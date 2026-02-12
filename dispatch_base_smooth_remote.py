#!/usr/bin/env python3
"""
Dispatch base-only smoothness scans to remote machines.

This launcher is designed for remote-only execution:
- code is synced to remote code directories;
- worker processes are launched via SSH on remote hosts;
- GPU availability is determined by low utilization and sufficient free memory.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from run_base_smooth_hyper_scan import generate_candidates


@dataclass(frozen=True)
class GpuState:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    util_percent: int

    @property
    def memory_free_mb(self) -> int:
        return max(0, int(self.memory_total_mb) - int(self.memory_used_mb))


@dataclass(frozen=True)
class WorkerSlot:
    host: str
    gpu0: int
    gpu1: int
    alias: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch base smooth scan workers to remote hosts.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Local project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="Local 3dnet dir (default <work-root>/3dnet).")
    parser.add_argument(
        "--remote-root",
        type=str,
        default="/data/zhenghao/fg_rmw",
        help="Remote project root that contains code/ and data/.",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="zhenghao@119.78.226.31,zhenghao@202.127.24.58",
        help="Comma-separated remote SSH hosts.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="/home/zhenghao/miniconda3/envs/torch/bin/python",
        help="Default remote python executable used to run scan workers.",
    )
    parser.add_argument(
        "--python-bin-map",
        type=str,
        default=(
            "zhenghao@119.78.226.31:/home/zhenghao/miniconda3/envs/torch/bin/python,"
            "zhenghao@202.127.24.58:/home/zhenghao/miniconda3/envs/pytorch/bin/python"
        ),
        help=(
            "Optional host-specific python map: host:path,host:path. "
            "When provided, host entry overrides --python-bin."
        ),
    )
    parser.add_argument("--phase", type=str, default="phase_a", choices=["phase_a", "phase_b", "phase_c"])
    parser.add_argument("--samples-per-mode", type=int, default=24)
    parser.add_argument("--fixed-controls-per-mode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument("--candidate-names", type=str, default="")
    parser.add_argument("--smooth-modes", type=str, default="")
    parser.add_argument("--prior-sources", type=str, default="")
    parser.add_argument("--explicit-fg-mean-list", type=str, default="0.0")
    parser.add_argument("--explicit-fg-sigma-list", type=str, default="")
    parser.add_argument("--num-iters", type=int, default=1200)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--cut-size-frac", type=float, default=0.30)
    parser.add_argument("--eor-amp-threshold", type=float, default=0.1)
    parser.add_argument("--data-error", type=float, default=0.005)
    parser.add_argument("--freq-start-mhz", type=float, default=106.0)
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1)
    parser.add_argument("--gpu-util-max", type=int, default=5, help="Max GPU util (%) to treat as available.")
    parser.add_argument(
        "--gpu-used-max-mb",
        type=int,
        default=0,
        help=(
            "Optional max used memory (MB) filter. "
            "Set to 0 to disable (default). This is usually unnecessary when gpu-free-min-mb is set."
        ),
    )
    parser.add_argument("--gpu-free-min-mb", type=int, default=20000, help="Min free memory (MB) to treat as available.")
    parser.add_argument("--sync-code", action="store_true", help="Rsync local 3dnet code to remote hosts before launch.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned launch commands only.")
    parser.add_argument(
        "--launch-manifest",
        type=Path,
        default=None,
        help="Local manifest path (default: <work-root>/remote_tests/<date>/base_smooth_dispatch_<timestamp>.json).",
    )
    return parser.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def _run_capture(cmd: Sequence[str]) -> str:
    out = subprocess.check_output(list(cmd), text=True)
    return out


def query_remote_gpus(host: str) -> List[GpuState]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        host,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
    ]
    out = _run_capture(cmd)
    states: List[GpuState] = []
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            states.append(
                GpuState(
                    index=int(parts[0]),
                    memory_used_mb=int(float(parts[1])),
                    memory_total_mb=int(float(parts[2])),
                    util_percent=int(float(parts[3])),
                )
            )
        except ValueError:
            continue
    return states


def available_gpus(states: Sequence[GpuState], *, util_max: int, used_max_mb: int, free_min_mb: int) -> List[GpuState]:
    out = []
    for s in states:
        if s.util_percent > int(util_max):
            continue
        # Keep the primary criterion aligned with our workflow:
        # low volatile utilization + enough free memory. A "used" cap is optional.
        if int(used_max_mb) > 0 and s.memory_used_mb > int(used_max_mb):
            continue
        if s.memory_free_mb < int(free_min_mb):
            continue
        out.append(s)
    out.sort(key=lambda x: (x.memory_used_mb, x.index))
    return out


def make_worker_slots(host: str, gpus: Sequence[GpuState]) -> List[WorkerSlot]:
    idxs = [g.index for g in gpus]
    slots: List[WorkerSlot] = []
    for i in range(0, len(idxs) - 1, 2):
        g0 = idxs[i]
        g1 = idxs[i + 1]
        alias = host.split("@")[-1].replace(".", "_")
        slots.append(WorkerSlot(host=host, gpu0=g0, gpu1=g1, alias=alias))
    return slots


def split_round_robin(items: Sequence[str], n_bins: int) -> List[List[str]]:
    bins: List[List[str]] = [[] for _ in range(max(1, n_bins))]
    for i, item in enumerate(items):
        bins[i % len(bins)].append(item)
    return bins


def rsync_code(local_code_dir: Path, host: str, remote_root: str) -> None:
    remote_code = f"{host}:{remote_root.rstrip('/')}/code/3dnet/"
    cmd = [
        "rsync",
        "-az",
        "--exclude",
        ".git",
        "--exclude",
        "__pycache__",
        "--exclude",
        "outputs",
        "--exclude",
        "*.pyc",
        f"{str(local_code_dir).rstrip('/')}/",
        remote_code,
    ]
    subprocess.check_call(cmd)


def _shq(text: str) -> str:
    return shlex.quote(text)


def parse_host_path_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid host:path token in --python-bin-map: {token}")
        host, path = token.split(":", 1)
        host = host.strip()
        path = path.strip()
        if not host or not path:
            raise ValueError(f"Invalid host:path token in --python-bin-map: {token}")
        out[host] = path
    return out


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        raise ValueError("No remote hosts provided.")
    python_map = parse_host_path_map(args.python_bin_map)

    candidates = generate_candidates(
        phase=args.phase,
        samples_per_mode=int(args.samples_per_mode),
        fixed_controls_per_mode=int(args.fixed_controls_per_mode),
        seed=int(args.seed),
        smooth_modes=_parse_csv_tokens(args.smooth_modes),
        prior_sources=_parse_csv_tokens(args.prior_sources),
        explicit_fg_smooth_mean_list=_parse_float_list(args.explicit_fg_mean_list),
        explicit_fg_smooth_sigma_list=(
            _parse_float_list(args.explicit_fg_sigma_list) if args.explicit_fg_sigma_list.strip() else None
        ),
    )
    names = [c.name for c in candidates]
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        known = set(names)
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        names = [n for n in names if n in allow]
    if not names:
        raise ValueError("No candidates selected for dispatch.")

    host_gpu_raw: Dict[str, List[GpuState]] = {}
    host_gpu_avail: Dict[str, List[GpuState]] = {}
    slots: List[WorkerSlot] = []
    for host in hosts:
        raw = query_remote_gpus(host)
        avail = available_gpus(
            raw,
            util_max=int(args.gpu_util_max),
            used_max_mb=int(args.gpu_used_max_mb),
            free_min_mb=int(args.gpu_free_min_mb),
        )
        host_gpu_raw[host] = raw
        host_gpu_avail[host] = avail
        slots.extend(make_worker_slots(host, avail))

    if not slots:
        raise RuntimeError("No worker slots available under current GPU availability thresholds.")

    if args.sync_code and not args.dry_run:
        for host in hosts:
            rsync_code(code_dir, host, args.remote_root)

    chunks = split_round_robin(names, len(slots))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_root = f"{args.remote_root.rstrip('/')}/runs/base_smooth_scan_{stamp}_remote"

    launches: List[Dict[str, object]] = []
    for slot, cand_chunk in zip(slots, chunks):
        if not cand_chunk:
            continue
        worker_name = f"{slot.alias}_g{slot.gpu0}_{slot.gpu1}"
        worker_out = f"{remote_run_root}/{worker_name}"
        worker_log = f"{worker_out}/worker.log"
        gpu_map = f"cube1:{slot.gpu0},cube2:{slot.gpu1}"
        cand_text = ",".join(cand_chunk)
        worker_python_bin = python_map.get(slot.host, args.python_bin)

        cmd_parts = [
            _shq(worker_python_bin),
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet/run_base_smooth_hyper_scan.py"),
            "--work-root",
            _shq(args.remote_root),
            "--code-dir",
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet"),
            "--data-dir",
            _shq(f"{args.remote_root.rstrip('/')}/data"),
            "--output-dir",
            _shq(worker_out),
            "--phase",
            _shq(args.phase),
            "--samples-per-mode",
            _shq(str(args.samples_per_mode)),
            "--fixed-controls-per-mode",
            _shq(str(args.fixed_controls_per_mode)),
            "--seed",
            _shq(str(args.seed)),
            "--candidate-names",
            _shq(cand_text),
            "--smooth-modes",
            _shq(args.smooth_modes),
            "--prior-sources",
            _shq(args.prior_sources),
            "--explicit-fg-mean-list",
            _shq(args.explicit_fg_mean_list),
            "--explicit-fg-sigma-list",
            _shq(args.explicit_fg_sigma_list),
            "--gpu-map",
            _shq(gpu_map),
            "--max-concurrent-jobs",
            "2",
            "--num-iters",
            _shq(str(args.num_iters)),
            "--print-every",
            _shq(str(args.print_every)),
            "--cut-size-frac",
            _shq(str(args.cut_size_frac)),
            "--eor-amp-threshold",
            _shq(str(args.eor_amp_threshold)),
            "--data-error",
            _shq(str(args.data_error)),
            "--freq-start-mhz",
            _shq(str(args.freq_start_mhz)),
            "--freq-delta-mhz",
            _shq(str(args.freq_delta_mhz)),
            "--python-bin",
            _shq(worker_python_bin),
        ]
        worker_cmd = " ".join(cmd_parts)
        remote_cmd = (
            f"mkdir -p {_shq(worker_out)} && "
            f"nohup {worker_cmd} > {_shq(worker_log)} 2>&1 & echo $!"
        )

        launch_rec: Dict[str, object] = {
            "host": slot.host,
            "worker": worker_name,
            "gpu_map": gpu_map,
            "candidate_count": len(cand_chunk),
            "candidates": cand_chunk,
            "python_bin": worker_python_bin,
            "output_dir": worker_out,
            "log_path": worker_log,
            "remote_cmd": remote_cmd,
        }
        if args.dry_run:
            launch_rec["pid"] = None
        else:
            pid = _run_capture(["ssh", "-o", "BatchMode=yes", slot.host, remote_cmd]).strip()
            launch_rec["pid"] = pid
        launches.append(launch_rec)

    date_tag = datetime.now().strftime("%Y%m%d")
    default_manifest = work_root / "remote_tests" / date_tag / f"base_smooth_dispatch_{stamp}.json"
    manifest_path = args.launch_manifest.resolve() if args.launch_manifest else default_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "remote_run_root": remote_run_root,
        "candidate_count": len(names),
        "hosts": hosts,
        "gpu_raw": {h: [asdict(s) for s in states] for h, states in host_gpu_raw.items()},
        "gpu_available": {h: [asdict(s) for s in states] for h, states in host_gpu_avail.items()},
        "worker_slots": [asdict(s) for s in slots],
        "launches": launches,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[dispatch] candidates={len(names)} slots={len(slots)} launches={len(launches)}")
    print(f"[dispatch] manifest={manifest_path}")
    for rec in launches:
        print(
            f"[worker] host={rec['host']} worker={rec['worker']} "
            f"gpu_map={rec['gpu_map']} n={rec['candidate_count']} pid={rec.get('pid')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
