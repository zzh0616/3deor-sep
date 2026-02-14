#!/usr/bin/env python3
"""
Dispatch poly/rfft scan workers to remote machines.

Remote layout expectation:
  <remote_root>/code/3dnet/  (synced code)
  <remote_root>/data/        (FITS cubes)
  <remote_root>/runs/        (outputs)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from run_poly_rfft_scan import generate_candidates


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
    p = argparse.ArgumentParser(description="Dispatch poly/rfft scan workers to remote hosts.")
    p.add_argument("--work-root", type=Path, default=Path.cwd(), help="Local project root.")
    p.add_argument("--code-dir", type=Path, default=None, help="Local 3dnet dir (default <work-root>/3dnet).")
    p.add_argument(
        "--remote-root",
        type=str,
        default="/data/zhenghao/fg_rmw",
        help="Remote project root that contains code/ and data/.",
    )
    p.add_argument(
        "--hosts",
        type=str,
        default="zhenghao@119.78.226.31,zhenghao@202.127.24.58",
        help="Comma-separated remote SSH hosts.",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default="/home/zhenghao/miniconda3/envs/torch/bin/python",
        help="Default remote python executable used to run scan workers.",
    )
    p.add_argument(
        "--python-bin-map",
        type=str,
        default=(
            "zhenghao@119.78.226.31:/home/zhenghao/miniconda3/envs/torch/bin/python,"
            "zhenghao@202.127.24.58:/home/zhenghao/miniconda3/envs/pytorch/bin/python"
        ),
        help="Optional host-specific python map: host:path,host:path.",
    )

    # Scan knobs (mirrors run_poly_rfft_scan.py).
    p.add_argument("--datasets", type=str, default="cube1,cube2")
    p.add_argument("--exclude-from-ranking", type=str, default="cube1")
    p.add_argument("--num-iters", type=int, default=3000)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-start-mhz", type=float, default=106.0)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)
    p.add_argument("--data-error", type=float, default=0.005)

    # Fixed base baseline.
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument(
        "--base-eor-amp-prior-mode",
        type=str,
        default="slice_rms_hinge",
        choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"],
    )
    p.add_argument("--base-eor-hybrid-voxel-factor", type=float, default=5.0)
    p.add_argument("--base-eor-hybrid-voxel-weight", type=float, default=0.1)
    p.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2")
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Optimizer knobs.
    p.add_argument("--optimizer-name", type=str, default="adam")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="plateau")
    p.add_argument("--lr-plateau-patience", type=int, default=240)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-cooldown", type=int, default=80)
    p.add_argument("--lr-min", type=float, default=1e-6)

    p.add_argument("--extra-loss-start-iter", type=int, default=500)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=0)

    p.add_argument(
        "--power-config",
        type=str,
        default="configs/power_eor_window.json",
        help="Path relative to code-dir on remote host.",
    )

    # Candidate generation toggles.
    p.add_argument("--include-base", action="store_true")
    p.add_argument("--include-poly", action="store_true")
    p.add_argument("--include-rfft", action="store_true")
    p.add_argument("--include-combos", action="store_true")

    # Poly grid.
    p.add_argument("--poly-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--poly-degree-list", type=str, default="2,3")
    p.add_argument("--poly-sigma-list", type=str, default="0.05,0.1")

    # rFFT grid.
    p.add_argument("--fft-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--fft-sigma-list", type=str, default="0.5,1.0")
    p.add_argument("--fft-percent-list", type=str, default="0.7")
    p.add_argument("--fft-use-log-energy", action="store_true")
    p.add_argument("--fft-z-clip", type=float, default=None)

    p.add_argument("--candidate-names", type=str, default="")

    # GPU availability policy (util low + enough free memory).
    p.add_argument("--gpu-util-max", type=int, default=5)
    p.add_argument("--gpu-used-max-mb", type=int, default=0)
    p.add_argument("--gpu-free-min-mb", type=int, default=20000)

    p.add_argument("--sync-code", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--launch-manifest",
        type=Path,
        default=None,
        help="Local manifest path (default: <work-root>/runs/remote/<date>/poly_rfft_dispatch_<timestamp>.json).",
    )
    return p.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for token in _parse_csv_tokens(text):
        out.append(int(float(token)))
    return out


def _run_capture(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), text=True)


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
    out: List[GpuState] = []
    for s in states:
        if int(s.util_percent) > int(util_max):
            continue
        if int(used_max_mb) > 0 and int(s.memory_used_mb) > int(used_max_mb):
            continue
        if int(s.memory_free_mb) < int(free_min_mb):
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


def make_gpu_map_for_slot(datasets: Sequence[str], *, gpu0: int, gpu1: int) -> str:
    names = [str(x).strip() for x in datasets if str(x).strip()]
    if not names:
        raise ValueError("No datasets provided for gpu_map.")
    parts: List[str] = []
    for i, name in enumerate(names):
        gpu = int(gpu0) if (i % 2 == 0) else int(gpu1)
        parts.append(f"{name}:{gpu}")
    return ",".join(parts)


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
            raise ValueError(f"Invalid host:path token: {token}")
        host, path = token.split(":", 1)
        out[host.strip()] = path.strip()
    return out


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    hosts = [h.strip() for h in str(args.hosts).split(",") if h.strip()]
    if not hosts:
        raise ValueError("No remote hosts provided.")
    python_map = parse_host_path_map(args.python_bin_map)

    # Default include set aligns with run_poly_rfft_scan.py.
    include_any = args.include_base or args.include_poly or args.include_rfft or args.include_combos
    include_base = bool(args.include_base) if include_any else True
    include_poly = bool(args.include_poly) if include_any else True
    include_rfft = bool(args.include_rfft) if include_any else True
    include_combos = bool(args.include_combos) if include_any else False

    poly_weight_list = _parse_float_list(args.poly_weight_list)
    poly_degree_list = _parse_int_list(args.poly_degree_list)
    poly_sigma_list = _parse_float_list(args.poly_sigma_list)
    fft_weight_list = _parse_float_list(args.fft_weight_list)
    fft_sigma_list = _parse_float_list(args.fft_sigma_list)
    fft_percent_list = _parse_float_list(args.fft_percent_list)
    if (include_poly or include_combos) and (not poly_weight_list or not poly_degree_list or not poly_sigma_list):
        raise ValueError("Poly lists must be non-empty when poly/combo candidates are enabled.")
    if (include_rfft or include_combos) and (not fft_weight_list or not fft_sigma_list or not fft_percent_list):
        raise ValueError("FFT lists must be non-empty when rfft/combo candidates are enabled.")

    candidates = generate_candidates(
        include_base=include_base,
        include_poly=include_poly,
        include_rfft=include_rfft,
        include_combos=include_combos,
        poly_weight_list=poly_weight_list,
        poly_degree_list=poly_degree_list,
        poly_sigma_list=poly_sigma_list,
        fft_weight_list=fft_weight_list,
        fft_sigma_list=fft_sigma_list,
        fft_percent_list=fft_percent_list,
        fft_use_log_energy=bool(args.fft_use_log_energy),
        fft_z_clip=args.fft_z_clip,
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
        raise RuntimeError("No worker slots available under current GPU thresholds.")

    if args.sync_code and not args.dry_run:
        for host in hosts:
            rsync_code(code_dir, host, args.remote_root)

    dataset_names = _parse_csv_tokens(args.datasets)
    if not dataset_names:
        raise ValueError("No datasets provided.")

    chunks = split_round_robin(names, len(slots))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_root = f"{args.remote_root.rstrip('/')}/runs/poly_rfft_scan_{stamp}_remote"

    launches: List[Dict[str, object]] = []
    for slot, cand_chunk in zip(slots, chunks):
        if not cand_chunk:
            continue
        worker_name = f"{slot.alias}_g{slot.gpu0}_{slot.gpu1}"
        worker_out = f"{remote_run_root}/{worker_name}"
        worker_log = f"{worker_out}/worker.log"
        gpu_map = make_gpu_map_for_slot(dataset_names, gpu0=slot.gpu0, gpu1=slot.gpu1)
        cand_text = ",".join(cand_chunk)
        worker_python_bin = python_map.get(slot.host, args.python_bin)

        cmd_parts = [
            _shq(worker_python_bin),
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet/run_poly_rfft_scan.py"),
            "--work-root",
            _shq(args.remote_root),
            "--code-dir",
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet"),
            "--data-dir",
            _shq(f"{args.remote_root.rstrip('/')}/data"),
            "--output-dir",
            _shq(worker_out),
            "--datasets",
            _shq(args.datasets),
            "--exclude-from-ranking",
            _shq(str(args.exclude_from_ranking)),
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
            "--freq-start-mhz",
            _shq(str(args.freq_start_mhz)),
            "--freq-delta-mhz",
            _shq(str(args.freq_delta_mhz)),
            "--data-error",
            _shq(str(args.data_error)),
            "--base-beta",
            _shq(str(args.base_beta)),
            "--base-gamma",
            _shq(str(args.base_gamma)),
            "--base-eor-prior-sigma",
            _shq(str(args.base_eor_prior_sigma)),
            "--base-eor-amp-threshold",
            _shq(str(args.base_eor_amp_threshold)),
            "--base-eor-amp-prior-mode",
            _shq(str(args.base_eor_amp_prior_mode)),
            "--base-eor-hybrid-voxel-factor",
            _shq(str(args.base_eor_hybrid_voxel_factor)),
            "--base-eor-hybrid-voxel-weight",
            _shq(str(args.base_eor_hybrid_voxel_weight)),
            "--base-fg-smooth-mode",
            _shq(str(args.base_fg_smooth_mode)),
            "--base-fg-smooth-mean",
            _shq(str(args.base_fg_smooth_mean)),
            "--base-fg-smooth-sigma",
            _shq(str(args.base_fg_smooth_sigma)),
            "--base-fg-smooth-huber-delta",
            _shq(str(args.base_fg_smooth_huber_delta)),
            "--optimizer-name",
            _shq(str(args.optimizer_name)),
            "--lr",
            _shq(str(args.lr)),
            "--momentum",
            _shq(str(args.momentum)),
            "--lr-scheduler",
            _shq(str(args.lr_scheduler)),
            "--lr-plateau-patience",
            _shq(str(args.lr_plateau_patience)),
            "--lr-plateau-factor",
            _shq(str(args.lr_plateau_factor)),
            "--lr-plateau-min-delta",
            _shq(str(args.lr_plateau_min_delta)),
            "--lr-plateau-cooldown",
            _shq(str(args.lr_plateau_cooldown)),
            "--lr-min",
            _shq(str(args.lr_min)),
            "--extra-loss-start-iter",
            _shq(str(args.extra_loss_start_iter)),
            "--extra-loss-ramp-iters",
            _shq(str(args.extra_loss_ramp_iters)),
            "--power-config",
            _shq(str(args.power_config)),
            "--poly-weight-list",
            _shq(str(args.poly_weight_list)),
            "--poly-degree-list",
            _shq(str(args.poly_degree_list)),
            "--poly-sigma-list",
            _shq(str(args.poly_sigma_list)),
            "--fft-weight-list",
            _shq(str(args.fft_weight_list)),
            "--fft-sigma-list",
            _shq(str(args.fft_sigma_list)),
            "--fft-percent-list",
            _shq(str(args.fft_percent_list)),
            "--candidate-names",
            _shq(cand_text),
            "--python-bin",
            _shq(worker_python_bin),
        ]
        if include_base:
            cmd_parts.append("--include-base")
        if include_poly:
            cmd_parts.append("--include-poly")
        if include_rfft:
            cmd_parts.append("--include-rfft")
        if include_combos:
            cmd_parts.append("--include-combos")
        if bool(args.fft_use_log_energy):
            cmd_parts.append("--fft-use-log-energy")
        if args.fft_z_clip is not None:
            cmd_parts.extend(["--fft-z-clip", _shq(str(args.fft_z_clip))])

        worker_cmd = " ".join(cmd_parts)
        remote_cmd = f"mkdir -p {_shq(worker_out)} && nohup {worker_cmd} > {_shq(worker_log)} 2>&1 & echo $!"

        rec: Dict[str, object] = {
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
            rec["pid"] = None
        else:
            pid = _run_capture(["ssh", "-o", "BatchMode=yes", slot.host, remote_cmd]).strip()
            rec["pid"] = pid
        launches.append(rec)

    date_tag = datetime.now().strftime("%Y%m%d")
    default_manifest = work_root / "runs" / "remote" / date_tag / f"poly_rfft_dispatch_{stamp}.json"
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

