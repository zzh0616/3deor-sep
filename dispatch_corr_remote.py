#!/usr/bin/env python3
"""
Dispatch corr-term scans to remote machines.

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
from typing import Dict, List, Sequence, Tuple

from run_corr_hyper_scan import generate_candidates


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
    parser = argparse.ArgumentParser(description="Dispatch corr scan workers to remote hosts.")
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
        help="Optional host-specific python map: host:path,host:path.",
    )

    # Scan knobs (mirrors run_corr_hyper_scan.py).
    parser.add_argument("--datasets", type=str, default="cube1,cube2")
    parser.add_argument("--num-iters", type=int, default=2500)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--cut-size-frac", type=float, default=0.30)
    parser.add_argument("--freq-start-mhz", type=float, default=106.0)
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1)
    parser.add_argument("--data-error", type=float, default=0.005)

    # Fixed base baseline.
    parser.add_argument("--base-beta", type=float, default=0.5)
    parser.add_argument("--base-gamma", type=float, default=0.6)
    parser.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    parser.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    parser.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2")
    parser.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    parser.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    parser.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Optimizer knobs.
    parser.add_argument("--optimizer-name", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="plateau")
    parser.add_argument("--lr-plateau-patience", type=int, default=240)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=80)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    # Corr grid.
    parser.add_argument("--corr-weight-list", type=str, default="0.05,0.1,0.2,0.5,1.0,2.0")
    parser.add_argument("--corr-sigma-list", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--corr-abs-threshold-list", type=str, default="0.0")
    parser.add_argument("--corr-prior-mean", type=float, default=0.0)
    parser.add_argument(
        "--corr-reduce",
        type=str,
        default="mean",
        choices=["mean", "topk", "logsumexp"],
        help="Reduction over per-frequency corr penalties (default mean).",
    )
    parser.add_argument("--corr-topk", type=int, default=8, help="Top-k used when corr_reduce=topk.")
    parser.add_argument(
        "--corr-lse-alpha",
        type=float,
        default=10.0,
        help="Temperature used when corr_reduce=logsumexp (log-mean-exp).",
    )
    parser.add_argument("--extra-loss-start-iter", type=int, default=500)
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=0)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--candidate-names", type=str, default="")

    # GPU availability policy (util low + enough free memory).
    parser.add_argument("--gpu-util-max", type=int, default=5)
    parser.add_argument("--gpu-used-max-mb", type=int, default=0)
    parser.add_argument("--gpu-free-min-mb", type=int, default=20000)

    parser.add_argument("--sync-code", action="store_true", help="Rsync local 3dnet code to remote hosts before launch.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned launch commands only.")
    parser.add_argument(
        "--launch-manifest",
        type=Path,
        default=None,
        help="Local manifest path (default: <work-root>/runs/remote/<date>/corr_dispatch_<timestamp>.json).",
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
        host = host.strip()
        path = path.strip()
        if not host or not path:
            raise ValueError(f"Invalid host:path token: {token}")
        out[host] = path
    return out


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    hosts = [h.strip() for h in str(args.hosts).split(",") if h.strip()]
    if not hosts:
        raise ValueError("No remote hosts provided.")
    python_map = parse_host_path_map(args.python_bin_map)

    candidates = generate_candidates(
        corr_weight_list=_parse_float_list(args.corr_weight_list),
        corr_sigma_list=_parse_float_list(args.corr_sigma_list),
        corr_abs_threshold_list=_parse_float_list(args.corr_abs_threshold_list),
        corr_reduce=str(args.corr_reduce),
        corr_topk=int(args.corr_topk) if args.corr_topk is not None else None,
        corr_lse_alpha=float(args.corr_lse_alpha),
        extra_loss_start_iter=int(args.extra_loss_start_iter),
        extra_loss_ramp_iters=int(args.extra_loss_ramp_iters),
        include_control=bool(args.include_control),
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

    chunks = split_round_robin(names, len(slots))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_root = f"{args.remote_root.rstrip('/')}/runs/corr_scan_{stamp}_remote"

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
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet/run_corr_hyper_scan.py"),
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
            "--corr-weight-list",
            _shq(str(args.corr_weight_list)),
            "--corr-sigma-list",
            _shq(str(args.corr_sigma_list)),
            "--corr-abs-threshold-list",
            _shq(str(args.corr_abs_threshold_list)),
            "--corr-prior-mean",
            _shq(str(args.corr_prior_mean)),
            "--corr-reduce",
            _shq(str(args.corr_reduce)),
            "--corr-topk",
            _shq(str(args.corr_topk)),
            "--corr-lse-alpha",
            _shq(str(args.corr_lse_alpha)),
            "--extra-loss-start-iter",
            _shq(str(args.extra_loss_start_iter)),
            "--extra-loss-ramp-iters",
            _shq(str(args.extra_loss_ramp_iters)),
            "--candidate-names",
            _shq(cand_text),
            "--python-bin",
            _shq(worker_python_bin),
        ]
        if args.include_control:
            cmd_parts.append("--include-control")
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
    default_manifest = work_root / "runs" / "remote" / date_tag / f"corr_dispatch_{stamp}.json"
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
