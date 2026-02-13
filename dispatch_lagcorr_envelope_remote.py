#!/usr/bin/env python3
"""
Dispatch lagcorr-envelope scans to remote machines.

Remote layout expectation:
  <remote_root>/code/3dnet/  (synced code)
  <remote_root>/data/        (FITS cubes)
  <remote_root>/runs/        (outputs)

We launch one "worker" per 2-GPU slot (cube1+cube2 in parallel), and split
candidate names round-robin across slots.
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

from run_lagcorr_envelope_scan import generate_candidates


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
    parser = argparse.ArgumentParser(description="Dispatch lagcorr envelope scan workers to remote hosts.")
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

    # Scan knobs (mirrors run_lagcorr_envelope_scan.py).
    parser.add_argument("--datasets", type=str, default="cube1,cube2")
    parser.add_argument("--num-iters", type=int, default=2500)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--cut-size-frac", type=float, default=0.30)
    parser.add_argument("--freq-start-mhz", type=float, default=106.0)
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1)
    parser.add_argument("--data-error", type=float, default=0.005)
    parser.add_argument("--max-concurrent-jobs", type=int, default=2)

    # Optimizer knobs.
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--optimizer-name", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="plateau")
    parser.add_argument("--lr-plateau-patience", type=int, default=240)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=80)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    # Fixed base baseline.
    parser.add_argument("--base-beta", type=float, default=0.5)
    parser.add_argument("--base-gamma", type=float, default=0.6)
    parser.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    parser.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    parser.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2")
    parser.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    parser.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    parser.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Extra-loss schedule (corr ramp).
    parser.add_argument("--extra-loss-start-iter", type=int, default=300)
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=700)

    # Corr knobs (fixed for this scan).
    parser.add_argument("--corr-prior-mean", type=float, default=0.0)
    parser.add_argument("--corr-prior-sigma", type=float, default=0.2)
    parser.add_argument("--corr-abs-threshold", type=float, default=0.05)
    parser.add_argument("--corr-reduce", type=str, default="topk", choices=["mean", "topk", "logsumexp"])
    parser.add_argument("--corr-topk", type=int, default=8)
    parser.add_argument("--corr-lse-alpha", type=float, default=10.0)
    parser.add_argument("--corr-weight", type=float, default=0.2)
    parser.add_argument("--no-corr", action="store_true")

    # Lagcorr knobs.
    parser.add_argument("--lagcorr-weight", type=float, default=1.0)
    parser.add_argument("--lagcorr-spatial-pool-list", type=str, default="4")
    parser.add_argument("--lagcorr-eor-start-iter", type=int, default=1200)
    parser.add_argument("--lagcorr-eor-ramp-iters", type=int, default=800)
    parser.add_argument("--tail-eps-list", type=str, default="0.05,0.08")
    parser.add_argument("--neg-delta-list", type=str, default="0.0,0.02")
    parser.add_argument("--near-rho-min-list", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--rebound-delta-up-list", type=str, default="0.01,0.02")
    parser.add_argument("--rebound-eps-act", type=float, default=0.05)
    parser.add_argument("--w-tail", type=float, default=1.0)
    parser.add_argument("--w-neg", type=float, default=1.0)
    parser.add_argument("--w-near", type=float, default=1.0)
    parser.add_argument("--w-rebound", type=float, default=1.0)
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
        help="Local manifest path (default: <work-root>/runs/remote/<date>/lagcorr_env_dispatch_<timestamp>.json).",
    )
    return parser.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


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
    if not code_dir.is_dir():
        raise FileNotFoundError(f"Local code dir not found: {code_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_bucket = datetime.now().strftime("%Y%m%d")
    if args.launch_manifest is None:
        manifest_path = work_root / "runs" / "remote" / date_bucket / f"lagcorr_env_dispatch_{stamp}.json"
    else:
        manifest_path = args.launch_manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    remote_root = str(args.remote_root).rstrip("/")
    remote_run_root = f"{remote_root}/runs/lagcorr_env_scan_{stamp}_remote"

    hosts = [h.strip() for h in str(args.hosts).split(",") if h.strip()]
    python_map = parse_host_path_map(str(args.python_bin_map))

    # GPU discovery.
    gpu_raw: Dict[str, List[GpuState]] = {}
    gpu_available: Dict[str, List[GpuState]] = {}
    worker_slots: List[WorkerSlot] = []
    for host in hosts:
        states = query_remote_gpus(host)
        gpu_raw[host] = states
        avail = available_gpus(
            states,
            util_max=int(args.gpu_util_max),
            used_max_mb=int(args.gpu_used_max_mb),
            free_min_mb=int(args.gpu_free_min_mb),
        )
        gpu_available[host] = avail
        worker_slots.extend(make_worker_slots(host, avail))

    if not worker_slots:
        raise RuntimeError("No available 2-GPU slots found on remote hosts (check --gpu-* thresholds).")

    # Candidate names split across workers.
    candidates = generate_candidates(args)
    if str(args.candidate_names).strip():
        allow = {x.strip() for x in str(args.candidate_names).split(",") if x.strip()}
        unknown = sorted(allow - {c.name for c in candidates})
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
    if not candidates:
        raise RuntimeError("No candidates to dispatch.")
    cand_names = [c.name for c in candidates]
    splits = split_round_robin(cand_names, len(worker_slots))

    payload: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "remote_run_root": remote_run_root,
        "candidate_count": len(cand_names),
        "hosts": hosts,
        "gpu_raw": {h: [asdict(s) for s in gpu_raw[h]] for h in hosts},
        "gpu_available": {h: [asdict(s) for s in gpu_available[h]] for h in hosts},
        "worker_slots": [asdict(s) for s in worker_slots],
        "launches": [],
    }

    if bool(args.sync_code) and not bool(args.dry_run):
        for host in hosts:
            rsync_code(code_dir, host, remote_root)

    launches: List[Dict[str, object]] = []
    for slot, subset in zip(worker_slots, splits):
        if not subset:
            continue
        host = slot.host
        alias = slot.alias
        worker = f"{alias}_g{slot.gpu0}_{slot.gpu1}"
        python_bin = python_map.get(host, str(args.python_bin))
        gpu_map = f"cube1:{slot.gpu0},cube2:{slot.gpu1}"
        out_dir = f"{remote_run_root}/{worker}"
        log_path = f"{out_dir}/worker.log"

        run_cmd = [
            "nohup",
            _shq(python_bin),
            _shq(f"{remote_root}/code/3dnet/run_lagcorr_envelope_scan.py"),
            "--work-root",
            _shq(remote_root),
            "--code-dir",
            _shq(f"{remote_root}/code/3dnet"),
            "--data-dir",
            _shq(f"{remote_root}/data"),
            "--output-dir",
            _shq(out_dir),
            "--datasets",
            _shq(str(args.datasets)),
            "--gpu-map",
            _shq(gpu_map),
            "--max-concurrent-jobs",
            _shq(str(args.max_concurrent_jobs)),
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
            "--lr",
            _shq(str(args.lr)),
            "--optimizer-name",
            _shq(str(args.optimizer_name)),
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
            "--extra-loss-start-iter",
            _shq(str(args.extra_loss_start_iter)),
            "--extra-loss-ramp-iters",
            _shq(str(args.extra_loss_ramp_iters)),
            "--corr-prior-mean",
            _shq(str(args.corr_prior_mean)),
            "--corr-prior-sigma",
            _shq(str(args.corr_prior_sigma)),
            "--corr-abs-threshold",
            _shq(str(args.corr_abs_threshold)),
            "--corr-reduce",
            _shq(str(args.corr_reduce)),
            "--corr-topk",
            _shq(str(args.corr_topk)),
            "--corr-lse-alpha",
            _shq(str(args.corr_lse_alpha)),
            "--corr-weight",
            _shq(str(args.corr_weight)),
            "--lagcorr-weight",
            _shq(str(args.lagcorr_weight)),
            "--lagcorr-spatial-pool-list",
            _shq(str(args.lagcorr_spatial_pool_list)),
            "--lagcorr-eor-start-iter",
            _shq(str(args.lagcorr_eor_start_iter)),
            "--lagcorr-eor-ramp-iters",
            _shq(str(args.lagcorr_eor_ramp_iters)),
            "--tail-eps-list",
            _shq(str(args.tail_eps_list)),
            "--neg-delta-list",
            _shq(str(args.neg_delta_list)),
            "--near-rho-min-list",
            _shq(str(args.near_rho_min_list)),
            "--rebound-delta-up-list",
            _shq(str(args.rebound_delta_up_list)),
            "--rebound-eps-act",
            _shq(str(args.rebound_eps_act)),
            "--w-tail",
            _shq(str(args.w_tail)),
            "--w-neg",
            _shq(str(args.w_neg)),
            "--w-near",
            _shq(str(args.w_near)),
            "--w-rebound",
            _shq(str(args.w_rebound)),
            "--candidate-names",
            _shq(",".join(subset)),
            "--python-bin",
            _shq(python_bin),
        ]
        if bool(args.no_corr):
            run_cmd.append("--no-corr")
        if bool(args.dry_run):
            run_cmd.append("--dry-run")
        cmd_str = (
            f"mkdir -p {_shq(out_dir)} && "
            + " ".join(run_cmd)
            + f" > {_shq(log_path)} 2>&1 & echo $!"
        )

        rec = {
            "host": host,
            "worker": worker,
            "gpu_map": gpu_map,
            "candidate_count": len(subset),
            "candidates": list(subset),
            "python_bin": python_bin,
            "output_dir": out_dir,
            "log_path": log_path,
            "remote_cmd": cmd_str,
        }
        launches.append(rec)

    # Save manifest before launch (for reproducibility even if launch fails).
    payload["launches"] = launches
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if bool(args.dry_run):
        print(f"[dispatch] candidates={len(cand_names)} slots={len(worker_slots)} launches={len(launches)}")
        print(f"[dispatch] manifest={manifest_path}")
        for rec in launches:
            print(f"[cmd] host={rec['host']} worker={rec['worker']} n={rec['candidate_count']} out={rec['output_dir']}")
        return 0

    # Launch.
    for rec in launches:
        host = str(rec["host"])
        cmd = str(rec["remote_cmd"])
        pid = _run_capture(["ssh", "-o", "BatchMode=yes", host, cmd]).strip()
        rec["pid"] = pid
        print(
            f"[worker] host={host} worker={rec['worker']} "
            f"gpu_map={rec['gpu_map']} n={rec['candidate_count']} pid={pid}"
        )

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[dispatch] candidates={len(cand_names)} slots={len(worker_slots)} launches={len(launches)}")
    print(f"[dispatch] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
