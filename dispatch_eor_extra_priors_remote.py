#!/usr/bin/env python3
"""
Dispatch extra EoR priors scans (eor_mean/eor_hf) to remote machines.
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

from run_eor_extra_priors_scan import generate_candidates


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
    parser = argparse.ArgumentParser(description="Dispatch extra EoR priors scan workers to remote hosts.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd())
    parser.add_argument("--code-dir", type=Path, default=None)
    parser.add_argument("--remote-root", type=str, default="/data/zhenghao/fg_rmw")
    parser.add_argument("--hosts", type=str, default="zhenghao@119.78.226.31,zhenghao@202.127.24.58")
    parser.add_argument("--python-bin", type=str, default="/home/zhenghao/miniconda3/envs/torch/bin/python")
    parser.add_argument(
        "--python-bin-map",
        type=str,
        default=(
            "zhenghao@119.78.226.31:/home/zhenghao/miniconda3/envs/torch/bin/python,"
            "zhenghao@202.127.24.58:/home/zhenghao/miniconda3/envs/pytorch/bin/python"
        ),
    )

    # Scan knobs (mirrors run_eor_extra_priors_scan.py).
    parser.add_argument("--datasets", type=str, default="cube1,cube2")
    parser.add_argument("--exclude-from-ranking", type=str, default="cube1")
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

    # Base baseline.
    parser.add_argument("--base-beta", type=float, default=0.5)
    parser.add_argument("--base-gamma", type=float, default=0.6)
    parser.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    parser.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    parser.add_argument("--base-eor-amp-prior-mode", type=str, default="voxel_deadzone")
    parser.add_argument("--base-eor-hybrid-voxel-factor", type=float, default=5.0)
    parser.add_argument("--base-eor-hybrid-voxel-weight", type=float, default=0.1)
    parser.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2")
    parser.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    parser.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    parser.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Extra-loss schedule.
    parser.add_argument("--extra-loss-start-iter", type=int, default=300)
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=700)

    # Corr baseline.
    parser.add_argument("--corr-prior-mean", type=float, default=0.0)
    parser.add_argument("--corr-prior-sigma", type=float, default=0.2)
    parser.add_argument("--corr-abs-threshold", type=float, default=0.08)
    parser.add_argument("--corr-reduce", type=str, default="logsumexp")
    parser.add_argument("--corr-topk", type=int, default=8)
    parser.add_argument("--corr-lse-alpha", type=float, default=10.0)
    parser.add_argument("--corr-weight", type=float, default=0.2)
    parser.add_argument("--corr-feature", type=str, default="raw")
    parser.add_argument("--corr-spatial-pool", type=int, default=1)

    # Lagcorr baseline.
    parser.add_argument("--lagcorr-weight", type=float, default=1.0)
    parser.add_argument("--lagcorr-spatial-pool", type=int, default=4)
    parser.add_argument("--lagcorr-rms-min", type=float, default=0.0)
    parser.add_argument("--lagcorr-eor-start-iter", type=int, default=1200)
    parser.add_argument("--lagcorr-eor-ramp-iters", type=int, default=800)
    parser.add_argument("--lagcorr-eor-subterm-schedule", type=str, default="static")
    parser.add_argument("--lagcorr-eor-tail-weight-mode", type=str, default="hard")
    parser.add_argument("--lagcorr-eor-tail-sigmoid-center-mpc", type=float, default=150.0)
    parser.add_argument("--lagcorr-eor-tail-sigmoid-width-mpc", type=float, default=20.0)
    parser.add_argument("--lagcorr-eor-tail-eps", type=float, default=0.05)
    parser.add_argument("--lagcorr-eor-neg-delta", type=float, default=0.0)
    parser.add_argument("--lagcorr-eor-near-rho-min", type=float, default=0.05)
    parser.add_argument("--lagcorr-eor-near-floor-mode", type=str, default="absolute_mean")
    parser.add_argument("--lagcorr-eor-near-rho1-coeffs", type=str, default="0,0.8,0.6,0.4,0.3,0.2,0.1,0,0")
    parser.add_argument("--lagcorr-eor-rebound-eps-act", type=float, default=0.05)
    parser.add_argument("--lagcorr-eor-rebound-delta-up", type=float, default=0.02)
    parser.add_argument("--lagcorr-eor-w-tail", type=float, default=1.0)
    parser.add_argument("--lagcorr-eor-w-neg", type=float, default=1.0)
    parser.add_argument("--lagcorr-eor-w-near", type=float, default=1.0)
    parser.add_argument("--lagcorr-eor-w-rebound", type=float, default=1.0)
    parser.add_argument("--lagcorr-eor-near-max-lag", type=int, default=10)
    parser.add_argument("--lagcorr-eor-mid-max-lag", type=int, default=50)
    parser.add_argument("--lagcorr-eor-far-min-lag", type=int, default=70)

    # Extra priors scan grid.
    parser.add_argument("--eor-mean-weight-list", type=str, default="0.0,0.1,0.3")
    parser.add_argument("--eor-hf-weight-list", type=str, default="0.0,0.1,0.3")
    parser.add_argument("--eor-hf-percent", type=float, default=0.7)
    parser.add_argument("--eor-hf-r-max", type=float, default=0.85)
    parser.add_argument("--candidate-names", type=str, default="")

    # GPU availability.
    parser.add_argument("--gpu-util-max", type=int, default=5)
    parser.add_argument("--gpu-used-max-mb", type=int, default=0)
    parser.add_argument("--gpu-free-min-mb", type=int, default=20000)

    parser.add_argument("--sync-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launch-manifest", type=Path, default=None)
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
    python_map = parse_host_path_map(args.python_bin_map)

    candidates = generate_candidates(args)
    names = [c.name for c in candidates]
    if str(args.candidate_names).strip():
        allow = {x.strip() for x in str(args.candidate_names).split(",") if x.strip()}
        unknown = sorted(allow - set(names))
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        names = [n for n in names if n in allow]
    if not names:
        raise RuntimeError("No candidates to dispatch.")

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
        raise RuntimeError("No available 2-GPU slots found on remote hosts.")

    if args.sync_code and not args.dry_run:
        for host in hosts:
            rsync_code(code_dir, host, args.remote_root)

    dataset_names = _parse_csv_tokens(args.datasets)
    if not dataset_names:
        raise ValueError("No datasets provided.")

    chunks = split_round_robin(names, len(slots))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_run_root = f"{str(args.remote_root).rstrip('/')}/runs/eor_extra_priors_scan_{stamp}_remote"

    launches: List[Dict[str, object]] = []
    for slot, subset in zip(slots, chunks):
        if not subset:
            continue
        worker = f"{slot.alias}_g{slot.gpu0}_{slot.gpu1}"
        host = slot.host
        python_bin = python_map.get(host, str(args.python_bin))
        gpu_map = make_gpu_map_for_slot(dataset_names, gpu0=slot.gpu0, gpu1=slot.gpu1)
        out_dir = f"{remote_run_root}/{worker}"
        log_path = f"{out_dir}/worker.log"

        run_cmd = [
            _shq(python_bin),
            _shq(f"{str(args.remote_root).rstrip('/')}/code/3dnet/run_eor_extra_priors_scan.py"),
            "--work-root",
            _shq(str(args.remote_root)),
            "--code-dir",
            _shq(f"{str(args.remote_root).rstrip('/')}/code/3dnet"),
            "--data-dir",
            _shq(f"{str(args.remote_root).rstrip('/')}/data"),
            "--output-dir",
            _shq(out_dir),
            "--datasets",
            _shq(str(args.datasets)),
            "--exclude-from-ranking",
            _shq(str(args.exclude_from_ranking)),
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
            "--corr-feature",
            _shq(str(args.corr_feature)),
            "--corr-spatial-pool",
            _shq(str(args.corr_spatial_pool)),
            "--lagcorr-weight",
            _shq(str(args.lagcorr_weight)),
            "--lagcorr-spatial-pool",
            _shq(str(args.lagcorr_spatial_pool)),
            "--lagcorr-rms-min",
            _shq(str(args.lagcorr_rms_min)),
            "--lagcorr-eor-start-iter",
            _shq(str(args.lagcorr_eor_start_iter)),
            "--lagcorr-eor-ramp-iters",
            _shq(str(args.lagcorr_eor_ramp_iters)),
            "--lagcorr-eor-subterm-schedule",
            _shq(str(args.lagcorr_eor_subterm_schedule)),
            "--lagcorr-eor-tail-weight-mode",
            _shq(str(args.lagcorr_eor_tail_weight_mode)),
            "--lagcorr-eor-tail-sigmoid-center-mpc",
            _shq(str(args.lagcorr_eor_tail_sigmoid_center_mpc)),
            "--lagcorr-eor-tail-sigmoid-width-mpc",
            _shq(str(args.lagcorr_eor_tail_sigmoid_width_mpc)),
            "--lagcorr-eor-tail-eps",
            _shq(str(args.lagcorr_eor_tail_eps)),
            "--lagcorr-eor-neg-delta",
            _shq(str(args.lagcorr_eor_neg_delta)),
            "--lagcorr-eor-near-rho-min",
            _shq(str(args.lagcorr_eor_near_rho_min)),
            "--lagcorr-eor-near-floor-mode",
            _shq(str(args.lagcorr_eor_near_floor_mode)),
            "--lagcorr-eor-near-rho1-coeffs",
            _shq(str(args.lagcorr_eor_near_rho1_coeffs)),
            "--lagcorr-eor-rebound-eps-act",
            _shq(str(args.lagcorr_eor_rebound_eps_act)),
            "--lagcorr-eor-rebound-delta-up",
            _shq(str(args.lagcorr_eor_rebound_delta_up)),
            "--lagcorr-eor-w-tail",
            _shq(str(args.lagcorr_eor_w_tail)),
            "--lagcorr-eor-w-neg",
            _shq(str(args.lagcorr_eor_w_neg)),
            "--lagcorr-eor-w-near",
            _shq(str(args.lagcorr_eor_w_near)),
            "--lagcorr-eor-w-rebound",
            _shq(str(args.lagcorr_eor_w_rebound)),
            "--lagcorr-eor-near-max-lag",
            _shq(str(args.lagcorr_eor_near_max_lag)),
            "--lagcorr-eor-mid-max-lag",
            _shq(str(args.lagcorr_eor_mid_max_lag)),
            "--lagcorr-eor-far-min-lag",
            _shq(str(args.lagcorr_eor_far_min_lag)),
            "--eor-mean-weight-list",
            _shq(str(args.eor_mean_weight_list)),
            "--eor-hf-weight-list",
            _shq(str(args.eor_hf_weight_list)),
            "--eor-hf-percent",
            _shq(str(args.eor_hf_percent)),
            "--eor-hf-r-max",
            _shq(str(args.eor_hf_r_max)),
            "--candidate-names",
            _shq(",".join(subset)),
            "--python-bin",
            _shq(python_bin),
        ]
        if bool(args.dry_run):
            run_cmd.append("--dry-run")

        cmd_str = (
            f"mkdir -p {_shq(out_dir)} && nohup "
            + " ".join(run_cmd)
            + f" > {_shq(log_path)} 2>&1 & echo $!"
        )
        rec: Dict[str, object] = {
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
        if args.dry_run:
            rec["pid"] = None
        else:
            pid = _run_capture(["ssh", "-o", "BatchMode=yes", host, cmd_str]).strip()
            rec["pid"] = pid
        launches.append(rec)

    date_tag = datetime.now().strftime("%Y%m%d")
    default_manifest = work_root / "runs" / "remote" / date_tag / f"eor_extra_priors_dispatch_{stamp}.json"
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
        print(f"[worker] host={rec['host']} worker={rec['worker']} gpu_map={rec['gpu_map']} n={rec['candidate_count']} pid={rec.get('pid')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

