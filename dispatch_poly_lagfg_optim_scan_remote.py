#!/usr/bin/env python3
"""
Dispatch poly+lag_fg_corr+optimizer scan workers to remote machines.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from run_poly_lagfg_optim_scan import generate_candidates


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
    p = argparse.ArgumentParser(description="Dispatch poly+lag_fg_corr+optimizer scan workers to remote hosts.")
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

    # Scan knobs (mirrors run_poly_lagfg_optim_scan.py).
    p.add_argument("--datasets", type=str, default="cube1,cube2")
    p.add_argument("--exclude-from-ranking", type=str, default="cube1")
    p.add_argument("--num-iters", type=int, default=12000)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)
    p.add_argument("--power-config", type=str, default="configs/power_eor_window.json")

    # Candidate generation knobs.
    p.add_argument("--seed", type=int, default=20260214)
    p.add_argument("--num-candidates", type=int, default=160)
    p.add_argument("--num-controls", type=int, default=12)
    p.add_argument("--candidate-names", type=str, default="")

    p.add_argument("--fg-smooth-modes", type=str, default="diff2_l2")
    p.add_argument("--poly-degrees", type=str, default="2,3,4,5")
    p.add_argument(
        "--poly-bases",
        type=str,
        default="power",
        help="Comma-separated polynomial bases for poly_reparam: power,chebyshev,legendre.",
    )
    p.add_argument("--poly-x-modes", type=str, default="lin,log")
    p.add_argument("--poly-models", type=str, default="exp", help="Comma-separated poly_model choices: add,exp.")
    p.add_argument(
        "--poly-resid-enabled-list",
        type=str,
        default="false",
        help="Comma-separated booleans for poly_resid_enabled: true,false.",
    )
    p.add_argument("--poly-weights", type=str, default="0.3,1.0,3.0,10.0")
    p.add_argument("--poly-sigma-min", type=float, default=0.003)
    p.add_argument("--poly-sigma-max", type=float, default=0.2)
    p.add_argument("--beta-min", type=float, default=0.05)
    p.add_argument("--beta-max", type=float, default=2.0)
    p.add_argument("--gamma-min", type=float, default=0.05)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--data-error-min", type=float, default=0.002)
    p.add_argument("--data-error-max", type=float, default=0.02)
    p.add_argument("--fg-smooth-mean-list", type=str, default="0.0,0.002")
    p.add_argument("--fg-smooth-sigma-min", type=float, default=0.001)
    p.add_argument("--fg-smooth-sigma-max", type=float, default=0.03)
    p.add_argument("--eor-prior-sigma-list", type=str, default="0.01,0.02,0.04,0.08")
    p.add_argument("--eor-amp-prior-modes", type=str, default="slice_rms_hinge,voxel_deadzone,hybrid")
    p.add_argument("--eor-amp-threshold", type=float, default=0.1)

    p.add_argument("--optimizer-names", type=str, default="adam,sgd")
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--lr-max", type=float, default=2e-3)
    p.add_argument("--lr-fg-factor-list", type=str, default="0.2,0.5,1.0,2.0")
    p.add_argument("--lr-schedulers", type=str, default="plateau,none")
    p.add_argument("--plateau-patience-list", type=str, default="200,400,800")
    p.add_argument("--plateau-factor-list", type=str, default="0.3,0.5")
    p.add_argument("--plateau-min-delta-list", type=str, default="1e-5,1e-4,1e-3")
    p.add_argument("--plateau-cooldown-list", type=str, default="80,160")
    p.add_argument("--init-modes", type=str, default="smooth_zero,smooth_residual,poly_residual")
    p.add_argument("--alt-update-modes", type=str, default="none,fg_then_eor")
    p.add_argument("--alt-fg-steps-list", type=str, default="10,50,200")
    p.add_argument("--alt-eor-steps-list", type=str, default="1,5")
    p.add_argument("--extra-loss-start-list", type=str, default="0,200,500,1000")
    p.add_argument("--extra-loss-ramp-list", type=str, default="0,500,2000")

    p.add_argument("--lagfg-prob", type=float, default=0.7)
    p.add_argument("--lagcorr-weight-min", type=float, default=0.005)
    p.add_argument("--lagcorr-weight-max", type=float, default=0.3)
    p.add_argument("--lagcorr-features", type=str, default="diff1,raw")
    p.add_argument("--lagcorr-unit", type=str, default="mhz")
    p.add_argument("--lagcorr-pair-sampling", type=str, default="random")
    p.add_argument("--lagcorr-random-seed", type=int, default=20260214)
    p.add_argument("--lagcorr-max-pairs-list", type=str, default="64,128,256")
    p.add_argument("--lagcorr-spatial-pool-list", type=str, default="2,4,8")
    p.add_argument("--lagcorr-rms-min-list", type=str, default="0.0,0.01,0.05")
    p.add_argument("--lagcorr-sigma-floor", type=float, default=0.02)
    p.add_argument("--lagfg-prior-source", type=str, default="obs_smooth")
    p.add_argument("--lagfg-const-mean", type=float, default=0.99)
    p.add_argument("--lagfg-const-sigma", type=float, default=0.05)
    p.add_argument(
        "--lagfg-prior-robust",
        action="store_true",
        help="Use median+MAD (robust) stats for fg_lagcorr_mean/sigma estimation on the worker.",
    )

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
        help="Local manifest path (default: <work-root>/runs/remote/<date>/poly_lagfg_optim_dispatch_<timestamp>.json).",
    )
    return p.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _shq(s: str) -> str:
    return shlex.quote(str(s))


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
    out.sort(key=lambda x: x.index)
    return out


def make_worker_slots(host: str, gpus: Sequence[GpuState]) -> List[WorkerSlot]:
    slots: List[WorkerSlot] = []
    idxs = [int(g.index) for g in gpus]
    for i in range(0, len(idxs) - 1, 2):
        g0 = idxs[i]
        g1 = idxs[i + 1]
        alias = host.split("@")[-1].replace(".", "_")
        slots.append(WorkerSlot(host=host, gpu0=g0, gpu1=g1, alias=alias))
    return slots


def split_round_robin(items: Sequence[str], n: int) -> List[List[str]]:
    out: List[List[str]] = [[] for _ in range(int(n))]
    for i, it in enumerate(items):
        out[i % int(n)].append(str(it))
    return out


def make_gpu_map_for_slot(dataset_names: Sequence[str], *, gpu0: int, gpu1: int) -> str:
    if len(dataset_names) == 1:
        return f"{dataset_names[0]}:{int(gpu0)}"
    if len(dataset_names) >= 2:
        return f"{dataset_names[0]}:{int(gpu0)},{dataset_names[1]}:{int(gpu1)}"
    raise ValueError("No datasets.")


def _parse_python_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in _parse_csv_tokens(text):
        if ":" not in tok:
            continue
        host, path = tok.split(":", 1)
        out[host.strip()] = path.strip()
    return out


def rsync_code(code_dir: Path, host: str, remote_root: str) -> None:
    # Keep it simple: sync the entire 3dnet folder (small enough for our use).
    src = str(code_dir.resolve()).rstrip("/") + "/"
    dst = f"{host}:{remote_root.rstrip('/')}/code/3dnet/"
    subprocess.check_call(["rsync", "-az", "--delete", src, dst])


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    hosts = _parse_csv_tokens(args.hosts)
    if not hosts:
        raise ValueError("No hosts provided.")
    python_map = _parse_python_map(args.python_bin_map)

    # Generate candidate names deterministically to split across slots.
    candidates = generate_candidates(args)
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
    remote_run_root = f"{args.remote_root.rstrip('/')}/runs/poly_lagfg_optim_scan_{stamp}_remote"

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
            _shq(f"{args.remote_root.rstrip('/')}/code/3dnet/run_poly_lagfg_optim_scan.py"),
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
            "--freq-delta-mhz",
            _shq(str(args.freq_delta_mhz)),
            "--power-config",
            _shq(str(args.power_config)),
            "--seed",
            _shq(str(args.seed)),
            "--num-candidates",
            _shq(str(args.num_candidates)),
            "--num-controls",
            _shq(str(args.num_controls)),
            "--candidate-names",
            _shq(cand_text),
            # Candidate generation knobs
            "--fg-smooth-modes",
            _shq(str(args.fg_smooth_modes)),
            "--poly-degrees",
            _shq(str(args.poly_degrees)),
            "--poly-x-modes",
            _shq(str(args.poly_x_modes)),
            "--poly-models",
            _shq(str(args.poly_models)),
            "--poly-resid-enabled-list",
            _shq(str(args.poly_resid_enabled_list)),
            "--poly-weights",
            _shq(str(args.poly_weights)),
            "--poly-sigma-min",
            _shq(str(args.poly_sigma_min)),
            "--poly-sigma-max",
            _shq(str(args.poly_sigma_max)),
            "--beta-min",
            _shq(str(args.beta_min)),
            "--beta-max",
            _shq(str(args.beta_max)),
            "--gamma-min",
            _shq(str(args.gamma_min)),
            "--gamma-max",
            _shq(str(args.gamma_max)),
            "--data-error-min",
            _shq(str(args.data_error_min)),
            "--data-error-max",
            _shq(str(args.data_error_max)),
            "--fg-smooth-mean-list",
            _shq(str(args.fg_smooth_mean_list)),
            "--fg-smooth-sigma-min",
            _shq(str(args.fg_smooth_sigma_min)),
            "--fg-smooth-sigma-max",
            _shq(str(args.fg_smooth_sigma_max)),
            "--eor-prior-sigma-list",
            _shq(str(args.eor_prior_sigma_list)),
            "--eor-amp-prior-modes",
            _shq(str(args.eor_amp_prior_modes)),
            "--eor-amp-threshold",
            _shq(str(args.eor_amp_threshold)),
            "--optimizer-names",
            _shq(str(args.optimizer_names)),
            "--lr-min",
            _shq(str(args.lr_min)),
            "--lr-max",
            _shq(str(args.lr_max)),
            "--lr-fg-factor-list",
            _shq(str(args.lr_fg_factor_list)),
            "--lr-schedulers",
            _shq(str(args.lr_schedulers)),
            "--plateau-patience-list",
            _shq(str(args.plateau_patience_list)),
            "--plateau-factor-list",
            _shq(str(args.plateau_factor_list)),
            "--plateau-min-delta-list",
            _shq(str(args.plateau_min_delta_list)),
            "--plateau-cooldown-list",
            _shq(str(args.plateau_cooldown_list)),
            "--init-modes",
            _shq(str(args.init_modes)),
            "--alt-update-modes",
            _shq(str(args.alt_update_modes)),
            "--alt-fg-steps-list",
            _shq(str(args.alt_fg_steps_list)),
            "--alt-eor-steps-list",
            _shq(str(args.alt_eor_steps_list)),
            "--extra-loss-start-list",
            _shq(str(args.extra_loss_start_list)),
            "--extra-loss-ramp-list",
            _shq(str(args.extra_loss_ramp_list)),
            "--lagfg-prob",
            _shq(str(args.lagfg_prob)),
            "--lagcorr-weight-min",
            _shq(str(args.lagcorr_weight_min)),
            "--lagcorr-weight-max",
            _shq(str(args.lagcorr_weight_max)),
            "--lagcorr-features",
            _shq(str(args.lagcorr_features)),
            "--lagcorr-unit",
            _shq(str(args.lagcorr_unit)),
            "--lagcorr-pair-sampling",
            _shq(str(args.lagcorr_pair_sampling)),
            "--lagcorr-random-seed",
            _shq(str(args.lagcorr_random_seed)),
            "--lagcorr-max-pairs-list",
            _shq(str(args.lagcorr_max_pairs_list)),
            "--lagcorr-spatial-pool-list",
            _shq(str(args.lagcorr_spatial_pool_list)),
            "--lagcorr-rms-min-list",
            _shq(str(args.lagcorr_rms_min_list)),
            "--lagcorr-sigma-floor",
            _shq(str(args.lagcorr_sigma_floor)),
            "--lagfg-prior-source",
            _shq(str(args.lagfg_prior_source)),
            "--lagfg-const-mean",
            _shq(str(args.lagfg_const_mean)),
            "--lagfg-const-sigma",
            _shq(str(args.lagfg_const_sigma)),
            *(
                ["--lagfg-prior-robust"]
                if bool(args.lagfg_prior_robust)
                else []
            ),
            "--python-bin",
            _shq(worker_python_bin),
        ]

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
    default_manifest = work_root / "runs" / "remote" / date_tag / f"poly_lagfg_optim_dispatch_{stamp}.json"
    manifest_path = args.launch_manifest.resolve() if args.launch_manifest else default_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "remote_run_root": remote_run_root,
        "candidate_count": len(names),
        "slots": [slot.__dict__ for slot in slots],
        "host_gpu_raw": {h: [s.__dict__ for s in ss] for h, ss in host_gpu_raw.items()},
        "host_gpu_avail": {h: [s.__dict__ for s in ss] for h, ss in host_gpu_avail.items()},
        "launches": launches,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[dispatch] candidates={len(names)} slots={len(slots)} launches={len(launches)}")
    print(f"[dispatch] manifest={manifest_path}")
    for rec in launches:
        print(f"  host={rec['host']} worker={rec['worker']} pid={rec.get('pid')} out={rec['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
