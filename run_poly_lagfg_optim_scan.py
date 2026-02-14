#!/usr/bin/env python3
"""
Poly + (optional) FG-only lagcorr + optimizer hyperparameter scan on injected cubes.

Goal:
- Keep (or improve) EoR-window PS2D agreement while pushing per-frequency corr(EoR_est, EoR_true) higher.

Notes:
- This scan does NOT use injected EoR truth in the loss. Truth is only used for evaluation metrics.
- FG-only lagcorr ("lag_fg_corr") uses lagcorr with FG component weight=1 and EoR component weight=0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from dataset_registry import build_datasets, filter_datasets, parse_dataset_names


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


@dataclass(frozen=True)
class CandidateSpec:
    name: str

    # Base weights/priors
    beta: float
    gamma: float
    data_error: float
    fg_smooth_mode: str
    fg_smooth_mean: float
    fg_smooth_sigma: float
    fg_smooth_huber_delta: float
    eor_prior_sigma: float
    eor_amp_prior_mode: str
    eor_hybrid_voxel_factor: float
    eor_hybrid_voxel_weight: float
    eor_amp_threshold: float

    # Poly (extra term + reparam)
    poly_weight: float
    poly_degree: int
    poly_sigma: float
    poly_x_mode: str  # lin/log

    # Optimizer
    optimizer_name: str
    lr: float
    lr_fg_factor: float
    lr_scheduler: str
    lr_plateau_patience: int
    lr_plateau_factor: float
    lr_plateau_min_delta: float
    lr_plateau_cooldown: int
    lr_min: float
    alt_update_mode: str
    alt_fg_steps: int
    alt_eor_steps: int
    extra_loss_start_iter: int
    extra_loss_ramp_iters: int
    init_mode: str  # smooth_zero/smooth_residual/poly_residual

    # FG-only lagcorr
    lagfg_enabled: bool
    lagcorr_weight: float
    lagcorr_feature: str  # raw/diff1
    lagcorr_unit: str  # mhz/chan
    lagcorr_pair_sampling: str  # head/random
    lagcorr_random_seed: int
    lagcorr_max_pairs: Optional[int]
    lagcorr_spatial_pool: int
    lagcorr_rms_min: float
    lagcorr_sigma_floor: float


@dataclass(frozen=True)
class JobSpec:
    dataset_name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path
    freq_start_mhz: float
    gpu_index: int
    candidate: CandidateSpec
    run_dir: Path
    config_path: Path
    log_path: Path
    fg_output: Path
    eor_output: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run poly + lag_fg_corr + optimizer hyperparameter scan.")
    p.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    p.add_argument("--code-dir", type=Path, default=None, help="3dnet dir (default <work-root>/3dnet).")
    p.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/poly_lagfg_optim_scan_<timestamp>).",
    )
    p.add_argument("--seed", type=int, default=20260214, help="Random seed for candidate sampling.")
    p.add_argument("--num-candidates", type=int, default=160, help="Total candidates (incl controls).")
    p.add_argument("--num-controls", type=int, default=12, help="Deterministic control candidates.")
    p.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Comma-separated candidate names to run; empty means all generated candidates.",
    )

    p.add_argument("--datasets", type=str, default="cube1,cube2", help="Comma-separated datasets to run.")
    p.add_argument("--exclude-from-ranking", type=str, default="cube1", help="Comma-separated dataset names excluded.")
    p.add_argument("--gpu-map", type=str, default="cube1:0,cube2:1", help="Dataset->GPU mapping.")
    p.add_argument("--max-concurrent-jobs", type=int, default=2, help="Max concurrent dataset jobs per candidate.")

    p.add_argument("--num-iters", type=int, default=12000, help="Iterations per run.")
    p.add_argument("--print-every", type=int, default=200, help="Iteration logging interval.")
    p.add_argument("--cut-size-frac", type=float, default=0.30, help="Spatial center cut fraction.")

    p.add_argument("--freq-delta-mhz", type=float, default=0.1, help="Channel spacing in MHz.")
    p.add_argument("--power-config", type=str, default="configs/power_eor_window.json")

    # Candidate space knobs (kept intentionally compact; tune by editing this file if needed).
    p.add_argument("--fg-smooth-modes", type=str, default="diff2_l2", help="Comma-separated fg_smooth_mode choices.")
    p.add_argument("--poly-degrees", type=str, default="2,3,4,5")
    p.add_argument("--poly-x-modes", type=str, default="lin,log")
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
    p.add_argument(
        "--eor-amp-prior-modes",
        type=str,
        default="slice_rms_hinge,voxel_deadzone,hybrid",
        help="Comma-separated eor_amp_prior_mode choices.",
    )
    p.add_argument("--eor-amp-threshold", type=float, default=0.1)

    # Optimizer search
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

    # FG-only lagcorr search
    p.add_argument("--lagfg-prob", type=float, default=0.7, help="Probability a random candidate enables lag_fg_corr.")
    p.add_argument("--lagcorr-weight-min", type=float, default=0.005)
    p.add_argument("--lagcorr-weight-max", type=float, default=0.3)
    p.add_argument("--lagcorr-features", type=str, default="diff1,raw")
    p.add_argument("--lagcorr-unit", type=str, default="mhz", choices=["mhz", "chan"])
    p.add_argument("--lagcorr-pair-sampling", type=str, default="random", choices=["head", "random"])
    p.add_argument("--lagcorr-random-seed", type=int, default=20260214)
    p.add_argument("--lagcorr-max-pairs-list", type=str, default="64,128,256")
    p.add_argument("--lagcorr-spatial-pool-list", type=str, default="2,4,8")
    p.add_argument("--lagcorr-rms-min-list", type=str, default="0.0,0.01,0.05")
    p.add_argument("--lagcorr-sigma-floor", type=float, default=0.02)
    p.add_argument(
        "--lagfg-prior-source",
        type=str,
        default="obs_smooth",
        choices=["obs_smooth", "obs_raw", "truth", "constant"],
        help="Source for fg_lagcorr_mean/sigma when lag_fg_corr is enabled.",
    )
    p.add_argument("--lagfg-const-mean", type=float, default=0.99)
    p.add_argument("--lagfg-const-sigma", type=float, default=0.05)

    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run separation_cli.py.",
    )
    p.add_argument("--dry-run", action="store_true", help="Generate configs/candidates but do not execute jobs.")
    return p.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(float(x)) for x in _parse_csv_tokens(text)]


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in _parse_csv_tokens(text)]


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo_f = float(lo)
    hi_f = float(hi)
    if not (math.isfinite(lo_f) and math.isfinite(hi_f) and lo_f > 0.0 and hi_f > lo_f):
        raise ValueError("Invalid log-uniform bounds.")
    return float(math.exp(float(rng.uniform(math.log(lo_f), math.log(hi_f)))))


def _fmt_float_token(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace(".", "p").replace("-", "m")


def generate_candidates(args: argparse.Namespace) -> List[CandidateSpec]:
    rng = np.random.default_rng(int(args.seed))
    controls: List[CandidateSpec] = []

    def _base_control(
        name: str,
        *,
        poly_x_mode: str,
        init_mode: str,
        poly_degree: int,
        poly_sigma: float,
        poly_weight: float,
        lagfg: bool,
    ) -> CandidateSpec:
        # Baseline anchored to the previously-best poly config, but with new init/poly_x_mode levers.
        return CandidateSpec(
            name=name,
            beta=0.5,
            gamma=0.6,
            data_error=0.005,
            fg_smooth_mode="diff2_l2",
            fg_smooth_mean=0.002,
            fg_smooth_sigma=0.004,
            fg_smooth_huber_delta=1.0,
            eor_prior_sigma=0.02,
            eor_amp_prior_mode="slice_rms_hinge",
            eor_hybrid_voxel_factor=5.0,
            eor_hybrid_voxel_weight=0.1,
            eor_amp_threshold=float(args.eor_amp_threshold),
            poly_weight=float(poly_weight),
            poly_degree=int(poly_degree),
            poly_sigma=float(poly_sigma),
            poly_x_mode=str(poly_x_mode),
            optimizer_name="adam",
            lr=4e-4,
            lr_fg_factor=1.0,
            lr_scheduler="plateau",
            lr_plateau_patience=400,
            lr_plateau_factor=0.5,
            lr_plateau_min_delta=1e-4,
            lr_plateau_cooldown=80,
            lr_min=1e-6,
            alt_update_mode="none",
            alt_fg_steps=50,
            alt_eor_steps=1,
            extra_loss_start_iter=500,
            extra_loss_ramp_iters=0,
            init_mode=str(init_mode),
            lagfg_enabled=bool(lagfg),
            lagcorr_weight=0.03 if lagfg else 0.0,
            lagcorr_feature="diff1",
            lagcorr_unit=str(args.lagcorr_unit),
            lagcorr_pair_sampling=str(args.lagcorr_pair_sampling),
            lagcorr_random_seed=int(args.lagcorr_random_seed),
            lagcorr_max_pairs=128,
            lagcorr_spatial_pool=4,
            lagcorr_rms_min=0.01,
            lagcorr_sigma_floor=float(args.lagcorr_sigma_floor),
        )

    controls.extend(
        [
            _base_control(
                "ctrl00_poly_lin_init_smooth_zero",
                poly_x_mode="lin",
                init_mode="smooth_zero",
                poly_degree=3,
                poly_sigma=0.05,
                poly_weight=1.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl01_poly_lin_init_smooth_resid",
                poly_x_mode="lin",
                init_mode="smooth_residual",
                poly_degree=3,
                poly_sigma=0.05,
                poly_weight=1.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl02_poly_lin_init_poly_resid",
                poly_x_mode="lin",
                init_mode="poly_residual",
                poly_degree=3,
                poly_sigma=0.05,
                poly_weight=1.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl03_poly_log_init_poly_resid",
                poly_x_mode="log",
                init_mode="poly_residual",
                poly_degree=3,
                poly_sigma=0.05,
                poly_weight=1.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl04_poly_log_d3_s0p01_pw3_init_poly_resid",
                poly_x_mode="log",
                init_mode="poly_residual",
                poly_degree=3,
                poly_sigma=0.01,
                poly_weight=3.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl05_poly_log_d4_s0p01_pw3_init_poly_resid",
                poly_x_mode="log",
                init_mode="poly_residual",
                poly_degree=4,
                poly_sigma=0.01,
                poly_weight=3.0,
                lagfg=False,
            ),
            _base_control(
                "ctrl06_poly_log_d3_s0p01_pw3_init_poly_resid_lagfg",
                poly_x_mode="log",
                init_mode="poly_residual",
                poly_degree=3,
                poly_sigma=0.01,
                poly_weight=3.0,
                lagfg=True,
            ),
        ]
    )
    controls = controls[: max(0, int(args.num_controls))]

    fg_smooth_modes = _parse_csv_tokens(args.fg_smooth_modes)
    poly_degrees = _parse_int_list(args.poly_degrees)
    poly_x_modes = _parse_csv_tokens(args.poly_x_modes)
    poly_weights = _parse_float_list(args.poly_weights)
    fg_means = _parse_float_list(args.fg_smooth_mean_list)
    eor_sigmas = _parse_float_list(args.eor_prior_sigma_list)
    eor_amp_modes = _parse_csv_tokens(args.eor_amp_prior_modes)
    optimizer_names = _parse_csv_tokens(args.optimizer_names)
    lr_fg_factors = _parse_float_list(args.lr_fg_factor_list)
    lr_schedulers = _parse_csv_tokens(args.lr_schedulers)
    plateau_pat = _parse_int_list(args.plateau_patience_list)
    plateau_fac = _parse_float_list(args.plateau_factor_list)
    plateau_mind = _parse_float_list(args.plateau_min_delta_list)
    plateau_cd = _parse_int_list(args.plateau_cooldown_list)
    init_modes = _parse_csv_tokens(args.init_modes)
    alt_modes = _parse_csv_tokens(args.alt_update_modes)
    alt_fg_steps = _parse_int_list(args.alt_fg_steps_list)
    alt_eor_steps = _parse_int_list(args.alt_eor_steps_list)
    extra_start = _parse_int_list(args.extra_loss_start_list)
    extra_ramp = _parse_int_list(args.extra_loss_ramp_list)

    lagcorr_features = _parse_csv_tokens(args.lagcorr_features)
    lagcorr_max_pairs_list = _parse_int_list(args.lagcorr_max_pairs_list)
    lagcorr_pools = _parse_int_list(args.lagcorr_spatial_pool_list)
    lagcorr_rms_min_list = _parse_float_list(args.lagcorr_rms_min_list)

    out: List[CandidateSpec] = []
    out.extend(controls)

    remaining = max(0, int(args.num_candidates) - len(out))
    for i in range(remaining):
        poly_degree = int(rng.choice(poly_degrees))
        poly_x_mode = str(rng.choice(poly_x_modes)).strip().lower()
        poly_sigma = _log_uniform(rng, float(args.poly_sigma_min), float(args.poly_sigma_max))
        poly_weight = float(rng.choice(poly_weights))

        beta = _log_uniform(rng, float(args.beta_min), float(args.beta_max))
        gamma = _log_uniform(rng, float(args.gamma_min), float(args.gamma_max))
        data_error = _log_uniform(rng, float(args.data_error_min), float(args.data_error_max))

        fg_smooth_mode = str(rng.choice(fg_smooth_modes)).strip()
        fg_smooth_mean = float(rng.choice(fg_means))
        fg_smooth_sigma = _log_uniform(rng, float(args.fg_smooth_sigma_min), float(args.fg_smooth_sigma_max))

        eor_prior_sigma = float(rng.choice(eor_sigmas))
        eor_amp_mode = str(rng.choice(eor_amp_modes)).strip()

        opt_name = str(rng.choice(optimizer_names)).strip().lower()
        lr = _log_uniform(rng, float(args.lr_min), float(args.lr_max))
        lr_fg_factor = float(rng.choice(lr_fg_factors))
        lr_sched = str(rng.choice(lr_schedulers)).strip().lower()
        pat = int(rng.choice(plateau_pat))
        fac = float(rng.choice(plateau_fac))
        mind = float(rng.choice(plateau_mind))
        cd = int(rng.choice(plateau_cd))

        init_mode = str(rng.choice(init_modes)).strip().lower()
        alt_mode = str(rng.choice(alt_modes)).strip().lower()
        if alt_mode == "fg_then_eor":
            a_fg = int(rng.choice(alt_fg_steps))
            a_eor = int(rng.choice(alt_eor_steps))
        else:
            a_fg = 50
            a_eor = 1

        ex_start = int(rng.choice(extra_start))
        ex_ramp = int(rng.choice(extra_ramp))

        lagfg = bool(rng.uniform(0.0, 1.0) < float(args.lagfg_prob))
        lagcorr_weight = _log_uniform(rng, float(args.lagcorr_weight_min), float(args.lagcorr_weight_max)) if lagfg else 0.0
        lagfeat = str(rng.choice(lagcorr_features)).strip().lower()
        lagpool = int(rng.choice(lagcorr_pools))
        lagpairs = int(rng.choice(lagcorr_max_pairs_list))
        lagrms = float(rng.choice(lagcorr_rms_min_list))

        name = (
            f"r{i:04d}_px{poly_x_mode[0]}d{poly_degree}_ps{_fmt_float_token(poly_sigma)}"
            f"_pw{_fmt_float_token(poly_weight)}"
            f"_b{_fmt_float_token(beta)}_g{_fmt_float_token(gamma)}"
            f"_lr{_fmt_float_token(lr)}_lrf{_fmt_float_token(lr_fg_factor)}"
            f"_init{init_mode[:3]}"
            + (f"_lag{_fmt_float_token(lagcorr_weight)}{lagfeat[0]}p{lagpool}" if lagfg else "_nolag")
        )
        out.append(
            CandidateSpec(
                name=name,
                beta=float(beta),
                gamma=float(gamma),
                data_error=float(data_error),
                fg_smooth_mode=str(fg_smooth_mode),
                fg_smooth_mean=float(fg_smooth_mean),
                fg_smooth_sigma=float(fg_smooth_sigma),
                fg_smooth_huber_delta=1.0,
                eor_prior_sigma=float(eor_prior_sigma),
                eor_amp_prior_mode=str(eor_amp_mode),
                eor_hybrid_voxel_factor=5.0,
                eor_hybrid_voxel_weight=0.1,
                eor_amp_threshold=float(args.eor_amp_threshold),
                poly_weight=float(poly_weight),
                poly_degree=int(poly_degree),
                poly_sigma=float(poly_sigma),
                poly_x_mode=str(poly_x_mode),
                optimizer_name=str(opt_name),
                lr=float(lr),
                lr_fg_factor=float(lr_fg_factor),
                lr_scheduler=str(lr_sched),
                lr_plateau_patience=int(pat),
                lr_plateau_factor=float(fac),
                lr_plateau_min_delta=float(mind),
                lr_plateau_cooldown=int(cd),
                lr_min=1e-6,
                alt_update_mode=str(alt_mode),
                alt_fg_steps=int(a_fg),
                alt_eor_steps=int(a_eor),
                extra_loss_start_iter=int(ex_start),
                extra_loss_ramp_iters=int(ex_ramp),
                init_mode=str(init_mode),
                lagfg_enabled=bool(lagfg),
                lagcorr_weight=float(lagcorr_weight),
                lagcorr_feature=str(lagfeat),
                lagcorr_unit=str(args.lagcorr_unit),
                lagcorr_pair_sampling=str(args.lagcorr_pair_sampling),
                lagcorr_random_seed=int(args.lagcorr_random_seed),
                lagcorr_max_pairs=int(lagpairs),
                lagcorr_spatial_pool=int(lagpool),
                lagcorr_rms_min=float(lagrms),
                lagcorr_sigma_floor=float(args.lagcorr_sigma_floor),
            )
        )
    return out


def _extract_cut_indices(shape: Tuple[int, int, int], cut_frac: float) -> Tuple[int, int, int, int]:
    _, ny, nx = shape
    size = int(round(float(cut_frac) * float(min(ny, nx))))
    size = max(1, min(size, ny, nx))
    x0 = (nx - size) // 2
    y0 = (ny - size) // 2
    return int(x0), int(x0 + size), int(y0), int(y0 + size)


def _load_cube_cut(path: Path, *, cut: Tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = cut
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        # Expected order (F, Y, X)
        out = np.asarray(data[:, y0:y1, x0:x1], dtype=np.float32)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    worst_frac = 0.20
    k = max(1, int(math.ceil(float(finite.size) * worst_frac)))
    return float(np.mean(np.sort(finite)[:k]))


def _frequency_correlations(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Shape mismatch in correlation.")
    f = a.shape[0]
    out = np.full((f,), np.nan, dtype=np.float64)
    a2 = a.reshape(f, -1).astype(np.float64, copy=False)
    b2 = b.reshape(f, -1).astype(np.float64, copy=False)
    a2 = a2 - np.mean(a2, axis=1, keepdims=True)
    b2 = b2 - np.mean(b2, axis=1, keepdims=True)
    na = np.linalg.norm(a2, axis=1)
    nb = np.linalg.norm(b2, axis=1)
    denom = np.maximum(na * nb, float(eps))
    out = np.sum(a2 * b2, axis=1) / denom
    return out.astype(np.float64)


def _read_eor_window_metrics(power_dir: Path) -> Dict[str, object]:
    metrics_path = power_dir / "power2d_eor_window_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {f"ps2d_win_{k}": v for k, v in metrics.items()}


_ITER_RE = re.compile(r"^\\[iter\\s+(\\d+)\\]\\s+total=([-0-9eE+.]+)")
_CHECK_RE = re.compile(r"^\\[check\\]\\s+iter\\s+(\\d+):\\s+mean EoR corr=([-0-9eE+.]+)")


def _parse_convergence_from_log(log_path: Path) -> Dict[str, object]:
    iters: List[int] = []
    totals: List[float] = []
    checks_i: List[int] = []
    checks_corr: List[float] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _ITER_RE.match(line.strip())
            if m:
                iters.append(int(m.group(1)))
                totals.append(float(m.group(2)))
                continue
            m = _CHECK_RE.match(line.strip())
            if m:
                checks_i.append(int(m.group(1)))
                checks_corr.append(float(m.group(2)))
    except OSError:
        return {"converged": False, "conv_error": "read_failed"}

    def _last_delta(vals: List[float], n: int) -> Optional[float]:
        if len(vals) < n + 1:
            return None
        return float(vals[-1] - vals[-(n + 1)])

    total_delta_2 = _last_delta(totals, 2)
    corr_delta_2 = _last_delta(checks_corr, 2)
    total_rel_delta_2: Optional[float] = None
    if total_delta_2 is not None and totals:
        denom = abs(float(totals[-1])) + 1e-12
        total_rel_delta_2 = float(total_delta_2) / float(denom)
    out: Dict[str, object] = {
        "conv_total_delta_last2": total_delta_2,
        "conv_total_rel_delta_last2": total_rel_delta_2,
        "conv_eor_corr_delta_last2": corr_delta_2,
        "conv_total_last": totals[-1] if totals else None,
        "conv_iter_last": iters[-1] if iters else None,
        "conv_check_iter_last": checks_i[-1] if checks_i else None,
        "conv_check_corr_last": checks_corr[-1] if checks_corr else None,
    }

    corr_stable = True
    if corr_delta_2 is not None:
        corr_stable = abs(float(corr_delta_2)) < 2e-3
    total_stable = True
    if total_rel_delta_2 is not None:
        total_stable = abs(float(total_rel_delta_2)) < 2e-2
    out["converged"] = bool(corr_stable and total_stable)
    return out


def _avg_pool_xy(cube: np.ndarray, pool: int) -> np.ndarray:
    if pool <= 1:
        return cube
    if cube.ndim != 3:
        raise ValueError("Expected cube with shape (F, Y, X).")
    f, ny, nx = cube.shape
    ny_t = (ny // pool) * pool
    nx_t = (nx // pool) * pool
    if ny_t <= 0 or nx_t <= 0:
        raise ValueError("Pooling factor too large.")
    view = cube[:, :ny_t, :nx_t].reshape(f, ny_t // pool, pool, nx_t // pool, pool)
    return view.mean(axis=(2, 4))


def _smooth_initial_foreground_np(y: np.ndarray) -> np.ndarray:
    if y.ndim != 3:
        raise ValueError("Expected cube with shape (F, Y, X).")
    sm = y.astype(np.float32, copy=True)
    if sm.shape[0] >= 3:
        sm[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    return sm


def _lagcorr_stats_from_cube(
    cube_in: np.ndarray,
    *,
    lag_channels: Sequence[int],
    feature: str,
    spatial_pool: int,
    max_pairs: Optional[int],
    pair_sampling: str,
    seed: int,
    rms_min: float,
    sigma_floor: float,
    eps: float = 1e-12,
) -> Tuple[List[float], List[float]]:
    feat = str(feature).strip().lower()
    if feat == "raw":
        cube = cube_in
    elif feat == "diff1":
        if cube_in.shape[0] < 2:
            raise ValueError("diff1 requires at least 2 channels.")
        cube = np.diff(cube_in, axis=0)
    else:
        raise ValueError("feature must be raw or diff1.")
    if int(spatial_pool) > 1:
        cube = _avg_pool_xy(cube, int(spatial_pool))
    f = cube.shape[0]
    flat = cube.reshape(f, -1).astype(np.float64, copy=False)
    centered = flat - np.mean(flat, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms = np.maximum(norms, float(eps))
    if float(rms_min) > 0.0:
        scale = math.sqrt(max(1.0, float(centered.shape[1])))
        slice_rms = norms / scale
        valid_slice = slice_rms >= float(rms_min)
    else:
        valid_slice = np.ones_like(norms, dtype=bool)

    rng = np.random.default_rng(int(seed))
    means: List[float] = []
    sigmas: List[float] = []
    for lag in lag_channels:
        lag_i = int(lag)
        if lag_i < 1 or lag_i >= f:
            raise ValueError(f"Invalid lag {lag_i} for f={f}.")
        num_total = f - lag_i
        if max_pairs is None or int(max_pairs) <= 0 or int(max_pairs) >= num_total:
            idx = np.arange(num_total, dtype=np.int64)
        else:
            k = int(max_pairs)
            if str(pair_sampling).strip().lower() == "head":
                idx = np.arange(k, dtype=np.int64)
            elif str(pair_sampling).strip().lower() == "random":
                idx = rng.choice(num_total, size=k, replace=False)
                idx = np.sort(idx)
            else:
                raise ValueError("pair_sampling must be head or random.")
        jdx = idx + lag_i
        pair_valid = valid_slice[idx] & valid_slice[jdx]
        if not np.any(pair_valid):
            means.append(0.0)
            sigmas.append(float(max(1e-6, float(sigma_floor))))
            continue
        idx = idx[pair_valid]
        jdx = jdx[pair_valid]
        a = centered[idx]
        b = centered[jdx]
        dot = np.sum(a * b, axis=1)
        denom = norms[idx] * norms[jdx]
        denom = np.maximum(denom, float(eps))
        corr = dot / denom
        mu = float(np.mean(corr))
        var = float(np.var(corr))
        sigma = math.sqrt(max(0.0, var))
        sigma = float(max(sigma, float(sigma_floor)))
        means.append(mu)
        sigmas.append(sigma)
    return means, sigmas


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    return [str(args.python_bin), str(cli_path), "--config", str(config_path)]


def _parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in _parse_csv_tokens(text):
        if ":" not in token:
            raise ValueError(f"Bad gpu-map token: {token}")
        name, idx = token.split(":", 1)
        out[name.strip()] = int(idx.strip())
    return out


def _build_config(
    *,
    job: JobSpec,
    args: argparse.Namespace,
    code_dir: Path,
    fg_lagcorr_cache: Dict[Tuple[str, str, int], Tuple[List[float], List[float]]],
) -> Dict[str, object]:
    cand = job.candidate
    power_cfg_path = Path(args.power_config)
    if not power_cfg_path.is_absolute():
        power_cfg_path = (code_dir / str(power_cfg_path)).resolve()

    extra_terms: List[str] = ["poly_reparam"]
    if bool(cand.lagfg_enabled):
        extra_terms.append("lagcorr")

    cfg: Dict[str, object] = {
        "input_cube": str(job.input_cube),
        "fg_output": str(job.fg_output),
        "eor_output": str(job.eor_output),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(cand.lr),
            "lr_fg_factor": float(cand.lr_fg_factor),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(job.gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": list(extra_terms),
            "extra_loss_start_iter": int(cand.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(cand.extra_loss_ramp_iters),
            "optimizer_name": str(cand.optimizer_name),
            "momentum": 0.9,
            "lr_scheduler": str(cand.lr_scheduler),
            "lr_plateau_patience": int(cand.lr_plateau_patience),
            "lr_plateau_factor": float(cand.lr_plateau_factor),
            "lr_plateau_min_delta": float(cand.lr_plateau_min_delta),
            "lr_plateau_cooldown": int(cand.lr_plateau_cooldown),
            "lr_min": float(cand.lr_min),
            "alt_update_mode": str(cand.alt_update_mode),
            "alt_fg_steps": int(cand.alt_fg_steps),
            "alt_eor_steps": int(cand.alt_eor_steps),
            "freq_start_mhz": float(job.freq_start_mhz),
            "freq_delta_mhz": float(args.freq_delta_mhz),
            "poly_degree": int(cand.poly_degree),
            "poly_sigma": float(cand.poly_sigma),
            "poly_x_mode": str(cand.poly_x_mode),
        },
        "cut_xy": {
            "enabled": True,
            "unit": "frac",
            "center_x": 0.5,
            "center_y": 0.5,
            "size": float(args.cut_size_frac),
        },
        "weights": {
            "alpha": 1.0,
            "beta": float(cand.beta),
            "gamma": float(cand.gamma),
            "corr_weight": 0.0,
            "lagcorr_weight": float(cand.lagcorr_weight) if cand.lagfg_enabled else 0.0,
            "lagcorr_fg_component_weight": 1.0 if cand.lagfg_enabled else 0.0,
            "lagcorr_eor_component_weight": 0.0,
            "lagcorr_gap_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": float(cand.poly_weight),
            "fg_logcurv_weight": 0.0,
            "fg_lowrank_weight": 0.0,
            "eor_lagshape_weight": 0.0,
            "eor_iso_weight": 0.0,
            "eor_mean_weight": 0.0,
            "eor_hf_weight": 0.0,
        },
        "priors": {
            "data_error": float(cand.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(cand.eor_prior_sigma),
            "eor_prior_amp_threshold": float(cand.eor_amp_threshold),
            "eor_amp_prior_mode": str(cand.eor_amp_prior_mode),
            "eor_hybrid_voxel_factor": float(cand.eor_hybrid_voxel_factor),
            "eor_hybrid_voxel_weight": float(cand.eor_hybrid_voxel_weight),
            "fg_smooth_mode": str(cand.fg_smooth_mode),
            "fg_smooth_mean": float(cand.fg_smooth_mean),
            "fg_smooth_sigma": float(cand.fg_smooth_sigma),
            "fg_smooth_huber_delta": float(cand.fg_smooth_huber_delta),
        },
        "init": {"init_fg_cube": "", "init_eor_cube": "", "init_mode": str(cand.init_mode)},
        "evaluation": {
            "true_eor_cube": str(job.eor_true_cube),
            "diagnose_input": False,
            "enable_corr_check": True,
            "corr_check_every": max(50, int(args.print_every)),
            "corr_plot": str(job.run_dir / "eor_corr.png"),
        },
        "power": {
            "power_config": str(power_cfg_path),
            "power_output_dir": str(job.run_dir / "powerspec"),
        },
        "scan_meta": {"candidate_name": cand.name},
    }

    if cand.lagfg_enabled:
        key = (job.dataset_name, str(cand.lagcorr_feature), int(cand.lagcorr_spatial_pool))
        if key not in fg_lagcorr_cache:
            raise ValueError(f"Missing fg lagcorr cache for key={key}")
        mu, sig = fg_lagcorr_cache[key]
        cfg["priors"].update(
            {
                "lagcorr_unit": str(cand.lagcorr_unit),
                "lagcorr_feature": str(cand.lagcorr_feature),
                "lagcorr_spatial_pool": int(cand.lagcorr_spatial_pool),
                "lagcorr_max_pairs": None if cand.lagcorr_max_pairs is None else int(cand.lagcorr_max_pairs),
                "lagcorr_pair_sampling": str(cand.lagcorr_pair_sampling),
                "lagcorr_random_seed": int(cand.lagcorr_random_seed),
                "lagcorr_rms_min": float(cand.lagcorr_rms_min),
                "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                "lagcorr_lag_weights": 1.0,
                "fg_lagcorr_mean": list(mu),
                "fg_lagcorr_sigma": list(sig),
            }
        )
    return cfg


def _run_job_result_only(
    *,
    job: JobSpec,
    return_code: int,
    runtime: float,
    ds_cache: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "candidate": job.candidate.name,
        "dataset": job.dataset_name,
        "status": "ok" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "runtime_sec": float(runtime),
        "config_path": str(job.config_path),
        "log_path": str(job.log_path),
        "eor_output": str(job.eor_output),
        "fg_output": str(job.fg_output),
        "beta": float(job.candidate.beta),
        "gamma": float(job.candidate.gamma),
        "data_error": float(job.candidate.data_error),
        "fg_smooth_mode": str(job.candidate.fg_smooth_mode),
        "fg_smooth_mean": float(job.candidate.fg_smooth_mean),
        "fg_smooth_sigma": float(job.candidate.fg_smooth_sigma),
        "eor_prior_sigma": float(job.candidate.eor_prior_sigma),
        "eor_amp_prior_mode": str(job.candidate.eor_amp_prior_mode),
        "eor_amp_threshold": float(job.candidate.eor_amp_threshold),
        "poly_weight": float(job.candidate.poly_weight),
        "poly_degree": int(job.candidate.poly_degree),
        "poly_sigma": float(job.candidate.poly_sigma),
        "poly_x_mode": str(job.candidate.poly_x_mode),
        "init_mode": str(job.candidate.init_mode),
        "optimizer_name": str(job.candidate.optimizer_name),
        "lr": float(job.candidate.lr),
        "lr_fg_factor": float(job.candidate.lr_fg_factor),
        "lr_scheduler": str(job.candidate.lr_scheduler),
        "lr_plateau_patience": int(job.candidate.lr_plateau_patience),
        "alt_update_mode": str(job.candidate.alt_update_mode),
        "alt_fg_steps": int(job.candidate.alt_fg_steps),
        "alt_eor_steps": int(job.candidate.alt_eor_steps),
        "extra_loss_start_iter": int(job.candidate.extra_loss_start_iter),
        "extra_loss_ramp_iters": int(job.candidate.extra_loss_ramp_iters),
        "lagfg_enabled": bool(job.candidate.lagfg_enabled),
        "lagcorr_weight": float(job.candidate.lagcorr_weight),
        "lagcorr_feature": str(job.candidate.lagcorr_feature),
        "lagcorr_spatial_pool": int(job.candidate.lagcorr_spatial_pool),
        "lagcorr_max_pairs": (None if job.candidate.lagcorr_max_pairs is None else int(job.candidate.lagcorr_max_pairs)),
        "lagcorr_rms_min": float(job.candidate.lagcorr_rms_min),
    }
    row.update(_parse_convergence_from_log(job.log_path))

    if int(return_code) != 0:
        return row
    if not job.eor_output.exists():
        row["status"] = "failed"
        row["note"] = f"Missing EoR output: {job.eor_output}"
        return row

    cache = ds_cache[job.dataset_name]
    true_eor = cache["true_eor"]
    assert isinstance(true_eor, np.ndarray)
    with fits.open(job.eor_output, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)

    if eor_est.shape != true_eor.shape:
        # Best-effort center crop if shapes drift.
        f, ny, nx = true_eor.shape
        fy, fx = eor_est.shape[1], eor_est.shape[2]
        y0 = max(0, (ny - fy) // 2)
        x0 = max(0, (nx - fx) // 2)
        true_eor = true_eor[:, y0 : y0 + fy, x0 : x0 + fx]

    corr = _frequency_correlations(eor_est, true_eor)
    corr_profile_path = job.run_dir / "eor_corr_profile.csv"
    corr_profile_path.parent.mkdir(parents=True, exist_ok=True)
    with corr_profile_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, v in enumerate(corr.tolist()):
            w.writerow([int(i), float(v)])
    row["eor_corr_profile_path"] = str(corr_profile_path)
    finite = corr[np.isfinite(corr)]
    row["eor_corr_count"] = int(finite.size)
    if finite.size == 0:
        row["status"] = "failed"
        row["note"] = "No finite per-frequency correlations."
        return row
    row["eor_corr_mean"] = float(np.mean(finite))
    row["eor_corr_median"] = float(np.median(finite))
    row["eor_corr_p10"] = float(np.percentile(finite, 10))
    row["eor_corr_min"] = float(np.min(finite))
    row["eor_corr_max"] = float(np.max(finite))
    row["eor_corr_score"] = float(_score_from_corr(corr))

    row.update(_read_eor_window_metrics(job.run_dir / "powerspec"))
    return row


def _candidate_summary(rows: Sequence[Dict[str, object]], exclude_datasets: Sequence[str]) -> List[Dict[str, object]]:
    exclude = set(exclude_datasets)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        if not bool(row.get("converged", True)):
            continue
        if str(row.get("dataset")) in exclude:
            continue
        grouped.setdefault(str(row.get("candidate")), []).append(row)

    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        scores = [float(x.get("eor_corr_score")) for x in items if x.get("eor_corr_score") is not None]
        ps_mad = [float(x.get("ps2d_win_log10_mad")) for x in items if x.get("ps2d_win_log10_mad") is not None]
        out.append(
            {
                "candidate": cand,
                "n_ok_converged": int(len(items)),
                "eor_corr_score_mean": float(np.mean(scores)) if scores else float("nan"),
                "ps2d_win_log10_mad_mean": float(np.mean(ps_mad)) if ps_mad else float("nan"),
            }
        )
    out.sort(key=lambda r: (-(r["eor_corr_score_mean"]), r["ps2d_win_log10_mad_mean"]))
    for i, row in enumerate(out, start=1):
        row["rank"] = int(i)
    return out


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Poly + lag_fg_corr + Optim Scan Summary\n\n")
    lines.append("## Meta\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("| rank | candidate | n_ok_converged | eor_corr_score_mean | ps2d_win_log10_mad_mean |\n")
    lines.append("|---:|---|---:|---:|---:|\n")
    for row in ranked:
        lines.append(
            f"| {int(row['rank'])} | {row['candidate']} | {int(row['n_ok_converged'])} | "
            f"{float(row['eor_corr_score_mean']):.6f} | {float(row['ps2d_win_log10_mad_mean']):.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "code" / "3dnet" if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet"))
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"poly_lagfg_optim_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir)
    enabled_names = parse_dataset_names(args.datasets)
    datasets = filter_datasets(datasets_all, enabled_names)
    if not datasets:
        raise ValueError("No datasets selected.")
    gpu_map = _parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing gpu-map entry for dataset '{ds.name}'.")

    candidates = generate_candidates(args)
    if args.candidate_names.strip():
        allow = {t.strip() for t in args.candidate_names.split(",") if t.strip()}
        known = {c.name for c in candidates}
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

    # Prepare dataset cache: cut indices + true cubes + obs (for lagfg priors).
    ds_cache: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        with fits.open(ds.input_cube, memmap=True) as hdul:
            in_shape = tuple(int(v) for v in hdul[0].data.shape)
        cut = _extract_cut_indices(in_shape, float(args.cut_size_frac))
        ds_cache[ds.name] = {
            "cut": cut,
            "obs": _load_cube_cut(ds.input_cube, cut=cut),
            "true_eor": _load_cube_cut(ds.eor_true_cube, cut=cut),
            "true_fg": _load_cube_cut(ds.fg_true_cube, cut=cut),
        }

    # Precompute fg_lagcorr_mean/sigma caches for all (ds, feature, pool) combos used.
    fg_lagcorr_cache: Dict[Tuple[str, str, int], Tuple[List[float], List[float]]] = {}
    lag_channels: List[int] = []
    if str(args.lagcorr_unit).strip().lower() == "chan":
        lag_channels = [max(1, int(round(float(x)))) for x in LAG_INTERVALS_MHZ]
    else:
        for mhz in LAG_INTERVALS_MHZ:
            lag_channels.append(max(1, int(round(float(mhz) / float(args.freq_delta_mhz)))))

    need_lagfg = any(bool(c.lagfg_enabled) for c in candidates)
    if need_lagfg:
        source = str(args.lagfg_prior_source).strip().lower()
        keys_needed = {(ds.name, str(c.lagcorr_feature), int(c.lagcorr_spatial_pool)) for ds in datasets for c in candidates if c.lagfg_enabled}
        for ds_name, feat, pool in sorted(keys_needed):
            cache = ds_cache[ds_name]
            obs = cache["obs"]
            true_fg = cache["true_fg"]
            assert isinstance(obs, np.ndarray) and isinstance(true_fg, np.ndarray)
            if source == "truth":
                cube_for_prior = true_fg
            elif source == "obs_raw":
                cube_for_prior = obs
            elif source == "obs_smooth":
                cube_for_prior = _smooth_initial_foreground_np(obs)
            elif source == "constant":
                mu = [float(args.lagfg_const_mean)] * len(lag_channels)
                sig = [float(args.lagfg_const_sigma)] * len(lag_channels)
                fg_lagcorr_cache[(ds_name, feat, pool)] = (mu, sig)
                continue
            else:
                raise ValueError("Unsupported lagfg_prior_source.")

            mu, sig = _lagcorr_stats_from_cube(
                cube_for_prior,
                lag_channels=lag_channels,
                feature=feat,
                spatial_pool=int(pool),
                max_pairs=None,
                pair_sampling=str(args.lagcorr_pair_sampling),
                seed=int(args.lagcorr_random_seed),
                rms_min=0.0,
                sigma_floor=float(args.lagcorr_sigma_floor),
            )
            fg_lagcorr_cache[(ds_name, feat, pool)] = (mu, sig)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "datasets": [ds.name for ds in datasets],
        "exclude_from_ranking": str(args.exclude_from_ranking),
        "lagfg_prior_source": str(args.lagfg_prior_source),
        "fg_lagcorr_cache_keys": [list(k) for k in sorted(fg_lagcorr_cache.keys())],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] generated {len(candidates)} candidates under {output_dir}")
        return 0

    rows: List[Dict[str, object]] = []
    max_jobs = max(1, int(args.max_concurrent_jobs))
    for idx, cand in enumerate(candidates, start=1):
        print(f"[candidate {idx}/{len(candidates)}] {cand.name}")
        jobs: List[JobSpec] = []
        for ds in datasets:
            run_dir = output_dir / cand.name / ds.name
            jobs.append(
                JobSpec(
                    dataset_name=ds.name,
                    input_cube=ds.input_cube,
                    fg_true_cube=ds.fg_true_cube,
                    eor_true_cube=ds.eor_true_cube,
                    freq_start_mhz=float(ds.freq_start_mhz),
                    gpu_index=int(gpu_map[ds.name]),
                    candidate=cand,
                    run_dir=run_dir,
                    config_path=run_dir / "config.json",
                    log_path=run_dir / "run.log",
                    fg_output=run_dir / "fg_est.fits",
                    eor_output=run_dir / "eor_est.fits",
                )
            )

        active: List[Tuple[subprocess.Popen[str], JobSpec, float, object]] = []
        queued = list(jobs)
        while queued or active:
            active_gpus = {int(j.gpu_index) for (_, j, _, _) in active}
            while queued and len(active) < max_jobs:
                pick_idx: Optional[int] = None
                for i, cand_job in enumerate(queued):
                    if int(cand_job.gpu_index) in active_gpus:
                        continue
                    pick_idx = i
                    break
                if pick_idx is None:
                    break
                job = queued.pop(pick_idx)
                job.run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_config(job=job, args=args, code_dir=code_dir, fg_lagcorr_cache=fg_lagcorr_cache)
                with job.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(cfg, handle, indent=2)
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active.append((proc, job, time.time(), log_handle))
                active_gpus.add(int(job.gpu_index))
                print(f"  [launch] {job.dataset_name} gpu={job.gpu_index} pid={proc.pid}")

            still_active: List[Tuple[subprocess.Popen[str], JobSpec, float, object]] = []
            for proc, job, t0, log_handle in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, t0, log_handle))
                    continue
                log_handle.close()
                runtime = time.time() - t0
                row = _run_job_result_only(job=job, return_code=int(ret), runtime=float(runtime), ds_cache=ds_cache)
                rows.append(row)
                print(
                    f"  [done] {job.dataset_name} status={row['status']} "
                    f"converged={row.get('converged')} score={row.get('eor_corr_score')} "
                    f"ps_mad={row.get('ps2d_win_log10_mad')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "poly_lagfg_optim_scan_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows, exclude_datasets=_parse_csv_tokens(args.exclude_from_ranking))
    rank_csv = output_dir / "poly_lagfg_optim_scan_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "poly_lagfg_optim_scan_summary.md", ranked, manifest)
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

