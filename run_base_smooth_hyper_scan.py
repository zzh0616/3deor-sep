#!/usr/bin/env python3
"""
Base-only smoothness hyperparameter scan.

Design constraints for this scan:
1) Only run base objective terms (no lagcorr/corr/rfft/poly extra terms).
2) Evaluate candidates primarily by per-frequency EoR correlation profile
   between recovered and injected EoR cubes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

VALID_FG_SMOOTH_MODES: Tuple[str, ...] = ("diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    smooth_mode: str
    beta: float
    gamma: float
    eor_prior_sigma: float
    prior_source: str  # reference_robust/reference_std/explicit_scalar
    use_robust_fg_stats: bool
    mae_to_sigma_factor: float
    fg_smooth_mean: Optional[float]
    fg_smooth_sigma: Optional[float]
    fg_smooth_huber_delta: float
    optimizer_name: str
    lr: float
    momentum: float
    lr_scheduler: str
    lr_plateau_patience: int
    lr_plateau_factor: float
    lr_plateau_min_delta: float
    lr_plateau_cooldown: int
    lr_min: float

    def key(self) -> Tuple[object, ...]:
        return (
            self.smooth_mode,
            round(self.beta, 8),
            round(self.gamma, 8),
            round(self.eor_prior_sigma, 8),
            self.prior_source,
            bool(self.use_robust_fg_stats),
            round(self.mae_to_sigma_factor, 8),
            None if self.fg_smooth_mean is None else round(float(self.fg_smooth_mean), 10),
            None if self.fg_smooth_sigma is None else round(float(self.fg_smooth_sigma), 10),
            round(self.fg_smooth_huber_delta, 8),
            self.optimizer_name,
            round(self.lr, 12),
            round(self.momentum, 8),
            self.lr_scheduler,
            int(self.lr_plateau_patience),
            round(self.lr_plateau_factor, 8),
            round(self.lr_plateau_min_delta, 12),
            int(self.lr_plateau_cooldown),
            round(self.lr_min, 12),
        )


@dataclass
class JobSpec:
    dataset: DatasetSpec
    candidate: CandidateSpec
    gpu_index: int
    run_dir: Path
    config_path: Path
    log_path: Path
    fg_output: Path
    eor_output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base-only smoothness hyperparameter scan.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="3dnet dir (default <work-root>/3dnet).")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/base_smooth_scan_<timestamp>).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="phase_a",
        choices=["phase_a", "phase_b", "phase_c"],
        help="Phase preset for scan density.",
    )
    parser.add_argument(
        "--samples-per-mode",
        type=int,
        default=24,
        help="Total candidate count per smooth mode (including fixed controls).",
    )
    parser.add_argument(
        "--fixed-controls-per-mode",
        type=int,
        default=4,
        help="Number of deterministic control candidates per smooth mode.",
    )
    parser.add_argument("--seed", type=int, default=20260212, help="Random seed for candidate sampling.")
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Comma-separated candidate names to run; empty means all generated candidates.",
    )
    parser.add_argument(
        "--smooth-modes",
        type=str,
        default="",
        help=(
            "Comma-separated subset of smooth modes to scan (default: all). "
            f"Valid: {', '.join(VALID_FG_SMOOTH_MODES)}"
        ),
    )
    parser.add_argument(
        "--prior-sources",
        type=str,
        default="",
        help="Comma-separated subset of FG smooth prior sources to scan (default: all).",
    )
    parser.add_argument(
        "--explicit-fg-mean-list",
        type=str,
        default="0.0",
        help=(
            "Comma-separated fg_smooth_mean values used when prior_source=explicit_scalar. "
            "Default: 0.0 (keeps legacy behavior)."
        ),
    )
    parser.add_argument(
        "--explicit-fg-sigma-list",
        type=str,
        default="",
        help=(
            "Comma-separated fg_smooth_sigma values used when prior_source=explicit_scalar. "
            "When empty, mode-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="cube1,cube2",
        help="Comma-separated datasets. Available: cube1,cube2.",
    )
    parser.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset->GPU mapping, e.g. cube1:0,cube2:1",
    )
    parser.add_argument("--max-concurrent-jobs", type=int, default=2, help="Max concurrent dataset jobs per candidate.")
    parser.add_argument("--num-iters", type=int, default=1200, help="Iterations per run.")
    parser.add_argument("--print-every", type=int, default=200, help="Iteration logging interval.")
    parser.add_argument("--cut-size-frac", type=float, default=0.30, help="Spatial center cut fraction.")
    parser.add_argument("--eor-amp-threshold", type=float, default=0.1, help="Fixed EoR dead-zone threshold.")
    parser.add_argument("--data-error", type=float, default=0.005, help="Data error scalar prior.")
    parser.add_argument("--freq-start-mhz", type=float, default=106.0, help="Starting frequency in MHz.")
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1, help="Channel spacing in MHz.")
    parser.add_argument(
        "--conda-env",
        type=str,
        default="",
        help="Conda env name for separation_cli.py. Empty uses --python-bin.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run separation_cli.py when --conda-env is empty.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs/candidates but do not execute jobs.")
    return parser.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def _validate_subset(name: str, values: Sequence[str], valid: Sequence[str]) -> List[str]:
    valid_set = set(str(v) for v in valid)
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    unknown = sorted({v for v in cleaned if v not in valid_set})
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}; valid={sorted(valid_set)}")
    return cleaned


def parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid gpu map token: {token}")
        key, value = token.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


def _sanitize_token(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value)


def _fmt_float_token(value: float, digits: int = 4) -> str:
    text = f"{value:.{digits}g}"
    return _sanitize_token(text.replace(".", "p").replace("+", "").replace("-", "m"))


def build_datasets(data_dir: Path) -> List[DatasetSpec]:
    datasets = [
        DatasetSpec(
            name="cube1",
            input_cube=data_dir / "back" / "all_cube1.fits",
            fg_true_cube=data_dir / "fg_cube1.fits",
            eor_true_cube=data_dir / "eor_cube1.fits",
        ),
        DatasetSpec(
            name="cube2",
            input_cube=data_dir / "all_cube2.fits",
            fg_true_cube=data_dir / "fg_cube2.fits",
            eor_true_cube=data_dir / "eor_cube2.fits",
        ),
    ]
    for ds in datasets:
        for p in (ds.input_cube, ds.fg_true_cube, ds.eor_true_cube):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")
    return datasets


def _explicit_sigma_values(mode: str) -> Sequence[float]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "diff3_l2":
        return (2e-4, 5e-4, 1e-3)
    if mode_norm in {"diff2_l2", "diff2_huber"}:
        return (8e-4, 2e-3, 5e-3)
    if mode_norm == "diff1_l1":
        return (8e-3, 2e-2, 5e-2)
    raise ValueError(f"Unsupported smooth mode: {mode}")


def _phase_default_samples(phase: str) -> int:
    phase_norm = str(phase).strip().lower()
    if phase_norm == "phase_a":
        return 24
    if phase_norm == "phase_b":
        return 12
    if phase_norm == "phase_c":
        return 6
    raise ValueError(f"Unsupported phase: {phase}")


def generate_candidates(
    *,
    phase: str,
    samples_per_mode: int,
    fixed_controls_per_mode: int,
    seed: int,
    smooth_modes: Optional[Sequence[str]] = None,
    prior_sources: Optional[Sequence[str]] = None,
    explicit_fg_smooth_mean_list: Optional[Sequence[float]] = None,
    explicit_fg_smooth_sigma_list: Optional[Sequence[float]] = None,
) -> List[CandidateSpec]:
    mode_list = list(VALID_FG_SMOOTH_MODES)
    if smooth_modes:
        mode_list = _validate_subset("smooth_modes", smooth_modes, VALID_FG_SMOOTH_MODES)
    rng = random.Random(int(seed))
    if samples_per_mode <= 0:
        raise ValueError("samples_per_mode must be > 0.")
    if fixed_controls_per_mode < 0 or fixed_controls_per_mode > samples_per_mode:
        raise ValueError("fixed_controls_per_mode must be in [0, samples_per_mode].")

    beta_list = (0.2, 0.5, 1.0, 2.0, 4.0)
    gamma_list = (0.05, 0.15, 0.3, 0.6)
    eor_sigma_list = (0.02, 0.03, 0.05)
    prior_source_list = ("reference_robust", "reference_std", "explicit_scalar")
    if prior_sources:
        prior_source_list = tuple(_validate_subset("prior_sources", prior_sources, prior_source_list))
    mae_factor_list = (1.0, 1.4826, 2.0)
    optimizer_list = ("adam", "sgd")
    adam_lr_list = (1.0e-4, 2.0e-4, 4.0e-4)
    sgd_lr_list = (5.0e-4, 1.0e-3, 2.0e-3)
    sgd_momentum_list = (0.85, 0.90, 0.95)
    scheduler_list = ("plateau", "plateau", "none")
    plateau_patience_list = (120, 240, 360)
    plateau_factor_list = (0.5, 0.3)
    plateau_min_delta_list = (1e-4, 5e-5)
    plateau_cooldown_list = (40, 80)
    plateau_min_lr_list = (1e-6, 5e-6)
    huber_delta_list = (0.7, 1.2, 2.0, 3.0)

    explicit_mean_list: Sequence[float] = (0.0,)
    if explicit_fg_smooth_mean_list is not None:
        if len(explicit_fg_smooth_mean_list) == 0:
            raise ValueError("explicit_fg_smooth_mean_list must not be empty.")
        explicit_mean_list = tuple(float(v) for v in explicit_fg_smooth_mean_list)
    # If provided, this overrides the mode-specific sigma presets for explicit_scalar priors.
    explicit_sigma_override: Optional[Sequence[float]] = None
    if explicit_fg_smooth_sigma_list is not None:
        if len(explicit_fg_smooth_sigma_list) == 0:
            raise ValueError("explicit_fg_smooth_sigma_list must not be empty.")
        explicit_sigma_override = tuple(float(v) for v in explicit_fg_smooth_sigma_list)

    out: List[CandidateSpec] = []
    seen = set()

    for mode in mode_list:
        controls: List[CandidateSpec] = []
        if fixed_controls_per_mode > 0:
            controls = [
                CandidateSpec(
                    name="",
                    smooth_mode=mode,
                    beta=1.0,
                    gamma=0.3,
                    eor_prior_sigma=0.03,
                    prior_source="reference_robust",
                    use_robust_fg_stats=True,
                    mae_to_sigma_factor=1.4826,
                    fg_smooth_mean=None,
                    fg_smooth_sigma=None,
                    fg_smooth_huber_delta=1.2 if mode == "diff2_huber" else 1.0,
                    optimizer_name="adam",
                    lr=2.0e-4,
                    momentum=0.9,
                    lr_scheduler="none",
                    lr_plateau_patience=240,
                    lr_plateau_factor=0.5,
                    lr_plateau_min_delta=1e-4,
                    lr_plateau_cooldown=80,
                    lr_min=1e-6,
                ),
                CandidateSpec(
                    name="",
                    smooth_mode=mode,
                    beta=1.0,
                    gamma=0.3,
                    eor_prior_sigma=0.03,
                    prior_source="reference_robust",
                    use_robust_fg_stats=True,
                    mae_to_sigma_factor=1.4826,
                    fg_smooth_mean=None,
                    fg_smooth_sigma=None,
                    fg_smooth_huber_delta=1.2 if mode == "diff2_huber" else 1.0,
                    optimizer_name="adam",
                    lr=2.0e-4,
                    momentum=0.9,
                    lr_scheduler="plateau",
                    lr_plateau_patience=240,
                    lr_plateau_factor=0.5,
                    lr_plateau_min_delta=1e-4,
                    lr_plateau_cooldown=80,
                    lr_min=1e-6,
                ),
                CandidateSpec(
                    name="",
                    smooth_mode=mode,
                    beta=0.5,
                    gamma=0.3,
                    eor_prior_sigma=0.03,
                    prior_source="reference_std",
                    use_robust_fg_stats=False,
                    mae_to_sigma_factor=1.4826,
                    fg_smooth_mean=None,
                    fg_smooth_sigma=None,
                    fg_smooth_huber_delta=1.2 if mode == "diff2_huber" else 1.0,
                    optimizer_name="sgd",
                    lr=1.0e-3,
                    momentum=0.9,
                    lr_scheduler="none",
                    lr_plateau_patience=240,
                    lr_plateau_factor=0.5,
                    lr_plateau_min_delta=1e-4,
                    lr_plateau_cooldown=80,
                    lr_min=1e-6,
                ),
                CandidateSpec(
                    name="",
                    smooth_mode=mode,
                    beta=0.5,
                    gamma=0.3,
                    eor_prior_sigma=0.03,
                    prior_source="reference_std",
                    use_robust_fg_stats=False,
                    mae_to_sigma_factor=1.4826,
                    fg_smooth_mean=None,
                    fg_smooth_sigma=None,
                    fg_smooth_huber_delta=1.2 if mode == "diff2_huber" else 1.0,
                    optimizer_name="sgd",
                    lr=1.0e-3,
                    momentum=0.9,
                    lr_scheduler="plateau",
                    lr_plateau_patience=240,
                    lr_plateau_factor=0.5,
                    lr_plateau_min_delta=1e-4,
                    lr_plateau_cooldown=80,
                    lr_min=1e-6,
                ),
            ][:fixed_controls_per_mode]

        per_mode: List[CandidateSpec] = []
        for cand in controls:
            if cand.key() not in seen:
                seen.add(cand.key())
                per_mode.append(cand)

        target_n = samples_per_mode
        attempts = 0
        max_attempts = max(2000, samples_per_mode * 200)
        while len(per_mode) < target_n and attempts < max_attempts:
            attempts += 1
            prior_source = rng.choice(prior_source_list)
            use_robust = prior_source == "reference_robust"
            mae_factor = rng.choice(mae_factor_list) if use_robust else 1.4826
            fg_mean = None
            fg_sigma = None
            if prior_source == "explicit_scalar":
                fg_mean = float(explicit_mean_list[0]) if len(explicit_mean_list) == 1 else float(rng.choice(explicit_mean_list))
                sigma_list = explicit_sigma_override if explicit_sigma_override is not None else _explicit_sigma_values(mode)
                fg_sigma = float(rng.choice(sigma_list))

            optimizer_name = rng.choice(optimizer_list)
            if optimizer_name == "adam":
                lr_val = float(rng.choice(adam_lr_list))
                momentum = 0.9
            else:
                lr_val = float(rng.choice(sgd_lr_list))
                momentum = float(rng.choice(sgd_momentum_list))

            scheduler = rng.choice(scheduler_list)
            if scheduler == "plateau":
                plateau_patience = int(rng.choice(plateau_patience_list))
                plateau_factor = float(rng.choice(plateau_factor_list))
                plateau_min_delta = float(rng.choice(plateau_min_delta_list))
                plateau_cooldown = int(rng.choice(plateau_cooldown_list))
                plateau_min_lr = float(rng.choice(plateau_min_lr_list))
            else:
                plateau_patience = 240
                plateau_factor = 0.5
                plateau_min_delta = 1e-4
                plateau_cooldown = 80
                plateau_min_lr = 1e-6

            cand = CandidateSpec(
                name="",
                smooth_mode=mode,
                beta=float(rng.choice(beta_list)),
                gamma=float(rng.choice(gamma_list)),
                eor_prior_sigma=float(rng.choice(eor_sigma_list)),
                prior_source=prior_source,
                use_robust_fg_stats=bool(use_robust),
                mae_to_sigma_factor=float(mae_factor),
                fg_smooth_mean=fg_mean,
                fg_smooth_sigma=fg_sigma,
                fg_smooth_huber_delta=float(rng.choice(huber_delta_list) if mode == "diff2_huber" else 1.0),
                optimizer_name=optimizer_name,
                lr=lr_val,
                momentum=momentum,
                lr_scheduler=scheduler,
                lr_plateau_patience=plateau_patience,
                lr_plateau_factor=plateau_factor,
                lr_plateau_min_delta=plateau_min_delta,
                lr_plateau_cooldown=plateau_cooldown,
                lr_min=plateau_min_lr,
            )
            key = cand.key()
            if key in seen:
                continue
            seen.add(key)
            per_mode.append(cand)

        if len(per_mode) < target_n:
            raise RuntimeError(
                f"Unable to generate enough unique candidates for mode={mode}: "
                f"{len(per_mode)} < {target_n}."
            )

        for idx, cand in enumerate(per_mode, start=1):
            name = (
                f"{mode}_n{idx:03d}_b{_fmt_float_token(cand.beta)}_g{_fmt_float_token(cand.gamma)}"
                f"_es{_fmt_float_token(cand.eor_prior_sigma)}_"
                f"{_sanitize_token(cand.prior_source)}_{cand.optimizer_name}_lr{_fmt_float_token(cand.lr)}"
                f"_sch{cand.lr_scheduler}"
            )
            if cand.prior_source == "explicit_scalar" and cand.fg_smooth_mean is not None and abs(float(cand.fg_smooth_mean)) > 0:
                name += f"_fm{_fmt_float_token(float(cand.fg_smooth_mean))}"
            if cand.fg_smooth_sigma is not None:
                name += f"_fs{_fmt_float_token(cand.fg_smooth_sigma)}"
            if cand.smooth_mode == "diff2_huber":
                name += f"_hd{_fmt_float_token(cand.fg_smooth_huber_delta)}"
            out.append(
                CandidateSpec(
                    name=name,
                    smooth_mode=cand.smooth_mode,
                    beta=cand.beta,
                    gamma=cand.gamma,
                    eor_prior_sigma=cand.eor_prior_sigma,
                    prior_source=cand.prior_source,
                    use_robust_fg_stats=cand.use_robust_fg_stats,
                    mae_to_sigma_factor=cand.mae_to_sigma_factor,
                    fg_smooth_mean=cand.fg_smooth_mean,
                    fg_smooth_sigma=cand.fg_smooth_sigma,
                    fg_smooth_huber_delta=cand.fg_smooth_huber_delta,
                    optimizer_name=cand.optimizer_name,
                    lr=cand.lr,
                    momentum=cand.momentum,
                    lr_scheduler=cand.lr_scheduler,
                    lr_plateau_patience=cand.lr_plateau_patience,
                    lr_plateau_factor=cand.lr_plateau_factor,
                    lr_plateau_min_delta=cand.lr_plateau_min_delta,
                    lr_plateau_cooldown=cand.lr_plateau_cooldown,
                    lr_min=cand.lr_min,
                )
            )
    return out


def _extract_cut_indices(shape: Sequence[int], cut_size_frac: float) -> Optional[Tuple[int, int, int, int]]:
    from separation_optim import OptimizationConfig, build_cut_xy_indices

    cfg = OptimizationConfig()
    cfg.freq_axis = 0
    cfg.cut_xy_enabled = True
    cfg.cut_xy_unit = "frac"
    cfg.cut_xy_center_x = 0.5
    cfg.cut_xy_center_y = 0.5
    cfg.cut_xy_size = float(cut_size_frac)
    indices = build_cut_xy_indices(shape=shape, freq_axis=0, config=cfg)
    if indices is None:
        return None
    return indices.x0, indices.x1, indices.y0, indices.y1


def _load_cube_cut(path: Path, cut: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if cut is not None:
            x0, x1, y0, y1 = cut
            data = data[:, y0:y1, x0:x1]
        return np.asarray(data, dtype=np.float32)


def _center_crop_xy(cube: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={cube.shape}")
    f, ny, nx = cube.shape
    tf, ty, tx = map(int, target_shape)
    if f != tf:
        raise ValueError(f"Frequency dimension mismatch: {cube.shape} vs {tuple(target_shape)}")
    if ny < ty or nx < tx:
        raise ValueError(f"Cannot center-crop from {cube.shape} to {tuple(target_shape)}")
    y0 = (ny - ty) // 2
    x0 = (nx - tx) // 2
    return cube[:, y0 : y0 + ty, x0 : x0 + tx]


def _frequency_correlations(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    if est.shape != true.shape:
        raise ValueError(f"Shape mismatch for correlation: {est.shape} vs {true.shape}")
    out = np.full((est.shape[0],), np.nan, dtype=np.float64)
    for i in range(est.shape[0]):
        a = est[i].reshape(-1).astype(np.float64, copy=False)
        b = true[i].reshape(-1).astype(np.float64, copy=False)
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            out[i] = float(np.dot(a, b) / denom)
    return out


def _write_frequency_corr_profile(path: Path, corr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_idx", "corr"])
        for idx, value in enumerate(corr):
            if np.isfinite(value):
                writer.writerow([int(idx), f"{float(value):.10g}"])
            else:
                writer.writerow([int(idx), "nan"])


def _score_from_corr(corr: np.ndarray) -> float:
    vals = corr[np.isfinite(corr)]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    return 0.7 * mean + 0.2 * median + 0.1 * p10


def _build_config(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(candidate.lr),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": [],
            "optimizer_name": candidate.optimizer_name,
            "momentum": float(candidate.momentum),
            "lr_scheduler": candidate.lr_scheduler,
            "lr_plateau_patience": int(candidate.lr_plateau_patience),
            "lr_plateau_factor": float(candidate.lr_plateau_factor),
            "lr_plateau_min_delta": float(candidate.lr_plateau_min_delta),
            "lr_plateau_cooldown": int(candidate.lr_plateau_cooldown),
            "lr_min": float(candidate.lr_min),
            "freq_start_mhz": float(args.freq_start_mhz),
            "freq_delta_mhz": float(args.freq_delta_mhz),
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
            "beta": float(candidate.beta),
            "gamma": float(candidate.gamma),
            "corr_weight": 0.0,
            "lagcorr_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(candidate.eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.eor_amp_threshold),
            "fg_smooth_mode": candidate.smooth_mode,
            "fg_smooth_huber_delta": float(candidate.fg_smooth_huber_delta),
            "corr_prior_mean": 0.0,
            "corr_prior_sigma": 1.0,
        },
        "evaluation": {
            "true_eor_cube": str(dataset.eor_true_cube),
            "diagnose_input": False,
            "enable_corr_check": True,
            "corr_check_every": max(50, int(args.print_every)),
            "corr_plot": str(run_dir / "eor_corr.png"),
        },
        "init": {
            "init_fg_cube": "",
            "init_eor_cube": "",
        },
        "scan_meta": {
            "candidate_name": candidate.name,
            "prior_source": candidate.prior_source,
        },
    }

    priors = cfg["priors"]
    assert isinstance(priors, dict)
    if candidate.prior_source == "explicit_scalar":
        priors["fg_smooth_mean"] = float(candidate.fg_smooth_mean) if candidate.fg_smooth_mean is not None else 0.0
        priors["fg_smooth_sigma"] = float(candidate.fg_smooth_sigma if candidate.fg_smooth_sigma is not None else 1.0)
    else:
        priors["fg_reference_cube"] = str(dataset.fg_true_cube)
        priors["use_robust_fg_stats"] = bool(candidate.use_robust_fg_stats)
        priors["mae_to_sigma_factor"] = float(candidate.mae_to_sigma_factor)
    return cfg


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    if args.conda_env.strip():
        return ["conda", "run", "-n", args.conda_env.strip(), "python", str(cli_path), "--config", str(config_path)]
    return [args.python_bin, str(cli_path), "--config", str(config_path)]


def _run_job(args: argparse.Namespace, code_dir: Path, job: JobSpec) -> Dict[str, object]:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    with job.config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _build_config(
                dataset=job.dataset,
                candidate=job.candidate,
                run_dir=job.run_dir,
                gpu_index=job.gpu_index,
                args=args,
            ),
            handle,
            indent=2,
        )

    cmd = _command_for_config(args, code_dir, job.config_path)
    t0 = time.time()
    with job.log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    runtime = time.time() - t0

    row: Dict[str, object] = {
        "candidate": job.candidate.name,
        "dataset": job.dataset.name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": int(proc.returncode),
        "runtime_sec": float(runtime),
        "smooth_mode": job.candidate.smooth_mode,
        "beta": float(job.candidate.beta),
        "gamma": float(job.candidate.gamma),
        "eor_prior_sigma": float(job.candidate.eor_prior_sigma),
        "eor_amp_threshold": float(args.eor_amp_threshold),
        "prior_source": job.candidate.prior_source,
        "use_robust_fg_stats": bool(job.candidate.use_robust_fg_stats),
        "mae_to_sigma_factor": float(job.candidate.mae_to_sigma_factor),
        "fg_smooth_mean": (
            float(job.candidate.fg_smooth_mean) if job.candidate.fg_smooth_mean is not None else None
        ),
        "fg_smooth_sigma": (
            float(job.candidate.fg_smooth_sigma) if job.candidate.fg_smooth_sigma is not None else None
        ),
        "fg_smooth_huber_delta": float(job.candidate.fg_smooth_huber_delta),
        "optimizer_name": job.candidate.optimizer_name,
        "lr": float(job.candidate.lr),
        "momentum": float(job.candidate.momentum),
        "lr_scheduler": job.candidate.lr_scheduler,
        "lr_plateau_patience": int(job.candidate.lr_plateau_patience),
        "lr_plateau_factor": float(job.candidate.lr_plateau_factor),
        "lr_plateau_min_delta": float(job.candidate.lr_plateau_min_delta),
        "lr_plateau_cooldown": int(job.candidate.lr_plateau_cooldown),
        "lr_min": float(job.candidate.lr_min),
        "config_path": str(job.config_path),
        "log_path": str(job.log_path),
        "eor_output": str(job.eor_output),
    }

    if proc.returncode != 0:
        return row

    if not job.eor_output.exists():
        row["status"] = "failed"
        row["note"] = f"Missing EoR output: {job.eor_output}"
        return row

    with fits.open(job.dataset.input_cube, memmap=True) as hdul:
        in_shape = tuple(int(v) for v in hdul[0].data.shape)
    cut = _extract_cut_indices(in_shape, args.cut_size_frac)
    eor_true = _load_cube_cut(job.dataset.eor_true_cube, cut=cut)
    with fits.open(job.eor_output, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    if eor_true.shape != eor_est.shape:
        eor_true = _center_crop_xy(eor_true, eor_est.shape)

    corr = _frequency_correlations(eor_est, eor_true)
    corr_profile_path = run_dir / "eor_corr_profile.csv"
    _write_frequency_corr_profile(corr_profile_path, corr)
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
    row["corr_score"] = float(_score_from_corr(corr))
    return row


def _candidate_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate")), []).append(row)

    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        ok_items = [r for r in items if str(r.get("status")) == "ok"]
        score_vals = [float(r["corr_score"]) for r in ok_items if r.get("corr_score") is not None]
        mean_vals = [float(r["eor_corr_mean"]) for r in ok_items if r.get("eor_corr_mean") is not None]
        med_vals = [float(r["eor_corr_median"]) for r in ok_items if r.get("eor_corr_median") is not None]
        p10_vals = [float(r["eor_corr_p10"]) for r in ok_items if r.get("eor_corr_p10") is not None]
        summary: Dict[str, object] = {
            "candidate": cand,
            "n": len(items),
            "n_ok": len(ok_items),
            "corr_score_mean": float(np.mean(score_vals)) if score_vals else float("nan"),
            "eor_corr_mean_mean": float(np.mean(mean_vals)) if mean_vals else float("nan"),
            "eor_corr_median_mean": float(np.mean(med_vals)) if med_vals else float("nan"),
            "eor_corr_p10_mean": float(np.mean(p10_vals)) if p10_vals else float("nan"),
        }
        out.append(summary)
    out.sort(key=lambda x: float(x.get("corr_score_mean", float("-inf"))), reverse=True)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys: List[str] = []
    key_set = set()
    for row in rows:
        for key in row.keys():
            if key not in key_set:
                key_set.add(key)
                all_keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Base Smooth Scan Summary\n\n")
    lines.append("| rank | candidate | n_ok/n | corr_score_mean | eor_corr_mean | eor_corr_median | eor_corr_p10 |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['candidate']} | {row['n_ok']}/{row['n']} | "
            f"{float(row['corr_score_mean']):.6f} | "
            f"{float(row['eor_corr_mean_mean']):.6f} | "
            f"{float(row['eor_corr_median_mean']):.6f} | "
            f"{float(row['eor_corr_p10_mean']):.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"base_smooth_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase and args.samples_per_mode == 24:
        args.samples_per_mode = _phase_default_samples(args.phase)

    datasets_all = build_datasets(data_dir)
    enabled = {x.strip() for x in args.datasets.split(",") if x.strip()}
    datasets = [d for d in datasets_all if d.name in enabled]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")

    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

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
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        known = {c.name for c in candidates}
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(d).items()} for d in datasets
        ],
        "candidates": [asdict(c) for c in candidates],
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
                    dataset=ds,
                    candidate=cand,
                    gpu_index=int(gpu_map[ds.name]),
                    run_dir=run_dir,
                    config_path=run_dir / "config.json",
                    log_path=run_dir / "run.log",
                    fg_output=run_dir / "fg_est.fits",
                    eor_output=run_dir / "eor_est.fits",
                )
            )

        active: List[Tuple[subprocess.Popen, JobSpec, float, object]] = []
        queued = list(jobs)
        while queued or active:
            while queued and len(active) < max_jobs:
                job = queued.pop(0)
                job.run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_config(dataset=job.dataset, candidate=job.candidate, run_dir=job.run_dir, gpu_index=job.gpu_index, args=args)
                with job.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(cfg, handle, indent=2)
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active.append((proc, job, time.time(), log_handle))
                print(f"  [launch] {job.dataset.name} gpu={job.gpu_index} pid={proc.pid}")

            still_active: List[Tuple[subprocess.Popen, JobSpec, float, object]] = []
            for proc, job, t0, log_handle in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, t0, log_handle))
                    continue
                log_handle.close()
                runtime = time.time() - t0
                row = _run_job_result_only(args=args, dataset=job.dataset, candidate=job.candidate, run_dir=job.run_dir, return_code=ret, runtime=runtime)
                rows.append(row)
                print(f"  [done] {job.dataset.name} status={row['status']} score={row.get('corr_score')}")
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "base_smooth_scan_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows)
    summary_csv = output_dir / "base_smooth_scan_rank.csv"
    _write_csv(summary_csv, ranked)
    _write_markdown(output_dir / "base_smooth_scan_summary.md", ranked)
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={summary_csv}")
    return 0


def _run_job_result_only(
    *,
    args: argparse.Namespace,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    return_code: int,
    runtime: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "candidate": candidate.name,
        "dataset": dataset.name,
        "status": "ok" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "runtime_sec": float(runtime),
        "smooth_mode": candidate.smooth_mode,
        "beta": float(candidate.beta),
        "gamma": float(candidate.gamma),
        "eor_prior_sigma": float(candidate.eor_prior_sigma),
        "eor_amp_threshold": float(args.eor_amp_threshold),
        "prior_source": candidate.prior_source,
        "use_robust_fg_stats": bool(candidate.use_robust_fg_stats),
        "mae_to_sigma_factor": float(candidate.mae_to_sigma_factor),
        "fg_smooth_mean": float(candidate.fg_smooth_mean) if candidate.fg_smooth_mean is not None else None,
        "fg_smooth_sigma": float(candidate.fg_smooth_sigma) if candidate.fg_smooth_sigma is not None else None,
        "fg_smooth_huber_delta": float(candidate.fg_smooth_huber_delta),
        "optimizer_name": candidate.optimizer_name,
        "lr": float(candidate.lr),
        "momentum": float(candidate.momentum),
        "lr_scheduler": candidate.lr_scheduler,
        "lr_plateau_patience": int(candidate.lr_plateau_patience),
        "lr_plateau_factor": float(candidate.lr_plateau_factor),
        "lr_plateau_min_delta": float(candidate.lr_plateau_min_delta),
        "lr_plateau_cooldown": int(candidate.lr_plateau_cooldown),
        "lr_min": float(candidate.lr_min),
        "config_path": str(run_dir / "config.json"),
        "log_path": str(run_dir / "run.log"),
        "eor_output": str(run_dir / "eor_est.fits"),
    }
    if int(return_code) != 0:
        return row

    eor_out = run_dir / "eor_est.fits"
    if not eor_out.exists():
        row["status"] = "failed"
        row["note"] = "missing_eor_output"
        return row

    with fits.open(dataset.input_cube, memmap=True) as hdul:
        in_shape = tuple(int(v) for v in hdul[0].data.shape)
    cut = _extract_cut_indices(in_shape, args.cut_size_frac)
    eor_true = _load_cube_cut(dataset.eor_true_cube, cut=cut)
    with fits.open(eor_out, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    if eor_true.shape != eor_est.shape:
        eor_true = _center_crop_xy(eor_true, eor_est.shape)

    corr = _frequency_correlations(eor_est, eor_true)
    finite = corr[np.isfinite(corr)]
    row["eor_corr_count"] = int(finite.size)
    if finite.size == 0:
        row["status"] = "failed"
        row["note"] = "no_finite_corr"
        return row
    row["eor_corr_mean"] = float(np.mean(finite))
    row["eor_corr_median"] = float(np.median(finite))
    row["eor_corr_p10"] = float(np.percentile(finite, 10))
    row["eor_corr_min"] = float(np.min(finite))
    row["eor_corr_max"] = float(np.max(finite))
    row["corr_score"] = float(_score_from_corr(corr))
    return row


if __name__ == "__main__":
    raise SystemExit(main())
