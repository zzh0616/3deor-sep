#!/usr/bin/env python3
"""
Numerical experiment for EoR lagcorr prior design under physics-stable assumptions.

Design principles:
- Allow EoR prior terms only from robust physical facts:
  1) EoR brightness temperature scale is small (~10 mK level);
  2) EoR frequency correlation decays rapidly with lag;
  3) EoR is less spectrally smooth than foregrounds.
- Avoid priors tied to specific cosmological/astrophysical scenario details.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


@dataclass
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path


@dataclass
class CandidateSpec:
    name: str
    loss_mode: str
    lagcorr_feature: str
    lagcorr_fg_component_weight: float
    lagcorr_eor_component_weight: float
    fg_lagcorr_sigma_scale: float
    gamma: float
    eor_prior_sigma: float
    eor_lagcorr_mean: List[float]
    eor_lagcorr_sigma: List[float]
    note: str


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


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EoR lagcorr prior design experiment on two cubes."
    )
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="3dnet dir (default <work-root>/3dnet).")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/lagcorr_eor_prior_design_<timestamp>).",
    )
    parser.add_argument("--num-iters", type=int, default=500, help="Iterations per run.")
    parser.add_argument("--print-every", type=int, default=50, help="Logging interval.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--cut-size-frac", type=float, default=0.30, help="Spatial center cut fraction.")
    parser.add_argument("--extra-loss-start-iter", type=int, default=80, help="Extra-loss start iter.")
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=120, help="Extra-loss ramp iters.")
    parser.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset->GPU mapping, e.g. cube1:0,cube2:1",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="torch",
        help="Conda env for separation_cli.py (empty uses current python).",
    )
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Comma-separated candidate names to run. Empty runs all.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=2,
        help="Maximum concurrent dataset jobs per candidate.",
    )
    return parser.parse_args()


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
        for path in (ds.input_cube, ds.fg_true_cube, ds.eor_true_cube):
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")
    return datasets


def build_candidates() -> List[CandidateSpec]:
    zeros = [0.0] * len(LAG_INTERVALS_MHZ)

    # Physically stable EoR-lag priors in diff1 domain:
    # mean close to 0 (rapid decorrelation after differencing),
    # with different prior widths for loose/mid/strict tests.
    loose_sigma = [0.35, 0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.12, 0.12]
    mid_sigma = [0.20, 0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08]
    strict_sigma = [0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04]

    return [
        CandidateSpec(
            name="base_noeorprior",
            loss_mode="base",
            lagcorr_feature="raw",
            lagcorr_fg_component_weight=0.0,
            lagcorr_eor_component_weight=0.0,
            fg_lagcorr_sigma_scale=2.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            eor_lagcorr_mean=zeros,
            eor_lagcorr_sigma=loose_sigma,
            note="reference base mode",
        ),
        CandidateSpec(
            name="lag_diff1_fg_only",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            fg_lagcorr_sigma_scale=2.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            eor_lagcorr_mean=zeros,
            eor_lagcorr_sigma=loose_sigma,
            note="best previous fg-only lagcorr baseline",
        ),
        CandidateSpec(
            name="lag_diff1_eor_phys_loose",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            fg_lagcorr_sigma_scale=2.0,
            gamma=0.2,
            eor_prior_sigma=0.02,
            eor_lagcorr_mean=zeros,
            eor_lagcorr_sigma=loose_sigma,
            note="10-20mK scale, fast decorrelation, loose bounds",
        ),
        CandidateSpec(
            name="lag_diff1_eor_phys_mid",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.6,
            fg_lagcorr_sigma_scale=2.0,
            gamma=0.4,
            eor_prior_sigma=0.015,
            eor_lagcorr_mean=zeros,
            eor_lagcorr_sigma=mid_sigma,
            note="moderate EoR lag prior strength",
        ),
        CandidateSpec(
            name="lag_diff1_eor_phys_strict",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=1.0,
            fg_lagcorr_sigma_scale=2.0,
            gamma=0.6,
            eor_prior_sigma=0.01,
            eor_lagcorr_mean=zeros,
            eor_lagcorr_sigma=strict_sigma,
            note="~10mK tight amplitude prior + tighter lag prior",
        ),
    ]


def _center_cut(arr: np.ndarray, frac: float) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {arr.shape}")
    _, ny, nx = arr.shape
    size = int(round(min(ny, nx) * float(frac)))
    size = max(1, min(size, ny, nx))
    y0 = (ny - size) // 2
    x0 = (nx - size) // 2
    return arr[:, y0 : y0 + size, x0 : x0 + size]


def _compute_lagcorr_profile(
    cube: np.ndarray,
    lag_channels: Sequence[int],
    *,
    max_pairs: Optional[int],
) -> Tuple[List[float], List[float]]:
    nfreq = int(cube.shape[0])
    flat = cube.reshape(nfreq, -1).astype(np.float64, copy=False)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms = np.clip(norms, 1e-12, None)

    means: List[float] = []
    sigmas: List[float] = []
    for lag in lag_channels:
        lag_i = int(lag)
        if lag_i < 1 or lag_i >= nfreq:
            raise ValueError(f"Invalid lag {lag_i} for nfreq={nfreq}")
        total = nfreq - lag_i
        use_pairs = total if max_pairs is None else min(total, int(max_pairs))
        idx = np.arange(use_pairs, dtype=np.int64)
        jdx = idx + lag_i
        corr = np.sum(centered[idx] * centered[jdx], axis=1) / np.clip(norms[idx] * norms[jdx], 1e-12, None)
        means.append(float(np.mean(corr)))
        sigmas.append(float(np.std(corr)))
    return means, sigmas


def derive_fg_prior(
    ds: DatasetSpec,
    *,
    lagcorr_feature: str,
    cut_size_frac: float,
    lag_intervals_mhz: Sequence[float],
    freq_delta_mhz: float,
    max_pairs: Optional[int],
    cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]],
) -> Tuple[List[float], List[float]]:
    key = (ds.name, lagcorr_feature, float(cut_size_frac))
    if key in cache:
        return cache[key]

    with fits.open(ds.fg_true_cube, memmap=True) as hdul:
        fg = np.asarray(hdul[0].data, dtype=np.float32)
    fg = _center_cut(fg, frac=cut_size_frac)
    if lagcorr_feature == "diff1":
        fg = np.diff(fg, n=1, axis=0)

    lag_channels = [max(1, int(round(float(v) / float(freq_delta_mhz)))) for v in lag_intervals_mhz]
    means, sigmas = _compute_lagcorr_profile(fg, lag_channels=lag_channels, max_pairs=max_pairs)
    cache[key] = (means, sigmas)
    return means, sigmas


def build_config(
    args: argparse.Namespace,
    ds: DatasetSpec,
    cand: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    fg_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]],
) -> Dict[str, object]:
    lag_intervals = list(LAG_INTERVALS_MHZ)

    fg_means, fg_sigmas_native = derive_fg_prior(
        ds,
        lagcorr_feature=cand.lagcorr_feature,
        cut_size_frac=float(args.cut_size_frac),
        lag_intervals_mhz=lag_intervals,
        freq_delta_mhz=0.1,
        max_pairs=256,
        cache=fg_cache,
    )
    fg_sigmas = [max(float(v) * float(cand.fg_lagcorr_sigma_scale), 1e-6) for v in fg_sigmas_native]

    lagcorr_weight = 1.0 if cand.loss_mode == "lagcorr" else 0.0

    cfg: Dict[str, object] = {
        "input_cube": str(ds.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "num_iters": int(args.num_iters),
        "lr": float(args.lr),
        "freq_start_mhz": 106.0,
        "freq_delta_mhz": 0.1,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": float(cand.gamma),
        "freq_axis": 0,
        "print_every": int(args.print_every),
        "device": f"cuda:{gpu_index}",
        "dtype": "float32",
        "loss_mode": cand.loss_mode,
        "extra_loss_start_iter": int(args.extra_loss_start_iter),
        "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
        "cut_xy": {
            "enabled": True,
            "unit": "frac",
            "center_x": 0.5,
            "center_y": 0.5,
            "size": float(args.cut_size_frac),
        },
        "weights": {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": float(cand.gamma),
            "corr_weight": 0.0,
            "lagcorr_weight": float(lagcorr_weight),
            "lagcorr_fg_component_weight": float(cand.lagcorr_fg_component_weight),
            "lagcorr_eor_component_weight": float(cand.lagcorr_eor_component_weight),
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": 0.005,
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(cand.eor_prior_sigma),
            "fg_smooth_mean": 0.0,
            "fg_smooth_sigma": 0.0005,
            "fg_reference_cube": str(ds.fg_true_cube),
            "use_robust_fg_stats": True,
            "mae_to_sigma_factor": 1.4826,
            "corr_prior_mean": 0.0,
            "corr_prior_sigma": 0.5,
            "lagcorr_feature": cand.lagcorr_feature,
            "lagcorr_unit": "mhz",
            "lagcorr_pair_sampling": "random",
            "lagcorr_random_seed": 20260211,
            "lagcorr_intervals": lag_intervals,
            "fg_lagcorr_mean": fg_means,
            "fg_lagcorr_sigma": fg_sigmas,
            "eor_lagcorr_mean": list(cand.eor_lagcorr_mean),
            "eor_lagcorr_sigma": list(cand.eor_lagcorr_sigma),
            "lagcorr_max_pairs": 256,
            "fft_highfreq_percent": 0.7,
            "fft_use_log_energy": True,
            "fft_z_clip": 6.0,
            "fft_prior_mean": 0.0,
            "fft_prior_sigma": 1.0,
            "poly_degree": 3,
            "poly_sigma": 0.01,
        },
        "evaluation": {
            "true_eor_cube": str(ds.eor_true_cube),
            "diagnose_input": False,
            "enable_corr_check": True,
            "corr_check_every": 100,
            "corr_plot": str(run_dir / "eor_corr.png"),
        },
        "optimizer": {
            "optimizer_name": "adam",
            "momentum": 0.9,
            "freq_start_mhz": 106.0,
            "freq_delta_mhz": 0.1,
        },
        "init": {
            "init_fg_cube": "",
            "init_eor_cube": "",
        },
    }
    return cfg


def _center_crop_to_shape(arr: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    if arr.shape[0] != shape[0]:
        raise ValueError(f"Frequency mismatch: {arr.shape} vs {shape}")
    _, ny, nx = arr.shape
    _, sy, sx = shape
    if sy > ny or sx > nx:
        raise ValueError(f"Cannot crop {arr.shape} to {shape}")
    y0 = (ny - sy) // 2
    x0 = (nx - sx) // 2
    return arr[:, y0 : y0 + sy, x0 : x0 + sx]


def _frequency_corr_mean(est: np.ndarray, true: np.ndarray) -> float:
    vals: List[float] = []
    for i in range(est.shape[0]):
        a = est[i].reshape(-1).astype(np.float64)
        b = true[i].reshape(-1).astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        vals.append(0.0 if den < 1e-18 else float(np.dot(a, b) / den))
    return float(np.mean(vals))


def _safe_profile_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size != b.size or a.size < 2:
        return None
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-18:
        return None
    return float(np.dot(a, b) / den)


def _apply_feature(cube: np.ndarray, feature: str) -> np.ndarray:
    if feature == "raw":
        return cube
    if feature == "diff1":
        if cube.shape[0] < 2:
            raise ValueError("diff1 requires at least 2 frequency channels.")
        return np.diff(cube, n=1, axis=0)
    raise ValueError(f"Unsupported feature: {feature}")


def _lag_profile_for_cube(
    cube: np.ndarray,
    *,
    lag_intervals_mhz: Sequence[float],
    freq_delta_mhz: float,
    feature: str,
    max_pairs: Optional[int] = None,
) -> List[float]:
    arr = _apply_feature(cube, feature)
    lag_channels = [max(1, int(round(float(v) / float(freq_delta_mhz)))) for v in lag_intervals_mhz]
    means, _ = _compute_lagcorr_profile(arr, lag_channels=lag_channels, max_pairs=max_pairs)
    return means


def _lag_profile_metrics(
    est_profile: Sequence[float],
    true_profile: Sequence[float],
    *,
    lag_intervals_mhz: Sequence[float],
    tail_threshold_mhz: float = 2.0,
) -> Dict[str, Optional[float]]:
    est = np.asarray(est_profile, dtype=np.float64)
    true = np.asarray(true_profile, dtype=np.float64)
    if est.size != true.size or est.size == 0:
        return {
            "mae": None,
            "rmse": None,
            "profile_corr": None,
            "decay_est": None,
            "decay_true": None,
            "decay_gap_abs": None,
            "tail_abs_est": None,
            "tail_abs_true": None,
            "tail_abs_gap": None,
        }

    diff = est - true
    tail_mask = np.asarray([float(v) >= float(tail_threshold_mhz) for v in lag_intervals_mhz], dtype=bool)
    if not np.any(tail_mask):
        tail_mask = np.ones_like(est, dtype=bool)
    tail_abs_est = float(np.mean(np.abs(est[tail_mask])))
    tail_abs_true = float(np.mean(np.abs(true[tail_mask])))
    decay_est = float(est[0] - est[-1])
    decay_true = float(true[0] - true[-1])
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "profile_corr": _safe_profile_corr(est, true),
        "decay_est": decay_est,
        "decay_true": decay_true,
        "decay_gap_abs": float(abs(decay_est - decay_true)),
        "tail_abs_est": tail_abs_est,
        "tail_abs_true": tail_abs_true,
        "tail_abs_gap": float(tail_abs_est - tail_abs_true),
    }


def _empty_metrics() -> Dict[str, Optional[float]]:
    return {
        "eor_mse": None,
        "eor_corr_mean": None,
        "eor_std_ratio": None,
        "recon_mse": None,
        "eor_lag_raw_mae": None,
        "eor_lag_raw_rmse": None,
        "eor_lag_raw_profile_corr": None,
        "eor_lag_raw_decay_est": None,
        "eor_lag_raw_decay_true": None,
        "eor_lag_raw_decay_gap_abs": None,
        "eor_lag_raw_tail_abs_est": None,
        "eor_lag_raw_tail_abs_true": None,
        "eor_lag_raw_tail_abs_gap": None,
        "eor_lag_diff1_mae": None,
        "eor_lag_diff1_rmse": None,
        "eor_lag_diff1_profile_corr": None,
        "eor_lag_diff1_decay_est": None,
        "eor_lag_diff1_decay_true": None,
        "eor_lag_diff1_decay_gap_abs": None,
        "eor_lag_diff1_tail_abs_est": None,
        "eor_lag_diff1_tail_abs_true": None,
        "eor_lag_diff1_tail_abs_gap": None,
    }


def compute_metrics(
    eor_est_path: Path,
    fg_est_path: Path,
    ds: DatasetSpec,
    *,
    run_dir: Optional[Path] = None,
    lag_intervals_mhz: Sequence[float] = LAG_INTERVALS_MHZ,
    freq_delta_mhz: float = 0.1,
) -> Dict[str, Optional[float]]:
    metrics = _empty_metrics()
    if not eor_est_path.exists() or not fg_est_path.exists():
        return metrics

    with fits.open(eor_est_path, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(fg_est_path, memmap=True) as hdul:
        fg_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.eor_true_cube, memmap=True) as hdul:
        eor_true_full = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.input_cube, memmap=True) as hdul:
        y_full = np.asarray(hdul[0].data, dtype=np.float32)

    eor_true = _center_crop_to_shape(eor_true_full, eor_est.shape)
    y_obs = _center_crop_to_shape(y_full, eor_est.shape)

    eor_mse = float(np.mean((eor_est - eor_true) ** 2))
    eor_corr_mean = _frequency_corr_mean(eor_est, eor_true)
    std_true = float(eor_true.std())
    eor_std_ratio = float(eor_est.std() / (std_true + 1e-12))
    recon_mse = float(np.mean((fg_est + eor_est - y_obs) ** 2))

    raw_est = _lag_profile_for_cube(
        eor_est,
        lag_intervals_mhz=lag_intervals_mhz,
        freq_delta_mhz=freq_delta_mhz,
        feature="raw",
    )
    raw_true = _lag_profile_for_cube(
        eor_true,
        lag_intervals_mhz=lag_intervals_mhz,
        freq_delta_mhz=freq_delta_mhz,
        feature="raw",
    )
    diff1_est = _lag_profile_for_cube(
        eor_est,
        lag_intervals_mhz=lag_intervals_mhz,
        freq_delta_mhz=freq_delta_mhz,
        feature="diff1",
    )
    diff1_true = _lag_profile_for_cube(
        eor_true,
        lag_intervals_mhz=lag_intervals_mhz,
        freq_delta_mhz=freq_delta_mhz,
        feature="diff1",
    )

    raw_metrics = _lag_profile_metrics(raw_est, raw_true, lag_intervals_mhz=lag_intervals_mhz)
    diff1_metrics = _lag_profile_metrics(diff1_est, diff1_true, lag_intervals_mhz=lag_intervals_mhz)

    metrics.update(
        {
        "eor_mse": eor_mse,
        "eor_corr_mean": eor_corr_mean,
        "eor_std_ratio": eor_std_ratio,
        "recon_mse": recon_mse,
        "eor_lag_raw_mae": raw_metrics["mae"],
        "eor_lag_raw_rmse": raw_metrics["rmse"],
        "eor_lag_raw_profile_corr": raw_metrics["profile_corr"],
        "eor_lag_raw_decay_est": raw_metrics["decay_est"],
        "eor_lag_raw_decay_true": raw_metrics["decay_true"],
        "eor_lag_raw_decay_gap_abs": raw_metrics["decay_gap_abs"],
        "eor_lag_raw_tail_abs_est": raw_metrics["tail_abs_est"],
        "eor_lag_raw_tail_abs_true": raw_metrics["tail_abs_true"],
        "eor_lag_raw_tail_abs_gap": raw_metrics["tail_abs_gap"],
        "eor_lag_diff1_mae": diff1_metrics["mae"],
        "eor_lag_diff1_rmse": diff1_metrics["rmse"],
        "eor_lag_diff1_profile_corr": diff1_metrics["profile_corr"],
        "eor_lag_diff1_decay_est": diff1_metrics["decay_est"],
        "eor_lag_diff1_decay_true": diff1_metrics["decay_true"],
        "eor_lag_diff1_decay_gap_abs": diff1_metrics["decay_gap_abs"],
        "eor_lag_diff1_tail_abs_est": diff1_metrics["tail_abs_est"],
        "eor_lag_diff1_tail_abs_true": diff1_metrics["tail_abs_true"],
        "eor_lag_diff1_tail_abs_gap": diff1_metrics["tail_abs_gap"],
        }
    )

    if run_dir is not None:
        payload = {
            "lag_intervals_mhz": [float(v) for v in lag_intervals_mhz],
            "raw": {"est": [float(v) for v in raw_est], "true": [float(v) for v in raw_true]},
            "diff1": {"est": [float(v) for v in diff1_est], "true": [float(v) for v in diff1_true]},
        }
        with (run_dir / "eor_lag_profile.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return metrics


def launch_job(code_dir: Path, conda_env: str, job: JobSpec) -> Tuple[subprocess.Popen[str], float]:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    if conda_env.strip():
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env.strip(),
            "python",
            str(code_dir / "separation_cli.py"),
            "--config",
            str(job.config_path),
        ]
    else:
        cmd = [sys.executable, str(code_dir / "separation_cli.py"), "--config", str(job.config_path)]

    handle = job.log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(code_dir.parent),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_log_handle", handle)
    return proc, time.time()


def close_job_process(proc: subprocess.Popen[str]) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        try:
            handle.flush()
        finally:
            handle.close()


def write_summary(output_dir: Path, rows: Sequence[Dict[str, object]]) -> Tuple[Path, Path]:
    csv_path = output_dir / "lagcorr_eor_prior_results.csv"
    md_path = output_dir / "lagcorr_eor_prior_results.md"
    fields = [
        "candidate",
        "dataset",
        "loss_mode",
        "lagcorr_feature",
        "gamma",
        "eor_prior_sigma",
        "lagcorr_fg_component_weight",
        "lagcorr_eor_component_weight",
        "status",
        "return_code",
        "runtime_sec",
        "eor_mse",
        "eor_corr_mean",
        "eor_std_ratio",
        "recon_mse",
        "eor_lag_raw_mae",
        "eor_lag_raw_rmse",
        "eor_lag_raw_profile_corr",
        "eor_lag_raw_decay_est",
        "eor_lag_raw_decay_true",
        "eor_lag_raw_decay_gap_abs",
        "eor_lag_raw_tail_abs_est",
        "eor_lag_raw_tail_abs_true",
        "eor_lag_raw_tail_abs_gap",
        "eor_lag_diff1_mae",
        "eor_lag_diff1_rmse",
        "eor_lag_diff1_profile_corr",
        "eor_lag_diff1_decay_est",
        "eor_lag_diff1_decay_true",
        "eor_lag_diff1_decay_gap_abs",
        "eor_lag_diff1_tail_abs_est",
        "eor_lag_diff1_tail_abs_true",
        "eor_lag_diff1_tail_abs_gap",
        "note",
        "config_path",
        "log_path",
        "eor_lag_profile_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    def _fmt(v: object) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)):
            x = float(v)
            if not math.isfinite(x):
                return "nan"
            if abs(x) >= 1e-3 and abs(x) < 1e3:
                return f"{x:.6f}"
            return f"{x:.6e}"
        return str(v)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# lagcorr EoR prior design experiment\n\n")
        handle.write(
            "| candidate | dataset | feature | gamma | eor_sigma | w_fg | w_eor | status | runtime_sec | eor_mse | eor_corr_mean | std_ratio | lag_raw_rmse | lag_raw_corr | lag_diff1_rmse | lag_diff1_corr |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row.get('candidate')} | {row.get('dataset')} | {row.get('lagcorr_feature')} | "
                f"{_fmt(row.get('gamma'))} | {_fmt(row.get('eor_prior_sigma'))} | "
                f"{_fmt(row.get('lagcorr_fg_component_weight'))} | {_fmt(row.get('lagcorr_eor_component_weight'))} | "
                f"{row.get('status')} | {_fmt(row.get('runtime_sec'))} | {_fmt(row.get('eor_mse'))} | "
                f"{_fmt(row.get('eor_corr_mean'))} | {_fmt(row.get('eor_std_ratio'))} | "
                f"{_fmt(row.get('eor_lag_raw_rmse'))} | {_fmt(row.get('eor_lag_raw_profile_corr'))} | "
                f"{_fmt(row.get('eor_lag_diff1_rmse'))} | {_fmt(row.get('eor_lag_diff1_profile_corr'))} |\n"
            )
    return csv_path, md_path


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = args.code_dir.resolve() if args.code_dir else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"lagcorr_eor_prior_design_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(data_dir)
    candidates = build_candidates()
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        known = {c.name for c in candidates}
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}. Known: {sorted(known)}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")
    gpu_map = parse_gpu_map(args.gpu_map)
    max_concurrent_jobs = max(1, int(args.max_concurrent_jobs))

    fg_prior_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]] = {}
    rows: List[Dict[str, object]] = []

    for cand in candidates:
        print(f"[candidate] {cand.name}")

        jobs: List[JobSpec] = []
        for ds in datasets:
            run_dir = output_dir / cand.name / ds.name
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.json"
            log_path = run_dir / "run.log"
            fg_output = run_dir / "fg_est.fits"
            eor_output = run_dir / "eor_est.fits"
            gpu_idx = gpu_map.get(ds.name, 0)

            cfg = build_config(args, ds, cand, run_dir, gpu_idx, fg_prior_cache)
            with cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle, indent=2)

            jobs.append(
                JobSpec(
                    dataset=ds,
                    candidate=cand,
                    gpu_index=gpu_idx,
                    run_dir=run_dir,
                    config_path=cfg_path,
                    log_path=log_path,
                    fg_output=fg_output,
                    eor_output=eor_output,
                )
            )

        queued: List[JobSpec] = list(jobs)
        procs: List[Tuple[JobSpec, subprocess.Popen[str], float]] = []
        while queued or procs:
            while queued and len(procs) < max_concurrent_jobs:
                job = queued.pop(0)
                print(f"  [launch] dataset={job.dataset.name} gpu={job.gpu_index} config={job.config_path}")
                proc, t0 = launch_job(code_dir=code_dir, conda_env=args.conda_env, job=job)
                procs.append((job, proc, t0))

            pending: List[Tuple[JobSpec, subprocess.Popen[str], float]] = []
            for job, proc, t0 in procs:
                ret = proc.poll()
                if ret is None:
                    pending.append((job, proc, t0))
                    continue

                runtime = time.time() - t0
                close_job_process(proc)
                status = "ok" if ret == 0 else "failed"
                metrics = (
                    compute_metrics(
                        job.eor_output,
                        job.fg_output,
                        job.dataset,
                        run_dir=job.run_dir,
                        lag_intervals_mhz=LAG_INTERVALS_MHZ,
                        freq_delta_mhz=0.1,
                    )
                    if ret == 0
                    else _empty_metrics()
                )

                row: Dict[str, object] = {
                    "candidate": job.candidate.name,
                    "dataset": job.dataset.name,
                    "loss_mode": job.candidate.loss_mode,
                    "lagcorr_feature": job.candidate.lagcorr_feature,
                    "gamma": job.candidate.gamma,
                    "eor_prior_sigma": job.candidate.eor_prior_sigma,
                    "lagcorr_fg_component_weight": job.candidate.lagcorr_fg_component_weight,
                    "lagcorr_eor_component_weight": job.candidate.lagcorr_eor_component_weight,
                    "status": status,
                    "return_code": ret,
                    "runtime_sec": runtime,
                    "note": job.candidate.note,
                    "config_path": str(job.config_path),
                    "log_path": str(job.log_path),
                    "eor_lag_profile_path": str(job.run_dir / "eor_lag_profile.json"),
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"  [done] dataset={job.dataset.name} status={status} "
                    f"runtime={runtime:.1f}s eor_mse={metrics.get('eor_mse')}"
                )
            procs = pending
            if queued or procs:
                time.sleep(5.0)

    csv_path, md_path = write_summary(output_dir, rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "work_root": str(work_root),
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "args": _json_safe(vars(args)),
        "candidates": _json_safe([asdict(c) for c in candidates]),
        "datasets": _json_safe([asdict(d) for d in datasets]),
        "result_csv": str(csv_path),
        "result_md": str(md_path),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[done] output_dir={output_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
