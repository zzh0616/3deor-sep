#!/usr/bin/env python3
"""
Ablation study for potential conflicts between corr prior and lagcorr_eor design.

Main questions:
1) Does enabling corr prior hurt lagcorr-based separation?
2) Is degradation mainly from corr prior or from stronger EoR amplitude prior (gamma/sigma)?
3) Should we remove corr from base when designing lagcorr variants?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


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
    gamma: float
    eor_prior_sigma: float
    corr_weight: float
    corr_prior_mode: str  # off | zero | truth
    eor_lag_sigma_profile: str  # loose | strict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corr/lagcorr conflict ablation.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="3dnet dir.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/lagcorr_conflict_ablation_<timestamp>).",
    )
    parser.add_argument("--num-iters", type=int, default=400, help="Iterations per run.")
    parser.add_argument("--print-every", type=int, default=50, help="Log interval.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--cut-size-frac", type=float, default=0.30, help="Center cut fraction.")
    parser.add_argument("--extra-loss-start-iter", type=int, default=80, help="Extra loss start iter.")
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=120, help="Extra loss ramp iters.")
    parser.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset->GPU map, e.g. cube1:0,cube2:1",
    )
    parser.add_argument("--conda-env", type=str, default="torch", help="Conda env for runs.")
    return parser.parse_args()


def parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid gpu map token: {token}")
        key, val = token.split(":", 1)
        out[key.strip()] = int(val.strip())
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
        for p in (ds.input_cube, ds.fg_true_cube, ds.eor_true_cube):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")
    return datasets


def build_candidates() -> List[CandidateSpec]:
    return [
        CandidateSpec(
            name="base_corr_off",
            loss_mode="base",
            lagcorr_feature="raw",
            lagcorr_fg_component_weight=0.0,
            lagcorr_eor_component_weight=0.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=0.0,
            corr_prior_mode="off",
            eor_lag_sigma_profile="loose",
            note="base without corr, no eor prior force",
        ),
        CandidateSpec(
            name="base_corr_zero",
            loss_mode="base",
            lagcorr_feature="raw",
            lagcorr_fg_component_weight=0.0,
            lagcorr_eor_component_weight=0.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=1.0,
            corr_prior_mode="zero",
            eor_lag_sigma_profile="loose",
            note="base with corr prior mean=0",
        ),
        CandidateSpec(
            name="lag_fgonly_corr_off",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=0.0,
            corr_prior_mode="off",
            eor_lag_sigma_profile="loose",
            note="fg-only lagcorr baseline",
        ),
        CandidateSpec(
            name="lag_fgonly_corr_zero",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=1.0,
            corr_prior_mode="zero",
            eor_lag_sigma_profile="loose",
            note="fg-only lagcorr plus corr mean=0",
        ),
        CandidateSpec(
            name="lag_eorweak_nogamma_corr_off",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=0.0,
            corr_prior_mode="off",
            eor_lag_sigma_profile="loose",
            note="add weak eor lag prior only (no gamma)",
        ),
        CandidateSpec(
            name="lag_eorweak_nogamma_corr_zero",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            gamma=0.0,
            eor_prior_sigma=0.02,
            corr_weight=1.0,
            corr_prior_mode="zero",
            eor_lag_sigma_profile="loose",
            note="add weak eor lag prior + corr mean=0",
        ),
        CandidateSpec(
            name="lag_eorloose_corr_off",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            gamma=0.2,
            eor_prior_sigma=0.02,
            corr_weight=0.0,
            corr_prior_mode="off",
            eor_lag_sigma_profile="loose",
            note="weak eor lag + weak eor amplitude prior",
        ),
        CandidateSpec(
            name="lag_eorloose_corr_zero",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            gamma=0.2,
            eor_prior_sigma=0.02,
            corr_weight=1.0,
            corr_prior_mode="zero",
            eor_lag_sigma_profile="loose",
            note="weak eor lag + gamma + corr mean=0",
        ),
        CandidateSpec(
            name="lag_eorloose_corr_truth",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.3,
            gamma=0.2,
            eor_prior_sigma=0.02,
            corr_weight=1.0,
            corr_prior_mode="truth",
            eor_lag_sigma_profile="loose",
            note="weak eor lag + gamma + corr mean=true-global",
        ),
        CandidateSpec(
            name="lag_eorstrict_corr_off",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=1.0,
            gamma=0.6,
            eor_prior_sigma=0.01,
            corr_weight=0.0,
            corr_prior_mode="off",
            eor_lag_sigma_profile="strict",
            note="strict eor lag + strict amplitude prior",
        ),
        CandidateSpec(
            name="lag_eorstrict_corr_zero",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=1.0,
            gamma=0.6,
            eor_prior_sigma=0.01,
            corr_weight=1.0,
            corr_prior_mode="zero",
            eor_lag_sigma_profile="strict",
            note="strict eor lag + strict amplitude + corr mean=0",
        ),
        CandidateSpec(
            name="lag_eorstrict_corr_truth",
            loss_mode="lagcorr",
            lagcorr_feature="diff1",
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=1.0,
            gamma=0.6,
            eor_prior_sigma=0.01,
            corr_weight=1.0,
            corr_prior_mode="truth",
            eor_lag_sigma_profile="strict",
            note="strict eor lag + strict amplitude + corr mean=true-global",
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


def _global_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den < 1e-18:
        return 0.0
    return float(np.dot(x, y) / den)


def _per_freq_corr_mean(a: np.ndarray, b: np.ndarray) -> float:
    vals: List[float] = []
    for i in range(a.shape[0]):
        vals.append(_global_corr(a[i], b[i]))
    return float(np.mean(vals))


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


def truth_global_corr(ds: DatasetSpec, cut_size_frac: float) -> float:
    with fits.open(ds.fg_true_cube, memmap=True) as hdul:
        fg = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.eor_true_cube, memmap=True) as hdul:
        eor = np.asarray(hdul[0].data, dtype=np.float32)
    fg = _center_cut(fg, cut_size_frac)
    eor = _center_cut(eor, cut_size_frac)
    return _global_corr(fg, eor)


def build_config(
    args: argparse.Namespace,
    ds: DatasetSpec,
    cand: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    fg_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]],
    truth_corr_cache: Dict[str, float],
) -> Dict[str, object]:
    lag_intervals = list(LAG_INTERVALS_MHZ)
    loose_sigma = [0.35, 0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.12, 0.12]
    strict_sigma = [0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04]
    eor_lag_mean = [0.0] * len(lag_intervals)
    eor_lag_sigma = loose_sigma if cand.eor_lag_sigma_profile == "loose" else strict_sigma

    if cand.loss_mode == "lagcorr":
        fg_means, fg_sigmas_native = derive_fg_prior(
            ds,
            lagcorr_feature=cand.lagcorr_feature,
            cut_size_frac=float(args.cut_size_frac),
            lag_intervals_mhz=lag_intervals,
            freq_delta_mhz=0.1,
            max_pairs=256,
            cache=fg_cache,
        )
        fg_sigmas = [max(float(v) * 2.0, 1e-6) for v in fg_sigmas_native]
        lagcorr_weight = 1.0
    else:
        fg_means = [1.0] * len(lag_intervals)
        fg_sigmas = [1.0] * len(lag_intervals)
        lagcorr_weight = 0.0

    if cand.corr_prior_mode == "off":
        corr_prior_mean = 0.0
    elif cand.corr_prior_mode == "zero":
        corr_prior_mean = 0.0
    elif cand.corr_prior_mode == "truth":
        corr_prior_mean = float(truth_corr_cache[ds.name])
    else:
        raise ValueError(f"Unsupported corr_prior_mode: {cand.corr_prior_mode}")

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
            "corr_weight": float(cand.corr_weight),
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
            "corr_prior_mean": float(corr_prior_mean),
            "corr_prior_sigma": 0.5,
            "lagcorr_feature": cand.lagcorr_feature,
            "lagcorr_unit": "mhz",
            "lagcorr_pair_sampling": "random",
            "lagcorr_random_seed": 20260211,
            "lagcorr_intervals": lag_intervals,
            "fg_lagcorr_mean": fg_means,
            "fg_lagcorr_sigma": fg_sigmas,
            "eor_lagcorr_mean": eor_lag_mean,
            "eor_lagcorr_sigma": eor_lag_sigma,
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
        "init": {"init_fg_cube": "", "init_eor_cube": ""},
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
        vals.append(_global_corr(est[i], true[i]))
    return float(np.mean(vals))


def compute_metrics(eor_est_path: Path, fg_est_path: Path, ds: DatasetSpec) -> Dict[str, Optional[float]]:
    if not eor_est_path.exists() or not fg_est_path.exists():
        return {
            "eor_mse": None,
            "eor_corr_mean": None,
            "eor_std_ratio": None,
            "recon_mse": None,
            "fg_eor_corr_est_global": None,
            "fg_eor_corr_est_perfreq_mean": None,
            "fg_eor_corr_true_global": None,
            "fg_eor_corr_true_perfreq_mean": None,
        }

    with fits.open(eor_est_path, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(fg_est_path, memmap=True) as hdul:
        fg_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.eor_true_cube, memmap=True) as hdul:
        eor_true_full = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.fg_true_cube, memmap=True) as hdul:
        fg_true_full = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(ds.input_cube, memmap=True) as hdul:
        y_full = np.asarray(hdul[0].data, dtype=np.float32)

    eor_true = _center_crop_to_shape(eor_true_full, eor_est.shape)
    fg_true = _center_crop_to_shape(fg_true_full, eor_est.shape)
    y_obs = _center_crop_to_shape(y_full, eor_est.shape)

    eor_mse = float(np.mean((eor_est - eor_true) ** 2))
    eor_corr_mean = _frequency_corr_mean(eor_est, eor_true)
    eor_std_ratio = float(eor_est.std() / (float(eor_true.std()) + 1e-12))
    recon_mse = float(np.mean((fg_est + eor_est - y_obs) ** 2))

    fg_eor_corr_est_global = _global_corr(fg_est, eor_est)
    fg_eor_corr_est_perfreq_mean = _per_freq_corr_mean(fg_est, eor_est)
    fg_eor_corr_true_global = _global_corr(fg_true, eor_true)
    fg_eor_corr_true_perfreq_mean = _per_freq_corr_mean(fg_true, eor_true)
    return {
        "eor_mse": eor_mse,
        "eor_corr_mean": eor_corr_mean,
        "eor_std_ratio": eor_std_ratio,
        "recon_mse": recon_mse,
        "fg_eor_corr_est_global": fg_eor_corr_est_global,
        "fg_eor_corr_est_perfreq_mean": fg_eor_corr_est_perfreq_mean,
        "fg_eor_corr_true_global": fg_eor_corr_true_global,
        "fg_eor_corr_true_perfreq_mean": fg_eor_corr_true_perfreq_mean,
    }


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
    csv_path = output_dir / "lagcorr_conflict_ablation_results.csv"
    md_path = output_dir / "lagcorr_conflict_ablation_results.md"
    fields = [
        "candidate",
        "dataset",
        "loss_mode",
        "lagcorr_feature",
        "gamma",
        "eor_prior_sigma",
        "corr_weight",
        "corr_prior_mode",
        "lagcorr_fg_component_weight",
        "lagcorr_eor_component_weight",
        "status",
        "return_code",
        "runtime_sec",
        "eor_mse",
        "eor_corr_mean",
        "eor_std_ratio",
        "recon_mse",
        "fg_eor_corr_est_global",
        "fg_eor_corr_est_perfreq_mean",
        "fg_eor_corr_true_global",
        "fg_eor_corr_true_perfreq_mean",
        "note",
        "config_path",
        "log_path",
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
        handle.write("# lagcorr conflict ablation results\n\n")
        handle.write(
            "| candidate | dataset | corr_mode | gamma | eor_sigma | w_fg | w_eor | eor_corr_mean | eor_mse | std_ratio | corr_est_global | corr_true_global | recon_mse |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row.get('candidate')} | {row.get('dataset')} | {row.get('corr_prior_mode')} | "
                f"{_fmt(row.get('gamma'))} | {_fmt(row.get('eor_prior_sigma'))} | "
                f"{_fmt(row.get('lagcorr_fg_component_weight'))} | {_fmt(row.get('lagcorr_eor_component_weight'))} | "
                f"{_fmt(row.get('eor_corr_mean'))} | {_fmt(row.get('eor_mse'))} | {_fmt(row.get('eor_std_ratio'))} | "
                f"{_fmt(row.get('fg_eor_corr_est_global'))} | {_fmt(row.get('fg_eor_corr_true_global'))} | {_fmt(row.get('recon_mse'))} |\n"
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
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"lagcorr_conflict_ablation_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(data_dir)
    candidates = build_candidates()
    gpu_map = parse_gpu_map(args.gpu_map)
    fg_prior_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]] = {}
    truth_corr_cache: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []

    for ds in datasets:
        truth_corr_cache[ds.name] = truth_global_corr(ds, cut_size_frac=float(args.cut_size_frac))

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

            cfg = build_config(
                args=args,
                ds=ds,
                cand=cand,
                run_dir=run_dir,
                gpu_index=gpu_idx,
                fg_cache=fg_prior_cache,
                truth_corr_cache=truth_corr_cache,
            )
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

        procs: List[Tuple[JobSpec, subprocess.Popen[str], float]] = []
        for job in jobs:
            print(f"  [launch] dataset={job.dataset.name} gpu={job.gpu_index}")
            proc, t0 = launch_job(code_dir=code_dir, conda_env=args.conda_env, job=job)
            procs.append((job, proc, t0))

        while procs:
            pending: List[Tuple[JobSpec, subprocess.Popen[str], float]] = []
            for job, proc, t0 in procs:
                ret = proc.poll()
                if ret is None:
                    pending.append((job, proc, t0))
                    continue
                runtime = time.time() - t0
                close_job_process(proc)
                status = "ok" if ret == 0 else "failed"
                metrics = compute_metrics(job.eor_output, job.fg_output, job.dataset) if ret == 0 else {
                    "eor_mse": None,
                    "eor_corr_mean": None,
                    "eor_std_ratio": None,
                    "recon_mse": None,
                    "fg_eor_corr_est_global": None,
                    "fg_eor_corr_est_perfreq_mean": None,
                    "fg_eor_corr_true_global": None,
                    "fg_eor_corr_true_perfreq_mean": None,
                }
                row = {
                    "candidate": job.candidate.name,
                    "dataset": job.dataset.name,
                    "loss_mode": job.candidate.loss_mode,
                    "lagcorr_feature": job.candidate.lagcorr_feature,
                    "gamma": job.candidate.gamma,
                    "eor_prior_sigma": job.candidate.eor_prior_sigma,
                    "corr_weight": job.candidate.corr_weight,
                    "corr_prior_mode": job.candidate.corr_prior_mode,
                    "lagcorr_fg_component_weight": job.candidate.lagcorr_fg_component_weight,
                    "lagcorr_eor_component_weight": job.candidate.lagcorr_eor_component_weight,
                    "status": status,
                    "return_code": ret,
                    "runtime_sec": runtime,
                    "note": job.candidate.note,
                    "config_path": str(job.config_path),
                    "log_path": str(job.log_path),
                }
                row.update(metrics)
                rows.append(row)
                print(
                    "  [done] "
                    f"dataset={job.dataset.name} status={status} corr={row.get('eor_corr_mean')} mse={row.get('eor_mse')}"
                )
            procs = pending
            if procs:
                time.sleep(2.0)

    csv_path, md_path = write_summary(output_dir, rows)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "work_root": str(work_root),
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "args": _json_safe(vars(args)),
        "datasets": [_json_safe(ds.__dict__) for ds in datasets],
        "candidates": [_json_safe(c.__dict__) for c in candidates],
        "truth_global_corr_cut": truth_corr_cache,
        "result_csv": str(csv_path),
        "result_md": str(md_path),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[summary] csv={csv_path}")
    print(f"[summary] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
