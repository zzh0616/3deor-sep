#!/usr/bin/env python3
"""
Numerical experiment for lag-correlation loss design under minimal EoR priors.

Design principle for this experiment:
- Do not use EoR lag-shape priors in the tested candidates.
- Keep only decomposition/data constraints + FG smoothness + FG lagcorr variants.
- Compare separation quality and runtime to find lagcorr settings that are
  discriminative but not too expensive.
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
    lagcorr_weight: float
    lagcorr_fg_component_weight: float
    lagcorr_eor_component_weight: float
    lagcorr_feature: str
    fg_prior_mode: str
    fg_sigma_mode: str
    fg_sigma_scale: float
    fg_sigma_floor: float


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
FG_SIGMA_LEGACY: List[float] = [0.005, 0.005, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lagcorr design experiment on two cubes."
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd).",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help="3dnet directory (default: <work-root>/3dnet).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: <work-root>/data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <work-root>/runs/lagcorr_design_<timestamp>).",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=400,
        help="Iterations per run.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Logging interval.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--cut-size-frac",
        type=float,
        default=0.30,
        help="Spatial cut fraction.",
    )
    parser.add_argument(
        "--extra-loss-start-iter",
        type=int,
        default=80,
        help="Start iteration of extra loss terms.",
    )
    parser.add_argument(
        "--extra-loss-ramp-iters",
        type=int,
        default=120,
        help="Ramp iterations for extra loss terms.",
    )
    parser.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset to GPU mapping, e.g. cube1:0,cube2:1",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="torch",
        help="Conda env for running separation_cli.py. Use empty string to use current python.",
    )
    parser.add_argument(
        "--include-reference-mixed",
        action="store_true",
        help="Include current mixed lagcorr (FG+EoR) as reference.",
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


def build_candidates(include_reference_mixed: bool) -> List[CandidateSpec]:
    candidates: List[CandidateSpec] = [
        CandidateSpec(
            name="base_noeorprior",
            loss_mode="base",
            lagcorr_weight=0.0,
            lagcorr_fg_component_weight=0.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="raw",
            fg_prior_mode="fixed",
            fg_sigma_mode="legacy",
            fg_sigma_scale=1.0,
            fg_sigma_floor=5e-4,
        ),
        CandidateSpec(
            name="lag_raw_fixed_legacy",
            loss_mode="lagcorr",
            lagcorr_weight=1.0,
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="raw",
            fg_prior_mode="fixed",
            fg_sigma_mode="legacy",
            fg_sigma_scale=1.0,
            fg_sigma_floor=5e-4,
        ),
        CandidateSpec(
            name="lag_raw_fgref_s5",
            loss_mode="lagcorr",
            lagcorr_weight=1.0,
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="raw",
            fg_prior_mode="fg_reference",
            fg_sigma_mode="legacy",
            fg_sigma_scale=5.0,
            fg_sigma_floor=1e-6,
        ),
        CandidateSpec(
            name="lag_raw_fgref_s2",
            loss_mode="lagcorr",
            lagcorr_weight=1.0,
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="raw",
            fg_prior_mode="fg_reference",
            fg_sigma_mode="legacy",
            fg_sigma_scale=2.0,
            fg_sigma_floor=1e-6,
        ),
        CandidateSpec(
            name="lag_diff1_fgref_s5",
            loss_mode="lagcorr",
            lagcorr_weight=1.0,
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="diff1",
            fg_prior_mode="fg_reference",
            fg_sigma_mode="flat",
            fg_sigma_scale=5.0,
            fg_sigma_floor=1e-6,
        ),
        CandidateSpec(
            name="lag_diff1_fgref_s2",
            loss_mode="lagcorr",
            lagcorr_weight=1.0,
            lagcorr_fg_component_weight=1.0,
            lagcorr_eor_component_weight=0.0,
            lagcorr_feature="diff1",
            fg_prior_mode="fg_reference",
            fg_sigma_mode="flat",
            fg_sigma_scale=2.0,
            fg_sigma_floor=1e-6,
        ),
    ]
    if include_reference_mixed:
        candidates.append(
            CandidateSpec(
                name="lag_mixed_reference",
                loss_mode="lagcorr",
                lagcorr_weight=1.0,
                lagcorr_fg_component_weight=0.5,
                lagcorr_eor_component_weight=0.5,
                lagcorr_feature="raw",
                fg_prior_mode="fixed",
                fg_sigma_mode="legacy",
                fg_sigma_scale=1.0,
                fg_sigma_floor=5e-4,
            )
        )
    return candidates


def _build_fg_sigma_template(mode: str, n_lags: int) -> List[float]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "legacy":
        if n_lags != len(FG_SIGMA_LEGACY):
            raise ValueError(f"legacy sigma template requires {len(FG_SIGMA_LEGACY)} lags.")
        return list(FG_SIGMA_LEGACY)
    if mode_norm == "flat":
        return [0.005] * n_lags
    raise ValueError(f"Unsupported fg_sigma_mode: {mode}")


def _resolve_center_cut(arr: np.ndarray, cut_size_frac: float) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {arr.shape}")
    _, ny, nx = arr.shape
    frac = float(cut_size_frac)
    if frac <= 0:
        raise ValueError("cut_size_frac must be > 0.")
    size = int(round(min(ny, nx) * frac))
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
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")
    num_freqs = int(cube.shape[0])
    flat = cube.reshape(num_freqs, -1).astype(np.float64, copy=False)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms = np.clip(norms, 1e-12, None)

    means: List[float] = []
    sigmas: List[float] = []
    for lag in lag_channels:
        lag_i = int(lag)
        if lag_i < 1 or lag_i >= num_freqs:
            raise ValueError(f"Invalid lag {lag_i} for num_freqs={num_freqs}")
        total = num_freqs - lag_i
        use_pairs = total if max_pairs is None else min(total, int(max_pairs))
        if use_pairs <= 0:
            raise ValueError(f"No valid pairs for lag={lag_i}")
        idx = np.arange(use_pairs, dtype=np.int64)
        jdx = idx + lag_i
        dot = np.sum(centered[idx] * centered[jdx], axis=1)
        den = norms[idx] * norms[jdx]
        corr = dot / np.clip(den, 1e-12, None)
        means.append(float(np.mean(corr)))
        sigmas.append(float(np.std(corr)))
    return means, sigmas


def _derive_fg_lag_prior(
    *,
    ds: DatasetSpec,
    cand: CandidateSpec,
    cut_size_frac: float,
    lag_intervals_mhz: Sequence[float],
    freq_delta_mhz: float,
    max_pairs: Optional[int],
    cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]],
) -> Tuple[List[float], List[float]]:
    key = (ds.name, cand.lagcorr_feature, float(cut_size_frac))
    cached = cache.get(key)
    if cached is not None:
        return cached

    with fits.open(ds.fg_true_cube, memmap=True) as hdul:
        fg_true = np.asarray(hdul[0].data, dtype=np.float32)
    fg_true = _resolve_center_cut(fg_true, cut_size_frac=cut_size_frac)
    if cand.lagcorr_feature == "diff1":
        if fg_true.shape[0] < 2:
            raise ValueError("lagcorr_feature='diff1' requires at least 2 frequency channels.")
        fg_true = np.diff(fg_true, n=1, axis=0)

    lag_channels = [
        max(1, int(round(float(v) / float(freq_delta_mhz)))) for v in lag_intervals_mhz
    ]
    means, sigmas = _compute_lagcorr_profile(
        fg_true,
        lag_channels=lag_channels,
        max_pairs=max_pairs,
    )
    cache[key] = (means, sigmas)
    return means, sigmas


def build_config(
    args: argparse.Namespace,
    ds: DatasetSpec,
    cand: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    fg_prior_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]],
) -> Dict[str, object]:
    lag_intervals = list(LAG_INTERVALS_MHZ)
    if cand.fg_prior_mode == "fg_reference":
        fg_means, fg_sigmas_native = _derive_fg_lag_prior(
            ds=ds,
            cand=cand,
            cut_size_frac=float(args.cut_size_frac),
            lag_intervals_mhz=lag_intervals,
            freq_delta_mhz=0.1,
            max_pairs=256,
            cache=fg_prior_cache,
        )
        fg_sigma = [
            max(float(v) * float(cand.fg_sigma_scale), float(cand.fg_sigma_floor))
            for v in fg_sigmas_native
        ]
        fg_mean = [float(v) for v in fg_means]
    elif cand.fg_prior_mode == "fixed":
        fg_sigma_base = _build_fg_sigma_template(cand.fg_sigma_mode, n_lags=len(lag_intervals))
        fg_sigma = [
            max(float(v) * float(cand.fg_sigma_scale), float(cand.fg_sigma_floor))
            for v in fg_sigma_base
        ]
        fg_mean = [1.0] * len(lag_intervals)
    else:
        raise ValueError(f"Unsupported fg_prior_mode: {cand.fg_prior_mode}")

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
        # No explicit EoR shape prior in this experiment.
        "gamma": 0.0,
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
            "gamma": 0.0,
            "corr_weight": 0.0,
            "lagcorr_weight": float(cand.lagcorr_weight),
            "lagcorr_fg_component_weight": float(cand.lagcorr_fg_component_weight),
            "lagcorr_eor_component_weight": float(cand.lagcorr_eor_component_weight),
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": 0.005,
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": 0.1,
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
            "lagcorr_random_seed": 20260210,
            "lagcorr_intervals": lag_intervals,
            "fg_lagcorr_mean": fg_mean,
            "fg_lagcorr_sigma": fg_sigma,
            # Kept only for compatibility; in FG-only candidates the EoR component weight is 0.
            "eor_lagcorr_mean": [0.0] * len(lag_intervals),
            "eor_lagcorr_sigma": [1.0] * len(lag_intervals),
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


def compute_metrics(eor_est_path: Path, fg_est_path: Path, ds: DatasetSpec) -> Dict[str, Optional[float]]:
    if not eor_est_path.exists() or not fg_est_path.exists():
        return {
            "eor_mse": None,
            "eor_corr_mean": None,
            "eor_std_ratio": None,
            "recon_mse": None,
        }

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
    return {
        "eor_mse": eor_mse,
        "eor_corr_mean": eor_corr_mean,
        "eor_std_ratio": eor_std_ratio,
        "recon_mse": recon_mse,
    }


def launch_job(
    code_dir: Path,
    conda_env: str,
    job: JobSpec,
) -> Tuple[subprocess.Popen[str], float]:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str]
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
        cmd = [
            sys.executable,
            str(code_dir / "separation_cli.py"),
            "--config",
            str(job.config_path),
        ]
    handle = job.log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(code_dir.parent),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # keep handle alive via proc object attribute
    setattr(proc, "_log_handle", handle)
    return proc, time.time()


def close_job_process(proc: subprocess.Popen[str]) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        try:
            handle.flush()
        finally:
            handle.close()


def write_summary(
    output_dir: Path,
    rows: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    csv_path = output_dir / "lagcorr_design_results.csv"
    md_path = output_dir / "lagcorr_design_results.md"
    fields = [
        "candidate",
        "loss_mode",
        "lagcorr_feature",
        "fg_prior_mode",
        "fg_sigma_mode",
        "fg_sigma_scale",
        "dataset",
        "gpu",
        "status",
        "return_code",
        "runtime_sec",
        "eor_mse",
        "eor_corr_mean",
        "eor_std_ratio",
        "recon_mse",
        "config_path",
        "log_path",
        "fg_output",
        "eor_output",
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
        handle.write("# lagcorr design experiment\n\n")
        handle.write(
            "| candidate | feature | prior | sigma_mode | sigma_scale | dataset | gpu | status | runtime_sec | eor_mse | eor_corr_mean | eor_std_ratio | recon_mse |\n"
        )
        handle.write("|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row.get('candidate')} | {row.get('lagcorr_feature')} | {row.get('fg_prior_mode')} | "
                f"{row.get('fg_sigma_mode')} | {_fmt(row.get('fg_sigma_scale'))} | "
                f"{row.get('dataset')} | {row.get('gpu')} | {row.get('status')} | "
                f"{_fmt(row.get('runtime_sec'))} | {_fmt(row.get('eor_mse'))} | {_fmt(row.get('eor_corr_mean'))} | "
                f"{_fmt(row.get('eor_std_ratio'))} | {_fmt(row.get('recon_mse'))} |\n"
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
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"lagcorr_design_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_map = parse_gpu_map(args.gpu_map)
    datasets = build_datasets(data_dir)
    candidates = build_candidates(include_reference_mixed=bool(args.include_reference_mixed))

    rows: List[Dict[str, object]] = []
    fg_prior_cache: Dict[Tuple[str, str, float], Tuple[List[float], List[float]]] = {}

    for cand in candidates:
        print(f"[candidate] {cand.name}")
        jobs: List[JobSpec] = []
        for ds in datasets:
            gpu_index = gpu_map.get(ds.name, 0)
            run_dir = output_dir / cand.name / ds.name
            cfg_path = run_dir / "config.json"
            log_path = run_dir / "run.log"
            fg_output = run_dir / "fg_est.fits"
            eor_output = run_dir / "eor_est.fits"
            cfg = build_config(args, ds, cand, run_dir, gpu_index, fg_prior_cache)
            run_dir.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle, indent=2)
            jobs.append(
                JobSpec(
                    dataset=ds,
                    candidate=cand,
                    gpu_index=gpu_index,
                    run_dir=run_dir,
                    config_path=cfg_path,
                    log_path=log_path,
                    fg_output=fg_output,
                    eor_output=eor_output,
                )
            )

        procs: List[Tuple[JobSpec, subprocess.Popen[str], float]] = []
        for job in jobs:
            print(
                f"  [launch] dataset={job.dataset.name} gpu={job.gpu_index} "
                f"config={job.config_path}"
            )
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
                }
                row: Dict[str, object] = {
                    "candidate": job.candidate.name,
                    "loss_mode": job.candidate.loss_mode,
                    "lagcorr_feature": job.candidate.lagcorr_feature,
                    "fg_prior_mode": job.candidate.fg_prior_mode,
                    "fg_sigma_mode": job.candidate.fg_sigma_mode,
                    "fg_sigma_scale": job.candidate.fg_sigma_scale,
                    "dataset": job.dataset.name,
                    "gpu": job.gpu_index,
                    "status": status,
                    "return_code": ret,
                    "runtime_sec": runtime,
                    "config_path": str(job.config_path),
                    "log_path": str(job.log_path),
                    "fg_output": str(job.fg_output),
                    "eor_output": str(job.eor_output),
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"  [done] dataset={job.dataset.name} status={status} "
                    f"runtime={runtime:.1f}s eor_mse={metrics.get('eor_mse')}"
                )
            procs = pending
            if procs:
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
