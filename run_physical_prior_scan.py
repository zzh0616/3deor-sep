#!/usr/bin/env python3
"""
Scan candidate "physics-driven" extra loss terms (A1-A5) on injected cubes.

Primary evaluation metric (truth-based, post-hoc):
  - per-frequency corr(EoR_est[f], EoR_true[f]) across spatial pixels.

Secondary evaluation:
  - EoR-window 2D power-spectrum metrics (if power_config enables it).

This script is intended for fast ablations: it keeps the base loss fixed and
enables one (or a small set) of extra terms per candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from dataset_registry import DatasetSpec, build_datasets, default_dataset_name_hint


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    extra_loss_terms: Tuple[str, ...]
    optim_overrides: Dict[str, object]
    weight_overrides: Dict[str, object]
    prior_overrides: Dict[str, object]


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
    power_dir: Path


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(t) for t in _parse_csv_tokens(text)]


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for t in _parse_csv_tokens(text):
        out.append(int(float(t)))
    return out


def parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid gpu map token: {token}")
        key, value = token.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


def _fmt_float_token(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    s = f"{float(value):.6g}"
    if "e" in s or "E" in s:
        s = f"{float(value):.12f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan A1-A5 physical-prior loss candidates.")
    p.add_argument("--work-root", type=Path, default=Path.cwd())
    p.add_argument("--code-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--datasets", type=str, default="cube2", help=f"Comma-separated datasets. Common: {default_dataset_name_hint()}")
    p.add_argument("--exclude-from-ranking", type=str, default="cube1", help="Datasets excluded from rank aggregation.")
    p.add_argument("--gpu-map", type=str, default="cube2:0", help="Dataset->GPU mapping.")
    p.add_argument("--max-concurrent-jobs", type=int, default=1)
    p.add_argument("--num-iters", type=int, default=2500)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-start-mhz", type=float, default=106.0)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")

    # Base baseline (fixed across candidates unless overridden).
    p.add_argument("--data-error", type=float, default=0.005)
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument("--base-eor-amp-prior-mode", type=str, default="slice_rms_hinge", choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"])
    p.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2", choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"])
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Extra-loss schedule (shared).
    p.add_argument("--extra-loss-start-iter", type=int, default=500)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=0)

    # Optimizer (fixed unless overridden).
    p.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-patience", type=int, default=240)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-cooldown", type=int, default=80)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # Power spectrum + EoR-window evaluation.
    p.add_argument("--power-config", type=str, default="configs/power_eor_window.json", help="Path relative to code-dir.")

    # Candidate grids.
    p.add_argument("--include-control", action="store_true", help="Include base-only control.")
    p.add_argument("--fg-logcurv-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--fg-logcurv-sigma-list", type=str, default="1.0")
    p.add_argument("--fg-logcurv-eps", type=float, default=1e-6)
    p.add_argument("--fg-logcurv-softplus-scale", type=float, default=1.0)

    p.add_argument("--fg-lowrank-weight-list", type=str, default="0.3,1.0")
    p.add_argument("--fg-lowrank-rank-list", type=str, default="3")
    p.add_argument("--fg-lowrank-num-samples", type=int, default=4096)
    p.add_argument("--fg-lowrank-spatial-pool", type=int, default=8)
    p.add_argument("--fg-lowrank-normalize", type=str, default="rms", choices=["none", "rms"])
    p.add_argument("--fg-lowrank-tail-max-list", type=str, default="0.0")
    p.add_argument("--fg-lowrank-sigma-list", type=str, default="0.2")
    p.add_argument("--fg-lowrank-sample-mode", type=str, default="stride", choices=["stride", "random"])
    p.add_argument("--fg-lowrank-random-seed", type=int, default=0)
    p.add_argument("--fg-lowrank-eps", type=float, default=1e-12)

    p.add_argument("--eor-lagshape-weight-list", type=str, default="0.3,1.0")
    p.add_argument("--eor-lagshape-feature-list", type=str, default="raw", help="Comma-separated: raw,diff1.")
    p.add_argument("--eor-lagshape-spatial-pool-list", type=str, default="4")
    p.add_argument("--eor-lagshape-far-min-lag", type=int, default=70)
    p.add_argument("--eor-lagshape-tail-eps", type=float, default=0.05)
    p.add_argument("--eor-lagshape-mid-max-lag", type=int, default=50)
    p.add_argument("--eor-lagshape-rebound-eps-act", type=float, default=0.05)
    p.add_argument("--eor-lagshape-rebound-delta-up", type=float, default=0.02)

    p.add_argument("--laggap-weight-list", type=str, default="0.5,1.0")
    p.add_argument("--laggap-margin-list", type=str, default="0.1")
    p.add_argument("--laggap-sigma-list", type=str, default="0.2")
    p.add_argument("--lagcorr-unit", type=str, default="mhz", choices=["mhz", "chan"])
    p.add_argument("--lagcorr-feature", type=str, default="raw", choices=["raw", "diff1"])
    p.add_argument("--lagcorr-spatial-pool", type=int, default=4)
    p.add_argument("--lagcorr-max-pairs", type=int, default=0, help="0 means all pairs.")
    p.add_argument("--lagcorr-pair-sampling", type=str, default="head", choices=["head", "random"])
    p.add_argument("--lagcorr-random-seed", type=int, default=0)
    p.add_argument("--lagcorr-eor-start-iter", type=int, default=1200)
    p.add_argument("--lagcorr-eor-ramp-iters", type=int, default=800)

    p.add_argument("--eor-iso-weight-list", type=str, default="0.1,0.3")
    p.add_argument("--eor-iso-spatial-pool-list", type=str, default="16")
    p.add_argument("--eor-iso-num-freq-samples", type=int, default=8)
    p.add_argument("--eor-iso-num-radial-bins", type=int, default=20)
    p.add_argument("--eor-iso-min-count", type=int, default=32)
    p.add_argument("--eor-iso-use-log-power", action="store_true")
    return p.parse_args()


def _extract_cut_indices(shape: Tuple[int, int, int], frac: float) -> Tuple[int, int, int, int]:
    _, nx, ny = shape
    if not (0.0 < frac <= 1.0):
        raise ValueError("cut-size-frac must be in (0,1].")
    size_x = max(1, int(round(nx * frac)))
    size_y = max(1, int(round(ny * frac)))
    size = min(size_x, size_y)
    x0 = (nx - size) // 2
    y0 = (ny - size) // 2
    return x0, x0 + size, y0, y0 + size


def _load_cube_cut(path: Path, *, cut: Tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = cut
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        cube = np.asarray(data[:, x0:x1, y0:y1], dtype=np.float32)
    return cube


def _center_crop_xy(cube: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    f, nx, ny = cube.shape
    tf, tx, ty = target_shape
    if f != tf:
        raise ValueError("Frequency dimension mismatch for center crop.")
    if nx == tx and ny == ty:
        return cube
    x0 = (nx - tx) // 2
    y0 = (ny - ty) // 2
    return cube[:, x0 : x0 + tx, y0 : y0 + ty]


def _frequency_correlations(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    f = a.shape[0]
    out = np.full((f,), np.nan, dtype=np.float64)
    for i in range(f):
        x = a[i].reshape(-1).astype(np.float64, copy=False)
        y = b[i].reshape(-1).astype(np.float64, copy=False)
        x = x - np.mean(x)
        y = y - np.mean(y)
        denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y))) + float(eps)
        out[i] = float(np.sum(x * y) / denom)
    return out


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    # Robust scalar score: mean of the worst 20% correlations.
    n = max(1, int(round(0.2 * finite.size)))
    worst = np.sort(finite)[:n]
    return float(np.mean(worst))


def _summarize_corr_stats(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_mean": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def _write_frequency_corr_profile(path: Path, corr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, v in enumerate(corr.tolist()):
            w.writerow([i, v])


def _read_eor_window_metrics(power_dir: Path) -> Dict[str, object]:
    metrics_path = power_dir / "power2d_eor_window_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        if isinstance(metrics, dict):
            return {f"ps2d_win_{k}": v for k, v in metrics.items()}
        return {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _candidate_summary(rows: Sequence[Dict[str, object]], exclude_datasets: Sequence[str]) -> List[Dict[str, object]]:
    exclude = set(exclude_datasets)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        if str(row.get("dataset")) in exclude:
            continue
        grouped.setdefault(str(row.get("candidate")), []).append(dict(row))
    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        eor_scores = [float(x.get("eor_corr_score", float("nan"))) for x in items]
        eor_scores = [x for x in eor_scores if math.isfinite(x)]
        ps_mad = [float(x.get("ps2d_win_log10_mad", float("nan"))) for x in items]
        ps_mad = [x for x in ps_mad if math.isfinite(x)]
        out.append(
            {
                "candidate": cand,
                "n_ok": int(len(items)),
                "eor_corr_score_mean": float(np.mean(eor_scores)) if eor_scores else float("nan"),
                "ps2d_win_log10_mad_mean": float(np.mean(ps_mad)) if ps_mad else float("nan"),
            }
        )
    out.sort(key=lambda r: (-(float(r["eor_corr_score_mean"]) if math.isfinite(float(r["eor_corr_score_mean"])) else -1e9)))
    return out


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Physical Prior Scan Summary\n\n")
    lines.append("## Meta\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("| rank | candidate | n_ok | eor_corr_score_mean | ps2d_win_log10_mad_mean |\n")
    lines.append("|---:|---|---:|---:|---:|\n")
    for i, row in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {row['candidate']} | {row['n_ok']} | "
            f"{float(row['eor_corr_score_mean']):.6f} | "
            f"{float(row['ps2d_win_log10_mad_mean']):.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    return [str(args.python_bin), str(cli_path), "--config", str(config_path)]


def _build_config(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    args: argparse.Namespace,
    code_dir: Path,
) -> Dict[str, object]:
    power_cfg_path = (code_dir / str(args.power_config)).resolve()
    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": list(candidate.extra_loss_terms),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "optimizer_name": str(args.optimizer_name),
            "momentum": float(args.momentum),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_plateau_patience": int(args.lr_plateau_patience),
            "lr_plateau_factor": float(args.lr_plateau_factor),
            "lr_plateau_min_delta": float(args.lr_plateau_min_delta),
            "lr_plateau_cooldown": int(args.lr_plateau_cooldown),
            "lr_min": float(args.lr_min),
            "freq_start_mhz": float(dataset.freq_start_mhz),
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
            "beta": float(args.base_beta),
            "gamma": float(args.base_gamma),
            "corr_weight": 0.0,
            "lagcorr_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
            "fg_logcurv_weight": 0.0,
            "fg_lowrank_weight": 0.0,
            "eor_lagshape_weight": 0.0,
            "eor_iso_weight": 0.0,
            "eor_mean_weight": 0.0,
            "eor_hf_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "eor_amp_prior_mode": str(args.base_eor_amp_prior_mode),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            # Defaults for A3/A4 to avoid relying on external FG references.
            "lagcorr_unit": str(args.lagcorr_unit),
            "lagcorr_feature": str(args.lagcorr_feature),
            "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
            "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
            "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
            "lagcorr_random_seed": int(args.lagcorr_random_seed),
            "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
            "lagcorr_lag_weights": 1.0,
            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
            "lagcorr_eor_far_min_lag": int(args.eor_lagshape_far_min_lag),
            "lagcorr_eor_tail_eps": float(args.eor_lagshape_tail_eps),
            "lagcorr_eor_mid_max_lag": int(args.eor_lagshape_mid_max_lag),
            "lagcorr_eor_rebound_eps_act": float(args.eor_lagshape_rebound_eps_act),
            "lagcorr_eor_rebound_delta_up": float(args.eor_lagshape_rebound_delta_up),
        },
        "power": {
            "power_config": str(power_cfg_path),
            "power_output_dir": str(run_dir / "powerspec"),
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
        },
    }

    # Apply candidate overrides.
    cfg["optim"].update(candidate.optim_overrides)
    cfg["weights"].update(candidate.weight_overrides)
    cfg["priors"].update(candidate.prior_overrides)
    return cfg


def generate_candidates(args: argparse.Namespace) -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    if bool(args.include_control):
        out.append(
            CandidateSpec(
                name="base",
                extra_loss_terms=(),
                optim_overrides={},
                weight_overrides={},
                prior_overrides={},
            )
        )

    for w in _parse_float_list(args.fg_logcurv_weight_list):
        for sig in _parse_float_list(args.fg_logcurv_sigma_list):
            name = f"a1_logcurv_w{_fmt_float_token(w)}_s{_fmt_float_token(sig)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("fg_logcurv",),
                    optim_overrides={},
                    weight_overrides={"fg_logcurv_weight": float(w)},
                    prior_overrides={
                        "fg_logcurv_mean": 0.0,
                        "fg_logcurv_sigma": float(sig),
                        "fg_logcurv_eps": float(args.fg_logcurv_eps),
                        "fg_logcurv_softplus_scale": float(args.fg_logcurv_softplus_scale),
                    },
                )
            )

    for w in _parse_float_list(args.fg_lowrank_weight_list):
        for r in _parse_int_list(args.fg_lowrank_rank_list):
            for tail_max in _parse_float_list(args.fg_lowrank_tail_max_list):
                for sig in _parse_float_list(args.fg_lowrank_sigma_list):
                    name = (
                        f"a2_lowrank_w{_fmt_float_token(w)}"
                        f"_r{int(r)}_t{_fmt_float_token(tail_max)}_s{_fmt_float_token(sig)}"
                    )
                    out.append(
                        CandidateSpec(
                            name=name,
                            extra_loss_terms=("fg_lowrank",),
                            optim_overrides={},
                            weight_overrides={"fg_lowrank_weight": float(w)},
                            prior_overrides={
                                "fg_lowrank_rank": int(r),
                                "fg_lowrank_num_samples": int(args.fg_lowrank_num_samples),
                                "fg_lowrank_spatial_pool": int(args.fg_lowrank_spatial_pool),
                                "fg_lowrank_normalize": str(args.fg_lowrank_normalize),
                                "fg_lowrank_tail_max": float(tail_max),
                                "fg_lowrank_sigma": float(sig),
                                "fg_lowrank_sample_mode": str(args.fg_lowrank_sample_mode),
                                "fg_lowrank_random_seed": int(args.fg_lowrank_random_seed),
                                "fg_lowrank_eps": float(args.fg_lowrank_eps),
                            },
                        )
                    )

    for w in _parse_float_list(args.eor_lagshape_weight_list):
        for feat in _parse_csv_tokens(args.eor_lagshape_feature_list):
            for pool in _parse_int_list(args.eor_lagshape_spatial_pool_list):
                feat_norm = str(feat).strip().lower()
                name = (
                    f"a3_lagshape_w{_fmt_float_token(w)}"
                    f"_{feat_norm}_p{int(pool)}"
                )
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("eor_lagshape",),
                        optim_overrides={},
                        weight_overrides={"eor_lagshape_weight": float(w)},
                        prior_overrides={
                            "eor_lagshape_feature": feat_norm,
                            "eor_lagshape_spatial_pool": int(pool),
                        },
                    )
                )

    for w in _parse_float_list(args.laggap_weight_list):
        for margin in _parse_float_list(args.laggap_margin_list):
            for sig in _parse_float_list(args.laggap_sigma_list):
                name = (
                    f"a4_laggap_w{_fmt_float_token(w)}"
                    f"_m{_fmt_float_token(margin)}_s{_fmt_float_token(sig)}"
                )
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("lagcorr",),
                        optim_overrides={},
                        weight_overrides={"lagcorr_weight": 1.0},
                        prior_overrides={
                            "lagcorr_fg_component_weight": 0.0,
                            "lagcorr_eor_component_weight": 0.0,
                            "lagcorr_gap_weight": float(w),
                            "lagcorr_gap_mode": "hinge",
                            "lagcorr_gap_margin": float(margin),
                            "lagcorr_gap_sigma": float(sig),
                            # Keep schedule on to reduce early-stage conflict.
                            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                        },
                    )
                )

    for w in _parse_float_list(args.eor_iso_weight_list):
        for pool in _parse_int_list(args.eor_iso_spatial_pool_list):
            name = f"a5_iso_w{_fmt_float_token(w)}_p{int(pool)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("eor_iso",),
                    optim_overrides={},
                    weight_overrides={"eor_iso_weight": float(w)},
                    prior_overrides={
                        "eor_iso_spatial_pool": int(pool),
                        "eor_iso_num_freq_samples": int(args.eor_iso_num_freq_samples),
                        "eor_iso_num_radial_bins": int(args.eor_iso_num_radial_bins),
                        "eor_iso_min_count": int(args.eor_iso_min_count),
                        "eor_iso_use_log_power": bool(args.eor_iso_use_log_power),
                        "eor_iso_eps": 1e-12,
                    },
                )
            )

    # De-dup by name if lists overlap.
    seen = set()
    uniq: List[CandidateSpec] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(c)
    return uniq


def _run_job_result_only(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    true_eor: np.ndarray,
    true_fg: np.ndarray,
    return_code: int,
    runtime: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "candidate": candidate.name,
        "dataset": dataset.name,
        "freq_start_mhz": float(dataset.freq_start_mhz),
        "status": "ok" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "runtime_sec": float(runtime),
        "extra_loss_terms": ",".join(candidate.extra_loss_terms),
        "config_path": str(run_dir / "config.json"),
        "log_path": str(run_dir / "run.log"),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "power_dir": str(run_dir / "powerspec"),
    }
    if int(return_code) != 0:
        return row

    fg_out = run_dir / "fg_est.fits"
    eor_out = run_dir / "eor_est.fits"
    if not fg_out.exists() or not eor_out.exists():
        row["status"] = "failed"
        row["note"] = "missing_output"
        return row

    with fits.open(eor_out, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(fg_out, memmap=True) as hdul:
        fg_est = np.asarray(hdul[0].data, dtype=np.float32)

    true_eor_use = true_eor if true_eor.shape == eor_est.shape else _center_crop_xy(true_eor, eor_est.shape)
    true_fg_use = true_fg if true_fg.shape == fg_est.shape else _center_crop_xy(true_fg, fg_est.shape)

    eor_corr = _frequency_correlations(eor_est, true_eor_use)
    _write_frequency_corr_profile(run_dir / "eor_corr_profile.csv", eor_corr)
    eor_stats = _summarize_corr_stats(eor_corr)
    row["eor_corr_mean"] = eor_stats["mean"]
    row["eor_corr_p10"] = eor_stats["p10"]
    row["eor_corr_min"] = eor_stats["min"]
    row["eor_corr_max"] = eor_stats["max"]
    row["eor_corr_score"] = float(_score_from_corr(eor_corr))

    fg_eor_corr = _frequency_correlations(fg_est, eor_est)
    fg_eor_stats = _summarize_corr_stats(fg_eor_corr)
    row["fg_eor_corr_abs_mean"] = fg_eor_stats["abs_mean"]
    inj_fg_eor_corr = _frequency_correlations(true_fg_use, true_eor_use)
    inj_stats = _summarize_corr_stats(inj_fg_eor_corr)
    row["inj_fg_eor_corr_abs_mean"] = inj_stats["abs_mean"]

    row.update(_read_eor_window_metrics(run_dir / "powerspec"))
    return row


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()

    if args.code_dir:
        code_dir = args.code_dir.resolve()
    else:
        code_dir = (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"physical_prior_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir, cube12_start_mhz=float(args.freq_start_mhz))
    enabled = {x.strip() for x in str(args.datasets).split(",") if x.strip()}
    datasets = [d for d in datasets_all if d.name in enabled]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")
    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

    candidates = generate_candidates(args)
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": [{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(d).items()} for d in datasets],
        "candidates": [
            {
                "name": c.name,
                "extra_loss_terms": list(c.extra_loss_terms),
                "optim_overrides": c.optim_overrides,
                "weight_overrides": c.weight_overrides,
                "prior_overrides": c.prior_overrides,
            }
            for c in candidates
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] generated {len(candidates)} candidates under {output_dir}")
        return 0

    # Prepare dataset caches (cut + truth).
    ds_cache: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        with fits.open(ds.input_cube, memmap=True) as hdul:
            in_shape = tuple(int(v) for v in hdul[0].data.shape)
        cut = _extract_cut_indices(in_shape, float(args.cut_size_frac))
        ds_cache[ds.name] = {
            "cut": cut,
            "true_eor": _load_cube_cut(ds.eor_true_cube, cut=cut),
            "true_fg": _load_cube_cut(ds.fg_true_cube, cut=cut),
        }

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
                    power_dir=run_dir / "powerspec",
                )
            )

        active: List[Tuple[object, JobSpec, float, object]] = []
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
                cfg = _build_config(
                    dataset=job.dataset,
                    candidate=job.candidate,
                    run_dir=job.run_dir,
                    gpu_index=job.gpu_index,
                    args=args,
                    code_dir=code_dir,
                )
                job.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                import subprocess  # local import to keep import side-effects minimal

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(code_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append((proc, job, time.time(), log_handle))
                active_gpus.add(int(job.gpu_index))
                print(f"  [launch] {job.dataset.name} gpu={job.gpu_index} pid={proc.pid}")

            still_active: List[Tuple[object, JobSpec, float, object]] = []
            for proc, job, t0, log_handle in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, t0, log_handle))
                    continue
                log_handle.close()
                runtime = time.time() - t0
                cache = ds_cache[job.dataset.name]
                true_eor = cache["true_eor"]
                true_fg = cache["true_fg"]
                assert isinstance(true_eor, np.ndarray) and isinstance(true_fg, np.ndarray)
                row = _run_job_result_only(
                    dataset=job.dataset,
                    candidate=job.candidate,
                    run_dir=job.run_dir,
                    true_eor=true_eor,
                    true_fg=true_fg,
                    return_code=int(ret),
                    runtime=float(runtime),
                )
                rows.append(row)
                print(
                    f"  [done] {job.dataset.name} status={row['status']} "
                    f"eor_score={row.get('eor_corr_score')} ps_mad={row.get('ps2d_win_log10_mad')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "physical_prior_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows, exclude_datasets=_parse_csv_tokens(args.exclude_from_ranking))
    rank_csv = output_dir / "physical_prior_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "physical_prior_summary.md", ranked, manifest)
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

