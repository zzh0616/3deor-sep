#!/usr/bin/env python3
"""
Scan poly_reparam combined with previously-tested extra loss terms on injected cubes.

Primary evaluation metric (truth-based, post-hoc):
  - per-frequency corr(EoR_est[f], EoR_true[f]) across spatial pixels.

Secondary evaluation:
  - EoR-window 2D power-spectrum metrics inside a configurable EoR window.

Notes
-----
- This script deliberately evaluates *outputs*, not training loss values.
- Ranking typically excludes contaminated cube1 but still reports it.
- For lagcorr FG-only ("lag_fg_corr") we support an "oracle" prior derived from
  injected FG truth, so we can test whether the constraint is potentially useful.
  (This is not intended as a real-data prior; it's an ablation tool.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for token in _parse_csv_tokens(text):
        out.append(int(float(token)))
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
    p = argparse.ArgumentParser(description="Scan poly_reparam combos with other extra loss terms.")
    p.add_argument("--work-root", type=Path, default=Path.cwd())
    p.add_argument("--code-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--datasets",
        type=str,
        default="cube1,cube2",
        help=f"Comma-separated datasets. Common: {default_dataset_name_hint()}",
    )
    p.add_argument("--exclude-from-ranking", type=str, default="cube1")
    p.add_argument("--gpu-map", type=str, default="cube1:0,cube2:1", help="Dataset->GPU mapping.")
    p.add_argument("--max-concurrent-jobs", type=int, default=2)
    p.add_argument("--num-iters", type=int, default=3000)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-start-mhz", type=float, default=106.0)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--candidate-names", type=str, default="", help="Optional comma-separated candidate filter.")

    # Base baseline (kept fixed unless overridden by a candidate).
    p.add_argument("--data-error", type=float, default=0.005)
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument(
        "--base-eor-amp-prior-mode",
        type=str,
        default="slice_rms_hinge",
        choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"],
    )
    p.add_argument("--base-eor-hybrid-voxel-factor", type=float, default=5.0)
    p.add_argument("--base-eor-hybrid-voxel-weight", type=float, default=0.1)
    p.add_argument(
        "--base-fg-smooth-mode",
        type=str,
        default="diff2_l2",
        choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"],
    )
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Extra-loss schedule (shared).
    p.add_argument("--extra-loss-start-iter", type=int, default=500)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=0)

    # Optimizer.
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
    p.add_argument("--power-config", type=str, default="configs/power_eor_window.json")

    # Poly baseline (defaults set to current best on injected cube2).
    p.add_argument("--poly-weight", type=float, default=1.0)
    p.add_argument("--poly-degree", type=int, default=3)
    p.add_argument("--poly-sigma", type=float, default=0.05)
    p.add_argument(
        "--poly-degree-list",
        type=str,
        default="",
        help="Optional comma-separated poly degrees to generate multiple poly-only candidates (overrides --poly-degree).",
    )
    p.add_argument(
        "--poly-basis",
        type=str,
        default="power",
        choices=["power", "chebyshev", "legendre", "dct", "bspline"],
        help="Basis used by poly_reparam: power (legacy), chebyshev, legendre, dct, bspline.",
    )
    p.add_argument(
        "--poly-basis-list",
        type=str,
        default="",
        help="Optional comma-separated poly bases to generate multiple poly-only candidates (overrides --poly-basis).",
    )
    p.add_argument(
        "--poly-x-mode",
        type=str,
        default="log",
        choices=["lin", "log"],
        help="Polynomial coordinate for poly_reparam (default log).",
    )
    p.add_argument(
        "--poly-x-mode-list",
        type=str,
        default="",
        help="Optional comma-separated poly x-modes to generate multiple poly-only candidates (overrides --poly-x-mode).",
    )
    p.add_argument(
        "--poly-model",
        type=str,
        default="exp_mul",
        choices=["add", "exp", "exp_mul"],
        help="Polynomial foreground model for poly_reparam (default exp_mul).",
    )
    p.add_argument(
        "--poly-model-list",
        type=str,
        default="",
        help="Optional comma-separated poly models to generate multiple poly-only candidates (overrides --poly-model).",
    )
    p.add_argument(
        "--poly-enable-resid",
        dest="poly_resid_enabled",
        action="store_true",
        help="Enable explicit per-voxel residual when poly_reparam is active (default disabled).",
    )
    p.add_argument(
        "--poly-resid-enabled-list",
        type=str,
        default="",
        help=(
            "Optional comma-separated booleans to generate multiple poly-only candidates "
            "(overrides --poly-enable-resid). Example: true,false"
        ),
    )
    p.add_argument(
        "--eor-as-residual",
        action="store_true",
        help="Set EoR = y - FG (optimize FG only) to reduce FG/EoR degeneracy.",
    )
    p.add_argument(
        "--init-mode",
        type=str,
        default="smooth_residual",
        choices=["smooth_zero", "smooth_residual", "poly_residual"],
        help="Initialization policy (default smooth_residual).",
    )
    p.add_argument(
        "--lr-fg-factor",
        type=float,
        default=0.5,
        help="Learning-rate multiplier for FG params (lr_fg = lr * lr_fg_factor).",
    )

    # What to include.
    p.add_argument("--include-control", action="store_true", help="Include base-only control candidate.")
    p.add_argument("--include-corr", action="store_true")
    p.add_argument("--include-lagcorr-fg", action="store_true", help="Include FG-only lagcorr (lag_fg_corr).")
    p.add_argument("--include-laggap", action="store_true", help="Include lag-gap-only candidates (A4 style).")
    p.add_argument("--include-fg-logcurv", action="store_true")
    p.add_argument("--include-fg-lowrank", action="store_true")
    p.add_argument("--include-eor-mean", action="store_true")
    p.add_argument("--include-eor-hf", action="store_true")
    p.add_argument("--include-eor-lagshape", action="store_true")
    p.add_argument("--include-eor-iso", action="store_true")
    p.add_argument("--include-combos", action="store_true", help="Include a few hand-picked 2-term combos on top of poly.")

    # Corr config (weak by default).
    p.add_argument("--corr-weight-list", type=str, default="0.05,0.1")
    p.add_argument("--corr-abs-threshold", type=float, default=0.08)
    p.add_argument("--corr-sigma", type=float, default=0.2)
    p.add_argument("--corr-reduce", type=str, default="logsumexp", choices=["mean", "topk", "logsumexp"])
    p.add_argument("--corr-topk", type=int, default=0)
    p.add_argument("--corr-lse-alpha", type=float, default=10.0)
    p.add_argument("--corr-feature", type=str, default="diff1", choices=["raw", "diff1", "diff2"])
    p.add_argument("--corr-spatial-pool", type=int, default=4)

    # Lagcorr FG-only config.
    p.add_argument("--lagcorr-fg-weight-list", type=str, default="0.03,0.1,0.3")
    p.add_argument("--lagcorr-feature-list", type=str, default="diff1,raw", help="Comma-separated: raw,diff1.")
    p.add_argument("--lagcorr-unit", type=str, default="mhz", choices=["mhz", "chan"])
    p.add_argument("--lagcorr-spatial-pool", type=int, default=4)
    p.add_argument("--lagcorr-max-pairs", type=int, default=256)
    p.add_argument("--lagcorr-pair-sampling", type=str, default="random", choices=["head", "random"])
    p.add_argument("--lagcorr-random-seed", type=int, default=20260214)
    p.add_argument("--lagcorr-rms-min", type=float, default=0.0)
    p.add_argument(
        "--lagcorr-fg-prior-source",
        type=str,
        default="truth",
        choices=["truth", "constant"],
        help="How to set fg_lagcorr_mean/sigma for lagcorr_fg-only.",
    )
    p.add_argument("--lagcorr-fg-const-mean", type=float, default=0.95)
    p.add_argument("--lagcorr-fg-const-sigma", type=float, default=0.15)
    p.add_argument("--lagcorr-fg-sigma-floor", type=float, default=0.08, help="Floor added to oracle sigma.")

    # Lag-gap (no absolute priors).
    p.add_argument("--laggap-weight-list", type=str, default="0.3,1.0")
    p.add_argument("--laggap-margin-list", type=str, default="0.05,0.1,0.2")
    p.add_argument("--laggap-sigma-list", type=str, default="0.2")
    p.add_argument("--lagcorr-eor-start-iter", type=int, default=500)
    p.add_argument("--lagcorr-eor-ramp-iters", type=int, default=0)

    # FG logcurv.
    p.add_argument("--fg-logcurv-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--fg-logcurv-sigma", type=float, default=1.0)
    p.add_argument(
        "--fg-logcurv-sigma-list",
        type=str,
        default="",
        help="Optional comma-separated sigma list. If empty, uses --fg-logcurv-sigma.",
    )
    p.add_argument("--fg-logcurv-eps", type=float, default=1e-6)
    p.add_argument("--fg-logcurv-softplus-scale", type=float, default=1.0)

    # FG lowrank.
    p.add_argument("--fg-lowrank-weight-list", type=str, default="0.3,1.0")
    p.add_argument("--fg-lowrank-rank", type=int, default=3)
    p.add_argument(
        "--fg-lowrank-rank-list",
        type=str,
        default="",
        help="Optional comma-separated rank list. If empty, uses --fg-lowrank-rank.",
    )
    p.add_argument("--fg-lowrank-num-samples", type=int, default=4096)
    p.add_argument("--fg-lowrank-spatial-pool", type=int, default=8)
    p.add_argument("--fg-lowrank-normalize", type=str, default="rms", choices=["none", "rms"])
    p.add_argument("--fg-lowrank-tail-max", type=float, default=0.0)
    p.add_argument(
        "--fg-lowrank-tail-max-list",
        type=str,
        default="",
        help="Optional comma-separated tail_max list. If empty, uses --fg-lowrank-tail-max.",
    )
    p.add_argument("--fg-lowrank-sigma", type=float, default=0.2)
    p.add_argument(
        "--fg-lowrank-sigma-list",
        type=str,
        default="",
        help="Optional comma-separated sigma list. If empty, uses --fg-lowrank-sigma.",
    )
    p.add_argument("--fg-lowrank-sample-mode", type=str, default="stride", choices=["stride", "random"])
    p.add_argument("--fg-lowrank-random-seed", type=int, default=0)
    p.add_argument("--fg-lowrank-eps", type=float, default=1e-12)

    # EoR mean/hf.
    p.add_argument("--eor-mean-weight-list", type=str, default="0.01,0.03,0.1")
    p.add_argument("--eor-hf-weight-list", type=str, default="0.1,0.3")
    p.add_argument("--eor-hf-percent", type=float, default=0.7)
    p.add_argument("--eor-hf-rmax-list", type=str, default="0.7,0.85")

    # EoR lagshape envelope (A3).
    p.add_argument("--eor-lagshape-weight-list", type=str, default="0.1,0.3")
    p.add_argument("--eor-lagshape-feature", type=str, default="diff1", choices=["raw", "diff1"])
    p.add_argument("--eor-lagshape-spatial-pool", type=int, default=4)
    p.add_argument("--eor-lagshape-far-min-lag", type=int, default=70)
    p.add_argument("--eor-lagshape-tail-eps", type=float, default=0.05)
    p.add_argument("--eor-lagshape-mid-max-lag", type=int, default=50)
    p.add_argument("--eor-lagshape-rebound-eps-act", type=float, default=0.05)
    p.add_argument("--eor-lagshape-rebound-delta-up", type=float, default=0.02)

    # EoR iso (A5).
    p.add_argument("--eor-iso-weight-list", type=str, default="0.1")
    p.add_argument("--eor-iso-spatial-pool", type=int, default=16)
    p.add_argument("--eor-iso-num-freq-samples", type=int, default=8)
    p.add_argument("--eor-iso-num-radial-bins", type=int, default=20)
    p.add_argument("--eor-iso-min-count", type=int, default=32)
    p.add_argument("--eor-iso-use-log-power", action="store_true")
    return p.parse_args()


def _extract_cut_indices(shape: Sequence[int], cut_frac: float) -> Optional[Tuple[int, int, int, int]]:
    if len(shape) < 3:
        return None
    f, ny, nx = int(shape[0]), int(shape[1]), int(shape[2])
    if f <= 0 or nx <= 0 or ny <= 0:
        return None
    min_dim = min(nx, ny)
    size_px = int(round(float(cut_frac) * float(min_dim)))
    size_px = max(1, min(size_px, min_dim))

    # Match separation_optim.build_cut_xy_indices default behavior for unit='frac', center=(0.5,0.5).
    center_x_px = int(round(0.5 * float(max(nx - 1, 0))))
    center_y_px = int(round(0.5 * float(max(ny - 1, 0))))

    def _clamp_fixed_window(start: int, size: int, length: int) -> Tuple[int, int]:
        if size > length:
            raise ValueError(f"Requested crop size {size} exceeds axis length {length}.")
        end = start + size
        if start < 0:
            start = 0
            end = size
        if end > length:
            end = length
            start = length - size
        return int(start), int(end)

    start_x = int(center_x_px) - int(size_px) // 2
    start_y = int(center_y_px) - int(size_px) // 2
    x0, x1 = _clamp_fixed_window(start_x, int(size_px), int(nx))
    y0, y1 = _clamp_fixed_window(start_y, int(size_px), int(ny))
    return (int(x0), int(x1), int(y0), int(y1))


def _load_cube_cut(path: Path, cut: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if cut is not None:
            x0, x1, y0, y1 = cut
            data = data[:, y0:y1, x0:x1]
        return np.asarray(data, dtype=np.float32)


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


def _summarize_corr_stats(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "min": float("nan"), "max": float("nan"), "abs_mean": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    worst_frac = 0.20
    k = max(1, int(math.ceil(float(finite.size) * worst_frac)))
    return float(np.mean(np.sort(finite)[:k]))


def _write_frequency_corr_profile(path: Path, corr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, v in enumerate(corr.tolist()):
            w.writerow([int(i), float(v)])


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
    """
    Best-effort convergence indicators from run.log.
    """
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


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    # Use unbuffered mode so run logs update continuously when redirected to files on remote hosts.
    return [str(args.python_bin), "-u", str(cli_path), "--config", str(config_path)]


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


def _lagcorr_oracle_from_true_fg(
    true_fg: np.ndarray,
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
    """
    Estimate fg_lagcorr_mean/sigma vectors from injected FG truth (oracle).
    """
    feat = str(feature).strip().lower()
    if feat == "raw":
        cube = true_fg
    elif feat == "diff1":
        if true_fg.shape[0] < 2:
            raise ValueError("diff1 requires at least 2 channels.")
        cube = np.diff(true_fg, axis=0)
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
        # norms / sqrt(Npix) is slice RMS in the centered domain.
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
        # variance guard
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


def _build_config(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    args: argparse.Namespace,
    code_dir: Path,
    fg_lagcorr_cache: Optional[Dict[Tuple[str, str, int], Tuple[List[float], List[float]]]] = None,
) -> Dict[str, object]:
    power_cfg_path = Path(args.power_config)
    if not power_cfg_path.is_absolute():
        power_cfg_path = (code_dir / str(power_cfg_path)).resolve()

    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "lr_fg_factor": float(args.lr_fg_factor),
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
            # poly baseline (in optim namespace)
            "poly_degree": int(args.poly_degree),
            "poly_sigma": float(args.poly_sigma),
            "poly_basis": str(args.poly_basis),
            "poly_x_mode": str(args.poly_x_mode),
            "poly_model": str(args.poly_model),
            "poly_resid_enabled": bool(args.poly_resid_enabled),
            "eor_as_residual": bool(args.eor_as_residual),
            "init_mode": str(args.init_mode),
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
            "lagcorr_fg_component_weight": 0.0,
            "lagcorr_eor_component_weight": 0.0,
            "lagcorr_gap_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": float(args.poly_weight),
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
            "eor_hybrid_voxel_factor": float(args.base_eor_hybrid_voxel_factor),
            "eor_hybrid_voxel_weight": float(args.base_eor_hybrid_voxel_weight),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            # Power config is per-run (for convenience).
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
        "init": {"init_fg_cube": "", "init_eor_cube": ""},
        "scan_meta": {"candidate_name": candidate.name},
    }

    # Apply candidate overrides.
    cfg["optim"].update(candidate.optim_overrides)
    cfg["weights"].update(candidate.weight_overrides)
    cfg["priors"].update(candidate.prior_overrides)

    # Wire lagcorr oracle priors if needed.
    extras = set(str(x).strip().lower() for x in candidate.extra_loss_terms)
    if "lagcorr" in extras and float(cfg["weights"].get("lagcorr_fg_component_weight", 0.0)) > 0.0:
        src = str(args.lagcorr_fg_prior_source).strip().lower()
        if src == "truth":
            if fg_lagcorr_cache is None:
                raise ValueError("Internal error: fg_lagcorr_cache missing.")
            feat = str(cfg["priors"].get("lagcorr_feature", "raw")).strip().lower()
            pool = int(cfg["priors"].get("lagcorr_spatial_pool", int(args.lagcorr_spatial_pool)))
            key = (dataset.name, feat, pool)
            if key not in fg_lagcorr_cache:
                raise ValueError(f"Missing FG lagcorr oracle cache for key={key}")
            mu, sig = fg_lagcorr_cache[key]
            cfg["priors"]["fg_lagcorr_mean"] = list(mu)
            cfg["priors"]["fg_lagcorr_sigma"] = list(sig)
        elif src == "constant":
            mu = float(args.lagcorr_fg_const_mean)
            sig = float(args.lagcorr_fg_const_sigma)
            cfg["priors"]["fg_lagcorr_mean"] = [mu] * len(LAG_INTERVALS_MHZ)
            cfg["priors"]["fg_lagcorr_sigma"] = [sig] * len(LAG_INTERVALS_MHZ)
        else:
            raise ValueError("lagcorr_fg_prior_source must be truth or constant.")
    return cfg


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
    lines.append("# Poly Combo Scan Summary\n\n")
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


def generate_candidates(args: argparse.Namespace) -> List[CandidateSpec]:
    out: List[CandidateSpec] = []

    poly_degrees = _parse_int_list(args.poly_degree_list) if str(args.poly_degree_list).strip() else [int(args.poly_degree)]
    poly_bases = _parse_csv_tokens(args.poly_basis_list) if str(args.poly_basis_list).strip() else [str(args.poly_basis)]
    poly_x_modes = _parse_csv_tokens(args.poly_x_mode_list) if str(args.poly_x_mode_list).strip() else [str(args.poly_x_mode)]
    poly_models = _parse_csv_tokens(args.poly_model_list) if str(args.poly_model_list).strip() else [str(args.poly_model)]

    resid_list: List[bool] = []
    if str(args.poly_resid_enabled_list).strip():
        for tok in _parse_csv_tokens(args.poly_resid_enabled_list):
            t = tok.strip().lower()
            if t in {"true", "1", "yes", "y"}:
                resid_list.append(True)
            elif t in {"false", "0", "no", "n"}:
                resid_list.append(False)
            else:
                raise ValueError("--poly-resid-enabled-list expects booleans like true/false.")
    else:
        resid_list = [bool(args.poly_resid_enabled)]

    poly_basis_allowed = {"power", "chebyshev", "legendre", "dct", "bspline"}
    poly_model_allowed = {"add", "exp", "exp_mul"}
    poly_x_allowed = {"lin", "log"}

    poly_variants: List[Tuple[int, str, str, str, bool]] = []
    seen: set = set()
    for d in poly_degrees:
        for b in poly_bases:
            b_norm = str(b).strip().lower()
            if b_norm not in poly_basis_allowed:
                raise ValueError(f"Invalid poly_basis '{b_norm}'. Expected one of: {sorted(poly_basis_allowed)}")
            for x in poly_x_modes:
                x_norm = str(x).strip().lower()
                if x_norm not in poly_x_allowed:
                    raise ValueError(f"Invalid poly_x_mode '{x_norm}'. Expected lin or log.")
                for m in poly_models:
                    m_norm = str(m).strip().lower()
                    if m_norm not in poly_model_allowed:
                        raise ValueError(f"Invalid poly_model '{m_norm}'. Expected one of: {sorted(poly_model_allowed)}")
                    for r in resid_list:
                        key = (int(d), b_norm, x_norm, m_norm, bool(r))
                        if key in seen:
                            continue
                        seen.add(key)
                        poly_variants.append(key)

    # Avoid accidental combinatorial explosions: only allow other extra-term scans when a single poly baseline is selected.
    if len(poly_variants) > 1 and any(
        bool(getattr(args, name))
        for name in (
            "include_corr",
            "include_lagcorr_fg",
            "include_laggap",
            "include_fg_logcurv",
            "include_fg_lowrank",
            "include_eor_mean",
            "include_eor_hf",
            "include_eor_lagshape",
            "include_eor_iso",
            "include_combos",
        )
    ):
        raise ValueError(
            "Multiple poly baselines were requested via --poly-*-list, but extra-term scans were also enabled. "
            "Run the poly baseline grid first (poly-only), then run extra-term scans with a single poly baseline."
        )

    basis_tag = {"power": "pow", "chebyshev": "cheb", "legendre": "leg", "dct": "dct", "bspline": "bsp"}
    legacy_single = (
        len(poly_variants) == 1
        and (not str(args.poly_degree_list).strip())
        and (not str(args.poly_basis_list).strip())
        and (not str(args.poly_x_mode_list).strip())
        and (not str(args.poly_model_list).strip())
        and (not str(args.poly_resid_enabled_list).strip())
    )
    for d, b, x, m, r in poly_variants:
        out.append(
            CandidateSpec(
                name=(
                    f"poly_w{_fmt_float_token(float(args.poly_weight))}_d{int(d)}_s{_fmt_float_token(float(args.poly_sigma))}"
                    if legacy_single
                    else (
                        f"poly_{basis_tag.get(b,b)}_{x}_{m}_r{1 if r else 0}"
                        f"_w{_fmt_float_token(float(args.poly_weight))}"
                        f"_d{int(d)}_s{_fmt_float_token(float(args.poly_sigma))}"
                    )
                ),
                extra_loss_terms=("poly_reparam",),
                optim_overrides={
                    "poly_degree": int(d),
                    "poly_sigma": float(args.poly_sigma),
                    "poly_basis": str(b),
                    "poly_x_mode": str(x),
                    "poly_model": str(m),
                    "poly_resid_enabled": bool(r),
                },
                weight_overrides={"poly_weight": float(args.poly_weight)},
                prior_overrides={},
            )
        )

    if bool(args.include_control):
        out.append(
            CandidateSpec(
                name="base",
                extra_loss_terms=(),
                optim_overrides={},
                weight_overrides={"poly_weight": 0.0},
                prior_overrides={},
            )
        )

    if bool(args.include_corr):
        for w in _parse_float_list(args.corr_weight_list):
            name = f"poly+corr_w{_fmt_float_token(w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "corr"),
                    optim_overrides={},
                    weight_overrides={"corr_weight": float(w)},
                    prior_overrides={
                        "corr_prior_mean": 0.0,
                        "corr_prior_sigma": float(args.corr_sigma),
                        "corr_prior_abs_threshold": float(args.corr_abs_threshold),
                        "corr_reduce": str(args.corr_reduce),
                        "corr_topk": None if int(args.corr_topk) <= 0 else int(args.corr_topk),
                        "corr_lse_alpha": float(args.corr_lse_alpha),
                        "corr_feature": str(args.corr_feature),
                        "corr_spatial_pool": int(args.corr_spatial_pool),
                    },
                )
            )

    if bool(args.include_lagcorr_fg):
        for w in _parse_float_list(args.lagcorr_fg_weight_list):
            for feat in _parse_csv_tokens(args.lagcorr_feature_list):
                feat_norm = str(feat).strip().lower()
                name = f"poly+lagfg_w{_fmt_float_token(w)}_{feat_norm}"
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("poly_reparam", "lagcorr"),
                        optim_overrides={},
                        weight_overrides={
                            "lagcorr_weight": float(w),
                            "lagcorr_fg_component_weight": 1.0,
                            "lagcorr_eor_component_weight": 0.0,
                            "lagcorr_gap_weight": 0.0,
                        },
                        prior_overrides={
                            "lagcorr_unit": str(args.lagcorr_unit),
                            "lagcorr_feature": feat_norm,
                            "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
                            "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                            "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
                            "lagcorr_random_seed": int(args.lagcorr_random_seed),
                            "lagcorr_rms_min": float(args.lagcorr_rms_min),
                            "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                            "lagcorr_lag_weights": 1.0,
                            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                            # fg_lagcorr_mean/sigma are wired in _build_config based on prior-source.
                        },
                    )
                )

    if bool(args.include_laggap):
        for w in _parse_float_list(args.laggap_weight_list):
            for margin in _parse_float_list(args.laggap_margin_list):
                for sig in _parse_float_list(args.laggap_sigma_list):
                    name = (
                        f"poly+laggap_w{_fmt_float_token(w)}_m{_fmt_float_token(margin)}_s{_fmt_float_token(sig)}"
                    )
                    out.append(
                        CandidateSpec(
                            name=name,
                            extra_loss_terms=("poly_reparam", "lagcorr"),
                            optim_overrides={},
                            weight_overrides={
                                "lagcorr_weight": float(w),
                                "lagcorr_fg_component_weight": 0.0,
                                "lagcorr_eor_component_weight": 0.0,
                                "lagcorr_gap_weight": 1.0,
                                "lagcorr_gap_margin": float(margin),
                                "lagcorr_gap_sigma": float(sig),
                                "lagcorr_gap_mode": "hinge",
                            },
                            prior_overrides={
                                "lagcorr_unit": str(args.lagcorr_unit),
                                "lagcorr_feature": "diff1",
                                "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
                                "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                                "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
                                "lagcorr_random_seed": int(args.lagcorr_random_seed),
                                "lagcorr_rms_min": float(args.lagcorr_rms_min),
                                "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                                "lagcorr_lag_weights": 1.0,
                                "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                                "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                            },
                        )
                    )

    if bool(args.include_fg_logcurv):
        sigmas = (
            _parse_float_list(args.fg_logcurv_sigma_list)
            if str(args.fg_logcurv_sigma_list).strip()
            else [float(args.fg_logcurv_sigma)]
        )
        for w in _parse_float_list(args.fg_logcurv_weight_list):
            for sig in sigmas:
                name = f"poly+logcurv_w{_fmt_float_token(w)}_s{_fmt_float_token(sig)}"
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("poly_reparam", "fg_logcurv"),
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

    if bool(args.include_fg_lowrank):
        ranks = (
            _parse_int_list(args.fg_lowrank_rank_list)
            if str(args.fg_lowrank_rank_list).strip()
            else [int(args.fg_lowrank_rank)]
        )
        tail_max_list = (
            _parse_float_list(args.fg_lowrank_tail_max_list)
            if str(args.fg_lowrank_tail_max_list).strip()
            else [float(args.fg_lowrank_tail_max)]
        )
        sigmas = (
            _parse_float_list(args.fg_lowrank_sigma_list)
            if str(args.fg_lowrank_sigma_list).strip()
            else [float(args.fg_lowrank_sigma)]
        )
        for w in _parse_float_list(args.fg_lowrank_weight_list):
            for r in ranks:
                for tail_max in tail_max_list:
                    for sig in sigmas:
                        name = (
                            f"poly+lowrank_w{_fmt_float_token(w)}_r{int(r)}"
                            f"_t{_fmt_float_token(tail_max)}_s{_fmt_float_token(sig)}"
                        )
                        out.append(
                            CandidateSpec(
                                name=name,
                                extra_loss_terms=("poly_reparam", "fg_lowrank"),
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

    if bool(args.include_eor_mean):
        for w in _parse_float_list(args.eor_mean_weight_list):
            name = f"poly+eormean_w{_fmt_float_token(w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "eor_mean"),
                    optim_overrides={},
                    weight_overrides={"eor_mean_weight": float(w)},
                    prior_overrides={},
                )
            )

    if bool(args.include_eor_hf):
        for w in _parse_float_list(args.eor_hf_weight_list):
            for rmax in _parse_float_list(args.eor_hf_rmax_list):
                name = f"poly+eorhf_w{_fmt_float_token(w)}_r{_fmt_float_token(rmax)}"
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("poly_reparam", "eor_hf"),
                        optim_overrides={},
                        weight_overrides={"eor_hf_weight": float(w)},
                        prior_overrides={
                            "eor_hf_percent": float(args.eor_hf_percent),
                            "eor_hf_r_max": float(rmax),
                        },
                    )
                )

    if bool(args.include_eor_lagshape):
        for w in _parse_float_list(args.eor_lagshape_weight_list):
            name = f"poly+lagshape_w{_fmt_float_token(w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "eor_lagshape"),
                    optim_overrides={},
                    weight_overrides={"eor_lagshape_weight": float(w)},
                    prior_overrides={
                        "lagcorr_unit": str(args.lagcorr_unit),
                        "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                        "eor_lagshape_feature": str(args.eor_lagshape_feature),
                        "eor_lagshape_spatial_pool": int(args.eor_lagshape_spatial_pool),
                        "eor_lagshape_far_min_lag": int(args.eor_lagshape_far_min_lag),
                        "eor_lagshape_tail_eps": float(args.eor_lagshape_tail_eps),
                        "eor_lagshape_mid_max_lag": int(args.eor_lagshape_mid_max_lag),
                        "eor_lagshape_rebound_eps_act": float(args.eor_lagshape_rebound_eps_act),
                        "eor_lagshape_rebound_delta_up": float(args.eor_lagshape_rebound_delta_up),
                    },
                )
            )

    if bool(args.include_eor_iso):
        for w in _parse_float_list(args.eor_iso_weight_list):
            name = f"poly+iso_w{_fmt_float_token(w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "eor_iso"),
                    optim_overrides={},
                    weight_overrides={"eor_iso_weight": float(w)},
                    prior_overrides={
                        "eor_iso_spatial_pool": int(args.eor_iso_spatial_pool),
                        "eor_iso_num_freq_samples": int(args.eor_iso_num_freq_samples),
                        "eor_iso_num_radial_bins": int(args.eor_iso_num_radial_bins),
                        "eor_iso_min_count": int(args.eor_iso_min_count),
                        "eor_iso_use_log_power": bool(args.eor_iso_use_log_power),
                    },
                )
            )

    if bool(args.include_combos):
        # A few 2-term combos on top of poly (kept small to avoid combinatorial explosion).
        for lag_w in [0.03, 0.1]:
            for mean_w in [0.03, 0.1]:
                name = f"poly+lagfg_w{_fmt_float_token(lag_w)}+mean_w{_fmt_float_token(mean_w)}"
                out.append(
                    CandidateSpec(
                        name=name,
                        extra_loss_terms=("poly_reparam", "lagcorr", "eor_mean"),
                        optim_overrides={},
                        weight_overrides={
                            "lagcorr_weight": float(lag_w),
                            "lagcorr_fg_component_weight": 1.0,
                            "lagcorr_eor_component_weight": 0.0,
                            "lagcorr_gap_weight": 0.0,
                            "eor_mean_weight": float(mean_w),
                        },
                        prior_overrides={
                            "lagcorr_unit": str(args.lagcorr_unit),
                            "lagcorr_feature": "diff1",
                            "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
                            "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                            "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
                            "lagcorr_random_seed": int(args.lagcorr_random_seed),
                            "lagcorr_rms_min": float(args.lagcorr_rms_min),
                            "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                            "lagcorr_lag_weights": 1.0,
                            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                        },
                    )
                )

        # A few physics-motivated multi-term combos with FG-only priors.
        # These are intended as "stack non-poly priors" long-run re-tests.
        logcurv_sigmas = (
            _parse_float_list(args.fg_logcurv_sigma_list)
            if str(args.fg_logcurv_sigma_list).strip()
            else [float(args.fg_logcurv_sigma)]
        )
        lowrank_ranks = (
            _parse_int_list(args.fg_lowrank_rank_list)
            if str(args.fg_lowrank_rank_list).strip()
            else [int(args.fg_lowrank_rank)]
        )
        lowrank_tail_max_list = (
            _parse_float_list(args.fg_lowrank_tail_max_list)
            if str(args.fg_lowrank_tail_max_list).strip()
            else [float(args.fg_lowrank_tail_max)]
        )
        lowrank_sigmas = (
            _parse_float_list(args.fg_lowrank_sigma_list)
            if str(args.fg_lowrank_sigma_list).strip()
            else [float(args.fg_lowrank_sigma)]
        )

        # Pick a single representative setting from each list (the first element) to keep combos small.
        logcurv_sig = float(logcurv_sigmas[0])
        lowrank_rank = int(lowrank_ranks[0])
        lowrank_tail = float(lowrank_tail_max_list[0])
        lowrank_sig = float(lowrank_sigmas[0])

        lag_w = 0.1
        logcurv_w = 0.3
        lowrank_w = 0.3
        for feat in ["diff1"]:
            feat_norm = str(feat).strip().lower()
            # poly + lagFG + logcurv
            name = f"poly+lagfg_w{_fmt_float_token(lag_w)}+logcurv_w{_fmt_float_token(logcurv_w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "lagcorr", "fg_logcurv"),
                    optim_overrides={},
                    weight_overrides={
                        "lagcorr_weight": float(lag_w),
                        "lagcorr_fg_component_weight": 1.0,
                        "lagcorr_eor_component_weight": 0.0,
                        "lagcorr_gap_weight": 0.0,
                        "fg_logcurv_weight": float(logcurv_w),
                    },
                    prior_overrides={
                        "lagcorr_unit": str(args.lagcorr_unit),
                        "lagcorr_feature": feat_norm,
                        "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
                        "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                        "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
                        "lagcorr_random_seed": int(args.lagcorr_random_seed),
                        "lagcorr_rms_min": float(args.lagcorr_rms_min),
                        "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                        "lagcorr_lag_weights": 1.0,
                        "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                        "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                        "fg_logcurv_mean": 0.0,
                        "fg_logcurv_sigma": float(logcurv_sig),
                        "fg_logcurv_eps": float(args.fg_logcurv_eps),
                        "fg_logcurv_softplus_scale": float(args.fg_logcurv_softplus_scale),
                    },
                )
            )

            # poly + lagFG + lowrank
            name = f"poly+lagfg_w{_fmt_float_token(lag_w)}+lowrank_w{_fmt_float_token(lowrank_w)}"
            out.append(
                CandidateSpec(
                    name=name,
                    extra_loss_terms=("poly_reparam", "lagcorr", "fg_lowrank"),
                    optim_overrides={},
                    weight_overrides={
                        "lagcorr_weight": float(lag_w),
                        "lagcorr_fg_component_weight": 1.0,
                        "lagcorr_eor_component_weight": 0.0,
                        "lagcorr_gap_weight": 0.0,
                        "fg_lowrank_weight": float(lowrank_w),
                    },
                    prior_overrides={
                        "lagcorr_unit": str(args.lagcorr_unit),
                        "lagcorr_feature": feat_norm,
                        "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
                        "lagcorr_max_pairs": None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                        "lagcorr_pair_sampling": str(args.lagcorr_pair_sampling),
                        "lagcorr_random_seed": int(args.lagcorr_random_seed),
                        "lagcorr_rms_min": float(args.lagcorr_rms_min),
                        "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
                        "lagcorr_lag_weights": 1.0,
                        "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
                        "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
                        "fg_lowrank_rank": int(lowrank_rank),
                        "fg_lowrank_num_samples": int(args.fg_lowrank_num_samples),
                        "fg_lowrank_spatial_pool": int(args.fg_lowrank_spatial_pool),
                        "fg_lowrank_normalize": str(args.fg_lowrank_normalize),
                        "fg_lowrank_tail_max": float(lowrank_tail),
                        "fg_lowrank_sigma": float(lowrank_sig),
                        "fg_lowrank_sample_mode": str(args.fg_lowrank_sample_mode),
                        "fg_lowrank_random_seed": int(args.fg_lowrank_random_seed),
                        "fg_lowrank_eps": float(args.fg_lowrank_eps),
                    },
                )
            )

        # poly + logcurv + lowrank (no lagcorr)
        name = f"poly+logcurv_w{_fmt_float_token(logcurv_w)}+lowrank_w{_fmt_float_token(lowrank_w)}"
        out.append(
            CandidateSpec(
                name=name,
                extra_loss_terms=("poly_reparam", "fg_logcurv", "fg_lowrank"),
                optim_overrides={},
                weight_overrides={
                    "fg_logcurv_weight": float(logcurv_w),
                    "fg_lowrank_weight": float(lowrank_w),
                },
                prior_overrides={
                    "fg_logcurv_mean": 0.0,
                    "fg_logcurv_sigma": float(logcurv_sig),
                    "fg_logcurv_eps": float(args.fg_logcurv_eps),
                    "fg_logcurv_softplus_scale": float(args.fg_logcurv_softplus_scale),
                    "fg_lowrank_rank": int(lowrank_rank),
                    "fg_lowrank_num_samples": int(args.fg_lowrank_num_samples),
                    "fg_lowrank_spatial_pool": int(args.fg_lowrank_spatial_pool),
                    "fg_lowrank_normalize": str(args.fg_lowrank_normalize),
                    "fg_lowrank_tail_max": float(lowrank_tail),
                    "fg_lowrank_sigma": float(lowrank_sig),
                    "fg_lowrank_sample_mode": str(args.fg_lowrank_sample_mode),
                    "fg_lowrank_random_seed": int(args.fg_lowrank_random_seed),
                    "fg_lowrank_eps": float(args.fg_lowrank_eps),
                },
            )
        )

    # De-dup by name.
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

    if true_eor.shape != eor_est.shape:
        row["status"] = "failed"
        row["note"] = f"true_shape_mismatch true={true_eor.shape} est={eor_est.shape}"
        return row

    eor_corr = _frequency_correlations(eor_est, true_eor)
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
    inj_fg_eor_corr = _frequency_correlations(true_fg, true_eor)
    inj_stats = _summarize_corr_stats(inj_fg_eor_corr)
    row["inj_fg_eor_corr_abs_mean"] = inj_stats["abs_mean"]

    row.update(_read_eor_window_metrics(run_dir / "powerspec"))
    row.update(_parse_convergence_from_log(run_dir / "run.log"))
    return row


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    code_dir = (
        args.code_dir.resolve()
        if args.code_dir
        else (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    )
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"poly_combo_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir, cube12_start_mhz=float(args.freq_start_mhz))
    enabled = [t.strip() for t in str(args.datasets).split(",") if t.strip()]
    datasets = [d for d in datasets_all if d.name in set(enabled)]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")
    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

    # Default include set: keep it explicit so this script does not accidentally explode.
    include_any = any(
        bool(getattr(args, k))
        for k in [
            "include_control",
            "include_corr",
            "include_lagcorr_fg",
            "include_laggap",
            "include_fg_logcurv",
            "include_fg_lowrank",
            "include_eor_mean",
            "include_eor_hf",
            "include_eor_lagshape",
            "include_eor_iso",
            "include_combos",
        ]
    )
    if not include_any:
        args.include_lagcorr_fg = True
        args.include_laggap = True
        args.include_eor_mean = True
        args.include_eor_hf = True
        args.include_fg_logcurv = True
        args.include_fg_lowrank = True

    candidates = generate_candidates(args)
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        known = {c.name for c in candidates}
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

    # Prepare dataset caches: cut indices + true cubes.
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

    # Optional cache for oracle FG lagcorr priors (per dataset/feature/pool).
    fg_lagcorr_cache: Dict[Tuple[str, str, int], Tuple[List[float], List[float]]] = {}
    if bool(args.include_lagcorr_fg) and str(args.lagcorr_fg_prior_source).strip().lower() == "truth":
        lag_channels: List[int] = []
        if str(args.lagcorr_unit).strip().lower() == "chan":
            lag_channels = [int(round(float(x))) for x in LAG_INTERVALS_MHZ]
        else:
            for mhz in LAG_INTERVALS_MHZ:
                lag_channels.append(max(1, int(round(float(mhz) / float(args.freq_delta_mhz)))))
        for ds in datasets:
            cache = ds_cache[ds.name]
            true_fg = cache["true_fg"]
            assert isinstance(true_fg, np.ndarray)
            for feat in _parse_csv_tokens(args.lagcorr_feature_list):
                feat_norm = str(feat).strip().lower()
                pool = int(args.lagcorr_spatial_pool)
                key = (ds.name, feat_norm, pool)
                if key in fg_lagcorr_cache:
                    continue
                mu, sig = _lagcorr_oracle_from_true_fg(
                    true_fg,
                    lag_channels=lag_channels,
                    feature=feat_norm,
                    spatial_pool=pool,
                    max_pairs=None if int(args.lagcorr_max_pairs) <= 0 else int(args.lagcorr_max_pairs),
                    pair_sampling=str(args.lagcorr_pair_sampling),
                    seed=int(args.lagcorr_random_seed),
                    rms_min=float(args.lagcorr_rms_min),
                    sigma_floor=float(args.lagcorr_fg_sigma_floor),
                )
                fg_lagcorr_cache[key] = (mu, sig)

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": [{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(d).items()} for d in datasets],
        "candidates": [asdict(c) for c in candidates],
        "baseline_fixed": {
            "beta": float(args.base_beta),
            "gamma": float(args.base_gamma),
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_amp_threshold": float(args.base_eor_amp_threshold),
            "eor_amp_prior_mode": str(args.base_eor_amp_prior_mode),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "data_error": float(args.data_error),
            "optimizer_name": str(args.optimizer_name),
            "lr": float(args.lr),
            "lr_scheduler": str(args.lr_scheduler),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "power_config": str(args.power_config),
            "exclude_from_ranking": str(args.exclude_from_ranking),
            "poly_weight": float(args.poly_weight),
            "poly_degree": int(args.poly_degree),
            "poly_sigma": float(args.poly_sigma),
        },
        "fg_lagcorr_oracle_cache_keys": [list(k) for k in sorted(fg_lagcorr_cache.keys())],
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
                    fg_lagcorr_cache=fg_lagcorr_cache,
                )
                with job.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(cfg, handle, indent=2)
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                import subprocess  # local import to keep module import minimal

                proc = subprocess.Popen(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
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
                    f"converged={row.get('converged')} eor_score={row.get('eor_corr_score')} "
                    f"ps_mad={row.get('ps2d_win_log10_mad')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "poly_combo_scan_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows, exclude_datasets=_parse_csv_tokens(args.exclude_from_ranking))
    rank_csv = output_dir / "poly_combo_scan_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "poly_combo_scan_summary.md", ranked, manifest["baseline_fixed"])
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
