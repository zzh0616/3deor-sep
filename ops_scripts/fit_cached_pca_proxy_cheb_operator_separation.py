#!/usr/bin/env python3
"""Fit a Chebyshev foreground model through a cached PCA response operator.

This is the first practical optimizer-loop version of the response-bank route.
It fits foreground Chebyshev coefficient maps from dirty total data while the
forward model is a tiled stride-2-proxy PCA operator.  Dirty EoR truth is used
only for post-fit recovery metrics, not for the optimization loss.

The implemented operator intentionally supports only pure PCA interpolation.
The older ``train_hybrid`` evaluator reads exact response FITS files for train
support points, which is useful for closure diagnostics but not viable inside an
optimizer inner loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from torch.utils.checkpoint import checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate_edge_tiled_hybrid_fullsky_subtraction import (  # noqa: E402
    _make_tiles,
    _pca_basis,
    _rbf_coefficients_many,
    _select_train_rows,
)
from evaluate_edge_tiled_hybrid_stride2_proxy_subtraction import (  # noqa: E402
    _assign_grid_indices,
    _select_dense_rows,
)
from evaluate_fullsky_response_interpolation import (  # noqa: E402
    _central_crop,
    _fmt,
    _k_to_jy_per_pixel,
    _load_fits_2d,
)
from evaluate_fullsky_stride2_integer_shift import _axis_linear_contribs  # noqa: E402
from evaluate_sparse_response_interpolation_holdout import _load_grid_csv, _shift_zero  # noqa: E402


SUSPICIOUS_TEMPLATE_BASIS_TOKENS = (
    "simulation",
    "simulator",
    "skymap/osm_prepare",
    "osm_prepare",
    "/osm/",
    ".osm",
    "fg_cube",
    "eor_cube",
    "truth",
    "oracle",
    "synthetic",
    "perturbed_inits",
    "cube2_fullsky",
    "cheb_foreground",
)


@dataclass
class PcaTileCache:
    freq_index: int
    freq_mhz: float
    tile_name: str
    model_size: int
    eval_size: int
    crop_start: int
    src_y: np.ndarray
    src_x: np.ndarray
    flat_idx: np.ndarray
    mean_weight: np.ndarray
    coeff_weight: np.ndarray
    kernel_stack: np.ndarray


@dataclass(frozen=True)
class C0CorrectionStage:
    index: int
    start_iter: int
    end_iter: int
    degrees: Tuple[int, ...]
    blocks: Tuple[int, ...]
    label: str


@dataclass(frozen=True)
class ScalarSchedulePoint:
    start_iter: int
    value: float


def _crop2d(
    array: np.ndarray,
    *,
    size: int,
    center_x: int | None,
    center_y: int | None,
) -> np.ndarray:
    if int(size) <= 0:
        return array
    height, width = array.shape
    size = int(size)
    center_x = int(center_x) if center_x is not None else width // 2
    center_y = int(center_y) if center_y is not None else height // 2
    x_start = center_x - size // 2
    y_start = center_y - size // 2
    if (
        x_start < 0
        or y_start < 0
        or x_start + size > width
        or y_start + size > height
    ):
        raise ValueError(
            f"Invalid crop size={size}, center=({center_x},{center_y}) "
            f"for image shape {(height, width)}"
        )
    return np.asarray(array[y_start : y_start + size, x_start : x_start + size])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--freqs-mhz", required=True)
    ap.add_argument("--freq-index-offset", type=int, default=0)
    ap.add_argument("--dense-grid-csv-pattern", required=True)
    ap.add_argument("--train-grid-csv-pattern", required=True)
    ap.add_argument("--train-response-pattern", required=True)
    ap.add_argument("--truth-fg-pattern", required=True)
    ap.add_argument("--truth-eor-pattern", required=True)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--response-crop-size", type=int, default=512)
    ap.add_argument("--eval-crop-size", type=int, default=256)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--train-halo-px", type=int, default=16)
    ap.add_argument("--model-margin", type=int, default=-1)
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--rbf-scale-px", type=float, default=32.0)
    ap.add_argument("--cheb-degree", type=int, default=2)
    ap.add_argument("--pixel-arcsec", type=float, default=32.0)
    ap.add_argument("--num-iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument(
        "--lr-schedule",
        default="",
        help=(
            "Optional piecewise-constant LR schedule, e.g. '0:0.5,30:0.2,60:0.05'. "
            "Empty keeps --lr fixed."
        ),
    )
    ap.add_argument(
        "--optimizer-name",
        choices=("adam", "adamw", "nadam", "sgd", "rmsprop", "lbfgs"),
        default="adam",
        help="Optimizer used for the Cheb coefficient fit. Defaults to the historical Adam path.",
    )
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--nesterov", action="store_true")
    ap.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=4,
        help="Maximum inner iterations per outer step for --optimizer-name lbfgs.",
    )
    ap.add_argument(
        "--lbfgs-history-size",
        type=int,
        default=20,
        help="History size for --optimizer-name lbfgs.",
    )
    ap.add_argument(
        "--lbfgs-line-search",
        choices=("none", "strong_wolfe"),
        default="strong_wolfe",
        help="Line search used by --optimizer-name lbfgs.",
    )
    ap.add_argument(
        "--lbfgs-tolerance-grad",
        type=float,
        default=1e-7,
        help="Gradient tolerance for --optimizer-name lbfgs.",
    )
    ap.add_argument(
        "--lbfgs-tolerance-change",
        type=float,
        default=1e-9,
        help="Parameter/loss change tolerance for --optimizer-name lbfgs.",
    )
    ap.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Clip total gradient norm before optimizer.step(). 0 disables clipping.",
    )
    ap.add_argument(
        "--step-control-mode",
        choices=("none", "train_loss_backtracking"),
        default="none",
        help=(
            "Optional deployable step acceptance control. train_loss_backtracking "
            "uses the training objective, not EoR truth, to backtrack or reject "
            "optimizer-proposed steps that overshoot."
        ),
    )
    ap.add_argument(
        "--step-control-max-backtracks",
        type=int,
        default=6,
        help="Maximum shrink attempts for --step-control-mode train_loss_backtracking.",
    )
    ap.add_argument(
        "--step-control-shrink",
        type=float,
        default=0.5,
        help="Multiplicative shrink factor for step-control backtracking.",
    )
    ap.add_argument(
        "--step-control-rel-tol",
        type=float,
        default=1e-8,
        help="Relative objective increase tolerated by step-control acceptance.",
    )
    ap.add_argument(
        "--step-control-abs-tol",
        type=float,
        default=0.0,
        help="Absolute objective increase tolerated by step-control acceptance.",
    )
    ap.add_argument(
        "--step-control-fallback-gradient",
        action="store_true",
        help=(
            "If optimizer-proposal backtracking fails, try a fresh RMS-normalized "
            "negative-gradient step from the pre-step parameters and clear "
            "optimizer momentum when that fallback is accepted."
        ),
    )
    ap.add_argument(
        "--step-control-gradient-eps",
        type=float,
        default=1e-30,
        help="Gradient RMS floor for --step-control-fallback-gradient.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--progress-every-tile", type=int, default=1)
    ap.add_argument("--keep-kernel-fft-on-device", action="store_true")
    ap.add_argument(
        "--checkpoint-tiles",
        action="store_true",
        help="Recompute tile forwards during backward to reduce rank64 optimizer memory.",
    )
    ap.add_argument("--data-weight", type=float, default=1.0)
    ap.add_argument(
        "--optimization-loss-scale",
        type=float,
        default=1.0,
        help=(
            "Positive scalar multiplier applied only to the loss used for "
            "backpropagation/optimizer steps. Logged losses and selection "
            "metrics remain on the physical objective scale."
        ),
    )
    ap.add_argument(
        "--data-loss-mode",
        choices=("mse", "upper_hinge", "target_log", "window_hinge"),
        default="mse",
        help=(
            "mse preserves the historical squared residual/data-error loss. "
            "upper_hinge only penalizes residuals above the data-error scale. "
            "target_log penalizes deviations above or below the data-error scale. "
            "window_hinge only penalizes residual RMS ratios outside "
            "[--data-window-lower, --data-window-upper]."
        ),
    )
    ap.add_argument(
        "--data-window-lower",
        type=float,
        default=0.0,
        help="Lower residual/data-error RMS ratio for --data-loss-mode window_hinge.",
    )
    ap.add_argument(
        "--data-window-lower-schedule",
        default="",
        help="Optional piecewise schedule for --data-window-lower, e.g. '0:1.02,40:1.003'.",
    )
    ap.add_argument(
        "--data-window-upper",
        type=float,
        default=1.0,
        help=(
            "Upper residual/data-error RMS ratio for --data-loss-mode "
            "upper_hinge/window_hinge."
        ),
    )
    ap.add_argument(
        "--data-window-upper-schedule",
        default="",
        help="Optional piecewise schedule for --data-window-upper, e.g. '0:1.04,40:1.007'.",
    )
    ap.add_argument(
        "--data-error-rms",
        type=float,
        default=0.0,
        help=(
            "RMS scale used to normalize the data misfit. "
            "0 preserves the historical RMS(dirty_total) normalization."
        ),
    )
    ap.add_argument("--residual-cheb-weight", type=float, default=0.2)
    ap.add_argument(
        "--residual-cheb-weight-schedule",
        default="",
        help="Optional piecewise schedule for --residual-cheb-weight.",
    )
    ap.add_argument(
        "--residual-rms-target",
        type=float,
        default=0.0,
        help="Optional expected dirty-EoR residual RMS. 0 disables the amplitude prior.",
    )
    ap.add_argument(
        "--residual-rms-weight",
        type=float,
        default=0.0,
        help="Weight for the log-RMS residual amplitude prior.",
    )
    ap.add_argument(
        "--residual-cheb-norm-mode",
        choices=("dirty_total_rms", "data_error_rms"),
        default="dirty_total_rms",
        help=(
            "Normalization for the residual Cheb/smooth leakage loss. "
            "dirty_total_rms preserves the historical weak scaling; "
            "data_error_rms measures smooth residual leakage on the explicit "
            "EoR/error scale and is intended for active recovery tests."
        ),
    )
    ap.add_argument(
        "--residual-spatial-lowpass-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a non-oracle penalty on low-spatial-frequency residual "
            "DCT power. 0 disables the term."
        ),
    )
    ap.add_argument(
        "--residual-spatial-lowpass-dct-size",
        type=int,
        default=8,
        help="Number of low-frequency DCT modes per axis used by the residual spatial-lowpass loss.",
    )
    ap.add_argument(
        "--residual-spatial-lowpass-norm-mode",
        choices=("dirty_total_rms", "data_error_rms"),
        default="data_error_rms",
        help="RMS scale used to normalize the residual spatial-lowpass loss.",
    )
    ap.add_argument(
        "--residual-spatial-lowpass-skip-dc",
        action="store_true",
        help="Exclude the DC spatial mode from the residual spatial-lowpass loss.",
    )
    ap.add_argument(
        "--residual-power-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a radial spatial-power prior on the residual/EoR estimate. "
            "0 disables the term."
        ),
    )
    ap.add_argument(
        "--residual-power-target-source",
        choices=("none", "dirty_eor", "npz", "operator_white"),
        default="none",
        help=(
            "Target source for --residual-power-weight. dirty_eor is an oracle "
            "diagnostic; npz is intended for theory/non-truth targets; "
            "operator_white builds a non-truth target by passing random white "
            "sky maps through the cached forward operator."
        ),
    )
    ap.add_argument(
        "--residual-power-target-npz",
        type=Path,
        default=None,
        help=(
            "NPZ containing residual radial-power target. Expected key 'power' "
            "with shape (n_freq, n_bins) or (n_bins,). Optional key 'bin_edges'."
        ),
    )
    ap.add_argument(
        "--residual-power-loss-mode",
        choices=("mse", "log_upper_hinge", "log_window_hinge"),
        default="mse",
        help=(
            "mse attracts residual radial power to the target. log_upper_hinge "
            "only penalizes residual power above target*exp(tolerance). "
            "log_window_hinge penalizes deviations outside a symmetric log "
            "window. Hinge modes are weak envelope priors, not exact EoR "
            "templates."
        ),
    )
    ap.add_argument(
        "--residual-power-log-tolerance",
        type=float,
        default=0.0,
        help="Allowed absolute log-power tolerance for residual-power hinge modes.",
    )
    ap.add_argument("--residual-power-num-bins", type=int, default=12)
    ap.add_argument("--residual-power-kmin", type=float, default=0.0)
    ap.add_argument("--residual-power-kmax", type=float, default=0.5)
    ap.add_argument(
        "--residual-power-log-eps",
        type=float,
        default=1e-30,
        help="Power floor used inside the log-ratio radial-power loss.",
    )
    ap.add_argument(
        "--residual-power-operator-white-samples",
        type=int,
        default=8,
        help="Number of random white-sky samples for --residual-power-target-source operator_white.",
    )
    ap.add_argument(
        "--residual-power-operator-white-seed",
        type=int,
        default=12345,
        help="Random seed for --residual-power-target-source operator_white.",
    )
    ap.add_argument(
        "--residual-power-isotropy-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a non-oracle angular-isotropy loss on residual log power. "
            "The loss removes each radial bin mean, so it does not prescribe a "
            "radial power profile."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a frequency-correlation prior on the residual/EoR estimate. "
            "0 disables the term."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-target-source",
        choices=("none", "dirty_eor", "npz", "operator_white"),
        default="none",
        help=(
            "Target source for --residual-freq-corr-weight. dirty_eor is an oracle "
            "diagnostic; npz is intended for theory/non-truth targets; "
            "operator_white builds a non-truth target by passing random white "
            "sky maps through the cached forward operator."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-target-npz",
        type=Path,
        default=None,
        help=(
            "NPZ containing residual frequency-correlation target. Expected key "
            "'corr', 'freq_corr', or 'frequency_corr' with shape (n_freq, n_freq)."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-loss-mode",
        choices=("mse", "upper_hinge"),
        default="mse",
        help=(
            "mse attracts the residual frequency-correlation matrix to the target. "
            "upper_hinge only penalizes correlation-matrix MSE above "
            "--residual-freq-corr-hinge-target."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-hinge-target",
        type=float,
        default=0.0,
        help=(
            "Allowed residual frequency-correlation MSE for "
            "--residual-freq-corr-loss-mode upper_hinge."
        ),
    )
    ap.add_argument(
        "--residual-freq-corr-eps",
        type=float,
        default=1e-30,
        help="Variance floor used when normalizing the residual frequency-correlation matrix.",
    )
    ap.add_argument(
        "--residual-freq-corr-operator-white-samples",
        type=int,
        default=8,
        help="Number of random white-sky samples for --residual-freq-corr-target-source operator_white.",
    )
    ap.add_argument(
        "--residual-freq-corr-operator-white-seed",
        type=int,
        default=22345,
        help="Random seed for --residual-freq-corr-target-source operator_white.",
    )
    ap.add_argument("--fg-l2-weight", type=float, default=1e-6)
    ap.add_argument("--fg-positivity-weight", type=float, default=1e-3)
    ap.add_argument("--coeff-tv-weight", type=float, default=1e-7)
    ap.add_argument(
        "--prior-cheb-coeffs",
        type=Path,
        default=None,
        help="Optional external foreground coefficient prior, stored as degree+1 x image x image FITS.",
    )
    ap.add_argument(
        "--coeff-prior-weight",
        type=float,
        default=0.0,
        help="Weight for the external coefficient prior loss.",
    )
    ap.add_argument(
        "--coeff-prior-weight-schedule",
        default="",
        help="Optional piecewise schedule for --coeff-prior-weight.",
    )
    ap.add_argument(
        "--coeff-prior-loss-mode",
        choices=("mse", "global_rms_hinge"),
        default="mse",
        help=(
            "mse treats the external coefficient prior as a Gaussian attractor. "
            "global_rms_hinge treats it as a broad uncertainty tube and only "
            "penalizes per-degree normalized RMS deviations above "
            "--coeff-prior-hinge-sigma."
        ),
    )
    ap.add_argument(
        "--coeff-prior-hinge-sigma",
        type=float,
        default=1.0,
        help=(
            "Per-degree normalized RMS threshold for "
            "--coeff-prior-loss-mode global_rms_hinge."
        ),
    )
    ap.add_argument(
        "--coeff-prior-hinge-sigma-schedule",
        default="",
        help="Optional piecewise schedule for --coeff-prior-hinge-sigma.",
    )
    ap.add_argument(
        "--coeff-prior-scales-k",
        default="",
        help=(
            "Absolute per-degree K scales for the external coefficient prior. "
            "Empty uses the prior RMS per degree, falling back to --coeff-scale-k."
        ),
    )
    ap.add_argument(
        "--coeff-prior-relative-scales",
        default="",
        help="Optional scalar or per-degree multiplier for --coeff-prior-scales-k/default scales.",
    )
    ap.add_argument(
        "--coeff-delta-trust-weight",
        type=float,
        default=0.0,
        help="Weight for a per-degree trust-region penalty on coefficient changes from the initializer.",
    )
    ap.add_argument(
        "--coeff-delta-trust-scales-k",
        default="",
        help=(
            "Absolute per-degree K scales for the coefficient-delta trust penalty. "
            "Empty uses the initializer RMS per degree, falling back to --coeff-scale-k."
        ),
    )
    ap.add_argument(
        "--coeff-delta-trust-relative-scales",
        default="",
        help="Optional scalar or per-degree multiplier for --coeff-delta-trust-scales-k/default scales.",
    )
    ap.add_argument("--coeff-scale-k", type=float, default=1000.0)
    ap.add_argument(
        "--param-mode",
        choices=("direct", "scaled_delta"),
        default="direct",
        help=(
            "direct optimizes coefficient values directly. scaled_delta keeps a fixed base "
            "coefficient cube and optimizes base + scale[degree] * theta."
        ),
    )
    ap.add_argument(
        "--optimize-degrees",
        default="",
        help="Comma-separated Cheb degree indices to update. Empty means all degrees; none/off freezes all full-pixel degrees.",
    )
    ap.add_argument(
        "--degree-param-scales-k",
        default="",
        help=(
            "Absolute per-degree K scales for scaled_delta. Empty uses the base coefficient "
            "RMS per degree, falling back to --coeff-scale-k for zero-RMS degrees."
        ),
    )
    ap.add_argument(
        "--degree-param-relative-scales",
        default="",
        help=(
            "Optional scalar or per-degree multiplier applied to --degree-param-scales-k "
            "or the default base RMS scales."
        ),
    )
    ap.add_argument(
        "--degree-param-max-abs",
        default="",
        help=(
            "Optional scalar or per-degree hard clamp for scaled_delta theta values. "
            "0 disables clamping for that degree."
        ),
    )
    ap.add_argument(
        "--c0-correction-mode",
        choices=("none", "global", "block", "multiblock", "dct"),
        default="none",
        help=(
            "Optional low-dimensional additive correction for selected Cheb degrees. "
            "Use with --optimize-degrees none/1,2 to avoid full-pixel C0 freedom."
        ),
    )
    ap.add_argument(
        "--c0-correction-degrees",
        default="0",
        help=(
            "Comma-separated Cheb degree indices receiving the low-dimensional "
            "correction. Default 0 preserves the historical C0-only behavior."
        ),
    )
    ap.add_argument(
        "--c0-correction-blocks",
        type=int,
        default=4,
        help="Number of blocks per axis for --c0-correction-mode block.",
    )
    ap.add_argument(
        "--c0-correction-block-list",
        default="",
        help=(
            "Comma-separated block counts for --c0-correction-mode multiblock. "
            "Empty defaults to --c0-correction-blocks."
        ),
    )
    ap.add_argument(
        "--c0-correction-dct-size",
        type=int,
        default=8,
        help="Number of low-frequency DCT modes per axis for --c0-correction-mode dct.",
    )
    ap.add_argument(
        "--c0-correction-scale-k",
        type=float,
        default=0.0,
        help="Absolute K scale for a unit C0 correction parameter. 0 uses C0 RMS times --c0-correction-relative-scale.",
    )
    ap.add_argument(
        "--c0-correction-scales-k",
        default="",
        help=(
            "Optional absolute K scale list for selected --c0-correction-degrees. "
            "Accepts one scalar or one value per selected degree. Overrides "
            "--c0-correction-scale-k when non-empty."
        ),
    )
    ap.add_argument(
        "--c0-correction-relative-scale",
        type=float,
        default=1e-6,
        help="Relative scale used when --c0-correction-scale-k is 0.",
    )
    ap.add_argument(
        "--c0-correction-param-max-abs",
        type=float,
        default=0.0,
        help="Hard absolute clamp for the low-dimensional C0 correction parameters. 0 disables clamping.",
    )
    ap.add_argument(
        "--c0-correction-stage-spec",
        default="",
        help=(
            "Optional staged activation schedule for multiblock correction params. "
            "Format: 'off@10,0:4@20,0:4+8@20,0+1:4+8+16@40'. "
            "The left side lists Cheb degrees, the right side lists active block "
            "sizes, and @N is the number of optimizer steps for that stage. "
            "'off@N' freezes all multiblock correction parameters for N steps. "
            "After the final listed stage, the final active set is kept."
        ),
    )
    ap.add_argument(
        "--c0-correction-stage-reset-optimizer",
        action="store_true",
        help=(
            "When a new C0 correction stage activates, clear the optimizer state "
            "for the C0 correction parameter so Adam/RMSProp momentum from the "
            "previous active subspace does not dominate the newly released block."
        ),
    )
    ap.add_argument(
        "--template-gain-mode",
        choices=("none", "global", "per_degree"),
        default="none",
        help="Optional low-dimensional multiplicative gain on the loaded/init Cheb template.",
    )
    ap.add_argument(
        "--template-gain-init",
        default="",
        help="Initial template gain. Scalar or comma-separated per degree. Empty defaults to 1.",
    )
    ap.add_argument(
        "--template-gain-param-scale",
        type=float,
        default=1.0,
        help="Gain change represented by a unit trainable template-gain parameter.",
    )
    ap.add_argument(
        "--template-gain-param-max-abs",
        type=float,
        default=0.0,
        help="Optional clamp on template-gain trainable parameters. 0 disables.",
    )
    ap.add_argument(
        "--template-basis-cheb-coeffs-list",
        default="",
        help=(
            "Optional comma/semicolon/newline separated list of observed Cheb coefficient FITS "
            "basis templates. These are linearly combined with low-dimensional gains."
        ),
    )
    ap.add_argument(
        "--template-basis-gain-mode",
        choices=("none", "global", "per_basis", "per_basis_degree"),
        default="none",
        help="Low-dimensional gain parameterization for --template-basis-cheb-coeffs-list.",
    )
    ap.add_argument(
        "--template-basis-gain-init",
        default="",
        help=(
            "Initial basis gains. Empty/scalar broadcasts. per_basis accepts one value per basis; "
            "per_basis_degree accepts one value per basis or basis*degree values."
        ),
    )
    ap.add_argument(
        "--template-basis-param-scale",
        type=float,
        default=1.0,
        help="Basis-gain change represented by a unit trainable template-basis parameter.",
    )
    ap.add_argument(
        "--template-basis-param-max-abs",
        type=float,
        default=0.0,
        help="Optional clamp on template-basis trainable parameters. 0 disables.",
    )
    ap.add_argument(
        "--allow-suspicious-template-basis",
        action="store_true",
        help="Allow template-basis paths that look simulated/oracle. Use only for labelled diagnostics.",
    )
    ap.add_argument(
        "--max-tiles-per-freq",
        type=int,
        default=0,
        help="Debug limit. 0 means use all tiles.",
    )
    ap.add_argument(
        "--init-cheb-coeffs",
        type=Path,
        default=None,
        help="Optional oracle/debug initializer, stored as degree+1 x image x image FITS.",
    )
    ap.add_argument(
        "--init-cheb-coeffs-list",
        default="",
        help=(
            "Optional comma/semicolon/newline separated initializer list. "
            "Use NONE or ZERO for a zero init entry. When set, all entries share one cached operator."
        ),
    )
    ap.add_argument(
        "--run-labels",
        default="",
        help="Optional comma/semicolon/newline separated labels for --init-cheb-coeffs-list entries.",
    )
    ap.add_argument(
        "--tile-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for persistent per-frequency/tile PCA cache reuse.",
    )
    ap.add_argument(
        "--tile-cache-refresh",
        action="store_true",
        help="Ignore existing tile cache files and overwrite them.",
    )
    ap.add_argument(
        "--tile-cache-meta-path-rewrite",
        default="",
        help=(
            "Optional path rewrite list used only when comparing tile-cache "
            "metadata, e.g. '/data1/zhenghao/fg_rmw=/data/zhenghao/fg_rmw'. "
            "This allows copied caches to be reused after moving the project "
            "root while still checking all non-path metadata."
        ),
    )
    ap.add_argument(
        "--preload-train-responses",
        action="store_true",
        help=(
            "Preload all training response crops for each frequency before tiled PCA cache construction. "
            "This trades memory for much less repeated FITS I/O."
        ),
    )
    ap.add_argument(
        "--cache-build-only",
        action="store_true",
        help="Build or validate the persistent tile cache, then exit before constructing the optimizer.",
    )
    ap.add_argument(
        "--save-selection-mode",
        choices=(
            "truth_metric",
            "train_loss",
            "last",
            "data_prior_boundary",
            "data_window_min_prior",
            "data_window_max_data",
            "data_window_max_data_skip0",
            "data_window_latest",
            "data_window_earliest",
            "data_window_earliest_skip0",
        ),
        default="truth_metric",
        help=(
            "State written to cheb_coeffs_best_k.fits. truth_metric preserves the "
            "historical diagnostic behavior and uses dirty EoR truth. train_loss "
            "last, data_prior_boundary, data_window_min_prior, "
            "data_window_max_data, data_window_max_data_skip0, and "
            "data_window_latest, data_window_earliest, and "
            "data_window_earliest_skip0 are deployable because they do not use EoR "
            "truth. The skip0 variant applies the same max-data rule but never "
            "saves the initial state if later states exist. data_window_latest "
            "prefers the latest iteration inside the configured data window. "
            "data_window_earliest prefers the earliest iteration inside the "
            "configured data window to reduce late EoR absorption."
        ),
    )
    ap.add_argument(
        "--selection-prior-degree",
        type=int,
        default=0,
        help="Cheb degree used by --save-selection-mode data_prior_boundary.",
    )
    ap.add_argument(
        "--selection-prior-target-rms",
        type=float,
        default=1.0,
        help=(
            "Target normalized prior RMS for --save-selection-mode "
            "data_prior_boundary."
        ),
    )
    ap.add_argument(
        "--selection-prior-weight",
        type=float,
        default=1e-4,
        help=(
            "Weight for |prior_rms[degree] - target| in "
            "--save-selection-mode data_prior_boundary."
        ),
    )
    ap.add_argument(
        "--selection-data-window-lower",
        type=float,
        default=0.0,
        help=(
            "Lower residual/data-error RMS ratio for data-window based "
            "--save-selection-mode values."
        ),
    )
    ap.add_argument(
        "--selection-data-window-upper",
        type=float,
        default=1.0,
        help=(
            "Upper residual/data-error RMS ratio for data-window based "
            "--save-selection-mode values."
        ),
    )
    ap.add_argument(
        "--selection-data-window-penalty",
        type=float,
        default=1000.0,
        help=(
            "Penalty multiplier for data-window violation in "
            "data-window based --save-selection-mode values."
        ),
    )
    ap.add_argument("--save-products", action="store_true")
    return ap.parse_args()


def _parse_floats(spec: str) -> List[float]:
    return [float(v.strip()) for v in str(spec).replace(" ", ",").split(",") if v.strip()]


def _parse_ints(spec: str) -> List[int]:
    if str(spec or "").strip().lower() in {"none", "off", "false", "freeze", "frozen"}:
        return []
    return [int(v.strip()) for v in str(spec).replace(" ", ",").split(",") if v.strip()]


def _parse_degree_float_values(spec: str, n_degree: int, default: Sequence[float]) -> List[float]:
    values = _parse_floats(spec)
    if not values:
        return [float(v) for v in default]
    if len(values) == 1:
        return [float(values[0]) for _ in range(int(n_degree))]
    if len(values) != int(n_degree):
        raise ValueError(f"Expected 1 or {n_degree} degree values, got {len(values)} from {spec!r}")
    return [float(v) for v in values]


def _parse_template_basis_gain_init(spec: str, n_basis: int, n_degree: int, mode: str) -> np.ndarray:
    values = _parse_floats(spec)
    if str(mode) == "global":
        if not values:
            values = [1.0]
        if len(values) != 1:
            raise ValueError(f"--template-basis-gain-init for global mode expects 1 value, got {len(values)}")
        return np.full((int(n_basis), int(n_degree)), float(values[0]), dtype=np.float64)
    if str(mode) == "per_basis":
        if not values:
            values = [1.0] * int(n_basis)
        elif len(values) == 1:
            values = [float(values[0])] * int(n_basis)
        elif len(values) != int(n_basis):
            raise ValueError(f"--template-basis-gain-init for per_basis expects 1 or {n_basis} values, got {len(values)}")
        return np.asarray(values, dtype=np.float64).reshape(int(n_basis), 1).repeat(int(n_degree), axis=1)
    if str(mode) == "per_basis_degree":
        if not values:
            return np.ones((int(n_basis), int(n_degree)), dtype=np.float64)
        if len(values) == 1:
            return np.full((int(n_basis), int(n_degree)), float(values[0]), dtype=np.float64)
        if len(values) == int(n_basis):
            return np.asarray(values, dtype=np.float64).reshape(int(n_basis), 1).repeat(int(n_degree), axis=1)
        if len(values) == int(n_basis) * int(n_degree):
            return np.asarray(values, dtype=np.float64).reshape(int(n_basis), int(n_degree))
        raise ValueError(
            "--template-basis-gain-init for per_basis_degree expects 1, "
            f"{n_basis}, or {int(n_basis) * int(n_degree)} values, got {len(values)}"
        )
    raise ValueError(f"Unsupported template basis gain mode: {mode}")


def _parse_list(spec: str) -> List[str]:
    text = str(spec or "").replace("\n", ",").replace(";", ",")
    return [v.strip() for v in text.split(",") if v.strip()]


def _parse_scalar_schedule(spec: str, *, name: str) -> List[ScalarSchedulePoint]:
    text = str(spec or "").strip()
    if not text:
        return []
    points: List[ScalarSchedulePoint] = []
    for piece in _parse_list(text):
        if ":" in piece:
            iter_text, value_text = piece.split(":", 1)
        elif "=" in piece:
            iter_text, value_text = piece.split("=", 1)
        else:
            raise ValueError(f"Invalid {name} schedule entry {piece!r}; expected ITER:VALUE")
        try:
            start_iter = int(iter_text.strip())
            value = float(value_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid {name} schedule entry {piece!r}; expected ITER:VALUE") from exc
        if start_iter < 0:
            raise ValueError(f"{name} schedule iteration must be non-negative: {piece!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} schedule value must be finite: {piece!r}")
        points.append(ScalarSchedulePoint(start_iter=int(start_iter), value=float(value)))
    if not points:
        return []
    dedup: Dict[int, float] = {}
    for point in points:
        dedup[int(point.start_iter)] = float(point.value)
    return [
        ScalarSchedulePoint(start_iter=int(start_iter), value=float(value))
        for start_iter, value in sorted(dedup.items())
    ]


def _scalar_schedule_value(
    schedule: Sequence[ScalarSchedulePoint],
    iteration: int,
    default: float,
) -> float:
    value = float(default)
    for point in schedule:
        if int(iteration) < int(point.start_iter):
            break
        value = float(point.value)
    return float(value)


def _scalar_schedule_to_dicts(schedule: Sequence[ScalarSchedulePoint]) -> List[Dict[str, Any]]:
    return [
        {"start_iter": int(point.start_iter), "value": float(point.value)}
        for point in schedule
    ]


def _parse_plus_list(spec: str) -> List[str]:
    return [v.strip() for v in str(spec or "").split("+") if v.strip()]


def _parse_stage_degree_token(token: str) -> int:
    text = str(token).strip().lower()
    if text.startswith("c"):
        text = text[1:]
    if not text:
        raise ValueError(f"Invalid empty C0 correction stage degree token: {token!r}")
    return int(text)


def _parse_c0_correction_stage_spec(
    spec: str,
    *,
    correction_degrees: Sequence[int],
    block_list: Sequence[int],
) -> List[C0CorrectionStage]:
    text = str(spec or "").strip()
    if not text:
        return []
    allowed_degrees = {int(v) for v in correction_degrees}
    allowed_blocks = {int(v) for v in block_list}
    if not allowed_degrees:
        raise ValueError("--c0-correction-stage-spec requires --c0-correction-degrees")
    if not allowed_blocks:
        raise ValueError("--c0-correction-stage-spec requires a non-empty multiblock block list")

    stages: List[C0CorrectionStage] = []
    start_iter = 0
    for index, raw_piece in enumerate(text.replace(";", ",").split(",")):
        piece = raw_piece.strip()
        if not piece:
            continue
        if "@" not in piece:
            raise ValueError(
                f"Invalid C0 correction stage {piece!r}: expected '<degrees>:<blocks>@<steps>' or 'off@<steps>'"
            )
        body, steps_text = piece.rsplit("@", 1)
        try:
            n_steps = int(str(steps_text).strip())
        except ValueError as exc:
            raise ValueError(f"Invalid C0 correction stage step count in {piece!r}") from exc
        if n_steps <= 0:
            raise ValueError(f"C0 correction stage step count must be positive in {piece!r}")
        body_text = str(body).strip()
        if body_text.lower() in {"none", "off", "freeze", "frozen"}:
            degrees = tuple()
            blocks = tuple()
        elif ":" not in body:
            raise ValueError(
                f"Invalid C0 correction stage {piece!r}: expected '<degrees>:<blocks>@<steps>' or 'off@<steps>'"
            )
        else:
            degrees_text, blocks_text = body.split(":", 1)
            if str(degrees_text).strip().lower() in {"*", "all"}:
                degrees = tuple(sorted(allowed_degrees))
            else:
                degrees = tuple(_parse_stage_degree_token(v) for v in _parse_plus_list(degrees_text))
            if str(blocks_text).strip().lower() in {"*", "all"}:
                blocks = tuple(sorted(allowed_blocks))
            else:
                blocks = tuple(int(v) for v in _parse_plus_list(blocks_text))
            if not degrees:
                raise ValueError(f"C0 correction stage has no active degrees: {piece!r}")
            if not blocks:
                raise ValueError(f"C0 correction stage has no active blocks: {piece!r}")
        bad_degrees = [int(v) for v in degrees if int(v) not in allowed_degrees]
        if bad_degrees:
            raise ValueError(
                f"C0 correction stage {piece!r} uses degrees not present in "
                f"--c0-correction-degrees: {bad_degrees}"
            )
        bad_blocks = [int(v) for v in blocks if int(v) not in allowed_blocks]
        if bad_blocks:
            raise ValueError(
                f"C0 correction stage {piece!r} uses blocks not present in "
                f"--c0-correction-block-list/--c0-correction-blocks: {bad_blocks}"
            )
        end_iter = start_iter + n_steps
        if degrees and blocks:
            label = (
                "+".join(f"C{int(v)}" for v in degrees)
                + ":"
                + "+".join(str(int(v)) for v in blocks)
                + f"@{int(n_steps)}"
            )
        else:
            label = f"off@{int(n_steps)}"
        stages.append(
            C0CorrectionStage(
                index=int(index),
                start_iter=int(start_iter),
                end_iter=int(end_iter),
                degrees=tuple(int(v) for v in degrees),
                blocks=tuple(int(v) for v in blocks),
                label=label,
            )
        )
        start_iter = end_iter
    if not stages:
        raise ValueError(f"--c0-correction-stage-spec did not contain any valid stages: {spec!r}")
    return stages


def _c0_correction_stage_for_iter(
    stages: Sequence[C0CorrectionStage],
    iteration: int,
) -> C0CorrectionStage | None:
    if not stages:
        return None
    for stage in stages:
        if int(iteration) < int(stage.end_iter):
            return stage
    return stages[-1]


def _c0_correction_stage_to_dict(stage: C0CorrectionStage) -> Dict[str, Any]:
    return {
        "index": int(stage.index),
        "start_iter": int(stage.start_iter),
        "end_iter": int(stage.end_iter),
        "degrees": [int(v) for v in stage.degrees],
        "blocks": [int(v) for v in stage.blocks],
        "label": str(stage.label),
    }


def _make_multiblock_c0_stage_mask(
    *,
    stage: C0CorrectionStage | None,
    correction_degrees: Sequence[int],
    block_list: Sequence[int],
    like: torch.Tensor,
) -> torch.Tensor:
    if stage is None:
        return torch.ones_like(like)
    active_degrees = {int(v) for v in stage.degrees}
    active_blocks = {int(v) for v in stage.blocks}
    pieces: List[np.ndarray] = []
    for block_count in block_list:
        block_count_int = int(block_count)
        block_mask = np.zeros(
            (int(len(correction_degrees)), block_count_int, block_count_int),
            dtype=np.float32,
        )
        if block_count_int in active_blocks:
            for degree_pos, degree_index in enumerate(correction_degrees):
                if int(degree_index) in active_degrees:
                    block_mask[int(degree_pos), :, :] = 1.0
        pieces.append(block_mask.reshape(-1))
    if not pieces:
        return torch.ones_like(like)
    mask_np = np.concatenate(pieces, axis=0)
    if int(mask_np.size) != int(like.numel()):
        raise ValueError(
            f"C0 correction stage mask size mismatch: mask={int(mask_np.size)} param={int(like.numel())}"
        )
    return torch.as_tensor(mask_np, device=like.device, dtype=like.dtype)


def _check_template_basis_path(path: Path, *, allow_suspicious: bool) -> None:
    text = str(path).replace("\\", "/").lower()
    hits = [token for token in SUSPICIOUS_TEMPLATE_BASIS_TOKENS if token in text]
    if hits and not allow_suspicious:
        raise ValueError(
            f"template-basis path looks simulated/oracle, not observed: {path} "
            f"(matched tokens: {hits}). Use observed catalog/map templates or pass "
            "--allow-suspicious-template-basis only for labelled diagnostics."
        )


def _freqtag(freq_mhz: float) -> str:
    return f"{int(round(float(freq_mhz) * 100)):05d}"


def _format_pattern(pattern: str, *, freq: float) -> str:
    try:
        return str(pattern).format(freq=float(freq), freqtag=_freqtag(float(freq)))
    except ValueError as exc:
        raise ValueError(f"invalid format pattern {str(pattern)!r} for freq={float(freq):.2f}") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if str(name) == "float64" else torch.float32


def _make_optimizer(args: argparse.Namespace, params: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
    name = str(args.optimizer_name).lower()
    lr = float(args.lr)
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
    if name == "nadam":
        return torch.optim.NAdam(
            params,
            lr=lr,
            betas=(float(args.adam_beta1), float(args.adam_beta2)),
            eps=float(args.adam_eps),
            weight_decay=float(args.weight_decay),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
            nesterov=bool(args.nesterov),
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    if name == "lbfgs":
        if int(args.lbfgs_max_iter) <= 0:
            raise ValueError("--lbfgs-max-iter must be positive")
        if int(args.lbfgs_history_size) <= 0:
            raise ValueError("--lbfgs-history-size must be positive")
        line_search_fn = None if str(args.lbfgs_line_search) == "none" else str(args.lbfgs_line_search)
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=int(args.lbfgs_max_iter),
            history_size=int(args.lbfgs_history_size),
            line_search_fn=line_search_fn,
            tolerance_grad=float(args.lbfgs_tolerance_grad),
            tolerance_change=float(args.lbfgs_tolerance_change),
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer_name}")


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    device_set = False
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not device_set:
            total = total.to(device=grad.device)
            device_set = True
        total = total + torch.sum(grad.double() * grad.double())
    return float(torch.sqrt(total).detach().cpu())


def _clone_optimizer_params(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _restore_optimizer_params(params: Sequence[torch.nn.Parameter], values: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def _blend_optimizer_params(
    params: Sequence[torch.nn.Parameter],
    start_values: Sequence[torch.Tensor],
    end_values: Sequence[torch.Tensor],
    alpha: float,
) -> None:
    with torch.no_grad():
        for param, start, end in zip(params, start_values, end_values):
            param.copy_(start + float(alpha) * (end - start))


def _clone_optimizer_state(opt: torch.optim.Optimizer) -> Dict[str, Any]:
    state_dict = opt.state_dict()

    def clone_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: clone_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [clone_value(val) for val in value]
        if isinstance(value, tuple):
            return tuple(clone_value(val) for val in value)
        return value

    return clone_value(state_dict)


def _radial_power_masks(
    image_size: int,
    num_bins: int,
    kmin: float,
    kmax: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, np.ndarray]:
    if num_bins <= 0:
        raise ValueError("--residual-power-num-bins must be positive")
    if not np.isfinite(kmin) or not np.isfinite(kmax) or not (0.0 <= kmin < kmax):
        raise ValueError("--residual-power-kmin/kmax must be finite with 0 <= kmin < kmax")
    ky = np.fft.fftfreq(int(image_size))
    kx = np.fft.fftfreq(int(image_size))
    rr = np.sqrt(ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :])
    edges = np.linspace(float(kmin), float(kmax), int(num_bins) + 1, dtype=np.float64)
    masks = []
    for bin_index in range(int(num_bins)):
        mask_np = (rr >= edges[bin_index]) & (rr < edges[bin_index + 1])
        count = int(np.count_nonzero(mask_np))
        if count == 0:
            raise ValueError(
                f"Empty residual radial-power bin {bin_index}: "
                f"[{edges[bin_index]}, {edges[bin_index + 1]})"
            )
        masks.append(mask_np.astype(np.float64) / float(count))
    mask = torch.as_tensor(np.stack(masks, axis=0), device=device, dtype=dtype)
    return mask, edges


def _radial_power(cube: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    fft = torch.fft.fft2(cube, dim=(-2, -1), norm="ortho")
    power = fft.real * fft.real + fft.imag * fft.imag
    return torch.einsum("fxy,bxy->fb", power, masks)


def _radial_log_power_angular_variance(
    cube: torch.Tensor,
    masks: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    fft = torch.fft.fft2(cube, dim=(-2, -1), norm="ortho")
    power = fft.real * fft.real + fft.imag * fft.imag
    log_power = torch.log(torch.clamp(power, min=float(eps)))
    mean = torch.einsum("fxy,bxy->fb", log_power, masks)
    mean_sq = torch.einsum("fxy,bxy->fb", log_power * log_power, masks)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    return torch.mean(var)


def _frequency_corr(cube: torch.Tensor, eps: float) -> torch.Tensor:
    flat = cube.reshape(cube.shape[0], -1)
    flat = flat - torch.mean(flat, dim=1, keepdim=True)
    cov = flat @ flat.transpose(0, 1)
    cov = cov / max(int(flat.shape[1]), 1)
    var = torch.clamp(
        torch.diag(cov),
        min=torch.as_tensor(float(eps), device=cube.device, dtype=cube.dtype),
    )
    std = torch.sqrt(var)
    return cov / torch.clamp(std[:, None] * std[None, :], min=float(eps))


def _cheb_design(freqs: Sequence[float], degree: int) -> np.ndarray:
    x = np.asarray(freqs, dtype=np.float64)
    if x.size == 1:
        z = np.zeros_like(x)
    else:
        mid = 0.5 * (float(np.min(x)) + float(np.max(x)))
        half = 0.5 * (float(np.max(x)) - float(np.min(x)))
        z = (x - mid) / max(half, 1e-12)
    cols = [np.ones_like(z)]
    if int(degree) >= 1:
        cols.append(z)
    for k in range(2, int(degree) + 1):
        cols.append(2.0 * z * cols[-1] - cols[-2])
    return np.stack(cols, axis=1).astype(np.float64)


def _load_dirty_cube(
    freqs: Sequence[float],
    pattern: str,
    *,
    eval_size: int,
    dtype: np.dtype,
) -> Tuple[np.ndarray, str]:
    slices: List[np.ndarray] = []
    first_path = ""
    for freq in freqs:
        path = _format_pattern(pattern, freq=float(freq))
        if not first_path:
            first_path = path
        slices.append(_central_crop(_load_fits_2d(path, dtype=dtype), int(eval_size)))
    return np.stack(slices, axis=0).astype(dtype, copy=False), first_path


def _load_wsclean_2d_eager(path: Path | str, dtype: np.dtype) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data)
        hdr = hdul[0].header.copy()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D WSClean product after squeezing, got {arr.shape} for {path}")
    return np.asarray(arr, dtype=dtype), hdr


def _write_cube(path: Path, cube: np.ndarray, template_path: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    if template_path:
        try:
            hdr = fits.getheader(template_path)
        except Exception:
            hdr = fits.Header()
    fits.writeto(path, np.asarray(cube, dtype=np.float32), header=hdr, overwrite=True)


def _safe_label(text: str) -> str:
    keep: List[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    label = "".join(keep).strip("._")
    return label or "run"


def _init_label(path: Path | None, index: int) -> str:
    if path is None:
        return f"{index:02d}_zero"
    name = path.name
    for suffix in ("_coeff_k.fits", ".fits", ".fit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{index:02d}_{_safe_label(name)}"


def _json_dumps_stable(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _parse_path_rewrites(spec: str) -> List[Tuple[str, str]]:
    rewrites: List[Tuple[str, str]] = []
    for piece in _parse_list(spec):
        if "=" not in piece:
            raise ValueError(
                f"Invalid --tile-cache-meta-path-rewrite entry {piece!r}; expected OLD=NEW"
            )
        old, new = piece.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old:
            raise ValueError(f"Invalid --tile-cache-meta-path-rewrite entry {piece!r}: OLD is empty")
        rewrites.append((old, new))
    return rewrites


def _rewrite_strings_for_meta_compare(value: Any, rewrites: Sequence[Tuple[str, str]]) -> Any:
    if not rewrites:
        return value
    if isinstance(value, str):
        out = value
        for old, new in rewrites:
            out = out.replace(str(old), str(new))
        return out
    if isinstance(value, dict):
        return {key: _rewrite_strings_for_meta_compare(val, rewrites) for key, val in value.items()}
    if isinstance(value, list):
        return [_rewrite_strings_for_meta_compare(val, rewrites) for val in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_strings_for_meta_compare(val, rewrites) for val in value)
    return value


def _tile_cache_meta(
    *,
    freq_index: int,
    freq_mhz: float,
    dense_grid_csv: Path,
    train_grid_csv: Path,
    train_response_pattern: str,
    image_size: int,
    response_crop_size: int,
    eval_size: int,
    tile_size: int,
    train_halo_px: int,
    model_margin_arg: int,
    pca_rank: int,
    rbf_scale_px: float,
    dtype: np.dtype,
    tile: Any,
) -> Dict[str, Any]:
    return {
        "cache_version": 1,
        "freq_index": int(freq_index),
        "freq_mhz": float(freq_mhz),
        "dense_grid_csv": str(dense_grid_csv),
        "train_grid_csv": str(train_grid_csv),
        "train_response_pattern": str(train_response_pattern),
        "image_size": int(image_size),
        "response_crop_size": int(response_crop_size),
        "eval_size": int(eval_size),
        "tile_size": int(tile_size),
        "train_halo_px": int(train_halo_px),
        "model_margin_arg": int(model_margin_arg),
        "pca_rank": int(pca_rank),
        "rbf_scale_px": float(rbf_scale_px),
        "dtype": str(np.dtype(dtype).name),
        "tile": {
            "name": str(tile.name),
            "x0": int(tile.x0),
            "x1": int(tile.x1),
            "y0": int(tile.y0),
            "y1": int(tile.y1),
        },
    }


def _tile_cache_path(cache_dir: Path, meta: Dict[str, Any]) -> Path:
    meta_json = _json_dumps_stable(meta)
    digest = hashlib.sha1(meta_json.encode("utf-8")).hexdigest()[:16]
    freqtag = _freqtag(float(meta["freq_mhz"]))
    tile_name = _safe_label(str(meta["tile"]["name"]))
    return cache_dir / f"freq{freqtag}_{tile_name}_{digest}.npz"


def _load_tile_cache(
    path: Path,
    expected_meta: Dict[str, Any],
    *,
    meta_path_rewrites: Sequence[Tuple[str, str]] = (),
) -> PcaTileCache | None:
    if not path.exists():
        return None
    expected_compare = _rewrite_strings_for_meta_compare(expected_meta, meta_path_rewrites)
    expected_json = _json_dumps_stable(expected_compare)
    cache_dtype = np.dtype(str(expected_meta.get("dtype", "float32")))
    try:
        with np.load(path, allow_pickle=False) as data:
            meta_json = str(np.asarray(data["meta_json"]).item())
            if meta_path_rewrites:
                try:
                    cached_meta = json.loads(meta_json)
                except json.JSONDecodeError:
                    return None
                cached_compare = _rewrite_strings_for_meta_compare(cached_meta, meta_path_rewrites)
                cached_json = _json_dumps_stable(cached_compare)
            else:
                cached_json = meta_json
            if cached_json != expected_json:
                return None
            return PcaTileCache(
                freq_index=int(np.asarray(data["freq_index"]).item()),
                freq_mhz=float(np.asarray(data["freq_mhz"]).item()),
                tile_name=str(np.asarray(data["tile_name"]).item()),
                model_size=int(np.asarray(data["model_size"]).item()),
                eval_size=int(np.asarray(data["eval_size"]).item()),
                crop_start=int(np.asarray(data["crop_start"]).item()),
                src_y=np.asarray(data["src_y"], dtype=np.int64),
                src_x=np.asarray(data["src_x"], dtype=np.int64),
                flat_idx=np.asarray(data["flat_idx"], dtype=np.int64),
                mean_weight=np.asarray(data["mean_weight"], dtype=cache_dtype),
                coeff_weight=np.asarray(data["coeff_weight"], dtype=cache_dtype),
                kernel_stack=np.asarray(data["kernel_stack"], dtype=cache_dtype),
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "tile_cache_load_failed",
                    "path": str(path),
                    "error": str(exc),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return None


def _save_tile_cache(path: Path, tile: PcaTileCache, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(
        tmp,
        meta_json=np.asarray(_json_dumps_stable(meta)),
        freq_index=np.asarray(tile.freq_index, dtype=np.int64),
        freq_mhz=np.asarray(tile.freq_mhz, dtype=np.float64),
        tile_name=np.asarray(tile.tile_name),
        model_size=np.asarray(tile.model_size, dtype=np.int64),
        eval_size=np.asarray(tile.eval_size, dtype=np.int64),
        crop_start=np.asarray(tile.crop_start, dtype=np.int64),
        src_y=tile.src_y,
        src_x=tile.src_x,
        flat_idx=tile.flat_idx,
        mean_weight=tile.mean_weight,
        coeff_weight=tile.coeff_weight,
        kernel_stack=tile.kernel_stack,
    )
    if tmp.exists():
        tmp.replace(path)
    else:
        # numpy appends .npz when given a path whose suffix is not exactly .npz.
        appended = Path(str(tmp) + ".npz")
        appended.replace(path)


def _load_train_aligned(
    *,
    rows: Sequence[Dict[str, Any]],
    pattern: str,
    freq_mhz: float,
    response_crop_size: int,
    model_size: int,
    dtype: np.dtype,
    align_x: float,
    align_y: float,
    response_cache: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    aligned: List[np.ndarray] = []
    for row in rows:
        label = str(row["label"])
        cropped = None if response_cache is None else response_cache.get(label)
        if cropped is None:
            arr, _ = _load_wsclean_2d_eager(_fmt(pattern, label=label, freq=float(freq_mhz)), dtype)
            cropped = _crop2d(arr, size=int(response_crop_size), center_x=None, center_y=None)
        shifted = _shift_zero(
            cropped,
            dx=int(round(float(align_x) - float(row["x"]))),
            dy=int(round(float(align_y) - float(row["y"]))),
        )
        aligned.append(np.asarray(_central_crop(shifted, int(model_size)), dtype=dtype).reshape(-1))
    return np.stack(aligned, axis=0)


def _support_contrib_entries(
    *,
    dense_tile: Sequence[Dict[str, Any]],
    x_contribs: Sequence[Sequence[Tuple[int, float, int]]],
    y_contribs: Sequence[Sequence[Tuple[int, float, int]]],
    coeff_interp: np.ndarray,
    align_x: float,
    align_y: float,
    model_size: int,
    weight_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = int(model_size) // 2
    src_y: List[int] = []
    src_x: List[int] = []
    flat_idx: List[int] = []
    mean_weight: List[float] = []
    coeff_weight: List[np.ndarray] = []
    rank = int(coeff_interp.shape[1])
    for i, row in enumerate(dense_tile):
        sx = float(row["x"])
        sy = float(row["y"])
        row_coeff = np.asarray(coeff_interp[i, :rank], dtype=weight_dtype)
        for y_pix, wy, dy in y_contribs[int(row["iy"])]:
            for x_pix, wx, dx in x_contribs[int(row["ix"])]:
                w = float(wy) * float(wx)
                if w == 0.0:
                    continue
                ox = center + int(round((sx + float(dx)) - float(align_x)))
                oy = center + int(round((sy + float(dy)) - float(align_y)))
                if 0 <= ox < int(model_size) and 0 <= oy < int(model_size):
                    src_y.append(int(y_pix))
                    src_x.append(int(x_pix))
                    flat_idx.append(int(oy) * int(model_size) + int(ox))
                    mean_weight.append(float(w))
                    coeff_weight.append((float(w) * row_coeff).astype(weight_dtype, copy=False))
    if not src_y:
        raise ValueError("No contribution entries were generated for tile")
    return (
        np.asarray(src_y, dtype=np.int64),
        np.asarray(src_x, dtype=np.int64),
        np.asarray(flat_idx, dtype=np.int64),
        np.asarray(mean_weight, dtype=weight_dtype),
        np.stack(coeff_weight, axis=0).astype(weight_dtype, copy=False),
        np.asarray([rank], dtype=np.int64),
    )


def _build_freq_tile_caches(
    *,
    freq_index: int,
    freq_mhz: float,
    dense_grid_csv: Path,
    train_grid_csv: Path,
    train_response_pattern: str,
    image_size: int,
    response_crop_size: int,
    eval_size: int,
    tile_size: int,
    train_halo_px: int,
    model_margin_arg: int,
    pca_rank: int,
    rbf_scale_px: float,
    dtype: np.dtype,
    progress_every_tile: int,
    max_tiles: int,
    tile_cache_dir: Path | None,
    tile_cache_refresh: bool,
    tile_cache_meta_path_rewrites: Sequence[Tuple[str, str]],
    preload_train_responses: bool,
    cache_build_only: bool,
    tile_indices: Sequence[int] = (),
) -> List[PcaTileCache]:
    weight_dtype = np.dtype(dtype)
    dense_rows = _load_grid_csv(dense_grid_csv)
    x_support, y_support = _assign_grid_indices(dense_rows)
    x_contribs = _axis_linear_contribs(int(image_size), x_support)
    y_contribs = _axis_linear_contribs(int(image_size), y_support)
    train_rows_all = _load_grid_csv(train_grid_csv)
    all_tiles = _make_tiles(int(image_size), int(tile_size))
    selected_tile_indices = [int(v) for v in tile_indices]
    if selected_tile_indices:
        if int(max_tiles) > 0:
            raise ValueError("tile_indices and max_tiles are mutually exclusive")
        if len(set(selected_tile_indices)) != len(selected_tile_indices):
            raise ValueError("tile_indices must not contain duplicates")
        bad_indices = [v for v in selected_tile_indices if v < 0 or v >= len(all_tiles)]
        if bad_indices:
            raise ValueError(
                f"tile_indices out of range for {len(all_tiles)} tiles: {bad_indices}"
            )
        tiles = [all_tiles[v] for v in selected_tile_indices]
    else:
        tiles = all_tiles
        if int(max_tiles) > 0:
            tiles = tiles[: int(max_tiles)]
    train_response_cache: Dict[str, np.ndarray] | None = None
    if bool(preload_train_responses):
        preload_tiles = list(tiles)
        if tile_cache_dir is not None and not bool(tile_cache_refresh):
            preload_tiles = []
            for tile in tiles:
                cache_meta = _tile_cache_meta(
                    freq_index=int(freq_index),
                    freq_mhz=float(freq_mhz),
                    dense_grid_csv=dense_grid_csv,
                    train_grid_csv=train_grid_csv,
                    train_response_pattern=str(train_response_pattern),
                    image_size=int(image_size),
                    response_crop_size=int(response_crop_size),
                    eval_size=int(eval_size),
                    tile_size=int(tile_size),
                    train_halo_px=int(train_halo_px),
                    model_margin_arg=int(model_margin_arg),
                    pca_rank=int(pca_rank),
                    rbf_scale_px=float(rbf_scale_px),
                    dtype=dtype,
                    tile=tile,
                )
                cache_path = _tile_cache_path(tile_cache_dir, cache_meta)
                if not cache_path.exists():
                    preload_tiles.append(tile)
        preload_rows_by_label: Dict[str, Dict[str, Any]] = {}
        for tile in preload_tiles:
            for row in _select_train_rows(
                train_rows_all,
                tile,
                halo_px=int(train_halo_px),
                image_size=int(image_size),
            ):
                preload_rows_by_label[str(row["label"])] = row
        preload_rows = list(preload_rows_by_label.values())
        train_response_cache = {}
        print(
            json.dumps(
                {
                    "event": "train_response_preload_start",
                    "freq_mhz": float(freq_mhz),
                    "n_tiles_for_preload": int(len(preload_tiles)),
                    "n_train_rows": int(len(preload_rows)),
                    "response_crop_size": int(response_crop_size),
                    "dtype": str(weight_dtype.name),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for row_idx, row in enumerate(preload_rows, start=1):
            label = str(row["label"])
            arr, _ = _load_wsclean_2d_eager(_fmt(train_response_pattern, label=label, freq=float(freq_mhz)), weight_dtype)
            train_response_cache[label] = np.asarray(
                _crop2d(arr, size=int(response_crop_size), center_x=None, center_y=None),
                dtype=weight_dtype,
            )
            if row_idx == 1 or row_idx % 512 == 0 or row_idx == len(preload_rows):
                print(
                    json.dumps(
                        {
                            "event": "train_response_preload_progress",
                            "freq_mhz": float(freq_mhz),
                            "rows_loaded": int(row_idx),
                            "rows_total": int(len(preload_rows)),
                            "time_utc": _now(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "event": "train_response_preload_done",
                    "freq_mhz": float(freq_mhz),
                    "n_cached": int(len(train_response_cache)),
                    "bytes_cached": int(sum(arr.nbytes for arr in train_response_cache.values())),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    caches: List[PcaTileCache] = []
    for tile_idx, tile in enumerate(tiles, start=1):
        cache_meta = _tile_cache_meta(
            freq_index=int(freq_index),
            freq_mhz=float(freq_mhz),
            dense_grid_csv=dense_grid_csv,
            train_grid_csv=train_grid_csv,
            train_response_pattern=str(train_response_pattern),
            image_size=int(image_size),
            response_crop_size=int(response_crop_size),
            eval_size=int(eval_size),
            tile_size=int(tile_size),
            train_halo_px=int(train_halo_px),
            model_margin_arg=int(model_margin_arg),
            pca_rank=int(pca_rank),
            rbf_scale_px=float(rbf_scale_px),
            dtype=dtype,
            tile=tile,
        )
        cache_path = _tile_cache_path(tile_cache_dir, cache_meta) if tile_cache_dir is not None else None
        if cache_path is not None and not bool(tile_cache_refresh):
            if bool(cache_build_only) and cache_path.exists() and cache_path.stat().st_size > 0:
                if int(progress_every_tile) > 0 and (
                    tile_idx == 1 or tile_idx % int(progress_every_tile) == 0 or tile_idx == len(tiles)
                ):
                    print(
                        json.dumps(
                            {
                                "event": "tile_cache_hit",
                                "freq_mhz": float(freq_mhz),
                                "tile": tile.name,
                                "tile_index": int(tile_idx),
                                "path": str(cache_path),
                                "fast_cache_build_only": True,
                                "time_utc": _now(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                continue
            cached_tile = _load_tile_cache(
                cache_path,
                cache_meta,
                meta_path_rewrites=tile_cache_meta_path_rewrites,
            )
            if cached_tile is not None:
                caches.append(cached_tile)
                if int(progress_every_tile) > 0 and (
                    tile_idx == 1 or tile_idx % int(progress_every_tile) == 0 or tile_idx == len(tiles)
                ):
                    print(
                        json.dumps(
                            {
                                "event": "tile_cache_hit",
                                "freq_mhz": float(freq_mhz),
                                "tile": tile.name,
                                "tile_index": int(tile_idx),
                                "rank": int(cached_tile.kernel_stack.shape[0] - 1),
                                "n_entries": int(cached_tile.src_y.size),
                                "path": str(cache_path),
                                "time_utc": _now(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                continue

        dense_tile = _select_dense_rows(dense_rows, tile)
        train_rows = _select_train_rows(
            train_rows_all,
            tile,
            halo_px=int(train_halo_px),
            image_size=int(image_size),
        )
        if not dense_tile:
            raise ValueError(f"{freq_mhz:.2f} {tile.name}: zero dense rows")
        if len(train_rows) < 4:
            raise ValueError(f"{freq_mhz:.2f} {tile.name}: too few train rows ({len(train_rows)})")

        dense_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in dense_tile], dtype=np.float64)
        train_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in train_rows], dtype=np.float64)
        align_x = float(np.median(dense_xy[:, 0]))
        align_y = float(np.median(dense_xy[:, 1]))

        contrib_positions: List[Tuple[float, float]] = []
        for row in dense_tile:
            sx = float(row["x"])
            sy = float(row["y"])
            for _, _, dy in y_contribs[int(row["iy"])]:
                for _, _, dx in x_contribs[int(row["ix"])]:
                    contrib_positions.append((sx + float(dx), sy + float(dy)))
        contrib_xy = np.asarray(contrib_positions, dtype=np.float64)
        max_offset = max(
            float(np.max(np.abs(dense_xy[:, 0] - align_x))),
            float(np.max(np.abs(dense_xy[:, 1] - align_y))),
            float(np.max(np.abs(train_xy[:, 0] - align_x))),
            float(np.max(np.abs(train_xy[:, 1] - align_y))),
            float(np.max(np.abs(contrib_xy[:, 0] - align_x))),
            float(np.max(np.abs(contrib_xy[:, 1] - align_y))),
        )
        model_margin = int(model_margin_arg) if int(model_margin_arg) >= 0 else int(math.ceil(max_offset)) + 8
        model_size = int(eval_size) + 2 * int(model_margin)
        if model_size > int(response_crop_size):
            raise ValueError(f"{freq_mhz:.2f} {tile.name}: model_size={model_size} > response_crop_size")

        print(
            json.dumps(
                {
                    "event": "build_tile_cache_start",
                    "freq_mhz": float(freq_mhz),
                    "tile": tile.name,
                    "tile_index": int(tile_idx),
                    "tiles_total": int(len(tiles)),
                    "n_dense": int(len(dense_tile)),
                    "n_train": int(len(train_rows)),
                    "model_size": int(model_size),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        train_aligned = _load_train_aligned(
            rows=train_rows,
            pattern=str(train_response_pattern),
            freq_mhz=float(freq_mhz),
            response_crop_size=int(response_crop_size),
            model_size=int(model_size),
            dtype=dtype,
            align_x=align_x,
            align_y=align_y,
            response_cache=train_response_cache,
        )
        rank = min(int(pca_rank), max(1, len(train_rows) - 1))
        mean, basis, coeff_train = _pca_basis(train_aligned, int(rank))
        actual_rank = int(basis.shape[0])
        coeff_interp = _rbf_coefficients_many(
            target_xy=dense_xy,
            train_xy=train_xy,
            coeff_train=coeff_train[:, :actual_rank],
            scale_px=float(rbf_scale_px),
        ).astype(weight_dtype, copy=False)
        src_y, src_x, flat_idx, mean_weight, coeff_weight, _rank_arr = _support_contrib_entries(
            dense_tile=dense_tile,
            x_contribs=x_contribs,
            y_contribs=y_contribs,
            coeff_interp=coeff_interp[:, :actual_rank],
            align_x=align_x,
            align_y=align_y,
            model_size=int(model_size),
            weight_dtype=weight_dtype,
        )
        kernel_stack = np.concatenate(
            [
                np.asarray(mean, dtype=weight_dtype).reshape(1, int(model_size), int(model_size)),
                np.asarray(basis[:actual_rank], dtype=weight_dtype).reshape(actual_rank, int(model_size), int(model_size)),
            ],
            axis=0,
        )
        crop_start = int(model_size) // 2 + (int(model_size) - int(eval_size)) // 2
        cache_entry = PcaTileCache(
            freq_index=int(freq_index),
            freq_mhz=float(freq_mhz),
            tile_name=str(tile.name),
            model_size=int(model_size),
            eval_size=int(eval_size),
            crop_start=int(crop_start),
            src_y=src_y,
            src_x=src_x,
            flat_idx=flat_idx,
            mean_weight=mean_weight,
            coeff_weight=coeff_weight,
            kernel_stack=kernel_stack,
        )
        if not bool(cache_build_only):
            caches.append(cache_entry)
        if cache_path is not None:
            _save_tile_cache(cache_path, cache_entry, cache_meta)
            print(
                json.dumps(
                    {
                        "event": "tile_cache_saved",
                        "freq_mhz": float(freq_mhz),
                        "tile": tile.name,
                        "tile_index": int(tile_idx),
                        "path": str(cache_path),
                        "time_utc": _now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if int(progress_every_tile) > 0 and (
            tile_idx == 1 or tile_idx % int(progress_every_tile) == 0 or tile_idx == len(tiles)
        ):
            print(
                json.dumps(
                    {
                        "event": "build_tile_cache_done",
                        "freq_mhz": float(freq_mhz),
                        "tile": tile.name,
                        "tile_index": int(tile_idx),
                        "rank": int(actual_rank),
                        "n_entries": int(src_y.size),
                        "time_utc": _now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return caches


class CachedPcaProxyForward:
    def __init__(
        self,
        caches: Sequence[PcaTileCache],
        *,
        n_freq: int,
        eval_size: int,
        device: torch.device,
        dtype: torch.dtype,
        keep_kernel_fft_on_device: bool,
        checkpoint_tiles: bool,
    ) -> None:
        self.caches = list(caches)
        self.n_freq = int(n_freq)
        self.eval_size = int(eval_size)
        self.device = device
        self.dtype = dtype
        self.keep_kernel_fft_on_device = bool(keep_kernel_fft_on_device)
        self.checkpoint_tiles = bool(checkpoint_tiles)
        self._kernel_fft_cache: Dict[int, torch.Tensor] = {}

    def _kernel_fft(self, tile_id: int, tile: PcaTileCache) -> torch.Tensor:
        cached = self._kernel_fft_cache.get(tile_id)
        if cached is not None:
            return cached
        kernel = torch.as_tensor(tile.kernel_stack, device=self.device, dtype=self.dtype)
        m = int(tile.model_size)
        full_size = 2 * m - 1
        kernel_pad = F.pad(kernel, (0, full_size - m, 0, full_size - m))
        kernel_fft = torch.fft.rfft2(kernel_pad)
        if self.keep_kernel_fft_on_device:
            self._kernel_fft_cache[tile_id] = kernel_fft
        return kernel_fft

    def _tile_forward(self, tile_id: int, tile: PcaTileCache, flux: torch.Tensor) -> torch.Tensor:
        m = int(tile.model_size)
        rank = int(tile.coeff_weight.shape[1])
        src_y = torch.as_tensor(tile.src_y, device=self.device, dtype=torch.long)
        src_x = torch.as_tensor(tile.src_x, device=self.device, dtype=torch.long)
        flat_idx = torch.as_tensor(tile.flat_idx, device=self.device, dtype=torch.long)
        mean_weight = torch.as_tensor(tile.mean_weight, device=self.device, dtype=self.dtype)
        coeff_weight = torch.as_tensor(tile.coeff_weight, device=self.device, dtype=self.dtype)
        vals = flux[src_y, src_x]

        mean_flat = torch.zeros((m * m,), device=self.device, dtype=self.dtype)
        mean_flat.index_add_(0, flat_idx, vals * mean_weight)
        coeff_flat = torch.zeros((rank, m * m), device=self.device, dtype=self.dtype)
        coeff_flat.index_add_(1, flat_idx, coeff_weight.transpose(0, 1) * vals.unsqueeze(0))
        maps = torch.cat([mean_flat.view(1, m, m), coeff_flat.view(rank, m, m)], dim=0)

        full_size = 2 * m - 1
        maps_pad = F.pad(maps, (0, full_size - m, 0, full_size - m))
        pred_full = torch.fft.irfft2(
            torch.fft.rfft2(maps_pad) * self._kernel_fft(tile_id, tile),
            s=(full_size, full_size),
        ).sum(dim=0)
        s = int(tile.crop_start)
        return pred_full[s : s + int(tile.eval_size), s : s + int(tile.eval_size)]

    def _tile_forward_batch(
        self,
        tile_id: int,
        tile: PcaTileCache,
        flux: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one tile to a batch without rebuilding its FFT for every sample."""
        if flux.ndim != 3:
            raise ValueError(
                f"Batched tile flux must have shape [batch,y,x], got {tuple(flux.shape)}"
            )
        batch_size = int(flux.shape[0])
        m = int(tile.model_size)
        rank = int(tile.coeff_weight.shape[1])
        src_y = torch.as_tensor(tile.src_y, device=self.device, dtype=torch.long)
        src_x = torch.as_tensor(tile.src_x, device=self.device, dtype=torch.long)
        flat_idx = torch.as_tensor(tile.flat_idx, device=self.device, dtype=torch.long)
        mean_weight = torch.as_tensor(tile.mean_weight, device=self.device, dtype=self.dtype)
        coeff_weight = torch.as_tensor(tile.coeff_weight, device=self.device, dtype=self.dtype)
        vals = flux[:, src_y, src_x]

        mean_flat = torch.zeros(
            (batch_size, m * m),
            device=self.device,
            dtype=self.dtype,
        )
        mean_flat.index_add_(1, flat_idx, vals * mean_weight.unsqueeze(0))
        coeff_flat = torch.zeros(
            (batch_size, rank, m * m),
            device=self.device,
            dtype=self.dtype,
        )
        coeff_flat.index_add_(
            2,
            flat_idx,
            coeff_weight.transpose(0, 1).unsqueeze(0) * vals.unsqueeze(1),
        )
        maps = torch.cat(
            [
                mean_flat.view(batch_size, 1, m, m),
                coeff_flat.view(batch_size, rank, m, m),
            ],
            dim=1,
        )

        full_size = 2 * m - 1
        maps_pad = F.pad(maps, (0, full_size - m, 0, full_size - m))
        pred_full = torch.fft.irfft2(
            torch.fft.rfft2(maps_pad)
            * self._kernel_fft(tile_id, tile).unsqueeze(0),
            s=(full_size, full_size),
        ).sum(dim=1)
        s = int(tile.crop_start)
        return pred_full[
            :,
            s : s + int(tile.eval_size),
            s : s + int(tile.eval_size),
        ]

    def __call__(self, flux_cube: torch.Tensor) -> torch.Tensor:
        if flux_cube.ndim == 4:
            pred = torch.zeros(
                (
                    int(flux_cube.shape[0]),
                    self.n_freq,
                    self.eval_size,
                    self.eval_size,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            for tile_id, tile in enumerate(self.caches):
                flux_slice = flux_cube[:, tile.freq_index]
                if self.checkpoint_tiles and torch.is_grad_enabled() and flux_slice.requires_grad:
                    pred_tile = checkpoint(
                        lambda x, tid=tile_id, t=tile: self._tile_forward_batch(tid, t, x),
                        flux_slice,
                        use_reentrant=False,
                    )
                else:
                    pred_tile = self._tile_forward_batch(tile_id, tile, flux_slice)
                pred[:, tile.freq_index] = pred[:, tile.freq_index] + pred_tile
            return pred
        if flux_cube.ndim != 3:
            raise ValueError(
                "CachedPcaProxyForward expects [freq,y,x] or [batch,freq,y,x], "
                f"got {tuple(flux_cube.shape)}"
            )
        pred = torch.zeros(
            (self.n_freq, self.eval_size, self.eval_size),
            device=self.device,
            dtype=self.dtype,
        )
        for tile_id, tile in enumerate(self.caches):
            flux_slice = flux_cube[tile.freq_index]
            if self.checkpoint_tiles and torch.is_grad_enabled() and flux_slice.requires_grad:
                pred_tile = checkpoint(
                    lambda x, tid=tile_id, t=tile: self._tile_forward(tid, t, x),
                    flux_slice,
                    use_reentrant=False,
                )
            else:
                pred_tile = self._tile_forward(tile_id, tile, flux_slice)
            pred[tile.freq_index] = pred[tile.freq_index] + pred_tile
        return pred


def _tensor_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().double().reshape(-1)
    bb = b.detach().double().reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(den.cpu()) <= 0.0:
        return float("nan")
    return float((aa @ bb / den).cpu())


def _cube_metrics(est_eor: torch.Tensor, true_eor: torch.Tensor, dirty_total: torch.Tensor, pred_fg: torch.Tensor) -> Dict[str, Any]:
    err = est_eor - true_eor
    eor_rms = torch.sqrt(torch.mean(true_eor.double() ** 2))
    err_rms = torch.sqrt(torch.mean(err.double() ** 2))
    est_rms = torch.sqrt(torch.mean(est_eor.double() ** 2))
    data_rms = torch.sqrt(torch.mean((pred_fg - dirty_total).double() ** 2))
    per_freq: List[Dict[str, float]] = []
    for i in range(int(true_eor.shape[0])):
        er = torch.sqrt(torch.mean((est_eor[i].double() - true_eor[i].double()) ** 2))
        tr = torch.sqrt(torch.mean(true_eor[i].double() ** 2))
        per_freq.append(
            {
                "freq_index": int(i),
                "residual_over_dirty_eor_rms": float((er / torch.clamp(tr, min=1e-300)).cpu()),
                "corr": _tensor_corr(est_eor[i], true_eor[i]),
                "estimate_over_truth_rms": float((torch.sqrt(torch.mean(est_eor[i].double() ** 2)) / torch.clamp(tr, min=1e-300)).cpu()),
            }
        )
    return {
        "global_residual_over_dirty_eor_rms": float((err_rms / torch.clamp(eor_rms, min=1e-300)).cpu()),
        "global_corr": _tensor_corr(est_eor, true_eor),
        "global_estimate_over_truth_rms": float((est_rms / torch.clamp(eor_rms, min=1e-300)).cpu()),
        "data_residual_rms": float(data_rms.cpu()),
        "dirty_eor_rms": float(eor_rms.cpu()),
        "per_freq": per_freq,
    }


def _spatial_tv(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return torch.mean(dx * dx) + torch.mean(dy * dy)


def _dct_rms_normalized_basis(
    n_modes: int,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return low-frequency DCT-II vectors with unit spatial RMS per mode."""

    if int(n_modes) <= 0:
        raise ValueError("DCT correction size must be positive")
    if int(n_modes) > int(size):
        raise ValueError(f"DCT correction size {n_modes} exceeds image size {size}")
    x = torch.arange(int(size), device=device, dtype=dtype)
    k = torch.arange(int(n_modes), device=device, dtype=dtype).reshape(int(n_modes), 1)
    basis = torch.cos(math.pi * (x.reshape(1, int(size)) + 0.5) * k / float(size))
    if int(n_modes) > 1:
        basis[1:] = basis[1:] * math.sqrt(2.0)
    return basis


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    freqs = _parse_floats(args.freqs_mhz)
    if not bool(args.cache_build_only) and len(freqs) < int(args.cheb_degree) + 1:
        raise ValueError("Need at least degree+1 frequencies to fit Chebyshev foreground coefficients")
    np_dtype = np.dtype(np.float32 if args.dtype == "float32" else np.float64)
    torch_dtype = _torch_dtype(args.dtype)
    device = torch.device(args.device)
    tile_cache_meta_path_rewrites = _parse_path_rewrites(args.tile_cache_meta_path_rewrite)

    caches: List[PcaTileCache] = []
    for fi, freq in enumerate(freqs):
        dense_grid = Path(_format_pattern(args.dense_grid_csv_pattern, freq=float(freq)))
        train_grid = Path(_format_pattern(args.train_grid_csv_pattern, freq=float(freq)))
        caches.extend(
            _build_freq_tile_caches(
                freq_index=int(fi) + int(args.freq_index_offset),
                freq_mhz=float(freq),
                dense_grid_csv=dense_grid,
                train_grid_csv=train_grid,
                train_response_pattern=str(args.train_response_pattern),
                image_size=int(args.image_size),
                response_crop_size=int(args.response_crop_size),
                eval_size=int(args.eval_crop_size),
                tile_size=int(args.tile_size),
                train_halo_px=int(args.train_halo_px),
                model_margin_arg=int(args.model_margin),
                pca_rank=int(args.pca_rank),
                rbf_scale_px=float(args.rbf_scale_px),
                dtype=np_dtype,
                progress_every_tile=int(args.progress_every_tile),
                max_tiles=int(args.max_tiles_per_freq),
                tile_cache_dir=args.tile_cache_dir,
                tile_cache_refresh=bool(args.tile_cache_refresh),
                tile_cache_meta_path_rewrites=tile_cache_meta_path_rewrites,
                preload_train_responses=bool(args.preload_train_responses),
                cache_build_only=bool(args.cache_build_only),
            )
        )

    if bool(args.cache_build_only):
        summary = {
            "event": "cache_build_only_done",
            "n_freqs": int(len(freqs)),
            "freq_index_offset": int(args.freq_index_offset),
            "n_tile_caches": int(len(caches)),
            "tile_cache_dir": str(args.tile_cache_dir) if args.tile_cache_dir is not None else None,
            "preload_train_responses": bool(args.preload_train_responses),
            "time_utc": _now(),
        }
        (args.out_dir / "cache_build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        return

    dirty_fg_np, first_template = _load_dirty_cube(
        freqs,
        args.truth_fg_pattern,
        eval_size=int(args.eval_crop_size),
        dtype=np_dtype,
    )
    dirty_eor_np, _ = _load_dirty_cube(
        freqs,
        args.truth_eor_pattern,
        eval_size=int(args.eval_crop_size),
        dtype=np_dtype,
    )
    dirty_total_np = dirty_fg_np.astype(np.float64) + dirty_eor_np.astype(np.float64)

    forward_op = CachedPcaProxyForward(
        caches,
        n_freq=len(freqs),
        eval_size=int(args.eval_crop_size),
        device=device,
        dtype=torch_dtype,
        keep_kernel_fft_on_device=bool(args.keep_kernel_fft_on_device),
        checkpoint_tiles=bool(args.checkpoint_tiles),
    )
    cheb_np = _cheb_design(freqs, int(args.cheb_degree))
    cheb = torch.as_tensor(cheb_np, device=device, dtype=torch_dtype)
    q_cheb, _ = torch.linalg.qr(cheb)
    k_to_jy = torch.as_tensor(
        [_k_to_jy_per_pixel(float(freq), float(args.pixel_arcsec)) for freq in freqs],
        device=device,
        dtype=torch_dtype,
    )
    dirty_total = torch.as_tensor(dirty_total_np, device=device, dtype=torch_dtype)
    dirty_eor = torch.as_tensor(dirty_eor_np, device=device, dtype=torch_dtype)
    total_rms = torch.sqrt(torch.mean(dirty_total.double() ** 2)).to(device=device, dtype=torch_dtype)
    if float(args.data_error_rms) > 0.0:
        data_error_rms = torch.as_tensor(float(args.data_error_rms), device=device, dtype=torch_dtype)
        data_error_mode = "explicit"
    else:
        data_error_rms = total_rms
        data_error_mode = "dirty_total_rms"
    if str(args.residual_cheb_norm_mode) == "data_error_rms":
        residual_cheb_norm_rms = data_error_rms
    elif str(args.residual_cheb_norm_mode) == "dirty_total_rms":
        residual_cheb_norm_rms = total_rms
    else:
        raise ValueError(f"Unsupported residual Cheb norm mode: {args.residual_cheb_norm_mode}")
    if str(args.residual_spatial_lowpass_norm_mode) == "data_error_rms":
        residual_spatial_lowpass_norm_rms = data_error_rms
    elif str(args.residual_spatial_lowpass_norm_mode) == "dirty_total_rms":
        residual_spatial_lowpass_norm_rms = total_rms
    else:
        raise ValueError(
            f"Unsupported residual spatial lowpass norm mode: {args.residual_spatial_lowpass_norm_mode}"
        )
    if (
        not np.isfinite(float(args.residual_spatial_lowpass_weight))
        or float(args.residual_spatial_lowpass_weight) < 0.0
    ):
        raise ValueError("--residual-spatial-lowpass-weight must be finite and non-negative")
    residual_spatial_lowpass_basis: torch.Tensor | None = None
    if float(args.residual_spatial_lowpass_weight) != 0.0:
        residual_spatial_lowpass_basis = _dct_rms_normalized_basis(
            int(args.residual_spatial_lowpass_dct_size),
            int(args.eval_crop_size),
            device=device,
            dtype=torch_dtype,
        )
    if float(args.residual_power_weight) != 0.0 and str(args.residual_power_target_source) == "none":
        raise ValueError("--residual-power-weight requires --residual-power-target-source")
    if str(args.residual_power_target_source) == "npz" and args.residual_power_target_npz is None:
        raise ValueError("--residual-power-target-source npz requires --residual-power-target-npz")
    if not np.isfinite(float(args.residual_power_log_eps)) or float(args.residual_power_log_eps) <= 0.0:
        raise ValueError("--residual-power-log-eps must be finite and positive")
    if not np.isfinite(float(args.residual_power_log_tolerance)) or float(args.residual_power_log_tolerance) < 0.0:
        raise ValueError("--residual-power-log-tolerance must be finite and non-negative")
    if (
        not np.isfinite(float(args.residual_power_isotropy_weight))
        or float(args.residual_power_isotropy_weight) < 0.0
    ):
        raise ValueError("--residual-power-isotropy-weight must be finite and non-negative")
    residual_power_masks: torch.Tensor | None = None
    residual_power_bin_edges = np.zeros((0,), dtype=np.float64)
    residual_power_target: torch.Tensor | None = None
    if (
        str(args.residual_power_target_source) != "none"
        or float(args.residual_power_weight) != 0.0
        or float(args.residual_power_isotropy_weight) != 0.0
    ):
        residual_power_masks, residual_power_bin_edges = _radial_power_masks(
            int(args.eval_crop_size),
            int(args.residual_power_num_bins),
            float(args.residual_power_kmin),
            float(args.residual_power_kmax),
            device=device,
            dtype=torch_dtype,
        )
    if str(args.residual_power_target_source) == "dirty_eor":
        if residual_power_masks is None:
            raise AssertionError("residual_power_masks was not initialized")
        with torch.no_grad():
            residual_power_target = _radial_power(dirty_eor, residual_power_masks).detach()
    elif str(args.residual_power_target_source) == "npz":
        if residual_power_masks is None:
            raise AssertionError("residual_power_masks was not initialized")
        payload = np.load(args.residual_power_target_npz)
        if "power" not in payload:
            raise ValueError("--residual-power-target-npz must contain key 'power'")
        target_np = np.asarray(payload["power"], dtype=np.float64)
        if target_np.ndim == 1:
            target_np = np.broadcast_to(target_np[None, :], (len(freqs), target_np.shape[0])).copy()
        if target_np.shape != (len(freqs), int(args.residual_power_num_bins)):
            raise ValueError(
                "Residual power target shape mismatch: "
                f"expected {(len(freqs), int(args.residual_power_num_bins))}, got {target_np.shape}"
            )
        if "bin_edges" in payload:
            edges_np = np.asarray(payload["bin_edges"], dtype=np.float64)
            if edges_np.shape != residual_power_bin_edges.shape or not np.allclose(
                edges_np,
                residual_power_bin_edges,
                rtol=1e-6,
                atol=1e-12,
            ):
                raise ValueError("Residual power target bin_edges do not match requested bins")
        residual_power_target = torch.as_tensor(target_np, device=device, dtype=torch_dtype)
    elif str(args.residual_power_target_source) == "operator_white":
        if residual_power_masks is None:
            raise AssertionError("residual_power_masks was not initialized")
        if int(args.residual_power_operator_white_samples) <= 0:
            raise ValueError("--residual-power-operator-white-samples must be positive")
        rng = np.random.default_rng(int(args.residual_power_operator_white_seed))
        target_accum = torch.zeros(
            (len(freqs), int(args.residual_power_num_bins)),
            device=device,
            dtype=torch_dtype,
        )
        with torch.no_grad():
            for _sample_index in range(int(args.residual_power_operator_white_samples)):
                sky_np = rng.standard_normal(
                    (len(freqs), int(args.image_size), int(args.image_size))
                ).astype(np_dtype, copy=False)
                sky = torch.as_tensor(sky_np, device=device, dtype=torch_dtype)
                sky_rms = torch.sqrt(torch.mean(sky * sky, dim=(1, 2), keepdim=True))
                sky = sky / torch.clamp(sky_rms, min=1e-30)
                dirty_sample = forward_op(sky * k_to_jy[:, None, None])
                dirty_rms = torch.sqrt(torch.mean(dirty_sample * dirty_sample, dim=(1, 2), keepdim=True))
                dirty_sample = dirty_sample * (
                    data_error_rms / torch.clamp(dirty_rms, min=1e-30)
                )
                target_accum = target_accum + _radial_power(dirty_sample, residual_power_masks)
        residual_power_target = (target_accum / float(args.residual_power_operator_white_samples)).detach()
    if float(args.residual_freq_corr_weight) != 0.0 and str(args.residual_freq_corr_target_source) == "none":
        raise ValueError("--residual-freq-corr-weight requires --residual-freq-corr-target-source")
    if str(args.residual_freq_corr_target_source) == "npz" and args.residual_freq_corr_target_npz is None:
        raise ValueError("--residual-freq-corr-target-source npz requires --residual-freq-corr-target-npz")
    if not np.isfinite(float(args.residual_freq_corr_eps)) or float(args.residual_freq_corr_eps) <= 0.0:
        raise ValueError("--residual-freq-corr-eps must be finite and positive")
    if str(args.residual_freq_corr_loss_mode) == "upper_hinge" and (
        not np.isfinite(float(args.residual_freq_corr_hinge_target))
        or float(args.residual_freq_corr_hinge_target) <= 0.0
    ):
        raise ValueError(
            "--residual-freq-corr-hinge-target must be finite and positive "
            "for --residual-freq-corr-loss-mode upper_hinge"
        )
    if int(args.residual_freq_corr_operator_white_samples) <= 0:
        raise ValueError("--residual-freq-corr-operator-white-samples must be positive")
    residual_freq_corr_target: torch.Tensor | None = None
    if str(args.residual_freq_corr_target_source) == "dirty_eor":
        with torch.no_grad():
            residual_freq_corr_target = _frequency_corr(
                dirty_eor,
                float(args.residual_freq_corr_eps),
            ).detach()
    elif str(args.residual_freq_corr_target_source) == "npz":
        payload = np.load(args.residual_freq_corr_target_npz)
        corr_np = None
        for key in ("corr", "freq_corr", "frequency_corr"):
            if key in payload:
                corr_np = np.asarray(payload[key], dtype=np.float64)
                break
        if corr_np is None:
            raise ValueError(
                "--residual-freq-corr-target-npz must contain key "
                "'corr', 'freq_corr', or 'frequency_corr'"
            )
        if corr_np.shape != (len(freqs), len(freqs)):
            raise ValueError(
                "Residual frequency-correlation target shape mismatch: "
                f"expected {(len(freqs), len(freqs))}, got {corr_np.shape}"
            )
        residual_freq_corr_target = torch.as_tensor(corr_np, device=device, dtype=torch_dtype)
    elif str(args.residual_freq_corr_target_source) == "operator_white":
        rng = np.random.default_rng(int(args.residual_freq_corr_operator_white_seed))
        target_accum = torch.zeros(
            (len(freqs), len(freqs)),
            device=device,
            dtype=torch_dtype,
        )
        with torch.no_grad():
            for _sample_index in range(int(args.residual_freq_corr_operator_white_samples)):
                sky_np = rng.standard_normal(
                    (len(freqs), int(args.image_size), int(args.image_size))
                ).astype(np_dtype, copy=False)
                sky = torch.as_tensor(sky_np, device=device, dtype=torch_dtype)
                sky_rms = torch.sqrt(torch.mean(sky * sky, dim=(1, 2), keepdim=True))
                sky = sky / torch.clamp(sky_rms, min=1e-30)
                dirty_sample = forward_op(sky * k_to_jy[:, None, None])
                dirty_rms = torch.sqrt(torch.mean(dirty_sample * dirty_sample, dim=(1, 2), keepdim=True))
                dirty_sample = dirty_sample * (
                    data_error_rms / torch.clamp(dirty_rms, min=1e-30)
                )
                target_accum = target_accum + _frequency_corr(
                    dirty_sample,
                    float(args.residual_freq_corr_eps),
                )
        residual_freq_corr_target = (
            target_accum / float(args.residual_freq_corr_operator_white_samples)
        ).detach()
    coeff_scale = torch.as_tensor(float(args.coeff_scale_k), device=device, dtype=torch_dtype)

    coeff_shape = (int(args.cheb_degree) + 1, int(args.image_size), int(args.image_size))
    n_degree = int(args.cheb_degree) + 1
    if int(args.selection_prior_degree) < 0 or int(args.selection_prior_degree) >= n_degree:
        raise ValueError("--selection-prior-degree is out of range for --cheb-degree")
    if not np.isfinite(float(args.optimization_loss_scale)) or float(args.optimization_loss_scale) <= 0.0:
        raise ValueError("--optimization-loss-scale must be finite and positive")
    if (
        not np.isfinite(float(args.data_window_lower))
        or not np.isfinite(float(args.data_window_upper))
        or float(args.data_window_lower) < 0.0
        or float(args.data_window_upper) <= float(args.data_window_lower)
    ):
        raise ValueError("--data-window-lower/upper must be finite with 0 <= lower < upper")
    if not np.isfinite(float(args.selection_prior_target_rms)) or float(args.selection_prior_target_rms) < 0.0:
        raise ValueError("--selection-prior-target-rms must be finite and non-negative")
    if not np.isfinite(float(args.selection_prior_weight)) or float(args.selection_prior_weight) < 0.0:
        raise ValueError("--selection-prior-weight must be finite and non-negative")
    if (
        not np.isfinite(float(args.selection_data_window_lower))
        or not np.isfinite(float(args.selection_data_window_upper))
        or float(args.selection_data_window_lower) < 0.0
        or float(args.selection_data_window_upper) <= float(args.selection_data_window_lower)
    ):
        raise ValueError(
            "--selection-data-window-lower/upper must be finite with 0 <= lower < upper"
        )
    if (
        not np.isfinite(float(args.selection_data_window_penalty))
        or float(args.selection_data_window_penalty) < 0.0
    ):
        raise ValueError("--selection-data-window-penalty must be finite and non-negative")
    template_basis_paths = [Path(v) for v in _parse_list(args.template_basis_cheb_coeffs_list)]
    if str(args.template_basis_gain_mode) != "none" and not template_basis_paths:
        raise ValueError("--template-basis-gain-mode requires --template-basis-cheb-coeffs-list")
    if template_basis_paths and str(args.template_basis_gain_mode) == "none":
        raise ValueError("--template-basis-cheb-coeffs-list requires --template-basis-gain-mode")
    if not np.isfinite(float(args.coeff_prior_hinge_sigma)) or float(args.coeff_prior_hinge_sigma) < 0.0:
        raise ValueError("--coeff-prior-hinge-sigma must be finite and non-negative")
    if int(args.step_control_max_backtracks) < 0:
        raise ValueError("--step-control-max-backtracks must be non-negative")
    if not (0.0 < float(args.step_control_shrink) < 1.0):
        raise ValueError("--step-control-shrink must be in (0, 1)")
    if float(args.step_control_rel_tol) < 0.0 or float(args.step_control_abs_tol) < 0.0:
        raise ValueError("--step-control tolerances must be non-negative")
    if not np.isfinite(float(args.step_control_gradient_eps)) or float(args.step_control_gradient_eps) <= 0.0:
        raise ValueError("--step-control-gradient-eps must be finite and positive")
    template_basis_np: np.ndarray | None = None
    template_basis_rms_np = np.zeros((0, n_degree), dtype=np.float64)
    if template_basis_paths:
        basis_arrays = []
        for path in template_basis_paths:
            _check_template_basis_path(path, allow_suspicious=bool(args.allow_suspicious_template_basis))
            arr = np.asarray(fits.getdata(path), dtype=np.float64)
            if tuple(arr.shape) != coeff_shape:
                raise ValueError(f"template-basis coeff shape {arr.shape} from {path} does not match {coeff_shape}")
            if not np.any(np.isfinite(arr)):
                raise ValueError(f"template-basis contains no finite values: {path}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            basis_arrays.append(arr)
        template_basis_np = np.stack(basis_arrays, axis=0)
        template_basis_rms_np = np.sqrt(np.mean(template_basis_np * template_basis_np, axis=(2, 3)))
    n_template_basis = int(0 if template_basis_np is None else template_basis_np.shape[0])
    if n_template_basis > 0:
        if not np.isfinite(float(args.template_basis_param_scale)) or float(args.template_basis_param_scale) <= 0.0:
            raise ValueError("--template-basis-param-scale must be positive")
        if not np.isfinite(float(args.template_basis_param_max_abs)) or float(args.template_basis_param_max_abs) < 0.0:
            raise ValueError("--template-basis-param-max-abs must be non-negative")
        template_basis_gain_init_np = _parse_template_basis_gain_init(
            args.template_basis_gain_init,
            n_template_basis,
            n_degree,
            str(args.template_basis_gain_mode),
        )
        if np.any(~np.isfinite(template_basis_gain_init_np)):
            raise ValueError(f"Invalid template-basis gain init: {template_basis_gain_init_np.tolist()}")
        template_basis_coeffs = torch.as_tensor(template_basis_np, device=device, dtype=torch_dtype)
        template_basis_gain_init = torch.as_tensor(
            template_basis_gain_init_np.reshape(n_template_basis, n_degree, 1, 1),
            device=device,
            dtype=torch_dtype,
        )
    else:
        template_basis_gain_init_np = np.zeros((0, n_degree), dtype=np.float64)
        template_basis_coeffs = None
        template_basis_gain_init = torch.zeros((0, n_degree, 1, 1), device=device, dtype=torch_dtype)
    if args.prior_cheb_coeffs is not None:
        prior_coeffs_np = np.asarray(fits.getdata(args.prior_cheb_coeffs), dtype=np.float64)
        if tuple(prior_coeffs_np.shape) != coeff_shape:
            raise ValueError(f"prior coeff shape {prior_coeffs_np.shape} does not match expected {coeff_shape}")
        prior_rms_np = np.sqrt(np.mean(prior_coeffs_np * prior_coeffs_np, axis=(1, 2)))
        prior_default_scales = [
            float(v) if float(v) > 0.0 else float(args.coeff_scale_k)
            for v in np.asarray(prior_rms_np, dtype=np.float64)
        ]
        prior_abs_scales = _parse_degree_float_values(args.coeff_prior_scales_k, n_degree, prior_default_scales)
        prior_rel_scales = _parse_degree_float_values(args.coeff_prior_relative_scales, n_degree, [1.0] * n_degree)
        prior_scales_np = np.asarray(
            [float(a) * float(r) for a, r in zip(prior_abs_scales, prior_rel_scales)],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(prior_scales_np)) or np.any(prior_scales_np <= 0.0):
            raise ValueError(f"Invalid coefficient-prior scales: {prior_scales_np.tolist()}")
        prior_coeffs = torch.as_tensor(prior_coeffs_np, device=device, dtype=torch_dtype)
        prior_scales = torch.as_tensor(prior_scales_np.reshape(n_degree, 1, 1), device=device, dtype=torch_dtype)
        prior_mode = "external_fits"
    else:
        prior_scales_np = np.ones((n_degree,), dtype=np.float64)
        prior_rel_scales = [1.0] * n_degree
        prior_coeffs = None
        prior_scales = torch.ones((n_degree, 1, 1), device=device, dtype=torch_dtype)
        prior_mode = "none"
    if str(args.save_selection_mode) in {"data_prior_boundary", "data_window_min_prior"} and prior_coeffs is None:
        raise ValueError(
            f"--save-selection-mode {args.save_selection_mode} requires --prior-cheb-coeffs"
        )
    optimize_degrees_spec = str(args.optimize_degrees or "").strip().lower()
    freeze_full_pixel_degrees = optimize_degrees_spec in {"none", "off", "false", "freeze", "frozen"}
    optimize_degrees = _parse_ints(args.optimize_degrees)
    if not optimize_degrees and not freeze_full_pixel_degrees:
        optimize_degrees = list(range(n_degree))
    bad_degrees = [int(v) for v in optimize_degrees if int(v) < 0 or int(v) >= n_degree]
    if bad_degrees:
        raise ValueError(f"--optimize-degrees contains out-of-range entries: {bad_degrees}")
    optimize_mask_np = np.zeros((n_degree, 1, 1), dtype=np.float32)
    for degree_index in optimize_degrees:
        optimize_mask_np[int(degree_index), 0, 0] = 1.0
    optimize_mask = torch.as_tensor(optimize_mask_np, device=device, dtype=torch_dtype)

    def run_fit_once(run_out_dir: Path, init_path: Path | None, run_label: str) -> Dict[str, Any]:
        run_out_dir.mkdir(parents=True, exist_ok=True)
        if init_path is not None:
            init_np = np.asarray(fits.getdata(init_path), dtype=np.float64)
            if tuple(init_np.shape) != coeff_shape:
                raise ValueError(f"init coeff shape {init_np.shape} does not match expected {coeff_shape}")
            init_mode = "oracle_or_debug_fits"
        else:
            init_np = np.zeros(coeff_shape, dtype=np.float64)
            init_mode = "zero"

        base_coeffs = torch.as_tensor(init_np, device=device, dtype=torch_dtype)
        base_rms_np = np.sqrt(np.mean(init_np * init_np, axis=(1, 2)))
        base_default_scales = [
            float(v) if float(v) > 0.0 else float(args.coeff_scale_k)
            for v in np.asarray(base_rms_np, dtype=np.float64)
        ]
        degree_abs_scales = _parse_degree_float_values(args.degree_param_scales_k, n_degree, base_default_scales)
        degree_rel_scales = _parse_degree_float_values(args.degree_param_relative_scales, n_degree, [1.0] * n_degree)
        degree_param_scales_np = np.asarray(
            [float(a) * float(r) for a, r in zip(degree_abs_scales, degree_rel_scales)],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(degree_param_scales_np)) or np.any(degree_param_scales_np <= 0.0):
            raise ValueError(f"Invalid degree parameter scales: {degree_param_scales_np.tolist()}")
        degree_param_scales = torch.as_tensor(
            degree_param_scales_np.reshape(n_degree, 1, 1),
            device=device,
            dtype=torch_dtype,
        )
        degree_param_max_abs = _parse_degree_float_values(args.degree_param_max_abs, n_degree, [0.0] * n_degree)
        degree_param_max_abs_np = np.asarray(degree_param_max_abs, dtype=np.float64)
        if np.any(~np.isfinite(degree_param_max_abs_np)) or np.any(degree_param_max_abs_np < 0.0):
            raise ValueError(f"Invalid degree parameter clamps: {degree_param_max_abs_np.tolist()}")
        trust_abs_scales = _parse_degree_float_values(args.coeff_delta_trust_scales_k, n_degree, base_default_scales)
        trust_rel_scales = _parse_degree_float_values(
            args.coeff_delta_trust_relative_scales,
            n_degree,
            [1.0] * n_degree,
        )
        trust_scales_np = np.asarray(
            [float(a) * float(r) for a, r in zip(trust_abs_scales, trust_rel_scales)],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(trust_scales_np)) or np.any(trust_scales_np <= 0.0):
            raise ValueError(f"Invalid coefficient-delta trust scales: {trust_scales_np.tolist()}")
        trust_scales = torch.as_tensor(
            trust_scales_np.reshape(n_degree, 1, 1),
            device=device,
            dtype=torch_dtype,
        )
        lr_schedule = _parse_scalar_schedule(args.lr_schedule, name="lr")
        data_window_lower_schedule = _parse_scalar_schedule(
            args.data_window_lower_schedule,
            name="data_window_lower",
        )
        data_window_upper_schedule = _parse_scalar_schedule(
            args.data_window_upper_schedule,
            name="data_window_upper",
        )
        residual_cheb_weight_schedule = _parse_scalar_schedule(
            args.residual_cheb_weight_schedule,
            name="residual_cheb_weight",
        )
        coeff_prior_weight_schedule = _parse_scalar_schedule(
            args.coeff_prior_weight_schedule,
            name="coeff_prior_weight",
        )
        coeff_prior_hinge_sigma_schedule = _parse_scalar_schedule(
            args.coeff_prior_hinge_sigma_schedule,
            name="coeff_prior_hinge_sigma",
        )

        c0_correction_param: torch.nn.Parameter | None = None
        c0_correction_degrees: List[int] = []
        c0_correction_block_list: List[int] = []
        c0_correction_scales_np = np.zeros((0,), dtype=np.float64)
        c0_correction_scale = 0.0
        if str(args.c0_correction_mode) != "none":
            if int(args.cheb_degree) < 0:
                raise ValueError("C0 correction requires at least one Cheb degree")
            c0_correction_degrees = _parse_ints(args.c0_correction_degrees)
            if not c0_correction_degrees:
                raise ValueError("--c0-correction-degrees must list at least one degree when correction is enabled")
            bad_correction_degrees = [
                int(v) for v in c0_correction_degrees if int(v) < 0 or int(v) >= n_degree
            ]
            if bad_correction_degrees:
                raise ValueError(f"--c0-correction-degrees contains out-of-range entries: {bad_correction_degrees}")
            explicit_c0_correction_scales = _parse_floats(args.c0_correction_scales_k)
            if explicit_c0_correction_scales:
                if len(explicit_c0_correction_scales) == 1:
                    c0_correction_scales_np = np.asarray(
                        [float(explicit_c0_correction_scales[0]) for _ in c0_correction_degrees],
                        dtype=np.float64,
                    )
                elif len(explicit_c0_correction_scales) == len(c0_correction_degrees):
                    c0_correction_scales_np = np.asarray(explicit_c0_correction_scales, dtype=np.float64)
                else:
                    raise ValueError(
                        "--c0-correction-scales-k expects one scalar or one value per selected "
                        f"correction degree ({len(c0_correction_degrees)}), got "
                        f"{len(explicit_c0_correction_scales)}"
                    )
            elif float(args.c0_correction_scale_k) > 0.0:
                c0_correction_scales_np = np.asarray(
                    [float(args.c0_correction_scale_k) for _ in c0_correction_degrees],
                    dtype=np.float64,
                )
            else:
                c0_correction_scales_np = np.asarray(
                    [
                        float(base_default_scales[int(degree_index)]) * float(args.c0_correction_relative_scale)
                        for degree_index in c0_correction_degrees
                    ],
                    dtype=np.float64,
                )
            if np.any(~np.isfinite(c0_correction_scales_np)) or np.any(c0_correction_scales_np <= 0.0):
                raise ValueError(f"Invalid C0 correction scales: {c0_correction_scales_np.tolist()}")
            c0_correction_scale = float(c0_correction_scales_np[0])
            n_correction_degrees = int(len(c0_correction_degrees))
            if str(args.c0_correction_mode) == "global":
                c0_correction_param = torch.nn.Parameter(
                    torch.zeros((n_correction_degrees, 1, 1), device=device, dtype=torch_dtype)
                )
            elif str(args.c0_correction_mode) == "block":
                if int(args.c0_correction_blocks) <= 0:
                    raise ValueError("--c0-correction-blocks must be positive")
                c0_correction_block_list = [int(args.c0_correction_blocks)]
                c0_correction_param = torch.nn.Parameter(
                    torch.zeros(
                        (n_correction_degrees, 1, int(args.c0_correction_blocks), int(args.c0_correction_blocks)),
                        device=device,
                        dtype=torch_dtype,
                    )
                )
            elif str(args.c0_correction_mode) == "multiblock":
                c0_correction_block_list = _parse_ints(args.c0_correction_block_list)
                if not c0_correction_block_list:
                    c0_correction_block_list = [int(args.c0_correction_blocks)]
                bad_blocks = [
                    int(v)
                    for v in c0_correction_block_list
                    if int(v) <= 0 or int(v) > int(args.image_size)
                ]
                if bad_blocks:
                    raise ValueError(
                        "--c0-correction-block-list values must be positive and <= image size: "
                        f"{bad_blocks}"
                    )
                n_multiblock_param = int(
                    n_correction_degrees * sum(int(v) * int(v) for v in c0_correction_block_list)
                )
                c0_correction_param = torch.nn.Parameter(
                    torch.zeros((n_multiblock_param,), device=device, dtype=torch_dtype)
                )
            elif str(args.c0_correction_mode) == "dct":
                if int(args.c0_correction_dct_size) <= 0:
                    raise ValueError("--c0-correction-dct-size must be positive")
                if int(args.c0_correction_dct_size) > int(args.image_size):
                    raise ValueError("--c0-correction-dct-size must not exceed --image-size")
                c0_correction_param = torch.nn.Parameter(
                    torch.zeros(
                        (n_correction_degrees, int(args.c0_correction_dct_size), int(args.c0_correction_dct_size)),
                        device=device,
                        dtype=torch_dtype,
                    )
                )
            else:
                raise ValueError(f"Unsupported C0 correction mode: {args.c0_correction_mode}")

        c0_correction_stages: List[C0CorrectionStage] = []
        if str(args.c0_correction_stage_spec or "").strip():
            if c0_correction_param is None:
                raise ValueError("--c0-correction-stage-spec requires --c0-correction-mode other than none")
            if str(args.c0_correction_mode) != "multiblock":
                raise ValueError("--c0-correction-stage-spec currently supports --c0-correction-mode multiblock only")
            c0_correction_stages = _parse_c0_correction_stage_spec(
                args.c0_correction_stage_spec,
                correction_degrees=c0_correction_degrees,
                block_list=c0_correction_block_list,
            )
        c0_correction_stage_mask = (
            torch.ones_like(c0_correction_param) if c0_correction_param is not None else None
        )
        current_c0_stage: C0CorrectionStage | None = None
        current_c0_stage_index = -1
        current_c0_stage_label = "all"
        current_c0_stage_degrees: Tuple[int, ...] = tuple(int(v) for v in c0_correction_degrees)
        current_c0_stage_blocks: Tuple[int, ...] = tuple(int(v) for v in c0_correction_block_list)
        c0_stage_active_param_count = (
            int(c0_correction_param.numel()) if c0_correction_param is not None else 0
        )

        template_gain_param: torch.nn.Parameter | None = None
        template_gain_init_np = np.ones((n_degree,), dtype=np.float64)
        if str(args.template_gain_mode) != "none":
            if init_path is None:
                raise ValueError("--template-gain-mode requires a nonzero --init-cheb-coeffs template")
            if not np.any(base_rms_np > 0.0):
                raise ValueError("--template-gain-mode requires a nonzero coefficient template")
            if not np.isfinite(float(args.template_gain_param_scale)) or float(args.template_gain_param_scale) <= 0.0:
                raise ValueError("--template-gain-param-scale must be positive")
            if not np.isfinite(float(args.template_gain_param_max_abs)) or float(args.template_gain_param_max_abs) < 0.0:
                raise ValueError("--template-gain-param-max-abs must be non-negative")
            template_gain_init_np = np.asarray(
                _parse_degree_float_values(args.template_gain_init, n_degree, [1.0] * n_degree),
                dtype=np.float64,
            )
            if np.any(~np.isfinite(template_gain_init_np)):
                raise ValueError(f"Invalid template gain init: {template_gain_init_np.tolist()}")
            if str(args.template_gain_mode) == "global":
                template_gain_param = torch.nn.Parameter(torch.zeros((1, 1, 1), device=device, dtype=torch_dtype))
            elif str(args.template_gain_mode) == "per_degree":
                template_gain_param = torch.nn.Parameter(torch.zeros((n_degree, 1, 1), device=device, dtype=torch_dtype))
            else:
                raise ValueError(f"Unsupported template gain mode: {args.template_gain_mode}")
        template_gain_init = torch.as_tensor(
            template_gain_init_np.reshape(n_degree, 1, 1),
            device=device,
            dtype=torch_dtype,
        )

        def materialize_template_gain() -> torch.Tensor:
            if template_gain_param is None:
                return torch.ones((n_degree, 1, 1), device=device, dtype=torch_dtype)
            if str(args.template_gain_mode) == "global":
                param = template_gain_param.reshape(1, 1, 1).expand(n_degree, 1, 1)
            else:
                param = template_gain_param
            return template_gain_init + float(args.template_gain_param_scale) * param

        def apply_template_gain(coeffs: torch.Tensor) -> torch.Tensor:
            if template_gain_param is None:
                return coeffs
            return coeffs * materialize_template_gain()

        template_basis_param: torch.nn.Parameter | None = None
        if n_template_basis > 0:
            if str(args.template_basis_gain_mode) == "global":
                template_basis_param = torch.nn.Parameter(torch.zeros((1, 1, 1, 1), device=device, dtype=torch_dtype))
            elif str(args.template_basis_gain_mode) == "per_basis":
                template_basis_param = torch.nn.Parameter(
                    torch.zeros((n_template_basis, 1, 1, 1), device=device, dtype=torch_dtype)
                )
            elif str(args.template_basis_gain_mode) == "per_basis_degree":
                template_basis_param = torch.nn.Parameter(
                    torch.zeros((n_template_basis, n_degree, 1, 1), device=device, dtype=torch_dtype)
                )
            else:
                raise ValueError(f"Unsupported template basis gain mode: {args.template_basis_gain_mode}")

        def materialize_template_basis_gain() -> torch.Tensor:
            if template_basis_param is None:
                return torch.zeros((0, n_degree, 1, 1), device=device, dtype=torch_dtype)
            if str(args.template_basis_gain_mode) == "global":
                param = template_basis_param.reshape(1, 1, 1, 1).expand(n_template_basis, n_degree, 1, 1)
            elif str(args.template_basis_gain_mode) == "per_basis":
                param = template_basis_param.expand(n_template_basis, n_degree, 1, 1)
            else:
                param = template_basis_param
            return template_basis_gain_init + float(args.template_basis_param_scale) * param

        def materialize_template_basis_coeffs() -> torch.Tensor:
            if template_basis_coeffs is None:
                return torch.zeros(coeff_shape, device=device, dtype=torch_dtype)
            gains = materialize_template_basis_gain()
            return torch.sum(template_basis_coeffs * gains, dim=0)

        c0_correction_degrees_tensor = torch.as_tensor(
            c0_correction_degrees,
            device=device,
            dtype=torch.long,
        )
        c0_correction_scales = torch.as_tensor(
            c0_correction_scales_np.reshape(-1, 1, 1),
            device=device,
            dtype=torch_dtype,
        )

        def materialize_c0_correction() -> torch.Tensor | None:
            if c0_correction_param is None:
                return None
            if str(args.c0_correction_mode) == "global":
                correction = c0_correction_param.expand(
                    int(len(c0_correction_degrees)),
                    int(args.image_size),
                    int(args.image_size),
                )
                return correction * c0_correction_scales
            if str(args.c0_correction_mode) == "dct":
                dct_basis = _dct_rms_normalized_basis(
                    int(args.c0_correction_dct_size),
                    int(args.image_size),
                    device=device,
                    dtype=torch_dtype,
                )
                correction = torch.einsum("dab,ay,bx->dyx", c0_correction_param, dct_basis, dct_basis)
                return correction * c0_correction_scales
            if str(args.c0_correction_mode) == "multiblock":
                offset = 0
                flat_param = c0_correction_param
                if c0_correction_stage_mask is not None:
                    flat_param = flat_param * c0_correction_stage_mask
                correction_sum = torch.zeros(
                    (
                        int(len(c0_correction_degrees)),
                        int(args.image_size),
                        int(args.image_size),
                    ),
                    device=device,
                    dtype=torch_dtype,
                )
                for block_count in c0_correction_block_list:
                    n_values = int(len(c0_correction_degrees)) * int(block_count) * int(block_count)
                    block_param = flat_param[offset : offset + n_values].reshape(
                        int(len(c0_correction_degrees)),
                        1,
                        int(block_count),
                        int(block_count),
                    )
                    offset += n_values
                    correction_sum = correction_sum + F.interpolate(
                        block_param,
                        size=(int(args.image_size), int(args.image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                return correction_sum * c0_correction_scales
            correction = F.interpolate(
                c0_correction_param,
                size=(int(args.image_size), int(args.image_size)),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            return correction * c0_correction_scales

        def add_c0_correction(coeffs: torch.Tensor) -> torch.Tensor:
            c0_delta = materialize_c0_correction()
            if c0_delta is None:
                return coeffs
            delta_cube = torch.zeros_like(coeffs)
            delta_cube.index_copy_(0, c0_correction_degrees_tensor, c0_delta)
            return coeffs + delta_cube

        if str(args.param_mode) == "scaled_delta":
            params = torch.nn.Parameter(torch.zeros(coeff_shape, device=device, dtype=torch_dtype))

            def materialize_coeffs(params: torch.nn.Parameter = params) -> torch.Tensor:
                coeffs = apply_template_gain(base_coeffs + optimize_mask * degree_param_scales * params)
                return add_c0_correction(coeffs + materialize_template_basis_coeffs())

            optimizer_params = [params]
        else:
            params = torch.nn.Parameter(base_coeffs.detach().clone())

            def materialize_coeffs(params: torch.nn.Parameter = params) -> torch.Tensor:
                coeffs = apply_template_gain(base_coeffs + optimize_mask * (params - base_coeffs))
                return add_c0_correction(coeffs + materialize_template_basis_coeffs())

            optimizer_params = [params]
        if template_gain_param is not None:
            optimizer_params.append(template_gain_param)
        if template_basis_param is not None:
            optimizer_params.append(template_basis_param)
        if c0_correction_param is not None:
            optimizer_params.append(c0_correction_param)

        print(
            json.dumps(
                {
                    "event": "fit_run_start",
                    "run_label": str(run_label),
                    "out_dir": str(run_out_dir),
                    "init_cheb_coeffs": str(init_path) if init_path is not None else None,
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        opt = _make_optimizer(args, optimizer_params)
        use_lbfgs = str(args.optimizer_name).lower() == "lbfgs"
        if use_lbfgs and str(args.step_control_mode) != "none":
            raise ValueError("--optimizer-name lbfgs is incompatible with --step-control-mode")
        if use_lbfgs and float(args.grad_clip_norm) > 0.0:
            raise ValueError("--optimizer-name lbfgs does not support --grad-clip-norm in this script")
        history: List[Dict[str, Any]] = []
        best_metric = float("inf")
        best_state: torch.Tensor | None = None
        best_iter = -1
        selected_metric = float("inf")
        selected_state: torch.Tensor | None = None
        selected_iter = -1
        last_total_grad_norm = float("nan")
        last_step_control = {
            "accepted": None,
            "source": "none",
            "alpha": float("nan"),
            "attempts": 0,
            "old_loss": float("nan"),
            "candidate_loss": float("nan"),
        }

        def project_optimizer_params(params: torch.nn.Parameter = params) -> None:
            with torch.no_grad():
                if str(args.param_mode) == "scaled_delta" and np.any(degree_param_max_abs_np > 0.0):
                    for degree_index, max_abs in enumerate(degree_param_max_abs_np.tolist()):
                        if float(max_abs) > 0.0:
                            params[int(degree_index)].clamp_(min=-float(max_abs), max=float(max_abs))
                if c0_correction_param is not None and float(args.c0_correction_param_max_abs) > 0.0:
                    c0_correction_param.clamp_(
                        min=-float(args.c0_correction_param_max_abs),
                        max=float(args.c0_correction_param_max_abs),
                    )
                if c0_correction_param is not None and c0_correction_stage_mask is not None:
                    c0_correction_param.mul_(c0_correction_stage_mask)
                if template_gain_param is not None and float(args.template_gain_param_max_abs) > 0.0:
                    template_gain_param.clamp_(
                        min=-float(args.template_gain_param_max_abs),
                        max=float(args.template_gain_param_max_abs),
                    )
                if template_basis_param is not None and float(args.template_basis_param_max_abs) > 0.0:
                    template_basis_param.clamp_(
                        min=-float(args.template_basis_param_max_abs),
                        max=float(args.template_basis_param_max_abs),
                    )

        def set_c0_correction_stage_for_iter(
            iteration: int,
            *,
            force: bool = False,
            opt: torch.optim.Optimizer = opt,
        ) -> None:
            nonlocal c0_correction_stage_mask
            nonlocal current_c0_stage
            nonlocal current_c0_stage_index
            nonlocal current_c0_stage_label
            nonlocal current_c0_stage_degrees
            nonlocal current_c0_stage_blocks
            nonlocal c0_stage_active_param_count

            if c0_correction_param is None or not c0_correction_stages:
                return
            target_stage = _c0_correction_stage_for_iter(c0_correction_stages, int(iteration))
            target_stage_index = -1 if target_stage is None else int(target_stage.index)
            if not force and target_stage_index == int(current_c0_stage_index):
                return

            previous_stage_index = int(current_c0_stage_index)
            reset_optimizer_state = False
            if (
                not force
                and previous_stage_index != target_stage_index
                and int(iteration) > 0
                and bool(args.c0_correction_stage_reset_optimizer)
            ):
                opt.state.pop(c0_correction_param, None)
                reset_optimizer_state = True

            current_c0_stage = target_stage
            current_c0_stage_index = int(target_stage_index)
            if target_stage is None:
                current_c0_stage_label = "all"
                current_c0_stage_degrees = tuple(int(v) for v in c0_correction_degrees)
                current_c0_stage_blocks = tuple(int(v) for v in c0_correction_block_list)
            else:
                current_c0_stage_label = str(target_stage.label)
                current_c0_stage_degrees = tuple(int(v) for v in target_stage.degrees)
                current_c0_stage_blocks = tuple(int(v) for v in target_stage.blocks)
            c0_correction_stage_mask = _make_multiblock_c0_stage_mask(
                stage=target_stage,
                correction_degrees=c0_correction_degrees,
                block_list=c0_correction_block_list,
                like=c0_correction_param,
            )
            c0_stage_active_param_count = int(torch.count_nonzero(c0_correction_stage_mask).detach().cpu())
            project_optimizer_params()
            print(
                json.dumps(
                    {
                        "event": "c0_correction_stage_change",
                        "iter": int(iteration),
                        "previous_stage_index": int(previous_stage_index),
                        "stage_index": int(current_c0_stage_index),
                        "stage_label": str(current_c0_stage_label),
                        "active_degrees": [int(v) for v in current_c0_stage_degrees],
                        "active_blocks": [int(v) for v in current_c0_stage_blocks],
                        "active_param_count": int(c0_stage_active_param_count),
                        "optimizer_state_reset": bool(reset_optimizer_state),
                        "time_utc": _now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if c0_correction_stages:
            print(
                json.dumps(
                    {
                        "event": "c0_correction_stage_schedule",
                        "stage_spec": str(args.c0_correction_stage_spec),
                        "stage_reset_optimizer": bool(args.c0_correction_stage_reset_optimizer),
                        "stages": [_c0_correction_stage_to_dict(stage) for stage in c0_correction_stages],
                        "time_utc": _now(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def evaluate_objective(needs_grad: bool, iteration: int) -> Dict[str, torch.Tensor]:
            with torch.set_grad_enabled(needs_grad):
                current_data_window_lower = _scalar_schedule_value(
                    data_window_lower_schedule,
                    int(iteration),
                    float(args.data_window_lower),
                )
                current_data_window_upper = _scalar_schedule_value(
                    data_window_upper_schedule,
                    int(iteration),
                    float(args.data_window_upper),
                )
                current_residual_cheb_weight = _scalar_schedule_value(
                    residual_cheb_weight_schedule,
                    int(iteration),
                    float(args.residual_cheb_weight),
                )
                current_coeff_prior_weight = _scalar_schedule_value(
                    coeff_prior_weight_schedule,
                    int(iteration),
                    float(args.coeff_prior_weight),
                )
                current_coeff_prior_hinge_sigma = _scalar_schedule_value(
                    coeff_prior_hinge_sigma_schedule,
                    int(iteration),
                    float(args.coeff_prior_hinge_sigma),
                )
                coeffs_current = materialize_coeffs()
                fg_k = torch.einsum("fd,dxy->fxy", cheb, coeffs_current)
                fg_flux = fg_k * k_to_jy[:, None, None]
                pred_fg = forward_op(fg_flux)
                est_eor = dirty_total - pred_fg
                data_residual_rms = torch.sqrt(torch.mean((pred_fg - dirty_total) ** 2))
                data_residual_over_error = data_residual_rms / torch.clamp(data_error_rms, min=1e-30)
                if str(args.data_loss_mode) == "mse":
                    data_loss = data_residual_over_error * data_residual_over_error
                elif str(args.data_loss_mode) == "upper_hinge":
                    data_upper = torch.as_tensor(
                        float(current_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    data_loss = torch.relu(data_residual_over_error - data_upper) ** 2
                elif str(args.data_loss_mode) == "target_log":
                    data_loss = torch.log(torch.clamp(data_residual_over_error, min=1e-12)) ** 2
                elif str(args.data_loss_mode) == "window_hinge":
                    data_lower = torch.as_tensor(
                        float(current_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    )
                    data_upper = torch.as_tensor(
                        float(current_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    data_loss = (
                        torch.relu(data_lower - data_residual_over_error) ** 2
                        + torch.relu(data_residual_over_error - data_upper) ** 2
                    )
                else:
                    raise ValueError(f"Unsupported data loss mode: {args.data_loss_mode}")
                resid_flat = est_eor.reshape(len(freqs), -1)
                smooth_resid = q_cheb @ (q_cheb.transpose(0, 1) @ resid_flat)
                residual_cheb_loss = torch.mean(smooth_resid * smooth_resid) / torch.clamp(
                    residual_cheb_norm_rms * residual_cheb_norm_rms,
                    min=1e-30,
                )
                if residual_spatial_lowpass_basis is not None:
                    lowpass_coeff = torch.einsum(
                        "ky,fxy,lx->fkl",
                        residual_spatial_lowpass_basis,
                        est_eor,
                        residual_spatial_lowpass_basis,
                    ) / float(int(args.eval_crop_size) * int(args.eval_crop_size))
                    if bool(args.residual_spatial_lowpass_skip_dc):
                        lowpass_coeff = lowpass_coeff.clone()
                        lowpass_coeff[:, 0, 0] = 0.0
                    residual_spatial_lowpass_loss = torch.mean(lowpass_coeff * lowpass_coeff) / torch.clamp(
                        residual_spatial_lowpass_norm_rms * residual_spatial_lowpass_norm_rms,
                        min=1e-30,
                    )
                else:
                    residual_spatial_lowpass_loss = torch.zeros((), device=device, dtype=torch_dtype)
                if float(args.residual_rms_target) > 0.0 and float(args.residual_rms_weight) != 0.0:
                    target_rms = torch.as_tensor(float(args.residual_rms_target), device=device, dtype=torch_dtype)
                    resid_rms = torch.sqrt(torch.mean(est_eor * est_eor))
                    residual_rms_loss = torch.log(
                        torch.clamp(resid_rms / torch.clamp(target_rms, min=1e-30), min=1e-12)
                    ) ** 2
                else:
                    residual_rms_loss = torch.zeros((), device=device, dtype=torch_dtype)
                if residual_power_target is not None and float(args.residual_power_weight) != 0.0:
                    if residual_power_masks is None:
                        raise AssertionError("residual_power_masks was not initialized")
                    residual_power = _radial_power(est_eor, residual_power_masks)
                    eps = torch.as_tensor(float(args.residual_power_log_eps), device=device, dtype=torch_dtype)
                    log_delta = torch.log(torch.clamp(residual_power, min=eps)) - torch.log(
                        torch.clamp(residual_power_target, min=eps)
                    )
                    tolerance = torch.as_tensor(
                        float(args.residual_power_log_tolerance),
                        device=device,
                        dtype=torch_dtype,
                    )
                    if str(args.residual_power_loss_mode) == "mse":
                        residual_power_loss = torch.mean(log_delta * log_delta)
                    elif str(args.residual_power_loss_mode) == "log_upper_hinge":
                        residual_power_loss = torch.mean(torch.relu(log_delta - tolerance) ** 2)
                    elif str(args.residual_power_loss_mode) == "log_window_hinge":
                        residual_power_loss = torch.mean(torch.relu(torch.abs(log_delta) - tolerance) ** 2)
                    else:
                        raise ValueError(f"Unsupported residual power loss mode: {args.residual_power_loss_mode}")
                else:
                    residual_power_loss = torch.zeros((), device=device, dtype=torch_dtype)
                if float(args.residual_power_isotropy_weight) != 0.0:
                    if residual_power_masks is None:
                        raise AssertionError("residual_power_masks was not initialized")
                    residual_power_isotropy_loss = _radial_log_power_angular_variance(
                        est_eor,
                        residual_power_masks,
                        float(args.residual_power_log_eps),
                    )
                else:
                    residual_power_isotropy_loss = torch.zeros((), device=device, dtype=torch_dtype)
                if residual_freq_corr_target is not None and float(args.residual_freq_corr_weight) != 0.0:
                    residual_freq_corr = _frequency_corr(
                        est_eor,
                        float(args.residual_freq_corr_eps),
                    )
                    residual_freq_corr_raw_loss = torch.mean(
                        (residual_freq_corr - residual_freq_corr_target) ** 2
                    )
                    if str(args.residual_freq_corr_loss_mode) == "mse":
                        residual_freq_corr_loss = residual_freq_corr_raw_loss
                    elif str(args.residual_freq_corr_loss_mode) == "upper_hinge":
                        target = torch.as_tensor(
                            float(args.residual_freq_corr_hinge_target),
                            device=device,
                            dtype=torch_dtype,
                        )
                        residual_freq_corr_loss = torch.relu(
                            residual_freq_corr_raw_loss / torch.clamp(target, min=1e-30) - 1.0
                        ) ** 2
                    else:
                        raise ValueError(f"Unsupported residual freq-corr loss mode: {args.residual_freq_corr_loss_mode}")
                else:
                    residual_freq_corr_raw_loss = torch.zeros((), device=device, dtype=torch_dtype)
                    residual_freq_corr_loss = torch.zeros((), device=device, dtype=torch_dtype)
                fg_l2_loss = torch.mean((fg_k / coeff_scale) ** 2)
                fg_pos_loss = torch.mean((torch.relu(-fg_k) / coeff_scale) ** 2)
                tv_loss = _spatial_tv(coeffs_current / coeff_scale)
                if prior_coeffs is not None:
                    coeff_prior_normed = (coeffs_current - prior_coeffs) / prior_scales
                    coeff_prior_mse_by_degree = torch.mean(coeff_prior_normed.double() ** 2, dim=(1, 2))
                    coeff_prior_rms_by_degree = torch.sqrt(torch.clamp(coeff_prior_mse_by_degree.detach(), min=0.0))
                else:
                    coeff_prior_normed = torch.zeros_like(coeffs_current)
                    coeff_prior_mse_by_degree = torch.zeros((n_degree,), device=device, dtype=torch.float64)
                    coeff_prior_rms_by_degree = torch.zeros((n_degree,), device=device, dtype=torch.float64)
                if prior_coeffs is not None and float(current_coeff_prior_weight) != 0.0:
                    if str(args.coeff_prior_loss_mode) == "mse":
                        coeff_prior_loss = torch.mean(coeff_prior_normed * coeff_prior_normed)
                    elif str(args.coeff_prior_loss_mode) == "global_rms_hinge":
                        # Compare squared RMS values to avoid the singular
                        # derivative of sqrt() when the current state exactly
                        # equals the prior.
                        hinge = torch.relu(
                            coeff_prior_mse_by_degree.to(device=device, dtype=torch_dtype)
                            - float(current_coeff_prior_hinge_sigma) ** 2
                        )
                        coeff_prior_loss = torch.mean(hinge * hinge)
                    else:
                        raise ValueError(f"Unsupported coefficient-prior loss mode: {args.coeff_prior_loss_mode}")
                else:
                    coeff_prior_loss = torch.zeros((), device=device, dtype=torch_dtype)
                coeff_delta = coeffs_current - base_coeffs
                coeff_delta_trust_loss = torch.mean((coeff_delta / trust_scales) ** 2)
                loss = (
                    float(args.data_weight) * data_loss
                    + float(current_residual_cheb_weight) * residual_cheb_loss
                    + float(args.residual_spatial_lowpass_weight) * residual_spatial_lowpass_loss
                    + float(args.residual_rms_weight) * residual_rms_loss
                    + float(args.residual_power_weight) * residual_power_loss
                    + float(args.residual_power_isotropy_weight) * residual_power_isotropy_loss
                    + float(args.residual_freq_corr_weight) * residual_freq_corr_loss
                    + float(args.fg_l2_weight) * fg_l2_loss
                    + float(args.fg_positivity_weight) * fg_pos_loss
                    + float(args.coeff_tv_weight) * tv_loss
                    + float(current_coeff_prior_weight) * coeff_prior_loss
                    + float(args.coeff_delta_trust_weight) * coeff_delta_trust_loss
                )
                optimization_loss = loss * float(args.optimization_loss_scale)
                return {
                    "coeffs_current": coeffs_current,
                    "pred_fg": pred_fg,
                    "est_eor": est_eor,
                    "loss": loss,
                    "optimization_loss": optimization_loss,
                    "data_loss": data_loss,
                    "data_residual_over_error": data_residual_over_error,
                    "residual_cheb_loss": residual_cheb_loss,
                    "residual_spatial_lowpass_loss": residual_spatial_lowpass_loss,
                    "residual_rms_loss": residual_rms_loss,
                    "residual_power_loss": residual_power_loss,
                    "residual_power_isotropy_loss": residual_power_isotropy_loss,
                    "residual_freq_corr_raw_loss": residual_freq_corr_raw_loss,
                    "residual_freq_corr_loss": residual_freq_corr_loss,
                    "fg_l2_loss": fg_l2_loss,
                    "fg_pos_loss": fg_pos_loss,
                    "tv_loss": tv_loss,
                    "coeff_prior_loss": coeff_prior_loss,
                    "coeff_prior_rms_by_degree": coeff_prior_rms_by_degree,
                    "coeff_delta_trust_loss": coeff_delta_trust_loss,
                    "current_data_window_lower": torch.as_tensor(
                        float(current_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    ),
                    "current_data_window_upper": torch.as_tensor(
                        float(current_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    ),
                    "current_residual_cheb_weight": torch.as_tensor(
                        float(current_residual_cheb_weight),
                        device=device,
                        dtype=torch_dtype,
                    ),
                    "current_coeff_prior_weight": torch.as_tensor(
                        float(current_coeff_prior_weight),
                        device=device,
                        dtype=torch_dtype,
                    ),
                    "current_coeff_prior_hinge_sigma": torch.as_tensor(
                        float(current_coeff_prior_hinge_sigma),
                        device=device,
                        dtype=torch_dtype,
                    ),
                }

        for it in range(int(args.num_iters) + 1):
            set_c0_correction_stage_for_iter(int(it), force=int(it) == 0)
            current_lr = _scalar_schedule_value(lr_schedule, int(it), float(args.lr))
            for group in opt.param_groups:
                group["lr"] = float(current_lr)
            opt.zero_grad(set_to_none=True)
            needs_grad = it < int(args.num_iters)
            objective = evaluate_objective(needs_grad=needs_grad, iteration=int(it))
            coeffs_current = objective["coeffs_current"]
            pred_fg = objective["pred_fg"]
            est_eor = objective["est_eor"]
            loss = objective["loss"]
            optimization_loss = objective["optimization_loss"]
            data_loss = objective["data_loss"]
            data_residual_over_error = objective["data_residual_over_error"]
            residual_cheb_loss = objective["residual_cheb_loss"]
            residual_spatial_lowpass_loss = objective["residual_spatial_lowpass_loss"]
            residual_rms_loss = objective["residual_rms_loss"]
            residual_power_loss = objective["residual_power_loss"]
            residual_power_isotropy_loss = objective["residual_power_isotropy_loss"]
            residual_freq_corr_raw_loss = objective["residual_freq_corr_raw_loss"]
            residual_freq_corr_loss = objective["residual_freq_corr_loss"]
            fg_l2_loss = objective["fg_l2_loss"]
            fg_pos_loss = objective["fg_pos_loss"]
            tv_loss = objective["tv_loss"]
            coeff_prior_loss = objective["coeff_prior_loss"]
            coeff_prior_rms_by_degree = objective["coeff_prior_rms_by_degree"]
            coeff_delta_trust_loss = objective["coeff_delta_trust_loss"]
            current_data_window_lower = objective["current_data_window_lower"]
            current_data_window_upper = objective["current_data_window_upper"]
            current_residual_cheb_weight = objective["current_residual_cheb_weight"]
            current_coeff_prior_weight = objective["current_coeff_prior_weight"]
            current_coeff_prior_hinge_sigma = objective["current_coeff_prior_hinge_sigma"]
            if str(args.save_selection_mode) in {
                "train_loss",
                "last",
                "data_prior_boundary",
                "data_window_min_prior",
                "data_window_max_data",
                "data_window_max_data_skip0",
                "data_window_latest",
                "data_window_earliest",
                "data_window_earliest_skip0",
            }:
                if str(args.save_selection_mode) == "data_prior_boundary":
                    selection_prior_rms = coeff_prior_rms_by_degree[int(args.selection_prior_degree)].to(
                        device=device,
                        dtype=torch_dtype,
                    )
                    selection_metric_tensor = data_loss + float(args.selection_prior_weight) * torch.abs(
                        selection_prior_rms - float(args.selection_prior_target_rms)
                    )
                    current_selection_metric = float(selection_metric_tensor.detach().cpu())
                elif str(args.save_selection_mode) == "data_window_min_prior":
                    selection_prior_rms = coeff_prior_rms_by_degree[int(args.selection_prior_degree)].to(
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_lower = torch.as_tensor(
                        float(args.selection_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_upper = torch.as_tensor(
                        float(args.selection_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_violation = (
                        torch.relu(window_lower - data_residual_over_error)
                        + torch.relu(data_residual_over_error - window_upper)
                    )
                    selection_metric_tensor = selection_prior_rms + float(
                        args.selection_data_window_penalty
                    ) * window_violation
                    current_selection_metric = float(selection_metric_tensor.detach().cpu())
                elif str(args.save_selection_mode) in {"data_window_max_data", "data_window_max_data_skip0"}:
                    window_lower = torch.as_tensor(
                        float(args.selection_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_upper = torch.as_tensor(
                        float(args.selection_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_violation = (
                        torch.relu(window_lower - data_residual_over_error)
                        + torch.relu(data_residual_over_error - window_upper)
                    )
                    selection_metric_tensor = -data_residual_over_error + float(
                        args.selection_data_window_penalty
                    ) * window_violation
                    current_selection_metric = float(selection_metric_tensor.detach().cpu())
                elif str(args.save_selection_mode) == "data_window_latest":
                    window_lower = torch.as_tensor(
                        float(args.selection_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_upper = torch.as_tensor(
                        float(args.selection_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_violation = (
                        torch.relu(window_lower - data_residual_over_error)
                        + torch.relu(data_residual_over_error - window_upper)
                    )
                    selection_metric_tensor = -torch.as_tensor(
                        float(it),
                        device=device,
                        dtype=torch_dtype,
                    ) + float(args.selection_data_window_penalty) * window_violation
                    current_selection_metric = float(selection_metric_tensor.detach().cpu())
                elif str(args.save_selection_mode) in {"data_window_earliest", "data_window_earliest_skip0"}:
                    window_lower = torch.as_tensor(
                        float(args.selection_data_window_lower),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_upper = torch.as_tensor(
                        float(args.selection_data_window_upper),
                        device=device,
                        dtype=torch_dtype,
                    )
                    window_violation = (
                        torch.relu(window_lower - data_residual_over_error)
                        + torch.relu(data_residual_over_error - window_upper)
                    )
                    outside_window = (window_violation > 0).to(dtype=torch_dtype)
                    selection_metric_tensor = (
                        float(it)
                        + outside_window * 1.0e9
                        + float(args.selection_data_window_penalty) * window_violation
                    )
                    current_selection_metric = float(selection_metric_tensor.detach().cpu())
                else:
                    current_selection_metric = float(loss.detach().cpu())
                skip_initial = str(args.save_selection_mode) in {
                    "data_window_max_data_skip0",
                    "data_window_earliest_skip0",
                } and int(it) == 0
                should_select = (
                    not skip_initial
                    and (str(args.save_selection_mode) == "last" or current_selection_metric < selected_metric)
                )
                if should_select:
                    selected_metric = current_selection_metric
                    selected_iter = int(it)
                    selected_state = coeffs_current.detach().cpu().clone()
            if it < int(args.num_iters):
                old_loss_float = float(loss.detach().cpu())
                if use_lbfgs:
                    def lbfgs_closure(opt: torch.optim.Optimizer = opt) -> torch.Tensor:
                        opt.zero_grad(set_to_none=True)
                        closure_objective = evaluate_objective(needs_grad=True, iteration=int(it))
                        closure_loss = closure_objective["optimization_loss"]
                        if not torch.isfinite(closure_loss):
                            raise FloatingPointError(
                                f"Non-finite LBFGS closure loss at iter {it}: "
                                f"{float(closure_loss.detach().cpu())}"
                            )
                        closure_loss.backward()
                        return closure_loss

                    opt.step(lbfgs_closure)
                    project_optimizer_params()
                    last_total_grad_norm = _grad_norm(optimizer_params)
                    last_step_control = {
                        "accepted": True,
                        "source": "lbfgs",
                        "alpha": float("nan"),
                        "attempts": int(args.lbfgs_max_iter),
                        "old_loss": float(old_loss_float),
                        "candidate_loss": float("nan"),
                    }
                elif str(args.step_control_mode) == "train_loss_backtracking":
                    old_param_values = _clone_optimizer_params(optimizer_params)
                    old_optimizer_state = _clone_optimizer_state(opt)
                else:
                    old_param_values = []
                    old_optimizer_state = {}
                if not use_lbfgs:
                    optimization_loss.backward()
                    if float(args.grad_clip_norm) > 0.0:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(optimizer_params, float(args.grad_clip_norm))
                        last_total_grad_norm = float(grad_norm_tensor.detach().cpu())
                    else:
                        last_total_grad_norm = _grad_norm(optimizer_params)
                    if str(args.step_control_mode) == "train_loss_backtracking":
                        grad_param_values = [
                            (
                                param.grad.detach().clone()
                                if param.grad is not None
                                else torch.zeros_like(param.detach())
                            )
                            for param in optimizer_params
                        ]
                    else:
                        grad_param_values = []
                    opt.step()
                    project_optimizer_params()
                    if str(args.step_control_mode) == "train_loss_backtracking":
                        proposed_param_values = _clone_optimizer_params(optimizer_params)
                        accept_limit = (
                            old_loss_float * (1.0 + float(args.step_control_rel_tol))
                            + float(args.step_control_abs_tol)
                        )
                        accepted = False
                        accepted_source = "rejected"
                        accepted_alpha = 0.0
                        accepted_loss = float("nan")
                        attempts = 0
                        for attempt in range(int(args.step_control_max_backtracks) + 1):
                            alpha = float(args.step_control_shrink) ** int(attempt)
                            _blend_optimizer_params(optimizer_params, old_param_values, proposed_param_values, alpha)
                            project_optimizer_params()
                            with torch.no_grad():
                                candidate_loss = float(
                                    evaluate_objective(needs_grad=False, iteration=int(it))["loss"].detach().cpu()
                                )
                            attempts = int(attempt) + 1
                            if math.isfinite(candidate_loss) and candidate_loss <= accept_limit:
                                accepted = True
                                accepted_source = "optimizer"
                                accepted_alpha = alpha
                                accepted_loss = candidate_loss
                                break
                        if not accepted and bool(args.step_control_fallback_gradient):
                            _restore_optimizer_params(optimizer_params, old_param_values)
                            opt.load_state_dict(old_optimizer_state)
                            grad_eps = float(args.step_control_gradient_eps)
                            for attempt in range(int(args.step_control_max_backtracks) + 1):
                                alpha = float(args.step_control_shrink) ** int(attempt)
                                with torch.no_grad():
                                    for param, start, grad in zip(optimizer_params, old_param_values, grad_param_values):
                                        grad_rms = torch.sqrt(torch.mean(grad.detach().double() * grad.detach().double()))
                                        grad_rms = grad_rms.to(device=param.device, dtype=param.dtype)
                                        denom = torch.clamp(
                                            grad_rms,
                                            min=torch.as_tensor(grad_eps, device=param.device, dtype=param.dtype),
                                        )
                                        param.copy_(start - float(args.lr) * alpha * grad.to(dtype=param.dtype) / denom)
                                project_optimizer_params()
                                with torch.no_grad():
                                    candidate_loss = float(
                                        evaluate_objective(needs_grad=False, iteration=int(it))["loss"].detach().cpu()
                                    )
                                attempts += 1
                                if math.isfinite(candidate_loss) and candidate_loss <= accept_limit:
                                    accepted = True
                                    accepted_source = "gradient"
                                    accepted_alpha = alpha
                                    accepted_loss = candidate_loss
                                    opt.state.clear()
                                    break
                        if not accepted:
                            _restore_optimizer_params(optimizer_params, old_param_values)
                            opt.load_state_dict(old_optimizer_state)
                            project_optimizer_params()
                        last_step_control = {
                            "accepted": bool(accepted),
                            "source": str(accepted_source),
                            "alpha": float(accepted_alpha),
                            "attempts": int(attempts),
                            "old_loss": float(old_loss_float),
                            "candidate_loss": float(accepted_loss),
                        }

            do_log = it == 0 or it == int(args.num_iters) or (
                int(args.print_every) > 0 and it % int(args.print_every) == 0
            )
            if do_log:
                with torch.no_grad():
                    metrics = _cube_metrics(est_eor, dirty_eor, dirty_total, pred_fg)
                    coeff_delta = coeffs_current - base_coeffs
                    delta_rms_by_degree = torch.sqrt(torch.mean(coeff_delta.double() ** 2, dim=(1, 2)))
                    base_rms = torch.sqrt(torch.mean(base_coeffs.double() ** 2, dim=(1, 2)))
                    param_grad = params.grad.detach() if params.grad is not None else torch.zeros_like(params)
                    grad_rms_by_degree = torch.sqrt(torch.mean(param_grad.double() ** 2, dim=(1, 2)))
                    template_gain = materialize_template_gain().detach().reshape(n_degree)
                    if template_gain_param is not None:
                        if str(args.template_gain_mode) == "global":
                            template_gain_param_values = template_gain_param.detach().reshape(1)
                            template_gain_grad_values = (
                                template_gain_param.grad.detach().reshape(1)
                                if template_gain_param.grad is not None
                                else torch.zeros((1,), device=device, dtype=torch_dtype)
                            )
                        else:
                            template_gain_param_values = template_gain_param.detach().reshape(n_degree)
                            template_gain_grad_values = (
                                template_gain_param.grad.detach().reshape(n_degree)
                                if template_gain_param.grad is not None
                                else torch.zeros((n_degree,), device=device, dtype=torch_dtype)
                            )
                    else:
                        template_gain_param_values = torch.zeros((0,), device=device, dtype=torch_dtype)
                        template_gain_grad_values = torch.zeros((0,), device=device, dtype=torch_dtype)
                    template_basis_gain = materialize_template_basis_gain().detach().reshape(n_template_basis, n_degree)
                    if template_basis_param is not None:
                        template_basis_param_values = template_basis_param.detach().reshape(-1)
                        template_basis_grad_values = (
                            template_basis_param.grad.detach().reshape(-1)
                            if template_basis_param.grad is not None
                            else torch.zeros((int(template_basis_param.numel()),), device=device, dtype=torch_dtype)
                        )
                    else:
                        template_basis_param_values = torch.zeros((0,), device=device, dtype=torch_dtype)
                        template_basis_grad_values = torch.zeros((0,), device=device, dtype=torch_dtype)
                    c0_correction_current = materialize_c0_correction()
                    if c0_correction_current is not None:
                        c0_correction_rms_by_degree = torch.sqrt(
                            torch.mean(c0_correction_current.double() ** 2, dim=(1, 2))
                        )
                        c0_base_rms = base_rms[c0_correction_degrees_tensor].double()
                        c0_correction_over_base_by_degree = c0_correction_rms_by_degree / torch.clamp(
                            c0_base_rms,
                            min=torch.as_tensor(1e-300, device=device, dtype=torch.float64),
                        )
                    else:
                        c0_correction_rms_by_degree = torch.zeros((0,), device=device, dtype=torch.float64)
                        c0_correction_over_base_by_degree = torch.zeros((0,), device=device, dtype=torch.float64)
                    row = {
                        "iter": int(it),
                        "run_label": str(run_label),
                        "loss": float(loss.detach().cpu()),
                        "optimization_loss": float(optimization_loss.detach().cpu()),
                        "lr_current": float(current_lr),
                        "data_loss": float(data_loss.detach().cpu()),
                        "data_error_rms": float(data_error_rms.detach().cpu()),
                        "data_residual_over_data_error_rms": float(data_residual_over_error.detach().cpu()),
                        "data_window_lower_current": float(current_data_window_lower.detach().cpu()),
                        "data_window_upper_current": float(current_data_window_upper.detach().cpu()),
                        "residual_cheb_loss": float(residual_cheb_loss.detach().cpu()),
                        "residual_cheb_weight_current": float(current_residual_cheb_weight.detach().cpu()),
                        "residual_cheb_norm_rms": float(residual_cheb_norm_rms.detach().cpu()),
                        "residual_spatial_lowpass_loss": float(
                            residual_spatial_lowpass_loss.detach().cpu()
                        ),
                        "residual_spatial_lowpass_norm_rms": float(
                            residual_spatial_lowpass_norm_rms.detach().cpu()
                        ),
                        "residual_rms_loss": float(residual_rms_loss.detach().cpu()),
                        "residual_power_loss": float(residual_power_loss.detach().cpu()),
                        "residual_power_isotropy_loss": float(
                            residual_power_isotropy_loss.detach().cpu()
                        ),
                        "residual_freq_corr_raw_loss": float(residual_freq_corr_raw_loss.detach().cpu()),
                        "residual_freq_corr_loss": float(residual_freq_corr_loss.detach().cpu()),
                        "fg_l2_loss": float(fg_l2_loss.detach().cpu()),
                        "fg_pos_loss": float(fg_pos_loss.detach().cpu()),
                        "tv_loss": float(tv_loss.detach().cpu()),
                        "coeff_prior_loss": float(coeff_prior_loss.detach().cpu()),
                        "coeff_prior_weight_current": float(current_coeff_prior_weight.detach().cpu()),
                        "coeff_prior_hinge_sigma_current": float(
                            current_coeff_prior_hinge_sigma.detach().cpu()
                        ),
                        "coeff_prior_rms_by_degree": [
                            float(v) for v in coeff_prior_rms_by_degree.detach().cpu().numpy()
                        ],
                        "coeff_delta_trust_loss": float(coeff_delta_trust_loss.detach().cpu()),
                        "coeff_delta_rms_by_degree": [float(v) for v in delta_rms_by_degree.detach().cpu().numpy()],
                        "coeff_delta_over_base_rms_by_degree": [
                            float(v)
                            for v in (
                                delta_rms_by_degree
                                / torch.clamp(base_rms, min=torch.as_tensor(1e-300, device=device, dtype=torch.float64))
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        ],
                        "param_grad_rms_by_degree": [float(v) for v in grad_rms_by_degree.detach().cpu().numpy()],
                        "total_grad_norm": float(last_total_grad_norm),
                        "step_control_mode": str(args.step_control_mode),
                        "step_control_last_accepted": last_step_control["accepted"],
                        "step_control_last_source": str(last_step_control["source"]),
                        "step_control_last_alpha": float(last_step_control["alpha"]),
                        "step_control_last_attempts": int(last_step_control["attempts"]),
                        "step_control_last_old_loss": float(last_step_control["old_loss"]),
                        "step_control_last_candidate_loss": float(last_step_control["candidate_loss"]),
                        "template_gain_by_degree": [float(v) for v in template_gain.detach().cpu().numpy()],
                        "template_gain_param_values": [
                            float(v) for v in template_gain_param_values.detach().cpu().numpy()
                        ],
                        "template_gain_param_grad_values": [
                            float(v) for v in template_gain_grad_values.detach().cpu().numpy()
                        ],
                        "template_basis_gain_by_basis_degree": [
                            [float(v) for v in row_vals]
                            for row_vals in template_basis_gain.detach().cpu().numpy().tolist()
                        ],
                        "template_basis_param_values": [
                            float(v) for v in template_basis_param_values.detach().cpu().numpy()
                        ],
                        "template_basis_param_grad_values": [
                            float(v) for v in template_basis_grad_values.detach().cpu().numpy()
                        ],
                        "c0_correction_rms_k": float(
                            (
                                torch.sqrt(torch.mean(c0_correction_current.double() ** 2))
                                if c0_correction_current is not None
                                else torch.zeros((), device=device, dtype=torch.float64)
                            )
                            .detach()
                            .cpu()
                        ),
                        "c0_correction_degrees": [int(v) for v in c0_correction_degrees],
                        "c0_correction_stage_index": int(current_c0_stage_index),
                        "c0_correction_stage_label": str(current_c0_stage_label),
                        "c0_correction_stage_active_degrees": [
                            int(v) for v in current_c0_stage_degrees
                        ],
                        "c0_correction_stage_active_blocks": [
                            int(v) for v in current_c0_stage_blocks
                        ],
                        "c0_correction_stage_active_param_count": int(c0_stage_active_param_count),
                        "c0_correction_rms_k_by_degree": [
                            float(v) for v in c0_correction_rms_by_degree.detach().cpu().numpy()
                        ],
                        "c0_correction_over_base_rms": float(
                            (
                                torch.sqrt(torch.mean(c0_correction_current.double() ** 2))
                                / torch.clamp(base_rms[0], min=torch.as_tensor(1e-300, device=device, dtype=torch.float64))
                                if c0_correction_current is not None
                                else torch.zeros((), device=device, dtype=torch.float64)
                            )
                            .detach()
                            .cpu()
                        ),
                        "c0_correction_over_base_rms_by_degree": [
                            float(v) for v in c0_correction_over_base_by_degree.detach().cpu().numpy()
                        ],
                        "c0_correction_grad_rms": float(
                            (
                                torch.sqrt(torch.mean(c0_correction_param.grad.detach().double() ** 2))
                                if c0_correction_param is not None and c0_correction_param.grad is not None
                                else torch.zeros((), device=device, dtype=torch.float64)
                            )
                            .detach()
                            .cpu()
                        ),
                        "global_residual_over_dirty_eor_rms": metrics["global_residual_over_dirty_eor_rms"],
                        "global_corr": metrics["global_corr"],
                        "time_utc": _now(),
                    }
                    history.append(row)
                    print(json.dumps({"event": "optim_progress", **row}, sort_keys=True), flush=True)
                    current_metric = float(metrics["global_residual_over_dirty_eor_rms"])
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_iter = int(it)
                        best_state = coeffs_current.detach().cpu().clone()
            if not torch.isfinite(loss) or not torch.isfinite(optimization_loss):
                raise FloatingPointError(
                    f"Non-finite loss at iter {it}: "
                    f"loss={float(loss.detach().cpu())}, "
                    f"optimization_loss={float(optimization_loss.detach().cpu())}"
                )

        if str(args.save_selection_mode) == "truth_metric" and best_state is not None:
            coeffs_eval = best_state.to(device=device, dtype=torch_dtype)
            selected_iter_out = int(best_iter)
            selected_metric_out = float(best_metric)
        elif selected_state is not None:
            coeffs_eval = selected_state.to(device=device, dtype=torch_dtype)
            selected_iter_out = int(selected_iter)
            selected_metric_out = float(selected_metric)
        else:
            coeffs_eval = materialize_coeffs().detach()
            selected_iter_out = int(args.num_iters)
            selected_metric_out = float("nan")
        with torch.no_grad():
            fg_k_best = torch.einsum("fd,dxy->fxy", cheb, coeffs_eval)
            fg_flux_best = fg_k_best * k_to_jy[:, None, None]
            pred_fg_best = forward_op(fg_flux_best)
            est_eor_best = dirty_total - pred_fg_best
            final_metrics = _cube_metrics(est_eor_best, dirty_eor, dirty_total, pred_fg_best)

        coeffs_np = coeffs_eval.detach().cpu().numpy().astype(np.float32)
        pred_fg_np = pred_fg_best.detach().cpu().numpy().astype(np.float32)
        est_eor_np = est_eor_best.detach().cpu().numpy().astype(np.float32)
        err_np = (est_eor_best - dirty_eor).detach().cpu().numpy().astype(np.float32)

        coeff_path = run_out_dir / "cheb_coeffs_best_k.fits"
        pred_fg_path = run_out_dir / "pred_fg_dirty_best.fits"
        est_eor_path = run_out_dir / "eor_est_dirty_best.fits"
        err_path = run_out_dir / "eor_error_dirty_best.fits"
        _write_cube(coeff_path, coeffs_np, None)
        if args.save_products:
            _write_cube(pred_fg_path, pred_fg_np, first_template)
            _write_cube(est_eor_path, est_eor_np, first_template)
            _write_cube(err_path, err_np, first_template)

        manifest = {
            "created_at": _now(),
            "method": "actual_optimizer_loop_cached_pca_proxy_cheb",
            "important_limitations": [
                "Optimizer input is dirty total only; dirty FG/EoR component files are summed only because a combined observation product is not present.",
                "Dirty EoR truth is used only for post-fit recovery metrics.",
                "Pure PCA interpolation is used; train_hybrid exact-response reads are intentionally excluded from the optimizer loop.",
                "Residual Cheb projection penalty is an EoR-preservation heuristic, not a formal identifiability guarantee.",
                "residual_power_target_source=dirty_eor is an oracle statistical-prior diagnostic, not a deployable setting.",
            ],
            "settings": {
                "freqs_mhz": [float(v) for v in freqs],
                "dense_grid_csv_pattern": str(args.dense_grid_csv_pattern),
                "train_grid_csv_pattern": str(args.train_grid_csv_pattern),
                "train_response_pattern": str(args.train_response_pattern),
                "truth_fg_pattern": str(args.truth_fg_pattern),
                "truth_eor_pattern": str(args.truth_eor_pattern),
                "image_size": int(args.image_size),
                "response_crop_size": int(args.response_crop_size),
                "eval_crop_size": int(args.eval_crop_size),
                "tile_size": int(args.tile_size),
                "train_halo_px": int(args.train_halo_px),
                "model_margin": int(args.model_margin),
                "pca_rank": int(args.pca_rank),
                "rbf_scale_px": float(args.rbf_scale_px),
                "cheb_degree": int(args.cheb_degree),
                "pixel_arcsec": float(args.pixel_arcsec),
                "num_iters": int(args.num_iters),
                "lr": float(args.lr),
                "lr_schedule": _scalar_schedule_to_dicts(lr_schedule),
                "optimizer_name": str(args.optimizer_name),
                "adam_beta1": float(args.adam_beta1),
                "adam_beta2": float(args.adam_beta2),
                "adam_eps": float(args.adam_eps),
                "weight_decay": float(args.weight_decay),
                "momentum": float(args.momentum),
                "nesterov": bool(args.nesterov),
                "lbfgs_max_iter": int(args.lbfgs_max_iter),
                "lbfgs_history_size": int(args.lbfgs_history_size),
                "lbfgs_line_search": str(args.lbfgs_line_search),
                "lbfgs_tolerance_grad": float(args.lbfgs_tolerance_grad),
                "lbfgs_tolerance_change": float(args.lbfgs_tolerance_change),
                "grad_clip_norm": float(args.grad_clip_norm),
                "step_control_mode": str(args.step_control_mode),
                "step_control_max_backtracks": int(args.step_control_max_backtracks),
                "step_control_shrink": float(args.step_control_shrink),
                "step_control_rel_tol": float(args.step_control_rel_tol),
                "step_control_abs_tol": float(args.step_control_abs_tol),
                "step_control_fallback_gradient": bool(args.step_control_fallback_gradient),
                "step_control_gradient_eps": float(args.step_control_gradient_eps),
                "device": str(args.device),
                "dtype": str(args.dtype),
                "keep_kernel_fft_on_device": bool(args.keep_kernel_fft_on_device),
                "checkpoint_tiles": bool(args.checkpoint_tiles),
                "data_weight": float(args.data_weight),
                "optimization_loss_scale": float(args.optimization_loss_scale),
                "data_loss_mode": str(args.data_loss_mode),
                "data_window_lower": float(args.data_window_lower),
                "data_window_upper": float(args.data_window_upper),
                "data_window_lower_schedule": _scalar_schedule_to_dicts(data_window_lower_schedule),
                "data_window_upper_schedule": _scalar_schedule_to_dicts(data_window_upper_schedule),
                "data_error_rms": float(args.data_error_rms),
                "data_error_mode": str(data_error_mode),
                "residual_cheb_weight": float(args.residual_cheb_weight),
                "residual_cheb_weight_schedule": _scalar_schedule_to_dicts(
                    residual_cheb_weight_schedule
                ),
                "residual_cheb_norm_mode": str(args.residual_cheb_norm_mode),
                "residual_cheb_norm_rms": float(residual_cheb_norm_rms.detach().cpu()),
                "residual_spatial_lowpass_weight": float(args.residual_spatial_lowpass_weight),
                "residual_spatial_lowpass_dct_size": int(args.residual_spatial_lowpass_dct_size),
                "residual_spatial_lowpass_norm_mode": str(args.residual_spatial_lowpass_norm_mode),
                "residual_spatial_lowpass_norm_rms": float(
                    residual_spatial_lowpass_norm_rms.detach().cpu()
                ),
                "residual_spatial_lowpass_skip_dc": bool(args.residual_spatial_lowpass_skip_dc),
                "residual_rms_target": float(args.residual_rms_target),
                "residual_rms_weight": float(args.residual_rms_weight),
                "residual_power_weight": float(args.residual_power_weight),
                "residual_power_target_source": str(args.residual_power_target_source),
                "residual_power_target_npz": (
                    str(args.residual_power_target_npz)
                    if args.residual_power_target_npz is not None
                    else None
                ),
                "residual_power_loss_mode": str(args.residual_power_loss_mode),
                "residual_power_num_bins": int(args.residual_power_num_bins),
                "residual_power_kmin": float(args.residual_power_kmin),
                "residual_power_kmax": float(args.residual_power_kmax),
                "residual_power_bin_edges": [float(v) for v in residual_power_bin_edges.tolist()],
                "residual_power_log_eps": float(args.residual_power_log_eps),
                "residual_power_log_tolerance": float(args.residual_power_log_tolerance),
                "residual_power_operator_white_samples": int(
                    args.residual_power_operator_white_samples
                ),
                "residual_power_operator_white_seed": int(args.residual_power_operator_white_seed),
                "residual_power_isotropy_weight": float(args.residual_power_isotropy_weight),
                "residual_freq_corr_weight": float(args.residual_freq_corr_weight),
                "residual_freq_corr_target_source": str(args.residual_freq_corr_target_source),
                "residual_freq_corr_target_npz": (
                    str(args.residual_freq_corr_target_npz)
                    if args.residual_freq_corr_target_npz is not None
                    else None
                ),
                "residual_freq_corr_loss_mode": str(args.residual_freq_corr_loss_mode),
                "residual_freq_corr_hinge_target": float(args.residual_freq_corr_hinge_target),
                "residual_freq_corr_eps": float(args.residual_freq_corr_eps),
                "residual_freq_corr_operator_white_samples": int(
                    args.residual_freq_corr_operator_white_samples
                ),
                "residual_freq_corr_operator_white_seed": int(
                    args.residual_freq_corr_operator_white_seed
                ),
                "fg_l2_weight": float(args.fg_l2_weight),
                "fg_positivity_weight": float(args.fg_positivity_weight),
                "coeff_tv_weight": float(args.coeff_tv_weight),
                "prior_mode": str(prior_mode),
                "prior_cheb_coeffs": str(args.prior_cheb_coeffs) if args.prior_cheb_coeffs is not None else None,
                "coeff_prior_weight": float(args.coeff_prior_weight),
                "coeff_prior_weight_schedule": _scalar_schedule_to_dicts(coeff_prior_weight_schedule),
                "coeff_prior_loss_mode": str(args.coeff_prior_loss_mode),
                "coeff_prior_hinge_sigma": float(args.coeff_prior_hinge_sigma),
                "coeff_prior_hinge_sigma_schedule": _scalar_schedule_to_dicts(
                    coeff_prior_hinge_sigma_schedule
                ),
                "coeff_prior_scales_k": [float(v) for v in prior_scales_np.tolist()],
                "coeff_prior_relative_scales": [float(v) for v in prior_rel_scales],
                "coeff_delta_trust_weight": float(args.coeff_delta_trust_weight),
                "coeff_delta_trust_scales_k": [float(v) for v in trust_scales_np.tolist()],
                "coeff_delta_trust_relative_scales": [float(v) for v in trust_rel_scales],
                "coeff_scale_k": float(args.coeff_scale_k),
                "param_mode": str(args.param_mode),
                "freeze_full_pixel_degrees": bool(freeze_full_pixel_degrees),
                "optimize_degrees": [int(v) for v in optimize_degrees],
                "degree_param_scales_k": [float(v) for v in degree_param_scales_np.tolist()],
                "degree_param_relative_scales": [float(v) for v in degree_rel_scales],
                "degree_param_max_abs": [float(v) for v in degree_param_max_abs_np.tolist()],
                "template_gain_mode": str(args.template_gain_mode),
                "template_gain_init": [float(v) for v in template_gain_init_np.tolist()],
                "template_gain_param_scale": float(args.template_gain_param_scale),
                "template_gain_param_max_abs": float(args.template_gain_param_max_abs),
                "template_basis_cheb_coeffs_list": [str(v) for v in template_basis_paths],
                "template_basis_gain_mode": str(args.template_basis_gain_mode),
                "template_basis_gain_init": [
                    [float(v) for v in row_vals]
                    for row_vals in template_basis_gain_init_np.tolist()
                ],
                "template_basis_param_scale": float(args.template_basis_param_scale),
                "template_basis_param_max_abs": float(args.template_basis_param_max_abs),
                "allow_suspicious_template_basis": bool(args.allow_suspicious_template_basis),
                "template_basis_rms_k_by_basis_degree": [
                    [float(v) for v in row_vals]
                    for row_vals in template_basis_rms_np.tolist()
                ],
                "c0_correction_mode": str(args.c0_correction_mode),
                "c0_correction_degrees": [int(v) for v in c0_correction_degrees],
                "c0_correction_blocks": int(args.c0_correction_blocks),
                "c0_correction_block_list": [int(v) for v in c0_correction_block_list],
                "c0_correction_dct_size": int(args.c0_correction_dct_size),
                "c0_correction_scale_k": float(c0_correction_scale),
                "c0_correction_scales_k": [float(v) for v in c0_correction_scales_np.tolist()],
                "c0_correction_relative_scale": float(args.c0_correction_relative_scale),
                "c0_correction_param_max_abs": float(args.c0_correction_param_max_abs),
                "c0_correction_stage_spec": str(args.c0_correction_stage_spec),
                "c0_correction_stage_reset_optimizer": bool(args.c0_correction_stage_reset_optimizer),
                "c0_correction_stage_schedule": [
                    _c0_correction_stage_to_dict(stage) for stage in c0_correction_stages
                ],
                "init_mode": init_mode,
                "init_cheb_coeffs": str(init_path) if init_path is not None else None,
                "run_label": str(run_label),
                "n_tile_caches": int(len(caches)),
                "tile_cache_dir": str(args.tile_cache_dir) if args.tile_cache_dir is not None else None,
                "tile_cache_refresh": bool(args.tile_cache_refresh),
                "tile_cache_meta_path_rewrite": [
                    [str(old), str(new)] for old, new in tile_cache_meta_path_rewrites
                ],
                "save_selection_mode": str(args.save_selection_mode),
                "selection_prior_degree": int(args.selection_prior_degree),
                "selection_prior_target_rms": float(args.selection_prior_target_rms),
                "selection_prior_weight": float(args.selection_prior_weight),
                "selection_data_window_lower": float(args.selection_data_window_lower),
                "selection_data_window_upper": float(args.selection_data_window_upper),
                "selection_data_window_penalty": float(args.selection_data_window_penalty),
            },
            "best_iter": int(best_iter),
            "best_metric": float(best_metric),
            "best_metric_name": "truth_global_residual_over_dirty_eor_rms",
            "selected_iter": int(selected_iter_out),
            "selected_metric": float(selected_metric_out),
            "selected_metric_name": (
                "truth_global_residual_over_dirty_eor_rms"
                if str(args.save_selection_mode) == "truth_metric"
                else str(args.save_selection_mode)
            ),
            "final_metrics": final_metrics,
            "history": history,
            "products": {
                "cheb_coeffs_best_k": str(coeff_path),
                "pred_fg_dirty_best": str(pred_fg_path) if args.save_products else None,
                "eor_est_dirty_best": str(est_eor_path) if args.save_products else None,
                "eor_error_dirty_best": str(err_path) if args.save_products else None,
            },
        }
        manifest_path = run_out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "fit_run_done",
                    "run_label": str(run_label),
                    "manifest": str(manifest_path),
                    "metrics": final_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        summary = {
            "run_label": str(run_label),
            "out_dir": str(run_out_dir),
            "manifest": str(manifest_path),
            "init_cheb_coeffs": str(init_path) if init_path is not None else None,
            "best_iter": int(best_iter),
            "best_metric": float(best_metric),
            "selected_iter": int(selected_iter_out),
            "selected_metric": float(selected_metric_out),
            "save_selection_mode": str(args.save_selection_mode),
            "final_metrics": final_metrics,
        }
        del params, opt, coeffs_eval, pred_fg_best, est_eor_best, fg_k_best, fg_flux_best
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return summary

    if args.init_cheb_coeffs_list:
        raw_inits = _parse_list(args.init_cheb_coeffs_list)
        if not raw_inits:
            raise ValueError("--init-cheb-coeffs-list was set but parsed to an empty list")
        init_paths: List[Path | None] = [
            None if item.upper() in {"NONE", "ZERO", "NULL"} else Path(item) for item in raw_inits
        ]
        labels = _parse_list(args.run_labels)
        if labels and len(labels) != len(init_paths):
            raise ValueError(f"--run-labels has {len(labels)} entries, expected {len(init_paths)}")
        if not labels:
            labels = [_init_label(path, i) for i, path in enumerate(init_paths)]
        use_subdirs = True
    else:
        init_paths = [args.init_cheb_coeffs]
        labels = _parse_list(args.run_labels)
        if labels and len(labels) != 1:
            raise ValueError("--run-labels without --init-cheb-coeffs-list must contain exactly one label")
        if not labels:
            labels = [_init_label(args.init_cheb_coeffs, 0)]
        use_subdirs = False

    run_summaries: List[Dict[str, Any]] = []
    for i, (init_path, label) in enumerate(zip(init_paths, labels)):
        safe = _safe_label(label)
        run_out_dir = args.out_dir / safe if use_subdirs else args.out_dir
        run_summaries.append(run_fit_once(run_out_dir, init_path, safe))

    if use_subdirs:
        aggregate = {
            "created_at": _now(),
            "method": "multi_init_shared_cached_pca_proxy_cheb",
            "out_dir": str(args.out_dir),
            "n_runs": int(len(run_summaries)),
            "n_tile_caches": int(len(caches)),
            "tile_cache_dir": str(args.tile_cache_dir) if args.tile_cache_dir is not None else None,
            "runs": run_summaries,
        }
        manifest_path = args.out_dir / "multi_run_manifest.json"
        manifest_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"event": "multi_run_done", "manifest": str(manifest_path), "n_runs": int(len(run_summaries))},
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
