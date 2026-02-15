#!/usr/bin/env python3
"""
Command-line interface for foreground/EoR separation.

This module wires together argument parsing, JSON config loading, and the
core optimization routines defined in separation_optim.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

from separation_optim import (
    OptimizationConfig,
    _optimize_from_fits,
    load_config_file,
    run_synthetic_demo,
    write_true_signal_correlation_report,
)


def parse_cli_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Optimize foreground/EoR separation with configurable priors."
    )
    parser.add_argument("--config", type=str, help="Path to JSON config file with defaults.")
    parser.add_argument("--input-cube", type=str, help="Input FITS cube containing observed data.")
    parser.add_argument(
        "--mask-cube",
        type=str,
        help="Optional FITS mask (2D or 3D) applied to all inputs before cut_xy and loss evaluation.",
    )
    parser.add_argument(
        "--psf-cube",
        type=str,
        help=(
            "Optional 3D FITS cube containing a PSF used as a forward operator in the data term. "
            "When set, the data term compares PSF(fg+eor) to the observation instead of fg+eor directly."
        ),
    )
    parser.add_argument(
        "--psf-scale",
        type=float,
        help="Optional scalar multiplier applied after the PSF operator (default 1.0).",
    )
    parser.add_argument("--fg-output", type=str, help="Output FITS path for foreground estimate.")
    parser.add_argument("--eor-output", type=str, help="Output FITS path for EoR estimate.")
    parser.add_argument("--num-iters", type=int, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, help="Learning rate for Adam.")
    parser.add_argument("--alpha", type=float, help="Weight for data fidelity term.")
    parser.add_argument("--beta", type=float, help="Weight for foreground smoothness term.")
    parser.add_argument("--gamma", type=float, help="Weight for EoR regularization term.")
    parser.add_argument("--freq-axis", type=int, help="Frequency axis index in the cube.")
    parser.add_argument("--print-every", type=int, help="Logging frequency.")
    parser.add_argument("--device", type=str, help="Device string, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--dtype", type=str, help="Torch dtype name, e.g., float32.")
    parser.add_argument(
        "--extra-loss-start-iter",
        type=int,
        help=(
            "When extra loss terms are enabled, only base terms are used before this iteration (default 500)."
        ),
    )
    parser.add_argument(
        "--extra-loss-ramp-iters",
        type=int,
        help="If >0, ramp extra loss scale to 1 over this many iterations after start.",
    )
    parser.add_argument(
        "--cut-xy-enabled",
        action="store_true",
        default=None,
        help="Enable spatial XY cropping before optimization (see config for details).",
    )
    parser.add_argument(
        "--cut-xy-unit",
        choices=["frac", "px"],
        type=str,
        help="cut_xy unit: 'frac' or 'px'.",
    )
    parser.add_argument(
        "--cut-xy-center-x",
        type=float,
        help="cut_xy center x (fraction or pixel index depending on unit).",
    )
    parser.add_argument(
        "--cut-xy-center-y",
        type=float,
        help="cut_xy center y (fraction or pixel index depending on unit).",
    )
    parser.add_argument(
        "--cut-xy-size",
        type=float,
        help="cut_xy output square side length (fraction or pixels depending on unit).",
    )
    parser.add_argument("--data-error", type=float, help="Scalar measurement error (default 0.05).")
    parser.add_argument(
        "--eor-mean",
        dest="eor_prior_mean",
        type=float,
        help="Scalar EoR prior mean (default 0.0).",
    )
    parser.add_argument(
        "--eor-sigma",
        dest="eor_prior_sigma",
        type=float,
        help="Scalar EoR prior std (default 0.1).",
    )
    parser.add_argument(
        "--eor-amp-threshold",
        dest="eor_prior_amp_threshold",
        type=float,
        help=(
            "Dead-zone threshold for EoR amplitude prior: "
            "no penalty when |eor-eor_mean| is below this value."
        ),
    )
    parser.add_argument(
        "--eor-amp-prior-mode",
        dest="eor_amp_prior_mode",
        choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"],
        type=str,
        help=(
            "EoR amplitude prior mode. "
            "voxel_deadzone: dead-zone on |eor| per voxel; "
            "slice_rms_hinge: hinge on per-slice std_xy(eor); "
            "hybrid: slice_rms_hinge plus loose voxel outlier guard."
        ),
    )
    parser.add_argument(
        "--eor-hybrid-voxel-factor",
        dest="eor_hybrid_voxel_factor",
        type=float,
        help="Hybrid amp prior: voxel outlier threshold factor (guard_thr = factor * threshold).",
    )
    parser.add_argument(
        "--eor-hybrid-voxel-weight",
        dest="eor_hybrid_voxel_weight",
        type=float,
        help="Hybrid amp prior: relative weight of the voxel outlier guard term.",
    )
    parser.add_argument(
        "--fg-smooth-mean",
        type=float,
        help="Scalar prior mean for FG third differences (default 0.0).",
    )
    parser.add_argument(
        "--fg-smooth-sigma",
        type=float,
        help="Scalar prior std/scale for FG smoothness finite differences (default 0.05).",
    )
    parser.add_argument(
        "--fg-smooth-mode",
        type=str,
        choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"],
        help="Foreground smoothness mode (default diff3_l2).",
    )
    parser.add_argument(
        "--fg-smooth-huber-delta",
        type=float,
        help="Huber delta used when fg_smooth_mode=diff2_huber (default 1.0).",
    )
    parser.add_argument(
        "--fg-reference-cube",
        type=str,
        help="FITS cube used to derive FG smoothness mean/std automatically.",
    )
    parser.add_argument(
        "--use-robust-fg-stats",
        action="store_true",
        default=None,
        help="Use median/MAE (scaled) instead of mean/std when deriving FG smoothness stats.",
    )
    parser.add_argument(
        "--mae-to-sigma-factor",
        type=float,
        help="Scaling factor to convert MAE to sigma when using robust FG stats (default 1.4826).",
    )
    parser.add_argument(
        "--corr-mean",
        dest="corr_prior_mean",
        type=float,
        help="Prior mean for FG/EoR correlation coefficient (default 0.0).",
    )
    parser.add_argument(
        "--corr-sigma",
        dest="corr_prior_sigma",
        type=float,
        help="Prior std for FG/EoR correlation coefficient (default 0.2).",
    )
    parser.add_argument(
        "--corr-abs-threshold",
        dest="corr_prior_abs_threshold",
        type=float,
        help="Dead-zone threshold for |corr-mean| (values below are not penalized; default 0.0).",
    )
    parser.add_argument(
        "--corr-reduce",
        dest="corr_reduce",
        choices=["mean", "topk", "logsumexp"],
        type=str,
        help="Reduction over per-frequency corr penalties (default mean).",
    )
    parser.add_argument(
        "--corr-topk",
        dest="corr_topk",
        type=int,
        help="Top-k used when corr_reduce=topk.",
    )
    parser.add_argument(
        "--corr-lse-alpha",
        dest="corr_lse_alpha",
        type=float,
        help="Temperature used when corr_reduce=logsumexp (log-mean-exp).",
    )
    parser.add_argument(
        "--corr-weight",
        dest="corr_weight",
        type=float,
        help="Weight applied to the correlation prior term (default 1.0).",
    )
    parser.add_argument(
        "--corr-feature",
        dest="corr_feature",
        choices=["raw", "diff1", "diff2"],
        type=str,
        help="Feature used for corr prior: raw, diff1, or diff2 along frequency.",
    )
    parser.add_argument(
        "--corr-spatial-pool",
        dest="corr_spatial_pool",
        type=int,
        help="Spatial avg-pooling factor (>=1) applied before corr statistics.",
    )
    parser.add_argument(
        "--lagcorr-weight",
        dest="lagcorr_weight",
        type=float,
        help="Weight applied to the frequency-lag autocorrelation prior term (lagcorr mode).",
    )
    parser.add_argument(
        "--lagcorr-fg-component-weight",
        dest="lagcorr_fg_component_weight",
        type=float,
        help="Relative weight of the FG lagcorr component (lagcorr mode, default 0.5).",
    )
    parser.add_argument(
        "--lagcorr-eor-component-weight",
        dest="lagcorr_eor_component_weight",
        type=float,
        help="Relative weight of the EoR lagcorr component (lagcorr mode, default 0.5).",
    )
    parser.add_argument(
        "--lagcorr-gap-weight",
        dest="lagcorr_gap_weight",
        type=float,
        help="Weight for FG-vs-EoR lag-gap constraint (requires both lagcorr components).",
    )
    parser.add_argument(
        "--lagcorr-gap-margin",
        dest="lagcorr_gap_margin",
        type=float,
        help="Target margin for FG lagcorr mean minus EoR lagcorr mean (scalar CLI value).",
    )
    parser.add_argument(
        "--lagcorr-gap-sigma",
        dest="lagcorr_gap_sigma",
        type=float,
        help="Scale for lag-gap residual normalization (scalar CLI value).",
    )
    parser.add_argument(
        "--lagcorr-gap-mode",
        dest="lagcorr_gap_mode",
        choices=["hinge", "squared"],
        type=str,
        help="Residual mode for lag-gap constraint (default hinge).",
    )
    parser.add_argument(
        "--lagcorr-feature",
        dest="lagcorr_feature",
        choices=["raw", "diff1"],
        type=str,
        help="Feature transform used by lagcorr: raw cube or first frequency difference.",
    )
    parser.add_argument(
        "--lagcorr-unit",
        dest="lagcorr_unit",
        choices=["mhz", "chan"],
        type=str,
        help="Units used by lagcorr_intervals (lagcorr mode).",
    )
    parser.add_argument(
        "--lagcorr-max-pairs",
        dest="lagcorr_max_pairs",
        type=int,
        help="Maximum number of frequency-slice pairs used per lag (lagcorr mode).",
    )
    parser.add_argument(
        "--lagcorr-spatial-pool",
        dest="lagcorr_spatial_pool",
        type=int,
        help="Spatial avg-pooling factor (>=1) for lagcorr statistics (default 1).",
    )
    parser.add_argument(
        "--lagcorr-eor-mode",
        dest="lagcorr_eor_mode",
        choices=["gaussian", "envelope_v2", "envelope"],
        type=str,
        help="EoR lagcorr prior mode (default gaussian).",
    )
    parser.add_argument(
        "--lagcorr-eor-near-max-lag",
        dest="lagcorr_eor_near_max_lag",
        type=int,
        help="Max lag (in channels) considered 'near' for envelope_v2 priors.",
    )
    parser.add_argument(
        "--lagcorr-eor-mid-max-lag",
        dest="lagcorr_eor_mid_max_lag",
        type=int,
        help="Max lag (in channels) considered 'mid' for rebound suppression (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-eor-far-min-lag",
        dest="lagcorr_eor_far_min_lag",
        type=int,
        help="Min lag (in channels) considered 'far' for tail envelope (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-eor-tail-eps",
        dest="lagcorr_eor_tail_eps",
        type=float,
        help="Allowed |rho(lag)| envelope at far lags before penalty (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-eor-neg-delta",
        dest="lagcorr_eor_neg_delta",
        type=float,
        help="Allowance for negative rho at near lags (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-eor-near-rho-min",
        dest="lagcorr_eor_near_rho_min",
        type=float,
        help="Weak floor on mean near-lag rho (envelope_v2; 0 disables).",
    )
    parser.add_argument(
        "--lagcorr-eor-rebound-eps-act",
        dest="lagcorr_eor_rebound_eps_act",
        type=float,
        help="Activation threshold on |rho| for rebound suppression (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-eor-rebound-delta-up",
        dest="lagcorr_eor_rebound_delta_up",
        type=float,
        help="Allowed increase in |rho| per step before rebound penalty (envelope_v2).",
    )
    parser.add_argument(
        "--lagcorr-pair-sampling",
        dest="lagcorr_pair_sampling",
        choices=["head", "random"],
        type=str,
        help="Pair sampling strategy when lagcorr_max_pairs limits pairs (default head).",
    )
    parser.add_argument(
        "--lagcorr-random-seed",
        dest="lagcorr_random_seed",
        type=int,
        help="Optional random seed used when lagcorr_pair_sampling=random.",
    )
    parser.add_argument(
        "--lagcorr-lag-weights",
        dest="lagcorr_lag_weights",
        type=float,
        help=(
            "Scalar per-lag weight multiplier for lagcorr residual aggregation. "
            "For vector weights, set in config JSON."
        ),
    )
    parser.add_argument(
        "--lagcorr-eor-start-iter",
        dest="lagcorr_eor_start_iter",
        type=int,
        help="Iteration at which EoR lagcorr sub-term starts (defaults to extra_loss_start_iter).",
    )
    parser.add_argument(
        "--lagcorr-eor-ramp-iters",
        dest="lagcorr_eor_ramp_iters",
        type=int,
        help="Ramp iterations for EoR lagcorr sub-term after lagcorr_eor_start_iter.",
    )
    parser.add_argument(
        "--fft-weight",
        dest="fft_weight",
        type=float,
        help="Weight applied to the high-frequency rFFT prior term (default 1.0).",
    )
    parser.add_argument(
        "--eor-mean-weight",
        dest="eor_mean_weight",
        type=float,
        help="Weight applied to the EoR slice-mean prior term (eor_mean extra term).",
    )
    parser.add_argument(
        "--eor-hf-weight",
        dest="eor_hf_weight",
        type=float,
        help="Weight applied to the EoR high-frequency ratio prior term (eor_hf extra term).",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        help=(
            "Legacy selector for loss terms. Accepts 'base' or single/combined terms "
            "such as 'rfft', 'lagcorr', or 'rfft,lagcorr'."
        ),
    )
    parser.add_argument(
        "--extra-loss-terms",
        type=str,
        help=(
            "Comma/plus-separated extra terms to add on top of base, e.g. "
            "'corr', 'rfft+lagcorr', 'corr,rfft,lagcorr'."
        ),
    )
    parser.add_argument(
        "--fft-percent",
        type=float,
        help="Fraction of highest frequency bins penalized in rFFT mode (default 0.7).",
    )
    parser.add_argument(
        "--eor-hf-percent",
        dest="eor_hf_percent",
        type=float,
        help="Fraction of highest rFFT bins used to compute EoR high-frequency energy ratio.",
    )
    parser.add_argument(
        "--eor-hf-r-max",
        dest="eor_hf_r_max",
        type=float,
        help="Allowed maximum high-frequency energy ratio before EoR hf penalty activates.",
    )
    parser.add_argument(
        "--fft-mean",
        dest="fft_prior_mean",
        type=float,
        help="Scalar prior mean for high-frequency energy (default 0.0).",
    )
    parser.add_argument(
        "--fft-sigma",
        dest="fft_prior_sigma",
        type=float,
        help="Scalar prior std for high-frequency energy (default 1.0).",
    )
    parser.add_argument(
        "--fft-use-log-energy",
        dest="fft_use_log_energy",
        action="store_true",
        default=None,
        help="Apply log1p transform to high-frequency energy before rFFT prior matching.",
    )
    parser.add_argument(
        "--fft-z-clip",
        dest="fft_z_clip",
        type=float,
        help="Optional absolute z-score clip for rFFT residuals (robust against outliers).",
    )
    parser.add_argument(
        "--optimizer",
        dest="optimizer_name",
        choices=["adam", "sgd"],
        help="Optimizer choice (default adam).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum for SGD optimizer (default 0.9).",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        type=str,
        help="Learning-rate scheduler (default none).",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        help="Plateau scheduler patience in iterations (default 240).",
    )
    parser.add_argument(
        "--lr-plateau-factor",
        type=float,
        help="Plateau scheduler LR decay factor in (0,1), e.g. 0.5.",
    )
    parser.add_argument(
        "--lr-plateau-min-delta",
        type=float,
        help="Minimum absolute loss improvement to reset plateau patience.",
    )
    parser.add_argument(
        "--lr-plateau-cooldown",
        type=int,
        help="Plateau scheduler cooldown in iterations after each LR drop.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        help="Minimum LR allowed when scheduler is enabled.",
    )
    parser.add_argument(
        "--poly-weight",
        dest="poly_weight",
        type=float,
        help="Weight applied to the polynomial residual prior term (poly_reparam mode).",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        help="Polynomial degree for foreground reparameterization in poly_reparam mode (default 3).",
    )
    parser.add_argument(
        "--poly-sigma",
        type=float,
        help="Std used to scale polynomial prior residuals (default 0.05).",
    )
    parser.add_argument(
        "--poly-basis",
        type=str,
        choices=["power", "chebyshev", "legendre", "dct", "bspline"],
        help="Basis for poly_reparam: power, chebyshev, legendre, dct, bspline (default power).",
    )
    parser.add_argument(
        "--poly-x-mode",
        type=str,
        choices=["lin", "log"],
        help="Polynomial coordinate: lin=freq, log=log(freq) (default lin).",
    )
    parser.add_argument(
        "--poly-model",
        type=str,
        choices=["add", "exp", "exp_mul"],
        help=(
            "Polynomial foreground model in poly_reparam: "
            "add=poly+resid, exp=exp(poly)+resid, exp_mul=exp(poly+resid) (default add)."
        ),
    )
    parser.add_argument(
        "--poly-disable-resid",
        dest="poly_resid_enabled",
        action="store_false",
        default=None,
        help="Disable explicit per-voxel residual when poly_reparam is active (FG becomes pure polynomial).",
    )
    parser.add_argument(
        "--eor-as-residual",
        dest="eor_as_residual",
        action="store_true",
        default=None,
        help=(
            "Do not optimize EoR as a free parameter; instead set EoR = y - FG and backpropagate "
            "EoR priors through FG. This removes FG/EoR degeneracy in near-noiseless injected cubes."
        ),
    )
    parser.add_argument(
        "--freq-start-mhz",
        type=float,
        help="Starting frequency of the cube in MHz (for poly_reparam mode).",
    )
    parser.add_argument(
        "--freq-delta-mhz",
        type=float,
        help="Frequency spacing of the cube in MHz (for poly_reparam mode).",
    )
    parser.add_argument(
        "--true-eor-cube",
        type=str,
        help="FITS cube containing reference EoR for evaluation plots (not used for training).",
    )
    parser.add_argument(
        "--diagnose-input",
        "--report-true-signal-corr",
        dest="diagnose_input",
        action="store_true",
        default=None,
        help=(
            "Diagnose inputs (requires fg_reference_cube + true_eor_cube): compute FG/EoR frequency-lag "
            "autocorrelations plus a loss breakdown, then exit. "
            "('--report-true-signal-corr' is a deprecated alias.)"
        ),
    )
    parser.add_argument(
        "--corr-plot",
        type=str,
        help="File path for saving the per-frequency correlation plot.",
    )
    parser.add_argument(
        "--enable-corr-check",
        action="store_true",
        default=None,
        help=(
            "Enable periodic correlation checks between the recovered EoR and the reference EoR "
            "(requires --true-eor-cube or 'true_eor_cube' in the config)."
        ),
    )
    parser.add_argument(
        "--corr-check-every",
        type=int,
        help="Iteration interval for correlation checks (default 500).",
    )
    parser.add_argument(
        "--init-fg-cube",
        type=str,
        help="FITS cube providing an initial foreground guess.",
    )
    parser.add_argument(
        "--init-eor-cube",
        type=str,
        help="FITS cube providing an initial EoR guess.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["smooth_zero", "smooth_residual", "poly_residual"],
        help="Initialization policy when init cubes are not provided (default smooth_zero).",
    )
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Run the built-in synthetic demo instead of loading a FITS cube.",
    )
    parser.add_argument(
        "--power-config",
        type=str,
        help="Path to JSON config for power spectrum computation.",
    )
    parser.add_argument(
        "--power-output-dir",
        type=str,
        help="Directory to write power spectrum outputs (overrides power config).",
    )
    return parser, parser.parse_args()


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "num_iters",
        "lr",
        "alpha",
        "beta",
        "gamma",
        "fft_weight",
        "eor_mean_weight",
        "eor_hf_weight",
        "loss_mode",
        "extra_loss_terms",
        "freq_axis",
        "print_every",
        "device",
        "dtype",
        "extra_loss_start_iter",
        "extra_loss_ramp_iters",
        "cut_xy_enabled",
        "cut_xy_unit",
        "cut_xy_center_x",
        "cut_xy_center_y",
        "cut_xy_size",
        "data_error",
        "eor_prior_mean",
        "eor_prior_sigma",
        "eor_prior_amp_threshold",
        "eor_amp_prior_mode",
        "eor_hybrid_voxel_factor",
        "eor_hybrid_voxel_weight",
        "fg_smooth_mean",
        "fg_smooth_sigma",
        "fg_smooth_mode",
        "fg_smooth_huber_delta",
        "fg_reference_cube",
        "use_robust_fg_stats",
        "mae_to_sigma_factor",
        "corr_prior_mean",
        "corr_prior_sigma",
        "corr_prior_abs_threshold",
        "corr_reduce",
        "corr_topk",
        "corr_lse_alpha",
        "corr_weight",
        "corr_feature",
        "corr_spatial_pool",
        "lagcorr_weight",
        "lagcorr_fg_component_weight",
        "lagcorr_eor_component_weight",
        "lagcorr_gap_weight",
        "lagcorr_gap_margin",
        "lagcorr_gap_sigma",
        "lagcorr_gap_mode",
        "lagcorr_feature",
        "lagcorr_unit",
        "lagcorr_pair_sampling",
        "lagcorr_random_seed",
        "lagcorr_max_pairs",
        "lagcorr_spatial_pool",
        "lagcorr_eor_mode",
        "lagcorr_eor_near_max_lag",
        "lagcorr_eor_mid_max_lag",
        "lagcorr_eor_far_min_lag",
        "lagcorr_eor_tail_eps",
        "lagcorr_eor_neg_delta",
        "lagcorr_eor_near_rho_min",
        "lagcorr_eor_rebound_eps_act",
        "lagcorr_eor_rebound_delta_up",
        "lagcorr_lag_weights",
        "lagcorr_eor_start_iter",
        "lagcorr_eor_ramp_iters",
        "fft_highfreq_percent",
        "fft_use_log_energy",
        "fft_z_clip",
        "eor_hf_percent",
        "eor_hf_r_max",
        "fft_prior_mean",
        "fft_prior_sigma",
        "poly_weight",
        "poly_degree",
        "poly_sigma",
        "poly_basis",
        "poly_x_mode",
        "poly_model",
        "poly_resid_enabled",
        "eor_as_residual",
        "freq_start_mhz",
        "freq_delta_mhz",
        "power_config",
        "power_output_dir",
        "true_eor_cube",
        "diagnose_input",
        "corr_plot",
        "init_fg_cube",
        "init_eor_cube",
        "init_mode",
        "mask_cube",
        "psf_cube",
        "psf_scale",
        "enable_corr_check",
        "corr_check_every",
        "optimizer_name",
        "momentum",
        "lr_scheduler",
        "lr_plateau_patience",
        "lr_plateau_factor",
        "lr_plateau_min_delta",
        "lr_plateau_cooldown",
        "lr_min",
    ]
    overrides = {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}
    if getattr(args, "fft_percent", None) is not None:
        overrides["fft_highfreq_percent"] = args.fft_percent
    return overrides


def main() -> None:
    parser, args = parse_cli_args()

    config_data: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Config file '{config_path}' not found.")
        config_data = load_config_file(config_path)

    config = OptimizationConfig()

    def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
        transparent_sections = {"optim", "weights", "priors", "evaluation", "init", "power"}
        flattened: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                nested = _flatten_config(value)
                if key in transparent_sections:
                    flattened.update(nested)
                else:
                    for nested_key, nested_value in nested.items():
                        flattened[f"{key}_{nested_key}"] = nested_value
            else:
                flattened[key] = value
        return flattened

    config.update_from_dict(_flatten_config(config_data))
    config.update_from_dict(_collect_cli_overrides(args))

    if args.run_demo:
        run_synthetic_demo(config)
        return

    if config.diagnose_input or config.report_true_signal_corr:
        if not config.fg_reference_cube:
            parser.error(
                "Input diagnostics requested, but foreground reference cube is missing "
                "(set --fg-reference-cube or 'fg_reference_cube' in the config)."
            )
        if not config.true_eor_cube:
            parser.error(
                "Input diagnostics requested, but true EoR cube is missing "
                "(set --true-eor-cube or 'true_eor_cube' in the config)."
            )
        fg_true_path = Path(config.fg_reference_cube)
        if not fg_true_path.exists():
            parser.error(f"Foreground reference cube '{fg_true_path}' not found.")
        eor_true_path = Path(config.true_eor_cube)
        if not eor_true_path.exists():
            parser.error(f"Reference EoR cube '{eor_true_path}' not found.")

        if config.mask_cube:
            mask_path = Path(config.mask_cube)
            if not mask_path.exists():
                parser.error(f"Mask cube '{mask_path}' not found.")

        input_cube = args.input_cube or config_data.get("input_cube")
        fg_output = args.fg_output or config_data.get("fg_output")
        eor_output = args.eor_output or config_data.get("eor_output")

        if eor_output:
            output_dir = Path(eor_output).parent
        elif fg_output:
            output_dir = Path(fg_output).parent
        elif args.config:
            output_dir = Path(args.config).parent
        elif input_cube:
            output_dir = Path(input_cube).parent
        else:
            output_dir = Path(".")

        if args.config:
            prefix = Path(args.config).stem
        elif input_cube:
            prefix = Path(str(input_cube)).stem
        else:
            prefix = "input_diagnostics"

        input_cube_path = Path(str(input_cube)) if input_cube else None
        summary_path, details_path, plot_path = write_true_signal_correlation_report(
            fg_true_path,
            eor_true_path,
            config,
            output_dir=output_dir,
            filename_prefix=prefix,
            input_cube_path=input_cube_path,
        )
        print(f"Saved input diagnostics summary to {summary_path}")
        print(f"Saved input diagnostics details to {details_path}")
        print(f"Saved input diagnostics plot to {plot_path}")
        return

    input_cube = args.input_cube or config_data.get("input_cube")
    if input_cube is None:
        parser.error("Please specify --input-cube or set 'input_cube' in the config file.")
    input_path = Path(input_cube)
    if not input_path.exists():
        parser.error(f"Input cube '{input_path}' not found.")

    if config.mask_cube:
        mask_path = Path(config.mask_cube)
        if not mask_path.exists():
            parser.error(f"Mask cube '{mask_path}' not found.")

    fg_output = args.fg_output or config_data.get("fg_output")
    eor_output = args.eor_output or config_data.get("eor_output")
    if fg_output is None:
        fg_output = str(input_path.with_name(f"{input_path.stem}_fg.fits"))
    if eor_output is None:
        eor_output = str(input_path.with_name(f"{input_path.stem}_eor.fits"))

    if config.true_eor_cube:
        true_eor_path = Path(config.true_eor_cube)
        if not true_eor_path.exists():
            parser.error(f"Reference EoR cube '{true_eor_path}' not found.")
        if config.corr_plot is None:
            default_plot = input_path.with_name(f"{input_path.stem}_eor_corr.png")
            config.corr_plot = str(default_plot)

    if config.init_fg_cube:
        fg_init_path = Path(config.init_fg_cube)
        if not fg_init_path.exists():
            parser.error(f"Initial foreground cube '{fg_init_path}' not found.")
    if config.init_eor_cube:
        eor_init_path = Path(config.init_eor_cube)
        if not eor_init_path.exists():
            parser.error(f"Initial EoR cube '{eor_init_path}' not found.")

    if config.fg_reference_cube:
        ref_path = Path(config.fg_reference_cube)
        if not ref_path.exists():
            parser.error(f"fg_reference_cube path '{ref_path}' not found.")

    _optimize_from_fits(input_path, Path(fg_output), Path(eor_output), config)


if __name__ == "__main__":
    main()
