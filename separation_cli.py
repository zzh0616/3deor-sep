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

from separation_optim import OptimizationConfig, _optimize_from_fits, load_config_file, run_synthetic_demo


def parse_cli_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Optimize foreground/EoR separation with configurable priors."
    )
    parser.add_argument("--config", type=str, help="Path to JSON config file with defaults.")
    parser.add_argument("--input-cube", type=str, help="Input FITS cube containing observed data.")
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
        "--fg-smooth-mean",
        type=float,
        help="Scalar prior mean for FG third differences (default 0.0).",
    )
    parser.add_argument(
        "--fg-smooth-sigma",
        type=float,
        help="Scalar prior std for FG third differences (default 0.05).",
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
        "--corr-weight",
        dest="corr_weight",
        type=float,
        help="Weight applied to the correlation prior term (default 1.0).",
    )
    parser.add_argument(
        "--fft-weight",
        dest="fft_weight",
        type=float,
        help="Weight applied to the high-frequency rFFT prior term (default 1.0).",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["base", "rfft", "poly", "poly_reparam"],
        help=(
            "Loss mode: 'base' (default), 'rfft' with high-frequency penalty, "
            "'poly' polynomial prior on the recovered foreground, or 'poly_reparam' polynomial reparameterization."
        ),
    )
    parser.add_argument(
        "--fft-percent",
        type=float,
        help="Fraction of highest frequency bins penalized in rFFT mode (default 0.7).",
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
        "--poly-weight",
        dest="poly_weight",
        type=float,
        help="Weight applied to the polynomial prior term (poly mode).",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        help="Polynomial degree for the foreground prior in poly mode (default 3).",
    )
    parser.add_argument(
        "--poly-sigma",
        type=float,
        help="Std used to scale polynomial prior residuals (default 0.05).",
    )
    parser.add_argument(
        "--freq-start-mhz",
        type=float,
        help="Starting frequency of the cube in MHz (for polynomial modes).",
    )
    parser.add_argument(
        "--freq-delta-mhz",
        type=float,
        help="Frequency spacing of the cube in MHz (for polynomial modes).",
    )
    parser.add_argument(
        "--true-eor-cube",
        type=str,
        help="FITS cube containing reference EoR for evaluation plots (not used for training).",
    )
    parser.add_argument(
        "--corr-plot",
        type=str,
        help="File path for saving the per-frequency correlation plot.",
    )
    parser.add_argument(
        "--enable-corr-check",
        action="store_true",
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
        "loss_mode",
        "freq_axis",
        "print_every",
        "device",
        "dtype",
        "data_error",
        "eor_prior_mean",
        "eor_prior_sigma",
        "fg_smooth_mean",
        "fg_smooth_sigma",
        "fg_reference_cube",
        "use_robust_fg_stats",
        "mae_to_sigma_factor",
        "corr_prior_mean",
        "corr_prior_sigma",
        "corr_weight",
        "fft_highfreq_percent",
        "fft_prior_mean",
        "fft_prior_sigma",
        "poly_weight",
        "poly_degree",
        "poly_sigma",
        "freq_start_mhz",
        "freq_delta_mhz",
        "power_config",
        "power_output_dir",
        "true_eor_cube",
        "corr_plot",
        "init_fg_cube",
        "init_eor_cube",
        "enable_corr_check",
        "corr_check_every",
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
        flattened: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flattened.update(_flatten_config(value))
            else:
                flattened[key] = value
        return flattened

    config.update_from_dict(_flatten_config(config_data))
    config.update_from_dict(_collect_cli_overrides(args))

    if args.run_demo:
        run_synthetic_demo(config)
        return

    input_cube = args.input_cube or config_data.get("input_cube")
    if input_cube is None:
        parser.error("Please specify --input-cube or set 'input_cube' in the config file.")
    input_path = Path(input_cube)
    if not input_path.exists():
        parser.error(f"Input cube '{input_path}' not found.")

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

