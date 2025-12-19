# Usage Guide

This document summarizes the main command-line interface and configuration options.

## Command-Line Interface

The main entry point is `separation_cli.py`. Typical invocation:

```bash
python separation_cli.py \
  --config configs/example.json \
  --device cuda:0
```

Key CLI arguments:

- `--config`: Path to the main JSON configuration file.
- `--input-cube`: FITS cube containing observed data (overrides config).
- `--mask-cube`: Optional FITS mask (2D or 3D) applied to all inputs before `cut_xy` and loss evaluation.
- `--fg-output` / `--eor-output`: Output FITS paths for foreground/EoR estimates.
- `--device`: Device string, e.g., `cuda:0` or `cpu`.
- `--loss-mode`: `base`, `rfft`, `poly`, `poly_reparam`, or `lagcorr`.
- `--extra-loss-start-iter` / `--extra-loss-ramp-iters`: Delay and (optionally) ramp extra loss terms for non-base modes.
- `--cut-xy-enabled` / `--cut-xy-unit` / `--cut-xy-center-x` / `--cut-xy-center-y` / `--cut-xy-size`: Spatial XY cropping controls (see `docs/config.md`).
- `--optimizer`: `adam` (default) or `sgd`.
- `--momentum`: Momentum for SGD (default 0.9).
- `--freq-start-mhz` / `--freq-delta-mhz`: Starting frequency and spacing (MHz) for polynomial modes.
- `--lagcorr-weight`: Weight for the frequency-lag autocorrelation prior (lagcorr mode).
- `--lagcorr-unit`: Units for `lagcorr_intervals` (`mhz` requires `--freq-delta-mhz`; `chan` requires integer intervals, lagcorr mode).
- `--lagcorr-max-pairs`: Cap the number of pairs evaluated per lag (lagcorr mode).
- `--true-eor-cube`: Reference EoR FITS cube (evaluation only, not used in training).
- `--diagnose-input`: Compute FG↔FG and EoR↔EoR frequency-lag correlations for the true cubes over lags `1..floor((F-1)/2)` and write a summary + CSV + plot, plus a loss breakdown and FG smoothness stats (requires `fg_reference_cube` and `true_eor_cube`), then exit. (`--report-true-signal-corr` is a deprecated alias.)
- `--corr-plot`: Path for saving EoR correlation plot.
- `--enable-corr-check`: Periodically compute the mean correlation between recovered and reference EoR during optimization (requires a true EoR cube).
- `--corr-check-every`: Iteration interval for the correlation checks (default 500).
- `--power-config`: JSON config for power spectra (see `docs/powerspec.md`).
- `--power-output-dir`: Directory for power-spectrum outputs.
- `--run-demo`: Run the synthetic demo instead of loading a FITS cube.

Other arguments (loss weights, priors, etc.) mirror the config fields and are rarely needed on the CLI unless you want to override specific settings.

## Configuration Structure

The main config is a JSON file with the following top-level sections:

- `input_cube`: Path to observed cube (FITS).
- `fg_output` / `eor_output`: Output FITS paths.
- `optim`: Optimization parameters and high-level behavior.
- `weights`: Loss weights.
- `priors`: Priors and hyperparameters for loss terms.
- `evaluation`: Evaluation-only inputs (true EoR cube, correlation plot path).
- `power`: Power-spectrum configuration (optional).
- `init`: Optional initial guesses.

See `docs/config.md` for a detailed field reference.
