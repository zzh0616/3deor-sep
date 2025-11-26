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
- `--fg-output` / `--eor-output`: Output FITS paths for foreground/EoR estimates.
- `--device`: Device string, e.g., `cuda:0` or `cpu`.
- `--loss-mode`: `base`, `rfft`, `poly`, or `poly_reparam`.
- `--optimizer`: `adam` (default) or `sgd`.
- `--momentum`: Momentum for SGD (default 0.9).
- `--freq-start-mhz` / `--freq-delta-mhz`: Starting frequency and spacing (MHz) for polynomial modes.
- `--true-eor-cube`: Reference EoR FITS cube (evaluation only, not used in training).
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
