# Configuration Reference

This document describes the JSON configuration fields used by the optimizer.

## Top-Level Fields

- `input_cube` (`str`): Path to observed FITS cube.
- `fg_output` (`str`): Output FITS path for the recovered foreground.
- `eor_output` (`str`): Output FITS path for the recovered EoR.

## `optim` Section

- `num_iters` (`int`): Number of optimization iterations.
- `lr` (`float`): Learning rate.
- `freq_axis` (`int`): Index of the frequency axis in the cube.
- `print_every` (`int`): Logging frequency (iterations).
- `device` (`str`): Device string, e.g. `cuda:0` or `cpu`.
- `dtype` (`str`): Torch dtype name (e.g., `float32`).
- `loss_mode` (`str`): One of:
  - `"base"`: Only the base loss (data + smoothness + EoR prior + correlation).
  - `"rfft"`: Base loss + high-frequency (rFFT) penalty.
  - `"poly"`: Base loss + polynomial prior (fitted on the recovered foreground).
  - `"poly_reparam"`: Base loss with foreground parameterized as polynomial coefficients + residual.
- `optimizer_name` (`str`): `"adam"` (default) or `"sgd"`.
- `momentum` (`float`): SGD momentum (ignored for Adam).
- `freq_start_mhz` (`float`): Starting frequency of the cube (MHz) for polynomial modes.
- `freq_delta_mhz` (`float`): Frequency spacing of the cube (MHz) for polynomial modes.
- `power_config` (`str`): Path to power-spectrum config JSON (optional).

## `weights` Section

- `alpha` (`float`): Data term weight.
- `beta` (`float`): Foreground smoothness weight.
- `gamma` (`float`): EoR prior weight.
- `corr_weight` (`float`): Correlation prior weight.
- `fft_weight` (`float`): rFFT high-frequency penalty weight.
- `poly_weight` (`float`): Polynomial prior weight.

All weights default to `1.0`. The code prints a warning if you deviate from 1.0.

## `priors` Section

- `data_error` (`float`): Scalar data-error sigma.
- `eor_prior_mean` (`float`): EoR prior mean.
- `eor_prior_sigma` (`float`): EoR prior sigma.
- `fg_smooth_mean` (`float`): Foreground third-difference prior mean.
- `fg_smooth_sigma` (`float`): Foreground third-difference prior sigma.
- `fg_reference_cube` (`str`): FITS cube used to derive foreground smoothness stats (mean/std of third differences).
- `use_robust_fg_stats` (`bool`): If `true`, use median/MAE (scaled) instead of mean/std for smoothness/FFT priors.
- `mae_to_sigma_factor` (`float`): MAE-to-sigma scaling (default ~1.48).
- `corr_prior_mean` (`float`): Correlation prior mean.
- `corr_prior_sigma` (`float`): Correlation prior sigma.
- `fft_highfreq_percent` (`float`): Fraction of highest-frequency bins to penalize in rFFT mode (0–1).
- `fft_prior_mean` (`float`): High-frequency energy prior mean.
- `fft_prior_sigma` (`float`): High-frequency energy prior sigma.
- `poly_degree` (`int`): Polynomial degree for polynomial priors.
- `poly_sigma` (`float`): Residual sigma for polynomial priors.

## `evaluation` Section

- `true_eor_cube` (`str`): Reference EoR FITS cube (evaluation only).
- `corr_plot` (`str`): Output path for the EoR correlation plot.
- `enable_corr_check` (`bool`): If `true`, periodically compute the mean correlation between the recovered and reference EoR during optimization. Requires `true_eor_cube`.
- `corr_check_every` (`int`): Iteration interval for correlation checks (default 500).

## `power` Section

See `docs/powerspec.md` for a dedicated description.

## `init` Section

- `init_fg_cube` (`str`): Initial foreground guess (FITS cube).
- `init_eor_cube` (`str`): Initial EoR guess (FITS cube).
