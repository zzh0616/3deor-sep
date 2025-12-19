# Configuration Reference

This document describes the JSON configuration fields used by the optimizer.

## Top-Level Fields

- `input_cube` (`str`): Path to observed FITS cube.
- `mask_cube` (`str`): Optional FITS mask path (2D or 3D) applied to all inputs before `cut_xy` and loss evaluation.
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
  - `"lagcorr"`: Base loss + frequency-lag autocorrelation priors for FG and EoR (see lagcorr settings below).
- `optimizer_name` (`str`): `"adam"` (default) or `"sgd"`.
- `momentum` (`float`): SGD momentum (ignored for Adam).
- `freq_start_mhz` (`float`): Starting frequency of the cube (MHz) for polynomial modes.
- `freq_delta_mhz` (`float`): Frequency spacing of the cube (MHz) for polynomial modes.
- `freqs_mhz_path` (`str`): Optional path to a 1D array (text or `.npy`) containing the cube frequency values in MHz.
- `power_config` (`str`): Path to power-spectrum config JSON (optional).
- `extra_loss_start_iter` (`int`): When `loss_mode != "base"`, only base loss terms are used before this iteration (default 500).
- `extra_loss_ramp_iters` (`int`): If > 0, ramp the extra loss scale from 0 to 1 over this many iterations after `extra_loss_start_iter` (default 0 = hard switch).

## `cut_xy` Section

If enabled, all inputs are cropped in the spatial (XY) plane before optimization and analysis. Frequency is never cropped.

- `enabled` (`bool`): Enable spatial cropping (default `false`).
- `unit` (`str`): `"frac"` or `"px"`. `"frac"` interprets `center_x/center_y/size` as fractions of the image; `"px"` uses pixel indices and pixel sizes.
- `center_x` / `center_y` (`float` or `int`): Crop center (default image center; `0.5/0.5` for `"frac"`, `Nx//2` / `Ny//2` for `"px"`).
- `size` (`float` or `int`): Output square side length (default `0.5 * min(Nx, Ny)`; `0.5` for `"frac"`, integer pixels for `"px"`). When the crop window exceeds image bounds, the window is clamped so the output size stays fixed.
  
The crop indices are saved into the output FITS headers (e.g., `CUTX0/CUTX1/CUTY0/CUTY1`).

## `weights` Section

- `alpha` (`float`): Data term weight.
- `beta` (`float`): Foreground smoothness weight.
- `gamma` (`float`): EoR prior weight.
- `corr_weight` (`float`): Correlation prior weight.
- `lagcorr_weight` (`float`): Frequency-lag autocorrelation prior weight (lagcorr mode).
- `fft_weight` (`float`): rFFT high-frequency penalty weight.
- `poly_weight` (`float`): Polynomial prior weight.

All weights default to `1.0`. The code prints a warning if you deviate from 1.0.

## `priors` Section

- `data_error` (`float`): Scalar data-error sigma.
- `eor_prior_mean` (`float`): EoR prior mean.
- `eor_prior_sigma` (`float`): EoR prior sigma.
- `fg_smooth_mean` (`float`): Foreground third-difference prior mean.
- `fg_smooth_sigma` (`float`): Foreground third-difference prior sigma.
- `fg_reference_cube` (`str`): FITS cube used to derive foreground smoothness stats (mean/std of third differences). If `fg_smooth_mean` or `fg_smooth_sigma` are explicitly set, those values take precedence and the derivation is skipped.
- `use_robust_fg_stats` (`bool`): If `true`, use median/MAE (scaled) instead of mean/std for smoothness/FFT priors.
- `mae_to_sigma_factor` (`float`): MAE-to-sigma scaling (default ~1.48).
- `corr_prior_mean` (`float`): Correlation prior mean.
- `corr_prior_sigma` (`float`): Correlation prior sigma.
- `lagcorr_unit` (`str`): Units for `lagcorr_intervals`, one of `"mhz"` or `"chan"` (`"mhz"` requires `freq_delta_mhz`; `"chan"` requires integer intervals).
- `lagcorr_intervals` (`list[float]`): Frequency-lag list (default `[0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 7.5]` in MHz).
- `fg_lagcorr_mean` (`list[float]`): Expected FG autocorrelation for each lag (same length as `lagcorr_intervals`).
- `fg_lagcorr_sigma` (`list[float]`): Expected FG autocorrelation sigma for each lag (same length as `lagcorr_intervals`).
- `eor_lagcorr_mean` (`list[float]`): Expected EoR autocorrelation for each lag (same length as `lagcorr_intervals`).
- `eor_lagcorr_sigma` (`list[float]`): Expected EoR autocorrelation sigma for each lag (same length as `lagcorr_intervals`).
- `lagcorr_max_pairs` (`int`): Optional cap on the number of slice pairs used per lag (default: use all available pairs).
- `fft_highfreq_percent` (`float`): Fraction of highest-frequency bins to penalize in rFFT mode (0–1).
- `fft_prior_mean` (`float`): High-frequency energy prior mean.
- `fft_prior_sigma` (`float`): High-frequency energy prior sigma.
- `poly_degree` (`int`): Polynomial degree for polynomial priors.
- `poly_sigma` (`float`): Residual sigma for polynomial priors.

## `evaluation` Section

- `true_eor_cube` (`str`): Reference EoR FITS cube (evaluation only).
- `corr_plot` (`str`): Output path for the EoR correlation plot.
- `diagnose_input` (`bool`): If `true`, compute frequency-lag correlations for the true foreground (`fg_reference_cube`) and true EoR (`true_eor_cube`) cubes over lags `1..floor((F-1)/2)` (no FG↔EoR cross-correlation), write a summary report + detailed CSV + plot, and include a loss breakdown + smoothness stats (2nd/3rd finite differences for FG and EoR), then exit. (`report_true_signal_corr` is a deprecated alias.)
- `enable_corr_check` (`bool`): If `true`, periodically compute the mean correlation between the recovered and reference EoR during optimization. Requires `true_eor_cube`.
- `corr_check_every` (`int`): Iteration interval for correlation checks (default 500).

## `power` Section

See `docs/powerspec.md` for a dedicated description.

## `init` Section

- `init_fg_cube` (`str`): Initial foreground guess (FITS cube).
- `init_eor_cube` (`str`): Initial EoR guess (FITS cube).
