# Foreground–EoR Separation Prototype

This repository contains a small optimization-based prototype for separating a smooth foreground component and a fluctuating EoR component from a 3D radio data cube. The current focus is on building a clean, modular foundation that can later accommodate realistic PSFs and alternative optimization strategies (e.g., proximal or ADMM-style solvers).

## Background

In many low-frequency radio experiments we observe a cube `y[frequency, x, y]` that mixes bright, spectrally smooth foreground emission with a faint, rapidly varying EoR signal. Instead of forcing `eor = y - fg`, we treat both components as independent tensors and optimize them jointly under four uncertainty-aware objectives:

1. **Data consistency** enforced through a configurable forward model `y_pred = forward_model(fg, eor, psf=None)` and per-pixel measurement errors.
2. **Foreground smoothness** via a third-order finite-difference prior whose mean/variance can come from user inputs or a reference foreground cube.
3. **EoR amplitude prior** that constrains the stochastic nature of the EoR component through configurable means/variances.
4. **Foreground–EoR correlation prior** that encourages weak correlation between the two recovered cubes.

The current implementation assumes an identity PSF but keeps the forward model abstract so it can be swapped out for a convolutional operator without touching the optimization loop.

## Repository Layout

- `separation_optim.py`: Core module with optimizer loop, CLI, config handling, and a synthetic demo in the `__main__` block.
- `losses.py`: Forward model and all loss/regularization utilities (data, smoothness, correlation, rFFT, polynomial priors).
- `powerspec.py`: 1D/2D power spectrum utilities (configurable k-grid, FITS/PNG outputs).

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) (CPU-only build is sufficient for the demo)
- [Astropy](https://www.astropy.org/) for FITS I/O
- [Matplotlib](https://matplotlib.org/) for optional evaluation plots

You can install PyTorch via pip:

```bash
pip install torch
```

or via conda:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Install Astropy via:

```bash
pip install astropy
```

Install Matplotlib via:

```bash
pip install matplotlib
```

## Usage

### Command-line interface

The main entry point expects a FITS cube and produces two FITS files for the separated components:

```bash
python separation_optim.py \
  --input-cube /path/to/observed_cube.fits \
  --fg-output /path/to/foreground_estimate.fits \
  --eor-output /path/to/eor_estimate.fits
```

If you omit the output paths they default to `<input>_fg.fits` and `<input>_eor.fits`. By default the script runs on GPU when `torch.cuda.is_available()`, otherwise it falls back to CPU. You can override this with `--device cpu` or `--device cuda:1`, and optionally set the working dtype through `--dtype float32`.

Each loss term is normalized by its uncertainty:

- **Data fidelity:** use `--data-error 0.05` for a scalar noise level.
- **EoR prior:** specify scalar `--eor-mean` / `--eor-sigma` (defaults are mean = 0, sigma = 0.1).
- **Foreground smoothness:** provide `--fg-smooth-mean`, `--fg-smooth-sigma`, or let the code derive both from a reference cube via `--fg-reference-cube fg_truth.fits` (the reference cube defines the mean/std of the third-order differences and the per-frequency summaries are printed). If you prefer robust statistics when deriving these priors, add `--use-robust-fg-stats` (median + MAE scaled by `--mae-to-sigma-factor`, default 1.4826 to match Gaussian std).
- **FG/EoR correlation:** enforce decorrelation with `--corr-mean` (default 0), `--corr-sigma` (default 0.2), and scale the penalty via `--corr-weight`.
- **Optional rFFT penalty:** set `--loss-mode rfft` to penalize high-frequency energy of the foreground. Configure the fraction of penalized bins via `--fft-percent` (default 0.7, meaning the top 70% of frequency bins are treated as high-frequency), and set priors with `--fft-mean`, `--fft-sigma`.
- **Polynomial priors:** `--loss-mode poly` fits a low-order polynomial to the recovered foreground and penalizes residuals (degree via `--poly-degree`, scale via `--poly-sigma`, weight via `--poly-weight`). `--loss-mode poly_reparam` re-parameterizes the foreground as “polynomial coefficients + residual cube” and penalizes the residual magnitude with the same `poly-*` settings; data/smoothness/EoR/correlation losses are still applied to the reconstructed foreground.
- **Optimizer:** choose Adam (default) or SGD via `--optimizer`; configure SGD momentum with `--momentum`. Polynomial modes on large cubes can be slow—use primarily for testing or smaller inputs.

The weights `alpha` (data), `beta` (smoothness), `gamma` (EoR prior), and `corr_weight` all default to 1.0. The program will emit a warning if you override them so you can double-check that change is intentional.

When coarse initial guesses for either component are available, pass `--init-fg-cube <path>` and/or `--init-eor-cube <path>`. These cubes are only used as starting points for optimization; the usual smoothness/data priors still drive the update.

If you have a reference EoR cube for diagnostics only, add `--true-eor-cube /path/to/reference_eor.fits`. The script will compute the per-frequency correlation between the recovered and reference EoR, log summary statistics, and save a plot (configure the destination with `--corr-plot path/to/plot.png`). The reference cube is **never** used during optimization—it is only for evaluation.

For power-spectrum diagnostics, provide a power config (e.g., `--power-config configs/power.json`). After optimization, the script will compute recovered (and, if available, true) 1D/2D power spectra and save both FITS tables and PNG plots to the configured output directory.

### Configuration files

Repeated argument sets can be stored in a JSON config and overridden from the CLI when needed:

```json
{
  "input_cube": "data/example_cube.fits",
  "fg_output": "outputs/example_fg.fits",
  "eor_output": "outputs/example_eor.fits",
  "optim": {
    "num_iters": 500,
    "lr": 0.05,
    "freq_axis": 0,
    "print_every": 50,
    "device": "cuda:0",
    "dtype": "float32",
    "loss_mode": "base",
    "optimizer_name": "adam",
    "momentum": 0.9,
    "power_config": "configs/power.json"
  },
  "weights": {
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "corr_weight": 1.0,
    "fft_weight": 1.0,
    "poly_weight": 1.0
  },
  "priors": {
    "data_error": 0.05,
    "eor_prior_mean": 0.0,
    "eor_prior_sigma": 0.1,
    "fg_smooth_mean": 0.0,
    "fg_smooth_sigma": 0.05,
    "fg_reference_cube": "data/example_fg_reference.fits",
    "use_robust_fg_stats": false,
    "mae_to_sigma_factor": 1.4826,
    "corr_prior_mean": 0.0,
    "corr_prior_sigma": 0.2,
    "fft_highfreq_percent": 0.7,
    "fft_prior_mean": 0.0,
    "fft_prior_sigma": 1.0,
    "poly_degree": 3,
    "poly_sigma": 0.05
  },
  "evaluation": {
    "true_eor_cube": "data/example_eor_truth.fits",
    "corr_plot": "outputs/example_eor_corr.png"
  },
  "power": {
    "dx": 1.0,
    "dy": 1.0,
    "df": 1.0,
    "df_unit": "mhz",
    "ref_freq_mhz": 150.0,
    "rest_freq_mhz": 1420.40575,
    "H0": 70.0,
    "Om0": 0.3,
    "Ode0": 0.7,
    "freq_axis": 0,
    "nbins_1d": 30,
    "nbins_kperp": 30,
    "nbins_kpar": 30,
    "output_dir": "powerspec"
  },
  "init": {
    "init_fg_cube": "data/example_fg_init.fits",
    "init_eor_cube": "data/example_eor_init.fits"
  }
}
```

Run it via:

```bash
python separation_optim.py --config configs/example.json
```

CLI arguments always override config fields, so you can keep defaults in JSON and tweak particular settings per run.

### Synthetic demo

To quickly verify the environment without FITS files, run the demo:

```bash
python separation_optim.py --run-demo
```

This generates a smooth foreground plus oscillatory EoR cube, adds mild noise, and optimizes the components using Adam while logging the loss terms and final reconstruction errors.

## Customization

- **Forward model / PSF:** Implement a callable that applies your PSF to `fg + eor` and pass it via the `psf` argument.
- **Optimization routine:** Swap Adam for another optimizer by adapting `optimize_components`, or wrap the module in a higher-level solver.
- **Device placement:** Keep your data on GPU (or pass `device=torch.device("cuda")`) so the optimizer stays on the desired accelerator.
- **Regularization:** Replace the placeholder EoR term (`gamma * mean(eor**2)`) with a domain-specific prior when needed.
- **Power spectra:** Set `dx/dy/df` in the power config to the physical spacing of your cube (e.g., Mpc or wavelength units); incorrect spacings will rescale k-axes and power values.

## License

Released under the MIT License © 2024 Zhenghao Zhu. See `LICENSE` for details.

## Next Steps

Future iterations can add realistic interferometric PSFs, multi-resolution smoothness penalties, or plug this loss into a custom FISTA/ADMM solver while reusing the same forward-model interface.
