# Foreground–EoR Separation: Overview

This document gives a high-level overview of the foreground/EoR separation prototype.

## Goal

We model a 3D data cube `y[frequency, x, y]` as a sum of:

- A smooth foreground component `fg[f, x, y]`
- A fluctuating EoR component `eor[f, x, y]`

Both `fg` and `eor` are treated as independent optimization variables and are constrained via:

- Data consistency: a forward model `forward_model(fg, eor, psf=None)`
- Foreground smoothness along the frequency axis
- Optional EoR priors
- Optional foreground–EoR decorrelation
- Optional high-frequency (rFFT) suppression
- Optional polynomial priors (including a reparameterized mode)

The prototype is written in PyTorch and is designed to be modular so that loss terms, optimizers, and power-spectrum diagnostics can be swapped in and out.

## Repository Layout

- `separation_optim.py`: Core optimization module. Implements the optimizer loop, loss wiring, FITS I/O helpers, and the synthetic demo.
- `separation_cli.py`: Command-line entry point. Parses arguments, loads JSON configs, and calls the optimizer.
- `losses.py`: All forward-model and loss/regularization logic (data term, smoothness, EoR prior, correlation prior, rFFT, polynomial priors).
- `powerspec.py`: 1D/2D power-spectrum utilities (physical k-axes, FITS/PNG outputs).
- `utils.py`: Shared helpers (tensor/device management, broadcasted priors, numeric clamps).
- `constants.py`: Numeric constants (epsilons, default sigmas).
- `configs/example.json`: Example training configuration.
- `configs/power.json`: Example power-spectrum configuration.

## Typical Workflow

1. Prepare a cube (FITS) and an optional reference EoR cube.
2. Write a JSON config (e.g., `configs/example.json`) describing:
   - Optimization hyperparameters
   - Loss weights and priors
   - Frequency information (start and spacing)
   - Optional power-spectrum config file
3. Run:

   ```bash
   python separation_cli.py --config configs/example.json
   ```

4. Inspect:
   - Optimization logs (per-iteration loss terms)
   - Optional per-interval EoR correlation checks (if a true EoR cube and `enable_corr_check` are configured)
   - Recovered foreground/EoR cubes (FITS)
   - Power spectra (1D/2D FITS and PNG; 2D k-bins and log/linear plotting are controlled via `configs/power.json`)
