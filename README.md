# Foreground–EoR Separation Prototype

This repository contains an optimization-based prototype for separating a smooth foreground component and a fluctuating EoR component from a 3D radio data cube.

For a high-level overview and full documentation, see the files under `docs/`:

- `docs/overview.md`: Scientific background and design goals.
- `docs/usage.md`: Command-line interface and usage patterns.
- `docs/config.md`: JSON configuration reference.
- `docs/powerspec.md`: Power-spectrum configuration and outputs.

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

## Quickstart

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
