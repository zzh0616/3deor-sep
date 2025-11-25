# Power Spectrum Configuration

This document describes the configuration used for 1D/2D power-spectrum diagnostics.

## Power Config (`power.json`)

Example (`configs/power.json`):

```json
{
  "dx": 1.0,
  "dy": 1.0,
  "df": 0.1,
  "unit_x": "mpc",
  "unit_y": "mpc",
  "unit_f": "mhz",
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
}
```

### Spatial Spacing

- `dx`, `dy` (`float`): Spatial spacing along x/y axes.
- `unit_x`, `unit_y` (`str`): Units for dx/dy, currently `"mpc"` or `"mpc/h"`.

### Frequency Axis

- `df` (`float`): Frequency spacing (if `unit_f` is a frequency unit).
- `unit_f` (`str`): `"mpc"` if df is already a comoving spacing, otherwise `"mhz"` or `"hz"`.
- `ref_freq_mhz` (`float`): Starting (or central) frequency of the cube in MHz; required when `unit_f` is a frequency unit.
- `rest_freq_mhz` (`float`): Rest frequency of the line (default 1420.40575 MHz for HI 21cm).

When `unit_f` is `"mhz"`/`"hz"`, the code:

1. Builds the frequency array: `freq_n = ref_freq_mhz + n * df`.
2. Converts to redshift `z = (rest_freq / freq) - 1`.
3. Uses the configured cosmology to map `z` to comoving distance `χ(z)`.
4. Uses the mean spacing in `χ` as the effective comoving df.

### Cosmology

- `H0` (`float`): Hubble parameter in km/s/Mpc.
- `Om0` (`float`): Matter density parameter.
- `Ode0` (`float`): Dark-energy density parameter (optional; if omitted, `1 - Om0` is used).

### Binning

- `freq_axis` (`int`): Frequency axis index (should match `freq_axis` in the main config).
- `nbins_1d` (`int`): Number of k bins for the 1D spherical average.
- `nbins_kperp`, `nbins_kpar` (`int`): Number of bins in k_perp and k_par for the 2D power spectrum.

### Outputs

For a given cube and power config, `powerspec.py` produces:

- 1D:
  - `power1d_rec.fits` / `power1d_true.fits` / `power1d_rel.fits`
  - `power1d.png`
- 2D:
  - `power2d_rec.fits` / `power2d_true.fits` / `power2d_rel.fits`
  - `power2d.png` / `power2d_rel.png` (relative plot is clipped to ±500%).

All power spectra are windowed (Hann) and normalized by the window energy and physical volume.

