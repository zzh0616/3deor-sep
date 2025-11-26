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
  "output_dir": "powerspec",
  "stat_mode": "median"
}
```

### Spatial Spacing

- `dx`, `dy` (`float`): Spatial spacing along x/y axes.
- `unit_x`, `unit_y` (`str`): Units for `dx`/`dy`. Supported:
  - Length units: `"mpc"`, `"mpc/h"`, `"kpc"`, `"kpc/h"`, `"gpc"`, `"gpc/h"`.
  - Angular units: `"rad"`, `"deg"`, `"arcmin"`, `"arcsec"`.

When using angular units, the code converts angles on the sky to transverse comoving distances using a reference redshift. The reference redshift is obtained from:

- `ref_freq_mhz` and `rest_freq_mhz` if a line frequency is provided (via `z = (rest_freq / freq) - 1`), or
- `ref_redshift` (see below) if given explicitly.

### Frequency Axis

- `df` (`float`): Radial spacing along the z-axis (frequency or redshift, depending on `unit_f`).
- `unit_f` (`str`): Units for `df`. Supported:
  - Length units: `"mpc"`, `"mpc/h"`, `"kpc"`, `"kpc/h"`, `"gpc"`, `"gpc/h"` (interpreted directly as comoving spacing).
  - Frequency units: `"mhz"`, `"hz"` (requires `ref_freq_mhz` and `rest_freq_mhz`).
  - Redshift units: `"redshift"` or `"z"` (requires `ref_redshift`).
- `ref_freq_mhz` (`float`): Starting (or central) frequency of the cube in MHz; required when `unit_f` is a frequency unit.
- `rest_freq_mhz` (`float`): Rest frequency of the line (default 1420.40575 MHz for HI 21cm).
- `ref_redshift` (`float`): Reference redshift when `unit_f` is `"redshift"`/`"z"` or when `dx`/`dy` use angular units without a line frequency.

When `unit_f` is `"mhz"`/`"hz"`, the code:

1. Builds the frequency array: `freq_n = ref_freq_mhz + n * df`.
2. Converts to redshift `z = (rest_freq / freq) - 1`.
3. Uses the configured cosmology to map `z` to comoving distance `χ(z)`.
4. Uses the mean spacing in `χ` as the effective comoving `df` (and warns if the spacing is significantly non-uniform).

When `unit_f` is `"redshift"`/`"z"`, the code:

1. Builds a redshift grid: `z_n = ref_redshift + n * df`.
2. Maps `z_n` to comoving distance `χ(z_n)`.
3. Uses the mean spacing in `χ` as the effective comoving `df` (and warns if the spacing is significantly non-uniform).

### Cosmology

- `H0` (`float`): Hubble parameter in km/s/Mpc.
- `Om0` (`float`): Matter density parameter.
- `Ode0` (`float`): Dark-energy density parameter (optional; used only for sanity checks; the actual cosmology is flat with `Ω_Λ = 1 - Ω_m`).

### Binning

- `freq_axis` (`int`): Frequency axis index (should match `freq_axis` in the main config).
- `nbins_1d` (`int`): Number of k bins for the 1D spherical average.
- `nbins_kperp`, `nbins_kpar` (`int`): Number of bins in k_perp and k_par for the 2D power spectrum.
- `stat_mode` (`str`): How to average within each k-bin:
  - `"median"` (default): Robust median over all modes in the bin (recommended for noisy or non-Gaussian fields).
  - `"mean"`: Simple arithmetic mean (useful for idealized simulations).

### Outputs

For a given cube and power config, `powerspec.py` produces:

- 1D:
  - `power1d_rec.fits` / `power1d_true.fits` / `power1d_rel.fits`
  - `power1d.png`
- 2D:
  - `power2d_rec.fits` / `power2d_true.fits` / `power2d_rel.fits`
  - `power2d.png` / `power2d_rel.png` (relative plot is clipped to ±500%).

All power spectra are windowed (Hann) and normalized by the window energy and physical volume.

Internally, only Fourier modes inside the inscribed k-space sphere (where all three spatial
dimensions have support) are used when forming 1D and 2D averages. A placeholder hook for a
future uv/PSF/visibility mask is present in the implementation and can be activated when the
pipeline is extended to interferometric data.
