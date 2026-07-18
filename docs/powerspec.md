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
  "freq_grid_start_mhz": 148.6,
  "rest_freq_mhz": 1420.40575,
  "H0": 67.8,
  "Om0": 0.308,
  "Ode0": 0.692,
  "freq_axis": 0,
  "nbins_1d": 30,
  "nbins_kperp": 30,
  "nbins_kpar": 30,
  "output_dir": "powerspec",
  "stat_mode": "median",
  "demean_mode": "global",
  "log_bins_2d": true,
  "log_power_2d": true
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
- `ref_freq_mhz` (`float`): Reference frequency used for angular-to-transverse conversion, normally the cube center; required when `unit_f` is a frequency unit.
- `freq_grid_start_mhz` (`float`, optional): Frequency of the first cube sample. If omitted, the code uses `ref_freq_mhz` as the first sample for backward compatibility.
- `rest_freq_mhz` (`float`): Rest frequency of the line (default 1420.40575 MHz for HI 21cm).
- `ref_redshift` (`float`): Reference redshift when `unit_f` is `"redshift"`/`"z"` or when `dx`/`dy` use angular units without a line frequency.

When `unit_f` is `"mhz"`/`"hz"`, the code:

1. Builds the frequency array: `freq_n = freq_grid_start_mhz + n * df` (or uses `ref_freq_mhz` as the legacy fallback).
2. Converts to redshift `z = (rest_freq / freq) - 1`.
3. Uses the configured cosmology to map `z` to comoving distance `χ(z)`.
4. Uses the mean spacing in `χ` as the effective comoving `df` (and warns if the spacing is significantly non-uniform).

When `unit_f` is `"redshift"`/`"z"`, the code:

1. Builds a redshift grid: `z_n = ref_redshift + n * df`.
2. Maps `z_n` to comoving distance `χ(z_n)`.
3. Uses the mean spacing in `χ` as the effective comoving `df` (and warns if the spacing is significantly non-uniform).

### Cosmology

- `H0` (`float`): Hubble parameter in km/s/Mpc. The project default is aligned with the
  EOS lightcone used in `e2esim` (`H0=67.8`).
- `Om0` (`float`): Matter density parameter. The project default is `Om0=0.308`.
- `Ode0` (`float`): Dark-energy density parameter (optional; used only for sanity checks; the actual cosmology is flat with `Ω_Λ = 1 - Ω_m`, i.e. `0.692` for the project default).

### Binning

- `freq_axis` (`int`): Frequency axis index (should match `freq_axis` in the main config).
- `nbins_1d` (`int`): Number of k bins for the 1D spherical average.
- `nbins_kperp`, `nbins_kpar` (`int`): Number of bins in k_perp and k_par for the 2D power spectrum.
- `stat_mode` (`str`): How to average within each k-bin:
  - `"median"` (default): Robust median over all modes in the bin (recommended for noisy or non-Gaussian fields).
  - `"mean"`: Simple arithmetic mean (useful for idealized simulations).
- `demean_mode` (`str`): Pre-FFT mean subtraction mode:
  - `"global"` (default): subtract one cube-wide mean.
  - `"per_freq_spatial"`: subtract the spatial mean of each frequency slice separately.
  - `"none"`: do not subtract a mean before FFT.
  This matters for lightcone cubes because slice-to-slice mean evolution can otherwise feed very low-`k_parallel` structure.
- `log_bins_2d` (`bool`): If `true` (default), use logarithmically spaced k_perp/k_par bins for the 2D spectrum; if `false`, use linear spacing.
- `log_power_2d` (`bool`): If `true` (default), plot 2D power spectra in log10 scale; if `false`, plot linear P(k).

### EoR-window selection

- `eor_window_kpar_min`: Low-line-of-sight-mode floor.
- `eor_window_wedge_slope` and `eor_window_wedge_intercept`: Require
  `kpar >= slope * kperp + intercept`.
- `eor_window_kperp_min`, `eor_window_kperp_max`, and `eor_window_kpar_max`:
  Optional instrument/support bounds.
- `eor_window_bin_policy`: `"center"` preserves the historical bin-center cut;
  `"all_modes"`, `"any_modes"`, and `"majority"` use the exact Fourier modes
  contributing to each bin. Coarse frequency grids should not use `"center"` for
  a physical threshold.

`compute_power_spectra` records a per-profile mode fraction for every populated
PS2D bin. A partially selected bin must not silently be treated as wholly inside
the window. The strict dataset configs use `all_modes`; analyses that need the
partial support itself should cut modes before cylindrical averaging or redesign
the bins to align with the physical boundary.

### Outputs

For a given cube and power config, `powerspec.py` produces:

- 1D:
  - `power1d_rec.fits` / `power1d_true.fits` / `power1d_rel.fits`
  - `power1d_counts.fits`
  - `power1d.png`
- 2D:
  - `power2d_rec.fits` / `power2d_true.fits` / `power2d_rel.fits`
  - `power2d_counts.fits`
  - `power2d_eor_window_mode_fraction*.fits` when EoR-window evaluation is enabled
  - `power2d.png` / `power2d_rel.png` (relative plot uses absolute relative error clipped to 1%-100% for log display).
  - `power_axes.json` (stores centers/edges and the demeaning mode used)

All power spectra are windowed (Hann) and normalized by the window energy and physical volume.

Internally, only Fourier modes inside the inscribed k-space sphere (where all three spatial
dimensions have support) are used when forming the 1D spectrum. For the 2D spectrum, the full
line-of-sight range is retained while transverse modes are restricted to the inscribed circle in
(kx, ky). A placeholder hook for a future uv/PSF/visibility mask is present in the implementation
and can be activated when the
pipeline is extended to interferometric data. When `compute_power_spectra` is given a CUDA tensor
as input (e.g., from the training pipeline), the 3D FFT is evaluated on the GPU via `torch.fft`
before results are moved back to CPU/NumPy for binning and plotting.

Each `k_perp` and `k_par` coordinate is range-checked before the pair is flattened into a 2D bin
index. Modes exactly beyond a rightmost bin edge are excluded; they must not wrap into the next
`k_perp` row. For an even number of frequency samples this currently excludes the radial Nyquist
layer; reports that compare integrated power must state this convention explicitly.

## Shared e2esim Backend Plan

`e2esim.analysis.powerspec` is the planned shared implementation for 1D/2D
power-spectrum diagnostics across e2esim and fg_rmw. The e2esim implementation
intentionally keeps a light dependency profile: it supports NumPy for CPU and
CuPy for GPU FFT, but does not import or depend on Torch.

For fg_rmw, the intended migration path is to call e2esim through a thin
compatibility wrapper. If the input cube is already a CUDA `torch.Tensor`, pass
it directly to e2esim with the CuPy backend and require explicit failure on
fallback:

```python
from e2esim.analysis.powerspec import PowerSpecConfig, compute_power_spectra

result = compute_power_spectra(
    cube_torch_cuda,
    cfg,
    backend="cupy",
    device="cuda:0",
    dtype="float32",
    fail_on_fallback=True,
)
```

This path should use DLPack (`__dlpack__`) to convert the Torch CUDA tensor into
a CuPy array without making Torch an e2esim dependency. For large production
runs, silent fallback through CPU should be avoided because it can hide large
data transfers and make runtime hard to interpret.

Do not remove the existing Torch FFT path in `3dnet/powerspec.py` yet. Keep it
as a legacy fallback until these checks have passed:

- Numerical parity: compare fg_rmw Torch FFT, e2esim NumPy, and e2esim CuPy on
  representative cubes.
- Metric parity: compare `p1d`, `p2d`, EoR-window masks, and EoR-window metrics.
- Performance parity: verify the CuPy/DLPack path does not introduce hidden CPU
  roundtrips for CUDA tensors.
- Reproducibility: regenerate at least one existing fg_rmw analysis result
  within expected floating-point tolerance.

After those checks, fg_rmw can default to the e2esim backend while keeping the
Torch route available as `legacy_torch` or a similar explicit fallback for old
result reproduction.
