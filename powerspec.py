#!/usr/bin/env python3
"""
Power spectrum utilities for foreground/EoR separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from constants import EPS_LOSS, EPS_STD

try:  # optional torch dependency for GPU FFT
    import torch  # type: ignore

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch not available
    torch = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class PowerSpecConfig:
    dx: float  # spacing along x
    dy: float  # spacing along y
    df: float  # spacing along frequency axis (Hz or MHz if unit_f is a frequency unit)
    unit_x: str = "mpc"  # e.g., "mpc", "mpc/h", "kpc", "arcmin", etc.
    unit_y: str = "mpc"
    unit_f: str = "mpc"  # "mpc"/"mpc/h"/"kpc" if already physical, otherwise "mhz", "hz", or "redshift"
    ref_freq_mhz: Optional[float] = None  # central/starting frequency (MHz) if unit_f is frequency
    ref_redshift: Optional[float] = None  # reference redshift (for angular units or redshift axis)
    rest_freq_mhz: float = 1420.40575  # HI 21cm default
    H0: float = 70.0
    Om0: float = 0.3
    Ode0: Optional[float] = None
    freq_axis: int = 0
    nbins_1d: int = 30
    nbins_kperp: int = 30
    nbins_kpar: int = 30
    output_dir: str = "powerspec"
    stat_mode: str = "median"  # "median" (default, robust) or "mean"
    log_bins_2d: bool = True  # use log-spaced k_perp/k_par bins for 2D spectra
    log_power_2d: bool = True  # plot 2D power spectra in log10 scale

def _cosmo_from_cfg(cfg: PowerSpecConfig) -> FlatLambdaCDM:
    """
    Build a flat LambdaCDM cosmology from configuration.
    Astropy's FlatLambdaCDM enforces flatness via Om0; Ode0 is implied as 1-Om0.
    """
    if cfg.Ode0 is not None:
        implied = 1.0 - cfg.Om0
        if abs(cfg.Ode0 - implied) / (abs(implied) + EPS_STD) > 0.05:
            print(
                "Warning: Ode0 is ignored by FlatLambdaCDM; using a flat model with "
                f"Om0={cfg.Om0:.3f} (implied Ode0={implied:.3f})."
            )
    return FlatLambdaCDM(H0=cfg.H0 * u.km / u.s / u.Mpc, Om0=cfg.Om0)


def _resolve_length_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert length-like spacings to Mpc.
    Supports Mpc, Mpc/h, kpc, kpc/h, Gpc, and Gpc/h.
    """
    unit_low = unit.lower()
    h = cfg.H0 / 100.0
    if unit_low == "mpc":
        return value
    if unit_low == "mpc/h":
        return value / h
    if unit_low == "kpc":
        return value / 1e3
    if unit_low == "kpc/h":
        return (value / 1e3) / h
    if unit_low == "gpc":
        return value * 1e3
    if unit_low == "gpc/h":
        return (value * 1e3) / h
    raise ValueError(
        f"Unsupported length unit '{unit}'. Use 'mpc', 'mpc/h', 'kpc', 'kpc/h', 'gpc', or 'gpc/h'."
    )


def _resolve_angular_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert angular spacing on the sky to Mpc using a reference redshift.
    """
    unit_low = unit.lower()
    if unit_low == "rad":
        angle_rad = value
    elif unit_low == "deg":
        angle_rad = math.radians(value)
    elif unit_low == "arcmin":
        angle_rad = math.radians(value / 60.0)
    elif unit_low == "arcsec":
        angle_rad = math.radians(value / 3600.0)
    else:
        raise ValueError(
            f"Unsupported angular unit '{unit}'. Use 'rad', 'deg', 'arcmin', or 'arcsec'."
        )

    cosmo = _cosmo_from_cfg(cfg)
    if cfg.ref_freq_mhz is not None:
        z_ref = (cfg.rest_freq_mhz / cfg.ref_freq_mhz) - 1.0
    elif cfg.ref_redshift is not None:
        z_ref = cfg.ref_redshift
    else:
        raise ValueError(
            "ref_freq_mhz or ref_redshift is required when using angular units for dx/dy."
        )
    chi = cosmo.comoving_distance(z_ref).to(u.Mpc).value
    return float(chi * angle_rad)


def _resolve_spatial_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert a spatial spacing along x or y to Mpc, supporting both length and angular units.
    """
    unit_low = unit.lower()
    if unit_low in {"mpc", "mpc/h", "kpc", "kpc/h", "gpc", "gpc/h"}:
        return _resolve_length_spacing(value, unit_low, cfg)
    if unit_low in {"rad", "deg", "arcmin", "arcsec"}:
        return _resolve_angular_spacing(value, unit_low, cfg)
    raise ValueError(
        f"Unsupported spatial unit '{unit}'. Use length units "
        "('mpc', 'mpc/h', 'kpc', 'kpc/h', 'gpc', 'gpc/h') or angular units "
        "('rad', 'deg', 'arcmin', 'arcsec')."
    )


def _frequency_spacing_to_mpc(cfg: PowerSpecConfig, nf: int) -> float:
    """
    Convert spacing along the radial axis to Mpc.
    Supports:
      - Length units: mpc, mpc/h, kpc, kpc/h, gpc, gpc/h
      - Frequency units: mhz, hz (needs ref_freq_mhz and rest_freq_mhz)
      - Redshift units: redshift, z (needs ref_redshift)
    """
    unit_low = cfg.unit_f.lower()
    if unit_low in {"mpc", "mpc/h", "kpc", "kpc/h", "gpc", "gpc/h"}:
        return _resolve_length_spacing(cfg.df, unit_low, cfg)

    if unit_low in {"mhz", "hz"}:
        if cfg.ref_freq_mhz is None:
            raise ValueError("ref_freq_mhz is required when unit_f is a frequency unit.")

        cosmo = _cosmo_from_cfg(cfg)
        ref_freq = cfg.ref_freq_mhz * u.MHz
        df = cfg.df * (u.MHz if unit_low == "mhz" else u.Hz)
        freqs = ref_freq + np.arange(nf) * df
        z = (cfg.rest_freq_mhz * u.MHz / freqs - 1.0).decompose()
        chi = cosmo.comoving_distance(z).to(u.Mpc).value
        dchi = np.diff(chi)
        dchi_abs = np.abs(dchi)
        dchi_mean = float(np.mean(dchi_abs))
        if np.std(dchi_abs) / (np.abs(dchi_mean) + EPS_STD) > 0.05:
            print(
                "Warning: frequency spacing is not uniform in comoving distance; using mean spacing."
            )
        return dchi_mean

    if unit_low in {"redshift", "z"}:
        if cfg.ref_redshift is None:
            raise ValueError("ref_redshift is required when unit_f is a redshift unit.")
        cosmo = _cosmo_from_cfg(cfg)
        z0 = cfg.ref_redshift
        zs = z0 + np.arange(nf) * cfg.df
        chi = cosmo.comoving_distance(zs).to(u.Mpc).value
        dchi = np.diff(chi)
        dchi_abs = np.abs(dchi)
        dchi_mean = float(np.mean(dchi_abs))
        if np.std(dchi_abs) / (np.abs(dchi_mean) + EPS_STD) > 0.05:
            print(
                "Warning: redshift spacing is not uniform in comoving distance; using mean spacing."
            )
        return dchi_mean

    raise ValueError(
        f"Unsupported radial unit '{cfg.unit_f}'. Use 'mpc', 'mpc/h', 'kpc', 'kpc/h', "
        "'gpc', 'gpc/h', 'mhz', 'hz', or 'redshift'."
    )


def _compute_k_axes(shape: Tuple[int, int, int], dx: float, dy: float, df: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nf, nx, ny = shape
    kf = 2 * math.pi * np.fft.fftfreq(nf, d=df)
    kx = 2 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * math.pi * np.fft.fftfreq(ny, d=dy)
    return kf, kx, ky


def _apply_uv_mask(power: np.ndarray, config: PowerSpecConfig) -> np.ndarray:
    """
    Placeholder for applying a uv / PSF / visibility mask in (kx, ky).

    Currently this is a no-op and simply returns the input power cube.
    In future, when working with interferometric visibilities or an
    explicit PSF model, this hook can be used to zero out unsampled
    transverse modes before cylindrical averaging.
    """
    # Example future extension:
    # - Add uvmin/uvmax fields to PowerSpecConfig (in meters or wavelengths)
    # - Convert them to k_perp ranges using the cosmology
    # - Construct a boolean mask in (kx, ky) and apply it here.
    return power


def compute_power_spectra(
    cube: np.ndarray,
    config: PowerSpecConfig,
    window: str = "hann",
) -> Dict[str, np.ndarray]:
    """
    Compute 1D (spherical) and 2D (kperp, kpar) power spectra for a 3D cube.

    For the 1D spectrum, only Fourier modes inside the inscribed k-space
    sphere (i.e., where all three dimensions have support) are used when
    forming averages. For the 2D spectrum, the full line-of-sight range
    is retained while transverse modes are restricted to the inscribed
    circle in (kx, ky). By default, 2D k-bins are logarithmically spaced
    in both k_perp and k_par.
    """
    use_torch = _HAS_TORCH and isinstance(cube, torch.Tensor)  # type: ignore[name-defined]

    if use_torch:
        # Torch/GPU path: keep FFT on the tensor device, then move power to CPU.
        tensor = cube  # type: ignore[assignment]
        cube_reorder_t = tensor.movedim(config.freq_axis, 0)
        nf, nx, ny = cube_reorder_t.shape
        cube_demean_t = cube_reorder_t - cube_reorder_t.mean()

        if window:
            win_f = torch.hann_window(nf, periodic=True, device=tensor.device, dtype=tensor.dtype)
            win_x = torch.hann_window(nx, periodic=True, device=tensor.device, dtype=tensor.dtype)
            win_y = torch.hann_window(ny, periodic=True, device=tensor.device, dtype=tensor.dtype)
            win3d = win_f[:, None, None] * win_x[None, :, None] * win_y[None, None, :]
            cube_demean_t = cube_demean_t * win3d
            norm = float((win3d**2).mean().item())
        else:
            norm = 1.0

        df_mpc = _frequency_spacing_to_mpc(config, nf)
        dx_mpc = _resolve_spatial_spacing(config.dx, config.unit_x, config)
        dy_mpc = _resolve_spatial_spacing(config.dy, config.unit_y, config)
        v_cell = df_mpc * dx_mpc * dy_mpc
        n_vox = float(nf * nx * ny)

        Fk_t = torch.fft.fftn(cube_demean_t)
        power = (v_cell / (norm * n_vox)) * (Fk_t.abs() ** 2)
        power = power.detach().cpu().numpy()
        cube_reorder = cube_reorder_t.detach().cpu().numpy()
    else:
        # NumPy/CPU path.
        if hasattr(cube, "detach"):
            cube = cube.detach().cpu().numpy()  # type: ignore[assignment]
        if not isinstance(cube, np.ndarray):
            cube = np.asarray(cube)
        cube_reorder = np.moveaxis(cube, config.freq_axis, 0)
        nf, nx, ny = cube_reorder.shape
        cube_demean = cube_reorder - cube_reorder.mean()

        if window:
            win_f = np.hanning(nf) if window == "hann" else np.ones(nf)
            win_x = np.hanning(nx) if window == "hann" else np.ones(nx)
            win_y = np.hanning(ny) if window == "hann" else np.ones(ny)
            win3d = win_f[:, None, None] * win_x[None, :, None] * win_y[None, None, :]
            cube_demean = cube_demean * win3d
            norm = np.mean(win3d**2)
        else:
            norm = 1.0

        df_mpc = _frequency_spacing_to_mpc(config, nf)
        dx_mpc = _resolve_spatial_spacing(config.dx, config.unit_x, config)
        dy_mpc = _resolve_spatial_spacing(config.dy, config.unit_y, config)
        v_cell = df_mpc * dx_mpc * dy_mpc
        n_vox = float(nf * nx * ny)

        Fk = np.fft.fftn(cube_demean)
        power = (v_cell / (norm * n_vox)) * (np.abs(Fk) ** 2)
    power = _apply_uv_mask(power, config)
    kf, kx, ky = _compute_k_axes((nf, nx, ny), df_mpc, dx_mpc, dy_mpc)
    kf_grid, kx_grid, ky_grid = np.meshgrid(kf, kx, ky, indexing="ij")
    kperp = np.sqrt(kx_grid**2 + ky_grid**2)
    kmag = np.sqrt(kperp**2 + kf_grid**2)

    # Restrict 1D averages to the inscribed k-space sphere where all
    # three dimensions have support.
    max_kf = float(np.max(np.abs(kf)))
    max_kx = float(np.max(np.abs(kx)))
    max_ky = float(np.max(np.abs(ky)))
    kmax_sphere = min(max_kf, max_kx, max_ky)
    sphere_mask = kmag <= kmax_sphere

    power_1d = power[sphere_mask].reshape(-1)
    kmag_flat = kmag[sphere_mask].reshape(-1)

    # 1D spherical average (robust or mean) with linear k bins.
    k_bins = np.linspace(0.0, kmax_sphere, config.nbins_1d + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    bin_idx_1d = np.digitize(kmag_flat, k_bins) - 1
    p1d = np.zeros(config.nbins_1d, dtype=float)
    stat_mode = config.stat_mode.lower()
    if stat_mode not in {"median", "mean"}:
        raise ValueError("stat_mode must be 'median' or 'mean'.")
    for i in range(config.nbins_1d):
        mask = bin_idx_1d == i
        if not np.any(mask):
            continue
        vals = power_1d[mask]
        if stat_mode == "mean":
            p1d[i] = float(np.mean(vals))
        else:
            p1d[i] = float(np.median(vals))

    # 2D (kperp, kpar) average (robust or mean).
    # Relax the inscribed-sphere constraint: keep the full k_par range,
    # but restrict transverse modes to the inscribed circle in (kx, ky)
    # so that both spatial dimensions have support.
    kmax_perp_circle = min(max_kx, max_ky)
    circle_mask = kperp <= kmax_perp_circle
    power_2d = power[circle_mask].reshape(-1)
    kperp_flat = kperp[circle_mask].reshape(-1)
    kpar_flat = np.abs(kf_grid[circle_mask]).reshape(-1)

    # Define 2D k-bins: log or linear spacing.
    if config.log_bins_2d:
        valid_kperp = kperp_flat[kperp_flat > 0]
        valid_kpar = kpar_flat[kpar_flat > 0]
        if valid_kperp.size == 0 or valid_kpar.size == 0:
            raise ValueError("No positive k_perp or k_par values available for 2D binning.")
        kperp_min = float(valid_kperp.min())
        kperp_max = float(valid_kperp.max())
        kpar_min = float(valid_kpar.min())
        kpar_max = float(valid_kpar.max())
        kperp_bins = np.logspace(
            math.log10(kperp_min), math.log10(kperp_max), config.nbins_kperp + 1
        )
        kpar_bins = np.logspace(
            math.log10(kpar_min), math.log10(kpar_max), config.nbins_kpar + 1
        )
    else:
        kperp_bins = np.linspace(0.0, kmax_perp_circle, config.nbins_kperp + 1)
        kpar_bins = np.linspace(0.0, float(np.max(kpar_flat)), config.nbins_kpar + 1)

    bin_kperp = np.digitize(kperp_flat, kperp_bins) - 1
    bin_kpar = np.digitize(kpar_flat, kpar_bins) - 1
    # Ensure k_perp = 0 and k_par = 0 modes are included in the first bin
    # when using logarithmic k-bins. This prevents entire first rows/columns
    # from being empty purely due to the placement of the log-spaced edges.
    if config.log_bins_2d:
        bin_kperp[kperp_flat == 0.0] = 0
        bin_kpar[kpar_flat == 0.0] = 0

    p2d = np.zeros((config.nbins_kperp, config.nbins_kpar), dtype=float)
    for ix in range(config.nbins_kperp):
        for iy in range(config.nbins_kpar):
            mask = (bin_kperp == ix) & (bin_kpar == iy)
            if not np.any(mask):
                continue
            vals = power_2d[mask]
            if stat_mode == "mean":
                p2d[ix, iy] = float(np.mean(vals))
            else:
                p2d[ix, iy] = float(np.median(vals))

    return {
        "k_centers": k_centers,
        "p1d": p1d,
        "kperp_centers": 0.5 * (kperp_bins[:-1] + kperp_bins[1:]),
        "kpar_centers": 0.5 * (kpar_bins[:-1] + kpar_bins[1:]),
        "p2d": p2d,
    }


def _save_1d_fits(path: Path, k: np.ndarray, p: np.ndarray) -> None:
    cols = [fits.Column(name="k", format="E", array=k.astype(np.float32)),
            fits.Column(name="power", format="E", array=p.astype(np.float32))]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)


def _save_2d_fits(path: Path, power2d: np.ndarray) -> None:
    fits.PrimaryHDU(data=power2d.astype(np.float32)).writeto(path, overwrite=True)


def _plot_1d(
    path: Path,
    k: np.ndarray,
    rec: np.ndarray,
    true: Optional[np.ndarray],
    rel: Optional[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(k, rec, label="recovered", color="C0")
    if true is not None:
        ax1.plot(k, true, label="true", color="C1", linestyle="--")
    ax1.set_xlabel("k [1/Mpc]")
    ax1.set_ylabel("P(k)")
    has_positive = np.any(rec > 0) or (true is not None and np.any(true > 0))
    if has_positive:
        ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if rel is not None:
        rel_clip = np.clip(rel, 1.0, 100.0)
        ax2 = ax1.twinx()
        ax2.plot(k, rel_clip, color="C2", alpha=0.6, label="|rel| %")
        ax2.set_yscale("log")
        ax2.set_ylim(1.0, 100.0)
        ax2.set_ylabel("Relative % (abs, log)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_2d(
    path: Path,
    kperp: np.ndarray,
    kpar: np.ndarray,
    p2d: np.ndarray,
    title: str,
    cbar_label: str,
    log_scale: bool = False,
    log_axes: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    data = np.array(p2d, copy=True)
    if log_scale:
        # Map non-positive or empty bins to a small floor value so that
        # they appear as the darkest color rather than blank. This avoids
        # large NaN regions in the plot while preserving the dynamic range
        # of bins with actual measurements.
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = data[data > 0]
            if positive.size > 0:
                min_pos = float(np.nanmin(positive))
                floor = min_pos / 10.0
                safe = np.where(data > 0, data, floor)
                data = np.log10(safe)
            else:  # no positive values; fall back to zeros
                data = np.zeros_like(data)
    finite_vals = data[np.isfinite(data)]
    vmax = np.nanmax(finite_vals) if finite_vals.size > 0 else None
    vmin = np.nanmin(finite_vals) if finite_vals.size > 0 else None
    im = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=[kperp.min(), kperp.max(), kpar.min(), kpar.max()],
        vmin=vmin,
        vmax=vmax if vmax is not None and np.isfinite(vmax) else None,
    )
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("k_perp [1/Mpc]")
    ax.set_ylabel("k_par [1/Mpc]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_power_outputs(
    output_dir: Path,
    rec: Dict[str, np.ndarray],
    true: Optional[Dict[str, np.ndarray]] = None,
    log_power_2d: bool = True,
    log_axes_2d: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    k = rec["k_centers"]
    p1d_rec = rec["p1d"]
    p2d_rec = rec["p2d"]
    kperp = rec["kperp_centers"]
    kpar = rec["kpar_centers"]

    true_p1d = true["p1d"] if true is not None else None
    true_p2d = true["p2d"] if true is not None else None

    rel1d = None
    rel2d = None
    if true is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel1d = np.where(
                np.abs(true_p1d) > EPS_LOSS,
                100.0 * np.abs(p1d_rec - true_p1d) / np.abs(true_p1d),
                0.0,
            )
            rel2d = np.where(
                np.abs(true_p2d) > EPS_LOSS,
                100.0 * np.abs(p2d_rec - true_p2d) / np.abs(true_p2d),
                0.0,
            )

    _save_1d_fits(output_dir / "power1d_rec.fits", k, p1d_rec)
    _plot_1d(output_dir / "power1d.png", k, p1d_rec, true_p1d, rel1d)
    _save_2d_fits(output_dir / "power2d_rec.fits", p2d_rec)
    _plot_2d(
        output_dir / "power2d.png",
        kperp,
        kpar,
        p2d_rec,
        "Recovered 2D Power",
        cbar_label="log10 P(k)" if log_power_2d else "P(k)",
        log_scale=log_power_2d,
        log_axes=log_axes_2d,
    )

    if true is not None:
        _save_1d_fits(output_dir / "power1d_true.fits", k, true_p1d)
        _save_2d_fits(output_dir / "power2d_true.fits", true_p2d)
        _plot_2d(
            output_dir / "power2d_true.png",
            kperp,
            kpar,
            true_p2d,
            "True 2D Power",
            cbar_label="log10 P(k)" if log_power_2d else "P(k)",
            log_scale=log_power_2d,
            log_axes=log_axes_2d,
        )
        if rel1d is not None:
            rel1d_clip = np.clip(rel1d, 1.0, 100.0)
            _save_1d_fits(output_dir / "power1d_rel.fits", k, rel1d_clip)
        if rel2d is not None:
            rel2d_clip = np.clip(rel2d, 1.0, 100.0)
            _save_2d_fits(output_dir / "power2d_rel.fits", rel2d_clip)
            _plot_2d(
                output_dir / "power2d_rel.png",
                kperp,
                kpar,
                rel2d_clip,
                "Relative % 2D Power (clipped, abs)",
                cbar_label="log10 |rel| %",
                log_scale=True,
                log_axes=log_axes_2d,
            )
