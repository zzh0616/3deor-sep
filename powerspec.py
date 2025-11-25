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


@dataclass
class PowerSpecConfig:
    dx: float  # spacing along x
    dy: float  # spacing along y
    df: float  # spacing along frequency axis (Hz or MHz if unit_f is a frequency unit)
    unit_x: str = "mpc"  # e.g., "mpc" or "mpc/h"
    unit_y: str = "mpc"
    unit_f: str = "mpc"  # "mpc" if already physical, otherwise "mhz" or "hz"
    ref_freq_mhz: Optional[float] = None  # central/starting frequency (MHz) if unit_f is frequency
    rest_freq_mhz: float = 1420.40575  # HI 21cm default
    H0: float = 70.0
    Om0: float = 0.3
    Ode0: Optional[float] = None
    freq_axis: int = 0
    nbins_1d: int = 30
    nbins_kperp: int = 30
    nbins_kpar: int = 30
    output_dir: str = "powerspec"

def _cosmo_from_cfg(cfg: PowerSpecConfig) -> FlatLambdaCDM:
    ode0 = 1.0 - cfg.Om0 if cfg.Ode0 is None else cfg.Ode0
    return FlatLambdaCDM(H0=cfg.H0 * u.km / u.s / u.Mpc, Om0=cfg.Om0, Ode0=ode0)


def _resolve_spacing(value: float, unit: str) -> float:
    unit_low = unit.lower()
    if unit_low in {"mpc", "mpc/h"}:
        if unit_low == "mpc/h":
            return value / 1.0  # assume h=1.0 unless user encoded it in value
        return value
    raise ValueError(f"Unsupported spatial unit '{unit}'. Use 'mpc' or 'mpc/h'.")


def _frequency_spacing_to_mpc(cfg: PowerSpecConfig, nf: int) -> float:
    if cfg.unit_f.lower() in {"mpc", "mpc/h"}:
        return _resolve_spacing(cfg.df, cfg.unit_f)
    if cfg.unit_f.lower() not in {"mhz", "hz"}:
        raise ValueError(f"Unsupported frequency unit '{cfg.unit_f}'. Use 'MHz' or 'Hz'.")
    if cfg.ref_freq_mhz is None:
        raise ValueError("ref_freq_mhz is required when unit_f is a frequency unit.")

    cosmo = _cosmo_from_cfg(cfg)
    ref_freq = cfg.ref_freq_mhz * u.MHz
    df = cfg.df * (u.MHz if cfg.unit_f.lower() == "mhz" else u.Hz)
    freqs = ref_freq + np.arange(nf) * df
    z = (cfg.rest_freq_mhz * u.MHz / freqs - 1.0).decompose()
    chi = cosmo.comoving_distance(z).to(u.Mpc).value
    dchi = np.diff(chi)
    dchi_mean = float(np.mean(dchi))
    if np.std(dchi) / (np.abs(dchi_mean) + EPS_STD) > 0.05:
        print("Warning: frequency spacing is not uniform in comoving distance; using mean spacing.")
    return dchi_mean


def _compute_k_axes(shape: Tuple[int, int, int], dx: float, dy: float, df: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nf, nx, ny = shape
    kf = 2 * math.pi * np.fft.fftfreq(nf, d=df)
    kx = 2 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * math.pi * np.fft.fftfreq(ny, d=dy)
    return kf, kx, ky


def compute_power_spectra(
    cube: np.ndarray,
    config: PowerSpecConfig,
    window: str = "hann",
) -> Dict[str, np.ndarray]:
    """
    Compute 1D (spherical) and 2D (kperp, kpar) power spectra for a 3D cube.
    """
    if hasattr(cube, "detach"):
        cube = cube.detach().cpu().numpy()
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
    Fk = np.fft.fftn(cube_demean)
    volume = (df_mpc * dx_mpc * dy_mpc)
    power = np.abs(Fk) ** 2 / (norm * volume)

    df_mpc = _frequency_spacing_to_mpc(config, nf)
    dx_mpc = _resolve_spacing(config.dx, config.unit_x)
    dy_mpc = _resolve_spacing(config.dy, config.unit_y)
    kf, kx, ky = _compute_k_axes((nf, nx, ny), df_mpc, dx_mpc, dy_mpc)
    kf_grid, kx_grid, ky_grid = np.meshgrid(kf, kx, ky, indexing="ij")
    kperp = np.sqrt(kx_grid**2 + ky_grid**2)
    kmag = np.sqrt(kperp**2 + kf_grid**2)

    power_flat = power.flatten()
    kmag_flat = kmag.flatten()
    kperp_flat = kperp.flatten()
    kpar_flat = np.abs(kf_grid).flatten()

    # 1D spherical average
    k_bins = np.linspace(0, kmag_flat.max(), config.nbins_1d + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    counts_1d, _ = np.histogram(kmag_flat, bins=k_bins)
    weights_1d, _ = np.histogram(kmag_flat, bins=k_bins, weights=power_flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        p1d = np.where(counts_1d > 0, weights_1d / counts_1d, 0.0)

    # 2D (kperp, kpar) average
    kperp_bins = np.linspace(0, kperp_flat.max(), config.nbins_kperp + 1)
    kpar_bins = np.linspace(0, kpar_flat.max(), config.nbins_kpar + 1)
    counts_2d, _, _ = np.histogram2d(kperp_flat, kpar_flat, bins=(kperp_bins, kpar_bins))
    weights_2d, _, _ = np.histogram2d(kperp_flat, kpar_flat, bins=(kperp_bins, kpar_bins), weights=power_flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        p2d = np.where(counts_2d > 0, weights_2d / counts_2d, 0.0)

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


def _plot_1d(path: Path, k: np.ndarray, rec: np.ndarray, true: Optional[np.ndarray], rel: Optional[np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(k, rec, label="recovered", color="C0")
    if true is not None:
        ax1.plot(k, true, label="true", color="C1", linestyle="--")
    ax1.set_xlabel("k")
    ax1.set_ylabel("P(k)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if rel is not None:
        ax2 = ax1.twinx()
        ax2.plot(k, rel, color="C2", alpha=0.6, label="rel %")
        ax2.set_ylabel("Relative %")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_2d(path: Path, kperp: np.ndarray, kpar: np.ndarray, p2d: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    finite_vals = p2d[np.isfinite(p2d)]
    vmax = np.nanmax(finite_vals) if finite_vals.size > 0 else None
    vmin = 0.0
    im = ax.imshow(
        p2d.T,
        origin="lower",
        aspect="auto",
        extent=[kperp.min(), kperp.max(), kpar.min(), kpar.max()],
        vmin=vmin,
        vmax=vmax if vmax is not None and np.isfinite(vmax) else None,
    )
    ax.set_xlabel("k_perp")
    ax.set_ylabel("k_par")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="P(k)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_power_outputs(
    output_dir: Path,
    rec: Dict[str, np.ndarray],
    true: Optional[Dict[str, np.ndarray]] = None,
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
        rel1d = np.where(np.abs(true_p1d) > EPS_LOSS, 100.0 * (p1d_rec - true_p1d) / true_p1d, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel2d = np.where(np.abs(true_p2d) > EPS_LOSS, 100.0 * (p2d_rec - true_p2d) / true_p2d, 0.0)

    _save_1d_fits(output_dir / "power1d_rec.fits", k, p1d_rec)
    _plot_1d(output_dir / "power1d.png", k, p1d_rec, true_p1d, rel1d)
    _save_2d_fits(output_dir / "power2d_rec.fits", p2d_rec)
    _plot_2d(output_dir / "power2d.png", kperp, kpar, p2d_rec, "Recovered 2D Power")

    if true is not None:
        _save_1d_fits(output_dir / "power1d_true.fits", k, true_p1d)
        _save_2d_fits(output_dir / "power2d_true.fits", true_p2d)
        if rel1d is not None:
            _save_1d_fits(output_dir / "power1d_rel.fits", k, rel1d)
        if rel2d is not None:
            rel2d_clip = np.clip(rel2d, -500.0, 500.0)
            _save_2d_fits(output_dir / "power2d_rel.fits", rel2d_clip)
            _plot_2d(output_dir / "power2d_rel.png", kperp, kpar, rel2d_clip, "Relative % 2D Power (clipped)")
