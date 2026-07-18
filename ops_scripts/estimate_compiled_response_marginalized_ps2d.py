#!/usr/bin/env python3
"""Estimate dirty-EoR PS2D after marginalizing a compiled foreground response.

This is a response-calibrated quadratic estimator, not a map-recovery loss.  A
linear compiled Chebyshev response is treated as a foreground nuisance space.
Its coefficients are analytically projected out, and observation-independent
Gaussian band probes calibrate the resulting bandpower transfer/window matrix.

Injected foreground/EoR truth is optional and is used only to construct a
synthetic observation and report post-estimation simulation diagnostics.  It
is never used by the nuisance projection or transfer calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
for candidate in (
    CODE_DIR,
    CODE_DIR / "code" / "3dnet",
    Path.cwd() / "code" / "3dnet",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from powerspec import (  # noqa: E402
    PowerSpecConfig,
    _frequency_spacing_to_mpc,
    _resolve_spatial_spacing,
    compute_eor_window_mask,
    compute_eor_window_mode_mask_from_params,
    compute_power_spectra,
    select_eor_window_bins,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("candidate name must not be empty")
    return name.strip(), Path(raw_path)


def _format_pattern(pattern: str, freq: float) -> Path:
    freqtag = f"{float(freq):.2f}".replace(".", "")
    return Path(str(pattern).format(freq=float(freq), freqtag=freqtag))


def _central_crop(array: np.ndarray, size: int) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image, got {squeezed.shape}")
    height, width = squeezed.shape
    if size > height or size > width:
        raise ValueError(f"Cannot crop {size} from {squeezed.shape}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return np.asarray(squeezed[y0 : y0 + size, x0 : x0 + size], dtype=np.float64)


def _load_pattern_cube(pattern: str, freqs: Sequence[float], size: int) -> np.ndarray:
    planes = []
    for freq in freqs:
        path = _format_pattern(pattern, float(freq))
        if not path.is_file():
            raise FileNotFoundError(path)
        planes.append(_central_crop(fits.getdata(path), size))
    cube = np.stack(planes, axis=0)
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Non-finite values loaded from pattern {pattern}")
    return cube


def _load_cube(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    cube = np.squeeze(np.asarray(fits.getdata(path), dtype=np.float64))
    if cube.shape != shape:
        raise ValueError(f"Cube shape mismatch: {path}: {cube.shape} != {shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Cube contains non-finite values: {path}")
    return cube


def _rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    # bool subclasses int, so preserve JSON booleans before the integer case.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _svd_pinv(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=np.float64)
    left, singular, right_t = np.linalg.svd(values, full_matrices=False)
    threshold = (
        float(rcond) * float(singular[0]) if singular.size and singular[0] > 0.0 else 0.0
    )
    keep = singular > threshold
    inverse = np.zeros_like(singular)
    inverse[keep] = 1.0 / singular[keep]
    pinv = (right_t.T * inverse[None, :]) @ left.T
    positive = singular[singular > 0.0]
    condition = (
        float(positive[0] / positive[-1]) if positive.size else float("inf")
    )
    retained_condition = (
        float(singular[0] / singular[keep][-1]) if np.any(keep) else float("inf")
    )
    return pinv, {
        "shape": [int(value) for value in values.shape],
        "rcond": float(rcond),
        "threshold": float(threshold),
        "rank": int(np.sum(keep)),
        "condition_number": condition,
        "retained_condition_number": retained_condition,
        "singular_values": singular,
    }


@dataclass
class FourierGeometry:
    cube_shape: tuple[int, int, int]
    window: np.ndarray
    power_scale: float
    mode_indices: np.ndarray
    mode_active_bins: np.ndarray
    active_linear_bins: np.ndarray
    full_mode_linear_bins: np.ndarray
    counts: np.ndarray
    kperp_edges: np.ndarray
    kpar_edges: np.ndarray
    kperp_centers: np.ndarray
    kpar_centers: np.ndarray
    active_kperp_indices: np.ndarray
    active_kpar_indices: np.ndarray
    eor_window_active: np.ndarray
    eor_window_mode_fraction: np.ndarray
    demean_mode: str

    @property
    def band_count(self) -> int:
        return int(self.active_linear_bins.size)


def _build_fourier_geometry(
    shape: tuple[int, int, int],
    cfg: PowerSpecConfig,
) -> FourierGeometry:
    nf, ny, nx = shape
    if int(cfg.freq_axis) != 0:
        raise ValueError("The compiled-response estimator requires freq_axis=0")
    df_mpc = _frequency_spacing_to_mpc(cfg, nf)
    dy_mpc = _resolve_spatial_spacing(cfg.dy, cfg.unit_y, cfg)
    dx_mpc = _resolve_spatial_spacing(cfg.dx, cfg.unit_x, cfg)
    kf = 2.0 * math.pi * np.fft.fftfreq(nf, d=df_mpc)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy_mpc)
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx_mpc)
    kf_grid, ky_grid, kx_grid = np.meshgrid(kf, ky, kx, indexing="ij")
    kperp = np.sqrt(ky_grid**2 + kx_grid**2)
    kpar = np.abs(kf_grid)
    kmax_perp = min(float(np.max(np.abs(ky))), float(np.max(np.abs(kx))))
    circle = kperp <= kmax_perp

    kperp_flat = kperp[circle]
    kpar_flat = kpar[circle]
    if bool(cfg.log_bins_2d):
        positive_perp = kperp_flat[kperp_flat > 0.0]
        positive_par = kpar_flat[kpar_flat > 0.0]
        if positive_perp.size == 0 or positive_par.size == 0:
            raise ValueError("No positive Fourier modes for logarithmic PS2D bins")
        kperp_edges = np.logspace(
            math.log10(float(np.min(positive_perp))),
            math.log10(float(np.max(positive_perp))),
            int(cfg.nbins_kperp) + 1,
        )
        kpar_edges = np.logspace(
            math.log10(float(np.min(positive_par))),
            math.log10(float(np.max(positive_par))),
            int(cfg.nbins_kpar) + 1,
        )
    else:
        kperp_edges = np.linspace(0.0, kmax_perp, int(cfg.nbins_kperp) + 1)
        kpar_edges = np.linspace(
            0.0, float(np.max(kpar_flat)), int(cfg.nbins_kpar) + 1
        )

    perp_bin = np.digitize(kperp_flat, kperp_edges) - 1
    par_bin = np.digitize(kpar_flat, kpar_edges) - 1
    if bool(cfg.log_bins_2d):
        perp_bin[kperp_flat == 0.0] = 0
        par_bin[kpar_flat == 0.0] = 0
    valid = (
        (perp_bin >= 0)
        & (perp_bin < int(cfg.nbins_kperp))
        & (par_bin >= 0)
        & (par_bin < int(cfg.nbins_kpar))
    )
    circle_indices = np.flatnonzero(circle.reshape(-1))
    mode_indices = circle_indices[valid]
    linear_bins = perp_bin[valid] * int(cfg.nbins_kpar) + par_bin[valid]
    full_bin_count = int(cfg.nbins_kperp) * int(cfg.nbins_kpar)
    full_counts = np.bincount(linear_bins, minlength=full_bin_count)
    active_linear = np.flatnonzero(full_counts > 0)
    active_lookup = np.full((full_bin_count,), -1, dtype=np.int64)
    active_lookup[active_linear] = np.arange(active_linear.size, dtype=np.int64)
    active_mode_bins = active_lookup[linear_bins]
    counts = full_counts[active_linear].astype(np.float64)
    full_mode_bins = np.full((nf * ny * nx,), -1, dtype=np.int64)
    full_mode_bins[mode_indices] = active_mode_bins
    active_perp = active_linear // int(cfg.nbins_kpar)
    active_par = active_linear % int(cfg.nbins_kpar)
    kperp_centers = 0.5 * (kperp_edges[:-1] + kperp_edges[1:])
    kpar_centers = 0.5 * (kpar_edges[:-1] + kpar_edges[1:])
    eor_grid = compute_eor_window_mask(kperp_centers, kpar_centers, cfg)
    eor_center_active = eor_grid[active_perp, active_par]
    mode_window = compute_eor_window_mode_mask_from_params(
        kperp_flat[valid],
        kpar_flat[valid],
        kpar_min=float(cfg.eor_window_kpar_min),
        wedge_slope=float(cfg.eor_window_wedge_slope),
        wedge_intercept=float(cfg.eor_window_wedge_intercept),
        kperp_min=cfg.eor_window_kperp_min,
        kperp_max=cfg.eor_window_kperp_max,
        kpar_max=cfg.eor_window_kpar_max,
        exclude_dc=bool(cfg.eor_window_exclude_dc),
    )
    selected_counts = np.bincount(
        linear_bins[mode_window], minlength=full_bin_count
    ).astype(np.float64, copy=False)
    full_mode_fraction = np.zeros((full_bin_count,), dtype=np.float64)
    populated = full_counts > 0
    full_mode_fraction[populated] = (
        selected_counts[populated] / full_counts[populated]
    )
    eor_mode_fraction = full_mode_fraction[active_linear]
    eor_active = select_eor_window_bins(
        eor_center_active,
        eor_mode_fraction,
        cfg.eor_window_bin_policy,
    )

    window = (
        np.hanning(nf)[:, None, None]
        * np.hanning(ny)[None, :, None]
        * np.hanning(nx)[None, None, :]
    )
    window_energy = float(np.mean(window * window))
    if window_energy <= 0.0:
        raise ValueError("Hann window has zero energy")
    voxel_volume = float(df_mpc * dy_mpc * dx_mpc)
    power_scale = voxel_volume / (window_energy * float(nf * ny * nx))
    return FourierGeometry(
        cube_shape=shape,
        window=np.asarray(window, dtype=np.float64),
        power_scale=power_scale,
        mode_indices=np.asarray(mode_indices, dtype=np.int64),
        mode_active_bins=np.asarray(active_mode_bins, dtype=np.int64),
        active_linear_bins=np.asarray(active_linear, dtype=np.int64),
        full_mode_linear_bins=full_mode_bins.reshape(shape),
        counts=counts,
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
        kperp_centers=kperp_centers,
        kpar_centers=kpar_centers,
        active_kperp_indices=active_perp,
        active_kpar_indices=active_par,
        eor_window_active=np.asarray(eor_active, dtype=bool),
        eor_window_mode_fraction=np.asarray(eor_mode_fraction, dtype=np.float64),
        demean_mode=str(cfg.demean_mode),
    )


class BandpowerTransform:
    def __init__(self, geometry: FourierGeometry, device: torch.device):
        self.geometry = geometry
        self.device = device
        self.window = torch.as_tensor(
            geometry.window, dtype=torch.float64, device=device
        )
        self.mode_indices = torch.as_tensor(
            geometry.mode_indices, dtype=torch.int64, device=device
        )
        self.mode_bins = torch.as_tensor(
            geometry.mode_active_bins, dtype=torch.int64, device=device
        )
        self.counts = torch.as_tensor(
            geometry.counts, dtype=torch.float64, device=device
        )

    def _demean(self, cubes: torch.Tensor) -> torch.Tensor:
        mode = self.geometry.demean_mode.strip().lower()
        if mode in {"global", "global_demean"}:
            return cubes - cubes.mean(dim=(1, 2, 3), keepdim=True)
        if mode in {"per_freq_spatial", "per_freq_demean"}:
            return cubes - cubes.mean(dim=(2, 3), keepdim=True)
        if mode == "none":
            return cubes
        raise ValueError(f"Unsupported demean_mode: {self.geometry.demean_mode}")

    def fourier(self, cubes: torch.Tensor) -> torch.Tensor:
        if cubes.ndim == 3:
            cubes = cubes.unsqueeze(0)
        if tuple(cubes.shape[1:]) != self.geometry.cube_shape:
            raise ValueError(
                f"Bandpower cube shape mismatch: {tuple(cubes.shape[1:])} "
                f"!= {self.geometry.cube_shape}"
            )
        tapered = self._demean(cubes) * self.window.unsqueeze(0)
        return torch.fft.fftn(tapered, dim=(-3, -2, -1))

    def __call__(self, cubes: torch.Tensor) -> torch.Tensor:
        spectrum = self.fourier(cubes)
        power = spectrum.real.square() + spectrum.imag.square()
        selected = power.reshape(power.shape[0], -1).index_select(1, self.mode_indices)
        out = torch.zeros(
            (power.shape[0], self.geometry.band_count),
            dtype=torch.float64,
            device=self.device,
        )
        out.scatter_add_(1, self.mode_bins.unsqueeze(0).expand(power.shape[0], -1), selected)
        out = out / self.counts.unsqueeze(0)
        return out * float(self.geometry.power_scale)


def _gram_inverse(
    gram: torch.Tensor,
    *,
    rcond: float,
    ridge_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    maximum = float(torch.max(eigenvalues).item()) if eigenvalues.numel() else 0.0
    threshold = float(rcond) * maximum
    keep = eigenvalues > threshold
    if not bool(torch.any(keep)):
        raise ValueError("Compiled nuisance response has zero numerical rank")
    retained_values = eigenvalues[keep]
    retained_vectors = eigenvectors[:, keep]
    ridge = float(ridge_fraction) * maximum
    inverse = (
        retained_vectors
        @ torch.diag(torch.reciprocal(retained_values + ridge))
        @ retained_vectors.T
    )
    return inverse, {
        "rcond": float(rcond),
        "threshold": threshold,
        "rank": int(torch.sum(keep).item()),
        "ridge_fraction": float(ridge_fraction),
        "ridge_absolute": ridge,
        "condition_number_retained": float(
            (retained_values[-1] + ridge).item()
            / (retained_values[0] + ridge).item()
        ),
        "eigenvalue_min_retained": float(retained_values[0].item()),
        "eigenvalue_max": maximum,
    }


def _compress_design_svd(
    design: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_count = int(design.shape[0])
    requested = int(rank)
    if requested <= 0 or requested >= source_count:
        return design, {
            "enabled": False,
            "source_parameter_count": source_count,
            "retained_parameter_count": source_count,
            "response_power_fraction": 1.0,
        }
    flat = design.reshape(source_count, -1)
    gram = flat @ flat.T
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    order = torch.argsort(eigenvalues, descending=True)
    retained_indices = order[:requested]
    retained_vectors = eigenvectors[:, retained_indices]
    compressed = (retained_vectors.T @ flat).reshape(
        requested, *design.shape[1:]
    )
    nonnegative = torch.clamp(eigenvalues, min=0.0)
    total = float(torch.sum(nonnegative).item())
    retained = float(torch.sum(nonnegative[retained_indices]).item())
    return compressed, {
        "enabled": True,
        "source_parameter_count": source_count,
        "retained_parameter_count": requested,
        "response_power_fraction": retained / max(total, 1e-300),
        "source_gram_condition_number": float(
            nonnegative[order[0]].item()
            / max(float(nonnegative[order[-1]].item()), 1e-300)
        ),
        "retained_eigenvalue_min": float(nonnegative[retained_indices[-1]].item()),
        "retained_eigenvalue_max": float(nonnegative[retained_indices[0]].item()),
    }


class NuisanceProjector:
    def __init__(self, design: torch.Tensor, rcond: float, ridge_fraction: float):
        if design.ndim != 4:
            raise ValueError("Compiled response must have shape [param,freq,y,x]")
        self.design = design.reshape(design.shape[0], -1)
        gram = self.design @ self.design.T
        self.gram_pinv, inverse_stats = _gram_inverse(
            gram, rcond=rcond, ridge_fraction=ridge_fraction
        )
        self.stats = {
            "fit_scope": "image_all",
            "parameter_count": int(design.shape[0]),
            "voxel_count": int(self.design.shape[1]),
            **inverse_stats,
        }

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = cubes.ndim == 3
        if squeeze:
            cubes = cubes.unsqueeze(0)
        flat = cubes.reshape(cubes.shape[0], -1)
        rhs = flat @ self.design.T
        params = rhs @ self.gram_pinv
        residual = flat - params @ self.design
        result = residual.reshape_as(cubes)
        return (result[0] if squeeze else result), (params[0] if squeeze else params)


class IdentityNuisanceProjector:
    def __init__(self, design: torch.Tensor):
        self.stats = {
            "fit_scope": "none",
            "parameter_count": 0,
            "source_parameter_count": int(design.shape[0]),
            "voxel_count": int(np.prod(design.shape[1:])),
            "rank": 0,
            "ridge_fraction": 0.0,
        }

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = cubes.ndim == 3
        batch_count = 1 if squeeze else int(cubes.shape[0])
        params = torch.zeros(
            (batch_count, 0), dtype=cubes.dtype, device=cubes.device
        )
        return cubes, (params[0] if squeeze else params)


class FourierWedgeNuisanceProjector:
    def __init__(
        self,
        design: torch.Tensor,
        transform: BandpowerTransform,
        rcond: float,
        ridge_fraction: float,
    ):
        if design.ndim != 4:
            raise ValueError("Compiled response must have shape [param,freq,y,x]")
        self.design = design.reshape(design.shape[0], -1)
        geometry = transform.geometry
        wedge_modes = ~geometry.eor_window_active[geometry.mode_active_bins]
        fit_indices_np = geometry.mode_indices[wedge_modes]
        if fit_indices_np.size == 0:
            raise ValueError("The configured PS2D geometry contains no wedge fit modes")
        self.fit_indices = torch.as_tensor(
            fit_indices_np, dtype=torch.int64, device=design.device
        )
        design_spectrum = transform.fourier(design).reshape(design.shape[0], -1)
        self.fit_design = design_spectrum.index_select(1, self.fit_indices)
        gram = (
            self.fit_design.real @ self.fit_design.real.T
            + self.fit_design.imag @ self.fit_design.imag.T
        )
        self.gram_pinv, inverse_stats = _gram_inverse(
            gram, rcond=rcond, ridge_fraction=ridge_fraction
        )
        self.transform = transform
        self.stats = {
            "fit_scope": "fourier_wedge",
            "parameter_count": int(design.shape[0]),
            "voxel_count": int(self.design.shape[1]),
            "fit_fourier_mode_count": int(fit_indices_np.size),
            "fit_active_band_count": int(
                np.sum(~geometry.eor_window_active)
            ),
            "excluded_eor_window_band_count": int(
                np.sum(geometry.eor_window_active)
            ),
            **inverse_stats,
        }

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = cubes.ndim == 3
        if squeeze:
            cubes = cubes.unsqueeze(0)
        spectrum = self.transform.fourier(cubes).reshape(cubes.shape[0], -1)
        fit_data = spectrum.index_select(1, self.fit_indices)
        rhs = (
            fit_data.real @ self.fit_design.real.T
            + fit_data.imag @ self.fit_design.imag.T
        )
        params = rhs @ self.gram_pinv
        flat = cubes.reshape(cubes.shape[0], -1)
        residual = flat - params @ self.design
        result = residual.reshape_as(cubes)
        return (result[0] if squeeze else result), (params[0] if squeeze else params)


def _calibrate_transfer(
    *,
    geometry: FourierGeometry,
    bandpowers: BandpowerTransform,
    projector: NuisanceProjector
    | FourierWedgeNuisanceProjector
    | IdentityNuisanceProjector,
    probes_per_band: int,
    batch_size: int,
    seed: int,
    transfer_rcond: float,
    probe_fourier_psd: np.ndarray | None,
    dirty_probe_bank: np.ndarray | None,
) -> dict[str, Any]:
    device = bandpowers.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    band_count = geometry.band_count
    input_response = np.zeros((band_count, band_count), dtype=np.float64)
    projected_response = np.zeros_like(input_response)
    projected_square_sum = np.zeros_like(input_response)
    full_bins = torch.as_tensor(
        geometry.full_mode_linear_bins, dtype=torch.int64, device=device
    )
    bank_tensor = None
    if dirty_probe_bank is not None:
        bank = np.asarray(dirty_probe_bank, dtype=np.float64)
        if bank.ndim != 4 or tuple(bank.shape[1:]) != geometry.cube_shape:
            raise ValueError(
                "Dirty operator-probe bank must have shape "
                f"[probe,{','.join(str(value) for value in geometry.cube_shape)}]"
            )
        if int(probes_per_band) > int(bank.shape[0]):
            raise ValueError(
                f"Requested {probes_per_band} probes per band but bank contains "
                f"only {bank.shape[0]}"
            )
        if not np.all(np.isfinite(bank)):
            raise ValueError("Dirty operator-probe bank contains non-finite values")
        bank_tensor = torch.as_tensor(
            bank[: int(probes_per_band)], dtype=torch.float64, device=device
        )
        probe_sqrt_psd = None
        probe_covariance_source = "operator_propagated_dirty_probe_bank"
    elif probe_fourier_psd is None:
        probe_sqrt_psd = None
        probe_covariance_source = "flat_dirty_fourier_power"
    else:
        psd = np.asarray(probe_fourier_psd, dtype=np.float64)
        if psd.shape != geometry.cube_shape:
            raise ValueError(
                f"Operator-probe PSD shape mismatch: {psd.shape} != {geometry.cube_shape}"
            )
        if not np.all(np.isfinite(psd)) or np.any(psd < 0.0):
            raise ValueError("Operator-probe PSD must be finite and non-negative")
        if not np.any(psd > 0.0):
            raise ValueError("Operator-probe PSD contains no positive power")
        probe_sqrt_psd = torch.as_tensor(
            np.sqrt(psd), dtype=torch.float64, device=device
        )
        probe_covariance_source = "operator_propagated_white_gaussian_sky_psd"
    completed = 0
    for source_band in range(band_count):
        input_sum = torch.zeros((band_count,), dtype=torch.float64, device=device)
        projected_sum = torch.zeros_like(input_sum)
        projected_sq = torch.zeros_like(input_sum)
        source_mask = full_bins == int(source_band)
        for first in range(0, int(probes_per_band), int(batch_size)):
            current_batch = min(int(batch_size), int(probes_per_band) - first)
            if bank_tensor is None:
                white = torch.randn(
                    (current_batch, *geometry.cube_shape),
                    dtype=torch.float64,
                    device=device,
                    generator=generator,
                )
                spectrum = torch.fft.fftn(
                    white, dim=(-3, -2, -1), norm="ortho"
                )
                if probe_sqrt_psd is not None:
                    spectrum = spectrum * probe_sqrt_psd.unsqueeze(0)
            else:
                spectrum = torch.fft.fftn(
                    bank_tensor[first : first + current_batch],
                    dim=(-3, -2, -1),
                    norm="ortho",
                )
            probes = torch.fft.ifftn(
                spectrum * source_mask.unsqueeze(0),
                dim=(-3, -2, -1),
                norm="ortho",
            ).real
            before = bandpowers(probes)
            after_cubes, _params = projector.project(probes)
            after = bandpowers(after_cubes)
            input_sum += torch.sum(before, dim=0)
            projected_sum += torch.sum(after, dim=0)
            projected_sq += torch.sum(after * after, dim=0)
            completed += current_batch
        input_response[:, source_band] = (
            input_sum / float(probes_per_band)
        ).detach().cpu().numpy()
        projected_response[:, source_band] = (
            projected_sum / float(probes_per_band)
        ).detach().cpu().numpy()
        projected_square_sum[:, source_band] = (
            projected_sq / float(probes_per_band)
        ).detach().cpu().numpy()

    input_pinv, input_svd = _svd_pinv(input_response, transfer_rcond)
    transfer = projected_response @ input_pinv
    transfer_pinv, transfer_svd = _svd_pinv(transfer, transfer_rcond)
    row_sums = np.sum(transfer, axis=1)
    row_scale = np.zeros_like(row_sums)
    safe = np.abs(row_sums) > max(
        float(np.max(np.abs(row_sums))) * float(transfer_rcond), 1e-300
    )
    row_scale[safe] = 1.0 / row_sums[safe]
    row_normalizer = np.diag(row_scale)
    window_matrix = row_normalizer @ transfer
    variance = np.maximum(
        projected_square_sum - projected_response * projected_response, 0.0
    )
    projected_standard_error = np.sqrt(variance / float(probes_per_band))
    return {
        "input_response": input_response,
        "projected_response": projected_response,
        "projected_response_standard_error": projected_standard_error,
        "transfer_matrix": transfer,
        "deconvolution_matrix": transfer_pinv,
        "row_normalization_matrix": row_normalizer,
        "window_matrix": window_matrix,
        "input_response_svd": input_svd,
        "transfer_svd": transfer_svd,
        "row_sums": row_sums,
        "row_normalization_valid": safe,
        "probe_count_total": int(completed),
        "probes_per_band": int(probes_per_band),
        "probe_batch_size": int(batch_size),
        "seed": int(seed),
        "probe_covariance_source": probe_covariance_source,
    }


def _estimate_from_projected(
    projected_bandpower: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(projected_bandpower, dtype=np.float64)
    row_normalized = np.asarray(calibration["row_normalization_matrix"]) @ values
    deconvolved = np.asarray(calibration["deconvolution_matrix"]) @ values
    return row_normalized, deconvolved


def _bandpower_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(estimate)
        & np.isfinite(target)
        & (target > 0.0)
    )
    if not np.any(valid):
        return {"n_bins": 0}
    est = estimate[valid]
    truth = target[valid]
    denominator = max(float(np.linalg.norm(truth)), 1e-300)
    centered_est = est - np.mean(est)
    centered_truth = truth - np.mean(truth)
    corr_denominator = float(
        np.linalg.norm(centered_est) * np.linalg.norm(centered_truth)
    )
    positive = est > 0.0
    result = {
        "n_bins": int(est.size),
        "relative_l2_error": float(np.linalg.norm(est - truth) / denominator),
        "power_sum_ratio": float(np.sum(est) / max(float(np.sum(truth)), 1e-300)),
        "median_signed_fractional_error": float(np.median((est - truth) / truth)),
        "positive_fraction": float(np.mean(positive)),
        "correlation": (
            float(np.dot(centered_est, centered_truth) / corr_denominator)
            if corr_denominator > 0.0
            else float("nan")
        ),
    }
    if np.any(positive):
        log_ratio = np.log10(est[positive] / truth[positive])
        result.update(
            {
                "positive_log10_bias": float(np.mean(log_ratio)),
                "positive_log10_mad": float(np.median(np.abs(log_ratio))),
                "positive_log10_rmse": float(np.sqrt(np.mean(log_ratio**2))),
            }
        )
    return result


def _foreground_leakage_metrics(
    foreground_power: np.ndarray,
    eor_power: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    foreground = np.asarray(foreground_power, dtype=np.float64)
    eor = np.asarray(eor_power, dtype=np.float64)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(foreground)
        & np.isfinite(eor)
        & (foreground >= 0.0)
        & (eor > 0.0)
    )
    if not np.any(valid):
        return {"n_bins": 0}
    ratio = foreground[valid] / eor[valid]
    eor_valid = eor[valid]
    total_eor = max(float(np.sum(eor_valid)), 1e-300)
    result: dict[str, Any] = {
        "n_bins": int(ratio.size),
        "ratio_median": float(np.median(ratio)),
        "ratio_p90": float(np.percentile(ratio, 90.0)),
        "ratio_max": float(np.max(ratio)),
    }
    for threshold in (0.1, 1.0):
        below = ratio <= float(threshold)
        label = "0p1" if threshold == 0.1 else "1"
        result[f"bin_fraction_le_{label}"] = float(np.mean(below))
        result[f"eor_power_fraction_in_bins_le_{label}"] = float(
            np.sum(eor_valid[below]) / total_eor
        )
    return result


def _cube_bandpowers(
    cube: np.ndarray,
    transform: BandpowerTransform,
) -> np.ndarray:
    tensor = torch.as_tensor(cube, dtype=torch.float64, device=transform.device)
    return transform(tensor).detach().cpu().numpy()[0]


def _self_test() -> None:
    rng = np.random.default_rng(20260711)
    shape = (6, 8, 8)
    design_np = rng.normal(size=(3, *shape))
    design = torch.as_tensor(design_np, dtype=torch.float64)
    projector = NuisanceProjector(design, rcond=1e-12, ridge_fraction=0.0)
    foreground_params = torch.as_tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    foreground = torch.einsum("p,pfij->fij", foreground_params, design)
    projected, fitted = projector.project(foreground)
    np.testing.assert_allclose(projected.numpy(), 0.0, atol=2e-12)
    np.testing.assert_allclose(fitted.numpy(), foreground_params.numpy(), atol=2e-12)

    cfg = PowerSpecConfig(
        dx=1.0,
        dy=1.0,
        df=1.0,
        unit_x="mpc",
        unit_y="mpc",
        unit_f="mpc",
        freq_axis=0,
        nbins_1d=4,
        nbins_kperp=2,
        nbins_kpar=2,
        stat_mode="mean",
        log_bins_2d=False,
        demean_mode="global",
        eor_window_enabled=True,
        eor_window_kpar_min=2.0,
    )
    geometry = _build_fourier_geometry(shape, cfg)
    transform = BandpowerTransform(geometry, torch.device("cpu"))
    wedge_projector = FourierWedgeNuisanceProjector(
        design,
        transform,
        rcond=1e-12,
        ridge_fraction=0.0,
    )
    wedge_projected, wedge_fitted = wedge_projector.project(foreground)
    np.testing.assert_allclose(wedge_projected.numpy(), 0.0, atol=2e-11)
    np.testing.assert_allclose(
        wedge_fitted.numpy(), foreground_params.numpy(), atol=2e-11
    )
    parity_cube = rng.normal(size=shape)
    direct = transform(torch.as_tensor(parity_cube, dtype=torch.float64)).numpy()[0]
    legacy = compute_power_spectra(parity_cube, cfg, window="hann")["p2d"].reshape(-1)
    np.testing.assert_allclose(
        direct,
        legacy[geometry.active_linear_bins],
        rtol=2e-6,
        atol=max(float(np.max(np.abs(direct))) * 1e-12, 1e-12),
    )
    calibration = _calibrate_transfer(
        geometry=geometry,
        bandpowers=transform,
        projector=projector,
        probes_per_band=8,
        batch_size=4,
        seed=20260711,
        transfer_rcond=1e-8,
        probe_fourier_psd=None,
        dirty_probe_bank=None,
    )
    window = np.asarray(calibration["window_matrix"])
    valid = np.asarray(calibration["row_normalization_valid"], dtype=bool)
    np.testing.assert_allclose(np.sum(window[valid], axis=1), 1.0, atol=1e-10)
    if not np.all(np.isfinite(np.asarray(calibration["transfer_matrix"]))):
        raise AssertionError("Synthetic transfer matrix is not finite")
    print(
        json.dumps(
            {
                "event": "self_test_pass",
                "band_count": geometry.band_count,
                "nuisance_rank": projector.stats["rank"],
                "transfer_rank": calibration["transfer_svd"]["rank"],
                "time_utc": _now(),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-npz", type=Path)
    parser.add_argument("--extra-design-npz", action="append", type=Path, default=[])
    parser.add_argument("--candidate", action="append", type=_parse_named_path)
    parser.add_argument("--observed-cube-fits", type=Path)
    parser.add_argument("--truth-fg-pattern")
    parser.add_argument("--truth-eor-pattern")
    parser.add_argument("--freqs-mhz", default="")
    parser.add_argument("--eval-crop-size", type=int, default=256)
    parser.add_argument(
        "--power-config", type=Path, default=CODE_DIR / "configs/power_eor_window.json"
    )
    parser.add_argument("--nbins-kperp", type=int, default=4)
    parser.add_argument("--nbins-kpar", type=int, default=4)
    parser.add_argument("--nuisance-rcond", type=float, default=1e-12)
    parser.add_argument(
        "--nuisance-design-rank",
        type=int,
        default=0,
        help="Top compiled-response SVD modes retained; 0 keeps every mode.",
    )
    parser.add_argument(
        "--nuisance-fit-scope",
        choices=("none", "image_all", "fourier_wedge"),
        default="image_all",
    )
    parser.add_argument("--nuisance-ridge-fraction", type=float, default=0.0)
    parser.add_argument("--transfer-rcond", type=float, default=1e-8)
    parser.add_argument("--transfer-probes-per-band", type=int, default=32)
    parser.add_argument("--transfer-probe-batch-size", type=int, default=4)
    parser.add_argument("--transfer-seed", type=int, default=20260711)
    parser.add_argument(
        "--operator-probe-psd-npz",
        type=Path,
        help=(
            "Optional observation-independent operator-propagated sky-probe PSD. "
            "When present, transfer probes use this dirty-domain covariance shape."
        ),
    )
    parser.add_argument(
        "--operator-probe-bank-npz",
        type=Path,
        help=(
            "Optional bank of individual dirty responses from independent white-sky "
            "probes propagated through the complete operator. It takes precedence "
            "over the diagonal probe PSD for transfer calibration."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    required = {
        "design_npz": args.design_npz,
        "candidate": args.candidate,
        "freqs_mhz": args.freqs_mhz,
        "out_json": args.out_json,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    if args.observed_cube_fits is None and args.truth_fg_pattern is None:
        parser.error("Provide --observed-cube-fits or --truth-fg-pattern")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if int(args.nbins_kperp) < 1 or int(args.nbins_kpar) < 1:
        raise ValueError("PS2D bin counts must be positive")
    if int(args.transfer_probes_per_band) < 2:
        raise ValueError("At least two transfer probes per band are required")
    if int(args.transfer_probe_batch_size) < 1:
        raise ValueError("Transfer probe batch size must be positive")
    if not 0.0 < float(args.nuisance_rcond) < 1.0:
        raise ValueError("--nuisance-rcond must be in (0, 1)")
    if float(args.nuisance_ridge_fraction) < 0.0:
        raise ValueError("--nuisance-ridge-fraction must be non-negative")
    if int(args.nuisance_design_rank) < 0:
        raise ValueError("--nuisance-design-rank must be non-negative")
    if not 0.0 < float(args.transfer_rcond) < 1.0:
        raise ValueError("--transfer-rcond must be in (0, 1)")
    freqs = np.asarray(_parse_floats(args.freqs_mhz), dtype=np.float64)
    if freqs.size < 3:
        raise ValueError("At least three frequencies are required")
    device = torch.device(str(args.device))
    started = time.monotonic()

    design_paths = [Path(args.design_npz), *[Path(path) for path in args.extra_design_npz]]
    design_arrays = []
    metadata_records = []
    identities = []
    for path in design_paths:
        with np.load(path, allow_pickle=False) as payload:
            design_array = np.asarray(payload["design"], dtype=np.float64)
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        identity = metadata.get("identity")
        if not isinstance(identity, dict) or int(identity.get("format_version", -1)) != 3:
            raise ValueError("Every compiled response must use canonical v3 identity")
        if design_array.ndim != 4 or not np.all(np.isfinite(design_array)):
            raise ValueError("Compiled response must be finite [param,freq,y,x]")
        if design_arrays and design_array.shape[1:] != design_arrays[0].shape[1:]:
            raise ValueError("Additional compiled response dimensions do not match")
        if identities and identity.get("operator_identity") != identities[0].get(
            "operator_identity"
        ):
            raise ValueError("Additional compiled response operator identity mismatch")
        design_arrays.append(design_array)
        metadata_records.append(metadata)
        identities.append(identity)
    design_np = np.concatenate(design_arrays, axis=0)
    parameter_count, frequency_count, height, width = design_np.shape
    shape = (frequency_count, height, width)
    if frequency_count != freqs.size or height != width or height != int(args.eval_crop_size):
        raise ValueError("Design dimensions do not match requested frequencies/crop")
    identity_freqs = identities[0].get("operator_identity", {}).get("freqs_mhz", [])
    if not np.allclose(identity_freqs, freqs, rtol=0.0, atol=1e-10):
        raise ValueError("Compiled response frequencies do not match --freqs-mhz")

    config_payload = json.loads(args.power_config.read_text(encoding="utf-8"))
    config_payload["ref_freq_mhz"] = float(np.mean(freqs))
    config_payload["freq_grid_start_mhz"] = float(freqs[0])
    config_payload["df"] = float(np.median(np.diff(freqs)))
    config_payload["nbins_kperp"] = int(args.nbins_kperp)
    config_payload["nbins_kpar"] = int(args.nbins_kpar)
    config_payload["stat_mode"] = "mean"
    config = PowerSpecConfig(**config_payload)
    geometry = _build_fourier_geometry(shape, config)
    transform = BandpowerTransform(geometry, device)
    design = torch.as_tensor(design_np, dtype=torch.float64, device=device)
    design, design_compression = _compress_design_svd(
        design, int(args.nuisance_design_rank)
    )
    if str(args.nuisance_fit_scope) == "none":
        projector = IdentityNuisanceProjector(design)
    elif str(args.nuisance_fit_scope) == "fourier_wedge":
        projector = FourierWedgeNuisanceProjector(
            design,
            transform,
            rcond=float(args.nuisance_rcond),
            ridge_fraction=float(args.nuisance_ridge_fraction),
        )
    else:
        projector = NuisanceProjector(
            design,
            rcond=float(args.nuisance_rcond),
            ridge_fraction=float(args.nuisance_ridge_fraction),
        )
    del design_np

    operator_probe_psd = None
    operator_probe_metadata = None
    if args.operator_probe_psd_npz is not None:
        with np.load(args.operator_probe_psd_npz, allow_pickle=False) as payload:
            operator_probe_psd = np.asarray(payload["probe_psd"], dtype=np.float64)
            operator_probe_metadata = json.loads(
                str(np.asarray(payload["metadata_json"]).item())
            )
        if operator_probe_psd.shape != shape:
            raise ValueError(
                f"Operator-probe PSD shape mismatch: {operator_probe_psd.shape} != {shape}"
            )
        expected_probe_metadata = {
            "prior": "zero_mean_white_gaussian_sky",
            "freqs_mhz": [float(value) for value in freqs],
            "eval_crop_size": int(args.eval_crop_size),
        }
        for key, expected in expected_probe_metadata.items():
            if operator_probe_metadata.get(key) != expected:
                raise ValueError(
                    f"Operator-probe metadata mismatch for {key}: "
                    f"{operator_probe_metadata.get(key)!r} != {expected!r}"
                )

    operator_probe_bank = None
    operator_probe_bank_metadata = None
    if args.operator_probe_bank_npz is not None:
        with np.load(args.operator_probe_bank_npz, allow_pickle=False) as payload:
            operator_probe_bank = np.asarray(
                payload["dirty_probes"], dtype=np.float64
            )
            operator_probe_bank_metadata = json.loads(
                str(np.asarray(payload["metadata_json"]).item())
            )
        expected_bank_shape = (
            int(operator_probe_bank_metadata.get("probe_count", -1)),
            *shape,
        )
        if operator_probe_bank.shape != expected_bank_shape:
            raise ValueError(
                f"Operator-probe bank shape mismatch: {operator_probe_bank.shape} "
                f"!= {expected_bank_shape}"
            )
        expected_bank_metadata = {
            "prior": "zero_mean_white_gaussian_sky",
            "freqs_mhz": [float(value) for value in freqs],
            "eval_crop_size": int(args.eval_crop_size),
        }
        for key, expected in expected_bank_metadata.items():
            if operator_probe_bank_metadata.get(key) != expected:
                raise ValueError(
                    f"Operator-probe bank metadata mismatch for {key}: "
                    f"{operator_probe_bank_metadata.get(key)!r} != {expected!r}"
                )

    truth_fg = (
        _load_pattern_cube(str(args.truth_fg_pattern), freqs, int(args.eval_crop_size))
        if args.truth_fg_pattern is not None
        else None
    )
    truth_eor = (
        _load_pattern_cube(str(args.truth_eor_pattern), freqs, int(args.eval_crop_size))
        if args.truth_eor_pattern is not None
        else None
    )
    if args.observed_cube_fits is not None:
        observed = _load_cube(args.observed_cube_fits, shape)
    else:
        if truth_fg is None:
            raise AssertionError("Synthetic observation requires foreground truth")
        observed = truth_fg + (truth_eor if truth_eor is not None else 0.0)

    print(
        json.dumps(
            {
                "event": "marginalized_ps2d_loaded",
                "design_shape": [parameter_count, frequency_count, height, width],
                "band_count": geometry.band_count,
                "eor_window_band_count": int(np.sum(geometry.eor_window_active)),
                "device": str(device),
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    calibration_started = time.monotonic()
    calibration = _calibrate_transfer(
        geometry=geometry,
        bandpowers=transform,
        projector=projector,
        probes_per_band=int(args.transfer_probes_per_band),
        batch_size=int(args.transfer_probe_batch_size),
        seed=int(args.transfer_seed),
        transfer_rcond=float(args.transfer_rcond),
        probe_fourier_psd=operator_probe_psd,
        dirty_probe_bank=operator_probe_bank,
    )
    calibration_elapsed = float(time.monotonic() - calibration_started)
    print(
        json.dumps(
            {
                "event": "transfer_calibration_done",
                "elapsed_seconds": calibration_elapsed,
                "input_rank": calibration["input_response_svd"]["rank"],
                "transfer_rank": calibration["transfer_svd"]["rank"],
                "transfer_condition_retained": calibration["transfer_svd"][
                    "retained_condition_number"
                ],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    truth_bandpower = _cube_bandpowers(truth_eor, transform) if truth_eor is not None else None
    window_matrix = np.asarray(calibration["window_matrix"], dtype=np.float64)
    windowed_truth = (
        window_matrix @ truth_bandpower if truth_bandpower is not None else None
    )
    results: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {
        "transfer_matrix": np.asarray(calibration["transfer_matrix"]),
        "window_matrix": window_matrix,
        "deconvolution_matrix": np.asarray(calibration["deconvolution_matrix"]),
        "input_response": np.asarray(calibration["input_response"]),
        "projected_response": np.asarray(calibration["projected_response"]),
        "projected_response_standard_error": np.asarray(
            calibration["projected_response_standard_error"]
        ),
        "active_linear_bins": geometry.active_linear_bins,
        "active_kperp_indices": geometry.active_kperp_indices,
        "active_kpar_indices": geometry.active_kpar_indices,
        "eor_window_active": geometry.eor_window_active,
        "kperp_edges": geometry.kperp_edges,
        "kpar_edges": geometry.kpar_edges,
    }
    if truth_bandpower is not None:
        npz_payload["dirty_eor_bandpower"] = truth_bandpower
        npz_payload["windowed_dirty_eor_bandpower"] = windowed_truth

    for name, reference_path in args.candidate:
        reference = _load_cube(reference_path, shape)
        initial_residual = observed - reference
        residual_tensor = torch.as_tensor(
            initial_residual, dtype=torch.float64, device=device
        )
        projected_tensor, params_tensor = projector.project(residual_tensor)
        projected = projected_tensor.detach().cpu().numpy()
        fitted_params = params_tensor.detach().cpu().numpy()
        raw_bandpower = transform(residual_tensor).detach().cpu().numpy()[0]
        projected_bandpower = transform(projected_tensor).detach().cpu().numpy()[0]
        row_estimate, deconvolved_estimate = _estimate_from_projected(
            projected_bandpower, calibration
        )
        candidate_result: dict[str, Any] = {
            "reference_fits": str(reference_path),
            "reference_sha256": _sha256(reference_path),
            "fit_uses_truth": False,
            "nuisance_parameter_rms": (
                _rms(fitted_params) if fitted_params.size else 0.0
            ),
            "nuisance_parameter_l2": float(np.linalg.norm(fitted_params)),
            "initial_residual_rms": _rms(initial_residual),
            "projected_residual_rms": _rms(projected),
            "raw_bandpower": raw_bandpower,
            "projected_bandpower": projected_bandpower,
            "row_normalized_bandpower": row_estimate,
            "deconvolved_bandpower": deconvolved_estimate,
        }
        if truth_bandpower is not None and windowed_truth is not None:
            candidate_result["truth_diagnostic_only"] = {
                "raw_vs_dirty_eor": _bandpower_metrics(
                    raw_bandpower, truth_bandpower, geometry.eor_window_active
                ),
                "projected_uncorrected_vs_dirty_eor": _bandpower_metrics(
                    projected_bandpower, truth_bandpower, geometry.eor_window_active
                ),
                "row_normalized_vs_windowed_dirty_eor": _bandpower_metrics(
                    row_estimate, windowed_truth, geometry.eor_window_active
                ),
                "deconvolved_vs_dirty_eor": _bandpower_metrics(
                    deconvolved_estimate, truth_bandpower, geometry.eor_window_active
                ),
            }
            projected_eor_tensor, _ = projector.project(
                torch.as_tensor(truth_eor, dtype=torch.float64, device=device)
            )
            projected_eor_bp = transform(projected_eor_tensor).detach().cpu().numpy()[0]
            row_eor, deconvolved_eor = _estimate_from_projected(
                projected_eor_bp, calibration
            )
            candidate_result["truth_diagnostic_only"].update(
                {
                    "pure_eor_row_normalized_vs_windowed_dirty_eor": _bandpower_metrics(
                        row_eor, windowed_truth, geometry.eor_window_active
                    ),
                    "pure_eor_deconvolved_vs_dirty_eor": _bandpower_metrics(
                        deconvolved_eor, truth_bandpower, geometry.eor_window_active
                    ),
                }
            )
            if truth_fg is not None:
                fg_mismatch = truth_fg - reference
                projected_fg_tensor, _ = projector.project(
                    torch.as_tensor(fg_mismatch, dtype=torch.float64, device=device)
                )
                projected_fg_bp = transform(projected_fg_tensor).detach().cpu().numpy()[0]
                row_fg, deconvolved_fg = _estimate_from_projected(
                    projected_fg_bp, calibration
                )
                candidate_result["truth_diagnostic_only"].update(
                    {
                        "projected_foreground_mismatch_rms_over_dirty_eor": (
                            _rms(projected_fg_tensor.detach().cpu().numpy())
                            / max(_rms(truth_eor), 1e-300)
                        ),
                        "row_normalized_foreground_leakage_over_windowed_eor_sum": float(
                            np.sum(row_fg[geometry.eor_window_active])
                            / max(
                                float(
                                    np.sum(windowed_truth[geometry.eor_window_active])
                                ),
                                1e-300,
                            )
                        ),
                        "row_normalized_foreground_leakage_bins": (
                            _foreground_leakage_metrics(
                                row_fg,
                                windowed_truth,
                                geometry.eor_window_active,
                            )
                        ),
                        "deconvolved_foreground_leakage_over_eor_sum": float(
                            np.sum(deconvolved_fg[geometry.eor_window_active])
                            / max(
                                float(
                                    np.sum(truth_bandpower[geometry.eor_window_active])
                                ),
                                1e-300,
                            )
                        ),
                    }
                )
                npz_payload[f"{name}_projected_fg_bandpower"] = projected_fg_bp
        results[name] = candidate_result
        npz_payload[f"{name}_fitted_nuisance_params"] = fitted_params
        npz_payload[f"{name}_raw_bandpower"] = raw_bandpower
        npz_payload[f"{name}_projected_bandpower"] = projected_bandpower
        npz_payload[f"{name}_row_normalized_bandpower"] = row_estimate
        npz_payload[f"{name}_deconvolved_bandpower"] = deconvolved_estimate
        print(
            json.dumps(
                {
                    "event": "candidate_done",
                    "candidate": name,
                    "projected_residual_rms": candidate_result[
                        "projected_residual_rms"
                    ],
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = {
        "time_utc": _now(),
        "method": "compiled_foreground_nuisance_marginalized_quadratic_ps2d",
        "scientific_target": "dirty_eor_cylindrical_power_spectrum",
        "fit_uses_truth": False,
        "transfer_calibration_uses_truth": False,
        "truth_used_only_for_synthetic_observation_and_postfit_diagnostics": bool(
            truth_fg is not None or truth_eor is not None
        ),
        "design_npz_paths": [str(path) for path in design_paths],
        "design_sha256_list": [_sha256(path) for path in design_paths],
        "design_identities": identities,
        "design_shape": [parameter_count, frequency_count, height, width],
        "operator_probe_psd_npz": (
            str(args.operator_probe_psd_npz)
            if args.operator_probe_psd_npz is not None
            else None
        ),
        "operator_probe_psd_sha256": (
            _sha256(args.operator_probe_psd_npz)
            if args.operator_probe_psd_npz is not None
            else None
        ),
        "operator_probe_metadata": operator_probe_metadata,
        "operator_probe_bank_npz": (
            str(args.operator_probe_bank_npz)
            if args.operator_probe_bank_npz is not None
            else None
        ),
        "operator_probe_bank_sha256": (
            _sha256(args.operator_probe_bank_npz)
            if args.operator_probe_bank_npz is not None
            else None
        ),
        "operator_probe_bank_metadata": operator_probe_bank_metadata,
        "observed_cube_fits": (
            str(args.observed_cube_fits) if args.observed_cube_fits is not None else None
        ),
        "freqs_mhz": freqs,
        "power_config": config_payload,
        "geometry": {
            "active_band_count": geometry.band_count,
            "eor_window_active_band_count": int(np.sum(geometry.eor_window_active)),
            "active_linear_bins": geometry.active_linear_bins,
            "active_kperp_indices": geometry.active_kperp_indices,
            "active_kpar_indices": geometry.active_kpar_indices,
            "active_mode_counts": geometry.counts,
            "eor_window_active": geometry.eor_window_active,
            "kperp_edges": geometry.kperp_edges,
            "kpar_edges": geometry.kpar_edges,
            "kperp_centers": geometry.kperp_centers,
            "kpar_centers": geometry.kpar_centers,
            "power_scale": geometry.power_scale,
        },
        "nuisance_projection": projector.stats,
        "nuisance_design_compression": design_compression,
        "calibration": calibration,
        "results": results,
        "calibration_elapsed_seconds": calibration_elapsed,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    safe_output = _json_safe(output)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = Path(str(args.out_json) + f".tmp.{int(time.time())}")
    temporary_json.write_text(
        json.dumps(safe_output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_json.replace(args.out_json)

    out_npz = args.out_npz or args.out_json.with_suffix(".npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    npz_payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                "method": output["method"],
                "scientific_target": output["scientific_target"],
                "design_sha256_list": output["design_sha256_list"],
                "freqs_mhz": freqs.tolist(),
                "power_config": config_payload,
                "nuisance_projection": projector.stats,
                "nuisance_design_compression": design_compression,
                "calibration_seed": int(args.transfer_seed),
                "operator_probe_psd_sha256": output["operator_probe_psd_sha256"],
                "operator_probe_bank_sha256": output[
                    "operator_probe_bank_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    temporary_npz = Path(str(out_npz) + f".tmp.{int(time.time())}")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **npz_payload)
    temporary_npz.replace(out_npz)
    print(
        json.dumps(
            {
                "event": "marginalized_ps2d_done",
                "out_json": str(args.out_json),
                "out_npz": str(out_npz),
                "elapsed_seconds": output["elapsed_seconds"],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
