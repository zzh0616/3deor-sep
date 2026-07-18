#!/usr/bin/env python3
"""Mode-first cylindrical power-spectrum binning.

The v2 contract keeps the complete diagnostic plane separate from a science
window.  Window modes are selected before aggregation, so a bin intersected by
a physical boundary contains only the modes that pass that boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class EoRWindowSpec:
    kpar_min: float
    wedge_slope: float
    wedge_intercept: float
    kperp_min: Optional[float] = None
    kperp_max: Optional[float] = None
    kpar_max: Optional[float] = None
    exclude_exact_dc: bool = True

    def __post_init__(self) -> None:
        required = (self.kpar_min, self.wedge_slope, self.wedge_intercept)
        if not np.all(np.isfinite(required)):
            raise ValueError("EoR-window floor, slope, and intercept must be finite")
        optional = (self.kperp_min, self.kperp_max, self.kpar_max)
        if any(value is not None and not math.isfinite(float(value)) for value in optional):
            raise ValueError("Optional EoR-window bounds must be finite")
        if (
            self.kperp_min is not None
            and self.kperp_max is not None
            and float(self.kperp_max) < float(self.kperp_min)
        ):
            raise ValueError("EoR-window kperp bounds are inverted")
        if self.kpar_max is not None and float(self.kpar_max) < float(self.kpar_min):
            raise ValueError("EoR-window kpar_max is below its floor")

    def mask(self, kperp: np.ndarray, kpar: np.ndarray) -> np.ndarray:
        kp, kz = np.broadcast_arrays(
            np.asarray(kperp, dtype=np.float64),
            np.asarray(kpar, dtype=np.float64),
        )
        selected = (kz >= float(self.kpar_min)) & (
            kz >= float(self.wedge_slope) * kp + float(self.wedge_intercept)
        )
        if self.kperp_min is not None:
            selected &= kp >= float(self.kperp_min)
        if self.kperp_max is not None:
            selected &= kp <= float(self.kperp_max)
        if self.kpar_max is not None:
            selected &= kz <= float(self.kpar_max)
        if self.exclude_exact_dc:
            selected &= ~((kp == 0.0) & (kz == 0.0))
        return selected


@dataclass
class CylindricalModeLayout:
    cube_shape: tuple[int, int, int]
    dx_mpc: float
    dy_mpc: float
    dpar_mpc: float
    radial_nyquist_policy: str
    kperp_edges: np.ndarray
    kperp_centers: np.ndarray
    kpar_values: np.ndarray
    kpar_edges: np.ndarray
    full_mode_indices: np.ndarray
    full_mode_bands: np.ndarray
    selected_mode_indices: np.ndarray
    selected_mode_bands: np.ndarray
    full_fft_mode_counts: np.ndarray
    selected_fft_mode_counts: np.ndarray
    full_independent_mode_counts: np.ndarray
    selected_independent_mode_counts: np.ndarray
    selected_mode_fraction: np.ndarray
    full_kperp_mode_mean: np.ndarray
    selected_kperp_mode_mean: np.ndarray
    full_kpar_mode_mean: np.ndarray
    selected_kpar_mode_mean: np.ndarray
    transverse_circle_max: float

    @property
    def shape_2d(self) -> tuple[int, int]:
        return (int(self.kperp_centers.size), int(self.kpar_values.size))

    @property
    def band_count(self) -> int:
        return int(self.kperp_centers.size * self.kpar_values.size)


@dataclass
class BandpowerProduct:
    mean: np.ndarray
    power_sum: np.ndarray
    fft_mode_counts: np.ndarray
    independent_mode_counts: np.ndarray
    within_bin_std: np.ndarray

    @property
    def total_power(self) -> float:
        return float(np.sum(self.power_sum, dtype=np.float64))


@dataclass
class PS2DProducts:
    power_cube: np.ndarray
    fft_metadata: dict[str, Any]
    full_layout: CylindricalModeLayout
    window_layout: CylindricalModeLayout
    full: BandpowerProduct
    window: BandpowerProduct
    window_rectangular: BandpowerProduct

    @property
    def window_power_fraction(self) -> float:
        return self.window.total_power / max(self.full.total_power, 1e-300)


@dataclass
class ModeFirstAnalysisContract:
    full_layout: CylindricalModeLayout
    window_layout: CylindricalModeLayout
    analysis_window: np.ndarray
    window_energy: float
    power_scale: float
    demean_mode: str
    radial_taper: str
    spatial_taper: str
    layout_sha256: str
    analysis_contract_sha256: str


def _native_abs_kpar_groups(
    kpar_axis: np.ndarray,
    radial_nyquist_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.abs(np.asarray(kpar_axis, dtype=np.float64).reshape(-1))
    unique_values, inverse = np.unique(values, return_inverse=True)
    policy = str(radial_nyquist_policy).strip().lower()
    if policy not in {"include", "exclude"}:
        raise ValueError("radial_nyquist_policy must be 'include' or 'exclude'")

    keep = np.ones(unique_values.shape, dtype=bool)
    if policy == "exclude" and values.size % 2 == 0:
        nyquist_value = float(values[values.size // 2])
        keep &= ~np.isclose(
            unique_values, nyquist_value, rtol=1e-12, atol=1e-14
        )
    kept_values = unique_values[keep]
    if kept_values.size == 0:
        raise ValueError("No radial Fourier modes remain after Nyquist selection")

    unique_to_kept = np.full(unique_values.shape, -1, dtype=np.int64)
    unique_to_kept[keep] = np.arange(np.sum(keep), dtype=np.int64)
    per_frequency_group = unique_to_kept[inverse]
    return kept_values, per_frequency_group, _mode_edges(kept_values)


def _mode_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64).reshape(-1)
    if centers.size == 1:
        width = max(abs(float(centers[0])), 1.0)
        return np.asarray(
            [max(0.0, float(centers[0]) - 0.5 * width), float(centers[0]) + 0.5 * width],
            dtype=np.float64,
        )
    interior = 0.5 * (centers[:-1] + centers[1:])
    first = max(0.0, float(centers[0]) - 0.5 * float(centers[1] - centers[0]))
    last = float(centers[-1]) + 0.5 * float(centers[-1] - centers[-2])
    return np.concatenate(([first], interior, [last])).astype(np.float64)


def linear_kperp_edges(kperp_min: float, kperp_max: float, nbins: int) -> np.ndarray:
    lower = float(kperp_min)
    upper = float(kperp_max)
    count = int(nbins)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("kperp bounds must be finite and increasing")
    if count <= 0:
        raise ValueError("nbins must be positive")
    return np.linspace(lower, upper, count + 1, dtype=np.float64)


def _bin_kperp(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    kp = np.asarray(values, dtype=np.float64)
    bins = np.searchsorted(edges, kp, side="right") - 1
    on_right_edge = np.isclose(kp, float(edges[-1]), rtol=1e-12, atol=1e-14)
    bins[on_right_edge] = int(edges.size - 2)
    return bins.astype(np.int64, copy=False)


def _canonical_conjugate_mask(
    flat_indices: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    fi, yi, xi = np.unravel_index(flat_indices, shape)
    conjugate = np.ravel_multi_index(
        ((-fi) % shape[0], (-yi) % shape[1], (-xi) % shape[2]),
        shape,
    )
    return flat_indices <= conjugate


def _counts(
    bands: np.ndarray,
    band_count: int,
    canonical: Optional[np.ndarray] = None,
) -> np.ndarray:
    use = bands if canonical is None else bands[np.asarray(canonical, dtype=bool)]
    return np.bincount(use, minlength=int(band_count)).astype(np.int64, copy=False)


def _binned_mean(
    values: np.ndarray,
    bands: np.ndarray,
    counts: np.ndarray,
    band_count: int,
) -> np.ndarray:
    sums = np.bincount(
        bands,
        weights=np.asarray(values, dtype=np.float64),
        minlength=int(band_count),
    )
    output = np.full((int(band_count),), np.nan, dtype=np.float64)
    populated = counts > 0
    output[populated] = sums[populated] / counts[populated]
    return output


def build_cylindrical_mode_layout(
    shape: tuple[int, int, int],
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    kperp_edges: np.ndarray,
    radial_nyquist_policy: str = "exclude",
    window_spec: Optional[EoRWindowSpec] = None,
    transverse_circle_max: Optional[float] = None,
) -> CylindricalModeLayout:
    """Build a reusable, data-independent mapping from FFT modes to PS2D bands."""
    nf, ny, nx = (int(value) for value in shape)
    if min(nf, ny, nx) <= 0:
        raise ValueError("All cube dimensions must be positive")
    if min(float(dx_mpc), float(dy_mpc), float(dpar_mpc)) <= 0.0:
        raise ValueError("Physical spacings must be positive")
    edges = np.asarray(kperp_edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("kperp_edges must be finite and strictly increasing")

    kpar_axis = 2.0 * math.pi * np.fft.fftfreq(nf, d=float(dpar_mpc))
    ky_axis = 2.0 * math.pi * np.fft.fftfreq(ny, d=float(dy_mpc))
    kx_axis = 2.0 * math.pi * np.fft.fftfreq(nx, d=float(dx_mpc))
    kpar_values, frequency_groups, kpar_edges = _native_abs_kpar_groups(
        kpar_axis, radial_nyquist_policy
    )

    circle_max = (
        min(float(np.max(np.abs(ky_axis))), float(np.max(np.abs(kx_axis))))
        if transverse_circle_max is None
        else float(transverse_circle_max)
    )
    if not math.isfinite(circle_max) or circle_max < 0.0:
        raise ValueError("transverse_circle_max must be finite and non-negative")
    kpar_grid, ky_grid, kx_grid = np.meshgrid(
        np.abs(kpar_axis), ky_axis, kx_axis, indexing="ij"
    )
    kperp_grid = np.sqrt(np.square(ky_grid) + np.square(kx_grid))
    kperp_bins = _bin_kperp(kperp_grid, edges)
    kpar_bins = np.broadcast_to(frequency_groups[:, None, None], shape)
    nkperp = int(edges.size - 1)
    nkpar = int(kpar_values.size)
    band_count = nkperp * nkpar

    full_mask = (
        (kperp_grid <= circle_max)
        & (kperp_bins >= 0)
        & (kperp_bins < nkperp)
        & (kpar_bins >= 0)
        & (kpar_bins < nkpar)
    )
    selected_mask = np.array(full_mask, copy=True)
    if window_spec is not None:
        selected_mask &= window_spec.mask(kperp_grid, kpar_grid)

    full_indices = np.flatnonzero(full_mask.reshape(-1)).astype(np.int64)
    selected_indices = np.flatnonzero(selected_mask.reshape(-1)).astype(np.int64)
    linear_grid = kperp_bins * nkpar + kpar_bins
    full_bands = linear_grid[full_mask].astype(np.int64, copy=False)
    selected_bands = linear_grid[selected_mask].astype(np.int64, copy=False)
    full_canonical = _canonical_conjugate_mask(full_indices, shape)
    selected_canonical = _canonical_conjugate_mask(selected_indices, shape)

    full_counts_flat = _counts(full_bands, band_count)
    selected_counts_flat = _counts(selected_bands, band_count)
    full_independent_flat = _counts(full_bands, band_count, full_canonical)
    selected_independent_flat = _counts(
        selected_bands, band_count, selected_canonical
    )
    fraction = np.zeros((band_count,), dtype=np.float64)
    populated = full_counts_flat > 0
    fraction[populated] = (
        selected_counts_flat[populated] / full_counts_flat[populated]
    )

    kperp_flat = kperp_grid.reshape(-1)
    kpar_flat = kpar_grid.reshape(-1)
    full_kperp_mean = _binned_mean(
        kperp_flat[full_indices], full_bands, full_counts_flat, band_count
    )
    selected_kperp_mean = _binned_mean(
        kperp_flat[selected_indices],
        selected_bands,
        selected_counts_flat,
        band_count,
    )
    full_kpar_mean = _binned_mean(
        kpar_flat[full_indices], full_bands, full_counts_flat, band_count
    )
    selected_kpar_mean = _binned_mean(
        kpar_flat[selected_indices],
        selected_bands,
        selected_counts_flat,
        band_count,
    )
    shape_2d = (nkperp, nkpar)
    return CylindricalModeLayout(
        cube_shape=(nf, ny, nx),
        dx_mpc=float(dx_mpc),
        dy_mpc=float(dy_mpc),
        dpar_mpc=float(dpar_mpc),
        radial_nyquist_policy=str(radial_nyquist_policy).strip().lower(),
        kperp_edges=edges,
        kperp_centers=0.5 * (edges[:-1] + edges[1:]),
        kpar_values=kpar_values,
        kpar_edges=kpar_edges,
        full_mode_indices=full_indices,
        full_mode_bands=full_bands,
        selected_mode_indices=selected_indices,
        selected_mode_bands=selected_bands,
        full_fft_mode_counts=full_counts_flat.reshape(shape_2d),
        selected_fft_mode_counts=selected_counts_flat.reshape(shape_2d),
        full_independent_mode_counts=full_independent_flat.reshape(shape_2d),
        selected_independent_mode_counts=selected_independent_flat.reshape(shape_2d),
        selected_mode_fraction=fraction.reshape(shape_2d),
        full_kperp_mode_mean=full_kperp_mean.reshape(shape_2d),
        selected_kperp_mode_mean=selected_kperp_mean.reshape(shape_2d),
        full_kpar_mode_mean=full_kpar_mean.reshape(shape_2d),
        selected_kpar_mode_mean=selected_kpar_mean.reshape(shape_2d),
        transverse_circle_max=circle_max,
    )


def aggregate_power_cube(
    power_cube: np.ndarray,
    layout: CylindricalModeLayout,
    *,
    selected: bool,
    allow_negative: bool = False,
) -> BandpowerProduct:
    power = np.asarray(power_cube, dtype=np.float64)
    if tuple(power.shape) != tuple(layout.cube_shape):
        raise ValueError(
            f"Power cube shape {tuple(power.shape)} != layout {layout.cube_shape}"
        )
    if not np.all(np.isfinite(power)):
        raise ValueError("Mode-power cube must be finite")
    if not bool(allow_negative) and np.any(power < 0.0):
        raise ValueError("Auto-power cube must be non-negative")
    if selected:
        indices = layout.selected_mode_indices
        bands = layout.selected_mode_bands
        counts = layout.selected_fft_mode_counts.reshape(-1)
        independent = layout.selected_independent_mode_counts
    else:
        indices = layout.full_mode_indices
        bands = layout.full_mode_bands
        counts = layout.full_fft_mode_counts.reshape(-1)
        independent = layout.full_independent_mode_counts

    values = power.reshape(-1)[indices]
    band_count = int(layout.band_count)
    sums = np.bincount(bands, weights=values, minlength=band_count)
    sums_sq = np.bincount(bands, weights=np.square(values), minlength=band_count)
    means = np.full((band_count,), np.nan, dtype=np.float64)
    std = np.full((band_count,), np.nan, dtype=np.float64)
    populated = counts > 0
    means[populated] = sums[populated] / counts[populated]
    enough = counts > 1
    variance = np.zeros((band_count,), dtype=np.float64)
    variance[enough] = (
        sums_sq[enough]
        - np.square(sums[enough]) / counts[enough]
    ) / (counts[enough] - 1)
    std[enough] = np.sqrt(np.maximum(variance[enough], 0.0))
    shape_2d = layout.shape_2d
    return BandpowerProduct(
        mean=means.reshape(shape_2d),
        power_sum=sums.reshape(shape_2d),
        fft_mode_counts=counts.reshape(shape_2d),
        independent_mode_counts=np.asarray(independent, dtype=np.int64),
        within_bin_std=std.reshape(shape_2d),
    )


def _window_1d(length: int, name: str) -> np.ndarray:
    count = int(length)
    if count <= 0:
        raise ValueError("Window length must be positive")
    window_name = str(name).strip().lower()
    if window_name == "none":
        return np.ones((count,), dtype=np.float64)
    if window_name == "hann":
        return np.hanning(count).astype(np.float64, copy=False)
    if window_name == "blackman_harris":
        if count == 1:
            return np.ones((1,), dtype=np.float64)
        phase = 2.0 * math.pi * np.arange(count, dtype=np.float64) / (count - 1)
        return (
            0.35875
            - 0.48829 * np.cos(phase)
            + 0.14128 * np.cos(2.0 * phase)
            - 0.01168 * np.cos(3.0 * phase)
        )
    raise ValueError("Taper must be 'none', 'hann', or 'blackman_harris'")


def build_analysis_window(
    shape: tuple[int, int, int],
    *,
    radial_taper: str,
    spatial_taper: str,
) -> tuple[np.ndarray, float]:
    nf, ny, nx = (int(value) for value in shape)
    if min(nf, ny, nx) <= 0:
        raise ValueError("All analysis-window dimensions must be positive")
    radial = _window_1d(nf, radial_taper)
    spatial_y = _window_1d(ny, spatial_taper)
    spatial_x = _window_1d(nx, spatial_taper)
    window = radial[:, None, None] * spatial_y[None, :, None] * spatial_x[None, None, :]
    energy = float(np.mean(np.square(window), dtype=np.float64))
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("Taper has zero or invalid energy")
    return np.asarray(window, dtype=np.float64), energy


def mode_layout_sha256(
    full_layout: CylindricalModeLayout,
    window_layout: CylindricalModeLayout,
) -> str:
    if full_layout.cube_shape != window_layout.cube_shape:
        raise ValueError("Full and window layouts must use the same cube shape")
    payload = {
        "cube_shape": window_layout.cube_shape,
        "spacings_mpc": [
            window_layout.dx_mpc,
            window_layout.dy_mpc,
            window_layout.dpar_mpc,
        ],
        "radial_nyquist_policy": window_layout.radial_nyquist_policy,
        "full_kperp_edges": full_layout.kperp_edges.tolist(),
        "window_kperp_edges": window_layout.kperp_edges.tolist(),
        "kpar_values": window_layout.kpar_values.tolist(),
        "full_mode_indices_sha256": hashlib.sha256(
            full_layout.full_mode_indices.tobytes()
        ).hexdigest(),
        "full_mode_bands_sha256": hashlib.sha256(
            full_layout.full_mode_bands.tobytes()
        ).hexdigest(),
        "selected_mode_indices_sha256": hashlib.sha256(
            window_layout.selected_mode_indices.tobytes()
        ).hexdigest(),
        "selected_mode_bands_sha256": hashlib.sha256(
            window_layout.selected_mode_bands.tobytes()
        ).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def analysis_contract_sha256(
    *,
    layout_sha256: str,
    demean_mode: str,
    radial_taper: str,
    spatial_taper: str,
    window_energy: float,
    voxel_volume_mpc3: float,
    power_scale: float,
) -> str:
    payload = {
        "schema": "ps2d_v2_analysis_contract",
        "layout_sha256": str(layout_sha256),
        "demean_mode": str(demean_mode).strip().lower(),
        "radial_taper": str(radial_taper).strip().lower(),
        "spatial_taper": str(spatial_taper).strip().lower(),
        "window_energy": float(window_energy),
        "voxel_volume_mpc3": float(voxel_volume_mpc3),
        "power_scale": float(power_scale),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_mode_first_analysis_contract(
    shape: tuple[int, int, int],
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    full_kperp_edges: np.ndarray,
    window_kperp_edges: np.ndarray,
    window_spec: EoRWindowSpec,
    radial_nyquist_policy: str,
    demean_mode: str,
    radial_taper: str,
    spatial_taper: str,
) -> ModeFirstAnalysisContract:
    full_layout = build_cylindrical_mode_layout(
        shape,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        kperp_edges=full_kperp_edges,
        radial_nyquist_policy=radial_nyquist_policy,
    )
    window_layout = build_cylindrical_mode_layout(
        shape,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        kperp_edges=window_kperp_edges,
        radial_nyquist_policy=radial_nyquist_policy,
        window_spec=window_spec,
        transverse_circle_max=full_layout.transverse_circle_max,
    )
    window, window_energy = build_analysis_window(
        shape,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    spacings = np.asarray([dx_mpc, dy_mpc, dpar_mpc], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("Physical spacings must be finite and positive")
    voxel_volume = float(np.prod(spacings))
    power_scale = voxel_volume / (window_energy * float(np.prod(shape)))
    layout_hash = mode_layout_sha256(full_layout, window_layout)
    contract_hash = analysis_contract_sha256(
        layout_sha256=layout_hash,
        demean_mode=demean_mode,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
        window_energy=window_energy,
        voxel_volume_mpc3=voxel_volume,
        power_scale=power_scale,
    )
    return ModeFirstAnalysisContract(
        full_layout=full_layout,
        window_layout=window_layout,
        analysis_window=window,
        window_energy=window_energy,
        power_scale=power_scale,
        demean_mode=str(demean_mode).strip().lower(),
        radial_taper=str(radial_taper).strip().lower(),
        spatial_taper=str(spatial_taper).strip().lower(),
        layout_sha256=layout_hash,
        analysis_contract_sha256=contract_hash,
    )


def _prepare_spectrum(
    cube: np.ndarray,
    *,
    demean_mode: str,
    radial_taper: str,
    spatial_taper: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(cube, dtype=np.float64)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("cube must be a finite [frequency,y,x] array")
    mode = str(demean_mode).strip().lower()
    if mode == "global":
        prepared = values - np.mean(values, dtype=np.float64)
    elif mode == "per_freq_spatial":
        prepared = values - np.mean(values, axis=(1, 2), keepdims=True)
    elif mode == "none":
        prepared = np.array(values, copy=True)
    else:
        raise ValueError("demean_mode must be global, per_freq_spatial, or none")

    window, window_energy = build_analysis_window(
        tuple(int(value) for value in values.shape),
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    tapered = prepared * window
    spectrum = np.fft.fftn(tapered)
    return spectrum, tapered, {
        "cube_shape": list(values.shape),
        "demean_mode": mode,
        "radial_taper": str(radial_taper).strip().lower(),
        "spatial_taper": str(spatial_taper).strip().lower(),
        "window_energy": window_energy,
    }


def fft_auto_power_cube(
    cube: np.ndarray,
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    demean_mode: str = "global",
    radial_taper: str = "hann",
    spatial_taper: str = "hann",
) -> tuple[np.ndarray, dict[str, Any]]:
    spectrum, tapered, metadata = _prepare_spectrum(
        cube,
        demean_mode=demean_mode,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    window_energy = float(metadata["window_energy"])
    spacings = np.asarray([dx_mpc, dy_mpc, dpar_mpc], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("Physical spacings must be finite and positive")
    voxel_volume = float(np.prod(spacings))
    scale = voxel_volume / (window_energy * float(tapered.size))
    power = scale * np.square(np.abs(spectrum))
    parseval_real = voxel_volume / window_energy * float(
        np.sum(np.square(tapered), dtype=np.float64)
    )
    parseval_fourier = float(np.sum(power, dtype=np.float64))
    metadata.update({
        "schema": "ps2d_v2_mode_first",
        "voxel_volume_mpc3": voxel_volume,
        "power_scale": scale,
        "parseval_real_space_power": parseval_real,
        "parseval_fourier_power": parseval_fourier,
        "parseval_relative_error": parseval_fourier / max(parseval_real, 1e-300) - 1.0,
    })
    return np.asarray(power, dtype=np.float64), metadata


def fft_cross_power_cube(
    cube_a: np.ndarray,
    cube_b: np.ndarray,
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    demean_mode: str = "global",
    radial_taper: str = "hann",
    spatial_taper: str = "hann",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Re[F(a) F(b)*] with the same normalization as auto-power."""
    spectrum_a, tapered_a, metadata_a = _prepare_spectrum(
        cube_a,
        demean_mode=demean_mode,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    spectrum_b, tapered_b, metadata_b = _prepare_spectrum(
        cube_b,
        demean_mode=demean_mode,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    if tapered_a.shape != tapered_b.shape:
        raise ValueError("Cross-power cubes must have the same shape")
    if metadata_a != metadata_b:
        raise ValueError("Cross-power cubes must use identical analysis settings")
    window_energy = float(metadata_a["window_energy"])
    spacings = np.asarray([dx_mpc, dy_mpc, dpar_mpc], dtype=np.float64)
    if not np.all(np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("Physical spacings must be finite and positive")
    voxel_volume = float(np.prod(spacings))
    scale = voxel_volume / (window_energy * float(tapered_a.size))
    cross = scale * np.real(spectrum_a * np.conjugate(spectrum_b))
    metadata = dict(metadata_a)
    metadata.update(
        {
            "schema": "ps2d_v2_mode_first_cross",
            "voxel_volume_mpc3": voxel_volume,
            "power_scale": scale,
        }
    )
    return np.asarray(cross, dtype=np.float64), metadata


def compute_ps2d_products(
    cube: np.ndarray,
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    full_kperp_edges: np.ndarray,
    window_kperp_edges: np.ndarray,
    window_spec: EoRWindowSpec,
    radial_nyquist_policy: str = "exclude",
    demean_mode: str = "global",
    radial_taper: str = "hann",
    spatial_taper: str = "hann",
) -> PS2DProducts:
    power, metadata = fft_auto_power_cube(
        cube,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        demean_mode=demean_mode,
        radial_taper=radial_taper,
        spatial_taper=spatial_taper,
    )
    shape = tuple(int(value) for value in power.shape)
    full_layout = build_cylindrical_mode_layout(
        shape,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        kperp_edges=full_kperp_edges,
        radial_nyquist_policy=radial_nyquist_policy,
    )
    window_layout = build_cylindrical_mode_layout(
        shape,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        kperp_edges=window_kperp_edges,
        radial_nyquist_policy=radial_nyquist_policy,
        window_spec=window_spec,
        transverse_circle_max=full_layout.transverse_circle_max,
    )
    return PS2DProducts(
        power_cube=power,
        fft_metadata=metadata,
        full_layout=full_layout,
        window_layout=window_layout,
        full=aggregate_power_cube(power, full_layout, selected=False),
        window=aggregate_power_cube(power, window_layout, selected=True),
        window_rectangular=aggregate_power_cube(
            power, window_layout, selected=False
        ),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return float("nan")
    cdf = np.cumsum(sorted_weights) / total
    index = int(np.searchsorted(cdf, float(q), side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def compare_bandpowers(
    recovered: BandpowerProduct,
    truth: BandpowerProduct,
    *,
    eps: float = 1e-300,
) -> dict[str, float]:
    if recovered.mean.shape != truth.mean.shape:
        raise ValueError("Recovered and truth bandpower shapes differ")
    if not np.array_equal(recovered.fft_mode_counts, truth.fft_mode_counts):
        raise ValueError("Recovered and truth products must use the same mode layout")
    valid = (
        (truth.fft_mode_counts > 0)
        & np.isfinite(recovered.mean)
        & np.isfinite(truth.mean)
        & (truth.mean > 0.0)
    )
    if not np.any(valid):
        return {
            "n_bands": 0.0,
            "fft_mode_count": 0.0,
            "independent_mode_count": 0.0,
            "band_log10_mad": float("nan"),
            "independent_weighted_log10_mad": float("nan"),
            "independent_weighted_log10_rmse": float("nan"),
            "power_sum_ratio": float("nan"),
        }
    log_valid = valid & (recovered.mean > 0.0)
    rec_sum = float(np.sum(recovered.power_sum[valid], dtype=np.float64))
    truth_sum = float(np.sum(truth.power_sum[valid], dtype=np.float64))
    if not np.any(log_valid):
        return {
            "n_bands": float(np.sum(valid)),
            "n_log_bands": 0.0,
            "fft_mode_count": float(np.sum(truth.fft_mode_counts[valid])),
            "independent_mode_count": float(
                np.sum(truth.independent_mode_counts[valid])
            ),
            "band_log10_mad": float("nan"),
            "independent_weighted_log10_mad": float("nan"),
            "independent_weighted_log10_rmse": float("nan"),
            "power_sum_ratio": rec_sum / max(truth_sum, float(eps)),
        }
    dlog = np.log10(recovered.mean[log_valid] + float(eps)) - np.log10(
        truth.mean[log_valid] + float(eps)
    )
    weights = truth.independent_mode_counts[log_valid].astype(np.float64)
    weight_sum = float(np.sum(weights))
    weighted_rmse = (
        float(np.sqrt(np.sum(weights * np.square(dlog)) / weight_sum))
        if weight_sum > 0.0
        else float("nan")
    )
    return {
        "n_bands": float(np.sum(valid)),
        "n_log_bands": float(np.sum(log_valid)),
        "fft_mode_count": float(np.sum(truth.fft_mode_counts[valid])),
        "independent_mode_count": float(
            np.sum(truth.independent_mode_counts[valid])
        ),
        "band_log10_mad": float(np.median(np.abs(dlog))),
        "independent_weighted_log10_mad": _weighted_quantile(
            np.abs(dlog), weights, 0.5
        ),
        "independent_weighted_log10_rmse": weighted_rmse,
        "power_sum_ratio": rec_sum / max(truth_sum, float(eps)),
    }
