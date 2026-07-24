#!/usr/bin/env python3
"""Geometry and linear-algebra helpers for visibility-domain sky bandpowers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

C_M_S = 299_792_458.0
OMEGA_EARTH_RAD_S = 7.272205217e-5


@dataclass(frozen=True)
class SkyBandLayout:
    cube_shape: tuple[int, int, int]
    mode_bands: np.ndarray
    band_count: int
    kperp_edges: np.ndarray
    kpar_values: np.ndarray
    active_kperp_indices: np.ndarray
    active_kpar_indices: np.ndarray
    counts: np.ndarray


def direction_cosines(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    phase_ra_deg: float,
    phase_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert equatorial directions to phase-centre direction cosines."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    ra0 = math.radians(float(phase_ra_deg))
    dec0 = math.radians(float(phase_dec_deg))
    dra = np.mod(ra - ra0 + math.pi, 2.0 * math.pi) - math.pi
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_dec0 = math.sin(dec0)
    cos_dec0 = math.cos(dec0)
    l_cosine = cos_dec * np.sin(dra)
    m_cosine = sin_dec * cos_dec0 - cos_dec * sin_dec0 * np.cos(dra)
    n = sin_dec * sin_dec0 + cos_dec * cos_dec0 * np.cos(dra)
    return l_cosine, m_cosine, n


def oskar_time_smearing_cycles(
    uvw_m: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    n_minus_one: np.ndarray,
    *,
    frequency_hz: float,
    integration_time_s: float,
    phase_dec_deg: float,
) -> np.ndarray:
    """Return the argument of numpy.sinc for OSKAR time averaging.

    This is algebraically equivalent to OSKAR's station-baseline expression,
    but uses the row-centre UVW coordinates already stored in the bank.
    """
    uvw = np.asarray(uvw_m, dtype=np.float64)
    if uvw.ndim != 2 or uvw.shape[1] != 3:
        raise ValueError("uvw_m must have shape [row,3]")
    ll = np.asarray(l_cosine, dtype=np.float64).reshape(-1)
    mm = np.asarray(m_cosine, dtype=np.float64).reshape(-1)
    nn = np.asarray(n_minus_one, dtype=np.float64).reshape(-1)
    if ll.shape != mm.shape or ll.shape != nn.shape:
        raise ValueError("Direction-cosine vectors must have identical shapes")
    dec0 = math.radians(float(phase_dec_deg))
    u = uvw[:, 0:1]
    v = uvw[:, 1:2]
    w = uvw[:, 2:3]
    transverse = -math.sin(dec0) * v + math.cos(dec0) * w
    path_rate_per_radian = (
        transverse * ll[None, :]
        + u * math.sin(dec0) * mm[None, :]
        - u * math.cos(dec0) * nn[None, :]
    )
    return (
        float(frequency_hz)
        * float(integration_time_s)
        * OMEGA_EARTH_RAD_S
        * path_rate_per_radian
        / C_M_S
    )


def direct_dft_kernel_numpy(
    uvw_m: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    n_minus_one: np.ndarray,
    *,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    integration_time_s: float,
    phase_dec_deg: float,
) -> np.ndarray:
    """Build the no-PB direct-DFT kernel with OSKAR rectangular smearing."""
    uvw = np.asarray(uvw_m, dtype=np.float64)
    ll = np.asarray(l_cosine, dtype=np.float64).reshape(-1)
    mm = np.asarray(m_cosine, dtype=np.float64).reshape(-1)
    nn = np.asarray(n_minus_one, dtype=np.float64).reshape(-1)
    path_m = (
        uvw[:, 0:1] * ll[None, :]
        + uvw[:, 1:2] * mm[None, :]
        + uvw[:, 2:3] * nn[None, :]
    )
    delay_s = path_m / C_M_S
    bandwidth = np.sinc(delay_s * float(channel_bandwidth_hz))
    time = np.sinc(
        oskar_time_smearing_cycles(
            uvw,
            ll,
            mm,
            nn,
            frequency_hz=float(frequency_hz),
            integration_time_s=float(integration_time_s),
            phase_dec_deg=float(phase_dec_deg),
        )
    )
    phase = 2.0 * math.pi * float(frequency_hz) * delay_s
    return bandwidth * time * np.exp(1j * phase)


def build_sky_band_layout(
    cube_shape: tuple[int, int, int],
    *,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    kperp_edges: np.ndarray,
    exclude_radial_nyquist: bool = True,
) -> SkyBandLayout:
    """Assign a Hermitian-symmetric cylindrical band to each sky FFT mode."""
    nf, ny, nx = (int(value) for value in cube_shape)
    if min(nf, ny, nx) < 2:
        raise ValueError("Sky cube dimensions must all be at least two")
    edges = np.asarray(kperp_edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("kperp_edges must be finite and strictly increasing")
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=float(dx_mpc))
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=float(dy_mpc))
    kpar_signed = 2.0 * math.pi * np.fft.fftfreq(nf, d=float(dpar_mpc))
    absolute_indices = np.minimum(
        np.arange(nf, dtype=np.int64),
        (-np.arange(nf, dtype=np.int64)) % nf,
    )
    radial_keep = np.ones(nf, dtype=bool)
    if exclude_radial_nyquist and nf % 2 == 0:
        radial_keep[nf // 2] = False
    kept_abs = np.unique(absolute_indices[radial_keep])
    radial_lookup = np.full(int(np.max(absolute_indices)) + 1, -1, dtype=np.int64)
    radial_lookup[kept_abs] = np.arange(kept_abs.size, dtype=np.int64)
    radial_bands = radial_lookup[absolute_indices]
    kpar_values = np.abs(kpar_signed[kept_abs])

    kperp = np.hypot(ky[:, None], kx[None, :])
    transverse_bands = np.digitize(kperp, edges, right=False) - 1
    transverse_bands[np.isclose(kperp, edges[-1], rtol=1e-12, atol=1e-14)] = (
        edges.size - 2
    )
    transverse_valid = (transverse_bands >= 0) & (transverse_bands < edges.size - 1)
    nkpar = int(kpar_values.size)
    mode_bands = np.full((nf, ny, nx), -1, dtype=np.int64)
    for frequency_index in range(nf):
        radial_band = int(radial_bands[frequency_index])
        if radial_band < 0:
            continue
        plane = transverse_bands * nkpar + radial_band
        mode_bands[frequency_index, transverse_valid] = plane[transverse_valid]
    band_count = int((edges.size - 1) * nkpar)
    selected = mode_bands >= 0
    counts = np.bincount(mode_bands[selected], minlength=band_count).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("Sky layout contains an empty cylindrical band")
    return SkyBandLayout(
        cube_shape=(nf, ny, nx),
        mode_bands=mode_bands,
        band_count=band_count,
        kperp_edges=edges,
        kpar_values=kpar_values,
        active_kperp_indices=np.repeat(np.arange(edges.size - 1), nkpar),
        active_kpar_indices=np.tile(np.arange(nkpar), edges.size - 1),
        counts=counts,
    )


def reporting_band_ids(
    layout: SkyBandLayout,
    *,
    high_kpar_fraction: float,
    mid_kperp_fraction_range: tuple[float, float],
    radial_band_count: int | None = None,
) -> np.ndarray:
    """Return the predeclared high-kpar, intermediate-kperp band IDs."""
    high_fraction = float(high_kpar_fraction)
    low_perp, high_perp = (float(value) for value in mid_kperp_fraction_range)
    if not 0.0 <= high_fraction < 1.0:
        raise ValueError("high_kpar_fraction must lie in [0,1)")
    if not 0.0 <= low_perp < high_perp <= 1.0:
        raise ValueError("Invalid mid_kperp_fraction_range")
    nkperp = int(layout.kperp_edges.size - 1)
    available_nkpar = int(layout.kpar_values.size)
    nkpar = available_nkpar if radial_band_count is None else int(radial_band_count)
    if not 1 <= nkpar <= available_nkpar:
        raise ValueError("radial_band_count exceeds the sky layout")
    first_perp = int(math.floor(low_perp * nkperp))
    stop_perp = int(math.ceil(high_perp * nkperp))
    first_par = int(math.floor(high_fraction * nkpar))
    mask = (
        (layout.active_kperp_indices >= first_perp)
        & (layout.active_kperp_indices < stop_perp)
        & (layout.active_kpar_indices >= first_par)
        & (layout.active_kpar_indices < nkpar)
    )
    return np.flatnonzero(mask).astype(np.int64)


def band_selection_coverage(
    *,
    geometric_window: np.ndarray,
    selected_output_band_ids: np.ndarray,
    source_kperp_indices: np.ndarray,
    source_kpar_indices: np.ndarray,
    source_mode_counts: np.ndarray,
    source_bandpower: np.ndarray,
    output_cell_weights: np.ndarray | None = None,
    reporting_source_positions: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Measure a selected output footprint against a cylindrical EoR window.

    Source bands may contain an extra radial-Nyquist layer. Such bands are
    excluded from the geometric-window denominator because they are input
    response context rather than reported output bands.
    """
    window = np.asarray(geometric_window, dtype=bool)
    if window.ndim != 2 or not np.any(window):
        raise ValueError("geometric_window must be a non-empty 2D mask")
    selected_ids = np.asarray(selected_output_band_ids, dtype=np.int64).reshape(-1)
    if (
        selected_ids.size == 0
        or np.unique(selected_ids).size != selected_ids.size
        or np.any(selected_ids < 0)
        or np.any(selected_ids >= window.size)
    ):
        raise ValueError("selected_output_band_ids must be unique in-range IDs")
    selected_output = np.zeros(window.size, dtype=bool)
    selected_output[selected_ids] = True
    selected_output = selected_output.reshape(window.shape)
    if np.any(selected_output & ~window):
        raise ValueError("selected output bands must lie inside geometric_window")

    if output_cell_weights is None:
        cell_weights = np.ones(window.shape, dtype=np.float64)
    else:
        cell_weights = np.asarray(output_cell_weights, dtype=np.float64)
        if (
            cell_weights.shape != window.shape
            or np.any(~np.isfinite(cell_weights))
            or np.any(cell_weights <= 0.0)
        ):
            raise ValueError(
                "output_cell_weights must be finite, positive, and match the window"
            )

    source_kperp = np.asarray(source_kperp_indices, dtype=np.int64).reshape(-1)
    source_kpar = np.asarray(source_kpar_indices, dtype=np.int64).reshape(-1)
    mode_counts = np.asarray(source_mode_counts, dtype=np.float64).reshape(-1)
    bandpower = np.asarray(source_bandpower, dtype=np.float64).reshape(-1)
    source_size = source_kperp.size
    if not (source_kpar.size == mode_counts.size == bandpower.size == source_size):
        raise ValueError("source-band arrays must have identical lengths")
    if (
        np.any(source_kperp < 0)
        or np.any(source_kperp >= window.shape[0])
        or np.any(source_kpar < 0)
        or np.any(~np.isfinite(mode_counts))
        or np.any(mode_counts <= 0.0)
        or np.any(~np.isfinite(bandpower))
        or np.any(bandpower < 0.0)
    ):
        raise ValueError("invalid source-band indices, counts, or powers")

    output_radial = source_kpar < window.shape[1]
    source_in_window = np.zeros(source_size, dtype=bool)
    source_selected = np.zeros(source_size, dtype=bool)
    source_in_window[output_radial] = window[
        source_kperp[output_radial], source_kpar[output_radial]
    ]
    source_selected[output_radial] = selected_output[
        source_kperp[output_radial], source_kpar[output_radial]
    ]
    weighted_power = mode_counts * bandpower

    result: dict[str, float | int] = {
        "geometric_window_band_count": int(np.count_nonzero(window)),
        "selected_band_count": int(selected_ids.size),
        "selected_fraction_of_geometric_window_bands": float(
            selected_ids.size / np.count_nonzero(window)
        ),
        "selected_fraction_of_geometric_window_plot_area": float(
            np.sum(cell_weights[selected_output]) / np.sum(cell_weights[window])
        ),
        "geometric_window_fft_mode_count": int(np.sum(mode_counts[source_in_window])),
        "selected_fft_mode_count": int(np.sum(mode_counts[source_selected])),
        "selected_fraction_of_geometric_window_fft_modes": float(
            np.sum(mode_counts[source_selected]) / np.sum(mode_counts[source_in_window])
        ),
        "geometric_window_injected_power": float(
            np.sum(weighted_power[source_in_window])
        ),
        "selected_injected_power": float(np.sum(weighted_power[source_selected])),
        "selected_fraction_of_geometric_window_injected_power": float(
            np.sum(weighted_power[source_selected])
            / np.sum(weighted_power[source_in_window])
        ),
        "selected_fraction_of_all_response_scope_injected_power": float(
            np.sum(weighted_power[source_selected]) / np.sum(weighted_power)
        ),
    }
    if reporting_source_positions is not None:
        reporting_positions = np.asarray(
            reporting_source_positions, dtype=np.int64
        ).reshape(-1)
        if (
            reporting_positions.size == 0
            or np.unique(reporting_positions).size != reporting_positions.size
            or np.any(reporting_positions < 0)
            or np.any(reporting_positions >= source_size)
        ):
            raise ValueError(
                "reporting_source_positions must be unique in-range positions"
            )
        reporting = np.zeros(source_size, dtype=bool)
        reporting[reporting_positions] = True
        if np.any(reporting & ~source_in_window):
            raise ValueError("reporting source bands must lie in geometric_window")
        result.update(
            {
                "reporting_band_count": int(reporting_positions.size),
                "selected_fraction_of_reporting_bands": float(
                    np.count_nonzero(source_selected) / reporting_positions.size
                ),
                "selected_fraction_of_reporting_fft_modes": float(
                    np.sum(mode_counts[source_selected])
                    / np.sum(mode_counts[reporting])
                ),
                "selected_fraction_of_reporting_injected_power": float(
                    np.sum(weighted_power[source_selected])
                    / np.sum(weighted_power[reporting])
                ),
            }
        )
    return result


def stratified_row_indices(
    kperp_mpc_inv: np.ndarray,
    kperp_edges: np.ndarray,
    *,
    rows_per_bin: int,
    seed: int,
    partition_index: int = 0,
    partition_count: int = 1,
    bin_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select a deterministic random row subset in every transverse bin."""
    values = np.asarray(kperp_mpc_inv, dtype=np.float64).reshape(-1)
    edges = np.asarray(kperp_edges, dtype=np.float64).reshape(-1)
    count = int(rows_per_bin)
    part_index = int(partition_index)
    part_count = int(partition_count)
    if count < 1:
        raise ValueError("rows_per_bin must be positive")
    if part_count < 1 or not 0 <= part_index < part_count:
        raise ValueError("Invalid row partition index/count")
    if bin_indices is None:
        selected_bins = np.arange(edges.size - 1, dtype=np.int64)
    else:
        selected_bins = np.asarray(bin_indices, dtype=np.int64).reshape(-1)
        if (
            selected_bins.size == 0
            or np.unique(selected_bins).size != selected_bins.size
            or np.any(selected_bins < 0)
            or np.any(selected_bins >= edges.size - 1)
        ):
            raise ValueError("bin_indices must select unique transverse bins")
    generator = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    for index in selected_bins:
        index = int(index)
        members = np.flatnonzero(
            (values >= edges[index])
            & (
                (values < edges[index + 1])
                | (index == edges.size - 2 and np.isclose(values, edges[index + 1]))
            )
        )
        required = count * part_count
        if members.size < required:
            raise ValueError(
                f"kperp bin {index} has {members.size} rows, fewer than "
                f"{required} required by the disjoint partitions"
            )
        shuffled = generator.permutation(members)
        first = part_index * count
        chosen.append(np.sort(shuffled[first : first + count]))
    return np.concatenate(chosen).astype(np.int64, copy=False)


def source_bandpowers(
    cube: np.ndarray,
    layout: SkyBandLayout,
) -> np.ndarray:
    """Measure untapered orthonormal-FFT power in every source band."""
    values = np.asarray(cube)
    if values.shape[-3:] != layout.cube_shape:
        raise ValueError("Cube and sky-band layout shapes differ")
    spectrum = np.fft.fftn(values, axes=(-3, -2, -1), norm="ortho")
    leading = int(np.prod(values.shape[:-3])) if values.ndim > 3 else 1
    flat_power = np.square(np.abs(spectrum)).reshape(leading, -1)
    flat_bands = layout.mode_bands.reshape(-1)
    output = np.empty((leading, layout.band_count), dtype=np.float64)
    selected = flat_bands >= 0
    for row in range(leading):
        sums = np.bincount(
            flat_bands[selected],
            weights=flat_power[row, selected],
            minlength=layout.band_count,
        )
        output[row] = sums / layout.counts
    if values.ndim == 3:
        return output[0]
    return output.reshape((*values.shape[:-3], layout.band_count))


def weighted_response_pseudoinverse(
    response: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a row-scaled pseudoinverse for a rectangular response matrix."""
    matrix = np.asarray(response, dtype=np.float64)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("Response must be a finite matrix")
    row_scale = np.sqrt(np.sum(np.square(matrix), axis=1))
    threshold = max(float(np.max(row_scale)) * 1e-12, 1e-300)
    keep_rows = row_scale > threshold
    scaled = matrix[keep_rows] / row_scale[keep_rows, None]
    left, singular, right_t = np.linalg.svd(scaled, full_matrices=False)
    cutoff = float(rcond) * float(singular[0]) if singular.size else math.inf
    keep_singular = singular > cutoff
    inverse = np.zeros_like(singular)
    inverse[keep_singular] = 1.0 / singular[keep_singular]
    scaled_pinv = (right_t.T * inverse[None, :]) @ left.T
    pseudoinverse = np.zeros((matrix.shape[1], matrix.shape[0]), dtype=np.float64)
    pseudoinverse[:, keep_rows] = scaled_pinv / row_scale[keep_rows][None, :]
    retained_condition = (
        float(singular[0] / singular[keep_singular][-1])
        if np.any(keep_singular)
        else math.inf
    )
    return pseudoinverse, {
        "shape": [int(value) for value in matrix.shape],
        "kept_rows": int(np.count_nonzero(keep_rows)),
        "rank": int(np.count_nonzero(keep_singular)),
        "rcond": float(rcond),
        "cutoff": cutoff,
        "retained_condition_number": retained_condition,
        "singular_values": singular,
    }
