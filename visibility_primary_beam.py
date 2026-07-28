#!/usr/bin/env python3
"""Primary-beam factors for the matrix-free visibility operator."""

from __future__ import annotations

import hashlib
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def direction_cosine_geometry_sha256(
    *,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    n_minus_one: np.ndarray,
) -> str:
    """Hash the exact source-direction order used by a visibility operator."""
    directions = np.ascontiguousarray(
        np.stack(
            (
                np.asarray(l_cosine, dtype=np.float64),
                np.asarray(m_cosine, dtype=np.float64),
                np.asarray(n_minus_one, dtype=np.float64) + 1.0,
            )
        ),
        dtype=np.float64,
    )
    return hashlib.sha256(directions.view(np.uint8)).hexdigest()


def oskar_circular_gaussian_stokes_i_power(
    *,
    frequencies_hz: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    fwhm_deg: float,
    reference_frequency_hz: float,
) -> np.ndarray:
    """Return OSKAR's normalised circular-Gaussian Stokes-I response.

    OSKAR defines ``gaussian_beam/fwhm_deg`` as a voltage FWHM at the
    reference frequency. For an unpolarised Stokes-I source the baseline
    response is therefore the product of two voltage beams.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    l_values = np.asarray(l_cosine, dtype=np.float64).reshape(-1)
    m_values = np.asarray(m_cosine, dtype=np.float64).reshape(-1)
    if (
        frequencies.size < 1
        or l_values.shape != m_values.shape
        or not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.isfinite(l_values))
        or not np.all(np.isfinite(m_values))
        or not math.isfinite(float(fwhm_deg))
        or float(fwhm_deg) <= 0.0
        or not math.isfinite(float(reference_frequency_hz))
        or float(reference_frequency_hz) <= 0.0
    ):
        raise ValueError("Invalid Gaussian-beam frequency or direction geometry")
    sine_theta_squared = l_values**2 + m_values**2
    if np.any(sine_theta_squared > 1.0 + 1e-12):
        raise ValueError("Direction cosines lie outside the visible hemisphere")
    sine_theta_squared = np.clip(sine_theta_squared, 0.0, 1.0)
    effective_fwhm_rad = (
        math.radians(float(fwhm_deg))
        * float(reference_frequency_hz)
        / frequencies
    )
    sine_fwhm = np.sin(effective_fwhm_rad)
    if np.any(np.abs(sine_fwhm) < np.finfo(np.float64).tiny):
        raise ValueError("Effective Gaussian FWHM is numerically singular")
    exponent = (
        -8.0
        * math.log(2.0)
        * sine_theta_squared[None, :]
        / sine_fwhm[:, None] ** 2
    )
    return np.exp(exponent)


@dataclass(frozen=True)
class DirectionOnlyKernelMultiplier:
    """Broadcast a cached frequency/direction factor over visibility rows."""

    values: Any

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        del row_first, row_stop, device, complex_dtype
        return self.values[
            int(frequency_index), int(source_first) : int(source_stop)
        ][None, :]


@dataclass(frozen=True)
class TimeDirectionKernelMultiplier:
    """Select a cached frequency/time/direction factor for each row."""

    values: Any
    row_time_indices: Any

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        del device, complex_dtype
        time_indices = self.row_time_indices[int(row_first) : int(row_stop)]
        return self.values[
            int(frequency_index),
            time_indices,
            int(source_first) : int(source_stop),
        ]


@dataclass(frozen=True)
class RowDirectionFileKernelMultiplier:
    """Stream a complex baseline beam indexed by row and direction."""

    values: np.memmap

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        del frequency_index
        block = np.array(
            self.values[
                int(row_first) : int(row_stop),
                int(source_first) : int(source_stop),
            ],
            copy=True,
            order="C",
        )
        import torch

        return torch.as_tensor(
            block,
            dtype=complex_dtype,
            device=device,
        )


@dataclass(frozen=True)
class FrequencyRowDirectionFileKernelMultiplier:
    """Select a disk-backed station-pair beam for each frequency."""

    values: tuple[np.memmap, ...]

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        block = np.array(
            self.values[int(frequency_index)][
                int(row_first) : int(row_stop),
                int(source_first) : int(source_stop),
            ],
            copy=True,
            order="C",
        )
        import torch

        return torch.as_tensor(
            block,
            dtype=complex_dtype,
            device=device,
        )


@dataclass(frozen=True)
class IndexedFrequencyRowDirectionFileKernelMultiplier:
    """Select partition rows from a shared frequency/row beam cache."""

    values: tuple[np.memmap, ...]
    row_indices: tuple[np.ndarray, ...]

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        frequency = int(frequency_index)
        selected_rows = self.row_indices[frequency][
            int(row_first) : int(row_stop)
        ]
        block = np.array(
            self.values[frequency][
                selected_rows,
                int(source_first) : int(source_stop),
            ],
            copy=True,
            order="C",
        )
        import torch

        return torch.as_tensor(
            block,
            dtype=complex_dtype,
            device=device,
        )


@dataclass(frozen=True)
class DirectionCorrectedKernelMultiplier:
    """Apply a deterministic estimator-side correction to a cached PB."""

    base: Any
    correction: Any

    def __call__(
        self,
        frequency_index: int,
        row_first: int,
        row_stop: int,
        source_first: int,
        source_stop: int,
        device: Any,
        complex_dtype: Any,
    ) -> Any:
        values = self.base(
            frequency_index,
            row_first,
            row_stop,
            source_first,
            source_stop,
            device,
            complex_dtype,
        )
        return values * self.correction[
            int(frequency_index),
            int(source_first) : int(source_stop),
        ][None, :]


def primary_beam_model_correction(
    *,
    frequencies_hz: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    mode: str,
    edge_error_fraction: float,
    ripple_cycles: float = 2.0,
) -> np.ndarray:
    """Return a controlled multiplicative error for the estimator-side PB.

    The correction is one at phase centre. ``edge_error_fraction`` is the
    largest absolute fractional error at the most distant source direction.
    These profiles are sensitivity probes, not hardware-error simulations.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    l_values = np.asarray(l_cosine, dtype=np.float64).reshape(-1)
    m_values = np.asarray(m_cosine, dtype=np.float64).reshape(-1)
    error = float(edge_error_fraction)
    cycles = float(ripple_cycles)
    supported_modes = {
        "exact",
        "radial_static",
        "radial_linear",
        "radial_ripple",
    }
    if str(mode) not in supported_modes:
        raise ValueError(f"Unsupported primary-beam model error: {mode}")
    if (
        frequencies.size < 1
        or l_values.shape != m_values.shape
        or l_values.size < 1
        or not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.isfinite(l_values))
        or not np.all(np.isfinite(m_values))
        or not math.isfinite(error)
        or abs(error) > 0.5
        or not math.isfinite(cycles)
        or cycles <= 0.0
    ):
        raise ValueError("Invalid primary-beam model-error geometry")
    if str(mode) == "exact" and error != 0.0:
        raise ValueError("The exact PB model requires zero edge error")

    radius_squared = l_values**2 + m_values**2
    maximum_radius_squared = float(np.max(radius_squared))
    radial_profile = (
        radius_squared / maximum_radius_squared
        if maximum_radius_squared > 0.0
        else np.zeros_like(radius_squared)
    )
    if str(mode) == "exact" or error == 0.0:
        profile = np.zeros((frequencies.size, l_values.size), dtype=np.float64)
    elif str(mode) == "radial_static":
        profile = np.broadcast_to(
            radial_profile[None, :],
            (frequencies.size, l_values.size),
        )
    else:
        frequency_span = float(np.ptp(frequencies))
        if frequency_span <= 0.0:
            raise ValueError("A chromatic PB error requires multiple frequencies")
        frequency_phase = (frequencies - float(np.min(frequencies))) / frequency_span
        if str(mode) == "radial_linear":
            spectral_profile = 2.0 * frequency_phase - 1.0
        else:
            spectral_profile = np.sin(
                2.0 * math.pi * cycles * frequency_phase
            )
        profile = spectral_profile[:, None] * radial_profile[None, :]
    correction = 1.0 + error * profile
    if np.any(correction <= 0.0) or not np.all(np.isfinite(correction)):
        raise ValueError("Primary-beam model correction is non-positive")
    return np.asarray(correction, dtype=np.float64)


def build_direction_corrected_kernel_multiplier(
    *,
    base: Any,
    torch: Any,
    frequencies_hz: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    mode: str,
    edge_error_fraction: float,
    ripple_cycles: float,
    device: Any,
    operator_dtype: str,
) -> DirectionCorrectedKernelMultiplier:
    """Wrap an exact cached beam with a controlled estimator-side error."""
    correction = primary_beam_model_correction(
        frequencies_hz=frequencies_hz,
        l_cosine=l_cosine,
        m_cosine=m_cosine,
        mode=str(mode),
        edge_error_fraction=float(edge_error_fraction),
        ripple_cycles=float(ripple_cycles),
    )
    real_dtype = (
        torch.float32
        if str(operator_dtype) == "complex64"
        else torch.float64
    )
    return DirectionCorrectedKernelMultiplier(
        base=base,
        correction=torch.as_tensor(
            correction,
            dtype=real_dtype,
            device=device,
        ),
    )


def build_oskar_circular_gaussian_kernel_multiplier(
    *,
    torch: Any,
    frequencies_hz: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    fwhm_deg: float,
    reference_frequency_hz: float,
    device: Any,
    operator_dtype: str,
) -> DirectionOnlyKernelMultiplier:
    """Cache the OSKAR Gaussian beam on-device for matrix-free DFT calls."""
    values = oskar_circular_gaussian_stokes_i_power(
        frequencies_hz=frequencies_hz,
        l_cosine=l_cosine,
        m_cosine=m_cosine,
        fwhm_deg=float(fwhm_deg),
        reference_frequency_hz=float(reference_frequency_hz),
    )
    real_dtype = (
        torch.float32
        if str(operator_dtype) == "complex64"
        else torch.float64
    )
    return DirectionOnlyKernelMultiplier(
        values=torch.as_tensor(values, dtype=real_dtype, device=device)
    )


def build_time_direction_kernel_multiplier(
    *,
    torch: Any,
    values: np.ndarray,
    row_time_indices: np.ndarray,
    device: Any,
    operator_dtype: str,
) -> TimeDirectionKernelMultiplier:
    """Cache a scalar PB indexed by frequency, time, and direction."""
    beam = np.asarray(values)
    time_indices = np.asarray(row_time_indices, dtype=np.int64).reshape(-1)
    if (
        beam.ndim != 3
        or time_indices.size < 1
        or np.any(time_indices < 0)
        or np.any(time_indices >= beam.shape[1])
        or not np.all(np.isfinite(beam))
    ):
        raise ValueError("Invalid time-dependent primary-beam cache")
    real_dtype = (
        torch.float32
        if str(operator_dtype) == "complex64"
        else torch.float64
    )
    return TimeDirectionKernelMultiplier(
        values=torch.as_tensor(beam, dtype=real_dtype, device=device),
        row_time_indices=torch.as_tensor(
            time_indices, dtype=torch.int64, device=device
        ),
    )


def open_row_direction_kernel_multiplier(
    cache_dir: Path | str,
) -> tuple[RowDirectionFileKernelMultiplier, dict[str, Any]]:
    """Open an OSKAR station-pair coherency cache without loading it in RAM."""
    directory = Path(cache_dir)
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema") != "oskar_aperture_row_direction_coherency":
        raise ValueError("Unsupported aperture row-beam cache schema")
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Invalid aperture row-beam cache shape")
    dtype = np.dtype(str(metadata["dtype"]))
    if dtype not in (np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("Aperture row-beam cache must be complex")
    data_path = directory / str(metadata["data_file"])
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if data_path.stat().st_size != expected_bytes:
        raise ValueError("Aperture row-beam cache byte count differs")
    values = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
    return RowDirectionFileKernelMultiplier(values=values), metadata


def open_frequency_row_direction_kernel_multiplier(
    cache_dirs: list[Path | str] | tuple[Path | str, ...],
) -> tuple[FrequencyRowDirectionFileKernelMultiplier, list[dict[str, Any]]]:
    """Open one station-pair coherency cache per input frequency."""
    values: list[np.memmap] = []
    metadata: list[dict[str, Any]] = []
    expected_shape: tuple[int, ...] | None = None
    for cache_dir in cache_dirs:
        multiplier, current = open_row_direction_kernel_multiplier(cache_dir)
        shape = tuple(int(value) for value in current["shape"])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("Frequency row-beam cache shapes differ")
        values.append(multiplier.values)
        metadata.append(current)
    if not values:
        raise ValueError("At least one frequency row-beam cache is required")
    return (
        FrequencyRowDirectionFileKernelMultiplier(values=tuple(values)),
        metadata,
    )


def open_indexed_frequency_row_direction_kernel_multiplier(
    cache_dirs: list[Path | str] | tuple[Path | str, ...],
    *,
    selected_bank_rows: np.ndarray,
) -> tuple[
    IndexedFrequencyRowDirectionFileKernelMultiplier,
    list[dict[str, Any]],
    list[np.ndarray],
]:
    """Open shared row caches and map one evaluator partition into each."""
    requested = np.asarray(selected_bank_rows, dtype=np.int64).reshape(-1)
    if requested.size < 1 or np.unique(requested).size != requested.size:
        raise ValueError("Selected evaluator rows must be unique")
    values: list[np.memmap] = []
    metadata: list[dict[str, Any]] = []
    row_indices: list[np.ndarray] = []
    cached_rows_by_frequency: list[np.ndarray] = []
    expected_source_count: int | None = None
    for cache_dir in cache_dirs:
        directory = Path(cache_dir)
        multiplier, current = open_row_direction_kernel_multiplier(directory)
        with np.load(directory / "geometry.npz", allow_pickle=False) as geometry:
            cached_rows = np.asarray(
                geometry["selected_bank_rows"], dtype=np.int64
            ).reshape(-1)
        if (
            cached_rows.size != multiplier.values.shape[0]
            or np.unique(cached_rows).size != cached_rows.size
        ):
            raise ValueError("Aperture row-beam cache rows are invalid")
        source_count = int(multiplier.values.shape[1])
        if expected_source_count is None:
            expected_source_count = source_count
        elif source_count != expected_source_count:
            raise ValueError("Frequency row-beam source counts differ")
        lookup = {int(row): index for index, row in enumerate(cached_rows)}
        if any(int(row) not in lookup for row in requested):
            raise ValueError(
                "Aperture row-beam cache does not contain every evaluator row"
            )
        row_indices.append(
            np.asarray(
                [lookup[int(row)] for row in requested],
                dtype=np.int64,
            )
        )
        values.append(multiplier.values)
        metadata.append(current)
        cached_rows_by_frequency.append(cached_rows)
    if not values:
        raise ValueError("At least one frequency row-beam cache is required")
    return (
        IndexedFrequencyRowDirectionFileKernelMultiplier(
            values=tuple(values),
            row_indices=tuple(row_indices),
        ),
        metadata,
        cached_rows_by_frequency,
    )
