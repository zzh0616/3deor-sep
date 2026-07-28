#!/usr/bin/env python3
"""Local-redshift window definitions for visibility Q_beta estimators."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ps2d_v2_config import (
    FROZEN_GEOMETRY_KEYS,
    resolve_mode_first_analysis,
    resolve_mode_first_geometry,
)


@dataclass(frozen=True)
class LocalRedshiftWindow:
    """One guarded input band and its central analysis band."""

    index: int
    input_start: int
    input_stop: int
    analysis_start: int
    analysis_stop: int
    input_frequencies_mhz: np.ndarray
    analysis_frequencies_mhz: np.ndarray

    @property
    def label(self) -> str:
        return (
            f"local_{self.index:02d}_"
            f"{self.analysis_frequencies_mhz[0]:.1f}_"
            f"{self.analysis_frequencies_mhz[-1]:.1f}mhz"
        ).replace(".", "p")

    @property
    def reference_frequency_mhz(self) -> float:
        return float(np.mean(self.analysis_frequencies_mhz))


def frequency_subset_indices(
    available_frequencies: np.ndarray,
    requested_frequencies: np.ndarray,
    *,
    atol: float,
) -> np.ndarray:
    """Match a strictly ordered requested frequency view to a parent bank."""
    available = np.asarray(available_frequencies, dtype=np.float64).reshape(-1)
    requested = np.asarray(requested_frequencies, dtype=np.float64).reshape(-1)
    if (
        available.size < 1
        or requested.size < 1
        or np.any(~np.isfinite(available))
        or np.any(~np.isfinite(requested))
        or np.any(np.diff(available) <= 0.0)
        or np.any(np.diff(requested) <= 0.0)
    ):
        raise ValueError("Frequency axes must be finite and strictly increasing")
    indices = np.searchsorted(available, requested)
    if (
        np.any(indices >= available.size)
        or not np.allclose(
            available[np.minimum(indices, available.size - 1)],
            requested,
            rtol=0.0,
            atol=float(atol),
        )
    ):
        raise ValueError("Requested frequencies are not a subset of the parent axis")
    return indices.astype(np.int64, copy=False)


def build_local_redshift_windows(
    frequencies_mhz: np.ndarray,
    *,
    input_channel_count: int,
    analysis_channel_count: int,
    stride_channels: int,
    target_start: int,
    target_stop: int,
) -> list[LocalRedshiftWindow]:
    """Build overlapping guarded windows whose centres span a target interval."""
    frequencies = np.asarray(frequencies_mhz, dtype=np.float64).reshape(-1)
    input_count = int(input_channel_count)
    analysis_count = int(analysis_channel_count)
    stride = int(stride_channels)
    first_target = int(target_start)
    stop_target = int(target_stop)
    if (
        frequencies.size < 2
        or np.any(~np.isfinite(frequencies))
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError("frequencies_mhz must be finite and strictly increasing")
    if (
        input_count < 2
        or analysis_count < 2
        or analysis_count > input_count
        or (input_count - analysis_count) % 2
        or stride < 1
    ):
        raise ValueError("Invalid guarded local-window dimensions")
    if not 0 <= first_target < stop_target <= frequencies.size:
        raise ValueError("Invalid target channel interval")
    guard = (input_count - analysis_count) // 2
    windows: list[LocalRedshiftWindow] = []
    for input_start in range(
        0, frequencies.size - input_count + 1, stride
    ):
        input_stop = input_start + input_count
        analysis_start = input_start + guard
        analysis_stop = analysis_start + analysis_count
        centre = 0.5 * (analysis_start + analysis_stop - 1)
        if not first_target - 0.5 <= centre < stop_target:
            continue
        windows.append(
            LocalRedshiftWindow(
                index=len(windows),
                input_start=input_start,
                input_stop=input_stop,
                analysis_start=analysis_start,
                analysis_stop=analysis_stop,
                input_frequencies_mhz=frequencies[input_start:input_stop].copy(),
                analysis_frequencies_mhz=frequencies[
                    analysis_start:analysis_stop
                ].copy(),
            )
        )
    if not windows:
        raise ValueError("No local windows span the requested target interval")
    covered = np.zeros(frequencies.size, dtype=bool)
    for window in windows:
        covered[window.analysis_start : window.analysis_stop] = True
    if not np.all(covered[first_target:stop_target]):
        raise ValueError("Local analysis windows do not cover the target interval")
    return windows


def freeze_local_config(
    template: dict[str, Any],
    *,
    frequencies_mhz: np.ndarray,
    reference_frequency_mhz: float,
    status: str,
    frozen_geometry: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create a self-contained frozen config for one local frequency view."""
    config = copy.deepcopy(template)
    config["frequencies_mhz"] = [
        float(value)
        for value in np.asarray(frequencies_mhz, dtype=np.float64)
    ]
    config["reference_frequency_mhz"] = float(reference_frequency_mhz)
    config.pop("frozen_analysis_contract_sha256", None)
    config.pop("frozen_analysis_window_energy", None)
    config.pop("frozen_geometry", None)
    config.setdefault("legacy_reproduction", {})["status"] = str(status)
    if frozen_geometry is None:
        live = resolve_mode_first_analysis(config)
        config["frozen_geometry"] = {
            name: float(live.geometry[name])
            for name in FROZEN_GEOMETRY_KEYS
        }
    else:
        config["frozen_geometry"] = {
            name: float(frozen_geometry[name])
            for name in FROZEN_GEOMETRY_KEYS
        }
    live = resolve_mode_first_analysis(config)
    config["frozen_analysis_window_energy"] = float(
        live.contract.window_energy
    )
    frozen = resolve_mode_first_analysis(config)
    config["frozen_analysis_contract_sha256"] = (
        frozen.contract.analysis_contract_sha256
    )
    resolve_mode_first_analysis(config)
    return config


def freeze_frequency_view_config(
    template: dict[str, Any],
    *,
    frequencies_mhz: np.ndarray,
    reference_frequency_mhz: float,
    status: str,
    frozen_geometry: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Freeze only geometry and identity for an input-frequency view."""
    config = copy.deepcopy(template)
    config["frequencies_mhz"] = [
        float(value)
        for value in np.asarray(frequencies_mhz, dtype=np.float64)
    ]
    config["reference_frequency_mhz"] = float(reference_frequency_mhz)
    config.pop("frozen_analysis_contract_sha256", None)
    config.pop("frozen_analysis_window_energy", None)
    config.pop("frequency_view_contract_sha256", None)
    config.pop("frozen_geometry", None)
    config.setdefault("legacy_reproduction", {})["status"] = str(status)
    if frozen_geometry is None:
        live = resolve_mode_first_geometry(config)
        config["frozen_geometry"] = {
            name: float(live[name]) for name in FROZEN_GEOMETRY_KEYS
        }
    else:
        config["frozen_geometry"] = {
            name: float(frozen_geometry[name])
            for name in FROZEN_GEOMETRY_KEYS
        }
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    config["frequency_view_contract_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    resolve_mode_first_geometry(config)
    return config
