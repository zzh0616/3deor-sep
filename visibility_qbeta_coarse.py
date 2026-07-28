#!/usr/bin/env python3
"""Response-aware coarse bandpowers for exact-visibility Q_beta products."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoarseGroup:
    output_positions: np.ndarray
    kperp_first: int
    kperp_stop: int
    kpar_first: int
    kpar_stop: int


def kperp_intervals(profile: str, count: int) -> list[tuple[int, int]]:
    """Return predeclared contiguous transverse-bin intervals."""
    if count < 1:
        raise ValueError("count must be positive")
    name = str(profile)
    if name == "fine":
        return [(index, index + 1) for index in range(count)]
    if name == "low4":
        if count < 4:
            raise ValueError("low4 requires at least four transverse bins")
        return [(0, 4), *[(index, index + 1) for index in range(4, count)]]
    if name == "low4_high2":
        if count < 6:
            raise ValueError("low4_high2 requires at least six transverse bins")
        return [
            (0, 4),
            *[(index, index + 1) for index in range(4, count - 2)],
            (count - 2, count),
        ]
    if name in {"low4_mid3_high2", "low4_mid4_high2"}:
        mid_first = 18 if name == "low4_mid3_high2" else 17
        mid_stop = 21
        if count < mid_stop + 2:
            raise ValueError(f"{name} requires at least {mid_stop + 2} bins")
        return [
            (0, 4),
            *[(index, index + 1) for index in range(4, mid_first)],
            (mid_first, mid_stop),
            *[(index, index + 1) for index in range(mid_stop, count - 2)],
            (count - 2, count),
        ]
    if name in {"pair", "quad"}:
        width = 2 if name == "pair" else 4
        return [
            (first, min(first + width, count))
            for first in range(0, count, width)
        ]
    raise ValueError(f"Unsupported kperp profile: {profile}")


def build_coarse_groups(
    output_band_ids: np.ndarray,
    *,
    kperp_count: int,
    kpar_count: int,
    kperp_profile: str,
    kpar_group_size: int,
) -> list[CoarseGroup]:
    """Group existing geometric-window output rows into rectangles."""
    output_ids = np.asarray(output_band_ids, dtype=np.int64).reshape(-1)
    if (
        output_ids.size == 0
        or np.unique(output_ids).size != output_ids.size
        or np.any(output_ids < 0)
        or np.any(output_ids >= int(kperp_count) * int(kpar_count))
    ):
        raise ValueError("output_band_ids must be unique and in range")
    radial_width = int(kpar_group_size)
    if radial_width < 1:
        raise ValueError("kpar_group_size must be positive")
    output_kperp = output_ids // int(kpar_count)
    output_kpar = output_ids % int(kpar_count)
    groups: list[CoarseGroup] = []
    for kperp_first, kperp_stop in kperp_intervals(
        kperp_profile, int(kperp_count)
    ):
        for kpar_first in range(0, int(kpar_count), radial_width):
            kpar_stop = min(kpar_first + radial_width, int(kpar_count))
            selected = np.flatnonzero(
                (output_kperp >= kperp_first)
                & (output_kperp < kperp_stop)
                & (output_kpar >= kpar_first)
                & (output_kpar < kpar_stop)
            )
            if selected.size:
                groups.append(
                    CoarseGroup(
                        output_positions=selected,
                        kperp_first=int(kperp_first),
                        kperp_stop=int(kperp_stop),
                        kpar_first=int(kpar_first),
                        kpar_stop=int(kpar_stop),
                    )
                )
    assigned = np.concatenate([group.output_positions for group in groups])
    if (
        assigned.size != output_ids.size
        or np.unique(assigned).size != output_ids.size
    ):
        raise ValueError("Coarse groups must partition every output exactly once")
    return groups


def matching_mode_counts(
    output_band_ids: np.ndarray,
    *,
    kpar_count: int,
    source_kperp_indices: np.ndarray,
    source_kpar_indices: np.ndarray,
    source_mode_counts: np.ndarray,
) -> np.ndarray:
    """Look up intrinsic FFT-mode counts for nominal output cells."""
    output_ids = np.asarray(output_band_ids, dtype=np.int64).reshape(-1)
    output_kperp = output_ids // int(kpar_count)
    output_kpar = output_ids % int(kpar_count)
    source_kperp = np.asarray(source_kperp_indices, dtype=np.int64).reshape(-1)
    source_kpar = np.asarray(source_kpar_indices, dtype=np.int64).reshape(-1)
    mode_counts = np.asarray(source_mode_counts, dtype=np.float64).reshape(-1)
    if not (
        source_kperp.size == source_kpar.size == mode_counts.size
        and np.all(np.isfinite(mode_counts))
        and np.all(mode_counts > 0.0)
    ):
        raise ValueError("Invalid source-band geometry or mode counts")
    lookup = {
        (int(kp), int(ka)): float(count)
        for kp, ka, count in zip(source_kperp, source_kpar, mode_counts)
    }
    result = np.asarray(
        [
            lookup.get((int(kp), int(ka)), math.nan)
            for kp, ka in zip(output_kperp, output_kpar)
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("A nominal output cell has no matching source band")
    return result


def mode_weighted_transform(
    response: np.ndarray,
    groups: list[CoarseGroup],
    output_mode_counts: np.ndarray,
) -> np.ndarray:
    """Build a raw-q transform whose normalized output is mode weighted."""
    return normalized_weighted_transform(response, groups, output_mode_counts)


def normalized_weighted_transform(
    response: np.ndarray,
    groups: list[CoarseGroup],
    output_weights: np.ndarray,
) -> np.ndarray:
    """Build a transform that averages normalized estimates with fixed weights."""
    matrix = np.asarray(response, dtype=np.float64)
    weights = np.asarray(output_weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != weights.size:
        raise ValueError("response rows and output_weights differ")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("output_weights must be finite and positive")
    row_sum = np.sum(matrix, axis=1)
    if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
        raise ValueError("response rows must have finite positive sums")
    transform = np.zeros((len(groups), matrix.shape[0]), dtype=np.float64)
    for group_index, group in enumerate(groups):
        positions = np.asarray(group.output_positions, dtype=np.int64)
        transform[group_index, positions] = (
            weights[positions] / row_sum[positions]
        )
    return transform


def transform_q(values: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply an output transform while preserving arbitrary leading batches."""
    q_values = np.asarray(values, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if q_values.shape[-1] != matrix.shape[1]:
        raise ValueError("q output axis and transform columns differ")
    return q_values @ matrix.T


def _weighted_l2(
    estimate: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    first = np.asarray(estimate, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    weight = np.broadcast_to(np.asarray(weights, dtype=np.float64), first.shape)
    return math.sqrt(
        float(np.sum(weight * np.square(first - second)))
        / max(float(np.sum(weight * np.square(second))), 1e-300)
    )


def _realization_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    first = np.asarray(estimate, dtype=np.float64).reshape(-1)
    second = np.asarray(target, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    relative = np.abs(first - second) / np.maximum(np.abs(second), 1e-300)
    return {
        "relative_l2": _weighted_l2(first, second, weight),
        "integrated_power_ratio": float(
            np.sum(weight * first) / np.sum(weight * second)
        ),
        "median_absolute_error_fraction": float(np.median(relative)),
        "p90_absolute_error_fraction": float(np.percentile(relative, 90.0)),
        "maximum_absolute_error_fraction": float(np.max(relative)),
        "passing_10pct_count": int(np.count_nonzero(relative < 0.1)),
        "passing_20pct_count": int(np.count_nonzero(relative < 0.2)),
    }


def _effective_rank(matrix: np.ndarray) -> dict[str, float | int]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    power = np.square(singular)
    participation = float(
        np.square(np.sum(power)) / max(float(np.sum(np.square(power))), 1e-300)
    )
    cutoff = 1e-3 * float(singular[0]) if singular.size else math.inf
    return {
        "numerical_rank_rcond_1e3": int(np.count_nonzero(singular >= cutoff)),
        "participation_rank": participation,
    }


def evaluate_coarse_profile(
    *,
    response: np.ndarray,
    source_power: np.ndarray,
    source_in_geometric_window: np.ndarray,
    source_kperp_indices: np.ndarray,
    source_kpar_values: np.ndarray,
    output_kpar_values: np.ndarray,
    output_mode_counts: np.ndarray,
    groups: list[CoarseGroup],
    restricted_q: np.ndarray,
    heldout_q: np.ndarray,
    heldout_total_q: np.ndarray,
    bank_foreground_q: np.ndarray,
    bank_eor_q: np.ndarray,
    bank_total_q: np.ndarray,
    physical_shift_q: np.ndarray | None = None,
    physical_shift_total_q: np.ndarray | None = None,
    aggregation_weighting: str = "mode_count",
    minimum_relative_response: float = 0.1,
    minimum_window_fraction: float = 0.95,
    minimum_group_mode_count: float = 0.0,
    minimum_nominal_window_fraction: float = 0.0,
    maximum_window_effective_width: float = math.inf,
    minimum_kperp_index: int = 0,
    maximum_kperp_index_exclusive: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate a truth-blind coarse grouping against held-out products."""
    fine_response = np.asarray(response, dtype=np.float64)
    power = np.asarray(source_power, dtype=np.float64).reshape(-1)
    geometric = np.asarray(source_in_geometric_window, dtype=bool).reshape(-1)
    source_kperp = np.asarray(source_kperp_indices, dtype=np.int64).reshape(-1)
    source_kpar = np.asarray(source_kpar_values, dtype=np.float64).reshape(-1)
    output_kpar = np.asarray(output_kpar_values, dtype=np.float64).reshape(-1)
    modes = np.asarray(output_mode_counts, dtype=np.float64).reshape(-1)
    if (
        fine_response.ndim != 2
        or fine_response.shape[1] != power.size
        or geometric.size != power.size
        or source_kperp.size != power.size
        or source_kpar.size != power.size
        or output_kpar.size
        < max(group.kpar_stop for group in groups)
        or fine_response.shape[0] != modes.size
    ):
        raise ValueError("Response, source, and output arrays are inconsistent")

    fine_row_sum = np.sum(fine_response, axis=1)
    fine_relative_response = fine_row_sum / np.max(fine_row_sum)
    weighting = str(aggregation_weighting)
    if weighting == "mode_count":
        fine_aggregation_weight = modes
    elif weighting == "response":
        fine_aggregation_weight = fine_relative_response
    else:
        raise ValueError(f"Unsupported aggregation weighting: {weighting}")
    transform = normalized_weighted_transform(
        fine_response, groups, fine_aggregation_weight
    )
    coarse_response = transform @ fine_response
    row_sum = np.sum(coarse_response, axis=1)
    window = coarse_response / row_sum[:, None]
    target = window @ power
    group_modes = np.asarray(
        [np.sum(modes[group.output_positions]) for group in groups],
        dtype=np.float64,
    )
    group_metric_weights = np.asarray(
        [
            np.sum(fine_aggregation_weight[group.output_positions])
            for group in groups
        ],
        dtype=np.float64,
    )
    minimum_input_response = np.asarray(
        [
            np.min(fine_relative_response[group.output_positions])
            for group in groups
        ],
        dtype=np.float64,
    )
    window_fraction = np.sum(window[:, geometric], axis=1)
    square_sum = np.sum(np.square(window), axis=1)
    effective_width = np.divide(
        1.0,
        square_sum,
        out=np.full(row_sum.shape, np.inf),
        where=square_sum > 0.0,
    )
    nominal_window_fraction = np.empty(len(groups), dtype=np.float64)
    group_kperp_first = np.asarray(
        [group.kperp_first for group in groups], dtype=np.int64
    )
    group_kperp_stop = np.asarray(
        [group.kperp_stop for group in groups], dtype=np.int64
    )
    for group_index, group in enumerate(groups):
        radial_match = np.any(
            np.isclose(
                source_kpar[:, None],
                output_kpar[group.kpar_first : group.kpar_stop][None, :],
                rtol=1e-10,
                atol=1e-12,
            ),
            axis=1,
        )
        nominal_source = (
            (source_kperp >= int(group.kperp_first))
            & (source_kperp < int(group.kperp_stop))
            & radial_match
        )
        nominal_window_fraction[group_index] = float(
            np.sum(window[group_index, nominal_source])
        )
    maximum_kperp = (
        max(group.kperp_stop for group in groups)
        if maximum_kperp_index_exclusive is None
        else int(maximum_kperp_index_exclusive)
    )
    selected = (
        (minimum_input_response >= float(minimum_relative_response))
        & (window_fraction >= float(minimum_window_fraction))
        & (group_modes >= float(minimum_group_mode_count))
        & (
            nominal_window_fraction
            >= float(minimum_nominal_window_fraction)
        )
        & (effective_width <= float(maximum_window_effective_width))
        & (group_kperp_first >= int(minimum_kperp_index))
        & (group_kperp_stop <= maximum_kperp)
    )
    if not np.any(selected):
        raise ValueError("No coarse group passes the response-only selection")

    def estimate(values: np.ndarray) -> np.ndarray:
        coarse_q = transform_q(values, transform)
        return coarse_q / row_sum

    restricted_estimate = estimate(restricted_q)
    heldout_estimate = estimate(heldout_q)
    heldout_total_estimate = estimate(heldout_total_q)
    foreground_estimate = estimate(bank_foreground_q)
    bank_eor_estimate = estimate(bank_eor_q)
    bank_total_estimate = estimate(bank_total_q)
    physical_shift_estimate = (
        None if physical_shift_q is None else estimate(physical_shift_q)
    )
    physical_shift_total_estimate = (
        None
        if physical_shift_total_q is None
        else estimate(physical_shift_total_q)
    )
    if (physical_shift_estimate is None) != (
        physical_shift_total_estimate is None
    ):
        raise ValueError("Pure and total physical-shift products must coexist")
    selected_target = target[selected]
    selected_metric_weights = group_metric_weights[selected]

    heldout_rows = [
        _realization_metrics(
            row[selected], selected_target, selected_metric_weights
        )
        for row in heldout_estimate
    ]
    heldout_total_rows = [
        _realization_metrics(
            row[selected], selected_target, selected_metric_weights
        )
        for row in heldout_total_estimate
    ]
    bank_eor_metrics = _realization_metrics(
        bank_eor_estimate[selected], selected_target, selected_metric_weights
    )
    bank_total_metrics = _realization_metrics(
        bank_total_estimate[selected], selected_target, selected_metric_weights
    )
    restricted_metrics = _realization_metrics(
        restricted_estimate[selected], selected_target, selected_metric_weights
    )
    heldout_worst_error = np.max(
        np.abs(heldout_total_estimate[:, selected] - selected_target[None, :])
        / np.maximum(np.abs(selected_target[None, :]), 1e-300),
        axis=0,
    )
    bank_error = (
        np.abs(bank_total_estimate[selected] - selected_target)
        / np.maximum(np.abs(selected_target), 1e-300)
    )
    strict_without_shifts = (bank_error < 0.2) & (heldout_worst_error < 0.2)
    physical_shift_pure_worst_error = np.full(
        selected_target.shape, np.nan, dtype=np.float64
    )
    physical_shift_total_worst_error = np.full(
        selected_target.shape, np.nan, dtype=np.float64
    )
    physical_shift_pure_rows: list[dict[str, float | int]] = []
    physical_shift_total_rows: list[dict[str, float | int]] = []
    if physical_shift_total_estimate is not None:
        physical_shift_pure_worst_error = np.max(
            np.abs(
                physical_shift_estimate[:, selected]
                - selected_target[None, :]
            )
            / np.maximum(np.abs(selected_target[None, :]), 1e-300),
            axis=0,
        )
        physical_shift_total_worst_error = np.max(
            np.abs(
                physical_shift_total_estimate[:, selected]
                - selected_target[None, :]
            )
            / np.maximum(np.abs(selected_target[None, :]), 1e-300),
            axis=0,
        )
        physical_shift_pure_rows = [
            _realization_metrics(
                row[selected], selected_target, selected_metric_weights
            )
            for row in physical_shift_estimate
        ]
        physical_shift_total_rows = [
            _realization_metrics(
                row[selected], selected_target, selected_metric_weights
            )
            for row in physical_shift_total_estimate
        ]
        strict = (
            strict_without_shifts
            & (physical_shift_pure_worst_error < 0.2)
            & (physical_shift_total_worst_error < 0.2)
        )
    else:
        strict = strict_without_shifts
    foreground_effect = (
        np.abs(bank_total_estimate[selected] - bank_eor_estimate[selected])
        / np.maximum(np.abs(selected_target), 1e-300)
    )
    heldout_mean_metrics = _realization_metrics(
        np.mean(heldout_estimate[:, selected], axis=0),
        selected_target,
        selected_metric_weights,
    )
    heldout_total_error = (
        heldout_total_estimate[:, selected] - selected_target[None, :]
    )
    if heldout_total_error.shape[0] > 1:
        heldout_total_error_covariance = np.atleast_2d(
            np.cov(heldout_total_error, rowvar=False, ddof=1)
        )
        heldout_total_fractional_std = np.sqrt(
            np.maximum(np.diag(heldout_total_error_covariance), 0.0)
        ) / np.maximum(np.abs(selected_target), 1e-300)
    else:
        heldout_total_error_covariance = np.full(
            (selected_target.size, selected_target.size), np.nan
        )
        heldout_total_fractional_std = np.full(
            selected_target.shape, np.nan
        )
    summary: dict[str, Any] = {
        "aggregation_weighting": weighting,
        "group_count": int(len(groups)),
        "selected_group_count": int(np.count_nonzero(selected)),
        "strict_group_count": int(np.count_nonzero(strict)),
        "strict_without_physical_shifts_group_count": int(
            np.count_nonzero(strict_without_shifts)
        ),
        "strict_fraction_of_selected": float(np.mean(strict)),
        "selected_nominal_cell_count": int(
            sum(
                groups[index].output_positions.size
                for index in np.flatnonzero(selected)
            )
        ),
        "minimum_relative_response": float(minimum_relative_response),
        "minimum_window_fraction": float(minimum_window_fraction),
        "minimum_group_mode_count": float(minimum_group_mode_count),
        "minimum_nominal_window_fraction": float(
            minimum_nominal_window_fraction
        ),
        "maximum_window_effective_width": float(
            maximum_window_effective_width
        ),
        "minimum_kperp_index": int(minimum_kperp_index),
        "maximum_kperp_index_exclusive": int(maximum_kperp),
        "selected_window_fraction_minimum": float(
            np.min(window_fraction[selected])
        ),
        "selected_nominal_window_fraction_minimum": float(
            np.min(nominal_window_fraction[selected])
        ),
        "selected_window_effective_width_median": float(
            np.median(effective_width[selected])
        ),
        "selected_window_effective_width_maximum": float(
            np.max(effective_width[selected])
        ),
        "restricted_random_probe": restricted_metrics,
        "heldout_mean": heldout_mean_metrics,
        "heldout_worst_relative_l2": float(
            max(row["relative_l2"] for row in heldout_rows)
        ),
        "heldout_total_worst_relative_l2": float(
            max(row["relative_l2"] for row in heldout_total_rows)
        ),
        "heldout_total_integrated_ratio_minimum": float(
            min(row["integrated_power_ratio"] for row in heldout_total_rows)
        ),
        "heldout_total_integrated_ratio_maximum": float(
            max(row["integrated_power_ratio"] for row in heldout_total_rows)
        ),
        "heldout_total_error_covariance": {
            "used_for_selection": False,
            "realization_count": int(heldout_total_error.shape[0]),
            "fractional_std_median": float(
                np.nanmedian(heldout_total_fractional_std)
            ),
            "fractional_std_maximum": float(
                np.nanmax(heldout_total_fractional_std)
            ),
            "participation_rank": _effective_rank(heldout_total_error)[
                "participation_rank"
            ],
        },
        "bank_eor": bank_eor_metrics,
        "bank_total": bank_total_metrics,
        "foreground_effect_maximum_fraction": float(
            np.max(foreground_effect)
        ),
        "foreground_effect_median_fraction": float(
            np.median(foreground_effect)
        ),
        "response_window_rank": _effective_rank(window[selected]),
    }
    if physical_shift_total_rows:
        def summarize_physical_shifts(
            rows: list[dict[str, float | int]],
            worst_error: np.ndarray,
        ) -> dict[str, Any]:
            return {
                "realization_count": int(len(rows)),
                "worst_relative_l2": float(
                    max(row["relative_l2"] for row in rows)
                ),
                "integrated_ratio_minimum": float(
                    min(row["integrated_power_ratio"] for row in rows)
                ),
                "integrated_ratio_maximum": float(
                    max(row["integrated_power_ratio"] for row in rows)
                ),
                "maximum_per_group_error_fraction": float(
                    np.max(worst_error)
                ),
            }

        summary["physical_shifts_pure"] = summarize_physical_shifts(
            physical_shift_pure_rows, physical_shift_pure_worst_error
        )
        summary["physical_shifts_total"] = summarize_physical_shifts(
            physical_shift_total_rows, physical_shift_total_worst_error
        )
        summary["physical_shifts"] = {
            **summary["physical_shifts_total"],
            "deprecated_alias_for": "physical_shifts_total",
        }
    arrays = {
        "transform": transform,
        "response": coarse_response,
        "window": window,
        "target": target,
        "group_modes": group_modes,
        "group_kperp_first": group_kperp_first,
        "group_kperp_stop": group_kperp_stop,
        "group_metric_weights": group_metric_weights,
        "minimum_input_relative_response": minimum_input_response,
        "window_fraction": window_fraction,
        "nominal_window_fraction": nominal_window_fraction,
        "effective_width": effective_width,
        "selected": selected.astype(np.int8),
        "strict_selected": strict.astype(np.int8),
        "restricted_estimate": restricted_estimate,
        "heldout_estimate": heldout_estimate,
        "heldout_total_estimate": heldout_total_estimate,
        "foreground_estimate": foreground_estimate,
        "bank_eor_estimate": bank_eor_estimate,
        "bank_total_estimate": bank_total_estimate,
        "selected_bank_error_fraction": bank_error,
        "selected_heldout_worst_error_fraction": heldout_worst_error,
        "selected_foreground_effect_fraction": foreground_effect,
        "selected_physical_shift_pure_worst_error_fraction": (
            physical_shift_pure_worst_error
        ),
        "selected_physical_shift_total_worst_error_fraction": (
            physical_shift_total_worst_error
        ),
        "selected_physical_shift_worst_error_fraction": (
            physical_shift_total_worst_error
        ),
        "heldout_total_error_covariance": heldout_total_error_covariance,
        "heldout_total_fractional_std": heldout_total_fractional_std,
    }
    if physical_shift_estimate is not None:
        arrays["physical_shift_estimate"] = physical_shift_estimate
        arrays["physical_shift_total_estimate"] = (
            physical_shift_total_estimate
        )
    return summary, arrays
