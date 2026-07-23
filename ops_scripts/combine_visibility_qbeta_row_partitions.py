#!/usr/bin/env python3
"""Combine disjoint-row visibility Q_beta calibrations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from visibility_qbeta import weighted_response_pseudoinverse  # noqa: E402


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--response-rcond", type=float, default=1e-4)
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _relative_l2(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    first = np.asarray(estimate, dtype=np.float64)
    second = np.asarray(truth, dtype=np.float64)
    if weights is None:
        weight = np.ones_like(first)
    else:
        weight = np.broadcast_to(np.asarray(weights, dtype=np.float64), first.shape)
    return math.sqrt(
        float(np.sum(weight * np.square(first - second)))
        / max(float(np.sum(weight * np.square(second))), 1e-300)
    )


def _operator_closure(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.complex128)
    truth = np.asarray(target, dtype=np.complex128)
    residual = prediction - truth
    denominator = max(float(np.sum(np.abs(truth) ** 2)), 1e-300)
    return {
        "relative_l2": math.sqrt(
            float(np.sum(np.abs(residual) ** 2)) / denominator
        ),
        "selected_visibility_count": int(truth.size),
    }


def _windowed_metrics(
    *,
    response: np.ndarray,
    observed_q: np.ndarray,
    source_power: np.ndarray,
    minimum_relative_response: float,
    target_source_positions: np.ndarray,
    minimum_target_window_fraction: float,
) -> dict[str, Any]:
    matrix = np.asarray(response, dtype=np.float64)
    q_values = np.asarray(observed_q, dtype=np.float64)
    if q_values.ndim == 1:
        q_values = q_values[None, :]
    row_sum = np.sum(matrix, axis=1)
    relative_response = row_sum / max(float(np.max(row_sum)), 1e-300)
    window = np.zeros_like(matrix)
    valid = row_sum > 0.0
    window[valid] = matrix[valid] / row_sum[valid, None]
    target_positions = np.asarray(
        target_source_positions, dtype=np.int64
    ).reshape(-1)
    if (
        target_positions.size == 0
        or np.any(target_positions < 0)
        or np.any(target_positions >= matrix.shape[1])
    ):
        raise ValueError("target_source_positions must select response columns")
    target_window_fraction = np.sum(
        window[:, target_positions], axis=1
    )
    selected = (
        (relative_response >= float(minimum_relative_response))
        & (
            target_window_fraction
            >= float(minimum_target_window_fraction)
        )
    )
    target = window @ np.asarray(source_power, dtype=np.float64)
    estimate = np.full(q_values.shape, np.nan, dtype=np.float64)
    estimate[:, valid] = q_values[:, valid] / row_sum[valid][None, :]
    weights = relative_response[selected]
    target_selected = target[selected]
    rows: list[dict[str, Any]] = []
    for row in estimate[:, selected]:
        if target_selected.size == 0:
            rows.append(
                {
                    "relative_l2": math.nan,
                    "integrated_power_ratio": math.nan,
                    "maximum_relative_window_error": math.nan,
                    "passing_window_count": 0,
                    "passing_window_fraction": math.nan,
                }
            )
            continue
        relative_error = np.abs(row - target_selected) / np.maximum(
            np.abs(target_selected), 1e-300
        )
        rows.append(
            {
                "relative_l2": math.sqrt(
                    float(
                        np.sum(
                            weights * np.square(row - target_selected)
                        )
                    )
                    / max(
                        float(
                            np.sum(
                                weights * np.square(target_selected)
                            )
                        ),
                        1e-300,
                    )
                ),
                "integrated_power_ratio": float(
                    np.sum(weights * row)
                    / np.sum(weights * target_selected)
                ),
                "maximum_relative_window_error": float(
                    np.max(relative_error)
                ),
                "passing_window_count": int(
                    np.count_nonzero(relative_error < 0.2)
                ),
                "passing_window_fraction": float(
                    np.mean(relative_error < 0.2)
                ),
            }
        )
    square_sum = np.sum(np.square(window), axis=1)
    return {
        "minimum_relative_response": float(minimum_relative_response),
        "minimum_target_window_fraction": float(
            minimum_target_window_fraction
        ),
        "selected_window_positions": np.flatnonzero(selected),
        "selected_window_count": int(np.count_nonzero(selected)),
        "relative_response": relative_response,
        "target_window_fraction": target_window_fraction,
        "window": window,
        "window_effective_width": np.divide(
            1.0,
            square_sum,
            out=np.full(row_sum.shape, np.inf),
            where=square_sum > 0.0,
        ),
        "target_windowed_power": target,
        "estimated_windowed_power": estimate,
        "realizations": rows,
    }


def _coarse_recovery_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    estimated = np.asarray(estimate, dtype=np.float64).reshape(10, 4)
    target = np.asarray(truth, dtype=np.float64).reshape(10, 4)
    mode_weights = np.asarray(weights, dtype=np.float64).reshape(10, 4)
    rows: list[dict[str, Any]] = []
    for transverse_group in (1, 2, 5, 10):
        for radial_group in (1, 2, 4):
            grouped_estimate: list[float] = []
            grouped_truth: list[float] = []
            grouped_weights: list[float] = []
            for first_perp in range(0, 10, transverse_group):
                for first_par in range(0, 4, radial_group):
                    block_weights = mode_weights[
                        first_perp : first_perp + transverse_group,
                        first_par : first_par + radial_group,
                    ]
                    total_weight = float(np.sum(block_weights))
                    grouped_estimate.append(
                        float(
                            np.sum(
                                block_weights
                                * estimated[
                                    first_perp : first_perp + transverse_group,
                                    first_par : first_par + radial_group,
                                ]
                            )
                            / total_weight
                        )
                    )
                    grouped_truth.append(
                        float(
                            np.sum(
                                block_weights
                                * target[
                                    first_perp : first_perp + transverse_group,
                                    first_par : first_par + radial_group,
                                ]
                            )
                            / total_weight
                        )
                    )
                    grouped_weights.append(total_weight)
            estimate_array = np.asarray(grouped_estimate)
            truth_array = np.asarray(grouped_truth)
            weight_array = np.asarray(grouped_weights)
            rows.append(
                {
                    "kperp_group_size": int(transverse_group),
                    "kpar_group_size": int(radial_group),
                    "bin_count": int(estimate_array.size),
                    "relative_l2": _relative_l2(
                        estimate_array, truth_array, weight_array
                    ),
                    "integrated_power_ratio": float(
                        np.sum(weight_array * estimate_array)
                        / np.sum(weight_array * truth_array)
                    ),
                    "maximum_relative_bin_error": float(
                        np.max(
                            np.abs(estimate_array - truth_array)
                            / np.maximum(np.abs(truth_array), 1e-300)
                        )
                    ),
                }
            )
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(args.input_dir) < 2:
        raise ValueError("At least two row partitions are required")
    metadata: list[dict[str, Any]] = []
    archives: list[dict[str, np.ndarray]] = []
    for directory in args.input_dir:
        metadata.append(
            json.loads((directory / "result.json").read_text(encoding="utf-8"))
        )
        with np.load(directory / "result.npz", allow_pickle=False) as archive:
            archives.append(
                {name: np.asarray(archive[name]) for name in archive.files}
            )
    reference_meta = metadata[0]
    reference = archives[0]
    for meta, archive in zip(metadata, archives):
        if "reporting_source_positions" not in archive:
            if meta["settings"].get("source_scope", "reporting") != "reporting":
                raise ValueError(
                    "Missing reporting_source_positions for a non-reporting source scope"
                )
            archive["reporting_source_positions"] = np.arange(
                archive["source_band_ids"].size, dtype=np.int64
            )
    invariant_arrays = (
        "source_band_ids",
        "reporting_source_positions",
        "source_band_kperp_indices",
        "source_band_kpar_indices",
        "source_band_mode_counts",
        "output_band_ids",
        "support",
    )
    if all("selected_row_kperp_indices" in archive for archive in archives):
        invariant_arrays += ("selected_row_kperp_indices",)
    for meta, archive in zip(metadata[1:], archives[1:]):
        for key in (
            "analysis_contract_sha256",
            "visibility_bank_sha256",
            "sky_cache_sha256",
        ):
            if meta[key] != reference_meta[key]:
                raise ValueError(f"Partition metadata differ in {key}")
        for key in invariant_arrays:
            if not np.array_equal(archive[key], reference[key]):
                raise ValueError(f"Partition arrays differ in {key}")
        for key in (
            "calibration_repeats",
            "validation_repeats",
            "mixture_repeats",
            "probe_seed",
            "rows_per_kperp_bin",
        ):
            if meta["settings"][key] != reference_meta["settings"][key]:
                raise ValueError(f"Partition settings differ in {key}")
        for key, default in (
            ("source_scope", "reporting"),
            ("row_scope", "all"),
        ):
            if meta["settings"].get(key, default) != reference_meta[
                "settings"
            ].get(key, default):
                raise ValueError(f"Partition settings differ in {key}")
    selected_rows = np.concatenate(
        [archive["selected_bank_rows"] for archive in archives]
    )
    if np.unique(selected_rows).size != selected_rows.size:
        raise ValueError("Input row partitions overlap")

    calibration_samples = np.mean(
        np.stack([archive["calibration_samples"] for archive in archives]),
        axis=0,
    )
    validation_samples = np.mean(
        np.stack([archive["validation_samples"] for archive in archives]),
        axis=0,
    )
    source_count = int(reference["source_band_ids"].size)
    output_ids = reference["output_band_ids"]
    calibration_response = np.mean(calibration_samples, axis=0).reshape(
        source_count, -1
    )[:, output_ids].T
    validation_response = np.mean(validation_samples, axis=0).reshape(
        source_count, -1
    )[:, output_ids].T
    response_pinv, response_svd = weighted_response_pseudoinverse(
        calibration_response, rcond=float(args.response_rcond)
    )
    validation_recovery = response_pinv @ validation_response
    identity = np.eye(source_count, dtype=np.float64)
    restricted_q = np.mean(
        np.stack([archive["restricted_eor_q"] for archive in archives]),
        axis=0,
    )
    restricted_power = reference["restricted_eor_source_power"]
    restricted_prediction = calibration_response @ restricted_power
    restricted_recovery = response_pinv @ restricted_q
    source_weights = reference["source_band_mode_counts"].astype(np.float64)
    heldout_mixture_q = np.mean(
        np.stack([archive["heldout_mixture_q"] for archive in archives]),
        axis=0,
    )
    bank_foreground_q = np.mean(
        np.stack([archive["bank_foreground_q"] for archive in archives]),
        axis=0,
    )
    bank_eor_q = np.mean(
        np.stack([archive["bank_eor_q"] for archive in archives]),
        axis=0,
    )
    bank_total_q = np.mean(
        np.stack([archive["bank_total_q"] for archive in archives]),
        axis=0,
    )
    minimum_qbeta_response = 0.1
    source_scope = str(
        reference_meta["settings"].get("source_scope", "reporting")
    )
    row_scope = str(reference_meta["settings"].get("row_scope", "all"))
    minimum_target_window_fraction = (
        0.0 if source_scope == "reporting" else 0.8
    )
    reporting_source_positions = reference[
        "reporting_source_positions"
    ]
    restricted_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=restricted_q,
        source_power=restricted_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    mixture_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=heldout_mixture_q,
        source_power=restricted_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    full_eor_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=bank_eor_q,
        source_power=restricted_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    total_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=bank_total_q,
        source_power=restricted_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    selected_window_positions = np.asarray(
        restricted_windowed["selected_window_positions"], dtype=np.int64
    )
    row_sum = np.sum(calibration_response, axis=1)
    foreground_windowed = (
        bank_foreground_q[selected_window_positions]
        / row_sum[selected_window_positions]
    )
    target_windowed = np.asarray(
        restricted_windowed["target_windowed_power"], dtype=np.float64
    )[selected_window_positions]
    foreground_weights = np.asarray(
        restricted_windowed["relative_response"], dtype=np.float64
    )[selected_window_positions]
    if selected_window_positions.size:
        foreground_to_target = float(
            np.sum(foreground_weights * np.abs(foreground_windowed))
            / np.sum(foreground_weights * np.abs(target_windowed))
        )
        median_window_effective_width = float(
            np.median(
                np.asarray(
                    restricted_windowed["window_effective_width"]
                )[selected_window_positions]
            )
        )
    else:
        foreground_to_target = math.nan
        median_window_effective_width = math.nan
    predicted_vis = np.concatenate(
        [archive["predicted_eor_vis"] for archive in archives], axis=1
    )
    target_vis = np.concatenate(
        [archive["target_eor_vis"] for archive in archives], axis=1
    )
    integrated_truth = float(np.sum(source_weights * restricted_power))
    coarse_metrics = (
        _coarse_recovery_metrics(
            restricted_recovery, restricted_power, source_weights
        )
        if source_count == 40
        else []
    )
    products = {
        "selected_bank_rows": selected_rows,
        **{key: reference[key] for key in invariant_arrays},
        "calibration_samples": calibration_samples,
        "validation_samples": validation_samples,
        "calibration_response": calibration_response,
        "validation_response": validation_response,
        "response_pseudoinverse": response_pinv,
        "validation_recovery": validation_recovery,
        "restricted_eor_source_power": restricted_power,
        "restricted_eor_q": restricted_q,
        "restricted_eor_q_prediction": restricted_prediction,
        "restricted_eor_recovery": restricted_recovery,
        "heldout_mixture_q": heldout_mixture_q,
        "bank_foreground_q": bank_foreground_q,
        "bank_eor_q": bank_eor_q,
        "bank_total_q": bank_total_q,
        "qbeta_window": restricted_windowed["window"],
        "qbeta_relative_response": restricted_windowed["relative_response"],
        "qbeta_target_window_fraction": restricted_windowed[
            "target_window_fraction"
        ],
        "qbeta_selected_window_positions": selected_window_positions,
        "restricted_eor_windowed_power": restricted_windowed[
            "estimated_windowed_power"
        ],
        "heldout_mixture_windowed_power": mixture_windowed[
            "estimated_windowed_power"
        ],
        "full_eor_windowed_power": full_eor_windowed[
            "estimated_windowed_power"
        ],
        "total_windowed_power": total_windowed["estimated_windowed_power"],
    }
    result = {
        "schema": "visibility_qbeta_row_partition_combination",
        "schema_version": 1,
        "analysis_contract_sha256": reference_meta["analysis_contract_sha256"],
        "visibility_bank_sha256": reference_meta["visibility_bank_sha256"],
        "sky_cache_sha256": reference_meta["sky_cache_sha256"],
        "input_directories": [str(path) for path in args.input_dir],
        "partition_count": int(len(archives)),
        "rows_per_kperp_bin_per_partition": int(
            reference_meta["settings"]["rows_per_kperp_bin"]
        ),
        "combined_rows_per_kperp_bin": int(
            len(archives)
            * reference_meta["settings"]["rows_per_kperp_bin"]
        ),
        "selected_row_count": int(selected_rows.size),
        "operator_closure": _operator_closure(predicted_vis, target_vis),
        "qbeta": {
            "source_scope": source_scope,
            "source_band_count": source_count,
            "supported_output_band_count": int(output_ids.size),
            "calibration_validation_response_relative_l2": _relative_l2(
                validation_response, calibration_response
            ),
            "validation_projected_response_relative_l2": _relative_l2(
                calibration_response
                @ response_pinv
                @ validation_response,
                validation_response,
            ),
            "validation_identity_relative_l2": _relative_l2(
                validation_recovery, identity
            ),
            "validation_diagonal_median": float(
                np.median(np.diag(validation_recovery))
            ),
            "validation_offdiagonal_absolute_l1_per_column": float(
                np.sum(np.abs(validation_recovery - identity)) / source_count
            ),
            "response_svd": response_svd,
        },
        "restricted_eor_closure": {
            "forward_q_relative_l2": _relative_l2(
                restricted_q, restricted_prediction
            ),
            "recovered_power_relative_l2": _relative_l2(
                restricted_recovery, restricted_power, source_weights
            ),
            "integrated_power_ratio": float(
                np.sum(source_weights * restricted_recovery)
                / integrated_truth
            ),
            "maximum_relative_band_error": float(
                np.max(
                    np.abs(restricted_recovery - restricted_power)
                    / np.maximum(np.abs(restricted_power), 1e-300)
                )
            ),
            "coarse_group_metrics": coarse_metrics,
        },
        "windowed_candidate": {
            "selection_status": (
                "response-only threshold frozen after the initial row pilot; "
                "heldout mixture realizations are independent of calibration"
            ),
            "minimum_relative_qbeta_response": minimum_qbeta_response,
            "minimum_target_window_fraction": (
                minimum_target_window_fraction
            ),
            "selected_window_count": int(
                restricted_windowed["selected_window_count"]
            ),
            "median_window_effective_width": (
                median_window_effective_width
            ),
            "restricted_eor": restricted_windowed["realizations"][0],
            "heldout_eor_like_mixtures": mixture_windowed["realizations"],
            "full_eor_including_context": full_eor_windowed["realizations"][0],
            "foreground_to_target_integrated_absolute_ratio": (
                foreground_to_target
            ),
            "total_fg_plus_eor": total_windowed["realizations"][0],
        },
        "limitations": [
            "no thermal noise",
            "fixed baseline-time rows; no uv gridding",
            (
                "rows cover every configured kperp bin"
                if row_scope == "all"
                else "rows are concentrated in the predeclared reporting kperp range"
            ),
            (
                "40-band target-subspace Q_beta calibration only"
                if source_scope == "reporting"
                else (
                    "all 512 in-range sky bands are included in Q_beta"
                    if source_scope == "all_in_range"
                    else "all 544 in-range sky bands including radial Nyquist are included in Q_beta"
                )
            ),
            (
                "foreground and out-of-region EoR nuisance bands are not yet included"
                if source_scope == "reporting"
                else (
                    "radial-Nyquist and out-of-kperp-range sky modes remain outside the response"
                    if source_scope == "all_in_range"
                    else "out-of-kperp-range sky modes remain outside the response"
                )
            ),
            (
                "coarse groups are diagnostics derived from predeclared fine bands"
                if source_count == 40
                else (
                    f"fully deconvolved {source_count}-band estimates are "
                    "underdetermined and diagnostic only"
                )
            ),
        ],
    }
    _atomic_npz(args.out_dir / "result.npz", products)
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
