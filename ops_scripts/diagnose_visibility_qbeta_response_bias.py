#!/usr/bin/env python3
"""Diagnose row-sampling and phase-probe bias in visibility Q_beta responses."""

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

from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_qbeta_coarse import (  # noqa: E402
    build_coarse_groups,
    evaluate_coarse_profile,
    matching_mode_counts,
    transform_q,
)


SUMMED_ARRAYS = (
    "calibration_samples",
    "validation_samples",
    "restricted_eor_q",
    "heldout_mixture_q",
    "heldout_total_mixture_q",
    "bank_foreground_q",
    "bank_eor_q",
    "bank_total_q",
)

INVARIANT_ARRAYS = (
    "source_band_ids",
    "source_band_kperp_indices",
    "source_band_kpar_indices",
    "source_band_mode_counts",
    "source_band_kpar_mpc_inv",
    "source_band_in_geometric_window",
    "restricted_eor_source_power",
    "output_band_ids",
    "support",
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="Partition evaluator directory containing result.npz; repeat.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Discover numerically ordered part_N/evaluate inputs here.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--surrogate-run-dir",
        type=Path,
        help="Optional run containing part_N/result.npz surrogate products.",
    )
    parser.add_argument(
        "--surrogate-label",
        help="Prefix of LABEL_actual_amplitude_random_phase_q.",
    )
    parser.add_argument(
        "--localized-block-count",
        type=int,
        action="append",
        help="Analyze LABEL_localized_Nblock_random_phase_q; repeat.",
    )
    parser.add_argument(
        "--spectral-coherence",
        action="store_true",
        help=(
            "Analyze LABEL_spectral_coherence_random_spatial_phase_q "
            "from the surrogate run."
        ),
    )
    parser.add_argument(
        "--frozen-combined-npz",
        type=Path,
        help="Freeze selected groups from this formal combined response.",
    )
    parser.add_argument(
        "--partition-counts",
        default="1,2,4,8,12,16,20",
        help="Comma-separated cumulative partition counts.",
    )
    parser.add_argument(
        "--kperp-profile",
        choices=(
            "fine",
            "low4",
            "low4_high2",
            "low4_mid3_high2",
            "low4_mid4_high2",
            "pair",
            "quad",
        ),
        default="quad",
    )
    parser.add_argument("--kpar-group-size", type=int, default=1)
    parser.add_argument(
        "--aggregation-weighting",
        choices=("response", "mode_count"),
        default="response",
    )
    parser.add_argument("--minimum-relative-response", type=float, default=0.1)
    parser.add_argument("--minimum-window-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-kperp-index", type=int, default=4)
    parser.add_argument(
        "--maximum-kperp-index-exclusive", type=int, default=None
    )
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


def _relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    denominator = float(np.linalg.norm(truth_array))
    if denominator <= 0.0:
        return math.nan
    return float(np.linalg.norm(estimate_array - truth_array) / denominator)


def _weighted_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    estimate_array = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth_array = np.asarray(truth, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    denominator = float(np.sum(weight_array * np.square(truth_array)))
    integrated_truth = float(np.sum(weight_array * truth_array))
    fractional_error = np.abs(estimate_array - truth_array) / np.maximum(
        np.abs(truth_array), 1e-300
    )
    return {
        "relative_l2": float(
            np.sqrt(
                np.sum(weight_array * np.square(estimate_array - truth_array))
                / denominator
            )
        ),
        "integrated_power_ratio": float(
            np.sum(weight_array * estimate_array) / integrated_truth
        ),
        "maximum_absolute_error_fraction": float(np.max(fractional_error)),
        "median_absolute_error_fraction": float(
            np.median(fractional_error)
        ),
        "passing_10pct_count": int(np.count_nonzero(fractional_error < 0.1)),
        "passing_20pct_count": int(np.count_nonzero(fractional_error < 0.2)),
    }


def _response_from_samples(
    samples: np.ndarray,
    *,
    source_count: int,
    output_band_ids: np.ndarray,
) -> np.ndarray:
    return np.mean(samples, axis=0).reshape(source_count, -1)[
        :, output_band_ids
    ].T


def _evaluate(
    *,
    response: np.ndarray,
    arrays: dict[str, np.ndarray],
    invariant: dict[str, np.ndarray],
    groups: list[Any],
    mode_counts: np.ndarray,
    output_kpar_values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return evaluate_coarse_profile(
        response=response,
        source_power=invariant["restricted_eor_source_power"],
        source_in_geometric_window=invariant[
            "source_band_in_geometric_window"
        ],
        source_kperp_indices=invariant["source_band_kperp_indices"],
        source_kpar_values=invariant["source_band_kpar_mpc_inv"],
        output_kpar_values=output_kpar_values,
        output_mode_counts=mode_counts,
        groups=groups,
        restricted_q=arrays["restricted_eor_q"],
        heldout_q=arrays["heldout_mixture_q"],
        heldout_total_q=arrays["heldout_total_mixture_q"],
        bank_foreground_q=arrays["bank_foreground_q"],
        bank_eor_q=arrays["bank_eor_q"],
        bank_total_q=arrays["bank_total_q"],
        aggregation_weighting=str(args.aggregation_weighting),
        minimum_relative_response=float(args.minimum_relative_response),
        minimum_window_fraction=float(args.minimum_window_fraction),
        minimum_kperp_index=int(args.minimum_kperp_index),
        maximum_kperp_index_exclusive=args.maximum_kperp_index_exclusive,
    )


def _frozen_metrics(
    *,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    frozen_selected: np.ndarray,
    frozen_weights: np.ndarray,
    surrogate_q: np.ndarray | None = None,
    spectral_coherence_q: np.ndarray | None = None,
    localized_q: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(frozen_selected, dtype=bool)
    target = arrays["target"][selected]
    weights = np.asarray(frozen_weights, dtype=np.float64)
    heldout = arrays["heldout_estimate"][:, selected]
    heldout_total = arrays["heldout_total_estimate"][:, selected]
    current_selected = np.asarray(arrays["selected"], dtype=bool)
    heldout_rows = [
        _weighted_metrics(row, target, weights) for row in heldout
    ]
    heldout_total_rows = [
        _weighted_metrics(row, target, weights) for row in heldout_total
    ]
    result = {
        "response_selected_group_count": int(
            summary["selected_group_count"]
        ),
        "response_selection_overlap_with_frozen_count": int(
            np.count_nonzero(current_selected & selected)
        ),
        "restricted_random_probe": _weighted_metrics(
            arrays["restricted_estimate"][selected], target, weights
        ),
        "bank_eor": _weighted_metrics(
            arrays["bank_eor_estimate"][selected], target, weights
        ),
        "bank_total": _weighted_metrics(
            arrays["bank_total_estimate"][selected], target, weights
        ),
        "heldout_mean": _weighted_metrics(
            np.mean(heldout, axis=0), target, weights
        ),
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
    }
    if surrogate_q is not None:
        transform = np.asarray(arrays["transform"], dtype=np.float64)
        coarse_response = np.asarray(arrays["response"], dtype=np.float64)
        surrogate_estimate = transform_q(surrogate_q, transform) / np.sum(
            coarse_response, axis=1
        )[None, :]
        surrogate_rows = [
            _weighted_metrics(row[selected], target, weights)
            for row in surrogate_estimate
        ]
        surrogate_ratios = np.asarray(
            [row["integrated_power_ratio"] for row in surrogate_rows],
            dtype=np.float64,
        )
        surrogate_ratio_std = float(
            np.std(surrogate_ratios, ddof=1)
            if surrogate_ratios.size > 1
            else math.nan
        )
        actual_ratio = float(
            result["restricted_random_probe"]["integrated_power_ratio"]
        )
        result["actual_amplitude_random_phase"] = {
            "realization_count": int(surrogate_estimate.shape[0]),
            "mean": _weighted_metrics(
                np.mean(surrogate_estimate[:, selected], axis=0),
                target,
                weights,
            ),
            "integrated_ratios": surrogate_ratios,
            "integrated_ratio_mean": float(np.mean(surrogate_ratios)),
            "integrated_ratio_std": surrogate_ratio_std,
            "integrated_ratio_minimum": float(
                np.min(surrogate_ratios)
            ),
            "integrated_ratio_maximum": float(
                np.max(surrogate_ratios)
            ),
            "worst_relative_l2": float(
                max(row["relative_l2"] for row in surrogate_rows)
            ),
            "actual_restricted_ratio_minus_surrogate_mean": float(
                actual_ratio - np.mean(surrogate_ratios)
            ),
            "diagonal_in_band_ratio_bias": float(
                np.mean(surrogate_ratios) - 1.0
            ),
            "coherent_phase_ratio_contribution": float(
                actual_ratio - np.mean(surrogate_ratios)
            ),
            "actual_restricted_ratio_z_score": float(
                (actual_ratio - np.mean(surrogate_ratios))
                / surrogate_ratio_std
            ),
        }
    if spectral_coherence_q is not None:
        transform = np.asarray(arrays["transform"], dtype=np.float64)
        coarse_response = np.asarray(arrays["response"], dtype=np.float64)
        estimates = transform_q(
            spectral_coherence_q, transform
        ) / np.sum(coarse_response, axis=1)[None, :]
        rows = [
            _weighted_metrics(row[selected], target, weights)
            for row in estimates
        ]
        ratios = np.asarray(
            [row["integrated_power_ratio"] for row in rows],
            dtype=np.float64,
        )
        ratio_std = float(
            np.std(ratios, ddof=1) if ratios.size > 1 else math.nan
        )
        actual_ratio = float(
            result["restricted_random_probe"]["integrated_power_ratio"]
        )
        result["spectral_coherence_random_spatial_phase"] = {
            "realization_count": int(estimates.shape[0]),
            "mean": _weighted_metrics(
                np.mean(estimates[:, selected], axis=0),
                target,
                weights,
            ),
            "integrated_ratios": ratios,
            "integrated_ratio_mean": float(np.mean(ratios)),
            "integrated_ratio_std": ratio_std,
            "integrated_ratio_minimum": float(np.min(ratios)),
            "integrated_ratio_maximum": float(np.max(ratios)),
            "worst_relative_l2": float(
                max(row["relative_l2"] for row in rows)
            ),
            "actual_restricted_ratio_minus_surrogate_mean": float(
                actual_ratio - np.mean(ratios)
            ),
            "actual_restricted_ratio_z_score": float(
                (actual_ratio - np.mean(ratios)) / ratio_std
            ),
        }
    if localized_q:
        transform = np.asarray(arrays["transform"], dtype=np.float64)
        coarse_response = np.asarray(arrays["response"], dtype=np.float64)
        row_sum = np.sum(coarse_response, axis=1)
        actual_estimate = np.asarray(
            arrays["restricted_estimate"], dtype=np.float64
        )[selected]
        global_target = np.asarray(arrays["target"], dtype=np.float64)[
            selected
        ]
        localized_rows: dict[str, Any] = {}
        for block_count, q_values in sorted(localized_q.items()):
            estimates = transform_q(q_values, transform) / row_sum[None, :]
            predicted = np.mean(estimates[:, selected], axis=0)
            realization_ratios = np.asarray(
                [
                    _weighted_metrics(
                        row[selected], predicted, weights
                    )["integrated_power_ratio"]
                    for row in estimates
                ],
                dtype=np.float64,
            )
            localized_rows[str(block_count)] = {
                "realization_count": int(estimates.shape[0]),
                "predicted_vs_global_target": _weighted_metrics(
                    predicted, global_target, weights
                ),
                "actual_vs_localized_prediction": _weighted_metrics(
                    actual_estimate, predicted, weights
                ),
                "prediction_realization_ratio_std": float(
                    np.std(realization_ratios, ddof=1)
                    if realization_ratios.size > 1
                    else math.nan
                ),
            }
        result["localized_block_random_phase"] = localized_rows
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    input_dirs = list(args.input_dir or [])
    if args.run_dir is not None:
        if input_dirs:
            raise ValueError("--run-dir and --input-dir are mutually exclusive")
        discovered: list[tuple[int, Path]] = []
        for path in args.run_dir.glob("part_*/evaluate"):
            try:
                partition_index = int(path.parent.name.removeprefix("part_"))
            except ValueError:
                continue
            if (path / "result.npz").is_file():
                discovered.append((partition_index, path))
        input_dirs = [
            path for _, path in sorted(discovered, key=lambda item: item[0])
        ]
    if not input_dirs:
        raise ValueError("Supply --run-dir or at least one --input-dir")
    if (args.surrogate_run_dir is None) != (
        args.surrogate_label is None
    ):
        raise ValueError(
            "--surrogate-run-dir and --surrogate-label must be supplied together"
        )
    surrogate_dirs: list[Path] = []
    surrogate_key: str | None = None
    localized_block_counts = sorted(
        set(int(value) for value in (args.localized_block_count or []))
    )
    if any(value < 1 for value in localized_block_counts):
        raise ValueError("localized-block-count values must be positive")
    if localized_block_counts and args.surrogate_run_dir is None:
        raise ValueError(
            "Localized products require --surrogate-run-dir and label"
        )
    if args.spectral_coherence and args.surrogate_run_dir is None:
        raise ValueError(
            "Spectral-coherence products require --surrogate-run-dir "
            "and label"
        )
    if args.surrogate_run_dir is not None:
        discovered_surrogates: list[tuple[int, Path]] = []
        for path in args.surrogate_run_dir.glob("part_*"):
            try:
                partition_index = int(path.name.removeprefix("part_"))
            except ValueError:
                continue
            if (path / "result.npz").is_file():
                discovered_surrogates.append((partition_index, path))
        surrogate_dirs = [
            path
            for _, path in sorted(
                discovered_surrogates, key=lambda item: item[0]
            )
        ]
        if [
            index
            for index, _ in sorted(
                discovered_surrogates, key=lambda item: item[0]
            )
        ] != list(range(len(surrogate_dirs))):
            raise ValueError(
                "Surrogate partition indices must be contiguous from zero"
            )
        surrogate_key = (
            f"{args.surrogate_label}_actual_amplitude_random_phase_q"
        )
    requested_counts = sorted(
        {
            int(item)
            for item in str(args.partition_counts).split(",")
            if item.strip()
        }
    )
    if not requested_counts or requested_counts[0] < 1:
        raise ValueError("Partition counts must be positive")
    if requested_counts[-1] > len(input_dirs):
        raise ValueError(
            "Largest requested partition count exceeds the supplied inputs"
        )
    if surrogate_dirs and requested_counts[-1] > len(surrogate_dirs):
        raise ValueError(
            "Largest requested count exceeds available surrogate partitions"
        )

    invariant: dict[str, np.ndarray] = {}
    running: dict[str, np.ndarray] = {}
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    for partition_index, directory in enumerate(
        input_dirs[: requested_counts[-1]], start=1
    ):
        current_selected_rows: np.ndarray
        with np.load(directory / "result.npz", allow_pickle=False) as archive:
            missing = sorted(
                (set(SUMMED_ARRAYS) | set(INVARIANT_ARRAYS))
                - set(archive.files)
            )
            if missing:
                raise ValueError(
                    f"{directory} lacks: {', '.join(missing)}"
                )
            if not invariant:
                invariant = {
                    name: np.asarray(archive[name]) for name in INVARIANT_ARRAYS
                }
            else:
                for name in INVARIANT_ARRAYS:
                    if not np.array_equal(archive[name], invariant[name]):
                        raise ValueError(
                            f"Partition invariant differs for {name}: "
                            f"{directory}"
                        )
            for name in SUMMED_ARRAYS:
                value = np.asarray(archive[name], dtype=np.float64)
                if name not in running:
                    running[name] = value.copy()
                else:
                    running[name] += value
            current_selected_rows = np.asarray(
                archive["selected_bank_rows"], dtype=np.int64
            )
        if surrogate_dirs:
            with np.load(
                surrogate_dirs[partition_index - 1] / "result.npz",
                allow_pickle=False,
            ) as surrogate_archive:
                assert surrogate_key is not None
                if surrogate_key not in surrogate_archive:
                    raise ValueError(
                        f"Surrogate archive lacks {surrogate_key}"
                    )
                if not np.array_equal(
                    surrogate_archive["selected_bank_rows"],
                    current_selected_rows,
                ):
                    raise ValueError("Base and surrogate selected rows differ")
                surrogate_value = np.asarray(
                    surrogate_archive[surrogate_key], dtype=np.float64
                )
                localized_values: dict[int, np.ndarray] = {}
                spectral_coherence_value: np.ndarray | None = None
                if args.spectral_coherence:
                    key = (
                        f"{args.surrogate_label}_spectral_coherence_"
                        "random_spatial_phase_q"
                    )
                    if key not in surrogate_archive:
                        raise ValueError(f"Surrogate archive lacks {key}")
                    spectral_coherence_value = np.asarray(
                        surrogate_archive[key], dtype=np.float64
                    )
                for block_count in localized_block_counts:
                    key = (
                        f"{args.surrogate_label}_localized_"
                        f"{block_count}block_random_phase_q"
                    )
                    if key not in surrogate_archive:
                        raise ValueError(f"Surrogate archive lacks {key}")
                    localized_values[block_count] = np.asarray(
                        surrogate_archive[key], dtype=np.float64
                    )
            if "actual_amplitude_random_phase_q" not in running:
                running["actual_amplitude_random_phase_q"] = (
                    surrogate_value.copy()
                )
            else:
                running["actual_amplitude_random_phase_q"] += surrogate_value
            if spectral_coherence_value is not None:
                key = "spectral_coherence_random_spatial_phase_q"
                if key not in running:
                    running[key] = spectral_coherence_value.copy()
                else:
                    running[key] += spectral_coherence_value
            for block_count, localized_value in localized_values.items():
                key = f"localized_{block_count}block_random_phase_q"
                if key not in running:
                    running[key] = localized_value.copy()
                else:
                    running[key] += localized_value
        if partition_index in requested_counts:
            snapshots[partition_index] = {
                name: value / float(partition_index)
                for name, value in running.items()
            }

    support = np.asarray(invariant["support"])
    if support.ndim != 2:
        raise ValueError("support must be two-dimensional")
    kperp_count, kpar_count = support.shape
    output_ids = np.asarray(invariant["output_band_ids"], dtype=np.int64)
    source_count = int(invariant["source_band_ids"].size)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    output_kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    if output_kpar_values.shape != (kpar_count,):
        raise ValueError("Config kpar layout differs from result support")
    mode_counts = matching_mode_counts(
        output_ids,
        kpar_count=kpar_count,
        source_kperp_indices=invariant["source_band_kperp_indices"],
        source_kpar_indices=invariant["source_band_kpar_indices"],
        source_mode_counts=invariant["source_band_mode_counts"],
    )
    groups = build_coarse_groups(
        output_ids,
        kperp_count=kperp_count,
        kpar_count=kpar_count,
        kperp_profile=str(args.kperp_profile),
        kpar_group_size=int(args.kpar_group_size),
    )

    final_count = requested_counts[-1]
    final_snapshot = snapshots[final_count]
    final_calibration = _response_from_samples(
        final_snapshot["calibration_samples"],
        source_count=source_count,
        output_band_ids=output_ids,
    )
    frozen_response = final_calibration
    if args.frozen_combined_npz is not None:
        with np.load(
            args.frozen_combined_npz, allow_pickle=False
        ) as frozen_archive:
            if not np.array_equal(
                frozen_archive["output_band_ids"], output_ids
            ):
                raise ValueError(
                    "Frozen combined and partition output bands differ"
                )
            frozen_response = np.asarray(
                frozen_archive["calibration_response"], dtype=np.float64
            )
    baseline_summary, baseline_arrays = _evaluate(
        response=frozen_response,
        arrays=final_snapshot,
        invariant=invariant,
        groups=groups,
        mode_counts=mode_counts,
        output_kpar_values=output_kpar_values,
        args=args,
    )
    frozen_selected = np.asarray(baseline_arrays["selected"], dtype=bool)
    frozen_weights = np.asarray(
        baseline_arrays["group_metric_weights"][frozen_selected],
        dtype=np.float64,
    )

    count_rows: dict[str, Any] = {}
    for count in requested_counts:
        snapshot = snapshots[count]
        calibration = _response_from_samples(
            snapshot["calibration_samples"],
            source_count=source_count,
            output_band_ids=output_ids,
        )
        validation = _response_from_samples(
            snapshot["validation_samples"],
            source_count=source_count,
            output_band_ids=output_ids,
        )
        variants = {
            "calibration": calibration,
            "validation": validation,
            "calibration_validation_mean": 0.5
            * (calibration + validation),
        }
        variant_rows: dict[str, Any] = {}
        for name, response in variants.items():
            summary, evaluated = _evaluate(
                response=response,
                arrays=snapshot,
                invariant=invariant,
                groups=groups,
                mode_counts=mode_counts,
                output_kpar_values=output_kpar_values,
                args=args,
            )
            variant_rows[name] = {
                "response_relative_l2_from_final_calibration": _relative_l2(
                    response, final_calibration
                ),
                "frozen": _frozen_metrics(
                    summary=summary,
                    arrays=evaluated,
                    frozen_selected=frozen_selected,
                    frozen_weights=frozen_weights,
                    surrogate_q=snapshot.get(
                        "actual_amplitude_random_phase_q"
                    ),
                    spectral_coherence_q=snapshot.get(
                        "spectral_coherence_random_spatial_phase_q"
                    ),
                    localized_q={
                        block_count: snapshot[
                            f"localized_{block_count}block_random_phase_q"
                        ]
                        for block_count in localized_block_counts
                    },
                ),
            }
        count_rows[str(count)] = {
            "calibration_validation_response_relative_l2": _relative_l2(
                validation, calibration
            ),
            "variants": variant_rows,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "visibility_qbeta_response_bias_diagnostic",
        "schema_version": 1,
        "input_dirs": [str(path) for path in input_dirs],
        "config": str(args.config),
        "surrogate_run_dir": (
            None
            if args.surrogate_run_dir is None
            else str(args.surrogate_run_dir)
        ),
        "surrogate_label": args.surrogate_label,
        "localized_block_counts": localized_block_counts,
        "spectral_coherence": bool(args.spectral_coherence),
        "frozen_combined_npz": (
            None
            if args.frozen_combined_npz is None
            else str(args.frozen_combined_npz)
        ),
        "profile": {
            "kperp_profile": str(args.kperp_profile),
            "kpar_group_size": int(args.kpar_group_size),
            "aggregation_weighting": str(args.aggregation_weighting),
        },
        "selection": {
            "minimum_relative_response": float(
                args.minimum_relative_response
            ),
            "minimum_window_fraction": float(args.minimum_window_fraction),
            "minimum_kperp_index": int(args.minimum_kperp_index),
            "maximum_kperp_index_exclusive": (
                None
                if args.maximum_kperp_index_exclusive is None
                else int(args.maximum_kperp_index_exclusive)
            ),
            "frozen_from_partition_count": int(final_count),
            "frozen_group_count": int(np.count_nonzero(frozen_selected)),
        },
        "partition_counts": count_rows,
    }
    _atomic_json(args.out_dir / "summary.json", result)
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
