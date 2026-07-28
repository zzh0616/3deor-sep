#!/usr/bin/env python3
"""Measure calibration-probe convergence at fixed visibility rows."""

from __future__ import annotations

import argparse
import itertools
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
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-count",
        type=int,
        action="append",
        help="Cumulative calibration repeat count; defaults to powers of two.",
    )
    parser.add_argument("--minimum-kperp-index", type=int, default=4)
    parser.add_argument("--minimum-relative-response", type=float, default=0.1)
    parser.add_argument("--minimum-window-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-subsets-per-count", type=int, default=128)
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


def cumulative_probe_counts(
    repeat_count: int,
    requested: list[int] | None,
) -> list[int]:
    """Return validated cumulative counts including the final repeat."""
    repeats = int(repeat_count)
    if repeats < 1:
        raise ValueError("repeat_count must be positive")
    if requested:
        counts = sorted(set(int(value) for value in requested))
    else:
        counts = []
        value = 1
        while value < repeats:
            counts.append(value)
            value *= 2
        counts.append(repeats)
    if any(value < 1 or value > repeats for value in counts):
        raise ValueError("probe counts must lie inside available repeats")
    if repeats not in counts:
        counts.append(repeats)
    return sorted(set(counts))


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.sum(np.square(first - second)))
    denominator = max(float(np.sum(np.square(second))), 1e-300)
    return math.sqrt(numerator / denominator)


def _fixed_selection_metrics(
    *,
    estimate: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    current_estimate = np.asarray(estimate, dtype=np.float64)[selected]
    current_target = np.asarray(target, dtype=np.float64)[selected]
    current_weights = np.asarray(weights, dtype=np.float64)[selected]
    return {
        "integrated_power_ratio": float(
            np.sum(current_weights * current_estimate)
            / np.sum(current_weights * current_target)
        ),
        "relative_l2": float(
            math.sqrt(
                np.sum(
                    current_weights
                    * np.square(current_estimate - current_target)
                )
                / max(
                    float(
                        np.sum(
                            current_weights * np.square(current_target)
                        )
                    ),
                    1e-300,
                )
            )
        ),
        "maximum_relative_error": float(
            np.max(
                np.abs(current_estimate - current_target)
                / np.maximum(np.abs(current_target), 1e-300)
            )
        ),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.result_npz, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    calibration_samples = np.asarray(
        data["calibration_samples"], dtype=np.float64
    )
    repeat_count = int(calibration_samples.shape[0])
    counts = cumulative_probe_counts(repeat_count, args.probe_count)
    source_count = int(data["source_band_ids"].size)
    output_ids = np.asarray(data["output_band_ids"], dtype=np.int64)
    support = np.asarray(data["support"], dtype=bool)
    kperp_count, kpar_count = support.shape
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    output_kpar = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    mode_counts = matching_mode_counts(
        output_ids,
        kpar_count=int(kpar_count),
        source_kperp_indices=data["source_band_kperp_indices"],
        source_kpar_indices=data["source_band_kpar_indices"],
        source_mode_counts=data["source_band_mode_counts"],
    )
    groups = build_coarse_groups(
        output_ids,
        kperp_count=int(kperp_count),
        kpar_count=int(kpar_count),
        kperp_profile="quad",
        kpar_group_size=1,
    )
    responses = {
        count: np.mean(calibration_samples[:count], axis=0).reshape(
            source_count, -1
        )[:, output_ids].T
        for count in counts
    }
    final_response = responses[counts[-1]]
    summaries: list[dict[str, Any]] = []
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for count in counts:
        summary, current = evaluate_coarse_profile(
            response=responses[count],
            source_power=data["restricted_eor_source_power"],
            source_in_geometric_window=data[
                "source_band_in_geometric_window"
            ],
            output_mode_counts=mode_counts,
            groups=groups,
            restricted_q=data["restricted_eor_q"],
            heldout_q=data["heldout_mixture_q"],
            heldout_total_q=data["heldout_total_mixture_q"],
            bank_foreground_q=data["bank_foreground_q"],
            bank_eor_q=data["bank_eor_q"],
            bank_total_q=data["bank_total_q"],
            source_kperp_indices=data["source_band_kperp_indices"],
            source_kpar_values=data["source_band_kpar_mpc_inv"],
            output_kpar_values=output_kpar,
            aggregation_weighting="response",
            minimum_relative_response=float(
                args.minimum_relative_response
            ),
            minimum_window_fraction=float(args.minimum_window_fraction),
            minimum_kperp_index=int(args.minimum_kperp_index),
        )
        arrays[count] = current
        summaries.append(
            {
                "probe_count": int(count),
                "response_relative_l2_to_final": _relative_l2(
                    responses[count], final_response
                ),
                "response_row_sum_relative_l2_to_final": _relative_l2(
                    np.sum(responses[count], axis=1),
                    np.sum(final_response, axis=1),
                ),
                "selected_group_count": int(
                    summary["selected_group_count"]
                ),
                "bank_total_native_selection": summary["bank_total"],
                "heldout_total_worst_relative_l2": float(
                    summary["heldout_total_worst_relative_l2"]
                ),
            }
        )
    final_selected = np.asarray(
        arrays[counts[-1]]["selected"], dtype=bool
    )
    final_weights = np.asarray(
        arrays[counts[-1]]["group_metric_weights"], dtype=np.float64
    )
    for summary, count in zip(summaries, counts, strict=True):
        current = arrays[count]
        summary["bank_total_final_selection"] = _fixed_selection_metrics(
            estimate=current["bank_total_estimate"],
            target=current["target"],
            selected=final_selected,
            weights=final_weights,
        )
        current_selected = np.asarray(current["selected"], dtype=bool)
        summary["selection_jaccard_with_final"] = float(
            np.count_nonzero(current_selected & final_selected)
            / np.count_nonzero(current_selected | final_selected)
        )
    maximum_subsets = int(args.maximum_subsets_per_count)
    if maximum_subsets < 1:
        raise ValueError("--maximum-subsets-per-count must be positive")
    subset_distributions: list[dict[str, Any]] = []
    repeat_indices = range(repeat_count)
    for count in counts:
        subsets = list(itertools.combinations(repeat_indices, count))
        if len(subsets) > maximum_subsets:
            positions = np.rint(
                np.linspace(
                    0,
                    len(subsets) - 1,
                    num=maximum_subsets,
                )
            ).astype(np.int64)
            subsets = [subsets[position] for position in positions]
        ratios = []
        relative_l2_values = []
        row_sum_l2_values = []
        selected_counts = []
        jaccards = []
        for subset in subsets:
            response = np.mean(
                calibration_samples[np.asarray(subset, dtype=np.int64)],
                axis=0,
            ).reshape(source_count, -1)[:, output_ids].T
            _, current = evaluate_coarse_profile(
                response=response,
                source_power=data["restricted_eor_source_power"],
                source_in_geometric_window=data[
                    "source_band_in_geometric_window"
                ],
                output_mode_counts=mode_counts,
                groups=groups,
                restricted_q=data["restricted_eor_q"],
                heldout_q=data["heldout_mixture_q"],
                heldout_total_q=data["heldout_total_mixture_q"],
                bank_foreground_q=data["bank_foreground_q"],
                bank_eor_q=data["bank_eor_q"],
                bank_total_q=data["bank_total_q"],
                source_kperp_indices=data["source_band_kperp_indices"],
                source_kpar_values=data["source_band_kpar_mpc_inv"],
                output_kpar_values=output_kpar,
                aggregation_weighting="response",
                minimum_relative_response=float(
                    args.minimum_relative_response
                ),
                minimum_window_fraction=float(
                    args.minimum_window_fraction
                ),
                minimum_kperp_index=int(args.minimum_kperp_index),
            )
            metrics = _fixed_selection_metrics(
                estimate=current["bank_total_estimate"],
                target=current["target"],
                selected=final_selected,
                weights=final_weights,
            )
            current_selected = np.asarray(current["selected"], dtype=bool)
            union = np.count_nonzero(current_selected | final_selected)
            ratios.append(metrics["integrated_power_ratio"])
            relative_l2_values.append(metrics["relative_l2"])
            row_sum_l2_values.append(
                _relative_l2(
                    np.sum(response, axis=1),
                    np.sum(final_response, axis=1),
                )
            )
            selected_counts.append(float(np.count_nonzero(current_selected)))
            jaccards.append(
                float(
                    np.count_nonzero(current_selected & final_selected)
                    / union
                )
            )
        subset_distributions.append(
            {
                "probe_count": int(count),
                "enumerated_subset_count": int(len(subsets)),
                "bank_total_final_selection_integrated_power_ratio": (
                    _distribution(ratios)
                ),
                "bank_total_final_selection_relative_l2": _distribution(
                    relative_l2_values
                ),
                "response_row_sum_relative_l2_to_full": _distribution(
                    row_sum_l2_values
                ),
                "selected_group_count": _distribution(selected_counts),
                "selection_jaccard_with_full": _distribution(jaccards),
            }
        )
    result = {
        "schema": "visibility_qbeta_calibration_probe_convergence",
        "schema_version": 1,
        "fixed_rows": True,
        "result_npz": str(args.result_npz),
        "calibration_repeat_count": int(repeat_count),
        "probe_counts": counts,
        "profile": "quad_kperp_response",
        "selection_uses_truth": False,
        "rows": summaries,
        "all_subset_distributions": subset_distributions,
    }
    products: dict[str, np.ndarray] = {
        "probe_counts": np.asarray(counts, dtype=np.int64),
        "final_selected": final_selected.astype(np.int8),
    }
    for count in counts:
        products[f"response_{count}"] = responses[count]
        products[f"target_{count}"] = arrays[count]["target"]
        products[f"bank_total_estimate_{count}"] = arrays[count][
            "bank_total_estimate"
        ]
        products[f"selected_{count}"] = arrays[count]["selected"]
    _atomic_json(args.out_dir / "summary.json", result)
    _atomic_npz(args.out_dir / "products.npz", products)
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
