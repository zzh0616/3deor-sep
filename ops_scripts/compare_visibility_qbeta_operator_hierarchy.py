#!/usr/bin/env python3
"""Compare delay-diagonal, common-beam, and exact Q_beta responses."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-coarse", type=Path, required=True)
    parser.add_argument("--common-coarse", type=Path, required=True)
    parser.add_argument("--delay-coarse", type=Path, required=True)
    parser.add_argument("--exact-combined-npz", type=Path, required=True)
    parser.add_argument("--common-combined-npz", type=Path, required=True)
    parser.add_argument("--exact-combined-json", type=Path, required=True)
    parser.add_argument("--common-combined-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Coarse product prefix; repeat for multiple profiles.",
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def weighted_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metric_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        values.size < 1
        or values.shape != truth.shape
        or values.shape != metric_weights.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(metric_weights))
        or np.any(metric_weights < 0.0)
        or not np.any(metric_weights > 0.0)
    ):
        raise ValueError("Invalid estimate, target, or metric weights")
    difference = values - truth
    fractional = np.abs(difference) / np.maximum(np.abs(truth), 1e-300)
    return {
        "count": int(values.size),
        "integrated_power_ratio": float(
            np.sum(metric_weights * values)
            / np.sum(metric_weights * truth)
        ),
        "relative_l2": float(
            np.sqrt(
                np.sum(metric_weights * np.square(difference))
                / np.sum(metric_weights * np.square(truth))
            )
        ),
        "median_absolute_error_fraction": float(np.median(fractional)),
        "p90_absolute_error_fraction": float(np.quantile(fractional, 0.9)),
        "maximum_absolute_error_fraction": float(np.max(fractional)),
        "passing_10pct_count": int(np.count_nonzero(fractional <= 0.1)),
        "passing_20pct_count": int(np.count_nonzero(fractional <= 0.2)),
    }


def combined_response_diagnostics(
    *,
    exact: dict[str, np.ndarray],
    common: dict[str, np.ndarray],
) -> dict[str, Any]:
    required = {
        "calibration_response",
        "bank_foreground_q",
        "bank_eor_q",
        "bank_total_q",
        "selected_bank_rows",
    }
    for label, data in (("exact", exact), ("common", common)):
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"{label} combined archive lacks: {', '.join(missing)}")
    if not np.array_equal(
        exact["selected_bank_rows"], common["selected_bank_rows"]
    ):
        raise ValueError("Exact and common-beam selected rows differ")
    exact_response = np.asarray(exact["calibration_response"], dtype=np.float64)
    common_response = np.asarray(common["calibration_response"], dtype=np.float64)
    if exact_response.shape != common_response.shape:
        raise ValueError("Exact and common-beam response shapes differ")
    exact_row_sum = np.sum(exact_response, axis=1)
    common_row_sum = np.sum(common_response, axis=1)
    if np.any(exact_row_sum <= 0.0) or np.any(common_row_sum <= 0.0):
        raise ValueError("Response rows must have positive total gain")
    row_sum_ratio = common_row_sum / exact_row_sum
    exact_window = exact_response / exact_row_sum[:, None]
    common_window = common_response / common_row_sum[:, None]

    q_diagnostics: dict[str, Any] = {}
    for name in ("bank_foreground_q", "bank_eor_q", "bank_total_q"):
        exact_q = np.asarray(exact[name], dtype=np.float64)
        common_q = np.asarray(common[name], dtype=np.float64)
        q_diagnostics[name] = {
            "array_equal": bool(np.array_equal(exact_q, common_q)),
            "relative_l2": float(
                np.linalg.norm(common_q - exact_q)
                / max(float(np.linalg.norm(exact_q)), 1e-300)
            ),
        }
    return {
        "selected_rows_equal": True,
        "selected_visibility_row_count": int(
            exact["selected_bank_rows"].size
        ),
        "response_relative_l2": float(
            np.linalg.norm(common_response - exact_response)
            / np.linalg.norm(exact_response)
        ),
        "normalized_window_relative_l2": float(
            np.linalg.norm(common_window - exact_window)
            / np.linalg.norm(exact_window)
        ),
        "common_to_exact_response_row_sum_ratio": {
            "minimum": float(np.min(row_sum_ratio)),
            "median": float(np.median(row_sum_ratio)),
            "mean": float(np.mean(row_sum_ratio)),
            "maximum": float(np.max(row_sum_ratio)),
        },
        "observed_q": q_diagnostics,
    }


def _profile_comparison(
    *,
    profile: str,
    products: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    prefix = f"{profile}_"
    required_suffixes = {
        "selected",
        "target",
        "group_metric_weights",
        "bank_eor_estimate",
        "bank_total_estimate",
        "foreground_estimate",
        "window_fraction",
        "nominal_window_fraction",
        "effective_width",
        "group_kperp_first",
        "group_kperp_stop",
    }
    for arm, current in products.items():
        missing = sorted(
            prefix + suffix
            for suffix in required_suffixes
            if prefix + suffix not in current
        )
        if missing:
            raise ValueError(f"{arm} lacks profile arrays: {', '.join(missing)}")
    reference = products["exact_station_pair"]
    for arm, current in products.items():
        for suffix in ("group_kperp_first", "group_kperp_stop"):
            if not np.array_equal(
                current[prefix + suffix], reference[prefix + suffix]
            ):
                raise ValueError(f"{arm} and exact group geometry differ")
    selected_by_arm = {
        arm: np.asarray(current[prefix + "selected"], dtype=bool)
        for arm, current in products.items()
    }
    common = np.logical_and.reduce(list(selected_by_arm.values()))
    if not np.any(common):
        raise ValueError(f"No common selected groups for {profile}")
    fixed_target = np.asarray(reference[prefix + "target"])[common]
    weights = np.asarray(reference[prefix + "group_metric_weights"])[common]
    arms: dict[str, Any] = {}
    for arm, current in products.items():
        eor = np.asarray(current[prefix + "bank_eor_estimate"])[common]
        total = np.asarray(current[prefix + "bank_total_estimate"])[common]
        foreground = np.asarray(current[prefix + "foreground_estimate"])[common]
        native_target = np.asarray(current[prefix + "target"])[common]
        foreground_fraction = np.abs(total - eor) / np.maximum(
            np.abs(fixed_target), 1e-300
        )
        arms[arm] = {
            "selected_group_count": int(np.count_nonzero(selected_by_arm[arm])),
            "fixed_exact_target": weighted_metrics(total, fixed_target, weights),
            "fixed_exact_target_eor_only": weighted_metrics(
                eor, fixed_target, weights
            ),
            "native_response_target": weighted_metrics(
                total, native_target, weights
            ),
            "response_target_vs_exact": weighted_metrics(
                native_target, fixed_target, weights
            ),
            "foreground_effect_maximum_fraction": float(
                np.max(foreground_fraction)
            ),
            "foreground_effect_median_fraction": float(
                np.median(foreground_fraction)
            ),
            "foreground_estimate_maximum_absolute": float(
                np.max(np.abs(foreground))
            ),
            "window_fraction_minimum": float(
                np.min(current[prefix + "window_fraction"][common])
            ),
            "nominal_window_fraction_minimum": float(
                np.min(current[prefix + "nominal_window_fraction"][common])
            ),
            "window_effective_width_median": float(
                np.median(current[prefix + "effective_width"][common])
            ),
        }
    pairwise: dict[str, Any] = {}
    for arm in ("common_scalar_power", "delay_diagonal"):
        pair_selected = selected_by_arm["exact_station_pair"] & selected_by_arm[
            arm
        ]
        pair_target = np.asarray(reference[prefix + "target"])[pair_selected]
        pair_weights = np.asarray(
            reference[prefix + "group_metric_weights"]
        )[pair_selected]
        candidate = products[arm]
        pairwise[arm] = {
            "common_selected_group_count": int(np.count_nonzero(pair_selected)),
            "candidate_fixed_exact_target": weighted_metrics(
                candidate[prefix + "bank_total_estimate"][pair_selected],
                pair_target,
                pair_weights,
            ),
            "exact_fixed_exact_target": weighted_metrics(
                reference[prefix + "bank_total_estimate"][pair_selected],
                pair_target,
                pair_weights,
            ),
            "candidate_response_target_vs_exact": weighted_metrics(
                candidate[prefix + "target"][pair_selected],
                pair_target,
                pair_weights,
            ),
        }
    return {
        "common_selected_group_count": int(np.count_nonzero(common)),
        "group_count": int(common.size),
        "arms": arms,
        "pairwise_with_exact": pairwise,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    products = {
        "exact_station_pair": _load_npz(args.exact_coarse),
        "common_scalar_power": _load_npz(args.common_coarse),
        "delay_diagonal": _load_npz(args.delay_coarse),
    }
    exact_combined = _load_npz(args.exact_combined_npz)
    common_combined = _load_npz(args.common_combined_npz)
    profiles = args.profile or [
        "fine_response",
        "pair_kperp_response",
        "quad_kperp_response",
        "quad_kperp_kpar2_response",
    ]
    exact_meta = json.loads(args.exact_combined_json.read_text(encoding="utf-8"))
    common_meta = json.loads(
        args.common_combined_json.read_text(encoding="utf-8")
    )
    result = {
        "schema": "visibility_qbeta_operator_hierarchy_comparison",
        "schema_version": 1,
        "fixed_target": (
            "The exact station-pair response window applied to the injected "
            "EoR source bandpowers."
        ),
        "delay_diagonal_definition": (
            "Exact response row sums assigned to nominal single cells; this "
            "is an interpretation control, not a visibility closure model."
        ),
        "operator_closure": {
            "exact_station_pair": exact_meta.get("operator_closure"),
            "common_scalar_power": common_meta.get("operator_closure"),
            "delay_diagonal": None,
        },
        "combined_response_diagnostics": combined_response_diagnostics(
            exact=exact_combined,
            common=common_combined,
        ),
        "profiles": {
            profile: _profile_comparison(profile=profile, products=products)
            for profile in profiles
        },
    }
    _atomic_json(args.out_dir / "summary.json", result)
    print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
