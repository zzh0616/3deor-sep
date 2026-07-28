#!/usr/bin/env python3
"""Evaluate response-aware coarse visibility-Q_beta bandpowers."""

from __future__ import annotations

import argparse
import csv
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

from visibility_qbeta_coarse import (  # noqa: E402
    build_coarse_groups,
    evaluate_coarse_profile,
    matching_mode_counts,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


BASE_PROFILES = (
    ("fine", "fine", 1),
    ("low4", "low4", 1),
    ("low4_high2", "low4_high2", 1),
    ("low4_mid3_high2", "low4_mid3_high2", 1),
    ("low4_mid4_high2", "low4_mid4_high2", 1),
    ("pair_kperp", "pair", 1),
    ("quad_kperp", "quad", 1),
    ("low4_high2_kpar2", "low4_high2", 2),
    ("low4_mid3_high2_kpar2", "low4_mid3_high2", 2),
    ("pair_kperp_kpar2", "pair", 2),
    ("quad_kperp_kpar2", "quad", 2),
    ("quad_kperp_kpar4", "quad", 4),
)
PROFILES = tuple(
    (
        f"{name}_{weighting}",
        kperp_profile,
        kpar_group_size,
        weighting,
    )
    for name, kperp_profile, kpar_group_size in BASE_PROFILES
    for weighting in ("response", "mode_count")
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-npz", type=Path, required=True)
    parser.add_argument("--physical-shifts-npz", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-relative-response", type=float, default=0.1)
    parser.add_argument("--minimum-window-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-group-mode-count", type=float, default=0.0)
    parser.add_argument(
        "--minimum-nominal-window-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--maximum-window-effective-width", type=float, default=math.inf
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(profile[0] for profile in PROFILES),
        help="Evaluate only the named profile; repeat for multiple profiles.",
    )
    parser.add_argument("--minimum-kperp-index", type=int, default=0)
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


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _write_group_csv(
    path: Path,
    *,
    profile: str,
    groups: list[Any],
    arrays: dict[str, np.ndarray],
) -> None:
    selected_positions = np.flatnonzero(arrays["selected"])
    strict = arrays["strict_selected"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "profile",
                "group_index",
                "kperp_first",
                "kperp_stop",
                "kpar_first",
                "kpar_stop",
                "nominal_cell_count",
                "group_mode_count",
                "minimum_input_relative_response",
                "window_fraction",
                "nominal_window_fraction",
                "window_effective_width",
                "bank_total_error_percent",
                "heldout_worst_error_percent",
                "physical_shift_pure_worst_error_percent",
                "physical_shift_total_worst_error_percent",
                "foreground_effect_percent",
                "strict_pass",
            )
        )
        for selected_offset, group_index in enumerate(selected_positions):
            group = groups[int(group_index)]
            writer.writerow(
                (
                    profile,
                    int(group_index),
                    group.kperp_first,
                    group.kperp_stop,
                    group.kpar_first,
                    group.kpar_stop,
                    int(group.output_positions.size),
                    float(arrays["group_modes"][group_index]),
                    float(
                        arrays["minimum_input_relative_response"][group_index]
                    ),
                    float(arrays["window_fraction"][group_index]),
                    float(arrays["nominal_window_fraction"][group_index]),
                    float(arrays["effective_width"][group_index]),
                    100.0
                    * float(
                        arrays["selected_bank_error_fraction"][selected_offset]
                    ),
                    100.0
                    * float(
                        arrays["selected_heldout_worst_error_fraction"][
                            selected_offset
                        ]
                    ),
                    100.0
                    * float(
                        arrays[
                            "selected_physical_shift_pure_worst_error_fraction"
                        ][selected_offset]
                    ),
                    100.0
                    * float(
                        arrays[
                            "selected_physical_shift_total_worst_error_fraction"
                        ][selected_offset]
                    ),
                    100.0
                    * float(
                        arrays["selected_foreground_effect_fraction"][
                            selected_offset
                        ]
                    ),
                    int(strict[selected_offset]),
                )
            )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.combined_npz, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    shifts: dict[str, np.ndarray] | None = None
    if args.physical_shifts_npz is not None:
        with np.load(
            args.physical_shifts_npz, allow_pickle=False
        ) as archive:
            shifts = {
                name: np.asarray(archive[name]) for name in archive.files
            }
        if not np.array_equal(
            shifts["output_band_ids"], data["output_band_ids"]
        ):
            raise ValueError("Physical-shift and combined output bands differ")
    required = {
        "calibration_response",
        "restricted_eor_source_power",
        "source_band_in_geometric_window",
        "source_band_kperp_indices",
        "source_band_kpar_indices",
        "source_band_kpar_mpc_inv",
        "source_band_mode_counts",
        "output_band_ids",
        "restricted_eor_q",
        "heldout_mixture_q",
        "heldout_total_mixture_q",
        "bank_foreground_q",
        "bank_eor_q",
        "bank_total_q",
        "support",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Combined archive lacks: {', '.join(missing)}")
    support = np.asarray(data["support"])
    if support.ndim != 2:
        raise ValueError("support must be a two-dimensional output grid")
    kperp_count, kpar_count = support.shape
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    output_kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    if output_kpar_values.shape != (int(kpar_count),):
        raise ValueError("Config kpar layout differs from the combined archive")
    mode_counts = matching_mode_counts(
        data["output_band_ids"],
        kpar_count=int(kpar_count),
        source_kperp_indices=data["source_band_kperp_indices"],
        source_kpar_indices=data["source_band_kpar_indices"],
        source_mode_counts=data["source_band_mode_counts"],
    )

    summaries: dict[str, Any] = {}
    products: dict[str, np.ndarray] = {}
    for (
        profile_name,
        kperp_profile,
        kpar_group_size,
        weighting,
    ) in PROFILES:
        if args.profile and profile_name not in set(args.profile):
            continue
        groups = build_coarse_groups(
            data["output_band_ids"],
            kperp_count=int(kperp_count),
            kpar_count=int(kpar_count),
            kperp_profile=kperp_profile,
            kpar_group_size=int(kpar_group_size),
        )
        summary, arrays = evaluate_coarse_profile(
            response=data["calibration_response"],
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
            physical_shift_q=(
                None if shifts is None else shifts["physical_shift_q"]
            ),
            physical_shift_total_q=(
                None
                if shifts is None
                else shifts["physical_shift_total_q"]
            ),
            source_kperp_indices=data["source_band_kperp_indices"],
            source_kpar_values=data["source_band_kpar_mpc_inv"],
            output_kpar_values=output_kpar_values,
            aggregation_weighting=weighting,
            minimum_relative_response=float(args.minimum_relative_response),
            minimum_window_fraction=float(args.minimum_window_fraction),
            minimum_group_mode_count=float(args.minimum_group_mode_count),
            minimum_nominal_window_fraction=float(
                args.minimum_nominal_window_fraction
            ),
            maximum_window_effective_width=float(
                args.maximum_window_effective_width
            ),
            minimum_kperp_index=int(args.minimum_kperp_index),
            maximum_kperp_index_exclusive=args.maximum_kperp_index_exclusive,
        )
        summary.update(
            {
                "kperp_profile": kperp_profile,
                "kpar_group_size": int(kpar_group_size),
                "aggregation_weighting": weighting,
            }
        )
        summaries[profile_name] = summary
        for name, values in arrays.items():
            products[f"{profile_name}_{name}"] = values
        _write_group_csv(
            args.out_dir / f"{profile_name}_groups.csv",
            profile=profile_name,
            groups=groups,
            arrays=arrays,
        )

    result = {
        "schema": "visibility_qbeta_coarse_covariance_evaluation",
        "schema_version": 2,
        "combined_npz": str(args.combined_npz),
        "physical_shifts_npz": (
            None
            if args.physical_shifts_npz is None
            else str(args.physical_shifts_npz)
        ),
        "config": str(args.config),
        "selection": {
            "truth_blind": True,
            "minimum_relative_response": float(
                args.minimum_relative_response
            ),
            "minimum_window_fraction": float(args.minimum_window_fraction),
            "minimum_group_mode_count": float(
                args.minimum_group_mode_count
            ),
            "minimum_nominal_window_fraction": float(
                args.minimum_nominal_window_fraction
            ),
            "maximum_window_effective_width": float(
                args.maximum_window_effective_width
            ),
            "minimum_kperp_index": int(args.minimum_kperp_index),
            "maximum_kperp_index_exclusive": (
                None
                if args.maximum_kperp_index_exclusive is None
                else int(args.maximum_kperp_index_exclusive)
            ),
        },
        "profiles": summaries,
    }
    _atomic_json(args.out_dir / "summary.json", result)
    _atomic_npz(args.out_dir / "products.npz", products)
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
