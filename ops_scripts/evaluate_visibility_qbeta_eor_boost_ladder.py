#!/usr/bin/env python3
"""Evaluate EoR-amplitude scaling through frozen Q_beta responses."""

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

from ops_scripts.calibrate_visibility_qbeta_noiseless import (  # noqa: E402
    _load_bank,
    _maximum_patch_delays,
    _row_kperp,
    _visibility_bandpowers,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("exact", "common", "delay"):
        parser.add_argument(
            f"--{arm}-combined-npz", type=Path, required=True
        )
        parser.add_argument(f"--{arm}-coarse", type=Path, required=True)
    parser.add_argument("--exact-combined-json", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--amplitude-factor",
        action="append",
        type=float,
        default=[],
        help=(
            "EoR temperature/visibility amplitude multiplier; repeat for a "
            "ladder. Power scales as the square of this value."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Coarse response profile prefix; repeat for multiple profiles.",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


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


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(current - truth)
        / max(float(np.linalg.norm(truth)), 1e-300)
    )


def _quadratic_at_factor(
    factors: np.ndarray,
    quadratic_values: np.ndarray,
    factor: float,
) -> np.ndarray:
    ladder = np.asarray(factors, dtype=np.float64).reshape(-1)
    values = np.asarray(quadratic_values)
    if values.shape[0] != ladder.size:
        raise ValueError("Factor and quadratic-value axes differ")
    position = np.flatnonzero(np.isclose(ladder, factor, rtol=0.0, atol=1e-14))
    if position.size != 1:
        raise ValueError(f"Direct visibility ladder lacks factor {factor}")
    return values[int(position[0])]


def amplified_quadratic_q(
    *,
    foreground_q: np.ndarray,
    eor_q: np.ndarray,
    total_q: np.ndarray,
    amplitude_factor: float,
) -> np.ndarray:
    """Scale one component of a quadratic estimate, retaining its cross term."""
    foreground = np.asarray(foreground_q, dtype=np.float64)
    eor = np.asarray(eor_q, dtype=np.float64)
    total = np.asarray(total_q, dtype=np.float64)
    if foreground.shape != eor.shape or foreground.shape != total.shape:
        raise ValueError("Foreground, EoR, and total q arrays must match")
    factor = float(amplitude_factor)
    if not math.isfinite(factor) or factor < 0.0:
        raise ValueError("EoR amplitude factor must be finite and nonnegative")
    unit_cross = total - foreground - eor
    return foreground + factor * unit_cross + factor * factor * eor


def weighted_metrics(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    estimate = np.asarray(values, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metric_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        estimate.size < 1
        or estimate.shape != truth.shape
        or estimate.shape != metric_weights.shape
        or not np.all(np.isfinite(estimate))
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(metric_weights))
        or np.any(metric_weights < 0.0)
        or not np.any(metric_weights > 0.0)
    ):
        raise ValueError("Invalid estimate, target, or metric weights")
    difference = estimate - truth
    fractional = np.abs(difference) / np.maximum(np.abs(truth), 1e-300)
    return {
        "count": int(estimate.size),
        "integrated_power_ratio": float(
            np.sum(metric_weights * estimate)
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


def _component_fraction(
    component: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(component, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metric_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    fractional = np.abs(values) / np.maximum(np.abs(truth), 1e-300)
    return {
        "weighted_absolute_fraction": float(
            np.sum(metric_weights * np.abs(values))
            / np.sum(metric_weights * np.abs(truth))
        ),
        "relative_l2": float(
            np.sqrt(
                np.sum(metric_weights * np.square(values))
                / np.sum(metric_weights * np.square(truth))
            )
        ),
        "median_absolute_fraction": float(np.median(fractional)),
        "maximum_absolute_fraction": float(np.max(fractional)),
    }


def _coarse_estimate(
    q_values: np.ndarray,
    products: dict[str, np.ndarray],
    profile: str,
) -> np.ndarray:
    prefix = f"{profile}_"
    transform = np.asarray(products[prefix + "transform"], dtype=np.float64)
    response = np.asarray(products[prefix + "response"], dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if transform.shape[1] != q.size or response.shape[0] != transform.shape[0]:
        raise ValueError(f"{profile} transform, response, and q differ")
    row_sum = np.sum(response, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{profile} has a nonpositive response row sum")
    return (transform @ q) / row_sum


def _profile_ladder(
    *,
    profile: str,
    factors: np.ndarray,
    exact_products: dict[str, np.ndarray],
    arm_products: dict[str, np.ndarray],
    arm_combined: dict[str, np.ndarray],
) -> dict[str, Any]:
    prefix = f"{profile}_"
    required = {
        prefix + "target",
        prefix + "selected",
        prefix + "group_metric_weights",
        prefix + "transform",
        prefix + "response",
    }
    for label, products in (
        ("exact", exact_products),
        ("candidate", arm_products),
    ):
        missing = sorted(required - products.keys())
        if missing:
            raise ValueError(f"{label} products lack: {', '.join(missing)}")
    q_required = {"bank_foreground_q", "bank_eor_q", "bank_total_q"}
    missing_q = sorted(q_required - arm_combined.keys())
    if missing_q:
        raise ValueError(f"Combined archive lacks: {', '.join(missing_q)}")

    exact_selected = np.asarray(
        exact_products[prefix + "selected"], dtype=bool
    )
    arm_selected = np.asarray(arm_products[prefix + "selected"], dtype=bool)
    if exact_selected.shape != arm_selected.shape:
        raise ValueError(f"{profile} exact and candidate groups differ")
    selected = exact_selected & arm_selected
    if not np.any(selected):
        raise ValueError(f"{profile} has no pairwise selected groups")
    base_target = np.asarray(exact_products[prefix + "target"])[selected]
    native_target = np.asarray(arm_products[prefix + "target"])[selected]
    weights = np.asarray(
        exact_products[prefix + "group_metric_weights"]
    )[selected]

    foreground_q = np.asarray(arm_combined["bank_foreground_q"])
    eor_q = np.asarray(arm_combined["bank_eor_q"])
    total_q = np.asarray(arm_combined["bank_total_q"])
    foreground_estimate = _coarse_estimate(
        foreground_q, arm_products, profile
    )
    unit_cross_estimate = _coarse_estimate(
        total_q - foreground_q - eor_q,
        arm_products,
        profile,
    )
    eor_estimate = _coarse_estimate(eor_q, arm_products, profile)

    rows: list[dict[str, Any]] = []
    for factor in factors:
        power_factor = float(factor * factor)
        exact_target = power_factor * base_target
        candidate_target = power_factor * native_target
        pure_eor = power_factor * eor_estimate
        contamination = (
            foreground_estimate + float(factor) * unit_cross_estimate
        )
        total_estimate = pure_eor + contamination
        direct_formula = _coarse_estimate(
            amplified_quadratic_q(
                foreground_q=foreground_q,
                eor_q=eor_q,
                total_q=total_q,
                amplitude_factor=float(factor),
            ),
            arm_products,
            profile,
        )
        if not np.allclose(
            direct_formula, total_estimate, rtol=2e-12, atol=1e-18
        ):
            raise ValueError("Coarse EoR scaling decomposition failed")
        rows.append(
            {
                "amplitude_factor": float(factor),
                "power_factor": power_factor,
                "fixed_exact_target": weighted_metrics(
                    total_estimate[selected], exact_target, weights
                ),
                "fixed_exact_target_eor_only": weighted_metrics(
                    pure_eor[selected], exact_target, weights
                ),
                "native_response_target": weighted_metrics(
                    total_estimate[selected], candidate_target, weights
                ),
                "foreground_plus_cross_fraction": _component_fraction(
                    contamination[selected], exact_target, weights
                ),
            }
        )
    return {
        "pairwise_selected_group_count": int(np.count_nonzero(selected)),
        "candidate_selected_group_count": int(np.count_nonzero(arm_selected)),
        "rows": rows,
    }


def _direct_visibility_check(
    *,
    factors: np.ndarray,
    combined: dict[str, np.ndarray],
    metadata: dict[str, Any],
    bank_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    frequencies_hz = (
        np.asarray(combined["input_frequencies_mhz"], dtype=np.float64) * 1e6
    )
    bank, _ = _load_bank(bank_dir, requested_frequencies_hz=frequencies_hz)
    selected_rows = np.asarray(combined["selected_bank_rows"], dtype=np.int64)
    foreground_vis = np.asarray(
        bank["sample_fg"][:, selected_rows], dtype=np.complex128
    )
    eor_vis = np.asarray(
        bank["sample_eor"][:, selected_rows], dtype=np.complex128
    )
    reference_frequency_hz = (
        float(resolved.geometry["reference_frequency_mhz"]) * 1e6
    )
    row_kperp = _row_kperp(
        np.asarray(bank["sample_uvw_m"])[selected_rows],
        reference_frequency_hz=reference_frequency_hz,
        transverse_distance_mpc=float(
            resolved.geometry["transverse_distance_mpc"]
        ),
    )
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    radial_mpc_per_hz = float(
        resolved.geometry["radial_spacing_mpc"]
    ) / float(np.mean(np.diff(frequencies_hz)))
    maximum_delays = _maximum_patch_delays(
        kperp_edges=kperp_edges,
        transverse_distance_mpc=float(
            resolved.geometry["transverse_distance_mpc"]
        ),
        reference_frequency_hz=reference_frequency_hz,
        source_corner_angle_deg=float(
            resolved.geometry["source_corner_angle_deg"]
        ),
        wedge_buffer_mpc_inv=float(
            resolved.geometry["wedge_buffer_mpc_inv"]
        ),
        radial_mpc_per_hz=radial_mpc_per_hz,
    )
    settings = metadata["qbeta"]
    kwargs = {
        "frequencies_hz": frequencies_hz,
        "analysis_frequency_indices": np.asarray(
            combined["analysis_frequency_indices"], dtype=np.int64
        ),
        "filter_bandwidth_scope": str(settings["filter_bandwidth_scope"]),
        "row_kperp": row_kperp,
        "kperp_edges": kperp_edges,
        "maximum_delays_s": maximum_delays,
        "dpss_eigenvalue_threshold": float(
            settings["dpss_eigenvalue_threshold"]
        ),
        "foreground_filter": str(settings["foreground_filter"]),
        "suppression_strength": float(settings["suppression_strength"]),
        "polynomial_degree": int(settings["polynomial_degree"]),
        "spectral_taper": str(settings["spectral_taper"]),
    }
    signed_factors = np.unique(
        np.concatenate((-factors, np.asarray([0.0]), factors))
    )
    mixed_vis = np.stack(
        [foreground_vis + factor * eor_vis for factor in signed_factors]
    )
    direct_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=mixed_vis,
        **kwargs,
    )
    direct_eor_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=eor_vis,
        **kwargs,
    )
    output_ids = np.asarray(combined["output_band_ids"], dtype=np.int64)
    direct_q = np.asarray(direct_q).reshape(signed_factors.size, -1)[
        :, output_ids
    ]
    direct_eor_q = np.asarray(direct_eor_q).reshape(-1)[output_ids]
    saved_foreground = np.asarray(combined["bank_foreground_q"])
    saved_eor = np.asarray(combined["bank_eor_q"])
    saved_total = np.asarray(combined["bank_total_q"])
    def q_at(factor: float) -> np.ndarray:
        return _quadratic_at_factor(signed_factors, direct_q, factor)

    rows = []
    for factor in np.concatenate(([0.0], factors)):
        q_values = q_at(float(factor))
        algebra = amplified_quadratic_q(
            foreground_q=saved_foreground,
            eor_q=saved_eor,
            total_q=saved_total,
            amplitude_factor=float(factor),
        )
        rows.append(
            {
                "amplitude_factor": float(factor),
                "power_factor": float(factor * factor),
                "direct_vs_saved_quadratic_relative_l2": _relative_l2(
                    q_values, algebra
                ),
            }
        )
    zero_q = q_at(0.0)
    unit_cross = saved_total - saved_foreground - saved_eor
    parity_rows = []
    for factor in factors:
        plus = q_at(float(factor))
        minus = q_at(float(-factor))
        even_eor = 0.5 * (plus + minus) - zero_q
        odd_cross = (plus - minus) / (2.0 * float(factor))
        parity_rows.append(
            {
                "absolute_amplitude_factor": float(factor),
                "even_eor_relative_l2": _relative_l2(
                    even_eor, float(factor * factor) * saved_eor
                ),
                "odd_cross_relative_l2": _relative_l2(
                    odd_cross, unit_cross
                ),
            }
        )
    total_closure = (
        _relative_l2(q_at(1.0), saved_total)
        if np.any(np.isclose(signed_factors, 1.0))
        else None
    )
    return {
        "selected_visibility_row_count": int(selected_rows.size),
        "saved_component_closure": {
            "foreground_relative_l2": _relative_l2(
                q_at(0.0), saved_foreground
            ),
            "eor_relative_l2": _relative_l2(direct_eor_q, saved_eor),
            "total_relative_l2": total_closure,
        },
        "rows": rows,
        "positive_negative_parity_rows": parity_rows,
        "maximum_ladder_relative_l2": float(
            max(row["direct_vs_saved_quadratic_relative_l2"] for row in rows)
        ),
        "maximum_positive_ladder_relative_l2": float(
            max(
                row["direct_vs_saved_quadratic_relative_l2"]
                for row in rows
                if row["amplitude_factor"] > 0.0
            )
        ),
        "maximum_even_eor_relative_l2": float(
            max(row["even_eor_relative_l2"] for row in parity_rows)
        ),
        "maximum_odd_cross_relative_l2": float(
            max(row["odd_cross_relative_l2"] for row in parity_rows)
        ),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    factors = np.asarray(
        args.amplitude_factor
        or [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        dtype=np.float64,
    )
    if (
        factors.size < 2
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
        or np.unique(factors).size != factors.size
    ):
        raise ValueError("Amplitude factors must be distinct, finite, and positive")
    factors.sort()
    profiles = args.profile or [
        "fine_response",
        "pair_kperp_response",
        "quad_kperp_response",
        "quad_kperp_kpar2_response",
    ]
    combined = {
        "exact_station_pair": _load_npz(args.exact_combined_npz),
        "common_scalar_power": _load_npz(args.common_combined_npz),
        "delay_diagonal": _load_npz(args.delay_combined_npz),
    }
    coarse = {
        "exact_station_pair": _load_npz(args.exact_coarse),
        "common_scalar_power": _load_npz(args.common_coarse),
        "delay_diagonal": _load_npz(args.delay_coarse),
    }
    exact_products = coarse["exact_station_pair"]
    metadata = json.loads(
        args.exact_combined_json.read_text(encoding="utf-8")
    )
    result = {
        "schema": "visibility_qbeta_eor_boost_ladder",
        "schema_version": 1,
        "amplitude_factor_definition": (
            "Multiplier applied to the EoR brightness-temperature and "
            "visibility amplitude; EoR power scales as factor squared."
        ),
        "selection_contract": (
            "All response-only masks are frozen before examining the EoR "
            "amplitude ladder."
        ),
        "amplitude_factors": factors,
        "power_factors": np.square(factors),
        "direct_visibility_check": _direct_visibility_check(
            factors=factors,
            combined=combined["exact_station_pair"],
            metadata=metadata,
            bank_dir=args.bank_dir,
            config_path=args.config,
        ),
        "profiles": {
            profile: {
                arm: _profile_ladder(
                    profile=profile,
                    factors=factors,
                    exact_products=exact_products,
                    arm_products=coarse[arm],
                    arm_combined=combined[arm],
                )
                for arm in combined
            }
            for profile in profiles
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "summary.json", result)
    print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
