#!/usr/bin/env python3
"""Estimate partial-window bandpowers by subtracting foreground covariance bias."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, CODE_DIR, CODE_DIR / "code" / "3dnet"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from estimate_partial_window_covariance_ps2d import (  # noqa: E402
    _load_product,
    _mask_metrics,
    _product_arrays,
)
from ps2d_v2 import fft_auto_power_cube  # noqa: E402
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_floats(value: str) -> list[float]:
    return [float(piece.strip()) for piece in str(value).split(",") if piece.strip()]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--reference-result-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--precision-levels",
        default="0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01",
    )
    parser.add_argument("--quick-relative-tolerance", type=float, default=0.3)
    parser.add_argument("--strict-relative-tolerance", type=float, default=0.2)
    parser.add_argument("--foreground-bias-tolerance", type=float, default=0.1)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _power(cube: np.ndarray, resolved: Any) -> np.ndarray:
    layout = resolved.contract.window_layout
    power, _ = fft_auto_power_cube(
        np.asarray(cube, dtype=np.float64),
        dx_mpc=float(layout.dx_mpc),
        dy_mpc=float(layout.dy_mpc),
        dpar_mpc=float(layout.dpar_mpc),
        demean_mode=str(resolved.contract.demean_mode),
        radial_taper=str(resolved.contract.radial_taper),
        spatial_taper=str(resolved.contract.spatial_taper),
    )
    return np.asarray(power, dtype=np.float64)


def _evaluate(
    total_power_cube: np.ndarray,
    predicted_foreground_power_cube: np.ndarray,
    actual_foreground_power_cube: np.ndarray,
    truth_product: dict[str, np.ndarray],
    layout: Any,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    total = _product_arrays(total_power_cube, layout)
    predicted_foreground = _product_arrays(predicted_foreground_power_cube, layout)
    actual_foreground = _product_arrays(actual_foreground_power_cube, layout)
    estimate = {
        "mean": total["mean"] - predicted_foreground["mean"],
        "power_sum": total["power_sum"] - predicted_foreground["power_sum"],
    }
    foreground_bias = {
        "mean": actual_foreground["mean"] - predicted_foreground["mean"],
        "power_sum": (
            actual_foreground["power_sum"] - predicted_foreground["power_sum"]
        ),
    }
    metrics = {
        mask_name: _mask_metrics(
            estimate["mean"],
            truth_product["mean"],
            np.abs(foreground_bias["mean"]),
            truth_product["fft_mode_counts"],
            mask,
            independent_counts=truth_product["independent_mode_counts"],
            quick_tolerance=float(args.quick_relative_tolerance),
            strict_tolerance=float(args.strict_relative_tolerance),
            foreground_tolerance=float(args.foreground_bias_tolerance),
        )
        for mask_name, mask in masks.items()
    }
    return metrics, {
        "estimate_mean": estimate["mean"],
        "estimate_power_sum": estimate["power_sum"],
        "foreground_bias_mean": foreground_bias["mean"],
        "foreground_bias_power_sum": foreground_bias["power_sum"],
    }


def _control_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    output: dict[str, Any] = {"count": len(records)}
    for key in (
        "integrated_power_ratio",
        "count_weighted_relative_l2",
        "foreground_integrated_over_eor",
        "median_power_ratio",
    ):
        values = np.asarray(
            [record[key] for record in records if key in record and np.isfinite(record[key])],
            dtype=np.float64,
        )
        if values.size:
            output[key] = {
                "minimum": float(np.min(values)),
                "p10": float(np.percentile(values, 10.0)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90.0)),
                "maximum": float(np.max(values)),
            }
    ratios = np.asarray(
        [
            record["integrated_power_ratio"]
            for record in records
            if "integrated_power_ratio" in record
            and np.isfinite(record["integrated_power_ratio"])
        ],
        dtype=np.float64,
    )
    output["fraction_integrated_within_quick_tolerance"] = float(
        np.mean(np.abs(ratios - 1.0) <= 0.3)
    ) if ratios.size else 0.0
    output["fraction_integrated_within_strict_tolerance"] = float(
        np.mean(np.abs(ratios - 1.0) <= 0.2)
    ) if ratios.size else 0.0
    return output


def _add_geometric_contractions(
    masks: dict[str, np.ndarray],
    layout: Any,
) -> dict[str, np.ndarray]:
    """Add fixed high-kpar contractions without consulting any sky truth."""
    output = dict(masks)
    standard = np.asarray(masks["standard_window"], dtype=bool)
    kperp = np.asarray(layout.kperp_centers, dtype=np.float64)[:, None]
    lower = float(layout.kperp_edges[0])
    width = float(layout.kperp_edges[-1] - layout.kperp_edges[0])
    radial_index = np.arange(layout.kpar_values.size, dtype=np.int64)[None, :]
    top_two_start = max(0, int(layout.kpar_values.size) - 2)
    top_one_start = max(0, int(layout.kpar_values.size) - 1)
    mid = (kperp >= lower + 0.20 * width) & (kperp <= lower + 0.50 * width)
    core = (kperp >= lower + 0.30 * width) & (kperp <= lower + 0.45 * width)
    output["top2_kpar_mid_kperp"] = standard & (radial_index >= top_two_start) & mid
    output["top1_kpar_mid_kperp"] = standard & (radial_index >= top_one_start) & mid
    output["top2_kpar_core_kperp"] = standard & (radial_index >= top_two_start) & core
    output["top1_kpar_core_kperp"] = standard & (radial_index >= top_one_start) & core
    return output


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    manifest = json.loads((args.bank_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("analysis_contract_sha256") != resolved.contract.analysis_contract_sha256:
        raise ValueError("bank and config analysis contracts differ")
    products = manifest["products"]
    exact_fg = _load_product(args.bank_dir, products["exact_dirty_fg"])
    exact_eor = _load_product(args.bank_dir, products["exact_dirty_eor"])
    operator_fg = _load_product(args.bank_dir, products["operator_dirty_fg"])
    heldout_unit = _load_product(args.bank_dir, products["fg_error_heldout_unit"])
    foreground_draws_unit = _load_product(args.bank_dir, products["fg_error_draws_unit"])
    closure_residual = exact_fg - operator_fg

    reference = np.load(args.reference_result_npz)
    masks = {
        key.removeprefix("mask_"): np.asarray(reference[key], dtype=bool)
        for key in reference.files
        if key.startswith("mask_")
    }
    if not masks:
        raise ValueError("reference result does not contain reporting masks")
    layout = resolved.contract.window_layout
    masks = _add_geometric_contractions(masks, layout)
    truth_power = _power(exact_eor, resolved)
    truth_product = _product_arrays(truth_power, layout)
    standard_mask = np.asarray(masks["standard_window"], dtype=bool)
    standard_truth_sum = float(np.sum(truth_product["power_sum"][standard_mask]))
    reporting_mask_metadata: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        rows, columns = np.nonzero(mask)
        truth_sum = float(np.sum(truth_product["power_sum"][mask]))
        reporting_mask_metadata[mask_name] = {
            "bin_count": int(np.count_nonzero(mask)),
            "independent_mode_count": int(
                np.sum(truth_product["independent_mode_counts"][mask])
            ),
            "kperp_min_mpc_inv": float(np.min(layout.kperp_centers[rows])) if rows.size else None,
            "kperp_max_mpc_inv": float(np.max(layout.kperp_centers[rows])) if rows.size else None,
            "kpar_values_mpc_inv": (
                np.asarray(layout.kpar_values[np.unique(columns)], dtype=np.float64).tolist()
                if columns.size
                else []
            ),
            "postfit_truth_eor_power_fraction_of_standard_window": float(
                truth_sum / max(standard_truth_sum, 1e-300)
            ),
        }
    heldout_power = _power(heldout_unit, resolved)
    closure_power = _power(closure_residual, resolved)
    draw_powers = np.stack([_power(draw, resolved) for draw in foreground_draws_unit])
    mean_draw_power = np.mean(draw_powers, axis=0)
    sum_draw_power = np.sum(draw_powers, axis=0)

    npz_payload: dict[str, np.ndarray] = {
        "truth_eor_mean": truth_product["mean"],
        "selected_independent_mode_counts": truth_product["independent_mode_counts"],
        "kperp_centers_mpc_inv": np.asarray(layout.kperp_centers),
        "kpar_values_mpc_inv": np.asarray(layout.kpar_values),
    }
    for mask_name, mask in masks.items():
        npz_payload[f"mask_{mask_name}"] = np.asarray(mask, dtype=np.uint8)

    results: dict[str, Any] = {}
    for precision in _parse_floats(args.precision_levels):
        if precision <= 0.0:
            raise ValueError("precision levels must be positive")
        predicted_fg_power = precision**2 * mean_draw_power
        heldout_variants = {
            "minus": closure_residual - precision * heldout_unit,
            "plus": closure_residual + precision * heldout_unit,
        }
        heldout_results: dict[str, Any] = {}
        for sign_name, fg_residual in heldout_variants.items():
            total_power = _power(exact_eor + fg_residual, resolved)
            actual_fg_power = _power(fg_residual, resolved)
            metrics, arrays = _evaluate(
                total_power,
                predicted_fg_power,
                actual_fg_power,
                truth_product,
                layout,
                masks,
                args,
            )
            heldout_results[sign_name] = metrics
            label = f"p{precision:.8g}".replace(".", "p").replace("-", "m")
            for name, values in arrays.items():
                npz_payload[f"{label}_heldout_{sign_name}_{name}"] = values

        control_records = {mask_name: [] for mask_name in masks}
        for draw_index, draw in enumerate(foreground_draws_unit):
            predicted_loo = precision**2 * (
                sum_draw_power - draw_powers[draw_index]
            ) / float(foreground_draws_unit.shape[0] - 1)
            for sign in (-1.0, 1.0):
                fg_residual = closure_residual + sign * precision * draw
                total_power = _power(exact_eor + fg_residual, resolved)
                actual_fg_power = _power(fg_residual, resolved)
                metrics, _ = _evaluate(
                    total_power,
                    predicted_loo,
                    actual_fg_power,
                    truth_product,
                    layout,
                    masks,
                    args,
                )
                for mask_name in masks:
                    control_records[mask_name].append(metrics[mask_name])
        controls = {
            mask_name: _control_summary(records)
            for mask_name, records in control_records.items()
        }
        results[f"{precision:.8g}"] = {
            "foreground_precision": {
                "fractional_amplitude_rms": precision,
                "spectral_index_rms": 2.0 * precision,
                "astrometric_shift_rms_px": 4.0 * precision,
                "unresolved_confusion_rms_fraction": 0.3 * precision,
            },
            "heldout": heldout_results,
            "leave_one_out_sign_controls": controls,
            "foreground_residual_over_eor_rms": {
                sign_name: float(
                    np.sqrt(np.mean(np.square(residual)))
                    / max(float(np.sqrt(np.mean(np.square(exact_eor)))), 1e-300)
                )
                for sign_name, residual in heldout_variants.items()
            },
        }
        print(
            json.dumps(
                {
                    "event": "precision_done",
                    "precision": precision,
                    "heldout": heldout_results,
                    "controls": controls,
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema": "partial_window_debiased_ps2d_result",
        "schema_version": 1,
        "created_at": _now(),
        "method": "mode_power_minus_operator_propagated_foreground_covariance_bias",
        "scientific_target": "dirty_eor_ps2d_bandpower",
        "map_recovery_claim": False,
        "noise_model": "none",
        "fit_uses_eor_truth": False,
        "foreground_prior_is_truth_derived_emulator": True,
        "truth_use": (
            "foreground truth is degraded once to emulate an external observed prior; "
            "EoR truth and hidden foreground residuals are used only for post-fit validation"
        ),
        "config": str(args.config),
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "bank_manifest": str(args.bank_dir / "manifest.json"),
        "foreground_prior_emulator": manifest["foreground_prior_emulator"],
        "foreground_covariance_training_draw_count": int(foreground_draws_unit.shape[0]),
        "validation": "independent heldout draw plus signed leave-one-out controls",
        "geometric_contractions": {
            "selection_uses_truth": False,
            "top2_kpar_mid_kperp": "highest two native |kpar| groups and 20--50% kperp span",
            "top1_kpar_mid_kperp": "highest native |kpar| group and 20--50% kperp span",
            "top2_kpar_core_kperp": "highest two native |kpar| groups and 30--45% kperp span",
            "top1_kpar_core_kperp": "highest native |kpar| group and 30--45% kperp span",
        },
        "reporting_masks": {
            "selection_uses_truth": False,
            "truth_power_fractions_are_postfit_diagnostics_only": True,
            "metadata": reporting_mask_metadata,
        },
        "operator_identity": manifest["operator_identity"],
        "operator_closure": manifest["closure"],
        "operator_closure_residual_over_eor_rms": float(
            np.sqrt(np.mean(np.square(closure_residual)))
            / max(float(np.sqrt(np.mean(np.square(exact_eor)))), 1e-300)
        ),
        "diagnostic_power_ratios": {
            "heldout_unit_over_training_mean": float(
                np.sum(heldout_power) / max(float(np.sum(mean_draw_power)), 1e-300)
            ),
            "closure_over_eor": float(
                np.sum(closure_power) / max(float(np.sum(truth_power)), 1e-300)
            ),
        },
        "gates": {
            "quick_relative_tolerance": float(args.quick_relative_tolerance),
            "strict_relative_tolerance": float(args.strict_relative_tolerance),
            "foreground_bias_tolerance": float(args.foreground_bias_tolerance),
        },
        "results": results,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(args.out_dir / "result.json", result)
    np.savez_compressed(args.out_dir / "result.npz", **npz_payload)
    print(
        json.dumps(
            {
                "event": "partial_window_debiased_ps2d_done",
                "result_json": str(args.out_dir / "result.json"),
                "elapsed_seconds": result["elapsed_seconds"],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
