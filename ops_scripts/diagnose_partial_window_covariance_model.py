#!/usr/bin/env python3
"""Diagnose EoR covariance mismatch in the partial-window likelihood."""

from __future__ import annotations

import argparse
import json
import math
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

from covariance_partial_window import (  # noqa: E402
    fill_conjugate_power_cube,
    fit_covariance_grid,
    fit_to_dict,
    independent_spatial_coordinates,
    posterior_radial_powers,
    regularize_covariance,
    second_moment_covariance,
    spatial_fourier_modes,
)
from estimate_partial_window_covariance_ps2d import (  # noqa: E402
    _load_product,
    _mask_metrics,
    _mode_samples,
    _product_arrays,
    _single_cube_samples,
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
    parser.add_argument("--precision", type=float, default=0.001)
    parser.add_argument("--q-fg-grid", default="0.01,0.03,0.1,0.25,0.5,1,2,4")
    parser.add_argument("--q-eor-log10-min", type=float, default=-10.0)
    parser.add_argument("--q-eor-log10-max", type=float, default=0.0)
    parser.add_argument("--q-eor-grid-size", type=int, default=51)
    parser.add_argument("--q-fg-log-sigma", type=float, default=math.log(3.0))
    parser.add_argument("--diagonal-shrinkage", type=float, default=0.05)
    parser.add_argument("--eigen-floor-fraction", type=float, default=1e-8)
    parser.add_argument("--minimum-independent-modes", type=int, default=8)
    parser.add_argument("--max-retained-states", type=int, default=256)
    parser.add_argument("--quick-relative-tolerance", type=float, default=0.3)
    parser.add_argument("--strict-relative-tolerance", type=float, default=0.2)
    parser.add_argument("--foreground-leakage-tolerance", type=float, default=0.1)
    return parser.parse_args(argv)


def _normalized_shape(covariance: np.ndarray) -> np.ndarray:
    values = np.asarray(covariance, dtype=np.complex128)
    trace = float(np.real(np.trace(values)))
    return values / max(trace, np.finfo(np.float64).tiny)


def _coherence(covariance: np.ndarray) -> np.ndarray:
    values = np.asarray(covariance, dtype=np.complex128)
    diagonal = np.maximum(np.real(np.diag(values)), np.finfo(np.float64).tiny)
    return values / np.sqrt(diagonal[:, None] * diagonal[None, :])


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / denominator)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.monotonic()
    if float(args.precision) <= 0.0:
        raise ValueError("precision must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    manifest = json.loads((args.bank_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("analysis_contract_sha256") != resolved.contract.analysis_contract_sha256:
        raise ValueError("bank and config analysis contracts differ")
    products = manifest["products"]
    exact_eor = _load_product(args.bank_dir, products["exact_dirty_eor"])
    heldout_unit = _load_product(args.bank_dir, products["fg_error_heldout_unit"])
    foreground_draws_unit = _load_product(args.bank_dir, products["fg_error_draws_unit"])
    eor_probe_cubes = {
        float(ell): _load_product(args.bank_dir, record)
        for ell, record in products["eor_probes"].items()
    }

    shape = tuple(int(value) for value in resolved.contract.full_layout.cube_shape)
    if exact_eor.shape != shape or heldout_unit.shape != shape:
        raise ValueError("bank cube shape does not match config")
    nfreq, height, width = shape
    radial_window = np.hanning(nfreq)
    spatial_window_y = np.hanning(height)
    spatial_window_x = np.hanning(width)
    demean_mode = str(resolved.contract.demean_mode)
    foreground_draw_modes = spatial_fourier_modes(
        foreground_draws_unit,
        demean_mode=demean_mode,
        spatial_window_y=spatial_window_y,
        spatial_window_x=spatial_window_x,
    )
    exact_eor_modes = spatial_fourier_modes(
        exact_eor,
        demean_mode=demean_mode,
        spatial_window_y=spatial_window_y,
        spatial_window_x=spatial_window_x,
    )
    heldout_modes = spatial_fourier_modes(
        heldout_unit,
        demean_mode=demean_mode,
        spatial_window_y=spatial_window_y,
        spatial_window_x=spatial_window_x,
    )
    eor_probe_modes = {
        ell: spatial_fourier_modes(
            cubes,
            demean_mode=demean_mode,
            spatial_window_y=spatial_window_y,
            spatial_window_x=spatial_window_x,
        )
        for ell, cubes in eor_probe_cubes.items()
    }
    del foreground_draws_unit, eor_probe_cubes

    layout = resolved.contract.window_layout
    coordinates_all = independent_spatial_coordinates(height, width)
    ky = 2.0 * math.pi * np.fft.fftfreq(height, d=float(layout.dy_mpc))
    kx = 2.0 * math.pi * np.fft.fftfreq(width, d=float(layout.dx_mpc))
    coordinate_kperp = np.sqrt(
        np.square(ky[coordinates_all[:, 0]]) + np.square(kx[coordinates_all[:, 1]])
    )
    coordinate_bins = np.searchsorted(layout.kperp_edges, coordinate_kperp, side="right") - 1
    coordinate_bins[
        np.isclose(coordinate_kperp, layout.kperp_edges[-1], rtol=1e-12, atol=1e-14)
    ] = layout.kperp_edges.size - 2
    coordinate_valid = (
        (coordinate_kperp <= float(layout.transverse_circle_max))
        & (coordinate_bins >= 0)
        & (coordinate_bins < layout.kperp_centers.size)
    )

    reference = np.load(args.reference_result_npz)
    reporting_masks = {
        key.removeprefix("mask_"): np.asarray(reference[key], dtype=bool)
        for key in reference.files
        if key.startswith("mask_")
    }
    if not reporting_masks:
        raise ValueError("reference result does not contain reporting masks")

    truth_power_cube, _ = fft_auto_power_cube(
        exact_eor,
        dx_mpc=float(layout.dx_mpc),
        dy_mpc=float(layout.dy_mpc),
        dpar_mpc=float(layout.dpar_mpc),
        demean_mode=demean_mode,
        radial_taper=str(resolved.contract.radial_taper),
        spatial_taper=str(resolved.contract.spatial_taper),
    )
    truth_product = _product_arrays(truth_power_cube, layout)
    q_fg_grid = _parse_floats(args.q_fg_grid)
    q_eor_grid = np.logspace(
        float(args.q_eor_log10_min),
        float(args.q_eor_log10_max),
        int(args.q_eor_grid_size),
    )
    oracle_q_grid = np.logspace(-2.0, 2.0, 41)
    precision = float(args.precision)
    output_cubes = {
        name: {
            power: np.zeros(shape, dtype=np.float64)
            for power in (
                "posterior_second_moment",
                "posterior_mean",
                "foreground_leakage_mean",
                "eor_transfer_mean",
            )
        }
        for name in ("matched_current_probe", "matched_oracle_covariance")
    }
    diagnostics_by_kperp: dict[str, Any] = {}

    for kperp_bin in range(layout.kperp_centers.size):
        coordinates = coordinates_all[coordinate_valid & (coordinate_bins == kperp_bin)]
        if coordinates.shape[0] < int(args.minimum_independent_modes):
            continue
        fg_covariance_unit, fg_stats = regularize_covariance(
            second_moment_covariance(_mode_samples(foreground_draw_modes, coordinates)),
            diagonal_shrinkage=float(args.diagonal_shrinkage),
            eigen_floor_fraction=float(args.eigen_floor_fraction),
        )
        fg_covariance = precision**2 * fg_covariance_unit
        eor_samples = _single_cube_samples(exact_eor_modes, coordinates)
        fg_samples = precision * _single_cube_samples(heldout_modes, coordinates)
        total_samples = fg_samples + eor_samples
        exact_eor_covariance, exact_stats = regularize_covariance(
            second_moment_covariance(eor_samples),
            diagonal_shrinkage=float(args.diagonal_shrinkage),
            eigen_floor_fraction=float(args.eigen_floor_fraction),
        )
        current_covariances: dict[float, np.ndarray] = {}
        covariance_shape: dict[str, Any] = {}
        for ell, modes in sorted(eor_probe_modes.items()):
            covariance, _ = regularize_covariance(
                second_moment_covariance(_mode_samples(modes, coordinates)),
                diagonal_shrinkage=float(args.diagonal_shrinkage),
                eigen_floor_fraction=float(args.eigen_floor_fraction),
            )
            current_covariances[float(ell)] = covariance
            covariance_shape[f"{float(ell):.8g}"] = {
                "trace_ratio_probe_over_exact": float(
                    np.real(np.trace(covariance))
                    / max(float(np.real(np.trace(exact_eor_covariance))), 1e-300)
                ),
                "normalized_covariance_relative_frobenius": _relative_frobenius(
                    _normalized_shape(exact_eor_covariance), _normalized_shape(covariance)
                ),
                "coherence_relative_frobenius": _relative_frobenius(
                    _coherence(exact_eor_covariance), _coherence(covariance)
                ),
            }
        best_ell = min(
            current_covariances,
            key=lambda ell: covariance_shape[f"{ell:.8g}"][
                "normalized_covariance_relative_frobenius"
            ],
        )
        variants = {
            "matched_current_probe": (current_covariances, q_eor_grid),
            "matched_oracle_covariance": ({1.0: exact_eor_covariance}, oracle_q_grid),
        }
        fit_records: dict[str, Any] = {}
        for name, (eor_covariances, amplitude_grid) in variants.items():
            fit = fit_covariance_grid(
                total_samples,
                fg_covariance,
                eor_covariances,
                q_fg_grid=q_fg_grid,
                q_eor_k2_grid=amplitude_grid,
                q_fg_log_sigma=float(args.q_fg_log_sigma),
                max_retained_states=int(args.max_retained_states),
            )
            powers = posterior_radial_powers(
                total_samples,
                fg_samples,
                eor_samples,
                fg_covariance,
                eor_covariances,
                fit,
                radial_window=radial_window,
            )
            for power_name, values in powers.items():
                fill_conjugate_power_cube(
                    output_cubes[name][power_name],
                    coordinates,
                    float(resolved.contract.power_scale) * values,
                )
            fit_records[name] = fit_to_dict(fit)
        heldout_covariance = second_moment_covariance(fg_samples)
        diagnostics_by_kperp[str(kperp_bin)] = {
            "independent_spatial_modes": int(coordinates.shape[0]),
            "heldout_fg_trace_over_ensemble": float(
                np.real(np.trace(heldout_covariance))
                / max(float(np.real(np.trace(fg_covariance))), 1e-300)
            ),
            "foreground_covariance_stats": fg_stats,
            "exact_eor_covariance_stats": exact_stats,
            "best_current_probe_ell_by_shape_mhz": float(best_ell),
            "best_current_probe_shape": covariance_shape[f"{best_ell:.8g}"],
            "all_current_probe_shapes": covariance_shape,
            "fits": fit_records,
        }
        print(
            json.dumps(
                {
                    "event": "kperp_done",
                    "kperp_bin": int(kperp_bin),
                    "mode_count": int(coordinates.shape[0]),
                    "heldout_fg_trace_over_ensemble": diagnostics_by_kperp[str(kperp_bin)][
                        "heldout_fg_trace_over_ensemble"
                    ],
                    "best_probe_ell_mhz": float(best_ell),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    variant_metrics: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {
        "truth_eor_mean": truth_product["mean"],
        "selected_independent_mode_counts": truth_product["independent_mode_counts"],
        "kperp_centers_mpc_inv": np.asarray(layout.kperp_centers),
        "kpar_values_mpc_inv": np.asarray(layout.kpar_values),
    }
    for mask_name, mask in reporting_masks.items():
        npz_payload[f"mask_{mask_name}"] = np.asarray(mask, dtype=np.uint8)
    for variant_name, cubes in output_cubes.items():
        products_by_power = {
            name: _product_arrays(cube, layout) for name, cube in cubes.items()
        }
        variant_metrics[variant_name] = {
            mask_name: _mask_metrics(
                products_by_power["posterior_second_moment"]["mean"],
                truth_product["mean"],
                products_by_power["foreground_leakage_mean"]["mean"],
                truth_product["fft_mode_counts"],
                mask,
                independent_counts=truth_product["independent_mode_counts"],
                quick_tolerance=float(args.quick_relative_tolerance),
                strict_tolerance=float(args.strict_relative_tolerance),
                foreground_tolerance=float(args.foreground_leakage_tolerance),
            )
            for mask_name, mask in reporting_masks.items()
        }
        for power_name, product in products_by_power.items():
            npz_payload[f"{variant_name}_{power_name}_mean"] = product["mean"]

    result = {
        "schema": "partial_window_covariance_model_diagnostic",
        "schema_version": 1,
        "created_at": _now(),
        "precision": precision,
        "control_interpretation": (
            "matched synthetic total uses an independent operator-propagated foreground-error "
            "draw plus exact dirty EoR; oracle EoR covariance is a truth-only mathematical control"
        ),
        "fit_uses_eor_truth": {
            "matched_current_probe": False,
            "matched_oracle_covariance": True,
        },
        "q_fg_grid": q_fg_grid,
        "q_eor_grid_current_probe": q_eor_grid.tolist(),
        "q_eor_grid_oracle_covariance": oracle_q_grid.tolist(),
        "metrics": variant_metrics,
        "diagnostics_by_kperp": diagnostics_by_kperp,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _write_json(args.out_dir / "result.json", result)
    np.savez_compressed(args.out_dir / "result.npz", **npz_payload)
    print(
        json.dumps(
            {
                "event": "covariance_model_diagnostic_done",
                "metrics": variant_metrics,
                "elapsed_seconds": result["elapsed_seconds"],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
