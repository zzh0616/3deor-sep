#!/usr/bin/env python3
"""Estimate a partial-window EoR PS2D with marginalized foreground covariance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
for candidate in (CODE_DIR, CODE_DIR / "code" / "3dnet"):
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
from ps2d_v2 import aggregate_power_cube, fft_auto_power_cube  # noqa: E402
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_floats(value: str) -> list[float]:
    return [float(piece.strip()) for piece in str(value).split(",") if piece.strip()]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--precision-levels",
        default="0.001,0.003,0.01,0.03,0.1",
        help="Fractional foreground-template amplitude RMS p.",
    )
    parser.add_argument("--q-fg-grid", default="0.25,0.5,1,2,4")
    parser.add_argument("--q-eor-log10-min", type=float, default=-8.0)
    parser.add_argument("--q-eor-log10-max", type=float, default=-1.0)
    parser.add_argument("--q-eor-grid-size", type=int, default=36)
    parser.add_argument("--q-fg-log-sigma", type=float, default=math.log(2.0))
    parser.add_argument("--diagonal-shrinkage", type=float, default=0.05)
    parser.add_argument("--eigen-floor-fraction", type=float, default=1e-8)
    parser.add_argument("--minimum-independent-modes", type=int, default=8)
    parser.add_argument("--prior-clean-fraction", type=float, default=0.2)
    parser.add_argument("--quick-relative-tolerance", type=float, default=0.3)
    parser.add_argument("--strict-relative-tolerance", type=float, default=0.2)
    parser.add_argument("--foreground-leakage-tolerance", type=float, default=0.1)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_product(bank_dir: Path, record: dict[str, Any]) -> np.ndarray:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = bank_dir / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"bank product hash mismatch: {path}")
    values = np.load(path, mmap_mode="r")
    if list(values.shape) != list(record["shape"]):
        raise ValueError(f"bank product shape mismatch: {path}")
    return np.asarray(values)


def _radial_covariance_power(covariance: np.ndarray, radial_window: np.ndarray) -> np.ndarray:
    transform = np.fft.fft(np.diag(radial_window), axis=0)
    return np.maximum(
        np.real(np.diag(transform @ covariance @ transform.conj().T)),
        np.finfo(np.float64).tiny,
    )


def _native_kpar_groups(frequencies: int) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(np.fft.fftfreq(int(frequencies)))
    values, groups = np.unique(absolute, return_inverse=True)
    if int(frequencies) % 2 == 0:
        nyquist = int(frequencies) // 2
        nyquist_value = absolute[nyquist]
        keep = ~np.isclose(values, nyquist_value, rtol=1e-12, atol=1e-14)
        remap = np.full(values.shape, -1, dtype=np.int64)
        remap[keep] = np.arange(np.sum(keep), dtype=np.int64)
        return values[keep], remap[groups]
    return values, groups


def _mode_samples(modes: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    selected = modes[:, :, y, x]
    return np.transpose(selected, (0, 2, 1)).reshape(-1, modes.shape[1])


def _single_cube_samples(modes: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    y = coordinates[:, 0]
    x = coordinates[:, 1]
    return np.asarray(modes[0][:, y, x].T, dtype=np.complex128)


def _product_arrays(power_cube: np.ndarray, layout: Any) -> dict[str, np.ndarray]:
    product = aggregate_power_cube(power_cube, layout, selected=True)
    return {
        "mean": np.asarray(product.mean),
        "power_sum": np.asarray(product.power_sum),
        "fft_mode_counts": np.asarray(product.fft_mode_counts),
        "independent_mode_counts": np.asarray(product.independent_mode_counts),
    }


def _mask_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    foreground: np.ndarray,
    fft_counts: np.ndarray,
    mask: np.ndarray,
    *,
    independent_counts: np.ndarray | None = None,
    quick_tolerance: float,
    strict_tolerance: float,
    foreground_tolerance: float,
) -> dict[str, Any]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(estimate)
        & np.isfinite(truth)
        & np.isfinite(foreground)
        & (truth > 0.0)
        & (fft_counts > 0)
    )
    if not np.any(valid):
        return {"n_bins": 0, "independent_mode_count": 0}
    ratio = estimate[valid] / truth[valid]
    fg_ratio = foreground[valid] / truth[valid]
    weight = np.asarray(fft_counts[valid], dtype=np.float64)
    reported_counts = fft_counts if independent_counts is None else independent_counts
    difference = estimate[valid] - truth[valid]
    denominator = max(float(np.sum(weight * np.square(truth[valid]))), 1e-300)
    quick = (np.abs(ratio - 1.0) <= float(quick_tolerance)) & (
        fg_ratio <= float(foreground_tolerance)
    )
    strict = (np.abs(ratio - 1.0) <= float(strict_tolerance)) & (
        fg_ratio <= float(foreground_tolerance)
    )
    return {
        "n_bins": int(np.count_nonzero(valid)),
        "independent_mode_count": int(np.sum(reported_counts[valid])),
        "integrated_power_ratio": float(
            np.sum(weight * estimate[valid]) / max(float(np.sum(weight * truth[valid])), 1e-300)
        ),
        "count_weighted_relative_l2": float(
            math.sqrt(float(np.sum(weight * np.square(difference))) / denominator)
        ),
        "median_power_ratio": float(np.median(ratio)),
        "maximum_absolute_fractional_error": float(np.max(np.abs(ratio - 1.0))),
        "foreground_integrated_over_eor": float(
            np.sum(weight * foreground[valid]) / max(float(np.sum(weight * truth[valid])), 1e-300)
        ),
        "foreground_median_over_eor": float(np.median(fg_ratio)),
        "quick_pass_bins": int(np.count_nonzero(quick)),
        "strict_pass_bins": int(np.count_nonzero(strict)),
        "all_quick_pass": bool(np.all(quick)),
        "all_strict_pass": bool(np.all(strict)),
    }


def _make_reporting_masks(
    config: dict[str, Any],
    layout: Any,
    prior_score: np.ndarray,
    *,
    minimum_independent_modes: int,
    prior_clean_fraction: float,
) -> dict[str, np.ndarray]:
    support = np.asarray(layout.selected_independent_mode_counts) >= int(minimum_independent_modes)
    settings = config.get("reporting_masks", {})
    high_fraction = float(settings.get("high_kpar_fraction", 0.75))
    mid_low, mid_high = [
        float(value) for value in settings.get("mid_kperp_fraction_range", [0.2, 0.5])
    ]
    kpar_index = np.arange(layout.kpar_values.size, dtype=np.int64)[None, :]
    high_start = int(math.ceil(high_fraction * max(0, layout.kpar_values.size - 1)))
    high_kpar = kpar_index >= high_start
    kperp = np.asarray(layout.kperp_centers, dtype=np.float64)[:, None]
    lower = float(layout.kperp_edges[0])
    width = float(layout.kperp_edges[-1] - layout.kperp_edges[0])
    mid_kperp = (kperp >= lower + mid_low * width) & (kperp <= lower + mid_high * width)
    standard = support & (np.asarray(layout.selected_fft_mode_counts) > 0)
    clean_candidates = np.flatnonzero(standard.reshape(-1) & np.isfinite(prior_score.reshape(-1)))
    clean = np.zeros_like(standard, dtype=bool).reshape(-1)
    if clean_candidates.size:
        retain = max(1, int(math.ceil(float(prior_clean_fraction) * clean_candidates.size)))
        order = clean_candidates[np.argsort(prior_score.reshape(-1)[clean_candidates])]
        clean[order[:retain]] = True
    return {
        "standard_window": standard,
        "high_kpar": standard & high_kpar,
        "high_kpar_mid_kperp": standard & high_kpar & mid_kperp,
        "prior_cleanest_fraction": clean.reshape(standard.shape),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    manifest_path = args.bank_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "partial_window_covariance_operator_bank":
        raise ValueError("unsupported covariance bank schema")
    if manifest.get("analysis_contract_sha256") != resolved.contract.analysis_contract_sha256:
        raise ValueError("bank and estimator analysis contracts differ")
    if manifest.get("operator_identity", {}).get("frequency_interpolation") is not False:
        raise ValueError("the estimator refuses a frequency-interpolated operator")

    products = manifest["products"]
    exact_fg = _load_product(args.bank_dir, products["exact_dirty_fg"])
    exact_eor = _load_product(args.bank_dir, products["exact_dirty_eor"])
    operator_fg = _load_product(args.bank_dir, products["operator_dirty_fg"])
    heldout_unit = _load_product(args.bank_dir, products["fg_error_heldout_unit"])
    foreground_draws_unit = _load_product(args.bank_dir, products["fg_error_draws_unit"])
    eor_probe_cubes = {
        float(ell): _load_product(args.bank_dir, record)
        for ell, record in products["eor_probes"].items()
    }
    shape = tuple(int(value) for value in resolved.contract.full_layout.cube_shape)
    for name, values in {
        "exact_fg": exact_fg,
        "exact_eor": exact_eor,
        "operator_fg": operator_fg,
        "heldout_unit": heldout_unit,
    }.items():
        if values.shape != shape:
            raise ValueError(f"{name} shape {values.shape} != {shape}")

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
    independent_coordinates = independent_spatial_coordinates(height, width)
    ky = 2.0 * math.pi * np.fft.fftfreq(height, d=float(layout.dy_mpc))
    kx = 2.0 * math.pi * np.fft.fftfreq(width, d=float(layout.dx_mpc))
    coordinate_kperp = np.sqrt(
        np.square(ky[independent_coordinates[:, 0]])
        + np.square(kx[independent_coordinates[:, 1]])
    )
    coordinate_bins = np.searchsorted(layout.kperp_edges, coordinate_kperp, side="right") - 1
    on_right = np.isclose(coordinate_kperp, layout.kperp_edges[-1], rtol=1e-12, atol=1e-14)
    coordinate_bins[on_right] = layout.kperp_edges.size - 2
    coordinate_valid = (
        (coordinate_kperp <= float(layout.transverse_circle_max))
        & (coordinate_bins >= 0)
        & (coordinate_bins < layout.kperp_centers.size)
    )
    frequencies_mhz = np.asarray(manifest["frequencies_mhz"], dtype=np.float64)
    if frequencies_mhz.size != nfreq:
        raise ValueError("manifest frequency count mismatch")

    diagonal_shrinkage = float(args.diagonal_shrinkage)
    eigen_floor = float(args.eigen_floor_fraction)
    base_covariances: dict[int, tuple[np.ndarray, dict[float, np.ndarray]]] = {}
    covariance_stats: dict[str, Any] = {}
    prior_score = np.full(layout.shape_2d, np.nan, dtype=np.float64)
    _, radial_groups = _native_kpar_groups(nfreq)
    for kperp_bin in range(layout.kperp_centers.size):
        coordinates = independent_coordinates[coordinate_valid & (coordinate_bins == kperp_bin)]
        if coordinates.shape[0] < int(args.minimum_independent_modes):
            continue
        fg_samples = _mode_samples(foreground_draw_modes, coordinates)
        fg_covariance, fg_stats = regularize_covariance(
            second_moment_covariance(fg_samples),
            diagonal_shrinkage=diagonal_shrinkage,
            eigen_floor_fraction=eigen_floor,
        )
        eor_covariances: dict[float, np.ndarray] = {}
        eor_stats: dict[str, Any] = {}
        eor_radial_shapes = []
        for ell, modes in eor_probe_modes.items():
            covariance, stats = regularize_covariance(
                second_moment_covariance(_mode_samples(modes, coordinates)),
                diagonal_shrinkage=diagonal_shrinkage,
                eigen_floor_fraction=eigen_floor,
            )
            eor_covariances[float(ell)] = covariance
            eor_stats[f"{float(ell):.8g}"] = stats
            radial = _radial_covariance_power(covariance, radial_window)
            eor_radial_shapes.append(radial / float(np.sum(radial)))
        base_covariances[kperp_bin] = (fg_covariance, eor_covariances)
        fg_radial = _radial_covariance_power(fg_covariance, radial_window)
        fg_radial /= float(np.sum(fg_radial))
        eor_radial = np.mean(np.stack(eor_radial_shapes), axis=0)
        for radial_group in range(layout.kpar_values.size):
            radial_indices = np.flatnonzero(radial_groups == radial_group)
            if radial_indices.size:
                prior_score[kperp_bin, radial_group] = float(
                    np.mean(fg_radial[radial_indices])
                    / max(float(np.mean(eor_radial[radial_indices])), 1e-300)
                )
        covariance_stats[str(kperp_bin)] = {
            "independent_spatial_modes": int(coordinates.shape[0]),
            "foreground": fg_stats,
            "eor": eor_stats,
        }

    reporting_masks = _make_reporting_masks(
        config,
        layout,
        prior_score,
        minimum_independent_modes=int(args.minimum_independent_modes),
        prior_clean_fraction=float(args.prior_clean_fraction),
    )
    mask_signature = hashlib.sha256(
        b"".join(
            name.encode("utf-8") + np.asarray(mask, dtype=np.uint8).tobytes()
            for name, mask in sorted(reporting_masks.items())
        )
    ).hexdigest()

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
    precision_levels = _parse_floats(args.precision_levels)
    if any(value <= 0.0 for value in precision_levels):
        raise ValueError("precision levels must be positive")

    result_by_precision: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {
        "truth_eor_mean": truth_product["mean"],
        "truth_eor_power_sum": truth_product["power_sum"],
        "selected_independent_mode_counts": truth_product["independent_mode_counts"],
        "prior_shape_score": prior_score,
        "kperp_edges_mpc_inv": np.asarray(layout.kperp_edges),
        "kperp_centers_mpc_inv": np.asarray(layout.kperp_centers),
        "kpar_values_mpc_inv": np.asarray(layout.kpar_values),
    }
    for mask_name, mask in reporting_masks.items():
        npz_payload[f"mask_{mask_name}"] = np.asarray(mask, dtype=np.uint8)

    for precision in precision_levels:
        label = f"p{float(precision):.6g}".replace(".", "p").replace("-", "m")
        prior_mean = operator_fg + float(precision) * heldout_unit
        foreground_residual = exact_fg - prior_mean
        total_residual = foreground_residual + exact_eor
        total_modes = spatial_fourier_modes(
            total_residual,
            demean_mode=demean_mode,
            spatial_window_y=spatial_window_y,
            spatial_window_x=spatial_window_x,
        )
        foreground_modes = spatial_fourier_modes(
            foreground_residual,
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
        output_cubes = {
            "posterior_second_moment": np.zeros(shape, dtype=np.float64),
            "posterior_mean": np.zeros(shape, dtype=np.float64),
            "foreground_leakage_mean": np.zeros(shape, dtype=np.float64),
            "eor_transfer_mean": np.zeros(shape, dtype=np.float64),
        }
        hyperparameters: dict[str, Any] = {}
        for kperp_bin, (fg_covariance_unit, eor_covariances) in base_covariances.items():
            coordinates = independent_coordinates[coordinate_valid & (coordinate_bins == kperp_bin)]
            total_samples = _single_cube_samples(total_modes, coordinates)
            foreground_samples = _single_cube_samples(foreground_modes, coordinates)
            eor_samples = _single_cube_samples(exact_eor_modes, coordinates)
            fit = fit_covariance_grid(
                total_samples,
                float(precision) ** 2 * fg_covariance_unit,
                eor_covariances,
                q_fg_grid=q_fg_grid,
                q_eor_k2_grid=q_eor_grid,
                q_fg_log_sigma=float(args.q_fg_log_sigma),
            )
            powers = posterior_radial_powers(
                total_samples,
                foreground_samples,
                eor_samples,
                float(precision) ** 2 * fg_covariance_unit,
                eor_covariances,
                fit,
                radial_window=radial_window,
            )
            for name, values in powers.items():
                fill_conjugate_power_cube(
                    output_cubes[name],
                    coordinates,
                    float(resolved.contract.power_scale) * values,
                )
            hyperparameters[str(kperp_bin)] = fit_to_dict(fit)
        estimated_products = {
            name: _product_arrays(power_cube, layout)
            for name, power_cube in output_cubes.items()
        }
        metrics = {
            mask_name: _mask_metrics(
                estimated_products["posterior_second_moment"]["mean"],
                truth_product["mean"],
                estimated_products["foreground_leakage_mean"]["mean"],
                truth_product["fft_mode_counts"],
                mask,
                independent_counts=truth_product["independent_mode_counts"],
                quick_tolerance=float(args.quick_relative_tolerance),
                strict_tolerance=float(args.strict_relative_tolerance),
                foreground_tolerance=float(args.foreground_leakage_tolerance),
            )
            for mask_name, mask in reporting_masks.items()
        }
        result_by_precision[f"{float(precision):.8g}"] = {
            "foreground_precision": {
                "fractional_amplitude_rms": float(precision),
                "spectral_index_rms": 2.0 * float(precision),
                "astrometric_shift_rms_px": 4.0 * float(precision),
                "unresolved_confusion_rms_fraction": 0.3 * float(precision),
            },
            "hyperparameters_by_kperp": hyperparameters,
            "metrics": metrics,
            "operator_foreground_closure_residual_over_eor_rms": float(
                np.sqrt(np.mean(np.square(exact_fg - operator_fg)))
                / max(float(np.sqrt(np.mean(np.square(exact_eor)))), 1e-300)
            ),
            "foreground_residual_over_eor_rms": float(
                np.sqrt(np.mean(np.square(foreground_residual)))
                / max(float(np.sqrt(np.mean(np.square(exact_eor)))), 1e-300)
            ),
        }
        for name, product in estimated_products.items():
            npz_payload[f"{label}_{name}_mean"] = product["mean"]
            npz_payload[f"{label}_{name}_power_sum"] = product["power_sum"]
        print(
            json.dumps(
                {
                    "event": "precision_done",
                    "precision": float(precision),
                    "metrics": metrics,
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = {
        "schema": "partial_window_covariance_ps2d_result",
        "schema_version": 1,
        "created_at": _now(),
        "method": "per_kperp_frequency_covariance_likelihood_with_fg_and_eor_hyperparameter_marginalization",
        "scientific_target": "dirty_eor_ps2d_posterior_second_moment",
        "map_recovery_claim": False,
        "noise_model": "none",
        "fit_uses_eor_truth": False,
        "fit_uses_hidden_foreground_component_after_prior_construction": False,
        "foreground_prior_is_truth_derived_emulator": True,
        "truth_use": (
            "foreground truth is degraded once to emulate an external observed prior; "
            "EoR truth and the hidden foreground residual are used only in frozen post-fit diagnostics"
        ),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "bank_manifest": str(manifest_path),
        "bank_manifest_sha256": _sha256(manifest_path),
        "operator_identity": manifest["operator_identity"],
        "operator_closure": manifest["closure"],
        "frequency_covariance": {
            "foreground_source": "operator-propagated observation-precision emulator ensemble",
            "eor_source": "operator-propagated stationary probes",
            "eor_lengths_mhz": sorted(eor_probe_modes),
            "q_eor_k2_grid": q_eor_grid.tolist(),
            "q_fg_grid": q_fg_grid,
            "q_fg_log_sigma": float(args.q_fg_log_sigma),
            "diagonal_shrinkage": diagonal_shrinkage,
            "eigen_floor_fraction": eigen_floor,
            "transverse_assumption": "independent kperp annuli with angular pooling; no fixed morphology template",
        },
        "reporting_masks": {
            "selection_uses_truth": False,
            "mask_sha256": mask_signature,
            "minimum_independent_modes": int(args.minimum_independent_modes),
            "prior_clean_fraction": float(args.prior_clean_fraction),
            "counts": {
                name: int(np.count_nonzero(mask)) for name, mask in reporting_masks.items()
            },
        },
        "gates": {
            "quick_relative_tolerance": float(args.quick_relative_tolerance),
            "strict_relative_tolerance": float(args.strict_relative_tolerance),
            "foreground_leakage_tolerance": float(args.foreground_leakage_tolerance),
        },
        "covariance_stats_by_kperp": covariance_stats,
        "results": result_by_precision,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(args.out_dir / "result.json", output)
    _atomic_npz(args.out_dir / "result.npz", npz_payload)
    print(
        json.dumps(
            {
                "event": "partial_window_covariance_ps2d_done",
                "result_json": str(args.out_dir / "result.json"),
                "result_npz": str(args.out_dir / "result.npz"),
                "elapsed_seconds": output["elapsed_seconds"],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
