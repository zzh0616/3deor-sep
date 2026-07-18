#!/usr/bin/env python3
"""Evaluate observation-anchored foreground priors through the cached operator.

The benchmark uses a deep observed catalog and an observed spatial spectral-
index map only to construct a same-sky foreground truth.  The estimator sees a
shallower catalog basis, a Haslam mean map with one global spectral index, and
declared finite prior widths.  It fits control Fourier modes only; guard and
science modes remain held out.  Simulation EoR component labels are used only
for post-fit transfer and leakage diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = Path(os.environ.get("FG_RMW_CODE_DIR", str(SCRIPT_DIR.parent))).resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_template_free_sky_smooth_projection as flat  # noqa: E402
from build_observed_sky_template_cheb_prior import (  # noqa: E402
    _cheb_design,
    _insert_catalog_jypix,
    _load_diffuse_base_map,
    _pixel_arcsec_from_header,
    _read_catalog,
    _target_header,
    _target_shape,
)
from observed_prior_separation import (  # noqa: E402
    partition_of_unity_grid,
    posterior_mean_for_data,
    posterior_predictive_score,
    prior_predictive_feature_scale,
    relative_linearity_error,
    solve_linear_gaussian_control,
)
from ps2d_v2_estimator import (  # noqa: E402
    EstimatorBandGeometry,
    TorchBandpowerTransform,
    build_mode_first_estimator_contract,
)


@dataclass(frozen=True)
class SourceGroupSpec:
    label: str
    coefficient_path: Path
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class NuisanceBasisSpec:
    name: str
    family: str
    variation: str
    prior_std: float
    source_group: int | None = None
    diffuse_cell: int | None = None


def _parse_grid(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in str(value).lower().replace("x", ",").split(",") if item.strip()]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("grid must be NY,NX with positive integers")
    return int(parts[0]), int(parts[1])


def _parse_bands(value: str) -> list[int] | None:
    if str(value).strip().lower() in {"", "all"}:
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _resolve_manifest_path(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _catalog_reader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        catalog_flux_unit="Jy",
        catalog_min_flux_jy=float(args.truth_catalog_min_flux_jy),
        catalog_ref_freq_mhz=float(args.catalog_ref_freq_mhz),
        catalog_spectral_index_default=float(args.catalog_spectral_index_default),
        catalog_curvature_default=0.0,
    )


def _catalog_cube_k(
    path: Path,
    *,
    args: argparse.Namespace,
    frequencies: np.ndarray,
    header: fits.Header,
    shape: tuple[int, int],
    pixel_arcsec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    catalog = _read_catalog(path, _catalog_reader_args(args))
    cube, rows = _catalog_to_cube_k(
        catalog,
        frequencies=frequencies,
        header=header,
        shape=shape,
        pixel_arcsec=pixel_arcsec,
        insert_mode=str(args.catalog_insert_mode),
    )
    return cube, {
        "path": str(path),
        "catalog_rows_after_flux_filter": int(len(catalog["ra_deg"])),
        "per_frequency": rows,
    }


def _catalog_to_cube_k(
    catalog: dict[str, np.ndarray],
    *,
    frequencies: np.ndarray,
    header: fits.Header,
    shape: tuple[int, int],
    pixel_arcsec: float,
    insert_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cube = np.zeros((len(frequencies), shape[0], shape[1]), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, frequency in enumerate(frequencies.tolist()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            plane_jy, stats = _insert_catalog_jypix(
                catalog=catalog,
                freq_mhz=float(frequency),
                header=header,
                shape=shape,
                insert_mode=str(insert_mode),
            )
        conversion = flat._k_to_jy_per_pixel(float(frequency), float(pixel_arcsec))
        cube[index] = plane_jy / conversion
        rows.append({"freq_mhz": float(frequency), **stats})
    return cube, rows


def _source_basis_manifest(
    path: Path,
    frequencies: np.ndarray,
    shape: tuple[int, int],
) -> tuple[list[SourceGroupSpec], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = payload.get("settings", {})
    if str(payload.get("method")) != "observed_catalog_source_support_basis":
        raise ValueError("source basis manifest has an unexpected method")
    if bool(settings.get("include_all_inside", False)):
        raise ValueError("include_all_inside duplicates grouped sources and is not valid here")
    if not np.allclose(settings.get("freqs_mhz", []), frequencies, rtol=0.0, atol=1.0e-10):
        raise ValueError("source basis frequencies do not match the estimator config")
    if tuple(int(value) for value in settings.get("image_shape", [])) != tuple(shape):
        raise ValueError("source basis image shape does not match the operator input")
    groups = [
        SourceGroupSpec(
            label=str(item.get("label", f"group_{index:03d}")),
            coefficient_path=_resolve_manifest_path(str(item["path"]), path),
            source_indices=tuple(int(value) for value in item.get("source_indices", [])),
        )
        for index, item in enumerate(payload.get("basis", []))
    ]
    paths = [group.coefficient_path for group in groups]
    if not groups or any(not item.exists() for item in paths):
        missing = [str(item) for item in paths if not item.exists()]
        raise FileNotFoundError(f"source basis is empty or missing files: {missing[:4]}")
    indexed = [index for group in groups for index in group.source_indices]
    if indexed and len(indexed) != len(set(indexed)):
        raise ValueError("source basis groups contain duplicate catalog rows")
    return groups, payload


def _prior_catalog_from_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
    groups: Sequence[SourceGroupSpec],
) -> tuple[dict[str, np.ndarray] | None, Path | None, str]:
    settings = payload.get("settings", {})
    insert_mode = str(settings.get("catalog_insert_mode", "bilinear"))
    if not any(group.source_indices for group in groups):
        return None, None, insert_mode
    raw_path = settings.get("catalog_csv")
    if not raw_path:
        raise ValueError("exact source membership requires settings.catalog_csv")
    catalog_path = _resolve_manifest_path(str(raw_path), manifest_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"prior source catalog is missing: {catalog_path}")
    reader_args = argparse.Namespace(
        catalog_flux_unit=str(settings.get("catalog_flux_unit", "Jy")),
        catalog_min_flux_jy=float(settings.get("catalog_min_flux_jy", 0.0)),
        catalog_ref_freq_mhz=float(settings.get("catalog_ref_freq_mhz", 150.0)),
        catalog_spectral_index_default=float(
            settings.get("catalog_spectral_index_default", -0.8)
        ),
        catalog_curvature_default=float(settings.get("catalog_curvature_default", 0.0)),
    )
    return _read_catalog(catalog_path, reader_args), catalog_path, insert_mode


def _source_group_cube(
    group: SourceGroupSpec,
    frequencies: np.ndarray,
    shape: tuple[int, int],
    *,
    prior_catalog: dict[str, np.ndarray] | None,
    header: fits.Header,
    pixel_arcsec: float,
    insert_mode: str,
) -> np.ndarray:
    if group.source_indices:
        if prior_catalog is None:
            raise ValueError("exact source membership requires the prior catalog")
        indices = np.asarray(group.source_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(prior_catalog["ra_deg"])):
            raise ValueError(f"source group {group.label} has an invalid catalog row")
        subset = {key: np.asarray(value)[indices] for key, value in prior_catalog.items()}
        cube, _ = _catalog_to_cube_k(
            subset,
            frequencies=frequencies,
            header=header,
            shape=shape,
            pixel_arcsec=float(pixel_arcsec),
            insert_mode=str(insert_mode),
        )
        return cube
    coefficient = np.squeeze(np.asarray(fits.getdata(group.coefficient_path), dtype=np.float64))
    if coefficient.ndim != 3 or tuple(coefficient.shape[1:]) != tuple(shape):
        raise ValueError(
            f"Invalid source coefficient cube shape {coefficient.shape}: {group.coefficient_path}"
        )
    design = _cheb_design(frequencies.tolist(), int(coefficient.shape[0]) - 1)
    return (design @ coefficient.reshape(coefficient.shape[0], -1)).reshape(
        len(frequencies), shape[0], shape[1]
    )


def _diffuse_cubes(
    *,
    args: argparse.Namespace,
    frequencies: np.ndarray,
    header: fits.Header,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base, base_stats = _load_diffuse_base_map(
        args.diffuse_base_fits,
        header,
        shape,
        unit=str(args.diffuse_unit),
    )
    index_map, index_stats = _load_diffuse_base_map(
        args.truth_diffuse_index_fits,
        header,
        shape,
        unit="K",
    )
    if str(args.truth_diffuse_index_convention) == "positive_beta":
        truth_alpha = -np.abs(index_map)
    else:
        truth_alpha = index_map
    finite_alpha = truth_alpha[np.isfinite(truth_alpha)]
    if finite_alpha.size == 0:
        raise ValueError("truth diffuse index map has no finite values")
    if float(np.median(finite_alpha)) > -1.0 or float(np.median(finite_alpha)) < -4.5:
        raise ValueError(
            "truth diffuse spectral-index convention is implausible: "
            f"median alpha={float(np.median(finite_alpha)):.4g}"
        )
    prior = np.zeros((len(frequencies), shape[0], shape[1]), dtype=np.float64)
    truth = np.zeros_like(prior)
    for index, frequency in enumerate(frequencies.tolist()):
        ratio = float(frequency) / float(args.diffuse_ref_freq_mhz)
        prior[index] = base * ratio ** float(args.prior_diffuse_spectral_index)
        truth[index] = base * np.exp(truth_alpha * math.log(ratio))
    return prior, truth, {
        "base_map": str(args.diffuse_base_fits),
        "base_map_stats": base_stats,
        "index_map": str(args.truth_diffuse_index_fits),
        "index_map_stats": index_stats,
        "index_convention": str(args.truth_diffuse_index_convention),
        "truth_alpha_min": float(np.min(finite_alpha)),
        "truth_alpha_median": float(np.median(finite_alpha)),
        "truth_alpha_max": float(np.max(finite_alpha)),
        "truth_alpha_std": float(np.std(finite_alpha)),
        "prior_global_alpha": float(args.prior_diffuse_spectral_index),
    }


def _build_specs(
    source_groups: Sequence[SourceGroupSpec],
    *,
    args: argparse.Namespace,
    diffuse_cell_count: int,
) -> list[NuisanceBasisSpec]:
    specs: list[NuisanceBasisSpec] = []
    if bool(args.catalog_nuisance):
        for index, _group in enumerate(source_groups):
            specs.append(
                NuisanceBasisSpec(
                    name=f"catalog_group_{index:03d}_amplitude",
                    family="catalog",
                    variation="fractional_amplitude",
                    prior_std=float(args.catalog_amplitude_prior_std),
                    source_group=int(index),
                )
            )
            specs.append(
                NuisanceBasisSpec(
                    name=f"catalog_group_{index:03d}_slope",
                    family="catalog",
                    variation="spectral_index",
                    prior_std=float(args.catalog_slope_prior_std),
                    source_group=int(index),
                )
            )
    if bool(args.diffuse_nuisance):
        for cell in range(int(diffuse_cell_count)):
            specs.append(
                NuisanceBasisSpec(
                    name=f"diffuse_cell_{cell:03d}_amplitude",
                    family="diffuse",
                    variation="fractional_amplitude",
                    prior_std=float(args.diffuse_amplitude_prior_std),
                    diffuse_cell=int(cell),
                )
            )
            specs.append(
                NuisanceBasisSpec(
                    name=f"diffuse_cell_{cell:03d}_slope",
                    family="diffuse",
                    variation="spectral_index",
                    prior_std=float(args.diffuse_slope_prior_std),
                    diffuse_cell=int(cell),
                )
            )
    if not specs:
        raise ValueError("at least one nuisance family must be enabled")
    if any(not np.isfinite(spec.prior_std) or spec.prior_std <= 0.0 for spec in specs):
        raise ValueError("all nuisance prior standard deviations must be finite and positive")
    return specs


def _basis_cube(
    spec: NuisanceBasisSpec,
    *,
    frequencies: np.ndarray,
    shape: tuple[int, int],
    source_groups: Sequence[SourceGroupSpec],
    prior_catalog: dict[str, np.ndarray] | None,
    header: fits.Header,
    pixel_arcsec: float,
    catalog_insert_mode: str,
    diffuse_prior: np.ndarray,
    diffuse_weights: np.ndarray,
    pivot_mhz: float,
) -> np.ndarray:
    if spec.family == "catalog":
        if spec.source_group is None:
            raise AssertionError("catalog basis is missing its source group")
        cube = _source_group_cube(
            source_groups[int(spec.source_group)],
            frequencies,
            shape,
            prior_catalog=prior_catalog,
            header=header,
            pixel_arcsec=float(pixel_arcsec),
            insert_mode=str(catalog_insert_mode),
        )
    elif spec.family == "diffuse":
        if spec.diffuse_cell is None:
            raise AssertionError("diffuse basis is missing its cell")
        cube = diffuse_prior * diffuse_weights[int(spec.diffuse_cell)][None, :, :]
    else:
        raise ValueError(f"Unknown basis family: {spec.family}")
    if spec.variation == "spectral_index":
        log_ratio = np.log(frequencies / float(pivot_mhz))
        cube = cube * log_ratio[:, None, None]
    elif spec.variation != "fractional_amplitude":
        raise ValueError(f"Unknown basis variation: {spec.variation}")
    return np.asarray(cube * float(spec.prior_std), dtype=np.float64)


def _prior_catalog_mean(
    source_groups: Sequence[SourceGroupSpec],
    frequencies: np.ndarray,
    shape: tuple[int, int],
    *,
    prior_catalog: dict[str, np.ndarray] | None,
    header: fits.Header,
    pixel_arcsec: float,
    insert_mode: str,
) -> np.ndarray:
    total = np.zeros((len(frequencies), shape[0], shape[1]), dtype=np.float64)
    for group in source_groups:
        total += _source_group_cube(
            group,
            frequencies,
            shape,
            prior_catalog=prior_catalog,
            header=header,
            pixel_arcsec=float(pixel_arcsec),
            insert_mode=str(insert_mode),
        )
    return total


def _selected_features(
    cubes: torch.Tensor,
    transform: TorchBandpowerTransform,
    indices: torch.Tensor,
) -> torch.Tensor:
    spectrum = transform.fourier(cubes).reshape(cubes.shape[0], -1).index_select(1, indices)
    return torch.cat((spectrum.real, spectrum.imag), dim=1)


def _build_feature_design(
    specs: Sequence[NuisanceBasisSpec],
    *,
    args: argparse.Namespace,
    frequencies: np.ndarray,
    shape: tuple[int, int],
    source_groups: Sequence[SourceGroupSpec],
    prior_catalog: dict[str, np.ndarray] | None,
    header: fits.Header,
    pixel_arcsec: float,
    catalog_insert_mode: str,
    diffuse_prior: np.ndarray,
    diffuse_weights: np.ndarray,
    k_to_jy: torch.Tensor,
    forward_operator: Any,
    full_transform: TorchBandpowerTransform,
    control_indices: torch.Tensor,
    guard_indices: torch.Tensor,
    device: torch.device,
    started: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    control_blocks: list[torch.Tensor] = []
    guard_blocks: list[torch.Tensor] = []
    batch_size = max(int(args.basis_batch_size), 1)
    for start in range(0, len(specs), batch_size):
        stop = min(start + batch_size, len(specs))
        cube = np.stack(
            [
                _basis_cube(
                    spec,
                    frequencies=frequencies,
                    shape=shape,
                    source_groups=source_groups,
                    prior_catalog=prior_catalog,
                    header=header,
                    pixel_arcsec=float(pixel_arcsec),
                    catalog_insert_mode=str(catalog_insert_mode),
                    diffuse_prior=diffuse_prior,
                    diffuse_weights=diffuse_weights,
                    pivot_mhz=float(args.nuisance_pivot_mhz),
                )
                for spec in specs[start:stop]
            ],
            axis=0,
        )
        sky = torch.as_tensor(cube, dtype=torch.float64, device=device)
        with torch.no_grad():
            dirty = forward_operator(sky * k_to_jy[None, :, None, None])
            control_blocks.append(
                _selected_features(dirty, full_transform, control_indices).T.cpu()
            )
            guard_blocks.append(
                _selected_features(dirty, full_transform, guard_indices).T.cpu()
            )
        del sky, dirty, cube
        print(
            json.dumps(
                {
                    "event": "observed_prior_basis_batch_done",
                    "basis_done": int(stop),
                    "basis_total": int(len(specs)),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    control = torch.cat(control_blocks, dim=1).to(device=device, dtype=torch.float64)
    guard = torch.cat(guard_blocks, dim=1).to(device=device, dtype=torch.float64)
    return control, guard


def _compose_correction_sky(
    specs: Sequence[NuisanceBasisSpec],
    coefficients: torch.Tensor,
    *,
    frequencies: np.ndarray,
    shape: tuple[int, int],
    source_groups: Sequence[SourceGroupSpec],
    prior_catalog: dict[str, np.ndarray] | None,
    header: fits.Header,
    pixel_arcsec: float,
    catalog_insert_mode: str,
    diffuse_prior: np.ndarray,
    diffuse_weights: np.ndarray,
    pivot_mhz: float,
) -> np.ndarray:
    values = np.asarray(coefficients.detach().cpu(), dtype=np.float64)
    if values.shape != (len(specs),):
        raise ValueError("coefficient vector has the wrong shape")
    total = np.zeros((len(frequencies), shape[0], shape[1]), dtype=np.float64)
    for value, spec in zip(values.tolist(), specs):
        if value == 0.0:
            continue
        total += float(value) * _basis_cube(
            spec,
            frequencies=frequencies,
            shape=shape,
            source_groups=source_groups,
            prior_catalog=prior_catalog,
            header=header,
            pixel_arcsec=float(pixel_arcsec),
            catalog_insert_mode=str(catalog_insert_mode),
            diffuse_prior=diffuse_prior,
            diffuse_weights=diffuse_weights,
            pivot_mhz=float(pivot_mhz),
        )
    return total


def _weighted_metrics(
    estimate_power: np.ndarray,
    truth_power: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    estimate = estimate_power[indices]
    truth = truth_power[indices]
    weight = counts[indices]
    difference = estimate - truth
    ratio = estimate / np.maximum(truth, 1.0e-300)
    return {
        "band_count": int(indices.size),
        "count_weighted_relative_l2": float(
            np.sqrt(
                np.sum(weight * difference * difference)
                / max(float(np.sum(weight * truth * truth)), 1.0e-300)
            )
        ),
        "exact_mode_power_sum_ratio": float(
            np.sum(weight * estimate) / max(float(np.sum(weight * truth)), 1.0e-300)
        ),
        "median_band_ratio": float(np.median(ratio)),
        "maximum_per_band_relative_error": float(np.max(np.abs(ratio - 1.0))),
    }


def _power_diagnostics(
    *,
    total_residual: torch.Tensor,
    foreground_residual: torch.Tensor,
    eor_transfer: torch.Tensor,
    eor_truth: torch.Tensor,
    transform: TorchBandpowerTransform,
    geometry: EstimatorBandGeometry,
    target_bands: list[int] | None,
    quick_tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cubes = torch.stack((total_residual, foreground_residual, eor_transfer, eor_truth), dim=0)
    power = np.asarray(transform(cubes).detach().cpu(), dtype=np.float64)
    total_power, foreground_power, transfer_power, truth_power = power
    all_indices = np.arange(geometry.band_count, dtype=np.int64)
    selected = all_indices if target_bands is None else np.asarray(target_bands, dtype=np.int64)
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= geometry.band_count):
        raise ValueError("target bands are empty or outside the science geometry")
    transfer_ratio = transfer_power / np.maximum(truth_power, 1.0e-300)
    foreground_ratio = foreground_power / np.maximum(truth_power, 1.0e-300)
    total_ratio = total_power / np.maximum(truth_power, 1.0e-300)
    strict = (
        (np.abs(transfer_ratio - 1.0) <= 0.10)
        & (foreground_ratio <= 0.10)
        & (np.abs(total_ratio - 1.0) <= 0.10)
    )
    quick = (
        (np.abs(transfer_ratio - 1.0) <= float(quick_tolerance))
        & (foreground_ratio <= float(quick_tolerance))
        & (np.abs(total_ratio - 1.0) <= float(quick_tolerance))
    )
    rows: list[dict[str, Any]] = []
    for band in all_indices.tolist():
        kperp_index = int(geometry.active_kperp_indices[band])
        kpar_index = int(geometry.active_kpar_indices[band])
        rows.append(
            {
                "band": int(band),
                "kperp_h_mpc": float(geometry.kperp_centers[kperp_index]),
                "kpar_h_mpc": float(geometry.kpar_values[kpar_index]),
                "fft_mode_count": int(geometry.counts[band]),
                "truth_eor_power": float(truth_power[band]),
                "total_residual_power_ratio": float(total_ratio[band]),
                "foreground_residual_power_ratio": float(foreground_ratio[band]),
                "pure_eor_transfer_power_ratio": float(transfer_ratio[band]),
                "strict_gate_pass": bool(strict[band]),
                "quick_gate_pass": bool(quick[band]),
            }
        )
    diagnostics = {
        "target_bands": selected.tolist(),
        "total_residual_vs_eor": _weighted_metrics(total_power, truth_power, geometry.counts, selected),
        "pure_eor_transfer_vs_eor": _weighted_metrics(transfer_power, truth_power, geometry.counts, selected),
        "foreground_false_eor": {
            "count_weighted_power_ratio": float(
                np.sum(geometry.counts[selected] * foreground_power[selected])
                / max(float(np.sum(geometry.counts[selected] * truth_power[selected])), 1.0e-300)
            ),
            "median_band_power_ratio": float(np.median(foreground_ratio[selected])),
            "maximum_band_power_ratio": float(np.max(foreground_ratio[selected])),
        },
        "gates": {
            "strict_definition": "transfer,total within 10%; foreground power <=10% of EoR",
            "quick_tolerance": float(quick_tolerance),
            "strict_pass_count": int(np.sum(strict[selected])),
            "quick_pass_count": int(np.sum(quick[selected])),
            "target_band_count": int(selected.size),
            "strict_all_target_bands": bool(np.all(strict[selected])),
            "quick_all_target_bands": bool(np.all(quick[selected])),
        },
        "bands": rows,
    }
    arrays = {
        "total_residual_power": total_power,
        "foreground_residual_power": foreground_power,
        "pure_eor_transfer_power": transfer_power,
        "truth_eor_power": truth_power,
        "strict_gate": strict,
        "quick_gate": quick,
    }
    return diagnostics, arrays


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-npz", type=Path, required=True)
    parser.add_argument("--tile-cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-dirty", type=Path, required=True)
    parser.add_argument("--truth-eor-pattern", required=True)
    parser.add_argument("--truth-catalog-csv", type=Path, required=True)
    parser.add_argument("--prior-source-basis-manifest", type=Path, required=True)
    parser.add_argument("--diffuse-base-fits", type=Path, required=True)
    parser.add_argument("--truth-diffuse-index-fits", type=Path, required=True)
    parser.add_argument("--truth-diffuse-index-convention", choices=("alpha", "positive_beta"), default="positive_beta")
    parser.add_argument("--diffuse-ref-freq-mhz", type=float, default=408.0)
    parser.add_argument("--diffuse-unit", choices=("K", "mK"), default="K")
    parser.add_argument("--prior-diffuse-spectral-index", type=float, default=-2.55)
    parser.add_argument("--diffuse-grid", type=_parse_grid, default=(4, 4))
    parser.add_argument("--catalog-amplitude-prior-std", type=float, default=0.20)
    parser.add_argument("--catalog-slope-prior-std", type=float, default=0.30)
    parser.add_argument("--diffuse-amplitude-prior-std", type=float, default=0.30)
    parser.add_argument("--diffuse-slope-prior-std", type=float, default=0.30)
    parser.add_argument("--catalog-nuisance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffuse-nuisance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nuisance-pivot-mhz", type=float, default=119.3)
    parser.add_argument("--truth-catalog-min-flux-jy", type=float, default=0.005)
    parser.add_argument("--catalog-ref-freq-mhz", type=float, default=150.0)
    parser.add_argument("--catalog-spectral-index-default", type=float, default=-0.8)
    parser.add_argument(
        "--catalog-insert-mode",
        choices=("bilinear", "nearest", "gaussian_observed", "gaussian_deconv"),
        default="gaussian_deconv",
    )
    parser.add_argument("--basis-batch-size", type=int, default=4)
    parser.add_argument("--feature-floor-quantile", type=float, default=0.10)
    parser.add_argument("--feature-floor-fraction-of-max", type=float, default=1.0e-6)
    parser.add_argument("--target-bands", type=_parse_bands, default=None)
    parser.add_argument("--quick-gate-tolerance", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-tiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-cubes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = build_mode_first_estimator_contract(config)
    analysis = contract.analysis
    output_shape = tuple(int(value) for value in analysis.full_layout.cube_shape)
    frequencies = np.asarray(contract.resolved.geometry["frequencies_mhz"], dtype=np.float64)
    device = torch.device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty((), dtype=torch.float64, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    forward_operator, operator_identity = flat._load_operator(
        design_path=args.design_npz,
        tile_cache_dir=args.tile_cache_dir,
        frequencies=frequencies,
        analysis_shape=output_shape,
        device=device,
        checkpoint_tiles=bool(args.checkpoint_tiles),
    )
    image_size = int(operator_identity["image_size"])
    header = _target_header(args.reference_dirty, image_size)
    input_shape = _target_shape(header)
    if input_shape != (image_size, image_size):
        raise ValueError("reference WCS does not resolve to the operator input shape")
    pixel_arcsec = _pixel_arcsec_from_header(header)
    if not math.isclose(pixel_arcsec, float(operator_identity["pixel_arcsec"]), rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("reference WCS pixel scale does not match the operator")

    source_groups, source_manifest = _source_basis_manifest(
        args.prior_source_basis_manifest,
        frequencies,
        input_shape,
    )
    prior_catalog_data, prior_catalog_path, prior_catalog_insert_mode = (
        _prior_catalog_from_manifest(
            args.prior_source_basis_manifest,
            source_manifest,
            source_groups,
        )
    )
    truth_catalog_sky, truth_catalog_stats = _catalog_cube_k(
        args.truth_catalog_csv,
        args=args,
        frequencies=frequencies,
        header=header,
        shape=input_shape,
        pixel_arcsec=pixel_arcsec,
    )
    prior_catalog_sky = _prior_catalog_mean(
        source_groups,
        frequencies,
        input_shape,
        prior_catalog=prior_catalog_data,
        header=header,
        pixel_arcsec=float(pixel_arcsec),
        insert_mode=str(prior_catalog_insert_mode),
    )
    prior_diffuse, truth_diffuse, diffuse_stats = _diffuse_cubes(
        args=args,
        frequencies=frequencies,
        header=header,
        shape=input_shape,
    )
    truth_foreground_sky = truth_catalog_sky + truth_diffuse
    prior_foreground_sky = prior_catalog_sky + prior_diffuse
    diffuse_weights = partition_of_unity_grid(input_shape, tuple(args.diffuse_grid))
    specs = _build_specs(
        source_groups,
        args=args,
        diffuse_cell_count=int(diffuse_weights.shape[0]),
    )
    k_to_jy = torch.as_tensor(
        [
            flat._k_to_jy_per_pixel(float(frequency), float(pixel_arcsec))
            for frequency in frequencies.tolist()
        ],
        dtype=torch.float64,
        device=device,
    )

    print(
        json.dumps(
            {
                "event": "observed_prior_inputs_ready",
                "frequency_count": int(len(frequencies)),
                "source_group_count": int(len(source_groups)),
                "nuisance_parameter_count": int(len(specs)),
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    sky_pair = torch.as_tensor(
        np.stack((truth_foreground_sky, prior_foreground_sky), axis=0),
        dtype=torch.float64,
        device=device,
    )
    with torch.no_grad():
        dirty_pair = forward_operator(sky_pair * k_to_jy[None, :, None, None])
    truth_foreground_dirty = dirty_pair[0]
    prior_foreground_dirty = dirty_pair[1]
    del sky_pair, dirty_pair
    eor_cube, first_eor_path = flat.base._load_dirty_cube(
        frequencies.tolist(),
        str(args.truth_eor_pattern),
        eval_size=int(output_shape[1]),
        dtype=np.dtype(np.float64),
    )
    eor_dirty = torch.as_tensor(eor_cube, dtype=torch.float64, device=device)
    foreground_delta = truth_foreground_dirty - prior_foreground_dirty
    centered_total = foreground_delta + eor_dirty

    full_transform = TorchBandpowerTransform(contract.full_geometry, analysis, device)
    science_transform = TorchBandpowerTransform(contract.science_geometry, analysis, device)
    control_indices = torch.as_tensor(contract.control_mode_indices, dtype=torch.int64, device=device)
    guard_indices = torch.as_tensor(contract.guard_mode_indices, dtype=torch.int64, device=device)
    control_design, guard_design = _build_feature_design(
        specs,
        args=args,
        frequencies=frequencies,
        shape=input_shape,
        source_groups=source_groups,
        prior_catalog=prior_catalog_data,
        header=header,
        pixel_arcsec=float(pixel_arcsec),
        catalog_insert_mode=str(prior_catalog_insert_mode),
        diffuse_prior=prior_diffuse,
        diffuse_weights=diffuse_weights,
        k_to_jy=k_to_jy,
        forward_operator=forward_operator,
        full_transform=full_transform,
        control_indices=control_indices,
        guard_indices=guard_indices,
        device=device,
        started=started,
    )
    control_total = _selected_features(centered_total.unsqueeze(0), full_transform, control_indices)[0]
    control_eor = _selected_features(eor_dirty.unsqueeze(0), full_transform, control_indices)[0]
    guard_total = _selected_features(centered_total.unsqueeze(0), full_transform, guard_indices)[0]
    control_scale = prior_predictive_feature_scale(
        control_design,
        floor_quantile=float(args.feature_floor_quantile),
        floor_fraction_of_max=float(args.feature_floor_fraction_of_max),
    )
    total_fit = solve_linear_gaussian_control(
        control_design,
        control_total,
        feature_scale=control_scale,
    )
    eor_coefficients = posterior_mean_for_data(control_design, control_eor, total_fit)
    foreground_coefficients = total_fit.posterior_mean - eor_coefficients
    guard_scale = prior_predictive_feature_scale(
        guard_design,
        floor_quantile=float(args.feature_floor_quantile),
        floor_fraction_of_max=float(args.feature_floor_fraction_of_max),
    )
    guard_score = posterior_predictive_score(
        guard_design,
        guard_total,
        total_fit,
        feature_scale=guard_scale,
    )

    correction_total_sky = _compose_correction_sky(
        specs,
        total_fit.posterior_mean,
        frequencies=frequencies,
        shape=input_shape,
        source_groups=source_groups,
        prior_catalog=prior_catalog_data,
        header=header,
        pixel_arcsec=float(pixel_arcsec),
        catalog_insert_mode=str(prior_catalog_insert_mode),
        diffuse_prior=prior_diffuse,
        diffuse_weights=diffuse_weights,
        pivot_mhz=float(args.nuisance_pivot_mhz),
    )
    correction_eor_sky = _compose_correction_sky(
        specs,
        eor_coefficients,
        frequencies=frequencies,
        shape=input_shape,
        source_groups=source_groups,
        prior_catalog=prior_catalog_data,
        header=header,
        pixel_arcsec=float(pixel_arcsec),
        catalog_insert_mode=str(prior_catalog_insert_mode),
        diffuse_prior=prior_diffuse,
        diffuse_weights=diffuse_weights,
        pivot_mhz=float(args.nuisance_pivot_mhz),
    )
    correction_sky = torch.as_tensor(
        np.stack((correction_total_sky, correction_eor_sky), axis=0),
        dtype=torch.float64,
        device=device,
    )
    with torch.no_grad():
        correction_dirty = forward_operator(correction_sky * k_to_jy[None, :, None, None])
    total_residual = centered_total - correction_dirty[0]
    eor_transfer = eor_dirty - correction_dirty[1]
    foreground_residual = foreground_delta - (correction_dirty[0] - correction_dirty[1])
    linearity_error = relative_linearity_error(
        total_residual,
        (foreground_residual, eor_transfer),
    )
    if linearity_error > 1.0e-10:
        raise RuntimeError(f"foreground/EoR transfer decomposition failed: {linearity_error}")

    power_diagnostics, power_arrays = _power_diagnostics(
        total_residual=total_residual,
        foreground_residual=foreground_residual,
        eor_transfer=eor_transfer,
        eor_truth=eor_dirty,
        transform=science_transform,
        geometry=contract.science_geometry,
        target_bands=args.target_bands,
        quick_tolerance=float(args.quick_gate_tolerance),
    )
    prior_only_diagnostics, prior_only_arrays = _power_diagnostics(
        total_residual=centered_total,
        foreground_residual=foreground_delta,
        eor_transfer=eor_dirty,
        eor_truth=eor_dirty,
        transform=science_transform,
        geometry=contract.science_geometry,
        target_bands=args.target_bands,
        quick_tolerance=float(args.quick_gate_tolerance),
    )
    result: dict[str, Any] = {
        "format_version": 1,
        "method": "observed_prior_control_linear_gaussian_v1",
        "scientific_scope": "noiseless_same_sky_prior_information_screen",
        "elapsed_seconds": time.monotonic() - started,
        "non_cheating_policy": {
            "fit_uses_simulated_foreground_truth": False,
            "fit_uses_eor_truth": False,
            "fit_modes": "control_only",
            "guard_and_science_modes_held_out": True,
            "deep_catalog_and_index_map_roles": "benchmark_truth_construction_only",
            "simulation_component_labels_role": "post_fit_diagnostics_only",
        },
        "inputs": {
            "config": str(args.config),
            "analysis_contract_sha256": analysis.analysis_contract_sha256,
            "estimator_contract_sha256": contract.estimator_contract_sha256,
            "design_npz": str(args.design_npz),
            "tile_cache_dir": str(args.tile_cache_dir),
            "frequencies_mhz": frequencies.tolist(),
            "truth_eor_pattern": str(args.truth_eor_pattern),
            "first_truth_eor_path": str(first_eor_path),
            "truth_catalog": truth_catalog_stats,
            "prior_source_basis_manifest": str(args.prior_source_basis_manifest),
            "prior_source_catalog": str(prior_catalog_path),
            "prior_source_reconstruction": (
                "exact_catalog_membership"
                if prior_catalog_data is not None
                else "legacy_chebyshev_coefficients"
            ),
            "diffuse": diffuse_stats,
        },
        "nuisance": {
            "parameter_count": int(len(specs)),
            "source_group_count": int(len(source_groups)),
            "catalog_enabled": bool(args.catalog_nuisance),
            "diffuse_enabled": bool(args.diffuse_nuisance),
            "diffuse_grid": [int(value) for value in args.diffuse_grid],
            "catalog_amplitude_prior_std": float(args.catalog_amplitude_prior_std),
            "catalog_slope_prior_std": float(args.catalog_slope_prior_std),
            "diffuse_amplitude_prior_std": float(args.diffuse_amplitude_prior_std),
            "diffuse_slope_prior_std": float(args.diffuse_slope_prior_std),
            "pivot_mhz": float(args.nuisance_pivot_mhz),
            "basis": [
                {
                    "name": spec.name,
                    "family": spec.family,
                    "variation": spec.variation,
                    "prior_std": float(spec.prior_std),
                }
                for spec in specs
            ],
        },
        "fit": {
            "control_complex_mode_count": int(control_indices.numel()),
            "guard_complex_mode_count": int(guard_indices.numel()),
            "control": total_fit.stats,
            "guard_posterior_predictive": guard_score,
            "posterior_mean_max_abs_sigma": float(torch.max(torch.abs(total_fit.posterior_mean)).cpu()),
            "posterior_mean_over_3sigma_count": int(torch.sum(torch.abs(total_fit.posterior_mean) > 3.0).cpu()),
            "foreground_component_coefficient_norm": float(torch.linalg.vector_norm(foreground_coefficients).cpu()),
            "eor_component_coefficient_norm": float(torch.linalg.vector_norm(eor_coefficients).cpu()),
            "component_linearity_relative_error": float(linearity_error),
        },
        "ps2d": {
            "after_control_fit": power_diagnostics,
            "prior_mean_only_baseline": prior_only_diagnostics,
        },
        "resource": {
            "device": str(device),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
    }
    flat._atomic_json(args.out_dir / "result.json", result)
    products: dict[str, np.ndarray] = {
        **power_arrays,
        **{f"prior_only_{key}": value for key, value in prior_only_arrays.items()},
        "posterior_mean": np.asarray(total_fit.posterior_mean.detach().cpu(), dtype=np.float64),
        "posterior_covariance": np.asarray(total_fit.posterior_covariance.detach().cpu(), dtype=np.float64),
        "foreground_component_coefficients": np.asarray(foreground_coefficients.detach().cpu(), dtype=np.float64),
        "eor_component_coefficients": np.asarray(eor_coefficients.detach().cpu(), dtype=np.float64),
    }
    if bool(args.write_cubes):
        products.update(
            {
                "total_residual": np.asarray(total_residual.detach().cpu(), dtype=np.float32),
                "foreground_residual": np.asarray(foreground_residual.detach().cpu(), dtype=np.float32),
                "pure_eor_transfer": np.asarray(eor_transfer.detach().cpu(), dtype=np.float32),
                "truth_eor_dirty": np.asarray(eor_dirty.detach().cpu(), dtype=np.float32),
            }
        )
    flat._atomic_npz(args.out_dir / "products.npz", products)
    print(
        json.dumps(
            {
                "event": "observed_prior_control_screen_done",
                "out_dir": str(args.out_dir),
                "quick_pass_count": power_diagnostics["gates"]["quick_pass_count"],
                "target_band_count": power_diagnostics["gates"]["target_band_count"],
                "total_relative_l2": power_diagnostics["total_residual_vs_eor"]["count_weighted_relative_l2"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
