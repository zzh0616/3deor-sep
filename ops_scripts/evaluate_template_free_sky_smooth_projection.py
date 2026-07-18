#!/usr/bin/env python3
"""Run a full-cube template-free sky-smooth nuisance identifiability screen.

The fitted sky model has free spatial coefficient maps multiplied only by a
small smooth spectral basis. No foreground morphology or coefficient template
is read by the fit. Simulation component cubes are summed once to construct the
synthetic observed cube; their component labels are not exposed to the solve,
and they are separated again only for post-fit diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
OPERATOR_SCRIPT_DIR = Path(
    os.environ.get("FG_RMW_OPERATOR_SCRIPT_DIR", str(SCRIPT_DIR))
).resolve()
if str(OPERATOR_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(OPERATOR_SCRIPT_DIR))

import fit_cached_pca_proxy_cheb_operator_separation as base  # noqa: E402
from evaluate_fullsky_response_interpolation import (  # noqa: E402
    _k_to_jy_per_pixel,
)
from ps2d_v2_estimator import (  # noqa: E402
    TorchBandpowerTransform,
    build_mode_first_estimator_contract,
)
from ps2d_v2_sky_smooth import (  # noqa: E402
    compose_sky_smooth_cube,
    matrix_free_lsqr,
    orthonormal_chebyshev_spectral_basis,
    relative_adjoint_error,
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _tensor_corr(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.detach().double().reshape(-1)
    right = second.detach().double().reshape(-1)
    left = left - torch.mean(left)
    right = right - torch.mean(right)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.cpu()) <= 0.0:
        return 0.0
    return float(((left @ right) / denominator).cpu())


def _cube_metrics(estimate: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    error = estimate.detach().double() - truth.detach().double()
    truth_rms = torch.sqrt(torch.mean(truth.detach().double().square()))
    estimate_rms = torch.sqrt(torch.mean(estimate.detach().double().square()))
    error_rms = torch.sqrt(torch.mean(error.square()))
    denominator = torch.clamp(truth_rms, min=1e-300)
    return {
        "residual_over_truth_rms": float((error_rms / denominator).cpu()),
        "estimate_over_truth_rms": float((estimate_rms / denominator).cpu()),
        "correlation": _tensor_corr(estimate, truth),
    }


def _band_metrics(
    estimate: torch.Tensor,
    truth: torch.Tensor,
    *,
    transform: TorchBandpowerTransform,
    target_bands: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    estimate_power = transform(estimate)[0]
    truth_power = transform(truth)[0]
    indices = torch.as_tensor(
        target_bands, dtype=torch.int64, device=estimate.device
    )
    selected_estimate = estimate_power.index_select(0, indices)
    selected_truth = truth_power.index_select(0, indices)
    counts = transform.counts.index_select(0, indices)
    difference = selected_estimate - selected_truth
    l2_denominator = torch.sum(counts * selected_truth.square())
    power_denominator = torch.sum(counts * selected_truth)
    relative = torch.abs(difference) / torch.clamp(
        torch.abs(selected_truth), min=1e-300
    )
    metrics = {
        "count_weighted_relative_l2": float(
            torch.sqrt(
                torch.sum(counts * difference.square())
                / torch.clamp(l2_denominator, min=1e-300)
            ).cpu()
        ),
        "exact_mode_power_sum_ratio": float(
            (
                torch.sum(counts * selected_estimate)
                / torch.clamp(power_denominator, min=1e-300)
            ).cpu()
        ),
        "maximum_per_band_relative_error": float(torch.max(relative).cpu()),
        "all_positive": bool(torch.all(selected_estimate > 0.0).cpu()),
        "target_bandpowers": [
            float(value) for value in selected_estimate.detach().cpu().tolist()
        ],
        "truth_target_bandpowers": [
            float(value) for value in selected_truth.detach().cpu().tolist()
        ],
        "target_band_ratios": [
            float(value)
            for value in (
                selected_estimate
                / torch.clamp(selected_truth, min=1e-300)
            )
            .detach()
            .cpu()
            .tolist()
        ],
    }
    return (
        metrics,
        np.asarray(estimate_power.detach().cpu(), dtype=np.float64),
        np.asarray(truth_power.detach().cpu(), dtype=np.float64),
    )


def _load_operator(
    *,
    design_path: Path,
    tile_cache_dir: Path,
    frequencies: np.ndarray,
    analysis_shape: tuple[int, int, int],
    device: torch.device,
    checkpoint_tiles: bool,
) -> tuple[Any, dict[str, Any]]:
    with np.load(design_path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    identity = metadata.get("identity", {})
    operator_identity = identity.get("operator_identity", {})
    if int(identity.get("format_version", -1)) != 3:
        raise ValueError("Compiled design must use canonical identity v3")
    if not np.allclose(
        operator_identity.get("freqs_mhz", []),
        frequencies,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Compiled operator frequencies do not match the config")
    image_size = int(operator_identity["image_size"])
    eval_size = int(operator_identity["eval_crop_size"])
    if (len(frequencies), eval_size, eval_size) != tuple(analysis_shape):
        raise ValueError("Compiled operator output shape does not match the config")

    cache_indices = np.asarray(
        operator_identity["freq_cache_indices"], dtype=np.int64
    )
    if cache_indices.shape != frequencies.shape:
        raise ValueError("Operator cache-index list has the wrong length")
    caches: list[Any] = []
    for local_index, frequency in enumerate(frequencies.tolist()):
        frequency_caches = base._build_freq_tile_caches(
            freq_index=int(cache_indices[local_index]),
            freq_mhz=float(frequency),
            dense_grid_csv=Path(
                base._format_pattern(
                    str(operator_identity["dense_grid_csv_pattern"]),
                    freq=float(frequency),
                )
            ),
            train_grid_csv=Path(
                base._format_pattern(
                    str(operator_identity["train_grid_csv_pattern"]),
                    freq=float(frequency),
                )
            ),
            train_response_pattern=str(operator_identity["train_response_pattern"]),
            image_size=image_size,
            response_crop_size=int(operator_identity["response_crop_size"]),
            eval_size=eval_size,
            tile_size=int(operator_identity["tile_size"]),
            train_halo_px=int(operator_identity["train_halo_px"]),
            model_margin_arg=int(operator_identity["model_margin"]),
            pca_rank=int(operator_identity["pca_rank"]),
            rbf_scale_px=float(operator_identity["rbf_scale_px"]),
            dtype=np.dtype(np.float64),
            progress_every_tile=0,
            max_tiles=0,
            tile_cache_dir=tile_cache_dir,
            tile_cache_refresh=False,
            tile_cache_meta_path_rewrites=[],
            preload_train_responses=False,
            cache_build_only=False,
        )
        caches.extend(
            replace(tile, freq_index=int(local_index)) for tile in frequency_caches
        )
    expected_tiles = len(frequencies) * int(
        operator_identity["full_tile_count_per_freq"]
    )
    if len(caches) != expected_tiles:
        raise ValueError(
            f"Cached operator loaded {len(caches)} tiles, expected {expected_tiles}"
        )
    forward = base.CachedPcaProxyForward(
        caches,
        n_freq=len(frequencies),
        eval_size=eval_size,
        device=device,
        dtype=torch.float64,
        keep_kernel_fft_on_device=False,
        checkpoint_tiles=bool(checkpoint_tiles),
    )
    return forward, operator_identity


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-npz", type=Path, required=True)
    parser.add_argument("--tile-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_DIR / "configs/ps2d_v2_8wide_isobeam_patch.json",
    )
    parser.add_argument("--truth-fg-pattern", required=True)
    parser.add_argument("--truth-eor-pattern", required=True)
    parser.add_argument("--spectral-degrees", type=_parse_ints, default=[0, 1])
    parser.add_argument("--target-bands", type=_parse_ints, default=[4, 10, 16, 7, 13, 19])
    parser.add_argument("--lsqr-max-iters", type=int, default=24)
    parser.add_argument("--lsqr-relative-residual-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--jacobi-probes",
        type=int,
        default=0,
        help="Operator-only Hutchinson probes for a right Jacobi preconditioner.",
    )
    parser.add_argument("--jacobi-floor-fraction", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-pure-eor-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = build_mode_first_estimator_contract(config)
    analysis = contract.analysis
    shape = tuple(int(value) for value in analysis.full_layout.cube_shape)
    frequencies = np.asarray(
        contract.resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    device = torch.device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty((), dtype=torch.float32, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    dirty_fg, _ = base._load_dirty_cube(
        frequencies.tolist(),
        args.truth_fg_pattern,
        eval_size=int(shape[1]),
        dtype=np.dtype(np.float64),
    )
    dirty_eor, _ = base._load_dirty_cube(
        frequencies.tolist(),
        args.truth_eor_pattern,
        eval_size=int(shape[1]),
        dtype=np.dtype(np.float64),
    )
    observed = dirty_fg + dirty_eor
    observed_tensor = torch.as_tensor(observed, dtype=torch.float64, device=device)
    eor_tensor = torch.as_tensor(dirty_eor, dtype=torch.float64, device=device)

    print(
        json.dumps(
            {
                "event": "operator_load_start",
                "spectral_degrees": list(args.spectral_degrees),
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    forward_operator, operator_identity = _load_operator(
        design_path=args.design_npz,
        tile_cache_dir=args.tile_cache_dir,
        frequencies=frequencies,
        analysis_shape=shape,
        device=device,
        checkpoint_tiles=bool(args.checkpoint_tiles),
    )
    image_size = int(operator_identity["image_size"])
    spectral_basis = orthonormal_chebyshev_spectral_basis(
        len(frequencies),
        args.spectral_degrees,
        dtype=torch.float64,
        device=device,
    )
    k_to_jy = torch.as_tensor(
        [
            _k_to_jy_per_pixel(
                float(frequency), float(operator_identity["pixel_arcsec"])
            )
            for frequency in frequencies.tolist()
        ],
        dtype=torch.float64,
        device=device,
    )

    def apply_full(coefficient_maps: torch.Tensor) -> torch.Tensor:
        sky_kelvin = compose_sky_smooth_cube(coefficient_maps, spectral_basis)
        sky_flux = sky_kelvin * k_to_jy[:, None, None]
        return forward_operator(sky_flux)

    adjoint_probe = torch.zeros(
        (spectral_basis.shape[1], image_size, image_size),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    adjoint_output = apply_full(adjoint_probe)

    def apply_physical_adjoint(dirty_cube: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=adjoint_output,
            inputs=adjoint_probe,
            grad_outputs=dirty_cube,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

    def apply_physical_forward(coefficient_maps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return apply_full(coefficient_maps.detach()).detach()

    jacobi_probe_count = int(args.jacobi_probes)
    if jacobi_probe_count < 0:
        raise ValueError("Jacobi probe count must be non-negative")
    if not 0.0 < float(args.jacobi_floor_fraction) <= 1.0:
        raise ValueError("Jacobi floor fraction must be in (0, 1]")
    inverse_sqrt_diagonal = torch.ones_like(adjoint_probe.detach())
    preconditioner_stats: dict[str, Any] = {
        "enabled": False,
        "method": "identity",
        "probe_count": 0,
    }
    if jacobi_probe_count > 0:
        preconditioner_started = time.monotonic()
        diagonal_sum = torch.zeros_like(inverse_sqrt_diagonal)
        preconditioner_generator = torch.Generator(device=device)
        preconditioner_generator.manual_seed(int(args.seed) + 7919)
        for probe_index in range(jacobi_probe_count):
            sign = torch.randint(
                0,
                2,
                adjoint_probe.shape,
                dtype=torch.int8,
                device=device,
                generator=preconditioner_generator,
            ).to(dtype=torch.float64)
            sign = 2.0 * sign - 1.0
            normal_sign = apply_physical_adjoint(
                apply_physical_forward(sign)
            )
            diagonal_sum = diagonal_sum + sign * normal_sign
            print(
                json.dumps(
                    {
                        "event": "jacobi_probe_done",
                        "probe": probe_index + 1,
                        "probe_count": jacobi_probe_count,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        diagonal_estimate = diagonal_sum / float(jacobi_probe_count)
        finite = torch.isfinite(diagonal_estimate)
        positive = finite & (diagonal_estimate > 0.0)
        if not bool(torch.any(positive)):
            raise RuntimeError("Jacobi estimator has no positive finite entries")
        positive_values = diagonal_estimate[positive]
        reference = torch.median(positive_values)
        floor = float(args.jacobi_floor_fraction) * reference
        stabilized = torch.where(
            finite,
            torch.clamp(diagonal_estimate, min=floor),
            floor,
        )
        inverse_sqrt_diagonal = torch.rsqrt(stabilized)
        preconditioner_stats = {
            "enabled": True,
            "method": "operator_hutchinson_diagonal_right_jacobi",
            "probe_count": jacobi_probe_count,
            "seed": int(args.seed) + 7919,
            "floor_fraction": float(args.jacobi_floor_fraction),
            "floor_absolute": float(floor.detach().cpu()),
            "nonpositive_or_nonfinite_fraction": float(
                torch.mean((~positive).to(dtype=torch.float64)).cpu()
            ),
            "positive_diagonal_min": float(torch.min(positive_values).cpu()),
            "positive_diagonal_median": float(reference.cpu()),
            "positive_diagonal_max": float(torch.max(positive_values).cpu()),
            "inverse_sqrt_min": float(torch.min(inverse_sqrt_diagonal).cpu()),
            "inverse_sqrt_median": float(
                torch.median(inverse_sqrt_diagonal).cpu()
            ),
            "inverse_sqrt_max": float(torch.max(inverse_sqrt_diagonal).cpu()),
            "elapsed_seconds": time.monotonic() - preconditioner_started,
        }

    def solver_to_physical(coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates * inverse_sqrt_diagonal

    def apply_forward(coordinates: torch.Tensor) -> torch.Tensor:
        return apply_physical_forward(solver_to_physical(coordinates))

    def apply_adjoint(dirty_cube: torch.Tensor) -> torch.Tensor:
        return inverse_sqrt_diagonal * apply_physical_adjoint(dirty_cube)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    domain_probe = torch.randn(
        adjoint_probe.shape,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    range_probe = torch.randn(
        shape,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    adjoint_stats = relative_adjoint_error(
        apply_forward, apply_adjoint, domain_probe, range_probe
    )
    if adjoint_stats["relative_error"] > 1e-8:
        raise RuntimeError(f"Cached operator adjoint closure failed: {adjoint_stats}")
    print(
        json.dumps(
            {
                "event": "operator_adjoint_checked",
                **adjoint_stats,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    def solve(label: str, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        solve_started = time.monotonic()

        def progress(record: dict[str, float | int]) -> None:
            print(
                json.dumps(
                    {
                        "event": "lsqr_iteration",
                        "label": label,
                        **record,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        result = matrix_free_lsqr(
            apply_forward,
            apply_adjoint,
            rhs,
            max_iters=int(args.lsqr_max_iters),
            relative_residual_tolerance=float(
                args.lsqr_relative_residual_tolerance
            ),
            progress_callback=progress,
        )
        physical_solution = solver_to_physical(result.solution)
        prediction = apply_physical_forward(physical_solution)
        residual = rhs - prediction
        stats = {
            **result.stats,
            "elapsed_seconds": time.monotonic() - solve_started,
            "prediction_over_rhs_rms": float(
                (
                    torch.sqrt(torch.mean(prediction.double().square()))
                    / torch.clamp(
                        torch.sqrt(torch.mean(rhs.double().square())), min=1e-300
                    )
                ).cpu()
            ),
        }
        return physical_solution, residual, stats

    observed_coefficients, observed_residual, observed_solver = solve(
        "observed_total", observed_tensor
    )
    if bool(args.run_pure_eor_transfer):
        eor_coefficients, eor_residual, eor_solver = solve("pure_eor", eor_tensor)
    else:
        eor_coefficients = torch.empty((0,), dtype=torch.float64, device=device)
        eor_residual = None
        eor_solver = {"enabled": False}

    transform = TorchBandpowerTransform(contract.science_geometry, analysis, device)
    target_bands = np.asarray(args.target_bands, dtype=np.int64)
    if target_bands.size == 0 or np.any(target_bands < 0) or np.any(
        target_bands >= contract.science_geometry.band_count
    ):
        raise ValueError("Target bands are empty or outside the science geometry")
    separation_bands, observed_power, truth_power = _band_metrics(
        observed_residual.unsqueeze(0),
        eor_tensor.unsqueeze(0),
        transform=transform,
        target_bands=target_bands,
    )
    if eor_residual is not None:
        transfer_bands, eor_residual_power, _ = _band_metrics(
            eor_residual.unsqueeze(0),
            eor_tensor.unsqueeze(0),
            transform=transform,
            target_bands=target_bands,
        )
        pure_eor_cube_metrics = _cube_metrics(eor_residual, eor_tensor)
        pure_eor_residual_product = np.asarray(
            eor_residual.detach().cpu(), dtype=np.float64
        )
    else:
        transfer_bands = None
        eor_residual_power = np.empty((0,), dtype=np.float64)
        pure_eor_cube_metrics = None
        pure_eor_residual_product = np.empty((0,), dtype=np.float64)

    raw_dimension = int(np.prod(shape))
    coefficient_dimension = int(
        spectral_basis.shape[1] * image_size * image_size
    )
    result_payload: dict[str, Any] = {
        "format_version": 1,
        "method": "template_free_sky_smooth_full_cube_flat_projection_v1",
        "scientific_scope": "noiseless_identifiability_screen_not_final_qml",
        "elapsed_seconds": time.monotonic() - started,
        "input": {
            "config": str(args.config),
            "design_npz": str(args.design_npz),
            "tile_cache_dir": str(args.tile_cache_dir),
            "synthetic_observed_foreground_component_pattern": str(
                args.truth_fg_pattern
            ),
            "synthetic_observed_eor_component_pattern": str(
                args.truth_eor_pattern
            ),
            "frequencies_mhz": frequencies.tolist(),
            "dirty_cube_shape": list(shape),
            "target_bands": target_bands.tolist(),
        },
        "nuisance": {
            "fixed_foreground_template_used": False,
            "spatial_coefficient_maps": "one_free_full_sky_map_per_spectral_basis_column",
            "spectral_family": "orthonormalized_chebyshev",
            "spectral_degrees": list(args.spectral_degrees),
            "spectral_basis": np.asarray(
                spectral_basis.detach().cpu(), dtype=np.float64
            ).tolist(),
            "coefficient_dimension": coefficient_dimension,
            "dirty_data_dimension": raw_dimension,
            "coefficient_to_data_dimension_ratio": coefficient_dimension
            / raw_dimension,
            "fit_weighting": "identity_full_dirty_cube",
            "coefficient_prior": "flat_minimum_norm_lsqr",
        },
        "operator": {
            "identity": operator_identity,
            "adjoint_dot_test": adjoint_stats,
            "checkpoint_tiles": bool(args.checkpoint_tiles),
        },
        "solver": {
            "name": "matrix_free_lsqr",
            "max_iters": int(args.lsqr_max_iters),
            "relative_residual_tolerance": float(
                args.lsqr_relative_residual_tolerance
            ),
            "observed_total": observed_solver,
            "pure_eor": eor_solver,
            "right_preconditioner": preconditioner_stats,
        },
        "truth_diagnostic_only": {
            "pure_eor_transfer_run": bool(args.run_pure_eor_transfer),
            "observed_residual_vs_dirty_eor_cube": _cube_metrics(
                observed_residual, eor_tensor
            ),
            "pure_eor_after_projection_vs_dirty_eor_cube": pure_eor_cube_metrics,
            "observed_residual_vs_dirty_eor_ps2d_targets": separation_bands,
            "pure_eor_transfer_ps2d_targets": transfer_bands,
        },
        "resource": {
            "device": str(device),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
    }
    _atomic_json(args.out_dir / "result.json", result_payload)
    _atomic_npz(
        args.out_dir / "products.npz",
        {
            "spectral_basis": np.asarray(
                spectral_basis.detach().cpu(), dtype=np.float64
            ),
            "observed_coefficients": np.asarray(
                observed_coefficients.detach().cpu(), dtype=np.float64
            ),
            "observed_residual": np.asarray(
                observed_residual.detach().cpu(), dtype=np.float64
            ),
            "pure_eor_coefficients": np.asarray(
                eor_coefficients.detach().cpu(), dtype=np.float64
            ),
            "pure_eor_residual": pure_eor_residual_product,
            "truth_eor_power": truth_power,
            "observed_residual_power": observed_power,
            "pure_eor_residual_power": eor_residual_power,
            "target_bands": target_bands,
        },
    )
    print(
        json.dumps(
            {
                "event": "template_free_sky_smooth_screen_done",
                "out_dir": str(args.out_dir),
                "separation_l2": separation_bands["count_weighted_relative_l2"],
                "pure_eor_transfer_l2": (
                    None
                    if transfer_bands is None
                    else transfer_bands["count_weighted_relative_l2"]
                ),
                "elapsed_seconds": result_payload["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
