#!/usr/bin/env python3
"""Propagate a physical outer foreground through the exact PB Q_beta chain."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PROJECT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from calibrate_visibility_qbeta_noiseless import (  # noqa: E402
    _analysis_frequency_indices,
    _format_pattern,
    _load_bank,
    _maximum_patch_delays,
    _relative_l2,
    _row_kperp,
    _visibility_bandpowers,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_matrix_free import (  # noqa: E402
    apply_exact_visibility_operator_matrix_free,
)
from visibility_primary_beam import (  # noqa: E402
    open_indexed_frequency_row_direction_kernel_multiplier,
)
from visibility_qbeta_coarse import (  # noqa: E402
    _realization_metrics,
    transform_q,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frequency-config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--outer-sky", type=Path, required=True)
    parser.add_argument("--combined-result-dir", type=Path, required=True)
    parser.add_argument("--coarse-products", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    parser.add_argument(
        "--partition-result-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--aperture-row-beam-cache-pattern", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument("--row-chunk", type=int, default=32)
    parser.add_argument("--source-chunk", type=int, default=8192)
    parser.add_argument(
        "--operator-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    return parser.parse_args(argv)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


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


def _load_result(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(directory / "result.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    return arrays, metadata


def _foreground_effect(
    *,
    q: np.ndarray,
    response: np.ndarray,
    target: np.ndarray,
    selected_positions: np.ndarray,
    relative_response: np.ndarray,
) -> dict[str, Any]:
    row_sum = np.sum(response, axis=1)
    estimate = np.divide(
        np.asarray(q, dtype=np.float64),
        row_sum,
        out=np.full(row_sum.shape, np.nan, dtype=np.float64),
        where=row_sum > 0.0,
    )
    positions = np.asarray(selected_positions, dtype=np.int64)
    weights = np.asarray(relative_response, dtype=np.float64)[positions]
    reference = np.asarray(target, dtype=np.float64)[positions]
    values = estimate[positions]
    fractions = np.abs(values) / np.maximum(
        np.abs(reference), 1e-300
    )
    denominator = max(
        float(np.sum(weights * np.abs(reference))), 1e-300
    )
    return {
        "integrated_signed_ratio": float(
            np.sum(weights * values) / denominator
        ),
        "integrated_absolute_ratio": float(
            np.sum(weights * np.abs(values)) / denominator
        ),
        "maximum_absolute_window_ratio": float(
            np.max(fractions)
        ),
        "median_absolute_window_ratio": float(np.median(fractions)),
        "p90_absolute_window_ratio": float(
            np.quantile(fractions, 0.9)
        ),
        "above_10pct_count": int(np.count_nonzero(fractions > 0.1)),
        "above_20pct_count": int(np.count_nonzero(fractions > 0.2)),
    }


def _coarse_estimate(
    q: np.ndarray,
    *,
    transform: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    row_sum = np.sum(np.asarray(response, dtype=np.float64), axis=1)
    return np.divide(
        transform_q(q, transform),
        row_sum,
        out=np.full(
            (*np.asarray(q).shape[:-1], row_sum.size),
            np.nan,
            dtype=np.float64,
        ),
        where=row_sum > 0.0,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frequency_config = json.loads(
        args.frequency_config.read_text(encoding="utf-8")
    )
    resolved = resolve_mode_first_analysis(config)
    analysis_frequencies_mhz = np.asarray(
        resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    frequencies_mhz = np.asarray(
        frequency_config["frequencies_mhz"], dtype=np.float64
    )
    frequencies_hz = frequencies_mhz * 1e6
    analysis_indices = _analysis_frequency_indices(
        frequencies_mhz, analysis_frequencies_mhz
    )
    bank, bank_manifest = _load_bank(
        args.bank_dir, requested_frequencies_hz=frequencies_hz
    )
    with np.load(args.outer_sky, allow_pickle=False) as archive:
        outer = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        not np.allclose(
            outer["frequencies_mhz"],
            frequencies_mhz,
            rtol=0.0,
            atol=1e-9,
        )
        or outer["fg_jy"].shape[0] != frequencies_mhz.size
    ):
        raise ValueError("Outer sky and frequency config differ")
    combined, combined_metadata = _load_result(
        args.combined_result_dir
    )
    del combined_metadata
    with np.load(args.coarse_products, allow_pickle=False) as archive:
        coarse = {name: np.asarray(archive[name]) for name in archive.files}
    prefix = f"{args.profile}_"
    required_coarse = {
        f"{prefix}selected",
        f"{prefix}target",
        f"{prefix}bank_total_estimate",
        f"{prefix}transform",
        f"{prefix}response",
        f"{prefix}group_metric_weights",
    }
    missing_coarse = sorted(required_coarse - coarse.keys())
    if missing_coarse:
        raise ValueError(
            "Coarse products lack required outer-field arrays: "
            + ", ".join(missing_coarse)
        )
    partition_results = [
        _load_result(directory) for directory in args.partition_result_dir
    ]
    if len(partition_results) < 2:
        raise ValueError("At least two row partitions are required")
    combined_rows = np.concatenate(
        [
            np.asarray(arrays["selected_bank_rows"], dtype=np.int64)
            for arrays, _ in partition_results
        ]
    )
    if not np.array_equal(
        np.sort(combined_rows),
        np.sort(
            np.asarray(combined["selected_bank_rows"], dtype=np.int64)
        ),
    ):
        raise ValueError("Partition rows do not reproduce the combined result")
    reference_settings = partition_results[0][1]["settings"]
    for arrays, metadata in partition_results[1:]:
        del arrays
        for key in (
            "foreground_filter",
            "filter_bandwidth_scope",
            "dpss_eigenvalue_threshold",
            "suppression_strength",
            "polynomial_degree",
            "spectral_taper",
        ):
            if metadata["settings"].get(key) != reference_settings.get(key):
                raise ValueError(f"Partition settings differ in {key}")

    region_ids = np.asarray(outer["region_id"], dtype=np.int64)
    unique_regions = np.unique(region_ids)
    sky_batches = [np.asarray(outer["fg_jy"], dtype=np.float32)]
    for region_id in unique_regions:
        sky_batches.append(
            np.where(
                region_ids[None, :] == region_id,
                outer["fg_jy"],
                0.0,
            ).astype(np.float32)
        )
    skies = np.stack(sky_batches, axis=0)[:, :, None, :]

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Outer-field propagation requires CUDA")
    torch.cuda.set_device(device)
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    reference_frequency_hz = (
        float(resolved.geometry["reference_frequency_mhz"]) * 1e6
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
    outer_q_by_partition = []
    region_q_by_partition = []
    base_total_q_by_partition = []
    extended_total_q_by_partition = []
    region_extended_total_q_by_partition = []
    cache_hashes: list[list[str]] = []
    for partition_index, (arrays, metadata) in enumerate(partition_results):
        selected_rows = np.asarray(
            arrays["selected_bank_rows"], dtype=np.int64
        )
        uvw = np.asarray(
            bank["sample_uvw_m"][selected_rows], dtype=np.float64
        )
        row_kperp = _row_kperp(
            uvw,
            reference_frequency_hz=reference_frequency_hz,
            transverse_distance_mpc=float(
                resolved.geometry["transverse_distance_mpc"]
            ),
        )
        cache_dirs = [
            _format_pattern(
                str(args.aperture_row_beam_cache_pattern),
                float(frequency_mhz),
            )
            for frequency_mhz in frequencies_mhz
        ]
        (
            beam_multiplier,
            cache_metadata,
            _,
        ) = open_indexed_frequency_row_direction_kernel_multiplier(
            cache_dirs, selected_bank_rows=selected_rows
        )
        cache_hashes.append(
            [str(row["data_sha256"]) for row in cache_metadata]
        )
        for frequency_hz, row in zip(
            frequencies_hz, cache_metadata, strict=True
        ):
            if (
                tuple(int(value) for value in row["shape"])[1]
                != int(outer["l_cosine"].size)
                or not np.isclose(
                    float(row["frequency_hz"]),
                    float(frequency_hz),
                    rtol=0.0,
                    atol=1e-3,
                )
            ):
                raise ValueError(
                    "Outer-field PB cache geometry or frequency differs"
                )

        def report_frequency(
            frequency_index: int, frequency_hz: float
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "outer_operator_frequency",
                        "partition_index": int(partition_index),
                        "frequency_index": int(frequency_index),
                        "frequency_mhz": float(frequency_hz / 1e6),
                        "elapsed_seconds": float(
                            time.monotonic() - started
                        ),
                    }
                ),
                flush=True,
            )

        outer_vis = apply_exact_visibility_operator_matrix_free(
            torch=torch,
            frequencies_hz=frequencies_hz,
            uvw_m=uvw,
            l_cosine=outer["l_cosine"],
            m_cosine=outer["m_cosine"],
            n_minus_one=outer["n_minus_one"],
            sky_jy=skies,
            channel_bandwidth_hz=float(args.channel_bandwidth_hz),
            integration_time_s=float(args.integration_time_s),
            phase_dec_deg=float(args.phase_dec_deg),
            device=device,
            operator_dtype=str(args.operator_dtype),
            row_chunk=int(args.row_chunk),
            source_chunk=int(args.source_chunk),
            kernel_multiplier=beam_multiplier,
            progress_callback=report_frequency,
        )
        bandpower_kwargs = {
            "frequencies_hz": frequencies_hz,
            "analysis_frequency_indices": analysis_indices,
            "filter_bandwidth_scope": str(
                metadata["settings"]["filter_bandwidth_scope"]
            ),
            "row_kperp": row_kperp,
            "kperp_edges": kperp_edges,
            "maximum_delays_s": maximum_delays,
            "dpss_eigenvalue_threshold": float(
                metadata["settings"]["dpss_eigenvalue_threshold"]
            ),
            "foreground_filter": str(
                metadata["settings"]["foreground_filter"]
            ),
            "suppression_strength": float(
                metadata["settings"]["suppression_strength"]
            ),
            "polynomial_degree": int(
                metadata["settings"]["polynomial_degree"]
            ),
            "spectral_taper": str(
                metadata["settings"]["spectral_taper"]
            ),
        }
        output_band_ids = np.asarray(
            arrays["output_band_ids"], dtype=np.int64
        )
        outer_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=outer_vis, **bandpower_kwargs
        )
        outer_q = outer_q.reshape(
            outer_vis.shape[0], -1
        )[:, output_band_ids]
        base_total_vis = (
            np.asarray(
                bank["sample_fg"][:, selected_rows],
                dtype=np.complex128,
            )
            + np.asarray(
                bank["sample_eor"][:, selected_rows],
                dtype=np.complex128,
            )
        )
        base_total_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=base_total_vis, **bandpower_kwargs
        )
        extended_total_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=base_total_vis + outer_vis[0],
            **bandpower_kwargs,
        )
        region_extended_total_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=base_total_vis[None, ...] + outer_vis[1:],
            **bandpower_kwargs,
        )
        outer_q_by_partition.append(outer_q[0])
        region_q_by_partition.append(outer_q[1:])
        base_total_q_by_partition.append(
            base_total_q.reshape(-1)[output_band_ids]
        )
        extended_total_q_by_partition.append(
            extended_total_q.reshape(-1)[output_band_ids]
        )
        region_extended_total_q_by_partition.append(
            region_extended_total_q.reshape(
                outer_vis.shape[0] - 1, -1
            )[:, output_band_ids]
        )

    outer_q = np.mean(np.stack(outer_q_by_partition), axis=0)
    region_q = np.mean(np.stack(region_q_by_partition), axis=0)
    base_total_q = np.mean(
        np.stack(base_total_q_by_partition), axis=0
    )
    extended_total_q = np.mean(
        np.stack(extended_total_q_by_partition), axis=0
    )
    region_extended_total_q = np.mean(
        np.stack(region_extended_total_q_by_partition), axis=0
    )
    transform = np.asarray(
        coarse[f"{prefix}transform"], dtype=np.float64
    )
    coarse_response = np.asarray(
        coarse[f"{prefix}response"], dtype=np.float64
    )
    selected = np.asarray(coarse[f"{prefix}selected"], dtype=bool)
    selected_positions = np.flatnonzero(selected)
    target_windowed = np.asarray(
        coarse[f"{prefix}target"], dtype=np.float64
    )
    metric_weights = np.asarray(
        coarse[f"{prefix}group_metric_weights"], dtype=np.float64
    )
    base_estimate = _coarse_estimate(
        base_total_q,
        transform=transform,
        response=coarse_response,
    )
    extended_estimate = _coarse_estimate(
        extended_total_q,
        transform=transform,
        response=coarse_response,
    )
    outer_estimate = _coarse_estimate(
        outer_q,
        transform=transform,
        response=coarse_response,
    )
    delta_q = extended_total_q - base_total_q
    delta_estimate = extended_estimate - base_estimate
    region_outer_estimate = _coarse_estimate(
        region_q,
        transform=transform,
        response=coarse_response,
    )
    region_delta_q = region_extended_total_q - base_total_q[None, :]
    region_delta_estimate = _coarse_estimate(
        region_delta_q,
        transform=transform,
        response=coarse_response,
    )
    summary = {
        "schema": "visibility_qbeta_outer_field_evaluation",
        "schema_version": 3,
        "profile": str(args.profile),
        "elapsed_seconds": float(time.monotonic() - started),
        "partition_count": int(len(partition_results)),
        "bank_sha256": bank_manifest["bank_sha256"],
        "base_q_recomputation_relative_l2": _relative_l2(
            base_total_q, combined["bank_total_q"]
        ),
        "base_coarse_recomputation_relative_l2": _relative_l2(
            base_estimate,
            coarse[f"{prefix}bank_total_estimate"],
        ),
        "selected_window_count": int(selected_positions.size),
        "base_total": _realization_metrics(
            base_estimate[selected],
            target_windowed[selected],
            metric_weights[selected],
        ),
        "extended_total": _realization_metrics(
            extended_estimate[selected],
            target_windowed[selected],
            metric_weights[selected],
        ),
        "outer_only_effect": _foreground_effect(
            q=transform_q(outer_q, transform),
            response=coarse_response,
            target=target_windowed,
            selected_positions=selected_positions,
            relative_response=metric_weights,
        ),
        "outer_induced_total_change": _foreground_effect(
            q=transform_q(delta_q, transform),
            response=coarse_response,
            target=target_windowed,
            selected_positions=selected_positions,
            relative_response=metric_weights,
        ),
        "regions": [
            {
                "region_id": int(region_id),
                "outer_only_effect": _foreground_effect(
                    q=transform_q(region_q[position], transform),
                    response=coarse_response,
                    target=target_windowed,
                    selected_positions=selected_positions,
                    relative_response=metric_weights,
                ),
                "induced_total_change": _foreground_effect(
                    q=transform_q(region_delta_q[position], transform),
                    response=coarse_response,
                    target=target_windowed,
                    selected_positions=selected_positions,
                    relative_response=metric_weights,
                ),
            }
            for position, region_id in enumerate(unique_regions)
        ],
        "cache_data_sha256_by_partition": cache_hashes,
        "limitations": [
            "outer sky is the same simulated foreground outside the central square",
            "outer sky is flux-conserving block-averaged and brightness-selected",
            "the finite full cube reaches only its square boundary, not the horizon",
            "thermal noise and calibration errors are absent",
        ],
    }
    products = {
        "outer_q": outer_q,
        "region_ids": unique_regions,
        "region_q": region_q,
        "region_extended_total_q": region_extended_total_q,
        "region_delta_total_q": region_delta_q,
        "base_total_q": base_total_q,
        "extended_total_q": extended_total_q,
        "delta_total_q": delta_q,
        "selected_window_positions": selected_positions,
        "target_windowed_power": target_windowed,
        "group_metric_weights": metric_weights,
        "base_total_windowed_power": base_estimate,
        "extended_total_windowed_power": extended_estimate,
        "outer_windowed_power": outer_estimate,
        "delta_total_windowed_power": delta_estimate,
        "region_outer_windowed_power": region_outer_estimate,
        "region_delta_total_windowed_power": region_delta_estimate,
    }
    for name in (
        "group_kperp_first",
        "group_kperp_stop",
        "minimum_input_relative_response",
        "window_fraction",
        "nominal_window_fraction",
        "effective_width",
    ):
        key = f"{prefix}{name}"
        if key in coarse:
            products[name] = coarse[key]
    _atomic_npz(args.out_dir / "products.npz", products)
    _atomic_json(args.out_dir / "result.json", summary)
    print(json.dumps(_json_safe(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
