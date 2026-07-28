#!/usr/bin/env python3
"""Create truly shared full-band realizations for local-redshift covariance."""

from __future__ import annotations

import argparse
import hashlib
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
    _row_kperp,
    _visibility_bandpowers,
)
from evaluate_visibility_qbeta_amplitude_phase_surrogates import (  # noqa: E402
    _spectrally_coherent_spatial_phase_surrogates,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_matrix_free import (  # noqa: E402
    apply_exact_visibility_operator_matrix_free,
)
from visibility_primary_beam import (  # noqa: E402
    open_indexed_frequency_row_direction_kernel_multiplier,
)
from visibility_qbeta import (  # noqa: E402
    build_sky_band_layout,
    source_bandpowers,
)
from visibility_qbeta_coarse import _realization_metrics  # noqa: E402
from visibility_qbeta_local_redshift import (  # noqa: E402
    frequency_subset_indices,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument(
        "--evaluation-sky-cache",
        type=Path,
        help=(
            "Optional independent full-band lightcone propagated with the "
            "shared covariance skies but never used to select windows."
        ),
    )
    parser.add_argument(
        "--aperture-row-beam-cache-pattern", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    parser.add_argument("--partition-count", type=int, default=4)
    parser.add_argument("--realization-count", type=int, default=64)
    parser.add_argument("--realization-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260731)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _load_sky_cache(
    path: Path, frequencies_mhz: np.ndarray
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    indices = frequency_subset_indices(
        np.asarray(data["frequencies_mhz"], dtype=np.float64),
        np.asarray(frequencies_mhz, dtype=np.float64),
        atol=1e-9,
    )
    if indices.size != np.asarray(data["frequencies_mhz"]).size:
        raise ValueError("Shared covariance requires the complete sky cache")
    return data


def realization_batches(
    realization_count: int,
    batch_size: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Return deterministic streamed realization ranges and seeds."""
    count = int(realization_count)
    size = int(batch_size)
    if count < 1 or size < 1:
        raise ValueError("Realization count and batch size must be positive")
    return [
        (
            first,
            min(count, first + size),
            int(seed) + 104729 * batch_index,
        )
        for batch_index, first in enumerate(range(0, count, size))
    ]


def _window_records(
    *,
    manifest_path: Path,
    local_root: Path,
    profile: str,
    partition_count: int,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in manifest["windows"]:
        label = str(row["label"])
        analysis_path = local_root / "configs" / str(
            row["analysis_config"]
        )
        input_path = local_root / "configs" / str(row["input_config"])
        analysis_config = json.loads(
            analysis_path.read_text(encoding="utf-8")
        )
        input_config = json.loads(
            input_path.read_text(encoding="utf-8")
        )
        resolved = resolve_mode_first_analysis(analysis_config)
        combined, combined_metadata = _load_result(
            local_root / label / "combined"
        )
        coarse_path = local_root / label / "coarse" / "products.npz"
        with np.load(coarse_path, allow_pickle=False) as archive:
            coarse = {
                name: np.asarray(archive[name]) for name in archive.files
            }
        partition_rows = []
        partition_metadata = []
        for partition in range(int(partition_count)):
            arrays, metadata = _load_result(
                local_root / label / f"part_{partition}" / "evaluate"
            )
            partition_rows.append(
                np.asarray(arrays["selected_bank_rows"], dtype=np.int64)
            )
            partition_metadata.append(metadata)
        prefix = f"{profile}_"
        required = {
            f"{prefix}selected",
            f"{prefix}target",
            f"{prefix}bank_total_estimate",
            f"{prefix}window",
            f"{prefix}transform",
            f"{prefix}response",
        }
        missing = sorted(required - coarse.keys())
        if missing:
            raise ValueError(
                f"{label} lacks shared-covariance products: {missing}"
            )
        records.append(
            {
                "label": label,
                "analysis_path": analysis_path,
                "input_path": input_path,
                "analysis_config": analysis_config,
                "input_config": input_config,
                "resolved": resolved,
                "combined": combined,
                "combined_metadata": combined_metadata,
                "coarse": coarse,
                "partition_rows": partition_rows,
                "partition_metadata": partition_metadata,
            }
        )
    if not records:
        raise ValueError("Local manifest contains no windows")
    reference_rows = records[0]["partition_rows"]
    for record in records[1:]:
        for current, reference in zip(
            record["partition_rows"], reference_rows, strict=True
        ):
            if not np.array_equal(current, reference):
                raise ValueError(
                    "Local windows do not use identical row partitions"
                )
    return records


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.realization_count) < 3:
        raise ValueError("At least three shared realizations are required")
    batches = realization_batches(
        int(args.realization_count),
        int(args.realization_batch_size),
        int(args.seed),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    products_dir = args.out_dir / "window_products"
    products_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    full_config = json.loads(
        args.full_config.read_text(encoding="utf-8")
    )
    full_resolved = resolve_mode_first_analysis(full_config)
    frequencies_mhz = np.asarray(
        full_resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    frequencies_hz = frequencies_mhz * 1e6
    bank, bank_manifest = _load_bank(
        args.bank_dir, requested_frequencies_hz=frequencies_hz
    )
    sky = _load_sky_cache(args.sky_cache, frequencies_mhz)
    if (
        not np.allclose(
            bank["frequencies_hz"],
            frequencies_hz,
            rtol=0.0,
            atol=1e-3,
        )
        or not np.allclose(
            sky["frequencies_mhz"],
            frequencies_mhz,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError("Full bank, sky, and config frequencies differ")
    records = _window_records(
        manifest_path=args.local_manifest,
        local_root=args.local_root,
        profile=str(args.profile),
        partition_count=int(args.partition_count),
    )
    for record in records:
        input_frequencies = np.asarray(
            record["input_config"]["frequencies_mhz"], dtype=np.float64
        )
        record["global_frequency_indices"] = frequency_subset_indices(
            frequencies_mhz, input_frequencies, atol=1e-9
        )
        analysis_frequencies = np.asarray(
            record["resolved"].geometry["frequencies_mhz"],
            dtype=np.float64,
        )
        record["analysis_frequency_indices"] = (
            _analysis_frequency_indices(
                input_frequencies, analysis_frequencies
            )
        )
        local_frequency_hz = input_frequencies * 1e6
        radial_mpc_per_hz = float(
            record["resolved"].geometry["radial_spacing_mpc"]
        ) / float(np.mean(np.diff(local_frequency_hz)))
        kperp_edges = np.asarray(
            record["resolved"].contract.window_layout.kperp_edges,
            dtype=np.float64,
        )
        record["kperp_edges"] = kperp_edges
        record["maximum_delays"] = _maximum_patch_delays(
            kperp_edges=kperp_edges,
            transverse_distance_mpc=float(
                record["resolved"].geometry["transverse_distance_mpc"]
            ),
            reference_frequency_hz=float(
                record["resolved"].geometry["reference_frequency_mhz"]
            )
            * 1e6,
            source_corner_angle_deg=float(
                record["resolved"].geometry["source_corner_angle_deg"]
            ),
            wedge_buffer_mpc_inv=float(
                record["resolved"].geometry["wedge_buffer_mpc_inv"]
            ),
            radial_mpc_per_hz=radial_mpc_per_hz,
        )
        output_ids = np.asarray(
            record["combined"]["output_band_ids"], dtype=np.int64
        )
        record["output_band_ids"] = output_ids
        record["q_sum"] = np.zeros(
            (int(args.realization_count), output_ids.size),
            dtype=np.float64,
        )
        if args.evaluation_sky_cache is not None:
            record["evaluation_pure_q_sum"] = np.zeros(
                output_ids.size, dtype=np.float64
            )
            record["evaluation_total_q_sum"] = np.zeros(
                output_ids.size, dtype=np.float64
            )

    contract_payload = {
        "schema": "visibility_qbeta_shared_fullband_realizations",
        "schema_version": 2,
        "frequencies_mhz": frequencies_mhz.tolist(),
        "bank_sha256": str(bank_manifest["bank_sha256"]),
        "sky_cache_sha256": _sha256(args.sky_cache),
        "seed": int(args.seed),
        "realization_count": int(args.realization_count),
        "realization_batch_size": int(args.realization_batch_size),
        "batch_seed_stride": 104729,
        "method": (
            "common 2D spatial phase randomization retaining every spatial "
            "mode's complete 128-frequency complex vector"
        ),
        "partition_rows": [
            rows.tolist() for rows in records[0]["partition_rows"]
        ],
    }
    shared_contract = _contract_sha256(contract_payload)
    evaluation_sky = None
    if args.evaluation_sky_cache is not None:
        evaluation_sky = _load_sky_cache(
            args.evaluation_sky_cache, frequencies_mhz
        )
        for name in ("l_cosine", "m_cosine", "n_minus_one"):
            if not np.array_equal(evaluation_sky[name], sky[name]):
                raise ValueError(
                    f"Independent full-band sky differs in {name}"
                )
        if not np.allclose(
            evaluation_sky["k2jy_per_pixel"],
            sky["k2jy_per_pixel"],
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "Independent full-band sky uses different K-to-Jy factors"
            )

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Shared-covariance propagation requires CUDA")
    torch.cuda.set_device(device)
    real_dtype = (
        torch.float32
        if str(args.operator_dtype) == "complex64"
        else torch.float64
    )
    source_size = int(full_config["image_geometry"]["source_image_size"])
    eor_jy = torch.as_tensor(
        sky["eor_jy"].reshape(
            frequencies_mhz.size, source_size, source_size
        ),
        dtype=real_dtype,
        device=device,
    )
    k2jy = torch.as_tensor(
        sky["k2jy_per_pixel"],
        dtype=real_dtype,
        device=device,
    )[:, None, None]
    eor_k = eor_jy / k2jy
    evaluation_tensor = None
    if evaluation_sky is not None:
        evaluation_tensor = torch.as_tensor(
            evaluation_sky["eor_jy"].reshape(
                1, frequencies_mhz.size, source_size, source_size
            ),
            dtype=real_dtype,
            device=device,
        )
    del eor_jy
    torch.cuda.empty_cache()

    cache_dirs = [
        _format_pattern(
            str(args.aperture_row_beam_cache_pattern),
            float(frequency_mhz),
        )
        for frequency_mhz in frequencies_mhz
    ]
    partition_contexts = []
    for partition_index, selected_rows in enumerate(
        records[0]["partition_rows"]
    ):
        uvw = np.asarray(
            bank["sample_uvw_m"][selected_rows], dtype=np.float64
        )
        (
            beam_multiplier,
            cache_metadata,
            _,
        ) = open_indexed_frequency_row_direction_kernel_multiplier(
            cache_dirs, selected_bank_rows=selected_rows
        )
        for frequency_hz, row in zip(
            frequencies_hz, cache_metadata, strict=True
        ):
            if (
                tuple(int(value) for value in row["shape"])[1]
                != int(sky["l_cosine"].size)
                or not np.isclose(
                    float(row["frequency_hz"]),
                    float(frequency_hz),
                    rtol=0.0,
                    atol=1e-3,
                )
            ):
                raise ValueError(
                    "Full-band aperture cache geometry differs"
                )
        partition_contexts.append(
            {
                "partition_index": int(partition_index),
                "selected_rows": selected_rows,
                "uvw": uvw,
                "beam_multiplier": beam_multiplier,
            }
        )

    for batch_index, (batch_first, batch_stop, batch_seed) in enumerate(
        batches
    ):
        shared_skies = _spectrally_coherent_spatial_phase_surrogates(
            torch=torch,
            restricted_k=eor_k,
            k2jy=k2jy,
            repeats=batch_stop - batch_first,
            seed=batch_seed,
            real_dtype=real_dtype,
        )
        evaluation_offset = 0
        if evaluation_tensor is not None and batch_index == 0:
            shared_skies = torch.cat(
                (evaluation_tensor, shared_skies), dim=0
            )
            evaluation_offset = 1
        for context in partition_contexts:
            partition_index = int(context["partition_index"])
            selected_rows = context["selected_rows"]
            uvw = context["uvw"]

            def report_frequency(
                frequency_index: int, frequency_hz: float
            ) -> None:
                print(
                    json.dumps(
                        {
                            "event": (
                                "shared_covariance_operator_frequency"
                            ),
                            "batch_index": int(batch_index),
                            "batch_first": int(batch_first),
                            "batch_stop": int(batch_stop),
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

            visibilities = apply_exact_visibility_operator_matrix_free(
                torch=torch,
                frequencies_hz=frequencies_hz,
                uvw_m=uvw,
                l_cosine=sky["l_cosine"],
                m_cosine=sky["m_cosine"],
                n_minus_one=sky["n_minus_one"],
                sky_jy=shared_skies,
                channel_bandwidth_hz=float(args.channel_bandwidth_hz),
                integration_time_s=float(args.integration_time_s),
                phase_dec_deg=float(args.phase_dec_deg),
                device=device,
                operator_dtype=str(args.operator_dtype),
                row_chunk=int(args.row_chunk),
                source_chunk=int(args.source_chunk),
                kernel_multiplier=context["beam_multiplier"],
                progress_callback=report_frequency,
            )
            for record in records:
                indices = record["global_frequency_indices"]
                pure_local_vis = visibilities[:, indices]
                local_vis = (
                    pure_local_vis
                    + np.asarray(
                        bank["sample_fg"][indices][:, selected_rows],
                        dtype=np.complex128,
                    )[None, ...]
                )
                reference_frequency_hz = (
                    float(
                        record["resolved"].geometry[
                            "reference_frequency_mhz"
                        ]
                    )
                    * 1e6
                )
                row_kperp = _row_kperp(
                    uvw,
                    reference_frequency_hz=reference_frequency_hz,
                    transverse_distance_mpc=float(
                        record["resolved"].geometry[
                            "transverse_distance_mpc"
                        ]
                    ),
                )
                settings = record["partition_metadata"][
                    partition_index
                ]["settings"]
                q, _, _, _, _ = _visibility_bandpowers(
                    visibilities=local_vis,
                    frequencies_hz=frequencies_hz[indices],
                    analysis_frequency_indices=record[
                        "analysis_frequency_indices"
                    ],
                    filter_bandwidth_scope=str(
                        settings["filter_bandwidth_scope"]
                    ),
                    row_kperp=row_kperp,
                    kperp_edges=record["kperp_edges"],
                    maximum_delays_s=record["maximum_delays"],
                    dpss_eigenvalue_threshold=float(
                        settings["dpss_eigenvalue_threshold"]
                    ),
                    foreground_filter=str(
                        settings["foreground_filter"]
                    ),
                    suppression_strength=float(
                        settings["suppression_strength"]
                    ),
                    polynomial_degree=int(
                        settings["polynomial_degree"]
                    ),
                    spectral_taper=str(settings["spectral_taper"]),
                )
                flat_q = q.reshape(q.shape[0], -1)[
                    :, record["output_band_ids"]
                ]
                record["q_sum"][batch_first:batch_stop] += flat_q[
                    evaluation_offset:
                ]
                if evaluation_offset:
                    pure_q, _, _, _, _ = _visibility_bandpowers(
                        visibilities=pure_local_vis[0],
                        frequencies_hz=frequencies_hz[indices],
                        analysis_frequency_indices=record[
                            "analysis_frequency_indices"
                        ],
                        filter_bandwidth_scope=str(
                            settings["filter_bandwidth_scope"]
                        ),
                        row_kperp=row_kperp,
                        kperp_edges=record["kperp_edges"],
                        maximum_delays_s=record["maximum_delays"],
                        dpss_eigenvalue_threshold=float(
                            settings["dpss_eigenvalue_threshold"]
                        ),
                        foreground_filter=str(
                            settings["foreground_filter"]
                        ),
                        suppression_strength=float(
                            settings["suppression_strength"]
                        ),
                        polynomial_degree=int(
                            settings["polynomial_degree"]
                        ),
                        spectral_taper=str(settings["spectral_taper"]),
                    )
                    record["evaluation_pure_q_sum"] += (
                        pure_q.reshape(-1)[record["output_band_ids"]]
                    )
                    record["evaluation_total_q_sum"] += flat_q[0]
            del visibilities
        del shared_skies
        torch.cuda.empty_cache()

    profile = str(args.profile)
    prefix = f"{profile}_"
    output_paths = []
    independent_results = []
    for record in records:
        mean_q = record["q_sum"] / int(args.partition_count)
        coarse = record["coarse"]
        transform = np.asarray(
            coarse[f"{prefix}transform"], dtype=np.float64
        )
        response = np.asarray(
            coarse[f"{prefix}response"], dtype=np.float64
        )
        estimate = (mean_q @ transform.T) / np.sum(response, axis=1)[
            None, :
        ]
        output_path = products_dir / f"{record['label']}.npz"
        payload = {
            f"{prefix}selected": coarse[f"{prefix}selected"],
            f"{prefix}target": coarse[f"{prefix}target"],
            f"{prefix}bank_total_estimate": coarse[
                f"{prefix}bank_total_estimate"
            ],
            f"{prefix}heldout_total_estimate": estimate,
            f"{prefix}window": coarse[f"{prefix}window"],
            "shared_realization_contract_sha256": np.asarray(
                shared_contract
            ),
            "shared_realization_source": np.asarray(
                contract_payload["method"]
            ),
        }
        if evaluation_sky is not None:
            indices = record["global_frequency_indices"]
            input_count = int(indices.size)
            evaluation_k = (
                evaluation_sky["eor_jy"][indices]
                / evaluation_sky["k2jy_per_pixel"][indices, None]
            ).reshape(input_count, source_size, source_size)
            layout = build_sky_band_layout(
                (input_count, source_size, source_size),
                dx_mpc=float(
                    record["resolved"].contract.full_layout.dx_mpc
                ),
                dy_mpc=float(
                    record["resolved"].contract.full_layout.dy_mpc
                ),
                dpar_mpc=float(
                    record["resolved"].contract.full_layout.dpar_mpc
                ),
                kperp_edges=record["kperp_edges"],
                exclude_radial_nyquist=False,
            )
            source_ids = np.asarray(
                record["combined"]["source_band_ids"], dtype=np.int64
            )
            if (
                layout.band_count <= int(np.max(source_ids))
                or not np.array_equal(
                    layout.active_kperp_indices[source_ids],
                    record["combined"]["source_band_kperp_indices"],
                )
                or not np.array_equal(
                    layout.active_kpar_indices[source_ids],
                    record["combined"]["source_band_kpar_indices"],
                )
            ):
                raise ValueError(
                    "Independent lightcone source layout differs"
                )
            spectrum = np.fft.fftn(evaluation_k, norm="ortho")
            restricted_k = np.fft.ifftn(
                np.where(
                    np.isin(layout.mode_bands, source_ids),
                    spectrum,
                    0.0,
                ),
                norm="ortho",
            ).real
            source_power = source_bandpowers(
                restricted_k, layout
            )[source_ids]
            evaluation_target = np.asarray(
                coarse[f"{prefix}window"], dtype=np.float64
            ) @ source_power
            pure_q = (
                record["evaluation_pure_q_sum"]
                / int(args.partition_count)
            )
            total_q = (
                record["evaluation_total_q_sum"]
                / int(args.partition_count)
            )
            row_sum = np.sum(response, axis=1)
            pure_estimate = (pure_q @ transform.T) / row_sum
            total_estimate = (total_q @ transform.T) / row_sum
            selected = np.asarray(
                coarse[f"{prefix}selected"], dtype=bool
            )
            metric_weights = np.asarray(
                coarse[f"{prefix}group_metric_weights"],
                dtype=np.float64,
            )
            independent_results.append(
                {
                    "label": record["label"],
                    "pure_eor": _realization_metrics(
                        pure_estimate[selected],
                        evaluation_target[selected],
                        metric_weights[selected],
                    ),
                    "fg_plus_eor": _realization_metrics(
                        total_estimate[selected],
                        evaluation_target[selected],
                        metric_weights[selected],
                    ),
                }
            )
            payload.update(
                {
                    "independent_source_power": source_power,
                    f"{prefix}independent_target": evaluation_target,
                    f"{prefix}independent_pure_estimate": pure_estimate,
                    f"{prefix}independent_total_estimate": total_estimate,
                }
            )
        _atomic_npz(
            output_path,
            payload,
        )
        output_paths.append(
            {"label": record["label"], "path": str(output_path)}
        )
    result = {
        "schema": "visibility_qbeta_shared_local_covariance_inputs",
        "schema_version": 1,
        "elapsed_seconds": float(time.monotonic() - started),
        "shared_realization_contract_sha256": shared_contract,
        "contract": contract_payload,
        "window_products": output_paths,
        "independent_evaluation_sky_cache_sha256": (
            None
            if args.evaluation_sky_cache is None
            else _sha256(args.evaluation_sky_cache)
        ),
        "independent_lightcone_by_window": independent_results,
        "selection_uses_shared_realizations": False,
        "selection_uses_independent_lightcone": False,
    }
    _atomic_json(args.out_dir / "manifest.json", result)
    print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
