#!/usr/bin/env python3
"""Evaluate a frozen Q_beta response on an independent physical lightcone."""

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
    _operator_closure_metrics,
    _row_kperp,
    _visibility_bandpowers,
    _windowed_metrics,
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
from visibility_qbeta_local_redshift import (  # noqa: E402
    frequency_subset_indices,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frequency-config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--calibration-sky-cache", type=Path, required=True)
    parser.add_argument("--evaluation-sky-cache", type=Path, required=True)
    parser.add_argument("--combined-result-dir", type=Path, required=True)
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
    parser.add_argument("--maximum-operator-closure", type=float, default=1e-4)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_result(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(directory / "result.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    return arrays, metadata


def _load_sky_cache(
    path: Path, requested_frequencies_mhz: np.ndarray
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    indices = frequency_subset_indices(
        np.asarray(data["frequencies_mhz"], dtype=np.float64),
        np.asarray(requested_frequencies_mhz, dtype=np.float64),
        atol=1e-9,
    )
    data["frequencies_mhz"] = np.asarray(
        data["frequencies_mhz"], dtype=np.float64
    )[indices]
    data["eor_jy"] = np.asarray(data["eor_jy"])[indices]
    data["k2jy_per_pixel"] = np.asarray(
        data["k2jy_per_pixel"]
    )[indices]
    data["parent_frequency_indices"] = indices
    return data


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
    calibration_sky = _load_sky_cache(
        args.calibration_sky_cache, frequencies_mhz
    )
    evaluation_sky = _load_sky_cache(
        args.evaluation_sky_cache, frequencies_mhz
    )
    for name in ("l_cosine", "m_cosine", "n_minus_one"):
        if not np.array_equal(calibration_sky[name], evaluation_sky[name]):
            raise ValueError(f"Independent sky differs in {name}")
    if not np.allclose(
        calibration_sky["k2jy_per_pixel"],
        evaluation_sky["k2jy_per_pixel"],
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError("Independent sky uses different K-to-Jy factors")
    bank, bank_manifest = _load_bank(
        args.bank_dir, requested_frequencies_hz=frequencies_hz
    )
    combined, combined_metadata = _load_result(
        args.combined_result_dir
    )
    partition_results = [
        _load_result(directory) for directory in args.partition_result_dir
    ]
    if len(partition_results) < 2:
        raise ValueError("At least two row partitions are required")
    partition_rows = [
        np.asarray(arrays["selected_bank_rows"], dtype=np.int64)
        for arrays, _ in partition_results
    ]
    if not np.array_equal(
        np.sort(np.concatenate(partition_rows)),
        np.sort(
            np.asarray(combined["selected_bank_rows"], dtype=np.int64)
        ),
    ):
        raise ValueError("Partition rows do not reproduce the combined result")
    for arrays, metadata in partition_results:
        if not np.array_equal(
            arrays["source_band_ids"], combined["source_band_ids"]
        ):
            raise ValueError("Partition source bands differ")
        if (
            metadata["analysis_contract_sha256"]
            != combined_metadata["analysis_contract_sha256"]
        ):
            raise ValueError("Partition analysis contracts differ")

    source_size = int(config["image_geometry"]["source_image_size"])
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    source_layout = build_sky_band_layout(
        (frequencies_mhz.size, source_size, source_size),
        dx_mpc=float(resolved.contract.full_layout.dx_mpc),
        dy_mpc=float(resolved.contract.full_layout.dy_mpc),
        dpar_mpc=float(resolved.contract.full_layout.dpar_mpc),
        kperp_edges=kperp_edges,
        exclude_radial_nyquist=False,
    )
    source_band_ids = np.asarray(
        combined["source_band_ids"], dtype=np.int64
    )
    if (
        source_layout.band_count <= int(np.max(source_band_ids))
        or not np.array_equal(
            source_layout.active_kperp_indices[source_band_ids],
            combined["source_band_kperp_indices"],
        )
        or not np.array_equal(
            source_layout.active_kpar_indices[source_band_ids],
            combined["source_band_kpar_indices"],
        )
    ):
        raise ValueError("Independent sky-band layout differs")
    evaluation_k = (
        evaluation_sky["eor_jy"]
        / evaluation_sky["k2jy_per_pixel"][:, None]
    ).reshape(frequencies_mhz.size, source_size, source_size)
    evaluation_spectrum = np.fft.fftn(evaluation_k, norm="ortho")
    source_mask = np.isin(source_layout.mode_bands, source_band_ids)
    restricted_k = np.fft.ifftn(
        np.where(source_mask, evaluation_spectrum, 0.0),
        norm="ortho",
    ).real
    evaluation_source_power = source_bandpowers(
        restricted_k, source_layout
    )[source_band_ids]

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Independent-lightcone propagation requires CUDA")
    torch.cuda.set_device(device)
    real_dtype = (
        torch.float32
        if str(args.operator_dtype) == "complex64"
        else torch.float64
    )
    skies = torch.as_tensor(
        np.stack(
            (
                calibration_sky["eor_jy"],
                evaluation_sky["eor_jy"],
            )
        ).reshape(
            2, frequencies_mhz.size, source_size, source_size
        ),
        dtype=real_dtype,
        device=device,
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
    evaluation_q_by_partition = []
    total_q_by_partition = []
    closure_by_partition = []
    cache_hashes: list[list[str]] = []
    for partition_index, ((arrays, metadata), selected_rows) in enumerate(
        zip(partition_results, partition_rows, strict=True)
    ):
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
                != int(calibration_sky["l_cosine"].size)
                or not np.isclose(
                    float(row["frequency_hz"]),
                    float(frequency_hz),
                    rtol=0.0,
                    atol=1e-3,
                )
            ):
                raise ValueError(
                    "Aperture cache geometry or frequency differs"
                )

        def report_frequency(
            frequency_index: int, frequency_hz: float
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "independent_operator_frequency",
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

        propagated = apply_exact_visibility_operator_matrix_free(
            torch=torch,
            frequencies_hz=frequencies_hz,
            uvw_m=uvw,
            l_cosine=calibration_sky["l_cosine"],
            m_cosine=calibration_sky["m_cosine"],
            n_minus_one=calibration_sky["n_minus_one"],
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
        closure = _operator_closure_metrics(
            propagated[0],
            np.asarray(
                bank["sample_eor"][:, selected_rows],
                dtype=np.complex128,
            ),
        )
        if float(closure["relative_l2"]) > float(
            args.maximum_operator_closure
        ):
            raise RuntimeError(
                "Independent-lightcone base closure failed: "
                f"{closure['relative_l2']:.6g}"
            )
        closure_by_partition.append(closure)
        settings = metadata["settings"]
        bandpower_kwargs = {
            "frequencies_hz": frequencies_hz,
            "analysis_frequency_indices": analysis_indices,
            "filter_bandwidth_scope": str(
                settings["filter_bandwidth_scope"]
            ),
            "row_kperp": row_kperp,
            "kperp_edges": kperp_edges,
            "maximum_delays_s": maximum_delays,
            "dpss_eigenvalue_threshold": float(
                settings["dpss_eigenvalue_threshold"]
            ),
            "foreground_filter": str(settings["foreground_filter"]),
            "suppression_strength": float(
                settings["suppression_strength"]
            ),
            "polynomial_degree": int(settings["polynomial_degree"]),
            "spectral_taper": str(settings["spectral_taper"]),
        }
        output_band_ids = np.asarray(
            arrays["output_band_ids"], dtype=np.int64
        )
        evaluation_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=propagated[1], **bandpower_kwargs
        )
        total_q, _, _, _, _ = _visibility_bandpowers(
            visibilities=(
                propagated[1]
                + np.asarray(
                    bank["sample_fg"][:, selected_rows],
                    dtype=np.complex128,
                )
            ),
            **bandpower_kwargs,
        )
        evaluation_q_by_partition.append(
            evaluation_q.reshape(-1)[output_band_ids]
        )
        total_q_by_partition.append(
            total_q.reshape(-1)[output_band_ids]
        )
    evaluation_q = np.mean(
        np.stack(evaluation_q_by_partition), axis=0
    )
    total_q = np.mean(np.stack(total_q_by_partition), axis=0)
    response = np.asarray(
        combined["calibration_response"], dtype=np.float64
    )
    reporting_positions = np.asarray(
        combined["reporting_source_positions"], dtype=np.int64
    )
    evaluation_windowed = _windowed_metrics(
        response=response,
        observed_q=evaluation_q,
        source_power=evaluation_source_power,
        minimum_relative_response=0.1,
        target_source_positions=reporting_positions,
        minimum_target_window_fraction=0.8,
    )
    total_windowed = _windowed_metrics(
        response=response,
        observed_q=total_q,
        source_power=evaluation_source_power,
        minimum_relative_response=0.1,
        target_source_positions=reporting_positions,
        minimum_target_window_fraction=0.8,
    )
    selected_positions = np.asarray(
        evaluation_windowed["selected_output_positions"], dtype=np.int64
    )
    if not np.array_equal(
        selected_positions,
        np.asarray(
            combined["qbeta_selected_window_positions"], dtype=np.int64
        ),
    ):
        raise ValueError("Frozen response selection is not reproducible")
    result = {
        "schema": "visibility_qbeta_independent_lightcone_evaluation",
        "schema_version": 1,
        "elapsed_seconds": float(time.monotonic() - started),
        "partition_count": int(len(partition_results)),
        "bank_sha256": bank_manifest["bank_sha256"],
        "calibration_sky_cache_sha256": _sha256(
            args.calibration_sky_cache
        ),
        "evaluation_sky_cache_sha256": _sha256(
            args.evaluation_sky_cache
        ),
        "calibration_parent_frequency_indices": calibration_sky[
            "parent_frequency_indices"
        ],
        "evaluation_parent_frequency_indices": evaluation_sky[
            "parent_frequency_indices"
        ],
        "operator_closure_by_partition": closure_by_partition,
        "selected_window_count": int(selected_positions.size),
        "independent_eor": evaluation_windowed["realizations"][0],
        "independent_fg_plus_eor": total_windowed["realizations"][0],
        "base_source_power_integrated_ratio": float(
            np.sum(
                combined["source_band_mode_counts"]
                * evaluation_source_power
            )
            / np.sum(
                combined["source_band_mode_counts"]
                * combined["restricted_eor_source_power"]
            )
        ),
        "cache_data_sha256_by_partition": cache_hashes,
        "limitations": [
            "no thermal noise",
            "the independent lightcone shares only the observing geometry",
            "response calibration and selection use no independent-sky truth",
        ],
    }
    _atomic_npz(
        args.out_dir / "products.npz",
        {
            "evaluation_source_power": evaluation_source_power,
            "evaluation_q": evaluation_q,
            "total_q": total_q,
            "selected_window_positions": selected_positions,
            "target_windowed_power": np.asarray(
                evaluation_windowed["target_windowed_power"]
            ),
            "evaluation_windowed_power": np.asarray(
                evaluation_windowed["estimated_windowed_power"][0]
            ),
            "total_windowed_power": np.asarray(
                total_windowed["estimated_windowed_power"][0]
            ),
        },
    )
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
