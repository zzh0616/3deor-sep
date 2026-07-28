#!/usr/bin/env python3
"""Close the exact OSKAR station-pair aperture-beam operator."""

from __future__ import annotations

import argparse
import json
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

from evaluate_oskar_gaussian_beam_closure import (  # noqa: E402
    _sky_plane,
    visibility_closure_metrics,
)
from visibility_matrix_free import (  # noqa: E402
    apply_exact_visibility_operator_matrix_free,
)
from visibility_primary_beam import (  # noqa: E402
    direction_cosine_geometry_sha256,
    open_indexed_frequency_row_direction_kernel_multiplier,
    open_row_direction_kernel_multiplier,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-shard", type=Path, required=True)
    parser.add_argument("--row-beam-cache", type=Path, required=True)
    parser.add_argument("--selected-row-result", type=Path)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument("--row-chunk", type=int, default=32)
    parser.add_argument("--source-chunk", type=int, default=32768)
    parser.add_argument(
        "--operator-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--maximum-relative-l2", type=float, default=1e-5)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank_shard, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        sky = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(
        args.row_beam_cache / "geometry.npz", allow_pickle=False
    ) as archive:
        geometry = {name: np.asarray(archive[name]) for name in archive.files}
    cached_rows = np.asarray(
        geometry["selected_bank_rows"], dtype=np.int64
    )
    if args.selected_row_result is None:
        rows = cached_rows
        multiplier, beam_metadata = open_row_direction_kernel_multiplier(
            args.row_beam_cache
        )
    else:
        with np.load(
            args.selected_row_result, allow_pickle=False
        ) as selected_archive:
            rows = np.asarray(
                selected_archive["selected_bank_rows"], dtype=np.int64
            )
        multiplier, metadata, cached_rows_by_frequency = (
            open_indexed_frequency_row_direction_kernel_multiplier(
                [args.row_beam_cache],
                selected_bank_rows=rows,
            )
        )
        beam_metadata = metadata[0]
        if not np.array_equal(
            cached_rows_by_frequency[0], cached_rows
        ):
            raise ValueError("Indexed row-beam geometry changed while opening")
    if tuple(int(value) for value in beam_metadata["shape"]) != (
        int(cached_rows.size),
        int(np.asarray(sky["l_cosine"]).size),
    ):
        raise ValueError("Row-beam cache geometry differs from the sky")
    expected_direction_sha256 = direction_cosine_geometry_sha256(
        l_cosine=sky["l_cosine"],
        m_cosine=sky["m_cosine"],
        n_minus_one=sky["n_minus_one"],
    )
    cached_direction_sha256 = beam_metadata.get("direction_sha256")
    if (
        cached_direction_sha256 is not None
        and str(cached_direction_sha256) != expected_direction_sha256
    ):
        raise ValueError("Row-beam cache source-direction order differs")
    frequency_hz = float(bank["frequency_hz"].item())
    if not np.isclose(
        float(beam_metadata["frequency_hz"]),
        frequency_hz,
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError("Row-beam cache and visibility frequencies differ")

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Aperture-array closure requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    predicted = apply_exact_visibility_operator_matrix_free(
        torch=torch,
        frequencies_hz=np.asarray([frequency_hz]),
        uvw_m=np.asarray(bank["sample_uvw_m"][rows], dtype=np.float64),
        l_cosine=sky["l_cosine"],
        m_cosine=sky["m_cosine"],
        n_minus_one=sky["n_minus_one"],
        sky_jy=_sky_plane(sky, frequency_hz),
        channel_bandwidth_hz=float(args.channel_bandwidth_hz),
        integration_time_s=float(args.integration_time_s),
        phase_dec_deg=float(args.phase_dec_deg),
        device=device,
        operator_dtype=str(args.operator_dtype),
        row_chunk=int(args.row_chunk),
        source_chunk=int(args.source_chunk),
        kernel_multiplier=multiplier,
    )
    elapsed = float(time.monotonic() - started)
    target = np.asarray(bank["sample_eor"][rows], dtype=np.complex128)[None, :]
    metrics = visibility_closure_metrics(predicted, target)
    result = {
        "schema": "oskar_aperture_array_row_beam_matrix_free_closure",
        "schema_version": 1,
        "passed": bool(
            metrics["relative_l2"] <= float(args.maximum_relative_l2)
        ),
        "bank_shard": str(args.bank_shard),
        "row_beam_cache": str(args.row_beam_cache),
        "sky_cache": str(args.sky_cache),
        "frequency_mhz": float(frequency_hz / 1e6),
        "selected_row_count": int(rows.size),
        "cached_row_count": int(cached_rows.size),
        "beam": {
            "definition": str(beam_metadata["definition"]),
            "dtype": str(beam_metadata["dtype"]),
            "shape": list(beam_metadata["shape"]),
            "cache_elapsed_seconds": float(
                beam_metadata["elapsed_seconds"]
            ),
        },
        "operator": {
            "implementation": (
                "matrix_free_exact_dft_with_oskar_station_pair_coherency"
            ),
            "dtype": str(args.operator_dtype),
            "row_chunk": int(args.row_chunk),
            "source_chunk": int(args.source_chunk),
            "channel_bandwidth_hz": float(args.channel_bandwidth_hz),
            "integration_time_s": float(args.integration_time_s),
            "elapsed_seconds": elapsed,
            "rows_per_second": float(rows.size / max(elapsed, 1e-12)),
            "device_name": str(torch.cuda.get_device_name(device)),
            "peak_allocated_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            "peak_reserved_gib": float(
                torch.cuda.max_memory_reserved(device) / 2**30
            ),
        },
        "maximum_relative_l2": float(args.maximum_relative_l2),
        "closure": metrics,
    }
    np.savez_compressed(
        args.out_dir / "result.npz",
        selected_bank_rows=rows,
        predicted_eor_visibility=np.asarray(predicted),
        target_eor_visibility=target,
    )
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "Aperture-array row-beam closure failed: "
            f"{metrics['relative_l2']:.6g} > "
            f"{float(args.maximum_relative_l2):.6g}"
        )


if __name__ == "__main__":
    main()
