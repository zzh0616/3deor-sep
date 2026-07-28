#!/usr/bin/env python3
"""Close a cached aperture-array PB operator against OSKAR visibilities."""

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
    _selected_rows,
    _sky_plane,
    visibility_closure_metrics,
)
from visibility_matrix_free import (  # noqa: E402
    apply_exact_visibility_operator_matrix_free,
)
from visibility_primary_beam import (  # noqa: E402
    build_time_direction_kernel_multiplier,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-shard", type=Path, required=True)
    parser.add_argument("--beam-cache", type=Path, required=True)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selected-row-result", type=Path)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument("--row-chunk", type=int, default=64)
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


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _row_time_indices(
    all_times: np.ndarray, selected_rows: np.ndarray, time_count: int
) -> np.ndarray:
    times = np.asarray(all_times, dtype=np.float64).reshape(-1)
    unique = np.unique(times)
    if unique.size != int(time_count):
        raise ValueError(
            f"Bank has {unique.size} times but beam cache has {time_count}"
        )
    indices = np.searchsorted(unique, times[selected_rows])
    if not np.array_equal(unique[indices], times[selected_rows]):
        raise ValueError("Selected visibility time does not map to beam cache")
    return indices.astype(np.int64)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank_shard, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.beam_cache, allow_pickle=False) as archive:
        beam = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        sky = {name: np.asarray(archive[name]) for name in archive.files}
    station_type = str(bank.get("station_type", np.asarray("")).item())
    if station_type != "aperture_array":
        raise ValueError("Closure requires an aperture-array OSKAR shard")
    frequency_hz = float(bank["frequency_hz"].item())
    if not np.isclose(
        float(beam["frequency_hz"].item()),
        frequency_hz,
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError("Beam-cache and visibility frequencies differ")
    rows = _selected_rows(
        bank_row_count=int(bank["sample_uvw_m"].shape[0]),
        selected_result=args.selected_row_result,
        maximum_rows=args.maximum_rows,
    )
    beam_power = np.asarray(beam["stokes_i_power"], dtype=np.float64)
    if beam_power.ndim != 2 or beam_power.shape[1] != int(
        np.asarray(sky["l_cosine"]).size
    ):
        raise ValueError("Beam cache and sky directions differ")
    time_indices = _row_time_indices(
        bank["sample_time_s"], rows, beam_power.shape[0]
    )

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Aperture-array closure requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    multiplier = build_time_direction_kernel_multiplier(
        torch=torch,
        values=beam_power[None, ...],
        row_time_indices=time_indices,
        device=device,
        operator_dtype=str(args.operator_dtype),
    )
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
        "schema": "oskar_aperture_array_beam_matrix_free_closure",
        "schema_version": 1,
        "passed": bool(
            metrics["relative_l2"] <= float(args.maximum_relative_l2)
        ),
        "bank_shard": str(args.bank_shard),
        "beam_cache": str(args.beam_cache),
        "sky_cache": str(args.sky_cache),
        "frequency_mhz": float(frequency_hz / 1e6),
        "selected_row_count": int(rows.size),
        "beam": {
            "station_type": station_type,
            "station_id": int(beam["station_id"].item()),
            "time_count": int(beam_power.shape[0]),
            "source_count": int(beam_power.shape[1]),
            "station_beam_duplication": True,
        },
        "operator": {
            "implementation": (
                "matrix_free_exact_dft_with_oskar_aperture_power"
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
    _atomic_npz(
        args.out_dir / "result.npz",
        {
            "selected_bank_rows": rows,
            "selected_row_time_indices": time_indices,
            "predicted_eor_visibility": np.asarray(predicted),
            "target_eor_visibility": target,
        },
    )
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "Aperture-array PB operator closure failed: "
            f"{metrics['relative_l2']:.6g} > "
            f"{float(args.maximum_relative_l2):.6g}"
        )


if __name__ == "__main__":
    main()
