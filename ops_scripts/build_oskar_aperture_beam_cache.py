#!/usr/bin/env python3
"""Build an OSKAR aperture-array Stokes-I beam cache on OSM directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


INTEGER_HEADER = re.compile(r"^# ([^:]+):\s+(\d+)\s*$")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--oskar", required=True)
    parser.add_argument("--telescope-dir", type=Path, required=True)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--frequency-mhz", type=float, required=True)
    parser.add_argument("--phase-ra-deg", type=float, default=0.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument("--start-time-utc", default="2030-01-01T06:30:00.0")
    parser.add_argument("--observation-length-s", type=float, default=320.0)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--station-id", type=int, default=0)
    parser.add_argument("--max-sources-per-chunk", type=int, default=131072)
    parser.add_argument("--use-gpus", action="store_true")
    parser.add_argument("--keep-text", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_config(path: Path, *, args: argparse.Namespace, root: Path) -> None:
    path.write_text(
        "[General]\n"
        "app=oskar_sim_beam_pattern\n"
        "version=2.12.2\n\n"
        "[simulator]\n"
        "double_precision=true\n"
        f"use_gpus={'true' if args.use_gpus else 'false'}\n"
        "num_devices=1\n"
        f"max_sources_per_chunk={int(args.max_sources_per_chunk)}\n"
        "keep_log_file=false\n\n"
        "[observation]\n"
        f"phase_centre_ra_deg={float(args.phase_ra_deg):.12g}\n"
        f"phase_centre_dec_deg={float(args.phase_dec_deg):.12g}\n"
        f"start_frequency_hz={float(args.frequency_mhz) * 1e6:.1f}\n"
        "num_channels=1\n"
        "frequency_inc_hz=0.0\n"
        f"start_time_utc={args.start_time_utc}\n"
        f"length={float(args.observation_length_s):.12g}\n"
        f"num_time_steps={int(args.time_steps)}\n\n"
        "[telescope]\n"
        f"input_directory={args.telescope_dir}\n"
        "normalise_beams_at_phase_centre=true\n"
        "allow_station_beam_duplication=true\n"
        "pol_mode=Full\n"
        "station_type=Aperture array\n\n"
        "[beam_pattern]\n"
        "all_stations=false\n"
        f"station_ids={int(args.station_id)}\n"
        "coordinate_frame=Equatorial\n"
        "coordinate_type=Sky model\n"
        f"sky_model/file={args.osm}\n"
        f"root_path={root}\n"
        "output/separate_time_and_channel=true\n"
        "station_outputs/text_file/auto_power=true\n"
        "test_source/stokes_i=true\n",
        encoding="utf-8",
    )


def parse_oskar_auto_power_i(path: Path) -> np.ndarray:
    """Return [time, channel, source] values from OSKAR's chunked text."""
    header: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = INTEGER_HEADER.match(line.rstrip())
            if match:
                header[match.group(1)] = int(match.group(2))
    required = {
        "Number of pixel chunks",
        "Number of times (output)",
        "Number of channels (output)",
        "Maximum pixel chunk size",
        "Total number of pixels",
    }
    missing = sorted(required - header.keys())
    if missing:
        raise ValueError(f"Beam text lacks header keys: {', '.join(missing)}")
    values = np.asarray(np.loadtxt(path, comments="#"), dtype=np.float64).reshape(-1)
    chunk_count = header["Number of pixel chunks"]
    time_count = header["Number of times (output)"]
    channel_count = header["Number of channels (output)"]
    maximum_chunk = header["Maximum pixel chunk size"]
    source_count = header["Total number of pixels"]
    expected = int(time_count) * int(channel_count) * int(source_count)
    if values.size != expected:
        raise ValueError(f"Beam text has {values.size} values; expected {expected}")
    output = np.empty(
        (int(time_count), int(channel_count), int(source_count)),
        dtype=np.float64,
    )
    value_first = 0
    source_first = 0
    for chunk_index in range(int(chunk_count)):
        current = min(int(maximum_chunk), int(source_count) - source_first)
        value_count = int(time_count) * int(channel_count) * current
        block = values[value_first : value_first + value_count].reshape(
            int(time_count), int(channel_count), current
        )
        output[:, :, source_first : source_first + current] = block
        source_first += current
        value_first += value_count
    if source_first != int(source_count) or value_first != values.size:
        raise ValueError("Beam text chunk geometry does not close")
    return output


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if min(
        int(args.time_steps),
        int(args.max_sources_per_chunk),
        int(args.station_id) + 1,
    ) < 1:
        raise ValueError("Time, chunk, and station settings must be valid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = args.out_dir / "beam_pattern.ini"
    root = args.out_dir / "beam"
    _write_config(config, args=args, root=root)
    log = args.out_dir / "oskar.log"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [str(args.oskar), str(config)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    matches = sorted(
        args.out_dir.glob(
            f"beam_S{int(args.station_id):04d}_"
            "*_AUTO_POWER_AMP_I_I.txt"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Stokes-I auto-power file, got {matches}")
    beam = parse_oskar_auto_power_i(matches[0])
    if beam.shape[1] != 1 or beam.shape[0] != int(args.time_steps):
        raise ValueError("Parsed beam time/frequency geometry differs")
    result_path = args.out_dir / "beam_cache.npz"
    _atomic_npz(
        result_path,
        {
            "frequency_hz": np.asarray(
                float(args.frequency_mhz) * 1e6, dtype=np.float64
            ),
            "station_id": np.asarray(int(args.station_id), dtype=np.int32),
            "stokes_i_power": np.asarray(beam[:, 0], dtype=np.float32),
        },
    )
    result = {
        "schema": "oskar_aperture_array_stokes_i_beam_cache",
        "schema_version": 1,
        "beam_cache": str(result_path),
        "beam_cache_sha256": _sha256(result_path),
        "frequency_mhz": float(args.frequency_mhz),
        "time_steps": int(args.time_steps),
        "source_count": int(beam.shape[-1]),
        "station_id": int(args.station_id),
        "station_beam_duplication": True,
        "normalised_at_phase_centre": True,
        "osm": str(args.osm),
        "minimum_power": float(np.min(beam)),
        "maximum_power": float(np.max(beam)),
    }
    _atomic_json(args.out_dir / "result.json", result)
    if not args.keep_text:
        for path in args.out_dir.glob("beam_*.txt"):
            path.unlink()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
