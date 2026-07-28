#!/usr/bin/env python3
"""Build an exact OSKAR station-pair aperture-beam coherency cache."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import subprocess
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

from evaluate_oskar_aperture_beam_closure import _row_time_indices  # noqa: E402
from evaluate_oskar_gaussian_beam_closure import _selected_rows  # noqa: E402
from visibility_primary_beam import (  # noqa: E402
    direction_cosine_geometry_sha256,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-shard", type=Path, required=True)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--oskar-config", type=Path, required=True)
    parser.add_argument("--oskar-prefix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selected-row-result", type=Path)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument("--source-chunk", type=int, default=32768)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--compiler-library-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Library directory placed before OSKAR libraries, for example "
            "the Conda runtime used to build OSKAR"
        ),
    )
    parser.add_argument("--helper-binary", type=Path)
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


def _oskar_operator_settings(path: Path) -> dict[str, float]:
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path):
        raise ValueError(f"Cannot read OSKAR config: {path}")
    try:
        return {
            "frequency_hz": config.getfloat(
                "observation", "start_frequency_hz"
            ),
            "phase_centre_dec_deg": config.getfloat(
                "observation", "phase_centre_dec_deg"
            ),
            "channel_bandwidth_hz": config.getfloat(
                "interferometer", "channel_bandwidth_hz"
            ),
            "time_average_sec": config.getfloat(
                "interferometer", "time_average_sec"
            ),
        }
    except (configparser.Error, ValueError) as error:
        raise ValueError(
            f"OSKAR config lacks an operator setting: {path}"
        ) from error


def _compile_helper(
    *,
    source: Path,
    output: Path,
    prefix: Path,
    cxx: str,
    compiler_library_dirs: list[Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cxx),
        "-std=c++17",
        "-O3",
        "-fopenmp",
        f"-I{prefix / 'include' / 'oskar'}",
        str(source),
    ]
    for directory in compiler_library_dirs:
        command.extend(
            (
                f"-L{directory}",
                f"-Wl,-rpath,{directory}",
                f"-Wl,-rpath-link,{directory}",
            )
        )
    command.extend(
        [
            f"-L{prefix / 'lib'}",
            f"-Wl,-rpath,{prefix / 'lib'}",
            f"-Wl,-rpath-link,{prefix / 'lib'}",
        "-loskar_apps",
        "-loskar_settings",
        "-loskar",
        "-o",
        str(output),
        ]
    )
    subprocess.run(command, check=True)
    _atomic_json(
        output.with_name(output.name + ".build.json"),
        {
            "schema": "oskar_aperture_row_beam_helper_build",
            "schema_version": 1,
            "source_sha256": _sha256(source),
            "oskar_prefix": str(prefix),
            "cxx": str(cxx),
            "compiler_library_dirs": [
                str(path) for path in compiler_library_dirs
            ],
            "binary_sha256": _sha256(output),
        },
    )


def _helper_matches_build_contract(
    *,
    helper: Path,
    source: Path,
    prefix: Path,
    cxx: str,
    compiler_library_dirs: list[Path],
) -> bool:
    manifest_path = helper.with_name(helper.name + ".build.json")
    if not helper.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema")
        == "oskar_aperture_row_beam_helper_build"
        and str(manifest.get("source_sha256", "")) == _sha256(source)
        and str(manifest.get("oskar_prefix", "")) == str(prefix)
        and str(manifest.get("cxx", "")) == str(cxx)
        and list(manifest.get("compiler_library_dirs", []))
        == [str(path) for path in compiler_library_dirs]
        and str(manifest.get("binary_sha256", "")) == _sha256(helper)
    )


def _validate_finite_file(
    path: Path,
    *,
    shape: tuple[int, int],
    source_chunk: int,
) -> tuple[float, float]:
    values = np.memmap(path, mode="r", dtype=np.complex64, shape=shape)
    minimum = np.inf
    maximum = 0.0
    for first in range(0, shape[1], int(source_chunk)):
        block = np.asarray(values[:, first : first + int(source_chunk)])
        if not np.all(np.isfinite(block)):
            raise ValueError("Row-beam cache contains non-finite values")
        amplitude = np.abs(block)
        minimum = min(minimum, float(np.min(amplitude)))
        maximum = max(maximum, float(np.max(amplitude)))
    return minimum, maximum


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.source_chunk) < 1:
        raise ValueError("--source-chunk must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank_shard, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        sky = {name: np.asarray(archive[name]) for name in archive.files}
    if str(bank.get("station_type", np.asarray("")).item()) != "aperture_array":
        raise ValueError("Bank shard is not an aperture-array simulation")
    for name in ("sample_antenna1", "sample_antenna2"):
        if name not in bank:
            raise ValueError(f"Bank shard lacks {name}")
    rows = _selected_rows(
        bank_row_count=int(bank["sample_uvw_m"].shape[0]),
        selected_result=args.selected_row_result,
        maximum_rows=args.maximum_rows,
    )
    unique_times = np.unique(np.asarray(bank["sample_time_s"], dtype=np.float64))
    time_indices = _row_time_indices(
        bank["sample_time_s"], rows, int(unique_times.size)
    )
    source_count = int(np.asarray(sky["l_cosine"]).size)
    n_cosine = np.asarray(sky["n_minus_one"], dtype=np.float64) + 1.0
    directions = np.stack(
        (
            np.asarray(sky["l_cosine"], dtype=np.float64),
            np.asarray(sky["m_cosine"], dtype=np.float64),
            n_cosine,
        )
    )
    if directions.shape != (3, source_count) or not np.all(
        np.isfinite(directions)
    ):
        raise ValueError("Invalid sky direction geometry")
    row_geometry = np.column_stack(
        (
            time_indices,
            np.asarray(bank["sample_antenna1"], dtype=np.int32)[rows],
            np.asarray(bank["sample_antenna2"], dtype=np.int32)[rows],
        )
    ).astype(np.int32, copy=False)

    directions_path = args.out_dir / "directions.float64.bin"
    rows_path = args.out_dir / "rows.int32.bin"
    data_path = args.out_dir / "coherency.complex64.bin"
    directions.tofile(directions_path)
    row_geometry.tofile(rows_path)
    helper = (
        args.helper_binary
        if args.helper_binary is not None
        else args.out_dir / "evaluate_oskar_aperture_row_beam_factors"
    )
    helper_source = SCRIPT_DIR / "evaluate_oskar_aperture_row_beam_factors.cc"
    if not _helper_matches_build_contract(
        helper=helper,
        source=helper_source,
        prefix=args.oskar_prefix,
        cxx=str(args.cxx),
        compiler_library_dirs=list(args.compiler_library_dir),
    ):
        _compile_helper(
            source=helper_source,
            output=helper,
            prefix=args.oskar_prefix,
            cxx=str(args.cxx),
            compiler_library_dirs=list(args.compiler_library_dir),
        )

    command = [
        str(helper),
        "--config",
        str(args.oskar_config),
        "--directions",
        str(directions_path),
        "--num-sources",
        str(source_count),
        "--rows",
        str(rows_path),
        "--num-rows",
        str(rows.size),
        "--output",
        str(data_path),
        "--source-chunk",
        str(int(args.source_chunk)),
    ]
    started = time.monotonic()
    subprocess.run(command, check=True)
    elapsed = float(time.monotonic() - started)
    shape = (int(rows.size), source_count)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(
        np.complex64
    ).itemsize
    if data_path.stat().st_size != expected_bytes:
        raise ValueError("Row-beam helper wrote an unexpected byte count")
    minimum, maximum = _validate_finite_file(
        data_path,
        shape=shape,
        source_chunk=int(args.source_chunk),
    )
    oskar_operator_settings = _oskar_operator_settings(args.oskar_config)
    if not np.isclose(
        oskar_operator_settings["frequency_hz"],
        float(bank["frequency_hz"].item()),
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError("OSKAR config frequency differs from the bank shard")
    np.savez_compressed(
        args.out_dir / "geometry.npz",
        selected_bank_rows=rows,
        row_time_indices=time_indices,
        row_antenna1=row_geometry[:, 1],
        row_antenna2=row_geometry[:, 2],
        unique_sample_times_s=unique_times,
    )
    metadata = {
        "schema": "oskar_aperture_row_direction_coherency",
        "schema_version": 2,
        "data_file": data_path.name,
        "dtype": "complex64",
        "shape": list(shape),
        "axis_order": ["selected_visibility_row", "sky_direction"],
        "definition": "0.5 * Tr(E_p E_q^H)",
        "jones_scope": (
            "OSKAR E-Jones; the common duplicated parallactic rotation "
            "cancels from the unpolarised trace"
        ),
        "scope_guards": {
            "allow_station_beam_duplication": True,
            "station_gains_defined": False,
            "thermal_noise_enabled": False,
        },
        "frequency_hz": float(bank["frequency_hz"].item()),
        "selected_row_count": int(rows.size),
        "source_count": source_count,
        "direction_sha256": direction_cosine_geometry_sha256(
            l_cosine=directions[0],
            m_cosine=directions[1],
            n_minus_one=directions[2] - 1.0,
        ),
        "source_chunk": int(args.source_chunk),
        "time_count": int(unique_times.size),
        "elapsed_seconds": elapsed,
        "amplitude_minimum": minimum,
        "amplitude_maximum": maximum,
        "data_sha256": _sha256(data_path),
        "bank_shard": str(args.bank_shard),
        "bank_shard_sha256": _sha256(args.bank_shard),
        "sky_cache": str(args.sky_cache),
        "sky_cache_sha256": _sha256(args.sky_cache),
        "oskar_config": str(args.oskar_config),
        "oskar_config_sha256": _sha256(args.oskar_config),
        "oskar_operator_settings": oskar_operator_settings,
        "helper_binary": str(helper),
        "helper_binary_sha256": _sha256(helper),
        "helper_source_sha256": _sha256(helper_source),
    }
    _atomic_json(args.out_dir / "metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
