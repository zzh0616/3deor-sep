#!/usr/bin/env python3
"""Simulate, grid, and consolidate a noiseless CHIPS-style visibility bank.

Each frequency is processed independently.  The temporary foreground and EoR
Measurement Sets are immediately reduced to split uv grids and a deterministic
row sample, then may be deleted.  This keeps the 32-frequency pilot small while
retaining component labels strictly for post-estimator diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

C_M_S = 299792458.0


def _parse_frequencies(spec: str) -> list[float]:
    values = [float(piece.strip()) for piece in str(spec).split(",") if piece.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("At least one finite frequency is required")
    if len(values) > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError("Frequencies must be strictly increasing")
    return values


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "combine"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frequency-mhz", type=float)
    parser.add_argument("--frequencies-mhz")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--oskar")
    parser.add_argument("--telescope-dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--min-uv-lambda", type=float, default=30.0)
    parser.add_argument("--max-uv-lambda", type=float, default=2500.0)
    parser.add_argument("--reference-frequency-mhz", type=float, default=119.45)
    parser.add_argument("--chunk-rows", type=int, default=262144)
    parser.add_argument("--sample-kperp-bins", type=int, default=16)
    parser.add_argument("--sample-rows-per-bin", type=int, default=2048)
    parser.add_argument("--delete-ms", action="store_true")
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _write_oskar_config(
    path: Path,
    *,
    osm: Path,
    ms: Path,
    frequency_mhz: float,
    telescope_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""[General]
app=oskar_sim_interferometer
version=2.12.2

[simulator]
max_sources_per_chunk=131072
keep_log_file=true
write_status_to_log_file=true
double_precision=true

[sky]
oskar_sky_model/file={osm}
advanced/apply_horizon_clip=false

[observation]
phase_centre_dec_deg=-27.0
start_frequency_hz={float(frequency_mhz) * 1e6:.1f}
start_time_utc=2030-01-01T06:30:00.0
length=320.0
num_time_steps=32

[telescope]
input_directory={telescope_dir}
station_type=Isotropic beam
allow_station_beam_duplication=true

[interferometer]
channel_bandwidth_hz=100000.0
time_average_sec=10.0
max_time_samples_per_block=4
noise/enable=false
ms_filename={ms}
""",
        encoding="utf-8",
    )


def _run_oskar(
    executable: str,
    config: Path,
    log: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {executable} {config}\n")
        handle.flush()
        subprocess.run(
            [str(executable), str(config)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _corr_indices(ms: Path) -> tuple[int, int]:
    from casacore.tables import table

    with table(str(ms / "POLARIZATION"), readonly=True, ack=False) as pol:
        corr_types = np.asarray(pol.getcol("CORR_TYPE"))[0].reshape(-1)
    xx = np.flatnonzero(corr_types == 9)
    yy = np.flatnonzero(corr_types == 12)
    if xx.size != 1 or yy.size != 1:
        raise ValueError(f"Expected exactly XX/YY correlations, got {corr_types.tolist()}")
    return int(xx[0]), int(yy[0])


def _channel_frequency_hz(ms: Path) -> float:
    from casacore.tables import table

    with table(str(ms / "SPECTRAL_WINDOW"), readonly=True, ack=False) as spectral:
        frequencies = np.asarray(spectral.getcol("CHAN_FREQ"), dtype=np.float64)
    if frequencies.size != 1:
        raise ValueError("Each pilot Measurement Set must contain exactly one channel")
    return float(frequencies.reshape(-1)[0])


def _stokes_i(
    data: np.ndarray,
    flags: np.ndarray,
    *,
    xx_index: int,
    yy_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data)
    bad = np.asarray(flags, dtype=bool)
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError(f"Unsupported DATA shape {values.shape}")
    stokes = 0.5 * (
        values[:, 0, int(xx_index)] + values[:, 0, int(yy_index)]
    )
    valid = ~(
        bad[:, 0, int(xx_index)] | bad[:, 0, int(yy_index)]
    )
    valid &= np.isfinite(stokes.real) & np.isfinite(stokes.imag)
    return np.asarray(stokes, dtype=np.complex128), valid


def _deposit_bilinear(
    sums: np.ndarray,
    weights: np.ndarray,
    *,
    u_lambda: np.ndarray,
    v_lambda: np.ndarray,
    visibilities: np.ndarray,
    split: np.ndarray,
    max_uv_lambda: float,
) -> None:
    grid_size = int(sums.shape[-1])
    cell = 2.0 * float(max_uv_lambda) / float(grid_size)
    u = np.concatenate((u_lambda, -u_lambda))
    v = np.concatenate((v_lambda, -v_lambda))
    values = np.concatenate((visibilities, np.conjugate(visibilities)))
    halves = np.concatenate((split, split)).astype(np.int64, copy=False)
    x = (u + float(max_uv_lambda)) / cell - 0.5
    y = (v + float(max_uv_lambda)) / cell - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            xi = x0 + dx
            yi = y0 + dy
            kernel = wx * wy
            valid = (
                (xi >= 0)
                & (xi < grid_size)
                & (yi >= 0)
                & (yi < grid_size)
                & (kernel > 0.0)
            )
            for half in (0, 1):
                selected = valid & (halves == half)
                if not np.any(selected):
                    continue
                flat = yi[selected] * grid_size + xi[selected]
                current_weight = kernel[selected]
                weight_sum = np.bincount(
                    flat,
                    weights=current_weight,
                    minlength=grid_size * grid_size,
                )
                real_sum = np.bincount(
                    flat,
                    weights=current_weight * values[selected].real,
                    minlength=grid_size * grid_size,
                )
                imag_sum = np.bincount(
                    flat,
                    weights=current_weight * values[selected].imag,
                    minlength=grid_size * grid_size,
                )
                weights[half].reshape(-1)[:] += weight_sum
                sums[half].reshape(-1)[:] += real_sum + 1j * imag_sum


def _select_row_sample(
    *,
    row_indices: np.ndarray,
    uvw_m: np.ndarray,
    times: np.ndarray,
    fg: np.ndarray,
    eor: np.ndarray,
    reference_frequency_mhz: float,
    min_uv_lambda: float,
    max_uv_lambda: float,
    bins: int,
    rows_per_bin: int,
) -> dict[str, np.ndarray]:
    radius = (
        np.hypot(uvw_m[:, 0], uvw_m[:, 1])
        * float(reference_frequency_mhz)
        * 1e6
        / C_M_S
    )
    edges = np.linspace(float(min_uv_lambda), float(max_uv_lambda), int(bins) + 1)
    groups = np.searchsorted(edges, radius, side="right") - 1
    groups[np.isclose(radius, edges[-1], rtol=1e-12, atol=1e-12)] = int(bins) - 1
    chosen: list[np.ndarray] = []
    for group in range(int(bins)):
        candidates = np.flatnonzero(groups == group)
        if candidates.size == 0:
            continue
        retain = min(int(rows_per_bin), int(candidates.size))
        positions = np.linspace(0, candidates.size - 1, retain).round().astype(np.int64)
        chosen.append(candidates[positions])
    if not chosen:
        raise ValueError("No deterministic visibility rows lie inside the UV support")
    selected = np.unique(np.concatenate(chosen))
    unique_times = np.unique(times)
    split = np.searchsorted(unique_times, times[selected]) % 2
    return {
        "sample_row_indices": np.asarray(row_indices[selected], dtype=np.int64),
        "sample_uvw_m": np.asarray(uvw_m[selected], dtype=np.float64),
        "sample_time_s": np.asarray(times[selected], dtype=np.float64),
        "sample_split": np.asarray(split, dtype=np.int8),
        "sample_fg": np.asarray(fg[selected], dtype=np.complex64),
        "sample_eor": np.asarray(eor[selected], dtype=np.complex64),
    }


def grid_ms_pair(
    fg_ms: Path,
    eor_ms: Path,
    *,
    grid_size: int,
    min_uv_lambda: float,
    max_uv_lambda: float,
    reference_frequency_mhz: float,
    chunk_rows: int,
    sample_kperp_bins: int,
    sample_rows_per_bin: int,
) -> dict[str, np.ndarray]:
    """Grid a foreground/EoR MS pair and retain a deterministic row sample."""
    from casacore.tables import table

    fg_frequency = _channel_frequency_hz(fg_ms)
    eor_frequency = _channel_frequency_hz(eor_ms)
    if not math.isclose(fg_frequency, eor_frequency, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError("Foreground and EoR MS frequencies differ")
    fg_xx, fg_yy = _corr_indices(fg_ms)
    eor_xx, eor_yy = _corr_indices(eor_ms)
    if (fg_xx, fg_yy) != (eor_xx, eor_yy):
        raise ValueError("Foreground and EoR correlation layouts differ")

    shape = (2, int(grid_size), int(grid_size))
    fg_sum = np.zeros(shape, dtype=np.complex128)
    eor_sum = np.zeros(shape, dtype=np.complex128)
    grid_weight = np.zeros(shape, dtype=np.float64)
    candidate_stride: int
    candidates: dict[str, list[np.ndarray]] = {
        "row_indices": [],
        "uvw_m": [],
        "times": [],
        "fg": [],
        "eor": [],
    }
    with table(str(fg_ms), readonly=True, ack=False) as fg_table, table(
        str(eor_ms), readonly=True, ack=False
    ) as eor_table:
        if int(fg_table.nrows()) != int(eor_table.nrows()):
            raise ValueError("Foreground and EoR row counts differ")
        rows = int(fg_table.nrows())
        candidate_target = max(
            1, 2 * int(sample_kperp_bins) * int(sample_rows_per_bin)
        )
        candidate_stride = max(1, rows // candidate_target)
        all_times = np.asarray(fg_table.getcol("TIME"), dtype=np.float64)
        unique_times = np.unique(all_times)
        del all_times

        for first in range(0, rows, int(chunk_rows)):
            count = min(int(chunk_rows), rows - first)
            uvw_fg = np.asarray(
                fg_table.getcol("UVW", first, count), dtype=np.float64
            )
            uvw_eor = np.asarray(
                eor_table.getcol("UVW", first, count), dtype=np.float64
            )
            if not np.allclose(uvw_fg, uvw_eor, rtol=0.0, atol=1e-9):
                raise ValueError("Foreground and EoR UVW rows differ")
            time_fg = np.asarray(
                fg_table.getcol("TIME", first, count), dtype=np.float64
            )
            time_eor = np.asarray(
                eor_table.getcol("TIME", first, count), dtype=np.float64
            )
            if not np.array_equal(time_fg, time_eor):
                raise ValueError("Foreground and EoR time rows differ")
            flag_rows = np.asarray(
                fg_table.getcol("FLAG_ROW", first, count), dtype=bool
            ) | np.asarray(eor_table.getcol("FLAG_ROW", first, count), dtype=bool)
            fg, fg_valid = _stokes_i(
                fg_table.getcol("DATA", first, count),
                fg_table.getcol("FLAG", first, count),
                xx_index=fg_xx,
                yy_index=fg_yy,
            )
            eor, eor_valid = _stokes_i(
                eor_table.getcol("DATA", first, count),
                eor_table.getcol("FLAG", first, count),
                xx_index=eor_xx,
                yy_index=eor_yy,
            )
            u_lambda = uvw_fg[:, 0] * fg_frequency / C_M_S
            v_lambda = uvw_fg[:, 1] * fg_frequency / C_M_S
            radius = np.hypot(u_lambda, v_lambda)
            valid = (
                fg_valid
                & eor_valid
                & ~flag_rows
                & (radius >= float(min_uv_lambda))
                & (radius <= float(max_uv_lambda))
            )
            split = np.searchsorted(unique_times, time_fg) % 2
            weight_before = np.array(grid_weight, copy=True)
            _deposit_bilinear(
                fg_sum,
                grid_weight,
                u_lambda=u_lambda[valid],
                v_lambda=v_lambda[valid],
                visibilities=fg[valid],
                split=split[valid],
                max_uv_lambda=max_uv_lambda,
            )
            # Geometry weights are shared; avoid depositing them twice.
            eor_weight = np.zeros_like(grid_weight)
            _deposit_bilinear(
                eor_sum,
                eor_weight,
                u_lambda=u_lambda[valid],
                v_lambda=v_lambda[valid],
                visibilities=eor[valid],
                split=split[valid],
                max_uv_lambda=max_uv_lambda,
            )
            if not np.allclose(
                eor_weight,
                grid_weight - weight_before,
                rtol=1e-12,
                atol=1e-8,
            ):
                raise ValueError("Component gridding weights differ")

            global_rows = first + np.arange(count, dtype=np.int64)
            reference_radius = (
                np.hypot(uvw_fg[:, 0], uvw_fg[:, 1])
                * float(reference_frequency_mhz)
                * 1e6
                / C_M_S
            )
            candidate = (
                fg_valid
                & eor_valid
                & ~flag_rows
                & (reference_radius >= float(min_uv_lambda))
                & (reference_radius <= float(max_uv_lambda))
                & ((global_rows % candidate_stride) == 0)
            )
            if np.any(candidate):
                candidates["row_indices"].append(global_rows[candidate])
                candidates["uvw_m"].append(uvw_fg[candidate])
                candidates["times"].append(time_fg[candidate])
                candidates["fg"].append(fg[candidate])
                candidates["eor"].append(eor[candidate])

    fg_grid = np.zeros_like(fg_sum)
    eor_grid = np.zeros_like(eor_sum)
    occupied = grid_weight > 0.0
    fg_grid[occupied] = fg_sum[occupied] / grid_weight[occupied]
    eor_grid[occupied] = eor_sum[occupied] / grid_weight[occupied]
    sampled = _select_row_sample(
        row_indices=np.concatenate(candidates["row_indices"]),
        uvw_m=np.concatenate(candidates["uvw_m"]),
        times=np.concatenate(candidates["times"]),
        fg=np.concatenate(candidates["fg"]),
        eor=np.concatenate(candidates["eor"]),
        reference_frequency_mhz=reference_frequency_mhz,
        min_uv_lambda=min_uv_lambda,
        max_uv_lambda=max_uv_lambda,
        bins=sample_kperp_bins,
        rows_per_bin=sample_rows_per_bin,
    )
    cell = 2.0 * float(max_uv_lambda) / float(grid_size)
    centers = -float(max_uv_lambda) + (np.arange(grid_size) + 0.5) * cell
    return {
        "frequency_hz": np.asarray(fg_frequency, dtype=np.float64),
        "u_centers_lambda": np.asarray(centers, dtype=np.float64),
        "v_centers_lambda": np.asarray(centers, dtype=np.float64),
        "grid_weight": np.asarray(grid_weight, dtype=np.float32),
        "fg_grid": np.asarray(fg_grid, dtype=np.complex64),
        "eor_grid": np.asarray(eor_grid, dtype=np.complex64),
        **sampled,
    }


def _simulate_shard(args: argparse.Namespace) -> None:
    required = {
        "--frequency-mhz": args.frequency_mhz,
        "--source-root": args.source_root,
        "--oskar": args.oskar,
        "--telescope-dir": args.telescope_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Shard mode requires {', '.join(missing)}")
    frequency = float(args.frequency_mhz)
    tag = f"{frequency:.2f}"
    shard_dir = args.out_dir / "shards"
    shard_path = shard_dir / f"freq_{tag}.npz"
    if shard_path.is_file():
        print(f"existing shard: {shard_path}", flush=True)
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "tmp" / f"freq_{tag}"
    configs = args.out_dir / "configs"
    logs = args.out_dir / "logs"
    work.mkdir(parents=True, exist_ok=True)
    ms_paths: dict[str, Path] = {}
    started = time.monotonic()
    for label in ("fg", "eor"):
        osm = Path(args.source_root) / "osm" / f"{label}_{tag}.osm"
        if not osm.is_file():
            raise FileNotFoundError(osm)
        ms = work / f"{label}_{tag}.ms"
        config = configs / f"sim_{label}_{tag}.ini"
        ms_paths[label] = ms
        if not ms.is_dir():
            _write_oskar_config(
                config,
                osm=osm,
                ms=ms,
                frequency_mhz=frequency,
                telescope_dir=Path(args.telescope_dir),
            )
            _run_oskar(str(args.oskar), config, logs / f"oskar_{label}_{tag}.log")

    payload = grid_ms_pair(
        ms_paths["fg"],
        ms_paths["eor"],
        grid_size=int(args.grid_size),
        min_uv_lambda=float(args.min_uv_lambda),
        max_uv_lambda=float(args.max_uv_lambda),
        reference_frequency_mhz=float(args.reference_frequency_mhz),
        chunk_rows=int(args.chunk_rows),
        sample_kperp_bins=int(args.sample_kperp_bins),
        sample_rows_per_bin=int(args.sample_rows_per_bin),
    )
    payload["elapsed_seconds"] = np.asarray(time.monotonic() - started, dtype=np.float64)
    _atomic_npz(shard_path, payload)
    if args.delete_ms:
        for ms in ms_paths.values():
            shutil.rmtree(ms)
    print(f"wrote shard: {shard_path}", flush=True)


def _combine(args: argparse.Namespace) -> None:
    if not args.frequencies_mhz:
        raise ValueError("Combine mode requires --frequencies-mhz")
    frequencies = _parse_frequencies(args.frequencies_mhz)
    shards: list[dict[str, np.ndarray]] = []
    shard_paths: list[Path] = []
    for frequency in frequencies:
        path = args.out_dir / "shards" / f"freq_{frequency:.2f}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as loaded:
            shards.append({name: np.asarray(loaded[name]) for name in loaded.files})
        shard_paths.append(path)
    reference = shards[0]
    for shard in shards[1:]:
        for name in (
            "u_centers_lambda",
            "v_centers_lambda",
            "sample_row_indices",
            "sample_uvw_m",
            "sample_time_s",
            "sample_split",
        ):
            if not np.array_equal(shard[name], reference[name]):
                raise ValueError(f"Visibility shards disagree in {name}")
    bank_payload = {
        "frequencies_hz": np.asarray(
            [float(shard["frequency_hz"].item()) for shard in shards],
            dtype=np.float64,
        ),
        "u_centers_lambda": reference["u_centers_lambda"],
        "v_centers_lambda": reference["v_centers_lambda"],
        "grid_weight": np.stack(
            [shard["grid_weight"] for shard in shards], axis=1
        ),
        "fg_grid": np.stack([shard["fg_grid"] for shard in shards], axis=1),
        "eor_grid": np.stack([shard["eor_grid"] for shard in shards], axis=1),
        "sample_row_indices": reference["sample_row_indices"],
        "sample_uvw_m": reference["sample_uvw_m"],
        "sample_time_s": reference["sample_time_s"],
        "sample_split": reference["sample_split"],
        "sample_fg": np.stack([shard["sample_fg"] for shard in shards], axis=0),
        "sample_eor": np.stack([shard["sample_eor"] for shard in shards], axis=0),
    }
    bank_path = args.out_dir / "visibility_bank.npz"
    _atomic_npz(bank_path, bank_payload)
    manifest = {
        "schema": "chips_visibility_bank",
        "schema_version": 1,
        "bank_path": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "frequencies_mhz": frequencies,
        "grid_size": int(reference["fg_grid"].shape[-1]),
        "split_count": int(reference["fg_grid"].shape[0]),
        "sample_row_count": int(reference["sample_row_indices"].size),
        "source_shards": [
            {"path": str(path), "sha256": _sha256(path)} for path in shard_paths
        ],
        "gridding": {
            "kernel": "bilinear",
            "conjugate_completion": True,
            "split": "alternating_oskar_time",
            "min_uv_lambda": float(args.min_uv_lambda),
            "max_uv_lambda": float(args.max_uv_lambda),
            "reference_frequency_mhz": float(args.reference_frequency_mhz),
        },
        "instrument": {
            "simulator": "OSKAR 2.12.2",
            "station_type": "isotropic",
            "noise": False,
            "time_steps": 32,
            "observation_length_s": 320.0,
            "channel_bandwidth_hz": 100000.0,
            "time_average_s": 10.0,
        },
    }
    _atomic_json(args.out_dir / "manifest.json", manifest)
    print(f"wrote bank: {bank_path}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "shard":
        _simulate_shard(args)
    else:
        _combine(args)


if __name__ == "__main__":
    main()
