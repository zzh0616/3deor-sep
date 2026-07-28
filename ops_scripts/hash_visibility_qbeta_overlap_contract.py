#!/usr/bin/env python3
"""Hash frequency-resolved sky and visibility inputs for overlap audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VISIBILITY_KEYS = (
    "sample_row_indices",
    "sample_uvw_m",
    "sample_time_s",
    "sample_split",
    "sample_fg",
    "sample_eor",
    "sample_antenna1",
    "sample_antenna2",
)

SKY_GEOMETRY_KEYS = (
    "l_cosine",
    "m_cosine",
    "n_minus_one",
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--frequency-first-mhz", type=float, required=True)
    parser.add_argument("--frequency-last-mhz", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _update_array_hash(
    digest: Any,
    *,
    name: str,
    value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))


def _array_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        _update_array_hash(digest, name=name, value=value)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        frequencies_mhz = np.asarray(
            archive["frequencies_mhz"], dtype=np.float64
        )
        selected = np.flatnonzero(
            (frequencies_mhz >= float(args.frequency_first_mhz) - 1e-9)
            & (frequencies_mhz <= float(args.frequency_last_mhz) + 1e-9)
        )
        if selected.size == 0:
            raise ValueError("The requested frequency interval is empty")
        geometry_digest = _array_digest(
            {
                name: np.asarray(archive[name])
                for name in SKY_GEOMETRY_KEYS
            }
        )
        sky_rows = {
            f"{frequencies_mhz[index]:.2f}": _array_digest(
                {
                    "frequency_mhz": frequencies_mhz[index],
                    "eor_jy": np.asarray(archive["eor_jy"][index]),
                    "k2jy_per_pixel": np.asarray(
                        archive["k2jy_per_pixel"][index]
                    ),
                }
            )
            for index in selected
        }

    visibility_rows: dict[str, str] = {}
    for frequency_label in sky_rows:
        shard_path = (
            args.bank_dir / "shards" / f"freq_{frequency_label}.npz"
        )
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with np.load(shard_path, allow_pickle=False) as archive:
            missing = sorted(set(VISIBILITY_KEYS) - set(archive.files))
            if missing:
                raise ValueError(
                    f"{shard_path} lacks: {', '.join(missing)}"
                )
            visibility_rows[frequency_label] = _array_digest(
                {
                    "frequency_hz": np.asarray(archive["frequency_hz"]),
                    **{
                        name: np.asarray(archive[name])
                        for name in VISIBILITY_KEYS
                    },
                }
            )

    _atomic_json(
        args.out,
        {
            "schema": "visibility_qbeta_overlap_contract_hashes",
            "schema_version": 1,
            "sky_cache": str(args.sky_cache),
            "bank_dir": str(args.bank_dir),
            "frequency_count": int(len(sky_rows)),
            "frequency_first_mhz": float(next(iter(sky_rows))),
            "frequency_last_mhz": float(next(reversed(sky_rows))),
            "sky_geometry_sha256": geometry_digest,
            "sky_frequency_sha256": sky_rows,
            "visibility_frequency_sha256": visibility_rows,
            "visibility_keys": VISIBILITY_KEYS,
        },
    )


if __name__ == "__main__":
    main()
