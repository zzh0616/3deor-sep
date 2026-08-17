#!/usr/bin/env python3
"""Build a favorable delay-diagonal control from a calibrated Q_beta run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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


def row_sum_matched_diagonal_response(
    *,
    response: np.ndarray,
    output_band_ids: np.ndarray,
    source_band_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each response row onto its nominal source band.

    The exact row sum is retained, so this control does not incur an
    additional scalar-normalisation error for a locally flat spectrum.
    """
    matrix = np.asarray(response, dtype=np.float64)
    output_ids = np.asarray(output_band_ids, dtype=np.int64).reshape(-1)
    source_ids = np.asarray(source_band_ids, dtype=np.int64).reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape != (output_ids.size, source_ids.size)
        or np.unique(source_ids).size != source_ids.size
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Response and band identifiers are inconsistent")
    source_lookup = {int(value): index for index, value in enumerate(source_ids)}
    if any(int(value) not in source_lookup for value in output_ids):
        raise ValueError("An output band has no matching nominal source band")
    nominal_positions = np.asarray(
        [source_lookup[int(value)] for value in output_ids], dtype=np.int64
    )
    row_sum = np.sum(matrix, axis=1)
    if np.any(row_sum <= 0.0) or not np.all(np.isfinite(row_sum)):
        raise ValueError("Every response row must have finite positive gain")
    diagonal = np.zeros_like(matrix)
    diagonal[np.arange(matrix.shape[0]), nominal_positions] = row_sum
    return diagonal, nominal_positions


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.combined_npz, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"calibration_response", "output_band_ids", "source_band_ids"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("Combined archive lacks: " + ", ".join(missing))
    exact_response = np.asarray(data["calibration_response"], dtype=np.float64)
    diagonal, nominal_positions = row_sum_matched_diagonal_response(
        response=exact_response,
        output_band_ids=data["output_band_ids"],
        source_band_ids=data["source_band_ids"],
    )
    data["calibration_response"] = diagonal
    data["delay_diagonal_nominal_source_positions"] = nominal_positions
    data["delay_diagonal_exact_response_row_sum"] = np.sum(
        exact_response, axis=1
    )
    output_path = args.out_dir / "result.npz"
    _atomic_npz(output_path, data)
    result = {
        "schema": "visibility_qbeta_delay_diagonal_control",
        "schema_version": 1,
        "combined_npz": str(args.combined_npz),
        "combined_npz_sha256": _sha256(args.combined_npz),
        "output_npz": str(output_path),
        "output_npz_sha256": _sha256(output_path),
        "normalisation": "exact_response_row_sum",
        "interpretation": (
            "Each calibrated response row is assigned to its nominal single "
            "source band. The exact row sum is retained; non-local window "
            "mixing is intentionally ignored."
        ),
        "output_count": int(diagonal.shape[0]),
        "source_count": int(diagonal.shape[1]),
    }
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
