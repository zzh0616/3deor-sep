#!/usr/bin/env python3
"""Freeze the deterministic visibility rows used by a Q_beta partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibrate_visibility_qbeta_noiseless import (  # noqa: E402
    _load_bank,
    _select_qbeta_rows,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows-per-kperp-bin", type=int, default=12)
    parser.add_argument(
        "--row-scope",
        choices=("all", "reporting_kperp"),
        default="all",
    )
    parser.add_argument("--maximum-kperp-index-exclusive", type=int)
    parser.add_argument("--row-seed", type=int, default=20260725)
    parser.add_argument("--row-partition-index", type=int, default=0)
    parser.add_argument("--row-partition-count", type=int, default=1)
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


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    bank, manifest = _load_bank(args.bank_dir)
    (
        all_row_kperp,
        kperp_edges,
        selected_bins,
        selected_rows,
    ) = _select_qbeta_rows(
        bank=bank,
        resolved=resolved,
        config=config,
        row_scope=str(args.row_scope),
        maximum_kperp_index_exclusive=(
            None
            if args.maximum_kperp_index_exclusive is None
            else int(args.maximum_kperp_index_exclusive)
        ),
        rows_per_kperp_bin=int(args.rows_per_kperp_bin),
        row_seed=int(args.row_seed),
        row_partition_index=int(args.row_partition_index),
        row_partition_count=int(args.row_partition_count),
    )
    np.savez_compressed(
        args.out_dir / "result.npz",
        selected_bank_rows=selected_rows,
        selected_row_kperp_mpc_inv=all_row_kperp[selected_rows],
        selected_row_kperp_indices=selected_bins,
        kperp_edges_mpc_inv=kperp_edges,
    )
    result = {
        "schema": "visibility_qbeta_row_selection",
        "schema_version": 1,
        "analysis_contract_sha256": (
            resolved.contract.analysis_contract_sha256
        ),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "bank_dir": str(args.bank_dir),
        "visibility_bank_sha256": str(manifest["bank_sha256"]),
        "selected_row_count": int(selected_rows.size),
        "selected_kperp_bin_count": int(selected_bins.size),
        "settings": {
            "rows_per_kperp_bin": int(args.rows_per_kperp_bin),
            "row_scope": str(args.row_scope),
            "maximum_kperp_index_exclusive": (
                None
                if args.maximum_kperp_index_exclusive is None
                else int(args.maximum_kperp_index_exclusive)
            ),
            "row_seed": int(args.row_seed),
            "row_partition_index": int(args.row_partition_index),
            "row_partition_count": int(args.row_partition_count),
        },
    }
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
