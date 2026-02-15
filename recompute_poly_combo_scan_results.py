#!/usr/bin/env python3
"""
Recompute poly_combo_scan aggregated CSV/MD artifacts from an existing run directory.

This is useful when a scan is interrupted/crashes after producing per-candidate outputs,
but before writing:
  - poly_combo_scan_results.csv
  - poly_combo_scan_rank.csv
  - poly_combo_scan_summary.md

The recomputation evaluates *outputs* (FITS + powerspec artifacts), not training loss values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from astropy.io import fits

from dataset_registry import DatasetSpec
from run_poly_combo_scan import (
    CandidateSpec,
    _candidate_summary,
    _extract_cut_indices,
    _load_cube_cut,
    _parse_csv_tokens,
    _run_job_result_only,
    _write_csv,
    _write_markdown,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute poly_combo_scan aggregated results from outputs.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory containing manifest.json and outputs.")
    p.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset filter.")
    p.add_argument("--candidate-names", type=str, default="", help="Optional comma-separated candidate filter.")
    p.add_argument(
        "--exclude-from-ranking",
        type=str,
        default="",
        help="Override the exclude list for ranking (comma-separated). Default: use manifest baseline value.",
    )
    return p.parse_args()


def _load_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _datasets_from_manifest(raw: Sequence[object]) -> List[DatasetSpec]:
    out: List[DatasetSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            DatasetSpec(
                name=str(item["name"]),
                input_cube=Path(str(item["input_cube"])),
                fg_true_cube=Path(str(item["fg_true_cube"])),
                eor_true_cube=Path(str(item["eor_true_cube"])),
                freq_start_mhz=float(item["freq_start_mhz"]),
            )
        )
    return out


def _candidates_from_manifest(raw: Sequence[object]) -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        extra = item.get("extra_loss_terms") or []
        out.append(
            CandidateSpec(
                name=str(item["name"]),
                extra_loss_terms=tuple(str(x) for x in extra),
                optim_overrides=dict(item.get("optim_overrides") or {}),
                weight_overrides=dict(item.get("weight_overrides") or {}),
                prior_overrides=dict(item.get("prior_overrides") or {}),
            )
        )
    return out


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    datasets = _datasets_from_manifest(manifest.get("datasets", []))
    if not datasets:
        raise ValueError("No datasets found in manifest.json")
    candidates = _candidates_from_manifest(manifest.get("candidates", []))
    if not candidates:
        raise ValueError("No candidates found in manifest.json")

    if args.datasets.strip():
        allow = set(_parse_csv_tokens(args.datasets))
        datasets = [d for d in datasets if d.name in allow]
        if not datasets:
            raise ValueError("No datasets selected after --datasets filter.")

    if args.candidate_names.strip():
        allow = set(_parse_csv_tokens(args.candidate_names))
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

    raw_args = manifest.get("args", {})
    if not isinstance(raw_args, dict):
        raw_args = {}
    cut_size_frac = float(raw_args.get("cut_size_frac", 0.30))

    # Prepare dataset caches: cut indices + true cubes.
    ds_cache: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        with fits.open(ds.input_cube, memmap=True) as hdul:
            in_shape = tuple(int(v) for v in hdul[0].data.shape)
        cut = _extract_cut_indices(in_shape, cut_size_frac)
        ds_cache[ds.name] = {
            "true_eor": _load_cube_cut(ds.eor_true_cube, cut=cut),
            "true_fg": _load_cube_cut(ds.fg_true_cube, cut=cut),
        }

    rows: List[Dict[str, object]] = []
    for cand in candidates:
        for ds in datasets:
            out_dir = run_dir / cand.name / ds.name
            eor_out = out_dir / "eor_est.fits"
            fg_out = out_dir / "fg_est.fits"
            return_code = 0 if (eor_out.exists() and fg_out.exists()) else 1
            cache = ds_cache[ds.name]
            true_eor = cache["true_eor"]
            true_fg = cache["true_fg"]
            row = _run_job_result_only(
                dataset=ds,
                candidate=cand,
                run_dir=out_dir,
                true_eor=true_eor,  # type: ignore[arg-type]
                true_fg=true_fg,  # type: ignore[arg-type]
                return_code=int(return_code),
                runtime=float("nan"),
            )
            rows.append(row)

    detail_csv = run_dir / "poly_combo_scan_results.csv"
    _write_csv(detail_csv, rows)

    exclude = args.exclude_from_ranking.strip()
    if not exclude:
        baseline = manifest.get("baseline_fixed", {})
        if isinstance(baseline, dict):
            exclude = str(baseline.get("exclude_from_ranking", ""))
    ranked = _candidate_summary(rows, exclude_datasets=_parse_csv_tokens(exclude))
    rank_csv = run_dir / "poly_combo_scan_rank.csv"
    _write_csv(rank_csv, ranked)

    meta = manifest.get("baseline_fixed", {})
    if not isinstance(meta, dict):
        meta = {}
    _write_markdown(run_dir / "poly_combo_scan_summary.md", ranked, meta)  # type: ignore[arg-type]

    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

