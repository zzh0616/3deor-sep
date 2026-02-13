#!/usr/bin/env python3
"""
Collect and merge extra EoR priors scan results from remote workers.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from run_eor_extra_priors_scan import _candidate_summary, _write_markdown


HOST_ALIAS = {
    "119.78.226.31": "genoa",
    "202.127.24.58": "milan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect extra EoR priors scan outputs from remote workers.")
    parser.add_argument(
        "--dispatch-manifest",
        type=Path,
        required=True,
        help="Local dispatch manifest produced by dispatch_eor_extra_priors_remote.py.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run_capture(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), text=True)


def _host_ip(host: str) -> str:
    return host.split("@")[-1].strip()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fetch_text(host: str, remote_path: str) -> str:
    return _run_capture(["ssh", "-o", "BatchMode=yes", host, f"cat {remote_path}"])


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    manifest_path = args.dispatch_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    remote_run_root = str(payload["remote_run_root"])
    exclude_from_ranking = str(payload.get("args", {}).get("exclude_from_ranking", "cube1"))
    exclude_names = [t.strip() for t in exclude_from_ranking.split(",") if t.strip()]

    if args.output_root is None:
        date_bucket = manifest_path.parent
        run_name = Path(remote_run_root).name
        output_root = date_bucket / run_name
    else:
        output_root = args.output_root.resolve()

    collected_root = output_root / "collected"
    merged_root = output_root / "merged"
    _ensure_dir(collected_root)
    _ensure_dir(merged_root)
    (merged_root / "dispatch_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    merged_rows: List[Dict[str, object]] = []
    for rec in payload.get("launches", []):
        host = str(rec["host"])
        out_dir = str(rec["output_dir"]).rstrip("/")
        ip = _host_ip(host)
        alias = HOST_ALIAS.get(ip, ip.replace(".", "_"))
        worker = str(rec["worker"])

        local_worker_dir = collected_root / alias / worker
        _ensure_dir(local_worker_dir)

        for filename in (
            "eor_extra_priors_results.csv",
            "eor_extra_priors_rank.csv",
            "eor_extra_priors_summary.md",
            "manifest.json",
            "worker.log",
        ):
            remote_file = f"{out_dir}/{filename}"
            local_file = local_worker_dir / filename
            if local_file.exists() and not args.overwrite:
                continue
            try:
                text = _fetch_text(host, remote_file)
            except subprocess.CalledProcessError:
                continue
            local_file.write_text(text, encoding="utf-8")

        results_path = local_worker_dir / "eor_extra_priors_results.csv"
        if results_path.exists():
            for row in _read_csv_rows(results_path):
                row["_source"] = str(results_path)
                merged_rows.append(row)

    partial_path = merged_root / "eor_extra_priors_results_partial_merged.csv"
    _write_csv(partial_path, merged_rows)
    merged_path = merged_root / "eor_extra_priors_results_merged.csv"
    _write_csv(merged_path, merged_rows)

    ranked = _candidate_summary(merged_rows, exclude_datasets=exclude_names)
    rank_path = merged_root / "eor_extra_priors_rank_merged.csv"
    _write_csv(rank_path, ranked)

    baseline_fixed: Dict[str, object] = {}
    for rec in payload.get("launches", []):
        host = str(rec["host"])
        out_dir = str(rec["output_dir"]).rstrip("/")
        try:
            text = _fetch_text(host, f"{out_dir}/manifest.json")
        except subprocess.CalledProcessError:
            continue
        try:
            worker_manifest = json.loads(text)
        except json.JSONDecodeError:
            continue
        baseline_fixed = worker_manifest.get("baseline_fixed", {}) or {}
        break
    _write_markdown(merged_root / "eor_extra_priors_summary_merged.md", ranked, baseline_fixed)

    print(f"[collect] output_root={output_root}")
    print(f"[collect] merged_rows={len(merged_rows)}")
    print(f"[collect] merged_csv={merged_path}")
    print(f"[collect] rank_csv={rank_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

