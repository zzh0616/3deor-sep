#!/usr/bin/env python3
"""Build frozen overlapping local-redshift Q_beta analysis configs."""

from __future__ import annotations

import argparse
import copy
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

from visibility_qbeta_local_redshift import (  # noqa: E402
    build_local_redshift_windows,
    freeze_frequency_view_config,
    freeze_local_config,
)
from ps2d_v2_config import (  # noqa: E402
    FROZEN_GEOMETRY_KEYS,
    resolve_mode_first_geometry,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--analysis-template", type=Path, required=True)
    parser.add_argument("--input-template", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-channel-count", type=int, default=64)
    parser.add_argument("--analysis-channel-count", type=int, default=32)
    parser.add_argument("--stride-channels", type=int, default=16)
    parser.add_argument("--target-start", type=int, default=32)
    parser.add_argument("--target-stop", type=int, default=96)
    parser.add_argument(
        "--foreground-support-angle-deg",
        type=float,
        default=None,
        help=(
            "Truth-blind angular foreground support used by both DPSS delays "
            "and the frozen EoR-window wedge."
        ),
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    full = json.loads(args.full_config.read_text(encoding="utf-8"))
    analysis_template = json.loads(
        args.analysis_template.read_text(encoding="utf-8")
    )
    input_template = json.loads(
        args.input_template.read_text(encoding="utf-8")
    )
    if args.foreground_support_angle_deg is not None:
        support_angle = float(args.foreground_support_angle_deg)
        for config in (full, analysis_template, input_template):
            config.setdefault("eor_window", {})[
                "foreground_support_angle_deg"
            ] = support_angle
    frequencies = np.asarray(full["frequencies_mhz"], dtype=np.float64)
    common_reference = float(full["reference_frequency_mhz"])
    common_geometry = {
        name: float(value)
        for name, value in full["frozen_geometry"].items()
    }
    if args.foreground_support_angle_deg is not None:
        live_full = copy.deepcopy(full)
        live_full.pop("frozen_geometry", None)
        live_full.pop("frozen_analysis_contract_sha256", None)
        live_full.pop("frozen_analysis_window_energy", None)
        live_geometry = resolve_mode_first_geometry(live_full)
        # Keep all established geometry frozen; only the explicit angular
        # support is allowed to move the patch wedge.
        common_geometry["patch_wedge_slope"] = float(
            live_geometry["patch_wedge_slope"]
        )
        if set(common_geometry) != set(FROZEN_GEOMETRY_KEYS):
            raise ValueError("Unexpected frozen-geometry contract")
    windows = build_local_redshift_windows(
        frequencies,
        input_channel_count=int(args.input_channel_count),
        analysis_channel_count=int(args.analysis_channel_count),
        stride_channels=int(args.stride_channels),
        target_start=int(args.target_start),
        target_stop=int(args.target_stop),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_windows: list[dict[str, Any]] = []
    for window in windows:
        analysis = freeze_local_config(
            analysis_template,
            frequencies_mhz=window.analysis_frequencies_mhz,
            reference_frequency_mhz=common_reference,
            status=f"{window.label}_analysis",
            frozen_geometry=common_geometry,
        )
        input_config = freeze_frequency_view_config(
            input_template,
            frequencies_mhz=window.input_frequencies_mhz,
            reference_frequency_mhz=common_reference,
            status=f"{window.label}_input",
            frozen_geometry=common_geometry,
        )
        analysis_name = f"{window.label}_analysis.json"
        input_name = f"{window.label}_input.json"
        _atomic_json(args.out_dir / analysis_name, analysis)
        _atomic_json(args.out_dir / input_name, input_config)
        manifest_windows.append(
            {
                "label": window.label,
                "index": int(window.index),
                "input_start": int(window.input_start),
                "input_stop": int(window.input_stop),
                "analysis_start": int(window.analysis_start),
                "analysis_stop": int(window.analysis_stop),
                "input_frequency_range_mhz": [
                    float(window.input_frequencies_mhz[0]),
                    float(window.input_frequencies_mhz[-1]),
                ],
                "analysis_frequency_range_mhz": [
                    float(window.analysis_frequencies_mhz[0]),
                    float(window.analysis_frequencies_mhz[-1]),
                ],
                "reference_frequency_mhz": (
                    window.reference_frequency_mhz
                ),
                "geometry_reference_frequency_mhz": common_reference,
                "analysis_config": analysis_name,
                "input_config": input_name,
                "analysis_contract_sha256": analysis[
                    "frozen_analysis_contract_sha256"
                ],
                "input_contract_sha256": input_config[
                    "frequency_view_contract_sha256"
                ],
            }
        )
    manifest = {
        "schema": "visibility_qbeta_local_redshift_config_set",
        "schema_version": 1,
        "full_config": str(args.full_config),
        "input_channel_count": int(args.input_channel_count),
        "analysis_channel_count": int(args.analysis_channel_count),
        "stride_channels": int(args.stride_channels),
        "target_start": int(args.target_start),
        "target_stop": int(args.target_stop),
        "geometry_reference_frequency_mhz": common_reference,
        "foreground_support_angle_deg": (
            None
            if args.foreground_support_angle_deg is None
            else float(args.foreground_support_angle_deg)
        ),
        "windows": manifest_windows,
    }
    _atomic_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
