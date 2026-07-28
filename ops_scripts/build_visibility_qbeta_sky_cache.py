#!/usr/bin/env python3
"""Build the intrinsic multifrequency sky cache used by visibility Q_beta."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibrate_visibility_qbeta_noiseless import _build_sky_cache  # noqa: E402
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--osm-pattern", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase-ra-deg", type=float, default=0.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    source_size = int(config["image_geometry"]["source_image_size"])
    _build_sky_cache(
        path=args.out,
        osm_pattern=str(args.osm_pattern),
        frequencies_mhz=resolved.geometry["frequencies_mhz"],
        expected_source_count=source_size * source_size,
        phase_ra_deg=float(args.phase_ra_deg),
        phase_dec_deg=float(args.phase_dec_deg),
    )


if __name__ == "__main__":
    main()
