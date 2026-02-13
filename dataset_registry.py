#!/usr/bin/env python3
"""
Dataset registry helpers for scan/dispatch scripts.

We keep dataset definitions centralized so that:
- all scan scripts operate on the same dataset naming convention;
- per-dataset frequency metadata (freq_start_mhz) is available for physics-driven priors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


_FREQ_TAG_RE = re.compile(r"^f(\d{3})$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path
    freq_start_mhz: float


def _iter_multiband_dirs(multiband_root: Path) -> Iterable[Tuple[str, int, Path]]:
    """
    Yield (name, start_mhz_int, path) from multiband_root directories like f060, f080, ...
    """
    if not multiband_root.is_dir():
        return
    for entry in sorted(multiband_root.iterdir()):
        if not entry.is_dir():
            continue
        m = _FREQ_TAG_RE.match(entry.name)
        if not m:
            continue
        start_mhz = int(m.group(1))
        yield (entry.name, start_mhz, entry)


def build_datasets(
    data_dir: Path,
    *,
    cube12_start_mhz: float = 106.0,
    multiband_dirname: str = "multiband_1024_20260213",
) -> List[DatasetSpec]:
    """
    Build the default dataset list used across scans.

    Names:
    - cube1: 106 MHz start (all_cube1)
    - cube2: 106 MHz start (all_cube2)
    - f060/f080/...: from data/<multiband_dirname>/fXXX/
    """
    data_dir = Path(data_dir)
    datasets: List[DatasetSpec] = [
        DatasetSpec(
            name="cube1",
            input_cube=data_dir / "back" / "all_cube1.fits",
            fg_true_cube=data_dir / "fg_cube1.fits",
            eor_true_cube=data_dir / "eor_cube1.fits",
            freq_start_mhz=float(cube12_start_mhz),
        ),
        DatasetSpec(
            name="cube2",
            input_cube=data_dir / "all_cube2.fits",
            fg_true_cube=data_dir / "fg_cube2.fits",
            eor_true_cube=data_dir / "eor_cube2.fits",
            freq_start_mhz=float(cube12_start_mhz),
        ),
    ]

    multiband_root = data_dir / str(multiband_dirname)
    for name, start_mhz, ds_dir in _iter_multiband_dirs(multiband_root):
        datasets.append(
            DatasetSpec(
                name=name,
                input_cube=ds_dir / "all_cube.fits",
                fg_true_cube=ds_dir / "fg_cube.fits",
                eor_true_cube=ds_dir / "eor_cube.fits",
                freq_start_mhz=float(start_mhz),
            )
        )

    # Validate existence early so scans fail fast and loudly.
    for ds in datasets:
        for p in (ds.input_cube, ds.fg_true_cube, ds.eor_true_cube):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file for dataset '{ds.name}': {p}")
    return datasets


def parse_dataset_names(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def filter_datasets(datasets: Sequence[DatasetSpec], enabled: Sequence[str]) -> List[DatasetSpec]:
    enabled_set = set(str(x).strip() for x in enabled if str(x).strip())
    return [d for d in datasets if d.name in enabled_set]


def parse_excluded_names(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def default_dataset_name_hint() -> str:
    # Keep this stable for help-text: it's fine if new multiband dirs appear;
    # scan scripts can still pass explicit --datasets.
    return "cube1,cube2,f060,f080,f120,f140,f160,f180"

