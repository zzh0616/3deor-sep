#!/usr/bin/env python3
"""
Recompute per-frequency corr(EoR_est, EoR_true) metrics for existing runs.

Why: some older scan scripts computed truth-cube cuts with a slightly different
center convention than separation_optim.cut_xy, shifting the crop by ~1 px and
corrupting correlation metrics. This utility recomputes correlations using the
cut_xy settings stored in each run's config.json.

Inputs:
- a CSV produced by scan scripts (expects columns including config_path,eor_output,status)

Outputs:
- a corrected CSV with updated eor_corr_* fields (and inj_fg_eor_corr_abs_mean if possible)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute EoR corr metrics for a scan results CSV.")
    p.add_argument("--results-csv", type=Path, required=True, help="Input results CSV.")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <results>.corrected.csv).",
    )
    p.add_argument(
        "--write-profiles",
        action="store_true",
        help="Write corrected per-frequency profiles to <run_dir>/eor_corr_profile_corrected.csv.",
    )
    return p.parse_args()


def _clamp_fixed_window(start: int, size: int, length: int) -> Tuple[int, int]:
    if size > length:
        raise ValueError(f"Requested crop size {size} exceeds axis length {length}.")
    end = start + size
    if start < 0:
        start = 0
        end = size
    if end > length:
        end = length
        start = length - size
    return int(start), int(end)


def compute_cut_xy(
    shape: Sequence[int],
    *,
    freq_axis: int,
    enabled: bool,
    unit: str,
    center_x: Optional[float],
    center_y: Optional[float],
    size: Optional[float],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    if not enabled:
        return None
    if len(shape) != 3:
        raise ValueError(f"cut_xy expects a 3D cube, got shape {tuple(shape)}")
    if not (0 <= int(freq_axis) < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    spatial_axes = [ax for ax in range(3) if ax != int(freq_axis)]
    x_axis, y_axis = spatial_axes[0], spatial_axes[1]
    nx, ny = int(shape[x_axis]), int(shape[y_axis])
    min_dim = min(nx, ny)

    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"frac", "px"}:
        raise ValueError("cut_xy.unit must be 'frac' or 'px'.")

    if unit_norm == "frac":
        cx = 0.5 if center_x is None else float(center_x)
        cy = 0.5 if center_y is None else float(center_y)
        frac = 0.5 if size is None else float(size)
        size_px = int(round(frac * float(min_dim)))
        center_x_px = int(round(cx * float(max(nx - 1, 0))))
        center_y_px = int(round(cy * float(max(ny - 1, 0))))
    else:
        center_x_px = nx // 2 if center_x is None else int(round(float(center_x)))
        center_y_px = ny // 2 if center_y is None else int(round(float(center_y)))
        size_px = int(round(0.5 * float(min_dim))) if size is None else int(round(float(size)))

    size_px = max(1, int(size_px))
    if size_px > min_dim:
        raise ValueError(f"cut_xy.size={size_px} exceeds min(Nx,Ny)={min_dim}.")

    start_x = int(center_x_px) - int(size_px) // 2
    start_y = int(center_y_px) - int(size_px) // 2
    x0, x1 = _clamp_fixed_window(start_x, int(size_px), int(nx))
    y0, y1 = _clamp_fixed_window(start_y, int(size_px), int(ny))
    return (int(x_axis), int(y_axis), int(x0), int(x1), int(y0), int(y1))


def load_fits_cut(path: Path, *, cut: Optional[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None or getattr(data, "ndim", None) != 3:
            raise ValueError(f"Expected 3D cube in {path}")
        view = data
        if cut is not None:
            x_axis, y_axis, x0, x1, y0, y1 = cut
            slices = [slice(None)] * 3
            slices[int(x_axis)] = slice(int(x0), int(x1))
            slices[int(y_axis)] = slice(int(y0), int(y1))
            view = view[tuple(slices)]
        return np.asarray(view, dtype=np.float32)


def frequency_correlations(est: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if est.shape != true.shape:
        raise ValueError(f"Shape mismatch: est={est.shape} true={true.shape}")
    f = est.shape[0]
    out = np.full((f,), np.nan, dtype=np.float64)
    for i in range(f):
        a = est[i].reshape(-1).astype(np.float64, copy=False)
        b = true[i].reshape(-1).astype(np.float64, copy=False)
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + float(eps)
        out[i] = float(np.dot(a, b) / denom)
    return out


def corr_score(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    n = max(1, int(round(0.2 * finite.size)))
    worst = np.sort(finite)[:n]
    return float(np.mean(worst))


def summarize_corr(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "min": float("nan"), "max": float("nan"), "score": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "score": float(corr_score(finite)),
    }


def write_profile_csv(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, v in enumerate(vec.tolist()):
            w.writerow([i, v])


def main() -> int:
    args = parse_args()
    in_csv = args.results_csv.resolve()
    out_csv = args.out_csv.resolve() if args.out_csv else in_csv.with_suffix(in_csv.suffix + ".corrected.csv")

    rows: List[Dict[str, str]] = []
    with in_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        cfg_path = Path(str(row["config_path"]))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        optim = cfg.get("optim") if isinstance(cfg.get("optim"), dict) else {}
        freq_axis = int(optim.get("freq_axis", 0)) if isinstance(optim, dict) else int(cfg.get("freq_axis", 0))
        cut_xy = cfg.get("cut_xy") if isinstance(cfg.get("cut_xy"), dict) else {}

        enabled = bool(cut_xy.get("enabled", False)) if isinstance(cut_xy, dict) else False
        unit = str(cut_xy.get("unit", "frac")) if isinstance(cut_xy, dict) else "frac"
        center_x = float(cut_xy.get("center_x", 0.5)) if isinstance(cut_xy, dict) and "center_x" in cut_xy else None
        center_y = float(cut_xy.get("center_y", 0.5)) if isinstance(cut_xy, dict) and "center_y" in cut_xy else None
        size = float(cut_xy.get("size", 1.0)) if isinstance(cut_xy, dict) and "size" in cut_xy else None

        # Paths.
        eor_est_path = Path(str(row["eor_output"]))
        ev = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
        true_eor_path = None
        if isinstance(ev, dict) and ev.get("true_eor_cube"):
            true_eor_path = Path(str(ev["true_eor_cube"]))
        elif cfg.get("true_eor_cube"):
            true_eor_path = Path(str(cfg["true_eor_cube"]))
        if true_eor_path is None:
            raise ValueError(f"Missing true_eor_cube in config {cfg_path}")
        if not true_eor_path.exists():
            raise FileNotFoundError(f"true_eor_cube not found: {true_eor_path}")

        # Compute cut from the truth cube header shape.
        with fits.open(true_eor_path, memmap=True) as hdul:
            shape = tuple(int(v) for v in hdul[0].data.shape)
        cut = compute_cut_xy(
            shape,
            freq_axis=freq_axis,
            enabled=enabled,
            unit=unit,
            center_x=center_x,
            center_y=center_y,
            size=size,
        )

        est = load_fits_cut(eor_est_path, cut=None)
        true = load_fits_cut(true_eor_path, cut=cut)
        if est.shape != true.shape:
            raise ValueError(f"Shape mismatch after cut: est={est.shape} true={true.shape} ({eor_est_path})")

        vec = frequency_correlations(est, true)
        stats = summarize_corr(vec)
        row["eor_corr_mean"] = f"{stats['mean']:.10g}"
        row["eor_corr_p10"] = f"{stats['p10']:.10g}"
        row["eor_corr_min"] = f"{stats['min']:.10g}"
        row["eor_corr_max"] = f"{stats['max']:.10g}"
        row["eor_corr_score"] = f"{stats['score']:.10g}"

        if bool(args.write_profiles):
            write_profile_csv(eor_est_path.parent / "eor_corr_profile_corrected.csv", vec)

        # Try to fix injected FG/EoR corr abs mean if present.
        if row.get("inj_fg_eor_corr_abs_mean"):
            try:
                true_fg_path = Path(str(true_eor_path).replace("eor_cube", "fg_cube"))
                if true_fg_path.exists():
                    fg_true = load_fits_cut(true_fg_path, cut=cut)
                    fg_eor = frequency_correlations(fg_true, true)
                    abs_mean = float(np.mean(np.abs(fg_eor[np.isfinite(fg_eor)]))) if np.any(np.isfinite(fg_eor)) else float("nan")
                    row["inj_fg_eor_corr_abs_mean"] = f"{abs_mean:.10g}"
            except Exception:
                pass

    # Write corrected CSV.
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote corrected CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
