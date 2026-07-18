#!/usr/bin/env python3
"""Evaluate sparse image-domain response interpolation by grid hold-out.

Given a dense grid of exact point-response dirty images, this script keeps a
coarser sub-grid as interpolation support and predicts the held-out responses.
The intended use is to estimate how dense an OSKAR/WSClean response grid must
be before it can replace per-source point-response simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
from scipy import ndimage

from build_planA_real_wsclean_dirty_base import _corr, _crop2d, _fit_gain, _load_wsclean_2d


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--grid-csv", type=Path, required=True)
    ap.add_argument("--response-pattern", required=True, help="FITS path pattern formatted with {label} and {freq:.2f}.")
    ap.add_argument("--freq-mhz", type=float, default=106.0)
    ap.add_argument(
        "--position-mode",
        choices=("csv", "wcs"),
        default="csv",
        help=(
            "Use pixel_x/pixel_y from the CSV, or recompute pixel coordinates "
            "from ra_deg/dec_deg using the response FITS WCS. Use wcs for "
            "re-imaged lower-resolution products."
        ),
    )
    ap.add_argument("--train-stride", type=int, default=2, help="Keep every Nth grid point in x/y as interpolation support.")
    ap.add_argument("--holdout-mode", choices=("all", "cell-centres", "random"), default="cell-centres")
    ap.add_argument("--max-holdouts", type=int, default=0, help="Limit held-out positions for a quick smoke test.")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--crop-size", type=int, default=2048)
    ap.add_argument("--eval-crop-size", type=int, default=1536)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    ap.add_argument("--shift-method", choices=("integer", "linear", "cubic"), default="linear")
    ap.add_argument("--fit-demean", action="store_true")
    return ap.parse_args()


def _fmt(pattern: str, *, label: str, freq: float) -> Path:
    return Path(str(pattern).format(label=label, freq=float(freq)))


def _load_grid_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"label", "pixel_x", "pixel_y", "ra_deg", "dec_deg"}
    if not required.issubset(rows[0].keys() if rows else set()):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "label": row["label"],
                "x": float(row["pixel_x"]),
                "y": float(row["pixel_y"]),
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
            }
        )
    return out


def _assign_grid_indices(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Assign dense-grid integer indices from the original CSV pixel positions."""
    orig_x = np.array(sorted({round(float(row["x"]), 6) for row in rows}), dtype=np.float64)
    orig_y = np.array(sorted({round(float(row["y"]), 6) for row in rows}), dtype=np.float64)
    for row in rows:
        row["ix"] = int(np.argmin(np.abs(orig_x - float(row["x"]))))
        row["iy"] = int(np.argmin(np.abs(orig_y - float(row["y"]))))
    return orig_x, orig_y


def _apply_wcs_positions(rows: List[Dict[str, Any]], header: fits.Header) -> None:
    wcs = WCS(header).celestial
    for row in rows:
        coord = SkyCoord(float(row["ra_deg"]) * u.deg, float(row["dec_deg"]) * u.deg, frame="icrs")
        x, y = skycoord_to_pixel(coord, wcs, origin=0)
        row["x"] = float(x)
        row["y"] = float(y)


def _nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(values - value)))


def _bracket(values: np.ndarray, value: float) -> Tuple[int, int, float]:
    if value < values[0] or value > values[-1]:
        raise ValueError(f"value {value} outside interpolation support {values[0]}..{values[-1]}")
    if value == values[0]:
        return 0, 0, 0.0
    if value == values[-1]:
        n = len(values) - 1
        return n, n, 0.0
    hi = int(np.searchsorted(values, value, side="right"))
    lo = hi - 1
    t = float((value - values[lo]) / (values[hi] - values[lo]))
    return lo, hi, t


def _shift_zero(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(arr)
    h, w = arr.shape
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def _shift_subpixel_zero(arr: np.ndarray, dx: float, dy: float, order: int) -> np.ndarray:
    return np.asarray(ndimage.shift(arr, shift=(dy, dx), order=order, mode="constant", cval=0.0), dtype=arr.dtype)


def _central_crop(arr: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size >= min(arr.shape):
        return arr
    h, w = arr.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0 : y0 + size, x0 : x0 + size]


def _stats(target: np.ndarray, pred: np.ndarray, *, fit_demean: bool) -> Dict[str, float]:
    gain, stats = _fit_gain(target, pred, demean=fit_demean)
    stats["gain"] = float(gain)
    stats["corr_before_gain"] = float(_corr(target, pred))
    stats["target_rms"] = float(np.sqrt(np.mean(np.asarray(target, dtype=np.float64) ** 2)))
    stats["pred_rms"] = float(np.sqrt(np.mean(np.asarray(pred, dtype=np.float64) ** 2)))
    stats["resid_rms_abs"] = float(np.sqrt(np.mean((np.asarray(target, dtype=np.float64) - gain * np.asarray(pred, dtype=np.float64)) ** 2)))
    return {k: float(v) for k, v in stats.items()}


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.float64 if args.dtype == "float64" else np.float32)

    rows = _load_grid_csv(args.grid_csv)
    orig_x_values, orig_y_values = _assign_grid_indices(rows)
    if args.position_mode == "wcs":
        first_label = str(rows[0]["label"])
        _arr, first_header = _load_wsclean_2d(_fmt(args.response_pattern, label=first_label, freq=float(args.freq_mhz)), dtype)
        _apply_wcs_positions(rows, first_header)
    n_x = int(len(orig_x_values))
    n_y = int(len(orig_y_values))
    grid_by_ij: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        ix = int(row["ix"])
        iy = int(row["iy"])
        grid_by_ij[(ix, iy)] = row
    if len(grid_by_ij) != n_x * n_y:
        raise ValueError(f"Grid is incomplete in CSV: {len(grid_by_ij)} rows for {n_x}x{n_y}")

    x_values = np.empty(n_x, dtype=np.float64)
    y_values = np.empty(n_y, dtype=np.float64)
    for ix in range(n_x):
        x_values[ix] = float(np.median([row["x"] for row in rows if int(row["ix"]) == ix]))
    for iy in range(n_y):
        y_values[iy] = float(np.median([row["y"] for row in rows if int(row["iy"]) == iy]))

    train_ix = list(range(0, n_x, int(args.train_stride)))
    train_iy = list(range(0, n_y, int(args.train_stride)))
    if train_ix[-1] != n_x - 1:
        train_ix.append(n_x - 1)
    if train_iy[-1] != n_y - 1:
        train_iy.append(n_y - 1)
    train_x = x_values[train_ix]
    train_y = y_values[train_iy]
    train_set = {(ix, iy) for ix in train_ix for iy in train_iy}

    holdouts: List[Tuple[int, int]] = []
    cell_centre_offset = max(1, int(args.train_stride) // 2)
    for iy in range(n_y):
        for ix in range(n_x):
            if (ix, iy) in train_set:
                continue
            if args.holdout_mode == "cell-centres" and (
                ix % int(args.train_stride) != cell_centre_offset
                or iy % int(args.train_stride) != cell_centre_offset
            ):
                continue
            holdouts.append((ix, iy))
    if args.holdout_mode == "random":
        rng = random.Random(int(args.random_seed))
        rng.shuffle(holdouts)
    if int(args.max_holdouts) > 0:
        holdouts = holdouts[: int(args.max_holdouts)]

    response_cache: Dict[str, np.ndarray] = {}

    def load_response(label: str) -> np.ndarray:
        if label not in response_cache:
            arr, _ = _load_wsclean_2d(_fmt(args.response_pattern, label=label, freq=float(args.freq_mhz)), dtype)
            response_cache[label] = _crop2d(arr, size=int(args.crop_size), center_x=None, center_y=None)
        return response_cache[label]

    per_holdout: List[Dict[str, Any]] = []
    for ix, iy in holdouts:
        target_row = grid_by_ij[(ix, iy)]
        x = float(target_row["x"])
        y = float(target_row["y"])
        tx0, tx1, ax = _bracket(train_x, x)
        ty0, ty1, ay = _bracket(train_y, y)
        candidate_train = [
            (train_ix[tx0], train_iy[ty0], (1.0 - ax) * (1.0 - ay)),
            (train_ix[tx1], train_iy[ty0], ax * (1.0 - ay)),
            (train_ix[tx0], train_iy[ty1], (1.0 - ax) * ay),
            (train_ix[tx1], train_iy[ty1], ax * ay),
        ]
        pred: np.ndarray | None = None
        weights: List[Dict[str, Any]] = []
        for jx, jy, weight in candidate_train:
            if weight == 0.0:
                continue
            support_row = grid_by_ij[(jx, jy)]
            response = load_response(str(support_row["label"]))
            dx = x - float(support_row["x"])
            dy = y - float(support_row["y"])
            if args.shift_method == "integer":
                shifted = _shift_zero(response, dx=int(round(dx)), dy=int(round(dy)))
            else:
                shifted = _shift_subpixel_zero(response, dx=dx, dy=dy, order=1 if args.shift_method == "linear" else 3)
            if pred is None:
                pred = np.zeros_like(shifted, dtype=np.float64)
            pred += float(weight) * shifted
            weights.append({"label": support_row["label"], "ix": int(jx), "iy": int(jy), "weight": float(weight), "dx": float(dx), "dy": float(dy)})

        if pred is None:
            raise RuntimeError(f"No interpolation support for {target_row['label']}")
        target = load_response(str(target_row["label"]))
        full_stats = _stats(target, pred, fit_demean=bool(args.fit_demean))
        crop_stats = _stats(
            _central_crop(target, int(args.eval_crop_size)),
            _central_crop(pred, int(args.eval_crop_size)),
            fit_demean=bool(args.fit_demean),
        )
        per_holdout.append(
            {
                "label": target_row["label"],
                "ix": int(ix),
                "iy": int(iy),
                "x": x,
                "y": y,
                "weights": weights,
                "full": full_stats,
                "eval_crop": crop_stats,
            }
        )

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "settings": {
            "grid_csv": str(args.grid_csv),
            "response_pattern": str(args.response_pattern),
            "freq_mhz": float(args.freq_mhz),
            "position_mode": str(args.position_mode),
            "train_stride": int(args.train_stride),
            "holdout_mode": str(args.holdout_mode),
            "max_holdouts": int(args.max_holdouts),
            "crop_size": int(args.crop_size),
            "eval_crop_size": int(args.eval_crop_size),
            "dtype": str(dtype),
            "shift_method": str(args.shift_method),
            "fit_demean": bool(args.fit_demean),
        },
        "grid": {
            "n_x_dense": int(n_x),
            "n_y_dense": int(n_y),
            "n_x_train": int(len(train_x)),
            "n_y_train": int(len(train_y)),
            "x_train_indices": [int(v) for v in train_ix],
            "y_train_indices": [int(v) for v in train_iy],
            "n_holdouts": int(len(per_holdout)),
        },
        "summary": {
            "full_resid_over_dirty_rms": _summarize(row["full"]["resid_over_dirty_rms"] for row in per_holdout),
            "crop_resid_over_dirty_rms": _summarize(row["eval_crop"]["resid_over_dirty_rms"] for row in per_holdout),
            "full_corr": _summarize(row["full"]["corr"] for row in per_holdout),
            "crop_corr": _summarize(row["eval_crop"]["corr"] for row in per_holdout),
            "full_gain": _summarize(row["full"]["gain"] for row in per_holdout),
            "crop_gain": _summarize(row["eval_crop"]["gain"] for row in per_holdout),
        },
        "holdouts": per_holdout,
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "n_holdouts": len(per_holdout),
                "full_resid_median": output["summary"]["full_resid_over_dirty_rms"].get("median"),
                "crop_resid_median": output["summary"]["crop_resid_over_dirty_rms"].get("median"),
                "crop_resid_p90": output["summary"]["crop_resid_over_dirty_rms"].get("p90"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
