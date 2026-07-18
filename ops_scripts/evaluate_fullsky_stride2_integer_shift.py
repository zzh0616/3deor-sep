#!/usr/bin/env python3
"""Fast local-bilinear full-sky closure for integer response grids.

The generic ``local_bilinear_conv`` evaluator aligns every support response to a
common centre and runs one FFT convolution per support point.  For the
stride2edge grid used here, both support positions and foreground source pixels
are integer pixel coordinates.  The same prediction can therefore be assembled
by integer-shifting each exact response by the source/support pixel offset and
accumulating the corresponding foreground weights.  Optional stride filtering
lets the completed stride2edge bank emulate coarser integer sub-grids without
generating more WSClean responses.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_fullsky_response_interpolation import (  # noqa: E402
    GridRow,
    _central_crop,
    _fmt,
    _freq_to_index,
    _k_to_jy_per_pixel,
    _load_cube_slice,
    _load_fits_2d,
    _load_grid_csv,
    _metric,
    _parse_floats,
    _parse_ints,
    _stack_fft_metric,
    _write_image,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--response-root", type=Path, required=True)
    ap.add_argument("--truth-root", type=Path, required=True)
    ap.add_argument("--fg-cube-k", type=Path, required=True)
    ap.add_argument("--freqs-mhz", required=True)
    ap.add_argument("--freq0-mhz", type=float, default=106.0)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument("--pixel-arcsec", type=float, default=32.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--eval-crop-sizes", default="64,96,128,160,192,224,256,320,384,448,512")
    ap.add_argument("--response-pattern", required=True)
    ap.add_argument("--grid-csv-pattern", required=True)
    ap.add_argument("--truth-fg-pattern", required=True)
    ap.add_argument("--truth-eor-pattern", required=True)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--save-products", action="store_true")
    ap.add_argument("--progress-every", type=int, default=2048)
    ap.add_argument(
        "--support-stride-pixels",
        type=int,
        default=0,
        help="If positive, keep only integer support coordinates divisible by this stride.",
    )
    ap.add_argument(
        "--include-edge",
        action="store_true",
        help="When filtering support stride, also keep image_size-1 on each axis.",
    )
    return ap.parse_args()


AxisContrib = Tuple[int, float, int]


def _axis_linear_contribs(size: int, support: np.ndarray) -> List[List[AxisContrib]]:
    """Return per-support pixel contributions ``(pixel, weight, pixel-support)``."""
    contribs: List[List[AxisContrib]] = [[] for _ in range(int(support.size))]
    for pix in range(int(size)):
        coord = float(pix)
        hi = int(np.searchsorted(support, coord, side="right"))
        hi = min(max(hi, 1), int(support.size) - 1)
        lo = hi - 1
        denom = float(support[hi] - support[lo])
        t = (coord - float(support[lo])) / denom if denom != 0.0 else 0.0
        pairs: Sequence[Tuple[int, float]]
        if coord <= float(support[0]):
            pairs = ((0, 1.0),)
        elif coord >= float(support[-1]):
            pairs = ((int(support.size) - 1, 1.0),)
        else:
            pairs = ((lo, 1.0 - t), (hi, t))
        for idx, weight in pairs:
            if weight == 0.0:
                continue
            offset = int(round(float(pix) - float(support[idx])))
            contribs[idx].append((pix, float(weight), offset))
    return contribs


def _filter_integer_support(
    rows: Sequence[GridRow],
    x_support: np.ndarray,
    y_support: np.ndarray,
    *,
    stride_pixels: int,
    include_edge: bool,
    image_size: int,
) -> Tuple[List[GridRow], np.ndarray, np.ndarray]:
    stride = int(stride_pixels)
    if stride <= 0:
        return list(rows), np.asarray(x_support), np.asarray(y_support)

    edge = int(image_size) - 1

    def keep(v: float) -> bool:
        iv = int(round(float(v)))
        if abs(float(v) - float(iv)) > 1e-6:
            return False
        return (iv % stride == 0) or (bool(include_edge) and iv == edge)

    new_x = np.asarray([float(v) for v in x_support if keep(float(v))], dtype=np.float64)
    new_y = np.asarray([float(v) for v in y_support if keep(float(v))], dtype=np.float64)
    if new_x.size < 2 or new_y.size < 2:
        raise ValueError(f"Filtered support is too small: {new_x.size} x {new_y.size}")

    by_xy = {(int(round(float(r.x))), int(round(float(r.y)))): r for r in rows}
    new_rows: List[GridRow] = []
    for iy, y in enumerate(new_y):
        for ix, x in enumerate(new_x):
            key = (int(round(float(x))), int(round(float(y))))
            src = by_xy.get(key)
            if src is None:
                raise ValueError(f"Missing response row for filtered support {key}")
            new_rows.append(GridRow(label=src.label, x=float(src.x), y=float(src.y), ix=int(ix), iy=int(iy)))
    return new_rows, new_x, new_y


def _add_shifted(pred: np.ndarray, response: np.ndarray, *, coeff: float, dy: int, dx: int) -> None:
    if coeff == 0.0:
        return
    h, w = response.shape
    sy0 = max(0, -int(dy))
    sy1 = min(h, h - int(dy))
    sx0 = max(0, -int(dx))
    sx1 = min(w, w - int(dx))
    dy0 = max(0, int(dy))
    dx0 = max(0, int(dx))
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)
    if sy1 > sy0 and sx1 > sx0:
        pred[dy0:dy1, dx0:dx1] += float(coeff) * np.asarray(response[sy0:sy1, sx0:sx1], dtype=np.float64)


def _predict_fast_integer_shift(
    *,
    fg_flux: np.ndarray,
    rows: Sequence[GridRow],
    x_support: np.ndarray,
    y_support: np.ndarray,
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
    image_size: int,
    progress_every: int,
) -> np.ndarray:
    x_contribs = _axis_linear_contribs(int(image_size), x_support)
    y_contribs = _axis_linear_contribs(int(image_size), y_support)
    pred = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    flux = np.asarray(fg_flux, dtype=np.float64)
    total = len(rows)
    for n, row in enumerate(rows, start=1):
        coeff_by_offset: Dict[Tuple[int, int], float] = {}
        for y_pix, wy, dy in y_contribs[int(row.iy)]:
            for x_pix, wx, dx in x_contribs[int(row.ix)]:
                coeff = float(flux[y_pix, x_pix]) * float(wy) * float(wx)
                if coeff != 0.0:
                    key = (int(dy), int(dx))
                    coeff_by_offset[key] = coeff_by_offset.get(key, 0.0) + coeff
        if coeff_by_offset:
            response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
            for (dy, dx), coeff in coeff_by_offset.items():
                _add_shifted(pred, response, coeff=coeff, dy=dy, dx=dx)
        if progress_every > 0 and (n == 1 or n % int(progress_every) == 0 or n == total):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "rows_done": int(n),
                        "rows_total": int(total),
                        "frac": float(n / max(total, 1)),
                        "time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return pred


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.float64 if args.dtype == "float64" else np.float32)
    freqs = _parse_floats(args.freqs_mhz)
    crop_sizes = _parse_ints(args.eval_crop_sizes)
    method = "local_bilinear_conv"

    per_freq: List[Dict[str, Any]] = []
    stacks: Dict[int, Dict[str, List[np.ndarray]]] = {
        crop: {"residual": [], "eor": []} for crop in crop_sizes
    }
    for freq in freqs:
        grid_csv = _fmt(args.grid_csv_pattern, freq=freq)
        rows, x_support, y_support = _load_grid_csv(grid_csv)
        rows, x_support, y_support = _filter_integer_support(
            rows,
            x_support,
            y_support,
            stride_pixels=int(args.support_stride_pixels),
            include_edge=bool(args.include_edge),
            image_size=int(args.image_size),
        )
        freq_index = _freq_to_index(freq, float(args.freq0_mhz), float(args.freq_step_mhz))
        fg_k = _load_cube_slice(args.fg_cube_k, freq_index, dtype=np.float64)
        fg_flux = fg_k * _k_to_jy_per_pixel(freq, float(args.pixel_arcsec))
        truth_fg_path = _fmt(args.truth_fg_pattern, freq=freq)
        truth_eor_path = _fmt(args.truth_eor_pattern, freq=freq)
        dirty_fg = _load_fits_2d(truth_fg_path, dtype=dtype)
        dirty_eor = _load_fits_2d(truth_eor_path, dtype=dtype)
        if dirty_fg.shape != (int(args.image_size), int(args.image_size)):
            raise ValueError(f"dirty_fg shape {dirty_fg.shape} does not match image_size={args.image_size}")

        pred_fg = _predict_fast_integer_shift(
            fg_flux=fg_flux,
            rows=rows,
            x_support=x_support,
            y_support=y_support,
            response_pattern=str(args.response_pattern),
            freq=float(freq),
            dtype=dtype,
            image_size=int(args.image_size),
            progress_every=int(args.progress_every),
        )
        residual = np.asarray(dirty_fg, dtype=np.float64) - np.asarray(pred_fg, dtype=np.float64)
        method_result: Dict[str, Any] = {
            "support_flux_sum_jy": None,
            "input_flux_sum_jy": float(np.sum(fg_flux, dtype=np.float64)),
            "support_flux_abs_sum_jy": None,
            "input_flux_abs_sum_jy": float(np.sum(np.abs(fg_flux), dtype=np.float64)),
            "fast_integer_shift_equivalent": True,
            "support_stride_pixels": int(args.support_stride_pixels),
            "include_edge": bool(args.include_edge),
            "crops": {},
        }
        for crop in crop_sizes:
            fg_c = _central_crop(dirty_fg, crop)
            pred_c = _central_crop(pred_fg, crop)
            eor_c = _central_crop(dirty_eor, crop)
            res_c = fg_c - pred_c
            method_result["crops"][str(int(crop))] = _metric(fg_c, pred_c, eor_c)
            stacks[int(crop)]["residual"].append(np.asarray(res_c, dtype=np.float32))
            stacks[int(crop)]["eor"].append(np.asarray(eor_c, dtype=np.float32))
        if args.save_products:
            stem = f"{method}_fast_integer_shift_{freq:.2f}"
            _write_image(out_dir / f"{stem}_pred_fg-dirty.fits", pred_fg, truth_fg_path)
            _write_image(out_dir / f"{stem}_fg_residual-dirty.fits", residual, truth_fg_path)
        freq_result = {
            "freq_mhz": float(freq),
            "freq_index": int(freq_index),
            "grid_csv": str(grid_csv),
            "truth_fg": str(truth_fg_path),
            "truth_eor": str(truth_eor_path),
            "n_support": int(len(rows)),
            "support_x_minmax": [float(x_support[0]), float(x_support[-1])],
            "support_y_minmax": [float(y_support[0]), float(y_support[-1])],
            "methods": {method: method_result},
        }
        print(
            json.dumps(
                {
                    "freq_mhz": float(freq),
                    "method": method,
                    "best_crop": min(
                        (
                            (int(crop), float(method_result["crops"][str(int(crop))]["fg_residual_over_dirty_eor_rms"]))
                            for crop in crop_sizes
                        ),
                        key=lambda item: item[1],
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        per_freq.append(freq_result)

    stack_results: Dict[str, Dict[str, Any]] = {method: {}}
    for crop in crop_sizes:
        residual_stack = np.stack(stacks[int(crop)]["residual"], axis=0)
        eor_stack = np.stack(stacks[int(crop)]["eor"], axis=0)
        stack_results[method][str(int(crop))] = _stack_fft_metric(residual_stack, eor_stack)

    output: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "settings": {
            "response_root": str(args.response_root),
            "truth_root": str(args.truth_root),
            "fg_cube_k": str(args.fg_cube_k),
            "freqs_mhz": [float(v) for v in freqs],
            "freq0_mhz": float(args.freq0_mhz),
            "freq_step_mhz": float(args.freq_step_mhz),
            "pixel_arcsec": float(args.pixel_arcsec),
            "image_size": int(args.image_size),
            "methods": [method],
            "eval_crop_sizes": [int(v) for v in crop_sizes],
            "response_pattern": str(args.response_pattern),
            "grid_csv_pattern": str(args.grid_csv_pattern),
            "truth_fg_pattern": str(args.truth_fg_pattern),
            "truth_eor_pattern": str(args.truth_eor_pattern),
            "dtype": str(dtype),
            "fast_integer_shift_equivalent": True,
            "support_stride_pixels": int(args.support_stride_pixels),
            "include_edge": bool(args.include_edge),
        },
        "per_frequency": per_freq,
        "stack_results": stack_results,
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
