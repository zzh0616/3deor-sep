#!/usr/bin/env python3
"""Evaluate full-image foreground closure from sparse dirty responses.

This diagnostic compares a directly simulated full-sky dirty foreground image
against a prediction assembled from a sparse dirty-response bank.  The sparse
bank contains one dirty image per unit-flux point source.  For each full-sky
foreground pixel, its Jy flux is distributed to the response support grid using
nearest-neighbour, bilinear, or cubic interpolation weights; the weighted
response sum is then compared with the direct dirty foreground image.

The key leakage metric is the foreground closure residual divided by the direct
dirty-EoR RMS, evaluated on one or more central crops.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@dataclass(frozen=True)
class GridRow:
    label: str
    x: float
    y: float
    ix: int
    iy: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--response-root", type=Path, required=True)
    ap.add_argument("--truth-root", type=Path, required=True)
    ap.add_argument("--fg-cube-k", type=Path, required=True)
    ap.add_argument("--freqs-mhz", required=True, help="Comma list or start:stop:step in MHz.")
    ap.add_argument("--freq0-mhz", type=float, default=106.0)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument("--pixel-arcsec", type=float, default=32.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument(
        "--methods",
        default=(
            "nearest,bilinear,cubic,cubic_pos,global_center_conv,"
            "local_nearest_conv,local_bilinear_conv"
        ),
    )
    ap.add_argument("--eval-crop-sizes", default="128,256,512")
    ap.add_argument("--response-pattern", default="")
    ap.add_argument("--grid-csv-pattern", default="")
    ap.add_argument("--truth-fg-pattern", default="")
    ap.add_argument("--truth-eor-pattern", default="")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument(
        "--local-shift-method",
        choices=("integer", "linear", "cubic"),
        default="integer",
        help=(
            "Shift method used when local convolution methods align an off-axis "
            "point response to the common kernel centre. The historical default "
            "is integer; linear/cubic preserve fractional support coordinates."
        ),
    )
    ap.add_argument(
        "--local-pad-pixels",
        type=int,
        default=0,
        help=(
            "Extra zero padding on each side for local convolution kernels. "
            "Use this when aligning off-axis responses to avoid truncating the "
            "finite dirty-response image before convolution."
        ),
    )
    ap.add_argument("--save-products", action="store_true")
    ap.add_argument("--power-window", action="store_true")
    ap.add_argument("--power-nbins-kperp", type=int, default=36)
    ap.add_argument("--power-nbins-kpar", type=int, default=36)
    ap.add_argument("--power-kpar-min", type=float, default=0.1356)
    ap.add_argument("--power-ref-freq-mhz", type=float, default=0.0)
    return ap.parse_args()


def _parse_floats(spec: str) -> List[float]:
    text = str(spec).strip()
    if ":" in text:
        start, stop, step = [float(v) for v in text.split(":")]
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_ints(spec: str) -> List[int]:
    return [int(v.strip()) for v in str(spec).split(",") if v.strip()]


def _parse_methods(spec: str) -> List[str]:
    methods = [v.strip().lower() for v in str(spec).split(",") if v.strip()]
    allowed = {
        "nearest",
        "bilinear",
        "cubic",
        "cubic_pos",
        "global_center_conv",
        "local_nearest_conv",
        "local_bilinear_conv",
        "local_cubic_conv",
        "local_cubic_pos_conv",
    }
    bad = sorted(set(methods) - allowed)
    if bad:
        raise ValueError(f"Unknown methods {bad}; allowed methods are {sorted(allowed)}")
    if not methods:
        raise ValueError("No interpolation methods requested.")
    return methods


def _freq_to_index(freq_mhz: float, freq0_mhz: float, freq_step_mhz: float) -> int:
    return int(round((float(freq_mhz) - float(freq0_mhz)) / float(freq_step_mhz)))


def _k_to_jy_per_pixel(freq_mhz: float, pixel_arcsec: float) -> float:
    k_b = 1.380649e-23
    c = 299792458.0
    pix_rad = (float(pixel_arcsec) / 3600.0) * (math.pi / 180.0)
    omega_pix = pix_rad * pix_rad
    nu_hz = float(freq_mhz) * 1e6
    return float(2.0 * k_b * nu_hz * nu_hz / (c * c) * omega_pix / 1e-26)


def _fmt(pattern: str, *, freq: float, label: str | None = None) -> Path:
    kwargs: Dict[str, Any] = {"freq": float(freq)}
    if label is not None:
        kwargs["label"] = label
    return Path(str(pattern).format(**kwargs))


def _default_response_pattern(root: Path) -> str:
    return str(root / "freq_{freq:.2f}" / "image_natural" / "image_{label}" / "{label}_{freq:.2f}_isobeam_32t-dirty.fits")


def _default_grid_csv_pattern(root: Path) -> str:
    return str(root / "freq_{freq:.2f}" / "patch_fov512_grid33_unit_fluxes.csv")


def _load_grid_csv(path: Path) -> Tuple[List[GridRow], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    if not raw_rows:
        raise ValueError(f"Empty grid CSV: {path}")
    x_key = "pixel_x" if "pixel_x" in raw_rows[0] else "x"
    y_key = "pixel_y" if "pixel_y" in raw_rows[0] else "y"
    x_values = np.asarray(sorted({float(row[x_key]) for row in raw_rows}), dtype=np.float64)
    y_values = np.asarray(sorted({float(row[y_key]) for row in raw_rows}), dtype=np.float64)
    rows: List[GridRow] = []
    for row in raw_rows:
        x = float(row[x_key])
        y = float(row[y_key])
        ix = int(np.argmin(np.abs(x_values - x)))
        iy = int(np.argmin(np.abs(y_values - y)))
        rows.append(GridRow(label=str(row["label"]), x=x, y=y, ix=ix, iy=iy))
    if len(rows) != int(x_values.size * y_values.size):
        raise ValueError(f"Grid is not rectangular: {len(rows)} rows for {x_values.size}x{y_values.size}.")
    rows = sorted(rows, key=lambda r: (r.iy, r.ix))
    return rows, x_values, y_values


def _squeeze_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after squeeze, got {arr.shape}")
    return np.asarray(arr)


def _load_fits_2d(path: Path, dtype: np.dtype) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(_squeeze_2d(hdul[0].data), dtype=dtype)


def _load_cube_slice(path: Path, index: int, dtype: np.dtype) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if np.asarray(data).ndim != 3:
            raise ValueError(f"Expected 3D cube in {path}, got {np.asarray(data).shape}")
        if index < 0 or index >= data.shape[0]:
            raise IndexError(f"Cube index {index} outside {path} shape {data.shape}")
        return np.asarray(data[index], dtype=dtype)


def _central_crop(arr: np.ndarray, size: int) -> np.ndarray:
    if int(size) <= 0:
        return np.asarray(arr)
    h, w = np.asarray(arr).shape[-2:]
    if int(size) > h or int(size) > w:
        raise ValueError(f"Crop size {size} exceeds image shape {(h, w)}")
    y0 = (h - int(size)) // 2
    x0 = (w - int(size)) // 2
    return np.asarray(arr)[y0 : y0 + int(size), x0 : x0 + int(size)]


def _rms(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    den = math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / den) if den > 0.0 else float("nan")


def _axis_nearest_indices(size: int, support: np.ndarray) -> np.ndarray:
    coords = np.arange(int(size), dtype=np.float64)
    return np.argmin(np.abs(coords[:, None] - support[None, :]), axis=1).astype(np.int64)


def _axis_linear_weights(size: int, support: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.arange(int(size), dtype=np.float64)
    hi = np.searchsorted(support, coords, side="right")
    hi = np.clip(hi, 1, support.size - 1)
    lo = hi - 1
    denom = support[hi] - support[lo]
    t = np.divide(coords - support[lo], denom, out=np.zeros_like(coords), where=denom != 0.0)
    outside_lo = coords <= support[0]
    outside_hi = coords >= support[-1]
    lo[outside_lo] = 0
    hi[outside_lo] = 0
    t[outside_lo] = 0.0
    lo[outside_hi] = support.size - 1
    hi[outside_hi] = support.size - 1
    t[outside_hi] = 0.0
    return lo.astype(np.int64), hi.astype(np.int64), (1.0 - t), t


def _catmull_rom_weights(t: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float64)
    t2 = tt * tt
    t3 = t2 * tt
    return np.stack(
        (
            -0.5 * tt + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * tt + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ),
        axis=0,
    )


def _axis_cubic_weights(size: int, support: np.ndarray, positive: bool) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.arange(int(size), dtype=np.float64)
    hi = np.searchsorted(support, coords, side="right")
    hi = np.clip(hi, 1, support.size - 1)
    lo = hi - 1
    denom = support[hi] - support[lo]
    t = np.divide(coords - support[lo], denom, out=np.zeros_like(coords), where=denom != 0.0)
    idx = np.stack((lo - 1, lo, lo + 1, lo + 2), axis=0)
    idx = np.clip(idx, 0, support.size - 1).astype(np.int64)
    w = _catmull_rom_weights(t)
    outside_lo = coords <= support[0]
    outside_hi = coords >= support[-1]
    w[:, outside_lo] = 0.0
    w[1, outside_lo] = 1.0
    idx[:, outside_lo] = 0
    w[:, outside_hi] = 0.0
    w[1, outside_hi] = 1.0
    idx[:, outside_hi] = support.size - 1
    if positive:
        w = np.maximum(w, 0.0)
    norm = np.sum(w, axis=0)
    w = np.divide(w, norm[None, :], out=np.zeros_like(w), where=norm[None, :] != 0.0)
    return idx, w


def _support_fluxes(flux: np.ndarray, x_support: np.ndarray, y_support: np.ndarray, method: str) -> np.ndarray:
    image_size = int(flux.shape[0])
    if flux.shape[0] != flux.shape[1]:
        raise ValueError(f"Expected square flux image, got {flux.shape}")
    out = np.zeros((y_support.size, x_support.size), dtype=np.float64)
    if method == "nearest":
        ix = _axis_nearest_indices(image_size, x_support)
        iy = _axis_nearest_indices(image_size, y_support)
        yy, xx = np.indices(flux.shape)
        np.add.at(out, (iy[yy].reshape(-1), ix[xx].reshape(-1)), np.asarray(flux, dtype=np.float64).reshape(-1))
        return out
    if method == "bilinear":
        x0, x1, wx0, wx1 = _axis_linear_weights(image_size, x_support)
        y0, y1, wy0, wy1 = _axis_linear_weights(image_size, y_support)
        yy, xx = np.indices(flux.shape)
        vals = np.asarray(flux, dtype=np.float64)
        combos = (
            (y0, x0, wy0, wx0),
            (y0, x1, wy0, wx1),
            (y1, x0, wy1, wx0),
            (y1, x1, wy1, wx1),
        )
        for y_idx, x_idx, wy, wx in combos:
            weight = wy[yy] * wx[xx]
            np.add.at(out, (y_idx[yy].reshape(-1), x_idx[xx].reshape(-1)), (vals * weight).reshape(-1))
        return out
    if method in {"cubic", "cubic_pos"}:
        positive = method == "cubic_pos"
        x_idx, wx = _axis_cubic_weights(image_size, x_support, positive=positive)
        y_idx, wy = _axis_cubic_weights(image_size, y_support, positive=positive)
        vals = np.asarray(flux, dtype=np.float64)
        for ky in range(4):
            for kx in range(4):
                weight = wy[ky][:, None] * wx[kx][None, :]
                yy = np.repeat(y_idx[ky][:, None], image_size, axis=1)
                xx = np.repeat(x_idx[kx][None, :], image_size, axis=0)
                np.add.at(out, (yy.reshape(-1), xx.reshape(-1)), (vals * weight).reshape(-1))
        return out
    raise ValueError(method)


def _predict_from_support(
    *,
    support_flux: np.ndarray,
    rows: Sequence[GridRow],
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
) -> np.ndarray:
    pred: np.ndarray | None = None
    for row in rows:
        coeff = float(support_flux[row.iy, row.ix])
        if coeff == 0.0:
            continue
        response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
        if pred is None:
            pred = np.zeros_like(response, dtype=np.float64)
        pred += coeff * np.asarray(response, dtype=np.float64)
    if pred is None:
        raise ValueError("All support coefficients are zero.")
    return pred


def _shift_zero(arr: np.ndarray, *, dy: int, dx: int) -> np.ndarray:
    src = np.asarray(arr)
    out = np.zeros_like(src)
    sy0 = max(0, -int(dy))
    sy1 = min(src.shape[0], src.shape[0] - int(dy))
    sx0 = max(0, -int(dx))
    sx1 = min(src.shape[1], src.shape[1] - int(dx))
    dy0 = max(0, int(dy))
    dx0 = max(0, int(dx))
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)
    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]
    return out


def _shift_response(
    arr: np.ndarray,
    *,
    dy: float,
    dx: float,
    method: str,
) -> np.ndarray:
    if method == "integer":
        return _shift_zero(arr, dy=int(round(float(dy))), dx=int(round(float(dx))))
    order = 1 if method == "linear" else 3
    return np.asarray(
        ndimage.shift(
            np.asarray(arr),
            shift=(float(dy), float(dx)),
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=(order > 1),
        ),
        dtype=np.asarray(arr).dtype,
    )


def _fftconvolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve  # type: ignore
    except Exception as exc:  # pragma: no cover - remote diagnostic dependency
        raise RuntimeError("local convolution methods require scipy.signal.fftconvolve") from exc
    return np.asarray(fftconvolve(np.asarray(image, dtype=np.float64), np.asarray(kernel, dtype=np.float64), mode="same"), dtype=np.float64)


def _fftconvolve_crop_at_origin(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    origin_y: float,
    origin_x: float,
    image_size: int,
) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve  # type: ignore
    except Exception as exc:  # pragma: no cover - remote diagnostic dependency
        raise RuntimeError("local convolution methods require scipy.signal.fftconvolve") from exc
    full = fftconvolve(np.asarray(image, dtype=np.float64), np.asarray(kernel, dtype=np.float64), mode="full")
    y0 = int(round(float(origin_y)))
    x0 = int(round(float(origin_x)))
    return np.asarray(full[y0 : y0 + int(image_size), x0 : x0 + int(image_size)], dtype=np.float64)


def _align_response_kernel(
    response: np.ndarray,
    *,
    dy: float,
    dx: float,
    method: str,
    pad_pixels: int,
) -> np.ndarray:
    pad = max(0, int(pad_pixels))
    if pad == 0:
        return _shift_response(response, dy=float(dy), dx=float(dx), method=method)
    arr = np.asarray(response)
    canvas = np.zeros((arr.shape[0] + 2 * pad, arr.shape[1] + 2 * pad), dtype=arr.dtype)
    canvas[pad : pad + arr.shape[0], pad : pad + arr.shape[1]] = arr
    return _shift_response(canvas, dy=float(dy), dx=float(dx), method=method)


def _central_row(rows: Sequence[GridRow], image_size: int) -> GridRow:
    cx = (int(image_size) - 1) / 2.0
    cy = (int(image_size) - 1) / 2.0
    return min(rows, key=lambda r: (float(r.x) - cx) ** 2 + (float(r.y) - cy) ** 2)


def _nearest_cell_indices(size: int, x_support: np.ndarray, y_support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.arange(int(size), dtype=np.float64)
    ix = np.argmin(np.abs(coords[:, None] - x_support[None, :]), axis=1).astype(np.int64)
    iy = np.argmin(np.abs(coords[:, None] - y_support[None, :]), axis=1).astype(np.int64)
    return ix, iy


def _predict_global_center_conv(
    *,
    fg_flux: np.ndarray,
    rows: Sequence[GridRow],
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
    image_size: int,
) -> np.ndarray:
    row = _central_row(rows, int(image_size))
    response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
    # Treat the centre-source response as a direction-independent dirty beam.
    # scipy's same-mode linear convolution uses the kernel centre as origin.
    return _fftconvolve_same(fg_flux, response)


def _predict_local_nearest_conv(
    *,
    fg_flux: np.ndarray,
    rows: Sequence[GridRow],
    x_support: np.ndarray,
    y_support: np.ndarray,
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
    image_size: int,
    shift_method: str,
    pad_pixels: int,
) -> np.ndarray:
    ix_of_x, iy_of_y = _nearest_cell_indices(int(image_size), x_support, y_support)
    yy, xx = np.indices((int(image_size), int(image_size)))
    align_row = _central_row(rows, int(image_size))
    align_x = float(align_row.x)
    align_y = float(align_row.y)
    pred = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    for row in rows:
        mask = (ix_of_x[xx] == int(row.ix)) & (iy_of_y[yy] == int(row.iy))
        if not np.any(mask):
            continue
        cell_flux = np.zeros_like(fg_flux, dtype=np.float64)
        cell_flux[mask] = np.asarray(fg_flux, dtype=np.float64)[mask]
        response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
        kernel = _align_response_kernel(
            response,
            dy=float(align_y - float(row.y)),
            dx=float(align_x - float(row.x)),
            method=shift_method,
            pad_pixels=int(pad_pixels),
        )
        if int(pad_pixels) > 0:
            pred += _fftconvolve_crop_at_origin(
                cell_flux,
                kernel,
                origin_y=float(pad_pixels) + align_y,
                origin_x=float(pad_pixels) + align_x,
                image_size=int(image_size),
            )
        else:
            pred += _fftconvolve_same(cell_flux, kernel)
    return pred


def _axis_linear_weight_table(size: int, support: np.ndarray) -> np.ndarray:
    """Return table W[i, x] for linear interpolation weights on one axis."""
    lo, hi, w_lo, w_hi = _axis_linear_weights(int(size), support)
    table = np.zeros((support.size, int(size)), dtype=np.float64)
    pix = np.arange(int(size), dtype=np.int64)
    np.add.at(table, (lo, pix), w_lo)
    np.add.at(table, (hi, pix), w_hi)
    return table


def _axis_cubic_weight_table(size: int, support: np.ndarray, *, positive: bool) -> np.ndarray:
    """Return table W[i, x] for cubic interpolation weights on one axis."""
    idx, weights = _axis_cubic_weights(int(size), support, positive=bool(positive))
    table = np.zeros((support.size, int(size)), dtype=np.float64)
    pix = np.arange(int(size), dtype=np.int64)
    for k in range(4):
        np.add.at(table, (idx[k], pix), weights[k])
    return table


def _predict_local_bilinear_conv(
    *,
    fg_flux: np.ndarray,
    rows: Sequence[GridRow],
    x_support: np.ndarray,
    y_support: np.ndarray,
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
    image_size: int,
    shift_method: str,
    pad_pixels: int,
) -> np.ndarray:
    wx_table = _axis_linear_weight_table(int(image_size), x_support)
    wy_table = _axis_linear_weight_table(int(image_size), y_support)
    align_row = _central_row(rows, int(image_size))
    align_x = float(align_row.x)
    align_y = float(align_row.y)
    pred = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    base_flux = np.asarray(fg_flux, dtype=np.float64)
    for row in rows:
        wx = wx_table[int(row.ix)]
        wy = wy_table[int(row.iy)]
        if not np.any(wx) or not np.any(wy):
            continue
        weighted_flux = base_flux * wy[:, None] * wx[None, :]
        if not np.any(weighted_flux):
            continue
        response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
        kernel = _align_response_kernel(
            response,
            dy=float(align_y - float(row.y)),
            dx=float(align_x - float(row.x)),
            method=shift_method,
            pad_pixels=int(pad_pixels),
        )
        if int(pad_pixels) > 0:
            pred += _fftconvolve_crop_at_origin(
                weighted_flux,
                kernel,
                origin_y=float(pad_pixels) + align_y,
                origin_x=float(pad_pixels) + align_x,
                image_size=int(image_size),
            )
        else:
            pred += _fftconvolve_same(weighted_flux, kernel)
    return pred


def _predict_local_cubic_conv(
    *,
    fg_flux: np.ndarray,
    rows: Sequence[GridRow],
    x_support: np.ndarray,
    y_support: np.ndarray,
    response_pattern: str,
    freq: float,
    dtype: np.dtype,
    image_size: int,
    shift_method: str,
    pad_pixels: int,
    positive: bool,
) -> np.ndarray:
    wx_table = _axis_cubic_weight_table(int(image_size), x_support, positive=bool(positive))
    wy_table = _axis_cubic_weight_table(int(image_size), y_support, positive=bool(positive))
    align_row = _central_row(rows, int(image_size))
    align_x = float(align_row.x)
    align_y = float(align_row.y)
    pred = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    base_flux = np.asarray(fg_flux, dtype=np.float64)
    for row in rows:
        wx = wx_table[int(row.ix)]
        wy = wy_table[int(row.iy)]
        if not np.any(wx) or not np.any(wy):
            continue
        weighted_flux = base_flux * wy[:, None] * wx[None, :]
        if not np.any(weighted_flux):
            continue
        response = _load_fits_2d(_fmt(response_pattern, freq=freq, label=row.label), dtype=dtype)
        kernel = _align_response_kernel(
            response,
            dy=float(align_y - float(row.y)),
            dx=float(align_x - float(row.x)),
            method=shift_method,
            pad_pixels=int(pad_pixels),
        )
        if int(pad_pixels) > 0:
            pred += _fftconvolve_crop_at_origin(
                weighted_flux,
                kernel,
                origin_y=float(pad_pixels) + align_y,
                origin_x=float(pad_pixels) + align_x,
                image_size=int(image_size),
            )
        else:
            pred += _fftconvolve_same(weighted_flux, kernel)
    return pred


def _metric(target_fg: np.ndarray, pred_fg: np.ndarray, dirty_eor: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(target_fg, dtype=np.float64) - np.asarray(pred_fg, dtype=np.float64)
    return {
        "dirty_fg_rms": _rms(target_fg),
        "pred_fg_rms": _rms(pred_fg),
        "dirty_eor_rms": _rms(dirty_eor),
        "fg_residual_rms": _rms(residual),
        "fg_residual_over_dirty_eor_rms": _rms(residual) / max(_rms(dirty_eor), 1e-300),
        "fg_residual_over_dirty_fg_rms": _rms(residual) / max(_rms(target_fg), 1e-300),
        "pred_vs_dirty_fg_corr": _corr(target_fg, pred_fg),
        "residual_vs_dirty_eor_corr": _corr(residual, dirty_eor),
    }


def _stack_fft_metric(residual_stack: np.ndarray, eor_stack: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(residual_stack, dtype=np.float64)
    eor = np.asarray(eor_stack, dtype=np.float64)
    residual_power = float(np.sum(np.abs(np.fft.fftn(residual)) ** 2))
    eor_power = float(np.sum(np.abs(np.fft.fftn(eor)) ** 2))
    return {
        "cube_fg_residual_rms": _rms(residual),
        "cube_dirty_eor_rms": _rms(eor),
        "cube_fg_residual_over_dirty_eor_rms": _rms(residual) / max(_rms(eor), 1e-300),
        "diagnostic_fft_residual_power": residual_power,
        "diagnostic_fft_dirty_eor_power": eor_power,
        "diagnostic_fft_residual_over_dirty_eor_power": residual_power / max(eor_power, 1e-300),
    }


def _window_power_metric(
    residual_stack: np.ndarray,
    eor_stack: np.ndarray,
    *,
    pixel_arcsec: float,
    freq_step_mhz: float,
    ref_freq_mhz: float,
    freq_grid_start_mhz: float,
    nbins_kperp: int,
    nbins_kpar: int,
    kpar_min: float,
) -> Dict[str, Any]:
    from powerspec import (  # type: ignore
        PowerSpecConfig,
        compute_eor_window_mask,
        compute_power2d_window_metrics,
        compute_power_spectra,
        select_eor_window_bins,
    )

    cfg = PowerSpecConfig(
        dx=float(pixel_arcsec),
        dy=float(pixel_arcsec),
        df=float(freq_step_mhz),
        unit_x="arcsec",
        unit_y="arcsec",
        unit_f="mhz",
        ref_freq_mhz=float(ref_freq_mhz),
        freq_grid_start_mhz=float(freq_grid_start_mhz),
        rest_freq_mhz=1420.40575,
        H0=67.8,
        Om0=0.308,
        Ode0=0.692,
        freq_axis=0,
        nbins_1d=50,
        nbins_kperp=int(nbins_kperp),
        nbins_kpar=int(nbins_kpar),
        stat_mode="median",
        log_bins_2d=False,
        demean_mode="global",
        eor_window_enabled=True,
        eor_window_kpar_min=float(kpar_min),
        eor_window_wedge_slope=0.0,
        eor_window_wedge_intercept=0.0,
        eor_window_exclude_dc=False,
        eor_window_bin_policy="all_modes",
        eor_window_eps=1.0e-20,
    )
    leak_ps = compute_power_spectra(np.asarray(residual_stack, dtype=np.float64), cfg, window="hann")
    eor_ps = compute_power_spectra(np.asarray(eor_stack, dtype=np.float64), cfg, window="hann")
    center_mask = compute_eor_window_mask(
        eor_ps["kperp_centers"], eor_ps["kpar_centers"], cfg
    )
    mask = select_eor_window_bins(
        center_mask,
        eor_ps["eor_window_mode_fractions"]["default"],
        cfg.eor_window_bin_policy,
    )
    metrics = compute_power2d_window_metrics(leak_ps["p2d"], eor_ps["p2d"], mask, eps=float(cfg.eor_window_eps))
    return {
        "enabled": True,
        "ref_freq_mhz": float(ref_freq_mhz),
        "pixel_arcsec": float(pixel_arcsec),
        "freq_step_mhz": float(freq_step_mhz),
        "kpar_min": float(kpar_min),
        "nbins_kperp": int(nbins_kperp),
        "nbins_kpar": int(nbins_kpar),
        "window_n_bins_bool": int(np.sum(mask)),
        "window_metrics": metrics,
    }


def _write_image(path: Path, data: np.ndarray, template: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.getheader(template).copy()
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.float64 if args.dtype == "float64" else np.float32)
    freqs = _parse_floats(args.freqs_mhz)
    methods = _parse_methods(args.methods)
    crop_sizes = _parse_ints(args.eval_crop_sizes)
    response_pattern = args.response_pattern or _default_response_pattern(args.response_root)
    grid_csv_pattern = args.grid_csv_pattern or _default_grid_csv_pattern(args.response_root)
    truth_fg_pattern = args.truth_fg_pattern or str(args.truth_root / "image_natural" / "fg_{freq:.2f}-dirty.fits")
    truth_eor_pattern = args.truth_eor_pattern or str(args.truth_root / "image_natural" / "eor_{freq:.2f}-dirty.fits")

    per_freq: List[Dict[str, Any]] = []
    stacks: Dict[str, Dict[int, Dict[str, List[np.ndarray]]]] = {
        method: {crop: {"residual": [], "eor": []} for crop in crop_sizes} for method in methods
    }

    for freq in freqs:
        grid_csv = _fmt(grid_csv_pattern, freq=freq)
        rows, x_support, y_support = _load_grid_csv(grid_csv)
        freq_index = _freq_to_index(freq, float(args.freq0_mhz), float(args.freq_step_mhz))
        fg_k = _load_cube_slice(args.fg_cube_k, freq_index, dtype=np.float64)
        if fg_k.shape != (int(args.image_size), int(args.image_size)):
            raise ValueError(f"FG slice shape {fg_k.shape} does not match image_size={args.image_size}.")
        fg_flux = fg_k * _k_to_jy_per_pixel(freq, float(args.pixel_arcsec))
        truth_fg_path = _fmt(truth_fg_pattern, freq=freq)
        truth_eor_path = _fmt(truth_eor_pattern, freq=freq)
        dirty_fg = _load_fits_2d(truth_fg_path, dtype=dtype)
        dirty_eor = _load_fits_2d(truth_eor_path, dtype=dtype)
        freq_result: Dict[str, Any] = {
            "freq_mhz": float(freq),
            "freq_index": int(freq_index),
            "grid_csv": str(grid_csv),
            "truth_fg": str(truth_fg_path),
            "truth_eor": str(truth_eor_path),
            "n_support": int(len(rows)),
            "support_x_minmax": [float(x_support[0]), float(x_support[-1])],
            "support_y_minmax": [float(y_support[0]), float(y_support[-1])],
            "methods": {},
        }
        for method in methods:
            support_flux: np.ndarray | None = None
            if method in {"nearest", "bilinear", "cubic", "cubic_pos"}:
                support_flux = _support_fluxes(fg_flux, x_support, y_support, method)
                pred_fg = _predict_from_support(
                    support_flux=support_flux,
                    rows=rows,
                    response_pattern=response_pattern,
                    freq=float(freq),
                    dtype=dtype,
                )
            elif method == "global_center_conv":
                pred_fg = _predict_global_center_conv(
                    fg_flux=fg_flux,
                    rows=rows,
                    response_pattern=response_pattern,
                    freq=float(freq),
                    dtype=dtype,
                    image_size=int(args.image_size),
                )
            elif method == "local_nearest_conv":
                pred_fg = _predict_local_nearest_conv(
                    fg_flux=fg_flux,
                    rows=rows,
                    x_support=x_support,
                    y_support=y_support,
                    response_pattern=response_pattern,
                    freq=float(freq),
                    dtype=dtype,
                    image_size=int(args.image_size),
                    shift_method=str(args.local_shift_method),
                    pad_pixels=int(args.local_pad_pixels),
                )
            elif method == "local_bilinear_conv":
                pred_fg = _predict_local_bilinear_conv(
                    fg_flux=fg_flux,
                    rows=rows,
                    x_support=x_support,
                    y_support=y_support,
                    response_pattern=response_pattern,
                    freq=float(freq),
                    dtype=dtype,
                    image_size=int(args.image_size),
                    shift_method=str(args.local_shift_method),
                    pad_pixels=int(args.local_pad_pixels),
                )
            elif method in {"local_cubic_conv", "local_cubic_pos_conv"}:
                pred_fg = _predict_local_cubic_conv(
                    fg_flux=fg_flux,
                    rows=rows,
                    x_support=x_support,
                    y_support=y_support,
                    response_pattern=response_pattern,
                    freq=float(freq),
                    dtype=dtype,
                    image_size=int(args.image_size),
                    shift_method=str(args.local_shift_method),
                    pad_pixels=int(args.local_pad_pixels),
                    positive=(method == "local_cubic_pos_conv"),
                )
            else:  # pragma: no cover - guarded by _parse_methods()
                raise ValueError(method)
            residual = np.asarray(dirty_fg, dtype=np.float64) - np.asarray(pred_fg, dtype=np.float64)
            method_result: Dict[str, Any] = {
                "support_flux_sum_jy": None if support_flux is None else float(np.sum(support_flux, dtype=np.float64)),
                "input_flux_sum_jy": float(np.sum(fg_flux, dtype=np.float64)),
                "support_flux_abs_sum_jy": None if support_flux is None else float(np.sum(np.abs(support_flux), dtype=np.float64)),
                "input_flux_abs_sum_jy": float(np.sum(np.abs(fg_flux), dtype=np.float64)),
                "crops": {},
            }
            for crop in crop_sizes:
                fg_c = _central_crop(dirty_fg, crop)
                pred_c = _central_crop(pred_fg, crop)
                eor_c = _central_crop(dirty_eor, crop)
                res_c = fg_c - pred_c
                method_result["crops"][str(int(crop))] = _metric(fg_c, pred_c, eor_c)
                stacks[method][int(crop)]["residual"].append(np.asarray(res_c, dtype=np.float32))
                stacks[method][int(crop)]["eor"].append(np.asarray(eor_c, dtype=np.float32))
            if args.save_products:
                stem = f"{method}_{freq:.2f}"
                _write_image(out_dir / f"{stem}_pred_fg-dirty.fits", pred_fg, truth_fg_path)
                _write_image(out_dir / f"{stem}_fg_residual-dirty.fits", residual, truth_fg_path)
            freq_result["methods"][method] = method_result
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

    stack_results: Dict[str, Dict[str, Any]] = {}
    ref_freq_mhz = float(args.power_ref_freq_mhz) if float(args.power_ref_freq_mhz) > 0.0 else float(np.median(freqs))
    for method in methods:
        stack_results[method] = {}
        for crop in crop_sizes:
            residual_stack = np.stack(stacks[method][int(crop)]["residual"], axis=0)
            eor_stack = np.stack(stacks[method][int(crop)]["eor"], axis=0)
            crop_result: Dict[str, Any] = _stack_fft_metric(residual_stack, eor_stack)
            if args.power_window and len(freqs) >= 4:
                crop_result["eor_window_ps2d"] = _window_power_metric(
                    residual_stack,
                    eor_stack,
                    pixel_arcsec=float(args.pixel_arcsec),
                    freq_step_mhz=float(args.freq_step_mhz),
                    ref_freq_mhz=ref_freq_mhz,
                    freq_grid_start_mhz=float(freqs[0]),
                    nbins_kperp=int(args.power_nbins_kperp),
                    nbins_kpar=int(args.power_nbins_kpar),
                    kpar_min=float(args.power_kpar_min),
                )
            elif args.power_window:
                crop_result["eor_window_ps2d"] = {
                    "enabled": False,
                    "reason": "At least four frequency channels are required for this diagnostic.",
                }
            stack_results[method][str(int(crop))] = crop_result

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
            "methods": methods,
            "eval_crop_sizes": [int(v) for v in crop_sizes],
            "response_pattern": response_pattern,
            "grid_csv_pattern": grid_csv_pattern,
            "truth_fg_pattern": truth_fg_pattern,
            "truth_eor_pattern": truth_eor_pattern,
            "dtype": str(dtype),
            "local_shift_method": str(args.local_shift_method),
            "local_pad_pixels": int(args.local_pad_pixels),
            "power_window": bool(args.power_window),
            "power_ref_freq_mhz": ref_freq_mhz,
            "power_kpar_min": float(args.power_kpar_min),
        },
        "per_frequency": per_freq,
        "stack_results": stack_results,
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
