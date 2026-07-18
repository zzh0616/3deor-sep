#!/usr/bin/env python3
"""Evaluate tiled local-hybrid foreground subtraction from one edge bank.

This is stricter than ``evaluate_existing_edge_tiled_local_hybrid.py``.  The
older diagnostic reconstructs a dense response bank at stride-2 support points.
This script uses only the supplied train response bank to predict the dirty
foreground contribution from every integer sky pixel, then compares the central
dirty-image crop against the direct dirty foreground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_planA_real_wsclean_dirty_base import _crop2d, _load_wsclean_2d  # noqa: E402
from evaluate_fullsky_response_interpolation import (  # noqa: E402
    _central_crop,
    _corr,
    _fmt,
    _freq_to_index,
    _k_to_jy_per_pixel,
    _load_cube_slice,
    _load_fits_2d,
    _rms,
    _write_image,
)
from evaluate_sparse_response_interpolation_holdout import _load_grid_csv, _shift_zero  # noqa: E402


@dataclass(frozen=True)
class TileSpec:
    name: str
    x0: int
    x1: int
    y0: int
    y1: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--train-grid-csv", type=Path, required=True)
    ap.add_argument("--train-response-pattern", required=True)
    ap.add_argument("--truth-root", type=Path, required=True)
    ap.add_argument("--truth-fg-pattern", required=True)
    ap.add_argument("--truth-eor-pattern", required=True)
    ap.add_argument("--fg-cube-k", type=Path, required=True)
    ap.add_argument("--freq-mhz", type=float, default=117.90)
    ap.add_argument("--freq0-mhz", type=float, default=106.0)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument("--pixel-arcsec", type=float, default=32.0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--response-crop-size", type=int, default=512)
    ap.add_argument("--eval-crop-size", type=int, default=256)
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument(
        "--train-halo-px",
        type=int,
        default=0,
        help="Extra support-response halo around each source tile. Target foreground pixels remain tile-local.",
    )
    ap.add_argument("--model-margin", type=int, default=-1)
    ap.add_argument("--ranks", default="32")
    ap.add_argument("--rbf-scales-px", default="32")
    ap.add_argument("--modes", default="interp,train_hybrid")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--progress-every-tile", type=int, default=1)
    ap.add_argument("--save-products", action="store_true")
    return ap.parse_args()


def _parse_ints(spec: str) -> List[int]:
    return [int(v) for v in str(spec).split(",") if str(v).strip()]


def _parse_floats(spec: str) -> List[float]:
    return [float(v) for v in str(spec).split(",") if str(v).strip()]


def _parse_modes(spec: str) -> List[str]:
    modes = [v.strip() for v in str(spec).split(",") if v.strip()]
    allowed = {"interp", "train_hybrid"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise ValueError(f"Unsupported modes {bad}; allowed={sorted(allowed)}")
    if not modes:
        raise ValueError("No modes selected")
    return modes


def _make_tiles(image_size: int, tile_size: int) -> List[TileSpec]:
    tiles: List[TileSpec] = []
    for y0 in range(0, int(image_size), int(tile_size)):
        y1 = min(y0 + int(tile_size) - 1, int(image_size) - 1)
        for x0 in range(0, int(image_size), int(tile_size)):
            x1 = min(x0 + int(tile_size) - 1, int(image_size) - 1)
            tiles.append(TileSpec(f"t{x0:03d}_{y0:03d}", x0, x1, y0, y1))
    return tiles


def _row_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(round(float(row["x"]))), int(round(float(row["y"])))


def _select_train_rows(
    rows: Sequence[Dict[str, Any]],
    tile: TileSpec,
    *,
    halo_px: int,
    image_size: int,
) -> List[Dict[str, Any]]:
    halo = max(0, int(halo_px))
    x0 = max(0, int(tile.x0) - halo)
    x1 = min(int(image_size) - 1, int(tile.x1) + halo)
    y0 = max(0, int(tile.y0) - halo)
    y1 = min(int(image_size) - 1, int(tile.y1) + halo)
    return [
        row
        for row in rows
        if x0 <= int(round(float(row["x"]))) <= x1
        and y0 <= int(round(float(row["y"]))) <= y1
    ]


def _inside_tile_key(row: Dict[str, Any], tile: TileSpec) -> bool:
    x, y = _row_key(row)
    return int(tile.x0) <= x <= int(tile.x1) and int(tile.y0) <= y <= int(tile.y1)


def _load_train_aligned_and_exact(
    *,
    rows: Sequence[Dict[str, Any]],
    pattern: str,
    freq_mhz: float,
    response_crop_size: int,
    model_size: int,
    eval_size: int,
    dtype: np.dtype,
    align_x: float,
    align_y: float,
    fg_flux: np.ndarray,
    exact_keys: set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    aligned: List[np.ndarray] = []
    exact_pred = np.zeros((int(eval_size), int(eval_size)), dtype=np.float64)
    for row in rows:
        arr, _ = _load_wsclean_2d(_fmt(pattern, label=str(row["label"]), freq=float(freq_mhz)), dtype)
        cropped = _crop2d(arr, size=int(response_crop_size), center_x=None, center_y=None)
        shifted = _shift_zero(
            cropped,
            dx=int(round(float(align_x) - float(row["x"]))),
            dy=int(round(float(align_y) - float(row["y"]))),
        )
        aligned.append(np.asarray(_central_crop(shifted, int(model_size)), dtype=dtype).reshape(-1))
        x, y = _row_key(row)
        if (x, y) in exact_keys:
            exact_pred += float(fg_flux[y, x]) * np.asarray(_central_crop(cropped, int(eval_size)), dtype=np.float64)
    return np.stack(aligned, axis=0), exact_pred


def _pca_basis(train_aligned: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_dtype = np.asarray(train_aligned).dtype
    x = np.asarray(train_aligned, dtype=np.float64)
    mean = np.mean(x, axis=0)
    centered = x - mean[np.newaxis, :]
    gram = centered @ centered.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    keep = min(int(rank), int(np.sum(vals > 0.0)), centered.shape[0])
    basis: List[np.ndarray] = []
    for i in range(keep):
        basis.append((vecs[:, i] @ centered) / np.sqrt(vals[i]))
    basis_arr = np.asarray(basis, dtype=out_dtype)
    coeff_train = centered @ basis_arr.T
    return mean.astype(out_dtype), basis_arr, coeff_train.astype(out_dtype)


def _rbf_coefficients_many(
    *,
    target_xy: np.ndarray,
    train_xy: np.ndarray,
    coeff_train: np.ndarray,
    scale_px: float,
    ridge: float = 1e-10,
) -> np.ndarray:
    train = np.asarray(train_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    dx = train[:, None, 0] - train[None, :, 0]
    dy = train[:, None, 1] - train[None, :, 1]
    q = np.sqrt(dx * dx + dy * dy) / float(scale_px)
    mat = np.sqrt(1.0 + q * q)
    mat = mat + float(ridge) * np.eye(mat.shape[0], dtype=np.float64)
    rhs_dx = train[:, None, 0] - target[None, :, 0]
    rhs_dy = train[:, None, 1] - target[None, :, 1]
    rhs_q = np.sqrt(rhs_dx * rhs_dx + rhs_dy * rhs_dy) / float(scale_px)
    rhs = np.sqrt(1.0 + rhs_q * rhs_q)
    try:
        weights_t = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        weights_t = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    coeff = weights_t.T @ np.asarray(coeff_train, dtype=np.float64)
    by_key = {(int(round(x)), int(round(y))): i for i, (x, y) in enumerate(train)}
    for i, (x, y) in enumerate(target):
        src = by_key.get((int(round(float(x))), int(round(float(y)))))
        if src is not None:
            coeff[i] = np.asarray(coeff_train[src], dtype=np.float64)
    return coeff


def _tile_pixels(tile: TileSpec) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[int] = []
    ys: List[int] = []
    for y in range(tile.y0, tile.y1 + 1):
        for x in range(tile.x0, tile.x1 + 1):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.int16), np.asarray(ys, dtype=np.int16)


def _component_weight_maps(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    fluxes: np.ndarray,
    coeff: np.ndarray,
    exact_mask: np.ndarray,
    align_x: float,
    align_y: float,
    model_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_comp = int(coeff.shape[1])
    mean_map = np.zeros((int(model_size), int(model_size)), dtype=np.float64)
    coeff_maps = np.zeros((n_comp, int(model_size), int(model_size)), dtype=np.float64)
    center = int(model_size) // 2
    for i in range(int(xs.size)):
        if bool(exact_mask[i]):
            continue
        ox = center + int(round(float(xs[i]) - float(align_x)))
        oy = center + int(round(float(ys[i]) - float(align_y)))
        if 0 <= ox < int(model_size) and 0 <= oy < int(model_size):
            f = float(fluxes[i])
            mean_map[oy, ox] += f
            coeff_maps[:, oy, ox] += f * np.asarray(coeff[i], dtype=np.float64)
    return mean_map, coeff_maps


def _accumulate_components(
    *,
    mean: np.ndarray,
    basis: np.ndarray,
    mean_map: np.ndarray,
    coeff_maps: np.ndarray,
    model_size: int,
    eval_size: int,
) -> np.ndarray:
    pred_full = signal.fftconvolve(
        mean_map,
        np.asarray(mean, dtype=np.float64).reshape(int(model_size), int(model_size)),
        mode="full",
    )
    for k in range(int(basis.shape[0])):
        pred_full += signal.fftconvolve(
            coeff_maps[k],
            np.asarray(basis[k], dtype=np.float64).reshape(int(model_size), int(model_size)),
            mode="full",
        )
    start = int(model_size) // 2 + (int(model_size) - int(eval_size)) // 2
    return np.asarray(pred_full[start : start + int(eval_size), start : start + int(eval_size)], dtype=np.float64)


def _metrics(dirty_fg: np.ndarray, pred_fg: np.ndarray, dirty_eor: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(dirty_fg, dtype=np.float64) - np.asarray(pred_fg, dtype=np.float64)
    return {
        "dirty_fg_rms": _rms(dirty_fg),
        "pred_fg_rms": _rms(pred_fg),
        "dirty_eor_rms": _rms(dirty_eor),
        "fg_residual_rms": _rms(residual),
        "fg_residual_over_dirty_fg_rms": _rms(residual) / max(_rms(dirty_fg), 1e-300),
        "fg_residual_over_dirty_eor_rms": _rms(residual) / max(_rms(dirty_eor), 1e-300),
        "pred_vs_dirty_fg_corr": _corr(pred_fg, dirty_fg),
        "residual_plus_dirty_eor_corr_dirty_eor": _corr(residual + dirty_eor, dirty_eor),
    }


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir if args.out_dir is not None else args.out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.float64 if args.dtype == "float64" else np.float32)
    ranks = sorted(set(_parse_ints(args.ranks)))
    scales = sorted(set(_parse_floats(args.rbf_scales_px)))
    modes = _parse_modes(args.modes)

    rows_all = _load_grid_csv(args.train_grid_csv)
    freq_index = _freq_to_index(float(args.freq_mhz), float(args.freq0_mhz), float(args.freq_step_mhz))
    fg_k = _load_cube_slice(args.fg_cube_k, int(freq_index), dtype=np.float64)
    fg_flux = fg_k * _k_to_jy_per_pixel(float(args.freq_mhz), float(args.pixel_arcsec))
    truth_fg_path = _fmt(args.truth_fg_pattern, freq=float(args.freq_mhz))
    truth_eor_path = _fmt(args.truth_eor_pattern, freq=float(args.freq_mhz))
    dirty_fg = _central_crop(_load_fits_2d(truth_fg_path, dtype=dtype), int(args.eval_crop_size))
    dirty_eor = _central_crop(_load_fits_2d(truth_eor_path, dtype=dtype), int(args.eval_crop_size))

    pred_by_config: Dict[str, np.ndarray] = {}
    tile_summaries: List[Dict[str, Any]] = []
    tiles = _make_tiles(int(args.image_size), int(args.tile_size))

    for tile_idx, tile in enumerate(tiles, start=1):
        train_rows = _select_train_rows(
            rows_all,
            tile,
            halo_px=int(args.train_halo_px),
            image_size=int(args.image_size),
        )
        if len(train_rows) < 4:
            raise ValueError(f"Tile {tile.name} selected too few train rows: {len(train_rows)}")
        xs, ys = _tile_pixels(tile)
        target_xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        train_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in train_rows], dtype=np.float64)
        align_x = float(np.median(xs.astype(np.float64)))
        align_y = float(np.median(ys.astype(np.float64)))
        max_target_offset = max(
            float(np.max(np.abs(xs.astype(np.float64) - align_x))),
            float(np.max(np.abs(ys.astype(np.float64) - align_y))),
        )
        max_train_offset = max(
            float(np.max(np.abs(train_xy[:, 0] - align_x))),
            float(np.max(np.abs(train_xy[:, 1] - align_y))),
        )
        max_offset = int(math.ceil(max(max_target_offset, max_train_offset)))
        model_margin = int(args.model_margin) if int(args.model_margin) >= 0 else max_offset + 8
        model_size = int(args.eval_crop_size) + 2 * int(model_margin)
        if model_size > int(args.response_crop_size):
            raise ValueError(f"Tile {tile.name}: model_size={model_size} > response_crop_size={args.response_crop_size}")

        print(
            json.dumps(
                {
                    "event": "tile_start",
                    "tile": tile.name,
                    "tile_index": int(tile_idx),
                    "tiles_total": int(len(tiles)),
                    "n_pixels": int(xs.size),
                    "n_train": int(len(train_rows)),
                    "model_size": int(model_size),
                    "time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                sort_keys=True,
            ),
            flush=True,
        )

        exact_tile_keys = {_row_key(row) for row in train_rows if _inside_tile_key(row, tile)}
        train_aligned, exact_train_pred = _load_train_aligned_and_exact(
            rows=train_rows,
            pattern=str(args.train_response_pattern),
            freq_mhz=float(args.freq_mhz),
            response_crop_size=int(args.response_crop_size),
            model_size=int(model_size),
            eval_size=int(args.eval_crop_size),
            dtype=dtype,
            align_x=align_x,
            align_y=align_y,
            fg_flux=fg_flux,
            exact_keys=exact_tile_keys,
        )
        max_rank = min(max(ranks), max(1, len(train_rows) - 1))
        mean, basis_max, coeff_train_max = _pca_basis(train_aligned, int(max_rank))
        fluxes = fg_flux[ys.astype(np.int64), xs.astype(np.int64)]
        train_mask = np.asarray([(int(x), int(y)) in exact_tile_keys for x, y in zip(xs, ys)], dtype=bool)

        for scale in scales:
            coeff_interp_max = _rbf_coefficients_many(
                target_xy=target_xy,
                train_xy=train_xy,
                coeff_train=coeff_train_max,
                scale_px=float(scale),
            )
            for rank in ranks:
                k = min(int(rank), int(basis_max.shape[0]))
                for mode in modes:
                    exact_mask = train_mask if mode == "train_hybrid" else np.zeros(xs.size, dtype=bool)
                    mean_map, coeff_maps = _component_weight_maps(
                        xs=xs,
                        ys=ys,
                        fluxes=fluxes,
                        coeff=coeff_interp_max[:, :k],
                        exact_mask=exact_mask,
                        align_x=align_x,
                        align_y=align_y,
                        model_size=int(model_size),
                    )
                    pred_tile = _accumulate_components(
                        mean=mean,
                        basis=basis_max[:k],
                        mean_map=mean_map,
                        coeff_maps=coeff_maps,
                        model_size=int(model_size),
                        eval_size=int(args.eval_crop_size),
                    )
                    if mode == "train_hybrid":
                        pred_tile = pred_tile + exact_train_pred
                    label = f"scale{scale:g}_rank{k}_{mode}"
                    pred_by_config.setdefault(
                        label,
                        np.zeros((int(args.eval_crop_size), int(args.eval_crop_size)), dtype=np.float64),
                    )
                    pred_by_config[label] += pred_tile

        tile_summaries.append(
            {
                "name": tile.name,
                "bounds": {"x0": tile.x0, "x1": tile.x1, "y0": tile.y0, "y1": tile.y1},
                "n_pixels": int(xs.size),
                "n_train": int(len(train_rows)),
                "train_fraction": float(np.mean(train_mask)),
                "train_halo_px": int(args.train_halo_px),
                "align_x": float(align_x),
                "align_y": float(align_y),
                "model_size": int(model_size),
                "model_margin": int(model_margin),
            }
        )
        if int(args.progress_every_tile) > 0 and (
            tile_idx == 1 or tile_idx % int(args.progress_every_tile) == 0 or tile_idx == len(tiles)
        ):
            best_so_far = []
            for label, pred in pred_by_config.items():
                best_so_far.append((label, _metrics(dirty_fg, pred, dirty_eor)["fg_residual_over_dirty_eor_rms"]))
            best_so_far.sort(key=lambda item: item[1])
            print(
                json.dumps(
                    {
                        "event": "tile_done",
                        "tile": tile.name,
                        "tile_index": int(tile_idx),
                        "best_so_far": best_so_far[:3],
                        "time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    configs: Dict[str, Any] = {}
    best: List[Dict[str, Any]] = []
    for label, pred in sorted(pred_by_config.items()):
        metrics = _metrics(dirty_fg, pred, dirty_eor)
        configs[label] = metrics
        best.append({"config": label, "fg_residual_over_dirty_eor_rms": metrics["fg_residual_over_dirty_eor_rms"]})
        if args.save_products:
            residual = np.asarray(dirty_fg, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
            _write_image(out_dir / f"{label}_pred_fg-dirty.fits", pred, truth_fg_path)
            _write_image(out_dir / f"{label}_fg_residual-dirty.fits", residual, truth_fg_path)
    best.sort(key=lambda item: float(item["fg_residual_over_dirty_eor_rms"]))

    output = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "settings": {
            "train_grid_csv": str(args.train_grid_csv),
            "train_response_pattern": str(args.train_response_pattern),
            "truth_root": str(args.truth_root),
            "truth_fg": str(truth_fg_path),
            "truth_eor": str(truth_eor_path),
            "fg_cube_k": str(args.fg_cube_k),
            "freq_mhz": float(args.freq_mhz),
            "freq_index": int(freq_index),
            "pixel_arcsec": float(args.pixel_arcsec),
            "image_size": int(args.image_size),
            "response_crop_size": int(args.response_crop_size),
            "eval_crop_size": int(args.eval_crop_size),
            "tile_size": int(args.tile_size),
            "train_halo_px": int(args.train_halo_px),
            "ranks": [int(v) for v in ranks],
            "rbf_scales_px": [float(v) for v in scales],
            "modes": list(modes),
            "dtype": str(dtype),
        },
        "tiles": tile_summaries,
        "configs": configs,
        "best": best,
    }
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
