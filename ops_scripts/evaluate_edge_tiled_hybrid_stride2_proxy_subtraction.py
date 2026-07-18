#!/usr/bin/env python3
"""Evaluate foreground subtraction using a hybrid-reconstructed stride-2 bank.

The current proven single-frequency route is exact ``stride2edge`` responses
plus local bilinear source interpolation.  This evaluator tests whether a
coarser train bank can replace the exact stride-2 response generation by
locally reconstructing the stride-2 support responses, then using the same
stride-2 local-bilinear foreground assembly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_planA_real_wsclean_dirty_base import _crop2d, _load_wsclean_2d  # noqa: E402
from evaluate_edge_tiled_hybrid_fullsky_subtraction import (  # noqa: E402
    _accumulate_components,
    _make_tiles,
    _pca_basis,
    _rbf_coefficients_many,
    _row_key,
    _select_train_rows,
)
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
from evaluate_fullsky_stride2_integer_shift import _axis_linear_contribs  # noqa: E402
from evaluate_sparse_response_interpolation_holdout import (  # noqa: E402
    _load_grid_csv,
    _shift_zero,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dense-grid-csv", type=Path, required=True)
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
    ap.add_argument("--train-halo-px", type=int, default=16)
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


def _assign_grid_indices(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x_support = np.asarray(sorted({float(row["x"]) for row in rows}), dtype=np.float64)
    y_support = np.asarray(sorted({float(row["y"]) for row in rows}), dtype=np.float64)
    for row in rows:
        row["ix"] = int(np.argmin(np.abs(x_support - float(row["x"]))))
        row["iy"] = int(np.argmin(np.abs(y_support - float(row["y"]))))
    return x_support, y_support


def _select_dense_rows(rows: Sequence[Dict[str, Any]], tile: Any) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if int(tile.x0) <= int(round(float(row["x"]))) <= int(tile.x1)
        and int(tile.y0) <= int(round(float(row["y"]))) <= int(tile.y1)
    ]


def _load_train_aligned(
    *,
    rows: Sequence[Dict[str, Any]],
    pattern: str,
    freq_mhz: float,
    response_crop_size: int,
    model_size: int,
    dtype: np.dtype,
    align_x: float,
    align_y: float,
) -> np.ndarray:
    aligned: List[np.ndarray] = []
    for row in rows:
        arr, _ = _load_wsclean_2d(_fmt(pattern, label=str(row["label"]), freq=float(freq_mhz)), dtype)
        cropped = _crop2d(arr, size=int(response_crop_size), center_x=None, center_y=None)
        shifted = _shift_zero(
            cropped,
            dx=int(round(float(align_x) - float(row["x"]))),
            dy=int(round(float(align_y) - float(row["y"]))),
        )
        aligned.append(np.asarray(_central_crop(shifted, int(model_size)), dtype=dtype).reshape(-1))
    return np.stack(aligned, axis=0)


def _add_exact_shifted(
    pred: np.ndarray,
    *,
    row: Dict[str, Any],
    coeff_by_offset: Dict[Tuple[int, int], float],
    pattern: str,
    freq_mhz: float,
    response_crop_size: int,
    eval_size: int,
    dtype: np.dtype,
) -> None:
    arr, _ = _load_wsclean_2d(_fmt(pattern, label=str(row["label"]), freq=float(freq_mhz)), dtype)
    cropped = _crop2d(arr, size=int(response_crop_size), center_x=None, center_y=None)
    for (dy, dx), coeff in coeff_by_offset.items():
        if coeff != 0.0:
            pred += float(coeff) * np.asarray(
                _central_crop(_shift_zero(cropped, dx=int(dx), dy=int(dy)), int(eval_size)),
                dtype=np.float64,
            )


def _support_coeffs_by_offset(
    *,
    row: Dict[str, Any],
    fg_flux: np.ndarray,
    x_contribs: Sequence[Sequence[Tuple[int, float, int]]],
    y_contribs: Sequence[Sequence[Tuple[int, float, int]]],
) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for y_pix, wy, dy in y_contribs[int(row["iy"])]:
        for x_pix, wx, dx in x_contribs[int(row["ix"])]:
            coeff = float(fg_flux[y_pix, x_pix]) * float(wy) * float(wx)
            if coeff != 0.0:
                key = (int(dy), int(dx))
                out[key] = out.get(key, 0.0) + coeff
    return out


def _fill_component_maps(
    *,
    mean_map: np.ndarray,
    coeff_maps: np.ndarray,
    row: Dict[str, Any],
    coeff_by_offset: Dict[Tuple[int, int], float],
    row_coeff: np.ndarray,
    align_x: float,
    align_y: float,
) -> None:
    center = int(mean_map.shape[0]) // 2
    sx = float(row["x"])
    sy = float(row["y"])
    for (dy, dx), weight in coeff_by_offset.items():
        ox = center + int(round((sx + float(dx)) - float(align_x)))
        oy = center + int(round((sy + float(dy)) - float(align_y)))
        if 0 <= ox < mean_map.shape[1] and 0 <= oy < mean_map.shape[0]:
            w = float(weight)
            mean_map[oy, ox] += w
            coeff_maps[:, oy, ox] += w * np.asarray(row_coeff, dtype=np.float64)


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

    dense_rows = _load_grid_csv(args.dense_grid_csv)
    x_support, y_support = _assign_grid_indices(dense_rows)
    x_contribs = _axis_linear_contribs(int(args.image_size), x_support)
    y_contribs = _axis_linear_contribs(int(args.image_size), y_support)
    train_rows_all = _load_grid_csv(args.train_grid_csv)
    train_by_key = {_row_key(row): row for row in train_rows_all}

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
        dense_tile = _select_dense_rows(dense_rows, tile)
        train_rows = _select_train_rows(
            train_rows_all,
            tile,
            halo_px=int(args.train_halo_px),
            image_size=int(args.image_size),
        )
        if not dense_tile:
            raise ValueError(f"Tile {tile.name} selected zero dense rows")
        if len(train_rows) < 4:
            raise ValueError(f"Tile {tile.name} selected too few train rows: {len(train_rows)}")
        dense_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in dense_tile], dtype=np.float64)
        train_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in train_rows], dtype=np.float64)
        align_x = float(np.median(dense_xy[:, 0]))
        align_y = float(np.median(dense_xy[:, 1]))

        contrib_positions: List[Tuple[float, float]] = []
        coeffs_by_dense: List[Dict[Tuple[int, int], float]] = []
        for row in dense_tile:
            coeff_by_offset = _support_coeffs_by_offset(
                row=row,
                fg_flux=fg_flux,
                x_contribs=x_contribs,
                y_contribs=y_contribs,
            )
            coeffs_by_dense.append(coeff_by_offset)
            for dy, dx in coeff_by_offset:
                contrib_positions.append((float(row["x"]) + float(dx), float(row["y"]) + float(dy)))
        contrib_xy = np.asarray(contrib_positions, dtype=np.float64)
        max_offset = max(
            float(np.max(np.abs(dense_xy[:, 0] - align_x))),
            float(np.max(np.abs(dense_xy[:, 1] - align_y))),
            float(np.max(np.abs(train_xy[:, 0] - align_x))),
            float(np.max(np.abs(train_xy[:, 1] - align_y))),
            float(np.max(np.abs(contrib_xy[:, 0] - align_x))),
            float(np.max(np.abs(contrib_xy[:, 1] - align_y))),
        )
        model_margin = int(args.model_margin) if int(args.model_margin) >= 0 else int(math.ceil(max_offset)) + 8
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
                    "n_dense": int(len(dense_tile)),
                    "n_train": int(len(train_rows)),
                    "model_size": int(model_size),
                    "time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                sort_keys=True,
            ),
            flush=True,
        )

        train_aligned = _load_train_aligned(
            rows=train_rows,
            pattern=str(args.train_response_pattern),
            freq_mhz=float(args.freq_mhz),
            response_crop_size=int(args.response_crop_size),
            model_size=int(model_size),
            dtype=dtype,
            align_x=align_x,
            align_y=align_y,
        )
        max_rank = min(max(ranks), max(1, len(train_rows) - 1))
        mean, basis_max, coeff_train_max = _pca_basis(train_aligned, int(max_rank))
        coeff_interp_max = {
            scale: _rbf_coefficients_many(
                target_xy=dense_xy,
                train_xy=train_xy,
                coeff_train=coeff_train_max,
                scale_px=float(scale),
            )
            for scale in scales
        }
        exact_dense_mask = np.asarray([_row_key(row) in train_by_key for row in dense_tile], dtype=bool)

        for scale in scales:
            for rank in ranks:
                k = min(int(rank), int(basis_max.shape[0]))
                for mode in modes:
                    mean_map = np.zeros((int(model_size), int(model_size)), dtype=np.float64)
                    coeff_maps = np.zeros((k, int(model_size), int(model_size)), dtype=np.float64)
                    exact_pred = np.zeros((int(args.eval_crop_size), int(args.eval_crop_size)), dtype=np.float64)
                    for i, row in enumerate(dense_tile):
                        coeff_by_offset = coeffs_by_dense[i]
                        use_exact = mode == "train_hybrid" and bool(exact_dense_mask[i])
                        if use_exact:
                            _add_exact_shifted(
                                exact_pred,
                                row=train_by_key[_row_key(row)],
                                coeff_by_offset=coeff_by_offset,
                                pattern=str(args.train_response_pattern),
                                freq_mhz=float(args.freq_mhz),
                                response_crop_size=int(args.response_crop_size),
                                eval_size=int(args.eval_crop_size),
                                dtype=dtype,
                            )
                        else:
                            _fill_component_maps(
                                mean_map=mean_map,
                                coeff_maps=coeff_maps,
                                row=row,
                                coeff_by_offset=coeff_by_offset,
                                row_coeff=coeff_interp_max[scale][i, :k],
                                align_x=align_x,
                                align_y=align_y,
                            )
                    pred_tile = _accumulate_components(
                        mean=mean,
                        basis=basis_max[:k],
                        mean_map=mean_map,
                        coeff_maps=coeff_maps,
                        model_size=int(model_size),
                        eval_size=int(args.eval_crop_size),
                    )
                    pred_tile += exact_pred
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
                "n_dense": int(len(dense_tile)),
                "n_train": int(len(train_rows)),
                "exact_dense_fraction": float(np.mean(exact_dense_mask)),
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
            "dense_grid_csv": str(args.dense_grid_csv),
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
