#!/usr/bin/env python3
"""Build non-oracle source-support bases from an observed catalog.

This is a stronger point-source prior than a single fixed catalog template, but
it still keeps the spatial support external: source positions and reference
fluxes come only from an observed catalog CSV.  The script splits sources that
land on the target WCS into a small set of predeclared groups, then writes one
Chebyshev coefficient cube per group for use as template bases in the cached
operator optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_observed_sky_template_cheb_prior import (  # noqa: E402
    _check_non_oracle_path,
    _cheb_design,
    _insert_catalog_jypix,
    _k_to_jy_per_pixel,
    _parse_freqs,
    _pixel_arcsec_from_header,
    _read_catalog,
    _target_header,
    _target_shape,
    _write_cube,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_spatial_grid(spec: str) -> Tuple[int, int]:
    text = str(spec or "1,1").lower().replace("x", ",")
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) == 1:
        vals = [vals[0], vals[0]]
    if len(vals) != 2 or vals[0] <= 0 or vals[1] <= 0:
        raise ValueError(f"--spatial-grid expects NX,NY with positive integers, got {spec!r}")
    return int(vals[0]), int(vals[1])


def _slice_catalog(catalog: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {key: np.asarray(value)[idx] for key, value in catalog.items()}


def _catalog_pixels(catalog: Dict[str, np.ndarray], header: fits.Header) -> Tuple[np.ndarray, np.ndarray]:
    wcs = WCS(header).celestial
    x, y = wcs.all_world2pix(catalog["ra_deg"], catalog["dec_deg"], 0)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _inside_target(
    x: np.ndarray,
    y: np.ndarray,
    shape: Tuple[int, int],
    *,
    insert_mode: str,
) -> np.ndarray:
    height, width = shape
    if str(insert_mode) in {"nearest", "gaussian_observed", "gaussian_deconv"}:
        xi = np.rint(x)
        yi = np.rint(y)
        return (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    x0 = np.floor(x)
    y0 = np.floor(y)
    return (x0 >= 0) & (y0 >= 0) & ((x0 + 1) < width) & ((y0 + 1) < height)


def _safe_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out.strip("._") or "basis"


def _make_groups(
    *,
    catalog: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    inside: np.ndarray,
    shape: Tuple[int, int],
    top_n_singletons: int,
    flux_bins: int,
    spatial_grid: Tuple[int, int],
    min_sources_per_bin: int,
    include_all_inside: bool,
) -> List[Tuple[str, np.ndarray]]:
    flux = np.asarray(catalog["flux_ref_jy"], dtype=np.float64)
    inside_idx = np.flatnonzero(inside & np.isfinite(flux) & (flux > 0.0))
    if inside_idx.size == 0:
        raise ValueError("No observed catalog sources land inside the target WCS")
    ordered = inside_idx[np.argsort(flux[inside_idx])[::-1]]
    groups: List[Tuple[str, np.ndarray]] = []
    if include_all_inside:
        groups.append(("all_inside", ordered.copy()))

    n_single = min(max(int(top_n_singletons), 0), int(ordered.size))
    for rank, idx in enumerate(ordered[:n_single]):
        label = f"single_{rank:02d}_flux{flux[idx]:.3g}jy"
        groups.append((_safe_label(label), np.asarray([idx], dtype=np.int64)))

    remaining = ordered[n_single:]
    if remaining.size == 0 or int(flux_bins) <= 0:
        return groups

    nx, ny = spatial_grid
    height, width = shape
    ix = np.clip(np.floor(x[remaining] / max(float(width), 1.0) * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor(y[remaining] / max(float(height), 1.0) * ny).astype(np.int64), 0, ny - 1)

    flux_chunks = np.array_split(remaining, int(flux_bins))
    for b, chunk in enumerate(flux_chunks):
        if chunk.size == 0:
            continue
        chunk_set = set(int(v) for v in chunk.tolist())
        if nx == 1 and ny == 1:
            if chunk.size >= int(min_sources_per_bin):
                label = f"fluxbin_{b:02d}_n{chunk.size:03d}"
                groups.append((_safe_label(label), np.asarray(chunk, dtype=np.int64)))
            continue
        for cy in range(ny):
            for cx in range(nx):
                cell_members = [
                    int(src_idx)
                    for src_idx, sx_cell, sy_cell in zip(remaining.tolist(), ix.tolist(), iy.tolist())
                    if int(src_idx) in chunk_set and int(sx_cell) == cx and int(sy_cell) == cy
                ]
                if len(cell_members) < int(min_sources_per_bin):
                    continue
                label = f"fluxbin_{b:02d}_x{cx}_y{cy}_n{len(cell_members):03d}"
                groups.append((_safe_label(label), np.asarray(cell_members, dtype=np.int64)))
    return groups


def _build_coeff_for_group(
    *,
    catalog: Dict[str, np.ndarray],
    indices: Sequence[int],
    freqs_mhz: Sequence[float],
    cheb_degree: int,
    header: fits.Header,
    shape: Tuple[int, int],
    pixel_arcsec: float,
    insert_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    subcat = _slice_catalog(catalog, indices)
    cube_k = np.zeros((len(freqs_mhz), shape[0], shape[1]), dtype=np.float64)
    per_freq: List[Dict[str, Any]] = []
    for i, freq in enumerate(freqs_mhz):
        model_jy, stats = _insert_catalog_jypix(
            catalog=subcat,
            freq_mhz=float(freq),
            header=header,
            shape=shape,
            insert_mode=str(insert_mode),
        )
        model_k = model_jy / _k_to_jy_per_pixel(float(freq), float(pixel_arcsec))
        cube_k[i] = model_k
        per_freq.append(
            {
                "freq_mhz": float(freq),
                "n_inserted": int(stats["n_inserted"]),
                "sum_jy": float(stats["sum_jy"]),
                "rms_k": float(np.sqrt(np.mean(model_k * model_k))),
            }
        )
    basis = _cheb_design(freqs_mhz, int(cheb_degree))
    coeff = np.linalg.pinv(basis) @ cube_k.reshape(len(freqs_mhz), -1)
    coeff_cube = coeff.reshape(int(cheb_degree) + 1, shape[0], shape[1])
    recon = (basis @ coeff).reshape(cube_k.shape)
    diff = recon - cube_k
    return coeff_cube, {
        "n_sources": int(len(indices)),
        "sum_flux_ref_jy": float(np.sum(subcat["flux_ref_jy"])),
        "max_flux_ref_jy": float(np.max(subcat["flux_ref_jy"])) if len(indices) else 0.0,
        "coeff_c0_rms_k": float(np.sqrt(np.mean(coeff_cube[0] * coeff_cube[0]))),
        "cheb_fit_rms_abs_k": float(np.sqrt(np.mean(diff * diff))),
        "per_frequency": per_freq,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reference-dirty", type=Path, required=True)
    ap.add_argument("--catalog-csv", type=Path, required=True)
    ap.add_argument("--freqs-mhz", required=True)
    ap.add_argument("--cheb-degree", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--catalog-ref-freq-mhz", type=float, default=150.0)
    ap.add_argument("--catalog-flux-unit", choices=("Jy", "mJy"), default="Jy")
    ap.add_argument("--catalog-min-flux-jy", type=float, default=0.0)
    ap.add_argument("--catalog-spectral-index-default", type=float, default=-0.8)
    ap.add_argument("--catalog-curvature-default", type=float, default=0.0)
    ap.add_argument(
        "--catalog-insert-mode",
        choices=("bilinear", "nearest", "gaussian_observed", "gaussian_deconv"),
        default="bilinear",
    )
    ap.add_argument("--top-n-singletons", type=int, default=16)
    ap.add_argument("--flux-bins", type=int, default=4)
    ap.add_argument("--spatial-grid", default="2,2")
    ap.add_argument("--min-sources-per-bin", type=int, default=1)
    ap.add_argument("--include-all-inside", action="store_true")
    ap.add_argument("--allow-suspicious-input", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    _check_non_oracle_path(args.catalog_csv, role="catalog", allow_suspicious_input=bool(args.allow_suspicious_input))
    _check_non_oracle_path(args.reference_dirty, role="reference dirty", allow_suspicious_input=True)

    freqs = _parse_freqs(args.freqs_mhz)
    header = _target_header(args.reference_dirty, int(args.image_size) if int(args.image_size) > 0 else None)
    shape = _target_shape(header)
    pixel_arcsec = _pixel_arcsec_from_header(header)
    spatial_grid = _parse_spatial_grid(str(args.spatial_grid))
    catalog = _read_catalog(args.catalog_csv, args)
    x, y = _catalog_pixels(catalog, header)
    inside = _inside_target(x, y, shape, insert_mode=str(args.catalog_insert_mode))
    groups = _make_groups(
        catalog=catalog,
        x=x,
        y=y,
        inside=inside,
        shape=shape,
        top_n_singletons=int(args.top_n_singletons),
        flux_bins=int(args.flux_bins),
        spatial_grid=spatial_grid,
        min_sources_per_bin=int(args.min_sources_per_bin),
        include_all_inside=bool(args.include_all_inside),
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "created_at": _now(),
        "script": str(Path(__file__).resolve()),
        "method": "observed_catalog_source_support_basis",
        "non_cheating_policy": {
            "source_support_from_observed_catalog_only": True,
            "reference_dirty_used_only_for_wcs": True,
            "must_not_use": [
                "simulated foreground truth",
                "truth/oracle Chebyshev coefficients",
                "simulator OSM/source lists",
                "synthetic perturbation priors",
            ],
            "allow_suspicious_input": bool(args.allow_suspicious_input),
        },
        "settings": {
            "catalog_csv": str(args.catalog_csv),
            "reference_dirty": str(args.reference_dirty),
            "freqs_mhz": [float(v) for v in freqs],
            "cheb_degree": int(args.cheb_degree),
            "image_shape": [int(v) for v in shape],
            "pixel_arcsec": float(pixel_arcsec),
            "catalog_ref_freq_mhz": float(args.catalog_ref_freq_mhz),
            "catalog_flux_unit": str(args.catalog_flux_unit),
            "catalog_min_flux_jy": float(args.catalog_min_flux_jy),
            "catalog_spectral_index_default": float(args.catalog_spectral_index_default),
            "catalog_curvature_default": float(args.catalog_curvature_default),
            "catalog_insert_mode": str(args.catalog_insert_mode),
            "top_n_singletons": int(args.top_n_singletons),
            "flux_bins": int(args.flux_bins),
            "spatial_grid": [int(spatial_grid[0]), int(spatial_grid[1])],
            "min_sources_per_bin": int(args.min_sources_per_bin),
            "include_all_inside": bool(args.include_all_inside),
        },
        "catalog_stats": {
            "n_catalog_rows": int(len(catalog["ra_deg"])),
            "n_inside_target": int(np.sum(inside)),
            "inside_flux_sum_jy": float(np.sum(catalog["flux_ref_jy"][inside])),
            "inside_flux_max_jy": float(np.max(catalog["flux_ref_jy"][inside])) if np.any(inside) else 0.0,
        },
        "groups_requested": [
            {"label": label, "n_sources": int(len(indices))}
            for label, indices in groups
        ],
        "dry_run": bool(args.dry_run),
        "basis": [],
        "products": {},
    }
    if args.dry_run:
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "dry_run_done", "manifest": str(manifest_path), "n_basis": int(len(groups))}, sort_keys=True))
        return

    basis_paths: List[str] = []
    for out_index, (label, indices) in enumerate(groups):
        coeff_cube, stats = _build_coeff_for_group(
            catalog=catalog,
            indices=indices,
            freqs_mhz=freqs,
            cheb_degree=int(args.cheb_degree),
            header=header,
            shape=shape,
            pixel_arcsec=float(pixel_arcsec),
            insert_mode=str(args.catalog_insert_mode),
        )
        if not np.isfinite(coeff_cube).all() or float(np.sqrt(np.mean(coeff_cube * coeff_cube))) <= 0.0:
            continue
        path = out_dir / f"basis_{out_index:03d}_{_safe_label(label)}_cheb_coeff_k.fits"
        _write_cube(path, coeff_cube, None)
        basis_paths.append(str(path))
        manifest["basis"].append(
            {
                "index": int(len(basis_paths) - 1),
                "label": str(label),
                "path": str(path),
                "source_indices": [int(value) for value in np.asarray(indices).tolist()],
                "exact_reconstruction": "catalog_rows_at_requested_frequencies",
                **stats,
            }
        )

    if not basis_paths:
        raise RuntimeError("No nonzero observed catalog source-support basis was written")
    (out_dir / "basis_paths.txt").write_text("\n".join(basis_paths) + "\n", encoding="utf-8")
    (out_dir / "basis_paths_comma.txt").write_text(",".join(basis_paths) + "\n", encoding="utf-8")
    (out_dir / "basis_gain_init_ones.txt").write_text(",".join(["1"] * len(basis_paths)) + "\n", encoding="utf-8")
    (out_dir / "basis_gain_init_zeros.txt").write_text(",".join(["0"] * len(basis_paths)) + "\n", encoding="utf-8")
    manifest["products"] = {
        "basis_paths": str(out_dir / "basis_paths.txt"),
        "basis_paths_comma": str(out_dir / "basis_paths_comma.txt"),
        "basis_gain_init_ones": str(out_dir / "basis_gain_init_ones.txt"),
        "basis_gain_init_zeros": str(out_dir / "basis_gain_init_zeros.txt"),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "observed_catalog_source_basis_done",
                "manifest": str(manifest_path),
                "n_basis": int(len(basis_paths)),
                "n_inside_target": int(np.sum(inside)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
