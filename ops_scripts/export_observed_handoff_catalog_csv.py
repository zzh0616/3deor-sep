#!/usr/bin/env python3
"""Export an observed catalog-handoff NPZ into the template-builder CSV format.

The intended input is an external observed catalog cache, such as the
GLEAM-X/EGC handoff arrays used by e2esim.  The output CSV is deliberately
minimal and matches ``build_observed_sky_template_cheb_prior.py``:

``ra_deg, dec_deg, flux_jy, ref_freq_mhz, spectral_index, curvature``.
When requested, it also preserves observed source-shape columns from the
handoff cache so downstream builders can construct non-point source-support
bases from catalog information alone.

This script is a data-format bridge only.  It must not be used with simulated
foreground truth, simulator OSM/source lists, or oracle Chebyshev coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


SUSPICIOUS_INPUT_TOKENS = (
    "simulation",
    "simulator",
    "skymap/osm_prepare",
    "osm_prepare",
    "/osm/",
    ".osm",
    "fg_cube",
    "eor_cube",
    "truth",
    "oracle",
    "synthetic",
    "perturbed_inits",
    "cube2_fullsky",
    "cheb_foreground",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _check_non_oracle_path(path: Path, *, role: str, allow_suspicious_input: bool) -> None:
    text = str(path).replace("\\", "/").lower()
    hits = [token for token in SUSPICIOUS_INPUT_TOKENS if token in text]
    if hits and not allow_suspicious_input:
        raise ValueError(
            f"{role} path looks like simulated/oracle input, not observed catalog data: {path} "
            f"(matched tokens: {hits})."
        )


def _target_header(reference_dirty: Path, image_size: int | None) -> fits.Header:
    hdr = fits.getheader(reference_dirty)
    if int(hdr.get("NAXIS", 0)) > 2:
        for key in list(hdr.keys()):
            if key.startswith("NAXIS") and key not in {"NAXIS", "NAXIS1", "NAXIS2"}:
                del hdr[key]
            elif key.endswith("3") or key.endswith("4"):
                del hdr[key]
        hdr["NAXIS"] = 2
    if image_size is not None and int(image_size) > 0:
        size = int(image_size)
        old_nx = int(hdr.get("NAXIS1", size))
        old_ny = int(hdr.get("NAXIS2", size))
        hdr["NAXIS1"] = size
        hdr["NAXIS2"] = size
        hdr["CRPIX1"] = float(hdr.get("CRPIX1", old_nx / 2.0 + 1.0)) + 0.5 * (size - old_nx)
        hdr["CRPIX2"] = float(hdr.get("CRPIX2", old_ny / 2.0 + 1.0)) + 0.5 * (size - old_ny)
    return hdr


def _field_from_reference(reference_dirty: Path, image_size: int | None, margin_deg: float) -> Tuple[float, float, float]:
    hdr = _target_header(reference_dirty, image_size)
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    wcs = WCS(hdr).celestial
    center_ra, center_dec = wcs.pixel_to_world_values((nx - 1.0) / 2.0, (ny - 1.0) / 2.0)
    corners_x = np.asarray([0.0, nx - 1.0, 0.0, nx - 1.0], dtype=np.float64)
    corners_y = np.asarray([0.0, 0.0, ny - 1.0, ny - 1.0], dtype=np.float64)
    ra_c, dec_c = wcs.pixel_to_world_values(corners_x, corners_y)
    radius = float(np.max(_angular_sep_deg(center_ra, center_dec, ra_c, dec_c)) + float(margin_deg))
    return float(center_ra) % 360.0, float(center_dec), radius


def _angular_sep_deg(ra0_deg: float, dec0_deg: float, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra0 = math.radians(float(ra0_deg))
    dec0 = math.radians(float(dec0_deg))
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    dra = (ra - ra0 + math.pi) % (2.0 * math.pi) - math.pi
    sin_ddec = np.sin(0.5 * (dec - dec0))
    sin_dra = np.sin(0.5 * dra)
    a = sin_ddec * sin_ddec + math.cos(dec0) * np.cos(dec) * sin_dra * sin_dra
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _load_array(npz: np.lib.npyio.NpzFile, name: str, default: float | str | None = None) -> np.ndarray:
    if name in npz.files:
        return np.asarray(npz[name])
    if default is None:
        raise KeyError(f"Missing required NPZ array: {name}")
    n = int(np.asarray(npz["ra_deg"]).shape[0])
    return np.full(n, default)


def _optional_array(npz: np.lib.npyio.NpzFile, name: str, n: int) -> np.ndarray:
    if name not in npz.files:
        return np.full(int(n), np.nan, dtype=np.float64)
    arr = np.asarray(npz[name], dtype=np.float64).reshape(-1)
    if arr.size == 1 and int(n) != 1:
        return np.full(int(n), float(arr[0]), dtype=np.float64)
    if arr.size != int(n):
        return np.full(int(n), np.nan, dtype=np.float64)
    return arr


def _first_scalar(npz: np.lib.npyio.NpzFile, name: str) -> Any:
    if name not in npz.files:
        return None
    arr = np.asarray(npz[name])
    if arr.size == 0:
        return None
    value = arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog-npz", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument("--reference-dirty", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--center-ra-deg", type=float, default=None)
    ap.add_argument("--center-dec-deg", type=float, default=None)
    ap.add_argument("--radius-deg", type=float, default=None)
    ap.add_argument("--margin-deg", type=float, default=0.5)
    ap.add_argument("--min-flux-jy", type=float, default=0.0)
    ap.add_argument("--max-sources", type=int, default=0, help="Optional brightest-source cap after filtering. 0 keeps all.")
    ap.add_argument("--default-alpha", type=float, default=-0.8)
    ap.add_argument(
        "--include-shape-columns",
        action="store_true",
        help="Preserve observed major/minor/PA and catalog PSF columns when available.",
    )
    ap.add_argument("--allow-suspicious-input", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    _check_non_oracle_path(args.catalog_npz, role="catalog", allow_suspicious_input=bool(args.allow_suspicious_input))
    if args.reference_dirty is not None:
        center_ra, center_dec, radius = _field_from_reference(
            args.reference_dirty,
            int(args.image_size) if int(args.image_size) > 0 else None,
            float(args.margin_deg),
        )
    else:
        if args.center_ra_deg is None or args.center_dec_deg is None or args.radius_deg is None:
            raise ValueError("Provide either --reference-dirty or explicit --center-ra-deg/--center-dec-deg/--radius-deg")
        center_ra = float(args.center_ra_deg) % 360.0
        center_dec = float(args.center_dec_deg)
        radius = float(args.radius_deg)
    if args.radius_deg is not None:
        radius = float(args.radius_deg)

    with np.load(args.catalog_npz, allow_pickle=True) as npz:
        ra = np.asarray(_load_array(npz, "ra_deg"), dtype=np.float64).reshape(-1)
        dec = np.asarray(_load_array(npz, "dec_deg"), dtype=np.float64).reshape(-1)
        if "flux_ref_mjy" in npz.files:
            flux_ref_mjy = np.asarray(npz["flux_ref_mjy"], dtype=np.float64).reshape(-1)
        else:
            flux_ref_mjy = np.asarray(_load_array(npz, "flux_150_mjy"), dtype=np.float64).reshape(-1)
        ref_freq = np.asarray(_load_array(npz, "ref_freq_mhz", 150.0), dtype=np.float64).reshape(-1)
        alpha = np.asarray(_load_array(npz, "alpha", float(args.default_alpha)), dtype=np.float64).reshape(-1)
        if "source_names" in npz.files and np.asarray(npz["source_names"]).shape[0] == ra.shape[0]:
            names = np.asarray(npz["source_names"]).astype(str).reshape(-1)
        else:
            names = np.asarray([f"src_{i:07d}" for i in range(ra.shape[0])], dtype=str)
        shape_arrays = {
            "major_obs_arcsec": _optional_array(npz, "major_obs_arcsec", int(ra.shape[0])),
            "minor_obs_arcsec": _optional_array(npz, "minor_obs_arcsec", int(ra.shape[0])),
            "pa_obs_deg": _optional_array(npz, "pa_obs_deg", int(ra.shape[0])),
            "psf_major_arcsec": _optional_array(npz, "psf_major_arcsec", int(ra.shape[0])),
            "psf_minor_arcsec": _optional_array(npz, "psf_minor_arcsec", int(ra.shape[0])),
            "psf_pa_deg": _optional_array(npz, "psf_pa_deg", int(ra.shape[0])),
        }
        meta = {
            "profile_name": _first_scalar(npz, "profile_name"),
            "coverage_kind": _first_scalar(npz, "coverage_kind"),
            "coverage_center_ra_deg": _first_scalar(npz, "coverage_center_ra_deg"),
            "coverage_center_dec_deg": _first_scalar(npz, "coverage_center_dec_deg"),
            "coverage_radius_deg": _first_scalar(npz, "coverage_radius_deg"),
            "source_size": _first_scalar(npz, "source_size"),
        }

    if not (ra.shape == dec.shape == flux_ref_mjy.shape == ref_freq.shape == alpha.shape):
        raise ValueError(
            "Catalog arrays have inconsistent shapes: "
            f"ra={ra.shape}, dec={dec.shape}, flux={flux_ref_mjy.shape}, ref_freq={ref_freq.shape}, alpha={alpha.shape}"
        )

    flux_jy = flux_ref_mjy * 1.0e-3
    sep = _angular_sep_deg(center_ra, center_dec, ra, dec)
    mask = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(flux_jy)
        & np.isfinite(ref_freq)
        & np.isfinite(alpha)
        & (flux_jy >= float(args.min_flux_jy))
        & (sep <= radius)
    )
    idx = np.flatnonzero(mask)
    if int(args.max_sources) > 0 and idx.size > int(args.max_sources):
        order = np.argsort(flux_jy[idx])[::-1]
        idx = idx[order[: int(args.max_sources)]]
    else:
        idx = idx[np.argsort(flux_jy[idx])[::-1]]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "ra_deg",
        "dec_deg",
        "flux_jy",
        "ref_freq_mhz",
        "spectral_index",
        "curvature",
        "sep_deg",
    ]
    if bool(args.include_shape_columns):
        fieldnames.extend(
            [
                "major_obs_arcsec",
                "minor_obs_arcsec",
                "pa_obs_deg",
                "psf_major_arcsec",
                "psf_minor_arcsec",
                "psf_pa_deg",
            ]
        )
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in idx:
            row = {
                "name": str(names[i]),
                "ra_deg": f"{ra[i]:.10f}",
                "dec_deg": f"{dec[i]:.10f}",
                "flux_jy": f"{flux_jy[i]:.10e}",
                "ref_freq_mhz": f"{ref_freq[i]:.6f}",
                "spectral_index": f"{alpha[i]:.8f}",
                "curvature": "0.0",
                "sep_deg": f"{sep[i]:.8f}",
            }
            if bool(args.include_shape_columns):
                for key, values in shape_arrays.items():
                    val = float(values[i])
                    row[key] = f"{val:.8f}" if math.isfinite(val) else ""
            writer.writerow(row)

    manifest_path = args.out_manifest or args.out_csv.with_suffix(".manifest.json")
    manifest = {
        "created_at": _now(),
        "script": str(Path(__file__).resolve()),
        "method": "observed_catalog_handoff_npz_to_template_csv",
        "non_cheating_policy": {
            "input_must_be_observed_catalog_cache": True,
            "suspicious_input_tokens": list(SUSPICIOUS_INPUT_TOKENS),
            "allow_suspicious_input": bool(args.allow_suspicious_input),
        },
        "input": {
            "catalog_npz": str(args.catalog_npz),
            "metadata": meta,
        },
        "selection": {
            "center_ra_deg": float(center_ra),
            "center_dec_deg": float(center_dec),
            "radius_deg": float(radius),
            "min_flux_jy": float(args.min_flux_jy),
            "max_sources": int(args.max_sources),
            "include_shape_columns": bool(args.include_shape_columns),
            "n_input": int(ra.size),
            "n_selected": int(idx.size),
            "flux_jy_max": float(np.max(flux_jy[idx])) if idx.size else 0.0,
            "flux_jy_median": float(np.median(flux_jy[idx])) if idx.size else 0.0,
            "flux_jy_min": float(np.min(flux_jy[idx])) if idx.size else 0.0,
        },
        "products": {
            "catalog_csv": str(args.out_csv),
        },
    }
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "export_done", "csv": str(args.out_csv), "manifest": str(manifest_path), "n_selected": int(idx.size)}, sort_keys=True))


if __name__ == "__main__":
    main()
