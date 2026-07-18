#!/usr/bin/env python3
"""Build a non-oracle sky-template Chebyshev prior from observed inputs.

This script intentionally does not read simulated foreground truth, oracle Cheb
coefficients, or simulator OSM source lists.  It accepts only explicit observed
catalog/diffuse-map files and writes a sky-side coefficient cube in Kelvin that
can be passed to ``fit_cached_pca_proxy_cheb_operator_separation.py`` via
``--prior-cheb-coeffs``.

The supported first-pass template is deliberately simple:

* point-source catalog: RA/Dec plus reference flux, spectral index, optional
  curvature, deposited into the target WCS as Jy/pixel then converted to K;
* diffuse sky map: 2D FITS image with WCS or HEALPix FITS vector, scaled with a
  temperature spectral index and optional curvature.

The output is a prior/template, not a validation result.  EoR truth and
simulated foreground products must remain outside this builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate_fullsky_response_interpolation import _k_to_jy_per_pixel  # noqa: E402


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


def _parse_freqs(spec: str) -> List[float]:
    freqs = [float(v.strip()) for v in str(spec).replace(";", ",").split(",") if v.strip()]
    if not freqs:
        raise ValueError("--freqs-mhz parsed to an empty list")
    return freqs


def _cheb_design(freqs: Sequence[float], degree: int) -> np.ndarray:
    if int(degree) < 0:
        raise ValueError("--cheb-degree must be non-negative")
    if int(degree) + 1 > len(freqs):
        raise ValueError("--cheb-degree is too high for the number of frequencies")
    x = np.asarray(freqs, dtype=np.float64)
    if x.size == 1:
        z = np.zeros_like(x)
    else:
        mid = 0.5 * (float(np.min(x)) + float(np.max(x)))
        half = 0.5 * (float(np.max(x)) - float(np.min(x)))
        z = (x - mid) / max(half, 1e-12)
    return np.polynomial.chebyshev.chebvander(z, int(degree)).astype(np.float64, copy=False)


def _squeeze_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing, got {arr.shape}")
    return np.asarray(arr)


def _target_header(reference_dirty: Path, image_size: int | None) -> fits.Header:
    hdr = fits.getheader(reference_dirty)
    if int(hdr.get("NAXIS", 0)) > 2:
        # Keep celestial WCS and force a plain 2D output image.
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


def _target_shape(header: fits.Header) -> Tuple[int, int]:
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def _pixel_arcsec_from_header(header: fits.Header) -> float:
    w = WCS(header).celestial
    try:
        scales = np.abs(w.proj_plane_pixel_scales()) * 3600.0
        scale = float(np.sqrt(float(scales[0]) * float(scales[1])))
    except Exception:
        scale = float(abs(header.get("CDELT2", header.get("CDELT1", 32.0 / 3600.0))) * 3600.0)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid pixel scale inferred from reference header: {scale}")
    return scale


def _check_non_oracle_path(path: Path | None, *, role: str, allow_suspicious_input: bool) -> None:
    if path is None:
        return
    text = str(path).replace("\\", "/").lower()
    hits = [token for token in SUSPICIOUS_INPUT_TOKENS if token in text]
    if hits and not allow_suspicious_input:
        raise ValueError(
            f"{role} path looks like simulated/oracle input, not observed data: {path} "
            f"(matched tokens: {hits}). Use a real observed catalog/map, or pass "
            "--allow-suspicious-input only for explicitly labelled oracle diagnostics."
        )


def _find_col(row: Dict[str, str], candidates: Sequence[str], *, required: bool) -> str | None:
    lower = {k.lower(): k for k in row.keys()}
    for cand in candidates:
        key = lower.get(cand.lower())
        if key is not None:
            return key
    if required:
        raise KeyError(f"Missing required column. Tried: {', '.join(candidates)}")
    return None


def _read_catalog(path: Path, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty catalog CSV: {path}")
    first = rows[0]
    ra_col = _find_col(first, ("ra_deg", "ra", "raj2000", "ra_j2000"), required=True)
    dec_col = _find_col(first, ("dec_deg", "dec", "dej2000", "dec_j2000"), required=True)
    flux_col = _find_col(first, ("flux_jy", "flux", "s_jy", "i_jy", "int_flux_jy", "peak_flux_jy"), required=True)
    ref_col = _find_col(first, ("ref_freq_mhz", "freq_mhz", "nu_mhz"), required=False)
    alpha_col = _find_col(first, ("spectral_index", "alpha", "spidx", "si"), required=False)
    curve_col = _find_col(first, ("curvature", "beta", "curve"), required=False)
    shape_cols = {
        "major_obs_arcsec": _find_col(first, ("major_obs_arcsec", "major_arcsec", "maj_arcsec"), required=False),
        "minor_obs_arcsec": _find_col(first, ("minor_obs_arcsec", "minor_arcsec", "min_arcsec"), required=False),
        "pa_obs_deg": _find_col(first, ("pa_obs_deg", "pa_deg", "position_angle_deg"), required=False),
        "psf_major_arcsec": _find_col(first, ("psf_major_arcsec", "beam_major_arcsec"), required=False),
        "psf_minor_arcsec": _find_col(first, ("psf_minor_arcsec", "beam_minor_arcsec"), required=False),
        "psf_pa_deg": _find_col(first, ("psf_pa_deg", "beam_pa_deg"), required=False),
    }

    ra: List[float] = []
    dec: List[float] = []
    flux: List[float] = []
    ref_freq: List[float] = []
    alpha: List[float] = []
    curve: List[float] = []
    shape_values: Dict[str, List[float]] = {key: [] for key in shape_cols}
    unit_scale = 1.0 if str(args.catalog_flux_unit).lower() == "jy" else 1.0e-3
    for row in rows:
        try:
            f = float(row[flux_col]) * unit_scale  # type: ignore[index]
            if not np.isfinite(f) or f < float(args.catalog_min_flux_jy):
                continue
            ra.append(float(row[ra_col]))  # type: ignore[index]
            dec.append(float(row[dec_col]))  # type: ignore[index]
            flux.append(f)
            ref_freq.append(float(row[ref_col]) if ref_col and row.get(ref_col, "").strip() else float(args.catalog_ref_freq_mhz))
            alpha.append(float(row[alpha_col]) if alpha_col and row.get(alpha_col, "").strip() else float(args.catalog_spectral_index_default))
            curve.append(float(row[curve_col]) if curve_col and row.get(curve_col, "").strip() else float(args.catalog_curvature_default))
            for key, col in shape_cols.items():
                val = float("nan")
                if col and row.get(col, "").strip():
                    try:
                        val = float(row[col])
                    except ValueError:
                        val = float("nan")
                shape_values[key].append(val)
        except (TypeError, ValueError):
            continue
    if not ra:
        raise ValueError(f"No valid catalog rows after filtering: {path}")
    catalog = {
        "ra_deg": np.asarray(ra, dtype=np.float64),
        "dec_deg": np.asarray(dec, dtype=np.float64),
        "flux_ref_jy": np.asarray(flux, dtype=np.float64),
        "ref_freq_mhz": np.asarray(ref_freq, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "curvature": np.asarray(curve, dtype=np.float64),
    }
    for key, values in shape_values.items():
        catalog[key] = np.asarray(values, dtype=np.float64)
    return catalog


def _insert_gaussian_source(
    model: np.ndarray,
    *,
    x: float,
    y: float,
    flux: float,
    major_fwhm_px: float,
    minor_fwhm_px: float,
    pa_deg: float,
    truncate_sigma: float = 4.0,
) -> bool:
    if not all(np.isfinite(v) for v in (x, y, flux, major_fwhm_px, minor_fwhm_px, pa_deg)):
        return False
    if float(flux) == 0.0:
        return False
    sigma_major = max(float(major_fwhm_px), float(minor_fwhm_px)) / 2.35482004503
    sigma_minor = min(float(major_fwhm_px), float(minor_fwhm_px)) / 2.35482004503
    if sigma_major < 0.25 or sigma_minor <= 0.0:
        return False

    height, width = model.shape
    radius = max(1, int(math.ceil(float(truncate_sigma) * sigma_major)))
    xc = int(round(float(x)))
    yc = int(round(float(y)))
    x0 = max(0, xc - radius)
    x1 = min(width - 1, xc + radius)
    y0 = max(0, yc - radius)
    y1 = min(height - 1, yc + radius)
    if x0 > x1 or y0 > y1:
        return False

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
    dx = xx - float(x)
    dy = yy - float(y)
    theta = math.radians(float(pa_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    major_coord = dx * sin_t + dy * cos_t
    minor_coord = dx * cos_t - dy * sin_t
    weights = np.exp(
        -0.5
        * (
            (major_coord / max(sigma_major, 1e-12)) ** 2
            + (minor_coord / max(sigma_minor, 1e-12)) ** 2
        )
    )
    norm = float(np.sum(weights))
    if not np.isfinite(norm) or norm <= 0.0:
        return False
    model[y0 : y1 + 1, x0 : x1 + 1] += float(flux) * weights / norm
    return True


def _insert_catalog_jypix(
    *,
    catalog: Dict[str, np.ndarray],
    freq_mhz: float,
    header: fits.Header,
    shape: Tuple[int, int],
    insert_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    wcs = WCS(header).celestial
    x, y = wcs.all_world2pix(catalog["ra_deg"], catalog["dec_deg"], 0)
    ratio = np.asarray(freq_mhz / catalog["ref_freq_mhz"], dtype=np.float64)
    # Curvature is defined in log flux: log S = log S0 + alpha log r + beta log(r)^2.
    log_ratio = np.log(np.clip(ratio, 1e-300, None))
    flux = catalog["flux_ref_jy"] * np.exp(catalog["alpha"] * log_ratio + catalog["curvature"] * log_ratio * log_ratio)

    height, width = shape
    model = np.zeros(shape, dtype=np.float64)
    mode = str(insert_mode)
    if mode == "nearest":
        xi = np.rint(x).astype(np.int64)
        yi = np.rint(y).astype(np.int64)
        ok = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height) & np.isfinite(flux)
        np.add.at(model, (yi[ok], xi[ok]), flux[ok])
        n_gaussian = 0
        n_point_fallback = 0
    elif mode == "bilinear":
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        ok = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height) & np.isfinite(flux)
        dx = x[ok] - x0[ok]
        dy = y[ok] - y0[ok]
        f = flux[ok]
        np.add.at(model, (y0[ok], x0[ok]), f * (1.0 - dx) * (1.0 - dy))
        np.add.at(model, (y0[ok], x1[ok]), f * dx * (1.0 - dy))
        np.add.at(model, (y1[ok], x0[ok]), f * (1.0 - dx) * dy)
        np.add.at(model, (y1[ok], x1[ok]), f * dx * dy)
        n_gaussian = 0
        n_point_fallback = 0
    elif mode in {"gaussian_observed", "gaussian_deconv"}:
        pixel_arcsec = _pixel_arcsec_from_header(header)
        major = np.asarray(catalog.get("major_obs_arcsec", np.full_like(flux, np.nan)), dtype=np.float64)
        minor = np.asarray(catalog.get("minor_obs_arcsec", np.full_like(flux, np.nan)), dtype=np.float64)
        pa = np.asarray(catalog.get("pa_obs_deg", np.zeros_like(flux)), dtype=np.float64)
        if mode == "gaussian_deconv":
            psf_major = np.asarray(catalog.get("psf_major_arcsec", np.zeros_like(flux)), dtype=np.float64)
            psf_minor = np.asarray(catalog.get("psf_minor_arcsec", np.zeros_like(flux)), dtype=np.float64)
            major = np.sqrt(np.maximum(major * major - psf_major * psf_major, 0.0))
            minor = np.sqrt(np.maximum(minor * minor - psf_minor * psf_minor, 0.0))
        ok = (x >= 0) & (y >= 0) & (x < width) & (y < height) & np.isfinite(flux)
        n_gaussian = 0
        n_point_fallback = 0
        for i in np.flatnonzero(ok):
            inserted = _insert_gaussian_source(
                model,
                x=float(x[i]),
                y=float(y[i]),
                flux=float(flux[i]),
                major_fwhm_px=float(major[i]) / max(float(pixel_arcsec), 1e-12),
                minor_fwhm_px=float(minor[i]) / max(float(pixel_arcsec), 1e-12),
                pa_deg=float(pa[i]) if np.isfinite(pa[i]) else 0.0,
            )
            if inserted:
                n_gaussian += 1
                continue
            n_point_fallback += 1
            xi0 = int(math.floor(float(x[i])))
            yi0 = int(math.floor(float(y[i])))
            xi1 = xi0 + 1
            yi1 = yi0 + 1
            if xi0 < 0 or yi0 < 0 or xi1 >= width or yi1 >= height:
                continue
            dx = float(x[i]) - xi0
            dy = float(y[i]) - yi0
            f = float(flux[i])
            model[yi0, xi0] += f * (1.0 - dx) * (1.0 - dy)
            model[yi0, xi1] += f * dx * (1.0 - dy)
            model[yi1, xi0] += f * (1.0 - dx) * dy
            model[yi1, xi1] += f * dx * dy
    else:
        raise ValueError(f"Unsupported catalog insert mode: {insert_mode}")
    return model, {
        "n_catalog_rows": int(len(catalog["ra_deg"])),
        "n_inserted": int(np.sum(ok)),
        "n_gaussian_inserted": int(n_gaussian),
        "n_point_fallback": int(n_point_fallback),
        "sum_jy": float(np.sum(model)),
        "rms_jypix": float(np.sqrt(np.mean(model * model))),
    }


def _is_healpix_hdu(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> bool:
    data = np.asarray(hdu.data)
    if data.ndim == 1:
        return True
    return "ORDERING" in hdu.header or "NSIDE" in hdu.header or "PIXTYPE" in hdu.header


def _reproject_wcs_bilinear(source_data: np.ndarray, source_header: fits.Header, target_header: fits.Header, shape: Tuple[int, int]) -> np.ndarray:
    source = np.asarray(source_data, dtype=np.float64)
    target_wcs = WCS(target_header).celestial
    source_wcs = WCS(source_header).celestial
    yy, xx = np.indices(shape, dtype=np.float64)
    lon, lat = target_wcs.all_pix2world(xx, yy, 0)
    sx, sy = source_wcs.all_world2pix(lon, lat, 0)

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < source.shape[1]) & (y1 < source.shape[0])
    out = np.zeros(shape, dtype=np.float64)
    if np.any(valid):
        dx = sx[valid] - x0[valid]
        dy = sy[valid] - y0[valid]
        out[valid] = (
            source[y0[valid], x0[valid]] * (1.0 - dx) * (1.0 - dy)
            + source[y0[valid], x1[valid]] * dx * (1.0 - dy)
            + source[y1[valid], x0[valid]] * (1.0 - dx) * dy
            + source[y1[valid], x1[valid]] * dx * dy
        )
    return out


def _load_diffuse_base_map(path: Path, header: fits.Header, shape: Tuple[int, int], *, unit: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
        if _is_healpix_hdu(hdu):
            import healpy as hp

            data = np.asarray(hdu.data)
            if data.dtype.fields:
                first_name = list(data.dtype.fields.keys())[0]
                values = np.asarray(data[first_name], dtype=np.float64)
            else:
                values = np.asarray(data, dtype=np.float64).reshape(-1)
            nside = hp.npix2nside(values.size)
            ordering = str(hdu.header.get("ORDERING", "RING")).upper()
            y, x = np.indices(shape, dtype=np.float64)
            lon, lat = WCS(header).celestial.all_pix2world(x, y, 0)
            theta = np.deg2rad(90.0 - lat)
            phi = np.deg2rad(lon)
            pix = hp.ang2pix(nside, theta, phi, nest=(ordering == "NESTED"))
            out = values[pix]
            mode = "healpix_nearest"
        else:
            source_data = _squeeze_2d(np.asarray(hdu.data))
            try:
                from reproject import reproject_interp
            except ModuleNotFoundError:
                out = _reproject_wcs_bilinear(source_data, hdu.header, header, shape)
                mode = "wcs_bilinear_fallback"
            else:
                source_hdu = fits.PrimaryHDU(data=source_data, header=hdu.header)
                out, _footprint = reproject_interp(source_hdu, header, shape_out=shape)
                out = np.asarray(out, dtype=np.float64)
                mode = "wcs_reproject_interp"
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if str(unit).lower() in {"mk", "millikelvin"}:
        out *= 1.0e-3
    elif str(unit).lower() not in {"k", "kelvin"}:
        raise ValueError("--diffuse-unit must be K or mK")
    return out, {
        "mode": mode,
        "shape": [int(v) for v in out.shape],
        "min_k": float(np.min(out)),
        "max_k": float(np.max(out)),
        "rms_k": float(np.sqrt(np.mean(out * out))),
    }


def _scale_diffuse_k(base_k: np.ndarray, freq_mhz: float, ref_freq_mhz: float, alpha: float, curvature: float) -> np.ndarray:
    ratio = float(freq_mhz) / float(ref_freq_mhz)
    log_ratio = math.log(max(ratio, 1e-300))
    scale = math.exp(float(alpha) * log_ratio + float(curvature) * log_ratio * log_ratio)
    return np.asarray(base_k * scale, dtype=np.float64)


def _write_cube(path: Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    hdr = fits.Header() if header is None else header.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reference-dirty", type=Path, required=True, help="Observed/target dirty FITS whose WCS defines the prior grid.")
    ap.add_argument("--freqs-mhz", required=True)
    ap.add_argument("--cheb-degree", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=0, help="Optional 2D output size. 0 uses reference dirty size.")
    ap.add_argument("--catalog-csv", type=Path, default=None)
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
    ap.add_argument("--diffuse-fits", type=Path, default=None)
    ap.add_argument("--diffuse-ref-freq-mhz", type=float, default=408.0)
    ap.add_argument("--diffuse-unit", choices=("K", "mK"), default="K")
    ap.add_argument("--diffuse-spectral-index", type=float, default=-2.55)
    ap.add_argument("--diffuse-curvature", type=float, default=0.0)
    ap.add_argument("--allow-suspicious-input", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate paths/policy and write only a manifest.")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.catalog_csv is None and args.diffuse_fits is None:
        raise ValueError("At least one of --catalog-csv or --diffuse-fits is required.")
    _check_non_oracle_path(args.catalog_csv, role="catalog", allow_suspicious_input=bool(args.allow_suspicious_input))
    _check_non_oracle_path(args.diffuse_fits, role="diffuse map", allow_suspicious_input=bool(args.allow_suspicious_input))
    _check_non_oracle_path(args.reference_dirty, role="reference dirty", allow_suspicious_input=True)

    freqs = _parse_freqs(args.freqs_mhz)
    header = _target_header(args.reference_dirty, int(args.image_size) if int(args.image_size) > 0 else None)
    shape = _target_shape(header)
    pixel_arcsec = _pixel_arcsec_from_header(header)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_at": _now(),
        "script": str(Path(__file__).resolve()),
        "method": "observed_catalog_diffuse_sky_template_cheb_prior",
        "non_cheating_policy": {
            "main_result_must_not_use": [
                "simulated foreground truth",
                "truth/oracle Chebyshev coefficients",
                "simulator OSM/source lists",
                "synthetic perturbation priors",
            ],
            "suspicious_input_tokens": list(SUSPICIOUS_INPUT_TOKENS),
            "allow_suspicious_input": bool(args.allow_suspicious_input),
        },
        "settings": {
            "reference_dirty": str(args.reference_dirty),
            "freqs_mhz": [float(v) for v in freqs],
            "cheb_degree": int(args.cheb_degree),
            "image_shape": [int(v) for v in shape],
            "pixel_arcsec": float(pixel_arcsec),
            "catalog_csv": str(args.catalog_csv) if args.catalog_csv is not None else None,
            "diffuse_fits": str(args.diffuse_fits) if args.diffuse_fits is not None else None,
        },
    }
    if args.dry_run:
        manifest["dry_run"] = True
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "dry_run_done", "manifest": str(manifest_path)}, sort_keys=True))
        return

    catalog = _read_catalog(args.catalog_csv, args) if args.catalog_csv is not None else None
    diffuse_base = None
    diffuse_stats = None
    if args.diffuse_fits is not None:
        diffuse_base, diffuse_stats = _load_diffuse_base_map(args.diffuse_fits, header, shape, unit=str(args.diffuse_unit))

    cube_k = np.zeros((len(freqs), shape[0], shape[1]), dtype=np.float64)
    per_freq: List[Dict[str, Any]] = []
    for i, freq in enumerate(freqs):
        row: Dict[str, Any] = {"freq_mhz": float(freq)}
        if catalog is not None:
            cat_jy, cat_stats = _insert_catalog_jypix(
                catalog=catalog,
                freq_mhz=float(freq),
                header=header,
                shape=shape,
                insert_mode=str(args.catalog_insert_mode),
            )
            cat_k = cat_jy / _k_to_jy_per_pixel(float(freq), float(pixel_arcsec))
            cube_k[i] += cat_k
            row["catalog"] = cat_stats
            row["catalog_rms_k"] = float(np.sqrt(np.mean(cat_k * cat_k)))
        if diffuse_base is not None:
            diffuse_k = _scale_diffuse_k(
                diffuse_base,
                float(freq),
                float(args.diffuse_ref_freq_mhz),
                float(args.diffuse_spectral_index),
                float(args.diffuse_curvature),
            )
            cube_k[i] += diffuse_k
            row["diffuse_rms_k"] = float(np.sqrt(np.mean(diffuse_k * diffuse_k)))
        row["template_rms_k"] = float(np.sqrt(np.mean(cube_k[i] * cube_k[i])))
        row["template_min_k"] = float(np.min(cube_k[i]))
        row["template_max_k"] = float(np.max(cube_k[i]))
        per_freq.append(row)

    basis = _cheb_design(freqs, int(args.cheb_degree))
    coeff = np.linalg.pinv(basis) @ cube_k.reshape(len(freqs), -1)
    coeff_cube = coeff.reshape(int(args.cheb_degree) + 1, shape[0], shape[1])
    recon = (basis @ coeff).reshape(cube_k.shape)
    diff = recon - cube_k

    template_cube_path = out_dir / "observed_template_cube_k.fits"
    coeff_path = out_dir / f"observed_template_cheb_deg{int(args.cheb_degree)}_coeff_k.fits"
    _write_cube(template_cube_path, cube_k, header)
    _write_cube(coeff_path, coeff_cube, None)

    manifest.update(
        {
            "dry_run": False,
            "diffuse_input": diffuse_stats,
            "per_frequency": per_freq,
            "cheb_fit": {
                "basis": "chebyshev_frequency_normalized_to_minus1_plus1",
                "degree": int(args.cheb_degree),
                "max_abs_k": float(np.max(np.abs(diff))),
                "rms_abs_k": float(np.sqrt(np.mean(diff * diff))),
                "max_rel": float(np.max(np.abs(diff) / np.maximum(np.abs(cube_k), 1e-30))),
            },
            "products": {
                "template_cube_k": str(template_cube_path),
                "cheb_coeffs_k": str(coeff_path),
            },
        }
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "observed_template_prior_done",
                "manifest": str(manifest_path),
                "cheb_coeffs_k": str(coeff_path),
                "template_rms_k_median": float(np.median([row["template_rms_k"] for row in per_freq])),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
