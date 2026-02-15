#!/usr/bin/env python3
"""
Build small multi-frequency dirty/PSF cubes from downloaded SDC3 WSClean products.

Expected local layout (as downloaded from Kunshan):

  data/sdc3_hpc/all_106.00/image_natural/all_106.00-dirty.fits
  data/sdc3_hpc/all_106.00/image_natural/all_106.00-psf.fits
  data/sdc3_hpc/all_106.10/image_natural/all_106.10-dirty.fits
  ...

This script stacks per-frequency 2D products into 3D cubes (F, Y, X) and (optionally)
generates corresponding sky-truth cubes in Jy/pixel from local temperature maps.

NOTE: this is a *preparation* utility for stage-2 experiments; it does not run separation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits


def _parse_freq_list(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("freq_step_mhz must be > 0")
    count = int(round((stop - start) / step)) + 1
    if count <= 0:
        raise ValueError("Invalid frequency range")
    return [round(start + i * step, 2) for i in range(count)]


def _squeeze_wsclean_2d(data: np.ndarray) -> np.ndarray:
    # WSClean commonly writes (Nstokes, Nfreq, Y, X) with singleton leading axes.
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D after squeezing, got shape {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _load_wsclean_2d(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    return _squeeze_wsclean_2d(np.asarray(data)), hdr


def _write_cube(path: Path, cube: np.ndarray, *, freq_start_mhz: float, freq_step_mhz: float) -> None:
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (F,Y,X), got {cube.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["CTYPE3"] = ("FREQ", "Frequency axis")
    hdr["CUNIT3"] = ("MHz", "Frequency unit")
    hdr["CRVAL3"] = (float(freq_start_mhz), "Reference frequency (MHz) at CRPIX3")
    hdr["CDELT3"] = (float(freq_step_mhz), "Frequency step (MHz)")
    hdr["CRPIX3"] = (1.0, "Reference pixel (1-based)")
    fits.writeto(path, np.asarray(cube, dtype=np.float32), header=hdr, overwrite=True)


def _infer_osm_clip_max_k(osm_path: Path) -> Optional[float]:
    # Read a few header lines to find "Maximum I value = ... [K]"
    try:
        with osm_path.open("r", encoding="utf-8") as f:
            for _ in range(64):
                line = f.readline()
                if not line:
                    break
                if "Maximum I value" in line:
                    # example: "# Maximum I value = 5.0000e+05 [K]"
                    parts = line.split("=")
                    if len(parts) >= 2:
                        num = parts[1].split("[")[0].strip()
                        return float(num)
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare stacked dirty/psf cubes from SDC3 WSClean FITS products.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory (e.g. runs/stage2_instrument/...)")
    ap.add_argument("--freq-start-mhz", type=float, required=True)
    ap.add_argument("--freq-stop-mhz", type=float, required=True)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument(
        "--image-root-pattern",
        type=str,
        default="data/sdc3_hpc/all_{freq:.2f}/image_natural",
        help="Local pattern pointing to per-frequency image folders.",
    )
    ap.add_argument("--weight", type=str, default="natural", choices=["natural", "uniform", "briggs"])
    ap.add_argument(
        "--write-sky-truth",
        action="store_true",
        help="Also write a sky-truth FG cube in Jy/pixel from local /data2 sky maps (clipped like OSM).",
    )
    ap.add_argument(
        "--write-truth",
        action="store_true",
        help=(
            "Also write truth cubes in Jy/pixel from local /data2 sky maps (FG clipped like OSM). "
            "This supersedes --write-sky-truth by additionally writing EoR truth."
        ),
    )
    ap.add_argument(
        "--sky-pattern",
        type=str,
        default="/data2/sdc3/simulation/skymap/osm_prepare/all_{freq:.2f}.fits",
        help="Local sky-map pattern (K).",
    )
    ap.add_argument(
        "--eor-pattern",
        type=str,
        default="/data2/sdc3/simulation/skymap/eor/deltaTb_f{freq:.2f}_N2048_fov9.1.fits",
        help="Local EoR map pattern (K). Used when --write-truth is enabled.",
    )
    ap.add_argument(
        "--osm-pattern",
        type=str,
        default="/data2/sdc3/simulation/skymap/osm_prepare/osm/all_{freq:.2f}.osm",
        help="Local OSM pattern used only to infer clipping max and K2JyPixel sanity.",
    )
    ap.add_argument(
        "--beam-fwhm-deg",
        type=float,
        default=None,
        help=(
            "If set, generate a Gaussian primary beam cube (power beam) with this FWHM in degrees "
            "at the reference frequency (see --beam-ref-mhz)."
        ),
    )
    ap.add_argument(
        "--beam-ref-mhz",
        type=float,
        default=None,
        help=(
            "Reference frequency (MHz) for --beam-fwhm-deg. When set, beam FWHM scales ~ nu_ref/nu."
        ),
    )
    ap.add_argument(
        "--write-config",
        action="store_true",
        help="Write an example separation config JSON for stage-2 (dirty/psf/beam forward model).",
    )
    ap.add_argument(
        "--psf-pad-to",
        type=int,
        default=None,
        help="If set, include psf_pad_to in the written config (e.g. 2460).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    freqs = _parse_freq_list(args.freq_start_mhz, args.freq_stop_mhz, args.freq_step_mhz)
    if not freqs:
        raise ValueError("No frequencies selected.")

    dirty_slices: List[np.ndarray] = []
    psf_slices: List[np.ndarray] = []
    dirty_hdr0: Optional[fits.Header] = None
    psf_hdr0: Optional[fits.Header] = None

    for f in freqs:
        img_dir = Path(args.image_root_pattern.format(freq=f))
        prefix = f"all_{f:.2f}"
        dirty_path = img_dir / f"{prefix}-dirty.fits"
        psf_path = img_dir / f"{prefix}-psf.fits"
        if not dirty_path.exists():
            raise FileNotFoundError(f"Missing dirty FITS: {dirty_path}")
        if not psf_path.exists():
            raise FileNotFoundError(f"Missing PSF FITS: {psf_path}")
        dirty2d, hdr_d = _load_wsclean_2d(dirty_path)
        psf2d, hdr_p = _load_wsclean_2d(psf_path)
        dirty_slices.append(dirty2d)
        psf_slices.append(psf2d)
        if dirty_hdr0 is None:
            dirty_hdr0 = hdr_d
        if psf_hdr0 is None:
            psf_hdr0 = hdr_p

    dirty_cube = np.stack(dirty_slices, axis=0)
    psf_cube = np.stack(psf_slices, axis=0)

    dirty_out = out_dir / f"dirty_{args.weight}.fits"
    psf_out = out_dir / f"psf_{args.weight}.fits"
    _write_cube(dirty_out, dirty_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(psf_out, psf_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)

    meta = {
        "freqs_mhz": freqs,
        "dirty_out": str(dirty_out),
        "psf_out": str(psf_out),
        "dirty_header_excerpt": {k: dirty_hdr0.get(k) for k in ("BUNIT", "BMAJ", "BMIN", "BPA") if dirty_hdr0},
        "psf_header_excerpt": {k: psf_hdr0.get(k) for k in ("BUNIT", "BMAJ", "BMIN", "BPA") if psf_hdr0},
    }

    want_fg_truth = bool(args.write_sky_truth or args.write_truth)
    want_eor_truth = bool(args.write_truth)
    if want_fg_truth:
        # Convert local K maps to Jy/pixel using the RJ conversion (pixel solid angle).
        # Match OSM bright-source clipping when possible.
        k_B = 1.380649e-23
        c = 299792458.0
        pix_rad = (16.0 / 3600.0) * (np.pi / 180.0)
        omega_pix = pix_rad**2
        K_to_JyPix_first = 2 * k_B * (float(freqs[0]) * 1e6) ** 2 / c**2 * omega_pix / 1e-26

        fg_slices: List[np.ndarray] = []
        eor_slices: List[np.ndarray] = []
        clip_max_k = None
        osm0 = Path(args.osm_pattern.format(freq=freqs[0]))
        if osm0.exists():
            clip_max_k = _infer_osm_clip_max_k(osm0)

        for f in freqs:
            nu_hz = float(f) * 1e6
            K_to_JyPix_f = 2 * k_B * nu_hz**2 / c**2 * omega_pix / 1e-26

            sky_path = Path(args.sky_pattern.format(freq=f))
            if not sky_path.exists():
                raise FileNotFoundError(f"Missing sky FITS: {sky_path}")
            sky_k = fits.getdata(sky_path, memmap=True).astype(np.float32)
            if sky_k.ndim != 2:
                raise ValueError(f"Expected 2D sky map: {sky_path} shape={sky_k.shape}")
            if clip_max_k is not None and np.isfinite(clip_max_k):
                sky_k = np.clip(sky_k, 0.0, float(clip_max_k))
            fg_slices.append(sky_k * float(K_to_JyPix_f))

            if want_eor_truth:
                eor_path = Path(args.eor_pattern.format(freq=f))
                if not eor_path.exists():
                    raise FileNotFoundError(f"Missing EoR FITS: {eor_path}")
                eor_k = fits.getdata(eor_path, memmap=True).astype(np.float32)
                if eor_k.ndim != 2:
                    raise ValueError(f"Expected 2D EoR map: {eor_path} shape={eor_k.shape}")
                eor_slices.append(eor_k * float(K_to_JyPix_f))

        fg_cube = np.stack(fg_slices, axis=0)
        fg_out = out_dir / "fg_true_jypix.fits"
        _write_cube(fg_out, fg_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
        meta["fg_true_jypix_out"] = str(fg_out)
        meta["K_to_JyPix_first_freq"] = float(K_to_JyPix_first)
        meta["osm_clip_max_k"] = float(clip_max_k) if clip_max_k is not None else None

        if want_eor_truth:
            eor_cube = np.stack(eor_slices, axis=0)
            eor_out = out_dir / "eor_true_jypix.fits"
            all_out = out_dir / "all_true_jypix.fits"
            _write_cube(eor_out, eor_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
            _write_cube(all_out, fg_cube + eor_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
            meta["eor_true_jypix_out"] = str(eor_out)
            meta["all_true_jypix_out"] = str(all_out)

    if args.beam_fwhm_deg is not None:
        if dirty_hdr0 is None:
            raise RuntimeError("Internal error: dirty_hdr0 is None while generating beam.")
        fwhm0 = float(args.beam_fwhm_deg)
        if fwhm0 <= 0.0 or not np.isfinite(fwhm0):
            raise ValueError("--beam-fwhm-deg must be a positive finite number.")
        nu_ref = float(args.beam_ref_mhz) if args.beam_ref_mhz is not None else float(freqs[0])
        if nu_ref <= 0.0 or not np.isfinite(nu_ref):
            raise ValueError("--beam-ref-mhz must be a positive finite number.")

        pix_deg = float(abs(dirty_hdr0.get("CDELT1", 16.0 / 3600.0)))
        cy = int(round(float(dirty_hdr0.get("CRPIX2", dirty_cube.shape[-2] // 2 + 1)) - 1.0))
        cx = int(round(float(dirty_hdr0.get("CRPIX1", dirty_cube.shape[-1] // 2 + 1)) - 1.0))
        yy, xx = np.mgrid[0 : dirty_cube.shape[-2], 0 : dirty_cube.shape[-1]]
        r_deg = np.sqrt(((yy - cy) * pix_deg) ** 2 + ((xx - cx) * pix_deg) ** 2).astype(np.float32)

        beam_slices: List[np.ndarray] = []
        for f in freqs:
            fwhm_f = fwhm0 * (nu_ref / float(f))
            beam = np.exp(-4.0 * np.log(2.0) * (r_deg**2) / float(fwhm_f**2)).astype(np.float32)
            beam_slices.append(beam)
        beam_cube = np.stack(beam_slices, axis=0)
        beam_out = out_dir / "beam_gaussian.fits"
        _write_cube(beam_out, beam_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
        meta["beam_out"] = str(beam_out)
        meta["beam_fwhm_deg_ref"] = fwhm0
        meta["beam_ref_mhz"] = nu_ref

    if args.write_config:
        cfg_path = out_dir / "config_separation.json"
        cfg = {
            "input_cube": str(dirty_out),
            "psf_cube": str(psf_out),
            "psf_scale": 1.0,
            "psf_pad_to": int(args.psf_pad_to) if args.psf_pad_to is not None else None,
            "beam_cube": meta.get("beam_out"),
            "beam_scale": 1.0,
            "fg_output": str(out_dir / "fg_est_jypix.fits"),
            "eor_output": str(out_dir / "eor_est_jypix.fits"),
            "num_iters": 1500,
            "lr": 5e-2,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "freq_axis": 0,
            "data_error": 0.05,
            "fg_smooth_mode": "diff2_l2",
            "fg_smooth_mean": 0.0,
            "fg_smooth_sigma": 0.05,
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": 0.05,
            "eor_amp_prior_mode": "voxel_deadzone",
            "eor_prior_amp_threshold": 0.0,
            "true_eor_cube": meta.get("eor_true_jypix_out"),
            "freq_start_mhz": float(freqs[0]),
            "freq_delta_mhz": float(args.freq_step_mhz),
            "print_every": 50,
        }
        cfg = {k: v for k, v in cfg.items() if v is not None}
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
        meta["config_out"] = str(cfg_path)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote: {dirty_out}")
    print(f"Wrote: {psf_out}")
    if want_fg_truth:
        print(f"Wrote: {meta.get('fg_true_jypix_out')}")
    if want_eor_truth:
        print(f"Wrote: {meta.get('eor_true_jypix_out')}")
        print(f"Wrote: {meta.get('all_true_jypix_out')}")
    if args.beam_fwhm_deg is not None:
        print(f"Wrote: {meta.get('beam_out')}")
    if args.write_config:
        print(f"Wrote: {meta.get('config_out')}")
    print(f"Wrote: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
