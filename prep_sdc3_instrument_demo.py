#!/usr/bin/env python3
"""
Prepare a small instrument-effect demo dataset using SDC3 sky maps:

- Foreground truth: /data2/sdc3/simulation/skymap/osm_prepare/all_{freq}.fits
- EoR truth: /data2/sdc3/simulation/skymap/eor/deltaTb_f{freq}_N2048_fov9.1.fits
- PSF: a WSClean PSF FITS (typically 4D with leading singleton axes)

It writes 3D FITS cubes (F, Y, X) under an output directory:
  - fg_true.fits
  - eor_true.fits
  - psf_cube.fits
  - obs_dirty_synth.fits  (PSF-convolved fg+eor)
  - config_separation.json (example config to run separation_cli.py)

This is intended as a *stage-2* instrument-effect sanity check: the observation
is no longer fg+eor, but A(fg+eor) where A is a per-frequency PSF convolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits


def _parse_freq_list(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("freq_step_mhz must be > 0")
    # Inclusive stop (within float tolerance).
    count = int(round((stop - start) / step)) + 1
    if count <= 0:
        raise ValueError("Invalid frequency range")
    freqs = [start + i * step for i in range(count)]
    # Round to 2 decimals to match filenames (e.g. 106.10).
    return [round(f, 2) for f in freqs]


def _load_2d(path: Path) -> np.ndarray:
    data = fits.getdata(path, memmap=True)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D FITS image in {path}, got shape {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _load_wsclean_2d(path: Path) -> np.ndarray:
    data = fits.getdata(path, memmap=True)
    # WSClean commonly writes (Nstokes, Nfreq, Y, X) with singleton leading axes.
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected WSClean-like 2D image in {path}, got shape {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _psf_convolve_cube(x: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Circular convolution via FFT for a cube (F, Y, X).
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D (F,Y,X), got {x.shape}")
    if psf.ndim != 3:
        raise ValueError(f"psf must be 3D (Fpsf,Y,X), got {psf.shape}")
    if x.shape[1:] != psf.shape[1:]:
        raise ValueError(f"spatial shape mismatch: x={x.shape[1:]}, psf={psf.shape[1:]}")
    if psf.shape[0] not in (1, x.shape[0]):
        raise ValueError(f"psf F must be 1 or match x: psf_F={psf.shape[0]}, x_F={x.shape[0]}")

    # Broadcast PSF if needed.
    if psf.shape[0] == 1 and x.shape[0] != 1:
        psf = np.repeat(psf, x.shape[0], axis=0)

    psf0 = np.fft.ifftshift(psf, axes=(-2, -1))
    psf_fft = np.fft.rfft2(psf0, axes=(-2, -1))
    x_fft = np.fft.rfft2(x, axes=(-2, -1))
    y = np.fft.irfft2(x_fft * psf_fft, s=x.shape[-2:], axes=(-2, -1))
    return np.asarray(y, dtype=np.float32)


def _write_cube(path: Path, cube: np.ndarray, *, freq_start_mhz: float, freq_step_mhz: float) -> None:
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got {cube.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["CTYPE3"] = ("FREQ", "Frequency axis")
    hdr["CUNIT3"] = ("MHz", "Frequency unit")
    hdr["CRVAL3"] = (float(freq_start_mhz), "Reference frequency (MHz) at CRPIX3")
    hdr["CDELT3"] = (float(freq_step_mhz), "Frequency step (MHz)")
    hdr["CRPIX3"] = (1.0, "Reference pixel (1-based)")
    fits.writeto(path, np.asarray(cube, dtype=np.float32), header=hdr, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SDC3 instrument-effect demo cubes.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory under the project root.")
    ap.add_argument("--freq-start-mhz", type=float, default=106.0)
    ap.add_argument("--freq-stop-mhz", type=float, default=107.0)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument(
        "--psf-fits",
        type=str,
        default="data/sdc3_hpc/all_106.00/image_natural/all_106.00-psf.fits",
        help="Path to a WSClean PSF FITS file (will be squeezed to 2D and broadcast across frequency).",
    )
    ap.add_argument(
        "--fg-pattern",
        type=str,
        default="/data2/sdc3/simulation/skymap/osm_prepare/all_{freq:.2f}.fits",
        help="Foreground FITS filename pattern.",
    )
    ap.add_argument(
        "--eor-pattern",
        type=str,
        default="/data2/sdc3/simulation/skymap/eor/deltaTb_f{freq:.2f}_N2048_fov9.1.fits",
        help="EoR FITS filename pattern.",
    )
    ap.add_argument("--num-iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--data-error", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    freqs = _parse_freq_list(args.freq_start_mhz, args.freq_stop_mhz, args.freq_step_mhz)
    if not freqs:
        raise ValueError("No frequencies selected.")

    # Load FG/EoR stacks.
    fg_slices: List[np.ndarray] = []
    eor_slices: List[np.ndarray] = []
    for f in freqs:
        fg_path = Path(args.fg_pattern.format(freq=f))
        eor_path = Path(args.eor_pattern.format(freq=f))
        if not fg_path.exists():
            raise FileNotFoundError(f"Missing FG FITS: {fg_path}")
        if not eor_path.exists():
            raise FileNotFoundError(f"Missing EoR FITS: {eor_path}")
        fg_slices.append(_load_2d(fg_path))
        eor_slices.append(_load_2d(eor_path))

    fg_cube = np.stack(fg_slices, axis=0)
    eor_cube = np.stack(eor_slices, axis=0)

    # Load PSF (2D) and broadcast across F.
    psf_path = Path(args.psf_fits)
    if not psf_path.exists():
        raise FileNotFoundError(f"Missing PSF FITS: {psf_path}")
    psf_2d = _load_wsclean_2d(psf_path)
    psf_cube = psf_2d[None, :, :]

    obs_cube = _psf_convolve_cube(fg_cube + eor_cube, psf_cube)

    fg_out = out_dir / "fg_true.fits"
    eor_out = out_dir / "eor_true.fits"
    psf_out = out_dir / "psf_cube.fits"
    obs_out = out_dir / "obs_dirty_synth.fits"
    cfg_out = out_dir / "config_separation.json"

    _write_cube(fg_out, fg_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(eor_out, eor_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(psf_out, psf_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(obs_out, obs_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)

    cfg = {
        "input_cube": str(obs_out),
        "fg_output": str(out_dir / "fg_est.fits"),
        "eor_output": str(out_dir / "eor_est.fits"),
        "psf_cube": str(psf_out),
        "psf_scale": 1.0,
        "num_iters": int(args.num_iters),
        "lr": float(args.lr),
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "freq_axis": 0,
        "data_error": float(args.data_error),
        "fg_smooth_mode": "diff2_l2",
        "fg_smooth_mean": 0.0,
        "fg_smooth_sigma": 0.05,
        "eor_prior_mean": 0.0,
        "eor_prior_sigma": 0.1,
        "eor_amp_prior_mode": "voxel_deadzone",
        "eor_prior_amp_threshold": 0.0,
        "true_eor_cube": str(eor_out),
        "freq_start_mhz": float(freqs[0]),
        "freq_delta_mhz": float(args.freq_step_mhz),
        "print_every": 50,
        "device": "cpu",
    }
    cfg_out.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote demo cubes to: {out_dir}")
    print(f"  FG   : {fg_out}")
    print(f"  EoR  : {eor_out}")
    print(f"  PSF  : {psf_out}")
    print(f"  OBS  : {obs_out}")
    print(f"  CFG  : {cfg_out}")


if __name__ == "__main__":
    main()
