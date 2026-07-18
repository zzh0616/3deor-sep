#!/usr/bin/env python3
"""
Build the Plan-A PSF-aware controlled dirty-image dataset.

The intended use is a bounded response to the ApJ referee report:

    d_nu = B_nu * (F_nu + E_nu) + n_nu

where B_nu is a known, frequency-dependent dirty PSF.  The script deliberately defaults
to the safer cube2 foreground rebuilt from component maps without the simplified
cluster/radio-halo component.  It does not use the anomalously bright cube1 branch.

All direct convolutions are performed in float64 by default.  The OSKAR/WSClean PSF FITS
files may have been produced by a single-precision radio-imaging chain, but the controlled
dataset generation itself should not add an extra float32 truncation unless explicitly
requested with --dtype float32.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a cube2 drop-cluster PSF-aware dirty-image dataset.")
    ap.add_argument("--work-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/planA_psf_dirty_cube2_dropcluster_20260528"),
        help="Output directory, relative to work-root unless absolute.",
    )
    ap.add_argument("--freq-start-mhz", type=float, default=106.0)
    ap.add_argument("--freq-stop-mhz", type=float, default=121.0)
    ap.add_argument("--freq-step-mhz", type=float, default=0.1)
    ap.add_argument(
        "--eor-cube",
        type=Path,
        default=Path("data/eor_cube2.fits"),
        help="Reference EoR cube for cube2.",
    )
    ap.add_argument(
        "--component-root",
        type=Path,
        default=Path("e2esim_runs/cube2/sky_model"),
        help="Root containing cube2 component maps.",
    )
    ap.add_argument(
        "--bright-mask",
        type=Path,
        default=Path("data/mask_cube2.fits"),
        help=(
            "Bright-source mask from the e2esim postprocess step. Pixels with mask>0 "
            "are replaced by the postprocessed fallback foreground cube."
        ),
    )
    ap.add_argument(
        "--masked-fg-fallback",
        type=Path,
        default=Path("data/fg_cube2.fits"),
        help="Postprocessed foreground cube used only inside the bright-source mask.",
    )
    ap.add_argument(
        "--disable-bright-mask",
        action="store_true",
        help="Disable bright-source-mask replacement. This is not recommended for Plan A.",
    )
    ap.add_argument(
        "--psf-pattern",
        type=str,
        default="data/sdc3_hpc/all_{freq:.2f}/image_natural/all_{freq:.2f}-psf.fits",
        help="Frequency-dependent PSF FITS pattern.",
    )
    ap.add_argument(
        "--psf-cube",
        type=Path,
        default=None,
        help="Optional pre-stacked PSF cube. When set, psf-pattern is ignored.",
    )
    ap.add_argument(
        "--cut-size-px",
        type=int,
        default=512,
        help="Center crop size for a first controlled run. Use 0 for full image.",
    )
    ap.add_argument("--noise-mk", type=float, default=0.0, help="Optional image-domain white-noise RMS in mK.")
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument(
        "--fg-max-k-fail",
        type=float,
        default=5.0e4,
        help="Abort if the processed foreground exceeds this brightness temperature in K.",
    )
    ap.add_argument(
        "--keep-temp-npy",
        action="store_true",
        help="Keep intermediate memmap .npy files under out-dir/_tmp.",
    )
    ap.add_argument(
        "--psf-pad-to",
        type=int,
        default=None,
        help="Optional FFT pad size for convolution and for the written separation config.",
    )
    ap.add_argument(
        "--no-normalize-psf-peak",
        action="store_true",
        help="Do not normalize each PSF by its central pixel/peak value.",
    )
    ap.add_argument(
        "--keep-psf-mean",
        action="store_true",
        help=(
            "Keep the finite-grid PSF mean. By default the PSF mean is removed to "
            "avoid an artificial total-power response in this interferometric control test."
        ),
    )
    ap.add_argument("--num-iters", type=int, default=1200, help="Iteration count for the example config.")
    ap.add_argument("--lr", type=float, default=5e-2, help="Learning rate for the example config.")
    ap.add_argument(
        "--data-error",
        type=float,
        default=None,
        help="Data sigma for config. Defaults to noise_mk/1000 when noise_mk>0, otherwise 0.005.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device recorded in the example separation config.",
    )
    return ap.parse_args()


def _resolve(work_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else work_root / path


def _parse_freqs(start: float, stop: float, step: float) -> List[float]:
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step) or step <= 0:
        raise ValueError("Invalid frequency range.")
    count = int(round((stop - start) / step)) + 1
    if count <= 0:
        raise ValueError("Frequency range is empty.")
    freqs = [round(start + i * step, 2) for i in range(count)]
    if abs(freqs[-1] - stop) > 1e-5:
        raise ValueError("Frequency range is not aligned with freq_step_mhz.")
    return freqs


def _squeeze_2d(data: np.ndarray, *, path: Path) -> np.ndarray:
    arr = np.asarray(data)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing {path}, got shape {arr.shape}")
    return arr


def _load_2d(path: Path, *, dtype: np.dtype) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = _squeeze_2d(hdul[0].data, path=path)
        return np.asarray(data, dtype=dtype)


def _load_mask2d(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = _squeeze_2d(hdul[0].data, path=path)
        return np.asarray(data > 0.5, dtype=bool)


def _component_paths(component_root: Path) -> Dict[str, List[Path]]:
    roots = {
        "gsync": component_root / "galactic" / "synchrotron",
        "gfree": component_root / "galactic" / "freefree",
        "ptr": component_root / "extragalactic" / "pointsource",
    }
    out: Dict[str, List[Path]] = {}
    for name, root in roots.items():
        paths = sorted(root.glob("*.fits"))
        if not paths:
            raise FileNotFoundError(f"No FITS component maps found under {root}")
        out[name] = paths
    return out


def _center_crop2d(x: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if size > h or size > w:
        raise ValueError(f"cut-size-px={size} is larger than image shape {(h, w)}")
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return x[..., y0 : y0 + size, x0 : x0 + size]


def _center_pad2d(x: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    th, tw = int(shape[0]), int(shape[1])
    if th < h or tw < w:
        raise ValueError(f"pad shape {shape} is smaller than image shape {(h, w)}")
    top = (th - h) // 2
    left = (tw - w) // 2
    out = np.zeros((th, tw), dtype=x.dtype)
    out[top : top + h, left : left + w] = x
    return out, (top, left)


def _center_crop_from(x: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    oh, ow = int(out_shape[0]), int(out_shape[1])
    top = (h - oh) // 2
    left = (w - ow) // 2
    return x[top : top + oh, left : left + ow]


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    cy = int(psf.shape[-2] // 2)
    cx = int(psf.shape[-1] // 2)
    peak = float(psf[cy, cx])
    if not math.isfinite(peak) or abs(peak) < 1e-12:
        peak = float(np.nanmax(np.abs(psf)))
    if not math.isfinite(peak) or abs(peak) < 1e-12:
        raise ValueError("Cannot normalize PSF with zero/non-finite peak.")
    return psf / peak


def _prepare_psf(psf: np.ndarray, *, zero_mean: bool, normalize_peak: bool) -> np.ndarray:
    out = np.asarray(psf)
    if zero_mean:
        out = out - np.mean(out, dtype=np.float64)
    if normalize_peak:
        out = _normalize_psf(out)
    return np.asarray(out, dtype=psf.dtype)


def _fft_convolve2d(image: np.ndarray, psf: np.ndarray, *, pad_to: Optional[int]) -> np.ndarray:
    if image.shape != psf.shape:
        raise ValueError(f"image/psf shape mismatch: {image.shape} vs {psf.shape}")
    in_shape = (int(image.shape[-2]), int(image.shape[-1]))
    if pad_to is not None:
        pad_shape = (int(pad_to), int(pad_to))
        image_work, _ = _center_pad2d(image, pad_shape)
        psf_work, _ = _center_pad2d(psf, pad_shape)
    else:
        image_work = image
        psf_work = psf
        pad_shape = in_shape

    psf_shifted = np.fft.ifftshift(psf_work)
    out = np.fft.irfft2(
        np.fft.rfft2(image_work) * np.fft.rfft2(psf_shifted),
        s=pad_shape,
    )
    if pad_to is not None:
        out = _center_crop_from(out, in_shape)
    return np.asarray(out, dtype=image.dtype)


def _write_cube(path: Path, data: np.ndarray, *, freq_start_mhz: float, freq_step_mhz: float) -> None:
    hdr = fits.Header()
    hdr["CTYPE3"] = ("FREQ", "Frequency axis")
    hdr["CUNIT3"] = ("MHz", "Frequency unit")
    hdr["CRVAL3"] = (float(freq_start_mhz), "Reference frequency at CRPIX3")
    hdr["CDELT3"] = (float(freq_step_mhz), "Frequency step")
    hdr["CRPIX3"] = (1.0, "Reference pixel")
    hdr["BUNIT"] = ("K", "Brightness temperature or linear dirty-image units")
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _open_memmap(path: Path, *, shape: Sequence[int], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    out_dir = _resolve(work_root, args.out_dir)
    dtype = np.dtype(np.float64 if args.dtype == "float64" else np.float32)
    freqs = _parse_freqs(args.freq_start_mhz, args.freq_stop_mhz, args.freq_step_mhz)
    nfreq = len(freqs)

    eor_cube_path = _resolve(work_root, args.eor_cube)
    component_root = _resolve(work_root, args.component_root)
    bright_mask_path = _resolve(work_root, args.bright_mask)
    masked_fg_fallback_path = _resolve(work_root, args.masked_fg_fallback)
    psf_cube_path = _resolve(work_root, args.psf_cube) if args.psf_cube is not None else None

    if not eor_cube_path.exists():
        raise FileNotFoundError(eor_cube_path)
    bright_mask = None
    if not args.disable_bright_mask:
        if not bright_mask_path.exists():
            raise FileNotFoundError(bright_mask_path)
        if not masked_fg_fallback_path.exists():
            raise FileNotFoundError(masked_fg_fallback_path)
        bright_mask = _load_mask2d(bright_mask_path)
    comp_paths = _component_paths(component_root)
    for name, paths in comp_paths.items():
        if len(paths) < nfreq:
            raise FileNotFoundError(f"Component {name} has {len(paths)} maps, need {nfreq}.")

    with fits.open(eor_cube_path, memmap=True) as hdul:
        eor_data = hdul[0].data
        if eor_data.ndim != 3:
            raise ValueError(f"Expected 3D EoR cube in {eor_cube_path}, got {eor_data.shape}")
        if int(eor_data.shape[0]) < nfreq:
            raise ValueError(f"EoR cube has {eor_data.shape[0]} channels, need {nfreq}.")
        sample = _center_crop2d(np.asarray(eor_data[0], dtype=dtype), int(args.cut_size_px))
        out_shape = (nfreq, int(sample.shape[-2]), int(sample.shape[-1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    fg_tmp = tmp_dir / "fg_true.npy"
    eor_tmp = tmp_dir / "eor_true.npy"
    psf_tmp = tmp_dir / "psf_cube.npy"
    dirty_clean_tmp = tmp_dir / "dirty_clean.npy"
    dirty_obs_tmp = tmp_dir / "dirty_obs.npy"
    fg_cube = _open_memmap(fg_tmp, shape=out_shape, dtype=dtype)
    eor_cube = _open_memmap(eor_tmp, shape=out_shape, dtype=dtype)
    psf_cube = _open_memmap(psf_tmp, shape=out_shape, dtype=dtype)
    dirty_clean = _open_memmap(dirty_clean_tmp, shape=out_shape, dtype=dtype)
    dirty_obs = _open_memmap(dirty_obs_tmp, shape=out_shape, dtype=dtype)
    rng = np.random.default_rng(int(args.seed))
    noise_sigma = float(args.noise_mk) / 1000.0

    psf_stack = None
    if psf_cube_path is not None:
        with fits.open(psf_cube_path, memmap=True) as hdul:
            psf_stack = hdul[0].data
            if psf_stack.ndim != 3:
                raise ValueError(f"Expected 3D PSF cube in {psf_cube_path}, got {psf_stack.shape}")
            if int(psf_stack.shape[0]) not in (1, nfreq):
                raise ValueError(f"PSF cube has {psf_stack.shape[0]} channels; expected 1 or {nfreq}.")

    for idx, freq in enumerate(freqs):
        fg = (
            _load_2d(comp_paths["gsync"][idx], dtype=dtype)
            + _load_2d(comp_paths["gfree"][idx], dtype=dtype)
            + _load_2d(comp_paths["ptr"][idx], dtype=dtype)
        )
        with fits.open(eor_cube_path, memmap=True) as hdul:
            eor = np.asarray(hdul[0].data[idx], dtype=dtype)
        if bright_mask is not None:
            with fits.open(masked_fg_fallback_path, memmap=True) as hdul:
                fallback = np.asarray(hdul[0].data[idx], dtype=dtype)
            if fallback.shape != fg.shape:
                raise ValueError(f"Fallback FG shape mismatch: {fallback.shape} vs {fg.shape}")
            if bright_mask.shape != fg.shape:
                raise ValueError(f"Bright mask shape mismatch: {bright_mask.shape} vs {fg.shape}")
            fg = np.where(bright_mask, fallback, fg)
        fg_max = float(np.nanmax(fg))
        if math.isfinite(float(args.fg_max_k_fail)) and fg_max > float(args.fg_max_k_fail):
            raise ValueError(
                f"Processed foreground slice exceeds fg-max-k-fail: "
                f"freq={freq:.2f} MHz max={fg_max:.6g} K limit={float(args.fg_max_k_fail):.6g} K. "
                "This usually means the bright-source mask/fallback was not applied as intended."
            )

        if psf_stack is not None:
            psf_idx = 0 if int(psf_stack.shape[0]) == 1 else idx
            psf = np.asarray(psf_stack[psf_idx], dtype=dtype)
        else:
            psf_path = _resolve(work_root, Path(args.psf_pattern.format(freq=freq)))
            if not psf_path.exists():
                raise FileNotFoundError(psf_path)
            psf = _load_2d(psf_path, dtype=dtype)

        fg = _center_crop2d(fg, int(args.cut_size_px))
        eor = _center_crop2d(eor, int(args.cut_size_px))
        psf = _center_crop2d(psf, int(args.cut_size_px))
        psf = _prepare_psf(
            psf,
            zero_mean=not bool(args.keep_psf_mean),
            normalize_peak=not bool(args.no_normalize_psf_peak),
        )

        fg_cube[idx] = fg
        eor_cube[idx] = eor
        psf_cube[idx] = psf
        dirty_slice = _fft_convolve2d(fg + eor, psf, pad_to=args.psf_pad_to)
        dirty_clean[idx] = dirty_slice
        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma, size=dirty_slice.shape).astype(dtype, copy=False)
            dirty_obs[idx] = dirty_slice + noise
        else:
            dirty_obs[idx] = dirty_slice

        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == nfreq:
            print(f"[{idx + 1:03d}/{nfreq:03d}] freq={freq:.2f} MHz", flush=True)

    for mm in (fg_cube, eor_cube, psf_cube, dirty_clean, dirty_obs):
        mm.flush()

    fg_out = out_dir / "fg_true_cube2_dropcluster.fits"
    eor_out = out_dir / "eor_true_cube2.fits"
    psf_out = out_dir / "psf_cube.fits"
    dirty_clean_out = out_dir / "dirty_clean.fits"
    dirty_obs_out = out_dir / "dirty_obs.fits"
    config_out = out_dir / "config_psf_aware_separation.json"
    manifest_out = out_dir / "manifest.json"

    _write_cube(fg_out, fg_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(eor_out, eor_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(psf_out, psf_cube, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(dirty_clean_out, dirty_clean, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)
    _write_cube(dirty_obs_out, dirty_obs, freq_start_mhz=freqs[0], freq_step_mhz=args.freq_step_mhz)

    data_error = float(args.data_error) if args.data_error is not None else (noise_sigma if noise_sigma > 0 else 0.005)
    config = {
        "input_cube": str(dirty_obs_out),
        "fg_output": str(out_dir / "fg_est_psfaware.fits"),
        "eor_output": str(out_dir / "eor_est_psfaware.fits"),
        "psf_cube": str(psf_out),
        "psf_scale": 1.0,
        "psf_pad_to": int(args.psf_pad_to) if args.psf_pad_to is not None else None,
        "true_eor_cube": str(eor_out),
        "fg_reference_cube": str(fg_out),
        "fg_smooth_prior_source": "reference_cube",
        "use_robust_fg_stats": True,
        "num_iters": int(args.num_iters),
        "lr": float(args.lr),
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "freq_axis": 0,
        "data_error": float(data_error),
        "data_sigma_mode": "explicit",
        "noise_component_enabled": False,
        "noise_weight": 0.0,
        "noise_rms_weight": 0.0,
        "noise_gauss_weight": 0.0,
        "noise_spatial_weight": 0.0,
        "noise_cross_weight": 0.0,
        "noise_ps_flat_weight": 0.0,
        "noise_slice_std_weight": 0.0,
        "en_coherence_weight": 0.0,
        "fg_smooth_mode": "diff2_l2",
        "eor_prior_mean": 0.0,
        "eor_prior_sigma": 0.1,
        "eor_amp_prior_mode": "voxel_deadzone",
        "eor_prior_amp_threshold": 0.0,
        "freq_start_mhz": float(freqs[0]),
        "freq_delta_mhz": float(args.freq_step_mhz),
        "dtype": str(args.dtype),
        "device": str(args.device),
        "print_every": 50,
    }
    config = {k: v for k, v in config.items() if v is not None}
    config_out.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "work_root": str(work_root),
        "out_dir": str(out_dir),
        "freqs_mhz": freqs,
        "shape": list(out_shape),
        "dtype": str(dtype),
        "dataset": "cube2",
        "foreground_mode": "drop_cluster_components_gsync_gfree_pointsource",
        "eor_cube": str(eor_cube_path),
        "component_root": str(component_root),
        "bright_mask": None if args.disable_bright_mask else str(bright_mask_path),
        "masked_fg_fallback": None if args.disable_bright_mask else str(masked_fg_fallback_path),
        "psf_source": str(psf_cube_path) if psf_cube_path is not None else str(args.psf_pattern),
        "psf_normalized_by_peak": not bool(args.no_normalize_psf_peak),
        "psf_zero_mean": not bool(args.keep_psf_mean),
        "psf_pad_to": int(args.psf_pad_to) if args.psf_pad_to is not None else None,
        "noise_mk": float(args.noise_mk),
        "fg_max_k_fail": float(args.fg_max_k_fail),
        "seed": int(args.seed),
        "outputs": {
            "fg_true": str(fg_out),
            "eor_true": str(eor_out),
            "psf_cube": str(psf_out),
            "dirty_clean": str(dirty_clean_out),
            "dirty_obs": str(dirty_obs_out),
            "config": str(config_out),
        },
        "notes": [
            "This Plan-A dataset deliberately avoids the anomalously bright cube1 branch.",
            "The foreground is rebuilt from cube2 component maps and excludes simplified clusters/radio halos.",
            "By default, e2esim bright-source masked pixels are replaced by the postprocessed cube2 FG fallback.",
            "The PSF mean is removed by default to avoid an artificial total-power response.",
            "Direct PSF convolution is performed in the requested numpy dtype, float64 by default.",
            "This is a known-operator dirty-image-domain test, not a full visibility-domain OSKAR inversion.",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not args.keep_temp_npy:
        for tmp_path in (fg_tmp, eor_tmp, psf_tmp, dirty_clean_tmp, dirty_obs_tmp):
            tmp_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    print(f"Wrote Plan-A dataset to {out_dir}")
    print(f"Config: {config_out}")


if __name__ == "__main__":
    main()
