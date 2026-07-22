#!/usr/bin/env python3
"""Rebuild a dense-frequency cached operator and propagate covariance probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, CODE_DIR, CODE_DIR / "code" / "3dnet"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import fit_cached_pca_proxy_cheb_operator_separation as base  # noqa: E402
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_floats(value: str) -> list[float]:
    return [float(piece.strip()) for piece in str(value).split(",") if piece.strip()]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fg-cube-k", type=Path, required=True)
    parser.add_argument("--eor-cube-k", type=Path, required=True)
    parser.add_argument("--cube-freq0-mhz", type=float, default=106.0)
    parser.add_argument("--cube-freq-step-mhz", type=float, default=0.1)
    parser.add_argument("--truth-dirty-pattern", required=True)
    parser.add_argument("--dense-grid-csv-pattern", required=True)
    parser.add_argument("--train-grid-csv-pattern", required=True)
    parser.add_argument("--train-response-pattern", required=True)
    parser.add_argument("--tile-cache-dir", type=Path, required=True)
    parser.add_argument("--tile-cache-meta-path-rewrite", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--response-crop-size", type=int, default=512)
    parser.add_argument("--eval-crop-size", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--train-halo-px", type=int, default=16)
    parser.add_argument("--model-margin", type=int, default=-1)
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--rbf-scale-px", type=float, default=32.0)
    parser.add_argument("--pixel-arcsec", type=float, default=32.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--operator-batch-size", type=int, default=8)
    parser.add_argument("--fg-draw-count", type=int, default=16)
    parser.add_argument("--fg-heldout-seed", type=int, default=2026072201)
    parser.add_argument("--fg-ensemble-seed", type=int, default=2026072301)
    parser.add_argument("--eor-probes-per-length", type=int, default=8)
    parser.add_argument("--eor-probe-seed", type=int, default=2026072401)
    parser.add_argument(
        "--eor-lengths-mhz",
        default="0.06,0.10,0.18,0.32,0.56,1.00,2.00",
    )
    parser.add_argument("--closure-threshold", type=float, default=1e-4)
    parser.add_argument("--progress-every-tile", type=int, default=64)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temporary, np.asarray(values))
    temporary.replace(path)
    return {
        "path": str(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _format(pattern: str, *, label: str, freq: float) -> str:
    return str(pattern).format(
        label=str(label),
        freq=float(freq),
        freqtag=f"{float(freq):.2f}".replace(".", ""),
    )


def _load_intrinsic_cube(
    path: Path,
    frequencies: Sequence[float],
    *,
    cube_freq0_mhz: float,
    cube_freq_step_mhz: float,
    image_size: int,
) -> np.ndarray:
    indices = [
        int(round((float(freq) - float(cube_freq0_mhz)) / float(cube_freq_step_mhz)))
        for freq in frequencies
    ]
    with fits.open(path, memmap=True) as hdul:
        source = hdul[0].data
        if min(indices) < 0 or max(indices) >= source.shape[0]:
            raise IndexError(f"frequency indices {min(indices)}..{max(indices)} outside {source.shape}")
        rows = [np.asarray(source[index], dtype=np.float64) for index in indices]
    cube = np.stack(rows, axis=0)
    if cube.shape[1:] != (int(image_size), int(image_size)):
        raise ValueError(f"intrinsic cube shape {cube.shape} does not match image size {image_size}")
    return cube


def _load_exact_dirty(
    pattern: str,
    label: str,
    frequencies: Sequence[float],
    eval_size: int,
) -> np.ndarray:
    rows = []
    for freq in frequencies:
        path = Path(_format(pattern, label=label, freq=float(freq)))
        data = np.squeeze(np.asarray(fits.getdata(path), dtype=np.float64))
        if data.ndim != 2:
            raise ValueError(f"expected 2D dirty image: {path}: {data.shape}")
        start_y = (data.shape[0] - int(eval_size)) // 2
        start_x = (data.shape[1] - int(eval_size)) // 2
        rows.append(data[start_y : start_y + int(eval_size), start_x : start_x + int(eval_size)])
    return np.stack(rows, axis=0)


def _sky_k_to_jy(cubes_k: np.ndarray, frequencies: Sequence[float], pixel_arcsec: float) -> np.ndarray:
    scales = np.asarray(
        [base._k_to_jy_per_pixel(float(freq), float(pixel_arcsec)) for freq in frequencies],
        dtype=np.float64,
    )
    return np.asarray(cubes_k, dtype=np.float64) * scales.reshape((1,) * (cubes_k.ndim - 3) + (-1, 1, 1))


def _normal_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    scales_px: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    radius2 = ky * ky + kx * kx
    spectrum_filter = np.zeros_like(radius2, dtype=np.float64)
    for scale, weight in zip(scales_px, weights):
        spectrum_filter += float(weight) * np.exp(-2.0 * math.pi**2 * float(scale) ** 2 * radius2)
    white = rng.normal(size=shape)
    field = np.fft.irfft2(np.fft.rfft2(white) * spectrum_filter, s=shape)
    field -= float(np.mean(field))
    rms = float(np.sqrt(np.mean(field * field)))
    return field / max(rms, np.finfo(np.float64).tiny)


def _foreground_unit_error(
    foreground_k: np.ndarray,
    frequencies: Sequence[float],
    seed: int,
) -> np.ndarray:
    """One unit draw; multiplying by p gives p fractional amplitude RMS."""
    rng = np.random.default_rng(int(seed))
    shape = tuple(int(value) for value in foreground_k.shape[-2:])
    broad_scales = (1.0, 4.0, 16.0, 64.0)
    broad_weights = (0.35, 0.30, 0.25, 0.10)
    amplitude = _normal_field(rng, shape, broad_scales, broad_weights)
    slope = _normal_field(rng, shape, broad_scales, broad_weights)
    shift_x = _normal_field(rng, shape, (4.0, 16.0, 64.0), (0.4, 0.4, 0.2))
    shift_y = _normal_field(rng, shape, (4.0, 16.0, 64.0), (0.4, 0.4, 0.2))
    confusion = _normal_field(rng, shape, (0.5, 1.0, 2.0), (0.5, 0.3, 0.2))
    frequencies_array = np.asarray(frequencies, dtype=np.float64)
    reference = float(np.exp(np.mean(np.log(frequencies_array))))
    log_frequency = np.log(frequencies_array / reference)
    grad_y, grad_x = np.gradient(foreground_k, axis=(-2, -1))
    foreground_rms = np.sqrt(np.mean(np.square(foreground_k), axis=(-2, -1)))
    return (
        foreground_k * amplitude[None, :, :]
        + 2.0 * foreground_k * log_frequency[:, None, None] * slope[None, :, :]
        + 4.0 * (grad_x * shift_x[None, :, :] + grad_y * shift_y[None, :, :])
        + 0.3 * foreground_rms[:, None, None] * confusion[None, :, :]
    )


def _eor_probe_batch(
    frequencies: Sequence[float],
    *,
    count: int,
    image_size: int,
    ell_mhz: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    frequency = np.asarray(frequencies, dtype=np.float64)
    correlation = np.exp(-np.abs(frequency[:, None] - frequency[None, :]) / float(ell_mhz))
    factor = np.linalg.cholesky(correlation + 1e-10 * np.eye(frequency.size))
    white = rng.normal(size=(int(count), frequency.size, int(image_size), int(image_size)))
    probes = np.einsum("fg,bgxy->bfxy", factor, white, optimize=True)
    probes -= np.mean(probes, axis=(-2, -1), keepdims=True)
    return np.asarray(probes, dtype=np.float64)


def _apply_operator(
    operator: base.CachedPcaProxyForward,
    cubes_jy: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(cubes_jy, dtype=np.float64)
    if values.ndim == 3:
        values = values[None, ...]
    output = np.empty(
        (values.shape[0], operator.n_freq, operator.eval_size, operator.eval_size),
        dtype=np.float64,
    )
    for start in range(0, values.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), values.shape[0])
        started = time.monotonic()
        tensor = torch.as_tensor(values[start:stop], device=operator.device, dtype=operator.dtype)
        with torch.no_grad():
            prediction = operator(tensor)
        output[start:stop] = prediction.detach().cpu().numpy()
        del tensor, prediction
        if operator.device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "operator_batch_done",
                    "batch_start": int(start),
                    "batch_stop": int(stop),
                    "elapsed_seconds": float(time.monotonic() - started),
                    "time_utc": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return output


def _closure(exact: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(predicted) - np.asarray(exact)
    exact_rms = float(np.sqrt(np.mean(np.square(exact))))
    per_frequency = []
    for index in range(exact.shape[0]):
        denominator = float(np.sqrt(np.mean(np.square(exact[index]))))
        numerator = float(np.sqrt(np.mean(np.square(residual[index]))))
        per_frequency.append(numerator / max(denominator, np.finfo(np.float64).tiny))
    return {
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(exact), 1e-300)),
        "residual_over_exact_rms": float(
            np.sqrt(np.mean(np.square(residual))) / max(exact_rms, 1e-300)
        ),
        "per_frequency_residual_over_exact_rms": per_frequency,
        "per_frequency_max": float(np.max(per_frequency)),
    }


def _cache_signature(cache_dir: Path) -> dict[str, Any]:
    paths = sorted(cache_dir.glob("*.npz"))
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        size = int(path.stat().st_size)
        total += size
        digest.update(f"{path.name}\t{size}\n".encode("utf-8"))
    return {
        "file_count": len(paths),
        "total_size_bytes": total,
        "filename_size_sha256": digest.hexdigest(),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    frequencies = [float(value) for value in resolved.geometry["frequencies_mhz"]]
    if int(args.eval_crop_size) != int(config["image_geometry"]["eval_crop_size"]):
        raise ValueError("eval crop does not match PS2D config")
    lengths = _parse_floats(args.eor_lengths_mhz)
    if len(lengths) < 2 or any(value <= 0.0 for value in lengths):
        raise ValueError("at least two positive EoR lengths are required")
    if int(args.fg_draw_count) < 4 or int(args.eor_probes_per_length) < 4:
        raise ValueError("covariance banks require at least four probes")

    cache_signature = _cache_signature(args.tile_cache_dir)
    expected_cache_count = len(frequencies) * (int(args.image_size) // int(args.tile_size)) ** 2
    if cache_signature["file_count"] != expected_cache_count:
        raise ValueError(
            f"cache has {cache_signature['file_count']} files; expected {expected_cache_count}"
        )
    path_rewrites = base._parse_path_rewrites(args.tile_cache_meta_path_rewrite)
    caches: list[base.PcaTileCache] = []
    for frequency_index, frequency in enumerate(frequencies):
        frequency_caches = base._build_freq_tile_caches(
            freq_index=int(frequency_index),
            freq_mhz=float(frequency),
            dense_grid_csv=Path(base._format_pattern(args.dense_grid_csv_pattern, freq=float(frequency))),
            train_grid_csv=Path(base._format_pattern(args.train_grid_csv_pattern, freq=float(frequency))),
            train_response_pattern=str(args.train_response_pattern),
            image_size=int(args.image_size),
            response_crop_size=int(args.response_crop_size),
            eval_size=int(args.eval_crop_size),
            tile_size=int(args.tile_size),
            train_halo_px=int(args.train_halo_px),
            model_margin_arg=int(args.model_margin),
            pca_rank=int(args.pca_rank),
            rbf_scale_px=float(args.rbf_scale_px),
            dtype=np.dtype(np.float64),
            progress_every_tile=int(args.progress_every_tile),
            max_tiles=0,
            tile_cache_dir=args.tile_cache_dir,
            tile_cache_refresh=False,
            tile_cache_meta_path_rewrites=path_rewrites,
            preload_train_responses=False,
            cache_build_only=False,
        )
        expected_tiles = (int(args.image_size) // int(args.tile_size)) ** 2
        if len(frequency_caches) != expected_tiles:
            raise ValueError(f"frequency {frequency:.2f} loaded {len(frequency_caches)} tiles")
        caches.extend(frequency_caches)
    operator = base.CachedPcaProxyForward(
        caches,
        n_freq=len(frequencies),
        eval_size=int(args.eval_crop_size),
        device=torch.device(str(args.device)),
        dtype=torch.float64,
        keep_kernel_fft_on_device=False,
        checkpoint_tiles=False,
    )
    print(
        json.dumps(
            {
                "event": "dense_frequency_operator_loaded",
                "frequency_count": len(frequencies),
                "tile_count": len(caches),
                "device": str(args.device),
                "frequency_interpolation": False,
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    foreground_k = _load_intrinsic_cube(
        args.fg_cube_k,
        frequencies,
        cube_freq0_mhz=float(args.cube_freq0_mhz),
        cube_freq_step_mhz=float(args.cube_freq_step_mhz),
        image_size=int(args.image_size),
    )
    eor_k = _load_intrinsic_cube(
        args.eor_cube_k,
        frequencies,
        cube_freq0_mhz=float(args.cube_freq0_mhz),
        cube_freq_step_mhz=float(args.cube_freq_step_mhz),
        image_size=int(args.image_size),
    )
    exact_fg = _load_exact_dirty(args.truth_dirty_pattern, "fg", frequencies, int(args.eval_crop_size))
    exact_eor = _load_exact_dirty(args.truth_dirty_pattern, "eor", frequencies, int(args.eval_crop_size))
    predicted_truth = _apply_operator(
        operator,
        _sky_k_to_jy(np.stack([foreground_k, eor_k]), frequencies, float(args.pixel_arcsec)),
        batch_size=min(2, int(args.operator_batch_size)),
    )
    closure = {
        "foreground": _closure(exact_fg, predicted_truth[0]),
        "eor": _closure(exact_eor, predicted_truth[1]),
    }
    maximum_closure = max(
        closure["foreground"]["relative_l2"], closure["eor"]["relative_l2"]
    )
    if maximum_closure > float(args.closure_threshold):
        raise RuntimeError(
            f"rebuilt operator closure {maximum_closure:.6e} exceeds {args.closure_threshold:.6e}"
        )

    products: dict[str, Any] = {}
    products["exact_dirty_fg"] = _save_array(args.out_dir / "exact_dirty_fg.npy", exact_fg)
    products["exact_dirty_eor"] = _save_array(args.out_dir / "exact_dirty_eor.npy", exact_eor)
    products["operator_dirty_fg"] = _save_array(
        args.out_dir / "operator_dirty_fg.npy", predicted_truth[0]
    )
    products["operator_dirty_eor"] = _save_array(
        args.out_dir / "operator_dirty_eor.npy", predicted_truth[1]
    )

    foreground_errors = [
        _foreground_unit_error(foreground_k, frequencies, int(args.fg_heldout_seed))
    ]
    foreground_errors.extend(
        _foreground_unit_error(
            foreground_k, frequencies, int(args.fg_ensemble_seed) + draw_index
        )
        for draw_index in range(int(args.fg_draw_count))
    )
    foreground_error_dirty = _apply_operator(
        operator,
        _sky_k_to_jy(np.stack(foreground_errors), frequencies, float(args.pixel_arcsec)),
        batch_size=int(args.operator_batch_size),
    )
    products["fg_error_heldout_unit"] = _save_array(
        args.out_dir / "fg_error_heldout_unit.npy",
        foreground_error_dirty[0].astype(np.float32),
    )
    products["fg_error_draws_unit"] = _save_array(
        args.out_dir / "fg_error_draws_unit.npy",
        foreground_error_dirty[1:].astype(np.float32),
    )
    del foreground_errors, foreground_error_dirty

    eor_probe_products: dict[str, Any] = {}
    for length_index, ell_mhz in enumerate(lengths):
        probes_k = _eor_probe_batch(
            frequencies,
            count=int(args.eor_probes_per_length),
            image_size=int(args.image_size),
            ell_mhz=float(ell_mhz),
            seed=int(args.eor_probe_seed) + 1000 * int(length_index),
        )
        probes_dirty = _apply_operator(
            operator,
            _sky_k_to_jy(probes_k, frequencies, float(args.pixel_arcsec)),
            batch_size=int(args.operator_batch_size),
        )
        label = f"ell_{float(ell_mhz):.4f}".replace(".", "p")
        eor_probe_products[f"{float(ell_mhz):.8g}"] = _save_array(
            args.out_dir / f"eor_probes_{label}.npy",
            probes_dirty.astype(np.float32),
        )
        del probes_k, probes_dirty
    products["eor_probes"] = eor_probe_products

    manifest = {
        "schema": "partial_window_covariance_operator_bank",
        "schema_version": 1,
        "created_at": _now(),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "frequencies_mhz": frequencies,
        "operator_identity": {
            "kind": "per_frequency_stride4_rank64_pca_response",
            "frequency_interpolation": False,
            "spatial_training_stride_px": 4,
            "pca_rank": int(args.pca_rank),
            "rbf_scale_px": float(args.rbf_scale_px),
            "tile_size": int(args.tile_size),
            "tile_count_per_frequency": (int(args.image_size) // int(args.tile_size)) ** 2,
            "tile_cache_dir": str(args.tile_cache_dir),
            "tile_cache_signature": cache_signature,
            "dense_grid_csv_pattern": str(args.dense_grid_csv_pattern),
            "train_grid_csv_pattern": str(args.train_grid_csv_pattern),
            "train_response_pattern": str(args.train_response_pattern),
            "dtype": "float64",
        },
        "closure": closure,
        "closure_threshold": float(args.closure_threshold),
        "foreground_prior_emulator": {
            "interpretation": "simulation template plus independent observation-like error draw",
            "unit_scale": "multiply a unit draw by p for p fractional amplitude RMS",
            "amplitude_sigma_over_p": 1.0,
            "spectral_index_sigma_over_p": 2.0,
            "astrometric_shift_sigma_px_over_p": 4.0,
            "unresolved_confusion_rms_over_template_rms_over_p": 0.3,
            "heldout_seed": int(args.fg_heldout_seed),
            "ensemble_seed_start": int(args.fg_ensemble_seed),
            "ensemble_draw_count": int(args.fg_draw_count),
            "estimator_does_not_receive_intrinsic_foreground_truth": True,
        },
        "eor_covariance_probes": {
            "kernel": "exponential_frequency_correlation",
            "lengths_mhz": lengths,
            "probes_per_length": int(args.eor_probes_per_length),
            "spatial_prior": "zero_mean_stationary_flat_transverse_probe; per-kperp amplitude remains free",
            "input_rms_k": 1.0,
            "seed": int(args.eor_probe_seed),
            "uses_eor_simulation_truth": False,
        },
        "products": products,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "partial_window_covariance_bank_done",
                "out_dir": str(args.out_dir),
                "closure": closure,
                "elapsed_seconds": manifest["elapsed_seconds"],
                "time_utc": _now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
