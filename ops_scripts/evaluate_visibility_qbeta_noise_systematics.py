#!/usr/bin/env python3
"""Evaluate split-noise and gain-residual robustness of frozen Q_beta windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402

from ops_scripts.calibrate_visibility_qbeta_noiseless import (  # noqa: E402
    _load_bank,
    _maximum_patch_delays,
    _row_kperp,
    _visibility_bandpowers,
)


SKAO_SEFD_SOURCE_URL = (
    "https://gitlab.com/ska-telescope/ost/ska-ost-senscalc/-/blob/"
    "master/src/ska_ost_senscalc/static/lookups/"
    "ska_station_sensitivity_AAVS2.h5"
)
SKAO_SEFD_SOURCE_REVISION = "f2905865f5d276b46dfc2f7ac9861e16de0772a0"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-npz", type=Path, required=True)
    parser.add_argument("--combined-json", type=Path, required=True)
    parser.add_argument("--coarse-npz", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sefd-h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    parser.add_argument("--realizations", type=int, default=512)
    parser.add_argument("--gain-realizations", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--integration-hours", action="append", type=float, default=[]
    )
    parser.add_argument(
        "--gain-rms", action="append", type=float, default=[]
    )
    parser.add_argument(
        "--gain-profile",
        action="append",
        choices=("smooth", "ripple"),
        default=[],
    )
    parser.add_argument(
        "--target-significance", action="append", type=float, default=[]
    )
    parser.add_argument("--thermal-seed", type=int, default=2026081701)
    parser.add_argument("--gain-seed", type=int, default=2026081702)
    parser.add_argument("--reference-k-h-mpc", type=float, default=0.2)
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--az-deg", type=float, default=100.64)
    parser.add_argument("--el-deg", type=float, default=50.41)
    parser.add_argument("--start-lst-hour", type=float, default=21.009)
    parser.add_argument("--end-lst-hour", type=float, default=21.098)
    parser.add_argument("--gain-ripple-cycles", type=float, default=2.0)
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def lookup_ska_low_stokes_i_sefd(
    path: Path,
    *,
    frequencies_mhz: np.ndarray,
    az_deg: float,
    el_deg: float,
    start_lst_hour: float,
    end_lst_hour: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the public SKAO nearest-cell and spline SEFD lookup."""
    import h5py
    from scipy.interpolate import UnivariateSpline

    with h5py.File(path, "r") as handle:
        sefd = np.asarray(handle["sefd"][:], dtype=np.float64)
        coarse_frequencies = np.asarray(
            handle["dimensions/frequency"][:], dtype=np.float64
        )
        azimuths = np.asarray(handle["dimensions/azimuth"][:], dtype=np.float64)
        zenith_angles = np.asarray(
            handle["dimensions/zenith_angle"][:], dtype=np.float64
        )
        lsts = np.asarray(handle["dimensions/lst"][:], dtype=np.float64)

    zenith_angle = 90.0 - float(el_deg)
    if zenith_angle <= 2.5:
        azimuth_index = 0
        zenith_index = 0
    else:
        azimuth_index = int(np.argmin(np.abs(azimuths - float(az_deg))))
        zenith_index = int(
            np.argmin(np.abs(zenith_angles - float(zenith_angle)))
        )
    start_index = int(np.argmin(np.abs(lsts - float(start_lst_hour))))
    end_index = int(np.argmin(np.abs(lsts - float(end_lst_hour))))
    selected_lst = np.zeros(lsts.size, dtype=bool)
    if start_index < end_index:
        selected_lst[start_index:end_index] = True
    elif start_index == end_index:
        selected_lst[start_index] = True
    else:
        selected_lst[start_index:] = True
        selected_lst[:end_index] = True
    values = sefd[
        selected_lst, azimuth_index, zenith_index, :, :
    ]

    def effective(component: np.ndarray) -> np.ndarray:
        return np.sqrt(component.shape[0]) / np.sqrt(
            np.sum(1.0 / np.square(component), axis=0)
        )

    sefd_x = UnivariateSpline(
        coarse_frequencies, effective(values[..., 0]), s=0
    )(frequencies_mhz)
    sefd_y = UnivariateSpline(
        coarse_frequencies, effective(values[..., 1]), s=0
    )(frequencies_mhz)
    stokes_i = 0.5 * np.sqrt(np.square(sefd_x) + np.square(sefd_y))
    return np.asarray(stokes_i, dtype=np.float64), {
        "requested_az_deg": float(az_deg),
        "requested_el_deg": float(el_deg),
        "requested_start_lst_hour": float(start_lst_hour),
        "requested_end_lst_hour": float(end_lst_hour),
        "selected_az_deg": float(azimuths[azimuth_index]),
        "selected_zenith_angle_deg": float(zenith_angles[zenith_index]),
        "selected_lst_hours": lsts[selected_lst],
    }


def thermal_noise_sigma_per_real_component(
    sefd_jy: np.ndarray,
    *,
    channel_bandwidth_hz: float,
    total_integration_hours: float,
    time_step_count: int,
    split_count: int = 2,
) -> tuple[np.ndarray, float]:
    """Return per-split real/imag visibility noise for a repeated LST track."""
    if (
        channel_bandwidth_hz <= 0.0
        or total_integration_hours <= 0.0
        or time_step_count <= 0
        or split_count < 2
    ):
        raise ValueError("Invalid thermal-noise integration contract")
    seconds_per_row_per_split = (
        float(total_integration_hours)
        * 3600.0
        / (int(time_step_count) * int(split_count))
    )
    sigma = np.asarray(sefd_jy, dtype=np.float64) / np.sqrt(
        2.0 * float(channel_bandwidth_hz) * seconds_per_row_per_split
    )
    return sigma, seconds_per_row_per_split


def _complex_noise(
    rng: np.random.Generator,
    *,
    size: tuple[int, int, int],
    sigma_per_frequency: np.ndarray,
) -> np.ndarray:
    scale = np.asarray(sigma_per_frequency, dtype=np.float64)[None, :, None]
    return scale * (
        rng.standard_normal(size) + 1j * rng.standard_normal(size)
    )


def _coarse_estimate(
    q_values: np.ndarray,
    products: dict[str, np.ndarray],
    profile: str,
) -> np.ndarray:
    values = np.asarray(q_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("Q values must have shape [output] or [batch,output]")
    prefix = f"{profile}_"
    transform = np.asarray(products[prefix + "transform"], dtype=np.float64)
    response = np.asarray(products[prefix + "response"], dtype=np.float64)
    if values.shape[1] != transform.shape[1]:
        raise ValueError("Q values and coarse transform differ")
    row_sum = np.sum(response, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError("Coarse response has a nonpositive row sum")
    estimate = (values @ transform.T) / row_sum[None, :]
    return estimate[0] if squeeze else estimate


def _weighted_metrics_batch(
    estimates: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(estimates, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metric_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    ratios = (values @ metric_weights) / float(metric_weights @ truth)
    denominator = float(metric_weights @ np.square(truth))
    relative_l2 = np.sqrt(
        np.sum(metric_weights[None, :] * np.square(values - truth), axis=1)
        / denominator
    )
    return ratios, relative_l2


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "median": float(np.median(data)),
        "p05": float(np.quantile(data, 0.05)),
        "p95": float(np.quantile(data, 0.95)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _mean_metrics(
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    ratios, l2 = _weighted_metrics_batch(samples, target, weights)
    mean_ratio, mean_l2 = _weighted_metrics_batch(
        np.mean(samples, axis=0, keepdims=True), target, weights
    )
    return {
        "mean_estimate_integrated_power_ratio": float(mean_ratio[0]),
        "mean_estimate_relative_l2": float(mean_l2[0]),
        "per_realization_integrated_power_ratio": _distribution(ratios),
        "per_realization_relative_l2": _distribution(l2),
    }


def _foreground_contamination_metrics(
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    metric_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    denominator = max(
        float(np.sum(metric_weights * np.abs(truth))), 1e-300
    )
    integrated = (
        np.sum(metric_weights[None, :] * np.abs(values), axis=1)
        / denominator
    )
    mean = np.mean(values, axis=0)
    cell_fraction = np.abs(mean) / np.maximum(np.abs(truth), 1e-300)
    return {
        "mean_integrated_absolute_foreground_to_target": float(
            np.mean(integrated)
        ),
        "integrated_absolute_foreground_to_target_distribution": (
            _distribution(integrated)
        ),
        "maximum_mean_cell_foreground_to_target": float(
            np.max(cell_fraction)
        ),
        "median_mean_cell_foreground_to_target": float(
            np.median(cell_fraction)
        ),
        "p90_mean_cell_foreground_to_target": float(
            np.quantile(cell_fraction, 0.9)
        ),
    }


def _coverage(
    samples: np.ndarray,
    expectation: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    expected = np.asarray(expectation, dtype=np.float64).reshape(-1)
    split = values.shape[0] // 2
    if split < 2 or values.shape[0] - split < 2:
        raise ValueError("Coverage requires at least four realizations")
    train = values[:split]
    test = values[split:]
    sigma = np.std(train, axis=0, ddof=1)
    valid = sigma > np.finfo(np.float64).tiny
    pulls = (test[:, valid] - expected[valid]) / sigma[valid][None, :]
    return {
        "training_realizations": int(split),
        "evaluation_realizations": int(values.shape[0] - split),
        "valid_group_count": int(np.count_nonzero(valid)),
        "pooled_pull_mean": float(np.mean(pulls)),
        "pooled_pull_std": float(np.std(pulls, ddof=1)),
        "coverage_68": float(np.mean(np.abs(pulls) <= 1.0)),
        "coverage_95": float(np.mean(np.abs(pulls) <= 1.959963984540054)),
        "per_group_sigma": sigma,
    }


def _gain_products(
    rng: np.random.Generator,
    *,
    frequencies_mhz: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    realization_count: int,
    rms: float,
    profile: str,
    ripple_cycles: float,
) -> np.ndarray:
    station_count = int(max(np.max(antenna1), np.max(antenna2)) + 1)
    x = np.linspace(-1.0, 1.0, frequencies_mhz.size, dtype=np.float64)
    if str(profile) == "smooth":
        basis = np.stack((np.ones_like(x), x, 0.5 * (3.0 * x * x - 1.0)))
        amplitude = np.einsum(
            "rsc,cf->rfs",
            rng.standard_normal((realization_count, station_count, 3)),
            basis,
        )
        phase = np.einsum(
            "rsc,cf->rfs",
            rng.standard_normal((realization_count, station_count, 3)),
            basis,
        )
    elif str(profile) == "ripple":
        argument = math.pi * float(ripple_cycles) * (x + 1.0)
        amplitude_phase = rng.uniform(
            0.0, 2.0 * math.pi, size=(realization_count, 1, station_count)
        )
        phase_phase = rng.uniform(
            0.0, 2.0 * math.pi, size=(realization_count, 1, station_count)
        )
        amplitude = np.sin(argument[None, :, None] + amplitude_phase)
        phase = np.sin(argument[None, :, None] + phase_phase)
    else:
        raise ValueError(f"Unsupported gain profile: {profile}")

    def unit_rms(field: np.ndarray) -> np.ndarray:
        centered = field - np.mean(field, axis=(1, 2), keepdims=True)
        scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2), keepdims=True))
        return centered / scale

    log_gain = float(rms) * unit_rms(amplitude) + 1j * float(rms) * unit_rms(phase)
    gains = np.exp(log_gain)
    return gains[:, :, antenna1] * np.conjugate(gains[:, :, antenna2])


def _cross_q(
    left: np.ndarray,
    right: np.ndarray,
    *,
    output_band_ids: np.ndarray,
    bandpower_kwargs: dict[str, Any],
) -> np.ndarray:
    values, _, _, _, _ = _visibility_bandpowers(
        visibilities=left,
        cross_visibilities=right,
        **bandpower_kwargs,
    )
    return np.asarray(values, dtype=np.float64).reshape(left.shape[0], -1)[
        :, output_band_ids
    ]


def _thermal_samples(
    *,
    foreground_vis: np.ndarray,
    eor_vis: np.ndarray,
    eor_q: np.ndarray,
    foreground_eor_cross_q: np.ndarray,
    sigma_per_frequency: np.ndarray,
    realization_count: int,
    chunk_size: int,
    seed: int,
    output_band_ids: np.ndarray,
    bandpower_kwargs: dict[str, Any],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    output_count = int(output_band_ids.size)
    products = {
        name: np.empty((realization_count, output_count), dtype=np.float64)
        for name in ("noise", "null", "eor_only", "total_linear")
    }
    frequency_count, row_count = foreground_vis.shape
    for first in range(0, realization_count, chunk_size):
        stop = min(realization_count, first + chunk_size)
        count = stop - first
        shape = (count, frequency_count, row_count)
        noise_a = _complex_noise(
            rng, size=shape, sigma_per_frequency=sigma_per_frequency
        )
        noise_b = _complex_noise(
            rng, size=shape, sigma_per_frequency=sigma_per_frequency
        )
        left = np.concatenate(
            (
                noise_a,
                foreground_vis[None, ...] + noise_a,
                foreground_vis[None, ...] + eor_vis[None, ...] + noise_a,
                foreground_vis[None, ...] - eor_vis[None, ...] + noise_a,
            ),
            axis=0,
        )
        right = np.concatenate(
            (
                noise_b,
                foreground_vis[None, ...] + noise_b,
                foreground_vis[None, ...] + eor_vis[None, ...] + noise_b,
                foreground_vis[None, ...] - eor_vis[None, ...] + noise_b,
            ),
            axis=0,
        )
        q = _cross_q(
            left,
            right,
            output_band_ids=output_band_ids,
            bandpower_kwargs=bandpower_kwargs,
        ).reshape(4, count, output_count)
        linear = 0.5 * (q[2] - q[3])
        products["noise"][first:stop] = q[0]
        products["null"][first:stop] = q[1]
        products["eor_only"][first:stop] = (
            q[0] + linear - foreground_eor_cross_q[None, :] + eor_q[None, :]
        )
        products["total_linear"][first:stop] = linear
    return products


def _factor_samples(
    thermal: dict[str, np.ndarray],
    eor_q: np.ndarray,
    factor: float,
) -> np.ndarray:
    return (
        thermal["null"]
        + float(factor) * thermal["total_linear"]
        + float(factor * factor) * eor_q[None, :]
    )


def _solve_significance_factor(
    *,
    thermal: dict[str, np.ndarray],
    eor_q: np.ndarray,
    target: np.ndarray,
    reference_group: int,
    requested_significance: float,
) -> tuple[float, float]:
    def significance(log_factor: float) -> float:
        factor = math.exp(log_factor)
        samples = _factor_samples(thermal, eor_q, factor)
        sigma = float(np.std(samples[:, reference_group], ddof=1))
        return abs(float(factor * factor * target[reference_group])) / max(
            sigma, 1e-300
        )

    low = math.log(1e-6)
    high = math.log(1e8)
    if significance(high) < float(requested_significance):
        raise ValueError("Requested significance is outside the factor bracket")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if significance(middle) < float(requested_significance):
            low = middle
        else:
            high = middle
    factor = math.exp(high)
    return factor, significance(high)


def _coarse_significance_components(
    *,
    thermal: dict[str, np.ndarray],
    eor_q: np.ndarray,
    coarse: dict[str, np.ndarray],
    profile: str,
    selected: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    coarse_thermal = {
        key: _coarse_estimate(thermal[key], coarse, profile)[:, selected]
        for key in ("null", "total_linear")
    }
    coarse_eor = _coarse_estimate(eor_q, coarse, profile)[selected]
    return coarse_thermal, coarse_eor


def _group_coordinates(
    *,
    combined: dict[str, np.ndarray],
    coarse: dict[str, np.ndarray],
    profile: str,
    kperp_edges: np.ndarray,
    kpar_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"{profile}_"
    transform = np.asarray(coarse[prefix + "transform"], dtype=np.float64)
    output_ids = np.asarray(combined["output_band_ids"], dtype=np.int64)
    kpar_count = int(np.asarray(combined["support"]).shape[1])
    group_kperp = np.empty(transform.shape[0], dtype=np.float64)
    group_kpar = np.empty(transform.shape[0], dtype=np.float64)
    for index, row in enumerate(transform):
        positions = np.flatnonzero(np.abs(row) > np.max(np.abs(row)) * 1e-12)
        cells = output_ids[positions]
        kperp_indices = cells // kpar_count
        kpar_indices = cells % kpar_count
        centers = 0.5 * (
            kperp_edges[kperp_indices] + kperp_edges[kperp_indices + 1]
        )
        group_kperp[index] = float(np.mean(centers))
        group_kpar[index] = float(np.mean(kpar_values[kpar_indices]))
    return group_kperp, group_kpar, np.hypot(group_kperp, group_kpar)


def _systematic_samples(
    *,
    foreground_vis: np.ndarray,
    eor_vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    frequencies_mhz: np.ndarray,
    sigma_per_frequency: np.ndarray,
    realization_count: int,
    chunk_size: int,
    rms: float,
    profile: str,
    ripple_cycles: float,
    seed: int,
    output_band_ids: np.ndarray,
    bandpower_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    output_count = int(output_band_ids.size)
    foreground_only = np.empty(
        (realization_count, output_count), dtype=np.float64
    )
    gain_only = np.empty_like(foreground_only)
    gain_thermal = np.empty_like(foreground_only)
    signal = foreground_vis + eor_vis
    frequency_count, row_count = signal.shape
    for first in range(0, realization_count, chunk_size):
        stop = min(realization_count, first + chunk_size)
        count = stop - first
        gains = _gain_products(
            rng,
            frequencies_mhz=frequencies_mhz,
            antenna1=antenna1,
            antenna2=antenna2,
            realization_count=count,
            rms=rms,
            profile=profile,
            ripple_cycles=ripple_cycles,
        )
        shape = (count, frequency_count, row_count)
        noise_a = _complex_noise(
            rng, size=shape, sigma_per_frequency=sigma_per_frequency
        )
        noise_b = _complex_noise(
            rng, size=shape, sigma_per_frequency=sigma_per_frequency
        )
        signal_with_gain = gains * signal[None, ...]
        foreground_with_gain = gains * foreground_vis[None, ...]
        foreground_only[first:stop] = _cross_q(
            foreground_with_gain,
            foreground_with_gain,
            output_band_ids=output_band_ids,
            bandpower_kwargs=bandpower_kwargs,
        )
        gain_only[first:stop] = _cross_q(
            signal_with_gain,
            signal_with_gain,
            output_band_ids=output_band_ids,
            bandpower_kwargs=bandpower_kwargs,
        )
        gain_thermal[first:stop] = _cross_q(
            gains * (signal[None, ...] + noise_a),
            gains * (signal[None, ...] + noise_b),
            output_band_ids=output_band_ids,
            bandpower_kwargs=bandpower_kwargs,
        )
    return foreground_only, gain_only, gain_thermal


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.realizations < 8 or args.gain_realizations < 8 or args.chunk_size < 1:
        raise ValueError("Realization counts must be >=8 and chunk size positive")
    integration_hours = np.asarray(
        args.integration_hours or [100.0, 1000.0], dtype=np.float64
    )
    gain_rms_values = np.asarray(
        args.gain_rms or [1e-4, 3e-4, 1e-3, 3e-3], dtype=np.float64
    )
    gain_profiles = args.gain_profile or ["smooth", "ripple"]
    target_significances = np.asarray(
        args.target_significance or [10.0, 25.0], dtype=np.float64
    )
    if (
        np.any(integration_hours <= 0.0)
        or np.any(gain_rms_values <= 0.0)
        or np.any(target_significances <= 0.0)
    ):
        raise ValueError(
            "Integration, gain RMS, and significance values must be positive"
        )

    combined = _load_npz(args.combined_npz)
    coarse = _load_npz(args.coarse_npz)
    metadata = json.loads(args.combined_json.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    frequencies_mhz = np.asarray(combined["input_frequencies_mhz"], dtype=np.float64)
    frequencies_hz = frequencies_mhz * 1e6
    bank, bank_manifest = _load_bank(
        args.bank_dir, requested_frequencies_hz=frequencies_hz
    )
    selected_rows = np.asarray(combined["selected_bank_rows"], dtype=np.int64)
    foreground_vis = np.asarray(
        bank["sample_fg"][:, selected_rows], dtype=np.complex128
    )
    eor_vis = np.asarray(
        bank["sample_eor"][:, selected_rows], dtype=np.complex128
    )
    row_kperp = _row_kperp(
        np.asarray(bank["sample_uvw_m"])[selected_rows],
        reference_frequency_hz=float(resolved.geometry["reference_frequency_mhz"])
        * 1e6,
        transverse_distance_mpc=float(resolved.geometry["transverse_distance_mpc"]),
    )
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    radial_mpc_per_hz = float(resolved.geometry["radial_spacing_mpc"]) / float(
        np.mean(np.diff(frequencies_hz))
    )
    maximum_delays = _maximum_patch_delays(
        kperp_edges=kperp_edges,
        transverse_distance_mpc=float(resolved.geometry["transverse_distance_mpc"]),
        reference_frequency_hz=float(resolved.geometry["reference_frequency_mhz"])
        * 1e6,
        source_corner_angle_deg=float(resolved.geometry["source_corner_angle_deg"]),
        wedge_buffer_mpc_inv=float(resolved.geometry["wedge_buffer_mpc_inv"]),
        radial_mpc_per_hz=radial_mpc_per_hz,
    )
    settings = metadata["qbeta"]
    input_frequency_weights = np.asarray(
        combined.get("input_frequency_weights", np.ones(frequencies_mhz.size)),
        dtype=np.float64,
    )
    bandpower_kwargs = {
        "frequencies_hz": frequencies_hz,
        "analysis_frequency_indices": np.asarray(
            combined["analysis_frequency_indices"], dtype=np.int64
        ),
        "filter_bandwidth_scope": str(settings["filter_bandwidth_scope"]),
        "row_kperp": row_kperp,
        "kperp_edges": kperp_edges,
        "maximum_delays_s": maximum_delays,
        "dpss_eigenvalue_threshold": float(settings["dpss_eigenvalue_threshold"]),
        "foreground_filter": str(settings["foreground_filter"]),
        "suppression_strength": float(settings["suppression_strength"]),
        "polynomial_degree": int(settings["polynomial_degree"]),
        "spectral_taper": str(settings["spectral_taper"]),
        "input_frequency_weights": input_frequency_weights,
    }
    output_band_ids = np.asarray(combined["output_band_ids"], dtype=np.int64)

    auto_total, _, _, _, _ = _visibility_bandpowers(
        visibilities=foreground_vis + eor_vis, **bandpower_kwargs
    )
    cross_total = _cross_q(
        (foreground_vis + eor_vis)[None, ...],
        (foreground_vis + eor_vis)[None, ...],
        output_band_ids=output_band_ids,
        bandpower_kwargs=bandpower_kwargs,
    )[0]
    auto_total = np.asarray(auto_total).reshape(-1)[output_band_ids]
    cross_auto_relative_l2 = float(
        np.linalg.norm(cross_total - auto_total)
        / max(float(np.linalg.norm(auto_total)), 1e-300)
    )
    foreground_q = np.asarray(combined["bank_foreground_q"], dtype=np.float64)
    eor_q = np.asarray(combined["bank_eor_q"], dtype=np.float64)
    total_q = np.asarray(combined["bank_total_q"], dtype=np.float64)
    foreground_eor_cross_q = total_q - foreground_q - eor_q

    prefix = f"{args.profile}_"
    selected = np.asarray(coarse[prefix + "selected"], dtype=bool)
    target_all = np.asarray(coarse[prefix + "target"], dtype=np.float64)
    metric_weights_all = np.asarray(
        coarse[prefix + "group_metric_weights"], dtype=np.float64
    )
    target = target_all[selected]
    metric_weights = metric_weights_all[selected]
    foreground_coarse = _coarse_estimate(foreground_q, coarse, args.profile)
    eor_coarse = _coarse_estimate(eor_q, coarse, args.profile)
    total_coarse = _coarse_estimate(total_q, coarse, args.profile)
    group_kperp, group_kpar, group_k = _group_coordinates(
        combined=combined,
        coarse=coarse,
        profile=args.profile,
        kperp_edges=kperp_edges,
        kpar_values=kpar_values,
    )
    h = float(config["cosmology"]["H0_km_s_mpc"]) / 100.0
    reference_k_mpc = float(args.reference_k_h_mpc) * h
    selected_positions = np.flatnonzero(selected)
    reference_selected_position = int(
        np.argmin(np.abs(group_k[selected] - reference_k_mpc))
    )
    reference_group = int(selected_positions[reference_selected_position])

    sefd_jy, sefd_lookup = lookup_ska_low_stokes_i_sefd(
        args.sefd_h5,
        frequencies_mhz=frequencies_mhz,
        az_deg=float(args.az_deg),
        el_deg=float(args.el_deg),
        start_lst_hour=float(args.start_lst_hour),
        end_lst_hour=float(args.end_lst_hour),
    )
    time_steps = int(bank_manifest["instrument"]["time_steps"])
    thermal_results: dict[str, Any] = {}
    product_arrays: dict[str, np.ndarray] = {
        "selected": selected.astype(np.int8),
        "target": target_all,
        "metric_weights": metric_weights_all,
        "group_kperp_mpc_inv": group_kperp,
        "group_kpar_mpc_inv": group_kpar,
        "group_k_mpc_inv": group_k,
        "sefd_jy": sefd_jy,
        "input_frequency_weights": input_frequency_weights,
    }
    sigma_cache: dict[float, np.ndarray] = {}
    for integration_index, hours in enumerate(integration_hours):
        sigma, seconds_per_row_per_split = thermal_noise_sigma_per_real_component(
            sefd_jy,
            channel_bandwidth_hz=float(args.channel_bandwidth_hz),
            total_integration_hours=float(hours),
            time_step_count=time_steps,
        )
        thermal = _thermal_samples(
            foreground_vis=foreground_vis,
            eor_vis=eor_vis,
            eor_q=eor_q,
            foreground_eor_cross_q=foreground_eor_cross_q,
            sigma_per_frequency=sigma,
            realization_count=int(args.realizations),
            chunk_size=int(args.chunk_size),
            seed=int(args.thermal_seed) + 10007 * integration_index,
            output_band_ids=output_band_ids,
            bandpower_kwargs=bandpower_kwargs,
        )
        sigma_cache[float(hours)] = sigma
        null_selected = _coarse_estimate(
            thermal["null"], coarse, args.profile
        )[:, selected]
        noise_selected = _coarse_estimate(
            thermal["noise"], coarse, args.profile
        )[:, selected]
        eor_only_selected = _coarse_estimate(
            thermal["eor_only"], coarse, args.profile
        )[:, selected]
        total_selected = _coarse_estimate(
            _factor_samples(thermal, eor_q, 1.0), coarse, args.profile
        )[:, selected]
        coarse_thermal, coarse_eor_selected = _coarse_significance_components(
            thermal=thermal,
            eor_q=eor_q,
            coarse=coarse,
            profile=args.profile,
            selected=selected,
        )
        null_std = np.std(null_selected, axis=0, ddof=1)
        null_z = np.mean(null_selected, axis=0) / np.maximum(null_std, 1e-300)
        boost_results: dict[str, Any] = {}
        for requested in target_significances:
            factor, achieved = _solve_significance_factor(
                thermal=coarse_thermal,
                eor_q=coarse_eor_selected,
                target=target,
                reference_group=reference_selected_position,
                requested_significance=float(requested),
            )
            boosted = _factor_samples(
                coarse_thermal, coarse_eor_selected, factor
            )
            boosted_target = factor * factor * target
            boost_results[f"{float(requested):g}sigma"] = {
                "amplitude_factor": factor,
                "power_factor": factor * factor,
                "achieved_reference_group_significance": achieved,
                "metrics": _mean_metrics(
                    boosted, boosted_target, metric_weights
                ),
                "noise_coverage_about_noiseless_expectation": _coverage(
                    boosted,
                    (
                        foreground_coarse
                        + factor
                        * (total_coarse - foreground_coarse - eor_coarse)
                        + factor * factor * eor_coarse
                    )[selected],
                ),
                "scientific_target_coverage": _coverage(
                    boosted, boosted_target
                ),
            }
        label = f"{float(hours):g}h"
        thermal_results[label] = {
            "total_integration_hours": float(hours),
            "seconds_per_row_per_split": seconds_per_row_per_split,
            "per_real_component_sigma_jy": {
                "minimum": float(np.min(sigma)),
                "median": float(np.median(sigma)),
                "maximum": float(np.max(sigma)),
            },
            "noise_only": {
                "maximum_absolute_group_mean_over_sigma": float(
                    np.max(
                        np.abs(np.mean(noise_selected, axis=0))
                        / np.maximum(np.std(noise_selected, axis=0, ddof=1), 1e-300)
                    )
                ),
                "coverage_about_zero": _coverage(
                    noise_selected, np.zeros(target.size)
                ),
            },
            "foreground_null": {
                "maximum_absolute_single_observation_significance": float(
                    np.max(np.abs(null_z))
                ),
                "groups_above_3sigma": int(np.count_nonzero(np.abs(null_z) > 3.0)),
                "groups_above_5sigma": int(np.count_nonzero(np.abs(null_z) > 5.0)),
                "mean_group_estimate": np.mean(null_selected, axis=0),
                "group_standard_deviation": null_std,
            },
            "eor_only_unboosted": {
                "reference_group_significance": float(
                    abs(target[reference_selected_position])
                    / max(
                        float(
                            np.std(
                                eor_only_selected[:, reference_selected_position],
                                ddof=1,
                            )
                        ),
                        1e-300,
                    )
                ),
                "metrics": _mean_metrics(
                    eor_only_selected, target, metric_weights
                ),
                "noise_coverage_about_noiseless_expectation": _coverage(
                    eor_only_selected, eor_coarse[selected]
                ),
            },
            "foreground_plus_eor_unboosted": {
                "reference_group_significance": float(
                    abs(target[reference_selected_position])
                    / max(
                        float(
                            np.std(
                                total_selected[:, reference_selected_position], ddof=1
                            )
                        ),
                        1e-300,
                    )
                ),
                "metrics": _mean_metrics(total_selected, target, metric_weights),
                "noise_coverage_about_noiseless_expectation": _coverage(
                    total_selected, total_coarse[selected]
                ),
                "scientific_target_coverage": _coverage(total_selected, target),
            },
            "boosted_benchmarks": boost_results,
        }
        for name, values in (
            ("noise", noise_selected),
            ("null", null_selected),
            ("eor_only", eor_only_selected),
            ("total", total_selected),
        ):
            product_arrays[f"thermal_{label}_{name}_selected"] = values

    longest_hours = float(np.max(integration_hours))
    gain_results: dict[str, Any] = {}
    for profile_index, profile in enumerate(gain_profiles):
        profile_results: dict[str, Any] = {}
        for rms_index, rms in enumerate(gain_rms_values):
            gain_foreground_q, gain_only_q, gain_thermal_q = _systematic_samples(
                foreground_vis=foreground_vis,
                eor_vis=eor_vis,
                antenna1=np.asarray(bank["sample_antenna1"])[selected_rows],
                antenna2=np.asarray(bank["sample_antenna2"])[selected_rows],
                frequencies_mhz=frequencies_mhz,
                sigma_per_frequency=sigma_cache[longest_hours],
                realization_count=int(args.gain_realizations),
                chunk_size=int(args.chunk_size),
                rms=float(rms),
                profile=str(profile),
                ripple_cycles=float(args.gain_ripple_cycles),
                seed=(
                    int(args.gain_seed)
                    + 1000003 * profile_index
                    + 10007 * rms_index
                ),
                output_band_ids=output_band_ids,
                bandpower_kwargs=bandpower_kwargs,
            )
            gain_foreground = _coarse_estimate(
                gain_foreground_q, coarse, args.profile
            )[:, selected]
            gain_only = _coarse_estimate(
                gain_only_q, coarse, args.profile
            )[:, selected]
            gain_thermal = _coarse_estimate(
                gain_thermal_q, coarse, args.profile
            )[:, selected]
            key = f"{float(rms):.0e}"
            profile_results[key] = {
                "station_log_amplitude_rms": float(rms),
                "station_phase_rms_radian": float(rms),
                "foreground_contamination": (
                    _foreground_contamination_metrics(
                        gain_foreground, target, metric_weights
                    )
                ),
                "gain_only_metrics": _mean_metrics(
                    gain_only, target, metric_weights
                ),
                f"gain_plus_thermal_{longest_hours:g}h_metrics": _mean_metrics(
                    gain_thermal, target, metric_weights
                ),
                "gain_only_integrated_ratio_distribution": _distribution(
                    _weighted_metrics_batch(gain_only, target, metric_weights)[0]
                ),
            }
            product_arrays[
                f"gain_{profile}_{key}_foreground_selected"
            ] = gain_foreground
            product_arrays[
                f"gain_{profile}_{key}_only_selected"
            ] = gain_only
            product_arrays[
                f"gain_{profile}_{key}_{longest_hours:g}h_selected"
            ] = gain_thermal
        gain_results[str(profile)] = profile_results

    result = {
        "schema": "visibility_qbeta_noise_systematics",
        "schema_version": 1,
        "selection_contract": (
            "Coarse profile and response-only selected groups are frozen before "
            "thermal-noise, EoR-amplitude, or gain-residual realizations."
        ),
        "inputs": {
            "combined_npz": str(args.combined_npz),
            "coarse_npz": str(args.coarse_npz),
            "config": str(args.config),
            "visibility_bank_sha256": bank_manifest["bank_sha256"],
            "selected_visibility_row_count": int(selected_rows.size),
            "profile": str(args.profile),
            "selected_group_count": int(np.count_nonzero(selected)),
        },
        "cross_power_closure": {
            "cross_equal_inputs_vs_auto_relative_l2": cross_auto_relative_l2,
        },
        "thermal_noise_contract": {
            "estimator": (
                "Re[(L V_A) conj(L V_B)] with independent Gaussian noise splits"
            ),
            "sefd_definition": "SKA-Low station Stokes-I SEFD",
            "sefd_h5_sha256": _sha256(args.sefd_h5),
            "sefd_source_url": SKAO_SEFD_SOURCE_URL,
            "sefd_source_revision": SKAO_SEFD_SOURCE_REVISION,
            "sefd_lookup": sefd_lookup,
            "sefd_jy": {
                "minimum": float(np.min(sefd_jy)),
                "median": float(np.median(sefd_jy)),
                "maximum": float(np.max(sefd_jy)),
            },
            "channel_bandwidth_hz": float(args.channel_bandwidth_hz),
            "time_step_count": time_steps,
            "split_count": 2,
            "integration_interpretation": (
                "Equivalent repeats of the frozen 320-s LST track; total time "
                "is distributed uniformly over its time bins and two splits."
            ),
        },
        "reference_group": {
            "requested_k_h_mpc": float(args.reference_k_h_mpc),
            "requested_k_mpc_inv": reference_k_mpc,
            "coarse_group_index": reference_group,
            "selected_group_position": reference_selected_position,
            "kperp_mpc_inv": float(group_kperp[reference_group]),
            "kpar_mpc_inv": float(group_kpar[reference_group]),
            "k_mpc_inv": float(group_k[reference_group]),
            "k_h_mpc": float(group_k[reference_group] / h),
            "target_power": float(target_all[reference_group]),
        },
        "thermal_results": thermal_results,
        "gain_residual_contract": {
            "profiles": gain_profiles,
            "profile_definitions": {
                "smooth": "Independent station Legendre degrees 0--2",
                "ripple": (
                    "Independent station sinusoids with "
                    f"{float(args.gain_ripple_cycles):g} "
                    "cycles across the input band"
                ),
            },
            "rms_values": gain_rms_values,
            "ripple_cycles": float(args.gain_ripple_cycles),
            "rms_normalization": (
                "Per-realization RMS over the station-frequency grid after "
                "removing its realization-wide mean"
            ),
            "station_time_dependence": "constant over the frozen 320-s track",
            "application": (
                "g_p g_q* multiplies signal and thermal noise; nominal response "
                "remains fixed"
            ),
            "thermal_integration_hours": longest_hours,
        },
        "gain_results": gain_results,
        "limitations": [
            "Thermal samples repeat one frozen LST track rather than adding new "
            "uv tracks.",
            "Gain residuals are direction-independent phenomenological stress models.",
            "RFI, ionosphere, and correlated receiver noise are not injected.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "summary.json", result)
    _atomic_npz(args.out_dir / "products.npz", product_arrays)
    print(json.dumps(_json_safe(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
