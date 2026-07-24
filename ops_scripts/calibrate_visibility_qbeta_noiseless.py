#!/usr/bin/env python3
"""Calibrate sky-band Q_beta responses through an exact visibility operator.

The pilot uses fixed baseline-time rows so that chromatic baseline migration is
preserved without introducing gridding ambiguity.  Independent Gaussian
unit-band skies calibrate and validate the quadratic response.  Injected EoR is
used only for direct-DFT closure and a held-out target-subspace test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chips_visibility import (  # noqa: E402
    build_chebyshev_quadratic_response,
    build_quadratic_response,
    dpss_foreground_basis,
    fold_absolute_delay,
    fold_window_absolute_delay,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_qbeta import (  # noqa: E402
    C_M_S,
    OMEGA_EARTH_RAD_S,
    build_sky_band_layout,
    direction_cosines,
    reporting_band_ids,
    source_bandpowers,
    stratified_row_indices,
    weighted_response_pseudoinverse,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--osm-pattern", required=True)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-kperp-bin", type=int, default=8)
    parser.add_argument(
        "--row-scope",
        choices=("all", "reporting_kperp"),
        default="all",
    )
    parser.add_argument("--row-seed", type=int, default=20260724)
    parser.add_argument("--row-partition-index", type=int, default=0)
    parser.add_argument("--row-partition-count", type=int, default=1)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--mixture-repeats", type=int, default=4)
    parser.add_argument(
        "--source-scope",
        choices=(
            "reporting",
            "all_in_range",
            "all_in_range_with_nyquist",
        ),
        default="reporting",
    )
    parser.add_argument("--probe-batch-size", type=int, default=20)
    parser.add_argument("--probe-seed", type=int, default=51021)
    parser.add_argument("--source-chunk", type=int, default=4096)
    parser.add_argument("--row-chunk", type=int, default=32)
    parser.add_argument(
        "--operator-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--phase-ra-deg", type=float, default=0.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument(
        "--foreground-filter",
        choices=(
            "none",
            "dpss_hard",
            "dpss_soft",
            "chebyshev",
            "chebyshev_rank_matched",
        ),
        default="dpss_hard",
    )
    parser.add_argument("--suppression-strength", type=float, default=1e4)
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--dpss-eigenvalue-threshold", type=float, default=1e-12)
    parser.add_argument(
        "--spectral-taper",
        choices=("none", "hann", "blackman_harris"),
        default="hann",
    )
    parser.add_argument("--minimum-window-self-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-relative-sensitivity", type=float, default=1e-4)
    parser.add_argument("--response-rcond", type=float, default=1e-4)
    return parser.parse_args(argv)


def _format_pattern(pattern: str, frequency_mhz: float) -> Path:
    return Path(str(pattern).format(freq=float(frequency_mhz)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_k2jy(path: Path) -> float:
    pattern = re.compile(r"K2JyPixel\s*=\s*([0-9.eE+-]+)")
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(16):
            line = handle.readline()
            if not line:
                break
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"K2JyPixel header not found in {path}")


def _build_sky_cache(
    *,
    path: Path,
    osm_pattern: str,
    frequencies_mhz: np.ndarray,
    expected_source_count: int,
    phase_ra_deg: float,
    phase_dec_deg: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_path = _format_pattern(osm_pattern, float(frequencies_mhz[0]))
    first = np.loadtxt(
        first_path,
        delimiter=",",
        comments="#",
        usecols=(0, 1, 2),
        dtype=np.float64,
    )
    if first.shape != (expected_source_count, 3):
        raise ValueError(
            f"Unexpected first OSM shape {first.shape}; "
            f"expected {(expected_source_count, 3)}"
        )
    l_cosine, m_cosine, n_cosine = direction_cosines(
        first[:, 0],
        first[:, 1],
        phase_ra_deg=float(phase_ra_deg),
        phase_dec_deg=float(phase_dec_deg),
    )
    eor_jy = np.empty(
        (frequencies_mhz.size, expected_source_count), dtype=np.float64
    )
    k2jy = np.empty(frequencies_mhz.size, dtype=np.float64)
    for index, frequency in enumerate(frequencies_mhz):
        osm_path = _format_pattern(osm_pattern, float(frequency))
        if index == 0:
            eor_jy[index] = first[:, 2]
        else:
            values = np.loadtxt(
                osm_path,
                delimiter=",",
                comments="#",
                usecols=(2,),
                dtype=np.float64,
            )
            if values.shape != (expected_source_count,):
                raise ValueError(f"Unexpected OSM flux shape in {osm_path}")
            eor_jy[index] = values
        k2jy[index] = _read_k2jy(osm_path)
        print(
            json.dumps(
                {
                    "event": "sky_cache_frequency",
                    "frequency_mhz": float(frequency),
                    "index": int(index),
                }
            ),
            flush=True,
        )
    _atomic_npz(
        path,
        {
            "frequencies_mhz": frequencies_mhz.astype(np.float64),
            "l_cosine": l_cosine.astype(np.float64),
            "m_cosine": m_cosine.astype(np.float64),
            "n_minus_one": (n_cosine - 1.0).astype(np.float64),
            "eor_jy": eor_jy,
            "k2jy_per_pixel": k2jy,
        },
    )


def _load_or_build_sky_cache(
    args: argparse.Namespace,
    frequencies_mhz: np.ndarray,
    expected_source_count: int,
) -> dict[str, np.ndarray]:
    if not args.sky_cache.is_file():
        _build_sky_cache(
            path=args.sky_cache,
            osm_pattern=str(args.osm_pattern),
            frequencies_mhz=frequencies_mhz,
            expected_source_count=int(expected_source_count),
            phase_ra_deg=float(args.phase_ra_deg),
            phase_dec_deg=float(args.phase_dec_deg),
        )
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    if not np.allclose(
        cache["frequencies_mhz"], frequencies_mhz, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Sky-cache frequencies differ from the analysis")
    if cache["eor_jy"].shape != (frequencies_mhz.size, expected_source_count):
        raise ValueError("Sky-cache flux cube has the wrong shape")
    return cache


def _load_bank(
    bank_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    bank_path = bank_dir / "visibility_bank.npz"
    if _sha256(bank_path) != str(manifest["bank_sha256"]):
        raise ValueError("Visibility-bank SHA256 differs from its manifest")
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    return bank, manifest


def _row_kperp(
    uvw_m: np.ndarray,
    *,
    reference_frequency_hz: float,
    transverse_distance_mpc: float,
) -> np.ndarray:
    uv_radius_lambda = (
        np.hypot(uvw_m[:, 0], uvw_m[:, 1])
        * float(reference_frequency_hz)
        / C_M_S
    )
    return 2.0 * math.pi * uv_radius_lambda / float(transverse_distance_mpc)


def _build_exact_operator(
    *,
    torch: Any,
    frequencies_hz: np.ndarray,
    uvw_m: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    n_minus_one: np.ndarray,
    channel_bandwidth_hz: float,
    integration_time_s: float,
    phase_dec_deg: float,
    device: Any,
    operator_dtype: str,
    row_chunk: int,
    source_chunk: int,
) -> Any:
    complex_dtype = (
        torch.complex64 if str(operator_dtype) == "complex64" else torch.complex128
    )
    phase_dtype = torch.float64
    frequencies = torch.as_tensor(
        frequencies_hz, dtype=phase_dtype, device=device
    )
    uvw = torch.as_tensor(uvw_m, dtype=phase_dtype, device=device)
    l_tensor = torch.as_tensor(l_cosine, dtype=phase_dtype, device=device)
    m_tensor = torch.as_tensor(m_cosine, dtype=phase_dtype, device=device)
    n_tensor = torch.as_tensor(n_minus_one, dtype=phase_dtype, device=device)
    n_frequency = int(frequencies.numel())
    n_row = int(uvw.shape[0])
    n_source = int(l_tensor.numel())
    operator = torch.empty(
        (n_frequency, n_row, n_source),
        dtype=complex_dtype,
        device=device,
    )
    dec0 = math.radians(float(phase_dec_deg))
    started = time.monotonic()
    for frequency_index in range(n_frequency):
        frequency = frequencies[frequency_index]
        for row_first in range(0, n_row, int(row_chunk)):
            row_stop = min(n_row, row_first + int(row_chunk))
            uvw_block = uvw[row_first:row_stop]
            u = uvw_block[:, 0:1]
            v = uvw_block[:, 1:2]
            w = uvw_block[:, 2:3]
            transverse = -math.sin(dec0) * v + math.cos(dec0) * w
            for source_first in range(0, n_source, int(source_chunk)):
                source_stop = min(n_source, source_first + int(source_chunk))
                ll = l_tensor[source_first:source_stop][None, :]
                mm = m_tensor[source_first:source_stop][None, :]
                nn = n_tensor[source_first:source_stop][None, :]
                path_m = u * ll + v * mm + w * nn
                delay_s = path_m / C_M_S
                bandwidth = torch.sinc(
                    delay_s * float(channel_bandwidth_hz)
                )
                path_rate = (
                    transverse * ll
                    + u * math.sin(dec0) * mm
                    - u * math.cos(dec0) * nn
                )
                time_cycles = (
                    frequency
                    * float(integration_time_s)
                    * OMEGA_EARTH_RAD_S
                    * path_rate
                    / C_M_S
                )
                amplitude = bandwidth * torch.sinc(time_cycles)
                phase = 2.0 * math.pi * frequency * delay_s
                kernel = torch.complex(
                    amplitude * torch.cos(phase),
                    amplitude * torch.sin(phase),
                )
                operator[
                    frequency_index,
                    row_first:row_stop,
                    source_first:source_stop,
                ] = kernel.to(complex_dtype)
        print(
            json.dumps(
                {
                    "event": "operator_frequency",
                    "frequency_index": int(frequency_index),
                    "frequency_mhz": float(frequencies_hz[frequency_index] / 1e6),
                    "elapsed_seconds": float(time.monotonic() - started),
                }
            ),
            flush=True,
        )
    return operator


def _apply_operator(
    *,
    torch: Any,
    operator: Any,
    sky_jy: Any,
) -> np.ndarray:
    if sky_jy.ndim == 3:
        sky = sky_jy.unsqueeze(0)
        squeeze = True
    elif sky_jy.ndim == 4:
        sky = sky_jy
        squeeze = False
    else:
        raise ValueError("Sky tensor must have shape [freq,y,x] or [batch,freq,y,x]")
    batch, n_frequency, _, _ = sky.shape
    output = torch.empty(
        (batch, n_frequency, operator.shape[1]),
        dtype=operator.dtype,
        device=operator.device,
    )
    for frequency_index in range(n_frequency):
        flux = sky[:, frequency_index].reshape(batch, -1).T
        output[:, frequency_index] = (
            operator[frequency_index] @ flux.to(operator.dtype)
        ).T
    result = np.asarray(output.detach().cpu())
    return result[0] if squeeze else result


def _operator_closure_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=np.complex128)
    truth = np.asarray(target, dtype=np.complex128)
    residual = prediction - truth
    denominator = max(float(np.sum(np.abs(truth) ** 2)), 1e-300)
    gain_denominator = max(float(np.sum(np.abs(prediction) ** 2)), 1e-300)
    gain = np.sum(np.conjugate(prediction) * truth) / gain_denominator
    gain_residual = gain * prediction - truth
    per_frequency = []
    for index in range(truth.shape[0]):
        scale = max(float(np.sum(np.abs(truth[index]) ** 2)), 1e-300)
        per_frequency.append(
            math.sqrt(float(np.sum(np.abs(residual[index]) ** 2)) / scale)
        )
    return {
        "relative_l2": math.sqrt(
            float(np.sum(np.abs(residual) ** 2)) / denominator
        ),
        "gain_corrected_relative_l2": math.sqrt(
            float(np.sum(np.abs(gain_residual) ** 2)) / denominator
        ),
        "complex_gain_real": float(np.real(gain)),
        "complex_gain_imag": float(np.imag(gain)),
        "per_frequency_relative_l2": per_frequency,
        "maximum_per_frequency_relative_l2": float(np.max(per_frequency)),
    }


def _maximum_patch_delays(
    *,
    kperp_edges: np.ndarray,
    transverse_distance_mpc: float,
    reference_frequency_hz: float,
    source_corner_angle_deg: float,
    wedge_buffer_mpc_inv: float,
    radial_mpc_per_hz: float,
) -> np.ndarray:
    u_upper = (
        np.asarray(kperp_edges[1:], dtype=np.float64)
        * float(transverse_distance_mpc)
        / (2.0 * math.pi)
    )
    buffer_delay = (
        float(wedge_buffer_mpc_inv)
        * float(radial_mpc_per_hz)
        / (2.0 * math.pi)
    )
    return (
        u_upper
        * math.sin(math.radians(float(source_corner_angle_deg)))
        / float(reference_frequency_hz)
        + buffer_delay
    )


def _visibility_bandpowers(
    *,
    visibilities: np.ndarray,
    frequencies_hz: np.ndarray,
    row_kperp: np.ndarray,
    kperp_edges: np.ndarray,
    maximum_delays_s: np.ndarray,
    dpss_eigenvalue_threshold: float,
    foreground_filter: str,
    suppression_strength: float,
    polynomial_degree: int,
    spectral_taper: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(visibilities)
    if values.ndim == 2:
        values = values[None, ...]
        squeeze = True
    elif values.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Visibilities must have shape [freq,row] or [batch,freq,row]")
    batch, n_frequency, _ = values.shape
    n_kperp = int(kperp_edges.size - 1)
    output: np.ndarray | None = None
    window_self: np.ndarray | None = None
    relative_sensitivity: np.ndarray | None = None
    counts = np.zeros(n_kperp, dtype=np.int64)
    foreground_ranks = np.zeros(n_kperp, dtype=np.int64)
    for transverse_index in range(n_kperp):
        members = np.flatnonzero(
            (row_kperp >= kperp_edges[transverse_index])
            & (
                (row_kperp < kperp_edges[transverse_index + 1])
                | (
                    transverse_index == n_kperp - 1
                    and np.isclose(
                        row_kperp, kperp_edges[transverse_index + 1]
                    )
                )
            )
        )
        counts[transverse_index] = int(members.size)
        maximum_delay = float(maximum_delays_s[transverse_index])
        filter_name = str(foreground_filter)
        if filter_name == "none":
            response = build_quadratic_response(
                frequencies_hz,
                max_delay_s=maximum_delay,
                suppression_strength=0.0,
                dpss_eigenvalue_threshold=float(dpss_eigenvalue_threshold),
                taper=spectral_taper,
            )
        elif filter_name in {"dpss_hard", "dpss_soft"}:
            response = build_quadratic_response(
                frequencies_hz,
                max_delay_s=maximum_delay,
                suppression_strength=(
                    math.inf
                    if filter_name == "dpss_hard"
                    else float(suppression_strength)
                ),
                dpss_eigenvalue_threshold=float(dpss_eigenvalue_threshold),
                taper=spectral_taper,
            )
        elif filter_name == "chebyshev":
            response = build_chebyshev_quadratic_response(
                frequencies_hz,
                degree=int(polynomial_degree),
                suppression_strength=math.inf,
                taper=spectral_taper,
            )
        elif filter_name == "chebyshev_rank_matched":
            dpss_basis = dpss_foreground_basis(
                frequencies_hz,
                maximum_delay,
                eigenvalue_threshold=float(dpss_eigenvalue_threshold),
            )
            response = build_chebyshev_quadratic_response(
                frequencies_hz,
                degree=dpss_basis.rank - 1,
                suppression_strength=math.inf,
                taper=spectral_taper,
            )
        else:
            raise ValueError(f"Unsupported foreground filter: {filter_name}")
        foreground_ranks[transverse_index] = (
            0 if filter_name == "none" else int(response.foreground_rank)
        )
        raw = build_quadratic_response(
            frequencies_hz,
            max_delay_s=maximum_delay,
            suppression_strength=0.0,
            dpss_eigenvalue_threshold=float(dpss_eigenvalue_threshold),
            taper=spectral_taper,
        )
        selected = np.transpose(values[:, :, members], (0, 2, 1))
        transformed = selected @ response.analysis_matrix.T
        estimate = np.mean(np.square(np.abs(transformed)), axis=1)
        estimate[:, response.supported] /= response.row_normalization[
            response.supported
        ][None, :]
        folded, delays, _ = fold_absolute_delay(estimate, response.delays_s)
        folded_norm, _, _ = fold_absolute_delay(
            response.row_normalization, response.delays_s
        )
        raw_norm, _, _ = fold_absolute_delay(
            raw.row_normalization, raw.delays_s
        )
        folded_window, _ = fold_window_absolute_delay(
            response.window, response.delays_s
        )
        if output is None:
            n_delay = int(folded.shape[-1])
            output = np.empty((batch, n_kperp, n_delay), dtype=np.float64)
            window_self = np.empty((n_kperp, n_delay), dtype=np.float64)
            relative_sensitivity = np.empty(
                (n_kperp, n_delay), dtype=np.float64
            )
        output[:, transverse_index] = folded
        assert window_self is not None
        assert relative_sensitivity is not None
        window_self[transverse_index] = np.diag(folded_window)
        relative_sensitivity[transverse_index] = np.divide(
            folded_norm,
            raw_norm,
            out=np.zeros_like(folded_norm),
            where=raw_norm > 0.0,
        )
    assert output is not None
    assert window_self is not None
    assert relative_sensitivity is not None
    return (
        output[0] if squeeze else output,
        counts,
        window_self,
        relative_sensitivity,
        foreground_ranks,
    )


def _generate_probe_outputs(
    *,
    torch: Any,
    operator: Any,
    layout: Any,
    source_band_ids: np.ndarray,
    k2jy_per_pixel: np.ndarray,
    repeats: int,
    batch_size: int,
    seed: int,
    bandpower_kwargs: dict[str, Any],
) -> np.ndarray:
    device = operator.device
    real_dtype = (
        torch.float32 if operator.dtype == torch.complex64 else torch.float64
    )
    mode_bands = torch.as_tensor(
        layout.mode_bands, dtype=torch.int64, device=device
    )
    k2jy = torch.as_tensor(
        k2jy_per_pixel, dtype=real_dtype, device=device
    )
    output: np.ndarray | None = None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    started = time.monotonic()
    for repeat in range(int(repeats)):
        for first in range(0, source_band_ids.size, int(batch_size)):
            band_ids = source_band_ids[first : first + int(batch_size)]
            current = int(band_ids.size)
            white = torch.randn(
                (current, *layout.cube_shape),
                dtype=real_dtype,
                device=device,
                generator=generator,
            )
            spectrum = torch.fft.fftn(
                white, dim=(-3, -2, -1), norm="ortho"
            )
            ids = torch.as_tensor(band_ids, dtype=torch.int64, device=device)
            mask = mode_bands.unsqueeze(0) == ids[:, None, None, None]
            phase = spectrum / torch.clamp(
                torch.abs(spectrum), min=torch.finfo(real_dtype).tiny
            )
            selected = torch.where(mask, phase, torch.zeros_like(phase))
            sky_k = torch.fft.ifftn(
                selected, dim=(-3, -2, -1), norm="ortho"
            ).real
            sky_jy = sky_k * k2jy[None, :, None, None]
            vis = _apply_operator(torch=torch, operator=operator, sky_jy=sky_jy)
            bandpowers, _, _, _, _ = _visibility_bandpowers(
                visibilities=vis,
                **bandpower_kwargs,
            )
            if output is None:
                output = np.empty(
                    (
                        int(repeats),
                        source_band_ids.size,
                        *bandpowers.shape[1:],
                    ),
                    dtype=np.float64,
                )
            output[repeat, first : first + current] = bandpowers
            del white, spectrum, mask, phase, selected, sky_k, sky_jy
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "event": "probe_batch",
                        "repeat": int(repeat),
                        "first": int(first),
                        "count": int(current),
                        "elapsed_seconds": float(time.monotonic() - started),
                    }
                ),
                flush=True,
            )
    assert output is not None
    return output


def _generate_mixture_outputs(
    *,
    torch: Any,
    operator: Any,
    layout: Any,
    source_band_ids: np.ndarray,
    source_band_power: np.ndarray,
    k2jy_per_pixel: np.ndarray,
    repeats: int,
    seed: int,
    bandpower_kwargs: dict[str, Any],
    additive_visibilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    device = operator.device
    real_dtype = (
        torch.float32 if operator.dtype == torch.complex64 else torch.float64
    )
    power_by_band = np.zeros(layout.band_count, dtype=np.float64)
    power_by_band[source_band_ids] = np.asarray(
        source_band_power, dtype=np.float64
    )
    mode_bands = torch.as_tensor(
        layout.mode_bands, dtype=torch.int64, device=device
    )
    power = torch.as_tensor(power_by_band, dtype=real_dtype, device=device)
    selected = mode_bands >= 0
    mode_amplitude = torch.zeros(
        layout.cube_shape, dtype=real_dtype, device=device
    )
    mode_amplitude[selected] = torch.sqrt(power[mode_bands[selected]])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    white = torch.randn(
        (int(repeats), *layout.cube_shape),
        dtype=real_dtype,
        device=device,
        generator=generator,
    )
    spectrum = torch.fft.fftn(white, dim=(-3, -2, -1), norm="ortho")
    phase = spectrum / torch.clamp(
        torch.abs(spectrum), min=torch.finfo(real_dtype).tiny
    )
    sky_k = torch.fft.ifftn(
        phase * mode_amplitude[None, ...],
        dim=(-3, -2, -1),
        norm="ortho",
    ).real
    k2jy = torch.as_tensor(
        k2jy_per_pixel, dtype=real_dtype, device=device
    )
    sky_jy = sky_k * k2jy[None, :, None, None]
    vis = _apply_operator(torch=torch, operator=operator, sky_jy=sky_jy)
    bandpowers, _, _, _, _ = _visibility_bandpowers(
        visibilities=vis,
        **bandpower_kwargs,
    )
    if additive_visibilities is None:
        return bandpowers, None
    additive = np.asarray(additive_visibilities, dtype=np.complex128)
    if additive.shape != vis.shape[1:]:
        raise ValueError(
            "Additive mixture visibilities must have shape [frequency,row]"
        )
    total_bandpowers, _, _, _, _ = _visibility_bandpowers(
        visibilities=vis + additive[None, ...],
        **bandpower_kwargs,
    )
    return bandpowers, total_bandpowers


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    return math.sqrt(
        float(np.sum(np.square(left - right)))
        / max(float(np.sum(np.square(right))), 1e-300)
    )


def _windowed_metrics(
    *,
    response: np.ndarray,
    observed_q: np.ndarray,
    source_power: np.ndarray,
    minimum_relative_response: float,
    target_source_positions: np.ndarray,
    minimum_target_window_fraction: float,
) -> dict[str, Any]:
    matrix = np.asarray(response, dtype=np.float64)
    q_values = np.asarray(observed_q, dtype=np.float64)
    if q_values.ndim == 1:
        q_values = q_values[None, :]
    row_sum = np.sum(matrix, axis=1)
    relative_response = row_sum / max(float(np.max(row_sum)), 1e-300)
    window = np.zeros_like(matrix)
    valid = row_sum > 0.0
    window[valid] = matrix[valid] / row_sum[valid, None]
    target_positions = np.asarray(
        target_source_positions, dtype=np.int64
    ).reshape(-1)
    if (
        target_positions.size == 0
        or np.any(target_positions < 0)
        or np.any(target_positions >= matrix.shape[1])
    ):
        raise ValueError("target_source_positions must select response columns")
    target_window_fraction = np.sum(
        window[:, target_positions], axis=1
    )
    selected = (
        (relative_response >= float(minimum_relative_response))
        & (
            target_window_fraction
            >= float(minimum_target_window_fraction)
        )
    )
    target = window @ np.asarray(source_power, dtype=np.float64)
    estimate = np.full(q_values.shape, np.nan, dtype=np.float64)
    estimate[:, valid] = q_values[:, valid] / row_sum[valid][None, :]
    weights = relative_response[selected]
    target_selected = target[selected]
    rows: list[dict[str, Any]] = []
    for row in estimate[:, selected]:
        if target_selected.size == 0:
            rows.append(
                {
                    "relative_l2": math.nan,
                    "integrated_power_ratio": math.nan,
                    "maximum_relative_window_error": math.nan,
                    "passing_window_count": 0,
                    "passing_window_fraction": math.nan,
                }
            )
            continue
        relative_error = np.abs(row - target_selected) / np.maximum(
            np.abs(target_selected), 1e-300
        )
        rows.append(
            {
                "relative_l2": math.sqrt(
                    float(
                        np.sum(
                            weights * np.square(row - target_selected)
                        )
                    )
                    / max(
                        float(
                            np.sum(
                                weights * np.square(target_selected)
                            )
                        ),
                        1e-300,
                    )
                ),
                "integrated_power_ratio": float(
                    np.sum(weights * row)
                    / np.sum(weights * target_selected)
                ),
                "maximum_relative_window_error": float(
                    np.max(relative_error)
                ),
                "passing_window_count": int(
                    np.count_nonzero(relative_error < 0.2)
                ),
                "passing_window_fraction": float(
                    np.mean(relative_error < 0.2)
                ),
            }
        )
    return {
        "minimum_relative_response": float(minimum_relative_response),
        "minimum_target_window_fraction": float(
            minimum_target_window_fraction
        ),
        "selected_window_count": int(np.count_nonzero(selected)),
        "selected_output_positions": np.flatnonzero(selected),
        "relative_response": relative_response,
        "target_window_fraction": target_window_fraction,
        "window": window,
        "window_effective_width": np.divide(
            1.0,
            np.sum(np.square(window), axis=1),
            out=np.full(row_sum.shape, np.inf),
            where=np.sum(np.square(window), axis=1) > 0.0,
        ),
        "target_windowed_power": target,
        "estimated_windowed_power": estimate,
        "realizations": rows,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    frequencies_mhz = np.asarray(
        resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    frequencies_hz = frequencies_mhz * 1e6
    bank, manifest = _load_bank(args.bank_dir)
    if not np.allclose(
        bank["frequencies_hz"], frequencies_hz, rtol=0.0, atol=1e-3
    ):
        raise ValueError("Visibility-bank frequencies differ from the config")
    source_size = int(config["image_geometry"]["source_image_size"])
    source_count = source_size * source_size
    sky = _load_or_build_sky_cache(args, frequencies_mhz, source_count)

    reference_frequency_hz = (
        float(resolved.geometry["reference_frequency_mhz"]) * 1e6
    )
    all_row_kperp = _row_kperp(
        bank["sample_uvw_m"],
        reference_frequency_hz=reference_frequency_hz,
        transverse_distance_mpc=float(resolved.geometry["transverse_distance_mpc"]),
    )
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    reporting = config["reporting_masks"]
    if args.row_scope == "reporting_kperp":
        low_fraction, high_fraction = (
            float(value)
            for value in reporting["mid_kperp_fraction_range"]
        )
        transverse_bin_count = int(kperp_edges.size - 1)
        first_reporting_kperp = int(
            math.floor(low_fraction * transverse_bin_count)
        )
        stop_reporting_kperp = int(
            math.ceil(high_fraction * transverse_bin_count)
        )
        selected_row_kperp_indices = np.arange(
            first_reporting_kperp,
            stop_reporting_kperp,
            dtype=np.int64,
        )
    else:
        selected_row_kperp_indices = np.arange(
            kperp_edges.size - 1, dtype=np.int64
        )
    selected_rows = stratified_row_indices(
        all_row_kperp,
        kperp_edges,
        rows_per_bin=int(args.rows_per_kperp_bin),
        seed=int(args.row_seed),
        partition_index=int(args.row_partition_index),
        partition_count=int(args.row_partition_count),
        bin_indices=selected_row_kperp_indices,
    )
    uvw = np.asarray(bank["sample_uvw_m"][selected_rows], dtype=np.float64)
    row_kperp = all_row_kperp[selected_rows]

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This formal Q_beta pilot requires a CUDA device")
    operator = _build_exact_operator(
        torch=torch,
        frequencies_hz=frequencies_hz,
        uvw_m=uvw,
        l_cosine=sky["l_cosine"],
        m_cosine=sky["m_cosine"],
        n_minus_one=sky["n_minus_one"],
        channel_bandwidth_hz=float(args.channel_bandwidth_hz),
        integration_time_s=float(args.integration_time_s),
        phase_dec_deg=float(args.phase_dec_deg),
        device=device,
        operator_dtype=str(args.operator_dtype),
        row_chunk=int(args.row_chunk),
        source_chunk=int(args.source_chunk),
    )
    real_dtype = (
        torch.float32 if operator.dtype == torch.complex64 else torch.float64
    )
    eor_jy_tensor = torch.as_tensor(
        sky["eor_jy"].reshape(frequencies_hz.size, source_size, source_size),
        dtype=real_dtype,
        device=device,
    )
    predicted_eor_vis = _apply_operator(
        torch=torch, operator=operator, sky_jy=eor_jy_tensor
    )
    target_eor_vis = np.asarray(
        bank["sample_eor"][:, selected_rows], dtype=np.complex128
    )
    operator_closure = _operator_closure_metrics(
        predicted_eor_vis, target_eor_vis
    )
    print(
        json.dumps({"event": "operator_closure", **operator_closure}),
        flush=True,
    )

    source_layout = build_sky_band_layout(
        (frequencies_hz.size, source_size, source_size),
        dx_mpc=float(resolved.contract.full_layout.dx_mpc),
        dy_mpc=float(resolved.contract.full_layout.dy_mpc),
        dpar_mpc=float(resolved.contract.full_layout.dpar_mpc),
        kperp_edges=kperp_edges,
        exclude_radial_nyquist=(
            args.source_scope != "all_in_range_with_nyquist"
        ),
    )
    reporting_source_band_ids = reporting_band_ids(
        source_layout,
        high_kpar_fraction=float(reporting["high_kpar_fraction"]),
        mid_kperp_fraction_range=tuple(
            float(value)
            for value in reporting["mid_kperp_fraction_range"]
        ),
        radial_band_count=int(
            resolved.contract.window_layout.kpar_values.size
        ),
    )
    if args.source_scope == "reporting":
        source_band_ids = reporting_source_band_ids
    else:
        source_band_ids = np.arange(
            source_layout.band_count, dtype=np.int64
        )
    reporting_source_positions = np.flatnonzero(
        np.isin(source_band_ids, reporting_source_band_ids)
    ).astype(np.int64)
    radial_mpc_per_hz = float(resolved.geometry["radial_spacing_mpc"]) / float(
        np.mean(np.diff(frequencies_hz))
    )
    maximum_delays = _maximum_patch_delays(
        kperp_edges=kperp_edges,
        transverse_distance_mpc=float(resolved.geometry["transverse_distance_mpc"]),
        reference_frequency_hz=reference_frequency_hz,
        source_corner_angle_deg=float(resolved.geometry["source_corner_angle_deg"]),
        wedge_buffer_mpc_inv=float(resolved.geometry["wedge_buffer_mpc_inv"]),
        radial_mpc_per_hz=radial_mpc_per_hz,
    )
    bandpower_kwargs = {
        "frequencies_hz": frequencies_hz,
        "row_kperp": row_kperp,
        "kperp_edges": kperp_edges,
        "maximum_delays_s": maximum_delays,
        "dpss_eigenvalue_threshold": float(args.dpss_eigenvalue_threshold),
        "foreground_filter": str(args.foreground_filter),
        "suppression_strength": float(args.suppression_strength),
        "polynomial_degree": int(args.polynomial_degree),
        "spectral_taper": str(args.spectral_taper),
    }
    calibration_samples = _generate_probe_outputs(
        torch=torch,
        operator=operator,
        layout=source_layout,
        source_band_ids=source_band_ids,
        k2jy_per_pixel=sky["k2jy_per_pixel"],
        repeats=int(args.calibration_repeats),
        batch_size=int(args.probe_batch_size),
        seed=int(args.probe_seed),
        bandpower_kwargs=bandpower_kwargs,
    )
    validation_samples = _generate_probe_outputs(
        torch=torch,
        operator=operator,
        layout=source_layout,
        source_band_ids=source_band_ids,
        k2jy_per_pixel=sky["k2jy_per_pixel"],
        repeats=int(args.validation_repeats),
        batch_size=int(args.probe_batch_size),
        seed=int(args.probe_seed) + 1000003,
        bandpower_kwargs=bandpower_kwargs,
    )
    _, sample_counts, window_self, relative_sensitivity, foreground_ranks = (
        _visibility_bandpowers(
            visibilities=predicted_eor_vis,
            **bandpower_kwargs,
        )
    )
    kpar = np.asarray(resolved.contract.window_layout.kpar_values, dtype=np.float64)
    geometric_window = resolved.window_spec.mask(
        kperp_edges[1:, None], kpar[None, :]
    )
    support = (
        geometric_window
        & (sample_counts[:, None] >= int(args.rows_per_kperp_bin))
        & (
            window_self
            >= float(args.minimum_window_self_fraction)
        )
        & (
            relative_sensitivity
            >= float(args.minimum_relative_sensitivity)
        )
    )
    output_band_ids = np.flatnonzero(support.reshape(-1))
    calibration_response = np.mean(calibration_samples, axis=0).reshape(
        source_band_ids.size, -1
    )[:, output_band_ids].T
    validation_response = np.mean(validation_samples, axis=0).reshape(
        source_band_ids.size, -1
    )[:, output_band_ids].T
    response_pinv, response_svd = weighted_response_pseudoinverse(
        calibration_response, rcond=float(args.response_rcond)
    )
    validation_recovery = response_pinv @ validation_response
    identity = np.eye(source_band_ids.size, dtype=np.float64)

    eor_k = (
        sky["eor_jy"]
        / np.asarray(sky["k2jy_per_pixel"], dtype=np.float64)[:, None]
    ).reshape(frequencies_hz.size, source_size, source_size)
    eor_spectrum = np.fft.fftn(eor_k, norm="ortho")
    source_scope_mask = np.isin(source_layout.mode_bands, source_band_ids)
    restricted_spectrum = np.where(source_scope_mask, eor_spectrum, 0.0)
    restricted_eor_k = np.fft.ifftn(
        restricted_spectrum, norm="ortho"
    ).real
    restricted_source_power_all = source_bandpowers(
        restricted_eor_k, source_layout
    )
    restricted_source_power = restricted_source_power_all[source_band_ids]
    restricted_tensor = torch.as_tensor(
        restricted_eor_k,
        dtype=real_dtype,
        device=device,
    )
    restricted_jy = restricted_tensor * torch.as_tensor(
        sky["k2jy_per_pixel"],
        dtype=real_dtype,
        device=device,
    )[:, None, None]
    restricted_vis = _apply_operator(
        torch=torch, operator=operator, sky_jy=restricted_jy
    )
    restricted_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=restricted_vis,
        **bandpower_kwargs,
    )
    restricted_q_supported = restricted_q.reshape(-1)[output_band_ids]
    restricted_prediction = calibration_response @ restricted_source_power
    restricted_recovery = response_pinv @ restricted_q_supported
    source_weights = source_layout.counts[source_band_ids].astype(np.float64)
    integrated_truth = float(
        np.sum(source_weights * restricted_source_power)
    )
    mixture_q_all, mixture_total_q_all = _generate_mixture_outputs(
        torch=torch,
        operator=operator,
        layout=source_layout,
        source_band_ids=source_band_ids,
        source_band_power=restricted_source_power,
        k2jy_per_pixel=sky["k2jy_per_pixel"],
        repeats=int(args.mixture_repeats),
        seed=int(args.probe_seed) + 2000003,
        bandpower_kwargs=bandpower_kwargs,
        additive_visibilities=np.asarray(
            bank["sample_fg"][:, selected_rows], dtype=np.complex128
        ),
    )
    mixture_q = mixture_q_all.reshape(
        int(args.mixture_repeats), -1
    )[:, output_band_ids]
    assert mixture_total_q_all is not None
    mixture_total_q = mixture_total_q_all.reshape(
        int(args.mixture_repeats), -1
    )[:, output_band_ids]
    bank_fg_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=np.asarray(
            bank["sample_fg"][:, selected_rows], dtype=np.complex128
        ),
        **bandpower_kwargs,
    )
    bank_eor_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=target_eor_vis,
        **bandpower_kwargs,
    )
    bank_total_q, _, _, _, _ = _visibility_bandpowers(
        visibilities=(
            np.asarray(
                bank["sample_fg"][:, selected_rows], dtype=np.complex128
            )
            + target_eor_vis
        ),
        **bandpower_kwargs,
    )
    bank_fg_q = bank_fg_q.reshape(-1)[output_band_ids]
    bank_eor_q = bank_eor_q.reshape(-1)[output_band_ids]
    bank_total_q = bank_total_q.reshape(-1)[output_band_ids]
    minimum_qbeta_response = 0.1
    minimum_target_window_fraction = (
        0.0 if args.source_scope == "reporting" else 0.8
    )
    restricted_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=restricted_q_supported,
        source_power=restricted_source_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    mixture_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=mixture_q,
        source_power=restricted_source_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    mixture_total_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=mixture_total_q,
        source_power=restricted_source_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    full_eor_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=bank_eor_q,
        source_power=restricted_source_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    total_windowed = _windowed_metrics(
        response=calibration_response,
        observed_q=bank_total_q,
        source_power=restricted_source_power,
        minimum_relative_response=minimum_qbeta_response,
        target_source_positions=reporting_source_positions,
        minimum_target_window_fraction=minimum_target_window_fraction,
    )
    selected_window_positions = np.asarray(
        restricted_windowed["selected_output_positions"], dtype=np.int64
    )
    foreground_windowed_power = (
        bank_fg_q[selected_window_positions]
        / np.sum(calibration_response, axis=1)[selected_window_positions]
    )
    target_windowed_power = np.asarray(
        restricted_windowed["target_windowed_power"], dtype=np.float64
    )[selected_window_positions]
    foreground_weights = np.asarray(
        restricted_windowed["relative_response"], dtype=np.float64
    )[selected_window_positions]
    if selected_window_positions.size:
        foreground_to_target = float(
            np.sum(
                foreground_weights * np.abs(foreground_windowed_power)
            )
            / np.sum(
                foreground_weights * np.abs(target_windowed_power)
            )
        )
        median_window_effective_width = float(
            np.median(
                np.asarray(
                    restricted_windowed["window_effective_width"]
                )[selected_window_positions]
            )
        )
    else:
        foreground_to_target = math.nan
        median_window_effective_width = math.nan

    products = {
        "selected_bank_rows": selected_rows,
        "selected_row_kperp_mpc_inv": row_kperp,
        "selected_row_kperp_indices": selected_row_kperp_indices,
        "source_band_ids": source_band_ids,
        "reporting_source_positions": reporting_source_positions,
        "source_band_kperp_indices": source_layout.active_kperp_indices[
            source_band_ids
        ],
        "source_band_kpar_indices": source_layout.active_kpar_indices[
            source_band_ids
        ],
        "source_band_mode_counts": source_layout.counts[source_band_ids],
        "output_band_ids": output_band_ids,
        "support": support.astype(np.int8),
        "sample_counts": sample_counts,
        "foreground_ranks": foreground_ranks,
        "window_self": window_self,
        "relative_sensitivity": relative_sensitivity,
        "calibration_samples": calibration_samples,
        "validation_samples": validation_samples,
        "calibration_response": calibration_response,
        "validation_response": validation_response,
        "response_pseudoinverse": response_pinv,
        "validation_recovery": validation_recovery,
        "restricted_eor_source_power": restricted_source_power,
        "restricted_eor_q": restricted_q_supported,
        "restricted_eor_q_prediction": restricted_prediction,
        "restricted_eor_recovery": restricted_recovery,
        "heldout_mixture_q": mixture_q,
        "heldout_total_mixture_q": mixture_total_q,
        "bank_foreground_q": bank_fg_q,
        "bank_eor_q": bank_eor_q,
        "bank_total_q": bank_total_q,
        "qbeta_window": restricted_windowed["window"],
        "qbeta_relative_response": restricted_windowed["relative_response"],
        "qbeta_target_window_fraction": restricted_windowed[
            "target_window_fraction"
        ],
        "qbeta_selected_window_positions": selected_window_positions,
        "restricted_eor_windowed_power": restricted_windowed[
            "estimated_windowed_power"
        ],
        "heldout_mixture_windowed_power": mixture_windowed[
            "estimated_windowed_power"
        ],
        "heldout_total_mixture_windowed_power": mixture_total_windowed[
            "estimated_windowed_power"
        ],
        "full_eor_windowed_power": full_eor_windowed[
            "estimated_windowed_power"
        ],
        "total_windowed_power": total_windowed["estimated_windowed_power"],
        "predicted_eor_vis": predicted_eor_vis,
        "target_eor_vis": target_eor_vis,
    }
    result = {
        "schema": "visibility_qbeta_noiseless_calibration",
        "schema_version": 1,
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "visibility_bank_sha256": manifest["bank_sha256"],
        "sky_cache_sha256": _sha256(args.sky_cache),
        "settings": {
            "rows_per_kperp_bin": int(args.rows_per_kperp_bin),
            "row_scope": str(args.row_scope),
            "selected_row_count": int(selected_rows.size),
            "row_seed": int(args.row_seed),
            "row_partition_index": int(args.row_partition_index),
            "row_partition_count": int(args.row_partition_count),
            "calibration_repeats": int(args.calibration_repeats),
            "validation_repeats": int(args.validation_repeats),
            "mixture_repeats": int(args.mixture_repeats),
            "source_scope": str(args.source_scope),
            "probe_batch_size": int(args.probe_batch_size),
            "probe_seed": int(args.probe_seed),
            "operator_dtype": str(args.operator_dtype),
            "channel_bandwidth_hz": float(args.channel_bandwidth_hz),
            "integration_time_s": float(args.integration_time_s),
            "phase_ra_deg": float(args.phase_ra_deg),
            "phase_dec_deg": float(args.phase_dec_deg),
            "foreground_filter": str(args.foreground_filter),
            "suppression_strength": float(args.suppression_strength),
            "effective_suppression": (
                "none"
                if args.foreground_filter == "none"
                else (
                    f"finite:{float(args.suppression_strength):.12g}"
                    if args.foreground_filter == "dpss_soft"
                    else "hard"
                )
            ),
            "polynomial_degree": int(args.polynomial_degree),
            "dpss_eigenvalue_threshold": float(
                args.dpss_eigenvalue_threshold
            ),
            "spectral_taper": str(args.spectral_taper),
            "response_rcond": float(args.response_rcond),
        },
        "operator_closure": operator_closure,
        "qbeta": {
            "source_scope": str(args.source_scope),
            "source_band_count": int(source_band_ids.size),
            "supported_output_band_count": int(output_band_ids.size),
            "calibration_validation_response_relative_l2": _relative_l2(
                validation_response, calibration_response
            ),
            "validation_projected_response_relative_l2": _relative_l2(
                calibration_response
                @ response_pinv
                @ validation_response,
                validation_response,
            ),
            "validation_identity_relative_l2": _relative_l2(
                validation_recovery, identity
            ),
            "validation_diagonal_median": float(
                np.median(np.diag(validation_recovery))
            ),
            "validation_offdiagonal_absolute_l1_per_column": float(
                (
                    np.sum(np.abs(validation_recovery - identity))
                    / source_band_ids.size
                )
            ),
            "response_svd": response_svd,
        },
        "restricted_eor_closure": {
            "forward_q_relative_l2": _relative_l2(
                restricted_q_supported, restricted_prediction
            ),
            "recovered_power_relative_l2": _relative_l2(
                restricted_recovery, restricted_source_power
            ),
            "integrated_power_ratio": (
                float(
                    np.sum(source_weights * restricted_recovery)
                    / integrated_truth
                )
                if integrated_truth > 0.0
                else math.nan
            ),
            "maximum_relative_band_error": float(
                np.max(
                    np.abs(
                        restricted_recovery - restricted_source_power
                    )
                    / np.maximum(np.abs(restricted_source_power), 1e-300)
                )
            ),
        },
        "windowed_candidate": {
            "selection_status": (
                "response-only threshold frozen after the initial row pilot; "
                "heldout mixture realizations are independent of calibration"
            ),
            "minimum_relative_qbeta_response": minimum_qbeta_response,
            "minimum_target_window_fraction": (
                minimum_target_window_fraction
            ),
            "selected_window_count": int(
                restricted_windowed["selected_window_count"]
            ),
            "median_window_effective_width": (
                median_window_effective_width
            ),
            "restricted_eor": restricted_windowed["realizations"][0],
            "heldout_eor_like_mixtures": mixture_windowed["realizations"],
            "heldout_fg_plus_eor_like_mixtures": mixture_total_windowed[
                "realizations"
            ],
            "full_eor_including_context": full_eor_windowed["realizations"][0],
            "foreground_to_target_integrated_absolute_ratio": (
                foreground_to_target
            ),
            "total_fg_plus_eor": total_windowed["realizations"][0],
        },
        "limitations": [
            "no thermal noise",
            "fixed baseline-time row pilot; no uv gridding",
            (
                "rows cover every configured kperp bin"
                if args.row_scope == "all"
                else "rows are concentrated in the predeclared reporting kperp range"
            ),
            (
                f"{int(args.rows_per_kperp_bin)} sampled rows per kperp bin "
                "in this run"
            ),
            (
                "Q_beta source scope is the predeclared 40-band reporting region"
                if args.source_scope == "reporting"
                else (
                    "Q_beta source scope includes all 512 in-range sky bands"
                    if args.source_scope == "all_in_range"
                    else "Q_beta source scope includes all 544 in-range sky bands including radial Nyquist"
                )
            ),
            (
                "foreground and out-of-region EoR nuisance bands are not yet included"
                if args.source_scope == "reporting"
                else (
                    "radial-Nyquist and out-of-kperp-range sky modes remain outside the response"
                    if args.source_scope == "all_in_range"
                    else "out-of-kperp-range sky modes remain outside the response"
                )
            ),
            (
                "one injected physical EoR realization is used for noiseless closure; "
                "random-phase mixtures are validation diagnostics only"
            ),
        ],
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _atomic_npz(args.out_dir / "result.npz", products)
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
