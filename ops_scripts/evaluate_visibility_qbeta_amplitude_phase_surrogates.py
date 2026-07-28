#!/usr/bin/env python3
"""Propagate actual-amplitude, randomized-phase EoR surrogates through Q_beta."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibrate_visibility_qbeta_noiseless import (  # noqa: E402
    _analysis_frequency_indices,
    _format_pattern,
    _load_bank,
    _maximum_patch_delays,
    _operator_closure_metrics,
    _row_kperp,
    _visibility_bandpowers,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_matrix_free import (  # noqa: E402
    apply_exact_visibility_operator_matrix_free,
)
from visibility_primary_beam import (  # noqa: E402
    open_indexed_frequency_row_direction_kernel_multiplier,
)
from visibility_qbeta import build_sky_band_layout  # noqa: E402


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frequency-config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument(
        "--base-result",
        action="append",
        required=True,
        help="Label=partition evaluator directory; repeat for filter variants.",
    )
    parser.add_argument(
        "--aperture-row-beam-cache-pattern", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surrogate-repeats", type=int, default=16)
    parser.add_argument("--surrogate-seed", type=int, default=20260728)
    parser.add_argument(
        "--localized-block-count",
        type=int,
        action="append",
        help=(
            "Also generate equal-bandpower random-phase skies in this many "
            "contiguous frequency blocks; repeat for multiple resolutions."
        ),
    )
    parser.add_argument("--localized-repeats", type=int, default=8)
    parser.add_argument("--localized-seed", type=int, default=20260729)
    parser.add_argument(
        "--spectral-coherence-repeats",
        type=int,
        default=0,
        help=(
            "Randomize only 2D spatial phases while retaining each spatial "
            "mode's complete cross-frequency complex vector."
        ),
    )
    parser.add_argument("--spectral-coherence-seed", type=int, default=20260730)
    parser.add_argument(
        "--sky-batch-size",
        type=int,
        default=0,
        help="Zero propagates the full sky batch in one matrix-free pass.",
    )
    parser.add_argument("--channel-bandwidth-hz", type=float, default=100000.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    parser.add_argument("--row-chunk", type=int, default=32)
    parser.add_argument("--source-chunk", type=int, default=8192)
    parser.add_argument(
        "--operator-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--maximum-operator-closure", type=float, default=1e-4)
    parser.add_argument(
        "--maximum-restricted-q-closure", type=float, default=1e-4
    )
    return parser.parse_args(argv)


def _parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = str(value).partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--base-result must use Label=/path syntax")
        if label in output:
            raise ValueError(f"Duplicate base-result label: {label}")
        output[label] = Path(raw_path)
    return output


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


def _relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_array = np.asarray(estimate)
    truth_array = np.asarray(truth)
    denominator = float(np.linalg.norm(truth_array))
    if denominator <= 0.0:
        return math.nan
    return float(np.linalg.norm(estimate_array - truth_array) / denominator)


def _propagate_in_batches(
    *,
    torch: Any,
    skies: Any,
    batch_size: int,
    operator_kwargs: dict[str, Any],
    started: float,
) -> np.ndarray:
    count = int(skies.shape[0])
    current_batch = count if int(batch_size) <= 0 else int(batch_size)
    outputs: list[np.ndarray] = []
    for first in range(0, count, current_batch):
        stop = min(count, first + current_batch)

        def report_frequency(
            frequency_index: int, frequency_hz: float
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "matrix_free_operator_frequency",
                        "sky_first": int(first),
                        "sky_stop": int(stop),
                        "frequency_index": int(frequency_index),
                        "frequency_mhz": float(frequency_hz / 1e6),
                        "elapsed_seconds": float(
                            time.monotonic() - started
                        ),
                    }
                ),
                flush=True,
            )

        outputs.append(
            apply_exact_visibility_operator_matrix_free(
                torch=torch,
                sky_jy=skies[first:stop],
                progress_callback=report_frequency,
                **operator_kwargs,
            )
        )
    return np.concatenate(outputs, axis=0)


def _localized_bandpower_surrogates(
    *,
    torch: Any,
    eor_k: Any,
    k2jy: Any,
    kperp_edges: np.ndarray,
    dx_mpc: float,
    dy_mpc: float,
    dpar_mpc: float,
    block_count: int,
    repeats: int,
    seed: int,
    real_dtype: Any,
) -> Any:
    frequency_count, size_y, size_x = (
        int(value) for value in eor_k.shape
    )
    if (
        int(block_count) < 1
        or frequency_count % int(block_count) != 0
        or int(repeats) < 1
    ):
        raise ValueError(
            "Localized blocks must divide the frequency axis and repeats "
            "must be positive"
        )
    block_size = frequency_count // int(block_count)
    output = torch.zeros(
        (int(repeats), frequency_count, size_y, size_x),
        dtype=real_dtype,
        device=eor_k.device,
    )
    generator = torch.Generator(device=eor_k.device)
    generator.manual_seed(int(seed))
    for block_index in range(int(block_count)):
        first = block_index * block_size
        stop = first + block_size
        layout = build_sky_band_layout(
            (block_size, size_y, size_x),
            dx_mpc=float(dx_mpc),
            dy_mpc=float(dy_mpc),
            dpar_mpc=float(dpar_mpc),
            kperp_edges=kperp_edges,
            exclude_radial_nyquist=False,
        )
        mode_bands = torch.as_tensor(
            layout.mode_bands,
            dtype=torch.int64,
            device=eor_k.device,
        )
        selected = mode_bands >= 0
        actual_spectrum = torch.fft.fftn(
            eor_k[first:stop], dim=(-3, -2, -1), norm="ortho"
        )
        flat_ids = mode_bands[selected]
        power_sums = torch.zeros(
            layout.band_count, dtype=real_dtype, device=eor_k.device
        )
        power_sums.scatter_add_(
            0,
            flat_ids,
            torch.square(torch.abs(actual_spectrum[selected])),
        )
        band_power = power_sums / torch.as_tensor(
            layout.counts, dtype=real_dtype, device=eor_k.device
        )
        mode_amplitude = torch.zeros(
            layout.cube_shape, dtype=real_dtype, device=eor_k.device
        )
        mode_amplitude[selected] = torch.sqrt(band_power[flat_ids])
        random_spectrum = torch.fft.fftn(
            torch.randn(
                (int(repeats), *layout.cube_shape),
                dtype=real_dtype,
                device=eor_k.device,
                generator=generator,
            ),
            dim=(-3, -2, -1),
            norm="ortho",
        )
        random_spectrum /= torch.clamp(
            torch.abs(random_spectrum), min=torch.finfo(real_dtype).tiny
        )
        random_spectrum *= mode_amplitude[None, ...]
        output[:, first:stop] = torch.fft.ifftn(
            random_spectrum, dim=(-3, -2, -1), norm="ortho"
        ).real
        del actual_spectrum, power_sums, band_power
        del mode_amplitude, random_spectrum
    return output * k2jy[None, ...]


def _spectrally_coherent_spatial_phase_surrogates(
    *,
    torch: Any,
    restricted_k: Any,
    k2jy: Any,
    repeats: int,
    seed: int,
    real_dtype: Any,
) -> Any:
    if int(repeats) < 1:
        raise ValueError("Spectral-coherence repeats must be positive")
    size_y, size_x = (int(value) for value in restricted_k.shape[-2:])
    spatial_spectrum = torch.fft.fftn(
        restricted_k, dim=(-2, -1), norm="ortho"
    )
    generator = torch.Generator(device=restricted_k.device)
    generator.manual_seed(int(seed))
    random_phase = torch.fft.fftn(
        torch.randn(
            (int(repeats), size_y, size_x),
            dtype=real_dtype,
            device=restricted_k.device,
            generator=generator,
        ),
        dim=(-2, -1),
        norm="ortho",
    )
    random_phase /= torch.clamp(
        torch.abs(random_phase), min=torch.finfo(real_dtype).tiny
    )
    surrogate_k = torch.fft.ifftn(
        spatial_spectrum[None, ...] * random_phase[:, None, ...],
        dim=(-2, -1),
        norm="ortho",
    ).real
    return surrogate_k * k2jy[None, ...]


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if int(args.surrogate_repeats) < 1:
        raise ValueError("surrogate-repeats must be positive")
    localized_block_counts = sorted(
        set(int(value) for value in (args.localized_block_count or []))
    )
    if any(value < 1 for value in localized_block_counts):
        raise ValueError("localized-block-count values must be positive")
    if localized_block_counts and int(args.localized_repeats) < 1:
        raise ValueError("localized-repeats must be positive")
    if int(args.spectral_coherence_repeats) < 0:
        raise ValueError("spectral-coherence-repeats must be non-negative")
    base_paths = _parse_labeled_paths(args.base_result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frequency_config = json.loads(
        args.frequency_config.read_text(encoding="utf-8")
    )
    resolved = resolve_mode_first_analysis(config)
    frequency_resolved = resolve_mode_first_analysis(frequency_config)
    analysis_frequencies_mhz = np.asarray(
        resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    frequencies_mhz = np.asarray(
        frequency_resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    frequencies_hz = frequencies_mhz * 1e6
    analysis_indices = _analysis_frequency_indices(
        frequencies_mhz, analysis_frequencies_mhz
    )
    bank, _ = _load_bank(args.bank_dir)
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        sky = {name: np.asarray(archive[name]) for name in archive.files}
    if not np.allclose(
        bank["frequencies_hz"], frequencies_hz, rtol=0.0, atol=1e-3
    ):
        raise ValueError("Visibility bank frequencies differ from config")
    if not np.allclose(
        sky["frequencies_mhz"], frequencies_mhz, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Sky cache frequencies differ from config")

    bases: dict[str, dict[str, Any]] = {}
    selected_rows: np.ndarray | None = None
    source_band_ids: np.ndarray | None = None
    for label, directory in base_paths.items():
        with np.load(directory / "result.npz", allow_pickle=False) as archive:
            data = {name: np.asarray(archive[name]) for name in archive.files}
        metadata = json.loads(
            (directory / "result.json").read_text(encoding="utf-8")
        )
        current_rows = np.asarray(data["selected_bank_rows"], dtype=np.int64)
        current_source_ids = np.asarray(
            data["source_band_ids"], dtype=np.int64
        )
        if selected_rows is None:
            selected_rows = current_rows
            source_band_ids = current_source_ids
        elif not np.array_equal(current_rows, selected_rows):
            raise ValueError("Base-result row selections differ")
        elif not np.array_equal(current_source_ids, source_band_ids):
            raise ValueError("Base-result source bands differ")
        bases[label] = {"arrays": data, "metadata": metadata}
    assert selected_rows is not None
    assert source_band_ids is not None

    uvw = np.asarray(bank["sample_uvw_m"][selected_rows], dtype=np.float64)
    reference_frequency_hz = (
        float(resolved.geometry["reference_frequency_mhz"]) * 1e6
    )
    row_kperp = _row_kperp(
        uvw,
        reference_frequency_hz=reference_frequency_hz,
        transverse_distance_mpc=float(
            resolved.geometry["transverse_distance_mpc"]
        ),
    )
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    radial_mpc_per_hz = float(
        resolved.geometry["radial_spacing_mpc"]
    ) / float(np.mean(np.diff(frequencies_hz)))
    maximum_delays = _maximum_patch_delays(
        kperp_edges=kperp_edges,
        transverse_distance_mpc=float(
            resolved.geometry["transverse_distance_mpc"]
        ),
        reference_frequency_hz=reference_frequency_hz,
        source_corner_angle_deg=float(
            resolved.geometry["source_corner_angle_deg"]
        ),
        wedge_buffer_mpc_inv=float(
            resolved.geometry["wedge_buffer_mpc_inv"]
        ),
        radial_mpc_per_hz=radial_mpc_per_hz,
    )

    import torch

    device = torch.device(str(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Amplitude-phase propagation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    real_dtype = (
        torch.float32
        if str(args.operator_dtype) == "complex64"
        else torch.float64
    )

    cache_dirs = [
        _format_pattern(
            str(args.aperture_row_beam_cache_pattern),
            float(frequency_mhz),
        )
        for frequency_mhz in frequencies_mhz
    ]
    (
        beam_multiplier,
        cache_metadata,
        _,
    ) = open_indexed_frequency_row_direction_kernel_multiplier(
        cache_dirs, selected_bank_rows=selected_rows
    )
    for frequency_hz, metadata in zip(
        frequencies_hz, cache_metadata, strict=True
    ):
        if (
            tuple(int(value) for value in metadata["shape"])[1]
            != int(sky["l_cosine"].size)
            or not np.isclose(
                float(metadata["frequency_hz"]),
                float(frequency_hz),
                rtol=0.0,
                atol=1e-3,
            )
        ):
            raise ValueError("Aperture cache geometry or frequency differs")

    source_size = int(config["image_geometry"]["source_image_size"])
    source_layout = build_sky_band_layout(
        (frequencies_hz.size, source_size, source_size),
        dx_mpc=float(resolved.contract.full_layout.dx_mpc),
        dy_mpc=float(resolved.contract.full_layout.dy_mpc),
        dpar_mpc=float(resolved.contract.full_layout.dpar_mpc),
        kperp_edges=kperp_edges,
        exclude_radial_nyquist=False,
    )
    first_arrays = next(iter(bases.values()))["arrays"]
    if (
        source_layout.band_count <= int(np.max(source_band_ids))
        or not np.array_equal(
            source_layout.active_kperp_indices[source_band_ids],
            first_arrays["source_band_kperp_indices"],
        )
        or not np.array_equal(
            source_layout.active_kpar_indices[source_band_ids],
            first_arrays["source_band_kpar_indices"],
        )
    ):
        raise ValueError("Reconstructed sky-band layout differs from base")

    eor_jy = torch.as_tensor(
        sky["eor_jy"].reshape(
            frequencies_hz.size, source_size, source_size
        ),
        dtype=real_dtype,
        device=device,
    )
    k2jy = torch.as_tensor(
        sky["k2jy_per_pixel"],
        dtype=real_dtype,
        device=device,
    )[:, None, None]
    eor_k = eor_jy / k2jy
    eor_spectrum = torch.fft.fftn(
        eor_k, dim=(-3, -2, -1), norm="ortho"
    )
    source_mask = torch.as_tensor(
        np.isin(source_layout.mode_bands, source_band_ids),
        dtype=torch.bool,
        device=device,
    )
    restricted_spectrum = torch.where(
        source_mask, eor_spectrum, torch.zeros_like(eor_spectrum)
    )
    actual_amplitude = torch.abs(restricted_spectrum)
    restricted_k = torch.fft.ifftn(
        restricted_spectrum, dim=(-3, -2, -1), norm="ortho"
    ).real
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.surrogate_seed))
    random_spectrum = torch.fft.fftn(
        torch.randn(
            (int(args.surrogate_repeats), *source_layout.cube_shape),
            dtype=real_dtype,
            device=device,
            generator=generator,
        ),
        dim=(-3, -2, -1),
        norm="ortho",
    )
    random_spectrum /= torch.clamp(
        torch.abs(random_spectrum), min=torch.finfo(real_dtype).tiny
    )
    random_spectrum *= actual_amplitude[None, ...]
    surrogate_k = torch.fft.ifftn(
        random_spectrum, dim=(-3, -2, -1), norm="ortho"
    ).real
    sky_batches = [
        eor_jy[None, ...],
        (restricted_k * k2jy)[None, ...],
        surrogate_k * k2jy[None, ...],
    ]
    actual_amplitude_slice = slice(2, 2 + int(args.surrogate_repeats))
    localized_slices: dict[int, slice] = {}
    spectral_coherence_slice: slice | None = None
    next_sky_position = int(actual_amplitude_slice.stop)
    spectral_coherence = None
    if int(args.spectral_coherence_repeats) > 0:
        spectral_coherence = (
            _spectrally_coherent_spatial_phase_surrogates(
                torch=torch,
                restricted_k=restricted_k,
                k2jy=k2jy,
                repeats=int(args.spectral_coherence_repeats),
                seed=int(args.spectral_coherence_seed),
                real_dtype=real_dtype,
            )
        )
        sky_batches.append(spectral_coherence)
        spectral_coherence_slice = slice(
            next_sky_position,
            next_sky_position + int(args.spectral_coherence_repeats),
        )
        next_sky_position += int(args.spectral_coherence_repeats)
    localized = None
    for block_count in localized_block_counts:
        if frequencies_hz.size % int(block_count) != 0:
            raise ValueError(
                f"Localized block count {block_count} does not divide "
                f"{frequencies_hz.size} frequencies"
            )
        localized = _localized_bandpower_surrogates(
            torch=torch,
            eor_k=eor_k,
            k2jy=k2jy,
            kperp_edges=kperp_edges,
            dx_mpc=float(resolved.contract.full_layout.dx_mpc),
            dy_mpc=float(resolved.contract.full_layout.dy_mpc),
            dpar_mpc=float(resolved.contract.full_layout.dpar_mpc),
            block_count=int(block_count),
            repeats=int(args.localized_repeats),
            seed=int(args.localized_seed) + 1009 * int(block_count),
            real_dtype=real_dtype,
        )
        sky_batches.append(localized)
        localized_slices[block_count] = slice(
            next_sky_position,
            next_sky_position + int(args.localized_repeats),
        )
        next_sky_position += int(args.localized_repeats)
    propagated_skies = torch.cat(sky_batches, dim=0)
    del eor_k, eor_spectrum, restricted_spectrum
    del actual_amplitude, random_spectrum, surrogate_k, restricted_k
    del sky_batches
    del localized
    del spectral_coherence
    torch.cuda.empty_cache()

    operator_kwargs = {
        "frequencies_hz": frequencies_hz,
        "uvw_m": uvw,
        "l_cosine": sky["l_cosine"],
        "m_cosine": sky["m_cosine"],
        "n_minus_one": sky["n_minus_one"],
        "channel_bandwidth_hz": float(args.channel_bandwidth_hz),
        "integration_time_s": float(args.integration_time_s),
        "phase_dec_deg": float(args.phase_dec_deg),
        "device": device,
        "operator_dtype": str(args.operator_dtype),
        "row_chunk": int(args.row_chunk),
        "source_chunk": int(args.source_chunk),
        "kernel_multiplier": beam_multiplier,
    }
    propagated = _propagate_in_batches(
        torch=torch,
        skies=propagated_skies,
        batch_size=int(args.sky_batch_size),
        operator_kwargs=operator_kwargs,
        started=started,
    )
    target = np.asarray(
        bank["sample_eor"][:, selected_rows], dtype=np.complex128
    )
    operator_closure = _operator_closure_metrics(propagated[0], target)
    if float(operator_closure["relative_l2"]) > float(
        args.maximum_operator_closure
    ):
        raise RuntimeError(
            "Matrix-free PB operator does not close the OSKAR bank: "
            f"{operator_closure['relative_l2']:.6g}"
        )

    products: dict[str, np.ndarray] = {
        "selected_bank_rows": selected_rows,
        "surrogate_ids": np.arange(
            int(args.surrogate_repeats), dtype=np.int64
        ),
    }
    if spectral_coherence_slice is not None:
        products["spectral_coherence_ids"] = np.arange(
            int(args.spectral_coherence_repeats), dtype=np.int64
        )
    filter_summaries: dict[str, Any] = {}
    for label, base in bases.items():
        arrays = base["arrays"]
        settings = base["metadata"]["settings"]
        output_band_ids = np.asarray(
            arrays["output_band_ids"], dtype=np.int64
        )
        bandpower_kwargs = {
            "frequencies_hz": frequencies_hz,
            "analysis_frequency_indices": analysis_indices,
            "filter_bandwidth_scope": str(
                settings["filter_bandwidth_scope"]
            ),
            "row_kperp": row_kperp,
            "kperp_edges": kperp_edges,
            "maximum_delays_s": maximum_delays,
            "dpss_eigenvalue_threshold": float(
                settings["dpss_eigenvalue_threshold"]
            ),
            "foreground_filter": str(settings["foreground_filter"]),
            "suppression_strength": float(
                settings["suppression_strength"]
            ),
            "polynomial_degree": int(settings["polynomial_degree"]),
            "spectral_taper": str(settings["spectral_taper"]),
        }
        bandpowers, _, _, _, _ = _visibility_bandpowers(
            visibilities=propagated,
            **bandpower_kwargs,
        )
        flat_bandpowers = bandpowers.reshape(propagated.shape[0], -1)
        restricted_q = flat_bandpowers[1, output_band_ids]
        surrogate_q = flat_bandpowers[
            actual_amplitude_slice, :
        ][:, output_band_ids]
        base_restricted_q = np.asarray(arrays["restricted_eor_q"])
        restricted_q_closure = _relative_l2(
            restricted_q, base_restricted_q
        )
        if restricted_q_closure > float(
            args.maximum_restricted_q_closure
        ):
            raise RuntimeError(
                f"{label} restricted-Q closure failed: "
                f"{restricted_q_closure:.6g}"
            )
        products[f"{label}_output_band_ids"] = output_band_ids
        products[f"{label}_restricted_q"] = restricted_q
        products[f"{label}_actual_amplitude_random_phase_q"] = surrogate_q
        if spectral_coherence_slice is not None:
            products[
                f"{label}_spectral_coherence_random_spatial_phase_q"
            ] = flat_bandpowers[
                spectral_coherence_slice, :
            ][:, output_band_ids]
        for block_count, positions in localized_slices.items():
            products[
                f"{label}_localized_{block_count}block_random_phase_q"
            ] = flat_bandpowers[positions, :][:, output_band_ids]
        filter_summaries[label] = {
            "base_result": str(base_paths[label]),
            "filter_bandwidth_scope": str(
                settings["filter_bandwidth_scope"]
            ),
            "output_band_count": int(output_band_ids.size),
            "restricted_q_relative_l2_to_base": restricted_q_closure,
        }

    _atomic_npz(args.out_dir / "result.npz", products)
    _atomic_json(
        args.out_dir / "result.json",
        {
            "schema": "visibility_qbeta_amplitude_phase_surrogate_partition",
            "schema_version": 1,
            "base_results": {
                label: str(path) for label, path in base_paths.items()
            },
            "selected_row_count": int(selected_rows.size),
            "surrogate_repeats": int(args.surrogate_repeats),
            "surrogate_seed": int(args.surrogate_seed),
            "localized_block_counts": localized_block_counts,
            "localized_repeats": int(args.localized_repeats),
            "localized_seed": int(args.localized_seed),
            "spectral_coherence_repeats": int(
                args.spectral_coherence_repeats
            ),
            "spectral_coherence_seed": int(args.spectral_coherence_seed),
            "construction": {
                "amplitude": "actual restricted EoR Fourier-mode amplitude",
                "phase": "independent Hermitian random phase",
                "source_power_is_exactly_preserved": True,
                "localized": (
                    "piecewise-stationary equal-bandpower Hermitian random "
                    "phase within each requested contiguous frequency block"
                ),
                "spectral_coherence": (
                    "one random Hermitian 2D spatial phase per spatial mode, "
                    "shared by all frequencies so that the complete "
                    "cross-frequency complex vector is retained"
                ),
            },
            "operator_closure": operator_closure,
            "filters": filter_summaries,
            "operator": {
                "implementation": "matrix_free_exact_dft_with_cached_pb",
                "dtype": str(args.operator_dtype),
                "row_chunk": int(args.row_chunk),
                "source_chunk": int(args.source_chunk),
                "sky_batch_size": int(args.sky_batch_size),
                "device_name": str(torch.cuda.get_device_name()),
                "peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated() / 2**30
                ),
                "peak_reserved_gib": float(
                    torch.cuda.max_memory_reserved() / 2**30
                ),
            },
            "elapsed_seconds": float(time.monotonic() - started),
        },
    )


if __name__ == "__main__":
    main()
