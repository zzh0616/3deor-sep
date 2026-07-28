#!/usr/bin/env python3
"""Memory-bounded direct visibility operators compatible with OSKAR banks."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from visibility_qbeta import C_M_S, OMEGA_EARTH_RAD_S


KernelMultiplier = Callable[
    [int, int, int, int, int, Any, Any],
    Any,
]
ProgressCallback = Callable[[int, float], None]


def apply_exact_visibility_operator_matrix_free(
    *,
    torch: Any,
    frequencies_hz: np.ndarray,
    uvw_m: np.ndarray,
    l_cosine: np.ndarray,
    m_cosine: np.ndarray,
    n_minus_one: np.ndarray,
    sky_jy: Any,
    channel_bandwidth_hz: float,
    integration_time_s: float,
    phase_dec_deg: float,
    device: Any,
    operator_dtype: str,
    row_chunk: int,
    source_chunk: int,
    kernel_multiplier: KernelMultiplier | None = None,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Apply the exact DFT kernel without materializing its full matrix.

    ``kernel_multiplier`` is reserved for direction-, time-, frequency-, and
    baseline-dependent effects such as a primary-beam coherency product.
    """
    complex_dtype = (
        torch.complex64
        if str(operator_dtype) == "complex64"
        else torch.complex128
    )
    real_dtype = (
        torch.float32
        if complex_dtype == torch.complex64
        else torch.float64
    )
    phase_dtype = torch.float64
    frequencies = torch.as_tensor(
        frequencies_hz, dtype=phase_dtype, device=device
    ).reshape(-1)
    uvw = torch.as_tensor(uvw_m, dtype=phase_dtype, device=device)
    l_tensor = torch.as_tensor(
        l_cosine, dtype=phase_dtype, device=device
    ).reshape(-1)
    m_tensor = torch.as_tensor(
        m_cosine, dtype=phase_dtype, device=device
    ).reshape(-1)
    n_tensor = torch.as_tensor(
        n_minus_one, dtype=phase_dtype, device=device
    ).reshape(-1)
    sky = torch.as_tensor(sky_jy, dtype=real_dtype, device=device)
    if sky.ndim == 3:
        sky = sky.unsqueeze(0)
        squeeze = True
    elif sky.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            "Sky tensor must have shape [freq,y,x] or [batch,freq,y,x]"
        )
    batch, n_frequency, _, _ = sky.shape
    n_row = int(uvw.shape[0])
    n_source = int(l_tensor.numel())
    if (
        int(frequencies.numel()) != int(n_frequency)
        or int(sky.shape[-2] * sky.shape[-1]) != n_source
        or uvw.ndim != 2
        or uvw.shape[1] != 3
        or int(row_chunk) < 1
        or int(source_chunk) < 1
    ):
        raise ValueError("Frequency, row, source, or chunk geometry differs")

    output = torch.empty(
        (int(batch), int(n_frequency), n_row),
        dtype=complex_dtype,
        device=device,
    )
    dec0 = math.radians(float(phase_dec_deg))
    for frequency_index in range(int(n_frequency)):
        frequency = frequencies[frequency_index]
        frequency_sky = sky[:, frequency_index].reshape(batch, -1).to(
            complex_dtype
        )
        for row_first in range(0, n_row, int(row_chunk)):
            row_stop = min(n_row, row_first + int(row_chunk))
            uvw_block = uvw[row_first:row_stop]
            u = uvw_block[:, 0:1]
            v = uvw_block[:, 1:2]
            w = uvw_block[:, 2:3]
            transverse = -math.sin(dec0) * v + math.cos(dec0) * w
            accumulated = torch.zeros(
                (int(batch), row_stop - row_first),
                dtype=complex_dtype,
                device=device,
            )
            for source_first in range(0, n_source, int(source_chunk)):
                source_stop = min(
                    n_source, source_first + int(source_chunk)
                )
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
                ).to(complex_dtype)
                if kernel_multiplier is not None:
                    multiplier = kernel_multiplier(
                        int(frequency_index),
                        int(row_first),
                        int(row_stop),
                        int(source_first),
                        int(source_stop),
                        device,
                        complex_dtype,
                    )
                    kernel = kernel * torch.as_tensor(
                        multiplier, dtype=complex_dtype, device=device
                    )
                accumulated.add_(
                    frequency_sky[:, source_first:source_stop]
                    @ kernel.transpose(0, 1)
                )
            output[:, frequency_index, row_first:row_stop] = accumulated
        if progress_callback is not None:
            progress_callback(
                int(frequency_index), float(frequencies_hz[frequency_index])
            )
    result = np.asarray(output.detach().cpu())
    return result[0] if squeeze else result
