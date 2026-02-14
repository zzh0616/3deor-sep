"""
Instrument / forward-operator utilities.

This module intentionally keeps the interface simple: a forward operator is
represented as a callable `op(x: Tensor) -> Tensor` that maps a cube in the
model domain to the observation domain.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def _ifftshift2(x: Tensor) -> Tensor:
    """
    Like numpy.fft.ifftshift for the last two (spatial) dims.

    We avoid relying on torch.fft.ifftshift availability/version differences.
    """

    if x.ndim < 2:
        raise ValueError(f"ifftshift2 expects at least 2 dims, got shape {tuple(x.shape)}")
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    return torch.roll(torch.roll(x, shifts=-(h // 2), dims=-2), shifts=-(w // 2), dims=-1)


def make_psf_convolution_operator(
    psf_cube: Tensor,
    *,
    freq_axis: int = 0,
    scale: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """
    Build a forward operator that applies per-frequency PSF convolution:

      y = scale * (x (*) psf)

    - `psf_cube` must be a real tensor with the same spatial shape as `x`.
    - `psf_cube` can have F=1 (broadcast) or F matching the input cube.
    - Convolution is implemented via FFT (circular convolution on the image grid).
    """

    if psf_cube.ndim != 3:
        raise ValueError(f"psf_cube must be 3D (F, Y, X) in some axis order, got {tuple(psf_cube.shape)}")
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    psf_perm = psf_cube.movedim(freq_axis, 0)  # (Fpsf, H, W)
    if psf_perm.shape[-2] < 2 or psf_perm.shape[-1] < 2:
        raise ValueError(f"psf spatial dims too small: {tuple(psf_perm.shape)}")

    psf_shifted = _ifftshift2(psf_perm)
    psf_fft = torch.fft.rfft2(psf_shifted, dim=(-2, -1))

    scale_val = float(scale)
    if not torch.isfinite(torch.tensor(scale_val)):
        raise ValueError("psf scale must be finite.")

    def _op(x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"PSF operator expects a 3D cube, got shape {tuple(x.shape)}")
        x_perm = x.movedim(freq_axis, 0)  # (F, H, W)
        if x_perm.shape[-2:] != psf_perm.shape[-2:]:
            raise ValueError(
                "Input spatial shape does not match PSF spatial shape: "
                f"x={tuple(x_perm.shape[-2:])}, psf={tuple(psf_perm.shape[-2:])}"
            )
        if psf_fft.shape[0] not in (1, x_perm.shape[0]):
            raise ValueError(
                "PSF frequency length must be 1 or match input: "
                f"psf_F={int(psf_fft.shape[0])}, x_F={int(x_perm.shape[0])}"
            )

        x_fft = torch.fft.rfft2(x_perm, dim=(-2, -1))
        y_perm = torch.fft.irfft2(x_fft * psf_fft, s=x_perm.shape[-2:], dim=(-2, -1))
        if scale_val != 1.0:
            y_perm = y_perm * scale_val
        return y_perm.movedim(0, freq_axis)

    return _op

