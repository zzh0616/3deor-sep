"""
Instrument / forward-operator utilities.

This module intentionally keeps the interface simple: a forward operator is
represented as a callable `op(x: Tensor) -> Tensor` that maps a cube in the
model domain to the observation domain.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
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


def _center_pad2(x: Tensor, target_hw: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int, int, int]]:
    """
    Center-pad the last two dims (H, W) of `x` up to `target_hw`.

    Returns (x_padded, (pad_left, pad_right, pad_top, pad_bottom)).
    """

    if x.ndim < 2:
        raise ValueError(f"center_pad2 expects at least 2 dims, got shape {tuple(x.shape)}")
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if th < h or tw < w:
        raise ValueError(f"target_hw must be >= input hw, got target=({th},{tw}) input=({h},{w})")
    pad_h = th - h
    pad_w = tw - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)


def _center_crop2(x: Tensor, out_hw: Tuple[int, int]) -> Tensor:
    """
    Center-crop the last two dims (H, W) of `x` to `out_hw`.
    """

    if x.ndim < 2:
        raise ValueError(f"center_crop2 expects at least 2 dims, got shape {tuple(x.shape)}")
    oh, ow = int(out_hw[0]), int(out_hw[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if oh > h or ow > w:
        raise ValueError(f"out_hw must be <= input hw, got out=({oh},{ow}) input=({h},{w})")
    top = (h - oh) // 2
    left = (w - ow) // 2
    return x[..., top : top + oh, left : left + ow]


def make_multiplicative_operator(
    multiplier_cube: Tensor,
    *,
    freq_axis: int = 0,
    scale: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """
    Build a forward operator that multiplies by a per-frequency (or broadcast) cube:

      y = scale * (x ⊙ m)

    - `multiplier_cube` can be 2D (H, W) or 3D with a frequency axis.
    - If 3D, `multiplier_cube` must have F=1 (broadcast) or F matching the input cube.
    """

    if multiplier_cube.ndim not in (2, 3):
        raise ValueError(
            f"multiplier_cube must be 2D or 3D, got shape {tuple(multiplier_cube.shape)}"
        )
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    scale_val = float(scale)
    if not torch.isfinite(torch.tensor(scale_val)):
        raise ValueError("multiplier scale must be finite.")

    if multiplier_cube.ndim == 2:
        m2 = multiplier_cube

        def _op2(x: Tensor) -> Tensor:
            if x.ndim != 3:
                raise ValueError(f"Multiplier operator expects a 3D cube, got shape {tuple(x.shape)}")
            spatial_axes = [ax for ax in range(3) if ax != freq_axis]
            x_axis, y_axis = spatial_axes[0], spatial_axes[1]
            expected = (int(x.shape[x_axis]), int(x.shape[y_axis]))
            if tuple(m2.shape) != expected:
                raise ValueError(
                    f"2D multiplier shape {tuple(m2.shape)} does not match spatial shape {expected}"
                )
            m3 = m2.to(device=x.device, dtype=x.dtype).unsqueeze(freq_axis).expand_as(x)
            y = x * m3
            if scale_val != 1.0:
                y = y * scale_val
            return y

        return _op2

    m_perm = multiplier_cube.movedim(freq_axis, 0)  # (Fm, H, W)
    if m_perm.shape[-2] < 2 or m_perm.shape[-1] < 2:
        raise ValueError(f"multiplier spatial dims too small: {tuple(m_perm.shape)}")

    def _op3(x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Multiplier operator expects a 3D cube, got shape {tuple(x.shape)}")
        x_perm = x.movedim(freq_axis, 0)  # (F, H, W)
        if x_perm.shape[-2:] != m_perm.shape[-2:]:
            raise ValueError(
                "Input spatial shape does not match multiplier spatial shape: "
                f"x={tuple(x_perm.shape[-2:])}, m={tuple(m_perm.shape[-2:])}"
            )
        if m_perm.shape[0] not in (1, x_perm.shape[0]):
            raise ValueError(
                "Multiplier frequency length must be 1 or match input: "
                f"m_F={int(m_perm.shape[0])}, x_F={int(x_perm.shape[0])}"
            )
        m_cast = m_perm.to(device=x_perm.device, dtype=x_perm.dtype)
        if m_cast.shape[0] == 1 and x_perm.shape[0] != 1:
            m_cast = m_cast.expand(x_perm.shape[0], -1, -1)
        y_perm = x_perm * m_cast
        if scale_val != 1.0:
            y_perm = y_perm * scale_val
        return y_perm.movedim(0, freq_axis)

    return _op3


def compose_operators(*ops: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """
    Compose forward operators in order: compose(op1, op2)(x) = op2(op1(x)).
    """

    valid_ops: Sequence[Callable[[Tensor], Tensor]] = [op for op in ops if op is not None]
    if not valid_ops:
        raise ValueError("compose_operators expects at least one operator")

    def _op(x: Tensor) -> Tensor:
        y = x
        for op in valid_ops:
            y = op(y)
        return y

    return _op


def make_psf_convolution_operator(
    psf_cube: Tensor,
    *,
    freq_axis: int = 0,
    scale: float = 1.0,
    pad_to: Optional[Union[int, Sequence[int]]] = None,
) -> Callable[[Tensor], Tensor]:
    """
    Build a forward operator that applies per-frequency PSF convolution:

      y = scale * (x (*) psf)

    - `psf_cube` must be a real tensor with the same spatial shape as `x`.
    - `psf_cube` can have F=1 (broadcast) or F matching the input cube.
    - Convolution is implemented via FFT. By default this is *circular* convolution on
      the image grid. When `pad_to` is provided, both `x` and `psf` are center-padded to
      the specified size, convolved, and center-cropped back (a closer match to WSClean's
      padded inversion).
    """

    if psf_cube.ndim != 3:
        raise ValueError(f"psf_cube must be 3D (F, Y, X) in some axis order, got {tuple(psf_cube.shape)}")
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    psf_perm = psf_cube.movedim(freq_axis, 0)  # (Fpsf, H, W)
    if psf_perm.shape[-2] < 2 or psf_perm.shape[-1] < 2:
        raise ValueError(f"psf spatial dims too small: {tuple(psf_perm.shape)}")

    in_hw = (int(psf_perm.shape[-2]), int(psf_perm.shape[-1]))
    pad_hw = in_hw
    if pad_to is not None:
        if isinstance(pad_to, int):
            pad_hw = (int(pad_to), int(pad_to))
        else:
            pad_to_seq = list(pad_to)
            if len(pad_to_seq) != 2:
                raise ValueError("pad_to must be an int or a 2-sequence (H, W).")
            pad_hw = (int(pad_to_seq[0]), int(pad_to_seq[1]))
        if pad_hw[0] < in_hw[0] or pad_hw[1] < in_hw[1]:
            raise ValueError(f"pad_to must be >= PSF spatial size, got pad_to={pad_hw}, psf={in_hw}")

    psf_pad = psf_perm
    if pad_hw != in_hw:
        psf_pad, _ = _center_pad2(psf_perm, pad_hw)
    psf_shifted = _ifftshift2(psf_pad)
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

        x_work = x_perm
        if pad_hw != in_hw:
            x_work, _ = _center_pad2(x_perm, pad_hw)

        x_fft = torch.fft.rfft2(x_work, dim=(-2, -1))
        y_work = torch.fft.irfft2(x_fft * psf_fft, s=pad_hw, dim=(-2, -1))
        y_perm = _center_crop2(y_work, in_hw) if pad_hw != in_hw else y_work
        if scale_val != 1.0:
            y_perm = y_perm * scale_val
        return y_perm.movedim(0, freq_axis)

    return _op
