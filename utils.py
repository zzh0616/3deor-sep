#!/usr/bin/env python3
"""
Utility helpers shared across modules.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from constants import EPS_LOSS, EPS_STD

Tensor = torch.Tensor


def ensure_tensor_on(
    value: Optional[Union[Tensor, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def prepare_broadcastable_prior(
    value: Optional[Union[Tensor, float]],
    reference: Tensor,
    name: str,
) -> Optional[Tensor]:
    if value is None:
        return None
    tensor = ensure_tensor_on(value, reference.device, reference.dtype)
    if tensor.ndim == 0:
        return tensor
    try:
        torch.broadcast_shapes(reference.shape, tensor.shape)
    except RuntimeError as exc:  # pragma: no cover - broadcaster
        raise ValueError(
            f"{name} with shape {tuple(tensor.shape)} is not broadcastable to {tuple(reference.shape)}"
        ) from exc
    return tensor


def clamp_eps(value: torch.Tensor, eps: float = EPS_LOSS) -> torch.Tensor:
    return torch.clamp(value, min=eps)


def clamp_std(value: torch.Tensor, eps: float = EPS_STD) -> torch.Tensor:
    return torch.clamp(value, min=eps)
