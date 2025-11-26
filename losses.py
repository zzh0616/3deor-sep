#!/usr/bin/env python3
"""
Loss computation utilities for foreground/EoR separation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from constants import EPS_LOSS, DEFAULT_FFT_SIGMA
from utils import clamp_eps, ensure_tensor_on, prepare_broadcastable_prior

Tensor = torch.Tensor


def forward_model(fg: Tensor, eor: Tensor, psf: Optional[callable] = None) -> Tensor:
    """
    Simple forward model for forming the observed cube.
    """
    combined = fg + eor
    if psf is None:
        return combined
    if callable(psf):
        return psf(combined)
    raise TypeError("psf must be None or a callable that maps a tensor to a tensor.")


def foreground_smoothness_loss(
    fg: Tensor,
    freq_axis: int = 0,
    prior_mean: Optional[Tensor] = None,
    prior_sigma: Optional[Tensor] = None,
) -> Tensor:
    """
    Normalized mean squared third-order finite differences along the frequency axis.
    """
    if fg.shape[freq_axis] < 4:
        return torch.zeros((), dtype=fg.dtype, device=fg.device)
    third_diff = torch.diff(fg, n=3, dim=freq_axis)
    mean_tensor = prepare_broadcastable_prior(prior_mean, third_diff, "fg_smooth_mean")
    sigma_tensor = prepare_broadcastable_prior(prior_sigma, third_diff, "fg_smooth_sigma")
    if mean_tensor is None:
        mean_tensor = torch.zeros(1, device=third_diff.device, dtype=third_diff.dtype)
    if sigma_tensor is None:
        sigma_tensor = torch.ones(1, device=third_diff.device, dtype=third_diff.dtype)
    sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)
    normalized = (third_diff - mean_tensor) / sigma_tensor
    return torch.mean(normalized**2)


def correlation_penalty(
    fg: Tensor,
    eor: Tensor,
    prior_mean: Tensor,
    prior_sigma: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Penalize deviations of the FG/EoR correlation coefficient from a prior mean.
    """
    fg_flat = fg.reshape(-1)
    eor_flat = eor.reshape(-1)
    fg_centered = fg_flat - fg_flat.mean()
    eor_centered = eor_flat - eor_flat.mean()
    denom = torch.norm(fg_centered) * torch.norm(eor_centered)
    denom = torch.clamp(denom, min=1e-8)
    corr = torch.dot(fg_centered, eor_centered) / denom
    sigma = torch.clamp(prior_sigma, min=1e-8)
    penalty = ((corr - prior_mean) / sigma) ** 2
    return penalty, corr


def compute_highfreq_energy(
    tensor: Tensor, freq_axis: int, percent: float
) -> Tensor:
    """
    Compute high-frequency energy map from the rFFT along the frequency axis.
    """
    if not (0.0 <= percent <= 1.0):
        raise ValueError("fft_highfreq_percent must be in [0, 1].")
    rfft = torch.fft.rfft(tensor, dim=freq_axis)
    num_bins = rfft.shape[freq_axis]
    if num_bins == 0:
        return torch.zeros_like(tensor.sum(dim=freq_axis))
    # percent means the fraction of highest-frequency bins to penalize.
    num_high = max(1, int(math.ceil(percent * num_bins)))
    start_idx = max(num_bins - num_high, 0)
    freq_slice = [slice(None)] * rfft.ndim
    freq_slice[freq_axis] = slice(start_idx, None)
    highfreq = rfft[tuple(freq_slice)]
    power = highfreq.real**2 + highfreq.imag**2
    energy = power.mean(dim=freq_axis)
    return energy


def derive_fft_prior_from_cube(
    fg_cube: Tensor,
    freq_axis: int,
    percent: float,
    use_robust: bool = False,
    mae_to_sigma_factor: float = 1.4826,
) -> Tuple[Tensor, Tensor]:
    energy_map = compute_highfreq_energy(fg_cube, freq_axis=freq_axis, percent=percent)
    if use_robust:
        median = energy_map.median().view(1)
        mae = (energy_map - median).abs().mean().view(1)
        sigma = torch.clamp(mae * mae_to_sigma_factor, min=1e-6)
        return median, sigma
    mean = energy_map.mean().view(1)
    std = torch.clamp(energy_map.std(), min=1e-6).view(1)
    return mean, std


def polynomial_prior_loss(
    fg: Tensor,
    freq_axis: int,
    degree: int,
    sigma: Union[Tensor, float],
    freqs: Optional[Tensor] = None,
) -> Tensor:
    """
    Fit a low-order polynomial along frequency and penalize residuals.
    """
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative.")
    fg_moved = fg.movedim(freq_axis, 0)
    num_freqs = fg_moved.shape[0]
    if freqs is None:
        coords = torch.linspace(0.0, 1.0, num_freqs, device=fg.device, dtype=fg.dtype)
    else:
        if freqs.numel() != num_freqs:
            raise ValueError("Frequency array length does not match foreground frequency dimension.")
        freqs = freqs.to(device=fg.device, dtype=fg.dtype)
        f_min = freqs.min()
        f_max = freqs.max()
        scale = torch.clamp(f_max - f_min, min=1e-8)
        coords = (freqs - f_min) / scale
    design = torch.stack([coords**i for i in range(degree + 1)], dim=1)  # (F, degree+1)
    y_flat = fg_moved.reshape(num_freqs, -1)
    safe_dtype = torch.float32 if fg.dtype not in (torch.float32, torch.float64) else fg.dtype
    y_flat_cast = y_flat.to(dtype=safe_dtype)
    design_cast = design.to(dtype=safe_dtype)
    coeffs = torch.linalg.lstsq(design_cast, y_flat_cast).solution  # (degree+1, Npix)
    coeffs = coeffs.to(dtype=fg.dtype)
    fitted = (design_cast.to(dtype=fg.dtype) @ coeffs).reshape_as(fg_moved)
    residual = fg_moved - fitted
    residual = residual.movedim(0, freq_axis)
    sigma_tensor = prepare_broadcastable_prior(sigma, residual, "poly_sigma")
    if sigma_tensor is None:
        sigma_tensor = torch.ones(1, device=fg.device, dtype=fg.dtype)
    sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)
    return torch.mean((residual / sigma_tensor) ** 2)


def loss_function(
    fg: Tensor,
    eor: Tensor,
    y: Tensor,
    *,
    psf: Optional[callable] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    delta: float = 1.0,
    fft_weight: float = 1.0,
    poly_weight: float = 1.0,
    freq_axis: int = 0,
    data_error: Tensor,
    eor_mean: Tensor,
    eor_sigma: Tensor,
    fg_smooth_mean: Optional[Tensor],
    fg_smooth_sigma: Optional[Tensor],
    corr_prior_mean: Tensor,
    corr_prior_sigma: Tensor,
    loss_mode: str = "base",
    fft_prior_mean: Optional[Tensor] = None,
    fft_prior_sigma: Optional[Tensor] = None,
    fft_percent: float = 0.7,
    poly_degree: int = 3,
    poly_sigma: Optional[Union[Tensor, float]] = None,
    poly_residual: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Combined loss used for optimizing foreground and EoR components.
    """
    y_pred = forward_model(fg, eor, psf=psf)
    sigma_data = clamp_eps(data_error, eps=EPS_LOSS)
    data_loss = torch.mean(((y_pred - y) / sigma_data) ** 2)

    smooth_loss = foreground_smoothness_loss(
        fg, freq_axis=freq_axis, prior_mean=fg_smooth_mean, prior_sigma=fg_smooth_sigma
    )

    eor_mean_tensor = eor_mean
    eor_sigma_tensor = clamp_eps(eor_sigma, eps=EPS_LOSS)
    eor_reg = torch.mean(((eor - eor_mean_tensor) / eor_sigma_tensor) ** 2)

    corr_loss, corr_coeff = correlation_penalty(fg, eor, corr_prior_mean, corr_prior_sigma)

    fft_loss = torch.zeros_like(data_loss)
    if loss_mode == "rfft":
        prior_mean = fft_prior_mean
        prior_sigma = (
            clamp_eps(fft_prior_sigma, eps=EPS_LOSS) if fft_prior_sigma is not None else None
        )
        energy_map = compute_highfreq_energy(fg, freq_axis=freq_axis, percent=fft_percent)
        if prior_mean is None:
            prior_mean = torch.zeros(1, device=energy_map.device, dtype=energy_map.dtype)
        if prior_sigma is None:
            prior_sigma = torch.full(
                (1,),
                DEFAULT_FFT_SIGMA,
                device=energy_map.device,
                dtype=energy_map.dtype,
            )
        fft_loss = torch.mean(((energy_map - prior_mean) / prior_sigma) ** 2)
    poly_loss = torch.zeros_like(data_loss)
    if loss_mode == "poly":
        poly_loss = polynomial_prior_loss(
            fg,
            freq_axis=freq_axis,
            degree=poly_degree,
            sigma=poly_sigma,
            freqs=None,
        )
    elif loss_mode == "poly_reparam":
        if poly_residual is None:
            poly_residual = torch.zeros_like(fg)
        sigma_tensor = prepare_broadcastable_prior(poly_sigma, poly_residual, "poly_sigma_reparam")
        if sigma_tensor is None:
            sigma_tensor = torch.ones(1, device=fg.device, dtype=fg.dtype)
        sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)
        poly_loss = torch.mean((poly_residual / sigma_tensor) ** 2)

    total_loss = (
        alpha * data_loss
        + beta * smooth_loss
        + gamma * eor_reg
        + delta * corr_loss
        + fft_weight * fft_loss
        + poly_weight * poly_loss
    )
    return total_loss, {
        "data": data_loss,
        "smooth": smooth_loss,
        "eor_reg": eor_reg,
        "corr": corr_loss,
        "corr_coeff": corr_coeff,
        "fft_highfreq": fft_loss,
        "poly": poly_loss,
    }
