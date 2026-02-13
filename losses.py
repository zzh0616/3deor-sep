#!/usr/bin/env python3
"""
Loss computation utilities for foreground/EoR separation.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from constants import EPS_LOSS, DEFAULT_FFT_SIGMA
from utils import clamp_eps, ensure_tensor_on, prepare_broadcastable_prior

Tensor = torch.Tensor
VALID_EXTRA_LOSS_TERMS: Tuple[str, ...] = (
    "corr",
    "rfft",
    "poly_reparam",
    "lagcorr",
    "eor_mean",
    "eor_hf",
)
VALID_FG_SMOOTH_MODES: Tuple[str, ...] = (
    "diff3_l2",
    "diff2_l2",
    "diff2_huber",
    "diff1_l1",
)


def normalize_extra_loss_terms(
    *,
    loss_mode: Optional[str] = None,
    extra_loss_terms: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[str, ...]:
    """
    Normalize active extra loss terms.

    Backward compatibility:
      - legacy `loss_mode` values ("base"/single extra term) still work.
      - comma/plus separated strings are supported.
    """
    if extra_loss_terms is None:
        if loss_mode is None:
            return ()
        raw_items: List[str] = [str(loss_mode)]
    elif isinstance(extra_loss_terms, str):
        raw_items = [extra_loss_terms]
    else:
        raw_items = [str(item) for item in extra_loss_terms]

    normalized: List[str] = []
    seen = set()
    for raw in raw_items:
        for token in re.split(r"[,+]", raw):
            term = token.strip().lower()
            if not term or term == "base":
                continue
            if term not in VALID_EXTRA_LOSS_TERMS:
                valid = ", ".join(["base", *VALID_EXTRA_LOSS_TERMS])
                raise ValueError(f"Unsupported loss term '{term}'. Valid values: {valid}.")
            if term not in seen:
                seen.add(term)
                normalized.append(term)
    return tuple(normalized)


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
    mode: str = "diff3_l2",
    huber_delta: float = 1.0,
) -> Tensor:
    """
    Frequency-domain smoothness priors for the foreground component.

    Supported modes:
      - diff3_l2: L2 on normalized 3rd-order differences (legacy behavior)
      - diff2_l2: L2 on normalized 2nd-order differences
      - diff2_huber: Huber on normalized 2nd-order differences
      - diff1_l1: L1 on normalized 1st-order differences
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm == "diff3_l2":
        diff_order, penalty = 3, "l2"
    elif mode_norm == "diff2_l2":
        diff_order, penalty = 2, "l2"
    elif mode_norm == "diff2_huber":
        diff_order, penalty = 2, "huber"
    elif mode_norm == "diff1_l1":
        diff_order, penalty = 1, "l1"
    else:
        valid = ", ".join(VALID_FG_SMOOTH_MODES)
        raise ValueError(f"Unsupported fg_smooth_mode '{mode}'. Valid values: {valid}.")

    if fg.shape[freq_axis] < (diff_order + 1):
        return torch.zeros((), dtype=fg.dtype, device=fg.device)
    diff_tensor = torch.diff(fg, n=diff_order, dim=freq_axis)
    mean_tensor = prepare_broadcastable_prior(prior_mean, diff_tensor, "fg_smooth_mean")
    sigma_tensor = prepare_broadcastable_prior(prior_sigma, diff_tensor, "fg_smooth_sigma")
    if mean_tensor is None:
        mean_tensor = torch.zeros(1, device=diff_tensor.device, dtype=diff_tensor.dtype)
    if sigma_tensor is None:
        sigma_tensor = torch.ones(1, device=diff_tensor.device, dtype=diff_tensor.dtype)
    sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)
    normalized = (diff_tensor - mean_tensor) / sigma_tensor
    if penalty == "l2":
        return torch.mean(normalized**2)
    if penalty == "l1":
        return torch.mean(torch.abs(normalized))
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("fg_smooth_huber_delta must be a finite positive value.")
    return F.huber_loss(normalized, torch.zeros_like(normalized), reduction="mean", delta=delta)


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


def frequency_slice_correlation_penalty(
    fg: Tensor,
    eor: Tensor,
    *,
    freq_axis: int,
    prior_mean: Tensor,
    prior_sigma: Tensor,
    prior_abs_threshold: Optional[Tensor] = None,
    reduce: str = "mean",
    topk: Optional[int] = None,
    lse_alpha: float = 10.0,
) -> Tuple[Tensor, Tensor]:
    """
    Penalize FG/EoR correlation per frequency slice (no cross-frequency coupling).

    The correlation is computed independently at each frequency slice over spatial
    pixels, then averaged:

        residual_f = max(|corr_f - mu_f| - t_f, 0) / sigma_f
        loss = Reduce_f [ residual_f^2 ].

    Reduce options:
      - mean: average over frequency slices
      - topk: mean over largest-k residuals (requires topk>0)
      - logsumexp: smooth max via log-mean-exp (zero when all residuals are zero)
    """
    if fg.shape != eor.shape:
        raise ValueError(
            f"fg/eor shape mismatch for correlation penalty: {tuple(fg.shape)} vs {tuple(eor.shape)}"
        )
    if fg.ndim != 3:
        raise ValueError(f"Expected 3D cubes for correlation penalty, got shape {tuple(fg.shape)}")

    fg_moved = fg.movedim(freq_axis, 0)
    eor_moved = eor.movedim(freq_axis, 0)
    fg_flat = fg_moved.reshape(fg_moved.shape[0], -1)
    eor_flat = eor_moved.reshape(eor_moved.shape[0], -1)

    fg_centered = fg_flat - fg_flat.mean(dim=1, keepdim=True)
    eor_centered = eor_flat - eor_flat.mean(dim=1, keepdim=True)
    dot = torch.sum(fg_centered * eor_centered, dim=1)
    denom = torch.norm(fg_centered, dim=1) * torch.norm(eor_centered, dim=1)
    denom = torch.clamp(denom, min=1e-8)
    corr_per_freq = dot / denom

    mean_tensor = prepare_broadcastable_prior(prior_mean, corr_per_freq, "corr_prior_mean")
    sigma_tensor = prepare_broadcastable_prior(prior_sigma, corr_per_freq, "corr_prior_sigma")
    threshold_tensor = prepare_broadcastable_prior(
        prior_abs_threshold, corr_per_freq, "corr_prior_abs_threshold"
    )
    if mean_tensor is None:
        mean_tensor = torch.zeros(1, device=corr_per_freq.device, dtype=corr_per_freq.dtype)
    if sigma_tensor is None:
        sigma_tensor = torch.ones(1, device=corr_per_freq.device, dtype=corr_per_freq.dtype)
    if threshold_tensor is None:
        threshold_tensor = torch.zeros(1, device=corr_per_freq.device, dtype=corr_per_freq.dtype)
    if torch.any(~torch.isfinite(threshold_tensor)):
        raise ValueError("corr_prior_abs_threshold must be finite.")
    if torch.any(threshold_tensor < 0.0):
        raise ValueError("corr_prior_abs_threshold must be non-negative.")
    sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)

    residual = torch.clamp(torch.abs(corr_per_freq - mean_tensor) - threshold_tensor, min=0.0)
    penalty_per_freq = (residual / sigma_tensor) ** 2
    reduce_norm = str(reduce).strip().lower()
    if reduce_norm == "mean":
        penalty = torch.mean(penalty_per_freq)
    elif reduce_norm == "topk":
        if topk is None:
            raise ValueError("corr reduce=topk requires topk to be set.")
        k = int(topk)
        if k <= 0:
            raise ValueError("corr reduce=topk requires topk>0.")
        k = min(k, int(penalty_per_freq.numel()))
        penalty = torch.mean(torch.topk(penalty_per_freq, k=k, largest=True).values)
    elif reduce_norm == "logsumexp":
        alpha = float(lse_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("corr reduce=logsumexp requires lse_alpha to be a finite positive value.")
        n = int(penalty_per_freq.numel())
        # log-mean-exp (LME) keeps penalty=0 when all residuals are 0.
        penalty = (torch.logsumexp(alpha * penalty_per_freq, dim=0) - math.log(max(1, n))) / alpha
    else:
        raise ValueError("corr reduce must be one of: mean, topk, logsumexp.")
    corr_mean = torch.mean(corr_per_freq)
    return penalty, corr_mean


def _frequency_lag_correlation_loss(
    cube: Tensor,
    *,
    freq_axis: int,
    lag_channels: Tensor,
    prior_mean: Tensor,
    prior_sigma: Tensor,
    lag_weights: Optional[Tensor] = None,
    max_pairs: Optional[int] = None,
    pair_sampling: str = "head",
    rng: Optional[torch.Generator] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Penalize per-lag autocorrelations against a Gaussian prior.

    For each lag `L`, compute Pearson correlations between slice pairs
    `(i, i+L)` across spatial pixels (flattened), optionally limited to the first
    `max_pairs` pairs, then return:

        mean_L mean_pairs [ ((corr - mean_L) / sigma_L)^2 ] averaged over L.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")

    lag_tensor = lag_channels
    if not torch.is_tensor(lag_tensor):
        lag_tensor = torch.as_tensor(lag_tensor, device=cube.device)
    lag_tensor = lag_tensor.to(device=cube.device)
    if lag_tensor.ndim == 0:
        lag_tensor = lag_tensor.view(1)
    lag_tensor = lag_tensor.to(dtype=torch.int64).reshape(-1)

    if max_pairs is not None:
        if not isinstance(max_pairs, int):
            raise TypeError("max_pairs must be an int or None.")
        if max_pairs <= 0:
            max_pairs = None
    pair_sampling_norm = str(pair_sampling).strip().lower()
    if pair_sampling_norm not in {"head", "random"}:
        raise ValueError("pair_sampling must be 'head' or 'random'.")

    mean_tensor = prior_mean if torch.is_tensor(prior_mean) else torch.as_tensor(prior_mean, device=cube.device)
    sigma_tensor = prior_sigma if torch.is_tensor(prior_sigma) else torch.as_tensor(prior_sigma, device=cube.device)
    mean_tensor = mean_tensor.to(device=cube.device, dtype=cube.dtype)
    sigma_tensor = sigma_tensor.to(device=cube.device, dtype=cube.dtype)
    lag_weight_tensor: Optional[Tensor] = None
    if lag_weights is not None:
        lag_weight_tensor = (
            lag_weights if torch.is_tensor(lag_weights) else torch.as_tensor(lag_weights, device=cube.device)
        )
        lag_weight_tensor = lag_weight_tensor.to(device=cube.device, dtype=cube.dtype).reshape(-1)
    if mean_tensor.ndim == 0:
        mean_tensor = mean_tensor.view(1)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.view(1)
    mean_tensor = mean_tensor.reshape(-1)
    sigma_tensor = sigma_tensor.reshape(-1)
    if mean_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr prior_mean must be a scalar or match lag_channels length.")
    if sigma_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr prior_sigma must be a scalar or match lag_channels length.")
    if lag_weight_tensor is not None and lag_weight_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr lag_weights must be a scalar or match lag_channels length.")

    moved = cube.movedim(freq_axis, 0)
    num_freqs = moved.shape[0]
    flat = moved.reshape(num_freqs, -1)
    means = flat.mean(dim=1, keepdim=True)
    centered = flat - means
    norms = torch.norm(centered, dim=1)
    norms = torch.clamp(norms, min=eps)

    losses = []
    lag_w = []
    for idx, lag in enumerate(lag_tensor.tolist()):
        if lag < 1:
            raise ValueError(f"lag_channels entries must be >= 1, got {lag}.")
        if lag >= num_freqs:
            raise ValueError(f"lag_channels entry {lag} exceeds num_freqs={num_freqs}.")

        num_total = num_freqs - lag
        num_pairs = min(num_total, max_pairs) if max_pairs is not None else num_total
        if num_pairs <= 0:
            continue

        if max_pairs is None or num_pairs == num_total or pair_sampling_norm == "head":
            pair_idx = torch.arange(num_pairs, device=cube.device)
        else:
            # Sample uniformly without replacement to avoid fixed early-frequency bias.
            pair_idx = torch.randperm(num_total, device="cpu", generator=rng)[:num_pairs]
            pair_idx = pair_idx.to(device=cube.device, dtype=torch.int64)
        pair_j = pair_idx + lag

        a = centered.index_select(0, pair_idx)
        b = centered.index_select(0, pair_j)
        dot = torch.sum(a * b, dim=1)
        denom = norms.index_select(0, pair_idx) * norms.index_select(0, pair_j)
        denom = torch.clamp(denom, min=eps)
        corr = dot / denom

        mean_k = mean_tensor[idx] if mean_tensor.numel() > 1 else mean_tensor[0]
        sigma_k = sigma_tensor[idx] if sigma_tensor.numel() > 1 else sigma_tensor[0]
        sigma_k = clamp_eps(sigma_k, eps=EPS_LOSS)
        losses.append(torch.mean(((corr - mean_k) / sigma_k) ** 2))
        if lag_weight_tensor is not None:
            w_k = lag_weight_tensor[idx] if lag_weight_tensor.numel() > 1 else lag_weight_tensor[0]
            lag_w.append(torch.clamp(w_k, min=0.0))

    if not losses:
        return torch.zeros((), device=cube.device, dtype=cube.dtype)
    losses_t = torch.stack(losses)
    if lag_w:
        lag_w_t = torch.stack(lag_w)
        w_sum = torch.sum(lag_w_t)
        if bool(torch.isfinite(w_sum)) and float(w_sum.item()) > 0.0:
            return torch.sum(losses_t * lag_w_t) / w_sum
    return torch.mean(losses_t)


def _frequency_lag_correlation_stats(
    cube: Tensor,
    *,
    freq_axis: int,
    lag_channels: Tensor,
    prior_mean: Tensor,
    prior_sigma: Tensor,
    lag_weights: Optional[Tensor] = None,
    max_pairs: Optional[int] = None,
    pair_sampling: str = "head",
    rng: Optional[torch.Generator] = None,
    rms_min: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return lagcorr prior loss plus per-lag correlation mean/variance.

    The first return value matches `_frequency_lag_correlation_loss`:
    mean over configured lags of normalized squared residuals.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")

    lag_tensor = lag_channels
    if not torch.is_tensor(lag_tensor):
        lag_tensor = torch.as_tensor(lag_tensor, device=cube.device)
    lag_tensor = lag_tensor.to(device=cube.device)
    if lag_tensor.ndim == 0:
        lag_tensor = lag_tensor.view(1)
    lag_tensor = lag_tensor.to(dtype=torch.int64).reshape(-1)

    if max_pairs is not None:
        if not isinstance(max_pairs, int):
            raise TypeError("max_pairs must be an int or None.")
        if max_pairs <= 0:
            max_pairs = None
    pair_sampling_norm = str(pair_sampling).strip().lower()
    if pair_sampling_norm not in {"head", "random"}:
        raise ValueError("pair_sampling must be 'head' or 'random'.")

    mean_tensor = prior_mean if torch.is_tensor(prior_mean) else torch.as_tensor(prior_mean, device=cube.device)
    sigma_tensor = prior_sigma if torch.is_tensor(prior_sigma) else torch.as_tensor(prior_sigma, device=cube.device)
    mean_tensor = mean_tensor.to(device=cube.device, dtype=cube.dtype)
    sigma_tensor = sigma_tensor.to(device=cube.device, dtype=cube.dtype)
    lag_weight_tensor: Optional[Tensor] = None
    if lag_weights is not None:
        lag_weight_tensor = (
            lag_weights if torch.is_tensor(lag_weights) else torch.as_tensor(lag_weights, device=cube.device)
        )
        lag_weight_tensor = lag_weight_tensor.to(device=cube.device, dtype=cube.dtype).reshape(-1)
    if mean_tensor.ndim == 0:
        mean_tensor = mean_tensor.view(1)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.view(1)
    mean_tensor = mean_tensor.reshape(-1)
    sigma_tensor = sigma_tensor.reshape(-1)
    if mean_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr prior_mean must be a scalar or match lag_channels length.")
    if sigma_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr prior_sigma must be a scalar or match lag_channels length.")
    if lag_weight_tensor is not None and lag_weight_tensor.numel() not in (1, lag_tensor.numel()):
        raise ValueError("lagcorr lag_weights must be a scalar or match lag_channels length.")

    moved = cube.movedim(freq_axis, 0)
    num_freqs = moved.shape[0]
    flat = moved.reshape(num_freqs, -1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = torch.norm(centered, dim=1)
    norms = torch.clamp(norms, min=eps)
    # Optional variance guard: ignore slice-pairs where either slice has too-small RMS,
    # which can make corr statistics extremely noisy/unstable.
    rms_min_val = float(rms_min)
    if not math.isfinite(rms_min_val) or rms_min_val < 0.0:
        raise ValueError("rms_min must be a finite non-negative value.")
    if rms_min_val > 0.0:
        pix = float(centered.shape[1])
        scale = math.sqrt(max(1.0, pix))
        slice_rms = norms / torch.as_tensor(scale, device=norms.device, dtype=norms.dtype)
        valid_slice = slice_rms >= rms_min_val
    else:
        valid_slice = torch.ones_like(norms, dtype=torch.bool)

    losses = []
    lag_means = []
    lag_vars = []
    lag_w = []
    for idx, lag in enumerate(lag_tensor.tolist()):
        if lag < 1:
            raise ValueError(f"lag_channels entries must be >= 1, got {lag}.")
        if lag >= num_freqs:
            raise ValueError(f"lag_channels entry {lag} exceeds num_freqs={num_freqs}.")

        num_total = num_freqs - lag
        num_pairs = min(num_total, max_pairs) if max_pairs is not None else num_total
        if num_pairs <= 0:
            continue

        if max_pairs is None or num_pairs == num_total or pair_sampling_norm == "head":
            pair_idx = torch.arange(num_pairs, device=cube.device)
        else:
            pair_idx = torch.randperm(num_total, device="cpu", generator=rng)[:num_pairs]
            pair_idx = pair_idx.to(device=cube.device, dtype=torch.int64)
        pair_j = pair_idx + lag

        a = centered.index_select(0, pair_idx)
        b = centered.index_select(0, pair_j)
        dot = torch.sum(a * b, dim=1)
        denom = norms.index_select(0, pair_idx) * norms.index_select(0, pair_j)
        denom = torch.clamp(denom, min=eps)
        corr = dot / denom
        pair_valid = valid_slice.index_select(0, pair_idx) & valid_slice.index_select(0, pair_j)
        if bool(torch.sum(pair_valid.to(dtype=torch.int64)).item()) == 0:
            # No stable pairs: do not apply a penalty for this lag.
            zero = torch.zeros((), device=cube.device, dtype=cube.dtype)
            losses.append(zero)
            lag_means.append(zero)
            lag_vars.append(zero)
            if lag_weight_tensor is not None:
                w_k = lag_weight_tensor[idx] if lag_weight_tensor.numel() > 1 else lag_weight_tensor[0]
                lag_w.append(torch.clamp(w_k, min=0.0))
            continue
        corr = corr[pair_valid]

        mean_k = mean_tensor[idx] if mean_tensor.numel() > 1 else mean_tensor[0]
        sigma_k = sigma_tensor[idx] if sigma_tensor.numel() > 1 else sigma_tensor[0]
        sigma_k = clamp_eps(sigma_k, eps=EPS_LOSS)
        losses.append(torch.mean(((corr - mean_k) / sigma_k) ** 2))
        lag_means.append(torch.mean(corr))
        lag_vars.append(torch.var(corr, unbiased=False))
        if lag_weight_tensor is not None:
            w_k = lag_weight_tensor[idx] if lag_weight_tensor.numel() > 1 else lag_weight_tensor[0]
            lag_w.append(torch.clamp(w_k, min=0.0))

    if not losses:
        zero = torch.zeros((), device=cube.device, dtype=cube.dtype)
        empty = torch.empty((0,), device=cube.device, dtype=cube.dtype)
        return zero, empty, empty
    losses_t = torch.stack(losses)
    if lag_w:
        lag_w_t = torch.stack(lag_w)
        w_sum = torch.sum(lag_w_t)
        if bool(torch.isfinite(w_sum)) and float(w_sum.item()) > 0.0:
            loss_val = torch.sum(losses_t * lag_w_t) / w_sum
        else:
            loss_val = torch.mean(losses_t)
    else:
        loss_val = torch.mean(losses_t)
    return loss_val, torch.stack(lag_means), torch.stack(lag_vars)


def _maybe_avg_pool_spatial(cube: Tensor, *, freq_axis: int, pool: int) -> Tensor:
    """
    Optionally downsample spatial dimensions using avg-pooling (per frequency slice).

    This is used to make correlation-style losses cheaper and less noisy. Pooling is
    applied only on the two spatial axes (the axes other than freq_axis).
    """
    if pool is None:
        return cube
    pool_int = int(pool)
    if pool_int <= 1:
        return cube
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube for spatial pooling, got shape {tuple(cube.shape)}")
    # Move frequency to dim0 => (F, X, Y), then pool as (F, 1, X, Y).
    moved = cube.movedim(freq_axis, 0)
    if moved.shape[1] < pool_int or moved.shape[2] < pool_int:
        raise ValueError(
            f"lagcorr_spatial_pool={pool_int} is larger than spatial dims {tuple(moved.shape[1:])}."
        )
    pooled = F.avg_pool2d(moved.unsqueeze(1), kernel_size=pool_int, stride=pool_int).squeeze(1)
    return pooled.movedim(0, freq_axis)


def _frequency_lag_correlation_profile(
    cube: Tensor,
    *,
    freq_axis: int,
    lag_channels: Tensor,
    max_pairs: Optional[int] = None,
    pair_sampling: str = "head",
    rng: Optional[torch.Generator] = None,
    rms_min: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    Compute per-lag mean/variance of autocorrelation across frequency.

    For each lag `L`, compute Pearson correlations between slice pairs `(i, i+L)`
    across spatial pixels (flattened), optionally limited to `max_pairs`.

    Returns:
      - lag_means: Tensor of shape (K,) matching lag_channels ordering
      - lag_vars:  Tensor of shape (K,) matching lag_channels ordering
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")

    lag_tensor = lag_channels
    if not torch.is_tensor(lag_tensor):
        lag_tensor = torch.as_tensor(lag_tensor, device=cube.device)
    lag_tensor = lag_tensor.to(device=cube.device)
    if lag_tensor.ndim == 0:
        lag_tensor = lag_tensor.view(1)
    lag_tensor = lag_tensor.to(dtype=torch.int64).reshape(-1)

    if max_pairs is not None:
        if not isinstance(max_pairs, int):
            raise TypeError("max_pairs must be an int or None.")
        if max_pairs <= 0:
            max_pairs = None
    pair_sampling_norm = str(pair_sampling).strip().lower()
    if pair_sampling_norm not in {"head", "random"}:
        raise ValueError("pair_sampling must be 'head' or 'random'.")

    moved = cube.movedim(freq_axis, 0)
    num_freqs = moved.shape[0]
    flat = moved.reshape(num_freqs, -1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = torch.norm(centered, dim=1)
    norms = torch.clamp(norms, min=eps)
    rms_min_val = float(rms_min)
    if not math.isfinite(rms_min_val) or rms_min_val < 0.0:
        raise ValueError("rms_min must be a finite non-negative value.")
    if rms_min_val > 0.0:
        pix = float(centered.shape[1])
        scale = math.sqrt(max(1.0, pix))
        slice_rms = norms / torch.as_tensor(scale, device=norms.device, dtype=norms.dtype)
        valid_slice = slice_rms >= rms_min_val
    else:
        valid_slice = torch.ones_like(norms, dtype=torch.bool)

    lag_means: List[Tensor] = []
    lag_vars: List[Tensor] = []
    for lag in lag_tensor.tolist():
        if lag < 1:
            raise ValueError(f"lag_channels entries must be >= 1, got {lag}.")
        if lag >= num_freqs:
            raise ValueError(f"lag_channels entry {lag} exceeds num_freqs={num_freqs}.")

        num_total = num_freqs - lag
        num_pairs = min(num_total, max_pairs) if max_pairs is not None else num_total
        if num_pairs <= 0:
            raise ValueError("No valid lagcorr slice pairs available for correlation profile.")

        if max_pairs is None or num_pairs == num_total or pair_sampling_norm == "head":
            pair_idx = torch.arange(num_pairs, device=cube.device)
        else:
            pair_idx = torch.randperm(num_total, device="cpu", generator=rng)[:num_pairs]
            pair_idx = pair_idx.to(device=cube.device, dtype=torch.int64)
        pair_j = pair_idx + lag

        a = centered.index_select(0, pair_idx)
        b = centered.index_select(0, pair_j)
        dot = torch.sum(a * b, dim=1)
        denom = norms.index_select(0, pair_idx) * norms.index_select(0, pair_j)
        denom = torch.clamp(denom, min=eps)
        corr = dot / denom
        pair_valid = valid_slice.index_select(0, pair_idx) & valid_slice.index_select(0, pair_j)
        if bool(torch.sum(pair_valid.to(dtype=torch.int64)).item()) == 0:
            lag_means.append(torch.zeros((), device=cube.device, dtype=cube.dtype))
            lag_vars.append(torch.zeros((), device=cube.device, dtype=cube.dtype))
            continue
        corr = corr[pair_valid]

        lag_means.append(torch.mean(corr))
        lag_vars.append(torch.var(corr, unbiased=False))

    return torch.stack(lag_means), torch.stack(lag_vars)


def _masked_weighted_mean(values: Tensor, weights: Optional[Tensor], mask: Tensor) -> Tensor:
    device = values.device
    dtype = values.dtype
    if mask.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)
    mask = mask.to(device=device)
    if bool(torch.sum(mask.to(dtype=torch.int64)).item()) == 0:
        return torch.zeros((), device=device, dtype=dtype)
    v = values[mask]
    if weights is None:
        return torch.mean(v)
    w = weights.to(device=device, dtype=dtype)[mask]
    w = torch.clamp(w, min=0.0)
    w_sum = torch.sum(w)
    if bool(torch.isfinite(w_sum)) and float(w_sum.item()) > 0.0:
        return torch.sum(v * w) / w_sum
    return torch.mean(v)


def eor_lagcorr_envelope_loss(
    *,
    lag_channels: Tensor,
    lag_means: Tensor,
    lag_weights: Optional[Tensor] = None,
    tail_weights: Optional[Tensor] = None,
    near_max_lag: Optional[int] = None,
    mid_max_lag: Optional[int] = None,
    far_min_lag: Optional[int] = None,
    tail_eps: float = 0.05,
    neg_delta: float = 0.0,
    near_rho_min: float = 0.0,
    near_floor_mode: str = "absolute_mean",
    near_rho1_coeffs: Optional[Tensor] = None,
    rebound_eps_act: float = 0.05,
    rebound_delta_up: float = 0.02,
    w_tail: float = 1.0,
    w_neg: float = 1.0,
    w_near: float = 1.0,
    w_rebound: float = 1.0,
) -> Tensor:
    """
    Weak, physically-motivated envelope priors for EoR lag autocorrelation profile.

    This loss depends only on the estimated EoR cube statistics (no truth/templates).

    Terms (all hinge-style):
      - tail:     penalize large-|rho| at far lags (decorrelation at large separation)
      - neg:      penalize clearly negative rho at near lags
      - near:     enforce a weak positive floor on near-lag correlation
      - rebound:  discourage large increases in |rho(l)| with increasing lag (ringing)
    """
    if lag_channels.ndim != 1 or lag_means.ndim != 1:
        raise ValueError("lag_channels and lag_means must be 1D tensors.")
    if int(lag_channels.numel()) != int(lag_means.numel()):
        raise ValueError("lag_channels and lag_means must have the same length.")

    tail_eps_val = float(tail_eps)
    neg_delta_val = float(neg_delta)
    near_rho_min_val = float(near_rho_min)
    rebound_eps_act_val = float(rebound_eps_act)
    rebound_delta_up_val = float(rebound_delta_up)
    for name, v in [
        ("tail_eps", tail_eps_val),
        ("neg_delta", neg_delta_val),
        ("near_rho_min", near_rho_min_val),
        ("rebound_eps_act", rebound_eps_act_val),
        ("rebound_delta_up", rebound_delta_up_val),
    ]:
        if not math.isfinite(v):
            raise ValueError(f"lagcorr envelope param {name} must be finite.")
    if tail_eps_val < 0.0:
        raise ValueError("tail_eps must be >= 0.")
    if rebound_eps_act_val < 0.0:
        raise ValueError("rebound_eps_act must be >= 0.")
    if rebound_delta_up_val < 0.0:
        raise ValueError("rebound_delta_up must be >= 0.")

    weights = lag_weights
    if weights is not None:
        weights = weights.reshape(-1)
        if int(weights.numel()) == 1:
            weights = weights.repeat(int(lag_channels.numel()))
        if int(weights.numel()) != int(lag_channels.numel()):
            raise ValueError("lag_weights must be scalar or match lag_channels length.")

    lags = lag_channels.to(dtype=torch.int64).reshape(-1)
    order = torch.argsort(lags)
    lags = lags.index_select(0, order)
    rho = lag_means.reshape(-1).index_select(0, order)
    w_sorted = weights.index_select(0, order) if weights is not None else None
    tail_w_sorted: Optional[Tensor] = None
    if tail_weights is not None:
        tail_w = tail_weights.to(device=rho.device, dtype=rho.dtype).reshape(-1)
        if int(tail_w.numel()) == 1:
            tail_w = tail_w.repeat(int(lag_channels.numel()))
        if int(tail_w.numel()) != int(lag_channels.numel()):
            raise ValueError("tail_weights must be scalar or match lag_channels length.")
        if torch.any(~torch.isfinite(tail_w)):
            raise ValueError("tail_weights must be finite.")
        if torch.any(tail_w < 0.0):
            raise ValueError("tail_weights must be non-negative.")
        tail_w_sorted = tail_w.index_select(0, order)

    max_lag = int(torch.max(lags).item()) if lags.numel() > 0 else 1
    near_max = int(near_max_lag) if near_max_lag is not None else min(10, max_lag)
    mid_max = int(mid_max_lag) if mid_max_lag is not None else min(30, max_lag)
    far_min = int(far_min_lag) if far_min_lag is not None else max_lag

    if near_max < 1 or mid_max < 1 or far_min < 1:
        raise ValueError("near_max_lag/mid_max_lag/far_min_lag must be >= 1 when provided.")

    abs_rho = torch.abs(rho)

    # (1) Far-lag envelope towards 0: allow small sign-changing fluctuations around 0.
    far_resid = torch.clamp(abs_rho - tail_eps_val, min=0.0) ** 2
    if tail_w_sorted is None:
        far_mask = lags >= far_min
        loss_tail = _masked_weighted_mean(far_resid, w_sorted, far_mask)
    else:
        far_mask = tail_w_sorted > 0.0
        eff_w = tail_w_sorted if w_sorted is None else (w_sorted * tail_w_sorted)
        loss_tail = _masked_weighted_mean(far_resid, eff_w, far_mask)

    # (2) Near-lag negative suppression (weak): don't force strict non-negativity everywhere.
    near_mask = lags <= near_max
    neg_resid = torch.clamp(-rho - neg_delta_val, min=0.0) ** 2
    loss_neg = _masked_weighted_mean(neg_resid, w_sorted, near_mask)

    # (3) Near-lag floor: prevent "instant decorrelation" at small separation.
    loss_near = torch.zeros_like(loss_tail)
    floor_mode = str(near_floor_mode).strip().lower()
    if floor_mode in {"absolute_mean", "absolute", "mean"}:
        if near_rho_min_val > 0.0:
            rho_near = _masked_weighted_mean(rho, w_sorted, near_mask)
            loss_near = (
                torch.clamp(
                    torch.as_tensor(near_rho_min_val, device=rho.device, dtype=rho.dtype) - rho_near,
                    min=0.0,
                )
                ** 2
            )
    elif floor_mode in {"relative_rho1", "rho1"}:
        if near_rho1_coeffs is None:
            raise ValueError("near_rho1_coeffs must be provided when near_floor_mode='relative_rho1'.")
        coeffs = near_rho1_coeffs.to(device=rho.device, dtype=rho.dtype).reshape(-1)
        if int(coeffs.numel()) == 1:
            coeffs = coeffs.repeat(int(lag_channels.numel()))
        if int(coeffs.numel()) != int(lag_channels.numel()):
            raise ValueError("near_rho1_coeffs must be scalar or match lag_channels length.")
        coeffs = coeffs.index_select(0, order)
        if torch.any(~torch.isfinite(coeffs)):
            raise ValueError("near_rho1_coeffs must be finite.")
        if torch.any(coeffs < 0.0):
            raise ValueError("near_rho1_coeffs must be non-negative.")
        rho1_mask = lags == 1
        if bool(torch.sum(rho1_mask.to(dtype=torch.int64)).item()) == 0:
            raise ValueError("near_floor_mode='relative_rho1' requires lag_channels to include lag=1.")
        rho1 = rho[rho1_mask][0]
        use_mask = near_mask & (lags != 1) & (coeffs > 0.0)
        resid = torch.clamp(coeffs * rho1 - rho, min=0.0) ** 2
        loss_near = _masked_weighted_mean(resid, w_sorted, use_mask)
    else:
        raise ValueError("near_floor_mode must be one of: absolute_mean, relative_rho1.")

    # (4) Rebound suppression on |rho| up to mid_max: penalize large increases with lag.
    loss_rebound = torch.zeros_like(loss_tail)
    if rho.numel() >= 2:
        rebound_mask = (lags[1:] <= mid_max) & (abs_rho[:-1] > rebound_eps_act_val)
        rebound_resid = torch.clamp(abs_rho[1:] - abs_rho[:-1] - rebound_delta_up_val, min=0.0) ** 2
        w_pairs = w_sorted[:-1] if w_sorted is not None else None
        loss_rebound = _masked_weighted_mean(rebound_resid, w_pairs, rebound_mask)

    # Weighted combination (weights are scalar multipliers inside the EoR lagcorr component).
    total = (
        float(w_tail) * loss_tail
        + float(w_neg) * loss_neg
        + float(w_near) * loss_near
        + float(w_rebound) * loss_rebound
    )
    return total


def _resolve_lag_vector_prior(
    value: Optional[Union[Tensor, Sequence[float], float]],
    *,
    lag_count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    default: Optional[float] = None,
) -> Optional[Tensor]:
    if value is None:
        if default is None:
            return None
        return torch.full((lag_count,), float(default), device=device, dtype=dtype)
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.repeat(lag_count)
    if tensor.numel() != lag_count:
        raise ValueError(
            f"{name} must be a scalar or match lagcorr_lags length ({tensor.numel()} vs {lag_count})."
        )
    return tensor


def _apply_lagcorr_feature(cube: Tensor, *, freq_axis: int, feature: str) -> Tensor:
    feature_norm = str(feature).strip().lower()
    if feature_norm == "raw":
        return cube
    if feature_norm == "diff1":
        if cube.shape[freq_axis] < 2:
            raise ValueError("lagcorr_feature='diff1' requires at least 2 frequency channels.")
        return torch.diff(cube, n=1, dim=freq_axis)
    raise ValueError("lagcorr_feature must be 'raw' or 'diff1'.")


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
    if percent == 0.0:
        return torch.zeros_like(tensor.sum(dim=freq_axis))
    # percent means the fraction of highest-frequency bins to penalize.
    num_high = int(math.ceil(percent * num_bins))
    if num_high <= 0:
        return torch.zeros_like(tensor.sum(dim=freq_axis))
    start_idx = max(num_bins - num_high, 0)
    freq_slice = [slice(None)] * rfft.ndim
    freq_slice[freq_axis] = slice(start_idx, None)
    highfreq = rfft[tuple(freq_slice)]
    power = highfreq.real**2 + highfreq.imag**2
    energy = power.mean(dim=freq_axis)
    return energy


def compute_highfreq_energy_ratio(
    tensor: Tensor,
    *,
    freq_axis: int,
    percent: float,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute per-pixel ratio: high-frequency energy / total energy along the frequency axis.
    """
    if not (0.0 <= percent <= 1.0):
        raise ValueError("percent must be in [0, 1].")
    rfft = torch.fft.rfft(tensor, dim=freq_axis)
    num_bins = rfft.shape[freq_axis]
    power = rfft.real**2 + rfft.imag**2
    total = power.mean(dim=freq_axis)
    if num_bins == 0 or percent == 0.0:
        return torch.zeros_like(total)
    num_high = int(math.ceil(percent * num_bins))
    if num_high <= 0:
        return torch.zeros_like(total)
    start_idx = max(num_bins - num_high, 0)
    freq_slice = [slice(None)] * power.ndim
    freq_slice[freq_axis] = slice(start_idx, None)
    high = power[tuple(freq_slice)].mean(dim=freq_axis)
    denom = torch.clamp(total, min=float(eps))
    return high / denom


def derive_fft_prior_from_cube(
    fg_cube: Tensor,
    freq_axis: int,
    percent: float,
    use_robust: bool = False,
    use_log_energy: bool = False,
    mae_to_sigma_factor: float = 1.4826,
) -> Tuple[Tensor, Tensor]:
    energy_map = compute_highfreq_energy(fg_cube, freq_axis=freq_axis, percent=percent)
    if use_log_energy:
        energy_map = torch.log1p(torch.clamp(energy_map, min=0.0))
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
    corr_weight: float = 1.0,
    delta: Optional[float] = None,
    fft_weight: float = 1.0,
    poly_weight: float = 1.0,
    eor_mean_weight: float = 1.0,
    eor_hf_weight: float = 1.0,
    lagcorr_weight: float = 1.0,
    lagcorr_fg_component_weight: float = 0.5,
    lagcorr_eor_component_weight: float = 0.5,
    extra_loss_scale: float = 1.0,
    freq_axis: int = 0,
    data_error: Tensor,
    eor_mean: Tensor,
    eor_sigma: Tensor,
    eor_amp_threshold: Tensor,
    eor_amp_prior_mode: str = "voxel_deadzone",
    eor_hybrid_voxel_factor: float = 5.0,
    eor_hybrid_voxel_weight: float = 0.1,
    fg_smooth_mean: Optional[Tensor],
    fg_smooth_sigma: Optional[Tensor],
    corr_prior_mean: Tensor,
    corr_prior_sigma: Tensor,
    corr_prior_abs_threshold: Optional[Tensor] = None,
    corr_reduce: str = "mean",
    corr_topk: Optional[int] = None,
    corr_lse_alpha: float = 10.0,
    corr_feature: str = "raw",
    corr_spatial_pool: int = 1,
    fg_smooth_mode: str = "diff3_l2",
    fg_smooth_huber_delta: float = 1.0,
    loss_mode: str = "base",
    active_extra_terms: Optional[Union[str, Sequence[str]]] = None,
    lagcorr_unit: str = "mhz",
    lagcorr_feature: str = "raw",
    lagcorr_lags: Optional[Tensor] = None,
    fg_lagcorr_mean: Optional[Tensor] = None,
    fg_lagcorr_sigma: Optional[Tensor] = None,
    eor_lagcorr_mean: Optional[Tensor] = None,
    eor_lagcorr_sigma: Optional[Tensor] = None,
    lagcorr_max_pairs: Optional[int] = None,
    lagcorr_spatial_pool: int = 1,
    lagcorr_rms_min: float = 0.0,
    lagcorr_eor_mode: str = "gaussian",
    lagcorr_eor_near_max_lag: Optional[int] = None,
    lagcorr_eor_mid_max_lag: Optional[int] = None,
    lagcorr_eor_far_min_lag: Optional[int] = None,
    lagcorr_eor_tail_weights: Optional[Tensor] = None,
    lagcorr_eor_tail_eps: float = 0.05,
    lagcorr_eor_neg_delta: float = 0.0,
    lagcorr_eor_near_rho_min: float = 0.0,
    lagcorr_eor_near_floor_mode: str = "absolute_mean",
    lagcorr_eor_near_rho1_coeffs: Optional[Tensor] = None,
    lagcorr_eor_rebound_eps_act: float = 0.05,
    lagcorr_eor_rebound_delta_up: float = 0.02,
    lagcorr_eor_w_tail: float = 1.0,
    lagcorr_eor_w_neg: float = 1.0,
    lagcorr_eor_w_near: float = 1.0,
    lagcorr_eor_w_rebound: float = 1.0,
    fft_prior_mean: Optional[Tensor] = None,
    fft_prior_sigma: Optional[Tensor] = None,
    fft_percent: float = 0.7,
    fft_use_log_energy: bool = False,
    fft_z_clip: Optional[float] = None,
    eor_hf_percent: float = 0.7,
    eor_hf_r_max: float = 0.85,
    poly_degree: int = 3,
    poly_sigma: Optional[Union[Tensor, float]] = None,
    poly_residual: Optional[Tensor] = None,
    lagcorr_pair_sampling: str = "head",
    lagcorr_rng: Optional[torch.Generator] = None,
    lagcorr_prior_weights: Optional[Union[Tensor, Sequence[float], float]] = None,
    lagcorr_eor_scale: float = 1.0,
    lagcorr_gap_weight: float = 0.0,
    lagcorr_gap_margin: Optional[Union[Tensor, Sequence[float], float]] = 0.0,
    lagcorr_gap_sigma: Optional[Union[Tensor, Sequence[float], float]] = 1.0,
    lagcorr_gap_mode: str = "hinge",
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Combined loss used for optimizing foreground and EoR components.
    """
    y_pred = forward_model(fg, eor, psf=psf)
    sigma_data = clamp_eps(data_error, eps=EPS_LOSS)
    data_loss = torch.mean(((y_pred - y) / sigma_data) ** 2)

    smooth_loss = foreground_smoothness_loss(
        fg,
        freq_axis=freq_axis,
        prior_mean=fg_smooth_mean,
        prior_sigma=fg_smooth_sigma,
        mode=fg_smooth_mode,
        huber_delta=fg_smooth_huber_delta,
    )

    eor_mean_tensor = eor_mean
    eor_sigma_tensor = clamp_eps(eor_sigma, eps=EPS_LOSS)

    eor_amp_mode = str(eor_amp_prior_mode).strip().lower()
    if eor_amp_mode in {"voxel_deadzone", "voxel", "deadzone"}:
        eor_amp_threshold_tensor = prepare_broadcastable_prior(
            eor_amp_threshold, eor, "eor_prior_amp_threshold"
        )
        if eor_amp_threshold_tensor is None:
            eor_amp_threshold_tensor = torch.zeros(1, device=eor.device, dtype=eor.dtype)
        if torch.any(~torch.isfinite(eor_amp_threshold_tensor)):
            raise ValueError("eor_prior_amp_threshold must be finite.")
        if torch.any(eor_amp_threshold_tensor < 0.0):
            raise ValueError("eor_prior_amp_threshold must be non-negative.")
        # Dead-zone EoR amplitude prior:
        #   residual = max(|eor - mean| - threshold, 0), then normalized squared penalty.
        eor_residual = torch.clamp(
            torch.abs(eor - eor_mean_tensor) - eor_amp_threshold_tensor, min=0.0
        )
        eor_reg = torch.mean((eor_residual / eor_sigma_tensor) ** 2)
    elif eor_amp_mode in {"slice_rms_hinge", "slice_rms", "rms"}:
        # Slice RMS upper bound (weak, physical): for each frequency slice, penalize when
        # std_xy(eor[f]) exceeds a threshold.
        eor_moved = eor.movedim(freq_axis, 0)
        eor_flat = eor_moved.reshape(eor_moved.shape[0], -1)
        slice_std = eor_flat.std(dim=1, unbiased=False)
        thr = prepare_broadcastable_prior(eor_amp_threshold, slice_std, "eor_prior_amp_threshold")
        if thr is None:
            thr = torch.zeros(1, device=slice_std.device, dtype=slice_std.dtype)
        if torch.any(~torch.isfinite(thr)):
            raise ValueError("eor_prior_amp_threshold must be finite.")
        if torch.any(thr < 0.0):
            raise ValueError("eor_prior_amp_threshold must be non-negative.")
        resid = torch.clamp(slice_std - thr, min=0.0)
        sigma_scale = prepare_broadcastable_prior(eor_sigma_tensor, resid, "eor_prior_sigma")
        if sigma_scale is None:
            sigma_scale = torch.ones(1, device=resid.device, dtype=resid.dtype)
        sigma_scale = clamp_eps(sigma_scale, eps=EPS_LOSS)
        eor_reg = torch.mean((resid / sigma_scale) ** 2)
    elif eor_amp_mode in {"hybrid"}:
        # Hybrid: slice RMS as primary constraint, plus a very loose voxel-wise guard
        # against extreme outliers (useful when RMS is small but rare pixels explode).
        eor_moved = eor.movedim(freq_axis, 0)
        eor_flat = eor_moved.reshape(eor_moved.shape[0], -1)
        slice_std = eor_flat.std(dim=1, unbiased=False)
        thr = prepare_broadcastable_prior(eor_amp_threshold, slice_std, "eor_prior_amp_threshold")
        if thr is None:
            thr = torch.zeros(1, device=slice_std.device, dtype=slice_std.dtype)
        if torch.any(~torch.isfinite(thr)):
            raise ValueError("eor_prior_amp_threshold must be finite.")
        if torch.any(thr < 0.0):
            raise ValueError("eor_prior_amp_threshold must be non-negative.")
        resid_rms = torch.clamp(slice_std - thr, min=0.0)
        sigma_scale = prepare_broadcastable_prior(eor_sigma_tensor, resid_rms, "eor_prior_sigma")
        if sigma_scale is None:
            sigma_scale = torch.ones(1, device=resid_rms.device, dtype=resid_rms.dtype)
        sigma_scale = clamp_eps(sigma_scale, eps=EPS_LOSS)
        rms_term = torch.mean((resid_rms / sigma_scale) ** 2)

        voxel_factor = float(eor_hybrid_voxel_factor)
        voxel_weight = float(eor_hybrid_voxel_weight)
        if not math.isfinite(voxel_factor) or voxel_factor <= 0.0:
            raise ValueError("eor_hybrid_voxel_factor must be a finite positive value.")
        if not math.isfinite(voxel_weight) or voxel_weight < 0.0:
            raise ValueError("eor_hybrid_voxel_weight must be a finite non-negative value.")
        guard_thr = voxel_factor * prepare_broadcastable_prior(thr, eor, "eor_prior_amp_threshold_guard")
        if guard_thr is None:
            guard_thr = voxel_factor * torch.zeros(1, device=eor.device, dtype=eor.dtype)
        guard_resid = torch.clamp(torch.abs(eor - eor_mean_tensor) - guard_thr, min=0.0)
        guard_term = torch.mean((guard_resid / eor_sigma_tensor) ** 2)
        eor_reg = rms_term + voxel_weight * guard_term
    else:
        raise ValueError(
            "eor_amp_prior_mode must be one of: voxel_deadzone, slice_rms_hinge, hybrid."
        )

    if delta is not None:
        corr_weight = float(delta)
    active_terms = normalize_extra_loss_terms(
        loss_mode=loss_mode, extra_loss_terms=active_extra_terms
    )
    active_term_set = set(active_terms)

    corr_feature_norm = str(corr_feature).strip().lower()
    if corr_feature_norm not in {"raw", "diff1", "diff2"}:
        raise ValueError("corr_feature must be one of: raw, diff1, diff2.")
    fg_corr = fg
    eor_corr = eor
    if corr_feature_norm == "diff1":
        if fg.shape[freq_axis] < 2:
            raise ValueError("corr_feature='diff1' requires at least 2 frequency channels.")
        fg_corr = torch.diff(fg_corr, n=1, dim=freq_axis)
        eor_corr = torch.diff(eor_corr, n=1, dim=freq_axis)
    elif corr_feature_norm == "diff2":
        if fg.shape[freq_axis] < 3:
            raise ValueError("corr_feature='diff2' requires at least 3 frequency channels.")
        fg_corr = torch.diff(fg_corr, n=2, dim=freq_axis)
        eor_corr = torch.diff(eor_corr, n=2, dim=freq_axis)

    pool_int = int(corr_spatial_pool)
    if pool_int < 1:
        raise ValueError("corr_spatial_pool must be >= 1.")
    if pool_int > 1:
        fg_corr = _maybe_avg_pool_spatial(fg_corr, freq_axis=freq_axis, pool=pool_int)
        eor_corr = _maybe_avg_pool_spatial(eor_corr, freq_axis=freq_axis, pool=pool_int)

    corr_loss, corr_coeff = frequency_slice_correlation_penalty(
        fg_corr,
        eor_corr,
        freq_axis=freq_axis,
        prior_mean=corr_prior_mean,
        prior_sigma=corr_prior_sigma,
        prior_abs_threshold=corr_prior_abs_threshold,
        reduce=corr_reduce,
        topk=corr_topk,
        lse_alpha=corr_lse_alpha,
    )

    extra_scale = float(extra_loss_scale)
    if not math.isfinite(extra_scale):
        raise ValueError("extra_loss_scale must be finite.")
    extra_scale = max(0.0, min(1.0, extra_scale))

    lagcorr_loss = torch.zeros_like(data_loss)
    lagcorr_gap_loss = torch.zeros_like(data_loss)
    if "lagcorr" in active_term_set and extra_scale > 0.0:
        unit_norm = lagcorr_unit.strip().lower()
        if unit_norm not in {"mhz", "chan"}:
            raise ValueError("lagcorr_unit must be 'mhz' or 'chan'.")
        if lagcorr_lags is None:
            raise ValueError("lagcorr_lags must be provided when enabling 'lagcorr'.")
        fg_component = float(lagcorr_fg_component_weight)
        eor_component = float(lagcorr_eor_component_weight)
        if not math.isfinite(fg_component) or fg_component < 0:
            raise ValueError("lagcorr_fg_component_weight must be a finite non-negative value.")
        if not math.isfinite(eor_component) or eor_component < 0:
            raise ValueError("lagcorr_eor_component_weight must be a finite non-negative value.")
        eor_component_scale = float(lagcorr_eor_scale)
        if not math.isfinite(eor_component_scale):
            raise ValueError("lagcorr_eor_scale must be finite.")
        eor_component_scale = max(0.0, min(1.0, eor_component_scale))
        effective_eor_component = eor_component * eor_component_scale
        total_component = fg_component + effective_eor_component
        if fg_component <= 0.0 and eor_component <= 0.0:
            raise ValueError(
                "At least one lagcorr component weight must be > 0 "
                "(lagcorr_fg_component_weight or lagcorr_eor_component_weight)."
            )
        gap_weight = float(lagcorr_gap_weight)
        if not math.isfinite(gap_weight) or gap_weight < 0.0:
            raise ValueError("lagcorr_gap_weight must be a finite non-negative value.")
        effective_gap_weight = gap_weight * eor_component_scale

        if total_component <= 0.0:
            # Allow scheduling: when lagcorr_eor_scale=0 and lagcorr_fg_component_weight=0,
            # treat lagcorr as temporarily inactive (no penalty) rather than erroring.
            pass
        else:
            lag_weight_vec = _resolve_lag_vector_prior(
                lagcorr_prior_weights,
                lag_count=int(lagcorr_lags.numel()),
                device=fg.device,
                dtype=fg.dtype,
                name="lagcorr_prior_weights",
                default=1.0,
            )
            assert lag_weight_vec is not None
            if torch.any(~torch.isfinite(lag_weight_vec)):
                raise ValueError("lagcorr_prior_weights must be finite.")
            if torch.any(lag_weight_vec < 0.0):
                raise ValueError("lagcorr_prior_weights must be non-negative.")
            if float(torch.sum(lag_weight_vec).item()) <= 0.0:
                raise ValueError("lagcorr_prior_weights must contain at least one positive entry.")

            fg_for_lag = _apply_lagcorr_feature(fg, freq_axis=freq_axis, feature=lagcorr_feature)
            eor_for_lag = _apply_lagcorr_feature(eor, freq_axis=freq_axis, feature=lagcorr_feature)
            pool_int = int(lagcorr_spatial_pool)
            if pool_int < 1:
                raise ValueError("lagcorr_spatial_pool must be >= 1.")
            if pool_int > 1:
                fg_for_lag = _maybe_avg_pool_spatial(
                    fg_for_lag, freq_axis=freq_axis, pool=pool_int
                )
                eor_for_lag = _maybe_avg_pool_spatial(
                    eor_for_lag, freq_axis=freq_axis, pool=pool_int
                )

            fg_loss = torch.zeros_like(data_loss)
            fg_corr_lag_mean: Optional[Tensor] = None
            if fg_component > 0.0:
                if fg_lagcorr_mean is None or fg_lagcorr_sigma is None:
                    raise ValueError(
                        "fg_lagcorr_mean/fg_lagcorr_sigma must be provided when "
                        "lagcorr_fg_component_weight > 0."
                    )
                fg_loss, fg_corr_lag_mean, _ = _frequency_lag_correlation_stats(
                    fg_for_lag,
                    freq_axis=freq_axis,
                    lag_channels=lagcorr_lags,
                    prior_mean=fg_lagcorr_mean,
                    prior_sigma=fg_lagcorr_sigma,
                    lag_weights=lag_weight_vec,
                    max_pairs=lagcorr_max_pairs,
                    pair_sampling=lagcorr_pair_sampling,
                    rng=lagcorr_rng,
                    rms_min=float(lagcorr_rms_min),
                )

            eor_loss = torch.zeros_like(data_loss)
            eor_corr_lag_mean: Optional[Tensor] = None
            if eor_component > 0.0 and (effective_eor_component > 0.0 or effective_gap_weight > 0.0):
                mode_norm = str(lagcorr_eor_mode).strip().lower()
                if mode_norm in {"gaussian", "normal"}:
                    if eor_lagcorr_mean is None or eor_lagcorr_sigma is None:
                        raise ValueError(
                            "eor_lagcorr_mean/eor_lagcorr_sigma must be provided when "
                            "lagcorr_eor_component_weight > 0 and lagcorr_eor_mode='gaussian'."
                        )
                    eor_loss, eor_corr_lag_mean, _ = _frequency_lag_correlation_stats(
                        eor_for_lag,
                        freq_axis=freq_axis,
                        lag_channels=lagcorr_lags,
                        prior_mean=eor_lagcorr_mean,
                        prior_sigma=eor_lagcorr_sigma,
                        lag_weights=lag_weight_vec,
                        max_pairs=lagcorr_max_pairs,
                        pair_sampling=lagcorr_pair_sampling,
                        rng=lagcorr_rng,
                        rms_min=float(lagcorr_rms_min),
                    )
                elif mode_norm in {"envelope", "envelope_v2"}:
                    lag_means, _ = _frequency_lag_correlation_profile(
                        eor_for_lag,
                        freq_axis=freq_axis,
                        lag_channels=lagcorr_lags,
                        max_pairs=lagcorr_max_pairs,
                        pair_sampling=lagcorr_pair_sampling,
                        rng=lagcorr_rng,
                        rms_min=float(lagcorr_rms_min),
                    )
                    eor_corr_lag_mean = lag_means
                    eor_loss = eor_lagcorr_envelope_loss(
                        lag_channels=lagcorr_lags,
                        lag_means=lag_means,
                        lag_weights=lag_weight_vec,
                        tail_weights=lagcorr_eor_tail_weights,
                        near_max_lag=lagcorr_eor_near_max_lag,
                        mid_max_lag=lagcorr_eor_mid_max_lag,
                        far_min_lag=lagcorr_eor_far_min_lag,
                        tail_eps=lagcorr_eor_tail_eps,
                        neg_delta=lagcorr_eor_neg_delta,
                        near_rho_min=lagcorr_eor_near_rho_min,
                        near_floor_mode=lagcorr_eor_near_floor_mode,
                        near_rho1_coeffs=lagcorr_eor_near_rho1_coeffs,
                        rebound_eps_act=lagcorr_eor_rebound_eps_act,
                        rebound_delta_up=lagcorr_eor_rebound_delta_up,
                        w_tail=lagcorr_eor_w_tail,
                        w_neg=lagcorr_eor_w_neg,
                        w_near=lagcorr_eor_w_near,
                        w_rebound=lagcorr_eor_w_rebound,
                    )
                else:
                    raise ValueError(
                        "lagcorr_eor_mode must be one of: gaussian, envelope_v2 (alias: envelope)."
                    )

            lagcorr_loss = (fg_component * fg_loss + effective_eor_component * eor_loss) / total_component

            gap_mode = str(lagcorr_gap_mode).strip().lower()
            if gap_mode not in {"hinge", "squared"}:
                raise ValueError("lagcorr_gap_mode must be 'hinge' or 'squared'.")
            if effective_gap_weight > 0.0:
                if fg_corr_lag_mean is None or eor_corr_lag_mean is None:
                    raise ValueError(
                        "lagcorr_gap_weight > 0 requires both FG and EoR lagcorr components enabled."
                    )
                lag_count = int(fg_corr_lag_mean.numel())
                margin_vec = _resolve_lag_vector_prior(
                    lagcorr_gap_margin,
                    lag_count=lag_count,
                    device=fg_corr_lag_mean.device,
                    dtype=fg_corr_lag_mean.dtype,
                    name="lagcorr_gap_margin",
                    default=0.0,
                )
                sigma_vec = _resolve_lag_vector_prior(
                    lagcorr_gap_sigma,
                    lag_count=lag_count,
                    device=fg_corr_lag_mean.device,
                    dtype=fg_corr_lag_mean.dtype,
                    name="lagcorr_gap_sigma",
                    default=1.0,
                )
                assert margin_vec is not None and sigma_vec is not None
                sigma_vec = clamp_eps(sigma_vec, eps=EPS_LOSS)
                lag_gap = fg_corr_lag_mean - eor_corr_lag_mean
                if gap_mode == "hinge":
                    residual = torch.clamp(margin_vec - lag_gap, min=0.0)
                else:
                    residual = lag_gap - margin_vec
                lagcorr_gap_loss = torch.mean((residual / sigma_vec) ** 2)
                lagcorr_loss = lagcorr_loss + effective_gap_weight * lagcorr_gap_loss

    fft_loss = torch.zeros_like(data_loss)
    if "rfft" in active_term_set and extra_scale > 0.0:
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
        if fft_use_log_energy:
            energy_map = torch.log1p(torch.clamp(energy_map, min=0.0))
        z = (energy_map - prior_mean) / prior_sigma
        if fft_z_clip is not None:
            clip = float(fft_z_clip)
            if not math.isfinite(clip) or clip <= 0:
                raise ValueError("fft_z_clip must be a finite positive value when provided.")
            z = torch.clamp(z, min=-clip, max=clip)
        fft_loss = torch.mean(z**2)
    eor_mean_loss = torch.zeros_like(data_loss)
    if "eor_mean" in active_term_set and extra_scale > 0.0:
        eor_moved = eor.movedim(freq_axis, 0)
        slice_mean = eor_moved.mean(dim=(1, 2))
        eor_mean_loss = torch.mean(slice_mean**2)
    eor_hf_loss = torch.zeros_like(data_loss)
    if "eor_hf" in active_term_set and extra_scale > 0.0:
        r_max = float(eor_hf_r_max)
        if not math.isfinite(r_max) or r_max < 0.0:
            raise ValueError("eor_hf_r_max must be a finite non-negative value.")
        ratio = compute_highfreq_energy_ratio(
            eor, freq_axis=freq_axis, percent=float(eor_hf_percent), eps=1e-12
        )
        resid = torch.clamp(ratio - r_max, min=0.0) ** 2
        eor_hf_loss = torch.mean(resid)
    poly_loss = torch.zeros_like(data_loss)
    if "poly_reparam" in active_term_set and extra_scale > 0.0:
        if poly_residual is None:
            poly_residual = torch.zeros_like(fg)
        sigma_tensor = prepare_broadcastable_prior(poly_sigma, poly_residual, "poly_sigma_reparam")
        if sigma_tensor is None:
            sigma_tensor = torch.ones(1, device=fg.device, dtype=fg.dtype)
        sigma_tensor = clamp_eps(sigma_tensor, eps=EPS_LOSS)
        poly_loss = torch.mean((poly_residual / sigma_tensor) ** 2)

    corr_term = corr_weight * corr_loss if "corr" in active_term_set else torch.zeros_like(data_loss)
    extra_terms = (
        corr_term
        + lagcorr_weight * lagcorr_loss
        + fft_weight * fft_loss
        + poly_weight * poly_loss
        + float(eor_mean_weight) * eor_mean_loss
        + float(eor_hf_weight) * eor_hf_loss
    )
    total_loss = alpha * data_loss + beta * smooth_loss + gamma * eor_reg + extra_scale * extra_terms
    return total_loss, {
        "data": data_loss,
        "smooth": smooth_loss,
        "eor_reg": eor_reg,
        "corr": corr_loss,
        "corr_coeff": corr_coeff,
        "lagcorr": lagcorr_loss,
        "lagcorr_gap": lagcorr_gap_loss,
        "fft_highfreq": fft_loss,
        "eor_mean": eor_mean_loss,
        "eor_hf": eor_hf_loss,
        "poly": poly_loss,
    }
