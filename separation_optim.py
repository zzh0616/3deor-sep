#!/usr/bin/env python3
"""
Optimization-based prototype for separating smooth foreground and fluctuating
EoR components from a 3D data cube.

Copyright (c) 2025 Zhenghao Zhu
Licensed under the MIT License. See LICENSE file for details.
"""

from __future__ import annotations

import json
import csv
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from constants import (
    DEFAULT_CORR_SIGMA,
    DEFAULT_DATA_ERROR,
    DEFAULT_EOR_SIGMA,
    DEFAULT_FFT_SIGMA,
    DEFAULT_POLY_SIGMA,
)
from losses import (
    compute_highfreq_energy,
    correlation_penalty,
    derive_fft_prior_from_cube,
    forward_model,
    foreground_smoothness_loss,
    loss_function,
    polynomial_prior_loss,
    _frequency_lag_correlation_loss,
)
from powerspec import PowerSpecConfig, compute_power_spectra, save_power_outputs
from utils import ensure_tensor_on, prepare_broadcastable_prior


Tensor = torch.Tensor
ForwardOperator = Callable[[Tensor], Tensor]
Device = Union[torch.device, str]


def derive_smoothness_stats_from_cube(
    fg_cube: Tensor,
    freq_axis: int,
    use_robust: bool = False,
    mae_to_sigma_factor: float = 1.4826,
) -> Tuple[Tensor, Tensor]:
    """
    Compute mean/std of third-order finite differences from a reference foreground cube.
    """
    if fg_cube.shape[freq_axis] < 4:
        raise ValueError("Foreground reference cube must have at least 4 frequency channels.")
    ref_diff = torch.diff(fg_cube, n=3, dim=freq_axis)
    if use_robust:
        median = ref_diff.median(dim=freq_axis, keepdim=True).values
        mae = (ref_diff - median).abs().mean(dim=freq_axis, keepdim=True)
        sigma = mae * mae_to_sigma_factor
        sigma = torch.clamp(sigma, min=1e-6)
        return median, sigma
    mean = ref_diff.mean(dim=freq_axis, keepdim=True)
    std = ref_diff.std(dim=freq_axis, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def _summarize_fg_stats(mean_tensor: Tensor, sigma_tensor: Tensor, freq_axis: int) -> None:
    mean_flat = mean_tensor.movedim(freq_axis, 0).reshape(mean_tensor.shape[freq_axis], -1)
    sigma_flat = sigma_tensor.movedim(freq_axis, 0).reshape(sigma_tensor.shape[freq_axis], -1)
    mean_vec = mean_flat.mean(dim=1)
    sigma_vec = sigma_flat.mean(dim=1)
    print("Foreground smoothness stats (per frequency channel after differencing):")
    print(f"  mean (first 5):  {mean_vec[:5].tolist()}")
    print(f"  sigma (first 5): {sigma_vec[:5].tolist()}")


def _warn_if_weight_not_one(name: str, value: float) -> None:
    if not math.isclose(float(value), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        print(f"Warning: {name}={value} (default 1.0). Ensure this is intentional.")


def _warn_weight_defaults(config: OptimizationConfig) -> None:
    _warn_if_weight_not_one("alpha", config.alpha)
    _warn_if_weight_not_one("beta", config.beta)
    _warn_if_weight_not_one("gamma", config.gamma)
    _warn_if_weight_not_one("corr_weight", config.corr_weight)
    _warn_if_weight_not_one("lagcorr_weight", config.lagcorr_weight)
    _warn_if_weight_not_one("fft_weight", config.fft_weight)
    _warn_if_weight_not_one("poly_weight", config.poly_weight)


@dataclass
class LossComponents:
    total: float
    data: float
    smooth: float
    eor_reg: float
    corr: float
    corr_coeff: float
    lagcorr: float
    fft_highfreq: float
    poly: float


@dataclass
class OptimizationConfig:
    num_iters: int = 400
    lr: float = 5e-2
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    fft_weight: float = 1.0
    extra_loss_start_iter: int = 500
    extra_loss_ramp_iters: int = 0
    poly_weight: float = 1.0
    poly_degree: int = 3
    poly_sigma: float = DEFAULT_POLY_SIGMA
    loss_mode: str = "base"  # "base", "rfft", "poly", "poly_reparam", or "lagcorr"
    optimizer_name: str = "adam"
    momentum: float = 0.9
    power_config: Optional[str] = None
    power_output_dir: Optional[str] = None
    freq_axis: int = 0
    cut_xy_enabled: bool = False
    cut_xy_unit: str = "frac"
    cut_xy_center_x: Optional[float] = None
    cut_xy_center_y: Optional[float] = None
    cut_xy_size: Optional[float] = None
    print_every: int = 50
    device: Optional[str] = None
    dtype: Optional[str] = None
    true_eor_cube: Optional[str] = None
    diagnose_input: bool = False
    report_true_signal_corr: bool = False  # deprecated alias for diagnose_input
    corr_plot: Optional[str] = None
    init_fg_cube: Optional[str] = None
    init_eor_cube: Optional[str] = None
    mask_cube: Optional[str] = None
    data_error: float = DEFAULT_DATA_ERROR
    eor_prior_mean: float = 0.0
    eor_prior_sigma: float = DEFAULT_EOR_SIGMA
    fg_smooth_mean: float = 0.0
    fg_smooth_sigma: float = 0.05
    fg_reference_cube: Optional[str] = None
    use_robust_fg_stats: bool = False
    mae_to_sigma_factor: float = 1.4826
    corr_prior_mean: float = 0.0
    corr_prior_sigma: float = DEFAULT_CORR_SIGMA
    corr_weight: float = 1.0
    lagcorr_weight: float = 1.0
    lagcorr_unit: str = "mhz"
    lagcorr_intervals: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]
    )
    fg_lagcorr_mean: List[float] = field(default_factory=lambda: [0.0] * 9)
    fg_lagcorr_sigma: List[float] = field(default_factory=lambda: [1.0] * 9)
    eor_lagcorr_mean: List[float] = field(default_factory=lambda: [0.0] * 9)
    eor_lagcorr_sigma: List[float] = field(default_factory=lambda: [1.0] * 9)
    lagcorr_max_pairs: Optional[int] = None
    fft_highfreq_percent: float = 0.7
    fft_prior_mean: float = 0.0
    fft_prior_sigma: float = DEFAULT_FFT_SIGMA
    enable_corr_check: bool = False
    corr_check_every: int = 500
    freq_start_mhz: Optional[float] = None
    freq_delta_mhz: Optional[float] = None
    freqs_mhz_path: Optional[str] = None
    provided_fields: Set[str] = field(default_factory=set, init=False, repr=False)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for config_field in fields(self):
            if config_field.name in data and data[config_field.name] is not None:
                setattr(self, config_field.name, data[config_field.name])
                if config_field.name != "provided_fields":
                    self.provided_fields.add(config_field.name)

    def resolved_device(self) -> torch.device:
        return torch.device(self.device) if self.device is not None else _default_device()

    def resolved_dtype(self) -> Optional[torch.dtype]:
        return _resolve_dtype_name(self.dtype)


def _default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _resolve_dtype_name(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    lookup = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'. Supported: {sorted(lookup)}")
    return lookup[key]




def _smooth_initial_foreground(y: Tensor, freq_axis: int = 0) -> Tensor:
    """
    Lightweight smoothing along frequency for foreground initialization.
    """
    moved = y.movedim(freq_axis, 0)
    smoothed = moved.clone()
    if moved.shape[0] >= 3:
        smoothed[1:-1] = 0.25 * moved[:-2] + 0.5 * moved[1:-1] + 0.25 * moved[2:]
    if freq_axis != 0:
        smoothed = smoothed.movedim(0, freq_axis)
    return smoothed


def initialize_components(y: Tensor, freq_axis: int = 0) -> Tuple[Tensor, Tensor]:
    """
    Generate initial guesses for fg and eor optimization variables.
    """
    fg_init = _smooth_initial_foreground(y, freq_axis=freq_axis)
    eor_init = torch.zeros_like(y)
    return fg_init, eor_init


def _polynomial_design(num_freqs: int, degree: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    freqs = torch.linspace(0.0, 1.0, num_freqs, device=device, dtype=dtype)
    return torch.stack([freqs**i for i in range(degree + 1)], dim=1)  # (F, degree+1)


def _fit_polynomial_coeffs(
    y: Tensor,
    freq_axis: int,
    degree: int,
    freqs: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    y_front = y.movedim(freq_axis, 0)
    num_freqs = y_front.shape[0]
    if freqs is None:
        design = _polynomial_design(num_freqs, degree, device=y.device, dtype=y.dtype)
    else:
        if freqs.numel() != num_freqs:
            raise ValueError("Frequency array length does not match cube frequency dimension.")
        freqs = freqs.to(device=y.device, dtype=y.dtype)
        f_min = freqs.min()
        f_max = freqs.max()
        scale = torch.clamp(f_max - f_min, min=1e-8)
        coords = (freqs - f_min) / scale
        design = torch.stack([coords**i for i in range(degree + 1)], dim=1)
    y_flat = y_front.reshape(num_freqs, -1)
    safe_dtype = torch.float32 if y.dtype not in (torch.float32, torch.float64) else y.dtype
    y_flat_cast = y_flat.to(dtype=safe_dtype)
    design_cast = design.to(dtype=safe_dtype)
    coeffs = torch.linalg.lstsq(design_cast, y_flat_cast).solution  # (degree+1, Npix)
    coeffs = coeffs.to(dtype=y.dtype)
    coeffs = coeffs.reshape(degree + 1, *y_front.shape[1:])
    fitted = (design_cast.to(dtype=y.dtype) @ coeffs.reshape(degree + 1, -1)).reshape_as(y_front)
    fitted = fitted.movedim(0, freq_axis)
    return coeffs, fitted


def _eval_polynomial_from_coeffs(
    coeffs: Tensor,
    freq_axis: int,
    target_shape: Sequence[int],
    freqs: Optional[Tensor] = None,
) -> Tensor:
    """
    Evaluate a polynomial foreground model from coefficients.

    If `freqs` is provided, it is interpreted as the physical frequency grid
    (1D tensor of length F) and used to build the design matrix; otherwise a
    normalized [0, 1] grid is used.
    """
    num_freqs = target_shape[freq_axis]
    if freqs is None:
        design = _polynomial_design(
            num_freqs, coeffs.shape[0] - 1, device=coeffs.device, dtype=coeffs.dtype
        )
    else:
        if freqs.numel() != num_freqs:
            raise ValueError("Frequency array length does not match cube frequency dimension.")
        freqs = freqs.to(device=coeffs.device, dtype=coeffs.dtype)
        f_min = freqs.min()
        f_max = freqs.max()
        scale = torch.clamp(f_max - f_min, min=1e-8)
        coords = (freqs - f_min) / scale
        design = torch.stack([coords**i for i in range(coeffs.shape[0])], dim=1)
    poly = torch.tensordot(design, coeffs, dims=([1], [0]))  # (F, spatial...)
    poly = poly.movedim(0, freq_axis)
    return poly


def _prepare_observation(
    y: Tensor,
    *,
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, torch.device, torch.dtype]:
    """
    Convert input data to a tensor on the requested device/dtype without forcing CPU copies.
    """
    target_device = torch.device(device) if device is not None else None
    target_dtype = dtype

    if torch.is_tensor(y):
        if target_device is None:
            target_device = y.device
        if target_dtype is None:
            target_dtype = y.dtype
        tensor = y.to(device=target_device, dtype=target_dtype)
    else:
        if target_device is None:
            target_device = _default_device()
        if target_dtype is None:
            target_dtype = torch.get_default_dtype()
        tensor = torch.as_tensor(y, dtype=target_dtype).to(device=target_device)

    return tensor, target_device, target_dtype


def optimize_components(
    y: Tensor,
    *,
    psf: Optional[ForwardOperator] = None,
    num_iters: int = 500,
    lr: float = 1e-2,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    freq_axis: int = 0,
    print_every: int = 50,
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = None,
    fg_init_tensor: Optional[Tensor] = None,
    eor_init_tensor: Optional[Tensor] = None,
    data_error: Union[Tensor, float] = DEFAULT_DATA_ERROR,
    eor_prior_mean: Union[Tensor, float] = 0.0,
    eor_prior_sigma: Union[Tensor, float] = DEFAULT_EOR_SIGMA,
    fg_smooth_mean: Optional[Union[Tensor, float]] = 0.0,
    fg_smooth_sigma: Optional[Union[Tensor, float]] = 0.05,
    corr_prior_mean: Union[Tensor, float] = 0.0,
    corr_prior_sigma: Union[Tensor, float] = DEFAULT_CORR_SIGMA,
    corr_weight: float = 1.0,
    lagcorr_weight: float = 1.0,
    lagcorr_unit: str = "mhz",
    lagcorr_intervals: Optional[Sequence[float]] = None,
    fg_lagcorr_mean: Optional[Sequence[float]] = None,
    fg_lagcorr_sigma: Optional[Sequence[float]] = None,
    eor_lagcorr_mean: Optional[Sequence[float]] = None,
    eor_lagcorr_sigma: Optional[Sequence[float]] = None,
    lagcorr_max_pairs: Optional[int] = None,
    extra_loss_start_iter: int = 500,
    extra_loss_ramp_iters: int = 0,
    fft_weight: float = 1.0,
    poly_weight: float = 1.0,
    poly_degree: int = 3,
    poly_sigma: Optional[Union[Tensor, float]] = DEFAULT_POLY_SIGMA,
    loss_mode: str = "base",
    fft_prior_mean: Optional[Union[Tensor, float]] = None,
    fft_prior_sigma: Optional[Union[Tensor, float]] = None,
    fft_highfreq_percent: float = 0.7,
    freq_start_mhz: Optional[float] = None,
    freq_delta_mhz: Optional[float] = None,
    optimizer_name: str = "adam",
    momentum: float = 0.9,
    eor_true_tensor: Optional[Tensor] = None,
    corr_check_every: int = 0,
) -> Tuple[Tensor, Tensor, List[LossComponents]]:
    """
    Optimize fg and eor to fit the observed cube y.

    Args:
        y: Observed cube (numpy array or tensor). If already on GPU, it stays there.
        device: Optional device override (e.g., "cuda:0").
        dtype: Optional dtype override.
        fg_init_tensor: Optional initial foreground tensor/array (same shape as y).
        eor_init_tensor: Optional initial EoR tensor/array (same shape as y).
        data_error: Measurement error (scalar or tensor broadcastable to y).
        eor_prior_mean: Prior mean for the EoR cube.
        eor_prior_sigma: Prior standard deviation for the EoR cube.
        fg_smooth_mean: Prior mean for the third-order FG differences.
        fg_smooth_sigma: Prior std for the third-order FG differences.
        corr_prior_mean: Prior mean for FG/EoR correlation coefficient.
        corr_prior_sigma: Prior std for FG/EoR correlation coefficient.
        corr_weight: Weight for the correlation consistency loss.
        lagcorr_weight: Weight for the frequency-lag autocorrelation loss (lagcorr mode).
        extra_loss_start_iter: Iteration at which extra loss terms activate (non-base modes).
        extra_loss_ramp_iters: If >0, ramp extra loss scale from 0 to 1 over this many iters after start.
        fft_weight: Weight for the high-frequency penalty (rFFT mode).
        poly_weight: Weight for the polynomial prior term.
        poly_degree: Degree for polynomial priors.
        poly_sigma: Std for polynomial prior residuals.
        loss_mode: "base" (default), "rfft", "poly", "poly_reparam", or "lagcorr".
        fft_prior_mean: Prior mean for high-frequency energy (scalar or tensor).
        fft_prior_sigma: Prior std for high-frequency energy (scalar or tensor).
        fft_highfreq_percent: Fraction (0-1) of the highest frequency bins to penalize.
        freq_start_mhz: Starting frequency of the cube (MHz) for polynomial modes.
        freq_delta_mhz: Frequency spacing of the cube (MHz) for polynomial modes.
    """
    if loss_mode not in {"base", "rfft", "poly", "poly_reparam", "lagcorr"}:
        raise ValueError("loss_mode must be 'base', 'rfft', 'poly', 'poly_reparam', or 'lagcorr'.")
    y_tensor, _, _ = _prepare_observation(y, device=device, dtype=dtype)

    aligned_eor_true: Optional[Tensor] = None
    if eor_true_tensor is not None:
        tensor = eor_true_tensor if torch.is_tensor(eor_true_tensor) else torch.as_tensor(eor_true_tensor)
        if tensor.shape != y_tensor.shape:
            raise ValueError(
                f"eor_true_tensor shape {tuple(tensor.shape)} does not match observation shape {tuple(y_tensor.shape)}"
            )
        aligned_eor_true = tensor.to(device=y_tensor.device, dtype=y_tensor.dtype)

    data_error_tensor = prepare_broadcastable_prior(data_error, y_tensor, "data_error")
    if data_error_tensor is None:
        data_error_tensor = torch.as_tensor(
            DEFAULT_DATA_ERROR, device=y_tensor.device, dtype=y_tensor.dtype
        )

    eor_mean_tensor = prepare_broadcastable_prior(eor_prior_mean, y_tensor, "eor_prior_mean")
    if eor_mean_tensor is None:
        eor_mean_tensor = torch.zeros(1, device=y_tensor.device, dtype=y_tensor.dtype)

    eor_sigma_tensor = prepare_broadcastable_prior(eor_prior_sigma, y_tensor, "eor_prior_sigma")
    if eor_sigma_tensor is None:
        eor_sigma_tensor = torch.full(
            (1,), DEFAULT_EOR_SIGMA, device=y_tensor.device, dtype=y_tensor.dtype
        )

    fg_mean_tensor = ensure_tensor_on(fg_smooth_mean, y_tensor.device, y_tensor.dtype)
    if fg_mean_tensor is None:
        fg_mean_tensor = torch.zeros(1, device=y_tensor.device, dtype=y_tensor.dtype)

    fg_sigma_tensor = ensure_tensor_on(fg_smooth_sigma, y_tensor.device, y_tensor.dtype)
    if fg_sigma_tensor is None:
        fg_sigma_tensor = torch.ones(1, device=y_tensor.device, dtype=y_tensor.dtype)

    corr_mean_tensor = ensure_tensor_on(corr_prior_mean, y_tensor.device, y_tensor.dtype)
    if corr_mean_tensor is None:
        corr_mean_tensor = torch.zeros(1, device=y_tensor.device, dtype=y_tensor.dtype)

    corr_sigma_tensor = ensure_tensor_on(corr_prior_sigma, y_tensor.device, y_tensor.dtype)
    if corr_sigma_tensor is None:
        corr_sigma_tensor = torch.full(
            (1,), DEFAULT_CORR_SIGMA, device=y_tensor.device, dtype=y_tensor.dtype
        )

    lagcorr_lags_tensor: Optional[Tensor] = None
    fg_lagcorr_mean_tensor: Optional[Tensor] = None
    fg_lagcorr_sigma_tensor: Optional[Tensor] = None
    eor_lagcorr_mean_tensor: Optional[Tensor] = None
    eor_lagcorr_sigma_tensor: Optional[Tensor] = None
    if loss_mode == "lagcorr":
        unit_norm = lagcorr_unit.strip().lower()
        if unit_norm not in {"mhz", "chan"}:
            raise ValueError("lagcorr_unit must be 'mhz' or 'chan'.")
        if lagcorr_intervals is None:
            raise ValueError("lagcorr_intervals must be provided when loss_mode='lagcorr'.")
        interval_tensor = ensure_tensor_on(lagcorr_intervals, y_tensor.device, y_tensor.dtype)
        if interval_tensor is None:
            raise ValueError("lagcorr_intervals must be provided when loss_mode='lagcorr'.")
        interval_tensor = interval_tensor.reshape(-1)
        if interval_tensor.numel() == 0:
            raise ValueError("lagcorr_intervals must contain at least one interval.")

        if unit_norm == "chan":
            rounded = interval_tensor.round()
            if not torch.allclose(interval_tensor, rounded, rtol=0.0, atol=1e-6):
                raise ValueError("lagcorr_intervals must be integers when lagcorr_unit='chan'.")
            lagcorr_lags_tensor = rounded.to(dtype=torch.int64)
            if torch.any(lagcorr_lags_tensor < 1):
                raise ValueError("lagcorr_intervals entries must be >= 1 when lagcorr_unit='chan'.")
        else:
            if freq_delta_mhz is None or freq_delta_mhz <= 0:
                raise ValueError(
                    "freq_delta_mhz must be set (>0) when using loss_mode='lagcorr' with lagcorr_unit='mhz'."
                )
            lagcorr_lags_tensor = torch.round(interval_tensor / float(freq_delta_mhz)).to(dtype=torch.int64)
            lagcorr_lags_tensor = torch.clamp(lagcorr_lags_tensor, min=1)

        num_freqs = int(y_tensor.shape[freq_axis])
        if torch.any(lagcorr_lags_tensor >= num_freqs):
            raise ValueError(
                f"lagcorr intervals produce lag_channels >= num_freqs ({num_freqs}); "
                "reduce lagcorr_intervals or use more frequency channels."
            )

        def _require_vec(name: str, values: Optional[Sequence[float]]) -> Tensor:
            tensor = ensure_tensor_on(values, y_tensor.device, y_tensor.dtype)
            if tensor is None:
                raise ValueError(f"{name} must be provided when loss_mode='lagcorr'.")
            tensor = tensor.reshape(-1)
            if tensor.numel() != interval_tensor.numel():
                raise ValueError(
                    f"{name} must have the same length as lagcorr_intervals "
                    f"({tensor.numel()} vs {interval_tensor.numel()})."
                )
            return tensor

        fg_lagcorr_mean_tensor = _require_vec("fg_lagcorr_mean", fg_lagcorr_mean)
        fg_lagcorr_sigma_tensor = _require_vec("fg_lagcorr_sigma", fg_lagcorr_sigma)
        eor_lagcorr_mean_tensor = _require_vec("eor_lagcorr_mean", eor_lagcorr_mean)
        eor_lagcorr_sigma_tensor = _require_vec("eor_lagcorr_sigma", eor_lagcorr_sigma)

        if lagcorr_max_pairs is not None and lagcorr_max_pairs <= 0:
            raise ValueError("lagcorr_max_pairs must be a positive int or None.")

    fft_mean_tensor = ensure_tensor_on(fft_prior_mean, y_tensor.device, y_tensor.dtype)
    fft_sigma_tensor = ensure_tensor_on(fft_prior_sigma, y_tensor.device, y_tensor.dtype)
    poly_sigma_tensor = ensure_tensor_on(poly_sigma, y_tensor.device, y_tensor.dtype)

    freqs_tensor: Optional[Tensor] = None
    if freq_start_mhz is not None and freq_delta_mhz is not None:
        nfreq = y_tensor.shape[freq_axis]
        freqs_tensor = (
            freq_start_mhz
            + freq_delta_mhz * torch.arange(nfreq, device=y_tensor.device, dtype=y_tensor.dtype)
        )

    if loss_mode == "poly":
        print(
            "Warning: loss_mode=poly runs a full polynomial fit over the foreground; "
            "this can be slow on large cubes and is recommended only for testing or small inputs."
        )

    fg_init, eor_init = initialize_components(y_tensor, freq_axis=freq_axis)

    # Poly reparam initialization
    poly_coeffs_param: Optional[torch.nn.Parameter] = None
    fg_resid_param: Optional[torch.nn.Parameter] = None
    fg_param: torch.nn.Parameter

    def _prepare_init_guess(init_value: Optional[Tensor], name: str) -> Optional[Tensor]:
        if init_value is None:
            return None
        tensor = init_value if torch.is_tensor(init_value) else torch.as_tensor(init_value)
        if tensor.shape != y_tensor.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match observation shape {tuple(y_tensor.shape)}"
            )
        return tensor.to(device=y_tensor.device, dtype=y_tensor.dtype)

    fg_override = _prepare_init_guess(fg_init_tensor, "fg_init")
    eor_override = _prepare_init_guess(eor_init_tensor, "eor_init")

    if fg_override is not None:
        fg_init = fg_override
    if eor_override is not None:
        eor_init = eor_override

    if loss_mode == "poly_reparam":
        coeffs_init, fitted = _fit_polynomial_coeffs(
            fg_init,
            freq_axis=freq_axis,
            degree=poly_degree,
            freqs=freqs_tensor,
        )
        resid_init = fg_init - fitted
        poly_coeffs_param = torch.nn.Parameter(coeffs_init.clone())
        fg_resid_param = torch.nn.Parameter(resid_init.clone())
        fg_param = fg_resid_param
    else:
        fg_param = torch.nn.Parameter(fg_init.clone())

    eor = torch.nn.Parameter(eor_init.clone())

    params = [eor]
    if poly_coeffs_param is not None and fg_resid_param is not None:
        params.extend([fg_resid_param, poly_coeffs_param])
    else:
        params.append(fg_param)
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    history: List[LossComponents] = []

    for it in range(1, num_iters + 1):
        optimizer.zero_grad()
        extra_scale = 0.0
        if loss_mode != "base":
            start_iter = max(0, int(extra_loss_start_iter))
            ramp_iters = max(0, int(extra_loss_ramp_iters))
            if it < start_iter:
                extra_scale = 0.0
            else:
                if ramp_iters > 0:
                    extra_scale = min(1.0, float(it - start_iter) / float(ramp_iters))
                else:
                    extra_scale = 1.0
        if loss_mode == "poly_reparam":
            assert poly_coeffs_param is not None and fg_resid_param is not None
            poly_component = _eval_polynomial_from_coeffs(
                poly_coeffs_param, freq_axis, y_tensor.shape, freqs=freqs_tensor
            )
            fg_current = poly_component + fg_resid_param
            poly_residual = fg_resid_param
        else:
            fg_current = fg_param
            poly_residual = None

        total_loss, components = loss_function(
            fg_current,
            eor,
            y_tensor,
            psf=psf,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=corr_weight,
            fft_weight=fft_weight,
            poly_weight=poly_weight,
            lagcorr_weight=lagcorr_weight,
            extra_loss_scale=extra_scale,
            freq_axis=freq_axis,
            data_error=data_error_tensor,
            eor_mean=eor_mean_tensor,
            eor_sigma=eor_sigma_tensor,
            fg_smooth_mean=fg_mean_tensor,
            fg_smooth_sigma=fg_sigma_tensor,
            corr_prior_mean=corr_mean_tensor,
            corr_prior_sigma=corr_sigma_tensor,
            loss_mode=loss_mode,
            lagcorr_unit=lagcorr_unit,
            lagcorr_lags=lagcorr_lags_tensor,
            fg_lagcorr_mean=fg_lagcorr_mean_tensor,
            fg_lagcorr_sigma=fg_lagcorr_sigma_tensor,
            eor_lagcorr_mean=eor_lagcorr_mean_tensor,
            eor_lagcorr_sigma=eor_lagcorr_sigma_tensor,
            lagcorr_max_pairs=lagcorr_max_pairs,
            fft_prior_mean=fft_mean_tensor,
            fft_prior_sigma=fft_sigma_tensor,
            fft_percent=fft_highfreq_percent,
            poly_degree=poly_degree,
            poly_sigma=poly_sigma_tensor,
            poly_residual=poly_residual,
        )
        total_loss.backward()
        optimizer.step()

        if (
            aligned_eor_true is not None
            and corr_check_every > 0
            and (it % corr_check_every == 0 or it == num_iters)
        ):
            with torch.no_grad():
                corr_values = compute_frequency_correlations(
                    eor, aligned_eor_true, freq_axis=freq_axis
                )
            mean_corr = float(np.nanmean(corr_values))
            print(f"[check] iter {it:04d}: mean EoR corr={mean_corr:.4f}")

        entry = LossComponents(
            total=float(total_loss.item()),
            data=float(components["data"].item()),
            smooth=float(components["smooth"].item()),
            eor_reg=float(components["eor_reg"].item()),
            corr=float(components["corr"].item()),
            corr_coeff=float(components["corr_coeff"].item()),
            lagcorr=float(components["lagcorr"].item()),
            fft_highfreq=float(components["fft_highfreq"].item()),
            poly=float(components["poly"].item()),
        )
        history.append(entry)

        if print_every and (it == 1 or it % print_every == 0 or it == num_iters):
            print(
                f"[iter {it:04d}] total={entry.total:.4e} "
                f"data={entry.data:.4e} smooth={entry.smooth:.4e} "
                f"eor={entry.eor_reg:.4e} corr={entry.corr:.4e} lagcorr={entry.lagcorr:.4e} "
                f"corr_coeff={entry.corr_coeff:.3f} fft={entry.fft_highfreq:.4e} "
                f"poly={entry.poly:.4e}"
            )

    if loss_mode == "poly_reparam":
        assert poly_coeffs_param is not None and fg_resid_param is not None
        fg_final = _eval_polynomial_from_coeffs(
            poly_coeffs_param, freq_axis, y_tensor.shape, freqs=freqs_tensor
        ) + fg_resid_param
    else:
        fg_final = fg_param

    return fg_final.detach(), eor.detach(), history


def _create_synthetic_cube(
    num_freqs: int = 32,
    image_shape: Sequence[int] = (32, 32),
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Construct a toy dataset with known fg and eor components.
    """
    device = device or torch.device("cpu")
    freqs = torch.linspace(0.0, 1.0, num_freqs, device=device)

    x = torch.linspace(-1.0, 1.0, image_shape[0], device=device)
    y = torch.linspace(-1.0, 1.0, image_shape[1], device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    spatial_template = torch.exp(-2.0 * (xx**2 + yy**2))

    spectral_poly = 0.4 * freqs**2 - 0.1 * freqs + 0.3
    fg_true = spectral_poly.view(-1, 1, 1) * spatial_template

    random_texture = torch.randn(1, *image_shape, device=device)
    oscillation = torch.sin(2 * torch.pi * freqs * 5.0).view(-1, 1, 1)
    eor_true = 0.05 * oscillation * random_texture + 0.02 * torch.randn(
        num_freqs, *image_shape, device=device
    )

    return fg_true, eor_true


def read_fits_cube(path: Path) -> Tensor:
    """
    Load a 3D cube from a FITS file.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Reading FITS files requires the 'astropy' package.") from exc

    data = fits.getdata(path)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, found shape {data.shape}")
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def read_fits_array(path: Path) -> Tensor:
    """
    Load a 2D image or 3D cube from a FITS file.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Reading FITS files requires the 'astropy' package.") from exc

    data = fits.getdata(path)
    if data.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D/3D array in {path}, found shape {data.shape}")
    return torch.from_numpy(np.asarray(data, dtype=np.float32))


def apply_mask_xy(cube: Tensor, mask: Tensor, freq_axis: int) -> Tensor:
    """
    Apply a mask to a 3D cube.

    Supported mask shapes:
      - 3D: same shape as cube.
      - 2D: spatial mask matching the two non-frequency axes (broadcast across frequency).
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")
    if mask.ndim not in (2, 3):
        raise ValueError(f"Mask must be 2D or 3D, got shape {tuple(mask.shape)}")
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    mask_clean = torch.nan_to_num(mask, nan=0.0)
    mask_clean = mask_clean.to(device=cube.device, dtype=cube.dtype)

    if mask_clean.ndim == 3:
        if tuple(mask_clean.shape) != tuple(cube.shape):
            raise ValueError(
                f"3D mask shape {tuple(mask_clean.shape)} does not match cube shape {tuple(cube.shape)}"
            )
        return cube * mask_clean

    spatial_axes = [ax for ax in range(3) if ax != freq_axis]
    x_axis, y_axis = spatial_axes[0], spatial_axes[1]
    expected = (int(cube.shape[x_axis]), int(cube.shape[y_axis]))
    if tuple(mask_clean.shape) != expected:
        raise ValueError(f"2D mask shape {tuple(mask_clean.shape)} does not match spatial shape {expected}")

    mask_3d = mask_clean.unsqueeze(freq_axis).expand_as(cube)
    return cube * mask_3d


@dataclass(frozen=True)
class CutXYIndices:
    freq_axis: int
    x_axis: int
    y_axis: int
    x0: int
    x1: int
    y0: int
    y1: int
    size_px: int
    center_x_px: int
    center_y_px: int
    nx: int
    ny: int
    unit: str


def _clamp_fixed_window(start: int, size: int, length: int) -> Tuple[int, int]:
    if size > length:
        raise ValueError(f"Requested crop size {size} exceeds axis length {length}.")
    end = start + size
    if start < 0:
        start = 0
        end = size
    if end > length:
        end = length
        start = length - size
    return int(start), int(end)


def build_cut_xy_indices(shape: Sequence[int], freq_axis: int, config: OptimizationConfig) -> Optional[CutXYIndices]:
    if not config.cut_xy_enabled:
        return None
    if len(shape) != 3:
        raise ValueError(f"cut_xy expects a 3D cube, got shape {tuple(shape)}")
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    spatial_axes = [ax for ax in range(3) if ax != freq_axis]
    x_axis, y_axis = spatial_axes[0], spatial_axes[1]
    nx, ny = int(shape[x_axis]), int(shape[y_axis])
    min_dim = min(nx, ny)

    unit = str(config.cut_xy_unit).strip().lower()
    if unit not in {"frac", "px"}:
        raise ValueError("cut_xy_unit must be 'frac' or 'px'.")

    if unit == "frac":
        cx = 0.5 if config.cut_xy_center_x is None else float(config.cut_xy_center_x)
        cy = 0.5 if config.cut_xy_center_y is None else float(config.cut_xy_center_y)
        size = 0.5 if config.cut_xy_size is None else float(config.cut_xy_size)
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError("cut_xy.center_x/center_y must be in [0, 1] when unit='frac'.")
        if size <= 0.0:
            raise ValueError("cut_xy.size must be > 0 when unit='frac'.")
        size_px = int(round(size * float(min_dim)))
        center_x_px = int(round(cx * float(max(nx - 1, 0))))
        center_y_px = int(round(cy * float(max(ny - 1, 0))))
    else:
        cx_val = nx // 2 if config.cut_xy_center_x is None else config.cut_xy_center_x
        cy_val = ny // 2 if config.cut_xy_center_y is None else config.cut_xy_center_y
        size_val = None if config.cut_xy_size is None else config.cut_xy_size
        center_x_px = int(round(float(cx_val)))
        center_y_px = int(round(float(cy_val)))
        if size_val is None:
            size_px = int(round(0.5 * float(min_dim)))
        else:
            size_px = int(round(float(size_val)))
        if size_px <= 0:
            raise ValueError("cut_xy.size must be >= 1 when unit='px'.")

    size_px = max(1, int(size_px))
    if size_px > min_dim:
        raise ValueError(f"cut_xy.size={size_px} exceeds min(Nx,Ny)={min_dim}.")

    start_x = center_x_px - size_px // 2
    start_y = center_y_px - size_px // 2
    x0, x1 = _clamp_fixed_window(start_x, size_px, nx)
    y0, y1 = _clamp_fixed_window(start_y, size_px, ny)

    return CutXYIndices(
        freq_axis=int(freq_axis),
        x_axis=int(x_axis),
        y_axis=int(y_axis),
        x0=int(x0),
        x1=int(x1),
        y0=int(y0),
        y1=int(y1),
        size_px=int(size_px),
        center_x_px=int(center_x_px),
        center_y_px=int(center_y_px),
        nx=int(nx),
        ny=int(ny),
        unit=unit,
    )


def apply_cut_xy(cube: Tensor, indices: CutXYIndices) -> Tensor:
    if cube.ndim != 3:
        raise ValueError(f"cut_xy expects a 3D cube, got shape {tuple(cube.shape)}")
    if int(cube.shape[indices.x_axis]) != indices.nx or int(cube.shape[indices.y_axis]) != indices.ny:
        raise ValueError(
            "cut_xy requires all cubes share the same spatial shape; "
            f"expected (Nx,Ny)=({indices.nx},{indices.ny}) on axes ({indices.x_axis},{indices.y_axis}), "
            f"got ({int(cube.shape[indices.x_axis])},{int(cube.shape[indices.y_axis])})."
        )
    slices: List[slice] = [slice(None)] * cube.ndim
    slices[indices.x_axis] = slice(indices.x0, indices.x1)
    slices[indices.y_axis] = slice(indices.y0, indices.y1)
    return cube[tuple(slices)]


def cut_xy_fits_header(indices: CutXYIndices) -> Dict[str, Tuple[object, str]]:
    return {
        "CUTXY": (True, "XY crop applied"),
        "CUTUNIT": (indices.unit, "cut_xy unit"),
        "CUTFRAX": (indices.freq_axis, "0-based freq axis"),
        "CUTXAX": (indices.x_axis, "0-based x axis"),
        "CUTYAX": (indices.y_axis, "0-based y axis"),
        "CUTX0": (indices.x0, "0-based crop x start"),
        "CUTX1": (indices.x1, "0-based crop x end (exclusive)"),
        "CUTY0": (indices.y0, "0-based crop y start"),
        "CUTY1": (indices.y1, "0-based crop y end (exclusive)"),
        "CUTSIZ": (indices.size_px, "crop size (pixels)"),
        "CUTCTX": (indices.center_x_px, "requested center x (pixels)"),
        "CUTCTY": (indices.center_y_px, "requested center y (pixels)"),
        "CUTNX": (indices.nx, "original Nx"),
        "CUTNY": (indices.ny, "original Ny"),
    }


def write_fits_cube(
    tensor: Tensor,
    path: Path,
    header_extras: Optional[Dict[str, Tuple[object, str]]] = None,
) -> None:
    """
    Save a tensor to a FITS file, moving to CPU if needed.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Writing FITS files requires the 'astropy' package.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().numpy()
    header = None
    if header_extras:
        header = fits.Header()
        for key, value in header_extras.items():
            if isinstance(value, tuple) and len(value) == 2:
                header[key] = value
            else:
                header[key] = value
    fits.writeto(path, array, header=header, overwrite=True)


def load_config_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_frequency_correlations(
    eor_est: Tensor, eor_true: Tensor, freq_axis: int = 0
) -> np.ndarray:
    """
    Compute per-frequency Pearson correlations between estimated and reference EoR cubes.
    """
    if eor_est.shape != eor_true.shape:
        raise ValueError(
            f"EoR estimate shape {tuple(eor_est.shape)} does not match reference {tuple(eor_true.shape)}"
        )

    est = eor_est.detach().cpu().movedim(freq_axis, 0)
    ref = eor_true.detach().cpu().movedim(freq_axis, 0)

    est_flat = est.reshape(est.shape[0], -1).numpy()
    ref_flat = ref.reshape(ref.shape[0], -1).numpy()

    correlations = []
    for idx in range(est_flat.shape[0]):
        est_slice = est_flat[idx]
        ref_slice = ref_flat[idx]
        est_centered = est_slice - est_slice.mean()
        ref_centered = ref_slice - ref_slice.mean()
        denom = np.linalg.norm(est_centered) * np.linalg.norm(ref_centered)
        corr = float(np.dot(est_centered, ref_centered) / denom) if denom > 0 else 0.0
        correlations.append(corr)
    return np.asarray(correlations, dtype=np.float32)


def _unit_scale_to_mhz(unit: str) -> Optional[float]:
    normalized = unit.strip().lower()
    if not normalized:
        return None
    mapping = {
        "hz": 1e-6,
        "hertz": 1e-6,
        "khz": 1e-3,
        "kilohz": 1e-3,
        "kilohertz": 1e-3,
        "mhz": 1.0,
        "megahz": 1.0,
        "megahertz": 1.0,
        "ghz": 1e3,
        "gigahz": 1e3,
        "gigahertz": 1e3,
    }
    return mapping.get(normalized)


def _infer_freqs_mhz_from_fits_header(path: Path, freq_axis: int, num_freqs: int, ndim: int) -> Optional[np.ndarray]:
    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover - dependency check
        return None

    header = fits.getheader(path)
    fits_axis = ndim - freq_axis

    crval = header.get(f"CRVAL{fits_axis}")
    cdelt = header.get(f"CDELT{fits_axis}")
    crpix = header.get(f"CRPIX{fits_axis}", 1.0)
    if crval is None or cdelt is None:
        return None

    cunit = str(header.get(f"CUNIT{fits_axis}", "")).strip()
    scale = _unit_scale_to_mhz(cunit)
    if scale is None:
        if not cunit:
            abs_val = abs(float(crval))
            if abs_val > 1e6:
                scale = 1e-6
            else:
                scale = 1.0
        else:
            return None

    pix = np.arange(1, num_freqs + 1, dtype=np.float64)
    freqs_native = float(crval) + (pix - float(crpix)) * float(cdelt)
    return freqs_native * float(scale)


def _load_freqs_mhz_from_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        freqs = np.load(path)
    else:
        freqs = np.loadtxt(path, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError(f"freqs_mhz_path '{path}' did not contain any values.")
    return freqs


def resolve_frequency_axis_mhz(
    cube_path: Optional[Path],
    num_freqs: int,
    freq_axis: int,
    config: OptimizationConfig,
    cube_ndim: int = 3,
) -> Optional[np.ndarray]:
    """
    Resolve the frequency axis values in MHz for a cube.

    Priority:
      1) config.freqs_mhz_path (text or .npy)
      2) config.freq_start_mhz + config.freq_delta_mhz
      3) FITS header WCS keywords (CRVAL/CDELT/CRPIX/CUNIT)
    """
    if config.freqs_mhz_path:
        freqs_path = Path(config.freqs_mhz_path)
        if not freqs_path.exists():
            raise FileNotFoundError(f"freqs_mhz_path '{freqs_path}' not found.")
        freqs = _load_freqs_mhz_from_file(freqs_path)
        if freqs.shape[0] != num_freqs:
            raise ValueError(
                f"freqs_mhz_path '{freqs_path}' has length {freqs.shape[0]}, expected {num_freqs}."
            )
        return freqs

    if config.freq_start_mhz is not None and config.freq_delta_mhz is not None:
        return config.freq_start_mhz + np.arange(num_freqs, dtype=np.float64) * config.freq_delta_mhz

    if cube_path is not None:
        freqs = _infer_freqs_mhz_from_fits_header(
            cube_path, freq_axis=freq_axis, num_freqs=num_freqs, ndim=cube_ndim
        )
        if freqs is not None and freqs.shape[0] == num_freqs:
            return freqs

    return None


def compute_frequency_lag_correlations(cube: Tensor, lag_channels: int, freq_axis: int = 0) -> np.ndarray:
    """
    Compute Pearson correlations between slices separated by a fixed frequency lag.

    Returns an array of length (num_freqs - lag_channels), where entry i is the
    correlation between cube[i] and cube[i + lag_channels] along the frequency axis.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")

    moved = cube.detach().cpu().movedim(freq_axis, 0)
    num_freqs = moved.shape[0]
    if lag_channels < 1 or lag_channels >= num_freqs:
        raise ValueError(
            f"lag_channels must be in [1, {num_freqs - 1}], got {lag_channels}."
        )

    flat = moved.reshape(num_freqs, -1).numpy()
    num_pixels = flat.shape[1]

    sums = flat.sum(axis=1, dtype=np.float64)
    means = sums / float(num_pixels)
    sumsq = np.sum(flat * flat, axis=1, dtype=np.float64)
    centered_energy = sumsq - float(num_pixels) * means * means
    centered_energy = np.maximum(centered_energy, 0.0)
    norms = np.sqrt(centered_energy)

    dots = np.sum(flat[:-lag_channels] * flat[lag_channels:], axis=1, dtype=np.float64)
    numerators = dots - float(num_pixels) * means[:-lag_channels] * means[lag_channels:]
    denominators = norms[:-lag_channels] * norms[lag_channels:]

    correlations = np.zeros(num_freqs - lag_channels, dtype=np.float64)
    valid = denominators > 0
    correlations[valid] = numerators[valid] / denominators[valid]
    return correlations.astype(np.float32)


def compute_frequency_slice_correlation_matrix(
    cube: Tensor,
    freq_axis: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Compute the full Pearson correlation matrix between all frequency slices.

    The result is a (F, F) tensor where entry (i, j) is the correlation coefficient
    between cube slices at frequency indices i and j.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {tuple(cube.shape)}")

    moved = cube.movedim(freq_axis, 0)
    if device is not None or dtype is not None:
        moved = moved.to(device=device or moved.device, dtype=dtype or moved.dtype)
    flat = moved.reshape(moved.shape[0], -1)

    means = flat.mean(dim=1, keepdim=True)
    centered = flat - means
    norms = torch.norm(centered, dim=1, keepdim=True)
    normalized = torch.where(norms > 0, centered / norms, torch.zeros_like(centered))
    return normalized @ normalized.transpose(0, 1)


def save_true_signal_corr_vs_lag_plot(
    lag_x: np.ndarray,
    fg_mean: np.ndarray,
    fg_std: np.ndarray,
    eor_mean: np.ndarray,
    eor_std: np.ndarray,
    x_label: str,
    path: Path,
) -> None:
    """
    Plot mean correlation vs. frequency lag for FG and EoR, with +/-1σ bands.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Plotting correlations requires the 'matplotlib' package.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lag_x, fg_mean, label="FG", color="tab:blue")
    ax.fill_between(lag_x, fg_mean - fg_std, fg_mean + fg_std, color="tab:blue", alpha=0.2)
    ax.plot(lag_x, eor_mean, label="EoR", color="tab:orange")
    ax.fill_between(lag_x, eor_mean - eor_std, eor_mean + eor_std, color="tab:orange", alpha=0.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Correlation")
    ax.set_title("True-signal autocorrelation vs. frequency lag")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def write_input_diagnostics_report(
    fg_true_cube_path: Path,
    eor_true_cube_path: Path,
    config: OptimizationConfig,
    output_dir: Path,
    filename_prefix: str,
    input_cube_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Diagnose input cubes by writing:
      - FG↔FG and EoR↔EoR frequency-lag autocorrelation summaries + per-pair CSV + plot
      - A loss breakdown (base + all extra loss terms) evaluated on the provided cubes
      - Foreground smoothness statistics (third differences mean/variance)

    The report computes correlations between frequency slices separated by lags
    ranging from 1 channel up to half the total frequency span (floor((F-1)/2)), separately for:
      - foreground ↔ foreground
      - EoR ↔ EoR

    No FG↔EoR cross-correlation is computed.
    """
    fg_true = read_fits_cube(fg_true_cube_path)
    eor_true = read_fits_cube(eor_true_cube_path)

    mask_tensor: Optional[Tensor] = None
    if config.mask_cube:
        mask_path = Path(config.mask_cube)
        if not mask_path.exists():
            raise FileNotFoundError(f"mask_cube '{mask_path}' not found.")
        mask_tensor = read_fits_array(mask_path)
        fg_true = apply_mask_xy(fg_true, mask_tensor, freq_axis=config.freq_axis)
        eor_true = apply_mask_xy(eor_true, mask_tensor, freq_axis=config.freq_axis)

    cut_indices = build_cut_xy_indices(tuple(fg_true.shape), freq_axis=config.freq_axis, config=config)
    if cut_indices is not None:
        fg_true = apply_cut_xy(fg_true, cut_indices)
        eor_true = apply_cut_xy(eor_true, cut_indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{filename_prefix}_input_diagnostics_summary.txt"
    details_path = output_dir / f"{filename_prefix}_input_diagnostics_corr_details.csv"
    plot_path = output_dir / f"{filename_prefix}_input_diagnostics_corr_vs_lag.png"

    device = config.resolved_device()
    dtype = config.resolved_dtype()
    diag_dtype = dtype if dtype is not None else torch.float32

    fg_true = fg_true.to(device=device, dtype=diag_dtype)
    eor_true = eor_true.to(device=device, dtype=diag_dtype)

    y_obs: Optional[Tensor] = None
    if input_cube_path is not None:
        if input_cube_path.exists():
            y_obs = read_fits_cube(input_cube_path)
            if mask_tensor is not None:
                y_obs = apply_mask_xy(y_obs, mask_tensor, freq_axis=config.freq_axis)
            if cut_indices is not None:
                y_obs = apply_cut_xy(y_obs, cut_indices)
            y_obs = y_obs.to(device=device, dtype=diag_dtype)

    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "component",
                "lag_channels",
                "freq_index_i",
                "freq_index_j",
                "freq_i_mhz",
                "freq_j_mhz",
                "delta_mhz",
                "corr_coeff",
            ]
        )

        def _analyze_component(
            name: str, cube: Tensor, cube_path: Path
        ) -> Dict[str, Any]:
            num_freqs = int(cube.shape[config.freq_axis])
            if num_freqs < 2:
                raise ValueError(f"Cube '{cube_path}' must have at least 2 frequency channels.")

            max_lag = max(1, (num_freqs - 1) // 2)
            corr_mat = compute_frequency_slice_correlation_matrix(
                cube, freq_axis=config.freq_axis, device=device, dtype=dtype
            ).detach().cpu().numpy()

            freqs_mhz = resolve_frequency_axis_mhz(
                cube_path,
                num_freqs=num_freqs,
                freq_axis=config.freq_axis,
                config=config,
                cube_ndim=cube.ndim,
            )

            lag_list: List[int] = []
            delta_median_list: List[float] = []
            corr_mean_list: List[float] = []
            corr_var_list: List[float] = []
            corr_min_list: List[float] = []
            corr_max_list: List[float] = []
            pair_count_list: List[int] = []

            for lag in range(1, max_lag + 1):
                corr_vals = np.diagonal(corr_mat, offset=lag).astype(np.float32, copy=False)
                pair_count_list.append(int(corr_vals.shape[0]))

                if freqs_mhz is not None:
                    fi = freqs_mhz[:-lag]
                    fj = freqs_mhz[lag:]
                    delta_vals = np.abs(fj - fi).astype(np.float64, copy=False)
                    positive = delta_vals[np.isfinite(delta_vals) & (delta_vals > 0)]
                    delta_median = float(np.median(positive)) if positive.size else math.nan
                else:
                    fi = None
                    fj = None
                    delta_vals = None
                    delta_median = math.nan

                lag_list.append(lag)
                delta_median_list.append(delta_median)
                corr_mean_list.append(float(np.mean(corr_vals)) if corr_vals.size else math.nan)
                corr_var_list.append(float(np.var(corr_vals)) if corr_vals.size else math.nan)
                corr_min_list.append(float(np.min(corr_vals)) if corr_vals.size else math.nan)
                corr_max_list.append(float(np.max(corr_vals)) if corr_vals.size else math.nan)

                for idx, corr in enumerate(corr_vals.tolist()):
                    i = idx
                    j = idx + lag
                    if fi is not None and fj is not None and delta_vals is not None:
                        freq_i_mhz = float(fi[idx])
                        freq_j_mhz = float(fj[idx])
                        delta_mhz = float(delta_vals[idx])
                    else:
                        freq_i_mhz = math.nan
                        freq_j_mhz = math.nan
                        delta_mhz = math.nan
                    writer.writerow([name, lag, i, j, freq_i_mhz, freq_j_mhz, delta_mhz, float(corr)])

            stats: Dict[str, Any] = {
                "component": name,
                "cube_path": str(cube_path),
                "shape": tuple(int(dim) for dim in cube.shape),
                "freq_axis": int(config.freq_axis),
                "num_freqs": num_freqs,
                "max_lag": int(max_lag),
                "freqs_mhz_available": bool(freqs_mhz is not None),
                "freq_mhz_min": float(np.nanmin(freqs_mhz)) if freqs_mhz is not None else None,
                "freq_mhz_max": float(np.nanmax(freqs_mhz)) if freqs_mhz is not None else None,
                "lag_channels": np.asarray(lag_list, dtype=np.int32),
                "delta_mhz_median": np.asarray(delta_median_list, dtype=np.float64),
                "corr_mean": np.asarray(corr_mean_list, dtype=np.float64),
                "corr_var": np.asarray(corr_var_list, dtype=np.float64),
                "corr_min": np.asarray(corr_min_list, dtype=np.float64),
                "corr_max": np.asarray(corr_max_list, dtype=np.float64),
                "pair_count": np.asarray(pair_count_list, dtype=np.int64),
            }
            return stats

        fg_stats = _analyze_component("foreground", fg_true, fg_true_cube_path)
        eor_stats = _analyze_component("eor", eor_true, eor_true_cube_path)

    def _format_stats(stats: Dict[str, Any]) -> str:
        freq_span = ""
        if stats.get("freqs_mhz_available"):
            fmin = stats.get("freq_mhz_min")
            fmax = stats.get("freq_mhz_max")
            if isinstance(fmin, (int, float)) and isinstance(fmax, (int, float)):
                freq_span = f"  freq_mhz_range: {fmin:.6g} .. {fmax:.6g}\n"
        return (
            f"- component: {stats['component']}\n"
            f"  cube: {stats['cube_path']}\n"
            f"  shape: {stats['shape']}, freq_axis={stats['freq_axis']}\n"
            f"{freq_span}"
            f"  num_freqs: {stats['num_freqs']}\n"
            f"  lags: 1..{stats['max_lag']} (channels)\n"
        )

    def _format_lag_table(stats: Dict[str, Any]) -> str:
        lag_arr: np.ndarray = stats["lag_channels"]
        delta_arr: np.ndarray = stats["delta_mhz_median"]
        mean_arr: np.ndarray = stats["corr_mean"]
        var_arr: np.ndarray = stats["corr_var"]
        count_arr: np.ndarray = stats["pair_count"]

        has_mhz = bool(stats.get("freqs_mhz_available"))
        if has_mhz:
            header = "lag_channels, delta_mhz_median, num_pairs, corr_mean, corr_var\n"
        else:
            header = "lag_channels, num_pairs, corr_mean, corr_var\n"

        lines = [header]
        for lag, delta, count, mean, var in zip(lag_arr.tolist(), delta_arr.tolist(), count_arr.tolist(), mean_arr.tolist(), var_arr.tolist()):
            if has_mhz:
                delta_str = f"{delta:.6g}" if np.isfinite(delta) else "nan"
                lines.append(f"{lag}, {delta_str}, {count}, {mean:.6g}, {var:.6g}\n")
            else:
                lines.append(f"{lag}, {count}, {mean:.6g}, {var:.6g}\n")
        return "".join(lines)

    def _format_float(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6g}"

    def _diff_stats(cube: Tensor, order: int) -> Tuple[Optional[float], Optional[float]]:
        if order < 1:
            raise ValueError("diff order must be >= 1.")
        if cube.shape[config.freq_axis] < (order + 1):
            return None, None
        diff = torch.diff(cube, n=order, dim=config.freq_axis)
        mean_val = float(diff.mean().item())
        var_val = float(diff.var(unbiased=False).item())
        return mean_val, var_val

    def _maybe_resolve_fg_smooth_prior() -> Tuple[Union[Tensor, float], Union[Tensor, float], str]:
        explicit = False
        if hasattr(config, "provided_fields"):
            provided = getattr(config, "provided_fields")
            explicit = ("fg_smooth_mean" in provided) or ("fg_smooth_sigma" in provided)
        if config.fg_reference_cube and not explicit:
            mean_t, sigma_t = derive_smoothness_stats_from_cube(
                fg_true.detach(),
                freq_axis=config.freq_axis,
                use_robust=config.use_robust_fg_stats,
                mae_to_sigma_factor=config.mae_to_sigma_factor,
            )
            return mean_t.to(device=device, dtype=diag_dtype), sigma_t.to(device=device, dtype=diag_dtype), "reference_cube"
        return float(config.fg_smooth_mean), float(config.fg_smooth_sigma), "config"

    def _compute_loss_breakdown() -> Dict[str, Any]:
        sigma_data = torch.as_tensor(float(config.data_error), device=device, dtype=diag_dtype)
        sigma_data = torch.clamp(sigma_data, min=1e-8)

        if y_obs is not None:
            y_pred = forward_model(fg_true, eor_true, psf=None)
            data_loss = float(torch.mean(((y_pred - y_obs) / sigma_data) ** 2).item())
        else:
            data_loss = None

        fg_prior_mean, fg_prior_sigma, fg_prior_source = _maybe_resolve_fg_smooth_prior()
        smooth_loss = float(
            foreground_smoothness_loss(
                fg_true,
                freq_axis=config.freq_axis,
                prior_mean=fg_prior_mean,
                prior_sigma=fg_prior_sigma,
            ).item()
        )

        eor_mean = torch.as_tensor(float(config.eor_prior_mean), device=device, dtype=diag_dtype)
        eor_sigma = torch.as_tensor(float(config.eor_prior_sigma), device=device, dtype=diag_dtype)
        eor_sigma = torch.clamp(eor_sigma, min=1e-8)
        eor_reg = float(torch.mean(((eor_true - eor_mean) / eor_sigma) ** 2).item())

        corr_prior_mean = torch.as_tensor(float(config.corr_prior_mean), device=device, dtype=diag_dtype)
        corr_prior_sigma = torch.as_tensor(float(config.corr_prior_sigma), device=device, dtype=diag_dtype)
        corr_prior_sigma = torch.clamp(corr_prior_sigma, min=1e-8)
        corr_loss_t, corr_coeff_t = correlation_penalty(
            fg_true, eor_true, corr_prior_mean, corr_prior_sigma
        )
        corr_loss = float(corr_loss_t.item())
        corr_coeff = float(corr_coeff_t.item())

        base_total_no_data = (
            float(config.beta) * smooth_loss
            + float(config.gamma) * eor_reg
            + float(config.corr_weight) * corr_loss
        )
        base_total = None if data_loss is None else float(config.alpha) * data_loss + base_total_no_data

        fft_loss_val: Optional[float] = None
        fft_weighted_val: Optional[float] = None
        fft_error: Optional[str] = None
        try:
            energy_map = compute_highfreq_energy(
                fg_true, freq_axis=config.freq_axis, percent=float(config.fft_highfreq_percent)
            )
            prior_mean_t = torch.as_tensor(float(config.fft_prior_mean), device=device, dtype=diag_dtype)
            prior_sigma_t = torch.as_tensor(float(config.fft_prior_sigma), device=device, dtype=diag_dtype)
            prior_sigma_t = torch.clamp(prior_sigma_t, min=1e-8)
            fft_loss_val = float(torch.mean(((energy_map - prior_mean_t) / prior_sigma_t) ** 2).item())
            fft_weighted_val = float(config.fft_weight) * fft_loss_val
        except Exception as exc:
            fft_error = str(exc)

        poly_loss_val: Optional[float] = None
        poly_weighted_val: Optional[float] = None
        poly_error: Optional[str] = None
        try:
            poly_loss_val = float(
                polynomial_prior_loss(
                    fg_true,
                    freq_axis=config.freq_axis,
                    degree=int(config.poly_degree),
                    sigma=float(config.poly_sigma),
                    freqs=None,
                ).item()
            )
            poly_weighted_val = float(config.poly_weight) * poly_loss_val
        except Exception as exc:
            poly_error = str(exc)

        lagcorr_loss_val: Optional[float] = None
        lagcorr_weighted_val: Optional[float] = None
        lagcorr_error: Optional[str] = None
        lagcorr_used_lags: Optional[int] = None
        lagcorr_total_lags: Optional[int] = None
        try:
            unit_norm = str(config.lagcorr_unit).strip().lower()
            if unit_norm not in {"mhz", "chan"}:
                raise ValueError("lagcorr_unit must be 'mhz' or 'chan'.")
            intervals = list(config.lagcorr_intervals)
            if not intervals:
                raise ValueError("lagcorr_intervals must contain at least one interval.")
            lagcorr_total_lags = int(len(intervals))

            interval_t = torch.as_tensor(intervals, device=device, dtype=diag_dtype).reshape(-1)
            if unit_norm == "chan":
                lag_t = torch.round(interval_t).to(dtype=torch.int64)
                if not torch.allclose(interval_t, lag_t.to(dtype=diag_dtype), rtol=0.0, atol=1e-6):
                    raise ValueError("lagcorr_intervals must be integers when lagcorr_unit='chan'.")
                if torch.any(lag_t < 1):
                    raise ValueError("lagcorr_intervals entries must be >= 1 when lagcorr_unit='chan'.")
            else:
                df_mhz = config.freq_delta_mhz
                if df_mhz is None or float(df_mhz) <= 0:
                    num_freqs = int(fg_true.shape[config.freq_axis])
                    freqs_mhz = resolve_frequency_axis_mhz(
                        fg_true_cube_path,
                        num_freqs=num_freqs,
                        freq_axis=config.freq_axis,
                        config=config,
                        cube_ndim=fg_true.ndim,
                    )
                    if freqs_mhz is None:
                        raise ValueError(
                            "Unable to resolve frequency axis in MHz to convert lagcorr_intervals; "
                            "set freq_delta_mhz or freqs_mhz_path, or use lagcorr_unit='chan'."
                        )
                    deltas = np.diff(freqs_mhz.astype(np.float64, copy=False))
                    positive = np.abs(deltas[np.isfinite(deltas) & (deltas != 0.0)])
                    if positive.size == 0:
                        raise ValueError("Could not infer freq_delta_mhz from the frequency axis.")
                    df_mhz = float(np.median(positive))
                lag_t = torch.round(interval_t / float(df_mhz)).to(dtype=torch.int64)
                lag_t = torch.clamp(lag_t, min=1)

            num_freqs = int(fg_true.shape[config.freq_axis])
            valid_lags = lag_t < num_freqs
            lag_t = lag_t[valid_lags]
            if lag_t.numel() == 0:
                raise ValueError(
                    f"All lagcorr intervals produce lag_channels >= num_freqs ({num_freqs})."
                )
            lagcorr_used_lags = int(lag_t.numel())

            fg_mean_t_full = torch.as_tensor(list(config.fg_lagcorr_mean), device=device, dtype=diag_dtype).reshape(-1)
            fg_sigma_t_full = torch.as_tensor(list(config.fg_lagcorr_sigma), device=device, dtype=diag_dtype).reshape(-1)
            eor_mean_t_full = torch.as_tensor(list(config.eor_lagcorr_mean), device=device, dtype=diag_dtype).reshape(-1)
            eor_sigma_t_full = torch.as_tensor(list(config.eor_lagcorr_sigma), device=device, dtype=diag_dtype).reshape(-1)
            if fg_mean_t_full.numel() != interval_t.numel() or fg_sigma_t_full.numel() != interval_t.numel():
                raise ValueError("fg_lagcorr_mean/sigma must match lagcorr_intervals length.")
            if eor_mean_t_full.numel() != interval_t.numel() or eor_sigma_t_full.numel() != interval_t.numel():
                raise ValueError("eor_lagcorr_mean/sigma must match lagcorr_intervals length.")

            fg_mean_t = fg_mean_t_full[valid_lags]
            fg_sigma_t = fg_sigma_t_full[valid_lags]
            eor_mean_t = eor_mean_t_full[valid_lags]
            eor_sigma_t = eor_sigma_t_full[valid_lags]

            fg_lag_loss = _frequency_lag_correlation_loss(
                fg_true,
                freq_axis=config.freq_axis,
                lag_channels=lag_t,
                prior_mean=fg_mean_t,
                prior_sigma=fg_sigma_t,
                max_pairs=config.lagcorr_max_pairs,
            )
            eor_lag_loss = _frequency_lag_correlation_loss(
                eor_true,
                freq_axis=config.freq_axis,
                lag_channels=lag_t,
                prior_mean=eor_mean_t,
                prior_sigma=eor_sigma_t,
                max_pairs=config.lagcorr_max_pairs,
            )
            lagcorr_loss_val = float((0.5 * (fg_lag_loss + eor_lag_loss)).item())
            lagcorr_weighted_val = float(config.lagcorr_weight) * lagcorr_loss_val
        except Exception as exc:
            lagcorr_error = str(exc)

        extra_weighted_terms = [
            val for val in (fft_weighted_val, poly_weighted_val, lagcorr_weighted_val) if val is not None
        ]
        extra_weighted_total = float(sum(extra_weighted_terms)) if extra_weighted_terms else 0.0

        full_total = None if base_total is None else base_total + extra_weighted_total
        full_total_no_data = base_total_no_data + extra_weighted_total

        return {
            "data": data_loss,
            "smooth": smooth_loss,
            "eor_reg": eor_reg,
            "corr": corr_loss,
            "corr_coeff": corr_coeff,
            "fft_highfreq": fft_loss_val,
            "fft_highfreq_weighted": fft_weighted_val,
            "fft_highfreq_error": fft_error,
            "poly": poly_loss_val,
            "poly_weighted": poly_weighted_val,
            "poly_error": poly_error,
            "lagcorr": lagcorr_loss_val,
            "lagcorr_weighted": lagcorr_weighted_val,
            "lagcorr_error": lagcorr_error,
            "lagcorr_used_lags": lagcorr_used_lags,
            "lagcorr_total_lags": lagcorr_total_lags,
            "extra_weighted_total": extra_weighted_total,
            "base_total": base_total,
            "base_total_no_data": base_total_no_data,
            "full_total": full_total,
            "full_total_no_data": full_total_no_data,
            "fg_smooth_prior_source": 0.0 if fg_prior_source == "config" else 1.0,
        }

    fg_diff2_mean, fg_diff2_var = _diff_stats(fg_true, 2)
    fg_diff3_mean, fg_diff3_var = _diff_stats(fg_true, 3)
    eor_diff2_mean, eor_diff2_var = _diff_stats(eor_true, 2)
    eor_diff3_mean, eor_diff3_var = _diff_stats(eor_true, 3)
    losses = _compute_loss_breakdown()

    def _format_int(value: Optional[int]) -> str:
        if value is None:
            return "n/a"
        return str(int(value))

    def _format_lagcorr_table(rows: List[Dict[str, Any]], df_mhz_used: Optional[float]) -> str:
        header = (
            "idx, interval, lag_channels, pairs_used, "
            "fg_prior_mean, fg_prior_sigma, fg_corr_mean, fg_corr_var, fg_loss, "
            "eor_prior_mean, eor_prior_sigma, eor_corr_mean, eor_corr_var, eor_loss\n"
        )
        unit_norm = str(config.lagcorr_unit).strip().lower()
        prefix = f"unit={unit_norm}"
        if unit_norm == "mhz":
            prefix += f", df_mhz={_format_float(df_mhz_used)}"
        if config.lagcorr_max_pairs is not None:
            prefix += f", max_pairs={int(config.lagcorr_max_pairs)}"
        else:
            prefix += ", max_pairs=all"
        lines = [f"{prefix}\n", header]
        for row in rows:
            lines.append(
                f"{row['idx']}, "
                f"{_format_float(row.get('interval'))}, "
                f"{_format_int(row.get('lag_channels'))}, "
                f"{_format_int(row.get('pairs_used'))}, "
                f"{_format_float(row.get('fg_prior_mean'))}, "
                f"{_format_float(row.get('fg_prior_sigma'))}, "
                f"{_format_float(row.get('fg_corr_mean'))}, "
                f"{_format_float(row.get('fg_corr_var'))}, "
                f"{_format_float(row.get('fg_loss'))}, "
                f"{_format_float(row.get('eor_prior_mean'))}, "
                f"{_format_float(row.get('eor_prior_sigma'))}, "
                f"{_format_float(row.get('eor_corr_mean'))}, "
                f"{_format_float(row.get('eor_corr_var'))}, "
                f"{_format_float(row.get('eor_loss'))}\n"
            )
        return "".join(lines)

    def _compute_lagcorr_distributions() -> Tuple[List[Dict[str, Any]], Optional[float], Optional[str]]:
        unit_norm = str(config.lagcorr_unit).strip().lower()
        if unit_norm not in {"mhz", "chan"}:
            return [], None, "lagcorr_unit must be 'mhz' or 'chan'."

        intervals = list(config.lagcorr_intervals)
        if not intervals:
            return [], None, "lagcorr_intervals is empty."

        fg_mean_list = list(config.fg_lagcorr_mean)
        fg_sigma_list = list(config.fg_lagcorr_sigma)
        eor_mean_list = list(config.eor_lagcorr_mean)
        eor_sigma_list = list(config.eor_lagcorr_sigma)
        if len(fg_mean_list) != len(intervals) or len(fg_sigma_list) != len(intervals):
            return [], None, "fg_lagcorr_mean/sigma must match lagcorr_intervals length."
        if len(eor_mean_list) != len(intervals) or len(eor_sigma_list) != len(intervals):
            return [], None, "eor_lagcorr_mean/sigma must match lagcorr_intervals length."

        df_mhz_used: Optional[float] = None
        if unit_norm == "mhz":
            df_cfg = config.freq_delta_mhz
            if df_cfg is not None and float(df_cfg) > 0:
                df_mhz_used = float(df_cfg)
            else:
                num_freqs = int(fg_true.shape[config.freq_axis])
                freqs_mhz = resolve_frequency_axis_mhz(
                    fg_true_cube_path,
                    num_freqs=num_freqs,
                    freq_axis=config.freq_axis,
                    config=config,
                    cube_ndim=fg_true.ndim,
                )
                if freqs_mhz is None:
                    return [], None, (
                        "Unable to resolve df_mhz for lagcorr_unit='mhz'; "
                        "set freq_delta_mhz or freqs_mhz_path, or use lagcorr_unit='chan'."
                    )
                deltas = np.diff(freqs_mhz.astype(np.float64, copy=False))
                positive = np.abs(deltas[np.isfinite(deltas) & (deltas != 0.0)])
                if positive.size == 0:
                    return [], None, "Could not infer df_mhz from the frequency axis."
                df_mhz_used = float(np.median(positive))

        def _pairwise_correlations(cube: Tensor, lag: int) -> Tuple[Tensor, int]:
            moved = cube.movedim(config.freq_axis, 0)
            num_freqs = int(moved.shape[0])
            flat = moved.reshape(num_freqs, -1)
            means = flat.mean(dim=1, keepdim=True)
            centered = flat - means
            norms = torch.norm(centered, dim=1)
            norms = torch.clamp(norms, min=1e-8)
            num_total = num_freqs - int(lag)
            if num_total <= 0:
                return torch.zeros((0,), device=cube.device, dtype=cube.dtype), 0
            max_pairs = config.lagcorr_max_pairs
            if max_pairs is not None and int(max_pairs) <= 0:
                max_pairs = None
            if max_pairs is None:
                num_pairs = num_total
            else:
                num_pairs = min(num_total, int(max_pairs))
            if num_pairs <= 0:
                return torch.zeros((0,), device=cube.device, dtype=cube.dtype), 0
            pair_idx = torch.arange(num_pairs, device=cube.device)
            pair_j = pair_idx + int(lag)
            a = centered.index_select(0, pair_idx)
            b = centered.index_select(0, pair_j)
            dot = torch.sum(a * b, dim=1)
            denom = norms.index_select(0, pair_idx) * norms.index_select(0, pair_j)
            denom = torch.clamp(denom, min=1e-8)
            corr = dot / denom
            return corr, int(num_pairs)

        rows: List[Dict[str, Any]] = []
        num_freqs_total = int(fg_true.shape[config.freq_axis])
        for idx, interval in enumerate(intervals):
            row: Dict[str, Any] = {
                "idx": int(idx),
                "interval": float(interval),
                "fg_prior_mean": float(fg_mean_list[idx]),
                "fg_prior_sigma": float(fg_sigma_list[idx]),
                "eor_prior_mean": float(eor_mean_list[idx]),
                "eor_prior_sigma": float(eor_sigma_list[idx]),
            }

            try:
                if unit_norm == "chan":
                    lag_round = int(round(float(interval)))
                    if not math.isclose(float(interval), float(lag_round), rel_tol=0.0, abs_tol=1e-6):
                        raise ValueError("interval is not an integer channel lag.")
                    if lag_round < 1:
                        raise ValueError("lag_channels must be >= 1.")
                    lag_channels = lag_round
                else:
                    assert df_mhz_used is not None
                    lag_channels = int(round(float(interval) / float(df_mhz_used)))
                    lag_channels = max(1, lag_channels)
                row["lag_channels"] = int(lag_channels)

                if lag_channels >= num_freqs_total:
                    row["pairs_used"] = 0
                    rows.append(row)
                    continue

                fg_corr, pairs_used = _pairwise_correlations(fg_true, lag_channels)
                eor_corr, _ = _pairwise_correlations(eor_true, lag_channels)
                row["pairs_used"] = int(pairs_used)

                if pairs_used > 0:
                    row["fg_corr_mean"] = float(fg_corr.mean().item())
                    row["fg_corr_var"] = float(fg_corr.var(unbiased=False).item())
                    row["eor_corr_mean"] = float(eor_corr.mean().item())
                    row["eor_corr_var"] = float(eor_corr.var(unbiased=False).item())

                    fg_sigma = max(float(row["fg_prior_sigma"]), 1e-8)
                    eor_sigma = max(float(row["eor_prior_sigma"]), 1e-8)
                    fg_loss_k = torch.mean(((fg_corr - float(row["fg_prior_mean"])) / fg_sigma) ** 2)
                    eor_loss_k = torch.mean(((eor_corr - float(row["eor_prior_mean"])) / eor_sigma) ** 2)
                    row["fg_loss"] = float(fg_loss_k.item())
                    row["eor_loss"] = float(eor_loss_k.item())
            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

        return rows, df_mhz_used, None

    lagcorr_rows, lagcorr_df_mhz, lagcorr_rows_error = _compute_lagcorr_distributions()

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Input diagnostics report\n")
        handle.write("========================\n\n")
        handle.write("Autocorrelations include FG↔FG and EoR↔EoR only.\n")
        handle.write("No FG↔EoR autocorrelation report is computed.\n\n")
        if cut_indices is not None:
            handle.write(
                "cut_xy: "
                f"x[{cut_indices.x0}:{cut_indices.x1}] "
                f"y[{cut_indices.y0}:{cut_indices.y1}] "
                f"(size={cut_indices.size_px}px, unit={cut_indices.unit}, freq_axis={cut_indices.freq_axis})\n\n"
            )
        handle.write("Smoothness stats (finite differences along freq axis):\n")
        handle.write("  foreground:\n")
        handle.write(f"    diff2 mean: {_format_float(fg_diff2_mean)}\n")
        handle.write(f"    diff2 var:  {_format_float(fg_diff2_var)}\n")
        handle.write(f"    diff3 mean: {_format_float(fg_diff3_mean)}\n")
        handle.write(f"    diff3 var:  {_format_float(fg_diff3_var)}\n")
        handle.write("  eor:\n")
        handle.write(f"    diff2 mean: {_format_float(eor_diff2_mean)}\n")
        handle.write(f"    diff2 var:  {_format_float(eor_diff2_var)}\n")
        handle.write(f"    diff3 mean: {_format_float(eor_diff3_mean)}\n")
        handle.write(f"    diff3 var:  {_format_float(eor_diff3_var)}\n\n")

        handle.write("Loss breakdown (evaluated on fg_reference_cube + true_eor_cube):\n")
        if input_cube_path is not None and y_obs is not None:
            handle.write(f"  data (vs input_cube={input_cube_path}): {_format_float(losses['data'])}\n")
        else:
            handle.write("  data: n/a (no input_cube provided or not found)\n")
        handle.write(f"  smooth:     {_format_float(losses['smooth'])}\n")
        handle.write(f"  eor_reg:    {_format_float(losses['eor_reg'])}\n")
        handle.write(f"  corr:       {_format_float(losses['corr'])}\n")
        handle.write(f"  corr_coeff: {_format_float(losses['corr_coeff'])}\n")
        handle.write(f"  lagcorr:    {_format_float(losses.get('lagcorr'))}\n")
        if losses.get("lagcorr_error"):
            handle.write(f"    lagcorr_error: {losses['lagcorr_error']}\n")
        if losses.get("lagcorr_used_lags") is not None and losses.get("lagcorr_total_lags") is not None:
            handle.write(
                f"    lagcorr_lags_used: {losses['lagcorr_used_lags']}/{losses['lagcorr_total_lags']}\n"
            )
        handle.write(f"  fft:       {_format_float(losses.get('fft_highfreq'))}\n")
        if losses.get("fft_highfreq_error"):
            handle.write(f"    fft_error: {losses['fft_highfreq_error']}\n")
        handle.write(f"  poly:      {_format_float(losses.get('poly'))}\n")
        if losses.get("poly_error"):
            handle.write(f"    poly_error: {losses['poly_error']}\n")
        handle.write("\n")
        handle.write(f"  base_total (with data):     {_format_float(losses.get('base_total'))}\n")
        handle.write(f"  base_total (no data):       {_format_float(losses.get('base_total_no_data'))}\n")
        handle.write(f"  extra_weighted_total:       {_format_float(losses.get('extra_weighted_total'))}\n")
        handle.write(f"  full_total (with data):     {_format_float(losses.get('full_total'))}\n")
        handle.write(f"  full_total (no data):       {_format_float(losses.get('full_total_no_data'))}\n\n")
        handle.write(
            "extra loss schedule (during optimization): "
            f"start_iter={config.extra_loss_start_iter}, ramp_iters={config.extra_loss_ramp_iters}\n\n"
        )

        handle.write("lagcorr distributions (configured intervals; pairs match lagcorr_max_pairs):\n")
        if lagcorr_rows_error:
            handle.write(f"  error: {lagcorr_rows_error}\n\n")
        else:
            handle.write(_format_lagcorr_table(lagcorr_rows, lagcorr_df_mhz))
            handle.write("\n")

        handle.write(_format_stats(fg_stats))
        handle.write("  per-lag summary (foreground):\n")
        handle.write(_format_lag_table(fg_stats))
        handle.write("\n")
        handle.write(_format_stats(eor_stats))
        handle.write("  per-lag summary (eor):\n")
        handle.write(_format_lag_table(eor_stats))
        handle.write("\n")
        handle.write(f"Detailed pair correlations: {details_path}\n")
        handle.write(f"Mean/variance vs lag plot:  {plot_path}\n")

    fg_has_mhz = bool(fg_stats.get("freqs_mhz_available"))
    eor_has_mhz = bool(eor_stats.get("freqs_mhz_available"))
    use_mhz_axis = fg_has_mhz and eor_has_mhz

    fg_x = (
        fg_stats["delta_mhz_median"].astype(np.float64, copy=False)
        if use_mhz_axis
        else fg_stats["lag_channels"].astype(np.float64, copy=False)
    )
    eor_x = (
        eor_stats["delta_mhz_median"].astype(np.float64, copy=False)
        if use_mhz_axis
        else eor_stats["lag_channels"].astype(np.float64, copy=False)
    )
    if fg_x.shape != eor_x.shape or (use_mhz_axis and not np.allclose(fg_x, eor_x, rtol=1e-3, atol=1e-6, equal_nan=True)):
        # Fall back to channel lags if the frequency axes don't align.
        fg_x = fg_stats["lag_channels"].astype(np.float64, copy=False)
        eor_x = eor_stats["lag_channels"].astype(np.float64, copy=False)
        use_mhz_axis = False

    x_label = "Frequency lag (MHz)" if use_mhz_axis else "Frequency lag (channels)"
    fg_mean = fg_stats["corr_mean"].astype(np.float64, copy=False)
    eor_mean = eor_stats["corr_mean"].astype(np.float64, copy=False)
    fg_std = np.sqrt(np.maximum(fg_stats["corr_var"].astype(np.float64, copy=False), 0.0))
    eor_std = np.sqrt(np.maximum(eor_stats["corr_var"].astype(np.float64, copy=False), 0.0))

    save_true_signal_corr_vs_lag_plot(
        lag_x=fg_x,
        fg_mean=fg_mean,
        fg_std=fg_std,
        eor_mean=eor_mean,
        eor_std=eor_std,
        x_label=x_label,
        path=plot_path,
    )

    return summary_path, details_path, plot_path


def write_true_signal_correlation_report(
    fg_true_cube_path: Path,
    eor_true_cube_path: Path,
    config: OptimizationConfig,
    output_dir: Path,
    filename_prefix: str,
    input_cube_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Deprecated alias for write_input_diagnostics_report.
    """
    return write_input_diagnostics_report(
        fg_true_cube_path,
        eor_true_cube_path,
        config,
        output_dir,
        filename_prefix,
        input_cube_path=input_cube_path,
    )


def save_correlation_plot(correlations: np.ndarray, path: Path) -> None:
    """
    Plot correlation vs. frequency index and save to disk.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Plotting correlations requires the 'matplotlib' package.") from exc

    freq_idx = np.arange(len(correlations))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq_idx, correlations, marker="o")
    ax.set_xlabel("Frequency channel")
    ax.set_ylabel("Correlation")
    ax.set_title("EoR estimate vs. reference correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_eor_estimate(
    eor_est: Tensor,
    eor_true: Tensor,
    freq_axis: int,
    output_path: Path,
) -> None:
    correlations = compute_frequency_correlations(eor_est, eor_true, freq_axis=freq_axis)
    save_correlation_plot(correlations, output_path)
    print(
        f"EoR correlation stats — mean: {np.nanmean(correlations):.4f}, "
        f"min: {np.nanmin(correlations):.4f}, max: {np.nanmax(correlations):.4f}"
    )
    print(f"Saved correlation plot to {output_path}")


def _optimize_from_fits(
    input_path: Path,
    fg_output: Path,
    eor_output: Path,
    config: OptimizationConfig,
) -> None:
    device = config.resolved_device()
    dtype = config.resolved_dtype()

    print(f"Loading cube from {input_path} on device {device} ...")
    y_cube = read_fits_cube(input_path)

    mask_tensor: Optional[Tensor] = None
    if config.mask_cube:
        mask_path = Path(config.mask_cube)
        print(f"Loading mask cube from {mask_path} ...")
        mask_tensor = read_fits_array(mask_path)
        y_cube = apply_mask_xy(y_cube, mask_tensor, freq_axis=config.freq_axis)

    cut_indices = build_cut_xy_indices(tuple(y_cube.shape), freq_axis=config.freq_axis, config=config)
    if cut_indices is not None:
        print(
            "Applying cut_xy: "
            f"x[{cut_indices.x0}:{cut_indices.x1}] y[{cut_indices.y0}:{cut_indices.y1}] "
            f"(size={cut_indices.size_px}px, unit={cut_indices.unit}, freq_axis={cut_indices.freq_axis})"
        )
        y_cube = apply_cut_xy(y_cube, cut_indices)

    eor_true: Optional[Tensor] = None
    if config.true_eor_cube:
        true_eor_path = Path(config.true_eor_cube)
        print(f"Loading reference EoR cube from {true_eor_path} ...")
        eor_true = read_fits_cube(true_eor_path)
        if mask_tensor is not None:
            eor_true = apply_mask_xy(eor_true, mask_tensor, freq_axis=config.freq_axis)
        if cut_indices is not None:
            eor_true = apply_cut_xy(eor_true, cut_indices)

    data_error_value: Union[Tensor, float] = config.data_error
    eor_mean_value: Union[Tensor, float] = config.eor_prior_mean
    eor_sigma_value: Union[Tensor, float] = config.eor_prior_sigma

    fg_mean_value: Union[Tensor, float, None]
    fg_sigma_value: Union[Tensor, float, None]

    explicit_smooth_prior = (
        "fg_smooth_mean" in config.provided_fields or "fg_smooth_sigma" in config.provided_fields
    )

    if config.fg_reference_cube:
        ref_path = Path(config.fg_reference_cube)
        fg_ref_cube = read_fits_cube(ref_path)
        if mask_tensor is not None:
            fg_ref_cube = apply_mask_xy(fg_ref_cube, mask_tensor, freq_axis=config.freq_axis)
        if cut_indices is not None:
            fg_ref_cube = apply_cut_xy(fg_ref_cube, cut_indices)

        if explicit_smooth_prior:
            fg_mean_value = config.fg_smooth_mean
            fg_sigma_value = config.fg_smooth_sigma
            print(
                "Using fg_smooth_mean/fg_smooth_sigma from config/CLI; "
                "skipping smoothness derivation from fg_reference_cube."
            )
        else:
            print(
                f"Deriving FG smoothness stats from reference cube {ref_path} "
                f"(robust={config.use_robust_fg_stats}, mae_to_sigma_factor={config.mae_to_sigma_factor}) ..."
            )
            fg_mean_value, fg_sigma_value = derive_smoothness_stats_from_cube(
                fg_ref_cube,
                freq_axis=config.freq_axis,
                use_robust=config.use_robust_fg_stats,
                mae_to_sigma_factor=config.mae_to_sigma_factor,
            )
            _summarize_fg_stats(fg_mean_value, fg_sigma_value, freq_axis=config.freq_axis)

        fft_mean_value: Optional[Union[Tensor, float]] = config.fft_prior_mean
        fft_sigma_value: Optional[Union[Tensor, float]] = config.fft_prior_sigma
        if config.loss_mode == "rfft":
            fft_mean_value, fft_sigma_value = derive_fft_prior_from_cube(
                fg_ref_cube,
                freq_axis=config.freq_axis,
                percent=config.fft_highfreq_percent,
                use_robust=config.use_robust_fg_stats,
                mae_to_sigma_factor=config.mae_to_sigma_factor,
            )
            print(
                f"High-frequency energy stats (reference): mean={float(fft_mean_value.item()):.4e}, "
                f"sigma={float(fft_sigma_value.item()):.4e}"
            )
        else:
            if fft_mean_value is None or fft_sigma_value is None:
                fft_mean_value = torch.zeros(1, dtype=y_cube.dtype)
                fft_sigma_value = torch.ones(1, dtype=y_cube.dtype)
    else:
        fg_mean_value = config.fg_smooth_mean
        fg_sigma_value = config.fg_smooth_sigma
        fft_mean_value = config.fft_prior_mean
        fft_sigma_value = (
            config.fft_prior_sigma
            if config.fft_prior_sigma is not None
            else DEFAULT_FFT_SIGMA
        )

    fg_init_tensor = None
    if config.init_fg_cube:
        fg_init_path = Path(config.init_fg_cube)
        print(f"Loading initial foreground guess from {fg_init_path} ...")
        fg_init_tensor = read_fits_cube(fg_init_path)
        if mask_tensor is not None:
            fg_init_tensor = apply_mask_xy(fg_init_tensor, mask_tensor, freq_axis=config.freq_axis)
        if cut_indices is not None:
            fg_init_tensor = apply_cut_xy(fg_init_tensor, cut_indices)

    eor_init_tensor = None
    if config.init_eor_cube:
        eor_init_path = Path(config.init_eor_cube)
        print(f"Loading initial EoR guess from {eor_init_path} ...")
        eor_init_tensor = read_fits_cube(eor_init_path)
        if mask_tensor is not None:
            eor_init_tensor = apply_mask_xy(eor_init_tensor, mask_tensor, freq_axis=config.freq_axis)
        if cut_indices is not None:
            eor_init_tensor = apply_cut_xy(eor_init_tensor, cut_indices)

    _warn_weight_defaults(config)

    fg_est, eor_est, history = optimize_components(
        y_cube,
        num_iters=config.num_iters,
        lr=config.lr,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        freq_axis=config.freq_axis,
        print_every=config.print_every,
        device=device,
        dtype=dtype,
        fg_init_tensor=fg_init_tensor,
        eor_init_tensor=eor_init_tensor,
        data_error=data_error_value,
        eor_prior_mean=eor_mean_value,
        eor_prior_sigma=eor_sigma_value,
        fg_smooth_mean=fg_mean_value,
        fg_smooth_sigma=fg_sigma_value,
        corr_prior_mean=config.corr_prior_mean,
        corr_prior_sigma=config.corr_prior_sigma,
        corr_weight=config.corr_weight,
        lagcorr_weight=config.lagcorr_weight,
        lagcorr_unit=config.lagcorr_unit,
        lagcorr_intervals=config.lagcorr_intervals,
        fg_lagcorr_mean=config.fg_lagcorr_mean,
        fg_lagcorr_sigma=config.fg_lagcorr_sigma,
        eor_lagcorr_mean=config.eor_lagcorr_mean,
        eor_lagcorr_sigma=config.eor_lagcorr_sigma,
        lagcorr_max_pairs=config.lagcorr_max_pairs,
        extra_loss_start_iter=config.extra_loss_start_iter,
        extra_loss_ramp_iters=config.extra_loss_ramp_iters,
        fft_weight=config.fft_weight,
        poly_weight=config.poly_weight,
        poly_degree=config.poly_degree,
        poly_sigma=config.poly_sigma,
        loss_mode=config.loss_mode,
        fft_prior_mean=fft_mean_value,
        fft_prior_sigma=fft_sigma_value,
        fft_highfreq_percent=config.fft_highfreq_percent,
        optimizer_name=config.optimizer_name,
        momentum=config.momentum,
        freq_start_mhz=config.freq_start_mhz,
        freq_delta_mhz=config.freq_delta_mhz,
        eor_true_tensor=eor_true if config.enable_corr_check else None,
        corr_check_every=config.corr_check_every if config.enable_corr_check else 0,
    )

    header_extras = cut_xy_fits_header(cut_indices) if cut_indices is not None else None
    write_fits_cube(fg_est, fg_output, header_extras=header_extras)
    write_fits_cube(eor_est, eor_output, header_extras=header_extras)

    final = history[-1] if history else None
    if final:
        print(
            f"Finished optimization: total={final.total:.4e}, "
            f"data={final.data:.4e}, smooth={final.smooth:.4e}, "
            f"eor={final.eor_reg:.4e}, corr={final.corr:.4e}, "
            f"lagcorr={final.lagcorr:.4e}, "
            f"corr_coeff={final.corr_coeff:.3f}, "
            f"fft={final.fft_highfreq:.4e}, poly={final.poly:.4e}"
        )
    print(f"Saved foreground estimate to {fg_output}")
    print(f"Saved EoR estimate to {eor_output}")

    if config.true_eor_cube and eor_true is not None:
        plot_path = (
            Path(config.corr_plot)
            if config.corr_plot
            else eor_output.with_name(f"{eor_output.stem}_corr.png")
        )
        evaluate_eor_estimate(eor_est, eor_true, freq_axis=config.freq_axis, output_path=plot_path)

    if config.power_config:
        power_cfg_path = Path(config.power_config)
        if not power_cfg_path.exists():
            print(f"Power config '{power_cfg_path}' not found; skipping power spectrum computation.")
        else:
            power_cfg_data = load_config_file(power_cfg_path)
            for key in ("dx", "dy", "df"):
                if key not in power_cfg_data:
                    print(f"Power config missing '{key}', skipping power spectrum computation.")
                    power_cfg_data = None
                    break
            if power_cfg_data is None:
                return
            power_cfg = PowerSpecConfig(**power_cfg_data)
            power_cfg.freq_axis = config.freq_axis
            if config.freq_start_mhz is not None and config.freq_delta_mhz is not None:
                power_cfg.ref_freq_mhz = config.freq_start_mhz
                power_cfg.df = config.freq_delta_mhz
                power_cfg.unit_f = "mhz"
            if config.power_output_dir:
                output_dir = Path(config.power_output_dir)
            else:
                output_dir = eor_output.with_name(f"{eor_output.stem}_powerspec")
            print(f"Computing power spectra to {output_dir} ...")
            # Keep FFT on the training device when possible by passing
            # the tensor directly to compute_power_spectra.
            rec_power = compute_power_spectra(eor_est.detach(), power_cfg)
            true_power = None
            if config.true_eor_cube:
                # For the true EoR cube, move to the same device as the
                # optimizer (if desired) to reuse the GPU FFT path.
                if device.type == "cuda":
                    eor_true_eval = eor_true.to(device)
                else:
                    eor_true_eval = eor_true
                true_power = compute_power_spectra(eor_true_eval, power_cfg)
            # Use PowerSpecConfig.log_power_2d to control whether 2D power
            # spectra are plotted in log10 or linear scale, and
            # PowerSpecConfig.log_bins_2d to decide whether to use
            # logarithmic axes for k_perp / k_par.
            save_power_outputs(
                Path(output_dir),
                rec_power,
                true_power,
                log_power_2d=power_cfg.log_power_2d,
                log_axes_2d=power_cfg.log_bins_2d,
            )


def run_synthetic_demo(config: OptimizationConfig) -> None:
    torch.manual_seed(0)
    device = config.resolved_device()
    dtype = config.resolved_dtype()
    print(f"Running synthetic demo on device: {device}")

    ignored_fields = [
        ("fg_reference_cube", config.fg_reference_cube),
    ]
    if any(path for _, path in ignored_fields if path):
        print("Warning: cube-based priors are ignored in the synthetic demo.")

    fg_true, eor_true = _create_synthetic_cube(num_freqs=40, image_shape=(24, 24), device=device)
    y_true = fg_true + eor_true
    noise = 0.01 * torch.randn_like(y_true)
    y_noisy = y_true + noise
    cut_indices = build_cut_xy_indices(tuple(y_noisy.shape), freq_axis=config.freq_axis, config=config)
    if cut_indices is not None:
        print(
            "Applying cut_xy: "
            f"x[{cut_indices.x0}:{cut_indices.x1}] y[{cut_indices.y0}:{cut_indices.y1}] "
            f"(size={cut_indices.size_px}px, unit={cut_indices.unit}, freq_axis={cut_indices.freq_axis})"
        )
        fg_true = apply_cut_xy(fg_true, cut_indices)
        eor_true = apply_cut_xy(eor_true, cut_indices)
        y_noisy = apply_cut_xy(y_noisy, cut_indices)

    print("Starting optimization...")
    _warn_weight_defaults(config)
    fg_est, eor_est, _ = optimize_components(
        y_noisy,
        num_iters=config.num_iters,
        lr=config.lr,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        freq_axis=config.freq_axis,
        print_every=config.print_every,
        device=device,
        dtype=dtype,
        data_error=config.data_error,
        eor_prior_mean=config.eor_prior_mean,
        eor_prior_sigma=config.eor_prior_sigma,
        fg_smooth_mean=config.fg_smooth_mean,
        fg_smooth_sigma=config.fg_smooth_sigma,
        corr_prior_mean=config.corr_prior_mean,
        corr_prior_sigma=config.corr_prior_sigma,
        corr_weight=config.corr_weight,
        lagcorr_weight=config.lagcorr_weight,
        lagcorr_unit=config.lagcorr_unit,
        lagcorr_intervals=config.lagcorr_intervals,
        fg_lagcorr_mean=config.fg_lagcorr_mean,
        fg_lagcorr_sigma=config.fg_lagcorr_sigma,
        eor_lagcorr_mean=config.eor_lagcorr_mean,
        eor_lagcorr_sigma=config.eor_lagcorr_sigma,
        lagcorr_max_pairs=config.lagcorr_max_pairs,
        extra_loss_start_iter=config.extra_loss_start_iter,
        extra_loss_ramp_iters=config.extra_loss_ramp_iters,
        fft_weight=config.fft_weight,
        poly_weight=config.poly_weight,
        poly_degree=config.poly_degree,
        poly_sigma=config.poly_sigma,
        loss_mode=config.loss_mode,
        fft_prior_mean=config.fft_prior_mean,
        fft_prior_sigma=config.fft_prior_sigma,
        fft_highfreq_percent=config.fft_highfreq_percent,
        optimizer_name=config.optimizer_name,
        momentum=config.momentum,
        freq_start_mhz=config.freq_start_mhz,
        freq_delta_mhz=config.freq_delta_mhz,
        eor_true_tensor=eor_true if config.enable_corr_check else None,
        corr_check_every=config.corr_check_every if config.enable_corr_check else 0,
    )

    fg_mse = torch.mean((fg_est - fg_true) ** 2).item()
    eor_mse = torch.mean((eor_est - eor_true) ** 2).item()
    recon_error = torch.mean((fg_est + eor_est - y_noisy) ** 2).item()

    print("\nRecovered component statistics:")
    print(f"Foreground MSE: {fg_mse:.4e}")
    print(f"EoR MSE:        {eor_mse:.4e}")
    print(f"Data fidelity:  {recon_error:.4e}")
