#!/usr/bin/env python3
"""
Optimization-based prototype for separating smooth foreground and fluctuating
EoR components from a 3D data cube.

Copyright (c) 2025 Zhenghao Zhu
Licensed under the MIT License. See LICENSE file for details.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    poly_weight: float = 1.0
    poly_degree: int = 3
    poly_sigma: float = DEFAULT_POLY_SIGMA
    loss_mode: str = "base"  # "base", "rfft", or "poly_reparam"
    optimizer_name: str = "adam"
    momentum: float = 0.9
    power_config: Optional[str] = None
    power_output_dir: Optional[str] = None
    freq_axis: int = 0
    print_every: int = 50
    device: Optional[str] = None
    dtype: Optional[str] = None
    true_eor_cube: Optional[str] = None
    corr_plot: Optional[str] = None
    init_fg_cube: Optional[str] = None
    init_eor_cube: Optional[str] = None
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
    fft_highfreq_percent: float = 0.7
    fft_prior_mean: float = 0.0
    fft_prior_sigma: float = DEFAULT_FFT_SIGMA
    enable_corr_check: bool = False
    corr_check_every: int = 500
    freq_start_mhz: Optional[float] = None
    freq_delta_mhz: Optional[float] = None
    freqs_mhz_path: Optional[str] = None

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for field in fields(self):
            if field.name in data and data[field.name] is not None:
                setattr(self, field.name, data[field.name])

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
        fft_weight: Weight for the high-frequency penalty (rFFT mode).
        poly_weight: Weight for the polynomial prior term.
        poly_degree: Degree for polynomial priors.
        poly_sigma: Std for polynomial prior residuals.
        loss_mode: "base" (default), "rfft", or "poly_reparam".
        fft_prior_mean: Prior mean for high-frequency energy (scalar or tensor).
        fft_prior_sigma: Prior std for high-frequency energy (scalar or tensor).
        fft_highfreq_percent: Fraction (0-1) of the highest frequency bins to penalize.
        freq_start_mhz: Starting frequency of the cube (MHz) for polynomial modes.
        freq_delta_mhz: Frequency spacing of the cube (MHz) for polynomial modes.
    """
    if loss_mode not in {"base", "rfft", "poly", "poly_reparam"}:
        raise ValueError("loss_mode must be 'base', 'rfft', 'poly', or 'poly_reparam'.")
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
            freq_axis=freq_axis,
            data_error=data_error_tensor,
            eor_mean=eor_mean_tensor,
            eor_sigma=eor_sigma_tensor,
            fg_smooth_mean=fg_mean_tensor,
            fg_smooth_sigma=fg_sigma_tensor,
            corr_prior_mean=corr_mean_tensor,
            corr_prior_sigma=corr_sigma_tensor,
            loss_mode=loss_mode,
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
            fft_highfreq=float(components["fft_highfreq"].item()),
            poly=float(components["poly"].item()),
        )
        history.append(entry)

        if print_every and (it == 1 or it % print_every == 0 or it == num_iters):
            print(
                f"[iter {it:04d}] total={entry.total:.4e} "
                f"data={entry.data:.4e} smooth={entry.smooth:.4e} "
                f"eor={entry.eor_reg:.4e} corr={entry.corr:.4e} "
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


def write_fits_cube(tensor: Tensor, path: Path) -> None:
    """
    Save a tensor to a FITS file, moving to CPU if needed.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError("Writing FITS files requires the 'astropy' package.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().numpy()
    fits.writeto(path, array, overwrite=True)


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

    eor_true: Optional[Tensor] = None
    if config.true_eor_cube:
        true_eor_path = Path(config.true_eor_cube)
        print(f"Loading reference EoR cube from {true_eor_path} ...")
        eor_true = read_fits_cube(true_eor_path)

    data_error_value: Union[Tensor, float] = config.data_error
    eor_mean_value: Union[Tensor, float] = config.eor_prior_mean
    eor_sigma_value: Union[Tensor, float] = config.eor_prior_sigma

    fg_mean_value: Union[Tensor, float, None]
    fg_sigma_value: Union[Tensor, float, None]

    if config.fg_reference_cube:
        ref_path = Path(config.fg_reference_cube)
        print(
            f"Deriving FG smoothness stats from reference cube {ref_path} "
            f"(robust={config.use_robust_fg_stats}, mae_to_sigma_factor={config.mae_to_sigma_factor}) ..."
        )
        fg_ref_cube = read_fits_cube(ref_path)
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

    eor_init_tensor = None
    if config.init_eor_cube:
        eor_init_path = Path(config.init_eor_cube)
        print(f"Loading initial EoR guess from {eor_init_path} ...")
        eor_init_tensor = read_fits_cube(eor_init_path)

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

    write_fits_cube(fg_est, fg_output)
    write_fits_cube(eor_est, eor_output)

    final = history[-1] if history else None
    if final:
        print(
            f"Finished optimization: total={final.total:.4e}, "
            f"data={final.data:.4e}, smooth={final.smooth:.4e}, "
            f"eor={final.eor_reg:.4e}, corr={final.corr:.4e}, "
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
