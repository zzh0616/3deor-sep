#!/usr/bin/env python3
"""Linear-Gaussian utilities for observation-anchored foreground separation.

The nuisance columns passed to this module are already scaled by their prior
standard deviations.  The latent coefficients therefore have an independent
standard-normal prior, which keeps heterogeneous catalog and diffuse
parameters on the same numerical scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LinearGaussianControlResult:
    posterior_mean: torch.Tensor
    posterior_covariance: torch.Tensor
    feature_scale: torch.Tensor
    prediction: torch.Tensor
    residual: torch.Tensor
    stats: dict[str, Any]


def _linear_hat_weights(length: int, cells: int) -> np.ndarray:
    if int(length) <= 0 or int(cells) <= 0:
        raise ValueError("length and cells must be positive")
    if int(cells) == 1:
        return np.ones((1, int(length)), dtype=np.float64)
    coordinate = np.linspace(0.0, float(cells - 1), int(length), dtype=np.float64)
    left = np.floor(coordinate).astype(np.int64)
    right = np.minimum(left + 1, int(cells) - 1)
    fraction = coordinate - left
    weights = np.zeros((int(cells), int(length)), dtype=np.float64)
    indices = np.arange(int(length), dtype=np.int64)
    weights[left, indices] += 1.0 - fraction
    weights[right, indices] += fraction
    return weights


def partition_of_unity_grid(
    shape: tuple[int, int],
    grid: tuple[int, int],
) -> np.ndarray:
    """Return smooth bilinear sky weights with shape ``[ny*nx, y, x]``."""

    height, width = (int(value) for value in shape)
    grid_y, grid_x = (int(value) for value in grid)
    wy = _linear_hat_weights(height, grid_y)
    wx = _linear_hat_weights(width, grid_x)
    weights = np.einsum("ay,bx->abyx", wy, wx, optimize=True).reshape(
        grid_y * grid_x, height, width
    )
    normalizer = np.sum(weights, axis=0, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise AssertionError("Partition of unity has uncovered pixels")
    return weights / normalizer


def prior_predictive_feature_scale(
    design: torch.Tensor,
    *,
    floor_quantile: float = 0.1,
    floor_fraction_of_max: float = 1.0e-6,
) -> torch.Tensor:
    """Estimate a stable feature scale from the nuisance prior predictive RMS."""

    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("design must be a non-empty [features, parameters] matrix")
    if not 0.0 <= float(floor_quantile) <= 1.0:
        raise ValueError("floor_quantile must be in [0, 1]")
    if float(floor_fraction_of_max) <= 0.0:
        raise ValueError("floor_fraction_of_max must be positive")
    row_rms = torch.linalg.vector_norm(design.double(), dim=1)
    positive = row_rms[row_rms > 0.0]
    if positive.numel() == 0:
        raise ValueError("design has no nonzero prior-predictive features")
    quantile_floor = torch.quantile(positive, float(floor_quantile))
    relative_floor = torch.max(positive) * float(floor_fraction_of_max)
    dtype_floor = torch.as_tensor(
        torch.finfo(torch.float64).tiny,
        dtype=torch.float64,
        device=design.device,
    )
    floor = torch.maximum(torch.maximum(quantile_floor, relative_floor), dtype_floor)
    return torch.clamp(row_rms, min=floor)


def solve_linear_gaussian_control(
    design: torch.Tensor,
    data: torch.Tensor,
    *,
    feature_scale: torch.Tensor | None = None,
    prior_precision: float | torch.Tensor = 1.0,
    feature_floor_quantile: float = 0.1,
    feature_floor_fraction_of_max: float = 1.0e-6,
    hessian_jitter_fraction: float = 1.0e-12,
) -> LinearGaussianControlResult:
    """Solve a whitened linear likelihood with a finite Gaussian prior."""

    matrix = design.double()
    vector = data.double().reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != vector.numel():
        raise ValueError("design/data feature dimensions differ")
    if not torch.isfinite(matrix).all() or not torch.isfinite(vector).all():
        raise ValueError("design and data must be finite")
    if feature_scale is None:
        scale = prior_predictive_feature_scale(
            matrix,
            floor_quantile=float(feature_floor_quantile),
            floor_fraction_of_max=float(feature_floor_fraction_of_max),
        )
    else:
        scale = feature_scale.double().reshape(-1)
        if scale.numel() != vector.numel():
            raise ValueError("feature_scale/data dimensions differ")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0.0):
            raise ValueError("feature_scale must be finite and positive")

    whitened_design = matrix / scale[:, None]
    whitened_data = vector / scale
    parameter_count = int(matrix.shape[1])
    precision = torch.as_tensor(
        prior_precision,
        dtype=torch.float64,
        device=matrix.device,
    ).reshape(-1)
    if precision.numel() == 1:
        precision = precision.expand(parameter_count)
    if precision.numel() != parameter_count:
        raise ValueError("prior_precision must be scalar or one value per parameter")
    if not torch.isfinite(precision).all() or torch.any(precision <= 0.0):
        raise ValueError("prior_precision must be finite and positive")

    hessian = whitened_design.T @ whitened_design
    hessian.diagonal().add_(precision)
    diagonal_scale = torch.clamp(torch.max(torch.diag(hessian)), min=1.0)
    jitter = diagonal_scale * float(hessian_jitter_fraction)
    hessian.diagonal().add_(jitter)
    rhs = whitened_design.T @ whitened_data
    cholesky, info = torch.linalg.cholesky_ex(hessian)
    if int(torch.max(info).cpu()) != 0:
        raise RuntimeError(f"Control Hessian is not positive definite: info={info.tolist()}")
    posterior_mean = torch.cholesky_solve(rhs[:, None], cholesky)[:, 0]
    posterior_covariance = torch.cholesky_inverse(cholesky)
    prediction = matrix @ posterior_mean
    residual = vector - prediction
    whitened_residual = residual / scale
    condition = torch.linalg.cond(hessian)
    stats: dict[str, Any] = {
        "feature_count": int(matrix.shape[0]),
        "parameter_count": parameter_count,
        "feature_scale_min": float(torch.min(scale).cpu()),
        "feature_scale_median": float(torch.median(scale).cpu()),
        "feature_scale_max": float(torch.max(scale).cpu()),
        "hessian_jitter": float(jitter.cpu()),
        "hessian_condition_number": float(condition.cpu()),
        "posterior_mean_norm": float(torch.linalg.vector_norm(posterior_mean).cpu()),
        "posterior_mean_max_abs": float(torch.max(torch.abs(posterior_mean)).cpu()),
        "data_standardized_rms": float(torch.sqrt(torch.mean(whitened_data.square())).cpu()),
        "residual_standardized_rms": float(torch.sqrt(torch.mean(whitened_residual.square())).cpu()),
        "residual_over_data_norm": float(
            (
                torch.linalg.vector_norm(residual)
                / torch.clamp(torch.linalg.vector_norm(vector), min=1.0e-300)
            ).cpu()
        ),
    }
    return LinearGaussianControlResult(
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        feature_scale=scale,
        prediction=prediction,
        residual=residual,
        stats=stats,
    )


def posterior_predictive_score(
    design: torch.Tensor,
    data: torch.Tensor,
    result: LinearGaussianControlResult,
    *,
    feature_scale: torch.Tensor | None = None,
    covariance_chunk_size: int = 8192,
) -> dict[str, float]:
    """Score held-out features under the fitted nuisance posterior."""

    matrix = design.double()
    vector = data.double().reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != vector.numel():
        raise ValueError("design/data feature dimensions differ")
    if matrix.shape[1] != result.posterior_mean.numel():
        raise ValueError("held-out design has the wrong parameter count")
    scale = (
        prior_predictive_feature_scale(matrix)
        if feature_scale is None
        else feature_scale.double().reshape(-1)
    )
    if scale.numel() != vector.numel():
        raise ValueError("feature_scale/data dimensions differ")
    residual = vector - matrix @ result.posterior_mean
    predictive_variance = scale.square().clone()
    chunk = max(int(covariance_chunk_size), 1)
    covariance = result.posterior_covariance
    for start in range(0, int(matrix.shape[0]), chunk):
        stop = min(start + chunk, int(matrix.shape[0]))
        block = matrix[start:stop]
        predictive_variance[start:stop] += torch.sum(
            (block @ covariance) * block,
            dim=1,
        )
    predictive_variance = torch.clamp(predictive_variance, min=1.0e-300)
    standardized_square = residual.square() / predictive_variance
    reference_variance = torch.median(predictive_variance)
    relative_nll = torch.mean(
        standardized_square + torch.log(predictive_variance / reference_variance)
    )
    return {
        "standardized_rms": float(torch.sqrt(torch.mean(standardized_square)).cpu()),
        "mean_standardized_chi2": float(torch.mean(standardized_square).cpu()),
        "relative_gaussian_nll": float(relative_nll.cpu()),
        "residual_over_data_norm": float(
            (
                torch.linalg.vector_norm(residual)
                / torch.clamp(torch.linalg.vector_norm(vector), min=1.0e-300)
            ).cpu()
        ),
        "predictive_std_min": float(torch.sqrt(torch.min(predictive_variance)).cpu()),
        "predictive_std_median": float(torch.sqrt(torch.median(predictive_variance)).cpu()),
        "predictive_std_max": float(torch.sqrt(torch.max(predictive_variance)).cpu()),
    }


def posterior_mean_for_data(
    design: torch.Tensor,
    data: torch.Tensor,
    result: LinearGaussianControlResult,
) -> torch.Tensor:
    """Apply an already factorized linear posterior map to another data vector."""

    matrix = design.double()
    vector = data.double().reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != vector.numel():
        raise ValueError("design/data feature dimensions differ")
    if matrix.shape[1] != result.posterior_mean.numel():
        raise ValueError("design has the wrong parameter count")
    scale = result.feature_scale
    whitened_design = matrix / scale[:, None]
    whitened_data = vector / scale
    rhs = whitened_design.T @ whitened_data
    return result.posterior_covariance @ rhs


def relative_linearity_error(
    total: torch.Tensor,
    components: tuple[torch.Tensor, ...],
) -> float:
    expected = torch.zeros_like(total)
    for component in components:
        expected = expected + component
    denominator = torch.clamp(torch.linalg.vector_norm(total.double()), min=1.0e-300)
    return float(
        (torch.linalg.vector_norm((total - expected).double()) / denominator).cpu()
    )
