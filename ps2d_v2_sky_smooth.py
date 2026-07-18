"""Template-free sky-smooth nuisance utilities for PS2D screens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


TensorAction = Callable[[torch.Tensor], torch.Tensor]


def orthonormal_chebyshev_spectral_basis(
    frequency_count: int,
    degrees: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return orthonormal Chebyshev columns with shape [frequency, basis]."""
    count = int(frequency_count)
    parsed = tuple(int(value) for value in degrees)
    if count < 2:
        raise ValueError("At least two frequencies are required")
    if not parsed or any(value < 0 for value in parsed):
        raise ValueError("Chebyshev degrees must be a non-empty non-negative list")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Chebyshev degrees must be unique")
    if len(parsed) > count:
        raise ValueError("Spectral basis cannot have more columns than frequencies")
    coordinate = torch.linspace(-1.0, 1.0, count, dtype=dtype, device=device)
    angle = torch.acos(torch.clamp(coordinate, -1.0, 1.0))
    raw = torch.stack(
        [torch.cos(float(degree) * angle) for degree in parsed], dim=1
    )
    basis, triangular = torch.linalg.qr(raw, mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    threshold = torch.finfo(dtype).eps * max(count, len(parsed))
    if bool(torch.any(diagonal <= threshold)):
        raise ValueError("Chebyshev spectral columns are numerically rank deficient")
    return basis


def compose_sky_smooth_cube(
    coefficient_maps: torch.Tensor,
    spectral_basis: torch.Tensor,
) -> torch.Tensor:
    """Combine free spatial coefficient maps with a shared smooth spectral basis."""
    if spectral_basis.ndim != 2:
        raise ValueError("Spectral basis must have shape [frequency,basis]")
    if coefficient_maps.ndim == 3:
        if coefficient_maps.shape[0] != spectral_basis.shape[1]:
            raise ValueError("Coefficient and spectral basis counts differ")
        return torch.einsum("fb,byx->fyx", spectral_basis, coefficient_maps)
    if coefficient_maps.ndim == 4:
        if coefficient_maps.shape[1] != spectral_basis.shape[1]:
            raise ValueError("Coefficient and spectral basis counts differ")
        return torch.einsum("fb,nbyx->nfyx", spectral_basis, coefficient_maps)
    raise ValueError(
        "Coefficient maps must have shape [basis,y,x] or [batch,basis,y,x]"
    )


def relative_adjoint_error(
    forward: TensorAction,
    adjoint: TensorAction,
    domain_probe: torch.Tensor,
    range_probe: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the normalized dot-product closure of a linear action pair."""
    left = torch.sum(forward(domain_probe).double() * range_probe.double())
    right = torch.sum(domain_probe.double() * adjoint(range_probe).double())
    denominator = torch.clamp(
        torch.maximum(torch.abs(left), torch.abs(right)), min=1e-300
    )
    return {
        "forward_dot": float(left.detach().cpu()),
        "adjoint_dot": float(right.detach().cpu()),
        "relative_error": float((torch.abs(left - right) / denominator).detach().cpu()),
    }


@dataclass
class MatrixFreeLeastSquaresResult:
    solution: torch.Tensor
    stats: dict[str, Any]


@dataclass
class AugmentedRidgeSystem:
    forward: TensorAction
    adjoint: TensorAction
    rhs: torch.Tensor
    data_size: int
    coefficient_shape: tuple[int, ...]


def build_augmented_ridge_system(
    forward: TensorAction,
    adjoint: TensorAction,
    rhs: torch.Tensor,
    coefficient_template: torch.Tensor,
    prior_precision_sqrt: torch.Tensor,
) -> AugmentedRidgeSystem:
    """Build [A; L] x = [b; 0] for a finite Gaussian coefficient prior."""
    if rhs.is_complex():
        raise ValueError("Augmented ridge RHS must use an explicit real representation")
    if coefficient_template.is_complex():
        raise ValueError("Coefficient template must be real")
    try:
        precision = torch.broadcast_to(
            prior_precision_sqrt,
            coefficient_template.shape,
        ).detach()
    except RuntimeError as error:
        raise ValueError("Prior precision is not broadcastable to coefficients") from error
    if not bool(torch.all(torch.isfinite(precision))):
        raise ValueError("Prior precision must be finite")
    if not bool(torch.all(precision > 0.0)):
        raise ValueError("Finite-covariance prior precision must be positive")

    data_shape = tuple(int(value) for value in rhs.shape)
    coefficient_shape = tuple(int(value) for value in coefficient_template.shape)
    data_size = int(rhs.numel())
    coefficient_size = int(coefficient_template.numel())
    augmented_rhs = torch.cat(
        (
            rhs.detach().reshape(-1),
            torch.zeros(
                (coefficient_size,),
                dtype=rhs.dtype,
                device=rhs.device,
            ),
        )
    )

    def augmented_forward(coefficients: torch.Tensor) -> torch.Tensor:
        if tuple(coefficients.shape) != coefficient_shape:
            raise ValueError("Coefficient shape differs from the ridge system")
        data = forward(coefficients)
        if tuple(data.shape) != data_shape:
            raise ValueError("Forward output shape differs from the ridge RHS")
        return torch.cat(
            (data.reshape(-1), (precision * coefficients).reshape(-1))
        )

    def augmented_adjoint(values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1)
        if int(flat.numel()) != data_size + coefficient_size:
            raise ValueError("Augmented residual has the wrong size")
        data_values = flat[:data_size].reshape(data_shape)
        prior_values = flat[data_size:].reshape(coefficient_shape)
        return adjoint(data_values) + precision * prior_values

    return AugmentedRidgeSystem(
        forward=augmented_forward,
        adjoint=augmented_adjoint,
        rhs=augmented_rhs,
        data_size=data_size,
        coefficient_shape=coefficient_shape,
    )


def matrix_free_lsqr(
    forward: TensorAction,
    adjoint: TensorAction,
    rhs: torch.Tensor,
    *,
    max_iters: int,
    relative_residual_tolerance: float,
    absolute_residual_tolerance: float = 0.0,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> MatrixFreeLeastSquaresResult:
    """Solve min ||A x - b|| with LSQR and an implicit linear operator."""
    if int(max_iters) <= 0:
        raise ValueError("LSQR iteration count must be positive")
    if float(relative_residual_tolerance) < 0.0:
        raise ValueError("Relative residual tolerance must be non-negative")
    if float(absolute_residual_tolerance) < 0.0:
        raise ValueError("Absolute residual tolerance must be non-negative")

    u = rhs.detach().clone()
    beta = torch.linalg.vector_norm(u)
    rhs_norm = float(beta.detach().double().cpu())
    target = max(
        float(absolute_residual_tolerance),
        float(relative_residual_tolerance) * rhs_norm,
    )
    if rhs_norm == 0.0:
        zero_solution = adjoint(rhs).detach()
        zero_solution.zero_()
        return MatrixFreeLeastSquaresResult(
            solution=zero_solution,
            stats={
                "iterations": 0,
                "converged": True,
                "reason": "zero_rhs",
                "rhs_norm": 0.0,
                "target_residual_norm": target,
                "estimated_residual_norm": 0.0,
                "actual_residual_norm": 0.0,
                "actual_relative_residual": 0.0,
                "solution_norm": 0.0,
                "history": [],
            },
        )

    u = u / beta
    v = adjoint(u).detach()
    alpha = torch.linalg.vector_norm(v)
    alpha_float = float(alpha.detach().double().cpu())
    if alpha_float == 0.0:
        solution = v.clone()
        residual_norm = float(torch.linalg.vector_norm(rhs.double()).cpu())
        return MatrixFreeLeastSquaresResult(
            solution=solution,
            stats={
                "iterations": 0,
                "converged": residual_norm <= target,
                "reason": "zero_adjoint_rhs",
                "rhs_norm": rhs_norm,
                "target_residual_norm": target,
                "estimated_residual_norm": residual_norm,
                "actual_residual_norm": residual_norm,
                "actual_relative_residual": residual_norm / rhs_norm,
                "solution_norm": 0.0,
                "history": [],
            },
        )
    v = v / alpha
    solution = torch.zeros_like(v)
    direction = v.clone()
    phi_bar = beta
    rho_bar = alpha
    history: list[dict[str, float | int]] = []
    converged = False
    reason = "max_iters"
    iterations = 0

    for iteration in range(int(max_iters)):
        u_next = forward(v).detach() - alpha * u
        beta = torch.linalg.vector_norm(u_next)
        beta_float = float(beta.detach().double().cpu())
        if beta_float > 0.0:
            u_next = u_next / beta

        v_next = adjoint(u_next).detach() - beta * v
        alpha = torch.linalg.vector_norm(v_next)
        alpha_float = float(alpha.detach().double().cpu())
        if alpha_float > 0.0:
            v_next = v_next / alpha

        rho = torch.sqrt(rho_bar.square() + beta.square())
        rho_float = float(rho.detach().double().cpu())
        if rho_float == 0.0 or not torch.isfinite(rho):
            reason = "invalid_rotation"
            break
        cosine = rho_bar / rho
        sine = beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        solution = (solution + (phi / rho) * direction).detach()
        direction = (v_next - (theta / rho) * direction).detach()
        u = u_next
        v = v_next
        iterations = iteration + 1
        estimated_residual = abs(float(phi_bar.detach().double().cpu()))
        record: dict[str, float | int] = {
            "iteration": iterations,
            "estimated_residual_norm": estimated_residual,
            "estimated_relative_residual": estimated_residual / rhs_norm,
            "alpha": alpha_float,
            "beta": beta_float,
            "solution_norm": float(
                torch.linalg.vector_norm(solution.detach().double()).cpu()
            ),
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
        if not torch.isfinite(solution).all():
            reason = "nonfinite_solution"
            break
        if estimated_residual <= target:
            converged = True
            reason = "residual_tolerance"
            break
        if alpha_float == 0.0 and beta_float == 0.0:
            converged = True
            reason = "exact_bidiagonalization"
            break

    actual_residual = forward(solution).detach() - rhs
    actual_residual_norm = float(
        torch.linalg.vector_norm(actual_residual.double()).cpu()
    )
    return MatrixFreeLeastSquaresResult(
        solution=solution,
        stats={
            "iterations": int(iterations),
            "converged": bool(converged),
            "reason": reason,
            "rhs_norm": rhs_norm,
            "target_residual_norm": target,
            "estimated_residual_norm": abs(
                float(phi_bar.detach().double().cpu())
            ),
            "actual_residual_norm": actual_residual_norm,
            "actual_relative_residual": actual_residual_norm / rhs_norm,
            "solution_norm": float(
                torch.linalg.vector_norm(solution.detach().double()).cpu()
            ),
            "history": history,
        },
    )
