#!/usr/bin/env python3
"""Small-matrix covariance marginalization for partial-window EoR PS2D tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HyperState:
    ell_mhz: float
    q_fg: float
    q_eor_k2: float
    log_weight: float
    weight: float


@dataclass
class CovarianceGridFit:
    states: list[HyperState]
    log_evidence: float
    effective_state_count: float
    retained_probability_mass: float
    edge_mass_q_eor: float
    edge_mass_ell: float
    map_state: HyperState
    posterior_mean_ell_mhz: float
    posterior_mean_q_fg: float
    posterior_mean_q_eor_k2: float


def second_moment_covariance(samples: np.ndarray) -> np.ndarray:
    """Return E[x x^H] for rows of zero-mean real or complex samples."""
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("samples must have shape [sample,frequency]")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contain non-finite values")
    covariance = values.T @ values.conj() / float(values.shape[0])
    return np.asarray(0.5 * (covariance + covariance.conj().T))


def regularize_covariance(
    covariance: np.ndarray,
    *,
    diagonal_shrinkage: float,
    eigen_floor_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(covariance, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be square")
    if not 0.0 <= float(diagonal_shrinkage) <= 1.0:
        raise ValueError("diagonal_shrinkage must be in [0,1]")
    if not 0.0 < float(eigen_floor_fraction) <= 1.0:
        raise ValueError("eigen_floor_fraction must be in (0,1]")
    hermitian = 0.5 * (values + values.conj().T)
    diagonal = np.diag(np.real(np.diag(hermitian))).astype(np.complex128)
    shrunk = (
        (1.0 - float(diagonal_shrinkage)) * hermitian
        + float(diagonal_shrinkage) * diagonal
    )
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    maximum = max(float(np.max(eigenvalues)), np.finfo(np.float64).tiny)
    floor = maximum * float(eigen_floor_fraction)
    effective = np.maximum(eigenvalues, floor)
    output = (eigenvectors * effective[None, :]) @ eigenvectors.conj().T
    return np.asarray(0.5 * (output + output.conj().T)), {
        "raw_min_eigenvalue": float(np.min(eigenvalues)),
        "raw_max_eigenvalue": maximum,
        "eigen_floor": floor,
        "condition_number": float(np.max(effective) / np.min(effective)),
        "diagonal_shrinkage": float(diagonal_shrinkage),
    }


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _complex_gaussian_log_likelihood(
    sample_covariance: np.ndarray,
    sample_count: int,
    covariance: np.ndarray,
) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if float(np.real(sign)) <= 0.0 or abs(float(np.imag(sign))) > 1e-8:
        return -math.inf
    try:
        solved = np.linalg.solve(covariance, sample_covariance)
    except np.linalg.LinAlgError:
        return -math.inf
    quadratic = float(sample_count) * float(np.real(np.trace(solved)))
    return -float(sample_count) * float(np.real(logdet)) - quadratic


def fit_covariance_grid(
    samples: np.ndarray,
    foreground_covariance: np.ndarray,
    eor_covariances: Mapping[float, np.ndarray],
    *,
    q_fg_grid: Sequence[float],
    q_eor_k2_grid: Sequence[float],
    q_fg_log_sigma: float = math.log(2.0),
    max_retained_states: int = 64,
    retained_mass: float = 0.9999,
) -> CovarianceGridFit:
    """Marginalize foreground scale, EoR scale, and coherence length on a grid."""
    y = np.asarray(samples, dtype=np.complex128)
    fg = np.asarray(foreground_covariance, dtype=np.complex128)
    if y.ndim != 2 or fg.shape != (y.shape[1], y.shape[1]):
        raise ValueError("sample and foreground covariance dimensions do not match")
    if not eor_covariances:
        raise ValueError("at least one EoR covariance is required")
    ell_values = np.asarray(sorted(float(value) for value in eor_covariances), dtype=np.float64)
    q_fg_values = np.asarray(q_fg_grid, dtype=np.float64)
    q_eor_values = np.asarray(q_eor_k2_grid, dtype=np.float64)
    if (
        np.any(ell_values <= 0.0)
        or np.any(q_fg_values <= 0.0)
        or np.any(q_eor_values <= 0.0)
    ):
        raise ValueError("all covariance-grid values must be positive")
    if float(q_fg_log_sigma) <= 0.0:
        raise ValueError("q_fg_log_sigma must be positive")

    observed_covariance = second_moment_covariance(y)
    records: list[tuple[float, float, float, float]] = []
    for ell in ell_values:
        eor = np.asarray(eor_covariances[float(ell)], dtype=np.complex128)
        if eor.shape != fg.shape:
            raise ValueError("EoR covariance shape mismatch")
        for q_fg in q_fg_values:
            log_prior_fg = -0.5 * (math.log(float(q_fg)) / float(q_fg_log_sigma)) ** 2
            for q_eor in q_eor_values:
                covariance = float(q_fg) * fg + float(q_eor) * eor
                log_like = _complex_gaussian_log_likelihood(
                    observed_covariance, int(y.shape[0]), covariance
                )
                records.append((float(ell), float(q_fg), float(q_eor), log_like + log_prior_fg))
    log_weights = np.asarray([record[3] for record in records], dtype=np.float64)
    if not np.any(np.isfinite(log_weights)):
        raise ValueError("every covariance-grid state has non-finite likelihood")
    log_norm = _logsumexp(log_weights[np.isfinite(log_weights)])
    weights = np.zeros_like(log_weights)
    finite = np.isfinite(log_weights)
    weights[finite] = np.exp(log_weights[finite] - log_norm)
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order])
    keep_count = int(np.searchsorted(cumulative, float(retained_mass), side="left") + 1)
    keep_count = min(max(1, keep_count), int(max_retained_states), len(records))
    keep = order[:keep_count]
    retained_probability_mass = float(np.sum(weights[keep]))
    retained_weights = weights[keep]
    retained_weights /= float(np.sum(retained_weights))
    states = [
        HyperState(
            ell_mhz=records[index][0],
            q_fg=records[index][1],
            q_eor_k2=records[index][2],
            log_weight=records[index][3] - log_norm,
            weight=float(retained_weights[position]),
        )
        for position, index in enumerate(keep)
    ]
    map_index = int(np.argmax(weights))
    map_record = records[map_index]
    map_state = HyperState(
        ell_mhz=map_record[0],
        q_fg=map_record[1],
        q_eor_k2=map_record[2],
        log_weight=map_record[3] - log_norm,
        weight=float(weights[map_index]),
    )
    q_eor_edge = (q_eor_values[0], q_eor_values[-1])
    ell_edge = (ell_values[0], ell_values[-1])
    return CovarianceGridFit(
        states=states,
        log_evidence=float(log_norm),
        effective_state_count=float(1.0 / np.sum(np.square(weights))),
        retained_probability_mass=retained_probability_mass,
        edge_mass_q_eor=float(
            sum(
                weight
                for weight, record in zip(weights, records)
                if record[2] in q_eor_edge
            )
        ),
        edge_mass_ell=float(
            sum(
                weight
                for weight, record in zip(weights, records)
                if record[0] in ell_edge
            )
        ),
        map_state=map_state,
        posterior_mean_ell_mhz=float(
            sum(weight * record[0] for weight, record in zip(weights, records))
        ),
        posterior_mean_q_fg=float(
            sum(weight * record[1] for weight, record in zip(weights, records))
        ),
        posterior_mean_q_eor_k2=float(
            sum(weight * record[2] for weight, record in zip(weights, records))
        ),
    )


def _radial_transform(length: int, window: np.ndarray) -> np.ndarray:
    diagonal = np.diag(np.asarray(window, dtype=np.float64))
    return np.fft.fft(diagonal, axis=0)


def posterior_radial_powers(
    total_samples: np.ndarray,
    foreground_only_samples: np.ndarray,
    eor_only_samples: np.ndarray,
    foreground_covariance: np.ndarray,
    eor_covariances: Mapping[float, np.ndarray],
    fit: CovarianceGridFit,
    *,
    radial_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate hyperparameter-marginalized Wiener powers for each spatial mode."""
    total = np.asarray(total_samples, dtype=np.complex128)
    fg_only = np.asarray(foreground_only_samples, dtype=np.complex128)
    eor_only = np.asarray(eor_only_samples, dtype=np.complex128)
    if total.shape != fg_only.shape or total.shape != eor_only.shape:
        raise ValueError("total, foreground-only, and EoR-only samples must match")
    nfreq = int(total.shape[1])
    window = np.asarray(radial_window, dtype=np.float64)
    if window.shape != (nfreq,):
        raise ValueError("radial window shape mismatch")
    transform = _radial_transform(nfreq, window)
    second_moment = np.zeros((total.shape[0], nfreq), dtype=np.float64)
    mixed_total_mean = np.zeros_like(total)
    mixed_fg_mean = np.zeros_like(total)
    mixed_eor_mean = np.zeros_like(total)

    for state in fit.states:
        ce = float(state.q_eor_k2) * np.asarray(
            eor_covariances[float(state.ell_mhz)], dtype=np.complex128
        )
        covariance = float(state.q_fg) * foreground_covariance + ce
        inverse = np.linalg.inv(covariance)
        gain = ce @ inverse
        total_mean = total @ gain.T
        covariance_post = ce - gain @ ce
        covariance_post = 0.5 * (covariance_post + covariance_post.conj().T)
        radial_variance = np.real(
            np.diag(transform @ covariance_post @ transform.conj().T)
        )
        total_fft = np.fft.fft(total_mean * window[None, :], axis=1)
        second_moment += float(state.weight) * (
            np.square(np.abs(total_fft)) + radial_variance[None, :]
        )
        mixed_total_mean += float(state.weight) * total_mean
        mixed_fg_mean += float(state.weight) * (fg_only @ gain.T)
        mixed_eor_mean += float(state.weight) * (eor_only @ gain.T)

    def mean_power(values: np.ndarray) -> np.ndarray:
        transformed = np.fft.fft(values * window[None, :], axis=1)
        return np.square(np.abs(transformed))

    return {
        "posterior_second_moment": second_moment,
        "posterior_mean": mean_power(mixed_total_mean),
        "foreground_leakage_mean": mean_power(mixed_fg_mean),
        "eor_transfer_mean": mean_power(mixed_eor_mean),
    }


def spatial_fourier_modes(
    cubes: np.ndarray,
    *,
    demean_mode: str,
    spatial_window_y: np.ndarray,
    spatial_window_x: np.ndarray,
) -> np.ndarray:
    """Return spatially tapered FFTs with shape [sample,freq,y,x]."""
    values = np.asarray(cubes, dtype=np.float64)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4:
        raise ValueError("cubes must have shape [freq,y,x] or [sample,freq,y,x]")
    mode = str(demean_mode).strip().lower()
    if mode == "global":
        prepared = values - np.mean(values, axis=(1, 2, 3), keepdims=True)
    elif mode == "per_freq_spatial":
        prepared = values - np.mean(values, axis=(2, 3), keepdims=True)
    elif mode == "none":
        prepared = values
    else:
        raise ValueError("unsupported demean mode")
    window = (
        np.asarray(spatial_window_y, dtype=np.float64)[None, None, :, None]
        * np.asarray(spatial_window_x, dtype=np.float64)[None, None, None, :]
    )
    return np.fft.fft2(prepared * window, axes=(-2, -1))


def independent_spatial_coordinates(height: int, width: int) -> np.ndarray:
    y, x = np.indices((int(height), int(width)), dtype=np.int64)
    linear = np.ravel_multi_index((y, x), (int(height), int(width)))
    conjugate = np.ravel_multi_index(
        ((-y) % int(height), (-x) % int(width)),
        (int(height), int(width)),
    )
    return np.column_stack(np.nonzero(linear <= conjugate)).astype(np.int64)


def fill_conjugate_power_cube(
    output: np.ndarray,
    coordinates: np.ndarray,
    radial_power: np.ndarray,
) -> None:
    """Fill a real-cube 3D power array from independent transverse modes."""
    if output.ndim != 3 or radial_power.shape != (coordinates.shape[0], output.shape[0]):
        raise ValueError("power output and independent-mode values do not match")
    nfreq, height, width = output.shape
    radial_conjugate = (-np.arange(nfreq, dtype=np.int64)) % nfreq
    for index, (y, x) in enumerate(np.asarray(coordinates, dtype=np.int64)):
        partner_y = (-int(y)) % height
        partner_x = (-int(x)) % width
        values = np.asarray(radial_power[index], dtype=np.float64)
        output[:, int(y), int(x)] = values
        if partner_y == int(y) and partner_x == int(x):
            output[:, int(y), int(x)] = 0.5 * (values + values[radial_conjugate])
        else:
            output[:, partner_y, partner_x] = values[radial_conjugate]


def fit_to_dict(fit: CovarianceGridFit) -> dict[str, Any]:
    return {
        "log_evidence": fit.log_evidence,
        "effective_state_count": fit.effective_state_count,
        "retained_probability_mass": fit.retained_probability_mass,
        "edge_mass_q_eor": fit.edge_mass_q_eor,
        "edge_mass_ell": fit.edge_mass_ell,
        "posterior_mean_ell_mhz": fit.posterior_mean_ell_mhz,
        "posterior_mean_q_fg": fit.posterior_mean_q_fg,
        "posterior_mean_q_eor_k2": fit.posterior_mean_q_eor_k2,
        "map_state": {
            "ell_mhz": fit.map_state.ell_mhz,
            "q_fg": fit.map_state.q_fg,
            "q_eor_k2": fit.map_state.q_eor_k2,
            "posterior_weight": fit.map_state.weight,
        },
        "retained_states": [
            {
                "ell_mhz": state.ell_mhz,
                "q_fg": state.q_fg,
                "q_eor_k2": state.q_eor_k2,
                "weight": state.weight,
            }
            for state in fit.states
        ],
    }
