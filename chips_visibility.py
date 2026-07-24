#!/usr/bin/env python3
"""Small visibility-domain building blocks for a CHIPS-like PS2D screen.

The module deliberately keeps the cosmological normalization outside the
quadratic estimator.  It estimates delay-band powers in observable visibility
units and exposes the complete response/window matrix needed for a later
OSKAR-calibrated conversion to sky bandpowers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal.windows import dpss


@dataclass(frozen=True)
class DpssBasis:
    vectors: np.ndarray
    eigenvalues: np.ndarray
    time_half_bandwidth: float
    max_delay_s: float

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True)
class QuadraticResponse:
    frequencies_hz: np.ndarray
    delays_s: np.ndarray
    analysis_matrix: np.ndarray
    fisher: np.ndarray
    window: np.ndarray
    row_normalization: np.ndarray
    foreground_rank: int
    dpss_eigenvalues: np.ndarray
    max_delay_s: float
    suppression_strength: float
    taper: np.ndarray
    foreground_basis: str = "dpss"
    polynomial_degree: Optional[int] = None

    @property
    def supported(self) -> np.ndarray:
        return self.row_normalization > np.finfo(np.float64).eps


def frequency_fourier_basis(
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a unitary frequency/delay basis for a uniform frequency grid."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if frequencies.size < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("At least two finite frequencies are required")
    spacings = np.diff(frequencies)
    spacing = float(np.median(spacings))
    if spacing <= 0.0 or not np.allclose(
        spacings, spacing, rtol=1e-8, atol=max(1e-6, abs(spacing) * 1e-10)
    ):
        raise ValueError("The visibility pilot currently requires uniform frequencies")
    delays = np.fft.fftfreq(frequencies.size, d=spacing)
    centered = frequencies - float(frequencies[0])
    basis = np.exp(
        2j * math.pi * centered[:, None] * delays[None, :]
    ) / math.sqrt(float(frequencies.size))
    return np.asarray(basis, dtype=np.complex128), delays


def weighted_lssa_matrix(
    frequencies_hz: np.ndarray,
    weights: np.ndarray,
    *,
    delays_s: Optional[np.ndarray] = None,
    ridge_fraction: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the weighted least-squares spectral-analysis matrix.

    The returned matrix maps frequency samples to coefficients in the supplied
    delay basis.  Missing channels are represented by zero weights.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if sample_weights.shape != frequencies.shape:
        raise ValueError("LSSA weights and frequencies must have the same shape")
    if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
        raise ValueError("LSSA weights must be finite and non-negative")
    if not np.any(sample_weights > 0.0):
        raise ValueError("LSSA requires at least one positive channel weight")

    default_basis, default_delays = frequency_fourier_basis(frequencies)
    if delays_s is None:
        basis = default_basis
        delays = default_delays
    else:
        delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)
        if delays.size == 0 or not np.all(np.isfinite(delays)):
            raise ValueError("LSSA delays must be finite and non-empty")
        centered = frequencies - float(frequencies[0])
        basis = np.exp(2j * math.pi * centered[:, None] * delays[None, :])
        basis /= math.sqrt(float(frequencies.size))

    weighted_basis = sample_weights[:, None] * basis
    gram = basis.conj().T @ weighted_basis
    scale = max(float(np.real(np.trace(gram))) / max(1, gram.shape[0]), 1.0)
    ridge = max(0.0, float(ridge_fraction)) * scale
    normal = gram + ridge * np.eye(gram.shape[0], dtype=np.complex128)
    rhs = basis.conj().T * sample_weights[None, :]
    try:
        transform = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        transform = np.linalg.pinv(normal, rcond=max(float(ridge_fraction), 1e-12)) @ rhs
    return np.asarray(transform, dtype=np.complex128), delays


def dpss_foreground_basis(
    frequencies_hz: np.ndarray,
    max_delay_s: float,
    *,
    eigenvalue_threshold: float = 1e-6,
) -> DpssBasis:
    """Return the DPSS subspace confined to ``|delay| <= max_delay_s``."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    _, _ = frequency_fourier_basis(frequencies)
    delay = float(max_delay_s)
    threshold = float(eigenvalue_threshold)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("max_delay_s must be finite and non-negative")
    if not 0.0 < threshold < 1.0:
        raise ValueError("DPSS eigenvalue threshold must lie strictly between zero and one")

    count = int(frequencies.size)
    spacing = float(np.median(np.diff(frequencies)))
    nyquist_delay = 0.5 / spacing
    if delay >= nyquist_delay * (1.0 - 1e-12):
        return DpssBasis(
            vectors=np.eye(count, dtype=np.float64),
            eigenvalues=np.ones(count, dtype=np.float64),
            time_half_bandwidth=0.5 * count,
            max_delay_s=delay,
        )
    if delay == 0.0:
        return DpssBasis(
            vectors=np.ones((count, 1), dtype=np.float64) / math.sqrt(float(count)),
            eigenvalues=np.ones(1, dtype=np.float64),
            time_half_bandwidth=0.0,
            max_delay_s=delay,
        )

    half_bandwidth = max(float(count) * spacing * delay, 1e-10)
    # Scipy requires NW < M/2.  The full-rank case was handled above.
    half_bandwidth = min(half_bandwidth, 0.5 * count * (1.0 - 1e-12))
    sequences, ratios = dpss(
        count,
        half_bandwidth,
        Kmax=count,
        sym=False,
        norm=2,
        return_ratios=True,
    )
    vectors = np.asarray(sequences, dtype=np.float64).T
    eigenvalues = np.asarray(ratios, dtype=np.float64)
    keep = eigenvalues >= threshold
    if not np.any(keep):
        keep[0] = True
    selected = vectors[:, keep]
    selected, _ = np.linalg.qr(selected, mode="reduced")
    return DpssBasis(
        vectors=np.asarray(selected, dtype=np.float64),
        eigenvalues=eigenvalues[keep],
        time_half_bandwidth=half_bandwidth,
        max_delay_s=delay,
    )


def chebyshev_foreground_basis(
    frequencies_hz: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Return an orthonormal Chebyshev nuisance basis through ``degree``."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    _, _ = frequency_fourier_basis(frequencies)
    polynomial_degree = int(degree)
    if polynomial_degree < 0:
        raise ValueError("Chebyshev degree must be non-negative")
    if polynomial_degree >= frequencies.size:
        raise ValueError("Chebyshev degree must be smaller than the channel count")
    span = float(frequencies[-1] - frequencies[0])
    if span <= 0.0:
        raise ValueError("Chebyshev frequencies must span a positive bandwidth")
    coordinate = 2.0 * (frequencies - frequencies[0]) / span - 1.0
    vandermonde = np.polynomial.chebyshev.chebvander(
        coordinate, polynomial_degree
    )
    vectors, _ = np.linalg.qr(vandermonde, mode="reduced")
    return np.asarray(vectors, dtype=np.float64)


def inverse_covariance_suppression(
    basis: np.ndarray,
    strength: float,
    *,
    eigenvalues: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an inverse nuisance covariance or its hard-projector limit."""
    vectors = np.asarray(basis, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError("Foreground basis must be a matrix")
    size = int(vectors.shape[0])
    identity = np.eye(size, dtype=np.complex128)
    if vectors.shape[1] == 0 or float(strength) == 0.0:
        return identity
    gram = vectors.conj().T @ vectors
    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=1e-9, atol=1e-10):
        vectors, _ = np.linalg.qr(vectors, mode="reduced")
    if math.isinf(float(strength)):
        coefficients = np.ones(vectors.shape[1], dtype=np.float64)
    else:
        value = float(strength)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Suppression strength must be non-negative or infinity")
        if eigenvalues is None:
            nuisance_scales = np.ones(vectors.shape[1], dtype=np.float64)
        else:
            nuisance_scales = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
            if nuisance_scales.shape != (vectors.shape[1],):
                raise ValueError("DPSS eigenvalues and basis rank differ")
            if np.any(~np.isfinite(nuisance_scales)) or np.any(
                nuisance_scales < 0.0
            ):
                raise ValueError("DPSS eigenvalues must be finite and non-negative")
        scaled = value * nuisance_scales
        coefficients = np.divide(
            scaled,
            1.0 + scaled,
            out=np.ones_like(scaled),
            where=np.isfinite(scaled),
        )
    result = identity - (vectors * coefficients[None, :]) @ vectors.conj().T
    return 0.5 * (result + result.conj().T)


def _build_quadratic_response_from_basis(
    frequencies_hz: np.ndarray,
    *,
    foreground_vectors: np.ndarray,
    foreground_eigenvalues: Optional[np.ndarray],
    foreground_basis: str,
    max_delay_s: float,
    polynomial_degree: Optional[int],
    suppression_strength: float,
    taper: str = "hann",
) -> QuadraticResponse:
    """Build a delay-bandpower response from an explicit nuisance basis."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    fourier, delays = frequency_fourier_basis(frequencies)
    inverse = inverse_covariance_suppression(
        foreground_vectors,
        float(suppression_strength),
        eigenvalues=foreground_eigenvalues,
    )
    taper_name = str(taper).strip().lower()
    if taper_name == "none":
        taper_values = np.ones(frequencies.size, dtype=np.float64)
    elif taper_name == "hann":
        taper_values = np.hanning(frequencies.size)
    elif taper_name == "blackman_harris":
        from scipy.signal.windows import blackmanharris

        taper_values = blackmanharris(frequencies.size, sym=True)
    else:
        raise ValueError("taper must be one of: none, hann, blackman_harris")
    taper_matrix = np.diag(taper_values.astype(np.complex128))

    # This is the generalized spectral transform inside the quadratic
    # estimator.  For zero suppression and no taper it is exactly the DFT.
    analysis = fourier.conj().T @ taper_matrix @ inverse
    mixing = analysis @ fourier
    fisher = np.square(np.abs(mixing))
    row_normalization = np.sum(fisher, axis=1)
    window = np.zeros_like(fisher, dtype=np.float64)
    supported = row_normalization > np.finfo(np.float64).eps
    window[supported] = fisher[supported] / row_normalization[supported, None]
    return QuadraticResponse(
        frequencies_hz=frequencies,
        delays_s=delays,
        analysis_matrix=np.asarray(analysis, dtype=np.complex128),
        fisher=np.asarray(fisher, dtype=np.float64),
        window=window,
        row_normalization=np.asarray(row_normalization, dtype=np.float64),
        foreground_rank=int(foreground_vectors.shape[1]),
        dpss_eigenvalues=(
            np.asarray(foreground_eigenvalues, dtype=np.float64)
            if foreground_eigenvalues is not None
            else np.empty(0, dtype=np.float64)
        ),
        max_delay_s=float(max_delay_s),
        suppression_strength=float(suppression_strength),
        taper=np.asarray(taper_values, dtype=np.float64),
        foreground_basis=str(foreground_basis),
        polynomial_degree=polynomial_degree,
    )


def build_quadratic_response(
    frequencies_hz: np.ndarray,
    *,
    max_delay_s: float,
    suppression_strength: float,
    dpss_eigenvalue_threshold: float = 1e-6,
    taper: str = "hann",
) -> QuadraticResponse:
    """Build a DPSS inverse-covariance weighted delay-bandpower response."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    foreground = dpss_foreground_basis(
        frequencies,
        max_delay_s,
        eigenvalue_threshold=dpss_eigenvalue_threshold,
    )
    return _build_quadratic_response_from_basis(
        frequencies,
        foreground_vectors=foreground.vectors,
        foreground_eigenvalues=foreground.eigenvalues,
        foreground_basis="dpss",
        max_delay_s=float(max_delay_s),
        polynomial_degree=None,
        suppression_strength=float(suppression_strength),
        taper=taper,
    )


def build_chebyshev_quadratic_response(
    frequencies_hz: np.ndarray,
    *,
    degree: int,
    suppression_strength: float,
    taper: str = "hann",
) -> QuadraticResponse:
    """Build a Chebyshev-nuisance weighted delay-bandpower response."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    foreground = chebyshev_foreground_basis(frequencies, int(degree))
    return _build_quadratic_response_from_basis(
        frequencies,
        foreground_vectors=foreground,
        foreground_eigenvalues=None,
        foreground_basis="chebyshev",
        max_delay_s=math.nan,
        polynomial_degree=int(degree),
        suppression_strength=float(suppression_strength),
        taper=taper,
    )


def cross_quadratic_bandpowers(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    response: QuadraticResponse,
    *,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate row-normalized cross-bandpowers and the unnormalized score."""
    first = np.asarray(samples_a, dtype=np.complex128)
    second = np.asarray(samples_b, dtype=np.complex128)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("Cross samples must have identical [sample,frequency] shapes")
    if first.shape[1] != response.frequencies_hz.size:
        raise ValueError("Sample frequency count does not match the response")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Cross samples must be finite")
    if sample_weights is None:
        weights = np.ones(first.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (first.shape[0],):
            raise ValueError("Sample weights have the wrong shape")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Sample weights must be finite and non-negative")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("At least one sample must have positive weight")

    transformed_a = first @ response.analysis_matrix.T
    transformed_b = second @ response.analysis_matrix.T
    per_sample = np.real(np.conjugate(transformed_a) * transformed_b)
    score = np.sum(weights[:, None] * per_sample, axis=0) / total_weight
    estimate = np.full(score.shape, np.nan, dtype=np.float64)
    estimate[response.supported] = (
        score[response.supported] / response.row_normalization[response.supported]
    )
    return estimate, np.asarray(score, dtype=np.float64)


def fold_absolute_delay(
    values: np.ndarray,
    delays_s: np.ndarray,
    *,
    exclude_nyquist: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold signed delay modes into unique absolute-delay groups."""
    data = np.asarray(values)
    delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)
    if data.shape[-1] != delays.size:
        raise ValueError("Delay axis length does not match values")
    absolute = np.abs(delays)
    unique, groups = np.unique(absolute, return_inverse=True)
    keep = np.ones(unique.size, dtype=bool)
    if exclude_nyquist and delays.size % 2 == 0:
        keep &= ~np.isclose(
            unique,
            abs(float(delays[delays.size // 2])),
            rtol=1e-12,
            atol=1e-18,
        )
    remap = np.full(unique.size, -1, dtype=np.int64)
    remap[keep] = np.arange(int(np.count_nonzero(keep)), dtype=np.int64)
    output_shape = data.shape[:-1] + (int(np.count_nonzero(keep)),)
    output = np.zeros(output_shape, dtype=data.dtype)
    counts = np.zeros(int(np.count_nonzero(keep)), dtype=np.int64)
    for source, target in enumerate(remap[groups]):
        if target < 0:
            continue
        output[..., target] += data[..., source]
        counts[target] += 1
    output /= counts.reshape((1,) * (output.ndim - 1) + (-1,))
    return output, unique[keep], counts


def fold_window_absolute_delay(
    window: np.ndarray,
    delays_s: np.ndarray,
    *,
    exclude_nyquist: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold both axes of a signed-delay response window."""
    matrix = np.asarray(window, dtype=np.float64)
    delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)
    if matrix.shape != (delays.size, delays.size):
        raise ValueError("Window must be square on the delay axis")
    absolute = np.abs(delays)
    unique, groups = np.unique(absolute, return_inverse=True)
    keep = np.ones(unique.size, dtype=bool)
    if exclude_nyquist and delays.size % 2 == 0:
        keep &= ~np.isclose(
            unique,
            abs(float(delays[delays.size // 2])),
            rtol=1e-12,
            atol=1e-18,
        )
    kept = np.flatnonzero(keep)
    result = np.zeros((kept.size, kept.size), dtype=np.float64)
    for row_out, row_group in enumerate(kept):
        rows = np.flatnonzero(groups == row_group)
        for col_out, col_group in enumerate(kept):
            cols = np.flatnonzero(groups == col_group)
            # A folded source band assigns the same power to both signed-delay
            # members, so source columns add while output rows average.
            result[row_out, col_out] = float(
                np.mean(np.sum(matrix[np.ix_(rows, cols)], axis=1))
            )
    row_sum = np.sum(result, axis=1)
    valid = row_sum > np.finfo(np.float64).eps
    result[valid] /= row_sum[valid, None]
    return result, unique[keep]
