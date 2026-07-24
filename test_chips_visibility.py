#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from chips_visibility import (
    build_chebyshev_quadratic_response,
    build_quadratic_response,
    chebyshev_foreground_basis,
    cross_quadratic_bandpowers,
    dpss_foreground_basis,
    fold_absolute_delay,
    fold_window_absolute_delay,
    frequency_fourier_basis,
    weighted_lssa_matrix,
)


def _frequencies(count: int = 32) -> np.ndarray:
    return 117.9e6 + np.arange(count, dtype=np.float64) * 0.1e6


def test_frequency_basis_is_unitary_and_lssa_closes() -> None:
    frequencies = _frequencies()
    basis, delays = frequency_fourier_basis(frequencies)
    np.testing.assert_allclose(
        basis.conj().T @ basis,
        np.eye(frequencies.size),
        rtol=1e-12,
        atol=1e-12,
    )
    transform, actual_delays = weighted_lssa_matrix(
        frequencies, np.ones(frequencies.size)
    )
    np.testing.assert_allclose(actual_delays, delays, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        transform @ basis,
        np.eye(frequencies.size),
        rtol=1e-10,
        atol=1e-10,
    )


def test_lssa_missing_channels_recovers_supported_model() -> None:
    frequencies = _frequencies(16)
    basis, delays = frequency_fourier_basis(frequencies)
    selected = np.asarray([1, 4, 7], dtype=np.int64)
    coefficients = np.asarray([2.0 + 1.0j, -0.5j, 0.25 - 0.2j])
    samples = basis[:, selected] @ coefficients
    weights = np.ones(frequencies.size)
    weights[[2, 9, 13]] = 0.0
    transform, _ = weighted_lssa_matrix(
        frequencies,
        weights,
        delays_s=delays[selected],
    )
    recovered = transform @ samples
    np.testing.assert_allclose(recovered, coefficients, rtol=1e-9, atol=1e-9)


def test_dpss_rank_increases_with_geometric_delay() -> None:
    frequencies = _frequencies()
    short = dpss_foreground_basis(
        frequencies, 0.4e-6, eigenvalue_threshold=1e-6
    )
    long = dpss_foreground_basis(
        frequencies, 4.0e-6, eigenvalue_threshold=1e-6
    )
    assert 0 < short.rank < long.rank < frequencies.size
    np.testing.assert_allclose(
        long.vectors.T @ long.vectors,
        np.eye(long.rank),
        rtol=1e-10,
        atol=1e-10,
    )


def test_chebyshev_basis_is_orthonormal_and_annihilated() -> None:
    frequencies = _frequencies()
    nuisance = chebyshev_foreground_basis(frequencies, degree=3)
    np.testing.assert_allclose(
        nuisance.T @ nuisance,
        np.eye(4),
        rtol=1e-12,
        atol=1e-12,
    )
    response = build_chebyshev_quadratic_response(
        frequencies,
        degree=3,
        suppression_strength=np.inf,
        taper="none",
    )
    np.testing.assert_allclose(
        response.analysis_matrix @ nuisance,
        np.zeros((frequencies.size, nuisance.shape[1])),
        rtol=0.0,
        atol=2e-14,
    )
    assert response.foreground_basis == "chebyshev"
    assert response.foreground_rank == 4
    assert response.polynomial_degree == 3


def test_chebyshev_response_matches_monte_carlo_window() -> None:
    rng = np.random.default_rng(2407)
    frequencies = _frequencies(16)
    basis, _ = frequency_fourier_basis(frequencies)
    response = build_chebyshev_quadratic_response(
        frequencies,
        degree=3,
        suppression_strength=np.inf,
        taper="hann",
    )
    true_power = np.linspace(0.5, 2.0, frequencies.size)
    coefficients = (
        rng.normal(size=(40000, frequencies.size))
        + 1j * rng.normal(size=(40000, frequencies.size))
    ) * np.sqrt(true_power[None, :] / 2.0)
    samples = coefficients @ basis.T
    estimate, _ = cross_quadratic_bandpowers(samples, samples, response)
    expected = response.window @ true_power
    np.testing.assert_allclose(
        estimate[response.supported],
        expected[response.supported],
        rtol=0.035,
        atol=0.035,
    )


def test_quadratic_response_matches_monte_carlo_window() -> None:
    rng = np.random.default_rng(7023)
    frequencies = _frequencies(16)
    basis, _ = frequency_fourier_basis(frequencies)
    response = build_quadratic_response(
        frequencies,
        max_delay_s=0.8e-6,
        suppression_strength=1e4,
        dpss_eigenvalue_threshold=1e-6,
        taper="hann",
    )
    true_power = np.linspace(0.5, 2.0, frequencies.size)
    count = 40000
    coefficients = (
        rng.normal(size=(count, frequencies.size))
        + 1j * rng.normal(size=(count, frequencies.size))
    ) * np.sqrt(true_power[None, :] / 2.0)
    samples = coefficients @ basis.T
    estimate, _ = cross_quadratic_bandpowers(samples, samples, response)
    expected = response.window @ true_power
    np.testing.assert_allclose(
        estimate[response.supported],
        expected[response.supported],
        rtol=0.035,
        atol=0.035,
    )


def test_dpss_suppression_reduces_smooth_foreground() -> None:
    rng = np.random.default_rng(17)
    frequencies = _frequencies()
    basis, delays = frequency_fourier_basis(frequencies)
    foreground = (
        1000.0
        * (1.0 + 0.02 * np.linspace(-1.0, 1.0, frequencies.size))[None, :]
        * np.exp(1j * rng.uniform(-0.1, 0.1, size=(128, 1)))
    )
    eor_coefficients = (
        rng.normal(size=(128, frequencies.size))
        + 1j * rng.normal(size=(128, frequencies.size))
    ) / np.sqrt(2.0)
    eor = eor_coefficients @ basis.T
    raw = build_quadratic_response(
        frequencies,
        max_delay_s=0.5e-6,
        suppression_strength=0.0,
        taper="hann",
    )
    suppressed = build_quadratic_response(
        frequencies,
        max_delay_s=0.5e-6,
        suppression_strength=np.inf,
        taper="hann",
    )
    raw_fg, _ = cross_quadratic_bandpowers(foreground, foreground, raw)
    clean_fg, _ = cross_quadratic_bandpowers(
        foreground, foreground, suppressed
    )
    clean_eor, _ = cross_quadratic_bandpowers(eor, eor, suppressed)
    folded_fg, folded_delay, _ = fold_absolute_delay(clean_fg, delays)
    folded_eor, _, _ = fold_absolute_delay(clean_eor, delays)
    raw_folded, _, _ = fold_absolute_delay(raw_fg, delays)
    high = folded_delay > 1.5e-6
    assert np.nanmean(np.abs(folded_fg[high])) < 1e-4 * np.nanmean(
        np.abs(raw_folded[high])
    )
    assert np.nanmean(folded_eor[high]) > 0.1


def test_folded_window_preserves_flat_bandpower() -> None:
    frequencies = _frequencies()
    response = build_quadratic_response(
        frequencies,
        max_delay_s=0.7e-6,
        suppression_strength=1e4,
        taper="hann",
    )
    folded, _ = fold_window_absolute_delay(
        response.window, response.delays_s
    )
    supported = np.sum(folded, axis=1) > 0.0
    np.testing.assert_allclose(
        np.sum(folded[supported], axis=1),
        np.ones(np.count_nonzero(supported)),
        rtol=1e-12,
        atol=1e-12,
    )
