from __future__ import annotations

import numpy as np

from covariance_partial_window import (
    fill_conjugate_power_cube,
    fit_covariance_grid,
    independent_spatial_coordinates,
    posterior_radial_powers,
    regularize_covariance,
    second_moment_covariance,
)


def _complex_samples(covariance: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(covariance)
    white = (
        rng.normal(size=(count, covariance.shape[0]))
        + 1j * rng.normal(size=(count, covariance.shape[0]))
    ) / np.sqrt(2.0)
    return white @ factor.T


def test_grid_fit_recovers_distinct_short_coherence_amplitude() -> None:
    nfreq = 8
    distance = np.abs(np.arange(nfreq)[:, None] - np.arange(nfreq)[None, :])
    foreground = 4.0 * np.exp(-distance / 20.0) + 0.1 * np.eye(nfreq)
    eor_short = np.exp(-distance / 0.7)
    eor_long = np.exp(-distance / 3.0)
    truth_q = 0.2
    samples = _complex_samples(foreground + truth_q * eor_short, 12000, 7)
    fit = fit_covariance_grid(
        samples,
        foreground,
        {0.7: eor_short, 3.0: eor_long},
        q_fg_grid=[0.8, 1.0, 1.2],
        q_eor_k2_grid=[0.05, 0.1, 0.2, 0.4, 0.8],
    )
    assert fit.map_state.ell_mhz == 0.7
    assert fit.map_state.q_eor_k2 == truth_q
    assert fit.edge_mass_q_eor < 0.05


def test_posterior_second_moment_exceeds_mean_power() -> None:
    nfreq = 6
    foreground = 3.0 * np.ones((nfreq, nfreq)) + 0.2 * np.eye(nfreq)
    eor = np.eye(nfreq)
    total = _complex_samples(foreground + 0.3 * eor, 64, 3)
    fg = _complex_samples(foreground, 64, 4)
    signal = total - fg
    fit = fit_covariance_grid(
        total,
        foreground,
        {0.2: eor},
        q_fg_grid=[1.0],
        q_eor_k2_grid=[0.3],
    )
    powers = posterior_radial_powers(
        total,
        fg,
        signal,
        foreground,
        {0.2: eor},
        fit,
        radial_window=np.hanning(nfreq),
    )
    assert np.all(
        powers["posterior_second_moment"]
        >= powers["posterior_mean"] - 1e-12
    )


def test_covariance_regularization_and_conjugate_fill() -> None:
    samples = np.asarray([[1.0, 1.0], [2.0, 2.0]])
    covariance = second_moment_covariance(samples)
    regularized, stats = regularize_covariance(
        covariance, diagonal_shrinkage=0.1, eigen_floor_fraction=1e-8
    )
    assert np.min(np.linalg.eigvalsh(regularized)) > 0.0
    assert stats["condition_number"] > 1.0

    coordinates = independent_spatial_coordinates(3, 4)
    output = np.zeros((5, 3, 4), dtype=np.float64)
    values = np.arange(coordinates.shape[0] * 5, dtype=np.float64).reshape(-1, 5)
    fill_conjugate_power_cube(output, coordinates, values)
    for y in range(3):
        for x in range(4):
            np.testing.assert_allclose(
                output[:, (-y) % 3, (-x) % 4],
                output[(-np.arange(5)) % 5, y, x],
            )
