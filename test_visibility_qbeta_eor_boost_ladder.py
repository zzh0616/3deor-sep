from __future__ import annotations

import numpy as np
import pytest

from ops_scripts.evaluate_visibility_qbeta_eor_boost_ladder import (
    _coarse_estimate,
    _quadratic_at_factor,
    amplified_quadratic_q,
    weighted_metrics,
)


def test_amplified_quadratic_q_matches_complex_visibility_power() -> None:
    rng = np.random.default_rng(51021)
    foreground = rng.normal(size=(7, 11)) + 1j * rng.normal(size=(7, 11))
    eor = rng.normal(size=(7, 11)) + 1j * rng.normal(size=(7, 11))
    projection = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))

    def q(values: np.ndarray) -> np.ndarray:
        transformed = projection @ values
        return np.mean(np.abs(transformed) ** 2, axis=1)

    foreground_q = q(foreground)
    eor_q = q(eor)
    total_q = q(foreground + eor)
    for factor in (0.0, 0.1, 1.0, 3.0, 30.0):
        expected = q(foreground + factor * eor)
        actual = amplified_quadratic_q(
            foreground_q=foreground_q,
            eor_q=eor_q,
            total_q=total_q,
            amplitude_factor=factor,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-13)
        if factor > 0.0:
            plus = q(foreground + factor * eor)
            minus = q(foreground - factor * eor)
            np.testing.assert_allclose(
                0.5 * (plus + minus) - foreground_q,
                factor * factor * eor_q,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                (plus - minus) / (2.0 * factor),
                total_q - foreground_q - eor_q,
                rtol=2e-13,
                atol=2e-13,
            )


def test_amplified_quadratic_q_rejects_invalid_factor() -> None:
    values = np.ones(3)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        amplified_quadratic_q(
            foreground_q=values,
            eor_q=values,
            total_q=values,
            amplitude_factor=-1.0,
        )


def test_quadratic_at_factor_does_not_assume_zero_is_first() -> None:
    factors = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0])
    values = np.arange(15).reshape(5, 3)
    np.testing.assert_array_equal(
        _quadratic_at_factor(factors, values, 0.0), values[2]
    )


def test_coarse_estimate_applies_transform_and_response_normalization() -> None:
    products = {
        "demo_transform": np.asarray([[1.0, 0.0], [0.25, 0.75]]),
        "demo_response": np.asarray([[2.0, 0.0], [1.0, 3.0]]),
    }
    estimate = _coarse_estimate(np.asarray([4.0, 8.0]), products, "demo")
    np.testing.assert_allclose(estimate, np.asarray([2.0, 1.75]))


def test_weighted_metrics_uses_power_ratio_and_mode_weights() -> None:
    metrics = weighted_metrics(
        np.asarray([1.0, 6.0]),
        np.asarray([1.0, 3.0]),
        np.asarray([3.0, 1.0]),
    )
    assert metrics["integrated_power_ratio"] == pytest.approx(1.5)
    assert metrics["passing_10pct_count"] == 1
