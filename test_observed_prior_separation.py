#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import torch

from observed_prior_separation import (
    partition_of_unity_grid,
    posterior_mean_for_data,
    posterior_predictive_score,
    prior_predictive_feature_scale,
    relative_linearity_error,
    solve_linear_gaussian_control,
)


class ObservedPriorSeparationTest(unittest.TestCase):
    def test_partition_of_unity_is_smooth_and_complete(self) -> None:
        weights = partition_of_unity_grid((17, 23), (4, 5))
        self.assertEqual(weights.shape, (20, 17, 23))
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, atol=1.0e-15)
        self.assertLessEqual(int(np.max(np.sum(weights > 0.0, axis=0))), 4)

    def test_prior_scaled_solve_recovers_well_measured_latent(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260717)
        design = 40.0 * torch.randn((300, 7), generator=generator, dtype=torch.float64)
        truth = torch.linspace(-1.5, 1.5, 7, dtype=torch.float64)
        data = design @ truth
        scale = torch.ones((300,), dtype=torch.float64)
        result = solve_linear_gaussian_control(
            design,
            data,
            feature_scale=scale,
        )
        torch.testing.assert_close(result.posterior_mean, truth, rtol=2.0e-4, atol=2.0e-4)
        self.assertLess(result.stats["residual_over_data_norm"], 2.0e-4)

    def test_feature_scale_floors_uncovered_rows(self) -> None:
        design = torch.tensor(
            [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=torch.float64
        )
        scale = prior_predictive_feature_scale(design, floor_quantile=0.0)
        self.assertGreater(float(scale[0]), 0.0)
        self.assertAlmostEqual(float(scale[1]), 5.0)
        self.assertAlmostEqual(float(scale[2]), 10.0)

    def test_heldout_score_prefers_consistent_features(self) -> None:
        design = torch.eye(4, dtype=torch.float64) * 20.0
        truth = torch.tensor([0.2, -0.3, 0.4, -0.1], dtype=torch.float64)
        result = solve_linear_gaussian_control(
            design,
            design @ truth,
            feature_scale=torch.ones(4, dtype=torch.float64),
        )
        good = posterior_predictive_score(
            design,
            design @ truth,
            result,
            feature_scale=torch.ones(4, dtype=torch.float64),
        )
        bad = posterior_predictive_score(
            design,
            design @ truth + 10.0,
            result,
            feature_scale=torch.ones(4, dtype=torch.float64),
        )
        self.assertLess(good["standardized_rms"], bad["standardized_rms"])

    def test_linearity_error_detects_exact_decomposition(self) -> None:
        first = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        second = torch.flip(first, dims=(0,))
        self.assertLess(relative_linearity_error(first + second, (first, second)), 1.0e-15)

    def test_reused_posterior_map_is_linear(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        design = torch.randn((40, 5), generator=generator, dtype=torch.float64)
        first = torch.randn((40,), generator=generator, dtype=torch.float64)
        second = torch.randn((40,), generator=generator, dtype=torch.float64)
        result = solve_linear_gaussian_control(
            design,
            first + second,
            feature_scale=torch.ones(40, dtype=torch.float64),
        )
        combined = posterior_mean_for_data(design, first + second, result)
        separate = posterior_mean_for_data(design, first, result) + posterior_mean_for_data(
            design, second, result
        )
        torch.testing.assert_close(combined, separate, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
