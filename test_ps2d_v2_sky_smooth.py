#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from ps2d_v2_sky_smooth import (
    build_augmented_ridge_system,
    compose_sky_smooth_cube,
    matrix_free_lsqr,
    orthonormal_chebyshev_spectral_basis,
    relative_adjoint_error,
)


class SkySmoothBasisTests(unittest.TestCase):
    def test_basis_is_orthonormal_and_spans_constant_linear_columns(self) -> None:
        basis = orthonormal_chebyshev_spectral_basis(
            8,
            [0, 1],
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(basis.shape), (8, 2))
        self.assertTrue(
            torch.allclose(basis.T @ basis, torch.eye(2, dtype=torch.float64))
        )
        coordinate = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
        projector = basis @ basis.T
        self.assertLess(
            float(
                torch.linalg.vector_norm(
                    projector @ torch.ones(8, dtype=torch.float64) - 1.0
                )
            ),
            1e-12,
        )
        self.assertLess(
            float(torch.linalg.vector_norm(projector @ coordinate - coordinate)),
            1e-12,
        )

    def test_compose_uses_free_spatial_maps_without_reference_template(self) -> None:
        basis = torch.tensor(
            [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
        )
        coefficients = torch.stack(
            (
                torch.ones((2, 2), dtype=torch.float64) * 3.0,
                torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64),
            )
        )
        cube = compose_sky_smooth_cube(coefficients, basis)
        self.assertTrue(torch.equal(cube[1], coefficients[0]))
        self.assertTrue(torch.equal(cube[0], coefficients[0] - coefficients[1]))
        self.assertTrue(torch.equal(cube[2], coefficients[0] + coefficients[1]))


class MatrixFreeLsqrTests(unittest.TestCase):
    def test_adjoint_dot_product_closure(self) -> None:
        matrix = torch.tensor(
            [[1.0, 2.0], [-0.5, 1.0], [3.0, -2.0]], dtype=torch.float64
        )
        stats = relative_adjoint_error(
            lambda value: matrix @ value,
            lambda value: matrix.T @ value,
            torch.tensor([0.3, -0.8], dtype=torch.float64),
            torch.tensor([1.1, 0.2, -0.7], dtype=torch.float64),
        )
        self.assertLess(stats["relative_error"], 1e-14)

    def test_lsqr_matches_explicit_overdetermined_least_squares(self) -> None:
        generator = torch.Generator().manual_seed(17)
        matrix = torch.randn((12, 5), generator=generator, dtype=torch.float64)
        rhs = torch.randn((12,), generator=generator, dtype=torch.float64)
        result = matrix_free_lsqr(
            lambda value: matrix @ value,
            lambda value: matrix.T @ value,
            rhs,
            max_iters=20,
            relative_residual_tolerance=0.0,
        )
        expected = torch.linalg.lstsq(matrix, rhs).solution
        self.assertTrue(
            torch.allclose(result.solution, expected, rtol=1e-10, atol=1e-10)
        )
        self.assertLess(
            abs(
                result.stats["actual_residual_norm"]
                - float(torch.linalg.vector_norm(matrix @ expected - rhs))
            ),
            1e-10,
        )

    def test_lsqr_returns_minimum_norm_underdetermined_solution(self) -> None:
        matrix = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64
        )
        rhs = torch.tensor([2.0, -1.0], dtype=torch.float64)
        result = matrix_free_lsqr(
            lambda value: matrix @ value,
            lambda value: matrix.T @ value,
            rhs,
            max_iters=10,
            relative_residual_tolerance=1e-13,
        )
        expected = torch.linalg.pinv(matrix) @ rhs
        self.assertTrue(
            torch.allclose(result.solution, expected, rtol=1e-11, atol=1e-11)
        )
        self.assertLess(result.stats["actual_relative_residual"], 1e-12)

    def test_augmented_ridge_matches_explicit_finite_covariance_map(self) -> None:
        matrix = torch.tensor(
            [[1.0, 2.0], [-0.5, 1.0], [3.0, -2.0]], dtype=torch.float64
        )
        rhs = torch.tensor([0.4, -1.2, 2.1], dtype=torch.float64)
        precision = torch.tensor([0.3, 1.7], dtype=torch.float64)
        system = build_augmented_ridge_system(
            lambda value: matrix @ value,
            lambda value: matrix.T @ value,
            rhs,
            torch.zeros((2,), dtype=torch.float64),
            precision,
        )
        result = matrix_free_lsqr(
            system.forward,
            system.adjoint,
            system.rhs,
            max_iters=20,
            relative_residual_tolerance=0.0,
        )
        expected = torch.linalg.solve(
            matrix.T @ matrix + torch.diag(precision.square()),
            matrix.T @ rhs,
        )
        self.assertLess(
            float(torch.linalg.vector_norm(result.solution - expected)), 1e-11
        )

    def test_augmented_ridge_rejects_nonfinite_covariance(self) -> None:
        with self.assertRaises(ValueError):
            build_augmented_ridge_system(
                lambda value: value,
                lambda value: value,
                torch.ones((2,), dtype=torch.float64),
                torch.zeros((2,), dtype=torch.float64),
                torch.tensor([1.0, 0.0], dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
