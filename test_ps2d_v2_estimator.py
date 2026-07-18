import unittest

import numpy as np
import torch

from ops_scripts.validate_ps2d_v2_estimator_identity import _source_window_metrics
from ps2d_v2 import (
    EoRWindowSpec,
    aggregate_power_cube,
    build_mode_first_analysis_contract,
    fft_auto_power_cube,
    linear_kperp_edges,
)
from ps2d_v2_config import ResolvedModeFirstAnalysis
from ps2d_v2_estimator import (
    FourierControlCubeProjector,
    IdentityCubeProjector,
    TorchBandpowerTransform,
    analytic_identity_source_response,
    build_mode_first_estimator_contract_from_analysis,
    calibrate_mode_first_transfer,
    estimate_projected_bandpower,
)


def _small_contract():
    shape = (6, 8, 8)
    window_spec = EoRWindowSpec(
        kpar_min=2.0,
        wedge_slope=0.2,
        wedge_intercept=0.0,
        kperp_min=0.1,
        kperp_max=2.0,
    )
    analysis = build_mode_first_analysis_contract(
        shape,
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=1.0,
        full_kperp_edges=linear_kperp_edges(0.0, np.pi, 4),
        window_kperp_edges=linear_kperp_edges(0.1, 2.0, 2),
        window_spec=window_spec,
        radial_nyquist_policy="exclude",
        demean_mode="global",
        radial_taper="hann",
        spatial_taper="hann",
    )
    config = {
        "estimator_partitions": {
            "control_kpar_indices": [0],
            "guard_kpar_indices": [1],
        },
        "calibration_source": {
            "transverse_support": "full_fft_square",
            "kperp_bins": 4,
            "radial_nyquist_policy": "include",
        },
    }
    resolved = ResolvedModeFirstAnalysis(
        config=config,
        geometry={},
        window_spec=window_spec,
        contract=analysis,
    )
    return build_mode_first_estimator_contract_from_analysis(resolved)


class PS2DV2EstimatorTest(unittest.TestCase):
    def test_source_window_metrics_returns_category_summaries(self) -> None:
        metrics = _source_window_metrics(
            np.asarray([[0.6, 0.1, 0.2, 0.1], [0.1, 0.7, 0.1, 0.1]]),
            2,
            np.asarray(["science", "science", "guard", "radial_nyquist"]),
        )
        self.assertAlmostEqual(metrics["self_response_median"], 0.65)
        self.assertAlmostEqual(metrics["category_fraction_median"]["guard"], 0.15)

    def test_three_layout_contract_is_complete_and_disjoint(self) -> None:
        contract = _small_contract()
        self.assertEqual(
            contract.calibration_source_geometry.mode_indices.size,
            int(np.prod(contract.analysis.full_layout.cube_shape)),
        )
        self.assertGreater(contract.science_geometry.band_count, 0)
        science_cube = contract.science_geometry.cube_mode_bands.reshape(-1)
        source_cube = contract.calibration_source_geometry.cube_mode_bands.reshape(-1)
        in_science = science_cube >= 0
        np.testing.assert_array_equal(source_cube[in_science], science_cube[in_science])
        self.assertTrue(
            np.all(
                source_cube[~in_science]
                >= contract.calibration_science_band_count
            )
        )
        self.assertEqual(
            int(np.sum(contract.calibration_source_kind == "science")),
            contract.science_geometry.band_count,
        )
        self.assertIn("control", set(contract.calibration_source_kind.tolist()))
        self.assertIn("guard", set(contract.calibration_source_kind.tolist()))
        self.assertIn("radial_nyquist", set(contract.calibration_source_kind.tolist()))
        self.assertEqual(
            np.intersect1d(
                contract.control_mode_indices,
                contract.analysis.window_layout.selected_mode_indices,
            ).size,
            0,
        )
        self.assertEqual(
            np.intersect1d(
                contract.guard_mode_indices,
                contract.analysis.window_layout.selected_mode_indices,
            ).size,
            0,
        )

    def test_torch_science_transform_matches_numpy_mode_first_product(self) -> None:
        contract = _small_contract()
        transform = TorchBandpowerTransform(
            contract.science_geometry, contract.analysis, torch.device("cpu")
        )
        rng = np.random.default_rng(20260712)
        cube = rng.normal(size=contract.analysis.full_layout.cube_shape)
        measured = transform(torch.as_tensor(cube, dtype=torch.float64)).numpy()[0]
        power, _ = fft_auto_power_cube(
            cube,
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=1.0,
            demean_mode="global",
            radial_taper="hann",
            spatial_taper="hann",
        )
        product = aggregate_power_cube(
            power, contract.analysis.window_layout, selected=True
        )
        expected = product.mean.reshape(-1)[
            contract.science_geometry.active_layout_bands
        ]
        np.testing.assert_allclose(measured, expected, rtol=2e-14, atol=2e-14)

    def test_identity_probe_calibration_and_pure_signal_close(self) -> None:
        contract = _small_contract()
        transform = TorchBandpowerTransform(
            contract.science_geometry, contract.analysis, torch.device("cpu")
        )
        identity = IdentityCubeProjector(contract.estimator_contract_sha256)
        calibration = calibrate_mode_first_transfer(
            contract=contract,
            target_transform=transform,
            projector=identity,
            probes_per_source_band=2,
            batch_size=2,
            seed=20260712,
            transfer_rcond=1e-10,
        )
        np.testing.assert_allclose(
            calibration["transfer_matrix"],
            np.eye(contract.science_geometry.band_count),
            rtol=1e-10,
            atol=1e-10,
        )
        raw_window = calibration["raw_source_window_matrix"]
        np.testing.assert_allclose(
            np.sum(raw_window, axis=1), 1.0, rtol=1e-12, atol=1e-12
        )
        analytic = analytic_identity_source_response(
            contract=contract, target_transform=transform, batch_size=4
        )
        np.testing.assert_allclose(
            np.sum(analytic["source_window_matrix"], axis=1),
            1.0,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            raw_window,
            analytic["source_window_matrix"],
            rtol=0.35,
            atol=0.15,
        )

        rng = np.random.default_rng(7)
        cube = torch.as_tensor(
            rng.normal(size=contract.analysis.full_layout.cube_shape),
            dtype=torch.float64,
        )
        direct = transform(cube).numpy()[0]
        projected, _ = identity.project(cube)
        projected_power = transform(projected).numpy()[0]
        row, deconvolved = estimate_projected_bandpower(
            projected_power, calibration
        )
        np.testing.assert_allclose(row, direct, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(deconvolved, direct, rtol=1e-10, atol=1e-10)

    def test_calibration_rejects_contract_mismatch(self) -> None:
        contract = _small_contract()
        transform = TorchBandpowerTransform(
            contract.science_geometry, contract.analysis, torch.device("cpu")
        )
        with self.assertRaises(ValueError):
            calibrate_mode_first_transfer(
                contract=contract,
                target_transform=transform,
                projector=IdentityCubeProjector("wrong"),
                probes_per_source_band=2,
                batch_size=2,
                seed=1,
                transfer_rcond=1e-10,
            )

    def test_control_projector_removes_an_in_span_foreground(self) -> None:
        contract = _small_contract()
        transform = TorchBandpowerTransform(
            contract.science_geometry, contract.analysis, torch.device("cpu")
        )
        rng = np.random.default_rng(11)
        design = torch.as_tensor(
            rng.normal(size=(2, *contract.analysis.full_layout.cube_shape)),
            dtype=torch.float64,
        )
        projector = FourierControlCubeProjector(
            design,
            transform,
            contract.control_mode_indices,
            estimator_contract_sha256=contract.estimator_contract_sha256,
            rcond=1e-12,
            ridge_fraction=0.0,
        )
        coefficients = torch.as_tensor([0.7, -1.3], dtype=torch.float64)
        foreground = torch.einsum("p,pfij->fij", coefficients, design)
        residual, fitted = projector.project(foreground)
        np.testing.assert_allclose(residual.numpy(), 0.0, atol=2e-12)
        np.testing.assert_allclose(fitted.numpy(), coefficients.numpy(), atol=2e-12)


if __name__ == "__main__":
    unittest.main()
