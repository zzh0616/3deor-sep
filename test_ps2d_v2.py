import json
import math
import unittest
from pathlib import Path

import numpy as np

from ps2d_v2 import (
    BandpowerProduct,
    EoRWindowSpec,
    aggregate_power_cube,
    build_mode_first_analysis_contract,
    build_cylindrical_mode_layout,
    compare_bandpowers,
    compute_ps2d_products,
    fft_auto_power_cube,
    fft_cross_power_cube,
    linear_kperp_edges,
)
from ops_scripts.evaluate_ps2d_v2_mode_first import _geometry


class PS2DV2Test(unittest.TestCase):
    def test_invalid_physical_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EoRWindowSpec(
                kpar_min=0.2,
                wedge_slope=0.3,
                wedge_intercept=0.1,
                kperp_min=2.0,
                kperp_max=1.0,
            )
        with self.assertRaises(ValueError):
            fft_auto_power_cube(
                np.ones((4, 4, 4)),
                dx_mpc=-1.0,
                dy_mpc=1.0,
                dpar_mpc=1.0,
            )

    def test_canonical_8wide_geometry_matches_frozen_audit(self) -> None:
        config_path = Path(__file__).resolve().parent / "configs" / "ps2d_v2_8wide_isobeam_patch.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        geometry = _geometry(config)
        self.assertAlmostEqual(geometry["radial_spacing_mpc"], 7.736914445464988)
        self.assertAlmostEqual(geometry["patch_wedge_slope"], 0.23941630957955987)
        self.assertAlmostEqual(geometry["kperp_uv_min_mpc_inv"], 0.01915393632618446)
        self.assertAlmostEqual(geometry["kperp_uv_max_mpc_inv"], 1.596161360515372)

        crop = int(config["image_geometry"]["eval_crop_size"])
        dx_mpc = float(geometry["spatial_spacing_mpc"])
        circle_max = float(np.max(np.abs(2.0 * math.pi * np.fft.fftfreq(crop, d=dx_mpc))))
        analysis = config["analysis"]
        contract = build_mode_first_analysis_contract(
            (len(config["frequencies_mhz"]), crop, crop),
            dx_mpc=dx_mpc,
            dy_mpc=dx_mpc,
            dpar_mpc=float(geometry["radial_spacing_mpc"]),
            full_kperp_edges=linear_kperp_edges(
                0.0, circle_max, int(analysis["full_kperp_bins"])
            ),
            window_kperp_edges=linear_kperp_edges(
                float(geometry["kperp_uv_min_mpc_inv"]),
                float(geometry["kperp_uv_max_mpc_inv"]),
                int(analysis["window_kperp_bins"]),
            ),
            window_spec=EoRWindowSpec(
                kpar_min=float(geometry["kpar_floor_mpc_inv"]),
                wedge_slope=float(geometry["patch_wedge_slope"]),
                wedge_intercept=float(geometry["wedge_buffer_mpc_inv"]),
                kperp_min=float(geometry["kperp_uv_min_mpc_inv"]),
                kperp_max=float(geometry["kperp_uv_max_mpc_inv"]),
                exclude_exact_dc=True,
            ),
            radial_nyquist_policy=str(analysis["radial_nyquist_policy"]),
            demean_mode=str(analysis["demean_mode"]),
            radial_taper=str(analysis["radial_taper"]),
            spatial_taper=str(analysis["spatial_taper"]),
        )
        self.assertEqual(
            contract.layout_sha256,
            "fe09288426ac1c118605ff53bfae52191ddf8ea2cb5f61e4feb28f8b800b2326",
        )
        self.assertEqual(
            contract.analysis_contract_sha256,
            "ce60b514464478c8d5543850805cc5f417f2bcae43f192e42adec06381bd8e64",
        )

    def test_native_kpar_coordinates_are_actual_fourier_modes(self) -> None:
        dpar = 7.736914445464988
        layout = build_cylindrical_mode_layout(
            (8, 8, 8),
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=dpar,
            kperp_edges=linear_kperp_edges(0.0, math.pi, 4),
            radial_nyquist_policy="exclude",
        )
        step = 2.0 * math.pi / (8.0 * dpar)
        np.testing.assert_allclose(
            layout.kpar_values,
            np.arange(4, dtype=np.float64) * step,
            rtol=1e-14,
            atol=1e-14,
        )

        included = build_cylindrical_mode_layout(
            (8, 8, 8),
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=dpar,
            kperp_edges=linear_kperp_edges(0.0, math.pi, 4),
            radial_nyquist_policy="include",
        )
        self.assertEqual(included.kpar_values.size, 5)
        self.assertAlmostEqual(included.kpar_values[-1], 4.0 * step)

    def test_window_modes_are_filtered_before_aggregation(self) -> None:
        rng = np.random.default_rng(20260712)
        cube = rng.normal(size=(8, 16, 16))
        products = compute_ps2d_products(
            cube,
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=1.0,
            full_kperp_edges=linear_kperp_edges(0.0, math.pi, 8),
            window_kperp_edges=linear_kperp_edges(0.1, 2.0, 2),
            window_spec=EoRWindowSpec(
                kpar_min=0.7,
                wedge_slope=1.0,
                wedge_intercept=0.1,
                kperp_min=0.1,
                kperp_max=2.0,
            ),
            radial_nyquist_policy="exclude",
        )
        layout = products.window_layout
        direct = float(
            np.sum(
                products.power_cube.reshape(-1)[layout.selected_mode_indices],
                dtype=np.float64,
            )
        )
        self.assertAlmostEqual(products.window.total_power, direct, places=10)
        self.assertTrue(
            np.any(
                (layout.selected_mode_fraction > 0.0)
                & (layout.selected_mode_fraction < 1.0)
            )
        )
        partial = (
            (layout.selected_mode_fraction > 0.0)
            & (layout.selected_mode_fraction < 1.0)
        )
        self.assertTrue(np.any(products.window.power_sum[partial] > 0.0))

    def test_full_power_sum_matches_layout_modes(self) -> None:
        rng = np.random.default_rng(7)
        power = np.square(rng.normal(size=(6, 8, 8)))
        layout = build_cylindrical_mode_layout(
            power.shape,
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=1.0,
            kperp_edges=linear_kperp_edges(0.0, math.pi, 4),
            radial_nyquist_policy="exclude",
        )
        product = aggregate_power_cube(power, layout, selected=False)
        expected = float(
            np.sum(power.reshape(-1)[layout.full_mode_indices], dtype=np.float64)
        )
        self.assertAlmostEqual(product.total_power, expected, places=12)
        self.assertTrue(
            np.all(
                product.independent_mode_counts
                <= product.fft_mode_counts
            )
        )

    def test_transverse_right_edge_is_explicitly_included(self) -> None:
        layout = build_cylindrical_mode_layout(
            (4, 8, 8),
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=1.0,
            kperp_edges=linear_kperp_edges(0.0, math.pi, 4),
            radial_nyquist_policy="exclude",
        )
        edge_mode = np.ravel_multi_index((0, 4, 0), layout.cube_shape)
        self.assertIn(edge_mode, set(layout.full_mode_indices.tolist()))

    def test_integrated_ratio_uses_mode_power_sums(self) -> None:
        counts = np.asarray([[1, 100]], dtype=np.int64)
        truth = BandpowerProduct(
            mean=np.asarray([[10.0, 1.0]]),
            power_sum=np.asarray([[10.0, 100.0]]),
            fft_mode_counts=counts,
            independent_mode_counts=counts,
            within_bin_std=np.zeros((1, 2)),
        )
        recovered = BandpowerProduct(
            mean=np.asarray([[20.0, 1.0]]),
            power_sum=np.asarray([[20.0, 100.0]]),
            fft_mode_counts=counts,
            independent_mode_counts=counts,
            within_bin_std=np.zeros((1, 2)),
        )
        metrics = compare_bandpowers(recovered, truth)
        self.assertAlmostEqual(metrics["power_sum_ratio"], 120.0 / 110.0)
        self.assertNotAlmostEqual(metrics["power_sum_ratio"], 21.0 / 11.0)

    def test_fft_normalization_closes_parseval(self) -> None:
        rng = np.random.default_rng(123)
        cube = rng.normal(size=(7, 9, 11))
        _, metadata = fft_auto_power_cube(
            cube,
            dx_mpc=1.2,
            dy_mpc=1.4,
            dpar_mpc=7.0,
            radial_taper="blackman_harris",
            spatial_taper="hann",
        )
        self.assertAlmostEqual(metadata["parseval_relative_error"], 0.0, places=14)

    def test_cross_power_uses_identical_normalization_and_allows_sign(self) -> None:
        rng = np.random.default_rng(456)
        cube = rng.normal(size=(6, 8, 8))
        auto, _ = fft_auto_power_cube(
            cube, dx_mpc=1.0, dy_mpc=1.0, dpar_mpc=2.0
        )
        same, _ = fft_cross_power_cube(
            cube, cube, dx_mpc=1.0, dy_mpc=1.0, dpar_mpc=2.0
        )
        opposite, _ = fft_cross_power_cube(
            cube, -cube, dx_mpc=1.0, dy_mpc=1.0, dpar_mpc=2.0
        )
        np.testing.assert_allclose(same, auto, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(opposite, -auto, rtol=1e-14, atol=1e-14)

        layout = build_cylindrical_mode_layout(
            cube.shape,
            dx_mpc=1.0,
            dy_mpc=1.0,
            dpar_mpc=2.0,
            kperp_edges=linear_kperp_edges(0.0, math.pi, 4),
        )
        product = aggregate_power_cube(
            opposite, layout, selected=False, allow_negative=True
        )
        self.assertLess(product.total_power, 0.0)


if __name__ == "__main__":
    unittest.main()
