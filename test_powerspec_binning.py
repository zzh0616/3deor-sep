import unittest

import numpy as np

from ops_scripts.estimate_compiled_response_marginalized_ps2d import (
    _build_fourier_geometry,
)
from powerspec import (
    PowerSpecConfig,
    _frequency_spacing_to_mpc,
    compute_eor_window_mask,
    compute_eor_window_mask_for_result,
    compute_power_spectra,
    select_eor_window_bins,
)


class PowerSpectrumBinningTest(unittest.TestCase):
    def test_right_edge_modes_do_not_wrap_between_2d_rows(self) -> None:
        rng = np.random.default_rng(20260711)
        cube = rng.normal(size=(6, 8, 8))
        config = PowerSpecConfig(
            dx=1.0,
            dy=1.0,
            df=1.0,
            unit_x="mpc",
            unit_y="mpc",
            unit_f="mpc",
            freq_axis=0,
            nbins_1d=4,
            nbins_kperp=2,
            nbins_kpar=2,
            stat_mode="mean",
            log_bins_2d=False,
            demean_mode="global",
        )

        result = compute_power_spectra(cube, config, window="hann")

        np.testing.assert_array_equal(
            result["p2d_counts"],
            np.asarray([[27, 18], [108, 72]], dtype=np.int32),
        )

    def test_all_modes_policy_rejects_center_only_kpar_bin(self) -> None:
        rng = np.random.default_rng(20260712)
        cube = rng.normal(size=(8, 8, 8))
        config = PowerSpecConfig(
            dx=1.0,
            dy=1.0,
            df=1.0,
            unit_x="mpc",
            unit_y="mpc",
            unit_f="mpc",
            freq_axis=0,
            nbins_1d=4,
            nbins_kperp=2,
            nbins_kpar=4,
            stat_mode="mean",
            log_bins_2d=False,
            eor_window_enabled=True,
            eor_window_kpar_min=1.0,
            eor_window_exclude_dc=False,
            eor_window_bin_policy="all_modes",
        )

        result = compute_power_spectra(cube, config, window="hann")
        center = compute_eor_window_mask(
            result["kperp_centers"], result["kpar_centers"], config
        )
        fraction = result["eor_window_mode_fractions"]["default"]
        strict = select_eor_window_bins(center, fraction, "all_modes")

        self.assertTrue(np.all(center[:, 1]))
        np.testing.assert_allclose(fraction[:, 1], 0.0, rtol=0.0, atol=0.0)
        self.assertFalse(np.any(strict[:, 1]))
        self.assertTrue(np.all(strict <= center))
        np.testing.assert_array_equal(
            compute_eor_window_mask_for_result(result, config), strict
        )

    def test_frequency_grid_start_is_separate_from_angular_reference(self) -> None:
        centered = PowerSpecConfig(
            dx=1.0,
            dy=1.0,
            df=0.4,
            unit_f="mhz",
            ref_freq_mhz=119.3,
            freq_grid_start_mhz=117.9,
        )
        legacy_equivalent = PowerSpecConfig(
            dx=1.0,
            dy=1.0,
            df=0.4,
            unit_f="mhz",
            ref_freq_mhz=117.9,
        )

        self.assertAlmostEqual(
            _frequency_spacing_to_mpc(centered, 8),
            _frequency_spacing_to_mpc(legacy_equivalent, 8),
            places=12,
        )

    def test_compiled_geometry_respects_x_y_axis_order(self) -> None:
        config = PowerSpecConfig(
            dx=1.0,
            dy=4.0,
            df=1.0,
            unit_x="mpc",
            unit_y="mpc",
            unit_f="mpc",
            freq_axis=0,
            nbins_1d=2,
            nbins_kperp=8,
            nbins_kpar=2,
            stat_mode="mean",
            log_bins_2d=False,
        )

        geometry = _build_fourier_geometry((4, 8, 4), config)

        self.assertGreaterEqual(geometry.full_mode_linear_bins[0, 1, 0], 0)
        self.assertEqual(geometry.full_mode_linear_bins[0, 0, 1], -1)


if __name__ == "__main__":
    unittest.main()
