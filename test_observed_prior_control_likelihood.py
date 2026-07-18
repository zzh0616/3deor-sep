#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from astropy.io import fits

OPS_DIR = Path(__file__).resolve().parent / "ops_scripts"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from evaluate_observed_prior_control_likelihood import (  # noqa: E402
    SourceGroupSpec,
    _build_specs,
    _catalog_to_cube_k,
    _source_group_cube,
)


def _header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 16
    header["NAXIS2"] = 16
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = -27.0
    header["CRPIX1"] = 8.5
    header["CRPIX2"] = 8.5
    header["CDELT1"] = -32.0 / 3600.0
    header["CDELT2"] = 32.0 / 3600.0
    return header


class ObservedPriorControlLikelihoodTest(unittest.TestCase):
    def test_nuisance_families_can_be_ablated_independently(self) -> None:
        args = Namespace(
            catalog_nuisance=False,
            diffuse_nuisance=True,
            catalog_amplitude_prior_std=0.2,
            catalog_slope_prior_std=0.3,
            diffuse_amplitude_prior_std=0.4,
            diffuse_slope_prior_std=0.5,
        )
        groups = [SourceGroupSpec("group", Path("unused.fits"), (0,))]
        specs = _build_specs(groups, args=args, diffuse_cell_count=2)
        self.assertEqual(len(specs), 4)
        self.assertEqual({spec.family for spec in specs}, {"diffuse"})
        self.assertEqual([spec.prior_std for spec in specs], [0.4, 0.5, 0.4, 0.5])

        args.catalog_nuisance = True
        args.diffuse_nuisance = False
        specs = _build_specs(groups, args=args, diffuse_cell_count=2)
        self.assertEqual(len(specs), 2)
        self.assertEqual({spec.family for spec in specs}, {"catalog"})

        args.catalog_nuisance = False
        with self.assertRaisesRegex(ValueError, "at least one nuisance family"):
            _build_specs(groups, args=args, diffuse_cell_count=2)

    def test_exact_groups_sum_to_direct_catalog_at_every_frequency(self) -> None:
        catalog = {
            "ra_deg": np.asarray([0.0, 0.02]),
            "dec_deg": np.asarray([-27.0, -27.01]),
            "flux_ref_jy": np.asarray([1.0, 0.25]),
            "ref_freq_mhz": np.asarray([150.0, 200.0]),
            "alpha": np.asarray([-0.7, -1.1]),
            "curvature": np.asarray([0.01, -0.02]),
            "major_obs_arcsec": np.asarray([np.nan, np.nan]),
            "minor_obs_arcsec": np.asarray([np.nan, np.nan]),
            "pa_obs_deg": np.asarray([np.nan, np.nan]),
            "psf_major_arcsec": np.asarray([np.nan, np.nan]),
            "psf_minor_arcsec": np.asarray([np.nan, np.nan]),
            "psf_pa_deg": np.asarray([np.nan, np.nan]),
        }
        frequencies = np.asarray([117.9, 119.3, 120.7], dtype=np.float64)
        header = _header()
        direct, _ = _catalog_to_cube_k(
            catalog,
            frequencies=frequencies,
            header=header,
            shape=(16, 16),
            pixel_arcsec=32.0,
            insert_mode="bilinear",
        )
        with tempfile.TemporaryDirectory() as temporary:
            placeholder = Path(temporary) / "unused.fits"
            groups = [
                SourceGroupSpec("first", placeholder, (0,)),
                SourceGroupSpec("second", placeholder, (1,)),
            ]
            grouped = sum(
                (
                    _source_group_cube(
                        group,
                        frequencies,
                        (16, 16),
                        prior_catalog=catalog,
                        header=header,
                        pixel_arcsec=32.0,
                        insert_mode="bilinear",
                    )
                    for group in groups
                ),
                np.zeros_like(direct),
            )
        np.testing.assert_allclose(grouped, direct, rtol=2.0e-15, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
