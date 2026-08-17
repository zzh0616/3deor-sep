from __future__ import annotations

import numpy as np
import pytest

from ops_scripts.calibrate_visibility_qbeta_noiseless import (
    _visibility_bandpowers,
)
from ops_scripts.evaluate_visibility_qbeta_noise_systematics import (
    _coarse_significance_components,
    _factor_samples,
    _foreground_contamination_metrics,
    _gain_products,
    _solve_significance_factor,
    lookup_ska_low_stokes_i_sefd,
    thermal_noise_sigma_per_real_component,
)


def _bandpower_kwargs() -> dict[str, object]:
    frequencies = np.linspace(100e6, 100.7e6, 8)
    return {
        "frequencies_hz": frequencies,
        "analysis_frequency_indices": np.arange(8),
        "filter_bandwidth_scope": "analysis_subband",
        "row_kperp": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "kperp_edges": np.asarray([0.0, 1.0]),
        "maximum_delays_s": np.asarray([0.0]),
        "dpss_eigenvalue_threshold": 1e-12,
        "foreground_filter": "none",
        "suppression_strength": 0.0,
        "polynomial_degree": 0,
        "spectral_taper": "none",
    }


def test_cross_bandpower_matches_auto_and_is_symmetric() -> None:
    rng = np.random.default_rng(42)
    left = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    right = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    auto, *_ = _visibility_bandpowers(
        visibilities=left, **_bandpower_kwargs()
    )
    self_cross, *_ = _visibility_bandpowers(
        visibilities=left,
        cross_visibilities=left,
        **_bandpower_kwargs(),
    )
    forward, *_ = _visibility_bandpowers(
        visibilities=left,
        cross_visibilities=right,
        **_bandpower_kwargs(),
    )
    reverse, *_ = _visibility_bandpowers(
        visibilities=right,
        cross_visibilities=left,
        **_bandpower_kwargs(),
    )
    np.testing.assert_allclose(self_cross, auto, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(forward, reverse, rtol=2e-15, atol=1e-15)


def test_all_one_frequency_weights_preserve_auto_bitwise() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    plain, *_ = _visibility_bandpowers(
        visibilities=values, **_bandpower_kwargs()
    )
    weighted, *_ = _visibility_bandpowers(
        visibilities=values,
        input_frequency_weights=np.ones(8),
        **_bandpower_kwargs(),
    )
    np.testing.assert_array_equal(weighted, plain)


def test_thermal_noise_sigma_uses_two_independent_splits() -> None:
    sigma, seconds = thermal_noise_sigma_per_real_component(
        np.asarray([1000.0, 2000.0]),
        channel_bandwidth_hz=100000.0,
        total_integration_hours=100.0,
        time_step_count=20,
        split_count=2,
    )
    assert seconds == pytest.approx(9000.0)
    expected = np.asarray([1000.0, 2000.0]) / np.sqrt(2.0 * 1e5 * 9000.0)
    np.testing.assert_allclose(sigma, expected)


def test_ska_sefd_lookup_matches_stokes_i_definition(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sefd.h5"
    with h5py.File(path, "w") as handle:
        dimensions = handle.create_group("dimensions")
        dimensions.create_dataset("frequency", data=[50.0, 100.0, 150.0, 200.0])
        dimensions.create_dataset("azimuth", data=[0.0])
        dimensions.create_dataset("zenith_angle", data=[0.0])
        dimensions.create_dataset("lst", data=[0.0])
        values = np.empty((1, 1, 1, 4, 2), dtype=np.float64)
        values[..., 0] = 100.0
        values[..., 1] = 200.0
        handle.create_dataset("sefd", data=values)
    actual, metadata = lookup_ska_low_stokes_i_sefd(
        path,
        frequencies_mhz=np.asarray([75.0, 125.0, 175.0]),
        az_deg=0.0,
        el_deg=90.0,
        start_lst_hour=0.0,
        end_lst_hour=0.0,
    )
    np.testing.assert_allclose(actual, 0.5 * np.sqrt(100.0**2 + 200.0**2))
    assert metadata["selected_az_deg"] == 0.0


def test_zero_rms_gain_products_are_identity() -> None:
    gains = _gain_products(
        np.random.default_rng(5),
        frequencies_mhz=np.linspace(100.0, 101.0, 8),
        antenna1=np.asarray([0, 1, 2]),
        antenna2=np.asarray([1, 2, 3]),
        realization_count=4,
        rms=0.0,
        profile="smooth",
        ripple_cycles=2.0,
    )
    assert gains.shape == (4, 8, 3)
    np.testing.assert_array_equal(gains, 1.0)


def test_significance_factor_solver_hits_requested_value() -> None:
    rng = np.random.default_rng(9)
    thermal = {
        "null": rng.normal(scale=4.0, size=(2000, 1)),
        "total_linear": rng.normal(scale=1.0, size=(2000, 1)),
    }
    factor, achieved = _solve_significance_factor(
        thermal=thermal,
        eor_q=np.asarray([1.0]),
        target=np.asarray([1.0]),
        reference_group=0,
        requested_significance=10.0,
    )
    samples = _factor_samples(thermal, np.asarray([1.0]), factor)
    measured = factor * factor / np.std(samples[:, 0], ddof=1)
    assert achieved == pytest.approx(10.0, rel=1e-10)
    assert measured == pytest.approx(10.0, rel=1e-10)


def test_significance_components_use_coarse_selected_axis() -> None:
    transform = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    coarse = {
        "demo_transform": transform,
        "demo_response": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    }
    thermal = {
        "null": np.arange(8, dtype=float).reshape(2, 4),
        "total_linear": np.arange(8, 16, dtype=float).reshape(2, 4),
    }
    selected = np.asarray([False, True, True])
    coarse_thermal, coarse_eor = _coarse_significance_components(
        thermal=thermal,
        eor_q=np.asarray([1.0, 2.0, 3.0, 4.0]),
        coarse=coarse,
        profile="demo",
        selected=selected,
    )
    np.testing.assert_array_equal(
        coarse_thermal["null"], thermal["null"] @ transform[selected].T
    )
    np.testing.assert_array_equal(
        coarse_thermal["total_linear"],
        thermal["total_linear"] @ transform[selected].T,
    )
    np.testing.assert_array_equal(coarse_eor, np.asarray([4.0, 16.0]))


def test_foreground_contamination_uses_absolute_power() -> None:
    samples = np.asarray([[1.0, -2.0], [-1.0, 4.0]])
    metrics = _foreground_contamination_metrics(
        samples,
        target=np.asarray([2.0, 2.0]),
        weights=np.asarray([1.0, 3.0]),
    )
    # Per-realization absolute weighted ratios are 7/8 and 13/8.
    assert metrics[
        "mean_integrated_absolute_foreground_to_target"
    ] == pytest.approx(1.25)
    assert metrics["maximum_mean_cell_foreground_to_target"] == pytest.approx(
        0.5
    )
