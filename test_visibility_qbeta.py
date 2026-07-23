#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np

from ops_scripts.combine_visibility_qbeta_row_partitions import (
    _windowed_metrics,
)
from visibility_qbeta import (
    C_M_S,
    OMEGA_EARTH_RAD_S,
    build_sky_band_layout,
    direct_dft_kernel_numpy,
    direction_cosines,
    oskar_time_smearing_cycles,
    reporting_band_ids,
    source_bandpowers,
    stratified_row_indices,
    weighted_response_pseudoinverse,
)


def test_direction_cosines_at_phase_centre() -> None:
    l_cosine, m_cosine, n = direction_cosines(
        np.asarray([13.0]),
        np.asarray([-27.0]),
        phase_ra_deg=13.0,
        phase_dec_deg=-27.0,
    )
    np.testing.assert_allclose(l_cosine, 0.0, atol=1e-15)
    np.testing.assert_allclose(m_cosine, 0.0, atol=1e-15)
    np.testing.assert_allclose(n, 1.0, atol=1e-15)


def test_time_smearing_uvw_algebra_matches_station_expression() -> None:
    dec0 = math.radians(-27.0)
    hour_angle = 0.37
    baseline_x = np.asarray([120.0, -90.0])
    baseline_y = np.asarray([-45.0, 70.0])
    baseline_z = np.asarray([20.0, 35.0])
    transverse = baseline_x * math.cos(hour_angle) - baseline_y * math.sin(
        hour_angle
    )
    u = baseline_x * math.sin(hour_angle) + baseline_y * math.cos(hour_angle)
    v = baseline_z * math.cos(dec0) - transverse * math.sin(dec0)
    w = baseline_z * math.sin(dec0) + transverse * math.cos(dec0)
    uvw = np.stack((u, v, w), axis=1)
    l_cosine = np.asarray([0.01, -0.02])
    m_cosine = np.asarray([-0.015, 0.005])
    n_minus_one = (
        np.sqrt(1.0 - l_cosine * l_cosine - m_cosine * m_cosine) - 1.0
    )
    frequency = 119.45e6
    interval = 10.0
    actual = oskar_time_smearing_cycles(
        uvw,
        l_cosine,
        m_cosine,
        n_minus_one,
        frequency_hz=frequency,
        integration_time_s=interval,
        phase_dec_deg=-27.0,
    )
    expected_path = (
        transverse[:, None] * l_cosine[None, :]
        + u[:, None] * math.sin(dec0) * m_cosine[None, :]
        - u[:, None] * math.cos(dec0) * n_minus_one[None, :]
    )
    expected = (
        frequency
        * interval
        * OMEGA_EARTH_RAD_S
        * expected_path
        / C_M_S
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_direct_dft_kernel_is_unity_at_phase_centre() -> None:
    uvw = np.asarray([[10.0, -20.0, 30.0], [50.0, 40.0, -10.0]])
    kernel = direct_dft_kernel_numpy(
        uvw,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
        frequency_hz=120e6,
        channel_bandwidth_hz=100e3,
        integration_time_s=10.0,
        phase_dec_deg=-27.0,
    )
    np.testing.assert_allclose(kernel, 1.0, rtol=0.0, atol=1e-15)


def test_sky_band_layout_and_reporting_region() -> None:
    layout = build_sky_band_layout(
        (8, 16, 16),
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=2.0,
        kperp_edges=np.linspace(0.0, math.sqrt(2.0) * math.pi, 5),
    )
    assert layout.band_count == 4 * 4
    assert int(np.sum(layout.counts)) == 7 * 16 * 16
    mirrored = layout.mode_bands[
        (-np.arange(8)) % 8,
    ][:, (-np.arange(16)) % 16][:, :, (-np.arange(16)) % 16]
    np.testing.assert_array_equal(layout.mode_bands, mirrored)
    selected = reporting_band_ids(
        layout,
        high_kpar_fraction=0.5,
        mid_kperp_fraction_range=(0.25, 0.75),
    )
    assert selected.size == 2 * 2
    layout_with_nyquist = build_sky_band_layout(
        (8, 16, 16),
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=2.0,
        kperp_edges=np.linspace(0.0, math.sqrt(2.0) * math.pi, 5),
        exclude_radial_nyquist=False,
    )
    selected_with_nyquist = reporting_band_ids(
        layout_with_nyquist,
        high_kpar_fraction=0.5,
        mid_kperp_fraction_range=(0.25, 0.75),
        radial_band_count=4,
    )
    assert selected_with_nyquist.size == selected.size
    assert np.max(
        layout_with_nyquist.active_kpar_indices[selected_with_nyquist]
    ) == 3


def test_source_bandpowers_for_unit_phase_band() -> None:
    layout = build_sky_band_layout(
        (8, 8, 8),
        dx_mpc=1.0,
        dy_mpc=1.0,
        dpar_mpc=1.0,
        kperp_edges=np.linspace(0.0, math.sqrt(2.0) * math.pi, 3),
    )
    rng = np.random.default_rng(4)
    white = rng.normal(size=layout.cube_shape)
    spectrum = np.fft.fftn(white, norm="ortho")
    band = 3
    mask = layout.mode_bands == band
    selected = np.zeros_like(spectrum)
    selected[mask] = spectrum[mask] / np.abs(spectrum[mask])
    cube = np.fft.ifftn(selected, norm="ortho").real
    power = source_bandpowers(cube, layout)
    np.testing.assert_allclose(power[band], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.delete(power, band), 0.0, atol=1e-14)


def test_stratified_rows_and_response_inverse() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
    chosen = stratified_row_indices(
        values,
        np.asarray([0.0, 1.0, 2.0]),
        rows_per_bin=2,
        seed=5,
    )
    assert chosen.size == 4
    assert np.count_nonzero(chosen < 3) == 2
    assert np.count_nonzero(chosen >= 3) == 2
    first = stratified_row_indices(
        values,
        np.asarray([0.0, 1.0, 2.0]),
        rows_per_bin=1,
        seed=9,
        partition_index=0,
        partition_count=2,
    )
    second = stratified_row_indices(
        values,
        np.asarray([0.0, 1.0, 2.0]),
        rows_per_bin=1,
        seed=9,
        partition_index=1,
        partition_count=2,
    )
    assert np.intersect1d(first, second).size == 0
    middle_only = stratified_row_indices(
        values,
        np.asarray([0.0, 1.0, 2.0]),
        rows_per_bin=2,
        seed=5,
        bin_indices=np.asarray([1]),
    )
    assert middle_only.size == 2
    assert np.all(middle_only >= 3)

    response = np.asarray(
        [[2.0, 0.1], [0.2, 3.0], [1.0, -0.5]], dtype=np.float64
    )
    inverse, diagnostics = weighted_response_pseudoinverse(
        response, rcond=1e-12
    )
    assert diagnostics["rank"] == 2
    np.testing.assert_allclose(
        inverse @ response, np.eye(2), rtol=1e-12, atol=1e-12
    )


def test_window_selection_uses_response_target_concentration() -> None:
    response = np.asarray(
        [
            [9.0, 1.0, 0.0],
            [1.0, 1.0, 8.0],
            [1.0, 1.0, 0.0],
        ]
    )
    source_power = np.asarray([2.0, 3.0, 7.0])
    observed_q = response @ source_power
    metrics = _windowed_metrics(
        response=response,
        observed_q=observed_q,
        source_power=source_power,
        minimum_relative_response=0.1,
        target_source_positions=np.asarray([0, 1]),
        minimum_target_window_fraction=0.8,
    )
    np.testing.assert_array_equal(
        metrics["selected_window_positions"], np.asarray([0, 2])
    )
    np.testing.assert_allclose(
        metrics["target_window_fraction"], np.asarray([1.0, 0.2, 1.0])
    )
    assert metrics["realizations"][0]["relative_l2"] == 0.0

    empty = _windowed_metrics(
        response=response,
        observed_q=observed_q,
        source_power=source_power,
        minimum_relative_response=1.1,
        target_source_positions=np.asarray([0, 1]),
        minimum_target_window_fraction=0.8,
    )
    assert empty["selected_window_count"] == 0
    assert math.isnan(empty["realizations"][0]["relative_l2"])


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} visibility_qbeta tests passed")
