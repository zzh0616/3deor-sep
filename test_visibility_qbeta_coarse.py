from __future__ import annotations

import numpy as np

from visibility_qbeta_coarse import (
    build_coarse_groups,
    evaluate_coarse_profile,
    matching_mode_counts,
    mode_weighted_transform,
    transform_q,
)


def test_mode_weighted_transform_matches_normalized_average() -> None:
    response = np.diag([2.0, 4.0, 8.0, 16.0])
    groups = build_coarse_groups(
        np.arange(4),
        kperp_count=2,
        kpar_count=2,
        kperp_profile="pair",
        kpar_group_size=1,
    )
    modes = np.asarray([1.0, 3.0, 5.0, 7.0])
    transform = mode_weighted_transform(response, groups, modes)
    q_values = response @ np.asarray([10.0, 20.0, 30.0, 40.0])
    coarse_response = transform @ response
    estimate = transform_q(q_values, transform) / np.sum(
        coarse_response, axis=1
    )
    np.testing.assert_allclose(
        estimate,
        np.asarray(
            [
                (1.0 * 10.0 + 5.0 * 30.0) / 6.0,
                (3.0 * 20.0 + 7.0 * 40.0) / 10.0,
            ]
        ),
    )


def test_matching_mode_counts_uses_nominal_output_cell() -> None:
    result = matching_mode_counts(
        np.asarray([1, 2, 5]),
        kpar_count=3,
        source_kperp_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        source_kpar_indices=np.asarray([0, 1, 2, 0, 1, 2]),
        source_mode_counts=np.asarray([2, 3, 5, 7, 11, 13]),
    )
    np.testing.assert_array_equal(result, np.asarray([3.0, 5.0, 13.0]))


def test_exact_identity_response_closes_after_coarsening() -> None:
    response = np.eye(4)
    source_power = np.asarray([2.0, 3.0, 5.0, 7.0])
    groups = build_coarse_groups(
        np.arange(4),
        kperp_count=2,
        kpar_count=2,
        kperp_profile="pair",
        kpar_group_size=1,
    )
    heldout = np.repeat(source_power[None, :], 3, axis=0)
    summary, arrays = evaluate_coarse_profile(
        response=response,
        source_power=source_power,
        source_in_geometric_window=np.ones(4, dtype=np.int8),
        output_mode_counts=np.asarray([1.0, 2.0, 3.0, 4.0]),
        groups=groups,
        restricted_q=source_power,
        heldout_q=heldout,
        heldout_total_q=heldout,
        bank_foreground_q=np.zeros(4),
        bank_eor_q=source_power,
        bank_total_q=source_power,
        source_kperp_indices=np.asarray([0, 0, 1, 1]),
        source_kpar_values=np.asarray([0.0, 1.0, 0.0, 1.0]),
        output_kpar_values=np.asarray([0.0, 1.0]),
        minimum_relative_response=0.0,
        minimum_window_fraction=1.0,
    )
    assert summary["strict_group_count"] == 2
    assert summary["bank_total"]["relative_l2"] < 1e-14
    assert summary["foreground_effect_maximum_fraction"] == 0.0
    assert np.all(arrays["strict_selected"])


def test_physical_shift_gate_checks_pure_and_total_separately() -> None:
    response = np.eye(2)
    target = np.asarray([2.0, 3.0])
    groups = build_coarse_groups(
        np.arange(2),
        kperp_count=1,
        kpar_count=2,
        kperp_profile="fine",
        kpar_group_size=1,
    )
    summary, arrays = evaluate_coarse_profile(
        response=response,
        source_power=target,
        source_in_geometric_window=np.ones(2, dtype=np.int8),
        output_mode_counts=np.ones(2),
        groups=groups,
        restricted_q=target,
        heldout_q=np.repeat(target[None, :], 2, axis=0),
        heldout_total_q=np.repeat(target[None, :], 2, axis=0),
        bank_foreground_q=np.zeros(2),
        bank_eor_q=target,
        bank_total_q=target,
        physical_shift_q=np.asarray([[2.5, 3.0], [2.0, 3.0]]),
        physical_shift_total_q=np.repeat(target[None, :], 2, axis=0),
        source_kperp_indices=np.asarray([0, 0]),
        source_kpar_values=np.asarray([0.0, 1.0]),
        output_kpar_values=np.asarray([0.0, 1.0]),
        minimum_relative_response=0.0,
        minimum_window_fraction=1.0,
    )
    assert summary["strict_without_physical_shifts_group_count"] == 2
    assert summary["strict_group_count"] == 1
    assert (
        summary["physical_shifts_pure"][
            "maximum_per_group_error_fraction"
        ]
        == 0.25
    )
    assert (
        summary["physical_shifts_total"][
            "maximum_per_group_error_fraction"
        ]
        == 0.0
    )
    np.testing.assert_array_equal(arrays["strict_selected"], [0, 1])
