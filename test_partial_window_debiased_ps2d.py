from types import SimpleNamespace

import numpy as np

from ops_scripts.estimate_partial_window_debiased_ps2d import (
    _add_geometric_contractions,
    _control_summary,
)
from ops_scripts.estimate_partial_window_covariance_ps2d import _mask_metrics


def test_geometric_contractions_use_fixed_layout_coordinates() -> None:
    layout = SimpleNamespace(
        kperp_centers=np.arange(10, dtype=np.float64) + 0.5,
        kperp_edges=np.arange(11, dtype=np.float64),
        kpar_values=np.arange(6, dtype=np.float64),
    )
    standard = np.ones((10, 6), dtype=bool)
    output = _add_geometric_contractions({"standard_window": standard}, layout)

    assert np.count_nonzero(output["top2_kpar_mid_kperp"]) == 6
    assert np.count_nonzero(output["top1_kpar_mid_kperp"]) == 3
    assert np.all(np.nonzero(output["top2_kpar_mid_kperp"])[1] >= 4)
    assert np.all(np.nonzero(output["top1_kpar_mid_kperp"])[1] == 5)


def test_control_summary_reports_signed_ensemble_stability() -> None:
    records = [
        {
            "integrated_power_ratio": ratio,
            "count_weighted_relative_l2": error,
            "foreground_integrated_over_eor": 0.05,
            "median_power_ratio": ratio,
        }
        for ratio, error in ((0.8, 0.2), (1.0, 0.1), (1.2, 0.2), (1.5, 0.5))
    ]
    summary = _control_summary(records)

    assert summary["count"] == 4
    assert summary["integrated_power_ratio"]["median"] == 1.1
    assert summary["fraction_integrated_within_quick_tolerance"] == 0.75
    assert summary["fraction_integrated_within_strict_tolerance"] == 0.75


def test_mask_metrics_integrate_with_fft_counts_but_report_independent_counts() -> None:
    metrics = _mask_metrics(
        np.asarray([[2.0, 10.0]]),
        np.asarray([[1.0, 10.0]]),
        np.zeros((1, 2)),
        np.asarray([[100, 1]]),
        np.ones((1, 2), dtype=bool),
        independent_counts=np.asarray([[3, 5]]),
        quick_tolerance=0.3,
        strict_tolerance=0.2,
        foreground_tolerance=0.1,
    )

    assert metrics["integrated_power_ratio"] == 210.0 / 110.0
    assert metrics["independent_mode_count"] == 8
