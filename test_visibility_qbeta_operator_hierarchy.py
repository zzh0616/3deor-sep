from __future__ import annotations

import numpy as np

from ops_scripts.compare_visibility_qbeta_operator_hierarchy import (
    combined_response_diagnostics,
    weighted_metrics,
)


def test_weighted_metrics_uses_mode_weights() -> None:
    metrics = weighted_metrics(
        np.asarray([1.0, 3.0]),
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 3.0]),
    )
    assert metrics["count"] == 2
    np.testing.assert_allclose(metrics["integrated_power_ratio"], 10.0 / 7.0)
    np.testing.assert_allclose(metrics["relative_l2"], np.sqrt(3.0 / 13.0))


def test_combined_response_diagnostics_separates_gain_from_window() -> None:
    exact_response = np.asarray([[1.0, 3.0], [2.0, 2.0]])
    common_response = 0.9 * exact_response
    shared = {
        "selected_bank_rows": np.asarray([4, 9]),
        "bank_foreground_q": np.asarray([1.0, 2.0]),
        "bank_eor_q": np.asarray([3.0, 4.0]),
        "bank_total_q": np.asarray([4.0, 6.0]),
    }
    diagnostics = combined_response_diagnostics(
        exact={"calibration_response": exact_response, **shared},
        common={"calibration_response": common_response, **shared},
    )
    np.testing.assert_allclose(diagnostics["response_relative_l2"], 0.1)
    np.testing.assert_allclose(
        diagnostics["normalized_window_relative_l2"], 0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        diagnostics["common_to_exact_response_row_sum_ratio"]["median"],
        0.9,
    )
    assert diagnostics["observed_q"]["bank_total_q"]["array_equal"]
