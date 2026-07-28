import numpy as np
import pytest

from ops_scripts.diagnose_visibility_qbeta_probe_convergence import (
    _distribution,
    cumulative_probe_counts,
)


def test_cumulative_probe_counts_uses_powers_of_two_and_final() -> None:
    assert cumulative_probe_counts(10, None) == [1, 2, 4, 8, 10]
    assert cumulative_probe_counts(8, [1, 4]) == [1, 4, 8]


def test_cumulative_probe_counts_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="inside"):
        cumulative_probe_counts(4, [0, 1])


def test_distribution_reports_population_spread() -> None:
    measured = _distribution([1.0, 2.0, 3.0])
    assert measured["count"] == 3
    assert np.isclose(measured["mean"], 2.0)
    assert np.isclose(
        measured["standard_deviation"], np.sqrt(2.0 / 3.0)
    )
