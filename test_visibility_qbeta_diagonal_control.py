from __future__ import annotations

import numpy as np
import pytest

from ops_scripts.build_visibility_qbeta_diagonal_response_control import (
    row_sum_matched_diagonal_response,
)


def test_row_sum_matched_diagonal_response_preserves_gain() -> None:
    response = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 0.5, 1.5]], dtype=np.float64
    )
    diagonal, positions = row_sum_matched_diagonal_response(
        response=response,
        output_band_ids=np.asarray([30, 10]),
        source_band_ids=np.asarray([10, 20, 30]),
    )
    np.testing.assert_array_equal(positions, np.asarray([2, 0]))
    np.testing.assert_allclose(np.sum(diagonal, axis=1), [6.0, 6.0])
    np.testing.assert_array_equal(
        diagonal,
        np.asarray([[0.0, 0.0, 6.0], [6.0, 0.0, 0.0]]),
    )


def test_row_sum_matched_diagonal_requires_nominal_source() -> None:
    with pytest.raises(ValueError, match="no matching nominal source"):
        row_sum_matched_diagonal_response(
            response=np.ones((1, 2)),
            output_band_ids=np.asarray([4]),
            source_band_ids=np.asarray([1, 2]),
        )
