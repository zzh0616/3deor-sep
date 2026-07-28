from __future__ import annotations

import numpy as np

from ops_scripts.evaluate_oskar_aperture_beam_closure import (
    _row_time_indices,
)


def test_row_time_indices_follow_selected_visibility_rows() -> None:
    times = np.asarray([10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    selected = np.asarray([4, 0, 3])
    np.testing.assert_array_equal(
        _row_time_indices(times, selected, 3),
        np.asarray([2, 0, 1]),
    )
