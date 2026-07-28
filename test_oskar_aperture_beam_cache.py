from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ops_scripts.build_oskar_aperture_beam_cache import (
    parse_oskar_auto_power_i,
)


def test_parse_oskar_auto_power_chunk_order() -> None:
    header = (
        "# Beam pixel list for station 0\n"
        "# Number of pixel chunks: 2\n"
        "# Number of times (output): 2\n"
        "# Number of channels (output): 1\n"
        "# Maximum pixel chunk size: 3\n"
        "# Total number of pixels: 5\n"
    )
    first = np.asarray([[1, 2, 3], [4, 5, 6]]).reshape(-1)
    second = np.asarray([[7, 8], [9, 10]]).reshape(-1)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "beam.txt"
        path.write_text(
            header
            + "\n".join(str(value) for value in np.concatenate((first, second)))
            + "\n",
            encoding="utf-8",
        )
        parsed = parse_oskar_auto_power_i(path)
    np.testing.assert_array_equal(
        parsed[:, 0],
        np.asarray([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]]),
    )
