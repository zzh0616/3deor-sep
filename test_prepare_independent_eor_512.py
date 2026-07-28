import numpy as np
import pytest
from astropy.io import fits

from ops_scripts.prepare_independent_eor_512 import (
    central_block_average,
    reference_transform_metrics,
)


def test_central_block_average() -> None:
    plane = np.arange(64, dtype=np.float64).reshape(8, 8)
    measured = central_block_average(
        plane, crop_size=4, downsample=2
    )
    expected = plane[2:6, 2:6].reshape(2, 2, 2, 2).mean(axis=(1, 3))
    assert np.array_equal(measured, expected)


def test_central_block_average_rejects_nondivisible_crop() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        central_block_average(
            np.ones((8, 8)), crop_size=5, downsample=2
        )


def test_reference_transform_metrics_streams_exact_contract(tmp_path) -> None:
    source = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8)
    reference = np.stack(
        [
            central_block_average(plane, crop_size=4, downsample=2)
            for plane in source
        ]
    )
    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    fits.PrimaryHDU(source).writeto(input_path)
    fits.PrimaryHDU(reference).writeto(output_path)

    relative_l2, maximum_absolute = reference_transform_metrics(
        input_path,
        output_path,
        crop_size=4,
        downsample=2,
    )

    assert relative_l2 == 0.0
    assert maximum_absolute == 0.0
