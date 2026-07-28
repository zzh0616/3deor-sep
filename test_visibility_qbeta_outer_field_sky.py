import numpy as np
import pytest

from ops_scripts.build_visibility_qbeta_outer_field_sky import (
    block_average_square,
    select_bright_outer_sources,
)
from ops_scripts.evaluate_visibility_qbeta_outer_field import (
    _coarse_estimate,
    _foreground_effect,
)


def test_block_average_square_conserves_flux_with_pixel_area() -> None:
    plane = np.arange(64, dtype=np.float64).reshape(8, 8)
    coarse = block_average_square(plane, 2)
    assert np.isclose(np.sum(coarse) * 4.0, np.sum(plane))


def test_outer_source_selection_keeps_regions_and_global_brightest() -> None:
    score = np.asarray([10.0, 9.0, 1.0, 0.5, 0.4, 0.3])
    regions = np.asarray([0, 0, 1, 1, 2, 2])
    selected = select_bright_outer_sources(
        score=score,
        region_ids=regions,
        maximum_sources=2,
        minimum_sources_per_region=1,
    )
    assert np.array_equal(selected, np.asarray([0, 1, 2, 4]))


def test_outer_source_selection_rejects_negative_scores() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        select_bright_outer_sources(
            score=np.asarray([1.0, -1.0]),
            region_ids=np.asarray([0, 1]),
            maximum_sources=1,
            minimum_sources_per_region=0,
        )


def test_foreground_effect_uses_frozen_response_normalization() -> None:
    response = np.asarray([[2.0, 0.0], [0.0, 4.0]])
    measured = _foreground_effect(
        q=np.asarray([0.2, -0.8]),
        response=response,
        target=np.asarray([1.0, 2.0]),
        selected_positions=np.asarray([0, 1]),
        relative_response=np.asarray([0.5, 1.0]),
    )
    assert np.isclose(measured["integrated_signed_ratio"], -0.06)
    assert np.isclose(measured["integrated_absolute_ratio"], 0.1)
    assert np.isclose(measured["maximum_absolute_window_ratio"], 0.1)
    assert np.isclose(measured["median_absolute_window_ratio"], 0.1)
    assert np.isclose(measured["p90_absolute_window_ratio"], 0.1)
    assert measured["above_10pct_count"] == 0
    assert measured["above_20pct_count"] == 0


def test_coarse_estimate_applies_frozen_transform_and_response() -> None:
    q = np.asarray([[2.0, 8.0], [6.0, 4.0]])
    transform = np.asarray([[0.5, 0.25], [0.0, 1.0]])
    response = np.asarray([[1.0, 2.0], [0.0, 4.0]])
    estimate = _coarse_estimate(
        q,
        transform=transform,
        response=response,
    )
    assert np.allclose(
        estimate,
        np.asarray([[1.0, 2.0], [4.0 / 3.0, 1.0]]),
    )
