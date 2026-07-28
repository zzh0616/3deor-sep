from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ops_scripts.build_chips_visibility_bank import (
    _select_row_sample,
    _write_oskar_config,
)


def test_gaussian_oskar_config_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / "simulation.ini"
        _write_oskar_config(
            config,
            osm=Path("/sky/eor.osm"),
            ms=Path("/output/eor.ms"),
            frequency_mhz=119.4,
            telescope_dir=Path("/telescope/ska1_low.tm"),
            station_type="gaussian",
            gaussian_fwhm_deg=5.0,
            gaussian_reference_frequency_mhz=119.4,
        )
        text = config.read_text(encoding="utf-8")
    assert "station_type=Gaussian beam" in text
    assert "gaussian_beam/ref_freq_hz=119400000.0" in text
    assert "gaussian_beam/fwhm_deg=5" in text
    assert "normalise_beams_at_phase_centre=true" in text
    assert "force_polarised_ms=true" in text


def test_aperture_array_oskar_config_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / "simulation.ini"
        _write_oskar_config(
            config,
            osm=Path("/sky/eor.osm"),
            ms=Path("/output/eor.ms"),
            frequency_mhz=119.4,
            telescope_dir=Path("/telescope/ska1_low.tm"),
            station_type="aperture_array",
            gaussian_fwhm_deg=5.0,
            gaussian_reference_frequency_mhz=119.4,
        )
        text = config.read_text(encoding="utf-8")
    assert "station_type=Aperture array" in text
    assert "pol_mode=Full" in text


def test_row_sample_preserves_antenna_alignment() -> None:
    count = 12
    antenna1 = np.arange(count, dtype=np.int32)
    antenna2 = antenna1 + 100
    sample = _select_row_sample(
        row_indices=np.arange(count, dtype=np.int64),
        uvw_m=np.column_stack(
            (
                np.linspace(100.0, 1000.0, count),
                np.zeros(count),
                np.zeros(count),
            )
        ),
        times=np.repeat(np.arange(3), 4),
        fg=np.arange(count, dtype=np.complex64),
        eor=(2 * np.arange(count)).astype(np.complex64),
        antenna1=antenna1,
        antenna2=antenna2,
        reference_frequency_mhz=120.0,
        min_uv_lambda=30.0,
        max_uv_lambda=500.0,
        bins=2,
        rows_per_bin=3,
    )
    rows = sample["sample_row_indices"]
    np.testing.assert_array_equal(sample["sample_antenna1"], antenna1[rows])
    np.testing.assert_array_equal(sample["sample_antenna2"], antenna2[rows])
