from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ops_scripts.build_oskar_aperture_row_beam_cache import (
    _helper_matches_build_contract,
    _oskar_operator_settings,
    _sha256,
    _validate_finite_file,
)
from visibility_primary_beam import (
    direction_cosine_geometry_sha256,
)


def test_direction_hash_contract_matches_evaluator_and_tracks_order() -> None:
    l_cosine = np.asarray([0.0, 0.1, -0.2], dtype=np.float64)
    m_cosine = np.asarray([0.2, -0.1, 0.0], dtype=np.float64)
    n_cosine = np.sqrt(1.0 - l_cosine**2 - m_cosine**2)
    expected = direction_cosine_geometry_sha256(
        l_cosine=l_cosine,
        m_cosine=m_cosine,
        n_minus_one=n_cosine - 1.0,
    )
    assert (
        direction_cosine_geometry_sha256(
            l_cosine=l_cosine[::-1],
            m_cosine=m_cosine[::-1],
            n_minus_one=n_cosine[::-1] - 1.0,
        )
        != expected
    )


def test_oskar_operator_settings_are_read_from_ini() -> None:
    content = """
[observation]
start_frequency_hz=119400000
phase_centre_dec_deg=-27

[interferometer]
channel_bandwidth_hz=100000
time_average_sec=10
"""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "simulation.ini"
        path.write_text(content, encoding="utf-8")
        settings = _oskar_operator_settings(path)
    assert settings == {
        "frequency_hz": 119400000.0,
        "phase_centre_dec_deg": -27.0,
        "channel_bandwidth_hz": 100000.0,
        "time_average_sec": 10.0,
    }


def test_helper_build_contract_rejects_missing_or_changed_binary() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        helper = root / "helper"
        source = root / "helper.cc"
        source.write_text("int main() { return 0; }\n", encoding="utf-8")
        helper.write_bytes(b"binary")
        assert not _helper_matches_build_contract(
            helper=helper,
            source=source,
            prefix=root / "oskar",
            cxx="g++",
            compiler_library_dirs=[root / "runtime"],
        )
        (root / "helper.build.json").write_text(
            json.dumps(
                {
                    "schema": "oskar_aperture_row_beam_helper_build",
                    "schema_version": 1,
                    "source_sha256": _sha256(source),
                    "oskar_prefix": str(root / "oskar"),
                    "cxx": "g++",
                    "compiler_library_dirs": [str(root / "runtime")],
                    "binary_sha256": _sha256(helper),
                }
            ),
            encoding="utf-8",
        )
        assert _helper_matches_build_contract(
            helper=helper,
            source=source,
            prefix=root / "oskar",
            cxx="g++",
            compiler_library_dirs=[root / "runtime"],
        )
        helper.write_bytes(b"changed")
        assert not _helper_matches_build_contract(
            helper=helper,
            source=source,
            prefix=root / "oskar",
            cxx="g++",
            compiler_library_dirs=[root / "runtime"],
        )
def test_validate_row_beam_cache_reports_amplitude_range() -> None:
    values = np.asarray(
        [[1.0 + 0.0j, 0.5j, 0.0], [0.25, -2.0j, 0.75]],
        dtype=np.complex64,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "beam.bin"
        values.tofile(path)
        minimum, maximum = _validate_finite_file(
            path, shape=values.shape, source_chunk=2
        )
    assert minimum == 0.0
    assert maximum == 2.0


def test_validate_row_beam_cache_rejects_nonfinite_values() -> None:
    values = np.asarray([[1.0, np.nan]], dtype=np.complex64)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "beam.bin"
        values.tofile(path)
        with pytest.raises(ValueError, match="non-finite"):
            _validate_finite_file(path, shape=values.shape, source_chunk=2)
