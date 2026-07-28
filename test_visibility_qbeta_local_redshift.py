import json

import numpy as np
import pytest

from ps2d_v2_config import resolve_mode_first_analysis
from visibility_qbeta_local_redshift import (
    build_local_redshift_windows,
    freeze_frequency_view_config,
    freeze_local_config,
    frequency_subset_indices,
)


def test_frequency_subset_indices_accepts_ordered_view() -> None:
    available = np.arange(100.0, 101.0, 0.1)
    requested = available[2:8]
    assert np.array_equal(
        frequency_subset_indices(available, requested, atol=1e-9),
        np.arange(2, 8),
    )
    with pytest.raises(ValueError, match="not a subset"):
        frequency_subset_indices(
            available, requested + 0.025, atol=1e-9
        )


def test_local_windows_cover_target_with_expected_overlap() -> None:
    frequencies = np.arange(108.3, 121.1, 0.1)
    windows = build_local_redshift_windows(
        frequencies,
        input_channel_count=64,
        analysis_channel_count=32,
        stride_channels=16,
        target_start=32,
        target_stop=96,
    )
    assert len(windows) == 5
    assert [window.input_start for window in windows] == [0, 16, 32, 48, 64]
    assert [window.analysis_start for window in windows] == [
        16,
        32,
        48,
        64,
        80,
    ]
    assert all(
        first.analysis_stop - second.analysis_start == 16
        for first, second in zip(windows[:-1], windows[1:], strict=True)
    )


def test_freeze_local_config_round_trips() -> None:
    template = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "configs"
            / "ps2d_v2_32central_isobeam_patch.json"
        ).read_text(encoding="utf-8")
    )
    frequencies = 111.5 + 0.1 * np.arange(32)
    config = freeze_local_config(
        template,
        frequencies_mhz=frequencies,
        reference_frequency_mhz=float(np.mean(frequencies)),
        status="unit_test",
    )
    resolved = resolve_mode_first_analysis(config)
    assert (
        config["frozen_analysis_contract_sha256"]
        == resolved.contract.analysis_contract_sha256
    )
    assert config["legacy_reproduction"]["status"] == "unit_test"


def test_freeze_local_config_recomputes_energy_for_shorter_band() -> None:
    template = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "configs"
            / "ps2d_v2_32central_isobeam_patch.json"
        ).read_text(encoding="utf-8")
    )
    frequencies = 113.9 + 0.1 * np.arange(16)
    config = freeze_local_config(
        template,
        frequencies_mhz=frequencies,
        reference_frequency_mhz=114.65,
        status="short_band",
    )
    resolved = resolve_mode_first_analysis(config)
    assert np.isclose(
        config["frozen_analysis_window_energy"],
        resolved.contract.window_energy,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        config["frozen_analysis_window_energy"]
        != template["frozen_analysis_window_energy"]
    )


def test_freeze_frequency_view_config_has_stable_identity() -> None:
    template = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "configs"
            / "ps2d_v2_64wide_isobeam_patch.json"
        ).read_text(encoding="utf-8")
    )
    frequencies = 108.3 + 0.1 * np.arange(64)
    first = freeze_frequency_view_config(
        template,
        frequencies_mhz=frequencies,
        reference_frequency_mhz=float(np.mean(frequencies)),
        status="unit_test",
    )
    second = freeze_frequency_view_config(
        template,
        frequencies_mhz=frequencies,
        reference_frequency_mhz=float(np.mean(frequencies)),
        status="unit_test",
    )
    assert (
        first["frequency_view_contract_sha256"]
        == second["frequency_view_contract_sha256"]
    )
    assert "frozen_analysis_contract_sha256" not in first


def test_common_geometry_gives_common_analysis_contract() -> None:
    path = __import__("pathlib").Path(__file__).parent / "configs"
    template = json.loads(
        (path / "ps2d_v2_32central_isobeam_patch.json").read_text(
            encoding="utf-8"
        )
    )
    full = json.loads(
        (path / "ps2d_v2_128wide_isobeam_patch.json").read_text(
            encoding="utf-8"
        )
    )
    first = freeze_local_config(
        template,
        frequencies_mhz=109.9 + 0.1 * np.arange(32),
        reference_frequency_mhz=float(full["reference_frequency_mhz"]),
        status="first",
        frozen_geometry=full["frozen_geometry"],
    )
    second = freeze_local_config(
        template,
        frequencies_mhz=116.3 + 0.1 * np.arange(32),
        reference_frequency_mhz=float(full["reference_frequency_mhz"]),
        status="second",
        frozen_geometry=full["frozen_geometry"],
    )
    assert (
        first["frozen_analysis_contract_sha256"]
        == second["frozen_analysis_contract_sha256"]
    )
