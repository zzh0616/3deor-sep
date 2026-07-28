from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ps2d_v2_config import resolve_mode_first_analysis
from ps2d_v2 import build_mode_first_analysis_contract


PROJECT_DIR = Path(__file__).resolve().parent


def _visibility_config() -> dict:
    path = (
        PROJECT_DIR
        / "configs"
        / "ps2d_v2_8contiguous_visibility_qbeta.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_geometry_preserves_contract_across_live_cosmology_change() -> None:
    config = _visibility_config()
    reference = resolve_mode_first_analysis(config)
    changed = copy.deepcopy(config)
    changed["cosmology"]["Om0"] = 0.5
    resolved = resolve_mode_first_analysis(changed)
    assert (
        resolved.contract.analysis_contract_sha256
        == reference.contract.analysis_contract_sha256
    )
    assert resolved.geometry["frozen_geometry_applied"]
    assert max(
        resolved.geometry[
            "frozen_geometry_live_relative_difference"
        ].values()
    ) > 0.0


def test_frozen_geometry_requires_complete_values() -> None:
    config = _visibility_config()
    del config["frozen_geometry"]["radial_spacing_mpc"]
    with pytest.raises(ValueError, match="radial_spacing_mpc"):
        resolve_mode_first_analysis(config)


def test_frozen_window_energy_rejects_a_material_difference() -> None:
    config = _visibility_config()
    resolved = resolve_mode_first_analysis(config)
    layout = resolved.contract.full_layout
    with pytest.raises(ValueError, match="window energy"):
        build_mode_first_analysis_contract(
            layout.cube_shape,
            dx_mpc=layout.dx_mpc,
            dy_mpc=layout.dy_mpc,
            dpar_mpc=layout.dpar_mpc,
            full_kperp_edges=layout.kperp_edges,
            window_kperp_edges=resolved.contract.window_layout.kperp_edges,
            window_spec=resolved.window_spec,
            radial_nyquist_policy=layout.radial_nyquist_policy,
            demean_mode=resolved.contract.demean_mode,
            radial_taper=resolved.contract.radial_taper,
            spatial_taper=resolved.contract.spatial_taper,
            window_energy_override=0.5,
        )


def test_explicit_foreground_support_moves_wedge_and_delay_alias() -> None:
    config = _visibility_config()
    baseline = resolve_mode_first_analysis(config)
    changed = copy.deepcopy(config)
    changed.pop("frozen_analysis_contract_sha256")
    changed.pop("frozen_analysis_window_energy")
    changed.pop("frozen_geometry")
    changed["eor_window"]["foreground_support_angle_deg"] = 6.45
    resolved = resolve_mode_first_analysis(changed)
    assert resolved.geometry["source_image_corner_angle_deg"] == pytest.approx(
        baseline.geometry["source_image_corner_angle_deg"]
    )
    assert resolved.geometry["foreground_support_angle_deg"] == 6.45
    assert resolved.geometry["source_corner_angle_deg"] == 6.45
    assert (
        resolved.geometry["patch_wedge_slope"]
        > baseline.geometry["patch_wedge_slope"]
    )


def test_foreground_support_must_cover_source_image() -> None:
    config = _visibility_config()
    config["eor_window"]["foreground_support_angle_deg"] = 1.0
    with pytest.raises(ValueError, match="must cover"):
        resolve_mode_first_analysis(config)
