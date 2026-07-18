#!/usr/bin/env python3
"""Resolve a PS2D v2 config into physical geometry and a frozen contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from ps2d_v2 import (
    EoRWindowSpec,
    ModeFirstAnalysisContract,
    build_mode_first_analysis_contract,
    linear_kperp_edges,
)


@dataclass
class ResolvedModeFirstAnalysis:
    config: dict[str, Any]
    geometry: dict[str, Any]
    window_spec: EoRWindowSpec
    contract: ModeFirstAnalysisContract


def _wedge_slope(
    cosmology: FlatLambdaCDM,
    redshift: float,
    theta_deg: float,
) -> float:
    distance = cosmology.comoving_transverse_distance(redshift).to_value(u.Mpc)
    expansion = cosmology.H(redshift).to_value(u.km / u.s / u.Mpc)
    return float(
        distance
        * expansion
        / (299792.458 * (1.0 + redshift))
        * math.sin(math.radians(float(theta_deg)))
    )


def resolve_mode_first_geometry(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "ps2d_v2_mode_first_config" or int(
        config.get("schema_version", -1)
    ) != 2:
        raise ValueError("Expected a PS2D v2 mode-first config")
    frequencies = np.asarray(config["frequencies_mhz"], dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("At least two one-dimensional frequencies are required")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.diff(frequencies) > 0.0):
        raise ValueError("frequencies_mhz must be finite and strictly increasing")

    cosmology_config = config["cosmology"]
    cosmology = FlatLambdaCDM(
        H0=float(cosmology_config["H0_km_s_mpc"]),
        Om0=float(cosmology_config["Om0"]),
        Tcmb0=0.0,
    )
    rest_frequency = float(config["rest_frequency_mhz"])
    reference_frequency = float(config["reference_frequency_mhz"])
    redshifts = rest_frequency / frequencies - 1.0
    distances = cosmology.comoving_distance(redshifts).to_value(u.Mpc)
    spacing_samples = np.abs(np.diff(distances))
    radial_spacing = float(np.mean(spacing_samples))
    max_spacing_deviation = float(
        np.max(np.abs(spacing_samples / radial_spacing - 1.0))
    )

    reference_redshift = rest_frequency / reference_frequency - 1.0
    transverse_distance = float(
        cosmology.comoving_transverse_distance(reference_redshift).to_value(u.Mpc)
    )
    image = config["image_geometry"]
    pixel_angle_rad = math.radians(float(image["spatial_pixel_arcsec"]) / 3600.0)
    spatial_spacing = transverse_distance * pixel_angle_rad
    source_corner_angle_deg = math.sqrt(2.0) * (
        float(image["source_image_size"])
        * float(image["spatial_pixel_arcsec"])
        / 2.0
        / 3600.0
    )

    instrument = config["instrument_support"]
    uv_min = float(instrument["min_uv_lambda"])
    uv_max = float(instrument["max_uv_lambda"])
    if not (0.0 <= uv_min < uv_max):
        raise ValueError("UV support must be finite, non-negative, and increasing")
    kperp_uv_min = 2.0 * math.pi * uv_min / transverse_distance
    kperp_uv_max = 2.0 * math.pi * uv_max / transverse_distance

    window = config["eor_window"]
    profile = str(window["profile"]).strip().lower()
    if profile != "finite_source_patch_corner":
        raise ValueError("v2 pilot currently requires finite_source_patch_corner")
    patch_slope = _wedge_slope(
        cosmology, reference_redshift, source_corner_angle_deg
    )
    horizon_slope = _wedge_slope(cosmology, reference_redshift, 90.0)
    h = float(cosmology_config["H0_km_s_mpc"]) / 100.0
    return {
        "frequencies_mhz": frequencies,
        "redshifts": redshifts,
        "comoving_distances_mpc": distances,
        "radial_spacing_samples_mpc": spacing_samples,
        "radial_spacing_mpc": radial_spacing,
        "radial_spacing_max_relative_deviation": max_spacing_deviation,
        "reference_frequency_mhz": reference_frequency,
        "reference_redshift": reference_redshift,
        "transverse_distance_mpc": transverse_distance,
        "spatial_spacing_mpc": spatial_spacing,
        "source_corner_angle_deg": source_corner_angle_deg,
        "patch_wedge_slope": patch_slope,
        "horizon_wedge_slope": horizon_slope,
        "h": h,
        "kpar_floor_mpc_inv": float(window["kpar_floor_h_mpc"]) * h,
        "wedge_buffer_mpc_inv": float(window["suprahorizon_buffer_h_mpc"]) * h,
        "kperp_uv_min_mpc_inv": kperp_uv_min,
        "kperp_uv_max_mpc_inv": kperp_uv_max,
    }


def resolve_mode_first_analysis(config: dict[str, Any]) -> ResolvedModeFirstAnalysis:
    geometry = resolve_mode_first_geometry(config)
    frequencies = np.asarray(geometry["frequencies_mhz"], dtype=np.float64)
    image = config["image_geometry"]
    analysis = config["analysis"]
    crop_size = int(image["eval_crop_size"])
    shape = (int(frequencies.size), crop_size, crop_size)
    dx_mpc = float(geometry["spatial_spacing_mpc"])
    dy_mpc = dx_mpc
    dpar_mpc = float(geometry["radial_spacing_mpc"])
    kx_axis = 2.0 * math.pi * np.fft.fftfreq(crop_size, d=dx_mpc)
    ky_axis = 2.0 * math.pi * np.fft.fftfreq(crop_size, d=dy_mpc)
    transverse_circle_max = min(
        float(np.max(np.abs(kx_axis))), float(np.max(np.abs(ky_axis)))
    )
    window_kperp_min = max(0.0, float(geometry["kperp_uv_min_mpc_inv"]))
    window_kperp_max = min(
        transverse_circle_max, float(geometry["kperp_uv_max_mpc_inv"])
    )
    if window_kperp_max <= window_kperp_min:
        raise ValueError("UV support does not overlap the transverse FFT circle")
    full_edges = linear_kperp_edges(
        0.0, transverse_circle_max, int(analysis["full_kperp_bins"])
    )
    window_edges = linear_kperp_edges(
        window_kperp_min,
        window_kperp_max,
        int(analysis["window_kperp_bins"]),
    )
    window_spec = EoRWindowSpec(
        kpar_min=float(geometry["kpar_floor_mpc_inv"]),
        wedge_slope=float(geometry["patch_wedge_slope"]),
        wedge_intercept=float(geometry["wedge_buffer_mpc_inv"]),
        kperp_min=window_kperp_min,
        kperp_max=window_kperp_max,
        exclude_exact_dc=bool(config["eor_window"]["exclude_exact_dc"]),
    )
    contract = build_mode_first_analysis_contract(
        shape,
        dx_mpc=dx_mpc,
        dy_mpc=dy_mpc,
        dpar_mpc=dpar_mpc,
        full_kperp_edges=full_edges,
        window_kperp_edges=window_edges,
        window_spec=window_spec,
        radial_nyquist_policy=str(analysis["radial_nyquist_policy"]),
        demean_mode=str(analysis["demean_mode"]),
        radial_taper=str(analysis["radial_taper"]),
        spatial_taper=str(analysis["spatial_taper"]),
    )
    expected_hash = config.get("frozen_analysis_contract_sha256")
    if expected_hash is not None and str(expected_hash) != contract.analysis_contract_sha256:
        raise ValueError(
            "Resolved analysis contract does not match frozen_analysis_contract_sha256"
        )
    geometry = {
        **geometry,
        "cube_shape": list(shape),
        "transverse_circle_max_mpc_inv": transverse_circle_max,
        "full_kperp_bins": int(analysis["full_kperp_bins"]),
        "window_kperp_bins": int(analysis["window_kperp_bins"]),
        "window_kperp_range_mpc_inv": [window_kperp_min, window_kperp_max],
        "kpar_values_mpc_inv": contract.full_layout.kpar_values,
        "kpar_display_edges_mpc_inv": contract.full_layout.kpar_edges,
        "radial_nyquist_policy": contract.full_layout.radial_nyquist_policy,
        "bin_edge_convention": "left_closed_right_open_last_right_inclusive",
    }
    return ResolvedModeFirstAnalysis(
        config=config,
        geometry=geometry,
        window_spec=window_spec,
        contract=contract,
    )
