#!/usr/bin/env python3
"""Estimator-facing adapters for the PS2D v2 mode-first contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from ps2d_v2 import (
    CylindricalModeLayout,
    ModeFirstAnalysisContract,
    build_cylindrical_mode_layout,
    linear_kperp_edges,
)
from ps2d_v2_config import ResolvedModeFirstAnalysis, resolve_mode_first_analysis


@dataclass
class EstimatorBandGeometry:
    cube_shape: tuple[int, int, int]
    mode_indices: np.ndarray
    mode_bands: np.ndarray
    cube_mode_bands: np.ndarray
    counts: np.ndarray
    active_layout_bands: np.ndarray
    active_kperp_indices: np.ndarray
    active_kpar_indices: np.ndarray
    kperp_edges: np.ndarray
    kperp_centers: np.ndarray
    kpar_values: np.ndarray
    kpar_display_edges: np.ndarray

    @property
    def band_count(self) -> int:
        return int(self.active_layout_bands.size)


@dataclass
class ModeFirstEstimatorContract:
    resolved: ResolvedModeFirstAnalysis
    full_geometry: EstimatorBandGeometry
    science_geometry: EstimatorBandGeometry
    calibration_source_geometry: EstimatorBandGeometry
    calibration_source_layout: CylindricalModeLayout
    calibration_source_kind: np.ndarray
    calibration_source_parent_bands: np.ndarray
    calibration_science_band_count: int
    control_mode_indices: np.ndarray
    guard_mode_indices: np.ndarray
    estimator_contract_sha256: str

    @property
    def analysis(self) -> ModeFirstAnalysisContract:
        return self.resolved.contract


def _geometry_from_layout(
    layout: CylindricalModeLayout,
    *,
    selected: bool,
) -> EstimatorBandGeometry:
    if selected:
        indices = np.asarray(layout.selected_mode_indices, dtype=np.int64)
        layout_bands = np.asarray(layout.selected_mode_bands, dtype=np.int64)
        layout_counts = layout.selected_fft_mode_counts.reshape(-1)
    else:
        indices = np.asarray(layout.full_mode_indices, dtype=np.int64)
        layout_bands = np.asarray(layout.full_mode_bands, dtype=np.int64)
        layout_counts = layout.full_fft_mode_counts.reshape(-1)
    active = np.flatnonzero(layout_counts > 0).astype(np.int64)
    lookup = np.full((layout.band_count,), -1, dtype=np.int64)
    lookup[active] = np.arange(active.size, dtype=np.int64)
    local_bands = lookup[layout_bands]
    if np.any(local_bands < 0):
        raise AssertionError("A selected mode maps to an empty layout band")
    cube_bands = np.full((int(np.prod(layout.cube_shape)),), -1, dtype=np.int64)
    cube_bands[indices] = local_bands
    nkpar = int(layout.kpar_values.size)
    return EstimatorBandGeometry(
        cube_shape=layout.cube_shape,
        mode_indices=indices,
        mode_bands=local_bands,
        cube_mode_bands=cube_bands.reshape(layout.cube_shape),
        counts=np.asarray(layout_counts[active], dtype=np.float64),
        active_layout_bands=active,
        active_kperp_indices=active // nkpar,
        active_kpar_indices=active % nkpar,
        kperp_edges=np.asarray(layout.kperp_edges, dtype=np.float64),
        kperp_centers=np.asarray(layout.kperp_centers, dtype=np.float64),
        kpar_values=np.asarray(layout.kpar_values, dtype=np.float64),
        kpar_display_edges=np.asarray(layout.kpar_edges, dtype=np.float64),
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _build_estimator_hash(
    *,
    analysis: ModeFirstAnalysisContract,
    source_layout: CylindricalModeLayout,
    source_geometry: EstimatorBandGeometry,
    control_mode_indices: np.ndarray,
    guard_mode_indices: np.ndarray,
) -> str:
    payload = {
        "schema": "ps2d_v2_estimator_contract",
        "analysis_contract_sha256": analysis.analysis_contract_sha256,
        "calibration_source": {
            "cube_shape": source_layout.cube_shape,
            "kperp_edges": source_layout.kperp_edges.tolist(),
            "kpar_values": source_layout.kpar_values.tolist(),
            "radial_nyquist_policy": source_layout.radial_nyquist_policy,
            "mode_indices_sha256": _array_sha256(source_layout.full_mode_indices),
            "mode_bands_sha256": _array_sha256(source_layout.full_mode_bands),
            "composite_cube_bands_sha256": _array_sha256(
                source_geometry.cube_mode_bands
            ),
        },
        "control_mode_indices_sha256": _array_sha256(control_mode_indices),
        "guard_mode_indices_sha256": _array_sha256(guard_mode_indices),
        "science_mode_indices_sha256": _array_sha256(
            analysis.window_layout.selected_mode_indices
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _composite_calibration_source_geometry(
    base: EstimatorBandGeometry,
    science: EstimatorBandGeometry,
    *,
    control_kpar_indices: np.ndarray,
    guard_kpar_indices: np.ndarray,
    analysis_kpar_count: int,
) -> tuple[EstimatorBandGeometry, np.ndarray, np.ndarray]:
    base_cube = np.asarray(base.cube_mode_bands, dtype=np.int64).reshape(-1)
    science_cube = np.asarray(science.cube_mode_bands, dtype=np.int64).reshape(-1)
    if np.any(base_cube < 0):
        raise ValueError("Base calibration geometry does not cover the full FFT cube")
    science_count = int(science.band_count)
    complement_parents = np.unique(base_cube[science_cube < 0])
    parent_lookup = np.full((base.band_count,), -1, dtype=np.int64)
    parent_lookup[complement_parents] = np.arange(
        complement_parents.size, dtype=np.int64
    )
    composite = np.empty_like(base_cube)
    in_science = science_cube >= 0
    composite[in_science] = science_cube[in_science]
    composite[~in_science] = (
        science_count + parent_lookup[base_cube[~in_science]]
    )
    if np.any(composite < 0):
        raise AssertionError("A calibration source mode has no composite band")
    total_count = science_count + int(complement_parents.size)
    counts = np.bincount(composite, minlength=total_count).astype(np.float64)
    if np.any(counts <= 0.0):
        raise AssertionError("Composite calibration source contains an empty band")
    parent_bands = np.concatenate(
        (science.active_layout_bands, base.active_layout_bands[complement_parents])
    ).astype(np.int64, copy=False)
    complement_kpar = base.active_kpar_indices[complement_parents]
    complement_kinds = np.full(
        (complement_parents.size,), "target_outside_window", dtype="<U24"
    )
    complement_kinds[np.isin(complement_kpar, control_kpar_indices)] = "control"
    complement_kinds[np.isin(complement_kpar, guard_kpar_indices)] = "guard"
    complement_kinds[complement_kpar >= int(analysis_kpar_count)] = "radial_nyquist"
    kinds = np.concatenate(
        (
            np.full((science_count,), "science", dtype="<U24"),
            complement_kinds,
        )
    )
    geometry = EstimatorBandGeometry(
        cube_shape=base.cube_shape,
        mode_indices=np.arange(base_cube.size, dtype=np.int64),
        mode_bands=composite,
        cube_mode_bands=composite.reshape(base.cube_shape),
        counts=counts,
        active_layout_bands=np.arange(total_count, dtype=np.int64),
        active_kperp_indices=np.concatenate(
            (
                science.active_kperp_indices,
                base.active_kperp_indices[complement_parents],
            )
        ),
        active_kpar_indices=np.concatenate(
            (
                science.active_kpar_indices,
                base.active_kpar_indices[complement_parents],
            )
        ),
        kperp_edges=base.kperp_edges,
        kperp_centers=base.kperp_centers,
        kpar_values=base.kpar_values,
        kpar_display_edges=base.kpar_display_edges,
    )
    return geometry, kinds, parent_bands


def build_mode_first_estimator_contract_from_analysis(
    resolved: ResolvedModeFirstAnalysis,
) -> ModeFirstEstimatorContract:
    config = resolved.config
    analysis = resolved.contract
    partition = config["estimator_partitions"]
    control_kpar = np.asarray(partition["control_kpar_indices"], dtype=np.int64)
    guard_kpar = np.asarray(partition["guard_kpar_indices"], dtype=np.int64)
    if control_kpar.size == 0:
        raise ValueError("At least one control kpar index is required")
    if np.intersect1d(control_kpar, guard_kpar).size:
        raise ValueError("Control and guard kpar partitions overlap")
    full_layout = analysis.full_layout
    nkpar = int(full_layout.kpar_values.size)
    if np.any(control_kpar < 0) or np.any(control_kpar >= nkpar):
        raise ValueError("Control kpar index is outside the full layout")
    if np.any(guard_kpar < 0) or np.any(guard_kpar >= nkpar):
        raise ValueError("Guard kpar index is outside the full layout")
    full_mode_kpar = full_layout.full_mode_bands % nkpar
    control_mask = np.isin(full_mode_kpar, control_kpar)
    guard_mask = np.isin(full_mode_kpar, guard_kpar)
    control_indices = np.asarray(
        full_layout.full_mode_indices[control_mask], dtype=np.int64
    )
    guard_indices = np.asarray(full_layout.full_mode_indices[guard_mask], dtype=np.int64)
    science_indices = np.asarray(
        analysis.window_layout.selected_mode_indices, dtype=np.int64
    )
    if np.intersect1d(control_indices, science_indices).size:
        raise ValueError("Control and science Fourier modes overlap")
    if np.intersect1d(guard_indices, science_indices).size:
        raise ValueError("Guard and science Fourier modes overlap")

    source_config = config["calibration_source"]
    if str(source_config["transverse_support"]).strip().lower() != "full_fft_square":
        raise ValueError("Calibration source must cover the full FFT square")
    nf, ny, nx = analysis.full_layout.cube_shape
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=analysis.full_layout.dy_mpc)
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=analysis.full_layout.dx_mpc)
    corner_max = float(
        math.sqrt(float(np.max(np.abs(ky))) ** 2 + float(np.max(np.abs(kx))) ** 2)
    )
    source_edges = linear_kperp_edges(
        0.0, corner_max, int(source_config["kperp_bins"])
    )
    source_layout = build_cylindrical_mode_layout(
        (nf, ny, nx),
        dx_mpc=analysis.full_layout.dx_mpc,
        dy_mpc=analysis.full_layout.dy_mpc,
        dpar_mpc=analysis.full_layout.dpar_mpc,
        kperp_edges=source_edges,
        radial_nyquist_policy=str(source_config["radial_nyquist_policy"]),
        transverse_circle_max=corner_max,
    )
    if source_layout.full_mode_indices.size != nf * ny * nx:
        raise ValueError("Calibration source layout does not cover every FFT mode")
    base_source_geometry = _geometry_from_layout(source_layout, selected=False)
    science_geometry = _geometry_from_layout(
        analysis.window_layout, selected=True
    )
    source_geometry, source_kind, source_parents = (
        _composite_calibration_source_geometry(
            base_source_geometry,
            science_geometry,
            control_kpar_indices=control_kpar,
            guard_kpar_indices=guard_kpar,
            analysis_kpar_count=nkpar,
        )
    )
    estimator_hash = _build_estimator_hash(
        analysis=analysis,
        source_layout=source_layout,
        source_geometry=source_geometry,
        control_mode_indices=control_indices,
        guard_mode_indices=guard_indices,
    )
    return ModeFirstEstimatorContract(
        resolved=resolved,
        full_geometry=_geometry_from_layout(full_layout, selected=False),
        science_geometry=science_geometry,
        calibration_source_geometry=source_geometry,
        calibration_source_layout=source_layout,
        calibration_source_kind=source_kind,
        calibration_source_parent_bands=source_parents,
        calibration_science_band_count=science_geometry.band_count,
        control_mode_indices=control_indices,
        guard_mode_indices=guard_indices,
        estimator_contract_sha256=estimator_hash,
    )


def build_mode_first_estimator_contract(
    config: dict[str, Any],
) -> ModeFirstEstimatorContract:
    return build_mode_first_estimator_contract_from_analysis(
        resolve_mode_first_analysis(config)
    )


class TorchBandpowerTransform:
    def __init__(
        self,
        geometry: EstimatorBandGeometry,
        analysis: ModeFirstAnalysisContract,
        device: torch.device,
    ) -> None:
        if geometry.cube_shape != analysis.full_layout.cube_shape:
            raise ValueError("Band geometry and analysis contract shapes differ")
        self.geometry = geometry
        self.analysis_contract_sha256 = analysis.analysis_contract_sha256
        self.device = device
        self.window = torch.as_tensor(
            analysis.analysis_window, dtype=torch.float64, device=device
        )
        self.mode_indices = torch.as_tensor(
            geometry.mode_indices, dtype=torch.int64, device=device
        )
        self.mode_bands = torch.as_tensor(
            geometry.mode_bands, dtype=torch.int64, device=device
        )
        self.counts = torch.as_tensor(
            geometry.counts, dtype=torch.float64, device=device
        )
        self.power_scale = float(analysis.power_scale)
        self.demean_mode = analysis.demean_mode

    def _demean(self, cubes: torch.Tensor) -> torch.Tensor:
        if self.demean_mode == "global":
            return cubes - cubes.mean(dim=(1, 2, 3), keepdim=True)
        if self.demean_mode == "per_freq_spatial":
            return cubes - cubes.mean(dim=(2, 3), keepdim=True)
        if self.demean_mode == "none":
            return cubes
        raise ValueError(f"Unsupported demean mode: {self.demean_mode}")

    def fourier(self, cubes: torch.Tensor) -> torch.Tensor:
        if cubes.ndim == 3:
            cubes = cubes.unsqueeze(0)
        if tuple(cubes.shape[1:]) != self.geometry.cube_shape:
            raise ValueError("Cube shape does not match the estimator contract")
        tapered = self._demean(cubes) * self.window.unsqueeze(0)
        return torch.fft.fftn(tapered, dim=(-3, -2, -1))

    def power_sums(self, cubes: torch.Tensor) -> torch.Tensor:
        spectrum = self.fourier(cubes)
        power = spectrum.real.square() + spectrum.imag.square()
        selected = power.reshape(power.shape[0], -1).index_select(
            1, self.mode_indices
        )
        output = torch.zeros(
            (power.shape[0], self.geometry.band_count),
            dtype=torch.float64,
            device=self.device,
        )
        output.scatter_add_(
            1,
            self.mode_bands.unsqueeze(0).expand(power.shape[0], -1),
            selected,
        )
        return output * self.power_scale

    def __call__(self, cubes: torch.Tensor) -> torch.Tensor:
        return self.power_sums(cubes) / self.counts.unsqueeze(0)


class CubeProjector(Protocol):
    contract_sha256: str

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class IdentityCubeProjector:
    def __init__(self, contract_sha256: str) -> None:
        self.contract_sha256 = str(contract_sha256)
        self.stats = {"fit_scope": "none", "parameter_count": 0}

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = cubes.ndim == 3
        batch = 1 if squeeze else int(cubes.shape[0])
        params = torch.zeros((batch, 0), dtype=cubes.dtype, device=cubes.device)
        return cubes, (params[0] if squeeze else params)


def _symmetric_gram_inverse(
    gram: torch.Tensor,
    *,
    rcond: float,
    ridge_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    maximum = float(torch.max(eigenvalues).item()) if eigenvalues.numel() else 0.0
    threshold = float(rcond) * maximum
    keep = eigenvalues > threshold
    if not bool(torch.any(keep)):
        raise ValueError("Control-mode nuisance response has zero numerical rank")
    retained = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    ridge = float(ridge_fraction) * maximum
    inverse = vectors @ torch.diag(torch.reciprocal(retained + ridge)) @ vectors.T
    return inverse, {
        "rank": int(torch.sum(keep).item()),
        "threshold": threshold,
        "ridge_absolute": ridge,
        "condition_number_retained": float(
            (retained[-1] + ridge).item() / (retained[0] + ridge).item()
        ),
    }


class FourierControlCubeProjector:
    def __init__(
        self,
        design: torch.Tensor,
        transform: TorchBandpowerTransform,
        control_mode_indices: np.ndarray,
        *,
        estimator_contract_sha256: str,
        rcond: float,
        ridge_fraction: float,
    ) -> None:
        if design.ndim != 4:
            raise ValueError("Compiled response must have shape [param,freq,y,x]")
        if tuple(design.shape[1:]) != transform.geometry.cube_shape:
            raise ValueError("Compiled response and analysis cube shapes differ")
        self.contract_sha256 = str(estimator_contract_sha256)
        self.design = design.reshape(design.shape[0], -1)
        self.control_indices = torch.as_tensor(
            np.asarray(control_mode_indices, dtype=np.int64),
            dtype=torch.int64,
            device=design.device,
        )
        design_spectrum = transform.fourier(design).reshape(design.shape[0], -1)
        self.control_design = design_spectrum.index_select(1, self.control_indices)
        gram = (
            self.control_design.real @ self.control_design.real.T
            + self.control_design.imag @ self.control_design.imag.T
        )
        self.gram_inverse, inverse_stats = _symmetric_gram_inverse(
            gram, rcond=rcond, ridge_fraction=ridge_fraction
        )
        self.transform = transform
        self.stats = {
            "fit_scope": "fourier_control_v2",
            "parameter_count": int(design.shape[0]),
            "control_fourier_mode_count": int(self.control_indices.numel()),
            **inverse_stats,
        }

    def project(self, cubes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = cubes.ndim == 3
        if squeeze:
            cubes = cubes.unsqueeze(0)
        spectrum = self.transform.fourier(cubes).reshape(cubes.shape[0], -1)
        control_data = spectrum.index_select(1, self.control_indices)
        rhs = (
            control_data.real @ self.control_design.real.T
            + control_data.imag @ self.control_design.imag.T
        )
        params = rhs @ self.gram_inverse
        residual = cubes.reshape(cubes.shape[0], -1) - params @ self.design
        projected = residual.reshape_as(cubes)
        return (projected[0] if squeeze else projected), (params[0] if squeeze else params)


def _svd_pinv(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=np.float64)
    left, singular, right_t = np.linalg.svd(values, full_matrices=False)
    threshold = float(rcond) * float(singular[0]) if singular.size else 0.0
    keep = singular > threshold
    inverse = np.zeros_like(singular)
    inverse[keep] = 1.0 / singular[keep]
    pinv = (right_t.T * inverse[None, :]) @ left.T
    return pinv, {
        "rank": int(np.sum(keep)),
        "threshold": threshold,
        "singular_values": singular,
        "retained_condition_number": (
            float(singular[0] / singular[keep][-1]) if np.any(keep) else float("inf")
        ),
    }


def _row_normalize(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    row_sums = np.sum(values, axis=1)
    threshold = max(float(np.max(np.abs(row_sums))) * float(rcond), 1e-300)
    valid = np.abs(row_sums) > threshold
    output = np.zeros_like(values)
    output[valid] = values[valid] / row_sums[valid, None]
    return output, valid


def _row_normalizer(
    matrix: np.ndarray, rcond: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    row_sums = np.sum(values, axis=1)
    threshold = max(float(np.max(np.abs(row_sums))) * float(rcond), 1e-300)
    valid = np.abs(row_sums) > threshold
    scale = np.zeros_like(row_sums)
    scale[valid] = 1.0 / row_sums[valid]
    return np.diag(scale), valid


def calibrate_mode_first_transfer(
    *,
    contract: ModeFirstEstimatorContract,
    target_transform: TorchBandpowerTransform,
    projector: CubeProjector,
    probes_per_source_band: int,
    batch_size: int,
    seed: int,
    transfer_rcond: float,
    analytic_input_response: np.ndarray | None = None,
) -> dict[str, Any]:
    if projector.contract_sha256 != contract.estimator_contract_sha256:
        raise ValueError("Projector and estimator contracts differ")
    if target_transform.analysis_contract_sha256 != contract.analysis.analysis_contract_sha256:
        raise ValueError("Bandpower transform and analysis contracts differ")
    if int(probes_per_source_band) < 2 or int(batch_size) < 1:
        raise ValueError("Probe count must be >=2 and batch size must be positive")
    source = contract.calibration_source_geometry
    target_count = target_transform.geometry.band_count
    source_count = source.band_count
    input_response = np.zeros((target_count, source_count), dtype=np.float64)
    projected_response = np.zeros_like(input_response)
    projected_square_sum = np.zeros_like(input_response)
    source_bands = torch.as_tensor(
        source.cube_mode_bands,
        dtype=torch.int64,
        device=target_transform.device,
    )
    generator = torch.Generator(device=target_transform.device)
    generator.manual_seed(int(seed))
    for source_band in range(source_count):
        input_sum = torch.zeros(
            (target_count,), dtype=torch.float64, device=target_transform.device
        )
        projected_sum = torch.zeros_like(input_sum)
        projected_square = torch.zeros_like(input_sum)
        source_mask = source_bands == int(source_band)
        for first in range(0, int(probes_per_source_band), int(batch_size)):
            current = min(int(batch_size), int(probes_per_source_band) - first)
            white = torch.randn(
                (current, *source.cube_shape),
                dtype=torch.float64,
                device=target_transform.device,
                generator=generator,
            )
            spectrum = torch.fft.fftn(
                white, dim=(-3, -2, -1), norm="ortho"
            )
            probes = torch.fft.ifftn(
                spectrum * source_mask.unsqueeze(0),
                dim=(-3, -2, -1),
                norm="ortho",
            ).real
            before = target_transform(probes)
            after_cube, _ = projector.project(probes)
            after = target_transform(after_cube)
            input_sum += torch.sum(before, dim=0)
            projected_sum += torch.sum(after, dim=0)
            projected_square += torch.sum(after.square(), dim=0)
        input_response[:, source_band] = (
            input_sum / float(probes_per_source_band)
        ).detach().cpu().numpy()
        projected_response[:, source_band] = (
            projected_sum / float(probes_per_source_band)
        ).detach().cpu().numpy()
        projected_square_sum[:, source_band] = (
            projected_square / float(probes_per_source_band)
        ).detach().cpu().numpy()

    monte_carlo_input_response = np.array(input_response, copy=True)
    if analytic_input_response is not None:
        exact_input = np.asarray(analytic_input_response, dtype=np.float64)
        if exact_input.shape != input_response.shape:
            raise ValueError("Analytic input response has the wrong shape")
        if not np.all(np.isfinite(exact_input)) or np.any(exact_input < 0.0):
            raise ValueError("Analytic input response must be finite and non-negative")
        input_response = exact_input
        input_response_source = "analytic_taper_convolution"
    else:
        input_response_source = "monte_carlo"
    input_pinv, input_svd = _svd_pinv(input_response, transfer_rcond)
    transfer = projected_response @ input_pinv
    transfer_pinv, transfer_svd = _svd_pinv(transfer, transfer_rcond)
    row_normalizer, transfer_valid = _row_normalizer(transfer, transfer_rcond)
    transfer_window = row_normalizer @ transfer
    raw_source_window, raw_source_valid = _row_normalize(
        input_response, transfer_rcond
    )
    projected_source_window, projected_source_valid = _row_normalize(
        projected_response, transfer_rcond
    )
    variance = np.maximum(
        projected_square_sum - np.square(projected_response), 0.0
    )
    return {
        "analysis_contract_sha256": contract.analysis.analysis_contract_sha256,
        "estimator_contract_sha256": contract.estimator_contract_sha256,
        "input_response": input_response,
        "monte_carlo_input_response": monte_carlo_input_response,
        "input_response_source": input_response_source,
        "projected_response": projected_response,
        "projected_response_standard_error": np.sqrt(
            variance / float(probes_per_source_band)
        ),
        "transfer_matrix": transfer,
        "deconvolution_matrix": transfer_pinv,
        "row_normalization_matrix": row_normalizer,
        "transfer_window_matrix": transfer_window,
        "window_matrix": transfer_window,
        "transfer_window_valid": transfer_valid,
        "raw_source_window_matrix": raw_source_window,
        "raw_source_window_valid": raw_source_valid,
        "projected_source_window_matrix": projected_source_window,
        "projected_source_window_valid": projected_source_valid,
        "input_response_svd": input_svd,
        "transfer_svd": transfer_svd,
        "probes_per_source_band": int(probes_per_source_band),
        "probe_batch_size": int(batch_size),
        "source_band_count": int(source_count),
        "target_band_count": int(target_count),
        "seed": int(seed),
    }


def analytic_identity_source_response(
    *,
    contract: ModeFirstEstimatorContract,
    target_transform: TorchBandpowerTransform,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """Compute the exact taper-induced source window for an identity projector."""
    if target_transform.analysis_contract_sha256 != contract.analysis.analysis_contract_sha256:
        raise ValueError("Bandpower transform and analysis contracts differ")
    if contract.analysis.demean_mode not in {"global", "none"}:
        raise ValueError("Analytic source response currently supports global or no demean")
    if int(batch_size) < 1:
        raise ValueError("Analytic source-response batch size must be positive")
    device = target_transform.device
    source = contract.calibration_source_geometry
    source_map = torch.as_tensor(
        source.cube_mode_bands, dtype=torch.int64, device=device
    )
    window = target_transform.window
    voxel_count = int(np.prod(source.cube_shape))
    kernel = torch.fft.fftn(window, dim=(-3, -2, -1)).abs().square()
    kernel = kernel / float(voxel_count)
    kernel_spectrum = torch.fft.fftn(kernel, dim=(-3, -2, -1))
    target_indices = target_transform.mode_indices
    target_bands = target_transform.mode_bands
    target_counts = target_transform.counts
    response = torch.empty(
        (target_transform.geometry.band_count, source.band_count),
        dtype=torch.float64,
        device=device,
    )
    for first in range(0, source.band_count, int(batch_size)):
        count = min(int(batch_size), source.band_count - first)
        band_ids = torch.arange(first, first + count, device=device)
        masks = source_map.unsqueeze(0) == band_ids[:, None, None, None]
        masks = masks.to(torch.float64)
        if contract.analysis.demean_mode == "global":
            masks[:, 0, 0, 0] = 0.0
        convolved = torch.fft.ifftn(
            torch.fft.fftn(masks, dim=(-3, -2, -1))
            * kernel_spectrum.unsqueeze(0),
            dim=(-3, -2, -1),
        ).real
        selected = convolved.reshape(count, -1).index_select(1, target_indices)
        sums = torch.zeros(
            (count, target_transform.geometry.band_count),
            dtype=torch.float64,
            device=device,
        )
        sums.scatter_add_(
            1,
            target_bands.unsqueeze(0).expand(count, -1),
            selected,
        )
        response[:, first : first + count] = (
            sums / target_counts.unsqueeze(0) * target_transform.power_scale
        ).T
    response_np = np.asarray(response.detach().cpu(), dtype=np.float64)
    response_np[np.abs(response_np) < np.max(response_np) * 1e-15] = 0.0
    source_window, valid = _row_normalize(response_np, 1e-14)
    if not np.all(valid):
        raise ValueError("Analytic identity source response contains an empty row")
    return {
        "input_response": response_np,
        "source_window_matrix": source_window,
    }


def estimate_projected_bandpower(
    projected_bandpower: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if calibration.get("estimator_contract_sha256") is None:
        raise ValueError("Calibration has no estimator contract identity")
    values = np.asarray(projected_bandpower, dtype=np.float64)
    row_normalizer = np.asarray(
        calibration["row_normalization_matrix"], dtype=np.float64
    )
    valid = np.asarray(calibration["transfer_window_valid"], dtype=bool)
    if not np.all(valid):
        raise ValueError("Transfer contains non-normalizable target rows")
    row_estimate = row_normalizer @ values
    deconvolved = np.asarray(calibration["deconvolution_matrix"]) @ values
    return row_estimate, deconvolved
