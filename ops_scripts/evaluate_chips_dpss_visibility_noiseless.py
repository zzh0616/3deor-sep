#!/usr/bin/env python3
"""Evaluate a noiseless CHIPS-DPSS visibility-domain PS2D pilot.

The primary view is a cross-quadratic estimate from alternating-time uv grids.
Coadded-grid and fixed-row auto estimates are retained only as diagnostics.
Foreground and EoR labels never enter the response, support, or window choices;
they are loaded after those choices to quantify leakage and signal transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chips_visibility import (  # noqa: E402
    QuadraticResponse,
    build_quadratic_response,
    cross_quadratic_bandpowers,
    fold_absolute_delay,
    fold_window_absolute_delay,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


@dataclass(frozen=True)
class VisibilityView:
    name: str
    role: str
    kperp_mpc_inv: np.ndarray
    fg_a: np.ndarray
    fg_b: np.ndarray
    eor_a: np.ndarray
    eor_b: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.kperp_mpc_inv.size)


METHODS: tuple[dict[str, Any], ...] = (
    {
        "name": "raw_none",
        "delay_model": "patch",
        "suppression_strength": 0.0,
        "taper": "none",
    },
    {
        "name": "raw_hann",
        "delay_model": "patch",
        "suppression_strength": 0.0,
        "taper": "hann",
    },
    {
        "name": "patch_cov_1e4",
        "delay_model": "patch",
        "suppression_strength": 1e4,
        "taper": "hann",
    },
    {
        "name": "patch_cov_1e8",
        "delay_model": "patch",
        "suppression_strength": 1e8,
        "taper": "hann",
    },
    {
        "name": "patch_cov_1e12",
        "delay_model": "patch",
        "suppression_strength": 1e12,
        "taper": "hann",
    },
    {
        "name": "patch_hard",
        "delay_model": "patch",
        "suppression_strength": math.inf,
        "taper": "hann",
    },
    {
        "name": "horizon_cov_1e8",
        "delay_model": "horizon",
        "suppression_strength": 1e8,
        "taper": "hann",
    },
    {
        "name": "horizon_hard",
        "delay_model": "horizon",
        "suppression_strength": math.inf,
        "taper": "hann",
    },
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-grid-weight", type=float, default=1.0)
    parser.add_argument("--minimum-samples-per-bin", type=int, default=8)
    parser.add_argument("--minimum-relative-sensitivity", type=float, default=1e-4)
    parser.add_argument("--minimum-window-self-fraction", type=float, default=0.1)
    parser.add_argument("--dpss-eigenvalue-threshold", type=float, default=1e-12)
    parser.add_argument("--foreground-leakage-tolerance", type=float, default=0.1)
    parser.add_argument("--total-relative-error-tolerance", type=float, default=0.2)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_bank(
    bank_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = bank_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "chips_visibility_bank"
        or int(manifest.get("schema_version", -1)) != 1
    ):
        raise ValueError("Unsupported CHIPS visibility bank schema")
    bank_path = Path(str(manifest["bank_path"]))
    if not bank_path.is_absolute():
        bank_path = bank_dir / bank_path
    if _sha256(bank_path) != str(manifest["bank_sha256"]):
        raise ValueError("Visibility bank hash mismatch")
    with np.load(bank_path) as loaded:
        bank = {name: np.asarray(loaded[name]) for name in loaded.files}
    return manifest, bank


def _make_views(
    bank: dict[str, np.ndarray],
    *,
    transverse_distance_mpc: float,
    reference_frequency_mhz: float,
    minimum_grid_weight: float,
) -> tuple[list[VisibilityView], dict[str, Any]]:
    frequencies = np.asarray(bank["frequencies_hz"], dtype=np.float64)
    grid_weight = np.asarray(bank["grid_weight"], dtype=np.float64)
    fg_grid = np.asarray(bank["fg_grid"], dtype=np.complex128)
    eor_grid = np.asarray(bank["eor_grid"], dtype=np.complex128)
    if (
        grid_weight.shape != fg_grid.shape
        or grid_weight.shape != eor_grid.shape
        or grid_weight.ndim != 4
        or grid_weight.shape[:2] != (2, frequencies.size)
    ):
        raise ValueError("Visibility grid arrays have incompatible shapes")
    complete = np.all(grid_weight >= float(minimum_grid_weight), axis=(0, 1))
    u, v = np.meshgrid(
        np.asarray(bank["u_centers_lambda"], dtype=np.float64),
        np.asarray(bank["v_centers_lambda"], dtype=np.float64),
        indexing="xy",
    )
    # Gridding explicitly inserted conjugate samples.  Retain one canonical
    # half-plane so sample counts and quadratic averages do not double-count
    # deterministic conjugates.
    coordinate_tolerance = max(
        float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1.0
    ) * 1e-12
    canonical = (v > coordinate_tolerance) | (
        (np.abs(v) <= coordinate_tolerance) & (u >= 0.0)
    )
    common = complete & canonical
    kperp_grid = 2.0 * math.pi * np.hypot(u, v) / float(transverse_distance_mpc)
    if not np.any(common):
        raise ValueError("No uv cells have complete split/frequency support")
    fg_a = fg_grid[0][:, common].T
    fg_b = fg_grid[1][:, common].T
    eor_a = eor_grid[0][:, common].T
    eor_b = eor_grid[1][:, common].T
    kperp_common = kperp_grid[common]

    total_weight = grid_weight[0] + grid_weight[1]
    coadd_fg = np.zeros_like(fg_grid[0])
    coadd_eor = np.zeros_like(eor_grid[0])
    occupied = total_weight > 0.0
    coadd_fg[occupied] = (
        grid_weight[0][occupied] * fg_grid[0][occupied]
        + grid_weight[1][occupied] * fg_grid[1][occupied]
    ) / total_weight[occupied]
    coadd_eor[occupied] = (
        grid_weight[0][occupied] * eor_grid[0][occupied]
        + grid_weight[1][occupied] * eor_grid[1][occupied]
    ) / total_weight[occupied]
    coadd_fg_samples = coadd_fg[:, common].T
    coadd_eor_samples = coadd_eor[:, common].T

    uvw = np.asarray(bank["sample_uvw_m"], dtype=np.float64)
    row_kperp = (
        2.0
        * math.pi
        * np.hypot(uvw[:, 0], uvw[:, 1])
        * float(reference_frequency_mhz)
        * 1e6
        / 299792458.0
        / float(transverse_distance_mpc)
    )
    row_fg = np.asarray(bank["sample_fg"], dtype=np.complex128).T
    row_eor = np.asarray(bank["sample_eor"], dtype=np.complex128).T
    if row_fg.shape != row_eor.shape or row_fg.shape[1] != frequencies.size:
        raise ValueError("Visibility row samples have incompatible shapes")

    views = [
        VisibilityView(
            name="uvgrid_cross",
            role="primary_noise_bias_free_architecture",
            kperp_mpc_inv=kperp_common,
            fg_a=fg_a,
            fg_b=fg_b,
            eor_a=eor_a,
            eor_b=eor_b,
        ),
        VisibilityView(
            name="uvgrid_auto",
            role="noiseless_gridding_diagnostic",
            kperp_mpc_inv=kperp_common,
            fg_a=coadd_fg_samples,
            fg_b=coadd_fg_samples,
            eor_a=coadd_eor_samples,
            eor_b=coadd_eor_samples,
        ),
        VisibilityView(
            name="row_auto",
            role="no_gridding_fixed_baseline_time_diagnostic",
            kperp_mpc_inv=row_kperp,
            fg_a=row_fg,
            fg_b=row_fg,
            eor_a=row_eor,
            eor_b=row_eor,
        ),
    ]
    return views, {
        "grid_cell_count": int(common.size),
        "complete_grid_cell_count_before_conjugate_dedup": int(
            np.count_nonzero(complete)
        ),
        "complete_grid_cell_count": int(np.count_nonzero(common)),
        "complete_grid_cell_fraction_before_conjugate_dedup": float(
            np.mean(complete)
        ),
        "complete_grid_cell_fraction": float(
            np.count_nonzero(common) / max(1, np.count_nonzero(canonical))
        ),
        "row_sample_count": int(row_fg.shape[0]),
        "minimum_grid_weight": float(minimum_grid_weight),
    }


def _fold_response_vector(
    values: np.ndarray,
    response: QuadraticResponse,
) -> np.ndarray:
    folded, _, _ = fold_absolute_delay(values, response.delays_s)
    return np.asarray(folded, dtype=np.float64)


def _estimate_components(
    view: VisibilityView,
    selected: np.ndarray,
    response: QuadraticResponse,
) -> dict[str, np.ndarray]:
    fg, _ = cross_quadratic_bandpowers(
        view.fg_a[selected], view.fg_b[selected], response
    )
    eor, _ = cross_quadratic_bandpowers(
        view.eor_a[selected], view.eor_b[selected], response
    )
    total_a = view.fg_a[selected] + view.eor_a[selected]
    total_b = view.fg_b[selected] + view.eor_b[selected]
    total, _ = cross_quadratic_bandpowers(total_a, total_b, response)
    folded_fg = _fold_response_vector(fg, response)
    folded_eor = _fold_response_vector(eor, response)
    folded_total = _fold_response_vector(total, response)
    return {
        "foreground": folded_fg,
        "eor": folded_eor,
        "total": folded_total,
        "cross": folded_total - folded_fg - folded_eor,
    }


def _metrics(
    *,
    foreground: np.ndarray,
    eor: np.ndarray,
    total: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    foreground_tolerance: float,
    total_tolerance: float,
) -> dict[str, Any]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(foreground)
        & np.isfinite(eor)
        & np.isfinite(total)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return {"n_bins": 0}
    fg = foreground[valid]
    signal = eor[valid]
    observed = total[valid]
    weight = np.asarray(weights, dtype=np.float64)[valid]
    signal_l1 = max(float(np.sum(weight * np.abs(signal))), 1e-300)
    signal_l2 = max(
        float(np.sum(weight * np.square(np.abs(signal)))), 1e-300
    )
    nonzero = np.abs(signal) > max(
        1e-12 * float(np.nanmax(np.abs(signal))), np.finfo(np.float64).tiny
    )
    per_bin_pass = np.zeros(signal.shape, dtype=bool)
    per_bin_pass[nonzero] = (
        np.abs(fg[nonzero]) <= float(foreground_tolerance) * np.abs(signal[nonzero])
    ) & (
        np.abs(observed[nonzero] - signal[nonzero])
        <= float(total_tolerance) * np.abs(signal[nonzero])
    )
    signed_denominator = float(np.sum(weight * signal))
    return {
        "n_bins": int(np.count_nonzero(valid)),
        "weighted_mode_count_proxy": float(np.sum(weight)),
        "eor_positive_fraction": float(np.mean(signal > 0.0)),
        "foreground_to_eor_absolute_l1": float(
            np.sum(weight * np.abs(fg)) / signal_l1
        ),
        "foreground_to_eor_l2": float(
            math.sqrt(
                float(np.sum(weight * np.square(np.abs(fg)))) / signal_l2
            )
        ),
        "total_minus_eor_relative_l2": float(
            math.sqrt(
                float(
                    np.sum(weight * np.square(np.abs(observed - signal)))
                )
                / signal_l2
            )
        ),
        "signed_integrated_total_over_eor": (
            float(np.sum(weight * observed) / signed_denominator)
            if abs(signed_denominator) > 1e-300
            else math.nan
        ),
        "passing_bins": int(np.count_nonzero(per_bin_pass)),
        "passing_fraction": float(np.mean(per_bin_pass)),
        "passing_weight_fraction": float(
            np.sum(weight[per_bin_pass]) / np.sum(weight)
        ),
    }


def _scientific_report(
    result: dict[str, Any],
) -> str:
    lines = [
        "# CHIPS-DPSS visibility-domain 无噪声测试",
        "",
        "## 口径",
        "",
        "- 主结果为交替时间 split 的 uv-grid cross-quadratic bandpower。",
        "- support 只由几何 EoR window、样本数、Fisher 相对灵敏度和 window self-fraction 决定。",
        "- FG/EoR 标签只用于冻结 support 后的泄漏诊断，不参与方法或 bin 选择。",
        "- 当前结果是 visibility-delay PS2D；尚未用独立 OSKAR unit-band probes 转换为绝对 sky PS2D。",
        "",
        "## 结果",
        "",
        "| view | method | support bins | FG/EoR L1 | total error L2 | passing bins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view_name, methods in result["metrics"].items():
        for method_name, metrics in methods.items():
            lines.append(
                "| {view} | {method} | {bins} | {fg:.6g} | {error:.6g} | {passed} |".format(
                    view=view_name,
                    method=method_name,
                    bins=int(metrics.get("n_bins", 0)),
                    fg=float(metrics.get("foreground_to_eor_absolute_l1", math.nan)),
                    error=float(metrics.get("total_minus_eor_relative_l2", math.nan)),
                    passed=int(metrics.get("passing_bins", 0)),
                )
            )
    lines.extend(
        [
            "",
            "## 解释约束",
            "",
            "- `raw_hann` 是未抑制前景的基线，不以 `FG/EoR<1` 作为预先停机门。",
            "- `patch_*` 只使用已知模拟视场角和预声明 supra-horizon buffer；`horizon_*` 是更保守控制。",
            "- covariance 强度扫描是预声明诊断。不能用注入真值事后挑选强度作为部署 selector。",
            "- auto 视图在加入热噪声后有 noise bias，不能替代 split cross-power。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    manifest, bank = _load_bank(args.bank_dir)
    frequencies_hz = np.asarray(bank["frequencies_hz"], dtype=np.float64)
    expected_frequencies_hz = (
        np.asarray(resolved.geometry["frequencies_mhz"], dtype=np.float64) * 1e6
    )
    if not np.allclose(
        frequencies_hz, expected_frequencies_hz, rtol=0.0, atol=1e-3
    ):
        raise ValueError("Visibility bank and PS2D config frequencies differ")

    views, bank_diagnostics = _make_views(
        bank,
        transverse_distance_mpc=float(resolved.geometry["transverse_distance_mpc"]),
        reference_frequency_mhz=float(resolved.geometry["reference_frequency_mhz"]),
        minimum_grid_weight=float(args.minimum_grid_weight),
    )
    source_side_rad = math.radians(
        float(config["image_geometry"]["source_image_size"])
        * float(config["image_geometry"]["spatial_pixel_arcsec"])
        / 3600.0
    )
    natural_uv_cell_lambda = 1.0 / source_side_rad
    u_centers = np.asarray(bank["u_centers_lambda"], dtype=np.float64)
    actual_uv_cell_lambda = float(np.median(np.diff(u_centers)))
    uv_cell_ratio = actual_uv_cell_lambda / natural_uv_cell_lambda
    bank_diagnostics.update(
        {
            "source_side_deg": math.degrees(source_side_rad),
            "natural_fourier_uv_cell_lambda": natural_uv_cell_lambda,
            "actual_uv_cell_lambda": actual_uv_cell_lambda,
            "actual_over_natural_uv_cell": uv_cell_ratio,
            "gridding_resolution_gate_max_ratio": 1.2,
            "gridding_resolution_gate_pass": bool(uv_cell_ratio <= 1.2),
        }
    )
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    kperp_centers = 0.5 * (kperp_edges[:-1] + kperp_edges[1:])
    delays = np.fft.fftfreq(
        frequencies_hz.size, d=float(np.median(np.diff(frequencies_hz)))
    )
    folded_delay, _, delay_degeneracy = fold_absolute_delay(
        np.abs(delays), delays
    )
    folded_delay = np.asarray(folded_delay, dtype=np.float64)
    radial_mpc_per_hz = float(resolved.geometry["radial_spacing_mpc"]) / float(
        np.mean(np.diff(frequencies_hz))
    )
    kpar = 2.0 * math.pi * folded_delay / radial_mpc_per_hz
    expected_kpar = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    if not np.allclose(kpar, expected_kpar, rtol=1e-8, atol=1e-12):
        raise ValueError("Visibility-delay and frozen PS2D kpar axes differ")

    upper_kperp = kperp_edges[1:, None]
    geometric_window = resolved.window_spec.mask(upper_kperp, kpar[None, :])
    nkperp = int(kperp_centers.size)
    nkpar = int(kpar.size)
    reference_frequency_hz = (
        float(resolved.geometry["reference_frequency_mhz"]) * 1e6
    )
    patch_angle_rad = math.radians(
        float(resolved.geometry["source_corner_angle_deg"])
    )
    buffer_delay_s = (
        float(resolved.geometry["wedge_buffer_mpc_inv"])
        * radial_mpc_per_hz
        / (2.0 * math.pi)
    )
    u_upper = (
        kperp_edges[1:]
        * float(resolved.geometry["transverse_distance_mpc"])
        / (2.0 * math.pi)
    )
    maximum_delays = {
        "patch": u_upper * math.sin(patch_angle_rad) / reference_frequency_hz
        + buffer_delay_s,
        "horizon": u_upper / reference_frequency_hz + buffer_delay_s,
    }

    products: dict[str, np.ndarray] = {
        "kperp_edges_mpc_inv": kperp_edges,
        "kperp_centers_mpc_inv": kperp_centers,
        "kpar_mpc_inv": kpar,
        "delay_s": folded_delay,
        "delay_mode_degeneracy": delay_degeneracy,
        "geometric_window": geometric_window.astype(np.int8),
    }
    all_metrics: dict[str, dict[str, Any]] = {}
    method_diagnostics: dict[str, dict[str, Any]] = {}

    for view in views:
        view_metrics: dict[str, Any] = {}
        sample_counts = np.zeros(nkperp, dtype=np.int64)
        band_members: list[np.ndarray] = []
        for index in range(nkperp):
            members = np.flatnonzero(
                (view.kperp_mpc_inv >= kperp_edges[index])
                & (
                    (view.kperp_mpc_inv < kperp_edges[index + 1])
                    | (
                        index == nkperp - 1
                        and np.isclose(
                            view.kperp_mpc_inv,
                            kperp_edges[index + 1],
                            rtol=1e-12,
                            atol=1e-14,
                        )
                    )
                )
            )
            band_members.append(members)
            sample_counts[index] = int(members.size)
        products[f"{view.name}__sample_counts"] = sample_counts

        raw_responses = [
            build_quadratic_response(
                frequencies_hz,
                max_delay_s=float(maximum_delays["patch"][index]),
                suppression_strength=0.0,
                dpss_eigenvalue_threshold=float(args.dpss_eigenvalue_threshold),
                taper="hann",
            )
            for index in range(nkperp)
        ]
        raw_normalization = np.stack(
            [
                _fold_response_vector(response.row_normalization, response)
                for response in raw_responses
            ],
            axis=0,
        )

        for method in METHODS:
            name = str(method["name"])
            component_arrays = {
                component: np.full((nkperp, nkpar), np.nan, dtype=np.float64)
                for component in ("foreground", "eor", "total", "cross")
            }
            sensitivity = np.zeros((nkperp, nkpar), dtype=np.float64)
            window_self = np.zeros((nkperp, nkpar), dtype=np.float64)
            window_effective_width = np.full(
                (nkperp, nkpar), np.inf, dtype=np.float64
            )
            ranks = np.zeros(nkperp, dtype=np.int64)
            windows = np.zeros((nkperp, nkpar, nkpar), dtype=np.float64)
            for index, members in enumerate(band_members):
                response = build_quadratic_response(
                    frequencies_hz,
                    max_delay_s=float(
                        maximum_delays[str(method["delay_model"])][index]
                    ),
                    suppression_strength=float(method["suppression_strength"]),
                    dpss_eigenvalue_threshold=float(
                        args.dpss_eigenvalue_threshold
                    ),
                    taper=str(method["taper"]),
                )
                ranks[index] = int(response.foreground_rank)
                folded_norm = _fold_response_vector(
                    response.row_normalization, response
                )
                sensitivity[index] = np.divide(
                    folded_norm,
                    raw_normalization[index],
                    out=np.zeros_like(folded_norm),
                    where=raw_normalization[index] > 0.0,
                )
                folded_window, window_delays = fold_window_absolute_delay(
                    response.window, response.delays_s
                )
                if not np.allclose(
                    window_delays, folded_delay, rtol=0.0, atol=1e-18
                ):
                    raise ValueError("Folded response delay axes differ")
                windows[index] = folded_window
                window_self[index] = np.diag(folded_window)
                square_sum = np.sum(np.square(folded_window), axis=1)
                window_effective_width[index] = np.divide(
                    1.0,
                    square_sum,
                    out=np.full(square_sum.shape, np.inf),
                    where=square_sum > 0.0,
                )
                if members.size < int(args.minimum_samples_per_bin):
                    continue
                estimated = _estimate_components(view, members, response)
                for component, values in estimated.items():
                    component_arrays[component][index] = values

            support = (
                geometric_window
                & (sample_counts[:, None] >= int(args.minimum_samples_per_bin))
                & (
                    sensitivity
                    >= float(args.minimum_relative_sensitivity)
                )
                & (
                    window_self
                    >= float(args.minimum_window_self_fraction)
                )
            )
            metrics = _metrics(
                foreground=component_arrays["foreground"],
                eor=component_arrays["eor"],
                total=component_arrays["total"],
                mask=support,
                weights=(
                    sample_counts[:, None]
                    * np.asarray(delay_degeneracy, dtype=np.float64)[None, :]
                ),
                foreground_tolerance=float(args.foreground_leakage_tolerance),
                total_tolerance=float(args.total_relative_error_tolerance),
            )
            supported_values = support
            metrics.update(
                {
                    "support_fraction_of_geometric_window": float(
                        np.count_nonzero(support)
                        / max(1, np.count_nonzero(geometric_window))
                    ),
                    "median_relative_sensitivity": (
                        float(np.median(sensitivity[supported_values]))
                        if np.any(supported_values)
                        else math.nan
                    ),
                    "median_window_self_fraction": (
                        float(np.median(window_self[supported_values]))
                        if np.any(supported_values)
                        else math.nan
                    ),
                    "median_window_effective_width": (
                        float(np.median(window_effective_width[supported_values]))
                        if np.any(supported_values)
                        else math.nan
                    ),
                }
            )
            view_metrics[name] = metrics
            prefix = f"{view.name}__{name}"
            for component, values in component_arrays.items():
                products[f"{prefix}__{component}"] = values
            products[f"{prefix}__support"] = support.astype(np.int8)
            products[f"{prefix}__relative_sensitivity"] = sensitivity
            products[f"{prefix}__window_self"] = window_self
            products[f"{prefix}__window_effective_width"] = window_effective_width
            products[f"{prefix}__window"] = windows
            products[f"{prefix}__foreground_rank"] = ranks
            method_diagnostics.setdefault(name, {
                "delay_model": str(method["delay_model"]),
                "suppression_strength": (
                    "infinity"
                    if math.isinf(float(method["suppression_strength"]))
                    else float(method["suppression_strength"])
                ),
                "taper": str(method["taper"]),
                "foreground_rank_by_kperp": ranks.tolist(),
                "maximum_delay_us_by_kperp": (
                    maximum_delays[str(method["delay_model"])] * 1e6
                ).tolist(),
            })
        all_metrics[view.name] = view_metrics

    for view in views:
        sample_counts = np.asarray(
            products[f"{view.name}__sample_counts"], dtype=np.float64
        )
        aggregate_weights = (
            sample_counts[:, None]
            * np.asarray(delay_degeneracy, dtype=np.float64)[None, :]
        )
        raw_eor = np.asarray(
            products[f"{view.name}__raw_hann__eor"], dtype=np.float64
        )
        identity_eor = np.asarray(
            products[f"{view.name}__raw_none__eor"], dtype=np.float64
        )
        for method in METHODS:
            name = str(method["name"])
            prefix = f"{view.name}__{name}"
            eor = np.asarray(products[f"{prefix}__eor"], dtype=np.float64)
            support = np.asarray(products[f"{prefix}__support"], dtype=bool)
            valid = (
                support
                & np.isfinite(raw_eor)
                & np.isfinite(eor)
                & (aggregate_weights > 0.0)
            )
            if not np.any(valid):
                all_metrics[view.name][name].update(
                    {
                        "pure_eor_vs_raw_integrated_ratio": math.nan,
                        "pure_eor_vs_raw_relative_l2": math.nan,
                        "pure_eor_window_closure_integrated_ratio": math.nan,
                        "pure_eor_window_closure_relative_l2": math.nan,
                    }
                )
                continue
            weight = aggregate_weights[valid]
            raw = raw_eor[valid]
            current = eor[valid]
            window = np.asarray(
                products[f"{prefix}__window"], dtype=np.float64
            )
            predicted_eor = np.einsum(
                "ijk,ik->ij", window, identity_eor, optimize=True
            )
            prediction = predicted_eor[valid]
            denominator_sum = float(np.sum(weight * raw))
            denominator_l2 = max(
                float(np.sum(weight * np.square(np.abs(raw)))), 1e-300
            )
            all_metrics[view.name][name].update(
                {
                    "pure_eor_vs_raw_integrated_ratio": (
                        float(np.sum(weight * current) / denominator_sum)
                        if abs(denominator_sum) > 1e-300
                        else math.nan
                    ),
                    "pure_eor_vs_raw_relative_l2": float(
                        math.sqrt(
                            float(
                                np.sum(
                                    weight
                                    * np.square(np.abs(current - raw))
                                )
                            )
                            / denominator_l2
                        )
                    ),
                    "pure_eor_window_closure_integrated_ratio": (
                        float(
                            np.sum(weight * current)
                            / np.sum(weight * prediction)
                        )
                        if abs(float(np.sum(weight * prediction))) > 1e-300
                        else math.nan
                    ),
                    "pure_eor_window_closure_relative_l2": float(
                        math.sqrt(
                            float(
                                np.sum(
                                    weight
                                    * np.square(np.abs(current - prediction))
                                )
                            )
                            / max(
                                float(
                                    np.sum(
                                        weight
                                        * np.square(np.abs(prediction))
                                    )
                                ),
                                1e-300,
                            )
                        )
                    ),
                }
            )

    result = {
        "schema": "chips_dpss_visibility_noiseless_result",
        "schema_version": 1,
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "visibility_bank_sha256": manifest["bank_sha256"],
        "primary_view": "uvgrid_cross",
        "primary_grid512_method": "patch_hard",
        "primary_method_selection_status": (
            "truth-informed engineering promotion from the fixed-row screen; "
            "requires an independent sky realization before a scientific claim"
        ),
        "primary_gridding_resolution_pass": bool(
            bank_diagnostics["gridding_resolution_gate_pass"]
        ),
        "bank_diagnostics": bank_diagnostics,
        "support_contract": {
            "geometric_window_uses_upper_kperp_edge": True,
            "minimum_samples_per_bin": int(args.minimum_samples_per_bin),
            "minimum_relative_sensitivity": float(
                args.minimum_relative_sensitivity
            ),
            "minimum_window_self_fraction": float(
                args.minimum_window_self_fraction
            ),
            "dpss_eigenvalue_threshold": float(
                args.dpss_eigenvalue_threshold
            ),
            "foreground_leakage_tolerance": float(
                args.foreground_leakage_tolerance
            ),
            "total_relative_error_tolerance": float(
                args.total_relative_error_tolerance
            ),
        },
        "method_diagnostics": method_diagnostics,
        "metrics": all_metrics,
        "limitations": [
            "no thermal noise",
            "single foreground and EoR realization",
            "isotropic station beam",
            "bilinear uv gridding does not yet include a beam or w-projection kernel",
            "observable visibility-delay power has not yet been converted to absolute sky PS2D",
            "covariance-strength alternatives are diagnostics, not a truth-selected deployable selector",
        ],
    }
    _atomic_npz(args.out_dir / "result.npz", products)
    _atomic_json(args.out_dir / "result.json", result)
    (args.out_dir / "summary_zh.md").write_text(
        _scientific_report(result), encoding="utf-8"
    )
    print(json.dumps(all_metrics["uvgrid_cross"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
