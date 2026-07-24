#!/usr/bin/env python3
"""Audit full-EoR-window recovery for an exact-visibility Q_beta result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fg-rmw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chips_visibility import (  # noqa: E402
    build_dpss_prefiltered_subband_response,
    fold_absolute_delay,
    matched_absolute_delay_window_fraction,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402


INK = "#17324d"
TEAL = "#147d7e"
VERMILION = "#d24b2a"
GOLD = "#e3a018"
PALE = "#dce9e6"
GRID = "#d7d2c8"

STATUS_COLORS = {
    "outside_eor_window": "#ece8df",
    "removed_by_dpss_response": "#aacbd3",
    "low_filter_sensitivity": "#8da0cb",
    "low_exact_qbeta_response": "#e6ab6c",
    "response_leaks_outside_eor": "#b597ba",
    "physical_eor_closure_failure": "#ef8a62",
    "heldout_eor_closure_failure": "#f6b26b",
    "foreground_leakage_failure": "#b2182b",
    "recoverable_windowed_bandpower": TEAL,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-window-self-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-filter-sensitivity", type=float, default=1e-4)
    parser.add_argument("--minimum-relative-qbeta-response", type=float, default=0.1)
    parser.add_argument("--minimum-geometric-window-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-cell-error", type=float, default=0.2)
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fbfaf6",
            "savefig.facecolor": "#fbfaf6",
            "savefig.bbox": "tight",
        }
    )


def _maximum_patch_delays(
    *,
    kperp_edges: np.ndarray,
    transverse_distance_mpc: float,
    reference_frequency_hz: float,
    source_corner_angle_deg: float,
    wedge_buffer_mpc_inv: float,
    radial_mpc_per_hz: float,
) -> np.ndarray:
    u_upper = (
        np.asarray(kperp_edges[1:], dtype=np.float64)
        * float(transverse_distance_mpc)
        / (2.0 * math.pi)
    )
    buffer_delay = (
        float(wedge_buffer_mpc_inv)
        * float(radial_mpc_per_hz)
        / (2.0 * math.pi)
    )
    return (
        u_upper
        * math.sin(math.radians(float(source_corner_angle_deg)))
        / float(reference_frequency_hz)
        + buffer_delay
    )


def _spectral_support_diagnostics(
    *,
    frequencies_hz: np.ndarray,
    analysis_frequency_indices: np.ndarray,
    maximum_delays_s: np.ndarray,
    dpss_eigenvalue_threshold: float,
    taper: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matched_rows: list[np.ndarray] = []
    sensitivity_rows: list[np.ndarray] = []
    ranks: list[int] = []
    for maximum_delay in maximum_delays_s:
        response = build_dpss_prefiltered_subband_response(
            frequencies_hz,
            analysis_frequency_indices=analysis_frequency_indices,
            max_delay_s=float(maximum_delay),
            suppression_strength=math.inf,
            dpss_eigenvalue_threshold=float(dpss_eigenvalue_threshold),
            taper=taper,
        )
        raw = build_dpss_prefiltered_subband_response(
            frequencies_hz,
            analysis_frequency_indices=analysis_frequency_indices,
            max_delay_s=float(maximum_delay),
            suppression_strength=0.0,
            dpss_eigenvalue_threshold=float(dpss_eigenvalue_threshold),
            taper=taper,
        )
        matched, _ = matched_absolute_delay_window_fraction(
            response.window,
            response.delays_s,
            response.input_delays_s,
        )
        filtered_norm, _, _ = fold_absolute_delay(
            response.row_normalization, response.delays_s
        )
        raw_norm, _, _ = fold_absolute_delay(
            raw.row_normalization, raw.delays_s
        )
        sensitivity = np.divide(
            filtered_norm,
            raw_norm,
            out=np.zeros_like(filtered_norm),
            where=raw_norm > 0.0,
        )
        matched_rows.append(matched)
        sensitivity_rows.append(sensitivity)
        ranks.append(int(response.foreground_rank))
    return (
        np.stack(matched_rows),
        np.stack(sensitivity_rows),
        np.asarray(ranks, dtype=np.int64),
    )


def _response_products(
    arrays: dict[str, np.ndarray],
    geometric: np.ndarray,
    kpar_values: np.ndarray,
) -> dict[str, np.ndarray]:
    response = np.asarray(arrays["calibration_response"], dtype=np.float64)
    row_sum = np.sum(response, axis=1)
    window = np.divide(
        response,
        row_sum[:, None],
        out=np.zeros_like(response),
        where=row_sum[:, None] > 0.0,
    )
    source_power = np.asarray(
        arrays["restricted_eor_source_power"], dtype=np.float64
    )
    target = window @ source_power
    source_in_geometric = np.asarray(
        arrays["source_band_in_geometric_window"], dtype=bool
    )
    geometric_fraction = np.sum(window[:, source_in_geometric], axis=1)
    relative_response = row_sum / max(float(np.max(row_sum)), 1e-300)
    pure = np.divide(
        arrays["bank_eor_q"],
        row_sum,
        out=np.full(row_sum.shape, np.nan),
        where=row_sum > 0.0,
    )
    total = np.divide(
        arrays["bank_total_q"],
        row_sum,
        out=np.full(row_sum.shape, np.nan),
        where=row_sum > 0.0,
    )
    foreground = np.divide(
        arrays["bank_foreground_q"],
        row_sum,
        out=np.full(row_sum.shape, np.nan),
        where=row_sum > 0.0,
    )
    heldout_pure = np.divide(
        arrays["heldout_mixture_q"],
        row_sum[None, :],
        out=np.full_like(arrays["heldout_mixture_q"], np.nan, dtype=np.float64),
        where=row_sum[None, :] > 0.0,
    )
    heldout_total = np.divide(
        arrays["heldout_total_mixture_q"],
        row_sum[None, :],
        out=np.full_like(
            arrays["heldout_total_mixture_q"], np.nan, dtype=np.float64
        ),
        where=row_sum[None, :] > 0.0,
    )
    denominator = np.maximum(np.abs(target), 1e-300)
    heldout_pure_error = np.abs(heldout_pure - target[None, :]) / denominator
    heldout_total_error = np.abs(heldout_total - target[None, :]) / denominator
    heldout_foreground_effect = (
        np.abs(heldout_total - heldout_pure) / denominator
    )
    square_sum = np.sum(np.square(window), axis=1)
    effective_width = np.divide(
        1.0,
        square_sum,
        out=np.full(row_sum.shape, np.inf),
        where=square_sum > 0.0,
    )

    source_kperp = np.asarray(
        arrays["source_band_kperp_indices"], dtype=np.int64
    )
    source_kpar = np.asarray(
        arrays["source_band_kpar_mpc_inv"], dtype=np.float64
    )
    output_ids = np.asarray(arrays["output_band_ids"], dtype=np.int64)
    same_cell_fraction = np.zeros(output_ids.size, dtype=np.float64)
    for position, band_id in enumerate(output_ids):
        kp_index, kpar_index = divmod(int(band_id), geometric.shape[1])
        source_match = (
            (source_kperp == kp_index)
            & np.isclose(
                source_kpar,
                float(kpar_values[kpar_index]),
                rtol=1e-10,
                atol=1e-12,
            )
        )
        if np.any(source_match):
            same_cell_fraction[position] = float(
                np.sum(window[position, source_match])
            )
    return {
        "row_sum": row_sum,
        "window": window,
        "target": target,
        "pure": pure,
        "total": total,
        "foreground": foreground,
        "geometric_fraction": geometric_fraction,
        "relative_response": relative_response,
        "heldout_pure_error": heldout_pure_error,
        "heldout_total_error": heldout_total_error,
        "heldout_foreground_effect": heldout_foreground_effect,
        "heldout_pure": heldout_pure,
        "heldout_total": heldout_total,
        "effective_width": effective_width,
        "same_cell_fraction": same_cell_fraction,
    }


def _plot(
    *,
    path: Path,
    kperp_edges: np.ndarray,
    kpar_edges: np.ndarray,
    boundary: np.ndarray,
    statuses: np.ndarray,
    worst_total_error_percent: np.ndarray,
    summary: dict[str, Any],
) -> None:
    _style()
    status_names = list(STATUS_COLORS)
    category = np.asarray(
        [[status_names.index(value) for value in row] for row in statuses],
        dtype=np.int64,
    )
    cmap = ListedColormap([STATUS_COLORS[name] for name in status_names])
    norm = BoundaryNorm(np.arange(-0.5, len(status_names) + 0.5), cmap.N)
    x_centers = 0.5 * (kperp_edges[:-1] + kperp_edges[1:])
    y_centers = 0.5 * (kpar_edges[:-1] + kpar_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(19.0, 7.2), constrained_layout=True)
    axes[0].pcolormesh(
        kperp_edges,
        kpar_edges,
        category.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        linewidth=0.14,
        edgecolors=(1.0, 1.0, 1.0, 0.40),
    )
    axes[0].legend(
        handles=[
            Patch(facecolor=STATUS_COLORS[name], edgecolor="none", label=name)
            for name in status_names
        ],
        loc="lower right",
        frameon=True,
        facecolor="#fbfaf6",
        edgecolor=GRID,
        fontsize=7.4,
    )
    axes[0].set_title("Reason assigned to every PS2D cell")

    background = np.where(statuses == "outside_eor_window", 0, 1)
    background_cmap = ListedColormap(["#ece8df", PALE])
    axes[1].pcolormesh(
        kperp_edges,
        kpar_edges,
        background.T,
        cmap=background_cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        linewidth=0.14,
        edgecolors=(1.0, 1.0, 1.0, 0.40),
    )
    image = axes[1].pcolormesh(
        kperp_edges,
        kpar_edges,
        np.ma.masked_invalid(worst_total_error_percent.T),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(
            20.0,
            min(100.0, float(np.nanmax(worst_total_error_percent))),
        ),
        shading="flat",
        linewidth=0.20,
        edgecolors="#fbfaf6",
    )
    for kp_index, kpar_index in zip(
        *np.nonzero(np.isfinite(worst_total_error_percent)), strict=True
    ):
        value = float(worst_total_error_percent[kp_index, kpar_index])
        axes[1].text(
            x_centers[kp_index],
            y_centers[kpar_index],
            f"{value:.0f}",
            ha="center",
            va="center",
            fontsize=4.0,
            color="#fbfaf6" if value >= 12.0 else INK,
        )
    fig.colorbar(
        image,
        ax=axes[1],
        label="worst FG+EoR error: physical or held-out phase [%]",
        shrink=0.92,
    )
    axes[1].set_title("Held-out error where exact broad response is available")

    for axis in axes:
        axis.plot(
            x_centers,
            boundary,
            color=INK,
            linewidth=1.25,
            linestyle=(0, (4, 3)),
        )
        axis.set(
            xlabel=r"$k_\perp\ [{\rm Mpc}^{-1}]$",
            ylabel=r"$|k_\parallel|\ [{\rm Mpc}^{-1}]$",
            xlim=(float(kperp_edges[0]), float(kperp_edges[-1])),
            ylim=(float(kpar_edges[0]), float(kpar_edges[-1])),
        )
        axis.set_xticks(x_centers[::4])
        axis.set_xticklabels([f"{value:.2f}" for value in x_centers[::4]])
        axis.set_yticks(y_centers)
        axis.set_yticklabels([f"{value:.3f}" for value in y_centers])
    fig.suptitle(
        "Full EoR-window exact-visibility Q-beta audit: "
        f"{summary['recoverable_cell_count']}/"
        f"{summary['geometric_eor_window_cell_count']} recoverable cells",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = _arguments()
    result_dir = args.result_dir
    if not (result_dir / "result.npz").is_file():
        result_dir = result_dir / "combined"
    metadata = json.loads(
        (result_dir / "result.json").read_text(encoding="utf-8")
    )
    with np.load(result_dir / "result.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    kpar_edges = np.asarray(
        resolved.contract.window_layout.kpar_edges, dtype=np.float64
    )
    geometric = resolved.window_spec.mask(
        kperp_edges[1:, None], kpar_values[None, :]
    )
    support = np.asarray(arrays["support"], dtype=bool)
    if support.shape != geometric.shape:
        raise ValueError("Result support does not match the analysis grid")
    settings = metadata["qbeta"]
    if settings.get("filter_bandwidth_scope") != "full_band":
        raise ValueError("This audit expects a full-band DPSS pre-filter")

    frequencies_hz = np.asarray(
        arrays["input_frequencies_mhz"], dtype=np.float64
    ) * 1e6
    analysis_indices = np.asarray(
        arrays["analysis_frequency_indices"], dtype=np.int64
    )
    radial_mpc_per_hz = float(
        resolved.geometry["radial_spacing_mpc"]
    ) / float(np.mean(np.diff(frequencies_hz)))
    maximum_delays = _maximum_patch_delays(
        kperp_edges=kperp_edges,
        transverse_distance_mpc=float(
            resolved.geometry["transverse_distance_mpc"]
        ),
        reference_frequency_hz=(
            float(resolved.geometry["reference_frequency_mhz"]) * 1e6
        ),
        source_corner_angle_deg=float(
            resolved.geometry["source_corner_angle_deg"]
        ),
        wedge_buffer_mpc_inv=float(
            resolved.geometry["wedge_buffer_mpc_inv"]
        ),
        radial_mpc_per_hz=radial_mpc_per_hz,
    )
    window_self, filter_sensitivity, foreground_ranks = (
        _spectral_support_diagnostics(
            frequencies_hz=frequencies_hz,
            analysis_frequency_indices=analysis_indices,
            maximum_delays_s=maximum_delays,
            dpss_eigenvalue_threshold=float(
                settings["dpss_eigenvalue_threshold"]
            ),
            taper=str(settings["spectral_taper"]),
        )
    )
    reconstructed_support = (
        geometric
        & (window_self >= float(args.minimum_window_self_fraction))
        & (
            filter_sensitivity
            >= float(args.minimum_filter_sensitivity)
        )
    )
    if not np.array_equal(reconstructed_support, support):
        difference = int(np.count_nonzero(reconstructed_support != support))
        raise ValueError(
            f"Reconstructed spectral support differs in {difference} cells"
        )

    products = _response_products(arrays, geometric, kpar_values)
    output_ids = np.asarray(arrays["output_band_ids"], dtype=np.int64)
    output_positions = {
        int(band_id): position for position, band_id in enumerate(output_ids)
    }
    exact_broad = (
        (
            products["relative_response"]
            >= float(args.minimum_relative_qbeta_response)
        )
        & (
            products["geometric_fraction"]
            >= float(args.minimum_geometric_window_fraction)
        )
    )
    pure_worst = np.max(products["heldout_pure_error"], axis=0)
    total_worst = np.max(products["heldout_total_error"], axis=0)
    foreground_worst = np.max(
        products["heldout_foreground_effect"], axis=0
    )
    target_denominator = np.maximum(np.abs(products["target"]), 1e-300)
    bank_pure_error = (
        np.abs(products["pure"] - products["target"]) / target_denominator
    )
    bank_total_error = (
        np.abs(products["total"] - products["target"]) / target_denominator
    )
    robust_total_worst = np.maximum(total_worst, bank_total_error)

    statuses = np.full(
        geometric.shape, "outside_eor_window", dtype=object
    )
    worst_total_error_percent = np.full(
        geometric.shape, np.nan, dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    for kp_index in range(geometric.shape[0]):
        for kpar_index in range(geometric.shape[1]):
            band_id = kp_index * geometric.shape[1] + kpar_index
            position = output_positions.get(band_id)
            if not geometric[kp_index, kpar_index]:
                status = "outside_eor_window"
            elif window_self[kp_index, kpar_index] < float(
                args.minimum_window_self_fraction
            ):
                status = "removed_by_dpss_response"
            elif filter_sensitivity[kp_index, kpar_index] < float(
                args.minimum_filter_sensitivity
            ):
                status = "low_filter_sensitivity"
            elif position is None:
                raise ValueError("A supported EoR cell has no Q_beta output")
            elif products["relative_response"][position] < float(
                args.minimum_relative_qbeta_response
            ):
                status = "low_exact_qbeta_response"
            elif products["geometric_fraction"][position] < float(
                args.minimum_geometric_window_fraction
            ):
                status = "response_leaks_outside_eor"
            elif bank_pure_error[position] >= float(args.maximum_cell_error):
                status = "physical_eor_closure_failure"
            elif bank_total_error[position] >= float(args.maximum_cell_error):
                status = "foreground_leakage_failure"
            elif pure_worst[position] >= float(args.maximum_cell_error):
                status = "heldout_eor_closure_failure"
            elif total_worst[position] >= float(args.maximum_cell_error):
                status = "foreground_leakage_failure"
            else:
                status = "recoverable_windowed_bandpower"
            statuses[kp_index, kpar_index] = status

            row: dict[str, Any] = {
                "kperp_index": kp_index,
                "kpar_index": kpar_index,
                "kperp_center_mpc_inv": float(
                    0.5
                    * (
                        kperp_edges[kp_index]
                        + kperp_edges[kp_index + 1]
                    )
                ),
                "kpar_mpc_inv": float(kpar_values[kpar_index]),
                "geometric_eor_window": int(
                    geometric[kp_index, kpar_index]
                ),
                "status": status,
                "dpss_rank": int(foreground_ranks[kp_index]),
                "spectral_window_self_fraction": float(
                    window_self[kp_index, kpar_index]
                ),
                "relative_filter_sensitivity": float(
                    filter_sensitivity[kp_index, kpar_index]
                ),
                "qbeta_evaluated": int(position is not None),
            }
            if position is not None:
                worst_total_error_percent[kp_index, kpar_index] = (
                    100.0 * robust_total_worst[position]
                    if exact_broad[position]
                    else math.nan
                )
                row.update(
                    {
                        "relative_qbeta_response": float(
                            products["relative_response"][position]
                        ),
                        "geometric_window_response_fraction": float(
                            products["geometric_fraction"][position]
                        ),
                        "same_cell_response_fraction": float(
                            products["same_cell_fraction"][position]
                        ),
                        "window_effective_width_source_bands": float(
                            products["effective_width"][position]
                        ),
                        "exact_broad_candidate": int(exact_broad[position]),
                        "bank_pure_error_percent": float(
                            100.0 * bank_pure_error[position]
                        ),
                        "bank_total_error_percent": float(
                            100.0 * bank_total_error[position]
                        ),
                        "heldout_worst_pure_error_percent": float(
                            100.0 * pure_worst[position]
                        ),
                        "heldout_worst_total_error_percent": float(
                            100.0 * total_worst[position]
                        ),
                        "heldout_worst_foreground_effect_percent": float(
                            100.0 * foreground_worst[position]
                        ),
                    }
                )
            rows.append(row)

    status_counts = Counter(
        status for status in statuses[geometric].reshape(-1)
    )
    recoverable_count = int(
        status_counts["recoverable_windowed_bandpower"]
    )
    geometric_count = int(np.count_nonzero(geometric))
    broad_positions = np.flatnonzero(exact_broad)
    broad_total_errors = total_worst[broad_positions]
    broad_pure_errors = pure_worst[broad_positions]
    broad_target = np.abs(products["target"][broad_positions])
    broad_weights = products["relative_response"][broad_positions]
    weighted_target_square = float(
        np.sum(broad_weights * np.square(broad_target))
    )
    broad_total_phase_errors = products["heldout_total_error"][
        :, broad_positions
    ]
    broad_pure_phase_errors = products["heldout_pure_error"][
        :, broad_positions
    ]
    if broad_positions.size and weighted_target_square > 0.0:
        broad_total_relative_l2 = np.sqrt(
            np.sum(
                broad_weights[None, :]
                * np.square(
                    broad_total_phase_errors * broad_target[None, :]
                ),
                axis=1,
            )
            / weighted_target_square
        )
        broad_pure_relative_l2 = np.sqrt(
            np.sum(
                broad_weights[None, :]
                * np.square(
                    broad_pure_phase_errors * broad_target[None, :]
                ),
                axis=1,
            )
            / weighted_target_square
        )
    else:
        phase_count = products["heldout_total_error"].shape[0]
        broad_total_relative_l2 = np.full(phase_count, np.nan)
        broad_pure_relative_l2 = np.full(phase_count, np.nan)
    broad_total_passing_counts = np.count_nonzero(
        broad_total_phase_errors < float(args.maximum_cell_error), axis=1
    )
    broad_pure_passing_counts = np.count_nonzero(
        broad_pure_phase_errors < float(args.maximum_cell_error), axis=1
    )
    broad_bank_pure_errors = bank_pure_error[broad_positions]
    broad_bank_total_errors = bank_total_error[broad_positions]
    if broad_positions.size and weighted_target_square > 0.0:
        broad_bank_pure_relative_l2 = math.sqrt(
            float(
                np.sum(
                    broad_weights
                    * np.square(
                        products["pure"][broad_positions]
                        - products["target"][broad_positions]
                    )
                )
                / weighted_target_square
            )
        )
        broad_bank_total_relative_l2 = math.sqrt(
            float(
                np.sum(
                    broad_weights
                    * np.square(
                        products["total"][broad_positions]
                        - products["target"][broad_positions]
                    )
                )
                / weighted_target_square
            )
        )
        broad_weighted_target_sum = float(
            np.sum(broad_weights * products["target"][broad_positions])
        )
        broad_bank_pure_integrated_ratio = float(
            np.sum(broad_weights * products["pure"][broad_positions])
            / broad_weighted_target_sum
        )
        broad_bank_total_integrated_ratio = float(
            np.sum(broad_weights * products["total"][broad_positions])
            / broad_weighted_target_sum
        )
        broad_heldout_pure_integrated_ratios = np.sum(
            broad_weights[None, :]
            * products["heldout_pure"][:, broad_positions],
            axis=1,
        ) / broad_weighted_target_sum
        broad_heldout_total_integrated_ratios = np.sum(
            broad_weights[None, :]
            * products["heldout_total"][:, broad_positions],
            axis=1,
        ) / broad_weighted_target_sum
    else:
        broad_bank_pure_relative_l2 = math.nan
        broad_bank_total_relative_l2 = math.nan
        broad_bank_pure_integrated_ratio = math.nan
        broad_bank_total_integrated_ratio = math.nan
        phase_count = products["heldout_total_error"].shape[0]
        broad_heldout_pure_integrated_ratios = np.full(
            phase_count, np.nan
        )
        broad_heldout_total_integrated_ratios = np.full(
            phase_count, np.nan
        )
    evaluated_positions = np.arange(output_ids.size, dtype=np.int64)
    evaluated_total_passing_counts = np.count_nonzero(
        products["heldout_total_error"] < float(args.maximum_cell_error),
        axis=1,
    )
    evaluated_pure_passing_counts = np.count_nonzero(
        products["heldout_pure_error"] < float(args.maximum_cell_error),
        axis=1,
    )
    evaluated_all_phase_total_pass = np.all(
        products["heldout_total_error"] < float(args.maximum_cell_error),
        axis=0,
    )
    evaluated_all_phase_pure_pass = np.all(
        products["heldout_pure_error"] < float(args.maximum_cell_error),
        axis=0,
    )
    evaluated_robust_total_pass = (
        evaluated_all_phase_total_pass
        & (bank_pure_error < float(args.maximum_cell_error))
        & (bank_total_error < float(args.maximum_cell_error))
    )
    response_percentiles = [0.0, 25.0, 50.0, 75.0, 100.0]
    summary = {
        "schema": "visibility_qbeta_full_window_audit",
        "schema_version": 1,
        "analysis_contract_sha256": metadata[
            "analysis_contract_sha256"
        ],
        "frequency_contract_sha256": metadata[
            "frequency_contract_sha256"
        ],
        "visibility_bank_sha256": metadata["visibility_bank_sha256"],
        "combined_rows_per_kperp_bin": metadata[
            "combined_rows_per_kperp_bin"
        ],
        "heldout_phase_count": int(
            products["heldout_total_error"].shape[0]
        ),
        "geometric_eor_window_cell_count": geometric_count,
        "spectral_filter_supported_cell_count": int(
            np.count_nonzero(support)
        ),
        "qbeta_evaluated_geometric_cell_count": int(output_ids.size),
        "evaluated_total_passing_cell_count_per_phase": [
            int(value) for value in evaluated_total_passing_counts
        ],
        "evaluated_pure_passing_cell_count_per_phase": [
            int(value) for value in evaluated_pure_passing_counts
        ],
        "evaluated_all_phase_total_passing_cell_count": int(
            np.count_nonzero(evaluated_all_phase_total_pass)
        ),
        "evaluated_all_phase_pure_passing_cell_count": int(
            np.count_nonzero(evaluated_all_phase_pure_pass)
        ),
        "evaluated_bank_pure_passing_cell_count": int(
            np.count_nonzero(
                bank_pure_error < float(args.maximum_cell_error)
            )
        ),
        "evaluated_bank_total_passing_cell_count": int(
            np.count_nonzero(
                bank_total_error < float(args.maximum_cell_error)
            )
        ),
        "evaluated_robust_total_passing_cell_count": int(
            np.count_nonzero(evaluated_robust_total_pass)
        ),
        "evaluated_worst_total_cell_error": float(
            np.max(
                products["heldout_total_error"][:, evaluated_positions]
            )
        ),
        "evaluated_worst_pure_cell_error": float(
            np.max(
                products["heldout_pure_error"][:, evaluated_positions]
            )
        ),
        "evaluated_worst_bank_pure_cell_error": float(
            np.max(bank_pure_error[evaluated_positions])
        ),
        "evaluated_worst_bank_total_cell_error": float(
            np.max(bank_total_error[evaluated_positions])
        ),
        "evaluated_worst_foreground_effect": float(
            np.max(
                products["heldout_foreground_effect"][
                    :, evaluated_positions
                ]
            )
        ),
        "evaluated_geometric_window_response_fraction_percentiles": [
            float(value)
            for value in np.percentile(
                products["geometric_fraction"][evaluated_positions],
                response_percentiles,
            )
        ],
        "evaluated_same_cell_response_fraction_percentiles": [
            float(value)
            for value in np.percentile(
                products["same_cell_fraction"][evaluated_positions],
                response_percentiles,
            )
        ],
        "exact_broad_candidate_count": int(broad_positions.size),
        "exact_broad_fraction_of_geometric_window": (
            int(broad_positions.size) / geometric_count
        ),
        "recoverable_cell_count": recoverable_count,
        "recoverable_fraction_of_geometric_window": (
            recoverable_count / geometric_count
        ),
        "status_counts_within_geometric_window": dict(status_counts),
        "broad_total_passing_cell_count_per_phase": [
            int(value) for value in broad_total_passing_counts
        ],
        "broad_pure_passing_cell_count_per_phase": [
            int(value) for value in broad_pure_passing_counts
        ],
        "broad_minimum_passing_cell_count_over_phases": int(
            np.min(broad_total_passing_counts)
        ),
        "broad_total_weighted_relative_l2_per_phase": [
            float(value) for value in broad_total_relative_l2
        ],
        "broad_pure_weighted_relative_l2_per_phase": [
            float(value) for value in broad_pure_relative_l2
        ],
        "broad_worst_total_weighted_relative_l2": float(
            np.max(broad_total_relative_l2)
        ),
        "broad_worst_pure_weighted_relative_l2": float(
            np.max(broad_pure_relative_l2)
        ),
        "broad_worst_total_cell_error": (
            float(np.max(broad_total_errors))
            if broad_total_errors.size
            else math.nan
        ),
        "broad_worst_pure_cell_error": (
            float(np.max(broad_pure_errors))
            if broad_pure_errors.size
            else math.nan
        ),
        "broad_worst_foreground_effect": (
            float(np.max(foreground_worst[broad_positions]))
            if broad_positions.size
            else math.nan
        ),
        "broad_median_worst_total_cell_error": (
            float(np.median(broad_total_errors))
            if broad_total_errors.size
            else math.nan
        ),
        "broad_bank_pure_passing_cell_count": int(
            np.count_nonzero(
                broad_bank_pure_errors < float(args.maximum_cell_error)
            )
        ),
        "broad_bank_total_passing_cell_count": int(
            np.count_nonzero(
                broad_bank_total_errors < float(args.maximum_cell_error)
            )
        ),
        "broad_bank_pure_weighted_relative_l2": (
            broad_bank_pure_relative_l2
        ),
        "broad_bank_total_weighted_relative_l2": (
            broad_bank_total_relative_l2
        ),
        "broad_bank_pure_integrated_power_ratio": (
            broad_bank_pure_integrated_ratio
        ),
        "broad_bank_total_integrated_power_ratio": (
            broad_bank_total_integrated_ratio
        ),
        "broad_heldout_pure_integrated_power_ratio_per_phase": [
            float(value) for value in broad_heldout_pure_integrated_ratios
        ],
        "broad_heldout_total_integrated_power_ratio_per_phase": [
            float(value) for value in broad_heldout_total_integrated_ratios
        ],
        "broad_robust_total_passing_cell_count": int(
            np.count_nonzero(
                (broad_bank_pure_errors < float(args.maximum_cell_error))
                & (broad_bank_total_errors < float(args.maximum_cell_error))
                & np.all(
                    broad_total_phase_errors
                    < float(args.maximum_cell_error),
                    axis=0,
                )
            )
        ),
        "broad_same_cell_response_fraction_percentiles": [
            float(value)
            for value in np.percentile(
                products["same_cell_fraction"][broad_positions],
                response_percentiles,
            )
        ],
        "broad_window_effective_width_source_bands_percentiles": [
            float(value)
            for value in np.percentile(
                products["effective_width"][broad_positions],
                response_percentiles,
            )
        ],
        "thresholds": {
            "minimum_window_self_fraction": float(
                args.minimum_window_self_fraction
            ),
            "minimum_filter_sensitivity": float(
                args.minimum_filter_sensitivity
            ),
            "minimum_relative_qbeta_response": float(
                args.minimum_relative_qbeta_response
            ),
            "minimum_geometric_window_fraction": float(
                args.minimum_geometric_window_fraction
            ),
            "maximum_cell_error": float(args.maximum_cell_error),
        },
        "interpretation": (
            "Recoverable cells are response-windowed bandpowers, not "
            "independent delta-function sky bins."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "cell_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    boundary = np.maximum(
        float(resolved.window_spec.kpar_min),
        float(resolved.window_spec.wedge_slope)
        * (0.5 * (kperp_edges[:-1] + kperp_edges[1:]))
        + float(resolved.window_spec.wedge_intercept),
    )
    _plot(
        path=args.output_dir / "full_window_audit.png",
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
        boundary=boundary,
        statuses=statuses,
        worst_total_error_percent=worst_total_error_percent,
        summary=summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
