#!/usr/bin/env python3
"""Plot the exact-visibility Q_beta recovery and audit its EoR-window coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fg-rmw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from visibility_qbeta import band_selection_coverage  # noqa: E402


INK = "#17324d"
TEAL = "#147d7e"
GOLD = "#e3a018"
VERMILION = "#d24b2a"
PALE = "#dce9e6"
GRID = "#d7d2c8"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ps2d_v2_32high_isobeam_patch.json",
    )
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=ROOT / "docs/results/visibility_qbeta_32freq_20260723_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument(
        "--coverage-summary",
        type=Path,
        default=ROOT / "docs/results/visibility_qbeta_coverage_20260724_summary.json",
    )
    parser.add_argument(
        "--cell-table",
        type=Path,
        default=ROOT / "docs/results/visibility_qbeta_cell_errors_20260724.csv",
    )
    parser.add_argument("--dpss-supported-band-count", type=int, default=337)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
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


def _reporting_mask(
    archive: np.lib.npyio.NpzFile, shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    positions = np.asarray(archive["reporting_source_positions"], dtype=np.int64)
    kperp = np.asarray(archive["source_band_kperp_indices"], dtype=np.int64)
    kpar = np.asarray(archive["source_band_kpar_indices"], dtype=np.int64)
    valid_positions = positions[kpar[positions] < shape[1]]
    mask[kperp[valid_positions], kpar[valid_positions]] = True
    return mask


def _selected_mask(
    archive: np.lib.npyio.NpzFile, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_positions = np.asarray(
        archive["qbeta_selected_window_positions"], dtype=np.int64
    )
    output_ids = np.asarray(archive["output_band_ids"], dtype=np.int64)
    selected_ids = output_ids[selected_positions]
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[selected_ids] = True
    return mask, selected_ids, selected_positions


def _plot_coverage(
    *,
    output: Path,
    kperp_edges: np.ndarray,
    kpar_edges: np.ndarray,
    geometric: np.ndarray,
    evaluated: np.ndarray,
    reporting: np.ndarray,
    selected: np.ndarray,
    boundary: np.ndarray,
    metrics: dict[str, float | int],
    dpss_supported_count: int,
) -> None:
    category = np.zeros(geometric.shape, dtype=np.int8)
    category[geometric] = 1
    category[evaluated & geometric] = 2
    category[reporting] = 3
    category[selected] = 4
    colors = ["#ece8df", PALE, "#a9d3d2", "#efc86f", VERMILION]
    labels = [
        "Outside geometric EoR window",
        "Geometric EoR window",
        "Q-beta evaluated support",
        "Predeclared reporting region",
        "Recovered output window",
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
    ax.pcolormesh(
        kperp_edges,
        kpar_edges,
        category.T,
        cmap=cmap,
        norm=norm,
        shading="flat",
        linewidth=0.16,
        edgecolors=(1.0, 1.0, 1.0, 0.26),
    )
    centers = 0.5 * (kperp_edges[:-1] + kperp_edges[1:])
    ax.plot(
        centers,
        boundary,
        color=INK,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        label="Frozen wedge/floor boundary",
    )
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(colors, labels, strict=True)
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color=INK,
            linewidth=1.5,
            linestyle=(0, (4, 3)),
            label="Frozen wedge/floor boundary",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        ncols=2,
        fontsize=8.3,
    )
    fraction = 100.0 * float(metrics["selected_fraction_of_geometric_window_bands"])
    mode_fraction = 100.0 * float(
        metrics["selected_fraction_of_geometric_window_fft_modes"]
    )
    power_fraction = 100.0 * float(
        metrics["selected_fraction_of_geometric_window_injected_power"]
    )
    selected_count = int(metrics["selected_band_count"])
    geometric_count = int(metrics["geometric_window_band_count"])
    note = (
        f"Direct output footprint: {selected_count}/{geometric_count} cells"
        f" = {fraction:.2f}%\n"
        f"FFT-mode weighted: {mode_fraction:.2f}%   "
        f"Injected-EoR-power weighted: {power_fraction:.2f}%\n"
        f"Earlier all-row DPSS support: {dpss_supported_count}/{geometric_count}; "
        f"selected/DPSS = {100.0 * selected_count / dpss_supported_count:.2f}%"
    )
    ax.text(
        0.985,
        0.035,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#fbfaf6",
            "edgecolor": GRID,
            "alpha": 0.96,
        },
    )
    ax.set(
        xlabel=r"$k_\perp\ [{\rm Mpc}^{-1}]$",
        ylabel=r"$|k_\parallel|\ [{\rm Mpc}^{-1}]$",
        xlim=(float(kperp_edges[0]), float(kperp_edges[-1])),
        ylim=(float(kpar_edges[0]), float(kpar_edges[-1])),
        title="Exact-visibility Q-beta recovery footprint",
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_recovery(
    *,
    output: Path,
    kperp_edges: np.ndarray,
    kpar_edges: np.ndarray,
    selected_ids: np.ndarray,
    target: np.ndarray,
    estimate: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    nkpar = kpar_edges.size - 1
    kperp_indices, kpar_indices = np.divmod(selected_ids, nkpar)
    unique_kperp = np.unique(kperp_indices)
    unique_kpar = np.unique(kpar_indices)
    if (
        selected_ids.size != unique_kperp.size * unique_kpar.size
        or np.any(np.diff(unique_kperp) != 1)
        or np.any(np.diff(unique_kpar) != 1)
    ):
        raise ValueError("selected recovery windows must form a rectangular grid")
    target_grid = np.full(
        (unique_kperp.size, unique_kpar.size), np.nan, dtype=np.float64
    )
    estimate_grid = np.full_like(target_grid, np.nan)
    for index, (kp_index, kz_index) in enumerate(
        zip(kperp_indices, kpar_indices, strict=True)
    ):
        x_index = int(np.flatnonzero(unique_kperp == kp_index)[0])
        y_index = int(np.flatnonzero(unique_kpar == kz_index)[0])
        target_grid[x_index, y_index] = target[index]
        estimate_grid[x_index, y_index] = estimate[index]
    relative = (estimate_grid - target_grid) / target_grid
    log_target = np.log10(target_grid)
    log_estimate = np.log10(estimate_grid)
    lower = float(min(np.nanmin(log_target), np.nanmin(log_estimate)))
    upper = float(max(np.nanmax(log_target), np.nanmax(log_estimate)))
    x_edges = kperp_edges[unique_kperp[0] : unique_kperp[-1] + 2]
    y_edges = kpar_edges[unique_kpar[0] : unique_kpar[-1] + 2]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    power_images = []
    for ax, values, title in (
        (axes[0], log_target, r"Window target $\log_{10}(WP_{\rm EoR})$"),
        (axes[1], log_estimate, r"Recovered $\log_{10}(\widehat P)$"),
    ):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values.T,
            cmap="magma",
            vmin=lower,
            vmax=upper,
            shading="flat",
            linewidth=0.45,
            edgecolors="#fbfaf6",
        )
        power_images.append(image)
        ax.set_title(title)
    residual_limit = max(0.2, float(np.nanmax(np.abs(relative))))
    residual_image = axes[2].pcolormesh(
        x_edges,
        y_edges,
        100.0 * relative.T,
        cmap="RdBu_r",
        vmin=-100.0 * residual_limit,
        vmax=100.0 * residual_limit,
        shading="flat",
        linewidth=0.45,
        edgecolors="#fbfaf6",
    )
    axes[2].set_title(r"Fractional residual $(\widehat P-WP)/WP$ [\%]")
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    for ax in axes:
        ax.set_xlabel(r"$k_\perp\ [{\rm Mpc}^{-1}]$")
        ax.set_yticks(y_centers)
        ax.set_yticklabels([f"{value:.3f}" for value in y_centers])
    axes[0].set_ylabel(r"$|k_\parallel|\ [{\rm Mpc}^{-1}]$")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    power_colorbar = fig.colorbar(power_images[0], ax=axes[:2], shrink=0.86, pad=0.02)
    power_colorbar.set_label("log10 bandpower [arbitrary normalization]")
    residual_colorbar = fig.colorbar(residual_image, ax=axes[2], shrink=0.86, pad=0.02)
    residual_colorbar.set_label("percent")
    fig.suptitle(
        "Noiseless full-EoR closure: "
        f"L2={100.0 * metrics['relative_l2']:.2f}%, "
        f"integrated ratio={metrics['integrated_power_ratio']:.3f}, "
        f"max error={100.0 * metrics['maximum_relative_window_error']:.2f}%",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _output_grid(
    values: np.ndarray,
    output_band_ids: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, np.nan, dtype=np.float64)
    flat = grid.reshape(-1)
    flat[np.asarray(output_band_ids, dtype=np.int64)] = np.asarray(
        values, dtype=np.float64
    )
    return grid


def _plot_full_grid_errors(
    *,
    output: Path,
    kperp_edges: np.ndarray,
    kpar_edges: np.ndarray,
    geometric: np.ndarray,
    selected: np.ndarray,
    boundary: np.ndarray,
    output_band_ids: np.ndarray,
    pure_error_percent: np.ndarray,
    total_error_percent: np.ndarray,
) -> None:
    shape = geometric.shape
    pure_grid = _output_grid(pure_error_percent, output_band_ids, shape)
    total_grid = _output_grid(total_error_percent, output_band_ids, shape)
    finite_values = np.concatenate(
        (pure_grid[np.isfinite(pure_grid)], total_grid[np.isfinite(total_grid)])
    )
    error_limit = max(
        20.0,
        5.0 * np.ceil(float(np.max(np.abs(finite_values))) / 5.0),
    )
    background = np.asarray(geometric, dtype=np.int8)
    background_cmap = ListedColormap(["#ece8df", "#dce9e6"])
    background_norm = BoundaryNorm([-0.5, 0.5, 1.5], background_cmap.N)
    x_centers = 0.5 * (kperp_edges[:-1] + kperp_edges[1:])
    y_centers = 0.5 * (kpar_edges[:-1] + kpar_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(19.0, 7.2), constrained_layout=True)
    images = []
    for ax, values, title in (
        (
            axes[0],
            pure_grid,
            r"Pure-EoR closure: $(\widehat P-WP_{\rm EoR})/WP_{\rm EoR}$",
        ),
        (
            axes[1],
            total_grid,
            r"FG+EoR closure: $(\widehat P_{\rm total}-WP_{\rm EoR})/WP_{\rm EoR}$",
        ),
    ):
        ax.pcolormesh(
            kperp_edges,
            kpar_edges,
            background.T,
            cmap=background_cmap,
            norm=background_norm,
            shading="flat",
            linewidth=0.13,
            edgecolors=(1.0, 1.0, 1.0, 0.42),
        )
        image = ax.pcolormesh(
            kperp_edges,
            kpar_edges,
            np.ma.masked_invalid(values.T),
            cmap="RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
            shading="flat",
            linewidth=0.22,
            edgecolors="#fbfaf6",
        )
        images.append(image)
        ax.plot(
            x_centers,
            boundary,
            color=INK,
            linewidth=1.25,
            linestyle=(0, (4, 3)),
        )
        for kp_index, kpar_index in zip(*np.nonzero(selected), strict=True):
            ax.add_patch(
                Rectangle(
                    (kperp_edges[kp_index], kpar_edges[kpar_index]),
                    kperp_edges[kp_index + 1] - kperp_edges[kp_index],
                    kpar_edges[kpar_index + 1] - kpar_edges[kpar_index],
                    fill=False,
                    edgecolor=GOLD,
                    linewidth=1.35,
                    zorder=5,
                )
            )
        for band_id in np.asarray(output_band_ids, dtype=np.int64):
            kp_index, kpar_index = divmod(int(band_id), shape[1])
            value = float(values[kp_index, kpar_index])
            text_color = "#fbfaf6" if abs(value) >= 11.0 else INK
            ax.text(
                x_centers[kp_index],
                y_centers[kpar_index],
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=4.6,
                color=text_color,
                zorder=6,
            )
        ax.set(
            xlabel=r"$k_\perp\ [{\rm Mpc}^{-1}]$",
            xlim=(float(kperp_edges[0]), float(kperp_edges[-1])),
            ylim=(float(kpar_edges[0]), float(kpar_edges[-1])),
            title=title,
        )
        ax.set_xticks(x_centers[::4])
        ax.set_xticklabels([f"{value:.2f}" for value in x_centers[::4]])
        ax.set_yticks(y_centers)
        ax.set_yticklabels([f"{value:.3f}" for value in y_centers])
    axes[0].set_ylabel(r"$|k_\parallel|\ [{\rm Mpc}^{-1}]$")
    axes[1].set_yticklabels([])
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.90, pad=0.012)
    colorbar.set_label("signed fractional error [%]")
    legend = [
        Patch(facecolor="#ece8df", edgecolor="none", label="Outside EoR window"),
        Patch(
            facecolor="#dce9e6",
            edgecolor="none",
            label="EoR-window cell not evaluated",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=GOLD,
            linewidth=1.35,
            label="Original 20 response-selected windows",
        ),
        plt.Line2D(
            [0],
            [0],
            color=INK,
            linewidth=1.25,
            linestyle=(0, (4, 3)),
            label="Frozen wedge/floor boundary",
        ),
    ]
    axes[0].legend(
        handles=legend,
        loc="lower right",
        ncols=2,
        frameon=True,
        facecolor="#fbfaf6",
        edgecolor=GRID,
        framealpha=0.94,
        fontsize=7.8,
    )
    fig.suptitle(
        "Exact-visibility Q-beta cell closure over the full cylindrical grid\n"
        "Numbers are signed percent errors for 112 evaluated response-windowed "
        "bandpowers; blank EoR-window cells were not evaluated",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output, dpi=240)
    plt.close(fig)


def _write_cell_table(
    *,
    output: Path,
    kperp_edges: np.ndarray,
    kpar_values: np.ndarray,
    geometric: np.ndarray,
    selected: np.ndarray,
    output_band_ids: np.ndarray,
    relative_response: np.ndarray,
    reporting_window_fraction: np.ndarray,
    geometric_window_fraction: np.ndarray,
    same_cell_fraction: np.ndarray,
    effective_width: np.ndarray,
    target_power: np.ndarray,
    pure_estimate: np.ndarray,
    total_estimate: np.ndarray,
    foreground_estimate: np.ndarray,
) -> None:
    shape = geometric.shape
    output_positions = {
        int(band_id): int(position)
        for position, band_id in enumerate(np.asarray(output_band_ids, dtype=np.int64))
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kperp_index",
        "kpar_index",
        "kperp_low_mpc_inv",
        "kperp_high_mpc_inv",
        "kperp_center_mpc_inv",
        "kpar_mpc_inv",
        "geometric_eor_window",
        "qbeta_evaluated",
        "original_response_selected",
        "qbeta_relative_response",
        "reporting_region_window_fraction",
        "geometric_eor_window_response_fraction",
        "same_source_cell_response_fraction",
        "window_effective_width_source_bands",
        "target_windowed_power",
        "pure_eor_recovered_windowed_power",
        "pure_eor_signed_error_percent",
        "total_recovered_windowed_power",
        "total_signed_error_percent",
        "foreground_windowed_power",
        "foreground_over_target",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for kp_index in range(shape[0]):
            for kpar_index in range(shape[1]):
                band_id = kp_index * shape[1] + kpar_index
                position = output_positions.get(band_id)
                row: dict[str, Any] = {
                    "kperp_index": kp_index,
                    "kpar_index": kpar_index,
                    "kperp_low_mpc_inv": float(kperp_edges[kp_index]),
                    "kperp_high_mpc_inv": float(kperp_edges[kp_index + 1]),
                    "kperp_center_mpc_inv": float(
                        0.5 * (kperp_edges[kp_index] + kperp_edges[kp_index + 1])
                    ),
                    "kpar_mpc_inv": float(kpar_values[kpar_index]),
                    "geometric_eor_window": int(geometric[kp_index, kpar_index]),
                    "qbeta_evaluated": int(position is not None),
                    "original_response_selected": int(selected[kp_index, kpar_index]),
                }
                if position is not None:
                    target = float(target_power[position])
                    pure = float(pure_estimate[position])
                    total = float(total_estimate[position])
                    foreground = float(foreground_estimate[position])
                    row.update(
                        {
                            "qbeta_relative_response": float(
                                relative_response[position]
                            ),
                            "reporting_region_window_fraction": float(
                                reporting_window_fraction[position]
                            ),
                            "geometric_eor_window_response_fraction": float(
                                geometric_window_fraction[position]
                            ),
                            "same_source_cell_response_fraction": float(
                                same_cell_fraction[position]
                            ),
                            "window_effective_width_source_bands": float(
                                effective_width[position]
                            ),
                            "target_windowed_power": target,
                            "pure_eor_recovered_windowed_power": pure,
                            "pure_eor_signed_error_percent": float(
                                100.0 * (pure - target) / target
                            ),
                            "total_recovered_windowed_power": total,
                            "total_signed_error_percent": float(
                                100.0 * (total - target) / target
                            ),
                            "foreground_windowed_power": foreground,
                            "foreground_over_target": foreground / target,
                        }
                    )
                writer.writerow(row)


def _plot_context_progression(
    *, output: Path, progression: list[dict[str, Any]]
) -> None:
    labels = [
        "40 target\nbands",
        "512 full\n(no Nyq.)",
        "544 full\n(+ Nyq.)",
        "544 + 2x\nrows",
    ]
    l2 = np.asarray([item["full_eor_relative_l2"] for item in progression], dtype=float)
    ratio = np.asarray(
        [item["full_eor_integrated_power_ratio"] for item in progression],
        dtype=float,
    )
    passing = np.asarray(
        [
            item["passing_window_count"] / item["selected_window_count"]
            for item in progression
        ],
        dtype=float,
    )
    x = np.arange(len(progression))
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.4), constrained_layout=True)
    axes[0].bar(x, l2, color=[GOLD, TEAL, TEAL, VERMILION], width=0.68)
    axes[0].axhline(0.2, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[0].set_ylabel("Full-EoR relative L2")
    axes[0].set_title("Source-context closure")
    for index, value in enumerate(l2):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)

    axes[1].plot(x, ratio, color=TEAL, marker="o", linewidth=2.0)
    axes[1].axhspan(0.8, 1.2, color=PALE, alpha=0.75)
    axes[1].axhline(1.0, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[1].set_ylim(0.75, 1.48)
    axes[1].set_ylabel("Integrated recovered / target")
    axes[1].set_title("Power normalization")
    for index, value in enumerate(ratio):
        axes[1].text(index, value + 0.035, f"{value:.3f}", ha="center", fontsize=8)

    axes[2].bar(x, 100.0 * passing, color=[GOLD, TEAL, TEAL, VERMILION], width=0.68)
    axes[2].axhline(100.0, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[2].set_ylim(0.0, 112.0)
    axes[2].set_ylabel("Windows passing 20% gate [%]")
    axes[2].set_title("Per-window gate")
    for index, value in enumerate(passing):
        axes[2].text(
            index, 100.0 * value + 2.5, f"{100.0 * value:.0f}%", ha="center", fontsize=8
        )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Why complete sky-band response context is required",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = _arguments()
    config = _load_json(args.config)
    resolved = resolve_mode_first_analysis(config)
    result_meta = _load_json(args.result_json)
    run_summary = _load_json(args.run_summary)
    archive = np.load(args.result_npz)

    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    kpar_edges = np.asarray(
        resolved.contract.window_layout.kpar_edges, dtype=np.float64
    )
    shape = (kperp_edges.size - 1, kpar_values.size)
    geometric = resolved.window_spec.mask(kperp_edges[1:, None], kpar_values[None, :])
    selected, selected_ids, selected_positions = _selected_mask(archive, shape)
    output_band_ids = np.asarray(archive["output_band_ids"], dtype=np.int64)
    reporting = _reporting_mask(archive, shape)
    evaluated = np.asarray(archive["support"], dtype=bool)
    if evaluated.shape != shape:
        raise ValueError("result support shape differs from frozen analysis grid")
    if not np.all(selected <= evaluated):
        raise ValueError("selected windows must lie in evaluated support")
    cell_weights = np.diff(kperp_edges)[:, None] * np.diff(kpar_edges)[None, :]
    coverage = band_selection_coverage(
        geometric_window=geometric,
        selected_output_band_ids=selected_ids,
        source_kperp_indices=archive["source_band_kperp_indices"],
        source_kpar_indices=archive["source_band_kpar_indices"],
        source_mode_counts=archive["source_band_mode_counts"],
        source_bandpower=archive["restricted_eor_source_power"],
        output_cell_weights=cell_weights,
        reporting_source_positions=archive["reporting_source_positions"],
    )
    dpss_supported_count = int(args.dpss_supported_band_count)
    coverage["dpss_supported_band_count"] = dpss_supported_count
    coverage["selected_fraction_of_dpss_supported_bands"] = float(
        int(coverage["selected_band_count"]) / dpss_supported_count
    )
    coverage["qbeta_evaluated_output_band_count"] = int(np.count_nonzero(evaluated))
    coverage["selected_fraction_of_qbeta_evaluated_output_bands"] = float(
        int(coverage["selected_band_count"]) / np.count_nonzero(evaluated)
    )

    qbeta_window = np.asarray(archive["qbeta_window"], dtype=np.float64)
    source_power = np.asarray(archive["restricted_eor_source_power"], dtype=np.float64)
    target_all = qbeta_window @ source_power
    estimate_all = np.asarray(archive["full_eor_windowed_power"], dtype=np.float64)[0]
    total_estimate_all = np.asarray(archive["total_windowed_power"], dtype=np.float64)[
        0
    ]
    response_row_sum = np.sum(
        np.asarray(archive["calibration_response"], dtype=np.float64), axis=1
    )
    foreground_estimate_all = np.divide(
        np.asarray(archive["bank_foreground_q"], dtype=np.float64),
        response_row_sum,
        out=np.full(response_row_sum.shape, np.nan, dtype=np.float64),
        where=response_row_sum > 0.0,
    )
    pure_error_percent = 100.0 * (estimate_all - target_all) / target_all
    total_error_percent = 100.0 * (total_estimate_all - target_all) / target_all
    source_kperp_indices = np.asarray(
        archive["source_band_kperp_indices"], dtype=np.int64
    )
    source_kpar_indices = np.asarray(
        archive["source_band_kpar_indices"], dtype=np.int64
    )
    source_in_geometric_window = np.zeros(source_kperp_indices.shape, dtype=bool)
    source_in_output_radial_range = source_kpar_indices < shape[1]
    source_in_geometric_window[source_in_output_radial_range] = geometric[
        source_kperp_indices[source_in_output_radial_range],
        source_kpar_indices[source_in_output_radial_range],
    ]
    geometric_window_response_fraction = np.sum(
        qbeta_window[:, source_in_geometric_window], axis=1
    )
    same_cell_response_fraction = np.empty(output_band_ids.size, dtype=np.float64)
    for output_position, output_band_id in enumerate(output_band_ids):
        output_kperp_index, output_kpar_index = divmod(int(output_band_id), shape[1])
        source_match = np.flatnonzero(
            (source_kperp_indices == output_kperp_index)
            & (source_kpar_indices == output_kpar_index)
        )
        if source_match.size != 1:
            raise ValueError("Each output cell must match exactly one source cell")
        same_cell_response_fraction[output_position] = qbeta_window[
            output_position, source_match[0]
        ]
    window_effective_width = np.divide(
        1.0,
        np.sum(np.square(qbeta_window), axis=1),
        out=np.full(output_band_ids.size, np.inf, dtype=np.float64),
        where=np.sum(np.square(qbeta_window), axis=1) > 0.0,
    )
    selected_target = target_all[selected_positions]
    selected_estimate = estimate_all[selected_positions]
    if np.any(selected_target <= 0.0) or np.any(selected_estimate <= 0.0):
        raise ValueError("log recovery plot requires positive selected powers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    boundary = np.maximum(
        float(resolved.window_spec.kpar_min),
        float(resolved.window_spec.wedge_slope)
        * (0.5 * (kperp_edges[:-1] + kperp_edges[1:]))
        + float(resolved.window_spec.wedge_intercept),
    )
    figure_paths = {
        "coverage": args.output_dir / "visibility_qbeta_eor_window_coverage.png",
        "recovery": args.output_dir / "visibility_qbeta_ps2d_recovery.png",
        "full_grid_errors": (
            args.output_dir / "visibility_qbeta_full_grid_cell_errors.png"
        ),
        "context": args.output_dir / "visibility_qbeta_context_progression.png",
    }
    _plot_coverage(
        output=figure_paths["coverage"],
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
        geometric=geometric,
        evaluated=evaluated,
        reporting=reporting,
        selected=selected,
        boundary=boundary,
        metrics=coverage,
        dpss_supported_count=dpss_supported_count,
    )
    full_eor_metrics = run_summary["windowed_results"]["full_eor"]
    _plot_recovery(
        output=figure_paths["recovery"],
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
        selected_ids=selected_ids,
        target=selected_target,
        estimate=selected_estimate,
        metrics=full_eor_metrics,
    )
    _plot_full_grid_errors(
        output=figure_paths["full_grid_errors"],
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
        geometric=geometric,
        selected=selected,
        boundary=boundary,
        output_band_ids=output_band_ids,
        pure_error_percent=pure_error_percent,
        total_error_percent=total_error_percent,
    )
    _plot_context_progression(
        output=figure_paths["context"],
        progression=run_summary["source_context_progression"],
    )
    _write_cell_table(
        output=args.cell_table,
        kperp_edges=kperp_edges,
        kpar_values=kpar_values,
        geometric=geometric,
        selected=selected,
        output_band_ids=output_band_ids,
        relative_response=np.asarray(
            archive["qbeta_relative_response"], dtype=np.float64
        ),
        reporting_window_fraction=np.asarray(
            archive["qbeta_target_window_fraction"], dtype=np.float64
        ),
        geometric_window_fraction=geometric_window_response_fraction,
        same_cell_fraction=same_cell_response_fraction,
        effective_width=window_effective_width,
        target_power=target_all,
        pure_estimate=estimate_all,
        total_estimate=total_estimate_all,
        foreground_estimate=foreground_estimate_all,
    )

    selected_output_mask = np.zeros(output_band_ids.size, dtype=bool)
    selected_output_mask[selected_positions] = True
    all_cell_diagnostic = {
        "status": (
            "post-hoc diagnostic only; the original 20-window response contract "
            "remains the frozen main result"
        ),
        "evaluated_cell_count": int(output_band_ids.size),
        "all_evaluated_cells_are_inside_geometric_eor_window": bool(
            np.all(geometric.reshape(-1)[output_band_ids])
        ),
        "pure_eor": {
            "maximum_absolute_error": float(np.max(np.abs(pure_error_percent)) / 100.0),
            "median_absolute_error": float(
                np.median(np.abs(pure_error_percent)) / 100.0
            ),
            "cells_below_20_percent": int(
                np.count_nonzero(np.abs(pure_error_percent) < 20.0)
            ),
        },
        "foreground_plus_eor": {
            "maximum_absolute_error": float(
                np.max(np.abs(total_error_percent)) / 100.0
            ),
            "median_absolute_error": float(
                np.median(np.abs(total_error_percent)) / 100.0
            ),
            "cells_below_20_percent": int(
                np.count_nonzero(np.abs(total_error_percent) < 20.0)
            ),
        },
        "not_originally_selected_but_below_20_percent": {
            "pure_eor": int(
                np.count_nonzero(
                    (~selected_output_mask) & (np.abs(pure_error_percent) < 20.0)
                )
            ),
            "foreground_plus_eor": int(
                np.count_nonzero(
                    (~selected_output_mask) & (np.abs(total_error_percent) < 20.0)
                )
            ),
        },
        "geometric_eor_window_response_fraction": {
            "minimum": float(np.min(geometric_window_response_fraction)),
            "median": float(np.median(geometric_window_response_fraction)),
            "cells_at_least_0p95": int(
                np.count_nonzero(geometric_window_response_fraction >= 0.95)
            ),
        },
        "window_localization": {
            "same_cell_fraction_minimum": float(np.min(same_cell_response_fraction)),
            "same_cell_fraction_median": float(np.median(same_cell_response_fraction)),
            "same_cell_fraction_maximum": float(np.max(same_cell_response_fraction)),
            "effective_width_minimum_source_bands": float(
                np.min(window_effective_width)
            ),
            "effective_width_median_source_bands": float(
                np.median(window_effective_width)
            ),
            "effective_width_maximum_source_bands": float(
                np.max(window_effective_width)
            ),
        },
    }

    output = {
        "schema": "visibility_qbeta_eor_window_coverage",
        "schema_version": 2,
        "date": "2026-07-24",
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "result_json_sha256": _sha256(args.result_json),
        "result_npz_sha256": _sha256(args.result_npz),
        "operator": {
            "primary_beam": False,
            "uv_gridding": False,
            "fixed_baseline_time_rows": True,
            "exact_direct_dft_with_w_term": True,
            "chromatic_baseline_migration": True,
            "channel_averaging_khz": 100.0,
            "time_averaging_s": 10.0,
        },
        "coverage": coverage,
        "selected_geometry": {
            "kperp_indices": np.unique(selected_ids // shape[1]).astype(int).tolist(),
            "kpar_indices": np.unique(selected_ids % shape[1]).astype(int).tolist(),
            "kperp_edge_range_mpc_inv": [
                float(kperp_edges[np.min(selected_ids // shape[1])]),
                float(kperp_edges[np.max(selected_ids // shape[1]) + 1]),
            ],
            "kpar_values_mpc_inv": kpar_values[
                np.unique(selected_ids % shape[1])
            ].tolist(),
            "median_window_effective_width_source_bands": float(
                result_meta["windowed_candidate"]["median_window_effective_width"]
            ),
        },
        "full_eor_recovery": full_eor_metrics,
        "all_evaluated_cell_diagnostic": all_cell_diagnostic,
        "foreground_to_target_integrated_absolute_ratio": float(
            run_summary["windowed_results"][
                "foreground_to_target_integrated_absolute_ratio"
            ]
        ),
        "selected_target_windowed_power": selected_target.tolist(),
        "selected_recovered_windowed_power": selected_estimate.tolist(),
        "figures": {
            key: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for key, path in figure_paths.items()
        },
        "cell_table": {
            "path": str(args.cell_table.relative_to(ROOT)),
            "sha256": _sha256(args.cell_table),
            "row_count": int(np.prod(shape)),
            "evaluated_row_count": int(output_band_ids.size),
        },
        "interpretation": (
            "The selected outputs are overlapping response-windowed bandpowers. "
            "The direct 20-cell footprint is a coverage diagnostic, not a claim "
            "that 20 independent delta-function sky bins were deconvolved. "
            "The 112-cell closure audit is post hoc and separates numerical "
            "closure from the original reporting-region localization gate; it "
            "does not replace the frozen 20-window main result."
        ),
    }
    args.coverage_summary.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_summary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
