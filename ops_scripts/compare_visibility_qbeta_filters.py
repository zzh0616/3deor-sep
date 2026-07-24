#!/usr/bin/env python3
"""Compare foreground filters under one exact-visibility Q_beta contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fg-rmw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from chips_visibility import (  # noqa: E402
    chebyshev_foreground_basis,
    dpss_foreground_basis,
)


COLORS = (
    "#17324d",
    "#147d7e",
    "#d24b2a",
    "#e3a018",
    "#6c8e55",
    "#8a5a44",
    "#4677a8",
    "#b66b88",
)


def _display_label(label: str) -> str:
    replacements = {
        "dpss_hard_e12_hann": "DPSS hard\ncutoff 1e-12",
        "dpss_hard_e10_hann": "DPSS hard\ncutoff 1e-10",
        "dpss_hard_e8_hann": "DPSS hard\ncutoff 1e-8",
        "dpss_hard_e6_hann": "DPSS hard\ncutoff 1e-6",
        "dpss_soft_e12_r1e4_hann": "DPSS soft\nrho 1e4",
        "dpss_soft_e12_r1e8_hann": "DPSS soft\nrho 1e8",
        "dpss_hard_e12_blackman_harris": "DPSS hard\nBlackman-Harris",
        "none_hann": "Hann only",
        "chebyshev_d3_hann": "Chebyshev\ndegree 3",
        "chebyshev_rankmatched_e12_hann": "Chebyshev\nrank matched",
    }
    return replacements.get(label, label)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ps2d_v2_32high_isobeam_patch.json",
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="LABEL=directory containing result.npz/json or a combined subdirectory",
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    return parser.parse_args()


def _load_result(specification: str) -> tuple[str, Path, dict[str, Any], dict[str, np.ndarray]]:
    if "=" not in specification:
        raise ValueError("--result must use LABEL=PATH")
    label, raw_path = specification.split("=", 1)
    directory = Path(raw_path)
    if not (directory / "result.npz").is_file():
        directory = directory / "combined"
    metadata = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    with np.load(directory / "result.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return label, directory, metadata, arrays


def _relative_l2(
    estimate: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = np.asarray(estimate) - np.asarray(truth)
    denominator = max(
        float(np.sum(weights * np.square(truth))),
        np.finfo(np.float64).tiny,
    )
    return math.sqrt(float(np.sum(weights * np.square(residual))) / denominator)


def _method_products(
    *,
    arrays: dict[str, np.ndarray],
    geometric: np.ndarray,
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
    source_kperp = np.asarray(
        arrays["source_band_kperp_indices"], dtype=np.int64
    )
    source_kpar = np.asarray(
        arrays["source_band_kpar_indices"], dtype=np.int64
    )
    source_in_geometric = np.zeros(source_kperp.shape, dtype=bool)
    in_radial_range = source_kpar < geometric.shape[1]
    source_in_geometric[in_radial_range] = geometric[
        source_kperp[in_radial_range], source_kpar[in_radial_range]
    ]
    geometric_fraction = np.sum(window[:, source_in_geometric], axis=1)
    square_sum = np.sum(np.square(window), axis=1)
    effective_width = np.divide(
        1.0,
        square_sum,
        out=np.full(row_sum.shape, np.inf),
        where=square_sum > 0.0,
    )
    relative_response = row_sum / max(
        float(np.max(row_sum)), np.finfo(np.float64).tiny
    )
    return {
        "row_sum": row_sum,
        "window": window,
        "target": target,
        "pure": pure,
        "total": total,
        "foreground": foreground,
        "geometric_fraction": geometric_fraction,
        "effective_width": effective_width,
        "relative_response": relative_response,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.edgecolor": "#17324d",
            "axes.labelcolor": "#17324d",
            "axes.titlecolor": "#17324d",
            "xtick.color": "#17324d",
            "ytick.color": "#17324d",
            "text.color": "#17324d",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fbfaf6",
            "savefig.facecolor": "#fbfaf6",
            "savefig.bbox": "tight",
        }
    )


def _rank_matched_subspace_diagnostic(
    *,
    config: dict[str, Any],
    resolved: Any,
    kperp_edges: np.ndarray,
    selected_kperp_indices: np.ndarray,
) -> dict[str, Any]:
    frequencies_hz = np.asarray(config["frequencies_mhz"], dtype=np.float64) * 1e6
    frequency_spacing_hz = float(np.mean(np.diff(frequencies_hz)))
    radial_mpc_per_hz = (
        float(resolved.geometry["radial_spacing_mpc"]) / frequency_spacing_hz
    )
    buffer_delay = (
        float(resolved.geometry["wedge_buffer_mpc_inv"])
        * radial_mpc_per_hz
        / (2.0 * math.pi)
    )
    maximum_delays = (
        kperp_edges[1:]
        * float(resolved.geometry["transverse_distance_mpc"])
        / (2.0 * math.pi)
        * math.sin(
            math.radians(float(resolved.geometry["source_corner_angle_deg"]))
        )
        / (float(config["reference_frequency_mhz"]) * 1e6)
        + buffer_delay
    )
    rows: list[dict[str, Any]] = []
    for transverse_index in selected_kperp_indices:
        dpss_basis = dpss_foreground_basis(
            frequencies_hz,
            float(maximum_delays[transverse_index]),
            eigenvalue_threshold=1e-12,
        )
        polynomial_basis = chebyshev_foreground_basis(
            frequencies_hz, dpss_basis.rank - 1
        )
        canonical_correlations = np.linalg.svd(
            dpss_basis.vectors.T @ polynomial_basis,
            compute_uv=False,
        )
        principal_angles_deg = np.degrees(
            np.arccos(np.clip(canonical_correlations, -1.0, 1.0))
        )
        rows.append(
            {
                "kperp_index": int(transverse_index),
                "dpss_rank": int(dpss_basis.rank),
                "maximum_delay_us": float(
                    1e6 * maximum_delays[transverse_index]
                ),
                "minimum_canonical_correlation": float(
                    np.min(canonical_correlations)
                ),
                "maximum_principal_angle_deg": float(
                    np.max(principal_angles_deg)
                ),
            }
        )
    return {
        "interpretation": (
            "The rank-matched Chebyshev control is high order and nearly spans "
            "the same discrete smooth subspace as the DPSS basis."
        ),
        "minimum_canonical_correlation": float(
            min(row["minimum_canonical_correlation"] for row in rows)
        ),
        "maximum_principal_angle_deg": float(
            max(row["maximum_principal_angle_deg"] for row in rows)
        ),
        "per_kperp": rows,
    }


def _plot_summary(
    *,
    path: Path,
    labels: list[str],
    rows: list[dict[str, Any]],
    total_errors: np.ndarray,
) -> None:
    _style()
    x = np.arange(len(labels))
    display_labels = [_display_label(label) for label in labels]
    colors = [COLORS[index % len(COLORS)] for index in range(len(labels))]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=((15.5, 7.4) if len(labels) > 6 else (12.0, 7.2)),
        constrained_layout=True,
    )

    foreground = np.asarray(
        [row["common20_foreground_auto_over_target"] for row in rows]
    )
    axes[0, 0].bar(x, foreground, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axhline(1.0, color="#17324d", linestyle=(0, (4, 3)))
    axes[0, 0].set_ylabel("FG auto / response-windowed EoR")
    axes[0, 0].set_title("Foreground suppression on baseline 20 cells")

    pure_l2 = np.asarray([row["common20_pure_relative_l2"] for row in rows])
    total_l2 = np.asarray([row["common20_total_relative_l2"] for row in rows])
    heldout_broad_l2 = [
        row.get("heldout_broad_total_worst_relative_l2") for row in rows
    ]
    width = 0.38
    if all(value is not None for value in heldout_broad_l2):
        broad_l2 = np.asarray(heldout_broad_l2, dtype=np.float64)
        axes[0, 1].bar(
            x - width / 2,
            total_l2,
            width,
            label="Physical FG + EoR, common 20",
            color="#147d7e",
        )
        axes[0, 1].bar(
            x + width / 2,
            broad_l2,
            width,
            label="Worst of 16 phases, broad support",
            color="#d24b2a",
        )
        axes[0, 1].set_title("Common support versus expanded-support stress test")
        plotted_maximum = float(np.max(broad_l2))
    else:
        axes[0, 1].bar(
            x - width / 2,
            pure_l2,
            width,
            label="Pure EoR",
            color="#147d7e",
        )
        axes[0, 1].bar(
            x + width / 2,
            total_l2,
            width,
            label="FG + EoR",
            color="#d24b2a",
        )
        axes[0, 1].set_title("Closure with and without foreground")
        plotted_maximum = float(np.max(total_l2))
    axes[0, 1].axhline(0.2, color="#17324d", linestyle=(0, (4, 3)))
    if plotted_maximum > 1.0:
        axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Response-windowed relative L2")
    axes[0, 1].legend(frameon=False)

    native = np.asarray([row["native_selected_window_count"] for row in rows])
    broad = np.asarray([row["broad_geometric_candidate_count"] for row in rows])
    broad_pass = np.asarray(
        [row["broad_candidates_total_passing_20pct_count"] for row in rows]
    )
    axes[1, 0].bar(x, broad, color="#dce9e6", label="Broad-window candidates")
    axes[1, 0].bar(x, broad_pass, color="#147d7e", label="Total passing 20%")
    axes[1, 0].scatter(
        x, native, color="#d24b2a", marker="D", s=28, label="Original-region selection"
    )
    axes[1, 0].set_ylabel("Output-cell count")
    axes[1, 0].set_title("Response-only support diagnostics")
    axes[1, 0].legend(frameon=False, fontsize=8)

    image = axes[1, 1].imshow(
        np.clip(100.0 * total_errors, -100.0, 100.0),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-100.0,
        vmax=100.0,
    )
    axes[1, 1].set_title("FG + EoR signed error on baseline 20 cells")
    axes[1, 1].set_ylabel("Filter")
    axes[1, 1].set_xlabel("Baseline selected-cell index")
    axes[1, 1].set_yticks(x)
    axes[1, 1].set_yticklabels(display_labels)
    fig.colorbar(image, ax=axes[1, 1], label="Signed error [%]", shrink=0.9)

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x)
        axis.set_xticklabels(
            display_labels,
            rotation=0,
            ha="center",
            fontsize=7.2 if len(labels) > 6 else 8.0,
        )
        axis.grid(axis="y", color="#d7d2c8", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Noiseless exact-visibility foreground-filter ablation",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = _arguments()
    loaded = [_load_result(specification) for specification in args.result]
    labels = [item[0] for item in loaded]
    if args.baseline_label not in labels:
        raise ValueError("baseline label is not present in --result")
    contracts = {item[2]["analysis_contract_sha256"] for item in loaded}
    banks = {item[2]["visibility_bank_sha256"] for item in loaded}
    skies = {item[2]["sky_cache_sha256"] for item in loaded}
    if len(contracts) != 1 or len(banks) != 1 or len(skies) != 1:
        raise ValueError("Compared results do not share one exact operator contract")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    kperp_edges = np.asarray(
        resolved.contract.window_layout.kperp_edges, dtype=np.float64
    )
    kpar_values = np.asarray(
        resolved.contract.window_layout.kpar_values, dtype=np.float64
    )
    geometric = resolved.window_spec.mask(
        kperp_edges[1:, None], kpar_values[None, :]
    )

    baseline_arrays = loaded[labels.index(args.baseline_label)][3]
    baseline_positions = np.asarray(
        baseline_arrays["qbeta_selected_window_positions"], dtype=np.int64
    )
    baseline_ids = np.asarray(
        baseline_arrays["output_band_ids"], dtype=np.int64
    )[baseline_positions]
    selected_kperp_indices = np.unique(baseline_ids // geometric.shape[1])
    baseline_products = _method_products(
        arrays=baseline_arrays, geometric=geometric
    )
    baseline_id_to_position = {
        int(band_id): int(position)
        for position, band_id in enumerate(baseline_arrays["output_band_ids"])
    }
    baseline_weights = np.asarray(
        [
            baseline_products["relative_response"][baseline_id_to_position[int(band_id)]]
            for band_id in baseline_ids
        ]
    )

    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    total_error_rows: list[np.ndarray] = []
    for label, directory, metadata, arrays in loaded:
        products = _method_products(arrays=arrays, geometric=geometric)
        output_ids = np.asarray(arrays["output_band_ids"], dtype=np.int64)
        positions = {int(band_id): index for index, band_id in enumerate(output_ids)}
        common_positions = np.asarray(
            [positions.get(int(band_id), -1) for band_id in baseline_ids],
            dtype=np.int64,
        )
        available = common_positions >= 0
        if not np.all(available):
            raise ValueError(
                f"{label} lacks {np.count_nonzero(~available)} baseline-selected cells"
            )
        target = products["target"][common_positions]
        pure = products["pure"][common_positions]
        total = products["total"][common_positions]
        foreground = products["foreground"][common_positions]
        pure_error = (pure - target) / np.maximum(np.abs(target), 1e-300)
        total_error = (total - target) / np.maximum(np.abs(target), 1e-300)
        total_error_rows.append(total_error)
        broad = (
            (products["relative_response"] >= 0.1)
            & (products["geometric_fraction"] >= 0.95)
        )
        broad_total_error = np.abs(
            (products["total"] - products["target"])
            / np.maximum(np.abs(products["target"]), 1e-300)
        )
        broad_pure_error = np.abs(
            (products["pure"] - products["target"])
            / np.maximum(np.abs(products["target"]), 1e-300)
        )
        native_positions = np.asarray(
            arrays["qbeta_selected_window_positions"], dtype=np.int64
        )
        foreground_ranks = np.asarray(
            arrays.get("foreground_ranks", np.full(1, np.nan)), dtype=np.float64
        )
        finite_ranks = foreground_ranks[np.isfinite(foreground_ranks)]
        if "heldout_total_mixture_q" in arrays:
            all_total_mixtures = np.divide(
                arrays["heldout_total_mixture_q"],
                products["row_sum"][None, :],
                out=np.full_like(
                    arrays["heldout_total_mixture_q"], np.nan, dtype=np.float64
                ),
                where=products["row_sum"][None, :] > 0.0,
            )
            total_mixtures = all_total_mixtures[:, common_positions]
            heldout_total_l2 = [
                _relative_l2(realization, target, baseline_weights)
                for realization in total_mixtures
            ]
            heldout_total_maximum_error = [
                float(
                    np.max(
                        np.abs(realization - target)
                        / np.maximum(np.abs(target), 1e-300)
                    )
                )
                for realization in total_mixtures
            ]
            eor_mixtures = np.divide(
                arrays["heldout_mixture_q"],
                products["row_sum"][None, :],
                out=np.full_like(
                    arrays["heldout_mixture_q"], np.nan, dtype=np.float64
                ),
                where=products["row_sum"][None, :] > 0.0,
            )[:, common_positions]
            heldout_foreground_effect_l2 = [
                _relative_l2(
                    total_realization,
                    eor_realization,
                    baseline_weights,
                )
                for total_realization, eor_realization in zip(
                    total_mixtures, eor_mixtures, strict=True
                )
            ]
            heldout_foreground_effect_maximum = [
                float(
                    np.max(
                        np.abs(total_realization - eor_realization)
                        / np.maximum(np.abs(target), 1e-300)
                    )
                )
                for total_realization, eor_realization in zip(
                    total_mixtures, eor_mixtures, strict=True
                )
            ]
            broad_weights = products["relative_response"][broad]
            broad_target = products["target"][broad]
            heldout_broad_total_l2 = [
                _relative_l2(
                    realization[broad],
                    broad_target,
                    broad_weights,
                )
                for realization in all_total_mixtures
            ]
            heldout_broad_total_maximum = [
                float(
                    np.max(
                        np.abs(realization[broad] - broad_target)
                        / np.maximum(np.abs(broad_target), 1e-300)
                    )
                )
                for realization in all_total_mixtures
            ]
            heldout_broad_total_passing = [
                int(
                    np.count_nonzero(
                        np.abs(realization[broad] - broad_target)
                        / np.maximum(np.abs(broad_target), 1e-300)
                        < 0.2
                    )
                )
                for realization in all_total_mixtures
            ]
        else:
            heldout_total_l2 = []
            heldout_total_maximum_error = []
            heldout_foreground_effect_l2 = []
            heldout_foreground_effect_maximum = []
            heldout_broad_total_l2 = []
            heldout_broad_total_maximum = []
            heldout_broad_total_passing = []
        qbeta_meta = metadata["qbeta"]
        filter_name = qbeta_meta.get("foreground_filter", "dpss_hard")
        effective_suppression = qbeta_meta.get("effective_suppression")
        if effective_suppression is None:
            if filter_name == "none":
                effective_suppression = "none"
            elif filter_name == "dpss_soft":
                effective_suppression = (
                    f"finite:{float(qbeta_meta['suppression_strength']):.12g}"
                )
            else:
                effective_suppression = "hard"
        row = {
            "label": label,
            "directory": (
                str(
                    Path(metadata["input_directories"][0]).parents[1]
                    / "combined"
                )
                if metadata.get("input_directories")
                else str(directory)
            ),
            "foreground_filter": filter_name,
            "effective_suppression": effective_suppression,
            "requested_suppression_strength": qbeta_meta.get(
                "suppression_strength"
            ),
            "dpss_eigenvalue_threshold": qbeta_meta.get(
                "dpss_eigenvalue_threshold", 1e-12
            ),
            "spectral_taper": qbeta_meta.get("spectral_taper", "hann"),
            "foreground_rank_min": (
                float(np.min(finite_ranks)) if finite_ranks.size else None
            ),
            "foreground_rank_max": (
                float(np.max(finite_ranks)) if finite_ranks.size else None
            ),
            "supported_output_band_count": int(output_ids.size),
            "native_selected_window_count": int(native_positions.size),
            "broad_geometric_candidate_count": int(np.count_nonzero(broad)),
            "broad_candidates_pure_passing_20pct_count": int(
                np.count_nonzero(broad & (broad_pure_error < 0.2))
            ),
            "broad_candidates_total_passing_20pct_count": int(
                np.count_nonzero(broad & (broad_total_error < 0.2))
            ),
            "heldout_broad_total_worst_relative_l2": (
                float(np.max(heldout_broad_total_l2))
                if heldout_broad_total_l2
                else None
            ),
            "heldout_broad_total_worst_maximum_error": (
                float(np.max(heldout_broad_total_maximum))
                if heldout_broad_total_maximum
                else None
            ),
            "heldout_broad_total_minimum_passing_20pct_count": (
                int(np.min(heldout_broad_total_passing))
                if heldout_broad_total_passing
                else None
            ),
            "common20_pure_relative_l2": _relative_l2(
                pure, target, baseline_weights
            ),
            "common20_total_relative_l2": _relative_l2(
                total, target, baseline_weights
            ),
            "common20_pure_integrated_ratio": float(
                np.sum(baseline_weights * pure)
                / np.sum(baseline_weights * target)
            ),
            "common20_total_integrated_ratio": float(
                np.sum(baseline_weights * total)
                / np.sum(baseline_weights * target)
            ),
            "common20_pure_maximum_absolute_error": float(
                np.max(np.abs(pure_error))
            ),
            "common20_total_maximum_absolute_error": float(
                np.max(np.abs(total_error))
            ),
            "common20_foreground_auto_over_target": float(
                np.sum(baseline_weights * np.abs(foreground))
                / np.sum(baseline_weights * np.abs(target))
            ),
            "common20_total_minus_pure_absolute_over_target": float(
                np.sum(baseline_weights * np.abs(total - pure))
                / np.sum(baseline_weights * np.abs(target))
            ),
            "common20_median_effective_width": float(
                np.median(products["effective_width"][common_positions])
            ),
            "heldout_fg_plus_eor_common20_worst_relative_l2": (
                float(np.max(heldout_total_l2)) if heldout_total_l2 else None
            ),
            "heldout_fg_plus_eor_common20_worst_maximum_error": (
                float(np.max(heldout_total_maximum_error))
                if heldout_total_maximum_error
                else None
            ),
            "heldout_foreground_effect_common20_worst_relative_l2": (
                float(np.max(heldout_foreground_effect_l2))
                if heldout_foreground_effect_l2
                else None
            ),
            "heldout_foreground_effect_common20_worst_maximum_error": (
                float(np.max(heldout_foreground_effect_maximum))
                if heldout_foreground_effect_maximum
                else None
            ),
            "calibration_validation_response_relative_l2": float(
                qbeta_meta["calibration_validation_response_relative_l2"]
            ),
        }
        summary_rows.append(row)
        for common_index, (band_id, position) in enumerate(
            zip(baseline_ids, common_positions, strict=True)
        ):
            kp_index, kpar_index = divmod(int(band_id), geometric.shape[1])
            cell_rows.append(
                {
                    "label": label,
                    "baseline_cell_index": common_index,
                    "output_band_id": int(band_id),
                    "kperp_index": kp_index,
                    "kpar_index": kpar_index,
                    "kperp_center_mpc_inv": float(
                        0.5 * (kperp_edges[kp_index] + kperp_edges[kp_index + 1])
                    ),
                    "kpar_mpc_inv": float(kpar_values[kpar_index]),
                    "target_windowed_power": float(products["target"][position]),
                    "pure_eor_signed_error_percent": float(
                        100.0 * pure_error[common_index]
                    ),
                    "total_signed_error_percent": float(
                        100.0 * total_error[common_index]
                    ),
                    "foreground_auto_over_target": float(
                        products["foreground"][position]
                        / products["target"][position]
                    ),
                    "geometric_window_response_fraction": float(
                        products["geometric_fraction"][position]
                    ),
                    "window_effective_width": float(
                        products["effective_width"][position]
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "visibility_qbeta_filter_ablation",
        "schema_version": 1,
        "analysis_contract_sha256": next(iter(contracts)),
        "visibility_bank_sha256": next(iter(banks)),
        "sky_cache_sha256": next(iter(skies)),
        "baseline_label": args.baseline_label,
        "common_baseline_selected_cell_count": int(baseline_ids.size),
        "rank_matched_subspace_diagnostic": (
            _rank_matched_subspace_diagnostic(
                config=config,
                resolved=resolved,
                kperp_edges=kperp_edges,
                selected_kperp_indices=selected_kperp_indices,
            )
        ),
        "methods": summary_rows,
        "notes": [
            "all comparisons are noiseless and use the same exact visibility operator",
            "common20 metrics use the baseline response-selected cells and baseline response weights",
            "each method is compared to its own response-windowed W p target",
            "broad geometric support is a diagnostic, not an independently frozen promotion",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "cell_errors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(cell_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cell_rows)
    figure_path = (
        args.figure
        if args.figure is not None
        else args.output_dir / "filter_ablation.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_summary(
        path=figure_path,
        labels=labels,
        rows=summary_rows,
        total_errors=np.stack(total_error_rows),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
