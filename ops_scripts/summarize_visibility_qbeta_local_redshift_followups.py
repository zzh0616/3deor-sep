#!/usr/bin/env python3
"""Summarize and plot local-redshift Q_beta validation follow-ups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-figure", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bank_metrics(path: Path) -> dict[str, float]:
    profile = _load_json(path)["profiles"]["quad_kperp_response"]
    return {
        "selected_group_count": int(profile["selected_group_count"]),
        **{
            name: float(value)
            for name, value in profile["bank_total"].items()
            if isinstance(value, (int, float))
        },
        "heldout_total_worst_relative_l2": float(
            profile["heldout_total_worst_relative_l2"]
        ),
        "foreground_effect_maximum_fraction": float(
            profile["foreground_effect_maximum_fraction"]
        ),
        "response_window_participation_rank": float(
            profile["response_window_rank"]["participation_rank"]
        ),
    }


def _frequency_centers(
    independent_manifest: dict[str, Any],
) -> dict[str, float]:
    frequencies = np.asarray(
        independent_manifest["contract"]["frequencies_mhz"],
        dtype=np.float64,
    )
    centers: dict[str, float] = {}
    for record in independent_manifest["window_products"]:
        label = str(record["label"])
        fields = label.split("_")
        low = float(fields[-2].replace("p", "."))
        high = float(fields[-1].removesuffix("mhz").replace("p", "."))
        centers[label] = 0.5 * (low + high)
    if not centers or frequencies.size < 1:
        raise ValueError("Independent-lightcone manifest is incomplete")
    return centers


def _outer_diagnostics(
    *,
    products_path: Path,
    group_metadata_path: Path,
) -> dict[str, Any] | None:
    if not products_path.exists():
        return None
    with np.load(products_path, allow_pickle=False) as archive:
        products = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    positions = np.asarray(
        products["selected_window_positions"], dtype=np.int64
    )
    target = np.asarray(
        products["target_windowed_power"], dtype=np.float64
    )
    delta = np.asarray(
        products["delta_total_windowed_power"], dtype=np.float64
    )
    fractions = np.abs(delta[positions]) / np.maximum(
        np.abs(target[positions]), 1e-300
    )
    diagnostics: dict[str, Any] = {
        "median_absolute_window_ratio": float(np.median(fractions)),
        "p90_absolute_window_ratio": float(np.quantile(fractions, 0.9)),
        "above_10pct_count": int(np.count_nonzero(fractions > 0.1)),
        "above_20pct_count": int(np.count_nonzero(fractions > 0.2)),
        "maximum_absolute_window_ratio": float(np.max(fractions)),
    }
    if not group_metadata_path.exists():
        return diagnostics
    with np.load(group_metadata_path, allow_pickle=False) as archive:
        metadata = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    first = np.asarray(metadata["group_kperp_first"], dtype=np.int64)
    stop = np.asarray(metadata["group_kperp_stop"], dtype=np.int64)
    by_kperp = []
    for lower in np.unique(first[positions]):
        group_positions = positions[first[positions] == lower]
        group_fractions = np.abs(delta[group_positions]) / np.maximum(
            np.abs(target[group_positions]), 1e-300
        )
        by_kperp.append(
            {
                "kperp_index_first": int(lower),
                "kperp_index_stop": int(stop[group_positions[0]]),
                "selected_count": int(group_positions.size),
                "median_absolute_window_ratio": float(
                    np.median(group_fractions)
                ),
                "maximum_absolute_window_ratio": float(
                    np.max(group_fractions)
                ),
            }
        )
    diagnostics["by_kperp_group"] = by_kperp

    conservative = positions[first[positions] >= 8]
    weights = np.asarray(
        products["group_metric_weights"], dtype=np.float64
    )[conservative]
    conservative_target = target[conservative]
    denominator = max(
        float(np.sum(weights * np.abs(conservative_target))), 1e-300
    )
    conservative_delta = delta[conservative]
    base = np.asarray(
        products["base_total_windowed_power"], dtype=np.float64
    )[conservative]
    extended = np.asarray(
        products["extended_total_windowed_power"], dtype=np.float64
    )[conservative]
    diagnostics["posthoc_kperp_index_minimum_8"] = {
        "posthoc_diagnostic_only": True,
        "selected_count": int(conservative.size),
        "integrated_absolute_ratio": float(
            np.sum(weights * np.abs(conservative_delta)) / denominator
        ),
        "integrated_signed_ratio": float(
            np.sum(weights * conservative_delta) / denominator
        ),
        "maximum_absolute_window_ratio": float(
            np.max(
                np.abs(conservative_delta)
                / np.maximum(np.abs(conservative_target), 1e-300)
            )
        ),
        "base_integrated_power_ratio": float(
            np.sum(weights * base) / denominator
        ),
        "extended_integrated_power_ratio": float(
            np.sum(weights * extended) / denominator
        ),
    }
    return diagnostics


def _build_summary(results_dir: Path) -> dict[str, Any]:
    independent_manifest = _load_json(
        results_dir / "covariance" / "shared_inputs_manifest.json"
    )
    centers = _frequency_centers(independent_manifest)
    local_paths = sorted((results_dir / "local_windows").glob("*.json"))
    local_windows = []
    for path in local_paths:
        metrics = _bank_metrics(path)
        index = int(path.stem.split("_")[1])
        label = independent_manifest["window_products"][index]["label"]
        metrics["label"] = label
        metrics["frequency_center_mhz"] = centers[label]
        local_windows.append(metrics)

    independent = []
    for record in independent_manifest["independent_lightcone_by_window"]:
        independent.append(
            {
                "label": str(record["label"]),
                "frequency_center_mhz": centers[str(record["label"])],
                "fg_plus_eor": record["fg_plus_eor"],
                "pure_eor": record["pure_eor"],
            }
        )

    probe_raw = _load_json(
        results_dir / "probes" / "all_subsets_summary.json"
    )["all_subset_distributions"]
    probes = [
        {
            "probe_count": int(record["probe_count"]),
            "subset_count": int(record["enumerated_subset_count"]),
            "integrated_ratio_mean": float(
                record[
                    "bank_total_final_selection_integrated_power_ratio"
                ]["mean"]
            ),
            "integrated_ratio_standard_deviation": float(
                record[
                    "bank_total_final_selection_integrated_power_ratio"
                ]["standard_deviation"]
            ),
            "response_row_sum_relative_l2_mean": float(
                record["response_row_sum_relative_l2_to_full"]["mean"]
            ),
            "selection_jaccard_mean": float(
                record["selection_jaccard_with_full"]["mean"]
            ),
        }
        for record in probe_raw
    ]
    four_probe = _bank_metrics(
        results_dir
        / "probes"
        / "four_partition_four_probe_summary.json"
    )

    covariance = _load_json(results_dir / "covariance" / "summary.json")
    outer_results = []
    for path in sorted((results_dir / "outer_field").glob("support_*_result.json")):
        record = _load_json(path)
        name = path.stem.removesuffix("_result")
        output = {
            "name": name,
            "selected_window_count": int(record["selected_window_count"]),
            "base_total": record["base_total"],
            "extended_total": record["extended_total"],
            "outer_induced_total_change": record[
                "outer_induced_total_change"
            ],
            "base_q_recomputation_relative_l2": float(
                record["base_q_recomputation_relative_l2"]
            ),
            "base_coarse_recomputation_relative_l2": float(
                record["base_coarse_recomputation_relative_l2"]
            ),
        }
        diagnostics = _outer_diagnostics(
            products_path=path.with_name(f"{name}_products.npz"),
            group_metadata_path=path.with_name(
                f"{name}_group_metadata.npz"
            ),
        )
        if diagnostics is not None:
            output["selected_window_diagnostics"] = diagnostics
        outer_results.append(output)
    return {
        "schema": "visibility_qbeta_local_redshift_followup_summary",
        "schema_version": 1,
        "local_windows": local_windows,
        "independent_lightcone": independent,
        "probe_convergence": probes,
        "four_partition_four_probe": four_probe,
        "covariance": covariance,
        "outer_field": outer_results,
    }


def _plot(
    summary: dict[str, Any],
    *,
    covariance_products: Path,
    out: Path,
) -> None:
    local = summary["local_windows"]
    independent = summary["independent_lightcone"]
    frequencies = np.asarray(
        [record["frequency_center_mhz"] for record in local]
    )
    ratios = np.asarray(
        [record["integrated_power_ratio"] for record in local]
    )
    independent_ratios = np.asarray(
        [
            record["fg_plus_eor"]["integrated_power_ratio"]
            for record in independent
        ]
    )
    local_l2 = np.asarray([record["relative_l2"] for record in local])
    independent_l2 = np.asarray(
        [record["fg_plus_eor"]["relative_l2"] for record in independent]
    )

    probes = summary["probe_convergence"]
    probe_counts = np.asarray([record["probe_count"] for record in probes])
    response_l2 = np.asarray(
        [record["response_row_sum_relative_l2_mean"] for record in probes]
    )
    ratio_std = np.asarray(
        [
            record["integrated_ratio_standard_deviation"]
            for record in probes
        ]
    )

    covariance = np.load(covariance_products)
    correlation = np.asarray(
        covariance["heldout_error_correlation"], dtype=np.float64
    )
    offsets = np.asarray(covariance["window_offsets"], dtype=np.int64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    figure = plt.figure(figsize=(12.0, 7.1), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.12))

    ratio_axis = figure.add_subplot(grid[0, 0])
    ratio_axis.axhline(1.0, color="0.45", linewidth=1.0, linestyle="--")
    ratio_axis.plot(frequencies, ratios, "o-", label="cube2")
    ratio_axis.plot(
        frequencies,
        independent_ratios,
        "s-",
        label="independent cube1",
    )
    ratio_axis.set(
        xlabel="Local-window centre [MHz]",
        ylabel=r"$\sum\widehat P/\sum WP$",
        title="(a) Local-redshift recovery",
    )
    ratio_axis.legend(frameon=False)

    l2_axis = figure.add_subplot(grid[0, 1])
    l2_axis.plot(frequencies, 100.0 * local_l2, "o-", label="cube2")
    l2_axis.plot(
        frequencies,
        100.0 * independent_l2,
        "s-",
        label="independent cube1",
    )
    l2_axis.set(
        xlabel="Local-window centre [MHz]",
        ylabel="Response-weighted relative L2 [%]",
        title="(b) Realization dependence",
    )
    l2_axis.legend(frameon=False)

    probe_axis = figure.add_subplot(grid[0, 2])
    probe_axis.plot(
        probe_counts,
        100.0 * response_l2,
        "o-",
        color="#00796b",
        label="response row-sum L2",
    )
    probe_axis.plot(
        probe_counts,
        100.0 * ratio_std,
        "s-",
        color="#c75b12",
        label="bank-ratio std.",
    )
    probe_axis.set_xscale("log", base=2)
    probe_axis.set_xticks(probe_counts, labels=[str(x) for x in probe_counts])
    probe_axis.set(
        xlabel="Fixed-row calibration probes",
        ylabel="Fractional variation [%]",
        title="(c) Probe convergence",
    )
    probe_axis.legend(frameon=False)

    covariance_axis = figure.add_subplot(grid[1, :2])
    image = covariance_axis.imshow(
        correlation,
        origin="lower",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
        rasterized=True,
    )
    for offset in offsets[1:-1]:
        covariance_axis.axvline(offset - 0.5, color="black", linewidth=0.5)
        covariance_axis.axhline(offset - 0.5, color="black", linewidth=0.5)
    covariance_axis.set(
        xlabel="Concatenated local-window bandpower",
        ylabel="Concatenated local-window bandpower",
        title="(d) Shared-realization error correlation (512 skies)",
    )
    figure.colorbar(image, ax=covariance_axis, label="Correlation", shrink=0.9)

    outer_axis = figure.add_subplot(grid[1, 2])
    outer = summary["outer_field"]
    if outer:
        names = [record["name"].replace("support_", "") for record in outer]
        values = [
            record["outer_induced_total_change"]["integrated_absolute_ratio"]
            for record in outer
        ]
        outer_axis.bar(names, values, color="#325d79")
        outer_axis.set_yscale("log")
        outer_axis.axhline(1.0, color="0.45", linewidth=1.0, linestyle="--")
        outer_axis.set(
            xlabel="Foreground angular support",
            ylabel=r"$\sum|\Delta P_{\rm outer}|/\sum WP_{\rm EoR}$",
            title="(e) Outer-field gate",
        )
    else:
        outer_axis.text(0.5, 0.5, "No outer-field result", ha="center")
        outer_axis.set_axis_off()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=220)
    plt.close(figure)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = _build_summary(args.results_dir)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(
        summary,
        covariance_products=(
            args.results_dir / "covariance" / "products.npz"
        ),
        out=args.out_figure,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
