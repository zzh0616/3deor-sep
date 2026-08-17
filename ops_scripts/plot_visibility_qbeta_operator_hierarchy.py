#!/usr/bin/env python3
"""Plot the common-beam and delay-diagonal operator controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fg-rmw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#17324d"
TEAL = "#147d7e"
GOLD = "#e3a018"
VERMILION = "#d24b2a"
GRID = "#d7d2c8"
PAPER = "#fbfaf6"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-products", type=Path, required=True)
    parser.add_argument("--common-products", type=Path, required=True)
    parser.add_argument("--delay-products", type=Path, required=True)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    exact = _load(args.exact_products)
    common = _load(args.common_products)
    delay = _load(args.delay_products)
    summary = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
    prefix = f"{args.profile}_"

    exact_selected = np.asarray(exact[prefix + "selected"], dtype=bool)
    common_selected = np.asarray(common[prefix + "selected"], dtype=bool)
    delay_selected = np.asarray(delay[prefix + "selected"], dtype=bool)
    exact_common = exact_selected & common_selected
    exact_delay = exact_selected & delay_selected
    group_indices = np.arange(exact_selected.size)
    exact_target = np.asarray(exact[prefix + "target"], dtype=np.float64)

    _style()
    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.1), constrained_layout=True
    )
    ax = axes[0]
    ax.axhline(1.0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.scatter(
        group_indices[exact_common],
        exact[prefix + "bank_total_estimate"][exact_common]
        / exact_target[exact_common],
        s=22,
        color=TEAL,
        alpha=0.82,
        label="Exact station-pair response",
        zorder=3,
    )
    ax.scatter(
        group_indices[exact_common],
        common[prefix + "bank_total_estimate"][exact_common]
        / exact_target[exact_common],
        s=24,
        marker="x",
        linewidth=1.2,
        color=VERMILION,
        alpha=0.88,
        label="Common station-power response",
        zorder=4,
    )
    profile_summary = summary["profiles"][args.profile]
    common_pair = profile_summary["pairwise_with_exact"][
        "common_scalar_power"
    ]
    exact_metrics = common_pair["exact_fixed_exact_target"]
    common_metrics = common_pair["candidate_fixed_exact_target"]
    note = (
        f"{int(np.count_nonzero(exact_common))} common groups\n"
        f"exact: {exact_metrics['integrated_power_ratio']:.3f} / "
        f"{exact_metrics['relative_l2']:.3f}\n"
        f"common: {common_metrics['integrated_power_ratio']:.3f} / "
        f"{common_metrics['relative_l2']:.3f}\n"
        "(integrated ratio / relative L2)"
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": PAPER,
            "edgecolor": GRID,
            "alpha": 0.95,
        },
    )
    ax.set(
        xlabel="Coarse 4x1 group index",
        ylabel="Recovered / exact-window target",
        ylim=(0.45, 1.62),
        title="(a) Wrong common beam biases normalization",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=8.1)

    ax = axes[1]
    ax.axhline(1.0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.scatter(
        group_indices[exact_common],
        common[prefix + "target"][exact_common] / exact_target[exact_common],
        s=22,
        color=GOLD,
        alpha=0.82,
        label="Common-beam broad target",
        zorder=3,
    )
    ax.scatter(
        group_indices[exact_delay],
        delay[prefix + "target"][exact_delay] / exact_target[exact_delay],
        s=24,
        marker="x",
        linewidth=1.2,
        color=VERMILION,
        alpha=0.88,
        label="Delay-diagonal single-cell target",
        zorder=4,
    )
    delay_native = profile_summary["arms"]["delay_diagonal"][
        "native_response_target"
    ]
    target_match = common_pair["candidate_response_target_vs_exact"]
    note = (
        "common target vs exact:\n"
        f"L2 = {target_match['relative_l2']:.4f}\n"
        "delay interpretation:\n"
        f"ratio / L2 = {delay_native['integrated_power_ratio']:.3f} / "
        f"{delay_native['relative_l2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": PAPER,
            "edgecolor": GRID,
            "alpha": 0.95,
        },
    )
    ax.set_yscale("log")
    ax.set(
        xlabel="Coarse 4x1 group index",
        ylabel="Assumed target / exact-window target",
        ylim=(0.45, 25.0),
        title="(b) Diagonal-delay interpretation changes the target",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7, which="both")
    ax.legend(loc="lower right", frameon=False, fontsize=8.1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
