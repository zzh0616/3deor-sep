#!/usr/bin/env python3
"""Plot frozen-window EoR boost-ladder diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    return parser.parse_args()


def _series(
    rows: list[dict], metric: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["amplitude_factor"] for row in rows]),
        np.asarray([row[metric][field] for row in rows]),
    )


def main() -> None:
    args = _parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    profile = data["profiles"][args.profile]
    exact = profile["exact_station_pair"]["rows"]
    common = profile["common_scalar_power"]["rows"]
    delay = profile["delay_diagonal"]["rows"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.45))
    colors = {
        "exact": "#16697a",
        "common": "#d1495b",
        "delay": "#edae49",
        "closure": "#3f7d20",
    }

    ratio_series = (
        (exact, "fixed_exact_target", "Exact response, broad target", "exact"),
        (
            common,
            "fixed_exact_target",
            "Common beam, exact broad target",
            "common",
        ),
        (
            delay,
            "native_response_target",
            "Delay-diagonal, claimed single-cell target",
            "delay",
        ),
    )
    for rows, metric, label, color in ratio_series:
        x, y = _series(rows, metric, "integrated_power_ratio")
        axes[0].plot(
            x, y, marker="o", markersize=3.5, linewidth=1.5,
            color=colors[color], label=label,
        )
    axes[0].axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("EoR temperature amplitude factor")
    axes[0].set_ylabel("Integrated recovered / target power")
    axes[0].set_title("(a) Recovery normalization")
    axes[0].legend(frameon=False, loc="center right")
    axes[0].grid(alpha=0.2)

    for rows, metric, label, color in ratio_series:
        x, y = _series(rows, metric, "relative_l2")
        axes[1].plot(
            x, y, marker="o", markersize=3.5, linewidth=1.5,
            color=colors[color], label=label,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("EoR temperature amplitude factor")
    axes[1].set_ylabel("Relative L2 error")
    axes[1].set_title("(b) Fractional shape error")
    axes[1].grid(alpha=0.2, which="both")

    direct = data["direct_visibility_check"]
    direct_rows = [
        row for row in direct["rows"] if row["amplitude_factor"] > 0.0
    ]
    parity = direct["positive_negative_parity_rows"]
    axes[2].plot(
        [row["amplitude_factor"] for row in direct_rows],
        [row["direct_vs_saved_quadratic_relative_l2"] for row in direct_rows],
        marker="o", markersize=3.5, linewidth=1.5,
        color=colors["closure"], label="Direct visibility vs quadratic identity",
    )
    axes[2].plot(
        [row["absolute_amplitude_factor"] for row in parity],
        [row["even_eor_relative_l2"] for row in parity],
        marker="s", markersize=3.2, linewidth=1.2,
        color="#5c80bc", label="Even (+/-) EoR component",
    )
    axes[2].plot(
        [row["absolute_amplitude_factor"] for row in parity],
        [row["odd_cross_relative_l2"] for row in parity],
        marker="^", markersize=3.2, linewidth=1.2,
        color="#7a5195", label="Odd (+/-) cross component",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("EoR temperature amplitude factor")
    axes[2].set_ylabel("Closure relative L2")
    axes[2].set_title("(c) Raw-visibility injection null tests")
    axes[2].legend(frameon=False, loc="best")
    axes[2].grid(alpha=0.2, which="both")

    figure.suptitle(
        "Frozen-window EoR boost ladder (power factor = amplitude factor squared)",
        y=1.01,
        fontsize=11,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
