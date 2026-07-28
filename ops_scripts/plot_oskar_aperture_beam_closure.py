#!/usr/bin/env python3
"""Plot station-0 and station-pair aperture-beam closure residuals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station0-result", type=Path, required=True)
    parser.add_argument("--station-pair-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    station0 = _load(args.station0_result)
    exact = _load(args.station_pair_result)
    rows = np.asarray(exact["selected_bank_rows"], dtype=np.int64)
    if not np.array_equal(rows, station0["selected_bank_rows"]):
        raise ValueError("Closure results use different visibility rows")
    target = np.asarray(exact["target_eor_visibility"]).reshape(-1)
    station0_target = np.asarray(
        station0["target_eor_visibility"]
    ).reshape(-1)
    if not np.array_equal(target, station0_target):
        raise ValueError("Closure results use different target visibilities")
    station0_prediction = np.asarray(
        station0["predicted_eor_visibility"]
    ).reshape(-1)
    exact_prediction = np.asarray(
        exact["predicted_eor_visibility"]
    ).reshape(-1)
    scale = max(
        float(np.sqrt(np.mean(np.abs(target) ** 2))),
        np.finfo(np.float64).tiny,
    )
    station0_error = np.abs(station0_prediction - target) / scale
    exact_error = np.abs(exact_prediction - target) / scale
    time_indices = np.asarray(
        station0["selected_row_time_indices"], dtype=np.int64
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.0),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    order = np.argsort(time_indices, kind="stable")
    axes[0].semilogy(
        np.arange(rows.size),
        station0_error[order],
        color="#cc4c2f",
        linewidth=1.1,
        label="station-0 power beam",
    )
    axes[0].semilogy(
        np.arange(rows.size),
        exact_error[order],
        color="#147d92",
        linewidth=1.1,
        label="station-pair Jones",
    )
    axes[0].set_xlabel("selected visibility row, ordered by time")
    axes[0].set_ylabel(r"$|V_{\rm pred}-V_{\rm OSKAR}|/\mathrm{RMS}(V)$")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", alpha=0.18)

    bins = np.logspace(-8, 0, 48)
    axes[1].hist(
        station0_error,
        bins=bins,
        histtype="stepfilled",
        color="#cc4c2f",
        alpha=0.35,
        label="station-0",
    )
    axes[1].hist(
        exact_error,
        bins=bins,
        histtype="step",
        color="#147d92",
        linewidth=2.0,
        label="station-pair",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("normalized visibility residual")
    axes[1].set_ylabel("row count")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="x", alpha=0.18)
    figure.suptitle(
        "OSKAR aperture-array primary-beam closure at 119.4 MHz",
        fontsize=12,
        y=1.01,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
