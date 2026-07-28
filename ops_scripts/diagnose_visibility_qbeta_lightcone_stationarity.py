#!/usr/bin/env python3
"""Audit EoR light-cone stationarity across input and analysis subbands."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sky-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--band",
        action="append",
        required=True,
        help=(
            "LABEL,INPUT_FIRST,INPUT_LAST,ANALYSIS_FIRST,ANALYSIS_LAST; "
            "repeat for compared contracts."
        ),
    )
    return parser.parse_args(argv)


def _parse_bands(values: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels: set[str] = set()
    for value in values:
        fields = str(value).split(",")
        if len(fields) != 5:
            raise ValueError("--band requires five comma-separated fields")
        label = fields[0].strip()
        bounds = [float(field) for field in fields[1:]]
        if (
            not label
            or label in labels
            or any(not math.isfinite(bound) for bound in bounds)
            or bounds[0] > bounds[1]
            or bounds[2] > bounds[3]
            or bounds[2] < bounds[0]
            or bounds[3] > bounds[1]
        ):
            raise ValueError(f"Invalid band definition: {value}")
        labels.add(label)
        output.append(
            {
                "label": label,
                "input_first_mhz": bounds[0],
                "input_last_mhz": bounds[1],
                "analysis_first_mhz": bounds[2],
                "analysis_last_mhz": bounds[3],
            }
        )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _frequency_mask(
    frequencies_mhz: np.ndarray,
    first_mhz: float,
    last_mhz: float,
) -> np.ndarray:
    tolerance = 1e-8
    mask = (frequencies_mhz >= float(first_mhz) - tolerance) & (
        frequencies_mhz <= float(last_mhz) + tolerance
    )
    if not np.any(mask):
        raise ValueError(
            f"No channels in requested interval {first_mhz}--{last_mhz} MHz"
        )
    return mask


def _band_summary(
    *,
    definition: dict[str, Any],
    frequencies_mhz: np.ndarray,
    channel_variance: np.ndarray,
) -> dict[str, Any]:
    input_mask = _frequency_mask(
        frequencies_mhz,
        definition["input_first_mhz"],
        definition["input_last_mhz"],
    )
    analysis_mask = _frequency_mask(
        frequencies_mhz,
        definition["analysis_first_mhz"],
        definition["analysis_last_mhz"],
    )
    input_positions = np.flatnonzero(input_mask)
    first_half_positions, second_half_positions = np.array_split(
        input_positions, 2
    )
    input_variance = float(np.mean(channel_variance[input_mask]))
    analysis_variance = float(np.mean(channel_variance[analysis_mask]))
    return {
        **definition,
        "input_channel_count": int(np.count_nonzero(input_mask)),
        "analysis_channel_count": int(np.count_nonzero(analysis_mask)),
        "input_mean_spatial_variance_k2": input_variance,
        "analysis_mean_spatial_variance_k2": analysis_variance,
        "analysis_to_input_variance_ratio": float(
            analysis_variance / input_variance
        ),
        "input_first_half_mean_spatial_variance_k2": float(
            np.mean(channel_variance[first_half_positions])
        ),
        "input_second_half_mean_spatial_variance_k2": float(
            np.mean(channel_variance[second_half_positions])
        ),
        "second_to_first_half_variance_ratio": float(
            np.mean(channel_variance[second_half_positions])
            / np.mean(channel_variance[first_half_positions])
        ),
    }


def _plot(
    *,
    path: Path,
    frequencies_mhz: np.ndarray,
    channel_variance: np.ndarray,
    adjacent_correlation: np.ndarray,
    bands: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
        }
    )
    colors = ("#B7462A", "#176B87", "#5C6F3B", "#9A6A1F")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0), "hspace": 0.08},
    )
    normalized_variance = channel_variance / np.mean(channel_variance)
    axes[0].plot(
        frequencies_mhz,
        normalized_variance,
        color="#1E2528",
        linewidth=1.6,
        label="Per-channel EoR spatial variance",
    )
    for index, band in enumerate(bands):
        color = colors[index % len(colors)]
        axes[0].axvspan(
            band["analysis_first_mhz"],
            band["analysis_last_mhz"],
            color=color,
            alpha=0.13,
            label=f"{band['label']} analysis",
        )
        axes[0].axvline(
            band["input_first_mhz"],
            color=color,
            linewidth=0.8,
            linestyle="--",
        )
        axes[0].axvline(
            band["input_last_mhz"],
            color=color,
            linewidth=0.8,
            linestyle="--",
        )
    axes[0].axhline(1.0, color="#6C757D", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Variance / full-cache mean")
    axes[0].set_title("EOS light-cone evolution across Q-beta bands")
    axes[0].grid(color="#C7CDD1", linewidth=0.5, alpha=0.45)
    axes[0].legend(ncol=2, frameon=False, loc="upper right")

    adjacent_frequency = 0.5 * (
        frequencies_mhz[:-1] + frequencies_mhz[1:]
    )
    axes[1].plot(
        adjacent_frequency,
        adjacent_correlation,
        color="#176B87",
        linewidth=1.2,
    )
    axes[1].set_ylabel("Adjacent\ncorrelation")
    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].set_ylim(
        min(0.0, float(np.min(adjacent_correlation)) - 0.02), 1.01
    )
    axes[1].grid(color="#C7CDD1", linewidth=0.5, alpha=0.45)
    figure.align_ylabels(axes)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    bands = _parse_bands(args.band)
    with np.load(args.sky_cache, allow_pickle=False) as archive:
        frequencies_mhz = np.asarray(
            archive["frequencies_mhz"], dtype=np.float64
        )
        eor_jy = np.asarray(archive["eor_jy"], dtype=np.float64)
        k2jy = np.asarray(
            archive["k2jy_per_pixel"], dtype=np.float64
        ).reshape(-1)
    if (
        frequencies_mhz.ndim != 1
        or k2jy.shape != frequencies_mhz.shape
        or eor_jy.shape[0] != frequencies_mhz.size
    ):
        raise ValueError("Sky cache frequency and EoR arrays differ")
    source_count = int(eor_jy.shape[1])
    source_size = int(round(math.sqrt(source_count)))
    if source_size * source_size != source_count:
        raise ValueError("EoR sky does not contain square image planes")
    eor_k = (eor_jy / k2jy[:, None]).reshape(
        frequencies_mhz.size, source_size, source_size
    )
    centered = eor_k - np.mean(eor_k, axis=(1, 2), keepdims=True)
    channel_variance = np.mean(np.square(centered), axis=(1, 2))
    flattened = centered.reshape(frequencies_mhz.size, -1)
    adjacent_covariance = np.mean(
        flattened[:-1] * flattened[1:], axis=1
    )
    adjacent_correlation = adjacent_covariance / np.sqrt(
        channel_variance[:-1] * channel_variance[1:]
    )

    summaries = [
        _band_summary(
            definition=definition,
            frequencies_mhz=frequencies_mhz,
            channel_variance=channel_variance,
        )
        for definition in bands
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "visibility_qbeta_lightcone_stationarity_audit",
        "schema_version": 1,
        "sky_cache": str(args.sky_cache),
        "frequency_count": int(frequencies_mhz.size),
        "frequency_first_mhz": float(frequencies_mhz[0]),
        "frequency_last_mhz": float(frequencies_mhz[-1]),
        "channel_variance_mean_k2": float(np.mean(channel_variance)),
        "channel_variance_minimum_k2": float(np.min(channel_variance)),
        "channel_variance_maximum_k2": float(np.max(channel_variance)),
        "adjacent_correlation_minimum": float(
            np.min(adjacent_correlation)
        ),
        "adjacent_correlation_median": float(
            np.median(adjacent_correlation)
        ),
        "adjacent_correlation_maximum": float(
            np.max(adjacent_correlation)
        ),
        "bands": {summary["label"]: summary for summary in summaries},
        "frequencies_mhz": frequencies_mhz,
        "channel_spatial_variance_k2": channel_variance,
        "adjacent_channel_correlation": adjacent_correlation,
    }
    _atomic_json(args.out_dir / "summary.json", result)
    _plot(
        path=args.out_dir / "lightcone_stationarity.png",
        frequencies_mhz=frequencies_mhz,
        channel_variance=channel_variance,
        adjacent_correlation=adjacent_correlation,
        bands=summaries,
    )
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
