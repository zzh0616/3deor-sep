#!/usr/bin/env python3
"""Combine thermal, gain, flag, and existing PB robustness diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROFILE = "quad_kperp_response"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-summary", type=Path, required=True)
    parser.add_argument("--exact-coarse-summary", type=Path, required=True)
    parser.add_argument(
        "--flag-root",
        action="append",
        default=[],
        help="Flag scenario as label=/path/to/root.",
    )
    parser.add_argument("--pb-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _parse_flag_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--flag-root must have label=path form")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError("--flag-root must have non-empty label and path")
    return label, Path(raw_path)


def _flag_summary(label: str, root: Path) -> dict[str, Any]:
    combined = _load_json(root / "combined" / "result.json")
    coarse = _load_json(root / "coarse" / "summary.json")
    profile = coarse["profiles"][PROFILE]
    thermal_path = root / "thermal" / "summary.json"
    result = {
        "label": label,
        "flagged_input_frequency_indices": combined["qbeta"][
            "flagged_input_frequency_indices"
        ],
        "flagged_channel_count": len(
            combined["qbeta"]["flagged_input_frequency_indices"]
        ),
        "selected_group_count": profile["selected_group_count"],
        "strict_group_count": profile["strict_group_count"],
        "selected_nominal_cell_count": profile[
            "selected_nominal_cell_count"
        ],
        "selected_window_fraction_minimum": profile[
            "selected_window_fraction_minimum"
        ],
        "foreground_effect_maximum_fraction": profile[
            "foreground_effect_maximum_fraction"
        ],
        "bank_eor": profile["bank_eor"],
        "bank_total": profile["bank_total"],
        "heldout_total_worst_relative_l2": profile[
            "heldout_total_worst_relative_l2"
        ],
    }
    if thermal_path.exists():
        thermal = _load_json(thermal_path)
        result["thermal"] = thermal["thermal_results"]
        result["cross_power_closure"] = thermal["cross_power_closure"]
    return result


def _pb_comparison(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path)
    scenarios = payload["scenarios"]
    exact = next(row for row in scenarios if row["model"] == "exact")
    rows = []
    for scenario in scenarios:
        rows.append(
            {
                **scenario,
                "integrated_ratio_change_from_paired_exact": (
                    scenario["bank_total_integrated_power_ratio"]
                    - exact["bank_total_integrated_power_ratio"]
                ),
                "relative_l2_change_from_paired_exact": (
                    scenario["bank_total_relative_l2"]
                    - exact["bank_total_relative_l2"]
                ),
            }
        )
    return {
        "source": str(path),
        "comparison_contract": (
            "Paired PB result from the earlier 64-to-32 response run; absolute "
            "metrics must not be mixed with the local-redshift baseline."
        ),
        "scenarios": rows,
    }


def _write_flag_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "label",
        "flagged_channel_count",
        "flagged_input_frequency_indices",
        "selected_group_count",
        "strict_group_count",
        "selected_nominal_cell_count",
        "bank_total_integrated_power_ratio",
        "bank_total_relative_l2",
        "bank_total_p90_absolute_error_fraction",
        "foreground_effect_maximum_fraction",
        "heldout_total_worst_relative_l2",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "flagged_channel_count": row["flagged_channel_count"],
                    "flagged_input_frequency_indices": ";".join(
                        str(item)
                        for item in row["flagged_input_frequency_indices"]
                    ),
                    "selected_group_count": row["selected_group_count"],
                    "strict_group_count": row["strict_group_count"],
                    "selected_nominal_cell_count": row[
                        "selected_nominal_cell_count"
                    ],
                    "bank_total_integrated_power_ratio": row["bank_total"][
                        "integrated_power_ratio"
                    ],
                    "bank_total_relative_l2": row["bank_total"]["relative_l2"],
                    "bank_total_p90_absolute_error_fraction": row["bank_total"][
                        "p90_absolute_error_fraction"
                    ],
                    "foreground_effect_maximum_fraction": row[
                        "foreground_effect_maximum_fraction"
                    ],
                    "heldout_total_worst_relative_l2": row[
                        "heldout_total_worst_relative_l2"
                    ],
                }
            )


def _plot(
    path: Path,
    *,
    noise: dict[str, Any],
    exact: dict[str, Any],
    flags: list[dict[str, Any]],
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "fg_rmw_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    blue = "#176B87"
    orange = "#D96C2F"
    green = "#4F7D54"
    gray = "#6D7278"
    thermal_rows = sorted(
        noise["thermal_results"].items(),
        key=lambda item: float(item[1]["total_integration_hours"]),
    )
    hours = np.asarray(
        [row[1]["total_integration_hours"] for row in thermal_rows]
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.2), constrained_layout=True)
    ax = axes[0, 0]
    for significance, color in (("10sigma", blue), ("25sigma", orange)):
        factors = [
            row[1]["boosted_benchmarks"][significance]["amplitude_factor"]
            for row in thermal_rows
        ]
        ax.plot(
            hours,
            factors,
            marker="o",
            linewidth=2,
            color=color,
            label=significance.replace("sigma", r"$\sigma$"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Equivalent integration time [h]")
    ax.set_ylabel("Required EoR amplitude factor")
    ax.set_title("Detection-scale injection")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    labels = [f"{value:g} h" for value in hours]
    coverage68 = [
        row[1]["foreground_plus_eor_unboosted"]
        ["noise_coverage_about_noiseless_expectation"]["coverage_68"]
        for row in thermal_rows
    ]
    coverage95 = [
        row[1]["foreground_plus_eor_unboosted"]
        ["noise_coverage_about_noiseless_expectation"]["coverage_95"]
        for row in thermal_rows
    ]
    positions = np.arange(len(labels), dtype=float)
    width = 0.32
    ax.bar(positions - width / 2, coverage68, width, color=blue, label="68%")
    ax.bar(positions + width / 2, coverage95, width, color=orange, label="95%")
    ax.axhline(0.682689, color=blue, linestyle=":", linewidth=1)
    ax.axhline(0.95, color=orange, linestyle=":", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Independent-split cross-power")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.18, axis="y")

    ax = axes[1, 0]
    gain = noise["gain_results"]
    for profile, color, marker in (
        ("smooth", blue, "o"),
        ("ripple", orange, "s"),
    ):
        rows = sorted(
            gain[profile].values(), key=lambda row: row["station_log_amplitude_rms"]
        )
        rms = np.asarray([row["station_log_amplitude_rms"] for row in rows])
        l2 = np.asarray(
            [
                row["gain_only_metrics"]["mean_estimate_relative_l2"]
                for row in rows
            ]
        )
        contamination = np.asarray(
            [
                row["foreground_contamination"][
                    "mean_integrated_absolute_foreground_to_target"
                ]
                for row in rows
            ]
        )
        ax.plot(
            rms,
            100.0 * l2,
            color=color,
            marker=marker,
            linewidth=2,
            label=f"{profile}: total L2",
        )
        ax.plot(
            rms,
            100.0 * contamination,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{profile}: |FG|/EoR",
        )
    ax.axhline(
        100.0 * exact["bank_total"]["relative_l2"],
        color=gray,
        linestyle=":",
        linewidth=1.2,
        label="matched baseline L2",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Station gain residual RMS")
    ax.set_ylabel("Error or contamination [%]")
    ax.set_title("Direction-independent gain stress")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=7)

    ax = axes[1, 1]
    flag_labels = ["none"] + [row["label"] for row in flags]
    l2 = [exact["bank_total"]["relative_l2"]] + [
        row["bank_total"]["relative_l2"] for row in flags
    ]
    groups = [exact["selected_group_count"]] + [
        row["selected_group_count"] for row in flags
    ]
    x = np.arange(len(flag_labels), dtype=float)
    ax.bar(x, 100.0 * np.asarray(l2), color=green, alpha=0.82)
    ax.set_yscale("log")
    ax.set_xticks(x, flag_labels, rotation=18, ha="right")
    ax.set_ylabel("Noiseless total relative L2 [%]", color=green)
    ax.tick_params(axis="y", labelcolor=green)
    ax.set_title("Known-channel flag stress")
    ax.grid(alpha=0.18, axis="y")
    twin = ax.twinx()
    twin.plot(x, groups, color=gray, marker="o", linewidth=1.8)
    twin.set_ylabel("Selected coarse groups", color=gray)
    twin.tick_params(axis="y", labelcolor=gray)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    noise = _load_json(args.noise_summary)
    exact_payload = _load_json(args.exact_coarse_summary)
    exact = exact_payload["profiles"][PROFILE]
    flags = [
        _flag_summary(label, root)
        for label, root in map(_parse_flag_root, args.flag_root)
    ]
    pb = _pb_comparison(args.pb_summary)
    result = {
        "schema": "visibility_qbeta_combined_robustness_summary",
        "schema_version": 1,
        "profile": PROFILE,
        "noise_source": str(args.noise_summary),
        "exact_coarse_source": str(args.exact_coarse_summary),
        "cross_power_closure": noise["cross_power_closure"],
        "reference_group": noise["reference_group"],
        "thermal_noise_contract": noise["thermal_noise_contract"],
        "thermal_results": noise["thermal_results"],
        "gain_residual_contract": noise["gain_residual_contract"],
        "gain_results": noise["gain_results"],
        "exact_no_flag": exact,
        "flag_scenarios": flags,
        "primary_beam_mismatch": pb,
        "limitations": noise["limitations"]
        + [
            "Known flags are zero-filled inside the fixed DPSS basis; no "
            "DAYENUREST-style inpainting is used.",
            "The archived PB mismatch comparison comes from an earlier paired "
            "64-to-32 run rather than the local-redshift noise baseline.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "combined_summary.json", result)
    _write_flag_csv(args.out_dir / "flag_summary.csv", flags)
    _plot(
        args.out_dir / "visibility_qbeta_noise_systematics.png",
        noise=noise,
        exact=exact,
        flags=flags,
    )
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
