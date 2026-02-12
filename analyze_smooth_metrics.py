#!/usr/bin/env python3
"""
Analyze smoothness indicators directly from injected/recovered FG/EoR cubes.

This script does not evaluate optimization loss values. It computes derivative-domain
summary metrics from cubes and reports:
  - FG/EoR separability for injected cubes
  - FG/EoR separability for recovered cubes
  - alignment between recovered and injected separability
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from astropy.io import fits


EPS = 1e-12


@dataclass(frozen=True)
class SmoothSpec:
    name: str
    diff_order: int
    primary_metric: str


SMOOTH_SPECS: Tuple[SmoothSpec, ...] = (
    SmoothSpec(name="diff3_l2", diff_order=3, primary_metric="rms"),
    SmoothSpec(name="diff2_l2", diff_order=2, primary_metric="rms"),
    SmoothSpec(name="diff2_huber", diff_order=2, primary_metric="huber_raw"),
    SmoothSpec(name="diff1_l1", diff_order=1, primary_metric="abs_mean"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct smoothness-indicator comparison for FG/EoR cubes.")
    parser.add_argument("--fg-true", type=Path, required=True, help="Injected FG cube FITS.")
    parser.add_argument("--eor-true", type=Path, required=True, help="Injected EoR cube FITS.")
    parser.add_argument("--fg-est", type=Path, required=True, help="Recovered FG cube FITS.")
    parser.add_argument("--eor-est", type=Path, required=True, help="Recovered EoR cube FITS.")
    parser.add_argument("--freq-axis", type=int, default=0, help="Frequency axis index.")
    parser.add_argument("--huber-delta", type=float, default=0.02, help="Delta for raw Huber indicator.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--output-md", type=Path, default=None, help="Output markdown summary path.")
    return parser.parse_args()


def load_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube for {path}, got shape {data.shape}.")
    return data


def move_freq_axis(cube: np.ndarray, freq_axis: int) -> np.ndarray:
    axis = int(freq_axis)
    if axis < 0:
        axis = cube.ndim + axis
    if axis < 0 or axis >= cube.ndim:
        raise ValueError(f"Invalid freq_axis={freq_axis} for shape {cube.shape}.")
    if axis == 0:
        return cube
    return np.moveaxis(cube, axis, 0)


def _huber_raw(x: np.ndarray, delta: float) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def derivative_metrics(cube: np.ndarray, *, diff_order: int, huber_delta: float) -> Dict[str, float]:
    if cube.shape[0] < diff_order + 1:
        raise ValueError(
            f"Need at least {diff_order + 1} frequency channels for diff_order={diff_order}; got {cube.shape[0]}."
        )
    diff = np.diff(cube, n=diff_order, axis=0)
    abs_diff = np.abs(diff)
    mean = float(np.mean(diff))
    std = float(np.std(diff))
    rms = float(np.sqrt(np.mean(diff * diff)))
    abs_mean = float(np.mean(abs_diff))
    p90_abs = float(np.percentile(abs_diff, 90.0))
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    robust_sigma = float(max(mad * 1.4826, EPS))
    huber_raw = float(np.mean(_huber_raw(diff, float(huber_delta))))
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "abs_mean": abs_mean,
        "p90_abs": p90_abs,
        "robust_sigma": robust_sigma,
        "huber_raw": huber_raw,
    }


def _safe_ratio(a: float, b: float) -> float:
    denom = b if abs(b) > EPS else (EPS if b >= 0 else -EPS)
    return float(a / denom)


def evaluate(
    fg_true: np.ndarray,
    eor_true: np.ndarray,
    fg_est: np.ndarray,
    eor_est: np.ndarray,
    *,
    huber_delta: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for spec in SMOOTH_SPECS:
        fg_true_m = derivative_metrics(fg_true, diff_order=spec.diff_order, huber_delta=huber_delta)
        eor_true_m = derivative_metrics(eor_true, diff_order=spec.diff_order, huber_delta=huber_delta)
        fg_est_m = derivative_metrics(fg_est, diff_order=spec.diff_order, huber_delta=huber_delta)
        eor_est_m = derivative_metrics(eor_est, diff_order=spec.diff_order, huber_delta=huber_delta)

        key = spec.primary_metric
        sep_true = _safe_ratio(fg_true_m[key], eor_true_m[key])
        sep_est = _safe_ratio(fg_est_m[key], eor_est_m[key])
        sep_alignment = _safe_ratio(sep_est, sep_true)

        row: Dict[str, float] = {
            "smooth_mode": spec.name,
            "diff_order": float(spec.diff_order),
            "primary_metric": key,
            "fg_true_primary": float(fg_true_m[key]),
            "eor_true_primary": float(eor_true_m[key]),
            "fg_est_primary": float(fg_est_m[key]),
            "eor_est_primary": float(eor_est_m[key]),
            "sep_true_fg_over_eor": sep_true,
            "sep_est_fg_over_eor": sep_est,
            "sep_alignment_est_over_true": sep_alignment,
            "fg_primary_alignment_est_over_true": _safe_ratio(fg_est_m[key], fg_true_m[key]),
            "eor_primary_alignment_est_over_true": _safe_ratio(eor_est_m[key], eor_true_m[key]),
            "fg_true_robust_sigma": fg_true_m["robust_sigma"],
            "eor_true_robust_sigma": eor_true_m["robust_sigma"],
            "fg_est_robust_sigma": fg_est_m["robust_sigma"],
            "eor_est_robust_sigma": eor_est_m["robust_sigma"],
            "fg_true_p90_abs": fg_true_m["p90_abs"],
            "eor_true_p90_abs": eor_true_m["p90_abs"],
            "fg_est_p90_abs": fg_est_m["p90_abs"],
            "eor_est_p90_abs": eor_est_m["p90_abs"],
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Smooth Indicator Comparison\n")
    lines.append(
        "| mode | metric | inj FG/EoR | rec FG/EoR | rec/inj sep | FG rec/inj | EoR rec/inj |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['smooth_mode']} | {r['primary_metric']} | "
            f"{r['sep_true_fg_over_eor']:.4g} | {r['sep_est_fg_over_eor']:.4g} | "
            f"{r['sep_alignment_est_over_true']:.4g} | "
            f"{r['fg_primary_alignment_est_over_true']:.4g} | "
            f"{r['eor_primary_alignment_est_over_true']:.4g} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive.")

    fg_true = move_freq_axis(load_fits(args.fg_true), args.freq_axis)
    eor_true = move_freq_axis(load_fits(args.eor_true), args.freq_axis)
    fg_est = move_freq_axis(load_fits(args.fg_est), args.freq_axis)
    eor_est = move_freq_axis(load_fits(args.eor_est), args.freq_axis)

    rows = evaluate(
        fg_true,
        eor_true,
        fg_est,
        eor_est,
        huber_delta=float(args.huber_delta),
    )

    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.output_md:
        write_md(args.output_md, rows)

    for row in rows:
        print(
            f"{row['smooth_mode']}: inj_sep={row['sep_true_fg_over_eor']:.4g}, "
            f"rec_sep={row['sep_est_fg_over_eor']:.4g}, "
            f"sep_align={row['sep_alignment_est_over_true']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
