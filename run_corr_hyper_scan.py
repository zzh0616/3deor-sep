#!/usr/bin/env python3
"""
Corr-term hyperparameter scan on top of a fixed base baseline.

This scan keeps base terms fixed (data + FG smoothness + EoR amplitude prior),
and enables the per-frequency FG/EoR correlation extra term ("corr") with
different weights and prior sigmas.

Evaluation is done post-hoc from FITS outputs (not loss values):
- primary: per-frequency corr(EoR_est[f], EoR_true[f]) across spatial pixels
- secondary: per-frequency corr(FG_est[f], EoR_est[f]) to detect spurious coupling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    corr_weight: float
    corr_prior_sigma: float
    extra_loss_start_iter: int
    extra_loss_ramp_iters: int

    def key(self) -> Tuple[object, ...]:
        return (
            round(float(self.corr_weight), 12),
            round(float(self.corr_prior_sigma), 12),
            int(self.extra_loss_start_iter),
            int(self.extra_loss_ramp_iters),
        )


@dataclass
class JobSpec:
    dataset: DatasetSpec
    candidate: CandidateSpec
    gpu_index: int
    run_dir: Path
    config_path: Path
    log_path: Path
    fg_output: Path
    eor_output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corr-term hyperparameter scan.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="3dnet dir (default <work-root>/code/3dnet or <work-root>/3dnet).")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/corr_scan_<timestamp>).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="cube1,cube2",
        help="Comma-separated datasets. Available: cube1,cube2.",
    )
    parser.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset->GPU mapping, e.g. cube1:0,cube2:1",
    )
    parser.add_argument("--max-concurrent-jobs", type=int, default=2, help="Max concurrent dataset jobs per candidate.")
    parser.add_argument("--num-iters", type=int, default=2500, help="Iterations per run.")
    parser.add_argument("--print-every", type=int, default=200, help="Iteration logging interval.")
    parser.add_argument("--cut-size-frac", type=float, default=0.30, help="Spatial center cut fraction.")
    parser.add_argument("--freq-start-mhz", type=float, default=106.0, help="Starting frequency in MHz.")
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1, help="Channel spacing in MHz.")
    parser.add_argument("--data-error", type=float, default=0.005, help="Data error scalar prior.")

    # Fixed base baseline (defaults reflect current best-known base baseline).
    parser.add_argument("--base-beta", type=float, default=0.5, help="FG smooth weight beta.")
    parser.add_argument("--base-gamma", type=float, default=0.6, help="EoR amplitude prior weight gamma.")
    parser.add_argument("--base-eor-prior-sigma", type=float, default=0.02, help="EoR amplitude sigma (dead-zone outside threshold).")
    parser.add_argument("--base-eor-amp-threshold", type=float, default=0.1, help="EoR dead-zone threshold.")
    parser.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2", choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"])
    parser.add_argument("--base-fg-smooth-mean", type=float, default=0.002, help="Scalar FG smooth prior mean (applied to finite differences).")
    parser.add_argument("--base-fg-smooth-sigma", type=float, default=0.004, help="Scalar FG smooth prior sigma (applied to finite differences).")
    parser.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0, help="Huber delta for diff2_huber.")

    # Optimizer knobs (kept fixed unless caller changes).
    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum when optimizer=sgd.")
    parser.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    parser.add_argument("--lr-plateau-patience", type=int, default=240)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=80)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    # Corr scan grid.
    parser.add_argument(
        "--corr-weight-list",
        type=str,
        default="0.05,0.1,0.2,0.5,1.0,2.0",
        help="Comma-separated corr weights.",
    )
    parser.add_argument(
        "--corr-sigma-list",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated corr prior sigma values.",
    )
    parser.add_argument("--corr-prior-mean", type=float, default=0.0, help="Corr prior mean (fixed for this scan).")
    parser.add_argument("--extra-loss-start-iter", type=int, default=500, help="Iteration where corr term activates.")
    parser.add_argument("--extra-loss-ramp-iters", type=int, default=0, help="Ramp iterations for corr term.")
    parser.add_argument("--include-control", action="store_true", help="Include a base-only control run (corr disabled).")
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Comma-separated candidate names to run; empty means all generated candidates.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run separation_cli.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate configs/candidates but do not execute jobs.")
    return parser.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid gpu map token: {token}")
        key, value = token.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


def build_datasets(data_dir: Path) -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="cube1",
            input_cube=data_dir / "all_cube2.fits",
            fg_true_cube=data_dir / "fg_cube2.fits",
            eor_true_cube=data_dir / "eor_cube2.fits",
        ),
        DatasetSpec(
            name="cube2",
            input_cube=data_dir / "all_cube1.fits",
            fg_true_cube=data_dir / "fg_cube1.fits",
            eor_true_cube=data_dir / "eor_cube1.fits",
        ),
    ]


def _fmt_float_token(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    s = f"{float(value):.6g}"
    if "e" in s or "E" in s:
        s = f"{float(value):.12f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def generate_candidates(
    *,
    corr_weight_list: Sequence[float],
    corr_sigma_list: Sequence[float],
    extra_loss_start_iter: int,
    extra_loss_ramp_iters: int,
    include_control: bool,
) -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    if include_control:
        out.append(
            CandidateSpec(
                name="control_base",
                corr_weight=0.0,
                corr_prior_sigma=float(corr_sigma_list[0]) if corr_sigma_list else 0.1,
                extra_loss_start_iter=int(extra_loss_start_iter),
                extra_loss_ramp_iters=int(extra_loss_ramp_iters),
            )
        )
    for w in corr_weight_list:
        for s in corr_sigma_list:
            name = f"corr_w{_fmt_float_token(float(w))}_s{_fmt_float_token(float(s))}"
            out.append(
                CandidateSpec(
                    name=name,
                    corr_weight=float(w),
                    corr_prior_sigma=float(s),
                    extra_loss_start_iter=int(extra_loss_start_iter),
                    extra_loss_ramp_iters=int(extra_loss_ramp_iters),
                )
            )
    # Deduplicate by key while preserving order.
    dedup: List[CandidateSpec] = []
    seen = set()
    for c in out:
        if c.key() in seen:
            continue
        seen.add(c.key())
        dedup.append(c)
    return dedup


def _extract_cut_indices(shape: Sequence[int], cut_size_frac: float) -> Optional[Tuple[int, int, int, int]]:
    if len(shape) != 3:
        return None
    _, ny, nx = (int(shape[0]), int(shape[1]), int(shape[2]))
    frac = float(cut_size_frac)
    if not math.isfinite(frac) or frac <= 0.0 or frac > 1.0:
        raise ValueError("cut_size_frac must be in (0, 1].")
    size_x = max(1, int(round(nx * frac)))
    size_y = max(1, int(round(ny * frac)))
    cx = nx // 2
    cy = ny // 2
    x0 = max(0, cx - size_x // 2)
    y0 = max(0, cy - size_y // 2)
    x1 = min(nx, x0 + size_x)
    y1 = min(ny, y0 + size_y)
    return (x0, x1, y0, y1)


def _load_cube_cut(path: Path, cut: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if cut is None:
        return data
    x0, x1, y0, y1 = cut
    return data[:, y0:y1, x0:x1]


def _center_crop_xy(cube: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    z, ty, tx = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
    if cube.shape[0] != z:
        cube = cube[:z, :, :]
    ny, nx = (cube.shape[1], cube.shape[2])
    x0 = max(0, (nx - tx) // 2)
    y0 = max(0, (ny - ty) // 2)
    return cube[:, y0 : y0 + ty, x0 : x0 + tx]


def _frequency_correlations(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    if est.shape != true.shape:
        raise ValueError(f"shape mismatch: est={est.shape} true={true.shape}")
    if est.ndim != 3:
        raise ValueError(f"expected 3D cubes, got {est.shape}")
    nfreq = est.shape[0]
    out = np.full((nfreq,), np.nan, dtype=np.float64)
    for i in range(nfreq):
        a = est[i].reshape(-1).astype(np.float64)
        b = true[i].reshape(-1).astype(np.float64)
        am = a.mean()
        bm = b.mean()
        ac = a - am
        bc = b - bm
        denom = np.linalg.norm(ac) * np.linalg.norm(bc)
        if denom <= 0:
            continue
        out[i] = float(np.dot(ac, bc) / denom)
    return out


def _write_frequency_corr_profile(path: Path, corr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, val in enumerate(corr.tolist()):
            w.writerow([int(i), "" if not math.isfinite(float(val)) else float(val)])


def _score_from_corr(corr: np.ndarray) -> float:
    vals = corr[np.isfinite(corr)]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    return 0.7 * mean + 0.2 * median + 0.1 * p10


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _candidate_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate")), []).append(row)
    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        ok = [r for r in items if str(r.get("status")) == "ok"]
        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in ok if r.get(key) is not None and math.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")
        summary: Dict[str, object] = {
            "candidate": cand,
            "n": len(items),
            "n_ok": len(ok),
            "eor_corr_score_mean": _mean("eor_corr_score"),
            "eor_corr_mean_mean": _mean("eor_corr_mean"),
            "eor_corr_p10_mean": _mean("eor_corr_p10"),
            "fg_eor_corr_abs_mean_mean": _mean("fg_eor_corr_abs_mean"),
            "fg_eor_corr_mean_mean": _mean("fg_eor_corr_mean"),
        }
        out.append(summary)
    out.sort(key=lambda r: float(r.get("eor_corr_score_mean", float("-inf"))), reverse=True)
    return out


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Corr Scan Summary\n\n")
    lines.append("## Fixed Baseline\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("## Ranked Candidates\n\n")
    lines.append("| rank | candidate | n_ok/n | eor_score | eor_corr_mean | eor_corr_p10 | fg_eor_abs_mean | fg_eor_mean |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['candidate']} | {row['n_ok']}/{row['n']} | "
            f"{float(row['eor_corr_score_mean']):.6f} | "
            f"{float(row['eor_corr_mean_mean']):.6f} | "
            f"{float(row['eor_corr_p10_mean']):.6f} | "
            f"{float(row['fg_eor_corr_abs_mean_mean']):.6f} | "
            f"{float(row['fg_eor_corr_mean_mean']):.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _build_config(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    corr_enabled = float(candidate.corr_weight) > 0.0
    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": ["corr"] if corr_enabled else [],
            "extra_loss_start_iter": int(candidate.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(candidate.extra_loss_ramp_iters),
            "optimizer_name": str(args.optimizer_name),
            "momentum": float(args.momentum),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_plateau_patience": int(args.lr_plateau_patience),
            "lr_plateau_factor": float(args.lr_plateau_factor),
            "lr_plateau_min_delta": float(args.lr_plateau_min_delta),
            "lr_plateau_cooldown": int(args.lr_plateau_cooldown),
            "lr_min": float(args.lr_min),
            "freq_start_mhz": float(args.freq_start_mhz),
            "freq_delta_mhz": float(args.freq_delta_mhz),
        },
        "cut_xy": {
            "enabled": True,
            "unit": "frac",
            "center_x": 0.5,
            "center_y": 0.5,
            "size": float(args.cut_size_frac),
        },
        "weights": {
            "alpha": 1.0,
            "beta": float(args.base_beta),
            "gamma": float(args.base_gamma),
            "corr_weight": float(candidate.corr_weight) if corr_enabled else 0.0,
            "lagcorr_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            "corr_prior_mean": float(args.corr_prior_mean),
            "corr_prior_sigma": float(candidate.corr_prior_sigma),
        },
        "evaluation": {
            "true_eor_cube": str(dataset.eor_true_cube),
            "diagnose_input": False,
            "enable_corr_check": True,
            "corr_check_every": max(50, int(args.print_every)),
            "corr_plot": str(run_dir / "eor_corr.png"),
        },
        "init": {
            "init_fg_cube": "",
            "init_eor_cube": "",
        },
        "scan_meta": {
            "candidate_name": candidate.name,
        },
    }
    return cfg


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    return [str(args.python_bin), str(cli_path), "--config", str(config_path)]


def _summarize_corr_stats(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_mean": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def _run_job_result_only(
    *,
    args: argparse.Namespace,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    true_eor: np.ndarray,
    true_fg: np.ndarray,
    return_code: int,
    runtime: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "candidate": candidate.name,
        "dataset": dataset.name,
        "status": "ok" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "runtime_sec": float(runtime),
        "corr_weight": float(candidate.corr_weight),
        "corr_prior_sigma": float(candidate.corr_prior_sigma),
        "extra_loss_start_iter": int(candidate.extra_loss_start_iter),
        "extra_loss_ramp_iters": int(candidate.extra_loss_ramp_iters),
        "beta": float(args.base_beta),
        "gamma": float(args.base_gamma),
        "eor_prior_sigma": float(args.base_eor_prior_sigma),
        "eor_amp_threshold": float(args.base_eor_amp_threshold),
        "fg_smooth_mode": str(args.base_fg_smooth_mode),
        "fg_smooth_mean": float(args.base_fg_smooth_mean),
        "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
        "optimizer_name": str(args.optimizer_name),
        "lr": float(args.lr),
        "lr_scheduler": str(args.lr_scheduler),
        "config_path": str(run_dir / "config.json"),
        "log_path": str(run_dir / "run.log"),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
    }
    if int(return_code) != 0:
        return row

    fg_out = run_dir / "fg_est.fits"
    eor_out = run_dir / "eor_est.fits"
    if not fg_out.exists() or not eor_out.exists():
        row["status"] = "failed"
        row["note"] = "missing_output"
        return row

    with fits.open(eor_out, memmap=True) as hdul:
        eor_est = np.asarray(hdul[0].data, dtype=np.float32)
    with fits.open(fg_out, memmap=True) as hdul:
        fg_est = np.asarray(hdul[0].data, dtype=np.float32)

    if true_eor.shape != eor_est.shape:
        true_eor_use = _center_crop_xy(true_eor, eor_est.shape)
    else:
        true_eor_use = true_eor
    if true_fg.shape != fg_est.shape:
        true_fg_use = _center_crop_xy(true_fg, fg_est.shape)
    else:
        true_fg_use = true_fg

    eor_corr = _frequency_correlations(eor_est, true_eor_use)
    _write_frequency_corr_profile(run_dir / "eor_corr_profile.csv", eor_corr)
    eor_stats = _summarize_corr_stats(eor_corr)
    row["eor_corr_mean"] = eor_stats["mean"]
    row["eor_corr_median"] = eor_stats["median"]
    row["eor_corr_p10"] = eor_stats["p10"]
    row["eor_corr_min"] = eor_stats["min"]
    row["eor_corr_max"] = eor_stats["max"]
    row["eor_corr_score"] = float(_score_from_corr(eor_corr))

    fg_eor_corr = _frequency_correlations(fg_est, eor_est)
    _write_frequency_corr_profile(run_dir / "fg_eor_corr_profile.csv", fg_eor_corr)
    fg_eor_stats = _summarize_corr_stats(fg_eor_corr)
    row["fg_eor_corr_mean"] = fg_eor_stats["mean"]
    row["fg_eor_corr_median"] = fg_eor_stats["median"]
    row["fg_eor_corr_p10"] = fg_eor_stats["p10"]
    row["fg_eor_corr_min"] = fg_eor_stats["min"]
    row["fg_eor_corr_max"] = fg_eor_stats["max"]
    row["fg_eor_corr_abs_mean"] = fg_eor_stats["abs_mean"]

    inj_fg_eor_corr = _frequency_correlations(true_fg_use, true_eor_use)
    inj_stats = _summarize_corr_stats(inj_fg_eor_corr)
    row["inj_fg_eor_corr_mean"] = inj_stats["mean"]
    row["inj_fg_eor_corr_abs_mean"] = inj_stats["abs_mean"]
    return row


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()

    # Local repo layout: either <work-root>/3dnet (local) or <work-root>/code/3dnet (remote root layout).
    if args.code_dir:
        code_dir = args.code_dir.resolve()
    else:
        code_dir = (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"corr_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir)
    enabled = {x.strip() for x in str(args.datasets).split(",") if x.strip()}
    datasets = [d for d in datasets_all if d.name in enabled]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")
    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

    corr_weight_list = _parse_float_list(args.corr_weight_list)
    corr_sigma_list = _parse_float_list(args.corr_sigma_list)
    if not corr_weight_list or not corr_sigma_list:
        raise ValueError("corr-weight-list and corr-sigma-list must be non-empty.")
    candidates = generate_candidates(
        corr_weight_list=corr_weight_list,
        corr_sigma_list=corr_sigma_list,
        extra_loss_start_iter=int(args.extra_loss_start_iter),
        extra_loss_ramp_iters=int(args.extra_loss_ramp_iters),
        include_control=bool(args.include_control),
    )
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        known = {c.name for c in candidates}
        unknown = sorted(allow - known)
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

    # Prepare dataset caches: cut indices + true cubes.
    ds_cache: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
        with fits.open(ds.input_cube, memmap=True) as hdul:
            in_shape = tuple(int(v) for v in hdul[0].data.shape)
        cut = _extract_cut_indices(in_shape, float(args.cut_size_frac))
        ds_cache[ds.name] = {
            "cut": cut,
            "true_eor": _load_cube_cut(ds.eor_true_cube, cut=cut),
            "true_fg": _load_cube_cut(ds.fg_true_cube, cut=cut),
        }

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": [{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(d).items()} for d in datasets],
        "candidates": [asdict(c) for c in candidates],
        "baseline_fixed": {
            "beta": float(args.base_beta),
            "gamma": float(args.base_gamma),
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_amp_threshold": float(args.base_eor_amp_threshold),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "data_error": float(args.data_error),
            "optimizer_name": str(args.optimizer_name),
            "lr": float(args.lr),
            "lr_scheduler": str(args.lr_scheduler),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "corr_prior_mean": float(args.corr_prior_mean),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] generated {len(candidates)} candidates under {output_dir}")
        return 0

    rows: List[Dict[str, object]] = []
    max_jobs = max(1, int(args.max_concurrent_jobs))
    for idx, cand in enumerate(candidates, start=1):
        print(f"[candidate {idx}/{len(candidates)}] {cand.name}")
        jobs: List[JobSpec] = []
        for ds in datasets:
            run_dir = output_dir / cand.name / ds.name
            jobs.append(
                JobSpec(
                    dataset=ds,
                    candidate=cand,
                    gpu_index=int(gpu_map[ds.name]),
                    run_dir=run_dir,
                    config_path=run_dir / "config.json",
                    log_path=run_dir / "run.log",
                    fg_output=run_dir / "fg_est.fits",
                    eor_output=run_dir / "eor_est.fits",
                )
            )

        active: List[Tuple[object, JobSpec, float, object]] = []
        queued = list(jobs)
        while queued or active:
            while queued and len(active) < max_jobs:
                job = queued.pop(0)
                job.run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_config(dataset=job.dataset, candidate=job.candidate, run_dir=job.run_dir, gpu_index=job.gpu_index, args=args)
                with job.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(cfg, handle, indent=2)
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                import subprocess  # local import to keep module import minimal

                proc = subprocess.Popen(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active.append((proc, job, time.time(), log_handle))
                print(f"  [launch] {job.dataset.name} gpu={job.gpu_index} pid={proc.pid}")

            still_active: List[Tuple[object, JobSpec, float, object]] = []
            for proc, job, t0, log_handle in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, t0, log_handle))
                    continue
                log_handle.close()
                runtime = time.time() - t0
                cache = ds_cache[job.dataset.name]
                true_eor = cache["true_eor"]
                true_fg = cache["true_fg"]
                assert isinstance(true_eor, np.ndarray) and isinstance(true_fg, np.ndarray)
                row = _run_job_result_only(
                    args=args,
                    dataset=job.dataset,
                    candidate=job.candidate,
                    run_dir=job.run_dir,
                    true_eor=true_eor,
                    true_fg=true_fg,
                    return_code=int(ret),
                    runtime=float(runtime),
                )
                rows.append(row)
                print(
                    f"  [done] {job.dataset.name} status={row['status']} "
                    f"eor_score={row.get('eor_corr_score')} fg_eor_abs_mean={row.get('fg_eor_corr_abs_mean')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "corr_scan_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows)
    rank_csv = output_dir / "corr_scan_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "corr_scan_summary.md", ranked, manifest["baseline_fixed"])
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

