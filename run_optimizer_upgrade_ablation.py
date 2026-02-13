#!/usr/bin/env python3
"""
Optimizer-upgrade ablation on top of a fixed best-known loss stack.

Purpose (Phase D):
- Compare a few optimizer engineering knobs (LR split / plateau reset / alternating updates)
  while keeping the loss configuration fixed, using truth only for evaluation.

This script is intentionally small and conservative: it runs a limited set of datasets
and candidates to de-risk optimizer changes before rolling out widely.
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

from dataset_registry import DatasetSpec, build_datasets, default_dataset_name_hint


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    lr_fg_factor: float
    lr_plateau_reset_on_phase_start: bool
    alt_update_mode: str
    alt_fg_steps: int
    alt_eor_steps: int

    def key(self) -> Tuple[object, ...]:
        return (
            round(float(self.lr_fg_factor), 8),
            bool(self.lr_plateau_reset_on_phase_start),
            str(self.alt_update_mode),
            int(self.alt_fg_steps),
            int(self.alt_eor_steps),
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
    p = argparse.ArgumentParser(description="Run optimizer upgrade ablation scan.")
    p.add_argument("--work-root", type=Path, default=Path.cwd())
    p.add_argument("--code-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--datasets", type=str, default="cube2,f140", help=f"Comma-separated datasets. Common: {default_dataset_name_hint()}")
    p.add_argument("--gpu-map", type=str, default="cube2:0,f140:1", help="Dataset->GPU mapping, e.g. cube2:0,f140:1")
    p.add_argument("--max-concurrent-jobs", type=int, default=2)
    p.add_argument("--num-iters", type=int, default=2500)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-start-mhz", type=float, default=106.0)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)

    # Best-known fixed loss stack (defaults set to current Phase C winner).
    p.add_argument("--data-error", type=float, default=0.005)
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument("--base-eor-amp-prior-mode", type=str, default="slice_rms_hinge", choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"])
    p.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2", choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"])
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Corr baseline (Phase A winner).
    p.add_argument("--extra-loss-start-iter", type=int, default=300)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=700)
    p.add_argument("--corr-weight", type=float, default=0.2)
    p.add_argument("--corr-prior-mean", type=float, default=0.0)
    p.add_argument("--corr-prior-sigma", type=float, default=0.2)
    p.add_argument("--corr-abs-threshold", type=float, default=0.08)
    p.add_argument("--corr-reduce", type=str, default="logsumexp", choices=["mean", "topk", "logsumexp"])
    p.add_argument("--corr-topk", type=int, default=8)
    p.add_argument("--corr-lse-alpha", type=float, default=10.0)
    p.add_argument("--corr-feature", type=str, default="diff1", choices=["raw", "diff1", "diff2"])
    p.add_argument("--corr-spatial-pool", type=int, default=1)

    # Lagcorr baseline (Phase B winner).
    p.add_argument("--lagcorr-weight", type=float, default=1.0)
    p.add_argument("--lagcorr-spatial-pool", type=int, default=4)
    p.add_argument("--lagcorr-rms-min", type=float, default=0.0)
    p.add_argument("--lagcorr-eor-start-iter", type=int, default=1200)
    p.add_argument("--lagcorr-eor-ramp-iters", type=int, default=800)
    p.add_argument("--lagcorr-eor-subterm-schedule", type=str, default="static", choices=["static", "staged_v1"])
    p.add_argument("--lagcorr-eor-tail-weight-mode", type=str, default="hard", choices=["hard", "sigmoid_chan", "sigmoid_chi_planck18"])
    p.add_argument("--lagcorr-eor-tail-eps", type=float, default=0.05)
    p.add_argument("--lagcorr-eor-neg-delta", type=float, default=0.0)
    p.add_argument("--lagcorr-eor-near-floor-mode", type=str, default="relative_rho1", choices=["absolute_mean", "relative_rho1"])
    p.add_argument("--lagcorr-eor-near-rho1-coeffs", type=str, default="0.5,0.25,0.15,0,0,0,0,0,0")
    p.add_argument("--lagcorr-eor-rebound-eps-act", type=float, default=0.05)
    p.add_argument("--lagcorr-eor-rebound-delta-up", type=float, default=0.02)
    p.add_argument("--lagcorr-eor-near-max-lag", type=int, default=10)
    p.add_argument("--lagcorr-eor-mid-max-lag", type=int, default=50)
    p.add_argument("--lagcorr-eor-far-min-lag", type=int, default=70)

    # Extra EoR priors (Phase C winner).
    p.add_argument("--eor-mean-weight", type=float, default=0.1)
    p.add_argument("--eor-hf-weight", type=float, default=0.3)
    p.add_argument("--eor-hf-percent", type=float, default=0.7)
    p.add_argument("--eor-hf-r-max", type=float, default=0.85)

    # Optimizer baseline.
    p.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-patience", type=int, default=240)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-cooldown", type=int, default=80)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # Ablation grid.
    p.add_argument("--lr-fg-factor-list", type=str, default="1.0,0.5")
    p.add_argument("--plateau-reset-list", type=str, default="0,1", help="Comma-separated 0/1 toggles.")
    p.add_argument("--alt-mode-list", type=str, default="none,fg_then_eor")
    p.add_argument("--alt-fg-steps", type=int, default=5)
    p.add_argument("--alt-eor-steps", type=int, default=1)
    p.add_argument("--candidate-names", type=str, default="")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(t) for t in _parse_csv_tokens(text)]


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for t in _parse_csv_tokens(text):
        out.append(int(float(t)))
    return out


def parse_gpu_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid gpu map token: {token}")
        key, value = token.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out


def _fmt_float_token(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    s = f"{float(value):.6g}"
    if "e" in s or "E" in s:
        s = f"{float(value):.12f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def generate_candidates(args: argparse.Namespace) -> List[CandidateSpec]:
    lr_fg_list = _parse_float_list(args.lr_fg_factor_list)
    reset_list = _parse_int_list(args.plateau_reset_list)
    alt_list = [t.strip() for t in str(args.alt_mode_list).split(",") if t.strip()]
    if not lr_fg_list or not reset_list or not alt_list:
        raise ValueError("lr-fg-factor-list / plateau-reset-list / alt-mode-list must be non-empty.")

    out: List[CandidateSpec] = []
    for lr_fg in lr_fg_list:
        for rst in reset_list:
            for alt in alt_list:
                alt_norm = str(alt).strip().lower()
                if alt_norm not in {"none", "fg_then_eor"}:
                    raise ValueError(f"Unsupported alt mode '{alt}'.")
                name = (
                    f"opt_fg{_fmt_float_token(float(lr_fg))}"
                    f"_rst{int(bool(int(rst)))}"
                    f"_alt{alt_norm}"
                )
                out.append(
                    CandidateSpec(
                        name=name,
                        lr_fg_factor=float(lr_fg),
                        lr_plateau_reset_on_phase_start=bool(int(rst)),
                        alt_update_mode=alt_norm,
                        alt_fg_steps=int(args.alt_fg_steps),
                        alt_eor_steps=int(args.alt_eor_steps),
                    )
                )

    # Dedup by key (keep first occurrence).
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


def _frequency_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    out = np.zeros((a.shape[0],), dtype=np.float64)
    for i in range(a.shape[0]):
        x = a[i].reshape(-1).astype(np.float64)
        y = b[i].reshape(-1).astype(np.float64)
        x -= x.mean()
        y -= y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        out[i] = 0.0 if denom < 1e-18 else float(np.dot(x, y) / denom)
    return out


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.clip(finite, -1.0, 1.0)))


def _summarize_corr_stats(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "abs_mean": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def _compute_lagcorr_profile(cube: np.ndarray, lag_channels: Sequence[int], *, max_pairs: Optional[int]) -> List[float]:
    nfreq = int(cube.shape[0])
    flat = cube.reshape(nfreq, -1).astype(np.float64, copy=False)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms = np.clip(norms, 1e-12, None)
    means: List[float] = []
    for lag in lag_channels:
        lag_i = int(lag)
        total = max(0, nfreq - lag_i)
        if total <= 0:
            means.append(float("nan"))
            continue
        use_pairs = total if max_pairs is None else min(total, int(max_pairs))
        idx = np.arange(use_pairs, dtype=np.int64)
        jdx = idx + lag_i
        corr = np.sum(centered[idx] * centered[jdx], axis=1) / np.clip(norms[idx] * norms[jdx], 1e-12, None)
        means.append(float(np.mean(corr)))
    return means


def _safe_profile_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size != b.size or a.size < 2:
        return None
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-18:
        return None
    return float(np.dot(a, b) / den)


def _lag_profile_metrics(est: Sequence[float], true: Sequence[float], *, tail_threshold_mhz: float = 2.0) -> Dict[str, Optional[float]]:
    est_a = np.asarray(est, dtype=np.float64)
    true_a = np.asarray(true, dtype=np.float64)
    if est_a.size != true_a.size or est_a.size == 0:
        return {"rmse": None, "profile_corr": None, "tail_abs_gap": None}
    diff = est_a - true_a
    tail_mask = np.asarray([float(v) >= float(tail_threshold_mhz) for v in LAG_INTERVALS_MHZ], dtype=bool)
    if not np.any(tail_mask):
        tail_mask = np.ones_like(est_a, dtype=bool)
    tail_abs_est = float(np.mean(np.abs(est_a[tail_mask])))
    tail_abs_true = float(np.mean(np.abs(true_a[tail_mask])))
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "profile_corr": _safe_profile_corr(est_a, true_a),
        "tail_abs_gap": float(tail_abs_est - tail_abs_true),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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

        out.append(
            {
                "candidate": cand,
                "n": len(items),
                "n_ok": len(ok),
                "eor_corr_score_mean": _mean("eor_corr_score"),
                "fg_eor_corr_abs_mean_mean": _mean("fg_eor_corr_abs_mean"),
                "eor_lag_rmse_mean": _mean("eor_lag_rmse"),
                "eor_lag_profile_corr_mean": _mean("eor_lag_profile_corr"),
                "runtime_sec_mean": _mean("runtime_sec"),
            }
        )

    def _sort_key(r: Dict[str, object]) -> Tuple[float, float, float]:
        lag_rmse = float(r.get("eor_lag_rmse_mean", float("nan")))
        lag_k = lag_rmse if math.isfinite(lag_rmse) else float("inf")
        score = float(r.get("eor_corr_score_mean", float("nan")))
        score_k = -score if math.isfinite(score) else float("inf")
        mix = float(r.get("fg_eor_corr_abs_mean_mean", float("nan")))
        mix_k = mix if math.isfinite(mix) else float("inf")
        return (lag_k, score_k, mix_k)

    out.sort(key=_sort_key)
    return out


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Optimizer Upgrade Ablation Summary\n\n")
    lines.append("## Fixed Baseline\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("## Ranked Candidates\n\n")
    lines.append("| rank | candidate | n_ok/n | lag_rmse | eor_score | fg_eor_abs | runtime_sec |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
    for idx, row in enumerate(ranked, start=1):
        def _fmt(v: object) -> str:
            try:
                fv = float(v)
            except Exception:
                return "nan"
            return f"{fv:.6f}" if math.isfinite(fv) else "nan"

        lines.append(
            f"| {idx} | {row.get('candidate')} | {row.get('n_ok')}/{row.get('n')} | "
            f"{_fmt(row.get('eor_lag_rmse_mean'))} | "
            f"{_fmt(row.get('eor_corr_score_mean'))} | "
            f"{_fmt(row.get('fg_eor_corr_abs_mean_mean'))} | "
            f"{_fmt(row.get('runtime_sec_mean'))} |\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _command_for_config(args: argparse.Namespace, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    return [str(args.python_bin), str(cli_path), "--config", str(config_path)]


def _build_config(
    *,
    dataset: DatasetSpec,
    candidate: CandidateSpec,
    run_dir: Path,
    gpu_index: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    extra_terms: List[str] = ["corr", "lagcorr", "eor_mean", "eor_hf"]
    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "lr_fg_factor": float(candidate.lr_fg_factor),
            "lr_plateau_reset_on_phase_start": bool(candidate.lr_plateau_reset_on_phase_start),
            "alt_update_mode": str(candidate.alt_update_mode),
            "alt_fg_steps": int(candidate.alt_fg_steps),
            "alt_eor_steps": int(candidate.alt_eor_steps),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": list(extra_terms),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "optimizer_name": str(args.optimizer_name),
            "momentum": float(args.momentum),
            "lr_scheduler": str(args.lr_scheduler),
            "lr_plateau_patience": int(args.lr_plateau_patience),
            "lr_plateau_factor": float(args.lr_plateau_factor),
            "lr_plateau_min_delta": float(args.lr_plateau_min_delta),
            "lr_plateau_cooldown": int(args.lr_plateau_cooldown),
            "lr_min": float(args.lr_min),
            "freq_start_mhz": float(dataset.freq_start_mhz),
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
            "corr_weight": float(args.corr_weight),
            "lagcorr_weight": float(args.lagcorr_weight),
            "eor_mean_weight": float(args.eor_mean_weight),
            "eor_hf_weight": float(args.eor_hf_weight),
            "fft_weight": 0.0,
            "poly_weight": 0.0,
            "lagcorr_fg_component_weight": 0.0,
            "lagcorr_eor_component_weight": 1.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "eor_amp_prior_mode": str(args.base_eor_amp_prior_mode),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            "corr_prior_mean": float(args.corr_prior_mean),
            "corr_prior_sigma": float(args.corr_prior_sigma),
            "corr_prior_abs_threshold": float(args.corr_abs_threshold),
            "corr_reduce": str(args.corr_reduce),
            "corr_topk": int(args.corr_topk),
            "corr_lse_alpha": float(args.corr_lse_alpha),
            "corr_feature": str(args.corr_feature),
            "corr_spatial_pool": int(args.corr_spatial_pool),
            # lagcorr envelope params
            "lagcorr_feature": "raw",
            "lagcorr_unit": "mhz",
            "lagcorr_pair_sampling": "random",
            "lagcorr_random_seed": 20260213,
            "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
            "lagcorr_max_pairs": 256,
            "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
            "lagcorr_rms_min": float(args.lagcorr_rms_min),
            "lagcorr_eor_mode": "envelope_v2",
            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
            "lagcorr_eor_subterm_schedule": str(args.lagcorr_eor_subterm_schedule),
            "lagcorr_eor_tail_weight_mode": str(args.lagcorr_eor_tail_weight_mode),
            "lagcorr_eor_tail_eps": float(args.lagcorr_eor_tail_eps),
            "lagcorr_eor_neg_delta": float(args.lagcorr_eor_neg_delta),
            "lagcorr_eor_near_floor_mode": str(args.lagcorr_eor_near_floor_mode),
            "lagcorr_eor_near_rho1_coeffs": [float(x) for x in _parse_float_list(args.lagcorr_eor_near_rho1_coeffs)],
            "lagcorr_eor_rebound_eps_act": float(args.lagcorr_eor_rebound_eps_act),
            "lagcorr_eor_rebound_delta_up": float(args.lagcorr_eor_rebound_delta_up),
            "lagcorr_eor_near_max_lag": int(args.lagcorr_eor_near_max_lag),
            "lagcorr_eor_mid_max_lag": int(args.lagcorr_eor_mid_max_lag),
            "lagcorr_eor_far_min_lag": int(args.lagcorr_eor_far_min_lag),
            # eor_hf prior params
            "eor_hf_percent": float(args.eor_hf_percent),
            "eor_hf_r_max": float(args.eor_hf_r_max),
        },
        "scan_meta": {
            "candidate_name": str(candidate.name),
        },
    }
    return cfg


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()
    if args.code_dir:
        code_dir = args.code_dir.resolve()
    else:
        code_dir = (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"optimizer_upgrade_ablation_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir, cube12_start_mhz=float(args.freq_start_mhz))
    enabled = {x.strip() for x in str(args.datasets).split(",") if x.strip()}
    datasets = [d for d in datasets_all if d.name in enabled]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")

    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

    candidates = generate_candidates(args)
    if str(args.candidate_names).strip():
        allow = {x.strip() for x in str(args.candidate_names).split(",") if x.strip()}
        unknown = sorted(allow - {c.name for c in candidates})
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]

    cut = _extract_cut_indices(_load_cube_cut(datasets[0].input_cube, cut=None).shape, float(args.cut_size_frac))
    ds_cache: Dict[str, Dict[str, object]] = {}
    for ds in datasets:
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
            "loss_stack": "base + corr(diff1) + lagcorr(env_v2+) + eor_mean + eor_hf",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] candidates={len(candidates)} output_dir={output_dir}")
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
            active_gpus = {int(j.gpu_index) for (_, j, _, _) in active}
            while queued and len(active) < max_jobs:
                pick_idx: Optional[int] = None
                for i, cand_job in enumerate(queued):
                    if int(cand_job.gpu_index) in active_gpus:
                        continue
                    pick_idx = i
                    break
                if pick_idx is None:
                    break
                job = queued.pop(pick_idx)
                job.run_dir.mkdir(parents=True, exist_ok=True)
                cfg = _build_config(dataset=job.dataset, candidate=job.candidate, run_dir=job.run_dir, gpu_index=job.gpu_index, args=args)
                job.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                import subprocess

                proc = subprocess.Popen(cmd, cwd=str(code_dir), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active.append((proc, job, time.time(), log_handle))
                active_gpus.add(int(job.gpu_index))
                print(f"  [launch] {job.dataset.name} gpu={job.gpu_index} pid={proc.pid}")

            still_active: List[Tuple[object, JobSpec, float, object]] = []
            for proc, job, t0, log_handle in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job, t0, log_handle))
                    continue
                log_handle.close()
                runtime = time.time() - t0
                row: Dict[str, object] = {
                    "candidate": job.candidate.name,
                    "dataset": job.dataset.name,
                    "status": "ok" if int(ret) == 0 and job.eor_output.exists() and job.fg_output.exists() else "failed",
                    "return_code": int(ret),
                    "runtime_sec": float(runtime),
                    "lr_fg_factor": float(job.candidate.lr_fg_factor),
                    "lr_plateau_reset_on_phase_start": int(bool(job.candidate.lr_plateau_reset_on_phase_start)),
                    "alt_update_mode": str(job.candidate.alt_update_mode),
                    "alt_fg_steps": int(job.candidate.alt_fg_steps),
                    "alt_eor_steps": int(job.candidate.alt_eor_steps),
                    "config_path": str(job.config_path),
                    "log_path": str(job.log_path),
                    "fg_output": str(job.fg_output),
                    "eor_output": str(job.eor_output),
                }
                if row["status"] != "ok":
                    rows.append(row)
                    continue

                cache = ds_cache[job.dataset.name]
                true_eor = cache["true_eor"]
                true_fg = cache["true_fg"]
                assert isinstance(true_eor, np.ndarray) and isinstance(true_fg, np.ndarray)

                with fits.open(job.eor_output, memmap=True) as hdul:
                    eor_est = np.asarray(hdul[0].data, dtype=np.float32)
                with fits.open(job.fg_output, memmap=True) as hdul:
                    fg_est = np.asarray(hdul[0].data, dtype=np.float32)

                eor_corr = _frequency_correlations(eor_est, true_eor)
                fg_eor_corr = _frequency_correlations(fg_est, eor_est)
                row["eor_corr_score"] = float(_score_from_corr(eor_corr))
                row["eor_corr_mean"] = float(_summarize_corr_stats(eor_corr)["mean"])
                row["fg_eor_corr_abs_mean"] = float(_summarize_corr_stats(fg_eor_corr)["abs_mean"])

                lag_channels = [
                    max(1, int(round(float(v) / float(args.freq_delta_mhz)))) for v in LAG_INTERVALS_MHZ
                ]
                est_prof = _compute_lagcorr_profile(eor_est, lag_channels, max_pairs=256)
                true_prof = _compute_lagcorr_profile(true_eor, lag_channels, max_pairs=256)
                (job.run_dir / "eor_lag_profile_est_raw.csv").write_text(
                    "lag_mhz,lag_chan,rho\n"
                    + "\n".join(
                        f"{LAG_INTERVALS_MHZ[i]},{lag_channels[i]},{est_prof[i]}"
                        for i in range(len(lag_channels))
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (job.run_dir / "eor_lag_profile_true_raw.csv").write_text(
                    "lag_mhz,lag_chan,rho\n"
                    + "\n".join(
                        f"{LAG_INTERVALS_MHZ[i]},{lag_channels[i]},{true_prof[i]}"
                        for i in range(len(lag_channels))
                    )
                    + "\n",
                    encoding="utf-8",
                )
                lag_m = _lag_profile_metrics(est_prof, true_prof)
                row["eor_lag_rmse"] = lag_m["rmse"]
                row["eor_lag_profile_corr"] = lag_m["profile_corr"]
                row["eor_lag_tail_abs_gap"] = lag_m["tail_abs_gap"]

                rows.append(row)
                print(
                    f"  [done] {job.dataset.name} status=ok "
                    f"eor_score={row.get('eor_corr_score')} lag_rmse={row.get('eor_lag_rmse')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "optimizer_upgrade_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows)
    rank_csv = output_dir / "optimizer_upgrade_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "optimizer_upgrade_summary.md", ranked, manifest["baseline_fixed"])
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

