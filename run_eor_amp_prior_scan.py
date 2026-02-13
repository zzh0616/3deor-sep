#!/usr/bin/env python3
"""
EoR amplitude-prior mode scan on top of a fixed base baseline.

We compare the base EoR amplitude prior implementations:
- voxel_deadzone: legacy voxel-wise dead-zone hinge
- slice_rms_hinge: per-frequency slice RMS (std_xy) upper bound hinge
- hybrid: slice RMS hinge + very loose voxel outlier guard

Training loss never uses injected truth; injected truth is used only for evaluation metrics.
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
    eor_amp_prior_mode: str
    eor_hybrid_voxel_factor: float
    eor_hybrid_voxel_weight: float

    def key(self) -> Tuple[object, ...]:
        return (
            str(self.eor_amp_prior_mode),
            round(float(self.eor_hybrid_voxel_factor), 8),
            round(float(self.eor_hybrid_voxel_weight), 8),
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
    parser = argparse.ArgumentParser(description="Run EoR amplitude-prior mode scan.")
    parser.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--code-dir", type=Path, default=None, help="3dnet dir.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default <work-root>/runs/eor_amp_scan_<timestamp>).")
    parser.add_argument("--datasets", type=str, default="cube1,cube2", help=f"Comma-separated datasets. Common: {default_dataset_name_hint()}")
    parser.add_argument("--exclude-from-ranking", type=str, default="cube1")
    parser.add_argument("--gpu-map", type=str, default="cube1:0,cube2:1", help="Dataset->GPU mapping, e.g. cube1:0,cube2:1")
    parser.add_argument("--max-concurrent-jobs", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=2500)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--cut-size-frac", type=float, default=0.30)
    parser.add_argument("--freq-start-mhz", type=float, default=106.0, help="Cube1/2 starting frequency in MHz.")
    parser.add_argument("--freq-delta-mhz", type=float, default=0.1)
    parser.add_argument("--data-error", type=float, default=0.005)

    # Fixed base baseline.
    parser.add_argument("--base-beta", type=float, default=0.5)
    parser.add_argument("--base-gamma", type=float, default=0.6)
    parser.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    parser.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    parser.add_argument(
        "--base-fg-smooth-mode",
        type=str,
        default="diff2_l2",
        choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"],
    )
    parser.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    parser.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    parser.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Optimizer knobs.
    parser.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    parser.add_argument("--lr-plateau-patience", type=int, default=240)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-plateau-cooldown", type=int, default=80)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    # Scan grid.
    parser.add_argument(
        "--eor-amp-prior-modes",
        type=str,
        default="voxel_deadzone,slice_rms_hinge,hybrid",
        help="Comma-separated: voxel_deadzone,slice_rms_hinge,hybrid",
    )
    parser.add_argument("--hybrid-voxel-factor", type=float, default=5.0)
    parser.add_argument("--hybrid-voxel-weight", type=float, default=0.1)
    parser.add_argument("--candidate-names", type=str, default="")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


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


def generate_candidates(args: argparse.Namespace) -> List[CandidateSpec]:
    modes = [m.strip() for m in str(args.eor_amp_prior_modes).split(",") if m.strip()]
    if not modes:
        raise ValueError("eor-amp-prior-modes must be non-empty.")

    out: List[CandidateSpec] = []
    for mode in modes:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"voxel_deadzone", "slice_rms_hinge", "hybrid"}:
            raise ValueError(f"Unknown eor_amp_prior_mode: {mode}")
        name = f"amp_{mode_norm}"
        if mode_norm == "hybrid":
            name = f"{name}_vf{_fmt_float_token(float(args.hybrid_voxel_factor))}_vw{_fmt_float_token(float(args.hybrid_voxel_weight))}"
        out.append(
            CandidateSpec(
                name=name,
                eor_amp_prior_mode=mode_norm,
                eor_hybrid_voxel_factor=float(args.hybrid_voxel_factor),
                eor_hybrid_voxel_weight=float(args.hybrid_voxel_weight),
            )
        )
    # Dedup by key preserving order.
    dedup: List[CandidateSpec] = []
    seen = set()
    for c in out:
        if c.key() in seen:
            continue
        seen.add(c.key())
        dedup.append(c)
    return dedup


def _fmt_float_token(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    s = f"{float(value):.6g}"
    if "e" in s or "E" in s:
        s = f"{float(value):.12f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


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


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    p10 = float(np.percentile(finite, 10))
    return 0.7 * mean + 0.2 * median + 0.1 * p10


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
        return {"mae": None, "rmse": None, "profile_corr": None, "tail_abs_gap": None}
    diff = est_a - true_a
    tail_mask = np.asarray([float(v) >= float(tail_threshold_mhz) for v in LAG_INTERVALS_MHZ], dtype=bool)
    if not np.any(tail_mask):
        tail_mask = np.ones_like(est_a, dtype=bool)
    tail_abs_est = float(np.mean(np.abs(est_a[tail_mask])))
    tail_abs_true = float(np.mean(np.abs(true_a[tail_mask])))
    return {
        "mae": float(np.mean(np.abs(diff))),
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


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# EoR Amplitude Prior Mode Scan Summary\n\n")
    lines.append("## Fixed Baseline\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("## Ranked Candidates\n\n")
    lines.append("| rank | candidate | n_ok/n | eor_score | eor_corr_mean | fg_eor_abs_mean | eor_lag_rmse | eor_lag_profile_corr |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for idx, row in enumerate(ranked, start=1):
        def _fmt(v: object) -> str:
            try:
                fv = float(v)
            except Exception:
                return "nan"
            return f"{fv:.6f}" if math.isfinite(fv) else "nan"

        lines.append(
            f"| {idx} | {row.get('candidate')} | {row.get('n_ok')}/{row.get('n')} | "
            f"{_fmt(row.get('eor_corr_score_mean'))} | "
            f"{_fmt(row.get('eor_corr_mean_mean'))} | "
            f"{_fmt(row.get('fg_eor_corr_abs_mean_mean'))} | "
            f"{_fmt(row.get('eor_lag_rmse_mean'))} | "
            f"{_fmt(row.get('eor_lag_profile_corr_mean'))} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _candidate_summary(rows: Sequence[Dict[str, object]], *, exclude_datasets: Sequence[str] = ()) -> List[Dict[str, object]]:
    exclude = {str(x).strip() for x in exclude_datasets if str(x).strip()}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate")), []).append(row)
    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        ranked_items = [r for r in items if str(r.get("dataset")) not in exclude]
        ok_rank = [r for r in ranked_items if str(r.get("status")) == "ok"]

        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in ok_rank if r.get(key) is not None and math.isfinite(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        out.append(
            {
                "candidate": cand,
                "n": len(ranked_items),
                "n_ok": len(ok_rank),
                "eor_corr_score_mean": _mean("eor_corr_score"),
                "eor_corr_mean_mean": _mean("eor_corr_mean"),
                "fg_eor_corr_abs_mean_mean": _mean("fg_eor_corr_abs_mean"),
                "eor_lag_rmse_mean": _mean("eor_lag_rmse"),
                "eor_lag_profile_corr_mean": _mean("eor_lag_profile_corr"),
            }
        )

    def _sort_key(r: Dict[str, object]) -> Tuple[float, float, float]:
        score = float(r.get("eor_corr_score_mean", float("nan")))
        score_k = -score if math.isfinite(score) else float("inf")
        mix = float(r.get("fg_eor_corr_abs_mean_mean", float("nan")))
        mix_k = mix if math.isfinite(mix) else float("inf")
        lag = float(r.get("eor_lag_rmse_mean", float("nan")))
        lag_k = lag if math.isfinite(lag) else float("inf")
        return (score_k, mix_k, lag_k)

    out.sort(key=_sort_key)
    return out


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
            "extra_loss_terms": [],
            "extra_loss_start_iter": 0,
            "extra_loss_ramp_iters": 0,
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
            "corr_weight": 0.0,
            "lagcorr_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "eor_amp_prior_mode": str(candidate.eor_amp_prior_mode),
            "eor_hybrid_voxel_factor": float(candidate.eor_hybrid_voxel_factor),
            "eor_hybrid_voxel_weight": float(candidate.eor_hybrid_voxel_weight),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
        },
        "scan_meta": {
            "candidate_name": candidate.name,
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
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"eor_amp_scan_{stamp}")
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
    if args.candidate_names.strip():
        allow = {x.strip() for x in args.candidate_names.split(",") if x.strip()}
        unknown = sorted(allow - {c.name for c in candidates})
        if unknown:
            raise ValueError(f"Unknown candidate names: {unknown}")
        candidates = [c for c in candidates if c.name in allow]
        if not candidates:
            raise ValueError("No candidates selected after --candidate-names filter.")

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

    baseline_fixed = {
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
        "exclude_from_ranking": str(args.exclude_from_ranking),
    }

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "datasets": [{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(d).items()} for d in datasets],
        "candidates": [asdict(c) for c in candidates],
        "baseline_fixed": baseline_fixed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] candidates={len(candidates)} output_dir={output_dir}")
        return 0

    rows: List[Dict[str, object]] = []
    max_jobs = max(1, int(args.max_concurrent_jobs))
    lag_channels = [max(1, int(round(float(v) / float(args.freq_delta_mhz)))) for v in LAG_INTERVALS_MHZ]
    exclude_rank = _parse_csv_tokens(args.exclude_from_ranking)

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
                import subprocess  # local import

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
                    "candidate": cand.name,
                    "dataset": job.dataset.name,
                    "freq_start_mhz": float(job.dataset.freq_start_mhz),
                    "status": "ok" if int(ret) == 0 else "failed",
                    "return_code": int(ret),
                    "runtime_sec": float(runtime),
                    "eor_amp_prior_mode": str(cand.eor_amp_prior_mode),
                    "eor_hybrid_voxel_factor": float(cand.eor_hybrid_voxel_factor),
                    "eor_hybrid_voxel_weight": float(cand.eor_hybrid_voxel_weight),
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
                    "config_path": str(job.config_path),
                    "log_path": str(job.log_path),
                    "fg_output": str(job.fg_output),
                    "eor_output": str(job.eor_output),
                }

                if int(ret) != 0 or not job.eor_output.exists() or not job.fg_output.exists():
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
                eor_stats = _summarize_corr_stats(eor_corr)
                fg_eor_stats = _summarize_corr_stats(fg_eor_corr)
                row["eor_corr_mean"] = eor_stats["mean"]
                row["eor_corr_p10"] = eor_stats["p10"]
                row["eor_corr_score"] = float(_score_from_corr(eor_corr))
                row["fg_eor_corr_abs_mean"] = fg_eor_stats["abs_mean"]

                est_prof = _compute_lagcorr_profile(eor_est, lag_channels, max_pairs=256)
                true_prof = _compute_lagcorr_profile(true_eor, lag_channels, max_pairs=256)
                (job.run_dir / "eor_lag_profile_est_raw.csv").write_text(
                    "lag_mhz,lag_chan,rho\n"
                    + "\n".join(
                        f"{LAG_INTERVALS_MHZ[i]},{lag_channels[i]},{est_prof[i]}" for i in range(len(lag_channels))
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (job.run_dir / "eor_lag_profile_true_raw.csv").write_text(
                    "lag_mhz,lag_chan,rho\n"
                    + "\n".join(
                        f"{LAG_INTERVALS_MHZ[i]},{lag_channels[i]},{true_prof[i]}" for i in range(len(lag_channels))
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
                    f"  [done] {job.dataset.name} status={row['status']} "
                    f"eor_score={row.get('eor_corr_score')} mix={row.get('fg_eor_corr_abs_mean')} lag_rmse={row.get('eor_lag_rmse')}"
                )

            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "eor_amp_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows, exclude_datasets=exclude_rank)
    rank_csv = output_dir / "eor_amp_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "eor_amp_summary.md", ranked, baseline_fixed)
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

