#!/usr/bin/env python3
"""
Run fixed loss-mode fits (base / base+lagcorr / base+corr+lagcorr) for many datasets.

This is intended for "multi-band" evaluation: different frequency-start cubes generated
from e2esim, plus the baseline cube1/cube2 at 106 MHz.

Training uses only physically-motivated priors. Injected truth cubes are used ONLY for
evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


LAG_INTERVALS_MHZ: List[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path
    freq_start_mhz: float
    freq_delta_mhz: float


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


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
        return {
            "rmse": None,
            "profile_corr": None,
            "tail_abs_gap": None,
            "tail_abs_est": None,
            "tail_abs_true": None,
        }
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
        "tail_abs_est": tail_abs_est,
        "tail_abs_true": tail_abs_true,
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


def load_datasets(path: Path) -> List[DatasetSpec]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out: List[DatasetSpec] = []
    for it in items:
        out.append(
            DatasetSpec(
                name=str(it["name"]),
                input_cube=Path(it["input_cube"]),
                fg_true_cube=Path(it["fg_true_cube"]),
                eor_true_cube=Path(it["eor_true_cube"]),
                freq_start_mhz=float(it.get("freq_start_mhz", 106.0)),
                freq_delta_mhz=float(it.get("freq_delta_mhz", 0.1)),
            )
        )
    return out


def _command_for_config(python_bin: str, code_dir: Path, config_path: Path) -> List[str]:
    cli_path = code_dir / "separation_cli.py"
    return [str(python_bin), str(cli_path), "--config", str(config_path)]


def build_config(
    *,
    dataset: DatasetSpec,
    run_dir: Path,
    gpu_index: int,
    mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    # Fixed base.
    extra_terms: List[str] = []
    corr_enabled = False
    lag_enabled = False

    if mode == "base":
        pass
    elif mode == "base+lagcorr":
        lag_enabled = True
        extra_terms.append("lagcorr")
    elif mode == "base+corr+lagcorr":
        corr_enabled = True
        lag_enabled = True
        extra_terms.extend(["corr", "lagcorr"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
            "freq_delta_mhz": float(dataset.freq_delta_mhz),
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
            "corr_weight": float(args.corr_weight) if corr_enabled else 0.0,
            "lagcorr_weight": float(args.lagcorr_weight) if lag_enabled else 0.0,
            "lagcorr_fg_component_weight": 0.0,
            "lagcorr_eor_component_weight": 1.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            # Base priors.
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            # Corr (v2 hinge).
            "corr_prior_mean": 0.0,
            "corr_prior_sigma": float(args.corr_prior_sigma),
            "corr_prior_abs_threshold": float(args.corr_abs_threshold),
            "corr_reduce": str(args.corr_reduce),
            "corr_topk": int(args.corr_topk),
            "corr_lse_alpha": float(args.corr_lse_alpha),
            # Lagcorr envelope v2 (EoR component only).
            "lagcorr_feature": "raw",
            "lagcorr_unit": "mhz",
            "lagcorr_pair_sampling": "random",
            "lagcorr_random_seed": 20260213,
            "lagcorr_intervals": list(LAG_INTERVALS_MHZ),
            "lagcorr_max_pairs": int(args.lagcorr_max_pairs),
            "lagcorr_spatial_pool": int(args.lagcorr_spatial_pool),
            "lagcorr_eor_mode": "envelope_v2",
            "lagcorr_eor_start_iter": int(args.lagcorr_eor_start_iter),
            "lagcorr_eor_ramp_iters": int(args.lagcorr_eor_ramp_iters),
            "lagcorr_eor_tail_eps": float(args.lagcorr_eor_tail_eps),
            "lagcorr_eor_neg_delta": float(args.lagcorr_eor_neg_delta),
            "lagcorr_eor_near_rho_min": float(args.lagcorr_eor_near_rho_min),
            "lagcorr_eor_rebound_eps_act": float(args.lagcorr_eor_rebound_eps_act),
            "lagcorr_eor_rebound_delta_up": float(args.lagcorr_eor_rebound_delta_up),
            "lagcorr_eor_w_tail": float(args.lagcorr_eor_w_tail),
            "lagcorr_eor_w_neg": float(args.lagcorr_eor_w_neg),
            "lagcorr_eor_w_near": float(args.lagcorr_eor_w_near),
            "lagcorr_eor_w_rebound": float(args.lagcorr_eor_w_rebound),
            "lagcorr_eor_near_max_lag": int(args.lagcorr_eor_near_max_lag),
            "lagcorr_eor_mid_max_lag": int(args.lagcorr_eor_mid_max_lag),
            "lagcorr_eor_far_min_lag": int(args.lagcorr_eor_far_min_lag),
        },
        "evaluation": {
            "true_eor_cube": str(dataset.eor_true_cube),
            "diagnose_input": False,
            "enable_corr_check": False,
        },
    }
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-dataset fixed-mode evaluation runner.")
    p.add_argument("--work-root", type=Path, default=Path.cwd())
    p.add_argument("--code-dir", type=Path, default=None)
    p.add_argument("--datasets-json", type=Path, required=True)
    p.add_argument("--dataset-names", type=str, default="", help="Optional comma list to run a subset.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--modes", type=str, default="base,base+lagcorr,base+corr+lagcorr")

    # Shared training settings.
    p.add_argument("--num-iters", type=int, default=2500)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--data-error", type=float, default=0.005)

    # Optimizer knobs.
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-patience", type=int, default=240)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-cooldown", type=int, default=80)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # Base priors.
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2", choices=["diff3_l2", "diff2_l2", "diff2_huber", "diff1_l1"])
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Corr v2.
    p.add_argument("--corr-weight", type=float, default=0.2)
    p.add_argument("--corr-prior-sigma", type=float, default=0.2)
    p.add_argument("--corr-abs-threshold", type=float, default=0.08)
    p.add_argument("--corr-reduce", type=str, default="logsumexp", choices=["mean", "topk", "logsumexp"])
    p.add_argument("--corr-topk", type=int, default=8)
    p.add_argument("--corr-lse-alpha", type=float, default=10.0)

    # Lagcorr envelope v2.
    p.add_argument("--lagcorr-weight", type=float, default=1.0)
    p.add_argument("--lagcorr-spatial-pool", type=int, default=4)
    p.add_argument("--lagcorr-max-pairs", type=int, default=256)
    p.add_argument("--lagcorr-eor-start-iter", type=int, default=1200)
    p.add_argument("--lagcorr-eor-ramp-iters", type=int, default=800)
    p.add_argument("--lagcorr-eor-tail-eps", type=float, default=0.05)
    p.add_argument("--lagcorr-eor-neg-delta", type=float, default=0.0)
    p.add_argument("--lagcorr-eor-near-rho-min", type=float, default=0.05)
    p.add_argument("--lagcorr-eor-rebound-eps-act", type=float, default=0.05)
    p.add_argument("--lagcorr-eor-rebound-delta-up", type=float, default=0.02)
    p.add_argument("--lagcorr-eor-w-tail", type=float, default=1.0)
    p.add_argument("--lagcorr-eor-w-neg", type=float, default=1.0)
    p.add_argument("--lagcorr-eor-w-near", type=float, default=1.0)
    p.add_argument("--lagcorr-eor-w-rebound", type=float, default=1.0)
    p.add_argument("--lagcorr-eor-near-max-lag", type=int, default=10)
    p.add_argument("--lagcorr-eor-mid-max-lag", type=int, default=50)
    p.add_argument("--lagcorr-eor-far-min-lag", type=int, default=70)

    # Schedule for corr/extra terms.
    p.add_argument("--extra-loss-start-iter", type=int, default=300)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=700)

    p.add_argument("--python-bin", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()

    if args.code_dir:
        code_dir = args.code_dir.resolve()
    else:
        code_dir = (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    python_bin = args.python_bin or str(Path(sys.executable))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"multiband_eval_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args.datasets_json.resolve())
    if args.dataset_names.strip():
        allow = set(_parse_csv_tokens(args.dataset_names))
        datasets = [d for d in datasets if d.name in allow]
    if not datasets:
        raise ValueError("No datasets selected.")

    modes = _parse_csv_tokens(args.modes)
    if not modes:
        raise ValueError("Empty --modes.")

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        with fits.open(ds.input_cube, memmap=True) as h:
            in_shape = tuple(int(v) for v in h[0].data.shape)
        cut = _extract_cut_indices(in_shape, float(args.cut_size_frac))
        true_eor = _load_cube_cut(ds.eor_true_cube, cut=cut)
        true_fg = _load_cube_cut(ds.fg_true_cube, cut=cut)

        for mode in modes:
            run_dir = output_dir / ds.name / mode.replace("+", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_config(dataset=ds, run_dir=run_dir, gpu_index=int(args.gpu_index), mode=mode, args=args)
            cfg_path = run_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            cmd = _command_for_config(python_bin, code_dir, cfg_path)
            t0 = time.time()
            if args.dry_run:
                rc = 0
            else:
                rc = int(subprocess.call(cmd, cwd=str(work_root)))
            runtime = float(time.time() - t0)

            row: Dict[str, object] = {
                "dataset": ds.name,
                "mode": mode,
                "status": "ok" if rc == 0 else "failed",
                "return_code": int(rc),
                "runtime_sec": runtime,
                "gpu_index": int(args.gpu_index),
                "input_cube": str(ds.input_cube),
                "fg_true_cube": str(ds.fg_true_cube),
                "eor_true_cube": str(ds.eor_true_cube),
                "fg_output": str(run_dir / "fg_est.fits"),
                "eor_output": str(run_dir / "eor_est.fits"),
                "freq_start_mhz": float(ds.freq_start_mhz),
                "freq_delta_mhz": float(ds.freq_delta_mhz),
            }
            if rc != 0:
                rows.append(row)
                continue

            with fits.open(run_dir / "eor_est.fits", memmap=True) as h:
                eor_est = np.asarray(h[0].data, dtype=np.float32)
            with fits.open(run_dir / "fg_est.fits", memmap=True) as h:
                fg_est = np.asarray(h[0].data, dtype=np.float32)

            eor_corr = _frequency_correlations(eor_est, true_eor)
            fg_eor_corr = _frequency_correlations(fg_est, eor_est)
            inj_fg_eor_corr = _frequency_correlations(true_fg, true_eor)

            eor_stats = _summarize_corr_stats(eor_corr)
            fg_eor_stats = _summarize_corr_stats(fg_eor_corr)
            inj_stats = _summarize_corr_stats(inj_fg_eor_corr)

            row.update(
                {
                    "eor_corr_mean": eor_stats["mean"],
                    "eor_corr_p10": eor_stats["p10"],
                    "eor_corr_min": eor_stats["min"],
                    "eor_corr_max": eor_stats["max"],
                    "fg_eor_corr_abs_mean": fg_eor_stats["abs_mean"],
                    "fg_eor_corr_mean": fg_eor_stats["mean"],
                    "inj_fg_eor_corr_abs_mean": inj_stats["abs_mean"],
                    "inj_fg_eor_corr_mean": inj_stats["mean"],
                }
            )

            lag_channels = [int(round(float(x) / float(ds.freq_delta_mhz))) for x in LAG_INTERVALS_MHZ]
            est_prof = _compute_lagcorr_profile(eor_est, lag_channels, max_pairs=None)
            true_prof = _compute_lagcorr_profile(true_eor, lag_channels, max_pairs=None)
            lm = _lag_profile_metrics(est_prof, true_prof, tail_threshold_mhz=2.0)
            row.update(
                {
                    "eor_lag_rmse": lm["rmse"],
                    "eor_lag_profile_corr": lm["profile_corr"],
                    "eor_lag_tail_abs_gap": lm["tail_abs_gap"],
                    "eor_lag_tail_abs_est": lm["tail_abs_est"],
                    "eor_lag_tail_abs_true": lm["tail_abs_true"],
                }
            )

            # Save profiles for later plotting.
            (run_dir / "eor_corr_profile.json").write_text(
                json.dumps({"corr": eor_corr.tolist()}, indent=2), encoding="utf-8"
            )
            (run_dir / "eor_lag_profile_est.json").write_text(
                json.dumps({"lag_mhz": LAG_INTERVALS_MHZ, "rho": list(map(float, est_prof))}, indent=2),
                encoding="utf-8",
            )
            (run_dir / "eor_lag_profile_true.json").write_text(
                json.dumps({"lag_mhz": LAG_INTERVALS_MHZ, "rho": list(map(float, true_prof))}, indent=2),
                encoding="utf-8",
            )

            rows.append(row)

    _write_csv(output_dir / "mode_sweep_results.csv", rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "code_dir": str(code_dir),
                "output_dir": str(output_dir),
                "datasets_json": str(args.datasets_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
