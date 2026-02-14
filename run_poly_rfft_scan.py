#!/usr/bin/env python3
"""
Scan poly_reparam and rfft extra loss terms on top of a fixed base baseline.

We evaluate candidates using metrics computed from outputs (not loss values):
- primary: per-frequency corr(EoR_est[f], EoR_true[f]) across spatial pixels
- secondary: EoR-window 2D power-spectrum agreement metrics (from powerspec outputs)

The intent is to test whether these extra terms improve separation under the same
physics-driven base prior setup, while avoiding analysis of obviously unconverged runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from dataset_registry import DatasetSpec, build_datasets, default_dataset_name_hint


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    extra_loss_terms: Tuple[str, ...]
    weight_overrides: Dict[str, float]
    optim_overrides: Dict[str, object]
    prior_overrides: Dict[str, object]


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


_ITER_RE = re.compile(r"^\\[iter\\s+(\\d+)\\]\\s+total=([0-9eE+\\-.]+)")
_CHECK_RE = re.compile(r"^\\[check\\]\\s+iter\\s+(\\d+):\\s+mean EoR corr=([0-9eE+\\-.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run poly_reparam + rfft scans on injected cubes.")
    p.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root.")
    p.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help="3dnet dir (default <work-root>/code/3dnet or <work-root>/3dnet).",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Data dir (default <work-root>/data).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default <work-root>/runs/poly_rfft_scan_<timestamp>).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="cube1,cube2",
        help=f"Comma-separated datasets. Common: {default_dataset_name_hint()}",
    )
    p.add_argument(
        "--exclude-from-ranking",
        type=str,
        default="cube1",
        help="Comma-separated datasets excluded from aggregation (still evaluated).",
    )
    p.add_argument(
        "--gpu-map",
        type=str,
        default="cube1:0,cube2:1",
        help="Dataset->GPU mapping, e.g. cube1:0,cube2:1",
    )
    p.add_argument("--max-concurrent-jobs", type=int, default=2, help="Max concurrent dataset jobs per candidate.")

    # Common run controls.
    p.add_argument("--num-iters", type=int, default=3000)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--cut-size-frac", type=float, default=0.30)
    p.add_argument("--freq-start-mhz", type=float, default=106.0)
    p.add_argument("--freq-delta-mhz", type=float, default=0.1)
    p.add_argument("--data-error", type=float, default=0.005)

    # Fixed base baseline (defaults reflect current best-known base baseline).
    p.add_argument("--base-beta", type=float, default=0.5)
    p.add_argument("--base-gamma", type=float, default=0.6)
    p.add_argument("--base-eor-prior-sigma", type=float, default=0.02)
    p.add_argument("--base-eor-amp-threshold", type=float, default=0.1)
    p.add_argument(
        "--base-eor-amp-prior-mode",
        type=str,
        default="slice_rms_hinge",
        choices=["voxel_deadzone", "slice_rms_hinge", "hybrid"],
    )
    p.add_argument("--base-eor-hybrid-voxel-factor", type=float, default=5.0)
    p.add_argument("--base-eor-hybrid-voxel-weight", type=float, default=0.1)
    p.add_argument("--base-fg-smooth-mode", type=str, default="diff2_l2")
    p.add_argument("--base-fg-smooth-mean", type=float, default=0.002)
    p.add_argument("--base-fg-smooth-sigma", type=float, default=0.004)
    p.add_argument("--base-fg-smooth-huber-delta", type=float, default=1.0)

    # Optimizer knobs.
    p.add_argument("--optimizer-name", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--lr-plateau-patience", type=int, default=240)
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-min-delta", type=float, default=1e-4)
    p.add_argument("--lr-plateau-cooldown", type=int, default=80)
    p.add_argument("--lr-min", type=float, default=1e-6)

    # Extra-term scheduling (applies to both poly and rfft penalties).
    p.add_argument("--extra-loss-start-iter", type=int, default=500)
    p.add_argument("--extra-loss-ramp-iters", type=int, default=0)

    # Power-spectrum outputs.
    p.add_argument(
        "--power-config",
        type=Path,
        default=Path("configs/power_eor_window.json"),
        help="Power-spectrum JSON config, relative to code-dir unless absolute.",
    )

    # Candidate generation toggles.
    p.add_argument("--include-base", action="store_true", help="Include base-only control.")
    p.add_argument("--include-poly", action="store_true", help="Include poly_reparam candidates.")
    p.add_argument("--include-rfft", action="store_true", help="Include rfft candidates.")
    p.add_argument("--include-combos", action="store_true", help="Include poly+rfft combination candidates.")

    # Poly grid.
    p.add_argument("--poly-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--poly-degree-list", type=str, default="2,3")
    p.add_argument("--poly-sigma-list", type=str, default="0.05,0.1")

    # rFFT grid.
    p.add_argument("--fft-weight-list", type=str, default="0.1,0.3,1.0")
    p.add_argument("--fft-sigma-list", type=str, default="0.5,1.0")
    p.add_argument("--fft-percent-list", type=str, default="0.7")
    p.add_argument("--fft-use-log-energy", action="store_true", help="Use log1p(energy) before penalty.")
    p.add_argument("--fft-z-clip", type=float, default=None, help="Optional absolute z-score clip.")

    p.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Comma-separated candidate names to run; empty means all generated candidates.",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run separation_cli.py.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv_tokens(text):
        out.append(float(token))
    return out


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for token in _parse_csv_tokens(text):
        out.append(int(float(token)))
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


def _extract_cut_indices(shape: Sequence[int], cut_frac: float) -> Optional[Tuple[int, int, int, int]]:
    if len(shape) < 3:
        return None
    f, ny, nx = int(shape[0]), int(shape[1]), int(shape[2])
    if f <= 0 or nx <= 0 or ny <= 0:
        return None
    size = int(round(float(cut_frac) * min(nx, ny)))
    size = max(1, min(size, nx, ny))
    x0 = (nx - size) // 2
    y0 = (ny - size) // 2
    return (x0, x0 + size, y0, y0 + size)


def _load_cube_cut(path: Path, cut: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if cut is not None:
            x0, x1, y0, y1 = cut
            data = data[:, y0:y1, x0:x1]
        return np.asarray(data, dtype=np.float32)


def _frequency_correlations(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    f = a.shape[0]
    out = np.full((f,), np.nan, dtype=np.float64)
    for i in range(f):
        x = a[i].reshape(-1).astype(np.float64, copy=False)
        y = b[i].reshape(-1).astype(np.float64, copy=False)
        x = x - np.mean(x)
        y = y - np.mean(y)
        denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y))) + float(eps)
        out[i] = float(np.sum(x * y) / denom)
    return out


def _summarize_corr_stats(vec: np.ndarray) -> Dict[str, float]:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "min": float("nan"), "max": float("nan"), "abs_mean": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
    }


def _score_from_corr(vec: np.ndarray) -> float:
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return float("nan")
    n = max(1, int(round(0.2 * finite.size)))
    worst = np.sort(finite)[:n]
    return float(np.mean(worst))


def _write_frequency_corr_profile(path: Path, corr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["freq_index", "corr"])
        for i, v in enumerate(corr.tolist()):
            w.writerow([i, v])


def _read_eor_window_metrics(power_dir: Path) -> Dict[str, object]:
    metrics_path = power_dir / "power2d_eor_window_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        if isinstance(metrics, dict):
            return {f"ps2d_win_{k}": v for k, v in metrics.items()}
        return {}
    except Exception:
        return {}


def _parse_convergence_from_log(log_path: Path) -> Dict[str, object]:
    """
    Best-effort convergence indicators from run.log.
    We avoid overly strict criteria; the goal is to flag obviously still-moving runs.
    """
    iters: List[int] = []
    totals: List[float] = []
    checks_i: List[int] = []
    checks_corr: List[float] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _ITER_RE.match(line.strip())
            if m:
                iters.append(int(m.group(1)))
                totals.append(float(m.group(2)))
                continue
            m = _CHECK_RE.match(line.strip())
            if m:
                checks_i.append(int(m.group(1)))
                checks_corr.append(float(m.group(2)))
    except OSError:
        return {"converged": False, "conv_error": "read_failed"}

    def _last_delta(vals: List[float], n: int) -> Optional[float]:
        if len(vals) < n + 1:
            return None
        return float(vals[-1] - vals[-(n + 1)])

    total_delta_2 = _last_delta(totals, 2)
    corr_delta_2 = _last_delta(checks_corr, 2)
    out: Dict[str, object] = {
        "conv_total_delta_last2": total_delta_2,
        "conv_eor_corr_delta_last2": corr_delta_2,
        "conv_total_last": totals[-1] if totals else None,
        "conv_iter_last": iters[-1] if iters else None,
        "conv_check_iter_last": checks_i[-1] if checks_i else None,
        "conv_check_corr_last": checks_corr[-1] if checks_corr else None,
    }

    # A pragmatic, metric-driven convergence flag:
    # - If we have at least 3 corr checks, require the last ~2 intervals to be stable.
    corr_stable = True
    if corr_delta_2 is not None:
        corr_stable = abs(float(corr_delta_2)) < 1e-3
    total_stable = True
    if total_delta_2 is not None:
        # total loss should not be changing too much at the end; keep loose.
        total_stable = abs(float(total_delta_2)) < 5e-3
    out["converged"] = bool(corr_stable and total_stable)
    return out


def _fmt_float_token(x: float) -> str:
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def generate_candidates(
    *,
    include_base: bool,
    include_poly: bool,
    include_rfft: bool,
    include_combos: bool,
    poly_weight_list: Sequence[float],
    poly_degree_list: Sequence[int],
    poly_sigma_list: Sequence[float],
    fft_weight_list: Sequence[float],
    fft_sigma_list: Sequence[float],
    fft_percent_list: Sequence[float],
    fft_use_log_energy: bool,
    fft_z_clip: Optional[float],
) -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    if bool(include_base):
        out.append(CandidateSpec(name="base", extra_loss_terms=(), weight_overrides={}, optim_overrides={}, prior_overrides={}))

    if bool(include_poly):
        for w in poly_weight_list:
            for d in poly_degree_list:
                for sig in poly_sigma_list:
                    name = f"poly_w{_fmt_float_token(w)}_d{int(d)}_s{_fmt_float_token(sig)}"
                    out.append(
                        CandidateSpec(
                            name=name,
                            extra_loss_terms=("poly_reparam",),
                            weight_overrides={"poly_weight": float(w)},
                            optim_overrides={"poly_degree": int(d), "poly_sigma": float(sig)},
                            prior_overrides={},
                        )
                    )

    if bool(include_rfft):
        for w in fft_weight_list:
            for sig in fft_sigma_list:
                for pct in fft_percent_list:
                    name = f"rfft_w{_fmt_float_token(w)}_s{_fmt_float_token(sig)}_p{_fmt_float_token(pct)}"
                    if bool(fft_use_log_energy):
                        name += "_log"
                    out.append(
                        CandidateSpec(
                            name=name,
                            extra_loss_terms=("rfft",),
                            weight_overrides={"fft_weight": float(w)},
                            optim_overrides={},
                            prior_overrides={
                                "fft_prior_mean": 0.0,
                                "fft_prior_sigma": float(sig),
                                "fft_highfreq_percent": float(pct),
                                "fft_use_log_energy": bool(fft_use_log_energy),
                                "fft_z_clip": fft_z_clip,
                            },
                        )
                    )

    if bool(include_combos):
        for pw in poly_weight_list:
            for pd in poly_degree_list:
                for ps in poly_sigma_list:
                    for fw in fft_weight_list:
                        for fs in fft_sigma_list:
                            for fp in fft_percent_list:
                                name = (
                                    f"poly_w{_fmt_float_token(pw)}_d{int(pd)}_s{_fmt_float_token(ps)}"
                                    f"__rfft_w{_fmt_float_token(fw)}_s{_fmt_float_token(fs)}_p{_fmt_float_token(fp)}"
                                )
                                if bool(fft_use_log_energy):
                                    name += "_log"
                                out.append(
                                    CandidateSpec(
                                        name=name,
                                        extra_loss_terms=("poly_reparam", "rfft"),
                                        weight_overrides={"poly_weight": float(pw), "fft_weight": float(fw)},
                                        optim_overrides={"poly_degree": int(pd), "poly_sigma": float(ps)},
                                        prior_overrides={
                                            "fft_prior_mean": 0.0,
                                            "fft_prior_sigma": float(fs),
                                            "fft_highfreq_percent": float(fp),
                                            "fft_use_log_energy": bool(fft_use_log_energy),
                                            "fft_z_clip": fft_z_clip,
                                        },
                                    )
                                )

    # De-dup by name.
    seen = set()
    uniq: List[CandidateSpec] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(c)
    return uniq


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
    code_dir: Path,
) -> Dict[str, object]:
    power_cfg = args.power_config
    if power_cfg is None:
        raise ValueError("power_config is required to compute EoR-window metrics.")
    power_cfg_path = Path(power_cfg)
    if not power_cfg_path.is_absolute():
        power_cfg_path = (code_dir / str(power_cfg_path)).resolve()

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
            "extra_loss_terms": list(candidate.extra_loss_terms),
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
            "corr_weight": 0.0,
            "lagcorr_weight": 0.0,
            "fft_weight": 0.0,
            "poly_weight": 0.0,
            "fg_logcurv_weight": 0.0,
            "fg_lowrank_weight": 0.0,
            "eor_lagshape_weight": 0.0,
            "eor_iso_weight": 0.0,
            "eor_mean_weight": 0.0,
            "eor_hf_weight": 0.0,
        },
        "priors": {
            "data_error": float(args.data_error),
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": float(args.base_eor_prior_sigma),
            "eor_prior_amp_threshold": float(args.base_eor_amp_threshold),
            "eor_amp_prior_mode": str(args.base_eor_amp_prior_mode),
            "eor_hybrid_voxel_factor": float(args.base_eor_hybrid_voxel_factor),
            "eor_hybrid_voxel_weight": float(args.base_eor_hybrid_voxel_weight),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "fg_smooth_huber_delta": float(args.base_fg_smooth_huber_delta),
            # rfft defaults (candidate may override)
            "fft_highfreq_percent": 0.7,
            "fft_use_log_energy": bool(args.fft_use_log_energy),
            "fft_z_clip": args.fft_z_clip,
            "fft_prior_mean": 0.0,
            "fft_prior_sigma": 1.0,
        },
        "power": {
            "power_config": str(power_cfg_path),
            "power_output_dir": str(run_dir / "powerspec"),
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

    cfg["optim"].update(candidate.optim_overrides)
    cfg["weights"].update(candidate.weight_overrides)
    cfg["priors"].update(candidate.prior_overrides)
    return cfg


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
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_summary(rows: Sequence[Dict[str, object]], exclude_datasets: Sequence[str]) -> List[Dict[str, object]]:
    exclude = set(exclude_datasets)
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "ok":
            continue
        if str(row.get("dataset")) in exclude:
            continue
        if not bool(row.get("converged", True)):
            continue
        grouped.setdefault(str(row.get("candidate")), []).append(dict(row))
    out: List[Dict[str, object]] = []
    for cand, items in grouped.items():
        eor_scores = [float(x.get("eor_corr_score", float("nan"))) for x in items]
        eor_scores = [x for x in eor_scores if math.isfinite(x)]
        ps_mad = [float(x.get("ps2d_win_log10_mad", float("nan"))) for x in items]
        ps_mad = [x for x in ps_mad if math.isfinite(x)]
        out.append(
            {
                "candidate": cand,
                "n_ok_converged": int(len(items)),
                "eor_corr_score_mean": float(np.mean(eor_scores)) if eor_scores else float("nan"),
                "ps2d_win_log10_mad_mean": float(np.mean(ps_mad)) if ps_mad else float("nan"),
            }
        )
    out.sort(
        key=lambda r: (
            -(float(r["eor_corr_score_mean"]) if math.isfinite(float(r["eor_corr_score_mean"])) else -1e9),
            float(r["ps2d_win_log10_mad_mean"]) if math.isfinite(float(r["ps2d_win_log10_mad_mean"])) else 1e9,
        )
    )
    return out


def _write_markdown(path: Path, ranked: Sequence[Dict[str, object]], meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Poly + rFFT Scan Summary\n\n")
    lines.append("## Meta\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("\n```\n\n")
    lines.append("| rank | candidate | n_ok_converged | eor_corr_score_mean | ps2d_win_log10_mad_mean |\n")
    lines.append("|---:|---|---:|---:|---:|\n")
    for i, row in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {row['candidate']} | {row['n_ok_converged']} | "
            f"{float(row['eor_corr_score_mean']):.6f} | "
            f"{float(row['ps2d_win_log10_mad_mean']):.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _run_job_result_only(
    *,
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
        "freq_start_mhz": float(dataset.freq_start_mhz),
        "status": "ok" if int(return_code) == 0 else "failed",
        "return_code": int(return_code),
        "runtime_sec": float(runtime),
        "extra_loss_terms": ",".join(candidate.extra_loss_terms),
        "config_path": str(run_dir / "config.json"),
        "log_path": str(run_dir / "run.log"),
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "power_dir": str(run_dir / "powerspec"),
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
        row["status"] = "failed"
        row["note"] = f"true_shape_mismatch true={true_eor.shape} est={eor_est.shape}"
        return row

    eor_corr = _frequency_correlations(eor_est, true_eor)
    _write_frequency_corr_profile(run_dir / "eor_corr_profile.csv", eor_corr)
    eor_stats = _summarize_corr_stats(eor_corr)
    row["eor_corr_mean"] = eor_stats["mean"]
    row["eor_corr_p10"] = eor_stats["p10"]
    row["eor_corr_min"] = eor_stats["min"]
    row["eor_corr_max"] = eor_stats["max"]
    row["eor_corr_score"] = float(_score_from_corr(eor_corr))

    fg_eor_corr = _frequency_correlations(fg_est, eor_est)
    fg_eor_stats = _summarize_corr_stats(fg_eor_corr)
    row["fg_eor_corr_abs_mean"] = fg_eor_stats["abs_mean"]
    inj_fg_eor_corr = _frequency_correlations(true_fg, true_eor)
    inj_stats = _summarize_corr_stats(inj_fg_eor_corr)
    row["inj_fg_eor_corr_abs_mean"] = inj_stats["abs_mean"]

    row.update(_read_eor_window_metrics(run_dir / "powerspec"))
    row.update(_parse_convergence_from_log(run_dir / "run.log"))
    return row


def main() -> int:
    args = parse_args()
    work_root = args.work_root.resolve()

    if args.code_dir:
        code_dir = args.code_dir.resolve()
    else:
        code_dir = (work_root / "code" / "3dnet") if (work_root / "code" / "3dnet").is_dir() else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir else (work_root / "data")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"poly_rfft_scan_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_all = build_datasets(data_dir, cube12_start_mhz=float(args.freq_start_mhz))
    enabled = [t.strip() for t in str(args.datasets).split(",") if t.strip()]
    datasets = [d for d in datasets_all if d.name in set(enabled)]
    if not datasets:
        raise ValueError("No datasets enabled after --datasets filter.")
    gpu_map = parse_gpu_map(args.gpu_map)
    for ds in datasets:
        if ds.name not in gpu_map:
            raise ValueError(f"Missing GPU mapping for dataset '{ds.name}'.")

    # Default: include base, poly, rfft if nothing is explicitly set.
    if not (args.include_base or args.include_poly or args.include_rfft or args.include_combos):
        args.include_base = True
        args.include_poly = True
        args.include_rfft = True

    poly_weight_list = _parse_float_list(args.poly_weight_list)
    poly_degree_list = _parse_int_list(args.poly_degree_list)
    poly_sigma_list = _parse_float_list(args.poly_sigma_list)
    fft_weight_list = _parse_float_list(args.fft_weight_list)
    fft_sigma_list = _parse_float_list(args.fft_sigma_list)
    fft_percent_list = _parse_float_list(args.fft_percent_list)
    if (args.include_poly or args.include_combos) and (not poly_weight_list or not poly_degree_list or not poly_sigma_list):
        raise ValueError("Poly lists must be non-empty when include-poly/include-combos is enabled.")
    if (args.include_rfft or args.include_combos) and (not fft_weight_list or not fft_sigma_list or not fft_percent_list):
        raise ValueError("FFT lists must be non-empty when include-rfft/include-combos is enabled.")

    candidates = generate_candidates(
        include_base=bool(args.include_base),
        include_poly=bool(args.include_poly),
        include_rfft=bool(args.include_rfft),
        include_combos=bool(args.include_combos),
        poly_weight_list=poly_weight_list,
        poly_degree_list=poly_degree_list,
        poly_sigma_list=poly_sigma_list,
        fft_weight_list=fft_weight_list,
        fft_sigma_list=fft_sigma_list,
        fft_percent_list=fft_percent_list,
        fft_use_log_energy=bool(args.fft_use_log_energy),
        fft_z_clip=args.fft_z_clip,
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
            "eor_amp_prior_mode": str(args.base_eor_amp_prior_mode),
            "eor_hybrid_voxel_factor": float(args.base_eor_hybrid_voxel_factor),
            "eor_hybrid_voxel_weight": float(args.base_eor_hybrid_voxel_weight),
            "fg_smooth_mode": str(args.base_fg_smooth_mode),
            "fg_smooth_mean": float(args.base_fg_smooth_mean),
            "fg_smooth_sigma": float(args.base_fg_smooth_sigma),
            "data_error": float(args.data_error),
            "optimizer_name": str(args.optimizer_name),
            "lr": float(args.lr),
            "lr_scheduler": str(args.lr_scheduler),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "power_config": str(args.power_config),
            "exclude_from_ranking": str(args.exclude_from_ranking),
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
                cfg = _build_config(
                    dataset=job.dataset,
                    candidate=job.candidate,
                    run_dir=job.run_dir,
                    gpu_index=job.gpu_index,
                    args=args,
                    code_dir=code_dir,
                )
                with job.config_path.open("w", encoding="utf-8") as handle:
                    json.dump(cfg, handle, indent=2)
                cmd = _command_for_config(args, code_dir, job.config_path)
                log_handle = job.log_path.open("w", encoding="utf-8")
                import subprocess  # local import to keep module import minimal

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
                cache = ds_cache[job.dataset.name]
                true_eor = cache["true_eor"]
                true_fg = cache["true_fg"]
                assert isinstance(true_eor, np.ndarray) and isinstance(true_fg, np.ndarray)
                row = _run_job_result_only(
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
                    f"converged={row.get('converged')} eor_score={row.get('eor_corr_score')} "
                    f"ps_mad={row.get('ps2d_win_log10_mad')}"
                )
            active = still_active
            if active:
                time.sleep(1.0)

    detail_csv = output_dir / "poly_rfft_scan_results.csv"
    _write_csv(detail_csv, rows)
    ranked = _candidate_summary(rows, exclude_datasets=_parse_csv_tokens(args.exclude_from_ranking))
    rank_csv = output_dir / "poly_rfft_scan_rank.csv"
    _write_csv(rank_csv, ranked)
    _write_markdown(output_dir / "poly_rfft_scan_summary.md", ranked, manifest["baseline_fixed"])
    print(f"[done] detail={detail_csv}")
    print(f"[done] rank={rank_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
