#!/usr/bin/env python3
"""
Run a loss-term sweep (base and selectable extra-term combinations) on two datasets with
temperature-aware throttling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency on remote hosts
    psutil = None

from losses import VALID_EXTRA_LOSS_TERMS, normalize_extra_loss_terms
from separation_optim import OptimizationConfig, build_cut_xy_indices


DEFAULT_MODE_SPECS: Tuple[str, ...] = ("base", "corr", "rfft", "poly_reparam", "lagcorr")


@dataclass
class DatasetSpec:
    name: str
    input_cube: Path
    fg_true_cube: Path
    eor_true_cube: Path
    mask_cube: Optional[Path] = None


@dataclass
class ThermalRunStats:
    return_code: int
    runtime_sec: float
    paused_sec: float
    pause_count: int
    max_gpu_temp_c: Optional[float]
    max_cpu_temp_c: Optional[float]
    temp_trace_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep loss modes across two datasets with thermal guard."
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/ and 3dnet/ (default: current directory).",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help="Optional code directory containing separation_cli.py (default: <work-root>/3dnet).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional data directory containing all_cube2.fits and back/all_cube1.fits (default: <work-root>/data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for sweep outputs. Default: <work-root>/runs/mode_sweep_<timestamp>.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=None,
        help="Directory for daily reports. Default: <work-root>/reports/daily.",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor/use.")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODE_SPECS),
        help=(
            "Comma-separated mode specs. Each spec can be 'base' or a '+'-joined extra-term set, "
            "e.g. 'corr', 'rfft+lagcorr', 'corr+rfft+lagcorr'."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="cube1,cube2",
        help="Comma-separated datasets to run. Available: cube1,cube2.",
    )
    parser.add_argument("--num-iters", type=int, default=300, help="Iterations per run.")
    parser.add_argument("--print-every", type=int, default=50, help="Iteration log interval.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--corr-weight",
        type=float,
        default=1.0,
        help="Weight for per-frequency FG/EoR correlation term when 'corr' is enabled.",
    )
    parser.add_argument(
        "--corr-prior-mean",
        type=float,
        default=0.0,
        help="Prior mean for per-frequency FG/EoR correlation term.",
    )
    parser.add_argument(
        "--corr-prior-sigma",
        type=float,
        default=0.5,
        help="Prior sigma for per-frequency FG/EoR correlation term.",
    )
    parser.add_argument(
        "--cut-size-frac",
        type=float,
        default=0.30,
        help="cut_xy size in fraction of the spatial field.",
    )
    parser.add_argument(
        "--extra-loss-start-iter",
        type=int,
        default=50,
        help="Iteration where non-base extra terms begin to activate.",
    )
    parser.add_argument(
        "--extra-loss-ramp-iters",
        type=int,
        default=50,
        help="Ramp iterations for extra loss terms.",
    )
    parser.add_argument(
        "--gpu-pause-temp-c",
        type=float,
        default=80.0,
        help="Pause run when GPU temperature reaches this threshold.",
    )
    parser.add_argument(
        "--gpu-resume-temp-c",
        type=float,
        default=72.0,
        help="Resume run after cooling below this threshold.",
    )
    parser.add_argument(
        "--cpu-pause-temp-c",
        type=float,
        default=88.0,
        help="Pause run when max CPU sensor temperature reaches this threshold.",
    )
    parser.add_argument(
        "--cpu-resume-temp-c",
        type=float,
        default=80.0,
        help="Resume run after CPU cools below this threshold.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Thermal polling interval in seconds.",
    )
    parser.add_argument(
        "--cooldown-check-seconds",
        type=float,
        default=15.0,
        help="Polling interval between runs while waiting for cool-down.",
    )
    parser.add_argument(
        "--power-config",
        type=Path,
        default=None,
        help="Optional power-spectrum JSON config. If set, each run writes power outputs to <run_dir>/powerspec.",
    )
    return parser.parse_args()


def _parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if value.upper() in {"N/A", "[NOT SUPPORTED]", "NOT SUPPORTED", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def query_gpu_stats() -> Dict[int, Dict[str, Optional[float]]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    stats: Dict[int, Dict[str, Optional[float]]] = {}
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_index = int(parts[0])
        except ValueError:
            continue
        stats[gpu_index] = {
            "temp": _parse_float(parts[1]),
            "util": _parse_float(parts[2]),
            "power": _parse_float(parts[3]),
        }
    return stats


def query_max_cpu_temp() -> Optional[float]:
    if psutil is None:
        return None
    try:
        temp_map = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None
    max_temp: Optional[float] = None
    for entries in temp_map.values():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue
            if max_temp is None or current > max_temp:
                max_temp = float(current)
    return max_temp


def _build_base_config(
    dataset: DatasetSpec,
    mode_label: str,
    extra_terms: Sequence[str],
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "input_cube": str(dataset.input_cube),
        "mask_cube": str(dataset.mask_cube) if dataset.mask_cube else "",
        "fg_output": str(run_dir / "fg_est.fits"),
        "eor_output": str(run_dir / "eor_est.fits"),
        "optim": {
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "freq_axis": 0,
            "print_every": int(args.print_every),
            "device": f"cuda:{int(args.gpu_index)}",
            "dtype": "float32",
            "loss_mode": "base",
            "extra_loss_terms": list(extra_terms),
            "extra_loss_start_iter": int(args.extra_loss_start_iter),
            "extra_loss_ramp_iters": int(args.extra_loss_ramp_iters),
            "optimizer_name": "adam",
            "momentum": 0.9,
            "freq_start_mhz": 106.0,
            "freq_delta_mhz": 0.1,
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
            "beta": 1.0,
            "gamma": 1.0,
            "corr_weight": float(args.corr_weight),
            "lagcorr_weight": 1.0,
            "fft_weight": 1.0,
            "poly_weight": 1.0,
        },
        "priors": {
            "data_error": 0.005,
            "eor_prior_mean": 0.0,
            "eor_prior_sigma": 0.1,
            "fg_smooth_mean": 0.0,
            "fg_smooth_sigma": 0.0005,
            "fg_reference_cube": str(dataset.fg_true_cube),
            "use_robust_fg_stats": True,
            "mae_to_sigma_factor": 1.4826,
            "corr_prior_mean": float(args.corr_prior_mean),
            "corr_prior_sigma": float(args.corr_prior_sigma),
            "lagcorr_feature": "raw",
            "lagcorr_unit": "mhz",
            "lagcorr_fg_component_weight": 0.5,
            "lagcorr_eor_component_weight": 0.5,
            "lagcorr_pair_sampling": "random",
            "lagcorr_random_seed": 20260210,
            "lagcorr_intervals": [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5],
            "fg_lagcorr_mean": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "fg_lagcorr_sigma": [0.005, 0.005, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03],
            "eor_lagcorr_mean": [0.89, 0.78, 0.52, 0.27, 0.15, 0.09, 0.03, 0.00, 0.00],
            "eor_lagcorr_sigma": [0.12, 0.12, 0.12, 0.08, 0.06, 0.05, 0.05, 0.08, 0.12],
            "lagcorr_max_pairs": 256,
            "fft_highfreq_percent": 0.7,
            "fft_use_log_energy": True,
            "fft_z_clip": 6.0,
            "fft_prior_mean": 0.0,
            "fft_prior_sigma": 1.0,
            "poly_degree": 3,
            "poly_sigma": 0.01,
        },
        "evaluation": {
            "true_eor_cube": str(dataset.eor_true_cube),
            "diagnose_input": False,
            "corr_plot": str(run_dir / "eor_corr.png"),
            "enable_corr_check": True,
            "corr_check_every": 50,
        },
        "init": {
            "init_fg_cube": "",
            "init_eor_cube": "",
        },
    }
    cfg["mode_label"] = mode_label
    if args.power_config is not None:
        cfg["power_config"] = str(args.power_config)
        cfg["power_output_dir"] = str(run_dir / "powerspec")
    return cfg


def _extract_cut_indices(shape: Sequence[int], cut_size_frac: float) -> Optional[Tuple[int, int, int, int]]:
    cfg = OptimizationConfig()
    cfg.freq_axis = 0
    cfg.cut_xy_enabled = True
    cfg.cut_xy_unit = "frac"
    cfg.cut_xy_center_x = 0.5
    cfg.cut_xy_center_y = 0.5
    cfg.cut_xy_size = float(cut_size_frac)
    indices = build_cut_xy_indices(shape=shape, freq_axis=0, config=cfg)
    if indices is None:
        return None
    return indices.x0, indices.x1, indices.y0, indices.y1


def _load_mask(mask_path: Optional[Path], cut: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    with fits.open(mask_path, memmap=True) as hdul:
        mask = hdul[0].data
        if cut is not None:
            x0, x1, y0, y1 = cut
            if mask.ndim == 2:
                mask = mask[y0:y1, x0:x1]
            elif mask.ndim == 3:
                mask = mask[:, y0:y1, x0:x1]
        mask_np = np.asarray(mask, dtype=np.float32)
    return mask_np


def _load_cube_cut(path: Path, cut: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if cut is not None:
            x0, x1, y0, y1 = cut
            data = data[:, y0:y1, x0:x1]
        return np.asarray(data, dtype=np.float32)


def _apply_mask(cube: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return cube
    if mask.ndim == 2:
        return cube * mask[None, :, :]
    if mask.ndim == 3:
        return cube * mask
    raise ValueError(f"Unsupported mask ndim={mask.ndim}, expected 2 or 3.")


def _frequency_correlations(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    if est.shape != true.shape:
        raise ValueError(f"Shape mismatch for correlation: {est.shape} vs {true.shape}.")
    out = np.full((est.shape[0],), np.nan, dtype=np.float64)
    for i in range(est.shape[0]):
        a = est[i].reshape(-1).astype(np.float64, copy=False)
        b = true[i].reshape(-1).astype(np.float64, copy=False)
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            out[i] = float(np.dot(a, b) / denom)
    return out


def _load_power1d(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            return None
        table = hdul[1].data
        if "power" not in table.names:
            return None
        return np.asarray(table["power"], dtype=np.float64)


def _load_power2d(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with fits.open(path, memmap=True) as hdul:
        if hdul[0].data is None:
            return None
        return np.asarray(hdul[0].data, dtype=np.float64)


def _relative_power_percent(rec: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rel = np.full(rec.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(rec) & np.isfinite(true) & (np.abs(true) > eps)
    if np.any(mask):
        rel[mask] = 100.0 * np.abs(rec[mask] - true[mask]) / np.abs(true[mask])
    return rel


def _append_percent_metrics(prefix: str, rel: np.ndarray, out: Dict[str, object]) -> None:
    vals = rel[np.isfinite(rel)]
    if vals.size == 0:
        return
    out[f"{prefix}_mean_pct"] = float(np.mean(vals))
    out[f"{prefix}_median_pct"] = float(np.median(vals))
    out[f"{prefix}_p90_pct"] = float(np.percentile(vals, 90))
    out[f"{prefix}_p95_pct"] = float(np.percentile(vals, 95))


def _collect_power_metrics(power_dir: Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {"power_dir": str(power_dir)}
    rec_1d = _load_power1d(power_dir / "power1d_rec.fits")
    true_1d = _load_power1d(power_dir / "power1d_true.fits")
    if rec_1d is not None and true_1d is not None and rec_1d.shape == true_1d.shape:
        rel_1d = _relative_power_percent(rec_1d, true_1d)
        _append_percent_metrics("ps1d_rel", rel_1d, metrics)

    rec_2d = _load_power2d(power_dir / "power2d_rec.fits")
    true_2d = _load_power2d(power_dir / "power2d_true.fits")
    if rec_2d is not None and true_2d is not None and rec_2d.shape == true_2d.shape:
        rel_2d = _relative_power_percent(rec_2d, true_2d)
        _append_percent_metrics("ps2d_rel", rel_2d, metrics)
    return metrics


def _parse_final_loss_components(log_path: Path) -> Dict[str, float]:
    target_line: Optional[str] = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Finished optimization:"):
                target_line = line.strip()
    if target_line is None:
        return {}
    pairs = re.findall(r"([a-z_]+)=([-+0-9.eE]+)", target_line)
    parsed: Dict[str, float] = {}
    for key, raw in pairs:
        try:
            parsed[key] = float(raw)
        except ValueError:
            continue
    return parsed


def _needs_pause(
    gpu_temp: Optional[float],
    cpu_temp: Optional[float],
    gpu_pause_temp: float,
    cpu_pause_temp: float,
) -> bool:
    return (gpu_temp is not None and gpu_temp >= gpu_pause_temp) or (
        cpu_temp is not None and cpu_temp >= cpu_pause_temp
    )


def _can_resume(
    gpu_temp: Optional[float],
    cpu_temp: Optional[float],
    gpu_resume_temp: float,
    cpu_resume_temp: float,
) -> bool:
    gpu_ok = (gpu_temp is None) or (gpu_temp <= gpu_resume_temp)
    cpu_ok = (cpu_temp is None) or (cpu_temp <= cpu_resume_temp)
    return gpu_ok and cpu_ok


def run_with_thermal_guard(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    gpu_index: int,
    gpu_pause_temp: float,
    gpu_resume_temp: float,
    cpu_pause_temp: float,
    cpu_resume_temp: float,
    poll_seconds: float,
) -> ThermalRunStats:
    temp_trace = log_path.with_suffix(".temps.csv")
    pause_count = 0
    paused_sec = 0.0
    pause_started: Optional[float] = None
    max_gpu_temp: Optional[float] = None
    max_cpu_temp: Optional[float] = None

    start = time.time()
    paused = False

    with log_path.open("w", encoding="utf-8") as log_handle, temp_trace.open(
        "w", encoding="utf-8", newline=""
    ) as temp_handle:
        writer = csv.writer(temp_handle)
        writer.writerow(
            ["timestamp", "gpu_index", "gpu_temp_c", "gpu_util_percent", "gpu_power_w", "cpu_max_temp_c", "paused"]
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            ret = proc.poll()
            now = time.time()

            gpu_stats_map = query_gpu_stats()
            gpu_stats = gpu_stats_map.get(gpu_index, {})
            gpu_temp = gpu_stats.get("temp")
            gpu_util = gpu_stats.get("util")
            gpu_power = gpu_stats.get("power")
            cpu_temp = query_max_cpu_temp()

            if gpu_temp is not None:
                max_gpu_temp = gpu_temp if max_gpu_temp is None else max(max_gpu_temp, gpu_temp)
            if cpu_temp is not None:
                max_cpu_temp = cpu_temp if max_cpu_temp is None else max(max_cpu_temp, cpu_temp)

            writer.writerow(
                [f"{now:.3f}", gpu_index, gpu_temp, gpu_util, gpu_power, cpu_temp, int(paused)]
            )
            temp_handle.flush()

            if ret is not None:
                if paused and pause_started is not None:
                    paused_sec += now - pause_started
                runtime_sec = now - start
                return ThermalRunStats(
                    return_code=ret,
                    runtime_sec=runtime_sec,
                    paused_sec=paused_sec,
                    pause_count=pause_count,
                    max_gpu_temp_c=max_gpu_temp,
                    max_cpu_temp_c=max_cpu_temp,
                    temp_trace_csv=temp_trace,
                )

            if (not paused) and _needs_pause(
                gpu_temp, cpu_temp, gpu_pause_temp=gpu_pause_temp, cpu_pause_temp=cpu_pause_temp
            ):
                os.kill(proc.pid, signal.SIGSTOP)
                paused = True
                pause_count += 1
                pause_started = now
                log_handle.write(
                    f"\n[thermal-guard] paused at t={now - start:.1f}s, "
                    f"gpu={gpu_temp}, cpu={cpu_temp}\n"
                )
                log_handle.flush()

            elif paused and _can_resume(
                gpu_temp, cpu_temp, gpu_resume_temp=gpu_resume_temp, cpu_resume_temp=cpu_resume_temp
            ):
                os.kill(proc.pid, signal.SIGCONT)
                paused = False
                if pause_started is not None:
                    paused_sec += now - pause_started
                pause_started = None
                log_handle.write(
                    f"[thermal-guard] resumed at t={now - start:.1f}s, "
                    f"gpu={gpu_temp}, cpu={cpu_temp}\n"
                )
                log_handle.flush()

            time.sleep(poll_seconds if not paused else max(5.0, poll_seconds * 0.5))


def wait_for_cooldown(
    gpu_index: int,
    gpu_resume_temp: float,
    cpu_resume_temp: float,
    poll_seconds: float,
) -> None:
    while True:
        gpu_stats = query_gpu_stats().get(gpu_index, {})
        gpu_temp = gpu_stats.get("temp")
        cpu_temp = query_max_cpu_temp()
        if _can_resume(
            gpu_temp, cpu_temp, gpu_resume_temp=gpu_resume_temp, cpu_resume_temp=cpu_resume_temp
        ):
            return
        print(
            f"[cooldown] waiting: gpu_temp={gpu_temp}, cpu_temp={cpu_temp}, "
            f"resume thresholds=({gpu_resume_temp}, {cpu_resume_temp})"
        )
        time.sleep(poll_seconds)


def ensure_cube2_observation(all_cube2_path: Path, fg_cube2_path: Path, eor_cube2_path: Path) -> None:
    if all_cube2_path.exists():
        return
    all_cube2_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prep] creating {all_cube2_path} from fg+eor ...")
    shutil.copy2(fg_cube2_path, all_cube2_path)
    with fits.open(all_cube2_path, mode="update", memmap=True) as out_hdul, fits.open(
        eor_cube2_path, memmap=True
    ) as eor_hdul:
        out_data = out_hdul[0].data
        eor_data = eor_hdul[0].data
        if out_data.shape != eor_data.shape:
            raise ValueError(
                f"cube2 shape mismatch: {out_data.shape} (output) vs {eor_data.shape} (eor)"
            )
        nfreq = int(out_data.shape[0])
        for i in range(nfreq):
            out_data[i, :, :] = out_data[i, :, :] + eor_data[i, :, :]
            if (i + 1) % 16 == 0:
                out_hdul.flush()
        out_hdul.flush()
    print(f"[prep] created {all_cube2_path}")


def _load_header_shape(path: Path) -> Tuple[int, int, int]:
    header = fits.getheader(path)
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    nf = int(header["NAXIS3"])
    return nf, ny, nx


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "nan"
        v = float(value)
        if abs(v) >= 1e-3 and abs(v) < 1e3:
            return f"{v:.6f}"
        return f"{v:.6e}"
    return str(value)


def _serialize_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [str(x) if isinstance(x, Path) else x for x in value]
        else:
            out[key] = value
    return out


def _path_metadata(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    info: Dict[str, object] = {
        "path": str(path),
        "realpath": str(path.resolve()),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    st = path.stat()
    info["size_bytes"] = int(st.st_size)
    info["mtime_epoch"] = float(st.st_mtime)
    if path.suffix.lower() == ".fits":
        try:
            header = fits.getheader(path)
            nx = int(header.get("NAXIS1", 0))
            ny = int(header.get("NAXIS2", 0))
            nf = int(header.get("NAXIS3", 0))
            info["fits_shape"] = [nf, ny, nx]
            bitpix = header.get("BITPIX")
            if bitpix is not None:
                info["fits_bitpix"] = int(bitpix)
        except Exception as exc:
            info["fits_header_error"] = str(exc)
    return info


def _write_run_manifest(
    output_dir: Path,
    work_root: Path,
    d3_root: Path,
    args: argparse.Namespace,
    datasets: Sequence[DatasetSpec],
) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "work_root": str(work_root),
        "work_root_realpath": str(work_root.resolve()),
        "code_dir": str(d3_root),
        "code_dir_realpath": str(d3_root.resolve()),
        "args": _serialize_args(args),
        "power_config": _path_metadata(args.power_config) if args.power_config else None,
        "datasets": {},
    }
    datasets_info: Dict[str, object] = {}
    for ds in datasets:
        datasets_info[ds.name] = {
            "input_cube": _path_metadata(ds.input_cube),
            "fg_true_cube": _path_metadata(ds.fg_true_cube),
            "eor_true_cube": _path_metadata(ds.eor_true_cube),
            "mask_cube": _path_metadata(ds.mask_cube) if ds.mask_cube else None,
        }
    manifest["datasets"] = datasets_info
    path = output_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def _append_daily_reports(
    reports_root: Path,
    stamp: str,
    output_dir: Path,
    csv_path: Path,
    results: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    reports_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    summary_path = reports_root / f"{day}_summary.md"
    detail_path = reports_root / f"{day}_detail.md"

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in results:
        grouped.setdefault(str(row.get("dataset", "unknown")), []).append(row)

    def _best_row(dataset_rows: Sequence[Dict[str, object]], key: str, reverse: bool = False) -> Optional[Dict[str, object]]:
        filtered = [r for r in dataset_rows if isinstance(r.get(key), (int, float))]
        if not filtered:
            return None
        return sorted(filtered, key=lambda r: float(r[key]), reverse=reverse)[0]

    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {time_str} {stamp}\n")
        handle.write(f"- output_dir: `{output_dir}`\n")
        handle.write(f"- csv: `{csv_path}`\n")
        all_gpu_temps = [
            float(row["max_gpu_temp_c"])
            for row in results
            if isinstance(row.get("max_gpu_temp_c"), (int, float))
        ]
        all_cpu_temps = [
            float(row["max_cpu_temp_c"])
            for row in results
            if isinstance(row.get("max_cpu_temp_c"), (int, float))
        ]
        max_gpu = max(all_gpu_temps) if all_gpu_temps else None
        max_cpu = max(all_cpu_temps) if all_cpu_temps else None
        handle.write(f"- max_gpu_temp_c: {_format_metric(max_gpu)}\n")
        handle.write(f"- max_cpu_temp_c: {_format_metric(max_cpu)}\n")
        handle.write("- per-dataset best (by EoR MSE):\n")
        for dataset, ds_rows in sorted(grouped.items()):
            best_eor = _best_row(ds_rows, "eor_mse", reverse=False)
            best_corr = _best_row(ds_rows, "eor_corr_mean", reverse=True)
            best_recon = _best_row(ds_rows, "recon_mse", reverse=False)
            best_ps1d = _best_row(ds_rows, "ps1d_rel_median_pct", reverse=False)
            best_ps2d = _best_row(ds_rows, "ps2d_rel_median_pct", reverse=False)
            handle.write(
                f"  - {dataset}: "
                f"best_eor={best_eor.get('mode') if best_eor else 'n/a'}({_format_metric(best_eor.get('eor_mse') if best_eor else None)}), "
                f"best_corr={best_corr.get('mode') if best_corr else 'n/a'}({_format_metric(best_corr.get('eor_corr_mean') if best_corr else None)}), "
                f"best_recon={best_recon.get('mode') if best_recon else 'n/a'}({_format_metric(best_recon.get('recon_mse') if best_recon else None)}), "
                f"best_ps1d={best_ps1d.get('mode') if best_ps1d else 'n/a'}({_format_metric(best_ps1d.get('ps1d_rel_median_pct') if best_ps1d else None)}), "
                f"best_ps2d={best_ps2d.get('mode') if best_ps2d else 'n/a'}({_format_metric(best_ps2d.get('ps2d_rel_median_pct') if best_ps2d else None)})\n"
            )

    with detail_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {time_str} {stamp}\n")
        handle.write(f"- output_dir: `{output_dir}`\n")
        handle.write(f"- csv: `{csv_path}`\n\n")
        handle.write(
            "| dataset | mode | status | fg_mse | eor_mse | recon_mse | eor_corr_mean | ps1d_rel_median_pct | ps2d_rel_median_pct | runtime_sec | max_gpu_temp_c | pause_count |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row.get('dataset')} | {row.get('mode')} | {row.get('status')} "
                f"| {_format_metric(row.get('fg_mse'))} | {_format_metric(row.get('eor_mse'))} "
                f"| {_format_metric(row.get('recon_mse'))} | {_format_metric(row.get('eor_corr_mean'))} "
                f"| {_format_metric(row.get('ps1d_rel_median_pct'))} | {_format_metric(row.get('ps2d_rel_median_pct'))} "
                f"| {_format_metric(row.get('runtime_sec'))} | {_format_metric(row.get('max_gpu_temp_c'))} "
                f"| {_format_metric(row.get('pause_count'))} |\n"
            )
        handle.write("\n")
        handle.write("- artifact paths:\n")
        for row in results:
            handle.write(
                f"  - {row.get('dataset')}/{row.get('mode')}: "
                f"log=`{row.get('log_path')}`, config=`{row.get('config_path')}`, "
                f"fg=`{row.get('fg_output')}`, eor=`{row.get('eor_output')}`, "
                f"temps=`{row.get('temp_trace_csv')}`\n"
            )

    return summary_path, detail_path


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    d3_root = args.code_dir.resolve() if args.code_dir is not None else (work_root / "3dnet")
    data_dir = args.data_dir.resolve() if args.data_dir is not None else (work_root / "data")

    if args.power_config is not None:
        power_cfg = args.power_config
        if not power_cfg.is_absolute():
            power_cfg = (d3_root / power_cfg).resolve() if args.code_dir is not None else (work_root / power_cfg).resolve()
        if not power_cfg.exists():
            raise FileNotFoundError(f"Power config not found: {power_cfg}")
        args.power_config = power_cfg

    if not d3_root.exists():
        raise FileNotFoundError(f"Code directory not found: {d3_root}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve() if args.output_dir else (work_root / "runs" / f"mode_sweep_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_root = args.reports_root.resolve() if args.reports_root else (work_root / "reports" / "daily")

    ensure_cube2_observation(
        all_cube2_path=data_dir / "all_cube2.fits",
        fg_cube2_path=data_dir / "fg_cube2.fits",
        eor_cube2_path=data_dir / "eor_cube2.fits",
    )

    all_datasets = [
        DatasetSpec(
            name="cube1",
            input_cube=data_dir / "back" / "all_cube1.fits",
            fg_true_cube=data_dir / "fg_cube1.fits",
            eor_true_cube=data_dir / "eor_cube1.fits",
            mask_cube=None,
        ),
        DatasetSpec(
            name="cube2",
            input_cube=data_dir / "all_cube2.fits",
            fg_true_cube=data_dir / "fg_cube2.fits",
            eor_true_cube=data_dir / "eor_cube2.fits",
            mask_cube=None,
        ),
    ]
    dataset_map: Dict[str, DatasetSpec] = {ds.name: ds for ds in all_datasets}
    selected_dataset_names = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    if not selected_dataset_names:
        raise ValueError("No datasets selected. Use --datasets cube1,cube2 (or subset).")
    unknown_datasets = [name for name in selected_dataset_names if name not in dataset_map]
    if unknown_datasets:
        raise ValueError(
            f"Unknown datasets {unknown_datasets}. Available datasets: {sorted(dataset_map.keys())}"
        )
    datasets = [dataset_map[name] for name in selected_dataset_names]

    selected_mode_specs = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    if not selected_mode_specs:
        raise ValueError(
            "No modes selected. Use --modes base,corr,rfft,poly_reparam,lagcorr (or '+' combinations)."
        )
    selected_modes: List[Tuple[str, Tuple[str, ...]]] = []
    for spec in selected_mode_specs:
        try:
            extras = normalize_extra_loss_terms(loss_mode=spec, extra_loss_terms=None)
        except ValueError as exc:
            valid = ", ".join(["base", *VALID_EXTRA_LOSS_TERMS])
            raise ValueError(f"Invalid mode spec '{spec}'. Valid terms: {valid}.") from exc
        label = "base" if not extras else "+".join(extras)
        selected_modes.append((label, extras))

    manifest_path = _write_run_manifest(
        output_dir=output_dir,
        work_root=work_root,
        d3_root=d3_root,
        args=args,
        datasets=datasets,
    )
    print(f"[manifest] wrote {manifest_path}")

    results: List[Dict[str, object]] = []

    for ds in datasets:
        if not ds.input_cube.exists():
            raise FileNotFoundError(f"Missing input cube: {ds.input_cube}")
        if not ds.fg_true_cube.exists():
            raise FileNotFoundError(f"Missing fg cube: {ds.fg_true_cube}")
        if not ds.eor_true_cube.exists():
            raise FileNotFoundError(f"Missing eor cube: {ds.eor_true_cube}")
        if ds.mask_cube and (not ds.mask_cube.exists()):
            raise FileNotFoundError(f"Missing mask cube: {ds.mask_cube}")

        full_shape = _load_header_shape(ds.input_cube)
        cut = _extract_cut_indices(shape=full_shape, cut_size_frac=float(args.cut_size_frac))
        print(f"[dataset:{ds.name}] full_shape={full_shape}, cut={cut}")

        mask_arr = _load_mask(ds.mask_cube, cut=cut)
        fg_true = _apply_mask(_load_cube_cut(ds.fg_true_cube, cut=cut), mask_arr)
        eor_true = _apply_mask(_load_cube_cut(ds.eor_true_cube, cut=cut), mask_arr)
        obs_true = fg_true + eor_true

        for mode_label, mode_terms in selected_modes:
            run_dir = output_dir / ds.name / mode_label
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = _build_base_config(
                ds, mode_label=mode_label, extra_terms=mode_terms, run_dir=run_dir, args=args
            )
            cfg_path = run_dir / "config.json"
            with cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle, indent=2)

            log_path = run_dir / "run.log"
            cmd = [sys.executable, str(d3_root / "separation_cli.py"), "--config", str(cfg_path)]
            print(f"[run] dataset={ds.name}, mode={mode_label}, cmd={' '.join(cmd)}")

            wait_for_cooldown(
                gpu_index=int(args.gpu_index),
                gpu_resume_temp=float(args.gpu_resume_temp_c),
                cpu_resume_temp=float(args.cpu_resume_temp_c),
                poll_seconds=float(args.cooldown_check_seconds),
            )

            thermal = run_with_thermal_guard(
                cmd=cmd,
                cwd=work_root,
                log_path=log_path,
                gpu_index=int(args.gpu_index),
                gpu_pause_temp=float(args.gpu_pause_temp_c),
                gpu_resume_temp=float(args.gpu_resume_temp_c),
                cpu_pause_temp=float(args.cpu_pause_temp_c),
                cpu_resume_temp=float(args.cpu_resume_temp_c),
                poll_seconds=float(args.poll_seconds),
            )

            fg_out_path = run_dir / "fg_est.fits"
            eor_out_path = run_dir / "eor_est.fits"
            record: Dict[str, object] = {
                "dataset": ds.name,
                "mode": mode_label,
                "extra_loss_terms": ",".join(mode_terms),
                "input_cube": str(ds.input_cube),
                "input_cube_realpath": str(ds.input_cube.resolve()),
                "full_shape": "x".join(str(v) for v in full_shape),
                "cut_bounds": f"{cut}" if cut is not None else "None",
                "cut_size_frac": float(args.cut_size_frac),
                "num_iters": int(args.num_iters),
                "status": "ok" if thermal.return_code == 0 else "failed",
                "return_code": thermal.return_code,
                "runtime_sec": thermal.runtime_sec,
                "paused_sec": thermal.paused_sec,
                "pause_count": thermal.pause_count,
                "max_gpu_temp_c": thermal.max_gpu_temp_c,
                "max_cpu_temp_c": thermal.max_cpu_temp_c,
                "log_path": str(log_path),
                "temp_trace_csv": str(thermal.temp_trace_csv),
                "config_path": str(cfg_path),
                "fg_output": str(fg_out_path),
                "eor_output": str(eor_out_path),
            }

            if thermal.return_code == 0 and fg_out_path.exists() and eor_out_path.exists():
                fg_est = np.asarray(fits.getdata(fg_out_path), dtype=np.float32)
                eor_est = np.asarray(fits.getdata(eor_out_path), dtype=np.float32)
                if fg_est.shape != fg_true.shape or eor_est.shape != eor_true.shape:
                    raise ValueError(
                        f"Output/true shape mismatch for {ds.name}/{mode_label}: "
                        f"fg_est={fg_est.shape}, fg_true={fg_true.shape}, "
                        f"eor_est={eor_est.shape}, eor_true={eor_true.shape}"
                    )

                fg_mse = float(np.mean((fg_est - fg_true) ** 2))
                eor_mse = float(np.mean((eor_est - eor_true) ** 2))
                recon_mse = float(np.mean((fg_est + eor_est - obs_true) ** 2))

                corrs = _frequency_correlations(eor_est, eor_true)
                corr_mean = float(np.nanmean(corrs))
                corr_min = float(np.nanmin(corrs))
                corr_max = float(np.nanmax(corrs))

                final_losses = _parse_final_loss_components(log_path)

                record.update(
                    {
                        "fg_mse": fg_mse,
                        "eor_mse": eor_mse,
                        "recon_mse": recon_mse,
                        "eor_corr_mean": corr_mean,
                        "eor_corr_min": corr_min,
                        "eor_corr_max": corr_max,
                        "final_total_loss": final_losses.get("total"),
                        "final_data_loss": final_losses.get("data"),
                        "final_smooth_loss": final_losses.get("smooth"),
                        "final_eor_loss": final_losses.get("eor"),
                        "final_corr_loss": final_losses.get("corr"),
                        "final_lagcorr_loss": final_losses.get("lagcorr"),
                        "final_fft_loss": final_losses.get("fft"),
                        "final_poly_loss": final_losses.get("poly"),
                    }
                )
                power_dir = run_dir / "powerspec"
                if power_dir.exists():
                    record.update(_collect_power_metrics(power_dir))

            results.append(record)

            summary_txt = run_dir / "summary.json"
            with summary_txt.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)

            print(
                f"[done] dataset={ds.name}, mode={mode_label}, status={record['status']}, "
                f"runtime={thermal.runtime_sec:.1f}s, pause_count={thermal.pause_count}, "
                f"max_gpu_temp={thermal.max_gpu_temp_c}"
            )

    csv_path = output_dir / "mode_sweep_results.csv"
    if results:
        fieldnames: List[str] = []
        for row in results:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    md_path = output_dir / "mode_sweep_results.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Loss Mode Sweep Results\n\n")
        handle.write(f"- output_dir: `{output_dir}`\n")
        handle.write(f"- csv: `{csv_path}`\n\n")
        handle.write(
            "| dataset | mode | status | fg_mse | eor_mse | recon_mse | eor_corr_mean | ps1d_rel_median_pct | ps2d_rel_median_pct | runtime_sec | max_gpu_temp_c |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            handle.write(
                f"| {row.get('dataset')} | {row.get('mode')} | {row.get('status')} "
                f"| {row.get('fg_mse', 'n/a')} | {row.get('eor_mse', 'n/a')} | {row.get('recon_mse', 'n/a')} "
                f"| {row.get('eor_corr_mean', 'n/a')} | {row.get('ps1d_rel_median_pct', 'n/a')} "
                f"| {row.get('ps2d_rel_median_pct', 'n/a')} | {row.get('runtime_sec', 'n/a')} | {row.get('max_gpu_temp_c', 'n/a')} |\n"
            )

    print("\n=== Sweep complete ===")
    print(f"Results CSV: {csv_path}")
    print(f"Results MD:  {md_path}")
    summary_path, detail_path = _append_daily_reports(
        reports_root=reports_root,
        stamp=stamp,
        output_dir=output_dir,
        csv_path=csv_path,
        results=results,
    )
    print(f"Daily summary report: {summary_path}")
    print(f"Daily detail report:  {detail_path}")
    for row in results:
        print(
            f"{row.get('dataset')}/{row.get('mode')}: status={row.get('status')}, "
            f"fg_mse={row.get('fg_mse')}, eor_mse={row.get('eor_mse')}, "
            f"recon_mse={row.get('recon_mse')}, eor_corr_mean={row.get('eor_corr_mean')}, "
            f"ps1d_rel_median_pct={row.get('ps1d_rel_median_pct')}, "
            f"ps2d_rel_median_pct={row.get('ps2d_rel_median_pct')}"
        )


if __name__ == "__main__":
    main()
