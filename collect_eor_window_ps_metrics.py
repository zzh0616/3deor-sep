#!/usr/bin/env python3
"""
Collect EoR-window 2D power-spectrum agreement metrics across historical runs.

This script crawls `runs/**/eor_est.fits` (and sibling `config.json`), computes
2D power spectra for recovered vs injected EoR, then evaluates agreement inside
an EoR window as defined by a PowerSpecConfig.

It writes a single CSV suitable for ranking/comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This script requires torch. Run it inside your conda torch env.") from exc

from powerspec import (
    PowerSpecConfig,
    compute_eor_window_mask,
    compute_power2d_window_metrics,
    compute_power_spectra,
)


@dataclass(frozen=True)
class CutXY:
    enabled: bool
    unit: str
    center_x: float
    center_y: float
    size: float


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cut_xy(cfg: Dict[str, Any]) -> CutXY:
    d = cfg.get("cut_xy") or {}
    enabled = bool(d.get("enabled", False))
    unit = str(d.get("unit", "frac")).strip().lower()
    cx = float(d.get("center_x", 0.5))
    cy = float(d.get("center_y", 0.5))
    size = float(d.get("size", 1.0))
    return CutXY(enabled=enabled, unit=unit, center_x=cx, center_y=cy, size=size)


def _parse_common_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Support both the newer schema (nested `optim`) and older flat schema.
    if isinstance(cfg.get("optim"), dict):
        optim = cfg["optim"]
        loss_mode = str(optim.get("loss_mode", cfg.get("loss_mode", "")))
        extra_terms = optim.get("extra_loss_terms", cfg.get("extra_loss_terms", []))
        if extra_terms is None:
            extra_terms = []
        return {
            "freq_axis": int(optim.get("freq_axis", cfg.get("freq_axis", 0))),
            "freq_start_mhz": float(optim.get("freq_start_mhz", cfg.get("freq_start_mhz", math.nan))),
            "freq_delta_mhz": float(optim.get("freq_delta_mhz", cfg.get("freq_delta_mhz", math.nan))),
            "num_iters": int(optim.get("num_iters", cfg.get("num_iters", -1))),
            "lr": float(optim.get("lr", cfg.get("lr", math.nan))),
            "loss_mode": loss_mode,
            "extra_loss_terms": list(extra_terms) if isinstance(extra_terms, list) else [str(extra_terms)],
        }

    # Older schema.
    extra_terms = cfg.get("extra_loss_terms", [])
    if extra_terms is None:
        extra_terms = []
    return {
        "freq_axis": int(cfg.get("freq_axis", 0)),
        "freq_start_mhz": float(cfg.get("freq_start_mhz", math.nan)),
        "freq_delta_mhz": float(cfg.get("freq_delta_mhz", math.nan)),
        "num_iters": int(cfg.get("num_iters", -1)),
        "lr": float(cfg.get("lr", math.nan)),
        "loss_mode": str(cfg.get("loss_mode", "")),
        "extra_loss_terms": list(extra_terms) if isinstance(extra_terms, list) else [str(extra_terms)],
    }


def _clamp_fixed_window(start: int, size: int, length: int) -> Tuple[int, int]:
    if size > length:
        raise ValueError(f"Requested crop size {size} exceeds axis length {length}.")
    end = start + size
    if start < 0:
        start = 0
        end = size
    if end > length:
        end = length
        start = length - size
    return int(start), int(end)


def _compute_cut_indices(shape: Tuple[int, int, int], freq_axis: int, cut: CutXY) -> Optional[Tuple[int, int, int, int, int, int]]:
    if not cut.enabled:
        return None
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}.")
    if not (0 <= freq_axis < 3):
        raise ValueError(f"freq_axis must be in [0, 2], got {freq_axis}.")

    spatial_axes = [ax for ax in range(3) if ax != freq_axis]
    x_axis, y_axis = spatial_axes[0], spatial_axes[1]
    nx, ny = int(shape[x_axis]), int(shape[y_axis])
    min_dim = min(nx, ny)

    unit = cut.unit
    if unit not in {"frac", "px"}:
        raise ValueError("cut_xy.unit must be 'frac' or 'px'.")

    if unit == "frac":
        cx = float(cut.center_x)
        cy = float(cut.center_y)
        size = float(cut.size)
        size_px = int(round(size * float(min_dim)))
        center_x_px = int(round(cx * float(max(nx - 1, 0))))
        center_y_px = int(round(cy * float(max(ny - 1, 0))))
    else:
        center_x_px = int(round(float(cut.center_x)))
        center_y_px = int(round(float(cut.center_y)))
        size_px = int(round(float(cut.size)))

    size_px = max(1, int(size_px))
    if size_px > min_dim:
        raise ValueError(f"cut_xy.size={size_px} exceeds min(Nx,Ny)={min_dim}.")

    start_x = center_x_px - size_px // 2
    start_y = center_y_px - size_px // 2
    x0, x1 = _clamp_fixed_window(start_x, size_px, nx)
    y0, y1 = _clamp_fixed_window(start_y, size_px, ny)
    return int(x_axis), int(y_axis), int(x0), int(x1), int(y0), int(y1)


def _read_fits_shape(path: Path) -> Tuple[int, int, int]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None or getattr(data, "ndim", None) != 3:
            raise ValueError(f"Expected 3D cube in {path}")
        return tuple(int(v) for v in data.shape)  # type: ignore[return-value]


def _read_fits_cube(path: Path, *, cut_indices: Optional[Tuple[int, int, int, int, int, int]] = None) -> torch.Tensor:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None or data.ndim != 3:
            raise ValueError(f"Expected 3D cube in {path}, found {None if data is None else data.shape}")
        view = data
        if cut_indices is not None:
            x_axis, y_axis, x0, x1, y0, y1 = cut_indices
            slices: List[slice] = [slice(None)] * 3
            slices[x_axis] = slice(x0, x1)
            slices[y_axis] = slice(y0, y1)
            view = view[tuple(slices)]
        return torch.from_numpy(np.asarray(view, dtype=np.float32))


def _extract_true_eor_path(cfg: Dict[str, Any]) -> Optional[str]:
    ev = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
    if isinstance(ev, dict) and ev.get("true_eor_cube"):
        return str(ev.get("true_eor_cube"))
    if cfg.get("true_eor_cube"):
        return str(cfg.get("true_eor_cube"))
    return None


def _infer_dataset_name(eor_true_path: Path, run_dir: Path) -> str:
    name = eor_true_path.name.lower()
    for key in ("cube1", "cube2"):
        if key in name:
            return key
    # Fall back to path parts.
    for part in run_dir.parts[::-1]:
        low = part.lower()
        if low in {"cube1", "cube2"}:
            return low
    return "unknown"


def _load_power_cfg(power_cfg_path: Path) -> PowerSpecConfig:
    data = _read_json(power_cfg_path)
    return PowerSpecConfig(**data)


def _format_extra_terms(loss_mode: str, extra_terms: List[str]) -> str:
    extra_terms = [str(x).strip() for x in extra_terms if str(x).strip()]
    if not extra_terms:
        return loss_mode
    return f"{loss_mode}+{'+'.join(extra_terms)}"


def collect_metrics(
    runs_root: Path,
    power_cfg_path: Path,
    device: str,
    *,
    limit: Optional[int],
    max_spatial: Optional[int],
) -> List[Dict[str, Any]]:
    power_base = _load_power_cfg(power_cfg_path)
    out_rows: List[Dict[str, Any]] = []

    # Cache: (true_path, cut_indices, freq_axis, ref_freq, df) -> computed power dict
    true_power_cache: Dict[Tuple[str, Optional[Tuple[int, int, int, int, int, int]], int, float, float], Dict[str, np.ndarray]] = {}

    eor_paths = sorted(runs_root.glob("**/eor_est.fits"))
    if limit is not None:
        eor_paths = eor_paths[: max(0, int(limit))]

    torch_device = torch.device(device)

    n_seen = 0
    n_kept = 0
    for eor_path in eor_paths:
        run_dir = eor_path.parent
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            n_seen += 1
            cfg = _read_json(cfg_path)
            fields = _parse_common_fields(cfg)
            cut = _parse_cut_xy(cfg)
            true_eor = _extract_true_eor_path(cfg)
            if not true_eor:
                continue
            true_eor_path = Path(true_eor)
            if not true_eor_path.exists():
                continue

            freq_axis = int(fields["freq_axis"])
            freq_start_mhz = float(fields["freq_start_mhz"])
            freq_delta_mhz = float(fields["freq_delta_mhz"])
            if not (math.isfinite(freq_start_mhz) and math.isfinite(freq_delta_mhz)):
                continue

            if max_spatial is not None:
                rec_shape = _read_fits_shape(eor_path)
                spatial_axes = [ax for ax in range(3) if ax != freq_axis]
                nx_rec = int(rec_shape[spatial_axes[0]])
                ny_rec = int(rec_shape[spatial_axes[1]])
                if max(nx_rec, ny_rec) > int(max_spatial):
                    continue

            dataset = _infer_dataset_name(true_eor_path, run_dir)
            exp_series = run_dir.relative_to(runs_root).parts[0] if run_dir != runs_root else "runs"

            # Configure powerspec with run-specific frequency axis and sampling.
            power_cfg = PowerSpecConfig(**power_base.__dict__)
            power_cfg.freq_axis = freq_axis
            power_cfg.ref_freq_mhz = freq_start_mhz
            power_cfg.df = freq_delta_mhz
            power_cfg.unit_f = "mhz"

            # Load recovered EoR estimate.
            eor_rec = _read_fits_cube(eor_path).to(torch_device)

            # Compute the corresponding cut of the injected EoR cube (for fair comparison).
            full_shape = _read_fits_shape(true_eor_path)
            cut_indices = _compute_cut_indices(full_shape, freq_axis, cut)
            # If the run did not cut but shapes still mismatch, try a centered crop.
            if cut_indices is None:
                # Ensure true cube matches recovered shape.
                # Only supports the typical case where cut applies to spatial axes.
                if tuple(full_shape) != tuple(int(x) for x in eor_rec.shape):
                    spatial_axes = [ax for ax in range(3) if ax != freq_axis]
                    x_axis, y_axis = spatial_axes[0], spatial_axes[1]
                    nx_full, ny_full = int(full_shape[x_axis]), int(full_shape[y_axis])
                    nx_rec, ny_rec = int(eor_rec.shape[x_axis]), int(eor_rec.shape[y_axis])
                    if nx_rec <= nx_full and ny_rec <= ny_full:
                        x0 = (nx_full - nx_rec) // 2
                        y0 = (ny_full - ny_rec) // 2
                        cut_indices = (x_axis, y_axis, x0, x0 + nx_rec, y0, y0 + ny_rec)

            true_key = (str(true_eor_path), cut_indices, freq_axis, float(freq_start_mhz), float(freq_delta_mhz))
            true_power = true_power_cache.get(true_key)
            if true_power is None:
                eor_true = _read_fits_cube(true_eor_path, cut_indices=cut_indices).to(torch_device)
                true_power = compute_power_spectra(eor_true, power_cfg)
                true_power_cache[true_key] = true_power

            rec_power = compute_power_spectra(eor_rec, power_cfg)

            mask = compute_eor_window_mask(rec_power["kperp_centers"], rec_power["kpar_centers"], power_cfg)
            metrics = compute_power2d_window_metrics(
                rec_power["p2d"],
                true_power["p2d"],
                mask,
                eps=float(power_cfg.eor_window_eps),
            )

            row = {
                "run_dir": str(run_dir),
                "series": exp_series,
                "dataset": dataset,
                "loss_label": _format_extra_terms(str(fields["loss_mode"]), list(fields["extra_loss_terms"])),
                "num_iters": int(fields["num_iters"]),
                "lr": float(fields["lr"]),
                "cut_enabled": bool(cut.enabled),
                "cut_unit": cut.unit,
                "cut_size": float(cut.size),
                "freq_start_mhz": float(freq_start_mhz),
                "freq_delta_mhz": float(freq_delta_mhz),
                "eor_window_kpar_min": float(power_cfg.eor_window_kpar_min),
                "eor_window_wedge_slope": float(power_cfg.eor_window_wedge_slope),
                "eor_window_wedge_intercept": float(power_cfg.eor_window_wedge_intercept),
                "eor_window_exclude_dc": bool(power_cfg.eor_window_exclude_dc),
                "ps2d_win_n_bins": float(metrics.get("n_bins", float("nan"))),
                "ps2d_win_log10_mad": float(metrics.get("log10_mad", float("nan"))),
                "ps2d_win_log10_rmse": float(metrics.get("log10_rmse", float("nan"))),
                "ps2d_win_log10_p90_abs": float(metrics.get("log10_p90_abs", float("nan"))),
                "ps2d_win_log10_corr": float(metrics.get("log10_corr", float("nan"))),
                "ps2d_win_power_sum_ratio": float(metrics.get("power_sum_ratio", float("nan"))),
            }
            out_rows.append(row)
            n_kept += 1
            if n_seen % 10 == 0:
                print(f"[progress] seen={n_seen} kept={n_kept} last={run_dir}")
        except Exception:
            # Keep the scan robust; a single bad run shouldn't fail collection.
            continue

    return out_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="runs", help="Root directory containing run outputs.")
    ap.add_argument(
        "--power-config",
        type=str,
        default=str(Path("configs") / "power_eor_window.json"),
        help="PowerSpecConfig JSON path (relative to 3dnet/ or absolute).",
    )
    ap.add_argument("--device", type=str, default="cuda:1", help="torch device for FFT (e.g. cuda:1 or cpu).")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of runs to process.")
    ap.add_argument(
        "--max-spatial",
        type=int,
        default=700,
        help="Skip runs whose spatial size exceeds this threshold (e.g. to avoid 1024^2 cubes).",
    )
    ap.add_argument(
        "--output-csv",
        type=str,
        default=str(Path("reports") / "analysis" / "eor_window_ps_all_runs.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    runs_root = Path(args.runs_root).resolve()
    power_cfg_path = Path(args.power_config)
    if not power_cfg_path.is_absolute():
        power_cfg_path = (Path(__file__).resolve().parent / power_cfg_path).resolve()
    out_csv = Path(args.output_csv).resolve()

    rows = collect_metrics(
        runs_root,
        power_cfg_path,
        args.device,
        limit=args.limit,
        max_spatial=args.max_spatial,
    )
    _write_csv(out_csv, rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
