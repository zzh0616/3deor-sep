#!/usr/bin/env python3
"""
Power spectrum utilities for foreground/EoR separation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from constants import EPS_LOSS, EPS_STD

try:  # optional torch dependency for GPU FFT
    import torch  # type: ignore

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch not available
    torch = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class PowerSpecConfig:
    dx: float  # spacing along x
    dy: float  # spacing along y
    df: float  # spacing along frequency axis (Hz or MHz if unit_f is a frequency unit)
    unit_x: str = "mpc"  # e.g., "mpc", "mpc/h", "kpc", "arcmin", etc.
    unit_y: str = "mpc"
    unit_f: str = "mpc"  # "mpc"/"mpc/h"/"kpc" if already physical, otherwise "mhz", "hz", or "redshift"
    ref_freq_mhz: Optional[float] = None  # angular/cosmology reference frequency (MHz)
    # First sample on the frequency grid.  If omitted, ref_freq_mhz is also
    # used as the first sample for backward compatibility.
    freq_grid_start_mhz: Optional[float] = None
    ref_redshift: Optional[float] = None  # reference redshift (for angular units or redshift axis)
    rest_freq_mhz: float = 1420.40575  # HI 21cm default
    H0: float = 67.8
    Om0: float = 0.308
    Ode0: Optional[float] = None
    freq_axis: int = 0
    nbins_1d: int = 30
    nbins_kperp: int = 30
    nbins_kpar: int = 30
    output_dir: str = "powerspec"
    stat_mode: str = "median"  # "median" (default, robust) or "mean"
    log_bins_2d: bool = True  # use log-spaced k_perp/k_par bins for 2D spectra
    log_power_2d: bool = True  # plot 2D power spectra in log10 scale
    demean_mode: str = "global"  # "global", "per_freq_spatial", or "none"
    # Optional EoR-window evaluation on 2D power spectra (cylindrical).
    eor_window_enabled: bool = False
    eor_window_kpar_min: float = 0.0  # require k_par >= this
    eor_window_wedge_slope: float = 0.0  # wedge line: k_par >= slope * k_perp + intercept
    eor_window_wedge_intercept: float = 0.0
    eor_window_kperp_min: Optional[float] = None
    eor_window_kperp_max: Optional[float] = None
    eor_window_kpar_max: Optional[float] = None
    eor_window_exclude_dc: bool = True  # drop k_perp=0 or k_par=0 bins
    # "center" preserves the historical bin-center mask.  The other policies
    # use the exact Fourier modes contributing to each cylindrical bin.
    eor_window_bin_policy: str = "center"  # center, all_modes, any_modes, majority
    eor_window_eps: float = 1e-20  # numerical floor for log ratios
    # Optional multi-profile EoR-window evaluation.
    # Example:
    #   "eor_window_profiles": {
    #     "loose": {"kpar_min": 0.4, "wedge_slope": 0.0},
    #     "strict": {"kpar_min": 0.6, "wedge_slope": 0.5}
    #   }
    eor_window_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    # Which profile is used as the primary one for backward-compatible top-level "metrics".
    eor_window_ranking_profile: str = "default"
    # Optional soft-window (taper) diagnostics.
    eor_window_soft_enabled: bool = False
    eor_window_soft_transition: float = 0.05  # transition width in 1/Mpc


def compute_eor_window_mask(
    kperp_centers: np.ndarray,
    kpar_centers: np.ndarray,
    cfg: PowerSpecConfig,
) -> np.ndarray:
    return compute_eor_window_mask_from_params(
        kperp_centers,
        kpar_centers,
        kpar_min=float(cfg.eor_window_kpar_min),
        wedge_slope=float(cfg.eor_window_wedge_slope),
        wedge_intercept=float(cfg.eor_window_wedge_intercept),
        kperp_min=None if cfg.eor_window_kperp_min is None else float(cfg.eor_window_kperp_min),
        kperp_max=None if cfg.eor_window_kperp_max is None else float(cfg.eor_window_kperp_max),
        kpar_max=None if cfg.eor_window_kpar_max is None else float(cfg.eor_window_kpar_max),
        exclude_dc=bool(cfg.eor_window_exclude_dc),
    )


def compute_eor_window_mask_from_params(
    kperp_centers: np.ndarray,
    kpar_centers: np.ndarray,
    *,
    kpar_min: float,
    wedge_slope: float,
    wedge_intercept: float,
    kperp_min: Optional[float] = None,
    kperp_max: Optional[float] = None,
    kpar_max: Optional[float] = None,
    exclude_dc: bool = True,
) -> np.ndarray:
    kperp = np.asarray(kperp_centers, dtype=float).reshape(-1)
    kpar = np.asarray(kpar_centers, dtype=float).reshape(-1)
    kperp_grid = kperp[:, None]
    kpar_grid = kpar[None, :]
    return compute_eor_window_mode_mask_from_params(
        kperp_grid,
        kpar_grid,
        kpar_min=kpar_min,
        wedge_slope=wedge_slope,
        wedge_intercept=wedge_intercept,
        kperp_min=kperp_min,
        kperp_max=kperp_max,
        kpar_max=kpar_max,
        exclude_dc=exclude_dc,
    )


def compute_eor_window_mode_mask_from_params(
    kperp: np.ndarray,
    kpar: np.ndarray,
    *,
    kpar_min: float,
    wedge_slope: float,
    wedge_intercept: float,
    kperp_min: Optional[float] = None,
    kperp_max: Optional[float] = None,
    kpar_max: Optional[float] = None,
    exclude_dc: bool = True,
) -> np.ndarray:
    """Evaluate an EoR-window profile on broadcast-compatible Fourier modes."""
    kperp_arr, kpar_arr = np.broadcast_arrays(
        np.asarray(kperp, dtype=float), np.asarray(kpar, dtype=float)
    )
    slope = float(wedge_slope)
    intercept = float(wedge_intercept)
    kpar_min_val = float(kpar_min)
    if not np.isfinite([slope, intercept, kpar_min_val]).all():
        raise ValueError("EoR window params must be finite.")

    mask = (kpar_arr >= kpar_min_val) & (
        kpar_arr >= slope * kperp_arr + intercept
    )
    if kperp_min is not None:
        mask &= kperp_arr >= float(kperp_min)
    if kperp_max is not None:
        mask &= kperp_arr <= float(kperp_max)
    if kpar_max is not None:
        mask &= kpar_arr <= float(kpar_max)
    if bool(exclude_dc):
        mask &= (kperp_arr > 0.0) & (kpar_arr > 0.0)
    return mask


def select_eor_window_bins(
    center_mask: np.ndarray,
    mode_fraction: Optional[np.ndarray],
    policy: str,
) -> np.ndarray:
    """Convert per-bin mode support into a hard evaluation mask."""
    center = np.asarray(center_mask, dtype=bool)
    policy_name = str(policy).strip().lower()
    if policy_name == "center":
        return center
    if mode_fraction is None:
        raise ValueError(f"EoR window policy '{policy_name}' requires mode fractions")
    fraction = np.asarray(mode_fraction, dtype=float)
    if fraction.shape != center.shape:
        raise ValueError("EoR-window mode fractions and center mask must match")
    if policy_name == "all_modes":
        return fraction >= 1.0 - 1e-12
    if policy_name == "any_modes":
        return fraction > 0.0
    if policy_name == "majority":
        return fraction >= 0.5
    raise ValueError(
        "eor_window_bin_policy must be 'center', 'all_modes', 'any_modes', or 'majority'"
    )


def compute_eor_window_mask_for_result(
    power_result: Dict[str, Any],
    cfg: PowerSpecConfig,
    profile: str = "default",
) -> np.ndarray:
    """Resolve the configured hard mask using a computed spectrum's mode support."""
    profiles = resolve_eor_window_profiles(cfg)
    profile_name = str(profile).strip() or "default"
    if profile_name not in profiles:
        raise KeyError(f"Unknown EoR-window profile: {profile_name}")
    params = profiles[profile_name]
    center_mask = compute_eor_window_mask_from_params(
        power_result["kperp_centers"],
        power_result["kpar_centers"],
        kpar_min=float(params["kpar_min"]),
        wedge_slope=float(params["wedge_slope"]),
        wedge_intercept=float(params["wedge_intercept"]),
        kperp_min=params.get("kperp_min"),
        kperp_max=params.get("kperp_max"),
        kpar_max=params.get("kpar_max"),
        exclude_dc=bool(params.get("exclude_dc", True)),
    )
    fractions = power_result.get("eor_window_mode_fractions", {})
    mode_fraction = fractions.get(profile_name) if isinstance(fractions, dict) else None
    return select_eor_window_bins(
        center_mask,
        mode_fraction,
        str(params.get("bin_policy", "center")),
    )


def compute_eor_window_soft_weights(
    kperp_centers: np.ndarray,
    kpar_centers: np.ndarray,
    *,
    kpar_min: float,
    wedge_slope: float,
    wedge_intercept: float,
    transition: float,
    kperp_min: Optional[float] = None,
    kperp_max: Optional[float] = None,
    kpar_max: Optional[float] = None,
    exclude_dc: bool = True,
) -> np.ndarray:
    kperp = np.asarray(kperp_centers, dtype=float).reshape(-1)
    kpar = np.asarray(kpar_centers, dtype=float).reshape(-1)
    kperp_grid = kperp[:, None]
    kpar_grid = kpar[None, :]
    tau = float(transition)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("Soft-window transition must be a finite positive value.")

    wedge_line = float(wedge_slope) * kperp_grid + float(wedge_intercept)
    w_kpar_min = 1.0 / (1.0 + np.exp(-(kpar_grid - float(kpar_min)) / tau))
    w_wedge = 1.0 / (1.0 + np.exp(-(kpar_grid - wedge_line) / tau))
    w = w_kpar_min * w_wedge
    if kperp_min is not None:
        w *= 1.0 / (1.0 + np.exp(-(kperp_grid - float(kperp_min)) / tau))
    if kperp_max is not None:
        w *= 1.0 / (1.0 + np.exp(-(float(kperp_max) - kperp_grid) / tau))
    if kpar_max is not None:
        w *= 1.0 / (1.0 + np.exp(-(float(kpar_max) - kpar_grid) / tau))
    if bool(exclude_dc):
        hard = (kperp_grid > 0.0) & (kpar_grid > 0.0)
        w *= hard.astype(np.float64)
    return np.clip(w, 0.0, 1.0)


def compute_power2d_window_metrics(
    p2d_rec: np.ndarray,
    p2d_true: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float,
) -> Dict[str, float]:
    eps_val = float(eps)
    if not math.isfinite(eps_val) or eps_val <= 0.0:
        raise ValueError("eps must be a finite positive value.")
    rec = np.asarray(p2d_rec, dtype=float)
    tru = np.asarray(p2d_true, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if rec.shape != tru.shape or rec.shape != m.shape:
        raise ValueError("p2d_rec/p2d_true/mask must have the same shape.")

    # Exclude empty bins (p2d=0 by construction when a bin has no contributing modes).
    valid = m & np.isfinite(rec) & np.isfinite(tru) & (rec > 0.0) & (tru > 0.0)
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n_bins": 0.0,
            "log10_mad": float("nan"),
            "log10_p90_abs": float("nan"),
            "log10_rmse": float("nan"),
            "log10_corr": float("nan"),
            "log10_rank_corr": float("nan"),
            "log10_shape_rmse": float("nan"),
            "power_sum_ratio": float("nan"),
        }

    log_rec = np.log10(rec[valid] + eps_val)
    log_tru = np.log10(tru[valid] + eps_val)
    dlog = log_rec - log_tru
    abs_dlog = np.abs(dlog)

    # Robust primary metric: median absolute log10 ratio.
    mad = float(np.median(abs_dlog))
    p90 = float(np.percentile(abs_dlog, 90))
    rmse = float(np.sqrt(np.mean(dlog**2)))

    # Correlation on log power inside window (scale-insensitive).
    if np.std(log_rec) < 1e-12 or np.std(log_tru) < 1e-12:
        corr = float("nan")
        shape_rmse = float("nan")
    else:
        corr = float(np.corrcoef(log_rec, log_tru)[0, 1])
        z_rec = (log_rec - np.mean(log_rec)) / np.std(log_rec)
        z_tru = (log_tru - np.mean(log_tru)) / np.std(log_tru)
        shape_rmse = float(np.sqrt(np.mean((z_rec - z_tru) ** 2)))

    # Rank correlation is an additional shape-sensitive metric that is less
    # affected by monotonic remapping of amplitudes.
    if log_rec.size < 2:
        rank_corr = float("nan")
    else:
        rx = np.argsort(np.argsort(log_rec))
        ry = np.argsort(np.argsort(log_tru))
        if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
            rank_corr = float("nan")
        else:
            rank_corr = float(np.corrcoef(rx, ry)[0, 1])

    sum_ratio = float(np.sum(rec[valid]) / (np.sum(tru[valid]) + eps_val))
    return {
        "n_bins": float(n),
        "log10_mad": mad,
        "log10_p90_abs": p90,
        "log10_rmse": rmse,
        "log10_corr": corr,
        "log10_rank_corr": rank_corr,
        "log10_shape_rmse": shape_rmse,
        "power_sum_ratio": sum_ratio,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    qv = float(q)
    if not math.isfinite(qv) or qv < 0.0 or qv > 1.0:
        raise ValueError("Weighted quantile q must be in [0,1].")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    wsum = float(np.sum(w))
    if not math.isfinite(wsum) or wsum <= 0.0:
        return float("nan")
    cdf = np.cumsum(w) / wsum
    idx = int(np.searchsorted(cdf, qv, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _aggregate_binned_stat(
    values: np.ndarray,
    bin_idx: np.ndarray,
    nbins: int,
    stat_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros(int(nbins), dtype=float)
    counts = np.zeros(int(nbins), dtype=np.int32)

    vals = np.asarray(values, dtype=float).reshape(-1)
    idx = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    valid = np.isfinite(vals) & (idx >= 0) & (idx < int(nbins))
    if not np.any(valid):
        return out, counts

    vals = vals[valid]
    idx = idx[valid]

    if stat_mode == "mean":
        sums = np.bincount(idx, weights=vals, minlength=int(nbins))
        counts = np.bincount(idx, minlength=int(nbins)).astype(np.int32, copy=False)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
        return out, counts

    if stat_mode != "median":
        raise ValueError("stat_mode must be 'median' or 'mean'.")

    order = np.argsort(idx, kind="mergesort")
    idx_sorted = idx[order]
    vals_sorted = vals[order]
    unique_bins, start_idx, unique_counts = np.unique(
        idx_sorted,
        return_index=True,
        return_counts=True,
    )
    counts[unique_bins] = unique_counts.astype(np.int32, copy=False)
    for bin_id, start, count in zip(unique_bins, start_idx, unique_counts):
        out[int(bin_id)] = float(np.median(vals_sorted[start:start + count]))
    return out, counts


def compute_power2d_window_metrics_weighted(
    p2d_rec: np.ndarray,
    p2d_true: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float,
) -> Dict[str, float]:
    eps_val = float(eps)
    if not math.isfinite(eps_val) or eps_val <= 0.0:
        raise ValueError("eps must be a finite positive value.")
    rec = np.asarray(p2d_rec, dtype=float)
    tru = np.asarray(p2d_true, dtype=float)
    w = np.asarray(weights, dtype=float)
    if rec.shape != tru.shape or rec.shape != w.shape:
        raise ValueError("p2d_rec/p2d_true/weights must have the same shape.")
    valid = (
        np.isfinite(rec)
        & np.isfinite(tru)
        & np.isfinite(w)
        & (rec > 0.0)
        & (tru > 0.0)
        & (w > 0.0)
    )
    n = int(np.sum(valid))
    if n == 0:
        return {
            "n_bins": 0.0,
            "weight_sum": 0.0,
            "log10_mad": float("nan"),
            "log10_p90_abs": float("nan"),
            "log10_rmse": float("nan"),
            "log10_corr": float("nan"),
            "log10_rank_corr": float("nan"),
            "log10_shape_rmse": float("nan"),
            "power_sum_ratio": float("nan"),
        }
    log_rec = np.log10(rec[valid] + eps_val)
    log_tru = np.log10(tru[valid] + eps_val)
    dlog = log_rec - log_tru
    abs_dlog = np.abs(dlog)
    wv = w[valid]
    wsum = float(np.sum(wv))
    if not math.isfinite(wsum) or wsum <= 0.0:
        return {
            "n_bins": float(n),
            "weight_sum": 0.0,
            "log10_mad": float("nan"),
            "log10_p90_abs": float("nan"),
            "log10_rmse": float("nan"),
            "log10_corr": float("nan"),
            "log10_rank_corr": float("nan"),
            "log10_shape_rmse": float("nan"),
            "power_sum_ratio": float("nan"),
        }
    wn = wv / wsum
    mad = _weighted_quantile(abs_dlog, wn, 0.5)
    p90 = _weighted_quantile(abs_dlog, wn, 0.9)
    rmse = float(math.sqrt(float(np.sum(wn * (dlog**2)))))
    mu_x = float(np.sum(wn * log_rec))
    mu_y = float(np.sum(wn * log_tru))
    dx = log_rec - mu_x
    dy = log_tru - mu_y
    vx = float(np.sum(wn * (dx**2)))
    vy = float(np.sum(wn * (dy**2)))
    if vx <= 1e-20 or vy <= 1e-20:
        corr = float("nan")
        shape_rmse = float("nan")
    else:
        cov = float(np.sum(wn * dx * dy))
        corr = float(cov / math.sqrt(vx * vy))
        z_rec = dx / math.sqrt(vx)
        z_tru = dy / math.sqrt(vy)
        shape_rmse = float(math.sqrt(float(np.sum(wn * ((z_rec - z_tru) ** 2)))))
    if log_rec.size < 2:
        rank_corr = float("nan")
    else:
        rx = np.argsort(np.argsort(log_rec))
        ry = np.argsort(np.argsort(log_tru))
        if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
            rank_corr = float("nan")
        else:
            rank_corr = float(np.corrcoef(rx, ry)[0, 1])
    sum_ratio = float(np.sum(wv * rec[valid]) / (np.sum(wv * tru[valid]) + eps_val))
    return {
        "n_bins": float(n),
        "weight_sum": float(wsum),
        "log10_mad": mad,
        "log10_p90_abs": p90,
        "log10_rmse": rmse,
        "log10_corr": corr,
        "log10_rank_corr": rank_corr,
        "log10_shape_rmse": shape_rmse,
        "power_sum_ratio": sum_ratio,
    }


def compute_wedge_leakage_metrics(
    p2d_rec: np.ndarray,
    p2d_true: np.ndarray,
    window_mask: np.ndarray,
    *,
    eps: float,
) -> Dict[str, float]:
    """
    Compute wedge-to-window leakage ratios.

    Here wedge is the complement of the EoR window in the 2D (k_perp, k_par)
    grid. These metrics are evaluated on positive, finite power bins only.
    """
    eps_val = float(eps)
    if not math.isfinite(eps_val) or eps_val <= 0.0:
        raise ValueError("eps must be a finite positive value.")
    rec = np.asarray(p2d_rec, dtype=float)
    tru = np.asarray(p2d_true, dtype=float)
    m = np.asarray(window_mask, dtype=bool)
    if rec.shape != tru.shape or rec.shape != m.shape:
        raise ValueError("p2d_rec/p2d_true/window_mask must have the same shape.")

    wedge = ~m
    rec_valid = np.isfinite(rec) & (rec > 0.0)
    tru_valid = np.isfinite(tru) & (tru > 0.0)

    rec_win = m & rec_valid
    rec_wedge = wedge & rec_valid
    tru_win = m & tru_valid
    tru_wedge = wedge & tru_valid

    rec_win_sum = float(np.sum(rec[rec_win])) if np.any(rec_win) else float("nan")
    rec_wedge_sum = float(np.sum(rec[rec_wedge])) if np.any(rec_wedge) else float("nan")
    tru_win_sum = float(np.sum(tru[tru_win])) if np.any(tru_win) else float("nan")
    tru_wedge_sum = float(np.sum(tru[tru_wedge])) if np.any(tru_wedge) else float("nan")

    rec_ratio = (
        float(rec_wedge_sum / (rec_win_sum + eps_val))
        if math.isfinite(rec_wedge_sum) and math.isfinite(rec_win_sum)
        else float("nan")
    )
    tru_ratio = (
        float(tru_wedge_sum / (tru_win_sum + eps_val))
        if math.isfinite(tru_wedge_sum) and math.isfinite(tru_win_sum)
        else float("nan")
    )
    if math.isfinite(rec_ratio) and math.isfinite(tru_ratio):
        excess = float(rec_ratio / (tru_ratio + eps_val))
    else:
        excess = float("nan")

    if (
        math.isfinite(rec_wedge_sum)
        and math.isfinite(tru_wedge_sum)
        and rec_wedge_sum > 0.0
        and tru_wedge_sum > 0.0
    ):
        wedge_log10_abs = float(abs(math.log10((rec_wedge_sum + eps_val) / (tru_wedge_sum + eps_val))))
    else:
        wedge_log10_abs = float("nan")

    return {
        "wedge_n_bins": float(int(np.sum(wedge))),
        "wedge_rec_power_sum": rec_wedge_sum,
        "wedge_true_power_sum": tru_wedge_sum,
        "window_rec_power_sum": rec_win_sum,
        "window_true_power_sum": tru_win_sum,
        "wedge_leakage_ratio": rec_ratio,
        "wedge_true_leakage_ratio": tru_ratio,
        "wedge_leakage_excess_ratio": excess,
        "wedge_power_log10_abs_ratio": wedge_log10_abs,
    }


def compute_transfer_function_metrics(
    p2d_rec: np.ndarray,
    p2d_true: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float,
) -> Dict[str, float]:
    """
    Compute transfer-function diagnostics in the EoR window.
    """
    eps_val = float(eps)
    if not math.isfinite(eps_val) or eps_val <= 0.0:
        raise ValueError("eps must be a finite positive value.")
    rec = np.asarray(p2d_rec, dtype=float)
    tru = np.asarray(p2d_true, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if rec.shape != tru.shape or rec.shape != m.shape:
        raise ValueError("p2d_rec/p2d_true/mask must have the same shape.")

    valid = m & np.isfinite(rec) & np.isfinite(tru) & (rec > 0.0) & (tru > 0.0)
    n = int(np.sum(valid))
    if n == 0:
        return {
            "transfer_n_bins": 0.0,
            "transfer_ratio_median": float("nan"),
            "transfer_ratio_p16": float("nan"),
            "transfer_ratio_p84": float("nan"),
            "transfer_log10_bias": float("nan"),
            "transfer_log10_mad": float("nan"),
            "transfer_log10_rmse": float("nan"),
        }

    ratio = rec[valid] / (tru[valid] + eps_val)
    log_ratio = np.log10(ratio + eps_val)
    abs_log_ratio = np.abs(log_ratio)

    return {
        "transfer_n_bins": float(n),
        "transfer_ratio_median": float(np.median(ratio)),
        "transfer_ratio_p16": float(np.percentile(ratio, 16)),
        "transfer_ratio_p84": float(np.percentile(ratio, 84)),
        "transfer_log10_bias": float(np.mean(log_ratio)),
        "transfer_log10_mad": float(np.median(abs_log_ratio)),
        "transfer_log10_rmse": float(np.sqrt(np.mean(log_ratio**2))),
    }


def resolve_eor_window_profiles(cfg: PowerSpecConfig) -> Dict[str, Dict[str, Any]]:
    base = {
        "kpar_min": float(cfg.eor_window_kpar_min),
        "wedge_slope": float(cfg.eor_window_wedge_slope),
        "wedge_intercept": float(cfg.eor_window_wedge_intercept),
        "kperp_min": None if cfg.eor_window_kperp_min is None else float(cfg.eor_window_kperp_min),
        "kperp_max": None if cfg.eor_window_kperp_max is None else float(cfg.eor_window_kperp_max),
        "kpar_max": None if cfg.eor_window_kpar_max is None else float(cfg.eor_window_kpar_max),
        "exclude_dc": bool(cfg.eor_window_exclude_dc),
        "bin_policy": str(cfg.eor_window_bin_policy).strip().lower(),
    }
    out: Dict[str, Dict[str, Any]] = {"default": dict(base)}
    raw = cfg.eor_window_profiles
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if not isinstance(val, dict):
            continue
        cur = dict(base)
        alias_map = {
            "kpar_min": ("kpar_min", "eor_window_kpar_min"),
            "wedge_slope": ("wedge_slope", "eor_window_wedge_slope"),
            "wedge_intercept": ("wedge_intercept", "eor_window_wedge_intercept"),
            "kperp_min": ("kperp_min", "eor_window_kperp_min"),
            "kperp_max": ("kperp_max", "eor_window_kperp_max"),
            "kpar_max": ("kpar_max", "eor_window_kpar_max"),
            "exclude_dc": ("exclude_dc", "eor_window_exclude_dc"),
            "bin_policy": ("bin_policy", "eor_window_bin_policy"),
        }
        for canonical, aliases in alias_map.items():
            for alias in aliases:
                if alias in val:
                    cur[canonical] = val[alias]
                    break
        if cur.get("kperp_min") is not None:
            cur["kperp_min"] = float(cur["kperp_min"])
        if cur.get("kperp_max") is not None:
            cur["kperp_max"] = float(cur["kperp_max"])
        if cur.get("kpar_max") is not None:
            cur["kpar_max"] = float(cur["kpar_max"])
        cur["kpar_min"] = float(cur["kpar_min"])
        cur["wedge_slope"] = float(cur["wedge_slope"])
        cur["wedge_intercept"] = float(cur["wedge_intercept"])
        cur["exclude_dc"] = bool(cur["exclude_dc"])
        cur["bin_policy"] = str(cur["bin_policy"]).strip().lower()
        out[name] = cur
    return out

def _cosmo_from_cfg(cfg: PowerSpecConfig) -> FlatLambdaCDM:
    """
    Build a flat LambdaCDM cosmology from configuration.
    Astropy's FlatLambdaCDM enforces flatness via Om0; Ode0 is implied as 1-Om0.
    """
    if cfg.Ode0 is not None:
        implied = 1.0 - cfg.Om0
        if abs(cfg.Ode0 - implied) / (abs(implied) + EPS_STD) > 0.05:
            print(
                "Warning: Ode0 is ignored by FlatLambdaCDM; using a flat model with "
                f"Om0={cfg.Om0:.3f} (implied Ode0={implied:.3f})."
            )
    return FlatLambdaCDM(H0=cfg.H0 * u.km / u.s / u.Mpc, Om0=cfg.Om0)


def _resolve_length_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert length-like spacings to Mpc.
    Supports Mpc, Mpc/h, kpc, kpc/h, Gpc, and Gpc/h.
    """
    unit_low = unit.lower()
    h = cfg.H0 / 100.0
    if unit_low == "mpc":
        return value
    if unit_low == "mpc/h":
        return value / h
    if unit_low == "kpc":
        return value / 1e3
    if unit_low == "kpc/h":
        return (value / 1e3) / h
    if unit_low == "gpc":
        return value * 1e3
    if unit_low == "gpc/h":
        return (value * 1e3) / h
    raise ValueError(
        f"Unsupported length unit '{unit}'. Use 'mpc', 'mpc/h', 'kpc', 'kpc/h', 'gpc', or 'gpc/h'."
    )


def _resolve_angular_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert angular spacing on the sky to Mpc using a reference redshift.
    """
    unit_low = unit.lower()
    if unit_low == "rad":
        angle_rad = value
    elif unit_low == "deg":
        angle_rad = math.radians(value)
    elif unit_low == "arcmin":
        angle_rad = math.radians(value / 60.0)
    elif unit_low == "arcsec":
        angle_rad = math.radians(value / 3600.0)
    else:
        raise ValueError(
            f"Unsupported angular unit '{unit}'. Use 'rad', 'deg', 'arcmin', or 'arcsec'."
        )

    cosmo = _cosmo_from_cfg(cfg)
    if cfg.ref_freq_mhz is not None:
        z_ref = (cfg.rest_freq_mhz / cfg.ref_freq_mhz) - 1.0
    elif cfg.ref_redshift is not None:
        z_ref = cfg.ref_redshift
    else:
        raise ValueError(
            "ref_freq_mhz or ref_redshift is required when using angular units for dx/dy."
        )
    chi = cosmo.comoving_distance(z_ref).to(u.Mpc).value
    return float(chi * angle_rad)


def _resolve_spatial_spacing(value: float, unit: str, cfg: PowerSpecConfig) -> float:
    """
    Convert a spatial spacing along x or y to Mpc, supporting both length and angular units.
    """
    unit_low = unit.lower()
    if unit_low in {"mpc", "mpc/h", "kpc", "kpc/h", "gpc", "gpc/h"}:
        return _resolve_length_spacing(value, unit_low, cfg)
    if unit_low in {"rad", "deg", "arcmin", "arcsec"}:
        return _resolve_angular_spacing(value, unit_low, cfg)
    raise ValueError(
        f"Unsupported spatial unit '{unit}'. Use length units "
        "('mpc', 'mpc/h', 'kpc', 'kpc/h', 'gpc', 'gpc/h') or angular units "
        "('rad', 'deg', 'arcmin', 'arcsec')."
    )


def _frequency_spacing_to_mpc(cfg: PowerSpecConfig, nf: int) -> float:
    """
    Convert spacing along the radial axis to Mpc.
    Supports:
      - Length units: mpc, mpc/h, kpc, kpc/h, gpc, gpc/h
      - Frequency units: mhz, hz (needs ref_freq_mhz and rest_freq_mhz)
      - Redshift units: redshift, z (needs ref_redshift)
    """
    unit_low = cfg.unit_f.lower()
    if unit_low in {"mpc", "mpc/h", "kpc", "kpc/h", "gpc", "gpc/h"}:
        return _resolve_length_spacing(cfg.df, unit_low, cfg)

    if unit_low in {"mhz", "hz"}:
        if cfg.ref_freq_mhz is None:
            raise ValueError("ref_freq_mhz is required when unit_f is a frequency unit.")

        cosmo = _cosmo_from_cfg(cfg)
        grid_start_mhz = (
            cfg.ref_freq_mhz
            if cfg.freq_grid_start_mhz is None
            else cfg.freq_grid_start_mhz
        )
        ref_freq = float(grid_start_mhz) * u.MHz
        df = cfg.df * (u.MHz if unit_low == "mhz" else u.Hz)
        freqs = ref_freq + np.arange(nf) * df
        z = (cfg.rest_freq_mhz * u.MHz / freqs - 1.0).decompose()
        chi = cosmo.comoving_distance(z).to(u.Mpc).value
        dchi = np.diff(chi)
        dchi_abs = np.abs(dchi)
        dchi_mean = float(np.mean(dchi_abs))
        if np.std(dchi_abs) / (np.abs(dchi_mean) + EPS_STD) > 0.05:
            print(
                "Warning: frequency spacing is not uniform in comoving distance; using mean spacing."
            )
        return dchi_mean

    if unit_low in {"redshift", "z"}:
        if cfg.ref_redshift is None:
            raise ValueError("ref_redshift is required when unit_f is a redshift unit.")
        cosmo = _cosmo_from_cfg(cfg)
        z0 = cfg.ref_redshift
        zs = z0 + np.arange(nf) * cfg.df
        chi = cosmo.comoving_distance(zs).to(u.Mpc).value
        dchi = np.diff(chi)
        dchi_abs = np.abs(dchi)
        dchi_mean = float(np.mean(dchi_abs))
        if np.std(dchi_abs) / (np.abs(dchi_mean) + EPS_STD) > 0.05:
            print(
                "Warning: redshift spacing is not uniform in comoving distance; using mean spacing."
            )
        return dchi_mean

    raise ValueError(
        f"Unsupported radial unit '{cfg.unit_f}'. Use 'mpc', 'mpc/h', 'kpc', 'kpc/h', "
        "'gpc', 'gpc/h', 'mhz', 'hz', or 'redshift'."
    )


def _compute_k_axes(shape: Tuple[int, int, int], dx: float, dy: float, df: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nf, nx, ny = shape
    kf = 2 * math.pi * np.fft.fftfreq(nf, d=df)
    kx = 2 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * math.pi * np.fft.fftfreq(ny, d=dy)
    return kf, kx, ky


def _apply_demean_mode_np(cube: np.ndarray, mode: str) -> np.ndarray:
    mode_low = str(mode).strip().lower()
    if mode_low in {"global", "global_demean"}:
        return cube - np.float32(cube.mean(dtype=np.float64))
    if mode_low in {"per_freq_spatial", "per_freq_demean"}:
        means = cube.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        return cube - means
    if mode_low == "none":
        return cube
    raise ValueError(
        f"Unsupported demean_mode '{mode}'. Use 'global', 'per_freq_spatial', or 'none'."
    )


def _apply_demean_mode_torch(cube: "torch.Tensor", mode: str) -> "torch.Tensor":
    mode_low = str(mode).strip().lower()
    if mode_low in {"global", "global_demean"}:
        return cube - cube.mean()
    if mode_low in {"per_freq_spatial", "per_freq_demean"}:
        return cube - cube.mean(dim=(1, 2), keepdim=True)
    if mode_low == "none":
        return cube
    raise ValueError(
        f"Unsupported demean_mode '{mode}'. Use 'global', 'per_freq_spatial', or 'none'."
    )


def _apply_uv_mask(power: np.ndarray, config: PowerSpecConfig) -> np.ndarray:
    """
    Placeholder for applying a uv / PSF / visibility mask in (kx, ky).

    Currently this is a no-op and simply returns the input power cube.
    In future, when working with interferometric visibilities or an
    explicit PSF model, this hook can be used to zero out unsampled
    transverse modes before cylindrical averaging.
    """
    # Example future extension:
    # - Add uvmin/uvmax fields to PowerSpecConfig (in meters or wavelengths)
    # - Convert them to k_perp ranges using the cosmology
    # - Construct a boolean mask in (kx, ky) and apply it here.
    return power


def compute_power_spectra(
    cube: np.ndarray,
    config: PowerSpecConfig,
    window: str = "hann",
) -> Dict[str, Any]:
    """
    Compute 1D (spherical) and 2D (kperp, kpar) power spectra for a 3D cube.

    For the 1D spectrum, only Fourier modes inside the inscribed k-space
    sphere (i.e., where all three dimensions have support) are used when
    forming averages. For the 2D spectrum, the full line-of-sight range
    is retained while transverse modes are restricted to the inscribed
    circle in (kx, ky). By default, 2D k-bins are logarithmically spaced
    in both k_perp and k_par.
    """
    use_torch = _HAS_TORCH and isinstance(cube, torch.Tensor)  # type: ignore[name-defined]

    if use_torch:
        # Torch/GPU path: keep FFT on the tensor device, then move power to CPU.
        tensor = cube  # type: ignore[assignment]
        cube_reorder_t = tensor.movedim(config.freq_axis, 0)
        nf, nx, ny = cube_reorder_t.shape
        cube_demean_t = _apply_demean_mode_torch(cube_reorder_t, config.demean_mode)

        if window:
            # Use non-periodic Hann windows to match np.hanning in the CPU path.
            win_f = torch.hann_window(nf, periodic=False, device=tensor.device, dtype=tensor.dtype)
            win_x = torch.hann_window(nx, periodic=False, device=tensor.device, dtype=tensor.dtype)
            win_y = torch.hann_window(ny, periodic=False, device=tensor.device, dtype=tensor.dtype)
            win3d = win_f[:, None, None] * win_x[None, :, None] * win_y[None, None, :]
            cube_demean_t = cube_demean_t * win3d
            norm = float((win3d**2).mean().item())
        else:
            norm = 1.0

        df_mpc = _frequency_spacing_to_mpc(config, nf)
        dx_mpc = _resolve_spatial_spacing(config.dx, config.unit_x, config)
        dy_mpc = _resolve_spatial_spacing(config.dy, config.unit_y, config)
        v_cell = df_mpc * dx_mpc * dy_mpc
        n_vox = float(nf * nx * ny)

        Fk_t = torch.fft.fftn(cube_demean_t)
        power = (v_cell / (norm * n_vox)) * (Fk_t.abs() ** 2)
        power = power.detach().cpu().numpy()
        cube_reorder = cube_reorder_t.detach().cpu().numpy()
    else:
        # NumPy/CPU path.
        if hasattr(cube, "detach"):
            cube = cube.detach().cpu().numpy()  # type: ignore[assignment]
        if not isinstance(cube, np.ndarray):
            cube = np.asarray(cube)
        cube_reorder = np.moveaxis(cube, config.freq_axis, 0)
        nf, nx, ny = cube_reorder.shape
        cube_demean = _apply_demean_mode_np(cube_reorder, config.demean_mode)

        if window:
            win_f = np.hanning(nf) if window == "hann" else np.ones(nf)
            win_x = np.hanning(nx) if window == "hann" else np.ones(nx)
            win_y = np.hanning(ny) if window == "hann" else np.ones(ny)
            win3d = win_f[:, None, None] * win_x[None, :, None] * win_y[None, None, :]
            cube_demean = cube_demean * win3d
            norm = np.mean(win3d**2)
        else:
            norm = 1.0

        df_mpc = _frequency_spacing_to_mpc(config, nf)
        dx_mpc = _resolve_spatial_spacing(config.dx, config.unit_x, config)
        dy_mpc = _resolve_spatial_spacing(config.dy, config.unit_y, config)
        v_cell = df_mpc * dx_mpc * dy_mpc
        n_vox = float(nf * nx * ny)

        Fk = np.fft.fftn(cube_demean)
        power = (v_cell / (norm * n_vox)) * (np.abs(Fk) ** 2)
    power = _apply_uv_mask(power, config)
    kf, kx, ky = _compute_k_axes((nf, nx, ny), dx_mpc, dy_mpc, df_mpc)
    kf_grid, kx_grid, ky_grid = np.meshgrid(kf, kx, ky, indexing="ij")
    kperp = np.sqrt(kx_grid**2 + ky_grid**2)
    kmag = np.sqrt(kperp**2 + kf_grid**2)

    # Restrict 1D averages to the inscribed k-space sphere where all
    # three dimensions have support.
    max_kf = float(np.max(np.abs(kf)))
    max_kx = float(np.max(np.abs(kx)))
    max_ky = float(np.max(np.abs(ky)))
    kmax_sphere = min(max_kf, max_kx, max_ky)
    sphere_mask = kmag <= kmax_sphere

    power_1d = power[sphere_mask].reshape(-1)
    kmag_flat = kmag[sphere_mask].reshape(-1)

    # 1D spherical average (robust or mean) with linear k bins.
    k_bins = np.linspace(0.0, kmax_sphere, config.nbins_1d + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    bin_idx_1d = np.digitize(kmag_flat, k_bins) - 1
    stat_mode = config.stat_mode.lower()
    p1d, p1d_counts = _aggregate_binned_stat(power_1d, bin_idx_1d, config.nbins_1d, stat_mode)

    # 2D (kperp, kpar) average (robust or mean).
    # Relax the inscribed-sphere constraint: keep the full k_par range,
    # but restrict transverse modes to the inscribed circle in (kx, ky)
    # so that both spatial dimensions have support.
    kmax_perp_circle = min(max_kx, max_ky)
    circle_mask = kperp <= kmax_perp_circle
    power_2d = power[circle_mask].reshape(-1)
    kperp_flat = kperp[circle_mask].reshape(-1)
    kpar_flat = np.abs(kf_grid[circle_mask]).reshape(-1)

    # Define 2D k-bins: log or linear spacing.
    if config.log_bins_2d:
        valid_kperp = kperp_flat[kperp_flat > 0]
        valid_kpar = kpar_flat[kpar_flat > 0]
        if valid_kperp.size == 0 or valid_kpar.size == 0:
            raise ValueError("No positive k_perp or k_par values available for 2D binning.")
        kperp_min = float(valid_kperp.min())
        kperp_max = float(valid_kperp.max())
        kpar_min = float(valid_kpar.min())
        kpar_max = float(valid_kpar.max())
        kperp_bins = np.logspace(
            math.log10(kperp_min), math.log10(kperp_max), config.nbins_kperp + 1
        )
        kpar_bins = np.logspace(
            math.log10(kpar_min), math.log10(kpar_max), config.nbins_kpar + 1
        )
    else:
        kperp_bins = np.linspace(0.0, kmax_perp_circle, config.nbins_kperp + 1)
        kpar_bins = np.linspace(0.0, float(np.max(kpar_flat)), config.nbins_kpar + 1)

    bin_kperp = np.digitize(kperp_flat, kperp_bins) - 1
    bin_kpar = np.digitize(kpar_flat, kpar_bins) - 1
    # Ensure k_perp = 0 and k_par = 0 modes are included in the first bin
    # when using logarithmic k-bins. This prevents entire first rows/columns
    # from being empty purely due to the placement of the log-spaced edges.
    if config.log_bins_2d:
        bin_kperp[kperp_flat == 0.0] = 0
        bin_kpar[kpar_flat == 0.0] = 0

    # Validate each cylindrical coordinate before flattening.  Flattening an
    # out-of-range k_par index directly can otherwise wrap it into the next
    # k_perp row (for example, (i, nbins_kpar) becomes (i + 1, 0)).
    valid_2d_bins = (
        (bin_kperp >= 0)
        & (bin_kperp < config.nbins_kperp)
        & (bin_kpar >= 0)
        & (bin_kpar < config.nbins_kpar)
    )
    linear_bins = np.full(bin_kperp.shape, -1, dtype=np.int64)
    linear_bins[valid_2d_bins] = (
        bin_kperp[valid_2d_bins] * config.nbins_kpar
        + bin_kpar[valid_2d_bins]
    )
    p2d_flat, p2d_counts_flat = _aggregate_binned_stat(
        power_2d,
        linear_bins,
        config.nbins_kperp * config.nbins_kpar,
        stat_mode,
    )
    p2d = p2d_flat.reshape(config.nbins_kperp, config.nbins_kpar)
    p2d_counts = p2d_counts_flat.reshape(config.nbins_kperp, config.nbins_kpar)

    eor_window_mode_fractions: Dict[str, np.ndarray] = {}
    if bool(config.eor_window_enabled):
        full_bin_count = int(config.nbins_kperp * config.nbins_kpar)
        total_counts = p2d_counts_flat.astype(np.float64, copy=False)
        for profile_name, params in resolve_eor_window_profiles(config).items():
            selected_modes = compute_eor_window_mode_mask_from_params(
                kperp_flat,
                kpar_flat,
                kpar_min=float(params["kpar_min"]),
                wedge_slope=float(params["wedge_slope"]),
                wedge_intercept=float(params["wedge_intercept"]),
                kperp_min=params.get("kperp_min"),
                kperp_max=params.get("kperp_max"),
                kpar_max=params.get("kpar_max"),
                exclude_dc=bool(params.get("exclude_dc", True)),
            )
            selected_valid = valid_2d_bins & selected_modes
            selected_counts = np.bincount(
                linear_bins[selected_valid], minlength=full_bin_count
            ).astype(np.float64, copy=False)
            fraction = np.zeros_like(total_counts, dtype=np.float64)
            populated = total_counts > 0.0
            fraction[populated] = selected_counts[populated] / total_counts[populated]
            eor_window_mode_fractions[profile_name] = fraction.reshape(
                config.nbins_kperp, config.nbins_kpar
            )

    return {
        "k_centers": k_centers,
        "k_edges": k_bins,
        "p1d": p1d,
        "p1d_counts": p1d_counts,
        "kperp_edges": kperp_bins,
        "kpar_edges": kpar_bins,
        "kperp_centers": 0.5 * (kperp_bins[:-1] + kperp_bins[1:]),
        "kpar_centers": 0.5 * (kpar_bins[:-1] + kpar_bins[1:]),
        "p2d": p2d,
        "p2d_counts": p2d_counts,
        "eor_window_mode_fractions": eor_window_mode_fractions,
    }


def _save_1d_fits(path: Path, k: np.ndarray, p: np.ndarray) -> None:
    cols = [fits.Column(name="k", format="E", array=k.astype(np.float32)),
            fits.Column(name="power", format="E", array=p.astype(np.float32))]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)


def _save_2d_fits(path: Path, power2d: np.ndarray) -> None:
    fits.PrimaryHDU(data=power2d.astype(np.float32)).writeto(path, overwrite=True)


def _plot_1d(
    path: Path,
    k: np.ndarray,
    rec: np.ndarray,
    true: Optional[np.ndarray],
    rel: Optional[np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(k, rec, label="recovered", color="C0")
    if true is not None:
        ax1.plot(k, true, label="true", color="C1", linestyle="--")
    ax1.set_xlabel("k [1/Mpc]")
    ax1.set_ylabel("P(k)")
    has_positive = np.any(rec > 0) or (true is not None and np.any(true > 0))
    if has_positive:
        ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if rel is not None:
        rel_clip = np.clip(rel, 1.0, 100.0)
        ax2 = ax1.twinx()
        ax2.plot(k, rel_clip, color="C2", alpha=0.6, label="|rel| %")
        ax2.set_yscale("log")
        ax2.set_ylim(1.0, 100.0)
        ax2.set_ylabel("Relative % (abs, log)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_2d(
    path: Path,
    kperp: np.ndarray,
    kpar: np.ndarray,
    p2d: np.ndarray,
    title: str,
    cbar_label: str,
    log_scale: bool = False,
    log_axes: bool = False,
    kperp_edges: Optional[np.ndarray] = None,
    kpar_edges: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    data = np.array(p2d, copy=True)
    if log_scale:
        # Map non-positive or empty bins to a small floor value so that
        # they appear as the darkest color rather than blank. This avoids
        # large NaN regions in the plot while preserving the dynamic range
        # of bins with actual measurements.
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = data[data > 0]
            if positive.size > 0:
                min_pos = float(np.nanmin(positive))
                floor = min_pos / 10.0
                safe = np.where(data > 0, data, floor)
                data = np.log10(safe)
            else:  # no positive values; fall back to zeros
                data = np.zeros_like(data)
    finite_vals = data[np.isfinite(data)]
    vmax = np.nanmax(finite_vals) if finite_vals.size > 0 else None
    vmin = np.nanmin(finite_vals) if finite_vals.size > 0 else None
    if kperp_edges is not None and kpar_edges is not None:
        extent = [float(kperp_edges[0]), float(kperp_edges[-1]), float(kpar_edges[0]), float(kpar_edges[-1])]
    else:
        extent = [float(kperp.min()), float(kperp.max()), float(kpar.min()), float(kpar.max())]
    im = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax if vmax is not None and np.isfinite(vmax) else None,
    )
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("k_perp [1/Mpc]")
    ax.set_ylabel("k_par [1/Mpc]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_power_outputs(
    output_dir: Path,
    rec: Dict[str, Any],
    true: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[PowerSpecConfig] = None,
    log_power_2d: Optional[bool] = None,
    log_axes_2d: Optional[bool] = None,
) -> None:
    if config is not None:
        if log_power_2d is None:
            log_power_2d = bool(config.log_power_2d)
        if log_axes_2d is None:
            log_axes_2d = bool(config.log_bins_2d)
    if log_power_2d is None:
        log_power_2d = True
    if log_axes_2d is None:
        log_axes_2d = False

    output_dir.mkdir(parents=True, exist_ok=True)
    k = rec["k_centers"]
    k_edges = rec.get("k_edges")
    p1d_rec = rec["p1d"]
    p1d_counts = rec.get("p1d_counts")
    p2d_rec = rec["p2d"]
    p2d_counts = rec.get("p2d_counts")
    eor_window_mode_fractions = rec.get("eor_window_mode_fractions", {})
    kperp = rec["kperp_centers"]
    kpar = rec["kpar_centers"]
    kperp_edges = rec.get("kperp_edges")
    kpar_edges = rec.get("kpar_edges")

    true_p1d = true["p1d"] if true is not None else None
    true_p2d = true["p2d"] if true is not None else None

    rel1d = None
    rel2d = None
    if true is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel1d = np.where(
                np.abs(true_p1d) > EPS_LOSS,
                100.0 * np.abs(p1d_rec - true_p1d) / np.abs(true_p1d),
                0.0,
            )
            rel2d = np.where(
                np.abs(true_p2d) > EPS_LOSS,
                100.0 * np.abs(p2d_rec - true_p2d) / np.abs(true_p2d),
                0.0,
            )

    _save_1d_fits(output_dir / "power1d_rec.fits", k, p1d_rec)
    if p1d_counts is not None:
        _save_1d_fits(output_dir / "power1d_counts.fits", k, np.asarray(p1d_counts, dtype=np.float32))
    _plot_1d(output_dir / "power1d.png", k, p1d_rec, true_p1d, rel1d)
    _save_2d_fits(output_dir / "power2d_rec.fits", p2d_rec)
    if p2d_counts is not None:
        _save_2d_fits(output_dir / "power2d_counts.fits", np.asarray(p2d_counts, dtype=np.float32))
    axes_payload = {
        "demean_mode": config.demean_mode if config is not None else "global",
        "k_centers": np.asarray(k, dtype=float).tolist(),
        "k_edges": None if k_edges is None else np.asarray(k_edges, dtype=float).tolist(),
        "kperp_centers": np.asarray(kperp, dtype=float).tolist(),
        "kpar_centers": np.asarray(kpar, dtype=float).tolist(),
        "kperp_edges": None if kperp_edges is None else np.asarray(kperp_edges, dtype=float).tolist(),
        "kpar_edges": None if kpar_edges is None else np.asarray(kpar_edges, dtype=float).tolist(),
    }
    (output_dir / "power_axes.json").write_text(json.dumps(axes_payload, indent=2), encoding="utf-8")
    _plot_2d(
        output_dir / "power2d.png",
        kperp,
        kpar,
        p2d_rec,
        "Recovered 2D Power",
        cbar_label="log10 P(k)" if log_power_2d else "P(k)",
        log_scale=log_power_2d,
        log_axes=log_axes_2d,
        kperp_edges=kperp_edges,
        kpar_edges=kpar_edges,
    )

    if true is not None:
        _save_1d_fits(output_dir / "power1d_true.fits", k, true_p1d)
        _save_2d_fits(output_dir / "power2d_true.fits", true_p2d)
        _plot_2d(
            output_dir / "power2d_true.png",
            kperp,
            kpar,
            true_p2d,
            "True 2D Power",
            cbar_label="log10 P(k)" if log_power_2d else "P(k)",
            log_scale=log_power_2d,
            log_axes=log_axes_2d,
            kperp_edges=kperp_edges,
            kpar_edges=kpar_edges,
        )
        if rel1d is not None:
            rel1d_clip = np.clip(rel1d, 1.0, 100.0)
            _save_1d_fits(output_dir / "power1d_rel.fits", k, rel1d_clip)
        if rel2d is not None:
            rel2d_clip = np.clip(rel2d, 1.0, 100.0)
            _save_2d_fits(output_dir / "power2d_rel.fits", rel2d_clip)
            _plot_2d(
                output_dir / "power2d_rel.png",
                kperp,
                kpar,
                rel2d_clip,
                "Relative % 2D Power (clipped, abs)",
                cbar_label="log10 |rel| %",
                log_scale=True,
                log_axes=log_axes_2d,
                kperp_edges=kperp_edges,
                kpar_edges=kpar_edges,
            )

        # Optional EoR-window metrics on 2D power spectra (inside the window only).
        if config is not None and bool(config.eor_window_enabled):
            try:
                eps_val = float(config.eor_window_eps)
                profiles = resolve_eor_window_profiles(config)
                soft_enabled = bool(config.eor_window_soft_enabled)
                soft_transition = float(config.eor_window_soft_transition)
                ranking_profile = str(config.eor_window_ranking_profile).strip() or "default"
                if ranking_profile not in profiles:
                    ranking_profile = "default"

                out_profiles: Dict[str, Dict[str, Any]] = {}
                for name, params in profiles.items():
                    center_mask = compute_eor_window_mask_from_params(
                        kperp,
                        kpar,
                        kpar_min=float(params["kpar_min"]),
                        wedge_slope=float(params["wedge_slope"]),
                        wedge_intercept=float(params["wedge_intercept"]),
                        kperp_min=params.get("kperp_min"),
                        kperp_max=params.get("kperp_max"),
                        kpar_max=params.get("kpar_max"),
                        exclude_dc=bool(params.get("exclude_dc", True)),
                    )
                    mode_fraction = (
                        eor_window_mode_fractions.get(name)
                        if isinstance(eor_window_mode_fractions, dict)
                        else None
                    )
                    bin_policy = str(params.get("bin_policy", "center"))
                    mask = select_eor_window_bins(
                        center_mask, mode_fraction, bin_policy
                    )
                    hard_metrics = compute_power2d_window_metrics(
                        p2d_rec,
                        true_p2d,
                        mask,
                        eps=eps_val,
                    )
                    hard_metrics.update(
                        compute_wedge_leakage_metrics(
                            p2d_rec,
                            true_p2d,
                            mask,
                            eps=eps_val,
                        )
                    )
                    hard_metrics.update(
                        compute_transfer_function_metrics(
                            p2d_rec,
                            true_p2d,
                            mask,
                            eps=eps_val,
                        )
                    )
                    profile_rec: Dict[str, Any] = {
                        "params": {
                            "kpar_min": float(params["kpar_min"]),
                            "wedge_slope": float(params["wedge_slope"]),
                            "wedge_intercept": float(params["wedge_intercept"]),
                            "kperp_min": params.get("kperp_min"),
                            "kperp_max": params.get("kperp_max"),
                            "kpar_max": params.get("kpar_max"),
                            "exclude_dc": bool(params.get("exclude_dc", True)),
                            "bin_policy": bin_policy,
                            "eps": eps_val,
                        },
                        "hard_metrics": hard_metrics,
                    }
                    suffix = "" if name == "default" else f"_{name}"
                    _save_2d_fits(
                        output_dir / f"power2d_eor_window_mask{suffix}.fits",
                        mask.astype(np.int16),
                    )
                    if mode_fraction is not None:
                        mode_fraction_arr = np.asarray(mode_fraction, dtype=np.float32)
                        _save_2d_fits(
                            output_dir
                            / f"power2d_eor_window_mode_fraction{suffix}.fits",
                            mode_fraction_arr,
                        )
                        populated = (
                            np.asarray(p2d_counts) > 0
                            if p2d_counts is not None
                            else np.ones_like(mode_fraction_arr, dtype=bool)
                        )
                        hard_metrics["fully_selected_bin_count"] = float(
                            np.sum(populated & (mode_fraction_arr >= 1.0 - 1e-12))
                        )
                        hard_metrics["partially_selected_bin_count"] = float(
                            np.sum(
                                populated
                                & (mode_fraction_arr > 0.0)
                                & (mode_fraction_arr < 1.0 - 1e-12)
                            )
                        )
                    dlog = np.full_like(p2d_rec, np.nan, dtype=np.float32)
                    win = mask & (p2d_rec > 0.0) & (true_p2d > 0.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dlog[win] = (
                            np.log10(p2d_rec[win] + eps_val)
                            - np.log10(true_p2d[win] + eps_val)
                        ).astype(np.float32)
                    _save_2d_fits(output_dir / f"power2d_eor_window_dlog10{suffix}.fits", dlog)

                    if soft_enabled:
                        soft_w = compute_eor_window_soft_weights(
                            kperp,
                            kpar,
                            kpar_min=float(params["kpar_min"]),
                            wedge_slope=float(params["wedge_slope"]),
                            wedge_intercept=float(params["wedge_intercept"]),
                            transition=soft_transition,
                            kperp_min=params.get("kperp_min"),
                            kperp_max=params.get("kperp_max"),
                            kpar_max=params.get("kpar_max"),
                            exclude_dc=bool(params.get("exclude_dc", True)),
                        )
                        soft_metrics = compute_power2d_window_metrics_weighted(
                            p2d_rec,
                            true_p2d,
                            soft_w,
                            eps=eps_val,
                        )
                        profile_rec["soft_metrics"] = soft_metrics
                        profile_rec["soft_transition"] = float(soft_transition)
                        _save_2d_fits(
                            output_dir / f"power2d_eor_window_soft_weights{suffix}.fits",
                            soft_w.astype(np.float32),
                        )
                    out_profiles[name] = profile_rec

                rank_rec = out_profiles[ranking_profile]
                out = {
                    "ranking_profile": ranking_profile,
                    "profiles": out_profiles,
                    # Backward compatibility fields:
                    "eor_window_params": rank_rec["params"],
                    "metrics": rank_rec["hard_metrics"],
                    "soft_metrics": rank_rec.get("soft_metrics"),
                }
                (output_dir / "power2d_eor_window_metrics.json").write_text(
                    json.dumps(out, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception as exc:
                (output_dir / "power2d_eor_window_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
