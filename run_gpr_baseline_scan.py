#!/usr/bin/env python3
"""
LOFAR-inspired GP regression (GPR) *baseline* on pure-sky cubes.

This script does NOT run iterative optimization. It performs a per-pixel 1D
Gaussian-process smoothing along the frequency axis to estimate FG, then sets:

  EoR_est = y_obs - FG_est

This is meant as a fast diagnostic baseline to answer:
  - "If we only use frequency-coherence separation, how good can we get?"

We implement the GP posterior mean with a stationary kernel K(Δν):

  FG_est = K (K + λ I)^(-1) y

where λ = σ_noise^2 / σ_fg^2 is a (dimensionless) noise-to-signal variance ratio.

Notes:
- This mirrors LOFAR GPR's line-of-sight modelling idea in the *no-instrument* stage.
- It is intentionally "FG-only": no simulation-trained 21cm kernel is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset_registry import build_datasets, filter_datasets, parse_dataset_names
from separation_optim import OptimizationConfig, build_cut_xy_indices, read_fits_cube


@dataclass(frozen=True)
class Candidate:
    mode: str  # "fg_only" or "fg_eor"
    kernel_fg: str
    l_fg_mhz: float
    # For fg_only, eor kernel/scale are unused.
    kernel_eor: str = "matern32"
    l_eor_mhz: float = 0.5
    eor_var_rel: float = 1.0  # var(K_eor) relative to var(K_fg)
    noise_lam: float = 0.01  # variance ratio for additive white noise in (K+noise I)

    @property
    def name(self) -> str:
        mode = str(self.mode).strip().lower()
        kfg = str(self.kernel_fg).strip().lower()
        lfg = float(self.l_fg_mhz)
        if mode == "fg_only":
            return f"{kfg}_l{lfg:g}_lam{float(self.noise_lam):g}"
        if mode == "fg_eor":
            ke = str(self.kernel_eor).strip().lower()
            le = float(self.l_eor_mhz)
            vr = float(self.eor_var_rel)
            nl = float(self.noise_lam)
            return f"fg_{kfg}_l{lfg:g}__eor_{ke}_l{le:g}_vr{vr:g}__n{nl:g}"
        return f"unknown_{kfg}_l{lfg:g}"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return out


def _score_from_corr(corr: np.ndarray) -> float:
    vals = corr[np.isfinite(corr)]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    return 0.7 * mean + 0.2 * median + 0.1 * p10


def _kernel_matrix(
    num_freqs: int,
    *,
    df_mhz: float,
    l_mhz: float,
    kernel: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive.")
    df = float(df_mhz)
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("df_mhz must be finite and > 0.")
    l = float(l_mhz)
    if not math.isfinite(l) or l <= 0.0:
        raise ValueError("l_mhz must be finite and > 0.")
    kname = str(kernel).strip().lower()
    if kname not in {"matern32", "rbf"}:
        raise ValueError("kernel must be one of: matern32, rbf.")

    # Build on CPU float64 for stability; then cast to target dtype/device.
    idx = torch.arange(int(num_freqs), device="cpu", dtype=torch.float64)
    dist = (idx[:, None] - idx[None, :]).abs() * df
    if kname == "matern32":
        a = math.sqrt(3.0) * dist / l
        K = (1.0 + a) * torch.exp(-a)
    else:
        z = dist / l
        K = torch.exp(-0.5 * z * z)
    return K.to(device=device, dtype=dtype)


def _gpr_filter_matrix(
    num_freqs: int,
    *,
    df_mhz: float,
    l_mhz: float,
    lam: float,
    kernel: str,
    jitter: float = 1e-6,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lam_val = float(lam)
    if not math.isfinite(lam_val) or lam_val <= 0.0:
        raise ValueError("lam must be finite and > 0 (noise-to-signal variance ratio).")
    jit = float(jitter)
    if not math.isfinite(jit) or jit <= 0.0:
        raise ValueError("jitter must be finite and > 0.")

    # Do the small (F,F) linear algebra in float64 for stability, then cast back.
    mat_dtype = torch.float64 if dtype == torch.float32 else dtype
    K = _kernel_matrix(
        num_freqs,
        df_mhz=float(df_mhz),
        l_mhz=float(l_mhz),
        kernel=kernel,
        device=device,
        dtype=mat_dtype,
    )
    eye = torch.eye(int(num_freqs), device=device, dtype=mat_dtype)
    base = K + lam_val * eye
    # M = K @ inv(A) (posterior mean linear filter)
    last_err: Optional[Exception] = None
    for _attempt in range(6):
        try:
            A = base + jit * eye
            L = torch.linalg.cholesky(A)
            Ainv = torch.cholesky_inverse(L)
            M = (K @ Ainv).to(device=device, dtype=dtype)
            return M
        except Exception as exc:  # pragma: no cover - numeric fallback
            last_err = exc
            jit *= 10.0
    raise RuntimeError(f"GPR filter Cholesky failed after retries: {last_err}")


def _gpr_two_component_filters(
    num_freqs: int,
    *,
    df_mhz: float,
    fg_kernel: str,
    fg_l_mhz: float,
    eor_kernel: str,
    eor_l_mhz: float,
    eor_var_rel: float = 1.0,
    noise_lam: float = 0.0,
    jitter: float = 1e-6,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-component decomposition y = fg + eor, with independent GP priors:
      fg ~ GP(0, K_fg), eor ~ GP(0, K_eor), (optional) noise ~ N(0, noise_lam I).

    Posterior mean linear filters (assuming shared frequency grid):
      fg_est  = K_fg  (K_fg + K_eor + noise_lam I)^(-1) y
      eor_est = K_eor (K_fg + K_eor + noise_lam I)^(-1) y
    """
    nl = float(noise_lam)
    if not math.isfinite(nl) or nl < 0.0:
        raise ValueError("noise_lam must be finite and >= 0.")
    vr = float(eor_var_rel)
    if not math.isfinite(vr) or vr < 0.0:
        raise ValueError("eor_var_rel must be finite and >= 0.")
    jit = float(jitter)
    if not math.isfinite(jit) or jit <= 0.0:
        raise ValueError("jitter must be finite and > 0.")

    mat_dtype = torch.float64 if dtype == torch.float32 else dtype
    K_fg = _kernel_matrix(
        num_freqs,
        df_mhz=float(df_mhz),
        l_mhz=float(fg_l_mhz),
        kernel=fg_kernel,
        device=device,
        dtype=mat_dtype,
    )
    K_e = _kernel_matrix(
        num_freqs,
        df_mhz=float(df_mhz),
        l_mhz=float(eor_l_mhz),
        kernel=eor_kernel,
        device=device,
        dtype=mat_dtype,
    )
    if vr != 1.0:
        K_e = K_e * vr
    eye = torch.eye(int(num_freqs), device=device, dtype=mat_dtype)
    base = K_fg + K_e + nl * eye
    last_err: Optional[Exception] = None
    for _attempt in range(6):
        try:
            C = base + jit * eye
            L = torch.linalg.cholesky(C)
            Cinv = torch.cholesky_inverse(L)
            M_fg = (K_fg @ Cinv).to(device=device, dtype=dtype)
            M_e = (K_e @ Cinv).to(device=device, dtype=dtype)
            return M_fg, M_e
        except Exception as exc:  # pragma: no cover - numeric fallback
            last_err = exc
            jit *= 10.0
    raise RuntimeError(f"GPR two-component Cholesky failed after retries: {last_err}")


def _per_frequency_corr(
    est: torch.Tensor,
    tru: torch.Tensor,
    *,
    freq_axis: int,
) -> torch.Tensor:
    if est.shape != tru.shape:
        raise ValueError(f"Shape mismatch: {tuple(est.shape)} vs {tuple(tru.shape)}")
    if est.ndim != 3:
        raise ValueError(f"Expected 3D cubes, got {tuple(est.shape)}")
    moved_est = est.movedim(freq_axis, 0).reshape(int(est.shape[freq_axis]), -1)
    moved_tru = tru.movedim(freq_axis, 0).reshape(int(tru.shape[freq_axis]), -1)
    est_c = moved_est - moved_est.mean(dim=1, keepdim=True)
    tru_c = moved_tru - moved_tru.mean(dim=1, keepdim=True)
    dot = torch.sum(est_c * tru_c, dim=1)
    denom = torch.norm(est_c, dim=1) * torch.norm(tru_c, dim=1)
    denom = torch.clamp(denom, min=1e-12)
    return dot / denom


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LOFAR-inspired GPR baseline scan (no optimization).")
    p.add_argument("--work-root", type=Path, default=Path.cwd(), help="Project root (default: cwd).")
    p.add_argument("--code-dir", type=Path, default=None, help="3dnet dir (default: <work-root>/3dnet).")
    p.add_argument("--data-dir", type=Path, default=None, help="Data dir (default: <work-root>/data).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <work-root>/runs/gpr_baseline_scan_<timestamp>).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="cube2",
        help="Comma-separated dataset names (default: cube2).",
    )
    p.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available).")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--freq-axis", type=int, default=0)
    p.add_argument("--df-mhz", type=float, default=0.1, help="Frequency spacing in MHz (default 0.1).")
    p.add_argument(
        "--decomp",
        type=str,
        default="fg_only",
        choices=["fg_only", "fg_eor"],
        help="Decomposition mode: fg_only uses a single FG kernel + white-noise residual; "
        "fg_eor uses two kernels (FG + EoR) like LOFAR's additive-kernel model.",
    )
    p.add_argument("--kernel", type=str, default="matern32", choices=["matern32", "rbf"])
    # fg_only grid:
    p.add_argument("--l-mhz-list", type=str, default="1,2,5,10,20,50", help="(fg_only) FG length scales (MHz).")
    p.add_argument("--lam-list", type=str, default="0.01,0.1,1", help="(fg_only) λ=σ_n^2/σ_fg^2 (white noise).")
    # fg_eor grid:
    p.add_argument("--l-fg-mhz-list", type=str, default="10,20,50", help="(fg_eor) FG length scales (MHz).")
    p.add_argument("--l-eor-mhz-list", type=str, default="0.1,0.2,0.5,1.0", help="(fg_eor) EoR length scales (MHz).")
    p.add_argument(
        "--eor-var-rel-list",
        type=str,
        default="1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4",
        help="(fg_eor) EoR kernel variance relative to FG kernel variance.",
    )
    p.add_argument(
        "--noise-lam-list",
        type=str,
        default="0,1e-4,1e-2",
        help="(fg_eor) additive white-noise variance ratio in C = K_fg + K_eor + noise_lam I.",
    )
    p.add_argument("--jitter", type=float, default=1e-6, help="Diagonal jitter added to (K+λI).")
    p.add_argument("--chunk-pixels", type=int, default=0, help="Optional pixel chunk size for matmul (0=all).")
    p.add_argument("--write-fits", action="store_true", help="Write fg_est/eor_est FITS for each candidate.")
    p.add_argument("--cut-size-frac", type=float, default=0.30, help="Spatial crop size fraction (default 0.30).")
    p.add_argument("--cut-center-x", type=float, default=0.5)
    p.add_argument("--cut-center-y", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    work_root = Path(args.work_root).resolve()
    code_dir = (Path(args.code_dir) if args.code_dir else work_root / "3dnet").resolve()
    data_dir = (Path(args.data_dir) if args.data_dir else work_root / "data").resolve()
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (work_root / "runs" / f"gpr_baseline_scan_{_now_tag()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    freq_axis = int(args.freq_axis)

    datasets = build_datasets(data_dir)
    enabled = parse_dataset_names(args.datasets)
    datasets = filter_datasets(datasets, enabled)
    if not datasets:
        raise SystemExit("No datasets selected.")

    decomp_mode = str(args.decomp).strip().lower()
    kernel = str(args.kernel).strip().lower()

    candidates: List[Candidate] = []
    if decomp_mode == "fg_only":
        l_list = _parse_float_list(args.l_mhz_list)
        lam_list = _parse_float_list(args.lam_list)
        for l in l_list:
            for lam in lam_list:
                candidates.append(
                    Candidate(
                        mode="fg_only",
                        kernel_fg=kernel,
                        l_fg_mhz=float(l),
                        noise_lam=float(lam),
                    )
                )
    else:
        l_fg_list = _parse_float_list(args.l_fg_mhz_list)
        l_eor_list = _parse_float_list(args.l_eor_mhz_list)
        vr_list = _parse_float_list(args.eor_var_rel_list)
        noise_list = _parse_float_list(args.noise_lam_list)
        for lfg in l_fg_list:
            for le in l_eor_list:
                for vr in vr_list:
                    for nl in noise_list:
                        candidates.append(
                            Candidate(
                                mode="fg_eor",
                                kernel_fg=kernel,
                                l_fg_mhz=float(lfg),
                                kernel_eor=kernel,
                                l_eor_mhz=float(le),
                                eor_var_rel=float(vr),
                                noise_lam=float(nl),
                            )
                        )

    # Record a top-level scan config for reproducibility.
    scan_cfg = {
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "datasets": [d.name for d in datasets],
        "decomp": decomp_mode,
        "kernel": kernel,
        "fg_only": {"l_mhz_list": _parse_float_list(args.l_mhz_list), "lam_list": _parse_float_list(args.lam_list)},
        "fg_eor": {
            "l_fg_mhz_list": _parse_float_list(args.l_fg_mhz_list),
            "l_eor_mhz_list": _parse_float_list(args.l_eor_mhz_list),
            "eor_var_rel_list": _parse_float_list(args.eor_var_rel_list),
            "noise_lam_list": _parse_float_list(args.noise_lam_list),
        },
        "df_mhz": float(args.df_mhz),
        "jitter": float(args.jitter),
        "device": str(device),
        "dtype": str(args.dtype),
        "cut_xy": {
            "enabled": True,
            "unit": "frac",
            "center_x": float(args.cut_center_x),
            "center_y": float(args.cut_center_y),
            "size": float(args.cut_size_frac),
        },
        "write_fits": bool(args.write_fits),
        "chunk_pixels": int(args.chunk_pixels),
    }
    (out_dir / "scan_config.json").write_text(json.dumps(scan_cfg, indent=2), encoding="utf-8")

    results_path = out_dir / "gpr_baseline_results.csv"
    rows: List[Dict[str, object]] = []

    for ds in datasets:
        # Build cut indices using existing helper.
        # We only need the shape: read it from the FITS header without materializing the whole cube.
        try:
            from astropy.io import fits
        except Exception as exc:  # pragma: no cover - dependency check
            raise SystemExit("This script requires astropy for FITS header inspection.") from exc

        with fits.open(ds.input_cube, memmap=True) as hdul:
            data = hdul[0].data
            if data is None or getattr(data, "ndim", 0) != 3:
                raise ValueError(f"Expected a 3D cube in {ds.input_cube}, found {getattr(data, 'shape', None)}")
            shape = tuple(int(x) for x in data.shape)
        cfg = OptimizationConfig()
        cfg.cut_xy_enabled = True
        cfg.cut_xy_unit = "frac"
        cfg.cut_xy_center_x = float(args.cut_center_x)
        cfg.cut_xy_center_y = float(args.cut_center_y)
        cfg.cut_xy_size = float(args.cut_size_frac)
        cfg.freq_axis = int(freq_axis)
        cut = build_cut_xy_indices(shape, freq_axis=freq_axis, config=cfg)
        y = read_fits_cube(ds.input_cube, cut_indices=cut)
        eor_true = read_fits_cube(ds.eor_true_cube, cut_indices=cut)

        y = y.to(device=device, dtype=dtype)
        eor_true = eor_true.to(device=device, dtype=dtype)

        F = int(y.shape[freq_axis])
        if F < 2:
            raise ValueError(f"Dataset {ds.name}: need at least 2 frequency channels, got {F}.")

        for cand in candidates:
            run_dir = out_dir / ds.name / cand.name
            run_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            mode = str(cand.mode).strip().lower()
            if mode == "fg_only":
                M_fg = _gpr_filter_matrix(
                    F,
                    df_mhz=float(args.df_mhz),
                    l_mhz=float(cand.l_fg_mhz),
                    lam=float(cand.noise_lam),
                    kernel=str(cand.kernel_fg),
                    jitter=float(args.jitter),
                    device=device,
                    dtype=dtype,
                )
                M_eor = None
            else:
                M_fg, M_eor = _gpr_two_component_filters(
                    F,
                    df_mhz=float(args.df_mhz),
                    fg_kernel=str(cand.kernel_fg),
                    fg_l_mhz=float(cand.l_fg_mhz),
                    eor_kernel=str(cand.kernel_eor),
                    eor_l_mhz=float(cand.l_eor_mhz),
                    eor_var_rel=float(cand.eor_var_rel),
                    noise_lam=float(cand.noise_lam),
                    jitter=float(args.jitter),
                    device=device,
                    dtype=dtype,
                )

            y_front = y.movedim(freq_axis, 0)  # (F, X, Y)
            flat = y_front.reshape(F, -1).transpose(0, 1)  # (Npix, F)
            n_pix = int(flat.shape[0])
            chunk = int(args.chunk_pixels)
            if chunk <= 0:
                fg_flat = flat @ M_fg.transpose(0, 1)
            else:
                fg_flat = torch.empty_like(flat)
                for start in range(0, n_pix, chunk):
                    end = min(start + chunk, n_pix)
                    fg_flat[start:end] = flat[start:end] @ M_fg.transpose(0, 1)
            fg_front = fg_flat.transpose(0, 1).reshape_as(y_front)
            fg_est = fg_front.movedim(0, freq_axis)
            if M_eor is None:
                eor_est = y - fg_est
            else:
                if chunk <= 0:
                    eor_flat = flat @ M_eor.transpose(0, 1)
                else:
                    eor_flat = torch.empty_like(flat)
                    for start in range(0, n_pix, chunk):
                        end = min(start + chunk, n_pix)
                        eor_flat[start:end] = flat[start:end] @ M_eor.transpose(0, 1)
                eor_front = eor_flat.transpose(0, 1).reshape_as(y_front)
                eor_est = eor_front.movedim(0, freq_axis)

            corr = _per_frequency_corr(eor_est, eor_true, freq_axis=freq_axis)
            corr_np = corr.detach().cpu().numpy().astype(np.float32, copy=False)
            dt = float(time.time() - t0)

            finite = corr_np[np.isfinite(corr_np)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            median = float(np.median(finite)) if finite.size else float("nan")
            p10 = float(np.percentile(finite, 10)) if finite.size else float("nan")
            cmin = float(np.min(finite)) if finite.size else float("nan")
            cmax = float(np.max(finite)) if finite.size else float("nan")
            score = _score_from_corr(corr_np)

            # Persist small artifacts (corr profile + minimal run config).
            prof_path = run_dir / "eor_corr_profile.csv"
            with prof_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["freq_idx", "corr"])
                for idx, val in enumerate(corr_np.tolist()):
                    writer.writerow([int(idx), f"{float(val):.10g}" if np.isfinite(val) else "nan"])

            run_cfg = {
                "input_cube": str(ds.input_cube),
                "fg_output": str(run_dir / "fg_est.fits"),
                "eor_output": str(run_dir / "eor_est.fits"),
                "cut_xy": scan_cfg["cut_xy"],
                "evaluation": {
                    "true_eor_cube": str(ds.eor_true_cube),
                },
                "optim": {
                    "num_iters": 0,
                    "device": str(device),
                    "dtype": str(args.dtype),
                    "loss_mode": "gpr_baseline",
                    "extra_loss_terms": [],
                    "freq_axis": int(freq_axis),
                    "freq_start_mhz": float(getattr(ds, "freq_start_mhz", float("nan"))),
                    "freq_delta_mhz": float(args.df_mhz),
                },
                "gpr": asdict(cand),
            }
            (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

            if bool(args.write_fits):
                # Avoid importing FITS I/O unless requested; keep outputs aligned with other runs.
                from separation_optim import write_fits_cube, cut_xy_fits_header

                extras = cut_xy_fits_header(cut) if cut is not None else None
                write_fits_cube(fg_est, run_dir / "fg_est.fits", header_extras=extras)
                write_fits_cube(eor_est, run_dir / "eor_est.fits", header_extras=extras)

            rows.append(
                {
                    "dataset": ds.name,
                    "candidate": cand.name,
                    "decomp": str(cand.mode),
                    "kernel_fg": cand.kernel_fg,
                    "l_fg_mhz": float(cand.l_fg_mhz),
                    "kernel_eor": cand.kernel_eor,
                    "l_eor_mhz": float(cand.l_eor_mhz),
                    "eor_var_rel": float(cand.eor_var_rel),
                    "noise_lam": float(cand.noise_lam),
                    "cut_size_frac": float(args.cut_size_frac),
                    "runtime_sec": dt,
                    "eor_corr_score": float(score),
                    "eor_corr_mean": mean,
                    "eor_corr_median": median,
                    "eor_corr_p10": p10,
                    "eor_corr_min": cmin,
                    "eor_corr_max": cmax,
                    "eor_corr_profile_path": str(prof_path),
                    "run_dir": str(run_dir),
                }
            )

            # Free large intermediates early.
            del fg_est, eor_est, fg_front, fg_flat, M_fg
            if mode != "fg_only":
                del eor_front, eor_flat, M_eor
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # Write combined CSV.
    fieldnames = list(rows[0].keys()) if rows else []
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write a short markdown ranking per dataset.
    md_path = out_dir / "gpr_baseline_summary.md"
    lines: List[str] = []
    lines.append("# GPR Baseline Scan Summary\n\n")
    lines.append(f"- output_dir: `{out_dir}`\n")
    lines.append(f"- decomp: `{decomp_mode}`\n")
    lines.append(f"- kernel: `{kernel}`\n")
    lines.append(f"- df_mhz: `{float(args.df_mhz):g}`\n")
    lines.append(f"- cut_size_frac: `{float(args.cut_size_frac):g}`\n\n")

    for ds_name in sorted(set(r["dataset"] for r in rows)):
        subset = [r for r in rows if r["dataset"] == ds_name]
        subset.sort(key=lambda x: float(x.get("eor_corr_score", float("-inf"))), reverse=True)
        lines.append(f"## {ds_name}\n\n")
        lines.append("| rank | candidate | score | mean | median | p10 | runtime_sec |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
        for rank, r in enumerate(subset[:10], start=1):
            lines.append(
                f"| {rank} | {r['candidate']} | {float(r['eor_corr_score']):.6f} | "
                f"{float(r['eor_corr_mean']):.6f} | {float(r['eor_corr_median']):.6f} | "
                f"{float(r['eor_corr_p10']):.6f} | {float(r['runtime_sec']):.2f} |\n"
            )
        lines.append("\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"[done] wrote: {results_path}")
    print(f"[done] wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
