#!/usr/bin/env python3
"""Validate the PS2D v2 estimator contract on a real signal-only cube."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ops_scripts.evaluate_ps2d_v2_mode_first import _load_pattern_cube  # noqa: E402
from ps2d_v2 import compute_ps2d_products  # noqa: E402
from ps2d_v2_estimator import (  # noqa: E402
    IdentityCubeProjector,
    TorchBandpowerTransform,
    analytic_identity_source_response,
    build_mode_first_estimator_contract,
    calibrate_mode_first_transfer,
    estimate_projected_bandpower,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _weighted_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    denominator = float(np.sum(weights * np.square(truth), dtype=np.float64))
    l2 = math.sqrt(
        float(np.sum(weights * np.square(estimate - truth), dtype=np.float64))
        / max(denominator, 1e-300)
    )
    estimate_sum = float(np.sum(weights * estimate, dtype=np.float64))
    truth_sum = float(np.sum(weights * truth, dtype=np.float64))
    return {
        "count_weighted_relative_l2": l2,
        "exact_mode_power_sum_ratio": estimate_sum / max(truth_sum, 1e-300),
        "max_absolute_error": float(np.max(np.abs(estimate - truth))),
    }


def _source_window_metrics(
    window: np.ndarray,
    science_count: int,
    source_kind: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(window, dtype=np.float64)
    if values.shape[0] != int(science_count) or values.shape[1] < int(science_count):
        raise ValueError("Source-window dimensions do not match science bands")
    self_response = np.diag(values[:, :science_count])
    science_total = np.sum(values[:, :science_count], axis=1)
    complement = np.sum(values[:, science_count:], axis=1)
    kinds = np.asarray(source_kind).astype(str)
    if kinds.shape != (values.shape[1],):
        raise ValueError("Source-window categories have the wrong shape")
    category_fractions = {
        kind: np.sum(values[:, kinds == kind], axis=1)
        for kind in np.unique(kinds)
    }
    return {
        "row_sum_max_abs_error": float(
            np.max(np.abs(np.sum(values, axis=1) - 1.0))
        ),
        "self_response_min": float(np.min(self_response)),
        "self_response_median": float(np.median(self_response)),
        "science_source_fraction_min": float(np.min(science_total)),
        "science_source_fraction_median": float(np.median(science_total)),
        "complement_source_fraction_max": float(np.max(complement)),
        "complement_source_fraction_median": float(np.median(complement)),
        "self_response": self_response,
        "science_source_fraction": science_total,
        "complement_source_fraction": complement,
        "category_fraction_median": {
            kind: float(np.median(fraction))
            for kind, fraction in category_fractions.items()
        },
        "category_fraction_max": {
            kind: float(np.max(fraction))
            for kind, fraction in category_fractions.items()
        },
        "category_fractions": category_fractions,
    }


def _truth_blind_support(
    source_window: np.ndarray,
    science_count: int,
    kperp_indices: np.ndarray,
    kpar_indices: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(source_window, dtype=np.float64)
    kp = np.asarray(kperp_indices, dtype=np.int64)
    kz = np.asarray(kpar_indices, dtype=np.int64)
    science = values[:, : int(science_count)]
    self_fraction = np.diag(science)
    neighbors = (
        np.abs(kp[:, None] - kp[None, :])
        + np.abs(kz[:, None] - kz[None, :])
    ) <= 1
    far_leakage = 1.0 - np.sum(science * neighbors, axis=1)
    supported = (self_fraction >= 0.5) & (far_leakage <= 0.25)
    return {
        "defined_without_eor_truth": True,
        "self_fraction_minimum": 0.5,
        "far_leakage_maximum": 0.25,
        "self_fraction": self_fraction,
        "far_leakage": far_leakage,
        "supported": supported,
        "supported_band_count": int(np.sum(supported)),
        "unsupported_band_count": int(np.sum(~supported)),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    closure = result["pure_signal_closure"]
    calibration = result["identity_calibration"]
    source = result["raw_source_window"]
    support = result["truth_blind_analysis_support"]
    lines = [
        "# PS2D v2 estimator identity / pure-EoR 闭环",
        "",
        f"状态：`{'PASS' if result['all_gates_pass'] else 'FAIL'}`。",
        "",
        f"- analysis contract：`{result['analysis_contract_sha256']}`。",
        f"- estimator contract：`{result['estimator_contract_sha256']}`。",
        f"- science/source bands：`{result['geometry']['science_band_count']}` / `{result['geometry']['calibration_source_band_count']}`。",
        f"- 完整 source basis 覆盖 `{result['geometry']['calibration_source_mode_count']}` 个 FFT modes。",
        f"- identity transfer 最大单位阵误差：`{calibration['transfer_identity_max_abs_error']:.3e}`。",
        f"- Monte Carlo/解析 source window 的 row-max 差异中位数/最大值：`{calibration['monte_carlo_vs_analytic_window_row_max_median']:.3e}` / `{calibration['monte_carlo_vs_analytic_window_row_max_maximum']:.3e}`。",
        f"- pure-EoR row-normalized L2 / power ratio：`{closure['row_normalized']['count_weighted_relative_l2']:.3e}` / `{closure['row_normalized']['exact_mode_power_sum_ratio']:.12g}`。",
        f"- NumPy/Torch v2 bandpower 最大相对误差：`{closure['numpy_torch_max_relative_error']:.3e}`。",
        f"- taper source window 的 self response 中位数：`{source['self_response_median']:.6f}`；窗口外补集贡献中位数：`{source['complement_source_fraction_median']:.6f}`。",
        f"- 其中 control / guard / target-window 外 / radial-Nyquist 的中位贡献分别为：`{source['category_fraction_median'].get('control', 0.0):.6f}` / `{source['category_fraction_median'].get('guard', 0.0):.6f}` / `{source['category_fraction_median'].get('target_outside_window', 0.0):.6f}` / `{source['category_fraction_median'].get('radial_nyquist', 0.0):.6f}`。",
        f"- 按预登记 self >= 0.5、far leakage <= 0.25 的纯分析算子 support 为 `{support['supported_band_count']}/{result['geometry']['science_band_count']}` bands。",
        "",
        "该测试只证明 v2 layout、probe、transfer normalization 和 signal-only identity 链条闭环；不代表 foreground nuisance 已通过。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_DIR / "configs/ps2d_v2_8wide_isobeam_patch.json",
    )
    parser.add_argument("--truth-eor-pattern", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probes-per-source-band", type=int, default=2)
    parser.add_argument("--probe-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--transfer-rcond", type=float, default=1e-10)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = build_mode_first_estimator_contract(config)
    analysis = contract.analysis
    frequencies = np.asarray(
        contract.resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    crop_size = int(config["image_geometry"]["eval_crop_size"])
    cube, input_records = _load_pattern_cube(
        args.truth_eor_pattern, frequencies, crop_size
    )
    device = torch.device(str(args.device))
    transform = TorchBandpowerTransform(
        contract.science_geometry, analysis, device
    )
    identity = IdentityCubeProjector(contract.estimator_contract_sha256)
    calibration = calibrate_mode_first_transfer(
        contract=contract,
        target_transform=transform,
        projector=identity,
        probes_per_source_band=int(args.probes_per_source_band),
        batch_size=int(args.probe_batch_size),
        seed=int(args.seed),
        transfer_rcond=float(args.transfer_rcond),
    )
    analytic = analytic_identity_source_response(
        contract=contract,
        target_transform=transform,
        batch_size=max(int(args.probe_batch_size), 1),
    )

    cube_tensor = torch.as_tensor(cube, dtype=torch.float64, device=device)
    direct = transform(cube_tensor).detach().cpu().numpy()[0]
    projected_cube, _ = identity.project(cube_tensor)
    projected = transform(projected_cube).detach().cpu().numpy()[0]
    row_estimate, deconvolved = estimate_projected_bandpower(
        projected, calibration
    )
    numpy_products = compute_ps2d_products(
        cube,
        dx_mpc=analysis.full_layout.dx_mpc,
        dy_mpc=analysis.full_layout.dy_mpc,
        dpar_mpc=analysis.full_layout.dpar_mpc,
        full_kperp_edges=analysis.full_layout.kperp_edges,
        window_kperp_edges=analysis.window_layout.kperp_edges,
        window_spec=contract.resolved.window_spec,
        radial_nyquist_policy=analysis.full_layout.radial_nyquist_policy,
        demean_mode=analysis.demean_mode,
        radial_taper=analysis.radial_taper,
        spatial_taper=analysis.spatial_taper,
    )
    numpy_mean = numpy_products.window.mean.reshape(-1)[
        contract.science_geometry.active_layout_bands
    ]
    numpy_relative = float(
        np.max(np.abs(direct - numpy_mean) / np.maximum(np.abs(numpy_mean), 1e-300))
    )
    counts = contract.science_geometry.counts
    row_metrics = _weighted_metrics(row_estimate, direct, counts)
    deconvolved_metrics = _weighted_metrics(deconvolved, direct, counts)
    transfer = np.asarray(calibration["transfer_matrix"], dtype=np.float64)
    identity_error = float(
        np.max(np.abs(transfer - np.eye(contract.science_geometry.band_count)))
    )
    analytic_source_window = np.asarray(
        analytic["source_window_matrix"], dtype=np.float64
    )
    monte_carlo_source_window = np.asarray(
        calibration["raw_source_window_matrix"], dtype=np.float64
    )
    raw_source_metrics = _source_window_metrics(
        analytic_source_window,
        contract.calibration_science_band_count,
        contract.calibration_source_kind,
    )
    monte_carlo_source_metrics = _source_window_metrics(
        monte_carlo_source_window,
        contract.calibration_science_band_count,
        contract.calibration_source_kind,
    )
    window_difference = np.abs(
        monte_carlo_source_window - analytic_source_window
    )
    row_max_difference = np.max(window_difference, axis=1)
    support = _truth_blind_support(
        analytic_source_window,
        contract.calibration_science_band_count,
        contract.science_geometry.active_kperp_indices,
        contract.science_geometry.active_kpar_indices,
    )

    gates = {
        "analysis_hash_frozen": analysis.analysis_contract_sha256
        == str(config["frozen_analysis_contract_sha256"]),
        "source_covers_every_fft_mode": int(
            contract.calibration_source_geometry.mode_indices.size
        )
        == int(np.prod(analysis.full_layout.cube_shape)),
        "input_response_full_row_rank": int(
            calibration["input_response_svd"]["rank"]
        )
        == contract.science_geometry.band_count,
        "transfer_full_rank": int(calibration["transfer_svd"]["rank"])
        == contract.science_geometry.band_count,
        "identity_transfer_closes": identity_error <= 1e-8,
        "numpy_torch_bandpower_closes": numpy_relative <= 1e-12,
        "row_normalized_pure_signal_closes": (
            row_metrics["count_weighted_relative_l2"] <= 1e-8
            and abs(row_metrics["exact_mode_power_sum_ratio"] - 1.0) <= 1e-8
        ),
        "deconvolved_pure_signal_closes": (
            deconvolved_metrics["count_weighted_relative_l2"] <= 1e-8
            and abs(deconvolved_metrics["exact_mode_power_sum_ratio"] - 1.0)
            <= 1e-8
        ),
        "source_windows_normalized": max(
            raw_source_metrics["row_sum_max_abs_error"],
            monte_carlo_source_metrics["row_sum_max_abs_error"],
        )
        <= 1e-10,
        "monte_carlo_matches_analytic_window": (
            float(np.median(row_max_difference)) <= 0.02
            and float(np.max(row_max_difference)) <= 0.05
        ),
    }
    result = {
        "time_utc": _now(),
        "method": "ps2d_v2_estimator_identity_pure_signal_validation",
        "all_gates_pass": bool(all(gates.values())),
        "gates": gates,
        "analysis_contract_sha256": analysis.analysis_contract_sha256,
        "estimator_contract_sha256": contract.estimator_contract_sha256,
        "implementation": {
            "core_sha256": _sha256(CODE_DIR / "ps2d_v2.py"),
            "config_resolver_sha256": _sha256(CODE_DIR / "ps2d_v2_config.py"),
            "estimator_adapter_sha256": _sha256(
                CODE_DIR / "ps2d_v2_estimator.py"
            ),
            "validator_sha256": _sha256(Path(__file__).resolve()),
            "config_sha256": _sha256(args.config),
        },
        "input": {
            "truth_eor_pattern": args.truth_eor_pattern,
            "files": input_records,
        },
        "geometry": {
            "cube_shape": list(analysis.full_layout.cube_shape),
            "full_band_count": contract.full_geometry.band_count,
            "science_band_count": contract.science_geometry.band_count,
            "science_fft_mode_count": int(np.sum(counts)),
            "calibration_source_band_count": (
                contract.calibration_source_geometry.band_count
            ),
            "calibration_science_band_count": (
                contract.calibration_science_band_count
            ),
            "calibration_source_mode_count": int(
                contract.calibration_source_geometry.mode_indices.size
            ),
            "control_mode_count": int(contract.control_mode_indices.size),
            "guard_mode_count": int(contract.guard_mode_indices.size),
            "kpar_values_mpc_inv": analysis.full_layout.kpar_values,
        },
        "identity_calibration": {
            "probes_per_source_band": int(args.probes_per_source_band),
            "source_band_count": int(calibration["source_band_count"]),
            "target_band_count": int(calibration["target_band_count"]),
            "input_response_rank": int(calibration["input_response_svd"]["rank"]),
            "transfer_rank": int(calibration["transfer_svd"]["rank"]),
            "transfer_retained_condition_number": calibration["transfer_svd"][
                "retained_condition_number"
            ],
            "transfer_identity_max_abs_error": identity_error,
            "monte_carlo_vs_analytic_window_row_max_median": float(
                np.median(row_max_difference)
            ),
            "monte_carlo_vs_analytic_window_row_max_maximum": float(
                np.max(row_max_difference)
            ),
        },
        "pure_signal_closure": {
            "numpy_torch_max_relative_error": numpy_relative,
            "row_normalized": row_metrics,
            "deconvolved": deconvolved_metrics,
        },
        "raw_source_window": raw_source_metrics,
        "monte_carlo_source_window": monte_carlo_source_metrics,
        "truth_blind_analysis_support": support,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "result.json", result)
    _atomic_npz(
        args.out_dir / "result.npz",
        {
            "input_response": calibration["input_response"],
            "projected_response": calibration["projected_response"],
            "transfer_matrix": calibration["transfer_matrix"],
            "transfer_window_matrix": calibration["transfer_window_matrix"],
            "raw_source_window_matrix": calibration["raw_source_window_matrix"],
            "analytic_source_window_matrix": analytic_source_window,
            "projected_source_window_matrix": calibration[
                "projected_source_window_matrix"
            ],
            "source_band_kind": contract.calibration_source_kind,
            "source_parent_bands": contract.calibration_source_parent_bands,
            "science_bandpower": direct,
            "row_normalized_bandpower": row_estimate,
            "deconvolved_bandpower": deconvolved,
            "science_mode_counts": counts,
            "analysis_supported": support["supported"],
        },
    )
    _write_report(args.out_dir / "report.md", _json_safe(result))
    print(
        json.dumps(
            {
                "event": "ps2d_v2_estimator_identity_done",
                "all_gates_pass": result["all_gates_pass"],
                "analysis_contract_sha256": analysis.analysis_contract_sha256,
                "estimator_contract_sha256": contract.estimator_contract_sha256,
                "out_dir": str(args.out_dir),
                "time_utc": _now(),
            },
            sort_keys=True,
        )
    )
    if not result["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
