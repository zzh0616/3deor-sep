#!/usr/bin/env python3
"""Screen a CHIPS-like cross-fitted gridded-visibility covariance estimator.

For each kperp bin and held-out uv-angle fold, a complete complex frequency
covariance is estimated from the other folds.  The deployable screen uses only
the summed observation.  A foreground-label covariance is evaluated separately
as an oracle control and never participates in method selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chips_visibility import (  # noqa: E402
    QuadraticResponse,
    build_quadratic_response,
    fold_absolute_delay,
    fold_window_absolute_delay,
    frequency_fourier_basis,
)
from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from ops_scripts.evaluate_chips_dpss_visibility_noiseless import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _load_bank,
    _metrics,
)


def _parse_csv_floats(spec: str) -> list[float]:
    values = [float(piece.strip()) for piece in str(spec).split(",") if piece.strip()]
    if not values or np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError("Covariance floors must be finite and positive")
    return values


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--eigen-floor-fractions",
        default="1e-2,1e-4,1e-6,1e-8,1e-10,1e-12",
    )
    parser.add_argument("--diagonal-shrinkage", type=float, default=0.05)
    parser.add_argument("--angular-folds", type=int, default=4)
    parser.add_argument("--minimum-grid-weight", type=float, default=1.0)
    parser.add_argument("--minimum-training-samples", type=int, default=64)
    parser.add_argument("--minimum-test-samples", type=int, default=8)
    parser.add_argument("--minimum-relative-sensitivity", type=float, default=1e-4)
    parser.add_argument("--minimum-window-self-fraction", type=float, default=0.1)
    parser.add_argument("--foreground-leakage-tolerance", type=float, default=0.1)
    parser.add_argument("--total-relative-error-tolerance", type=float, default=0.2)
    return parser.parse_args(argv)


def _taper_values(size: int) -> np.ndarray:
    return np.hanning(int(size)).astype(np.float64)


def _response_from_inverse(
    frequencies_hz: np.ndarray,
    inverse_covariance: np.ndarray,
) -> QuadraticResponse:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    inverse = np.asarray(inverse_covariance, dtype=np.complex128)
    if inverse.shape != (frequencies.size, frequencies.size):
        raise ValueError("Inverse covariance has the wrong shape")
    fourier, delays = frequency_fourier_basis(frequencies)
    taper = _taper_values(frequencies.size)
    analysis = fourier.conj().T @ np.diag(taper) @ inverse
    fisher = np.square(np.abs(analysis @ fourier))
    row_normalization = np.sum(fisher, axis=1)
    window = np.zeros_like(fisher)
    supported = row_normalization > np.finfo(np.float64).eps
    window[supported] = fisher[supported] / row_normalization[supported, None]
    return QuadraticResponse(
        frequencies_hz=frequencies,
        delays_s=delays,
        analysis_matrix=analysis,
        fisher=fisher,
        window=window,
        row_normalization=row_normalization,
        foreground_rank=int(frequencies.size),
        dpss_eigenvalues=np.empty(0, dtype=np.float64),
        max_delay_s=math.nan,
        suppression_strength=math.nan,
        taper=taper,
    )


def _inverse_empirical_covariance(
    samples: np.ndarray,
    *,
    diagonal_shrinkage: float,
    eigen_floor_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(samples, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Covariance samples must have shape [sample,frequency]")
    # Rows store d^T.  The column-vector covariance is E[d d^H], not
    # E[d^* d^T]; the distinction is material for chromatic visibility phases.
    covariance = values.T @ values.conj() / float(values.shape[0])
    covariance = 0.5 * (covariance + covariance.conj().T)
    shrinkage = float(diagonal_shrinkage)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("Diagonal shrinkage must lie in [0,1]")
    diagonal = np.diag(np.real(np.diag(covariance)))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.real(np.trace(covariance))) / values.shape[1], 1e-300)
    floor = float(eigen_floor_fraction) * scale
    regularized = np.maximum(np.real(eigenvalues), floor)
    inverse_eigenvalues = 1.0 / regularized
    inverse = (eigenvectors * inverse_eigenvalues[None, :]) @ eigenvectors.conj().T
    inverse_rms = math.sqrt(
        float(np.real(np.trace(inverse.conj().T @ inverse))) / values.shape[1]
    )
    inverse /= max(inverse_rms, 1e-300)
    return 0.5 * (inverse + inverse.conj().T), {
        "covariance_min_eigenvalue_over_scale": float(np.min(eigenvalues) / scale),
        "covariance_max_eigenvalue_over_scale": float(np.max(eigenvalues) / scale),
        "regularized_condition_number": float(
            np.max(regularized) / np.min(regularized)
        ),
    }


def _cross_score(
    first: np.ndarray,
    second: np.ndarray,
    response: QuadraticResponse,
) -> np.ndarray:
    transformed_first = np.asarray(first) @ response.analysis_matrix.T
    transformed_second = np.asarray(second) @ response.analysis_matrix.T
    return np.mean(
        np.real(np.conjugate(transformed_first) * transformed_second), axis=0
    )


def _fold_vector(values: np.ndarray, response: QuadraticResponse) -> np.ndarray:
    folded, _, _ = fold_absolute_delay(values, response.delays_s)
    return np.asarray(folded, dtype=np.float64)


def _evaluate_candidate(
    *,
    frequencies_hz: np.ndarray,
    covariance_samples: np.ndarray,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    eor_a: np.ndarray,
    eor_b: np.ndarray,
    fold_ids: np.ndarray,
    members: np.ndarray,
    angular_folds: int,
    minimum_training_samples: int,
    minimum_test_samples: int,
    diagonal_shrinkage: float,
    eigen_floor_fraction: float,
) -> tuple[dict[str, np.ndarray], QuadraticResponse | None, dict[str, Any]]:
    score_sums = {
        component: np.zeros(frequencies_hz.size, dtype=np.float64)
        for component in ("foreground", "eor", "total")
    }
    fisher_sum = np.zeros(
        (frequencies_hz.size, frequencies_hz.size), dtype=np.float64
    )
    test_count = 0
    fold_diagnostics: list[dict[str, Any]] = []
    for fold in range(int(angular_folds)):
        test = members[fold_ids[members] == fold]
        train = members[fold_ids[members] != fold]
        if (
            train.size < int(minimum_training_samples)
            or test.size < int(minimum_test_samples)
        ):
            continue
        inverse, diagnostics = _inverse_empirical_covariance(
            covariance_samples[train],
            diagonal_shrinkage=diagonal_shrinkage,
            eigen_floor_fraction=eigen_floor_fraction,
        )
        response = _response_from_inverse(frequencies_hz, inverse)
        fg_score = _cross_score(fg_a[test], fg_b[test], response)
        eor_score = _cross_score(eor_a[test], eor_b[test], response)
        total_score = _cross_score(
            fg_a[test] + eor_a[test],
            fg_b[test] + eor_b[test],
            response,
        )
        for name, score in (
            ("foreground", fg_score),
            ("eor", eor_score),
            ("total", total_score),
        ):
            score_sums[name] += float(test.size) * score
        fisher_sum += float(test.size) * response.fisher
        test_count += int(test.size)
        fold_diagnostics.append(
            {
                "fold": int(fold),
                "training_samples": int(train.size),
                "test_samples": int(test.size),
                **diagnostics,
            }
        )
    if test_count == 0:
        return {}, None, {"test_samples": 0, "folds": fold_diagnostics}
    fisher = fisher_sum / float(test_count)
    row_normalization = np.sum(fisher, axis=1)
    window = np.zeros_like(fisher)
    supported = row_normalization > np.finfo(np.float64).eps
    window[supported] = fisher[supported] / row_normalization[supported, None]
    reference = _response_from_inverse(
        frequencies_hz, np.eye(frequencies_hz.size, dtype=np.complex128)
    )
    response = QuadraticResponse(
        frequencies_hz=frequencies_hz,
        delays_s=reference.delays_s,
        analysis_matrix=reference.analysis_matrix,
        fisher=fisher,
        window=window,
        row_normalization=row_normalization,
        foreground_rank=int(frequencies_hz.size),
        dpss_eigenvalues=np.empty(0, dtype=np.float64),
        max_delay_s=math.nan,
        suppression_strength=math.nan,
        taper=reference.taper,
    )
    estimates: dict[str, np.ndarray] = {}
    for name, score in score_sums.items():
        mean_score = score / float(test_count)
        estimate = np.full(mean_score.shape, np.nan, dtype=np.float64)
        estimate[supported] = mean_score[supported] / row_normalization[supported]
        estimates[name] = _fold_vector(estimate, response)
    estimates["cross"] = (
        estimates["total"] - estimates["foreground"] - estimates["eor"]
    )
    return estimates, response, {
        "test_samples": int(test_count),
        "folds": fold_diagnostics,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    floors = _parse_csv_floats(args.eigen_floor_fractions)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    manifest, bank = _load_bank(args.bank_dir)
    frequencies_hz = np.asarray(bank["frequencies_hz"], dtype=np.float64)
    if not np.allclose(
        frequencies_hz,
        np.asarray(resolved.geometry["frequencies_mhz"]) * 1e6,
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError("Visibility bank and config frequencies differ")

    grid_weight = np.asarray(bank["grid_weight"], dtype=np.float64)
    fg_grid = np.asarray(bank["fg_grid"], dtype=np.complex128)
    eor_grid = np.asarray(bank["eor_grid"], dtype=np.complex128)
    complete = np.all(
        grid_weight >= float(args.minimum_grid_weight), axis=(0, 1)
    )
    u, v = np.meshgrid(
        np.asarray(bank["u_centers_lambda"], dtype=np.float64),
        np.asarray(bank["v_centers_lambda"], dtype=np.float64),
        indexing="xy",
    )
    tolerance = max(float(np.max(np.abs(u))), 1.0) * 1e-12
    canonical = (v > tolerance) | ((np.abs(v) <= tolerance) & (u >= 0.0))
    selected = complete & canonical
    if not np.any(selected):
        raise ValueError("No complete canonical uv cells")
    u_selected = u[selected]
    v_selected = v[selected]
    kperp = (
        2.0
        * math.pi
        * np.hypot(u_selected, v_selected)
        / float(resolved.geometry["transverse_distance_mpc"])
    )
    angle = np.mod(np.arctan2(v_selected, u_selected), math.pi)
    fold_ids = np.minimum(
        (angle / math.pi * int(args.angular_folds)).astype(np.int64),
        int(args.angular_folds) - 1,
    )
    fg_a = fg_grid[0][:, selected].T
    fg_b = fg_grid[1][:, selected].T
    eor_a = eor_grid[0][:, selected].T
    eor_b = eor_grid[1][:, selected].T
    total_weight = grid_weight[0][:, selected] + grid_weight[1][:, selected]
    fg_coadd = (
        grid_weight[0][:, selected] * fg_grid[0][:, selected]
        + grid_weight[1][:, selected] * fg_grid[1][:, selected]
    ) / total_weight
    eor_coadd = (
        grid_weight[0][:, selected] * eor_grid[0][:, selected]
        + grid_weight[1][:, selected] * eor_grid[1][:, selected]
    ) / total_weight
    covariance_sources = {
        "observed_total": (fg_coadd + eor_coadd).T,
        "oracle_foreground": fg_coadd.T,
    }

    edges = np.asarray(resolved.contract.window_layout.kperp_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nkperp = centers.size
    signed_delays = np.fft.fftfreq(
        frequencies_hz.size, d=float(np.median(np.diff(frequencies_hz)))
    )
    delays, _, degeneracy = fold_absolute_delay(
        np.abs(signed_delays), signed_delays
    )
    radial_mpc_per_hz = float(resolved.geometry["radial_spacing_mpc"]) / float(
        np.mean(np.diff(frequencies_hz))
    )
    kpar = 2.0 * math.pi * delays / radial_mpc_per_hz
    geometric_window = resolved.window_spec.mask(edges[1:, None], kpar[None, :])
    raw_response = build_quadratic_response(
        frequencies_hz,
        max_delay_s=0.0,
        suppression_strength=0.0,
        taper="hann",
    )
    raw_norm = _fold_vector(raw_response.row_normalization, raw_response)
    products: dict[str, np.ndarray] = {
        "kperp_edges_mpc_inv": edges,
        "kperp_centers_mpc_inv": centers,
        "kpar_mpc_inv": kpar,
        "geometric_window": geometric_window.astype(np.int8),
        "delay_mode_degeneracy": degeneracy,
    }
    metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}

    for source_name, covariance_samples in covariance_sources.items():
        source_metrics: dict[str, Any] = {}
        for floor in floors:
            name = f"floor_{floor:.0e}"
            arrays = {
                component: np.full((nkperp, kpar.size), np.nan, dtype=np.float64)
                for component in ("foreground", "eor", "total", "cross")
            }
            sensitivity = np.zeros((nkperp, kpar.size), dtype=np.float64)
            window_self = np.zeros_like(sensitivity)
            windows = np.zeros(
                (nkperp, kpar.size, kpar.size), dtype=np.float64
            )
            test_counts = np.zeros(nkperp, dtype=np.int64)
            candidate_diagnostics: list[dict[str, Any]] = []
            for index in range(nkperp):
                members = np.flatnonzero(
                    (kperp >= edges[index])
                    & (
                        (kperp < edges[index + 1])
                        | (
                            index == nkperp - 1
                            and np.isclose(
                                kperp,
                                edges[index + 1],
                                rtol=1e-12,
                                atol=1e-14,
                            )
                        )
                    )
                )
                estimated, response, current_diagnostics = _evaluate_candidate(
                    frequencies_hz=frequencies_hz,
                    covariance_samples=covariance_samples,
                    fg_a=fg_a,
                    fg_b=fg_b,
                    eor_a=eor_a,
                    eor_b=eor_b,
                    fold_ids=fold_ids,
                    members=members,
                    angular_folds=int(args.angular_folds),
                    minimum_training_samples=int(args.minimum_training_samples),
                    minimum_test_samples=int(args.minimum_test_samples),
                    diagonal_shrinkage=float(args.diagonal_shrinkage),
                    eigen_floor_fraction=float(floor),
                )
                current_diagnostics["kperp_bin"] = int(index)
                candidate_diagnostics.append(current_diagnostics)
                if response is None:
                    continue
                test_counts[index] = int(current_diagnostics["test_samples"])
                for component, values in estimated.items():
                    arrays[component][index] = values
                folded_norm = _fold_vector(
                    response.row_normalization, response
                )
                sensitivity[index] = np.divide(
                    folded_norm,
                    raw_norm,
                    out=np.zeros_like(folded_norm),
                    where=raw_norm > 0.0,
                )
                folded_window, _ = fold_window_absolute_delay(
                    response.window, response.delays_s
                )
                windows[index] = folded_window
                window_self[index] = np.diag(folded_window)
            support = (
                geometric_window
                & (test_counts[:, None] >= int(args.minimum_test_samples))
                & (
                    sensitivity
                    >= float(args.minimum_relative_sensitivity)
                )
                & (
                    window_self
                    >= float(args.minimum_window_self_fraction)
                )
            )
            aggregate_weights = (
                test_counts[:, None]
                * np.asarray(degeneracy, dtype=np.float64)[None, :]
            )
            current_metrics = _metrics(
                foreground=arrays["foreground"],
                eor=arrays["eor"],
                total=arrays["total"],
                mask=support,
                weights=aggregate_weights,
                foreground_tolerance=float(args.foreground_leakage_tolerance),
                total_tolerance=float(args.total_relative_error_tolerance),
            )
            current_metrics["support_fraction_of_geometric_window"] = float(
                np.count_nonzero(support)
                / max(1, np.count_nonzero(geometric_window))
            )
            source_metrics[name] = current_metrics
            prefix = f"{source_name}__{name}"
            for component, values in arrays.items():
                products[f"{prefix}__{component}"] = values
            products[f"{prefix}__support"] = support.astype(np.int8)
            products[f"{prefix}__relative_sensitivity"] = sensitivity
            products[f"{prefix}__window_self"] = window_self
            products[f"{prefix}__window"] = windows
            products[f"{prefix}__test_counts"] = test_counts
            diagnostics[prefix] = candidate_diagnostics
        metrics[source_name] = source_metrics

    result = {
        "schema": "chips_crossfit_covariance_noiseless_result",
        "schema_version": 1,
        "analysis_contract_sha256": resolved.contract.analysis_contract_sha256,
        "visibility_bank_sha256": manifest["bank_sha256"],
        "covariance_sources": {
            "observed_total": "deployable noiseless screen; component labels hidden",
            "oracle_foreground": "diagnostic upper bound; uses true foreground labels",
        },
        "settings": {
            "eigen_floor_fractions": floors,
            "diagonal_shrinkage": float(args.diagonal_shrinkage),
            "angular_folds": int(args.angular_folds),
            "minimum_training_samples": int(args.minimum_training_samples),
            "minimum_test_samples": int(args.minimum_test_samples),
            "minimum_relative_sensitivity": float(
                args.minimum_relative_sensitivity
            ),
            "minimum_window_self_fraction": float(
                args.minimum_window_self_fraction
            ),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "limitations": [
            "no thermal noise",
            "single sky realization",
            "same data choose the covariance floor only as a diagnostic scan",
            "delay-diagonal signal response remains an incomplete sky-PS2D model",
        ],
    }
    _atomic_npz(args.out_dir / "result.npz", products)
    _atomic_json(args.out_dir / "result.json", result)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
