#!/usr/bin/env python3
"""Assemble cross-window covariance and decorrelated local Q_beta modes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        help="LABEL=coarse/products.npz; repeat in redshift order.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", default="quad_kperp_response")
    parser.add_argument("--eigen-rcond", type=float, default=1e-6)
    parser.add_argument(
        "--allow-legacy-unaligned-realizations",
        action="store_true",
        help=(
            "Allow old per-window same-seed products that do not certify a "
            "shared full-band sky. Never use for formal covariance."
        ),
    )
    return parser.parse_args(argv)


def _parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    used: set[str] = set()
    for value in values:
        label, separator, path = value.partition("=")
        label = label.strip()
        if not separator or not label or not path:
            raise ValueError("--window must use LABEL=PATH")
        if label in used:
            raise ValueError(f"Duplicate window label: {label}")
        used.add(label)
        output.append((label, Path(path)))
    return output


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
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _participation_rank_from_eigenvalues(values: np.ndarray) -> float:
    eigenvalues = np.maximum(
        np.asarray(values, dtype=np.float64).reshape(-1), 0.0
    )
    return float(
        np.square(np.sum(eigenvalues))
        / max(float(np.sum(np.square(eigenvalues))), 1e-300)
    )


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = diagonal[:, None] * diagonal[None, :]
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )


def _orient_transform_rows(transform: np.ndarray) -> np.ndarray:
    """Choose deterministic signs without consulting simulated bandpowers."""
    oriented = np.asarray(transform, dtype=np.float64).copy()
    for row in oriented:
        reference = float(np.sum(row))
        if abs(reference) <= 10.0 * np.finfo(np.float64).eps:
            reference = float(row[np.argmax(np.abs(row))])
        if reference < 0.0:
            row *= -1.0
    return oriented


def _maximum_off_diagonal(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape[0] < 2:
        return 0.0
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.max(np.abs(matrix[off_diagonal])))


def _crossfit_covariance_diagnostics(
    errors: np.ndarray,
    *,
    eigen_rcond: float,
) -> list[dict[str, Any]]:
    """Fit KL modes on one realization fold and evaluate the other."""
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 6:
        raise ValueError("Cross-fit covariance requires at least six rows")
    diagnostics = []
    indices = np.arange(values.shape[0], dtype=np.int64)
    for fold in range(2):
        train = values[indices % 2 == fold]
        test = values[indices % 2 != fold]
        train_covariance = np.atleast_2d(
            np.cov(train, rowvar=False, ddof=1)
        )
        train_covariance = 0.5 * (
            train_covariance + train_covariance.T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(train_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        cutoff = float(eigen_rcond) * max(float(eigenvalues[0]), 0.0)
        retained = eigenvalues > cutoff
        retained_values = eigenvalues[retained]
        if retained_values.size < 1:
            raise ValueError("Cross-fit covariance retains no eigenmodes")
        transform = _orient_transform_rows(
            eigenvectors[:, retained].T
        )
        test_modes = test @ transform.T
        test_covariance = np.atleast_2d(
            np.cov(test_modes, rowvar=False, ddof=1)
        )
        test_correlation = _covariance_to_correlation(test_covariance)
        variance_ratio = np.diag(test_covariance) / retained_values
        standardized_mean = (
            np.mean(test_modes, axis=0) / np.sqrt(retained_values)
        )
        diagnostics.append(
            {
                "train_fold": int(fold),
                "train_realization_count": int(train.shape[0]),
                "test_realization_count": int(test.shape[0]),
                "retained_eigenmode_count": int(retained_values.size),
                "train_covariance_participation_rank": (
                    _participation_rank_from_eigenvalues(retained_values)
                ),
                "test_to_train_variance_ratio_median": float(
                    np.median(variance_ratio)
                ),
                "test_to_train_variance_ratio_p10": float(
                    np.percentile(variance_ratio, 10.0)
                ),
                "test_to_train_variance_ratio_p90": float(
                    np.percentile(variance_ratio, 90.0)
                ),
                "test_kl_correlation_maximum_off_diagonal": (
                    _maximum_off_diagonal(test_correlation)
                ),
                "test_standardized_mean_rms": float(
                    np.sqrt(np.mean(np.square(standardized_mean)))
                ),
            }
        )
    return diagnostics


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    windows = _parse_labeled_paths(args.window)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile = str(args.profile)
    labels: list[str] = []
    offsets = [0]
    targets: list[np.ndarray] = []
    bank_estimates: list[np.ndarray] = []
    realization_estimates: list[np.ndarray] = []
    window_ranks: list[dict[str, Any]] = []
    realization_count: int | None = None
    shared_contract: str | None = None
    realization_source: str | None = None
    for label, path in windows:
        with np.load(path, allow_pickle=False) as archive:
            data = {name: np.asarray(archive[name]) for name in archive.files}
        prefix = f"{profile}_"
        required = {
            f"{prefix}selected",
            f"{prefix}target",
            f"{prefix}bank_total_estimate",
            f"{prefix}heldout_total_estimate",
            f"{prefix}window",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(
                f"{label} lacks covariance products: {', '.join(missing)}"
            )
        if "shared_realization_contract_sha256" in data:
            current_contract = str(
                np.asarray(
                    data["shared_realization_contract_sha256"]
                ).item()
            )
            current_source = str(
                np.asarray(
                    data["shared_realization_source"]
                ).item()
            )
            if shared_contract is None:
                shared_contract = current_contract
                realization_source = current_source
            elif current_contract != shared_contract:
                raise ValueError(
                    "Local windows use different shared-realization contracts"
                )
            elif current_source != realization_source:
                raise ValueError(
                    "Local windows describe shared realizations differently"
                )
        elif not args.allow_legacy_unaligned_realizations:
            raise ValueError(
                f"{label} has no certified shared full-band realizations"
            )
        selected = np.asarray(data[f"{prefix}selected"], dtype=bool)
        target = np.asarray(data[f"{prefix}target"], dtype=np.float64)[
            selected
        ]
        bank = np.asarray(
            data[f"{prefix}bank_total_estimate"], dtype=np.float64
        )[selected]
        realizations = np.asarray(
            data[f"{prefix}heldout_total_estimate"], dtype=np.float64
        )[:, selected]
        if realization_count is None:
            realization_count = int(realizations.shape[0])
        elif realizations.shape[0] != realization_count:
            raise ValueError("Local windows use different realization counts")
        response_window = np.asarray(
            data[f"{prefix}window"], dtype=np.float64
        )[selected]
        singular = np.linalg.svd(response_window, compute_uv=False)
        squared = np.square(singular)
        window_ranks.append(
            {
                "label": label,
                "selected_bandpower_count": int(target.size),
                "numerical_rank_rcond_1e3": int(
                    np.count_nonzero(singular >= 1e-3 * singular[0])
                ),
                "participation_rank": _participation_rank_from_eigenvalues(
                    squared
                ),
            }
        )
        labels.append(label)
        targets.append(target)
        bank_estimates.append(bank)
        realization_estimates.append(realizations)
        offsets.append(offsets[-1] + target.size)
    assert realization_count is not None
    if realization_count < 3:
        raise ValueError("At least three aligned realizations are required")

    target_all = np.concatenate(targets)
    bank_all = np.concatenate(bank_estimates)
    realization_all = np.concatenate(realization_estimates, axis=1)
    errors = realization_all - target_all[None, :]
    covariance = np.atleast_2d(np.cov(errors, rowvar=False, ddof=1))
    covariance = 0.5 * (covariance + covariance.T)
    correlation = _covariance_to_correlation(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    cutoff = float(args.eigen_rcond) * max(float(eigenvalues[0]), 0.0)
    retained = eigenvalues > cutoff
    retained_eigenvalues = eigenvalues[retained]
    kl_transform = _orient_transform_rows(
        eigenvectors[:, retained].T
    )
    kl_target = kl_transform @ target_all
    kl_bank = kl_transform @ bank_all
    kl_errors = errors @ kl_transform.T
    kl_covariance = np.atleast_2d(
        np.cov(kl_errors, rowvar=False, ddof=1)
    )
    kl_correlation = _covariance_to_correlation(kl_covariance)
    whitening = kl_transform / np.sqrt(retained_eigenvalues)[:, None]
    whitened_target = whitening @ target_all
    whitened_bank = whitening @ bank_all
    whitened_errors = errors @ whitening.T
    decorrelated_covariance = np.atleast_2d(
        np.cov(whitened_errors, rowvar=False, ddof=1)
    )
    decorrelated_correlation = _covariance_to_correlation(
        decorrelated_covariance
    )
    off_diagonal = ~np.eye(
        int(np.count_nonzero(retained)), dtype=bool
    )
    block_correlations: list[dict[str, Any]] = []
    for first_index in range(len(labels)):
        for second_index in range(first_index + 1, len(labels)):
            first = slice(offsets[first_index], offsets[first_index + 1])
            second = slice(offsets[second_index], offsets[second_index + 1])
            block = np.abs(correlation[first, second])
            block_correlations.append(
                {
                    "first": labels[first_index],
                    "second": labels[second_index],
                    "median_absolute_correlation": float(np.median(block)),
                    "maximum_absolute_correlation": float(np.max(block)),
                }
            )
    result = {
        "schema": "visibility_qbeta_local_redshift_covariance",
        "schema_version": 1,
        "profile": profile,
        "window_labels": labels,
        "window_offsets": offsets,
        "realization_source": (
            realization_source
            if realization_source is not None
            else "legacy unaligned per-window probes"
        ),
        "shared_realization_contract_sha256": shared_contract,
        "realization_count": int(realization_count),
        "bandpower_count": int(target_all.size),
        "covariance_rank_limit": int(realization_count - 1),
        "retained_eigenmode_count": int(np.count_nonzero(retained)),
        "eigen_rcond": float(args.eigen_rcond),
        "covariance_participation_rank": (
            _participation_rank_from_eigenvalues(eigenvalues)
        ),
        "response_window_ranks": window_ranks,
        "response_window_participation_rank_sum": float(
            sum(item["participation_rank"] for item in window_ranks)
        ),
        "cross_window_correlations": block_correlations,
        "decorrelated_covariance_maximum_off_diagonal": float(
            np.max(np.abs(decorrelated_covariance[off_diagonal]))
            if np.any(off_diagonal)
            else 0.0
        ),
        "decorrelated_correlation_maximum_off_diagonal": float(
            np.max(np.abs(decorrelated_correlation[off_diagonal]))
            if np.any(off_diagonal)
            else 0.0
        ),
        "kl_correlation_maximum_off_diagonal": float(
            np.max(np.abs(kl_correlation[off_diagonal]))
            if np.any(off_diagonal)
            else 0.0
        ),
        "bank_kl_relative_l2": float(
            np.linalg.norm(kl_bank - kl_target)
            / max(np.linalg.norm(kl_target), 1e-300)
        ),
        "bank_whitened_relative_l2": float(
            np.linalg.norm(whitened_bank - whitened_target)
            / max(np.linalg.norm(whitened_target), 1e-300)
        ),
        "crossfit_covariance_diagnostics": (
            _crossfit_covariance_diagnostics(
                errors,
                eigen_rcond=float(args.eigen_rcond),
            )
        ),
        "selection_uses_covariance": False,
    }
    products = {
        "window_labels": np.asarray(labels),
        "window_offsets": np.asarray(offsets, dtype=np.int64),
        "target": target_all,
        "bank_estimate": bank_all,
        "heldout_total_estimate": realization_all,
        "heldout_error_covariance": covariance,
        "heldout_error_correlation": correlation,
        "covariance_eigenvalues": eigenvalues,
        "covariance_eigenvectors": eigenvectors,
        "retained_eigenmodes": retained.astype(np.int8),
        "kl_transform": kl_transform,
        "kl_target": kl_target,
        "kl_bank_estimate": kl_bank,
        "kl_error_covariance": kl_covariance,
        "kl_error_correlation": kl_correlation,
        "whitening_matrix": whitening,
        "whitened_target": whitened_target,
        "whitened_bank_estimate": whitened_bank,
        "decorrelated_target": kl_target,
        "decorrelated_bank_estimate": kl_bank,
        "decorrelated_error_covariance": decorrelated_covariance,
        "decorrelated_error_correlation": decorrelated_correlation,
    }
    _atomic_json(args.out_dir / "summary.json", result)
    _atomic_npz(args.out_dir / "products.npz", products)
    print(json.dumps(_json_safe(result), sort_keys=True))


if __name__ == "__main__":
    main()
