import json
from pathlib import Path

import numpy as np

from ops_scripts.evaluate_visibility_qbeta_local_redshift_covariance import (
    _covariance_to_correlation,
    _crossfit_covariance_diagnostics,
    _orient_transform_rows,
    _parse_labeled_paths,
    _participation_rank_from_eigenvalues,
    main,
)
from ops_scripts.evaluate_visibility_qbeta_shared_local_covariance import (
    _contract_sha256,
    realization_batches,
)


def test_parse_labeled_paths_preserves_redshift_order() -> None:
    assert _parse_labeled_paths(["low=/tmp/a", "high=/tmp/b"]) == [
        ("low", Path("/tmp/a")),
        ("high", Path("/tmp/b")),
    ]


def test_covariance_to_correlation_handles_zero_variance() -> None:
    covariance = np.asarray([[4.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
    correlation = _covariance_to_correlation(covariance)
    assert np.allclose(np.diag(correlation)[:2], 1.0)
    assert np.isclose(correlation[0, 1], 1.0 / 3.0)
    assert np.all(correlation[2] == 0.0)


def test_participation_rank() -> None:
    assert np.isclose(
        _participation_rank_from_eigenvalues(np.ones(4)), 4.0
    )
    assert np.isclose(
        _participation_rank_from_eigenvalues(np.asarray([4.0, 0.0])), 1.0
    )


def test_shared_realization_contract_hash_is_order_independent() -> None:
    assert _contract_sha256({"a": 1, "b": [2, 3]}) == _contract_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_realization_batches_cover_range_with_distinct_seeds() -> None:
    assert realization_batches(130, 64, 11) == [
        (0, 64, 11),
        (64, 128, 104740),
        (128, 130, 209469),
    ]


def test_transform_orientation_is_truth_blind_and_deterministic() -> None:
    transform = np.asarray(
        [[-2.0, 1.0], [-1.0, 1.0], [0.0, -3.0]]
    )
    oriented = _orient_transform_rows(transform)
    assert np.array_equal(
        oriented,
        np.asarray([[2.0, -1.0], [1.0, -1.0], [0.0, 3.0]]),
    )


def test_covariance_main_writes_kl_and_whitened_products(
    tmp_path: Path,
) -> None:
    profile = "quad_kperp_response"
    prefix = f"{profile}_"
    contract = "shared-contract"
    paths = []
    errors = np.asarray(
        [
            [-2.0, -1.0, 0.5, 0.0],
            [-1.0, 1.0, -0.5, 1.0],
            [0.0, 0.5, 0.0, -1.0],
            [1.0, -0.5, 1.0, 0.5],
            [2.0, 0.0, -1.0, -0.5],
            [0.5, 1.5, 0.5, 0.0],
        ]
    )
    target = np.asarray([10.0, 20.0, 30.0, 40.0])
    for index in range(2):
        path = tmp_path / f"window_{index}.npz"
        current = slice(2 * index, 2 * index + 2)
        np.savez_compressed(
            path,
            **{
                f"{prefix}selected": np.ones(2, dtype=np.int8),
                f"{prefix}target": target[current],
                f"{prefix}bank_total_estimate": target[current] * 1.01,
                f"{prefix}heldout_total_estimate": (
                    target[None, current] + errors[:, current]
                ),
                f"{prefix}window": np.eye(2),
                "shared_realization_contract_sha256": np.asarray(contract),
                "shared_realization_source": np.asarray("test source"),
            },
        )
        paths.append(path)
    out_dir = tmp_path / "out"
    main(
        [
            "--window",
            f"low={paths[0]}",
            "--window",
            f"high={paths[1]}",
            "--out-dir",
            str(out_dir),
            "--profile",
            profile,
            "--eigen-rcond",
            "1e-10",
        ]
    )
    summary = json.loads(
        (out_dir / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(out_dir / "products.npz", allow_pickle=False) as archive:
        products = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    retained = summary["retained_eigenmode_count"]
    assert retained == 4
    assert products["kl_transform"].shape == (retained, 4)
    assert products["whitening_matrix"].shape == (retained, 4)
    assert np.allclose(
        products["decorrelated_error_covariance"],
        np.eye(retained),
        atol=1e-12,
    )
    kl_covariance = products["kl_error_covariance"]
    assert np.allclose(
        kl_covariance,
        np.diag(np.diag(kl_covariance)),
        atol=1e-12,
    )


def test_crossfit_covariance_uses_disjoint_realization_folds() -> None:
    generator = np.random.default_rng(7)
    errors = generator.normal(size=(200, 5)) @ np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.5],
        ]
    )
    diagnostics = _crossfit_covariance_diagnostics(
        errors,
        eigen_rcond=1e-8,
    )
    assert len(diagnostics) == 2
    assert all(
        row["train_realization_count"] == 100
        and row["test_realization_count"] == 100
        and row["retained_eigenmode_count"] == 5
        for row in diagnostics
    )
    assert all(
        0.5 < row["test_to_train_variance_ratio_median"] < 1.5
        for row in diagnostics
    )
