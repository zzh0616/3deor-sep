from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from visibility_primary_beam import (
    build_oskar_circular_gaussian_kernel_multiplier,
    build_time_direction_kernel_multiplier,
    open_frequency_row_direction_kernel_multiplier,
    open_indexed_frequency_row_direction_kernel_multiplier,
    open_row_direction_kernel_multiplier,
    oskar_circular_gaussian_stokes_i_power,
)


def test_oskar_gaussian_matches_measured_visibility_ratios() -> None:
    separation_deg = np.asarray(
        [0.0, 0.9999967047519429, 2.499948503787396, 3.9997890108761864]
    )
    measured = np.asarray(
        [1.0, 0.800637522658, 0.249354465069, 0.0286733985164]
    )
    response = oskar_circular_gaussian_stokes_i_power(
        frequencies_hz=np.asarray([119.4e6]),
        l_cosine=np.sin(np.deg2rad(separation_deg)),
        m_cosine=np.zeros(separation_deg.size),
        fwhm_deg=5.0,
        reference_frequency_hz=119.4e6,
    )
    np.testing.assert_allclose(response[0], measured, rtol=0.0, atol=2e-9)


def test_oskar_gaussian_frequency_scaling_and_centre_normalisation() -> None:
    response = oskar_circular_gaussian_stokes_i_power(
        frequencies_hz=np.asarray([100.0e6, 200.0e6]),
        l_cosine=np.sin(np.deg2rad([0.0, 1.0])),
        m_cosine=np.zeros(2),
        fwhm_deg=5.0,
        reference_frequency_hz=100.0e6,
    )
    np.testing.assert_array_equal(response[:, 0], np.ones(2))
    assert response[1, 1] < response[0, 1]


def test_gaussian_kernel_multiplier_broadcasts_over_rows() -> None:
    import torch

    multiplier = build_oskar_circular_gaussian_kernel_multiplier(
        torch=torch,
        frequencies_hz=np.asarray([100.0e6]),
        l_cosine=np.asarray([0.0, 0.01, 0.02]),
        m_cosine=np.zeros(3),
        fwhm_deg=5.0,
        reference_frequency_hz=100.0e6,
        device=torch.device("cpu"),
        operator_dtype="complex128",
    )
    block = multiplier(
        0,
        4,
        9,
        1,
        3,
        torch.device("cpu"),
        torch.complex128,
    )
    assert tuple(block.shape) == (1, 2)
    assert block.dtype == torch.float64


def test_time_direction_multiplier_selects_each_row_time() -> None:
    import torch

    values = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    multiplier = build_time_direction_kernel_multiplier(
        torch=torch,
        values=values,
        row_time_indices=np.asarray([2, 0, 1]),
        device=torch.device("cpu"),
        operator_dtype="complex128",
    )
    block = multiplier(
        1, 0, 2, 1, 4, torch.device("cpu"), torch.complex128
    )
    np.testing.assert_array_equal(
        block.numpy(),
        values[1, np.asarray([2, 0]), 1:4],
    )


def test_row_direction_file_multiplier_streams_complex_blocks() -> None:
    import torch

    values = (
        np.arange(15, dtype=np.float32).reshape(3, 5)
        + 1j * np.arange(15, dtype=np.float32).reshape(3, 5)[::-1]
    ).astype(np.complex64)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        values.tofile(directory / "coherency.complex64.bin")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "schema": "oskar_aperture_row_direction_coherency",
                    "data_file": "coherency.complex64.bin",
                    "dtype": "complex64",
                    "shape": [3, 5],
                }
            ),
            encoding="utf-8",
        )
        multiplier, metadata = open_row_direction_kernel_multiplier(directory)
        block = multiplier(
            0,
            1,
            3,
            2,
            5,
            torch.device("cpu"),
            torch.complex64,
        )
    assert metadata["shape"] == [3, 5]
    np.testing.assert_array_equal(block.numpy(), values[1:3, 2:5])


def test_frequency_row_direction_multiplier_selects_cache() -> None:
    import torch

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directories = []
        expected = []
        for index in range(2):
            directory = root / f"freq_{index}"
            directory.mkdir()
            values = np.full(
                (2, 4), index + 1j * (index + 1), dtype=np.complex64
            )
            values.tofile(directory / "beam.bin")
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema": "oskar_aperture_row_direction_coherency",
                        "data_file": "beam.bin",
                        "dtype": "complex64",
                        "shape": [2, 4],
                    }
                ),
                encoding="utf-8",
            )
            directories.append(directory)
            expected.append(values)
        multiplier, metadata = (
            open_frequency_row_direction_kernel_multiplier(directories)
        )
        block = multiplier(
            1,
            0,
            2,
            1,
            4,
            torch.device("cpu"),
            torch.complex64,
        )
    assert len(metadata) == 2
    np.testing.assert_array_equal(block.numpy(), expected[1][:, 1:4])


def test_indexed_frequency_multiplier_selects_partition_rows() -> None:
    import torch

    cached_rows = np.asarray([10, 20, 30, 40], dtype=np.int64)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directories = []
        expected = []
        for index in range(2):
            directory = root / f"freq_{index}"
            directory.mkdir()
            values = (
                np.arange(20, dtype=np.float32).reshape(4, 5)
                + 100 * index
            ).astype(np.complex64)
            values.tofile(directory / "beam.bin")
            np.savez_compressed(
                directory / "geometry.npz",
                selected_bank_rows=cached_rows,
            )
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema": "oskar_aperture_row_direction_coherency",
                        "data_file": "beam.bin",
                        "dtype": "complex64",
                        "shape": [4, 5],
                    }
                ),
                encoding="utf-8",
            )
            directories.append(directory)
            expected.append(values)
        multiplier, metadata, rows = (
            open_indexed_frequency_row_direction_kernel_multiplier(
                directories,
                selected_bank_rows=np.asarray([30, 10]),
            )
        )
        block = multiplier(
            1,
            0,
            2,
            1,
            4,
            torch.device("cpu"),
            torch.complex64,
        )
    assert len(metadata) == 2
    assert all(np.array_equal(value, cached_rows) for value in rows)
    np.testing.assert_array_equal(
        block.numpy(),
        expected[1][np.asarray([2, 0]), 1:4],
    )
