#!/usr/bin/env python3
"""Create the canonical 512-pixel EoR view from a 2048-pixel cube."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--reference-input", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument(
        "--validate-reference-only",
        action="store_true",
        help="Validate the reference crop contract without writing an output.",
    )
    return parser.parse_args(argv)


def central_block_average(
    plane: np.ndarray,
    *,
    crop_size: int,
    downsample: int,
) -> np.ndarray:
    """Centrally crop a square plane and average non-overlapping blocks."""
    values = np.asarray(plane)
    crop = int(crop_size)
    factor = int(downsample)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or crop < 1
        or crop > values.shape[0]
        or crop % factor
        or factor < 1
    ):
        raise ValueError("Invalid square crop/downsample geometry")
    first = (values.shape[0] - crop) // 2
    selected = np.asarray(
        values[first : first + crop, first : first + crop],
        dtype=np.float64,
    )
    output_size = crop // factor
    return selected.reshape(
        output_size, factor, output_size, factor
    ).mean(axis=(1, 3))


def _transform_cube(
    input_path: Path,
    *,
    crop_size: int,
    downsample: int,
) -> tuple[np.ndarray, fits.Header, tuple[int, ...]]:
    with fits.open(input_path, memmap=True) as hdul:
        source = hdul[0].data
        if source.ndim != 3:
            raise ValueError("Input EoR cube must have shape [freq,y,x]")
        input_shape = tuple(int(value) for value in source.shape)
        output_size = int(crop_size) // int(downsample)
        output = np.empty(
            (source.shape[0], output_size, output_size),
            dtype=np.float64,
        )
        for index in range(source.shape[0]):
            output[index] = central_block_average(
                source[index],
                crop_size=int(crop_size),
                downsample=int(downsample),
            )
        header = hdul[0].header.copy()
    header["HISTORY"] = (
        f"central {int(crop_size)} crop; "
        f"{int(downsample)}x{int(downsample)} block mean"
    )
    return output, header, input_shape


def reference_transform_metrics(
    reference_input: Path,
    reference_output: Path,
    *,
    crop_size: int,
    downsample: int,
) -> tuple[float, float]:
    """Compare a reference transform plane by plane without a large copy."""
    residual_squared = 0.0
    reference_squared = 0.0
    maximum_absolute = 0.0
    with fits.open(reference_input, memmap=True) as input_hdul, fits.open(
        reference_output, memmap=True
    ) as output_hdul:
        source = input_hdul[0].data
        reference = output_hdul[0].data
        expected_size = int(crop_size) // int(downsample)
        expected_shape = (source.shape[0], expected_size, expected_size)
        if source.ndim != 3 or reference.shape != expected_shape:
            raise ValueError("Reference input/output geometry differs")
        for index in range(source.shape[0]):
            transformed = central_block_average(
                source[index],
                crop_size=int(crop_size),
                downsample=int(downsample),
            )
            target = np.asarray(reference[index], dtype=np.float64)
            residual = transformed - target
            residual_squared += float(np.sum(np.square(residual)))
            reference_squared += float(np.sum(np.square(target)))
            maximum_absolute = max(
                maximum_absolute, float(np.max(np.abs(residual)))
            )
    return (
        math.sqrt(residual_squared / max(reference_squared, 1e-300)),
        maximum_absolute,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if (args.reference_input is None) != (
        args.reference_output is None
    ):
        raise ValueError(
            "--reference-input and --reference-output must coexist"
        )
    if args.validate_reference_only and args.reference_input is None:
        raise ValueError(
            "--validate-reference-only requires the reference pair"
        )
    if not args.validate_reference_only and (
        args.input is None or args.out is None
    ):
        raise ValueError("--input and --out are required when writing")
    reference_relative_l2 = None
    reference_maximum_absolute = None
    if args.reference_input is not None:
        (
            reference_relative_l2,
            reference_maximum_absolute,
        ) = reference_transform_metrics(
            args.reference_input,
            args.reference_output,
            crop_size=int(args.crop_size),
            downsample=int(args.downsample),
        )
        if reference_relative_l2 > 1e-12:
            raise RuntimeError(
                "Crop/downsample contract does not reproduce the reference"
            )
    output_shape = None
    input_shape = None
    if not args.validate_reference_only:
        assert args.input is not None
        assert args.out is not None
        output, header, input_shape = _transform_cube(
            args.input,
            crop_size=int(args.crop_size),
            downsample=int(args.downsample),
        )
        output_shape = tuple(int(value) for value in output.shape)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(
            args.out.suffix + f".tmp.{os.getpid()}"
        )
        fits.PrimaryHDU(data=output, header=header).writeto(
            temporary, overwrite=True
        )
        temporary.replace(args.out)
    result = {
        "schema": "independent_eor_512_preparation",
        "schema_version": 1,
        "input": None if args.input is None else str(args.input),
        "out": None if args.out is None else str(args.out),
        "input_shape": (
            None if input_shape is None else [int(value) for value in input_shape]
        ),
        "output_shape": (
            None
            if output_shape is None
            else [int(value) for value in output_shape]
        ),
        "crop_size": int(args.crop_size),
        "downsample": int(args.downsample),
        "validate_reference_only": bool(args.validate_reference_only),
        "reference_relative_l2": reference_relative_l2,
        "reference_maximum_absolute": reference_maximum_absolute,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
