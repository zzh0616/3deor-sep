#!/usr/bin/env python3
"""Build a flux-conserving outer-field foreground sky for Q_beta tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PROJECT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ps2d_v2_config import resolve_mode_first_analysis  # noqa: E402
from prepare_independent_eor_512 import (  # noqa: E402
    reference_transform_metrics,
)
from visibility_primary_beam import (  # noqa: E402
    direction_cosine_geometry_sha256,
)
from visibility_qbeta import direction_cosines  # noqa: E402


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fg-cube", type=Path, required=True)
    parser.add_argument("--template-fits", type=Path, required=True)
    parser.add_argument(
        "--reference-inner-cube",
        type=Path,
        help=(
            "Optional canonical central cube used to prove that the outer "
            "field and visibility-bank foreground share one input cube."
        ),
    )
    parser.add_argument("--frequency-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cube-frequency0-mhz", type=float, default=106.0)
    parser.add_argument("--cube-frequency-step-mhz", type=float, default=0.1)
    parser.add_argument("--inner-crop-size", type=int, default=1024)
    parser.add_argument("--outer-downsample", type=int, default=4)
    parser.add_argument("--template-downsample", type=int, default=2)
    parser.add_argument("--maximum-sources", type=int, default=65536)
    parser.add_argument("--minimum-sources-per-region", type=int, default=2048)
    parser.add_argument("--phase-ra-deg", type=float, default=0.0)
    parser.add_argument("--phase-dec-deg", type=float, default=-27.0)
    return parser.parse_args(argv)


def block_average_square(plane: np.ndarray, factor: int) -> np.ndarray:
    """Average a square image in non-overlapping square blocks."""
    values = np.asarray(plane)
    downsample = int(factor)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or downsample < 1
        or values.shape[0] % downsample
    ):
        raise ValueError("Invalid square block-average geometry")
    size = values.shape[0] // downsample
    return np.asarray(values, dtype=np.float64).reshape(
        size, downsample, size, downsample
    ).mean(axis=(1, 3))


def select_bright_outer_sources(
    *,
    score: np.ndarray,
    region_ids: np.ndarray,
    maximum_sources: int,
    minimum_sources_per_region: int,
) -> np.ndarray:
    """Select the brightest directions while retaining every outer region."""
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    regions = np.asarray(region_ids, dtype=np.int64).reshape(-1)
    if (
        values.shape != regions.shape
        or values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("Invalid outer-source score or region IDs")
    maximum = int(maximum_sources)
    minimum = int(minimum_sources_per_region)
    if maximum < 1 or minimum < 0:
        raise ValueError("Source-selection counts must be non-negative")
    maximum = min(maximum, values.size)
    order = np.argsort(values, kind="stable")[::-1]
    selected = np.zeros(values.size, dtype=bool)
    selected[order[:maximum]] = True
    for region in np.unique(regions):
        members = np.flatnonzero(regions == region)
        count = min(minimum, members.size)
        if count:
            region_order = members[
                np.argsort(values[members], kind="stable")[::-1]
            ]
            selected[region_order[:count]] = True
    return np.flatnonzero(selected)


def _k_to_jy_per_pixel(
    frequencies_mhz: np.ndarray, pixel_area_sr: float
) -> np.ndarray:
    boltzmann = 1.380649e-23
    speed_of_light = 299792458.0
    frequencies_hz = np.asarray(frequencies_mhz, dtype=np.float64) * 1e6
    return (
        2.0
        * boltzmann
        * np.square(frequencies_hz)
        / speed_of_light**2
        * float(pixel_area_sr)
        / 1e-26
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _region_metrics(
    *,
    names: tuple[str, ...],
    region_ids: np.ndarray,
    selected_positions: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    selected = np.zeros(score.size, dtype=bool)
    selected[np.asarray(selected_positions, dtype=np.int64)] = True
    rows = []
    for region_id, name in enumerate(names):
        members = region_ids == region_id
        kept = members & selected
        l1 = float(np.sum(score[members]))
        l2 = float(np.sum(np.square(score[members])))
        rows.append(
            {
                "name": name,
                "candidate_count": int(np.count_nonzero(members)),
                "selected_count": int(np.count_nonzero(kept)),
                "retained_l1_fraction": float(
                    np.sum(score[kept]) / max(l1, 1e-300)
                ),
                "retained_l2_fraction": float(
                    np.sum(np.square(score[kept])) / max(l2, 1e-300)
                ),
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config = json.loads(args.frequency_config.read_text(encoding="utf-8"))
    resolved = resolve_mode_first_analysis(config)
    frequencies_mhz = np.asarray(
        resolved.geometry["frequencies_mhz"], dtype=np.float64
    )
    reference_relative_l2 = None
    reference_maximum_absolute = None
    if args.reference_inner_cube is not None:
        (
            reference_relative_l2,
            reference_maximum_absolute,
        ) = reference_transform_metrics(
            args.fg_cube,
            args.reference_inner_cube,
            crop_size=int(args.inner_crop_size),
            downsample=int(args.template_downsample),
        )
        if reference_relative_l2 > 1e-12:
            raise RuntimeError(
                "Full foreground cube does not reproduce the inner reference"
            )
    with fits.open(args.fg_cube, memmap=True) as hdul:
        cube = hdul[0].data
        if cube.ndim != 3 or cube.shape[1] != cube.shape[2]:
            raise ValueError("Foreground cube must have shape [freq,y,x]")
        native_size = int(cube.shape[1])
        factor = int(args.outer_downsample)
        if native_size % factor:
            raise ValueError("Outer downsample does not divide the cube")
        coarse_size = native_size // factor
        inner_crop = int(args.inner_crop_size)
        if (
            inner_crop < 1
            or inner_crop > native_size
            or inner_crop % factor
            or (native_size - inner_crop) % 2
        ):
            raise ValueError("Invalid inner crop for outer-field extraction")
        cube_indices = np.rint(
            (
                frequencies_mhz
                - float(args.cube_frequency0_mhz)
            )
            / float(args.cube_frequency_step_mhz)
        ).astype(np.int64)
        reconstructed = (
            float(args.cube_frequency0_mhz)
            + cube_indices * float(args.cube_frequency_step_mhz)
        )
        if (
            np.any(cube_indices < 0)
            or np.any(cube_indices >= cube.shape[0])
            or not np.allclose(
                reconstructed, frequencies_mhz, rtol=0.0, atol=1e-9
            )
        ):
            raise ValueError("Requested frequencies are outside the cube")
        reference_position = int(frequencies_mhz.size // 2)
        reference_plane = block_average_square(
            cube[cube_indices[reference_position]], factor
        )

        first_inner = (native_size - inner_crop) // (2 * factor)
        stop_inner = first_inner + inner_crop // factor
        yy, xx = np.indices((coarse_size, coarse_size), dtype=np.int64)
        outside = ~(
            (yy >= first_inner)
            & (yy < stop_inner)
            & (xx >= first_inner)
            & (xx < stop_inner)
        )
        candidate_y = yy[outside]
        candidate_x = xx[outside]
        score = np.abs(reference_plane[outside])

        template_header = fits.getheader(args.template_fits)
        template_wcs = WCS(template_header).celestial
        template_scale_deg = float(
            np.mean(np.abs(proj_plane_pixel_scales(template_wcs)))
        )
        template_factor = int(args.template_downsample)
        if template_factor < 1:
            raise ValueError("--template-downsample must be positive")
        first_native = (native_size - inner_crop) // 2
        native_centres_x = factor * candidate_x + 0.5 * (factor - 1)
        native_centres_y = factor * candidate_y + 0.5 * (factor - 1)
        template_x = (
            native_centres_x
            - first_native
            - 0.5 * (template_factor - 1)
        ) / template_factor
        template_y = (
            native_centres_y
            - first_native
            - 0.5 * (template_factor - 1)
        ) / template_factor
        ra_deg, dec_deg = template_wcs.all_pix2world(
            template_x, template_y, 0
        )
        candidate_l, candidate_m, candidate_n = direction_cosines(
            ra_deg,
            dec_deg,
            phase_ra_deg=float(args.phase_ra_deg),
            phase_dec_deg=float(args.phase_dec_deg),
        )
        radius_deg = np.degrees(
            np.arccos(np.clip(candidate_n, -1.0, 1.0))
        )
        patch_corner_deg = float(
            resolved.geometry["source_corner_angle_deg"]
        )
        region_names = (
            "outside_square_inside_patch_corner",
            "patch_corner_to_5deg",
            "beyond_5deg_within_cube",
        )
        region_ids = np.where(
            radius_deg < patch_corner_deg,
            0,
            np.where(radius_deg < 5.0, 1, 2),
        ).astype(np.int64)
        selected_positions = select_bright_outer_sources(
            score=score,
            region_ids=region_ids,
            maximum_sources=int(args.maximum_sources),
            minimum_sources_per_region=int(
                args.minimum_sources_per_region
            ),
        )
        selected_y = candidate_y[selected_positions]
        selected_x = candidate_x[selected_positions]
        fg_k = np.empty(
            (frequencies_mhz.size, selected_positions.size),
            dtype=np.float64,
        )
        for output_index, cube_index in enumerate(cube_indices):
            plane = block_average_square(cube[cube_index], factor)
            fg_k[output_index] = plane[selected_y, selected_x]

    coarse_scale_deg = (
        template_scale_deg * factor / int(args.template_downsample)
    )
    pixel_area_sr = math.radians(coarse_scale_deg) ** 2
    k2jy = _k_to_jy_per_pixel(frequencies_mhz, pixel_area_sr)
    selected_l = candidate_l[selected_positions]
    selected_m = candidate_m[selected_positions]
    selected_n = candidate_n[selected_positions]
    selected_regions = region_ids[selected_positions]
    _atomic_npz(
        args.out,
        {
            "frequencies_mhz": frequencies_mhz,
            "l_cosine": selected_l.astype(np.float64),
            "m_cosine": selected_m.astype(np.float64),
            "n_minus_one": (selected_n - 1.0).astype(np.float64),
            "fg_jy": fg_k * k2jy[:, None],
            "k2jy_per_pixel": k2jy,
            "region_id": selected_regions,
            "coarse_y": selected_y,
            "coarse_x": selected_x,
            "candidate_position": selected_positions,
        },
    )
    metadata = {
        "schema": "visibility_qbeta_outer_field_sky",
        "schema_version": 1,
        "fg_cube": str(args.fg_cube),
        "fg_cube_sha256": _sha256(args.fg_cube),
        "template_fits": str(args.template_fits),
        "template_fits_sha256": _sha256(args.template_fits),
        "reference_inner_cube": (
            None
            if args.reference_inner_cube is None
            else str(args.reference_inner_cube)
        ),
        "reference_inner_cube_sha256": (
            None
            if args.reference_inner_cube is None
            else _sha256(args.reference_inner_cube)
        ),
        "reference_inner_relative_l2": reference_relative_l2,
        "reference_inner_maximum_absolute": reference_maximum_absolute,
        "frequency_config": str(args.frequency_config),
        "frequencies_mhz": frequencies_mhz.tolist(),
        "native_image_size": native_size,
        "inner_crop_size": int(args.inner_crop_size),
        "outer_downsample": int(args.outer_downsample),
        "template_downsample": int(args.template_downsample),
        "coarse_image_size": coarse_size,
        "coarse_pixel_scale_arcsec": coarse_scale_deg * 3600.0,
        "candidate_source_count": int(score.size),
        "selected_source_count": int(selected_positions.size),
        "maximum_sources": int(args.maximum_sources),
        "minimum_sources_per_region": int(
            args.minimum_sources_per_region
        ),
        "selection_frequency_mhz": float(
            frequencies_mhz[reference_position]
        ),
        "patch_corner_angle_deg": patch_corner_deg,
        "maximum_selected_angle_deg": float(
            np.max(radius_deg[selected_positions])
        ),
        "region_metrics": _region_metrics(
            names=region_names,
            region_ids=region_ids,
            selected_positions=selected_positions,
            score=score,
        ),
        "direction_sha256": direction_cosine_geometry_sha256(
            l_cosine=selected_l,
            m_cosine=selected_m,
            n_minus_one=selected_n - 1.0,
        ),
        "out": str(args.out),
    }
    metadata_path = args.out.with_suffix(".json")
    metadata["out_sha256"] = _sha256(args.out)
    _atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
