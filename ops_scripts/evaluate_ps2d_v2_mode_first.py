#!/usr/bin/env python3
"""Evaluate mode-first full-plane and EoR-window cylindrical power spectra."""

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
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ps2d_v2 import (  # noqa: E402
    PS2DProducts,
    analysis_contract_sha256,
    compare_bandpowers,
    compute_ps2d_products,
    mode_layout_sha256,
)
from ps2d_v2_config import (  # noqa: E402
    resolve_mode_first_analysis,
    resolve_mode_first_geometry,
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


def _parse_named_pattern(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cube must use NAME=FITS_PATTERN")
    name, pattern = value.split("=", 1)
    if not name.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError("cube name and FITS pattern must be non-empty")
    return name.strip(), pattern.strip()


def _format_pattern(pattern: str, frequency_mhz: float) -> Path:
    freqtag = f"{float(frequency_mhz):.2f}".replace(".", "")
    return Path(
        str(pattern).format(freq=float(frequency_mhz), freqtag=freqtag)
    )


def _central_crop(array: np.ndarray, size: int) -> np.ndarray:
    image = np.squeeze(np.asarray(array, dtype=np.float64))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS plane, got {image.shape}")
    if int(size) > min(image.shape):
        raise ValueError(f"Cannot crop {size} pixels from {image.shape}")
    y0 = (image.shape[0] - int(size)) // 2
    x0 = (image.shape[1] - int(size)) // 2
    return np.asarray(image[y0 : y0 + size, x0 : x0 + size], dtype=np.float64)


def _load_pattern_cube(
    pattern: str,
    frequencies_mhz: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    planes: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frequency in frequencies_mhz:
        path = _format_pattern(pattern, float(frequency))
        if not path.is_file():
            raise FileNotFoundError(path)
        planes.append(_central_crop(fits.getdata(path), int(crop_size)))
        records.append(
            {
                "frequency_mhz": float(frequency),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )
    cube = np.stack(planes, axis=0)
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Non-finite values loaded from {pattern}")
    return cube, records


def _geometry(config: dict[str, Any]) -> dict[str, Any]:
    return resolve_mode_first_geometry(config)


def _layout_hash(products: PS2DProducts) -> str:
    return mode_layout_sha256(products.full_layout, products.window_layout)


def _analysis_contract_hash(products: PS2DProducts) -> str:
    fft = products.fft_metadata
    return analysis_contract_sha256(
        layout_sha256=_layout_hash(products),
        demean_mode=fft["demean_mode"],
        radial_taper=fft["radial_taper"],
        spatial_taper=fft["spatial_taper"],
        window_energy=fft["window_energy"],
        voxel_volume_mpc3=fft["voxel_volume_mpc3"],
        power_scale=fft["power_scale"],
    )


def _product_summary(products: PS2DProducts) -> dict[str, Any]:
    full_layout = products.full_layout
    window_layout = products.window_layout
    power_flat = products.power_cube.reshape(-1)
    full_direct = float(
        np.sum(power_flat[full_layout.full_mode_indices], dtype=np.float64)
    )
    window_direct = float(
        np.sum(power_flat[window_layout.selected_mode_indices], dtype=np.float64)
    )
    rectangular_power = max(products.window_rectangular.total_power, 1e-300)
    full_power = max(products.full.total_power, 1e-300)
    partial = (
        (window_layout.selected_mode_fraction > 0.0)
        & (window_layout.selected_mode_fraction < 1.0)
    )
    partial_records = []
    for kperp_index, kpar_index in np.argwhere(partial):
        partial_records.append(
            {
                "kperp_index": int(kperp_index),
                "kpar_index": int(kpar_index),
                "kperp_edge_mpc_inv": [
                    float(window_layout.kperp_edges[kperp_index]),
                    float(window_layout.kperp_edges[kperp_index + 1]),
                ],
                "kpar_value_mpc_inv": float(
                    window_layout.kpar_values[kpar_index]
                ),
                "selected_fft_mode_count": int(
                    window_layout.selected_fft_mode_counts[kperp_index, kpar_index]
                ),
                "rectangular_fft_mode_count": int(
                    window_layout.full_fft_mode_counts[kperp_index, kpar_index]
                ),
                "selected_mode_fraction": float(
                    window_layout.selected_mode_fraction[kperp_index, kpar_index]
                ),
                "selected_power_fraction_of_full": float(
                    products.window.power_sum[kperp_index, kpar_index] / full_power
                ),
            }
        )
    return {
        "layout_sha256": _layout_hash(products),
        "analysis_contract_sha256": _analysis_contract_hash(products),
        "fft": products.fft_metadata,
        "full": {
            "populated_band_count": int(np.sum(products.full.fft_mode_counts > 0)),
            "fft_mode_count": int(np.sum(products.full.fft_mode_counts)),
            "conjugate_unique_mode_count": int(
                np.sum(products.full.independent_mode_counts)
            ),
            "power_sum": products.full.total_power,
            "direct_mode_power_sum": full_direct,
            "aggregation_relative_error": products.full.total_power
            / max(full_direct, 1e-300)
            - 1.0,
        },
        "window_rectangle": {
            "populated_band_count": int(
                np.sum(products.window_rectangular.fft_mode_counts > 0)
            ),
            "fft_mode_count": int(
                np.sum(products.window_rectangular.fft_mode_counts)
            ),
            "power_sum": products.window_rectangular.total_power,
        },
        "science_window": {
            "populated_band_count": int(np.sum(products.window.fft_mode_counts > 0)),
            "partially_selected_band_count": int(np.sum(partial)),
            "fft_mode_count": int(np.sum(products.window.fft_mode_counts)),
            "conjugate_unique_mode_count": int(
                np.sum(products.window.independent_mode_counts)
            ),
            "selected_mode_fraction_of_window_rectangle": float(
                np.sum(products.window.fft_mode_counts)
                / max(np.sum(products.window_rectangular.fft_mode_counts), 1)
            ),
            "power_sum": products.window.total_power,
            "power_fraction_of_window_rectangle": products.window.total_power
            / rectangular_power,
            "power_fraction_of_full": products.window.total_power / full_power,
            "direct_mode_power_sum": window_direct,
            "aggregation_relative_error": products.window.total_power
            / max(window_direct, 1e-300)
            - 1.0,
            "partially_selected_bins": partial_records,
        },
    }


def _npz_arrays(
    products_by_name: dict[str, PS2DProducts],
) -> dict[str, np.ndarray]:
    first = next(iter(products_by_name.values()))
    arrays: dict[str, np.ndarray] = {
        "full_kperp_edges": first.full_layout.kperp_edges,
        "full_kperp_centers": first.full_layout.kperp_centers,
        "full_kpar_values": first.full_layout.kpar_values,
        "full_kpar_display_edges": first.full_layout.kpar_edges,
        "full_fft_mode_counts": first.full_layout.full_fft_mode_counts,
        "full_conjugate_unique_mode_counts": first.full_layout.full_independent_mode_counts,
        "window_kperp_edges": first.window_layout.kperp_edges,
        "window_kperp_centers": first.window_layout.kperp_centers,
        "window_kpar_values": first.window_layout.kpar_values,
        "window_kpar_display_edges": first.window_layout.kpar_edges,
        "window_rectangular_fft_mode_counts": first.window_layout.full_fft_mode_counts,
        "window_selected_fft_mode_counts": first.window_layout.selected_fft_mode_counts,
        "window_selected_conjugate_unique_mode_counts": first.window_layout.selected_independent_mode_counts,
        "window_selected_mode_fraction": first.window_layout.selected_mode_fraction,
        "window_rectangular_kperp_mode_mean": first.window_layout.full_kperp_mode_mean,
        "window_selected_kperp_mode_mean": first.window_layout.selected_kperp_mode_mean,
        "window_selected_kpar_mode_mean": first.window_layout.selected_kpar_mode_mean,
    }
    for name, products in products_by_name.items():
        prefix = str(name).replace("-", "_")
        arrays[f"{prefix}_full_power_mean"] = products.full.mean
        arrays[f"{prefix}_full_power_sum"] = products.full.power_sum
        arrays[f"{prefix}_full_within_bin_std"] = products.full.within_bin_std
        arrays[f"{prefix}_window_power_mean"] = products.window.mean
        arrays[f"{prefix}_window_power_sum"] = products.window.power_sum
        arrays[f"{prefix}_window_within_bin_std"] = products.window.within_bin_std
    return arrays


def _write_report(path: Path, result: dict[str, Any]) -> None:
    geometry = result["geometry"]
    lines = [
        "# PS2D v2 mode-first 验证报告",
        "",
        f"生成时间：`{result['time_utc']}`。",
        "",
        "## 固定定义",
        "",
        "- `full_ps2d` 只用于完整诊断；`window_ps2d` 是逐 Fourier mode 通过 floor、wedge、buffer 和 UV support 后再聚合的科学产品。",
        "- $k_\\parallel$ 直接使用离散 FFT 的绝对 mode 值；径向 Nyquist 的保留策略显式记录。",
        "- 每个 band 同时保存 mode power 的均值与精确求和；跨 band 的总功率比只使用后者。",
        "- 横向 bin 为左闭右开，最后一个右边界显式包含；部分相交 bin 不整体删除。",
        "",
        "## 几何",
        "",
        f"- 实际 $k_\\parallel$：`{geometry['kpar_values_mpc_inv']}` Mpc^-1。",
        f"- $d_\\parallel={geometry['radial_spacing_mpc']:.9g}$ Mpc；各频率间距相对均值最大偏差为 `{geometry['radial_spacing_max_relative_deviation']:.4%}`。",
        f"- 科学 $k_\\perp$ support：`{geometry['window_kperp_range_mpc_inv']}` Mpc^-1，共 `{geometry['window_kperp_bins']}` bins。",
        f"- 有限源图 patch wedge slope 为 `{geometry['patch_wedge_slope']:.9g}`，buffer 为 `{geometry['wedge_buffer_mpc_inv']:.9g}` Mpc^-1。",
        "",
        "## 数据闭环",
        "",
    ]
    for name, record in result["cubes"].items():
        science = record["summary"]["science_window"]
        full = record["summary"]["full"]
        fft = record["summary"]["fft"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Parseval 相对误差：`{fft['parseval_relative_error']:.3e}`；full/window 聚合相对误差：`{full['aggregation_relative_error']:.3e}` / `{science['aggregation_relative_error']:.3e}`。",
                f"- science window 保留 `{science['fft_mode_count']}` 个 FFT modes、`{science['conjugate_unique_mode_count']}` 个共轭唯一 modes，覆盖 `{science['populated_band_count']}` 个 bands。",
                f"- 与 window rectangle 相交的部分 bins 为 `{science['partially_selected_band_count']}` 个；这些 bins 仅聚合真正过窗的 modes。",
                f"- science-window power/full diagnostic power = `{science['power_fraction_of_full']:.8f}`。该值只用于注入真值的事后覆盖审计，不参与窗口定义。",
                "",
            ]
        )
    lines.extend(
        [
            "## 使用限制",
            "",
            "当前配置的 finite-source-patch wedge 只适用于本次无 PB、有限 512 像素源图仿真。真实 SKA 全视场分析必须换成由相位中心、beam/视场和 baseline chromaticity 冻结的窗口；不得从注入 EoR 表现反推窗口。",
            "",
            "旧 `powerspec.py` 入口仅保留历史复现。后续 signal、probe、Fisher/window calibration 必须共享本文件记录的同一个 analysis-contract hash，不能混用 legacy bin-center mask。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_DIR / "configs/ps2d_v2_8wide_isobeam_patch.json",
    )
    parser.add_argument(
        "--cube",
        action="append",
        type=_parse_named_pattern,
        required=True,
        metavar="NAME=FITS_PATTERN",
    )
    parser.add_argument("--reference-name")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "ps2d_v2_mode_first_config" or int(
        config.get("schema_version", -1)
    ) != 2:
        raise ValueError("Expected a PS2D v2 mode-first config")
    named_patterns = dict(args.cube)
    if len(named_patterns) != len(args.cube):
        raise ValueError("Cube names must be unique")
    reference_name = args.reference_name or next(iter(named_patterns))
    if reference_name not in named_patterns:
        raise KeyError(f"Unknown reference cube: {reference_name}")

    resolved = resolve_mode_first_analysis(config)
    geometry = resolved.geometry
    contract = resolved.contract
    frequencies = geometry["frequencies_mhz"]
    image = config["image_geometry"]
    analysis = config["analysis"]
    crop_size = int(image["eval_crop_size"])
    dx_mpc = float(geometry["spatial_spacing_mpc"])
    dy_mpc = dx_mpc
    dpar_mpc = float(geometry["radial_spacing_mpc"])
    transverse_circle_max = float(geometry["transverse_circle_max_mpc_inv"])
    window_kperp_min, window_kperp_max = (
        float(value) for value in geometry["window_kperp_range_mpc_inv"]
    )
    full_edges = contract.full_layout.kperp_edges
    window_edges = contract.window_layout.kperp_edges
    window_spec = resolved.window_spec

    products_by_name: dict[str, PS2DProducts] = {}
    cube_summaries: dict[str, dict[str, Any]] = {}
    for name, pattern in named_patterns.items():
        cube, records = _load_pattern_cube(pattern, frequencies, crop_size)
        products = compute_ps2d_products(
            cube,
            dx_mpc=dx_mpc,
            dy_mpc=dy_mpc,
            dpar_mpc=dpar_mpc,
            full_kperp_edges=full_edges,
            window_kperp_edges=window_edges,
            window_spec=window_spec,
            radial_nyquist_policy=str(analysis["radial_nyquist_policy"]),
            demean_mode=str(analysis["demean_mode"]),
            radial_taper=str(analysis["radial_taper"]),
            spatial_taper=str(analysis["spatial_taper"]),
        )
        products_by_name[name] = products
        cube_summaries[name] = {
            "pattern": pattern,
            "files": records,
            "summary": _product_summary(products),
        }

    reference = products_by_name[reference_name]
    comparisons: dict[str, Any] = {}
    for name, products in products_by_name.items():
        if name == reference_name:
            continue
        if _layout_hash(products) != _layout_hash(reference):
            raise ValueError("All cubes must share an identical PS2D layout")
        if _analysis_contract_hash(products) != _analysis_contract_hash(reference):
            raise ValueError("All cubes must share an identical analysis contract")
        comparisons[name] = {
            "reference_name": reference_name,
            "full": compare_bandpowers(products.full, reference.full),
            "science_window": compare_bandpowers(products.window, reference.window),
        }

    first = next(iter(products_by_name.values()))
    if _layout_hash(first) != contract.layout_sha256:
        raise ValueError("Evaluator layout does not match the resolved v2 contract")
    if _analysis_contract_hash(first) != contract.analysis_contract_sha256:
        raise ValueError("Evaluator analysis settings do not match the v2 contract")
    output = {
        "time_utc": _now(),
        "method": "ps2d_v2_mode_first",
        "schema_version": 2,
        "implementation": {
            "core_path": str((CODE_DIR / "ps2d_v2.py").resolve()),
            "core_sha256": _sha256(CODE_DIR / "ps2d_v2.py"),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
        },
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "reference_name": reference_name,
        "layout_sha256": contract.layout_sha256,
        "analysis_contract_sha256": contract.analysis_contract_sha256,
        "geometry": {
            **geometry,
            "cube_shape": [int(frequencies.size), crop_size, crop_size],
            "transverse_circle_max_mpc_inv": transverse_circle_max,
            "full_kperp_bins": int(analysis["full_kperp_bins"]),
            "window_kperp_bins": int(analysis["window_kperp_bins"]),
            "window_kperp_range_mpc_inv": [window_kperp_min, window_kperp_max],
            "kpar_values_mpc_inv": first.full_layout.kpar_values,
            "kpar_display_edges_mpc_inv": first.full_layout.kpar_edges,
            "radial_nyquist_policy": first.full_layout.radial_nyquist_policy,
            "bin_edge_convention": "left_closed_right_open_last_right_inclusive",
        },
        "window_definition": {
            "defined_without_eor_truth": True,
            "profile": config["eor_window"]["profile"],
            "kpar_floor_mpc_inv": geometry["kpar_floor_mpc_inv"],
            "wedge_slope": geometry["patch_wedge_slope"],
            "wedge_intercept_mpc_inv": geometry["wedge_buffer_mpc_inv"],
            "kperp_min_mpc_inv": window_kperp_min,
            "kperp_max_mpc_inv": window_kperp_max,
            "exclude_exact_dc": window_spec.exclude_exact_dc,
        },
        "analysis": analysis,
        "legacy_reproduction": config.get("legacy_reproduction"),
        "cubes": cube_summaries,
        "comparisons": comparisons,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "result.json", output)
    _atomic_npz(args.out_dir / "bandpowers.npz", _npz_arrays(products_by_name))
    _write_report(args.out_dir / "report.md", _json_safe(output))
    print(
        json.dumps(
            {
                "event": "ps2d_v2_mode_first_done",
                "out_dir": str(args.out_dir),
                "layout_sha256": _layout_hash(first),
                "analysis_contract_sha256": _analysis_contract_hash(first),
                "time_utc": _now(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
