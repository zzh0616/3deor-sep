#!/usr/bin/env python3
"""
Generate multi-band (different frequency start) sky cubes using e2esim.

This script is intentionally project-local (under 3dnet/) to keep the workflow
reproducible without creating ad-hoc scripts elsewhere.

Outputs:
- e2esim per-frequency component FITS under: <work_root>/e2esim_runs/<run_name>/fXXX/sky_model/...
- stacked cubes (fg/eor/all) under: <work_root>/data/<out_data_subdir>/fXXX/{fg_cube,eor_cube,all_cube}.fits

Notes:
- We only use e2esim sky maps (no OSKAR/WSClean).
- We do NOT apply any spatial mask for evaluation purposes; cube1 is known to
  contain very bright point-source pixels and should be *annotated* in reports
  instead of being masked.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class BandSpec:
    start_mhz: float
    nchan: int
    step_mhz: float

    @property
    def stop_mhz(self) -> float:
        return float(self.start_mhz) + float(self.step_mhz) * float(int(self.nchan) - 1)


def _fmt_band_label(start_mhz: float) -> str:
    # Keep filenames stable and simple. We only expect integer starts here (60,80,...,180),
    # but handle fractional starts defensively.
    if abs(float(start_mhz) - round(float(start_mhz))) < 1e-9:
        return f"f{int(round(float(start_mhz))):03d}"
    s = f"{float(start_mhz):.3f}".rstrip("0").rstrip(".")
    return "f" + s.replace("-", "m").replace(".", "p")


def parse_float_csv(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def generate_config_from_template(
    *,
    template_path: Path,
    output_path: Path,
    work_root: Path,
    run_name: str,
    band: BandSpec,
    sky_xsize: int,
    sky_ysize: int,
    sky_pixelsize_arcsec: float,
    component_backend: str,
    postprocess_enabled: bool,
) -> None:
    """
    Generate an e2esim config by applying targeted replacements on a known-good template.

    We avoid fully parsing the nested config format; this keeps the generator robust
    to minor upstream schema changes while still being explicit about what we modify.
    """
    text = template_path.read_text(encoding="utf-8")
    lines = text.splitlines(True)

    band_label = _fmt_band_label(band.start_mhz)
    base_dir = (work_root / "e2esim_runs" / run_name / band_label).resolve()
    work_dir = base_dir
    log_dir = base_dir / "logs"
    sky_dir = base_dir / "sky_model"

    def replace_in_section(section: str, key: str, value: str) -> None:
        nonlocal lines
        out: List[str] = []
        in_sec = False
        sec_pat = re.compile(r"^\[" + re.escape(section) + r"\]\s*$")
        any_pat = re.compile(r"^\[[^\]]+\]\s*$")
        key_pat = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*).*$")
        replaced = False
        for ln in lines:
            if sec_pat.match(ln.strip()):
                in_sec = True
                out.append(ln)
                continue
            if in_sec and any_pat.match(ln.strip()):
                in_sec = False
            if in_sec:
                m = key_pat.match(ln)
                if m:
                    out.append(m.group(1) + value + "\n")
                    replaced = True
                    continue
            out.append(ln)
        if not replaced:
            raise RuntimeError(f"Failed to replace '{key}' in section [{section}] in template: {template_path}")
        lines = out

    def replace_in_subsection(parent: str, subsection: str, key: str, value: str) -> None:
        """
        Replace `key = ...` inside a nested config block:

          [parent]
            [[subsection]]
            key = ...

        This is needed for per-component cache paths like:
        - [galactic][[synchrotron]] cache_dir
        - [galactic][[freefree]] cache_dir
        - [extragalactic][[pointsource]] db_cache_dir
        """
        nonlocal lines
        out: List[str] = []
        in_parent = False
        in_sub = False
        parent_pat = re.compile(r"^\[" + re.escape(parent) + r"\]\s*$")
        section_pat = re.compile(r"^\[[^\]]+\]\s*$")
        sub_pat = re.compile(r"^\s*\[\[" + re.escape(subsection) + r"\]\]\s*$")
        any_sub_pat = re.compile(r"^\s*\[\[[^\]]+\]\]\s*$")
        key_pat = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*).*$")
        replaced = False

        for ln in lines:
            stripped = ln.strip()
            if parent_pat.match(stripped):
                in_parent = True
                in_sub = False
                out.append(ln)
                continue
            if in_parent and section_pat.match(stripped):
                in_parent = False
                in_sub = False
            if in_parent and sub_pat.match(stripped):
                in_sub = True
                out.append(ln)
                continue
            if in_parent and in_sub and any_sub_pat.match(stripped) and (not sub_pat.match(stripped)):
                in_sub = False

            if in_parent and in_sub:
                m = key_pat.match(ln)
                if m:
                    out.append(m.group(1) + value + "\n")
                    replaced = True
                    continue
            out.append(ln)

        if not replaced:
            raise RuntimeError(
                f"Failed to replace '{key}' in subsection [{parent}][[{subsection}]] in template: {template_path}"
            )
        lines = out

    def replace_anywhere(key: str, value: str, *, max_replacements: Optional[int] = None) -> None:
        nonlocal lines
        key_pat = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*).*$")
        out: List[str] = []
        n = 0
        for ln in lines:
            m = key_pat.match(ln)
            if m and (max_replacements is None or n < int(max_replacements)):
                out.append(m.group(1) + value + "\n")
                n += 1
            else:
                out.append(ln)
        if n == 0:
            raise RuntimeError(f"Failed to replace any '{key}=' in template: {template_path}")
        lines = out

    # paths
    replace_in_section("paths", "work_dir", str(work_dir))
    replace_in_section("paths", "log_dir", str(log_dir))
    replace_in_section("paths", "sky_dir", str(sky_dir))

    # frequency (only inside [frequency])
    replace_in_section("frequency", "start", f"{float(band.start_mhz):.6g}")
    replace_in_section("frequency", "stop", f"{float(band.stop_mhz):.6g}")
    replace_in_section("frequency", "step", f"{float(band.step_mhz):.6g}")

    # sky patch geometry (xsize/ysize/pixelsize appear only once in our template; replace globally).
    replace_anywhere("xsize", str(int(sky_xsize)), max_replacements=1)
    replace_anywhere("ysize", str(int(sky_ysize)), max_replacements=1)
    replace_anywhere("pixelsize", f"{float(sky_pixelsize_arcsec):.6g}", max_replacements=1)

    # parallel backend
    replace_in_section("parallel", "component_backend", str(component_backend))

    # postprocess toggle (inside [postprocess])
    replace_in_section("postprocess", "enabled", "true" if bool(postprocess_enabled) else "false")

    # IMPORTANT: In this Codex sandbox, we cannot write outside the workspace (writable roots).
    # e2esim's defaults put caches under `{paths/data_dir}/cache/...` which lives in the e2esim repo
    # and is not writable here. Override caches to workspace-local absolute paths.
    replace_in_subsection("galactic", "synchrotron", "cache_dir", str((base_dir / "cache" / "galactic" / "synchrotron").resolve()))
    replace_in_subsection("galactic", "freefree", "cache_dir", str((base_dir / "cache" / "galactic" / "freefree").resolve()))
    replace_in_subsection(
        "extragalactic",
        "pointsource",
        "db_cache_dir",
        str((work_root / "e2esim_runs" / run_name / "shared_cache" / "wilman").resolve()),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")

    meta = {
        "template": str(template_path),
        "band": {"start_mhz": band.start_mhz, "stop_mhz": band.stop_mhz, "step_mhz": band.step_mhz, "nchan": band.nchan},
        "paths": {"work_dir": str(work_dir), "log_dir": str(log_dir), "sky_dir": str(sky_dir)},
        "sky": {"xsize": sky_xsize, "ysize": sky_ysize, "pixelsize_arcsec": sky_pixelsize_arcsec},
        "parallel": {"component_backend": component_backend},
        "postprocess_enabled": bool(postprocess_enabled),
    }
    (output_path.parent / f"{output_path.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _parse_freq_from_filename(path: Path) -> float:
    m = re.search(r"_([0-9]+\.[0-9]+)MHz\.fits$", path.name)
    if not m:
        raise ValueError(f"Cannot parse frequency from filename: {path}")
    return float(m.group(1))


def _sorted_freq_files(dir_path: Path, prefix: str) -> List[Path]:
    files = sorted(dir_path.glob(f"{prefix}_*.fits"))
    if not files:
        raise FileNotFoundError(f"No files found for prefix '{prefix}' in {dir_path}")
    files = sorted(files, key=_parse_freq_from_filename)
    return files


def stack_component_maps_to_cubes(
    *,
    sky_dir: Path,
    out_dir: Path,
    dtype: str,
    clobber: bool,
) -> Dict[str, str]:
    """
    Stack e2esim per-frequency component FITS maps into 3D FITS cubes:
    - fg_cube = gsync + gfree + cluster + ptr
    - eor_cube = eor
    - all_cube = fg + eor
    """
    sky_dir = sky_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_np = np.float32 if str(dtype).lower() in {"f32", "float32"} else np.float64

    gsync_dir = sky_dir / "galactic" / "synchrotron"
    gfree_dir = sky_dir / "galactic" / "freefree"
    clu_dir = sky_dir / "extragalactic" / "clusters"
    ptr_dir = sky_dir / "extragalactic" / "pointsource"
    eor_dir = sky_dir / "eor" / "signal"

    gsync_files = _sorted_freq_files(gsync_dir, "gsync")
    gfree_files = _sorted_freq_files(gfree_dir, "gfree")
    clu_files = _sorted_freq_files(clu_dir, "cluster")
    ptr_files = _sorted_freq_files(ptr_dir, "ptr")
    eor_files = _sorted_freq_files(eor_dir, "eor")

    n = len(eor_files)
    for name, arr in [("gsync", gsync_files), ("gfree", gfree_files), ("cluster", clu_files), ("ptr", ptr_files)]:
        if len(arr) != n:
            raise RuntimeError(f"component {name} has {len(arr)} maps, but eor has {n}")

    freqs = [float(_parse_freq_from_filename(p)) for p in eor_files]
    if len(freqs) >= 2:
        step = float(np.median(np.diff(freqs)))
    else:
        step = float("nan")

    with fits.open(eor_files[0], memmap=True) as h:
        sample = np.asarray(h[0].data)
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D component maps; got {sample.shape} from {eor_files[0]}")
    ny, nx = int(sample.shape[0]), int(sample.shape[1])

    fg = np.zeros((n, ny, nx), dtype=dtype_np)
    eor = np.zeros((n, ny, nx), dtype=dtype_np)

    for i in range(n):
        def _read2(path: Path) -> np.ndarray:
            with fits.open(path, memmap=True) as h:
                return np.asarray(h[0].data, dtype=dtype_np)

        gsync = _read2(gsync_files[i])
        gfree = _read2(gfree_files[i])
        clu = _read2(clu_files[i])
        ptr = _read2(ptr_files[i])
        e = _read2(eor_files[i])
        fg[i] = gsync + gfree + clu + ptr
        eor[i] = e

    all_cube = fg + eor

    fg_path = out_dir / "fg_cube.fits"
    eor_path = out_dir / "eor_cube.fits"
    all_path = out_dir / "all_cube.fits"

    for p in [fg_path, eor_path, all_path]:
        if p.exists() and not clobber:
            raise FileExistsError(f"Output exists (use --clobber): {p}")

    def _write(path: Path, data: np.ndarray) -> None:
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["FREQ0"] = float(freqs[0]) if freqs else float("nan")
        hdu.header["DFREQ"] = float(step)
        hdu.header["NFREQ"] = int(n)
        hdu.writeto(path, overwrite=True)

    _write(fg_path, fg)
    _write(eor_path, eor)
    _write(all_path, all_cube)

    manifest = {
        "sky_dir": str(sky_dir),
        "out_dir": str(out_dir),
        "dtype": str(dtype),
        "shape": [int(n), int(ny), int(nx)],
        "freq_start_mhz": float(freqs[0]) if freqs else None,
        "freq_stop_mhz": float(freqs[-1]) if freqs else None,
        "freq_step_mhz_median": float(step),
        "components": {
            "gsync_dir": str(gsync_dir),
            "gfree_dir": str(gfree_dir),
            "cluster_dir": str(clu_dir),
            "ptr_dir": str(ptr_dir),
            "eor_dir": str(eor_dir),
        },
        "outputs": {"fg_cube": str(fg_path), "eor_cube": str(eor_path), "all_cube": str(all_path)},
    }
    (out_dir / "stack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in manifest["outputs"].items()}  # type: ignore[index]


def run_e2esim_sky(*, config_path: Path, python_bin: str) -> None:
    # Important: this project has a top-level symlink named "e2esim" which points to the
    # e2esim *repo root* (not the python package directory). If we run `python -m e2esim`
    # from the project root, that symlink shadows the real package and `-m e2esim` fails.
    #
    # Workaround: run with cwd set to the e2esim repo root so that the python package
    # directory `e2esim/` is importable as a normal package.
    cfg_txt = config_path.read_text(encoding="utf-8", errors="ignore")
    data_dir: Optional[str] = None
    in_paths = False
    for raw in cfg_txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[paths]":
            in_paths = True
            continue
        if in_paths and line.startswith("[") and line.endswith("]"):
            in_paths = False
        if in_paths and line.lower().startswith("data_dir"):
            _, rhs = line.split("=", 1)
            data_dir = rhs.strip()
            break
    if not data_dir:
        raise RuntimeError(f"Cannot locate [paths]/data_dir in config: {config_path}")
    repo_root = str(Path(data_dir).resolve().parent)

    cmd_init = [python_bin, "-m", "e2esim", "init", "--config", str(config_path)]
    cmd_sky = [python_bin, "-m", "e2esim", "sky", "--config", str(config_path)]
    subprocess.check_call(cmd_init, cwd=repo_root)
    subprocess.check_call(cmd_sky, cwd=repo_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-band cubes via e2esim (sky only) and stack to 3D FITS.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-configs", help="Generate per-band e2esim configs from a template.")
    g.add_argument("--work-root", type=Path, default=Path.cwd())
    g.add_argument("--template", type=Path, default=Path.cwd() / "e2esim_cube2.conf")
    g.add_argument("--run-name", type=str, default="multiband_1024")
    g.add_argument("--starts-mhz", type=str, default="60,80,120,140,160,180")
    g.add_argument("--nchan", type=int, default=151)
    g.add_argument("--step-mhz", type=float, default=0.1)
    g.add_argument("--sky-xsize", type=int, default=1024)
    g.add_argument("--sky-ysize", type=int, default=1024)
    g.add_argument("--sky-pixelsize-arcsec", type=float, default=40.0)
    g.add_argument("--component-backend", type=str, default="threads", choices=["threads", "processes"])
    g.add_argument("--postprocess", action="store_true", help="Keep e2esim postprocess enabled (default disabled).")

    r = sub.add_parser("run-sky", help="Run e2esim sky for a list of generated configs.")
    r.add_argument("--python-bin", type=str, default="python3")
    r.add_argument("--configs-dir", type=Path, required=True)
    r.add_argument("--starts-mhz", type=str, default="60,80,120,140,160,180")

    s = sub.add_parser("stack", help="Stack one band (or all bands) sky maps into fg/eor/all cubes.")
    s.add_argument("--work-root", type=Path, default=Path.cwd())
    s.add_argument("--run-name", type=str, default="multiband_1024")
    s.add_argument("--out-data-subdir", type=str, default="multiband_1024_20260213")
    s.add_argument("--starts-mhz", type=str, default="60,80,120,140,160,180")
    s.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    s.add_argument("--clobber", action="store_true")

    a = sub.add_parser("all", help="Generate configs, run e2esim sky, then stack cubes.")
    a.add_argument("--work-root", type=Path, default=Path.cwd())
    a.add_argument("--template", type=Path, default=Path.cwd() / "e2esim_cube2.conf")
    a.add_argument("--run-name", type=str, default="multiband_1024")
    a.add_argument("--out-data-subdir", type=str, default="multiband_1024_20260213")
    a.add_argument("--starts-mhz", type=str, default="60,80,120,140,160,180")
    a.add_argument("--nchan", type=int, default=151)
    a.add_argument("--step-mhz", type=float, default=0.1)
    a.add_argument("--sky-xsize", type=int, default=1024)
    a.add_argument("--sky-ysize", type=int, default=1024)
    a.add_argument("--sky-pixelsize-arcsec", type=float, default=40.0)
    a.add_argument("--component-backend", type=str, default="threads", choices=["threads", "processes"])
    a.add_argument("--postprocess", action="store_true")
    a.add_argument("--python-bin", type=str, default="python3")
    a.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    a.add_argument("--clobber", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.cmd == "gen-configs":
        work_root = args.work_root.resolve()
        configs_dir = work_root / "e2esim_runs" / args.run_name / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        starts = parse_float_csv(args.starts_mhz)
        for start in starts:
            band = BandSpec(start_mhz=float(start), nchan=int(args.nchan), step_mhz=float(args.step_mhz))
            band_label = _fmt_band_label(band.start_mhz)
            out_path = configs_dir / f"{band_label}.conf"
            generate_config_from_template(
                template_path=args.template.resolve(),
                output_path=out_path,
                work_root=work_root,
                run_name=str(args.run_name),
                band=band,
                sky_xsize=int(args.sky_xsize),
                sky_ysize=int(args.sky_ysize),
                sky_pixelsize_arcsec=float(args.sky_pixelsize_arcsec),
                component_backend=str(args.component_backend),
                postprocess_enabled=bool(args.postprocess),
            )
        print(str(configs_dir))
        return 0

    if args.cmd == "run-sky":
        starts = parse_float_csv(args.starts_mhz)
        for start in starts:
            band_label = _fmt_band_label(float(start))
            conf = args.configs_dir / f"{band_label}.conf"
            run_e2esim_sky(config_path=conf.resolve(), python_bin=str(args.python_bin))
        return 0

    if args.cmd == "stack":
        work_root = args.work_root.resolve()
        starts = parse_float_csv(args.starts_mhz)
        for start in starts:
            band_label = _fmt_band_label(float(start))
            sky_dir = work_root / "e2esim_runs" / args.run_name / band_label / "sky_model"
            out_dir = work_root / "data" / args.out_data_subdir / band_label
            stack_component_maps_to_cubes(
                sky_dir=sky_dir,
                out_dir=out_dir,
                dtype=str(args.dtype),
                clobber=bool(args.clobber),
            )
        return 0

    if args.cmd == "all":
        work_root = args.work_root.resolve()
        configs_dir = work_root / "e2esim_runs" / args.run_name / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        starts = parse_float_csv(args.starts_mhz)
        for start in starts:
            band = BandSpec(start_mhz=float(start), nchan=int(args.nchan), step_mhz=float(args.step_mhz))
            band_label = _fmt_band_label(band.start_mhz)
            conf = configs_dir / f"{band_label}.conf"
            generate_config_from_template(
                template_path=args.template.resolve(),
                output_path=conf,
                work_root=work_root,
                run_name=str(args.run_name),
                band=band,
                sky_xsize=int(args.sky_xsize),
                sky_ysize=int(args.sky_ysize),
                sky_pixelsize_arcsec=float(args.sky_pixelsize_arcsec),
                component_backend=str(args.component_backend),
                postprocess_enabled=bool(args.postprocess),
            )
            run_e2esim_sky(config_path=conf.resolve(), python_bin=str(args.python_bin))

            sky_dir = work_root / "e2esim_runs" / args.run_name / band_label / "sky_model"
            out_dir = work_root / "data" / args.out_data_subdir / band_label
            stack_component_maps_to_cubes(
                sky_dir=sky_dir,
                out_dir=out_dir,
                dtype=str(args.dtype),
                clobber=bool(args.clobber),
            )
        return 0

    raise AssertionError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
