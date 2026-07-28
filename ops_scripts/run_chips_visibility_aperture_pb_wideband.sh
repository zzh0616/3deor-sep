#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/3dnet}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380/bin/python}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64wide_isobeam_patch.json}"
RUNNER="${RUNNER:-${CODE_ROOT}/ops_scripts/run_chips_visibility_aperture_pb_pilot.sh}"

FREQUENCIES_MHZ="$(
  "${PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(f"{x:.2f}" for x in d["frequencies_mhz"]))' \
    "${FREQUENCY_CONFIG}"
)"
REFERENCE_FREQUENCY_MHZ="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["reference_frequency_mhz"])' \
    "${FREQUENCY_CONFIG}"
)"

export PROJECT_ROOT
export CODE_ROOT
export PYTHON
export SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_64freq_20260725}"
export OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_64freq_20260725}"
export FREQUENCIES_MHZ
export REFERENCE_FREQUENCY_MHZ
export GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-300}"

exec bash "${RUNNER}"
