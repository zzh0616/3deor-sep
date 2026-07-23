#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_32freq_grid512_20260723}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_32freq_20260617}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_32freq_20260724}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
SCRIPT="${SCRIPT:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32high_isobeam_patch.json}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/cache"

if [[ -n "${GPU:-}" ]]; then
  SELECTED_GPU="${GPU}"
else
  SELECTED_GPU="$(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F, \
        -v memory_limit="${GPU_MEMORY_LIMIT_MIB}" \
        -v util_limit="${GPU_UTIL_LIMIT_PERCENT}" \
        '{
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          gsub(/ /, "", $3);
          if (($2 + 0) < memory_limit && ($3 + 0) < util_limit) {
            print $1;
            exit;
          }
        }'
  )"
fi
if [[ -z "${SELECTED_GPU}" ]]; then
  echo "No GPU is currently below the launch limits." >&2
  exit 1
fi

printf 'selected GPU at launch: %s\n' "${SELECTED_GPU}" |
  tee "${OUT_DIR}/logs/gpu_selection.log"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  >>"${OUT_DIR}/logs/gpu_selection.log"

CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" "${PYTHON}" "${SCRIPT}" \
  --config "${CONFIG}" \
  --bank-dir "${BANK_DIR}" \
  --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
  --sky-cache "${OUT_DIR}/cache/eor_intrinsic_sky.npz" \
  --out-dir "${OUT_DIR}/evaluate" \
  --device cuda:0 \
  --rows-per-kperp-bin 8 \
  --calibration-repeats 2 \
  --validation-repeats 2 \
  --probe-batch-size 20 \
  --operator-dtype complex64 \
  --source-chunk 4096 \
  --row-chunk 32 \
  --response-rcond 1e-4 \
  >"${OUT_DIR}/logs/evaluate.log" 2>&1

date -Is >"${OUT_DIR}/COMPLETE"
echo "Visibility Q_beta pilot complete: ${OUT_DIR}"
