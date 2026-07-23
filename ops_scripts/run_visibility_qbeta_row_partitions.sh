#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_32freq_grid512_20260723}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_32freq_20260617}"
BASE_RUN="${BASE_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_32freq_20260724}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_rowpart6_20260724}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32high_isobeam_patch.json}"
PARTITION_COUNT="${PARTITION_COUNT:-6}"
PARTITION_FIRST="${PARTITION_FIRST:-0}"
ROWS_PER_BIN="${ROWS_PER_BIN:-32}"
ROW_SCOPE="${ROW_SCOPE:-all}"
CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-8}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-8}"
MIXTURE_REPEATS="${MIXTURE_REPEATS:-4}"
SOURCE_SCOPE="${SOURCE_SCOPE:-reporting}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"

mkdir -p "${OUT_DIR}/logs"
mapfile -t AVAILABLE_GPUS < <(
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
        if (($2 + 0) < memory_limit && ($3 + 0) < util_limit) print $1;
      }'
)
if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
  echo "No GPU is currently below the launch limits." >&2
  exit 1
fi

run_partition() {
  local partition="$1"
  local gpu="$2"
  local partition_dir="${OUT_DIR}/part_${partition}"
  mkdir -p "${partition_dir}/evaluate"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
    --config "${CONFIG}" \
    --bank-dir "${BANK_DIR}" \
    --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
    --sky-cache "${BASE_RUN}/cache/eor_intrinsic_sky.npz" \
    --out-dir "${partition_dir}/evaluate" \
    --device cuda:0 \
    --rows-per-kperp-bin "${ROWS_PER_BIN}" \
    --row-scope "${ROW_SCOPE}" \
    --row-seed 20260724 \
    --row-partition-index "${partition}" \
    --row-partition-count "${PARTITION_COUNT}" \
    --calibration-repeats "${CALIBRATION_REPEATS}" \
    --validation-repeats "${VALIDATION_REPEATS}" \
    --mixture-repeats "${MIXTURE_REPEATS}" \
    --source-scope "${SOURCE_SCOPE}" \
    --probe-batch-size 8 \
    --operator-dtype complex64 \
    --source-chunk 8192 \
    --row-chunk 32 \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/part_${partition}.log" 2>&1
}

for ((
  first = PARTITION_FIRST;
  first < PARTITION_COUNT;
  first += ${#AVAILABLE_GPUS[@]}
)); do
  pids=()
  for gpu_offset in "${!AVAILABLE_GPUS[@]}"; do
    partition=$((first + gpu_offset))
    if ((partition >= PARTITION_COUNT)); then
      break
    fi
    run_partition "${partition}" "${AVAILABLE_GPUS[gpu_offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one Q_beta row partition failed." >&2
    exit "${status}"
  fi
done

combine_args=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  combine_args+=(--input-dir "${OUT_DIR}/part_${partition}/evaluate")
done
"${PYTHON}" "${COMBINER}" \
  "${combine_args[@]}" \
  --out-dir "${OUT_DIR}/combined" \
  --response-rcond 1e-4 \
  >"${OUT_DIR}/logs/combine.log" 2>&1

date -Is >"${OUT_DIR}/COMPLETE"
echo "Visibility Q_beta row partitions complete: ${OUT_DIR}"
