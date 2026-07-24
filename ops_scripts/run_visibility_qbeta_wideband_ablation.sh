#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_64freq_grid512_20260725}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_64freq_20260714}"
CASE_SET="${CASE_SET:-screen}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32central_isobeam_patch.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64wide_isobeam_patch.json}"
SKY_CACHE="${SKY_CACHE:-${PROJECT_ROOT}/runs/visibility_qbeta_wideband_common_20260725/eor_intrinsic_sky_64freq.npz}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"

if [[ "${CASE_SET}" == "screen" ]]; then
  RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_wideband_screen_20260725}"
  PARTITION_COUNT="${PARTITION_COUNT:-4}"
  ROWS_PER_BIN="${ROWS_PER_BIN:-24}"
  CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-1}"
  VALIDATION_REPEATS="${VALIDATION_REPEATS:-1}"
  MIXTURE_REPEATS="${MIXTURE_REPEATS:-4}"
elif [[ "${CASE_SET}" == "promotion" ]]; then
  RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_wideband_promotion_20260725}"
  PARTITION_COUNT="${PARTITION_COUNT:-8}"
  ROWS_PER_BIN="${ROWS_PER_BIN:-48}"
  CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-2}"
  VALIDATION_REPEATS="${VALIDATION_REPEATS:-1}"
  MIXTURE_REPEATS="${MIXTURE_REPEATS:-16}"
else
  echo "CASE_SET must be screen or promotion." >&2
  exit 2
fi

CASE_NAMES=(
  dpss_narrow32_e12_hann
  dpss_wide64_to32_e12_hann
)
CASE_SCOPES=(
  analysis_subband
  full_band
)

if [[ ! -s "${BANK_DIR}/COMPLETE" ]]; then
  echo "64-frequency visibility bank is not complete: ${BANK_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}" "$(dirname "${SKY_CACHE}")"

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
  local case_index="$1"
  local partition="$2"
  local gpu="$3"
  local name="${CASE_NAMES[case_index]}"
  local case_dir="${RUN_ROOT}/${name}"
  local evaluate_dir="${case_dir}/part_${partition}/evaluate"
  mkdir -p "${evaluate_dir}" "${case_dir}/logs"
  if [[ -s "${evaluate_dir}/result.npz" && -s "${evaluate_dir}/result.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
    --config "${CONFIG}" \
    --frequency-config "${FREQUENCY_CONFIG}" \
    --bank-dir "${BANK_DIR}" \
    --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
    --sky-cache "${SKY_CACHE}" \
    --out-dir "${evaluate_dir}" \
    --device cuda:0 \
    --rows-per-kperp-bin "${ROWS_PER_BIN}" \
    --row-scope reporting_kperp \
    --row-seed 20260725 \
    --row-partition-index "${partition}" \
    --row-partition-count "${PARTITION_COUNT}" \
    --calibration-repeats "${CALIBRATION_REPEATS}" \
    --validation-repeats "${VALIDATION_REPEATS}" \
    --mixture-repeats "${MIXTURE_REPEATS}" \
    --source-scope all_in_range_with_nyquist \
    --foreground-filter dpss_hard \
    --filter-bandwidth-scope "${CASE_SCOPES[case_index]}" \
    --dpss-eigenvalue-threshold 1e-12 \
    --spectral-taper hann \
    --probe-batch-size 8 \
    --operator-dtype complex64 \
    --source-chunk 8192 \
    --row-chunk 32 \
    --response-rcond 1e-4 \
    >"${case_dir}/logs/part_${partition}.log" 2>&1
}

run_case() {
  local case_index="$1"
  local gpu="$2"
  local name="${CASE_NAMES[case_index]}"
  local case_dir="${RUN_ROOT}/${name}"
  local combine_args=()
  mkdir -p "${case_dir}/logs"
  if [[ -f "${case_dir}/COMPLETE" ]]; then
    echo "Skipping completed case: ${name}"
    return
  fi
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    run_partition "${case_index}" "${partition}" "${gpu}"
  done
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    combine_args+=(
      --input-dir "${case_dir}/part_${partition}/evaluate"
    )
  done
  "${PYTHON}" "${COMBINER}" \
    "${combine_args[@]}" \
    --out-dir "${case_dir}/combined" \
    --response-rcond 1e-4 \
    >"${case_dir}/logs/combine.log" 2>&1
  date -Is >"${case_dir}/COMPLETE"
  echo "Completed wideband case: ${name}"
}

# Avoid racing two writers while the shared 64-frequency sky cache is created.
if [[ ! -s "${SKY_CACHE}" ]]; then
  run_partition 0 0 "${AVAILABLE_GPUS[0]}"
fi

if [[ "${#AVAILABLE_GPUS[@]}" -eq 1 ]]; then
  for case_index in "${!CASE_NAMES[@]}"; do
    run_case "${case_index}" "${AVAILABLE_GPUS[0]}"
  done
else
  pids=()
  for case_index in "${!CASE_NAMES[@]}"; do
    gpu="${AVAILABLE_GPUS[case_index]}"
    run_case "${case_index}" "${gpu}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one wideband case failed." >&2
    exit "${status}"
  fi
fi

date -Is >"${RUN_ROOT}/COMPLETE"
echo "Visibility Q_beta wideband ablation complete: ${RUN_ROOT}"
