#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_32freq_grid512_20260723}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_32freq_20260617}"
BASE_RUN="${BASE_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_32freq_20260724}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_filter_ablation_20260724}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32high_isobeam_patch.json}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
CASE_SET="${CASE_SET:-screen}"
MIXTURE_REPEATS="${MIXTURE_REPEATS:-2}"

if [[ "${CASE_SET}" == "screen" ]]; then
  CASE_NAMES=(
    none_hann
    dpss_hard_e6_hann
    dpss_soft_e12_r1e4_hann
    dpss_soft_e12_r1e8_hann
    dpss_hard_e12_blackman_harris
    chebyshev_d3_hann
    chebyshev_rankmatched_e12_hann
  )
  CASE_FILTERS=(
    none
    dpss_hard
    dpss_soft
    dpss_soft
    dpss_hard
    chebyshev
    chebyshev_rank_matched
  )
  CASE_STRENGTHS=(0 0 1e4 1e8 0 0 0)
  CASE_DEGREES=(3 3 3 3 3 3 3)
  CASE_THRESHOLDS=(1e-12 1e-6 1e-12 1e-12 1e-12 1e-12 1e-12)
  CASE_TAPERS=(hann hann hann hann blackman_harris hann hann)
elif [[ "${CASE_SET}" == "promotion" ]]; then
  CASE_NAMES=(
    dpss_hard_e12_hann
    dpss_hard_e10_hann
    dpss_hard_e8_hann
    dpss_hard_e6_hann
    dpss_soft_e12_r1e8_hann
    chebyshev_rankmatched_e12_hann
  )
  CASE_FILTERS=(
    dpss_hard
    dpss_hard
    dpss_hard
    dpss_hard
    dpss_soft
    chebyshev_rank_matched
  )
  CASE_STRENGTHS=(0 0 0 0 1e8 0)
  CASE_DEGREES=(3 3 3 3 3 3)
  CASE_THRESHOLDS=(1e-12 1e-10 1e-8 1e-6 1e-12 1e-12)
  CASE_TAPERS=(hann hann hann hann hann hann)
else
  echo "CASE_SET must be screen or promotion." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"
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

run_case() {
  local case_index="$1"
  local gpu="$2"
  local name="${CASE_NAMES[case_index]}"
  local case_dir="${RUN_ROOT}/${name}"
  mkdir -p "${case_dir}/logs"
  if [[ -f "${case_dir}/COMPLETE" ]]; then
    echo "Skipping completed case: ${name}"
    return
  fi
  for partition in 0 1 2 3; do
    local evaluate_dir="${case_dir}/part_${partition}/evaluate"
    mkdir -p "${evaluate_dir}"
    if [[ -s "${evaluate_dir}/result.npz" && -s "${evaluate_dir}/result.json" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
      --config "${CONFIG}" \
      --bank-dir "${BANK_DIR}" \
      --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
      --sky-cache "${BASE_RUN}/cache/eor_intrinsic_sky.npz" \
      --out-dir "${evaluate_dir}" \
      --device cuda:0 \
      --rows-per-kperp-bin 96 \
      --row-scope reporting_kperp \
      --row-seed 20260724 \
      --row-partition-index "${partition}" \
      --row-partition-count 4 \
      --calibration-repeats 2 \
      --validation-repeats 1 \
      --mixture-repeats "${MIXTURE_REPEATS}" \
      --source-scope all_in_range_with_nyquist \
      --foreground-filter "${CASE_FILTERS[case_index]}" \
      --suppression-strength "${CASE_STRENGTHS[case_index]}" \
      --polynomial-degree "${CASE_DEGREES[case_index]}" \
      --dpss-eigenvalue-threshold "${CASE_THRESHOLDS[case_index]}" \
      --spectral-taper "${CASE_TAPERS[case_index]}" \
      --probe-batch-size 8 \
      --operator-dtype complex64 \
      --source-chunk 8192 \
      --row-chunk 32 \
      --response-rcond 1e-4 \
      >"${case_dir}/logs/part_${partition}.log" 2>&1
  done
  "${PYTHON}" "${COMBINER}" \
    --input-dir "${case_dir}/part_0/evaluate" \
    --input-dir "${case_dir}/part_1/evaluate" \
    --input-dir "${case_dir}/part_2/evaluate" \
    --input-dir "${case_dir}/part_3/evaluate" \
    --out-dir "${case_dir}/combined" \
    --response-rcond 1e-4 \
    >"${case_dir}/logs/combine.log" 2>&1
  date -Is >"${case_dir}/COMPLETE"
  echo "Completed filter case: ${name}"
}

for ((first = 0; first < ${#CASE_NAMES[@]}; first += ${#AVAILABLE_GPUS[@]})); do
  pids=()
  for gpu_offset in "${!AVAILABLE_GPUS[@]}"; do
    case_index=$((first + gpu_offset))
    if ((case_index >= ${#CASE_NAMES[@]})); then
      break
    fi
    run_case "${case_index}" "${AVAILABLE_GPUS[gpu_offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one filter-ablation case failed." >&2
    exit "${status}"
  fi
done

date -Is >"${RUN_ROOT}/COMPLETE"
echo "Visibility Q_beta filter ablation complete: ${RUN_ROOT}"
