#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_noise_systematics_20260817}"
BASE_RUN="${BASE_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
PARENT_RUN="${PARENT_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_flag_stress_20260817}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"

LABEL="local_04_116p3_119p4mhz"
ANALYSIS_CONFIG="${BASE_RUN}/configs/${LABEL}_analysis.json"
INPUT_CONFIG="${BASE_RUN}/configs/${LABEL}_input.json"
SKY_CACHE="${PARENT_RUN}/eor_intrinsic_sky_128freq.npz"
BEAM_CACHE_ROOT="${PARENT_RUN}/beam_cache_shared"
EVALUATOR="${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py"
COMBINER="${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py"
COARSE_EVALUATOR="${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py"

PARTITION_COUNT="${PARTITION_COUNT:-4}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-70000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-300}"

# Indices refer to the 64-channel input band. The patterns stress sparse and
# contiguous gaps while retaining the same predeclared 32-channel analysis band.
PATTERNS=(
  "random5|5,17,41"
  "random10|5,12,17,29,41,54"
  "cluster6|29,30,31,32"
)

for required in \
  "${BANK_DIR}/COMPLETE" \
  "${PARENT_RUN}/COMPLETE" \
  "${ANALYSIS_CONFIG}" \
  "${INPUT_CONFIG}" \
  "${SKY_CACHE}" \
  "${EVALUATOR}" \
  "${COMBINER}" \
  "${COARSE_EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing flag-stress input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}/logs"

select_gpus() {
  nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, \
      -v minimum_free="${GPU_MIN_FREE_MIB}" \
      -v util_limit="${GPU_UTIL_LIMIT_PERCENT}" \
      -v maximum_count="${MAX_WORKERS}" \
      '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (($2 + 0) >= minimum_free && ($3 + 0) <= util_limit) {
          print $1;
          count += 1;
          if (count >= maximum_count) exit;
        }
      }'
}

run_partition() {
  local pattern="$1"
  local indices="$2"
  local partition="$3"
  local gpu="$4"
  local root="${OUT_DIR}/${pattern}"
  local evaluate_dir="${root}/part_${partition}/evaluate"
  mkdir -p "${evaluate_dir}"
  if [[ -s "${evaluate_dir}/result.npz" &&
        -s "${evaluate_dir}/result.json" ]]; then
    return
  fi

  local flag_args=()
  local index
  IFS="," read -r -a flag_indices <<<"${indices}"
  for index in "${flag_indices[@]}"; do
    flag_args+=(--flagged-input-frequency-index "${index}")
  done

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
    --config "${ANALYSIS_CONFIG}" \
    --frequency-config "${INPUT_CONFIG}" \
    --bank-dir "${BANK_DIR}" \
    --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
    --sky-cache "${SKY_CACHE}" \
    --out-dir "${evaluate_dir}" \
    --device cuda:0 \
    --rows-per-kperp-bin "${ROWS_PER_BIN}" \
    --row-scope all \
    --row-seed 20260725 \
    --row-partition-index "${partition}" \
    --row-partition-count "${PARTITION_COUNT}" \
    --calibration-repeats 1 \
    --validation-repeats 1 \
    --mixture-repeats 16 \
    --source-scope all_in_range_with_nyquist \
    --probe-batch-size 8 \
    --probe-seed 51021 \
    --operator-dtype complex64 \
    --operator-storage cpu_streamed \
    --primary-beam bank \
    --aperture-row-beam-cache-pattern \
      "${BEAM_CACHE_ROOT}/freq_{freq:.2f}" \
    --source-chunk 32768 \
    --row-chunk 32 \
    --foreground-filter dpss_hard \
    --filter-bandwidth-scope full_band \
    --spectral-taper hann \
    --minimum-window-self-fraction 0 \
    --minimum-relative-sensitivity 0 \
    --response-rcond 1e-4 \
    "${flag_args[@]}" \
    >"${OUT_DIR}/logs/${pattern}_part_${partition}.log" 2>&1
}

tasks=()
for row in "${PATTERNS[@]}"; do
  IFS="|" read -r pattern indices <<<"${row}"
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    tasks+=("${pattern}|${indices}|${partition}")
  done
done

next=0
while ((next < ${#tasks[@]})); do
  mapfile -t available_gpus < <(select_gpus)
  if [[ "${#available_gpus[@]}" -eq 0 ]]; then
    echo "no GPU currently satisfies the live flag-stress limits" >&2
    sleep "${GPU_RECHECK_SECONDS}"
    continue
  fi
  pids=()
  for offset in "${!available_gpus[@]}"; do
    task_index=$((next + offset))
    if ((task_index >= ${#tasks[@]})); then
      break
    fi
    IFS="|" read -r pattern indices partition <<<"${tasks[task_index]}"
    run_partition \
      "${pattern}" "${indices}" "${partition}" \
      "${available_gpus[offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one flag-stress evaluator failed" >&2
    exit "${status}"
  fi
  next=$((next + ${#pids[@]}))
done

for row in "${PATTERNS[@]}"; do
  IFS="|" read -r pattern indices <<<"${row}"
  root="${OUT_DIR}/${pattern}"
  combine_args=()
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    combine_args+=(--input-dir "${root}/part_${partition}/evaluate")
  done
  "${PYTHON}" "${COMBINER}" \
    "${combine_args[@]}" \
    --out-dir "${root}/combined" \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/${pattern}_combine.log" 2>&1
  "${PYTHON}" "${COARSE_EVALUATOR}" \
    --combined-npz "${root}/combined/result.npz" \
    --config "${ANALYSIS_CONFIG}" \
    --out-dir "${root}/coarse" \
    --minimum-kperp-index 4 \
    --minimum-relative-response 0.1 \
    --minimum-window-fraction 0.95 \
    --profile fine_response \
    --profile pair_kperp_response \
    --profile quad_kperp_response \
    --profile quad_kperp_kpar2_response \
    >"${OUT_DIR}/logs/${pattern}_coarse.log" 2>&1
done

touch "${OUT_DIR}/COMPLETE"
