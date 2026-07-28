#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
LOCAL_ROOT="${LOCAL_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_outer_field_20260728}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
OUTER_SKY="${OUTER_SKY:-${OUT_ROOT}/outer_fg_local02_full.npz}"
LABEL="${LABEL:-local_02_113p1_116p2mhz}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
MAX_WORKERS="${MAX_WORKERS:-4}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"

ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-${LOCAL_ROOT}/configs/${LABEL}_analysis.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${LOCAL_ROOT}/configs/${LABEL}_input.json}"
COMBINED_DIR="${COMBINED_DIR:-${LOCAL_ROOT}/${LABEL}/combined}"
COARSE_PRODUCTS="${COARSE_PRODUCTS:-${LOCAL_ROOT}/${LABEL}/coarse/products.npz}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}/beam_cache_${LABEL}}"
EVALUATION_DIR="${EVALUATION_DIR:-${OUT_ROOT}/evaluation_${LABEL}}"
BUILD_CACHE="${BUILD_CACHE:-${CODE_ROOT}/ops_scripts/build_oskar_aperture_row_beam_cache.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_outer_field.py}"
OSKAR_PREFIX="${OSKAR_PREFIX:-${PROJECT_ROOT}/../local/radio-202605-oskar212-cuda-casa380}"
COMPILER_LIBRARY_DIR="${COMPILER_LIBRARY_DIR:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380/lib}"
HELPER_BINARY="${HELPER_BINARY:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/tools/evaluate_oskar_aperture_row_beam_factors}"

for required in \
  "${OUTER_SKY}" \
  "${ANALYSIS_CONFIG}" \
  "${FREQUENCY_CONFIG}" \
  "${COMBINED_DIR}/result.npz" \
  "${COARSE_PRODUCTS}" \
  "${BUILD_CACHE}" \
  "${EVALUATOR}" \
  "${HELPER_BINARY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing outer-field input: ${required}" >&2
    exit 1
  fi
done
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  required="${LOCAL_ROOT}/${LABEL}/part_${partition}/evaluate/result.npz"
  if [[ ! -s "${required}" ]]; then
    echo "missing outer-field partition result: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${CACHE_ROOT}" "${EVALUATION_DIR}"
mapfile -t FREQUENCIES_MHZ < <(
  "${PYTHON}" -c \
    'import json,sys
for value in json.load(open(sys.argv[1]))["frequencies_mhz"]:
    print(f"{float(value):.2f}")' \
    "${FREQUENCY_CONFIG}"
)
if [[ "${#FREQUENCIES_MHZ[@]}" -lt 2 ]]; then
  echo "frequency config contains too few channels" >&2
  exit 2
fi

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

build_frequency_cache() {
  local frequency="$1"
  local gpu="$2"
  local output="${CACHE_ROOT}/freq_${frequency}"
  if [[ -s "${output}/metadata.json" &&
        -s "${output}/coherency.complex64.bin" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${BUILD_CACHE}" \
    --bank-shard "${BANK_DIR}/shards/freq_${frequency}.npz" \
    --sky-cache "${OUTER_SKY}" \
    --oskar-config "${BANK_DIR}/configs/sim_eor_${frequency}.ini" \
    --oskar-prefix "${OSKAR_PREFIX}" \
    --out-dir "${output}" \
    --selected-row-result "${COMBINED_DIR}/result.npz" \
    --source-chunk 32768 \
    --compiler-library-dir "${COMPILER_LIBRARY_DIR}" \
    --helper-binary "${HELPER_BINARY}" \
    >"${OUT_ROOT}/logs/beam_${frequency}.log" 2>&1
}

next=0
while ((next < ${#FREQUENCIES_MHZ[@]})); do
  mapfile -t AVAILABLE_GPUS < <(select_gpus)
  if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
    echo "no GPU currently satisfies the outer-field cache limits" >&2
    sleep "${GPU_RECHECK_SECONDS}"
    continue
  fi
  pids=()
  for offset in "${!AVAILABLE_GPUS[@]}"; do
    index=$((next + offset))
    if ((index >= ${#FREQUENCIES_MHZ[@]})); then
      break
    fi
    build_frequency_cache \
      "${FREQUENCIES_MHZ[index]}" "${AVAILABLE_GPUS[offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one outer-field PB cache worker failed" >&2
    exit "${status}"
  fi
  next=$((next + ${#pids[@]}))
done

while true; do
  mapfile -t AVAILABLE_GPUS < <(select_gpus)
  if [[ "${#AVAILABLE_GPUS[@]}" -gt 0 ]]; then
    break
  fi
  sleep "${GPU_RECHECK_SECONDS}"
done

partition_args=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  partition_args+=(
    --partition-result-dir
    "${LOCAL_ROOT}/${LABEL}/part_${partition}/evaluate"
  )
done
CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" "${PYTHON}" "${EVALUATOR}" \
  --config "${ANALYSIS_CONFIG}" \
  --frequency-config "${FREQUENCY_CONFIG}" \
  --bank-dir "${BANK_DIR}" \
  --outer-sky "${OUTER_SKY}" \
  --combined-result-dir "${COMBINED_DIR}" \
  --coarse-products "${COARSE_PRODUCTS}" \
  --profile quad_kperp_response \
  "${partition_args[@]}" \
  --aperture-row-beam-cache-pattern "${CACHE_ROOT}/freq_{freq:.2f}" \
  --out-dir "${EVALUATION_DIR}" \
  --device cuda:0 \
  --source-chunk 8192 \
  >"${OUT_ROOT}/logs/evaluation_${LABEL}.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_ROOT}/COMPLETE"
echo "outer-field Q_beta evaluation complete: ${OUT_ROOT}"
