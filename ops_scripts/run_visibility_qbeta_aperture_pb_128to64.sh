#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen${PARTITION_COUNT}_20260726}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64central_111p5_117p8_isobeam_patch.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_128wide_isobeam_patch.json}"
OSKAR_PREFIX="${OSKAR_PREFIX:-${PROJECT_ROOT}/../local/radio-202605-oskar212-cuda-casa380}"
OSKAR_RUNTIME="${OSKAR_RUNTIME:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
SELECTION_SCRIPT="${SELECTION_SCRIPT:-${CODE_ROOT}/ops_scripts/prepare_visibility_qbeta_row_selection.py}"
SKY_BUILDER="${SKY_BUILDER:-${CODE_ROOT}/ops_scripts/build_visibility_qbeta_sky_cache.py}"
BEAM_BUILDER="${BEAM_BUILDER:-${CODE_ROOT}/ops_scripts/build_oskar_aperture_row_beam_cache.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
COARSE_EVALUATOR="${COARSE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py}"
SKY_CACHE="${SKY_CACHE:-${OUT_DIR}/eor_intrinsic_sky_128freq.npz}"
ROW_SEED="${ROW_SEED:-20260725}"
MAXIMUM_KPERP_INDEX_EXCLUSIVE="${MAXIMUM_KPERP_INDEX_EXCLUSIVE:-}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-50}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-300}"

"${PYTHON}" -c \
  'import json,sys,torch; reporting=json.load(open(sys.argv[1]))["reporting_masks"]; assert "high_kpar_fraction" in reporting; assert len(reporting["mid_kperp_fraction_range"]) == 2; assert torch.cuda.is_available()' \
  "${CONFIG}" || {
    echo "the Q_beta runtime requires CUDA PyTorch and a complete reporting_masks config" >&2
    exit 2
  }

row_kperp_limit_args=()
if [[ -n "${MAXIMUM_KPERP_INDEX_EXCLUSIVE}" ]]; then
  if ! [[ "${MAXIMUM_KPERP_INDEX_EXCLUSIVE}" =~ ^[0-9]+$ ]]; then
    echo "MAXIMUM_KPERP_INDEX_EXCLUSIVE must be a positive integer" >&2
    exit 2
  fi
  row_kperp_limit_args=(
    --maximum-kperp-index-exclusive
    "${MAXIMUM_KPERP_INDEX_EXCLUSIVE}"
  )
fi

while [[ ! -s "${BANK_DIR}/COMPLETE" ]]; do
  echo "waiting for the 128-frequency aperture-PB visibility bank" >&2
  sleep "${GPU_RECHECK_SECONDS}"
done
mkdir -p \
  "${OUT_DIR}/logs" \
  "${OUT_DIR}/beam_cache_shared" \
  "${OUT_DIR}/tools"

mapfile -t FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; [print(f"{x:.2f}") for x in json.load(open(sys.argv[1]))["frequencies_mhz"]]' \
    "${FREQUENCY_CONFIG}"
)
if [[ "${#FREQUENCIES[@]}" -ne 128 ]]; then
  echo "the wideband screen requires exactly 128 input frequencies" >&2
  exit 2
fi

if [[ ! -s "${SKY_CACHE}" ]]; then
  "${PYTHON}" "${SKY_BUILDER}" \
    --config "${FREQUENCY_CONFIG}" \
    --osm-pattern "${SOURCE_ROOT}/osm/eor_{freq:.2f}.osm" \
    --out "${SKY_CACHE}" \
    >"${OUT_DIR}/logs/sky_cache.log" 2>&1
fi

shared_selection_dir="${OUT_DIR}/shared_row_selection"
shared_rows_per_bin=$((ROWS_PER_BIN * PARTITION_COUNT))
if [[ ! -s "${shared_selection_dir}/result.npz" ]]; then
  "${PYTHON}" "${SELECTION_SCRIPT}" \
    --config "${CONFIG}" \
    --bank-dir "${BANK_DIR}" \
    --out-dir "${shared_selection_dir}" \
    --rows-per-kperp-bin "${shared_rows_per_bin}" \
    --row-scope all \
    "${row_kperp_limit_args[@]}" \
    --row-seed "${ROW_SEED}" \
    --row-partition-index 0 \
    --row-partition-count 1 \
    >"${OUT_DIR}/logs/row_selection_shared.log" 2>&1
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
        if (($2 + 0) >= minimum_free && ($3 + 0) < util_limit) {
          print $1;
          count += 1;
          if (count >= maximum_count) exit;
        }
      }'
}

wait_for_gpus() {
  AVAILABLE_GPUS=()
  while [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; do
    mapfile -t AVAILABLE_GPUS < <(select_gpus)
    if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
      echo "no GPU currently satisfies the exact-PB response limits" >&2
      sleep "${GPU_RECHECK_SECONDS}"
    fi
  done
}

build_cache() {
  local frequency="$1"
  local gpu="$2"
  local cache_dir="${OUT_DIR}/beam_cache_shared/freq_${frequency}"
  if [[ -s "${cache_dir}/metadata.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=8 \
    "${PYTHON}" "${BEAM_BUILDER}" \
      --bank-shard "${BANK_DIR}/shards/freq_${frequency}.npz" \
      --sky-cache "${SKY_CACHE}" \
      --oskar-config "${BANK_DIR}/configs/sim_eor_${frequency}.ini" \
      --oskar-prefix "${OSKAR_PREFIX}" \
      --compiler-library-dir "${OSKAR_RUNTIME}/lib" \
      --helper-binary "${OUT_DIR}/tools/evaluate_oskar_aperture_row_beam_factors" \
      --out-dir "${cache_dir}" \
      --selected-row-result "${shared_selection_dir}/result.npz" \
      --source-chunk 32768 \
      >"${OUT_DIR}/logs/beam_cache_shared_${frequency}.log" 2>&1
}

wait_for_gpus
first_missing=""
for frequency in "${FREQUENCIES[@]}"; do
  if [[ ! -s "${OUT_DIR}/beam_cache_shared/freq_${frequency}/metadata.json" ]]; then
    first_missing="${frequency}"
    break
  fi
done
if [[ -n "${first_missing}" ]]; then
  build_cache "${first_missing}" "${AVAILABLE_GPUS[0]}"
fi

cache_worker() {
  local worker_index="$1"
  local gpu="$2"
  local index
  for ((index = worker_index; index < ${#FREQUENCIES[@]}; index += ${#AVAILABLE_GPUS[@]})); do
    build_cache "${FREQUENCIES[index]}" "${gpu}"
  done
}

pids=()
for index in "${!AVAILABLE_GPUS[@]}"; do
  cache_worker "${index}" "${AVAILABLE_GPUS[index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more exact-PB cache workers failed" >&2
  exit "${status}"
fi

for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  selection_dir="${OUT_DIR}/part_${partition}/row_selection"
  if [[ ! -s "${selection_dir}/result.npz" ]]; then
    "${PYTHON}" "${SELECTION_SCRIPT}" \
      --config "${CONFIG}" \
      --bank-dir "${BANK_DIR}" \
      --out-dir "${selection_dir}" \
      --rows-per-kperp-bin "${ROWS_PER_BIN}" \
      --row-scope all \
      "${row_kperp_limit_args[@]}" \
      --row-seed "${ROW_SEED}" \
      --row-partition-index "${partition}" \
      --row-partition-count "${PARTITION_COUNT}" \
      >"${OUT_DIR}/logs/row_selection_${partition}.log" 2>&1
  fi
done

evaluate_partition() {
  local partition="$1"
  local gpu="$2"
  local evaluate_dir="${OUT_DIR}/part_${partition}/evaluate"
  mkdir -p "${evaluate_dir}"
  if [[ -s "${evaluate_dir}/result.npz" &&
        -s "${evaluate_dir}/result.json" ]]; then
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
    --row-scope all \
    "${row_kperp_limit_args[@]}" \
    --row-seed "${ROW_SEED}" \
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
      "${OUT_DIR}/beam_cache_shared/freq_{freq:.2f}" \
    --source-chunk 32768 \
    --row-chunk 32 \
    --foreground-filter dpss_hard \
    --filter-bandwidth-scope full_band \
    --spectral-taper hann \
    --minimum-window-self-fraction 0 \
    --minimum-relative-sensitivity 0 \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/evaluate_${partition}.log" 2>&1
}

wait_for_gpus
evaluation_worker() {
  local worker_index="$1"
  local gpu="$2"
  local partition
  for ((partition = worker_index; partition < PARTITION_COUNT; partition += ${#AVAILABLE_GPUS[@]})); do
    evaluate_partition "${partition}" "${gpu}"
  done
}

pids=()
for index in "${!AVAILABLE_GPUS[@]}"; do
  evaluation_worker "${index}" "${AVAILABLE_GPUS[index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more Q_beta evaluation workers failed" >&2
  exit "${status}"
fi

combine_args=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  combine_args+=(
    --input-dir "${OUT_DIR}/part_${partition}/evaluate"
  )
done
"${PYTHON}" "${COMBINER}" \
  "${combine_args[@]}" \
  --out-dir "${OUT_DIR}/combined" \
  --response-rcond 1e-4 \
  >"${OUT_DIR}/logs/combine.log" 2>&1

"${PYTHON}" "${COARSE_EVALUATOR}" \
  --combined-npz "${OUT_DIR}/combined/result.npz" \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}/coarse" \
  --minimum-kperp-index 4 \
  "${row_kperp_limit_args[@]}" \
  --minimum-relative-response 0.1 \
  --minimum-window-fraction 0.95 \
  --profile fine_response \
  --profile pair_kperp_response \
  --profile quad_kperp_response \
  --profile quad_kperp_kpar2_response \
  >"${OUT_DIR}/logs/coarse.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_DIR}/COMPLETE"
echo "128-to-64 aperture-PB Q_beta screen complete: ${OUT_DIR}"
