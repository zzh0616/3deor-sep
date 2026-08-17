#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
EXACT_ROOT="${EXACT_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728/local_04_116p3_119p4mhz}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728/configs}"
CONFIG="${CONFIG:-${CONFIG_ROOT}/local_04_116p3_119p4mhz_analysis.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CONFIG_ROOT}/local_04_116p3_119p4mhz_input.json}"
SKY_CACHE="${SKY_CACHE:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/eor_intrinsic_sky_128freq.npz}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_operator_hierarchy_screen4_20260817}"
COMMON_ROOT="${OUT_DIR}/common_scalar_beam"
DELAY_ROOT="${OUT_DIR}/delay_diagonal"

OSKAR_PREFIX="${OSKAR_PREFIX:-${PROJECT_ROOT}/../local/radio-202605-oskar212-cuda-casa380}"
OSKAR="${OSKAR:-${OSKAR_PREFIX}/bin/oskar_sim_beam_pattern}"
OSKAR_RUNTIME="${OSKAR_RUNTIME:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380}"
TELESCOPE_DIR="${TELESCOPE_DIR:-/data/zhenghao/fg_rmw/runs/operator_pilot_106_20260530/telescope/ska1_low.tm}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"

BEAM_BUILDER="${BEAM_BUILDER:-${CODE_ROOT}/ops_scripts/build_oskar_aperture_beam_cache.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
COARSE_EVALUATOR="${COARSE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py}"
DIAGONAL_BUILDER="${DIAGONAL_BUILDER:-${CODE_ROOT}/ops_scripts/build_visibility_qbeta_diagonal_response_control.py}"
COMPARISON="${COMPARISON:-${CODE_ROOT}/ops_scripts/compare_visibility_qbeta_operator_hierarchy.py}"

PARTITION_COUNT="${PARTITION_COUNT:-4}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
ROW_SEED="${ROW_SEED:-20260725}"
MAX_WORKERS="${MAX_WORKERS:-4}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-80}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"

for required in \
  "${BANK_DIR}/COMPLETE" \
  "${CONFIG}" \
  "${FREQUENCY_CONFIG}" \
  "${SKY_CACHE}" \
  "${EXACT_ROOT}/combined/result.npz" \
  "${EXACT_ROOT}/coarse/products.npz" \
  "${OSKAR}" \
  "${TELESCOPE_DIR}" \
  "${BEAM_BUILDER}" \
  "${EVALUATOR}" \
  "${COMBINER}" \
  "${COARSE_EVALUATOR}" \
  "${DIAGONAL_BUILDER}" \
  "${COMPARISON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing operator-hierarchy input: ${required}" >&2
    exit 1
  fi
done

mkdir -p \
  "${OUT_DIR}/logs" \
  "${COMMON_ROOT}/beam_cache" \
  "${DELAY_ROOT}"

mapfile -t FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; [print(f"{x:.2f}") for x in json.load(open(sys.argv[1]))["frequencies_mhz"]]' \
    "${FREQUENCY_CONFIG}"
)
if [[ "${#FREQUENCIES[@]}" -ne 64 ]]; then
  echo "operator hierarchy screen requires exactly 64 input frequencies" >&2
  exit 2
fi

if [[ ! -s "${DELAY_ROOT}/coarse/products.npz" ]]; then
  "${PYTHON}" "${DIAGONAL_BUILDER}" \
    --combined-npz "${EXACT_ROOT}/combined/result.npz" \
    --out-dir "${DELAY_ROOT}/combined" \
    >"${OUT_DIR}/logs/delay_diagonal_build.log" 2>&1
  "${PYTHON}" "${COARSE_EVALUATOR}" \
    --combined-npz "${DELAY_ROOT}/combined/result.npz" \
    --config "${CONFIG}" \
    --out-dir "${DELAY_ROOT}/coarse" \
    --minimum-kperp-index 4 \
    --minimum-relative-response 0.1 \
    --minimum-window-fraction 0.95 \
    --profile fine_response \
    --profile pair_kperp_response \
    --profile quad_kperp_response \
    --profile quad_kperp_kpar2_response \
    >"${OUT_DIR}/logs/delay_diagonal_coarse.log" 2>&1
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

wait_for_gpus() {
  AVAILABLE_GPUS=()
  while [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; do
    mapfile -t AVAILABLE_GPUS < <(select_gpus)
    if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
      echo "no GPU currently satisfies the live hierarchy limits" >&2
      sleep "${GPU_RECHECK_SECONDS}"
    fi
  done
}

build_common_cache() {
  local frequency="$1"
  local gpu="$2"
  local cache_dir="${COMMON_ROOT}/beam_cache/freq_${frequency}"
  if [[ -s "${cache_dir}/result.json" &&
        -s "${cache_dir}/beam_cache.npz" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=8 \
    LD_LIBRARY_PATH="${OSKAR_RUNTIME}/lib:${OSKAR_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${PYTHON}" "${BEAM_BUILDER}" \
      --out-dir "${cache_dir}" \
      --oskar "${OSKAR}" \
      --telescope-dir "${TELESCOPE_DIR}" \
      --osm "${SOURCE_ROOT}/osm/eor_${frequency}.osm" \
      --frequency-mhz "${frequency}" \
      --phase-ra-deg 0 \
      --phase-dec-deg -27 \
      --start-time-utc 2030-01-01T06:30:00.0 \
      --observation-length-s 320 \
      --time-steps 32 \
      --station-id 0 \
      --max-sources-per-chunk 131072 \
      --use-gpus \
      >"${OUT_DIR}/logs/common_beam_${frequency}.log" 2>&1
}

missing_frequencies=()
for frequency in "${FREQUENCIES[@]}"; do
  if [[ ! -s "${COMMON_ROOT}/beam_cache/freq_${frequency}/result.json" ||
        ! -s "${COMMON_ROOT}/beam_cache/freq_${frequency}/beam_cache.npz" ]]; then
    missing_frequencies+=("${frequency}")
  fi
done
next=0
while ((next < ${#missing_frequencies[@]})); do
  wait_for_gpus
  pids=()
  for offset in "${!AVAILABLE_GPUS[@]}"; do
    task_index=$((next + offset))
    if ((task_index >= ${#missing_frequencies[@]})); then
      break
    fi
    build_common_cache \
      "${missing_frequencies[task_index]}" \
      "${AVAILABLE_GPUS[offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one common-beam cache worker failed" >&2
    exit "${status}"
  fi
  next=$((next + ${#pids[@]}))
done

evaluate_partition() {
  local partition="$1"
  local gpu="$2"
  local evaluate_dir="${COMMON_ROOT}/part_${partition}/evaluate"
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
    --aperture-common-beam-cache-pattern \
      "${COMMON_ROOT}/beam_cache/freq_{freq:.2f}" \
    --source-chunk 32768 \
    --row-chunk 32 \
    --foreground-filter dpss_hard \
    --filter-bandwidth-scope full_band \
    --spectral-taper hann \
    --minimum-window-self-fraction 0 \
    --minimum-relative-sensitivity 0 \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/common_evaluate_${partition}.log" 2>&1
}

remaining_partitions=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  if [[ ! -s "${COMMON_ROOT}/part_${partition}/evaluate/result.npz" ||
        ! -s "${COMMON_ROOT}/part_${partition}/evaluate/result.json" ]]; then
    remaining_partitions+=("${partition}")
  fi
done
next=0
while ((next < ${#remaining_partitions[@]})); do
  wait_for_gpus
  pids=()
  for offset in "${!AVAILABLE_GPUS[@]}"; do
    task_index=$((next + offset))
    if ((task_index >= ${#remaining_partitions[@]})); then
      break
    fi
    evaluate_partition \
      "${remaining_partitions[task_index]}" \
      "${AVAILABLE_GPUS[offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one common-beam evaluator failed" >&2
    exit "${status}"
  fi
  next=$((next + ${#pids[@]}))
done

combine_args=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  combine_args+=(
    --input-dir "${COMMON_ROOT}/part_${partition}/evaluate"
  )
done
"${PYTHON}" "${COMBINER}" \
  "${combine_args[@]}" \
  --out-dir "${COMMON_ROOT}/combined" \
  --response-rcond 1e-4 \
  >"${OUT_DIR}/logs/common_combine.log" 2>&1

"${PYTHON}" "${COARSE_EVALUATOR}" \
  --combined-npz "${COMMON_ROOT}/combined/result.npz" \
  --config "${CONFIG}" \
  --out-dir "${COMMON_ROOT}/coarse" \
  --minimum-kperp-index 4 \
  --minimum-relative-response 0.1 \
  --minimum-window-fraction 0.95 \
  --profile fine_response \
  --profile pair_kperp_response \
  --profile quad_kperp_response \
  --profile quad_kperp_kpar2_response \
  >"${OUT_DIR}/logs/common_coarse.log" 2>&1

"${PYTHON}" "${COMPARISON}" \
  --exact-coarse "${EXACT_ROOT}/coarse/products.npz" \
  --common-coarse "${COMMON_ROOT}/coarse/products.npz" \
  --delay-coarse "${DELAY_ROOT}/coarse/products.npz" \
  --exact-combined-npz "${EXACT_ROOT}/combined/result.npz" \
  --common-combined-npz "${COMMON_ROOT}/combined/result.npz" \
  --exact-combined-json "${EXACT_ROOT}/combined/result.json" \
  --common-combined-json "${COMMON_ROOT}/combined/result.json" \
  --out-dir "${OUT_DIR}/comparison" \
  >"${OUT_DIR}/logs/comparison.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_DIR}/COMPLETE"
echo "operator hierarchy screen complete: ${OUT_DIR}"
