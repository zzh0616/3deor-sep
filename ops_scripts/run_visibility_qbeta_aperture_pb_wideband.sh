#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/3dnet}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_64freq_20260725}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_64freq_20260725}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_64freq_20260725}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32central_isobeam_patch.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64wide_isobeam_patch.json}"
OSKAR_PREFIX="${OSKAR_PREFIX:-/data/zhenghao/local/radio-202605-oskar212-cuda-casa380}"
OSKAR_RUNTIME="${OSKAR_RUNTIME:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/bin/python}"
SELECTION_SCRIPT="${SELECTION_SCRIPT:-${CODE_ROOT}/ops_scripts/prepare_visibility_qbeta_row_selection.py}"
BEAM_BUILDER="${BEAM_BUILDER:-${CODE_ROOT}/ops_scripts/build_oskar_aperture_row_beam_cache.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
COARSE_EVALUATOR="${COARSE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py}"
SKY_GEOMETRY_CACHE="${SKY_GEOMETRY_CACHE:-${PROJECT_ROOT}/runs/visibility_gaussian_source_119p40_20260725/eor_intrinsic_sky_119p40.npz}"
SKY_CACHE="${SKY_CACHE:-${OUT_DIR}/eor_intrinsic_sky_64freq.npz}"
PARTITION_COUNT="${PARTITION_COUNT:-20}"
PARTITION_FIRST="${PARTITION_FIRST:-0}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
ROW_SEED="${ROW_SEED:-20260725}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-65000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-50}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-300}"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/beam_cache" "${OUT_DIR}/tools"
while [[ ! -s "${BANK_DIR}/COMPLETE" ]]; do
  echo "Waiting for the 64-frequency aperture-PB visibility bank." >&2
  sleep "${GPU_RECHECK_SECONDS}"
done

mapfile -t FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); [print(f"{x:.2f}") for x in d["frequencies_mhz"]]' \
    "${FREQUENCY_CONFIG}"
)
if [[ "${#FREQUENCIES[@]}" -ne 64 ]]; then
  echo "The wideband PB run requires exactly 64 frequencies." >&2
  exit 2
fi

select_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, \
      -v minimum_free="${GPU_MIN_FREE_MIB}" \
      -v util_limit="${GPU_UTIL_LIMIT_PERCENT}" \
      '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (($2 + 0) >= minimum_free && ($3 + 0) < util_limit) {
          print $1;
          exit;
        }
      }'
}

wait_for_gpu() {
  local selected_gpu=""
  while [[ -z "${selected_gpu}" ]]; do
    selected_gpu="$(select_gpu)"
    if [[ -z "${selected_gpu}" ]]; then
      echo "No GPU satisfies the live wideband Q_beta limits." >&2
      sleep "${GPU_RECHECK_SECONDS}"
    fi
  done
  printf '%s\n' "${selected_gpu}"
}

helper_binary="${OUT_DIR}/tools/evaluate_oskar_aperture_row_beam_factors"
shared_selection_dir="${OUT_DIR}/shared_row_selection"
shared_rows_per_bin=$((ROWS_PER_BIN * PARTITION_COUNT))
if [[ ! -s "${shared_selection_dir}/result.npz" ]]; then
  "${PYTHON}" "${SELECTION_SCRIPT}" \
    --config "${CONFIG}" \
    --bank-dir "${BANK_DIR}" \
    --out-dir "${shared_selection_dir}" \
    --rows-per-kperp-bin "${shared_rows_per_bin}" \
    --row-scope all \
    --row-seed "${ROW_SEED}" \
    --row-partition-index 0 \
    --row-partition-count 1 \
    >"${OUT_DIR}/logs/row_selection_shared.log" 2>&1
fi

for frequency in "${FREQUENCIES[@]}"; do
  cache_dir="${OUT_DIR}/beam_cache_shared/freq_${frequency}"
  if [[ -s "${cache_dir}/metadata.json" ]]; then
    continue
  fi
  selected_gpu="$(wait_for_gpu)"
  CUDA_VISIBLE_DEVICES="${selected_gpu}" OMP_NUM_THREADS=8 \
    "${PYTHON}" "${BEAM_BUILDER}" \
      --bank-shard "${BANK_DIR}/shards/freq_${frequency}.npz" \
      --sky-cache "${SKY_GEOMETRY_CACHE}" \
      --oskar-config "${BANK_DIR}/configs/sim_eor_${frequency}.ini" \
      --oskar-prefix "${OSKAR_PREFIX}" \
      --compiler-library-dir "${OSKAR_RUNTIME}/lib" \
      --helper-binary "${helper_binary}" \
      --out-dir "${cache_dir}" \
      --selected-row-result "${shared_selection_dir}/result.npz" \
      --source-chunk 32768 \
      >"${OUT_DIR}/logs/beam_cache_shared_${frequency}.log" 2>&1
done

for ((partition = PARTITION_FIRST; partition < PARTITION_COUNT; partition++)); do
  partition_dir="${OUT_DIR}/part_${partition}"
  selection_dir="${partition_dir}/row_selection"
  evaluate_dir="${partition_dir}/evaluate"
  mkdir -p "${partition_dir}" "${evaluate_dir}"
  if [[ -s "${evaluate_dir}/result.npz" &&
        -s "${evaluate_dir}/result.json" ]]; then
    continue
  fi
  if [[ ! -s "${selection_dir}/result.npz" ]]; then
    "${PYTHON}" "${SELECTION_SCRIPT}" \
      --config "${CONFIG}" \
      --bank-dir "${BANK_DIR}" \
      --out-dir "${selection_dir}" \
      --rows-per-kperp-bin "${ROWS_PER_BIN}" \
      --row-scope all \
      --row-seed "${ROW_SEED}" \
      --row-partition-index "${partition}" \
      --row-partition-count "${PARTITION_COUNT}" \
      >"${OUT_DIR}/logs/row_selection_${partition}.log" 2>&1
  fi

  selected_gpu="$(wait_for_gpu)"
  CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON}" "${EVALUATOR}" \
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
    --operator-storage gpu \
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
done

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
  --minimum-relative-response 0.1 \
  --minimum-window-fraction 0.95 \
  --profile fine_response \
  --profile pair_kperp_response \
  --profile quad_kperp_response \
  --profile quad_kperp_kpar2_response \
  >"${OUT_DIR}/logs/coarse.log" 2>&1

date -Is >"${OUT_DIR}/COMPLETE"
echo "Wideband aperture-PB visibility Q_beta run complete: ${OUT_DIR}"
