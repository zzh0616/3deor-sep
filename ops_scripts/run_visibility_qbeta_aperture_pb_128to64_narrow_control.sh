#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
WIDE_DIR="${WIDE_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen${PARTITION_COUNT}_20260726}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_narrow_control_screen${PARTITION_COUNT}_20260726}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64central_111p5_117p8_isobeam_patch.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_128wide_isobeam_patch.json}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
SELECTION_SCRIPT="${SELECTION_SCRIPT:-${CODE_ROOT}/ops_scripts/prepare_visibility_qbeta_row_selection.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
COARSE_EVALUATOR="${COARSE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py}"
SKY_CACHE="${SKY_CACHE:-${WIDE_DIR}/eor_intrinsic_sky_128freq.npz}"
ROW_SEED="${ROW_SEED:-20260725}"
MAXIMUM_KPERP_INDEX_EXCLUSIVE="${MAXIMUM_KPERP_INDEX_EXCLUSIVE:-}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-50}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"
UPSTREAM_PID="${UPSTREAM_PID:-}"

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

mkdir -p "${OUT_DIR}/logs"
while [[ ! -s "${WIDE_DIR}/COMPLETE" ]]; do
  if [[ -n "${UPSTREAM_PID}" ]] && ! kill -0 "${UPSTREAM_PID}" 2>/dev/null; then
    echo "upstream pipeline ${UPSTREAM_PID} exited before the wide screen completed" >&2
    exit 1
  fi
  echo "waiting for the wide 128-to-64 screen: $(date -Is)" >&2
  sleep "${GPU_RECHECK_SECONDS}"
done

if [[ ! -s "${SKY_CACHE}" ]]; then
  echo "missing reusable 128-frequency sky cache: ${SKY_CACHE}" >&2
  exit 1
fi
cache_count="$(
  find "${WIDE_DIR}/beam_cache_shared" \
    -mindepth 2 -maxdepth 2 -type f -name metadata.json |
    wc -l
)"
if [[ "${cache_count}" -ne 128 ]]; then
  echo "expected 128 reusable exact-PB caches, found ${cache_count}" >&2
  exit 1
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

AVAILABLE_GPUS=()
while [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; do
  mapfile -t AVAILABLE_GPUS < <(select_gpus)
  if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
    echo "no GPU currently satisfies the narrow-control limits" >&2
    sleep "${GPU_RECHECK_SECONDS}"
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
      "${WIDE_DIR}/beam_cache_shared/freq_{freq:.2f}" \
    --source-chunk 32768 \
    --row-chunk 32 \
    --foreground-filter dpss_hard \
    --filter-bandwidth-scope analysis_subband \
    --spectral-taper hann \
    --minimum-window-self-fraction 0 \
    --minimum-relative-sensitivity 0 \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/evaluate_${partition}.log" 2>&1
}

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
  echo "one or more narrow-control evaluation workers failed" >&2
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
echo "128-to-64 aperture-PB narrow control complete: ${OUT_DIR}"
