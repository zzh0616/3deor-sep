#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
PARENT_RUN="${PARENT_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
PARTITION_INDICES="${PARTITION_INDICES:-}"
ROWS_PER_BIN="${ROWS_PER_BIN:-12}"
CALIBRATION_REPEATS="${CALIBRATION_REPEATS:-1}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-1}"
MIXTURE_REPEATS="${MIXTURE_REPEATS:-64}"
WINDOW_INDICES="${WINDOW_INDICES:-0,1,2,3,4}"
INPUT_CHANNEL_COUNT="${INPUT_CHANNEL_COUNT:-64}"
ANALYSIS_CHANNEL_COUNT="${ANALYSIS_CHANNEL_COUNT:-32}"
STRIDE_CHANNELS="${STRIDE_CHANNELS:-16}"
TARGET_START="${TARGET_START:-32}"
TARGET_STOP="${TARGET_STOP:-96}"
FOREGROUND_SUPPORT_ANGLE_DEG="${FOREGROUND_SUPPORT_ANGLE_DEG:-}"
MAX_WORKERS="${MAX_WORKERS:-4}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-99}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"

FULL_CONFIG="${FULL_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_128wide_isobeam_patch.json}"
ANALYSIS_TEMPLATE="${ANALYSIS_TEMPLATE:-${CODE_ROOT}/configs/ps2d_v2_32central_isobeam_patch.json}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-${CODE_ROOT}/configs/ps2d_v2_64wide_isobeam_patch.json}"
CONFIG_BUILDER="${CONFIG_BUILDER:-${CODE_ROOT}/ops_scripts/build_visibility_qbeta_local_redshift_configs.py}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/calibrate_visibility_qbeta_noiseless.py}"
COMBINER="${COMBINER:-${CODE_ROOT}/ops_scripts/combine_visibility_qbeta_row_partitions.py}"
COARSE_EVALUATOR="${COARSE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py}"
COVARIANCE_EVALUATOR="${COVARIANCE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_local_redshift_covariance.py}"
SHARED_COVARIANCE_PRODUCTS="${SHARED_COVARIANCE_PRODUCTS:-}"
SKY_CACHE="${SKY_CACHE:-${PARENT_RUN}/eor_intrinsic_sky_128freq.npz}"
BEAM_CACHE_ROOT="${BEAM_CACHE_ROOT:-${PARENT_RUN}/beam_cache_shared}"
ROW_SEED="${ROW_SEED:-20260725}"

for required in \
  "${BANK_DIR}/COMPLETE" \
  "${PARENT_RUN}/COMPLETE" \
  "${SKY_CACHE}" \
  "${CONFIG_BUILDER}" \
  "${EVALUATOR}" \
  "${COMBINER}" \
  "${COARSE_EVALUATOR}" \
  "${COVARIANCE_EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing local-redshift input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}/configs" "${OUT_DIR}/logs"
config_builder_args=()
if [[ -n "${FOREGROUND_SUPPORT_ANGLE_DEG}" ]]; then
  config_builder_args+=(
    --foreground-support-angle-deg "${FOREGROUND_SUPPORT_ANGLE_DEG}"
  )
fi
"${PYTHON}" "${CONFIG_BUILDER}" \
  --full-config "${FULL_CONFIG}" \
  --analysis-template "${ANALYSIS_TEMPLATE}" \
  --input-template "${INPUT_TEMPLATE}" \
  --out-dir "${OUT_DIR}/configs" \
  --input-channel-count "${INPUT_CHANNEL_COUNT}" \
  --analysis-channel-count "${ANALYSIS_CHANNEL_COUNT}" \
  --stride-channels "${STRIDE_CHANNELS}" \
  --target-start "${TARGET_START}" \
  --target-stop "${TARGET_STOP}" \
  "${config_builder_args[@]}" \
  >"${OUT_DIR}/logs/configs.log" 2>&1

mapfile -t WINDOW_ROWS < <(
  "${PYTHON}" -c \
    'import json,sys
p=json.load(open(sys.argv[1]))
keep={int(x) for x in sys.argv[2].split(",") if x}
for w in p["windows"]:
    if int(w["index"]) in keep:
        print("|".join((w["label"],w["analysis_config"],w["input_config"])))' \
    "${OUT_DIR}/configs/manifest.json" "${WINDOW_INDICES}"
)
if [[ "${#WINDOW_ROWS[@]}" -eq 0 ]]; then
  echo "WINDOW_INDICES selects no local-redshift windows" >&2
  exit 2
fi
if [[ -n "${PARTITION_INDICES}" ]]; then
  IFS="," read -r -a PARTITIONS <<<"${PARTITION_INDICES}"
else
  PARTITIONS=()
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    PARTITIONS+=("${partition}")
  done
fi
for partition in "${PARTITIONS[@]}"; do
  if [[ ! "${partition}" =~ ^[0-9]+$ ]] ||
     ((partition < 0 || partition >= PARTITION_COUNT)); then
    echo "invalid partition index: ${partition}" >&2
    exit 2
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
        if (($2 + 0) >= minimum_free && ($3 + 0) <= util_limit) {
          print $1;
          count += 1;
          if (count >= maximum_count) exit;
        }
      }'
}

run_task() {
  local row="$1"
  local partition="$2"
  local gpu="$3"
  local label analysis_name input_name
  IFS="|" read -r label analysis_name input_name <<<"${row}"
  local root="${OUT_DIR}/${label}"
  local evaluate_dir="${root}/part_${partition}/evaluate"
  mkdir -p "${evaluate_dir}"
  if [[ -s "${evaluate_dir}/result.npz" &&
        -s "${evaluate_dir}/result.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
    --config "${OUT_DIR}/configs/${analysis_name}" \
    --frequency-config "${OUT_DIR}/configs/${input_name}" \
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
    --calibration-repeats "${CALIBRATION_REPEATS}" \
    --validation-repeats "${VALIDATION_REPEATS}" \
    --mixture-repeats "${MIXTURE_REPEATS}" \
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
    >"${OUT_DIR}/logs/${label}_part_${partition}.log" 2>&1
}

tasks=()
for row in "${WINDOW_ROWS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    tasks+=("${row}::${partition}")
  done
done

next=0
while ((next < ${#tasks[@]})); do
  mapfile -t AVAILABLE_GPUS < <(select_gpus)
  if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
    echo "no GPU currently satisfies the live local-redshift limits" >&2
    sleep "${GPU_RECHECK_SECONDS}"
    continue
  fi
  pids=()
  for offset in "${!AVAILABLE_GPUS[@]}"; do
    task_index=$((next + offset))
    if ((task_index >= ${#tasks[@]})); then
      break
    fi
    task="${tasks[task_index]}"
    row="${task%::*}"
    partition="${task##*::}"
    run_task "${row}" "${partition}" "${AVAILABLE_GPUS[offset]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one local-redshift evaluator failed" >&2
    exit "${status}"
  fi
  next=$((next + ${#pids[@]}))
done

for row in "${WINDOW_ROWS[@]}"; do
  IFS="|" read -r label analysis_name input_name <<<"${row}"
  root="${OUT_DIR}/${label}"
  combine_args=()
  for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
    combine_args+=(--input-dir "${root}/part_${partition}/evaluate")
  done
  "${PYTHON}" "${COMBINER}" \
    "${combine_args[@]}" \
    --out-dir "${root}/combined" \
    --response-rcond 1e-4 \
    >"${OUT_DIR}/logs/${label}_combine.log" 2>&1
  "${PYTHON}" "${COARSE_EVALUATOR}" \
    --combined-npz "${root}/combined/result.npz" \
    --config "${OUT_DIR}/configs/${analysis_name}" \
    --out-dir "${root}/coarse" \
    --minimum-kperp-index 4 \
    --minimum-relative-response 0.1 \
    --minimum-window-fraction 0.95 \
    --profile fine_response \
    --profile pair_kperp_response \
    --profile quad_kperp_response \
    --profile quad_kperp_kpar2_response \
    >"${OUT_DIR}/logs/${label}_coarse.log" 2>&1
done

if [[ -n "${SHARED_COVARIANCE_PRODUCTS}" ]]; then
  covariance_args=()
  for row in "${WINDOW_ROWS[@]}"; do
    IFS="|" read -r label analysis_name input_name <<<"${row}"
    covariance_args+=(
      --window "${label}=${SHARED_COVARIANCE_PRODUCTS}/${label}.npz"
    )
  done
  "${PYTHON}" "${COVARIANCE_EVALUATOR}" \
    "${covariance_args[@]}" \
    --out-dir "${OUT_DIR}/covariance" \
    --profile quad_kperp_response \
    >"${OUT_DIR}/logs/covariance.log" 2>&1
else
  mkdir -p "${OUT_DIR}/covariance"
  printf '%s\n' \
    'requires certified shared full-band realizations; per-window same-seed probes are not aligned' \
    >"${OUT_DIR}/covariance/PENDING_SHARED_REALIZATIONS"
fi

printf 'complete %s\n' "$(date -Is)" >"${OUT_DIR}/COMPLETE"
echo "local-redshift Q_beta mosaic complete: ${OUT_DIR}"
