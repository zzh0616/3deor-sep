#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
LOCAL_ROOT="${LOCAL_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
OUT_ROOT="${OUT_ROOT:-${LOCAL_ROOT}/shared_covariance}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SKY_CACHE="${SKY_CACHE:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/eor_intrinsic_sky_128freq.npz}"
EVALUATION_SKY_CACHE="${EVALUATION_SKY_CACHE:-${PROJECT_ROOT}/runs/cube1_fullsky_isobeam_512_128freq_20260728/eor_intrinsic_sky_128freq.npz}"
BEAM_CACHE_ROOT="${BEAM_CACHE_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/beam_cache_shared}"
FULL_CONFIG="${FULL_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_128wide_isobeam_patch.json}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
REALIZATION_COUNT="${REALIZATION_COUNT:-64}"
REALIZATION_BATCH_SIZE="${REALIZATION_BATCH_SIZE:-64}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-30000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"
GPU_INDEX="${GPU_INDEX:-}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
SHARED_EVALUATOR="${SHARED_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_shared_local_covariance.py}"
COVARIANCE_EVALUATOR="${COVARIANCE_EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_local_redshift_covariance.py}"

for required in \
  "${LOCAL_ROOT}/COMPLETE" \
  "${LOCAL_ROOT}/configs/manifest.json" \
  "${BANK_DIR}/COMPLETE" \
  "${SKY_CACHE}" \
  "${EVALUATION_SKY_CACHE}" \
  "${FULL_CONFIG}" \
  "${SHARED_EVALUATOR}" \
  "${COVARIANCE_EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing shared-covariance input: ${required}" >&2
    exit 1
  fi
done
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/covariance"
printf '%s\n' "$$" >"${OUT_ROOT}/RUN.pid"

if [[ -n "${GPU_INDEX}" ]]; then
  gpu="${GPU_INDEX}"
else
  while true; do
    gpu="$(
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
            if (($2 + 0) >= minimum_free && ($3 + 0) <= util_limit) {
              print $1;
              exit;
            }
          }'
    )"
    if [[ -n "${gpu}" ]]; then
      break
    fi
    sleep "${GPU_RECHECK_SECONDS}"
  done
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SHARED_EVALUATOR}" \
  --full-config "${FULL_CONFIG}" \
  --local-manifest "${LOCAL_ROOT}/configs/manifest.json" \
  --local-root "${LOCAL_ROOT}" \
  --bank-dir "${BANK_DIR}" \
  --sky-cache "${SKY_CACHE}" \
  --evaluation-sky-cache "${EVALUATION_SKY_CACHE}" \
  --aperture-row-beam-cache-pattern "${BEAM_CACHE_ROOT}/freq_{freq:.2f}" \
  --out-dir "${OUT_ROOT}/shared_inputs" \
  --profile quad_kperp_response \
  --partition-count "${PARTITION_COUNT}" \
  --realization-count "${REALIZATION_COUNT}" \
  --realization-batch-size "${REALIZATION_BATCH_SIZE}" \
  --device cuda:0 \
  --source-chunk 8192 \
  >"${OUT_ROOT}/logs/shared_inputs.log" 2>&1

mapfile -t WINDOW_ROWS < <(
  "${PYTHON}" -c \
    'import json,sys
for row in json.load(open(sys.argv[1]))["windows"]:
    print(row["label"])' \
    "${LOCAL_ROOT}/configs/manifest.json"
)
covariance_args=()
for label in "${WINDOW_ROWS[@]}"; do
  covariance_args+=(
    --window
    "${label}=${OUT_ROOT}/shared_inputs/window_products/${label}.npz"
  )
done
"${PYTHON}" "${COVARIANCE_EVALUATOR}" \
  "${covariance_args[@]}" \
  --out-dir "${OUT_ROOT}/covariance" \
  --profile quad_kperp_response \
  >"${OUT_ROOT}/logs/covariance.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_ROOT}/COMPLETE"
echo "shared local-redshift covariance complete: ${OUT_ROOT}"
