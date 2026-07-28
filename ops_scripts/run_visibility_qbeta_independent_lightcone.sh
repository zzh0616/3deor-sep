#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
LOCAL_ROOT="${LOCAL_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_smoke_20260728}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_independent_lightcone_20260728}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
LABEL="${LABEL:-local_02_113p1_116p2mhz}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-600}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"

ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-${LOCAL_ROOT}/configs/${LABEL}_analysis.json}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${LOCAL_ROOT}/configs/${LABEL}_input.json}"
COMBINED_DIR="${COMBINED_DIR:-${LOCAL_ROOT}/${LABEL}/combined}"
CALIBRATION_SKY_CACHE="${CALIBRATION_SKY_CACHE:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/eor_intrinsic_sky_128freq.npz}"
EVALUATION_SKY_CACHE="${EVALUATION_SKY_CACHE:-${PROJECT_ROOT}/runs/cube1_fullsky_isobeam_512_128freq_20260728/eor_intrinsic_sky_128freq.npz}"
BEAM_CACHE_ROOT="${BEAM_CACHE_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_aperture_pb_128to64_screen4_20260726/beam_cache_shared}"
EVALUATOR="${EVALUATOR:-${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_independent_lightcone.py}"

for required in \
  "${ANALYSIS_CONFIG}" \
  "${FREQUENCY_CONFIG}" \
  "${COMBINED_DIR}/result.npz" \
  "${CALIBRATION_SKY_CACHE}" \
  "${EVALUATION_SKY_CACHE}" \
  "${EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing independent-lightcone input: ${required}" >&2
    exit 1
  fi
done
partition_args=()
for ((partition = 0; partition < PARTITION_COUNT; partition++)); do
  directory="${LOCAL_ROOT}/${LABEL}/part_${partition}/evaluate"
  if [[ ! -s "${directory}/result.npz" ]]; then
    echo "missing independent-lightcone partition: ${directory}" >&2
    exit 1
  fi
  partition_args+=(--partition-result-dir "${directory}")
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/evaluation_${LABEL}"
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

CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${EVALUATOR}" \
  --config "${ANALYSIS_CONFIG}" \
  --frequency-config "${FREQUENCY_CONFIG}" \
  --bank-dir "${BANK_DIR}" \
  --calibration-sky-cache "${CALIBRATION_SKY_CACHE}" \
  --evaluation-sky-cache "${EVALUATION_SKY_CACHE}" \
  --combined-result-dir "${COMBINED_DIR}" \
  "${partition_args[@]}" \
  --aperture-row-beam-cache-pattern "${BEAM_CACHE_ROOT}/freq_{freq:.2f}" \
  --out-dir "${OUT_ROOT}/evaluation_${LABEL}" \
  --device cuda:0 \
  --source-chunk 8192 \
  >"${OUT_ROOT}/logs/evaluation_${LABEL}.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_ROOT}/COMPLETE"
echo "independent-lightcone Q_beta evaluation complete: ${OUT_ROOT}"
