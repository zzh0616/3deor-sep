#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_128freq_20260726}"
REUSE_CENTRAL_BANK="${REUSE_CENTRAL_BANK:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_64freq_20260725}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
FREQUENCY_CONFIG="${FREQUENCY_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_128wide_isobeam_patch.json}"
REUSE_CONFIG="${REUSE_CONFIG:-${CODE_ROOT}/configs/ps2d_v2_64wide_isobeam_patch.json}"
OSKAR_PREFIX="${OSKAR_PREFIX:-${PROJECT_ROOT}/../local/radio-202605-oskar212-cuda-casa380}"
CASA_PREFIX="${CASA_PREFIX:-${PROJECT_ROOT}/../local/radio-20260517-casa380-py312}"
CONDA_ENV="${CONDA_ENV:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
BUILDER="${BUILDER:-${CODE_ROOT}/ops_scripts/build_chips_visibility_bank.py}"
TELESCOPE_DIR="${TELESCOPE_DIR:-${PROJECT_ROOT}/runs/operator_pilot_106_20260530/telescope/ska1_low.tm}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-50}"
GPU_RECHECK_SECONDS="${GPU_RECHECK_SECONDS:-300}"

while [[ ! -s "${SOURCE_ROOT}/COMPLETE" ]]; do
  echo "waiting for the 128-frequency OSM extension" >&2
  sleep "${GPU_RECHECK_SECONDS}"
done
while [[ ! -s "${REUSE_CENTRAL_BANK}/COMPLETE" ]]; do
  echo "waiting for the imported central 64-frequency bank" >&2
  sleep "${GPU_RECHECK_SECONDS}"
done

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/shards" "${OUT_DIR}/configs"
mapfile -t ALL_FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; [print(f"{x:.2f}") for x in json.load(open(sys.argv[1]))["frequencies_mhz"]]' \
    "${FREQUENCY_CONFIG}"
)
mapfile -t CENTRAL_FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; [print(f"{x:.2f}") for x in json.load(open(sys.argv[1]))["frequencies_mhz"]]' \
    "${REUSE_CONFIG}"
)
mapfile -t OUTER_FREQUENCIES < <(
  "${PYTHON}" -c \
    'import json,sys; a=json.load(open(sys.argv[1]))["frequencies_mhz"]; c=set(json.load(open(sys.argv[2]))["frequencies_mhz"]); [print(f"{x:.2f}") for x in a if x not in c]' \
    "${FREQUENCY_CONFIG}" "${REUSE_CONFIG}"
)
if [[ "${#ALL_FREQUENCIES[@]}" -ne 128 ||
      "${#CENTRAL_FREQUENCIES[@]}" -ne 64 ||
      "${#OUTER_FREQUENCIES[@]}" -ne 64 ]]; then
  echo "invalid 128-to-64 frequency contract" >&2
  exit 2
fi

for frequency in "${CENTRAL_FREQUENCIES[@]}"; do
  source_shard="${REUSE_CENTRAL_BANK}/shards/freq_${frequency}.npz"
  target_shard="${OUT_DIR}/shards/freq_${frequency}.npz"
  if [[ ! -s "${source_shard}" ]]; then
    echo "missing reusable central bank shard: ${source_shard}" >&2
    exit 1
  fi
  if [[ ! -e "${target_shard}" ]]; then
    ln "${source_shard}" "${target_shard}"
  fi
  for label in fg eor; do
    source_config="${REUSE_CENTRAL_BANK}/configs/sim_${label}_${frequency}.ini"
    target_config="${OUT_DIR}/configs/sim_${label}_${frequency}.ini"
    if [[ ! -s "${source_config}" ]]; then
      echo "missing reusable central config: ${source_config}" >&2
      exit 1
    fi
    if [[ ! -e "${target_config}" && ! -L "${target_config}" ]]; then
      ln -s "${source_config}" "${target_config}"
    fi
  done
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
    echo "no GPU currently satisfies the aperture-PB bank limits" >&2
    sleep "${GPU_RECHECK_SECONDS}"
  fi
done
printf 'selected GPUs: %s\n' "${AVAILABLE_GPUS[*]}" |
  tee "${OUT_DIR}/logs/gpu_selection.log"

export LD_LIBRARY_PATH="${CASA_PREFIX}/lib:${CONDA_ENV}/lib:${OSKAR_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local index
  local frequency
  for ((index = worker_index; index < ${#OUTER_FREQUENCIES[@]}; index += ${#AVAILABLE_GPUS[@]})); do
    frequency="${OUTER_FREQUENCIES[index]}"
    if [[ -s "${OUT_DIR}/shards/freq_${frequency}.npz" ]]; then
      continue
    fi
    echo "worker=${worker_index} gpu=${gpu} frequency=${frequency} start=$(date -Is)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${BUILDER}" \
      --mode shard \
      --out-dir "${OUT_DIR}" \
      --frequency-mhz "${frequency}" \
      --source-root "${SOURCE_ROOT}" \
      --oskar "${OSKAR_PREFIX}/bin/oskar_sim_interferometer" \
      --telescope-dir "${TELESCOPE_DIR}" \
      --station-type aperture_array \
      --grid-size 512 \
      --min-uv-lambda 30 \
      --max-uv-lambda 2500 \
      --reference-frequency-mhz 117.85 \
      --sample-kperp-bins 16 \
      --sample-rows-per-bin 2048 \
      --delete-ms
  done
}

pids=()
for index in "${!AVAILABLE_GPUS[@]}"; do
  run_worker "${index}" "${AVAILABLE_GPUS[index]}" \
    >"${OUT_DIR}/logs/worker_${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more outer-frequency visibility workers failed" >&2
  exit "${status}"
fi

FREQUENCY_CSV="$(IFS=,; echo "${ALL_FREQUENCIES[*]}")"
"${PYTHON}" "${BUILDER}" \
  --mode combine \
  --out-dir "${OUT_DIR}" \
  --frequencies-mhz "${FREQUENCY_CSV}" \
  --min-uv-lambda 30 \
  --max-uv-lambda 2500 \
  --reference-frequency-mhz 117.85 \
  >"${OUT_DIR}/logs/combine.log" 2>&1

printf 'complete %s\n' "$(date -Is)" >"${OUT_DIR}/COMPLETE"
echo "128-frequency aperture-PB visibility bank complete: ${OUT_DIR}"
