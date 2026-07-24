#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_64freq_20260714}"
REUSE_HIGH_BANK="${REUSE_HIGH_BANK:-${PROJECT_ROOT}/runs/chips_visibility_32freq_grid512_20260723}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/chips_visibility_64freq_grid512_20260725}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/obs-eor-core-py312-casa380/bin/python}"
CASA_LIB="${CASA_LIB:-/data1/zhenghao/local/radio-20260517-casa380-py312/lib}"
OSKAR="${OSKAR:-/data1/zhenghao/local/radio-202605-oskar212-cuda-casa380/bin/oskar_sim_interferometer}"
TELESCOPE_DIR="${TELESCOPE_DIR:-/data/zhenghao/fg_rmw/runs/operator_pilot_106_20260530/telescope/ska1_low.tm}"
BUILDER="${BUILDER:-${CODE_ROOT}/ops_scripts/build_chips_visibility_bank.py}"
MAX_WORKERS="${MAX_WORKERS:-3}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"

LOW_FREQUENCIES=(
  114.70 114.80 114.90 115.00 115.10 115.20 115.30 115.40
  115.50 115.60 115.70 115.80 115.90 116.00 116.10 116.20
  116.30 116.40 116.50 116.60 116.70 116.80 116.90 117.00
  117.10 117.20 117.30 117.40 117.50 117.60 117.70 117.80
)
HIGH_FREQUENCIES=(
  117.90 118.00 118.10 118.20 118.30 118.40 118.50 118.60
  118.70 118.80 118.90 119.00 119.10 119.20 119.30 119.40
  119.50 119.60 119.70 119.80 119.90 120.00 120.10 120.20
  120.30 120.40 120.50 120.60 120.70 120.80 120.90 121.00
)
ALL_FREQUENCIES=("${LOW_FREQUENCIES[@]}" "${HIGH_FREQUENCIES[@]}")
FREQUENCY_CSV="$(IFS=,; echo "${ALL_FREQUENCIES[*]}")"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/shards"
for frequency in "${HIGH_FREQUENCIES[@]}"; do
  source="${REUSE_HIGH_BANK}/shards/freq_${frequency}.npz"
  target="${OUT_DIR}/shards/freq_${frequency}.npz"
  if [[ ! -s "${source}" ]]; then
    echo "Missing reusable high-band shard: ${source}" >&2
    exit 1
  fi
  if [[ ! -e "${target}" ]]; then
    ln "${source}" "${target}"
  fi
done

export LD_LIBRARY_PATH="${CASA_LIB}:${LD_LIBRARY_PATH:-}"
if [[ -n "${GPUS:-}" ]]; then
  read -r -a AVAILABLE_GPUS <<<"${GPUS}"
else
  mapfile -t AVAILABLE_GPUS < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
    awk -F, \
      -v memory_limit="${GPU_MEMORY_LIMIT_MIB}" \
      -v util_limit="${GPU_UTIL_LIMIT_PERCENT}" \
      '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (($2 + 0) < memory_limit && ($3 + 0) < util_limit) print $1;
      }'
  )
fi
if [[ "${#AVAILABLE_GPUS[@]}" -eq 0 ]]; then
  echo "No GPU is currently below the launch limits." >&2
  exit 1
fi
if [[ "${#AVAILABLE_GPUS[@]}" -gt "${MAX_WORKERS}" ]]; then
  AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:0:${MAX_WORKERS}}")
fi
printf 'selected GPUs at launch: %s\n' "${AVAILABLE_GPUS[*]}" |
  tee "${OUT_DIR}/logs/gpu_selection.log"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader >>"${OUT_DIR}/logs/gpu_selection.log"

worker() {
  local worker_index="$1"
  local gpu="$2"
  local frequency
  for ((
    index = worker_index;
    index < ${#LOW_FREQUENCIES[@]};
    index += ${#AVAILABLE_GPUS[@]}
  )); do
    frequency="${LOW_FREQUENCIES[index]}"
    echo "[worker ${worker_index}] GPU ${gpu}, ${frequency} MHz, $(date -Is)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${BUILDER}" \
      --mode shard \
      --out-dir "${OUT_DIR}" \
      --frequency-mhz "${frequency}" \
      --source-root "${SOURCE_ROOT}" \
      --oskar "${OSKAR}" \
      --telescope-dir "${TELESCOPE_DIR}" \
      --grid-size 512 \
      --min-uv-lambda 30 \
      --max-uv-lambda 2500 \
      --reference-frequency-mhz 119.45 \
      --chunk-rows 262144 \
      --sample-kperp-bins 16 \
      --sample-rows-per-bin 2048 \
      --delete-ms
  done
}

pids=()
for index in "${!AVAILABLE_GPUS[@]}"; do
  worker "${index}" "${AVAILABLE_GPUS[index]}" \
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
  echo "At least one low-frequency visibility worker failed." >&2
  exit "${status}"
fi

"${PYTHON}" "${BUILDER}" \
  --mode combine \
  --out-dir "${OUT_DIR}" \
  --frequencies-mhz "${FREQUENCY_CSV}" \
  --min-uv-lambda 30 \
  --max-uv-lambda 2500 \
  --reference-frequency-mhz 119.45

printf 'complete %s\n' "$(date -Is)" >"${OUT_DIR}/COMPLETE"
echo "64-frequency visibility bank complete: ${OUT_DIR}"
