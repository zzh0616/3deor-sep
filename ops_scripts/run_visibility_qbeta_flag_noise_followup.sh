#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_noise_systematics_20260817}"
FLAG_ROOT="${FLAG_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_flag_stress_20260817}"
BASE_RUN="${BASE_RUN:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
BANK_DIR="${BANK_DIR:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_128freq_20260726}"
SEFD_H5="${SEFD_H5:-${PROJECT_ROOT}/runs/visibility_qbeta_noise_systematics_20260817/inputs/ska_station_sensitivity_AAVS2.h5}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
EVALUATOR="${CODE_ROOT}/ops_scripts/evaluate_visibility_qbeta_noise_systematics.py"
CONFIG="${BASE_RUN}/configs/local_04_116p3_119p4mhz_analysis.json"
POLL_SECONDS="${POLL_SECONDS:-600}"
PATTERNS=(random5 random10 cluster6)

while [[ ! -f "${FLAG_ROOT}/COMPLETE" ]]; do
  sleep "${POLL_SECONDS}"
done

for pattern in "${PATTERNS[@]}"; do
  root="${FLAG_ROOT}/${pattern}"
  out="${root}/thermal"
  if [[ -s "${out}/summary.json" && -s "${out}/products.npz" ]]; then
    continue
  fi
  mkdir -p "${out}"
  /usr/bin/time -v "${PYTHON}" "${EVALUATOR}" \
    --combined-npz "${root}/combined/result.npz" \
    --combined-json "${root}/combined/result.json" \
    --coarse-npz "${root}/coarse/products.npz" \
    --bank-dir "${BANK_DIR}" \
    --config "${CONFIG}" \
    --sefd-h5 "${SEFD_H5}" \
    --out-dir "${out}" \
    --realizations 512 \
    --gain-realizations 8 \
    --chunk-size 16 \
    --integration-hours 1000 \
    --gain-rms 0.0001 \
    --gain-profile smooth \
    --target-significance 10 \
    --target-significance 25 \
    >"${FLAG_ROOT}/logs/${pattern}_thermal.log" 2>&1
done

touch "${FLAG_ROOT}/THERMAL_COMPLETE"
