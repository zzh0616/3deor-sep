#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}}"
CONFIG="${CONFIG:-${CODE_ROOT}/configs/ps2d_v2_32high_isobeam_patch.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/partial_window_covariance_32high_20260722}"
BANK_DIR="${BANK_DIR:-${OUT_ROOT}/operator_bank}"
RESULT_DIR="${RESULT_DIR:-${OUT_ROOT}/estimate}"
DEBIASED_RESULT_DIR="${DEBIASED_RESULT_DIR:-${OUT_ROOT}/debiased_ps2d}"
RESPONSE_ROOT="${RESPONSE_ROOT:-${PROJECT_ROOT}/runs/direct_dft_response_stride4train_32freq_20260628}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/runs/cached_pca_operator_cheb_fit_32freq_20260629_stride4train}"
TRUTH_ROOT="${TRUTH_ROOT:-${PROJECT_ROOT}/runs/cube2_fullsky_isobeam_512_32freq_20260617}"
PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
DEVICE="${DEVICE:-cuda:3}"

mkdir -p "${OUT_ROOT}/logs"

if [[ ! -s "${BANK_DIR}/manifest.json" ]]; then
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
  "${PYTHON}" "${CODE_ROOT}/ops_scripts/build_partial_window_covariance_bank.py" \
    --config "${CONFIG}" \
    --out-dir "${BANK_DIR}" \
    --fg-cube-k "${PROJECT_ROOT}/data/fg_cube2_512_32asec.fits" \
    --eor-cube-k "${PROJECT_ROOT}/data/eor_cube2_512_32asec.fits" \
    --truth-dirty-pattern "${TRUTH_ROOT}/image_natural/{label}_{freq:.2f}-dirty.fits" \
    --dense-grid-csv-pattern "${RESPONSE_ROOT}/freq_{freq:.2f}/full_stride4train/grid_direct_dft_stride2edge_{freqtag}.csv" \
    --train-grid-csv-pattern "${CACHE_ROOT}/pca_stride4_chebdeg2/subgrids/grid_stride4_{freqtag}.csv" \
    --train-response-pattern "${RESPONSE_ROOT}/freq_{freq:.2f}/full_stride4train/dirty/{label}_signp_pol1/{label}_signp_pol1-dirty.fits" \
    --tile-cache-dir "${CACHE_ROOT}/tile_cache_stride4_rank64_h16_scale32_truefloat64_v1" \
    --device "${DEVICE}" \
    --operator-batch-size "${OPERATOR_BATCH_SIZE:-8}" \
    --fg-draw-count "${FG_DRAW_COUNT:-16}" \
    --eor-probes-per-length "${EOR_PROBES_PER_LENGTH:-8}" \
    > "${OUT_ROOT}/logs/build_bank.log" 2>&1
fi

"${PYTHON}" "${CODE_ROOT}/ops_scripts/estimate_partial_window_covariance_ps2d.py" \
  --config "${CONFIG}" \
  --bank-dir "${BANK_DIR}" \
  --out-dir "${RESULT_DIR}" \
  --precision-levels "${PRECISION_LEVELS:-0.001,0.003,0.01,0.03,0.1}" \
  > "${OUT_ROOT}/logs/estimate.log" 2>&1

"${PYTHON}" "${CODE_ROOT}/ops_scripts/estimate_partial_window_debiased_ps2d.py" \
  --config "${CONFIG}" \
  --bank-dir "${BANK_DIR}" \
  --reference-result-npz "${RESULT_DIR}/result.npz" \
  --out-dir "${DEBIASED_RESULT_DIR}" \
  --precision-levels "${DEBIASED_PRECISION_LEVELS:-0.001,0.003,0.01,0.015,0.02,0.03}" \
  > "${OUT_ROOT}/logs/debiased_ps2d.log" 2>&1
