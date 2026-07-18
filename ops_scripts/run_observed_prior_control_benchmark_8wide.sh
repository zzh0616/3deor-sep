#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/zhenghao/miniconda3/envs/torch/bin/python}"
CODE_DIR="${CODE_DIR:-/data1/zhenghao/fg_rmw/code/3dnet}"
OPS_DIR="${OPS_DIR:-/data1/zhenghao/fg_rmw/ops_scripts}"
DEVICE="${DEVICE:?Set DEVICE after checking current GPU availability, for example cuda:1}"

RUN_ROOT="${RUN_ROOT:-/data1/zhenghao/fg_rmw/runs/observed_prior_control_8wide_20260718}"
INPUT_DIR="${RUN_ROOT}/inputs"
PRIOR_BASIS_DIR="${INPUT_DIR}/gleam_prior_basis"
RESULT_DIR="${RESULT_DIR:-${RUN_ROOT}/result_alpha_m2p55_amp0p30}"

REFERENCE_DIRTY="/data1/zhenghao/fg_rmw/runs/cube2_fullsky_isobeam_512_32freq_20260617/image_natural/eor_117.90-dirty.fits"
EOR_PATTERN="/data1/zhenghao/fg_rmw/runs/cube2_fullsky_isobeam_512_32freq_20260617/image_natural/eor_{freq:.2f}-dirty.fits"
CONFIG="${CODE_DIR}/configs/ps2d_v2_8wide_isobeam_patch.json"
DESIGN="/data1/zhenghao/fg_rmw/runs/cached_pca_joint_eor_separation_quickscreen_20260710_compiled_response/designs/8wide_full_absolute_maska_shared_deg0_block8_scale1e_6_additive_v3.npz"
TILE_CACHE="/dev/shm/zhenghao_fg_rmw/tile_cache_stride4_rank64_8wide_v1"

GLEAM_NPZ="/data1/zhenghao/e2esim/data/cache/catalog_handoff/catalog_handoff_v1_gleam_egc_v2_full_handoff_370e47c848d59811.npz"
GLEAMX_NPZ="/data1/zhenghao/e2esim/data/cache/catalog_handoff/catalog_handoff_v1_gleamx_dr2_full_handoff_9ecdbd2815d4dc79.npz"
HASLAM_FITS="/data1/zhenghao/e2esim/data/cache/galactic/synchrotron/v2_template_haslam408_dsds_Remazeilles2014_ns512_csG_pfI_prCAR_i1_ra0p000_decm27p000_x1800_y1800_pix20p000.fits"
INDEX_FITS="/data1/zhenghao/e2esim/data/cache/galactic/synchrotron/v2_indexmap_synchrotron_specind2_ns512_csG_pfI_prCAR_i1_ra0p000_decm27p000_x1800_y1800_pix20p000.fits"

FREQUENCIES="117.9,118.3,118.7,119.1,119.5,119.9,120.3,120.7"
GLEAM_CSV="${INPUT_DIR}/gleam_egc_v2_field_5mjy.csv"
GLEAMX_CSV="${INPUT_DIR}/gleamx_dr2_field_5mjy.csv"
TRUTH_CATALOG_CSV="${TRUTH_CATALOG_CSV:-${GLEAMX_CSV}}"

mkdir -p "${INPUT_DIR}" "${RESULT_DIR}"

if [[ ! -s "${GLEAM_CSV}" ]]; then
  "${PYTHON}" "${OPS_DIR}/export_observed_handoff_catalog_csv.py" \
    --catalog-npz "${GLEAM_NPZ}" \
    --out-csv "${GLEAM_CSV}" \
    --reference-dirty "${REFERENCE_DIRTY}" \
    --image-size 512 \
    --margin-deg 0.1 \
    --min-flux-jy 0.005 \
    --include-shape-columns
fi

if [[ ! -s "${GLEAMX_CSV}" ]]; then
  "${PYTHON}" "${OPS_DIR}/export_observed_handoff_catalog_csv.py" \
    --catalog-npz "${GLEAMX_NPZ}" \
    --out-csv "${GLEAMX_CSV}" \
    --reference-dirty "${REFERENCE_DIRTY}" \
    --image-size 512 \
    --margin-deg 0.1 \
    --min-flux-jy 0.005 \
    --include-shape-columns
fi

if [[ ! -s "${PRIOR_BASIS_DIR}/manifest.json" ]]; then
  "${PYTHON}" "${OPS_DIR}/build_observed_catalog_source_basis.py" \
    --out-dir "${PRIOR_BASIS_DIR}" \
    --reference-dirty "${REFERENCE_DIRTY}" \
    --catalog-csv "${GLEAM_CSV}" \
    --freqs-mhz "${FREQUENCIES}" \
    --cheb-degree 2 \
    --image-size 512 \
    --catalog-insert-mode gaussian_deconv \
    --top-n-singletons 32 \
    --flux-bins 6 \
    --spatial-grid 4,4 \
    --min-sources-per-bin 1
fi

export FG_RMW_CODE_DIR="${CODE_DIR}"
export FG_RMW_OPERATOR_SCRIPT_DIR="${OPS_DIR}"

NUISANCE_ARGS=()
if [[ "${CATALOG_NUISANCE:-1}" == "1" ]]; then
  NUISANCE_ARGS+=(--catalog-nuisance)
else
  NUISANCE_ARGS+=(--no-catalog-nuisance)
fi
if [[ "${DIFFUSE_NUISANCE:-1}" == "1" ]]; then
  NUISANCE_ARGS+=(--diffuse-nuisance)
else
  NUISANCE_ARGS+=(--no-diffuse-nuisance)
fi

exec "${PYTHON}" "${OPS_DIR}/evaluate_observed_prior_control_likelihood.py" \
  --design-npz "${DESIGN}" \
  --tile-cache-dir "${TILE_CACHE}" \
  --config "${CONFIG}" \
  --reference-dirty "${REFERENCE_DIRTY}" \
  --truth-eor-pattern "${EOR_PATTERN}" \
  --truth-catalog-csv "${TRUTH_CATALOG_CSV}" \
  --prior-source-basis-manifest "${PRIOR_BASIS_DIR}/manifest.json" \
  --diffuse-base-fits "${HASLAM_FITS}" \
  --truth-diffuse-index-fits "${INDEX_FITS}" \
  --truth-diffuse-index-convention alpha \
  --prior-diffuse-spectral-index "${PRIOR_DIFFUSE_ALPHA:--2.55}" \
  --catalog-amplitude-prior-std "${CATALOG_AMPLITUDE_STD:-0.20}" \
  --catalog-slope-prior-std "${CATALOG_SLOPE_STD:-0.30}" \
  --diffuse-amplitude-prior-std "${DIFFUSE_AMPLITUDE_STD:-0.30}" \
  --diffuse-slope-prior-std "${DIFFUSE_SLOPE_STD:-0.30}" \
  --diffuse-grid "${DIFFUSE_GRID:-4,4}" \
  "${NUISANCE_ARGS[@]}" \
  --nuisance-pivot-mhz 119.3 \
  --basis-batch-size "${BASIS_BATCH_SIZE:-4}" \
  --device "${DEVICE}" \
  --no-checkpoint-tiles \
  --out-dir "${RESULT_DIR}"
