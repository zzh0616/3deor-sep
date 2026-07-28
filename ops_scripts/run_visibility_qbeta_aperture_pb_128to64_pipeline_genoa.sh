#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
MILAN_HOST="${MILAN_HOST:-zhenghao@202.127.24.58}"
MILAN_SSH_KEY="${MILAN_SSH_KEY:-/home/zhenghao/.ssh/id_ed25519_fg_rmw_119_to_202}"
MILAN_CENTRAL_BANK="${MILAN_CENTRAL_BANK:-/data/zhenghao/fg_rmw/runs/chips_visibility_aperture_pb_64freq_20260725}"
CENTRAL_BANK="${CENTRAL_BANK:-${PROJECT_ROOT}/runs/chips_visibility_aperture_pb_64freq_20260725}"
SOURCE_RUNNER="${SOURCE_RUNNER:-${CODE_ROOT}/ops_scripts/run_cube2_fullsky_isobeam_128freq_osm_extension.sh}"
BANK_RUNNER="${BANK_RUNNER:-${CODE_ROOT}/ops_scripts/run_chips_visibility_aperture_pb_128freq_extension.sh}"
QBETA_RUNNER="${QBETA_RUNNER:-${CODE_ROOT}/ops_scripts/run_visibility_qbeta_aperture_pb_128to64.sh}"
PARTITION_COUNT="${PARTITION_COUNT:-4}"

mkdir -p "${CENTRAL_BANK}"
if [[ ! -s "${CENTRAL_BANK}/COMPLETE" ]]; then
  rsync -a --partial --info=stats2 \
    -e "ssh -i ${MILAN_SSH_KEY} -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=30" \
    "${MILAN_HOST}:${MILAN_CENTRAL_BANK}/" \
    "${CENTRAL_BANK}/"
fi

PROJECT_ROOT="${PROJECT_ROOT}" CODE_ROOT="${CODE_ROOT}" \
  bash "${SOURCE_RUNNER}"
PROJECT_ROOT="${PROJECT_ROOT}" CODE_ROOT="${CODE_ROOT}" \
  REUSE_CENTRAL_BANK="${CENTRAL_BANK}" \
  bash "${BANK_RUNNER}"
PROJECT_ROOT="${PROJECT_ROOT}" CODE_ROOT="${CODE_ROOT}" \
  PARTITION_COUNT="${PARTITION_COUNT}" \
  bash "${QBETA_RUNNER}"

echo "128-to-64 Genoa pipeline complete: $(date -Is)"
