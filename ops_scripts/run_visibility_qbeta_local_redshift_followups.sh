#!/usr/bin/env bash
set -euo pipefail

umask 0002

PROJECT_ROOT="${PROJECT_ROOT:-/data1/zhenghao/fg_rmw}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/code/3dnet_128freq_20260726}"
LOCAL_ROOT="${LOCAL_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_screen4_20260728}"
CONFIRM_ROOT="${CONFIRM_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_probe4_confirm_20260728}"
FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-${PROJECT_ROOT}/runs/visibility_qbeta_local_redshift_followups_20260728}"
LOCAL_RUNNER="${LOCAL_RUNNER:-${CODE_ROOT}/ops_scripts/run_visibility_qbeta_local_redshift_mosaic.sh}"
SHARED_RUNNER="${SHARED_RUNNER:-${CODE_ROOT}/ops_scripts/run_visibility_qbeta_shared_local_covariance.sh}"
OUTER_RUNNER="${OUTER_RUNNER:-${CODE_ROOT}/ops_scripts/run_visibility_qbeta_outer_field.sh}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-120}"

mkdir -p "${FOLLOWUP_ROOT}/logs"
printf '%s\n' "$$" >"${FOLLOWUP_ROOT}/RUN.pid"
while [[ ! -s "${LOCAL_ROOT}/COMPLETE" ]]; do
  if [[ -s "${LOCAL_ROOT}/FAILED" ]]; then
    echo "formal local-redshift run failed" >&2
    exit 1
  fi
  sleep "${WAIT_SECONDS}"
done

OUT_DIR="${CONFIRM_ROOT}" \
  WINDOW_INDICES=2 \
  PARTITION_COUNT=4 \
  ROWS_PER_BIN=12 \
  CALIBRATION_REPEATS=4 \
  VALIDATION_REPEATS=1 \
  MIXTURE_REPEATS=16 \
  GPU_MIN_FREE_MIB=78000 \
  bash "${LOCAL_RUNNER}" \
  >"${FOLLOWUP_ROOT}/logs/probe4_central_confirm.log" 2>&1 &
confirm_pid="$!"
printf '%s\n' "${confirm_pid}" >"${FOLLOWUP_ROOT}/probe4_confirm.pid"
if ! wait "${confirm_pid}"; then
  printf 'failed %s\n' "$(date -Is)" \
    >"${FOLLOWUP_ROOT}/PROBE4_CONFIRM_FAILED"
  exit 1
fi

LOCAL_ROOT="${LOCAL_ROOT}" \
  bash "${SHARED_RUNNER}" \
  >"${FOLLOWUP_ROOT}/logs/shared_covariance.log" 2>&1 &
shared_pid="$!"
printf '%s\n' "${shared_pid}" >"${FOLLOWUP_ROOT}/shared_covariance.pid"

# Let the shared job reserve one GPU before selecting three for PB caches.
sleep "${STARTUP_GRACE_SECONDS}"
LOCAL_ROOT="${CONFIRM_ROOT}" \
  MAX_WORKERS=3 \
  GPU_MIN_FREE_MIB=65000 \
  bash "${OUTER_RUNNER}" \
  >"${FOLLOWUP_ROOT}/logs/outer_field.log" 2>&1 &
outer_pid="$!"
printf '%s\n' "${outer_pid}" >"${FOLLOWUP_ROOT}/outer_field.pid"

status=0
if ! wait "${shared_pid}"; then
  status=1
  printf 'failed %s\n' "$(date -Is)" \
    >"${FOLLOWUP_ROOT}/SHARED_COVARIANCE_FAILED"
fi
if ! wait "${outer_pid}"; then
  status=1
  printf 'failed %s\n' "$(date -Is)" \
    >"${FOLLOWUP_ROOT}/OUTER_FIELD_FAILED"
fi
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

printf 'complete %s\n' "$(date -Is)" >"${FOLLOWUP_ROOT}/COMPLETE"
echo "local-redshift follow-ups complete: ${FOLLOWUP_ROOT}"
