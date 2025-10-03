#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/test_tp_gemm_parity"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] TP GEMM parity test binary not found: ${BIN}" >&2
  echo "Build with: cmake --build build --target test_tp_gemm_parity" >&2
  exit 2
fi

PASS=0; FAIL=0; SKIP=0; FAILED=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }

have_ranks(){ local np=$1; if mpirun -np "${np}" /bin/true &>/dev/null; then return 0; fi; return 1; }

run_case(){
  local np=$1; shift; local filter=$1; shift; local tag=$1; shift; local extra_env="$*"
  if ! have_ranks "${np}"; then log "SKIP np=${np} tag=${tag}"; ((SKIP++)); return 0; fi
  log "RUN np=${np} tag=${tag} filter=${filter} ${extra_env}" 
  set +e
  # shellcheck disable=SC2086
  ${extra_env} mpirun -np "${np}" "${BIN}" --gtest_filter="${filter}" 2>&1 | sed -E "s/^/[${tag}] /"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then log "PASS tag=${tag}"; ((PASS++)); else log "FAIL tag=${tag} rc=${rc}"; ((FAIL++)); FAILED+=("${tag}"); fi
}

log "=== TP GEMM Parity Matrix Driver ==="
log "Binary: ${BIN}"

# 2-rank cases (column + row)
run_case 2 TPGemmParity.ColumnEvenSplit_2way col_even_t2
run_case 2 TPGemmParity.RowEvenSplit_2way    row_even_t2

# 3-rank ragged (if available)
run_case 3 TPGemmParity.ColumnRaggedPrime_3way col_ragged_t3
run_case 3 TPGemmParity.RowRaggedPrime_3way    row_ragged_t3

# 4-rank even
run_case 4 TPGemmParity.ColumnEven_4way col_even_t4
run_case 4 TPGemmParity.RowEven_4way    row_even_t4

# Degenerate single (no MPI launch)
log "RUN single-rank degenerate"
if "${BIN}" --gtest_filter=TPGemmParity.DegenerateSingle >/dev/null 2>&1; then log "PASS tag=degenerate_single"; ((PASS++)); else log "FAIL tag=degenerate_single"; ((FAIL++)); FAILED+=(degenerate_single); fi

log "=== SUMMARY ==="
log "PASS=${PASS} FAIL=${FAIL} SKIP=${SKIP}"
if (( FAIL > 0 )); then echo "Failed: ${FAILED[*]}" >&2; exit 1; fi
exit 0
