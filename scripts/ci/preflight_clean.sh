#!/usr/bin/env bash
# Preflight cleanup for self-hosted runner: clear stale MPI / IPC / shared-memory
# state that can accumulate across jobs and cause the next MPI launch to hang.
#
# This is intended to run on the runner HOST (not inside the build container)
# at the start of every gate job. It is intentionally tolerant of failures —
# every command is best-effort and silenced. The script never touches GPUs.
#
# Coverage:
#   - Orphaned MPI / PMIx / Hydra daemons left by killed jobs
#   - Open MPI / MPICH / PMIx session directories under /tmp
#   - Vader / sm / UCX / NCCL / RCCL shared-memory segments under /dev/shm
#   - SysV shared-memory + semaphore arrays owned by the runner user
#
# Safe to invoke when no jobs are running. Will NOT kill anything user-owned
# outside of known MPI/Llaminar process names.
set -u

echo "[preflight-clean] starting MPI/IPC stale-state cleanup"

# ---------------------------------------------------------------------------
# 1. Kill orphaned MPI/launcher/llaminar processes.
#    pkill returns non-zero when nothing matched — that is the common case,
#    so we swallow the exit code.
# ---------------------------------------------------------------------------
PROC_PATTERN='orted|hydra_pmi|hydra_bstrap|pmix_server|^mpirun$|llaminar2|v2_integration_|v2_perf_|v2_unit_'
mapfile -t stragglers < <(pgrep -fa -- "${PROC_PATTERN}" 2>/dev/null || true)
if (( ${#stragglers[@]} > 0 )); then
  echo "[preflight-clean] killing ${#stragglers[@]} straggler process(es):"
  printf '  %s\n' "${stragglers[@]}"
  pkill -9 -f -- "${PROC_PATTERN}" 2>/dev/null || true
  # Give the kernel a beat to reap.
  sleep 1
else
  echo "[preflight-clean] no straggler processes found"
fi

# ---------------------------------------------------------------------------
# 2. Wipe shared-memory transports + collective rings under /dev/shm.
# ---------------------------------------------------------------------------
shm_globs=(
  '/dev/shm/vader_segment.*'
  '/dev/shm/sm_segment.*'
  '/dev/shm/psm2_*'
  '/dev/shm/psm3_*'
  '/dev/shm/ucx_*'
  '/dev/shm/nccl-*'
  '/dev/shm/rccl-*'
  '/dev/shm/hsa_*'
)
shm_removed=0
for pattern in "${shm_globs[@]}"; do
  for path in $pattern; do
    [[ -e "$path" ]] || continue
    rm -rf -- "$path" 2>/dev/null && shm_removed=$((shm_removed + 1)) || true
  done
done
echo "[preflight-clean] removed ${shm_removed} stale /dev/shm entr(y/ies)"

# ---------------------------------------------------------------------------
# 3. Wipe MPI / PMIx / hwloc session directories under /tmp.
# ---------------------------------------------------------------------------
tmp_globs=(
  '/tmp/ompi.*'
  '/tmp/openmpi-sessions-*'
  '/tmp/pmix-*'
  '/tmp/pmix_dstor_*'
  '/tmp/hwloc-*'
)
tmp_removed=0
for pattern in "${tmp_globs[@]}"; do
  for path in $pattern; do
    [[ -e "$path" ]] || continue
    rm -rf -- "$path" 2>/dev/null && tmp_removed=$((tmp_removed + 1)) || true
  done
done
echo "[preflight-clean] removed ${tmp_removed} stale /tmp session entr(y/ies)"

# ---------------------------------------------------------------------------
# 4. Drain SysV shared-memory + semaphore arrays owned by the current user.
#    Filters by uid so we never touch other users' segments. Older Open MPI
#    components (and a few RCCL paths) still leak these on crash.
# ---------------------------------------------------------------------------
my_uid="$(id -u)"
sysv_shm_removed=0
while read -r shmid owner_uid; do
  [[ "$owner_uid" == "$my_uid" ]] || continue
  ipcrm -m "$shmid" 2>/dev/null && sysv_shm_removed=$((sysv_shm_removed + 1)) || true
done < <(ipcs -m 2>/dev/null | awk 'NR>3 && $2 != "" {
           # Try to resolve owner name -> uid via getent
           cmd = "id -u " $3 " 2>/dev/null"
           cmd | getline u
           close(cmd)
           if (u != "") print $2, u
         }')

sysv_sem_removed=0
while read -r semid owner_uid; do
  [[ "$owner_uid" == "$my_uid" ]] || continue
  ipcrm -s "$semid" 2>/dev/null && sysv_sem_removed=$((sysv_sem_removed + 1)) || true
done < <(ipcs -s 2>/dev/null | awk 'NR>3 && $2 != "" {
           cmd = "id -u " $3 " 2>/dev/null"
           cmd | getline u
           close(cmd)
           if (u != "") print $2, u
         }')

echo "[preflight-clean] removed ${sysv_shm_removed} SysV shm + ${sysv_sem_removed} sem array(s)"

echo "[preflight-clean] done"
exit 0
