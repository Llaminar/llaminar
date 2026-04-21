#!/usr/bin/env bash
# Preflight cleanup for self-hosted runner: clear stale MPI / IPC / shared-memory
# state that can accumulate across jobs and cause the next MPI launch to hang.
#
# Runs on the runner HOST (not inside the build container) at the start of
# every gate job. Best-effort: every command tolerates failure and the script
# always exits 0. Never touches GPUs.
#
# Why sudo: most leaked state on the host is created by the test container
# running as root (Open MPI / NCCL / RCCL inside `docker run --ipc=host`),
# so the resulting files in /dev/shm and SysV IPC arrays are root-owned on
# the host. The runner user typically can't unlink them without sudo. We
# fall back to direct invocation when sudo is unavailable.
#
# NOTE on process kill: when MPI ranks run inside a container, their host
# PIDs live in the container's PID namespace and are reaped at container
# exit, so host-side pkill of "orted" etc. is generally a no-op for the CI
# case. The kill block is still useful for bare-metal debug runs and is
# kept here intentionally.
set -u

# Resolve a sudo prefix once. If sudo isn't available (or asks for a
# password) we fall back to running the command directly — failures are
# logged-and-swallowed below.
SUDO=""
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
  else
    echo "[preflight-clean] sudo present but requires a password; running unprivileged"
  fi
else
  echo "[preflight-clean] sudo not found; running unprivileged"
fi

echo "[preflight-clean] starting MPI/IPC stale-state cleanup (sudo='${SUDO}')"

# ---------------------------------------------------------------------------
# 1. Kill orphaned MPI/launcher/llaminar processes on the host.
#    We resolve PIDs via pgrep first and kill them by PID rather than using
#    `pkill -f <pattern>`, because the pattern would otherwise match the
#    sudo / pkill invocation itself (the pattern text lives in sudo's
#    cmdline) and self-terminate the script.
# ---------------------------------------------------------------------------
PROC_PATTERN='orted|hydra_pmi|hydra_bstrap|pmix_server|^mpirun$|llaminar2|v2_integration_|v2_perf_|v2_unit_'
mapfile -t stragglers < <(pgrep -fa -- "${PROC_PATTERN}" 2>/dev/null || true)
# Filter out our own process tree so we never kill ourselves.
declare -A skip_pids=( [$$]=1 [$PPID]=1 )
victim_pids=()
for line in "${stragglers[@]}"; do
  pid="${line%% *}"
  [[ -n "${skip_pids[$pid]:-}" ]] && continue
  victim_pids+=("$pid")
done
if (( ${#victim_pids[@]} > 0 )); then
  echo "[preflight-clean] killing ${#victim_pids[@]} straggler process(es):"
  printf '  %s\n' "${stragglers[@]}"
  ${SUDO} kill -9 "${victim_pids[@]}" 2>/dev/null || true
  sleep 1
else
  echo "[preflight-clean] no straggler processes found"
fi

# ---------------------------------------------------------------------------
# 2. Wipe shared-memory transports + collective rings under /dev/shm.
#    These are the most common source of "first job after a crash hangs"
#    and are typically root-owned when the container ran as root.
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
    if ${SUDO} rm -rf -- "$path" 2>/dev/null; then
      shm_removed=$((shm_removed + 1))
    fi
  done
done
echo "[preflight-clean] removed ${shm_removed} stale /dev/shm entr(y/ies)"

# ---------------------------------------------------------------------------
# 3. Wipe MPI / PMIx / hwloc session directories under /tmp on the host.
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
    if ${SUDO} rm -rf -- "$path" 2>/dev/null; then
      tmp_removed=$((tmp_removed + 1))
    fi
  done
done
echo "[preflight-clean] removed ${tmp_removed} stale /tmp session entr(y/ies)"

# ---------------------------------------------------------------------------
# 4. Drain SysV shared-memory + semaphore arrays. With sudo we drain ALL
#    leaked arrays (typically root-owned from the container); without sudo
#    we restrict to ones owned by the current uid.
# ---------------------------------------------------------------------------
my_uid="$(id -u)"
filter_uid=""
if [[ -z "${SUDO}" ]]; then
  filter_uid="${my_uid}"
fi

drain_ipc() {
  local flag="$1"   # "-m" (shm) or "-s" (sem)
  local removed=0
  while read -r ipcid owner_uid; do
    if [[ -n "${filter_uid}" && "${owner_uid}" != "${filter_uid}" ]]; then
      continue
    fi
    if ${SUDO} ipcrm "${flag}" "${ipcid}" 2>/dev/null; then
      removed=$((removed + 1))
    fi
  done < <(ipcs "${flag}" 2>/dev/null | awk 'NR>3 && $2 != "" {
             cmd = "id -u " $3 " 2>/dev/null"
             cmd | getline u
             close(cmd)
             if (u != "") print $2, u
           }')
  echo "${removed}"
}

sysv_shm_removed="$(drain_ipc -m)"
sysv_sem_removed="$(drain_ipc -s)"
echo "[preflight-clean] removed ${sysv_shm_removed} SysV shm + ${sysv_sem_removed} sem array(s)"

echo "[preflight-clean] done"
exit 0
