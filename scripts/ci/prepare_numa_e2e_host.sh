#!/usr/bin/env bash
# Prepare the self-hosted runner host for NUMA-sensitive container E2E tests.
#
# Llaminar treats failed NUMA binding as a hard failure. After large Docker
# builds, the host can have enough free memory overall while still being
# fragmented enough that strict page migration is unreliable. Compact memory
# immediately before E2E so strict mbind failures remain a useful canary for
# real placement or permission problems instead of stale build-side pressure.
set -u

SUDO=""
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    SUDO="sudo -n"
  else
    echo "[numa-prep] sudo present but requires a password; running unprivileged"
  fi
else
  echo "[numa-prep] sudo not found; running unprivileged"
fi

echo "[numa-prep] cgroup cpuset:"
for f in /sys/fs/cgroup/cpuset.cpus.effective \
         /sys/fs/cgroup/cpuset.mems.effective \
         /sys/fs/cgroup/cpuset.cpus \
         /sys/fs/cgroup/cpuset.mems; do
  [[ -e "$f" ]] && printf '[numa-prep]   %s=%s\n' "$f" "$(cat "$f")"
done

echo "[numa-prep] node memory before compaction:"
for f in /sys/devices/system/node/node*/meminfo; do
  [[ -e "$f" ]] || continue
  awk 'NR <= 4 { print "[numa-prep]   " $0 }' "$f"
done

if [[ -e /proc/sys/vm/compact_memory ]]; then
  echo "[numa-prep] requesting host memory compaction"
  compact_cmd=(sh -c 'echo 1 > /proc/sys/vm/compact_memory')
  if command -v timeout >/dev/null 2>&1; then
    compact_cmd=(timeout 120s "${compact_cmd[@]}")
  fi
  if [[ -n "$SUDO" ]]; then
    if ! ${SUDO} "${compact_cmd[@]}"; then
      echo "[numa-prep] warning: memory compaction request failed"
    fi
  elif [[ -w /proc/sys/vm/compact_memory ]]; then
    if ! "${compact_cmd[@]}"; then
      echo "[numa-prep] warning: memory compaction request failed"
    fi
  else
    echo "[numa-prep] warning: cannot compact memory without writable /proc/sys/vm/compact_memory"
  fi
else
  echo "[numa-prep] warning: /proc/sys/vm/compact_memory is unavailable"
fi

echo "[numa-prep] node memory after compaction:"
for f in /sys/devices/system/node/node*/meminfo; do
  [[ -e "$f" ]] || continue
  awk 'NR <= 4 { print "[numa-prep]   " $0 }' "$f"
done

echo "[numa-prep] done"
