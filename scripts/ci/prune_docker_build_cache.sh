#!/usr/bin/env bash
# Bound the persistent local BuildKit cache on the self-hosted runner.
#
# This is intentionally post-job GC, not pre-job cleanup. If the cache is below
# the configured budget, BuildKit keeps it and the next run can reuse layers.
set -euo pipefail

limit="${LLAMINAR_DOCKER_BUILD_CACHE_MAX_SIZE:-256GB}"
enabled="${LLAMINAR_DOCKER_BUILD_CACHE_GC:-1}"

if [[ "$enabled" == "0" ]]; then
    echo "[docker-cache] bounded BuildKit cache GC disabled"
    exit 0
fi

if ! docker buildx version >/dev/null 2>&1; then
    echo "[docker-cache] docker buildx is unavailable; skipping bounded cache GC"
    exit 0
fi

help="$(docker buildx prune --help 2>&1 || true)"
cmd=(docker buildx prune --force)
if grep -q -- "--max-used-space" <<<"$help"; then
    cmd+=(--max-used-space "$limit")
elif grep -q -- "--keep-storage" <<<"$help"; then
    cmd+=(--keep-storage "$limit")
else
    echo "[docker-cache] this docker buildx does not support size-bounded prune; skipping"
    exit 0
fi

echo "[docker-cache] bounding local BuildKit cache to ${limit}; reusable layers remain when under budget"
if ! "${cmd[@]}"; then
    echo "[docker-cache] warning: bounded BuildKit cache GC failed; continuing"
fi
