#!/usr/bin/env bash
# Bound the persistent local BuildKit cache on the self-hosted runner.
#
# This is intentionally post-job GC, not pre-job cleanup. If the cache is below
# the configured budget, BuildKit keeps it and the next run can reuse layers.
set -euo pipefail

# The self-hosted runner owns this cache.  Keep the default aligned with the
# Dockerfile's one shared ccache budget so a missing ARC environment variable
# cannot silently turn a bounded CI cache into an unbounded one.
limit="${LLAMINAR_DOCKER_BUILD_CACHE_MAX_SIZE:-50GB}"
enabled="${LLAMINAR_DOCKER_BUILD_CACHE_GC:-1}"

if [[ "$enabled" == "0" ]]; then
    echo "[docker-cache] bounded BuildKit cache GC disabled"
    exit 0
fi

if ! docker buildx version >/dev/null 2>&1; then
    # CCACHE_MAXSIZE remains the hard compiler-cache cap. Losing optional
    # post-job layer pruning must not turn a successful product gate red.
    echo "[docker-cache] warning: docker buildx is unavailable; skipping best-effort ${limit} layer-cache prune" >&2
    exit 0
fi

help="$(docker buildx prune --help 2>&1 || true)"
cmd=(docker buildx prune --force)
if grep -q -- "--max-used-space" <<<"$help"; then
    cmd+=(--max-used-space "$limit")
elif grep -q -- "--keep-storage" <<<"$help"; then
    cmd+=(--keep-storage "$limit")
else
    echo "[docker-cache] warning: this docker buildx cannot size-bound the layer cache; skipping best-effort prune" >&2
    exit 0
fi

echo "[docker-cache] bounding local BuildKit cache to ${limit}; reusable layers remain when under budget"
if ! "${cmd[@]}"; then
    echo "[docker-cache] warning: best-effort ${limit} layer-cache prune failed; retaining reusable cache" >&2
fi
