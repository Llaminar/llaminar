#!/usr/bin/env bash
# @file transfer_arm64_ccache.sh
# @brief Transfer the existing Dockerfile ccache mount between hosted ARM runners.
# The BuildKit layer exporter omits mutable mounts. Use the workflow's active
# builder and its exact mount ID so the native source builds can reuse objects.
set -euo pipefail

cache_dir="${RUNNER_TEMP:?GitHub runner temporary directory is required}/llaminar-neon-ccache"
transfer_dir=$(mktemp -d "$RUNNER_TEMP/llaminar-ccache-transfer.XXXXXXXX")
trap 'rm -rf -- "$transfer_dir"' EXIT
nonce=$(cat /proc/sys/kernel/random/uuid)
# Each transfer needs to observe the live mount even on a reused builder.
build=(docker buildx build --progress=plain --build-arg "TRANSFER_NONCE=$nonce" --file -)

case "${1:-}" in
    restore)
        # Drop the downloaded archive once BuildKit has consumed it.
        mv -- "$cache_dir" "$transfer_dir/seed"
        "${build[@]}" --build-context "seed=$transfer_dir/seed" --output type=cacheonly "$transfer_dir" <<'DOCKERFILE'
FROM ghcr.io/containerd/busybox@sha256:52f73a0a43a16cf37cd0720c90887ce972fe60ee06a687ee71fb93a7ca601df7
ARG TRANSFER_NONCE
RUN --mount=type=bind,from=seed,target=/seed \
    --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked \
    cp -a /seed/. /root/.ccache/
DOCKERFILE
        ;;
    save)
        # Export into a fresh directory so evicted objects from the restored
        # archive cannot sneak back into the new cache or grow it past its cap.
        "${build[@]}" --output "type=local,dest=$transfer_dir/export" "$transfer_dir" <<'DOCKERFILE'
FROM ghcr.io/containerd/busybox@sha256:52f73a0a43a16cf37cd0720c90887ce972fe60ee06a687ee71fb93a7ca601df7 AS copy
ARG TRANSFER_NONCE
RUN --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked \
    mkdir /export && cp -a /root/.ccache/. /export/
FROM scratch
COPY --from=copy /export/ /
DOCKERFILE
        rm -rf -- "$cache_dir"
        mv -- "$transfer_dir/export" "$cache_dir"
        du -sh -- "$cache_dir"
        ;;
    *)
        echo "Usage: $0 restore|save" >&2
        exit 2
        ;;
esac
