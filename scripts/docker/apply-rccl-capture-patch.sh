#!/usr/bin/env bash
# Apply canonical capture, live-row, host-storage, worker-lifetime and protocol contracts.
# Validate the complete chain before one source mutation; current sources are a no-op.
set -euo pipefail

rccl_source_dir="$(realpath -- "${1:?RCCL source directory is required}")"
rccl_patch_root="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")/patches")"
rccl_patch_args=()
for rccl_patch_name in rccl-hip-capture-event-wait.patch rccl-device-live-rows.patch rccl-native-host-storage.patch rccl-partial-file-cleanup.patch rccl-created-thread-lifetime.patch rccl-communicator-protocol-defaults.patch rccl-collective-selector-taxonomy.patch rccl-small-allgather-ll-policy.patch; do
    rccl_patch_args+=(--patch "${rccl_patch_root}/${rccl_patch_name}")
done
python3 "$(dirname "${BASH_SOURCE[0]}")/apply-rccl-patch-chain.py" \
    --source "${rccl_source_dir}" \
    --upgrade-patch "${rccl_patch_root}/rccl-partial-file-cleanup.patch" \
    --upgrade-patch "${rccl_patch_root}/rccl-created-thread-lifetime.patch" \
    "${rccl_patch_args[@]}"
