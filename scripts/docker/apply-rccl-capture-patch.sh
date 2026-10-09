#!/usr/bin/env bash
# Apply canonical capture, live-row payload, host-storage and protocol contracts.
# Each independent patch is idempotent; incompatible upstream source is fatal.
set -euo pipefail

rccl_source_dir="$(realpath -- "${1:?RCCL source directory is required}")"
rccl_patch_root="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")/patches")"
for rccl_patch_name in rccl-hip-capture-event-wait.patch rccl-device-live-rows.patch rccl-native-host-storage.patch rccl-communicator-protocol-defaults.patch; do
    rccl_patch_file="${rccl_patch_root}/${rccl_patch_name}"
    if git -C "${rccl_source_dir}" apply --reverse --check "${rccl_patch_file}" 2>/dev/null; then
        continue
    fi
    git -C "${rccl_source_dir}" apply --check "${rccl_patch_file}"
    git -C "${rccl_source_dir}" apply "${rccl_patch_file}"
done
