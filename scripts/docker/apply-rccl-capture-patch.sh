#!/usr/bin/env bash
# Apply the validated ROCm graph-capture event dependency fix to pinned RCCL.
# The operation is idempotent and fails if upstream changes invalidate the hunk.
set -euo pipefail

rccl_source_dir="$(realpath -- "${1:?RCCL source directory is required}")"
rccl_patch_file="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")/patches/rccl-hip-capture-event-wait.patch")"

if git -C "${rccl_source_dir}" apply --reverse --check "${rccl_patch_file}" 2>/dev/null; then
    exit 0
fi
git -C "${rccl_source_dir}" apply --check "${rccl_patch_file}"
git -C "${rccl_source_dir}" apply "${rccl_patch_file}"
