#!/usr/bin/env bash
# @file install-rocm-lto-linker.sh
# @brief Build and prove the selected SDK's parallel-LTO remark repair without changing the SDK.
# Only pinned LTOBackend/GVN members, the driver and a host diagnostic witness are compiled.
# The fixed executable remains builder-owned; no application loader path changes.
set -euo pipefail
lto_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${lto_script_dir}/rocm-release.env"
python3 "${lto_script_dir}/llvm-lto-linker/build.py" build \
    --sdk "${LLAMINAR_ROCM_ACTIVE_PREFIX}" \
    --prefix "${LLAMINAR_ROCM_LTO_LINKER_PREFIX}" \
    --revision "${LLAMINAR_ROCM_LLVM_REVISION}" \
    --backend-sha256 "${LLAMINAR_ROCM_LLVM_LTO_BACKEND_SHA256}" \
    --driver-sha256 "${LLAMINAR_ROCM_LLVM_LLD_DRIVER_SHA256}" \
    --gvn-sha256 "${LLAMINAR_ROCM_LLVM_GVN_SOURCE_SHA256}" \
    --gvn-patch-sha256 "${LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256}" \
    --gvn-patch "${lto_script_dir}/patches/llvm-gvn-filtered-remarks.patch" \
    --patch "${lto_script_dir}/patches/llvm-parallel-lto-remarks.patch" \
    --spill-guard "${lto_script_dir}/../build/gpu_spill_guard.py" \
    --architectures "${LLAMINAR_ROCM_GPU_TARGETS}"
