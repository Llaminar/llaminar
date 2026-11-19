#!/usr/bin/env bash
# @file install-rocm-lto-linker.sh
# @brief Build and prove the selected SDK's parallel-LTO remark repair without changing the SDK.
# Only pinned LTOBackend/GVN members, the driver and a host diagnostic witness are compiled.
# The fixed executable remains builder-owned; no application loader path changes.
set -euo pipefail
lto_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${lto_script_dir}/rocm-release.env"
# This compiler proves every shipping target even for a single-architecture
# RCCL build. A wider explicit image also needs compiler evidence for its
# additional targets before the dependency can consume the linker.
image_targets="$(bash "${lto_script_dir}/rocm-image-targets.sh" "${ROCM_IMAGE_GPU_TARGETS:-}")"
IFS=';' read -r -a requested <<< "${image_targets}"
compiler_targets="${LLAMINAR_ROCM_GPU_TARGETS}"
for target in "${requested[@]}"; do
    if [[ ";${compiler_targets};" != *";${target};"* ]]; then
        compiler_targets+=";${target}"
    fi
done
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
    --architectures "${compiler_targets}"
