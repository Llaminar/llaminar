#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    repo_root="$1"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

# ROCm hot paths must not hide device scratch behind static process-global
# caches or pointer-value upload caches. These patterns have caused stale
# pointer arrays and graph replay bugs; use IWorkspaceConsumer buffers and
# explicit GPU-side staging kernels instead.
checks=(
    "src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip:::s_blockwise_single_partial|ensure_blockwise_single_partial|s_selfreduce_counters|s_int8_selfreduce_counters|hipMalloc\\(&s_"
    "src/v2/kernels/rocm/gemm/ROCmFloatingPointGemmKernel.cpp:::cached_batch_.*ptrs|uploadBatchedPointersIfChanged"
    "src/v2/kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h:::cached_batch_.*ptrs|uploadBatchedPointersIfChanged"
    "src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp:::runtime_.*pointer_cache|Runtime.*PointerCache|ensureRuntime.*PointerArrays|hipMalloc\\(&entry\\.d_"
    "src/v2/kernels/rocm/moe/ROCmMoEKernel.h:::runtime_.*pointer_cache|Runtime.*PointerCache|ensureRuntime.*PointerArrays"
)

violations=()

for check in "${checks[@]}"; do
    rel_path="${check%%:::*}"
    pattern="${check#*:::}"
    abs_path="${repo_root}/${rel_path}"

    if [[ ! -f "${abs_path}" ]]; then
        violations+=("${rel_path}: file missing from ROCm hygiene scan")
        continue
    fi

    matches="$(grep -n -E "${pattern}" "${abs_path}" || true)"
    if [[ -n "${matches}" ]]; then
        while IFS= read -r match; do
            violations+=("${rel_path}:${match}")
        done <<< "${matches}"
    fi
done

if (( ${#violations[@]} > 0 )); then
    cat >&2 <<'MSG'
ROCm hot-path hidden allocation/cache hygiene failed.

Do not reintroduce static device scratch, pointer-value upload caches, or
runtime pointer-cache entries in ROCm kernels. Use declared IWorkspaceConsumer
buffers and explicit streamful GPU-side staging instead.

Violations:
MSG
    printf '%s\n' "${violations[@]}" >&2
    exit 1
fi

echo "ROCm hot-path hidden allocation/cache hygiene guard passed."
