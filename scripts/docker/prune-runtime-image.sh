#!/usr/bin/env bash
# Remove build/profiling/development payloads from the release runtime image.
#
# This script is intentionally run in the same Dockerfile RUN layer as the
# heavyweight runtime installs. Deleting these files in a later layer would
# hide them from the final filesystem but would not reduce the image size.
set -euo pipefail

ROCM_ROOT="${ROCM_ROOT:-/opt/rocm-7.1.1}"
ROCM_RUNTIME_GPU_TARGETS="${ROCM_RUNTIME_GPU_TARGETS:-}"

target_is_enabled() {
    local target="$1"
    local normalized
    normalized="${ROCM_RUNTIME_GPU_TARGETS//,/ }"
    normalized="${normalized//;/ }"
    local enabled
    for enabled in ${normalized}; do
        [[ "${target}" == "${enabled}" ]] && return 0
    done
    return 1
}

prune_gfx_library_dir() {
    local dir="$1"
    [[ -n "${ROCM_RUNTIME_GPU_TARGETS}" && -d "${dir}" ]] || return 0

    local file base target
    while IFS= read -r -d '' file; do
        base="$(basename "${file}")"
        if [[ "${base}" =~ (gfx[0-9a-z]+) ]]; then
            target="${BASH_REMATCH[1]}"
            if ! target_is_enabled "${target}"; then
                rm -f "${file}"
            fi
        fi
    done < <(find "${dir}" -maxdepth 1 -type f -print0)
}

if [[ -d "${ROCM_ROOT}" ]]; then
    echo "==> [prune] ROCm before: $(du -sh "${ROCM_ROOT}" | cut -f1)"

    # Static archives and compiler objects are only useful for building.
    find "${ROCM_ROOT}" -type f \( -name '*.a' -o -name '*.o' \) -delete || true

    # Runtime keeps shared libraries and command-line diagnostics such as
    # rocminfo/rocm-smi, but drops headers, CMake/pkg-config metadata, docs,
    # compiler/debugger/profiler tools, and ML/vision stacks Llaminar does not
    # link against.
    rm -rf \
        "${ROCM_ROOT}/include" \
        "${ROCM_ROOT}/lib/cmake" \
        "${ROCM_ROOT}/lib/pkgconfig" \
        "${ROCM_ROOT}/lib/llvm/bin" \
        "${ROCM_ROOT}/lib/llvm/lib-debug" \
        "${ROCM_ROOT}/lib/migraphx" \
        "${ROCM_ROOT}/lib/python3.12" \
        "${ROCM_ROOT}/lib/rocfft" \
        "${ROCM_ROOT}/lib/rocprofiler" \
        "${ROCM_ROOT}/lib/rocprofiler-sdk" \
        "${ROCM_ROOT}/lib/rocprofiler-systems" \
        "${ROCM_ROOT}/share/doc" \
        "${ROCM_ROOT}/share/html" \
        "${ROCM_ROOT}/share/info" \
        "${ROCM_ROOT}/share/man" \
        "${ROCM_ROOT}/share/miopen" \
        "${ROCM_ROOT}/share/mivisionx" || true

    rm -f \
        "${ROCM_ROOT}/bin/amdclang"* \
        "${ROCM_ROOT}/bin/clang"* \
        "${ROCM_ROOT}/bin/flang"* \
        "${ROCM_ROOT}/bin/hipcc" \
        "${ROCM_ROOT}/bin/hipconfig" \
        "${ROCM_ROOT}/bin/hipify"* \
        "${ROCM_ROOT}/bin/lld"* \
        "${ROCM_ROOT}/bin/llc" \
        "${ROCM_ROOT}/bin/llvm"* \
        "${ROCM_ROOT}/bin/opt" \
        "${ROCM_ROOT}/bin/rocgdb"* \
        "${ROCM_ROOT}/bin/rocprof"* || true

    rm -f \
        "${ROCM_ROOT}/lib/libMIOpen"* \
        "${ROCM_ROOT}/lib/libhiptensor"* \
        "${ROCM_ROOT}/lib/libmigraphx"* \
        "${ROCM_ROOT}/lib/libopenvx"* \
        "${ROCM_ROOT}/lib/librocalution"* \
        "${ROCM_ROOT}/lib/librocfft"* \
        "${ROCM_ROOT}/lib/librocrand"* \
        "${ROCM_ROOT}/lib/librocsparse"* \
        "${ROCM_ROOT}/lib/librpp"* \
        "${ROCM_ROOT}/lib/libvx"* || true

    if [[ -n "${ROCM_RUNTIME_GPU_TARGETS}" ]]; then
        echo "==> [prune] Keeping ROCm GPU targets: ${ROCM_RUNTIME_GPU_TARGETS}"
        prune_gfx_library_dir "${ROCM_ROOT}/lib/hipblaslt/library"
        prune_gfx_library_dir "${ROCM_ROOT}/lib/rocblas/library"
    fi

    echo "==> [prune] ROCm after:  $(du -sh "${ROCM_ROOT}" | cut -f1)"
fi

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ldconfig
