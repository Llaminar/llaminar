#!/usr/bin/env bash
# Build and install matching rocBLAS/Tensile device kernels for MI50/MI60.
#
# ROCm Core SDK does not publish a gfx906 kernel pack. This is a real source
# build of the pinned rocBLAS and paired Tensile, not a foreign-code-object
# graft. The installed DSO and library tree are one inseparable runtime unit;
# Docker copies both from the tested builder into each serving image.
# ROCBLAS_INSTALL_PREFIX supports isolated validation without replacing a live
# system library. The caller must retire users before changing its SDK root.
set -euo pipefail

blas_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${blas_script_dir}/rocm-release.env"
blas_sdk_root="${ROCM_PATH:-${LLAMINAR_ROCM_SDK_PREFIX}}"
blas_prefix="${ROCBLAS_INSTALL_PREFIX:-${blas_sdk_root}}"
blas_targets="${ROCBLAS_GPU_TARGETS:-gfx906:xnack-}"
blas_marker="${blas_prefix}/share/llaminar/rocblas-source.txt"
blas_library_tree="${blas_prefix}/lib/rocblas/library"
blas_library="${blas_prefix}/lib/librocblas.so"

if [[ ! -r "${blas_sdk_root}/.info/version" ||
      "$(<"${blas_sdk_root}/.info/version")" != "${LLAMINAR_ROCM_VERSION}" ]]; then
    echo "rocBLAS source build requires the pinned ROCm ${LLAMINAR_ROCM_VERSION} SDK" >&2
    exit 1
fi

# Authenticate source/policy plus the entire installed BLAS artifact closure.
# A copied receipt alone cannot excuse a missing .dat or device code object.
blas_identity="${LLAMINAR_ROCM_VERSION}:${LLAMINAR_ROCM_DEB_VERSION}:${LLAMINAR_ROCM_LIBRARIES_REVISION}:${blas_targets}"
blas_artifact_digest() {
    [[ -f "${blas_library}" && -d "${blas_library_tree}" ]] || return 1
    find "${blas_library_tree}" -type f \( -name '*gfx906*.co' -o -name '*gfx906*.hsaco' \) -print -quit | grep -q . || return 1
    find "${blas_library_tree}" -type f -name '*gfx906*.dat' -print -quit | grep -q . || return 1
    (
        cd -- "${blas_prefix}"
        sha256sum lib/librocblas.so
        find lib/rocblas/library -type f -print0 | sort -z | xargs -0 -r sha256sum
    ) | sha256sum | cut -d ' ' -f 1
}
if [[ -r "${blas_marker}" ]] && blas_digest="$(blas_artifact_digest)" &&
   [[ "$(<"${blas_marker}")" == "${blas_identity}:${blas_digest}" ]]; then
    echo "Matching rocBLAS source kernels already installed: ${blas_library_tree}"
    exit 0
fi
if [[ ! -f "${blas_sdk_root}/lib/llvm/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    echo "rocBLAS source build requires the matching amdrocm-llvm-dev${LLAMINAR_ROCM_SERIES} package" >&2
    exit 1
fi
for blas_dependency in cmake ninja ccache curl python3; do
    command -v "${blas_dependency}" >/dev/null || {
        echo "Missing rocBLAS build dependency: ${blas_dependency}" >&2; exit 1;
    }
done

blas_work="$(mktemp -d /tmp/llaminar-rocblas-build.XXXXXXXX)"
trap 'rm -rf -- "${blas_work}"' EXIT
mkdir -p "${blas_work}/libraries"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/rocm-libraries/tar.gz/${LLAMINAR_ROCM_LIBRARIES_REVISION}" \
    -o "${blas_work}/libraries.tar.gz"
tar -xzf "${blas_work}/libraries.tar.gz" --strip-components=1 \
    -C "${blas_work}/libraries" --wildcards '*/projects/rocblas' '*/shared/tensile'

# Use physical cores for Tensile generation; ordinary Ninja compilation remains
# unrestricted. This avoids expensive generated-library preparation going serial.
blas_physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l)"
cmake -S "${blas_work}/libraries/projects/rocblas" -B "${blas_work}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${blas_sdk_root}/lib/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="${blas_sdk_root}/bin/amdclang++" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DROCM_PATH="${blas_sdk_root}" -DCMAKE_PREFIX_PATH="${blas_sdk_root}" \
    -DCMAKE_INSTALL_PREFIX="${blas_prefix}" -DGPU_TARGETS="${blas_targets}" \
    -DBUILD_WITH_HIPBLASLT=OFF \
    -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF \
    -DROCTX_LIBRARY="${blas_sdk_root}/lib/librocprofiler-sdk-roctx.so" \
    -DROCTRACER_INCLUDE_DIR="${blas_sdk_root}/include" \
    -DTensile_SEPARATE_ARCHITECTURES=ON -DTensile_LAZY_LIBRARY_LOADING=ON \
    -DTensile_TEST_LOCAL_PATH="${blas_work}/libraries/shared/tensile" \
    -DTensile_CPU_THREADS="${blas_physical_cores}"
cmake --build "${blas_work}/build" --parallel
cmake --install "${blas_work}/build"

# All scalar precisions have generated libraries in this full build; it is not
# a tiny FP32-only fixture. Require actual device binaries and dispatch data for
# the mandatory shipping architecture before writing a successful receipt.
if ! find "${blas_library_tree}" -type f \( -name '*gfx906*.co' -o -name '*gfx906*.hsaco' \) -print -quit | grep -q . ||
   ! find "${blas_library_tree}" -type f -name '*gfx906*.dat' -print -quit | grep -q .; then
    echo "rocBLAS build did not install gfx906 kernels and dispatch metadata" >&2
    exit 1
fi
install -d "${blas_prefix}/share/llaminar" "${blas_prefix}/share/licenses/llaminar-rocblas"
install -m 644 "${blas_work}/libraries/projects/rocblas/LICENSE.md" \
    "${blas_prefix}/share/licenses/llaminar-rocblas/LICENSE.md"
blas_digest="$(blas_artifact_digest)"
printf '%s:%s\n' "${blas_identity}" "${blas_digest}" > "${blas_marker}"
echo "Matching rocBLAS source kernels installed: ${blas_library_tree}"
