#!/usr/bin/env bash
# Install Llaminar's pinned ROCm Core SDK user-space closure.
#
# MODE=build supplies the compiler and headers for HIP/RCCL/BLAS builds;
# MODE=full additionally supplies the developer debugger/profiler tools;
# MODE=runtime installs only host libraries. No mode installs a kernel driver.
# The compiler modes build the repaired HIP runtime and matching gfx906 BLAS
# kernels. Runtime images copy those exact artifacts from their tested builder;
# they must never graft kernels from another distribution or release.
set -euo pipefail

rocm_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${rocm_script_dir}/rocm-release.env"
MODE="${MODE:-full}"
export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

# The package pin and source pins form one dependency. An environment override
# may not silently install a different SDK underneath the retained source DSOs.
if [[ "${ROCM_VERSION:-${LLAMINAR_ROCM_VERSION}}" != "${LLAMINAR_ROCM_VERSION}" ||
      "${ROCM_DEB_VERSION:-${LLAMINAR_ROCM_DEB_VERSION}}" != "${LLAMINAR_ROCM_DEB_VERSION}" ]]; then
    echo "ROCm package override does not match rocm-release.env" >&2
    exit 1
fi
# hipBLAS's host DSO directly needs rocSOLVER even when inference only calls
# GEMM. Core SDK's blas-host package does not declare that solver dependency;
# include it explicitly in every profile rather than resolving an older SDK's
# same-SONAME library from the system loader cache.
rocm_packages=(base sysdeps runtime math-common blas-host solver-host amdsmi profiler-base)
rocm_system_packages=()
case "${MODE}" in
    full|build)
        rocm_packages+=(llvm llvm-dev runtime-dev blas-dev hipblas-common-dev ccl-dev rccl-dev hipify)
        # HIP ROCclr's headless build still compiles GL interop. Tensile's host
        # library needs MessagePack C++ headers; none may be accidental extras
        # inherited only by a full development image.
        rocm_system_packages+=(libdrm-dev libglvnd-dev libmsgpack-cxx-dev)
        if [[ "${MODE}" == full ]]; then rocm_packages+=(debugger profiler); fi
        ;;
    runtime) ;;
    *) echo "Unknown MODE='${MODE}' (expected full, build, or runtime)" >&2; exit 2 ;;
esac
rocm_package_args=()
for rocm_package in "${rocm_packages[@]}"; do
    rocm_package_args+=("amdrocm-${rocm_package}${LLAMINAR_ROCM_SERIES}=${LLAMINAR_ROCM_DEB_VERSION}")
done

bash "${rocm_script_dir}/configure-rocm-repository.sh"
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --allow-change-held-packages \
    "${rocm_package_args[@]}" "${rocm_system_packages[@]}"

# Core SDK installs below /opt/rocm/core-X.Y rather than replacing /opt/rocm.
# One stable SDK link lets Docker, CMake and shell commands share the active
# compiler/library root without independently spelling a release version.
test "$(<"${LLAMINAR_ROCM_SDK_PREFIX}/.info/version")" = "${LLAMINAR_ROCM_VERSION}"
ln -sfn "${LLAMINAR_ROCM_SDK_PREFIX}" "${LLAMINAR_ROCM_ACTIVE_PREFIX}"
export ROCM_PATH="${LLAMINAR_ROCM_ACTIVE_PREFIX}" HIP_PATH="${LLAMINAR_ROCM_ACTIVE_PREFIX}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ "${MODE}" != runtime ]]; then
    bash "${rocm_script_dir}/install-hip-graph-runtime.sh"
    bash "${rocm_script_dir}/install-rocblas-gfx906.sh"
fi

# The caller owns unrelated packages; never autoremove them during SDK setup.
apt-get clean
rm -rf /var/lib/apt/lists/*
