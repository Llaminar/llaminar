#!/usr/bin/env bash
# install-rocm.sh
#
# Install AMD ROCm 7.1.1. The user-space stack is installed inside the
# container; the kernel driver is expected on the host and passed through
# via /dev/kfd + /dev/dri + the render/video groups.
#
# MODE=full     (default) — AMD's full `rocm` usecase.
# MODE=build              — minimal compiler/dev package set for building
#                           Llaminar's ROCm backend and optional RCCL source.
# MODE=runtime            — minimal runtime package set.
#
# gfx906 (MI50) was dropped from ROCm 7.x. We restore support by grafting the
# gfx906 Tensile library files from Arch Linux's rocBLAS package, which
# still ships them.
set -euo pipefail

MODE="${MODE:-full}"
ROCM_VERSION="${ROCM_VERSION:-7.1.1}"
ROCM_DEB_VERSION="${ROCM_DEB_VERSION:-7.1.1.70101-1}"
export DEBIAN_FRONTEND=noninteractive

APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

curl_download() {
    local url="$1"
    local output="$2"

    curl -fsSL \
        --connect-timeout 30 \
        --max-time 300 \
        --retry 8 \
        --retry-all-errors \
        --retry-delay 5 \
        --retry-max-time 900 \
        -o "${output}" \
        "${url}"
}

# amdgpu-install is the canonical installer for ROCm on Ubuntu.
curl_download \
    "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_DEB_VERSION}_all.deb" \
    /tmp/amdgpu-install.deb
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y --allow-change-held-packages /tmp/amdgpu-install.deb
rm /tmp/amdgpu-install.deb

case "${MODE}" in
    full)
        # --no-dkms: the kernel driver comes from the host; don't rebuild it
        # inside the container.
        amdgpu-install --usecase=rocm --no-dkms -y
        ;;
    build)
        # Direct package closure needed by CMake:
        # - hipcc/amdclang + HIP headers/runtime
        # - hipBLAS/hipBLASLt headers and shared libraries
        # - ROCm CMake/device libs for HIP and RCCL source builds
        # - RCCL dev package for headers/system fallback when source build is off
        apt-get "${APT_OPTS[@]}" update
        apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --allow-change-held-packages \
            rocm-llvm \
            rocm-cmake \
            rocm-device-libs \
            hipcc \
            hipify-clang \
            hip-dev \
            hsa-rocr-dev \
            libdrm-dev \
            hipblas-dev \
            hipblaslt-dev \
            rocblas-dev \
            rccl-dev \
            rocminfo \
            rocm-smi-lib
        ;;
    runtime)
        apt-get "${APT_OPTS[@]}" update
        apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --allow-change-held-packages \
            hipblas \
            rocminfo \
            rocm-smi-lib
        ;;
    *)
        echo "Unknown MODE='${MODE}' (expected full, build, or runtime)" >&2
        exit 2
        ;;
esac

# Restore gfx906 (MI50 / Vega 20) Tensile kernels removed in ROCm 7.x.
apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --allow-change-held-packages zstd
curl_download \
    "https://archlinux.org/packages/extra/x86_64/rocblas/download" \
    /tmp/rocblas-arch.pkg.tar.zst
mkdir -p /tmp/rocblas-arch
tar -I zstd -xf /tmp/rocblas-arch.pkg.tar.zst -C /tmp/rocblas-arch
if compgen -G "/tmp/rocblas-arch/opt/rocm/lib/rocblas/library/*gfx906*" >/dev/null; then
    cp /tmp/rocblas-arch/opt/rocm/lib/rocblas/library/*gfx906* \
       /opt/rocm/lib/rocblas/library/
fi
rm -rf /tmp/rocblas-arch /tmp/rocblas-arch.pkg.tar.zst

apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
