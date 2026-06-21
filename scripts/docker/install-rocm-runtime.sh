#!/usr/bin/env bash
# Install the smallest ROCm runtime package set needed by the release image.
#
# Unlike install-rocm.sh MODE=full, this does not call `amdgpu-install
# --usecase=rocm`; that meta-usecase pulls compilers, profiler stacks, MIOpen,
# MIGraphX, OpenCV, and other development/ML packages. Runtime only needs the
# HIP host runtime plus the ROCm BLAS libraries currently linked by Llaminar.
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.1.1}"
ROCM_DEB_VERSION="${ROCM_DEB_VERSION:-7.1.1.70101-1}"
export DEBIAN_FRONTEND=noninteractive

APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

curl -fsSL --retry 5 --retry-delay 5 -o /tmp/amdgpu-install.deb \
    "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_DEB_VERSION}_all.deb"
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y --allow-change-held-packages /tmp/amdgpu-install.deb
rm /tmp/amdgpu-install.deb

# amdgpu-install installs the ROCm/graphics apt sources. Install the direct
# runtime package closure only; hipblas pulls rocblas, rocsolver, hipblaslt,
# roctracer, and HIP runtime dependencies.
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y --no-install-recommends --allow-change-held-packages \
    hipblas \
    rocminfo \
    rocm-smi-lib \
    zstd

# Restore gfx906 (MI50 / Vega 20) rocBLAS Tensile kernels removed in ROCm 7.x.
# The copy is small after architecture pruning and harmless on non-gfx906 hosts.
curl -fsSL --retry 5 --retry-delay 5 -o /tmp/rocblas-arch.pkg.tar.zst \
    "https://archlinux.org/packages/extra/x86_64/rocblas/download"
mkdir -p /tmp/rocblas-arch
tar -I zstd -xf /tmp/rocblas-arch.pkg.tar.zst -C /tmp/rocblas-arch
if compgen -G "/tmp/rocblas-arch/opt/rocm/lib/rocblas/library/*gfx906*" >/dev/null; then
    mkdir -p /opt/rocm/lib/rocblas/library
    cp /tmp/rocblas-arch/opt/rocm/lib/rocblas/library/*gfx906* \
       /opt/rocm/lib/rocblas/library/
fi
rm -rf /tmp/rocblas-arch /tmp/rocblas-arch.pkg.tar.zst

apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
