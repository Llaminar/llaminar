#!/usr/bin/env bash
# install-rocm.sh
#
# Install AMD ROCm 7.1.1. The user-space stack is installed inside the
# container; the kernel driver is expected on the host and passed through
# via /dev/kfd + /dev/dri + the render/video groups.
#
# MODE=full     (default) — rocm usecase (HIP compiler, rocBLAS dev, RCCL).
# MODE=runtime             — rocm-hip-runtime usecase + rocBLAS + RCCL runtime.
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

# amdgpu-install is the canonical installer for ROCm on Ubuntu.
curl -fsSL --retry 5 --retry-delay 5 -o /tmp/amdgpu-install.deb \
    "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_DEB_VERSION}_all.deb"
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y /tmp/amdgpu-install.deb
rm /tmp/amdgpu-install.deb

if [[ "${MODE}" == "full" ]]; then
    USECASE="rocm"
else
    USECASE="rocm"   # ROCm doesn't ship a split runtime usecase we can rely
                     # on; the "rocm" usecase installs both compiler and
                     # runtime. We trim at image-flatten time, not here.
fi

# --no-dkms: the kernel driver comes from the host; don't rebuild it inside
# the container.
amdgpu-install --usecase="${USECASE}" --no-dkms -y

# Restore gfx906 (MI50 / Vega 20) Tensile kernels removed in ROCm 7.x.
apt-get "${APT_OPTS[@]}" install -y --no-install-recommends zstd
curl -fsSL --retry 5 --retry-delay 5 -o /tmp/rocblas-arch.pkg.tar.zst \
    "https://archlinux.org/packages/extra/x86_64/rocblas/download"
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
