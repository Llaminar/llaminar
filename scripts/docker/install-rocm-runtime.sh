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

download_arch_rocblas() {
    local output="$1"
    local package_url="${ROCBLAS_ARCH_PACKAGE_URL:-}"
    local package_file=""
    local urls=()

    if [[ -n "${package_url}" ]]; then
        urls+=("${package_url}")
    else
        if ! package_url="$(
            curl -fsSLI \
                --connect-timeout 30 \
                --max-time 60 \
                --retry 3 \
                --retry-all-errors \
                "https://archlinux.org/packages/extra/x86_64/rocblas/download" \
            | awk 'tolower($1) == "location:" { loc=$2 } END { gsub(/\r/, "", loc); print loc }'
        )"; then
            package_url=""
        fi
        package_file="${package_url##*/}"

        if [[ "${package_file}" == rocblas-*.pkg.tar.zst ]]; then
            urls+=(
                "https://geo.mirror.pkgbuild.com/extra/os/x86_64/${package_file}"
                "https://mirror.rackspace.com/archlinux/extra/os/x86_64/${package_file}"
                "https://arch.mirror.constant.com/extra/os/x86_64/${package_file}"
                "https://mirrors.edge.kernel.org/archlinux/extra/os/x86_64/${package_file}"
            )
        fi
        urls+=("https://archlinux.org/packages/extra/x86_64/rocblas/download")
    fi

    for package_url in "${urls[@]}"; do
        echo "==> [rocm-runtime] downloading rocBLAS Arch package from ${package_url}"
        if curl -fsSL \
            --connect-timeout 30 \
            --max-time 300 \
            --speed-limit 262144 \
            --speed-time 60 \
            --retry 2 \
            --retry-all-errors \
            --retry-delay 5 \
            -o "${output}" \
            "${package_url}"; then
            return 0
        fi
        rm -f "${output}"
    done

    echo "Failed to download Arch rocBLAS package from all mirrors" >&2
    return 1
}

curl_download \
    "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_DEB_VERSION}_all.deb" \
    /tmp/amdgpu-install.deb
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
download_arch_rocblas /tmp/rocblas-arch.pkg.tar.zst
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
