#!/usr/bin/env bash
# install-cutlass.sh
#
# Clone NVIDIA CUTLASS (header-only Tensor Core templates). Used by the
# CUDA kernels for GEMM and attention. CMake discovers it via CUTLASS_DIR.
set -euo pipefail

CUTLASS_VERSION="${CUTLASS_VERSION:-v4.2.1}"
CUTLASS_DIR="${CUTLASS_DIR:-/opt/cutlass}"

if [[ -d "${CUTLASS_DIR}/include/cutlass" ]]; then
    echo "CUTLASS already present at ${CUTLASS_DIR}, skipping."
    exit 0
fi

APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

install_from_apt() {
    echo "==> [cutlass] installing Ubuntu libcutlass-dev package"
    apt-get "${APT_OPTS[@]}" update
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        libcutlass-dev

    if [[ ! -d /usr/include/cutlass || ! -d /usr/include/cute ]]; then
        echo "libcutlass-dev installed, but expected /usr/include/cutlass and /usr/include/cute" >&2
        return 1
    fi

    mkdir -p "${CUTLASS_DIR}/include"
    ln -sfn /usr/include/cutlass "${CUTLASS_DIR}/include/cutlass"
    ln -sfn /usr/include/cute "${CUTLASS_DIR}/include/cute"
    rm -rf /var/lib/apt/lists/*
    echo "==> [cutlass] headers linked at ${CUTLASS_DIR}/include"
}

if apt-cache show libcutlass-dev >/dev/null 2>&1; then
    install_from_apt
    exit 0
fi

if command -v add-apt-repository >/dev/null 2>&1; then
    add-apt-repository -y multiverse
    apt-get "${APT_OPTS[@]}" update
    if apt-cache show libcutlass-dev >/dev/null 2>&1; then
        install_from_apt
        exit 0
    fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "==> [cutlass] libcutlass-dev unavailable; downloading ${CUTLASS_VERSION} tarball"
curl -fsSL \
    --connect-timeout 30 \
    --max-time 300 \
    --retry 8 \
    --retry-all-errors \
    --retry-delay 5 \
    --retry-max-time 900 \
    -o "${tmpdir}/cutlass.tar.gz" \
    "https://github.com/NVIDIA/cutlass/archive/refs/tags/${CUTLASS_VERSION}.tar.gz"

mkdir -p "${CUTLASS_DIR}"
tar -xzf "${tmpdir}/cutlass.tar.gz" \
    --strip-components=1 \
    -C "${CUTLASS_DIR}"
