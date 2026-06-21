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

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

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
