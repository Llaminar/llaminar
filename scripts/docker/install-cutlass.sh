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

git clone --branch "${CUTLASS_VERSION}" --depth 1 \
    https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
