#!/bin/bash
# Download and build OneDNN (oneAPI Deep Neural Network Library)
# https://github.com/uxlfoundation/oneDNN

set -e

ONEDNN_VERSION="v3.6.1"  # Latest stable as of Nov 2025
INSTALL_DIR="/opt/onednn"
BUILD_THREADS=$(nproc)

echo "======================================================================"
echo "OneDNN (oneAPI Deep Neural Network Library) Installation"
echo "======================================================================"
echo "Version: ${ONEDNN_VERSION}"
echo "Install directory: ${INSTALL_DIR}"
echo "Build threads: ${BUILD_THREADS}"
echo ""

# Check if already installed
if [ -f "${INSTALL_DIR}/lib/libdnnl.so" ]; then
    echo "OneDNN already installed at ${INSTALL_DIR}"
    echo "Version check:"
    ls -lh "${INSTALL_DIR}/lib/libdnnl.so"*
    echo ""
    echo "To reinstall, run: sudo rm -rf ${INSTALL_DIR}"
    exit 0
fi

# Create temporary build directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

cd "${TEMP_DIR}"

echo "[1/4] Cloning OneDNN repository..."
git clone --depth 1 --branch ${ONEDNN_VERSION} https://github.com/uxlfoundation/oneDNN.git
cd oneDNN

echo "[2/4] Configuring CMake build..."
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_BUILD_TESTS=OFF \
    -DDNNL_BUILD_EXAMPLES=OFF \
    -DDNNL_ENABLE_PRIMITIVE_CACHE=ON \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native"

echo "[3/4] Building OneDNN (this may take 5-10 minutes)..."
cmake --build build --parallel ${BUILD_THREADS}

echo "[4/4] Installing to ${INSTALL_DIR}..."
sudo cmake --install build

echo ""
echo "======================================================================"
echo "✅ OneDNN installation complete!"
echo "======================================================================"
echo "Library: ${INSTALL_DIR}/lib/libdnnl.so"
echo "Headers: ${INSTALL_DIR}/include/oneapi/dnnl/"
echo ""
echo "To use in Llaminar, reconfigure CMake:"
echo "  cd /workspaces/llaminar"
echo "  cmake -B build_v2 -S src/v2 -DUSE_ONEDNN=ON"
echo "  cmake --build build_v2 --target llaminar2_core --parallel"
echo ""
