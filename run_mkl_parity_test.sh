#!/bin/bash
# Quick script to run MKL BF16 parity test against PyTorch
# Prerequisites: PyTorch snapshots must exist (run python/reference/run_reference.py first)

set -e

echo "═══════════════════════════════════════════════════════"
echo "   MKL BF16 Backend - PyTorch Parity Test"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if test executable exists
if [ ! -f "build/test_parity_framework" ]; then
    echo "❌ Test executable not found: build/test_parity_framework"
    echo "   Run: cmake --build build --target test_parity_framework --parallel"
    exit 1
fi

# Check for MKL library
if ! ldconfig -p | grep -q "libmkl_rt.so"; then
    echo "⚠️  WARNING: MKL runtime library not found in ldconfig cache"
    echo "   Ensure Intel MKL is installed and LD_LIBRARY_PATH is set"
    echo ""
fi

# Run the test
echo "Running MKL parity test with 2 MPI ranks..."
echo ""

mpirun -np 2 --bind-to socket --map-by socket \
  ./build/test_parity_framework \
  --gtest_filter="ParityFramework.MKLPrefillVsPyTorch" \
  --gtest_color=yes

echo ""
echo "═══════════════════════════════════════════════════════"
echo "   Test Complete"
echo "═══════════════════════════════════════════════════════"
