#!/bin/bash
# Check if CUDA GEMM heuristic needs retraining
# Returns 0 if retraining needed, 1 if up-to-date
# Useful for CI/CD pipelines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_v2"
CSV_FILE="${BUILD_DIR}/cuda_gemm_benchmark_data.csv"
HASH_FILE="${BUILD_DIR}/.cuda_gemm_benchmark_data.csv.sha256"
WEIGHTS_FILE="${PROJECT_ROOT}/src/v2/kernels/cuda/cuda_heuristic_weights.h"

# Check if CSV exists
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ No benchmark data found at $CSV_FILE"
    echo "   Run: ./build_v2/performance/v2_perf_cuda_heuristic_validation"
    exit 2
fi

# Check if weights file exists
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "🆕 No weights file found - training required"
    echo "   Run: ./scripts/train_cuda_heuristic.sh"
    exit 0
fi

# Compute current hash
NEW_HASH=$(sha256sum "$CSV_FILE" | awk '{print $1}')

# Check if hash file exists
if [ ! -f "$HASH_FILE" ]; then
    echo "🆕 No hash file found - training required"
    echo "   CSV hash: ${NEW_HASH:0:16}..."
    echo "   Run: ./scripts/train_cuda_heuristic.sh"
    exit 0
fi

# Compare hashes
OLD_HASH=$(cat "$HASH_FILE")

if [ "$NEW_HASH" == "$OLD_HASH" ]; then
    echo "✅ Heuristic weights up-to-date"
    echo "   CSV hash: ${NEW_HASH:0:16}..."
    echo "   No retraining needed"
    exit 1
else
    echo "🔄 Benchmark data changed - retraining required"
    echo "   Old hash: ${OLD_HASH:0:16}..."
    echo "   New hash: ${NEW_HASH:0:16}..."
    echo "   Run: ./scripts/train_cuda_heuristic.sh"
    exit 0
fi
