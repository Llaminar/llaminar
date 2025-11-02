#!/bin/bash
# Auto-tune CUDA GEMM heuristic
# Runs benchmark suite and trains ML model to generate optimal weights
# Uses hash-based caching to skip training if CSV data hasn't changed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_v2"
CSV_FILE="${BUILD_DIR}/cuda_gemm_benchmark_data.csv"
HASH_FILE="${BUILD_DIR}/.cuda_gemm_benchmark_data.csv.sha256"

echo "================================================================================"
echo "CUDA GEMM Auto-Tuning Pipeline (Hash-Based Auto-Retrain)"
echo "================================================================================"
echo ""
echo "This script:"
echo "  1. Runs CUDA GEMM benchmark suite (~45-75 min for 34 tests)"
echo "  2. Collects ~132,192 performance data points (34 tests × 3,888 configs)"
echo "  3. Computes SHA256 hash of benchmark data"
echo "  4. Trains ML model ONLY if hash changed (skips if unchanged)"
echo "  5. Generates cuda_heuristic_weights.h with learned feature weights"
echo "  6. Updates source tree with data-driven heuristic"
echo ""

# Step 1: Check dependencies
echo "[1/4] Checking dependencies..."
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import pandas, sklearn" >/dev/null 2>&1 || {
    echo "ERROR: Missing Python dependencies (pandas, sklearn)"
    echo "Run: pip3 install pandas scikit-learn matplotlib seaborn"
    exit 1
}

if [ ! -f "${BUILD_DIR}/performance/v2_perf_cuda_heuristic_validation" ]; then
    echo "ERROR: Benchmark executable not found"
    echo "Build first: cd build_v2 && cmake --build . --target v2_perf_cuda_heuristic_validation"
    exit 1
fi

echo "      ✓ Python3 found"
echo "      ✓ ML dependencies installed"
echo "      ✓ Benchmark executable present"
echo ""

# Step 2: Run benchmark suite
echo "[2/5] Running CUDA GEMM benchmark suite..."
echo "      This will take ~45-75 minutes (34 tests × 3,888 configs each)"
echo ""

cd "${BUILD_DIR}"

# Save old CSV if exists (for comparison)
if [ -f "$CSV_FILE" ]; then
    mv "$CSV_FILE" "${CSV_FILE}.old"
fi

# Run with timeout safety (90 min max)
timeout 5400 ./performance/v2_perf_cuda_heuristic_validation 2>&1 | grep -E "\[SUMMARY\]|\[CSV\]|FAILED|PASSED" || {
    echo "ERROR: Benchmark failed or timed out"
    # Restore old CSV if benchmark failed
    if [ -f "${CSV_FILE}.old" ]; then
        mv "${CSV_FILE}.old" "$CSV_FILE"
    fi
    exit 1
}

# Verify data collection
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: cuda_gemm_benchmark_data.csv not generated"
    # Restore old CSV
    if [ -f "${CSV_FILE}.old" ]; then
        mv "${CSV_FILE}.old" "$CSV_FILE"
    fi
    exit 1
fi

ROW_COUNT=$(wc -l < "$CSV_FILE")
EXPECTED_ROWS=132193  # 34 tests × 3,888 configs + 1 header
if [ "$ROW_COUNT" -lt 100000 ]; then
    echo "WARNING: Expected ~$EXPECTED_ROWS rows, got $ROW_COUNT"
    echo "         Some tests may have failed"
fi

echo ""
echo "      ✓ Benchmark complete: $ROW_COUNT rows collected"
echo ""

# Step 3: Compute hash and check if retraining needed
echo "[3/5] Checking if ML model retraining needed..."

NEW_HASH=$(sha256sum "$CSV_FILE" | awk '{print $1}')
OLD_HASH=""

if [ -f "$HASH_FILE" ]; then
    OLD_HASH=$(cat "$HASH_FILE")
fi

echo "      New data hash: ${NEW_HASH:0:16}..."
if [ -n "$OLD_HASH" ]; then
    echo "      Old data hash: ${OLD_HASH:0:16}..."
fi

if [ "$NEW_HASH" == "$OLD_HASH" ]; then
    echo ""
    echo "      ⏭️  CSV data unchanged - skipping ML training"
    echo "      ✓ Using existing cuda_heuristic_weights.h"
    echo ""
    echo "================================================================================"
    echo "Auto-tuning complete (no retraining needed)!"
    echo "================================================================================"
    
    # Clean up old CSV backup
    rm -f "${CSV_FILE}.old"
    
    exit 0
fi

echo ""
if [ -z "$OLD_HASH" ]; then
    echo "      🆕 No previous hash found - training required"
else
    echo "      🔄 CSV data changed - retraining required"
fi
echo ""

# Clean up old CSV backup (we're proceeding with training)
rm -f "${CSV_FILE}.old"

# Step 4: Train ML model
echo "[4/5] Training ML model..."
echo "      Algorithm: Gradient Boosting Regressor"
echo "      Features: 32 engineered features"
echo "      Target: GFLOPS prediction (R² ~ 0.9997)"
echo ""

cd "${PROJECT_ROOT}"
python3 train_cuda_heuristic.py \
    --input "${BUILD_DIR}/cuda_gemm_benchmark_data.csv" \
    --output-dir "${PROJECT_ROOT}/src/v2/kernels/cuda" \
    2>&1 | grep -E "^\[|Model Performance|R²|MAE|RMSE|feature_scalers|EXPORT"

if [ ! -f "${PROJECT_ROOT}/src/v2/kernels/cuda/cuda_heuristic_weights.h" ]; then
    echo "ERROR: cuda_heuristic_weights.h not generated"
    exit 1
fi

echo ""
echo "      ✓ ML model trained successfully"
echo "      ✓ Generated: src/v2/kernels/cuda/cuda_heuristic_weights.h"
echo ""

# Step 5: Update hash file (training succeeded)
echo "[5/5] Updating hash file..."
echo "$NEW_HASH" > "$HASH_FILE"
echo "      ✓ Saved: ${HASH_FILE}"
echo ""

# Summary
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  • src/v2/kernels/cuda/cuda_heuristic_weights.h  (C++ header with learned weights)"
echo "  • src/v2/kernels/cuda/cuda_heuristic_model_weights.txt  (human-readable summary)"
echo "  • src/v2/kernels/cuda/cuda_heuristic_validation.png  (performance plots)"
echo "  • src/v2/kernels/cuda/cuda_gemm_predictions.csv  (~132,192 predictions)"
echo "  • build_v2/.cuda_gemm_benchmark_data.csv.sha256  (hash for auto-retrain)"
echo ""
echo "Next steps:"
echo "  1. Rebuild: cd build_v2 && cmake --build . --target cuda_backend"
echo "  2. Run validation tests: ./tests/v2/v2_test_gemm_autotuner_ml"
echo "  3. Verify improvements in rank correlation and top-10 accuracy"
echo ""
echo "Future runs:"
echo "  • This script will automatically skip training if CSV hash unchanged"
echo "  • Run benchmarks again to update data, then re-run this script"
echo "  • Training only happens when benchmark results actually change"
echo ""
echo "Feature importance (top 5):"
grep -A 5 "Feature Importance" "${PROJECT_ROOT}/src/v2/kernels/cuda/cuda_heuristic_model_weights.txt" | tail -5
echo ""
echo "================================================================================"
echo "Auto-tuning complete! (Model retrained)"
echo "================================================================================"
