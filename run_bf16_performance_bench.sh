#!/bin/bash
# BF16 GEMM Performance Benchmark
# Compares BF16 (with CPU-aware automatic fallback) vs pure FP32 baseline
# Tests: decode operations, batch processing, and prefill scenarios

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_release"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   BF16 GEMM Performance Benchmark (Phase 4)               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect system topology
SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
THREADS_PER_CORE=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
TOTAL_CORES=$((SOCKETS * CORES_PER_SOCKET))

# CRITICAL: Check if this is a Release build
echo -e "${GREEN}Build Configuration:${NC}"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Release build directory not found at $BUILD_DIR${NC}"
    echo "Please build the Release version first:"
    echo "  cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build_release --parallel"
    exit 1
fi

if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE:STRING" ${BUILD_DIR}/CMakeCache.txt | cut -d= -f2)
    echo "  Build type: $BUILD_TYPE"
    if [ "$BUILD_TYPE" != "Release" ]; then
        echo -e "${RED}  ⚠ WARNING: Performance benchmarks require Release build!${NC}"
        echo -e "${RED}  ⚠ Debug builds are 5-10x slower - results will be misleading${NC}"
        echo -e "${RED}  ⚠ Reconfigure with: cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release${NC}"
        exit 1
    fi
else
    echo "  Build type: Unknown (CMakeCache.txt not found)"
fi
echo ""

echo -e "${GREEN}System Topology:${NC}"
echo "  Sockets: $SOCKETS"
echo "  Cores per socket: $CORES_PER_SOCKET"  
echo "  Threads per core: $THREADS_PER_CORE"
echo "  Total physical cores: $TOTAL_CORES"
echo ""

# Check CPU features
echo -e "${GREEN}CPU Feature Detection:${NC}"
if [ -f "${BUILD_DIR}/test_cpu_features" ]; then
    ${BUILD_DIR}/test_cpu_features 2>&1 | grep -E "CPU Features|Can use native" | head -2
else
    echo -e "${YELLOW}  Warning: test_cpu_features not found${NC}"
fi
echo ""

# Set optimal OpenMP/MPI configuration
export OMP_NUM_THREADS=$CORES_PER_SOCKET
export OPENBLAS_NUM_THREADS=$CORES_PER_SOCKET
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

# MPI settings
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

echo -e "${GREEN}Runtime Configuration:${NC}"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MPI Processes: $SOCKETS"
echo ""

# Results directory
RESULTS_DIR="${SCRIPT_DIR}/bf16_benchmark_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/bf16_perf_${TIMESTAMP}.txt"

echo -e "${GREEN}Results will be saved to:${NC} ${RESULT_FILE}"
echo ""

# Helper function to run benchmark
run_benchmark() {
    local test_name=$1
    local executable=$2
    local bf16_flag=$3
    local description=$4
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test: ${test_name}${NC}"
    echo -e "${CYAN}Mode: ${description}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Set BF16 flag
    export LLAMINAR_QUANT_BF16_GEMM=$bf16_flag
    
    # Write to results file
    echo "========================================" >> "$RESULT_FILE"
    echo "Test: ${test_name}" >> "$RESULT_FILE"
    echo "Mode: ${description}" >> "$RESULT_FILE"
    echo "BF16 Flag: ${bf16_flag}" >> "$RESULT_FILE"
    echo "Timestamp: $(date)" >> "$RESULT_FILE"
    echo "----------------------------------------" >> "$RESULT_FILE"
    
    # Run the test
    if mpirun -np $SOCKETS --bind-to socket --map-by socket \
        ${executable} 2>&1 | tee -a "$RESULT_FILE"; then
        echo -e "${GREEN}✓ Test completed successfully${NC}"
    else
        echo -e "${RED}✗ Test failed${NC}"
    fi
    
    echo "" | tee -a "$RESULT_FILE"
    echo ""
}

# ============================================================================
# Test 1: BF16 Conversion Precision (Baseline Validation)
# ============================================================================
if [ -f "${BUILD_DIR}/test_bf16_conversion" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Test Suite 1: BF16 Conversion Precision                  ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark \
        "BF16 Conversion" \
        "${BUILD_DIR}/test_bf16_conversion" \
        "1" \
        "Validate FP32↔BF16 conversion (baseline)"
else
    echo -e "${YELLOW}⊘ Skipping test_bf16_conversion (not found)${NC}"
    echo ""
fi

# ============================================================================
# Test 2: BF16 GEMM Numerical Parity
# ============================================================================
if [ -f "${BUILD_DIR}/test_bf16_gemm_parity" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Test Suite 2: BF16 GEMM Numerical Parity                 ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark \
        "BF16 GEMM Parity (BF16 Mode)" \
        "${BUILD_DIR}/test_bf16_gemm_parity" \
        "1" \
        "BF16 GEMM with automatic CPU fallback - 64×896×896 matrix"
    
    run_benchmark \
        "BF16 GEMM Parity (FP32 Baseline)" \
        "${BUILD_DIR}/test_bf16_gemm_parity" \
        "0" \
        "Pure FP32 GEMM (no BF16) - 64×896×896 matrix"
else
    echo -e "${YELLOW}⊘ Skipping test_bf16_gemm_parity (not found)${NC}"
    echo ""
fi

# ============================================================================
# Test 3: Batch Performance Comparison
# ============================================================================
if [ -f "${BUILD_DIR}/test_batch_performance" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Test Suite 3: Batch Processing Performance               ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark \
        "Batch Performance (BF16 Mode)" \
        "${BUILD_DIR}/test_batch_performance" \
        "1" \
        "BF16 batch processing with automatic CPU fallback"
    
    run_benchmark \
        "Batch Performance (FP32 Baseline)" \
        "${BUILD_DIR}/test_batch_performance" \
        "0" \
        "Pure FP32 batch processing"
else
    echo -e "${YELLOW}⊘ Skipping test_batch_performance (not found)${NC}"
    echo ""
fi

# ============================================================================
# Test 4: Prefill Performance Benchmark
# ============================================================================
if [ -f "${BUILD_DIR}/test_prefill_performance_bench" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Test Suite 4: Prefill Performance                        ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark \
        "Prefill Performance (BF16 Mode)" \
        "${BUILD_DIR}/test_prefill_performance_bench" \
        "1" \
        "BF16 prefill operations with automatic CPU fallback"
    
    run_benchmark \
        "Prefill Performance (FP32 Baseline)" \
        "${BUILD_DIR}/test_prefill_performance_bench" \
        "0" \
        "Pure FP32 prefill operations"
else
    echo -e "${YELLOW}⊘ Skipping test_prefill_performance_bench (not found)${NC}"
    echo ""
fi

# ============================================================================
# Summary and Analysis
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Benchmark Complete - Generating Summary                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Results saved to:${NC} ${RESULT_FILE}"
echo ""

# Extract key metrics if available
if [ -f "$RESULT_FILE" ]; then
    echo -e "${CYAN}Key Performance Metrics:${NC}"
    echo ""
    
    # Look for timing data
    if grep -q "ms\|tok/s\|GFLOPS" "$RESULT_FILE" 2>/dev/null; then
        echo -e "${YELLOW}Timing Summary:${NC}"
        grep -E "(Time:|Throughput:|GFLOPS:|ms)" "$RESULT_FILE" | head -20
        echo ""
    fi
    
    # Look for relative L2 errors
    if grep -q "Relative L2" "$RESULT_FILE" 2>/dev/null; then
        echo -e "${YELLOW}Numerical Accuracy:${NC}"
        grep "Relative L2" "$RESULT_FILE"
        echo ""
    fi
    
    # Count test passes/failures
    PASSED=$(grep -c "PASSED" "$RESULT_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "FAILED" "$RESULT_FILE" 2>/dev/null || echo "0")
    
    echo -e "${CYAN}Test Results:${NC}"
    echo -e "  ${GREEN}Passed: ${PASSED}${NC}"
    echo -e "  ${RED}Failed: ${FAILED}${NC}"
    echo ""
fi

# Analysis recommendations
echo -e "${CYAN}Performance Analysis Guidance:${NC}"
echo ""
echo "1. Compare BF16 vs FP32 timing for each test suite"
echo "2. On Cascade Lake (no AVX512_BF16): Expect slight overhead from BF16→FP32 expansion"
echo "3. On Cooper Lake+ (with AVX512_BF16): Expect performance improvement or parity"
echo "4. Memory savings: 50% reduction in weight storage (both CPU types)"
echo "5. Cache benefits: Better locality from smaller weights may offset expansion cost"
echo ""

echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "${GREEN}View full results: ${RESULT_FILE}${NC}"
