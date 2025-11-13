#!/bin/bash
#
# run_integer_gemm_sweep.sh
# Run Integer GEMM configuration space sweep for ML training data
#
# Usage:
#   ./run_integer_gemm_sweep.sh quick      # Quick test (100 configs, ~2 minutes)
#   ./run_integer_gemm_sweep.sh single     # All configs single workload (~1-2 hours)
#   ./run_integer_gemm_sweep.sh multi      # All configs × 5 workloads (~5-10 hours)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the workspace root
cd "$(dirname "$0")"

# Fresh Release build (clean rebuild for reproducibility)
echo -e "${GREEN}=== Rebuilding Release target (clean) ===${NC}"
if [ -d "build_v2_release" ]; then
    echo -e "${YELLOW}Removing existing build_v2_release directory${NC}"
    rm -rf build_v2_release
fi

echo "Configuring (cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release)"
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release

echo "Building v2_perf_integer_gemm_full_sweep (Release)"
cmake --build build_v2_release --target v2_perf_integer_gemm_full_sweep --parallel

# Path to performance sweep executable (built above)
EXE="build_v2_release/performance/v2_perf_integer_gemm_full_sweep"

# Parse mode
MODE="${1:-quick}"

case "$MODE" in
    quick)
        echo -e "${GREEN}=== Integer GEMM Quick Sweep ===${NC}"
        echo "Testing first 100 configurations (~2 minutes)"
        OUTPUT="integer_gemm_sweep_quick.csv"
        LOG="integer_gemm_sweep_quick.log"
        FILTER="*QuickSweep"
        ;;
    
    single)
        echo -e "${GREEN}=== Integer GEMM Full Sweep (Single Workload) ===${NC}"
        echo "Testing all 8000 configurations, single token workload (~1-2 hours)"
        OUTPUT="integer_gemm_sweep_single.csv"
        LOG="integer_gemm_sweep_single.log"
        FILTER="*AllConfigsSingleToken"
        ;;
    
    multi)
        echo -e "${GREEN}=== Integer GEMM Full Sweep (Multiple Workloads) ===${NC}"
        echo "Testing all 8000 configs × 5 workloads (~5-10 hours)"
        OUTPUT="integer_gemm_sweep_multi.csv"
        LOG="integer_gemm_sweep_multi.log"
        FILTER="*MultipleWorkloads"
        ;;
    
    *)
        echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
        echo "Usage: $0 {quick|single|multi}"
        exit 1
        ;;
esac

echo ""
echo "Output files:"
echo "  CSV data: $OUTPUT"
echo "  Log file: $LOG"
echo ""

# Clean up any existing output files (for CI/automation friendliness)
if [ -f "$OUTPUT" ]; then
    echo -e "${YELLOW}Warning: Overwriting existing $OUTPUT${NC}"
    rm -f "$OUTPUT"
fi
if [ -f "$LOG" ]; then
    rm -f "$LOG"
fi

echo -e "${GREEN}Starting sweep...${NC}"
echo ""

# Run the sweep via CTest to get optimal OpenMP/MPI environment setup
# CTest automatically sets:
#   - OMP_NUM_THREADS (physical cores per socket)
#   - OMP_PLACES=sockets, OMP_PROC_BIND=close
#   - MPI binding (--bind-to socket, --map-by socket)
#   - BLAS threading (OPENBLAS_NUM_THREADS, MKL_NUM_THREADS)
START_TIME=$(date +%s)

# Map our filter modes to CTest test names
case "$FILTER" in
    "*QuickSweep")
        TEST_NAME="V2_Perf_IntegerGEMM_FullSweep_Quick"
        # Use CTest with --verbose to capture test output
        # CTest prefixes output lines with "127: ", so we need to strip that
        cd build_v2_release
        ctest -R "$TEST_NAME" --verbose 2>&1 | tee "../$LOG" | sed 's/^127: //' | grep -E '^(m,n,k|[0-9]+,)' > "../$OUTPUT"
        EXIT_CODE=${PIPESTATUS[0]}
        cd ..
        ;;
        
    "*AllConfigsSingleToken"|"*MultipleWorkloads")
        # These aren't registered in CTest (too long-running)
        # Use direct execution with manual environment setup
        echo -e "${YELLOW}Note: Using direct execution (test not in CTest)${NC}"
        
        # Detect CPU topology
        NUM_SOCKETS=$(lscpu | grep 'Socket(s):' | awk '{print $2}')
        CORES_PER_SOCKET=$(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')
        CORES_PER_SOCKET=${CORES_PER_SOCKET:-1}
        
        # Set optimal OpenMP/BLAS environment (matches add_v2_perf_test)
        export OMP_NUM_THREADS=$CORES_PER_SOCKET
        export OMP_PLACES=sockets
        export OMP_PROC_BIND=close
        export OMP_NESTED=false
        export OMP_DYNAMIC=false
        export KMP_AFFINITY=granularity=fine,compact,1,0
        export KMP_BLOCKTIME=0
        export OPENBLAS_NUM_THREADS=$CORES_PER_SOCKET
        export GOTO_NUM_THREADS=$CORES_PER_SOCKET
        export MKL_NUM_THREADS=$CORES_PER_SOCKET
        export MKL_DYNAMIC=false
        export LLAMINAR_LOG_LEVEL=INFO
        
        echo "CPU topology: $NUM_SOCKETS socket(s), $CORES_PER_SOCKET cores/socket"
        echo "OpenMP threads: $OMP_NUM_THREADS"
        echo ""
        
        "$EXE" --gtest_filter="$FILTER" 2>&1 | tee "$LOG" | grep -E '^(m,n,k|[0-9]+,)' > "$OUTPUT"
        EXIT_CODE=${PIPESTATUS[0]}
        ;;
esac

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Sweep completed successfully!${NC}"
    echo ""
    echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    
    # Count results (CSV already filtered by grep pipeline)
    TOTAL_LINES=$(wc -l < "$OUTPUT")
    DATA_LINES=$((TOTAL_LINES - 1))  # Exclude CSV header
    
    echo "Results:"
    echo "  Total configs tested: $DATA_LINES"
    echo "  CSV file: $OUTPUT ($TOTAL_LINES lines including header)"
    echo "  Log file: $LOG"
    echo ""
    
    # Show first few lines
    echo "Preview (first 10 results):"
    head -11 "$OUTPUT"
    echo ""
    
    # Show performance range
    if command -v awk &> /dev/null; then
        echo "Performance summary:"
        awk -F',' 'NR>1 {
            gflops=$12; 
            if (gflops > max_gflops) max_gflops = gflops;
            if (min_gflops == "" || gflops < min_gflops) min_gflops = gflops;
            sum_gflops += gflops;
            count++;
        } END {
            printf "  Min GFLOPS: %.2f\n", min_gflops;
            printf "  Max GFLOPS: %.2f\n", max_gflops;
            printf "  Avg GFLOPS: %.2f\n", sum_gflops/count;
        }' "$OUTPUT"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Analyze data: python3 src/v2/kernels/cpu/gemm/python/analyze_sweep_results.py $OUTPUT"
    echo "  2. Train ML model: python3 src/v2/kernels/cpu/gemm/python/train_autotuner.py --input $OUTPUT"
    
else
    echo -e "${RED}❌ Sweep failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo "Check log file for details: $LOG"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$LOG"
    exit $EXIT_CODE
fi
