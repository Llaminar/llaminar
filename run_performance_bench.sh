#!/bin/bash
# Performance Benchmark Runner for Prefill Operations
# Runs comprehensive parallelization analysis with MPI/OpenMP

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_release"
BENCH_EXEC="${BUILD_DIR}/test_prefill_performance_bench"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Prefill Performance Benchmark Runner                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if benchmark exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo -e "${YELLOW}Warning: Benchmark executable not found. Building...${NC}"
    cmake --build "$BUILD_DIR" --target test_prefill_performance_bench --parallel
fi

# Detect system topology
SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
THREADS_PER_CORE=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
TOTAL_CORES=$((SOCKETS * CORES_PER_SOCKET))

echo -e "${GREEN}System Topology:${NC}"
echo "  Sockets: $SOCKETS"
echo "  Cores per socket: $CORES_PER_SOCKET"
echo "  Threads per core: $THREADS_PER_CORE"
echo "  Total physical cores: $TOTAL_CORES"
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
echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "  MPI ranks: $SOCKETS"
echo ""

# Parse command line arguments
TEST_FILTER=""
SHOW_HELP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            SHOW_HELP=1
            shift
            ;;
    esac
done

if [ $SHOW_HELP -eq 1 ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --filter <pattern>    Run only tests matching pattern"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Available test suites:"
    echo "  PrefillPerformanceBench.OpenBLAS_StrongScaling_Prefill"
    echo "  PrefillPerformanceBench.OpenBLAS_StrongScaling_LargePrefill"
    echo "  PrefillPerformanceBench.OpenBLAS_WeakScaling"
    echo "  PrefillPerformanceBench.OpenBLAS_ModelShapes"
    echo "  PrefillPerformanceBench.ThreadUtilizationAnalysis"
    echo "  PrefillPerformanceBench.COSMA_StrongScaling_Prefill"
    echo "  PrefillPerformanceBench.COSMA_ModelShapes"
    echo "  PrefillPerformanceBench.ComparativePerformance"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all benchmarks"
    echo "  $0 --filter '*ModelShapes*'           # Run both OpenBLAS and COSMA model shapes"
    echo "  $0 --filter '*StrongScaling*'         # Run all strong scaling tests"
    echo "  $0 --filter '*COSMA*'                 # Run only COSMA benchmarks"
    echo "  $0 --filter 'Comparative*'            # Run OpenBLAS vs COSMA comparison"
    exit 0
fi

# Build filter argument
FILTER_ARG=""
if [ -n "$TEST_FILTER" ]; then
    FILTER_ARG="--gtest_filter=$TEST_FILTER"
    echo -e "${YELLOW}Running filtered tests: $TEST_FILTER${NC}"
    echo ""
fi

# Run benchmark with MPI
echo -e "${GREEN}Starting benchmark...${NC}"
echo ""

set +e  # Don't exit on benchmark failure
mpirun -np $SOCKETS \
    --bind-to socket \
    --map-by socket \
    --mca mpi_leave_pinned 1 \
    --mca btl_vader_single_copy_mechanism none \
    --report-bindings \
    "$BENCH_EXEC" $FILTER_ARG

RESULT=$?
set -e

echo ""
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
else
    echo -e "${YELLOW}⚠ Benchmark exited with code $RESULT${NC}"
fi

echo ""
echo -e "${BLUE}Performance Analysis Tips:${NC}"
echo "  • Efficiency > 90%: Excellent parallelization"
echo "  • Efficiency 70-90%: Good parallelization"
echo "  • Efficiency 50-70%: Moderate overhead"
echo "  • Efficiency < 50%: Poor parallelization (bottleneck present)"
echo ""
echo "For detailed analysis, review the GFLOPS, memory bandwidth, and CPU utilization columns."

exit $RESULT
