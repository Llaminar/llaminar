#!/usr/bin/env bash
# Batch performance benchmark runner with optimal MPI/OpenMP settings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build_release"
BENCH_EXEC="${BUILD_DIR}/test_batch_performance"

# Check if benchmark executable exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo -e "${RED}Error: Benchmark executable not found at $BENCH_EXEC${NC}"
    echo "Please build the Release version first:"
    echo "  cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build_release --target test_batch_performance --parallel"
    exit 1
fi
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Batch Processing Performance Benchmark                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if benchmark exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo -e "${YELLOW}Warning: Benchmark executable not found. Building...${NC}"
    cmake --build "$BUILD_DIR" --target test_batch_performance --parallel
fi

# Detect system topology (mirrors run_llaminar.sh logic)
detect_cpu_topology() {
    local physical_ids=$(grep "^physical id" /proc/cpuinfo | awk '{print $NF}' | sort -u | wc -l)
    local total_cores=$(grep "^processor" /proc/cpuinfo | wc -l)
    
    local unique_cores=$(awk '
        /^processor/ { proc = $NF }
        /^physical id/ { phys_id = $NF }
        /^core id/ { core_id = $NF; print phys_id ":" core_id }
    ' /proc/cpuinfo | sort -u | wc -l)
    
    SOCKETS=$physical_ids
    PHYSICAL_CORES=$unique_cores
    TOTAL_CORES=$total_cores
    CORES_PER_SOCKET=$((PHYSICAL_CORES / SOCKETS))
    THREADS_PER_CORE=$((TOTAL_CORES / PHYSICAL_CORES))
    
    if [ $THREADS_PER_CORE -gt 1 ]; then
        HYPERTHREADING_DETECTED="Yes"
    else
        HYPERTHREADING_DETECTED="No"
    fi
    
    OMP_THREADS=$CORES_PER_SOCKET
}

detect_cpu_topology

# CRITICAL: Check if this is a Release build
echo -e "${GREEN}Build Configuration:${NC}"
if [ -f "build_release/CMakeCache.txt" ]; then
    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE:STRING" build_release/CMakeCache.txt | cut -d= -f2)
    echo "  Build type: $BUILD_TYPE"
    if [ "$BUILD_TYPE" != "Release" ]; then
        echo -e "${RED}  ⚠ WARNING: Performance benchmarks require Release build!${NC}"
        echo -e "${RED}  ⚠ Debug builds are 5-10x slower - results will be misleading${NC}"
        echo -e "${RED}  ⚠ Reconfigure with: cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release${NC}"
        echo ""
        read -p "Continue with Debug build anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "  Build type: Unknown (CMakeCache.txt not found)"
fi
echo ""

echo -e "${GREEN}System Topology:${NC}"
echo "  Sockets: $SOCKETS"
echo "  Physical cores: $PHYSICAL_CORES"
echo "  Cores per socket: $CORES_PER_SOCKET"
echo "  Threads per core: $THREADS_PER_CORE"
echo "  Hyperthreading: $HYPERTHREADING_DETECTED"
echo ""

# Canonical OpenMP settings (from run_llaminar.sh)
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export MKL_NUM_THREADS=$OMP_THREADS
export MKL_DYNAMIC=false

# Performance benchmark specific settings
export LLAMINAR_LOG_LEVEL=WARN           # Suppress debug output (test also sets this programmatically)
export ADAPTIVE_DISABLE_COSMA=1          # Baseline measurement with OpenBLAS only

# OpenBLAS settings (match_omp policy for benchmarking)
export OPENBLAS_NUM_THREADS=$OMP_THREADS
export GOTO_NUM_THREADS=$OMP_THREADS

# Canonical MPI optimizations (from run_llaminar.sh)
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1

# Disable COSMA for baseline measurement
export ADAPTIVE_DISABLE_COSMA=1

echo -e "${GREEN}Runtime Configuration:${NC}"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "  MPI ranks: $SOCKETS"
echo "  ADAPTIVE_DISABLE_COSMA: 1 (baseline measurement)"
echo ""

# Parse command line arguments
TEST_FILTER=""
BATCH_FILTER=""
SEQ_LEN_FILTER=""
SHOW_HELP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        --batch)
            BATCH_FILTER="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN_FILTER="$2"
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
    echo "  --batch <sizes>       Test only specific batch sizes (comma-separated)"
    echo "  --seq-len <lengths>   Test only specific sequence lengths (comma-separated)"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Available test suites:"
    echo "  BatchPerformanceTest.PrefillThroughputScaling"
    echo "  BatchPerformanceTest.PrefillThroughputScalingLongSequences"
    echo "  BatchPerformanceTest.DecodeThroughputScaling"
    echo "  BatchPerformanceTest.MemoryBandwidthAnalysis"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Run all benchmarks"
    echo "  $0 --filter '*Prefill*'                     # Run only prefill scaling"
    echo "  $0 --filter '*LongSequences'                # Run long sequence tests"
    echo "  $0 --filter '*LongSequences' --batch 32 --seq-len 512"
    echo "                                              # Test only batch=32, seq_len=512"
    echo "  $0 --batch 1,4,32 --seq-len 256,512         # Test specific combinations"
    echo ""
    echo "Environment Variables (for fine-grained control):"
    echo "  BATCH_SIZE_FILTER=1,32    Same as --batch"
    echo "  SEQ_LEN_FILTER=512        Same as --seq-len"
    exit 0
fi

# Build filter argument and set environment variables
FILTER_ARG=""
if [ -n "$TEST_FILTER" ]; then
    FILTER_ARG="--gtest_filter=$TEST_FILTER"
    echo -e "${YELLOW}Running filtered tests: $TEST_FILTER${NC}"
fi

if [ -n "$BATCH_FILTER" ]; then
    export BATCH_SIZE_FILTER="$BATCH_FILTER"
    echo -e "${YELLOW}Filtered batch sizes: $BATCH_FILTER${NC}"
fi

if [ -n "$SEQ_LEN_FILTER" ]; then
    export SEQ_LEN_FILTER="$SEQ_LEN_FILTER"
    echo -e "${YELLOW}Filtered sequence lengths: $SEQ_LEN_FILTER${NC}"
fi

if [ -n "$FILTER_ARG" ] || [ -n "$BATCH_FILTER" ] || [ -n "$SEQ_LEN_FILTER" ]; then
    echo ""
fi

# Run benchmark with MPI
echo -e "${GREEN}Starting batch performance benchmark...${NC}"
echo -e "${YELLOW}Note: This will take several minutes (loading model multiple times)${NC}"
echo ""

set +e  # Don't exit on benchmark failure
mpirun -np $SOCKETS \
    --bind-to socket \
    --map-by socket \
    --mca mpi_leave_pinned 1 \
    --mca btl_vader_single_copy_mechanism none \
    --report-bindings \
    -x OMP_NUM_THREADS \
    -x OMP_PLACES \
    -x OMP_PROC_BIND \
    -x OMP_NESTED \
    -x OMP_DYNAMIC \
    -x KMP_AFFINITY \
    -x KMP_BLOCKTIME \
    -x MKL_NUM_THREADS \
    -x MKL_DYNAMIC \
    -x OPENBLAS_NUM_THREADS \
    -x GOTO_NUM_THREADS \
    -x OMPI_MCA_mpi_leave_pinned \
    -x OMPI_MCA_btl_vader_single_copy_mechanism \
    -x OMPI_MCA_btl_openib_allow_ib \
    -x ADAPTIVE_DISABLE_COSMA \
    -x BATCH_SIZE_FILTER \
    -x SEQ_LEN_FILTER \
    "$BENCH_EXEC" $FILTER_ARG

RESULT=$?
set -e

echo ""
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
else
    echo -e "${RED}✗ Benchmark failed with exit code $RESULT${NC}"
fi

echo ""
echo -e "${BLUE}Performance Target:${NC}"
echo "  • Baseline: ~13 tok/s @ batch=1 (single sequence)"
echo "  • Target: 288-320 tok/s @ batch=32 (22× speedup)"
echo "  • Good: >20× speedup (>260 tok/s @ batch=32)"
echo "  • Moderate: 15-20× speedup (195-260 tok/s)"
echo "  • Needs optimization: <15× speedup (<195 tok/s)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  • If target met: Proceed to Phase 4.4 (main app integration)"
echo "  • If below target: Phase 5 optimizations needed"

exit $RESULT
