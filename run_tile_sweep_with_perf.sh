#!/bin/bash
# Run IQ4_NL tile sweep with perf profiling for cache analysis
#
# This script runs the tile sweep benchmark with Linux perf to collect
# detailed cache performance metrics (L1/L2/LLC miss rates, IPC, etc.)
#
# Usage:
#   ./run_tile_sweep_with_perf.sh [test_filter]
#
# Examples:
#   ./run_tile_sweep_with_perf.sh                    # Run all tests
#   ./run_tile_sweep_with_perf.sh L2_CacheOptimization  # Run specific test
#   ./run_tile_sweep_with_perf.sh ComprehensiveSweep    # Run comprehensive sweep

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
BUILD_DIR="build_v2_release"
TEST_EXECUTABLE="$BUILD_DIR/performance/v2_perf_iq4nl_tile_sweep"
TEST_FILTER="${1:-*}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ IQ4_NL Tile Sweep with Performance Profiling                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if executable exists
if [ ! -f "$TEST_EXECUTABLE" ]; then
    echo -e "${RED}Error: Test executable not found: $TEST_EXECUTABLE${NC}"
    echo "Building..."
    cmake --build $BUILD_DIR --target v2_perf_iq4nl_tile_sweep --parallel
fi

# Detect CPU topology
SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
THREADS_PER_CORE=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')

echo -e "${GREEN}CPU Topology:${NC}"
echo "  Sockets: $SOCKETS"
echo "  Cores per socket: $CORES_PER_SOCKET"
echo "  Threads per core: $THREADS_PER_CORE"
echo ""

# Cache information
echo -e "${GREEN}Cache Hierarchy:${NC}"
lscpu | grep -E "L1d|L1i|L2|L3" | while read line; do
    echo "  $line"
done
echo ""

# Set optimal environment variables
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

# MPI settings
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=ERROR

echo -e "${GREEN}Environment:${NC}"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Test filter: $TEST_FILTER"
echo ""

# Define perf events to monitor
PERF_EVENTS="L1-dcache-loads,L1-dcache-load-misses,L1-icache-load-misses"
PERF_EVENTS="$PERF_EVENTS,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
PERF_EVENTS="$PERF_EVENTS,dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses"
PERF_EVENTS="$PERF_EVENTS,cycles,instructions,branches,branch-misses"

echo -e "${YELLOW}Running tile sweep benchmark with perf profiling...${NC}"
echo ""

# Run with perf stat
perf stat -e $PERF_EVENTS \
    mpirun -np $SOCKETS \
        --bind-to socket \
        --map-by socket \
        --mca mpi_leave_pinned 1 \
        --mca btl_vader_single_copy_mechanism none \
        $TEST_EXECUTABLE --gtest_filter="IQ4_NL_TileSweep.$TEST_FILTER"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ Benchmark Complete                                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "CSV results saved to:"
echo "  - tile_sweep_l1.csv"
echo "  - tile_sweep_l2.csv"
echo "  - tile_sweep_l3.csv"
echo "  - tile_sweep_comprehensive.csv"
echo ""
echo -e "${BLUE}Performance Metrics Interpretation:${NC}"
echo ""
echo "  ${YELLOW}L1 Cache Metrics:${NC}"
echo "    L1-dcache-load-misses / L1-dcache-loads = L1 miss rate"
echo "    Target: <5% for well-optimized code"
echo ""
echo "  ${YELLOW}LLC (L3) Cache Metrics:${NC}"
echo "    LLC-load-misses / LLC-loads = LLC miss rate"
echo "    Target: <20% for cache-resident workloads"
echo ""
echo "  ${YELLOW}IPC (Instructions Per Cycle):${NC}"
echo "    instructions / cycles"
echo "    Target: >1.5 for compute-bound code"
echo ""
echo "  ${YELLOW}Branch Prediction:${NC}"
echo "    branch-misses / branches"
echo "    Target: <2% for predictable code"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Analyze CSV files to identify best tile configuration"
echo "  2. Check L1/LLC miss rates - lower is better"
echo "  3. Look for tile sizes with high GFLOPS + low miss rates"
echo "  4. Update QuantizedGemmKernel.h with optimal tile sizes"
echo ""
