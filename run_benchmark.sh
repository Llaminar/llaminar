#!/bin/bash
# run_benchmark.sh - Generic benchmark runner with optimal MPI/OpenMP settings
#
# Usage:
#   ./run_benchmark.sh <benchmark_executable> [benchmark_args...]
#
# Examples:
#   ./run_benchmark.sh ./build_release/benchmark_iq4nl_gemm
#   ./run_benchmark.sh ./build_release/test_performance --gtest_filter=MyTest.*
#   ./run_benchmark.sh benchmark_iq4nl_gemm  # Auto-detects build_release/

set -euo pipefail

# Function to detect CPU topology
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

# Check if benchmark executable is provided
if [ $# -lt 1 ]; then
    cat <<EOF
Usage: $0 <benchmark_executable> [benchmark_args...]

Generic benchmark runner with canonical Llaminar MPI/OpenMP settings.

Arguments:
  benchmark_executable  Path to benchmark binary (absolute or relative)
                       Auto-detects in build_release/ if just basename given
  benchmark_args...    Optional arguments to pass to the benchmark

Examples:
  $0 ./build_release/benchmark_iq4nl_gemm
  $0 benchmark_iq4nl_gemm
  $0 ./build_release/test_performance --gtest_filter=MyTest.*
  $0 test_batch_performance --gtest_filter='*.ThroughputScaling'

Environment Variables:
  LLAMINAR_MPI_PROCS   Override MPI process count (default: one per socket)
  OMP_NUM_THREADS      Override OpenMP thread count (default: cores per socket)

EOF
    exit 1
fi

BENCHMARK_PATH="$1"
shift  # Remove first arg, rest are benchmark arguments

# Auto-detect benchmark in build_release/ if just basename given
if [ ! -f "$BENCHMARK_PATH" ] && [ -f "./build_release/$BENCHMARK_PATH" ]; then
    BENCHMARK_PATH="./build_release/$BENCHMARK_PATH"
fi

# Also try build/ directory for debug builds
if [ ! -f "$BENCHMARK_PATH" ] && [ -f "./build/$BENCHMARK_PATH" ]; then
    BENCHMARK_PATH="./build/$BENCHMARK_PATH"
fi

# Validate binary exists
if [ ! -f "$BENCHMARK_PATH" ]; then
    echo "Error: Benchmark binary not found: $BENCHMARK_PATH"
    echo ""
    echo "Searched in:"
    echo "  - $1 (as given)"
    echo "  - ./build_release/$1"
    echo "  - ./build/$1"
    echo ""
    echo "Please build the benchmark first or provide the correct path."
    exit 1
fi

# Make path absolute for clarity
BENCHMARK_PATH=$(realpath "$BENCHMARK_PATH")
BENCHMARK_NAME=$(basename "$BENCHMARK_PATH")

# Detect system topology
detect_cpu_topology

# Allow override of OMP_NUM_THREADS if explicitly set
if [ -n "${OMP_NUM_THREADS:-}" ]; then
    OMP_THREADS=$OMP_NUM_THREADS
fi

# Canonical OpenMP settings for Llaminar (using detected core count)
export OMP_NUM_THREADS=$OMP_THREADS     # Physical cores per socket
export OMP_PLACES=sockets               # Place threads on sockets  
export OMP_PROC_BIND=close              # Bind threads close to each other
export OMP_NESTED=false                 # Disable nested parallelism
export OMP_DYNAMIC=false                # Disable dynamic thread adjustment

# Additional OpenMP optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0                  # Reduce thread blocking time

# OpenBLAS thread configuration - match OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_THREADS
export GOTO_NUM_THREADS=$OMP_THREADS

# Canonical MPI optimizations for Llaminar
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1

# MPI process count (default: one per socket)
MPI_PROCS=${LLAMINAR_MPI_PROCS:-$SOCKETS}

echo "========================================================================="
echo "Llaminar Benchmark Runner: $BENCHMARK_NAME"
echo "========================================================================="
echo "System Configuration:"
echo "  Sockets:          ${SOCKETS}"
echo "  Cores/socket:     ${CORES_PER_SOCKET}"
echo "  Physical cores:   ${PHYSICAL_CORES}"
echo "  Logical cores:    ${TOTAL_CORES}"
echo "  Hyperthreading:   ${HYPERTHREADING_DETECTED} (${THREADS_PER_CORE} threads/core)"
echo ""
echo "Runtime Configuration:"
echo "  OpenMP threads:   ${OMP_THREADS} (per process)"
echo "  Thread placement: ${OMP_PLACES}"
echo "  Thread binding:   ${OMP_PROC_BIND}"
echo "  OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
echo "  MPI processes:    ${MPI_PROCS}"
echo ""
echo "Benchmark:"
echo "  Executable:       ${BENCHMARK_PATH}"
if [ $# -gt 0 ]; then
    echo "  Arguments:        $*"
fi
echo "========================================================================="
echo ""

# Run benchmark with canonical MPI/OpenMP settings
exec mpirun -np ${MPI_PROCS} \
    --bind-to socket \
    --map-by socket \
    --mca mpi_leave_pinned 1 \
    --mca btl_vader_single_copy_mechanism none \
    --report-bindings \
    "$BENCHMARK_PATH" "$@"
