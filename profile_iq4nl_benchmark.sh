#!/bin/bash
# profile_iq4nl_benchmark.sh - Profile IQ4_NL GEMM benchmark with Linux perf
#
# This script runs the benchmark under perf to identify performance hotspots
# and generates detailed profiling reports.

set -euo pipefail

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo "Error: perf is not installed"
    echo "Install with: sudo apt-get install linux-tools-generic"
    exit 1
fi

# Output directory for profiling data
PROFILE_DIR="./profile_results"
mkdir -p "$PROFILE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$PROFILE_DIR/perf_${TIMESTAMP}.data"
REPORT_FILE="$PROFILE_DIR/perf_report_${TIMESTAMP}.txt"
ANNOTATE_FILE="$PROFILE_DIR/perf_annotate_${TIMESTAMP}.txt"
FLAMEGRAPH_FILE="$PROFILE_DIR/flamegraph_${TIMESTAMP}.svg"

echo "========================================================================="
echo "IQ4_NL GEMM Benchmark Profiling with perf"
echo "========================================================================="
echo "Profile data: $PERF_DATA"
echo "Report:       $REPORT_FILE"
echo "Annotations:  $ANNOTATE_FILE"
echo "Flamegraph:   $FLAMEGRAPH_FILE"
echo ""

# Get directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the generic benchmark runner to get proper MPI/OpenMP config
# But we'll run perf on the actual mpirun command
echo "=== Running benchmark under perf ==="
echo ""

# Detect CPU topology (copied from run_benchmark.sh)
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
}

detect_cpu_topology

# Set canonical environment variables
export OMP_NUM_THREADS=$CORES_PER_SOCKET
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export OPENBLAS_NUM_THREADS=$CORES_PER_SOCKET
export GOTO_NUM_THREADS=$CORES_PER_SOCKET
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1

MPI_PROCS=${LLAMINAR_MPI_PROCS:-$SOCKETS}
BENCHMARK_PATH="./build_release/benchmark_iq4nl_gemm"

if [ ! -f "$BENCHMARK_PATH" ]; then
    echo "Error: Benchmark not found at $BENCHMARK_PATH"
    echo "Build with: cmake --build build_release --target benchmark_iq4nl_gemm --parallel"
    exit 1
fi

echo "Configuration:"
echo "  OpenMP threads: $OMP_NUM_THREADS"
echo "  MPI processes:  $MPI_PROCS"
echo "  Benchmark:      $BENCHMARK_PATH"
echo ""

# Profile with perf record
# -F 999: Sample at 999 Hz (high frequency for detailed profile)
# -g: Enable call-graph (stack trace) recording
# --call-graph dwarf: Use DWARF debug info for accurate call graphs
# -o: Output file
# --: Everything after this is the command to profile

echo "Recording profile data (this will take a few minutes)..."
echo ""

# For MPI programs, we profile the entire mpirun tree
perf record \
    -F 999 \
    -g \
    --call-graph dwarf \
    -o "$PERF_DATA" \
    -- \
    mpirun -np ${MPI_PROCS} \
        --bind-to socket \
        --map-by socket \
        --mca mpi_leave_pinned 1 \
        --mca btl_vader_single_copy_mechanism none \
        "$BENCHMARK_PATH" 2>&1 | tee "$PROFILE_DIR/benchmark_output_${TIMESTAMP}.log"

echo ""
echo "=== Generating perf report ==="

# Generate detailed report
perf report -i "$PERF_DATA" --stdio --percent-limit 0.5 > "$REPORT_FILE"

# Show top 30 hotspots
echo ""
echo "=== Top 30 Hotspots ==="
perf report -i "$PERF_DATA" --stdio -n --sort symbol --percent-limit 0.5 | head -50 | tee "$PROFILE_DIR/top_hotspots_${TIMESTAMP}.txt"

echo ""
echo "=== Annotating hot functions ==="

# Get top 5 functions by percentage
TOP_FUNCTIONS=$(perf report -i "$PERF_DATA" --stdio -n --sort symbol --percent-limit 2.0 | \
    grep -E '^\s+[0-9]+\.[0-9]+%' | \
    awk '{print $NF}' | \
    head -5)

# Annotate each hot function
for func in $TOP_FUNCTIONS; do
    echo "Annotating: $func"
    perf annotate -i "$PERF_DATA" --stdio "$func" >> "$ANNOTATE_FILE" 2>/dev/null || true
done

echo ""
echo "=== Performance Counters Summary ==="

# Run again with stat to get hardware counters
echo "Running with hardware counters..."
perf stat \
    -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses \
    -- \
    mpirun -np ${MPI_PROCS} \
        --bind-to socket \
        --map-by socket \
        --mca mpi_leave_pinned 1 \
        --mca btl_vader_single_copy_mechanism none \
        "$BENCHMARK_PATH" 2>&1 | tee "$PROFILE_DIR/perf_stat_${TIMESTAMP}.log"

echo ""
echo "========================================================================="
echo "Profiling Complete!"
echo "========================================================================="
echo ""
echo "Results saved to $PROFILE_DIR/"
echo ""
echo "Key files:"
echo "  - $REPORT_FILE (full report)"
echo "  - $PROFILE_DIR/top_hotspots_${TIMESTAMP}.txt (top functions)"
echo "  - $ANNOTATE_FILE (source-level annotation)"
echo "  - $PROFILE_DIR/perf_stat_${TIMESTAMP}.log (hardware counters)"
echo ""
echo "To explore interactively:"
echo "  perf report -i $PERF_DATA"
echo ""
echo "To generate flamegraph (if available):"
echo "  perf script -i $PERF_DATA | flamegraph.pl > $FLAMEGRAPH_FILE"
echo ""
echo "Quick analysis:"
grep -E "Overhead|cache-miss|branch-miss|IPC" "$PROFILE_DIR/perf_stat_${TIMESTAMP}.log" 2>/dev/null || true
echo ""
echo "========================================================================="
