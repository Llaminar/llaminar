#!/bin/bash
# run_llaminar.sh - Canonical script to run Llaminar with optimal MPI/OpenMP settings
#
# This wrapper establishes a consistent performance baseline:
#   * One MPI rank per socket by default (unless --mpi-procs / LLAMINAR_MPI_PROCS override)
#   * Threads pinned to sockets; OpenMP restricted to physical cores
#   * Adaptive OpenBLAS threading policy (hybrid) with optional TP-aware downscaling
#   * Centralized thread policy flags (LLAMINAR_OMP_USE_PHYSICAL, LLAMINAR_BIND_PER_SOCKET, LLAMINAR_OMP_FORCE)
#
# Script-level options (intercepted before passing remaining args to the binary):
#   --help                     Show this help & exit
#   --mpi-procs=N              Override number of MPI processes (sets LLAMINAR_MPI_PROCS)
#   --force-threads=N          Force global OpenMP threads (sets LLAMINAR_OMP_FORCE)
#   --no-bind                  Do NOT enable one-rank-per-socket binding flag (even if counts match)
#   --openblas-policy=POLICY   Set OpenBLAS policy: single|match_omp|hybrid (maps to LLAMINAR_OPENBLAS_POLICY)
#   --seq-len-hint=N           Provide sequence length hint (LLAMINAR_SEQ_LEN_HINT) guiding BLAS downscale
#   --dry-run                  Print computed configuration then exit without running mpirun
#
# All other arguments are forwarded to the llaminar executable.
# Example: ./run_llaminar.sh --mpi-procs=4 --force-threads=16 -m models/qwen... -v

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./run_llaminar.sh [script-options] [--] [llaminar-args]

Script Options:
    --help                     Show this help text and exit.
    --mpi-procs=N              Number of MPI processes (default: #sockets).
    --force-threads=N          Force global OpenMP threads (LLAMINAR_OMP_FORCE).
    --no-bind                  Disable automatic per-socket binding flag (LLAMINAR_BIND_PER_SOCKET).
    --openblas-policy=POLICY   OpenBLAS threading policy: single|match_omp|hybrid (default: hybrid).
    --seq-len-hint=N           Hint sequence length (LLAMINAR_SEQ_LEN_HINT) for adaptive BLAS scaling.
    --dry-run                  Show resolved configuration and exit (no mpirun).

Environment Variables (selected):
    LLAMINAR_MPI_PROCS             Same as --mpi-procs.
    LLAMINAR_OMP_FORCE             Same as --force-threads.
    LLAMINAR_OMP_USE_PHYSICAL=1    Auto physical-core restriction (set by script unless forced threads).
    LLAMINAR_BIND_PER_SOCKET=1     Mark one-rank-per-socket policy (auto when ranks == sockets unless --no-bind).
    LLAMINAR_OPENBLAS_POLICY       BLAS policy (single|match_omp|hybrid).
    LLAMINAR_SEQ_LEN_HINT          Sequence length hint for decode scaling.

Examples:
    ./run_llaminar.sh -m models/model.gguf -v
    ./run_llaminar.sh --mpi-procs=4 --force-threads=12 -m models/model.gguf
    ./run_llaminar.sh --dry-run --openblas-policy=single -m models/model.gguf

Notes:
    * Unknown --script-switch triggers this help with non-zero exit.
    * After configuration, all remaining args are passed verbatim to the binary.
EOF
}

# --- Script Option Parsing -------------------------------------------------
BIN_ARGS=()
DRY_RUN=0
while (( "$#" )); do
    case "$1" in
        --help)
            print_help; exit 0 ;;
        --mpi-procs=*)
            export LLAMINAR_MPI_PROCS="${1#*=}" ;;
        --force-threads=*)
            export LLAMINAR_OMP_FORCE="${1#*=}" ;;
        --no-bind)
            export LL_NO_BIND=1 ;;
        --openblas-policy=*)
            export LLAMINAR_OPENBLAS_POLICY="${1#*=}" ;;
        --seq-len-hint=*)
            export LLAMINAR_SEQ_LEN_HINT="${1#*=}" ;;
        --dry-run)
            DRY_RUN=1 ;;
        --)
            shift; BIN_ARGS+=("$@" ); break ;;
        --*)
            echo "[run_llaminar] Error: unknown option '$1'" >&2
            print_help >&2
            exit 1 ;;
        *)
            BIN_ARGS+=("$1") ;;
    esac
    shift
done

set +u # underlying binary may expect to probe additional env, keep strictness relaxed after parse

# Function to detect CPU topology (mirrors our C++ logic in src/common.cpp)
detect_cpu_topology() {
    # Parse /proc/cpuinfo to extract topology information
    local physical_ids=$(grep "^physical id" /proc/cpuinfo | awk '{print $NF}' | sort -u | wc -l)
    local total_cores=$(grep "^processor" /proc/cpuinfo | wc -l)
    
    # Extract unique (socket, core) pairs to count physical cores
    # This mirrors the C++ logic: socket_core_to_threads[{physical_id, core_id}]
    local unique_cores=$(awk '
        /^processor/ { proc = $NF }
        /^physical id/ { phys_id = $NF }
        /^core id/ { core_id = $NF; print phys_id ":" core_id }
    ' /proc/cpuinfo | sort -u | wc -l)
    
    # Calculate topology values (same as C++ detectCPUTopology)
    SOCKETS=$physical_ids
    PHYSICAL_CORES=$unique_cores
    TOTAL_CORES=$total_cores
    CORES_PER_SOCKET=$((PHYSICAL_CORES / SOCKETS))
    THREADS_PER_CORE=$((TOTAL_CORES / PHYSICAL_CORES))
    
    # Hyperthreading detection
    if [ $THREADS_PER_CORE -gt 1 ]; then
        HYPERTHREADING_DETECTED="Yes"
    else
        HYPERTHREADING_DETECTED="No"
    fi
    
    # Set optimal OpenMP thread count (physical cores per socket)
    OMP_THREADS=$CORES_PER_SOCKET
}

# Detect system topology
detect_cpu_topology

# Canonical OpenMP settings for Llaminar (using detected core count)
export OMP_NUM_THREADS=$OMP_THREADS     # Physical cores per socket (auto-detected)
export OMP_PLACES=sockets               # Place threads on sockets  
export OMP_PROC_BIND=close              # Bind threads close to each other
export OMP_NESTED=false                 # Disable nested parallelism
export OMP_DYNAMIC=false                # Disable dynamic thread adjustment

# Additional OpenMP optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0                  # Reduce thread blocking time
export MKL_NUM_THREADS=$OMP_THREADS     # If using MKL (same as OMP)
export MKL_DYNAMIC=false

# Canonical MPI optimizations for Llaminar
export OMPI_MCA_mpi_leave_pinned=1                    # Keep memory pinned
export OMPI_MCA_btl_vader_single_copy_mechanism=none  # Avoid cross-NUMA copies
export OMPI_MCA_btl_openib_allow_ib=1                 # Enable InfiniBand if available

# Additional system information
NUMA_NODES=$(lscpu | grep 'NUMA node(s):' | awk '{print $3}')

# OpenBLAS thread configuration - always match OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export GOTO_NUM_THREADS=$OMP_NUM_THREADS

# Optional MPI process override (default: one process per socket)
MPI_PROCS=${LLAMINAR_MPI_PROCS:-$SOCKETS}
if ! [[ $MPI_PROCS =~ ^[0-9]+$ ]] || [ $MPI_PROCS -lt 1 ]; then
    echo "[run_llaminar] Invalid MPI proc count '$MPI_PROCS' -> falling back to $SOCKETS" >&2
    MPI_PROCS=$SOCKETS
fi

# Validate binary exists
if [ ! -f "./build_release/llaminar" ]; then
    echo "Error: llaminar binary not found at ./build_release/llaminar"
    echo "Please build the project first: cmake -B build_release -DCMAKE_BUILD_TYPE=Release && cmake --build build_release --parallel"
    exit 1
fi

echo "=== Llaminar Canonical Configuration ==="
echo "System: ${SOCKETS} sockets, ${CORES_PER_SOCKET} cores/socket, ${NUMA_NODES} NUMA nodes"
echo "Topology: ${PHYSICAL_CORES} physical cores, ${TOTAL_CORES} logical cores"
echo "Hyperthreading: ${HYPERTHREADING_DETECTED} (${THREADS_PER_CORE} threads/core)"
echo "OpenMP: ${OMP_THREADS} threads/socket, ${OMP_PLACES} placement, ${OMP_PROC_BIND} binding"
echo "OpenBLAS: ${OPENBLAS_NUM_THREADS} threads (matching OMP_NUM_THREADS)"
echo "MPI: ${MPI_PROCS} processes (requested via LLAMINAR_MPI_PROCS or sockets default)"
# Threading policy flags (new centralized runtime control)
if [ -z "${LLAMINAR_OMP_FORCE}" ]; then
    # Respect explicit user force override; otherwise, enable internal physical-core policy
    export LLAMINAR_OMP_USE_PHYSICAL=${LLAMINAR_OMP_USE_PHYSICAL:-1}
fi
if [ "$MPI_PROCS" = "$SOCKETS" ] && [ -z "${LL_NO_BIND:-}" ] && [ -z "${LLAMINAR_BIND_PER_SOCKET}" ]; then
    export LLAMINAR_BIND_PER_SOCKET=1
fi
echo "ThreadPolicy: force=${LLAMINAR_OMP_FORCE:-none} use_physical=${LLAMINAR_OMP_USE_PHYSICAL:-0} bind_per_socket=${LLAMINAR_BIND_PER_SOCKET:-0}"
echo "TP: partitions=${LLAMINAR_ATTN_TP_PARTITIONS:-unset} disabled=${LLAMINAR_ATTN_TP_DISABLE:-0} output_mode=${LLAMINAR_ATTN_OUTPUT_MODE:-auto}" 
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo "[run_llaminar] Dry run requested; exiting before launch." >&2
    exit 0
fi

# Run Llaminar with canonical MPI/OpenMP settings
echo "=== Starting Llaminar with Optimal Settings ==="
exec mpirun -np ${MPI_PROCS} \
    --bind-to socket \
    --map-by socket \
    --mca mpi_leave_pinned 1 \
    --mca btl_vader_single_copy_mechanism none \
    --report-bindings \
    ./build_release/llaminar "${BIN_ARGS[@]}"