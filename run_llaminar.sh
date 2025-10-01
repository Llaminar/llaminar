#!/bin/bash
# run-llaminar.sh - Canonical script to run Llaminar with optimal MPI/OpenMP settings
#
# Usage: ./run-llaminar.sh [llaminar arguments]
# Example: ./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v

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

# --- Adaptive OpenBLAS Thread Policy ---------------------------------------
# We allow a lightweight policy to prevent oversubscription during decode while
# still giving larger prefill GEMMs thread parallelism.
# Policies (env: LLAMINAR_OPENBLAS_POLICY):
#   single     -> Always use 1 thread (good for latency-sensitive decode)
#   match_omp  -> Match OMP_NUM_THREADS exactly
#   hybrid(*)  -> If many cores, use half (min 4); if small core count (<=8) use all
#                 Additionally, if LLAMINAR_SEQ_LEN_HINT < 1024, force 1 thread
# (*) default
if [ -z "${OPENBLAS_NUM_THREADS}" ]; then
    POLICY=${LLAMINAR_OPENBLAS_POLICY:-hybrid}
    case "$POLICY" in
        single)
            OPENBLAS_NUM_THREADS=1 ;;
        match_omp)
            OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS ;;
        hybrid)
            if [ $OMP_NUM_THREADS -le 8 ]; then
                OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
            else
                OPENBLAS_NUM_THREADS=$((OMP_NUM_THREADS/2))
                [ $OPENBLAS_NUM_THREADS -lt 4 ] && OPENBLAS_NUM_THREADS=4
            fi
            # Sequence length hint override for very short decode shapes
            if [ -n "${LLAMINAR_SEQ_LEN_HINT}" ]; then
                if [ "${LLAMINAR_SEQ_LEN_HINT}" -lt 1024 ] 2>/dev/null; then
                    OPENBLAS_NUM_THREADS=1
                fi
            fi
            ;;
        *)
            echo "[run-llaminar] Warning: Unknown LLAMINAR_OPENBLAS_POLICY='$POLICY' (fallback=hybrid)" >&2
            if [ $OMP_NUM_THREADS -le 8 ]; then
                OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
            else
                OPENBLAS_NUM_THREADS=$((OMP_NUM_THREADS/2))
                [ $OPENBLAS_NUM_THREADS -lt 4 ] && OPENBLAS_NUM_THREADS=4
            fi
            ;;
    esac
    export OPENBLAS_NUM_THREADS
    export GOTO_NUM_THREADS=$OPENBLAS_NUM_THREADS
fi

# Provide visibility if user forced a BLAS thread override already
if [ -n "${OPENBLAS_NUM_THREADS}" ]; then
    BLAS_THREAD_POLICY=${LLAMINAR_OPENBLAS_POLICY:-explicit_or_auto}
fi

# --- TP-Aware BLAS Thread Auto-Lowering ------------------------------------
# If we are doing column (or future row) TP partitioning, per-partition GEMMs
# already provide parallelism at the partition level; keeping large BLAS
# thread counts can oversubscribe cores. We downscale unless disabled.
# Disable with: export LLAMINAR_DISABLE_TP_BLAS_AUTO=1
BASE_OPENBLAS_THREADS=$OPENBLAS_NUM_THREADS
if [ -z "${LLAMINAR_DISABLE_TP_BLAS_AUTO}" ]; then
    if [ -n "${LLAMINAR_ATTN_TP_PARTITIONS}" ]; then
        if [[ ${LLAMINAR_ATTN_TP_PARTITIONS} =~ ^[0-9]+$ ]] && [ ${LLAMINAR_ATTN_TP_PARTITIONS} -gt 1 ]; then
            if [ ${OPENBLAS_NUM_THREADS} -gt 1 ]; then
                TP_PARTS=${LLAMINAR_ATTN_TP_PARTITIONS}
                NEW_THREADS=$((OPENBLAS_NUM_THREADS / TP_PARTS))
                # Floor based on socket count (reserve at least one thread per MPI rank/socket)
                SOCKET_FLOOR=${SOCKETS}
                if [ -n "${LLAMINAR_SEQ_LEN_HINT}" ] && [ "${LLAMINAR_SEQ_LEN_HINT}" -lt 512 ] 2>/dev/null; then
                    # For very small decode shapes we allow dropping below socket floor
                    SHORT_DECODE=1
                fi
                if [ -z "${SHORT_DECODE}" ]; then
                    [ $NEW_THREADS -lt $SOCKET_FLOOR ] && NEW_THREADS=$SOCKET_FLOOR
                else
                    # But never below 1 even in short decode
                    [ $NEW_THREADS -lt 1 ] && NEW_THREADS=1
                fi
                if [ $NEW_THREADS -lt $OPENBLAS_NUM_THREADS ]; then
                    export OPENBLAS_NUM_THREADS=$NEW_THREADS
                    export GOTO_NUM_THREADS=$NEW_THREADS
                    BLAS_TP_SCALING_APPLIED=1
                    BLAS_TP_SOCKET_FLOOR=$SOCKET_FLOOR
                fi
            fi
        fi
    fi
fi
EFFECTIVE_OPENBLAS_THREADS=$OPENBLAS_NUM_THREADS

# Optional MPI process override (default: one process per socket)
MPI_PROCS=${LLAMINAR_MPI_PROCS:-$SOCKETS}
if ! [[ $MPI_PROCS =~ ^[0-9]+$ ]] || [ $MPI_PROCS -lt 1 ]; then
    echo "[run-llaminar] Invalid MPI proc count '$MPI_PROCS' -> falling back to $SOCKETS" >&2
    MPI_PROCS=$SOCKETS
fi

# Validate binary exists
if [ ! -f "./build/llaminar" ]; then
    echo "Error: llaminar binary not found at ./build/llaminar"
    echo "Please build the project first: cmake --build build --parallel"
    exit 1
fi

echo "=== Llaminar Canonical Configuration ==="
echo "System: ${SOCKETS} sockets, ${CORES_PER_SOCKET} cores/socket, ${NUMA_NODES} NUMA nodes"
echo "Topology: ${PHYSICAL_CORES} physical cores, ${TOTAL_CORES} logical cores"
echo "Hyperthreading: ${HYPERTHREADING_DETECTED} (${THREADS_PER_CORE} threads/core)"
echo "OpenMP: ${OMP_THREADS} threads/socket, ${OMP_PLACES} placement, ${OMP_PROC_BIND} binding"
echo "MPI: ${MPI_PROCS} processes (requested via LLAMINAR_MPI_PROCS or sockets default)"
if [ -n "${BLAS_TP_SCALING_APPLIED}" ]; then
    if [ -n "${BLAS_TP_SOCKET_FLOOR}" ]; then
        echo "BLAS: OpenBLAS threads=${EFFECTIVE_OPENBLAS_THREADS} (base=${BASE_OPENBLAS_THREADS}) policy=${LLAMINAR_OPENBLAS_POLICY:-auto} tp_scale=on floor=socket(${BLAS_TP_SOCKET_FLOOR}) (seq_len_hint=${LLAMINAR_SEQ_LEN_HINT:-none})"
    else
        echo "BLAS: OpenBLAS threads=${EFFECTIVE_OPENBLAS_THREADS} (base=${BASE_OPENBLAS_THREADS}) policy=${LLAMINAR_OPENBLAS_POLICY:-auto} tp_scale=on (short_decode) (seq_len_hint=${LLAMINAR_SEQ_LEN_HINT:-none})"
    fi
else
    echo "BLAS: OpenBLAS threads=${EFFECTIVE_OPENBLAS_THREADS} policy=${LLAMINAR_OPENBLAS_POLICY:-auto} tp_scale=off (seq_len_hint=${LLAMINAR_SEQ_LEN_HINT:-none})"
fi
echo "TP: partitions=${LLAMINAR_ATTN_TP_PARTITIONS:-unset} disabled=${LLAMINAR_ATTN_TP_DISABLE:-0} output_mode=${LLAMINAR_ATTN_OUTPUT_MODE:-auto}" 
echo ""

# Run Llaminar with canonical MPI/OpenMP settings
echo "=== Starting Llaminar with Optimal Settings ==="
exec mpirun -np ${MPI_PROCS} \
  --bind-to socket \
  --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  --report-bindings \
  ./build/llaminar "$@"