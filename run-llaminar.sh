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
echo "MPI: ${SOCKETS} processes (1 per socket)"
echo ""

# Run Llaminar with canonical MPI/OpenMP settings
echo "=== Starting Llaminar with Optimal Settings ==="
exec mpirun -np ${SOCKETS} \
  --bind-to socket \
  --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  --report-bindings \
  ./build/llaminar "$@"