#!/bin/bash
# optimized_run.sh - Run production demo with optimal MPI and OpenMP settings

# Set OpenMP environment for optimal performance
export OMP_NUM_THREADS=28          # Cores per socket (adjust based on your system)
export OMP_PLACES=sockets          # Place threads on sockets
export OMP_PROC_BIND=close         # Bind threads close to each other
export OMP_NESTED=false            # Disable nested parallelism
export OMP_DYNAMIC=false           # Disable dynamic thread adjustment

# Additional OpenMP optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0             # Reduce thread blocking time
export MKL_NUM_THREADS=28          # If using MKL
export MKL_DYNAMIC=false

# MPI optimizations
export OMPI_MCA_mpi_leave_pinned=1                    # Keep memory pinned
export OMPI_MCA_btl_vader_single_copy_mechanism=none  # Avoid cross-NUMA copies
export OMPI_MCA_btl_openib_allow_ib=1                 # Enable InfiniBand if available

echo "=== Optimized MPI/OpenMP Configuration ==="
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "OMP_PLACES: $OMP_PLACES"
echo "OMP_PROC_BIND: $OMP_PROC_BIND"
echo ""

SOCKETS=$(lscpu | grep 'Socket(s):' | awk '{print $2}')

# Check system topology
echo "=== System Information ==="
echo "Physical cores: $(nproc)"
echo "NUMA nodes: $(lscpu | grep 'NUMA node(s):' | awk '{print $3}')"
echo "Sockets: ${SOCKETS}"
echo ""

# Run with optimal MPI settings
echo "=== Starting Production Demo with Optimized Settings ==="
mpirun -np ${SOCKETS} \
  --bind-to socket \
  --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  --report-bindings \
  ./build/production_demo

echo ""
echo "=== Run Complete ==="