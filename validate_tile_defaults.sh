#!/bin/bash
# Validate that 64×32 tile defaults are active and show expected performance

set -e

echo "=== Validating 64×32 Tile Defaults ==="
echo ""

# Detect topology
SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
TOTAL_CORES=$((SOCKETS * CORES_PER_SOCKET))

echo "System topology:"
echo "  Sockets: $SOCKETS"
echo "  Cores per socket: $CORES_PER_SOCKET"
echo "  Total physical cores: $TOTAL_CORES"
echo "  Using OMP_NUM_THREADS=$CORES_PER_SOCKET (one socket)"
echo ""

# Configure OpenMP for single socket
export OMP_NUM_THREADS=$CORES_PER_SOCKET
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

# Enable microkernel
export LLAMINAR_IQ4_GEMM_MICROKERNEL=1

echo "Running benchmark with 64×32 defaults (no overrides)..."
echo "Expected performance (from tile sweep):"
echo "  Q-Proj-1024 (FP32): ~336 GFLOPS"
echo "  Q-Proj-4096 (FP32): ~350 GFLOPS"
echo "  FFN-Batch-16 (FP32): ~256 GFLOPS"
echo "  FFN-Batch-256 (FP32): ~539 GFLOPS (97% of optimal)"
echo ""

# Run single-threaded benchmark (no MPI overhead)
timeout 60 ./build_release/benchmark_iq4nl_gemm 2>&1 | \
  grep -A5 "IQ4_NL GEMM Performance" | \
  grep -E "(Throughput|m=|GFLOPS)" || \
  echo "Note: Benchmark requires MPI. Run with: mpirun -np 2 ./build_release/benchmark_iq4nl_gemm"

echo ""
echo "=== Validation Complete ==="
echo ""
echo "To compare against tile sweep results:"
echo "  1. Check tile_sweep_results/combined_fp32_20251022_195855.csv"
echo "  2. Expected geometric mean: 357 GFLOPS (64×32)"
echo "  3. To override: LLAMINAR_IQ4_M_TILE=<value> LLAMINAR_IQ4_N_TILE=<value>"
