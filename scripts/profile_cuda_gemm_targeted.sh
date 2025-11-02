#!/bin/bash
# PHASE 1: Targeted Profiling Script
# Profiles only top-10 and bottom-10 configs from each test case
# Runtime: ~2-4 hours (vs 5 days for all configs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${WORKSPACE_ROOT}/build_v2_release"
PROFILE_OUTPUT="${BUILD_DIR}/cuda_gemm_profiler_data.csv"
BENCHMARK_CSV="${BUILD_DIR}/cuda_gemm_benchmark_data.csv"

echo "=============================================================================="
echo "PHASE 1: Targeted CUDA GEMM Profiling"
echo "=============================================================================="
echo ""
echo "Strategy: Profile top-10 and bottom-10 configs from each test"
echo "Metrics:  Bank conflicts, cache hit rates, occupancy, coalescing"
echo "Runtime:  ~2-4 hours (600 kernel launches)"
echo ""

# Check prerequisites
if [ ! -f "${BENCHMARK_CSV}" ]; then
    echo "[ERROR] Benchmark data not found: ${BENCHMARK_CSV}"
    echo "        Run validation test first:"
    echo "        ./build_v2_release/performance/v2_perf_cuda_heuristic_validation"
    exit 1
fi

if ! command -v ncu &> /dev/null; then
    echo "[ERROR] NVIDIA Nsight Compute (ncu) not found"
    echo "        Add to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

# Check if running with privileges (ncu may need it)
if [ "$EUID" -ne 0 ] && ! ncu --version &> /dev/null; then
    echo "[WARN] ncu may require sudo on some systems"
    echo "       If profiling fails, try: sudo $0"
fi

echo "[1/4] Analyzing benchmark data to select configs..."

# Python script to select top-10 and bottom-10 per test
python3 <<'PYTHON_SCRIPT'
import pandas as pd
import sys

csv_path = "${BENCHMARK_CSV}"
output_path = "${BUILD_DIR}/profiling_targets.csv"

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"[ERROR] Failed to load CSV: {e}", file=sys.stderr)
    sys.exit(1)

# Group by test, select top-10 and bottom-10 GFLOPS
targets = []
for test_name, group in df.groupby('test_name'):
    sorted_group = group.sort_values('gflops', ascending=False)
    top_10 = sorted_group.head(10)
    bottom_10 = sorted_group.tail(10)
    targets.append(top_10)
    targets.append(bottom_10)
    
result = pd.concat(targets, ignore_index=True)
result.to_csv(output_path, index=False)

print(f"[INFO] Selected {len(result)} configs for profiling")
print(f"       ({len(df['test_name'].unique())} tests × 20 configs each)")
print(f"[INFO] Saved to: {output_path}")
PYTHON_SCRIPT

TARGETS_CSV="${BUILD_DIR}/profiling_targets.csv"

echo ""
echo "[2/4] Setting up NVIDIA Nsight Compute profiling..."

# Key metrics to collect (carefully selected for GEMM analysis)
METRICS=(
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"                    # Bank conflicts
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"                    # DRAM utilization
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"                       # Global load sectors
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"                       # Global store sectors
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio" # Coalescing (load)
    "smsp__average_warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active" # Memory stalls
    "smsp__average_warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active" # Math stalls
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"                      # SM utilization
    "smsp__cycles_active.avg.pct_of_peak_sustained_active"                 # Active cycles
    "launch__occupancy_limit_blocks"                                        # Occupancy limiter
    "launch__occupancy_limit_warps"                                         # Warp occupancy
    "launch__registers_per_thread"                                          # Register usage
)

METRIC_STRING=$(IFS=,; echo "${METRICS[*]}")

echo "       Metrics: ${#METRICS[@]} hardware counters"
echo "       Output:  ${PROFILE_OUTPUT}"
echo ""

echo "[3/4] Running profiler (this will take 2-4 hours)..."
echo "       Progress will be shown below..."
echo ""

# Create profiler output header
echo "test_name,m,n,k,tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch_stages,transpose_smem,vectorize_load,gflops,bank_conflicts,dram_util_pct,global_load_sectors,global_store_sectors,coalescing_ratio,mem_stall_pct,math_stall_pct,sm_util_pct,active_cycles_pct,occupancy_limit_blocks,occupancy_limit_warps,registers_per_thread" > "${PROFILE_OUTPUT}"

# TODO: Implement per-config profiling loop
# This requires:
# 1. Building a custom test harness that runs specific configs
# 2. Wrapping with ncu --metrics
# 3. Parsing ncu CSV output
# 4. Appending to PROFILE_OUTPUT

echo "[WARN] Full profiling implementation requires custom test harness"
echo "       For now, demonstrating ncu usage on single test:"
echo ""

# Example: Profile one kernel launch
ncu --metrics ${METRIC_STRING} --csv \
    --target-processes all \
    --force-overwrite \
    --log-file "${BUILD_DIR}/ncu_example.csv" \
    "${BUILD_DIR}/performance/v2_perf_cuda_heuristic_validation" \
    --gtest_filter='*Qwen_0_5B_SingleToken_QKV*' 2>&1 | head -50

echo ""
echo "[4/4] Profiling summary"
echo "       See example output: ${BUILD_DIR}/ncu_example.csv"
echo ""
echo "=============================================================================="
echo "NEXT STEPS"
echo "=============================================================================="
echo ""
echo "To complete Phase 1 profiling:"
echo "1. Implement per-config test harness (TODO)"
echo "2. Run full profiling (~2-4 hours)"
echo "3. Merge profiler data with benchmark CSV"
echo "4. Add profiler metrics to feature engineering"
echo "5. Re-train neural network with enriched features"
echo ""
echo "For Phase 3 (model-guided profiling):"
echo "1. Train initial model (Phase 2 estimated features)"
echo "2. Identify high-uncertainty predictions"
echo "3. Profile only uncertain configs (~100-200)"
echo "4. Add as additional training data"
echo "5. Re-train final model"
echo ""
