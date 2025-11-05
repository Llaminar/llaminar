#!/bin/bash
# Profile the 3 best GEMM configs with NVIDIA Nsight Compute

set -e

export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

echo "================================================================"
echo "PROFILING BEST GEMM CONFIGURATIONS WITH NCU"
echo "================================================================"

# Test 1: SingleToken_QKV (1×896×896) - 33.5 GFLOPS
echo ""
echo "=== Test 1: SingleToken_QKV (1×896×896) ==="
echo "Expected: ~33-38 GFLOPS, 64 threads, atom layout 1×4×1"
echo ""
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed \
--launch-skip 5 --launch-count 1 \
./profile_best_configs 1 2>&1 | grep -E "(kernel-name|Metric|GFLOPS|Profiling|Config|Shape|Average)" || true

echo ""
echo "=== Test 2: Batch32_QKV (32×896×896) ==="
echo "Expected: ~900 GFLOPS, 64 threads, atom layout 4×2×1"
echo ""
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed \
--launch-skip 5 --launch-count 1 \
./profile_best_configs 2 2>&1 | grep -E "(kernel-name|Metric|GFLOPS|Profiling|Config|Shape|Average)" || true

echo ""
echo "=== Test 3: FFN_Gate (1×4864×896) ==="  
echo "Expected: ~128 GFLOPS, 64 threads, atom layout 1×1×1"
echo ""
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed \
--launch-skip 5 --launch-count 1 \
./profile_best_configs 3 2>&1 | grep -E "(kernel-name|Metric|GFLOPS|Profiling|Config|Shape|Average)" || true

echo ""
echo "================================================================"
echo "PROFILING COMPLETE"
echo "================================================================"
echo ""
echo "Key metrics to analyze:"
echo "  - SM throughput: Should be >50% for good utilization"
echo "  - Memory throughput: Indicates bandwidth bottleneck"
echo "  - Warp activity: Low % indicates poor occupancy"
echo "  - FMA instructions: Should dominate (indicates compute-bound)"
echo ""
