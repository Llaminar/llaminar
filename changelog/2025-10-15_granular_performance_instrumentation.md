# Granular Performance Instrumentation - October 15, 2025

## Summary
Added fine-grained performance tracing to attention, linear, and normalization operators to enable detailed profiling and bottleneck identification. This builds on the hierarchical performance tracing framework by instrumenting individual kernel operations and MPI collective communications.

## Changes Made

### 1. Attention Operator Granular Tracing (`src/operators/MPIAttentionOperator.cpp`)

**Stage-Level Tracing (6 major stages):**
- `distributeWeightsByHead()`: "weight_distribution" category
- `computeQKVProjections()`: "qkv_projections" category
- `applyRotaryPositionEmbeddings()`: "rope_application" category
- `handleGQAExpansion()`: "gqa_expansion" category
- `computeAttentionScores()`: "attention_scores" category
- `projectAndGatherOutput()`: "output_projection" category

**Kernel-Level Tracing (10 kernel operations):**
- `qk_matmul_unmasked`: Unmasked Q·K^T computation for snapshots (attention_kernel)
- `qk_matmul_masked`: Masked Q·K^T computation for actual attention (attention_kernel)
- `softmax_all_heads`: Softmax across all heads (attention_kernel)
- `scores_v_matmul`: Attention scores × V multiplication (attention_kernel)
- `output_proj_matmul`: Final output projection matmul (attention_kernel)

**MPI Collective Tracing (5 operations):**
- `allgatherv_unmasked_scores`: Gather unmasked scores for snapshot (mpi_collective)
- `allgatherv_softmax_scores`: Gather softmax probabilities (mpi_collective)
- `allgather_attended_rows`: Row-by-row gather of attended values (mpi_collective)
- `allreduce_output`: Final output aggregation across ranks (mpi_collective)

**Hierarchy Depth:**
- Depth 1: `mpi_attention_execute` (top-level operator)
- Depth 2: Stage methods (qkv_projections, rope_application, etc.)
- Depth 3: Kernel operations (matmuls, softmax, MPI collectives)

### 2. Linear Operator Granular Tracing (`src/operators/MPILinearOperator.cpp`)

**Kernel-Level Tracing (4 operations):**
- `distribute_weight`: Weight matrix distribution to ranks (linear_kernel)
- `distribute_bias`: Bias vector distribution (linear_kernel)
- `linear_matmul`: Adaptive matrix multiplication (linear_kernel)
- `add_bias`: Bias addition to outputs (linear_kernel)

**MPI Collective Tracing (1 operation):**
- `gather_output`: Gather distributed results (mpi_collective)

**Hierarchy Depth:**
- Depth 1: `mpi_linear_execute` (top-level operator)
- Depth 2: Kernel operations and collectives

### 3. RMSNorm Operator Granular Tracing (`src/operators/MPIRMSNormOperator.cpp`)

**Feature-Sharded Path Tracing:**
- `rmsnorm_feature_sharded`: Top-level sharded path (norm_kernel)
  - `rmsnorm_row_sumsq`: Per-row sum-of-squares computation (norm_kernel)
  - `allreduce_row_sumsq`: Global reduction of row statistics (mpi_collective)
  - `rmsnorm_apply_scaling`: Apply normalization and scaling (norm_kernel)

**Distributed Path Tracing:**
- `rmsnorm_distributed`: Row-distributed computation (norm_kernel)
- `distribute_input`: Input distribution to ranks (norm_kernel)
- `gather_output`: Output gathering (mpi_collective)

**Hierarchy Depth:**
- Depth 1: `mpi_rmsnorm_execute` (top-level operator)
- Depth 2: Computation and collective operations

## Performance Insights from Initial Trace

### Bottleneck Hierarchy (Prefill, 893 tokens, Q8_0 quantization):

**1. Linear Operations Dominate (73.1% of total time)**
- Total linear time: 1591.01ms across 73 calls
- Breakdown by projection type:
  - `lm_head`: 516.88ms (32.5%) - Single large projection (893x896 -> 893x151936)
  - `ffn_gate`: 365.17ms (23.0%) - 24 layers × 15.2ms avg
  - `ffn_up`: 362.15ms (22.8%) - 24 layers × 15.1ms avg
  - `ffn_down`: 347.21ms (21.8%) - 24 layers × 14.5ms avg
- Kernel breakdown within linear:
  - `distribute_weight`: 122.11ms (7.7%) - Weight distribution overhead
  - `linear_matmul`: 29.29ms (1.8%) - Actual computation (!!)
  - `gather_output`: 31.15ms (2.0%) - MPI gather overhead

**Critical Finding**: Weight distribution (122ms) + gather (31ms) = 153ms overhead vs 29ms compute!
- **Overhead ratio: 5.3× the actual computation time**
- **Optimization opportunity**: Investigate reusing distributed weights across layers

**2. Attention Block (50.91ms, 2.3% of total, 42.59ms in mpi_attention_execute)**
- Stage-level breakdown:
  - `attention_scores`: 11.95ms (23.5%) - Computing Q·K^T and softmax
  - `rope_application`: 8.27ms (16.3%) - Rotary position embeddings
  - `qkv_projections`: 8.23ms (16.2%) - Q/K/V matrix projections
  - `output_projection`: 3.85ms (7.6%) - Final output matmul
  - `gqa_expansion`: 3.01ms (5.9%) - Grouped-query attention expansion
  - `weight_distribution`: 0.03ms (0.1%) - Minimal overhead
  
- Kernel-level breakdown:
  - `qk_matmul_unmasked`: 2.87ms - Q·K^T for snapshots
  - `scores_v_matmul`: 2.75ms - Attention weights × V
  - `qk_matmul_masked`: 1.77ms - Q·K^T with causal masking
  - `output_proj_matmul`: 1.42ms - Output projection
  - `softmax_all_heads`: 0.87ms - Softmax normalization
  
- MPI collective overhead:
  - `allreduce_output`: 1.17ms - Final aggregation
  - `allgather_attended_rows`: 0.76ms - Gather attended values
  - `allgatherv_*`: 0.18ms - Score gathering (minimal)

**3. RMSNorm (13.80ms, 0.6% of total across 49 calls)**
- `rmsnorm_distributed`: 9.52ms (69.0%) - Actual normalization
- `distribute_input`: 0.15ms (1.1%) - Input distribution
- `gather_output`: Included in linear operator gather timing

### Key Performance Observations

1. **Linear operator overhead is the primary bottleneck**:
   - Weight distribution (122ms) and gather (31ms) = 153ms overhead
   - Actual matmul (29ms) is only 16% of linear operator time
   - **Action item**: Investigate weight reuse and reduce redundant distribution

2. **Attention is well-optimized**:
   - Only 2.3% of total execution time
   - Kernel operations are efficient (all <3ms per layer)
   - MPI overhead is minimal (<2ms total per layer)
   
3. **lm_head projection anomaly**:
   - Single 517ms operation vs 347-365ms for FFN layers
   - Suspiciously slow for similar dimensions
   - **Action item**: Investigate backend selection for lm_head

4. **Quantization impact visible**:
   - Q8_0 model shows balanced compute vs overhead
   - Weight distribution overhead likely includes dequantization
   - Q4_0/Q6_K may show different characteristics

## Trace Categories

The instrumentation uses hierarchical categories for easy filtering:

- **`attention`**: High-level attention stages (depth 2)
- **`attention_kernel`**: Low-level attention kernels (depth 3)
- **`linear`**: Linear operator top-level (depth 1)
- **`linear_kernel`**: Linear kernels and weight ops (depth 2)
- **`linear_projection`**: Specific projection layers (depth 0)
- **`normalization`**: RMSNorm top-level (depth 1)
- **`norm_kernel`**: Normalization computation (depth 2)
- **`mpi_collective`**: All MPI collective operations (depth 2-3)
- **`prefill`**: Top-level pipeline stages (depth 0)

## Usage

Enable tracing with environment variable:
```bash
export LLAMINAR_PERF_TRACE=1
./run_llaminar.sh -- --benchmark -m model.gguf -p "prompt" -n 50
```

The trace is automatically printed to console and saved to `llaminar_trace.json` for visualization in chrome://tracing.

## Future Work

1. **Optimize linear operator overhead**:
   - Implement weight caching/reuse across layers
   - Reduce redundant weight distribution
   - Investigate fused distribute+matmul kernels

2. **Investigate lm_head anomaly**:
   - Profile backend selection for large output dimensions
   - Compare OpenBLAS vs COSMA for different shapes
   - Check for unexpected memory allocations

3. **Add adaptive backend tracing**:
   - Instrument `adaptiveMatMul` to show COSMA vs OpenBLAS selection
   - Track decision criteria and overhead
   - Measure backend switching costs

4. **Profile quantization formats**:
   - Compare Q4_0, Q6_K, Q8_0 performance
   - Measure dequantization overhead
   - Identify optimal quantization for different shapes

5. **Extend kernel-level tracing**:
   - Add tracing to COSMA internal operations
   - Instrument OpenBLAS threading decisions
   - Profile cache behavior and NUMA effects

## Related Files

- `src/PerformanceTracer.h/cpp`: Core tracing framework
- `src/utils/DebugEnv.h/cpp`: Environment configuration
- `src/operators/MPIAttentionOperator.cpp`: Attention instrumentation
- `src/operators/MPILinearOperator.cpp`: Linear instrumentation
- `src/operators/MPIRMSNormOperator.cpp`: Normalization instrumentation

## References

- Initial framework: `changelog/2025-10-15_performance_tracing_framework.md`
- Validation control: `changelog/2025-10-15_validation_compile_control.md`
