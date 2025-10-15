# Performance Optimization Roadmap - Next Actions

## Current State
- **Throughput**: 42.86 tok/s (893 tokens, 20.84s)
- **Target**: ~1200 tok/s (llama.cpp baseline)
- **Gap**: 28x slower

## Critical Bottlenecks (From Trace)

### 1. Attention Blocks: 49.7% of execution time
- 24 layers × 431.7 ms = 10.36 seconds
- Each layer: 408-470 ms range
- **Priority**: CRITICAL

### 2. Linear Operations: 36.9% of execution time
- 73 operations × 105.3 ms = 7.69 seconds
- lm_head anomaly: 2176.5 ms (single call)
- **Priority**: HIGH

### 3. FFN Projections: 34% cumulative
- ffn_gate: 2524.7 ms (24 × 105.2 ms)
- ffn_up: 2257.3 ms (24 × 94.1 ms)
- ffn_down: 732.0 ms (24 × 30.5 ms)
- **Priority**: HIGH

## Immediate Next Steps

### Step 1: Deep Dive into Attention Block
**Goal**: Understand the 431.7 ms per-layer cost

Add finer-grained tracing to `MPIAttentionOperator::execute()`:
```cpp
// Inside execute(), add:
PERF_TRACE_SCOPE_CAT("qkv_projection", "attention");  // Q/K/V computation
PERF_TRACE_SCOPE_CAT("attention_scores", "attention");  // Score computation
PERF_TRACE_SCOPE_CAT("attention_output", "attention");  // Output projection
PERF_TRACE_SCOPE_CAT("mpi_barriers", "synchronization");  // MPI overhead
```

**Expected Insights**:
- Which sub-operation dominates (Q/K/V vs scores vs output)
- MPI barrier overhead measurement
- Memory allocation hotspots

### Step 2: Investigate lm_head Anomaly
**Goal**: Reduce 2176.5 ms → <100 ms

Profile the final vocabulary projection:
- Check matrix dimensions (should be 893 × d_model → 893 × vocab_size)
- Verify backend selection (should use COSMA for large vocab)
- Measure memory access patterns
- Check NUMA locality of vocab matrix

**Actions**:
```cpp
// In OpenblasPrefillProvider::executeLinearProjection(), add:
if (operation_name == "lm_head") {
    PERF_TRACE_SCOPE_CAT("lm_head_matmul", "linear");
    // Detailed profiling for lm_head specifically
}
```

### Step 3: Fill Tracing Gaps
**Goal**: Account for missing 2.9 seconds of "other" overhead

Add top-level tracing:
```cpp
// In QwenPipeline::prefill():
PERF_TRACE_SCOPE_CAT("prefill_total", "pipeline");

// Inside layer loop:
PERF_TRACE_SCOPE_CAT("layer_overhead", "pipeline");  // Loop mechanics
```

**Expected Findings**:
- Layer loop overhead
- Memory allocation between operations
- Logging/debugging overhead
- MPI communication not captured

### Step 4: Add Explicit MPI Timing
**Goal**: Measure synchronization cost separately

Add barrier timing:
```cpp
// Before MPI collectives:
PERF_TRACE_BEGIN("mpi_barrier");
MPI_Barrier(MPI_COMM_WORLD);
PERF_TRACE_END();
```

**Expected Results**:
- Quantify MPI overhead vs computation
- Identify unnecessary barriers
- Find opportunities for async communication

## Quick Wins to Try First

### Win 1: Disable Heavy Validation in Benchmark Mode
Current state likely has validation enabled. Try:
```bash
export LLAMINAR_ATTN_VALIDATE_OUTPUT=0
export LLAMINAR_ATTN_VALIDATE_PROJ=0
./run_llaminar.sh --benchmark -m model.gguf -n 0
```

**Expected Impact**: 5-10% improvement (if validation is on)

### Win 2: Adjust OpenBLAS Threading
Current: 28 threads per socket may have overhead. Try:
```bash
export OMP_NUM_THREADS=14  # Half threads
./run_llaminar.sh --benchmark -m model.gguf -n 0
```

**Expected Impact**: 10-20% improvement (reduced thread coordination)

### Win 3: Test COSMA for lm_head
Force COSMA for large final projection:
```bash
export LLAMINAR_COSMA_PREFILL_THRESHOLD=512  # Lower threshold
./run_llaminar.sh --benchmark -m model.gguf -n 0
```

**Expected Impact**: 20-30% improvement on lm_head (if it switches)

### Win 4: Disable Tracing Overhead
Recompile without tracing for baseline:
```bash
cmake -B build -S . -DLLAMINAR_ENABLE_PERF_TRACE=OFF
cmake --build build --parallel
./run_llaminar.sh --benchmark -m model.gguf -n 0
```

**Expected Impact**: 1-2% improvement (tracing overhead)

## Medium-Term Optimizations

### Optimization 1: Fuse RMSNorm + Q/K/V Projection
RMSNorm is only 1.9% of time but executes before every attention block.

**Implementation**:
- Create fused kernel: `RMSNorm(x, weight) → Q/K/V projections`
- Eliminate intermediate tensor allocation
- Reduce memory bandwidth requirements

**Expected Impact**: 5-10% overall improvement

### Optimization 2: SwiGLU Fusion (Gate + Up)
FFN uses separate gate/up projections but they could be fused.

**Implementation**:
```cpp
// Instead of:
gate = linear(x, wgate)
up = linear(x, wup)
// Do:
gate_up = linear(x, [wgate | wup])  // Fused weight matrix
```

**Expected Impact**: 15-20% overall improvement

### Optimization 3: Reduce MPI Barriers
Profile shows frequent MPI synchronization.

**Implementation**:
- Use MPI_Ibarrier (non-blocking) where possible
- Batch operations across layers before sync
- Overlap computation with communication

**Expected Impact**: 20-30% overall improvement

## Long-Term Goals

### Goal 1: Flash Attention
Replace standard attention with Flash Attention algorithm:
- Reduces memory bandwidth by 5-10x
- Enables longer context windows
- Significant speedup for attention-heavy workloads

**Expected Impact**: 2-3x overall improvement

### Goal 2: Quantized KV Cache
Use 4-bit or 8-bit quantization for K/V cache:
- Reduces memory bandwidth
- Enables larger batch sizes
- Minimal accuracy loss

**Expected Impact**: 1.5-2x improvement on decode

### Goal 3: Multi-Query Attention (MQA)
If model supports it, use MQA instead of GQA:
- Reduces KV cache memory
- Faster attention computation
- Better scaling with batch size

**Expected Impact**: 1.5x improvement on attention

## Validation Strategy

After each optimization:

1. **Run benchmark**: `./run_llaminar.sh --benchmark -m model.gguf -n 0`
2. **Check trace**: Verify operation times reduced
3. **Run parity tests**: Ensure correctness maintained
4. **Document results**: Update changelog with findings

**Success Criteria**:
- Phase 1: >100 tok/s (2.3x improvement) ✅ Achievable with quick wins
- Phase 2: >500 tok/s (11.7x improvement) ✅ Medium-term optimizations
- Phase 3: >1000 tok/s (23.3x improvement) ⚠️ Requires Flash Attention

## Files to Edit

### High-Priority Instrumentation
1. `src/operators/MPIAttentionOperator.cpp` - Add sub-operation tracing
2. `src/qwen_pipeline.cpp` - Add layer loop tracing
3. `src/OpenblasPrefillProvider.cpp` - Add lm_head specific profiling

### Optimization Targets
1. `src/operators/MPIAttentionOperator.cpp` - Attention optimizations
2. `src/MatmulBackendSelection.cpp` - Better backend selection
3. `src/qwen_pipeline.cpp` - Layer batching and fusion

### Configuration
1. `run_llaminar.sh` - Experiment with threading configurations
2. `src/utils/DebugEnv.cpp` - Add validation disable flags

## How to Use This Document

1. **Start with Quick Wins**: Test each quick win sequentially, document results
2. **Deep Dive**: Add finer-grained tracing, re-run benchmark
3. **Optimize**: Implement medium-term optimizations based on trace data
4. **Validate**: Run parity tests after each change
5. **Iterate**: Repeat until target performance achieved

---

**Next Command to Run**:
```bash
# Quick Win #1: Test with validation disabled
export LLAMINAR_ATTN_VALIDATE_OUTPUT=0
export LLAMINAR_ATTN_VALIDATE_PROJ=0
export LLAMINAR_PERF_TRACE=1
./run_llaminar.sh --benchmark -m ./models/qwen2.5-0.5b-instruct-q8_0.gguf -n 0
```

**Expected Result**: Baseline improvement of 5-10%, clearer trace without validation overhead

---

**Author**: David Sanftenberg  
**Date**: October 15, 2025  
**Status**: Ready for immediate action  
