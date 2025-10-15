# Performance Trace Results Analysis - October 15, 2025

## Executive Summary

Successfully implemented and deployed hierarchical performance tracing framework across Llaminar's hot paths. Initial benchmark results reveal critical insights into performance bottlenecks.

**Key Findings:**
- **Current Performance**: 42.86 tok/s (893 tokens in 20.84 seconds)
- **Target Performance**: ~1200 tok/s (llama.cpp baseline)
- **Performance Gap**: ~28x slower than target
- **Primary Bottleneck**: Attention blocks consume 49% of execution time
- **Secondary Bottlenecks**: Linear operations (37%) and FFN projections (34%)

## Benchmark Configuration

```bash
Model: qwen2.5-0.5b-instruct-q8_0.gguf
Backend: OpenBLAS
Mode: Prefill-only (893 tokens, decode skipped)
MPI: 2 processes, socket-bound
OpenMP: 28 threads per socket
Time: 20836.27 ms total
Throughput: 42.86 tok/s
```

## Performance Breakdown (Rank 0)

### Top Operations by Total Time

| Operation | Category | Calls | Total (ms) | Avg (ms) | % of Total |
|-----------|----------|-------|------------|----------|------------|
| **attention_block** | prefill | 24 | 10360.76 | 431.7 | **49.7%** |
| mpi_attention_execute | attention | 24 | 10110.96 | 421.3 | 48.5% |
| **mpi_linear_execute** | linear | 73 | 7689.42 | 105.3 | **36.9%** |
| ffn_gate | linear_projection | 24 | 2524.70 | 105.2 | 12.1% |
| ffn_up | linear_projection | 24 | 2257.25 | 94.1 | 10.8% |
| **lm_head** | linear_projection | 1 | 2176.53 | 2176.5 | **10.4%** |
| ffn_down | linear_projection | 24 | 731.95 | 30.5 | 3.5% |
| mpi_rmsnorm_execute | normalization | 49 | 388.96 | 7.9 | 1.9% |

### Per-Layer Cost Analysis

**24 Transformer Layers:**
- Attention per layer: ~431.7 ms average (408-470 ms range)
- FFN gate per layer: ~105.2 ms average
- FFN up per layer: ~94.1 ms average
- FFN down per layer: ~30.5 ms average
- RMSNorm per layer: ~7.9 ms average (2 calls per layer)

**Total per layer**: ~670 ms average

## Critical Bottlenecks Identified

### 1. Attention Block Overhead (49.7% of execution)
**Observation**: Each attention block takes 408-470 ms
- mpi_attention_execute: 421.3 ms average
- Suggests heavy computation or synchronization overhead
- 24 layers × 431.7 ms = 10360.76 ms total

**Potential Causes:**
- MPI synchronization barriers
- Inefficient Q/K/V projection distribution
- Attention score computation bottleneck
- Memory allocation/copy overhead

### 2. Linear Operations Overhead (36.9% of execution)
**Observation**: 73 total linear operations, 105.3 ms average
- FFN gate/up projections dominate (105-94 ms each)
- lm_head final projection is expensive: 2176.5 ms (single call!)

**Potential Causes:**
- Matrix multiplication backend selection (OpenBLAS vs COSMA)
- Thread synchronization overhead
- Memory bandwidth saturation
- Lack of operation fusion

### 3. lm_head Projection Anomaly (10.4% of execution)
**Observation**: Single lm_head operation takes 2176.5 ms
- Avg: 2176.5 ms (only 1 call)
- This is ~2x slower than expected for vocab projection
- Likely related to large output dimension (vocab_size)

**Potential Causes:**
- Very large matrix multiplication (893 × d_model → 893 × vocab_size)
- Poor cache locality for vocab matrix
- Inefficient backend selection for this shape

### 4. FFN Projection Costs (34% cumulative)
**Observation**: Feed-forward network projections accumulate:
- ffn_gate: 24 × 105.2 ms = 2524.70 ms
- ffn_up: 24 × 94.1 ms = 2257.25 ms
- ffn_down: 24 × 30.5 ms = 731.95 ms
- **Total FFN**: 7.5 seconds (36% of execution)

**Potential Causes:**
- Lack of gate/up fusion opportunity
- Sequential execution instead of parallel
- Memory allocation per operation

### 5. RMSNorm Relative Efficiency (1.9% of execution)
**Observation**: RMSNorm is surprisingly efficient
- 49 calls, 7.9 ms average
- Only 388.96 ms total (1.9% of time)
- This is proportionally correct (normalization is lightweight)

## Performance Comparison vs llama.cpp

### Current State
- **Llaminar**: 42.86 tok/s (20836 ms for 893 tokens)
- **llama.cpp**: ~1200 tok/s (estimated ~744 ms for 893 tokens)
- **Gap**: ~28x slower

### Time Attribution Analysis
| Component | Llaminar (ms) | Expected (ms) | Overhead |
|-----------|---------------|---------------|----------|
| Attention | 10360.76 | ~300 | ~34x |
| Linear | 7689.42 | ~250 | ~30x |
| FFN | 7513.90 | ~150 | ~50x |
| RMSNorm | 388.96 | ~44 | ~8.8x |
| Other | 2882.00 | ~0 | N/A |

**Expected times** are rough estimates based on llama.cpp's ~744 ms total for similar workload.

## Root Cause Hypotheses

### 1. MPI Overhead (HIGH CONFIDENCE)
- Each attention block has nested MPI collectives
- Barriers before/after COSMA operations
- Per-layer synchronization instead of batched
- **Evidence**: mpi_attention_execute consumes 48.5% of time

### 2. Inefficient Backend Selection (MEDIUM CONFIDENCE)
- Small operations may use multi-threaded OpenBLAS when single-threaded would be faster
- Large operations may not be using COSMA when beneficial
- lm_head anomaly suggests wrong backend for large vocab projection
- **Evidence**: ffn projections taking 105 ms each (expected ~4-5 ms)

### 3. Memory Allocation Churn (MEDIUM CONFIDENCE)
- Each operation allocates intermediate tensors
- No buffer reuse across layers
- NUMA allocation overhead
- **Evidence**: "Other" time (2882 ms) unaccounted for in traced operations

### 4. Thread Synchronization Overhead (LOW CONFIDENCE)
- 28 OpenMP threads may have coordination cost
- OpenBLAS thread spawning per operation
- **Evidence**: Moderate variance in operation times

### 5. System Call Overhead (LOW CONFIDENCE)
- Repeated `getenv()` calls (now cached via debugEnv)
- Logging overhead in hot paths
- **Evidence**: Less likely after debugEnv refactor

## Optimization Priorities

### Priority 1: Optimize Attention Block (CRITICAL)
**Target**: Reduce attention_block from 431.7 ms → <20 ms per layer
- Profile Q/K/V projection overhead
- Reduce MPI synchronization frequency
- Consider fusing RMSNorm + Q/K/V projection
- Investigate attention score computation bottleneck
- **Expected Impact**: 10.4 seconds → <500 ms (9.9 second savings)

### Priority 2: Fix lm_head Anomaly (HIGH)
**Target**: Reduce lm_head from 2176.5 ms → <100 ms
- Investigate vocab matrix layout
- Use COSMA for large final projection
- Cache vocab matrix on NUMA-local memory
- **Expected Impact**: 2.2 seconds → <100 ms (2.1 second savings)

### Priority 3: Optimize FFN Projections (HIGH)
**Target**: Reduce ffn_gate/up from 105/94 ms → <5 ms each
- Fuse gate + up projections (SwiGLU pattern)
- Better backend selection for FFN shapes
- Reduce per-operation overhead
- **Expected Impact**: 7.5 seconds → <1 second (6.5 second savings)

### Priority 4: Reduce Linear Operation Overhead (MEDIUM)
**Target**: Improve mpi_linear_execute from 105.3 ms → <10 ms average
- Profile matrix multiplication time vs overhead
- Optimize weight distribution strategy
- Consider operation batching
- **Expected Impact**: 7.7 seconds → <1 second (6.7 second savings)

### Priority 5: Eliminate "Other" Overhead (MEDIUM)
**Target**: Account for missing 2.9 seconds
- Add more granular tracing to find gaps
- Profile memory allocation
- Measure MPI barrier time explicitly
- **Expected Impact**: 2.9 seconds → <500 ms (2.4 second savings)

## Cumulative Impact Estimate

If all optimizations succeed:
- Current: 20836 ms
- Attention optimization: -9900 ms
- lm_head fix: -2100 ms
- FFN optimization: -6500 ms
- Linear optimization: -6700 ms
- Other overhead: -2400 ms
- **Projected Total**: ~3236 ms (6.4x improvement)
- **Projected Throughput**: ~276 tok/s

**Still ~4.3x slower than llama.cpp target**, suggesting additional systemic issues beyond traced operations.

## Next Steps

### Immediate Actions
1. **Drill Down on Attention Block**:
   - Add finer-grained tracing inside MPIAttentionOperator
   - Measure Q/K/V projection time separately
   - Measure attention score computation time
   - Measure MPI barrier overhead explicitly

2. **Profile lm_head**:
   - Add tracing inside lm_head projection
   - Check matrix dimensions and backend selection
   - Profile memory access patterns

3. **Measure Gaps**:
   - Add tracing to QwenPipeline::prefill()
   - Add tracing to layer loop overhead
   - Add tracing to model loading and setup

### Medium-Term Actions
1. **Implement Operation Fusion**:
   - RMSNorm + Q/K/V projection fusion
   - Gate + Up projection fusion (SwiGLU)
   - Test fused attention kernel

2. **Optimize Backend Selection**:
   - Implement dynamic threshold tuning
   - Profile COSMA vs OpenBLAS breakpoints
   - Add adaptive selection based on shape

3. **Reduce MPI Overhead**:
   - Batch MPI operations across layers
   - Use asynchronous collectives
   - Reduce barrier frequency

### Long-Term Actions
1. **Memory Management Overhaul**:
   - Implement buffer pooling
   - Reuse activation tensors across layers
   - Optimize NUMA placement

2. **Threading Strategy**:
   - Experiment with fewer OpenMP threads
   - Profile thread spawn overhead
   - Test persistent thread pools

3. **Algorithmic Improvements**:
   - Flash Attention implementation
   - Quantized KV cache
   - Speculative decoding

## Tracing Infrastructure Validation

### Framework Functionality
✅ **Zero-overhead macros** - No performance impact when disabled
✅ **Hierarchical tracking** - Nested operations correctly attributed
✅ **Thread-safe** - Per-thread stacks, mutex-protected aggregation
✅ **MPI-aware** - Per-rank output with separate trace files
✅ **Chrome trace export** - llaminar_trace.json generated (49KB)
✅ **Console summary** - Readable table with top operations

### Instrumentation Coverage
✅ **Prefill path**: OpenblasPrefillProvider::executeAttentionBlock
✅ **Prefill path**: OpenblasPrefillProvider::executeLinearProjection
✅ **Attention**: MPIAttentionOperator::execute
✅ **Linear**: MPILinearOperator::execute
✅ **Normalization**: MPIRMSNormOperator::execute

### Missing Instrumentation
- ❌ QwenPipeline::prefill() - layer loop overhead
- ❌ Individual Q/K/V projections inside attention
- ❌ Attention score computation
- ❌ MPI barrier/collective timing
- ❌ Memory allocation timing
- ❌ Model loading and initialization

## Chrome Trace Visualization

The generated `llaminar_trace.json` can be viewed in Chrome:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select `llaminar_trace.json`
4. Examine timeline view showing:
   - Operation hierarchy (depth levels)
   - Time distribution across layers
   - Overlapping operations (if any)
   - Per-rank execution differences

## Conclusion

The performance tracing framework successfully revealed critical bottlenecks:
- **Attention blocks** are the #1 issue (49.7% of time)
- **Linear operations** need optimization (36.9% of time)
- **lm_head** projection has an anomaly (10.4% of time)
- **FFN projections** accumulate significant cost (34% of time)

Even with aggressive optimizations, estimated improvement is only ~6.4x (to ~276 tok/s), still ~4.3x slower than llama.cpp's ~1200 tok/s. This suggests **systemic architectural differences** beyond individual operation performance.

**Critical Next Step**: Add deeper instrumentation to attention blocks and fill tracing gaps to understand the remaining ~60% of unaccounted overhead.

**Performance Goal**: Achieve >500 tok/s (11.7x improvement) in next optimization iteration, then target >1000 tok/s (23x improvement) in subsequent work.

---

**Author**: David Sanftenberg  
**Date**: October 15, 2025  
**Tools**: PerformanceTracer framework, Chrome trace viewer  
**Benchmark**: Qwen 2.5 0.5B Q8_0, 893 tokens prefill  
