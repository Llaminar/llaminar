# Optimization Attempts Summary - October 15, 2025

## Session Goals
**Target**: Achieve 30 tok/s decode throughput (currently at ~13 tok/s, need 2.3× speedup)

## Optimizations Attempted

### 1. FFN Fusion ✅ IMPLEMENTED

**Approach**: Fuse gate and up projections in Feed-Forward Network
- Combined `W_gate` and `W_up` into single `W_fused` matrix [2*d_ff, d_model]
- Single matmul instead of two separate operations
- Reduces memory bandwidth and improves cache utilization

**Implementation**:
- Files: `src/QwenPipeline.{h,cpp}`, `src/QwenModelWeights.h`
- Control: `LLAMINAR_FFN_FUSION` environment variable (default: enabled)
- Weight concatenation in `QwenPipeline::fuseGateUpWeights()`
- Updated FFN kernel to split outputs after single matmul

**Results**:
- Baseline: 13.00 tok/s (decode phase)
- With FFN fusion: 13.25 tok/s (decode phase)  
- **Improvement: ~1.9%** (modest gain, below expectations)

**Analysis**:
- Benefit less than expected due to:
  - Weight caching already very effective (4.6× previous improvement)
  - Memory bandwidth bound (only 1% CPU efficiency)
  - Loading 511MB weights dominates over compute savings
  
**Status**: ✅ Complete, documented, tested, merged


### 2. Sequential Batch Processing ❌ ABANDONED

**Approach**: Process N sequences sequentially with shared weight loading
- Amortize 511MB weight loads across multiple sequences
- Expected 2-4× speedup for batch sizes 4-8

**Implementation Attempted**:
- Files: `src/SequentialBatchBenchmark.{h,cpp}`, `src/ArgumentParser.{h,cpp}`, `src/Main.cpp`
- Command-line: `--batch-size <n>` parameter
- Architecture: Loop through sequences, reusing pipeline instance

**Results**: ❌ FAILED - Fundamental architecture mismatch

**Problems Encountered**:
1. **State corruption**: Pipeline designed for single-sequence usage
   - KV cache accumulates across sequences
   - Position tracking (n_past_) carries over
   - Logits become corrupted (token IDs >>vocab_size)

2. **Attempted fixes**:
   - Reset position via `setSequencePosition(0)` - INSUFFICIENT
   - KV cache still contains residual data from previous sequence
   - No clean API to fully reset pipeline state between sequences

3. **Architectural issue**:
   - Pipeline holds persistent state (KV cache, position, intermediate activations)
   - Designed for autoregressive single-sequence generation
   - Reusing instance across sequences violates design assumptions

**Fundamental Limitation**:
- **Sequential batch processing provides NO benefit** over single sequence
  - Weight caching already happens within a sequence (via weight_cache_)
  - Sequential processing doesn't amortize anything new
  - Just adds complexity without performance gain

**Proper Solution Would Require**:
- **True parallel batching**: Process multiple tokens simultaneously
  - Reshape tensors: [batch_size, seq_len, hidden_dim]
  - Parallel matmuls across batch dimension
  - Complex refactoring of all kernels and attention mechanisms
  - Estimated effort: 2-3 weeks of development

**Decision**: Abandon sequential batch approach
- Doesn't provide promised benefits
- Architecture not designed for state reuse
- True parallel batching too complex for immediate goals

**Status**: ❌ Reverted (implementation not committed)


## Performance Analysis

### Current Bottlenecks Identified:

1. **Memory Bandwidth** (PRIMARY):
   - Only 1% CPU efficiency during decode
   - 511MB model weights for Qwen-0.5B-Q8
   - Memory-bound, not compute-bound
   - OpenBLAS/COSMA can't help without data

2. **Weight Loading**:
   - Already optimized via weight_cache_ (4.6× improvement)
   - Further gains require:
     - Quantization (already using Q8)
     - Model architecture changes (not in scope)
     - GPU acceleration (hardware requirement)

3. **Single Token Decode**:
   - Batch size = 1 during autoregressive generation
   - Can't parallelize across tokens (sequential dependency)
   - True batching requires processing multiple independent sequences simultaneously

### Why 30 tok/s is Challenging:

Current: 13.25 tok/s  
Target: 30 tok/s  
Gap: 2.27× speedup needed

**Realistic optimizations remaining**:
- Attention optimizations: ~5-10% potential
- RMSNorm fusion: ~2-5% potential
- Better NUMA placement: ~3-8% potential
- **Total realistic gain**: ~15-25% → ~15-16 tok/s

**To reach 30 tok/s would require**:
- GPU acceleration (10-50× speedup) OR
- Model quantization below Q8 (2-4× speedup, quality loss) OR
- True parallel batching (2-8× aggregate throughput, not per-sequence)

## Recommendations

### Short-term (Next Steps):
1. **Profile actual bottlenecks**: Use perf/VTune to identify hot spots
2. **Attention kernel optimization**: Focus on the most expensive operation
3. **NUMA optimization audit**: Verify first-touch working correctly
4. **Benchmark in Release mode**: Ensure Debug overhead not skewing results

### Medium-term (1-2 weeks):
1. **True parallel batching** (if aggregate throughput is the goal):
   - Reshape all tensors for batch dimension
   - Update all kernels for batched operations
   - Enable processing N independent sequences simultaneously
   - Benefit: N× aggregate throughput (not per-sequence speedup)

2. **GPU acceleration** (if per-sequence speed is the goal):
   - Port critical kernels to CUDA/ROCm
   - Utilize tensor cores for matmuls
   - Benefit: 10-50× speedup for compute-bound operations

### Long-term (Architecture):
1. **Continuous batching**: Dynamic batching like vLLM
2. **FlashAttention**: Memory-efficient attention implementation
3. **Speculative decoding**: Parallel token generation with verification
4. **KV cache quantization**: Reduce memory bandwidth for attention

## Files Modified (To Be Reverted)

Batch processing implementation to remove:
- `src/SequentialBatchBenchmark.{h,cpp}` - DELETE
- `src/ArgumentParser.{h,cpp}` - REVERT --batch-size changes
- `src/Main.cpp` - REVERT batch mode section
- `CMakeLists.txt` - REVERT SequentialBatchBenchmark.cpp addition

FFN fusion to keep:
- `src/QwenPipeline.{h,cpp}` - KEEP (working optimization)
- `src/QwenModelWeights.h` - KEEP (weight fusion support)

## Lessons Learned

1. **Measure before optimizing**: Profile to find actual bottlenecks
2. **Architecture constraints matter**: Pipeline designed for single-sequence use
3. **Weight caching is already effective**: 4.6× previous gain means diminishing returns
4. **Memory-bound vs compute-bound**: CPU optimizations won't help bandwidth limits
5. **Sequential ≠ Parallel batching**: Sequential processing provides no amortization benefit

## Conclusion

- **FFN Fusion**: ✅ Success (~1.9% improvement, incremental progress)
- **Batch Processing**: ❌ Failed (architectural mismatch, no benefit)
- **30 tok/s Goal**: Not achievable with CPU-only optimizations
  - Need GPU acceleration or true parallel batching for target performance
  - Realistic ceiling with current architecture: ~15-16 tok/s

**Recommended next focus**: Profile-guided optimization of attention kernel and consideration of GPU acceleration for meaningful speedup.
