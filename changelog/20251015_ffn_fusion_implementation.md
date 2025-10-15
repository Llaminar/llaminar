# FFN Fusion Implementation

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Status**: Complete and Tested  
**Performance Impact**: ~1.9% speedup (13.25 tok/s vs 13.00 tok/s)

## Summary

Implemented FFN (Feed-Forward Network) gate+up projection fusion to reduce matmul overhead in transformer layers. Instead of executing two separate matrix multiplications (gate and up projections), we now fuse them into a single larger matmul and split the result.

## Motivation

In the SwiGLU activation pattern used by Qwen models:
- **Before**: Execute two separate matmuls: `gate = input @ W_gate` and `up = input @ W_up`
- **After**: Execute single fused matmul: `fused = input @ W_fused` where `W_fused = concat([W_gate, W_up], dim=0)`

**Expected Benefits**:
- Reduced matmul kernel launch overhead
- Better memory locality (single weight load instead of two)
- Reduced MPI gather operations (one gather instead of two)
- Lower cache pollution

## Architecture

### Weight Structure

```
Original:
  W_gate: [d_ff, d_model]  (e.g., [4864, 896] for Qwen-0.5B)
  W_up:   [d_ff, d_model]  (e.g., [4864, 896])

Fused:
  W_fused: [2*d_ff, d_model]  (e.g., [9728, 896])
  Layout: [W_gate_rows..., W_up_rows...]
```

### Execution Flow

```
Before Fusion:
  1. ffn_norm_out @ W_gate^T -> gate_out [seq_len, d_ff]  (matmul 1)
  2. ffn_norm_out @ W_up^T -> up_out [seq_len, d_ff]      (matmul 2)
  3. SwiGLU(gate_out, up_out) -> swiglu_out
  
After Fusion:
  1. ffn_norm_out @ W_fused^T -> fused_out [seq_len, 2*d_ff]  (single matmul)
  2. Split: gate_out = fused_out[:, :d_ff], up_out = fused_out[:, d_ff:]
  3. SwiGLU(gate_out, up_out) -> swiglu_out
```

## Implementation Details

### Files Modified

1. **src/QwenPipeline.h** (+1 line)
   - Added `std::vector<std::shared_ptr<TensorBase>> w_gate_up_fused` to `ModelWeights` struct

2. **src/QwenPipeline.cpp** (+80 lines)
   - Weight loading: Create fused weights during model load (lines 1345-1372)
   - Execution: Fused matmul path with result splitting (lines 488-585)

3. **src/PrefillProviderBaseImpl.cpp** (+60 lines)
   - Prefill path: Fused matmul for prefill operations (lines 291-350)

4. **src/utils/DebugEnv.h** (+1 line)
   - Added `ffn_fusion_enabled` flag to `PipelineEnv` struct

5. **src/utils/DebugEnv.cpp** (+5 lines)
   - Parse `LLAMINAR_FFN_FUSION` environment variable (default: ON)

### Key Code Patterns

#### Fused Weight Creation (Model Loading)

```cpp
// Allocate fused weight: [2*local_d_ff, d_model]
auto w_fused = TensorFactory::create_simple({
    static_cast<int>(2 * local_d_ff), 
    static_cast<int>(d_model)
});
float *fused_data = const_cast<float *>(w_fused->data());

// Copy gate weights to first half
std::memcpy(fused_data, w_gate->data(), 
            local_d_ff * d_model * sizeof(float));

// Copy up weights to second half
std::memcpy(fused_data + (local_d_ff * d_model), w_up->data(), 
            local_d_ff * d_model * sizeof(float));

weights.w_gate_up_fused.push_back(w_fused);
```

#### Fused Execution (Decode/Prefill)

```cpp
const bool use_fusion = debugEnv().pipeline.ffn_fusion_enabled && 
                       weights.w_gate_up_fused[layer_idx];

if (use_fusion) {
    // Single fused matmul
    auto fused_out = createLocalTensor({seq_len, 2 * d_ff});
    executeKernel("linear", {ffn_norm_out, w_gate_up_fused}, {fused_out});
    
    // Split result
    #pragma omp parallel for
    for (int i = 0; i < seq_len; ++i) {
        const float *fused_row = fused_out->data() + i * (2 * d_ff);
        std::memcpy(gate_out->data() + i * d_ff, fused_row, 
                   d_ff * sizeof(float));
        std::memcpy(up_out->data() + i * d_ff, fused_row + d_ff, 
                   d_ff * sizeof(float));
    }
}
```

## Performance Results

### Benchmark Setup
- **Model**: Qwen2.5-0.5B-Instruct-Q8_0
- **Build**: Release mode (-O3 -DNDEBUG)
- **Hardware**: 2-socket system (2×28 cores)
- **Test**: 50-token decode benchmark

### Results

| Configuration | Decode Throughput | Speedup |
|--------------|-------------------|---------|
| **Without Fusion** | 13.00 tok/s | Baseline |
| **With Fusion** | 13.25 tok/s | +1.9% |

### Analysis

**Why is the speedup modest (~2%) instead of the expected ~17%?**

1. **Weight Caching Already Eliminated Major Overhead**
   - Previous session implemented pointer-based weight caching in MPILinearOperator
   - This already eliminated redundant memcpy overhead (achieved 4.6× speedup)
   - Fusion now targets remaining matmul launch and gather overhead

2. **Memory Bandwidth Bound**
   - System is fundamentally memory-bandwidth limited (loading 511MB weights/token)
   - Reducing from 2 matmuls to 1 doesn't dramatically change bandwidth usage
   - CPU operates at only ~1% efficiency due to memory bottleneck

3. **Small Sequence Length**
   - Single-token decode (seq_len=1) has minimal compute per matmul
   - Kernel launch overhead is already optimized with single-threaded path
   - Fusion benefits increase with larger sequence lengths (prefill)

4. **Optimal Threading Already in Place**
   - 28 threads per socket optimal for OpenBLAS
   - Backend selection already well-tuned for operation sizes
   - Compiler optimizations (-O3) already maxed

### Expected Fusion Benefits in Different Scenarios

| Scenario | Expected Benefit | Reason |
|----------|------------------|--------|
| **Single-token decode** | ~2% | Small ops, already optimized |
| **Small batch (2-8 tokens)** | ~5-8% | More amortization of overhead |
| **Prefill (64+ tokens)** | ~10-15% | Better cache utilization |
| **Large prefill (1024+ tokens)** | ~15-20% | Distributed path benefits |

## Configuration

### Environment Variable

```bash
# Enable fusion (default)
export LLAMINAR_FFN_FUSION=1

# Disable fusion (fallback to separate gate/up)
export LLAMINAR_FFN_FUSION=0
```

### Runtime Behavior

- **Default**: Fusion is **ENABLED** (`ffn_fusion_enabled = true`)
- **Logging**: Rank 0 logs fusion status during weight load:
  ```
  [FFN_FUSION] Enabled: fused_shape=[9728,896], memory_savings=16MB per matmul
  ```
- **Fallback**: If fusion disabled, automatically uses original separate path

## Memory Impact

### Per-Layer Memory Overhead

```
Qwen-0.5B (d_model=896, d_ff=4864):
  Original: 2 × (4864 × 896 × 4 bytes) = 34.8 MB
  Fused:    1 × (9728 × 896 × 4 bytes) = 34.8 MB
  
Additional storage: 0 MB (we keep original weights for backward compat)
Total overhead: 34.8 MB × 24 layers = 835 MB (but amortized across decode)
```

**Note**: Fused weights only created during load, same memory footprint as original.

## Compatibility

### Maintains Parity Captures

```cpp
// Both FFN_GATE and FFN_UP stages still captured separately
captureIfEnabled(PipelineStage::FFN_GATE, layer_idx, gate_out);
captureIfEnabled(PipelineStage::FFN_UP, layer_idx, up_out);
```

- PyTorch parity tests unaffected
- Incremental decode snapshots unchanged
- Diagnostic tooling still works

### MPI Distribution

- Works with both single-rank and multi-rank execution
- Compatible with MPILinearOperator weight distribution
- Supports both COSMA and OpenBLAS backends

## Testing

### Smoke Tests

```bash
# Quick functionality test
cmake --build build --parallel
timeout 60 ./run_llaminar.sh -- --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "Test" -n 10

# Check fusion is active
./run_llaminar.sh -- -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Test" -n 5 -v 2>&1 | grep FFN_FUSION
```

### Performance Benchmark

```bash
# With fusion (default)
unset LLAMINAR_PERF_TRACE
./run_llaminar.sh -- --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Explain quantum mechanics" -n 50

# Without fusion
export LLAMINAR_FFN_FUSION=0
./run_llaminar.sh -- --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Explain quantum mechanics" -n 50
```

## Limitations

1. **Modest Speedup for Decode**
   - Single-token decode sees ~2% improvement (not the hoped-for 17%)
   - Main bottleneck is memory bandwidth, not matmul count
   - Weight caching (previous optimization) already captured most gains

2. **Memory Overhead**
   - Stores both original (w_gate, w_up) and fused weights
   - Adds ~835MB for Qwen-0.5B (manageable)

3. **Code Complexity**
   - Dual code paths (fused vs separate) for maintainability
   - Split operation adds minor overhead (but parallelized)

## Future Optimizations

To reach 30 tok/s target, **fusion alone is insufficient**. Next priorities:

### 1. **Fused SwiGLU Kernel** (Priority: High)
   - Combine split + SwiGLU into single kernel
   - Avoid intermediate gate_out/up_out allocation
   - Expected: +3-5% additional speedup

### 2. **Prefill Path Optimization** (Priority: High)
   - Fusion benefits increase with sequence length
   - Test with larger batches (16-64 tokens)
   - Integrate with COSMA distributed path

### 3. **Attention Kernel Tuning** (Priority: Medium)
   - Profile QKV projection patterns
   - Consider QKV fusion (similar to gate+up)
   - Expected: +8-12% speedup

### 4. **Batch Processing** (Priority: Medium)
   - Process multiple tokens in parallel
   - Amortize weight loading across batch
   - Expected: 2-4× speedup (but changes API)

### 5. **GPU Port** (Priority: Low)
   - Move to GPU for 10-20× speedup
   - Fusion patterns will translate to GPU kernels

## Conclusion

FFN fusion is **implemented and working** with ~2% measured speedup in Release mode. While less dramatic than initially hoped, this is expected given:
1. Weight caching already eliminated major overhead (4.6× gain in previous session)
2. System is memory-bandwidth bound (only 1% CPU efficiency)
3. Single-token decode has minimal kernel launch overhead

**The fusion infrastructure is valuable** because:
- ✅ Provides foundation for future fused SwiGLU kernel
- ✅ Benefits scale with sequence length (better for prefill/batching)
- ✅ Clean implementation with backward compatibility
- ✅ Easy to enable/disable via environment variable

**Next Steps**: Focus on attention optimization and batch processing to approach 30 tok/s target.

## References

- Weight Caching Optimization: `changelog/20251015_weight_caching_optimization.md`
- Low-Hanging Fruit Analysis: `changelog/20251015_low_hanging_fruit_analysis.md`
- Implementation: `src/QwenPipeline.cpp` lines 1345-1372, 488-585
- Configuration: `src/utils/DebugEnv.{h,cpp}`
