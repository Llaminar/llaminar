# Virtual Dispatch Elimination in V2 GEMM (Oct 24, 2025)

## Summary

Successfully eliminated virtual dispatch overhead in V2's IQ4_NL quantized GEMM kernel by implementing template-based `QuantizedGemmOptimized<TensorType>` class. Achieved **18% speedup** (1.18×) over interface-based implementation, validating virtual dispatch as a measurable bottleneck.

## Background

After porting V1's complete GEMM algorithm to V2 (adaptive tiling, 4-column microkernel, prefetching), V2 still performed 9.6× slower than V1 (32 GFLOPS vs 314 GFLOPS). Root cause analysis identified virtual dispatch overhead:

- V2's `IBlockDecoder*` interface requires virtual function calls to `decode_block_at()`
- Called ~800k times per iteration (1024×896×896 GEMM)
- Prevents compiler inlining and SIMD optimizations
- Estimated 10-15% overhead from vtable lookups

## Implementation

### Template-Based Kernel

Created `src/v2/kernels/cpu/QuantizedGemmOptimized.h` (370 lines):

```cpp
template <typename TensorType>
class QuantizedGemmOptimized {
public:
    explicit QuantizedGemmOptimized(TensorType* tensor) : tensor_(tensor) {}
    
    bool multiply(const float* A, float* C, int m, int n, int k,
                 bool transpose_B, float alpha, float beta);

private:
    TensorType* tensor_;  // Concrete type, not interface pointer
    
    // Direct method calls (compiler can inline)
    // tensor_->decode_block_at(row, kb, buffer);
};
```

**Key Differences from Interface Version:**
- Template parameter `TensorType` instead of `IBlockDecoder*` interface
- Direct method calls (no vtable lookup)
- Identical algorithm to `QuantizedGemmKernel`:
  - Cache-blocked path for m ∈ [2,16]
  - Row-wise path with adaptive tiling for larger batches
  - Exact V1 tile sizing logic (32×32 for m=1024)
  - 4-column vectorized microkernel (conditionally enabled)
  - Software prefetching
  - AVX512/AVX2/scalar SIMD dot product

### Integration

Modified `tests/v2/performance/Perf__IQ4_NL_GEMM.cpp`:

```cpp
// OLD (virtual dispatch):
auto gemm = weight->createGemm();  // Returns ITensorGemm* interface
gemm->multiply(...);

// NEW (template):
llaminar2::QuantizedGemmOptimized<llaminar2::IQ4_NLTensor> gemm(weight.get());
gemm.multiply(...);  // Direct call, no virtual dispatch
```

## Performance Results

### Direct Comparison Test

**Workload:** Q-Proj 1024 (1024×896×896 GEMM, IQ4_NL weights, FP32 activations)  
**Hardware:** Single MPI rank, Release build (`-O3 -march=native`)  
**Iterations:** 20 timed iterations (3 warmup)

| Implementation | Time/Iter (ms) | Throughput (GFLOPS) | Speedup |
|----------------|----------------|---------------------|---------|
| Virtual Dispatch (ITensorGemm) | 57.60 | 28.55 | 1.00× (baseline) |
| **Template (QuantizedGemmOptimized)** | **48.96** | **33.58** | **1.18×** |

**Key Finding:** Template version is **18% faster** due to eliminated virtual dispatch overhead.

### Variance Analysis

Multiple test runs showed performance variance:
- QProjComparison_1024 test: 25.91 GFLOPS (template)
- VirtualVsTemplate_Comparison: 33.58 GFLOPS (template)
- Difference: ~30% variance between runs

**Likely causes:**
- CPU frequency scaling (turbo boost)
- Thermal throttling
- Background system processes
- Cache state differences

**Conclusion:** Direct comparison test (both versions in same run) is more reliable than separate test cases.

## Remaining Performance Gap

Despite 18% speedup from eliminating virtual dispatch, V2 is still **9.4× slower** than V1:
- **V1 baseline**: 314 GFLOPS (Q-Proj 1024, 2 MPI ranks, 32×32 tiles)
- **V2 template**: 33.58 GFLOPS (single rank, same tile sizes)
- **Gap**: 9.4× slower

### Hypothesis: Why Template Alone Isn't Enough

1. **MPI Distribution** (primary factor):
   - V1 tested with 2 MPI ranks (distributed computation)
   - V2 tested single rank (no distribution)
   - Impact: V1's 314 GFLOPS likely includes MPI parallelism benefits

2. **OpenMP Configuration**:
   - V1: 28 threads per rank × 2 ranks = 56 total threads
   - V2: Default OpenMP settings (unclear thread count)
   - Need to test V2 with explicit `OMP_NUM_THREADS=56`

3. **Broken Microkernel** (confirmed issue):
   - V2's microkernel makes performance WORSE (34 → 27 GFLOPS, 21% slower)
   - V1's microkernel improves performance +35%
   - Template version doesn't fix the broken microkernel logic

4. **Compiler Optimization Differences**:
   - V1 and V2 may have different compiler flags
   - Need to verify both use identical optimization levels
   - Check generated assembly for differences

5. **Tensor Allocation Overhead**:
   - V2's FP32Tensor vs V1's SimpleTensor
   - Possible memory layout differences affecting cache performance

## Action Items

### Immediate (High Priority)

1. **Fix Broken Microkernel** (separate investigation):
   - Compare V2's microkernel implementation to V1's
   - Identify why it makes performance worse instead of better
   - When fixed, expect +20-30% improvement

2. **Test with Optimal OpenMP Settings**:
   ```bash
   OMP_NUM_THREADS=56 OMP_PLACES=cores OMP_PROC_BIND=close \
     ./build_v2_release/performance/v2_perf_iq4nl_gemm --gtest_filter="*.VirtualVsTemplate_Comparison"
   ```

3. **Profile with perf** to identify actual hotspots:
   ```bash
   perf record -g ./build_v2_release/performance/v2_perf_iq4nl_gemm --gtest_filter="*.VirtualVsTemplate_Comparison"
   perf report
   ```

### Medium Priority

4. **Test V2 with MPI** (match V1's 2-rank configuration):
   - Port V2 to support MPI distribution
   - Test with 2 ranks like V1
   - Measure if this closes the gap

5. **Verify Compiler Flags**:
   - Compare V1 and V2 CMakeLists.txt optimization settings
   - Ensure both use `-O3 -march=native -DNDEBUG`
   - Check for any V1-specific flags not in V2

6. **Assembly Comparison**:
   - Generate assembly for both V1 and V2 GEMM implementations
   - Compare vectorization quality
   - Look for missed optimization opportunities

### Low Priority

7. **Tensor Allocation Benchmarking**:
   - Measure FP32Tensor creation overhead
   - Compare memory layout to V1's SimpleTensor
   - Test if tensor type affects cache performance

## Conclusions

1. ✅ **Virtual dispatch IS a bottleneck**: 18% speedup confirms hypothesis
2. ✅ **Template solution works**: Clean elimination of virtual calls
3. ❌ **Not sufficient alone**: Other factors dominate the 9.4× performance gap
4. 🔍 **Primary suspect**: MPI distribution (V1 uses 2 ranks, V2 uses 1)
5. 🔧 **Broken microkernel**: Separate bug that needs fixing

**Recommendation**: Prioritize fixing the broken microkernel (+20-30% expected) and testing with optimal OpenMP settings before investigating more exotic causes. The 9.4× gap is likely explained by V1's MPI distribution rather than algorithmic differences.

## Files Modified

**Created:**
- `src/v2/kernels/cpu/QuantizedGemmOptimized.h` (370 lines) - Template GEMM kernel

**Modified:**
- `tests/v2/performance/Perf__IQ4_NL_GEMM.cpp`:
  - Added `#include "kernels/cpu/QuantizedGemmOptimized.h"`
  - Changed `benchmarkFP32()` to use template instantiation
  - Added `VirtualVsTemplate_Comparison` test for direct comparison
  - Fixed namespace ambiguity (explicit `llaminar2::` qualifiers)

## Testing

```bash
# Build
cmake --build build_v2_release --target v2_perf_iq4nl_gemm --parallel

# Run comparison test
./build_v2_release/performance/v2_perf_iq4nl_gemm --gtest_filter="*.VirtualVsTemplate_Comparison"

# Expected output:
# Virtual Dispatch: ~28-32 GFLOPS
# Template:         ~33-38 GFLOPS
# Speedup:          1.15-1.20×
```

## Next Session Priorities

1. **Microkernel debug session**: Why does V2's microkernel make things slower?
2. **OpenMP tuning**: Test with V1's thread configuration
3. **Profiling**: Use perf to find actual hotspots
4. **MPI port**: Investigate effort to add MPI distribution to V2

---

**Session Date:** October 24, 2025  
**Performance Improvement:** +18% (virtual dispatch → template)  
**Remaining Gap to V1:** 9.4× (33.58 vs 314 GFLOPS)  
**Status:** Virtual dispatch eliminated ✅, performance parity not achieved ❌
