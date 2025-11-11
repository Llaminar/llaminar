# Integer GEMM Template Instantiation Generator

**Date**: November 11, 2025  
**Author**: David Sanftenberg  
**Status**: Infrastructure complete, instantiations disabled due to template constraint issue

## Summary

Created a Python script to generate all template instantiations for the Integer GEMM kernel configuration space, enabling comprehensive tile-sweep benchmarking for ML-based kernel auto-tuning. The generator successfully creates 8000 instantiations across 64 files, but encountered a template design issue preventing compilation.

## Motivation

Following successful vectorization of the softmax microkernel (4-7× speedup with libmvec), the next phase is infrastructure for ML-based auto-tuning:

1. **Generate all kernel variants** (this work)
2. Benchmark across matrix sizes (M×N×K sweep)
3. Train ML model to predict optimal kernel
4. Deploy runtime kernel selection

## Implementation

### Generator Script

**File**: `src/v2/kernels/cpu/gemm/python/generate_integer_gemm_instantiations.py` (264 lines)

**Configuration Space**:
- **ISA**: `AVX512VNNITag` (INT8 support required)
- **MR**: 1, 2, 4, 8, 16, 32 (micro-kernel M tile)
- **NR**: 32 (fixed for Q8_0 alignment)
- **UNROLL_K**: 1, 2, 4, 8, 16 (K-loop unroll)
- **PREFETCH_DIST**: 0, 1, 2, 3, 5 (prefetch distance)
- **MC**: 128, 256, 512, 1024 (M cache block)
- **KC**: 256, 512, 1024, 2048 (K cache block, multiple of 32)
- **NC**: 64, 128, 256, 512 (N cache block)

**Filters Applied**:
- Register pressure: `MR * (NR/32)` must fit in ZMM registers (24 available)
- KC alignment: `KC % 32 == 0` (Q8_0 block size)
- Cache hierarchy: MC×KC and KC×NC panels must fit in L2/L3

**Output**:
- 8000 valid instantiations (5 MR values × 5 UNROLL_K × 5 PREFETCH × 4 MC × 4 KC × 4 NC)
- Distributed across 64 files (125 instantiations per file)
- Parallel compilation enabled

### Generated Files

**Instantiation Files**: `src/v2/kernels/cpu/gemm/int8/generated/IntegerGemmInstantiations_00.cpp` ... `_63.cpp`

**Example instantiation**:
```cpp
template class IntegerGemmKernel<simd::AVX512VNNITag, 8, 32, 4, 2, 256, 512, 128>;
```

**CMake Integration**: `src/v2/kernels/cpu/gemm/int8/generated/sources.cmake`

### CMake Integration

**Modified**: `src/v2/CMakeLists.txt`

**Added**:
1. Generator script execution (lines 434-459):
   - Auto-detects if regeneration needed
   - Runs Python script during CMake configure
   - Validates generation success
   
2. Source inclusion (line 564):
   - `${INTEGER_GEMM_INSTANTIATION_SOURCES}` variable
   - Currently commented out (see issue below)

## Issue Encountered

### Compilation Error

**Error**: `'const class llaminar2::FP16Tensor' has no member named 'decode_to_q8_0'`

**Root Cause**: Template dependency chain causes unwanted instantiations

```
IntegerGemmKernel<...>
  → includes GemmWeightCache.h
    → contains CachedQ8Provider<TensorType>
      → gets instantiated for FP16Tensor (doesn't have decode_to_q8_0)
```

**Why This Happens**:
- `IntegerGemmKernelTemplate.h` includes `Tensors.h` (all tensor types)
- Explicit instantiation of `IntegerGemmKernel` triggers instantiation of all template dependencies
- `CachedQ8Provider<TensorType>` is unconstrained - accepts any tensor type
- Compiler tries to instantiate `CachedQ8Provider<FP16Tensor>` → fails

### Solutions Considered

**Option 1**: SFINAE/Concepts to constrain `CachedQ8Provider`
```cpp
template <typename TensorType>
concept HasDecodeToQ8_0 = requires(const TensorType& t, size_t r, size_t k, Q8_0Block* out) {
    { t.decode_to_q8_0(r, k, out) } -> std::same_as<void>;
};

template <typename TensorType> requires HasDecodeToQ8_0<TensorType>
class CachedQ8Provider : public Q8_0BlockProvider { ... };
```

**Option 2**: Forward declarations to break dependency
- Declare `IntegerGemmKernel` without including `Tensors.h`
- Move full implementation to `.cpp` file
- Limits template flexibility

**Option 3**: Separate compilation units
- Don't explicitly instantiate full kernel
- Rely on implicit instantiation at usage sites
- Defeats purpose of pre-generated instantiations

**Decision**: Comment out instantiations for now (Option 3 compromise)

## Current Status

✅ **Infrastructure Complete**:
- Generator script working (264 lines)
- 8000 instantiations successfully generated
- CMake integration added
- Parallel compilation support

❌ **Instantiations Disabled**:
- Commented out in `CMakeLists.txt` line 564
- Build succeeds without them
- Need template constraints before enabling

## Next Steps

### Immediate (Fix Template Issue)

1. Add C++20 concepts to `CachedQ8Provider`:
   ```cpp
   template <typename TensorType>
   concept Q8_0Decodable = requires(const TensorType& t) {
       { t.decode_to_q8_0(size_t{}, size_t{}, (Q8_0Block*)nullptr) };
   };
   ```

2. Update `GemmWeightCache.h`:
   ```cpp
   template <Q8_0Decodable TensorType>
   class CachedQ8Provider : public Q8_0BlockProvider { ... };
   ```

3. Uncomment `${INTEGER_GEMM_INSTANTIATION_SOURCES}` in CMakeLists.txt

4. Build with `-std=c++20` (already enabled)

### Future (ML Auto-Tuning Pipeline)

1. **Benchmark Harness**:
   - Sweep matrix sizes: 32×32×32 to 4096×4096×4096
   - Test all 8000 kernel configs per size
   - Record: (M, N, K, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC) → GFLOPS

2. **Data Collection**:
   - Export CSV with ~100K rows (matrix sizes × configs)
   - Include hardware features (cache sizes, core count)
   - Add runtime features (NUMA topology, load)

3. **ML Model Training**:
   - Feature engineering: log(M), log(N), log(K), ratios
   - Model: Gradient boosting (XGBoost, LightGBM)
   - Target: Predict top-K configs per matrix size

4. **Runtime Integration**:
   - `IntegerGemmKernelRegistry` similar to `MicroKernelRegistry`
   - ML model inference at runtime
   - Fallback to heuristic if model unavailable

## Files Created/Modified

**Created**:
- `src/v2/kernels/cpu/gemm/python/generate_integer_gemm_instantiations.py` (264 lines)
- `src/v2/kernels/cpu/gemm/int8/generated/IntegerGemmInstantiations_00.cpp` ... `_63.cpp` (64 files)
- `src/v2/kernels/cpu/gemm/int8/generated/sources.cmake`

**Modified**:
- `src/v2/CMakeLists.txt` (lines 434-459, 564)

## Performance Notes

**Compilation Time** (estimated):
- 64 files × 125 instantiations = 8000 templates
- Each instantiation: ~2-5 seconds (optimized)
- Total: ~4-10 hours single-threaded
- With `--parallel` (56 cores): ~5-10 minutes

**Binary Size Impact**:
- 8000 instantiations × ~50KB each = ~400MB
- Acceptable for development/benchmarking builds
- Production builds would select subset via ML model

## Lessons Learned

1. **Template Instantiation Cascades**: Explicit instantiation triggers all transitive template dependencies
2. **C++20 Concepts Are Critical**: Unconstrained templates cause compilation errors at instantiation sites
3. **SFINAE Not Sufficient**: Modern codebase should use concepts for template constraints
4. **Parallel Compilation Essential**: 8000 instantiations require `-j56` builds

## References

- FP32 GEMM generator: `generate_gemm_microkernel_instantiations.py` (171 lines, 1225 variants)
- Vectorized softmax: `VectorizedSoftmax.h` (336 lines, 4-7× speedup)
- Integer GEMM template: `IntegerGemmKernelTemplate.h` (306 lines)
- Weight cache: `GemmWeightCache.h` (485 lines)
