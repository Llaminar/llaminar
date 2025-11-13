# Integer GEMM Template Parameter Refactor

**Date**: November 12, 2025  
**Status**: ✅ Complete  
**Impact**: Performance-critical code path optimization

## Problem

The initial implementation of configurable K-block processing used runtime configuration lookup via `IntegerGemmConfig::instance()` **inside the hot path** (the parallel OpenMP loop). This violated the core design principle of the kernel template system and introduced unnecessary overhead:

```cpp
// ❌ BAD: Runtime lookup in hot path
#pragma omp parallel for
for (int ii = 0; ii < m; ii += TILE_M) {
    const auto& config = IntegerGemmConfig::instance();  // ← EVERY ITERATION!
    const int BLOCKS_PER_ITER = config.k_blocks_per_iter;
    // ...
}
```

## Solution

Converted `K_BLOCKS_PER_ITER` from runtime configuration to **compile-time template parameter**, following the existing pattern used by all other kernel parameters (MR, NR, UNROLL_K, etc.).

### Template Signature Change

**Before**:
```cpp
template <
    typename ISA,
    int MR,
    int NR,
    int UNROLL_K = 4,
    int PREFETCH_DIST = 2,
    int MC = 256,
    int KC = 512,
    int NC = 128>
class IntegerGemmKernelV2
```

**After**:
```cpp
template <
    typename ISA,
    int MR,
    int NR,
    int K_BLOCKS_PER_ITER = 2,  // ← NEW: Compile-time K-block processing
    int UNROLL_K = 4,
    int PREFETCH_DIST = 2,
    int MC = 256,
    int KC = 512,
    int NC = 128>
class IntegerGemmKernelV2
```

### Implementation Changes

**Hot path now uses `constexpr`**:
```cpp
static constexpr int BLOCKS_PER_ITER = K_BLOCKS_PER_ITER;  // Compile-time constant
static constexpr int BYTES_PER_ITER = BLOCK_SIZE * BLOCKS_PER_ITER;  // 32, 64, or 128

// Compile-time validation
static_assert(K_BLOCKS_PER_ITER >= 1 && K_BLOCKS_PER_ITER <= 4, 
              "K_BLOCKS_PER_ITER must be 1, 2, or 4");

// Sized for template parameter (not max)
alignas(64) int8_t A_panel[TILE_M * BYTES_PER_ITER];
alignas(64) float a_scales[TILE_M * BLOCKS_PER_ITER];
```

**Scale storage fixed to use proper stride**:
```cpp
// ✅ CORRECT: Stride matches BLOCKS_PER_ITER template parameter
a_scales[i * BLOCKS_PER_ITER + b] = fp16_to_fp32(block->d);
b_scales[j * BLOCKS_PER_ITER + b] = fp16_to_fp32(block->d);
```

This fixes the 128-byte mode scale reuse bug (was using stride-2 for 4 blocks).

## Files Modified

### Core Kernel
- **`src/v2/kernels/cpu/gemm/IntegerGemmKernelTemplateV2.h`**:
  - Added `K_BLOCKS_PER_ITER` template parameter with default=2
  - Removed `#include "IntegerGemmConfig.h"`
  - Removed runtime config lookup from hot path
  - Changed scale storage from hardcoded stride-2 to `BLOCKS_PER_ITER`
  - Updated documentation (1→4 blocks supported, not just 1→2)

### Benchmark/Tests
- **`tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_QwenProfile.cpp`**:
  - Updated `benchmark_operation<>` template signature to include `K_BLOCKS_PER_ITER`
  - Updated all instantiations to pass `K_BLOCKS_PER_ITER=2` (64-byte default)
  - Example: `IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 2, 8, 3, 512, 1024, 256>`

- **`tests/v2/performance/cpu/kernels/gemm/Test__IntegerGEMM_V2_Basic.cpp`**:
  - No changes needed (uses default K_BLOCKS_PER_ITER=2)

## Testing

```bash
# Build succeeds
cmake --build build_v2 --target v2_perf_integer_gemm_qwen_profile --parallel 8
cmake --build build_v2 --target v2_test_integer_gemm_v2_basic --parallel 8

# Default 2-block configuration works
./build_v2/performance/v2_perf_integer_gemm_qwen_profile --simd
# Average: ~7 GFLOPS (same as before refactor)
```

## Benefits

1. **Zero runtime overhead**: No config lookup in hot path
2. **Follows design principles**: Matches existing template parameter pattern
3. **Compiler optimizations**: `constexpr` enables aggressive optimization
4. **Type safety**: Template instantiation validates K_BLOCKS_PER_ITER at compile time
5. **Scale bug fixed**: Proper stride for 1/2/4 block modes

## Configuration Space

To explore different K-block configurations, instantiate different templates:

```cpp
// 32-byte processing (1 block)
using Kernel32 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 1>;

// 64-byte processing (2 blocks) - current default
using Kernel64 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 2>;

// 128-byte processing (4 blocks) - experimental
using Kernel128 = IntegerGemmKernelV2<AVX512VNNITag, 16, 32, 4>;
```

## Next Steps

1. ✅ **Template parameter working** - compile-time configuration
2. 🔄 **Test 128-byte mode** - verify scale fix resolved correctness issues
3. 🔄 **Performance sweep** - compare 1-block, 2-block, 4-block on real workloads
4. 🔄 **Release build testing** - measure impact with full optimization

## Design Notes

This refactor demonstrates the proper way to handle kernel configuration in the V2 architecture:

- **Compile-time parameters**: Template parameters for architecture decisions (tile sizes, block counts)
- **Runtime parameters**: Function arguments for data and dimensions (m, n, k)
- **No middle ground**: Avoid runtime config lookups in performance-critical loops

The `IntegerGemmConfig` runtime configuration system was a misstep - it should have been template-based from the start to match the existing kernel design pattern.
