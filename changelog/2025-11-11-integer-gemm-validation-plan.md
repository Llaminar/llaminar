# Integer GEMM Validation Plan (Option A)

**Date**: November 11, 2025  
**Goal**: Validate current single-block `decode_to_q8_0()` API before optimizing  
**Status**: In Progress

---

## Phase 1: Core Functionality Tests ✅

### Existing Tests (Test__IntegerGemm.cpp)
1. ✅ `QuantizeFP32ToQ8_0_Simple` - Basic quantization with known values
2. ✅ `QuantizeFP32ToQ8_0_AllZeros` - Edge case: all-zero input
3. ✅ `QuantizeFP32ToQ8_0_LargeValues` - Edge case: large dynamic range
4. ✅ `IntegerGemm_SmallMatrix_IQ4NL` - Correctness test with real model weights
5. ✅ `IntegerGemm_Performance_Comparison` - Basic throughput measurement
6. ✅ `IntegerGemm_NoVNNI_Fallback` - Graceful degradation

---

## Phase 2: Cache Validation Tests (NEW)

### 2.1 Cache Hit Rate Test
**Goal**: Verify cache prevents redundant decoding

**Test**: `CacheHitRate_MultipleM Tiles`
```cpp
// Simulate GEMM pattern: Multiple M-tiles accessing same weight blocks
// Expected: >95% cache hit rate after first M-tile
```

**Validation**:
- First M-tile: 0% hit rate (cold cache)
- Subsequent M-tiles: >95% hit rate (hot cache)
- Overall: >90% hit rate

### 2.2 Zero-Copy Path Test
**Goal**: Verify Q8_0 weights use direct access (no cache)

**Test**: `ZeroCopyQ8_0_NoCache`
```cpp
// Load Q8_0 model
// Verify provider.is_zero_copy() == true
// Verify no cache allocation
```

**Validation**:
- Q8_0DirectProvider used for Q8_0 tensors
- No WeightDecodeCache instantiated
- Direct pointer access confirmed

### 2.3 Decode Correctness Tests
**Goal**: Verify tensor decode methods produce correct Q8_0 output

**Test**: `IQ4_NL_DecodeCorrectness`
```cpp
// Decode IQ4_NL block to Q8_0
// Compare against llama.cpp reference implementation
```

**Test**: `Q6_K_DecodeCorrectness`
```cpp
// Decode Q6_K super-block to Q8_0
// Verify hierarchical scale extraction
```

**Test**: `FP32_QuantizeCorrectness`
```cpp
// Quantize FP32 to Q8_0
// Verify symmetric quantization (scale = amax/127)
```

---

## Phase 3: Performance Profiling

### 3.1 Decode Overhead Measurement
**Goal**: Quantify decode cost vs total GEMM time

**Benchmark**: `DecodeOverhead_Breakdown`
```cpp
// Measure:
// - Time for first M-tile (with decoding)
// - Time for subsequent M-tiles (cached)
// - Total GEMM time
// Calculate: decode_time / total_time
```

**Target**: Decode overhead <5% of total GEMM time

### 3.2 Cache Size Sensitivity
**Goal**: Find optimal cache size

**Benchmark**: `CacheSizeSensitivity`
```cpp
// Test cache sizes: 256, 512, 1024, 2048, 4096 entries
// Measure: hit rate, memory overhead, performance
```

**Expected**:
- 256 entries: ~80-85% hit rate (insufficient)
- 1024 entries: ~95% hit rate (good)
- 4096 entries: ~99% hit rate (diminishing returns)

### 3.3 Format Comparison
**Goal**: Compare decode overhead across formats

**Benchmark**: `FormatDecodeComparison`
```cpp
// Test: IQ4_NL, Q6_K, Q4_K, FP32
// Measure: decode time per block
```

**Expected** (cycles per block):
- Q8_0: 0 (zero-copy)
- IQ4_NL: ~80-100 (4-bit LUT)
- Q6_K: ~150-200 (6-bit + hierarchical scales)
- FP32: ~100-120 (absmax + quantize)

---

## Phase 4: Integration Tests

### 4.1 Full Attention Layer
**Goal**: Test Q/K/V projection with cache reuse

**Test**: `AttentionLayer_CacheReuse`
```cpp
// Qwen 0.5B: 3 GEMMs (Q, K, V) share same hidden states
// Verify:
// - Q projection: decode weights (cold cache)
// - K projection: reuse cached weights
// - V projection: reuse cached weights
```

**Expected**: K and V projections ~4× faster than Q (cached)

### 4.2 Multi-Format Support
**Goal**: Verify all 20 quantized formats work

**Test**: `AllFormats_Supported`
```cpp
// Test each format: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
//                    Q4_0, Q4_1, Q5_0, Q5_1,
//                    IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS,
//                    IQ2_S, IQ3_S, IQ1_S, IQ1_M,
//                    FP32, FP16
// Verify: createWeightProvider() returns non-null
```

---

## Success Criteria

### Correctness
- ✅ All quantization tests pass
- ⏳ Cache hit rate >90% in typical GEMM patterns
- ⏳ Decode correctness <0.1% relative error vs reference
- ⏳ Zero-copy path confirmed for Q8_0

### Performance
- ⏳ Decode overhead <5% of total GEMM time
- ⏳ Cache provides >4× speedup vs no-cache
- ⏳ Throughput >20 GFLOPS (M=128, IQ4_NL weights)

### Coverage
- ⏳ All 20 quantized formats supported
- ⏳ Thread safety validated (OpenMP parallel access)
- ⏳ Edge cases handled (partial blocks, zero values)

---

## Decision Point: Batch Decode API

**After Phase 3 profiling, decide**:

### If decode overhead <2%:
- ✅ **Keep current single-block API** (good enough)
- No batch decode needed
- Focus on other optimizations (GEMM kernel, tiling)

### If decode overhead 5-10%:
- 🔄 **Consider batch decode API** (worthwhile optimization)
- Implement for IQ4_NL first (most common)
- Measure impact before extending to other formats

### If decode overhead >10%:
- ⚠️ **Batch decode API essential** (critical path)
- Implement for all formats
- Consider pre-decoding entire tiles (Option 3)

---

## Implementation Timeline

### Week 1 (Current)
- ✅ Refactor to GemmWeightCache (DONE)
- ⏳ Add cache validation tests
- ⏳ Measure decode overhead

### Week 2 (If needed)
- 🔄 Implement batch decode API (if profiling shows >5% overhead)
- 🔄 Optimize IQ4_NL with AVX512 vectorization
- 🔄 Benchmark improvement

### Week 3
- Integration with auto-tuner
- Full pipeline testing
- Documentation

---

## Test Execution Commands

```bash
# Build tests
cmake --build build_v2 --target v2_test_integer_gemm --parallel

# Run all tests
cd build_v2 && ./tests/v2/v2_test_integer_gemm

# Run specific test
./tests/v2/v2_test_integer_gemm --gtest_filter='Test__IntegerGemm.CacheHitRate*'

# Run with profiling
perf stat -e cycles,instructions,cache-misses,cache-references \
  ./tests/v2/v2_test_integer_gemm --gtest_filter='Test__IntegerGemm.DecodeOverhead*'
```

---

## Next Steps

1. **Immediate**: Run existing tests to establish baseline
2. **Add cache tests**: Validate hit rate and zero-copy path
3. **Profile decode overhead**: Measure actual cost
4. **Make decision**: Proceed with batch API or keep current

**Status**: Ready to proceed with test execution ✅
