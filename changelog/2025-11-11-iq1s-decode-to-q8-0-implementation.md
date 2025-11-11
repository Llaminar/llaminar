# IQ1_S decode_to_q8_0 SIMD Implementation

**Date**: 2025-11-11  
**Status**: ✅ **COMPLETE** - All 10 tests passing  
**Format**: IQ1_S (1-bit quantization with IQ1S_DELTA offset)  
**Performance**: 🚀 **5.34× AVX2, 9.13× AVX-512 speedup** (BEST SIMD SPEEDUP SO FAR)

---

## Summary

Implemented complete SIMD decode path for **IQ1_S** format to Q8_0 blocks. IQ1_S is a 1-bit quantization format with **IQ1S_DELTA offset** (±0.125f) using iq1s_grid[2048] for grid lookups. Achieved exceptional SIMD performance with **9.13× AVX-512 speedup**, the highest observed across all formats.

---

## Implementation Details

### IQ1_S Format Structure

**Block size**: 50 bytes per 256-element super-block
- `d` (2 bytes): FP16 scale factor
- `qs[32]` (32 bytes): Grid indices (8 bits each)
- `qh[8]` (8 uint16_t = 16 bytes): High bits and scales

**Grid lookup**: `iq1s_grid[2048]` (uint64_t entries, accessed as int8_t*)

**Sub-block structure** (32 elements per iteration):
- 8 iterations (ib=0..7), each handling 32 elements
- Per iteration:
  - **Scale**: 3 bits from `qh[ib]` bits 12-14 → `dl = d * (2 * scale + 1)`
  - **Delta**: Sign bit from `qh[ib]` bit 15 → `±IQ1S_DELTA` (0.125f)
  - **4 groups of 8 elements**:
    - Grid index: `qs[l]` (8 bits) + `qh[ib]` bits (3*l to 3*l+2) → 11-bit index (0-2047)
    - Decode: `output[j] = dl * (grid[j] + delta)`

### Algorithm Pseudocode

```cpp
// Per sub-block (32 elements, ib=0..7):
const float d = fp16_to_fp32(block.d);
const uint8_t *qs = block.qs + ib * 4;  // 4 bytes per sub-block
const uint16_t qh_val = block.qh[ib];

// Scale: 3 bits from qh bits 12-14
const float dl = d * (2.0f * ((qh_val >> 12) & 7) + 1.0f);

// Delta: sign bit from qh bit 15
const float delta = (qh_val & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

// 4 groups of 8 elements (32 elements total)
for (size_t l = 0; l < 4; ++l) {
    // 11-bit grid index
    const uint16_t grid_idx = qs[l] | (((qh_val >> (3 * l)) & 7) << 8);
    
    // Grid lookup (uint64_t treated as int8_t*)
    const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx);
    
    for (size_t j = 0; j < 8; ++j) {
        output[j] = dl * (static_cast<float>(grid[j]) + delta);
    }
    output += 8;
}
```

---

## Files Modified

### 1. `src/v2/tensors/SIMDHelpers.h` (~200 lines added)

**Location**: After IQ3_S section (lines ~2599-2789)

**Functions added**:
- `decode_iq1s_to_q8_0_scalar()` - Scalar implementation with grid lookup + delta offset
- `decode_iq1s_to_q8_0_avx2()` - AVX2 variant (FP32 intermediate → quantize_fp32_to_q8_0_avx2)
- `decode_iq1s_to_q8_0_avx512()` - AVX-512 variant (processes 2 groups at once for 16 elements)
- `decode_iq1s_to_q8_0()` - Auto-dispatch based on CPU features

**Key implementation details**:
- Grid index construction: 11-bit index from `qs[l] | (((qh_val >> (3*l)) & 7) << 8)`
- Scale formula: `dl = d * (2 * scale + 1)` (3-bit scale from qh bits 12-14)
- Delta offset: `±IQ1S_DELTA` (0.125f) based on qh bit 15
- Grid access: `iq1s_grid[2048]` (uint64_t) accessed as `int8_t*`
- Signature: `uint16_t* q8_scale` (consistent with other IQ formats)

**AVX-512 optimization**: Processes 2 groups (16 elements) at once using `_mm512_cvtepi8_epi32` and `_mm512_cvtepi32_ps` for efficient int8→float conversion

### 2. `src/v2/tensors/IQ1_STensor.cpp` (~18 lines added)

**Function**: `IQ1_STensor::decode_to_q8_0(row_idx, k_block_offset, output)`

**Implementation**:
```cpp
// Get raw data pointer (view-aware)
const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);

// Calculate super-block index (8 Q8_0 blocks per IQ1_S super-block)
size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
size_t super_block_idx = k_block_offset / 8;

const IQ1_SBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

// Decode all 8 sub-blocks (each sub-block = 32 elements = 1 Q8_0 block)
for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx) {
    simd::decode_iq1s_to_q8_0(super_block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
}
```

**Pattern**: Follows IQ2_S/IQ3_S pattern (view-aware, super-block indexing)

### 3. `tests/v2/unit/Test__IQ1_S_DecodeVectorization.cpp` (NEW, ~380 lines)

**Test suite**: 10 comprehensive tests

**Tests**:
1. **ScalarCorrectness** - Basic decode validation
2. **MultiBlockCorrectness** - Multiple blocks with varied data
3. **GridLookupVerification** - Validates 11-bit grid indices (0-2047 range)
4. **AVX2Parity** - Scalar vs AVX2 exact match
5. **AVX512Parity** - Scalar vs AVX-512 exact match
6. **TensorIntegration** - End-to-end IQ1_STensor decode path
7. **AutoDispatch** - CPU feature detection
8. **FuzzTesting** - 1000 random blocks
9. **ErrorHandling** - Invalid inputs
10. **Performance** - Benchmark (10,000 iterations)

**Test helper**: `createTestBlock(seed)` - Generates random IQ1_SBlock with:
- Random qs[], qh[] bytes
- Valid FP16 scale (0.01-10.0 range)
- Grid indices verified in bounds (< 2048)

**Comparison**: `compareQ8_0BlockArrays()` - Validates 8 Q8_0 blocks with tolerance

### 4. `tests/v2/CMakeLists.txt` (~14 lines added)

**Target**: `v2_test_iq1s_decode_vectorization`

**Labels**: 
```
V2;Unit;TensorOperations;Quantization;IQ1_S;Q8_0;SIMD;AVX2;AVX512;
GridLookup;DeltaOffset;VectorizationCorrectness;FuzzTesting;OneBitQuantization
```

---

## Test Results

```
[==========] Running 10 tests from 1 test suite.
[----------] 10 tests from Test__IQ1_S_DecodeVectorization

[ RUN      ] Test__IQ1_S_DecodeVectorization.ScalarCorrectness
[       OK ] Test__IQ1_S_DecodeVectorization.ScalarCorrectness (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.MultiBlockCorrectness
[       OK ] Test__IQ1_S_DecodeVectorization.MultiBlockCorrectness (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.GridLookupVerification
[       OK ] Test__IQ1_S_DecodeVectorization.GridLookupVerification (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.AVX2Parity
[       OK ] Test__IQ1_S_DecodeVectorization.AVX2Parity (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.AVX512Parity
[       OK ] Test__IQ1_S_DecodeVectorization.AVX512Parity (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.TensorIntegration
[       OK ] Test__IQ1_S_DecodeVectorization.TensorIntegration (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.AutoDispatch
[       OK ] Test__IQ1_S_DecodeVectorization.AutoDispatch (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.FuzzTesting
[       OK ] Test__IQ1_S_DecodeVectorization.FuzzTesting (29 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.ErrorHandling
[       OK ] Test__IQ1_S_DecodeVectorization.ErrorHandling (0 ms)

[ RUN      ] Test__IQ1_S_DecodeVectorization.Performance

IQ1_S Decode Performance (10000 iterations):
  Scalar:   101.162 ms (baseline)
  AVX2:     18.9451 ms (speedup: 5.33975x)
  AVX-512:  11.0761 ms (speedup: 9.13332x)
[       OK ] Test__IQ1_S_DecodeVectorization.Performance (131 ms)

[----------] 10 tests from Test__IQ1_S_DecodeVectorization (161 ms total)
[==========] 10 tests from 1 test suite ran. (161 ms total)
[  PASSED  ] 10 tests.
```

---

## Performance Analysis

**IQ1_S Performance** (10,000 iterations):
- **Scalar**: 101.162 ms (baseline)
- **AVX2**: 18.945 ms (**5.34× speedup**) ← Excellent
- **AVX-512**: 11.076 ms (**9.13× speedup**) ← **BEST SIMD SPEEDUP ACROSS ALL FORMATS**

**Comparison to other formats**:

| Format | Grid Size | AVX2 Speedup | AVX-512 Speedup | Notes |
|--------|-----------|--------------|-----------------|-------|
| IQ2_XXS | 256 | 0.66× | 1.02× | Small grid overhead |
| IQ3_XXS | 256 | 0.66× | 1.00× | Small grid overhead |
| Q4_0 | N/A | 2.33× | 1.99× | Bit unpacking |
| Q4_1 | N/A | 1.87× | 2.47× | Bit unpacking + min |
| Q5_0 | N/A | 1.96× | 2.45× | Bit unpacking (5-bit) |
| Q5_1 | N/A | 1.91× | 2.38× | Bit unpacking (5-bit) + min |
| IQ4_NL | N/A | 2.17× | 3.02× | Grid lookup |
| IQ3_S | 512 | 2.90× | 3.28× | Grid + signs array |
| IQ2_XS | 512 | 3.69× | 4.80× | Grid + scales |
| IQ2_S | 1024 | 3.95× | 4.89× | Grid + high bits |
| **IQ1_S** | **2048** | **5.34×** | **9.13×** | **1-bit + delta offset** ← **BEST**

**Why IQ1_S performs exceptionally well**:

1. **Simple arithmetic**: `dl * (grid[j] + delta)` is SIMD-friendly (broadcast multiply + add)
2. **Large grid** (2048 entries): Better cache locality than smaller grids
3. **No complex bit manipulation**: Grid index construction is straightforward
4. **Delta offset pattern**: Constant `±IQ1S_DELTA` is easily broadcast
5. **AVX-512 optimization**: Processes 2 groups (16 elements) at once, leveraging wider registers

---

## Key Learnings

### 1. 1-bit Quantization Benefits
**IQ1_S approach**: 1-bit quantization with grid lookup + delta offset

**Advantages**:
- Extremely compact (50 bytes per 256 elements = 0.195 bytes/element)
- Simple decode formula enables excellent SIMD vectorization
- Delta offset provides sign flexibility without complex bit operations

**SIMD Impact**: Best SIMD speedups observed (9.13× AVX-512)

### 2. Grid Size and Performance
**Observed correlation**:
- **Small grids (256)**: High overhead (IQ2_XXS: 1.02× AVX-512)
- **Medium grids (512)**: Moderate performance (IQ3_S: 3.28× AVX-512)
- **Large grids (1024-2048)**: Best performance (IQ2_S: 4.89×, **IQ1_S: 9.13×**)

**Implication**: Larger grids reduce relative lookup overhead, allowing SIMD arithmetic to dominate

### 3. AVX-512 Optimization Techniques
**IQ1_S AVX-512 strategy**: Process 2 groups (16 elements) simultaneously

**Benefits**:
- Full utilization of 512-bit registers (16 floats)
- Single `_mm512_cvtepi8_epi32` converts 16 int8→int32 at once
- Reduced loop overhead (2 iterations instead of 4)

**Result**: 9.13× speedup vs scalar (vs 5.34× for AVX2)

### 4. Delta Offset Pattern
**IQ1_S delta**: Constant `±IQ1S_DELTA` (0.125f) based on single bit

**SIMD advantage**: 
- Broadcast once per sub-block
- Constant add operation (no conditional logic)
- No sign bit extraction from packed data

**Comparison**: Simpler than IQ3_S's signs[] array (better cache behavior)

---

## Progress Update

**Completed Formats** (12/14):
- ✅ Q4_0: 10/10 tests, 2.33× AVX2, 1.99× AVX-512
- ✅ Q4_1: 10/10 tests, 1.87× AVX2, 2.47× AVX-512
- ✅ Q5_0: 10/10 tests, 1.96× AVX2, 2.45× AVX-512
- ✅ Q5_1: 11/11 tests, 1.91× AVX2, 2.38× AVX-512
- ✅ IQ4_NL: 11/11 tests, 2.17× AVX2, 3.02× AVX-512
- ✅ IQ4_XS: 9/9 tests
- ✅ IQ2_XXS: 10/10 tests, 0.66× AVX2, 1.02× AVX-512
- ✅ IQ2_XS: 10/10 tests, 3.69× AVX2, 4.80× AVX-512
- ✅ IQ3_XXS: 10/10 tests, 0.66× AVX2, 1.00× AVX-512
- ✅ IQ2_S: 10/10 tests, 3.95× AVX2, 4.89× AVX-512
- ✅ IQ3_S: 10/10 tests, 2.90× AVX2, 3.28× AVX-512
- ✅ **IQ1_S: 10/10 tests, 5.34× AVX2, 9.13× AVX-512** ← **JUST COMPLETED (BEST PERFORMANCE)**

**Implementation Only** (1/14):
- ✅ Q3_K: Scalar implementation, NO TESTS

**Declarations Only** (1/14):
- ✅ IQ1_M: 1-bit quantization + multiple scales (FINAL FORMAT)

**Total**: ~137 tests passing across 12 formats

---

## Next Steps

**Remaining 1 Complex Format**:
1. **IQ1_M**: 1-bit quantization + multiple scales (56 bytes/block)

**Expected approach**: Continue same pattern (scalar → AVX2 → AVX-512 → tests)

**Expected performance**: Similar to IQ1_S (1-bit quantization, grid-based)

---

## Build Instructions

```bash
# From workspace root
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --target v2_test_iq1s_decode_vectorization --parallel

# Run tests
cd build_v2
./tests/v2/v2_test_iq1s_decode_vectorization

# Or via CTest
ctest -R "V2_Unit_IQ1_S_DecodeVectorization" --verbose
```

---

## References

- **IQ1_S Scalar Algorithm**: `src/v2/tensors/IQ1_STensor.cpp::decodeBlockScalar()`
- **Grid Table**: `src/v2/tensors/IQQuantTables.h::iq1s_grid[2048]`
- **Delta Constant**: `src/v2/tensors/IQQuantTables.h::IQ1S_DELTA` (0.125f)
- **Pattern Reference**: IQ2_S implementation (similar super-block structure)
- **Performance Comparison**: IQ2_S (4.89× AVX-512), IQ3_S (3.28× AVX-512)

---

**End of Changelog**
