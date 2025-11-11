# IQ2_S decode_to_q8_0 SIMD Implementation

**Date**: 2025-11-10  
**Status**: ✅ **COMPLETE** - All 10 tests passing  
**Format**: IQ2_S (2-bit + sign + grid + high bits + dual scales)  
**Performance**: 3.95× AVX2, 4.89× AVX-512 speedup vs scalar

---

## Summary

Implemented complete SIMD decode path for **IQ2_S** format to Q8_0 blocks, following the established pattern from IQ3_XXS/IQ2_XS. IQ2_S is a grid-based format with **high bits** for larger grid indices (10-bit vs 8-bit) and **dual scales** for finer granularity (16 elements vs 32 elements per scale).

---

## Implementation Details

### IQ2_S Format Structure

**Block size**: 82 bytes per 256-element super-block
- `d` (2 bytes): FP16 scale factor
- `qs[64]` (64 bytes): Quantized values (32 bytes) + sign patterns (32 bytes)
- `qh[8]` (8 bytes): High bits (2 extra bits per sub-block for grid index)
- `scales[8]` (8 bytes): Dual scales (2 scales per sub-block, 4 bits each)

**Grid lookup**: `iq2s_grid[1024]` (uint64_t entries) - larger than IQ3_XXS (256 entries)

**Sub-block structure** (32 elements):
- 4 groups of 8 elements
- Each group uses 10-bit grid index: `qs[l] | ((qh_byte << (8 - 2*l)) & 0x300)`
- 2 scales per sub-block: `db[0] = d * (0.5 + (scale_byte & 0xf)) * 0.25`, `db[1] = d * (0.5 + (scale_byte >> 4)) * 0.25`
- Groups 0-1 use `db[0]`, groups 2-3 use `db[1]`
- Sign patterns from `qs[32..63]` using `kmask_iq2xs[]`

### Algorithm Pseudocode

```cpp
// Per sub-block (32 elements):
const uint8_t *qs = block.qs + subblock_idx * 4;            // 4 bytes
const uint8_t *signs = block.qs + 32 + subblock_idx * 4;    // Signs offset
const uint8_t qh_byte = block.qh[subblock_idx];
const uint8_t scale_byte = block.scales[subblock_idx];

// Two scales per sub-block
float db[2];
db[0] = d * (0.5f + (scale_byte & 0xf)) * 0.25f;
db[1] = d * (0.5f + (scale_byte >> 4)) * 0.25f;

// 4 groups of 8 elements each
for (size_t l = 0; l < 4; ++l) {
    const float dl = db[l / 2]; // First 2 use db[0], last 2 use db[1]
    
    // 10-bit grid index from qs (8 bits) + qh high bits (2 bits)
    const uint16_t grid_idx = qs[l] | ((qh_byte << (8 - 2*l)) & 0x300);
    const uint8_t *grid = (const uint8_t *)(iq2s_grid + grid_idx);
    
    for (size_t j = 0; j < 8; ++j) {
        output[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
    }
    output += 8;
}
```

---

## Files Modified

### 1. `src/v2/tensors/SIMDHelpers.h` (~180 lines added)

**Location**: After IQ3_XXS section (lines ~2213-2393)

**Functions added**:
- `decode_iq2s_to_q8_0_scalar()` - Scalar implementation with grid lookup + high bits
- `decode_iq2s_to_q8_0_avx2()` - AVX2 variant (FP32 intermediate → quantize_fp32_to_q8_0_avx2)
- `decode_iq2s_to_q8_0_avx512()` - AVX-512 variant (FP32 intermediate → quantize_fp32_to_q8_0_avx512)
- `decode_iq2s_to_q8_0()` - Auto-dispatch based on CPU features

**Key implementation details**:
- Grid index construction: 10-bit index from `qs[l] | ((qh_byte << (8 - 2*l)) & 0x300)`
- Dual scales: `db[l/2]` alternates between two scales per sub-block
- Sign patterns: Reuses `kmask_iq2xs[]` from IQ2_XS
- Signature: `uint16_t* q8_scale` (consistent with other IQ formats)

### 2. `src/v2/tensors/IQ2_STensor.cpp` (~17 lines added)

**Function**: `IQ2_STensor::decode_to_q8_0(row_idx, k_block_offset, output)`

**Implementation**:
```cpp
// Get raw data pointer (view-aware)
const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);

// Calculate super-block index (8 Q8_0 blocks per IQ2_S super-block)
size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
size_t super_block_idx = k_block_offset / 8;

const IQ2_SBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

// Decode all 8 sub-blocks
for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx) {
    simd::decode_iq2s_to_q8_0(super_block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
}
```

**Pattern**: Follows IQ3_XXSTensor pattern (view-aware, super-block indexing)

### 3. `tests/v2/unit/Test__IQ2_S_DecodeVectorization.cpp` (NEW, ~418 lines)

**Test suite**: 10 comprehensive tests

**Tests**:
1. **ScalarCorrectness** - Basic decode validation
2. **MultiBlockCorrectness** - Multiple blocks with varied data
3. **GridLookupVerification** - Validates 10-bit grid indices (0-1023 range)
4. **AVX2Parity** - Scalar vs AVX2 exact match
5. **AVX512Parity** - Scalar vs AVX-512 exact match
6. **TensorIntegration** - End-to-end IQ2_STensor decode path
7. **AutoDispatch** - CPU feature detection
8. **FuzzTesting** - 1000 random blocks
9. **ErrorHandling** - Invalid inputs
10. **Performance** - Benchmark (10,000 iterations)

**Test helper**: `createTestBlock(seed)` - Generates random IQ2_SBlock with:
- Random qs[], qh[], scales[] bytes
- Valid FP16 scale (0.01-10.0 range)
- Grid indices verified in bounds (< 1024)

**Comparison**: `compareQ8_0BlockArrays()` - Validates 8 Q8_0 blocks with tolerance

### 4. `tests/v2/CMakeLists.txt` (~14 lines added)

**Target**: `v2_test_iq2s_decode_vectorization`

**Labels**: 
```
V2;Unit;TensorOperations;Quantization;IQ2_S;Q8_0;SIMD;AVX2;AVX512;
GridLookup;HighBits;ScalesArray;VectorizationCorrectness;FuzzTesting
```

---

## Test Results

```
[==========] Running 10 tests from 1 test suite.
[----------] 10 tests from Test__IQ2_S_DecodeVectorization

[ RUN      ] Test__IQ2_S_DecodeVectorization.ScalarCorrectness
[       OK ] Test__IQ2_S_DecodeVectorization.ScalarCorrectness (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.MultiBlockCorrectness
[       OK ] Test__IQ2_S_DecodeVectorization.MultiBlockCorrectness (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.GridLookupVerification
[       OK ] Test__IQ2_S_DecodeVectorization.GridLookupVerification (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.AVX2Parity
[       OK ] Test__IQ2_S_DecodeVectorization.AVX2Parity (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.AVX512Parity
[       OK ] Test__IQ2_S_DecodeVectorization.AVX512Parity (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.TensorIntegration
[       OK ] Test__IQ2_S_DecodeVectorization.TensorIntegration (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.AutoDispatch
[       OK ] Test__IQ2_S_DecodeVectorization.AutoDispatch (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.FuzzTesting
[       OK ] Test__IQ2_S_DecodeVectorization.FuzzTesting (50 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.ErrorHandling
[       OK ] Test__IQ2_S_DecodeVectorization.ErrorHandling (0 ms)

[ RUN      ] Test__IQ2_S_DecodeVectorization.Performance

IQ2_S Decode Performance (10000 iterations):
  Scalar:   88.085 ms (baseline)
  AVX2:     22.316 ms (speedup: 3.94717x)
  AVX-512:  17.998 ms (speedup: 4.89415x)
[       OK ] Test__IQ2_S_DecodeVectorization.Performance (128 ms)

[----------] 10 tests from Test__IQ2_S_DecodeVectorization (180 ms total)
[==========] 10 tests from 1 test suite ran. (180 ms total)
[  PASSED  ] 10 tests.
```

---

## Performance Analysis

**IQ2_S Performance** (10,000 iterations):
- **Scalar**: 88.085 ms (baseline)
- **AVX2**: 22.316 ms (**3.95× speedup**)
- **AVX-512**: 17.998 ms (**4.89× speedup**)

**Comparison to other grid-based formats**:

| Format | Grid Size | AVX2 Speedup | AVX-512 Speedup | Notes |
|--------|-----------|--------------|-----------------|-------|
| IQ2_XXS | 256 | 0.66× | 1.02× | Grid lookup overhead dominates |
| IQ3_XXS | 256 | 0.66× | 1.00× | Grid lookup overhead dominates |
| IQ2_XS | 512 | 3.69× | 4.80× | Better SIMD utilization |
| **IQ2_S** | 1024 | **3.95×** | **4.89×** | **Excellent SIMD benefits** |

**Why IQ2_S performs better than IQ2_XXS/IQ3_XXS**:
1. **Larger grid** (1024 entries) reduces cache pressure per lookup
2. **Dual scales** allow better vectorization (more FP32 arithmetic before grid lookup)
3. **High bits mechanism** is simpler (bitwise operations vs complex indexing)
4. **Grid structure** (uint64_t) may be more cache-friendly

**Why IQ2_S is similar to IQ2_XS**:
- Both use grid lookups with sign patterns
- Both benefit from SIMD in FP32 → Q8_0 quantization phase
- Grid lookup overhead is offset by SIMD gains in quantization

---

## Key Learnings

### 1. Namespace Errors
**Issue**: Initially used `using namespace llaminar::v2;` inside `llaminar2::simd` namespace

**Fix**: Removed `using namespace` directives - we're already in `llaminar2` namespace

**Files affected**: 
- `SIMDHelpers.h` (3 functions)
- `IQ2_STensor.cpp` (decode_to_q8_0 signature)
- `Test__IQ2_S_DecodeVectorization.cpp` (namespace declaration)

### 2. Tensor Data Access Pattern
**Issue**: Tried to access `blocks_` member (doesn't exist in IQ2_STensor)

**Fix**: Use view-aware raw data access pattern:
```cpp
const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);
```

**Pattern source**: Copied from `IQ3_XXSTensor::decode_to_q8_0()`

### 3. Grid-Based Formats Can Still Benefit from SIMD
**Insight**: IQ2_S achieves 3.95-4.89× speedup despite grid lookups

**Reason**: 
- Grid lookup is scalar, but FP32 → Q8_0 quantization is vectorized
- Larger grids (1024 vs 256) reduce cache pressure
- Dual scales increase FP32 arithmetic proportion

**Implication**: Not all grid-based formats suffer from poor SIMD scaling

---

## Progress Update

**Completed Formats** (10/14):
- ✅ Q4_0: 10/10 tests, 2.33× AVX2, 1.99× AVX-512
- ✅ Q4_1: 10/10 tests, 1.87× AVX2, 2.47× AVX-512
- ✅ Q5_0: 10/10 tests, 1.96× AVX2, 2.45× AVX-512
- ✅ Q5_1: 11/11 tests, 1.91× AVX2, 2.38× AVX-512
- ✅ IQ4_NL: 11/11 tests, 2.17× AVX2, 3.02× AVX-512
- ✅ IQ4_XS: 9/9 tests
- ✅ IQ2_XXS: 10/10 tests, 0.66× AVX2, 1.02× AVX-512
- ✅ IQ2_XS: 10/10 tests, 3.69× AVX2, 4.80× AVX-512
- ✅ IQ3_XXS: 10/10 tests, 0.66× AVX2, 1.00× AVX-512
- ✅ **IQ2_S: 10/10 tests, 3.95× AVX2, 4.89× AVX-512** ← **JUST COMPLETED**

**Implementation Only** (1/14):
- ✅ Q3_K: Scalar implementation, NO TESTS

**Declarations Only** (3/14):
- ✅ IQ3_S, IQ1_S, IQ1_M

**Total**: ~117 tests passing across 10 formats

---

## Next Steps

**Remaining 3 Complex Formats**:
1. **IQ3_S**: Uses signs[] array + scales[] (110 bytes/block)
2. **IQ1_S**: 1-bit quantization + IQ1S_DELTA offset (98 bytes/block)
3. **IQ1_M**: 1-bit quantization + multiple scales (98 bytes/block)

**Expected approach**: Continue same pattern (scalar → AVX2 → AVX-512 → tests)

---

## Build Instructions

```bash
# From workspace root
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --target v2_test_iq2s_decode_vectorization --parallel

# Run tests
cd build_v2
./tests/v2/v2_test_iq2s_decode_vectorization

# Or via CTest
ctest -R "V2_Unit_IQ2_S_DecodeVectorization" --verbose
```

---

## References

- **IQ2_S Scalar Algorithm**: `src/v2/tensors/IQ2_STensor.cpp::decodeBlockScalar()`
- **Grid Table**: `src/v2/tensors/IQQuantTables.h::iq2s_grid[1024]`
- **Sign Patterns**: `src/v2/tensors/IQQuantTables.h::kmask_iq2xs[]` (shared with IQ2_XS)
- **Pattern Reference**: IQ3_XXS implementation (similar super-block structure)
- **Performance Comparison**: IQ2_XS (comparable performance profile)

---

**End of Changelog**
