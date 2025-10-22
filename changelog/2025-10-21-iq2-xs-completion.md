# IQ2_XS Implementation Complete - October 21, 2025

## Summary

Successfully implemented **IQ2_XS** (2.3125 bits per weight) quantized tensor support for Llaminar. Second format in the IQ2 series after IQ2_XXS, achieving **13.84× compression** with improved quality through explicit scales and a larger grid codebook.

## Test Results

**✅ 9/9 tests passing** (0 ms total)
- ✅ BasicDecoding
- ✅ DualScaleExtraction
- ✅ SignApplication
- ✅ GridLookup512
- ✅ MultipleBlocks
- ✅ ScaleAlternation
- ✅ BF16Decoding
- ✅ SpanDecoding
- ✅ ErrorHandling

## Technical Details

### IQ2_XS Specification

**Block Structure** (74 bytes for 256 elements):
```c
struct IQ2_XSBlock {
    uint16_t d;        // 2 bytes - FP16 scale factor
    uint16_t qs[32];   // 64 bytes - packed 9-bit grid indices + 7-bit sign indices
    uint8_t scales[8]; // 8 bytes - explicit scales (2 per sub-block as 4-bit nibbles)
};
```

**Compression Metrics:**
- Block size: 74 bytes (256 elements)
- Compression ratio: **13.84×** (1024 bytes FP32 → 74 bytes IQ2_XS)
- Bits per weight: **2.3125 bpw**
- Scale count: 16 (8 sub-blocks × 2 scales per sub-block)

### Decoding Algorithm

Implements GGML `dequantize_row_iq2_xs` algorithm (ggml-quants.c line 2303-2328):

1. **FP16 Scale Extraction**: `d = GGML_FP16_TO_FP32(block.d)`

2. **Dual Scale Extraction** (per sub-block):
   ```c
   db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25   // Low nibble
   db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25    // High nibble
   ```

3. **Grid Index and Sign Extraction**:
   ```c
   const uint16_t packed = qs[4*ib32 + l];
   const uint16_t grid_idx = packed & 511;        // 9 bits (0-511)
   const uint8_t signs = ksigns_iq2xs[packed >> 9]; // 7 bits (0-127)
   ```

4. **Scale Alternation**: `db[l/2]` alternates every 2 groups:
   - Groups 0-1: Use `db[0]`
   - Groups 2-3: Use `db[1]`

5. **Grid Lookup and Sign Application**:
   ```c
   const uint8_t *grid = (const uint8_t *)(iq2xs_grid + grid_idx);
   const float scale = db[l/2];
   for (int j = 0; j < 8; ++j) {
       y[j] = scale * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
   }
   ```

### Key Differences from IQ2_XXS

| Feature | IQ2_XXS | IQ2_XS | Notes |
|---------|---------|--------|-------|
| **Block size** | 66 bytes | 74 bytes | +8 bytes for explicit scales |
| **Compression** | 15.52× | 13.84× | Slightly lower but better quality |
| **Bits/weight** | 2.0625 | 2.3125 | +0.25 bits for improved accuracy |
| **Grid size** | 256 entries | 512 entries | Larger codebook |
| **Grid table** | iq2xxs_grid | iq2xs_grid | Different lookup table |
| **Scales storage** | Packed in qs[] | Explicit scales[8] | Cleaner design |
| **Scales/sub-block** | 1 (4-bit) | 2 (4-bit nibbles) | Dual scales |
| **Grid index bits** | 8 bits | 9 bits | Wider range |
| **qs[] format** | Complex packing | Simple: grid\|sign | Easier to implement |
| **Quality** | Extreme compression | Better accuracy | Trade-off |

### Lookup Tables Used

1. **kmask_iq2xs[8]**: Sign bit masks (shared with IQ2_XXS)
2. **ksigns_iq2xs[128]**: Sign patterns (shared with IQ2_XXS)
3. **iq2xs_grid[512]**: ✅ **NEW** - 512-entry grid table (4096 bytes)
   - Extracted from ggml-common.h lines 595-730
   - Each entry: uint64_t containing 8 packed bytes
   - Range: 0x0808080808080808 to 0x2b2b2b2b2b2b2b2b

## Implementation Files

### Created Files

1. **src/tensors/IQ2_XSTensor.h** (295 lines)
   - Complete tensor implementation
   - Key components:
     * `IQ2_XSBlock` struct with static_assert
     * `decodeBlock()` - Core GGML decode algorithm
     * `decodeRow()` - Row-wise streaming decode
     * `decodeSpan()` - Arbitrary range decode
     * `decode_to_fp32()` - Full tensor decode with OpenMP
     * `decode_to_bf16()` - BF16 conversion path
     * `decodeRowToBF16()` - Row-wise BF16 decode
   - All APIs implemented and tested

2. **tests/test_iq2_xs_tensor.cpp** (560 lines)
   - 9 comprehensive test cases
   - **New tests** (specific to IQ2_XS):
     * **DualScaleExtraction**: Validates nibble unpacking (low/high)
     * **GridLookup512**: Tests 9-bit grid indices (0, 1, 256, 511)
     * **ScaleAlternation**: Verifies `db[l/2]` pattern
   - **Adapted tests** (from IQ2_XXS):
     * BasicDecoding, SignApplication, MultipleBlocks
     * BF16Decoding, SpanDecoding, ErrorHandling
   - Helper function: `create_iq2_xs_raw_data()`
     - Takes `uint16_t grid_indices[32]` (9-bit values 0-511)
     - Takes `uint8_t sign_indices[32]` (7-bit values 0-127)
     - Takes `uint8_t scales[8]` (explicit scale bytes)
     - Packs into 74-byte IQ2_XSBlock format
     - Much simpler than IQ2_XXS packing

### Modified Files

3. **src/tensors/IQQuantTables.h**
   - Added `iq2xs_grid[512]` lookup table (4096 bytes)
   - 512 uint64_t entries extracted from GGML
   - Complete Doxygen documentation
   - File now contains 4 tables:
     * kmask_iq2xs[8] (shared)
     * ksigns_iq2xs[128] (shared)
     * iq2xxs_grid[256] (IQ2_XXS-specific)
     * iq2xs_grid[512] (IQ2_XS-specific)

4. **src/tensors/QuantizedTensorBase.h**
   - Line ~37: Added `IQ2_XS` to QuantType enum
   - Line ~260: Added `case QuantType::IQ2_XS: return "IQ2_XS";`
   - Now supports both IQ2_XXS and IQ2_XS

5. **CMakeLists.txt**
   - Lines 1864-1868: Added test_iq2_xs_tensor target
   - Placed after IQ2_XXSTensor tests
   - Standard test configuration (30s timeout)

### Bug Fixes

**Build Error**: Missing `fp32_to_bf16` function
- **Root Cause**: Forgot to include `BFloat16.h`
- **Fix**: Added `#include "../utils/BFloat16.h"` to IQ2_XSTensor.h
- **Correction**: Changed `fp32_to_bf16()` → `bfloat16::from_float()`
- **Result**: Clean build and all tests passing

## GGML Reference Sources

1. **llama.cpp/ggml/src/ggml-quants.c**
   - Line 2303-2328: `dequantize_row_iq2_xs()` ✅ EXTRACTED
   
2. **llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/types.comp**
   - Block structure definition ✅ EXTRACTED
   
3. **llama.cpp/ggml/src/ggml-common.h**
   - Line 595-730: `iq2xs_grid[512]` table ✅ EXTRACTED

## Code Quality

**Implementation Matches GGML Specification**:
- ✅ Block structure: 74 bytes (2 + 64 + 8)
- ✅ Decode algorithm: Line-by-line match with ggml-quants.c
- ✅ Dual scale extraction: Low/high nibble handling
- ✅ Grid lookup: 9-bit indices (0-511 range)
- ✅ Scale alternation: `db[l/2]` pattern verified
- ✅ Sign application: ksigns_iq2xs[128] table
- ✅ OpenMP parallelization for decode_to_fp32
- ✅ BF16 conversion support

**Test Coverage**:
- ✅ All decode paths tested (FP32, BF16)
- ✅ Edge cases covered (empty tensors, out-of-bounds)
- ✅ Multi-block parallelization validated
- ✅ All new features tested (dual scales, 512-grid, alternation)

## Performance Characteristics

**When to Use IQ2_XS**:
- ✅ Better quality needed than IQ2_XXS
- ✅ Can afford 8 extra bytes per block
- ✅ Models where 0.25 extra bits/weight matters
- ✅ Importance-weighted quantization (imatrix available)

**When to Use IQ2_XXS Instead**:
- ✅ Maximum compression required
- ✅ Every byte counts (embedded, edge devices)
- ✅ Quality trade-off acceptable
- ✅ Extreme memory constraints

**Compression Comparison**:
```
Format      Block   Compression   Bits/Weight   Quality
IQ2_XXS:    66 B    15.52×        2.0625 bpw    Extreme
IQ2_XS:     74 B    13.84×        2.3125 bpw    Better (+8 bytes)
IQ2_S:      TBD     TBD           2.5625 bpw    Best (next)
```

## Methodology

**Implementation Process** (same as IQ2_XXS):
1. ✅ Research GGML reference implementation
2. ✅ Extract decode algorithm from ggml-quants.c
3. ✅ Identify block structure from Vulkan shaders
4. ✅ Extract lookup tables from ggml-common.h
5. ✅ Implement tensor class (IQ2_XSTensor.h)
6. ✅ Create comprehensive tests (9 tests)
7. ✅ Update build configuration (CMakeLists.txt)
8. ✅ Build and verify (9/9 tests passing)
9. ✅ Document completion (this changelog)

**Key Learnings**:
- IQ2_XS is simpler to implement than IQ2_XXS
  * Simpler qs[] packing (plain uint16_t)
  * Explicit scales array (no complex bit extraction)
  * Straightforward grid lookup (simple 9-bit mask)
- Dual scale logic adds complexity but improves accuracy
- Larger grid (512 vs 256) provides more quantization options
- Test data generation is much easier than IQ2_XXS

## Next Steps

### Immediate (IQ2_S)
- [ ] Research IQ2_S format in ggml-quants.c (line 2330+)
- [ ] Determine block structure and size
- [ ] Extract iq2s_grid table (size TBD)
- [ ] Implement IQ2_STensor.h
- [ ] Create comprehensive tests
- [ ] Complete IQ2 family (XXS, XS, S)

### Future (Complete IQ Series)
- [ ] **IQ3 family**: IQ3_XXS, IQ3_S, IQ3_M (3.0625 - 3.4375 bpw)
- [ ] **IQ4 family**: IQ4_NL, IQ4_XS (4.0 - 4.5 bpw)
- [ ] **IQ1 family**: IQ1_S, IQ1_M (1.5 - 1.75 bpw)
- [ ] Complete IQ format coverage (9 formats total)

## Session Summary

**Duration**: ~30 operations (18 pre-summary + 12 post-summary)

**Achievements**:
1. ✅ Complete IQ2_XS implementation (295 lines)
2. ✅ Comprehensive test suite (560 lines, 9 tests)
3. ✅ All tests passing on first run
4. ✅ Build configuration updated
5. ✅ Documentation complete

**Challenges Overcome**:
- ✅ Missing BF16 include (quick fix)
- ✅ Wrong BF16 conversion function (corrected)
- ✅ CMake cache regeneration (standard workflow)

**Quality Metrics**:
- Code: Matches GGML specification line-by-line
- Tests: 9/9 passing (100%)
- Build: Clean with no warnings
- Documentation: Complete and comprehensive

## Status Update

**IQ Format Progress: 2/9 Complete**

| Format | Status | Compression | Bits/Weight | Notes |
|--------|--------|-------------|-------------|-------|
| IQ2_XXS | ✅ Complete | 15.52× | 2.0625 | Extreme compression |
| IQ2_XS | ✅ Complete | 13.84× | 2.3125 | **This release** |
| IQ2_S | ⏳ Next | TBD | 2.5625 | Best IQ2 quality |
| IQ3_XXS | 🔲 Pending | TBD | 3.0625 | - |
| IQ3_S | 🔲 Pending | TBD | 3.4375 | - |
| IQ3_M | 🔲 Pending | TBD | 3.4375 | - |
| IQ4_NL | 🔲 Pending | TBD | 4.0 | - |
| IQ4_XS | 🔲 Pending | TBD | 4.5 | - |
| IQ1_S | 🔲 Pending | TBD | 1.5 | Extreme (future) |
| IQ1_M | 🔲 Pending | TBD | 1.75 | Extreme (future) |

**Completion Rate**: 22.2% (2/9 formats)

---

**Author**: David Sanftenberg  
**Date**: October 21, 2025  
**Commit**: IQ2_XS implementation complete - 9/9 tests passing
