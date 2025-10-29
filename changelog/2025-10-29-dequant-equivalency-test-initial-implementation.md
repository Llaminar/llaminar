# Dequant Equivalency Test Suite - Initial Implementation

**Date**: October 29, 2025  
**Author**: David Sanftenberg  
**Status**: In Progress - Linking Issues

## Objective

Create integration tests that verify bit-exact equivalence between Llaminar's `IBlockDecoder` implementations and llama.cpp's reference dequantization routines (`ggml-quants`).

## Background

All 18 quantized tensor types in Llaminar V2 already have complete inline `IBlockDecoder` implementations (verified in `changelog/2025-10-29-iblockdecoder-implementation-verification.md`). User requested equivalency tests to prove these produce identical FP32 outputs to llama.cpp's canonical dequantization functions.

## Implementation Progress

### ✅ Completed

1. **Test File Created**: `tests/v2/integration/Test__DequantEquivalency.cpp`
   - Comprehensive test framework with 5 initial test cases (Q8_0, IQ4_NL, Q4_0, Q4_1, Q6_K)
   - Comparison utilities (`compareOutputs`) with configurable tolerance
   - Detailed mismatch reporting (first 5 mismatches, max abs/rel diff)
   - Transpose detection placeholder (user mentioned potential transpose in IBlockDecoder)
   - Model loading via `ModelLoader::loadModel` and `loadTensor`

2. **CMake Integration**: Added to `tests/v2/CMakeLists.txt`
   - Test target: `v2_test_dequant_equivalency`
   - Test label: `V2;Integration;Quantization;IBlockDecoder;LlamaCppParity;Correctness`
   - Single-rank execution (MPI_PROCS 1)

3. **Block Structure Verification**:
   - Confirmed Llaminar's block structures match llama.cpp (GGML):
     - `Q8_0Block` ↔ `block_q8_0` (34 bytes)
     - `IQ4_NLBlock` ↔ `block_iq4_nl` (18 bytes)
     - `Q4_0Block` ↔ `block_q4_0` (18 bytes)
     - All K-quant blocks match sizes
   - Both use `ggml_half` (uint16_t) for FP16 scale factors
   - Identical memory layouts (verified with `static_assert`)

4. **llama.cpp Dequant API Identified**:
   - Header: `llama.cpp/ggml/src/ggml-quants.h`
   - 20 dequant functions available:
     - Q-series: `dequantize_row_q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`
     - K-quants: `dequantize_row_q2_K` through `q8_K`
     - IQ-quants: `dequantize_row_iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`
   - Signature: `void dequantize_row_TYPE(const block_TYPE *x, float *y, int64_t k)`

### 🔄 Partially Complete

1. **Test Code Compiles**: All C++ code compiles successfully
   - Correct includes: `../../../src/v2/tensors/Tensors.h`, `../../../src/v2/loaders/ModelLoader.h`
   - Correct API usage: `ModelLoader::loadModel()`, `loadTensor()`
   - Correct tensor names: `"token_embd.weight"` (verified from existing tests)

2. **llama.cpp Integration Attempted**:
   - **Approach 1**: Link against pre-built llama.cpp libraries
     - Found: `llama.cpp/build/bin/libggml.so`, `libggml-cpu.so`
     - Issue: Dequant functions not exported in shared libraries (`nm -D` shows no symbols)
   
   - **Approach 2**: Direct compilation of `ggml-quants.c`
     - Files: `llama.cpp/ggml/src/ggml-quants.c`, `llama.cpp/ggml/src/ggml-cpu/quants.c`
     - Issue: Missing dependencies (`ggml_abort`, `ggml_row_size`, `ggml_type_size`, `ggml_table_f32_f16`)
     - These require linking the full GGML library ecosystem

### ❌ Blocking Issues

**Linking Problem**: Cannot link against llama.cpp's dequant functions without pulling in entire GGML dependency tree.

**Error Summary**:
```
undefined reference to `ggml_abort'
undefined reference to `ggml_row_size'  
undefined reference to `ggml_type_size'
undefined reference to `ggml_table_f32_f16'
```

## Alternative Solutions

### Option 1: Build llama.cpp as CMake ExternalProject (Recommended)

Configure llama.cpp as a proper CMake dependency:

```cmake
include(ExternalProject)
ExternalProject_Add(llama_cpp
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../llama.cpp
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
    BUILD_COMMAND cmake --build . --target ggml
    INSTALL_COMMAND ""
)
target_link_libraries(v2_test_dequant_equivalency ggml)
```

**Pros**: Cleanest integration, proper dependency management  
**Cons**: Adds build complexity, increases compilation time

### Option 2: Reference Implementation in Llaminar (Simpler)

Create simple standalone dequant reference implementations in Llaminar's test directory that mirror llama.cpp's algorithms:

```cpp
// tests/v2/integration/dequant_reference/Q8_0Dequant.h
namespace reference {
    void dequantize_q8_0(const Q8_0Block& block, float* output) {
        const float d = fp16_to_fp32(block.d);
        for (size_t i = 0; i < 32; ++i) {
            output[i] = block.qs[i] * d;
        }
    }
}
```

Then compare Llaminar's `IBlockDecoder::decode_block_at()` against `reference::dequantize_q8_0()`.

**Pros**: No external dependencies, fast build, easy to debug  
**Cons**: Must manually port llama.cpp algorithms (risk of divergence)

### Option 3: Static Stub Library

Extract minimal ggml functions needed for quants.c:

```c
// tests/v2/integration/ggml_stubs.c
void ggml_abort(const char* msg) { fprintf(stderr, "%s\n", msg); abort(); }
size_t ggml_row_size(int type, int n) { /* minimal implementation */ }
const float* ggml_table_f32_f16 = /* lookup table */;
```

**Pros**: Reuses llama.cpp's exact code  
**Cons**: Maintenance burden, fragile to llama.cpp updates

## Recommended Next Steps

1. **Prototype Option 2** (Reference Implementation):
   - Start with Q8_0 (simplest: `output[i] = qs[i] * d`)
   - Verify Llaminar matches reference
   - If successful, port remaining 17 types

2. **Document Transpose Issue**:
   - User mentioned IBlockDecoder may transpose
   - Add test to detect transpose: compare both `[i]` and `[transpose(i)]` indices
   - Document expected behavior

3. **Extend Test Coverage**:
   - Current: 5 tests (Q8_0, IQ4_NL, Q4_0, Q4_1, Q6_K)
   - Remaining: 13 types (Q2_K, Q3_K, Q4_K, Q5_K, Q8_K, IQ1_M, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS)
   - Goal: 18/18 coverage with all tests passing

4. **Model Availability Check**:
   - Q8_0 model: ✅ `/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf`
   - IQ4_NL model: Need to verify path (test uses `/workspaces/llaminar/models/qwen2.5-0.5b-instruct-iq4_nl.gguf`)
   - Other formats: Use `GTEST_SKIP()` if model not available

## Test Design Details

### Test Pattern

```cpp
TEST_F(DequantEquivalencyTest, FORMAT_Equivalency) {
    // 1. Load quantized weight
    auto weight = loadWeight("token_embd.weight");
    auto tensor = std::dynamic_pointer_cast<FORMATTensor>(weight);
    
    // 2. Dequant with Llaminar
    for (each block) {
        tensor->decode_block_at(row, block_idx, llaminar_output + offset);
    }
    
    // 3. Dequant with reference (llama.cpp or standalone)
    dequantize_row_FORMAT(blocks, reference_output, cols);
    
    // 4. Compare with tolerance
    EXPECT_TRUE(compareOutputs(llaminar_output, reference_output, cols, tolerance));
}
```

### Comparison Metrics

- **Absolute difference**: `|llaminar[i] - reference[i]|`
- **Relative difference**: `abs_diff / |reference[i]|`
- **Mismatch count**: Elements exceeding tolerance
- **Max abs/rel diff**: Worst-case error
- **First 5 mismatches**: Detailed diagnostics

### Tolerances

- **Q8_0, Q4_0, Q4_1**: `1e-6` (high precision)
- **IQ4_NL, K-quants**: `1e-5` (quantization error tolerance)
- **IQ1/IQ2**: `1e-4` (aggressive quantization)

## Files Created

- ✅ `tests/v2/integration/Test__DequantEquivalency.cpp` (420 lines)
- ✅ `tests/v2/CMakeLists.txt` (updated with test target)
- ✅ `changelog/2025-10-29-dequant-equivalency-test-initial-implementation.md` (this file)

## Files Modified

- ✅ `tests/v2/CMakeLists.txt` (added `v2_test_dequant_equivalency` target)

## Build Status

**Configuration**: ✅ Passes  
**Compilation**: ✅ C++ code compiles  
**Linking**: ❌ Undefined references to GGML functions

## Next Session Goals

1. Decide between Option 1 (ExternalProject) vs Option 2 (Reference Implementation)
2. Implement chosen solution for Q8_0 test
3. Run first test and verify bit-exact equivalence
4. Extend to remaining 17 quantized formats
5. Document any transpose behavior discovered

## References

- **Block structures**: `llama.cpp/ggml/src/ggml-common.h` (lines 186-420)
- **Dequant API**: `llama.cpp/ggml/src/ggml-quants.h`
- **IBlockDecoder**: `src/v2/tensors/Tensors.h` (18 tensor classes)
- **Existing test pattern**: `tests/v2/unit/loaders/Test__ModelLoader.cpp` (tensor loading)
- **IBlockDecoder verification**: `changelog/2025-10-29-iblockdecoder-implementation-verification.md`
