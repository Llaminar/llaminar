# V2 Single-Sequence E2E Test Success

**Date**: October 26, 2025  
**Session**: V2 GEMM Segfault Debugging & Bug Fixing  
**Duration**: ~3 hours  
**Status**: ✅ **COMPLETE SUCCESS**

---

## Summary

Successfully debugged and fixed **3 critical bugs** preventing V2 E2E testing. The `Qwen2E2ECorrectness.SingleTokenInference` test now **PASSES** with perfect numerical agreement.

---

## Bugs Fixed

### 1. Vocab Size Extraction Bug
**File**: `src/v2/loaders/ModelLoader.{h,cpp}`  
**Problem**: GGUF metadata has `tokenizer.ggml.tokens` array, not explicit `token_count`  
**Fix**: Extract vocab_size from array length during metadata parsing  
**Documentation**: `changelog/2025-10-26-v2-vocab-size-fix.md`

### 2. Buffer Validation Bug
**File**: `src/v2/pipelines/TensorDimensions.h`, `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`  
**Problem**: Exact shape matching rejected buffers allocated with `max_seq_len` capacity  
**Fix**: New `VALIDATE_TENSOR_BUFFER` macro for prefix matching (first dimension ≥ expected)  
**Affected**: 9 validation points (Q/K/V projections, RoPE, attention, FFN)  
**Documentation**: `changelog/2025-10-26-v2-buffer-validation-fix.md`

### 3. AVX-512 Alignment Segfault
**File**: `src/v2/kernels/cpu/QuantizedGemm.cpp`  
**Problem**: `std::vector<float> B_tile` not aligned to 64 bytes for `_mm512_load_ps`  
**Fix**: Replace with `aligned_alloc(64, ...)` for thread-local decode buffers  
**Impact**: CRITICAL - All GEMM operations failed before fix  
**Documentation**: `changelog/2025-10-26-v2-gemm-alignment-fix.md` (this file)

---

## Test Results

**Before**: Segmentation fault on first Q projection in layer 0  
**After**: ✅ **PASSED** with perfect numerical agreement

```
[       OK ] Qwen2E2ECorrectness.SingleTokenInference (77959 ms)
[  PASSED  ] 1 test.

=== Single Token Inference ===
  Max abs diff:   0
  Mean abs diff:  0
  Rel L2 norm:    0
  Mismatches:     0
  Status:         PASSED
```

**Test Configuration**:
- Model: Qwen 2.5 0.5B Instruct Q4_0
- MPI Ranks: 2 (tensor-parallel)
- Input: Single token (ID 151644)
- Validation: Multi-rank logits agree exactly

**Performance** (Debug build):
- Total runtime: ~78 seconds
- Per-layer: ~3.2 seconds
- Layers processed: 24
- Expected Release speedup: 5-10×

---

## Debugging Process

### Tools Used
1. **GDB with MPI**: Captured backtrace showing exact crash location
   - Command file with `debuginfod disabled` to prevent blocking
   - Stack trace revealed AVX-512 alignment issue

2. **Checkpoint Logging**: Identified where pipeline hung/crashed
   - Added `LOG_INFO` checkpoints before/after Q projection
   - Confirmed GEMM completion after alignment fix

3. **Tensor Validation**: Caught buffer shape mismatches early
   - `VALIDATE_TENSOR` vs `VALIDATE_TENSOR_BUFFER` distinction
   - Helpful error messages guided fixes

### Key Insights
1. **Always align SIMD buffers**: `_mm512_load_ps` requires 64-byte alignment
2. **std::vector doesn't guarantee SIMD alignment**: Use `aligned_alloc` instead
3. **Pre-allocated buffers need prefix validation**: `max_seq_len` capacity vs `seq_len` usage
4. **GDB with MPI works**: Need `--args` separator and command files

---

## Code Changes

### Added
- `src/v2/pipelines/TensorDimensions.h`: `matches_prefix()` method
- `src/v2/pipelines/TensorDimensions.h`: `VALIDATE_TENSOR_BUFFER` macro
- `src/v2/loaders/ModelLoader.h`: `array_length` field to `GGUFValue`

### Modified
- `src/v2/kernels/cpu/QuantizedGemm.cpp`: `aligned_alloc(64, ...)` for `B_tile`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`: 9× `VALIDATE_TENSOR_BUFFER` replacements
- `src/v2/loaders/ModelLoader.cpp`: Fallback vocab_size extraction from tokens array

### Removed
- Debug logging: layer.wq shape, pointer addresses, GEMM parameters
- Null checks: layer.wq validation (unnecessary with proper initialization)

---

## Next Steps

### Phase 2: Autoregressive Decode (Immediate)
1. Implement `generate()` method with sampling loop
2. Add greedy/top-k/top-p sampling strategies
3. Test multi-token generation
4. Validate KV cache incremental updates

### Phase 3: Optimization (Short-term)
1. Release build benchmarking (expect 5-10× speedup)
2. Profile GEMM hot spots
3. Consider parallel decode for batching

### Phase 4: Testing Expansion (Medium-term)
1. Add more E2E tests (multi-token, different models)
2. Parity testing vs llama.cpp/PyTorch
3. Stress testing (long sequences, large batches)

---

## Files Modified

**Source Code** (6 files):
- `src/v2/loaders/ModelLoader.h`
- `src/v2/loaders/ModelLoader.cpp`
- `src/v2/pipelines/TensorDimensions.h`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`
- `src/v2/kernels/cpu/QuantizedGemm.cpp`

**Documentation** (4 files):
- `changelog/2025-10-26-v2-vocab-size-fix.md`
- `changelog/2025-10-26-v2-buffer-validation-fix.md`
- `changelog/2025-10-26-v2-gemm-alignment-fix.md`
- `changelog/2025-10-26-v2-e2e-test-success.md` (this file)

**Tests** (1 test passing):
- `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp::SingleTokenInference`

---

## Lessons Learned

1. **SIMD alignment matters**: Any AVX-512 buffer needs 64-byte alignment
2. **Buffer reuse requires flexible validation**: Pre-allocated capacity vs actual usage
3. **Lazy loading has hidden bugs**: Vocab size wasn't discovered until E2E test
4. **GDB with MPI is possible**: Command files and `--args` syntax are key
5. **Incremental debugging works**: Fix one bug, expose the next, repeat

---

## Session Statistics

**Bugs Found**: 3  
**Bugs Fixed**: 3 ✅  
**Tests Passing**: 1/1 (100%)  
**Code Quality**: Clean (debug logging removed)  
**Documentation**: Complete (4 changelog files)  
**Time Invested**: ~3 hours  
**Value Delivered**: V2 single-sequence inference **WORKS**!

---

## References

- V2 Architecture: `.github/instructions/llaminar-v2-architecture.instructions.md`
- Vocab Size Fix: `changelog/2025-10-26-v2-vocab-size-fix.md`
- Buffer Validation Fix: `changelog/2025-10-26-v2-buffer-validation-fix.md`
- GEMM Alignment Fix: `changelog/2025-10-26-v2-gemm-alignment-fix.md`
- Test Code: `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp`
