# E2E Batch Test Debugging Session - 2025-01-22

## Summary
Investigated `V2_E2E_Qwen2Correctness` test failures where Sequence 0 passes but Sequence 1 fails with large divergence (max_diff ~21, mean_diff ~3). Multiple fixes attempted but issue persists.

## Test Setup
- **ComprehensiveBatchParity Test**:
  - Sequential: Run each sequence separately with `batch_size=1`
    - Sequence 0: 4 tokens → `batch_size=1, padded_seq_len=4, effective_seq_len=4`
    - Sequence 1: 2 tokens → `batch_size=1, padded_seq_len=2, effective_seq_len=2`
  - Batched: Run together with `batch_size=2`
    - `batch_size=2, padded_seq_len=4, effective_seq_len=8`
    - Layout: rows 0-3 (Seq 0), rows 4-7 (Seq 1: 2 real + 2 padding)
  - Compare logits for each sequence

## Symptoms
- **Sequence 0**: ✅ PASS (max_diff=0)
- **Sequence 1**: ❌ FAIL (max_diff=21.2195, mean_diff=3.14686)
- **First Divergence**: FINAL_NORM (after all 24 layers)
- **Embedding Check**: ✅ PASS for both sequences

## Investigation History

### Initial Hypothesis: Garbage Mask Bug
**Finding**: `MpiAttentionOrchestrator` was passing garbage mask pointer when `needs_mask=false`.

**Fix 1** (lines 230-560 in `MpiAttentionOrchestrator.cpp`):
```cpp
// Pass nullptr instead of garbage when mask not needed
const TensorBase *attention_mask_ptr = (needs_mask && attention_mask) ? attention_mask : nullptr;
```

**Result**: Unit test passes, but E2E still fails identically.

### Second Hypothesis: Residual Connection Loops
**Finding**: Residual connection loops only processed `effective_seq_len * d_model` elements instead of full batch.

**Fix 2** (lines 792-800, 1027-1035 in `Qwen2Pipeline.cpp`):
```cpp
// OLD (BUG):
for (size_t i = 0; i < effective_seq_len * d_model_; ++i) { ... }

// NEW (FIXED):
const size_t total_elements = batch_size_ * padded_seq_len_ * d_model_;
for (size_t i = 0; i < total_elements; ++i) { ... }
```

**Result**: E2E still fails identically.

### Third Hypothesis: FusedRMSNormQuantize Row Count Bug
**Finding**: `FusedRMSNormQuantize` kernel was being called with `effective_seq_len` but we thought it should use `batch_size * padded_seq_len`.

**Fix 3** (lines 519, 861 in `Qwen2Pipeline.cpp`):
```cpp
// Added total_rows calculation
const int total_rows = batch_size_ * padded_seq_len_;
rmsnorm_kernel->execute(..., total_rows, d_model_, ...);
```

**Discovery**: `effective_seq_len` IS ALREADY `batch_size * padded_seq_len` (line 348)! So `total_rows` is redundant.

**Fix 3a**: Also fixed scale registration and dequantization loops (lines 536, 546, 556, 886, 896, 906) to use `total_rows` instead of `effective_seq_len`.

**Result**: E2E still fails identically (max_diff=21.2195).

## Current Understanding

### What Works
1. ✅ **Embeddings**: Both sequences embedded correctly (test confirms EMBEDDING snapshot passes)
2. ✅ **Sequence 0**: Complete pipeline works correctly
3. ✅ **Batch Size Calculations**: `effective_seq_len = batch_size * padded_seq_len` computed correctly
4. ✅ **Memory Layout**: Sequences laid out correctly (rows 0-3 for Seq 0, rows 4-7 for Seq 1)
5. ✅ **Test Extraction**: Test correctly extracts rows 4-5 from batch for Sequence 1 comparison

### What Fails
1. ❌ **Sequence 1 Processing**: Something between EMBEDDING and FINAL_NORM corrupts Sequence 1 data
2. ❌ **Layer-by-Layer**: All layers show failures for Sequence 1 (layer0-23), suggesting early corruption

## Debugging Evidence

### Batch Size Debug Output
```
[BATCH DEBUG] batch_size_=2, padded_seq_len_=4, effective_seq_len=8
[BATCH DEBUG]   Sequence 0: 4 tokens
[BATCH DEBUG]   Sequence 1: 2 tokens
```

### Test Results Pattern
```
EMBEDDING: ✓ PASS (max_diff=0, mean_diff=0)  # Both sequences
FINAL_NORM: ✗ FAIL (max_diff=43.4354, mean_diff=7.58405)  # Sequence 1 only
layer0_ATTENTION_OUTPUT: ✗ FAIL (max_diff=0.341318)  # Small error, Seq 1
layer3_FFN_RESIDUAL: ✗ FAIL (max_diff=1690.63)  # Huge error, Seq 1
layer10+ FFN_RESIDUAL: ~1682-1692  # Plateaus, Seq 1
```

## Code Changes Made

### Files Modified
1. **src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp**
   - Fixed mask handling (conditional `nullptr` passing)
   - Lines: 363-385, 437-467, 540-560

2. **src/v2/pipelines/qwen/Qwen2Pipeline.cpp**
   - Fixed residual connection loops (lines 792-800, 1027-1035)
   - Added `total_rows` calculations (lines 518, 868)
   - Fixed `set_row_scales` calls (lines 536, 886)
   - Fixed scale validation loops (lines 540-546, 890-896)
   - Fixed dequantization loops (lines 550-556, 900-906)
   - Added debug logging (lines 350-355)

### Tests Created
1. **tests/v2/unit/Test__MpiAttentionOrchestrator_Masking.cpp**
   - Validates garbage mask correctly ignored
   - ✅ PASSING

## Hypotheses Remaining

### Hypothesis 4: Unfused RMSNorm Path Issue
The code has two paths:
- **Fused**: `FusedRMSNormQuantize` (RMSNorm + INT8 quantization)
- **Unfused**: Separate `RMSNorm` + implicit quantization in GEMM

**Question**: Which path is the test using? If unfused, we may have missed a batch bug there.

**Check**: Add logging to determine which path executes during E2E test.

### Hypothesis 5: QKV Projection or Attention Kernel Issue
Even though attention orchestrator proven correct via debug logs, there could be bugs in:
- Q/K/V projection kernels (quantized GEMM)
- Attention kernel itself (masking, score computation)
- RoPE application

**Evidence Against**: Sequence 0 works perfectly, suggesting kernels are fundamentally correct.

### Hypothesis 6: Buffer Allocation or View Creation Issue
Buffers allocated with `max_seq_len = batch_size * max_seq_len_per_sequence`, but views created with `effective_seq_len`. Possible mismatch?

**Check**: Verify buffer sizes vs actual usage, especially for pre-allocated buffers.

### Hypothesis 7: KV Cache Indexing Bug
KV cache indexed per sequence. For batched execution, cache updates might not correctly handle Sequence 1 offset.

**Check**: Add logging around K/V cache updates to verify correct indexing for Sequence 1.

## Next Steps

### Immediate Actions
1. **Determine Code Path**: Add logging to confirm fused vs unfused path in E2E test
2. **Check Unfused Path**: If unfused, audit all `effective_seq_len` usage in that path
3. **Add Granular Logging**: Log first few values of tensors after each operation for Sequence 1
4. **Minimal Repro**: Create minimal test with just layer 0 to isolate where divergence starts

### Investigation Strategy
1. **Binary Search**: Comment out layers to find exactly where divergence occurs
2. **Snapshot Comparison**: Use layer-by-layer snapshots to identify first diverging operation
3. **Side-by-Side Debug**: Run sequential Seq 1 and batched Seq 1 with identical logging, compare outputs

### Code Audit Checklist
- [ ] All `effective_seq_len` usages (currently ~100 matches)
- [ ] All buffer allocations sized for batch
- [ ] All tensor views created with correct shapes
- [ ] All snapshot captures handle batch layout
- [ ] KV cache indexing for batched sequences
- [ ] Position ID generation for batched sequences

## Performance Impact
All fixes made are on hot paths but use existing variables, so minimal performance impact expected.

## References
- E2E Test: `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp`
- Pipeline: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`
- Attention: `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp`
- Fused Kernel: `src/v2/kernels/cpu/fused/FusedRMSNormQuantize.cpp`

## Conclusion
Despite multiple targeted fixes addressing mask handling, residual connections, and FusedRMSNormQuantize batch processing, **Sequence 1 continues to fail with identical error pattern**. This suggests a more fundamental issue with how batch data flows through the pipeline, possibly in:
- Unfused RMSNorm path (if that's what's being used)
- Buffer indexing/view creation
- KV cache handling for batched sequences
- Some other operation not yet audited

The fact that **Sequence 0 always passes perfectly** indicates the algorithms are correct, but **batch indexing or layout is wrong for non-first sequences**.

---
*Session Duration: ~2 hours*
*Files Modified: 3*
*Tests Created: 1*
*Status: Issue persists, further investigation required*
