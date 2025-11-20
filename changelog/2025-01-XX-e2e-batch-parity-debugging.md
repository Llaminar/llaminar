# E2E Batch Parity Debugging Session

## Date: 2025-01-XX

## Objective
Fix failing E2E batch parity tests (ComprehensiveBatchParity) that compare sequential vs batched execution.

## Root Cause Analysis

### Discoveries

1. **First Divergence Point**: `layer0_ATTENTION_CONTEXT` (not final norm or logits)
   - EMBEDDING: PASS (identical)
   - layer0_Q/K/V_PROJECTION: PASS (identical)
   - layer0_Q/K_ROPE: PASS (identical)  
   - layer0_ATTENTION_CONTEXT: **FAIL** (max_diff ~0.295 for seq 0, ~0.124 for seq 1)
   - All subsequent layers cascade from this initial divergence

2. **Mask Selection Bugs** (FIXED):
   - `MpiAttentionOrchestrator` (tensor-parallel path, lines 270-295): Always used causal masks
   - `MpiAttentionOrchestrator` (single-sequence path, lines 274-281): Always used causal masks
   - `GQAAttention::build_sequence_mask` (lines 63-88): Always used causal masks for batch_size > 1
   
   **Fix**: Respect `config.causal` flag and use `create_batch_padding_mask()` for non-causal attention

3. **Batch Layout Issues** (ATTEMPTED FIX):
   - `MpiAttentionOrchestrator::copy_head_slice` (lines 230-280): Didn't preserve batch structure
   - Needed separate paths for `effective_batch_size > 1` vs single-sequence
   - Expected layout: `[batch_size, seq_len, n_heads, head_dim]` not `[total_tokens, n_heads, head_dim]`
   
   **Fix**: Added batch-aware copying that preserves batch/sequence boundaries

4. **Output Gathering** (ATTEMPTED FIX):
   - `MpiAttentionOrchestrator` output collection (lines 405-440): Didn't preserve batch structure
   - Added batch-aware output gathering with proper stride calculations

## Files Modified

1. `src/v2/pipelines/AttentionUtils.h` (+85 lines):
   - Added `create_batch_padding_mask()` for non-causal batched attention
   - Masks only padding tokens, allows bi-directional attention

2. `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp` (3 locations):
   - Lines 270-295: Batched path mask selection
   - Lines 274-281: Single-sequence path mask selection  
   - Lines 230-280: Batch-aware head slice copying
   - Lines 405-460: Batch-aware output gathering

3. `src/v2/pipelines/attention/GQAAttention.cpp` (lines 63-88):
   - Mask selection based on `config.causal` flag
   - Uses `create_batch_padding_mask()` when causal=false

4. `tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp` (extensive):
   - Added compile-time check for ENABLE_PIPELINE_SNAPSHOTS
   - Modified to store sequential pipelines for snapshot access
   - Added layer-by-layer snapshot comparison infrastructure
   - Enabled snapshot capture for both sequential and batched execution

## Current Status

**Tests Still Failing** with same magnitude of divergence (~19-21 max_abs_diff in final logits).

### Hypothesis: Attention Kernel compute_batch Implementation Issue

The test path: E2E test → GQAAttention::compute_batch → CPUAttentionKernelT::compute_batch

**Key Observation**: E2E test doesn't use tensor-parallel path (no MPI strategy set), so our orchestrator fixes don't apply!

**Actual Path**: Uses `CpuAttentionKernelT<FP32Tensor>::compute_batch` directly.

**compute_batch Implementation** (`src/v2/kernels/cpu/CpuAttentionKernelT.h`, lines 580-654):
- Extracts per-sequence mask tiles from combined `[total_tokens, total_tokens]` mask
- Calls `compute_typed()` for each batch item independently
- Mask extraction looks correct (diagonal blocks)
- Layout handling: casts to ElementType before pointer arithmetic

**Possible Issues**:
1. Mask extraction when no masking needed (but mask_ptr would be nullptr)
2. Pointer arithmetic in ElementType units (looks correct)
3. Something in `compute_typed()` doesn't work correctly when called from batch context

## Next Steps

1. **Add Debug Logging** to `CpuAttentionKernelT::compute_batch`:
   - Log input Q/K/V shapes and first few values
   - Log mask extraction (if applicable)
   - Log output first few values
   - Compare sequential vs batched execution logs

2. **Verify Mask Setup** in E2E test:
   - Check if `should_build_mask()` returns false (as expected for causal=false, no padding)
   - Verify workspace_mask is nullptr in batched path

3. **Compare compute() vs compute_batch()** implementations:
   - Are there subtle differences in how they handle Q/K/V?
   - Is there a layout assumption difference?

4. **Consider Numerical Precision**:
   - Max diff of 0.295 in first attention layer seems large for FP32
   - Could indicate accumulation order differences or SIMD issues

5. **Test with Simpler Case**:
   - Run with batch_size=1 in "batched" mode vs sequential mode
   - Should be identical - if not, proves compute_batch has fundamental issue

## Test Commands

```bash
# Rebuild
cmake --build build_v2_e2e --target v2_test_qwen2_e2e_correctness --parallel

# Run with detailed snapshot output
cd /workspaces/llaminar && export OMP_NUM_THREADS=28 ... && \
mpirun -np 2 --bind-to socket --map-by socket \
./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness \
--gtest_filter="Qwen2E2ECorrectness.ComprehensiveBatchParity" 2>&1 | \
grep -A 20 "FIRST DIVERGENCE"

# Run with ERROR level logging (clean output)
LLAMINAR_LOG_LEVEL=ERROR mpirun -np 2 ... (same as above)
```

## Lessons Learned

1. **Layer-by-Layer Diagnostics Essential**: Without snapshot comparison, we'd still think the issue was in final norm or logits
2. **Multiple Bug Layers**: Mask selection bugs + layout bugs, all needed fixing
3. **Path Awareness Critical**: Our tensor-parallel fixes didn't apply because E2E uses GQA direct path
4. **Test Infrastructure Pays Off**: Snapshot framework immediately pinpointed divergence source

## References

- Attention mask types: causal vs padding-only
- Batch layout: `[batch_size, seq_len, ...]` vs `[total_tokens, ...]` 
- V2 architecture: `.github/instructions/llaminar-architecture-v2.instructions.md`
- Snapshot framework: `docs/v2/SNAPSHOT_FRAMEWORK_DESIGN.md`
