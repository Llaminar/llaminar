# Phase 5 Status: Ready to Implement

**Date**: 2025-10-15  
**Phase**: 5 - Pipeline Batch Tensor Integration  
**Status**: Implementation plan complete, ready to start coding

---

## Summary

We've completed Phases 4.2-4.3 and **definitively identified the bottleneck**:

- ✅ **Phase 4.2**: Correctness validated (2/2 tests passing) - batch matches sequential exactly
- ✅ **Phase 4.3**: Bottleneck located - only 1.01× speedup due to sequential loops in QwenPipeline

**Root Cause**: Lines 2090-2150 (prefillBatch) and 2250-2300 (decodeBatch) process sequences one-at-a-time in for-loops.

**Gap**: 20.99× missing speedup (target: 22×, actual: 1.01×)

---

## What We Found in the Code

### Sequential Pattern (Both prefill and decode)

```cpp
// Current implementation - SEQUENTIAL
for (int i = 0; i < batch_size; ++i) {
    // Restore sequence i state
    n_past_ = n_past_batch_[i];
    k_cache_ = k_cache_batch_[i];
    v_cache_ = v_cache_batch_[i];
    
    // Process ONE sequence through ALL 28 layers
    prefill(token_batches[i], weights, ctx);
    
    // Save sequence i state
    n_past_batch_[i] = n_past_;
    k_cache_batch_[i] = k_cache_;
}
```

### Performance Evidence

**Prefill (batch=32)**:
- Time: 28,041ms (should be ~880ms)
- Throughput: 9.13 tok/s (should be ~200 tok/s)
- Sequential ratio: 1.06 (proves sequential execution)

**Decode (batch=16)**:
- Throughput: 5.19 tok/s (should be ~73 tok/s)
- Speedup: 1.13× (should be 16×)

---

## Implementation Plan

### Revised Complexity Estimate: 1-2 days

**Why Longer Than Initially Estimated**:
1. Per-sequence state management (n_past_batch_, k_cache_batch_, v_cache_batch_)
2. Each sequence at different generation positions
3. Need to create NEW batch-aware methods (prefill_batch_internal, decode_batch_internal)
4. Both prefill AND decode need same fix
5. Must preserve state isolation semantics

### Six-Step Plan

1. **Create prefill_batch_internal()** (2-3 hours)
   - Accept [batch, max_seq_len] stacked tensor
   - Process entire batch through transformer layers in ONE pass
   - Return [batch, seq, vocab] logits

2. **Modify prefillBatch()** (1-2 hours)
   - Replace for-loop with: stack → prefill_batch_internal() → extract
   - Update per-sequence states from batch processing

3. **Verify correctness** (30 minutes)
   - Run: `ctest --test-dir build -R "BatchCorrectnessTest"`
   - Must still pass 2/2 tests

4. **Create decode_batch_internal()** (1-2 hours)
   - Same pattern as prefill helper
   - Input: [batch, 1] next tokens

5. **Modify decodeBatch()** (1 hour)
   - Same pattern as prefillBatch() modification

6. **Measure performance** (1 hour)
   - Run: `ctest --test-dir build -R "BatchPerformanceTest"`
   - **Expected**: 22× speedup achieved!

---

## Expected Results

### Before (Current)
```
Prefill batch=32:  9.13 tok/s, speedup 1.01×
Decode batch=16:   5.19 tok/s, speedup 1.13×
Sequential ratio:  1.06 (SEQUENTIAL)
```

### After (Target)
```
Prefill batch=32:  199.3 tok/s, speedup 21.83× ✅
Decode batch=16:   73.3 tok/s, speedup 16.00× ✅
Sequential ratio:  0.045 (PARALLEL)
```

---

## Files to Modify

### Primary
- `/workspaces/llaminar/src/QwenPipeline.cpp`
  - Add prefill_batch_internal() (~80 lines)
  - Add decode_batch_internal() (~60 lines)
  - Modify prefillBatch() (~30 lines net change)
  - Modify decodeBatch() (~20 lines net change)
  - **Total**: ~190 lines modified/added

### Secondary (if needed)
- `/workspaces/llaminar/src/QwenPipeline.h`
  - Declare private helper methods (~4 lines)

---

## Risk Assessment

### Low Risk ✅
- Operators already support batch dimensions (validated Phase 2)
- BatchedKVCache handles per-sequence isolation (validated Phase 2.4)
- Correctness proven (Phase 4.2: 2/2 tests passing)

### Medium Risk ⚠️
- State extraction after batch processing (n_past updates)
- KV cache state management (should work via existing BatchedKVCache)

### High Risk ❌
- None identified

---

## Next Action

**Start with Step 1**: Implement `prefill_batch_internal()` helper function

**Location**: `/workspaces/llaminar/src/QwenPipeline.cpp` after line 2089

**Pattern**:
```cpp
std::shared_ptr<TensorBase> QwenPipeline::prefill_batch_internal(
    const std::shared_ptr<TensorBase>& batch_tokens,
    const WeightsData& weights,
    StageContext& ctx
) {
    // 1. Embedding: [batch, seq] → [batch, seq, hidden]
    auto embedded = embedding_op_->execute({batch_tokens, ...});
    
    // 2. Transformer layers (no state save/restore loop)
    for (int layer = 0; layer < num_layers; ++layer) {
        embedded = process_layer_batch(embedded, layer, weights);
    }
    
    // 3. Final projection: [batch, seq, hidden] → [batch, seq, vocab]
    return lm_head_op_->execute({embedded, ...});
}
```

---

## Documentation Created

1. **Implementation Plan**: `changelog/2025-10-15-phase5-implementation-plan.md` (detailed 400-line plan)
2. **This Status**: `changelog/2025-10-15-phase5-status.md` (concise summary)
3. **Phase 4 Results**: `changelog/2025-10-15-phase4-2-4-3-results.md` (technical findings)
4. **Day 5 Summary**: `changelog/2025-10-15-day5-summary.md` (complete day log)
5. **Progress Report**: `changelog/2025-10-15-progress-report-phase4.md` (comprehensive)

---

## Conclusion

We're ready to implement the critical fix that will unlock **22× speedup**. All foundation work is complete (46/49 tests passing), and we've definitively identified the bottleneck.

**Next**: Start coding `prefill_batch_internal()` to replace the sequential loop pattern.

---

**Status**: READY TO CODE  
**Confidence**: HIGH (all prerequisites validated)  
**Expected Timeline**: 1-2 days to complete Phase 5  
**Expected Outcome**: 22× speedup unlocked 🚀
