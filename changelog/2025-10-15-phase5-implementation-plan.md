# Phase 5: Pipeline Batch Tensor Integration - Implementation Plan

**Date**: 2025-10-15  
**Status**: Ready to implement  
**Priority**: CRITICAL (unlocks 22× speedup)  
**Estimated Effort**: 1-2 days

---

## Executive Summary

**Problem**: QwenPipeline processes batches sequentially in a for-loop, achieving only **1.01× speedup** instead of the target **22× speedup**.

**Root Cause**: Lines 2090-2150 (prefillBatch) and 2250-2300 (decodeBatch) iterate through sequences one-at-a-time:
```cpp
for (int i = 0; i < batch_size; ++i) {
    prefill(token_batches[i], ...);  // Process ONE sequence
}
```

**Solution**: Replace sequential loops with batch tensor processing that calls operators once with [batch, seq, hidden] tensors.

**Expected Outcome**: 
- Prefill: 9.06 tok/s → 200+ tok/s (22× speedup)
- Decode: 4.58 tok/s → 100+ tok/s (22× speedup)
- Sequential ratio: 1.06 → 0.045

---

## Background: What We've Proven

### Phase 4.2: Correctness ✅
- **Test**: BatchCorrectnessTest (2/2 passing, 94.5s)
- **Result**: Batch output matches sequential output exactly
- **Validation**: Element-wise comparison, zero divergence
- **Conclusion**: Implementation is functionally correct

### Phase 4.3: Performance Bottleneck ✅
- **Test**: BatchPerformanceTest (131s prefill, 169s+ decode)
- **Finding**: Only 1.01× speedup at batch=32
- **Evidence**:
  - Sequential ratio = 1.06 (processing sequentially)
  - Time scales linearly: batch=32 takes 31.75× batch=1 time
  - Throughput flat: 9.06 → 9.13 tok/s (no improvement)

**Performance Gap**: 20.99× missing speedup

| Batch | Current (tok/s) | Target (tok/s) | Gap |
|-------|-----------------|----------------|-----|
| 1 | 9.06 | 9.06 | 1.00× |
| 32 | 9.13 | 199.3 | 21.83× |

---

## Current Implementation Analysis

### Sequential Loop Pattern (Lines 2090-2150)

```cpp
// prefillBatch() - SEQUENTIAL EXECUTION
int orig_n_past = n_past_;
auto orig_k_cache = k_cache_;
auto orig_v_cache = v_cache_;

for (int i = 0; i < batch_size; ++i) {
    // Restore sequence i state
    n_past_ = n_past_batch_[i];
    k_cache_ = k_cache_batch_[i];
    v_cache_ = v_cache_batch_[i];
    
    // Process ONE sequence
    prefill(token_batches[i], weights_iface, seq_ctx);
    
    // Save sequence i updated state
    n_past_batch_[i] = n_past_;
    k_cache_batch_[i] = k_cache_;
    v_cache_batch_[i] = v_cache_;
    
    // Extract logits for sequence i
    logits(batch_logits_data[i]);
}

// Restore original state
n_past_ = orig_n_past;
k_cache_ = orig_k_cache;
v_cache_ = orig_v_cache;
```

### State Management Pattern
- **Per-sequence vectors**: 
  - `n_past_batch_[]` - position counter per sequence
  - `k_cache_batch_[]` - KV cache tensors per sequence
  - `v_cache_batch_[]` - V cache tensors per sequence
- **Single-sequence state**: `n_past_`, `k_cache_`, `v_cache_`
- **Pattern**: Save original → Loop (restore i → process → save i) → Restore original

### Why This Is Sequential
1. Calls `prefill(token_batches[i], ...)` ONE sequence at a time
2. Each call processes through all 28 transformer layers serially
3. Time complexity: O(batch_size × layers × seq_len)
4. **Expected**: O(layers × total_tokens) with batch parallel processing

---

## Implementation Challenges

### 1. Per-Sequence State Management
**Challenge**: Each sequence has independent `n_past_` value (position in generation)
- Sequence 0 might be at position 15
- Sequence 1 might be at position 42
- Operators need to know which KV cache slice to use

**Current Solution**: Works correctly via BatchedKVCache
- Each sequence gets isolated cache access
- Operators already handle this via sequence_idx parameter

### 2. Variable Sequence Lengths
**Challenge**: Sequences in batch have different lengths
- Sequence 0: 8 tokens
- Sequence 1: 32 tokens  
- Sequence 2: 16 tokens

**Current Solution**: BatchPaddingUtils handles this
- `stack_batch()` pads to max_seq_len
- Operators process padded tensors
- Output extraction handles variable lengths

### 3. KV Cache Indexing
**Challenge**: Attention must write to correct KV cache position per sequence

**Current Solution**: BatchedKVCache::update()
- Takes sequence_idx and position parameters
- Writes to correct cache slice
- Already validated in Phase 2.4 tests

### 4. Single-Sequence Method Integration
**Challenge**: Current `prefill()` and `decode()` assume single sequence state

**Solution Approach**:
- Create parallel batch-aware methods (`prefill_batch_internal()`)
- Bypass single-sequence state management
- Call operators directly with batch tensors

---

## Implementation Plan

### Step 1: Create prefill_batch_internal() Helper (2-3 hours)

**Purpose**: Process entire batch in one pass through transformer layers

**Signature**:
```cpp
std::shared_ptr<TensorBase> QwenPipeline::prefill_batch_internal(
    const std::shared_ptr<TensorBase>& batch_tokens,  // [batch_size, max_seq_len]
    const WeightsData& weights,
    StageContext& ctx
);
```

**Implementation Pattern**:
```cpp
// 1. Embedding lookup - [batch, seq] → [batch, seq, hidden]
auto embedded = embedding_op_->execute({batch_tokens, ...});

// 2. For each transformer layer
for (int layer = 0; layer < num_layers; ++layer) {
    // Layer norm
    auto normed = rms_norm_op_->execute({embedded, ...});
    
    // Attention (handles batched KV cache)
    auto attn_out = attention_op_->execute({normed, ...});
    
    // Feed-forward
    auto ff_out = feed_forward_batch({attn_out, ...});
    
    // Residual
    embedded = add(embedded, ff_out);
}

// 3. Final projection - [batch, seq, hidden] → [batch, seq, vocab]
auto logits = lm_head_op_->execute({embedded, ...});

return logits;
```

**Key Differences from single-sequence `prefill()`**:
- No state save/restore (n_past_, k_cache_)
- Direct operator calls with batch tensors
- No per-sequence iteration

### Step 2: Modify prefillBatch() to Use Helper (1-2 hours)

**Current Code** (lines 2090-2150):
```cpp
for (int i = 0; i < batch_size; ++i) {
    prefill(token_batches[i], ...);
}
```

**New Code**:
```cpp
// 1. Stack sequences into batch tensor [batch, max_seq_len]
auto batch_tokens = BatchPaddingUtils::stack_batch(token_batches);

// 2. Process entire batch in ONE call
auto batch_logits = prefill_batch_internal(batch_tokens, weights, ctx);

// 3. Extract per-sequence outputs and states
for (int i = 0; i < batch_size; ++i) {
    // Extract logits for sequence i
    auto seq_logits = batch_logits->slice_batch(i);
    std::memcpy(batch_logits_data[i].data(), seq_logits->data(), vocab_size * sizeof(float));
    
    // Update per-sequence state (n_past incremented by seq_len)
    n_past_batch_[i] += token_batches[i].size();
    
    // KV cache already updated via BatchedKVCache during processing
}
```

**Changes**:
- ~60 lines removed (for-loop with state management)
- ~30 lines added (stack → process → extract pattern)
- Net: ~30 lines shorter

### Step 3: Verify Correctness (30 minutes)

**Test Command**:
```bash
ctest --test-dir build -R "BatchCorrectnessTest" -V
```

**Expected**:
- ✅ 2/2 tests passing
- ✅ Batch still matches sequential exactly
- ✅ No regression in correctness

**If Tests Fail**:
- Check KV cache state extraction
- Verify n_past_ updates per sequence
- Debug logits extraction from batch tensor

### Step 4: Create decode_batch_internal() Helper (1-2 hours)

**Signature**:
```cpp
std::shared_ptr<TensorBase> QwenPipeline::decode_batch_internal(
    const std::shared_ptr<TensorBase>& next_tokens,  // [batch_size, 1]
    const WeightsData& weights,
    StageContext& ctx
);
```

**Implementation**: Same pattern as prefill_batch_internal(), but:
- Input shape: [batch, 1] instead of [batch, seq_len]
- KV cache appends one token per sequence
- Faster execution (single token vs many)

### Step 5: Modify decodeBatch() (1 hour)

**Current Code** (lines 2250-2300):
```cpp
for (int i = 0; i < batch_size; ++i) {
    decode(next_tokens[i], ...);
}
```

**New Code**: Identical pattern to prefillBatch() modification
- Stack next_tokens into [batch, 1] tensor
- Call decode_batch_internal() once
- Extract per-sequence logits and states

### Step 6: Performance Validation (1 hour)

**Test Command**:
```bash
ctest --test-dir build -R "BatchPerformanceTest" -V
```

**Expected Results**:

**Prefill**:
| Batch | Old (tok/s) | New (tok/s) | Speedup |
|-------|-------------|-------------|---------|
| 1 | 9.06 | 9.06 | 1.00× |
| 2 | 9.14 | 18.1 | 1.98× |
| 4 | 9.14 | 36.2 | 3.96× |
| 8 | 9.15 | 72.5 | 7.94× |
| 16 | 9.08 | 145.0 | 15.88× |
| 32 | 9.13 | 199.3 | **21.83×** ✅ |

**Decode**:
| Batch | Old (tok/s) | New (tok/s) | Speedup |
|-------|-------------|-------------|---------|
| 1 | 4.58 | 4.58 | 1.00× |
| 16 | 5.19 | 73.3 | 16.00× |

**Sequential Ratio**:
- Old: 1.06 (sequential execution)
- New: **0.045** (true parallel batching)

**Timing**:
- Batch=1: ~880ms (unchanged)
- Batch=32: 28,041ms → **~880ms** (constant time!)

---

## Risk Analysis

### Low Risk
- ✅ Operators already support batch dimensions (validated Phase 2)
- ✅ BatchedKVCache handles per-sequence isolation (validated Phase 2.4)
- ✅ BatchPaddingUtils handles variable lengths (validated Phase 3)
- ✅ Correctness proven via Phase 4.2 tests

### Medium Risk
- ⚠️ State extraction from batch processing (n_past updates)
  - **Mitigation**: Test carefully, compare with sequential behavior
- ⚠️ KV cache state not explicitly saved (handled internally)
  - **Mitigation**: Operators already manage this correctly

### High Risk
- ❌ None identified (all foundation components validated)

---

## Success Criteria

### Functional Requirements
- ✅ Correctness tests pass (2/2 BatchCorrectnessTest)
- ✅ No regression in accuracy
- ✅ Per-sequence state isolation maintained

### Performance Requirements
- ✅ Prefill speedup ≥ 20× at batch=32
- ✅ Decode speedup ≥ 15× at batch=16
- ✅ Sequential ratio < 0.1
- ✅ Throughput scaling near-linear with batch size

### Code Quality
- ✅ Reduced code complexity (~30 lines shorter)
- ✅ Clearer separation of concerns
- ✅ No duplicate state management logic

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| Create prefill_batch_internal() | 2-3 hours | None |
| Modify prefillBatch() | 1-2 hours | Step 1 |
| Verify correctness | 30 min | Step 2 |
| Create decode_batch_internal() | 1-2 hours | None |
| Modify decodeBatch() | 1 hour | Step 4 |
| Performance validation | 1 hour | Steps 2,5 |
| Documentation | 1-2 hours | All |
| **TOTAL** | **8-12 hours** | **1-2 days** |

---

## Next Steps

1. **Implement Step 1**: Create prefill_batch_internal() helper
   - Location: `/workspaces/llaminar/src/QwenPipeline.cpp`
   - Insert after line 2089 (before prefillBatch())
   - ~80-100 lines of code

2. **Implement Step 2**: Modify prefillBatch()
   - Replace lines 2090-2150 with stack → process → extract pattern
   - Net reduction: ~30 lines

3. **Run correctness test**:
   ```bash
   ctest --test-dir build -R "BatchCorrectnessTest" -V
   ```

4. **Repeat for decode**: Steps 4-5

5. **Measure performance**:
   ```bash
   ctest --test-dir build -R "BatchPerformanceTest" -V
   ```

6. **Document results**: Update progress reports and TODO

---

## Code Files to Modify

### Primary File
- `/workspaces/llaminar/src/QwenPipeline.cpp`
  - Add: `prefill_batch_internal()` (~80 lines)
  - Add: `decode_batch_internal()` (~60 lines)
  - Modify: `prefillBatch()` (replace loop, ~30 lines net change)
  - Modify: `decodeBatch()` (replace loop, ~20 lines net change)
  - **Total changes**: ~190 lines added/modified

### Header File (if needed)
- `/workspaces/llaminar/src/QwenPipeline.h`
  - May need to declare private helpers
  - ~4 lines added

---

## Expected Outcome

**Before** (Current):
```
Prefill batch=32: 28,041ms, 9.13 tok/s, speedup 1.01×
Decode batch=16: ~17,500ms, 5.19 tok/s, speedup 1.13×
Sequential ratio: 1.06 (SEQUENTIAL EXECUTION)
```

**After** (Target):
```
Prefill batch=32: 880ms, 199.3 tok/s, speedup 21.83×
Decode batch=16: 1,100ms, 73.3 tok/s, speedup 16.00×
Sequential ratio: 0.045 (TRUE PARALLEL BATCHING)
```

**Impact**: **22× speedup unlocked** 🚀

---

## References

- **Phase 4.2-4.3 Results**: `changelog/2025-10-15-phase4-2-4-3-results.md`
- **Day 5 Summary**: `changelog/2025-10-15-day5-summary.md`
- **Progress Report**: `changelog/2025-10-15-progress-report-phase4.md`
- **Source Code**: `/workspaces/llaminar/src/QwenPipeline.cpp` lines 2090-2300
- **Tests**: 
  - `tests/test_batch_correctness.cpp`
  - `tests/test_batch_performance.cpp`

---

**Status**: Ready to implement  
**Author**: GitHub Copilot  
**Review Date**: 2025-10-15
