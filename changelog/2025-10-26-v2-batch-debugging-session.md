# V2 Batch Implementation - Debugging Session

**Date**: October 26, 2025  
**Status**: 🔴 **In Progress** - Batch parity test failing, investigating root cause

---

## Objective

Implement and validate multi-sequence batch inference in V2 Qwen2Pipeline.

---

## Progress

### ✅ Completed

1. **Batch-first refactoring** (Phase A)
   - Single unified pipeline handles all batch sizes
   - `batch_size=1` validated (SingleTokenInference test passing)
   - BatchedKVCache integrated
   - All buffers sized for batch dimension

2. **Multi-sequence batch test** (Phase B)
   - Added `MultiSequenceBatchEqualLength` test
   - Added `padded_seq_len()` getter for test access
   - Proper logits extraction logic (accounting for padding)

###  Currently Debugging

**Test**: `Qwen2E2ECorrectness.MultiSequenceBatchEqualLength`
- Batch size: 2
- Sequences: Both 2 tokens (equal length, no padding)
  - Sequence 0: `[151644, 9906]` (BOS + "Hello")
  - Sequence 1: `[151644, 1374]` (BOS + different token)

**Results**:
```
Sequence 0: ✅ PERFECT match (max_diff=0, rel_l2=0)
Sequence 1: ❌ MASSIVE divergence (max_diff=23.7, rel_l2=1.40)
```

---

## Root Cause Analysis

### Observations

1. **Sequence 0 passes perfectly**
   - Batch execution matches sequential execution exactly
   - All logits identical (0 difference)

2. **Sequence 1 fails completely**
   - Not a small numerical error - massive divergence
   - 303,810 mismatches out of 303,872 elements (vocab_size * seq_len)
   - Suggests wrong data entirely, not just precision issues

3. **Equal-length test eliminates padding as cause**
   - Both sequences have 2 tokens
   - No padding positions
   - `padded_seq_len = 2` (no padding needed)

### Hypothesis

The batch is being processed as **one concatenated sequence** instead of **two separate sequences**.

**Evidence**:
- Sequence 0 matches because it occupies positions [0-1] in both batch and sequential modes
- Sequence 1 diverges because:
  - **Batch mode**: Processes positions [2-3] as "continuation of position [0-1]"
  - **Sequential mode**: Processes positions [0-1] as fresh sequence
- Position encoding, attention context, or other positional information treats batch as one 4-token sequence

### Areas Investigated

#### 1. Logits Extraction ✅ (Ruled Out)
- `getLogits(seq_idx)` correctly calculates offset: `seq_idx * padded_seq_len * vocab_size`
- For seq1: offset = `1 * 2 * 151936 = 303872` ✓
- Extraction logic copies correct rows from buffer ✓

#### 2. Position IDs (Suspicious)
Current implementation:
```cpp
for (int b = 0; b < batch_size_; ++b)
{
    for (int i = 0; i < padded_seq_len_; ++i)
    {
        position_ids[b * padded_seq_len_ + i] = current_position_ + i;
    }
}
```

For `batch_size=2`, `padded_seq_len=2`, `current_position_=0`:
```
position_ids = [0, 1, 0, 1]
```

This looks **correct** for prefill (both sequences start at position 0).

**But**: `current_position_` is a **single shared counter**!
- After this batch: `current_position_ += padded_seq_len_` (increments to 2)
- Next decode step: All sequences would get position IDs starting at 2
- **Each sequence needs its own position counter for autoregressive decode!**

#### 3. Embeddings ✅ (Looks Correct)
```cpp
// embedding_batch processes each sequence separately
for (int b = 0; b < batch_size_; ++b)
{
    for (int i = 0; i < seq_len; ++i)
    {
        // Lookup embedding for batch[b][i]
        std::memcpy(output + global_idx * d_model_,
                    embed_table + tokens[i] * d_model_, ...);
        global_idx++;
    }
    // Pad with zeros
}
```

This correctly looks up embeddings for each sequence's tokens.

#### 4. Attention (Needs Investigation)
Currently calling `attention_gqa_mpi()` which is **single-sequence** attention:
- Doesn't know about batch boundaries
- Treats all `effective_seq_len=4` positions as one sequence
- **This is likely the bug!**

**Causal masking issue**:
- Sequence 0, token 1 can attend to: [seq0_tok0, seq0_tok1] ✓
- Sequence 1, token 0 can attend to: [seq0_tok0, seq0_tok1, seq0_tok2, seq1_tok0] ✗
  - Should only attend to [seq1_tok0]!
  - But causal mask allows attending to all previous positions in the batch

**This explains the divergence**:
- Sequence 1's first token (position 2 in batch) can see sequence 0's tokens
- This contaminates sequence 1's hidden states
- Results propagate through all layers
- Final logits are completely wrong

---

## Identified Bugs

### 🔴 Critical: Attention Not Batch-Aware

**Location**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp:424`

**Problem**:
```cpp
VALIDATE_OP(attention_gqa_mpi(
    buffers.Q.get(), buffers.K.get(),
    buffers.V.get(), buffers.attn_output.get(),
    n_heads_, n_kv_heads_, head_dim_,
    /*causal=*/true, /*window_size=*/-1),
    "GQA attention");
```

Uses `attention_gqa_mpi()` which applies **causal masking** across the entire `effective_seq_len=4` dimension.

**Expected**: Each sequence should only attend within its own tokens:
```
Batch layout: [seq0_tok0, seq0_tok1, seq1_tok0, seq1_tok1]

Attention mask should be:
     s0t0  s0t1  s1t0  s1t1
s0t0  ✓     ✗     ✗     ✗
s0t1  ✓     ✓     ✗     ✗    
s1t0  ✗     ✗     ✓     ✗    ← Can only see itself!
s1t1  ✗     ✗     ✓     ✓
```

**Actual** (with current causal mask):
```
     s0t0  s0t1  s1t0  s1t1
s0t0  ✓     ✗     ✗     ✗
s0t1  ✓     ✓     ✗     ✗    
s1t0  ✓     ✓     ✓     ✗    ← BUG! Sees sequence 0!
s1t1  ✓     ✓     ✓     ✓    ← BUG! Sees sequence 0!
```

**Fix Required**: Implement `attention_gqa_batch()` with **block-diagonal causal masking**:
- Causal masking within each sequence
- Zero masking across sequence boundaries

### 🟡 Medium: Position Counter Not Per-Sequence

**Location**: `src/v2/pipelines/qwen/Qwen2Pipeline.h:151`

**Problem**:
```cpp
int current_position_ = 0;  // Shared across all sequences in batch!
```

**Impact**:
- Prefill: Not an issue (all sequences start at position 0)
- Autoregressive decode: **Critical bug**
  - After first decode step, `current_position_ = 1`
  - All sequences in batch get position ID 1
  - But some sequences might be at different lengths!

**Fix Required**: 
```cpp
std::vector<int> current_positions_;  // One per sequence in batch
```

---

## Next Steps

### Immediate (Phase B continued)

1. **Implement batched attention masking**
   - Create `attention_gqa_batch()` function
   - Block-diagonal causal mask (per-sequence causality)
   - Combined with padding mask for variable-length sequences
   - Location: `src/v2/kernels/cpu/AttentionGQA.{h,cpp}` (or similar)

2. **Fix position counter**
   - Change `current_position_` to `std::vector<int> current_positions_`
   - Update RoPE position ID generation to use per-sequence positions
   - Update `forward_batch()` to increment each sequence's position independently

3. **Re-run tests**
   - `MultiSequenceBatchEqualLength` should pass after attention fix
   - `MultiSequenceBatch` (variable-length) should pass after padding mask integration

### Future (Phase C+)

4. **Batch KV cache validation**
   - Test autoregressive decode with batches
   - Verify per-sequence KV cache management

5. **Performance optimization**
   - Batch GEMM kernels
   - Memory pooling for batch buffers

---

## Test Status

| Test | Status | Notes |
|------|--------|-------|
| `SingleTokenInference` | ✅ PASSING | Validates `batch_size=1` works |
| `MultiSequenceBatchEqualLength` | ❌ FAILING | Sequence 1 diverges (attention bug) |
| `MultiSequenceBatch` (variable-length) | 🚫 DISABLED | Awaiting attention+padding mask fix |

---

## Code Changes Made This Session

1. **Qwen2Pipeline.h**
   - Added `int padded_seq_len()` getter

2. **Test__Qwen2E2ECorrectness.cpp**
   - Added `MultiSequenceBatchEqualLength` test (equal-length sequences)
   - Disabled original `MultiSequenceBatch` test (variable-length)
   - Updated logits extraction logic to handle padding layout
   - Added diagnostic logging for batch parameters

---

## References

- **Batch-first refactoring**: `changelog/2025-10-26-v2-batch-first-refactor-complete.md`
- **V2 architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Batching plan**: `V2_BATCHING_IMPLEMENTATION_PLAN.md`
- **V1 batch attention reference**: `src/operators/MPIAttentionBatchOperator.cpp`

---

**Status**: Blocked on batched attention implementation. Need to create `attention_gqa_batch()` with block-diagonal causal masking before tests can pass.
