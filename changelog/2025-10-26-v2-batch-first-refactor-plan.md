# V2 Batch-First Refactor Plan

**Date**: October 26, 2025  
**Status**: In Progress

## Strategic Decision

After completing single-token inference, we recognize that **single-sequence is just `batch_size=1`**. Instead of maintaining separate single and batch pipelines (like V1), we're refactoring `Qwen2Pipeline` to be batch-first from the ground up.

## Key Changes

### 1. Constructor
- Add `batch_size` parameter (default=1)
- Initialize `BatchedKVCache` instead of `KVCache`
- Size all buffers for `batch_size * max_seq_len`

### 2. Forward Pass
- Primary interface: `forward_batch(vector<vector<int>> token_batches)`
- Legacy interface: `forward(int* tokens, int seq_len)` wraps batch version
- Padding logic: Use `BatchPaddingUtils` for variable-length sequences

### 3. Dimensions
- All tensors: `[batch_size * padded_seq_len, feature_dim]`
- `effective_seq_len = batch_size * padded_seq_len` (flattened batch dimension)
- Attention: Use `attention_gqa_batch()` with combined masking

### 4. State Management
- `batch_size_`: Number of sequences
- `padded_seq_len_`: Max sequence length in current batch
- `sequence_lengths_`: Actual length per sequence
- `logits_buffer_`: Sized for full batch output

### 5. Test Updates
- Update `Test__Qwen2E2ECorrectness` to use batch format:
  - Single token: `forward_batch({{151644}})`
  - Multi token: `forward_batch({{tok1, tok2, ...}})`
- Add batch tests: `forward_batch({{seq1}, {seq2}})`

## Implementation Phases

**Phase A**: Core refactoring (this session)
- Update constructor, forward signatures
- Add batch state management
- Update dimension calculations

**Phase B**: Batch logic (next session)
- Implement `forward_batch()` with padding
- Integrate `BatchedKVCache`
- Use `attention_gqa_batch()`

**Phase C**: Testing (next session)
- Update E2E tests
- Add batch correctness tests
- Validate parity (batch vs sequential loop)

## Benefits

✅ Single codebase (no duplication)  
✅ Natural API (`batch_size=1` for single-sequence)  
✅ Future-proof (already handles batching)  
✅ Avoids V1's maintenance burden

## Current Status

- [x] Header updated (batch_size parameter, BatchedKVCache, forward_batch signature)
- [x] Constructor updated (accepts batch_size)
- [ ] forward_batch() implementation
- [ ] embedding_batch() implementation
- [ ] attention_block updated for batch
- [ ] ffn_block updated for batch
- [ ] Test migration

---

**Next Steps**: Implement `forward_batch()` and `embedding_batch()` logic.
