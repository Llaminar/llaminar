# Batched Attention Implementation Fix Plan

## Problem Statement

The current batched attention implementation in Llaminar V2 treats concatenated sequences as a single long sequence rather than independent sequences within a batch. This causes **cross-sequence attention contamination** where tokens from one sequence incorrectly attend to tokens from another sequence.

### Evidence

From E2E test `Qwen2MPIRankParity.ComprehensiveBatchParity`:
- **First divergence point**: `layer0_ATTENTION_CONTEXT` (attention output)
- **Preceding stages match exactly**: RMSNorm, QKV projections, RoPE all show `max_diff=0`
- **Conclusion**: The attention computation itself is the source of divergence

### Current Behavior

```
Sequential Execution (2 runs):
  Sequence 0: [4 tokens] → Attention computes over 4×4 matrix
  Sequence 1: [2 tokens] → Attention computes over 2×2 matrix

Batched Execution (1 run):
  Both sequences: [6 tokens total] → Attention computes over 6×6 matrix
  ❌ Tokens from Seq0 attend to tokens from Seq1 and vice versa
```

### Correct Behavior

```
Batched Execution (with proper masking):
  Both sequences: [6 tokens total] → Attention computes over 6×6 matrix
  ✓ Causal mask for each sequence independently
  ✓ Cross-sequence positions masked to -inf
```

## Solution Architecture

### Approach: Batch-Aware Attention Mask

Implement a **combined attention mask** that encodes:
1. **Causal masking** (within each sequence)
2. **Batch separation masking** (between sequences)

For a batch with sequences of lengths [4, 2] (padded to 4):

```
Token positions in flattened batch:
  Seq0: [0, 1, 2, 3]    (4 real tokens)
  Seq1: [4, 5, P, P]    (2 real tokens + 2 padding)

Combined Attention Mask (6×6 for non-padded view, 8×8 with padding):

Query/Key:    0   1   2   3 | 4   5   P   P
         0 [  0  -∞  -∞  -∞ | -∞  -∞  -∞  -∞ ]  Seq0
         1 [  0   0  -∞  -∞ | -∞  -∞  -∞  -∞ ]  Seq0
         2 [  0   0   0  -∞ | -∞  -∞  -∞  -∞ ]  Seq0
         3 [  0   0   0   0 | -∞  -∞  -∞  -∞ ]  Seq0
         ---|---------------|------------------|
         4 [ -∞  -∞  -∞  -∞ |  0  -∞  -∞  -∞ ]  Seq1
         5 [ -∞  -∞  -∞  -∞ |  0   0  -∞  -∞ ]  Seq1
         P [ -∞  -∞  -∞  -∞ | -∞  -∞  -∞  -∞ ]  Pad
         P [ -∞  -∞  -∞  -∞ | -∞  -∞  -∞  -∞ ]  Pad

Legend:
  0  = attend (add 0 to logits)
  -∞ = mask (add -inf to logits, becomes 0 after softmax)
```

## Implementation Plan

### Phase 1: Attention Mask Infrastructure

**Files to modify:**
- `src/v2/pipelines/qwen/Qwen2Graph.h` - Add mask buffer to activation buffers
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Generate attention mask during graph building
- `src/v2/execution/ComputeStage.h` - Add mask parameter to attention stage params
- `src/v2/execution/ComputeStage.cpp` - Update attention stage to use mask

**New structures:**

```cpp
// In Qwen2ActivationBuffers (Qwen2Graph.h)
struct Qwen2ActivationBuffers {
    // ... existing fields ...
    
    /// Attention mask for batch-aware attention
    /// Shape: [total_tokens, max_kv_len] where positions to mask are -inf
    std::unique_ptr<TensorBase> attention_mask;
};
```

```cpp
// In AttentionWithKVCacheStage::Params (ComputeStage.h)
struct Params {
    // ... existing fields ...
    
    /// Optional attention mask tensor
    /// If provided, added to attention scores before softmax
    /// Shape: [batch_size * seq_len, kv_len] or broadcastable
    TensorBase* attention_mask = nullptr;
    
    /// Sequence lengths for each batch element (for mask generation)
    const int* seq_lengths = nullptr;
    int batch_size = 1;
};
```

### Phase 2: Mask Generation

**Add mask generation function:**

```cpp
// src/v2/pipelines/qwen/BatchedAttentionMask.h

namespace llaminar2 {

/// Generate a batched causal attention mask
/// @param seq_lengths Array of sequence lengths for each batch element
/// @param batch_size Number of sequences in batch
/// @param padded_len Padded sequence length (max of seq_lengths)
/// @param output Output tensor of shape [total_tokens, total_tokens]
/// @return true on success
bool generateBatchedCausalMask(
    const int* seq_lengths,
    int batch_size,
    int padded_len,
    TensorBase* output);

/// Generate attention mask with existing KV cache positions
/// For decode phase where Q is [batch, 1] and K is [batch, cached + 1]
bool generateDecodeMask(
    const int* cached_lengths,  // Per-sequence cached token counts
    int batch_size,
    TensorBase* output);

} // namespace llaminar2
```

### Phase 3: Attention Kernel Update

**Modify attention computation to apply mask:**

Current flow:
```
scores = Q @ K^T / sqrt(d_k)
scores = causal_mask(scores)  // Simple lower-triangular
probs = softmax(scores)
output = probs @ V
```

New flow:
```
scores = Q @ K^T / sqrt(d_k)
if (attention_mask) {
    scores = scores + attention_mask  // Mask has -inf for blocked positions
}
probs = softmax(scores)
output = probs @ V
```

**Files to modify:**
- `src/v2/kernels/cpu/attention/GQAAttention.cpp` - Add mask application
- `src/v2/kernels/cpu/attention/FlashAttention.cpp` - Add mask application (if used)

### Phase 4: Graph Building Integration

**Update `buildAttentionGraph()` to:**
1. Accept `seq_lengths` array for variable-length batches
2. Generate appropriate attention mask
3. Pass mask to attention stage

**Update `forward_batch()` to:**
1. Track per-sequence lengths
2. Build position IDs correctly per sequence
3. Generate mask once before graph execution

### Phase 5: Testing

1. **Unit test**: Mask generation correctness
2. **Integration test**: Single-sequence with mask (should match no-mask)
3. **E2E test**: Fix `Qwen2MPIRankParity.ComprehensiveBatchParity`
4. **E2E test**: Fix `Qwen2MPIRankParity.MultiSequenceBatchEqualLength`

## Implementation Steps

### Step 1: Add Mask Generation Utility (New File)

Create `src/v2/pipelines/qwen/BatchedAttentionMask.h` and `.cpp`:
- `generateBatchedCausalMask()` for prefill
- `generateDecodeMask()` for autoregressive decode

### Step 2: Update Attention Stage Params

Add mask-related fields to `AttentionWithKVCacheStage::Params` and `AttentionComputeStage::Params`.

### Step 3: Update Attention Kernel

Modify `GQAAttention::compute()` to apply mask before softmax.

### Step 4: Update Graph Building

Modify `Qwen2Graph::buildAttentionGraph()` to generate and pass mask.

### Step 5: Update Forward Path

Modify `InferenceRunner::forward_batch()` to track sequence lengths and pass to graph.

### Step 6: Test and Validate

Run E2E parity tests to verify fix.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Performance regression from mask application | Fuse mask into attention kernel; use sparse representation |
| Memory overhead from mask tensor | Reuse mask buffer across layers; generate on-the-fly if small batch |
| Complexity in decode phase | Start with prefill-only; decode can use simpler per-position masking |
| Breaking single-sequence path | Make mask optional; null mask = current behavior |

## Timeline Estimate

- Phase 1: 1-2 hours (infrastructure)
- Phase 2: 1 hour (mask generation)
- Phase 3: 2-3 hours (kernel changes)
- Phase 4: 1-2 hours (graph integration)
- Phase 5: 1 hour (testing)

**Total: ~6-9 hours**

## Success Criteria

1. `Qwen2MPIRankParity.MultiSequenceBatchEqualLength` passes with tolerance 1e-3
2. `Qwen2MPIRankParity.ComprehensiveBatchParity` passes with tolerance 1e-3
3. Single-sequence tests continue to pass with `max_diff=0`
4. No performance regression for batch_size=1 (null mask fast path)
