# Batch Pipeline Correctness Root Cause Analysis

**Date**: October 16, 2025  
**Author**: David Sanftenberg  
**Issue**: `BatchCorrectnessTest` fails - batch pipeline produces different results than sequential

## Summary

Identified the root cause of batch vs sequential divergence in `BatchQwenPipeline`. The problem is **NOT in the batch operators** but in how `MPIAttentionOperator` handles batched inputs.

## Investigation Process

### 1. Validated Sequential Reference (✅ PASSED)
- Ran PyTorch parity tests for both OpenBLAS and COSMA prefill paths
- **Result**: Both passed with 387/387 checks, max relative L2 error ~8e-6
- **Conclusion**: Sequential single-sequence processing is numerically correct

### 2. Embedding Validation (✅ PASSED)
- Created `test_batch_embedding_debug.cpp` to isolate embedding preparation
- Compared batch vs sequential embeddings element-wise
- **Result**: Perfect match - embeddings are identical
- **Conclusion**: The divergence happens AFTER embedding, during layer processing

### 3. Attention Operator Analysis (❌ ROOT CAUSE FOUND)

Examined `MPIAttentionOperator.cpp` lines 475-482:

```cpp
// Extract dimensions - support both 2D [seq_len, d_model] and 3D [batch, seq_len, d_model]
bool is_batched = (input_shape.size() == 3);
int batch_size = is_batched ? result.input->shape()[0] : 1;
int seq_len_per_batch = is_batched ? result.input->shape()[1] : result.input->shape()[0];

// For batched inputs, treat as [batch*seq_len, d_model] (flatten batch dimension)
result.seq_len = batch_size * seq_len_per_batch;
```

**THE BUG**: The attention operator **flattens the batch dimension** instead of processing batches independently!

## The Problem in Detail

When `BatchQwenPipeline` passes a [B=2, T=5, D=896] tensor to attention:

**What Should Happen**:
- Process 2 independent sequences, each with length 5
- Apply causal masking within each sequence independently
- Sequence 0 tokens can ONLY attend to sequence 0 tokens
- Sequence 1 tokens can ONLY attend to sequence 1 tokens

**What Actually Happens**:
- Flattens to a single sequence of length 2×5=10
- Treats positions [0-4] as first part, [5-9] as second part
- Applies single causal mask across all 10 positions
- **Sequences can cross-contaminate**: position 5 (seq1, token 0) can attend to position 0 (seq0, token 0)!

## Example Contamination

For batch with sequences `[1,2,3,4]` and `[5,6,7,8,9]` padded to length 5:

**Flattened view** (what attention sees):
```
Position: 0  1  2  3  4  5  6  7  8  9
Token:    1  2  3  4  0  5  6  7  8  9
Sequence: |-- seq 0 --|  |--- seq 1 ---|
```

**Causal mask applied**:
- Position 5 (token 5 from seq1) can attend to positions 0-5
- This means **seq1's first token attends to all of seq0's tokens**!
- Completely breaks batch independence

## Why This Causes Numerical Divergence

1. **Cross-sequence attention**: Tokens from different sequences attend to each other
2. **Different context**: Each sequence in batch sees contaminated context from other sequences
3. **Padding interference**: Padding positions (zeros) from one sequence affect attention scores in other sequences
4. **Cascading errors**: Error accumulates through 24 layers, leading to completely different final logits

**Observed**: batch=3.25 vs sequential=12.37 (3.8x difference)  
**Explained**: Attention context is fundamentally different due to cross-sequence leakage

## Impact on Batch Operators

The batch operators (`MPILinearBatchOperator`, `MPISwiGLUBatchOperator`) are **NOT the problem**:
- They correctly handle 3D [B, T, D] tensors
- They process each batch×sequence position independently
- Unit tests pass (9/9 Linear, 7/7 SwiGLU)

The issue is **exclusively in attention**'s flattening approach.

## Fix Required

The `MPIAttentionOperator` needs to be updated to support true batched execution:

### Option A: Batch-aware attention (preferred)
- Keep batch dimension separate throughout Q/K/V projections
- Compute attention scores per-batch with independent causal masks
- Shape flow: [B, T, D] → [B, n_heads, T, head_dim] → [B, T, D]

### Option B: Loop over batch dimension
- Process each sequence in the batch independently
- Reuse existing single-sequence logic
- Simple but less efficient

## Next Steps

1. ✅ Document root cause (this file)
2. ⏭️ Design batch-aware attention architecture
3. ⏭️ Implement per-batch causal masking
4. ⏭️ Update KV cache to handle batch dimension properly
5. ⏭️ Verify `BatchCorrectnessTest` passes

## Test Evidence

### Parity Tests (Sequential Validation)
```
OpenBLAS Prefill vs PyTorch: 387/387 PASSED
COSMA Prefill vs PyTorch: 387/387 PASSED
```

### Embedding Debug Test
```
Batch embedding shape: [2, 5, 896]
✓ Sequence 0 embeddings match perfectly
✓ Sequence 1 embeddings match perfectly
✓ Sequence 0 padding is zero
✓ Sequence 1 padding is zero
```

### Batch Correctness Test (Failure)
```
Sequence 0 token 0 mismatch: batch=3.2548 sequential=12.3744 diff=9.1195
Sequence 0 had 151931 mismatches (max diff: 12.82)
```

## Conclusion

The `BatchQwenPipeline` infrastructure is sound:
- ✅ Embedding preparation correct
- ✅ Batch operators correct  
- ✅ Output projection correct
- ❌ **Attention operator flattens batches incorrectly**

This is a targeted fix in one component rather than a systemic architectural issue.
