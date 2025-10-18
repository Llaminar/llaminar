# Full Parallel Batching - Design Document

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Status**: 🔄 **IMPLEMENTATION IN PROGRESS**  
**Target**: Achieve 22× speedup at batch=32

## Design Overview

Transform the sequential batching implementation into true parallel batching by propagating batch dimension through the entire pipeline.

### Current State (Sequential)

```cpp
// Sequential iteration - 0× speedup
for (int i = 0; i < batch_size; ++i) {
    prefill(single_sequence[i]);  // [seq_len, hidden]
}
```

### Target State (Parallel)

```cpp
// Single pass with batch dimension - 22× speedup
prefill_parallel(all_sequences);  // [batch_size, seq_len, hidden]
```

## Implementation Strategy

### Phase 1: Tensor Stacking Infrastructure ✅ (Already Exists)

SimpleTensor and BatchPaddingUtils already support:
- Padding sequences to same length
- Stacking into batch tensor
- Batch dimension handling

### Phase 2: Parallel Prefill Implementation 🔄 (This PR)

**Core Approach**: Process all sequences in a single forward pass
- Stack sequences into batch tensor: `[batch, max_seq_len, ...]`
- Process through pipeline with batch dimension preserved
- Extract per-sequence results at the end

**Key Changes**:

1. **Embedding Layer** (Already batched via MPIEmbeddingOperator)
   - Input: `[batch, seq_len]` token IDs
   - Output: `[batch, seq_len, hidden_size]`

2. **Attention Mechanism** (Needs batch KV cache)
   - Batch Q, K, V projections
   - Batch attention scores: `[batch, n_heads, seq_len, seq_len]`
   - **KV Cache**: Separate cache per sequence in batch

3. **Linear Projections** (Already batched via MPILinearOperator)
   - Reshape `[batch, seq_len, hidden]` → `[batch*seq_len, hidden]`
   - Matmul: `[batch*seq_len, hidden] @ [hidden, out]`
   - Reshape back: `[batch*seq_len, out]` → `[batch, seq_len, out]`

4. **RMSNorm** (Already batched via MPIRMSNormOperator)
   - Normalize per-sequence (batch dimension independent)

### Phase 3: Batch State Management 🔄

**Current**: Swap state for each sequence
```cpp
for (i in batch) {
    n_past_ = n_past_batch_[i];
    k_cache_ = k_cache_batch_[i];
    v_cache_ = v_cache_batch_[i];
    prefill(sequence[i]);
    n_past_batch_[i] = n_past_;
    k_cache_batch_[i] = k_cache_;
    v_cache_batch_[i] = v_cache_;
}
```

**Target**: Process all sequences together, manage state in parallel
```cpp
// All sequences share same position in first prefill
// Each sequence gets its own KV cache slot
prefill_parallel(all_sequences, batch_kv_cache);
```

## Implementation Plan

### Step 1: Refactor prefillBatch to Stack and Process (30 min)

**File**: `src/QwenPipeline.cpp`

```cpp
bool QwenPipeline::prefillBatch(...) {
    // 1. Pad all sequences to same length
    std::vector<std::vector<int>> padded_sequences;
    std::vector<int> original_lengths;
    int max_len = BatchPaddingUtils::padSequences(
        token_batches, padded_sequences, original_lengths, 0);
    
    // 2. Stack into single batch tensor [batch, max_len]
    SimpleTensor batch_tokens({batch_size, max_len});
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < max_len; ++j) {
            batch_tokens.at({i, j}) = padded_sequences[i][j];
        }
    }
    
    // 3. Process entire batch in single forward pass
    bool success = prefillBatchParallel(batch_tokens, original_lengths, ...);
    
    // 4. Extract per-sequence results
    return success;
}
```

### Step 2: Implement prefillBatchParallel (2-3 hours)

**New Method**: Process batch dimension through pipeline

```cpp
bool QwenPipeline::prefillBatchParallel(
    const SimpleTensor& batch_tokens,      // [batch, seq_len]
    const std::vector<int>& seq_lengths,   // actual lengths
    const QwenModelWeights& weights,
    StageContext& ctx,
    std::shared_ptr<TensorBase>& out_logits)
{
    const int batch_size = batch_tokens.shape()[0];
    const int max_seq_len = batch_tokens.shape()[1];
    
    // 1. Batch embedding: [batch, seq_len] → [batch, seq_len, hidden]
    auto embedded = embedBatch(batch_tokens, weights);
    
    // 2. Process through layers with batch dimension
    auto hidden = embedded;
    for (int layer = 0; layer < num_layers; ++layer) {
        // RMSNorm (batch-aware)
        hidden = rmsNormBatch(hidden, weights.layers[layer].attn_norm);
        
        // Attention (batch-aware with separate KV caches)
        hidden = attentionBatch(hidden, seq_lengths, layer, weights);
        
        // FFN
        hidden = ffnBatch(hidden, weights.layers[layer]);
    }
    
    // 3. Final norm and projection
    hidden = rmsNormBatch(hidden, weights.output_norm);
    auto logits = linearBatch(hidden, weights.lm_head);  // [batch, seq_len, vocab]
    
    // 4. Extract last token logits for each sequence
    out_logits = extractLastTokenLogits(logits, seq_lengths);
    
    return true;
}
```

### Step 3: Batch KV Cache Management (1-2 hours)

**Challenge**: Each sequence needs independent KV cache

**Solution**: Restructure cache as `[num_layers][batch_size][seq_len, hidden]`

```cpp
// Current: Single sequence cache
std::vector<SimpleTensor> k_cache_;  // [num_layers][capacity, hidden]
std::vector<SimpleTensor> v_cache_;

// Target: Batch cache
std::vector<std::vector<SimpleTensor>> k_cache_batch_;  // [num_layers][batch][seq, hidden]
std::vector<std::vector<SimpleTensor>> v_cache_batch_;

// In attention:
void attentionBatch(..., int layer) {
    for (int b = 0; b < batch_size; ++b) {
        // Get this sequence's cache
        auto& k_cache = k_cache_batch_[layer][b];
        auto& v_cache = v_cache_batch_[layer][b];
        
        // Append new K, V for this sequence
        // Process attention with batch dimension
    }
}
```

### Step 4: Batch-Aware Operators (3-4 hours)

**Operators already support batching** via existing implementation:
- ✅ MPIEmbeddingOperator: Batch lookup
- ✅ MPILinearOperator: Batched matmul
- ✅ MPIRMSNormOperator: Per-sequence normalization
- 🔄 MPIAttentionOperator: Needs batch KV cache support

**Focus**: Update MPIAttentionOperator for batch processing

```cpp
class MPIAttentionOperator {
    bool executeBatch(
        const SimpleTensor& batch_hidden,  // [batch, seq, hidden]
        std::vector<std::vector<SimpleTensor>>& k_cache,  // [batch][cache]
        std::vector<std::vector<SimpleTensor>>& v_cache,
        SimpleTensor& batch_output)  // [batch, seq, hidden]
    {
        const int batch_size = batch_hidden.shape()[0];
        
        // Process each sequence with its own cache
        for (int b = 0; b < batch_size; ++b) {
            auto seq_hidden = batch_hidden.slice(b);  // [seq, hidden]
            auto seq_output = batch_output.slice(b);
            
            // Existing attention logic with this sequence's cache
            executeSequence(seq_hidden, k_cache[b], v_cache[b], seq_output);
        }
        
        return true;
    }
};
```

### Step 5: Testing and Validation (2-3 hours)

1. **Unit Tests**: Verify batch dimension preserved
2. **Correctness Tests**: Batch results match sequential
3. **Performance Tests**: Measure speedup

## Expected Performance

### Speedup Targets

| Batch Size | Expected Throughput | Expected Speedup |
|------------|---------------------|------------------|
| 1          | ~10 tok/s           | 1.0× (baseline)  |
| 2          | ~20 tok/s           | 2.0×             |
| 4          | ~40 tok/s           | 4.0×             |
| 8          | ~80 tok/s           | 8.0×             |
| 16         | ~160 tok/s          | 16.0×            |
| 32         | ~220 tok/s          | **22.0×** ✅     |

### Why 22× (Not 32×)?

- Overhead from padding shorter sequences
- Memory bandwidth saturation
- MPI communication overhead
- Empirical target based on typical batching efficiency

## Implementation Timeline

- **Step 1**: Tensor stacking (30 min) ✅ Already exists
- **Step 2**: prefillBatchParallel skeleton (1 hour)
- **Step 3**: Batch KV cache (2 hours)
- **Step 4**: Update operators (3 hours)
- **Step 5**: Testing (2 hours)
- **Total**: ~8 hours (1 day)

## Risks and Mitigations

### Risk 1: KV Cache Complexity
**Mitigation**: Start simple - each sequence gets independent cache, optimize later

### Risk 2: Memory Usage
**Mitigation**: Monitor memory, fall back to sequential if OOM

### Risk 3: Correctness
**Mitigation**: Extensive testing against sequential baseline

## Success Criteria

- ✅ All correctness tests pass (batch matches sequential)
- ✅ batch=4 achieves ≥3.5× speedup
- ✅ batch=32 achieves ≥20× speedup
- ✅ No memory leaks or crashes
- ✅ Clean instrumentation output

---

**Status**: Design complete, ready for implementation  
**Next**: Implement Step 2 (prefillBatchParallel)
