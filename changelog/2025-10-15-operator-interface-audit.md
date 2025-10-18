# Operator Interface Audit for Batch Dimension Support

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Purpose**: Audit current operator interfaces and document required changes for batch dimension support

## Executive Summary

This document audits all MPI operators to identify interface changes needed for batch dimension support. Current operators handle 2D tensors `[seq, hidden]`. Target is 3D tensors `[batch, seq, hidden]` with single forward pass for entire batch.

## Audit Scope

**Operators Audited**:
1. MPIEmbeddingOperator
2. MPILinearOperator  
3. MPIRMSNormOperator
4. MPIAttentionOperator
5. MPISoftmaxOperator (used within attention)

**Files Reviewed**:
- `src/kernels/mpi_embedding_operator.{h,cpp}`
- `src/kernels/mpi_linear_operator.{h,cpp}`
- `src/kernels/mpi_rmsnorm_operator.{h,cpp}`
- `src/kernels/mpi_attention_operator.{h,cpp}`
- `src/kernels/mpi_softmax_operator.{h,cpp}`

## Current Interface Analysis

### 1. MPIEmbeddingOperator

**Current Interface**:
```cpp
class MPIEmbeddingOperator : public MPIKernelBase {
public:
    MPIEmbeddingOperator(int vocab_size, int embedding_dim, const float* embeddings);
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs
    ) override;
};
```

**Current Behavior**:
- Input: `inputs[0]` = token IDs, shape `[seq_len]` (1D)
- Output: `outputs[0]` = embeddings, shape `[seq_len, embedding_dim]` (2D)
- Processing: Loop over sequence, lookup embeddings, stack

**Required Changes for Batch Support**:
- Input: `inputs[0]` = token IDs, shape `[batch, seq_len]` (2D)
- Output: `outputs[0]` = embeddings, shape `[batch, seq_len, embedding_dim]` (3D)
- Processing: Nested loop over batch and sequence, handle padding tokens

**Interface Changes**:
- ✅ No signature changes needed (polymorphic via TensorBase)
- ⚠️ Implementation needs batch dimension handling
- ⚠️ Need padding token handling (PAD_TOKEN = -1 or special value)

**Backward Compatibility**:
- Can detect batch dimension from `inputs[0]->shape().size()`
- If 1D: Process as single sequence (current behavior)
- If 2D: Process as batch (new behavior)

**Risk Assessment**: **LOW** (straightforward extension of current logic)

---

### 2. MPILinearOperator

**Current Interface**:
```cpp
class MPILinearOperator : public MPIKernelBase {
public:
    MPILinearOperator(int in_features, int out_features, 
                      const float* weights, const float* bias = nullptr);
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs
    ) override;
};
```

**Current Behavior**:
- Input: `inputs[0]` = activations, shape `[seq_len, in_features]` (2D)
- Output: `outputs[0]` = projected, shape `[seq_len, out_features]` (2D)
- Processing: Matrix multiplication `[seq_len, in_features] @ [in_features, out_features]`

**Required Changes for Batch Support**:
- Input: `inputs[0]` = activations, shape `[batch, seq_len, in_features]` (3D)
- Output: `outputs[0]` = projected, shape `[batch, seq_len, out_features]` (3D)
- Processing: Reshape to 2D, matmul, reshape back
  ```
  [batch, seq, in] → [batch*seq, in] @ [in, out] → [batch*seq, out] → [batch, seq, out]
  ```

**Interface Changes**:
- ✅ No signature changes needed
- ⚠️ Implementation needs reshape logic
- ⚠️ Ensure contiguous memory layout for efficient BLAS

**Backward Compatibility**:
- Detect input dimension from `inputs[0]->shape().size()`
- If 2D: Direct matmul (current behavior)
- If 3D: Reshape, matmul, reshape back (new behavior)

**Risk Assessment**: **LOW** (standard reshape pattern, well-understood)

**Performance Consideration**:
- Reshaping is view operation (no copy if contiguous)
- BLAS libraries handle large batched matmuls efficiently
- Expected to scale well

---

### 3. MPIRMSNormOperator

**Current Interface**:
```cpp
class MPIRMSNormOperator : public MPIKernelBase {
public:
    MPIRMSNormOperator(int hidden_size, const float* scale, float eps = 1e-6);
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs
    ) override;
};
```

**Current Behavior**:
- Input: `inputs[0]` = activations, shape `[seq_len, hidden_size]` (2D)
- Output: `outputs[0]` = normalized, shape `[seq_len, hidden_size]` (2D)
- Processing: For each token:
  ```
  rms = sqrt(mean(x^2) + eps)
  output = (x / rms) * scale
  ```

**Required Changes for Batch Support**:
- Input: `inputs[0]` = activations, shape `[batch, seq_len, hidden_size]` (3D)
- Output: `outputs[0]` = normalized, shape `[batch, seq_len, hidden_size]` (3D)
- Processing: For each (batch, token) pair independently:
  ```
  for b in batch:
      for s in seq:
          rms[b,s] = sqrt(mean(x[b,s]^2) + eps)
          output[b,s] = (x[b,s] / rms[b,s]) * scale
  ```

**Interface Changes**:
- ✅ No signature changes needed
- ⚠️ Implementation needs nested loops over batch and sequence
- ⚠️ Each sequence normalized independently (no batch statistics)

**Backward Compatibility**:
- Detect input dimension
- If 2D: Current behavior (loop over seq only)
- If 3D: New behavior (nested loops over batch and seq)

**Risk Assessment**: **LOW** (independent per-token operation)

**Parallelization Opportunity**:
```cpp
#pragma omp parallel for collapse(2)
for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
        // Compute RMS and normalize for this token
    }
}
```

**Expected Speedup**: Linear with batch size (embarrassingly parallel)

---

### 4. MPIAttentionOperator

**Current Interface**:
```cpp
class MPIAttentionOperator : public MPIKernelBase {
public:
    MPIAttentionOperator(int hidden_size, int num_heads, int head_dim,
                         const float* q_weights, const float* k_weights,
                         const float* v_weights, const float* o_weights);
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs
    ) override;
    
    void set_kv_cache(const std::shared_ptr<TensorBase>& k_cache,
                      const std::shared_ptr<TensorBase>& v_cache);
};
```

**Current Behavior**:
- Input: `inputs[0]` = activations, shape `[seq_len, hidden_size]` (2D)
- Output: `outputs[0]` = attention output, shape `[seq_len, hidden_size]` (2D)
- KV Cache: Single shared cache for the sequence
- Processing:
  1. Q/K/V projections: `[seq, hidden] @ [hidden, n_heads*head_dim]`
  2. Reshape for heads: `[seq, n_heads, head_dim]`
  3. Concatenate with cache: `K_full = concat(k_cache, K_new)`
  4. Attention: `softmax(Q @ K_full^T / sqrt(head_dim)) @ V_full`
  5. Output projection

**Required Changes for Batch Support**:
- Input: `inputs[0]` = activations, shape `[batch, seq_len, hidden_size]` (3D)
- Output: `outputs[0]` = attention output, shape `[batch, seq_len, hidden_size]` (3D)
- KV Cache: **Per-sequence cache** (critical change!)
  - Need `BatchedKVCache` class
  - Structure: `[num_layers][batch_size]` each holding cache for one sequence
- Processing: For each sequence in batch independently:
  1. Q/K/V projections (batched): `[batch, seq, hidden] @ [hidden, n_heads*head_dim]`
  2. Reshape: `[batch, seq, n_heads, head_dim]`
  3. **Per sequence**: Concatenate with that sequence's cache
  4. **Per sequence**: Compute attention scores
  5. Combine results back into `[batch, seq, hidden]`

**Interface Changes**:
- ⚠️ **BREAKING**: KV cache interface needs complete redesign
- Current `set_kv_cache` assumes single cache
- Need new interface:
  ```cpp
  void set_kv_cache_batch(std::shared_ptr<BatchedKVCache> cache, size_t layer_idx);
  ```

**Backward Compatibility**:
- Detect batch dimension
- If 2D input: Use current single-cache behavior
- If 3D input: Require BatchedKVCache, process per-sequence

**Risk Assessment**: **HIGH** (most complex operator, KV cache restructuring)

**Critical Design Decisions**:

1. **KV Cache Isolation**: Each sequence must have independent cache
   - Prevents cross-contamination
   - Enables different context lengths per sequence

2. **Attention Computation Strategy**:
   - Option A: Process all sequences in parallel (OpenMP)
   - Option B: Vectorize where possible, sequential where needed
   - **Recommendation**: Option A (simpler, still fast)

3. **Memory Layout**:
   - KV cache can be large: `[batch, total_seq, hidden]` per layer
   - Need efficient allocation and access patterns

**Implementation Complexity**: **HIGH**
- Estimated 3-4 days for complete implementation
- Requires BatchedKVCache class first
- Extensive testing needed

---

### 5. MPISoftmaxOperator

**Current Interface**:
```cpp
class MPISoftmaxOperator : public MPIKernelBase {
public:
    MPISoftmaxOperator();
    
    bool execute(
        const std::vector<std::shared_ptr<TensorBase>>& inputs,
        std::vector<std::shared_ptr<TensorBase>>& outputs
    ) override;
};
```

**Current Behavior**:
- Input: `inputs[0]` = scores, shape `[n_heads, seq_query, seq_key]` (3D)
- Output: `outputs[0]` = probabilities, shape `[n_heads, seq_query, seq_key]` (3D)
- Processing: Softmax over last dimension (key dimension)

**Required Changes for Batch Support**:
- Input: `inputs[0]` = scores, shape `[batch, n_heads, seq_query, seq_key]` (4D)
- Output: `outputs[0]` = probabilities, shape `[batch, n_heads, seq_query, seq_key]` (4D)
- Processing: Softmax over last dimension, independently for each (batch, head, query)

**Interface Changes**:
- ✅ No signature changes needed
- ⚠️ Implementation needs to handle 4D tensors
- ⚠️ Softmax still over last dimension

**Backward Compatibility**:
- Detect input dimension
- If 3D: Current behavior
- If 4D: Add outer batch loop

**Risk Assessment**: **LOW** (straightforward extension)

**Parallelization**:
```cpp
#pragma omp parallel for collapse(3)
for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t q = 0; q < seq_query; ++q) {
            // Softmax over seq_key dimension
        }
    }
}
```

---

## Summary of Interface Changes

| Operator | Current Input Shape | Target Input Shape | Signature Changes | Risk | Complexity |
|----------|---------------------|-------------------|-------------------|------|------------|
| **MPIEmbeddingOperator** | `[seq]` | `[batch, seq]` | None | LOW | Low |
| **MPILinearOperator** | `[seq, hidden]` | `[batch, seq, hidden]` | None | LOW | Low |
| **MPIRMSNormOperator** | `[seq, hidden]` | `[batch, seq, hidden]` | None | LOW | Low |
| **MPIAttentionOperator** | `[seq, hidden]` | `[batch, seq, hidden]` | **Yes** (KV cache) | **HIGH** | **High** |
| **MPISoftmaxOperator** | `[heads, seq_q, seq_k]` | `[batch, heads, seq_q, seq_k]` | None | LOW | Low |

## Required New Components

### 1. BatchedKVCache Class

**Purpose**: Manage per-sequence KV cache for batched inference

**Interface Design**:
```cpp
class BatchedKVCache {
public:
    // Constructor
    BatchedKVCache(size_t num_layers, size_t batch_size, 
                   size_t max_seq_len, size_t hidden_dim);
    
    // Access
    std::shared_ptr<TensorBase> get_k(size_t layer, size_t batch_idx) const;
    std::shared_ptr<TensorBase> get_v(size_t layer, size_t batch_idx) const;
    
    // Modification
    void append_kv(size_t layer, size_t batch_idx,
                   const std::shared_ptr<TensorBase>& new_k,
                   const std::shared_ptr<TensorBase>& new_v);
    
    // Batch operations
    void reset_batch(size_t batch_idx);  // Clear one sequence
    void clear_all();                     // Clear all caches
    
    // Queries
    size_t num_layers() const;
    size_t batch_size() const;
    size_t sequence_length(size_t layer, size_t batch_idx) const;
    size_t max_sequence_length() const;
    
private:
    // Storage: [num_layers][batch_size] → TensorBase (variable length)
    std::vector<std::vector<std::shared_ptr<TensorBase>>> k_cache_;
    std::vector<std::vector<std::shared_ptr<TensorBase>>> v_cache_;
    
    // Metadata
    std::vector<std::vector<size_t>> seq_lengths_;  // Current length per sequence
    size_t num_layers_;
    size_t batch_size_;
    size_t max_seq_len_;
    size_t hidden_dim_;
};
```

**File Location**: `src/tensors/batched_kv_cache.{h,cpp}`

**Priority**: **HIGH** (blocking for attention operator)

### 2. BatchPaddingUtils Extensions

**Current**: Handles padding and stacking for 1D token sequences  
**Needed**: Handle any tensor dimension

**Additional Methods**:
```cpp
class BatchPaddingUtils {
public:
    // Existing methods...
    
    // Pad 2D tensors [seq, hidden] to [max_seq, hidden]
    static std::shared_ptr<TensorBase> pad_2d_tensor(
        const std::shared_ptr<TensorBase>& tensor,
        size_t target_seq_len,
        float pad_value = 0.0f
    );
    
    // Stack 2D tensors into 3D [batch, seq, hidden]
    static std::shared_ptr<TensorBase> stack_2d_tensors(
        const std::vector<std::shared_ptr<TensorBase>>& tensors,
        float pad_value = 0.0f
    );
    
    // Extract single sequence from batched tensor
    static std::shared_ptr<TensorBase> extract_from_batch(
        const std::shared_ptr<TensorBase>& batch_tensor,
        size_t batch_idx,
        size_t actual_seq_len  // Trim padding
    );
};
```

**Priority**: **MEDIUM** (needed for pipeline integration)

## Implementation Order

**Phase 1: Foundation** (Days 2-7)
1. SimpleTensor batch dimension support
2. BatchedKVCache class implementation
3. BatchPaddingUtils extensions
4. Operator interface updates (signatures only)

**Phase 2: Simple Operators** (Days 8-13)
1. MPIEmbeddingOperator (batched)
2. MPILinearOperator (batched)
3. MPIRMSNormOperator (batched)
4. MPISoftmaxOperator (batched)

**Phase 3: Complex Operators** (Days 14-18)
1. MPIAttentionOperator Phase 1 (Q/K/V projections)
2. MPIAttentionOperator Phase 2 (attention computation)
3. MPIAttentionOperator Phase 3 (KV cache integration)

**Phase 4: Integration** (Days 19-21)
1. QwenPipeline parallel batching
2. State management updates
3. Integration testing

## Testing Strategy

### Unit Tests (Per Operator)
- Shape validation (input/output dimensions)
- Correctness vs sequential (batch vs loop over sequences)
- Edge cases (batch=1, empty sequences, max length)
- Backward compatibility (2D inputs still work)

### Integration Tests
- End-to-end batch processing
- Multi-step generation with KV cache
- Variable sequence lengths in batch
- Memory leak detection

### Performance Tests
- Throughput measurement (tok/s)
- Speedup ratio (batch vs single)
- Scaling efficiency (linear expected)
- Profiling (identify bottlenecks)

## Risks and Mitigation

### High-Risk Items

**1. KV Cache Complexity**
- **Risk**: BatchedKVCache implementation bugs, memory issues
- **Mitigation**: Extensive unit testing, gradual rollout
- **Contingency**: 2-3 extra days budgeted for debugging

**2. Attention Operator Integration**
- **Risk**: Batched attention with per-sequence cache is complex
- **Mitigation**: Break into 3 phases, test incrementally
- **Contingency**: Can fall back to sequential per-sequence if needed

**3. Memory Consumption**
- **Risk**: Batched KV cache may exceed available memory
- **Mitigation**: Conservative max_batch_size, memory profiling
- **Contingency**: Add memory checks, reduce batch size limits

### Medium-Risk Items

**1. Backward Compatibility**
- **Risk**: Breaking existing single-sequence code paths
- **Mitigation**: Maintain batch=1 as special case, extensive regression testing
- **Contingency**: Feature flag to disable batching if issues found

**2. Performance Target Not Met**
- **Risk**: May not achieve full 22× speedup
- **Mitigation**: Profile early, optimize incrementally
- **Contingency**: Days 24-25 dedicated to optimization

## Backward Compatibility Strategy

**Goal**: All operators support both 2D (single sequence) and 3D (batch) inputs without breaking changes

**Detection Pattern**:
```cpp
bool MPIOperator::execute(inputs, outputs) {
    auto input = inputs[0];
    size_t ndim = input->shape().size();
    
    if (ndim == 2) {
        // Legacy path: single sequence
        return execute_single(input, outputs);
    } else if (ndim == 3) {
        // New path: batched
        return execute_batched(input, outputs);
    } else {
        LOG_ERROR("Unsupported input dimension: " << ndim);
        return false;
    }
}
```

**Benefits**:
- Gradual migration (operators can be updated independently)
- Existing tests continue to work
- Easy debugging (can compare batch=1 vs single sequence)

## Conclusion

**Audit Summary**:
- ✅ 5 operators audited
- ✅ Interface changes identified
- ✅ Backward compatibility strategy defined
- ✅ Risk assessment complete
- ✅ Implementation order planned

**Key Findings**:
1. Most operators need implementation changes but no signature changes
2. MPIAttentionOperator requires KV cache redesign (highest risk)
3. BatchedKVCache is critical blocking component
4. Backward compatibility maintainable via dimension detection

**Next Steps**:
1. ✅ Complete Day 1 audit (this document)
2. Tomorrow: Begin SimpleTensor batch dimension support
3. Day 4-5: Implement operator interface updates
4. Day 6-7: Implement BatchedKVCache class

**Status**: Day 1 Audit Complete ✅  
**Updated**: October 15, 2025
