# Option A: Full Parallel Batching - Detailed Implementation Plan

**Date**: October 15, 2025  
**Author**: David Sanftenberg  
**Status**: 🔄 IN PROGRESS - Day 1  
**Timeline**: 2-3 weeks (Oct 15 - Nov 7, 2025)  
**Goal**: Achieve 22× speedup at batch=32 through true parallel batch processing

## Executive Summary

This document provides a day-by-day implementation plan for Option A: Full Parallel Batching. The plan transforms the current sequential batching implementation into a true parallel system where all sequences in a batch are processed simultaneously through the pipeline.

**Current State**: Sequential for-loop processes sequences one-by-one (0.97× speedup)  
**Target State**: Single forward pass processes entire batch (22× speedup)

## Background

From Phase 4.3 investigation (Oct 15, 2025):
- **Root Cause**: Sequential for-loop in `QwenPipeline::prefillBatch` (lines 2090-2142)
- **Evidence**: 99.98% of time in sequential iteration, only 6.5ms overhead
- **Performance Gap**: 21× speedup missing (0.97× actual vs 22× target)
- **Baseline**: Clean, production-ready code ready for refactoring

## Architecture Overview

### Current Architecture (Sequential)
```
Input: vector<vector<int>> token_batches (batch_size sequences)
│
├─ for i in 0..batch_size:  ❌ SEQUENTIAL LOOP
│   ├─ Restore state for sequence i
│   ├─ prefill(token_batches[i])  ← Single sequence
│   ├─ Save state for sequence i
│   └─ Extract logits[i]
│
Output: Concatenated logits
```

### Target Architecture (Parallel)
```
Input: vector<vector<int>> token_batches (batch_size sequences)
│
├─ Pad/stack to [batch, max_seq] tensor
│
├─ Embedding: [batch, max_seq] → [batch, max_seq, hidden]  ✅ PARALLEL
│
├─ For each layer:
│   ├─ RMSNorm: [batch, seq, hidden] (per-sequence stats)  ✅ PARALLEL
│   ├─ Attention: [batch, seq, hidden] with batch KV cache  ✅ PARALLEL
│   ├─ Linear: [batch, seq, hidden] → reshape matmul       ✅ PARALLEL
│   └─ Residual: [batch, seq, hidden]                      ✅ PARALLEL
│
├─ Final norm & projection
│
├─ Extract per-sequence logits: logits[i] = output[i][-1]
│
Output: Per-sequence logits
```

## Implementation Plan by Week

### Week 1: Design & Tensor Batch Dimension (Oct 16-22)

#### Day 1 (Oct 16): Design Review & Architecture Validation ✅ TODAY

**Objectives**:
1. Review and validate the parallel batching design
2. Identify all code changes needed across operators
3. Create detailed task breakdown
4. Set up tracking and documentation

**Tasks**:
- [x] Review `changelog/2025-10-15-parallel-batching-design.md`
- [x] Update TODO.md with detailed day-by-day plan
- [x] Create this implementation plan document
- [ ] Audit current operator interfaces for batch dimension support
- [ ] Document required interface changes
- [ ] Identify potential risks and mitigation strategies
- [ ] Set up progress tracking structure

**Deliverables**:
- ✅ Updated TODO.md with 28-day plan
- ✅ This detailed implementation plan
- [ ] Operator interface audit document
- [ ] Risk assessment document

**Time Estimate**: 4-6 hours

#### Day 2-3 (Oct 17-18): SimpleTensor Batch Dimension Support

**Objectives**:
1. Add 3D tensor support to SimpleTensor
2. Implement batch indexing and slicing utilities
3. Create comprehensive unit tests

**Current SimpleTensor Interface**:
```cpp
class SimpleTensor : public TensorBase {
    SimpleTensor(const std::vector<size_t>& shape);  // Supports any dimension
    std::vector<float>& data();
    const std::vector<size_t>& shape() const;
};
```

**Additions Needed**:
```cpp
// Batch-aware constructors (convenience)
static std::shared_ptr<SimpleTensor> create_batch(
    size_t batch_size, size_t seq_len, size_t hidden_dim);

// Batch indexing (get view of single sequence in batch)
std::shared_ptr<SimpleTensor> get_batch(size_t batch_idx) const;

// Batch slicing (get view of range of sequences)
std::shared_ptr<SimpleTensor> slice_batch(size_t start, size_t end) const;

// Set batch data (copy into batch slot)
void set_batch(size_t batch_idx, const std::shared_ptr<TensorBase>& data);

// Batch utilities
size_t batch_size() const;      // Return shape[0] if 3D
size_t sequence_length() const; // Return shape[1] if 3D
size_t hidden_dim() const;      // Return shape[2] if 3D
```

**Tasks**:
- [ ] Implement `create_batch` factory method
- [ ] Implement `get_batch` with view semantics (shared data)
- [ ] Implement `slice_batch` for range access
- [ ] Implement `set_batch` for batch slot assignment
- [ ] Add batch dimension utility methods
- [ ] Create unit tests: `tests/test_batch_tensor_operations.cpp`
  - [ ] Test 3D tensor creation
  - [ ] Test batch indexing correctness
  - [ ] Test batch slicing
  - [ ] Test set_batch operations
  - [ ] Test batch dimension queries
  - [ ] Test edge cases (batch=1, empty, out of bounds)

**Deliverables**:
- SimpleTensor with batch operations
- Unit test suite (≥15 test cases)
- All tests passing

**Time Estimate**: 2 days (12-16 hours)

#### Day 4-5 (Oct 19-20): Operator Interface Updates

**Objectives**:
1. Update all operator interfaces to accept batch dimensions
2. Create stub implementations that pass shape validation
3. Add interface compatibility tests

**Operators to Update**:

**MPIEmbeddingOperator**:
```cpp
// Current: [seq_len] → [seq_len, hidden]
// Target:  [batch, seq_len] → [batch, seq_len, hidden]

bool execute(
    const std::vector<std::shared_ptr<TensorBase>>& inputs,  // [0] = [batch, seq]
    std::vector<std::shared_ptr<TensorBase>>& outputs        // [0] = [batch, seq, hidden]
) override;
```

**MPILinearOperator**:
```cpp
// Current: [seq_len, hidden] → [seq_len, out_dim]
// Target:  [batch, seq_len, hidden] → [batch, seq_len, out_dim]

bool execute(
    const std::vector<std::shared_ptr<TensorBase>>& inputs,  // [0] = [batch, seq, hidden]
    std::vector<std::shared_ptr<TensorBase>>& outputs        // [0] = [batch, seq, out_dim]
) override;
```

**MPIRMSNormOperator**:
```cpp
// Current: [seq_len, hidden] → [seq_len, hidden]
// Target:  [batch, seq_len, hidden] → [batch, seq_len, hidden]
// Note: Compute stats per sequence independently

bool execute(
    const std::vector<std::shared_ptr<TensorBase>>& inputs,  // [0] = [batch, seq, hidden]
    std::vector<std::shared_ptr<TensorBase>>& outputs        // [0] = [batch, seq, hidden]
) override;
```

**MPIAttentionOperator**:
```cpp
// Current: [seq_len, hidden] + KV cache → [seq_len, hidden]
// Target:  [batch, seq_len, hidden] + batch KV cache → [batch, seq_len, hidden]

bool execute(
    const std::vector<std::shared_ptr<TensorBase>>& inputs,  // [0] = [batch, seq, hidden]
    std::vector<std::shared_ptr<TensorBase>>& outputs        // [0] = [batch, seq, hidden]
) override;

// Additional: Batch KV cache management
void set_kv_cache_batch(
    size_t batch_idx,
    const std::vector<std::shared_ptr<TensorBase>>& k_cache,
    const std::vector<std::shared_ptr<TensorBase>>& v_cache
);
```

**Tasks**:
- [ ] Update MPIEmbeddingOperator interface (stub implementation)
- [ ] Update MPILinearOperator interface (stub implementation)
- [ ] Update MPIRMSNormOperator interface (stub implementation)
- [ ] Update MPIAttentionOperator interface (stub implementation)
- [ ] Create interface validation tests
  - [ ] Test input/output shape validation
  - [ ] Test batch dimension handling
  - [ ] Test backward compatibility (batch=1 case)
- [ ] Document interface changes

**Deliverables**:
- Updated operator interfaces
- Stub implementations (shape validation only)
- Interface compatibility tests
- API documentation updates

**Time Estimate**: 2 days (12-16 hours)

#### Day 6-7 (Oct 21-22): KV Cache Restructuring - Phase 1

**Objectives**:
1. Design and implement batched KV cache structure
2. Add batch-aware allocation and indexing
3. Create comprehensive tests

**Current KV Cache Structure**:
```cpp
// QwenPipeline.h
std::vector<std::shared_ptr<TensorBase>> k_cache_;  // [num_layers]
std::vector<std::shared_ptr<TensorBase>> v_cache_;  // [num_layers]

// Per sequence batch state (Phase 3.3)
std::vector<int> n_past_batch_;                                        // [batch_size]
std::vector<std::vector<std::shared_ptr<TensorBase>>> k_cache_batch_;  // [batch_size][num_layers]
std::vector<std::vector<std::shared_ptr<TensorBase>>> v_cache_batch_;  // [batch_size][num_layers]
```

**Target KV Cache Structure**:
```cpp
// New batched structure
class BatchedKVCache {
public:
    BatchedKVCache(size_t num_layers, size_t batch_size, size_t max_seq_len, size_t hidden_dim);
    
    // Access
    std::shared_ptr<TensorBase> get_k(size_t layer, size_t batch_idx);
    std::shared_ptr<TensorBase> get_v(size_t layer, size_t batch_idx);
    
    // Modification
    void append_kv(size_t layer, size_t batch_idx, 
                   const std::shared_ptr<TensorBase>& new_k,
                   const std::shared_ptr<TensorBase>& new_v);
    
    // Batch operations
    void reset_batch(size_t batch_idx);
    void clear_all();
    
    // Queries
    size_t num_layers() const;
    size_t batch_size() const;
    size_t sequence_length(size_t layer, size_t batch_idx) const;
    
private:
    std::vector<std::vector<std::shared_ptr<TensorBase>>> k_cache_;  // [num_layers][batch_size]
    std::vector<std::vector<std::shared_ptr<TensorBase>>> v_cache_;  // [num_layers][batch_size]
    std::vector<std::vector<size_t>> seq_lengths_;  // [num_layers][batch_size]
    
    size_t num_layers_;
    size_t batch_size_;
    size_t max_seq_len_;
    size_t hidden_dim_;
};
```

**Tasks**:
- [ ] Design `BatchedKVCache` class interface
- [ ] Implement allocation and initialization
- [ ] Implement batch indexing (`get_k`, `get_v`)
- [ ] Implement append operations
- [ ] Implement batch reset and clear
- [ ] Create unit tests: `tests/test_batched_kv_cache.cpp`
  - [ ] Test allocation and initialization
  - [ ] Test batch indexing correctness
  - [ ] Test append operations
  - [ ] Test sequence length tracking
  - [ ] Test batch isolation (no cross-contamination)
  - [ ] Test reset and clear operations
- [ ] Integration with QwenPipeline (interface only)

**Deliverables**:
- `BatchedKVCache` class implementation
- Unit test suite (≥12 test cases)
- All tests passing
- Integration interface defined

**Time Estimate**: 2 days (12-16 hours)

### Week 2: Operator Implementation - Part 1 (Oct 23-29)

#### Day 8-9 (Oct 23-24): Batched Embedding Operator

**Objectives**:
1. Implement parallel batch embedding lookup
2. Handle variable sequence lengths with padding
3. Comprehensive testing and validation

**Implementation Strategy**:
```cpp
// Input:  [batch, max_seq] with padding
// Output: [batch, max_seq, hidden]

bool MPIEmbeddingOperator::execute(inputs, outputs) {
    auto token_ids = inputs[0];  // [batch, max_seq]
    size_t batch_size = token_ids->shape()[0];
    size_t max_seq = token_ids->shape()[1];
    
    // Allocate output
    auto output = SimpleTensor::create_batch(batch_size, max_seq, hidden_dim_);
    
    // Process each sequence in batch
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < max_seq; ++s) {
            int token_id = token_ids->data()[b * max_seq + s];
            if (token_id == PAD_TOKEN) continue;  // Skip padding
            
            // Lookup embedding
            const float* embedding = &embeddings_[token_id * hidden_dim_];
            float* output_slot = &output->data()[(b * max_seq + s) * hidden_dim_];
            std::copy(embedding, embedding + hidden_dim_, output_slot);
        }
    }
    
    outputs[0] = output;
    return true;
}
```

**Tasks**:
- [ ] Implement batched embedding lookup
- [ ] Add padding token handling
- [ ] Optimize memory access patterns
- [ ] Add shape validation
- [ ] Create comprehensive tests
  - [ ] Correctness: Compare batch output vs sequential per-sequence
  - [ ] Variable lengths: Test different sequence lengths in batch
  - [ ] Edge cases: batch=1, all padding, empty sequences
  - [ ] Performance: Measure speedup vs sequential
- [ ] Benchmark batch sizes 1,4,8,16,32

**Deliverables**:
- Working batched embedding operator
- Test suite (≥8 test cases)
- Performance measurements
- All tests passing

**Time Estimate**: 2 days (12-16 hours)

#### Day 10-11 (Oct 25-26): Batched Linear Operator

**Objectives**:
1. Implement batched matrix multiplication
2. Optimize reshape operations
3. Testing and validation

**Implementation Strategy**:
```cpp
// Input:  [batch, seq, hidden]
// Output: [batch, seq, out_dim]

bool MPILinearOperator::execute(inputs, outputs) {
    auto input = inputs[0];  // [batch, seq, hidden]
    size_t batch_size = input->shape()[0];
    size_t seq_len = input->shape()[1];
    size_t hidden_dim = input->shape()[2];
    
    // Reshape to 2D: [batch*seq, hidden]
    size_t total_tokens = batch_size * seq_len;
    const float* input_2d = input->data();  // Already contiguous
    
    // Allocate output: [batch*seq, out_dim]
    std::vector<float> output_2d(total_tokens * out_dim_);
    
    // Batched matmul: [batch*seq, hidden] @ [hidden, out_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total_tokens, out_dim_, hidden_dim,
                1.0f,
                input_2d, hidden_dim,
                weights_, out_dim_,
                0.0f,
                output_2d.data(), out_dim_);
    
    // Add bias if present
    if (bias_) {
        #pragma omp parallel for
        for (size_t i = 0; i < total_tokens; ++i) {
            for (size_t j = 0; j < out_dim_; ++j) {
                output_2d[i * out_dim_ + j] += bias_[j];
            }
        }
    }
    
    // Reshape back to 3D: [batch, seq, out_dim]
    auto output = SimpleTensor::create_batch(batch_size, seq_len, out_dim_);
    std::copy(output_2d.begin(), output_2d.end(), output->data().begin());
    
    outputs[0] = output;
    return true;
}
```

**Tasks**:
- [ ] Implement batched matmul with reshape
- [ ] Add bias handling
- [ ] Optimize memory layout (avoid unnecessary copies)
- [ ] Add shape validation
- [ ] Create comprehensive tests
  - [ ] Correctness: Compare batch vs sequential
  - [ ] Various shapes: Different batch/seq/hidden combinations
  - [ ] Edge cases: batch=1, seq=1
  - [ ] Performance: Measure GFLOPS
- [ ] Benchmark and profile

**Deliverables**:
- Working batched linear operator
- Test suite (≥6 test cases)
- Performance measurements
- All tests passing

**Time Estimate**: 2 days (12-16 hours)

#### Day 12-13 (Oct 27-28): Batched RMSNorm Operator

**Objectives**:
1. Implement per-sequence normalization
2. Parallelize across batch dimension
3. Testing and validation

**Implementation Strategy**:
```cpp
// Input:  [batch, seq, hidden]
// Output: [batch, seq, hidden]
// Note: Compute RMS statistics per sequence independently

bool MPIRMSNormOperator::execute(inputs, outputs) {
    auto input = inputs[0];  // [batch, seq, hidden]
    size_t batch_size = input->shape()[0];
    size_t seq_len = input->shape()[1];
    size_t hidden_dim = input->shape()[2];
    
    auto output = SimpleTensor::create_batch(batch_size, seq_len, hidden_dim);
    
    // Process each sequence independently
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t offset = (b * seq_len + s) * hidden_dim;
            const float* input_ptr = &input->data()[offset];
            float* output_ptr = &output->data()[offset];
            
            // Compute RMS
            float sum_sq = 0.0f;
            for (size_t h = 0; h < hidden_dim; ++h) {
                sum_sq += input_ptr[h] * input_ptr[h];
            }
            float rms = std::sqrt(sum_sq / hidden_dim + eps_);
            
            // Normalize and scale
            for (size_t h = 0; h < hidden_dim; ++h) {
                output_ptr[h] = (input_ptr[h] / rms) * scale_[h];
            }
        }
    }
    
    outputs[0] = output;
    return true;
}
```

**Tasks**:
- [ ] Implement per-sequence RMS computation
- [ ] Add OpenMP parallelization
- [ ] Optimize for memory locality
- [ ] Add numerical stability checks
- [ ] Create comprehensive tests
  - [ ] Correctness: Compare batch vs sequential
  - [ ] Per-sequence independence: Verify no cross-contamination
  - [ ] Numerical stability: Test with extreme values
  - [ ] Performance: Measure parallel efficiency
- [ ] Benchmark and tune OpenMP settings

**Deliverables**:
- Working batched RMSNorm operator
- Test suite (≥6 test cases)
- Performance measurements
- All tests passing

**Time Estimate**: 2 days (10-14 hours)

#### Day 14 (Oct 29): Batched Attention - Phase 1 (Setup)

**Objectives**:
1. Design batched attention computation strategy
2. Implement Q/K/V projections in batch mode
3. Set up KV cache integration interface

**Implementation Strategy (Overview)**:
```cpp
// Input:  [batch, seq, hidden]
// Output: [batch, seq, hidden]
// KV Cache: BatchedKVCache (per layer, per sequence)

bool MPIAttentionOperator::execute(inputs, outputs) {
    auto x = inputs[0];  // [batch, seq, hidden]
    
    // Phase 1: Q/K/V projections (batched linear)
    auto Q = project_q(x);  // [batch, seq, n_heads * head_dim]
    auto K = project_k(x);  // [batch, seq, n_heads * head_dim]
    auto V = project_v(x);  // [batch, seq, n_heads * head_dim]
    
    // Phase 2: Attention computation (next day)
    // ...
}
```

**Tasks**:
- [ ] Design batched attention flow
- [ ] Implement batched Q/K/V linear projections
- [ ] Add shape validation for multi-head dimensions
- [ ] Create KV cache integration interface
- [ ] Initial tests: Q/K/V shape correctness

**Deliverables**:
- Q/K/V projection implementation
- Shape validation tests
- KV cache interface design

**Time Estimate**: 1 day (6-8 hours)

### Week 3: Attention & Pipeline Integration (Oct 30-Nov 5)

#### Day 15-16 (Oct 30-31): Batched Attention - Phase 2 (Core)

**Objectives**:
1. Implement batched attention scores computation
2. Implement batched softmax
3. Implement batched output generation

**Implementation Strategy**:
```cpp
// Continuation of execute()

// Reshape for multi-head: [batch, seq, n_heads, head_dim]
auto Q_heads = reshape_for_heads(Q);
auto K_heads = reshape_for_heads(K);
auto V_heads = reshape_for_heads(V);

auto output = SimpleTensor::create_batch(batch_size, seq_len, hidden_dim);

// Process each sequence independently
#pragma omp parallel for
for (size_t b = 0; b < batch_size; ++b) {
    // Get KV cache for this sequence
    auto k_cache = kv_cache_->get_k(layer_idx_, b);
    auto v_cache = kv_cache_->get_v(layer_idx_, b);
    
    // Concatenate current K/V with cache
    auto K_full = concat(k_cache, K_heads->get_batch(b));  // [n_heads, total_seq, head_dim]
    auto V_full = concat(v_cache, V_heads->get_batch(b));  // [n_heads, total_seq, head_dim]
    
    // Attention: Q @ K^T / sqrt(head_dim)
    auto scores = matmul(Q_heads->get_batch(b), transpose(K_full));  // [n_heads, seq, total_seq]
    scores = scale(scores, 1.0 / sqrt(head_dim));
    
    // Softmax over key dimension
    scores = softmax(scores, /*dim=*/-1);
    
    // scores @ V
    auto attn_output = matmul(scores, V_full);  // [n_heads, seq, head_dim]
    
    // Reshape and project
    attn_output = reshape_from_heads(attn_output);  // [seq, hidden]
    attn_output = output_projection(attn_output);   // [seq, hidden]
    
    // Copy to output batch
    output->set_batch(b, attn_output);
    
    // Update KV cache
    kv_cache_->append_kv(layer_idx_, b, K_heads->get_batch(b), V_heads->get_batch(b));
}

outputs[0] = output;
return true;
```

**Tasks**:
- [ ] Implement batched Q@K^T computation
- [ ] Implement batched softmax
- [ ] Implement batched scores@V
- [ ] Add causal masking (for prefill)
- [ ] Handle variable sequence lengths
- [ ] Create comprehensive tests
  - [ ] Correctness: Compare batch vs sequential
  - [ ] Causal masking validation
  - [ ] Variable length handling
  - [ ] Performance measurements

**Deliverables**:
- Complete batched attention core
- Test suite (≥8 test cases)
- Performance measurements
- All tests passing

**Time Estimate**: 2 days (14-18 hours)

#### Day 17-18 (Nov 1-2): Batched Attention - Phase 3 (KV Cache)

**Objectives**:
1. Integrate BatchedKVCache with attention
2. Handle prefill (append new KV) and decode (use cached KV)
3. Test multi-step generation

**Tasks**:
- [ ] Integrate `BatchedKVCache::append_kv` in attention
- [ ] Handle prefill mode (append current K/V to cache)
- [ ] Handle decode mode (use full cache, append single token)
- [ ] Add cache size management (max sequence length)
- [ ] Create multi-step generation tests
  - [ ] Test prefill + multiple decode steps
  - [ ] Verify cache grows correctly per sequence
  - [ ] Test cache isolation (no cross-sequence contamination)
  - [ ] Test cache reset between different prompts
- [ ] Performance tests for cached attention

**Deliverables**:
- Complete batched attention with KV cache
- Multi-step generation tests
- All tests passing

**Time Estimate**: 2 days (12-16 hours)

#### Day 19-20 (Nov 3-4): QwenPipeline Parallel Batching

**CRITICAL**: This is the main integration that eliminates the sequential loop

**Objectives**:
1. Replace sequential for-loop with single batch forward pass
2. Integrate all batched operators
3. Handle per-sequence logits extraction

**Current Implementation (Sequential)**:
```cpp
bool QwenPipeline::prefillBatch(...) {
    for (int i = 0; i < batch_size; ++i) {  // ❌ SEQUENTIAL
        prefill(token_batches[i], ...);      // Process one at a time
    }
}
```

**Target Implementation (Parallel)**:
```cpp
bool QwenPipeline::prefillBatch(
    const std::vector<std::vector<int>>& token_batches,
    std::vector<std::vector<float>>& batch_logits_out) 
{
    size_t batch_size = token_batches.size();
    
    // 1. Pad and stack into [batch, max_seq] tensor
    auto padded_tokens = BatchPaddingUtils::pad_and_stack(token_batches);
    size_t max_seq = padded_tokens->shape()[1];
    
    // 2. Embedding: [batch, max_seq] → [batch, max_seq, hidden]
    auto embeddings = embedding_op_->execute(padded_tokens);
    
    // 3. Process through all layers IN BATCH MODE
    auto x = embeddings;
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        // RMSNorm
        auto normed = rms_norm_ops_[layer]->execute(x);
        
        // Attention
        auto attn_out = attention_ops_[layer]->execute(normed);
        
        // Add residual
        x = add(x, attn_out);
        
        // FFN RMSNorm
        normed = ffn_rms_norm_ops_[layer]->execute(x);
        
        // FFN (gate, up, down projections)
        auto ffn_out = ffn_ops_[layer]->execute(normed);
        
        // Add residual
        x = add(x, ffn_out);
    }
    
    // 4. Final norm and projection
    x = final_norm_->execute(x);
    auto logits_full = output_projection_->execute(x);  // [batch, max_seq, vocab_size]
    
    // 5. Extract per-sequence last logits
    batch_logits_out.resize(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        size_t actual_seq_len = token_batches[b].size();
        size_t last_idx = (b * max_seq + actual_seq_len - 1) * vocab_size_;
        
        batch_logits_out[b].resize(vocab_size_);
        std::copy(&logits_full->data()[last_idx],
                  &logits_full->data()[last_idx + vocab_size_],
                  batch_logits_out[b].begin());
    }
    
    return true;
}
```

**Tasks**:
- [ ] Implement padding and stacking utility
- [ ] Replace sequential loop with batch forward pass
- [ ] Integrate all batched operators
- [ ] Handle per-sequence logits extraction
- [ ] Update state management for batch mode
- [ ] Add comprehensive logging
- [ ] Create integration tests
  - [ ] Test batch forward pass correctness
  - [ ] Compare batch output vs sequential (per sequence)
  - [ ] Test variable sequence lengths
  - [ ] Test all batch sizes (1, 4, 8, 16, 32)

**Deliverables**:
- Complete parallel batching in QwenPipeline
- Integration tests
- All tests passing

**Time Estimate**: 2 days (14-18 hours)

#### Day 21 (Nov 5): Comprehensive Testing & Debugging

**Objectives**:
1. Run full test suite
2. Debug any issues
3. Validate correctness across all batch sizes

**Tasks**:
- [ ] Run all unit tests
- [ ] Run integration tests
- [ ] Run batch correctness tests (batch vs sequential)
- [ ] Test edge cases:
  - [ ] batch=1 (backward compatibility)
  - [ ] Max batch size
  - [ ] Variable sequence lengths
  - [ ] Empty sequences
  - [ ] Very long sequences
- [ ] Memory profiling (check for leaks)
- [ ] Debug any failing tests
- [ ] Fix issues found

**Deliverables**:
- All tests passing
- No memory leaks
- Clean test output

**Time Estimate**: 1 day (8-10 hours)

### Week 4: Performance Validation & Finalization (Nov 6-12)

#### Day 22-23 (Nov 6-7): Performance Benchmarking

**Objectives**:
1. Measure actual throughput improvements
2. Verify 22× speedup target achieved
3. Profile any remaining bottlenecks

**Tasks**:
- [ ] Run `test_batch_performance` with new implementation
- [ ] Measure throughput for all batch sizes:
  - [ ] batch=1: Baseline (~10 tok/s)
  - [ ] batch=2: Target ~20 tok/s (2×)
  - [ ] batch=4: Target ~40 tok/s (4×)
  - [ ] batch=8: Target ~80 tok/s (8×)
  - [ ] batch=16: Target ~160 tok/s (16×)
  - [ ] batch=32: Target ≥220 tok/s (22×)
- [ ] Calculate actual speedup ratios
- [ ] Profile with `perf` if speedup < target:
  - [ ] Identify hotspots
  - [ ] Check for unexpected synchronization
  - [ ] Verify parallel execution
- [ ] Document results in performance report

**Success Criteria**:
- ✅ batch=32 achieves ≥220 tok/s (22× speedup)
- ✅ Linear scaling observed (batch=4 → 4×, batch=8 → 8×, etc.)
- ✅ No obvious bottlenecks in profiling

**Deliverables**:
- Performance measurements table
- Speedup analysis
- Profiling results (if needed)
- Performance report document

**Time Estimate**: 2 days (10-14 hours)

#### Day 24-25 (Nov 8-9): Optimization (if needed)

**Objectives**:
1. Address any performance bottlenecks
2. Optimize critical paths
3. Re-benchmark after optimizations

**Potential Optimizations**:
- Memory layout optimization (reduce copies)
- OpenMP tuning (thread count, scheduling)
- BLAS library tuning (thread affinity)
- Operator fusion (combine small operations)

**Tasks**:
- [ ] Analyze profiling results from Day 22-23
- [ ] Implement targeted optimizations
- [ ] Re-run benchmarks
- [ ] Verify improvements
- [ ] Document optimization impact

**Note**: This is contingent on Day 22-23 results. If 22× target is met, skip to Day 26.

**Deliverables**:
- Optimizations implemented (if needed)
- Updated performance measurements
- Optimization impact analysis

**Time Estimate**: 0-2 days (conditional)

#### Day 26-27 (Nov 10-11): Code Cleanup & Documentation

**Objectives**:
1. Remove all debug instrumentation
2. Add comprehensive code comments
3. Update documentation

**Tasks**:
- [ ] Remove debug logging and instrumentation
- [ ] Add Doxygen comments to all new/modified methods
- [ ] Update operator API documentation
- [ ] Create performance benchmark report
- [ ] Update changelog with implementation summary
- [ ] Update README with batching capabilities
- [ ] Code review and cleanup
  - [ ] Remove dead code
  - [ ] Consistent naming
  - [ ] Proper error handling

**Deliverables**:
- Clean, production-ready code
- Comprehensive documentation
- Performance report
- Updated changelog

**Time Estimate**: 2 days (12-16 hours)

#### Day 28 (Nov 12): Final Validation

**Objectives**:
1. Run complete test suite
2. Final performance verification
3. Mark Phase complete

**Tasks**:
- [ ] Run smoke tests (5s)
- [ ] Run unit tests (2m30s)
- [ ] Run integration tests (3m)
- [ ] Run parity tests (if applicable)
- [ ] Final performance benchmark
- [ ] Verify all success criteria met
- [ ] Update TODO.md to mark Option A complete
- [ ] Create completion summary document

**Success Criteria Checklist**:
- ✅ All tests passing (smoke, unit, integration)
- ✅ Performance: batch=32 ≥220 tok/s (22× speedup)
- ✅ Correctness: Batch output matches sequential
- ✅ Code quality: Clean, documented, production-ready
- ✅ No memory leaks
- ✅ Documentation complete

**Deliverables**:
- Final test results
- Performance verification
- Completion summary
- Updated TODO.md

**Time Estimate**: 1 day (4-6 hours)

## Risk Assessment & Mitigation

### High-Risk Items

1. **Batched Attention Complexity**
   - Risk: Multi-head attention with batched KV cache is complex
   - Mitigation: Incremental implementation, extensive testing
   - Contingency: 2-3 extra days budgeted

2. **Performance Target Not Met**
   - Risk: May not achieve full 22× speedup
   - Mitigation: Profile early, optimize incrementally
   - Contingency: Days 24-25 dedicated to optimization

3. **Memory Issues**
   - Risk: Batched tensors may exceed memory limits
   - Mitigation: Conservative batch sizes, memory profiling
   - Contingency: Add memory checks, reduce max batch size

### Medium-Risk Items

1. **Interface Changes Breaking Existing Code**
   - Risk: Backward compatibility issues
   - Mitigation: Maintain batch=1 as special case
   - Contingency: Add compatibility layer

2. **Variable Sequence Length Handling**
   - Risk: Padding/masking bugs
   - Mitigation: Comprehensive edge case testing
   - Contingency: Extra debug time budgeted

## Success Metrics

### Performance Metrics
| Batch Size | Target Throughput | Target Speedup | Status |
|------------|-------------------|----------------|--------|
| 1          | ~10 tok/s         | 1.0×           | -      |
| 2          | ~20 tok/s         | 2.0×           | -      |
| 4          | ~40 tok/s         | 4.0×           | -      |
| 8          | ~80 tok/s         | 8.0×           | -      |
| 16         | ~160 tok/s        | 16.0×          | -      |
| 32         | ≥220 tok/s        | ≥22.0×         | -      |

### Correctness Metrics
- ✅ All unit tests passing
- ✅ Batch vs sequential logits match (tolerance: 1e-4)
- ✅ KV cache isolation verified
- ✅ Variable length handling correct

### Code Quality Metrics
- ✅ Comprehensive Doxygen comments
- ✅ No memory leaks (valgrind clean)
- ✅ Clean code review
- ✅ Documentation complete

## Progress Tracking

**Week 1**: [ ] Day 1 [ ] Day 2 [ ] Day 3 [ ] Day 4 [ ] Day 5 [ ] Day 6 [ ] Day 7  
**Week 2**: [ ] Day 8 [ ] Day 9 [ ] Day 10 [ ] Day 11 [ ] Day 12 [ ] Day 13 [ ] Day 14  
**Week 3**: [ ] Day 15 [ ] Day 16 [ ] Day 17 [ ] Day 18 [ ] Day 19 [ ] Day 20 [ ] Day 21  
**Week 4**: [ ] Day 22 [ ] Day 23 [ ] Day 24 [ ] Day 25 [ ] Day 26 [ ] Day 27 [ ] Day 28

**Overall Status**: 🔄 Day 1/28 (4% complete)

## Next Steps

**Immediate** (Today, Oct 15):
1. Complete Day 1 tasks (this document ✅)
2. Audit operator interfaces
3. Create risk assessment document

**Tomorrow** (Oct 16):
1. Begin SimpleTensor batch dimension support
2. Implement batch indexing utilities
3. Start unit test creation

## References

- Investigation findings: `changelog/2025-10-15-INVESTIGATION-RESULTS-sequential-batching-proven.md`
- Design document: `changelog/2025-10-15-parallel-batching-design.md`
- Baseline code: `src/QwenPipeline.cpp` (lines 2090-2142)
- Test suite: `tests/test_batch_performance.cpp`

---

**Status**: Day 1 Complete ✅  
**Next**: Day 2 - SimpleTensor Batch Dimension Support  
**Updated**: October 15, 2025
