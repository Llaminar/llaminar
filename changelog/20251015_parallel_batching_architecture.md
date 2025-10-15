# Parallel Batching Architecture Design - October 15, 2025

## Overview

This document outlines the architectural changes needed to implement true parallel batching in Llaminar, enabling processing of multiple independent sequences simultaneously to leverage the available memory bandwidth (281 GB/s theoretical peak).

## Current Architecture Limitations

### Single-Sequence Design
```
Current flow (batch_size = 1):
Token IDs: [seq_len]                    e.g., [8]
Embeddings: [seq_len, d_model]          e.g., [8, 896]
Hidden states: [seq_len, d_model]       e.g., [8, 896]
Attention: [n_heads, seq_len, seq_len]  e.g., [14, 8, 8]
Logits: [seq_len, vocab_size]           e.g., [8, 151936]
```

**Problems**:
- Pipeline state tied to single sequence (n_past_, KV cache)
- No batch dimension in tensor shapes
- Sequential token generation within one sequence
- Can't process multiple sequences simultaneously

## Target Architecture: Parallel Batching

### Batched Tensor Shapes
```
Batched flow (batch_size = B):
Token IDs: [batch, seq_len]                    e.g., [4, 8]
Embeddings: [batch, seq_len, d_model]          e.g., [4, 8, 896]
Hidden states: [batch, seq_len, d_model]       e.g., [4, 8, 896]
Attention: [batch, n_heads, seq_len, seq_len]  e.g., [4, 14, 8, 8]
KV cache: [batch, n_layers, n_heads, max_seq, head_dim]
Logits: [batch, seq_len, vocab_size]           e.g., [4, 8, 151936]
```

**Benefits**:
- Process B independent sequences in one forward pass
- Amortize memory latency over larger transfers
- Better CPU cache utilization
- Aggregate throughput: B × single_throughput

## Architectural Changes Required

### 1. Tensor System Updates

#### SimpleTensor Enhancement
**File**: `src/tensors/SimpleTensor.h`

Add batch dimension support:
```cpp
class SimpleTensor : public TensorBase {
private:
    std::vector<size_t> shape_;     // Now: [batch, seq_len, d_model] or [batch, ...]
    std::vector<float> data_;
    
public:
    // Current: shape is [seq_len, d_model]
    // New: shape is [batch, seq_len, d_model]
    
    // Batch-aware accessors
    size_t batch_size() const { return shape_.size() >= 1 ? shape_[0] : 1; }
    size_t seq_len() const { return shape_.size() >= 2 ? shape_[1] : shape_[0]; }
    
    // Slice batch: Extract single sequence from batch
    std::shared_ptr<SimpleTensor> slice_batch(size_t batch_idx) const;
    
    // Stack batches: Combine multiple sequences into batch tensor
    static std::shared_ptr<SimpleTensor> stack_batch(
        const std::vector<std::shared_ptr<SimpleTensor>>& sequences);
};
```

**Changes needed**:
- Update constructors to handle 3D+ shapes
- Add batch slicing/stacking utilities
- Ensure NUMA first-touch works with batched allocations
- Update size() to return total elements (batch × seq_len × d_model)

#### COSMATensor Enhancement
**File**: `src/tensors/COSMATensor.h`

Support distributed batched operations:
```cpp
class COSMATensor : public TensorBase {
    // Batch dimension distributed across MPI ranks or local
    // Strategy: Keep batch local, distribute other dimensions (seq_len, d_model)
    
    // Current: [seq_len, d_model] distributed
    // New: [batch, seq_len, d_model] where batch is local, seq_len can be distributed
};
```

**Design decision**: Keep batch dimension **local** (not distributed)
- Each MPI rank processes same batch
- Distribute computation within each sequence
- Simpler than distributing batch across ranks

---

### 2. Pipeline Architecture Changes

#### AbstractPipeline Interface
**File**: `src/AbstractPipeline.h`

Add batch-aware methods:
```cpp
class AbstractPipeline {
public:
    // NEW: Batch prefill - process multiple prompts at once
    virtual bool prefillBatch(
        const std::vector<std::vector<int>>& token_batches,  // [batch_size][seq_len_i]
        std::shared_ptr<TensorBase>& out_logits_batch        // [batch, max_seq, vocab]
    ) = 0;
    
    // NEW: Batch decode - generate next token for each sequence in batch
    virtual bool decodeBatch(
        const std::vector<int>& next_tokens,                 // [batch_size]
        std::shared_ptr<TensorBase>& out_logits_batch        // [batch, 1, vocab]
    ) = 0;
    
    // KEEP: Single-sequence methods for compatibility
    virtual bool prefill(const std::vector<int>& tokens) = 0;
    virtual bool decode(int next_token) = 0;
    
    // NEW: Batch state management
    virtual void resetBatch(size_t batch_size) = 0;
    virtual size_t currentBatchSize() const = 0;
};
```

#### QwenPipeline Batch Support
**File**: `src/QwenPipeline.h`, `src/QwenPipeline.cpp`

**State Changes**:
```cpp
class QwenPipeline : public AbstractPipeline {
private:
    // OLD: Single sequence state
    // int n_past_;  // Current position in sequence
    
    // NEW: Batch state
    size_t batch_size_;                    // Current batch size
    std::vector<int> n_past_batch_;        // Position per sequence [batch_size]
    
    // OLD: KV cache per layer: [n_heads, max_seq, head_dim]
    // NEW: KV cache per layer: [batch, n_heads, max_seq, head_dim]
    std::vector<std::shared_ptr<TensorBase>> k_cache_batch_;  // Per layer
    std::vector<std::shared_ptr<TensorBase>> v_cache_batch_;  // Per layer
    
public:
    bool prefillBatch(
        const std::vector<std::vector<int>>& token_batches,
        std::shared_ptr<TensorBase>& out_logits_batch) override;
    
    bool decodeBatch(
        const std::vector<int>& next_tokens,
        std::shared_ptr<TensorBase>& out_logits_batch) override;
    
    void resetBatch(size_t batch_size) override {
        batch_size_ = batch_size;
        n_past_batch_.assign(batch_size, 0);
        // Resize KV caches to [batch, n_heads, max_seq, head_dim]
        allocateKVCacheBatch();
    }
};
```

**Key Implementation**:
```cpp
bool QwenPipeline::prefillBatch(
    const std::vector<std::vector<int>>& token_batches,
    std::shared_ptr<TensorBase>& out_logits_batch)
{
    // 1. Pad sequences to same length (required for batching)
    size_t max_len = 0;
    for (const auto& tokens : token_batches) {
        max_len = std::max(max_len, tokens.size());
    }
    
    // 2. Create padded token tensor [batch, max_len]
    auto token_tensor = createPaddedTokenTensor(token_batches, max_len);
    
    // 3. Embed all sequences: [batch, max_len] -> [batch, max_len, d_model]
    auto embeddings = embedBatch(token_tensor);
    
    // 4. Process through transformer layers
    auto hidden = embeddings;
    for (size_t layer = 0; layer < n_layers_; ++layer) {
        hidden = executeTransformerLayerBatch(layer, hidden, max_len);
    }
    
    // 5. Final layer norm + projection to logits
    hidden = rmsNormBatch(hidden, weights_.ln_f_weight);
    out_logits_batch = projectToVocabBatch(hidden);
    
    // 6. Update positions for each sequence
    for (size_t b = 0; b < batch_size_; ++b) {
        n_past_batch_[b] += token_batches[b].size();
    }
    
    return true;
}
```

---

### 3. Operator/Kernel Updates

#### Embedding Kernel
**File**: `src/kernels/EmbeddingKernel.h`

```cpp
class EmbeddingKernel : public MPIKernelBase {
public:
    // NEW: Batch embedding
    bool executeBatch(
        const std::vector<std::vector<int>>& token_batches,  // [batch][seq_len_i]
        const float* embedding_weights,                       // [vocab, d_model]
        std::shared_ptr<TensorBase>& output                  // [batch, max_seq, d_model]
    );
    
    // Implementation:
    // - Parallelize across batch dimension with OpenMP
    // - Each thread processes one sequence
    // - Pad shorter sequences with zeros
};
```

**Implementation strategy**:
```cpp
#pragma omp parallel for
for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < token_batches[b].size(); ++t) {
        int token_id = token_batches[b][t];
        float* out_ptr = output->data() + b * max_seq * d_model + t * d_model;
        const float* emb_ptr = embedding_weights + token_id * d_model;
        std::memcpy(out_ptr, emb_ptr, d_model * sizeof(float));
    }
    // Remaining positions already zero-initialized (padding)
}
```

#### Linear Operator (Matmul)
**File**: `src/operators/MPILinearOperator.cpp`

**Good news**: Matrix multiplication naturally handles batching!

```cpp
// Current: [seq_len, d_model] × [d_model, d_out] = [seq_len, d_out]
// Batched: [batch×seq_len, d_model] × [d_model, d_out] = [batch×seq_len, d_out]

// Simply reshape input:
// From: [batch, seq_len, d_model]
// To: [batch*seq_len, d_model]  (flatten batch and seq_len)
// Output: [batch*seq_len, d_out]
// Reshape back to: [batch, seq_len, d_out]
```

**Minimal changes needed**:
```cpp
bool MPILinearOperator::executeBatch(
    const std::shared_ptr<TensorBase>& input,   // [batch, seq_len, d_in]
    const std::shared_ptr<TensorBase>& weight,  // [d_out, d_in]
    std::shared_ptr<TensorBase>& output)        // [batch, seq_len, d_out]
{
    // Flatten batch dimension into seq_len
    size_t batch = input->shape()[0];
    size_t seq_len = input->shape()[1];
    size_t d_in = input->shape()[2];
    size_t d_out = weight->shape()[0];
    
    // Treat as [batch*seq_len, d_in] × [d_out, d_in]^T = [batch*seq_len, d_out]
    auto flat_input = input->reshape({batch * seq_len, d_in});
    auto flat_output = TensorFactory::create_simple({batch * seq_len, d_out});
    
    // Use existing matmul logic
    performMatmul(flat_input, weight, flat_output);
    
    // Reshape output back to [batch, seq_len, d_out]
    output = flat_output->reshape({batch, seq_len, d_out});
    return true;
}
```

#### Attention Kernel
**File**: `src/kernels/MPIAttentionKernel.h`

**Most complex change** - needs batch-aware masking:

```cpp
class MPIAttentionKernel : public MPIKernelBase {
public:
    bool executeBatch(
        const std::shared_ptr<TensorBase>& hidden,          // [batch, seq_len, d_model]
        const std::shared_ptr<TensorBase>& k_cache_batch,   // [batch, n_heads, cache_len, head_dim]
        const std::shared_ptr<TensorBase>& v_cache_batch,   // [batch, n_heads, cache_len, head_dim]
        const std::vector<int>& positions_batch,             // [batch] - position per sequence
        const std::vector<int>& seq_lengths_batch,           // [batch] - actual length (for masking)
        std::shared_ptr<TensorBase>& output                 // [batch, seq_len, d_model]
    );
};
```

**Key implementation details**:
```cpp
// For each sequence in batch:
#pragma omp parallel for
for (size_t b = 0; b < batch_size; ++b) {
    // 1. Project Q, K, V for this sequence
    auto Q_b = projectQuery(hidden, b);   // [seq_len, d_model] -> [n_heads, seq_len, head_dim]
    auto K_b = projectKey(hidden, b);
    auto V_b = projectValue(hidden, b);
    
    // 2. Update KV cache for this sequence
    updateKVCache(k_cache_batch, v_cache_batch, b, K_b, V_b, positions_batch[b]);
    
    // 3. Compute attention scores: Q × K^T
    auto scores_b = matmul(Q_b, K_b.transpose());  // [n_heads, seq_len, cache_len]
    
    // 4. Apply causal + padding mask
    applyCausalMask(scores_b, positions_batch[b]);
    applyPaddingMask(scores_b, seq_lengths_batch[b]);  // NEW for batching
    
    // 5. Softmax + weighted sum with V
    auto attn_b = softmax(scores_b);
    auto out_b = matmul(attn_b, V_b);
    
    // 6. Output projection
    output_b = projectOutput(out_b);
    
    // 7. Write to output batch tensor
    copyToOutputBatch(output, b, output_b);
}
```

**Padding mask**:
```cpp
void applyPaddingMask(
    std::shared_ptr<TensorBase>& scores,  // [n_heads, seq_len, cache_len]
    int actual_seq_len)                    // Actual length (rest is padding)
{
    // Mask out attention to padding positions
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = actual_seq_len; j < cache_len; ++j) {
                scores->at({h, i, j}) = -INFINITY;  // Mask padding
            }
        }
    }
}
```

#### RMSNorm Kernel
**File**: `src/kernels/MPIRMSNormKernel.h`

**Simple parallelization** - independent per batch:

```cpp
bool MPIRMSNormKernel::executeBatch(
    const std::shared_ptr<TensorBase>& input,   // [batch, seq_len, d_model]
    const std::shared_ptr<TensorBase>& weight,  // [d_model]
    std::shared_ptr<TensorBase>& output)        // [batch, seq_len, d_model]
{
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            // Apply RMSNorm independently to each position
            applyRMSNorm(input, weight, output, b, t);
        }
    }
    return true;
}
```

---

### 4. Benchmark Runner Updates

#### Batch Benchmark Mode
**File**: `src/BenchmarkRunner.h`, `src/BenchmarkRunner.cpp`

```cpp
struct BatchBenchmarkMetrics {
    size_t batch_size;
    size_t total_sequences;
    
    // Prefill metrics
    size_t total_prefill_tokens;
    double prefill_time_ms;
    double prefill_throughput;  // Total tokens/s across all sequences
    
    // Decode metrics
    size_t total_decode_tokens;
    double decode_time_ms;
    double decode_throughput;
    
    // Efficiency
    double batching_efficiency;  // actual_throughput / (batch_size × single_throughput)
    double memory_bandwidth_used;  // Estimated GB/s
};

BatchBenchmarkMetrics runBatchBenchmark(
    AbstractPipeline& pipeline,
    const QwenModelWeights& weights,
    chat::TokenizerInterface& tokenizer,
    const std::vector<std::string>& prompts,  // batch_size prompts
    int n_predict);
```

**Implementation**:
```cpp
BatchBenchmarkMetrics runBatchBenchmark(...) {
    // 1. Tokenize all prompts
    std::vector<std::vector<int>> token_batches;
    for (const auto& prompt : prompts) {
        token_batches.push_back(tokenizer.tokenize(prompt));
    }
    
    // 2. Batch prefill
    auto t0 = now();
    std::shared_ptr<TensorBase> logits_batch;
    pipeline.prefillBatch(token_batches, logits_batch);
    auto t1 = now();
    
    // 3. Batch decode loop
    std::vector<int> generated_tokens(batch_size);
    auto t2 = now();
    for (int step = 0; step < n_predict; ++step) {
        // Sample next token for each sequence
        for (size_t b = 0; b < batch_size; ++b) {
            generated_tokens[b] = greedySample(logits_batch, b);
        }
        
        // Decode all sequences together
        pipeline.decodeBatch(generated_tokens, logits_batch);
    }
    auto t3 = now();
    
    // 4. Calculate metrics
    metrics.prefill_time_ms = duration(t0, t1);
    metrics.decode_time_ms = duration(t2, t3);
    metrics.total_prefill_tokens = sum(token_batches.sizes());
    metrics.total_decode_tokens = batch_size * n_predict;
    metrics.prefill_throughput = metrics.total_prefill_tokens / (metrics.prefill_time_ms / 1000.0);
    metrics.decode_throughput = metrics.total_decode_tokens / (metrics.decode_time_ms / 1000.0);
    
    return metrics;
}
```

---

### 5. Padding Strategy

**Critical for batching**: Sequences in a batch must have same length for efficient tensor operations.

#### Dynamic Padding
**File**: `src/BatchPaddingUtils.h` (new)

```cpp
struct PaddedBatch {
    std::shared_ptr<TensorBase> tokens;       // [batch, max_len]
    std::vector<int> actual_lengths;           // [batch] - original lengths
    std::vector<int> padding_mask;             // [batch, max_len] - 1=real, 0=padding
    size_t max_length;
};

PaddedBatch createPaddedBatch(
    const std::vector<std::vector<int>>& token_sequences,
    int pad_token_id = 0)
{
    // 1. Find max length
    size_t max_len = 0;
    for (const auto& seq : token_sequences) {
        max_len = std::max(max_len, seq.size());
    }
    
    // 2. Create padded tensor
    size_t batch_size = token_sequences.size();
    auto padded = TensorFactory::create_simple({batch_size, max_len});
    std::fill(padded->data(), padded->data() + padded->size(), pad_token_id);
    
    // 3. Copy actual tokens
    std::vector<int> actual_lengths(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        actual_lengths[b] = token_sequences[b].size();
        std::memcpy(
            padded->data() + b * max_len,
            token_sequences[b].data(),
            token_sequences[b].size() * sizeof(float));
    }
    
    return {padded, actual_lengths, createPaddingMask(actual_lengths, max_len), max_len};
}
```

#### Bucketing Strategy (Advanced)
Group sequences by similar lengths to minimize padding waste:

```cpp
// Instead of one batch [varying lengths]
// Create multiple batches [similar lengths each]
auto batches = bucketSequencesByLength(prompts, batch_size);
// Bucket 1: lengths 5-8    -> max_len=8, minimal padding
// Bucket 2: lengths 15-20  -> max_len=20, minimal padding
// Bucket 3: lengths 50-64  -> max_len=64, minimal padding
```

---

### 6. Memory Management

#### KV Cache Sizing
**Challenge**: Batch dimension increases KV cache size significantly

```cpp
// Single sequence: [n_layers, n_heads, max_seq, head_dim]
// Qwen-0.5B: 24 layers × 14 heads × 2048 max_seq × 64 head_dim × 4 bytes
// = 24 × 14 × 2048 × 64 × 4 = 176 MB per sequence

// Batch of 32:
// = 32 × 176 MB = 5.6 GB KV cache

// With 768GB RAM, can support batch_size up to ~4000 (limited by other factors)
```

**Strategy**: Pre-allocate KV cache for max batch size
```cpp
void QwenPipeline::allocateKVCacheBatch() {
    size_t cache_size = batch_size_ * n_layers_ * n_heads_ * max_seq_len_ * head_dim_;
    
    // Pre-allocate with NUMA first-touch
    for (size_t layer = 0; layer < n_layers_; ++layer) {
        k_cache_batch_[layer] = TensorFactory::create_simple(
            {batch_size_, n_heads_, max_seq_len_, head_dim_});
        v_cache_batch_[layer] = TensorFactory::create_simple(
            {batch_size_, n_heads_, max_seq_len_, head_dim_});
    }
}
```

---

### 7. MPI Distribution Strategy

**Design decision**: Keep batch local, distribute computation per sequence

```
Rank 0 processes: batch[0:32], all handle same sequences
Rank 1 processes: batch[0:32], all handle same sequences

Distribution happens within each sequence:
- Tensor parallelism: shard weights/activations across ranks
- Each rank computes part of each sequence
- AllReduce to combine results
```

**Alternative** (more complex): Distribute batch across ranks
```
Rank 0 processes: batch[0:16]
Rank 1 processes: batch[16:32]

Pros: 2× more sequences
Cons: Complicated synchronization, imbalanced loads
```

**Recommendation**: Start with local batch, distributed per-sequence

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Update SimpleTensor for batch dimensions
- [ ] Add batch accessors and utilities
- [ ] Implement padding/stacking functions
- [ ] Update AbstractPipeline interface

### Phase 2: Core Kernels (Week 2)
- [ ] Batch embedding kernel
- [ ] Batch linear operator (reshape strategy)
- [ ] Batch RMSNorm kernel
- [ ] Batch attention kernel (with padding masks)

### Phase 3: Pipeline Integration (Week 3)
- [ ] Implement QwenPipeline::prefillBatch()
- [ ] Implement QwenPipeline::decodeBatch()
- [ ] KV cache batch management
- [ ] State tracking per sequence

### Phase 4: Benchmarking (Week 4)
- [ ] Batch benchmark runner
- [ ] Correctness tests (batch vs sequential)
- [ ] Performance measurement
- [ ] Memory bandwidth utilization analysis

### Phase 5: Optimization (Week 5)
- [ ] Bucketing by sequence length
- [ ] Kernel fusion opportunities
- [ ] Memory layout optimization
- [ ] Profiling and tuning

---

## Expected Performance

### Batch Size Scaling
```
Batch=1:   13 tok/s aggregate,  13 tok/s per-seq,  1.5 GB/s bandwidth
Batch=4:   48 tok/s aggregate,  12 tok/s per-seq,  6.0 GB/s bandwidth  (3.7× improvement)
Batch=8:   88 tok/s aggregate,  11 tok/s per-seq, 11.0 GB/s bandwidth  (6.8× improvement)
Batch=16: 160 tok/s aggregate,  10 tok/s per-seq, 20.0 GB/s bandwidth (12.3× improvement)
Batch=32: 288 tok/s aggregate,   9 tok/s per-seq, 36.0 GB/s bandwidth (22× improvement)
```

**Note**: Per-sequence throughput decreases slightly due to:
- Padding overhead
- Cache contention
- OpenMP thread competition

But **aggregate throughput** scales nearly linearly!

### Memory Bandwidth Utilization
```
Batch=1:   1.5 GB/s  (0.5% of 281 GB/s peak)
Batch=32:  36 GB/s   (13% of 281 GB/s peak)
Batch=64:  65 GB/s   (23% of 281 GB/s peak)
Batch=128: 115 GB/s  (41% of 281 GB/s peak)
```

Even at batch=128, still only using 41% of theoretical peak due to sequential dependencies within each sequence.

---

## Conclusion

Parallel batching transforms Llaminar from processing one sequence at 13 tok/s to processing 32 sequences at 9-10 tok/s each = **288-320 tok/s aggregate throughput**.

**Key insight**: You're not making single-sequence generation faster (still limited by sequential dependencies), but you're processing many sequences simultaneously to leverage your massive memory bandwidth.

This is exactly what production LLM serving systems (vLLM, TensorRT-LLM, Text-Generation-Inference) do to achieve high throughput on expensive hardware.
