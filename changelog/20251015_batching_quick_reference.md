# Parallel Batching Quick Reference

## Tensor Shape Transformations

### Current (Single Sequence)
```
embedSingleToken(token_id)
  └─> [1, d_model]

prefill(tokens: [seq_len])
  └─> embeddings: [seq_len, d_model]
  └─> Q,K,V: [n_heads, seq_len, head_dim]
  └─> attention: [n_heads, seq_len, seq_len]
  └─> logits: [seq_len, vocab_size]
```

### Batched (Multiple Sequences)
```
embedBatch(token_ids: [batch, seq_len])
  └─> [batch, seq_len, d_model]

prefillBatch(tokens: [batch, seq_len])
  └─> embeddings: [batch, seq_len, d_model]
  └─> Q,K,V: [batch, n_heads, seq_len, head_dim]
  └─> attention: [batch, n_heads, seq_len, seq_len]
  └─> logits: [batch, seq_len, vocab_size]
```

## Key Code Changes

### 1. Pipeline Interface (AbstractPipeline.h)
```cpp
// ADD these new methods:
virtual bool prefillBatch(
    const std::vector<std::vector<int>>& token_batches,
    std::shared_ptr<TensorBase>& out_logits) = 0;

virtual bool decodeBatch(
    const std::vector<int>& next_tokens,
    std::shared_ptr<TensorBase>& out_logits) = 0;

virtual void resetBatch(size_t batch_size) = 0;
```

### 2. QwenPipeline State (QwenPipeline.h)
```cpp
// CHANGE from single to batch state:
// int n_past_;  // OLD
std::vector<int> n_past_batch_;           // NEW: [batch_size]
size_t batch_size_;                        // NEW: current batch size

// KV cache shape changes:
// [n_heads, max_seq, head_dim]          // OLD per layer
// [batch, n_heads, max_seq, head_dim]   // NEW per layer
```

### 3. Embedding (EmbeddingKernel)
```cpp
// BEFORE:
for (size_t t = 0; t < seq_len; ++t) {
    embed_token(tokens[t], output + t * d_model);
}

// AFTER:
#pragma omp parallel for
for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < seq_lens[b]; ++t) {
        embed_token(tokens[b][t], output + (b*max_len + t)*d_model);
    }
}
```

### 4. Linear Operator (MPILinearOperator)
```cpp
// NO MAJOR CHANGES - just reshape!
// [batch, seq_len, d_in] -> flatten -> [batch*seq_len, d_in]
// Matmul: [batch*seq_len, d_in] × [d_out, d_in]^T = [batch*seq_len, d_out]
// Reshape: [batch*seq_len, d_out] -> [batch, seq_len, d_out]

auto flat_input = input->reshape({batch * seq_len, d_in});
performMatmul(flat_input, weight, flat_output);
output = flat_output->reshape({batch, seq_len, d_out});
```

### 5. Attention Masking
```cpp
// ADD padding mask on top of causal mask:
void applyMasks(scores, positions, actual_lengths) {
    for (int b = 0; b < batch; ++b) {
        // Causal mask
        for (int i = 0; i < seq_len; ++i) {
            for (int j = i + 1; j < cache_len; ++j) {
                scores[b][i][j] = -INF;
            }
        }
        
        // Padding mask (NEW)
        for (int i = 0; i < seq_len; ++i) {
            for (int j = actual_lengths[b]; j < cache_len; ++j) {
                scores[b][i][j] = -INF;  // Mask padding
            }
        }
    }
}
```

## File Modification Checklist

### Core Files to Modify
- [ ] `src/AbstractPipeline.h` - Add batch methods to interface
- [ ] `src/QwenPipeline.h` - Add batch state members
- [ ] `src/QwenPipeline.cpp` - Implement prefillBatch/decodeBatch
- [ ] `src/tensors/SimpleTensor.h` - Add reshape(), batch_size() accessors
- [ ] `src/kernels/EmbeddingKernel.cpp` - Parallelize across batch
- [ ] `src/operators/MPILinearOperator.cpp` - Add reshape logic
- [ ] `src/kernels/MPIAttentionKernel.cpp` - Add padding mask support
- [ ] `src/kernels/MPIRMSNormKernel.cpp` - Parallelize across batch
- [ ] `src/BenchmarkRunner.{h,cpp}` - Add batch benchmark mode
- [ ] `src/ArgumentParser.{h,cpp}` - Already has --batch-size!

### New Files to Create
- [ ] `src/BatchPaddingUtils.h` - Padding/stacking utilities
- [ ] `src/BatchPaddingUtils.cpp` - Implementation
- [ ] `tests/test_batch_correctness.cpp` - Verify batch=2 vs 2×batch=1
- [ ] `tests/test_batch_performance.cpp` - Benchmark scaling

## Testing Strategy

### 1. Correctness Test
```cpp
// Run same sequence twice:
// Method A: batch_size=1, run twice sequentially
auto result1_seq = run_single(prompt);
auto result2_seq = run_single(prompt);

// Method B: batch_size=2, run once
auto results_batch = run_batch({prompt, prompt});

// Verify: results_batch[0] == result1_seq
//         results_batch[1] == result2_seq
REQUIRE(allclose(results_batch[0], result1_seq, rtol=1e-5));
REQUIRE(allclose(results_batch[1], result2_seq, rtol=1e-5));
```

### 2. Performance Test
```bash
# Baseline
./run_llaminar.sh -- --benchmark --batch-size 1 -p "test" -n 50
# Expected: ~13 tok/s

# Batch of 4
./run_llaminar.sh -- --benchmark --batch-size 4 \
  -p "prompt1" -p "prompt2" -p "prompt3" -p "prompt4" -n 50
# Expected: ~48 tok/s aggregate (3.7× speedup)

# Batch of 32
./run_llaminar.sh -- --benchmark --batch-size 32 -p "test" -n 50
# Expected: ~288 tok/s aggregate (22× speedup)
```

## Performance Expectations

| Batch Size | Aggregate tok/s | Per-seq tok/s | Bandwidth | Speedup |
|------------|-----------------|---------------|-----------|---------|
| 1          | 13              | 13            | 1.5 GB/s  | 1.0×    |
| 2          | 24              | 12            | 3.0 GB/s  | 1.8×    |
| 4          | 48              | 12            | 6.0 GB/s  | 3.7×    |
| 8          | 88              | 11            | 11 GB/s   | 6.8×    |
| 16         | 160             | 10            | 20 GB/s   | 12.3×   |
| 32         | 288             | 9             | 36 GB/s   | 22×     |
| 64         | 512             | 8             | 64 GB/s   | 39×     |

**Note**: Per-sequence throughput drops slightly due to:
- Padding overhead (wasted compute on pad tokens)
- Cache contention (batch shares CPU cache)
- Memory bandwidth limits start to appear

## Common Pitfalls

### ❌ DON'T: Distribute batch across MPI ranks
```cpp
// Rank 0: batch[0:16]
// Rank 1: batch[16:32]
// Problem: Complex synchronization, load imbalance
```

### ✅ DO: Keep batch local, distribute per-sequence computation
```cpp
// Both ranks process full batch[0:32]
// Distribute tensor parallelism within each sequence
// Simpler, better load balance
```

### ❌ DON'T: Ignore padding in attention
```cpp
// This attends to padding tokens!
scores = Q @ K.T  // Wrong if sequences have different lengths
```

### ✅ DO: Apply padding mask
```cpp
scores = Q @ K.T
scores = apply_causal_mask(scores)
scores = apply_padding_mask(scores, actual_lengths)  // Critical!
attn = softmax(scores)
```

### ❌ DON'T: Assume batch dimension is distributed
```cpp
size_t my_batch_start = rank * batch_size / world_size;  // Wrong!
```

### ✅ DO: Keep batch local
```cpp
// All ranks process same batch
// Distribution is per-sequence (tensor parallelism)
for (size_t b = 0; b < batch_size; ++b) {
    process_sequence_distributed(b);  // This uses MPI
}
```

## Memory Requirements

### KV Cache Formula
```
Single sequence: n_layers × n_heads × max_seq × head_dim × 4 bytes
Qwen-0.5B:       24 × 14 × 2048 × 64 × 4 = 176 MB

Batch size B:    B × 176 MB
Batch=32:        5.6 GB
Batch=128:       22.5 GB
Batch=512:       90 GB
```

### Your System (768GB RAM)
```
Available for KV cache: ~400 GB (leaving 368 GB for weights/activations)
Max batch size (2K context): 400 GB / 176 MB = ~2,270 sequences
Max batch size (8K context): 400 GB / 704 MB = ~568 sequences
```

**Recommendation**: Start with batch_size=32 (uses 5.6 GB), then scale up based on profiling.

## Next Steps

1. **Start simple**: Implement batch_size=2 first
2. **Test correctness**: Verify batch[0] == sequential run
3. **Measure performance**: Does batch=2 give ~1.8× aggregate speedup?
4. **Scale up**: Try batch=4, 8, 16, 32
5. **Optimize**: Add bucketing, kernel fusion, better padding strategy
6. **Production**: Handle variable-length sequences efficiently
