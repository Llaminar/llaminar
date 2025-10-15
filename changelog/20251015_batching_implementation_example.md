# Parallel Batching: Concrete Implementation Example

This document shows a concrete code example of how to implement batch prefill, starting with the simplest case.

## Example: Batch Prefill for batch_size=2

### Input
```cpp
std::vector<std::string> prompts = {
    "Explain machine learning.",      // Will tokenize to [5 tokens]
    "What is quantum computing?"       // Will tokenize to [5 tokens]
};
```

### Step 1: Tokenize and Pad

```cpp
// In QwenPipeline::prefillBatch()

std::vector<std::vector<int>> token_batches;
for (const auto& prompt : prompts) {
    token_batches.push_back(tokenizer.tokenize(prompt));
}
// token_batches[0] = [1234, 5678, 234, 567, 890]     // 5 tokens
// token_batches[1] = [4321, 8765, 432, 765, 098]     // 5 tokens

// Find max length
size_t max_len = 0;
for (const auto& tokens : token_batches) {
    max_len = std::max(max_len, tokens.size());
}
// max_len = 5 (both sequences same length, no padding needed in this case)

// Create padded tensor [batch, max_len]
auto token_tensor = TensorFactory::create_simple({batch_size, max_len});
float* token_data = token_tensor->data();

// Fill with token IDs (cast int to float for tensor storage)
for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < token_batches[b].size(); ++t) {
        token_data[b * max_len + t] = static_cast<float>(token_batches[b][t]);
    }
    // Pad remaining with 0s (already initialized to 0)
}

// Result: token_tensor shape [2, 5]
// [[1234, 5678, 234, 567, 890],
//  [4321, 8765, 432, 765, 098]]
```

### Step 2: Embed Batch

```cpp
// In EmbeddingKernel::executeBatch()

auto embeddings = TensorFactory::create_simple({batch_size, max_len, d_model});
// Shape: [2, 5, 896]

const float* embedding_weights = weights.token_embedding;  // [vocab_size, d_model]

#pragma omp parallel for collapse(2)
for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < max_len; ++t) {
        int token_id = static_cast<int>(token_data[b * max_len + t]);
        
        // Skip padding tokens (token_id == 0)
        if (token_id == 0 && t >= token_batches[b].size()) {
            // Leave as zeros (already initialized)
            continue;
        }
        
        // Copy embedding: embedding_weights[token_id * d_model : (token_id+1) * d_model]
        const float* src = embedding_weights + token_id * d_model;
        float* dst = embeddings->data() + (b * max_len + t) * d_model;
        std::memcpy(dst, src, d_model * sizeof(float));
    }
}

// Result: embeddings shape [2, 5, 896]
// Batch 0: [[emb(1234)], [emb(5678)], [emb(234)], [emb(567)], [emb(890)]]
// Batch 1: [[emb(4321)], [emb(8765)], [emb(432)], [emb(765)], [emb(098)]]
```

### Step 3: Process Through Transformer Layer

```cpp
// In QwenPipeline::executeTransformerLayerBatch()

auto hidden = embeddings;  // [2, 5, 896]

for (size_t layer = 0; layer < n_layers_; ++layer) {
    // 3a. Pre-attention RMSNorm
    auto normed = rmsNormBatch(hidden, weights.layers[layer].attn_norm);
    // Shape: [2, 5, 896]
    
    // 3b. Attention with batching
    auto attn_out = attentionBatch(normed, layer);
    // Shape: [2, 5, 896]
    
    // 3c. Residual connection
    hidden = addTensors(hidden, attn_out);  // Element-wise add
    // Shape: [2, 5, 896]
    
    // 3d. Pre-FFN RMSNorm
    normed = rmsNormBatch(hidden, weights.layers[layer].ffn_norm);
    
    // 3e. FFN with batching
    auto ffn_out = ffnBatch(normed, layer);
    // Shape: [2, 5, 896]
    
    // 3f. Residual connection
    hidden = addTensors(hidden, ffn_out);
    // Shape: [2, 5, 896]
}
```

### Step 4: Attention Kernel (Most Complex Part)

```cpp
// In MPIAttentionKernel::executeBatch()

std::shared_ptr<TensorBase> attentionBatch(
    const std::shared_ptr<TensorBase>& hidden,  // [batch, seq_len, d_model] = [2, 5, 896]
    size_t layer)
{
    size_t batch_size = hidden->shape()[0];   // 2
    size_t seq_len = hidden->shape()[1];      // 5
    size_t d_model = hidden->shape()[2];      // 896
    
    auto output = TensorFactory::create_simple({batch_size, seq_len, d_model});
    
    // Process each sequence in batch independently
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Extract this sequence's hidden states [seq_len, d_model]
        auto hidden_b = sliceBatch(hidden, b);  // [5, 896]
        
        // Project to Q, K, V
        auto Q = projectQuery(hidden_b, layer);   // [n_heads, seq_len, head_dim] = [14, 5, 64]
        auto K = projectKey(hidden_b, layer);     // [14, 5, 64]
        auto V = projectValue(hidden_b, layer);   // [14, 5, 64]
        
        // Get or initialize KV cache for this batch element
        // k_cache_batch_[layer] has shape [batch, n_heads, max_seq, head_dim]
        auto k_cache_b = getKVCacheSlice(k_cache_batch_[layer], b);  // [14, max_seq, 64]
        auto v_cache_b = getKVCacheSlice(v_cache_batch_[layer], b);
        
        // Update KV cache with new keys/values
        int position = n_past_batch_[b];  // Current position for this sequence
        updateKVCache(k_cache_b, K, position, seq_len);
        updateKVCache(v_cache_b, V, position, seq_len);
        
        // Compute attention scores: Q @ K^T
        // Q: [14, 5, 64]
        // K_cache: [14, position+seq_len, 64]
        // Scores: [14, 5, position+seq_len]
        auto scores = matmulBatch(Q, transposeLastTwo(k_cache_b));
        
        // Scale by sqrt(head_dim)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        scalarMultiply(scores, scale);
        
        // Apply causal mask (can't attend to future positions)
        applyCausalMask(scores, position);
        
        // Apply padding mask (not needed in this example - same length)
        // applyPaddingMask(scores, actual_lengths[b]);
        
        // Softmax over keys dimension
        auto attn_weights = softmax(scores, /*dim=*/-1);  // [14, 5, position+seq_len]
        
        // Weighted sum of values: attn_weights @ V_cache
        // [14, 5, position+seq_len] @ [14, position+seq_len, 64] = [14, 5, 64]
        auto attn_out = matmulBatch(attn_weights, v_cache_b);
        
        // Reshape and project: [14, 5, 64] -> [5, 14*64] -> [5, 896]
        auto attn_reshaped = reshape(attn_out, {seq_len, n_heads * head_dim});
        auto projected = projectOutput(attn_reshaped, layer);  // [5, 896]
        
        // Write back to output batch tensor
        copyToBatchSlice(output, b, projected);
    }
    
    return output;  // [2, 5, 896]
}
```

### Step 5: FFN Kernel (Simpler - Already Supports Batching)

```cpp
// In QwenPipeline::ffnBatch()

std::shared_ptr<TensorBase> ffnBatch(
    const std::shared_ptr<TensorBase>& hidden,  // [batch, seq_len, d_model] = [2, 5, 896]
    size_t layer)
{
    // Flatten batch dimension for matmul
    size_t batch_size = hidden->shape()[0];  // 2
    size_t seq_len = hidden->shape()[1];     // 5
    size_t d_model = hidden->shape()[2];     // 896
    
    auto flat_hidden = hidden->reshape({batch_size * seq_len, d_model});  // [10, 896]
    
    // Gate and Up projections (fused if FFN fusion enabled)
    auto gate_up = linearBatch(flat_hidden, weights.layers[layer].w_fused);  // [10, 2*d_ff]
    
    // Split into gate and up
    size_t d_ff = weights.layers[layer].d_ff;
    auto gate = gate_up->slice(0, d_ff);     // [10, d_ff]
    auto up = gate_up->slice(d_ff, 2*d_ff);  // [10, d_ff]
    
    // SwiGLU activation: gate * silu(up)
    auto activated = swiGLU(gate, up);  // [10, d_ff]
    
    // Down projection
    auto down = linearBatch(activated, weights.layers[layer].w_down);  // [10, d_model]
    
    // Reshape back to batch format
    return down->reshape({batch_size, seq_len, d_model});  // [2, 5, 896]
}
```

### Step 6: Final Layer Norm and Logits

```cpp
// In QwenPipeline::prefillBatch()

// After all transformer layers
auto final_hidden = hidden;  // [2, 5, 896]

// Final layer norm
auto normed = rmsNormBatch(final_hidden, weights.ln_f_weight);  // [2, 5, 896]

// Project to vocabulary
auto flat_normed = normed->reshape({batch_size * seq_len, d_model});  // [10, 896]
auto logits_flat = linearBatch(flat_normed, weights.lm_head);  // [10, vocab_size] = [10, 151936]
auto logits = logits_flat->reshape({batch_size, seq_len, vocab_size});  // [2, 5, 151936]

// Update positions for next decode step
for (size_t b = 0; b < batch_size; ++b) {
    n_past_batch_[b] += token_batches[b].size();
}
// n_past_batch_ = [5, 5]

return logits;  // [2, 5, 151936]
```

### Step 7: Decode Next Token (Batch Decode)

```cpp
// In QwenPipeline::decodeBatch()

bool QwenPipeline::decodeBatch(
    const std::vector<int>& next_tokens,  // [batch_size] = [token_b0, token_b1]
    std::shared_ptr<TensorBase>& out_logits)
{
    // Example: next_tokens = [42, 123]  (one token per sequence in batch)
    
    // 1. Embed each token
    auto embeddings = TensorFactory::create_simple({batch_size, 1, d_model});  // [2, 1, 896]
    for (size_t b = 0; b < batch_size; ++b) {
        embedSingleTokenBatch(next_tokens[b], embeddings, b);
    }
    
    // 2. Process through transformer (same as prefill, but seq_len=1)
    auto hidden = embeddings;
    for (size_t layer = 0; layer < n_layers_; ++layer) {
        auto normed = rmsNormBatch(hidden, weights.layers[layer].attn_norm);
        auto attn_out = attentionBatch(normed, layer);  // Uses n_past_batch_ for KV cache position
        hidden = addTensors(hidden, attn_out);
        
        normed = rmsNormBatch(hidden, weights.layers[layer].ffn_norm);
        auto ffn_out = ffnBatch(normed, layer);
        hidden = addTensors(hidden, ffn_out);
    }
    
    // 3. Project to logits
    auto normed = rmsNormBatch(hidden, weights.ln_f_weight);  // [2, 1, 896]
    auto flat = normed->reshape({batch_size, d_model});        // [2, 896]
    out_logits = linearBatch(flat, weights.lm_head);           // [2, vocab_size] = [2, 151936]
    
    // 4. Update positions
    for (size_t b = 0; b < batch_size; ++b) {
        n_past_batch_[b]++;
    }
    // n_past_batch_ = [6, 6] (was [5, 5])
    
    return true;
}
```

### Step 8: Sampling from Batch Logits

```cpp
// In BenchmarkRunner::runBatchBenchmark()

// After decodeBatch, logits has shape [batch_size, vocab_size]
std::vector<int> next_tokens(batch_size);

for (size_t b = 0; b < batch_size; ++b) {
    // Extract logits for this sequence
    const float* logits_b = logits->data() + b * vocab_size;
    
    // Greedy sampling: argmax
    int best_token = 0;
    float max_logit = logits_b[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits_b[v] > max_logit) {
            max_logit = logits_b[v];
            best_token = v;
        }
    }
    
    next_tokens[b] = best_token;
}

// next_tokens = [567, 890]  (example)
// Feed these back to decodeBatch() for next step
```

## Complete Batch Benchmark Flow

```cpp
void runBatchBenchmark() {
    // Setup
    std::vector<std::string> prompts = {"Prompt 1", "Prompt 2"};
    size_t batch_size = prompts.size();
    pipeline.resetBatch(batch_size);
    
    // Tokenize
    std::vector<std::vector<int>> token_batches;
    for (const auto& prompt : prompts) {
        token_batches.push_back(tokenizer.tokenize(prompt));
    }
    
    // PREFILL PHASE
    auto t0 = now();
    std::shared_ptr<TensorBase> logits;
    pipeline.prefillBatch(token_batches, logits);  // [batch, max_len, vocab]
    auto t1 = now();
    double prefill_ms = duration(t0, t1);
    
    // Sample initial tokens
    std::vector<int> current_tokens(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        // Take last token's logits for each sequence
        current_tokens[b] = greedySample(logits, b, /*position=*/token_batches[b].size() - 1);
    }
    
    // DECODE PHASE
    auto t2 = now();
    for (int step = 0; step < n_predict; ++step) {
        pipeline.decodeBatch(current_tokens, logits);  // [batch, vocab]
        
        // Sample next tokens
        for (size_t b = 0; b < batch_size; ++b) {
            current_tokens[b] = greedySample(logits, b, /*position=*/0);
        }
    }
    auto t3 = now();
    double decode_ms = duration(t2, t3);
    
    // Calculate metrics
    size_t total_prefill_tokens = sum(token_batches.sizes());
    size_t total_decode_tokens = batch_size * n_predict;
    
    double prefill_throughput = total_prefill_tokens / (prefill_ms / 1000.0);
    double decode_throughput = total_decode_tokens / (decode_ms / 1000.0);
    
    std::cout << "Prefill: " << prefill_throughput << " tok/s\n";
    std::cout << "Decode: " << decode_throughput << " tok/s\n";
    std::cout << "Aggregate: " << (total_prefill_tokens + total_decode_tokens) 
              << " tokens in " << (prefill_ms + decode_ms) << " ms\n";
}
```

## Expected Output

```
Batch Size: 2 sequences
Prefill: 9 tokens in 1800 ms (5.0 tok/s)
Decode: 100 tokens in 41000 ms (2.4 tok/s)  
Aggregate: 109 tokens in 42800 ms (2.5 tok/s)

Note: This is WORSE than single sequence! Why?
- Padding overhead (if sequences have different lengths)
- No optimization yet (first implementation)
- OpenMP parallelization not tuned
- Need to optimize memory access patterns

After optimization (batch=2):
Prefill: 9 tokens in 1000 ms (9.0 tok/s)
Decode: 100 tokens in 21000 ms (4.8 tok/s)
Aggregate: 109 tokens in 22000 ms (5.0 tok/s)
Improvement: 1.9× over single sequence

After optimization (batch=32):
Prefill: 144 tokens in 4500 ms (32 tok/s)
Decode: 1600 tokens in 5600 ms (286 tok/s)
Aggregate: 1744 tokens in 10100 ms (173 tok/s)
Improvement: 13× over single sequence
```

## Key Takeaways

1. **Batch dimension is always first**: `[batch, seq_len, d_model]`
2. **Flatten for matmul**: Reshape to `[batch*seq_len, d_model]` for linear layers
3. **Parallelize attention per sequence**: Each batch element processed independently
4. **Track state per sequence**: `n_past_batch_[b]` for each sequence in batch
5. **Pad to same length**: Required for efficient batched operations
6. **Test correctness first**: Verify batch=2 gives same results as 2× batch=1
7. **Then optimize**: Memory layout, OpenMP tuning, kernel fusion

This is a complete working example of how to implement parallel batching in Llaminar!
