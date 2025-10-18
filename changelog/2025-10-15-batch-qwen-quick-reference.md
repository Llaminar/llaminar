# BatchQwenPipeline Quick Reference

## Basic Usage

```cpp
#include "BatchQwenPipeline.h"

// 1. Create configuration
TransformerLayerConfig layer_cfg;
layer_cfg.d_model = 896;
layer_cfg.n_layers = 24;
layer_cfg.n_head = 14;
layer_cfg.vocab_size = 151936;
// ... set other fields

ModelConfig config(layer_cfg, "qwen");

// 2. Instantiate pipeline
BatchQwenPipeline pipeline(config);

// 3. Prepare batch of sequences (variable lengths)
std::vector<std::vector<int>> token_batches = {
    {1, 2, 3, 4, 5},        // sequence 0: 5 tokens
    {10, 11, 12},           // sequence 1: 3 tokens
    {20, 21, 22, 23}        // sequence 2: 4 tokens
};

// 4. Execute prefill
StageContext ctx;
std::shared_ptr<TensorBase> logits;
bool success = pipeline.prefillBatch(token_batches, weights, ctx, logits);

// 5. Retrieve results
if (success) {
    // logits shape: [3, 151936] = [batch_size, vocab_size]
    const auto& shape = logits->shape();
    std::cout << "Batch size: " << shape[0] << std::endl;
    std::cout << "Vocab size: " << shape[1] << std::endl;
}
```

## Current Implementation Status

### ✅ Working
- Pipeline construction
- Batch shape allocation
- Sequence padding and length tracking
- Last-token extraction per sequence
- Output shape validation
- MPI compatibility

### ⏳ Pending (Stubs)
- Embedding lookup (returns zeros)
- Layer execution (passthrough)
- LM head projection (returns zeros)
- Decode path
- KV cache population

## Shape Reference

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| Padding | `vector<vector<int>>` varying | `[B, T_max]` | Pads to longest sequence |
| Embedding | `[B, T_max]` tokens | `[B, T_max, D]` | D = d_model |
| Layers | `[B, T, D]` | `[B, T, D]` | T may change in decode |
| Last-token | `[B, T, D]` | `[B, D]` | Extract final real token |
| LM Head | `[B, D]` | `[B, V]` | V = vocab_size |

## Memory Footprint

### Current (Skeleton)
- **Embedding tensor**: `B × T_max × D × 4 bytes` (FP32)
- **Logits tensor**: `B × vocab_size × 4 bytes`
- **Overhead**: ~minimal (padding mask, length tracking)

**Example** (B=32, T_max=512, D=896, vocab=151936):
- Embedding: 32 × 512 × 896 × 4 = 58 MB
- Logits: 32 × 151936 × 4 = 19 MB
- **Total**: ~77 MB per batch

### Future (With KV Cache)
- **K cache**: `B × n_layers × n_heads × T × head_dim × 4 bytes`
- **V cache**: Same as K cache

**Example** (24 layers, 14 heads, 64 head_dim):
- Per sequence: 24 × 14 × 512 × 64 × 4 = 44 MB
- Full batch (32 seqs): 1.4 GB

## Testing

### Run Tests
```bash
# Build test
cmake --build build --target test_batch_prefill_skeleton --parallel

# Run with CTest
ctest --test-dir build -R BatchPrefillSkeletonTest --output-on-failure

# Run directly (2 ranks)
mpirun -np 2 ./build/test_batch_prefill_skeleton
```

### Test Coverage
- ✅ Constructor basic
- ✅ Prefill batch shapes
- ✅ Empty batch handling
- ✅ Single sequence batch
- ✅ Logits retrieval

## Next Development Steps

1. **Embedding Lookup** (Top Priority)
   ```cpp
   // In prepareEmbedding, after allocation:
   const auto* emb_weight = weights.embedding()->data();
   float* emb_out = embedded->data();
   
   for (size_t b = 0; b < B; ++b) {
       for (size_t t = 0; t < T_max; ++t) {
           int token_id = padded.tokens->data()[b * T_max + t];
           if (!padded.is_padding(b, t)) {
               const float* src = emb_weight + token_id * D;
               float* dst = emb_out + (b * T_max + t) * D;
               std::copy(src, src + D, dst);
           }
       }
   }
   ```

2. **LM Head Projection**
   ```cpp
   // In projectOutput, replace zeros with:
   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               B, vocab, D,
               1.0f, last_hidden->data(), D,
               lm_head->data(), D,
               0.0f, logits_out->data(), vocab);
   ```

3. **Layer Execution Loop**
   ```cpp
   bool runBatchedLayers(TensorBase& hidden, bool is_prefill) {
       for (int layer = 0; layer < config_.n_layers; ++layer) {
           // RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
           // ... call operators with batch dims
       }
       return true;
   }
   ```

## Performance Expectations

### Current (Skeleton)
- Throughput: N/A (returns zeros)
- Latency: <1ms (shape allocation only)

### Target (Full Implementation)
- **Single sequence**: ~9 tok/s prefill, ~5 tok/s decode
- **Batch=32**: ~200 tok/s prefill, ~160 tok/s decode
- **Speedup**: ~22× vs sequential (based on layer-pass reduction)

## Debugging Tips

### Enable Logging
```bash
export LLAMINAR_LOG_LEVEL=DEBUG
```

### Check Shapes
```cpp
LOG_DEBUG("Hidden shape: [" << hidden->shape()[0] << "," 
          << hidden->shape()[1] << "," << hidden->shape()[2] << "]");
```

### Validate Padding
```cpp
for (size_t b = 0; b < batch_size; ++b) {
    std::cout << "Seq " << b << ": actual=" << sequence_lengths_[b]
              << " padded=" << padded_length_ << std::endl;
}
```

## Common Issues

### Linker Errors (Abstract Class)
**Problem**: `cannot declare variable 'pipeline' to be of abstract type`  
**Fix**: Ensure `execute()`, `validate()`, `getKernelType()` are implemented

### MPI Initialization
**Problem**: Pipeline crashes with MPI errors  
**Fix**: Call `MPI_Init()` before constructing pipeline

### Shape Mismatches
**Problem**: Dimensions don't match operator expectations  
**Fix**: Check `batch_size`, `padded_length_`, and `d_model` consistency

## API Reference

### BatchQwenPipeline
```cpp
// Constructor
BatchQwenPipeline(const ModelConfig& config);
BatchQwenPipeline(const ModelConfig& config, const MPIContext& ctx);

// Batch processing
bool prefillBatch(
    const std::vector<std::vector<int>>& token_batches,
    const IModelWeights& weights,
    StageContext& ctx,
    std::shared_ptr<TensorBase>& out_logits
);

bool decodeBatch(
    const std::vector<int>& next_tokens,
    const IModelWeights& weights,
    StageContext& ctx,
    std::shared_ptr<TensorBase>& out_logits
);

// Configuration
void setMaxBatchSize(size_t max_batch);
void clearState();

// Accessors
bool logits(std::shared_ptr<TensorBase>& out_logits) override;
const KVCacheState* kvCacheState() const override;
```

### BatchQwenWeights
```cpp
struct BatchQwenWeights : public IModelWeights {
    QwenPipeline::ModelWeights inner;
    
    const std::shared_ptr<TensorBase>& embedding() const;
    const std::shared_ptr<TensorBase>& lm_head() const;
    int layer_count() const;
    
    // Per-layer accessors
    const std::shared_ptr<TensorBase>& wq(int layer) const;
    const std::shared_ptr<TensorBase>& wk(int layer) const;
    // ... etc
};
```

---

**Last Updated**: October 15, 2025  
**Version**: Foundation (Embedding + Output Skeleton)
