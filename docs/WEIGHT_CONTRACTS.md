# Weight Contract System

## Overview

The weight contract system provides compile-time and load-time validation of model weight dimensions and orientations. It eliminates runtime shape detection and provides clear error messages when GGUF files don't match expected formats.

## Architecture

### Components

1. **`weight_contracts.h`** - Core contract system
   - `WeightShapeContract`: Individual weight tensor specification
   - `ModelWeightContracts`: Collection of contracts for a complete architecture
   - `getQwenWeightContracts()`: Qwen/Qwen2 canonical format specification

2. **`QwenModelWeights`** - Weight wrapper with validation
   - Inherits from `IModelWeights` interface
   - Provides `validate()` method called during loading
   - Wraps `QwenPipeline::ModelWeights` with type-safe accessors

3. **`QwenPipelineAdapter::loadWeights()`** - Load-time validation
   - Loads GGUF file via `ModelLoader`
   - Creates `QwenModelWeights` instance
   - Calls `validate()` - **fails fast with clear errors if contracts violated**
   - Returns validated weights to pipeline

## Canonical GGUF Format

All weight matrices are stored as **`[out_features, in_features]`** matching PyTorch `nn.Linear` convention.

### Attention Weights

```cpp
// Q/K/V Projections: [out_features, in_features]
attn_q.weight:      [n_head*head_dim, d_model]      // e.g., [896, 896]
attn_k.weight:      [n_head_kv*head_dim, d_model]   // e.g., [128, 896] for GQA
attn_v.weight:      [n_head_kv*head_dim, d_model]   // e.g., [128, 896] for GQA

// Output Projection: [out_features, in_features]
attn_output.weight: [d_model, n_head*head_dim]      // e.g., [896, 896]

// Biases (if present)
attn_q.bias:        [n_head*head_dim]
attn_k.bias:        [n_head_kv*head_dim]
attn_v.bias:        [n_head_kv*head_dim]
```

### FFN Weights

```cpp
// All FFN weights: [out_features, in_features]
ffn_gate.weight:    [d_ff, d_model]                 // e.g., [4864, 896]
ffn_up.weight:      [d_ff, d_model]
ffn_down.weight:    [d_model, d_ff]                 // e.g., [896, 4864]
```

### Global Weights

```cpp
token_embedding:    [vocab_size, d_model]           // e.g., [151669, 896]
output_norm.weight: [d_model]                       // RMSNorm gamma
lm_head (output.weight): [vocab_size, d_model]     // e.g., [151669, 896]
```

## Kernel Usage

Kernels can now **trust** that weights are in canonical format without runtime detection:

```cpp
// MPIAttentionKernel.cpp
// No longer needs to guess orientation!

// Sharding detection based on first dimension
const int wq_rows = wq_global->shape()[0];
const bool weights_are_sharded = (wq_rows == local_head_dim);

// Expected shapes are now deterministic
const int expected_wq_rows = weights_are_sharded ? local_head_dim : (n_head * head_dim);
// Q/K/V are ALWAYS [rows, d_model] in GGUF
// wo is ALWAYS [d_model, cols] in GGUF
```

### Weight Matrix Usage

```cpp
// Forward pass: output = input @ weight^T
// Example: Q projection
//   input:  [seq_len, 896]
//   wq:     [896, 896]  ← guaranteed by contract
//   matmul: input @ wq^T (CblasTrans on B)
//   result: [seq_len, 896] ✓
```

## Test Fixture Updates

All test fixtures must create weights matching GGUF format:

```cpp
// test_abstract_pipeline_parity.cpp
struct RandomWeightBuilder {
    QwenPipeline::ModelWeights build() {
        QwenPipeline::ModelWeights w;
        
        // Global weights
        w.token_embedding = randTensor({cfg.vocab_size, cfg.d_model});
        w.lm_head = randTensor({cfg.vocab_size, cfg.d_model});  // [vocab, d_model]!
        
        for (int i = 0; i < cfg.n_layers; ++i) {
            // Attention: [out_features, in_features]
            w.wq.push_back(randTensor({cfg.n_head * cfg.head_dim, cfg.d_model}));
            w.wk.push_back(randTensor({cfg.n_head_kv * cfg.head_dim, cfg.d_model}));
            w.wv.push_back(randTensor({cfg.n_head_kv * cfg.head_dim, cfg.d_model}));
            w.wo.push_back(randTensor({cfg.d_model, cfg.n_head * cfg.head_dim}));
            
            // FFN: [out_features, in_features]
            w.w_gate.push_back(randTensor({cfg.d_ff, cfg.d_model}));
            w.w_up.push_back(randTensor({cfg.d_ff, cfg.d_model}));
            w.w_down.push_back(randTensor({cfg.d_model, cfg.d_ff}));
        }
        return w;
    }
};
```

## Error Messages

When validation fails, you get clear errors:

```
[ERROR] Weight validation failed: Weight contract validation failed for 'attn_k.weight' (layer 0):
  Description: Key projection (GGUF format: [out, in])
  Reason: Dimension 0 mismatch
  Expected shape: [128, 896] (from [n_head_kv*head_dim, d_model])
  Actual shape:   [896, 128]
```

This immediately identifies:
- **Which weight** is wrong (`attn_k.weight`)
- **Which layer** (layer 0)
- **What was expected** (`[128, 896]`)
- **What was found** (`[896, 128]`)
- **The symbolic expression** (`[n_head_kv*head_dim, d_model]`)

## Benefits

### 1. **Fail Fast**
   - Validation happens at model load time
   - Clear error messages before any inference
   - No runtime shape mismatches during execution

### 2. **Self-Documenting**
   - Contracts serve as executable documentation
   - Symbolic expressions like `n_head*head_dim` clarify intent
   - Human-readable descriptions explain each weight's role

### 3. **Simplified Kernels**
   - No runtime shape detection needed
   - Kernels trust weight format from contracts
   - Reduced code complexity and potential bugs

### 4. **Test Fixture Consistency**
   - All tests must match GGUF canonical format
   - Eliminates synthetic data format mismatches
   - Ensures tests validate real production paths

### 5. **Multi-Architecture Support**
   - Easy to add contracts for new architectures (LLaMA, GPT, etc.)
   - Each architecture defines its own canonical format
   - Shared `IModelWeights` interface ensures consistency

## Future Extensions

### Additional Architectures

```cpp
inline ModelWeightContracts getLlamaWeightContracts() {
    ModelWeightContracts contracts;
    // LLaMA-specific format...
    return contracts;
}

inline ModelWeightContracts getGPTWeightContracts() {
    ModelWeightContracts contracts;
    // GPT-specific format...
    return contracts;
}
```

### Quantization Validation

```cpp
struct WeightShapeContract {
    QuantizationType expected_quant = QuantizationType::F32;
    
    void validate(const std::shared_ptr<TensorBase>& tensor,
                 const TransformerLayerConfig& cfg) const {
        // Validate quantization type matches
        if (tensor->quantization_type() != expected_quant) {
            throw std::runtime_error("Quantization mismatch");
        }
    }
};
```

### Value Range Validation

```cpp
struct WeightShapeContract {
    std::optional<float> expected_min;
    std::optional<float> expected_max;
    
    void validate_values(const std::shared_ptr<TensorBase>& tensor) const {
        // Sanity check value ranges (detect corruption)
    }
};
```

## Related Files

- `/workspaces/llaminar/src/weight_contracts.h` - Core system
- `/workspaces/llaminar/src/qwen_pipeline_adapter.h` - Qwen validation
- `/workspaces/llaminar/src/qwen_pipeline_adapter.cpp` - Load-time validation
- `/workspaces/llaminar/src/model_loader.h` - GGUF format documentation
- `/workspaces/llaminar/tests/test_abstract_pipeline_parity.cpp` - Example usage

## See Also

- [Model Loader Documentation](../src/model_loader.h) - GGUF format reference
- [Attention Stage Contracts](../src/kernels/attention/AttentionStageContracts.h) - Runtime tensor validation
- [PARITY_FRAMEWORK_SUMMARY.md](../PARITY_FRAMEWORK_SUMMARY.md) - Testing architecture
