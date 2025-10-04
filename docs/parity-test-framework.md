# Parity Test Framework

## Overview

The Parity Test Framework provides infrastructure for comparing intermediate tensor states between Llaminar's distributed attention pipeline and the llama.cpp reference implementation. This enables detailed validation of mathematical correctness at each stage of the transformer pipeline.

## Architecture

The framework consists of three main components:

1. **Snapshot Capture**: Hooks to capture tensor states at specific pipeline stages
2. **Snapshot Registry**: Central storage for captured snapshots during test execution
3. **Comparison Engine**: Metrics computation and tolerance checking

### Pipeline Stages

The framework defines these transformer pipeline stages for snapshot capture:

- `EMBEDDING`: Token embedding lookup
- `ATTENTION_NORM`: RMSNorm before attention
- `QKV_PROJECTION`: Q, K, V linear projections
- `ROPE_APPLICATION`: Rotary position embeddings
- `ATTENTION_SCORES`: Q @ K^T scores
- `ATTENTION_SOFTMAX`: Softmax over attention scores
- `ATTENTION_CONTEXT`: Attention weights @ V
- `ATTENTION_OUTPUT`: Output projection W_o
- `ATTENTION_RESIDUAL`: After attention residual connection
- `FFN_NORM`: RMSNorm before FFN
- `FFN_GATE`: Gate projection
- `FFN_UP`: Up projection
- `FFN_SWIGLU`: SwiGLU activation
- `FFN_DOWN`: Down projection
- `FFN_RESIDUAL`: After FFN residual connection
- `FINAL_NORM`: Final RMSNorm
- `LM_HEAD`: Language model head output (logits)

## Usage

### Basic Snapshot Capture

```cpp
#include "parity_test_framework.h"

using namespace llaminar::parity;

// Enable snapshot capture
LlaminarSnapshotHook::set_enabled(true);

// Capture a snapshot
float* tensor_data = ...;
int seq_len = 32;
int hidden_dim = 896;
int layer_idx = 0;

LlaminarSnapshotHook::capture(
    PipelineStage::ATTENTION_OUTPUT,
    layer_idx,
    tensor_data,
    seq_len,
    hidden_dim
);
```

### Comparing Snapshots

```cpp
// Retrieve snapshots from registry
SnapshotRegistry& registry = SnapshotRegistry::instance();

TensorSnapshot llaminar_snap, llama_snap;
registry.get_snapshot("llaminar_layer_0_attention_output", llaminar_snap);
registry.get_snapshot("llama.cpp_layer_0_attention_output", llama_snap);

// Compare with tolerance
ComparisonTolerance tolerance(1e-3f, 1e-4);  // max_abs, rel_l2
auto result = SnapshotComparator::compare(llama_snap, llaminar_snap, tolerance);

if (!result.passed()) {
    std::cout << "Comparison failed: " << result.error_message << std::endl;
    std::cout << "max_abs: " << result.metrics.max_abs_diff << std::endl;
    std::cout << "rel_l2: " << result.metrics.rel_l2 << std::endl;
}
```

### Full Pipeline Test Example

```cpp
TEST(ParityTest, LayerComparison)
{
    // 1. Run llama.cpp and capture snapshots
    llama_backend_init();
    // ... setup llama.cpp context ...
    llama_decode(ctx, batch);
    
    // Manually register llama.cpp snapshots
    SnapshotRegistry& registry = SnapshotRegistry::instance();
    SnapshotMetadata meta;
    meta.stage = PipelineStage::ATTENTION_OUTPUT;
    meta.layer_index = 0;
    meta.seq_len = seq_len;
    meta.feature_dim = d_model;
    
    TensorSnapshot snap(meta, llama_data_ptr, count);
    registry.register_snapshot("llama.cpp_layer_0_attention_output", snap);
    
    // 2. Run Llaminar with capture enabled
    LlaminarSnapshotHook::set_enabled(true);
    pipeline.execute(tokens, weights, output);
    
    // 3. Compare snapshots
    TensorSnapshot llaminar_snap;
    registry.get_snapshot("llaminar_layer_0_attention_output", llaminar_snap);
    
    auto result = SnapshotComparator::compare(snap, llaminar_snap);
    EXPECT_TRUE(result.passed());
}
```

## Integration with Llaminar Pipeline

To add parity testing to a specific pipeline stage:

1. **Identify the capture point** in the pipeline code (e.g., after attention output projection)

2. **Add capture hook**:
```cpp
// In distributed_transformer_pipeline.cpp or relevant kernel
if (LlaminarSnapshotHook::is_enabled())
{
    LlaminarSnapshotHook::capture(
        PipelineStage::ATTENTION_OUTPUT,
        layer_idx,
        attn_output->data(),
        seq_len,
        d_model
    );
}
```

3. **Extract corresponding data from llama.cpp** (this is the challenging part - see below)

## Extracting Intermediate States from llama.cpp

llama.cpp does not expose intermediate layer states through its public API. However, you can:

### Option 1: Use Embeddings API (Limited)
```cpp
// Only works for final hidden state before LM head
cparams.embeddings = true;
llama_context* ctx = llama_init_from_model(model, cparams);
llama_decode(ctx, batch);

for (int i = 0; i < seq_len; ++i) {
    float* embedding = llama_get_embeddings_ith(ctx, i);
    // This gives you the final normalized hidden state
}
```

### Option 2: Modify llama.cpp (For Development)
For comprehensive stage-by-stage comparison, you may need to:

1. Fork llama.cpp or create a debug build
2. Add hooks in `llama.cpp/src/llama.cpp` at key points:
   - After token embedding
   - After each attention layer
   - After each FFN layer
   - After RMSNorm operations
3. Export these intermediate states through a custom API

Example modification location in llama.cpp:
```cpp
// In llama_decode_internal or similar
// After attention:
if (debug_hook_enabled) {
    export_tensor_snapshot("attn_output", layer_idx, cur.data(), size);
}
```

### Option 3: Reference Implementation Parity (Recommended)
Instead of extracting all intermediate states from llama.cpp, you can:

1. **Validate critical checkpoints**: Embedding, Final Hidden State, Logits
2. **Use numerical analysis**: Compare statistics (mean, std, min, max) when full tensors aren't available
3. **Trust llama.cpp for reference** and focus on end-to-end logits parity

The existing `test_prefill_attention_golden.cpp` demonstrates this approach.

## Environment Variables

- `LLAMINAR_PARITY_CAPTURE`: Enable snapshot capture in Llaminar pipeline
- `LLAMINAR_PARITY_COMPARE`: Enable automatic comparison during execution

## Comparison Metrics

The framework computes these metrics:

- **max_abs_diff**: Maximum absolute difference across all elements
- **mean_abs_diff**: Mean absolute difference
- **rel_l2**: Relative L2 norm = ||actual - expected||_2 / ||expected||_2
- **worst_index**: Index of element with maximum difference

### Typical Tolerances

Based on empirical testing:

- **Quantized models (Q4_0, Q4_K_M)**: 
  - max_abs: 2e-3
  - rel_l2: 5e-4

- **FP16 models**:
  - max_abs: 1e-4
  - rel_l2: 1e-5

- **FP32 models**:
  - max_abs: 1e-6
  - rel_l2: 1e-7

## Extending to New Model Architectures

To extend the framework for a new model family (e.g., LLaMA, Mistral):

1. **Define architecture-specific stages** if needed:
```cpp
enum class CustomStage {
    SLIDING_WINDOW_ATTN,
    MOE_ROUTING,
    EXPERT_SELECTION,
    // ...
};
```

2. **Create architecture-specific hooks**:
```cpp
class MistralParityHook {
    static void capture_sliding_window(int layer, const float* data, ...);
    static void capture_moe_routing(int layer, const float* data, ...);
};
```

3. **Add corresponding llama.cpp extraction** (if the model is supported)

4. **Write architecture-specific test cases**

## Debugging Tips

### Identifying Divergence

When a comparison fails:

1. **Check tensor shapes**: Ensure both snapshots have matching dimensions
2. **Inspect worst differences**: Use `log_top_differences()` to see where divergence is largest
3. **Verify data alignment**: Check for off-by-one errors in indexing
4. **Test with simpler inputs**: Use deterministic token sequences

### Common Issues

- **Size mismatch**: Verify seq_len and feature_dim are correctly propagated
- **Layer indexing**: Ensure layer indices match between Llaminar and llama.cpp (0-indexed vs 1-indexed)
- **Precision differences**: Quantized models have inherent precision loss
- **Distributed state**: In MPI runs, ensure snapshots are captured on rank 0 only or properly gathered

## Performance Considerations

- Snapshot capture adds memory overhead (copy of tensor data)
- Only enable during testing, not production inference
- For large models, capture selectively (e.g., only first/last layers)
- Use environment variable gating to disable in normal CI runs

## Future Enhancements

Potential improvements to the framework:

1. **Automatic llama.cpp hooking**: Patches to extract intermediate states without manual modification
2. **Streaming comparison**: Compare snapshots as they're generated rather than storing all
3. **Statistical summaries**: When full tensor comparison isn't feasible, compare distributions
4. **Differential debugging**: Automatically bisect to find first diverging layer
5. **Multi-rank validation**: Compare distributed tensor shards correctly

## References

- `tests/test_parity_framework.cpp`: Complete usage examples
- `tests/parity_test_framework.h`: API documentation
- `tests/test_prefill_attention_golden.cpp`: Existing end-to-end parity test
- `docs/pipeline-vs-llama-cpp-comparison.md`: Detailed pipeline analysis

## Contributing

When adding new parity tests:

1. Document the expected tolerance for the stage
2. Include test cases for multiple model types (quantized, FP16, FP32)
3. Add MPI-aware tests for distributed execution
4. Update this documentation with new stages or patterns
