# Integration Guide: Adding Parity Checks to Llaminar Pipeline

This guide shows how to integrate the parity test framework into the Llaminar distributed transformer pipeline for stage-by-stage validation against llama.cpp.

## Overview

The parity framework is designed to work alongside your existing pipeline code with minimal modifications. You add capture hooks at key points, then run tests to compare against llama.cpp.

## Step 1: Add Capture Hooks to Pipeline

### Example: Capturing Attention Output

In `src/mpi_transformer_pipeline.cpp` or `src/kernels/MPIAttentionKernel.cpp`:

```cpp
#include "parity_test_framework.h"  // Add to includes

// After attention output projection is computed
bool DistributedTransformerPipeline::executeTransformerLayer(...)
{
    // ... existing attention computation ...
    
    // Add parity capture hook
    if (llaminar::parity::LlaminarSnapshotHook::is_enabled())
    {
        llaminar::parity::LlaminarSnapshotHook::capture(
            llaminar::parity::PipelineStage::ATTENTION_OUTPUT,
            layer_idx,
            attn_out->data(),
            seq_len,
            config_.d_model
        );
    }
    
    // ... rest of layer execution ...
}
```

### Example Locations for Capture Hooks

Based on the existing pipeline structure:

```cpp
// 1. After embedding lookup
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::EMBEDDING,
        -1,  // No layer index for embedding
        embedded->data(),
        seq_len,
        config_.d_model
    );
}

// 2. After attention RMSNorm
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::ATTENTION_NORM,
        layer_idx,
        attn_norm_out->data(),
        seq_len,
        config_.d_model
    );
}

// 3. After RoPE application (in MPIRoPEKernel)
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::ROPE_APPLICATION,
        layer_idx,
        output_data,  // Q and K after RoPE
        seq_len,
        n_heads * head_dim
    );
}

// 4. After FFN gate projection
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::FFN_GATE,
        layer_idx,
        gate_output->data(),
        seq_len,
        config_.d_ff
    );
}

// 5. After final RMSNorm
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::FINAL_NORM,
        -1,
        final_norm_out->data(),
        seq_len,
        config_.d_model
    );
}
```

## Step 2: Extract Reference from llama.cpp

The challenging part is getting intermediate states from llama.cpp. Here are practical approaches:

### Option A: Use Existing End-to-End Test (Recommended)

The simplest approach is to rely on the existing `test_prefill_attention_golden.cpp` pattern:

```cpp
// In your test
TEST(MyParityTest, ValidateStage)
{
    // 1. Run llama.cpp to get final outputs
    llama_decode(ctx, batch);
    float* embeddings = llama_get_embeddings_ith(ctx, 0);  // Final hidden state
    float* logits = llama_get_logits_ith(ctx, 0);           // Final logits
    
    // Register these as reference snapshots
    registry.register_snapshot("llama.cpp_final_norm", ...);
    registry.register_snapshot("llama.cpp_lm_head", ...);
    
    // 2. Run Llaminar with capture enabled
    LlaminarSnapshotHook::set_enabled(true);
    pipeline.execute(tokens, weights, output);
    
    // 3. Compare final states
    // This validates that the pipeline as a whole is correct
    // If final outputs match, intermediate states are likely correct
}
```

### Option B: Custom llama.cpp Build (For Deep Debugging)

If you need stage-by-stage validation, create a debug build of llama.cpp:

1. **Fork or create a local copy of llama.cpp**

2. **Add export hooks** in `llama.cpp/src/llama.cpp`:

```cpp
// Add this helper function
static std::unordered_map<std::string, std::vector<float>> debug_snapshots;

void llama_debug_export_tensor(const char* name, const ggml_tensor* tensor) {
    if (!tensor || !name) return;
    
    size_t nelems = ggml_nelements(tensor);
    std::vector<float> data(nelems);
    
    // Copy tensor data (simplified - actual implementation needs proper type handling)
    memcpy(data.data(), tensor->data, nelems * sizeof(float));
    
    debug_snapshots[name] = data;
}

// In llama_decode_internal, after key operations:
llama_debug_export_tensor("layer_0_attn_output", cur);
llama_debug_export_tensor("layer_0_ffn_gate", gate);
```

3. **Expose snapshots via API**:

```cpp
// Add to llama.h
LLAMA_API const float* llama_get_debug_snapshot(
    struct llama_context* ctx,
    const char* name,
    size_t* out_size
);
```

4. **Use in tests**:

```cpp
size_t size;
const float* llama_attn_out = llama_get_debug_snapshot(ctx, "layer_0_attn_output", &size);

// Register as reference
registry.register_snapshot("llama.cpp_layer_0_attention_output", 
    TensorSnapshot(meta, llama_attn_out, size));
```

### Option C: Statistical Validation (When Full Tensors Unavailable)

If you can't extract exact intermediate states, compare statistical properties:

```cpp
// Compute statistics for Llaminar tensor
auto compute_stats = [](const float* data, size_t count) {
    float min_val = data[0], max_val = data[0];
    double sum = 0.0, sum_sq = 0.0;
    
    for (size_t i = 0; i < count; ++i) {
        float v = data[i];
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
        sum_sq += v * v;
    }
    
    return Stats{
        .min = min_val,
        .max = max_val,
        .mean = sum / count,
        .std = std::sqrt(sum_sq / count - (sum/count) * (sum/count))
    };
};

// Compare statistics instead of full tensors
auto llaminar_stats = compute_stats(llaminar_data, count);
auto llama_stats = compute_stats(llama_data, count);

EXPECT_NEAR(llaminar_stats.mean, llama_stats.mean, 1e-3);
EXPECT_NEAR(llaminar_stats.std, llama_stats.std, 1e-3);
```

## Step 3: Write Parity Tests

### Basic Test Structure

```cpp
TEST(StageParityTest, AttentionOutputLayer0)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Setup model and tokens
    ModelLoader loader;
    loader.loadModel(model_path);
    auto config = loader.createLayerConfig();
    
    std::vector<int> tokens = {100, 101, 102};
    
    // Run llama.cpp
    if (rank == 0) {
        // ... llama.cpp execution ...
        // Register reference snapshot
        registry.register_snapshot(ref_key, ref_snapshot);
    }
    
    // Run Llaminar
    LlaminarSnapshotHook::set_enabled(true);
    pipeline.execute(tokens, weights, output);
    
    // Compare on rank 0
    if (rank == 0) {
        TensorSnapshot llaminar_snap, llama_snap;
        registry.get_snapshot(llaminar_key, llaminar_snap);
        registry.get_snapshot(ref_key, llama_snap);
        
        auto result = SnapshotComparator::compare(
            llama_snap,
            llaminar_snap,
            ComparisonTolerance(1e-3f, 1e-4)
        );
        
        EXPECT_TRUE(result.passed()) << result.error_message;
        
        if (!result.passed()) {
            SnapshotComparator::log_top_differences(
                llama_snap.data,
                llaminar_snap.data,
                config.d_model,
                10,
                "attention_output_layer_0"
            );
        }
    }
}
```

## Step 4: Running Parity Tests

### Local Development

```bash
# Build with parity hooks included
cmake --build build --target test_parity_framework

# Run basic tests
./build/test_parity_framework --gtest_filter="ParityFramework.Basic*"

# Run with MPI (requires model file)
export LLAMINAR_PARITY_CAPTURE=1
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.DistributedPipelineVsLlamaCpp"
```

### Debugging Divergence

```bash
# Enable detailed logging
export LLAMINAR_LOG_LEVEL=DEBUG
export LLAMINAR_PARITY_CAPTURE=1

# Run specific stage test
mpirun -np 2 ./build/test_parity_framework --gtest_filter="*AttentionOutput*"

# Check output for comparison metrics
# Look for: [PARITY_*] max_abs=... rel_l2=...
```

## Step 5: Interpreting Results

### Successful Parity

```
[PARITY_LOGITS] max_abs=0.00123 mean_abs=0.00045 rel_l2=0.000234
[PARITY_FINAL_HIDDEN] max_abs=0.00089 mean_abs=0.00032 rel_l2=0.000156
✓ All comparisons passed
```

### Failed Parity

```
[PARITY_TOP_DIFF] attention_output_layer_0 top_k=10
  [0,512] diff=0.025 expected=1.234 actual=1.209
  [2,128] diff=0.021 expected=-0.456 actual=-0.477
  ...
✗ Tolerance exceeded: max_abs=0.025 (tol=0.002)
```

When parity fails:

1. **Check the worst difference location** - Is it in a specific feature dimension?
2. **Examine the pattern** - Systematic bias vs random noise
3. **Verify tensor shapes** - Off-by-one errors common
4. **Test simpler inputs** - Use deterministic token sequences
5. **Bisect layers** - Find first diverging layer

## Example: Complete Integration

Here's a complete example showing how to add parity checking to a new stage:

```cpp
// In src/distributed_transformer_pipeline.cpp

#include "parity_test_framework.h"

// Add capture after computing FFN gate
bool DistributedTransformerPipeline::executeTransformerLayer(...)
{
    // ... existing code ...
    
    // Compute FFN gate projection
    auto gate_out = createLocalTensor({seq_len, config_.d_ff});
    bool gate_success = ffn_gate_kernel->execute({ffn_norm_out}, {gate_out});
    
    // ===== PARITY CAPTURE HOOK =====
    if (llaminar::parity::LlaminarSnapshotHook::is_enabled() && rank == 0)
    {
        llaminar::parity::LlaminarSnapshotHook::capture(
            llaminar::parity::PipelineStage::FFN_GATE,
            layer_idx,
            gate_out->data(),
            seq_len,
            config_.d_ff
        );
    }
    // ================================
    
    // ... rest of layer ...
}
```

Then in your test:

```cpp
TEST(FFNParityTest, GateProjection)
{
    // This test validates the FFN gate projection
    
    // 1. Setup
    ModelLoader loader;
    loader.loadModel("models/qwen2.5-0.5b-instruct-q4_0.gguf");
    
    // 2. Get llama.cpp reference (simplified - you'd run actual inference)
    // For this example, we trust end-to-end logits validation
    
    // 3. Run Llaminar with capture
    LlaminarSnapshotHook::set_enabled(true);
    pipeline.execute(tokens, weights, output);
    
    // 4. Validate snapshot was captured
    SnapshotRegistry& registry = SnapshotRegistry::instance();
    EXPECT_TRUE(registry.has_snapshot("llaminar_layer_0_ffn_gate"));
    
    // 5. Basic sanity checks on the captured data
    TensorSnapshot snap;
    registry.get_snapshot("llaminar_layer_0_ffn_gate", snap);
    
    // Check for NaN/Inf
    for (float v : snap.data) {
        EXPECT_TRUE(std::isfinite(v));
    }
    
    // Check reasonable value ranges for SwiGLU gate
    // (This is model-specific, adjust based on your observations)
    float min_val = *std::min_element(snap.data.begin(), snap.data.end());
    float max_val = *std::max_element(snap.data.begin(), snap.data.end());
    
    EXPECT_GT(max_val, -10.0f);  // Reasonable bounds
    EXPECT_LT(min_val, 10.0f);
}
```

## Best Practices

1. **Start with end-to-end validation** - Use the existing golden test first
2. **Add hooks incrementally** - Don't instrument everything at once
3. **Use environment gating** - Don't slow down normal tests
4. **Capture on rank 0 only** - For MPI tests, unless you need distributed validation
5. **Set appropriate tolerances** - Quantized models need looser tolerances
6. **Log top differences** - When comparison fails, examine the worst cases
7. **Test with simple inputs** - Use deterministic token sequences for reproducibility

## Troubleshooting

**Q: Snapshots not being captured**
- Check `LlaminarSnapshotHook::is_enabled()` is true
- Verify capture hooks are in executed code path
- Check MPI rank is correct (rank 0 only?)

**Q: Size mismatch errors**
- Verify seq_len and feature_dim match between Llaminar and llama.cpp
- Check for transposed tensors (swap seq_len and feature_dim)

**Q: Can't extract intermediate states from llama.cpp**
- Use Option A (end-to-end validation) as primary strategy
- Option B (custom build) is for deep debugging only
- Option C (statistical validation) can catch major issues

**Q: Tolerances too tight/loose**
- Adjust based on quantization format (Q4: ~1e-3, FP16: ~1e-4, FP32: ~1e-6)
- Different stages may need different tolerances
- Document your chosen tolerances in tests

## Summary

The parity test framework is a powerful tool for validating pipeline correctness, but it requires thoughtful integration:

1. Add capture hooks at strategic points
2. Focus on end-to-end validation first (logits, final hidden state)
3. Use stage-by-stage validation when debugging specific divergence
4. Be pragmatic about llama.cpp extraction - full intermediate states may not be feasible
5. Trust statistical validation and end-to-end tests as primary verification

The framework is designed to grow with your needs - start simple and add more detailed checks as required.
