# Parity Test Framework

## Overview

The Parity Test Framework provides infrastructure for comparing intermediate tensor states between Llaminar's execution and reference implementations (PyTorch and llama.cpp). This enables detailed validation of mathematical correctness at each stage of the transformer pipeline.

### Key Features (October 2025 Update)

- **PrefillProvider Integration** ✨: Built-in snapshot capture in all prefill providers (OpenBLAS, COSMA)
- **PyTorch Reference** ✨: Ground truth validation against PyTorch reference implementation
- **171 Comparison Points**: Complete coverage with 3 global + 168 per-layer (6 stages × 28 layers) snapshots
- **Stage-by-Stage Detection**: Pinpoint exact layer and stage where divergence begins
- **Zero Production Overhead**: Snapshot capture compiled out in release builds (`#ifdef NDEBUG`)
- **Multi-Backend Testing**: Compare OpenBLAS vs COSMA vs PyTorch for comprehensive validation

## Architecture

The framework consists of four main components:

1. **PrefillProvider Base Class**: Built-in snapshot capture utilities inherited by all providers
2. **Snapshot Capture Hooks**: Automatic capture at standardized pipeline stages
3. **Snapshot Registry**: Central storage for captured snapshots during test execution
4. **Comparison Engine**: Metrics computation and tolerance checking against PyTorch/llama.cpp references

### PrefillProvider Integration

All prefill providers automatically inherit snapshot capture capabilities:

```cpp
class PrefillProvider {
protected:
    void captureSnapshot(PipelineStage stage, int layer_index,
                        const float* data, int seq_len, int feature_dim);
    bool isSnapshotEnabled() const;
};

// Concrete providers inherit and use:
class OpenBLASPrefillProvider : public PrefillProvider {
    bool execute(...) override {
        // Automatic snapshot capture at each stage
        captureSnapshot(PipelineStage::EMBEDDING, -1, ...);
        // ... layer execution ...
        captureSnapshot(PipelineStage::ATTENTION_OUTPUT, layer_idx, ...);
    }
};
```

### Pipeline Stages

The framework captures snapshots at these standardized transformer pipeline stages:

#### Global Stages (layer=-1)
- `EMBEDDING`: Token embedding lookup output
- `FINAL_NORM`: After final RMSNorm (before LM head)
- `LM_HEAD`: Language model head output (logits)

#### Per-Layer Stages (layer=0..27 for Qwen-0.5B)
- `ATTENTION_NORM`: RMSNorm before attention (input to Q/K/V)
- `ATTENTION_OUTPUT`: After attention output projection W_o
- `ATTENTION_RESIDUAL`: After attention residual add
- `FFN_NORM`: RMSNorm before FFN (input to gate/up)
- `FFN_DOWN`: After FFN down projection  
- `FFN_RESIDUAL`: After FFN residual add (final layer output)

#### Detailed Attention Substages (optional, for debugging)
- `QKV_PROJECTION`: Q, K, V linear projections (combined or separate)
- `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`: Individual projections
- `ROPE_APPLICATION`: After rotary position embeddings
- `ATTENTION_SCORES`: Q @ K^T attention scores
- `ATTENTION_SOFTMAX`: After softmax over attention scores
- `ATTENTION_CONTEXT`: Attention weights @ V

#### FFN Substages (optional, for debugging)
- `FFN_GATE`: Gate projection output
- `FFN_UP`: Up projection output
- `FFN_SWIGLU`: SwiGLU activation output

**Total Snapshots**: 3 global + (6 × 28 layers) = **171 snapshots** for complete prefill validation

## Usage

### Quick Start: PrefillProvider Parity Testing

```cpp
#include "prefill_provider.h"
#include "openblas_prefill_provider.h"
#include "pipeline_snapshot_manager.h"

// 1. Enable snapshot capture
setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);
PipelineSnapshotManager::instance().setEnabled(true);

// 2. Create provider and execute
auto provider = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
PrefillMetrics metrics;
bool success = provider->execute(tokens, weights, output, ctx, metrics);

// 3. Snapshots automatically captured at all 171 stages!
LOG_INFO("Captured " << metrics.snapshots_captured << " snapshots");
```

### Manual Snapshot Capture (for custom kernels)

```cpp
#include "parity_test_framework.h"

using namespace llaminar::parity;

// Enable snapshot capture
LlaminarSnapshotHook::set_enabled(true);

// Capture a snapshot manually
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

### PrefillProvider Parity Test Example

```cpp
TEST(ParityFramework, OpenBLASPrefillVsPyTorch)
{
    // 1. Setup environment
    setenv("PYTORCH_SNAPSHOT_DIR", "pytorch_snapshots/", 1);
    setenv("PYTORCH_SNAPSHOT_TOKENS", "1,2,3,4,5", 1);
    setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);
    
    PipelineSnapshotManager::instance().setEnabled(true);
    
    // 2. Run OpenBLAS prefill provider
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    auto provider = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    
    PrefillMetrics metrics;
    StageContext ctx(tokens.size());
    std::shared_ptr<TensorBase> output;
    
    bool success = provider->execute(tokens, weights, output, ctx, metrics);
    ASSERT_TRUE(success);
    
    LOG_INFO("OpenBLAS: " << metrics.total_ms() << "ms, "
             << metrics.snapshots_captured << " snapshots");
    
    // 3. Load PyTorch reference snapshots
    PyTorchSnapshotLoader pytorch("pytorch_snapshots/");
    
    // 4. Compare stage-by-stage
    SnapshotRegistry& registry = SnapshotRegistry::instance();
    
    // Global stages
    compareSnapshot(registry, pytorch, PipelineStage::EMBEDDING, -1);
    
    // Per-layer stages
    for (int layer = 0; layer < config.n_layers; ++layer) {
        compareSnapshot(registry, pytorch, PipelineStage::ATTENTION_NORM, layer);
        compareSnapshot(registry, pytorch, PipelineStage::ATTENTION_OUTPUT, layer);
        compareSnapshot(registry, pytorch, PipelineStage::ATTENTION_RESIDUAL, layer);
        compareSnapshot(registry, pytorch, PipelineStage::FFN_NORM, layer);
        compareSnapshot(registry, pytorch, PipelineStage::FFN_DOWN, layer);
        compareSnapshot(registry, pytorch, PipelineStage::FFN_RESIDUAL, layer);
    }
    
    compareSnapshot(registry, pytorch, PipelineStage::FINAL_NORM, -1);
    compareSnapshot(registry, pytorch, PipelineStage::LM_HEAD, -1);
}

void compareSnapshot(SnapshotRegistry& registry, PyTorchSnapshotLoader& pytorch,
                    PipelineStage stage, int layer_idx) {
    TensorSnapshot llaminar_snap;
    std::string key = snapshotKey("llaminar", stage, layer_idx);
    ASSERT_TRUE(registry.get_snapshot(key, llaminar_snap));
    
    auto pytorch_snap = pytorch.load(stage, layer_idx);
    ASSERT_TRUE(pytorch_snap.has_value());
    
    auto result = SnapshotComparator::compare(*pytorch_snap, llaminar_snap);
    EXPECT_TRUE(result.passed()) << "Stage: " << stageToString(stage)
                                  << ", Layer: " << layer_idx
                                  << ", rel_l2: " << result.metrics.rel_l2
                                  << ", max_abs: " << result.metrics.max_abs_diff;
}
```

## Generating PyTorch Reference Snapshots

### Step 1: Run PyTorch Reference Implementation

```bash
cd python/reference
python run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --capture-stages all \
    --output pytorch_snapshots.npz
```

### Step 2: Convert NPZ to .npy Files

```bash
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/
```

**Output Directory Structure**:
```
pytorch_snapshots/
  EMBEDDING_-1.npy              (shape: [5, 896])
  ATTENTION_NORM_0.npy          (shape: [5, 896])
  ATTENTION_OUTPUT_0.npy        (shape: [5, 896])
  ...
  FFN_RESIDUAL_27.npy           (shape: [5, 896])
  FINAL_NORM_-1.npy             (shape: [5, 896])
  LM_HEAD_-1.npy                (shape: [5, 151936])
```

### Step 3: Set Environment Variables

```bash
export PYTORCH_SNAPSHOT_DIR=pytorch_snapshots/
export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4,5
```

## Integration with Custom Kernels

To add parity testing to a custom kernel or pipeline stage:

1. **Inherit from PrefillProvider** (recommended) or use manual capture

2. **Add capture hook at key points**:
```cpp
// In your custom provider or kernel
if (isSnapshotEnabled()) {
    captureSnapshot(
        PipelineStage::ATTENTION_OUTPUT,
        layer_idx,
        attn_output->data(),
        seq_len,
        d_model
    );
}
```

3. **Generate corresponding PyTorch reference** (update `python/reference/qwen.py` to capture new stage)

## Multi-Backend Comparison

### OpenBLAS vs COSMA Parity

Compare different prefill providers to validate consistency:

```cpp
TEST(MultiBackendParity, OpenBLASvsCOSMA)
{
    setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);
    PipelineSnapshotManager::instance().setEnabled(true);
    
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    
    // Execute with OpenBLAS
    auto openblas = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    PrefillMetrics openblas_metrics;
    std::shared_ptr<TensorBase> openblas_output;
    openblas->execute(tokens, weights, openblas_output, ctx, openblas_metrics);
    
    // Execute with COSMA
    auto cosma = std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    PrefillMetrics cosma_metrics;
    std::shared_ptr<TensorBase> cosma_output;
    cosma->execute(tokens, weights, cosma_output, ctx, cosma_metrics);
    
    // Compare stage-by-stage
    SnapshotRegistry& registry = SnapshotRegistry::instance();
    
    for (int layer = 0; layer < config.n_layers; ++layer) {
        TensorSnapshot openblas_snap, cosma_snap;
        
        registry.get_snapshot(snapshotKey("OpenBLAS", PipelineStage::FFN_DOWN, layer), 
                            openblas_snap);
        registry.get_snapshot(snapshotKey("COSMA", PipelineStage::FFN_DOWN, layer), 
                            cosma_snap);
        
        auto result = SnapshotComparator::compare(openblas_snap, cosma_snap);
        EXPECT_TRUE(result.passed()) << "Layer " << layer 
                                      << " OpenBLAS vs COSMA diverged";
    }
}
```

## Legacy llama.cpp Integration

For historical reference, llama.cpp integration:

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

### Llaminar Snapshot Capture

- **`LLAMINAR_PARITY_CAPTURE=1`**: Enable snapshot capture in PrefillProviders and custom kernels
- **`LLAMINAR_PARITY_COMPARE=1`**: Enable comparison with reference during execution (for debugging)
- **`LLAMINAR_PARITY_STAGE_FILTER=<regex>`**: Capture only matching stages (e.g., `ATTENTION.*`)
- **`LLAMINAR_PARITY_LAYER_RANGE=<start>-<end>`**: Capture only specific layers (e.g., `0-5`)

### PyTorch Reference Loading

- **`PYTORCH_SNAPSHOT_DIR=<path>`**: Directory containing PyTorch .npy snapshot files
- **`PYTORCH_SNAPSHOT_TOKENS=<csv>`**: Comma-separated token IDs used in PyTorch run (e.g., `1,2,3,4,5`)
- **`PYTORCH_SNAPSHOT_PREFIX=<name>`**: Prefix for snapshot filenames (default: stage name)

### Debugging and Tracing

- **`LLAMINAR_PARITY_TRACE=1`**: Verbose logging for snapshot capture/comparison
- **`LLAMINAR_PARITY_DUMP=1`**: Dump first 10 values of each snapshot for debugging
- **`LLAMINAR_PARITY_ABORT_ON_FAIL=1`**: Abort on first divergence (useful for debugging)

## Comparison Metrics

The framework computes these metrics when comparing snapshots (automatically in PrefillProviders):

- **max_abs_diff**: Maximum absolute difference across all elements
- **mean_abs_diff**: Mean absolute difference
- **rel_l2**: Relative L2 norm = ||actual - expected||_2 / ||expected||_2
- **worst_index**: Index of element with maximum difference

### Tolerance Configuration

Default tolerances (configurable via `SnapshotComparator::setTolerances()`):

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

**Numerical Precision**
- Small differences (rel_l2 < 1e-4) are expected due to float32 precision
- Quantized models (Q4_0, Q6_K) have higher tolerances
- COSMA may reorder operations → slightly different rounding

**Snapshot Mismatches**
- Ensure PyTorch snapshots use identical tokenization
- Verify layer indexing: Llaminar uses 0-based, some frameworks use 1-based
- Check shape broadcasting: PyTorch [batch, seq, dim] vs Llaminar [seq, dim]

**Performance Impact**
- Snapshot capture adds ~5-10% overhead in debug builds
- Compiled out completely in release builds (`#ifdef NDEBUG`)
- Use stage filtering to reduce capture overhead during development

## See Also

- **[PREFILL_PARITY_TESTING_GUIDE.md](PREFILL_PARITY_TESTING_GUIDE.md)**: Comprehensive guide to PrefillProvider stage-by-stage parity testing
  - Detailed 171-stage workflow
  - PyTorch reference generation scripts
  - Debugging divergences with layer-specific diagnostics
  - Tolerance calibration strategies
  - Performance impact analysis

- **[llaminar-architecture.instructions.md](../.github/instructions/llaminar-architecture.instructions.md)**: Architecture documentation
  - Section 4: Prefill Provider Abstraction (strategy pattern, factory selection)
  - Provider lifecycle and integration points
  - OpenBLAS vs COSMA backend selection logic

- **[copilot-instructions.md](../.github/copilot-instructions.md)**: Development guidelines
  - Testing standards and framework usage
  - Debugging workflows for parity tests
  - MPI testing best practices

## Conclusion

The Parity Test Framework has evolved from manual llama.cpp comparison to automated PrefillProvider integration:

**Modern Approach (October 2025)**:
- ✅ **Built-in snapshot capture** in all PrefillProviders (OpenBLAS, COSMA)
- ✅ **171 automatic checkpoints**: 3 global + 168 per-layer (6 stages × 28 layers)
- ✅ **PyTorch ground truth**: Reference implementation with identical architecture
- ✅ **Zero production overhead**: Compiled out in release builds
- ✅ **Stage-by-stage debugging**: Pinpoint exact divergence location ("Layer 15, FFN_DOWN" vs "somewhere...")
- ✅ **Multi-backend validation**: OpenBLAS vs COSMA consistency checks

**Legacy Approach (deprecated)**:
- ❌ Manual llama.cpp snapshot extraction
- ❌ Complex hooking and binary modification
- ❌ Limited comparison points
- ❌ Debugging required extensive instrumentation

**Key Benefits**:
1. **Precision**: Detect divergence at exact layer and stage
2. **Automation**: No manual extraction, providers handle capture
3. **Coverage**: 171 snapshots vs ~10-20 manual checkpoints
4. **Performance**: No overhead in production (release builds)
5. **Multi-backend**: Compare different implementations automatically

For detailed workflows, debugging strategies, and advanced usage, see [PREFILL_PARITY_TESTING_GUIDE.md](PREFILL_PARITY_TESTING_GUIDE.md).

---

## Legacy Documentation (Historical Reference)

The following sections describe the old llama.cpp-based approach. They are preserved for historical context but are no longer recommended.

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
