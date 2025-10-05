# PyTorch Reference Integration Guide

This guide explains how to use the PyTorch reference implementation to validate Llaminar's pipeline stages through comprehensive parity testing.

## Overview

The PyTorch reference implementation (`python/reference/`) provides ground truth snapshots of intermediate pipeline stages. These snapshots can be compared against Llaminar's C++ execution to validate correctness at every stage of the transformer pipeline.

**Key Advantage**: Unlike llama.cpp (which only exposes final outputs), PyTorch reference captures **all 21 pipeline stages** including:
- Embedding
- Per-layer attention (norm, QKV, RoPE, scores, softmax, context, output, residual)
- Per-layer FFN (norm, gate, up, SwiGLU, down, residual)
- Final norm
- LM head logits

## Quick Start

### 1. Generate PyTorch Reference Snapshots

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Generate snapshots for a test sequence
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4 \
    --output pytorch_snapshots.npz \
    --verbose
```

This creates `pytorch_snapshots.npz` containing all captured stages.

### 2. Extract Snapshots to .npy Files

The C++ test framework loads individual .npy files (simpler than parsing ZIP archives):

```bash
# Extract using helper script
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

# This creates:
# pytorch_snapshots/
#   EMBEDDING_-1.npy
#   ATTENTION_OUTPUT_0.npy
#   ATTENTION_OUTPUT_1.npy
#   FFN_DOWN_0.npy
#   ...
#   FINAL_NORM_-1.npy
#   LM_HEAD_-1.npy
```

### 3. Run Parity Test

```bash
# Enable the test and set snapshot directory
export PYTORCH_SNAPSHOT_DIR=pytorch_snapshots/
export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4  # Must match generation

# Run the test
ctest --test-dir build -R PyTorchReference --output-on-failure

# Or run directly with MPI
mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter=ParityFramework.DistributedPipelineVsPyTorchReference
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PYTORCH_SNAPSHOT_DIR` | Directory containing extracted .npy files | `pytorch_snapshots/` |
| `PYTORCH_SNAPSHOT_TOKENS` | Comma-separated token IDs (must match generation) | `1,2,3,4` |

## Workflow Details

### Stage 1: Python Reference Execution

The PyTorch reference model:
1. Loads a HuggingFace model (e.g., Qwen2-0.5B-Instruct)
2. Runs forward pass with specified tokens
3. Captures intermediate states via forward hooks
4. Exports to .npz format with keys: `STAGE_layer` (e.g., `EMBEDDING_-1`, `ATTENTION_OUTPUT_0`)

### Stage 2: Snapshot Extraction

The `npz_to_npy.py` helper:
1. Loads the .npz archive (ZIP of .npy files)
2. Extracts each array to an individual .npy file
3. Preserves original shapes and dtypes

### Stage 3: C++ Parity Test

The `test_parity_framework.cpp` test:
1. Loads extracted .npy files using `npz_loader.h`
2. Runs Llaminar pipeline with same tokens
3. Captures Llaminar snapshots via parity framework
4. Compares stage-by-stage using tolerance-based metrics
5. Reports pass/fail with detailed diagnostics

## File Format Details

### .npy Format

NumPy's .npy format is a simple binary format:
- Magic bytes: `\x93NUMPY`
- Version (1 or 2)
- Header length (uint16 or uint32)
- Python dict header (describes shape, dtype, byte order)
- Raw data (C-contiguous float32)

Our `npz_loader.h` parses this format directly (no external dependencies).

### Array Naming Convention

```
{STAGE_NAME}_{LAYER_INDEX}.npy
```

Examples:
- `EMBEDDING_-1.npy` - Token embeddings (global, layer=-1)
- `ATTENTION_OUTPUT_0.npy` - Attention output for layer 0
- `FFN_DOWN_5.npy` - FFN down projection for layer 5
- `FINAL_NORM_-1.npy` - Final normalization (global)
- `LM_HEAD_-1.npy` - Logits (global)

Stage names match `PipelineStage` enum in `src/pipeline_stages.h`.

## Comparison Metrics

The test compares snapshots using three metrics:

1. **Max Absolute Difference**
   ```
   max_abs = max(|pytorch[i] - llaminar[i]|)
   ```
   Tolerance: typically 1e-3 to 2e-3

2. **Mean Absolute Difference**
   ```
   mean_abs = mean(|pytorch[i] - llaminar[i]|)
   ```
   Informational (not used for pass/fail)

3. **Relative L2 Norm**
   ```
   rel_l2 = ||pytorch - llaminar||₂ / max(||pytorch||₂, ||llaminar||₂)
   ```
   Tolerance: typically 1e-4 to 5e-4

### Why These Tolerances?

- **Quantization**: GGUF Q4_0/Q6_K vs PyTorch FP32/BF16 introduces precision differences
- **Numerical stability**: Different accumulation order in distributed ops
- **Implementation details**: SwiGLU activation, RoPE interpolation, etc.

Tighter tolerances (< 1e-5) require FP32 end-to-end, which defeats the purpose of quantization.

## Test Structure

### Test Case: `DistributedPipelineVsPyTorchReference`

```cpp
TEST(ParityFramework, DISABLED_DistributedPipelineVsPyTorchReference)
{
    // 1. Load PyTorch snapshots from PYTORCH_SNAPSHOT_DIR
    PyTorchSnapshotLoader pytorch_loader(snapshot_dir);
    
    // 2. Run Llaminar pipeline with same tokens
    auto result = pipeline->prefill(token_ids);
    
    // 3. Compare stage-by-stage
    for (auto& stage : stages_to_compare) {
        NpyArray pytorch_array;
        pytorch_loader.load_snapshot(stage.name, stage.layer, pytorch_array);
        
        TensorSnapshot llaminar_snapshot;
        registry.get_snapshot(key, llaminar_snapshot);
        
        auto metrics = compare(pytorch_array.data, llaminar_snapshot.data);
        EXPECT_LT(metrics.max_abs_diff, tolerance.max_abs);
        EXPECT_LT(metrics.rel_l2, tolerance.rel_l2);
    }
}
```

**Note**: Test is `DISABLED_` by default because it requires pre-generated snapshots. Enable by:
1. Removing `DISABLED_` prefix
2. Setting `PYTORCH_SNAPSHOT_DIR`
3. Ensuring tokens match

## Troubleshooting

### Issue: "PyTorch snapshot not found"

**Cause**: .npy file doesn't exist in `PYTORCH_SNAPSHOT_DIR`

**Solution**:
```bash
# Check what was generated
ls -lh pytorch_snapshots/

# Regenerate with correct stages
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4 \
    --stages EMBEDDING,ATTENTION_OUTPUT,FFN_DOWN,FINAL_NORM,LM_HEAD \
    --output pytorch_snapshots.npz

# Re-extract
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/
```

### Issue: "Llaminar snapshot not captured"

**Cause**: Parity framework not enabled or stage not captured

**Solution**:
```bash
# Ensure LLAMINAR_PARITY_CAPTURE=1 is set in test
# Check qwen_pipeline.cpp has captureIfEnabled() calls for the stage

# Verify captures:
LLAMINAR_PARITY_CAPTURE=1 ./build/test_parity_framework \
    --gtest_filter=ParityFramework.DistributedPipelineVsPyTorchReference
```

### Issue: Large max_abs_diff

**Cause**: Quantization differences, numerical precision

**Diagnostics**:
```bash
# Run with verbose logging
LLAMINAR_PARITY_CAPTURE=1 LLAMINAR_LOG_LEVEL=DEBUG \
    mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter=ParityFramework.DistributedPipelineVsPyTorchReference

# Check top differences (logged automatically for failures)
```

**Solutions**:
- Use FP16 GGUF instead of Q4/Q6 for tighter tolerances
- Adjust tolerances if differences are acceptable
- Investigate specific layer causing issues

### Issue: Token mismatch

**Cause**: `PYTORCH_SNAPSHOT_TOKENS` doesn't match what was used to generate snapshots

**Solution**:
```bash
# Regenerate with explicit tokens
python python/reference/run_reference.py \
    --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4 --output pytorch_snapshots.npz

# Set matching tokens
export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4
```

## Advanced Usage

### Compare Quantized Models

```bash
# Generate PyTorch reference with quantization
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4 \
    --quantization 4bit \
    --output pytorch_q4_snapshots.npz

# Extract
python tests/npz_to_npy.py pytorch_q4_snapshots.npz pytorch_q4/

# Run test
export PYTORCH_SNAPSHOT_DIR=pytorch_q4/
ctest --test-dir build -R PyTorchReference
```

**Note**: Expect larger tolerances (~5e-3 max_abs) for quantized models.

### Compare Specific Layers

Modify `stages_to_compare` in test to focus on specific layers:

```cpp
std::vector<StageToCompare> stages_to_compare = {
    {"ATTENTION_OUTPUT", PipelineStage::ATTENTION_OUTPUT, 5, {1e-3f, 1e-4}},
    {"FFN_DOWN", PipelineStage::FFN_DOWN, 5, {1e-3f, 1e-4}},
};
```

### Capture Custom Stages

Add new capture points in `qwen_pipeline.cpp`:

```cpp
captureIfEnabled(PipelineStage::ATTENTION_SCORES, attention_scores, layer);
```

Regenerate PyTorch snapshots with new stages, re-extract, and update test.

## Integration with CI

See `add-ci-integration` TODO task for full CI workflow. Summary:

```yaml
# .github/workflows/parity_test.yml
- name: Generate PyTorch Reference
  run: |
    pip install -r requirements.txt
    python python/reference/run_reference.py \
      --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
      --tokens 1,2,3,4 --output pytorch_snapshots.npz
    python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

- name: Run Parity Tests
  run: |
    export PYTORCH_SNAPSHOT_DIR=pytorch_snapshots/
    export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4
    ctest --test-dir build -R PyTorchReference --output-on-failure
```

## References

- PyTorch reference implementation: `python/reference/README.md`
- Parity framework architecture: `.github/instructions/llaminar-architecture.instructions.md` §15
- Test developer guide: `tests/AGENTS.md` §13

## Summary

The PyTorch reference integration provides **comprehensive stage-by-stage validation** of Llaminar's transformer pipeline:

✅ **Complete coverage**: All 21 stages captured  
✅ **Flexible**: Support quantization, custom stages  
✅ **Simple**: Extract .npz → run test → get detailed comparison  
✅ **Extensible**: Easy to add new models (LLaMA, DeepSeek, etc.)  

This addresses the fundamental limitation of llama.cpp (final outputs only) and enables high-confidence validation of distributed pipeline correctness.
