# Parity Test Framework

This directory contains the parity test framework for validating Llaminar's distributed attention pipeline against llama.cpp reference implementation.

## Quick Start

```bash
# Build the parity framework test
cmake --build build --target test_parity_framework

# Run basic framework tests (no MPI required)
./build/test_parity_framework --gtest_filter="ParityFramework.Basic*"

# Run full pipeline comparison (requires MPI and model file)
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.DistributedPipelineVsLlamaCpp"
```

## Prerequisites

- GGUF model file in `models/` directory (e.g., `qwen2.5-0.5b-instruct-q4_0.gguf`)
- MPI installation (OpenMPI recommended)
- Built llama.cpp submodule

## Files

- **parity_test_framework.h**: Core framework API
- **parity_test_framework.cpp**: Framework implementation
- **test_parity_framework.cpp**: Test suite demonstrating framework usage

## Test Cases

### ParityFramework.BasicSnapshotCapture
Tests basic snapshot capture and retrieval functionality.

### ParityFramework.SnapshotComparison
Tests tensor comparison metrics (max_abs, rel_l2, etc.).

### ParityFramework.DistributedPipelineVsLlamaCpp
Full end-to-end comparison of Llaminar pipeline against llama.cpp:
- Runs llama.cpp inference to get reference outputs
- Runs Llaminar pipeline with snapshot hooks
- Compares final logits and pre-LM hidden states

## Environment Variables

- `LLAMINAR_PARITY_CAPTURE`: Enable snapshot capture (set to "1")
- `LLAMINAR_PARITY_COMPARE`: Enable automatic comparison (set to "1")
- `PYTORCH_SNAPSHOT_DIR`: Directory containing PyTorch reference snapshots (e.g., "python/reference/")
- `PYTORCH_SNAPSHOT_TOKENS`: Comma-separated token IDs for snapshot generation (e.g., "1,2,3,4,5")

## Key Pipeline Stages

The framework captures snapshots at these transformer pipeline stages:

**Global Stages:**
- `EMBEDDING`: Token embedding output
- `FINAL_NORM`: After final RMSNorm
- `LM_HEAD`: Logits output

**Per-Layer Stages (repeated for each transformer layer):**
- `ATTENTION_NORM`: RMSNorm before attention
- `ATTENTION_OUTPUT`: After attention output projection
- `ATTENTION_RESIDUAL`: After attention residual add
- `FFN_NORM`: RMSNorm before FFN
- `FFN_DOWN`: After FFN down projection
- `FFN_RESIDUAL`: After FFN residual add

**Attention Sub-Stages** (for detailed debugging) ✨:
- `ATTENTION_SCORES`: Q @ K^T scores
- `ATTENTION_SOFTMAX`: After softmax normalization
- **`ATTENTION_CONTEXT`**: Attention weights @ V **(before output projection)**
  - **Critical for debugging**: Isolates whether divergence is in attention mechanism or output projection
  - **Validated**: Achieves rel_l2 < 3e-06 vs PyTorch
  - **Usage**: If ATTENTION_CONTEXT ✅ but ATTENTION_OUTPUT ❌ → focus on W_o weight loading/orientation

## Quick ATTENTION_CONTEXT Debugging

```bash
# 1. Generate PyTorch reference with ATTENTION_CONTEXT
cd python/reference
python generate_test_snapshots.py --model qwen2.5-0.5b-instruct --tokens 1,2,3,4,5 --output snapshots.npz

# 2. Run Llaminar with parity capture
export LLAMINAR_PARITY_CAPTURE=1
export PYTORCH_SNAPSHOT_DIR=python/reference/
./build/test_parity_framework --gtest_filter="*OpenBLASPrefillVsPyTorch"

# 3. Check results
# If ATTENTION_CONTEXT passes but ATTENTION_OUTPUT fails:
#   → Problem is in output projection (W_o matrix multiplication)
# If ATTENTION_CONTEXT fails:
#   → Problem is earlier (Q/K/V projections, RoPE, scores, softmax)
```

## Adding New Parity Tests

To add parity checking for a new pipeline stage:

1. Add capture hook in the pipeline code:
```cpp
if (LlaminarSnapshotHook::is_enabled()) {
    LlaminarSnapshotHook::capture(
        PipelineStage::YOUR_STAGE,
        layer_idx,
        tensor_data,
        seq_len,
        feature_dim
    );
}
```

2. Extract corresponding data from llama.cpp (see docs/parity-test-framework.md)

3. Add test case comparing the snapshots

## Documentation

See [docs/parity-test-framework.md](../docs/parity-test-framework.md) for comprehensive documentation including:
- Detailed API reference
- Integration guide
- llama.cpp extraction strategies
- Extending to new model architectures
- Debugging tips

## Example Output

```
[PARITY_TEST] Using model: models/qwen2.5-0.5b-instruct-q4_0.gguf
[PARITY_TEST] Running llama.cpp reference...
[PARITY_TEST] Running Llaminar pipeline...
[PARITY_TEST] Comparing results...
[PARITY_LOGITS] max_abs=0.00123 mean_abs=0.00045 rel_l2=0.000234
[PARITY_FINAL_HIDDEN] max_abs=0.00089 mean_abs=0.00032 rel_l2=0.000156
[PARITY_TEST] Test complete
```

## Known Limitations

1. **llama.cpp intermediate states**: The public llama.cpp API doesn't expose intermediate layer states. The framework currently validates:
   - Final hidden state (via embeddings API)
   - Final logits
   
   For comprehensive stage-by-stage validation, you may need to modify llama.cpp or use the embeddings API creatively.

2. **MPI distribution**: Current implementation captures snapshots on rank 0 only. For multi-rank validation, snapshots need proper gathering.

3. **Quantization precision**: Quantized models (Q4, Q6) have inherent precision loss. Use appropriate tolerances.

## Future Work

- [ ] Automatic llama.cpp hooking for intermediate states
- [ ] Multi-rank snapshot validation
- [ ] Streaming comparison (avoid storing all snapshots)
- [ ] Statistical comparison when full tensors unavailable
- [ ] Integration with CI pipeline

## Support

For issues or questions about the parity test framework:
1. Check the comprehensive documentation in `docs/parity-test-framework.md`
2. Review existing test cases in `test_parity_framework.cpp`
3. See `test_prefill_attention_golden.cpp` for end-to-end parity patterns
