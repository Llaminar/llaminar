# Layer-by-Layer Parity Testing Guide

## Overview

This guide explains how to use the extended parity test framework to compare Llaminar's implementation against PyTorch reference on a layer-by-layer basis. This is the most effective way to pinpoint exactly where numerical divergence begins in the transformer stack.

## Architecture

The layer-by-layer parity system consists of three components:

1. **C++ Parity Framework** (`tests/parity_test_framework.{h,cpp}`)
   - `SnapshotRegistry`: Central storage for tensor snapshots
   - `SnapshotComparator`: Comparison engine with configurable tolerances
   - `LlaminarSnapshotHook`: Captures Llaminar layer outputs
   - `PytorchSnapshotLoader`: Loads PyTorch NPZ snapshots (planned)
   - `LayerByLayerComparator`: Automated comparison and reporting

2. **Python Capture Script** (`python/reference/capture_pytorch_layers.py`)
   - `LayerCaptureHook`: PyTorch forward hook system
   - `QwenLayerCapture`: Qwen-specific layer capture
   - `compare_layer_outputs()`: Python-side comparison
   - NPZ export for C++ integration

3. **Test Integration** (`tests/test_layer_by_layer_parity.cpp`)
   - GTest framework integration
   - MPI-aware testing
   - Automated comparison workflow

## Quick Start: Finding Divergence

### Step 1: Capture PyTorch Reference Layers

```bash
# Capture all intermediate layer outputs from PyTorch
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30 \
  -o pytorch_layer_captures.npz

# Output:
# ✓ Model loaded
# ✓ Registered 100+ hooks
# ✓ Captured 100+ layer outputs:
#   - embeddings: (1, 7, 896)
#   - layer_0_input_norm_out: (1, 7, 896)
#   - layer_0_attn_out: (1, 7, 896)
#   - layer_0_post_attn_norm_out: (1, 7, 896)
#   - layer_0_ffn_out: (1, 7, 896)
#   - layer_0_out: (1, 7, 896)
#   ... (repeat for all 24 layers)
#   - final_norm_out: (1, 7, 896)
#   - logits: (1, 7, 151936)
# ✓ Saved captures to pytorch_layer_captures.npz
```

### Step 2: Capture Llaminar Layers and Compare

Currently, you can use the Python script for direct comparison:

```bash
# First, generate Llaminar's output with JSON
./build/llaminar \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --output-json llaminar_output.json \
  <<< "1 + 1 ="

# Then compare with PyTorch (with Llaminar snapshots if available)
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30 \
  --llaminar-captures llaminar_layer_captures.npz

# Output will show layer-by-layer comparison:
# ================================================================================
# LAYER-BY-LAYER COMPARISON
# ================================================================================
# PyTorch layers: 102
# Llaminar layers: 102
# Common layers: 102
#
# ✓ embeddings:
#    Shape: (1, 7, 896)
#    Max abs diff: 0.000000
#    Mean abs diff: 0.000000
#    Cosine similarity: 1.000000
#
# ✓ layer_0_input_norm_out:
#    Shape: (1, 7, 896)
#    Max abs diff: 0.000012
#    Mean abs diff: 0.000003
#    Cosine similarity: 0.999999
#
# ❌ layer_0_attn_out:
#    Shape: (1, 7, 896)
#    Max abs diff: 2.451234
#    Mean abs diff: 0.345678
#    Cosine similarity: 0.923456
#
#    🔍 DIVERGENCE DETECTED!
#    Worst element: idx=234
#      PyTorch: 1.234567
#      Llaminar: -1.216567
#      Diff: 2.451234
#
# ================================================================================
# FIRST DIVERGING LAYER: layer_0_attn_out
# Mean abs diff: 0.345678
# ================================================================================
```

### Step 3: Interpret Results

The comparison will identify the **first layer where divergence exceeds tolerance**:

- ✓ **embeddings match** → Weight loading correct
- ✓ **layer_0_input_norm_out matches** → RMSNorm correct
- ❌ **layer_0_attn_out diverges** → Attention is the problem!

Now you know to focus on the attention mechanism in layer 0.

## Detailed Usage

### Python Capture Script Options

```bash
python3 python/reference/capture_pytorch_layers.py --help

Options:
  -m, --model PATH          Path to GGUF model file
  --tokens ID [ID ...]      Token IDs to test with (default: test sequence)
  -o, --output PATH         Output NPZ file (default: pytorch_layer_captures.npz)
  --llaminar-captures PATH  NPZ file with Llaminar captures for comparison
```

### Captured Layers

For Qwen models, the script captures:

**Global layers:**
- `embeddings`: Token embeddings after embedding lookup
- `final_norm_out`: Output after final RMSNorm
- `logits`: Final output logits

**Per-layer (0-23):**
- `layer_N_input_norm_out`: After input RMSNorm (before attention)
- `layer_N_attn_out`: After self-attention (before residual)
- `layer_N_post_attn_norm_out`: After post-attention RMSNorm (before FFN)
- `layer_N_ffn_out`: After FFN/MLP (before residual)
- `layer_N_out`: Final layer output (after both residuals)

### Comparison Metrics

Each layer comparison includes:

- **max_abs_diff**: Maximum absolute difference across all elements
- **mean_abs_diff**: Average absolute difference
- **mean_rel_error**: Average relative error (percentage)
- **max_rel_error**: Maximum relative error
- **cosine_similarity**: Cosine similarity (1.0 = perfect match)

### Tolerance Guidelines

Recommended tolerances based on our investigation:

```python
# Perfect match (embeddings, weight loading)
mean_abs_diff < 1e-6

# Excellent match (numerical precision differences only)
mean_abs_diff < 0.001

# Acceptable match (small accumulation of floating point errors)
mean_abs_diff < 0.01

# Warning (noticeable divergence, investigate)
mean_abs_diff < 0.1

# Failure (significant divergence, bug likely)
mean_abs_diff >= 0.1
```

## Integration with C++ Tests

### Future: C++ Test Workflow (when NPZ loading implemented)

```bash
# 1. Build tests
cmake --build build --target test_layer_by_layer_parity

# 2. Run Llaminar test to capture layers
mpirun -np 2 ./build/test_layer_by_layer_parity \
  --gtest_filter=LayerByLayerParityTest.CaptureLlaminarLayers

# 3. Capture PyTorch reference
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -o pytorch_layer_captures.npz

# 4. Run comparison test
mpirun -np 2 ./build/test_layer_by_layer_parity \
  --gtest_filter=LayerByLayerParityTest.CompareLlaminarVsPytorch

# Output:
# ================================================================================
# LAYER-BY-LAYER COMPARISON REPORT
# ================================================================================
# Total layers compared: 102
# Passed: 25
# Failed: 77
# ================================================================================
#
# ✓ embeddings:
#    Max abs diff: 0.000000
#    ...
#
# ❌ layer_0_attn_out:
#    Max abs diff: 2.451234
#    🔍 DIVERGENCE DETECTED
#    ...
#
# 🔍 FIRST DIVERGING LAYER: layer_0_attn_out
```

### Current C++ Framework Status

**Implemented:**
- ✅ `SnapshotRegistry` for storing captures
- ✅ `SnapshotComparator` with metrics calculation
- ✅ `LlaminarSnapshotHook` for Llaminar capture
- ✅ `PytorchSnapshotLoader` skeleton (key parsing)
- ✅ `LayerByLayerComparator` for automated comparison
- ✅ Test infrastructure in `test_layer_by_layer_parity.cpp`

**Pending:**
- ⏳ NPZ loading implementation in `PytorchSnapshotLoader::load_from_npz()`
- ⏳ Llaminar pipeline instrumentation to call capture hooks
- ⏳ Integration with pipeline factory for automatic capture

**Workaround:**
Use Python-side comparison (fully functional) until C++ NPZ loading is implemented.

## Binary Search Strategy

When you have many layers, use binary search to find first divergence:

1. **Check embeddings** (layer -1)
   - If diverge: weight loading problem
   - If match: continue to layer 0

2. **Check middle layer** (layer 12 of 24)
   - If diverge: problem in layers 0-12
   - If match: problem in layers 13-23

3. **Recursively bisect** until you find the exact layer
   - Example: Check layer 6 → diverges → check layer 3 → matches → check layer 4...

4. **Within the diverging layer**, check sub-components:
   - `layer_N_input_norm_out`: RMSNorm before attention
   - `layer_N_attn_out`: Attention mechanism
   - `layer_N_post_attn_norm_out`: RMSNorm after attention
   - `layer_N_ffn_out`: FFN/MLP
   - `layer_N_out`: Final residual

## Known Issues and Workarounds

### Issue: NPZ loading not yet implemented in C++

**Workaround**: Use Python-side comparison (fully functional):

```bash
python3 python/reference/capture_pytorch_layers.py \
  --llaminar-captures <path_to_llaminar_npz>
```

### Issue: Llaminar pipeline not instrumented for capture

**Workaround**: Manual instrumentation needed in transformer layers:

```cpp
// In transformer layer forward pass
auto& hook = LlaminarSnapshotHook::instance();
if (hook.is_enabled()) {
    // After attention
    hook.capture(
        PipelineStage::ATTENTION_OUTPUT,
        layer_index,
        attention_output.data(),
        seq_len,
        hidden_dim
    );
    
    // After FFN
    hook.capture(
        PipelineStage::FFN_OUTPUT,
        layer_index,
        ffn_output.data(),
        seq_len,
        hidden_dim
    );
}
```

## Example Investigation Workflow

Based on our current divergence investigation:

```bash
# 1. We know embeddings match perfectly
python3 python/reference/investigate_divergence.py --check-embeddings
# ✓ Embedding diff: 0.000000

# 2. Capture layer-by-layer to find where divergence starts
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30

# 3. Expected output will show:
# ✓ embeddings: perfect match
# ✓ layer_0_input_norm_out: perfect match (RMSNorm correct)
# ❌ layer_0_attn_out: DIVERGENCE! (mean_abs_diff ~ 3.83)
#
# Conclusion: Attention mechanism in layer 0 is the root cause

# 4. Next step: Deep-dive into attention components
#    - QKV projection
#    - RoPE application
#    - Attention scores
#    - Softmax
#    - Output projection
```

## Next Steps

Once first diverging layer is identified:

1. **Drill down into components**:
   - Add more granular hooks (Q, K, V separately)
   - Capture before/after RoPE
   - Capture attention scores
   - Capture after softmax

2. **Compare individual operations**:
   - Test RoPE in isolation
   - Test attention mask
   - Test softmax numerical stability

3. **Fix root cause**:
   - Update implementation
   - Re-run layer-by-layer comparison
   - Verify all layers now match

## Conclusion

The layer-by-layer parity framework provides a systematic approach to debugging numerical divergence. By capturing and comparing intermediate outputs at every stage, you can quickly identify the exact point where implementations diverge, making it much easier to fix the root cause.

Current status:
- ✅ Python-side capture and comparison: **FULLY FUNCTIONAL**
- ⏳ C++ NPZ integration: **PLANNED**
- ⏳ Automatic pipeline instrumentation: **PLANNED**

Use the Python workflow now, C++ integration coming soon!
