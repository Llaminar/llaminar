# Q8_1 Pipeline Divergence Handover Guide

**Date:** 2025-12-12  
**Updated:** 2025-01-17  
**Topic:** Debugging and improving Q8_1 pipeline accuracy vs FP32 ground truth

## 1. Current Status

We have successfully established a baseline for Q8_1 inference accuracy.
- **Major Fix Implemented:** We identified that `RMSNorm` gamma weights in Qwen2.5 models can exceed the range `[-8, 8]`. The previous Q12 fixed-point format (scale 4096) clipped these values, causing massive divergence. We switched to **Q11 format** (scale 2048) in `RMSNormPrimitives.cpp`, which supports `[-16, 16]`.
- **Test Status:** The `v2_test_q8_1_layer_divergence` test now passes.
  - **Logit Cosine Similarity:** ~0.91 (Threshold is 0.90).
  - **Layer 0-5 Similarity:** > 0.99.

### Update (2025-01-17): Attention Divergence Root Cause Analysis

Investigated the `ATTENTION_CONTEXT` divergence (~0.89 cosine at layer 0 while Q/K/V projections are >0.999):

**Key Finding:** The attention kernel is mathematically correct. The divergence is caused by **softmax amplification of quantization noise** in Q/K/V projections. 

- Implemented `LLAMINAR_Q8_ATTENTION_FP32_SCORES=1` mode (dequantizes Q/K for score computation)
- Result: No improvement - mathematically equivalent to integer path
- Root cause: Quantization noise is in Q/K/V *inputs*, not attention computation

**See:** `changelog/2025-01-17-q8_1-attention-precision-analysis.md` for full analysis.

**Future Improvements Require:**
1. Higher precision activations (Q16) for attention Q/K/V
2. Per-channel/per-head quantization for better outlier handling  
3. Mixed-precision approaches (FP16 Q/K, Q8_1 V)

---

## 2. The "Dual Pipeline" Test

We created a specialized test that runs **two pipelines simultaneously** in the same process:
1.  **FP32 Pipeline:** Reference implementation (unquantized activations).
2.  **Q8_1 Pipeline:** The pipeline under test (quantized activations).

**Test File:** `tests/v2/e2e/Test__Q8_1_LayerDivergence.cpp`

**How to Run:**
```bash
# Build the E2E test suite
cmake --build build_v2_e2e --parallel

# Run the divergence test
# Note: Requires 2 MPI ranks as per standard E2E setup
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_e2e/tests/v2/v2_test_q8_1_layer_divergence \
  --gtest_filter="Test__Q8_1_LayerByLayer.SnapshotComparison"
```

---

## 3. Debugging with Tensor Dumps

If the test fails or you need to trace a divergence, use the **Tensor Dump** feature. This dumps raw binary tensors from *both* pipelines to disk.

### Configuration
Set these environment variables before running the test:

```bash
# Enable dumping
export LLAMINAR_SNAPSHOT_TENSOR_DUMP=1

# Output directory (will be created)
export LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer_debug

# Which layers to dump? (comma separated or "all")
# Start with the first failing layer reported by the test logs.
export LLAMINAR_SNAPSHOT_DUMP_LAYERS=3

# Which stages? (comma separated or "all")
# "all" is recommended to trace within the layer.
export LLAMINAR_SNAPSHOT_DUMP_STAGES=all

# Run the test
mpirun -np 2 ... ./build_v2_e2e/tests/v2/v2_test_q8_1_layer_divergence ...
```

### Output Format
Files are named to distinguish the pipeline source:
- `fp32_pipeline_layer3_FFN_NORM_rank0_fp32.bin`
- `q8_1_pipeline_layer3_FFN_NORM_rank0_fp32.bin`

---

## 4. Analysis Workflow (Python)

Use this Python script pattern to find the exact operation causing divergence.

**Step 1: Find the "Cliff"**
Compare the output of each stage in the layer. Find where the error jumps.

```python
import numpy as np
import os

dump_dir = "/tmp/layer_debug"
layer = 3
stages = [
    "ATTENTION_NORM", "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
    "FFN_NORM", "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN", "FFN_RESIDUAL"
]

print(f"--- Analyzing Layer {layer} ---")
for stage in stages:
    fp32_path = f"{dump_dir}/fp32_pipeline_layer{layer}_{stage}_rank0_fp32.bin"
    q8_1_path = f"{dump_dir}/q8_1_pipeline_layer{layer}_{stage}_rank0_fp32.bin"
    
    if not os.path.exists(fp32_path): continue
    
    # Load data
    fp32 = np.fromfile(fp32_path, dtype=np.float32)
    q8_1 = np.fromfile(q8_1_path, dtype=np.float32)
    
    # Calculate error
    diff = np.abs(fp32 - q8_1)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"{stage:20} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f}")
```

**Step 2: Trace the Element**
Once you find the stage (e.g., `FFN_NORM`), find the index of the maximum error and check the input values at that index.

```python
# Example: Deep dive into FFN_NORM
stage = "FFN_NORM"
fp32 = np.fromfile(f"{dump_dir}/fp32_pipeline_layer{layer}_{stage}_rank0_fp32.bin", dtype=np.float32)
q8_1 = np.fromfile(f"{dump_dir}/q8_1_pipeline_layer{layer}_{stage}_rank0_fp32.bin", dtype=np.float32)

# Find worst offender
idx = np.argmax(np.abs(fp32 - q8_1))
print(f"Worst index: {idx}")
print(f"FP32 value: {fp32[idx]}")
print(f"Q8_1 value: {q8_1[idx]}")

# Now check the INPUT to this stage (e.g., ATTENTION_RESIDUAL) at the same index
# to see if the error existed before, or was created here.
prev_stage = "ATTENTION_RESIDUAL"
# ... load prev_stage files ...
print(f"Input FP32: {fp32_prev[idx]}")
print(f"Input Q8_1: {q8_1_prev[idx]}")
```

## 5. Key Code Locations

- **`src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp`**:
  - Contains the AVX512 implementation of RMSNorm.
  - Look for `rmsnorm_q8_1_pure_integer_row`.
  - We use `vscale = _mm512_set1_ps(2048.0f)` (Q11) to allow weights up to +/- 16.

- **`src/v2/pipelines/PipelineBase.cpp`**:
  - `maybeDumpTensor()`: Handles the file writing.
  - Note the prefix logic: `(is_q8_1_ ? "q8_1_pipeline_" : "fp32_pipeline_")`.

- **`src/v2/tensors/Q8_1Tensor.cpp`**:
  - Handles the quantization/dequantization logic for the activations.
