# Q8_1 ATTENTION_CONTEXT Divergence Investigation - Handover Document

**Date**: December 10, 2025  
**Branch**: `feature/typed-residuals`  
**Author**: GitHub Copilot  
**Status**: Ready for investigation

---

## Executive Summary

The Q8_1 attention kernel produces divergent output starting at `layer0_ATTENTION_CONTEXT` despite **all preceding stages being excellent** (≥0.999 cosine similarity). This indicates the divergence is **NOT accumulated quantization noise** - it's a specific bug in the attention computation itself.

### Critical Observation

```
║   1 │ EMBEDDING                                │ 0.999987 │ 0.005314 │ ✓ EXCELLENT    ║
║   2 │ layer0_ATTENTION_NORM                    │ 0.999964 │ 0.025192 │ ✓ EXCELLENT    ║
║   3 │ layer0_Q_PROJECTION                      │ 0.999963 │ 0.008629 │ ✓ EXCELLENT    ║
║   4 │ layer0_Q_ROPE                            │ 0.999782 │ 0.020860 │ ✓ EXCELLENT    ║
║   5 │ layer0_K_PROJECTION                      │ 0.999972 │ 0.007508 │ ✓ EXCELLENT    ║
║   6 │ layer0_K_ROPE                            │ 0.999835 │ 0.019419 │ ✓ EXCELLENT    ║
║   7 │ layer0_V_PROJECTION                      │ 0.999845 │ 0.027604 │ ✓ EXCELLENT    ║
║   8 │ layer0_ATTENTION_CONTEXT                 │ 0.829681 │ 0.605168 │ ⚠ DIVERGING    ║  <-- HERE!
║   9 │ layer0_ATTENTION_OUTPUT                  │ 0.908483 │ 0.435256 │ ⚠ DRIFT        ║
```

**Key Insight**: Q, K, V projections are all ≥0.9997 cosine similarity, yet ATTENTION_CONTEXT drops to 0.829. The bug is in the attention computation logic (softmax(Q×K^T) × V), not in quantization error accumulation.

---

## Previous Bug Fix: Q/K Normalization

### The Bug (Fixed in Previous Session)

The Q8_1 JIT attention kernel (`QuantisedAttentionJit_Q8_1_Fused.h`) was applying **Q/K normalization** that the FP32 path does NOT use:

```cpp
// Bug: The JIT kernel was doing:
score = (Q · K) * scale / (|Q| * |K|)  // WRONG - Q/K normalization

// FP32 standard attention does:
score = (Q · K) * scale                 // CORRECT - just scaled dot product
```

### The Fix

Q/K normalization code was REMOVED from the JIT kernel. The code now says:

```cpp
// --------------------------------------------------------
// Section 6b: [REMOVED] Q/K normalization removed for standard attention
// Standard attention uses: score = (Q · K) * scale
// NOT: score = (Q · K) * scale / (|Q| * |K|)
// --------------------------------------------------------
```

This fix improved parity from ~0.60 cosine to ~0.96 cosine in isolated replay tests, but the full E2E test still shows divergence at ATTENTION_CONTEXT.

---

## Current Test Status

### E2E Test Results (December 10, 2025)

```
SUMMARY:
  Total stages:  363    Good (≥0.99):   56    Diverged (<0.90):  182
  Average cosine: 0.848696
  Best:  EMBEDDING                           (cos=1.0000)
  Worst: layer22_ATTENTION_RESIDUAL          (cos=-0.3956)

FINAL LOGITS:
  Cosine Similarity: 0.876093    Rel L2: 0.483422
  FP32 Top-5: [11, 13, 624, 382, 345]
  Q8_1 Top-5: [13, 624, 382, 11, 271]
  Top-5 Overlap: 4/5 (80%)
  ✗ TOP-1 MISMATCH: FP32=11, Q8_1=13
```

The test correctly **FAILS** because:
1. Top-1 token mismatch (FP32 predicts 11, Q8_1 predicts 13)
2. Logits cosine (0.876) is below threshold (0.90)

---

## How to Run the E2E Test

### Build (E2E Release with Snapshot Support)

```bash
cd /workspaces/llaminar

# Build with snapshot capture enabled (required for E2E tests)
cmake -B build_v2_e2e_release -S src/v2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PIPELINE_SNAPSHOTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build_v2_e2e_release --target v2_test_q8_1_layer_divergence --parallel
```

### Run the Test

```bash
cd /workspaces/llaminar

# Set optimal MPI/OpenMP environment
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close \
       OMP_NESTED=false OMP_DYNAMIC=false \
       LLAMINAR_LOG_LEVEL=INFO

# Run the E2E divergence test (outputs stage-by-stage table)
timeout 180 mpirun -np 1 --bind-to socket --map-by socket \
    ./build_v2_e2e_release/tests/v2/v2_test_q8_1_layer_divergence \
    --gtest_filter="Test__Q8_1_LayerByLayer.SnapshotComparison"
```

The test outputs a formatted table showing all 363 stages in execution order, with cosine similarity and status for each.

---

## Files Involved

### Core JIT Kernel (Primary Investigation Target)

```
src/v2/kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h
```
- **Lines 1-100**: Header documentation and FusedQ8_1AttentionParams struct
- **Lines 430-550**: Score computation loop (Q · K dot product)
- **Lines 600-760**: Online softmax and V accumulation
- **Lines 441-446**: Q/K normalization removal marker

### Attention Kernel Dispatcher

```
src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h
```
- **Lines 400-520**: `compute_q8_1_native()` - calls the JIT kernel
- Sets up `FusedQ8_1AttentionParams` and dispatches to JIT

### E2E Test File

```
tests/v2/e2e/qwen2/Test__Q8_1_LayerByLayer_Divergence.cpp
```
- **SnapshotComparison test**: Runs both FP32 and Q8_1 pipelines with snapshot capture
- Compares all 363 stages and outputs formatted table
- Strict pass criteria: Top-1 match required, ≥80% top-5 overlap, ≥0.90 cosine

### Integration Test File

```
tests/v2/integration/Test__Q8_1_vs_FP32_Parity.cpp
```
- **PrefillParity**: Single forward pass comparison
- **IncrementalDecodeParity**: Decode step comparison
- **GreedySamplingSequence**: Multi-token generation comparison
- Uses same strict thresholds as E2E test

### Pipeline Files

```
src/v2/pipelines/qwen/Qwen2Pipeline.cpp
src/v2/pipelines/PipelineBase.cpp
```
- `attention_block()` - orchestrates attention computation
- `compute_attention_with_kv_cache()` - dispatches to CPUAttentionKernelTyped

---

## What the JIT Kernel Does

The `QuantisedAttentionJit_Q8_1_Fused` kernel computes attention in a single fused pass:

```
output[m] = softmax(Q[m] @ K^T * scale) @ V  →  Q8_1
```

### Key Algorithm Steps

1. **Load Q row**: Dequantize Q8_1 blocks for one query position
2. **Stream K/V**: For each K/V position:
   - Compute score = dot(Q, K) * scale (using VNNI vpdpbusd)
   - Online softmax: Update running max/sum
   - Accumulate weighted V: context += weight * V
3. **Finalize**: Divide by sum, requantize to Q8_1 output

### Register Allocation (AVX-512)

```cpp
ZMM0-3:   Context accumulators (head_dim=64: 4×16 floats)
ZMM4-9:   Scratch for Q/K/V blocks
ZMM10:    Current score (scalar broadcast)
ZMM11:    Current weight (exp(s - max))
ZMM12:    Running max
ZMM13:    Running sum
ZMM14:    Correction factor
ZMM26:    Constant: 128 (unsigned conversion)
ZMM27:    Constant: attention scale
ZMM28:    Constant: -inf
ZMM29:    Constant: 1.0f
```

---

## Potential Bug Areas to Investigate

### 1. Online Softmax Numerical Stability

The online softmax uses rescaling when a new max is found:

```cpp
// When s > max:
correction = exp(max_old - max_new)
sum *= correction
context_accum *= correction  // Rescale all head_dim values
max = s
```

Check if:
- Correction factor computation is correct
- Context accumulator rescaling is applied to all blocks (including spilled ones)
- FP32 precision loss during rescaling

### 2. V Dequantization During Accumulation

V blocks are dequantized on-the-fly during the K loop:

```cpp
context_accum += weight * dequant(V[n])
```

Check if:
- V dequantization uses correct scales (d_V, sum_qs_V)
- Signed/unsigned conversion is handled correctly
- Block stride calculations are correct

### 3. Sum_qs Compensation Term

Q8_1 format includes sum_qs for zero-point compensation:

```cpp
// Q8_1 block: { d: fp16, sum_qs: int16, qs[32]: int8 }
// Dot product correction: result -= d_A * sum_qs_B * 128 + d_B * sum_qs_A * 128
```

Check if:
- Compensation is applied correctly in score computation
- Signs are correct (Q is converted to unsigned, K remains signed)

### 4. Mask Application

The JIT kernel applies causal mask:

```cpp
if (mask[m * mask_stride + n] == -inf)
    skip this position
```

Check if:
- Mask stride is correct
- Mask is applied BEFORE softmax (not after)

### 5. Output Requantization

Final context is requantized to Q8_1:

```cpp
context[m] = context_accum / sum
output[m] = quantize_to_q8_1(context[m])
```

Check if:
- Division by sum is correct
- Quantization scale computation is correct
- Output block stride is correct

---

## Debugging Strategies

### 1. Single-Head Isolated Test

Run the JIT kernel on a single head with known inputs:

```cpp
// In Test__Q8_1_LayerByLayer_Divergence.cpp
// Extract Q, K, V for head 0 only
// Run both FP32 reference and JIT
// Compare output element-by-element
```

### 2. Add Debug Logging to JIT

The kernel has `debug_generate_` flag. Enable it:

```cpp
jit_fused_64 = std::make_unique<gemm_v4::QuantisedAttentionJit_Q8_1_Fused>(64, true);
```

### 3. Compare Intermediate Values

Add snapshot points inside the JIT kernel for:
- Score values before softmax
- Softmax weights
- Context accumulator before final division

### 4. Run Reference Implementation

The kernel has a reference implementation fallback. Force it:

```cpp
// In CPUAttentionKernelTyped.h, make jit_kernel = nullptr
// This will use fused_q8_1_attention_reference()
```

---

## Test Configuration

```cpp
// Test prompt: 9 tokens "The quick brown fox jumps over the lazy dog"
static const std::vector<int> TEST_TOKEN_IDS = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

// Model: qwen2.5-0.5b-instruct-q4_0.gguf
// 24 layers, 14 heads, 2 KV heads, head_dim=64, d_model=896

// Pass Criteria (strict):
// - Top-1 token MUST match
// - Top-5 overlap ≥ 80%
// - Cosine similarity ≥ 0.90
```

---

## Related Changelogs

- `2025-12-09-q8_1-parity-investigation-handover.md` - Previous investigation (buffer issues)
- `2025-12-09-jit-attention-mask-fix.md` - Mask handling fixes
- `2025-12-08-q8-1-vs-fp32-parity-analysis.md` - Initial parity analysis
- `2025-12-07-q8_1-gemm-jit-tuning.md` - JIT kernel optimizations

---

## Summary

The Q8_1 attention kernel has a bug that causes divergence at ATTENTION_CONTEXT even when all inputs (Q, K, V) are excellent (≥0.9997 cosine). The bug is NOT accumulated quantization noise - it's in the attention computation logic itself.

**Primary Investigation Target**: `QuantisedAttentionJit_Q8_1_Fused.h`

**Most Likely Bug Areas**:
1. Online softmax rescaling (correction factor application)
2. V dequantization during accumulation
3. Sum_qs compensation term in dot product
4. Output requantization

Good luck!
