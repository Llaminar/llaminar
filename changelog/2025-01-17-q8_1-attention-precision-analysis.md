# Q8_1 Attention Precision Analysis

**Date**: January 17, 2025  
**Author**: David Sanftenberg (AI Assistant)

## Summary

This document details the investigation into Q8_1 attention divergence from FP32 reference, including root cause analysis, attempted improvements, and recommendations for future work.

## Problem Statement

When running the Q8_1 pipeline against an FP32 reference pipeline with identical prompts, the `ATTENTION_CONTEXT` stage shows significant divergence:

| Stage | Cosine Similarity | Status |
|-------|------------------|--------|
| Q_PROJECTION | 0.999963 | ✓ EXCELLENT |
| K_PROJECTION | 0.999972 | ✓ EXCELLENT |
| V_PROJECTION | 0.999840 | ✓ EXCELLENT |
| **ATTENTION_CONTEXT** | 0.891348 | ⚠ DIVERGING |
| ATTENTION_OUTPUT | 0.929173 | ⚠ DRIFT |

The Q/K/V projection outputs show excellent parity (>0.999), but attention context drops to ~0.89 cosine similarity.

## Root Cause Analysis

### Key Insight: Softmax Amplification

The attention mechanism's softmax operation is highly sensitive to small differences in input scores:

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

Even a 0.1% difference in Q·K scores can lead to significantly different attention weight distributions, especially:
1. When scores are close together (flat distribution)
2. At the edges of the distribution (winner-take-all scenarios)

### Why Q8_1 Attention Diverges

1. **Q/K/V projections are quantized to Q8_1**: 
   - FP32 activation → Q8_1 (quantization noise introduced)
   - Q8_1 × quantized weights → Q8_1 output

2. **Quantization noise is baked into Q/K/V tensors**:
   - Q8_1 Q/K/V ≠ FP32 Q/K/V (they differ by quantization noise)
   - Even perfect dequantization recovers quantized values, not original FP32

3. **Attention computes Q·K^T which amplifies differences**:
   - Small differences in Q/K values → small differences in scores
   - Softmax amplifies small score differences → large weight differences
   - Different weights × V → significantly different context output

### Why Isolated Q8_1 Attention Tests Pass

The isolated attention unit tests show >0.999 cosine similarity because:
- Both JIT and reference implementations receive **identical Q8_1 inputs**
- The attention algorithm itself is mathematically correct
- No comparison to FP32 reference is made

## Attempted Improvement: FP32 Score Computation

### Implementation

Added `LLAMINAR_Q8_ATTENTION_FP32_SCORES=1` environment variable that:
1. Dequantizes Q and K to FP32 before attention
2. Computes Q·K^T in FP32 (higher precision)
3. Applies softmax in FP32
4. Accumulates V in FP32, then requantizes to Q8_1

### Files Modified

- `src/v2/utils/DebugEnv.h`: Added `AttentionConfig` struct with `fp32_scores` flag
- `src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h`: Added `compute_q8_1_fp32_scores()` function

### Result: No Improvement

```
Without FP32 scores: layer0_ATTENTION_CONTEXT = 0.891349
With FP32 scores:    layer0_ATTENTION_CONTEXT = 0.891348
```

### Why It Didn't Help

The FP32 scores mode is mathematically equivalent to the integer path:

**Integer JIT Kernel**:
```
int32_acc = Σ(Q_i8[i] × K_i8[i])
score = int32_acc × d_Q × d_K
```

**FP32 Scores Mode**:
```
Q_fp32[i] = Q_i8[i] × d_Q  (dequantize)
K_fp32[i] = K_i8[i] × d_K  (dequantize)
score = Σ(Q_fp32[i] × K_fp32[i])
      = Σ(Q_i8[i] × d_Q × K_i8[i] × d_K)
      = d_Q × d_K × Σ(Q_i8[i] × K_i8[i])  // Same result!
```

Both paths compute the same score (within floating-point rounding). The quantization noise is in the Q/K/V input tensors, not in how we compute the attention scores.

## Recommendations for Future Work

### Short-term (Incremental Improvements)

1. **Temperature Scaling for Attention**:
   - Apply smaller softmax temperature (e.g., `score / sqrt(head_dim) / temperature`)
   - Reduces softmax sensitivity to small score differences
   - Tradeoff: May affect model quality/behavior

2. **Attention Score Normalization**:
   - L2-normalize Q and K before computing Q·K^T
   - Reduces variance in score magnitudes
   - Common technique in some attention variants (cosine attention)

### Medium-term (Significant Changes)

3. **Q16 Activation Precision**:
   - Use 16-bit quantization for Q/K/V projections
   - Double the precision, halve the quantization noise
   - Requires new VNNI-based kernels (vpdpbusd only handles 8-bit)

4. **Per-Channel Quantization**:
   - Current: Single scale per block (32 elements)
   - Improved: Scale per row or per attention head
   - Better handling of outlier values

### Long-term (Research Direction)

5. **Mixed-Precision Attention**:
   - Keep Q/K projections in FP16/BF16
   - Use Q8_1 only for V accumulation (larger, more memory-bound)
   - Balances accuracy and memory bandwidth

6. **SmoothQuant-style Scaling**:
   - Migrate quantization difficulty from activations to weights
   - Pre-compute migration factors during model loading
   - Requires changes to weight loading and GEMM kernels

## Code Location

- **Attention kernel**: `src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h`
- **JIT fused attention**: `src/v2/kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h`
- **Q8_1 tensor**: `src/v2/tensors/Q8_1Tensor.cpp`
- **Divergence test**: `tests/v2/e2e/qwen2/Test__Q8_1_LayerByLayer_Divergence.cpp`

## Test Commands

```bash
# Run Q8_1 vs FP32 layer divergence test
export LLAMINAR_LOG_LEVEL=INFO
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2/tests/v2/v2_test_q8_1_layer_divergence \
  --gtest_filter="Test__Q8_1_LayerByLayer.SnapshotComparison"

# Enable FP32 scores mode (for comparison - no improvement expected)
export LLAMINAR_Q8_ATTENTION_FP32_SCORES=1
```

## Conclusion

The Q8_1 attention divergence is fundamentally caused by quantization noise in the Q/K/V projection outputs, not by the attention kernel implementation. The softmax operation amplifies small score differences caused by this quantization noise.

Improving accuracy while maintaining integer-domain inference requires either:
1. Higher precision activations (Q16) for attention inputs
2. Per-channel/per-head quantization for better outlier handling
3. Mixed-precision approaches (FP16 Q/K, Q8_1 V)

The current integer JIT attention kernel is mathematically correct and efficient. Any accuracy improvements must come from changes to how activations are quantized before attention.
