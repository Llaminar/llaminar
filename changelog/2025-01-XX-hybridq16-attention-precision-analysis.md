# HybridQ16 Attention Precision Analysis

**Date**: January 2025  
**Author**: GitHub Copilot

## Summary

Investigation into why HybridQ16 mode produces worse ATTENTION_CONTEXT parity (0.799 cosine) compared to Hybrid mode (0.897 cosine) despite both using the same Q/K/V inputs.

## Root Cause

HybridQ16 mode **deliberately uses fused Q8_1 attention** (FusedAttentionWoStage), while Hybrid mode uses decomposed FP32 attention (AttentionComputeStage + GEMMStage).

The fused Q8_1 attention has a fundamental precision limitation of ~0.89 cosine similarity due to **softmax amplification of quantization noise** in Q/K/V projections. This is well documented in previous investigations (see `changelog/2025-01-17-q8_1-attention-precision-analysis.md`).

## Evidence

From the parity test (`v2_e2e_parity_hybridq16_vs_fp32_pipeline`):

| Stage | HybridQ16 vs FP32 | Hybrid vs FP32 |
|-------|-------------------|----------------|
| layer0_Q_PROJECTION | 0.999965 | 0.999965 |
| layer0_K_PROJECTION | 0.999972 | 0.999972 |
| layer0_V_PROJECTION | 0.999976 | 0.999976 |
| layer0_ATTENTION_CONTEXT | **0.798975** | **0.897072** |
| layer0_ATTENTION_OUTPUT | -0.074758 | 0.927648 |
| layer0_FFN_NORM | 0.720338 | 0.957796 |

**Key Observations**:
1. Q/K/V projections are **identical** for both modes (0.9999+ cosine)
2. ATTENTION_CONTEXT diverges: HybridQ16=0.799, Hybrid=0.897
3. Error accumulates rapidly: by FFN_NORM, HybridQ16=0.720

## Code Reference

In `src/v2/models/qwen/Qwen2Graph.cpp` (lines 987-989):

```cpp
bool use_fused_wo = env.attention.fused_wo &&
                    ...
                    (config_.activation_precision == ActivationPrecision::Q8_1 ||
                     config_.activation_precision == ActivationPrecision::HybridQ16);
```

This explicitly enables fused Q8_1 attention for HybridQ16 mode.

## Design Rationale

From the comments in `Qwen2Graph.cpp`:

```cpp
// Note: Regular Hybrid mode should use decomposed attention for FP32 attention scoring.
// The fused path quantizes Q to Q8_1 and uses Q8_1 K/V, losing the precision benefit.
// HybridQ16 enables fused path with Q16_1 residual fusion for better precision.
```

HybridQ16 was designed to:
1. Use fused Q8_1 attention for efficiency
2. Use Q16_1 residual stream for better precision in residual accumulation
3. Fuse Wo projection with Q16_1 residual add to eliminate FP32 intermediate memory

The Q16_1 residual fusion happens **after** Wo projection - it doesn't improve attention context precision.

## Options for Improvement

### Option 1: Accept Current Behavior
- HybridQ16's attention is limited by Q8_1 precision
- Q16_1 residual helps with accumulation but not attention
- **Pros**: No code changes, preserves fused efficiency
- **Cons**: Lower attention precision than Hybrid

### Option 2: Use Decomposed FP32 Attention for HybridQ16
- Change `use_fused_wo` condition to exclude HybridQ16
- Implement separate Wo projection with Q16_1 residual fusion
- **Pros**: Better attention precision
- **Cons**: Loses fused Wo projection efficiency, more complex

### Option 3: Implement Higher-Precision Attention
- Use Q16_1 or FP32 for attention computation
- Requires new JIT kernel or decomposed path with Q16_1 fusion
- **Pros**: Best precision
- **Cons**: Significant engineering effort

## Additional Finding: ATTENTION_OUTPUT Snapshot Mismatch

The catastrophic ATTENTION_OUTPUT cosine (-0.075 for HybridQ16 vs 0.928 for Hybrid) is caused by a **snapshot capture issue**, not a computation bug.

### Root Cause

When `fuse_residual_add=true` (HybridQ16 mode):
- `params_.output` points to the Q16_1 **residual buffer**
- After execution, this contains `residual + Wo(context)` (fused result)
- `getDumpInfo()` captures this fused value

For FP32/Hybrid modes:
- `params_.output` points to pure `Wo(context)` output
- No residual fusion

So we're comparing:
- **HybridQ16**: `residual + Wo(context)` 
- **FP32**: `Wo(context)` only

This explains why ATTENTION_OUTPUT shows -0.075 cosine - we're comparing completely different values!

### Impact

The ATTENTION_OUTPUT comparison for HybridQ16 is **meaningless** in the current test. The ATTENTION_CONTEXT comparison (0.799 vs 0.897) is valid and shows the actual attention precision difference.

### Fix Options

1. **Document the limitation**: Note that ATTENTION_OUTPUT in HybridQ16 is the fused residual
2. **Capture pure Wo output**: Add separate buffer for pre-fusion Wo output in JIT kernel
3. **Compare against correct reference**: For HybridQ16, compute `fp32_residual + Wo(context)` as reference

## Recommendation

For now, **document the limitation** and consider it expected behavior for HybridQ16. The mode trades attention precision for:
1. Memory savings (Q16_1 residual vs FP32)
2. Fused Wo projection efficiency

Users who need better attention precision should use Hybrid mode (decomposed FP32 attention).

The ATTENTION_OUTPUT snapshot mismatch should be fixed by either:
1. Excluding ATTENTION_OUTPUT from HybridQ16 parity tests
2. Adding a separate Wo output capture buffer

## Files Involved

### Core Graph Building
- [Qwen2Graph.cpp](src/v2/models/qwen/Qwen2Graph.cpp):
  - Lines 988-991: `use_fused_wo` condition includes HybridQ16 but excludes Hybrid
  - Lines 1060-1065: HybridQ16 sets `fuse_residual_add=true` and `output=buffers.residual`
  - **KEY INSIGHT**: HybridQ16 uses fused Q8_1 attention, Hybrid uses decomposed FP32 attention

### Attention Mode Selection
| Mode | Attention Path | Attention Precision |
|------|---------------|---------------------|
| FP32 | Decomposed (AttentionComputeStage + GEMMStage) | Reference |
| Hybrid | Decomposed (FP32 Q/K/V scoring) | 0.897 cosine |
| HybridQ16 | **Fused Q8_1** (FusedAttentionWoStage) | 0.799 cosine |
| Q8_1 | Fused Q8_1 | ~0.89 cosine |

### Why HybridQ16 Attention is Worse than Hybrid

**Root Cause**: HybridQ16 uses fused Q8_1 attention for efficiency, but Hybrid uses decomposed FP32 attention for precision.

Code evidence (Qwen2Graph.cpp lines 988-991):
```cpp
bool use_fused_wo = env.attention.fused_wo &&
    ...
    (config_.activation_precision == ActivationPrecision::Q8_1 ||
     config_.activation_precision == ActivationPrecision::HybridQ16);  // ← HybridQ16 included
```

Notice that `Hybrid` is NOT in this condition, so it uses the decomposed path (lines 1095+).

## Recommended Architecture Change

To improve HybridQ16 attention precision:

**Option 1: Use Decomposed Attention for HybridQ16**
- Change: Remove `ActivationPrecision::HybridQ16` from `use_fused_wo` condition
- Benefit: HybridQ16 attention would match Hybrid's 0.897 precision
- Trade-off: Loss of fused efficiency (separate AttentionComputeStage + GEMMStage)

**Option 2: Keep Fused Attention but Document Limitation**
- Accept 0.799 attention precision as design trade-off
- Document that HybridQ16 prioritizes memory over attention precision
- Users needing better attention should use Hybrid mode

**Option 3: Create New Mode (HybridQ16Decomposed)**  
- New mode that combines Q16_1 residuals with decomposed FP32 attention
- Best precision + memory savings, but no fused efficiency

Current recommendation: **Option 2** - document the limitation. The current design is intentional for specific use cases.

- `src/v2/models/qwen/Qwen2Graph.cpp`: Mode selection logic (lines 980-1000)
- `src/v2/execution/InferenceMode.h`: Mode definitions
- `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h`: Fused Q8_1 attention
- `tests/v2/e2e/parity/internal/hybridq16_vs_fp32/Test__HybridQ16Pipeline_vs_FP32_LayerByLayer.cpp`: Parity test
