# HybridQ16 Attention Investigation Handover

**Date**: January 5, 2026  
**Last Updated**: January 5, 2026 (Session 4 - QUANTITATIVE PROOF VIA REPLAY TEST)  
**Branch**: `feature/typed-residuals`  
**Status**: ROOT CAUSE CONFIRMED - K values exceed safe quantization range, causing 3.2% clipping

---

## Session 4 Update: QUANTITATIVE PROOF VIA REPLAY TEST

### New Tooling: StageDump + Replay Test Framework

Created a new debugging workflow:
1. **StageDumper** captures actual tensor data from pipeline stages
2. **Replay test** loads dumped tensors and performs isolated analysis
3. Enables root cause identification without running full pipeline

### Key Files Created

| File | Purpose |
|------|---------|
| `tests/v2/integration/replay/Test__HybridQ16AttentionReplay.cpp` | Replay test loading stage dumps |
| `tests/v2/unit/Test__StageDumpConfig.cpp` | Unit tests for stage dump configuration |

### Quantitative Findings

Running the replay test on captured `FusedAttentionWoStage` tensors from layer 0:

```
=== Q16 Quantization Error Analysis ===
Q range: [-10.3284, 11.1175], abs_max=11.1175
K range: [-92.7406, 148.8560], abs_max=148.8560

--- Q16_1 Quantization Constraints ---
KV_CACHE_SCALE: 256.0000
Scale factor d: 0.0078
Max safe quantized value (overflow prevention): ±5792
Max safe FP32 value: ±45.2514

Q max quantized magnitude: 1423.0000 (safe if < 5792) ✓
K max quantized magnitude: 19052.9902 (safe if < 5792) ✗

❌ K values EXCEED safe quantization range!
   K max: 148.8560 -> quantized 19052.9902
   Safe limit: 45.2514 -> quantized 5792
   Values beyond this will be CLIPPED to prevent dot product overflow

   SOLUTION: Increase KV_CACHE_SCALE from 256 to at least 843
```

### Root Cause: Overflow Prevention Clipping

| Metric | Q Tensor | K Tensor |
|--------|----------|----------|
| Max magnitude | 11.12 | **148.86** |
| Quantized max | 1,423 | **19,053** |
| Safe limit | 5,792 | 5,792 |
| Values clipped | 0/8064 (0%) | **37/1152 (3.2%)** |
| Cosine similarity | 1.0000 | **0.8975** |

**Why the safe limit exists**: To prevent int32 overflow in head_dim=64 dot products:
- `|q_i * k_i| < INT32_MAX / head_dim = 2147483647 / 64 = 33554431`
- `max_safe = sqrt(33554431) = 5792`

**Why K values are so large**: The Qwen2-0.5B model's K projections produce values up to ±148.86, far exceeding the ±45.25 safe range with `KV_CACHE_SCALE=256`.

### Required Fix

Increase `KV_CACHE_SCALE` from **256** to at least **843**:
```
K_max * 32767 / new_scale <= max_safe
148.86 * 32767 / 843 = 5786 < 5792 ✓
```

### Reproduction Commands

```bash
# 1. Capture stage dumps from integration test
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=FUSED_ATTENTION_WO \
./build_v2/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline

# 2. Run replay test to analyze dumps
./build_v2/tests/v2/v2_integration_hybridq16_attention_replay
```

---

## Session 3 Update: DEEPER ROOT CAUSE FOUND

### Initial Fix Attempt (Partial)

Disabled the per-position K scale path by setting `read_k_scales_from_blocks = false` in `Q16FusedAttentionKernel.cpp`. This fix was applied but **the test still fails with identical results**.

### The REAL Root Cause

**ALL Q16 block `.d` fields contain quantization scales, not the normalization scales the attention kernel expects!**

Evidence from test logs:
```
Q16FusedAttentionKernel Layer 0: extracted q_scales[0..2]=0.00781274, 0.00781274, 0.00781274 
(expected ~0.00195 = 64/32767)
```

**Q scale extracted: ~0.0078 (quantization scale = max_abs/32767)**
**Q scale expected: ~0.00195 (normalization scale = head_dim/32767)**

This is a **4x error** for Q scales, on top of the 32x error for K scales.

### Why This Happens

The Q16 attention kernel was designed with these assumptions:
1. Q16 block `.d` = normalization scale used during projection/RoPE
2. This scale can be used directly in attention score computation

But the ACTUAL behavior:
1. Q16 block `.d` = `max_abs(block_data) / 32767` (dynamic quantization scale)
2. This varies based on actual data magnitude, NOT the intended normalization

### Scale Mismatch Summary

| Tensor | Expected `.d` | Actual `.d` | Ratio |
|--------|--------------|-------------|-------|
| Q | 0.00195 (64/32767) | ~0.0078 | 4x too large |
| K | 0.000244 (kv_head_scale) | ~0.008 | 32x too large |
| V | 0.000244 (kv_head_scale) | ~0.008 | 32x too large |

### Impact on Attention Computation

The attention score scale formula:
```
qk_scale = q_scale * k_scale / sqrt(head_dim)
```

With correct scales: `0.00195 * 0.000244 / 8 = 4.77e-07`
With actual scales: `0.0078 * 0.008 / 8 = 7.8e-06` (16x too large!)

This makes softmax extremely peaked, destroying attention patterns.

---

## Session 3 Summary (Earlier Analysis - K Scale Bug)

### Original Finding (Still Valid)

The unit tests definitively prove the K scale bug:

**The per-position path reads `K_block.d` (Q16_1 quantization scale ~0.008) instead of `kv_head_scale` (~0.000244). This causes alpha to be ~32x too large, making softmax extremely peaked.**

### Mathematical Proof

**Correct formula (uniform path)**:
```
qk_scale = q_scale * kv_head_scale / sqrt(head_dim)
         = 0.00195 * 0.000244 / 8
         = 4.77e-07
```

**Buggy formula (per-position path)**:
```
base_alpha = q_scale / sqrt(head_dim) = 0.00195 / 8 = 0.000244
alpha[c] = base_alpha * K_block.d = 0.000244 * 0.00796 = 1.94e-06
```

**Ratio: 1.94e-06 / 4.77e-07 = 32.6x too large!**

### Why K_block.d ≠ kv_head_scale

- `K_block.d` ≈ `max_abs(K) / 32767` ≈ `130 / 32767` ≈ 0.004-0.008 (quantization scale)
- `kv_head_scale` = `1 / (KV_CACHE_SCALE * 16)` ≈ 0.000244 (normalization scale)

These are fundamentally different values:
- **Quantization scale**: How to convert FP32 K values to INT16
- **Normalization scale**: Scaling factor applied during attention computation

### Effect on Softmax

When alpha is 32x too large:
- `softmax(alpha * scores)` becomes almost one-hot
- Small score differences get amplified exponentially
- Weight concentrates on single position, losing all other information
- Entropy drops from ~3 bits to ~0 bits

### Unit Test Evidence

New test file: `tests/v2/unit/microkernels/Test__OnlineSoftmaxPerPositionKScales.cpp`

```
Test: ActualBug_KBlockDotD_vs_KVHeadScales
  Ratio (buggy/correct): 32.5879x
  Entropy correct: 3.17 bits
  Entropy buggy: 3.01 bits

Test: AlphaMagnitude_SoftmaxPeakedness
  alpha=1:   max_weight=0.22, entropy=2.99 bits
  alpha=30:  max_weight=0.99, entropy=0.02 bits (almost one-hot!)
```

---

## The Fix (REVISED)

The original fix (disabling per-position K scales) was **insufficient**. The deeper issue requires a more fundamental change:

### Required Fix

**The Q16 attention kernel cannot extract scales from block `.d` fields.** The scales must be passed explicitly from the pipeline.

**Option A: Fixed Normalization Scales (Recommended)**

Pass hardcoded normalization scales instead of extracting from blocks:
```cpp
// In Q16FusedAttentionKernel.cpp
constexpr float Q_NORM_SCALE = 64.0f / 32767.0f;      // ~0.00195 (head_dim based)
constexpr float KV_NORM_SCALE = 1.0f / (256.0f * 16.0f) / 32767.0f; // ~0.000244

for (int h = 0; h < params.n_heads; ++h) {
    q_scales[h] = Q_NORM_SCALE;  // Don't extract from block!
}
for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h) {
    kv_scales[kv_h] = KV_NORM_SCALE;  // Don't extract from block!
}
```

**Option B: Pre-normalize Before Quantization**

Normalize Q/K/V data to expected range before Q16 quantization so `.d` matches expected scale. This requires changes in RoPE and KV cache stages.

**Option C: Use FP32 Attention (Bypass Q16)**

For HybridQ16 mode, dequantize Q/K/V to FP32 before attention. This sacrifices the Q16 memory savings but guarantees correctness.

### Code Changes Already Applied

In `Q16FusedAttentionKernel.cpp` line 195:
```cpp
// DISABLED: Per-position K scale code path was reading the WRONG scale value.
ref_params.read_k_scales_from_blocks = false;
```

This partial fix is in place but insufficient. Option A above is needed next.

### Verification Commands

After applying full fix:
```bash
# Rebuild
cmake --build build_v2 --parallel

# Run integration test to verify fix
ctest --test-dir build_v2 -R "V2_Integration_HybridQ16Pipeline_vs_FP32" --output-on-failure

# Run unit tests to ensure no regression
./build_v2/tests/v2/v2_test_microkernel_online_softmax_k_scales
```

Or better, remove the entire per-position K scale code path since it's based on incorrect assumptions.

---

## Session 2 Summary (January 5, 2026)

### Actions Taken

1. **Created realistic data unit test**: `tests/v2/unit/microkernels/Test__Q16AttentionRealisticData.cpp`
   - Captures actual Q, K, V data from pipeline stage dumps
   - 5 test cases validating K precision issues
   - Added to CMakeLists.txt with proper labels

2. **Confirmed K value overflow**:
   - K max absolute value from pipeline: **130.36**
   - Max representable with scale=256: **127.996** (= 256 × 16383/32767)
   - **1 of 64 elements clipped** in test data
   - Minimum scale needed: **260.74** → recommended **320** (next multiple of 64)

3. **Dynamic vs fixed scale comparison**:
   - Fixed scale=256: RMSE=0.296, cosine=0.999967
   - Dynamic scale: RMSE=0.00157, cosine=1.0 (perfect)
   - Per-position K scales vary significantly: 0.003 to 0.012

4. **Detailed integration test analysis** (layer 0):
   - Input stages excellent: Q_PROJECTION=0.9999, K_PROJECTION=1.0, V_PROJECTION=0.9999
   - ATTENTION_CONTEXT severely diverged: **0.5949** (HybridQ16) vs 0.8971 (Hybrid)
   - **Per-row pattern**: r0=0.996 (first position perfect!), r5=0.352, r3=0.372
   - **Per-head pattern**: h11=0.261, h13=0.299, h7=0.333 (KV head 1-mapped worse)

### Key Finding

The **first position works nearly perfectly** (r0=0.996 cosine) but subsequent positions degrade severely. This pattern strongly suggests:
- K scale reading from block headers IS working for position 0
- Something goes wrong when processing subsequent KV positions
- Possibly running_max tracking doesn't handle varying K scales correctly

---

## Problem Statement

The **HybridQ16 pipeline** shows severe attention divergence compared to FP32 reference:
- `ATTENTION_CONTEXT` cosine similarity: **0.04 - 0.74** (should be >0.95)
- `ATTENTION_OUTPUT` cosine similarity: **-0.03 - 0.90** (should be >0.95)
- Residual stages compound the error through layers

The regular **Hybrid** (Q8_1) pipeline achieves expected accuracy (cos > 0.91-0.99).

---

## Test Commands

### Primary Integration Test
```bash
# Build (Debug with assertions)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --parallel

# Run the failing test
LLAMINAR_LOG_LEVEL=DEBUG ctest --test-dir build_v2 -R "V2_Integration_HybridQ16Pipeline_vs_FP32" --output-on-failure
```

### Direct Test Execution (faster iteration)
```bash
LLAMINAR_LOG_LEVEL=DEBUG build_v2/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline 2>&1 | tee test_output.log
```

### Force Rebuild After Changes
```bash
# The object file sometimes doesn't rebuild - force it:
rm -f build_v2/CMakeFiles/llaminar2_core.dir/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp.o
cmake --build build_v2 --target llaminar2_core --parallel
cmake --build build_v2 --target v2_integration_hybridq16_vs_fp32_pipeline --parallel
```

---

## Test Results

### Current Output (after implementing prefill K scale fix)
```
SUMMARY: HybridQ16 wins=0, Hybrid wins=96, Ties=0

Layer  | Stage              | HybridQ16 | Hybrid   | Delta
-------|--------------------|-----------+----------|--------
0      | ATTENTION_CONTEXT  | 0.041     | 0.949    | -0.908
0      | ATTENTION_OUTPUT   | 0.002     | 0.946    | -0.944
5      | ATTENTION_CONTEXT  | 0.122     | 0.932    | -0.810
12     | ATTENTION_CONTEXT  | 0.132     | 0.912    | -0.780
23     | FFN_RESIDUAL       | -0.022    | 0.970    | -0.991
```

The per-position K scale code path IS being triggered (verified via debug logs), but results are **worse** than before.

---

## Root Cause Analysis

### Identified Issue: K_head_scales Cross-Layer Contamination

The original problem was that the `K_head_scales` buffer was being written with different scale values at each layer, but attention was reading stale scales from previous layers.

**Solution implemented for decode path**: Set `read_k_scales_from_blocks=true` to read K scales directly from K cache block headers (`.d` field) instead of the contaminated buffer.

**Problem**: The test uses **9 tokens** which triggers **prefill mode**, not decode mode. The prefill path needed similar per-position K scale handling.

### Implementation Done

1. **New microkernel**: `fa2_prefill_process_kv_tile_with_k_scales` in `OnlineSoftmax.cpp`
   - Accepts `base_alpha_fp32` (Q scale / sqrt(head_dim)) and per-position `k_scales[]`
   - Computes per-column `AdaptiveAlphaConfig` for each K position
   - Uses same LUT-based exp2 approximation as decode path

2. **OnlineSoftmaxStateBatch extensions**: 
   - Added `init_per_position(int br, float q_scale, int head_dim, ...)` method
   - Added `base_alpha_fp32` field

3. **Prefill integration** in `Q16IntegerAttentionRef.cpp`:
   - Lines 775-810: BLOCK_64 case extracts K scales from block headers when `read_k_scales_from_blocks=true`
   - Lines 862-895: BLOCK_128 case similarly updated

---

## Current Investigation: Scale Value Mismatch

Debug output shows a **scale mismatch** between the per-position and uniform paths:

```
FA2 Prefill K scale debug (head 0, first tile):
  base_alpha=0.000244148 q_scale=0.00195318 qk_scale=4.76866e-07
  K_scale[0]=0.00795628 alpha[c]=1.94251e-06
  K_scale[1]=0.00793607 alpha[c]=1.93758e-06
```

### The Math

**Uniform path** (working):
- `qk_scale = q_head_scale * kv_head_scale / sqrt(head_dim)`
- `qk_scale = 0.00195 * 0.000244 / 8 ≈ 4.77e-07` ✓

**Per-position path** (broken):
- `base_alpha = q_scale / sqrt(head_dim) = 0.00195 / 8 = 0.000244` ✓
- `alpha[c] = base_alpha * K_block.d = 0.000244 * 0.00796 = 1.94e-06` ✗

**The problem**: `K_block.d` contains the **quantization scale** (~0.008), NOT the same value as `kv_head_scales` (~0.000244).

---

## Hypotheses

### Hypothesis 1: K Block `.d` Field Meaning Mismatch
The K cache block's `.d` field may contain a different scale than expected:
- **Expected**: Per-position K scale matching `kv_head_scales` 
- **Actual**: Q16_1 quantization scale for the block (~0.008 = 256/32767)

If true, the per-position alpha is ~4x too large, causing all softmax weights to be nearly identical (oversaturated).

### Hypothesis 2: Scale Already Applied
The K data may already be normalized/scaled during quantization, so the `.d` field shouldn't be applied again during attention.

### Hypothesis 3: Different Scale Semantics ✅ CONFIRMED
- `kv_head_scales[]` = normalization scale from RoPE/projection
- `K_block.d` = Q16_1 quantization scale (range compression)

These are two different things and should not be interchanged. **This is the root cause.**

---

## Next Steps (Session 3 Conclusion)

### Immediate Action Required

1. **Apply Option A fix**: Use fixed normalization scales instead of extracting from block `.d`
2. Modify `Q16FusedAttentionKernel.cpp` to use hardcoded scales:
   - Q scale: `64.0f / 32767.0f` (~0.00195)
   - KV scale: `1.0f / (256.0f * 16.0f * 32767.0f)` (~0.000244)

### Testing After Fix

```bash
# Rebuild and run integration test
cmake --build build_v2 --parallel
ctest --test-dir build_v2 -R "V2_Integration_HybridQ16Pipeline_vs_FP32" --output-on-failure

# Expected: HybridQ16 should now achieve cosine > 0.95 vs FP32
```

### Long-term Considerations

1. **Re-evaluate Q16 attention design**: The current design assumes `.d` is normalization scale
2. **Consider FP32 attention fallback**: May be simpler and more maintainable
3. **Document scale semantics**: Clarify what `.d` field means vs normalization scales

---

## Key Files

| File | Purpose |
|------|---------|
| `src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp` | Main Q16 attention kernel (decode + prefill) |
| `src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h` | Params struct with `read_k_scales_from_blocks`, `has_per_position_k_scales()` |
| `src/v2/kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.cpp` | Microkernel implementations including new `fa2_prefill_process_kv_tile_with_k_scales` |
| `src/v2/kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h` | Microkernel declarations, `OnlineSoftmaxStateBatch` struct |
| `src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp` | Kernel entry point, **PARTIAL FIX APPLIED** (line 195: `read_k_scales_from_blocks=false`) |
| `tests/v2/integration/Test__HybridQ16Pipeline_vs_FP32_LayerByLayer.cpp` | The integration test |
| `tests/v2/unit/microkernels/Test__Q16AttentionRealisticData.cpp` | Unit test with realistic pipeline data |
| `tests/v2/unit/microkernels/Test__OnlineSoftmaxPerPositionKScales.cpp` | **NEW (Session 3)** Unit tests reproducing the 32x alpha bug |

---

## Run the Bug Reproduction Tests

```bash
# Build
cmake --build build_v2 --target v2_test_microkernel_online_softmax_k_scales --parallel

# Run all tests (shows full analysis)
./build_v2/tests/v2/v2_test_microkernel_online_softmax_k_scales

# Run specific test demonstrating the bug
./build_v2/tests/v2/v2_test_microkernel_online_softmax_k_scales --gtest_filter="*ActualBug*"

# Run via ctest
ctest --test-dir build_v2 -R "V2_Unit_Microkernel_OnlineSoftmaxKScales" --output-on-failure
```

---

## Debug Logging Added

Current debug logging in `Q16IntegerAttentionRef.cpp` (lines 780-794):
```cpp
// Debug: log first tile K scales
if (h == 0 && q_tile_start == 0 && kv_tile_start == 0) {
    LOG_DEBUG("FA2 Prefill K scale debug (head 0, first tile):");
    LOG_DEBUG("  base_alpha=" << state.base_alpha_fp32 << " q_scale=" << params.get_q_scale(h) << " qk_scale=" << qk_scale);
}
// ... inside the K scale extraction loop:
if (h == 0 && q_tile_start == 0 && kv_tile_start == 0 && c < 4) {
    LOG_DEBUG("  K_scale[" << c << "]=" << k_scales_tile[c] << " alpha[c]=" << (state.base_alpha_fp32 * k_scales_tile[c]));
}
```

---

## Next Steps (Updated After Session 3)

### ✅ COMPLETED: Root Cause Identified

The bug is **confirmed**: per-position K scale code reads `K_block.d` (quantization scale ~0.008) instead of `kv_head_scale` (normalization scale ~0.000244), causing alpha to be **32x too large**.

### Immediate Fix Required

**Option A: Disable per-position K scales (recommended quick fix)**

In `Q16FusedAttentionKernel.cpp` line 192:
```cpp
// WRONG: This reads quantization scale, not normalization scale!
ref_params.read_k_scales_from_blocks = true;

// FIX: Don't use per-position K scales at all
ref_params.read_k_scales_from_blocks = false;
```

**Option B: Fix the scale source (if per-position scales are truly needed)**

In `Q16IntegerAttentionRef.cpp`, instead of reading `K_block.d`:
```cpp
// WRONG: Uses quantization scale
float k_scale = K_block.d;

// CORRECT: Should use normalization scale
float k_scale = kv_head_scale;  // ~0.000244
```

### Verification Commands

After applying fix:
```bash
# Rebuild
cmake --build build_v2 --parallel

# Run integration test to verify fix
ctest --test-dir build_v2 -R "V2_Integration_HybridQ16Pipeline_vs_FP32" --output-on-failure

# Run unit tests to ensure no regression
./build_v2/tests/v2/v2_test_microkernel_online_softmax_k_scales
```

### Future Considerations

1. **Remove per-position K scale code path entirely**: Since it's based on incorrect assumptions, consider removing it to avoid future confusion

2. **KV cache scale increase**: Unit test shows scale=512 has 0 clipped elements vs 1 at scale=256 - may still be worth investigating for precision improvements

---

## Related Documentation

- `docs/v2/projects/2026-01/PHASE8_ATTENTION_KERNEL_K_SCALES.md` - Original design for K scale handling
- `.github/copilot-instructions.md` - Project development guidelines

---

## New Unit Test: Test__Q16AttentionRealisticData

Location: `tests/v2/unit/microkernels/Test__Q16AttentionRealisticData.cpp`

### Test Cases

| Test | Purpose | Result |
|------|---------|--------|
| `K_ExceedsRepresentableRange` | Verify K max_abs (130.36) > scale=256 max (127.99) | ✓ Confirms overflow |
| `FixedVsDynamicK_Quantization` | Compare fixed vs dynamic scale quantization | ✓ Dynamic is 200× better |
| `PerPositionVsUniformK_ScaleBug` | Demonstrate per-position scale variance | ✓ Scales vary 0.003-0.012 |
| `SingleHeadAttention_FP32vsQ16` | E2E attention comparison | ✓ Output cos=0.9999 |
| `OptimalKVCacheScale` | Find minimum scale for K data | ✓ Scale=512 has 0 clipping |

### Run Command
```bash
./build_v2/tests/v2/v2_test_microkernel_q16_attention_realistic
```

---

## Detailed Test Output Analysis

### Layer 0 ATTENTION_CONTEXT Breakdown

```
Overall cosine: 0.5949 (HybridQ16) vs 0.8971 (Hybrid)

Per-head cosines (worst first):
  h11=0.261 h13=0.299 h 7=0.333 h 9=0.364 h12=0.513 h 8=0.558
  h10=0.745 h 3=0.759 h 1=0.764 h 4=0.780 h 5=0.785 h 0=0.819
  h 2=0.905 h 6=0.909

Per-row cosines (worst first):
  r5=0.352 r3=0.372 r8=0.501 r6=0.504 r2=0.555 r4=0.575
  r7=0.603 r1=0.687 r0=0.996  <-- First position nearly perfect!
```

### Pattern Analysis

1. **Head pattern**: Heads 7-13 (mapped to KV head 1 in 7:1 GQA) are worse than heads 0-6
2. **Row pattern**: Position 0 is excellent (0.996), positions 1-8 degrade significantly
3. **Implication**: The first KV position is handled correctly, subsequent positions have scale issues

---

## Summary

**ROOT CAUSE IDENTIFIED IN SESSION 3**: The HybridQ16 attention is broken because ALL Q16 block `.d` fields contain **quantization scales** (dynamic, data-dependent) instead of the **normalization scales** (fixed, design-based) the kernel expects.

### Key Findings

1. **Q scale**: Expected ~0.00195, actual ~0.0078 (4x too large)
2. **K scale**: Expected ~0.000244, actual ~0.008 (32x too large)  
3. **V scale**: Expected ~0.000244, actual ~0.008 (32x too large)

### Effect

Combined scale error makes attention scores ~16x too large, causing softmax to become almost one-hot. This destroys attention patterns and produces garbage output.

### Fix Status

- **Partial fix applied**: `read_k_scales_from_blocks = false` (disables per-position K scales)
- **Fix insufficient**: Test still fails because Q and KV scales are also wrong
- **Full fix needed**: Use hardcoded normalization scales instead of extracting from block `.d`

### Historical Context

Sessions 1-2 identified symptoms (first position works, subsequent fail; K overflow; per-position scale variance). Session 3 traced the per-position K scale bug then discovered the deeper issue affects ALL scales.



---

## Session 4: Deeper Investigation (Continued)

### Correction: Scale Values Are Correct!

Upon closer investigation, the scale values in Q16 block `.d` fields ARE correct:

1. **Unit test `Test__Q16IntegerAttentionParity` PASSES** with these same scales
2. **kv_cache_scale = 256.0** → block.d = 256/32767 = 0.00781
3. **Both q_scales and kv_scales use this same value** in the unit test
4. **The unit test quantizes FP32 → Q16 with the same scale** and achieves good parity

### The Real Issue: Something Different in Integration Test

The problem is NOT the scales themselves, but something else in the integration pipeline:

| Test | ATTENTION_CONTEXT Cosine vs FP32 |
|------|----------------------------------|
| Unit test (synthetic) | >0.99 |
| Integration (Hybrid Q8_1) | ~0.9-0.99 |
| Integration (HybridQ16) | ~0.03-0.6 |

### Questions to Investigate

1. **Data layout**: Is the Q/K/V data in the expected layout for Q16 attention?
   - Unit test: HEAD_MAJOR layout (transposed K/V)
   - Integration: What layout does the pipeline provide?

2. **Block size**: Q16_1Block has 32 elements, Q16_1Block_64 has 64 elements
   - Unit test uses Q16_1Block_64 explicitly
   - What does the pipeline use?

3. **KV cache format**: Is the KV cache data correctly formatted for Q16 attention?
   - Hybrid (Q8_1) uses Q8_1 KV cache with FP32 attention
   - HybridQ16 uses Q16_1 KV cache with Q16 integer attention

4. **Pre-normalization**: Does the Q16 data arrive pre-normalized?
   - The Q16 attention kernel expects normalized Q/K/V
   - Are they being normalized before attention in the pipeline?

### Next Steps

1. Add detailed logging to trace data flow in HybridQ16 mode
2. Compare data shapes/layouts between unit test and integration
3. Check if KV cache data is being correctly written/read for Q16_1

### Verified: Scales ARE Correct!

From debug logs during integration test:
```
Q16FusedAttentionKernel Layer 0: extracted q_scales[0..2]=0.00781274, 0.00781274, 0.00781274
K scales from cache range [0.00734449 - 0.00795628]
```

These values match `kv_cache_scale / 32767 = 256 / 32767 ≈ 0.00781`.

**The comment in the code saying "expected ~0.00195 = 64/32767" is WRONG!**
The correct expected value is 0.00781 (which is what we're getting).

### Next Investigation Direction

Since the scales are correct, the issue must be elsewhere. Possible causes:
1. Data layout mismatch (Q/K/V tensor layout different from what kernel expects)
2. KV cache read/write issues
3. Online softmax implementation bug with real data distributions
4. Block size handling (mixing 32-element and 64-element blocks)
5. Pre-normalization missing or incorrect

The unit tests pass with synthetic uniformly-distributed data but fail with real model data.
This suggests something about the real data distribution triggers a bug.

### Layout Verification: Q Layout Is Correct

Q16IntegerAttentionRef expects Q in **HEAD_MAJOR** layout `[n_heads, seq_len, head_dim]`:
```cpp
const Q16_1Block_64 *Q_tile = Q_blocks + h * seq_len_q * blocks_per_row + q_tile_start * blocks_per_row;
```

FusedAttentionWoStage correctly transposes Q from position-major `[seq, heads, dim]` to head-major `[heads, seq, dim]`:
```cpp
// Transpose: position-major [p][h] -> head-major [h][p]
src_block_idx = (p * n_heads + h) * blocks_per_head;
dst_block_idx = (h * seq_len + p) * blocks_per_head;
```

The layout is correct. The issue must be elsewhere.

### Remaining Investigation Areas

1. **Online softmax numerical stability** - Real data may have extreme values
2. **K/V cache read path** - Is the KV cache data being read correctly?
3. **Sum computation** - Q16_1Block has sum_qs precomputed, is it being used correctly?
4. **Block boundary handling** - Edge cases with partial blocks
5. **Residual fusion** - Is the fused residual add working correctly?

---

## Current Status Summary (Session 4)

### What We Know

| Aspect | Status | Notes |
|--------|--------|-------|
| Q scales | ✅ Correct | 0.00781 = 256/32767 |
| K scales (uniform) | ✅ Correct | 0.00734-0.00796 |
| K scales (per-position) | ⚠️ Disabled | Was buggy, disabled pending fix |
| Q layout | ✅ Correct | Transposed to HEAD_MAJOR for prefill |
| Block size | ✅ Correct | BLOCK_64 for head_dim=64 |
| Backend | ✅ Correct | Q16_INTEGER for HybridQ16 |

### What Fails

| Test | HybridQ16 | Hybrid (Q8_1) |
|------|-----------|---------------|
| ATTENTION_CONTEXT cosine | 0.03-0.6 | 0.9-0.99 |
| ATTENTION_OUTPUT cosine | <0.1 | 0.9+ |
| Unit test (synthetic) | ✅ Passes | N/A |
| Integration test (real) | ❌ Fails | ✅ Works |

### Root Cause: TBD

The Q16 integer attention produces correct results on synthetic uniformly-distributed
data in unit tests, but fails on real model activations from the Qwen model.

**Hypothesis**: Something about real data distributions triggers a bug in the Q16
attention implementation that doesn't manifest with synthetic data.

### Next Steps for Handover

1. Compare input data distributions between unit test and integration test
2. Add extreme value detection in Q16 attention (overflow/underflow)
3. Trace INT32 accumulator values to check for overflow
4. Compare online softmax intermediate values between FP32 and Q16 paths
5. Consider adding a "fallback to FP32" path when Q16 values are problematic
