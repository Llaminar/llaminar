# Operator-Level Snapshot Capture Fix

**Date**: October 19, 2025  
**Status**: ✅ Complete  
**Branch**: feature/quantized-tensors

## Problem Summary

Sequential decode snapshots were partially missing - we had EMBEDDING and ATTENTION_NORM, but all operator-level stages (Q/K/V projections, RoPE, attention context) were absent. This prevented stage-level parity diagnostics from identifying where batch vs sequential decode diverge.

## Root Cause

The attention operator's snapshot callback was passing the base `snapshot_source_` to `AbstractPipeline::captureStageSnapshot()` **without** applying the decode step suffix (`_dec0`, `_dec1`, etc.). This caused operator snapshots to be registered with keys like:
- ❌ `llaminar_layer_0_Q_PROJECTION` (no decode step)

Instead of:
- ✅ `llaminar_dec0_layer_0_Q_PROJECTION` (with decode step)

The `QwenPipeline::captureIfEnabled()` method was already correctly applying the suffix, but the operator callback bypassed this logic.

## Solution

### 1. Created Helper Method

Added `QwenPipeline::getEffectiveSnapshotSource()` to centralize decode step suffix logic:

```cpp
inline std::string QwenPipeline::getEffectiveSnapshotSource() const
{
    std::string effective_source = snapshot_source_;
    const auto &env = debugEnv();
    if (env.pipeline.decode_stage_snapshots && current_decode_step_ >= 0)
    {
        effective_source += "_dec" + std::to_string(current_decode_step_);
    }
    return effective_source;
}
```

### 2. Updated Snapshot Callback

Modified attention operator initialization to use the helper:

```cpp
// BEFORE: Used snapshot_source_ directly (no decode step suffix)
attention_kernel->setSnapshotCallback([this](...) {
    AbstractPipeline::captureStageSnapshot(..., snapshot_source_);
});

// AFTER: Uses getEffectiveSnapshotSource() for consistent suffix
attention_kernel->setSnapshotCallback([this](...) {
    AbstractPipeline::captureStageSnapshot(..., getEffectiveSnapshotSource());
});
```

### 3. Simplified captureIfEnabled

Refactored `captureIfEnabled()` to use the same helper, eliminating code duplication.

## Results

### Before Fix
```
key_batch=batch_dec0_layer_0_Q_PROJECTION present=1 
key_seq=llaminar_dec0_layer_0_Q_PROJECTION present=0  ❌

key_batch=batch_dec0_layer_0_K_PROJECTION present=1
key_seq=llaminar_dec0_layer_0_K_PROJECTION present=0  ❌

key_batch=batch_dec0_layer_0_V_PROJECTION present=1
key_seq=llaminar_dec0_layer_0_V_PROJECTION present=0  ❌

key_batch=batch_dec0_layer_0_ROPE_APPLICATION present=1
key_seq=llaminar_dec0_layer_0_ROPE_APPLICATION present=0  ❌

key_batch=batch_dec0_layer_0_ATTENTION_CONTEXT present=1
key_seq=llaminar_dec0_layer_0_ATTENTION_CONTEXT present=0  ❌
```

### After Fix
```
key_batch=batch_dec0_layer_0_Q_PROJECTION present=1
key_seq=llaminar_dec0_layer_0_Q_PROJECTION present=1  ✅

key_batch=batch_dec0_layer_0_K_PROJECTION present=1
key_seq=llaminar_dec0_layer_0_K_PROJECTION present=1  ✅

key_batch=batch_dec0_layer_0_V_PROJECTION present=1
key_seq=llaminar_dec0_layer_0_V_PROJECTION present=1  ✅

key_batch=batch_dec0_layer_0_ROPE_APPLICATION present=1
key_seq=llaminar_dec0_layer_0_ROPE_APPLICATION present=1  ✅

key_batch=batch_dec0_layer_0_ATTENTION_CONTEXT present=1
key_seq=llaminar_dec0_layer_0_ATTENTION_CONTEXT present=1  ✅
```

**All 10 critical decode snapshots now captured!**

## Files Modified

1. **src/QwenPipeline.h**
   - Added `getEffectiveSnapshotSource()` declaration

2. **src/QwenPipeline.cpp**
   - Implemented `getEffectiveSnapshotSource()` helper
   - Updated attention operator callback to use helper
   - Refactored `captureIfEnabled()` to use helper

## Next Steps

1. **Per-Sequence Comparison**: Current issue is shape mismatch (batch: [10,896], sequential: [5,896]) because batch processes 2 sequences together while sequential processes one at a time. Need to implement per-sequence snapshot slicing or separate pipeline comparison.

2. **Stage-Level Diagnostics**: Once per-sequence comparison working, perform numeric diff at each stage to identify first divergence point:
   - EMBEDDING
   - ATTENTION_NORM
   - Q/K/V projections
   - RoPE application
   - Attention context

3. **Root Cause Investigation**: Likely culprits for decode divergence:
   - RoPE position encoding (full context vs incremental)
   - RMSNorm with padding tokens
   - Bias slicing differences
   - Residual connection handling

## Validation

Test run shows all snapshots present:
```bash
$ grep "key_seq=llaminar_dec0" decode_test_with_operators.log | head -10
  key_batch=batch_dec0_EMBEDDING present=1 key_seq=llaminar_dec0_EMBEDDING present=1
  key_batch=batch_dec0_layer_0_ATTENTION_NORM present=1 key_seq=llaminar_dec0_layer_0_ATTENTION_NORM present=1
  key_batch=batch_dec0_layer_0_Q_PROJECTION present=1 key_seq=llaminar_dec0_layer_0_Q_PROJECTION present=1
  key_batch=batch_dec0_layer_0_K_PROJECTION present=1 key_seq=llaminar_dec0_layer_0_K_PROJECTION present=1
  key_batch=batch_dec0_layer_0_V_PROJECTION present=1 key_seq=llaminar_dec0_layer_0_V_PROJECTION present=1
  key_batch=batch_dec0_layer_0_ROPE_APPLICATION present=1 key_seq=llaminar_dec0_layer_0_ROPE_APPLICATION present=1
  key_batch=batch_dec0_layer_0_ATTENTION_CONTEXT present=1 key_seq=llaminar_dec0_layer_0_ATTENTION_CONTEXT present=1
  key_batch=batch_dec0_layer_0_ATTENTION_OUTPUT present=1 key_seq=llaminar_dec0_layer_0_ATTENTION_OUTPUT present=1
  key_batch=batch_dec0_FINAL_NORM present=1 key_seq=llaminar_dec0_FINAL_NORM present=1
  key_batch=batch_dec0_LM_HEAD present=1 key_seq=llaminar_dec0_LM_HEAD present=1
```

## Summary

Successfully restored complete operator-level snapshot capture for sequential decode path by ensuring decode step suffix is consistently applied across all snapshot capture points (pipeline-level and operator-level). This unblocks stage-level parity diagnostics for investigating the ~151,920 vocabulary mismatch issue between batch and sequential decode.
