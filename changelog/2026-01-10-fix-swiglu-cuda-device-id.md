# Fix: SwiGLU Stage Missing CUDA device_id Assignment

**Date:** 2026-01-10
**Type:** Bug Fix
**Component:** Graph Construction, CUDA Execution

## Summary

Fixed a bug where the `SwiGLUStage` was always executing on CPU even when the compute graph was configured for CUDA execution. The stage's `device_id` parameter was not being set when constructing the FFN portion of the Qwen2 graph.

## Root Cause

In `Qwen2Graph.cpp::buildFFNGraph()`, the `SwiGLUStage::Params` struct was missing the `device_id` assignment:

```cpp
// BEFORE (BUG):
SwiGLUStage::Params swiglu_params;
swiglu_params.gate = buffers.gate;
swiglu_params.up = buffers.up;
swiglu_params.output = buffers.up;
swiglu_params.seq_len = total_tokens;
// device_id was NOT SET - defaulted to CPU

// AFTER (FIXED):
SwiGLUStage::Params swiglu_params;
swiglu_params.gate = buffers.gate;
swiglu_params.up = buffers.up;
swiglu_params.output = buffers.up;
swiglu_params.seq_len = total_tokens;
swiglu_params.device_id = device;  // Now correctly set
```

## Symptoms

- `DiagnosticStageByStageComparison` test showed `layer0_FFN_SWIGLU` failing with cosine similarity = 0.163 (instead of ~1.0)
- The SwiGLU output was identical to the `up` input tensor, indicating no computation was happening
- Standalone CUDA SwiGLU parity tests passed with cosine = 1.0, confirming the kernel was correct

## Diagnosis

1. Stage output debug facility (`LLAMINAR_STAGE_OUTPUT_PRINT`) revealed:
   - CPU run: SwiGLU output = `silu(gate) * up` (correct)
   - GPU run: SwiGLU output = `up` verbatim (kernel not called)

2. Added debug logging to `SwiGLUStage::execute()` showing `device_id=CPU` for all runs

3. Traced to missing `device_id` assignment in `Qwen2Graph.cpp`

## Files Changed

- [src/v2/models/qwen/Qwen2Graph.cpp](src/v2/models/qwen/Qwen2Graph.cpp#L1385): Added `swiglu_params.device_id = device;`

## Impact

- **Layer 0 SwiGLU**: Now passes with cosine = 0.999881 (previously 0.163)
- **CUDA full model inference**: First failure point moved from `layer0_FFN_SWIGLU` to `layer0_FFN_DOWN`
- **Remaining issues**: Some stages still have cosine < 0.999, likely due to accumulated numerical differences between CPU/GPU GEMM implementations

## Testing

```bash
# Run diagnostic test
./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference \
    --gtest_filter="*DiagnosticStageByStageComparison*"

# Expected: layer0_FFN_SWIGLU now shows ✓ instead of ❌
```

## Lessons Learned

1. Always verify `device_id` is set on stage params when adding new stages to the graph
2. Stage output debug facility (`LLAMINAR_STAGE_OUTPUT_PRINT`) is invaluable for diagnosing buffer wiring issues
3. When standalone kernel tests pass but pipeline fails, look for configuration/wiring issues rather than kernel bugs
