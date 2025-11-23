# Fix: MPI Attention Masking for Batched Inference

## Summary
Fixed a critical bug in `MpiAttentionOrchestrator` where a potentially uninitialized or garbage mask tensor was being passed to the attention kernel even when masking was not required (e.g., non-causal attention with equal sequence lengths). This caused numerical divergence in batched inference, specifically affecting the second sequence in a batch.

## Key Changes
- Modified `MpiAttentionOrchestrator::compute_tensor_parallel` to pass `nullptr` instead of `mask_tensor` when `needs_mask` is false.
- Added regression test `tests/v2/unit/pipelines/Test__MpiAttentionOrchestrator_Masking.cpp` to verify that garbage masks are ignored when not needed.

## Impact
- Resolves E2E failure in `MultiSequenceBatchEqualLength` where Sequence 1 was diverging.
- Ensures correct attention computation for non-causal batched inference.

## Test Results
- New unit test `V2_Unit_MpiAttentionOrchestrator_Masking` passes.
