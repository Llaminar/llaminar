# Prefill KV Cache Reuse Infrastructure

**Date**: December 13, 2025

## Summary

Added infrastructure for prefill-optimized attention kernel with K/V cache reuse.
The optimization itself is pending, but the dispatch framework is in place.

## Changes

### New Features

1. **AttentionMode Enum** (`JitFusedAttentionWo.h`)
   - `DECODE`: Optimized for single-token generation (seq_len_q = 1)
   - `PREFILL`: Optimized for multi-token processing (seq_len_q > 1)
   - `AUTO`: Automatically selects based on batch_size

2. **Automatic Mode Selection**
   - `JitAttentionConfig::effectiveMode()` returns appropriate mode
   - Batch size 1 → DECODE, batch size > 1 → PREFILL

3. **Kernel Cache Separation**
   - Hash function updated to include effective mode
   - Decode and prefill kernels cached separately

4. **V Accumulation Helper** (`JitVWeightedAccum.h`)
   - Added `emit_weighted_accum_from_cache()` for stack-cached V blocks
   - Supports head_dim up to 128 (4 Q8_1 blocks)

### Design Document

- Added `docs/v2/projects/2025-12/PREFILL_KV_CACHE_REUSE_DESIGN.md`
- Documents loop reordering strategy (Q→H→KV vs H→KV→Q)
- Memory bandwidth analysis
- Stack layout for per-query state
- Implementation phases

## Current State

The prefill kernel currently falls back to the decode kernel, which:
- ✅ Produces correct results
- ❌ Does not reuse K/V cache across queries

## Performance (Current)

| Config | JIT (ms) | REF (ms) | Speedup |
|--------|----------|----------|---------|
| Qwen0.5B decode | 0.21 | 0.76 | 3.6x |
| Qwen0.5B prefill (32 tokens) | 4.5 | 20.2 | 4.4x |
| Qwen7B decode | 3.5 | 21.1 | 6.1x |
| Qwen7B prefill (128 tokens) | 440 | 2816 | 6.4x |

## Expected Performance (After Optimization)

With K/V cache reuse, prefill should improve by ~2-3x:
- K/V memory traffic reduced by `seq_len_q` times
- Main limitation shifts from memory bandwidth to compute

## Files Changed

- `src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h`
- `src/v2/kernels/cpu/jit/q8_1/JitVWeightedAccum.h`
- `tests/v2/performance/kernels/cpu/attention/Perf__FusedAttentionWo.cpp`
- `docs/v2/projects/2025-12/PREFILL_KV_CACHE_REUSE_DESIGN.md` (new)

## Next Steps

1. Implement H→KV→Q loop order in `generate_prefill_kernel()`
2. Add per-query softmax state tracking
3. Add per-query context accumulation
4. Profile and optimize inner loop
