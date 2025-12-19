# Phase 10: Graph Caching for Decode Mode

**Date**: 2025-12-19
**Author**: David Sanftenberg

## Summary

Implemented graph caching optimization for decode mode to eliminate redundant graph construction overhead during autoregressive generation.

## Problem Statement

Previously, each layer execution during decode created new ComputeGraph objects:
- `executeAttention()` called `buildAttentionGraph()` → new graph
- `executeFFN()` called `buildFFNGraph()` → new graph

For a 24-layer model, this meant **48 graph constructions per token**. For 100 tokens of generation, that's 4,800 graph constructions.

## Solution

Implemented graph caching for decode mode (seq_len=1) when graph buffer management is enabled:

1. **Cache Structure**: `CachedLayerGraphs` struct per layer containing:
   - `attention_decode`: Cached attention graph
   - `ffn_decode`: Cached FFN graph
   - `cached_seq_len`: Sequence length the graph was built for
   - `valid`: Cache validity flag

2. **Dynamic Parameter Updates**: Only certain parameters change between decode iterations:
   - `pos_offset` for RoPE (rotary position embedding)
   - KV cache length (queried dynamically from cache, not stored in graph)

3. **updateDynamicParams() Interface**: New virtual method in `IComputeStage`:
   - `RoPEStage::updateDynamicParams()`: Updates `params_.pos_offset`
   - `AttentionComputeStage::updateDynamicParams()`: Updates `params_.position_offset`

## Performance Impact

**Before optimization (per token during decode)**:
- 24 layers × 2 graphs = 48 graph constructions

**After optimization**:
- First decode token: 48 graph constructions (cached)
- Subsequent tokens: 0 graph constructions (reuse cached)

**For 100-token generation**:
- Before: 4,800 graph constructions
- After: 48 graph constructions
- **Reduction: 99%**

## Files Modified

1. **`src/v2/pipelines/qwen/Qwen2Graph.h`**:
   - Added `CachedLayerGraphs` struct
   - Added `graph_caching_enabled_` flag
   - Added `layer_graph_cache_` vector
   - Added method declarations for cache management

2. **`src/v2/execution/ComputeStage.h`**:
   - Added `updateDynamicParams(int pos_offset, int seq_len)` virtual method
   - Implemented in `RoPEStage` and `AttentionComputeStage`

3. **`src/v2/pipelines/qwen/Qwen2Graph.cpp`**:
   - Modified `executeAttention()` to use/build cached graphs
   - Modified `executeFFN()` to use/build cached graphs
   - Modified `initializeBuffers()` to enable caching and resize cache vector
   - Modified `clearCache()` and `releaseBuffers()` to clear graph caches
   - Added `updateCachedGraphParams()` helper function

## Activation

Graph caching is automatically enabled when:
1. `LLAMINAR_USE_LAYER_EXECUTOR=1`
2. `LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1`

Prefill (seq_len > 1) still builds fresh graphs since the graph structure depends on sequence length.

## Testing

All existing tests pass:
- `V2_Unit_Qwen2GraphBufferManagement` ✓
- `V2_Integration_GraphBufferManagement_Parity` ✓
- All unit tests ✓
- All integration tests ✓

## Log Output

With `LLAMINAR_LOG_LEVEL=DEBUG`:
```
[DEBUG] [Qwen2Graph] Building and caching attention graph for layer 0 (decode mode)
[DEBUG] [Qwen2Graph] Building and caching FFN graph for layer 0 (decode mode)
...
[DEBUG] [Qwen2Graph] Reusing cached attention graph for layer 0 (pos_offset=5)
[DEBUG] [Qwen2Graph] Reusing cached FFN graph for layer 0
...
[DEBUG] [Qwen2Graph] Reusing cached attention graph for layer 0 (pos_offset=6)
```

## Future Improvements

1. Consider caching prefill graphs for fixed prompt lengths
2. Add cache hit/miss statistics to benchmark output
3. Profile actual wall-time savings from reduced allocations
