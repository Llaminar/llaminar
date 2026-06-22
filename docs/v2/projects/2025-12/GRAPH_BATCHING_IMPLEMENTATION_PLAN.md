# Graph Framework Batching Implementation Plan

**Author:** David Sanftenberg  
**Date:** December 2025  
**Status:** In Progress - BLOCKING ISSUE DISCOVERED

## Executive Summary

**BLOCKER IDENTIFIED (December 2025):** The Graph framework cannot support `batch_size > 1` until per-sequence KV cache handling is implemented. See [Phase 3: KV Cache Batched Append](#phase-3-kv-cache-batched-append-medium-effort) for details.

### Completed Work
- ✅ Phase 1: `batch_size` parameter now flows through `buildAttentionGraph()`
- ✅ Phase 2: `AttentionComputeStage::Params.batch_size` no longer hardcoded
- ✅ Phase 4: `KVCacheAppendStage` uses `batch_size * seq_len` for total tokens
- ✅ Unit tests pass with changes

### Blocking Issue
The `KVCacheAppendStage` appends all batch tokens to `seq_idx = 0`:
```cpp
// Qwen2Graph.cpp line 681
kv_append_params.seq_idx = 0;  // All sequences share same cache slot!
```

This means:
1. Sequential execution (batch_size=1 per sequence) → Each sequence has own cache
2. Batched execution (batch_size=N) → All N sequences share sequence 0's cache ❌

Result: Logits diverge massively (151K+ mismatches, max_diff > 100).

### Workaround
MPI tests continue using `force_pipeline = true` which routes to `Qwen2Pipeline::forward_batch()`.

## Overview

This document outlines the implementation plan for supporting `batch_size > 1` in the Llaminar V2 Graph execution framework. The goal is to enable batched inference through the declarative `GraphOrchestrator` path, replacing the legacy `Qwen2Pipeline::forward_batch()` which is blocked by assertion.

## Current State Analysis

### Working Components

| Component | Batch Support | Notes |
|-----------|--------------|-------|
| `GQAAttention::compute_batch()` | ✅ Full | Separate code path for batch_size > 1 |
| `IUnifiedKVCache` | ✅ Full | Has `seq_idx` parameter on all methods |
| `AttentionComputeStage::Params` | ✅ Partial | Has `batch_size` field, hardcoded to 1 |
| `GraphOrchestrator::InferenceState` | ✅ Partial | Has batch fields, not fully wired |
| `buildPositionIds()` | ✅ Full | Already accepts `batch_size` parameter |

### Blocking Issues

1. **Hardcoded `batch_size = 1`** in `Qwen2Graph.cpp`:
   - Line 716: `attn_params.batch_size = 1;` (Decomposed attention)
   - Line 765: `attn_params.batch_size = 1;` (Legacy attention)

2. **Position ID handling** assumes single sequence:
   - `position_ids[0]` used instead of per-batch positions

3. **KV Cache append** doesn't pass `seq_idx`:
   - `KVCacheAppendStage` always uses `seq_idx = 0`

4. **Buffer allocation** assumes single sequence dimensions

## Implementation Phases

### Phase 1: Core Parameter Passing (LOW effort)

**Goal:** Pass actual `batch_size` through the graph system.

#### 1.1 Qwen2Graph Attention Parameters

**File:** `src/v2/pipelines/qwen/Qwen2Graph.cpp`

```cpp
// Line 716 - Decomposed attention path
// BEFORE:
attn_params.batch_size = 1;

// AFTER:
attn_params.batch_size = batch_size;  // From LayerContext or input
```

```cpp
// Line 765 - Legacy attention path  
// BEFORE:
attn_params.batch_size = 1;

// AFTER:
attn_params.batch_size = batch_size;
```

#### 1.2 LayerContext Batch Propagation

**File:** `src/v2/pipelines/qwen/Qwen2Graph.cpp`

The `LayerContext` struct needs batch_size:

```cpp
struct LayerContext {
    int batch_size = 1;        // ADD: batch size
    int seq_len = 0;
    IUnifiedKVCache* kv_cache = nullptr;
    const int* position_ids = nullptr;
    int device_idx = -1;
};
```

#### 1.3 ForwardInput Batch Size Flow

Ensure `Qwen2ForwardInput::batch_size` flows to:
1. `buildFullForwardGraph()` 
2. `buildLayerGraph()`
3. `buildAttentionSubgraph()`
4. `AttentionComputeStage::Params`

---

### Phase 2: Position ID Handling (MEDIUM effort)

**Goal:** Support per-sequence position offsets for batched inference.

#### 2.1 Position Array Layout

For batch_size=B with seq_len=S:
- **Total position IDs:** `B * S`
- **Layout:** Contiguous per sequence: `[seq0_pos0, seq0_pos1, ..., seq1_pos0, seq1_pos1, ...]`

#### 2.2 RoPE Position Offset

**File:** `src/v2/pipelines/qwen/Qwen2Graph.cpp`

Current code takes only first position:
```cpp
// Line 640 and 731
int pos_offset = position_ids ? position_ids[0] : 0;
```

For batching, need per-sequence offset OR pass full position array:
```cpp
// Option A: Pass full position array to RoPE stage
rope_params.position_ids = position_ids;  // Full [batch_size * seq_len]
rope_params.batch_size = batch_size;

// Option B: Compute base offset per sequence
for (int b = 0; b < batch_size; ++b) {
    int seq_offset = b * seq_len;
    rope_params.position_offset[b] = position_ids[seq_offset];
}
```

#### 2.3 Attention Position Offset

The attention stage needs position info for causal masking in decode mode:

```cpp
// Instead of single offset:
attn_params.position_offset = position_ids ? position_ids[0] : 0;

// Need batch-aware handling:
attn_params.position_ids = position_ids;  // Pass full array
attn_params.batch_size = batch_size;
```

---

### Phase 3: KV Cache Batched Append (MEDIUM effort) ⚠️ CURRENT BLOCKER

**Goal:** Append K/V tensors per-sequence in batched mode.

**STATUS:** This phase is the primary blocker for Graph-based batching.

**Current Problem:**
```cpp
// Qwen2Graph.cpp line 681
kv_append_params.seq_idx = 0;  // PROBLEM: All batch sequences use seq_idx=0!
kv_append_params.num_tokens = batch_size * seq_len;  // Correct count but wrong slot
```

For batch_size=2 with seq_len=2:
- Sequential execution: seq0 → cache[0], seq1 → cache[1] ✅
- Batched execution: seq0+seq1 → cache[0] (4 tokens) ❌

The attention then reads from the wrong cache slots for sequences beyond index 0.

#### 3.1 KVCacheAppendStage Enhancement

**File:** `src/v2/execution/ComputeStage.cpp` (KVCacheAppendStage)

Current signature:
```cpp
bool append_kv(int layer, const TensorBase* new_k, const TensorBase* new_v);
// Calls: append_kv(layer, 0, new_k, new_v)  // Always seq_idx=0
```

For batching:
```cpp
// Option A: Batch append with slicing
for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    auto k_slice = slice_sequence(new_k, seq_idx, seq_len);
    auto v_slice = slice_sequence(new_v, seq_idx, seq_len);
    cache->append_kv(layer, seq_idx, k_slice, v_slice);
}

// Option B: Add batch append API to IUnifiedKVCache
bool append_kv_batch(int layer, const TensorBase* k, const TensorBase* v, 
                     int batch_size, int seq_len);
```

#### 3.2 Cache Query for Batched Attention

When computing attention with cache:
- **Prefill:** Use K/V from current batch
- **Decode with cache:** Query cached K/V per sequence

```cpp
// Per-sequence cached_tokens query
for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    int kv_len = cache->get_cached_tokens(layer, seq_idx);
    kv_lengths[seq_idx] = kv_len;
}
```

---

### Phase 4: Buffer Allocation for Batching (MEDIUM effort)

**Goal:** Ensure all activation buffers are sized for batched execution.

#### 4.1 DeviceGraphBufferManager Allocation

**File:** `src/v2/execution/DeviceGraphBufferManager.cpp`

Current allocation pattern:
```cpp
// Single sequence: [seq_len, dim]
auto buffer = TensorFactory::create_fp32({seq_len, dim}, device_idx);
```

Batched allocation:
```cpp
// Batched: [batch_size * seq_len, dim]
auto buffer = TensorFactory::create_fp32({batch_size * seq_len, dim}, device_idx);
```

#### 4.2 Affected Buffers

| Buffer | Single Seq Shape | Batched Shape |
|--------|-----------------|---------------|
| `current_hidden` | `[S, D]` | `[B*S, D]` |
| `normalized` | `[S, D]` | `[B*S, D]` |
| `Q` | `[S, n_heads*head_dim]` | `[B*S, n_heads*head_dim]` |
| `K` | `[S, n_kv_heads*head_dim]` | `[B*S, n_kv_heads*head_dim]` |
| `V` | `[S, n_kv_heads*head_dim]` | `[B*S, n_kv_heads*head_dim]` |
| `attn_output` | `[S, D]` | `[B*S, D]` |
| `ffn_input` | `[S, D]` | `[B*S, D]` |
| `gate` | `[S, d_ff]` | `[B*S, d_ff]` |
| `up` | `[S, d_ff]` | `[B*S, d_ff]` |
| `down` | `[S, D]` | `[B*S, D]` |
| `workspace_scores` | `[n_heads*S, kv_len]` | `[n_heads*B*S, kv_len]` |

#### 4.3 Logits Buffer

For LM head output:
```cpp
// Single: [S, vocab_size]
// Batched: [B*S, vocab_size] or [B, S, vocab_size] reshaped
```

---

### Phase 5: Attention Masking (MEDIUM effort)

**Goal:** Correct causal masking for batched attention.

#### 5.1 Equal-Length Batches (Simple Case)

When all sequences in batch have same length:
- **No padding mask needed**
- Standard causal mask applies per-sequence
- Attention scores: `[B*n_heads, S, kv_len]`

#### 5.2 Variable-Length Batches (Padding Required)

When sequences have different lengths:

```cpp
// Input: sequences of lengths [4, 6, 3]
// Padded to max_len=6: 
//   seq0: [t0, t1, t2, t3, PAD, PAD]
//   seq1: [t0, t1, t2, t3, t4, t5]
//   seq2: [t0, t1, t2, PAD, PAD, PAD]

// Padding mask: [B, S] - true where padded
// Combined mask: causal AND NOT(padding)
```

**File:** `src/v2/pipelines/attention/GQAAttention.cpp`

Already has `build_sequence_mask()`:
```cpp
bool build_sequence_mask(TensorBase* mask, int batch_size, int seq_len,
                         const std::vector<int>* sequence_lengths,
                         const GQAAttentionConfig& config);
```

#### 5.3 Decode Mode with KV Cache

In decode mode (seq_len=1, kv_len > 1):
- Query position determines which K/V tokens are visible
- Per-sequence position tracking required

```cpp
// Sequence 0: position 10 → attends to K/V[0:11]
// Sequence 1: position 5  → attends to K/V[0:6]
```

---

### Phase 6: GraphOrchestrator Integration (LOW effort)

**Goal:** Wire batch_size through orchestrator execution.

#### 6.1 InferenceState Batching

**File:** `src/v2/pipelines/qwen/GraphOrchestrator.h`

```cpp
struct InferenceState {
    int batch_size = 1;
    int seq_len = 0;
    std::vector<int> positions;       // [batch_size] current positions
    std::vector<int> sequence_lengths; // [batch_size] actual lengths (for padding)
    // ... existing fields
};
```

#### 6.2 executeForward Batch Flow

**File:** `src/v2/pipelines/qwen/GraphOrchestrator.cpp`

```cpp
bool GraphOrchestrator::executeForward(const Qwen2ForwardInput& input, 
                                        Qwen2ForwardOutput& output) {
    // Validate batch input
    if (input.batch_size > 1 && !input.batches) {
        LOG_ERROR("Batch mode requires batches array");
        return false;
    }
    
    // Build position IDs for all sequences
    position_ids_storage = Qwen2Graph::buildPositionIds(
        input.seq_len, input.batch_size, input.position_offset);
    
    // Build graph with batch awareness
    ComputeGraph graph = graph_builder_->buildFullForwardGraph(input, output);
    // ... execute
}
```

---

### Phase 7: Test Migration (LOW effort)

**Goal:** Update E2E tests to use Graph framework instead of Pipeline.

#### 7.1 Remove force_pipeline Flag

**File:** `tests/v2/e2e/qwen2/Test__Qwen2MPIRankParity.cpp`

```cpp
// BEFORE (lines ~450, ~480):
InferenceRunnerConfig config_batch;
config_batch.force_pipeline = true;

// AFTER:
InferenceRunnerConfig config_batch;
// Uses Graph framework by default
```

#### 7.2 Update Test Assertions

Ensure tests verify:
1. Batched logits match sequential execution
2. Per-sequence token predictions match
3. MPI rank parity maintained in batch mode

---

## Edge Cases & Considerations

### Edge Case 1: Empty Sequences in Batch

```cpp
// batch = [[1, 2, 3], [], [4, 5]]  // Sequence 1 is empty
// Handling: Skip empty sequences or use special token
```

### Edge Case 2: Max Sequence Length Exceeded

```cpp
// batch = [[tokens...], [very_long_sequence]]
// Handling: Truncate to max_seq_len or reject
if (seq_len > max_seq_len) {
    LOG_WARN("Truncating sequence from " << seq_len << " to " << max_seq_len);
    seq_len = max_seq_len;
}
```

### Edge Case 3: KV Cache Capacity

```cpp
// Each sequence in batch needs cache space
// Total cache: batch_size * max_seq_len * layers * (K + V) * dims
size_t cache_per_seq = max_seq_len * n_layers * 2 * n_kv_heads * head_dim * sizeof(float);
size_t total_cache = batch_size * cache_per_seq;
```

### Edge Case 4: MPI with Batching

- Each MPI rank processes same batch
- Tensor parallel: weights sharded, activations replicated
- AllReduce after attention output and FFN down projections
- **No special handling needed** if single-sequence MPI works

### Edge Case 5: Continuous Batching

Future enhancement for dynamic sequence completion:
```cpp
// Sequence 1 completes (EOS token)
cache->clear_sequence(1);
// Insert new sequence into slot 1
batch[1] = new_sequence;
positions[1] = 0;
```

---

## Implementation Order

### Milestone 1: Equal-Length Batch Support
1. ✅ Phase 1.1: Remove hardcoded `batch_size = 1`
2. ✅ Phase 1.2: LayerContext batch propagation
3. ✅ Phase 4.1: Buffer allocation for batching
4. ✅ Phase 7.1: Remove force_pipeline in tests

**Deliverable:** Equal-length batch inference works via Graph

### Milestone 2: Variable-Length Batch Support
1. Phase 2: Full position ID handling
2. Phase 3: KV cache batched append
3. Phase 5.2: Padding mask support

**Deliverable:** Variable-length batch inference with padding

### Milestone 3: Production Hardening
1. Edge case handling
2. Memory optimization
3. Performance tuning
4. Comprehensive test coverage

---

## Testing Strategy

### Unit Tests
- `Test__Qwen2Graph_BatchedAttention`: Verify batch params flow
- `Test__KVCacheAppendStage_BatchedAppend`: Per-sequence cache ops
- `Test__GraphBufferManager_BatchedAllocation`: Buffer sizing

### Integration Tests
- `Test__GraphOrchestrator_EqualLengthBatch`: Same-length sequences
- `Test__GraphOrchestrator_VariableLengthBatch`: Padded sequences
- `Test__GraphOrchestrator_BatchedDecode`: Decode with KV cache

### E2E Tests
- `Test__Qwen2MPIRankParity::MultiSequenceBatchEqualLength`: MPI + batch
- `Test__Qwen2MPIRankParity::MultiSequenceBatchVariableLength`: MPI + padding

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | Pass batch_size to attention, fix position handling | P0 |
| `src/v2/pipelines/qwen/Qwen2Graph.h` | Add batch_size to LayerContext | P0 |
| `src/v2/execution/ComputeStage.cpp` | KVCacheAppendStage batch support | P1 |
| `src/v2/execution/DeviceGraphBufferManager.cpp` | Batched buffer allocation | P1 |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | Batch flow in executeForward | P1 |
| `tests/v2/e2e/qwen2/Test__Qwen2MPIRankParity.cpp` | Remove force_pipeline | P2 |

---

## Success Criteria

1. **Equal-length batches**: `forward_batch([[a,b], [c,d]])` produces same logits as sequential `forward([a,b])` + `forward([c,d])`

2. **Variable-length batches**: Proper padding/masking for `[[a,b,c], [d,e]]`

3. **MPI parity**: Batched results identical across MPI rank configurations

4. **Performance**: Batched inference faster than N sequential forward passes

5. **Memory**: No hot-path allocations in batched mode

---

## Appendix: Code References

### GQAAttention Batch Path
```cpp
// src/v2/pipelines/attention/GQAAttention.cpp:328-350
if (effective_batch_size > 1) {
    success = attention_kernel->compute_batch(
        Q_ptr, K_ptr, V_ptr, output_ptr,
        effective_batch_size, seq_len,
        config.n_heads, config.n_kv_heads, config.head_dim,
        config.causal, config.window_size,
        config.workspace_scores.get(), config.workspace_qkv_buffer.get(),
        config.workspace_context.get(), mask_tensor,
        (config.precision == ActivationPrecision::BF16),
        config.mpi_ctx.get(), -1);
}
```

### UnifiedKVCache Per-Sequence API
```cpp
// src/v2/tensors/UnifiedKVCache.h:68-80
virtual bool append_kv(int layer, int seq_idx, 
                       const TensorBase* new_k, const TensorBase* new_v) = 0;
virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;
virtual void clear_sequence(int seq_idx) = 0;
```

### buildPositionIds Signature
```cpp
// src/v2/pipelines/qwen/Qwen2Graph.h
static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset = 0);
```
