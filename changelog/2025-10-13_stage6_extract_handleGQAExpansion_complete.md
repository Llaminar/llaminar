# Stage 6: Extract handleGQAExpansion() - COMPLETE ✅

**Date**: 2025-10-13  
**Author**: David Sanftenberg  
**Refactoring Phase**: 8 (Execute() Method Orchestration Refactoring)  
**Stage**: 6 of 9  
**Status**: ✅ **COMPLETE** - All tests passed (387+387+1170 = 2,544 comparisons)

---

## Executive Summary

Successfully extracted **STEP 6: Handle GQA Expansion** (~208 lines) from `MPIAttentionKernel::execute()` into dedicated `handleGQAExpansion()` helper method. This stage handles Grouped Query Attention (GQA) K/V head replication with Phase 3+4+5 multi-rank optimizations (fused gathering, metadata caching, zero-copy). Execute() reduced from **943 → 735 lines (-208 lines, -22%)**. Cumulative reduction: **-1,552 lines (-68%)** from original 2,287 lines.

**Key Achievement**: GQA expansion logic cleanly isolated with Phase 3+4+5 optimization pipeline preserved.

---

## Changes Made

### 1. Header File (`src/kernels/MPIAttentionKernel.h`)

**Added GQAExpansionResult Struct** (lines ~144-158, 15 lines):
```cpp
struct GQAExpansionResult
{
    // Expanded K/V tensors (replicated for GQA, or direct reference for MHA)
    std::shared_ptr<TensorBase> local_k_expanded;  // [attn_seq_len, local_head_dim]
    std::shared_ptr<TensorBase> local_v_expanded;  // [attn_seq_len, local_head_dim]
    
    // Metadata
    bool gqa_required;  // true if n_head != n_head_kv (GQA architecture)
};
```

**Added Method Declaration** (lines ~431-459, 29 lines):
```cpp
/**
 * @brief Handle Grouped Query Attention (GQA) expansion (STEP 6)
 * 
 * For GQA architectures, replicates K/V heads to match the number of query heads.
 * Each KV head serves multiple query heads (head_ratio = n_head / n_head_kv).
 * For MHA architectures (n_head == n_head_kv), passes through K/V unchanged.
 * 
 * Multi-rank optimization (Phase 3+4+5):
 * - Phase 3: Fuse K+V gathering into single MPI_Allgatherv
 * - Phase 4: Cache metadata to skip count gathering on repeated decode calls
 * - Phase 5: Use MPI derived datatype for zero-copy K+V interleaving
 * 
 * Single-rank path: Direct KV cache usage (no gathering needed)
 * 
 * Key operations:
 * 1. Check if GQA expansion needed (n_head != n_head_kv)
 * 2. For multi-rank: Fused KV cache gathering with Phase 3+4+5 optimizations
 * 3. For single-rank: Use local cache directly
 * 4. Call expand_kv_for_gqa() to replicate KV heads for local query heads
 * 5. Optional debug logging of expanded tensors
 * 
 * @param setup Input setup result from STEP 1
 * @param rope_result RoPE result from STEP 5 (contains KV cache)
 * @return GQAExpansionResult with expanded K/V tensors ready for attention
 */
GQAExpansionResult handleGQAExpansion(
    const InputSetupResult &setup,
    const RoPEResult &rope_result);
```

### 2. Implementation File (`src/kernels/MPIAttentionKernel.cpp`)

**Added Helper Method** (lines ~1858-2077, 220 lines):

**Structure Overview**:
1. **Parameter Extraction** (13 variables from setup and rope_result)
2. **GQA Check** (n_head != n_head_kv)
3. **Multi-rank Path** (world_size > 1):
   - **Phase 3**: Fused K+V gathering (2 allgatherv → 1)
   - **Phase 4**: Metadata caching (skip count gather on decode)
   - **Phase 5**: Zero-copy MPI derived datatype
4. **Single-rank Path**: Direct cache reference
5. **GQA Expansion Call**: `expand_kv_for_gqa()` with rank-major layout
6. **Debug Logging**: Expanded tensor stats (layer 0 only)
7. **MHA Fast Path**: Direct cache pass-through (no expansion needed)

**Phase 3+4+5 Optimizations** (Preserved from Original):
- **Phase 3 (Fused Gathering)**: Pack K+V into single buffer, eliminate second MPI_Allgatherv
  - Old: 2 separate MPI_Allgatherv calls (K, V)
  - New: 1 fused MPI_Allgatherv with interleaved K+V buffer
  - Benefit: ~50% reduction in MPI collective overhead
  
- **Phase 4 (Metadata Caching)**: Cache count/displacement arrays across decode steps
  - Predictable growth pattern: attn_seq_len increases by 1 each decode step
  - Skip MPI_Allgather for counts when pattern detected
  - Benefit: 1 fewer MPI collective per decode step
  
- **Phase 5 (Zero-Copy Interleaving)**: MPI derived datatype for pack operation
  - Old: Manual memcpy to pack K and V into single buffer before send
  - New: MPI_Type_create_struct tells MPI how to read from separate buffers
  - Benefit: Eliminate 1 memcpy per rank (pack operation)

**Key Implementation Details**:

```cpp
// Phase 4: Check if metadata can be reused (decode pattern)
bool can_use_cached_metadata = kv_cache_metadata_initialized_ &&
                               (attn_seq_len == last_attn_seq_len_ + 1);

if (can_use_cached_metadata) {
    // Reuse cached metadata (skip MPI_Allgather!)
    recvcounts_kv = cached_recvcounts_kv_;
    displs_kv = cached_displs_kv_;
    
    // Update counts for single token growth
    #pragma omp parallel for if (world_size > 4) schedule(static)
    for (int r = 0; r < world_size; ++r) {
        recvcounts_kv[r] += 2 * local_kv_head_dim;  // K + V for 1 token
    }
}

// Phase 5: Zero-copy K+V interleaving with MPI derived datatype
MPI_Datatype kv_type;
int blocklengths[2] = {attn_seq_len * local_kv_head_dim, attn_seq_len * local_kv_head_dim};
MPI_Aint displacements[2];
MPI_Get_address(local_k_cache->data(), &displacements[0]);
MPI_Get_address(local_v_cache->data(), &displacements[1]);
displacements[1] -= displacements[0];
displacements[0] = 0;

MPI_Datatype types[2] = {MPI_FLOAT, MPI_FLOAT};
MPI_Type_create_struct(2, blocklengths, displacements, types, &kv_type);
MPI_Type_commit(&kv_type);

// Fused gather with zero-copy (Phase 3+5)
MPI_Allgatherv(local_k_cache->data(), 1, kv_type,
               fused_kv_buffer->data(), recvcounts_kv.data(), displs_kv.data(),
               MPI_FLOAT, MPI_COMM_WORLD);

// GQA expansion with correct head mapping
auto [local_kv, kv_offset] = getKVHeadDistribution();
llaminar::attn::expand_kv_for_gqa(
    global_k_cache->data(), global_v_cache->data(),
    result.local_k_expanded->data(), result.local_v_expanded->data(),
    attn_seq_len, head_dim_, local_heads, n_head_kv_, head_offset, n_head_,
    world_size > 1,  // gathered_rank_major = true for multi-rank
    kv_offset);      // kv_head_offset_for_rank
```

**Modified Execute() Method** (lines ~2189-2203, 15 lines replaces ~208 lines):
```cpp
// ========================================================================
// STEP 6: Handle GQA - replicate K/V heads if needed (REFACTORED)
// ========================================================================
auto gqa_result = handleGQAExpansion(setup, rope_result);

// Extract expanded K/V tensors for attention computation
auto local_k_expanded = gqa_result.local_k_expanded;
auto local_v_expanded = gqa_result.local_v_expanded;
```

**Old Code Removed**: ~208 lines of inline GQA logic
**New Code Added**: 1 helper call + 2 result extractions (3 lines)
**Net Change in execute()**: -205 lines

---

## Test Results

### Test 1: OpenBLAS Prefill (`ParityFramework.OpenBLASPrefillVsPyTorch`)
- **Duration**: 90.7 seconds (90,662 ms)
- **Result**: ✅ **387/387 passed** (0 failed, 0 missing)
- **Sample Results**:
  - ROPE_APPLICATION_layer22: max_abs=5.29e-05, rel_l2=4.66e-06 ✓
  - ATTENTION_SCORES_layer22: max_abs=7.63e-05, rel_l2=3.87e-06 ✓
  - ATTENTION_CONTEXT_layer22: max_abs=6.68e-05, rel_l2=1.12e-05 ✓
  - All stages within tolerance

### Test 2: COSMA Prefill (`ParityFramework.COSMAPrefillVsPyTorch`)
- **Duration**: 105.7 seconds (105,717 ms)
- **Result**: ✅ **387/387 passed** (0 failed, 0 missing)
- **Sample Results**:
  - ROPE_APPLICATION_layer22: max_abs=3.93e-05, rel_l2=1.82e-06 ✓
  - ATTENTION_SCORES_layer22: max_abs=7.22e-05, rel_l2=2.02e-06 ✓
  - ATTENTION_CONTEXT_layer22: max_abs=6.28e-05, rel_l2=5.46e-06 ✓
  - All stages within tolerance

### Test 3: Incremental Decode (`ParityFramework.TrueIncrementalDecodeVsPyTorch`)
- **Duration**: 33.7 seconds (33,661 ms)
- **Result**: ✅ **585 stages, 1,170 comparisons passed** (0 failed)
- **Token Sequence**: ✓ MATCH (6 → 25010 → 10)
- **Validation**: All 3 tokens validated across 24 layers × 16 stages
- **GQA Expansion**: Working correctly in decode mode with metadata caching

**Overall Test Summary**:
- **Total Comparisons**: 387 + 387 + 1,170 = **2,544 comparisons**
- **Pass Rate**: **100%** (2,544/2,544)
- **Performance**: Within variance (OpenBLAS: -0.2%, COSMA: -1.3%, Decode: -1.4%)

---

## Metrics

### Code Size Changes
| Metric | Before Stage 6 | After Stage 6 | Change | Notes |
|--------|----------------|---------------|--------|-------|
| execute() size | 943 lines | 735 lines | **-208 lines** | -22% reduction |
| Helper methods | 5 (1,464 lines) | 6 (1,684 lines) | **+220 lines** | New handleGQAExpansion() |
| Struct definitions | 5 (124 lines) | 6 (139 lines) | **+15 lines** | New GQAExpansionResult |
| Method declarations | 89 lines | 118 lines | **+29 lines** | New handleGQAExpansion() decl |
| **Net change** | | | **+56 lines** | Struct + decl + helper |

### Cumulative Progress (Stages 1-6)
| Metric | Original | After Stage 6 | Total Change | % Complete |
|--------|----------|---------------|--------------|------------|
| execute() size | 2,287 lines | 735 lines | **-1,552 lines** | **68% reduction** |
| Stages complete | 0/9 | **6/9** | 6 stages | **67%** |
| Lines refactored | 0 | ~1,701 | ~1,701 lines | **85%** of target |
| Helper methods | 0 | 6 | 1,684 lines | 6 helpers |
| Test pass rate | N/A | 100% | 2,544/2,544 | Perfect parity |

### Stage-by-Stage Breakdown
| Stage | Lines Extracted | Execute() Reduction | Helper Size | Cumulative Reduction |
|-------|----------------|---------------------|-------------|----------------------|
| 1. validateAndSetupInputs | 332 | -237 | 281 | -237 (-10%) |
| 2. distributeWeightsByHead | 282 | -264 | 295 | -501 (-22%) |
| 3. computeQKVProjections | 242 | -229 | 239 | -730 (-32%) |
| 4. gatherAndSnapshotPreRoPE | 155 | -148 | 166 | -878 (-38%) |
| 5. applyRotaryPositionEmbeddings | 482 | -467 | 483 | -1,344 (-59%) |
| **6. handleGQAExpansion** | **208** | **-208** | **220** | **-1,552 (-68%)** |
| **Total (6 stages)** | **1,701** | **-1,552** | **1,684** | **-68%** |

### Remaining Work (Stages 7-9)
| Stage | Estimated Lines | Estimated Reduction | Status |
|-------|----------------|---------------------|--------|
| 7. computeAttentionScores | ~389 | ~380 | Not started |
| 8. projectAndGatherOutput | ~174 | ~170 | Not started |
| 9. Final cleanup | ~0 | ~50 | Not started |
| **Total remaining** | **~563** | **~600** | **3 stages** |

**Projected Final State**:
- execute() target: ~135 lines (orchestration only)
- Total helper methods: 9
- Total reduction: ~2,150 lines (-94%)

---

## Implementation Details

### GQA (Grouped Query Attention) Background

**Standard Multi-Head Attention (MHA)**:
- n_head query heads (e.g., 14)
- n_head key heads (e.g., 14)
- n_head value heads (e.g., 14)
- Each Q head has dedicated K/V head (1:1 mapping)

**Grouped Query Attention (GQA)**:
- n_head query heads (e.g., 14)
- n_head_kv key heads (e.g., 2)
- n_head_kv value heads (e.g., 2)
- Multiple Q heads share K/V heads (head_ratio = 14/2 = 7)

**Head Replication Logic**:
```
Example: 14 Q heads, 2 KV heads, 2 ranks
Rank 0: 7 Q heads [0-6], 1 KV head [0]
Rank 1: 7 Q heads [7-13], 1 KV head [1]

GQA expansion:
  Q heads [0-6] → replicate KV head 0 (7 times)
  Q heads [7-13] → replicate KV head 1 (7 times)

Formula: kv_head = (local_h + head_offset) / (n_head / n_head_kv)
```

### Multi-Rank Optimization Pipeline

**Phase 3: Fused K+V Gathering**
- **Before**: 2 separate MPI_Allgatherv calls
  ```cpp
  MPI_Allgatherv(local_k, ..., global_k, ...);  // First collective
  MPI_Allgatherv(local_v, ..., global_v, ...);  // Second collective
  ```
- **After**: Single fused gather with interleaved buffer
  ```cpp
  // Pack K+V into single send buffer (fused)
  MPI_Allgatherv(fused_send, ..., fused_recv, ...);  // Single collective
  // Unpack received buffer into separate K and V
  ```
- **Benefit**: ~50% reduction in MPI synchronization overhead

**Phase 4: Metadata Caching**
- **Challenge**: Each MPI_Allgatherv requires count/displacement arrays
- **Observation**: In decode mode, attn_seq_len grows predictably (+1 per token)
- **Optimization**: Cache metadata, update incrementally
  ```cpp
  // First call or prefill: gather counts
  MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  cached_recvcounts_ = recvcounts;  // Cache for future
  
  // Subsequent decode calls: reuse and update
  recvcounts = cached_recvcounts_;
  for (int r = 0; r < world_size; ++r) {
      recvcounts[r] += 2 * local_kv_head_dim;  // K + V for 1 token
  }
  ```
- **Benefit**: Eliminate 1 MPI_Allgather per decode step

**Phase 5: Zero-Copy Interleaving**
- **Challenge**: Packing K+V requires memcpy before MPI_Allgatherv
- **Optimization**: Use MPI derived datatype to describe memory layout
  ```cpp
  MPI_Datatype kv_type;
  int blocklengths[2] = {k_size, v_size};
  MPI_Aint displacements[2] = {0, k_size_bytes};
  MPI_Type_create_struct(2, blocklengths, displacements, types, &kv_type);
  
  // MPI reads directly from K and V buffers (no pack memcpy!)
  MPI_Allgatherv(local_k_cache->data(), 1, kv_type, ...);
  ```
- **Benefit**: Eliminate manual pack memcpy (1 per rank)

**Combined Impact**:
- **Prefill**: ~3 MPI collectives → ~1.5 (Phase 3)
- **Decode**: ~3 MPI collectives → ~1 (Phase 3+4)
- **Memory**: ~1 pack memcpy eliminated (Phase 5)
- **Typical decode speedup**: ~2x for small operations (<100 tokens)

### Debug Instrumentation

**Phase 4 Logging** (layer 0 only):
```cpp
if (debugEnv().attention.verbose && layer_index_ == 0) {
    if (can_use_cached_metadata) {
        LOG_DEBUG("[PHASE 4] Rank " << rank << ": Reused cached KV metadata");
    } else {
        LOG_DEBUG("[PHASE 4] Rank " << rank << ": Initialized KV metadata cache");
    }
}
```

**Post-Expansion Logging** (layer 0 only):
```cpp
if (debugEnv().attention.verbose && layer_index_ == 0) {
    LOG_DEBUG("[RANK=" << rank << "] After GQA expansion (using cache):");
    LOG_DEBUG("  attn_seq_len=" << attn_seq_len);
    LOG_DEBUG("  K_expanded shape: [" << attn_seq_len << ", " << local_head_dim << "]");
    LOG_DEBUG("  K_expanded[0,0:5]: " << values);
    LOG_DEBUG("  K_expanded range: [" << min << ", " << max << "]");
}
```

---

## Code Organization

### Before Stage 6 (execute() = 943 lines)
```cpp
bool MPIAttentionKernel::execute(...) {
    // STEP 1: validate inputs (REFACTORED)
    auto setup = validateAndSetupInputs(...);
    
    // STEP 2: distribute weights (REFACTORED)
    auto weights = distributeWeightsByHead(setup);
    
    // STEP 3: compute QKV projections (REFACTORED)
    auto projections = computeQKVProjections(setup, weights);
    
    // STEP 4: gather pre-RoPE (REFACTORED)
    auto gather_result = gatherAndSnapshotPreRoPE(setup, projections);
    
    // STEP 5: apply RoPE (REFACTORED)
    auto rope_result = applyRotaryPositionEmbeddings(setup, projections, k_cache_in, v_cache_in);
    
    // STEP 6: handle GQA expansion (~208 lines INLINE)
    std::shared_ptr<TensorBase> local_k_expanded, local_v_expanded;
    if (n_head_ != n_head_kv_) {
        // Allocate expanded tensors
        local_k_expanded = TensorFactory::create_simple({attn_seq_len, local_head_dim});
        local_v_expanded = TensorFactory::create_simple({attn_seq_len, local_head_dim});
        
        // Gather full cache from all ranks
        if (world_size > 1) {
            // Phase 3+4+5 optimizations (~150 lines)
            // - Fused K+V gathering
            // - Metadata caching
            // - Zero-copy derived datatype
            // ... 150 lines of MPI logic ...
        } else {
            global_k_cache = local_k_cache;
            global_v_cache = local_v_cache;
        }
        
        // Expand KV heads
        auto [local_kv, kv_offset] = getKVHeadDistribution();
        llaminar::attn::expand_kv_for_gqa(...);
        
        // Debug logging (~15 lines)
    } else {
        local_k_expanded = local_k_cache;
        local_v_expanded = local_v_cache;
    }
    
    // STEP 7-8: remaining stages (~563 lines)
}
```

### After Stage 6 (execute() = 735 lines)
```cpp
bool MPIAttentionKernel::execute(...) {
    // STEP 1: validate inputs (REFACTORED)
    auto setup = validateAndSetupInputs(...);
    
    // STEP 2: distribute weights (REFACTORED)
    auto weights = distributeWeightsByHead(setup);
    
    // STEP 3: compute QKV projections (REFACTORED)
    auto projections = computeQKVProjections(setup, weights);
    
    // STEP 4: gather pre-RoPE (REFACTORED)
    auto gather_result = gatherAndSnapshotPreRoPE(setup, projections);
    
    // STEP 5: apply RoPE (REFACTORED)
    auto rope_result = applyRotaryPositionEmbeddings(setup, projections, k_cache_in, v_cache_in);
    
    // STEP 6: handle GQA expansion (REFACTORED)
    auto gqa_result = handleGQAExpansion(setup, rope_result);
    auto local_k_expanded = gqa_result.local_k_expanded;
    auto local_v_expanded = gqa_result.local_v_expanded;
    
    // STEP 7-8: remaining stages (~563 lines)
}
```

---

## Lessons Learned

### 1. Phase 3+4+5 Optimizations Are Critical
- **Observation**: GQA expansion is one of the most MPI-intensive stages
- **Impact**: Phase 3+4+5 reduce ~3 MPI collectives to ~1 per decode step
- **Lesson**: Preserve optimization pipeline when refactoring (don't simplify prematurely)

### 2. Metadata Caching Exploit Predictable Patterns
- **Challenge**: MPI_Allgather for counts is expensive (1 per step)
- **Solution**: Recognize decode pattern (attn_seq_len += 1), cache counts
- **Lesson**: Profile to identify repeated patterns, cache predictable data

### 3. MPI Derived Datatypes Enable Zero-Copy
- **Challenge**: Packing K+V requires manual memcpy before gather
- **Solution**: MPI_Type_create_struct describes memory layout to MPI
- **Lesson**: Use MPI's advanced features for non-contiguous data

### 4. Helper Methods Preserve Optimization Complexity
- **Challenge**: 208-line extraction with intricate MPI logic
- **Solution**: Extract entire optimization pipeline into helper
- **Lesson**: Don't oversimplify—keep complex optimizations intact

---

## Next Steps

### Stage 7: Extract computeAttentionScores()
- **Scope**: STEP 7 (~389 lines) - Attention score computation and softmax
- **Logic**:
  1. Compute Q·Kᵀ scores (scaled dot product)
  2. Apply causal masking (decoder-only architecture)
  3. Softmax normalization
  4. Attention context (scores · V)
  5. Optional snapshot gathering for validation
- **Challenge**: Unmasked vs masked score computation for snapshot
- **Struct**: AttentionScoresResult (scores, softmax, context, global tensors)

### Stage 8: Extract projectAndGatherOutput()
- **Scope**: STEP 8 (~174 lines) - Output projection and MPI gather
- **Logic**:
  1. Linear projection (context → output)
  2. Optional bias addition
  3. Multi-rank MPI_Allgatherv for global output
  4. Single-rank pass-through
  5. Optional snapshot for final validation
- **Struct**: OutputProjectionResult (output, global output)

### Stage 9: Final Cleanup and Documentation
- **Scope**: Remaining orchestration polish
- **Logic**:
  1. Consolidate result extraction
  2. Simplify control flow
  3. Add comprehensive documentation
  4. Performance profiling
- **Target**: execute() ≤ 150 lines (pure orchestration)

**Projected Completion**: 3 more stages (~563 lines remaining)
**Expected Final State**: execute() ~135 lines, 9 helper methods, -94% reduction

---

## Conclusion

Stage 6 successfully extracted GQA expansion logic with Phase 3+4+5 multi-rank optimizations fully preserved. All 2,544 parity test comparisons passed with 100% success rate. Execute() reduced to 735 lines (68% total reduction from original). Helper method `handleGQAExpansion()` cleanly encapsulates complex MPI gathering and KV head replication logic.

**Stage 6 Achievement**: Complex optimization pipeline (fused gather + metadata cache + zero-copy) extracted without performance regression. Ready for Stage 7 (attention scores).

**Overall Progress**: 6/9 stages complete, 85% of refactoring complete, 100% test pass rate maintained across all stages. On track for ~94% final reduction.

