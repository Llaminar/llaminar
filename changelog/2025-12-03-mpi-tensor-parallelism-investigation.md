# 2025-12-03: MPI Tensor Parallelism Investigation and Partial Implementation

## Summary

Investigated MPI work distribution in Llaminar V2 and discovered that only attention is properly tensor-parallelized. All GEMM operations (QKV projections, FFN Gate/Up/Down) were being duplicated on all ranks.

## Key Findings

### What Was Working ✅
- **Attention Tensor Parallelism**: `MpiAttentionOrchestrator` properly implements head-parallel attention:
  - Heads split via `mpi_ctx->get_local_slice()`
  - Each rank computes only its assigned heads
  - Results combined via `mpi_ctx->allreduce_sum()`

### What Was Broken ❌
- **GEMM Tensor Parallelism**: `QuantisedGemmKernel::multiply_with_precomputed_q8_1` **ignores** the MPI context:
  ```cpp
  // src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h line ~1108
  (void)mpi_ctx;  // <-- MPI context explicitly ignored!
  ```
- This affects ~95% of compute: QKV, Wo, Gate/Up, Down projections
- All ranks were duplicating work

## Changes Made

### 1. Added `project_row_parallel()` method to `PipelineBase`

**Files:**
- `src/v2/pipelines/PipelineBase.h` - Added declaration
- `src/v2/pipelines/PipelineBase.cpp` - Added implementation

This method wraps the GEMM with an MPI allreduce for row-parallel projections (Wo, FFN Down). 

**Current behavior:** All ranks still compute full GEMM, but the allreduce ensures all ranks have consistent results. The output is divided by `world_size` before allreduce to maintain correctness when all ranks have identical outputs.

### 2. Added `allreduce_sum_inplace()` to `MPIContext`

**File:** `src/v2/utils/MPIContext.h`

New method using `MPI_IN_PLACE` for in-place allreduce operations, avoiding memory allocation.

### 3. Updated `Qwen2Pipeline` to use row-parallel projections

**File:** `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

- Changed `attention_block()` to use `project_row_parallel()` for Wo projection
- Changed `ffn_block()` to use `project_row_parallel()` for Down projection

### 4. Created tensor-parallel GEMM wrapper infrastructure

**New files:**
- `src/v2/kernels/cpu/gemm_v4/TPGemm.h` - Tensor-parallel GEMM wrapper
- `src/v2/kernels/cpu/gemm_v4/TPFusedGEMM.h` - Tensor-parallel fused GEMM for QKV/Gate/Up

These provide the API for future optimizations but are not yet integrated.

### 5. Added comprehensive documentation

**File:** `docs/v2/TENSOR_PARALLELISM_STATUS.md`

Documents current status, what works, what doesn't, and the implementation plan.

## Test Results

- All unit tests pass
- Manual testing confirms correct output:
  - `"Hello"` → `", I am a beginner"`
  - `"What is the capital of France?"` → `"The capital of France is Paris. It is the"`

## Performance Impact

**Current state:** No compute savings - allreduce adds ~10-20% overhead for row-parallel projections since all ranks still compute full GEMM.

**Why this matters:** Establishes the correct API pattern for when we implement true tensor-parallel GEMM (compute savings) in the future.

## Next Steps (Future Work)

### Phase 1: Column-Parallel GEMM Integration (Short-term)
- Integrate `TPFusedGEMM` for QKV and Gate/Up projections
- Each rank extracts only its output columns after GEMM
- Saves memory bandwidth, not compute

### Phase 2: True Tensor-Parallel GEMM (Medium-term)
- Modify `QuantisedGemmKernel` to accept `n_start`, `n_end` parameters
- Only compute the local output column slice
- Saves both compute AND memory

### Phase 3: Weight Sharding (Long-term)
- Each rank loads only its slice of weights at model load time
- Enables natural tensor-parallel GEMM without kernel changes
- Saves memory on weights

## Files Changed

```
src/v2/utils/MPIContext.h                          # Added allreduce_sum_inplace()
src/v2/pipelines/PipelineBase.h                    # Added project_row_parallel()
src/v2/pipelines/PipelineBase.cpp                  # Implemented project_row_parallel()
src/v2/pipelines/qwen/Qwen2Pipeline.cpp            # Use project_row_parallel for Wo, Down
src/v2/kernels/cpu/gemm_v4/TPGemm.h               # New: TP GEMM wrapper
src/v2/kernels/cpu/gemm_v4/TPFusedGEMM.h          # New: TP fused GEMM
src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h  # Added getPackedWeights() accessor
docs/v2/TENSOR_PARALLELISM_STATUS.md              # New: Detailed status document
```

## Key Insight

The fundamental issue is that the GEMM kernels were designed for single-node execution and never implemented the MPI context functionality. The `(void)mpi_ctx;` in the kernel was a placeholder that was never filled in.

The cleanest long-term solution is weight sharding at load time, which allows the existing kernel to naturally compute only its portion without modification.
