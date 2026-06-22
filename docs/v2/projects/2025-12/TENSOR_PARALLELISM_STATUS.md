# Tensor Parallelism Status in Llaminar V2

## Executive Summary

**Current State**: Only attention is properly tensor-parallelized. All GEMM operations (QKV projections, FFN Gate/Up/Down) are **duplicated on all ranks**.

**Impact**: With 2 MPI ranks, we're doing 2x the compute we need for ~95% of operations.

## Current Implementation Status

### ✅ Working: Attention Tensor Parallelism

The `MpiAttentionOrchestrator` properly implements head-parallel attention:

1. **Head Distribution**: `mpi_ctx->get_local_slice()` splits heads across ranks
2. **Local Compute**: Each rank computes only its assigned heads
3. **AllReduce**: Results combined with `mpi_ctx->allreduce_sum()`

Location: `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp`

```cpp
// Line ~500
config.mpi_ctx->allreduce_sum(
    send_buffer.data(),
    mpi_output_buffer,
    total_tokens * config.n_heads * config.head_dim);
```

### ❌ Not Working: GEMM Tensor Parallelism

The `QuantisedGemmKernel::multiply_with_precomputed_q8_1` **ignores** the MPI context:

```cpp
// src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h line ~1108
(void)mpi_ctx;  // <-- MPI context explicitly ignored!
```

This affects:
- QKV projections (column-parallel candidates)
- Wo output projection (row-parallel, needs allreduce)
- FFN Gate/Up projections (column-parallel candidates)
- FFN Down projection (row-parallel, needs allreduce)
- LM Head projection

## Tensor Parallelism Strategy

For a transformer with MPI tensor parallelism:

### Column-Parallel Operations (No Communication)
Split output columns across ranks:
- **QKV projections**: Each rank computes Q[:, local_cols], K[:, local_cols], V[:, local_cols]
- **FFN Gate/Up**: Each rank computes gate[:, local_cols], up[:, local_cols]

### Row-Parallel Operations (Requires AllReduce)
Split input, each rank produces partial output that must be summed:
- **Wo (attention output)**: After attention, need allreduce
- **FFN Down**: After SwiGLU, need allreduce

### Current Flow (BROKEN)
```
Input → QKV (full on all ranks) → Attention (parallel ✅) → Wo (full) → Residual
     → FFN Norm → Gate/Up (full) → SwiGLU → Down (full) → Residual
```

### Correct Flow
```
Input → QKV (column-parallel) → Attention (parallel) → Wo (row-parallel + allreduce) → Residual
     → FFN Norm → Gate/Up (column-parallel) → SwiGLU → Down (row-parallel + allreduce) → Residual
```

## Implementation Plan

### Phase 1: Correctness (No Compute Savings)
Keep full GEMM computation but add allreduce after row-parallel projections to ensure all ranks have consistent state.

Status: **Not yet implemented**

### Phase 2: Column-Parallel GEMM Wrapper
Use `TPFusedGEMM` to extract only local columns after full GEMM computation.
- Saves memory bandwidth on intermediate tensors
- Does NOT save compute

Status: Files created but not integrated
- `src/v2/kernels/cpu/gemm_v4/TPGemm.h`
- `src/v2/kernels/cpu/gemm_v4/TPFusedGEMM.h`

### Phase 3: True Column-Parallel GEMM
Modify `QuantisedGemmKernel` to accept `n_start`, `n_end` parameters and only compute local output columns.
- Saves both compute AND memory

Status: Attempted but reverted due to complexity
- Kernel internals (block zeroing, N-iteration loops) make this non-trivial

### Phase 4: Weight Sharding
Each rank loads only its slice of weights at model load time.
- Saves memory on weights
- Enables proper tensor-parallel GEMM naturally

Status: Not started

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` | Core GEMM kernel | MPI ignored |
| `src/v2/kernels/cpu/gemm_v4/FusedGEMM.h` | Fused Q/K/V and Gate/Up | Wrapper, no parallelism |
| `src/v2/kernels/cpu/gemm_v4/TPGemm.h` | TP GEMM wrapper | Created, not integrated |
| `src/v2/kernels/cpu/gemm_v4/TPFusedGEMM.h` | TP fused GEMM | Created, not integrated |
| `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp` | Attention orchestrator | Working ✅ |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Pipeline orchestration | Uses full GEMM |

## Verification

To verify the current state:

```bash
# Run with 2 ranks and watch for GEMM calls
LLAMINAR_LOG_LEVEL=TRACE mpirun -np 2 ./run_llaminar.sh -m model.gguf -p "test" -n 1

# Check if both ranks are computing identical GEMMs (they are)
```

## Performance Impact

With 2 ranks on Qwen2.5-0.5B:
- **Attention**: ~50% speedup (parallel heads) ✅
- **QKV GEMM**: 0% speedup (duplicated) ❌
- **FFN GEMM**: 0% speedup (duplicated) ❌

Estimated overall speedup: ~10-15% (only attention benefits)
Expected speedup with full TP: ~80-90%

## Next Steps

1. **Quick Fix**: Add documentation/warning about current limitations
2. **Medium-Term**: Implement weight sharding at load time (cleanest solution)
3. **Long-Term**: Consider kernel-level column slicing for maximum efficiency
