# RMSNorm & Softmax Implementation Audit

_Last updated: 2025-09-26_

This document inventories every production code path that implements RMSNorm or softmax so we can collapse them onto a single shared kernel.

## RMSNorm Call Sites

| Location | Scope | Data Layout / Distribution | Notes |
| --- | --- | --- | --- |
| `src/mpi_transformer_pipeline.cpp` (`small_seq_fast_path` lambda) | Rank-0 fast path for tiny sequences | Row-major vectors on rank 0 | Inline lambda duplicates algorithm, ignores MPI and uses `config_.eps`. |
| `src/mpi_transformer_pipeline.cpp` (`executePrefill*` paths via `executeKernel("rmsnorm", ...)`) | Standard MPI pipeline | Delegates to `MPIRMSNormKernel` | Entry point for most layers outside COSMA short-circuit paths. |
| `src/cosma_prefill_manager.cpp` (`rmsnorm_in_layout`) | COSMA prefill attention path (distributed) | Operates on `CosmaView` (distributed blocks) | Owns MPI-aware implementation with per-rank reductions; also has optional validation compare. |
| `src/cosma_prefill_manager.cpp` (`fused_rmsnorm_qkv`) | COSMA fused RMSNorm + Q/K/V projection | Converts activations into COSMA layout | Calls `rmsnorm_in_layout` internally but keeps additional buffering / allocation logic. |
| `src/kernels/MPIRMSNormKernel.cpp` | General-purpose MPI kernel | Sequence-wise distribution via `MPI_Allreduce` | Intended reusable kernel but not invoked from COSMA path. |
| `src/kernels/RMSNormKernel.h` | Legacy single-process kernel | Row-major | Header-only fallback used in older graph code. |

**Key overlaps**
- All variants compute classic RMSNorm (`x / sqrt(mean(x^2) + eps)`), but epsilon usage is inconsistent (`config_.eps` vs kernel member).
- COSMA path maintains its own MPI reductions instead of reusing `MPIRMSNormKernel`.
- Small-seq fast path duplicates logic with separate scaling path.

## Softmax Call Sites

| Location | Scope | Data Layout / Distribution | Notes |
| --- | --- | --- | --- |
| `src/mpi_transformer_pipeline.cpp` (`executePrefillAttentionCosma`) | COSMA attention | Uses `CosmaPrefillManager::softmax_in_layout` on distributed views | Converts row-major scores into COSMA layout, runs distributed softmax, converts back. |
| `src/cosma_prefill_manager.cpp` (`softmax_in_layout`) | Core COSMA distributed softmax | Works on `CosmaView`; uses MPI all-reductions per row | Handles single-rank optimization and instrumentation counters. |
| `src/kernels/MPIAttentionKernel.cpp` (`computeLocalAttentionScores`) | MPI attention kernel (non-COSMA) | Local buffers per head, row-major | Implements masking + softmax manually per head. |
| `src/adaptive_transformer_pipeline.h` (`applySoftmax`) | Adaptive pipeline prototype | Row-major, optional MPI barriers | Uses simple MPI reductions without head awareness. |
| `src/chat/response_generator.cpp` (`ResponseGenerator::softmax`) | Sampling logits | Pure row-major | Unrelated to attention but serves as another softmax implementation. |

**Key overlaps**
- Attention softmax appears in four distinct forms: COSMA distributed, MPI per-head, adaptive pipeline helper, and simple row-major.
- Only COSMA variant tracks statistics; others lack consistent masking or MPI handling.
- No shared header/utility exists; all versions inline exponential and normalization loops.

## Next Steps

1. Design unified RMSNorm helper(s) that handle both row-major buffers and distributed shards, exposing a lightweight interface that both COSMA and MPI code can call.
2. Extract a softmax utility that accepts row-major buffers plus optional MPI context, then refactor attention kernels to delegate through it (head-aware wrapper for attention, generic routine for sampling).
3. Update test coverage to validate the shared implementations against the existing call paths before removing legacy code.
