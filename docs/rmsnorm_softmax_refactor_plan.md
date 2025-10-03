# Shared RMSNorm & Softmax Refactor Plan

_Last updated: 2025-09-26_

## Goals

1. **Single source of truth** for RMSNorm and softmax math so OpenBLAS, MPI, and COSMA paths share identical behavior.
2. **Composable adapters** that let existing call sites (row-major tensors, COSMA distributed views, attention-head loops) use the shared core without rewriting their control flow.
3. **Deterministic validation harness** that cross-checks all execution paths against a llama.cpp golden reference for prefill.

## Proposed Architecture

### 1. Common Kernels Library

Create a lightweight utility module under `src/kernels/common/` holding core routines:

```
src/kernels/common/
    rmsnorm_core.h / .cpp     # RMSNorm core & helpers (implemented; supersedes legacy normalization.{h,cpp})
    softmax.h / .cpp          # Softmax core & helpers (pending consolidation)
    mpi_utils.h / .cpp        # Thin wrappers for shared MPI reductions (existing bits we can reuse)
```

#### RMSNorm API Sketch

```cpp
struct RMSNormArgs {
    const float *input;    // length = rows * cols (row major)
    const float *weight;   // length = cols, optional (nullptr -> skipped)
    float *output;         // same shape as input
    int rows;
    int cols;
    float epsilon;
};

// Pure row-major kernel (no MPI).
void rmsnorm_row_major(const RMSNormArgs &args);

// Distributed reducer interface.
struct DistributedRMSNormContext {
    MPI_Comm comm;
    int world_size;
    int rank;
    // Optional scratch buffers for reductions (std::vector reused by caller).
};

// Each rank passes its local shard (rows_local, cols) plus the global row count
// so the routine can perform the correct global mean square reduction.
void rmsnorm_distributed(const RMSNormArgs &local_args,
                         int global_rows,
                         const DistributedRMSNormContext &ctx);
```

Adapters:
- `MPIRMSNormKernel` delegates to `rmsnorm_distributed`.
- COSMA path converts `CosmaView` shard pointers to `RMSNormArgs` and calls shared routine.
- Small-seq fast path calls `rmsnorm_row_major` directly.

#### Softmax API Sketch

```cpp
struct SoftmaxArgs {
    float *scores;  // in-place row-major, rows x cols
    int rows;
    int cols;
    bool apply_causal_mask; // optional mask j>i
    float scale;            // optional scaling (1.0f default)
};

void softmax_row_major(const SoftmaxArgs &args);

struct DistributedSoftmaxContext {
    MPI_Comm comm;
    int world_size;
    int rank;
};

// Each rank provides its slice (rows_local) but still needs global reductions per row.
// The routine handles MPI max/sum with provided context.
void softmax_distributed(const SoftmaxArgs &local_args,
                         int global_rows,
                         const DistributedSoftmaxContext &ctx);
```

Adapters:
- COSMA distributed softmax wraps `softmax_distributed` (reusing existing row index mapping but without custom math loops).
- `MPIAttentionKernel::computeLocalAttentionScores` computes QK^T and then calls shared softmax with `apply_causal_mask=true`.
- `adaptive_transformer_pipeline` switches to shared row-major helper.
- Sampling (`ResponseGenerator`) keeps using a simplified wrapper (no masking) that forwards to `softmax_row_major`.

### 2. Integration Steps

1. **Introduce common module** with pure row-major implementations + optional MPI helpers.
2. **Refactor MPIRMSNormKernel** to call shared distributed helper and delete duplicate math.
3. **Refactor COSMA RMSNorm** to map `CosmaView` shards onto the shared distributed helper.
4. **Refactor small-seq fast path** to use shared row-major helper.
5. **Refactor attention softmax** (MPI + COSMA + adaptive) to use shared helpers.
6. **Update chat softmax** to call the row-major helper (no behavior change expected).
7. **Add unit tests** for both kernels (row-major + distributed) covering:
   - small tolerances
   - masking behavior
   - scale parameter
8. **Add integration tests** ensuring OpenBLAS and COSMA produce matching RMSNorm + attention outputs when `LLAMINAR_COSMA_COMPARE_REPLICATED=1` (should already be zero diff once routines unify).

### 3. Golden Prefill Harness (High-Level)

- Use llama.cpp as the reference implementation by linking against its static library (`llama.cpp/build/libllama.a` already available).
- Implement a test fixture (`tests/test_prefill_golden.cpp`):
  1. Load a small GGUF model via llama.cpp API.
  2. Run a single prefill forward pass (sequence length up to current COSMA thresholds).
  3. Capture per-layer RMSNorm outputs, attention scores, and final logits.
  4. Run Llaminar (OpenBLAS + COSMA paths) with the same inputs, compare tensors (tolerance configurable).
- Gate the test behind `LLAMINAR_ENABLE_GOLDEN_TESTS` to allow opt-in due to runtime cost.

### 4. Migration & Validation Checklist

- [ ] All RMSNorm code paths include shared header; no standalone inline implementations remain.
- [ ] Cosmos path instrumentation still emitted from wrapper (shared helper exposes hook for stats).
- [ ] Distributed reductions respect MPI barriers (existing wrapper or optional barrier flag).
- [ ] Softmax causal masking behavior matches existing semantics (verify with regression tests).
- [ ] Golden harness passes for both OpenBLAS and COSMA settings.
- [ ] Documentation updated (developer guide & inline comments) to reflect the unified kernels.

## Open Questions

- Do we need mixed-precision support in the shared kernels (fp16/bfloat16)? For now plan on float only; add templates later if needed.
- Should we expose scratch buffer ownership to callers to avoid allocations? Initial version can allocate internally; optimize later if profiling justifies.
- For COSMA `softmax_in_layout`, we still have to map global→local indices. We’ll keep the existing mapping logic but route per-row math through the shared helper.

## Next Steps

1. (DONE) Implement `src/kernels/common/rmsnorm_core.{h,cpp}` with row-major + distributed helpers (legacy normalization removed).
2. (DONE) Refactor `MPIRMSNormKernel` to use shared `rmsnorm_core` (parity test added: `RMSNormCoreParity`).
3. (DONE) Wire COSMA fused & small-seq paths to `rmsnorm_core` helpers (instrumentation preserved).
4. (PENDING) Implement `softmax.{h,cpp}` unification and migrate attention kernels.
5. (PENDING) Start scaffolding the llama.cpp golden test harness once softmax path unified.
