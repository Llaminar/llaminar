# Unit Testing Improvement Plan (Pipeline Kernel Parity Initiative)

Date: 2025-09-29
Author: Generated from diagnostic session

## Objective
Strengthen fine‑grained unit and micro-integration test coverage for all transformer pipeline kernel components so that functional or numeric divergence (currently only surfacing in the slow golden end‑to‑end test) is isolated early and cheaply. Goal: be able to localize first-differing operation in < 5 seconds wall time without invoking llama.cpp baseline.

## Current Situation (Summary)
The golden parity test is slow due to full llama.cpp model load + quant repack. Divergence origin remains unclear because we lack unit tests that:
- Validate weight orientation correctness
- Assert attention scaling & masking
- Exercise RoPE rotation math independently
- Confirm SwiGLU activation ordering
- Verify LM head orientation / tie logic
- Provide layerwise checksum differentials
- Validate KV cache indexing across ranks
- Quantify effects of environment overrides

Existing tests cover embedding, RMSNorm, linear, attention partially, and several COSMA elementwise paths; but *MPIRoPEKernel*, *MPISwiGLUKernel*, *MPIResidualKernel*, LM head orientation, weight layout enforcement, and systematic per-layer divergence detection are missing.

## Guiding Principles
1. **Determinism First**: OMP threads = 1 unless test explicitly exercises threading.
2. **Reference First**: Each kernel test includes a naive scalar reference implementation with tight tolerances.
3. **Tight Tolerances**: Start with float-only numeric paths (abs ≤ 1e-6, rel L2 ≤ 1e-6) to make subtle orientation bugs visible.
4. **Fail Fast**: Early exit at first mismatch with a concise diff (index, ref, got, absolute, relative error).
5. **Layerwise Binary Search**: Provide infrastructure to find earliest divergence boundary.
6. **Isolation From Env Noise**: Assert disallowed environment overrides are unset for core correctness tests.

## Target Coverage Matrix
| Component | Current | Target State | Risk Addressed |
|-----------|---------|--------------|----------------|
| MPIEmbeddingKernel | Basic test | Add edge cases (OOV/boundaries) | Hidden mismatch via vocab partition |
| MPIRMSNormKernel | Exists | Add extreme magnitude + override tests | Drift from gamma forcing |
| MPILinearKernel | Exists | Orientation parity test | Transpose misuse |
| MPIAttentionKernel | Exists | Micro attention math test | Missing 1/sqrt(Dh), mask errors |
| MPIRoPEKernel | Missing | Dedicated RoPE test | Wrong angles / dims |
| MPISwiGLUKernel | Missing | Direct activation test | Activation ordering bug |
| MPIResidualKernel | Missing | Residual safety test | In-place alias errors |
| LM Head (orientation / tie) | Indirect | Orientation + one-hot test | Projection transpose |
| Weight Layout Enforcement | Implicit | Idempotency test | Double-enforce issues |
| Layerwise Divergence Harness | Missing | Checkpoint checksums | Earliest drift localization |
| KV Cache Indexing | Missing | Multi-rank slot test | Attention key misalignment |
| Env Overrides Effects | Missing | Quantified delta tests | Silent global gamma forcing |

## Implementation Order (High → Low Impact)
1. **Linear Orientation Parity Test** (`test_linear_orientation_parity.cpp`)
2. **Attention Micro Test** (`test_attention_micro.cpp`)
3. **SwiGLU Kernel Test** (`test_mpi_swiglu_kernel.cpp`)
4. **RoPE Kernel Test** (`test_mpi_rope_kernel.cpp`)
5. **LM Head Orientation Test** (`test_lm_head_orientation.cpp`)
6. **Layerwise Checksum Harness** (`test_layerwise_checksum.cpp`)
7. **Residual Kernel Test** (`test_mpi_residual_kernel.cpp`)
8. **KV Cache Indexing Test** (`test_kv_cache_indexing.cpp`)
9. **Env Overrides Effect Test** (`test_env_overrides_effect.cpp`)
10. **Weight Layout Enforcement Test** (`test_weight_layout_enforcement.cpp`)

Support Utilities (created alongside first test wave):
- `tests/test_tensor_utils.h` (metrics + compare helpers)
- `tests/test_reference_impls.h` (rope_ref, swiglu_ref, linear_ref, attention_ref)
- Deterministic PRNG (xoroshiro128+)

## Detailed Test Specs
### 1. Linear Orientation Parity
- Generate X[S,in], W_raw[out,in] (or [in,out]) to mimic GGUF layout.
- Enforce layout using production helper → run MPILinearKernel.
- Reference: explicit matmul with correct transpose decision.
- Assert abs ≤ 1e-6, rel L2 ≤ 1e-6.

### 2. Attention Micro
- Single head variant for clarity (configurable Dh=64).
- Random Q,K,V (seeded); scores_ref = QK^T / sqrt(Dh), causal mask.
- Compare scaling presence & row softmax sums (~1.0 ±1e-6).

### 3. SwiGLU
- Random gate/up; ref: `SiLU(gate) * up`.
- Stress test extremes (gate values ±40) for stability.

### 4. RoPE
- Build input [S,H,Dh]; apply MPIRoPEKernel.
- Reference rotates even/odd channel pairs using same θ base.
- Check positions {0,1,S-1}.

### 5. LM Head Orientation
- Simulate W_out[D,V]; craft one-hot hidden row to probe column retrieval.
- Tie fallback path check (when output.weight missing) returns embedding row.

### 6. Layerwise Checksum Harness
- Manual mini-pipeline of 1–2 layers using kernels directly.
- Capture checksums after: embedding, attn residual, ffn residual.
- Compare to integrated pipeline execution (same weights) – isolate earliest mismatch.

### 7. Residual Kernel
- Cases: distinct buffers; output aliases input A; mismatched shapes (negative test).
- Assert value equivalence and no overwrite of second operand region.

### 8. KV Cache Indexing
- Insert deterministic K,V per token index across ranks.
- Gather and verify ordering & continuity.

### 9. Env Overrides Effects
- With + without `LLAMINAR_ALL_RMSNORM_FORCE_UNIT` → gamma stddev approaches 0.
- Clamp env yields min/max inside bounds.
- Rope theta override changes computed cos/sin pairs predictably (log ratio of angles).

### 10. Weight Layout Enforcement
- Apply enforce twice (idempotent) → identical memory & pointer.
- Compare to explicit transpose copy to ensure numerical equivalence.

## Tolerances
| Category | abs | rel L2 |
|----------|-----|--------|
| Deterministic float kernel | 1e-6 | 1e-6 |
| MPI 2-rank reduction | 2e-6 | 2e-6 |
| Softmax rows | 2e-6 | 1e-6 |
| Post 2-layer accumulation | 5e-5 | 1e-5 |
| LM head logits (large dot) | 1e-4 | 2e-5 |

## Environment Hygiene
- Assert disallowed env overrides: unit gamma/clamp/orientation flags unset in core correctness tests.
- Force `OMP_NUM_THREADS=1` unless test name ends with `_Threads`.

## CI Labeling
- Add CTest labels: `KernelUnit`, `KernelMPI`, `Layerwise`, `EnvBehavior`.
- Fast lane: run all `KernelUnit` (<1s each) pre-integration.

## Meta Test
Add a test enumerating kernel registry; fail if new kernel appears without a matching test file name containing its class stem.

## Success Criteria
- Time to localize first divergent stage ≤ 5s.
- 100% of kernel classes possess at least one dedicated test.
- Golden full test becomes confirmatory, not investigative.
- Orientation bugs or scaling omissions caught by pre-commit CI.

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Over-fragility (tolerances too tight) | Layer instruments fallback macro to relax via env (e.g., `LLAMINAR_TEST_TOL_FACTOR`). |
| Drift in reference formulas | Centralize reference math in one header file. |
| Test runtime creep | Keep per-test tensor sizes tiny (S≤8, D≤64). |
| Hidden dependence on global state | Explicit env sanitation at test start.

## Rollout Plan
1. Scaffold utilities + first two high-priority tests (linear, attention).
2. Add RoPE & SwiGLU; observe any failures; fix cores.
3. Introduce layerwise checksum harness; integrate into CI gating for pipeline modifications.
4. Add remaining medium-risk tests (LM head, residual, KV).
5. Add env & layout enforcement tests.
6. Integrate meta test & CI labels.

## Follow-Up (Post Implementation)
- Extend to decode path (token-by-token incremental attention test).
- Add performance guard tests (budgeted micro-bench thresholds) once correctness stable.

---
*End of plan.*
