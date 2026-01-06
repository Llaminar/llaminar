# Q16 Attention Kernel — Target Architecture (Accuracy-First, FP32 P×V)

**Date:** 2026-01-05  
**Status:** Draft (implementation plan included)  

This document defines the new target architecture for the Q16 attention kernel, with **accuracy as the primary goal** and an explicit initial choice:

- **Option B (initially):** compute **P×V in FP32**.

This is a pragmatic intermediate architecture: keep the pieces that are already working and well-tested (integer Q×K, exp2 LUT softmax weight generation, VNNI-friendly layout, Wo projection + residual add), while correcting the key broken assumption: **V is not uniformly-scaled in practice** (often Q8_1 with per-block scale), so “pure integer P×V” is not valid without additional fixed-point scale folding.

---

## 1) Current Code Layout (What Exists Today)

### Macro kernel
- `src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.{h,cpp}`
  - Validates dimensions and pointers.
  - Extracts Q head scales (from Q16 block headers).
  - Extracts K/V head scales (currently from block headers; per-position K scale path is **disabled** in the macro).
  - Allocates a temporary Wo output buffer.
  - Calls `q16_integer_attention_reference(params)` (reference pipeline).

### Reference pipeline (composition of microkernels)
- `src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.{h,cpp}`
  - Implements decode and prefill paths.
  - Calls microkernels under `src/v2/kernels/cpu/attention/q16_1/ref/microkernels/`.

Key microkernels:
- `Q16DotProduct` — integer Q×K (INT16×INT16 → INT32)
- `Exp2Core` + `Exp2FixedSoftmax` — LUT-based exp2 primitives + normalization
- `OnlineSoftmax` — streaming softmax update logic (V1 + V2 exist)
- `PVAccumulate` — integer P×V (INT16 weights × INT16 V → INT32)
- `WoProjection` — INT32 context → Q16 output using packed Wo + VPDPWSSD-friendly loops

---

## 2) What’s Working (Keep As-Is)

These pieces are delivering value today and should be retained in the target architecture.

### A. Macro-kernel responsibilities
Keep in `Q16FusedAttentionKernel`:
- Parameter validation (shape constraints, head_dim % 32, GQA divisibility).
- Decode vs prefill dispatch by `seq_len_q`.
- Output aliasing protection (temporary Wo output buffer to avoid residual add corruption).

### B. Integer Q×K scoring
Keep:
- `microkernels::q16_dot_single` / `microkernels::q16_qk_gemv` patterns.
- INT32 score domain and causal masking behavior.

Rationale:
- Q×K is the most compute-heavy piece; it maps well to VNNI.
- It does *not* require V’s scale uniformity.

### C. Exp2 LUT weight generation
Keep:
- `Exp2Core` LUT primitives (`exp2_compute_block_weights`, `exp2_compute_rescale`, etc.)
- Adaptive alpha strategy (keeps usable precision for small alpha).

Rationale:
- Stable, deterministic, JIT-friendly.
- Central place to manage alpha/scaling (including LOG2E handling).

### D. Wo projection + residual add
Keep:
- `wo_projection_vnni_int16` interface and packed weight usage.
- Final Q16 output layout (BLOCK_32 output for residual compatibility).
- `simd::q16_1_add_q16_1` residual add.

Rationale:
- Already integrates with the rest of the pipeline and packed weights.

---

## 3) What’s Broken (Must Change)

### A. The “uniform KV scale” assumption does not hold
Observed in real dumps / integration runs:
- **V is often Q8_1** with **per-block scale `d`**, not a single per-head scale.
- Even if Q/K are Q16 with a single scale per head (ideal), V is not.

Impact:
- The existing integer P×V (`PVAccumulate`) is only correct if V has a single consistent scale per head (or if we incorporate per-block scales into weights or accumulation).

### B. V1 OnlineSoftmax includes FP leakage in a hot-path invariant
V1 uses:
- `double l_processed` and repeated merge/normalize operations.

This creates:
- hard-to-reason-about numeric drift,
- repeated FP work in the loop,
- mismatch with the intended “integer core”.

We already have a V2 direction (deferred normalization) that reduces this problem.

### C. Per-position K scale path is currently disabled (and needs a correct data source)
Current macro disables `params.k_head_scales` per-position logic because it previously read the wrong quantity from K’s `.d` field.

Target architecture must either:
- provide a **correct per-position K normalization scale** source (sidecar array or augmented cache metadata), or
- intentionally stick to uniform K for now (accuracy impact likely for HybridQ16).

---

## 4) Target Architecture (Accuracy-First): End-to-End Dataflow

### 4.1 Data types and ownership
**Q:** Q16_1 (block size 64/128) — integer dot product input  
**K:** Q16_1 (block size 64/128) — integer dot product input  
**V:** Q8_1 (32-element blocks, per-block scale) — used only in FP32 path initially  
**Wo:** packed INT8 weights (existing `QuantisedPackedWeights`)  
**Output:** Q16_1 BLOCK_32 (for residual add compatibility)

### 4.2 Decode path (seq_len_q = 1)
For each query head `h` and its mapped kv head `kv_h`:

1) **Score block** (integer)
- Compute scores for a KV block:
  - `scores_i32[k] = dot_i16(Q_h, K_{kv_h, k})`

2) **Online softmax update** (integer weight generation + max tracking)
- Use V2-style state:
  - track running max `m`
  - compute `scale_num/scale_shift` when `m` increases
  - compute per-position weights using exp2 LUT
  - maintain running sums of weights (prefer integer sums; convert only when needed)

3) **Rescale previous state if max increased**
When max changes, previous accumulated numerator and denominator must be rescaled by:

$$\text{scale\_factor} \approx \frac{\text{scale\_num}}{2^{\text{scale\_shift}}}$$

For accuracy-first:
- apply `scale_factor` using `ldexp(scale_num, -scale_shift)` as FP32/FP64
- rescale:
  - `sum_w *= scale_factor`
  - `context_sum[d] *= scale_factor`

4) **P×V accumulation (FP32)**
- Dequantize V on the fly:
  - for each Q8_1 block: `v_fp32[i] = v_i8[i] * block.d`
- Accumulate unnormalized numerator:
  - `context_sum[d] += w_unorm[k] * v_fp32[d]`
  - `sum_w += w_unorm[k]`

5) **Finalize (single division)**
- After all KV blocks:
  - `context_fp32[d] = context_sum[d] / sum_w`

6) **Bridge back into existing Wo projection (minimal disruption)**
Because `wo_projection_vnni_int16` expects an INT32 context + `context_scale`:
- choose a per-query `ctx_scale` and quantize:
  - `context_i32[d] = round(context_fp32[d] / ctx_scale)`
- call:
  - `wo_projection_vnni_int16(context_i32, ctx_scale, Wo_packed, out_q16, ...)`

7) **Residual add**
- unchanged: `q16_1_add_q16_1(out_q16, residual_in, residual_out)`

### 4.3 Prefill path (seq_len_q > 1)
Same conceptual flow, but tiled:
- Q×K tiled (`Br×Bc`)
- softmax weights computed per row
- P×V FP32 accumulation per row
- finalize per row
- batched Wo projection (existing `wo_projection_vnni_int16_batched`) after converting each row’s FP32 context to INT32 + ctx_scale

---

## 5) “What We Keep” vs “What We Change” (Summary)

### Keep
- Macro-kernel structure and parameter validation.
- Q×K dot product integer domain.
- Exp2 LUT primitives and adaptive alpha machinery.
- Wo projection interface + packed weights.
- Residual add + output format.

### Change
- **V representation expectation:** update the pipeline to treat V as Q8_1 (at least in the accuracy-first phase).
- **Online softmax core:** use V2 deferred-normalization state; eliminate the `double l_processed` merge path.
- **P×V:** replace `PVAccumulate` (integer) with a new FP32 accumulation microkernel.
- **Scale plumbing:**
  - remove reliance on “single pv_scale per head” derived from V headers
  - introduce an explicit `ctx_scale` bridging step to reuse existing Wo projection
- **Per-position K scaling:** reintroduce only once the scale source is correct (likely a per-layer sidecar buffer).

---

## 6) Required API / Data-Contract Updates (Target)

### A. New attention params for HybridQ16+FP32PV
Current `Q16IntegerAttentionParams` assumes `V` is Q16_1 via `Q16BlockPtr`. Target needs a version that supports:
- `Q` / `K`: Q16BlockPtr
- `V`: pointer to Q8_1 blocks (or a typed Tensor interface)
- optional per-position K scale source:
  - `k_position_scales` OR a new KV-cache sidecar tensor

### B. Kernel snapshot info must reflect reality
If the production path uses Q8_1 V:
- update snapshot declarations so V is captured as Q8_1 (not Q16_1), or declare “V may be Q16_1 or Q8_1 depending on backend”.

---

## 7) Implementation Plan (Concrete Steps)

### Phase 0 — Uniform Q16_1 block size across the graph (activation contract refactor)

Goal: make **Q16_1 block size** a first-class part of the activation contract so we can stop hard-coding `BLOCK_32` in residuals / norms / embedding / allreduce, and stop using unsafe `typed_data()` accessors that silently assume 32-element blocks.

Why this is required:
- Today, `ActivationFormat` does not represent Q16_1 at all (Q16 kernels return `ActivationFormat::Q8_1`), and multiple stages hard-check `num_elements % 32 == 0` and use `Q16_1Block*` via `mutable_typed_data()`.
- `Q16_1Tensor` already supports custom block sizes (32/64/128) and provides safe accessors (`as_block_32/64/128`), but most generic graph stages are not wired to use them.

Deliverables:
1) **Pick a single Q16_1 activation block size per run/model** ("uniform across the graph")
   - Add a single source of truth in model/runtime config (e.g., derived from model metadata).
   - Selection rule should prioritize *graph-wide compatibility*, not head_dim convenience:
     - Must cleanly tile `d_model` (and other activation widths used by GEMMs / residuals / norms).
     - If ambiguous, default to 32 until the kernel ecosystem is updated.

2) **Make activation format encode Q16_1 + block size**
   - Option A (recommended): introduce a small struct (e.g., `ActivationLayout{ActivationFormat format; Q16BlockSize q16_block;}`) used in buffer specs + stage contracts.
   - Option B: extend `ActivationFormat` with explicit `Q16_1_32/Q16_1_64/Q16_1_128` (more invasive / enum growth).

3) **Plumb the block size through GraphOrchestrator buffer specs and stage params**
   - Ensure every Q16 residual/activation buffer is allocated as `Q16_1Tensor(shape, block_size)`.
   - Enforce at runtime: a stage that consumes Q16 activations asserts the tensor's block size matches the activation contract.

4) **Refactor the core graph stages that currently hard-code 32**
   - Residual add:
     - Update `ResidualAddStage` to remove `num_elements % 32` checks.
     - Replace `mutable_typed_data()` usage with block-size dispatch and block-size-aware SIMD helpers.
   - Norms:
     - Update Q16 RMSNorm paths to operate on variable block sizes (dispatch on `Q16BlockSize`).
   - Embedding / misc activation ops:
     - Replace "Q16_1 has 32 elements per block" assumptions with `block_size`-aware loops.
   - MPI allreduce:
     - Replace `allreduce_q16_1_inplace(Q16_1Block*)` usage in graph stages with the already-existing templated `allreduce_q16_inplace<BlockType>()` dispatch.

5) **Provide explicit conversion boundaries (only if needed)**
   - If some kernels remain BLOCK_32-only initially, add explicit conversion kernels/stages:
     - `Q16_1(block=64/128) -> Q16_1(block=32)` and/or the reverse.
   - The goal is to make conversions visible, not implicit.

6) **Tests + invariants**
   - Add small unit tests per stage type verifying it works for the configured Q16 block size.
   - Add a graph-level assertion: “all Q16 activation tensors share the configured block size”.

Notes / scope control:
- This Phase 0 is intentionally orthogonal to the attention math. It is about making the execution graph's activation contract consistent so attention work can stop carrying `BLOCK_32` baggage.

### Phase 0b — Document + constraints (this doc)
- Confirm the accuracy-first contract: FP32 P×V, reuse Wo projection via context bridging.

### Phase 1 — Reference implementation (decode first)
1. Add a new FP32 P×V microkernel (decode): `PVAccumulateFp32`.
2. Introduce a new `flash_decode_process_kv_block_fp32pv(...)`:
   - compute integer scores
   - compute LUT weights + rescale factor
   - rescale `context_sum_fp32` + `sum_w_fp32` when max increases
   - accumulate `w * dequant(V)`
3. Finalize context (single division).
4. Implement “bridge to Wo” quantization:
   - compute `ctx_scale` and `context_i32`
   - call existing `wo_projection_vnni_int16`

### Phase 2 — Prefill tiling
- Add FP32 P×V tile kernel and integrate into prefill online softmax path.
- Use `wo_projection_vnni_int16_batched` with per-row `ctx_scale`.

### Phase 3 — Correct per-position K scale integration
- Define and implement a correct scale source:
  - Option: per-layer sidecar tensor `K_scale[pos, kv_head]` written during RoPE / KV write.
- Thread it through stages and params.
- Enable `init_per_position` path without reading incorrect `.d` semantics.

### Phase 4 — JIT follow-up
- Once reference is correct, mirror the new microkernel boundaries in JIT:
  - JIT Q×K stays
  - JIT exp2 weights stays
  - JIT P×V becomes “dequant + FMA” (AVX512)

---

## 8) Validation Strategy (Accuracy First)

Minimum bar for acceptance:
- Replay tests on dumped tensors: V2/FP32PV ≥ baseline cosine and passes per-head checks.
- Integration tests: improved stability vs the old HybridQ16 failures.

Recommended:
- Add a focused unit test for per-position K scaling once scale source is correct.
- Maintain a small deterministic dump dataset for regression.

---

## 9) Open Questions

1) **Where does the correct per-position K scale live?**
- If `.d` is quant scale, we need an explicit normalization scale.

2) **What is the long-term V cache type?**
- If V stays Q8_1, Option A (pure integer P×V) requires fixed-point folding of per-block scales.

3) **Do we want to upgrade Wo to FP32 in the accuracy-first phase?**
- This doc keeps Wo integer by bridging context back into `wo_projection_vnni_int16`.

---

## References
- `docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md`
- `docs/v2/ANALYSIS_Q16_ONLINE_SOFTMAX_REWRITE.md`
- `docs/v2/ANALYSIS_VNNI_SAFE_DEFERRED_NORMALIZATION.md`
- `docs/v2/HANDOVER_HYBRIDQ16_ATTENTION_INVESTIGATION.md`
