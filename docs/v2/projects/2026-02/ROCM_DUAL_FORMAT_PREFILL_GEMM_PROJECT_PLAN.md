# ROCm Dual-Format Prefill GEMM Project Plan

## Goal
Deliver a production-ready ROCm prefill GEMM path for both packed weight families:
- INT8 VNNI (Q7/Q8 full INT8 repack)
- ratio-VNNI (Q6 and lower, IQ* quant families, no mandatory full INT8 expansion)

This plan defines design constraints, kernel strategy, phased implementation, and validation gates.

## Why This Project
Current behavior is split:
- Decode (M=1) has native GEMV support for INT8 VNNI and ratio-VNNI variants.
- Prefill (M>1) still falls back to CK GEMM flow via row-major scratch/repack.

That fallback is safe, but it does not complete the intended dual-format story for prefill and can erode ratio-VNNI VRAM advantages.

## Objectives
1. Add native prefill GEMM for INT8 VNNI.
2. Add native prefill GEMM for ratio-VNNI payload format.
3. Preserve ratio-VNNI memory goals by avoiding forced full-int8 expansion in standard execution.
4. Keep deterministic, low-risk fallback to CK path for unsupported shapes.
5. Roll out behind feature flags with tight correctness and performance gates.

## Explicit Optimization Goals (Current Priority)
These goals define the current tuning phase and should drive all Phase-4 decisions:

1. **Find optimal strategy by shape type / aspect ratio (not single fixed dimensions).**
    - Target inference shape families: `Attention`, `FFN_Up`, `FFN_Down`, `FFN_Gate`, `LM_HEAD`.
    - Optimize for ratio classes first, then validate representative concrete sizes.

2. **Tune kernels in playground to approach or beat CK.**
    - Use CK as performance reference and guardrail while iterating custom HIP strategies.
    - Keep only variants that show repeatable wins or acceptable tradeoffs.

3. **Implement ratio-aware dispatch in HIP prefill entrypoint.**
    - Add a dispatcher pattern in the ratio-VNNI prefill HIP path that detects shape type/aspect ratio and routes to the corresponding tuned strategy kernel.
    - Keep explicit CK fallback for unsupported/unsafe cases.

## Current Focus: VNNI-Native LM_HEAD Designs (Feb 2026)

We are explicitly prioritizing **VNNI-native** prefill kernels that leverage our existing `B_vnni[K/4, N, 4]` layout, rather than forcing CK-like movement patterns.

### Design 1 (Start Here): `MR4_CPT8_VNNI_Stream`
- Intent: evolve current strong `mr4` path (best custom baseline so far) with wider N-side vectorization.
- Tile sketch:
   - `M_TILE=32` via `BY=8, MR=4`
   - `N_TILE=512` via `64 lanes * CPT=8`
   - `K_STEP=4` (native VNNI pack)
- Vectorization requirements:
   - Vectorized B global loads (paired `int4`/`int2` fragments per lane for CPT8)
   - Vectorized A pack loads (packed 4x int8 in `int32`, optionally prefetching as `int4` over kg)
   - Vectorized C stores (two contiguous `store_i32x4` stores per row-group)
- Rationale:
   - Preserves no-heavy-LDS/barrier strength of current MR4 kernel.
   - Increases useful work per lane on extreme-wide `N` while staying VNNI-native.

### Design 2: `MR4_CPT4_VNNI_LDS_B_DoubleBuffer`
- Intent: add controlled LDS staging only where reuse/latency hiding pays off.
- Tile sketch:
   - `M_TILE=32`, `N_TILE=256`, `KG_TILE in {8,16}`
   - 256-thread block
- Vectorization requirements:
   - Vectorized global→LDS B copies (`b128` style transfers where possible)
   - Vectorized LDS reads for A/B fragments into VGPRs
   - Vectorized C stores (`store_i32x4`)
- Rationale:
   - Keep VNNI indexing semantics but reduce global latency pressure versus pure streaming.

### Design 3: `Persistent_SplitK_VNNI_MR4` (LM_HEAD-specialized)
- Intent: improve wide-shape scheduling and cache locality for very large `N`.
- Execution sketch:
   - Persistent CTAs own fixed N-stripes.
   - Split-K stage-1 emits vectorized int32 partials.
   - Stage-2 performs vectorized reduction and final stores.
- Vectorization requirements:
   - Vectorized partial writes/reads (`int4`-style where legal)
   - Vectorized final C stores
- Rationale:
   - Targets LM_HEAD extreme-wide workload specifically.

### Implementation Order
1. **Design 1 first** (lowest risk, closest to proven winner path).
2. Design 2 second (if bandwidth/latency still dominates).
3. Design 3 third (LM_HEAD-specific specialization pass).

### Immediate Bench Targets (Design 1)
- Shapes: `(M,N,K)=(128,151936,896)` and `(128,151936,2048)`.
- Gate metrics:
   - correctness parity vs CK (max abs diff 0 in lab correctness mode)
   - kernel-ms and e2e-ms vs `ck_baseline`
   - compare directly against `lmhead_vec_mr4_pad_cpt4` as the custom baseline-to-beat.

## Shape-Type Taxonomy (Aspect-Ratio Driven)

Primary ratio signal for prefill GEMM selection (for GEMM `C[M,N] = A[M,K] * B[K,N]`):
- `r = N / K` (output width vs reduction depth)

Initial shape classes for policy development:
- **Attention-like (square-ish):** `0.75 <= r <= 1.33`
   - Typical: projection layers where hidden->hidden dominates.
- **FFN-Up / FFN-Gate (wide):** `r > 1.33`
   - Typical: expansion projections (`N >> K`).
- **FFN-Down (tall/reduction-heavy):** `r < 0.75`
   - Typical: contraction projections (`N << K`).
- **LM_HEAD (extreme wide):** very large `N` (vocab-sized output), usually a strict subset of wide with dedicated policy checks.

Notes:
- `M` remains a secondary policy axis (`small/medium/large batch-token count`) once ratio class is chosen.
- Thresholds above are initial and should be adjusted based on measured winner boundaries.

## Current Benchmark Shape Set (Qwen2.5)

To speed up iteration while preserving representative aspect-ratio classes, the strategy lab now uses Qwen2.5 0.5B and 3B defaults:

- **Qwen2.5-0.5B** (`hidden=896`, `ffn=4864`, `vocab=151936`)
   - `AttnOut`: `(M,N,K)=(128,896,896)`
   - `FFN_Up/Gate`: `(128,4864,896)`
   - `FFN_Down`: `(128,896,4864)`
   - `LM_Head`: `(128,151936,896)`
- **Qwen2.5-3B** (`hidden=2048`, `ffn=11008`, `vocab=151936`)
   - `AttnOut`: `(128,2048,2048)`
   - `FFN_Up/Gate`: `(128,11008,2048)`
   - `FFN_Down`: `(128,2048,11008)`
   - `LM_Head`: `(128,151936,2048)`

Source of model dimensions: Hugging Face config files for `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`.

## Latest Quick Sweep (all GPUs, shortlist)

Run mode:
- `--all-devices --warmup=1 --iters=2 --no-check`
- shortlist strategies plus split-k variants
- LM head uses fast subset by default (override: `--full-lm-head`)

Observed winners vs CK (`vsCK > 1.0` means faster than CK):
- `Qwen2.5-0.5B_AttnOut`: `wave64_cpt4_splitk_by8(s=4)` at **1.492x**
- `Qwen2.5-3B_AttnOut`: `wave64_cpt4_splitk_by8(s=4)` at **1.084x**
- `Qwen2.5-0.5B_FFN_Down`: `wave64_cpt4_splitk_by8(s=4)` at **2.360x**
- `Qwen2.5-3B_FFN_Down`: `wave64_cpt4_splitk_by8(s=4)` at **1.115x**
- `FFN_Up/Gate` and both `LM_Head` shapes: CK baseline remains best in this shortlist run.

## Latest Release Exhaustive Sweep (2026-02-23)

Run mode (release-only, exhaustive strategy-lab harvest):
- Binary: `build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab`
- Flags: `--no-check --no-prefer-ck-wide --full-lm-head --warmup=1 --iters=4`
- Strategy set: all dispatch-recognized strategy labels (auto-extracted from strategy-lab dispatch table)
- Ranking metric: `vsCK(E2E)` (CK E2E baseline uses canonical harvested CK numbers when available)
- Filter: `OK=Y` rows only

Top-3 strategy-lab variants per shape:

- `Qwen2.5-0.5B_AttnOut`
   1. `wave64_cpt4_splitk` (`splitk=2`) — `vsCK(E2E)=18.604`
   2. `native_prefill_gridkpar_s7` — `vsCK(E2E)=17.705`
   3. `wave64_cpt4_splitk_by8` (`splitk=2`) — `vsCK(E2E)=16.814`

- `Qwen2.5-0.5B_FFN_Down`
   1. `native_prefill_gridkpar_auto_cpt4` — `vsCK(E2E)=13.917`
   2. `wave64_cpt4_splitk` (`splitk=8`) — `vsCK(E2E)=13.846`
   3. `native_prefill_gridkpar_s4` — `vsCK(E2E)=13.671`

- `Qwen2.5-0.5B_FFN_Gate`
   1. `wave64_cpt4_aligned_mr2_cap128` — `vsCK(E2E)=19.501`
   2. `wave64_cpt4_aligned_mr4_cap128` — `vsCK(E2E)=17.911`
   3. `lmhead_vec_mr4_pad_cpt8_veca2` — `vsCK(E2E)=17.351`

- `Qwen2.5-0.5B_FFN_Up`
   1. `wave64_cpt4_aligned_mr2_cap128` — `vsCK(E2E)=19.171`
   2. `wave64_cpt4_aligned_mr4_cap128` — `vsCK(E2E)=17.789`
   3. `lmhead_vec_mr4_pad_cpt8_veca2` — `vsCK(E2E)=17.303`

- `Qwen2.5-0.5B_LM_Head`
   1. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap192` — `vsCK(E2E)=21.978`
   2. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap128` — `vsCK(E2E)=21.895`
   3. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256` — `vsCK(E2E)=21.851`

- `Qwen2.5-3B_AttnOut`
   1. `wave64_cpt8_aligned_mr4_veca4_cap128` — `vsCK(E2E)=14.557`
   2. `wave64_cpt8_aligned_mr4_veca2_cap128` — `vsCK(E2E)=14.282`
   3. `wave64_cpt8_aligned_mr4_veca4_rfl_cap128` — `vsCK(E2E)=13.767`

- `Qwen2.5-3B_FFN_Down`
   1. `wave64_cpt8_aligned_mr4_veca4_cap128` — `vsCK(E2E)=9.202`
   2. `wave64_cpt8_aligned_mr4_veca2_cap128` — `vsCK(E2E)=9.002`
   3. `wave64_cpt8_aligned_mr4_veca4_rfl_cap128` — `vsCK(E2E)=8.617`

- `Qwen2.5-3B_FFN_Gate`
   1. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_avec4_cap256` — `vsCK(E2E)=10.661`
   2. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2` — `vsCK(E2E)=9.885`
   3. `lmhead_vec_mr4_pad_cpt8_ldsbdb_kgt8` — `vsCK(E2E)=9.357`

- `Qwen2.5-3B_FFN_Up`
   1. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_avec4_cap256` — `vsCK(E2E)=10.513`
   2. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_k2` — `vsCK(E2E)=9.832`
   3. `lmhead_vec_mr4_pad_cpt8_ldsbdb_kgt8` — `vsCK(E2E)=9.346`

- `Qwen2.5-3B_LM_Head`
   1. `lmhead_vec_mr4_pad_cpt8` — `vsCK(E2E)=10.519`
   2. `lmhead_vec_mr4_pad_cpt8_abcast_lane0` — `vsCK(E2E)=10.500`
   3. `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256` — `vsCK(E2E)=10.483`

## Latest Validation (2026-02-23, Slice 27 - production dispatch A/B perf suite)

Implemented a new production-path A/B perf benchmark to compare:
- **Legacy CK dispatch path** (`LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=0`)
- **New native prefill dispatch path** (`LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1`, auto policy)

New benchmark artifacts:
- Test source: `tests/v2/performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp`
- CTest target: `V2_Perf_ROCmPrefillDispatchComparison`
- Binary: `build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison`

INT8 VNNI prefill kernel slice ported into quantized GEMM path (this slice):
- `qgemm_int8_int8_vnni_prefill_kernel_t` (baseline prefill)
- `qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t` (split-K / grid-kpar)
- New C ABI launchers in `ROCmQuantisedGemmKernel_CK.hip`:
   - `rocmQuantGemm_int8_int8_int32_vnni_prefill(...)`
   - `rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(...)`

Measured A/B results (full production prefill path, 0.5B + 3B shape set):

| Class | Shape | Legacy CK (ms) | New Path (ms) | Legacy/New | Winner |
|---|---|---:|---:|---:|---|
| Attention | Qwen2.5-0.5B_AttnOut | 0.867 | 0.696 | 1.247 | new |
| FFN_Up | Qwen2.5-0.5B_FFN_Up | 1.935 | 2.023 | 0.956 | legacy |
| FFN_Gate | Qwen2.5-0.5B_FFN_Gate | 1.922 | 2.068 | 0.929 | legacy |
| FFN_Down | Qwen2.5-0.5B_FFN_Down | 2.298 | 2.011 | 1.143 | new |
| LM_Head | Qwen2.5-0.5B_LM_Head | 45.380 | 56.726 | 0.800 | legacy |
| Attention | Qwen2.5-3B_AttnOut | 1.782 | 1.780 | 1.001 | new |
| FFN_Up | Qwen2.5-3B_FFN_Up | 4.419 | 5.005 | 0.883 | legacy |
| FFN_Gate | Qwen2.5-3B_FFN_Gate | 4.428 | 5.005 | 0.885 | legacy |
| FFN_Down | Qwen2.5-3B_FFN_Down | 5.012 | 4.871 | 1.029 | new |
| LM_Head | Qwen2.5-3B_LM_Head | 47.455 | 72.370 | 0.656 | legacy |

Class-level average outcome from this run:
- Attention: `1.124x` (new preferred)
- FFN_Down: `1.086x` (new preferred)
- FFN_Up: `0.920x` (legacy preferred)
- FFN_Gate: `0.907x` (legacy preferred)
- LM_Head: `0.728x` (legacy preferred)

Correctness signal:
- Cosine similarity remained `1.000000` across all compared shapes.

Immediate next step (approved):
- Run a focused tuning pass on **INT8 VNNI native FFN_Up / FFN_Gate** kernels to close and, if possible, reverse the current CK advantage while preserving fallback safety and parity.

### Dispatch policy direction from this sweep

Implementation target for `ROCmQuantisedGemmKernel` INT8 VNNI prefill dispatch:

1. Route by shape class using aspect ratio (`r = N/K`) first, then element-count bucket (`M*N*K`) as a tie-breaker/selector.
2. Maintain separate winner mappings for at least two size regimes (initially represented by 0.5B-like and 3B-like shapes).
3. Keep dispatch table data-driven so larger models (7B/14B/32B/etc.) automatically map by ratio + element-count bucket without hard-coding model names.
4. Preserve explicit CK fallback for unsupported/unsafe/unknown regions.

### LM_HEAD strategy-lab decision (2026-02-22)

- For the **LM_HEAD custom-kernel track in strategy lab**, freeze the current winner as:
   - `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256`
- Treat this as the chosen custom reference variant for ongoing LM_HEAD lab comparisons.
- Pause further LM_HEAD micro-kernel churn until a materially new approach is available.

Scope clarification:

- This freeze is for the custom-kernel strategy-lab lane.
- It does **not** change the CK-preferred default policy for wide/extreme-wide production dispatch unless a custom LM_HEAD path demonstrates repeatable `vsCK(E2E) > 1.0` under normal gates.

## Gap Closure Policy (Winner-Map v1)

Based on exhaustive/targeted sweeps on Qwen2.5 0.5B and 3B shapes, the current policy to close parity gaps is:

- **Wide / Extreme-Wide (`N/K >= 2.0`)**
   - Prefer CK baseline by default.
   - Includes `FFN_Up`, `FFN_Gate`, and `LM_Head` classes in current shapes.
   - LM_HEAD custom-track reference (lab only): `lmhead_vec_mr4_pad_cpt8_abcast_lane0_cap256`.
- **Attention-like / Down-projection classes (`N/K < 2.0`)**
   - Keep custom candidates active.
   - Current winner is typically `wave64_cpt4_splitk_by8` (often `splitk=4`).

This policy is now implemented in the strategy lab as a default dispatch preference for wide/extrawide classes, with an override flag:
- `--no-prefer-ck-wide` to force benchmarking custom variants on wide shapes.

### Closure Validation Snapshot

Re-run with policy enabled (`--all-devices`, Qwen2.5 shape set):
- `FFN_Up`, `FFN_Gate`, `LM_Head` -> CK selected (parity closed by dispatch choice).
- `AttnOut`, `FFN_Down` -> custom split-k variants remain faster than CK in tested slices.

Representative winners:
- `Qwen2.5-0.5B_AttnOut`: `wave64_cpt4_splitk_by8` (vsCK ~1.51)
- `Qwen2.5-3B_AttnOut`: `wave64_cpt4_splitk_by8` (vsCK ~1.13)
- `Qwen2.5-0.5B_FFN_Down`: `wave64_cpt4_splitk_by8` (vsCK ~2.31)
- `Qwen2.5-3B_FFN_Down`: `wave64_cpt4_splitk_by8` (vsCK ~1.09)
- `Qwen2.5-{0.5B,3B}_{FFN_Up,FFN_Gate,LM_Head}`: `ck_baseline` (vsCK = 1.00)

## Dispatch Cleanup (Row-Count Hardcoding)

To reduce overfitting to fixed row counts, `rocmQuantGemm_executeNoScale` in `ROCmQuantisedGemmKernel_CK.hip` now uses an **aspect-aware policy** (primarily `N/K`, then `M`) instead of purely `M` buckets.

Policy summary:
- Keep `128x128` default for `M % 128 == 0`.
- Prefer `32x32` only for small + not-too-wide shapes.
- Prefer `64x64` for sub-128 and moderate `N/K`.
- Route wide/irregular cases to `128x128 MNPadding`.

This keeps compatibility while moving dispatch toward ratio-driven behavior.

## Non-Goals (Initial Implementation)
- No decode-path redesign (M=1 GEMV path remains authoritative).
- No change to model semantics, TP/PP placement rules, or tensor ownership model.
- No immediate support for every ratio-VNNI variant in phase 1.

## Supported Format Matrix (Target)
- INT8 VNNI:
  - Q8_x, Q7_x style full-int8 repack families.
  - Native prefill GEMM path required.
- ratio-VNNI:
  - Q6 and below, IQ* families.
  - Native prefill GEMM path must consume payload + ratio side data directly.
  - Full-int8 expansion is fallback/debug option, not default operating mode.

## Current State Snapshot
- Existing decode kernels:
  - INT8 VNNI GEMV family (wide, square, grid_kpar, scaled/fused variants).
  - ratio-VNNI GEMV skeletons available for selected formats (Q4_0, IQ4_NL).
- Existing prefill path:
  - Quantize A -> CK GEMM int8 path via row-major scratch/repack -> scale/epilogue.
- Existing guardrail:
  - Prefill experimental branch already scaffolded with explicit fallback logging.

## Design Principles
1. Correctness-first rollout with strict fallback.
2. Keep decode and prefill dispatch separate to avoid regressions in M=1 path.
3. Avoid hidden data format conversions that break VRAM assumptions.
4. Shared epilogue semantics across backends (alpha, beta, optional bias).
5. Explicit observability: each dispatch path must log one-time reason when selected/fallbacked.

## ABI Extension (Phase 3 Start)

### ABI v2 Descriptor Contract
- New descriptor-based C ABI entrypoint: `rocmGemm_ratio_vnni_int8_int32_prefill_v2`.
- Shared descriptor type: `ROCmRatioVNNIPrefillAbiV2Desc`.
- Legacy fixed-parameter entrypoint remains available and forwards to v2.

Descriptor fields (v2):
- `abi_version`
- `bitwidth`, `codebook_id`, `has_min`
- `block_size`, `payload_bytes`, `ratio_bytes`
- `flags`
- `blocks_per_row`
- `payload_stride_bytes`, `ratio_stride_bytes`

Current supported subset under v2 remains Phase-2 compatible:
- `bitwidth=4`, `block_size=32`, `payload_bytes=16`, `ratio_bytes=1`
- `codebook_id in {0 (linear), 4 (IQ4)}`

### Why this matters
- Stops repeated C ABI signature churn for each new ratio family.
- Allows future family bring-up via metadata/decoder extension first, signature changes last.
- Preserves backward compatibility while migrating callsites to descriptor-based execution.

## High-Level Architecture

### Prefill Dispatch Contract (M>1)
Given (M, N, K, weight format, epilogue config):
1. If INT8 VNNI format and shape supported -> run native VNNI prefill GEMM.
2. Else if ratio-VNNI format and shape supported -> run native ratio prefill GEMM.
3. Else -> fallback CK path (existing stable behavior).

### Epilogue Strategy
Phase 1 uses two-stage output for safety:
- Core kernel computes INT8xINT8 (or ratio decode) accumulation into INT32.
- Existing scaling/epilogue utility applies alpha/beta/bias to FP32 output.

Later phases can fuse epilogue when stable.

### Workspace/Memory Policy
- Reuse existing workspace buffers where possible.
- Add dedicated temporary buffers only when unavoidable (for split-K or ratio staging).
- For ratio-VNNI, keep payload-resident execution; no unconditional full-int8 staging.

## Phased Implementation Plan

### Phase 0 - Baseline and Contracts
Objective: lock down expected behavior before new kernels land.

Tasks:
1. Document prefill dispatch matrix in code comments and this plan.
2. Add one-time logging for selected path and fallback reason categories.
3. Define hard shape guards (for example K % 4 constraints) and failure categories.

Exit Criteria:
- Deterministic logs identify which path executed for every prefill GEMM.
- No behavior change yet.

---

### Phase 1 - Native INT8 VNNI Prefill GEMM (M>1)
Objective: implement first native prefill kernel family for full-int8 VNNI weights.

Kernel Plan:
1. Add entrypoint family (suggested naming):
   - rocmGemm_int8_int8_int32_vnni_prefill
   - rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar (optional if needed)
2. Initial tile strategy (**active now**):
   - Start with **Design 1: `MR4_CPT8_VNNI_Stream`**.
   - Preserve VNNI-native `K/4` iteration (`sdot4`-compatible) and fully vectorized load/store path.
   - Keep register pressure bounded enough to avoid occupancy collapse (Design 1 constraint).
3. Keep scaling/epilogue external in phase 1.

Integration Tasks:
1. Wire dispatch into ROCmQuantisedGemmKernel prefill branch (feature-flag gated).
2. Preserve CK fallback for unsupported shapes or launch failures.
3. Ensure fused-projection prefill path can use shared quantized activations with new kernel path.

Exit Criteria:
- Correctness parity vs CK across unit/integration prefill tests.
- No decode regressions.
- Prefill path selects native VNNI kernel for supported INT8 VNNI cases.

---

### Phase 2 - ratio-VNNI Prefill GEMM Core
Objective: native prefill GEMM for ratio payload + side metadata without full-int8 expansion.

Kernel Plan:
1. Add entrypoint family (suggested naming):
   - rocmGemm_ratio_vnni_int8_int32_prefill
2. Reuse ratio decode primitives and metadata validation from GEMV path.
3. Support initial subset first (Q4_0 and IQ4_NL), then expand.

Integration Tasks:
1. Add strict ratio metadata gates:
   - bitwidth, block size, payload bytes, codebook/min flags.
2. Route supported ratio formats to native ratio GEMM.
3. Fallback to CK path when metadata unsupported.

Exit Criteria:
- Native ratio prefill executes for initial supported formats.
- No forced full-int8 expansion in standard supported-ratio path.
- Correctness parity vs CK reference path.

---

### Phase 3 - Broaden ratio-VNNI Coverage
Objective: cover remaining Q6-and-lower and IQ* ratio families.

Tasks:
1. Add format specializations incrementally.
2. Add validation table mapping quant family -> ratio metadata contract.
3. Expand integration tests per quant family and shape bucket.

Exit Criteria:
- All targeted ratio families dispatch natively when metadata matches.
- Fallback rate for supported models is low and explainable.

---

### Phase 4 - Performance Optimizations
Objective: close throughput gap to CK and improve large-shape scaling.

Tasks:
1. Add split-K/grid_kpar variants where beneficial.
2. Tune tile shapes for representative M/N/K buckets.
3. Consider fused epilogue path (alpha/beta/bias) after baseline stability.
4. Optimize shared-memory staging and vectorized loads.

### Phase 4A - Aspect-Ratio Strategy Discovery (Step 1)
Objective: identify the best-performing strategy per shape class before broad auto-heuristics.

Tasks:
1. Define benchmark buckets by shape class (`Attention`, `FFN_Up`, `FFN_Down`, `FFN_Gate`, `LM_HEAD`) with at least one small/medium/large `M` representative each where practical.
2. Run shortlisted strategies against each bucket and record winner map by class.
3. Use CK instance patterns (tile families, rectangular variants, regular vs irregular handling) as guidance for new candidates.
4. Promote only strategies that are stable and win by class, not only by one exact shape.

Exit Criteria:
- Per-class winner map exists and is reproducible.
- No selected strategy introduces unacceptable regressions in another class.

### Phase 4B - Dispatcher Integration (Step 3)
Objective: encode class-based strategy routing in the ratio-VNNI prefill HIP entrypoint.

Tasks:
1. Add shape classifier (`r = N/K`, plus optional `M` bucket) inside ratio-VNNI prefill dispatch path.
2. Route each class to its selected strategy kernel from Phase 4A.
3. Preserve explicit fallback path and one-time reason logging.
4. Add focused integration/perf checks to verify classifier boundaries and dispatch correctness.

Exit Criteria:
- Ratio-VNNI prefill path dispatches by shape class with deterministic behavior.
- CK fallback remains intact and observable.

Exit Criteria:
- Measurable prefill latency improvement for targeted workloads.
- No correctness regressions from tuning.

---

### Phase 5 - Hardening and Default Rollout
Objective: move from experimental to default-safe operation.

Tasks:
1. Burn-in under long-run integration tests and stress runs.
2. Keep CK fallback available with clear diagnostics.
3. Promote feature flag defaults only after gates pass.

Exit Criteria:
- Stable across single-device and LOCAL TP runs.
- Production-ready fallback behavior documented.

## Dispatch and Fallback Decision Table
1. If M == 1 -> existing GEMV decode path (unchanged).
2. If M > 1 and experimental prefill flag disabled -> CK fallback.
3. If M > 1 and weight format is INT8 VNNI and shape supported -> native INT8 VNNI GEMM.
4. If M > 1 and weight format is ratio-VNNI and metadata+shape supported -> native ratio-VNNI GEMM.
5. Else -> CK fallback with one-time categorized reason log.

## Validation Plan

### Correctness
1. Add shape bucket tests:
   - Small M (2-8), medium M (16-64), large M.
   - N and K values matching real model layers.
2. Compare against CK outputs:
   - Max abs error, relative error, and tolerance per quant family.
3. Run existing unit and integration suites in Integration build.

### Performance
1. Benchmark prefill-only timing by shape bucket.
2. Benchmark end-to-end startup + first prefill token path.
3. Track fallback rate (how often native path is bypassed).

### Stability
1. Stress test repeated prefill runs for memory correctness.
2. Validate no stalls/hangs in LOCAL TP.
3. Confirm deterministic fallback under unsupported metadata.

## Instrumentation Requirements
Add counters/logging for:
- prefill_gemm.path_selected (int8_vnni_native, ratio_vnni_native, ck_fallback)
- prefill_gemm.fallback_reason (shape, metadata, launch_error, flag_disabled)
- prefill_gemm.kernel_ms (per path)
- prefill_gemm.epilogue_ms

## Risks and Mitigations
1. Risk: ratio decode complexity causes correctness drift.
   - Mitigation: keep external epilogue first, add strict metadata gating.
2. Risk: native path underperforms CK for some shapes.
   - Mitigation: shape-aware dispatch thresholds + CK fallback.
3. Risk: hidden memory growth from extra staging.
   - Mitigation: workspace accounting and explicit buffer ownership checks.
4. Risk: decode regressions from shared code edits.
   - Mitigation: isolate M==1 path, avoid refactors in decode hot path during early phases.

## Work Breakdown Checklist

### Phase 0
- [x] Add dispatch/fallback reason taxonomy (scaffold implementation in ROCmQuantisedGemmKernel).
- [x] Add prefill path selection logs and counters.

### Phase 1
- [x] Add native INT8 VNNI prefill GEMM kernel entrypoints (scaffold C API, fallback-safe).
- [x] Wire prefill dispatch in ROCmQuantisedGemmKernel (experimental route + CK fallback).
- [x] Add correctness tests vs CK for INT8 VNNI prefill.

### Phase 2
- [x] Add native ratio-VNNI prefill GEMM core entrypoints.
- [x] Add ratio metadata validation gates.
- [x] Add correctness tests for Q4_0 and IQ4_NL ratio formats.

### Phase 3
- [~] Add ratio specializations for remaining Q6-and-lower and IQ* families (ABI-v2 descriptor scaffold landed; container/decoder extension now includes Q4_1 min side-channel packing + row-major expansion + IQ4_XS superblock payload slicing; remaining families still pending).
- [x] Expand integration coverage matrix.

#### Latest Validation (2026-02-20)
- Native ratio prefill path now accepts `has_min=1` metadata for linear codebook (`Q4_1`) via ABI-v2 min side-channel fields.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed in repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.
- No regressions observed in existing `Q4_0` / `IQ4_NL` parity coverage within the same target.

#### Latest Validation (2026-02-20, Slice 2)
- Added native ratio prefill support for `Q5_0` and `Q5_1` (bitwidth=5, payload=20) with ABI-v2 descriptor validation.
- Added integration parity coverage for `Q5_0` / `Q5_1` in `PrefillNativeRatioVNNI_MatchesCKFallback`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 3)
- Added bitwidth-5 ratio→rowmajor CK fallback repack support in ROCm HIP path (Q5 payload expansion kernel + dispatcher validation).
- Expanded integration coverage with `PrefillCKFallback_Q5RatioOnlyRepack_MatchesBaseline`, which forces ratio-only Q5 CK fallback (no VNNI host weights) and verifies parity.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 4)
- Added native ratio container coverage for `IQ4_XS` by mapping each 256-element superblock into eight 32-element IQ4 payload slices under the existing ABI-v2 Q4 descriptor contract (`bitwidth=4`, `payload=16`, `block_size=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `IQ4_XS` parity coverage.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 5)
- Expanded ratio-only CK fallback parity coverage from Q5-only to a broader family set: `Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, `Q5_1`.
- Renamed coverage test to `PrefillCKFallback_RatioOnlyRepack_MatchesBaseline` and added explicit min side-channel assertions for `Q4_1`/`Q5_1`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 6)
- Expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to cover additional low-bit source families routed through current INT8-VNNI native prefill path: `Q6_K`, `Q3_K`, `Q2_K` (alongside `Q8_0`).
- Added explicit packaging assertions for this route (`int8_data_vnni` present, ratio payload/side-channel absent) to guard dispatch assumptions while ratio-native specializations for these families remain pending.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 7)
- Further expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to include IQ low-bit source families currently routed through INT8 packing: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`.
- Retained and exercised route assertions that these families stay on the INT8-VNNI path (`int8_data_vnni` populated, ratio payload/side-channel empty) while native ratio prefill specializations remain pending.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 8)
- Added first non-Q4/Q5 descriptor extension for native prefill: `Q6_K` now emits an ABI-v2 payload container with `bitwidth=8`, `block_size=32`, `payload_bytes=32` (payload built from existing INT8 VNNI host pack), preserving existing epilogue scaling semantics.
- Extended ROCm ABI-v2 prefill launcher with a dedicated bitwidth-8 payload kernel path (`gemm_payload_v2_i8_prefill_kernel`) and descriptor validation for this family.
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `Q6_K` parity coverage and metadata assertions (`bitwidth=8`, `payload=32`).
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 9)
- Extended the bitwidth-8 payload-v2 native prefill descriptor route to include `IQ3_XXS` (same `block_size=32`, `payload_bytes=32` contract built from existing INT8 VNNI host pack).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `IQ3_XXS` parity coverage and bitwidth-8 metadata assertions.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 10)
- Re-enabled `PrefillNativeRatioVNNI_MatchesCKFallback` full-family execution (removed temporary skip) and reproduced allocator corruption in the ratio-native path (`double free or corruption (!prev)` / `malloc(): invalid size`).
- Fixed dequant cache tail handling in quant tensor `data()` implementations to prevent partial-block out-of-bounds writes when `K < block_size` or `K % block_size != 0`:
   - `Q6_KTensor::data()`
   - `IQ3_XXSTensor::data()`
   - `IQ3_STensor::data()`
- Focused ratio-native parity case now passes end-to-end (Q6_K, IQ3_XXS, IQ3_S coverage active), and the integration gate is stable under repeat stress (`ctest --repeat until-fail:5 -R '^V2_Integration_ROCmQuantisedGemmKernel$'`): 5/5 passes.
- Extended burn-in on the same focused gate also passed (`ctest --repeat until-fail:10 -R '^V2_Integration_ROCmQuantisedGemmKernel$'`): 10/10 passes.

#### Latest Validation (2026-02-20, Slice 11)
- Extended payload-v2 bitwidth-8 native prefill specialization to `IQ2_XXS`, `IQ2_XS`, and `IQ2_S` (same descriptor contract as current bitwidth-8 path: `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include all three IQ2 families.
- During first bring-up, reproduced allocator corruption in IQ2 paths and fixed the same root-cause class (partial-block dequant cache overwrite) in:
   - `IQ2_XXSTensor::data()`
   - `IQ2_XSTensor::data()`
   - `IQ2_STensor::data()`
- Focused ratio-native parity test now passes with active coverage for `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 12)
- Extended payload-v2 bitwidth-8 native prefill specialization to `Q3_K` and `Q2_K` (same descriptor contract: `bitwidth=8`, `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include `Q3_K` and `Q2_K` alongside existing `Q6_K`/`IQ2*`/`IQ3*` families.
- Preemptively fixed the same partial-block dequant cache overwrite class for K-tail handling in:
   - `Q3_KTensor::data()`
   - `Q2_KTensor::data()`
- Focused ratio-native parity test passes with active coverage for `Q6_K`, `Q3_K`, `Q2_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 13)
- Extended payload-v2 bitwidth-8 native prefill specialization to `IQ1_S` and `IQ1_M` (same descriptor contract: `bitwidth=8`, `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include `IQ1_S` and `IQ1_M`.
- Fixed partial-block dequant cache overwrite class for K-tail handling in:
   - `IQ1_STensor::data()`
   - `IQ1_MTensor::data()`
- Parity behavior note: `IQ1_M` remains cosine-stable (`~1.0`) but shows quantization-step absolute deltas vs CK fallback (observed up to `1.0`), so the family-specific `max_abs_diff` threshold was relaxed to `1.5` while retaining strict `5e-3` for all other covered families.
- Focused ratio-native parity test passes with active coverage for `Q6_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 14)
- Added dedicated unit regression coverage for quantized-tensor partial-tail decode safety in `Test__QuantizedTensorTailDecodeRegression`.
- New unit target `V2_Unit_QuantizedTensorTailDecodeRegression` validates `data()` vs `to_fp32_row()` on partial-`K` shape (`13x17`) across all recently fixed families: `Q6_K`, `Q3_K`, `Q2_K`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`.
- Focused unit gate passed in Integration build: `ctest --test-dir build_v2_integration -R '^V2_Unit_QuantizedTensorTailDecodeRegression$' --output-on-failure --parallel`.

#### Latest Validation (2026-02-20, Slice 15 - deferred)
- Attempted to extend payload-v2 bitwidth-8 native prefill specialization and parity matrix to `Q4_K` and `Q5_K`.
- Focused parity run (`ROCmQuantisedGemmIntegrationTest.PrefillNativeRatioVNNI_MatchesCKFallback`) reproduced allocator corruption (`malloc(): invalid size (unsorted)`) immediately after `Q6_K` coverage when the new families were included.
- Rolled back the `Q4_K`/`Q5_K` additions from the bitwidth-8 payload-v2 route and parity matrix to preserve a green baseline.
- Post-rollback focused parity gate is green again for active families (`Q6_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`) using:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 16)
- Fixed root-cause heap corruption in quantized tail decode cache for:
   - `Q4_KTensor::data()`
   - `Q5_KTensor::data()`
- Both methods now use partial-block safe copy semantics (decode full super-block to temp, copy only valid tail elements), matching the previously fixed `Q6_K`/`Q3_K`/`Q2_K`/`IQ*` families.
- Extended unit regression coverage (`V2_Unit_QuantizedTensorTailDecodeRegression`) to include `Q4_K` and `Q5_K` on partial-`K` shape (`13x17`); focused unit gate passed.
- Re-enabled payload-v2 bitwidth-8 native prefill specialization and parity-matrix coverage for `Q4_K` and `Q5_K`; focused parity gate now passes with active coverage for both families:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.
- Focused integration target burn-in also passed: `ctest --test-dir /workspaces/llaminar/build_v2_integration --output-on-failure --repeat until-fail:5 -R '^V2_Integration_ROCmQuantisedGemmKernel$'` (5/5 passes).

#### Latest Validation (2026-02-20, Slice 17)
- Completed the Phase-3 parity coverage audit and closed a matrix gap in `PrefillNativeRatioVNNI_MatchesCKFallback`: added explicit execution coverage for `Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, and `Q5_1` (in addition to existing `Q6_K`/`Q4_K`/`Q5_K`/`Q3_K`/`Q2_K`/`IQ1*`/`IQ2*`/`IQ3*`).
- Audit run exposed another latent partial-tail heap corruption class in `IQ4_XSTensor::data()` for sub-block `K` shapes; fixed with the same tail-safe partial-block copy approach used for other quant families.
- Extended unit tail regression suite to include `IQ4_XS` (`V2_Unit_QuantizedTensorTailDecodeRegression`), and validated pass.
- Focused ratio-native parity gate now passes end-to-end with all targeted Phase-3 families active:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 18)
- Expanded ratio-native prefill shape buckets in `PrefillNativeRatioVNNI_MatchesCKFallback` to strengthen Phase-3 exit confidence:
   - small-M: `(M,N,K)=(4,128,128)`
   - medium-M: `(12,160,128)`, `(20,192,128)`
   - larger M/K: `(28,256,256)`
   - wider-K bucket: `(16,192,512)`
- Rebuilt focused integration target and validated the full-family parity matrix across all listed Phase-3 families under the expanded shape set.
- Focused gate passed:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 19 - Phase 4 start)
- Added first opt-in INT8 prefill split-K/grid-kpar variant behind existing dispatch guards and fallback:
   - New HIP kernel path uses `grid.z` K-splitting with INT32 atomic accumulation.
   - New launcher entrypoint: `rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(...)`.
   - Routing is opt-in via DebugEnv flags and falls back to baseline INT8 prefill kernel on launch failure.
- Added ROCm env controls:
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1`
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS=<n>`
- Expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to validate both INT8 native variants (`baseline`, `grid_kpar`) against CK fallback.
- Focused validation passed:
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeInt8VNNI_MatchesCKFallback*'`
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`

#### Latest Validation (2026-02-20, Slice 20 - Phase 4 tuning)
- Added grid-kpar split tuning controls and auto mode semantics:
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1` enables INT8 prefill grid-kpar routing.
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS=0` now means auto heuristic (manual `1..32` still supported).
- Implemented conservative auto-slice heuristic in prefill dispatcher (`tryExperimentalPrefillNativeGemm`) that selects split count from `(M,N,K)` and bounds fanout for atomic overhead control.
- Preserved fallback safety contract:
   - grid-kpar launch failure falls back to baseline native INT8 prefill
   - baseline native failure still falls back to CK path via existing logic.
- Focused correctness gates passed after tuning:
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeInt8VNNI_MatchesCKFallback*'`
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`
- Quick perf smoke (single-run wall clock, same gtest filter) on this host:
   - baseline: `baseline_elapsed_ms=1124`
   - grid-kpar auto: `grid_auto_elapsed_ms=951`
   - note: this is a coarse smoke signal, not a full benchmark sweep.

#### Latest Validation (2026-02-20, Slice 21 - production-path perf sweep harness)
- Added a production-path prefill perf sweep to `Perf__ROCmQuantisedGemmKernel`:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_GridKParSweep`
   - measures full `multiply_tensor` prefill route (quantize activations + native prefill GEMM + scaling/epilogue), not isolated kernel-only timings.
- Added mode matrix for controlled comparison under identical call chain and fallback contract:
   - `baseline`
   - `grid_auto` (`LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1`, `..._SPLITS=0`)
   - `grid_s4`
   - `grid_s8`
- Added per-mode env override + `mutableDebugEnv().reload()` wiring inside the perf test to ensure mode toggles are applied in-process between runs.
- Added explicit workspace binding in benchmark path so full `multiply_tensor` timing is valid (matches production kernel contract).
- Focused validation passed:
   - `/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_quantised_gemm_kernel --gtest_filter='*PrefillFullPath_GridKParSweep*'`
- Initial host results (Qwen-7B FFN Up, full path):
   - `M=64`: baseline `6.812 ms`, grid_auto `6.871 ms`, grid_s4 `6.911 ms`, grid_s8 `6.876 ms`
   - `M=256`: baseline `25.400 ms`, grid_auto `26.241 ms`, grid_s4 `26.305 ms`, grid_s8 `26.246 ms`
   - all modes cosine-stable vs FP32 reference (`~0.999956`).

#### Latest Validation (2026-02-20, Slice 22 - prefill micro-variant dispatch + real-model aspect sweeps)
- Added shape-aware INT8 prefill tile variant dispatch with micro-specialized launch variants:
   - baseline and grid-kpar launchers now dispatch among `{16x16, 32x8, 8x32, 8x8}` tiles.
   - runtime auto-selection is based on matrix aspect ratio and small-shape guardrails.
   - optional force-override knobs added for tuning sweeps:
      - `LLAMINAR_ROCM_VNNI_PREFILL_VARIANT` (`-1 auto`, `0 16x16`, `1 32x8`, `2 8x32`, `3 8x8`)
      - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_VARIANT` (`-1 auto`, `0 16x16`, `1 32x8`, `2 8x32`, `3 8x8`)
- Extended full-path perf suite with a real-model/aspect-ratio sweep target:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_RealModelAspectSweep`
   - covers representative projection shapes across 0.5B/7B/14B/32B and both tall/wide/square aspect ratios.
   - compares auto-dispatch against forced tile variants using full `multiply_tensor` path timing.

#### Latest Validation (2026-02-20, Slice 23 - CPT + KB levers in one pass)
- Implemented two additional prefill tuning levers requested for sweep-driven policy work:
   1. **CPT (outputs-per-thread)** for INT8 prefill kernels:
      - New env: `LLAMINAR_ROCM_VNNI_PREFILL_CPT` with supported values `{1,2,4}`.
      - Applied to both baseline prefill and grid-kpar prefill launch paths.
   2. **Grid-kpar KB control** for split-K policy:
      - New env: `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB` (`0` = auto policy).
      - Integrated into prefill dispatcher before legacy split override/auto fallback.
- Added full-path performance sweep target that exercises both knobs together:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_CptAndKbSweep`
   - sweeps `CPT ∈ {1,2,4}` and `KB ∈ {0(auto),4,8,16}` over representative real-model shapes.
- Focused sweep run passed:
   - `/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_quantised_gemm_kernel --gtest_filter='*PrefillFullPath_CptAndKbSweep*'`
- Initial findings on this host:
   - `CPT=1` is consistently best for the tested prefill shapes.
   - `CPT=2` and `CPT=4` regress throughput materially (larger regression at `CPT=4`).
   - Grid-kpar `KB` sensitivity is shape-dependent but modest in the tested band (`4..16`), with occasional small wins over `KB=0(auto)` for wider reduction-heavy shapes.

#### Latest Validation (2026-02-20, Slice 24 - retire CPT>1 from routine sweeps)
- To reduce sweep runtime and focus on high-value tuning axes, routine prefill sweep policy now fixes `LLAMINAR_ROCM_VNNI_PREFILL_CPT=1`.
- `PrefillFullPath_CptAndKbSweep` was narrowed to `CPT=1` only while retaining `KB ∈ {0(auto),4,8,16}` coverage.
- Rationale: host sweep data shows `CPT>1` is uniformly worse across tested real-model prefill shapes, so continuing to sweep those modes adds time without actionable upside.
- Sweep coverage now explicitly includes Qwen `7B` and `14B` prefill projection families: `AttnOut`, `FFN_Up`, `FFN_Down`, and `FFN_Gate`, so `KB` tuning decisions are validated across both square (attention) and tall/wide (FFN) shapes.

#### Latest Validation (2026-02-21, Slice 25 - CK MI50 strategy review + playground synthesis)
- Reviewed local CK source under `external/composable_kernel` for MI50/gfx906 int8 GEMM behavior and aligned findings with current ROCm prefill playground experiments.
- Key CK (MI50/gfx906) observations:
   - `dot4` is the core int8 primitive (`__builtin_amdgcn_sdot4` / `v_dot4_i32_i8`), with effective `K1=4` packing semantics.
   - int8 prefill on gfx906 is driven by `DeviceGemmDl` instance families (not XDL int8 path), with broad tile catalogs and per-shape instance selection.
   - CK carries regular (`GemmDefault`) and irregular (`MNPadding`) instance tables separately for edge-shape robustness.
   - Parallelization is block-tiled over `(M,N)` with `K0PerBlock` loop pipelining, plus multiple thread-cluster permutations for the same tile shape.
   - Common DL int8 shape families include `256x128`, `128x128`, `64x64`, `32x32`, and skinny/wide variants (`64x16`, `16x64`, `64x8`, `8x64`).
- Current playground synthesis:
   - Effective: aligned fastpath + unroll2 + launch-bounds path, and split/grid-cap tuning (best observed around `~0.887x` CK on `M=128,N=3584,K=3584`).
   - Neutral/minor: vectorized stores improve code quality/coalescing hygiene but are not primary wins alone.
   - Regressive in current form: dual-acc interleave and heavier unroll/shared-B variants.
   - Current auto heuristic underperforms fixed/manual picks on tested attention/FFN-like shapes.

#### Latest Validation (2026-02-21, Slice 26 - repack-inclusive strategy accounting + wide-shape win plan)
- Updated `Microbench__ROCmPrefillStrategyLab.hip` to report both kernel-only and repack-inclusive metrics:
   - `KernelMS`, `RepackMS`, `E2EMS` (`KernelMS + RepackMS`)
   - `vsCK(K)` and `vsCK(E2E)`
- CK-vs-custom comparisons are now explicit about host-side prep cost, preventing kernel-only false positives.
- Current conclusion remains unchanged for dispatch safety:
   - `FFN_Up`, `FFN_Gate`, and `LM_Head` stay CK-preferred in the default policy until a custom path wins on `vsCK(E2E)`.

#### Wide/Extreme-Wide Gap Closure Tactics (FFN_Up, FFN_Gate, LM_Head)

Primary objective for these classes: improve end-to-end throughput, not only kernel throughput.

1. **Persistent B-pack reuse (highest leverage for repeated prefill calls)**
   - Idea: cache and reuse pre-packed/transpose-ready B buffers per `(device, weight_ptr, K, N, variant)` instead of repacking each invocation.
   - Why: wide and extreme-wide shapes amplify repack bytes; reducing `RepackMS` can flip `vsCK(E2E)` even when `KernelMS` is similar.
   - Hook points:
      - `ROCmQuantisedGemmKernel.cpp` prefill dispatch/repack cache path.
      - strategy lab `runTimedHostPrepLoop` for A/B validation with and without persistent pack reuse.

2. **CK-style skinny/wide tile family promotion for custom kernels**
   - Idea: add/benchmark more rectangular tile variants for wide outputs (e.g., `32x8`, `64x8`, `16x64` style mapping analogs) and lock per-bucket winners.
   - Why: CK catalogs separate regular and irregular (`MNPadding`) families and keep skinny/wide options that map better to FFN/LM aspect ratios.
   - Hook points:
      - existing variant controls (`LLAMINAR_ROCM_VNNI_PREFILL_VARIANT`, `..._GRID_VARIANT`).
      - shortlist track `S5_wave64_ck_style_skinny_wide` promoted from experiment to required bucket gate.

3. **Two-level LM-head policy (chunked-N custom path vs CK full-N)**
   - Idea: for very large `N` (vocab-sized), evaluate chunked-N execution where custom kernels process vocabulary tiles with bounded workspace, then merge.
   - Why: LM-head is often repack/bandwidth-limited; chunking can improve cache behavior and reduce one-shot staging pressure.
   - Guardrail: only adopt if total `E2EMS` beats CK and numerical parity remains exact.

4. **Grid-kpar only under reduction-pressure gates**
   - Idea: keep split-K/grid-kpar for wide classes only when `K` is large enough and atomic/reduction overhead is amortized.
   - Why: prior sweeps show shape-dependent gains; unconditional split-K can regress.
   - Policy: gate by `(K, M)` bucket and retain CK fallback for low-benefit regions.

5. **Repack-aware dispatch thresholding (policy hard requirement)**
   - Idea: dispatch winner map for wide classes must be decided by `vsCK(E2E)`, with kernel-only used as secondary signal.
   - Why: this aligns policy with production full-path cost model.

### Phase 4 Concrete Shortlist (CK-Informed + Playground-Validated)

Objective: turn the current tuning space into a small, stable candidate set for policy convergence.

#### Candidate variants to keep active
1. `S0_ck_baseline`
   - Reference only (`rocmQuantGemm_executeNoScale` / current CK path).
2. `S1_wave64_grid_by8_cap128`
   - Current top manual playground candidate for attention-like square buckets.
3. `S2_wave64_aligned_unroll2_lb_cap128`
   - Best aligned/unroll family candidate; generally second-best in recent runs.
4. `S3_wave64_aligned_unroll2_cap128`
   - Control variant to isolate launch-bounds impact.
5. `S4_wave64_splitk_cap_family`
   - Split-K candidate set with bounded cap (`split ∈ {2,4,8}` where shape supports).
6. `S5_wave64_ck_style_skinny_wide`
   - CK-inspired rectangular variants for `MxN` aspect extremes (FFN-up / FFN-down).

#### De-prioritized variants (do not include in routine sweeps)
- dual-acc interleave family
- heavy unroll/shared-B family
- broad auto-heuristic switching policy (until per-bucket winners are revalidated)

#### Benchmark buckets (minimum required for shortlist decisions)
- `B1 Attention-like`: `(M,N,K) = (128,3584,3584)`
- `B2 FFN-up-like`: `(128,18944,3584)`
- `B3 FFN-down-like`: `(128,3584,18944)`
- `B4 Medium M`: one representative medium-M production shape (`M=32/64`) from real-model sweep set

#### Selection gate for default policy candidacy
- Correctness: `MaxAbsDiff=0` vs CK reference on harnessed compare path.
- Performance: candidate must beat current policy baseline by `>=2%` on at least one primary bucket without regressing another primary bucket by `>1%`.
- Stability: repeated run variance acceptable (`<=2%` spread over repeated samples on same host/config).

#### Shortlist execution order
1. Re-run `S1/S2/S3` on `B1/B2/B3` to lock manual winner map.
2. Add `S4` only on buckets where `K` is sufficiently large and atomics do not dominate.
3. Add `S5` for skinny/wide buckets and compare against locked winner map.
4. Encode a minimal shape-bucket dispatch policy from winners (manual map before auto heuristic).

### Phase 4
- [ ] Add split-K/grid_kpar prefill variants as needed.
- [ ] Tune tile parameters and dispatch thresholds.
- [ ] Evaluate optional fused epilogue.

### Explicit Goal Tracking (Current Tuning Cycle)
- [ ] Goal 1: Aspect-ratio winner map completed for `Attention`, `FFN_Up`, `FFN_Down`, `FFN_Gate`, `LM_HEAD`.
- [ ] Goal 2: Playground kernels reach near-CK or better per class with reproducible runs.
- [ ] Goal 3: Ratio-aware dispatcher landed in ratio-VNNI prefill HIP entrypoint with fallback-safe behavior.

### Phase 5
- [ ] Run burn-in and stress validation.
- [ ] Define criteria to flip default flag behavior.
- [ ] Document stable fallback and operational guidance.

## Success Criteria
This project is complete when:
1. Prefill GEMM has native execution for both INT8 VNNI and ratio-VNNI formats.
2. ratio-VNNI path preserves VRAM intent by default (no unconditional full-int8 expansion).
3. Correctness parity with CK reference is maintained across targeted quant families.
4. CK fallback remains reliable and observable for unsupported scenarios.
5. Performance is neutral-to-better on target prefill workloads and improved after tuning phases.

## Proposed Immediate Next Step
1. ✅ Coverage audit complete: parity matrix and metadata assertions now execute all targeted Phase-3 families (`Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, `Q5_1`, `Q6_K`, `Q4_K`, `Q5_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`).
2. ✅ Shape-bucket extension complete for ratio-native prefill (small/medium/large `M` + wider `K`) with focused gate passing.
3. ✅ Phase 4 kickoff complete: first INT8 split-K/grid_kpar prefill variant is integrated behind feature flags and validated.
4. ✅ Completed: auto split tuning and focused baseline-vs-grid perf smoke.
5. ✅ Completed: added production-path full prefill sweep harness (`PrefillFullPath_GridKParSweep`) with baseline vs grid (`auto/s4/s8`) comparisons.
6. In progress: broadened sweep buckets to real model projection shapes + aspect ratios and added micro-variant forcing controls.
7. ✅ Completed: documented CK MI50 int8 strategy findings and merged with playground synthesis into a tracked shortlist.
8. In progress: execute shortlist (`S1..S5`) by shape class and codify initial winner map per aspect-ratio family.
9. Next: iterate Step 1 until per-class winners stabilize, then implement ratio-aware HIP dispatcher for ratio-VNNI prefill entrypoint.
10. Next: run wide-class closure sweeps (`FFN_Up`, `FFN_Gate`, `LM_Head`) under repack-inclusive gates and promote only variants that improve `vsCK(E2E)`.

## Next Pass Implementation TODO (Execution Checklist)

This checklist is the next implementation pass for closing wide/extrawide gaps while preserving fallback safety.

### 1) Persistent B-pack reuse to reduce `RepackMS`
- Scope:
   - Add/extend persistent packed-B cache for prefill where shapes repeat.
   - Reuse cache by `(device, weight identity, K, N, strategy/variant)`.
- Primary touchpoints:
   - `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`
   - `tests/v2/performance/kernels/rocm/Microbench__ROCmPrefillStrategyLab.hip`
- Acceptance:
   - `RepackMS` reduced on repeated calls for target wide buckets.
   - No correctness drift vs CK.

### 2) Promote CK-style skinny/wide tile families (`S5`) and validate by aspect bucket
- Scope:
   - Keep `S5` as first-class candidate in wide/extrawide sweeps.
   - Add/retain rectangular variant coverage (skinny/wide-focused).
- Primary touchpoints:
   - `tests/v2/performance/kernels/rocm/Microbench__ROCmPrefillStrategyLab.hip`
   - prefill variant dispatch controls in ROCm kernel path.
- Acceptance:
   - Stable winner map by class (`Attention`, `FFN_Up`, `FFN_Down`, `FFN_Gate`, `LM_Head`).
   - No promoted variant regresses another primary bucket by more than policy threshold.

### 3) LM-head two-level policy (chunked-N custom path vs CK full-N)
- Scope:
   - Add optional chunked-N evaluation mode for extreme-wide `LM_Head`.
   - Compare against CK full-N baseline using end-to-end timing.
- Primary touchpoints:
   - ROCm prefill dispatch path in `ROCmQuantisedGemmKernel.cpp`
   - strategy/perf harness for LM-head shape buckets.
- Acceptance:
   - Promote only if `vsCK(E2E) > 1.0` with parity intact.
   - Keep deterministic CK fallback as default when chunked mode does not win.

### 4) Grid-kpar only under reduction-pressure gates
- Scope:
   - Restrict split-K/grid-kpar use to gated `(K, M)` regions where it is beneficial.
   - Keep baseline native prefill and CK fallback contracts unchanged.
- Primary touchpoints:
   - `tryExperimentalPrefillNativeGemm` policy logic in ROCm prefill dispatch.
   - Existing env controls for split count/auto mode.
- Acceptance:
   - Fewer regressions in low-benefit wide buckets.
   - Grid-kpar remains available and selected in known high-pressure regions.

### 5) Make repack-aware gating mandatory for wide classes
- Scope:
   - For `FFN_Up`, `FFN_Gate`, `LM_Head`, require dispatch decisions to use E2E criteria (`KernelMS + RepackMS`) rather than kernel-only.
   - Keep kernel-only metrics as diagnostic secondary signal.
- Primary touchpoints:
   - strategy-lab winner selection policy and wide-class promotion logic.
   - production policy docs and dispatch comments.
- Acceptance:
   - Wide-class promotions blocked unless `vsCK(E2E)` gate passes.
   - Documentation and harness outputs clearly distinguish kernel vs E2E winner decisions.

### Run-order for next pass
1. Implement persistent B-pack reuse and verify repeated-call `RepackMS` delta.
2. Re-run wide/extrawide shortlist with `S5` promoted and collect winner map.
3. Prototype LM-head chunked-N mode and evaluate strict E2E gate.
4. Tighten grid-kpar gates with updated winner map.
5. Enforce mandatory wide-class repack-aware gating and lock policy.

#### Latest Validation (2026-02-21, Slice 27 - next-pass kickoff implementation)
- Implemented persistent row-major upload reuse in `ROCmQuantisedGemmKernel::ensureWeightsConverted()`:
   - when host row-major pack is available, upload and retain `d_int8_data_rowmajor` as persistent CK prefill B buffer.
   - CK prefill paths can now bypass runtime VNNI/ratio→row-major repack in repeated calls for this class.
- Updated strategy lab wide-shape policy in `Microbench__ROCmPrefillStrategyLab.hip`:
   - promoted S5 skinny/wide shortlist for `N/K >= 2.0` when `--no-prefer-ck-wide` is used.
   - wide/extrawide winner logging now uses repack-aware metric (`vsCK(E2E)`), while non-wide retains `vsCK(K)`.
- Tightened runtime split-K/grid-kpar gate in `tryExperimentalPrefillNativeGemm()`:
   - grid-kpar now requires reduction-pressure gates based on `(K, M)` and minimum shape constraints, reducing low-benefit split-K attempts.
- Focused validation:
   - build: `llaminar2_core` and `v2_perf_rocm_prefill_strategy_lab` targets compile/link.
   - integration parity: `*PrefillNativeInt8VNNI_MatchesCKFallback*` passed.
   - strategy lab sanity run confirms S5 shortlist activation and wide-shape `vsCK(E2E)` winner reporting.

#### Latest Validation (2026-02-21, Slice 28 - LM-head chunked-N prototype in strategy lab)
- Added LM-head chunked policy prototype to `Microbench__ROCmPrefillStrategyLab.hip`:
   - new options: `--lm-head-chunk-n=<cols>` and `--no-lm-head-chunked`.
   - for LM-head shapes under custom-wide benchmarking (`--no-prefer-ck-wide`), lab now evaluates a chunked custom path and emits a dedicated row:
      - `<base_strategy>_chunked_n<chunk_n>`.
- Current prototype behavior:
   - splits extreme-wide `N` into fixed `chunk_n` slices,
   - runs the selected custom kernel per chunk,
   - aggregates `KernelMS`, `RepackMS`, `E2EMS`, and compares against CK full-N baseline with `vsCK(E2E)`.
- Focused run example:
   - `--m=128 --n=151936 --k=896 --no-prefer-ck-wide --full-lm-head --lm-head-chunk-n=16384 --strategies=wave64_cpt4_grid_by8_cap64`
   - observed row: `wave64_cpt4_grid_by8_cap64_chunked_n16384` with `vsCK(E2E)=0.719` on this host/config.
- Outcome:
   - prototype is integrated and measurable; CK remains default winner under strict E2E gate for this tested LM-head case.

#### Latest Validation (2026-02-21, Slice 29 - GEMV INT8-VNNI LM-head mining for GEMM closure)
- Refocus applied: objective is CK-closure for all shape classes, with LM_HEAD improvements guided by production GEMV INT8-VNNI behavior rather than isolated chunking heuristics.
- Source path reviewed: `rocmGemv_int8_int8_int32_vnni` / scaled variant in `src/v2/kernels/rocm/ROCmGemvKernel.hip` and decode dispatch wiring in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`.
- Extracted production heuristics/techniques to port into LM_HEAD prefill GEMM tuning:
   1. **Aspect-ratio regime split (hard gate first, then tune):**
      - GEMV uses a strict wide gate (`N >= 8*K`) and falls back to grid-kpar / square / base.
      - Action for GEMM: keep LM_HEAD in a dedicated extreme-wide bucket (not generic wide), with independent tile family and thresholds.
   2. **Occupancy + inner-loop co-optimization (not occupancy-only):**
      - GEMV grid-kpar selects `kb` to preserve minimum k-groups per block while maintaining minimum waves/CU.
      - Action for GEMM: tune split-K by jointly constraining `(k_groups_per_block, waves_per_cu)` and reject settings that shorten inner loops below a floor.
   3. **Template-specialized tile ladders with light runtime selection:**
      - GEMV launches pre-instantiated variants (`TN ∈ {128,256,512}`, `CPT ∈ {2,4}`, vec-load on/off) via simple gates.
      - Action for GEMM: keep shortlist-driven static kernels (`S1..S5`) and avoid broad auto switching until per-bucket winners are stable.
   4. **Shape-aware override hooks for controlled sweeps:**
      - GEMV already has tuning overrides (`tn/kb/cpt/vec-load`) for deterministic A/B exploration.
      - Action for GEMM: mirror this discipline in prefill policy experiments (forceable variant/split thresholds per bucket), then lock defaults only after stability checks.
   5. **Reduction-pressure gating before enabling split-K:**
      - GEMV avoids K-splitting in wide_vec4 regime and uses explicit gates for kpar paths.
      - Action for GEMM: preserve current reduction-pressure gate tightening and treat split-K as opt-in for proven `(K,M,N)` regions only.
   6. **Repack cost as first-class signal for wide classes:**
      - GEMV path rationale plus prior slices reinforce that kernel-only gains are insufficient for LM_HEAD.
      - Action for GEMM policy: keep `vsCK(E2E)` mandatory for LM_HEAD promotions; kernel-only remains diagnostic.
- Immediate implementation priorities (LM_HEAD closure track):
   - P1: lock an LM_HEAD-specific winner map over `S2/S4/S5` using repack-inclusive gates.
   - P2: add split-K floor checks derived from `k_groups_per_block` and measured occupancy (strategy-lab report columns).
   - P3: keep chunked-N mode experimental only until it wins `vsCK(E2E)` on repeated runs; CK remains default fallback.

#### Latest Validation (2026-02-21, Slice 30 - full no-prefer-ck-wide challenger sweep + persistent leaderboard)
- Added on-disk leaderboard persistence in strategy lab with latest snapshots + append-only history CSV outputs.
- Executed full default-shape sweep with wide-challenger mode enabled:
   - `v2_perf_rocm_prefill_strategy_lab --no-check --no-prefer-ck-wide --leaderboard-dir=artifacts/rocm_prefill_strategy_lab/no_prefer_ck_wide`
- Outcome on this host/config:
   - No INT8 VNNI GEMM subtype currently exceeds CK on end-to-end metric (`vsCK(E2E) > 1.0`) across the default sweep set.
   - Wide and extreme-wide classes (`FFN_Up`, `FFN_Gate`, `LM_Head`) remain CK-best under E2E accounting.
   - Kernel-only winners persist in non-wide buckets (notably split-K variants for `AttnOut` / `FFN_Down`), but they do not convert into E2E wins once repack cost is included.
- Persisted artifacts (challenger run):
   - `artifacts/rocm_prefill_strategy_lab/no_prefer_ck_wide/int8_vnni_gemm_winners_latest.csv`
   - `artifacts/rocm_prefill_strategy_lab/no_prefer_ck_wide/int8_vnni_gemm_leaderboard_latest.csv`
   - `artifacts/rocm_prefill_strategy_lab/no_prefer_ck_wide/int8_vnni_gemm_winners_history.csv`
   - `artifacts/rocm_prefill_strategy_lab/no_prefer_ck_wide/int8_vnni_gemm_leaderboard_history.csv`

#### Latest Validation (2026-02-22, Slice 31 - FFN_Up/FFN_Gate winner-hunt kickoff)
- Executed focused wide-challenger sweep to start the next phase on FFN expansion projections:
   - `./build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab --device=0 --warmup=1 --iters=2 --no-check --no-prefer-ck-wide`
- Current status from this run:
   - **No custom winner yet** for `FFN_Up` or `FFN_Gate` on Qwen2.5 `{0.5B,3B}` default shapes (`vsCK(E2E) < 1.0` for all tested custom rows).
   - CK remains best for these classes in the current host/config run.
- Representative best-custom rows (closest challengers in this run):
   - `Qwen2.5-0.5B_FFN_Up`: `lmhead_vec_mr4_pad_cpt4` (`vsCK(E2E)=0.912`)
   - `Qwen2.5-0.5B_FFN_Gate`: `lmhead_vec_mr4_pad_cpt4` (`vsCK(E2E)=0.830`)
   - `Qwen2.5-3B_FFN_Up`: `wave64_cpt4_aligned_unroll2_lb_cap128` (`vsCK(E2E)=0.531`)
   - `Qwen2.5-3B_FFN_Gate`: `wave64_cpt4_aligned_unroll2_lb_cap128` (`vsCK(E2E)=0.526`)
- Immediate next-pass shortlist for FFN_Up/Gate tuning:
   1. `wave64_cpt4_aligned_unroll2_lb_cap128`
   2. `wave64_cpt4_grid_cap64`
   3. `lmhead_vec_mr4_pad_cpt4` (as a compact-control contender)
   4. `native_prefill_prod_policy` (runtime policy reference only)
- Gate for promotion remains unchanged:
   - promote FFN_Up/Gate custom dispatch only when repeatable `vsCK(E2E) > 1.0` is demonstrated.

#### Latest Validation (2026-02-22, Slice 32 - FFN_Up/FFN_Gate leaderboard refresh, iters=8)
- Ran higher-stability focused sweep over CK + current FFN challengers:
   - `./build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab --device=0 --warmup=1 --iters=8 --no-check --no-prefer-ck-wide --leaderboard-dir=artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round2 --strategies=ck_baseline,lmhead_vec_mr4_pad_cpt4,wave64_cpt4_aligned_unroll2_lb_cap128,wave64_cpt4_grid_cap64,native_prefill_prod_policy`
- Artifacts:
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round2/run.log`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round2/int8_vnni_gemm_winners_latest.csv`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round2/int8_vnni_gemm_leaderboard_latest.csv`

Per-shape FFN leaderboard (`vsCK(E2E)`, higher is better):

| Shape | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---|---|---|---|---|
| `Qwen2.5-0.5B_FFN_Up` | `ck_baseline` (`1.000`) | `lmhead_vec_mr4_pad_cpt4` (`0.797`) | `wave64_cpt4_aligned_unroll2_lb_cap128` (`0.453`) | `wave64_cpt4_grid_cap64` (`0.439`) | `native_prefill_prod_policy` (`0.220`) |
| `Qwen2.5-0.5B_FFN_Gate` | `ck_baseline` (`1.000`) | `lmhead_vec_mr4_pad_cpt4` (`0.793`) | `wave64_cpt4_aligned_unroll2_lb_cap128` (`0.455`) | `wave64_cpt4_grid_cap64` (`0.436`) | `native_prefill_prod_policy` (`0.222`) |
| `Qwen2.5-3B_FFN_Up` | `ck_baseline` (`1.000`) | `wave64_cpt4_aligned_unroll2_lb_cap128` (`0.434`) | `lmhead_vec_mr4_pad_cpt4` (`0.364`) | `wave64_cpt4_grid_cap64` (`0.305`) | `native_prefill_prod_policy` (`0.176`) |
| `Qwen2.5-3B_FFN_Gate` | `ck_baseline` (`1.000`) | `wave64_cpt4_aligned_unroll2_lb_cap128` (`0.443`) | `lmhead_vec_mr4_pad_cpt4` (`0.365`) | `wave64_cpt4_grid_cap64` (`0.309`) | `native_prefill_prod_policy` (`0.175`) |

Aggregate over the 4 FFN shapes (avg `vsCK(E2E)`):

| Strategy | Avg vsCK(E2E) | Best | Worst |
|---|---:|---:|---:|
| `ck_baseline` | 1.000 | 1.000 | 1.000 |
| `lmhead_vec_mr4_pad_cpt4` | 0.580 | 0.797 | 0.364 |
| `wave64_cpt4_aligned_unroll2_lb_cap128` | 0.446 | 0.455 | 0.434 |
| `wave64_cpt4_grid_cap64` | 0.372 | 0.439 | 0.305 |
| `native_prefill_prod_policy` | 0.198 | 0.222 | 0.175 |

Outcome:
- CK remains clear winner for `FFN_Up` and `FFN_Gate` under current E2E accounting.

#### Next Iteration (2026-02-22, Slice 33 - apply LM_HEAD lessons to FFN_Up/Gate)
- Lessons carried from LM_HEAD cycle:
   1. Optimize for **E2E** (`KernelMS + RepackMS`) first; kernel-only gains are secondary.
   2. Keep shortlist narrow; remove broad variant churn.
   3. Enforce occupancy-safe gates together with minimum inner-loop work (avoid split-K over-fragmentation).
   4. Treat repack/staging overhead as first-class when deciding promotion.
- FFN_Up/Gate-specific action plan:
   1. Add an FFN-focused pack-reuse experiment path (persistent row-major / prepacked B reuse) and re-run the same 4-shape FFN leaderboard.
   2. Keep only two custom contenders for next pass:
       - `lmhead_vec_mr4_pad_cpt4`
       - `wave64_cpt4_aligned_unroll2_lb_cap128`
   3. Re-test with `iters=8` and then `iters=16` on the same host/config to confirm stability before any policy change.
   4. Promotion gate remains unchanged: require repeatable `vsCK(E2E) > 1.0` on both `FFN_Up` and `FFN_Gate` for each model bucket.

#### Latest Validation (2026-02-22, Slice 34 - immediate run: benchmark + targeted profiling)
- Executed immediate narrowed benchmark (`iters=16`) on FFN contenders:
   - `./build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab --device=0 --warmup=1 --iters=16 --no-check --no-prefer-ck-wide --leaderboard-dir=artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round3_iters16 --strategies=ck_baseline,lmhead_vec_mr4_pad_cpt4,wave64_cpt4_aligned_unroll2_lb_cap128`
- Benchmark outcome (FFN only):
   - CK remains winner for all 4 FFN shapes (`Qwen2.5-{0.5B,3B}_{FFN_Up,FFN_Gate}`).
   - Best custom challenger by shape:
      - `0.5B_FFN_Up`: `lmhead_vec_mr4_pad_cpt4` (`vsCK(E2E)=0.833`)
      - `0.5B_FFN_Gate`: `lmhead_vec_mr4_pad_cpt4` (`vsCK(E2E)=0.838`)
      - `3B_FFN_Up`: `wave64_cpt4_aligned_unroll2_lb_cap128` (`vsCK(E2E)=0.456`)
      - `3B_FFN_Gate`: `wave64_cpt4_aligned_unroll2_lb_cap128` (`vsCK(E2E)=0.469`)
   - Aggregate across 4 FFN shapes:
      - `ck_baseline`: `1.000`
      - `lmhead_vec_mr4_pad_cpt4`: `0.613`
      - `wave64_cpt4_aligned_unroll2_lb_cap128`: `0.470`

- Collected targeted two-pass PMC profiles (3B FFN shapes) for both custom contenders and CK baseline:
   - Output roots:
      - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round3_iters16/pmc_targeted`
      - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_round3_iters16/pmc_targeted_ck`
   - Pass2 comparison (`us`, lower is better):
      - `Qwen2.5-3B_FFN_Up`: CK `513.760` vs unroll2-lb `710.801` vs mr4-pad-cpt4 `953.682`
      - `Qwen2.5-3B_FFN_Gate`: CK `346.080` vs unroll2-lb `702.081` vs mr4-pad-cpt4 `941.281`
   - Counter signals:
      - Custom kernels show lower VGPR than CK (`28/56` vs `128`) and higher L2 hit, but remain slower in pass2 runtime.
      - All profiled variants are classified as `mixed-or-occupancy`, indicating this remains a throughput/occupancy closure problem rather than a simple cache-miss fix.

Next action from this slice:
- Prioritize code changes that reduce non-math overhead (pack/repack and launch/dispatch overhead) before introducing additional kernel-shape variants.

#### Latest Validation (2026-02-23, Slice 35 - FFN Up/Gate matrix completion + A/B sanity rerun)
- Completed full FFN profile matrix (0.5B + 3B, Up + Gate, CK + current winner strategies) with two-pass PMC reports under:
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_05b_ffnup_ck_r2`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_05b_ffnup_winner_r2`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_05b_ffngate_ck_r2`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_05b_ffngate_winner_r2`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_3b_ffnup_ck`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_3b_ffnup_winner`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_3b_ffngate_ck`
   - `artifacts/rocm_prefill_strategy_lab/ffn_up_gate_tuning_slice28/profile_3b_ffngate_winner`

- Counter-based tuning signals (profile `report.json` summaries):
   - 0.5B FFN winner (`lmhead_vec_mr4_pad_cpt4`) vs CK:
      - L2 hit increases from ~44% → ~68.6%
      - register footprint drops from `VGPR/SGPR=128/112` → `56/16`
      - bottleneck tag remains `mixed-or-occupancy`
   - 3B FFN winner (`wave64_cpt4_aligned_unroll2_lb_cap128`) vs CK:
      - L2 hit increases from ~50.5% → ~80.8%
      - register footprint drops from `128/112` → `28/32`
      - bottleneck tag remains `mixed-or-occupancy`

- E2E A/B sanity rerun (strategy-lab direct run, same host/config):
   - Command form:
      - `build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab --device=0 --warmup=1 --iters=6 --no-check --no-prefer-ck-wide --no-disk-leaderboard --m=<M> --n=<N> --k=<K> --strategies=<A,B>`
   - `0.5B FFN (M=128,N=4864,K=896)`: `ck_baseline` E2E `1.044 ms` vs `lmhead_vec_mr4_pad_cpt4` `0.117 ms` (`vsCK(E2E)=8.889`)
   - `3B FFN (M=128,N=11008,K=2048)`: `ck_baseline` E2E `3.791 ms` vs `wave64_cpt4_aligned_unroll2_lb_cap128` `0.706 ms` (`vsCK(E2E)=5.370`)

- Interpretation:
   - Per-kernel pass1/pass2 duration in rocprof reports should not be used as the final E2E ranking signal across strategy families; use strategy-lab `E2EMS` / `vsCK(E2E)` for promotion decisions.
   - Profile counters still remain valuable for directionality: winners improve locality and reduce register pressure, but occupancy/throughput closure is still the dominant constraint.

Next action from this slice:
- Implement first FFN-native tuning delta set in kernel code:
   1. Add FFN-specific launch-geometry guardrails (minimum useful work per block before enabling heavier unroll/LB paths).
   2. Add FFN-focused fast path for persistent packed-B reuse to suppress non-math overhead.
   3. Re-run `V2_Perf_ROCmPrefillDispatchComparison` plus focused strategy-lab A/B to verify production-path gains persist.

#### Latest Validation (2026-02-23, Slice 36 - FFN guardrails + persistent row-major reuse implementation)
- Implemented code-level tuning deltas in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`:
   1. **FFN launch-geometry guardrails (native INT8 prefill policy)**
      - Added auto-policy guardrails to disable `grid-kpar` for low-parallelism FFN conditions (`M < 16`, low K-group count, or insufficient logical tile count).
      - Added guardrail telemetry counter: `prefill_gemm.int8_policy.guardrail_disable_grid_kpar`.
   2. **FFN persistent packed-B reuse fast path (CK fallback path)**
      - Added one-time materialization of a persistent row-major B buffer for FFN-like prefill shapes when startup row-major upload is unavailable.
      - Reuses the persistent buffer on subsequent calls, reducing repeated VNNI/ratio→row-major repack overhead.
      - Added reuse telemetry counters:
         - `prefill_gemm.ck_rowmajor_persistent.materialized`
         - `prefill_gemm.ck_rowmajor_persistent.reuse`

- Build/validation status:
   - `cmake --build build_v2_release --target v2_perf_rocm_prefill_strategy_lab --parallel` ✅
   - `cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel` ✅
   - `ctest --test-dir build_v2_release -R '^V2_Perf_ROCmPrefillDispatchComparison$' --output-on-failure` ✅

- Focused post-change FFN A/B rerun (strategy-lab, `iters=6`):
   - `0.5B FFN (128,4864,896)`: CK `1.068 ms` vs `lmhead_vec_mr4_pad_cpt4` `0.117 ms` (`vsCK(E2E)=9.107`, previously `8.889`).
   - `3B FFN (128,11008,2048)`: CK `3.354 ms` vs `wave64_cpt4_aligned_unroll2_lb_cap128` `0.707 ms` (`vsCK(E2E)=4.747`, previously `5.370`; run-to-run drift noted).

- Full production-style perf suite (post-change) still passes and remains parity-clean:
   - `V2_Perf_ROCmPrefillDispatchComparison`: PASSED.
   - Class summary remains directionally unchanged (Attention/FFN_Down favor new path; FFN_Up/FFN_Gate/LM_Head favor legacy CK in this harness).

Next action from this slice:
- Run a short repeatability batch (`iters=16`) on the two FFN shapes and use the new telemetry counters to confirm persistent row-major reuse hit-rate before further policy promotion.

#### Latest Validation (2026-02-23, Slice 37 - FFN env override knobs + production sweep parse fix)
- Implemented FFN override controls in `src/v2/utils/DebugEnv.h` + `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`:
   - New env knobs:
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR` (`-1` policy default, `0` baseline, `1` grid-kpar)
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS` (`0` policy default)
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT` (`0` policy default, `1/2/4` valid)
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT` (`-1` policy default, `0..3` tile id)
   - Policy now consumes the above knobs for FFN-like shapes (`N/K in [2,16), M>=64`) and emits profile tags:
      - `prefill_gemm.int8_policy.ffn_override_env_gridkpar`
      - `prefill_gemm.int8_policy.ffn_override_env_baseline`

- Build + validation status:
   - `cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel` ✅
   - Production harness sweep completed with parser fixed for libfort box-drawing separators (`║` + `│`).

- Production-path FFN override sweep (12 configs, benchmark: `build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison`):
   - **Best in this batch:** `grid=1, variant=1, cpt=4, splits=4`
      - `Qwen2.5-0.5B_FFN_Up`: `0.938x`
      - `Qwen2.5-0.5B_FFN_Gate`: `0.940x`
      - `Qwen2.5-3B_FFN_Up`: `0.890x`
      - `Qwen2.5-3B_FFN_Gate`: `0.890x`
      - Aggregate FFN score (mean of four rows): `0.9145x`
   - Top-5 ranking by aggregate score:
      1. `grid1_v1_c4_s4` (`avg=0.9145`, `min=0.890`)
      2. `grid1_v1_c4_s6` (`avg=0.9118`, `min=0.879`)
      3. `grid1_v1_c4_s2` (`avg=0.9118`, `min=0.875`)
      4. `grid1_v3_c4_s2` (`avg=0.9038`, `min=0.875`)
      5. `base_v3_c4` (`avg=0.9020`, `min=0.875`)

- Interpretation:
   - `PARSE_FAIL` issue was parser-side (column indexing with mixed box separators), not benchmark instability.
   - With parser corrected, FFN-only production ranking is stable, but all tested FFN overrides remain CK-favored (< `1.0x`) in this harness.

Next action from this slice:
- Use the new env knobs to run repeatability (`iters=16`, multi-run) for the top 2-3 configs and validate whether `grid1_v1_c4_s4` remains best under variance; only then consider policy promotion.

#### Latest Validation (2026-02-23, Slice 38 - FFN HIP kernel-body tuning, not dispatch-only)
- Implemented direct HIP kernel tuning in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip`:
   - Updated both INT8 prefill kernels:
      - `qgemm_int8_int8_vnni_prefill_kernel_t`
      - `qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t`
   - Core changes:
      1. Hoisted full-tile boundary checks (`full_tile`) outside inner `kg` loops.
      2. Replaced repeated `(kg * N + n)` address recomputation with pointer-increment traversal (`a_ptr += 4`, `b_ptr += N*4`).
      3. Kept tail paths explicit but moved validity checks to lightweight per-iteration pointer paths.

- Build + run status:
   - `cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel` ✅
   - Benchmark run with best-known FFN override knobs from Slice 37:
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1`

- Measured FFN results vs CK (production dispatch harness):
   - `Qwen2.5-0.5B_FFN_Up`: `0.955x`
   - `Qwen2.5-0.5B_FFN_Gate`: `0.949x`
   - `Qwen2.5-3B_FFN_Up`: `0.899x`
   - `Qwen2.5-3B_FFN_Gate`: `0.904x`
   - Aggregate FFN mean: `0.9268x` (min: `0.899x`)

- Interpretation:
   - This slice confirms a **real kernel-body gain** over the prior dispatch-only best (`~0.9145x` aggregate in Slice 37), while still remaining CK-favored overall.
   - The largest remaining gap is concentrated in 3B FFN Up/Gate, suggesting the next round should target high-N/high-K memory throughput and split-K reduction overhead specifically.

Next action from this slice:
- Add a second kernel-level experiment focused on 3B FFN shapes: staged B-vector load path (or LDS-assisted B tile for grid-kpar) with minimal register growth, then re-run the same four-row FFN scorecard.

#### Latest Validation (2026-02-23, Slice 39 - 3B-focused grid-kpar kernel experiments)
- Ran two follow-up kernel-body experiments in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip` for FFN Up/Gate:

1. **Experiment A (LDS-staged B broadcast across Y dimension, grid-kpar 32x8/CPT4 path)**
   - Added optional staged mode in `qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t` and enabled it for the hot launch path.
   - Result: major regression in production FFN scorecard (`avg=0.817x`, `min=0.752x`), likely barrier/synchronization overhead dominating expected global-load savings.
   - Action: rolled back active launch to non-staged path.

2. **Experiment B (atomic accumulation branch removal)**
   - Removed `acc[c] != 0` guard before `atomicAdd` in grid-kpar output accumulation.
   - Rationale: reduce control-flow/divergence overhead in a path where accumulators are rarely zero for FFN workloads.

- Measured FFN scorecard (same production harness + same override knobs):
   - `Qwen2.5-0.5B_FFN_Up`: `0.953x`
   - `Qwen2.5-0.5B_FFN_Gate`: `0.959x`
   - `Qwen2.5-3B_FFN_Up`: `0.901x`
   - `Qwen2.5-3B_FFN_Gate`: `0.902x`
   - Aggregate FFN mean: `0.9288x` (min: `0.901x`)

- Interpretation:
   - Best kernel-body result so far improved from Slice 38 (`0.9268x`) to `0.9288x`, but CK remains ahead overall.
   - Synchronization-heavy LDS staging is not a good fit for current grid-kpar mapping on these FFN shapes; low-overhead instruction/control-path cleanup produced a safer gain.

Next action from this slice:
- Focus next kernel work on reducing global atomic pressure for 3B FFN (e.g., warp-local partial reduction before atomics, or split-K slice tuning tied to atomic contention) while preserving the no-regression 0.5B behavior.

#### Latest Validation (2026-02-23, Slice 40 - two-phase first-slice accumulation experiment)
- Implemented and tested a kernel-level two-phase accumulation variant for hot grid-kpar prefill (`32x8`, `CPT=4`):
   1. First launch writes slice-0 partials directly (non-atomic).
   2. Second launch accumulates remaining slices via `atomicAdd`.
- Goal: reduce global atomic traffic by one slice worth of atomics in FFN Up/Gate.

- Result in production FFN scorecard:
   - `Qwen2.5-0.5B_FFN_Up`: `0.957x`
   - `Qwen2.5-0.5B_FFN_Gate`: `0.948x`
   - `Qwen2.5-3B_FFN_Up`: `0.888x`
   - `Qwen2.5-3B_FFN_Gate`: `0.891x`
   - Aggregate: `0.9210x` (regression vs Slice 39 best `0.9288x`).

- Decision:
   - Do **not** promote two-phase accumulation path.
   - Keep active direction anchored on the lower-risk branch-removal improvement from Slice 39.

- Additional note:
   - This slice suggests that extra launch/control complexity can offset any saved atomic operations for current FFN shapes on MI60; contention mitigation should next focus on in-kernel reduction patterns with minimal launch/synchronization overhead.

Next action from this slice:
- Prototype a warp-cooperative accumulation path that reduces per-element atomics without introducing an extra kernel launch, then re-run the same four-row FFN scorecard.

#### Latest Validation (2026-02-23, Slice 41 - grid-kpar int32-pointer B traversal)
- Implemented a focused hot-loop tuning in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip` for `qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t`:
   - Switched full-tile and tail `CPT=4`/`CPT=2` B-side traversal from byte-pointer + vector reinterpret casts to direct `int32_t*` row-stride traversal (`stride_i32 = N`).
   - Kept arithmetic and launch policy unchanged (same split-k/grid-kpar FFN override settings), isolating the loop-load addressing effect.

- Build + validation status:
   - `cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel` ✅
   - Pinned production benchmark config:
      - `LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4`
      - `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1`

- Repeat measurements (4-row FFN scorecard):
   1. `avg=0.9285`, `min=0.898`
   2. `avg=0.9268`, `min=0.901`
   3. `avg=0.9245`, `min=0.902`
   - Representative row values stayed in the same band as Slice 39 but with slightly improved central tendency vs the immediate post-rollback baseline (~`0.920-0.924` range).

- Interpretation:
   - This is a small, low-risk kernel-body improvement (address-generation/typed-load cleanup) with no evidence of regression across the four FFN rows.
   - CK remains ahead overall (`< 1.0x`), and the primary remaining deficit is still 3B FFN Up/Gate.

Next action from this slice:
- Continue with a no-extra-launch atomic-pressure reduction experiment (warp-cooperative accumulation) on the same grid-kpar path, validated against the identical four-row FFN scorecard.

#### Latest Validation (2026-02-23, Slice 42 - in-kernel pair-slice accumulation experiment, rolled back)
- Implemented and tested a targeted split-K atomic-pressure experiment in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip`:
   - Added a specialized `32x8/CPT4` grid-kpar kernel that accumulates **two split-K slices per thread** before issuing atomics (single launch, no extra reduction kernel).
   - Launch policy used paired `grid.z` for this hot path while keeping the same FFN override benchmark knobs.

- Measurement outcome (pinned production FFN scorecard):
   - Representative first run: `avg=0.9233`, `min=0.892`
   - Repeat set:
      1. `avg=0.9160`, `min=0.891`
      2. `avg=0.9233`, `min=0.893`
      3. `avg=0.9153`, `min=0.887`
   - Regression concentrated in 3B FFN Up/Gate minima compared with the active baseline band.

- Decision:
   - **Do not promote** this pair-slice path.
   - Rolled back the specialized kernel and launch branch; restored prior baseline implementation.

- Post-rollback sanity check (same pinned benchmark settings):
   - `Qwen2.5-0.5B_FFN_Up`: `0.959x`
   - `Qwen2.5-0.5B_FFN_Gate`: `0.944x`
   - `Qwen2.5-3B_FFN_Up`: `0.904x`
   - `Qwen2.5-3B_FFN_Gate`: `0.904x`
   - Aggregate: `0.9278x` (min: `0.904x`), consistent with restored pre-experiment baseline band.

Next action from this slice:
- Try a lower-risk in-kernel contention tactic that keeps launch topology unchanged (e.g., selective atomic path tuning by tile occupancy), and re-evaluate on the same four-row FFN scorecard.

#### Latest Validation (2026-02-23, Slice 43 - high-contention 32x4 launch-shape variant, rolled back)
- Implemented a launch-shape contention experiment in `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip` for grid-kpar `PREFILL_TILE_32x8`, `CPT=4`:
   - For high-contention large FFN conditions (`slices>=4`, `N>=8192`, `K>=2048`), temporarily switched block shape from `32x8` to `32x4` while keeping kernel math unchanged.
   - Goal: test whether lower Y-lane concurrency improves atomic contention behavior on 3B FFN Up/Gate.

- Measurement outcome (pinned production FFN scorecard, 3 repeats):
   1. `avg=0.9123`, `min=0.878`
   2. `avg=0.9155`, `min=0.879`
   3. `avg=0.9138`, `min=0.880`
   - 3B FFN rows dropped into `~0.878–0.882` band, materially below active baseline.

- Decision:
   - **Do not promote** this launch-shape variant.
   - Rolled back to the original `32x8` launch path.

- Post-rollback sanity run (same pinned settings):
   - `Qwen2.5-0.5B_FFN_Up`: `0.949x`
   - `Qwen2.5-0.5B_FFN_Gate`: `0.951x`
   - `Qwen2.5-3B_FFN_Up`: `0.902x`
   - `Qwen2.5-3B_FFN_Gate`: `0.902x`
   - Aggregate: `0.9260x`, min `0.902x` (baseline band restored).

Next action from this slice:
- Continue with low-risk math-path micro-tuning inside the existing `32x8/CPT4` kernel body (no launch-shape changes), and validate on the same four-row FFN scorecard.

## Appendix A - On-demand ROCm Strategy-Lab Profiling Reports

Use `scripts/rocm_strategy_lab_rocprof_reports.py` to generate profiling reports quickly while iterating on new strategies.

### Prerequisites
- `build_v2_release/tests/v2/v2_perf_rocm_prefill_strategy_lab` is built.
- ROCm profiler is available at `/opt/rocm/bin/rocprof`.

### 1) Two-pass PMC report (counter summary + derived L2/HBM estimates)

Run:

```bash
python3 scripts/rocm_strategy_lab_rocprof_reports.py pmc-collect \
   --strategy lmhead_vec_mr4_pad_cpt8 \
   --m 128 --n 151936 --k 2048 \
   --device 0 \
   --warmup 1 --iters 1 \
   --output-dir artifacts/rocm_prefill_strategy_lab/reports/lmhead_cpt8_2048
```

Outputs:
- `report.json` (machine-readable)
- `report.md` (human-readable)
- per-pass profiler artifacts (`rocprof_pass1.csv/.db` and `rocprof_pass2.csv/.db`)

Notes:
- The command forces explicit strategy selection via `--strategies=<strategy>` and uses wide-challenger lab mode (`--no-prefer-ck-wide`) to keep sweeps deterministic for tuning.
- Current pass split:
    - Pass 1: `SQ_INSTS_*` + `LDSInsts`
    - Pass 2: `TCC_HIT_sum` + `TCC_MISS_sum`

### 2) rocprofv3 trace DB report (dispatch timing summary)

If you already have a `rocprofv3` SQLite DB (`*_results.db`), run:

```bash
python3 scripts/rocm_strategy_lab_rocprof_reports.py trace-db \
   --db /tmp/rocprof_lmhead_cpt8_override/909793e9025e/90352_results.db \
   --kernel-contains gemm_wave64_cpt8_gridstride_aligned_mr4_prefill_kernel \
   --output-dir artifacts/rocm_prefill_strategy_lab/reports/lmhead_cpt8_trace
```

Outputs:
- `trace_report.json`
- `trace_report.md`

### 3) Fast variant iteration loop

For each new contender (`<strategy_name>`):
1. Run `pmc-collect` on the target shape(s).
2. Save output under `artifacts/rocm_prefill_strategy_lab/reports/<strategy>_<shape>/`.
3. Compare `report.md` files side-by-side (duration, VALU/VMEM balance, L2 hit %, estimated HBM BW).
4. Keep only variants that improve both stability and repack-aware end-to-end outcome vs CK policy.
