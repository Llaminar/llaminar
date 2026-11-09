# Handover: ideas 7 + 25 — `codex/hotpath-cleanup-prefill` @ 39c24d8b4

Base: `feat/arm64-native-kernels` @ 5d4e3716a (includes R1; linear, 13 commits; no rename tokens in this diff). Commit SHAs are on 5d4e3716a; every patch is identical (range-diff `=`) to the gated pre-rebase commits. The branch
is pushed, with no PR yet.

## Commits

| SHA | Item | Summary |
|---|---|---|
| d2fcfe76b | cleanup | Retire the orphan, never-registered `Perf__RMSNormParallelSweep.cpp`. Its setenv sweeps could not take effect: `debugEnv()` is read once, and no kernel reads those thresholds. |
| a7108c5fa | 7d fix | Q8_1 pure-integer RMSNorm Q11 gamma cache is keyed by the owner control block (`weak_from_this`) plus address and width, not the raw address. Unowned or expired owners always requantize. The registered RMSNorm thread sweep quantizes gamma once, outside the timed loop. |
| 23dbc556c | 7b perf | `get_inv_freq_cached` hits are lock-free (8-entry per-thread pointer memo into a node-stable, never-erased table). |
| 74ce39218 | 25b build | `install-cuda.sh` chooses the NVIDIA repo per image (22.04/24.04 × amd64→x86_64, arm64→sbsa) and fails closed otherwise. New `MODE=profilers`. New `validate-cuda-profilers.sh`. |
| 9be1cfe8b | 7c perf | Decoded-row RMSNorm (BF16/FP16/Q8_1) keeps its scratch on the stack up to 8192 floats and fails fast beyond that. There is no heap path. `apply_with_residual_add` is untouched (Main is retiring it). |
| 298148fe5 | 7a/7a2 perf | RoPE angle planes come from the admitted `cpu_rope_angles` workspace (RoPEStage IWorkspaceConsumer). RoPEPersistentState, update_rope_cache, the decode memo, VerifierAngleRows, the Q8_1 TLS tables and the kernel TLS state are deleted. One row function serves every path. A single shared omp-for fill runs inside teams. The estimator prices the workspace. |
| 7b8983521 | 25a perf | Host-only `forward_graph` timers: `prefill_chunk_host_wall`, `prefill_inter_chunk_gap`, `prefill_family_host_wall`. No events and no sync. |
| 3e55bc6f9 | review fix | The RoPE stage kernel is created exactly once; a type mismatch hits LLAMINAR_UNREACHABLE. Planes are sized by the effective rotated width (profile `rope_dimension_count`, wire v7). Focused regressions for the two Q8_1 defects. Profile test for the generic `<arch>.rope.dimension_count` path. |
| 7b59ecf77 | reviewer P2 fix | Ordinary contiguous RoPE fills publish every row, including negative positions (EveryPosition); verifier padding stays VerifierPadding. RAII PerfStats restore in the host-timer test. |
| dca2a3021 | test claims | Raw dirty-vs-clean byte check, asserted actual team width, and exact comments (per-row comparison is modulo signed zero and uses the current one-row path). |
| 0b32f8edd | stale tests | Bind admitted angle workspaces in CPU RoPE stage tests (including a skip-masked case and the GPU suite's CPU reference). Price CPU angle planes in the estimator/planner expectations. Host-independent CUDA provisioning replay with repository-selection cases. |
| 912d01073 | test contract | CPU RoPE stage tests assert execute() directly; catch-all→skip and success-gated checks removed (exposed and bound a never-bound position_ids case). |
| 39c24d8b4 | test lifecycle | The angle-workspace lifecycle replay republishes each request (updateDynamicPositionIds) after resetSessionState, as production does, with new positions per replay. It previously executed reset stages with no published request. |

## Production defects fixed, each with a focused ProductionTestPreflight regression

- Q8_1 RMSNorm gamma: a stale quantized gamma was returned for new weights at
  a reused address. Regression: `V2_Integration_CPURMSNormGammaQ11OwnerIdentity`.
  The old key fails 3/3.
- Q8_1 RoPE, one row inside a retained team: every member rotated the shared
  banks. Regression: `V2_Integration_CPURoPEQ8OneRowTeamHeadOwnership`.
  Reproduced on the base.
- Q8_1 RoPE, negative one-row position: it read the previous call's TLS table.
  Regression: `V2_Integration_CPURoPEQ8OneRowNegativePosition`. Reproduced on
  the base.

- Ordinary RoPE at negative positions (introduced by 298148fe5, fixed in 7b59ecf77): one-row,
  partial and 32+ table rows rotated by stale borrowed planes. The exported primitive contract was
  affected; the stage path drops negative positions. Regression:
  `V2_Integration_CPURoPENegativePositionDirtyWorkspace`. Raw output bytes are independent of
  workspace content. Per-row equality is checked modulo signed zero. The team width is asserted.
  Against the 407347f6c primitives the per-row check fails 160/160 cells. The cross-TU harness
  (rows 1..64 contiguous) is byte-identical to the 425002104 TU at n_past -1/-3/-40/-100 via the
  base no-state API; the positive control used start argument 17231 (start 17231 + rows). The base
  persistent-state route aborts only at -1 (sentinel collision), and
  only the Q8_1 one-row negative cell, the fixed TLS defect, is excluded.
  Evidence: `/tmp/armqual/rope-negative-position-evidence.md`.

## Other new preflight regressions

- `V2_Integration_CPURoPEInverseFrequencyLookup`: 7b memo eviction across
  threads, and alternation on one thread and in teams.
- `V2_Integration_CPURoPESharedAngleFill`: a late team member over every
  format, including verifier and Q8. A `nowait` fill fails it.
- `V2_Integration_CPURoPEAngleBindingRejection`.
- `V2_Integration_CPURoPEAngleWorkspaceLifecycle`: decode, verifier and prefill stages share
  one buffer address across reset and replay. Each replay republishes a new request
  (`updateDynamicPositionIds`, as production does after a reset) and must match byte-for-byte.
  Also covers a partial stage, and unbound/undersized bindings failing fatally before any store.
  The oracle is a separately bound serial invocation of the SAME production kernel
  (`CPURoPEKernelT<FP32>`, one row per call), not independent arithmetic. The primitives' x86
  exactness (both trees) and the base equivalence are separate evidence (arm-qual diagnostics,
  cross-TU harness).
- `V2_Integration_CPURoPEAngleWorkspaceAdmission`: exact estimator bytes for
  full and partial widths.
- `V2_{Unit,Integration}_CPURoPEAngleStorageSourcePolicy` plus negative
  controls.
- `V2_Integration_CPURMSNormDecodedRowStackBound`.
- `V2_Integration_PrefillBucketHostTimers`.
- The repository mapping test in the image-contract suite
  (`V2_Integration_CTestBuildToolImageContracts`).

## Production reachability (gate B): the Q16/hybrid RoPE family is never reached in production

Anchors at 912d01073. Production builds RoPE stages in exactly three places, and none sets `Q_out`/`K_out`:
- `QwenGraphBase::addRoPE` (QwenGraphBase.cpp:4873; callers QwenStandardGraph.cpp:172 and
  Qwen35Graph.cpp:2170).
- The key-cache publication transform (QwenGraphBase.cpp:4830, `createKeyOnlyRoPE`). Any output
  is rejected at ComputeStageFactory.cpp:111 (condition) / :116 (throw).
- `GraphResolver` StageType::RoPE (GraphResolver.cpp:719–728), from activation inputs only.

Activations other than FP32 are rejected at ActivationPrecisionPolicy.h:33–36. So the stage
never sees a Q8_1/Q16_1 operand or an output tensor, which are the only selectors of these
paths (RoPEStage.cpp:160–174).

The source policy allowlists each function below by name under tracker 07 follow-up A. Their
numerics are untouched (Q15 authority and per-call tables).

| Function | Only caller chain in `src/` | Production status |
|---|---|---|
| `apply_rope_q8_1_to_fp32` | `CPURoPEKernelT<Q8_1>::apply_q8_1_to_fp32` (CPURoPEKernelT.cpp:986) ← RoPEStage hybrid mode, Q8_1 Q + FP32 `Q_out` (RoPEStage.cpp:280) | unreachable: no `Q_out`; Q8_1 activations rejected at ActivationPrecisionPolicy.h:33 |
| `apply_rope_q8_1_to_q16_fixed_scale`, `_dispatch`, `apply_rope_q8_1_to_q16_head_fixed_scale` | `apply_q8_1_to_q16_fixed_scale` (CPURoPEKernelT.cpp:1206) ← HybridQ16 (RoPEStage.cpp:398); `apply_mixed_q8_k16_to_q16` (CPURoPEKernelT.cpp:1286) ← RoPEStage.cpp:343 | unreachable: needs Q16_1 `Q_out` |
| `apply_rope_q16_to_q16_dynamic_scale`, `_dispatch` | `apply_mixed_q8_k16_to_q16` (CPURoPEKernelT.cpp:1302) ← RoPEStage.cpp:343 | unreachable: needs Q16_1 `Q_out` and K |
| `apply_rope_q16_integer<Block>` | `CPURoPEKernelT<Q16_1>::apply_typed` / `apply_typed_block` (CPURoPEKernelT.cpp:1355/1401), selected for a Q16_1 Q | unreachable: Q16_1 activations rejected at ActivationPrecisionPolicy.h:33 |
| `apply_rope_q8_1_to_q16_1` | `CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16_1` (CPURoPEKernelT.cpp:1056); that method has no `src/` caller (TensorKernels.h:3107 virtual only) | test-only |
| `apply_rope_q8_1_to_q16`, `_dispatch`, `apply_rope_q8_1_to_q16_head[_avx2]`, `apply_rope_q8_1_to_q16_head_scalar` | `CPURoPEKernelT<Q8_1>::apply_q8_1_to_q16` (CPURoPEKernelT.cpp:1130); no `src/` caller (TensorKernels.h:3160 virtual only) | test-only |
| `apply_rope_q16_1_integer` | none | test-only |
| `apply_rope_q16_to_q16_fixed_scale`, `_dispatch` | none: the dispatch has no `src/` caller | test-only |
| `compute_rope_sincos_q15` | `apply_rope_q16_to_q16_fixed_scale` only (RoPEPrimitives.cpp:6502) | test-only |

## Evidence

- Byte identity: a base/new harness covered every changed RoPE entry point
  (dims 64–256, rows 1–64, padding and jumped positions, partial RoPE, every
  verifier format, Q8_1). The 8 MB of output is identical on NEON and
  NEON_DOTPROD.
- Mac portability on dca2a3021 (912d01073 adds only full-engine test files the portability project does not build): NEON_DOTPROD 289/289 and NEON 289/289, 0 skips; preflight repeat-until-fail:20 is 226/226. Earlier, on 8ba25fc28: NEON_DOTPROD 270/270 and NEON 270/270, with 0
  skips. Preflight repeated until failure, up to 20 runs: 214/214.
- **WSL x86 (AVX512, GCC 14) final gate at 39c24d8b4: GREEN.**
  - Unit: 960/974. The 14 failures are exactly the base set.
  - ProductionTestPreflight: 2357/2373. The 16 failures are exactly the base/WSL set.
  - Every per-case line is OK, with 0 skips, including:
    - the four previously stale groups (ComputeStage, WorkspaceMemoryEstimator, MemoryPlanner,
      CUDABuilderProvisioning);
    - QFNSelectedAttentionWorkspaceAdmission;
    - CPURoPEAngleWorkspaceLifecycle;
    - CPURoPENegativePositionDirtyWorkspace;
    - CPURoPEAngleWorkspaceAdmission;
    - the RMSNorm thread sweep (89.6 s);
    - the negative controls.
  - This was the first real execution of the test-only commits 0b32f8edd, 912d01073 and
    39c24d8b4.
- `V2_Integration_ParityFixtureShardOwnership` double free: a stale ccache artifact, not
  attributable to the branch.
  - gdb frame: free() ← default_delete<IOrchestrationRunner> at the test's `cache.runner.reset()`.
  - The test object allocated 13800 bytes, from a stale precompiled header and object with the
    pre-branch `ModelMemoryProfile`. The core's `sizeof(OrchestrationRunner)` is 13808, because
    `rope_dimension_count` was added and the runner embeds the profile through
    `std::optional<PlanningModelMetadata>`.
  - ccache 4.9.1 reused the old PCH and object across worktrees (`CCACHE_BASEDIR`, NOHASHDIR,
    pch_defines/time_macros sloppiness).
  - After deleting that PCH family and rebuilding with `CCACHE_DISABLE=1`, the object allocates
    13808 and both cases pass.
  - The affected PCH family covers 9 preflight tests. All 9 were re-certified clean at
    39c24d8b4 and at 5d4e3716a.
  - Gates that share a ccache should clean that PCH family after any header layout change.
- `test_production_pipeline`: ImageIdentityTests 17/17. The 53 other failures are the identical set on pure 425002104 (base defects).
- GB10 CPU A/B (arm-sparks, 302318f1e vs 94d7cbe2f; the tree differs from
  407347f6c only by the base rename, Main's unused-code deletions and test-only changes; no hot-path codegen change). Text was byte-identical and
  the mode assertions passed.
  - 27B decode: 0.0% / −0.28%.
  - 35B MTP-dyn decode: +0.31%.
  - **35B no-MTP decode: +0.80%, real (4v4, about 8σ).**
  - Prefill: no gain (jitter).
  - The decode memo deletion causes no regression, so Main's condition is met.
  - Mechanism for the 35B gain: less per-layer host time. Lock-free inv_freq
    lookups, and one shared angle fill instead of 20 private ones.
- Capture policy: `allow_graph_replay=false` and
  `collectives_graph_capturable=false` are expected for CPU-only runs
  (DeviceGraphOrchestrator.cpp:14762 requires a GPU context). CPU prefill and
  decode both execute stage by stage.

## Review status

Final narrow source-only ACK at 39c24d8b4 (parent 912d01073). Earlier ACKs: dca2a3021
(rebase = b199ccfa0 + R1 exactly), b199ccfa0 (P2 claims), 912d01073 (CPU RoPE test contract).
The ACKs are source-level. WSL/x86 and gcc results are owner evidence.

## Reviewer qualifications (non-blocking)

- Per-row negative-position equality is modulo signed zero, and its reference uses the current
  one-row path. Raw dirty-vs-clean byte equality is the unrestricted workspace-independence check.
- The lifecycle replay's DGO/FEE republication sites were a bounded caller check, not
  certification of every path.

- Decoded-row RMSNorm intentionally rejects widths above 8192 floats.
- The generic partial-profile extraction is covered by a Unit test; the angle-admission preflight
  tests supply the profile geometry directly.

## Follow-ups (tracked, not in this PR)

- RoPEStage reset contract (base behaviour, Main): after `resetSessionState()`, executing without
  republishing a request falls back to the tensor's rows at contiguous positions from 0
  (RoPEStage.cpp:154). This is a silent default, not a failure, and maxim 29 suggests making it
  fatal. Production always republishes, so it is not reached today.

- Unrelated RMSNorm/GEMM/composition stage tests still wrap execution in catch-all→GTEST_SKIP.
  The reviewer noted them as out of scope; this PR converted only the CPU RoPE cases.

- Dead DebugEnv RMSNorm threshold fields (`parallel_min_rows`, `parallel_min_elems`, `min_elems_per_thread`, `max_threads`; `LLAMINAR_RMSNORM_*`): parsed but not consumed by any CPU kernel. For Main to retire with the env table/help.

- Tracker 07 follow-up A: Q16/hybrid RoPE onto the admitted workspace.
- RMSNorm: admitted `cpu_rmsnorm_rows` workspace above 8192 (RMSNormStage needs
  workspace plumbing).
- Idea 25: the per-bucket trace is pending. The nsys path in the uma4 image is
  validated by arm-sparks. Decision notes are in `/tmp/armqual/25c-design-notes.md`.
