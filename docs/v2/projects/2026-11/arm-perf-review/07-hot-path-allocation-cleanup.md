# Idea 7 — Hot-path allocation / cache cleanup (shared CPU code)

Status: **Open**. These are shared x86+ARM paths, not ARM-specific. They show
up in the ARM hot path once the engine runs and are best as separate small PRs.
Some already have a fatal/typed cleanup pattern to follow (`rejectNumeric`,
idea 10).

## Items

1. **RoPE pair-position tables** —
   `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp`:
   - `apply_rope_pair_positions()` (~:444) allocates two `std::vector<float>`
     cos/sin tables (rows × head_dim/2) on every call;
   - a second allocation site exists for the seq-len path (~:776).
   - `get_inv_freq_cached()` (~:51, :62) takes a **global mutex** on every call
     (including cache hits) and stores `std::unordered_map` keyed by
     (head_dim<<32 | freq_bits).
   Ideas: bounded stack/arena angle buffers or a persistent per-invocation
   buffer owned by the caller; lock-free/sharded or thread-local inv-freq cache
   (values are determined by (head_dim, freq_base) at model load).

2. **RMSNorm FP32 generic path heap fallback** —
   `src/v2/kernels/cpu/ops/CPURMSNormKernelT.cpp`:
   - `MAX_STACK_ROW_SIZE = 4096` (~:42); rows with `ucols > 4096` fall back to a
     per-call `std::vector` (`heap_scratch.resize`). Any hidden size >= 5120
     allocates per call.
   - Native BF16/FP16/INT32 branches already avoid decoded row banks; the FP32
     residual/remaining branches are the ones still needing work.

3. **Gamma Q12 TLS cache** —
   `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp` (~:3120):
   - `thread_local std::vector<int16_t> gamma_q12_cache` keyed by `gamma`
     pointer + cols. Keying by raw pointer is a stale-identity risk: freed and
     reallocated same-address memory with different values would return stale
     quantized gamma (correctness, not just speed). It is a known risk; the
     dedicated fix is to key by an authentic generation/owner identity or move
     the quantized copy into the retained owner.

## Exactness / risk

- Item 1: angle tables are pure coefficient data; caching must preserve exact
  values (the libm cos/sin merge defect found in the indexer shows separate
  calls matter — keep the same call sequence).
- Item 2: scratch is transient; no arithmetic change, but the fallback must be
  covered by the existing per-caller ownership regressions.
- Item 3: arithmetic is identical when the cache is correct; the risk is
  correctness under reuse (needs the typed identity).

## Validation

- Perf A/B for items 1–2 (allocations removed from per-token path);
  focused regressions for item 3 (same-address different-weights reuse).

## Ideas 25+7 lane plan (2026-11-09) — scope confirmed

- F1: per-bucket PerfStats counters ALREADY EXIST (forward_graph/
  prefill_graph_launch + stage_gpu/prefill_graph.replay + lifecycle + cache
  hit; LLAMINAR_PERF_STATS_CSV + GPU_STAGE_TIMING) — 01#6 'only
  moe_placement/mtp/physical_memory' is STALE (corrected). Per-bucket probe
  possible NOW without nsys (GPU-timing run = profiling; separate wall run).
- F2: MAX_STACK_ROW_SIZE heap fallback is non-production (FP32 path uses no
  scratch; BF16/FP16/Q8_1 only); hidden 5120 does NOT allocate in production.
  Still fixed (7c: in-place residual-add + single-row buffers, stack bound
  8192 floats, allocate-but-correct above).
- F3: gamma_q12 TLS cache = x86-AVX512-only (rmsnorm_q8_1_pure_integer), non-
  production; real stale-identity defect → fix via authentic owner identity
  (TensorBase::weak_from_this(); absent owner → per-call quantise); regression
  on WSL x86 AVX512.
- F4 (production): FP32 + partial RoPE allocate 2× std::vector cos/sin per
  call (per bucket per layer prefill; seq_len=1 decode in partial path; every
  team thread allocates+fills its own copy) → 7a admitted cpu_rope_angles
  workspace (CPUInvocationWorkspace doctrine; missing = fatal; shared fill via
  omp for; byte-identical angle elements; WorkspaceMemoryEstimator admission
  as gate) + 7a2 VerifierAngleRows (thread_local growable) onto the same
  buffer (approved). 7b inv-freq: lock-free thread_local memo (reference-
  stable unordered_map nodes), mutex only on miss.
- 25a: add host-clock-only counters (prefill_chunk_host_wall, inter_chunk_gap,
  family_host_wall, request-intercept split) — no sync/D2H, domain-gated.
- 25b: arch-aware install-cuda.sh (x86_64 vs sbsa) + probe/runbook; recipe
  owner = arm-sparks.
- Commit order: 7d → 7b → 7a → 7c → 25a → 25b (25c tracker-only). Reviewer at
  clean checkpoint → handover to Main. Dead-code flag (apply_with_residual_add
  no production callers) + RMSNorm-no-workspace sent to Main.

## Ideas 7+25 lane progress (2026-11-09, branch rebased onto 302318f1e, local)

- 06b205ccd fix(cpu) 7d: Q8_1 pure-integer gamma keyed by owner identity
  (weak_from_this + address + width; quantisation split into own primitive;
  empty/expired owners requantise per call, never cached). Core-free
  regression + negative control (address-only key fails the 3 cases) in
  ProductionTestPreflight; AVX512 syntax probe on Mac; real AVX512 exec = WSL.
- a588345ac perf(cpu) 7b: inv_freq hits lock-free via 8-entry per-thread memo
  of published node addresses (map never erased; values/storage unchanged);
  regression 12 geometries × 6 threads → single address + bit-exact values.
- c93a2d1a8 build(docker) 25b: install-cuda.sh arch-aware (22.04|24.04 ×
  amd64→x86_64 | arm64→sbsa; else fail closed; x86 unchanged); MODE=profilers
  (CUPTI + nsight-compute + nsight-systems); validate-cuda-profilers.sh
  (identity + cuda,nvtx trace must contain kernels + cudaGraphLaunch);
  mapping test in preflight image-contract suite. uma4 (FROM uma3 +
  MODE=profilers) + runbook → arm-sparks builds/validates.
- Mac: portability NEON_DOTPROD Release 261/261 after 7d; RoPE subset 20/20
  after 7b.
- Decisions folded: 7c does NOT touch apply_with_residual_add; >8192 → FAIL
  FAST (fatal naming width+bound) — declared cpu_rmsnorm_rows workspace =
  tracked follow-up; 7a/7a2 via RoPEStage IWorkspaceConsumer, sizing from
  admitted max rows (incl. grouped verifier rows), bind before capture/replay,
  lifecycle reset/replay test, no thread_local cos/sin left in RoPE TU.
- Process: local --no-verify OK (macOS can't configure full engine); WSL x86
  gate = real prerequisite; branch stays local until gate + handover.

## Follow-up A (from the ideas 7+25 lane, allowlisted '07 follow-up A')

Convert the TEST-ONLY Q16/hybrid RoPE family (apply_rope_q8_1_to_fp32,
q16_1_integer, q16_integer, q8_1→q16 incl. fixed-scale/head variants,
q16→q16 fixed/dynamic, compute_rope_sincos_q15) to the admitted angle
workspace. Proven unreachable in production (QwenGraphBase addRoPE /
addKeyCachePublicationTransforms = FP32 Q/K, no Q_out; key-only factory
rejects Q_out/K_out at ComputeStageFactory.cpp:111; ActivationPrecisionPolicy
rejects non-FP32; GraphResolver sets no outputs — allowlisted by function in
the source policy). Dead persistent_state parameter removed.

## Tip + Main follow-ups (2026-11-09) — pushed, authoritative = 94d7cbe2f

- RoPEStage creates its kernel EXACTLY ONCE; type mismatch FATAL
  (LLAMINAR_UNREACHABLE names both types); recreate/rebind removed.
- Angle workspace sized from EFFECTIVE ROTATED WIDTH (partial-RoPE prefix),
  NOT head_dim — stage/kernel/estimator same contract; estimator via
  ModelMemoryProfile::rope_dimension_count / effectiveRotaryDimension();
  profile wire version 6→7 (stale layouts fail); exact-byte asserts full +
  partial widths.
- Two focused preflight regressions (production defects fixed in e17c08412):
  CPURoPEQ8OneRowTeamHeadOwnership (retained team rotates each shared head
  once) + CPURoPEQ8OneRowNegativePosition (independent of thread's previous
  call) — both reproduce the defect against the BASE TU in standalone control.
- Mac portability: 270/270 NEON_DOTPROD + 268/268 NEON @8a553c922, 0 skips;
  source policy clean. Bundle/patches stale; origin branch authoritative.

- PR #43 = 5922cd1f5: Main RETIRED the unused KernelFactory RoPE cache + two
  dead unregistered test sources flagged by this lane (7d/7a review). The only
  remaining failure in that filter = MoEForbiddenDependencyScan (known base).

## A/B results (2026-11-09, 302318f1e vs 94d7cbe2f, .80, 20c -c8192, s42, n256)

- 35B-nomtp: prefill 47.03→47.13 (+0.20%, jitter); decode 23.395→23.550
  (**+0.66%** — BEYOND the ±0.05% memo gate; σ≈0.2–0.5% → 3rd ABBA repeat
  requested). 35B-mtp: prefill −1.62% (A r1 48.73 = outlier rep/jitter),
  decode +0.31%. 27B-nomtp: P −0.83%, D 0.00%. 27B-mtp: P +0.81%,
  D −0.28%.
- Mode assertions ✓ (mtp 1/0 per cell); text byte-identical A vs B (0/257
  mismatch) ✓.
- Extra-B forward_graph CSV (9 rows): notable
  decode_capture_policy allow_graph_replay=false;
  collectives_graph_capturable=false — DECODE CAPTURES DISABLED under
  CPU-only/20T (lane to confirm expected; would explain decode-changed/
  prefill-unchanged). No p50/p95 in bench stdout — latency via PerfStats
  CSV/timer table if needed.
- Expected-outcome framing: prefill gain did NOT materialize (+jitter);
  decode +0.66% on 35B-nomtp is the candidate → repeat or accept-as-noise
  before reviewer.

## Lane read of the A/B (2026-11-09)

- 3rd ABBA repeat accepted on 35B-nomtp (+0.66% ≈ 1.3–3σ, ambiguous; direction
  = improvement — the gate targets a REGRESSION from memo deletion; 27B neutral
  + 35B-mtp +0.31% → memo deletion cost nothing anywhere, Main condition met
  regardless). Prediction if real: 0.3–0.7% (per-layer host time: before =
  mutex + private cos/sin per team member; after = lock-free + one shared
  omp-for trig fill; ~42.7ms/token 35B/10 layers; 27B ~217ms → ≤0.2% noise).
  ≤0.2% → accept as noise. Prefill no-gain fits (GEMM-dominated; RoPE tiny).
- CAPTURE POLICY: EXPECTED, no defect — allow_cached_graph_replay =
  gpu_graphs && isGPU() && segmented (DeviceGraphOrchestrator.cpp:14762);
  CPU executes ComputeGraph stage-by-stage for decode AND prefill (bucket
  replay is GPU); collectives_graph_capturable=false for the same reason (no
  NCCL/RCCL on CPU). Does NOT explain decode-vs-prefill.
- LATENCY: p50/p95 not needed for the mean-tok/s gate; timer table in the
  extra-B CSV; new prefill_chunk_host_wall/family timers populate only on GPU
  bucketed prefill, not CPU runs.

## 3rd ABBA repeat — VERDICT (2026-11-09)

- 35B-nomtp decode, n=4/arm: A 23.450±0.100 vs B 23.638±0.104 →
  **+0.80%** (4v4, consistent sign every pair, ≈8σ vs rep noise). The +0.66%
  was NOT noise — reproducibly +0.7–0.8%.
- Memo-deletion gate condition MET (no regression anywhere: 27B 0.0%/−0.28%,
  35B-mtp +0.31%); prefill no-gain (+0.20% jitter). +0.8% decode = branch
  BONUS (recorded in the handover).

## RENAME REBASE onto 425002104 (2026-11-09)

- #43 rebased + force-pushed: tip **425002104** @ base `46c35f15b`; base
  renamed `NativeVNNI`->`NativeQuantised` (`a0fd16d04`); old tip `5922cd1f5`
  is pre-rename. 989b7bada must be replayed with the map
  (`/tmp/remap/vnni_map.py`) onto 425002104; new tip TBD; WSL x86 gate + Mac
  portability + reviewer on the new tip. A/B economy stands (rename =
  identifiers only; Main's deletions already rebased).

## Rebase onto arm tip 5922cd1f5 (2026-11-09)

- Main retired getOrCreateRoPE + two dead test sources (lane's flagged
  follow-ups). Trial rebase of 94d7cbe2f: CLEAN (7 commits; trial tip
  783d07efc). APPROVED (b): force-push rebased tip; WSL x86 gate + Mac
  portability re-run + reviewer handover all on the rebased tip; A/B economy
  stands (deletions = unused code, no codegen impact — noted in handover).

- FORCE-PUSHED: tip **989b7bada** (origin/codex/hotpath-cleanup-prefill, 7
  commits, base 5922cd1f5); tree differs from 94d7cbe2f only by Main's
  deletions (getOrCreateRoPE + 2 dead test sources; unused → no codegen
  impact, A/B stands). Handed to WSL arm-qual for the x86 gate (after addon
  R1 x86); Mac portability re-run on 989b7bada in progress (lane).

## Mac portability on rebased tip 989b7bada (2026-11-09) — GREEN

- AppleClang 21 Release: NEON_DOTPROD 270/270 + NEON 270/270, 0 ctest/gtest
  skips; ProductionTestPreflight repeat-until-fail:20 NEON_DOTPROD 214/214;
  RoPE angle-storage source policy clean. Handover drafting; only WSL x86 gate
  (@989b7bada) before reviewer.

## Rebased onto 425002104 — tip 8ba25fc28 (2026-11-09)

- Rebase --onto 425002104 5922cd1f5 (no NativeVNNI tokens in diff; clean).
- LATENT COMPILE BUG fixed (lane's, since original 7d; found by sweeping ALL
  callers of the changed APIs outside the diff): Perf__RMSNormThreadSweep
  (registered) + Perf__RMSNormParallelSweep (unregistered) passed float gamma
  to rmsnorm_q8_1_pure_integer (const int16_t* gamma_q11) — old tree fails
  ("cannot convert float* to const int16_t*"). Fixed in 7d (20d85fa31):
  quantize_rmsnorm_gamma_q11 ONCE, outside the timed loop.
- NEW: FromGGUF_ExtractsPartialRotaryDimension in Test__ModelMemoryProfile
  (generic <arch>.rope.dimension_count: undeclared→head_dim, 32→32 wire
  round-trip, 64→full head).
- range-diff vs 989b7bada: commits 2/4/5/6 =; commit 1 = sweep fix; commit 3
  (docker) context only (MODE=full cuobjdump + MODE=profilers/repo merge;
  bash -n clean); commit 7 = profile test.
- Checks: gcc-16 aarch64 syntax 25 TUs + 13 callers; Mac portability
  NEON_DOTPROD 270/270 + NEON 270/270 0 skips; preflight 214/214 ×20;
  source policy clean; ImageIdentityTests 17/17; test_production_pipeline
  53 failures = IDENTICAL on pure 425002104 (base defects). A/B stands.
- Pending: reviewer pass at 8ba25fc28; WSL (5'); handover draft
  /tmp/armqual/hotpath-handover.md.

## Orphan decision (2026-11-09): DELETE Perf__RMSNormParallelSweep.cpp

- Stale, not distinct: setenv sweep is a no-op (debugEnv() static singleton,
  read once; DebugEnv.h:4358); no kernel reads the four threshold fields
  (git grep empty in src/v2/kernels/cpu); never registered; ThreadSweep
  covers thread-count scaling. Lane adds a SEPARATE BOTTOM commit
  'test(perf): retire the orphan RMSNorm parallel-threshold sweep';
  20d85fa31 then edits only compiled code. New tip SHA follows; production
  deltas unchanged. Dead DebugEnv threshold fields = Main follow-up
  (handover list; lane didn't remove).

## Retired — tip 407347f6c (2026-11-09)

- 8 commits on 425002104; force-push from 8ba25fc28. Bottom commit
  36027718a = retire Perf__RMSNormParallelSweep.cpp (−440, git grep empty);
  2f635c796 = 7d touching only registered ThreadSweep; other 6 content-
  identical. `git diff 8ba25fc28 407347f6c` = one file deleted. Rebase/
  equivalence + Mac portability carry over (file never built). WSL (5') at
  407347f6c; ThreadSweep the only RMSNorm perf cell; dead DebugEnv
  LLAMINAR_RMSNORM_* fields = Main follow-up (handover).

## Reviewer interim (2026-11-09) — RoPE negative-position contract (P2 likely)

- Unchanged production code: ordinary floating RoPE fills with
  RoPEPaddingRows::Skip (negative-position angle rows untouched) but
  consumers rotate those rows unconditionally (RoPEPrimitives.cpp:531–542,
  694–696, 826–856; BF16/FP16 share). Old ordinary helper computed angles for
  EVERY abs position; direct primitive call with n_past=-1 consumes
  dirty/uninitialized workspace. Typed FP32 kernel skips negative positions
  BEFORE invoking primitives → reviewer NOT claiming ordinary-stage path
  impact. Likely bounded P2 (exported primitive contract) + focused
  dirty-workspace negative-position regression. Ownership TBD after pass
  (lane's 7a/7a2 RoPE workspace vs base). No probe; tip stays stable.

- Lane confirms OWNERSHIP: introduced by 7a (b5d9ebf9d), not base. Scope
  confirmed (5 fills :531/:694/:826/:1438/:1980; unaffected: verifier :895,
  Q8_1 :2709, typed paths, inline *_direct). FIX plan: 5 fills ->
  RoPEPaddingRows::Compute (base semantics, base-identical bytes); enum
  rename Skip->VerifierPadding (consumer must honour active) /
  Compute->EveryPosition; consumer-policy asserts; no contract change, no
  new fatal. Alternative (not preferred): fatal reject n_past<0.
  Regression: V2_Integration_CPURoPENegativePositionDirtyWorkspace
  (ProductionTestPreflight; poison+negatives, all formats/rotary/seq lengths,
  team in/out; negative control fails at 407347f6c; rope_bytes_driver
  n_past<0). Routing (a) into branch before handover preferred, (b) separate
  follow-up — awaiting reviewer final.

- ROUTING = (a): fix on the branch BEFORE handover (current tip NOT green for
  handover). Reviewer specifics: keep negative control at unfixed tip
  407347f6c; regression covers ≥32-row BF16/FP16 table paths + one-row/
  partial FP32; compare vs APPLICABLE original arithmetic; EveryPosition/
  VerifierPadding contract (no fatal). Reviewer finish HOLD on this P2
  (remaining source read + orphan-deletion check); will review exact fix
  SHA. Lane implementing; WSL (5') continues as unfixed-tip signal (final
  gate will be at fixed tip).

- 8ba25fc28 FINAL: HOLD on lane-owned RoPE P2 ONLY; no additional P1;
  Q11/sweep/owner-key/inv_freq/workspace/wire-v7/GGUF/host-timer/Docker all
  affirmed. Qualifications (nonblocking): >8192 decoded-row reject intended;
  generic partial-profile Unit + preflight focused geometry; host-timer
  restore lambda not RAII across ASSERT (optional fold). Nothing
  authenticated/promoted to this tip. Reviewer next: 407347f6c deletion
  delta; then fix SHA.

## RoPE fix — beffdbe13 (2026-11-09)

- One commit FF from 407347f6c. EveryPosition/VerifierPadding replaces
  Skip/Compute (no fatal; base contract kept). EveryPosition: 5 ordinary
  fills + Q8_1 (:540/:703/:835/:1447/:1989/:2718); VerifierPadding: verifier
  fill only (:904; consumers skip by active).
- Regression V2_Integration_CPURoPENegativePositionDirtyWorkspace
  (Preflight; CPURoPEWorkshare.OrdinaryNegativePositionsIgnoreDirtyAngle
  Workspace): FP32 full+partial 1/8/40; FP16/BF16 1+40 (32+ tables);
  n_past -1/-3; poison 3 ways; standalone + team of 3 (one late); reference =
  original serial arithmetic via odd symmetry (P<0 = conjugate of conjugated
  input rotated at -P; signed zeros folded).
- NEGATIVE CONTROL at 407347f6c: 120/120 FAIL (all entry/rows/poison/team,
  incl. 40-row tables) → after fix 13/13. Cross-TU byte proof (base TU vs
  fixed TU): -3/-40/-100 IDENTICAL; -1 rows>=2 IDENTICAL; 0/17231 incl Q8
  IDENTICAL; unfixed TU differs at byte 1. Exclusions (base defects):
  Q8_1 one-row negative (stale-TLS, own regression) and base rows==1
  n_past=-1 ABORT (last_pos -1 sentinel). RAII env restore folded in.
- Gates: Mac NEON_DOTPROD/NEON 271/271 0 skips; preflight 215/215 x20;
  source policy clean; gcc aarch64 + clang x86 syntax OK. WSL (5') final =
  beffdbe13; reviewer per-tip pass in flight.

## beffdbe13 production reread (2026-11-09)

- P2 SOURCE-CLOSED; timer guard restores on ASSERT exits; NO new production
  P1. EVIDENCE-LIMIT corrections (not production failures), relayed:
  (1) regression comparator = equality modulo signed zero (not byte-exact
  certificate) + calls current implementation → add complementary RAW
  dirty-vs-clean byte check; :716 comment overstates.
  (2) :760 never measures team (budget != OMP_THREAD_LIMIT) → can pass
  serialized; record/assert actual width w/o barrier before late entrant.
  (3) driver SKIP_Q8_INT suppresses entire Q8 block; MIN_ROWS drops one-row
  group; sampled rows {1,2,3,7,31,32,33,40,64} ≠ contiguous 1..64.
  (4) base −1 sentinel = fresh persistent-state route only (update_rope_cache
  empty planes + apply_rope_cached_heads throws); OLD NO-STATE ordinary +
  partial APIs still give valid one-row −1 → don't claim unavailable.
- Recommended small follow-up commit before handover (raw byte check, actual
  width assert, comment fix, accurate evidence description). Lane deciding.

## b199ccfa0 (2026-11-09) — test-only; all four qualifications folded

- (1) RAW byte check (cells = clean-workspace bytes, no zero folding; 4
  workspace contents); canonical documented 'modulo signed zero'; :716
  comment corrected (calls current one-row path, not independent arithmetic;
  base equality = cross-TU harness). (2) team width measured (member 0
  omp_get_num_threads(), no barrier; assert == request; OMP_THREAD_LIMIT=1
  FAILS as designed). (3) harness rows 1..64 contiguous; Q8 exclusion =
  rows==1 negative-start ONLY; MIN_ROWS removed.
  (4) base −1 = fresh persistent-state route only; BASE_NO_STATE=1 gives
  valid no-state one-row −1 → IDENTICAL.
- Harness: no-state −1/−3/−40/−100 IDENTICAL; state −3/−40/−100 IDENTICAL
  (−1 base abort); 0/17231 incl Q8 IDENTICAL; unfixed TU differs at byte 1.
- Negative control: canonical 160/160 fail; raw 112 (clean cells can't fail;
  some BF16 coincide); width 0. Gates: Mac 271/271 both ISAs; preflight
  215/215 ×20; gcc/clang x86 syntax OK. Reviewer test-delta pass in flight;
  WSL (5') final = b199ccfa0.

## b199ccfa0 FINAL source ACK (2026-11-09)

- Narrow source-only ACK, no new blocker; parent beffdbe13; ONLY
  Test__CPURoPEWorkshare.cpp changed (production repair untouched). Four
  qualifications confirmed source-level (rawBytes all bits + clean-standalone
  per-cell; PriorPositive real prior call; member-0 width after join, no new
  barrier; reference comment correct). Driver scope confirmed (1..64,
  MIN_ROWS out, Q8 rows==1 neg-start only, BASE_NO_STATE); inspected scopes,
  NOT output authentication. Minor wording limits (not blockers): sleep
  doesn't prove arrival order; argv[2]=17231 = starts 17231+rows (record as
  argument/offset). P2 stays source-closed. SOURCE COMPLETE for handover;
  remaining = WSL (5') only.

## Rebased tip dca2a3021 (2026-11-09)

- 10 commits on 5d4e3716a (correction from 9: orphan deletion + 7 original +
  review fix + P2 fix + claims commit). range-diff b199ccfa0..dca2a3021: all
  `=`; whole-tree diff = same 50-file set + identical diffstat as
  425002104..5d4e3716a; after stripping index/hunk numbers → identical
  (only offsets in the 2 overlapping CMake files). = b199ccfa0 + R1 exactly.
- Mac gate at dca2a3021: NEON_DOTPROD 289/289 + NEON 289/289 0 skips;
  preflight 226/226 ×20; RoPE source policy clean. Reviewer mechanical pass +
  WSL (5'') = dca2a3021 in flight; handover draft next.

## 4 UNIT REGRESSIONS on x86 (WSL b199ccfa0 signal; 2026-11-09)

- (1) ComputeStage ×3: RoPE runs without binding the admitted
  cpu_rope_angles workspace (b5d9ebf9d) → 'requires its admitted workspace
  binding' throw (RoPEPartialRotaryKeepsFullHeadStride,
  KeyOnlyRoPEIsByteIdenticalToTheSameQueryRotation,
  RoPEVerifierPrefillMatchesSerialDecodeRows); siblings GTEST_SKIP on other
  message.
- (2) WorkspaceMemoryEstimator: 180928 vs 177408 (:468); :2477/:2494
  786944 vs 0 + 24640 vs 0 (angle-workspace bytes expected 0).
- (3) MemoryPlanner.CPUOnly_AttentionWorkspaceUsesPhysicalWorkerBOM:
  103859456 vs 102282496 (:2836/2842) — BOM/expectation angle-plane bytes.
- (4) CUDABuilderProvisioning ×3: install-cuda.sh arch selection
  (3dbfe04a0/a8198400c) via dpkg --print-architecture vs test shim printing
  nothing → 'Unsupported architecture \'\''. Lane test code.
- ALL deterministic device-free Unit; pre-commit gate would catch; Mac
  dca2a3021 run = portability + preflight only (NOT V2_Unit_*). Production
  contract stays; stale tests updated. Lane fixing; WSL (5'') held until the
  fixed SHA. Rebase equivalence confirmed (0 lines, 45 files +2855/-1162).

- REVIEWER FINAL dca2a3021: narrow mechanical source-only ACK; no rebase
  discrepancy. 10 linear commits; range-diff all `=`; base-change deltas =
  same 50 paths/numstats (48 identical blobs; 2 CMake patches equal after
  index/hunk normalization; registrations retained). Final source =
  reviewed cleanup + R1 delta, no extra rebase content. No semantic
  non-interference claim; owner evidence boundaries; merge authority = Main.
  Fix commit (4 unit regressions) = new tip → quick test-delta re-ACK →
  WSL (5'').

## Fix 0b32f8edd (2026-11-09) — test-only; 4 regressions resolved

- +85/−11, 6 test files, no src/scripts-docker. (1) ComputeStage ×3 bound
  (stage-sweep found 2 more: 1316 norm-preservation hidden GTEST_SKIP now
  bound; RoPEGraphCaptureNoH2D CPU ref bound). (2)(3) estimator/planner
  expectations add canonical cpu_rope_workspace::requirements (3520/786944/
  24640/1576960 — match WSL actuals exactly). (4) provisioning dpkg shim +
  exactly-one os-release + replay cases (amd64/arm64 × 22.04/24.04 URLs;
  ppc64el/'', 20.04 fail pre-curl).
- HONEST LIMIT: Mac can't configure src/v2 (libnuma) → full V2_Unit_* gate
  NOT runnable on Mac (explains the slip); lane ran provisioning 4/4 vs real
  installer (HEAD 3/3 fail), gcc-16 syntax (2 macOS-env only), byte-exact
  contract compile. WSL (5'') = first real execution; reviewer test-delta
  pass + WSL in flight.

- WSL (5'') queued at 0b32f8edd (after (5') preflight, ahead of N2/N3);
  diff saved; equivalence dca2a3021 = b199ccfa0 + R1 holds through the fix.
- GPU-gate note: Test__RoPEGraphCaptureNoH2D registered only as
  V2_Integration_{CUDA,ROCm}RoPEGraphCaptureNoH2D → x86 can't certify it;
  plan: GB10 full CUDA gate at merged tip (post spill-fix) exercises it
  (arm-sparks informed; handover note for the lane — not a seal blocker).

- Reviewer interim 0b32f8edd: bindings/charges/provisioning consistent; ONE
  test-contract item: Test__ComputeStages.cpp:1320–1338 catch-all
  std::exception→GTEST_SKIP('requires device enumeration') + checks under
  `if (success)` (false execute returns pass silently) = hidden-skip
  MECHANISM intact. Required: ASSERT_TRUE(execute()), exceptions FAIL, norm
  assertions unconditional. Lane pushing test-only follow-up; WSL (5'') held
  for the final SHA. Not a production defect.

## 912d0107 (2026-11-09) — hidden-skip mechanism removed (test-only)

- ComputeStages norm-preservation: ASSERT_TRUE(execute); catch-all→skip
  removed; unconditional norm assertions. Same mechanism found in all four
  CPU RoPE cases (ComputeStage: RoPEBasic, RoPEPositionOffset,
  RoPEExplicitPositionIds_BatchedParity, position_ids-override) — converted;
  no RoPE skip path remains (grep empty). Removing it exposed the
  position_ids-override 3 stages UNBOUND (regex missed digit names — lane's
  error, acknowledged) — bound. Correct re-sweep: 33 RoPEStage constructions,
  all accounted. Mac gcc-16 -Wall 0 diagnostics; WSL (5'') = first execution.

## 912d0107 FINAL source ACK (2026-11-09)

- Narrow source-only ACK; 0b32f8edd RoPE test-proof HOLD CLOSED. Exact
  parent + two-file test-only; no production/installer change. All five CPU
  RoPE cases assert directly; position-override 3 stages bound (declaration
  sized, lifetime covers checks); both fixtures CPU; 33-site re-inventory
  clean (unbound = metadata/null-context/rejected/GPU/later-bound/intentional
  lifecycle negatives; GPURoPEProbeStage GPU graph-owned). Scope note:
  unrelated RMSNorm/GEMM catch-skip wrappers remain (not in scope).
  SOURCE COMPLETE. Remaining: WSL (5'') + handover (lane preparing doc) +
  N2/N3 (arm tip, independent).

## (5') signal FINAL (2026-11-09) — 2 open preflight items

- Build clean; Unit 948/966 (14 base + 4 flagged); Preflight 2341/2361
  (16 base + 4 new); ThreadSweep PASS 86s; NEGCTL rc=1 ×4; focused 7/8.
- COVERED by 0b32f8edd: QFNSelectedAttentionWorkspaceAdmission +
  CUDABuilderProvisioning (same causes).
- ❗ CPURoPEAngleWorkspaceLifecycle.SerialFamilyReusesOneBindingAcross
  ResetAndReplay: 12 memcmp EXPECT fails (q/k vs serial ref; −192/−202/−223),
  5/5; other 3 cases pass; includes grouped verifier −1 row + beffdbe13
  negative-position change in area → lane owning production-vs-reference
  analysis. ❗ ParityFixtureShardOwnership.EvidenceRequiresExactRetained
  RunnerIdentity: SIGABRT double-free 9/9 at b199ccfa0, PASSES at
  5d4e3716a/ae090ec81 → (5'') 912d0107 = direct control (attribution
  verdict). No seal until both resolved.

## RoPE lifecycle mismatch — ROOT CAUSE DIRECTION (2026-11-09)

- x86 diagnostics: primitives IDENTICAL at 912d0107 + 5d4e3716a (all routes,
  OMP 1/4/32, -ffp-contract=off) → defect ABOVE primitive. DIAG: all
  elements differ (all 3 stages, both banks); 'actual' = EXACTLY the
  UNROTATED seeds (q −0x1.e8p+1, k −0x1.ep+1) → execute() returns true but
  leaves Q/K UNTOUCHED; 2-of-3 replays per stage (6×2 = 12 EXPECTs); 3rd
  replay rotates. Pattern: first OK; after resetSessionState() execute =
  SILENT NO-OP (bound angle workspace or kernel state cleared by reset).
  Production-relevant (maxims 17/29 + unrepresentable-invalid). Lane fixing:
  reset must not invalidate the admitted binding (or re-admit); execute must
  fail fatally on unbound workspace (no silent-skip); focused regression
  (reset+replay rotate; fail-fast case). NOTE: adaptive-diag patch applied
  at b199ccfa0 (not 912d0107 — gate tree untouched; files byte-identical);
  reverted clean. WSL's own earlier null-adapt = disregard (not product).

## (B) resolved — 39c24d8b4 (2026-11-09) — TEST-SIDE root cause

- CORRECTION: execute() NOT a no-op; binding/kernel state NOT invalidated.
  resetSessionState() (base, RoPEStage.h:202) clears REQUEST-LOCAL params;
  test executed WITHOUT republishing → seq_len-from-tensor fallback
  (RoPEStage.cpp:154) + pos 0..n−1; row0 pos0 = identity (cos1/sin0);
  binding intact (getWorkspace/address EXPECTs passed); primitives exact on
  x86 (WSL). Production republishes every request (DGO:26161/26176/61286;
  FEE:653) — test never modelled it. FIX (39c24d8b4, +20/−5): each replay
  publishes via updateDynamicPositionIds (different live positions +1000×
  replay; −1 kept) → reset+republication byte-exact vs serial incl. verifier
  rows; binding/address survives reset. Main follow-up (handover): base
  RoPEStage silent 0..rows−1 fallback on reset-without-republication (fail-
  fast candidate; outside lane). (5'') final re-targeted to 39c24d8b4.

## 39c24d8b4 FINAL source ACK (2026-11-09)

- Narrow source-only ACK; no blocker. Correction matches unchanged contract
  (reset clears request-local; base reset + CPU reset don't retire the
  admitted workspace; seq_len==0 → rows from Q + offset 0 → identity row0;
  NOT no-op/binding issue). Per-iteration verify: copy+shift 1000×iter
  (non-cumulative), −1 preserved, Q/K restored, publish via
  updateDynamicPositionIds before execute (stage-owned copy → lifetime safe);
  serial uses same new positions; memcmp + workspace checks remain. Cited
  DGO/FEE paths publish — bounded caller check. Oracle = serial invocation of
  the SAME production kernel (NOT independent arithmetic — handover wording).
  execute-after-reset default-position/fail-fast = Main follow-up. SOURCE
  COMPLETE at 39c24d8b4; only WSL (5'') remains → handover.

## (C) VERDICT — STALE CCACHE ARTIFACT (2026-11-09) — (5'') GREEN

- WSL retracted the ideas-lane attribution. gdb frame: free() ←
  default_delete<IOrchestrationRunner> ← cache.runner.reset() (:110); failing
  binary alloc 0x35e8 (13800) vs core sizeof 13808 (ModelMemoryProfile 344 +
  rope_dimension_count) → 8-byte write past allocation; CCACHE_DISABLE=1 +
  PCH-family rebuild → 0x35f0, E alone + S alone PASS. ccache 4.9.1 served
  pre-change PCH/object across worktrees (/src mount, CCACHE_BASEDIR=,
  NOHASHDIR, pch_defines/time_macros sloppiness). Blast radius = 1 PCH
  family, 9 preflight tests; re-cert 39c24d8b4 9/9 + 5d4e3716a 9/9.
- (5'') FINAL GREEN: Unit 960/974 (14 base), Preflight 2357/2373 (16
  base/WSL; ParityFixture passes post-PCH-clean); per-case all OK
  (ComputeStage ×7, ComputeStages, estimator 53/53, MemoryPlanner,
  provisioning 4, lifecycle, thread sweep 89.6 s, NEGCTL rc=1 ×4, 0 skips).
  TRIGGER MET → lane handover to Main (FF 5d4e3716a → 39c24d8b4).
- Systemic: gate2.sh PCH-family clean rebuild before testing (PCHCLEAN);
  recommended for .80 r2gate.sh (addon lane). N2 relaunched PCH-clean
  20:14Z; N3 next.
