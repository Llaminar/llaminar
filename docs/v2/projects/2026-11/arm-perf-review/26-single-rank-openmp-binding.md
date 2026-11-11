# Idea 26 — Single-rank OpenMP binding on GB10 is A725-first (capacity-ordered binding needed)

Status: **Open**. Found 2026-11-08 by the arm-i8mm lane while designing idea 5's
FastestClass placement; code-reading finding, not yet measured.

## Finding

- `MPIBootstrapPhase::configureClusterRankThreads` binds workers
  fastest-first (`capacityOrderedCpus` + `sched_setaffinity` by omp thread id)
  — but only on **cluster paths**: hostfile / rank-selection / automatic
  multi-rank discovery.
- An **ordinary single-rank GB10 serve** with `OMP_PLACES=cores`,
  `OMP_PROC_BIND=close` gets **libgomp's CPU-id place order**: threads 0–4
  land on CPU 0–4, which are the **A725** cores (`core_id` ordering from the
  handoff: A725 = CPU 0-4, 10-14). So "thread ids 0..w-1 = X925" is **NOT
  guaranteed** in the common single-rank path.
- Consequence: worker 0 and any static partition can land on slow cores in
  single-rank serving; a FastestClass placement must not assume a fast prefix.

## Options

- **(a)** Apply the same capacity-ordered binding for **every rank shape**
  (single-rank included). This is a **base-behaviour change** — needs the ARM
  implementer's decision. It also fixes worker 0 / static partitions landing
  on A725 in single-rank serving (independent of the typed placement).
- **(b)** Per-thread class probe published at bootstrap (sched_getcpu →
  inventory class, cached once per team member); `FastestClass` **refuses to
  engage** (hard error or explicitly unadmitted placement) unless the fast
  workers form a prefix.

The lane prefers (a). Verify on `.80` (sched_getcpu per team member) before
deciding — the lane will run that check.

## Validation

- Recorded GB10 CPU topology provides the expected mapping; a runtime check on
  .80 for OMP_PLACES=cores / bind=close single-rank serve; then A/B of
  binding change: single-rank 35B/27B decode+prefill, worker classes measured
  (X925-only / A725-only / mixed).

## Implemented (2026-11-08) — binding-only SHA a7f719dfb (parent afe62c5ef)

- One helper bindOpenMPWorkersFastestFirst + capacityOrderedCorePlaces(mask)
  (one place per physical core, allowed siblings only); used by the cluster
  path and every self-launched OMP_PLACES=cores/close rank.
- Proof: V2_Integration_OpenMPCapacityOrderedCorePlaces (GB10 layout/SMT/
  partial mask) registered in preflight; Test__MPIBootstrap
  CorePlacesFollowCapacityWithinTheRankMask; Test__CPUStartupAffinity now
  asserts every full/half/single close team on capacity-ordered places
  (failed before: worker 0 on CPU 0; passes after; localhost:-1:cpu:0 /
  cpu:0 / localhost:0:cpu:0 on .80).
- Engine effect (C→B, 35B): no-MTP 23.42→23.55, MTP 33.94→34.01, prefill
  noisy — no regression, ≤ +0.5% (dynamic tiles already balance load; gain
  is in serial/static work, as Main predicted).
- Status: pending Main's review/cherry-pick onto PR #43. Rebase conflict with
  the upstream fused recorder was resolved inside the lane's tip
  (recordNativeNEONFusedLaunches, 21-tag schema).

## Taken by Main (2026-11-08) — binding is in PR #43

Main reviewed and cherry-picked a7f719dfb; PR #43 is now a7f719dfb (same
commit object, fast-forward). Local pass: OpenMPCapacityOrderedCorePlaces,
MPIBootstrap*, OpenMPIdleSpinPolicy. The i8mm lane rebases onto a7f719dfb and
drops its binding commit; remaining tip goes to the gate.

## Defect in the taken binding + fix (2026-11-08)

a7f719dfb reads LLAMINAR_SELF_BOOTSTRAPPED/OMP_PLACES/OMP_PROC_BIND via
std::getenv in MPIBootstrapPhase.cpp → fails the getenv-hygiene sanitizer
(V2_Unit_Static_NoDirectGetenvOutsideDebugEnv, V2_Integration_
HostServiceDebugEnvAuthority). Fix offered: ef19d19f1 (DebugEnv::envValue,
3 lines, behaviour unchanged; clean cherry-pick onto 0868a5573). Expected
gate set includes these two as base-class until Main takes the fix.

- Fix TAKEN by Main unchanged (as 12d3f23cd in PR #43 = 79c1f62ab): locally
  passing NoDirectGetenvOutsideDebugEnv, HostServiceDebugEnvAuthority,
  OpenMPCapacityOrderedCorePlaces, MPIBootstrap. The lane drops its copy on
  rebase onto 79c1f62ab.
- Main also fixed the PrefixArchiveBackgroundPublication base race at
  79c1f62ab (worker records a compaction request arriving during
  Copying/Publishing and schedules exactly one more rewrite after its own
  publication; 18/18 + 20/20 pass in-container; .80 repeat-proof requested).

## A1 LANDED (arm-addon3, 2026-11-10, ~21:46Z): 1d9141934 on codex/arm-idea26-openmp-wakeup (base e8866ec04; 8 files, +319)

"perf(cpu): measure OpenMP team fork/join counts at owned boundaries":
- Counter authority: inline-variable relaxed fetch-add noteOpenMPTeamFork +
  snapshot accessor; observed at EVERY real fork branch (OMP_WORKSHARE/_SYNC/
  _IF + both collapses, OMP_LAYER_PARALLEL, the two FA decode raw sites),
  inserted before the pragma (outlined bodies untouched); retained reuse
  counts nothing; prep-only raw sites exempt by name; consumer contract =
  snapshot DELTAS (never absolutes).
- Summary openmp_team_forks_decode_step: exact window delta in
  ForwardExecutionEngine::executeCacheHit under the PerfStats gate AND
  record-family selection — default serving materializes nothing.
- Per-stage family openmp_team_forks_stage: both DeviceGraphExecutor CPU
  stage loops, riding the SAME owned boundary as the stage-clock pair,
  under the stage-timing gate; GPU entries publish nothing.
- COST = EVIDENCE: preflight cells measure the amortized per-note cost over
  1M invocations (2.26 ns direct / 2.32 ns CTest; ceiling asserted at 100
  ns) + the off-default check (no policy → no omp record while forks count).
- TESTS: core-free fixture +2 (exact fork contract; registered as the
  SHARED preflight alias V2_Integration_OpenMPTeamForkCounterContract);
  publication/default/cost cell V2_Integration_OpenMPTeamForkCounterPublication
  (preflight-appended); FULL portability CTest 293/293 green (Mac);
  Linux-engine targets syntax-validated under stubs (runtime validation at
  the canonical Linux gates at landing). A2 rider set BOOKED next (.80).

### Reviewer #179 — A1 HOLD at 1d9141934 (comprehensive fix-back #180 required)

Reviewer findings (four P2s; the macro+two-FA coverage cannot support
"every real fork"; owner-scoped accounting required — NO production
serialization workaround):
- **A1-1 [P2] coverage not exhaustive:** live non-prep raw forks remain
  UNCCOUNTED — Qwen4ExpBranchConcat.cpp:52 (CPU execution),
  MoEOverlaySparseCollective.cpp:1059 (route publication),
  MoELocalExpertStage.cpp:4255, HostBackend.cpp:979ff (collective
  reductions). Audit the runtime inventory + explicit NAMED prep exemptions.
- **A1-2 [P2] process-global snapshots not owned per-device/stage:**
  concurrent TP/executor scopes count each other's forks; serializing one
  executor's waves does not serialize other participants; the new
  exact-attribution comments overclaim. REQUIRE owner-scoped accounting +
  an overlapping-scope regression (no production serialization).
- **A1-3 [P2] data race:** Test__OpenMPWorkshare.cpp:626 assigns shared
  non-atomic `before` from every thread inside a retained parallel region —
  race even when values happen to be equal. Fix: immutable snapshot outside
  or private per-thread snapshot.
- **A1-4 [P2] publication test doesn't exercise publishers:** the core
  test only checks family enablement + self-cost — deleting BOTH production
  publishers would still pass. Add actual-path off/on/filter/GPU-absence/
  exact-delta regressions; RESTORE the cached DebugEnv/collector config
  after env RAII (current final evidence config LEAKS beyond the test).
- Positive recorded: standalone macro branches + both FA sites note;
  summary CPU/decode/success/family-gated; both stage additions CPU-gated;
  both preflight registrations present; no new launcher/concurrency cap;
  self-cost loop = uncontended only (not contended/multi-executor).
  293/293 + 2.26/2.32 ns = unauthenticated owner reports. No A2 evidence
  acceptance or merge grant. Main: A1 HOLD respected; no FF.

### A1 fix-back (#179 successor) landed: 0ce033b6c (arm-addon3, 2026-11-10, ~23:46Z)

codex/arm-idea26-openmp-wakeup @ **0ce033b6c** (parent 1d9141934; 53 files,
+1128/−89; pushed). One comprehensive successor, all four P2s:
- **A1-1 exhaustive:** ALL live runtime raw forks now note — the 4 named
  misses (Qwen4ExpBranchConcat:52, MoEOverlaySparseCollective:1059,
  MoELocalExpertStage:4255, HostBackend ×16 collective sites) + full audit
  (ShmemSpinBackend, DeviceContext, DeviceGraphOrchestrator expert-mask,
  RoPEPrimitives, CPUBackend first-touch, the whole tensor family) = 174
  mechanical one-line notes across 42 files; NAMED one-time exemptions in
  the header (10 files, per-file reasons); LOCKED by
  check_cpu_openmp_fork_note_policy.py (note within 6 lines or file named;
  orphan rejected; fails closed; rc=0: 176 noted / 10 exempt). Negative
  controls (8 cases) green; both registered core-free.
- **A1-2 owner-scoped:** thread-affine owner-cell stack
  (OpenMPTeamForkCell/Scope); concurrent scopes count each other only in
  the total; nested scopes observe the same fork; NO serialization;
  publishers read CELLS never global deltas; overclaim comments rewritten.
  Concurrent-overlap + nesting regressions GREEN.
- **A1-3 race fixed:** baseline captured outside the retained region.
- **A1-4 publishers under test:** publishers moved to
  utils/OpenMPTeamForkEvidence.h owning their gates + GPU refusal
  (enforced in the publisher); actual-path regressions (engine decode
  off/filter/exact/repeat/GPU-absence; CPU graph executor off/exact 2+5/
  repeat/GPU-refusal); env evidence config RESTORES caller env + reloads
  both configuration authorities (leak closed, also in the bootstrap cell).
VALIDATION: portability 295/295 green (Mac); Linux targets syntax-validated
under stubs (runtime at the canonical gates). A2 outputs held pending the
re-review ACK.

### Review #179 successor IN PROGRESS — 0ce NOT ACKable as written (reviewer, ~23:47Z)

Immediate concrete defect: the engine passes scope.forks() into
publishOpenMPTeamForkSummary, but the publisher still computes
openMPTeamForkSnapshot()-entry_forks → mixes owner count with process-total
epoch, yielding unrelated historical/concurrent counts (fresh isolated 3
forks → 0, not 3). ALSO: the default-off call now eagerly constructs
forwardGraphPerfContext BEFORE the publisher gate (side-effect/timing leak
in the off path). Reviewer completing the remaining policy/test audit
before ONE consolidated finding set. 295 portability/stubs = NOT
engine-publication qualification.

### Reviewer verdict #179/#180 successor — HOLD at 0ce033b6c; comprehensive fix-back #183 (seven findings)

- **R1 [P2]** summary still computes globalSnapshot-entry_forks while the
  engine passes scope.forks() — different quantities (isolated 3 → 0;
  prior/concurrent forks contaminate). Publish the TYPED OWNER count
  directly; add a nonzero-prior-total + overlapping-owners test.
- **R2 [P2]** source checker NOT fail-closed/exhaustive: main skips files
  without a matching raw pragma BEFORE the scan (orphan-only notes pass);
  missing src/v2 passes empty; .inc/.inl and _Pragma/alternative whitespace
  omitted (CPUNativeQuantisedWindowPacking.inc:122 is outside inventory and
  NOT a named exemption); six-line matching accepts duplicate notes and
  control-flow-separated notes; ENTIRE exempt files admit new arbitrary
  runtime sites. Need actual-driver controls (the negative controls call
  scan(), not the CLI, missing main's orphan bypass) + narrow named site
  exemptions / explicit supported grammar.
- **R3 [P2]** repeat tests use the WRONG collector contract:
  PerfStatsCollector.cpp:722-723 adds count/value — the second execution
  must produce deltas 2/5 or cumulative 4/10 with count 2, not value 2/5;
  the engine repeat has the same assumption for unchanged keys. Do NOT
  alter collector semantics to satisfy the test.
- **R4 [P2]** ScopedForkEvidenceConfig destructors UNSET rather than
  restore original env values; null options leave ambient selections
  active; the MPIBootstrap reload runs BEFORE its live filter/summary RAII
  destructors → cached config still leaks. Save/restore the EXACT prior
  environment AFTER restoration incl. early assertions; test ambient
  nondefault values.
- **R5 [P2]** gate ownership incomplete: the summary call eagerly builds
  forwardGraphPerfContext before the family check (main_condition_batch
  can allocate) → off-default no-materialization regressed; the stage
  publisher checks family/device but not cpuStageTimingEnabled (its direct
  test publishes with timing off). The gate must own ALL materialization +
  the explicit stage-enable policy.
- **R6 [P2]** the new engine regression + the overlap/nesting test are NOT
  explicitly in preflight filters; the new stage tests exercise the
  ORDINARY executor, not the retained loop; the GPU engine call ignores
  success (absence can pass before the seam). Add discriminating
  actual-path controls/registration for BOTH loops +
  success/failure/prefill-exclusion + GPU refusal; source/stub checks do
  not replace these.
- **R7** economy evidence gap: the unchanged million-note cost loop has no
  owner scope — production now does global+owner atomics + stack walk
  (often 2 owners). Measure the 0/1/2-owner + relevant concurrent configs;
  the prior 2.26/2.32 ns does NOT price production instrumentation.
- POSITIVE: the retained shared-before race fixed; the missed raw sites +
  sampled tensor/DeviceContext/ShmemSpin/RoPE/SIMDHelpers branches now
  noted; TLS cells isolate simultaneous caller-thread owners without
  serialization. TLS does NOT propagate to worker threads — "workers never
  fork" remains a contract requiring proof/rejection or explicit ownership
  propagation, not a process-wide exactness guarantee.
No execution/authentication; 295/295 + stubs = owner evidence.

### #183 fix-back landed: 16d633c15 (arm-addon3, 2026-11-11, ~00:32Z)

Parent 0ce033b6c; 11 files +739/−296; pushed. One successor, all seven
+ TLS/worker boundary, per the reviewer's requirements:
- R1 publishers read the TYPED OWNER COUNT (window/stage cells; no
  global-delta arithmetic); regression = nonzero prior total (7 pre-forks)
  + concurrent owner (5 forks, other thread) on the REAL engine path →
  published == the window's 3 exactly.
- R2 policy fail-closed: scans every supported file (.inc/.inl + _Pragma +
  whitespace variants) with/without pragmas (orphan-only files fail);
  missing src/v2 → exit 2; notes bind ONLY to the immediately-adjacent
  pragma (duplicates = orphans); NAMED SITE exemptions (file + enclosing
  function + per-site reason) replace whole-file exemptions; the
  WindowPacking.inc publication site inventoried (appendNativeWeightGroups);
  negative controls drive the ACTUAL CLI over synthetic trees (8 cases) →
  176 noted sites / 28 named exempt / 54 inventoried files, rc=0.
- R3 repeat tests assert the accumulator contract (stage 4/10 count 2;
  engine 6 count 2); collector untouched.
- R4 guards save/restore the EXACT prior env incl. absence, restore before
  reloading both authorities; off-default FORCES the export vars absent;
  ambient-nondefault test (SUMMARY=0/FILTER=mtp restored verbatim);
  bootstrap reload runs AFTER both RAII guards.
- R5 publishers own ALL materialization (lazy graph-context after the
  gate — off-default decode allocates nothing); stage = cpuStageTimingEnabled
  + family + CPU device with an ungated-caller regression.
- R6 BOTH loops discriminated in preflight (retained multi-device control
  via executeMultiDevice; engine regressions cover success/failure/
  prefill-exclusion; GPU refusal reached PAST the seam via hook
  execution-count assert); CounterContract filter extended; NEW alias
  V2_Integration_OpenMPForkEvidenceEnginePath (preflight).
- R7 self-cost under production ownership: no/one/two-nested/two-concurrent
  owners over 1M notes ≈ 2.3/4.5/6.9/5.2 ns (dev host) — bare no-owner no
  longer stands in.
- TLS/worker boundary explicit + regressed: worker-context forks move the
  TOTAL only (non-zero member proof; team thread-zero = owner thread
  legitimately attributes).
VALIDATION: portability 295/295; Linux targets stub-syntax-validated
(runtime at the canonical gates). A2 held until the next verdict.
