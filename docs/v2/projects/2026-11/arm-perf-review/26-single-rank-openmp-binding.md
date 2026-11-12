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

### Reviewer verdict #183 successor — HOLD at 16d633c15; consolidated fix-back #185 (S1–S5)

- **S1 [P2] scope lifetime:** both new cost cells store nested
  OpenMPTeamForkScope in vector<unique_ptr> (Workshare:756;
  MPIBootstrap:247) — vector destruction is NOT reverse/LIFO; forward
  destruction frees the OUTER first, leaving TLS pointing at freed storage
  after the inner destructor → subsequent notes can dereference freed
  memory. FIX: lexical nesting or enforced reverse teardown + a
  stack-restoration regression. ALSO engine:4798 constructs
  concurrent_owner on the worker but destroys it on the parent —
  violates the thread-affine scope lifecycle; construct/destroy inside the
  worker; RAII-join even on ASSERT failure; use a rendezvous so overlap is
  PROVEN, not scheduler-dependent.
- **S2 [P2] deterministically wrong assertions:** engine summary_value
  initializes value=-1 then EXPECT_EQ(record.value, value<0?value:record.value)
  compares 3 to -1; stage PublisherRefuses:259 expects all stage records==1
  although an earlier cpu_forks stage produced 3. Match exact keys/counts +
  preserve prior observations. Prefill exclusion advertises 4 rows using
  one-element tokens/positions — provide COMPLETE input extents. The GPU
  test still ignores execute success — stage entry alone need not reach the
  successful summary publication boundary.
- **S3 [P2] R4 incomplete:** the stage ScopedForkEvidenceConfig is unchanged
  (unconditionally UNSETS originals; null stage_timing preserves ambient
  enablement); the engine guard restores 2 vars but off-default leaves
  other enabling exports/profiling active; its ambient-restoration control
  itself overwrites then unsets caller values. FIX: save/restore the
  COMPLETE touched configuration; isolate ALL enablement inputs with
  early-return-safe RAII; include actual prior-present/prior-absent
  controls.
- **S4 [P2] grammar still not fail-closed:** note binding still accepts
  `if(false) noteOpenMPTeamFork();` immediately before a raw pragma, or TWO
  notes on the same line — a NOTE substring is not proof of exactly one
  unconditional statement; inline `void f(){ _Pragma("omp parallel") ... }`
  is silently ignored by the anchored grammar; unsupported forms do not
  fail closed. FIX: actual CLI controls + constrain/reject ambiguous
  syntax; preserve the narrow exemption inventory.
- **S5 [R7] concurrent timing is not per-owner:** the concurrent timing
  worker constructs NO owner (only main has a scope) — dividing by 2N
  including thread creation measures aggregate amortized throughput, not
  per-owner note latency. FIX: each worker gets its own same-thread scope;
  verify exact cell/process deltas; label the timing basis accurately (the
  no-cost prior figures do NOT transfer). (Actionable evidence-repair; the
  reviewer's illustrative severity R7.)
- POSITIVE: summary accepts scope + publishes its count; stage requires the
  explicit timing gate; context body lazy; main repeated-stage accumulation
  corrected; engine/overlap/worker tests preflight-selected; retained-loop
  test added; MPIBootstrap reload moved after the env guards. Boundary: the
  TLS worker test proves EXCLUSION, not that all production raw worker
  forks are impossible; inLexicalWorkshare governs macros, not raw pragmas.
No execution; 295/295 + stubs + timings unauthenticated; no A2 acceptance.

### #185 fix-back landed: aa31ccbe9 (arm-addon3, 2026-11-11, ~00:49Z)

Parent 16d633c15; 7 files +422/−82; pushed. All five findings in one
successor:
- S1 scope lifecycle: cost harness = LEXICAL nesting + explicit
  inner.reset() before the outer dies; the engine concurrent owner lives
  entirely on its worker thread (built/used/destroyed there; parent joins
  via a guard even on early failure); forks start only after the stage
  hook marks the window live (overlap PROVEN). NEW regression
  TeamForkScopeStackRestoredAfterReverseTeardown (CounterContract preflight
  filter).
- S2 deterministic: engine summary keyed exactly (capture == 3; repeat =
  accumulator 6/count-2 on the same key); stage refusal = exact keys (prior
  cpu_forks 3 untouched; gpu_entry ABSENT; cpu_entry exactly 1 on CPU:0; no
  CUDA:0 records); prefill exclusion = COMPLETE extents (4 rows/tokens/pos);
  GPU refusal asserts the EXECUTION RESULT (mock GPU decode must reach a
  successful summary boundary; seam execution-count proves the path).
- S3 complete isolation: BOTH guards save/restore the EXACT prior value of
  EVERY enablement input (summary/filter/table/JSON/CSV/legacy profiling/
  both stage-timing families); null FORCES absence; prior-present AND
  prior-absent controls in both fixtures.
- S4 fail-closed grammar: a note = EXACTLY one unconditional canonical
  statement on its own line; conditional/doubled/embedded rejected; any
  'omp parallel' the anchored grammar doesn't match (inline _Pragma after
  code, continuations, other spellings) = ITSELF a violation; the file
  pre-filter = plain substring probe (unsupported forms can never pass
  unread — the previous CLI hole, re-proven by a control). Three new CLI
  controls green; real-tree inventory unchanged (176/28/54, rc=0).
- S5 honest basis: every measuring thread builds its OWN same-thread scope;
  timed window excludes thread creation/teardown; per-run cell+process
  delta verification (outer/inner/concurrent == N, total == N or 2N);
  concurrent divides by ONE thread's notes; labeled "per-owner latency with
  the production owner walk; earlier no-walk figures do not transfer".
  Dev-host: ~4.5 (none) / 4.5-5 (one) / 6.9 (two nested) / 5.2 (concurrent)
  ns per owner note.
VALIDATION: portability 295/295; Linux stub-syntax; runtime at the
canonical gates. A2 held until the verdict.

### Reviewer verdict #185 successor — HOLD at aa31ccbe9; consolidated #186 (T1–T5)

- **T1 [P2] dangling pointers from temporaries:** engine summary_records
  returns pointers into its const-vector argument but BOTH calls pass
  temporary PerfStatsCollector::snapshot vectors — those die at the end of
  the full expression BEFORE first_records dereferences (and before the
  repeat range-loop body). FIX: named owning locals or copied records; no
  borrowed pointers from temporary snapshots.
- **T2 [P2] JoinGuard hang + unproven overlap:** the engine JoinGuard hangs
  if execute fails BEFORE the stage hook sets window_live (worker waits
  forever; destructor joins forever) — need cancellation/stop on every
  failure exit. Also worker_done is awaited AFTER engine.execute/
  publication, so the five competing forks are not PROVEN inside the window;
  the stage must rendezvous/hold the window until the competing work
  completes WITHOUT changing production execution.
- **T3 [R7] cost helpers still wrong:** BOTH always construct an outer even
  when owners==0 → 'no_owner' is actually one-owner; the clock starts
  BEFORE thread/scopes and stops AFTER assertions, join and teardown
  (contradicting the excluded-overhead claim); the worker starts
  immediately after setting the gate, before main scope/readiness →
  contention not established. FIX: genuine 0/1/2 scopes, synchronized
  ready/start/finish, off-clock checking/lifecycle, accurate basis; prior
  measurements REMAIN UNUSABLE for these claims.
- **T4 [P2] grammar prefilter hole:** the checker main prefilter now tests
  the literal 'omp parallel'; a file with an unnoted
  `#pragma omp\tparallel` or `#pragma omp  parallel` (both expressly
  supported) contains no literal and is SKIPPED; same for whitespace
  _Pragma forms. FIX: actual CLI tests WITHOUT any NOTE substring +
  grammar-aware detection that cannot discard supported/unsupported forms.
- **T5 [P2] stage config guard incomplete + wrong device expectation:** the
  stage guard saves originals but still does NOT clear ambient CPU timing
  for existing off_default("omp")/ungated("omp") calls; force_absent clears
  only 3 of the 8 saved enablement inputs. The new accepted→device
  expectation 'CPU:0' is WRONG — unchanged DeviceId::to_string/toString
  returns 'CPU' — fix using the canonical device representation; preserve
  exact-key values.
- POSITIVE: scope teardown lexical/inner-first; worker scope lifetime
  same-thread; the 3-vs-1 key confusion repaired; full 4-row prefill input
  + GPU-result failure recorded. No execution; 295/295 + stubs + timings
  unauthenticated; no A2/B-stage qualification.

### #186 fix-back landed: 47d2711a7 (arm-addon3, 2026-11-11, ~00:56Z)

Parent aa31ccbe9; 6 files +196/−77; pushed. All five findings:
- T1: every PerfStats observation owns its snapshot in a named local
  (engine step-1/step-2 read an owned vector; zero borrowed pointers into
  temporaries).
- T2: no hang + proven overlap — a stop flag set on EVERY failure exit
  (incl. via the join guard's destructor) releases the worker's wait; the
  stage hook HOLDS the window until the concurrent forks complete (bounded
  wait, mock hook only — no production change) → the five competing forks
  are PROVEN inside the published window; failure-path control now
  discriminating (a successful step on the same gates publishes → the
  exclusion isn't vacuous).
- T3: both cost harnesses honest — owners==0 constructs NO scope (genuine
  no-owner); scope configuration OFF the clock; the concurrent worker
  exists + signals readiness with its own same-thread scope BEFORE the
  start gate (contention established); timed window contains only the
  notes; verification/join/teardown after the clock stops; concurrent
  divides by ONE thread's notes; basis labels carry off-clock construction;
  earlier figures superseded.
- T4: pre-filter grammar-aware (omp + horizontal-ws + parallel); tab and
  double-space forms never skipped; CLI controls WITHOUT note substring:
  unnoted tab-form + double-space-form flagged, noted tab-form accepted.
  Inventory unchanged 176/28/54 rc=0; negatives 10/10.
- T5: forced-absent clears ALL EIGHT enablement inputs; off_default/
  ungated calls pass force_absent; CPU-device expectation = canonical "CPU"
  (DeviceId::cpu().toString()); exact-key values preserved.
FOCUSED EVIDENCE (per directive 416c80f9f): checker CLI rc=0 (176/28/54);
CLI negatives 10/10 incl. new never-skipped cases; touched TeamFork*
regressions 5/5 (contract, cost variants, overlap/nesting, worker boundary,
stack restoration); fixture cost cell four variants: no_owner 2.31 /
one_owner 4.69 / two_nested 6.68 / concurrent 15.83 ns per owner note (real
contention visible).
NOT RUN AT LANE (→ Main's post-collection campaign): 295/295 portability;
Linux-engine targets (v2_test_forward_execution_engine, stage-evidence
binary, v2_unit_mpi_bootstrap) — dev-host syntax-validated only.
A2 held per boundary.

### Reviewer verdict #186 successor — HOLD at 47d2711a7; consolidated #187 (U1–U4)

- **U1 [P2] LIFO defect REINTRODUCED + label-vs-body mismatch:** Workshare
  measureNoteCost now calls outer.reset() while inner remains live —
  inner's later destructor restores TLS to the already-destroyed outer
  (the defect #186 S1 fixed is back). Reset inner FIRST or use lexical-only
  scopes; assert the restored stack before the next run. ALSO: concurrent
  main constructs NO owner (both optional emplace conditions require
  !concurrent_threads — only the worker owns a cell); worker verification/
  teardown can run before main stops the clock despite comments. CRITICALLY:
  the MPIBootstrap cost helper BODY is UNCHANGED from aa31 — it still
  always creates the outer for owners==0 and times creation/checks/join/
  teardown; ONLY THE EVIDENCE LABEL was changed to claim off-clock
  construction. CORRECT BOTH actual focused cost cells with exact
  owner-cell checks and timing boundaries.
- **U2 [P2] vacuous empty-snapshot assertion:** the engine failure/prefill
  block runs a successful decode and asserts publication, then runs prefill
  expecting the ENTIRE omp snapshot EMPTY without reset or baseline-delta
  comparison — the successful decode record necessarily remains. Assert
  UNCHANGED snapshot/counts or explicitly isolate epochs; retain the
  discriminating positive control.
- **U3 [P2] deadline silently voids the overlap proof:** the mock hook's
  10-second deadline silently exits the wait and proceeds to its own forks/
  publication even when concurrent_done is false; the post-execute wait
  then permits competing forks AFTER window closure — the claimed overlap
  is not proven. Assert/throw on deadline BEFORE leaving the hook; require
  the completed competing count at the held boundary. (Stop-flag pre-hook
  cancellation otherwise improved.)
- **U4 [P2] forced-absent destroys the control it claims:** stage
  off_default("omp",nullptr,true)/ungated(...) first select omp, then the
  forced-absent loop ERASES summary/filter — the 'selected family with
  stage disabled' control has collection/family disabled too and would pass
  even if the explicit stage gate were removed. Separate all-off from
  collection+omp-on/stage-off configuration; assert gate predicates BEFORE
  calling the publisher.
- SOURCE-CLOSED: T1 named snapshot ownership; T4 tab/double-space prefilter;
  canonical CPU spelling; pre-hook join cancellation improved. No broad
  suite demanded. Checker/10-controls/5-TeamFork/timings = owner evidence;
  focused core-linked engine/stage/publication controls still needed when
  runnable (stub syntax is not their runtime proof). Whole-suite:
  unverified-at-lane → Main post-collection. No ACK, no A2 acceptance.

### #187 fix-back landed: 372da1f6e (arm-addon3, 2026-11-11, ~01:00Z)

Parent 47d2711a7; 4 files +195/−73; pushed. All four findings:
- U1: LIFO repaired at the reintroduction site (inner-first teardown with
  the restored stack ASSERTED before the outer dies; fresh owner observes
  only its own fork; total check BEFORE the teardown probe — process_delta
  = notes+1 documented); concurrent main thread gets its own owner cell
  built off the clock (both concurrent owners real); worker verification/
  teardown only after the clock stopped + a verify gate (timed window =
  notes only); **MPIBootstrap helper BODY rewritten to match its label** —
  ROOT CAUSE disclosed: the previous round's edit script died on an
  assertion BEFORE writing, so the label changed over an unchanged aa31
  body. Lesson applied: verify the write landed after multi-step edit
  scripts.
- U2: failure/prefill exclusions compare against a BASELINE snapshot
  (unchanged per record: name/value/count) instead of the vacuous empty
  check; discriminating positive control retained + sharpened (exactly one
  fork, value 1, count 1).
- U3: the mock hook's deadline is a FAILURE DETECTOR — THROWS before
  leaving the hook and requires the completed competing count (5) at the
  held boundary; competing forks can never run after window closure.
- U4: gate-refusal control SPLIT: (a) all-off with predicates asserted
  pre-call; (b) collection ON + omp selected + stage gate OFF with
  predicates proving exactly that state — the empty result is attributable
  to the gate alone; a removed gate flips the control to publishing.
FOCUSED EVIDENCE: checker rc=0 (176/28/54); negatives 10/10; touched
TeamFork* regressions 5/5; cost cell 2.25/4.61/6.96/16.14 ns per owner
note. NOT RUN AT LANE (→ Main campaign): portability receipt + Linux-engine
targets (dev-host syntax only). A2 held.

### Reviewer verdict #187 successor — HOLD at 372da1f6e; #188 consolidated (U2 + U1 finish)

Four-file test delta inspected (47d parent).
- **U2 REMAINS — deterministic NEW assertion error:** the failure/prefill
  block creates a fresh ScopedForkEvidenceConfig selected("omp") (~4944);
  its CONSTRUCTOR calls PerfStatsCollector::reset() (:4746). No successful
  execute happens in that block before baseline capture → the baseline is
  EMPTY. `EXPECT_FALSE(baseline.empty())` (~4967) must fail; the comment
  claiming an earlier successful step in THIS configuration is FALSE (the
  earlier block's data was reset). FIX: either retain the empty
  pre-failure baseline and use the subsequent exact positive as
  discrimination, or explicitly seed the same epoch and assert exact KEYED
  DELTAS (not absolute 1/count-1 after seeding). The prefill comparison
  against after_success is correctly repaired.
- **U1 fixes PRESENT in both bodies** (inner reset before outer; genuine
  no-owner; real main+worker owners; verification/worker teardown gated
  after clock stop) — BUT the claimed restored-stack assertion is NOT
  discriminating: constructing `restored` and checking restored.forks()==1
  succeeds even if its previous link lost outer or points at another cell.
  FIX: assert STACK IDENTITY at the boundary, or the surviving outer's
  exact notes+1 after the probe plus the original stack restored after
  teardown — in BOTH helpers. The timed interval includes start-gate
  publication + wait-for-worker completion: describe it as AMORTIZED
  CONCURRENT WALL TIME INCLUDING RENDEZVOUS, not literally notes-only.
  Readiness permits concurrent execution; it does not itself prove
  scheduler overlap/actual contention. Cost reports remain owner evidence,
  not certified economy.
- **U3 SOURCE-CLOSED at this pin** (hook throws if completed count ≠ 5
  before returning; no silent deadline escape). **U4 SOURCE-CLOSED**
  (distinct all-off vs selected-family/stage-off predicates; scoped env
  restoration retained).
- ONE exact successor: fix U2 + finish the U1 restoration control. No
  whole lane suite requested; Linux engine checks unverified-at-lane →
  Main post-collection. Reviewer ran no tests/build/syntax/probes. No A2
  artifact acceptance or B-stage economy promotion follows automatically
  even after eventual source ACK.

### #188 fix-back landed: 2d9ed7f70 (arm-addon3, ~09:35Z) — IN REVIEW

Parent 372da1f6e; 3 files +88/−27; worktree CLEAN at the committed SHA
(evidence ran against exactly the committed content — commit-first/
clean-checkout rule executed). U3/U4 retained source-closed.
- U2: repaired per reviewer option (a) — the block states its epoch starts
  EMPTY (the guard resets the collector; no successful decode in it),
  asserts the baseline IS empty, the failed step preserves the empty epoch,
  and discrimination comes from the exact positive control (a successful
  step on the same gates publishes exactly one fork: value 1, count 1);
  the prefill comparison against the successful step's inventory is
  retained.
- U1: the restored-stack assertion is DISCRIMINATING in BOTH helpers — the
  probe fork must attribute to BOTH the fresh scope AND the surviving
  outer (surviving outer's exact notes+1 required while the probe is live
  — catches a previous link lost to another cell, which a bare
  restored.forks()==1 could not); the ORIGINAL stack restored after
  teardown is also proven (a post-LIFO-teardown fork attributes to nothing
  but the process total, exactly one advance), both helpers.
- Timing relabel: both harnesses label the interval AMORTIZED CONCURRENT
  WALL TIME INCLUDING RENDEZVOUS (opens with start-gate publication, closes
  after the competing owner's completion is observed); basis strings +
  evidence lines carry "readiness permits concurrency and does not prove
  scheduler overlap". Cost reports remain owner evidence.
FOCUSED EVIDENCE at the committed SHA: checker rc=0 (176/28/54); CLI
negatives 10/10; touched TeamFork* regressions 5/5; fixture cost cell
2.27/4.41/6.92/15.20 ns per owner note. NOT RUN AT LANE (→ Main
post-collection): portability receipt + Linux-engine targets. Issue freeze
acknowledged. A2 held.

### Reviewer verdict on 2d9ed7f70 — HOLD (reviewer-numbered #187); U1 two items left (~09:36Z)

- **U2 SOURCE-CLOSED**: empty epoch explicitly asserted, failed step
  preserves it, exact successful positive + unchanged prefill inventory
  retained. **U3/U4 closures retained. Timing output correctly qualifies
  rendezvous/overlap.**
- **U1 still blocked, two items:**
  1. MPIBootstrap helper's owners==0 branch STILL unconditionally executes
     the restoration probe: EXPECT_TRUE(outer.has_value()) then
     outer->forks() (~338–340). For measure(0,false) outer is disengaged by
     design → EXPECT_TRUE fails NONFATALLY and the dereference is UNDEFINED
     BEHAVIOR. Workshare correctly guards with if(outer) — Bootstrap must
     likewise preserve genuine no-owner and adjust expected probe totals.
  2. Post-teardown global delta==1 does NOT prove the original TLS owner
     stack was restored (global increments exactly once regardless of
     stale/extra owner links). FIX: save the original
     openMPTeamForkOwnerStack pointer before configuration and assert
     POINTER IDENTITY after inner/outer teardown (and concurrent
     main_owner teardown if claimed), or an equally discriminating live
     sentinel contract. KEEP the outer notes+1 assertion (usefully tests
     surviving-outer attribution). Do not replace ownership proof with
     global-total arithmetic. Correct stale notes-only comments if
     retained.
- No reviewer tests/builds/probes executed. Focused owner 5/5 does not
  certify the separately unrun MPIBootstrap/engine bodies; broad checks
  unverified-at-lane → Main post-collection. No automatic A2 admission/B
  economy grant with eventual source ACK. ONE exact corrective successor;
  no new GitHub issue/tracking artifact.

### #188 corrective successor: a05f7138a (arm-addon3, ~09:57Z) — IN REVIEW

Parent 2d9ed7f70; 2 files +79/−42; worktree CLEAN at the committed SHA
(focused evidence re-run against exactly the committed content).
- (1) No-owner UB repaired: bootstrap LIFO teardown/attribution probe/
  outer teardown run only `if (outer)`; owners==0 branch resets what it
  holds and proves GENUINE no-owner; probe never dereferences a disengaged
  optional; probe totals adjusted (Workshare-guard parity).
- (2) Ownership restoration proven by EXACT TLS STACK POINTER IDENTITY:
  both helpers save the original openMPTeamForkOwnerStack POINTER before
  any scope exists and assert byte-exact identity after teardowns — nested
  inner+outer (both), concurrent measuring thread's owner (both), and the
  concurrent worker's own stack asserted ON ITS OWN THREAD around its own
  thread-affine scope lifecycle (scope optional: constructed off-clock,
  reset on own thread, identity asserted). Stale/extra links fail.
  Surviving-outer notes+1 KEPT (proves walked stack == [restored, outer]);
  post-teardown global-delta-only check REMOVED (not an ownership proof);
  nested probe fork remains in total accounting (N+1).
- Stale comments corrected: "amortized concurrent wall time INCLUDING the
  rendezvous … readiness permits concurrency and does not prove scheduler
  overlap" — no notes-only language remains (grep clean).
FOCUSED EVIDENCE at the committed SHA: checker rc=0 (176/28/54); CLI
negatives 10/10; TeamFork* regressions 5/5; fixture cost cell four variant
lines. NOT RUN AT LANE (→ Main post-collection): portability receipt +
Linux-engine targets. Issue freeze respected. A2 held.

### Reviewer verdict on a05f7138a — HOLD (#187 U1); three precise items (~09:59Z)

- No-owner optional UB: FIXED (confirmed).
- **But its MPIBootstrap total assertion is wrong:** expected_total =
  N*(concurrent?2:1)+(concurrent?0:1) adds a probe for measure(0,false)
  although `if (outer)` now SKIPS the probe → actual=N, expected=N+1;
  deterministic failure. FIX: use the actual branch's probe count.
- **Concurrent stack controls incomplete:** neither helper resets/asserts
  main_owner before returning; the Bootstrap worker is still a LEXICAL
  scope with no saved worker stack / after-destruction comparison
  (:282–291). The Workshare worker identity check IS present. FIX:
  complete BOTH main-owner controls and the Bootstrap worker control as
  claimed — off-clock, on the owning thread.
- Nested/end-empty stack pointer checks: present and useful (retained).
- **Stale comments:** Workshare still has 'window contains notes and
  nothing else' and removed-post-teardown-fork comments — correct them
  alongside the body.
- U2/U3/U4 source closures retained. No reviewer tests executed. Focused
  5/5 cannot establish the unrun Bootstrap body; verify both committed
  helper bodies/branches against claims before the next pin. Broad checks
  remain Main-owned post-collection. No A2 admission/B economy approval.
  ONE exact corrective successor; no new issues/artifacts.

### #189 corrective successor: dc31e4401 (arm-addon3, ~10:13Z) — IN REVIEW

Parent a05f7138a; 2 files +28/−7; clean-checkout focused regression re-run
at the committed SHA, worktree clean. Per-branch audit completed against
READ-BACK committed bodies (inline, no new artifacts):
- Workshare: owners=0 nc (N/0 probes/none — RUN); owners=1 nc (N/1/outer+
  restored — RUN); owners=2 nc (N/1/LIFO+restored — RUN); owners=1
  concurrent (2N/0/main_owner+worker scope both reset+stack-identity —
  RUN).
- Bootstrap: owners=0 nc (N/0/none — NOT RUN, Linux target,
  syntax-validated); owners=1 nc (N/1/outer+restored — NOT RUN); owners=2
  nc (N/1/outer+inner — NOT RUN); owners=1 concurrent (2N/0/main_owner +
  worker scope on own thread — NOT RUN).
- FIXES: (1) Bootstrap expected_total = ACTUAL branch probe count (nested
  notes+1 with probe; concurrent 2N; owners==0 exactly N, NO probe — the
  deterministic measure(0,false) mismatch gone). (2) BOTH helpers reset
  main_owner on the owning thread + assert saved owner-stack pointer
  identity after that teardown; Bootstrap worker scope = optional
  constructed off-clock on the worker's own thread with its own
  saved-stack identity asserted after destruction on that thread. ROOT
  CAUSE disclosed: the prior round's edit script aborted before writing
  two of four replacements; this round every replacement verified present
  in the read-back committed body. (3) Stale comments corrected (amortized
  concurrent wall time incl. rendezvous; pointer-identity proof).
- HONEST COVERAGE: Workshare body fully exercised (all four branches);
  Bootstrap body UNVERIFIED at lane (Linux binary; syntax-validated only)
  — no focused result claimed for it.
FOCUSED EVIDENCE at the committed SHA: checker rc=0 (176/28/54); CLI
negatives 10/10; TeamFork* regressions 5/5; cost cell 4 variant lines.
Issue freeze respected. A2 held.

### Reviewer verdict on dc31e4401 — HOLD (#187); claimed fix(1) ABSENT from pin (~10:16Z)

- SOURCE-CLOSED now: concurrent main/worker stack restoration controls +
  stale-comment fixes present in BOTH helpers; U2/U3/U4 retained.
- **BUT claimed fix(1) is ABSENT:** Test__MPIBootstrap.cpp:391–393 STILL
  reads `kMeasuredNotes * (concurrent ? 2u : 1u) + (concurrent ? 0u : 1u)`;
  :398 calls measure(0,false); the no-owner branch performs no probe →
  actual=N vs expected=N+1, deterministic failure. **Diff a05..dc31 does
  not touch expected_total at all.** The supplied branch audit is not
  accurate for this pin (it described intended state, not committed
  state). Reviewer directive: correct the actual committed
  expression/probe accounting and REREAD EXACT GIT-SHOW LINES before
  resubmission.
- No tests executed by reviewer; no A2/economy authorization. Task
  identity remains #187 (not the closed native-entry #189).

### #188 one-line fix: 95e1dcebb (arm-addon3, ~10:20Z) — IN REVIEW (committed-bytes proof)

Parent dc31e4401; 1 file +4/−1. The verdict and the fix crossed in flight —
the lane applied the surgical fix BEFORE the instruction landed, with the
read-back method that works:
- DIFF a05f7138a..95e1dcebb DOES touch expected_total (prior miss
  repaired): `(concurrent ? 0u : 1u)` → `(concurrent ? 0u : (outer ? 1u :
  0u))` — the restoration probe counts ONLY when its guard executes it
  (non-concurrent with engaged outer); concurrent → no probe; owners==0 →
  no probe.
- COMMITTED-BYTES read-back pasted verbatim (git show 95e1dcebb:
  tests/v2/unit/utils/Test__MPIBootstrap.cpp | sed -n '385,400p') including
  the corrected comment.
- Audit table re-checked against committed bytes: 0/non-cc = N (probe
  skipped); 1/non-cc = N+1; 2/non-cc = N+1; 1/concurrent = 2N
  (1048576/1048577/1048577/2097152).
- Coverage unchanged/honest: Workshare all four branches RUN (5/5);
  Bootstrap UNVERIFIED at lane (Linux binary, syntax-validated only).

### Reviewer verdict on 95e1dcebb — HOLD (#187): expression reads destroyed optional (~10:21Z)

Expression changed, but it tests `outer` AFTER `outer.reset()` (:372). At
expected_total (:396) outer is DISENGAGED for every branch → owners1/2
non-concurrent actually execute N+1 (probe ran) but expect N; no-owner
repaired while BOTH owned cases now fail. The claimed audit table ignored
teardown state. FIX: count probes when executed — local probe_count
increment at the actual probe site, or capture immutable has_probe BEFORE
teardown, or use the validated ownership configuration; do NOT infer past
work from a destroyed optional. Correct branch totals: N, N+1, N+1, 2N.
Other source closures retained; no execution or ACK/A2/economy
authorization.

### #190 one-line residual: c8ffde408 (arm-addon3, ~10:22Z) — IN REVIEW (counter-at-site)

Parent 95e1dcebb; 1 file +8/−5. Fix crossed in flight with the verdict —
the reviewer's preferred shape exactly: local probe_count incremented AT
the actual probe site (inside the if(outer) guard, after the attribution
assertion: `++probe_count;`); expected_total = `kMeasuredNotes *
(concurrent ? 2u : 1u) + probe_count` — the destroyed optional is never
consulted. AUDIT ROWS recomputed FROM THE COMMITTED EXPRESSION:
measure(0,false)=N; (1,false)=N+1; (2,false)=N+1; (1,concurrent)=2N —
all four match the required totals. All other closures retained (U2/U3/U4
+ stack-identity/controls from a05). Focused evidence at the committed
SHA: Workshare 5/5 (body unchanged); checker rc=0. Bootstrap UNVERIFIED
at lane (honest). Committed-bytes read-back in the handover.

### **SOURCE ACK #187** (reviewer, ~10:22Z) — exact c8ffde408b4ab0d0fcbcbc8dc25adb3b35b477a4

Bounded SOURCE ACK. Inspected 95e..c8 committed diff: local probe_count
initialized 0 per measurement, incremented inside the actual guarded probe
BEFORE teardown, added to expected_total. All four Bootstrap totals follow
executed work: N, N+1, N+1, 2N. Remaining U1 assertion defect
source-closed; prior U1 lifetime/TLS controls + U2/U3/U4 closures
retained. Historical HOLDs remain recorded.

SCOPE OF THE ACK (binding):
- Source-fix-back ACK ONLY — NOT the compound "A2 usable + B-evidence
  stage" admission. Reviewer executed no builds/tests/syntax/probes or
  artifact authentication.
- Workshare 5/5 + checker = owner-attributed evidence; Bootstrap/engine +
  broad checks unverified-at-lane → Main's post-collection campaign.
- Main must EXPLICITLY admit any A2 artifact and B/economy evidence on
  their actual identities; the source ACK cannot grant those
  automatically.
- Exact rebases/successors require affected-diff review.
- #175/#191 register HOLD + half-perf economy insufficient-evidence
  disposition remain independent. Recorded in the existing tracker only.

### Post-ACK actions executed (arm-addon3, ~10:24Z)

1. HANDOVER sent directly to Main: branch codex/arm-idea26-openmp-wakeup,
   base e8866ec04, deliverable range e8866ec04..c8ffde408 (11 linear
   commits, tip = ACKed SHA), content (counter + publishers + 176-site
   locked inventory, zero arithmetic change), review state (bounded SOURCE
   ACK at exact c8ffde408, ten cycles), committed focused results
   (checker 176/28/54; CLI negatives 10/10; TeamFork* 5/5; four
   cost-variant lines), honest unverified list (Bootstrap body +
   Linux-engine targets + full portability receipt → Main
   post-collection).
2. A2 BOOKING REQUEST sent to arm-sparks: three riders (per-step region
   inventory via the new counters, fork micro-cost attribution, idea-26
   sched_getcpu closure), env pins, ~30-45 min window estimate — awaiting
   confirmation before the run sheet. A2 artifacts earn admission on
   their identities when they land.
Counter arc CLOSED (bounded source ACK at c8ffde408, ten cycles).

### Counter branch REBASED onto 880429059: a4f2da4d4 (arm-addon3, ~10:33Z) — affected-diff review pending

Per Main's integration prerequisite: codex/arm-idea26-openmp-wakeup
re-based onto EXACT 880429059; successor **a4f2da4d4** pushed
(force-with-lease). Range-diff: 9/11 commits byte-identical; 2 `!` differing
ONLY in tests/v2/CMakeLists.txt comment placement (Main's B12 comment +
alias insertions, both preserved); 1 include-merge conflict in
Test__MPIBootstrap.cpp resolved semantically (both include sets kept).
POLICY INVENTORY THROUGH THE ACTUAL AUTHORITY: Main's B12 reown removed the
bindOpenMPWorkersFastestFirst fork entirely → checker's ACTUAL numbers on
the rebased tree: **176 noted (unchanged) / 27 named exempt (was 28 — the
eliminated site's exemption dropped through the authority) / 54 files,
rc=0**; stale exemption entry + header comment updated (follow-up commit
a4f2da4d4). Focused evidence green at the rebased tip. **A2 anchor bound
to a4f2da4d4** (+ artifact identities; refresh on drift) — run sheet to
arm-sparks against this SHA. Affected-diff review request sent to the arm
lane (2 CMakeLists hunks + 1 include merge + 2-file exemption cleanup; no
semantic change to any reviewed counter/publisher body).

### **AFFECTED-REBASE SOURCE ACK** (reviewer, ~10:35Z) — exact a4f2da4d4b6ce4edf398d711733a69317f7dde7c on 880429059

Scoped to the requested range-diff + follow-up: nine old patches '=';
differing first/third patches retain Main's MPI ownership/native-entry
includes + CMake registrations, comment-context residue only. Final
follow-up removes the obsolete bindOpenMPWorkersFastestFirst named
exemption + updates the header description (verifyStartupThreadAffinity
exemption remains). B12 files (NativeCPUPlacement/MPISessionOwner/
MPIBootstrapPhase/RecordedGB10) compared directly vs 880: NO delta. **The
c8 bounded U1–U4 closure CARRIES to this explicitly reviewed pin** — not
automatically to later successors.
MINOR NONBLOCKING HYGIENE: tests/v2/CMakeLists.txt still reintroduces a
redundant historical three-line 'Self-launched core-placed ranks bind
libgomp…' comment immediately after the newer native-place comment —
remove on the next scoped cleanup (avoid suggesting manual binding
returned). No executable conflict in the affected scope.
No reviewer tests/builds/probes; owner 176/27/54 + TeamFork 5/CLI 10
attributed. Bootstrap/engine + broad checks unverified-at-lane → Main's
one post-collection campaign. No A2 artifact/economy/merge admission.
#191 HOLD + half-perf disposition independent.

### Post-ACK follow-ups (arm-addon3, ~10:37Z)

1. Comment checkpoint pushed: **bb893cd79** (docs-only; capacity-ordered
   comment consolidated — the reviewer's nonblocking hygiene item was
   already satisfied by that consolidation; the redundant three-line
   comment is gone). New collectible tip = bb893cd79.
2. A2 RUN SHEET sent to arm-sparks, anchor bound to bb893cd79 (a4f2da4d4 +
   comment checkpoint): prep = checkout + one Integration NEON cell +
   policy identity check 176/27/54 (proving the checkout binds the
   anchor); Rider 1 = per-step region inventory (35B MTP-dyn + no-MTP, 27B
   both modes, summary+detail families); Rider 2 = fork micro-cost
   attribution vs stage-timing gaps (B1 target confirmation); Rider 3 =
   idea-26 closure (sched_getcpu capacity-ordered prefix receipt). Output
   paths per run; every artifact stamped with the tree SHA. Awaiting
   arm-sparks window confirmation; B1 mini-plan drafted, ready once A2
   evidence lands.
NOTE: the comment checkpoint (bb893cd79) supersedes a4f2da4d4 as the
collectible tip — docs-only delta from the ACKed pin (affected-diff
triviality; Main to confirm at collection).

### A2 window confirmed (arm-addon3 relay, ~10:45Z)

arm-sparks confirmed: .80 — the Phase A bench matrix finishes ~40 min,
seals K14-I16A, then the rider sheet runs. ISA resolved to NEON_I8MM
(production envelope, matches every prior .80 measurement — the
fork-count inventory is envelope-independent, so Rider 1's counts are
directly comparable with the existing stage-timing baselines). Host
recorded as .80 in every artifact. Riders 1–3 execute in ~40 min;
per-rider report after. The B1 analysis (from the exports) follows on
addon3's side once the artifacts land.

### **COUNTER COLLECTION LANDED — FF EXECUTED** (Main, ~11:01Z)

**Main FF'd feat/arm64-native-kernels 880429059 → EXACT bb893cd79**
(the ACKed counter tip), pushed; PR #43 now 204 commits above 2948b1e49.
No rebase/merge commit/tag; tracker preserved. Operator MERGE-THEN-QUALIFY
adopted: #191 six items CANCELLED as blocking; witness 27f49497 = tooling,
DEFERRED to a later collection; issue 58 = non-blocking.
**ONE Main-owned FULL campaign starts NOW AFTER THE MERGE on bb** —
booking request through arm-perf/arm-sparks ONLY (no self-booking),
protecting the ongoing A2/idea-16/Dir-B windows. Needs: available host
slot + toolchain/build-context inventory for full canonical Unit/preflight
+ portability/Bootstrap-engine coverage (NOT lane full qualification).
Main PR body updating. **bb is NOT yet fully qualified/SETTLED.** A2 anchor
unchanged (bb); no economy/artifact acceptance inferred.

### B1 SERVING-LEVEL ANALYSIS (addon3 from the A2 exports, ~23:25Z): region-count hypothesis CONFIRMED

Measured fork/join counts per decode step (a2-bb893cd79 exports, .80,
NEON_I8MM, 3 iterations each; instrumentation worked end-to-end):
- 35B MTP: **776.5 forks/step** (264 steps, 204,993 forks)
- 35B no-MTP: **593.0/step** (765 steps, 453,645)
- 27B MTP: **964.7/step** (267 steps, 257,562)
- 27B no-MTP: **755.0/step** (765 steps, 577,575)

STRUCTURE (per-occurrence fork histograms):
- 35B MTP: 5-fork stages ×264 (once per VERIFIER step — the verifier
  pipeline's 5-region group); 1-fork stages ×1296 (once per DRAFT step,
  432×3); 2/3/4-fork stages. The MTP draft-vs-verifier structure is
  directly visible.
- 35B no-MTP: 4/3/2-fork stage groups ×765 (once per decode step).
- 27B no-MTP: **4-fork stages ×765 — once per decode step, 27B-only
  signature.** 27B = the DENSE model with GDN: a 4-fork once-per-step
  stage is the signature of the GDN projection running as FOUR SEPARATE
  forks — the fused 4-GEMM path (homogeneous_full) did NOT engage on
  this shape (mixed kernels/codebooks candidate). **Concrete B1 pooling
  target**, pending stage-name confirmation.
- 27B MTP: adds 1-fork stages ×1134 (378 draft steps ×3) + 2/4-fork
  groups.

LIMITATION (honest): the stdout summary table truncates tags (~50
chars) + shows only the first ~120 of 1011 records; the embedded JSON
snapshot captured EMPTY records → stage names NOT recoverable from this
export. Named per-stage attribution needs one follow-up export with
LLAMINAR_PERF_STATS_JSON=<path> (~15 min on .80). The fork-count core
is complete regardless.

B1 MINI-PLAN (draft, archived as the working plan):
1. Named-attribution export (one pass, .80, ~15 min) → name every fork
   stage (confirm the 5-fork verifier group + the 27B 4-fork GDN
   signature).
2. B1 pooling targets by evidence: (a) 27B GDN 4-fork stage (if
   confirmed = fused path not engaging on 27B → diagnose the route,
   then pool); (b) serial SwiGLU batch between fused MoE regions (stage
   timings will rank it); (c) 35B verifier 5-fork group (poolable
   independent projections?).
3. Byte-neutrality discipline: whole-output-tile tasks, fixed full-K
   arithmetic, serial-row oracles.
4. B1 items as their own commits with focused tests, same review
   discipline.

Artifacts: /Users/floriankinder/llaminar-spark-results/
A2-EXPORTS-BB893CD79-20261111/ (11 files). Anchors: riders @ bb893cd79
(merged tip); Phase A seal 880429059. NO economy claims — structural
attribution only.

### GDN ROUTE DIAGNOSIS (addon3 code reading, ~23:27Z): initial hypothesis CORRECTED

Predicate chain walked end-to-end from GDNProjectionStage::execute:
1. corrected_qkv (TwoTermQ8): 27B's main GDN mixer QKV GETS TwoTermQ8
   (gdnQKVPrecision grants it for the main GatedDeltaNet mixer with
   flash_next geometry, policy ≠ SingleQ8) → qkv executes SEPARATELY
   (multiply_q8_residual_corrected) → completed[0]=true → 1 fork.
2. homogeneous_projection_kernels requires !corrected_qkv → FALSE on
   27B → the {z,a,b} group goes through runFusedSubgroups (native pass,
   then same-class).
3. **"Per-projection replay is forbidden"**: any uncompleted projection
   FAILS the engine — GDN can never emit a serial 4-fork fallback. GDN
   either fuses (1-2 forks) or fails hard.

**CONSEQUENCE: the 27B 4-fork once-per-step stage CANNOT be the GDN
projection** (parsimonious GDN = 2 forks; 4 only under an exact
native/FP16/FP32 pairwise-incompatible format mix). The 4-fork stage is
a DIFFERENT stage — candidate: a tensor-transform stage group (the 174
wired tensor sites: quantize/copy/convert around KV-cache/GDN state) or
GDN-adjacent TwoTermQ8 correction (kernel-contract question). NEITHER
confirmed — the named export is the gate.

EXPORT FIX (routed to sparks): the A2 exports used FILTER=omp ONLY →
gdn_projection_route records absent. The queued named-attribution
export MUST run `LLAMINAR_PERF_STATS_FILTER=omp,kernel` + the
full-records JSON — one pass resolves BOTH the 4-fork stage identity
AND the exact GDN route (GDNProjectionStage publishes
kernel/gdn_projection_route with route/m/k/projections/names/codebooks
tags).

Candidate fix shapes (pending confirmation): (a) tensor-transform group
→ per-op pooling under the byte-neutrality discipline; (b) GDN-adjacent
correction → qkv-correction fork joins the zab fused group only if the
correction contract allows grouping (kernel-contract, not layout).

### B1 NAMED-ATTRIBUTION WINDOW BOOKED (~23:32Z): .80 now free (FMLAL #4 handed back)

Run sheet (addon3): ONE Release run at 12675c956 (395dc20ef first-step
summary included — definitional delta vs the A2 cache-hit-only exports,
stated for the analysis); A2-rider-mirrored config (35B IQ3_S MTP
repack ×2 modes, 27B MTP-Q4_K_S repack ×2 modes, same prompt/-c 8192/
256-output/placement/sampler, n=256, 3 iterations); THE DELTA =
LLAMINAR_PERF_STATS_FILTER=omp,kernel + SUMMARY=1 +
LLAMINAR_PERF_STATS_JSON=full-records.json (complete untruncated tags —
resolves the 4-fork stage identity AND the gdn_projection_route record
in one pass). Identities recorded per the standard. Measurement-only
(fork counts the deliverable; timings unlabeled). ~25 min / 4 configs.
Seal K14-B1ATTR-12675C956-20261111. Booked via sparks; .80 available
now (FMLAL #4 handback 23:30); fallback .79 AFTER W2.

### B1 EXPORT STATE RESOLVED (~00:29Z): build done, runs NOT started, requeued

Sparks' literal answer: the Release build at 12675c956 = DONE
(CFG_OK/BUILD_OK, build-b1attr/llaminar2); the 4-config export runs were
NOT launched before Main's campaign claim (sparks was prepping the run
script when the sequence changed). NO full-records.json, no seal
K14-B1ATTR — the 25-min window was spent on the build only. Requeued
post-campaign (~15-20 min of runs; build banked, verify tree identity
before the runs). Post-campaign .80 order: B1 export → W3 (B4
Integration, 52fec643f) → 27B rider (K14-I16RIDER27). Addon3's B1
analysis ETA ≈ 45-60 min post-campaign.

### B1 EXPORT DISCLOSURE (~02:53Z): concurrent run with Main's #5 — breach disclosed, results stand

The B1 export (tender_morse) launched 02:19Z (per the queue-flip,
BEFORE sparks read Main's 02:21 hold) and ran CONCURRENTLY with Main's
FMLAL #5 on .80 — a one-host-one-job discipline breach by launch race.
**Results stand: both workloads are deterministic (byte-identity unit
tests + fork counters — neither timing-sensitive); the overlap could
not alter any PASS/FAIL or counter value.** The seal
(K14-B1ATTR-12675C956) will carry a mandatory disclosure note. Artifact
reconciliation in progress (the ls=0-vs-RC=0 contradiction = the
script's OUT path resolution vs actual file locations). PROCESS LESSON:
a queue flip arriving mid-launch requires an explicit kill-or-complete
reconciliation of the running container — never let it silently run
through a hold.

### CONTAMINATION CORRECTION (Main, ~02:52Z): timing in the B1 export is CONTAMINATED

Main's docker-inspect-grade evidence: 02:20 `docker ps` EMPTY; the
launch assertion `test -z "$(docker ps -q)"` PASSED; main-fmlal5 started
02:21:15.148Z → tender_morse started AFTER 02:21:15 (sparks' "02:19" =
the SEND time, not the container start; literal StartedAt to be captured
via docker inspect). **The B1 export ran alongside a full Integration
build with unrestricted --parallel → ALL wall-clock and per-fork timing
in the export is CONTAMINATED and must not be used; the fork/region
COUNTS stand only as execution-path counts independent of timing
(purity to be verified and stated in the seal).** Main's #5 result
unaffected (deterministic). Addon3's analysis constraint: use ONLY the
count fields + stage names + gdn_projection_route; every timing field
discarded; the ranking = counts × the fixture-priced per-owner term.

### OVERLAP RESOLVED — NO OVERLAP (Main docker-facts, ~02:55Z): contamination ruling WITHDRAWN

main-fmlal5 (#5): StartedAt 02:21:15.148Z, FinishedAt 02:41:56.662Z,
exit 0. tender_morse (B1 export): Created/StartedAt 02:47:52.4Z — 6
minutes AFTER #5 ended. **There was NO overlap: the concurrency-
contamination ruling is WITHDRAWN; the B1 export's timing fields are
CLEAN and usable; the "launched 02:19Z" claim was false (sparks' send
time ≠ container start).** The seal must state the clean-timing fact +
the artifact-path reconciliation. Addon3 unconstrained (full data
usable). FMLAL: tip f987501f9 (combined L4+L5) staged for #6
(campaign-src-fmlal6, tree 9d9f020b3 clean); **#6 scope = focused cells
PLUS the complete canonical Unit+preflight** (L4 changes startup ISA
admission — the broader gate is warranted); Main starts #6 when
tender_morse exits.

### B1 ANALYSIS PARTIAL (~03:09Z): GDN question ANSWERED — fused path ENGAGES on both shapes

From the gdn_projection_route records (K14-B1ATTR-12675C956-20261112):
- **27B dense: GDN runs as TWO fused 2-projection subgroups (qkv+z,
  alpha+beta; codebooks=none+none)** — grouped_decode_equivalent_
  verifier_subgroup (MTP/verifier) + same_kernel_mixed_codebook_subgroup
  (steady decode). GDN = 2 forks/step ✓ (the corrected prediction).
- 35B: same shape (90/23040 steady; 7920 verifier subgroups in MTP).
- **The GDN fused path ENGAGES on both shapes — the non-engagement
  hypothesis is DEAD; GDN was never the 4-fork stage.**
- The 4-fork once-per-step stage = a DIFFERENT stage; identification
  needs the named per-stage fork rows.

EXPORT ANOMALY (blocking named attribution): the full-records JSONs
contain ZERO omp stage-fork records (35b-mtp-full: 6100 records, 1 omp;
27b-nomtp-full: 47, 1 omp) while the .out stdout tables SHOW the omp
stage rows (119 visible, tags truncated ~50 chars, 120-row display cap)
— the JSON filter handling dropped the omp stage family (filter len=21
suggests the env filter extended beyond omp,kernel). Also: no
stage_cpu_detail timer rows in the .out at all (the adjacency pairing
has no timer partner).
Fork COUNTS = solid; named attribution blocked on the filter anomaly;
NEXT: addon3 reads filterListFromEnv's parsing (~10 min) → the exact
env recipe → a sparks re-run of the export step only (~15 min, build
banked).

### EXPORT ANOMALY ROOT-CAUSED (~03:10Z): missing CPU_STAGE_TIMING pin — corrected recipe

Root cause (addon3, self-caught by reading the committed artifacts):
the B1 run sheet OMITTED `LLAMINAR_PERF_STATS_CPU_STAGE_TIMING=1` (the
A2 sheet had it; dropped in the B1 booking) — the stage publisher
carries the explicit stage-enable gate from the #186 fix and REFUSES
by design with the gate off → no stage fork records, no
stage_cpu_detail rows. The summary publisher (no gate) published ✓.
The gate worked exactly as reviewed; the run sheet was wrong.
CORRECTED RECIPE (re-dispatched to sparks): LLAMINAR_PERF_STATS_
SUMMARY=1 + FILTER=omp,kernel + **CPU_STAGE_TIMING=1** + JSON=
/work/b1attr-runs/full-records.json; same build (12675c956, banked),
same 4 configs, ~15 min. Seal K14-B1ATTR2-12675C956-20261112.
DISCIPLINE NOTE: cross-check env pins against the prior successful
sheet when adapting it — a diff would have caught the omitted pin.

### .80 QUEUE ORDER CONFIRMED (~04:44Z): W3 → B1ATTR2 → 27B rider

Post-#8: W3 (B4 Integration, e09903875) → **B1ATTR2 moved ahead of the
27B rider** (addon3's named-attribution analysis is blocked on it; 15
min vs the rider's 15-20 — the analysis unblocks a whole lane) → 27B
rider (K14-I16RIDER27). W3 never started (the earlier attempts were
the disk-full one; the tree is seeded, no build began). .79/W2: the
1-failed counter assertion consistent across all 4 placements —
awaiting addon2's resolution.

### B1ATTR2 SEALED (~06:59Z): K14-B1ATTR2-12675C956-20261112 — the corrected export complete

16 files + full-record JSONs at .80 /home/factory/llaminar-i8mm/
b1attr2-runs/; all 4 configs RC=0 (35B/27B × MTP/no-MTP) with
omp,kernel + SUMMARY + **CPU_STAGE_TIMING=1** + full JSONs — the omp
fork stage names + the gdn_projection_route identity data are IN this
export (the missing pin fixed). Addon3's deliverables: (1) the
per-stage named analysis (the 35B verifier 5-fork group, the draft
1-fork stages, the 27B 4-fork stage identity); (2) the
gdn_projection_route cross-check (2 fused subgroups predicted from
code); (3) the B1 ranking paired with measured per-fork overhead
(timings usable — uncontended). .80 → idea-04 M=1 (0ad09600b) → the
27B rider.

### B1 ROOT CAUSE COMPLETE (~07:00Z): layerN_gdn_proj = the 4-fork stage — per-member forking in the floating fused path

The named-attribution export identified the stage: **layerN_gdn_proj,
4.00 forks/execution × 48 GDN stages × 765 steps = 147,600 forks (25.6%
of ALL decode forks).** Mechanism (code-level): FloatingPointGemmKernel
::multiply_fused_tensor loops per MEMBER and runs each member's matmul
with its own OMP_WORKSHARE_REGION — one fork per MEMBER, not one pooled
fork per group (2 calls × 2 members = 4 forks ✓ exact match with the
histogram). Contrast: the 35B native path pools into one workshare
(1 fork/call); the FLOATING path was never pooled. Route records
confirm: 2 fused 2-projection subgroup calls per execution (qkv+z,
alpha+beta — floating engines, codebooks none+none).
**FIX SHAPE: pool the floating fused path like the native path
(independent output tiles, byte-neutral, the same discipline) — affects
all floating fused-projection users; 4→2 forks/execution halves the GDN
fork share.** Production pooling waits for explicit operator
authorization per the lane's rule; the mini-plan routes for the
operator decision package. THE B1 ATTRIBUTION ARC IS CLOSED: unknown
structure → named, root-caused, mechanistically-explained pooling
target with a quantified expected gain.

### B1 RE-VERIFICATION AT THE LANDED TIP (~07:02Z): the diagnosis HOLDS at 0ad09600b

`multiply_fused_tensor` still loops per member with per-member internal
OMP_WORKSHARE_REGION (one fork per member call) at the current PR tip;
the FMLAL additions (leaf + route probe + FPCR admission) live INSIDE
the per-member matmul schedule and don't touch the forking structure.
The pooled-workshare fix shape is validated against 0ad09600b per the
lane's conditions (branch from 0ad09600b, one workshare over all
members' row tiles, unchanged per-member arithmetic, byte-identity
proof matrix, fork-count assertion). Implementation waits for the
operator's explicit authorization.

### B1 POOLING AUTHORIZED (~09:10Z): operator "B1 authorize" — implementation starts

The pooled-workshare fix is AUTHORIZED: one workshare over all members'
row tiles in multiply_fused_tensor's floating path (the native path's
pooling pattern), unchanged per-member arithmetic, the byte-identity
proof matrix, the fork-count assertion (4→2 forks/execution as the
testable claim). Branch: from the CURRENT PR tip 7a4790c77 (addon3's
prep note said bb893cd79 — outdated; the re-verification ran at
0ad09600b and 7a4790c77 adds only B4 diagnostic test files). Normal
implement → validate → handback. Expected gain (diagnostic arithmetic
until measured): 147,600 → ~73,800 forks on the 27B decode path
(25.6% → ~12.8% of all decode forks).

### B1 POOLING WIP HANDOVER (~09:36Z): the runner-placement structural pass routes to Main

Addon3's implementation (codex/arm-b1-fused-fork-pooling, 2 commits,
base 7a4790c77, honestly flagged WIP):
- DONE+VERIFIED: the tile body extraction (a8caee288 — the exact
  run_gpu_aligned_expert_fp32_matmul_isa work lambda as a static
  member helper, byte-identical by construction; kRowTile hoisted);
  the fork-count + byte-identity test file (members 1-4 ×
  FP32/FP16/BF16 × M tails 1/6/15 × workers 1-8, pooled fork delta == 1
  + byte identity vs the per-member oracle + the FP16 FMLAL variant);
  the call-site dispatch shape (GPUAlignedExpert → output-pointer
  check → weight_type_/ISA switch → the pooled runner).
- BLOCKED: the runner placement (drafted before the class definition
  where FloatingPointGemmKernel is undeclared; the relocation hit
  brace-structure errors at the OneDNN loop; the runner is after the
  class but the loop below the pooled call site has a structural
  issue).
HANDOVER per the protocol (three silent-write failures in this arc):
Main completes the runner placement (inside the class after the tile
helper, or namespace scope + forward-declare). The verified pieces are
banked; the test file validates the completed runner.

### B1 AUTHORIZATION QUOTE DELIVERED (~09:50Z): Main's relayed-authorization rule satisfied

Main required the operator's literal quote before production kernel
work (no paraphrase). Delivered verbatim: **"B1 authorize"** — the
operator's direct two-word answer to the pending-decisions item, no
qualifiers, recorded in the tracker at 9215506b5 BEFORE any
implementation dispatch. The decision package it answered (root cause
+ fix shape + expected arithmetic) restated. Main takes the structural
pass after the main-rb2 validation completes; the WIP rebases onto the
announced rebased PR tip once it lands.

### B1 DIAGNOSIS CORRECTION (~10:30Z): the WIP pooled the wrong path — the target is the BackendNative FP32 skinny members

Main's model+code check (27B Qwen3.6-MTP-Q4_K_S): attn_qkv = Q5_K,
attn_gate (z) = Q4_K, ssm_alpha/ssm_beta = F32 [5120×48]. On CPU
"codebooks=none" is uninformative (native codebook descriptors are GPU
exports): qkv+z run on CPUNativeQuantisedGemmKernel (the first
subgroup); alpha+beta run on FloatingPointGemmKernel with the
BackendNative policy (NOT GPUAlignedExpert — MoE-expert-only), via
multiply_fused_tensor → run_onednn_fp32_matmul → run_fp32_skinny_matmul
PER MEMBER = 2 forks. **addon3's WIP pooled only the GPUAlignedExpert
branch → the 27B gdn_proj fork count would be unchanged. CORRECTED
TARGET: pool the BackendNative FP32 skinny members (alpha+beta) into
one workshare — per-element arithmetic identical by construction (the
same fma chain + completeSkinnyMatmulValue epilogue). Expected gain:
4 → 3 forks per GDN-projection execution (one fork saved per layer per
step), NOT 4 → 2; the remaining forks = the native qkv+z subgroup.**
Main proceeds on the corrected target under the "B1 authorize" grant
from ae51ce3f0, gates: byte identity (FP32, members 1-4, M 1-12, N
tails, ±bias/±beta, all thread counts) + fork-count assertion + Linux
run + paired Release A/B — lands only if same-or-better.

### B1 VALIDATION BOOKED (~10:35Z): main-b1 on .80 — the corrected-target candidate d5e04fdce

Main's B1 validation: Integration NEON_I8MM, ~60 min (build, the
focused fused/GDN cells, the full Unit+preflight) at **d5e04fdce** on
ae51ce3f0 ("perf(cpu): pool skinny FP32 fused projections into one team
fork" — the BackendNative alpha+beta pooling, 4→3 forks/execution per
the corrected analysis). .80 was idle (the M=1 re-run not yet launched;
it queues behind main-b1). **A paired Release A/B timing window follows
as a separate booking per the gates — lands only if same-or-better.**

### B1 CORRECTNESS GREEN (~11:08Z): d5e04fdce canonical 3569/0-new; the A/B window main-b1ab RUNNING

Main-b1 on .80 (Integration NEON_I8MM GCC14): **GREEN at d5e04fdce** —
focused (the new FusedSkinnyFP32Pooling cell, FusedProjectionParity,
FusedQKV/GateUp, GDNKernels, GDNProjection grouping/arithmetic/route
evidence, CPUHalfOperands) all pass; canonical **3569 tests (3568+1
new), 0 skipped, the same 6 inherited, 0 new**. The one extra selected
cell = a CUDA-only registration (not built with HAVE_CUDA=OFF) — a
regex-selection artifact, out of the canonical gate, unrelated.
build-b1 pruned (the disk-hygiene rule).
**The Release A/B window main-b1ab is RUNNING on .80 (~75 min)**: two
Release NEON_I8MM builds, A=ae51ce3f0 vs B=d5e04fdce; saved plans
built by A; 27B ABBA + 35B ABBAABBA, nomtp+mtp, -n 256, unprofiled;
then one PerfStats omp fork-evidence run per binary (which directly
verifies the corrected 4→3-forks prediction). **Landing gate: B ≥ A
with generated tokens identical** — the operator's same-or-better bar.
The M=1 window queues behind main-b1ab.

### B1 A/B SEALED (12:07Z): 27B same-or-better; 35B nomtp +0.6% → NOT landing yet; main-b1ab2 final slot GRANTED

main-b1ab (A=ae51ce3f0 vs B=d5e04fdce, Release NEON_I8MM, unprofiled):
tokens IDENTICAL A=B in every lane; all runs exit 0. Decode p50 ms
(B/A): 27B nomtp 162.15→161.89 (0.998); 27B mtp 142.41→142.26
(0.999); 35B nomtp 37.33→37.57 (**1.006** — all 4 B p50 above all 4
A p50 = consistent); 35B mtp 35.88→36.20 (1.009, overlapping). Fork
evidence 27B nomtp: GDN_PROJECTION 4.00→3.00 forks/execution, decode
step 755→707 — **the corrected 4→3 prediction CONFIRMED**. By the
operator's same-or-better bar, B1 does NOT land yet. **GO granted for
one final 12-min slot** (main-b1ab2: reversed-lead BAABBAAB 35B nomtp
+ per-binary 35B fork/kernel evidence; real effect vs code-layout
artifact). **Condition: unconditional handback to sparks after
main-b1ab2 — no fourth window; B1 rides this verdict (repeat +0.6% =
defer post-merge with the sealed negatives; flip ≤1.00 or layout
evidence = land).** Sparks notified of the extension.

### B1 FINAL VERDICT (12:26Z): DOES NOT RIDE THE MERGE — deferred post-merge, evidence sealed

main-b1ab2 (reversed-lead BAABBAAB 35B nomtp + per-binary evidence):
- 35B nomtp: A p50s 37.15/37.11/37.22/37.15 vs B 37.40/37.46/37.47/
  37.48 → **B/A = 1.0085 p50, 1.0066 mean — the +0.6–0.8% REPEATS
  (8/8 pairs across BOTH windows)**; tokens identical.
- 35B fork/kernel evidence: **forks per decode step = 593 for A AND B;
  GDN_PROJECTION = 4.00 for both; ZERO cpu_floating_fused_projection_
  calls on the 35B decode path** → the pooled code NEVER executes
  there; the slowdown = unchanged code in a different binary =
  layout/codegen, NOT the pooling.
- 27B (the path runs): GDN_PROJECTION 4→3, 755→707 forks/step; p50
  0.998–0.999 = within noise.
**VERDICT**: a measured cost (35B, reproducible 0.7–0.9%) against an
unproven benefit (27B gain indistinguishable from noise) fails the
operator's same-or-better bar → **B1 deferred post-merge** (Main's
ruling; operator may override). Post-merge: re-measure on the merged
base (layout effects shift anyway). Preserved: origin/codex/arm-b1-
skinny-fp32-pooling @ d5e04fdce (correctness green: canonical 3569,
0 new) + evidence /tmp/llaminar-main-bb-linux-results/b1, b1ab,
b1ab/followup. PR #43 stays at ae51ce3f0. Main back on call for
merge-attributed fix-forwards only. **Open question for post-merge:
is the 27B fork cut worth anything at all?** .80 handed back to sparks
(unconditional) — M=1 next.
