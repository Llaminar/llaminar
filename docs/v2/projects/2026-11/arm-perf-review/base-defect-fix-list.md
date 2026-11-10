# Base-defect fix list (consolidated; no public issues by direction)

Source: Bundle-1/15 unmasked classification + idea-20/31/32 findings; the
authoritative evidence = evidence/bundle1/{15-numa-less-host-classification.md,
classification-v2.csv, classify_v2.py}; DGX probes/unmasking logs. Owner =
base (Main) unless assigned otherwise. Each item = tracker-recorded, evidence-
bound; fixes land through the normal lane/Main flow (NO GitHub issues).

| Id | Defect | Severity | Evidence |
|----|--------|----------|----------|
| B1 | NUMAAllocator constexpr PAGE_SIZE=4096 -> 64K: madvise(DONTNEED) EINVAL/null; sub-64K range -> kernel rounds to whole 64K page -> SILENT ZEROING of neighbouring live heap | HIGH | unmask rerun log; 15 table |
| B2 | CPUBackend::allocate 4096 alignment + mbind(MPOL_BIND) -> EINVAL/null or neighbour page | HIGH | unmask rerun |
| B3 | DeviceRegistry registers CPU-less/zero-memory NUMA nodes (readdir order) -> cpus[0] capacity 0 on Grace | MED | probes_b1.txt |
| B4 | NUMATopology::detectLocalNUMANode silent node-0 fallback (maxim 29) | MED | G5; 15 |
| B5 | NUMATopology::detectViaProcfs substring cpulist parse + scans nodes 0-7 only (misses CPU 9) | MED | G7; probe |
| B6 | ExpertPreparedMemoryGeometry 4096 Heap min-align -> under-admission (PMA 'admitted=4096 requested=65536') | MED | 15 unmasked |
| B7 | MoE overlay planLayout literal 4096u first-touch granularity | LOW | 15 |
| B8 | Installed overlay table not bound to its build envelope (ISA = comment only; no runtime envelope dim) | MED | idea-20 residual R1 |
| B9 | Register-evidence classifier grammar: ldr x,mem,imm post-index unsupported (Q6_K S4 unclassifiable) | TOOLING | idea31 census |

Notes: G1/G2 masked by seccomp on some hosts (unmasking log = direct
reproduction); B8/B9 are work items for the respective owners (overlay table
owner / register-gate tooling). 64K-page hosts = Grace/Neoverse-V2 + any
future 64K ARM; x86 unaffected (hwloc path) for B5, 4K-normal for B1/B2/B6.

## Main triage (2026-11-10)

- GB10 + .79 = 4KiB (getconf 4096; 6.17.0-nvidia) -> B1/B2/B6/B7 latent here,
  REAL on 64KiB (Grace/Neoverse-V2) = ARM-port robustness; fixing on #43:
  ONE host page-size authority (cached sysconf, fatal if invalid/not power of
  two) replacing literals in NUMAAllocator (PAGE_SIZE, externalRangeAlignment),
  CPUBackend allocate/first-touch, ExpertPreparedMemoryGeometry, MoE fabric
  planLayout, route-exchange kPageBytes; regressions page-size-parameterized +
  device-free + source scan banning new hard-coded host-page literals there.
  LIMIT: B6 uses the PLANNING host page size; mixed-page-size clusters
  unsupported; admission rejects mismatch (not under-admit) - noted in plan.
- B5 (procfs substring parse + nodes 0-7 only): same series; exact range-list
  parse + full node enumeration. B3 (CPU-less/zero-memory nodes): after B5;
  skip as compute endpoints + Grace-shaped sysfs fixture. B4 (silent node-0
  fallback): decision after B3/B5; NUMA-less-unsupported => fail-fast likely,
  but Mac-VM behavior check first.
- B8 (overlay table envelope binding): LOW while installed table empty; next
  overlay-table change binds it (idea-20 residual).
- B9 (register-evidence classifier ldr post-index): OWNER = addon lane (R2).
- 64K unmasked window: DGX Grace (the only 64K host); booked via the DGX lane
  when the fix lands (Unit + page-size-parameterized regressions + B1/B2/B6/B7
  direct repro checks); .80 proves no 4KiB regression.

- B9 OWNER CONFIRMED (addon lane): separate follow-up AFTER the R2 handover
  (NOT inside R2 — 777f7a52a gates are running; tooling change would
  invalidate the tip review; nothing installed needs it — Q6_K S=2 classifies
  cleanly; forced-S4 probes only). Scope: admit post-index general-register
  load `ldr x,[xN],#imm` (analyzer register/footprint semantics) + positive/
  negative controls in the test shards + re-census at idea-31 S=4 showing Q6_K
  cells classify. Tracked in the R2 handover doc plus here.

## B-fix 64K window RESULT (2026-11-10, DGX Grace 0db136607)

- VERDICT: intended page-granularity fix WORKS on a real 64K host: 3/3 new
  preflight tests PASS (HostPageSize + Owners + LiteralPolicy; 6-site literal
  original failure confirmed); unmasked 52-set = 11 FLIPS / 0 new; 64K
  sub-page probe = 4K DONTNEED EINVAL (kernel rejects; canaries survive —
  no observed corruption on this kernel; pre-fix failures EINVAL-driven) +
  host-aligned 64K ret=0 (fix works).
- 3 REGRESSIONS (passing 2x pre-fix) = TEST EXPECTATION area (host-page
  rounding): V2_Unit_MoEOverlayDeviceControllerTopology + V2_Integration_
  CPUExpertShardExecution_PreparedNativeSourceAllFormatsAndSerialRowsReplay +
  V2_Integration_CPUShardPreparedPackedRecords (:258) — recommend updating
  those expectations to host-page rounding before merge (test-only; fix
  correct). Owner: Main/base suite.
- NOTE: page-owner tests on Docker need --cap-add SYS_NICE + seccomp=
  unconfined (get_mempolicy EPERM otherwise) — separate from the page fix;
  record in test/CI docs.

## B-series DONE on #43 — tip 9490b66bc (2026-11-10)
- 6e049131b: the 3 64K test-expectation fixes (fabric page = hostPageBytes();
  expert geometry = page-aligned slot bound; bound>=actual &&
  bound-actual < one host page).
- 4bf9ae83f B5: exact cpulist parse over every nodeN (both substring
  directions + sparse/duplicate/malformed).
- 524afd11e B3: DeviceRegistry = NodeTopology (CPU-bearing nodes only,
  ascending) + Grace-shaped fixture; registry == NodeTopology on host.
- 9490b66bc B4 (per G5): all planner/backend consumers already reject failed
  detection; the ONE real ignore = RuntimeInitPhase DeviceManager::initialize
  (took node0 as socket-local view) -> failed detection now publishes node -1,
  method "undetected" (host-wide view); no top-level fail-fast.
- All four in ProductionTestPreflight; Mac OK except known no-NUMA-sysfs
  env class. Open: B8 (idea-20 residual), B9 (addon R2). Grace re-run of the
  3 flipped tests at 9490b66bc = requested (would close B1-B7 there). Test
  docs: page-owner tests in Docker need SYS_NICE + seccomp=unconfined.

## B1-B7 CLOSED on real 64K — 9490b66bc EOL (2026-11-10)
- Focused unmasked run 17/21: 3 regressions fixed; HostPage* 3/3; B3/B4 cells
  4/4; 11-flip sample 5/5. 4 FAILs = ONE env class (memory-only NUMA nodes;
  node1 255GB/0CPU, 2-8 zero capacity), test-side, no page/affinity
  regression: NUMATopologyCPUList(+Membership)
  RegistryCPUEndpointsFollowNodeTopology (registry CPU-bearing-only vs
  NodeTopology incl. memory-only node1); V2_Unit_DeviceRegistry (same
  interplay); ClusterInventoryBootstrap (cpu:0 ambiguous; specify NUMA node).
- RECOMMENDED (test-side): align those cells with B3 semantics (filter
  memory-only nodes / compare CPU-bearing sets); no engine change.
  Memory-only-node Delta = next host-qual doc item (15 env classes).

## B10 PINNED + FIXED (2026-11-10) — ce9ddb326 (#43, 157 commits)
- ROOT CAUSE: Main's own bug in 524afd11e —
  `for (const auto &node : NodeTopology::detect().numaNodes())` iterates a
  reference into a TEMPORARY NodeTopology that C++20 range-for destroys
  before the loop body -> numa_node read from FREED memory (use-after-free)
  = the run-variant garbage (5 runs, 5 values). Registry test had the same
  pattern (Mac passed only because freed memory still held old values).
- FIX: registry binds the topology to a local; numaNodes()/interSocketLinks()
  REF-QUALIFIED (lvalue lends vector; temporary returns owned copy) ->
  dangling form unrepresentable; static asserts + temporary-iteration
  regression in the preflight membership cell. Mac build + cells pass.
- Counter-evidence story: (a) false (node1-8 cpulist EMPTY), (c) N/A (sysfs
  used); size/ABI/objdump probes clean; clean-rebuild reproduced new garbage
  -> real defect, NOT a test-semantics mismatch; tests stay as-is.
- Grace close-out at ce9ddb326 in flight (NUMATopologyCPUList* + DeviceRegistry
  + membership cells). Memory-only-node Delta = B10 (solved) + ClusterInventory
  Bootstrap (known env) + expectations correct.

## B10 CLOSED on real 64K (2026-11-10) — ce9ddb326 Grace close-out
- Focused unmasked 9/9 PASS (NUMATopologyCPUList(+Membership), DeviceRegistry,
  NUMATopology, NUMATopologySysfs, DeviceRegistryStartupPolicy, NodeTopology
  (+DensePackageIds+fixture)) — garbage values GONE; the ref-qualified
  numaNodes() lvalue/temporary contract fixed the use-after-free. B1-B7+B10
  COMPLETE on Grace. Evidence: build_v2_dgx/b-fix-ce9ddb326/. Sanitizer
  stopped (valgrind 3.23 build aborted; not needed).

## Cross-program classification (2026-11-10): station failures from the
## continuous-batching slice qualification (28ddbe253 base, NEON_I8MM
## Integration, canonical driver, exit 8; sets IDENTICAL pin-vs-slice, 140
## signatures identical, zero new)

Classification vs the ARM station baseline (39c24d8b4 classes): the
overwhelming majority are IN-CLASS — documented station env classes (64K
pages, memory-only NUMA nodes 1-8, MPI hwloc bind/abort), fixture-missing
(ModelLoaderRowSlice qwen2.5-0.5b GGUF; ROCm trainer CSV ×4; ModelKLDCanary),
and known-base classes (ROCm registration, GroupedSealedPlan,
MoEForbiddenDependencyScan, CommonDispatchPolicy, FusedFinalAdd,
registration/selection family). Strong non-ISA-specificity signal:
CPUShardFusedFinalAddPreweight fails IDENTICALLY on scalar, neon AND
neon_i8mm — common cause, not an ARM-kernel bug. The bulk
qwen4exp/immutable-lookup gtest-assert family is consistent with the
memory-only-node topology infeasibility class (placement semantics
untestable where nodes 1-8 have no CPUs) — matches the documented
"memory-only nodes" env class; bulk, not individually enumerated at
39c24d8b4.

TRIAGE CANDIDATES (entered, not confirmed):
- **B11-candidate: V2_Unit_NUMAAllocator SIGSEGV** (mpirun "process rank 0
  exited on signal 11") under the memory-only-node topology, AT 28ddbe253 —
  i.e. AFTER B1-B4 fixes. Either an incomplete B-fix for this topology
  corner or mpirun/hwloc env interference; needs a non-MPI reproduction to
  attribute. Evidence: station artifacts
  /home/tdsynnex/repo/llaminar-cb-c1/artifacts/cb-c1-staging/
  {pin-qualification,slice-qualification}/ (worker:
  subagent-chat-01a1c527-8869-727c@dgx-station; found by the
  continuous-batching lane's qualification — operator-initiated cross-check).
- Mild unclassified-vs-39c24d8b4 items to verify against the ARM station
  run identities: V2_Unit_NativeSourceFinalRetirement [assert,env-numa];
  V2_Unit_ProductionParityCampaigns [gtest-assert];
  V2_Integration_DevelopPRGateContract [gtest-assert]. Possibly in-class
  (env/registration families); verify before treating as defects.

Method note (batch-inference lane, accepted): identities+signatures
compared, not counts; unchanged failures do not prove non-ARM causality;
outside-class = triage candidate, not automatically an ARM regression; no
qualification transfer from historical B10 receipts (different rewritten
ancestry).

### Refinements from the lane's independent artifact inspection (2026-11-10)

The batch-inference coordinator independently copied + hashed the station
artifacts (9/9 SHA256s match; JUnit 3408→3410, failed sets identical by
lane) and extracted concrete diagnostics that REFINE the classification:

1. **ModelKLDCanary is NOT fixture-missing-GGUF** (correcting my entry): it
   fails on a MISSING DOCKER EXECUTABLE for a subprocess + installed-receipt
   build/inventory mismatch. Class: environment/tooling.
2. **GroupedSealedPlan = a PYTHON TEST-HARNESS defect**: 10 errors,
   `SimpleNamespace lacks physical_launch_evidence` at
   adaptive_tournament.py:188 — the tournament test harness expects an
   attribute the production/publication path may have changed. **Possibly
   ARM-STACK-INDUCED staleness** (the panel-tournament publication contract
   evolved in #43) → triage candidate for the addon lane.
3. **CPUShardRouteCollective/FusedFinalAdd** (:336): fails because
   `preweightMoEOverlayCanonicalRouteRows` returns FALSE (workers=1,
   columns=1, degree=2) — a POLICY REJECTION of the degenerate config, NOT
   a byte mismatch. Consistent across scalar/neon/neon_i8mm (common cause).
4. **NUMAAllocator B11 sequence**: logs show EPERM querying memory policy →
   null allocations → SIGSEGV during NUMABufferIterators. Hypothesis:
   unhandled EPERM (memory-policy query blocked, cf. the G1/G2 seccomp
   masking evidence) → null-propagation → deref. The non-MPI repro remains
   the discriminator; the null-deref path may be a REAL base defect
   (error-path hardening), not env.
5. CAVEATS: "zero skips" is CTest-level only — embedded GTest skips exist
   (NUMAFirstTouchPolicyIsolation, model-loader cases, CUDA PendingProducer,
   ISA-specific tests); some system-out truncated at 1024 bytes — do NOT
   claim all native cases executed or full-signature equality from those
   excerpts.

### GroupedSealedPlan candidate CLOSED (addon verdict, 2026-11-10): standing base failure, NOT ARM-induced

The "ARM-induced harness staleness" hypothesis is REFUTED:
- The failure is IDENTICAL at the R1 base 425002104 (r1e.unit.log: the same
  10 `AttributeError: 'types.SimpleNamespace' object has no attribute
  'physical_launch_evidence'` at adaptive_tournament.py:188; g-8fcc0fe0b
  has exactly 10 too).
- Authorship: adaptive_tournament.py + the SimpleNamespace _observation()
  stub = a0fd16d04 (2026-11-09), part of #43's new base — not the addon
  diff. git diff 425002104..28ddbe253 over adaptive_tournament.py /
  candidate_observation.py / exact_oracle.py / the grouped-sealed-plan test
  = EMPTY (nearby 83f82897a only refactored candidate_observation.py's
  core-count helper into host_cores).
- Mechanism: the stub omits `physical_launch_evidence`, an OPTIONAL
  observation column since 77e17645b (2026-10-22); is_promotion_grade()
  accesses it UNCONDITIONALLY → AttributeError. Already on the R1
  known-failure list; V2_Unit_CPUNativeQuantisedGroupedSealedPlan = the
  standing base failure in every gate (13/987 at 28ddbe253).
- The tournament work touched only Perf__CPUNativeNEONPanelTournament.cpp
  (separate binary) — never the dispatch corpus schema or is_promotion_grade.
- BONUS: this also explains the earlier "R2-2 partial" gate error at
  623e44ccf — same 10 errors, same stub, different path.
- OWNER: Main (one-line stub repair: getattr(observation,
  'physical_launch_evidence', {})-class or an explicit stub field), outside
  R2. Classification: in-class standing base failure.

### B11 CANDIDATE CLOSED — env class confirmed (arm-sparks non-MPI repro, 2026-11-10, ~18:43Z)

v2_test_numa_allocator run DIRECTLY (no mpirun/hwloc launcher; .79 GB10,
e8866ec04 Integration build, same container class): **38/39 PASSED, 0
failures, RC=0** (58 ms). The 1 SKIPPED = FirstTouchOverridesInherited-
RemoteMemoryPolicy (skip guard = the inherited-policy precondition, not a
fault). NO segfault without mpirun → the SIGSEGV seen in the
continuous-batching lane's canonical driver run = the **mpirun/hwloc env
class**, NOT a real allocator defect on the memory-only-node topology.
B11-candidate CLOSED (no tracker upgrade, no Main fix-back). Evidence:
.79 /home/factory/idea25/b11-nompi.log (sealed). Cross-program note: this
CONFIRMS the in-class classification given to the continuous-batching lane
(mpi-hwloc env).

### METHODOLOGY CORRECTIONS ACCEPTED (continuous-batching lane, 2026-11-10): three downgrades

1. **Classification ≠ green.** Paired failure parity + classified residuals
   is baseline-relative evidence, NOT passing canonical gates; no waiver
   authorized. Classification routes fixes/setup only. The ARM program's own
   station runs recorded UNQUALIFIED-with-classification — never green.
   (The SYS_NICE/seccomp=unconfined unmasked windows DID turn page-owner
   tests green — SOME "env" classes are SETUP-fixable, not environmental.)
2. **B11 downgraded to NON-REPRODUCTION on the .79 configuration** — not
   proof of cause. The original failure host was the DGX STATION; the
   non-MPI repro ran on .79 (different Spark host, different container).
   Controlled comparison required: the SAME non-MPI run ON THE STATION with
   matched host/source/binary/container/NUMA/policy identities, the one
   non-passing case in 38/39 identified, and both failure contexts
   documented. Route: the continuous-batching worker (matched host) or the
   64K validation owner to execute the station-side repro; until then B11 =
   open candidate with per-side identities recorded, prior evidence remains
   EPERM→null→SIGSEGV (unattributed).
3. **No blanket environment exemption for the 140.** Per-failure/per-class
   evidence + owners instead. Class decomposition to bound:
   (a) 64K-page mmap assumptions — possibly PAGE-SIZE-TEST-DEFECTS
   (parameterizable), not env; (b) stale fixtures/schema (ROCm trainer CSVs
   = fixture-missing; ModelKLDCanary = docker-missing setup); (c)
   registration/selection family — possibly real registration defects;
   (d) memory-only-NUMA placement semantics — topology-infeasible on this
   host but parameterizable; (e) MPI hwloc bind — SETUP-class candidate.
   Each class: evidence, owner, fixable-vs-not verdict.
