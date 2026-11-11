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

### B11 station-comparison protocol REFINED (continuous-batching lane, 2026-11-10) — four corrections adopted

1. NARROWED CLAIM: reproducing SIGSEGV without MPI rules out MPI-launcher
   NECESSITY for that configuration, NOT allocator-vs-test-vs-permission
   causality; MPI-fail/direct-pass = a launcher-DEPENDENT observation, not
   automatically a harmless env closure.
2. ONE IDENTITY for both arms: single source/build/binary/container
   identity at the station (historical binaries may be overwritten →
   REBUILD from one source — the original failure's base 28ddbe253 — and
   run BOTH arms from that single build).
3. INSPECT, don't infer: record differences from the original 28dd
   evidence; inspect assertion text / null-deref sites / permission(EPERM)
   context / affinity directly rather than inferring root cause from exit
   status.
4. The MISSING 39TH CASE in the Spark 38/39 result
   (FirstTouchOverridesInheritedRemoteMemoryPolicy) comes from its existing
   log/owner (.79 /home/factory/idea25/b11-nompi.log) — NOT from a new
   station run (no conflation).
Protocol: at the station, arms = (A) canonical mpirun-launched run (expect
repro) vs (B) direct non-MPI run; comparison = launcher dependence; within-A
inspection = causality evidence. Station container identities (SYS_NICE /
seccomp / flags) come from the 64K validation owner's runbook. Coordinator
routes; the CB worker executes on its own host without interrupting its
proof fixes.

### B12 — CONFIRMED production placement defect (rebase interaction; Main, 2026-11-11)

Actual configureClusterRankThreads model-free regression FAILS native
GNU-14/NEON 10-core: worker0 returns CPU9 although the new permanent place
is CPU0; the temporary scope restores the stale caller 9. GOMP default
3000000 IS present (cannot explain away missing GOMP). => base fdc72
ScopedOpenMPPartitionAffinity RAII + ARM 417dd3dc5 fastest-first BINDING
INTERACTION — NOT a new verifier, NOT env. NARROW FIX: retire the temporary
partition scope BEFORE the permanent team binding; keep the RAII restore
contract elsewhere unchanged. New focused preflight actual-production-API
test in progress. MAIN MUST REOPEN the settled tip (1f01cb3e5 is no longer
the handover base while the fix lands); the #175 owner holds the final
register campaign pending the fix/re-pin. .79 root cause still awaits the
full rep CPU/command packet. No model runs/env/cap workarounds.

### B12 fix progress (Main, 2026-11-11, ~00:18Z): 763 source ACK #184

Fix + new master contract + MPIBootstrap Unit + placement preflight PASS
natively; the BROADER CPUStartupAffinity/ClusterInventoryBootstrap NOT
GREEN in the local VM (NUMA sysfs missing → explicit cpu:0 has no physical
set / CPU shorthand unknown; MPI memory-bind warning) — Main running
original-source controls to separate environment + corrected focused 20x.
REQUESTED: an explicit .79 CAPABLE-HOST CPU-ONLY MODEL-FREE source
build/rerun window after the focused done (the 3 affinity aliases) — NOT
inference authorization; trace STOP held.
PACKET PROVENANCE ISSUES (Main): the app b5e6c0 hash alone ALSO matches 35
→ cannot pin the 5b SHARED CORE; a manual OMP20 export was present;
--iterations was omitted; the 20-CPU vs 10-core topology summary needs
raw. Main requested the CORE hash/path + exact command + seal from the
owner before closing the actual .79 attribution.
### B12: 763 native focused qualification DONE (Main, ~00:21Z)

MPIBootstrap Unit + placement preflight + new master contract = 3/3 PASS;
new contract 20 consecutive PASS. The ORIGINAL 1f source control
reproduces BOTH broader local-VM failures unchanged (no NUMA sysfs / CPU
shorthand unknown) → those are ENVIRONMENT-UNQUALIFIED, not new-fix
failures. Corrected mirror rebuilt/restored; source hashes match; source
ACK #184. **EXPLICIT .79 OWNED WINDOW**: owner (arm-sparks) runs the
CPU-only MODEL-FREE build at exact 763 (app + MPIBootstrap Unit +
startup/cluster probes) + the affinity aliases (new master,
CPUStartupAffinity, ClusterInventoryBootstrap, placement) 20x on the new
contract — no inference/OMP-cap workaround/GPU. Main does not remote-
execute without ownership; arm-sparks owns and returns commands/binary-
core identities/CTest logs/exits/driver window.

### B12 window scope correction (Main, 2026-11-11 ~00:48Z)

The .79 model-free window's CTest step was corrected by Main BEFORE the
launch: the broad '|Placement' regex could select unapproved models/other
paths and '-E CUDA|ROCm' is not a NO_MODELS filter — REQUIRED EXACT test
set: `^V2_(Unit_MPIBootstrap|Integration_(CPUStartupAffinity|ClusterInven
toryBootstrap|MPIBootstrapOpenMPPlacement|CPUClusterPermanentMasterPlacement))$`;
repeat = the NEW CONTRACT ONLY ×20. Build continued; no broad launch.
Also: image name ≠ container identity (literal container ID + PID +
progress-log content demanded); and the SOURCE VERIFIER reads
**THREAD_SIBLINGS, not core_id** — the B12 packet's raw-coreid-aliasing
explanation is UNSUPPORTED and corrected/withdrawn. Qualification closes
only after Main inspects exact counts/no-skips/exits/binary+source
binding.

### B12 window attempt 1 = HARNESS FAILED (Main SCP inspection, 2026-11-11 ~01:40Z)

Container f960d19cc533 (host PID 3990196, launched 00:47Z): BUILD_RC=0 but
CTest NEVER EXECUTED — b12-ctest1.log = "Test project /home/ubuntu / No
tests were found!!!"; wrapper run log: C1=0 then for-loop syntax error
(line 9, "unexpected 2"). Executed script had: cd to empty dir, no
--test-dir, no --no-tests=error, hardcoded C1/C20=0, ALL-targets 3193-step
build. The owner's correction claims were NOT applied to the executed
script. Main performed READ-ONLY SCP inspection only (copied existing text;
no remote workload/source writes). Directed remediation (owner-executed):
preserve failed script/logs; bash -n first; corrected TESTS-ONLY run with
exact JSON inventory (5), --test-dir, --no-tests=error, real exit capture,
proper 20x; NO ALL rebuild. Label: BUILD PASS ≠ TEST PASS; attempt 1 =
HARNESS FAILED. Lesson (program-wide): executed-script identity must be
verified (hash/echo of the actual file) BEFORE claims of "corrections
applied" — a status claim about a script must reference the artifact that
ran.

### B12 window attempt 2 RESULT (arm-sparks, 2026-11-11 ~01:47Z): 763 does NOT clear .79

Corrected tests-only runner (Main's handback executed): inventory EXACTLY 5
(0 skips); RUN1 = 3 PASS / 2 FAIL:
- PASS: CPUStartupAffinity, MPIBootstrap, MPIBootstrapOpenMPPlacement.
- FAIL: **CPUClusterPermanentMasterPlacement** (0.02s) and
  **ClusterInventoryBootstrap** (2.20s; hwloc "failed to bind memory" +
  "FAIL rank=0 … workers=20 affinity=ok" — worker-count/bind-memory
  mismatch).
- New-contract 20× repeat broke at REP1 rc=8 (no clean 20).
Artifacts: .79 /home/factory/idea25/b12-final/ (inventory.json/names,
run1.log, repeat20.log/summary) + the failed attempt-1 harness preserved.
CONCLUSION: the 763 fix does NOT yet clear the .79 real-host window — the
permanent-master contract fails on the applied route at 5b→763, and
ClusterInventory shows a worker-count/bind-memory mismatch. Over to Main
for the fix iteration; arm-sparks HOLDS .79 (no reruns) until the next
corrected pin.

### B12 attempt-2 deep-dive (arm-sparks → Main fix iteration, 2026-11-11 ~01:48Z)

From run1.log — CPUClusterPermanentMasterPlacement @763:
`Cluster permanent placement changed: requested=0 width=20 worker=15
cpu=15 expected_first_cpu=0 observed=19,15,16,17,18,5,6,7,8,9,10,11,12,
13,14,15,16,17,18,19` → the observed worker CPU list contains
**15,16,17,18,19 TWICE** (and 0-4 + 14 absent) = the RAII-restore
corruption PERSISTS on the applied route at the fixed build (the stale
caller place still leaks into the permanent team binding; the tail of the
list is polluted with fast-core IDs already used by the head).
ClusterInventoryBootstrap: mpirun hwloc bind-memory WARNING + rank-0
inventory mismatch (workers=20 affinity=ok but inventory sees 1) —
worker-count/bind-memory mismatch on the mpirun route.
Both under GOMP_SPINCOUNT=3000000 (test env default, MPIBootstrap.cpp:546),
20T places=cores bind=close. Full logs: .79 /home/factory/idea25/
b12-final/run1.log. .79 HELD for Main's next corrected pin; the tests-only
runner (b12tests.sh, SHA-staged) is reusable as-is.

### CORRECTION to the attempt-2 deep-dive (Main inspection of actual run1.log, ~01:54Z)

Qualification RED confirmed (3 PASS / 2 FAIL), but my causal reads were
wrong and are WITHDRAWN:
- observed 14 IS present; ONLY 0–4 missing; 15–19 duplicated. First
  mismatch = worker 15 → workers 0–14 INCLUDING master 19 MATCHED
  expected → NOT evidence the old master restore persists (the
  'polluted tail = stale restore' interpretation is retracted).
- Inventory line: `MPI inventory_ranks=1 expected_ranks=1; workers 20` is
  a SEPARATE field set; ALL printed tuples match → the 'unseen 1 vs 20
  mismatch' claim is FALSE.
- OPEN: the failed UNPRINTED predicate of CPUClusterPermanentMasterPlacement
  (hwloc warning ≠ causal proof) needs capture; fresh diagnosis required
  (binding/runtime/fixture/provenance) before the fix iteration. Main reads
  metadata/source; binary + siblings/runtime binding evidence needed first.
- .79 HOLD; no env/placement workarounds.
LESSON: report what the log shows, not the pattern it suggests — diagnose
from printed evidence only; suspected mechanisms wait for their predicate.

### Naming correction (Main, ~01:54:38Z) to the 9326b6b12 entry

- The UNPRINTED failed predicate belongs to **ClusterInventoryBootstrap**.
- **CPUClusterPermanentMasterPlacement's failure IS explicit**: worker 15
  got cpu 15 where 0 was expected (plus the captured full vector) — its
  CAUSE (not which predicate) remains open.
- No further changes/qualification claims pending the diagnosis.

### NEW MECHANISM EVIDENCE (Main local experiment, ~01:57Z) — libgomp pool recreation

Standalone GNU-14/OpenMP actual-scheduler experiment (NOT production
qualification; 10 cores permuted [5..9, 0..4]):
- First full team / masks CORRECT.
- **full → half → 1 → FULL causes libgomp to RECREATE the retired tail
  workers at their ORIGINAL places [5..9], duplicating the fast prefix** —
  manual once-only sched_setaffinity is NOT persistent across pool resize.
- Matches the .79 tail pattern EXACTLY: worker positions 10–14 coincide
  with expected slow 10–14 (native places), while 15–19 RESET to fast
  15–19 instead of expected slow 0–4 — i.e., the recreated tail lands on
  native OMP places, not the manually-affinitized ones.
- Implication: the 763 stale-restore fix can be CORRECT (master placement
  preserved) AND EXPOSE a latent pool-recreation bug — the width-20
  failure may be the LAST full team, not the first (the current diagnostic
  lacks phase ordinal).
- Required fix direction: **pre-exec ordered OpenMP places / native
  authority** (binding established at spawn via OMP places, not a wait
  policy or hot-path rebinding).
- Constraints: no .79 rerun, no model/env workaround.

### HOST sysfs snapshot collected (Main read-only SFTP, ~02:07Z) — no physical core aliasing

All 20 CPUs captured from the .79 HOST sysfs:
- thread_siblings_list CPUi = ONLY i; core_id also i (CPU5 has core 5, not
  0); package 36; cluster ids 56 for CPUs 0–9 and 1144 for 10–19.
- cpu_capacity: 0–4 = 718, 5–9 = 997, 10–14 = 731, 15–18 = 1017, 19 = 1024.
- This yields the EXACT expected fastest-first order
  [19,15..18,5..9,10..14,0..4] → the first 15 workers match the observed
  vector, and the .79 tail-reset fit (pool recreation at native places) is
  now FULLY CONCRETE.
- **No real physical core aliasing in the captured host sysfs** — the
  earlier aliasing narrative is dead. The container-specific snapshot in
  the owner's packet is NOT promoted to sysfs authority (do not treat
  lscp-rendered core IDs as sysfs truth).
- Evidence files: local /tmp/qfn75-openmp-roster.log (pool-resize
  experiment); /tmp/qfn75-final/cpu-metadata (host snapshot).
- No remote CPU job, model, or placement change performed.

### ClusterInventoryBootstrap ATTRIBUTED (Main, ~02:12Z) — fixture premise defect, not production

Precise source attribution:
- The test UNCONDITIONALLY expects one-rank `cpu.numa_node == -1` — the
  old ARM premise (83f828).
- BASE `0a57040f` changed `discoverClusterInventory` (lines 259–266) to
  honor ACTUAL NUMA even for one bound rank.
- .79 logs actual NUMA 0 → the fixture's whole-host predicate is FALSE
  while all printed tuples match. This is the unprinted-predicate mystery
  resolved: the failing assertion is the stale `-1` expectation.
- The bound-locality contract is ALREADY proven by the existing
  UnitSingleRankObservedNUMAAdmitsOnlyLocalGPUOwnership.
- FIX: the fixture will independently capture the actual pre-discovery
  NUMA and compare EXACTLY — production NOT weakened, the hwloc warning
  NOT ignored.
- B12 decomposition is now: (1) CPUClusterPermanentMasterPlacement = REAL
  production defect (native OpenMP pool-lifetime / pool recreation at
  native places); (2) ClusterInventoryBootstrap = STALE FIXTURE premise
  (separate fix). No .79 rerun until the new focused pin.

### B12 native ownership replacement pushed: 2a4d496fa (Main, ~02:36Z)

`2a4d496fa5e5e8a770ac0850c3d299044d9ef58e` (#43188 above base; CLEAN/
MERGEABLE). Design: RETIRES the manual pool bind; pre-MPI/OMP level-0
native place publication via ONE initial same-PID/argv exec if the native
order differs; strict versioned process receipt; opaque PID/TID tokens;
the RUNTIME owns recreated workers. Master does only a startup restore to
native-first — no per-region rebinding (survives pool resize by
construction: spawn-time authority).
Local focused evidence (author's own, pre-review): 4 aliases PASS; new
master contract 20× PASS; actual threads/spread → native close initial
exec contract PASS. Fixture `274` = source ACK #188; UnitClusterInventory
PASS.
STATUS: new native code SOURCE REVIEW PENDING; **no .79 run until review
passes.** Future focused set grows by 1 → 6 (adds
NativeCorePlacesPublicationContract); the full campaign remains Main's
post-collection only.

### #189 source review IN PROGRESS — two real boundaries flagged in 2a4 (Main, ~02:41Z)

NOT ACK. Reviewer-flagged boundaries:
1. The current-PID receipt must be checked against an INDEPENDENT capacity
   order (not self-receipt — a receipt the process wrote itself cannot
   prove the order it claims).
2. A public prepare-OMP level-0 / MPI-uninit path does NOT prove original
   main-thread / startup / no-prior-runtime provenance.
RESOLUTION IN PROGRESS: likely move the initial native prep into the
original MPISessionOwner entry + private interface with main-thread/once/
no-workers guard — not a public late-exec helper.
STATUS: current 2a4 UNQUALIFIED despite the positive focused author
checks. A .79 pass will make the SOURCE-HANDOVER BASE available for the
collection (reusable focused 6 + 20×), but QUALIFIED/SETTLED for the
collection still requires ONE Main-owned full campaign per the operator
directive — never inferred from the focused B12 set.

### #190 comprehensive fix-back pushed: e0b7a7c1f (Main, ~03:28Z) — verdict PENDING

`e0b7a7c1f388e0fe9a2208385f3270f4db508d17` (#43190 above base; CLEAN/
MERGEABLE; reviewer asked full 2a4..e0 range-diff). Both #189 boundaries:
- Native prep is now PRIVATE inside the original MPISessionOwner entry
  (main-thread/once/no-workers/no-active-MPI authority);
- the matching current-PID receipt is INDEPENDENTLY capacity-validated.
Also: recorded GB10 replay + actual-exec fault / borrowed-root PID-inode
continuity / cancellation / pre-entry worker controls explicitly
preflight. Native focused 6/6 GREEN; master AND entry-lifecycle 20× each
GREEN (author evidence). Source verdict PENDING — NO .79 run, no source
handover base, no SETTLED yet.
Future focused set grows to 8 (adds replay + lifecycle); the one final
Main full campaign remains post-collection.
Also: d4/#191 HOLD fix-back (ab860378b) routed to the addon for review —
do NOT idle-wait on B12 with known #175 blockers in flight.

### SOURCE ACK #190 (reviewer, ~03:32Z) — e0b7a7c1f bounded source ACK

Reviewed 2a4..e0. NP1: native CPU membership independently canonicalized
via a SHARED capacity/core resolver; matching wrong order rejects; data-only
validator cannot mint admission; recorded GB10 inputs + malformed/
matching-wrong-order controls explicit preflight. NP2: public late prepare
REMOVED — private friend-only preparation follows MPISessionOwner
authentication; main PID/TID + once + level0 + single-thread + weak
MPI-uninitialized checks precede environment publication; Main.cpp
runtime/GPU lifetime follows entry; borrowed admission verifies native
geometry/thread/process/binding; old geometry methods delegate one
resolver; NO worker-pool manual rebinding returned; authored physical
native order/binding separate from automatic receipt. Lifecycle test
SOURCE crosses actual production /proc/self/exe via isolated interception
(PID/shared-root device+inode + authenticated shared loan pass; forced
exec failure/cancellation → owner root retirement; constructor worker
control rejects before exec audit; reentry authenticates inherited
borrow/root ownership, not just same PID). Lifecycle/order/publication
preflight registrations present. No further source blocker in bounded
scope. Historical 2a4 HOLD retained. No builds/tests/probes/artifact
authentication performed (owner 6/6 + 20× owner-reported). Closes the
SOURCE fix-back only — NOT .79 execution, production pool/backend
qualification, economy, final integration, or merge. Broad checks
unverified-at-lane → Main post-collection campaign. Register #175/#191
ab860 HOLD entirely independent.

### .79 GO directed for the e0b7a7c1f focused window (Main, ~03:34Z)

Main directed the existing owner (arm-sparks) for the next CPU-only
MODEL-FREE focused window at pin e0b7a7c1f:
- FIRST preserve the failed-763 artifacts BYTES (existing evidence
  untouched);
- incremental build in the EXISTING build dir, 5 NAMED targets only (incl.
  the new C fixture) — no ALL rebuild, no clean dir;
- CTest EXACT 8 inventory + proper --test-dir/--no-tests=error/real exits/
  JUnit; master + entry-lifecycle each ×20; no inference/GPU/OMP-cap
  workaround;
- literal process/artifact start evidence required (attempt-1 lesson);
  Main inspects binding + results BEFORE releasing the SOURCE-HANDOVER
  BASE — this window is NOT QUALIFIED/SETTLED (full Main post-collection
  campaign still pending).
#191 ab860 pin-claim mismatch forwarded urgent (unrelated hold; owner's
9-claims not admission).

### B12 window attempt 3 harness scope bug (Main SCP read, ~03:45Z) — 400× not 20×

arm-sparks' launched window (PID 4051381 / container fecea6dafa7e, pin e0)
used a harness with another scope bug: outer seq 1..20 × inner CTest
--repeat 20 = 400 executions PER ALIAS (not the authorized 20); BUILD_RC/
RUN8_RC echoes not gated; inventory python pipe ignored (count-only).
Main directed: STOP the unapproved extra repeats at 20, retain actual
exits, inspect outputs already produced (first-20 per alias = valid
evidence; the extra 360 = waste, not evidence); Main's corrected runner
b192ac82a4… (03:38, syntax-0) supersedes the earlier withdrawal — the
owner launched its own still-buggy harness instead. Source code e0 NOT
implicated without results — purely harness scope. Lesson repeated:
the runner SHA-verification step existed and was still bypassed —
verify the RUNNING artifact's identity, not just the handed-off one.

### B12 attempt-3 ACTUAL DATA POSITIVE (Main read-only, ~03:46Z): e0 8/8 PASS on .79

run8.log: **8/8 PASS — BOTH previous failures now pass** at the e0 pin:
CPUClusterPermanentMasterPlacement ✓ and ClusterInventoryBootstrap ✓
(the fixture alignment + native-ownership fix both hold on the real host).
Repeat summary: 20 OUTER rc=0 for EACH name; inner --repeat 20 → 400 PASS
per alias — scope overrun RECORDED AS 400 (not relabeled 20). Main
directed: STOP additional work; final binding + retained-763 evidence +
JUnit + inventory + driver terminal + handoff packet NOW. No source
failure in the selected results; Main inspects binding before releasing
the SOURCE-HANDOVER BASE. Extra unapproved repeats ≠ full qualification/
economy. Owner's own-script issues require an owner ACK; Main's corrected
runner unused. No Main remote workload.

### Owner-narration contradiction at packet time (Main, ~04:10Z) — quiesce + handback directed

arm-sparks seq 70/71 claimed 5 tests / 20 repeats / missing e0 registration
and proposed a rerun — ALL contradicting Main's frozen SCP + Python
validation (8-test inventory, JUnit, run8.log 8/8 PASS, 400 raw passes
EACH, script actual N≠8 / inner repeat 20). Main directed: NO rerun/
reconfigure/overwrite; owner QUIESCE + terminal-state report + explicit
HANDBACK; Main finishes the binding packet READ-ONLY after explicit
transfer (no further execution). All artifacts/provenance preserved; no
git repair; no source changes. Classification: owner-reporting/harness-
custody issue — NOT missing registration, NOT production test failure.
The actual data (8/8 at e0) stands; the narration about it failed.
Meta-lesson: at handoff time, the owner narrates the frozen artifacts,
not a re-derived story — every packet claim must be re-checked against
the preserved bytes immediately before sending.

### Main read-only host hashes RESOLVE the packet discrepancies (~04:12Z)

- git HOST e0 full CLEAN; source cpp/h hashes MATCH.
- **CURRENT core = a9cb029e55d78c263a3f6a879d18fef450ac064aa554fe83545bf0
  70ad2f1898** (09c1… is the PRESERVED FAILED-763 core — the owner packet
  reused the WRONG EPOCH); current cluster = 32527dac15ca757942f5f65f65b
  99fb906b591b4300d57c0b14ee3b69c3663cc; startup dddddd… unchanged.
- **fault MODULE PRESENT**: tests/v2/libv2_native_cpu_entry_exec_fault.so
  SHA c6152c4164c518ef6409004d6a0e6edc46bb859c583a326631025faa43834cb9 —
  no 'in-memory fixture/gap'; the earlier flag is RESOLVED (wrong
  filename epoch in the packet, not a build gap).
- preserve763 = actual bytes, different inodes; app 09a0… unchanged
  legitimate (same binary, rebuilt tree).
- Inventory/JUnit actual 8 still match Main's hashes.
- Container fecea… REMOVED (docker inspect: NoSuch) — not verified
  running; terminal state consistent with quiesce.
- DO NOT echo the owner packet's epoch-mixed hashes as binding. Main's
  final read-only cache/compile/binary-loader inspection finishing; no
  reruns.

### SOURCE-HANDOVER BASE RELEASED (Main, ~04:22Z): exact 880429059

`880429059d368d88e9e3abae01a016ba0bbf467a` (#43191 above 2948b1e49; CLEAN/
MERGEABLE) — docs-only after the e0 source ACK + Main's independently
verified capable-8/no-skips binding (with the 400-per-alias overrun
recorded), source-current core a9cb029e…, Ninja graph. **#75/#76 bounded
B12 DONE**; issue 57 tracks integration.
EXPLICIT SCOPE: NO QUALIFIED/SETTLED/full-backend/inference/register
claim — the one Main full campaign on the final collection remains
pending. Addon notified: mechanical rebase onto 880 ALLOWED; #191 ab860
STILL HOLD (comprehensive fix-back required first). .79 needs NO further
work; STOP owner text packets mixing epochs — use Main's record/raw
hashes. Source-proof docs updated; all current bad/e0 artifacts kept. NO
benchmark/trace/GPU queue unblock inferred.

### CB/C1 BASE-FIX INVENTORY (Main→CB, 2026-11-11 ~12:14Z) — relayed verbatim to CB lead

Main inspected the C1 final JUnit + actual 28dd source; NO rebase/cherry-
pick/workspace/action authorized from ARM side.
1) **a0393b64a2** — GroupedSealedPlan synthetic observation explicitly
   physical_launch_evidence={}; PROD validator unchanged. Relevant file:
   tests/v2/unit/scripts/test_cpu_native_quantised_grouped_sealed_plan.py
   (temp-index file-only patch --check exit 0 against EXACT 28dd). Whole
   commit ALSO contains unrelated docs → prefer minimal relevant hunk,
   not blind whole-commit import. Focused: V2_Unit_CPUNativeQuantised
   GroupedSealedPlan; the 10 Python errors = ONE CTest alias; repaired
   full script passed natively.
2) **e33bf1de6** — sole file Test__CPUShardRouteCollective.cpp (28dd
   temp-index file patch CHECK exit 0). Diagnostic special-value
   arithmetic scopes fail_on_nan=false + cached-bool RAII restoration;
   FusedFinalAddPreweightHonorsNonFiniteValidationPolicy proves the
   finite/nonfinite/default-policy/empty/inactive/source-byte/restoration
   contracts. PROD admission/dispatch UNCHANGED. Addresses
   V2_Unit_CPUShardRouteCollective + V2_Integration_CPUShardFusedFinalAdd
   Preweight_{scalar,neon,neon_i8mm} — TWO Unit + THREE host aliases,
   NOT all 140. Check-clean-apply ≠ runtime green; CB owner decides
   minimal integration + focused checks + full canonical exit 0; NO
   waiver. I8MM post-integration check still needed on the Station.
   Arithmetic comparison = metamorphic/preweight only, NOT a new
   fused-ticket publication oracle.
3) **B12 mechanism IS PRESENT on 28dd** (MPIBootstrapPhase.cpp:585+
   ScopedOpenMPPartitionAffinity lives through
   bindOpenMPWorkersFastestFirst :615). Fix chain: 763a8e15b (scope-
   lifetime first fix) → 2a4d496fa (initial candidate — HELD/UNSAFE
   ALONE) → c2b1e779 (private authenticated original entry) → e0b7a7c1f
   (replay/exec/loan/worker/cancellation controls); final correct state
   merged in 880 then bb. **DO NOT cherry-pick 2a4 alone or a range
   omitting c2** — semantics require the B12 final API/lifetime chain.
   Pool defect CONDITIONAL on manual place order differing native +
   worker retirement/recreation — not 'any pool resize'. DOES NOT fix
   mbind EPERM/memory-only-NUMA selectors or prove the 29-Unit root
   cause. The C1 ClusterInventoryBootstrap failure shows 9 ranks → all
   NUMA 0 → CPU participant cpu:0 ambiguous across ranks;
   CPUStartupAffinity explicit cpu:5..8 memory-only targets refused (no
   physical CPU set) — NOT demonstrated B12 duplicate-core failures.
   Current Main still needs a Station-specific fixture/topology
   disposition; don't attribute wholesale to B12. 274d0cff4 independently
   fixes the obsolete SINGLE-rank inventory NUMA −1 expectation — not the
   9-rank ambiguity/permission issue.
4) **Other narrow candidates:** a847db3286 corrects observed-NUMA fixtures
   in Test__MoEPipelineNgramPublication/Test__OrchestrationCandidate
   NgramAdmission (+PP checks); 5b76a146c certifies mandatory capacity
   before optional ngram selection — applicability = MATCHING actual
   assertion, not generic immutable/NUMA failure. No later NUMA topology
   detection fix to offer (55/52-class already in the 28dd lineage). Many
   C1 raw outputs show mbind/set_mempolicy EPERM + CPU first-touch failed
   → inspect Station kernel/seccomp/caps/NUMA preconditions and match
   context — no blanket env waiver, no promise cherry-picks fix
   permissions. Unmapped classes: missing docker/model/ROCm-CSV fixtures;
   Developer PR test-method names absent; registration-config errors;
   trainer expectations 3-vs-2/10-vs-9 — no verified Main fix commits;
   route each concretely, don't label 140 'env'.
5) **B11 OPEN/UNATTRIBUTED.** .79 direct 38/39 + 1 skip = NON-reproduction
   on a DIFFERENT host, not MPI-cause proof. Station paired canonical
   MPI+direct from the SAME source/build/binary/container/affinity/NUMA/
   policy identities needed; direct-fail only rules out MPI necessity,
   not allocator-vs-fixture-vs-permission cause. Preserve the
   EPERM→null→SIGSEGV context + skip name; inspect the actual site.
   Tracker base-defect-fix-list methodology corrections authoritative
   over the old 'closed env' paragraph.
Main has NOT claimed canonical green for C1; unchanged failed names =
baseline-relative only. ARM post-merge campaign separate — do not
interrupt its booked .80/A2/DirB windows.

### B11 STATION MATCHED COMPARISON (C1 worker c527, ~12:19Z) — crash is MPI-launcher-specific

Same binary (v2_test_numa_allocator), same container, same source, same
config — the comparison Main prescribed:
- **DIRECT (non-MPI): exit 0.** gtest failures but NO SIGSEGV — the
  allocator fails GRACEFULLY under the memory-only-NUMA topology: EPERM on
  first-touch memory-policy observation → null buffer returns → test
  failures → normal process exit.
- **MPI-WRAPPED (mpirun -np 1): SIGSEGV (signal 11).** The same binary
  under the MPI launcher crashes.
CONCLUSION: the SIGSEGV is MPI-launcher-specific (hwloc binding or
memory-policy interaction under mpirun), not a pure allocator defect —
the allocator handles the topology gracefully; the crash REQUIRES the MPI
launcher environment. B11 classification: MPI-launcher-interaction
defect. Logs: artifacts/cb-c1-staging/current/b11-direct.log (sha256
7046e841…) + b11-mpi.log (sha256 0eeac5b5…).
NOTE: this is the matched-identity comparison the ARM methodology
required (the .79 38/39 non-reproduction was on a DIFFERENT host).
Next diagnostic step if pursued: inspect what mpirun changes that turns
graceful-null into SIGSEGV (launcher-set memory policy? hwloc bind?
different allocator code path under MPI init?).

### **B11 CONCLUSION WITHDRAWN** (CB lead, ~12:20Z) — capture quality failure, claim retracted

The CB lead inspected the actual logs and STOPPED the causal closure:
- b11-direct.log is **403 bytes**, ends at '[ RUN ] NUMABufferIterators' +
  EPERM with NO suite completion → the reported "exit 0" is NOT reliably
  the tested process's status (likely a pipeline/tail exit status). The
  direct log does NOT establish normal completion/no-crash.
- b11-mpi.log explicitly reports SIGSEGV yet ends 'mpi-exit=0' —
  internally contradictory capture.
- Neither tiny log binds source/binary/container/config identities.
- The "SIGSEGV is MPI-launcher-specific, not allocator-inherited"
  conclusion is **WITHDRAWN** pending full untruncated commands/stdout/
  stderr/native statuses. Even a genuine matched split alone would not
  identify hwloc or rule out allocator defects.
- Corrected capture requested from the SAME worker; no duplicate lane.
TRACKER CORRECTIONS: the 8fae2a61e entry's conclusion is superseded by
this entry; the claim as relayed to Main is also withdrawn. LESSON
(recurring class): pipeline exit status ≠ child process status; log-tail
capture ≠ suite completion; tiny logs bind nothing. Claim size must match
capture quality.

### **B11 CORRECTED RESULT** (worker c527, ~12:21Z) — BOTH modes SIGSEGV; allocator-level defect

Full untruncated capture (no pipeline masking): **BOTH direct and MPI
exit 139 (SIGSEGV)** — the earlier "direct exit 0" was a pipeline-masking
artifact (tail consumed the actual exit code). CORRECTED CONCLUSION: the
SIGSEGV is NOT MPI-specific — the same binary segfaults in both modes;
the allocator defect reproduces on the Station regardless of launcher.
**B11 attribution: allocator-level defect on the Station's 64K/
memory-only-NUMA topology, reproducible with and without MPI.** This
SUPERSEDES both the "direct exit 0" conclusion AND the withdrawn
"MPI-launcher-specific" claim (the corrected capture changes the answer
again — the first capture was too small to conclude anything).
SOURCE/BINARY/BINDING: source HEAD 975f72ccba… + working delta (diff
SHA256 133e4e6e…); binary sha256 1c7a6079… (build_v2_cb_c1/tests/v2/
v2_test_numa_allocator); container llaminar-dev:latest sha256:b3a70cb0…
--network none; affinity default (no pinning); permissions default (no
SYS_NICE, no seccomp override); commands: direct = binary directly, MPI =
mpirun -np 1 --bind-to none --map-by slot + same binary. LOGS (full
stdout+stderr): artifacts/cb-c1-staging/current/{b11-direct-full.log,
b11-mpi-full.log}.
STATUS: B11 = allocator-level defect on Station topology, now
REPRODUCED matched-identity with full capture. The ARM .79 38/39
non-reproduction stands as a DIFFERENT-HOST observation (the defect may
be Station-topology-specific). Next: site inspection (EPERM→null→SIGSEGV
path — where does the null dereference happen under the Station
topology?), then fix-or-classify.

### CLASSIFICATION CORRECTION (CB lead, ~12:22Z) — B11 category = crash-site OPEN

"BOTH 139 rules out MPI necessity" is TRUE, but it does NOT establish
'allocator-level defect on 64K/memory-only topology' — THAT
CLASSIFICATION IS WITHDRAWN. Correct category: **crash site / root cause
OPEN — unchecked-null TEST fixture vs allocator vs permission handling.**
Already identified: Test__NUMAAllocator.cpp ~601 range-writes NUMABuffer
without a validity assertion, matching the last direct-log test. NEED:
backtrace + the actual failed allocation path. Do not propagate a stronger
cause than the evidence. 'Default affinity/permissions' and truncated
hashes are not full binding by themselves.

### B11 SCOPE CORRECTION (Main, ~12:23Z) — remove 'allocator-level confirmed' from headline

IF corrected native captures verify BOTH-139: MPI launcher is NOT
NECESSARY for this matched crash — they still do NOT establish
'allocator-level defect' or topology causality without the stack. Main
independently read Test__NUMAAllocator.cpp:601: `NUMABuffer<int>
buffer(64,0)` then IMMEDIATE range WRITE with NO ASSERT_TRUE(buffer.valid());
NUMAAllocator.h constructor PRESERVES count_ even if allocation returns
null; end() computes null+count. CONCRETE unchecked-buffer TEST/ITERATOR
candidate matching where the truncated log stopped — not yet a proved
allocator cause. NEEDED: capture the actual crashing frame + failed
allocation return; distinguish legitimate EPERM refusal, invalid buffer/
test use, and any allocator bug. A guard avoids the secondary crash but
does NOT make the expected-allocation test green or waive permission/
topology admission. CB lead already pursuing matched native status/stack;
NO new ARM host job, no blanket env waiver. Tracker headline
'allocator-level' removed per this correction (headline now: crash-site
OPEN, unchecked-buffer fixture candidate identified).

### **B11 CRASH SITE ISOLATED** (worker c527, ~12:24Z) — unchecked-null fixture site confirmed

SIGSEGV isolated to **Test__NUMAAllocator.NUMABufferIterators (line 601)**
with the single-test gtest filter (exit 139 confirmed). MECHANISM: the
test constructs NUMABuffer<int>(64,0) — NUMAAllocator::allocateLocal
returns NULL due to EPERM on get_mempolicy (Station 64K/memory-only-NUMA
topology) — then range-iterates the buffer WITHOUT checking valid(),
dereferencing null → SIGSEGV. **CLASSIFICATION: unchecked-null FIXTURE
site, NOT an allocator defect** — the test should ASSERT_TRUE(buffer.
valid()) before iterating. The EPERM itself (get_mempolicy EPERM on the
Station topology) is a SEPARATE Station environment issue. Crash log:
artifacts/cb-c1-staging/current/b11-iterators-crash.log; fixture code:
tests/v2/unit/memory/Test__NUMAAllocator.cpp:601-616.
DISPOSITION IMPLICATIONS: (1) B11 as a standalone allocator defect is
CLOSED (fixture defect + environment issue); (2) the CB lane's fix = the
assert-before-iterate fixture hardening (which Main already noted would
avoid the secondary crash but NOT make the expected-allocation test green
— permission/topology admission unchanged); (3) the underlying get_mempolicy
EPERM on 64K/memory-only-NUMA remains a Station environment precondition
to record, not an ARM defect.

### B11 ENVIRONMENT PRECONDITION ATTRIBUTED (worker c527, ~12:27Z): get_mempolicy EPERM = seccomp

SYS_NICE-ONLY EXPERIMENT (controlled, same binary/container): SYS_NICE
does NOT resolve the EPERM — first-touch certification still fails
"Operation not permitted" with --cap-add SYS_NICE + DEFAULT seccomp.
**The get_mempolicy/set_mempolicy EPERM is caused by Docker's default
seccomp profile blocking those syscalls, not by a missing capability.**
The prior "SYS_NICE + seccomp=unconfined" fix required BOTH — the seccomp
relaxation was the actual fix. Two independent mechanisms confirmed:
SYS_NICE addresses rlimit-based restrictions; seccomp addresses syscall
filtering. Log: artifacts/cb-c1-staging/current/b11-sysnice-experiment.log.
RUNBOOK CONFIRMATION: the established "SYS_NICE + seccomp=unconfined for
page-owner tests" rule was correct, and now has the attribution experiment
proving which mechanism does what.

### B11 permission path RESOLVED by CB lead (~13:06Z) — OCI profile move_pages arg-restriction

Same guarded binary (6a60032d/source) on the Station:
- default Docker profile: DENIES get_mempolicy;
- cap-add SYS_NICE: allows the query BUT move_pages EPERM; effective
  SYS_NICE also fails;
- actual default OCI profile + ONE arg-restricted move_pages allow
  (pid 0, nodes NULL, flags 0): PASSES.
Same UID 1000/groups, CapEff 0, seccomp 2, no host changes/unconfined/
privileged. Sole profile diff verified; profile SHA
4b5631f1da22b270ac8169c8203eb3ae3cd6c794f08485d2357e6b16dd81f54b.
Evidence: cb-c1/artifacts/cb-c1-staging/lead-numa-self-query-20261111T1305.
Worker ownership returned; GO = missing preflight registration/full NUMA
suite + full canonical under this EXPLICIT NARROW environment. No blanket
140 closure or allocator/topology cause claim; remaining assertions still
need triage. RUNBOOK: the get_mempolicy/move_pages path needs the exact
arg-restricted profile (not broad seccomp=unconfined).

### CB progress (lead, ~13:33Z): NUMA fixtures green, canonical refresh RUNNING

Lead implemented 4 NUMA fixture corrections (test-only) + explicit
preflight registration; **39/39 direct, 4/4 registered (Unit + 3
preflight), 0 skips** — evidence
lead-numa-suite-20261111T1314-r2. COMPLETE CANONICAL REFRESH NOW
RUNNING (container 0554e50d78e0, cb-c1-canonical-20261111t1330, build.log
growing; artifacts lead-canonical-refresh-20261111T1330) under the same
query-only seccomp profile, pinned image, source/index/full runtime +
dependency bindings; parent .git read-only bind fixes container worktree
git visibility; compile DB export enabled, active Ninja preserved. No
gate waiver/commits/pushes. Lead retains C1 source/build ownership until
the run finishes; worker = read-only other-failure triage. GLM untouched;
no new lanes/ARM resources. (No acknowledgment expected.)

### BOUNDED SOURCE ACK — fix-forward 480de4344 (reviewer, ~14:10Z)

Direct child of bb893cd79. Inspected the complete 2-CMake 8+/4- delta +
dependency/consumer source: root-retirement TU still literally includes
production MPISessionOwner.cpp; normal STATIC archive linking resolves
its new NativeCPUPlacement references by extracting NativeCPUPlacement.cpp
(no whole-archive added); NativeCPUCollectiveProgram removes its separate
Owner.cpp compile and consumes the same canonical owner target. Target
PUBLIC closure = Threads/OpenMP/dl only; dependency direction core ->
owner (NOT owner -> core) → neither test newly links inference/GPU
machinery. Existing collective MPI boundary doubles unchanged. No
remaining independent MPISessionOwner.cpp source entry beyond the
canonical library + the intentional literal-body test. All four
Unit/Integration aliases + preflight membership retained; Doxygen header
accurate. No production/arithmetic/test-body changes. Reviewer ran no
builds/tests/log auth — focused execution + the same post-merge campaign
remain Main-owned and pending (NOT a new admission gate). No new
issue/artifact. STATUS: fix-forward source-ACKed; focused epoch-480
execution protocol already dispatched to arm-sparks (build 2 targets →
exact 4-name regex → readelf → focused PASS → full campaign at 480 with
new /work/campaign-480-results).

### CAMPAIGN480 FIX-FORWARD PROGRESS (Main, ~17:06Z — literal, .80 lease still Main)

1) **PUSHED ebbeee230:** authored OMP_PROC_BIND=false honored +
   NativeCPUPlacement getenv bootstrap exception. Focused GB10: 12/12
   placement/scan regressions pass incl. NativeCPUEntryLifecycle +
   MPIProcessAffinity (first candidate broke spread lifecycle; preserved,
   never pushed).
2) **PUSHED 395dc20ef:** A1 fork evidence — CPU decode cache-miss runs
   eagerly OUTSIDE executeCacheHit and never published the first-step
   fork summary (PRODUCTION GAP, gated counter only); the retained test
   used executeMultiDevice instead of a retained plan; restoration control
   used SUMMARY=0. 4/4 fork-evidence Unit/Integration aliases pass.
3) **ShmemSpin NP4/NP8 ROOT CAUSE — branch-introduced by B12:** under
   mpirun with no authored OpenMP places, every co-resident rank
   independently took canonical core admission and bound its master to
   the SAME first place (all 4 ranks observed Cpus_allowed_list=19),
   time-slicing on one core: 18 ms per 2048-float allreduce. Same defect
   affects EXTERNAL mpirun launches of llaminar2. Candidate: co-resident
   launcher ranks (OMPI_COMM_WORLD_LOCAL_SIZE/MPI_LOCALNRANKS >1) keep
   native partition; added 'shared' case to preflight MPIProcessAffinity.
   Building focused + full MPI-label validation (started 17:06:36Z); push
   only after pass.
Inherited/env classification unchanged; operator scope decision pending
(dispatched by coordinator: fix-forward (1); (2) routed to addon3 (A1
owner); (3) routed to addon (ShmemSpin ARM owner)).

### CAMPAIGN480 FIX-FORWARD CONTINUED (Main, ~17:40Z): 12675c956 pushed + NEW canonical campaign

**Main pushed 12675c956:** shared mpirun partition fix (ROOT CAUSE of
ShmemSpin NP4/NP8: all co-resident ranks pinned to CPU19 by the B12
native entry under mpirun with no authored OpenMP places). SAME commit
also fixes the MPIBootstrapEnvironmentPolicy fixture broken by the
earlier ebbeee230 getenv-allowlist change (caught by a FULL MPI-label
A/B vs the merged baseline; DISCLOSED). Focused 18/18 pass; MPI label:
only differences vs baseline = MPIProcessAffinity fixed + that fixture.
**NEW Main canonical campaign at pushed tree 0924175d STARTED 17:40:13Z**
(CID 30a7014e67a0…, host PID 2631094, fresh build-campaign-1267 +
campaign-1267-results). Lease still Main. Remaining expected failures =
the inherited/env list; operator scope decision still requested
(coordinator disposition: fix-forward what the collection introduced;
report inherited classes to their owners — standing unless the operator
overrides).

### CAMPAIGN 12675c956 RESULT (Main, ~18:13Z): 25 fail (was 35), 0 NEW, 10 FIXED — collection clean, tip not fully green

Container main-campaign-1267 exit 8 at 18:12:26Z; receipt rc8, 3421
tests, identical inventory to 480, 0 skips. **FIXED (10):**
Static_NoDirectGetenv, HostServiceDebugEnvAuthority, MPIProcessAffinity
U/I, OpenMPTeamForkStageEvidence U/I, OpenMPForkEvidenceEnginePath,
ForwardExecutionEngine, ShmemSpin NP4/NP8. **NEW failures: none.**
**REMAINING 25 = exactly the inherited/env set already reported** (MTP
NUMA fixtures ×3, NativeSourceFinalRetirement ×3, registration_selected
fixture projects ×7, ROCm trainer fixtures ×4, dispatch registry, trainer
stream wait, MoE scan, DevelopPRGate, ModelKLD ×2 docker, CPURootedGather
2-socket). Raw /tmp/llaminar-main-bb-linux-results/campaign1267/.
Collection NOT qualified (tip not fully green); operator decision pending
on base-defect scope. Lease: Main done executing, holding until the
operator decides.
DISPOSITION FRAME (coordinator): the collection introduced ZERO new
failures and fixed its own 10; the 25 inherited = base defects the
campaign was designed to SURFACE (the triage is concrete: exact tests +
lineage + base commits). Under merge-then-qualify, inherited defects are
fixed on Main in subsequent work, not inside this collection.

### ShmemSpin: independent-convergence datapoint (addon, ~20:36Z) — recorded, no action

The addon's parallel diagnosis (dispatch/recall race) independently
derived the starvation-quantum mechanism (full-speed spin in the single
wait funnel waitForPeerEpoch never releases the CPU; ~3ms scheduler
quantum per rendezvous × 6 waits/allreduce = the 18.17s stress) and
proposed escalating backoff (spin → sched_yield ladder → 50µs nanosleep).
Main's fix (12675c956) addressed the TRIGGER (B12 NativeCPUPlacement's
shared pin to CPU19 — the pinning cause); the addon's mechanism analysis
AGREES on the class and is recorded as an independent-convergence
datapoint. Backoff work: NOT pursued (production latency economics out of
scope without measured evidence — Main's rule, addon concurred).
Post-fix validation: NP4 0.43s, NP8 0.64s, campaign passing.
FMLAL leaf: addon RESUMED implementation (fresh branch
codex/arm-fmlal-half-leaf off bb893cd79; plan approved; syntax-only
locally; .80 legs deferred).

### CB CANONICAL REFRESH RESULT (CB lead audit report, ~20:41Z): exit 8, gate failed

cb-c1-canonical-20261111t1330 (0554e50d) exited 8 at 13:47:55Z (~18 min
runtime); receipt: 3415 tests, gate FAILED. Receipt/XML/logs:
cb-c1/artifacts/cb-c1-staging/lead-canonical-refresh-20261111T1330/
qualification/prerequisites.json. Interpretation: the refresh ran under
the narrowed seccomp profile (lead's query-only + arg-restricted
move_pages allow) with the two authorized test-backports applied; the
remaining failures = the inherited set from Main's base-fix inventory
(minus the backport-fixed aliases). Next: CB lead triages the remaining
classes against Main's inventory (concrete fixes vs environment
preconditions — the seccomp attribution already resolved one class).
CB-program work; ARM side informational only.

### STATION CLEANUP (c1d0 worker, operator-ordered, ~21:06Z) — evidence consolidated FIRST

All c1d0 build trees removed on the operator's direct "cleanup everything
you don't need" order. **Evidence consolidated FIRST and INTACT:**
/home/tdsynnex/repo/llaminar-evidence-2026-11-11/ (118MB, 40 files) —
b-fix-0db136607/, b-fix-9490b66bc/, b-fix-ce9ddb326/ (counter-evidence +
B10 discovery chain), canonical unit/preflight logs (NEON+I8MM), all
bench_*.json/logs + bench_baseline.md, TESTED_SHA.txt, README-PATH-MAP.txt
(old→new path map). TRACKERS TO REPOINT: e74703801/687fde099/df0370d80
references → the evidence dir.
ALSO REMOVED: llaminar-cb-c2 worktree+tree (34G, frozen C2 artifacts —
C1 owns the branch, advanced to 975f72ccb in their own checkout); the
22GB policy-rejected Qwen3.5-35B-A3B-Q4_K_M.gguf; host ~/tools; onednn
build dirs + stale locks (source clone kept, 99M). KEPT: standard model
set (11 GGUFs, 27G), main .venv (2.2G), llaminar-dev image, main checkout
(clean, detached ce9ddb326), all other lanes' worktrees. Disk: 90%→86%
(256G free).
ARM-SIDE NOTE: the removed C2 artifacts were already dispositioned (C2
closed; the coordinator-held C2 state was the frozen snapshot — the
branch lives with C1). The B10/B11 evidence chains (b-fix-* dirs) are the
ARM-relevant part — INTACT at the consolidated path.

### CROSS-PROGRAM (CB/C1, operator-confirmed): the 2-socket cell is HARDWARE-BLOCKED

V2_Integration_CPURootedGatherParallelCopyReplay (requires MPI 2 +
valid CPUExpertSocketMembership(endpoints, 2) = 2 actual physical
socket workers): **operator-confirmed NO eligible 2-socket host exists
in the known fleet** (Mac/GB10×2/GB300/WSL all single-socket; WSL
flattens NUMA besides); the stale factory-host candidate is RETRACTED.
**Status: HARDWARE-BLOCKED / unqualified, pending host — NOT
accepted/passed.** The C1 complete-green commit/push rule is unchanged;
NO inherited-failure waiver authorized. Recorded in the arm-perf
tracker as a cross-program fact so any future fleet expansion (a
2-socket machine) picks this cell up first. CB lead continues other
concrete fixes; no host probes/jobs requested.
