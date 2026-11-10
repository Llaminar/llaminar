# Idea 15: known-failure classification on NUMA-less and 64K-page ARM hosts

Status: classification record, revision 2 (reviewer #144). Report only, no production change. Gap
tickets go through arm-perf. Source anchors refer to `39c24d8b4` (the gated tree). The tracker
declares `7d1a94479` and `a27c7ed4f` content-equivalent for CPU-only code. Every count below is
produced by `classify_v2.py` from the raw logs listed in §6.

## 1. Host records

**DGX station: NVIDIA GB300, Grace / Neoverse-V2**

- **Topology:** 72 cores, one socket, one thread per core.
- **Page size:** `PAGESIZE=65536` (64K pages).
- **Features:**
  - Listed: sve, sve2, svei8mm, i8mm, bf16, svebf16, fp16, sha3, sha512, sm3, sm4, paca, pacg, bti.
  - I8MM and SVE2 are present; SME is not.
- **NUMA layout** (`probes_b1.txt`):

| Node | CPUs | MemTotal | Distance from node 0 |
|---|---|---|---|
| 0 | 0-71 | 518,468,096 kB (506 GB) | 10 |
| 1 | none | 261,619,712 kB (255 GB) | 80 |
| 2-8 | none | 0 kB | 80 (11 from node 1) |

- **Container:** `Mems_allowed_list=0-1` and `Cpus_allowed_list=0-71` inside the devcontainer. The
  policy may target node 1, which has no CPUs.
- **Gate launch identity:**
  - devcontainer `CapAdd=[] SecurityOpt=[] ShmSize=64 MiB`, i.e. Docker's default seccomp without
    `CAP_SYS_NICE`;
  - for comparison, the operator GLM container has the same seccomp settings but `ShmSize=32 GiB`.
- **Build:** gcc-14, cmake 4.0.3, ninja 1.13, Integration, CPU-only, `LLAMINAR_SHARED_CORE=ON`.
  Envelopes: `NEON` (default) and `NEON_I8MM` (second run).

**Mac Docker VM: Apple M1 Max under linuxkit 7.0.14**

Evidence: `mac/env-probe-a27c7ed4f.log` and `mac/env-probe-seccomp-split.log`.

- 10 CPUs, `PAGESIZE=4096`.
- Features: asimd, asimddp, fphp, asimdhp, sha3, sha512, asimdfhm, frint. There is **no** i8mm,
  sve or bf16.
- `/sys/devices/system/node` is absent and `numactl` reports 0 nodes.
- Cache `index*` entries publish level, type and `shared_cpu_list`, but **no** `size` and **no**
  `coherency_line_size`.
- `set_mempolicy`/`get_mempolicy` give different answers depending on how the container is launched:

| Launch | set_mempolicy / get_mempolicy |
|---|---|
| Docker default seccomp, no CAP_SYS_NICE | EPERM |
| `--cap-add SYS_NICE` or `--security-opt seccomp=unconfined` | **ENOSYS** (kernel built without NUMA) |

The kernel's true answer is ENOSYS. Default seccomp masks it as EPERM. This explains the earlier
"EPERM/NUMA" environment class. It also means **container launch policy is part of the environment
identity**: Main's `llaminar-armbuild` gate container (default seccomp, 64 MiB shm) and the earlier
`m1f` container (SYS_NICE, unconfined, 16 GiB shm) are different environments, and their failure sets
are not comparable as-is.

**WSL x86**

AVX512, GCC 14, at 39c24d8b4. These logs were captured **without `--output-on-failure`**. They
record which tests failed, but not why. An x86 failure of the same test name is therefore kept as a
separate baseline observation (§2) and is never used as the cause of an ARM failure.

## 2. Method: three independent fields per failing test

Each failing test of each ARM run gets three fields:

| Field | Meaning |
|---|---|
| `x86_same_name` | The same test name also failed on WSL x86 at 39c24d8b4. This is a baseline observation only; the x86 cause is unobservable (see above). |
| `tags` | Every cause rule whose evidence line occurs in the ARM failure output. Tags **overlap**. They are matched over the whole ctest output of that test, which includes passing gtest siblings; this is why G8 was withdrawn. |
| `primary` | The first matching tag in the priority order below, or **UNRESOLVED**. Exactly one per test, so the primary counts of a run are a **disjoint partition** of its tests. |

Tags in priority order:

| Tag | Evidence | Verdict |
|---|---|---|
| X1 | Target missing: unguarded AVX2 `.inc` from `cc502b95a` | BUILD BREAK |
| S1 | `mbind`/`set_mempolicy`/`get_mempolicy` EPERM, Open MPI "failed to bind memory" | ENVIRONMENT (container launch) |
| G1, G2, G3, G6 | See §4 | PRODUCT GAP |
| N2 | Multi-rank inventory or rank-local backend requires exact NUMA affinity (the G5/G7 detection path) | PRODUCT GAP |
| N1 | NUMA binding or policy required on a NUMA-less kernel (incl. `get_mempolicy: Function not implemented`) | PRODUCT GAP (WP J) |
| N3 | "Streaming CPU measurement requires observed aggregate cache coverage" | PRODUCT GAP (WP J) |
| M1 | Test targets a CPU-less node ("Topology exposes no CPU on requested NUMA node") | TEST ASSUMPTION |
| T1 | Test passes a 4096-byte mmap window; production default is 64 KiB | TEST ASSUMPTION |
| T2 | 4K-host byte literal or cap (4491052224, 8392704, `recordedFragmentBytes`, 49×4096, `< 31`, staging cap) | TEST ASSUMPTION |
| E1 | oneDNN origin-probe git behaviour in the devcontainer | ENVIRONMENT |
| E2 | `docker` CLI absent in the container | ENVIRONMENT |
| E3 | Nested CMake configure fails inside the test ("Configuring incomplete") | UNRESOLVED-ENV: the configure error itself was not extracted |
| F1 | ROCm trainer fixtures absent from the tree | FIXTURE MISSING |
| V1, V2 | Mac timeouts: pass standalone, or the 34 s suite against its 30 s budget | ENVIRONMENT (VM) |

**UNRESOLVED** means no rule matched. These rows are listed by name; they are not attributed to the
base merely because x86 also failed them.

## 3. Results (disjoint primary partition; x86 same-name overlap shown per row)

**DGX canonical run** (39c24d8b4, NEON, default seccomp): 130 failing tests, of which 27 have
`x86_same_name`.

| Primary | Tests | …of which x86_same_name |
|---|---|---|
| S1 seccomp EPERM | 47 | 5 |
| T1 mmap window | 40 | 0 |
| T2 4K literal | 17 | 0 |
| E3 nested configure | 9 | 9 |
| UNRESOLVED | 8 | 7 |
| F1 fixtures | 4 | 4 |
| E2 docker | 2 | 2 |
| E1 git | 2 | 0 |
| G3 | 1 | 0 |
| **Total** | **130** | **27** |

- Overlapping tag occurrences: S1 47, T1 47, T2 22, E3 9, F1 4, E1 2, E2 2, G1 1, G3 1, N2 1.
- UNRESOLVED: MoEForbiddenDependencyScan, ProductionParityCampaigns, CPUNativeQuantisedGroupedSealedPlan,
  CPUShardRouteCollective, NativeQuantisedCommonDispatchPolicy, DevelopPRGateContract,
  CPUShardFusedFinalAddPreweight_scalar, CPUShardFusedFinalAddPreweight_neon. All except the `_neon`
  cell also fail by name on x86.
- WSL-only failures, recorded separately: ParityFixtureShardOwnership (the ideas07-25 stale-ccache
  artifact) and MoEOwnedSharedBackingNativeStartup (passes on DGX). The `_avx2`/`_avx512` cells are not
  registered on ARM.

**DGX unmasked rerun** of 52 tests (`--cap-add SYS_NICE --security-opt seccomp=unconfined`, same image
and build, default 64 MiB shm). The sets are disjoint exact test sets.

PASS, 16 tests: these failed only because of the seccomp mask.

- 7 `CPUJoint*`: BufferedSourceEconomy, ColdGraphAdmissionLifecycle, ColocatedSerialArenaLifecycle,
  InitialBankPrepassContract, LateGraphSourceLifecycle, ReusableRegistryBinding,
  SerialSourceGeometryLifecycle.
- 2 `CPUFragment*`: EpochBankContract, PayloadContract.
- 6 `FrontendAutomaticSelection*_MPI`.
- `Qwen35MoEGraph`.

`CPUJointBufferedSourceEconomy` also fails by name on x86, yet passes here. This shows concretely why
same-name overlap is not causal attribution.

FAIL, 36 tests, by primary cause:

| Primary | Tests | …of which x86_same_name |
|---|---|---|
| G2 CPUBackend 4K mbind | 9 | 0 |
| T1 mmap window | 9 | 0 |
| T2 4K literal | 7 | 1 |
| G6 expert estimate | 4 | 0 |
| UNRESOLVED | 3 | 3 |
| G1 | 1 | 0 |
| G3 | 1 | 0 |
| N2 (ClusterInventoryBootstrap) | 1 | 0 |
| M1 | 1 | 0 |
| **Total** | **36** | **4** |

- Overlapping tag occurrences: G2 9, T1 9, T2 9, G6 4, M1 2, G1 1, G3 1, N2 1. ExpertGemmRegistry
  carries G6+T2, MemoryPlanner T1+T2, NUMAAllocator G1+M1.
- UNRESOLVED: CPUStartupAffinity (no failure line in its output), CPURootedGatherParallelCopyReplay
  (`validCPUExpertSocketMembership(endpoints,2)` false; a memory-only-node topology effect is plausible
  but unproven), CPUShardRouteCollective (the preweight assertion).

**DGX NEON_I8MM run**

| Run | Unit | Preflight |
|---|---|---|
| I8MM | 29/985 | 102/2373 |
| NEON | 29/984 | 101/2344 |

The failing-name set is identical to NEON, except for `CPUShardFusedFinalAddPreweight_neon_i8mm`, the
I8MM twin of the UNRESOLVED preweight cells. There are no I8MM-specific failures on Grace, and all
newly registered I8MM cells pass.

## 4. Product gaps (ticket candidates, filed through arm-perf)

| ID | Severity | Defect | Evidence |
|---|---|---|---|
| G1 | HIGH | `NUMAAllocator.cpp:43` `constexpr PAGE_SIZE = 4096` drives `aligned_alloc` alignment, `madvise(MADV_DONTNEED)`, the first-touch stride and `move_pages`. On 64K kernels a 4K-aligned pointer gives `madvise` EINVAL and a null allocation. If a pointer happens to be 64K-aligned with a range under 64K, the kernel rounds DONTNEED up to the whole 64K page and **silently zeroes neighbouring live heap data**. | Unmasked `V2_Unit_NUMAAllocator`: "Could not revoke 4096 bytes before first touch on node 0: Invalid argument" |
| G2 | HIGH (blast radius) | `CPUBackend::allocate` aligns to 4096 then calls `mbind(MPOL_BIND)` on that range. This gives EINVAL unless the pointer is 64K-aligned, and otherwise binds a neighbour's page. `DeviceWorkspaceManager` then cannot allocate ordinary CPU workspaces. | 9 unmasked tests: FlashAttention ×2, GroupedVerifierRows, MTPSpec, PlanCommandPublication_MPI, embeddings ×3, CPUBackend. Line: `mbind(…) failed: errno=22` |
| G3 | MED | `DeviceRegistry::discoverCPU` registers every `/sys/devices/system/node/nodeN` as a CPU device, in `readdir` order, including CPU-less and zero-memory nodes (node 1; nodes 2–8). | `V2_Unit_DeviceRegistry` capacity 0; probes |
| G4 | LOW | `MoEOverlayNodeLocalDeviceControllerFabric::planLayout` is called with a literal `4096u` as its "first-touch granularity". It is correct on 64K, but declared per-owner first-touch locality collapses to whichever rank touches a 64K page first. Moot on Grace's single CPU node. | Source only (`…Fabric.cpp:897`) |
| G5 | MED (maxim 29) | `NUMATopology::detectLocalNUMANode` silently falls back to node 0 with a WARN. | DGX logs: "NUMA detection failed, assuming node 0" |
| G6 | MED | `ExpertPreparedMemoryGeometry::cpuProjectionBytes` prices CPU expert storage as **Heap** with a 4096 minimum, which is promoted to the runtime page only for payloads of at least one page. A NUMA-node `CPUStorageRoot` materializes as an **AnonymousPageMapping**, which always rounds to the runtime page. Every sub-page projection is therefore under-admitted. PMA rejects the claim correctly; the admission figure comes from production. | 4 unmasked tests: "live claim exceeds admitted routed_expert_weights … admitted=4096 requested=65536" |
| G7 | MED | `NUMATopology::detectViaProcfs` decides a CPU's node by substring search in the node's `cpulist` string (`"0-71".find("9")` misses CPU 9) and scans only nodes 0–7. With hwloc also returning −1, multi-rank inventory hard-fails. Host-independent parse defect. | Unmasked `ClusterInventoryBootstrap`: "Multi-rank hardware discovery requires exact CPU NUMA affinity" |

**Benign (not gaps):**

- `CPUBackend` and `MmapRegion` touch strides of 4096 only cause redundant first-touch stores on 64K
  pages.
- `MmapRegion`'s `MADV_HUGEPAGE` and the immutable-row validator are page-size correct.
- The `ExpertPreparedMemoryGeometry` 4096 minimum alignment is benign for payloads of at least one
  page. G6 covers the sub-page case.

**Withdrawn:** G8 (activation arena). The ERROR line comes from the passing deliberate rejection test
`CPUGraphArenaDeclarations.SmallActualSchemaMaterializesExactlyAndRejectsShortOwnerBudget`.


## 5. Mac record (a27c7ed4f, NUMA-less kernel, 4K pages, no published cache sizes)

Run with the kernel-truth launch (SYS_NICE, seccomp unconfined, 16 GiB shm), NEON_DOTPROD envelope,
Integration build, private ccache. Unit 957/984, preflight 2296/2374: 105 failing tests, of which 28
have `x86_same_name`.

| Primary | Tests | …of which x86_same_name |
|---|---|---|
| N1 NUMA binding/policy required | 41 | 3 |
| N2 multi-rank/rank-local NUMA affinity | 25 | 2 |
| UNRESOLVED | 10 | 8 |
| E3 nested configure | 9 | 9 |
| X1 build break | 8 | 0 |
| F1 fixtures | 4 | 4 |
| V1 timeout, passes standalone | 3 | 0 |
| E2 docker | 2 | 2 |
| V2 suite over its 30 s budget | 2 | 0 |
| N3 unpublished cache sizes | 1 | 0 |
| **Total** | **105** | **28** |

- **UNRESOLVED (10):** MoEForbiddenDependencyScan, ProductionParityCampaigns (timeout under the gate),
  CPUNativeQuantisedGroupedSealedPlan, NativeQuantisedCommonDispatchPolicy (registry 10 vs 9),
  CPUShardRouteCollective, DevelopPRGateContract, CPUStartupAffinity, and
  CPUShardFusedFinalAddPreweight_scalar/_neon/_neon_dotprod. Eight of these also fail by name on x86.
- **X1 (historical; recorded, not closed by the fix).** These 8 cells failed at a27c7ed4f because their
  target did not compile on aarch64. The source defect is fixed in `f69446e3c`.
  - Closure is separate rerun evidence: at `f69446e3c` (plus the two Bundle-1 test-only commits) all 8
    built and passed on this Mac container (`01-tip-f69446e3c.log`).
  - That is a focused rerun of the 8 cells, not a full Unit/preflight rerun.
- **Revision-1 correction:** v1 put `TensorFactory{Aggregate,Bound}NUMAPolicy` in N1 because the test
  *name* matched `NUMAPolicy`. v2 classifies them by their output (`get_mempolicy: Function not
  implemented`, the kernel's ENOSYS), which is still N1.

Cross-host view:

- N1/N2/N3 appear only on the NUMA-less Mac.
- G1/G2/G6 appear only on 64K-page Grace.
- G3 (CPU-less/zero nodes) and G7 (procfs substring parse) are latent wherever that topology exists.
- E2/E3/F1 are container or tree conditions seen on both ARM hosts. Whether x86 fails for the same
  reasons cannot be determined from the WSL logs.

## 6. Evidence

Raw logs, relative to `/tmp/armqual-b1`:

| Path | Content | SHA-256 |
|---|---|---|
| `raw/dgx/unit_test3.log` | DGX canonical Unit | `d835e47f…` |
| `raw/dgx/preflight.log` | DGX canonical preflight | `21e55c1f…` |
| `raw/dgx/unmasking/unmask_rerun.log` | DGX unmasked rerun | `27b2a803…` |
| `raw/dgx/unmasking/unmask_cmd.txt` | Rerun command | `30a92faf…` |
| `raw/dgx/unmasking/probes_b1.txt` | Station probes | `ffb04ec0…` |
| `raw/dgx/i8mm/unit_i8mm.log` | DGX I8MM Unit | `59e593f3…` |
| `raw/dgx/i8mm/preflight_i8mm.log` | DGX I8MM preflight | `24c87a16…` |
| `raw/wsl/unit.log` | x86 failing-name baseline, no failure output | `f736ea26…` |
| `raw/wsl/preflight.log` | x86 failing-name baseline, no failure output | `442daceb…` |
| `mac/unit-a27c7ed4f.log` | Mac Unit | `d561d7c2…` |
| `mac/preflight-a27c7ed4f.log` | Mac preflight | `cd8b31f2…` |
| `mac/build-a27c7ed4f.log` | Mac `-k 0` build | `f27ad5ca…` |
| `mac/timeouts-standalone-a27c7ed4f.log` | Standalone reruns of the timeouts | `d22b6828…` |
| `01-tip-f69446e3c.log` | X1 closure rerun | — |

Mac probes: `mac/env-probe-a27c7ed4f.log`, `mac/env-probe-seccomp-split.log`, `mac/env_probe.c`.

Classifiers, archived in the tracker at `docs/v2/projects/2026-11/arm-perf-review/evidence/bundle1/`:

- `classify.py` and `classification-39c24d8b4.csv`: revision 1, DGX/WSL. **Superseded** by v2; it used
  base-by-name attribution.
- `classify_mac.py` and `classification-mac-a27c7ed4f.csv`: revision 1, Mac. **Superseded**, for the
  same reason plus the name-matched N1 rule.
- **To add (revision 2, authoritative):** `15/classify_v2.py` (`ac0a624f…`),
  `15/classification-v2.csv` (`35c21aff…`, one row per (run, test) with `x86_same_name`, `tags`,
  `primary`), and the shared block splitter `15/extract.py`.

## Reviewer #144 source-bounds addendum (2026-11-10) — wording fixes REQUIRED

- classify_v2.py:48-61 scans the ENTIRE CTest output (incl. passing/expected-
  negative siblings) then promotes first match to primary+cause -> the
  PRIMARY DIAGNOSTIC TAGS are disjoint tags, NOT established causes. EITHER
  bind evidence to the actual failed case OR label output HEURISTIC
  diagnostic classification and keep causes unresolved; stop 'by primary
  cause'; same for 'passes unmasked => seccomp-only' and 'timeout passes
  alone => VM cause'.
- 39c->7d1/a27 'CPU-only content-equivalent' claim is NOT true:
  ContributionContract.h +50/-7; PackedWeights.h +5/-2;
  UnsignedWeightDotLifecycle.inc +333/-2; AVX2/Gemv + registrations change.
  -> scope correspondence to exact unchanged files; no whole-CPU/test
  qualification transfer.
- 'No I8MM-specific failures' stronger than matched failed-name set (extra
  unresolved *neon_i8mm failure): state exact name-set observation + scoped
  dedicated-test passes; causes/qualification open.
- G1/G2: 'CAN affect neighbours' (static reachable hazard), NOT an observed
  corrupting run. G3: API defect proven; not automatically the planner's
  path. G4: byte-range nonoverlap != runtime-page first-touch ownership;
  'correct on 64K' too broad. G5: NUMATopology:82-87 publishes node0 +
  detection_succeeded=false + method=fallback; ClusterInventoryGatherer:261-
  263 + RuntimeInitPhase:59-60 REJECT failed detection -> narrow to
  misleading/synthetic failure metadata; NOT proof those paths accept node0.
  G6: 'EVERY sub-page projection' too broad (65532 FP32 -> 65536 both);
  'CAN under-admit sub-page projections' accurate; PMA rejection separate
  from estimator defect. G7: ALSO falsely ACCEPTS CPU1 in '10-19' (not only
  misses CPU9 in '0-71'); generic affinity failure doesn't identify detector.
