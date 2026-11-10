# Idea 15: heuristic failure classification on NUMA-less and 64K-page ARM hosts

Status: classification record, revision 3 (reviewer #144). Report only, no production change. Gap
tickets go through arm-perf.

What this record does and does not establish:

- The per-test labels are **heuristic diagnostic tags** derived from failure text. They are not
  established causes. A cause is established only where §4 cites independent source evidence.
- **Source correspondence is scoped to files, not trees.** 39c24d8b4 (DGX, WSL) and a27c7ed4f (Mac)
  differ in CPU kernel code: `CPUNativeQuantisedGemv.h`, `CPUNativeQuantisedContributionContract.h`,
  `CPUNativeQuantisedPackedWeights.h` and `CPUNativeAVX2Gemv.h` (+352/−143), plus tests and
  registrations. Whole-tree CPU or test qualification therefore does not transfer between those SHAs.
  - What does correspond: every §4 anchor file is byte-identical between 39c24d8b4 and a27c7ed4f
    (`git diff --stat` empty). Those files are `NUMAAllocator.cpp`, `CPUBackend.cpp`,
    `DeviceRegistry.cpp`, `NUMATopology.cpp`, `ExpertPreparedMemoryGeometry.h`, `CPUStorageRoot.h`,
    `AlignedVector.h`, `ImmutableRowTablePolicy.cpp` and `MoEOverlayNodeLocalDeviceControllerFabric.cpp`.

Every count below is produced by `classify_v3.py` from the raw logs in §6.

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

AVX512, GCC 14, at 39c24d8b4. These logs were captured **without `--output-on-failure`**: they record
which tests failed, not why. An x86 failure of the same name is kept as a separate baseline
observation and is never used as a cause.

## 2. Method

Each failing test of each ARM run gets these fields:

| Field | Meaning |
|---|---|
| `x86_same_name` | The same test name also failed on WSL x86 at 39c24d8b4. Observation only. |
| `evidence_scope` | `failed-gtest-cases` when the output contains failed gtest cases. Tag text is then taken **only** from `[ RUN ] X` … `[  FAILED  ] X` segments of the failed cases, so passing or expected-negative siblings are excluded. `whole-output` for Python, MPI and non-gtest output, where no case binding is possible. |
| `tags` | Rule matches within that scope. They overlap. |
| `whole_output_tags` | The same rules over the unbound whole ctest output, kept as weaker context. |
| `primary` | The first scoped tag in the priority order below, or **NO-TAG**. Exactly one per test, so each run's primary counts partition its tests. A primary is a **diagnostic tag, not a cause**. |

Binding limit: logger lines go to stderr and are not positioned inside a gtest case in ctest's merged
output (verified: `Test__NUMAAllocator`'s "Could not revoke" lines precede the first `[ RUN ]`). For
gtest rows, case-bound evidence therefore sees only assertion and exception text. Rows whose only
evidence is a logger line become NO-TAG, with that evidence kept in `whole_output_tags`.

Tags in priority order:

| Tag | Matched text | Reading |
|---|---|---|
| X1 | Test executable missing (`could not access … executable`, `cannot prepare exact child spawn`) | build break at a27c7ed4f |
| S1 | `mbind`/`set_mempolicy`/`get_mempolicy` errno=1/EPERM, Open MPI "failed to bind memory" | container launch (seccomp) |
| G1, G2, G3, G6 | See §4 | source-evidenced gap where §4 says so |
| N2 | "Multi-rank hardware discovery / Rank-local CPU backend … requires exact NUMA affinity", "CPU shorthand requires observed NUMA affinity" | NUMA-affinity requirement (G5/G7 detection path) |
| N1 | "libnuma reports NUMA unavailable", "NUMA policy APIs are unavailable", `get_mempolicy: Function not implemented`, node −1 | NUMA policy unavailable (WP J) |
| N3 | "Streaming CPU measurement requires observed aggregate cache coverage" | unpublished cache sizes (WP J) |
| M1 | "Topology exposes no CPU on requested NUMA node" | test targets a CPU-less node |
| T1 | "immutable row mmap window must be page-aligned and nonzero" | test passes a 4096-byte window; production default 64 KiB |
| T2 | 4K-host literals/caps (4491052224, 8392704, `recordedFragmentBytes`, 49×4096, `< 31`, staging cap) | test assumption |
| E1 | oneDNN origin-probe text | container git |
| E2 | `No such file or directory: 'docker'` | container |
| E3 | "Configuring incomplete" | nested configure failure; the configure error itself not extracted |
| F1 | `rocm/fixtures` missing | fixture absent from tree |
| V1 / V2 | Mac timeouts | see §5 |

## 3. Results

**DGX canonical run** (39c24d8b4, NEON, default seccomp): 130 failing tests, 27 with `x86_same_name`.

| Primary diagnostic tag | Tests | x86_same_name | failed-case-bound |
|---|---|---|---|
| T1 | 46 | 0 | 46 |
| S1 | 33 | 2 | 23 |
| NO-TAG | 18 | 10 | 14 |
| T2 | 15 | 0 | 15 |
| E3 | 9 | 9 | 0 |
| F1 | 4 | 4 | 0 |
| E2 | 2 | 2 | 0 |
| E1 | 2 | 0 | 0 |
| G3 | 1 | 0 | 1 |
| **Total** | **130** | **27** | |

- Of the 18 NO-TAG rows, 8 carry an S1 whole-output tag and 2 a T2 whole-output tag.
- **8 carry no tag at all:** MoEForbiddenDependencyScan, ProductionParityCampaigns,
  CPUNativeQuantisedGroupedSealedPlan, CPUShardRouteCollective, NativeQuantisedCommonDispatchPolicy,
  DevelopPRGateContract, and CPUShardFusedFinalAddPreweight_scalar/_neon.

**DGX rerun with a changed launch** (52 tests, `--cap-add SYS_NICE --security-opt seccomp=unconfined`,
same image and build, default 64 MiB shm, one run).

**Passed, 16 tests:**

- 7 `CPUJoint*`: BufferedSourceEconomy, ColdGraphAdmissionLifecycle, ColocatedSerialArenaLifecycle,
  InitialBankPrepassContract, LateGraphSourceLifecycle, ReusableRegistryBinding,
  SerialSourceGeometryLifecycle.
- 2 `CPUFragment*`: EpochBankContract, PayloadContract.
- 6 `FrontendAutomaticSelection*_MPI`.
- `Qwen35MoEGraph`.

These tests failed under the default launch and passed under the changed one. This is **consistent
with** the seccomp mask, but it is one run per launch, so flakiness and the other launch differences
(added capabilities) are not excluded. It is not proof that seccomp was the only cause.
`CPUJointBufferedSourceEconomy` also fails by name on x86.

**Failed, 36 tests:**

| Primary diagnostic tag | Tests | x86_same_name | failed-case-bound |
|---|---|---|---|
| NO-TAG | 10 | 4 | 9 |
| T1 | 8 | 0 | 8 |
| G2 | 8 | 0 | 7 |
| T2 | 5 | 0 | 5 |
| G6 | 2 | 0 | 1 |
| G1 | 1 | 0 | 1 |
| G3 | 1 | 0 | 1 |
| N2 | 1 | 0 | 0 |
| **Total** | **36** | **4** | |

Tag occurrences, case-bound vs whole-output: G1 1/1, G2 8/9, G3 1/1, G6 2/4, N2 1/1, M1 1/2, T1 8/9,
T2 7/9. Three NO-TAG rows carry no tag even in whole output: CPUStartupAffinity,
CPURootedGatherParallelCopyReplay and CPUShardRouteCollective.

**DGX NEON_I8MM run** (same SHA, `LLAMINAR_CPU_ISA=NEON_I8MM`): Unit 29/985 and preflight 102/2373
fail, against NEON's 29/984 and 101/2344.

- **Name-set observation:** the I8MM failing-name set equals the NEON set plus
  `V2_Integration_CPUShardFusedFinalAddPreweight_neon_i8mm`.
  - That extra row fails at `Test__CPUShardRouteCollective.cpp:331` (preweight assertion).
  - It is NO-TAG, the same as its `_scalar` and `_neon` siblings.
  - Its cause is **open**.
- **Scoped observation:** 28 of the 29 I8MM-named tests in that run passed; the one failure is the
  row above.
- Nothing here qualifies the I8MM envelope beyond those named tests.

## 4. Product gaps (ticket candidates, filed through arm-perf)

Source anchors are unchanged between 39c24d8b4 and a27c7ed4f (see header).

| ID | Severity | Source-evidenced defect | Observed in runs | Not established |
|---|---|---|---|---|
| G1 | HIGH | `NUMAAllocator.cpp:43` `constexpr PAGE_SIZE = 4096` drives `aligned_alloc` alignment, `madvise(MADV_DONTNEED)`, the first-touch stride and `move_pages`. On a 64K kernel a pointer that is only 4K-aligned makes `madvise` fail with EINVAL, and the allocation returns null. **Static hazard:** if a pointer happens to be 64K-aligned with a range under 64K, the kernel applies DONTNEED to the whole 64K page, which **can** zero a neighbouring allocation's data. | EINVAL → null: `V2_Unit_NUMAAllocator` (rerun; logger line "Could not revoke 4096 bytes … Invalid argument", whole-output) | No run observed corruption |
| G2 | HIGH (scope) | `CPUBackend::allocate` aligns to 4096, then calls `mbind(MPOL_BIND)` on that range. EINVAL unless the pointer is 64K-aligned. **Static hazard:** when it is 64K-aligned, the policy covers the whole runtime page and **can** bind memory belonging to a neighbouring allocation. | `mbind … errno=22` and workspace allocation failure in 8 case-bound / 9 whole-output rerun tests | Neighbour mis-binding not observed |
| G3 | MED | **API defect:** `DeviceRegistry::discoverCPU` registers every `/sys/devices/system/node/nodeN` as a CPU device, including CPU-less and zero-memory nodes (DGX nodes 1–8; probes), in `readdir` order. | `V2_Unit_DeviceRegistry`: first CPU device has capacity 0 | Which planner or runtime paths consume these entries |
| G4 | LOW | `MoEOverlayNodeLocalDeviceControllerFabric::planLayout` is called with a literal `4096u` "first-touch granularity" (`…Fabric.cpp:897`). Owner byte ranges stay non-overlapping, but on 64K pages adjacent owners can share one runtime page. Per-owner first-touch page ownership is therefore not guaranteed. | Source only | Any measured effect; moot on Grace's single CPU node |
| G5 | MED (maxim 29) | `NUMATopology::detectLocalNUMANode` falls back to `local_numa_node = 0` with `detection_succeeded = false` and a WARN "assuming node 0". The defect is **misleading synthetic metadata and log text**. The checked consumers reject failed detection: `RuntimeInitPhase.cpp:59,143` and `ClusterInventoryGatherer.cpp:262`. | WARN on DGX and Mac | That any consumer accepts node 0 |
| G6 | MED | `ExpertPreparedMemoryGeometry::cpuProjectionBytes` prices CPU expert storage as Heap with a 4096 minimum; Heap is promoted to the runtime page only for payloads of at least one page. A NUMA-node `CPUStorageRoot` materializes as an AnonymousPageMapping rounded to the runtime page. The admission **can** therefore be smaller than the mapping for sub-page payloads whose 4096-rounded size is below the page, i.e. ≤ 61440 B on 64K. A payload of 61441–65535 B rounds to 65536 either way. PMA's rejection is a separate, correct check. | "live claim exceeds admitted routed_expert_weights … admitted=4096 requested=65536" in 2 case-bound / 4 whole-output rerun tests | Size distribution of real model projections |
| G7 | MED | `NUMATopology::detectViaProcfs` decides a CPU's node by **substring** search in the node's `cpulist` and scans only nodes 0–7. It misses CPU 9 in `"0-71"` and **falsely accepts** CPU 1 for `"10-19"` and CPU 2 for `"0-1,12"`. This is a host-independent parse defect. | ClusterInventoryBootstrap N2 on the rerun (hwloc also returned −1) | Which hosts reach the procfs path in production |

**Benign (not gaps):**

- `CPUBackend` and `MmapRegion` touch strides of 4096 only cause redundant first-touch stores on 64K
  pages.
- `MmapRegion`'s `MADV_HUGEPAGE` and the immutable-row validator are page-size correct.
- The `ExpertPreparedMemoryGeometry` 4096 minimum alignment is benign for payloads of at least one
  page. G6 covers the sub-page case.

**Withdrawn:** G8 (activation arena). The ERROR line comes from the passing deliberate rejection test
`CPUGraphArenaDeclarations.SmallActualSchemaMaterializesExactlyAndRejectsShortOwnerBudget`.

**Withdrawn:** G8 (activation arena). Its ERROR line came from the passing deliberate rejection test
`CPUGraphArenaDeclarations.SmallActualSchemaMaterializesExactlyAndRejectsShortOwnerBudget`. That
withdrawal is one instance of the whole-output problem, which §2's case binding now addresses
generally.

## 5. Mac record (a27c7ed4f; NUMA-less kernel, 4K pages, no published cache sizes)

Launch: SYS_NICE, seccomp unconfined, 16 GiB shm. NEON_DOTPROD envelope, Integration build, private
ccache. Unit 957/984 and preflight 2296/2374 passed: 105 failing tests, 28 with `x86_same_name`.

| Primary diagnostic tag | Tests | x86_same_name | failed-case-bound |
|---|---|---|---|
| N1 | 32 | 1 | 31 |
| N2 | 25 | 2 | 13 |
| NO-TAG | 19 | 10 | 14 |
| E3 | 9 | 9 | 0 |
| X1 | 8 | 0 | 0 |
| F1 | 4 | 4 | 0 |
| V1 | 3 | 0 | 0 |
| E2 | 2 | 2 | 0 |
| V2 | 2 | 0 | 0 |
| N3 | 1 | 0 | 0 |
| **Total** | **105** | **28** | |

- **NO-TAG (19).** 9 carry an N1 whole-output tag (logger-line evidence only). 10 carry no tag at all:
  MoEForbiddenDependencyScan, ProductionParityCampaigns, CPUNativeQuantisedGroupedSealedPlan,
  NativeQuantisedCommonDispatchPolicy, CPUShardRouteCollective, DevelopPRGateContract,
  CPUStartupAffinity, and CPUShardFusedFinalAddPreweight_scalar/_neon/_neon_dotprod.
- **V1 (3).** These timed out under the parallel gate and passed once each standalone
  (`timeouts-standalone-a27c7ed4f.log`). VM contention is a plausible reading, not an established cause.
- **V2 (2).** The MTP benchmark-matrix suite took 34 s standalone against its 30 s budget. That is an
  observed duration, not a cause.
- **X1 (historical, recorded).** These 8 cells failed at a27c7ed4f because their executable was not
  built: the unguarded AVX2 `.inc` from `cc502b95a`. The source fix is `f69446e3c`.
  - Closure is separate rerun evidence: all 8 built and passed at `f69446e3c` plus the two
    Bundle-1 test-only commits (`01-tip-f69446e3c.log`).
  - That rerun covers those 8 cells only, not a full gate.

Cross-host reading (tags, not causes):

- N1/N2/N3 occur only on the NUMA-less Mac.
- G1/G2/G6 text occurs only on 64K-page Grace.
- E2/E3/F1 occur on both ARM hosts. Whether x86 shares them cannot be seen in the WSL logs.

## 6. Evidence

Raw logs, relative to `/tmp/armqual-b1`:

| Path | Content | SHA-256 |
|---|---|---|
| `raw/dgx/unit_test3.log` | DGX canonical Unit | `d835e47f…` |
| `raw/dgx/preflight.log` | DGX canonical preflight | `21e55c1f…` |
| `raw/dgx/unmasking/unmask_rerun.log` | DGX changed-launch rerun | `27b2a803…` |
| `raw/dgx/unmasking/unmask_cmd.txt` | Rerun command | `30a92faf…` |
| `raw/dgx/unmasking/probes_b1.txt` | Station probes | `ffb04ec0…` |
| `raw/dgx/i8mm/unit_i8mm.log` | DGX I8MM Unit | `59e593f3…` |
| `raw/dgx/i8mm/preflight_i8mm.log` | DGX I8MM preflight | `24c87a16…` |
| `raw/wsl/unit.log` | x86 failing names, no failure output | `f736ea26…` |
| `raw/wsl/preflight.log` | x86 failing names, no failure output | `442daceb…` |
| `mac/unit-a27c7ed4f.log` | Mac Unit | `d561d7c2…` |
| `mac/preflight-a27c7ed4f.log` | Mac preflight | `cd8b31f2…` |
| `mac/build-a27c7ed4f.log` | Mac `-k 0` build | `f27ad5ca…` |
| `mac/timeouts-standalone-a27c7ed4f.log` | Standalone reruns of the timeouts | `d22b6828…` |
| `01-tip-f69446e3c.log` | X1 rerun at the fix | — |

Mac probes: `mac/env-probe-a27c7ed4f.log`, `mac/env-probe-seccomp-split.log`, `mac/env_probe.c`.

Classifiers. Revision 1 is in the tracker at `docs/v2/projects/2026-11/arm-perf-review/evidence/bundle1/`: `classify.py`, `classification-39c24d8b4.csv`,
`classify_mac.py` and `classification-mac-a27c7ed4f.csv`. They are **superseded**: they used
base-by-name attribution and whole-output tags.

**To archive (revision 3, authoritative):**

| File | SHA-256 | Notes |
|---|---|---|
| `15/classify_v3.py` | `6aeda387…` | |
| `15/classification-v3.csv` | `4c6f2e3a…` | One row per (run, test): `x86_same_name`, `evidence_scope`, `tags`, `whole_output_tags`, `primary` |
| `15/extract.py` | — | Shared block splitter |

Revision 2 (`classify_v2.py`, `classification-v2.csv`) is not archived; it is superseded by v3.

## #146 recheck (2026-11-10) — rev4 items
- R1: 'scoped' mislabeled 'case-bound' (classify_v3.py:66-67 whole-output when
  no FAILED marker) + CSV split evidence (G2 8 scoped/7 bound/9 whole; G6 2/1/4;
  N2 1/0/1) + doc:165 + G2/G6 observed columns vs correct table -> use
  SCOPED/WHOLE labels or exact splits; narrow stderr claim to 'failed-gtest
  TEXT segments' (all segment lines appended; marker position = selection
  rule, not logger-proof; incomplete/no-marker output weaker incl. aborted/
  timed-out); no row reclassification without observations.
- R2: machine artifact 'verdict' column still carries ENVIRONMENT(launch/VM)/
  VM budget/PRODUCT GAP despite prose withholding -> rename heuristic
  category + cause=not-established unless supported + docstring:13-15.
- Cleanup: duplicate G8 withdrawal; stale archive notes; anchor list +=
  RuntimeInitPhase.cpp + ClusterInventoryGatherer.cpp.
- #146 = substantive ACK otherwise (287 unique rows; counts reconcile; 9
  anchors byte-identical + 2 consumers unchanged; G1-G7 wording + I8MM
  name-set/open-cause addressed; X1/rerun separate). #145 HOLD independent.
