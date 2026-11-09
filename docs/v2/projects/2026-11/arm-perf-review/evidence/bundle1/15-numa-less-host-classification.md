# Idea 15: known-failure classification on NUMA-less and 64K-page ARM hosts

Status: classification record. Report only, no production change. Gap tickets go through arm-perf.
Source anchors refer to `39c24d8b4` (the gated tree). The tracker declares `7d1a94479` and
`a27c7ed4f` content-equivalent for CPU-only code. Every count below is reproduced by
`classify.py` from the raw logs listed in §6.

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

AVX512, GCC 14, same SHA. It serves as the base reference: a failure that also occurs on x86 at
the same SHA is not attributable to the ARM host.

## 2. Classes and verdicts

| Class | Meaning | Verdict |
|---|---|---|
| B0 | Also fails on x86 WSL at 39c24d8b4, or is the `_neon`/`_neon_i8mm` twin of a failing base `_scalar` cell | BASE |
| S1 | Container seccomp: `mbind`/`set_mempolicy`/`get_mempolicy` EPERM, Open MPI hwloc "failed to bind memory" | ENVIRONMENT (launch) |
| T1 | Tests pass `ImmutableRowTableStoragePolicy::forStorage(…, 4096, …)` mmap windows. The production default is 64 KiB (`kImmutableRowDefaultWindowBytes`) and the validator's fail-fast is correct. | TEST ASSUMPTION |
| T2 | 4K-host byte literals or caps (4491052224, 8392704, `recordedFragmentBytes`, exact 49×4096 slices, `< 31` ngram boundary, staging cap 8192). Production rounds through the sysconf page in one authority, and its own actual==estimate checks pass. | TEST ASSUMPTION |
| M1 | Test iterates every NUMA node, including CPU-less ones. The production rejection "Topology exposes no CPU on requested NUMA node" is correct. | TEST ASSUMPTION |
| E1 | oneDNN origin-probe git behaviour in the devcontainer | ENVIRONMENT (container git) |
| G1–G7 | See §4 | PRODUCT GAP |

## 3. Results

**DGX canonical run** (default seccomp), NEON at 39c24d8b4. 130 DGX failures (Unit 29 + preflight
101), plus WSL-only rows for completeness, give 134 rows. None are unclassified.

| Class | Rows |
|---|---|
| B0 base | 32 (all 14 WSL Unit + 16 WSL preflight + `_neon` twin + `_scalar`) |
| S1 seccomp | 41 |
| T1 mmap window | 40 |
| T2 4K literal | 17 |
| E1 git | 2 |
| G1 / G3 (primary) | 1 / 1 |

Two WSL-only rows need explaining:

- `ParityFixtureShardOwnership` is the stale-ccache artifact recorded in ideas07-25.
- `MoEOwnedSharedBackingNativeStartup` fails on WSL and passes on DGX.

**DGX unmasked rerun**

The rerun covers the S1+P1 set: 52 tests, `--cap-add SYS_NICE --security-opt seccomp=unconfined`,
same image and build, default 64 MiB shm. Result: 16 PASS, 36 FAIL.

- The 16 that pass were seccomp-masked only. They are 9 `CPUJoint*`, 2 `CPUFragment*`, 1
  `CPUJointBufferedSourceEconomy`, 6 `FrontendAutomaticSelection*_MPI` and `Qwen35MoEGraph`.
- The 36 that still fail, by primary class:

| Class | Rows |
|---|---|
| B0 base | 4 |
| T1 | 9 |
| T2 | 9 |
| M1 | 2 |
| **G2** | **9** |
| **G6** | **4** |
| G1 | 1 |
| G3 | 1 |
| G7 | 1 |

Each row's deciding line is in `classification-39c24d8b4.csv` (columns `unmasked`,
`unmasked_classes`). The seccomp mask hid most product gaps: in the canonical run EPERM fired before
the 64K-page `madvise`/`mbind` was reached.

**DGX NEON_I8MM run**

| Run | Unit | Preflight |
|---|---|---|
| I8MM | 29/985 | 102/2373 |
| NEON | 29/984 | 101/2344 |

The failure set is identical to NEON. The only extra entry is `CPUShardFusedFinalAddPreweight_neon_i8mm`,
the I8MM twin of the base NaN-corpus defect. There are **no I8MM-specific failures on Grace**, and all
newly registered I8MM cells pass.

**Mac Docker VM at a27c7ed4f**

Run with the kernel-truth launch: SYS_NICE, seccomp unconfined, 16 GiB shm, NEON_DOTPROD,
Integration, `SHARED_CORE=ON`, private ccache. Results: Unit 957/984 and preflight 2296/2374. All
105 failures are classified; see §5.

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

Classified by `classify_mac.py` into `classification-mac-a27c7ed4f.csv`: 105 failures (Unit 27 +
preflight 78), none unclassified.

| Class | Rows | Verdict |
|---|---|---|
| B0: name fails on x86 WSL at 39c24d8b4 (or the `_neon`/`_neon_dotprod` twin of a base `_scalar` cell) | 30 | BASE |
| N1: NUMA binding/policy required ("libnuma reports NUMA unavailable; NUMA binding is required", "NUMA policy APIs are unavailable", "CPU root first-touch placement failed", node −1) | 41 | PRODUCT GAP, known: idea 15 proposal 1 (typed single-memory-domain endpoint) |
| N2: multi-rank inventory or rank-local backend requires NUMA affinity | 20 | PRODUCT GAP, known; shares the G5/G7 detection path |
| N3: cache sizes unpublished ("Streaming CPU measurement requires observed aggregate cache coverage") | 1 | PRODUCT GAP, known: idea 15 proposal 2 (measured cache geometry) |
| X1: build break, the 8 fused-completion NEON/NEON_DOTPROD cells. The target does not compile on aarch64: unguarded `__m256` in `CPUNativeQuantisedUnsignedWeightDotLifecycle.inc:90-94`, from develop commit `cc502b95a`. | 8 | BUILD BREAK (reported to Main) |
| V1: slow VM. Load timeouts under the parallel gate; pass standalone in 2–8 s. | 3 | ENVIRONMENT (VM) |
| V2: slow VM. The MTP benchmark-matrix suite takes 34 s against its 30 s budget (idea 01-4). | 2 | ENVIRONMENT (VM budget) |

Notes:

- The base attribution is by name at 39c24d8b4, while the Mac ran at a27c7ed4f (declared
  content-equivalent for CPU code). Several base rows fail on the Mac for a different reason than on
  x86. For example, `ModelKLDCanary` cannot find `docker` in the container. These rows stay BASE
  rather than being re-attributed.
- The Mac VM launch matters (§1). Under Docker's default seccomp, N1 failures surface as EPERM
  instead of ENOSYS.
- Cross-host view:
  - N1/N2/N3 are the NUMA-less and cache-unpublished classes this idea scopes. They do not appear on
    Grace, which has NUMA and published caches.
  - G1/G2/G6 are 64K-page gaps that do not appear on the 4K Mac.
  - G3 (CPU-less/zero nodes) and G7 (procfs substring parse) are latent on any host with that
    topology.

Evidence (SHA-256 prefixes):

| File | Content | SHA-256 |
|---|---|---|
| `mac/unit-a27c7ed4f.log` | Unit run | `d561d7c2…` |
| `mac/preflight-a27c7ed4f.log` | Preflight run | `cd8b31f2…` |
| `mac/build-a27c7ed4f.log` | Final `-k 0` build | `f27ad5ca…` |
| `mac/timeouts-standalone-a27c7ed4f.log` | Standalone reruns of the timeouts | `d22b6828…` |

Also kept: the two earlier aborted build attempts, `build-…attempt1-oom.log` (my concurrent 09
compiles) and `build-…attempt2-break.log`.

## 6. Evidence

Paths are relative to `/tmp/armqual-b1`; archive into the tracker.

| Path | Content | SHA-256 |
|---|---|---|
| `raw/dgx/unit_test3.log` | DGX canonical Unit | `d835e47f…` |
| `raw/dgx/preflight.log` | DGX canonical preflight | `21e55c1f…` |
| `raw/dgx/unmasking/unmask_rerun.log` | DGX unmasked rerun | `27b2a803…` |
| `raw/dgx/unmasking/unmask_cmd.txt` | Rerun command | `30a92faf…` |
| `raw/dgx/unmasking/probes_b1.txt` | Station probes | `ffb04ec0…` |
| `raw/dgx/i8mm/unit_i8mm.log` | DGX I8MM Unit | `59e593f3…` |
| `raw/dgx/i8mm/preflight_i8mm.log` | DGX I8MM preflight | `24c87a16…` |
| `raw/wsl/unit.log`, `raw/wsl/preflight.log` | x86 base reference | — |

Other files:

- `mac/env-probe-a27c7ed4f.log`, `mac/env-probe-seccomp-split.log`, `mac/env_probe.c`: Mac probes.
- `15/extract.py`, `15/classify.py`, `15/classification-39c24d8b4.csv`: the DGX/WSL classifier and
  its output, reproducible from the raw logs.
- `15/classify_mac.py`, `15/classification-mac-a27c7ed4f.csv`: the Mac classifier and its output.
