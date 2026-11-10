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
