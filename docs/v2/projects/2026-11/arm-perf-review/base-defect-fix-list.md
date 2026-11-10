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
