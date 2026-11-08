# Idea 2 — Shared CPU/GPU (UMA) memory admission on GB10

Status: **Partial**. Correction to the review below: the joint pool is
implemented on the branch in `80985c08e` ("admit integrated-GPU and host
memory as one physical pool").

- CUDA discovery records `cudaDeviceProp::integrated` as the typed
  `PhysicalMemoryBacking::RankHostPool`
  (`backends/cuda/CUDAHardwareMetadata.cu`, `backends/PhysicalMemoryBacking.h`).
- `PhysicalMemoryPlan::fits()` also checks `rankHostPoolViolation()`: per rank,
  CPU plus pool-GPU incremental demand must fit the pool observation
  (`planning/PhysicalMemoryAuthority.h`).
- Dense socket indices and the GB10 recorded topology fixture landed with it.

Still open: a live GB10 CUDA+CPU build and admission run, plus the
`cudaMemGetInfo`-vs-`MemAvailable` observation below. The finding text is kept
as written by the review.

## Finding

On GB10 (DGX Spark), CPU and GPU draw from one LPDDR5x pool, but the memory
authority treats them as independent allocators. Caller-chain evidence at the
reviewed head:

- GPU capacity: `backends/CUDAEnumeration.cu` (`cudaMemGetInfo`), `totalGlobalMem`
  in `backends/cuda/CUDAHardwareMetadata.cu`. No integrated/UMA attribute is
  queried anywhere in the tree (repo-wide search finds no `cudaDevAttrIntegrated`
  or equivalent).
- CPU capacity: `backends/HostMemoryCapacity.cpp` — node meminfo /
  `/proc/meminfo`, available = MemFree + Inactive(file).
- Normalization: `planning/ClusterInventoryGatherer.cpp` copies both into
  separate `rank.cpu` and `rank.gpus[]` entries;
  `planning/RankMemoryPlanInputs.cpp::inventoryForDevice` builds a separate
  `PhysicalMemoryResource` per `(rank, device)` identity.
- Admission: `PhysicalMemoryPlan::fits()` is `all_of(bom.fits())`
  (`planning/PhysicalMemoryAuthority.cpp`); `PhysicalMemoryBOM::fits()` is
  `incrementalBytes <= admission_available_bytes` per resource
  (`planning/PhysicalMemoryBOM.h`). `sameAllocator()` compares (rank, device,
  total, available), so `cpu` and `cuda:0` are never the same allocator.

Consequence: a plan can be admitted with CPU and GPU BOMs each sized against
approximately the full available memory, i.e. up to ~2x the physically free
UMA before a runtime failure. `MoEOverlayLocalCapacityPlanner` re-observes GPU
free memory at runtime (which on UMA sees CPU allocations) but that is after
admission and cannot repair the certificate.

## Proposed change

A typed shared-physical-pool constraint inside the sole PMA:
- one shared capacity whose admitted total = sum of the linked allocators'
  admitted bytes, checked by the aggregate certificate;
- keep per-allocator limits, backing, RAII and residency credits distinct;
- no new secondary ledger; no blind `min()` of capacities; no ID aliasing.

## Exactness / risk

- Planning-only; no inference arithmetic change.
- Must handle: same-rank CPU+CUDA on GB10 vs discrete GPU hosts (no coupling),
  cross-rank (MPI) coupling only where physically true, and admission/residency
  renewal paths.

## Validation

- One GB10 observation of what `cudaMemGetInfo` reports relative to
  `MemAvailable`/page cache is still needed (integrated devices may under-report
  reclaimable cache).
- Device-free planning unit tests with synthetic coupled/discrete resources;
  then a real GB10 admission run.

## Live GB10 observation (2026-11-08, .79 Qwen3.8 run)

Qwen3.8-Flash-Next-UD-Q4_K_XL (~111 GB, 177B MoE) on the 121 GB LPDDR5x
joint pool:

- After the model read, ~40 GB of the unified pool sits in **page cache**;
  the driver's reported free (used for `cudaMemGetInfo`-style capacity)
  **collapses to ~24 GB**, and PMA rejects with `Higher-priority PMA plan must
  fit`;
- Workaround today: `echo 3 > drop_caches` before admission.

Open sub-items for the GB10 workflow:

- Account (or at least allow) **reclaimable page cache** in the joint-pool
  available figure so a one-time model read does not block admission
  (downside: page-cache reclaim pressure on first-token latency — needs a
  deliberate policy, not a silent subtraction);
- Alternatively document a GB10-specific admission knob, and make the failure
  diagnostic say what to do (drop caches / free page cache) instead of a bare
  `Higher-priority PMA plan must fit`.

## Coordination

ARM session owns Task 39; Spark qualification task is blocked on it. This is
the prerequisite for trusting any CPU-side memory saving (idea 8) on GB10.
