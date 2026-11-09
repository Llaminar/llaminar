# Idea 15 — NUMA-less and cache-unpublished Linux ARM hosts

Status: **Partial**. `a572c3c9c` attributes system memory to a NUMA-less
single-package host (`assignNonNUMAHostMemory`, regression
`V2_Integration_NonNUMAHostMemoryAttribution`). Full serving support is
scoped in the ARM plan (work package J, "NUMA-less and cache-unpublished
Linux ARM hosts").

## Finding

Intel production hosts always publish NUMA node topology and CPUID cache
geometry. ARM has no user-mode cache-size query, and some ARM Linux kernels
publish neither: embedded kernels built without NUMA, QEMU/KVM guests without
PPTT cache entries, and Docker Desktop's linuxkit VM.

On the M1 Max Docker VM:

- `set_mempolicy` returns ENOSYS;
- `/sys/devices/system/node` is absent;
- cache `index*` entries have no `size` file.

`llaminar2` now fails fast there, with precise diagnostics:

- `plan`: `Streaming CPU measurement requires observed aggregate cache coverage`
- `--device cpu`: `CPU shorthand requires observed NUMA affinity`
- `serve --device cpu:0`: `Explicit CPU NUMA target 0 has no physical CPU set`

## Proposed

1. A typed "single memory domain without NUMA policy" CPU endpoint through
   backend binding, allocator placement, first-touch proofs and shorthand
   resolution. Never a fake libnuma.
2. A measured cache-geometry source (bandwidth/latency plateau probe) when
   sysfs publishes no sizes. Never a guessed capacity.
3. A NUMA-less Unit/preflight qualification lane: the Mac Docker VM, which
   currently fails about 60 NUMA-family tests as environment-unqualified.

## Value

This makes the Mac a whole-engine development loop, plus small ARM servers
and VMs. It is not needed for the qualified GB10 target.

## Bundle 1/15 classification (2026-11-09, lane) + tickets filed

- 134 failing rows classified, 0 unclassified: B0 base 32; S1 env seccomp-
  EPERM (DGX devcontainer default seccomp w/o CAP_SYS_NICE; mbind/
  set_mempolicy EPERM — NOT 64K/memory-only as earlier recorded) 41;
  T1 test assumption (hard-coded 4096 mmap windows; prod 64K passes;
  validator fail-fast correct) 40; T2 test assumption (4K byte literals) 17;
  E1 env oneDNN origin-probe 2. Benign: touch strides; min-alignment
  promoted by AlignedVector.
- PRODUCT GAPS → GitHub issues: #48 G1 NUMAAllocator 4096 (HIGH; DONTNEED
  EINVAL/null + SILENT 64K-round zeroing); #49 G2 CPUBackend mbind 4096;
  #50 G3 DeviceRegistry CPU-less nodes (cpus[0] capacity 0); #51 G4 MoE
  planLayout 4096 granularity; #52 G5 NUMATopology silent node-0 fallback
  (maxim 29).
- Station probes (numactl/node loop/docker inspect/Mems_allowed) + approved
  UNMASKING rerun (S1+P1 ~42 tests; SYS_NICE + seccomp=unconfined) in
  flight → will refine G1/G2 reproduction + S1 true answers.
