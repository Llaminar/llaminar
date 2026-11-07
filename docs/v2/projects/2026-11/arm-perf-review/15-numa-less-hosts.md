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
