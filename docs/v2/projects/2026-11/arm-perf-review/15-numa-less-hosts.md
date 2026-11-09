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

## DGX probes + unmasking + I8MM gate (2026-11-09, worker report 2/2)

- PROBES: node0 72c+506GB; node1 255GB memory-only (cpus EMPTY); nodes2-8
  ZERO-capacity (cpus EMPTY, MemTotal 0) → G3 confirmed. Devcontainer =
  default seccomp, NO CAP_SYS_NICE, 64MB shm (GLM container: same seccomp,
  32GB shm). Mems_allowed_list=0-1 (cgroup) → node1 CPU-less → mbind/
  first-touch targets are CPU-less (true topology nuance).
- UNMASKING (52 tests S1+P1; SYS_NICE + seccomp=unconfined, 70s): 16
  FLIPPED=seccomp-masked (Qwen35MoEGraph; FrontendAutomaticSelection ×7;
  CPUJoint*/Fragment ×8+) — S1 exact; 36 STILL FAIL = TRUE env (NUMAAllocator
  :105 NULL, CPUBackend NULL, CPUShardRouteCollective:331, DeviceRegistry
  capacity-0, TransferEngine:428, MappedHostTransferRegion:1311, Memory+
  Weight/ActivationEstimators, Qwen4Exp family, PLE, Embeddings, ExpertGemm).
  → G1+G2 CONFIRMED true 64K/memory-only (tickets #48/#49 stand).
- I8MM GATE: build EXIT=0 (571s); Unit 953/982 (SAME failing set as NEON,
  diff empty) + NEW census cell CPUFixedEpochGDN_NEON_I8MM PASS; ISA/census
  cells all PASS; preflight 2271/2373 = NEON's 96 + 1 new
  (CPUShardFusedFinalAddPreweight_neon_i8mm — same env family :331). 2180
  compile cmds -march=armv8-a+simd+dotprod+i8mm;
  LLAMINAR_COMPILED_WITH_NEON_{DOTPROD,I8MM}=1. NO I8MM-specific regression
  → Grace/Neoverse-V2 = first SMMLA datapoint GREEN.
- MODEL: qwen2.5-0.5b-instruct-q4_0.gguf ACTUAL SIZE 428,730,208 B (409 MB —
  correction: my '~35MB' estimate was wrong; canonical file, only one
  fetched). ModelLoaderRowSlice 6/6 OK (CTest still fails on same-binary
  CPUJointBufferedSource 64K env cases). ONESHOT SMOKE (I8MM build, greedy
  -t 0 -s 42 n=16, 16 threads): PASS — prefill 5 tok; 'Paris, and it is the
  capital of Europe…'; staging 0.4GB/291 tensors; prefix cache 1/1.
