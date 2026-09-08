---
name: cpu-tuning
description: Profile and tune Llaminar V2 CPU kernels with Linux perf, hardware-counter and ISA analysis, physical-core scaling, NUMA-aware placement, OpenMP worksharing, and AVX2/AVX-512 validation. Use when asked to find a CPU hotspot, explain low IPC or cache misses, inspect SIMD code generation or register spills, tune CPU GEMM/GEMV/attention/quantization kernels, compare thread counts or ISA regimes, or prove a CPU optimization improves production throughput without breaking parity or deterministic grouped decode.
---

# CPU Kernel Profiling and Tuning

Use an evidence-driven loop: establish the production baseline, isolate one
kernel and geometry, collect counters without contaminating canonical timing,
inspect the hot instructions, change one hypothesis, then repeat the focused
and end-to-end correctness/economy gates.

## Preserve the execution contract

- Keep hot-path state host-owned and persistent. Do not add allocation,
  repacking, format conversion, or topology changes to make an isolated kernel
  look faster.
- Preserve the production arithmetic order. Grouped verifier rows must remain
  byte-identical to serial M=1 decode for every supported format and positive
  thread count.
- Use physical cores as the default worker budget. Treat SMT, cross-socket
  placement, and nested BLAS/OpenMP teams as separate, explicitly measured
  modes.
- Use `OMP_WORKSHARE_REGION` and its companion macros from
  `src/v2/utils/OpenMPUtils.h`; do not create an unconditional nested
  `#pragma omp parallel` region inside a kernel.
- Keep profiler collection separate from canonical latency samples. Never use
  `perf` overhead, sampling runs, or hardware-counter collection as dispatch
  labels.
- For NativeVNNI corpus collection, policy fitting, certification, or install,
  also read [nativevnni-gemm-tuning](../nativevnni-gemm-tuning/SKILL.md). Its
  trainer-owned `perf_event_open` protocol is the production evidence
  authority; use its existing collector/driver rather than inventing profiler
  environment variables. The system-wide `perf` workflow below is for
  diagnosis.

## 1. Define one exact experiment

Record all dimensions that can change the result:

- binary digest and `Release` or `Integration` build type;
- compiled ISA (`LLAMINAR_CPU_ISA=AVX2` or `AVX512`) and runtime ISA route;
- socket, NUMA node, physical CPU list, and OpenMP team size;
- tensor format/codebook, serial M=1 versus grouped-verifier operation,
  `(M,N,K)`, and candidate;
- warmup count, sample count, prompt hash, and execution mode.

Inspect topology before choosing a CPU list:

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
numactl --hardware
```

Do not assume that logical CPUs `0-N` are one physical socket. Derive a
physical-core list from the actual host and use the same list for both
`taskset` and `perf -C`.

Build and measure the production baseline without profiler-only flags:

```bash
cmake -B build_v2_release -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMINAR_CPU_ISA=AVX512
cmake --build build_v2_release --parallel

./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cpu:0 \
  --prompt-file <fixed-prompt.txt>
```

Run enough clean repetitions to establish a noise band. Canonical benchmarks
retain normal MPI bootstrap, pinning, and NUMA setup. A change inside the noise
band is not a win.

Use explicit PerfStats output to prove the production route and host phases:

```bash
LLAMINAR_PERF_STATS_JSON=/tmp/llaminar-cpu-perfstats.json \
LLAMINAR_PERF_STATS_SUMMARY=1 \
./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cpu:0 \
  --prompt-file <fixed-prompt.txt>
```

`LLAMINAR_PROFILING=1` is a deprecated compatibility alias. Do not use it to
change execution topology, and do not treat the legacy
`LLAMINAR_PROFILE_KERNELS=1` table as whole-model attribution.

Prefer a focused performance binary under
`tests/v2/performance/kernels/cpu/` for iteration. Representative targets
include:

```bash
cmake --build build_v2_release --parallel --target \
  v2_perf_cpu_native_vnni_thread_scaling
cmake --build build_v2_release --parallel --target \
  v2_perf_cpu_flash_attention_sweep
cmake --build build_v2_release --parallel --target \
  v2_perf_cpu_flash_attention_tile_tournament
cmake --build build_v2_release --parallel --target \
  v2_perf_gemm_tuning_sweep
```

Search `tests/v2/CMakeLists.txt` for the exact target that owns the kernel; do
not profile a neighboring proxy when the production-shaped harness exists.

### CPU FlashAttention cache-tile policy

CPU FA2 keeps arithmetic and cache blocking separate. The shared scheduler in
`CPUFlashAttentionKernelT.h` always forms canonical 256-row online-softmax
summaries and merges them in ascending K/V order. A physical K/V tile may alter
which rows stream through private cache in one callback, but it must never alter
score indexing, four-row vector groups, summary boundaries, or output bytes.
Do not install a tile optimization that couples those two policies again.

Build and certify AVX2 and AVX-512 independently with Release code generation:

```bash
cmake -B build_v2_release_avx2 -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMINAR_CPU_ISA=AVX2
cmake --build build_v2_release_avx2 --parallel --target \
  v2_perf_cpu_flash_attention_tile_tournament

python tests/v2/performance/kernels/cpu/attention/run_cpu_fa2_tile_tournament.py \
  --binary build_v2_release_avx2/tests/v2/v2_perf_cpu_flash_attention_tile_tournament \
  --family-repeats 3 --max-p95-regret 5
```

Repeat with `-DLLAMINAR_CPU_ISA=AVX512` and a distinct AVX-512 build directory.
Also run that AVX-512 binary with `LLAMINAR_ISA_LEVEL=avx2`; compiler code
generation and runtime algorithm dispatch are independent measured axes, and
the mixed regime must not reuse AVX2-only evidence implicitly. Tournament CSV
and summaries must name both `codegen_isa` and `runtime_isa`.

The driver selects one hardware thread per physical core on one socket, runs
each `(native K/V format, head dimension)` family in a fresh process, and takes
the median domain regret across fresh-process repeats. Its complete compact
matrix is nine native K/V pairings, head dimensions 64/128/256, decode M=1,
grouped M=15, prefill M=128, and all seven compiled physical tiles. Every
candidate is byte-authenticated through the production kernel before timing;
missing candidates, mixed code-generation/runtime ISA profiles, malformed
evidence, and byte mismatches are fatal. Use `--cpu-list` when topology or
governor control requires an explicit socket.

For one isolated profiler experiment, filter the C++ worker directly with
`LLAMINAR_CPU_FA2_TOURNAMENT_FORMAT`, `_HEAD_DIM`, `_M`, `_KV`, and `_TILE`.
Set `_SAMPLES` to a complete forward/reverse candidate-cycle count, keep
`_WARMUP` outside timing, and use `_COLD=1` only for an explicitly cold-cache
question. One `_TILE` worker is profiler evidence, not a regret certificate.
Keep the all-tile integration regressions and the process-isolated policy gate
green after any scheduler, codec, cache rule, or SIMD change.

## 2. Prepare Linux perf safely

Check counter access and available events first:

```bash
sysctl kernel.perf_event_paranoid
perf list
```

If access is denied, change `kernel.perf_event_paranoid` only with authorization
because it is host-wide. Record the current value, temporarily enable access
with `sudo sysctl -w kernel.perf_event_paranoid=-1`, and restore the recorded
value after collection.

Use two attachment patterns:

1. Profile standalone test/performance binaries directly. They do not MPI
   bootstrap, so do not pass `--no-mpi-bootstrap`.
2. When profiling `llaminar2` directly, pass `--no-mpi-bootstrap` so `perf`
   attaches to the compute process instead of the `mpirun` wrapper. Pin the
   selected CPUs and set the OpenMP policy explicitly. This mode is diagnostic
   only because it does not reproduce the full bootstrap placement contract.

Capture all worker activity unambiguously with system-wide collection limited
to the pinned physical CPUs:

```bash
perf stat -a -C <physical-core-list> \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  -- taskset -c <physical-core-list> \
  env OMP_NUM_THREADS=<physical-core-count> OMP_PLACES=cores OMP_PROC_BIND=close \
  ./build_v2_release/llaminar2 benchmark --no-mpi-bootstrap \
    -d cpu:0 -m <model>.gguf --prompt-file <fixed-prompt.txt>
```

Never report the wall time from this profiler-only invocation as the production
benchmark result.

## 3. Use perf stat for counter evidence

Start with portable counters, then add model-specific events that `perf list`
confirms are supported:

```bash
perf stat -r 5 -a -C <physical-core-list> \
  -e cycles,instructions,branches,branch-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  -- taskset -c <physical-core-list> <focused-perf-binary> <focused-args>
```

Reject multiplexed or unsupported counter sets rather than silently comparing
different evidence. Profile one exact kernel/ISA/shape/M candidate per run.

Interpret ratios in context, not as universal pass/fail constants:

| Evidence | Likely interpretation | Next check |
|---|---|---|
| Low IPC with high LLC/DRAM traffic | memory or NUMA limited | placement, first touch, bytes per logical work |
| Low IPC with low cache traffic | dependency, branch, barrier, or front-end stalls | sampled hotspot and annotated ISA |
| High branch-miss rate | irregular dispatch or hot-loop control flow | split aligned and boundary paths |
| High instructions per logical work | excess decode, shuffle, reduction, or bookkeeping | instruction census against the baseline |
| More threads but flat throughput | barriers, false sharing, bandwidth, or too little work per worker | physical-core scaling and `libgomp` samples |

For VNNI kernels, calculate IPC, instructions per logical MAC, cache traffic per
payload byte, and speedup across one thread through the physical cores per
socket. Raw miss counts alone are misleading when candidates execute different
amounts of work.

## 4. Find and annotate the hotspot

Write profiler artifacts outside the repository:

```bash
perf record -o /tmp/llaminar-cpu.perf.data \
  -a -C <physical-core-list> -F 997 --call-graph fp \
  -- taskset -c <physical-core-list> <focused-perf-binary> <focused-args>

perf report -i /tmp/llaminar-cpu.perf.data \
  --no-children --sort=dso,symbol
perf report -i /tmp/llaminar-cpu.perf.data \
  --no-children --stdio --sort=symbol
perf annotate -i /tmp/llaminar-cpu.perf.data \
  --symbol='<hot-symbol>' --stdio
```

Use `--call-graph fp` only when the binary was compiled with frame pointers.
For a profiling-only optimized build, add `-fno-omit-frame-pointer` to the
`Integration` C/C++ flags and keep its timings separate from the canonical
`Release` baseline. If exact production code generation matters more than call
stacks, profile the `Release` binary with an unwinder supported by the host and
quantify lost samples before trusting the call graph.

In annotated assembly, look for:

- stack spill/reload traffic from excessive live vector registers;
- register-to-register ZMM/YMM shuffles that crowd out dot-product work;
- `vpdpbusd`/VNNI or the intended AVX2 sequence on the production-selected
  path;
- hot loads with poor locality or cross-NUMA traffic;
- time in `libgomp` barriers or `pause` rather than in the kernel;
- scalar tails or boundary branches dominating small shapes.

Use `objdump -dC -Mintel <binary>` for a stable instruction census when sample
annotation alone is ambiguous.

### Classify spills by execution regime

Do not treat a compiler spill count as either universally fatal or harmless.
First delimit the actual inner loop by symbol address, branch targets, and
source annotation; prologue saves, scalar pointer slots, exception paths, and
unreached ISA branches are not hot-loop vector spills. Build AVX2 and AVX-512
separately because an AVX-512 build can hide pressure in code that is meant to
represent the 16-register AVX2 regime.

For compute-bound GEMM and grouped-verifier candidates, a vector spill in the
K loop disqualifies the candidate. Keep a no-spill geometry available and
reduce the live set by narrowing the N tile, decoding/loading one subchunk at a
time, scoping broadcasts to their use, or disabling an unprofitable unroll.
Preserve independent accumulator chains so removing a spill does not replace
memory traffic with a long dependency chain.

A streaming GEMV candidate may retain measured spill traffic only when all of
the following are true:

- counters establish that the domain is memory-bound rather than
  front-end/compute-bound;
- a no-spill candidate for the same format, geometry, M, and ISA is measured in
  the same tournament;
- the spilling candidate wins canonical latency beyond the noise band and does
  not increase bytes moved enough to lose at neighboring shapes;
- byte-equivalence and thread/ISA totality gates pass.

Never infer that a spill is acceptable merely because one memory-bound Q6
shape tolerated it. Candidate evidence is domain-specific, and spilling
variants must not be eligible in compute-bound dispatch domains.

### Keep compact-format ILP while shortening decode live ranges

Compact formats such as Q6_K can need more decode state than nibble-LUT or
expanded-INT8 formats. When grouped M=2 work is under-filled, first widen the
column span so one worker owns enough independent dot products; do not create
two separate row launches. Then reduce register pressure inside that wider
tile without collapsing its independent accumulator chains:

- inspect the generated K loop itself, not prologue saves or cold scalar
  branches, before classifying a spill;
- load, decode, consume, and retire one low/high bitplane group before decoding
  the next group;
- combine disjoint bitplanes with masked integer adds when that removes a
  longer merge/shuffle sequence and preserves the exact decoded bytes;
- use the narrowest VNNI width that matches auxiliary reductions such as a
  half-block activation sum, instead of keeping unnecessary ZMM constants live;
- keep one accumulator per output row and column vector, and visit K blocks in
  the same ascending order as serial M=1 decode.

The acceptance gate is simultaneous: byte equality against independent serial
rows across the all-format/runtime-M sweep, no hot-loop spill for a
compute-bound candidate, and a repeated latency win over serial decode. A
lower register count that loses cross-row reuse or changes FP accumulation is
not an optimization.

### Tune NativeVNNI cache tiles from the prepared execution layout

Cache policy must model the bytes consumed by the production kernel, not the
compact source GGUF block. Use `NativeVNNIPreparedFootprint` and include all
live data owned by one physical microkernel tile:

- the exact interleaved weight stride, including compensation, scales, and
  optional minima;
- the Q8 activation blocks for every simultaneously live physical row;
- the FP32 output chunks that remain live while K is accumulated;
- the prepared encoding and asymmetry, because nibble decode, expanded INT8,
  and native dual-scale Q6 have different reuse/compute balances.

Derive capacity and associativity from `CacheInfo`, reserve explicit
associativity headroom, and keep the residency fraction a typed, testable
policy. Unit tests must inject synthetic cache topologies and assert exact
block counts for every physical prepared encoding. Do not use a guessed source
bytes-per-weight ratio or let a benchmark-host cache size become a constant.

Treat cache residency as a tournament parameter, not as a folklore percentage:

1. Cover symmetric/asymmetric nibble, symmetric/asymmetric expanded INT8, and
   native Q6. Source codebooks that map to the same prepared encoding still
   belong in the final all-format correctness sweep.
2. Measure grouped depths on both sides of any proposed mode transition and
   include representative attention, FFN, and terminal-head geometries.
3. Interleave candidates in rotating order. Bracket the automatic candidate
   above and below, retain adjacent odd block counts, and include full K. A
   tournament that only searches one side of a changed default cannot report
   honest regret.
4. Enclose one additional warmed production launch for one exact
   `(format, ISA, M, N, K, candidate)` in the process-owned perf interval.
   Setup, quantization, correctness, warmup, other candidates, and canonical
   timing must remain outside that interval.
5. Prefer a candidate when lower wall time and cycles coincide with an
   explanatory signal such as fewer L2/LLC misses at similar instruction
   count. A raw timing fluctuation without a mechanism is not enough to encode
   a new generic policy.

The focused NativeVNNI cache-tile tournament is
`v2_perf_cpu_native_vnni_dispatch_sweep`, test
`CacheKTilesCoverAllPreparedFootprints`. Select geometry with
`LLAMINAR_CPU_NVNNI_CACHE_TILE_M`, `_N`, and `_K`; select one exact profiler
launch with `_FORMAT`, `_CANDIDATE`,
`LLAMINAR_NATIVE_VNNI_PROFILE_REQUEST_ID`, and
`LLAMINAR_NATIVE_VNNI_PERF_STATS_PATH`. Pin one physical socket and set
`OMP_NUM_THREADS`, `OMP_PLACES=cores`, and `OMP_PROC_BIND=close` explicitly so
NUMA or SMT movement cannot masquerade as a tile result.

One 32-element K block is already the NativeVNNI SIMD/decode unit. Do not round
cache tiles to a larger block multiple unless annotated assembly and repeated
measurements prove a real alignment requirement; odd tile lengths are valid
and can win. Likewise, do not assume one residency regime fits every M: shallow
grouped verification and deeper prefill may have materially different reuse.

Every cache-policy change must keep the AVX-512 and forced-AVX2
`V2_Integration_CPU_NativeVNNI_PrefillCacheKTile_*` gates green. Those gates
compare every K-tiled production result byte-for-byte with independent serial
rows across all formats, odd/even M values, tail N, and tile boundaries. A
faster tile that changes FP parenthesization is ineligible.

## 5. Tune one bottleneck at a time

Apply changes in this order and remeasure after each one:

1. Fix placement, oversubscription, and work partitioning. Do not tune SIMD
   around a bad NUMA or OpenMP configuration.
2. Remove hot-path allocations, packing, conversions, and redundant passes.
3. Improve cache reuse and first-touch locality. Initialize allocations of at
   least 128 KiB in parallel on the workers that will consume them.
4. Reduce barriers and imbalance while preserving required phase ordering.
5. Improve SIMD instruction-level parallelism, interleave independent
   loads/accumulators, and prefetch only when counters prove it helps.
6. Reduce vector-register live ranges when annotation proves spills or shuffle
   pressure. Prefer bounded load/decode/use tiles and multiple independent
   accumulators; more unrolling is not automatically faster.
7. Keep vectorized tail handling explicit: AVX-512, AVX2, narrower vectors,
   then scalar. Validate every tail and runtime ISA route.

Use the nested-safe worksharing form:

```cpp
auto work = [&]() {
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < count; ++i) {
        output[i] = compute(input[i]);
    }
};
OMP_WORKSHARE_REGION(work);
```

Use `OMP_WORKSHARE_REGION_SYNC` only when the caller needs an explicit barrier,
`OMP_WORKSHARE_REGION_IF` for measured size thresholds, and `OMP_SINGLE` for
single-worker work such as an MPI collective. Keep thread-local storage inside
the work lambda so nested and top-level execution have identical ownership.

## 6. Validate correctness, dispatch, and economy

Before accepting a change:

1. Run the focused correctness/integration test for the kernel and add a
   regression for every defect discovered during tuning.
2. Sweep every supported tensor format/codebook, representative `(M,N,K)`
   geometries, all required M values, AVX2 and AVX-512 build/runtime regimes,
   tails, and every positive thread count required by dispatch totality.
3. Run the focused performance binary without `perf` and prove improvement
   beyond its noise band.
4. Run the production `Release` benchmark with normal MPI bootstrap and the
   fixed prompt; confirm the relevant prefill/decode throughput improves.
5. Inspect PerfStats/route counters and prove the intended optimized kernel and
   ISA path actually executed.
6. Retain the exact command, binary/build identity, topology, raw counters,
   samples, and parity result. Do not commit `perf.data` or local result
   directories.

Accept the change only when correctness and production economy pass together.
Do not retain a serial fallback, an exact-shape-only shortcut, or an
uneconomical generic path to hide a failed tuning candidate.
