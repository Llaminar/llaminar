# Idea 1 — Local kernel-benchmark A/B harness

Status: **Partial**. The `Perf__` kernel suite exists and is portable; the gaps
below are worth closing before the larger kernel changes land.

## Context

The operator wants performance work tested on local hardware only (M1 Max Mac +
two DGX Spark GB10 hosts); no ARM CI lane. The A/B loop is: baseline run →
change in a separate worktree build → identical run → compare speed, PerfStats
and byte-identical output. The repo already has the model for this:

- `Perf__CPUNativeVNNI_ThreadScaling.cpp`, `_Sweep.cpp`, `_DispatchSweep.cpp`
  are intrinsic-free and ARM-aware (`tests/v2/performance/kernels/cpu/native_vnni/`),
  and time output is authenticated against serial one-worker rows.
- `Perf__CPUNativeVNNI_GEMV.cpp` still contains x86 intrinsics and is not a
  portable ARM harness.
- Engine-level A/B (`llaminar2 benchmark --config plan.json
  --benchmark-json-output a.json`) is not yet possible on ARM because the full
  engine (app/gate target) is not built/qualified yet.

## Already available

- Projection thread scaling (1..N workers), format sweeps, dispatch sweeps,
  byte-authenticated timings; the ARM session already used these on an M1 Linux
  VM and on GB10 (see `144ece037`, `408ff90e5` commit messages).
- Warmup/repeat/cache-clear semantics inside `BenchmarkMode`/`BenchmarkRunner`
  for engine-level runs later.

## Proposed additions (each small, self-contained)

1. Per-encoding M=1 (decode) micro-benchmark with GB/s per core reported against
   the actual working set (L1/L2/DRAM phases separated), not a nominal ceiling.
2. RSS/load-peak probe around prepared-weight loading (`int8_flat` rotation
   path, mmap reclaim timing) so memory ideas below have numbers.
3. A committed baseline-JSON convention (SHA-tagged host/build + counters) so
   any follow-up run can diff against the last recorded baseline without a
   manual table.

## Exactness / risk

- No arithmetic change; timings only. Output bytes must still be authenticated
  (existing pattern).

## Validation

- Runs on the Mac and on GB10; 1..20 workers; X925/A725/mixed where applicable.
- The harness is a prerequisite for evaluating ideas 4, 6, 7, 8, 9, 12.
