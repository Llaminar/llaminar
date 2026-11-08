# Idea 5 — Heterogeneous (X925/A725) core scheduling on GB10

Status: **Partial**. `33690f9b1` ("claim native NEON projection tiles
dynamically") landed; a typed weighted policy is not implemented and the
residual questions below remain.

## Context

GB10 has 10 Cortex-X925 + 10 Cortex-A725, no SMT, one NUMA node; private L2
2 MiB / 512 KiB respectively, no shared core count. A static equal split lets
the slowest core set the pace; the commit message reports the prefill curve
rising to 9.6x at ten threads and falling to ~4x once A725 cores joined, and
dynamic one-tile claiming restoring most of that.

Current state (`CPUNativeNEONProjectionBatch.h`):

- Tasks are independent output tiles with fixed full-K register trees; which
  worker runs a tile cannot change its bytes. Workshare is
  `#pragma omp for schedule(dynamic, 1)` (~:106) after `33690f9b1`.
- Native/expert partition counts (1 / 16) are independent of worker count;
  early decisions remain byte-exact under rebalancing.
- `Perf__CPUNativeVNNI_ThreadScaling` authenticates output per worker count.

## Measured (2026-11-07)

GB10 Q4_K N=2048 K=4096 decode:

| Placement | Time |
|---|---|
| Static split, 20 cores | 78 us |
| X925-only, 10 cores | 52.7 us |
| A725-only, 10 cores | 153 us |
| Dynamic tiles, 20 cores (`OMP_PLACES=cores`, close) | 44.7 us (27x) |

Prefill M=64 with dynamic tiles at 20 cores is about 1450 us, against 3536 us
for the static split.

Two follow-ups have landed since:

- `85e74615d` orders cluster workers fastest core first;
- the bounded spin (idea 13) removed a 20-thread M=1 regression seen only in
  cold single-width runs.

## Remaining ideas / questions

1. **Weighted policy:** dynamic(1) is a proxy; a typed policy that knows core
   identity (sysfs topology, `CPUCacheObservation`) could assign column shares
   by measured capacity. Any weighting must keep whole output-column
   assignment per core (each column's K program fixed) — never a K repartition.
2. **X925-only / A725-only / mixed runs** with 1..20 physical workers measured
   separately (the handoff requirement is still open).
3. **Cache-topology-aware tile sizes** per core class (see idea 12) — X925
   2 MiB L2 vs A725 512 KiB changes profitable panel width.

## Exactness / risk

- Weighting by complete output columns: byte-exact by construction.
- K- or reduction-tree repartition: not assumed byte-exact; forbidden without a
  full arithmetic proof.
- Other certified leaf trees (FA2 fixed arithmetic spans etc.) must remain
  fixed.

## Measured placement economics (P1 final Perf round, K14-PERF-FINAL-264970E14)

- **Fast-only (X925, 10T) ≈ mixed 20T** for GEMV — the X925-only placement
  matches all-core throughput with half the workers.
- **Slow-only (A725) is 2.4–3.7x worse** than mixed.

So the current `dynamic(1)` all-core policy is not leaving much on the table
for GEMV; the economical win would be a typed policy that avoids putting
work on slow cores when fast cores can absorb it (or a P-core-only mode where
power/thermal matter). No such typed policy exists yet.

## Validation

- Thread-scaling Perf on GB10, M 1..8/12/15/16/31/64, all formats; per-class
  placement runs; unprofiled timing; then engine-level decode/prefill curves.
