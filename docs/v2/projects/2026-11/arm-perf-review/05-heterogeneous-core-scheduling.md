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

- **Fast-only (X925, 10T) ≈ mixed 20T** for GEMV decode: 217.7 vs 226.4 µs —
  the 10 X925 cores alone saturate decode; adding A725 cores gives nothing
  (slightly worse).
- **Grouped M=15: mixed 20T still wins 11–12%** (1569 vs 1750 µs) — the
  big-K grouped/prefill shapes do use the extra cores.
- **Slow-only (A725) is 2.4x (GEMV) to 3.7x (grouped) behind fast-only.**

Suggested typed placement policy (no implementation yet):
- **M=1 decode on X925 only** — frees A725 cores for other work and avoids
  straggler barriers on mixed-core teams;
- **grouped/prefill on all 20**.

This is a decision for a typed policy with per-shape placement evidence, not
another schedule heuristic.

## Validation

- Thread-scaling Perf on GB10, M 1..8/12/15/16/31/64, all formats; per-class
  placement runs; unprofiled timing; then engine-level decode/prefill curves.

## Placement evidence update (2026-11-08 perf window — task-count matters)

- BIG shape (17408x5120): mixed-20 beats X925-10 for decode (Q4_K 545 vs 573 µs) AND grouped M=15 (2349 vs 3075 µs).
- SMALL N=512 bundle: mixed-20 is much worse for grouped M=4 (IQ2_S 52.5 vs 19.3 µs; Q4_K 32.5 vs 11.2 µs) — only ~8 tasks, 20 workers to wake.
- Conclusion: the placement choice must depend on the projection's TASK COUNT (columns × chunk structure), not only M=1 vs grouped — the C8 default table is revised accordingly (fastest-class only when the task count is small AND M=1; all-workers for big shapes).
- Nuance vs the earlier P1 numbers: 'fast-only ≈ mixed for GEMV' was a different shape (2048x4096); at 17408x5120 mixed wins decode. The typed policy needs per-shape-class evidence, not a single switch.

## Decode projection composition (2026-11-08 diagnostic probe, 35B)

- no-MTP: **66%** start a standalone OpenMP region, **34%** enter an
  already-running team; **99%** are M=1; **17%** have FEWER tasks than workers.
- MTP: **65%** standalone.
- Implication for C7: a standalone-region placement policy (fastest-class
  width + task-count cap) covers ~2/3 of decode projections; the rest need the
  in-team path (measured separately, idea 13 caveat).
- Next: typed placement mechanism implementation (fastest-class width +
  task-count cap, CLI flag, PerfStats counter) — pending window measurement.

## Typed placement implemented (2026-11-08, 56158b57a)

- CPUWorkerPlacement = core-free authority; binding publishes the
  fastest-class width by MIDR (capacity fallback; GB10 → 10).
- `--cpu-decode-placement auto|all-workers|fastest-class`: auto never wakes
  more workers than independent tasks; fastest-class additionally limits
  single-row decode to the fastest class and fails at startup without the
  bound layout. Applies to STANDALONE regions only (C7) via
  OMP_WORKSHARE_REGION_WIDTH. PerfStats `cpu_native_neon_team_placement`.
- Config document omits auto → existing plans/expected_config stay
  byte-identical; explicit values round-trip.
- Tests (all green on .80, preflight): V2_Integration_CPUWorkerPlacement,
  V2_Integration_CPUNativeNEONTypedPlacement (byte equality across
  placements), MIDR class counting on recorded GB10 layout, parser round trip.
- OPEN QUESTION: `auto` default changes behaviour (task cap) before any engine
  measurement — window 2 (all-workers vs auto vs fastest-class, 35B/27B MTP
  off/dyn) decides whether auto keeps the cap / adopts fastest-class.
