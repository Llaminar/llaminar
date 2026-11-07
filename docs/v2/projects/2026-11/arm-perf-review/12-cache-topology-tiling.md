# Idea 12 — Cache-topology-derived tiling / prefetch on GB10

Status: **Partial**. GB10 exact overlays are installed (`144ece037`), FA2 uses
an intentionally cache-independent canonical-span policy, and observed
topology/unknowns are retained (`CPUCacheObservation.h`). Class-specific tuning
(and prefetch) remains open.

## Context

- Mac cache line is 128 B; GB10 is 64 B. GB10 has private per-core L2
  (2 MiB X925 / 512 KiB A725) and 8 MiB / 16 MiB L3 sharing domains; the M1 has
  shared L2. Sizes must derive from observed per-host topology, not M1 numbers.
- NativeVNNI 64-column × 32-K block geometry is the prepared ABI (and the
  installer's overlay set), not a cache-line guess.
- `CPUFlashAttentionLaunchPolicy.h` starts from a cache-independent canonical
  span deliberately; no invented private-L2 assumption.

## Remaining ideas

1. Collect per-class evidence (X925 vs A725) for the profitable panel width in
   idea 4 and prefetch distance for the interleaved weight stream — one isolated
   kernel per profiler launch, then fold into the overlay/selection tables
   (not a runtime heuristic that changes bytes).
2. Tune only after both core classes are measured separately; the GB10
   overlays were generated from a 20-worker core-pinned Release build and cover
   M 1..8/12/15/16/31/64 across 120 manifest shapes — the natural harness for
   this.
3. Any new table must stay exact-overlay based (no generic fitting/retraining;
   the plan explicitly allows only exact overlays).

## Exactness / risk

- Low: tables select implementations with identical arithmetic; but a wrong
  assumption on the A725's smaller L2 can be a pure regression — measure, don't
  assume.

## Validation

- Perf harness A/B per core class; byte-auth timed output; unprofiled final
  runs; engine confirmation when the full engine builds.
