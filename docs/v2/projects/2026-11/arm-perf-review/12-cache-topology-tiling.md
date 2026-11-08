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

## End-to-end overlay transfer (final A/B/C, verified 2026-11-08)

Verified per-commit runs on GB10 (35B Release, decode tok/s, n=256 greedy):

- A `878a33fea` (baseline): MTP dyn **18.18±0.40**, no-MTP **14.14**, prefill
  24.0–24.7.
- B `144ece037` (+2,519-key overlay table): MTP dyn **17.45±0.21** (−4%),
  no-MTP **13.27** (−6%), prefill 22.6–22.8 (−7%).
- C `268cd81de` (pooled bundle **plus** the same overlays): MTP dyn 27.19±0.18,
  no-MTP 20.48, prefill 37.1–37.8 — so the head is still carrying the overlay
  cost on top of the bundle gains.

Kernel-level wins on selected keys were ~20–25%, but the 20-worker
microbenchmark keys do **not** transfer to in-model execution: end to end the
overlays are confirmed to **regress ~4–7%** (warm caches, concurrent stages).

**Follow-up:** A/B at the head with the overlay table disabled; if confirmed,
either retire the table or re-collect overlays from **in-model context** (warm
caches, concurrent stages, per-layer shapes as they actually run).

## Verdict (2026-11-08, evidence K14-OVERLAY-AB-HEAD)

Head A/B confirmed the regression (X `15f87f755` with table vs Y `d397154da`
with table emptied, alternating 2×2, mode-asserted): removing the table gives
35B MTP decode 27.35±0.06 → **28.51±0.29** (+4.2%), no-MTP 20.54 → **20.95**
(+2.0%), prefill **+9–11%**, 27B unchanged, text byte-identical.

**The exact-overlay table is retired** in `264970e14` (empty table; resolver,
collector and generator retained). Future overlay collection must follow idea
20 (in-model context or per-format end-to-end validation) — microbench-selected
keys win in isolation but lose in-model.

## Exactness / risk

- Low: tables select implementations with identical arithmetic; but a wrong
  assumption on the A725's smaller L2 can be a pure regression — measure, don't
  assume.

## Validation

- Perf harness A/B per core class; byte-auth timed output; unprofiled final
  runs; engine confirmation when the full engine builds.
