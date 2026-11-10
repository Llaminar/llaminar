# Idea 20 — Overlay collection methodology (microbench wins that lose in-model)

Status: **ADDRESSED** — coverage methodology finalized via Bundle-1; B5/B7 fixed; B8 deferred to the next overlay-table change. (2026-11-10 status flip)
verdict).

## Context

The 2,519-key GB10 exact NEON overlay table (`144ece037`) was selected by
20-worker **microbenchmark** keys (two replicas, byte-exact, ~20–25% kernel
wins on selected keys). Head A/B (X `15f87f755` with table vs Y `d397154da`
with table emptied, alternating 2×2, mode-asserted) confirmed a **net
regression**; the table was retired in `264970e14` (empty table; resolver,
collector and generator kept). Evidence is in `00-measured-baseline.md` and
idea 12.

## Follow-up idea

Microbench-selected keys (Pair tile, more chunks at 20 workers) win in
isolation but **lose in-model**, probably from cache/concurrency effects:
other stages' data gets evicted, and more tasks per region means more barrier
work on mixed cores (ideas 5/13).

Rule going forward, until proven otherwise:

- Any future overlay table must be **collected from in-model context** (warm
  caches, concurrent stages, real per-layer shapes and worker counts as they
  actually run), or
- **validated end to end per format** before install — kernel-level wins alone
  are not admission evidence (matches `AGENTS.md`: kernel-only wins are not
  proof; measure production paths).

## Validation

- A/B with table present/absent per format on GB10 Release; text byte-identical
  as the correctness gate; end-to-end tok/s (mean±sd) as the economy gate.

## ISA identity gap in the overlay tooling (2026-11-08, arm-i8mm)

- `Perf__CPUNativeNEONExactOverlay` stamps `isa = isaLevelName(activeISALevel())`
  — an I8MM build writes **"NEON+DotProd"**, and the runtime overlay key is
  `(codebook, superblock, n, k, m, workers)` with **no ISA dimension**.
- So an SDOT-collected and an SMMLA-collected corpus are indistinguishable,
  and the generator's mixed-ISA guard cannot fire. Harmless today (table
  retired in `264970e14`) but REQUIRED before any I8MM overlay collection:
  stamp `compiledNativeVNNIBuildISAName()`, and bind installed tables to their
  build envelope (and add the envelope/ISA dimension to the runtime key).
- Owner: arm-i8mm lane (collector stamp goes in its commit 2); the rest is
  future-work per this idea.
