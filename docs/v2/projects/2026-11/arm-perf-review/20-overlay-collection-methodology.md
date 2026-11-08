# Idea 20 — Overlay collection methodology (microbench wins that lose in-model)

Status: **Open** (design/verification methodology; arises from idea 12's
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
