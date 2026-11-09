# Idea 28 — IQ4_XS M=1 multi-thread regression after the metadata seal (R1)

Status: **Open (investigation)**. Found in addon window (b), 2026-11-09.

## Finding

Window (b): base `302318f1e` vs R1 `035d1d948` (Release NEON_I8MM, gcc-14,
production gemv path, ABBA, median of two launches per arm, 31 iters; binding
OMP_PLACES=cores OMP_PROC_BIND=close).

- **IQ4_XS 17408×5120 M=1: +6.6% @1 thread, −4.5% @10, −4.2% @20** (both
  launches consistent).
- IQ4_XS 2048×4096 M=1: +5.0 @1, **−4.1 @10**, +1.4 @20.
- IQ4_XS 512×2048 M=1: +11.7/+11.1/+7.3%.
- Everything else grouped/single-thread: **+1–12%** (IQ3_S 512×2048 M=4
  +9.0/+7.0/+7.6%; IQ4_XS M=4/M=15 positive everywhere; IQ3_S 17408×5120
  M=1 +0.9/+2.7/+2.0% etc.).

Claim boundary (lane, honest): byte-identical; **+3.1% 35B MTP-off decode**
(23.31 → 24.04; p50 41.8/42.2→40.5/40.8 ms), neutral MTP (33.94 vs 34.0,
±0.3% within noise incl. one launch outlier; acceptance 0.99 both), prefill
unchanged (~49.5–50 tok/s). NOT claiming economy for the IQ4_XS M=1 cell.

## Suspected cause (within the R1 diff)

- `noexcept` / `__builtin_unreachable` dispatch change (in-loop throw removal);
- the unchecked admitted accessors changing scheduling in the Rows=1 IQ4_XS
  nibble-source-scale path — under bandwidth pressure (17408×5120, multi-
  thread).

## Investigation plan (bounded, ~30 min)

1. Disassembly diff Rows=1 IQ4_XS nibble-source-scale path (base vs R1) —
   scheduling/inlining + dispatch edge.
2. Isolated confirmation ≥5× 31-iter at 10/20t @17408×5120 (raw+p95); if it
   holds, A/B the two candidates (noexcept dispatch vs admitted accessor).
3. Decision rule: codegen-attributable → FIX IN R1 (small amendment); clears
   as artifact/noise → record + close with the item tracked (R2 watch item).

## Evidence

`.80:build/results/w-r1-20261109T031823Z` + `build/wr1-summary.py`.
