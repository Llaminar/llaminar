# Idea 31 — S=4 for the remaining expanded encodings (Q5_K + nibble)

Status: OPEN (2026-11-09; queued via R2 C4 per-encoding width evidence).

## What R2 measured/installed
- S=4 installed for ExpandedInt8-without-minimum (Q8_0) + IQ3SExpandedSourceScale
  (spill-free on GNU-14 AND AppleClang-21; the two encodings that win S4/S2
  ratios: IQ3_S 0.58-0.95, Q5_K 0.60-0.91, Q8_0 0.61-0.94).
- REJECTED for the others by the register witness: ExpandedInt8 WITH minimum
  (Q5_K) — AppleClang spills 4 (BackendNative); Nibble (Q4_K/IQ4_XS) — GNU
  spills 6-33; Q6_K outside the admitted grammar (needs ldr x post-index).

## The opportunity (unlocks):
- Q5_K: gave up a measured S4/S2 0.6-0.9 (≈5-40% at the decode path for
  Q5_K formats).
- Q4_K / IQ4_XS: 2-11% (S4/S2 nibble 0.94-1.02 t1 / 0.76-0.99 t10).

## Work items (impl lane; C4-adjacent)
- ExpandedInt8 minimum-correction EPILOGUE: make it spill-free at S=4 on
  AppleClang-21 (currently 4 spills in BackendNative) while keeping zero
  spills on GNU-14 — register-pressure redesign of the min-correction path.
- Nibble decode: make the NibbleLUT + NibbleLUTSourceScale maps spill-free at
  S=4 on GNU-14 (6-33 spills) — likely addressing/load-schedule redesign.
- Same gates as C4: witness census both compilers Rows=1=0 + grouped budgets
  unchanged + byte oracle + engine A/B (if the widths land) + reviewer.

## Scope correction (2026-11-09, perf-lane step-1 auth @ 777f7a52a)

- S=4 means FOUR COLUMN STRIPS (16 output columns) in the Rows=1 PANEL
  program — NOT four grouped rows. R2 panels installed ONLY for Rows=1
  SDOT / I8MM-single-row; grouped row/pair paths remain S1. The reported
  spills (NibbleLUT GNU 6-33; Q5_K AppleClang BackendNative 4) = PANEL
  (column-strip) pressure. Idea-31 v2 (panel scope) incoming; draft v1's
  4-row reading is superseded. FP16 chain diff vs 39c at the real base =
  EMPTY (verified) → FP16 plan base gate PASSED.

## Final-base forced-width census (2026-11-09, perf lane; NON-CERTIFYING)

- 48 explicit Rows1/SDOT programs per envelope (NEON_DOTPROD + NEON_I8MM),
  Strips1/2/4 inventory; identical across envelopes. S1/S2 all zero-spill.
- S4 NibbleLUT centered/nonlinear 22/22 spills (both policies); unsigned+
  minimum 6 (both); IQ4_XS source-scale 31 BackendNative / 20 GPUAligned.
- **ExpandedInt8+minimum (Q5_K) = ZERO spills on GNU14** — no GNU epilogue
  defect reproduced (Q5_K = AppleClang-secondary only).
- Q6_K forced-S4 = UNCLASSIFIABLE with the unchanged canonical classifier
  (`ldr x,mem,imm` post-index not understood) → no number claimed; TOOLING
  GAP: extend CPUNativeTileRegisterEvidence.py instruction classifier for
  the post-index form (owner = register-gate tooling/addon + reviewer; do
  NOT change in the perf slice).
- Classifier + installed witness untouched; .80 canonical gate separate.
