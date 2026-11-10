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

## Idea-31 candidate AppleClang secondary PASS (2026-11-10, b4dd753c8)

- M1 Max AppleClang-21.0.0 portability Release; cand + parent NEON/NEON_DOTPROD.
- Bytes 0-new/0-gone: ctest cand 292/292 DOTPROD + 291/291 NEON (parent 291/291);
  0 skips; only new = V2_Integration_CPUNativeNEONNibblePanelReplay. Entire
  prepared-projection binary: cand DOTPROD 15/15 + NEON 9/9 (new =
  DotProdNibbleS4RecordedInputReplay). In-tree v2 byte sweep all green
  (PanelWidthsMatchTheStripProgram + EverySourceAndEncoding/Compact +
  EveryHalfAndOpaqueWord + ActivationSums + tail-padding + new replay).
- Register witness: installed canonical passes; cand vs parent
  INSTRUCTION-IDENTICAL (128/128 DOTPROD, 64/64 NEON). Non-certifying forced
  census (same analyzer e6350f86): 48 programs all classifiable; candidate
  S4 = NibbleLUT Centered BN **0 (parent 9)** / GA 0(0); NonLinear BN 0
  (parent 4) / GA 0(0) -> ZERO on AppleClang + FIXES BN 9/4. Controls
  unchanged (cand = parent): Unsigned+min BN7/GA0; SourceScale BN4/GA0;
  Q5_K BN4 (gate-5 verified = 4); Q6_K 25/25 (known AppleClang spill,
  CLASSIFIABLE on AppleClang unlike GNU); ExpandedInt8/IQ3S 0; all S1/S2 0.
  44/48 forced programs identical; exactly the 4 candidate programs changed.
- VERDICT: AppleClang parity holds (spill-free both compilers). Limits:
  non-certifying census; the INSTALL needs the canonical witness extension +
  the .80 economy gate (already measured); no Mac timing.

## Idea-31 candidate #159 SOURCE ACK (2026-11-10, b4dd753c8)
- Bounded candidate SOURCE ACK; no blocking finding. Verified: retireNibbleGroup
  = compiler-dataflow empty asm (no HW sync); constexpr gate Rows1/S4/non-ASIMD/
  plain-NibbleLUT/no-minimum; maps reach the panel; arithmetic/order/scale/
  guards unchanged; panelStrips UNCHANGED (installed nibble still S2); recorded
  replay + independent test-owned reference = recorded-input (not a regenerated
  fixture); registration/coverage consistent. BOUNDS: 'leaves the activation
  block in registers' = intended compiler outcome (-> 'allows it to remain');
  176-installed witness != forced-S4 coverage; owner evidence unauthenticated;
  ACK = candidate delta on 777 only; flip/rebases = own exact-pin checks.

## Idea-31 flip — Mac AppleClang-21 canonical census GREEN (2026-11-10, 972c0081)
- 9 cells (Integration/Release/Debug identical): NEON 64/16 Rows1/0 spills +
  1415=1415; NEON_DOTPROD 128/32/0 + 2003=2003; NEON_I8MM 176/32/0 +
  2171=2171; budgets byte-equal; no rows>=2 key changed. The audited Rows1 =
  the INSTALLED ones at the flip → includes the S4 NibbleLUT centered/
  nonlinear programs both policies.
- Mac byte sweep: PreparedProjection DotProd 15/15 (incl.
  DotProdNibbleS4RecordedInputReplay + PanelWidths + TailPadding), NEON-only
  9/9, ProjectionBatch DotProd 10/10. I8MM byte/oracle cells (incl. the fixed
  codebook-5 lookup) = .80 window (GB10).
- Merge note: idea-31 branch on 777f7a52a; its 2 commits = empty intersection
  with 777f..8fcc0fe0b → rebase onto the final R2 tip = mechanical (at
  handover). INSTALLABLE = .80 GNU-14 canonical census + the rebase.

## Flip #160 SOURCE ACK (2026-11-10, 972c0081)
- Bounded SOURCE ACK; no blocking finding. Selection verified (:195-203 +
  :323-325 -> codebooks0/4 S4 both policies; 5/IQ4_XS/Q6/ASIMD/grouped
  unchanged); bodies/order/admission unchanged; S4 tiles 64-col chunk; stores
  reject inactive strips; 'allows' wording. Minimum discriminator (:807-811):
  asymmetric flag carried into the nibble width query; the 816-vs-864-vs-960
  case = source-derived explanation; NO rebaseline (the derived expectation
  CHANGES correctly -> 'unchanged counts' != unchanged erroneous 960).
  Strips0 added to the recorded checks (1/2/4 retained); the fresh
  installing-tip canonical witness includes the S4 bodies (supersedes #159's
  'installed S2' only at 972c; pre-flip receipt not retroactive). Bounds kept.
- Disposition: Mac 9-cell canonical GREEN; .80 GNU-14 canonical census +
  full-byte + economy (measured) = the last qualification at 972c0081; rebase
  onto the final R2 tip (empty intersection, mechanical) -> INSTALLABLE.
