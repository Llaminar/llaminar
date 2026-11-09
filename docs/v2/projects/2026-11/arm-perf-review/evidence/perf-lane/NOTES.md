# Perf-lane drafts (DGX station arm-addon, 2026-11-09)

- fp16-exact-tree-plan.md: native NEON FP16 projection, exact-256-lane contract
  (FloatingPointGemmKernel); FMLAL half-product candidate (HalfRNE/HalfStorage
  variants); identity proof finite-only; closed non-candidates (attention
  mixed, T5 RMSNorm, softmax); .80 gates + release; ~3-5d first family.
- idea31-gnu-s4-plan.md: GNU S4 nibble decode + Q5_K min epilogue spillfix;
  contract invariants (two policies, no reassociation, persistRoundedNEON
  boundaries); step1 authenticate final-base problem; gates (replay, byte
  sweep, register witness, .80 economy, AppleClang closure, canonical gate);
  ~half-1d analysis + 1-2d/slice.
- Both: HELD until final base (R2 rebase on f69446e3c) + .80 evidence + slice
  approval; no production edits/new modes/budget updates/branch pushes.
