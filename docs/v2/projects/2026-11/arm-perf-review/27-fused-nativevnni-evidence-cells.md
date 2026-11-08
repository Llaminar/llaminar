# Idea 27 — Fused NativeVNNI evidence cells fail on ARM (Pairwise/WideRows gap)

Status: **Open — fix in flight**. 2026-11-08: prepared on the PR #43 branch by
Main (evidence cell uses the native WideRows schedule on aarch64; x86
unchanged; Pairwise stays as the second distinct policy); not pushed yet
(container rebuilding oneDNN). Close on Main's SHA once both NEON
fused-completion cells pass locally.

## Finding (commit-1 preflight acadffd64, .80 → 2284/2291)

- 4× `CPU_NativeVNNIFused{ProjectionEvidenceIdentity,Completion}_{NEON,
  NEON_I8MM}` fail with **"ARM grouped projection requires a native
  Pairwise/WideRows candidate"** in
  `ProjectionEvidenceUsesGeometryAndPolicyInsteadOfBundleOrdinal`.
- They fail identically in the plain **NEON** cell, so they come from base
  `b2cf793bf` — the newly registered upstream **fused-verifier regressions**
  appear to force an x86 policy (`Pairwise/WideRows`) that ARM does not
  implement natively.
- (Also base, for completeness: 3× nested-configure target-selection failures,
  same OpenMP-target family as idea-26 context.)

## Open questions

- Do the upstream fused-verifier evidence cells need an ARM row-tile policy
  (Pair/Four) as a first-class `VerifierRowsPolicy`, or should the cells be
  ARM-admitted differently (per the "no skips; hardware present = must run"
  rule)? Main (implementer) owns this classification.

## Validation

- After any fix: the four cells pass on NEON and NEON_I8MM; unit/preflight
  counts return to 2291/2291 for the base + i8mm envelope.
