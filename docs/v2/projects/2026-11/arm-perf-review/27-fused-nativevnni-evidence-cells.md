# Idea 27 — Fused NativeVNNI evidence cells fail on ARM (Pairwise/WideRows gap)

Status: **ADDRESSED** (already recorded 2026-11-08 — confirmed).
two commits: `37bd75ffa` lineage (evidence cell uses the native WideRows
schedule on ARM) and `afe62c5ef` (real cause: ARM's fused launcher recorded
nothing — three call sites published a 12-tag record including the bundle
ordinal (an upstream-removed cardinality defect), and M=1 fused paths
published no launch evidence; the NEON launcher now publishes the shared
21-tag semantic schema with batched multiplicity and the call-site recorder
is gone). Local aarch64 NEON: FusedProjectionEvidenceIdentity, FusedCompletion
and OrderedNibbleReplay pass; v2_integration_cpu_native_vnni_gemv 67/67.

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
