# Idea 3 — Move projection metadata validation out of the K loop

Status: **Open** (at reviewed head the inner-loop throw is still present).

## Context

In `src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h`:

- `run()` (~:254) walks the K blocks inside the 4-column outer strip.
- For `IQ3SExpandedSourceScale` and `NibbleLUTSourceScale`, the reserved-bit
  check `Ops::metadataFits(...)` (in `CPUNativeNEONMultiScaleOps.h:129`) plus a
  `throw std::invalid_argument` executes on **every** K block (~:330).

The weights are an immutable publication, so the check result cannot change
during one invocation. This is both economy and collective safety:

- The check is pure overhead in the inner loop for a value that was already
  validated (or should be) at admission.
- `run()` executes inside the workshare (workers), so a throw from one worker
  while others are inside the `omp for` can leave peers at the barrier —
  the same collective-safety class as the RMSNorm `domain_error` issue
  (now fixed via typed `rejectNumeric`, see idea 10).

## Proposed change

1. Validate the complete metadata plane once before the workshare starts (or at
   view/publication admission), so an invalid publication is rejected before
   any worker enters.
2. Back it with an authentic immutable-publication proof: prepare, wire receive,
   slot reuse and migration must all prove generation/ownership, not a pointer
   cache or a shape-only check (admitted spans alone do not establish capacity
   or leases).
3. Keep a "fail fast and fatal" diagnostic (consistent with
   `rms_numeric::rejectNumeric`, `[[noreturn]] noexcept`) instead of a throw
   inside the team.

## Exactness / risk

- Removing the per-block check changes no arithmetic; the validation outcome is
  unchanged (reject before work vs throw mid-work).
- The publication/generation proof is the correctness-critical part; it must
  cover slot reuse and migration, otherwise staleness could be hidden.

## Validation

- Mutation controls: a metadata word with reserved bits must still fail, at
  admission rather than inside the loop; all-format/encoding sweeps (M 1..N,
  workers 1..20) and byte auth against serial.

## Scope

ARM NEON tile only; the x86 `CPUNativeVNNIMultiScaleKernels.h` checks are a
separate (shared) path with the same pattern but different owners.

## Addon implementation (2026-11-09) — f058a2953 part of R1 09cba9963

- Native weight bytes behind CPUNativeVNNIPublishedBytes generation counter;
  CPUNativeVNNIMetadataSeal validates metadata once per publication; the
  per-encoding coverage table (== old in-loop rules, padding included) is in
  the commit body; all 6 producers seal; overlay slot acquisition withdraws
  admission until the arrival reseals; NEON run path noexcept (static_asserts);
  writer-policy sanitizer (narrow named allowlist) + negative controls pass.
- M1 core-free (preliminary, not authoritative): MetadataSeal 6/6,
  WeightView 4/4, PreparedProjection 8/8, ProjectionBatch 10/10, NEONExecution
  10/10, NumericalContract 5/5, writer negatives 5/5; run<> 44045→37434
  instructions. Syntax-only: gcc-16 ARM + clang x86_64-AVX512. Pending:
  core-linked tests (.80) + WSL x86 gate. x86 loadScales checks untouched.

## Reviewer P1 #2 (2026-11-09, R1 @ 09cba9963) — stale test fixture

- CpuExpertSlotPool.AcquisitionWithdrawsAdmissionUntilArrival fills the whole
  native stream with byte(epoch): epoch1 → IQ3_S 0x0101=257, epoch2 →
  0x0202=514 — both violate ≤15 (not only epoch3's 0x0303) — so the new seal
  correctly rejects at epoch1 while the test expected success for 1/2.
- The PRODUCTION seal is right (in-loop-only validation previously let
  invalid metadata through until compute); the FIXTURE must be fixed: seed
  valid secondary words for accepted epochs (epoch-distinct in-range values
  OK), corrupt a single coefficient only in the rejected epoch. Needs the
  .80 core-linked run (not covered by core-free/M1 suites).
