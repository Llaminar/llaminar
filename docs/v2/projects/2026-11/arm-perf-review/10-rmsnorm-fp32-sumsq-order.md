# Idea 10 — FP32 RMSNorm scalar sum order + sanitizer warnings (issue #36)

Status: **Addressed** at the reviewed head by `3086894b5` (pin scalar FP32 RMS
order) and `ac3289bd9` (typed fatal for numeric corruption). Entry kept for
context plus residuals.

## Original finding (issue #36, 12 optimized-ASan/UBSan `-Wpass-failed`)

- `compute_sumsq_scalar` used `#pragma omp simd reduction(+:s)` over
  `s += double(row[c]) * double(row[c])`. The reduction can keep VF independent
  partials combined in an unspecified order, so the "scalar" reference depended
  on vector width: optimized builds (NEON 2, AVX2 4, AVX-512 8 doubles) could
  differ from sanitizer builds, which fail vectorization and run strictly in
  increasing order.
- Contraction was not the cause: an FP32 square is exact in FP64 (48-bit
  significand ≤ 53), so `fma(v,v,s) == s + v*v` bit-for-bit; only order matters.
- Apply loops (`(input*inverse)*gamma`, `input*inverse`) are elementwise with no
  reduction, so vectorization cannot change results — the warnings were pure
  economy, and only the reduction order needed source pinning.

## What landed (verified in source at head)

- `compute_sumsq_scalar` is now an explicit increasing-column FP64 FMA chain
  with `noexcept`-style declared order (`RMSNormPrimitives.cpp` ~:395),
  independent of vector width and instrumentation.
- Non-finite summaries now route through `rms_numeric::rejectNumeric`
  (`[[noreturn]] noexcept`) instead of `throw std::domain_error` inside the
  OpenMP workshare — an explicit typed fatal, resolving the "exception cannot
  escape `omp for`" lifecycle mismatch (standalone calls would have terminated
  anyway; retained teams could hang at the barrier).
- The apply path was refactored into a typed row application with an explicit
  publication ISA.

## Residuals / follow-up

1. Confirm the typed apply path still vectorizes in non-sanitizer Release on
   x86 (compiler remarks `-Rpass-missed=loop-vectorize`); on ARM the generic
   path is diagnostic-only (native NEON dispatch is used otherwise).
2. Confirm sanitizer builds no longer warn at the apply sites; if they still
   do, decide whether the reason is sanitizer instrumentation (expected) or a
   real missed vectorization.
3. Regression quality: the existing tolerance-based tests could not
   discriminate reassociation. The proposed deterministic witness (row =
   {2^27, then sixteen 1s} → exactly 2^54 in increasing order, larger under any
   VF≥2 split) is a candidate bit-equality regression for the detail seam
   (`detail::compute_sumsq_scalar` is already declared in the production detail
   header and used by existing tests).
