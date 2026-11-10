# Idea 11 — Attention validation hoist + integer-dot range proof

Status: **ADDRESSED** — merged via R2 C2c (FA invocation admission) + addon scope. (2026-11-10 status flip)
lives rather than the arithmetic itself.

## 11a. Per-call validation in NEON attention math

`src/v2/kernels/cpu/attention/NEONAttentionMath.h`:

- `dot()` (~:113) validates query/key span sizes and geometry on every call;
  `accumulate()` (~:150) validates spans, overlap and `isfinite` weights on
  every call — inside the KV loop over 1/2/4 key rows.
- These are per-invocation constants for one attention step; repeated checks
  for every small row group are pure overhead.
- Note: the checks are not a full `isfinite` scan of every key; `dot()` does
  not scan keys for finiteness (only query scale paths do, ~:241-245).

Proposed: hoist geometry/overlap/finite-weight validation to one per-tile or
per-invocation admission step, backed by an authentic publication/lifetime/
generation boundary (same requirement as idea 3). Do not substitute a decoded
bank or a hot full-cache pre-scan.

## 11b. `integerDot` accumulation width

`integerDot()` (~:188) accepts `int16 × int16` rows. Worst case at head_dim 256
is ~2^38, so INT32 accumulation is unsafe under the current API even if the
final result is checked against INT32 — the intermediate sums overflow.
A head-width-only bound cannot authorize INT32.

Proposed options (only with proof):
- keep INT64 accumulation (current, exact), or
- add a separate narrow typed range proof (e.g. per-format bounded activation
  and key values) that authorizes a second INT32 kernel, with overflow checks.

## Exactness / risk

- 11a: no arithmetic change; correctness depends on the publication/generation
  proof (slot reuse, migration) replacing the per-call check — must not weaken
  or drop the checks.
- 11b: math change; INT32 must be proven safe for every admitted input range,
  or the exact INT64 path stays.

## Validation

- Focused regressions: alias/geometry mutation tests still fail if checks are
  dropped; int32 overflow witness at large head_dim for 11b.

## Addon-lane decisions (2026-11-09)

- 11b GATE SATISFIED: limit(T) = min(2047, floor(INT32_MAX/(32768·T))), T =
  min(head_dim, block_elems); every integer reduction runs under
  blocks_per_head==1 (head_dim ≤ block_elems); multi-block = per-block INT32
  → FP32 sum, NO cross-block INT32 sum; worst cases 2,139,095,040 (T=256,
  limit 255) … 2,146,435,072 (T=32) ≤ INT32_MAX. Plan: typed
  cpu::Q16ReductionBound as single authority; NEON → INT32 (vmlal); INT64
  deleted.
- 11a measurement (M1, FP16 cache, 4-row groups, 4096 rows, L2-resident):
  dot checks +3/+6/+11% at head_dim 256/128/64; accumulate +5/+10/+20%.
  Vectorized finite check ~0.6 ns/call → kept inside accumulate, FATAL +
  noexcept (stronger than a throw).
- SCOPE DECISION (arm-perf 2026-11-09): full FA invocation admission (a) —
  ISA-neutral cold admission (Q/K/V/O banks, disjointness, head width) on
  decode/batch/partitioned/context-parallel/Q16 entries; x86 gains it too
  (today x86 has NO buffer admission); NEON helpers become unchecked-admitted
  (debug asserts). Structure: R1 = C1(03) + C2a(11b) + C2b(fatal finite
  check); R2 = C2c(FA admission) + C3/C4(04). R2 shared code validated via
  WSL x86 gate + Mac portability + GB10 attention parity/engine.

## Reviewer P1 (2026-11-09, R1 @ 09cba9963) — tail overflow before validation

- integerDot validates query magnitude only AFTER the scalar tail
  `result[row] += int32_t(q[c])*k[row][c]`; invalid input can overflow INT32
  before rejectAttentionData. Repro: T=n=65, bound.limit=1008, first 64
  q=-1008/k=-32768, last q=-32768 → vector fold 2,113,929,216 + tail
  1,073,741,824 > INT32_MAX.
- REQUIRED FIX: reject invalid query magnitude BEFORE any bound-dependent
  scalar op (tail included); retain the valid-input INT32 program; add a
  tail-bearing adversarial regression (current n64 fixture only exercises
  vector lanes with key=1). Amended R1 SHA → reviewer re-check (reviewer
  pins to exact SHAs; in-flight R2 kept out of the reviewed tree).

## P1 FIXED at amended R1 0501d506b (2026-11-09)

- Reproduced with UBSan on 09cba9963 ('signed integer overflow:
  2113929216 + 1073741824' at tail add).
- Fix: combined vector peak + tail-code scan checked BEFORE the horizontal
  fold and any scalar tail arithmetic (before that: modular SIMD adds only —
  no UB). Valid input runs the IDENTICAL INT32 program (same vector
  accumulation, fold, tail order).
- Regression: Q16BoundRejectsInvalidTailCodesBeforeTailArithmetic
  (V2_Integration_NEONAttentionQ16Int32Bound) — n=65, limit 1008, keys
  −32768; valid worst case 2,146,959,360 exact for Rows 1/2/4; tail codes
  −32768/±1009/32767 fatal per Rows 1/2/4; same codes in the vector part
  fatal. UBSan clean; 7/7 NEONAttentionMath on M1.
- Reviewed tree = 09cba9963 + fix only (no R2). R2 (C2c) uncommitted; will
  rebase onto 0501d506b and carry the same tail ordering into the admitted
  integerDotRows kernel.

## SOURCE110 final (2026-11-09, 5c89f981c) — STATIC GATE PASSED

- All prior SOURCE109 P1/P2 findings SOURCE-CLOSED; no new P1. R1 passes the
  static-review gate to R2 (runtime/economy gates separate). Closure: (1)
  magnitude checked vector+tail before fold/scalar (valid program unchanged;
  n65 fixture in Q16Int32Bound preflight); (2) slot epochs 1/2 valid seeds,
  epoch3 corrupt only, same-root/republication checks kept; (3) rebound
  private to deep-copy owner; admitted/executeAdmitted private to batch
  friend + checked public methods with public positive controls; (4) SIGABRT
  + diagnostic + terminate discriminator, MAP_SHARED child-visible, Rows1/2/4
  × every bad row, write-then-reject control, single stdio; (5) whole-head/
  per-block wording corrected in successor + source.
- EVIDENCE PRECISION (nonblocking): the Q16 fixture's exact worst-case
  positive = Rows4 only; invalid tail = 1/2/4; invalid vector = 4 only;
  EXPECT_DEATH-based → cannot be cited as explicit-abort proof. Preferred
  fix: extend fixture to Rows1/2/4 everywhere (lane doing); else narrow
  evidence claims. UBSan old-code failure/fix report = owner evidence only.

- Fixture extension done at **46c9ffc58** (parent 5c89f981c, test-only): worst-
  case positive + invalid tail + invalid vector codes each run Rows1/2/4; all
  assert KilledBySignal(SIGABRT) (terminate handler exits 77). 8/8
  NEONAttentionMath on M1. R1 evidence SHA = 46c9ffc58.
- R1 close run STARTED on .80 (Integration NEON_I8MM gcc-14, Unit + preflight
  + focused: CPUNativeVNNI/NEON/attention/Q16, ExpertTier, PackedWeights,
  CpuExpertSlot, MoEOverlay residency/migration/protocol/fabric, slot-reuse,
  seal, writer policy; CUDA/ROCm/MPI/Perf excluded). Status:
  build/r1-46c9ffc58.status. Then economy window (announced before start).
- objdump checker handed to arm-sparks (fndiff.py + TU + gcc-14 commands,
  normalization documented, STRICT=1 keeps immediates; mac result reproduces).
- R2 being rebased onto 46c9ffc58 locally; outside the reviewed tree.
