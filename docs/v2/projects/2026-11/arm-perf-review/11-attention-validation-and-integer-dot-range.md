# Idea 11 — Attention validation hoist + integer-dot range proof

Status: **Open**. Two related attention items, both about where validation
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
