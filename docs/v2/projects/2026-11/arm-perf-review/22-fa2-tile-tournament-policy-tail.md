# Idea 22 — FA2 tile-tournament policy tail on GB10

Status: **Open**. Evidence: `K14-PERF-FINAL-264970E14` (GB10/NEON @
`264970e14`; sealed summary
`~/llaminar-spark-results/K14-PERF-FINAL-264970E14-20261108`).

## Finding

`CPUFlashAttentionTileTournament`: the FA2 aggregate **p95 regret is 8.926%**
against the 5% acceptance threshold. Worst codec pairs:

- `tq4_tq4` — 17.6%,
- `q16_1` — 16.6%,
- `tq4_tq8` — 14.9%.

So the policy tail on GB10 is too long: a visible share of attention shapes run
with tile choices far from the tournament-optimal ones.

## Open questions / leads

- The FA2 launch policy intentionally starts from a cache-independent
  canonical-span policy (idea 12) — is the tail a codec-specific (TQ4/TQ8/Q16)
  tile-geometry mismatch, or an X925/A725 class effect?
- Do the TQ4/TQ8 paths (quantized KV) need their own tournament/grid rather
  than the FP32-derived policy?
- Where does the current policy admit picks from (per-codec tournament vs a
  shared span grid)?

## Validation

- Re-run the tournament per codec pair and per core class (X925-only /
  A725-only / mixed) on GB10; target p95 regret ≤ 5% per codec; byte-exact
  outputs preserved; final `Perf__` round after any policy change.

## r2fa evidence (2026-11-09, R2 lane, .80 Release solo ABBA) — base fails its own policy

- V2_Perf_CPUFlashAttentionTileTournament: base r1 507s exit1 p95 8.955%
  max 20.1%; R2 r1 496s 7.998%/22.4%; R2 r2 493s 8.487%/14.1%; base r2 508s
  5.129%/17.7%. Policy check requires p95 <= 5% -> FAILS ON BASE both runs
  (drift 5.1-9.0% = GB10 tile-policy calibration, not R2). R2 mean -2.6%
  wall vs base (C2c per-row check removal). Solo = 83-85% of the 600 s limit
  -> parallel-ctest timeout = load, not R2. CLASSIFICATION: base/env.
- Worst tails: grouped_verifier M=15 hd256 tq8_tq8 / q8_1. Options: GB10
  tile-policy recalibration or cell timeout/ISA policy review; owner = FA2
  tile policy (idea 22), not a lane. Gated-cell rerun (ctest solo) = wall
  time only; cannot flip the classification.

- Gated-cell solo rerun (f9f190059, Integration, ctest): FAILED after 494.6 s
  (NO timeout) — same base policy check, p95 regret 9.248% > 5%. Confirms:
  the 600 s timeout in the parallel gate = LOAD on a cell that fails on base
  anyway. Idea-22 classification FINAL (base/env).
