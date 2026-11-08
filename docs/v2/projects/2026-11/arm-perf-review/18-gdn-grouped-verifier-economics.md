# Idea 18 — GDN grouped verifier + publication economics (mixed cores)

Status: **Open**. Needs a clean-host look; GB10 Release 20T/1T numbers are
pending the final `Perf__` round. Related: plan item "GDN grouped verifier
economics on mixed cores"; ideas 14 and 16.

## Finding (Mac VM, Integration build, 10 threads, all byte-exact)

Source: `Perf__CPUGatedDeltaNetVerifierRows`.

- **Grouped kernel alone vs serial:** mostly **1.0–1.2x** for M=2..7,
  **2.5–3.4x** at M=13/15/31 — but **non-monotonic dips** at M=8 (0.59),
  M=12 (0.63), M=14 (0.54), M=16 (0.74).
- **Production stage path** (`RuntimeM_CaptureAndAcceptedPublication_
  ProductionStage`, capture + accepted-state publication):
  **0.18–1.0x** of serial for M≤11 full acceptance (e.g. M=10 0.32, M=13 0.18),
  and far below serial for partial acceptance (0.1–0.5x). The test's economy
  assertion failed in one run at M=31 (0.63).
- **Caveat:** samples swing ~10x because the VM was loaded by a concurrent
  gate. GB10 Release 20T/1T numbers come from the final `Perf__` round.

## GB10 confirmation (final Perf round, K14-PERF-FINAL-264970E14 @ `264970e14`)

`CPUGatedDeltaNetVerifierRows` on the sealed final round: **grouped-vs-serial
speedup < 1 for essentially every M at both 20T and 1T** (e.g. M=4 **0.447**,
M=6 **0.416** @ 20T; only M=2 @ 1T shows 2.08, likely noise). The grouped GDN
verifier path is **uneconomical on NEON** — confirming, not just suggesting,
the analysis above. Any fix must first
separate kernel vs capture/publication cost (the production-path numbers above
were the worse half). Since it fails at 1T too, the cause is the **grouped
kernel/publication structure, not core heterogeneity** (see idea 5).

## Open questions (before any kernel change)

1. Is per-row **state capture / accepted-state publication** the dominant cost
   in the production stage path (versus the grouped kernel itself)?
2. Do the M=8/12/14/16 dips come from a **tile/partition boundary** in the
   grouped kernel?

## Validation

- Clean-host rerun (no concurrent gate) on the Mac VM, then GB10 20T/1T in the
  final `Perf__` round; byte-exactness maintained; M sweep with the dips
  re-measured; separate the kernel-only and capture+publication costs before
  deciding where to optimize.
