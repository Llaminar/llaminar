# Idea 14 — Dense-model MTP verifier economics on CPU (Qwen3.6-27B)

Status: **Open**. Measured symptom; root-cause attribution on GB10 still
pending.

## Finding

On GB10 (Release, 20 cores, after the spin and workspace fixes), Qwen3.6-27B
Q4_K_S runs MTP off at 4.61 tok/s and MTP dynamic at 4.23 tok/s, with 52-65%
acceptance and depth 3-4. The 35B MoE model gains 30% from MTP, because its
experts are small and acceptance is 99%.

The verifier is a grouped NativeVNNI GEMM over the dense weights. On the Mac
VM, Q4_K at the 27B gate/up shape (N=17408, K=5120, 10 threads):

| M | Time | Ratio to M=1 |
|---|---|---|
| 1 | 1.10 ms | 1.0 |
| 4 | 2.56 ms | 2.3 |
| 15 | 11.7 ms | 10.6 |
| 4 rows serially | 6.23 ms | 5.7 |

Verifying 4 rows costs about 2.3 decode steps. At about 52% acceptance with
depth 4 a step yields about 2 tokens, so once the draft head is added MTP
cannot win on this kernel. The grouped CPU path is compute-bound in SDOT
integer dot products; it amortizes weight bandwidth but not arithmetic.

## Ideas

1. **I8MM (SMMLA) for M >= 2** (idea 6). This is the direct lever: two
   result rows per instruction against one for SDOT, with identical exact
   INT32 sums. GB10 X925/A725 support it; the M1 does not.
2. **Depth controller check.** Dynamic depth should back off to shallow depth,
   or effectively off, when the measured verifier cost per accepted token
   exceeds a decode step. A 9% loss suggests the controller's CPU cost model
   or hardware defaults (`MTPDepthDefaults.h`, `continuation_profile=portable`)
   do not reflect CPU verifier cost. Replay the recorded traces through the
   controller (controller replay tests) with CPU timings.
3. **Stage attribution first.** Run per-stage CPU timing for the 27B, MTP off
   against dynamic, to confirm the verifier GEMMs dominate rather than
   GDN-verifier rows or the LM head (5.7 ms per verifier row on the 248K
   vocabulary is itself significant).

## Exactness / risk

- I8MM: exact INT32 sums. The surrounding decode and scale arithmetic must be
  proven unchanged per encoding (see idea 6). Grouped rows must stay
  byte-identical to serial decode (MTP batch invariance).
- The controller change is policy only and does not affect bytes.

## Validation

- `Perf__CPUNativeVNNI_ThreadScaling`, `Perf__CPUFusedVerifier`, SDOT against
  SMMLA, all formats, M = 2..16, per core class.
- End-to-end 27B and 35B MTP off against dynamic, -n 256, unprofiled.
- The MTP/admission test lanes are available for exhaustive debugging runs
  (project owner, 2026-11-07: mostly for exhaustive debugging; freely runnable,
  not intended as mandatory CI/CD gates). Use them **later, on demand, as a
  perf-debug helper** when this idea is implemented — verifier-depth,
  acceptance rate and depth-controller behaviour under the measured spin
  policy — rather than as a now-obligation or a CI check.

## Verified numbers (corrected)

Verified-mode per-commit run, A2 `878a33fea`: 27B MTP dyn **4.20** vs off
**4.51 tok/s** (−6.9%). The earlier “≈ equal” figures came from **mislabeled
cells (both runs had MTP on)** — the verified comparison is the A2 pair above.
At `268cd81de` (C): MTP dyn 4.24–4.25 vs no-MTP 4.49–4.62 (−7…−8%). Mainline
evidence for the dense-MTP decision stays the verified pair; idea 19 covers
the prefill side.

## Depth-controller finding (2026-11-08)

At C the 27B depth controller reports **43.4% acceptance at depth=10** with
last decision `window_not_ready`, i.e. it runs depth 10 at 43% acceptance.
For any positive verifier cost that is economically wrong: most of each
grouped verify is wasted. Two open questions:

1. Why does the decision window never become ready (`window_not_ready`)?
2. Are the CPU continuation profile defaults in `execution/config/
   MTPDepthDefaults.h` sensible for CPU (they may be tuned for device
   continua)?
