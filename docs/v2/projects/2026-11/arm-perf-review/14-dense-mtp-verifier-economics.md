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

## Base-dependent result (2026-11-08, afe62c5ef + i8mm c3)

On the NEW base, 27B MTP dyn **5.455—5.545 vs no-MTP 5.28 tok/s** — MTP now beats no-MTP on this base (acceptance 43.4%), opposite of the older 268cd81de numbers (4.24/4.27 vs 4.49). The dense-MTP decision is base/version-sensitive: evaluate the economics for each base, not once. (Root causes of the flip are not yet separated — likely the upstream perf/prefill fixes.)

## Activation (2026-11-10, operator: address it)

Current-tip facts before any policy change:

1. **Lever 1 (i8mm for M>=2) is already implemented**: the grouped kernel
   (`CPUNativeQuantisedGemmKernel.h`) runs SMMLA row-pairs on shared-row
   tiles and executes the SDOT leaf for a lone row — that lone-row fallback
   is the "M=1 guard". The remaining economics are therefore acceptance +
   depth policy + LM-head-per-row, not missing SMMLA.
2. **`MTPDepthDefaults.h` has no CPU profile** (Portable/CUDARTX3090/
   ROCmMI50 only; Portable adaptive max depth = full range 15, demotion
   rate 0.30). The C-era finding (depth 10 at 43.4% acceptance, decision
   `window_not_ready`) is consistent with device-tuned defaults.
3. **Base sensitivity**: on the new base MTP already WINS dense
   (5.455-5.545 vs 5.28). Re-measure on the current tip before changing
   policy; the C-era loss may be stale.

Owner: **DGX station perf lane** (plan-first, policy-only). Scope in order:
(a) root-cause `window_not_ready` on the current tip (code + controller
replay with recorded traces + CPU timings — no live window needed);
(b) re-measure dense-MTP economics on the current tip (live 27B runs booked
through arm-perf/arm-sparks only, queued after the idea-31 census, B9
census and the Bundle-1 bound run; a Grace station run may serve as a
cross-check, not the canonical GB10 number);
(c) only if economics are still negative on the current tip: add a measured
CPU continuation profile / demotion backoff so the controller cannot run
depth whose verifier cost per accepted token exceeds a decode step.
Policy-only (no kernel byte changes); grouped rows stay byte-identical
(MTP batch invariance); base feedback to Main (01a1a8bf); reviewer at a
clean checkpoint; merge-ready handover to Main.

## Plan v1 APPROVED (2026-11-10, station perf lane; plan-first)

Anchor `c007a59a2` (SHA-authenticated); isolated branch
`feat/dense-mtp-controller-economics`; policy-only; no live work yet.
APPROVED: **Plan A** — offline root-cause (configure/reset→publication→
recordStep→summary lifecycle chronology from RETAINED traces; owning
controller/replay tests incl. boundary matrix; bookkeeping-overhead timing
kept separate from model economics; wall-nanoseconds reported, PMU
explicitly unavailable). No profile/threshold change at this checkpoint.
**Plan B** (fresh paired GB10 economy: Release MTP-off vs production
dynamic, rotated A/B, raw reps + p95, mode verification via terminal
PerfStats — never cell labels [historical cells were mislabeled], exact
source/binary hashes, anchor-refresh rule, prefill/generation split, full
transaction cost without double-counting, bounded fixed-depth controls only
if negative) = coordinator-booked after the idea-31 + B9 censuses and the
Bundle-1 bound run. **Plan C** strictly conditional on demonstrated current
loss; typed design review if no existing profile/ABI can represent the
policy.

**DIAGNOSTIC CORRECTION to the finding above:** a terminal
`window_not_ready` + request-wide 43.4% do NOT establish "depth 10 at 43%
acceptance" or a permanently blocked window — `recordStep` clears consumed
windows after evaluation and overwrites `last_decision` on later incomplete
windows; the runner reports the LAST reason beside request-wide acceptance
whose denominator is ALL draft steps. The C-era capture is a diagnostic
limitation until the A audit says otherwise. Also: Portable zero-accept
demotion 0.30 = fraction of zero-accept TRANSACTIONS (not 1 − token
acceptance); the CPU controller records counters, not clocks;
`MTPDepthCostProfile.h` is ROCm/QFN-MoE-scoped (no CPU-dense authority);
canonical replay = `v2_test_mtp_depth_trace_replay` (cannot produce
counterfactual acceptance streams or end-to-end speedups).

### A-execution note (2026-11-10): station build constraint + approved diagnostic path

The station's monitor-owned c007 configure stops BEFORE compile/test:
`OneDNNBuild.cmake:160` needs `external/onednn/build-neon_i8mm.llaminar.lock`
inside the READ-ONLY dependency mount (failed receipt retained; lane
correctly refused to make shared externals writable, install, disable
backends, or edit build policy). APPROVED resolution for the A audit only:
**core-free direct linkage of UNCHANGED owning sources** (MTPDepthTraceReplay
files + actual MTPDepthController.cpp, official MTPDepthReplayIdentity.cmake
generator, GTest, unchanged Perf__MTPDepthController.cpp for wall-ns),
labeled **owner-test/self-check only** — NOT canonical CTest/full-preflight
or production certification; exact compile flags + file SHAs recorded;
canonical CTest evidence, if the reviewer requires it later, runs on .80
(SHARED_CORE=ON runbook). Read-only lifecycle/fixture audit continues in
parallel.

## A DELIVERABLE COMPLETE (2026-11-10, station lane; self-check evidence class)

29/29 trace + 71/71 controller + 1/1 bookkeeping owner tests PASS at c007
(zero source changes); 15+5 hashes rechecked; failed attempts preserved
with SHAs. Bookkeeping overhead = **14.6-55.5 ns/step** (synthetic composite,
wall-ns) — negligible vs a ~200 ms decode step: controller bookkeeping is
NOT a candidate cause of any dense-MTP loss. Readiness semantics audited:
terminal WindowNotReady + windows>0 = partial-tail explanation, NOT
never-ready; request-wide 43.4% ≠ a depth-10-window sample; the CPU/dense
controller has NO timing authority (counters only), so acceptance alone
cannot establish economic loss; the C-era trajectory stays unproven without
its inputs. Full packet + digests: evidence/idea14-A-owner-selfcheck.md.
Decision gate = Plan B (booked paired GB10 wave). Routed to reviewer at
c007a59a2.

### Reviewer #163 on A: source audit SUPPORTED; two P2 record corrections en route to unqualified ACK

Disposition upheld (no readiness defect; no change from A alone; C
conditional; negative B = necessary not sufficient). Corrections: [I14-A-1]
candidate-path agreement ≠ baseline validation by this executable (exact
flags to be recorded); [I14-A-2] the owner build wrapper is not a c007 file
— 14/14 tracked SHAs blob-verified, wrapper/binary hashes = owner evidence.
Details in evidence/idea14-A-owner-selfcheck.md.

### B-wave booking confirmed (operator, 2026-11-10): first post-SETTLED .80 window

Operator confirmed the coordinator defaults: Plan B (fresh paired GB10
Release economy, rotated A/B, mode-verified, full transaction cost) runs in
the FIRST post-SETTLED .80 window, anchored at Main's settled tip; bounded
fixed-depth controls stay only-if-negative. Post-SETTLED execution order
confirmed: combined idea-31+B9 FF → Bundle-1 FF → Qwen3.8-UD trace → GPU
windows (ideas 19/14/18) → half-perf cell → native FMLAL leaf.

### 27B GREEDY-RIDER PROTOCOL SHEET (arm-addon3 draft, ~14:53Z) — operator-approved, contingent

Purpose: settle the dense-MTP economics question on the current base with
a controlled greedy reference — the base/version-sensitive MTP flip
(5.455-5.545 dyn vs 5.28 off on the new base; 43.4% acceptance at depth=10)
is protocol-confounded (bench config carried temp 0.8/top_k 40/top_p 0.9/
seed −1).
- **ARTIFACT (27B open item resolved to tracker extent):** Qwen3.6-27B-
  MTP-Q4_K_S.gguf (the unsloth MTP repack = the tracker's 27B baseline
  identity; the plain Qwen3.6-27B-Q4_K_S.gguf is OUT of scope). Byte
  sha256 + on-disk path CAPTURED AT STAGING; if the K14/R1 gate dirs hold
  an original 27B staging sha256, that supersedes the at-staging capture
  (arm-sparks checks its artifact index first). Binary = the campaign
  binary at bb893cd79 (identity recorded).
- **FIXED (identical to the 35B controlled-reference shape):** GGUF
  bytes; prompt bytes; −c 8192; 256-output policy; cache-reset; placement
  (20 cores, capacity-ordered); build + policy identities; host identity.
- **CHANGED: ONLY sampler intent** — temp 0 + explicit seed 42. No seed
  search.
- **RECORDED:** RESOLVED actual sampler (post-engine-resolution, never
  just requested flags); MTP counters (acceptance + depth-controller);
  text signature (hash of generated text = the controlled greedy
  reference); tok/s; stage timings.
- **PROPOSED: TWO transactions** (the 27B economics question is
  intrinsically a MODE PAIR): T1 = MTP dynamic; T2 = same file/flags with
  --no-mtp — sampler pinned in both. FLAGGED for approval since it extends
  the 35B one-transaction shape (else T2 becomes a separate follow-up
  sheet).
- **CONTINGENCY:** execution waits for the 35B greedy result + a campaign
  slot. No execution before either.
STATUS: draft for the tracker; awaiting (a) the T1/T2 two-transaction
approval call, (b) the 35B greedy result, (c) a slot.

### IDEA-14 B-WAVE DEPTH-SWEEP PROTOCOL (arm-addon3 design draft, ~15:17Z) — execution queued post-FF

**QUESTION:** is the Phase A acceptance drop (38.89% at depth=10 [1,15]
budget-limited vs tracker-verified 99% at depth 7) DEPTH-DRIVEN,
sampling-driven, or base-driven? The greedy rider settles the sampler
axis; this sweep isolates the DEPTH axis with everything else pinned.
- **MODE DEPENDENCY (upfront):** the sweep runs under the controlled-
  reference sampler (temp 0 + seed 42) IF the 35B greedy rider validates
  the sampling-mode repair; otherwise BOTH modes per the greedy outcome —
  the mode dependency is carried as a factor, not hidden. Depth is
  orthogonal to sampler by design: only the depth policy changes.
- **INDEPENDENT VARIABLE — depth policy:** fixed depths **5, 7, 10, 12**
  (7 = tracker-verified baseline; 10 = Phase A observed; 5/12 bracket);
  **dynamic** (the engine's depth controller) as a separate cell
  (production mode; its own acceptance behavior is part of the question).
  Budget pinned to default for fixed cells; budget-limited events recorded
  per cell via the existing `mtp/depth_transaction_trace` v3 (pre_depth/
  accepted_prefix/post_depth/recommended_depth/rollback/budget_limited/
  requested_depth/effective_depth — zero new instrumentation).
- **DEPENDENT VARIABLES per cell:** (1) acceptance curve — accepted-prefix
  distribution (mean/median/p99 + rollback rate); (2) tok/s per depth
  (idea-14's verifier-cost economics: acceptance vs per-row verifier
  cost); (3) budget_limited event rate (confound candidate — acceptance
  collapsing exactly where budget_limited fires = budget-driven, not
  depth-driven); (4) requested vs effective vs recommended depth
  divergence (controller behavior under each fixed depth).
- **MODELS (paired blocks):** 27B Q4_K_S MTP repack (dense — idea-14's
  subject; byte identity at staging, K14 artifact index checked first) +
  35B IQ3_S MTP repack (where 38.89% originated). Each model = its own
  paired block (same binary/GGUF/prompt/context/256-output/cache-reset/
  placement/sampler across all depth cells; ONLY the depth policy
  changes).
- **CONTROLLED-REFERENCE DISCIPLINE:** one benchmark transaction per
  cell; RESOLVED actual sampler + MTP counters + text signature recorded
  per cell; no seed search; host identity in every artifact. **The text
  signature doubles as a depth-invariance check:** at fixed sampler +
  prompt, different depths change only the VERIFICATION schedule, not the
  accepted-token sequence — divergent signatures across depths = itself a
  finding (rollback/budget interaction changing outputs).
- **EXECUTION SHAPE:** 2 models × 5 depth cells = 10 transactions, n=256
  decode, at the merged tip. Anchor + artifact identities bound at
  staging (refresh on drift). Queued post-FF, after the greedy rider (its
  outcome sets the sampler mode), same window family (same host class/
  binary).
- **INSTRUMENTATION: zero new code** — the existing mtp/
  depth_transaction_trace v3 family publishes every field the curve
  needs; the sweep = run-matrix + export-analysis only.
