
## REVIEWER VERDICT (B4 corrected spec, ~22:48Z): NOT YET SPEC-FAITHFUL

Verified at tracker f20600403: the ORIGINAL
evidence/b4-draft-vocab-budget-design-spec.md is STILL the exact old blob
69cf952677146da42b565223b1d511b92913cdac (unchanged since de7062e2). The
new evidence/b4-design-spec-corrected.md (blob
886baa6a7f07bd68473dede271b8bb925a8d6e73) is a 61-line CORRECTIONS
SUMMARY, not the corrected full spec (single-commit history 0fefe187) —
"full corrected text preserved in history" is NOT supported.
S1/S2/S3/S6: substantively right. S4: lifetime "specified" without the
actual lifecycle. S5: rider reference "resolvable" without any reference.
The contradictory Mode C/fence wording REMAINS in the claimed canonical
spec. FIX REQUIRED: apply the normative corrections to the EXISTING
original spec and identify the exact resulting blob; no more summaries.
No new design disagreement, implementation restriction or gate added.
The B4 diagnostic d7eb73708 source review proceeds separately.

## REVIEWER VERDICT (B4 diagnostic source, ~22:52Z): HOLD at d7eb73708

Static findings (complete 1000-line diff read; no execution by reviewer;
production routing/sampler/verifier untouched):
- D1: fixture contradictions (size==8 vs 3 IDs; V=96 skips 128/256/512;
  threshold 1 not 0.5; wrong-q negative evaluates identically-p).
- D2: **Mode P biased — accept_uniform reused untransformed for the
  residual draw; counterexample p=(1/2,1/4,1/8,1/8), q=delta_0 → token1
  never emitted; realizedLaw never calls rejectionStep.**
- D3: selected arithmetic ≠ production tree (one 64-block chain vs 16
  ordered partitions × 4 blocks + fold; EXPECT_EQ not byte-compare;
  M1/unsharded V96 only; GPUAlignedExpert ≠ the measured LM-head policy).
- D4: selector lifetime not enforced (no verified receipts; merge-not-
  replace; no stale/lookahead rejection; cold-start fallback needs
  explicit admission).
- D5: Modes G/C toy copies (new hash/double-CDF vs
  sampleMTPSerialEquivalentTargetRow; G tie retains first entry not
  lowest ID).
- D6: cross-arch build regression (CMakeLists:7448ff adds the target to
  ordinary Unit on every arch while the TU #errors off AArch64).
UNPROVED remaining: counted traffic, multi-N execution, 63/65/127/129/V-1
byte sweep, reset replay, normalized-logit draw.
DISPOSITION: safely non-installed; not source-ready/unfencing basis;
repair focused mechanisms in THIS diagnostic; no new gate/apparatus.
Note: the station W3 focused check is MOOT for now — the diagnostic is
under repair; re-run after the D1-D6 fix tip.

## REVIEWER RE-VERDICT (B4 canonical spec @ 82fe1917, ~22:54Z): body finding CLOSED; 2 corrections + precision edits remain

CLOSED: the missing-corrected-body finding (blob verified IN PLACE;
summary = pointer only). S1 Mode C, S2 full target/grouped arithmetic,
S3 enumerated/stream obligations, PMA-BOM wording, runtime-N +
repeated/observed sweep requirements = substantially faithful.
REMAINING:
1. §3 selector mismatch: text claims "prior verified target row
   ranking" but defines a committed-token RECENCY SET (rank-0 on every
   observation; |S| collapses to 1 after first commit; N>=256 arms
   can't explore differing supports on a 256-token request). Fix =
   either specify real prior-row ranking w/ causal identity+lifetime,
   OR name/version the recency selector explicitly with cold-start
   policy + actual-count/sweep limits. First-step policy explicit.
2. Header line 3 stale authority statement (contradicts corrected §9).
PRECISION: Mode P "CHANGES" → "MAY CHANGE / does not guarantee
preservation"; §5 880-seal = historical protocol exemplar, not a
future-binary identity requirement; b7696144c = owner-reported
reference context, not authenticated evidence.
NO new apparatus/issue/host requirement. Diagnostic repair stays
authorized. d7 HOLD D1-D6 separate, not cured by spec text. FMLAL
landing review = reviewer's next item.

## SPEC v3 APPLIED (~22:57Z): blob f26fd8ba (commit 93d836b14)

addon2's authored in-place replacement carries the reviewer-211
corrections: §3 selector v1 = COMMITTED-TOKEN RECENCY (explicitly
admitted LowestGlobalIds cold start, honest count/sweep limits,
realized distinct-support counts reported), v2 = PRIOR VERIFIED TARGET
ROW RANKING as versioned alternative (every N explorable, separate
staging decision); §7 policy-valid-history vs stale-receipt; header
synchronized with §9; Mode P MAY-CHANGE (§0+§4); §5 880-seal =
historical exemplar, not a future-binary identity requirement.
ONE DECLARED SUBSTITUTION: addon2's raw body §9 had reverted to the old
fence wording (contradicting the corrected header + Main's §9 point) —
restored to the reviewed §9 (diagnostic AUTHORIZED; production-path +
N-sweep + economy fenced). Blob f26fd8ba01af0533bed7073383ee3deabd66acfa
= the re-verification target. d7 HOLD (D1-D6) separate.

## DIAGNOSTIC REPAIR TIP (~22:58Z): a42ac312b — routed for D1-D6 re-review

codex/arm-b4-vocab-budget @ a42ac312b (parent d7eb73708, base 12675c956;
3 files +861/-646; arch-boundary split per D6). Key dispositions: D2
Mode P now replays the PRODUCTION rejection sampler with an independent
residual threshold (dyadic-bin enumeration of the EXECUTED sampler); D3
byte proof on the ACTUAL BackendNative LM-head policy (single ordered
chain = that policy's production tree; 16×4 fold belongs to
GPUAlignedExpert MoE) with BITWISE comparisons + counted traffic; D4
verified receipts w/ strict stale/duplicate rejection; D5 production
couplings (sampleMTPSerialEquivalentTargetRow; G tie-break = lowest ID);
D6 portable/NEON TU split. Station W3 re-queues after the re-review.

## SPEC v3 RE-VERDICT (~23:01Z): reviewer-211 findings CLOSED; 2 local §3 contradictions remain

f26fd8ba verified: header/§9 sync, Mode-P may-change wording, historical
protocol qualification, explicit selector separation — CLOSED.
REMAINING (precise text/meaning fixes, no new gate):
1. v1 backfill contradiction: backfilled policy contradicts the stale
   "collapses to 1" sweep-limits paragraph — update the paragraph or drop
   the backfill; both cannot be normative.
2. Freshness ordering: (token, position, REQUEST generation) ledger
   ordered "generation descending" is degenerate (one request = one
   generation) — order by commit position/observation sequence
   descending; generation = identity/staleness only; dedupe IDs before N.
   Plus: v2 numeric range stays 1<=N<V.
a42ac312b diagnostic re-review STARTED by the reviewer.

## SPEC §3 FIXES APPLIED (~23:03Z): blob 10137aa3 (commit 607d8c274)

All four reviewer-215 §3 fixes in place: backfill normative with the
honest |S| = min(N, live vocabulary) semantics (collapse claim deleted;
ranked-vs-backfilled split reported per arm); freshness ordering =
commit position DESCENDING (generation = identity/staleness only);
dedup-first construction bullet; v2 range 1<=N<V. §9/header unchanged
from reviewed f26fd8ba. DECLARED: addon2's raw body carried the §9
revert AGAIN (second recurrence of the assembly artifact) — kept the
reviewed §9. Reviewer re-verdict pending; a42 diagnostic re-review
running.

### §9 RECURRENCE ROOT-CAUSED (addon2, ~23:02Z)

The local working draft (/tmp/i16a/b4-design-spec.md) never received the
§9 correction (applied only at the tracker canonical location) — every
subsequent assembly from the local draft re-carried the stale §9. Local
draft now synced; applied §3 verified as authored; 10137aa3 = spec of
record. **PROGRAM LESSON: when a correction is applied at the canonical
location, synchronize the AUTHOR's local working copy in the same cycle —
a diverged shadow draft silently reverts applied fixes on every future
assembly (shadow-copy divergence, the label-vs-body class).**

## DIAGNOSTIC RE-REVIEW #2 (~23:06Z): a42ac312b SOURCE HOLD — partial closures, A1-A5

CLOSED: D6 arch split (source-closed); D3 tree mismatch (BackendNative +
bit patterns + real grouped verifier call); D2 reused-U (production
helpers, independent residual threshold).
NEW FINDINGS (portable TU/fixtures):
- A1: selector hard-failure controls cannot pass (has_high_water_ never
  set; shortlist() unsorted/uncapped; lifecycle contradictions in the
  fixtures; no request-epoch binding; stale v2 class comment).
- A2: head-traffic assertion arithmetically false (227 executions vs
  reported 5); rowBytes = logical prepared-operand volume, NOT measured
  DRAM traffic.
- A3: Mode P negatives test the wrong condition (same q both sides;
  undercoverage q sums to .8 → silently-invalid PMF, not an omitted
  residual).
- A4: 64³ enumeration not exact for non-dyadic PMFs (midpoint-bin
  quantization; predicted 1e-9 failure on the truthful positive is TEST
  bias, not production defect); no loosened tolerances.
- A5: coverage exceeds execution (heads[1] constructed-only; odd-N
  branch dead; ID95 unclaimed-coverage mismatch; G still copies the
  authored target token — header claim false).
DISPOSITION: repair authorization intact; fixes stay in existing files;
no new apparatus. Reviewer → spec 10137aa3 (§3-only) next.

## SPEC ACK (~23:07Z): SPEC-FAITHFUL / DESIGN-ONLY at blob 10137aa3

Final delta vs reviewed f26fd8ba verified: backfill/count semantics
agree; freshness = commit position/observation sequence (generation =
identity only); freshest-per-ID dedup precedes N; v2 obeys 1<=N<V.
**S1-S6 SPEC FINDINGS CLOSED at this blob.** Intact: typed axes, strict
Mode C seeded identity, Mode P truthful-q/full-residual contract,
selected/grouped byte obligations, runtime-N/capture/PMA + measured
sweep requirements. v1 (committed-token-recency + LowestGlobalIds) =
explicitly a distinct heuristic, NOT certified useful/economical; v2
separately versioned.
SCOPE: DESIGN FAITHFULNESS ONLY — no production-unfence, N-sweep,
execution, or economy grant; diagnostic authorization unchanged.
a42 diagnostic = SOURCE HOLD A1-A5 (repair round 2 in flight); FMLAL =
SOURCE HOLD F1-F5 (repair tip 0b53fa7a4 in validation).

## DIAGNOSTIC ROUND-3 TIP (~23:10Z): 3002d37cc — routed for A1-A5 re-review

codex/arm-b4-vocab-budget @ 3002d37cc (parent a42ac312b; 2 files
+328/-169). A1: CommittedTokenRecencySelector v1 final-named w/ frontier
+ epoch maintained (all rejections EXECUTE + asserted), dedup/freshness/
cap/backfill aligned, epoch rebinding tested. A2: per-arm traffic
accounting, not_hardware_traffic=1 label, aggregate 227 asserted. A3:
wrong-CORRECTION negative with fixed p+draw (deviation DETECTED);
invalid-PMF rejected by draw-side validation. A4: fully-dyadic Mode P
fixtures → exact 64³ proof; endpoint fixture restored via production
clamped thresholds. A5: second geometry actually evaluated; even-N
backfill branch exercised; V-1/ragged covered; header boundaries
explicit (G harness-level corrected). Portable TU ± __ARM_NEON verified.
NOTE: a brief TU clobber during patching was recovered from the
committed a42ac312b blob (committed diff = authoritative).

## DIAGNOSTIC RE-REVIEW #3 (~23:23Z): 3002d37cc SOURCE HOLD — A2/A3/A4 deterministic, A1 partial, A5 narrow-closed

- A2: aggregate 228 vs asserted 227 (1+3+2+64+63+95); fix the aggregate,
  don't remove an exercised row; per-arm accounting + labels stand.
- A3: draw_q and correction_q still the SAME input (enumerate(p,·)
  changes both) → EXPECT_TRUE(differs) false; separate the two inputs
  (draw (1/4×4) vs correction (1/8,1/8,1/4,1/2); algebra predicts
  (31/64,21/64,1/8,1/16)); the omitted-residual negative REMAINS ABSENT.
- A4: boundary oracle wrong — residual normalizes to (1,2,3)=(.5,.25,.25)
  with r<=cumulative → threshold .5 selects token 1, not 3;
  clamp_unit_threshold comment misstates endpoints; keep the production
  convention; positive fixtures' 64-bin alignment CLOSED for these
  fixtures (retain the narrow claim).
- A1: frontier/epoch machinery present BUT constructor takes no epoch,
  the frontier is advanced by the same untrusted receipt, future
  positions are accepted → "lookahead unrepresentable" still false;
  enforce provenance or explicitly limit to assumed-committed inputs;
  exercise cap with fewer slots than distinct IDs; stale v2 text at :190.
- A5: NARROW CLOSED (geometry evaluated, reverse arms incl. 95,
  boundaries explicit); rename "backfill" → "reverse ordering" in the
  head (:39/:258).
DISPOSITION: focused repairs in existing files; no new apparatus; spec
ACK intact; fences unchanged.

## DIAGNOSTIC ROUND-4 TIP (~23:27Z): 688b0c14b — routed for A1-A5 re-review

parent 3002d37cc; 2 files +196/-67. A2: aggregate 228 (own miscount);
A3: separate draw_q/correction_q — wrong-correction negative with fixed
p+draw, independently predicted law (31/64,21/64,1/8,1/16) asserted
EXPECT_DOUBLE_EQ; OMITTED-RESIDUAL negative ADDED (correction re-draw
variant DETECTED); A4: endpoint fixture follows production convention
(token 1 at threshold .5; clamp endpoints asserted; comment corrected);
A1: constructor binds epoch; provenance scope EXPLICIT (receipts
ASSUMED-COMMITTED); "lookahead unrepresentable" RETRACTED (the honest
limit); cap-with-fewer-slots case added; A5: "reverse ordering" wording.
D6/D3/D2 stand. Reviewer re-verdict pending.

## DIAGNOSTIC RE-REVIEW #4 (~23:30Z): 688b0c14b SOURCE HOLD — three focused issues

1. A1: the future-position fixture contradicts its own epoch (reset
   epoch=9 then submits {9,100,7,0} → epoch guard throws unexpectedly;
   cap checks never execute); reorder or isolate the lifecycle; cap
   positions are 5..10 not 10..15. Constructor epoch-binding ACCEPTED;
   ASSUMED-COMMITTED scope limit ACCEPTED; lookahead requirement NOT
   reopened.
2. A3: EXPECT_NE-every-component contradicts the predicted law (token2
   = p(2)) — assert DISTRIBUTIONS differ, not every component.
3. A3: the omitted-residual negative is confounded (wrong-correction
   q already deviates) — clean counterfactual: same valid q both sides,
   unmodified helper yields p, then change ONLY the rejected-token
   source (predicts (5/16,5/16,3/16,3/16) vs p).
CLOSED: A2 (228 assertions), A4 (token-1 boundary + endpoints), A5
(coverage + scope limits), .8-negatives as total-mass validation.
NONBLOCKING PROSE: :190, :251-254, head:259 wording.

## DIAGNOSTIC ROUND-5 TIP (~23:33Z): 52fec643f + W3 window requested

parent 688b0c14b; 2 files +77/-101. A1 sequencing (future case after the
cap observations, current epoch 9/position 100, cap 5..10 corrected);
A3(1) vector inequality; A3(2) DE-CONFOUNDED omitted-residual negative
(same q both sides, positive control proves p, then only the
rejected-token source changes → (5/16,5/16,3/16,3/16) vs p); prose
(:190, observe-contract, head:259). PROCESS: next tip Linux-validated
BEFORE re-review (reviewer 222/223 commitment).
W3 (updated card, replaces d7/688 chain): build 52fec643f on .80 Release
NEON_I8MM SHARED_CORE=ON, run BOTH Unit targets, JUnit into seal
K14-B4-52FEC643, JUnit+tip to the reviewer together. .80 sequence: B1
named-attribution export FIRST (~25 min, K14-B1ATTR-12675C956), then W3.
Fallbacks: .79 after W2, or the station (device-free Unit work).

### W3 RUN-SHEET CORRECTION (sparks, ~23:34Z): Integration build, not Release

Correctness tests belong in an INTEGRATION NEON_I8MM build with
SHARED_CORE=ON (assertions + symbols) — NOT Release (Release is for
timing). Both TUs run with --no-tests=error; JUnit + exact tip/tree
retained; reviewer re-reviews only after those results. Final .80
sequence: (1) B1 attribution export (Release, measurement-only,
omp+kernel filters) → (2) W3 (Integration 52fec643f, both TUs,
JUnit seal). Coordinator run-sheet error caught by the executor —
recorded as the correctness-vs-timing build-class discipline.

### W3 ORDER CORRECTED (~04:21Z): rebase B4 onto 74797687f before the Integration run

Main's catch: 52fec643f sits on the old 12675c956 lineage — a W3 run
there would pin the JUnit + the reviewer's round-5 verdict to a tip
that can't land. ORDER: (1) addon2 fixes the open J2 item (primitive-
level counter controls, reviewer 242); (2) rebase the B4 chain onto
74797687f (semantic conflict resolution — the registrations + Main's
GPU guards both stay); (3) push → report the tip → W3 books
(Integration NEON_I8MM, both Unit targets, --no-tests=error, JUnit
seal) — runtime evidence only; the 224 HOLD stands until the reviewer's
re-verdict at the rebased tip. B4 candidate chain: W3 → re-verdict →
checks → candidate.

### COORDINATOR CORRECTION (~04:25Z): the "J2 fix" in my W3-order message was MY ARC CONFLATION

The "reviewer-242 primitive-level counter controls" item belongs to the
FMLAL arc (Test__CPUHalfOperands.cpp — the reviewer's 67afc62ea verdict,
which Main received and I only partially saw). Addon2's B4 branch has NO
open J2. Corrected B4 pre-W3 steps: (1) rebase onto 74797687f; (2) push
→ report tip → W3 books. The round-4 HOLD (688b0c14b) stands formally
until the reviewer re-verdicts at the rebased + Linux-validated tip.
Addon2 instructed to NOT start any J2 work (nothing to fix beyond the
rebase). Lesson: arc-crossing references (FMLAL J-numbers vs B4
A-numbers) must carry their arc label explicitly in routings.

### B4 REBASE RESOLVED (~04:30Z): the rebase FORM was wrong, not the lineage

Addon2's abort facts verified by the coordinator: merge-base
12675c956 vs 74797687f = 2948b1e49 (confirmed); FMLAL 67afc62ea NOT
ancestral to 74797687f (merge-base --is-ancestor: NO — addon2's fact 2
incorrect); **d7eb73708 touches ONLY tests/v2/CMakeLists.txt +18 + the
new Test__CPUDraftVocabularyBudget.cpp +982 — the B4 chain touches NO
kernel files.** The kernel ADD/ADD conflicts = the rebase form: plain
`git rebase 74797687f` replays everything since the merge-base (all
209 old-lineage commits incl. the whole ARM kernel tree). CORRECT
COMMAND: `git rebase --onto 74797687f 12675c956 codex/arm-b4-vocab-
budget` (the fork-point form) → only the 5 B4 commits replay →
expected conflicts: CMakeLists registration lines only (semantic:
registrations + Main's GPU guards both stay). Escalation only if a
kernel-file conflict appears with the correct form.

### B4 REBASED AND PUSHED (~04:24Z): e09903875 on 74797687f — W3 BOOKS NOW

The fork-point form replayed ONLY the 5 commits, all clean (zero
conflicts — the registration block landed where Main's GPU guards don't
reach). New tip e09903875325906d7bba6a190046c68d94252d4f (force-with-
lease pinned to 52fec643f, disclosed; old tip reachable). Diff vs
74797687f = exactly the 3 diagnostic files (CMake +38, portable TU
+970, head TU +471); ZERO kernel files. Post-rebase syntax: both TUs
clean. **W3 BOOKED on .80: Integration NEON_I8MM at e09903875, both
Unit targets, --no-tests=error, JUnit seal K14-B4-E09903875** → the
reviewer's round-5 re-verdict binds to this landable tip. Then the 27B
rider.

### W3 RESULT (~06:15Z): BUILD_OK; 1 FAILED — GroupedTargetRowsMatchSerialM1 (:439)

B4 Integration at e09903875 (the landable tip): BUILD_OK; the ctest
found the recorded set; **1 FAILED: CPUDraftVocabularyBudgetHead.
GroupedTargetRowsMatchSerialM1 (:439 — grouped verifier-rows vs serial
M1 mismatch in the HEAD TU)**; the portable TU + the head's other
cases pass. JUnit re-run ordered (--gtest_output=xml); seal
K14-B4-E09903875 = the fail reported honestly + JUnit. The result
routes to addon2 (diagnosis: fixture-side comparison question vs a
production grouped-verifier gap) + the reviewer (the round-5 re-verdict
at the landable tip, with JUnit in hand).

### W3 SEAL ANALYZED + ROUND-6 FIX (~06:27Z): cb9255c9a — the :439 fail = missing CPU-backend init

Addon2 pulled the seal from llaminar-spark-results and the JUnit
pinpointed it: the GroupedTargetRowsMatchSerialM1 failure is NOT a
comparison mismatch — workspace.allocate(...) returned FALSE at the
grouped test's admission (:439), the SAME missing-CPU-backend-init
class as the B1 bench (the two-arg DeviceWorkspaceManager allocates
through the device backend; the fixture skipped initCPUBackend(-1)).
FIXED at cb9255c9a (one file/6 lines: hasCPUBackend()/initCPUBackend(-1)
before the allocation + the include). **W3 SPLIT (the valuable part):
the byte-proof head test (SelectedRowsAreBitwiseIdenticalAndCounted —
the full bitwise oracle incl. counted traffic, multi-N, reverse arms,
second geometry) PASSED on Linux GCC; the portable TU PASSED (A1/A3/A4
repairs executed-valid).** Only the grouped-vs-serial comparison needs
the one-line init + a re-run. Routing: the re-run rides any window;
after it passes, the reviewer's re-verdict lands.

### B4 ORDER CONFIRMED (~06:27Z): rebase onto 0ad09600b → full-pair re-run → Main's application default

(1) addon2 rebases onto 0ad09600b (cb9255c9a sits on 74797687f; the
FMLAL commits landed between; expected clean, semantic CMake
resolution); (2) the re-run = the FULL PAIR (portable + head incl. the
grouped test) on the rebased tip (JUnit + the re-verdict bind to a
landable tip); (3) **Main's application default: YES if the reviewer
ACKs + all-architecture builds (the D6 split) + no new canonical
failures — with one Main-run canonical on the combined tip before
applying.**

### B4 REBASED ONTO 0ad09600b (~06:28Z): addf3db78 — STEP 2 BOOKED on .80

All 6 commits replayed CLEAN (the FMLAL additions didn't conflict with
the CMake registrations — the expected interaction didn't materialize).
New tip addf3db78a9d5c613b21700fd98294f60dc20dbd (chain 5bc9bcc10 →
de4adc05c → f561a0a76 → ef6fd7c4d → dbc1d05c0 → addf3db78 on
0ad09600b; force-with-lease pinned to cb9255c9a, disclosed). Diff vs
0ad09600b = exactly the 3 diagnostic files (CMake +38, portable +970,
head +477), ZERO production files. Both TUs clean post-rebase.
**STEP 2 BOOKED on .80: the full-pair re-run at addf3db78 (both Unit
targets, --no-tests=error, JUnit seal K14-B4-ADDF3DB78).** Green →
the reviewer's re-verdict → Main's application default (YES if ACK +
all-arch builds + no new canonical failures, one Main-run canonical on
the combined tip first).

### #204 BOUNDARY CLARIFICATIONS (~06:31Z) + the handoff spec

(1) the e099 failure = an ADMISSION gap (workspace.allocate FALSE
before the grouped-vs-M1 comparison) — NOT a byte-mismatch signal; the
grouped byte contract UNEXECUTED in that run; the portable/other-head
passes = separate reported results. (2) Neither e099 nor cb is
reviewer-declared landable; the latest completed B4 source verdict
remains 688 HOLD; a passing rerun does not auto-close source review.
HANDOFF SPEC (addon2 assembles with the step-2 re-run): full candidate
SHA/lineage back to 52/688 + build/compiler/ISA/host context + raw
failure/fixed-run JUnit + exits + the actual 'dual compiler' checks.
The reviewer reviews after the combined handoff. Non-installed
diagnostic authorization + the byte-exact contract unchanged;
production follow-ups Main/operator-owned.

### HANDOFF PACKAGE DRAFT (~06:31Z): high quality; one mapping clarification requested

Addon2's draft covers: the full lineage (fork point 12675c956 → the 6
rounds → the two rebases with the old→new mapping), the build/compiler/
ISA/host contexts per run, the raw JUnit + exits for BOTH runs (the
e099 admission gap + the pending step-2 fixed run), and the enumerated
dual-compiler checks (clang++-18 + g++-14 × compact/STT TUs; the B4
Unit TUs ± __ARM_NEON — syntax/semantic only, NOT full builds). ONE
CLARIFICATION REQUESTED: which rebased commit carries the cb9255c9a
grouped-fix content (the draft self-corrected mid-stream; the final
mapping must be 1:1 and precise). The package completes when the
step-2 re-run lands.

### B4 STEP-2 RE-RUN GREEN (~06:59Z): addf3db78 — 2/2 PASSED, JUnit failures=0

Seal K14-B4-ADDF3DB78-20261112 (sparks, .80): BUILD_OK; 2/2 PASSED
(portable 0.13s, head 0.06s); both JUnit XMLs failures=0 — the grouped
GroupedTargetRowsMatchSerialM1 passes after the backend-init fix
(addmission gap resolved; the same device-free class as B1). **The
combined handoff package is COMPLETE** (the 6-for-6 lineage mapping +
build contexts + both runs' raw JUnit/exits + the dual-compiler
enumeration) — the reviewer's round-5/6 re-verdict at the landable tip
addf3db78 proceeds. Main's application default: YES if ACK + all-arch
builds + no new canonical failures + one Main-run canonical on the
combined tip before applying.

### B4 COMBINED-TIP CANONICAL GO (~06:59Z): Main's campaign-src-b4c (tree 54b9e8e28) started on .80

addf3db78 confirmed directly on 0ad09600b (6 commits, tests-only, the
head TU aarch64-guarded). Main's canonical = the pre-application gate
per the default: build + the focused CPUDraftVocabularyBudget pair +
the full Unit+preflight. B1ATTR2 already completed + sealed
(K14-B1ATTR2-12675C956-20261112, 16 files) — .80 was idle, so the GO
was immediate. After the canonical: the B4 application decision (YES
if no new canonical failures beyond the inherited set) → the .80 queue
resumes (idea-04 M=1 → 27B rider).

### #204 PRELIMINARY BLOCKER (~07:09Z): round 5 accidentally DELETED the A4 threshold-boundary block

In Test__CPUDraftVocabularyBudget.cpp the 688→52 replay removed the
entire block after the omission negative; the current Mode-P test ends
at :929 with no replacement endpoint/tie calls; production calls use
ONLY (bin+0.5)/64 thresholds; the header still claims the endpoint
fixture is restored (:34). **A native 2/2 pass cannot certify the
strict acceptance tie, residual CDF tie, or endpoint clamp behavior.**
RESTORE: p=(1/2,1/4,1/8,1/8), q=delta0; threshold 1/4 accepts, 1/2
rejects; residual 1/2 selects token 1; retain the 0/1 clamp endpoint
controls. Preservation of previously closed A4 coverage.
BOUNDED POSITIVES: 6-for-6 source equivalence; A1 closure; the vector-
inequality repair; the truthful omission counterexample; the CPU-
backend init (admission repair, not numerical).

### A4 RESTORED (~07:10Z): 7a4790c77 — the block back verbatim from the round-4 lineage

Pushed (parent addf3db78): p=(1/2,1/4,1/8,1/8), q=δ0; threshold 1/4
accepts, 1/2 rejects (strict <); residual 1/2 → TOKEN 1 (the residual
(.5,.25,.25) with r<=cumulative); clamp endpoints asserted (0 stays 0;
1 → 0.99999994f); the header claim fixed; a stray splice brace fixed;
both TUs syntax-clean. **W3 RE-BOOKED at 7a4790c77 (the third tip move
— e099→addf→7a479, each disclosed): build + both Unit targets + JUnit
seal K14-B4-7A4790C7 — the grouped fix AND the restored A4 block both
ride it.** The reviewer's re-verdict material = this tip + the fresh
JUnit (the Linux-GCC-first process). Main's canonical re-books at the
same tip.

### #204 COMPLETED VERDICT (~07:19Z): SOURCE HOLD at addf3db78, narrowly for R1 (superseded by 7a4790c77)

Full 0ad..addf range reviewed (3 files +1485); all 6 rebased commits
source-equivalent; no production delta. R1 = the lost A4 boundary
coverage (the :34 claim vs the deleted block) — **7a4790c77 is the
reported R1 successor, not covered by this verdict.** ALL OTHER
CLOSURES STAND (A1/A3/A2/A5/D6 + CPU-init-before-admission). The
reviewer read both local seals: e099 = admission ASSERT_FALSE at :439
(portable 6/6, head 1/2, CTEST 8) — not a numerical comparison
failure; addf = portable 6/6 + head 2/2, exits 0 — owner-produced
records.
RECEIPT CORRECTIONS (not blockers): the failed SUMMARY retains the
superseded 'byte mismatch' diagnosis + mistypes the tip; the fixed
SUMMARY says portable=2 but the XML has 6 (2/2 = CTest entries); §4
mixes Release/Integration; the B4 TUs' evidence = Clang syntax +
Linux GCC build/run (not the performance TUs' GCC syntax checks); the
pasted XML excerpts were abridged. Preserve raw history + attach
corrected descriptions.
#204 continues awaiting the 7a4790c77 fresh combined handoff under the
Linux-first process; production/N-sweep/economy + Main's
application/canonical decisions separately fenced.

### CORRECTED-DESCRIPTIONS PACKAGE (~07:19Z): C1-C6 attach to the handoff; raw seals untouched

C1: the failed-run diagnosis = the ADMISSION gap (not byte mismatch) +
the tip typo; C2: "tests=2" = CTest entries (the XML truth = 8/8 test
cases PASS); C3: both .80 runs = INTEGRATION builds (the §4 mix
corrected); C4: the B4 TUs' executed evidence = the Linux GCC seals
(the performance-TU g++-14 syntax checks are separate diagnostics); C5:
the pasted XML excerpts were abridged (disclosed); C6: the R1 gap is
superseded by the 7a4790c77 restoration (the #204 continuation
vehicle). The package awaits the 7a4790c77 W3 re-run's fresh JUnit.

### TERMINOLOGY CORRECTION (reviewer, ~07:20Z): e099 failed ASSERT_TRUE(workspace.allocate(...)) at :439 (actual=false/expected=true) — the shorthand 'ASSERT_FALSE(actual)' was not the source macro. The admission-before-comparison conclusion and the R1-only HOLD are unchanged.
