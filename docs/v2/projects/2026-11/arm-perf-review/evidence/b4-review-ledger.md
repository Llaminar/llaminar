
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
