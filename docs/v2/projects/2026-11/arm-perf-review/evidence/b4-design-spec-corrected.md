# B4 — DRAFT VOCABULARY BUDGET: design specification (corrected blob for reviewer re-review)

(Verbatim corrected blob received from arm-addon2 seq 16, ~15:24Z — S1–S6
applied to the existing spec per reviewer206. The full corrected text is
preserved in this file's git history at commit time; key corrections
summarized here with the structure unchanged from
b4-draft-vocab-budget-design-spec.md.)

STATUS header: DESIGN RECORD (corrected blob for reviewer re-review).
Grants NO implementation, NO N-sweep execution, NO economy admission;
implementation and the N-sweep stay FENCED until a separate explicit
operator instruction (S6 applied — "operator review = second go"
REMOVED). Only approved execution scope: B1 diagnostic bench + single
post-campaign greedy rider.

S1 APPLIED (§0/§4 Mode C): under the CPU seeded serial-target-draw
coupling (Mode C) the position-keyed target draw MUST preserve the same
seeded target tokens for every admitted N; a token change is a FAILED
compatibility proof, not licensed by q/version renaming. Blanket §0
sentence qualified per-coupling (Mode P: changed q changes acceptance +
old seeded stream, target law preserved; Mode G: emitted stream is the
target's greedy stream regardless of proposal). Terminology: one-hot
DRAFT proposal with stochastic target.

S2 APPLIED (§3 operator contract inlined as normative clauses): full
target head + COMPLETE sampler unchanged (temperature, top-k/top-p/
filters, penalties/masks, tie-breaks); p never renormalized over S; ALL
unchanged grouped target verifier rows byte-equal serial M=1 besides
selected draft rows; Mode G argmax AFTER existing target transforms;
accepted-state/bonus authority unchanged.

S3 APPLIED (§2d/§7 expanded proof coverage): tiny-vocab
analytic/enumeration + production replay for q=p, one-hot q, disjoint
support/omitted mass, p(y)=0, normalization/underflow, threshold
endpoints; finite-precision/RNG contract; stream cases all-accept →
first/later rejection → every depth → bonus → stop/EOS → partial work →
continued state after reset/prefix reuse; VALID zero probability ≠
INVALID zero total proposal mass; qS truthful across residual support;
wrong-q negatives kept.

S4 APPLIED (§3 selector v1 lifetime + §6 PMA precision): selector v1's
source rows/global positions/request generation/lifetime specified
across first-proposal/full-accept/bonus/rejection/prefix-reset; no hidden
full-head rescue. "PMA AUTHORITY unchanged; ALL shortlist/ranking/gather/
probability workspace + retained state included in BOM/admission" (same
authority ≠ unchanged memory requirements).

S5 APPLIED (§5 trial hygiene normative): REPEATED FRESH-PROCESS paired/
rotated trials; complete controller + cache start state recorded; raw
variation/uncertainty reported; exact sampler params/seed recorded;
rider-protocol reference resolvable; placement lists bound to the ACTUAL
host/topology/observed placement.

S6 APPLIED (§9): implementation + N-sweep FENCED until separate explicit
operator instruction; spec-faithful verdict grants nothing.

(Also noted: compact IQ3S decode bench committed d096438e2 on the idea-08
lane — syntax-clean, unaffected by the B4 hold; and the lane's B4 HOLD
compliance: no B4 code/harness written this round — the hold stood until
the operator's literal authorization resolved; the only B4 action was the
S1–S6 spec text.)
