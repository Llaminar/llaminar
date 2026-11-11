# B4 — DRAFT VOCABULARY BUDGET: design specification (design-only)

Status: DESIGN SPEC for operator review. Grants NO implementation, NO
N-sweep execution, NO economy admission. Operator review = second go for
implementation; until then the only approved execution scope = the B1
diagnostic bench + the single post-campaign greedy rider. Governing
inputs: reviewer design inputs (two tracker entries incl. reviewer200
terminology addendum), Main's fences, Phase A measured facts
(owner-reported). Terminology: draft vocabulary budget — never "proposal
length/depth" (that is MTP depth). Separate meanings: shortlist SELECTOR,
proposal DRAW LAW, verifier COUPLING. (Received from arm-addon2, seq 12
attachment; archived verbatim structure — full text available in the
attachment and in the commit history of this file.)

## 0. Five consulted questions, answered
1. Boundary = unchanged TARGET law p + truthful PROPOSAL law q + correct
   installed coupling; shortlisted qS with correct residual preserves p;
   changed q changes acceptance rate + old seeded stream, not target law.
2. Existing special cases in repo (one-hot/star target, CPU seeded
   serial-target-draw) are NOT new qualifications; never silently replaced.
3. Operator contract wording carried verbatim (see §1 in the attachment).
4. Greedy-target first = smallest slice; stochastic staged separately;
   staging = engineering order, NOT a mathematical ban.
5. Building S must NOT evaluate the full head; A = 1-TV(p,qS) only for
   probability-rejection; deterministic A = p(a); CPU seeded = own RNG
   coupling. 1.15-1.18x ceiling assumed eliminating ALL draft-head cost
   with everything else fixed; N controls closeness; not an absolute
   bound; no-MTP 10.2% fraction NOT additive.

## 1. Typed specification
DraftProposalPolicy { policy_version, selector, draw_law,
vocab_budget }; VocabBudget = FullVocabulary | ShortlistLimit{
checked_positive_N }; VerifierCouplingPolicy { coupling_version, kind =
GreedyTargetMismatch | CpuSeededSerialTargetDraw | ProbabilityRejection }
— SEPARATE typed axis, own version. FullVocabulary = CURRENT full-vocab
draft computation, MTP STILL ENABLED, today's draw law + installed seeded
coupling, byte-identical (reference 1.0x; NOT no-MTP; never aliased to N).
ShortlistLimit{N}: 1<=N<V; N=0/N>=V = admission error (no clamp/round);
N = runtime descriptor DATA in ONE binary; actual live |S(h)| explicit +
reported (smaller only via declared selector/eligibility semantics, never
resource-error recovery; empty legal support fails); NO silent N=V→full
aliasing (draw law differs; future byte-identical proof = separately
versioned policy); default = FullVocabulary; no default shortlist. One
build = same binary, one fresh startup per N, policy bound BEFORE capture;
N mutation inside a running request out of scope.

## 2. Selected-row byte-proof spec
a. Selected-row arithmetic identity: GLOBAL row ID (never shortlist
   position), same full-K reduction as the identical full-head row, same
   IQ3_S per-row blocks/scales + ragged/sharded tails, same dequant
   codebook/scales, same accumulator order + FP32 tree. Proof:
   byte-compare selected rows vs identical full-head rows (sharded/
   replicated layouts).
b. Proposal ownership: valid/unique/nonempty/LIVE S; deterministic
   versioned tie-breaking; NO lookahead (selector must not consume
   verifier rows ahead of the proposal or current full-head eval).
c. Greedy mode: identical serial target-greedy tokens + continued KV/GDN/
   sampler state (winner-absent, ties, EOS, rejection positions).
d. Probability-rejection (when staged): analytic/enumerated tiny-vocab +
   recorded-state/RNG replay; wrong-q + omitted-residual negatives FAIL.
e. Seed/version: unchanged coupling retains records/stream conventions;
   changed q = explicit policy/version approval, never silent.
f. Ownership/runtime: existing PMA/arena/live-payload/capture authorities
   only; capture identity includes policy_version + selector version +
   draw_law + coupling version + N where embedded.
g. Economy: counted rows/bytes (build-S, gather, normalize, accepted-
   prefix target work); cost per EMITTED token = primary measure.

## 3. Selector identity
ONE fixed versioned cheap CAUSAL selector available BEFORE the draft —
no lookahead, no current-step full-head evaluation (full-logits-then-top-N
= fake savings, not admitted). Selector v1 = prior verified target row
ranking as candidate-set HEURISTIC (explicitly NOT certified top-N). q
truthful w.r.t. actual current selected logits (defined winner q=delta_a
or actual normalized qS); rank scores / old full-head probabilities never
presented as q unless they generated the token. Selector identity (kind +
version + inputs) in config/results/capture; changing = policy-version
change.

## 4. Per-mode coupling specs (separate compatibility contracts)
Mode G (greedy target mismatch correction; first): accept iff
target-argmax == draft proposal; q=delta_a with a in S; miss = normal
speculative correction (never target restriction/full-head retry). Byte
proof: identical serial target-greedy tokens + continued KV/GDN/sampler.
Mode C (CPU seeded serial-target-draw; EXISTING, never silently
replaced): seed!=0 serial-sample-equivalent; rejection sampler draws from
target p with position-keyed threshold (accept iff token==draft
proposal); target RNG keyed by LOGICAL POSITION + existing stream/token-ID
conventions — NOT N/rank/depth/rejections; changed q can change acceptance
+ old seeded stream while preserving law → explicit policy/version
approval; plausibly the strongest same-seed compatibility experiment.
Mode P (probability-rejection; last): y~qS, accept min(1,p/qS), on
rejection r=[p-qS]+/Z over FULL target support; A=1-TV(p,qS)=sum min
ONLY here; qS = normalized shortlisted logits (does NOT reuse old
full-vocab normalizer); old-q-in-ratio = INCORRECT, negative controls
FAIL. Unsupported shortlist+coupling combinations REJECT at admission —
never silently disable MTP / fall back to full.

## 5. Sweep protocol (DESIGN ONLY — not permission to run)
Points: full + 64/128/256/512/1024/4096; N=1 extreme diagnostic; one
larger point only if quality poor near frontier; refine near measured
frontier. Pilot points NOT approved defaults/supported-envelope/permission
to run. Model/format: 35B IQ3_S primary + one other later. MTP dynamic
(full baseline = MTP-enabled full-draft). Placements: mixed@20T,
fast-only@10T (5-9,15-19), slow-only@10T (0-4,10-14), 1T. Release;
unprofiled; isolated per-fit comparisons; one fresh startup per point.
Measures: cost per EMITTED token (primary); accepted-prefix length;
rejection position; target transactions; depth/controller state; counted
draft-head traffic avoided + build-S/gather/normalize costs; fractions
NOT additive; profiler facts ≠ E2E speedup. Trial hygiene: equivalent
initial state + state reset (extra rejections = work amplification, not
hysteresis); hold-out prompts; paired-timing shape per the rider protocol
(same binary/prompt/context/256-policy/cache-reset/placement; only policy
descriptor changes); no seed search/retry-until-green. Neither acceptance
nor speed monotone in N. Calibration identity per point: weights
revision, per-tensor formats, ISA/backend, placement/sharding, workload,
sampler/coupling kinds+versions, MTP-depth policy, prefix policy,
policy_version, selector version, draw_law, N, binary/source identity,
PMA/workspace identity. No automatic tuner/gate apparatus; explicit
authored N only.

## 6. One-build mechanics
N = runtime descriptor data (no template/CMake/per-N binary);
compile/source/ISA identity fixed; separate fresh startups per N; policy
bound before capture; PMA admission unchanged; runtime-N dispatch handles
odd/tail live support; policy_version + selector + draw_law + coupling +
N in config/results/capture identity; no recapture/readback/hot alloc.

## 7. Correctness coverage
Non-aligned 63/65/127/129/V−1 + every positive admitted N defined with
BYTE/live-traffic proof (ragged IQ3_S tails); zero/invalid/duplicate/
stale-generation/zero-mass → precise errors or hard failure; ties =
deterministic versioned; reset/prefix + prefix-cache identity
revalidation; NO IQ3_S budget rounding; kernel padding ≠ live support;
negative controls (wrong-q, omitted residual, lookahead, duplicates)
must FAIL.

## 8. Recommendation
Greedy-target (Mode G) first on CPU/IQ3_S; then Mode C (existing seeded —
strongest same-seed compatibility experiment); Mode P last. full =
default; no default N justified. No economy result bounded by this design.

## 9. Fences
DESIGN ONLY. No implementation / N-sweep execution / economy admission /
tuner / gate apparatus / new issues/artifacts; tracker only. Operator
review of this spec = second go for implementation. B1 + single
post-campaign rider = only approved execution scope.
