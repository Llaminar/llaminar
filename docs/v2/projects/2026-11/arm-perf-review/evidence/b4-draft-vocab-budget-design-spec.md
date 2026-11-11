# B4 — DRAFT VOCABULARY BUDGET: design specification (design-only)

Status: DESIGN RECORD (corrected in place; this file supersedes blob 69cf9526). This document grants NO implementation, NO N-sweep execution, and NO economy admission; implementation and the N-sweep stay fenced until a separate explicit operator instruction. Until then the only approved execution scope remains the B1 diagnostic bench and the single post-campaign greedy rider. Governing inputs: the reviewer design inputs in the existing tracker (both entries, including reviewer200's terminology addendum), Main's fences, and the Phase A measured facts (owner-reported). Terminology: **draft vocabulary budget** — never "proposal length/depth" (that is MTP depth). The three independent meanings stay separate: shortlist SELECTOR, proposal DRAW LAW, verifier COUPLING.

## 0. The five consulted questions, answered

1. **The correct boundary is not greedy-vs-stochastic.** It is: unchanged TARGET law p + truthful PROPOSAL law q + the correct installed coupling/correction. A shortlisted qS with correct residual handling preserves p. What a changed q does depends on the installed coupling, and the three couplings differ: under the CPU seeded serial-target-draw coupling (Mode C) the position-keyed target draw MUST preserve the same seeded target tokens for every admitted N; a token change is a FAILED compatibility proof, not something a q or version renaming can license. Under probability-rejection (Mode P) a changed q changes acceptance rate and, through the changed proposal identity, the old seeded output stream while the target law stays preserved. Under the greedy target (Mode G) the emitted stream is the target's greedy stream regardless of the proposal.
2. **Existing special cases are already in the repo and are not new qualifications:** the one-hot DRAFT proposal with stochastic target (greedy draft / one-hot q) exists; the CPU seeded serial-target-draw coupling (seed != 0) draws directly from target p with a position-keyed threshold and preserves the serial seeded target tokens under the Mode C contract. Neither is silently replaced by shortlist machinery.
3. **Operator contract (approved wording, normative for every mode):** "B4 is an explicit versioned draft-proposal policy evaluating only admitted live vocabulary rows. Full target head/sampling policy, grouped target arithmetic and committed-state authority remain unchanged. Shortlist miss causes normal speculative correction, never target restriction to the shortlist or a full-draft-head retry. Proposal policy and verifier coupling are separate typed axes." Binding clauses:
   - The full target head and the COMPLETE sampler are unchanged; the target distribution p is never renormalized over S.
   - ALL unchanged grouped target verifier rows remain byte-equal to serial M=1, besides the selected draft rows the policy evaluates.
   - Mode G takes the argmax AFTER the existing target transforms.
   - Accepted-state and bonus authority are unchanged.
4. **Recommendation: greedy-target first** as the smallest engineering slice on the measured CPU/IQ3_S lane; stochastic modes staged afterward as separately proved modes. Staging is an engineering order, NOT a mathematical ban on stochastic shortlists.
5. **Building S must not evaluate the full head.** Exact top-N of current full logits requires the full head and saves nothing; S comes from one fixed cheap causal selector. A = sum min(p,qS) = 1 - TV(p,qS) <= p(S) applies only to probability-rejection; deterministic A = p(a); the CPU seeded mode has its own RNG coupling. The Phase-A-derived 1.15-1.18x ceiling assumed eliminating ALL draft-head work with everything else fixed; N controls how close the budget gets, and changed acceptance/work invalidates it as an absolute bound. The no-MTP 10.2% LM-head fraction is NOT additive.

## 1. Typed specification

```
DraftProposalPolicy {
    policy_version : uint32          // monotonic; change = new policy
    selector       : SelectorPolicy  // typed, versioned (§3)
    draw_law       : ProposalDrawLaw // DeltaOnShortlist | NormalizedShortlist
    vocab_budget   : VocabBudget
}
VocabBudget   = FullVocabulary | ShortlistLimit { checked_positive_N }
VerifierCouplingPolicy {           // SEPARATE typed axis, own version
    coupling_version : uint32
    kind             : CouplingKind // GreedyTargetMismatch |
                                     // CpuSeededSerialTargetDraw |
                                     // ProbabilityRejection
}
```

- **FullVocabulary ("full")**: the CURRENT full-vocabulary DRAFT proposal computation with MTP STILL ENABLED — today's sidecar final projection over all V rows, today's draw law, today's installed seeded coupling, byte-identical behavior. This is the reference point (1.0x). It is NOT no-MTP. The spelling "full" is explicit and reserved; it is never aliased to a numeric N.
- **ShortlistLimit{N}**: N = maximum LIVE distinct candidate token IDs in the shortlist S(h) for the current draft position.
  - 1 <= N < V (V = 248320 on the 35B). N = 0 or N >= V: admission error, no clamping, no rounding.
  - N is a checked positive runtime value and runtime DESCRIPTOR DATA in ONE binary (no template parameter, no CMake option, no per-N build).
  - The ACTUAL live |S(h)| is explicit and reported per draft position. |S(h)| < N only through the declared selector/eligibility semantics (deduplication, verified-row availability) — never through resource-error recovery. |S(h)| = 0 where the draw law requires nonempty support: hard failure.
  - No silent N=V-to-full aliasing: the one-build draw law on S differs from the full path's law (shortlisted/one-hot q vs the CPU sampled full-draft), so an N=V request is rejected as out of range. If a future build proves a byte-identical draw law at N=V, that is a separately versioned policy — never a silent alias.
  - Default = FullVocabulary. Every non-full N is an explicitly selected policy; there is no default shortlist.
- "One build" means: same Release binary, compile/source/ISA identity fixed, one fresh startup instance per N, policy bound BEFORE capture. Mutating N inside a running request is out of scope (that requires an admitted dynamic-capacity contract; no hot allocation, no readback, no recapture).

## 2. Selected-row byte-proof specification

The shortlisted draft head evaluates only rows in S(h). Proof obligations, minimum set (per the reviewer's clause list):

a. **Selected-row arithmetic identity**: every selected row is identified by its GLOBAL vocabulary row ID (never by shortlist position) and uses the SAME full-K reduction as the identical full-head row: same weight bytes (IQ3_S per-row blocks + scales, ragged/sharded tails included), same dequant codebook/scales, same accumulator order and FP32 reduction tree. Proof: byte-compare of the selected rows' logits against the identical rows of the CURRENT full head on the same hidden row, across ragged/sharded IQ3_S and sharded/replicated layouts.
b. **Proposal ownership**: S is valid, unique, nonempty, and LIVE (in-vocabulary, currently owned rows); deterministic, versioned tie-breaking; NO lookahead — the selector must not consume verifier rows ahead of the proposal or any current-step full-head evaluation.
c. **Greedy mode**: identical serial target-greedy token stream AND identical continued KV/GDN/sampler state, including winner-absent-from-S, ties, EOS, and rejection positions.
d. **Probability-rejection mode** (when staged): tiny-vocab analytic/enumeration proofs plus production replay for the cases q = p, one-hot q, disjoint support / omitted mass, p(y) = 0, normalization and underflow behavior, and acceptance-threshold endpoints; a finite-precision and RNG contract for the draw; the stream cases all-accept, first and later rejection, every depth, bonus, stop/EOS, partial work, and continued state after reset/prefix reuse. A VALID zero probability at a token is distinct from an INVALID zero total proposal mass. qS must be truthful across the residual support. The negatives (wrong-q ratio in the acceptance test, omitted residual mass) must FAIL.
e. **Seed/version**: an unchanged coupling retains its existing records and stream conventions; a changed q is an explicit policy/version approval event, never a silent substitution.
f. **Ownership/runtime**: PMA AUTHORITY unchanged, but ALL shortlist, ranking, gather, and probability workspace plus any retained selector state are included in the BOM and admission under that same authority — same authority does not mean unchanged memory. No new allocation classes; capture identity includes policy_version, selector version, draw_law, coupling version, and N where embedded.
g. **Economy**: counted rows/bytes (build-S cost, selected-row gather, normalize, accepted-prefix target work); cost per EMITTED token as the primary measure.

## 3. Selector identity

- ONE fixed, versioned, cheap, CAUSAL selector, available BEFORE the draft step runs. No lookahead into verifier rows ahead of the proposal; no current-step full-head evaluation (full logits then top-N saves no dominant weight traffic and is not an admitted selector).
- Selector v1 (recommended): prior verified target row ranking — the ranking of token IDs by their most recently VERIFIED target evidence, used as a candidate-set HEURISTIC. It is explicitly NOT certified top-N of the current logits. Concrete lifecycle (normative):
  * State: an upsert ledger of (global token ID, last verified rank, last observation generation) over the request's vocabulary, plus a generation counter; request-scoped, no cross-request carryover.
  * Initial state (request start / policy reset / prefix reuse): the ledger is empty; the first proposal's shortlist is the deterministic fallback = lowest global IDs [0, N) in ascending order, versioned.
  * Observation point: after each verifier commit at a logical position, the verified target token at that position is observed — under Mode G the committed greedy token (rank 0), under Mode C the committed target draw (rank 0), under Mode P the emitted token (rank 0) — as an upsert (token -> rank 0 at the new generation); no eviction within a request.
  * Rejection at a position: the correcting target token is itself verified evidence and is observed by the same upsert rule.
  * Full-accept/bonus continuation: every committed bonus position observes its verified token in commit order.
  * Global positions and request generation: logical per-request positions; a new request restarts from the initial state.
  * Shortlist construction: order the ledger by (rank ascending, generation descending, token ID ascending) and take the first N distinct, in-vocabulary, live IDs; fewer available IDs give |S| = available count under the declared eligibility semantics (reported); the empty-ledger fallback covers the first proposal.
  * No hidden full-head rescue: the selector's only inputs are the verified evidence stream and the vocabulary bound; no current-logit or full-head input exists in its interface.
- The proposal draw law q must be TRUTHFUL w.r.t. the actual current selected logits: on S, either the declared deterministic winner (q = delta_a) or the actual normalized shortlisted distribution qS = normalize(logits|S). Rank scores or OLD full-head probabilities are never presented as q unless they actually generated the token.
- The selector identity (kind + version + inputs) enters config/results/capture identity; changing the selector is a policy-version change.

## 4. Per-mode coupling specs (separate compatibility contracts)

The three installed/planned couplings are NOT interchangeable. p is unchanged at every admitted N in all modes.

- **Mode G — greedy target mismatch correction** (recommended first): acceptance iff target-argmax == draft proposal. With a shortlist proposal, q = delta_a with a in S. Shortlist miss = normal speculative correction (target token emitted, draft state resynced) — never target restriction to S, never a full-draft-head retry. Byte proof: identical serial target-greedy tokens + identical continued KV/GDN/sampler state.
- **Mode C — CPU seeded serial-target-draw** (EXISTING coupling; never silently replaced): seed != 0 selects the serial-sample-equivalent host stochastic path; the rejection sampler draws directly from target p with a position-keyed threshold (accept iff token == draft proposal). The target RNG stream is keyed by LOGICAL POSITION + existing stream/token-ID conventions — NOT by N, shortlist rank, depth, or rejection count. The Mode C compatibility contract is STRICT: identical p, the position-keyed target draw and committed state MUST preserve the same seeded target tokens for every admitted N; a token change is a FAILED compatibility proof, not licensed by q or version renaming. Proof: byte-equivalent target rows, accepted-state continuation, unchanged target RNG consumption per logical position, and the recorded seeded token stream identical at full and at every admitted N.
- **Mode P — probability-rejection** (new machinery; staged last): y ~ qS; accept min(1, p(y)/qS(y)); on rejection draw r(x) = [p(x) - qS(x)]+ / Z over the FULL target support. The TV formula A = 1 - TV(p, qS) = sum min(p, qS) belongs ONLY to this mode. qS = the normalized shortlisted logits and does NOT reuse the old full-vocab normalizer; keeping the OLD q in the ratio while the proposal changed is INCORRECT and must fail the negative controls. Under Mode P a changed q changes acceptance rate and, through the changed proposal identity, the old seeded output stream while the target law stays preserved — an explicit policy/version approval event.
- Unsupported shortlist+coupling combinations REJECT at admission — never silently disable MTP, never fall back to full.

## 5. Sweep protocol (DESIGN ONLY — not permission to run)

- Points: full + N in {64, 128, 256, 512, 1024, 4096}; N = 1 as an extreme diagnostic; add one larger point only if measured quality is poor near the frontier; refine near the measured frontier afterward. Pilot points are NOT approved defaults, NOT a supported-N envelope, and NOT permission to run.
- Model/format: 35B IQ3_S primary; at least one other format later, per the calibration list. MTP dynamic (full baseline = MTP-enabled full-draft).
- Placements: mixed@20T, fast-only@10T (taskset 5-9,15-19), slow-only@10T (taskset 0-4,10-14), 1T. Release; unprofiled canonical timings; isolated per-fit comparisons; one fresh startup instance per point.
- Measures per point: cost per EMITTED token (primary); accepted-prefix length distribution; rejection position distribution; target transactions; actual depth/controller state (updates, budget-limited events); counted draft-head weight traffic actually avoided (rows/bytes) alongside the added build-S/gather/normalize costs. Bandwidth fractions are NOT additive; profiler facts are not E2E speedup claims.
- Trial hygiene (normative): REPEATED FRESH-PROCESS paired/rotated trials — each comparison starts from an equivalent initial state with the COMPLETE controller and cache start state recorded; raw variation and uncertainty are reported, not only medians; the exact sampler parameters and seed are recorded per run; hold-out prompts; no seed search, no retry-until-green.
- Paired-timing protocol of record (normative reference): the approved greedy-rider protocol AS EXECUTED in seal K14-I16RIDER-880429059 (tracker @ b7696144c): the retained Release binary and artifacts at 880429059, the same GGUF/prompt bytes/context/256-output policy/cache-reset/placement, the RESOLVED-ACTUAL request sampler recorded (not plan/CLI), one benchmark transaction with normal warmup/repeats, exact generated-text signature and token digest captured, model/binary/config identities recorded, no seed search and no GPU tooling. Every B4 timing run cites this protocol and its own seal; only the policy descriptor changes between paired arms.
- Placement labels are not evidence: the CPU placement lists bind to the ACTUAL host, topology, and observed placement of the executing run.
- Neither acceptance nor speed is monotone in N — the sweep exists to measure, not to assume.
- Calibration identity list per point: weights revision, per-tensor formats, ISA/backend, placement/sharding, workload, sampler/coupling (kind + versions), MTP-depth policy, prefix policy, policy_version, selector version, draw_law, N, binary/source identity, PMA/workspace identity.
- No automatic tuner, no new gate apparatus: explicit authored N values only; calibration persistence via existing config/tracker surfaces.

## 6. One-build mechanics

- N = runtime descriptor data (no template/CMake/per-N binary); compile/source/ISA identity fixed across the sweep; separate fresh startup instances per N; policy bound BEFORE graph capture.
- PMA admission for workspace/weights/layout unchanged from the existing authority; runtime-N dispatch handles odd/tail live support.
- policy_version + selector version + draw_law + coupling version + N recorded in config, results, and capture identity. No recapture, no readback, no hot allocation.

## 7. Correctness coverage list

- Non-aligned counts and sparse IDs: N in {63, 65, 127, 129, V-1}; every positive admitted N has defined behavior with BYTE/live-traffic proof; ragged IQ3_S tails included.
- Zero/invalid: N = 0, N >= V, out-of-range token IDs in S, duplicate IDs -> precise admission/execution errors.
- Stale generation: selector IDs from a stale position/generation -> hard failure, never silent acceptance.
- Zero-mass support: S with zero mass under the current logits -> the declared draw-law behavior; empty legal support FAILS.
- Ties: deterministic, versioned tie-breaking.
- Reset/prefix: policy reset behavior; prefix-cache interaction where the policy identity is part of the cacheable identity; stale/prefix-hit revalidation; continued state after reset and prefix reuse.
- Stream cases: all-accept, first and later rejection, every depth, bonus, stop/EOS, partial work.
- No rounding of the IQ3_S budget; kernel padding is NOT live support.
- Probability semantics: a valid zero probability at a token is distinct from an invalid zero total proposal mass; qS is truthful across the residual support.
- Negative controls: wrong-q ratio, omitted residual mass, lookahead selector, duplicated IDs — each must FAIL.

## 8. Recommendation carried forward

Stage greedy-target (Mode G) first on the measured CPU/IQ3_S lane; then Mode C (the existing seeded coupling — strongest same-seed compatibility experiment); Mode P last. `full` remains the default; no default N is justified by this design. Unsupported combinations reject. The design bounds no economy result: nothing here is admitted until measured under the fences.

## 9. Fences (restated)

DESIGN RECORD. Production-path implementation, the N-sweep, and economy/admission remain FENCED until a separate explicit operator instruction; a spec-faithful review verdict grants no production implementation and no economy admission. The DIAGNOSTIC implementation is AUTHORIZED (operator answer recorded: "Yes — diagnostic authorized"); its scope is validation of the design's claims, not productionization. No tuner/gate apparatus, no new issues/artifacts; tracker-only. Approved execution scope: the B1 diagnostic bench, the single post-campaign greedy rider, and the authorized B4 diagnostic (d7eb73708).
