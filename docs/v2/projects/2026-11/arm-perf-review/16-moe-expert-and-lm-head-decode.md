# Idea 16 — MoE routed experts, GDN projection and LM head in CPU decode

Status: **Open**. The measured attribution is in `00-measured-baseline.md`;
**the bundle part is closed** — `408ff90e5` is verified end to end
(2026-11-08): router bundle 668→176 µs/call (16 projections, K=2048),
`MOE_EXPERT_FFN` 803→274 µs/call, 35B MTP dyn 17.45→27.19 (+56%), no-MTP
13.27→20.48 (+54%). Remaining in this idea: GDN (394 µs × ~30 layers ≈ 12 ms,
now the largest stage), LM head 5.7 ms/token, shared expert, and the grouped
verifier economics (idea 18).

## Finding (35B-A3B, GB10, MTP off, per decode step of about 75 ms)

| Stage | Share | Notes |
|---|---|---|
| Routed experts | 43% | 803 us per layer |
| Fused router + gate/up bundle | 35% | 668 us per layer; the bundle part is fixed by `408ff90e5` |
| GDN projection | 18% | 440 us |
| LM head | | 5.7 ms per token on a 248K-wide vocabulary |

Verifier context multiplies these further: experts 3.8 ms per grouped call,
LM head about 7 ms per row.

## Ideas

1. **Routed experts.** Pool the 8 selected experts' down projections into one
   workshare, as was done for gate/up. Experts are 512x2048 / 2048x512,
   so each down projection has only 32 tasks at chunk width 64. For grouped
   verifier rows, group rows that route to the same expert so its weights
   stream once.
2. **GDN projection.** 440 us for the QKV/Z projection pieces at M=1. Check
   whether these are issued as separate regions; if so, pool them (idea 13).
3. **LM head.** 248K x 2048 at about 5.7 ms per token is weight-bandwidth
   bound (about 0.5 GB per token for Q6_K/Q8_0 heads). Options:
   - top-k-only verification heads where the sampler contract allows;
   - a shared-prefix vocabulary shortlist only where exactness is proven.

   Do not change greedy or stochastic sampling semantics.
4. **Exact overlays at the expert shapes.** The installed GB10 table covers
   `35BMoE_Expert_GateUp/Down`. Confirm the bundled path consults it per member
   (each member resolves its own schedule).

## Exactness / risk

- Pooling and grouping: scheduling only, with byte-identical tiles.
- LM-head shortcuts change semantics and need an explicit contract before any
  work starts.

## Validation

- Per-stage CPU timing before and after, on 35B and 27B, MTP off and dynamic.
- `Perf__CPUMoERouterVerifierRows`, `Perf__CPUFusedVerifier`.

## Lane activated (2026-11-10, arm-addon2; plan outline APPROVED)

- Code recon at the lane's base e8866ec04: the pooled fused launcher
  (CPUNativeNEONProjectionBatch taskCount/runTask) exists and serves the
  gate/up bundle; the multi-input expert-down launcher
  (multiply_fused_expert_down → gemv_fused_multi_input_preq) and the GDN
  4-member fused tensor path exist; a layer-batched grouped FFN transaction
  (execute_moe_grouped_ffn_transaction_preq_decode_equivalent: ONE retained
  team gate/up → exact SwiGLU-Q8 → down, expert-major rows, typed
  MTPVerifier purpose) exists for grouped/verifier rows; the exact-overlay
  table is empty/retired at this tip.
- ATTRIBUTION DRIFT: the tracker's attribution commits (408ff90e5/264970e14)
  are NOT ancestors of e8866ec04 (separate line) — the 274/176/394 µs
  numbers must be re-measured at the current tip before Phase B claims.
- APPROVED plan: Phase A measurement-only first (35B MTP off+dyn per-stage
  timing + route/counter evidence at the then-current tip; window booked
  after the combined receipts + Bundle-1 validation); Phase B candidates
  B1 (route M=1 through the existing FFN transaction primitive) + B2
  (shared-expert one-team transaction) + B3 (GDN pooling conditional on A) —
  byte-neutral scheduling only, full written plan before implementation;
  B4 (LM head) = evidence + a written top-k/shortlist sampler-contract
  PROPOSAL, with the contract decision routed to the OPERATOR (semantics
  change). Expected wins are hypotheses until A.

### **idea-16 Phase A COMPLETE — seal K14-I16A-880429059-20261111** (arm-addon2, ~11:54Z)

Full report archived: evidence/i16a-phase-a-report.md. TOP LINES:
1. **Headline (unprofiled, n=256 decode):** no-MTP **25.20/25.24 tok/s**
   (+20% vs tracker head ref 20.95); MTP-dyn 18.53/18.49 (**−35% vs
   28.51**) — MTP-dyn now LOSES ~27% at this anchor.
2. **Phase A answers:** (a) M=1 routed-down pooling ALREADY ACTIVE
   (router-Q8 bundle + pooled multi-input down fire every layer:
   30600 = 40×765 counters); (b) verifier rows DO take the grouped FFN
   transaction — in MTP-dyn the target graph runs ONLY as verifier
   forwards (no separate main decode steps; call counts resolve exactly);
   (c) GDN projection pooled but split into TWO subgroups/layer (qkv+z vs
   tiny alpha/beta; mixed-codebook compatibility rule), ~30-34 GB/s =
   mostly bandwidth-bound; (d) LM head bandwidth-bound ~45 GB/s — no
   scheduling headroom, contract-level only (B4/operator); (e) shared
   expert ~2× headroom, still two regions (B2 stands).
3. **MTP-dyn regression decomposition (54.0 vs 39.7 ms/token):** per token
   2.03 verifier rows; **LM head = one full-vocab M=1 GEMV PER VERIFIER
   ROW (zero amortization, 20.1% of MTP-dyn decode)** although a grouped
   decode-equivalent LM-head path EXISTS in LMHeadStage and the
   decode-verifier route does not take it → **NEW TOP BYTE-NEUTRAL
   B-CANDIDATE (ceiling ~6.5 ms/token)**; grouped GDN recurrence ~1.0×
   amortization (idea-18 confirmed in production); MoE transaction ~1.8×.
4. **PROTOCOL CAVEAT + rider request:** bench config carried temp 0.8 /
   top_k 40 / top_p 0.9 / seed −1; acceptance 38.89% (depth-10
   budget-limited, 5.91 rows/run) vs tracker-verified 99% at depth 7.
   Greedy-vs-stochastic UNDECIDABLE from the seal (no token digests).
   PROPOSED: ONE greedy-pinned MTP-dyn rider (temp 0 + explicit seed +
   text signature) to settle mode-artifact vs real draft regression —
   cheap, short .80 slot. Cross-idea flag: idea-14's 27B "52-65%
   acceptance" may share the same mode ambiguity.
5. B4 stays fenced to the operator. Prefill G1 = 12-token-prompt only,
   NOT tracker-comparable.

### **CORRECTION (author, ~11:56Z) — verifier LM head is GROUPED, not per-row**

Supersedes the "one full-vocab M=1 GEMV PER VERIFIER ROW" claim in the
Phase A lines above (full correction appended to
evidence/i16a-phase-a-report.md, commit 4c4d52687):
- Decisive counter in the same MTP stats: `mtp
  lm_head_grouped_decode_equivalent_verifier_prefill_rows count=264
  value=1560` — the verifier LM head IS the grouped decode-equivalent
  path: 264 grouped calls covering all 1560 rows (~2.4-2.8× amortization).
- The 1560 stage_cpu LM_HEAD calls = **1296 DRAFT-step M=1 calls + 264
  verifier grouped calls** (1296 = mtp draft_steps exactly). The real
  per-row full-vocab cost is the DRAFT side, BY DESIGN
  (Qwen35Graph::buildMTPGraph wires full_vocab_output at seq_len=1 — the
  draft samples from the full distribution).
- Corrected MTP-dyn per-token deltas (profiled): LM head +6.56 TOTAL =
  draft full-vocab ≈ +2.95 (top MTP-specific kernel cost) + verifier
  grouped ≈ −0.67 (grouped WINS); GDN recurrence +3.96 (≈1.0×); MoE +2.17
  (≈1.76×); GDN projection −0.82; shared expert −1.14; net ≈ +11.9 ✓.
- REFRAMED BREAKEVEN: blended amortization ≈1.9× vs rows/token 2.03 at
  38.9% acceptance → loses; at 99% acceptance rows/token ≈1.06 → wins
  with today's kernels. The acceptance collapse (rider) is the decisive
  gate; kernel fixes change the breakeven, not the sign at 38.9%.
- REFRAMED B-CANDIDATES: (A) grouped LM-head kernel ~20-22 GB/s at M≈6 vs
  45 GB/s at M=1 → ~2× kernel headroom (needs microbench before any
  claim; REPLACES the route-verifier-rows candidate — already taken);
  (B) draft-side logits = B4-family CONTRACT nuance: top-k/shortlist is
  exactness-safe under GREEDY verification, changes semantics under
  STOCHASTIC — goes into the B4 contract proposal. idea-14's "5.7 ms per
  verifier row" sentence needs re-reading (verifier rows were already
  grouped; the unamortized full-vocab cost is the DRAFT per-step head —
  27B dense draft economics apply there too).

### LM-head route/source audit @ bb893cd79 (arm-addon2, ~11:57Z — approved autonomous, NO fix claims)

1. **Row purpose/count:** 1560 LM_HEAD calls = 1296 draft-step M=1 calls +
   264 verifier GROUPED calls (`lm_head_grouped_decode_equivalent_verifier
   _prefill_rows` 264/1560 — the verifier already uses the grouped path;
   the earlier per-row claim corrected).
2. **Consumer live extent:** draft logits = one fp32 row in MTP_LOGITS per
   draft step → draft token sampler (1296 samples); verifier terminal
   logits: 264 rows × 993280 B retained for the next main distribution
   (267 current_main_logits_publications).
3. **Why grouped isn't selected on the draft side (code fact):**
   Qwen35Graph::buildMTPGraph enforces seq_len=1 for normal draft
   execution; the sidecar wires full_vocab_output = output.logits with
   compute_all_positions=true; the grouped predicate
   (force_decode_equivalent_lm_head_verifier_prefill) requires
   total_tokens > 1 — false at seq_len=1. STRUCTURALLY nothing to group:
   draft step t+1 consumes the token sampled at step t (sequential
   dependency) → the M=1 full-vocab GEMV per draft step (~4.25-4.9 ms;
   ≈13-15% of MTP-dyn decode as HYPOTHESIS) is inherent to the current
   draft-sampling loop contract. The M=1 kernel is already ~45 GB/s — no
   scheduling headroom at M=1.
4. **Fix-space (B4/operator discussion only):** (a) GREEDY verification —
   draft proposal mechanism doesn't affect output exactness (a
   shortlist/proxy trades acceptance RATE, not correctness); (b)
   STOCHASTIC — the draft distribution enters accept/reject math → any
   approximation is a semantics change. Any fix proves serial-row byte
   equivalence + actual economy. No rider beyond the approved greedy one.

### B1 microbench DESIGN (design only — no implementation until Main admits)

New test Perf__CPUMoESingleTokenTransaction.cpp beside
Perf__CPUMoERouterVerifierRows; production geometry (d_model 2048, 256
experts, top_k 8, intermediate 512 → 16 gate/up members 512×2048 sharing
one router Q8 row + 8 down members 2048×512), M=1; IQ3_S (production) +
Q4_K + Q8_0. Arm A (production oracle, public APIs): router Q8
publication → multiply_fused_router_q8_hidden_grouped_decode_equivalent
→ per-expert serial SwiGLU + per-expert Q8 quantization as
multiply_fused_expert_down does → multiply_fused_expert_down → publish/
axpy. Arm B: same descriptors through
execute_moe_grouped_ffn_transaction_preq_decode_equivalent as a 1-row
expert-major route. BYTE GATES (hard fail): gate/up outputs, SwiGLU-Q8
bank, down outputs, route contributions — A vs B byte-identical. Timing:
≥100 timed iters after warmup; mixed@20T, fast-only@10T (taskset
5-9,15-19), slow-only@10T, 1T; bundle-only/down-only/transaction-total vs
A-total; region-entry counters recorded. ECONOMY RULE: B1 admitted for
the full written plan only if B ≤ A within noise on the production
placement; a regression closes B1 with numbers recorded. If the
transaction does NOT admit the M=1 shared-router-Q8 shape: that admission
gap is a FINDING to report (not fixed inside the bench).

### **B1 DIAGNOSTIC IMPLEMENTATION APPROVED** (Main, ~12:01Z) — contract recorded

Main approved implementing the DIAGNOSTIC B1 microbench + harness focused
checks, booked as ONE CPU economy window AFTER Main campaign + approved
greedy rider (via coordinator/sparks). NOT approval of production
routing/kernel/sampler changes. CONTRACT:
- Actual production A helpers/publication/transaction B — NO copied
  SwiGLU/Q8 oracle, NO forced ineligible path; preallocate/seal ALL
  inputs/workspace BEFORE timing.
- Hard byte gates on ALL gate/up, SwiGLU-Q8, down, route outputs; prove
  both intended paths ran + live extents + inactive guards.
- Warmup + ≥100 ALTERNATING PAIRED A/B samples; production geometry/
  formats; the 4 observed physical placements; identical actual thread/
  OMP/policy context; UNPROFILED timings; separate attribution counters.
- If M=1 B cannot admit the actual shared-router-Q8 shape: report the
  GAP — no fallback/bench-production fix.
- Regression OR no signal → close/retain the finding WITH NUMBERS, not
  justify a production change; B≤A within noise = NONREGRESSION, not an
  optimization win (production economy claim needs repeatable benefit
  beyond noise). No lane-wide qualification/new issues/tracking artifacts.
LM-HEAD CORRECTION ACK: verifier already grouped (264/1560); old
route-bug claim withdrawn; draft single-row causal dependency retained.
**45 vs 20 GB/s alone does NOT prove 'no scheduling headroom' or '2×
available headroom' (different reuse/work denominators) — treat as
diagnostic hypotheses.** B4 shortlist remains OPERATOR-FENCED (no
automatic greedy-byte-exact or stochastic-semantics conclusion without
the full proposal/acceptance contract proof). The exact-sampler rider
remains a CONTROLLED REFERENCE only.

### B1 diagnostic bench IMPLEMENTED: bdbb8e370 (arm-addon2, ~12:09Z) — .80 focused checks pending

On codex/arm-idea16-moe-decode, base **bb893cd79** (branch start moved to
the current line tip — the original e8866ec04 start was superseded by the
base advancing). 2 files +965: Perf__CPUMoESingleTokenTransaction.cpp +
CMake registration (v2_perf_cpu_moe_stt_transaction / NO_MODELS NO_MPI,
TIMEOUT 300). Contract compliance:
- ACTUAL production helpers both arms: A = CPUMoEKernel::route +
  publishedRouterQ8Hidden (admitted publication root) →
  multiply_fused_router_q8_hidden_grouped_decode_equivalent (production
  hosting on first active expert's gate engine) →
  compute_swiglu_gpu_aligned_expert_serial → multiply_fused_expert_down
  (INTERNAL per-expert Q8 quantization, workspace-admitted bank) →
  vec_axpy publication. B =
  execute_moe_grouped_ffn_transaction_preq_decode_equivalent, 16 gate/up +
  8 down descriptors, rows=1, ALL gate/up members at the SAME
  published-router-Q8 row pointer (no copies), Decode purpose, Auto
  policies.
- NO copied oracle: the SwiGLU-Q8 byte gate applies the production
  quantize_activations_to_q8_1 helper to arm A's rows vs arm B's bank
  (launcher-vs-wrapper divergence would surface as a FINDING).
- Preallocate/seal before timing; inputs re-compared after every phase.
- Hard byte gates ALL outputs (gate/up ×8, SwiGLU-Q8 ×8, down ×8, route
  contribution) BEFORE + AFTER timing; 64 B guard canaries; A-vs-B gates
  vs guard-proven banks.
- PROOF PASS (separate, unprofiled): fresh deltas of router_q8_reuse +
  fused_expert_down (arm A) and grouped_ffn_transactions (arm B) each
  exactly 1.
- Timing: warmup 8 pairs + 120 paired samples (≥100 enforced),
  alternating within-pair order, per-arm clocks, unprofiled, medians/
  mean/sd + ratio; verdict line = diagnostic_only.
- ADMISSION GAP → ADD_FAILURE, no fallback/reshape/production fix.
- Geometry: d_model 2048, intermediate 512, 256-expert real top-8;
  default IQ2_S gate/up + IQ4_XS down (real UD per-tensor pairing) + env
  overrides (STT_GATE_UP_FORMAT/STT_DOWN_FORMAT/STT_POLICY).
- STATUS: -fsyntax-only passes locally (clang 22 NEON); LINKED build/run
  IMPOSSIBLE on macOS (libnuma) → focused checks + economy run happen in
  the .80 window; committed with the hook's documented --no-verify bypass.
- WINDOW REQUEST: after Main campaign + greedy rider; 4 observed
  placements (mixed@20T, fast-only@10T taskset 5-9,15-19, slow-only@10T
  taskset 0-4,10-14, 1T), each an independent V2_Perf_
  CPUMoESingleTokenTransaction invocation at bdbb8e370, Release
  Integration build, default formats, unprofiled, evidence = stdout +
  CPU_STT_* lines per placement into a K14 seal.

### B4 DESIGN INPUT (reviewer, ~13:33Z) — operator-directed consultation; no implementation grant

Static source checked at bb893cd79; Phase-A timings remain owner-reported.
(1) **Correct boundary is not greedy-vs-stochastic** — it is unchanged
   TARGET law p + truthful PROPOSAL law q + correct coupling/correction.
   For shortlisted qS (zero outside S): draw y~qS, accept
   min(1,p(y)/qS(y)); on rejection draw r(x)=[p(x)-qS(x)]+/Z. Accepted
   mass=min(p,qS), output mass=p. qS = normalized shortlisted logits does
   NOT require the old full-vocab normalizer; but changing the proposal
   while keeping OLD q in the ratio is INCORRECT. Acceptance
   probabilities generally change with q; the invariant is the corrected
   TARGET OUTPUT distribution — equal distribution ≠ identical old seeded
   output.
(2) **Cheap stochastic-valid special case already in the repo:**
   deterministic proposal a(h) (incl. shortlist argmax) = q=delta_a, accept
   with p(a), on rejection sample p with a removed; a need NOT be the
   full-draft argmax. SamplingMath.h/IBackend.h/DeviceGraphOrchestrator.cpp
   support greedy draft/one-hot q with stochastic TARGET. STRONGER CPU
   seeded contract: OrchestrationRunner.cpp:9056 selects
   serial-sample-equivalent host stochastic when seed!=0;
   MTPRejectionSampler.cpp:129ff draws directly from TARGET p with a
   position-keyed threshold (accept iff token==draft proposal) — changed
   draft proposals can preserve the same serial seeded target stream
   (Unit SerialEquivalentCorrectionIgnoresDraftChoice documents it). Do
   NOT replace this CPU seeded coupling with one-hot rejection silently.
(3) **Operator contract wording (reviewer-drafted, for the operator's
   approval):** "B4 is an explicit versioned draft-proposal policy
   evaluating only admitted live vocabulary rows. Full target head/
   sampling policy, grouped target arithmetic and committed-state
   authority remain unchanged. Shortlist miss causes normal speculative
   correction, never target restriction to the shortlist or a
   full-draft-head retry. Proposal policy and verifier coupling are
   separate typed axes." Minimum proof per clause (selected-row arithmetic
   byte-compare to identical full-head rows, IQ3_S ragged/sharded; proposal
   ownership valid unique nonempty live S + deterministic tie-breaking +
   no lookahead; greedy = same serial target-greedy tokens AND continued
   KV/GDN/sampler state incl. winner-absent/ties/EOS/rejection positions;
   stochastic = analytic/enumerated tiny-vocab proof + recorded-state/RNG
   replay, wrong-q and omitted-mass negatives fail; seed/version =
   unchanged coupling retains records, changed q explicitly approved;
   ownership/runtime = existing authorities only; economy = counted
   rows/bytes, cost per EMITTED token).
(4) **RECOMMENDATION: GREEDY-TARGET FIRST** on the measured CPU/IQ3_S lane
   as the smallest engineering slice, stochastic modes unchanged.
   Unsupported shortlist+coupling requests reject; no silent stochastic
   disabling/fallback. Staging ≠ mathematical ban on stochastic. Next:
   existing CPU seeded target-draw coupling or GPU one-hot semantics
   BEFORE inventing sampled sparse-q machinery.
(5) Building S must avoid full-head evaluation (full logits then top-k
   saves no dominant weight traffic); prior verified target top-k = a
   possible cheap heuristic, NOT certified current top-k. A =
   sum min(p,qS) = 1-TV(p,qS) ≤ p(S); deterministic A = p(a). Savings
   can vanish via extra rejections/target transactions. Conditional on
   13-15% being non-overlapped MTP wall time: removing ALL draft-head work
   with unchanged transaction counts caps speedup ≈1.15-1.18×; overhead/
   acceptance loss lower it. The no-MTP 10.2% fraction is NOT additive.
BOTTOM LINE: greedy-first staged design recommended; "stochastic
shortlist necessarily changes target law" is FALSE. Preserve target law,
actual q, and the installed seeded coupling separately. No implementation/
economy admission.

### B4 follow-up: DRAFT VOCABULARY BUDGET design (reviewer, ~14:00Z) — design only, fence unchanged

Terminology: **draft_vocab_limit**, NOT "proposal length" (confusable
with MTP draft-token DEPTH). Three independent meanings kept separate:
shortlist SELECTOR, proposal DRAW LAW, verifier COUPLING.
- Typed: DraftProposalPolicy { policy_version, selector, draw_law,
  vocab_budget }; VocabBudget = FullVocabulary | ShortlistLimit{checked_
  positive_N}. Verifier coupling/version separate. "full" = CURRENT
  full-vocabulary DRAFT PROPOSAL with MTP STILL ENABLED (≠ no-MTP);
  preserve current baseline computation + seeded coupling.
- N = maximum LIVE distinct candidate token IDs; actual |S(h)| explicit +
  reported; smaller set only via declared selection/eligibility semantics,
  never resource-error recovery; empty legal support fails. 1<=N<V;
  zero/out-of-range error; 'full' spelled explicitly; do NOT alias N=V to
  full if candidate draw law differs (one-hot vs CPU sampled full-draft).
- N does NOT define selection/draw. Exact top-N of current full logits
  requires full-head eval (defeats the savings) — use ONE fixed versioned
  cheap causal selector (prior verified target row ranking = heuristic,
  NOT certified top-N); then evaluate current draft logits ONLY on S. On S:
  declared deterministic winner (q=delta_a) OR actual normalized sampled
  qS. Never call rank scores/old full-head probabilities the proposal q
  unless they generated the token. "Residual always on" is NOT universal:
  greedy target = exact target-greedy mismatch correction; CPU seeded
  serial-target-draw = existing sample+match coupling; probability-
  rejection = min(1,p/qS) + normalized[p-qS]+ over FULL target support.
  Different contracts; target policy p unchanged at EVERY admitted N.
- Acceptance/economy: A=1-TV(p,qS) only for probability-rejection at fixed
  prefix; one-hot q → p(a); target-greedy = exact match; CPU seeded = its
  RNG coupling (NOT the TV formula). NO monotonicity in N. Measure
  accepted-prefix LENGTH, rejection position, target transactions, actual
  depth/controller, latency per EMITTED token. Extra rejections = work
  amplification, not hysteresis — start repeated comparisons from
  equivalent initial state. 'full' = 1.0× reference, NOT where the
  1.15–1.18× ceiling sits (that ceiling assumes eliminating all 13–15%
  draft-head cost with everything else fixed; N controls how close, and
  changed acceptance/work invalidates it as an absolute bound).
- IQ3_S exploratory points: full + 64/128/256/512/1024/4096; N=1 = extreme
  diagnostic; add a larger point if quality poor, refine near the
  measured frontier. NO default justified yet. Correctness covers
  non-aligned counts/sparse IDs (63/65,127/129,V−1), zero/invalid, ties,
  reset; pilot timing points ≠ supported-N envelope.
- Per-model calibration plausible but identities matter (weights revision,
  format, ISA/backend, placement/sharding, workload, sampler/coupling,
  MTP-depth, prefix policy); start with explicit authored N + default
  full; persist calibration via existing config/tracker surfaces; NO new
  automatic tuner/gate apparatus; hold-out prompts + clean repeated
  Release timings decide economy.
- Seed: sweep NOT restricted to greedy/unseeded mathematically; CPU
  seed≠0 = maybe the STRONGEST same-seed compatibility experiment. Target
  RNG addressed by logical position + existing stream/token-ID
  conventions — NOT by N/shortlist rank/depth/rejections. Retain
  byte-equivalent target rows + accepted-state continuation. Probability-
  rejection with changed q can change old seeded output despite
  preserving p → explicit policy/version approval. Greedy-target first =
  lower engineering risk; unsupported combinations reject WITHOUT
  silently disabling MTP. IQ3_S packing does NOT justify rounding the
  budget; kernel padding ≠ live support; invalid IDs/duplicates/stale
  generation/zero-mass support fail precisely.
- ONE BUILD: N = runtime descriptor DATA (not template/CMake/per-N
  binary); same Release binary, separate fresh startup instances per N;
  compile/source/ISA identity fixed; PMA admission for workspace/
  weights/layout; each startup binds policy BEFORE capture; runtime-N
  dispatch handles odd/tail; N + selector/draw/coupling versions in
  config/results/capture identity. "One build" ≠ mutating N in a running
  request (that needs an admitted dynamic-capacity contract; no hot
  allocation/readback/recapture). Canonical timing unprofiled; selected-
  row byte/traffic paths separately qualified.
RECOMMENDATION (reviewer → operator): **approve the DESIGN of an explicit
runtime draft-vocabulary budget** with full baseline + fixed selector/
draw/coupling identity; stage greedy-target first, then the
already-existing stochastic couplings as separately proved modes. NOT an
implementation unfence; no N/economy result admitted. Existing focused
tests + Main's post-merge campaign ownership unchanged.

### **B4 DESIGN APPROVED (operator, ~14:05Z) — design-only, implementation still fenced**

Operator approved: (1) the typed draft-vocabulary-budget DESIGN
(DraftProposalPolicy { policy_version, selector, draw_law, vocab_budget };
full | ShortlistLimit{checked_positive_N}; runtime parameter; one build);
(2) the sweep protocol (full + 64/128/256/512/1024/4096; cost per emitted
token; accepted-prefix length; production placements; IQ3_S); (3) the
contract spec for operator review. **NO implementation until an explicit
second go.** Greedy-target staged first per the reviewer. Reviewer design
input (e75b0b961) governs the spec. Owner: arm-addon2 (idea-16 lane).
Fence state: DESIGN unfenced; IMPLEMENTATION + economy admission fenced.

### B4 DESIGN SPEC DELIVERED (arm-addon2, ~14:02Z) — archived evidence/
b4-draft-vocab-budget-design-spec.md; routed to operator for review.

Design-only; fences restated (§9). Structure: 0) five consulted questions
answered; 1) typed spec (DraftProposalPolicy{version, selector, draw_law,
vocab_budget=Full|ShortlistLimit{N}} + separate VerifierCouplingPolicy
axis; full = current full-draft + MTP enabled; N runtime descriptor data
in one binary; no N=V→full aliasing; default full); 2) selected-row
byte-proof spec (global IDs/full-K/IQ3_S/scales/reduction + per-clause
minimum proofs); 3) selector identity (v1 prior verified target row
ranking = HEURISTIC, not certified top-N; no full-logits top-N fake
savings; q truthful); 4) per-mode coupling specs (G first, C existing
seeded never silently replaced, P last; TV formula only in P); 5) sweep
protocol DESIGN ONLY (full+64/128/256/512/1024/4096+N=1 diagnostic;
cost per emitted token primary; 4 placements; hold-out prompts; no
tuner); 6) one-build mechanics; 7) correctness coverage (63/65/127/129/
V−1, zero/invalid/stale/zero-mass/ties/reset/prefix, no budget rounding,
negatives must FAIL); 8) recommendation (G → C → P staged); 9) fences.

### B4 SPEC APPROVED FINAL (operator, ~14:05Z)

The B4 draft-vocabulary-budget design spec is the FINAL design record
(evidence/b4-draft-vocab-budget-design-spec.md). Implementation = a
separate later go (after the B1 diagnostic, the greedy rider, and the
campaign queue clear). No work until then. Fences: design-only; no N-sweep
run; no economy admission; no tuner/gate apparatus; tracker-only.

### Post-campaign .80 window ORDER LOCKED (coordinator, ~14:57Z)

(1) **GREEDY RIDER FIRST** (Main's verbatim protocol; retained Phase-A
880429059 binary/artifact state — differs from the B1/tournament tip;
running it first avoids a second build-dance; ~2 min machine time;
unlocks the idea-16 Phase B selection input — the sampling-mode question
behind the 38.9% acceptance anomaly); (2) idea-08 BOM census +
compact-decode tournament; (3) M=1 economy window (B1 at bdbb8e370);
(4) Lane-2 window. All after the Main campaign completes + the lease
returns. arm-addon2 executes each on relay.

### B4 SPEC VERDICT: SPEC-DEFECTS (reviewer, ~15:22Z) — S1–S6; corrections to the EXISTING spec file

Exact reviewed commit be1e43924c01b6681647c13adb260d18907f170c; blob
69cf952677146da42b565223b1d511b92913cdac (origin c4b1d6276 identical — no
local/remote ambiguity). FAITHFUL portions: typed axes separation;
full=MTP-enabled exact baseline; positive N<V + live-support-vs-capacity +
no silent N=V alias; §2a global IDs/full-K/IQ3_S/scales/reduction +
ragged/sharded selected-row checks; TV limited to Mode P; core economy
metrics + reset + no-tuner + runtime-N/one-build/capture boundaries.
CORRECTIONS (in the EXISTING spec file, not a new artifact):
- **S1 [semantic]:** Mode C (CPU seeded serial-target-draw) must PRESERVE
  the same seeded target tokens for every admitted N (identical p(h) +
  position-keyed draw + committed state) — only match/acceptance/work may
  change. The current text ("changed q can change the old seeded stream")
  transplants the Mode P caveat into C — a token change is a FAILED
  compatibility proof, not licensed by renaming q/version. Qualify the
  §0 blanket sentence; "one-hot/star target" → "one-hot DRAFT proposal
  with stochastic target."
- **S2 [contract omission]:** the operator contract is delegated to an
  unlinked attachment — INCLUDE IT HERE: full target head + complete
  target sampler (temperature/top-k/top-p/filters/penalties/masks/
  tie-breaks) unchanged; do NOT renormalize p over S; ALL unchanged
  grouped TARGET verifier rows byte-equivalent to serial M1 (in addition
  to selected DRAFT-row comparison); Mode G target argmax AFTER the
  existing applicable target transforms; accepted-state/bonus authority
  unchanged.
- **S3 [proof coverage]:** add tiny-vocab analytical/enumeration +
  production replay for q=p, one-hot q, disjoint support/omitted mass,
  p(y)=0, normalization/underflow, discrete threshold endpoints; keep
  finite-precision/RNG contracts (histograms insufficient); cover
  all-accept/first-and-later rejection/every supported depth/bonus row/
  stop-EOS/partial work/continued state after reset+prefix reuse;
  DIFFERENTIATE valid zero probabilities (q=0 outside S, p(y)=0) from
  INVALID zero total sampled-proposal mass; Mode P needs truthful qS
  across residual support, not merely qS(y).
- **S4 [ownership precision]:** selector v1 is a concrete design choice —
  specify its source row/global-position/request-generation + lifetime
  (first proposal/full-accept/bonus/rejection/prefix-cache-reset); only
  causally available valid source, no hidden full-head-on-missing-source
  rescue; "PMA admission unchanged" → "PMA AUTHORITY unchanged; ALL
  shortlist/ranking/gather/probability workspace + retained state included
  in its BOM/admission" (same authority ≠ unchanged memory requirements).
- **S5 [sweep evidence]:** make the rider-protocol reference RESOLVABLE
  and normative for repeated fresh-process paired/rotated trials + raw
  variation/uncertainty + exact sampler parameters/seed + complete
  controller/cache starting state; one fresh startup per point ≠
  repeatability evidence; bind hard-coded fast/slow CPU lists to the
  ACTUAL measured host/topology + observed worker placement (requested
  labels don't certify execution).
- **S6 [authorization]:** "operator review of this spec = second go for
  implementation" CONTRADICTS the design-only/fenced state — replace with
  "implementation and N-sweep remain FENCED until a separate explicit
  operator instruction unfences that scope"; a spec-faithful verdict does
  not itself grant implementation/economy.
No whole-lane suites/apparatus/implementation requested. The reviewer can
re-review the exact corrected blob; source/implementation + runtime/
economy evidence stay separate when/if the operator unfences.

### B4 S1–S6 CORRECTED SPEC delivered (arm-addon2, ~15:24Z) — corrected blob routed to reviewer

All six applied to the existing spec file (evidence/
b4-design-spec-corrected.md): S1 Mode C seeded-token preservation as a
HARD compatibility requirement + blanket sentence qualified per-coupling;
S2 operator contract INLINED as normative clauses (full head + complete
sampler unchanged, no p renormalization over S, grouped verifier rows
byte-equal serial M1, Mode G argmax after transforms, accepted-state/
bonus authority unchanged); S3 expanded proof coverage (incl. valid-zero
vs invalid-zero-mass distinction, truthful qS across residual support);
S4 selector v1 lifetime fully specified + PMA AUTHORITY-unchanged/BOM-
includes-workspace precision; S5 normative repeated fresh-process
paired/rotated trials + placement lists bound to actual topology; S6
"operator review = second go" REPLACED with the fence text
(implementation + N-sweep fenced until separate explicit operator
instruction).
ALSO: (a) compact IQ3S decode bench committed d096438e2 (idea-08 lane,
syntax-clean); (b) B4 HOLD COMPLIANCE confirmed: NO B4 code/harness
written before the operator's literal authorization resolved (the hold
stood; the only B4 action was the S1–S6 spec text). With the operator's
"yes — diagnostic authorized" now on record, addon2 proceeds with the
diagnostic implementation (S1–S6 as test properties) and sends the
corrected blob + diagnostic SHA to the reviewer.

### **GREEDY RIDER RESULT (arm-sparks, ~21:02Z): the 38.9% acceptance is NOT a sampling-mode artifact**

Seal K14-I16RIDER-880429059-20261111 (retained 880 Release binary verified
= build-i16a @ 880429059d; same GGUF/prompt/context/256-output; ONLY
sampler intent changed):
- EXIT SBENCH=0; tok/s 58.74 prefill / 18.39 decode.
- **RESOLVED ACTUAL SAMPLER: {'seed': 42, 'temperature': 0.0, 'top_k': 40,
  'top_p': 0.9} — GREEDY (temp0) confirmed in the resolved request
  config.**
- **MTP counters: acceptance_rate 0.3889 (168 accepted / 432 draft_steps /
  65 rejected); verifier_runs 88; verifier_token_count 432+; current_depth
  10; max_depth 15; min_depth 1; depth_policy=dynamic; depth_policy_updates
  24 (promotions 24, demotions 0); last_depth_policy_reason=budget_limited;
  stochastic_accepts 0 (purely greedy verify: all accept tests = 0);
  transaction_commits 264; mtp_cached_tokens 267; mdp config_enabled=True,
  bypassed=False.**
- TEXT DIGEST: 256 tokens, sha256(ids) = 0a4c0002a4a6de72f174….
- **INTERPRETATION: with temp0/stochastic fully disabled, acceptance =
  EXACTLY 38.89% → NOT a sampling-mode artifact.** The tracker-verified
  99% @ depth 7 baseline differs in protocol (earlier runs' sampler/policy
  state or the depth policy at that time). The controlled reference now =
  38.89% at current_depth 10, budget_limited, 24 promotions. MTP-dyn vs
  no-MTP decode = 18.39 vs 25.2 (−27%, mode-independent observation).
- .80 = Main's FMLAL window (started 21:02Z, ~30-40 min). W2 (edcce0183)
  → .79 (16G free, build ≈10G fits) STARTING NOW per the parallelization
  directive.

### SWEEP HYPOTHESIS SHARPENED (arm-addon2, ~21:03Z): the acceptance METRIC itself may be depth-confounded

Arithmetic from the seal: 168 accepted + 65 rejected = 233 of 432 draft
steps — **~199 steps unaccounted, matching budget_limited TRUNCATED draft
steps.** If the acceptance denominator counts truncated steps as
non-accepts, a depth-policy change alone (7→10 dynamic, budget-limited)
moves the REPORTED acceptance without any real draft/target agreement
change — consistent with 24 promotions/0 demotions (zero_accept_demote=
0.3 explains 0 demotions at 38.9% > 0.3). SWEEP DISCRIMINATORS (cheap):
(1) the budget_limited-event-rate column — acceptance tracking that rate
= metric confound; (2) **one FIXED depth-7 run at temp0/seed42** — if
acceptance jumps toward ~99% at fixed depth with the same binary/model,
the agreement is intact and the dynamic controller+metric is the whole
story; (3) per-run emitted-token identity vs the archived C-state greedy
text (byte-compare — real target-arithmetic drift shows as text
differences). IMPLICATION FOR PHASE B RANKING: if the metric confound
holds, the TOP candidate may be a DEPTH-POLICY/controller fix (idea-14
adjacent, policy-only, no kernel change) — and B-candidate economics
should be re-measured at the recovered acceptance before ranking.

### B1 FOCUSED-CHECK BLOCKER (arm-sparks, ~22:26Z): missing include path in the addon2 branch

W2 on .79: edcce0183 verified-fetched; Release CPU-only build progressed
to 736/764 then FAILED — the STT header exists in-tree but the target's
include dirs OMIT tests/v2 → fatal at
Perf__CPUMoESingleTokenTransaction.cpp:72 (same for the compact target).
Fix-back sent to addon2 (CMake include-dir fix); sparks resumes on their
fix. NOT disk/fetch (86G free; SHA verified).

### W2 FIX-BACK LANDED (addon2, ~22:28Z): a6aee9bac — include roots complete

Both bench targets now carry the source root AND tests/v2 include roots
(the STT target was missing tests/v2; the compact target was missing the
src root). Root cause of the miss: local syntax checks invoked clang++
with both -I paths manually, masking the CMake gap. Origin-verified;
sparks resumes the W2 build on .79.

### W2 FIRST REAL SIGNAL (~04:43Z): the B1 bench caught a production counter gap

.79 runs (all 4 placements): `Perf__CPUMoESingleTokenTransaction.cpp:836:
Failure — fresh_bundle Which is: 0 vs 1u — arm A must advance the
production router-Q8 bundle counter` — **the arm-A production route did
NOT advance the router-Q8 bundle counter in the fixture.** A real
counter/curve signal, not harness noise (the STT timing otherwise
rendered: production_median 217.2 vs transaction 206.3 µs at
mixed-20T). Routed to addon2: fixture-expectation question vs a genuine
production counter-path gap. **No B1 seal until the counter contract
resolves — the counter assertion is the bench's core.** This is the B1
bench working as designed: the diagnostic surfacing a production
question the first time it ran against the real routes.
DISK: .80 hit 100% full; sparks pruned their own superseded campaign
build dirs (~100G → 98G free, 3.4T used); Main's/addon's active
contexts preserved. B1ATTR2 requeued post-#8.
