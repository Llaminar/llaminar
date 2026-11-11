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
