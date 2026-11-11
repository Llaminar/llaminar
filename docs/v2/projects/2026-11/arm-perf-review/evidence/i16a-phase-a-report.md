## Phase A re-measured attribution @ `880429059` (GB10 .80, seal `K14-I16A-880429059-20261111`)

Measurement-only (idea-16 Phase A; approved plan). Release NEON_I8MM SHARED_CORE build at
`880429059`; 20 cores, bench mode, 256 generated tokens ×3 iterations, warmup; stage timing
via `LLAMINAR_PERF_STATS_CPU_STAGE_TIMING=1` + `LLAMINAR_PERF_STATS_JSON` (all domains).
Stage-table numbers carry ~6% profiling overhead vs the unprofiled G1 runs.

**Protocol caveat (must be settled before headline comparisons):** the bench plan config
carried `temperature=0.8, top_k=40, top_p=0.9, seed=-1`. The MTP-dynamic run shows
**38.89% acceptance, depth=10 [1,15] budget-limited, 264 verifier runs, 1560 verifier rows**
(5.91 rows/run) — against the tracker's verified 99% acceptance at depth 7. Two candidate
causes: (a) the runs were **not greedy** (stochastic sampling active → low acceptance is
expected and the no-MTP-vs-MTP comparison changes meaning), or (b) greedy held and the
draft/target agreement genuinely regressed on this line. The stats exports contain no token
digests, so this is **not decidable from the seal**. Proposed: one greedy-pinned rider
(MTP-dyn, explicit temp 0 + seed, text signature captured) — cheap, settles it. Until then
treat the G1 headline comparison as mode-unverified. (Cross-idea flag: idea-14's 27B
"52-65% acceptance" may have the same mode ambiguity.)

### Headline (G1, unprofiled, n=256 decode; prefill = 12-token prompt only — NOT tracker-comparable)

| Run | decode tok/s | notes |
|---|---|---|
| 35B no-MTP ×2 | 25.20 / 25.24 | vs tracker head ref 20.95 (+20%) |
| 35B MTP-dyn ×2 | 18.53 / 18.49 | vs tracker head ref 28.51 (−35%); 38.89% acceptance |

### no-MTP decode attribution (per decode step ≈ 42.1 ms profiled; 40 layers = 30 GDN + 10 full-attn, 256 experts top-8)

| Stage | calls/step | per call | per step | share |
|---|---|---|---|---|
| MOE_EXPERT_FFN (routed experts) | 40 | 279.5 µs | 11.18 ms | 26.7% |
| GDN_PROJECTION | 30 | 324.9 µs | 9.75 ms | 23.2% |
| GEMM (per-layer output projection: GDN out-proj + attn o_proj) | 40 | 109.7 µs | 4.39 ms | 10.5% |
| LM_HEAD (248320×2048, M=1) | 1 | 4284.7 µs | 4.28 ms | 10.2% |
| GDN_RECURRENCE | 30 | 121.4 µs | 3.64 ms | 8.7% |
| MOE_SHARED_EXPERT_FFN | 40 | 87.7 µs | 3.51 ms | 8.4% |
| GEMM_FUSED_QKV (full-attn layers) | 10 | 217.2 µs | 2.17 ms | 5.2% |
| MOE_ROUTER | 40 | 30.7 µs | 1.23 ms | 2.9% |
| everything else (attention, norms, rope, kv, conv1d, …) | — | — | ≈1.9 ms | ≈4.5% |

### Phase A answers

1. **M=1 routed-expert down pooling: ALREADY ACTIVE.** `cpu_moe_decode_router_q8_reuse_calls`
   = 30600 (40×765) and `cpu_native_quantised_fused_expert_down_calls` = 30600 — the pooled
   router-Q8 bundle AND the pooled multi-input expert-down launcher fire every layer of every
   M=1 decode step. Idea-16 item 1's M=1 half is implemented at this tip (the tracker idea
   text predates it). Remaining inside MOE_EXPERT_FFN: bundle ≈176 µs + down/swiglu/publish
   ≈100 µs per call.
2. **Verifier rows use the grouped FFN transaction: CONFIRMED.** In MTP-dyn, the target graph
   runs **only as verifier forwards** (264 runs; no separate main decode steps — stage-call
   counts resolve exactly to 40×264 + 1296 draft-layer calls). All 10560 MoE layer events took
   `cpu_native_quantised_moe_grouped_ffn_transactions` (expert-major rows, weights stream once
   per expert); draft steps (1296; the 1-layer draft head) use the M=1 bundle+down path.
3. **GDN projection: pooled but split into TWO subgroups per layer.** Route counter
   `gdn_projection_route` = `same_kernel_mixed_codebook_subgroup` with members
   (qkv n=8192 + z n=4096) and (alpha n=32 + beta n=32), K=2048 — the a/b members cannot join
   the qkv/z subgroup under the current compatibility rule (mixed codebook/kernel), so each
   GDN layer pays one extra tiny region (≈64×2048 weights). Verifier rows take the same split
   (`grouped_decode_equivalent_verifier_mixed` + `_subgroup`, m=2..11).
   Bandwidth: qkv+z ≈ 25.2M params ≈ 9.6 MB (IQ3_S) per layer → at ~325 µs incl. the second
   region, effective ≈ 30–34 GB/s vs the bundle's ~36 GB/s and LM head's ~45 GB/s — mostly
   bandwidth-bound; residual scheduling headroom ≈ the second region.
4. **LM head: bandwidth-bound, zero scheduling headroom.** 4284.7 µs at M=1 for
   248320×2048 ≈ 194 MB (IQ3_S) ≈ 45 GB/s effective — at the pooled-kernel ceiling. The only
   meaningful wins are contract-level (B4: top-k-only verifier heads / shortlist — operator
   decision; no semantics change without it).
5. **Shared expert: still region-overhead-dominated.** 87.7 µs/call at M=1 for ≈1.2–2 MB
   weights ≈ 14–23 GB/s effective (vs 36–45 GB/s elsewhere) — ≈2× headroom; gate/up fused
   region + swiglu/down region remain two regions (B2 candidate).

### MTP-dynamic regression decomposition (per output token, 54.0 ms vs 39.7 unprofiled no-MTP)

Per token: 2.03 verifier rows (1560 rows / 768 tokens) + 1.69 draft steps. Amortization per
row vs an M=1 stage call: MoE transaction ≈ 150–160 µs/row vs 279.5 (≈1.8×), GDN projection
145.9 vs 324.9 (2.2×), **GDN recurrence 124.5 vs 121.4 (≈1.0× — grouped recurrence buys
nothing; idea-18's kernel finding confirmed in production)**, **LM head 5336.5 µs per row
with zero amortization — the verifier computes one full-vocab M=1 GEMV PER ROW (1560 calls,
8.33 s = 20.1% of MTP-dyn decode), although a grouped decode-equivalent LM-head path exists
in `LMHeadStage` and is not taken by the decode-verifier route**.

Stage-total deltas per token (MTP-dyn − no-MTP, profiled): LM head +6.5 ms, GDN recurrence
+4.0 ms, MoE +2.1 ms, router +0.4, GDN projection −0.9, shared expert −1.1, GEMM −0.3 —
net ≈ +9.4 ms/token profiled (≈ +14 ms unprofiled), i.e. the acceptance collapse turns
per-row inefficiencies (LM head ×1.0, recurrence ×1.0, MoE ×1.8 amortization) into a net
loss. Whether the collapse itself is a sampling-mode artifact or a real draft regression is
exactly the rider question above.

### Implications for Phase B (evidence-gated, unchanged order)

- **New top B-candidate (byte-neutral):** route decode-verifier LM-head rows through the
  existing grouped decode-equivalent path (weights stream once per run instead of per row);
  measured ceiling ≈ up to ~6.5 ms/token of the MTP-dyn gap. Needs the why-is-it-per-row
  code check first.
- **B2 (shared expert one-team transaction)** stands, measured headroom ≈ 2× on 3.51 ms/step.
- **B1 (M=1 through the FFN transaction)** now has direct production numbers to test against
  (transaction at M̄5.9 = ~150-160 µs/row vs M=1 path 279.5 µs/call); decision after a
  B-phase microbench (transaction-at-M=1 vs bundle+down, byte-identical).
- **B3 (GDN a/b subgroup merge)**: small (~1 region × 30 layers); conditional on the
  compatibility-rule question (mixed codebook members inside one pooled region).
- **B4 (LM head top-k/shortlist)**: unchanged — operator-level contract decision; this data
  quantifies the ceiling (20.1% of MTP-dyn decode; 10.2% of no-MTP decode).

Evidence: seal `K14-I16A-880429059-20261111` (SUMMARY.md, g1a–g2b outputs, both stats JSONs,
plan JSONs/logs, SHA256SUMS); reducer + reduction notes by the idea-16 addon session.
