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
