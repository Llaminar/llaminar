# Idea 17 — Why is CPU decode on GB10 so low? (post-merge bandwidth gap analysis)

Status: **Open**. Tracked follow-up requested by the operator after merge;
back-of-envelope analysis provided by the ARM implementer, first measurements
proposed below. No matched external baseline exists yet, so "low" is not yet a
measurement.

## Back-of-envelope (unmeasured ceiling)

Release, 20 cores, MTP off; GGUF sizes plus stage timings from the Spark lane:

- **27B dense Q4_K_S:** ~15 GB of weights read per token at 4.6 tok/s
  ≈ **69 GB/s effective** → kernel ceiling ~5 tok/s. The 35B LM head alone
  (248K×2048, ~0.42 GB in 5.7 ms) also hits ~73 GB/s, so large dense GEMVs
  stream at ~70 GB/s. **Unknown:** GB10's real CPU-side bandwidth ceiling
  (not measured).
- Mac cross-check: 27B gate/up projection (1.1 ms ≈ 45 GB/s) looked
  compute-bound; the exact-rounding contract (no FMA, fixed order, explicit
  rounding) costs compute versus looser kernels.
- **35B-A3B (8 active experts):** ~1.3–1.6 GB/token at 20.6 tok/s
  ≈ **27–33 GB/s**, i.e. only 30–45% of the dense rate → overhead-dominated:
  - (a) small 512×2048 expert matrices: thread launch/barrier cost rivals the
    weight stream. The pooled bundle already cut router+gate/up 668→177 µs and
    experts 803→275 µs (at `268cd81de`).
  - (b) ~40 layers × ~10 parallel regions per token, each a 20-core barrier
    where half the cores are A725 at ~1/2.8 speed → est. several ms of the
    ~48 ms step (idea 13).
  - (c) LM head 5.7 ms ≈ 12% of each token (idea 16).
  - (d) GDN projection 394 µs × 30 layers ≈ 12 ms — now the largest remaining
    stage (idea 16).

## Proposed first measurements (post-merge follow-up)

1. **CPU memory-bandwidth sweep on GB10**, 1..20 cores, mixed / X925-only /
   A725-only, to establish the real ceiling: if ~100 GB/s, the 27B is at ~70%;
   if ~200 GB/s, there is large headroom.
2. **Matched llama.cpp CPU-only comparison on GB10** (same 27B/35B GGUFs,
   prompt, greedy, 20 threads) via `.agents/llama-cpp-comparison`, to see
   whether the gap is dense, MoE, or both.
3. **Attribute the gap:** per-stage timing plus barrier counts per decode step
   (PerfStats counter), then cost the exact-rounding contract against a
   loose-kernel reference on the same shapes.

## Updated serving numbers (supersede the 00 rows where noted)

At `268cd81de`: 35B MTP dynamic **27.5** tok/s, MTP off **20.6**; 27B MTP
dynamic **4.27–4.29**. The per-commit A/B/C runs
(`878a33fea` baseline → `144ece037` overlays → `268cd81de` pooled bundle,
with repeats) were still running on `.80`; at A: 35B 17.9/17.8, 27B 4.16 (MTP)
/ 4.21 (off) tok/s decode.
