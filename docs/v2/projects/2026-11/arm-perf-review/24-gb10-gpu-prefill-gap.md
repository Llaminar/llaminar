# Idea 24 — GB10 GPU prefill gap vs llama.cpp (3.3×)

Status: **Open**. Tracked follow-up (CUDA/GPU side, GB10); discovered via the
idea 17 cross-engine matrix, 2026-11-08.

## Finding

Same qwen4exp UD-Q4_K_XL GGUF on `.79` (GB10, sm_121, Release); llama.cpp
`cf34d31` (GGML_CUDA sm_121, `-ngl 99`, host CUDA 13.0.88), llama-bench 3
repeats:

| Engine | Prefill pp2160 | Decode tg256 |
|---|---|---|
| llama.cpp CUDA | 1063.24 ± 13.10 tok/s | 28.80 ± 0.03 tok/s |
| Llaminar CUDA (all-CUDA:0) | 318.6 tok/s | 22.82 tok/s |
| llama.cpp CPU | 40.88 ± 0.09 | 3.06 ± 0.11 |

- **Prefill: 3.3× behind** (1063 vs 319) — a real engine perf gap on prefill.
- **Decode: ~21% behind** (22.8 vs 28.8) — smaller.
- The 4-digit prefill **is** reachable on GB10 (llama.cpp proves it), so the
  ceiling is not hardware; Llaminar-GPU decode is healthy vs CPU (7.5–8×).

## Candidate causes (now measured — see dig below)

- Captured **multi-bucket prefill pipeline** overhead (graph capture, bucket
  scheduling) vs llama.cpp's single-pass FlashAttention prefill;
- prefill GEMM / attention kernels (sm_121-specific tuning; recall the sm_120/
  sm_121 spill guard is still open — idea 21);
- GDN path in prefill.

## Stage-profile dig (2026-11-08, from existing benchmark evidence)

**Work profile** — captured stage execute totals, 512-token prefill bucket,
1283 stage objects, 305.1 ms total:

- GEMM: 175.0 ms / 364 samples (max 170.1 ms) = **57%**
- GDN_PROJECTION: 119.8 ms / 36 samples (max 119.1 ms) = **39%**
- MOE_EXPERT_FFN: 1.92 ms / 48; everything else < 1 ms

→ ~96% of prefill stage work is GEMM + GDN_PROJECTION; routed-expert FFN is
negligible in prefill.

**Kernel vs overhead split** (2160-token run):

- Captured stage kernel time ≈ **0.60 ms/token** (512 tok / 305 ms) → if
  replayed at stage speed, ~**1660 tok/s**;
- measured benchmark marginal slope **3.009 ms/token** (approx 332 tok/s) plus a
  fixed **~279 ms per-request intercept**;
- → measured execution is ~**5× the stage kernel time**, so ~80% of prefill
  wall time is **NOT kernel math**: per-bucket replay/launch/sync overhead of
  the multi-bucket captured pipeline (buckets 512/448/384/256/128/64, graph
  replay + event frontiers per bucket), plus ~280 ms fixed;
- llama.cpp GPU reference: 0.94 ms/token (1063 tok/s), single-pass, no bucket
  boundaries.

**Conclusion:** the 3.3× prefill gap is **dominated by captured
bucket-pipeline overhead, not GEMM/GDN kernel efficiency** (kernels would
support ~1600 tok/s). See idea 25.

## Next steps

- Dig into the per-stage prefill profile (`MOE_EXPERT_FFN`, GEMM buckets, GDN,
  attention) for the same prompt/model and compare against llama.cpp's
  single-pass structure;
- separate capture/bucket overhead (measure with capture disabled or a
  single-bucket path) from kernel efficiency;
- sm_121 kernel tuning once the spill guard is clean (idea 21).

## Validation

- Same model/prompt/length, byte-identical outputs vs Llaminar cells;
  target: close the 3.3×; then re-run the full matrix (CPU + both CUDA
  engines) for the record.
