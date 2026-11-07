# Idea 19 — Prefill vs decode gap on GB10 (short-prompt effect)

Status: **Open** (analysis; no long-prompt measurement yet).

## Finding

GB10 Release serving, short bench prompt:

- 35B: prefill ~24.7 tok/s vs decode 18.5 tok/s (MTP).
- 27B: prefill ~10.4 tok/s vs decode 4.2–4.5 tok/s.

The bench prompt is short, so the low prefill rate is probably **latency-bound
small-M prefill**, not the GEMM path. Cross-check: Mac VM, Q4_K 17408x5120,
M=64 prefill: 39.7 ms ≈ **287 GOPS** at 10 threads — the GEMM path itself is
not the bottleneck at M=64.

## First measurements

- Prompt-length sweep (e.g. 128 / 512 / 2048 tokens) on GB10 (Release 20T,
  mixed / X925-only / A725-only) and the Mac VM; report prefill tok/s and
  effective GOPS.
- Compare with decode scaling (idea 1/5, `00-measured-baseline.md` thread
  scaling) to separate launch/latency overhead from bandwidth saturation
  (idea 17).

## Validation

- Same model/GGUF/prompt/greedy/seed as the decode baseline; unprofiled;
  byte-identical outputs; results land in `00-measured-baseline.md`.
