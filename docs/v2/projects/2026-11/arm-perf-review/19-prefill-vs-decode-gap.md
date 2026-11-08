# Idea 19 — Prefill vs decode gap on GB10 (short-prompt effect)

Status: **Open** (analysis; no long-prompt measurement yet).

## Finding

GB10 Release serving, short bench prompt (head numbers, 2026-11-08):

- 35B: prefill **~37 tok/s** vs decode **~27 tok/s** (MTP), i.e. prefill is
  now *ahead* of decode after the bundle fix, but still below what a
  memory-bound M=64+ prefill implies (Mac VM Q4_K 17408x5120 M=64: 39.7 ms ≈
  **287 GOPS** at 10 threads).
- 27B: prefill **~10.5** vs decode **~4.6** tok/s.
- At A `878a33fea`: 35B prefill 24.0–24.7 vs decode 18.18; at B `144ece037`:
  22.6–22.8 (the overlay regression, idea 12); at C `268cd81de`: 37.1–37.8.

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
