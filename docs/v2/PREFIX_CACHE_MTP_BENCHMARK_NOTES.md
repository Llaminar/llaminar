# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard for Qwen3.6 MTP and prefix-cache tuning.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 877.55 | 43.81 | clean baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic fixed d1 | 707.96 | 54.60 | top-k small path, accepted short smoke |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic dynamic | 706.09 | 55.80 | effective d1 |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 233.17 | 30.14 | fresh retained baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic fixed d1 | 216.79 | 24.83 | sidecar-base restore skipped, 82.81% accept |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic dynamic | 216.55 | 24.84 | held depth 1, no updates |
| Dense qbf, c64/n48 | ROCm | Qwen3.6 27B Q4_K_S | retired selected-row d3 | 73.05 | 54.23 | historical target only |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | clean baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 | n/a | 42.04 | 2.13x ratchet |

## Current Findings

- Retired selected-row and sidecar-chain shortcut numbers are historical targets,
  not acceptance evidence. Do not restore their code or tests.
- Stateful Qwen3.6 greedy and stochastic MTP use the shared decode-equivalent
  verifier path. The old accepted-count publication candidate is not promoted
  for stateful rows that require replay.
- Fresh ROCm retained-path profiling shows stochastic fixed depth-1 still pays
  two one-token main forwards per speculative pair. Skipping the redundant
  verifier-base restore when the sidecar preserves main state improved decode
  from 18.98 to 24.83 tok/s; the next speed lever is a proven state transaction
  path that removes the extra replay work.
- Shared Phase 13.8 publication-plan metadata now derives accepted-state slots,
  target cached-token counts, correction replay spans, and bonus-ready rows
  without mutating live state. The device publication API is a separate opt-in
  runner seam and is still unpromoted.
- CUDA and ROCm direct GDN/short-conv kernel tests now both cover restored
  verifier slot state followed by multi-step continuation replay equivalence.
- A lagged-terminal-output experiment cut verifier replay calls in half but
  paid the same forward as next-step `condition_forward`, leaving ROCm d1 at
  24.86 tok/s. The code was removed; do not retry this as a speed path.
- CUDA/ROCm stochastic sampling keeps first-token, draft-token, target
  distribution, accept/residual, and ready-token work on compact device-resident
  tables. `top_k<=64` uses arena-declared graph-captured scratch.
- ROCm long clear-cache stochastic replay is a retained deterministic regression
  gate. Exact same-seed replay uses `LLAMINAR_DETERMINISTIC=1`; production
  stochastic benchmark repeats may still vary near distribution thresholds.
- ROCm NativeVNNI decode dispatch covers Qwen3.6 hot verifier shapes for
  codebooks 0/5/7/8. Batched small-M verifier dispatch training remains
  offline-only until a model-level stochastic acceptance/equivalence gate passes.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`; MTP-off uses `llama-bench`, MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Retained Actions

- Find a CUDA/ROCm-common fast verifier strategy that preserves
  decode-equivalent state before using dense MTP as Phase 14 evidence.
- Keep CUDA MoE 148.5 tok/s and ROCm MoE 42.04 tok/s as current ratchets.
- Return to TP/PP/EP overlay MTP after SingleDevice dense promotion is stable.
