# Prefix Cache And MTP Benchmark Notes

Concise Phase 14 scoreboard for Qwen3.6 MTP/prefix-cache tuning. Raw run
history stays in `benchmark_results/` and `/tmp/llaminar-mtp-bench`; the phased
plan carries detailed implementation status.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 877.55 | 43.81 | current clean baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | retired selected-row d3 MTP | 726.19 | 87.07 | historical 1.99x target, not accepted |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic accepted-count MTP | 727.01 | 36.82 | residual-batch, speed-negative |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic fixed d1 MTP | 707.96 | 54.60 | two-pass top-k, near l.cpp d1 |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic dynamic MTP | 706.09 | 55.80 | two-pass top-k, effective d1 |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 233.90 | 30.16 | dispatch-refresh baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | retired selected-row d3 MTP | 217.10 | 34.40 | failed equivalence, not accepted |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic accepted-count MTP | 217.43 | 19.99 | residual-batch, speed-negative |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic fixed d1 MTP | 214.07 | 35.85 | Q4/Q5/Q6 decode dispatch refresh |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic dynamic MTP | 214.88 | 35.56 | Q4/Q5/Q6 decode dispatch refresh |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 76.35 | 31.92 | short-lane baseline |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | retired selected-row d3 MTP | 73.05 | 54.23 | historical 1.70x target |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device promotion |

## Current Findings

- Phase 13.8 remains the vLLM-style accepted-count transaction path. Retired
  selected-row numbers are speed targets only, not acceptance evidence.
- Dead raw all-position and sidecar-chain shortcut code/tests are removed.
  Retained work uses speculative state slots, accepted-count publication,
  shifted-row commit, correction suffix forward, and ready-token handoff.
- CUDA/ROCm stochastic MTP now batches compact-distribution accept decisions and
  residual correction candidates in one graph-capturable verifier launch; the
  focused runner, GPU sampling, graph-smoke, and Qwen3.6 stochastic parity gates
  are green.
- CUDA and ROCm both use arena-declared scratch for graph-captured
  `top_k<=32` distribution building; focused smokes assert the scratch counter.
  CUDA fixed d1/dynamic reached 54.60/55.80 tok/s, essentially matching the
  llama.cpp d1 anchor.
- ROCm generated NativeVNNI decode dispatch now covers Qwen3.6 hot verifier
  shapes for codebooks 0/5/7/8. This improves coverage and keeps parity green,
  but default-lane throughput remains about 35.8 tok/s; verifier forward is
  still the catch-up target at about 41.6 ms versus CUDA's 28.5 ms.
- ROCm batched small-M verifier dispatch training is now offline-only. The
  generated candidate looked correct in the perf harness but collapsed
  real-model stochastic acceptance, so no batched table is promoted without a
  model-level acceptance/equivalence gate.

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

- Finish Phase 13.8 by broadening the shared transaction path across stop,
  prefix, continuation, MoE, and TP/PP cases, then tune CUDA and ROCm
  greedy/stochastic MTP until both post comparable wins or any remaining gap is
  trace-backed. Do not restore raw all-position verifier shortcut code.
- Keep CUDA MoE 148.5 tok/s and ROCm MoE 42.04 tok/s as current ratchets.
- Return to TP/PP/EP overlay MTP after SingleDevice dense promotion is stable.
