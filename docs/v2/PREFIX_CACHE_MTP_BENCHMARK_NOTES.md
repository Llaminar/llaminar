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
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | stochastic dynamic MTP | 727.40 | 42.43 | effective d1, near baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 233.54 | 30.21 | current clean baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | retired selected-row d3 MTP | 217.10 | 34.40 | failed equivalence, not accepted |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic accepted-count MTP | 217.43 | 19.99 | residual-batch, speed-negative |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic dynamic MTP | 217.92 | 26.37 | effective d1, below baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | stochastic fixed d1 MTP | 214.37 | 27.62 | small-k sampler, below baseline |
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
- Dense default residual-batch stochastic MTP is still speed-negative: CUDA
  36.82 tok/s at 62.07% acceptance, ROCm 19.99 tok/s at 61.18%.
- Dynamic stochastic depth avoids the default prompt's fixed-d3 acceptance trap
  and matches fixed d1: CUDA 42.43 vs 42.37 tok/s, ROCm 26.37 vs 26.71 tok/s.
- ROCm generated decode dispatch now obeys the graph-safe small-M `kb<=8`
  contract. The follow-up small-k stochastic sampler specialization keeps the
  graph-captured Qwen3.6 `top_k=20` path in `k<=32` kernels and raised fixed-d1
  stochastic ROCm dense decode from 25.92 to 27.62 tok/s. Sampler cost fell from
  about 9.24 to 8.67 ms/step; verifier forward remains the main blocker at
  about 41.24 ms.

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
