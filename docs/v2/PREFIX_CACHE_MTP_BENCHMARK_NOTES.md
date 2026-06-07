# Prefix Cache And MTP Benchmark Notes

Concise Phase 14 scoreboard for Qwen3.6 MTP/prefix-cache tuning. Raw run
history stays in `benchmark_results/` and `/tmp/llaminar-mtp-bench`; the phased
plan carries detailed implementation status.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 708.33 | 41.73 | current promoted-path baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct candidate | 602.79 | 67.92 | historical target, not accepted |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 609.73 | 38.19 | old blocker, 84.95% acceptance |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 233.21 | 31.10 | current promoted-path baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct candidate | 218.27 | 36.14 | historical target, not accepted |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | fixed d1 MTP, Phase 13.8 direct candidate | 218.24 | 39.84 | historical target, not accepted |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 76.35 | 31.92 | short-lane baseline |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct | 73.05 | 54.23 | 1.70x, 88.57% acceptance |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device promotion |

Artifacts:

- CUDA dense: `/tmp/llaminar-mtp-bench/phase138-accept/dense-cuda-{nomtp,promoted-d3}.json`
- ROCm dense: `/tmp/llaminar-mtp-bench/phase138-accept/dense-rocm-{nomtp,promoted-d3}.json`
- Prior direct-gate dense: `/tmp/llaminar-mtp-bench/dense-{cuda,rocm}-phase138-direct-*`
- ROCm qbf: `/tmp/llaminar-mtp-bench/dense-rocm-phase138-direct-{nomtp,d3}-qbf-c64-n48.json`
- CUDA MoE: `benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode`

## Current Findings

- Phase 13.8 vLLM-style transaction is the active dense SingleDevice GPU
  design, but the direct-candidate benchmark numbers above are historical
  targets until the accepted-count speculative state-slot path is fully proven.
- CUDA default-lane depth 3 is the best historical dense target: 67.92 tok/s
  versus 41.73 no-MTP, with 36 verifier runs, 159 verifier tokens, and zero
  transaction validation failures.
- ROCm demonstrates a smaller historical direct-candidate win: default prompt
  was 1.16x at 58.82% acceptance, while the high-acceptance `qbf` lane reached
  1.70x at 88.57% acceptance. After acceptance, ROCm must post wins comparable
  to CUDA or get a trace-backed tuning pass before Phase 14 claims.
- Sequential verifier MTP remains documented as the old blocker. It was correct
  but speed-negative because accepted speculative tokens paid repeated
  decode-equivalent verifier forwards.
- The raw all-position and sidecar-chain shortcut code/tests have been removed;
  live work must go through accepted-count speculative state slots.
- Source-owned parity covers CUDA/ROCm depth-3 normal decode, CUDA
  benchmark-prompt depth-3, CUDA/ROCm depth-1 prefix restore, and normal
  CUDA/ROCm stochastic MTP verifier cells through the shared Phase 13.8
  transaction gate. The latest focused Phase 13.8 gate passed 9/9 including the
  prefix-flow verifier-row restore regression. MoE, TP/PP, chained-prefix,
  stop-token, broader continuation coverage, and stochastic benchmark evidence
  remain pending.

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
