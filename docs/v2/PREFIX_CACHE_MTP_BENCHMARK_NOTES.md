# Prefix Cache And MTP Benchmark Notes

Concise Phase 14 scoreboard for Qwen3.6 MTP/prefix-cache tuning. Raw run
history stays in `benchmark_results/` and `/tmp/llaminar-mtp-bench`; the phased
plan carries detailed implementation status.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 707.17 | 41.72 | current direct-candidate baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct | 602.01 | 74.58 | 1.79x, 96.84% acceptance |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 609.73 | 38.19 | old blocker, 84.95% acceptance |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 233.97 | 31.22 | current direct-candidate baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct | 217.23 | 35.86 | 1.15x, 60.47% acceptance |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | no MTP | 76.35 | 31.92 | short-lane baseline |
| Dense `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP, Phase 13.8 direct | 73.05 | 54.23 | 1.70x, 88.57% acceptance |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device promotion |

Artifacts:

- CUDA dense: `/tmp/llaminar-mtp-bench/dense-cuda-phase138-direct-{nomtp,d3}-default.json`
- ROCm dense: `/tmp/llaminar-mtp-bench/dense-rocm-phase138-direct-{nomtp,d3}-default.json`
- ROCm qbf: `/tmp/llaminar-mtp-bench/dense-rocm-phase138-direct-{nomtp,d3}-qbf-c64-n48.json`
- CUDA MoE: `benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode`

## Current Findings

- Phase 13.8 direct mode is now the first dense SingleDevice path that turns the
  vLLM-style transaction into real decode speedup on both CUDA and ROCm.
- CUDA default-lane depth 3 is the best current dense signal: 74.58 tok/s versus
  41.72 no-MTP, with 33 verifier runs, 134 verifier tokens, and zero transaction
  validation failures.
- ROCm proves the same strategy but remains acceptance-sensitive: default prompt
  is only 1.15x at 60.47% acceptance, while the high-acceptance `qbf` lane is
  1.70x at 88.57% acceptance.
- Sequential verifier MTP remains documented as the old blocker. It was correct
  but speed-negative because accepted speculative tokens paid repeated
  decode-equivalent verifier forwards.
- The retired shortcut ledger still stands: sidecar-chain verifier state and raw
  all-position selected-state restore are not decode-equivalent for Qwen3.6
  dense.
- `LLAMINAR_MTP_PHASE138_DIRECT_CANDIDATE=1` is still an explicit development
  gate. Promotion requires broader parity for accept-all, reject-first,
  reject-after-prefix, stop-token, prefix-restore, and continuation cases.

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

- Broaden Phase 13.8 direct-candidate parity, then promote or retire the explicit
  gate based on evidence. Do not re-enable raw all-position verifier shortcuts.
- Keep CUDA MoE 148.5 tok/s and ROCm MoE 42.04 tok/s as current ratchets.
- Return to TP/PP/EP overlay MTP after SingleDevice dense promotion is stable.
