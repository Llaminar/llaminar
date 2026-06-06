# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard for current CUDA/ROCm evidence. Raw history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 707.92 | 41.73 | current baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, sequential verifier | 598.97 | 38.33 | common safe policy, speed-negative |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 609.73 | 38.19 | 84.95% acceptance, same verifier wall |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 610.88 | 38.52 | holds depth 1; still below baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive, beats lcpp MTP d1 |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1943.69 | 145.36 | speed-positive |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

Artifacts:
CUDA dense `/tmp/llaminar-mtp-bench/dense-cuda-phase138-{nomtp,shared-d3}-default.json`;
CUDA MoE `benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode`.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`. MTP-off uses `llama-bench`; MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`. Artifact:
`benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` no-MTP | MoE 35B A3B | 1120.1 | 112.6 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Current Findings

- CUDA dense no-MTP decode is restored by generated NativeVNNI GEMV dispatch
  rules. Dense MTP remains speed-negative because accepted speculative tokens
  still pay decode-equivalent verifier forwards.
- ROCm and CUDA now share the same GDN verifier policy and the same
  `MTPDecodeCatchup` contract. The promoted path is `shared_stepwise`: commit
  shifted row, forward one decode row, sample, and repeat. Long-prefix one-row
  restore and M=2 sequential-equivalence parity are green on both backends, while
  raw all-position GDN verifier rows remain unpromoted.
- Phase 13.8 now has an explicit optimized catch-up capability hook on
  `IInferenceRunner`; no CUDA/ROCm backend advertises it. The stats capture
  `/tmp/llaminar-mtp-bench/dense-cuda-phase138-shared-d3-default-stats.csv`
  shows why: `decode_equivalent_catchup_forward_one` is 9272.9 ms across 383
  forwards, while shifted commits are 98.6 ms and sampling is 11.8 ms. A wrapper
  cannot fuse away the full main decode required by the contract.
- CUDA transaction validation is green again: committed-state stamps are scalar,
  payload-bearing snapshots are not cloned for metadata, and moved snapshots
  leave nested payload handles empty.
- CUDA MoE workspace rebind and graph-capture crashes are fixed. Current MoE
  fixed d1 is 148.5 tok/s versus 119.9 no-MTP and 142.0 llama.cpp MTP d1.

## Retained Actions

- CUDA/ROCm dense: chase speedup through a proven state shortcut or lower
  main-decode kernel cost. Do not re-enable raw GDN all-position verifier-row
  shortcuts or call a wrapper around full decode replay an optimized catch-up.
- CUDA MoE: keep the 148.5 tok/s ratchet and extend long-prompt/controller
  evidence without weakening parity.
- ROCm: continue toward the 2x dense target by reducing captured verifier GPU
  work in GEMM, fused Gate/Up, GDN projection, recurrence, and LM head.
- Shared: CUDA/ROCm all-codebook M=2/3/4 GEMV equivalence is green;
  keep dispatch tables bucket-aligned and parity tests in the normal suite.
