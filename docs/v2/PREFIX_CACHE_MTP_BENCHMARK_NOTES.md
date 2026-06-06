# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard for current CUDA/ROCm evidence. Raw history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 704.05 | 41.61 | decode restored after GEMV dispatch fix |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, sequential verifier | 595.72 | 38.63 | correct, verifier still slower than baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 595.50 | 32.06 | correct, deeper verifier overhead |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2460.26 | 111.30 | stable after MoE graph-boundary fix |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1825.22 | 102.06 | sequential verifier is now default |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

Artifacts:
CUDA dense `benchmark_results/cuda_dense_mtp/20260606T070336Z-dense-cuda-stage-attribution`;
CUDA MoE crash fix `benchmark_results/cuda_moe_mtp/20260606T111408Z-d1-crash-isolation`;
CUDA MoE checkpoint pool `benchmark_results/cuda_moe_mtp/20260606T122032Z-checkpoint-pool`;
CUDA MoE sequential default `benchmark_results/cuda_moe_mtp/20260606T123010Z-sequential-default`.

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
  rules. CUDA verifier-row shortcuts remain disabled because full-graph
  all-position verifier state still diverges, despite lower-level recurrence and
  short-conv restore regressions passing.
- CUDA MoE workspace rebind and graph-capture crashes are fixed. `CUDAMoEKernel`
  invalidates stale scratch on bind, and `moe_combine` starts a fresh captured
  segment so it is not fused with routing/shared/routed expert scratch stages.
- Graph failure attribution is stricter: checked stream syncs, full segment
  names in failure logs, optional CUDA embedding pointer validation, and passive
  perf JSON export.
- CUDA MoE MTP live hybrid checkpoints now reuse pooled storage. The measured
  checkpoint storage timer dropped from about 25.9 ms/step to 0.002 ms/step,
  lifting fixed d1 decode from 32.0 to 64.1 tok/s. The remaining CUDA MoE MTP
  wall is verifier replay plus shifted-row commit, not checkpoint allocation.
- CUDA greedy MTP now defaults to the sequential verifier path. On MoE fixed d1,
  this lifts decode to 102.1 tok/s with parity green, close to but still below
  the 111.3 tok/s no-MTP baseline.

## Retained Actions

- CUDA: recover MTP speed after the correctness fixes. Fixed d1 MoE is stable
  at 102.1 tok/s versus 111.3 tok/s no-MTP, so target sidecar plus verifier
  overhead and acceptance-sensitive depth policy before claiming speedup.
- ROCm: continue toward the 2x dense target by reducing captured verifier GPU
  work in GEMM, fused Gate/Up, GDN projection, recurrence, and LM head.
- Shared: keep generated GEMM/GEMV dispatch tables aligned with prefill buckets,
  GPU streams explicit, and parity tests in the normal suite.
