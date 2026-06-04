# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence, llama.cpp anchors, and the
active gap. Keep this concise; raw tuning history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 671.37 | 40.82 | fits 24GB after planner fix |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 582.66 | 53.83 | near fixed d1; controller starts min |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 899.41 | 107.02 | source-token expert quant |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 799.47 | 135.46 | depth 1, 87.50% accept |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`.
`llama-bench -p 768 -n 64 -ngl 999 -r 3` is MTP-off only. MTP uses generated
`mtp-*` sidecars and `llama-cli --single-turn`.
Artifacts: `benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.

| Lane | Model | Prefill | Decode | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` no-MTP | MoE 35B A3B | 1120.1 | 112.6 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Latest Evidence

CUDA dense memory planner smoke:
`benchmark_results/memory_planner/20260604T203612Z-cuda-dense-activation-cap`

| Case | Context | Act.Seq | KV | Arena | Result |
|---|---:|---:|---:|---:|---|
| Qwen3.6 27B CUDA tiny prompt | 16384 | 4096 | 4.0 GB | 1785 MB | pass |

CUDA MoE after source-token expert activation quantization:
`benchmark_results/cuda_moe_mtp/20260604T210314Z-source-token-quant`
`benchmark_results/cuda_moe_mtp/20260604T210400Z-source-token-quant-mtp-dynamic`
Profiler export check:
`benchmark_results/cuda_moe_mtp/20260604T212136Z-perf-export-device-field`

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 899.41 | 107.02 | n/a |
| dynamic MTP | 799.47 | 135.46 | 87.50% |

Focused correctness gates:

- `V2_Integration_CUDAMoEKernel`
- Qwen3.6 MoE CUDA math prefill parity.
- Qwen3.6 MoE CUDA benchmark-style parity: no-MTP skip-gather argmax,
  fused verifier prefill, and MTP skip-gather reference.

## Retained Actions

- ROCm dense/MoE: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN restore, compact active-expert MoE prefill grids.
- CUDA dense/MoE: verifier-row GDN restore, graph-capturable small-M attention,
  grouped verifier prefill, stream-explicit shared experts, fused split-K MoE,
  graph-capturable cuBLAS batched GDN projections, parallel runtime router top-k,
  source-token MoE prefill activation quantization, workspace-bound FP32 mapped
  output redirects.
- CUDA and ROCm GPU greedy argmax tie-break to lowest token id, matching CPU.
- Memory planning charges terminal-row logits and prepared embedding workspace only.
- GPU activation arenas are capped to prefill-bucket capacity while KV keeps the
  requested context capacity; oversized monolithic graph shapes hard fail.
- Stage profiling split: `stage_gpu` is GPU-event eager timing, `stage_executor_cpu`
  is host attribution, and graph replay timing is under `forward_graph`.
- Explicit non-null GPU stream hard failures remain required.

## Next Work

Updated goal is to beat llama.cpp CUDA on dense and MoE, prefill and decode, with
MTP on and off. Dense decode is close; MoE dynamic MTP decode now trails llama.cpp
MTP d1 by about 5%. MoE prefill is the largest visible gap: latest CUDA no-MTP
prefill is 899.41 tok/s versus llama-cli 1120.1 and llama-bench 2413.54. Profiling
says MoE router/expert prefill dominates, so the next target is router prefill
math/layout, then remaining verifier GDN/router budget.
