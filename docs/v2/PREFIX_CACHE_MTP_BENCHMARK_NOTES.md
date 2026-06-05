# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence, llama.cpp anchors, and current
gaps. Keep this concise; raw tuning history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 671.37 | 40.82 | fits 24GB |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 582.66 | 53.83 | near fixed d1 |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2158.10 | 107.14 | inverse-map top-k scatter |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1657.39 | 131.27 | depth 1, 78.52% accept |
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

CUDA MoE ratchet:
`benchmark_results/cuda_moe_mtp/20260604T222925Z-token-direct-down-no-mtp`
`benchmark_results/cuda_moe_mtp/20260604T222925Z-token-direct-down-mtp-dynamic`

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2158.10 | 107.14 | n/a |
| dynamic MTP | 1657.39 | 131.27 | 78.52% |

Fresh checks:
- Perf export/stage fix:
  `20260604T230244Z-perf-export-no-stage-events`,
  `20260604T231108Z-stage-timing-accumulator-fix`.
- Qwen3.6 MoE CUDA MTP sidecar stage breakdown now passes after graph-aware
  snapshot keying and BF16 shared-gate materialization; the formerly bad
  shared-gate/combined/LM-head rows are above threshold.
- Greedy margin diagnostic:
  `20260604T220430Z-greedy-margin-verifier-rows`; margins show real
  draft/main disagreement, not mostly argmax tie noise.
- Rejected CUDA MoE sweeps:
  `20260604T233501Z-kpart-sweep-d1` kept 16/16 split-K;
  `20260604T233707Z-tilem-sweep-d1` kept auto verifier tile 2/64;
  `20260604T234649Z-smallm-router-d1` and rerun rejected a shared-gate
  small-M router path at 129.6-130.4 tok/s.

Focused CUDA correctness gates: `V2_Integration_CUDAMoEKernel`, Qwen3.6 MoE
CUDA math prefill parity, and Qwen3.6 MoE CUDA benchmark-style parity.

## Retained Actions

- ROCm dense/MoE: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN restore, compact active-expert MoE prefill grids.
- CUDA dense/MoE: verifier-row GDN restore, graph-capturable small-M attention,
  grouped verifier prefill, stream-explicit shared experts, fused split-K MoE,
  graph-capturable cuBLAS batched GDN projections, parallel runtime router top-k,
  source-token MoE prefill activation quantization, workspace-bound FP32 mapped
  output redirects, inverse-map MoE top-k scatter, token-direct verifier down
  accumulation.
- CUDA and ROCm GPU greedy argmax tie-break to lowest token id.
- Memory planning caps activation arenas to prefill-bucket capacity while KV
  keeps requested context capacity; oversized monolithic graph shapes hard fail.
- Stage profiling split: `stage_gpu` is explicit eager per-stage event timing,
  `stage_executor_cpu` is host attribution, and captured replay timing is under
  `forward_graph`; JSON/CSV export alone stays non-intrusive.
- Explicit non-null GPU stream hard failures remain required.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
Dense decode is close; CUDA MoE dynamic MTP trails llama.cpp d1 by about 8%.
Next targets are verifier/correction costs and acceptance quality without
giving back the token-direct verifier down and prefill scatter wins.
