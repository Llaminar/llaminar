# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 671.37 | 40.82 | fits 24GB |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 582.66 | 53.83 | near fixed d1 |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2156.47 | 108.19 | inverse-map top-k scatter |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1660.13 | 140.50 | fused runtime decode, 88.28% accept |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1659.24 | 141.55 | depth 1, 87.50% accept |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`.
`llama-bench -p 768 -n 64 -ngl 999 -r 3` is MTP-off only. MTP uses generated
`mtp-*` sidecars and `llama-cli --single-turn`.
Artifacts: `benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.
Note: `llama-cli` MTP uses the Qwen chat template plus assistant/thinking
prefix; Llaminar `benchmark` uses the raw 595-token prompt. Acceptance rates are
not exact apples-to-apples quality data.

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
`benchmark_results/cuda_moe_mtp/20260605T022856Z-fused-runtime-decode`

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2156.47 | 108.19 | n/a |
| fixed d1 MTP | 1660.13 | 140.50 | 88.28% |
| dynamic MTP | 1659.24 | 141.55 | 87.50% |

Fresh checks:
- Perf export/stage fix:
  `20260604T230244Z-perf-export-no-stage-events`,
  `20260604T231108Z-stage-timing-accumulator-fix`.
- CUDA runtime-routed MoE decode now has a graph-replay regression:
  `RuntimeGroupedDecodeFusedMatchesTwoStepAndGraphReplays`.
  Counter-only run shows `cuda_moe_grouped_decode_fused_calls`,
  `route=fused_kpart`. Throughput impact is neutral to slight-positive on
  dynamic, not a decisive speedup.
- Qwen3.6 MoE CUDA MTP sidecar stage breakdown now passes after graph-aware
  snapshot keying and BF16 shared-gate materialization.
- CUDA hybrid d1 skips the post-sidecar hybrid checkpoint when verifier-row
  restore is supported. Perf export confirms no post-sidecar capture and halves
  hybrid checkpoint exports from 384 to 192 per benchmark window.
- Greedy margin diagnostic:
  `20260604T220430Z-greedy-margin-verifier-rows`; margins show real
  draft/main disagreement, not mostly argmax tie noise.

Focused CUDA correctness gates: `V2_Integration_CUDAMoEKernel`, Qwen3.6 MoE
CUDA math prefill parity, and Qwen3.6 MoE CUDA benchmark-style parity.

## Retained Actions

- ROCm dense/MoE: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN restore, compact active-expert MoE prefill grids.
- CUDA dense/MoE: verifier-row GDN restore, graph-capturable small-M attention,
  grouped verifier prefill, stream-explicit shared experts, fused split-K MoE,
  cuBLAS batched GDN projections, parallel router top-k, source-token MoE
  activation quantization, mapped output redirects, inverse-map top-k scatter,
  token-direct verifier down accumulation, post-sidecar checkpoint elision, and
  fused runtime-routed MoE decode.
- CUDA and ROCm GPU greedy argmax tie-break to lowest token id.
- Memory planning caps activation arenas to prefill-bucket capacity while KV
  keeps requested context capacity; oversized monolithic graph shapes hard fail.
- Stage profiling split: `stage_gpu` is explicit eager per-stage event timing,
  `stage_executor_cpu` is host attribution, and captured replay timing is under
  `forward_graph`; JSON/CSV export alone stays non-intrusive.
- Explicit non-null GPU stream hard failures remain required.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
Dense decode is close; CUDA MoE dynamic is now within noise of the llama.cpp MTP
d1 anchor. Next targets are verifier/correction costs, prefill gap, and a strict
same-prompt llama.cpp/Llaminar comparison.
