# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 671.37 | 40.82 | fits 24GB |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 582.66 | 53.83 | near fixed d1 |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2156.95 | 109.91 | shared-expert grouped decode default |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1628.13 | 141.05 | post-reset, best 142.74 |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1653.16 | 139.46 | request reset, 85.94% accept |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`.
MTP-off uses `llama-bench`; MTP uses generated `mtp-*` sidecars with
`llama-cli --single-turn`. Artifact:
`benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.
Prompt/template differences mean acceptance is directional, not exact parity.

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

CUDA MoE latest:
`20260605T031355Z-reset-check-no-mtp`,
`20260605T031848Z-post-reset-rerun-mtp-fixed`,
`20260605T031906Z-post-reset-rerun-mtp-dynamic`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2156.95 | 109.91 | n/a |
| fixed d1 MTP | 1628.13 | 141.05 | 84.38% |
| dynamic MTP | 1653.16 | 139.46 | 85.94% |

Fresh checks:
- Perf export/stage fixes:
  `20260604T230244Z-perf-export-no-stage-events`,
  `20260604T231108Z-stage-timing-accumulator-fix`,
  `20260605T024434Z-graph-replay-stage-stats`.
- Captured replay exports `stage_gpu.graph_replay.*` wall timings and
  `stage_gpu.graph_replay_plan_*` metadata to JSON/CSV; attribution shows main
  verifier replay around 11.05 ms for 524 stages and MTP sidecar around 0.79 ms.
- CUDA runtime-routed MoE decode has graph-replay regression coverage via
  `RuntimeGroupedDecodeFusedMatchesTwoStepAndGraphReplays`; counter-only runs
  show `cuda_moe_grouped_decode_fused_calls`, `route=fused_kpart`.
- CUDA shared-expert decode now uses the grouped table path by default and has
  benchmark-style parity coverage with perf-counter evidence.
- Dynamic MTP request state resets across benchmark-style `clearCache()`; JSON
  counters now describe the measured request rather than warmup plus all runs.

Focused gates: `V2_Integration_CUDAMoEKernel`, Qwen3.6 MoE CUDA math parity,
and Qwen3.6 MoE CUDA benchmark-style parity.

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
- Stage profiling split: `stage_gpu` has eager per-stage event timing plus
  captured graph replay wall/plan stats, `stage_executor_cpu` is host
  attribution, and `forward_graph` remains graph-control attribution.
- Explicit non-null GPU stream hard failures remain required.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
Dense decode is close; CUDA MoE fixed d1 best observed edges the llama.cpp MTP
d1 anchor, while latest dynamic remains acceptance-sensitive. Next targets are
controller stability, verifier/correction costs, prefill gap, and same-prompt comparison.
