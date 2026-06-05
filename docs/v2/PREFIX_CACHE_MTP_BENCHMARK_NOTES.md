# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 716.00 | 40.94 | split-K workspace + KV route fixed |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP | 621.25 | 54.82 | accept 96.88%, graph clean |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 620.57 | 55.43 | depth 1, graph clean |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | beats l.cpp no-MTP |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | accept 71.88%, beats l.cpp d1 |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1943.69 | 145.36 | accept 68.75%, depth 1 |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`. MTP-off uses `llama-bench`; MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`. Artifact:
`benchmark_results/llama_cpp_cuda/20260604T191903Z-6ddc943-mtp-recalibration`.

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

CUDA dense latest:
`benchmark_results/cuda_dense_mtp/20260605T172021Z-dense-default-after-kv-dispatch-fix`;
route sweeps: `20260605T-current-refresh/ffn_m600_sweep` and
`20260605T083341Z-qwen36-gdn-m600-tile-sweep`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 716.00 | 40.94 | n/a |
| fixed d1 MTP | 621.25 | 54.82 | 96.88% |
| dynamic MTP | 620.57 | 55.43 | 95.31% |

CUDA MoE latest:
`20260605T070628Z-iq4nl-word-decode`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2707.70 | 119.91 | n/a |
| fixed d1 MTP | 1946.82 | 148.50 | 71.88% |
| dynamic MTP | 1943.69 | 145.36 | 68.75% |

Fresh checks:
- Trusted `stage_gpu` replay timing, CUDA NativeVNNI route counters, and GPU
  workspace allocation counters export through perf stats.
- CUDA NativeVNNI scratch, MoE request scratch, KV gather/conversion, GDN
  deinterleave, attention decode partials, and MTP sidecar KV buffers bind via
  `DeviceWorkspaceManager`.
- CUDA dense no-MTP prefill uses tested Qwen3.6 Q4_K-family selector fixes:
  FFN down `M=600` split-K 4, exact `M=595` split-1, and attention K/V
  `1024x5120` avoids the rejected generated split-K-2 route.
- NativeVNNI trainer infrastructure is now codebook-synchronized: shared
  tensor-derived codebook map, production generated-dispatch validation,
  CUDA alias tests, ROCm trainer schema tests,
  Qwen3.6 dense attention/FFN/GDN/LM-head shapes, and M=2/3/4 sweep buckets.
  Focused smokes passed for CUDA prefill M=2/3/4 and ROCm prefill/decode.
  ROCm decode `Q4_K` on `Qwen36_GDN_TimeProjection` is a follow-up accuracy
  edge: cosine 0.999840, while `Q4_1` there and `Q4_K` MoE expert
  decode pass.
- Focused coverage green: codebook/generated-dispatch validators, CUDA GEMM
  route/workspace regressions, CUDA graph stochastic smoke, and Qwen3.6 CUDA
  dense prefix/MTP parity.

## Retained Actions

- ROCm: graph-safe sidecar streams, M=2/3/4 VNNI, verifier-row GDN restore,
  compact MoE prefill grids.
- CUDA: verifier-row GDN restore, small-M attention, grouped verifier prefill,
  stream-explicit shared experts, fused split-K/runtime MoE, batched GDN
  projections, router/top-k/scatter work, and IQ4_NL word decode.
- Shared: lowest-id greedy argmax, capped activation arenas, trusted stage
  profiling, synchronized sweep trainers, and non-null GPU stream hard failures.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
CUDA MoE now clears the current no-MTP and MTP anchors on the default lane.
Next targets are CUDA dense prefill/decode gaps, controller stability, and
longer-prompt MoE evidence so the win is not just a short-lane artifact. Dense
prefill is now gate/up and GEMM-heavy; broad GDN split-K remains rejected on
24GB memory grounds, so the next meaningful prefill win needs kernel/fusion
work rather than another selector-only pass.
