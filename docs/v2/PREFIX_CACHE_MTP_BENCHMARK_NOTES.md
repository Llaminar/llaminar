# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 671.37 | 40.82 | fits 24GB |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 582.66 | 53.83 | near fixed d1 |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2465.72 | 109.64 | cuBLAS router prefill default |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1723.72 | 138.67 | latest accept 84.38%, best decode 142.74 |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1823.83 | 142.70 | accept 90.62%, depth 1 |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

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

CUDA MoE latest:
`20260605T043011Z-router-cublas-ab`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2465.72 | 109.64 | n/a |
| fixed d1 MTP | 1723.72 | 138.67 | 84.38% |
| dynamic MTP | 1823.83 | 142.70 | 90.62% |

Fresh checks:
- Graph replay stage stats export trusted GPU-event rows in `stage_gpu`; host
  bookkeeping stays in `forward_graph` (`20260605T024434Z`).
- CUDA runtime-routed MoE decode and shared-expert grouped decode have graph
  replay / benchmark-style parity coverage with perf-counter evidence.
- Dynamic MTP request state resets across benchmark-style `clearCache()`.
- CUDA MoE prefill checks: tile 16 remains best; down prefill decode-hoist
  was neutral/slightly negative (`2146p/109.79d`) and reverted.
- CUDA deterministic scatter now uses block-per-expert chunk scans. nsys
  `20260605T041258Z-nsys-parallel-scatter-skipzero` shows scatter down from
  `67.2ms` to `4.6ms`; no-MTP prefill improved `2156.95 -> 2286.11 tok/s`.
- CUDA MoE FP32 prompt routing now uses cuBLAS SGEMM for `seq_len >= 16`,
  lifting no-MTP prefill above the llama.cpp `llama-bench` anchor while decode
  remains the open no-MTP gap.

Focused gates: `V2_Integration_CUDAMoEKernel`, Qwen3.6 MoE CUDA math parity,
Qwen3.6 dense/MoE CUDA prefix+MTP restore parity, and Qwen3.6 MoE CUDA
benchmark-style parity. ROCm LocalTP prefix smoke currently has an unrelated
segfault breadcrumb from this run.

## Retained Actions

- ROCm: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN restore, compact active-expert MoE prefill grids.
- CUDA: verifier-row GDN restore, graph-capturable small-M attention, grouped
  verifier prefill, stream-explicit shared experts, fused split-K/runtime MoE,
  parallel router/top-k/scatter, source-token activation quantization, mapped
  outputs, checkpoint elision, batched GDN projections, and cuBLAS router
  prefill.
- Shared: lowest-id greedy argmax tie-break, capped activation arenas, trusted
  stage profiling split, and non-null GPU stream hard failures.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
Dense decode is close. CUDA MoE no-MTP prefill and dynamic-MTP decode now edge
the llama.cpp anchors; no-MTP decode still trails `118.26 tok/s`. Next targets
are controller stability plus the remaining CUDA MoE decode and verifier costs.
