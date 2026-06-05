# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 709.61 | 40.80 | prefill improved |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP | 606.46 | 54.32 | accept 96.88%, below prior decode |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 607.03 | 56.15 | depth 1, beats l.cpp d1 |
| Dense long `qbf`, `-c64 -n48` | CUDA | Qwen3.6 27B Q4_K_S | best MTP | n/a | 53.30 | depth 1 best |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | beats l.cpp no-MTP |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | accept 71.88%, beats l.cpp d1 |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1943.69 | 145.36 | accept 68.75%, depth 1 |
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

CUDA dense latest:
`20260605T090657Z-qwen36-dense-gdn-split1-overrides`; route sweeps:
`20260605T083341Z-qwen36-gdn-m600-tile-sweep`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 709.61 | 40.80 | n/a |
| fixed d1 MTP | 606.46 | 54.32 | 96.88% |
| dynamic MTP | 607.03 | 56.15 | 96.88% |

CUDA MoE latest:
`20260605T070628Z-iq4nl-word-decode`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2707.70 | 119.91 | n/a |
| fixed d1 MTP | 1946.82 | 148.50 | 71.88% |
| dynamic MTP | 1943.69 | 145.36 | 68.75% |

Fresh checks:
- Trusted `stage_gpu` graph-replay timing is now exportable without legacy
  profiling, and CUDA NativeVNNI prompt-prefill emits structured
  `kernel.cuda_native_vnni_prefill_calls` route counters.
- CUDA NativeVNNI scratch, CUDA/ROCm MoE request scratch, CUDA KV gather, GDN
  deinterleave scratch, and CUDA attention decode partials now bind through
  `DeviceWorkspaceManager`; workspace suballocs emit structured memory counters.
  The old CUDA cuBLAS attention fallback/env knob was removed rather than
  preserving hidden O(seq^2) `cudaMalloc` state.
- Dense diagnostic at graph bucket `M=600` recorded top prompt-prefill routes:
  `17408x5120` Q4_K tile 4, `5120x17408` Q4_K tile 2, and GDN projection
  shapes including `6144x5120`, `5120x6144`, `10240x5120`, and `1024x5120`.
- GDN Z/output prefill now use split-1 `T64x128_w4x2` overrides for Q4_K/Q5_1.
  The faster split-K 2/4 sweep winners were rejected because their partial
  buffers caused a 24GB dense warmup OOM.
- Focused coverage stayed green for CUDA GEMM parity, CUDA MoE graph replay,
  Qwen3.6 MoE CUDA math parity, verifier-row shortcut parity, and the focused
  Qwen3.6 CUDA dense SingleDevice prefix/MTP parity suite.

## Retained Actions

- ROCm: graph-safe sidecar streams, depth clamping, M=2/3/4 VNNI,
  verifier-row GDN restore, compact active-expert MoE prefill grids.
- CUDA: verifier-row GDN restore, graph-capturable small-M attention, grouped
  verifier prefill, stream-explicit shared experts, fused split-K/runtime MoE,
  parallel router/top-k/scatter, source-token activation quantization, mapped
  outputs, checkpoint elision, batched GDN projections, and cuBLAS router
  prefill, fused-runtime MoE down warp-reduce, and IQ4_NL word decode.
- Shared: lowest-id greedy argmax tie-break, capped activation arenas, trusted
  stage profiling split, and non-null GPU stream hard failures.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
CUDA MoE now clears the current no-MTP and MTP anchors on the default lane.
Next targets are CUDA dense prefill/decode gaps, controller stability, and
longer-prompt MoE evidence so the win is not just a short-lane artifact.
