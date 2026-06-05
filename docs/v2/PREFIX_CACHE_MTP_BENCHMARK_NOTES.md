# Prefix Cache And MTP Benchmark Notes

Phase 14 scoreboard: latest CUDA/ROCm evidence and current gaps. Raw tuning
history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill | Decode | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 702.93 | 40.82 | prefill improved |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP | 601.81 | 56.13 | accept 96.88%, beats l.cpp d1 |
| Dense default, 595p/64d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 601.02 | 55.23 | depth 1, near fixed d1 |
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
`20260605T075427Z-qwen36-q4k-prefill-tile-override`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 702.93 | 40.82 | n/a |
| fixed d1 MTP | 601.81 | 56.13 | 96.88% |
| dynamic MTP | 601.02 | 55.23 | 96.88% |

CUDA MoE latest:
`20260605T070628Z-iq4nl-word-decode`.

| Case | Prefill | Decode | Acceptance |
|---|---:|---:|---:|
| no MTP | 2707.70 | 119.91 | n/a |
| fixed d1 MTP | 1946.82 | 148.50 | 71.88% |
| dynamic MTP | 1943.69 | 145.36 | 68.75% |

Fresh checks:
- Graph replay stage stats export trusted GPU-event `stage_gpu` rows; fused
  CUDA MoE decode also exports capture-safe `kernel_cuda` sub-kernel rows.
- The sub-kernel export identified IQ4_NL down decode as the next hot path.
  CUDA IQ4_NL decode now reuses the packed word decoder in the shared VNNI
  helper, lifting MoE no-MTP and MTP decode beyond the llama.cpp anchors.
- CUDA dense Q4_K prompt-prefill now uses sweep-selected tiles for the Qwen3.6
  FFN gate/up, FFN down, and GDN inner projection M-bin 512 shapes.
- Focused coverage stayed green for CUDA GEMM IQ4_NL parity, CUDA MoE kernel
  graph replay, Qwen3.6 MoE CUDA math decode parity, verifier-row shortcut
  parity, dynamic-depth request reset parity, fused verifier prefill parity,
  skip-gather greedy parity, prefix-cache plus MTP restore parity, and the
  focused Qwen3.6 CUDA dense SingleDevice prefix/MTP parity suite.

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
  prefill, fused-runtime MoE down warp-reduce, and IQ4_NL word decode.
- Shared: lowest-id greedy argmax tie-break, capped activation arenas, trusted
  stage profiling split, and non-null GPU stream hard failures.

## Next Work

Beat llama.cpp CUDA on dense and MoE, prefill and decode, with MTP on/off.
CUDA MoE now clears the current no-MTP and MTP anchors on the default lane.
Next targets are CUDA dense prefill/decode gaps, controller stability, and
longer-prompt MoE evidence so the win is not just a short-lane artifact.
