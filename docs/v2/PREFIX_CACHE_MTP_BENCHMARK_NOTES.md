# Prefix Cache And MTP Benchmark Notes

Durable Phase 14 scoreboard: latest headline numbers, evidence artifacts, and
live gaps. Keep this file concise; rejected tuning history belongs in artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `qbf`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 | 53.81 | 1.75x | Correctness green, short of 2x |
| Dense default benchmark, 595p/128d | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 | 46.74 | 1.56x | Depth-sensitive |
| Dense long lane, `qbf`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 | 53.30 | 1.31x | Depth 1 best |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 | 9.50 | 1.64x | Short smoke only |
| MoE default lane, 595p/64d | ROCm `rocm:0` | Qwen3.6 35B A3B | 19.72 | 42.04 | 2.13x | Fixed d1, ratcheted |
| MoE default lane, 595p/64d | CUDA `cuda:0` | Qwen3.6 35B A3B | 102.09 | 87.62 | 0.86x | Correct, still perf-negative |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | After single-device lanes |

llama.cpp CUDA north star, `ggml-org/llama.cpp@6ddc943`,
`llama-bench -p 768 -n 64 -ngl 999 -r 3`:

| Model | Prefill | Decode | Artifact |
|---|---:|---:|---|
| Dense 27B | 1161.19 | 41.82 | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/dense.jsonl` |
| MoE 35B A3B | 2415.25 | 118.31 | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/moe.jsonl` |

## MoE ROCm Evidence

Qwen3.6 35B A3B on `rocm:0`, default benchmark lane:

| Case | Decode | Acceptance | Artifact |
|---|---:|---:|---|
| baseline | 19.72 | n/a | `benchmark_results/rocm_moe_mtp/20260604T034243Z-64e01724-active-expert-prefill-grid-n64-repeat` |
| fixed d1 | 42.04 | 78.12% | same |
| dynamic max d3 | 37.82 | 71.21% | same |

## MoE CUDA Evidence

Qwen3.6 35B A3B on `cuda:0`, default benchmark lane:

| Case | Decode | Acceptance | Artifact |
|---|---:|---:|---|
| latest baseline | 102.09 | n/a | `benchmark_results/cuda_moe_mtp/20260604T144911Z-post-fused-path-regression-refresh/baseline.json` |
| best fixed d1 | 87.62 | 84.38% | `benchmark_results/cuda_moe_mtp/20260604T134433Z-fused-swiglu-scratch-fixed-n64` |
| latest fixed d1 | 84.97 | 75.78% | `benchmark_results/cuda_moe_mtp/20260604T144911Z-post-fused-path-regression-refresh/mtp_d1.json` |
| latest fixed d3 | 68.57 | 66.86% | `benchmark_results/cuda_moe_mtp/20260604T144911Z-post-fused-path-regression-refresh/mtp_d3_after_suffix_commit_fix.json` |

## Retained Actions

- ROCm dense/MoE: graph-safe sidecar streams, sampling ordering, verifier lifetime,
  depth clamping, M=2/3/4 VNNI routes, verifier-row GDN rollback restore, and compact
  active-expert MoE prefill grids.
- CUDA dense/MoE: verifier-row GDN restore is enabled in release, small-M verifier
  attention is graph-capturable, grouped MoE prefill auto-selects small verifier tiles,
  and the MoE GDN qkv+z path uses fused native projections.
- CUDA grouped MoE prefill keeps fused SwiGLU+quant but uses independent SwiGLU scratch,
  so the fused epilogue no longer overwrites gate/up inputs before reading them.
- CUDA verifier-sized MoE prefill now emits fused/split branch counters; M=2/3/4
  integration coverage asserts default fused execution, split-path equivalence, and capture.
- CUDA MoE MTP depth-3 parity now uses the stable benchmark-prompt lane, and shared
  expert gate verification allows mathematically valid sigmoid-underflow zero rows.
- Depth>1 MTP rollback now commits suffix shifted-cache rows after an already-committed
  verifier prefix; regressions cover tiny graph construction and CUDA MoE depth-3 parity.
- CUDA and ROCm GPU greedy argmax tie-break to the lowest token id, matching CPU greedy.
- CUDA Qwen3.6 MoE production sampling is guarded by same-step gathered-logits argmax
  parity instead of an unstable independent-run near-tie token oracle.
- Explicit non-null GPU stream hard failures remain required; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.

## Next Work

CUDA MoE MTP is correctness-green but still slower than baseline. Next target is MoE
verifier economics and sidecar cost on CUDA before moving to non-single-device lanes.
ROCm MoE keeps the current 2.13x ratchet.
