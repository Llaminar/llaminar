# Prefix Cache And MTP Benchmark Notes

Durable Phase 14 scoreboard: headline numbers, retained actions, and negative
A/Bs that should not be rediscovered.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, mostly graph captured, short of 2x |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 46.74 tok/s | 1.56x | Bucketed attention capture, depth-sensitive |
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 tok/s | 53.30 tok/s | 1.31x | Correctness green, depth 1 best |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | ROCm `rocm:0` | Qwen3.6 35B A3B | 19.72 tok/s | 42.04 tok/s | 2.13x | Fixed d1, compact active-expert prefill grid |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | CUDA `cuda:0` | Qwen3.6 35B A3B | 101.37 tok/s | 64.72 tok/s | 0.64x | Correctness green; llama.cpp north star is 118.31 tok/s |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | After single-device lanes |

llama.cpp CUDA: `ggml-org/llama.cpp@6ddc943`, `GGML_CUDA=ON`, SM86 RTX 3090,
`llama-bench -p 768 -n 64 -ngl 999 -r 3`:

| Model | Prefill | Decode | File |
|---|---:|---:|---|
| Dense 27B | 1161.19 tok/s | 41.82 tok/s | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/dense.jsonl` |
| MoE 35B A3B | 2415.25 tok/s | 118.31 tok/s | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/moe.jsonl` |

## Adaptive Depth

`MTPDepthController` supports fixed, observe, and dynamic modes. The reusable
sweep script is `scripts/run_mtp_depth_hysteresis_sweep.sh`.

Latest ROCm dense sweeps showed dynamic preserves depth 3 on `qbf_short`
(48.36 tok/s, 86.33%) and learns depth 1 on default/code prompts. The useful
tuning change was controller lifetime across `clearCache()` and repeated
benchmark prefills.

## MoE ROCm Evidence

Qwen3.6 35B A3B on `rocm:0`, default benchmark lane, 2026-06-04:

| Case | Decode | Acceptance | Notes |
|---|---:|---:|---|
| baseline | 19.72 tok/s | n/a | no MTP |
| fixed d1 | 42.04 tok/s | 78.12% | best stable repeat after compact active-expert prefill grid |
| dynamic max d3 | 37.82 tok/s | 71.21% | demotes to depth 1; retest after next router/expert slice |
| fixed d2 | 25.18 tok/s | 69.7% | verifier and rollback cost dominate |
| fixed d3 | 25.90 tok/s | 69.7% | overreaches |

Artifact: `benchmark_results/rocm_moe_mtp/20260604T034243Z-64e01724-active-expert-prefill-grid-n64-repeat`.
Stats showed 16 active experts out of 256.

MoE ROCm A/Bs not to repeat: FP16/Q8/K-partition routers, hipBLAS M=2..4
router, token-dedup grouped quant, and row-batched router/expert `TILE_M=4`
all passed correctness but missed the retained 42.04 tok/s ratchet.

## MoE CUDA Evidence

Qwen3.6 35B A3B on `cuda:0`, default benchmark lane, 2026-06-04:

| Case | Decode | Acceptance | Notes |
|---|---:|---:|---|
| baseline | 101.37 tok/s | n/a | current no-MTP ratchet |
| fixed d1, CUDA GDN qkv+z fused | 64.72 tok/s | 42.97% | `kernel.gdn_projection_route`: qkv+z native subgroup at M=1/2 |
| opt-in shared grouped FFN | 64.26 tok/s | 40.62% | n64 no ratchet; GPU-stage n16 improved 48.17 -> 51.65 tok/s |
| fixed d1, after small-M attention | 41.92 tok/s | 32.03% | attention no longer uses FA2 prefill path |

Artifacts: `benchmark_results/cuda_moe_mtp/20260604T072857Z-bffe6cc7-gdn-qkvz-fused-n64`,
`20260604T074312Z-20a0cfed-shared-grouped-cuda-n64`,
`20260604T074809Z-b5cee090-shared-grouped-cuda-gpustage-n16`.

## Retained Tuning Actions

- Stabilized ROCm sidecar streams, fused sampling ordering, verifier graph lifetime, and depth clamping.
- Added explicit non-null GPU stream hard failures; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks, batched verifier-row argmax, and GDN verifier-row rollback restore.
- CUDA quantized GEMM now advertises fused projection support; Qwen3.6 MoE GDN qkv+z uses a native subgroup while alpha/beta remain FP32 single projections.
- CUDA shared-expert decode has an opt-in graph-capturable grouped table path; keep opt-in until n64 improves.
- Added graph-capturable CUDA FP16KV small-M verifier attention for M=2..4.
- Stabilized MoE prefix fingerprints, grouped-decode runtime-table rewarm, and streamful `TransferEngine` terminal-state uploads.
- Kept ROCm attention params out of HIP capture and compacted MoE grouped-prefill grids to active experts only.

## Next Work

CUDA MoE MTP is correctness-green but performance-negative; next work is MoE
verifier economics after attention. ROCm MoE keeps the 2.13x ratchet.
