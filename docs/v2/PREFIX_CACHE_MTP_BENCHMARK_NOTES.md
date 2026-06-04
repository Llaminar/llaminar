# Prefix Cache And MTP Benchmark Notes

Durable Phase 14 scoreboard: headline numbers, retained actions, and live gaps.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, mostly graph captured, short of 2x |
| Dense default benchmark, 595 prompt tokens, 128 decode | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 46.74 tok/s | 1.56x | Bucketed attention capture, depth-sensitive |
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 tok/s | 53.30 tok/s | 1.31x | Correctness green, depth 1 best |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | ROCm `rocm:0` | Qwen3.6 35B A3B | 19.72 tok/s | 42.04 tok/s | 2.13x | Fixed d1, compact active-expert prefill grid |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | CUDA `cuda:0` | Qwen3.6 35B A3B | 102.03 tok/s | 83.10 tok/s | 0.81x | Verifier-row shortcut restored; correctness green; llama.cpp north star is 118.31 tok/s |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | After single-device lanes |

llama.cpp CUDA north star: `ggml-org/llama.cpp@6ddc943`, `llama-bench -p 768 -n 64 -ngl 999 -r 3`:

| Model | Prefill | Decode | File |
|---|---:|---:|---|
| Dense 27B | 1161.19 tok/s | 41.82 tok/s | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/dense.jsonl` |
| MoE 35B A3B | 2415.25 tok/s | 118.31 tok/s | `benchmark_results/llama_cpp_cuda/20260604T080645Z-6ddc943/moe.jsonl` |

Clean Llaminar CUDA no-MTP after prefill-reset safety: dense 671.87/40.42
and latest MoE refresh 891.86/102.03 tok/s. Request-boundary `clear_cache()` invalidates
monolithic prefill graph entries; recover graph-safe prefill reuse next.

## Adaptive Depth

`MTPDepthController` supports fixed, observe, and dynamic modes. Reusable sweep:
`scripts/run_mtp_depth_hysteresis_sweep.sh`. Latest ROCm dense sweeps preserve
depth 3 on `qbf_short` and learn depth 1 on default/code prompts.

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

Avoid redoing rejected ROCm A/Bs: FP16/Q8/K-partition routers, hipBLAS router,
token-dedup grouped quant, and row-batched router/expert `TILE_M=4`.

## MoE CUDA Evidence

Qwen3.6 35B A3B on `cuda:0`, default benchmark lane, 2026-06-04:

| Case | Decode | Acceptance | Notes |
|---|---:|---:|---|
| baseline | 103.84 tok/s | n/a | prior no-MTP ratchet after prefill reset safety |
| baseline refresh after shortcut fix | 102.03 tok/s | n/a | `benchmark_results/cuda_moe_mtp/20260604T131731Z-release-shortcut-fixed-baseline-n64` |
| fixed d1, verifier-row shortcut restored | 83.10 tok/s | 75.78% | corrected release token path; `benchmark_results/cuda_moe_mtp/20260604T131657Z-release-shortcut-fixed-default-n64` |
| fixed d1, CUDA small-M grouped prefill tile | 65.93 tok/s | 39.84% | previous ratchet; auto `TILE_M=2/4`; `benchmark_results/cuda_moe_mtp/20260604T090428Z-smallm-prefill-tile-auto` |
| fixed d1, CUDA GDN qkv+z fused | 64.72 tok/s | 42.97% | qkv+z native subgroup at M=1/2 |
| opt-in shared grouped FFN | 64.26 tok/s | 40.62% | n64 no ratchet; GPU-stage n16 improved 48.17 -> 51.65 tok/s |

## Retained Tuning Actions

- Stabilized ROCm sidecar streams, sampling ordering, verifier graph lifetime, and depth clamping.
- Added explicit non-null GPU stream hard failures; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks, batched verifier-row argmax, and GDN verifier-row rollback restore.
- CUDA quantized GEMM now advertises fused projection support; Qwen3.6 MoE GDN qkv+z uses a native subgroup while alpha/beta remain FP32 single projections.
- CUDA MoE grouped prefill auto-selects verifier-sized `TILE_M=2/4`.
- CUDA verifier-row rollback now uses the GDN row-restore shortcut in production release builds; the release-only stale-token root cause was MoE runtime-table decode consuming a bank initialized after routing for the same pass.
- CUDA shared-expert decode has an opt-in graph-capturable grouped table path; keep opt-in until n64 improves.
- Added graph-capturable CUDA FP16KV small-M verifier attention for M=2..4.
- Request-boundary prefill graph replay now invalidates on session reset after a CUDA padded-bucket crash; intra-request chunk capture/replay remains covered.
- Stabilized MoE prefix fingerprints, grouped-decode rewarm, and streamful terminal-state uploads.
- Kept ROCm attention params out of HIP capture and compacted MoE grouped-prefill grids to active experts only.

## Next Work

CUDA MoE MTP is correctness-green and the verifier-row shortcut lifted the n64
lane to 83.10 tok/s, but it remains performance-negative versus the 102.03 tok/s
no-MTP refresh; next is MoE verifier economics after the shortcut. ROCm MoE keeps
the 2.13x ratchet.
