# Prefix Cache And MTP Benchmark Notes

Durable Phase 14 scoreboard. Keep this file concise: headline numbers, retained
tuning actions, and negative A/B results that should not be rediscovered.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, mostly graph captured, short of 2x |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 46.74 tok/s | 1.56x | Bucketed attention capture, depth-sensitive |
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 tok/s | 53.30 tok/s | 1.31x | Correctness green, depth 1 best |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | ROCm `rocm:0` | Qwen3.6 35B A3B | 19.72 tok/s | 42.04 tok/s | 2.13x | Fixed d1, compact active-expert prefill grid |
| MoE single-device | CUDA `cuda:0` | Qwen3.6 35B A3B | 31.20 tok/s | 50.89 tok/s | 1.63x | Deep math and prefix/MTP parity green; c768/n1 replay smoke green |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | After single-device lanes |

## Adaptive Depth

`MTPDepthController` supports fixed, observe, and dynamic modes. The reusable
sweep script is `scripts/run_mtp_depth_hysteresis_sweep.sh`.

Latest ROCm dense Qwen3.6 27B sweeps, 2026-06-03:

| Case | Policy | Decode | Acceptance | Final depth | Notes |
|---|---|---:|---:|---:|---|
| `qbf_short`, `-n 48` | fixed d3 | 48.29 tok/s | 86.33% | 3 | best fixed |
| `qbf_short`, `-n 48` | dynamic max d3 | 48.36 tok/s | 86.33% | 3 | preserves depth 3 |
| default prompt, `-c 768 -n 64` | fixed d1 | 45.78 tok/s | 90.62% | 1 | best fixed |
| default prompt, `-c 768 -n 64` | dynamic max d3 | 45.09 tok/s | 85.82% | 1 | learns depth 1 |
| code prompts, five-case mean | fixed d1 | 45.61 tok/s | 92.81% | 1 | best fixed |
| code prompts, five-case mean | dynamic max d3 | 44.89 tok/s | 86.35% | 1 | near fixed d1 |

Controller lifetime, not hysteresis thresholds, was the useful tuning change:
`clearCache()` and repeated benchmark prefills no longer reset the controller.

## MoE ROCm Evidence

Qwen3.6 35B A3B on `rocm:0`, default benchmark lane, 2026-06-04:

| Case | Decode | Acceptance | Notes |
|---|---:|---:|---|
| baseline | 19.72 tok/s | n/a | no MTP |
| fixed d1 | 42.04 tok/s | 78.12% | best stable repeat after compact active-expert prefill grid |
| dynamic max d3 | 37.82 tok/s | 71.21% | demotes to depth 1; retest after next router/expert slice |
| fixed d2 | 25.18 tok/s | 69.7% | verifier and rollback cost dominate |
| fixed d3 | 25.90 tok/s | 69.7% | overreaches |

Latest artifact: `benchmark_results/rocm_moe_mtp/20260604T034243Z-64e01724-active-expert-prefill-grid-n64-repeat`.
Stats recorded active grouped-prefill grids with 16 active experts out of 256.

MoE A/B results to avoid repeating:

| Experiment | Decode | Acceptance | Decision |
|---|---:|---:|---|
| exact FP32 router | 33.98 tok/s | 79.69% | retained |
| FP16 router | 27.42 tok/s | 64.84% | reject |
| Q8 router | 33.05 tok/s | 75.78% | reject |
| K-partition router | 30.13 tok/s | 71.88% | reject |
| hipBLAS strided-batched M=2..4 router | 30.92-32.08 tok/s | 69.53-74.22% | parity green but slower; reverted |
| token-dedup grouped-prefill quant | 38.81 tok/s | 74.22% | slower than ratchet; reverted |
| row-batched router / expert `TILE_M=4` | 40.97 / 41.83 tok/s | 77.34-78.91% | slower than 42.04; reverted |

## Retained Tuning Actions

- Stabilized ROCm MTP sidecar stream binding, fused sampling ordering, stable
  verifier graph lifetime, and draft-depth clamping.
- Added explicit non-null GPU stream hard failures; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks, batched
  ROCm verifier-row argmax, and GDN verifier-row rollback restore.
- Kept CUDA verifier projection dispatch on the known-good row-wise route until
  a batched route proves parity and speed; CUDA GEMM context scratch now survives
  request resets while captured prefill graphs are retained.
- Stabilized MoE prefix fingerprints, grouped-decode runtime-table rewarm, and
  streamful `TransferEngine` terminal-state uploads.
- Kept ROCm attention param uploads out of HIP capture; captured attention now
  consumes pre-uploaded workspace params and hard-fails if missing.
- Fused verifier-sized MoE grouping/router work into explicit-stream ROCm kernels
  and compacted grouped-prefill launch grids to active experts only.

## Next Work

MoE ROCm remains the priority: shrink remaining main-verifier router/expert
time and then repeat longer-lane evidence after the 2.13x fixed-depth ratchet.
