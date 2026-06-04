# Prefix Cache And MTP Benchmark Notes

This is the durable Phase 14 scoreboard. Keep it concise: update the table when
a real benchmark changes the baseline, graph-capture status, or best MTP
speedup. Detailed tuning history belongs in commit messages or perf artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, mostly graph captured, short of 2x target |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 46.74 tok/s | 1.56x | Bucketed attention graph capture, depth-sensitive |
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.75 tok/s | 53.30 tok/s | 1.31x | Correctness green, depth 1 best |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE default lane, 595 prompt tokens, `-c 768 -n 64` | ROCm `rocm:0` | Qwen3.6 35B A3B | 15.76 tok/s | 26.66 tok/s | 1.69x | Dynamic best; d3 overreaches |
| MoE single-device | CUDA `cuda:0` | Qwen3.6 35B A3B | 31.20 tok/s | 50.89 tok/s | 1.63x | Needs longer confirmation |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | Phase 14 after single-device lanes |

## Adaptive Depth Motivation

`MTPDepthController` is now implemented with fixed, observe, and dynamic modes.
The first reusable hysteresis sweep is
`scripts/run_mtp_depth_hysteresis_sweep.sh`.

Latest ROCm release sweeps after the controller lifetime fix, Qwen3.6 27B
Q4_K_S on `rocm:0`, 2026-06-03:

| Case | Depth policy | Decode | Acceptance | Final depth | Updates | Notes |
|---|---|---:|---:|---:|---:|---|
| `qbf_short`, `-n 48` | fixed d1 | 45.96 tok/s | 90.62% | 1 | 0 | shallow reference |
| `qbf_short`, `-n 48` | fixed d3 | 48.29 tok/s | 86.33% | 3 | 0 | best fixed |
| `qbf_short`, `-n 48` | dynamic max d3 | 48.36 tok/s | 86.33% | 3 | 0 | preserves depth 3 |
| default prompt, `-c 768 -n 64` | fixed d1 | 45.78 tok/s | 90.62% | 1 | 0 | best fixed |
| default prompt, `-c 768 -n 64` | fixed d3 | 31.50 tok/s | 63.53% | 3 | 0 | overreaches |
| default prompt, `-c 768 -n 64` | dynamic max d3 | 45.09 tok/s | 85.82% | 1 | 2 | learns depth 1 |
| code prompts, five-case mean | fixed d1 | 45.61 tok/s | 92.81% | 1 | 0 | best fixed |
| code prompts, five-case mean | fixed d3 | 35.52 tok/s | 72.98% | 3 | 0 | overreaches |
| code prompts, five-case mean | dynamic max d3 | 44.89 tok/s | 86.35% | 1 | 2 | near fixed d1 |

Artifacts live under `benchmark_results/mtp_depth_hysteresis/20260603T171951Z-b526c702`,
`20260603T173035Z-b526c702`, and `20260603T173202Z-b526c702`.
The useful tuning change was lifecycle, not another threshold tweak:
`clearCache()` and repeated benchmark prefills no longer reset the controller,
so dynamic mode keeps the running tally it needs. No further hysteresis change
is required for this slice.

The reusable sweep script defaults to production benchmark mode. Use
`--deterministic` only when the benchmark is meant to mirror deterministic
parity settings.

## Latest CUDA Dense Evidence

Production-mode Qwen3.6 27B Q4_K_S on `cuda:0`, 2026-06-04:

| Case | Decode | Acceptance | Notes |
|---|---:|---:|---|
| baseline | 40.75 tok/s | n/a | no MTP |
| fixed d1 | 53.30 tok/s | 95.83% | best CUDA lane |
| fixed d2 | 33.20 tok/s | 93.55% | verifier cost dominates |
| fixed d3 | 31.29 tok/s | 91.43% | slower than d1 |
| dynamic max d3 | 31.32 tok/s | 91.43% | no update before short run ends |

Artifacts:
`benchmark_results/cuda_dense_mtp/20260604T003037Z-27115c81-qbf-fast-prodauto-final`.
An attempted CUDA M=3/4 chunked small-M route passed focused parity but
regressed deeper-draft throughput, so it is not retained.

## Main Tuning Actions Landed

- Stabilized ROCm MTP sidecar stream binding and fused sampling ordering.
- Added hard failures for GPU stage execution, MTP deferred sampling, and greedy
  GPU sampling when an explicit non-null stream is unavailable; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added stable verifier graph lifetime handling and draft-depth clamping.
- Added batched ROCm verifier-row argmax through declared graph workspace.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks.
- Kept CUDA M=3/4 verifier projection dispatch on the known-good row-wise route
  until a batched route proves both parity and speed.
- Added GDN verifier-row rollback restore for short-conv and recurrence state.
- Stabilized MoE prefix fingerprints by excluding transient runtime-table caches.
- Moved touched graph scratch paths onto declared workspace buffers.

## Next Work

Tune Qwen3.6 MoE ROCm toward the 1.5x target at longer context, then return
through CUDA, CPU, and the multi-participant matrix.
