# Prefix Cache And MTP Benchmark Notes

This is the durable Phase 14 scoreboard. Keep it concise: update the table when
a real benchmark changes the baseline, graph-capture status, or best MTP
speedup. Detailed tuning history belongs in commit messages or perf artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, mostly graph captured, short of 2x target |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 46.74 tok/s | 1.56x | Bucketed attention graph capture, depth-sensitive |
| Dense long lane | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.44 tok/s | 54.02 tok/s | 1.34x | Correctness green, needs verifier work |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE single-device | ROCm `rocm:0` | Qwen3.6 35B A3B | 21.23 tok/s | 10.89 tok/s | 0.51x | Next tuning target |
| MoE single-device | CUDA `cuda:0` | Qwen3.6 35B A3B | 31.20 tok/s | 50.89 tok/s | 1.63x | Needs longer confirmation |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | Phase 14 after single-device lanes |

## Latest ROCm Dense Evidence

Fresh Release captures on 2026-06-03 used code-default GPU graphs. Depth-3 MTP
is speed-positive but still short of the 2x target. The default prompt remains
acceptance-limited, so adaptive depth is needed instead of hard-pinning the
deepest draft.

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

Artifacts live under
`benchmark_results/mtp_depth_hysteresis/20260603T171951Z-b526c702`,
`20260603T173035Z-b526c702`, and `20260603T173202Z-b526c702`.
The useful tuning change was lifecycle, not another threshold tweak:
`clearCache()` and repeated benchmark prefills no longer reset the controller,
so dynamic mode keeps the running tally it needs. No further hysteresis change
is required for this slice; promotion tuning can wait until we see a lane where
fixed depth 3 clearly wins but dynamic fails to climb.

## ROCm Attention Param-Copy Safety

The ROCm attention crash was traced to dynamic `AttentionDeviceParams` H2D
copies being recorded inside HIP graph capture. The fix moves that upload into
`prepareDynamicAttnParams()` on an explicit non-null stream before capture or
replay; the captured stage body now hard-fails if params were not already
uploaded. Cached forward graphs and MTP sidecar graphs bind all stages to the
worker/capture stream before dynamic-param updates.

Validation: the focused graph/attention/forward units passed. A short real
Qwen3.6 ROCm dynamic-MTP run completed with 268 MTP graph replay traces, no
ERROR/WARN param-copy diagnostics, 36.64 decode tok/s, and 65.85% acceptance.

## Main Tuning Actions Landed

- Stabilized ROCm MTP sidecar stream binding and fused sampling ordering.
- Added hard failures for GPU stage execution, MTP deferred sampling, and greedy
  GPU sampling when an explicit non-null stream is unavailable; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added stable verifier graph lifetime handling and draft-depth clamping.
- Added batched ROCm verifier-row argmax through declared graph workspace.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks.
- Added GDN verifier-row rollback restore for short-conv and recurrence state.
- Moved touched graph scratch paths onto declared workspace buffers.

## Next Work

Tune promotion hysteresis, then move to Qwen3.6 MoE MTP on ROCm before returning
through CUDA, CPU, and the multi-participant matrix.
