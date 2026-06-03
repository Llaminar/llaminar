# Prefix Cache And MTP Benchmark Notes

This is the durable Phase 14 scoreboard. Keep it concise: update the table when
a real benchmark changes the baseline, graph-capture status, or best MTP
speedup. Detailed tuning history belongs in commit messages or perf artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, graph captured, short of 2x target |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 39.72 tok/s | 1.33x | Graph captured, depth-sensitive |
| Dense long lane | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.44 tok/s | 54.02 tok/s | 1.34x | Correctness green, needs verifier work |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE single-device | ROCm `rocm:0` | Qwen3.6 35B A3B | 21.23 tok/s | 10.89 tok/s | 0.51x | Next tuning target |
| MoE single-device | CUDA `cuda:0` | Qwen3.6 35B A3B | 31.20 tok/s | 50.89 tok/s | 1.63x | Needs longer confirmation |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | Phase 14 after single-device lanes |

## Latest ROCm Dense Evidence

Fresh Release captures on 2026-06-03 used code-default GPU graphs and production
prefill buckets. The long lane shows graph-captured depth-3 MTP is speed-positive
but still short of the 2x target. The default benchmark prompt remains
acceptance-limited: depth 3 drops to about 62-63% acceptance, so adaptive depth is
needed instead of hard-pinning the deepest draft.

## Adaptive Depth Motivation

`MTPDepthController` is now implemented with fixed, observe, and dynamic modes.
The first reusable hysteresis sweep is
`scripts/run_mtp_depth_hysteresis_sweep.sh`.

Latest ROCm release sweep, Qwen3.6 27B Q4_K_S on `rocm:0`, 2026-06-03:

| Case | Depth policy | Decode | Acceptance | Final depth | Updates | Notes |
|---|---|---:|---:|---:|---:|---|
| `qbf_short` | fixed d1 | 45.52 tok/s | 89.58% | 1 | 0 | reference |
| `qbf_short` | fixed d3 | 43.76 tok/s | 85.61% | 3 | 0 | reference |
| `qbf_short` | dynamic max d3 | 49.08 tok/s | 85.61% | 3 | 0 | stayed deep |
| default benchmark prompt | fixed d1 | 35.58 tok/s | 69.92% | 1 | 0 | best fixed in this sweep |
| default benchmark prompt | fixed d3 | 31.50 tok/s | 63.48% | 3 | 0 | overreaches |
| default benchmark prompt | dynamic max d3 | 33.50 tok/s | 67.53% | 1 | 8 | demotes, but pays learning cost |
| `qbf_long` | fixed d1 | 47.49 tok/s | 92.97% | 1 | 0 | best fixed in this sweep |
| `qbf_long` | fixed d3 | 40.83 tok/s | 78.15% | 3 | 0 | overreaches here |
| `qbf_long` | dynamic max d3 | 38.75 tok/s | 78.79% | 2 | 5 | stable after stream fix |

Artifacts live under
`/tmp/llaminar-mtp-bench/adaptive-depth-20260603/`. Dynamic mode now makes the
right qualitative choices: it stays deep for high-acceptance short prompts and
demotes acceptance-limited prompts. Whole-run speed on the default prompt still
trails fixed d1 because the controller starts at max depth and spends early
windows learning; tuning initial depth/window policy is the next adaptive slice.

## Main Tuning Actions Landed

- Stabilized ROCm graph-captured MTP sidecar stream binding and fused sampling
  ordering; regression: `V2_Unit_GraphExecutorCollective` plus the focused
  draft-3 graph-stream stress parity test.
- Added hard failures for GPU stage execution, MTP deferred sampling, and greedy
  GPU sampling when an explicit non-null stream is unavailable; regression:
  `V2_Unit_Static_NoDefaultStreamInGPUCode`.
- Added stable verifier graph lifetime handling for all-position logits and
  budget-aware draft-depth clamping.
- Added batched ROCm verifier-row argmax through declared graph workspace.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks, fused
  QKV/GateUp/GDN projections, fused SwiGLU/down, and tiny FP32 alpha/beta.
- Added GDN verifier-row rollback restore for short-conv and recurrence state.
- Moved touched graph scratch paths onto declared workspace buffers.

## Next Work

Tune adaptive initial/window policy, then move to Qwen3.6 MoE MTP on ROCm before
returning through CUDA, CPU, and the multi-participant matrix.
