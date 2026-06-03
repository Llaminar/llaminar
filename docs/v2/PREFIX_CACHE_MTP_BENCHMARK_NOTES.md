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

Fresh Release captures on 2026-06-03 used code-default GPU graphs and production
prefill buckets. The long lane shows graph-captured depth-3 MTP is speed-positive
but still short of the 2x target. The default benchmark prompt remains
acceptance-limited: depth 3 drops to about 62-63% acceptance, so adaptive depth is
needed instead of hard-pinning the deepest draft.

## Adaptive Depth Motivation

`MTPDepthController` is now implemented with fixed, observe, and dynamic modes.
The first reusable hysteresis sweep is
`scripts/run_mtp_depth_hysteresis_sweep.sh`.

Latest ROCm release sweep after early-demotion tuning, Qwen3.6 27B Q4_K_S on
`rocm:0`, 2026-06-03:

| Case | Depth policy | Decode | Acceptance | Final depth | Updates | Notes |
|---|---|---:|---:|---:|---:|---|
| `qbf_short` | fixed d1 | 48.22 tok/s | 92.71% | 1 | 0 | reference |
| `qbf_short` | fixed d3 | 52.91 tok/s | 85.61% | 3 | 0 | best fixed |
| `qbf_short` | dynamic max d3 | 52.70 tok/s | 87.41% | 3 | 1 | tied with d3 |
| default benchmark prompt | fixed d1 | 36.36 tok/s | 71.88% | 1 | 0 | best fixed |
| default benchmark prompt | fixed d3 | 33.62 tok/s | 60.76% | 3 | 0 | overreaches |
| default benchmark prompt | dynamic max d3 | 37.81 tok/s | 73.45% | 1 | 5 | beats fixed in this sweep |
| `qbf_long` | fixed d1 | 47.49 tok/s | 92.97% | 1 | 0 | best fixed in this sweep |
| `qbf_long` | fixed d3 | 40.83 tok/s | 78.15% | 3 | 0 | overreaches here |
| `qbf_long` | dynamic max d3 | 41.50 tok/s | 79.30% | 1 | 9 | improved, still trails d1 |

Artifacts live under
`/tmp/llaminar-mtp-bench/adaptive-depth-20260603/`. Dynamic mode now evaluates
bad partial windows after `min_samples`, so acceptance-limited prompts demote
before a full window elapses. The default prompt improved from the previous
33.50 tok/s dynamic capture to 37.81 tok/s. qbf-long still trails fixed d1,
mostly because occasional perfect shallow windows can promote and then demote
again; promotion hysteresis is the next adaptive slice.

## ROCm Graph Replay Safety

A long dynamic-depth QBF run faulted during cached HIP graph replay of
`layer0_attention`. ROCm decode attention now remains graph-capturable while
reporting a launch-topology variant signature keyed to split-count buckets.
Segmented replay warms and recaptures when the bucket changes, while ordinary
within-bucket KV growth uses device-side dynamic params.

Validation: `V2_Unit_AttentionComputeStage_DynamicKVLen` pins the bucket
signature, `V2_Unit_ForwardGraphTypes` pins variant recapture, and
`FlashDecode_FP32_GraphReplayUsesUpdatedKVLenWithinBucket` proves graph replay
uses updated device-side KV length. The old one-stage trace shows
`layer0_attention` as `[GRAPH]`; graph-stream stress, dynamic MTP parity, and
prefix+dynamic restore passed. The original crash repro now completes at 46.74
tok/s, 85.67% acceptance, and no new `gpucore.*`.

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
