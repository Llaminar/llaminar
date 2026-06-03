# Prefix Cache And MTP Benchmark Notes

This is the durable Phase 14 scoreboard. Keep it concise: update the table when
a real benchmark changes the baseline, graph-capture status, or best MTP
speedup. Detailed tuning history belongs in commit messages or perf artifacts.

## Headline Matrix

| Scope | Device | Model | Baseline decode | Best MTP decode | Speedup | Status |
|---|---|---|---:|---:|---:|---|
| Dense long lane, `The quick brown fox`, `-c 64 -n 48` | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 30.76 tok/s | 53.81 tok/s | 1.75x | Correctness green, graph captured, short of 2x target |
| Dense default benchmark, 595 prompt tokens, 128 decode tokens | ROCm `rocm:0` | Qwen3.6 27B Q4_K_S | 29.91 tok/s | 34.96 tok/s | 1.17x | Graph captured, acceptance-limited |
| Dense long lane | CUDA `cuda:0` | Qwen3.6 27B Q4_K_S | 40.44 tok/s | 54.02 tok/s | 1.34x | Correctness green, needs verifier work |
| Dense short lane | CPU `cpu:0` | Qwen3.6 27B Q4_K_S | 5.80 tok/s | 9.50 tok/s | 1.64x | Short smoke only |
| MoE single-device | ROCm `rocm:0` | Qwen3.6 35B A3B | 21.23 tok/s | 10.89 tok/s | 0.51x | Next tuning target |
| MoE single-device | CUDA `cuda:0` | Qwen3.6 35B A3B | 31.20 tok/s | 50.89 tok/s | 1.63x | Needs longer confirmation |
| LocalTP / LocalPP / EP overlay | Mixed | Dense and MoE | Pending | Pending | Pending | Phase 14 after single-device lanes |

## Latest ROCm Dense Evidence

Fresh Release captures on 2026-06-03 used no ROCm tuning env overrides beyond
perf export and logging. Benchmark mode auto-enabled the production prefill
bucket defaults for single-device runs and GPU graphs were left at their code
default.

- Long lane baseline: `/tmp/llaminar-mtp-bench/dense-rocm-final-baseline-c64-n48-bench.json`
- Long lane MTP depth 3: `/tmp/llaminar-mtp-bench/dense-rocm-final-mtp-d3-c64-n48-bench.json`
- Default baseline: `/tmp/llaminar-mtp-bench/dense-rocm-final-default-baseline-bench.json`
- Default MTP depth 3: `/tmp/llaminar-mtp-bench/dense-rocm-final-default-mtp-d3-bench.json`
- Matching perf stats: same prefixes with `-stats.json` and `-stats.csv`.

Long lane MTP telemetry: 86.33% acceptance, 120 accepted tokens, 19 rejected
tokens, 19 rollbacks, 53 verifier runs, and 210 verifier tokens. Structured
stats show fully segmented graph replay for the dense verifier path, sidecar
sample-sync fusion, batched verifier-token sampling, GDN rollback-row shortcuts,
and graph-native small-M ROCm VNNI routes.

Default MTP telemetry: 61.99% acceptance, 212 accepted tokens, 130 rejected
tokens, 130 rollbacks, 170 verifier runs, and 675 verifier tokens. The lower
speedup is mainly acceptance-limited rather than graph-capture-limited.

## Main Tuning Actions Landed

- Stabilized ROCm graph-captured MTP sidecar stream binding and fused sampling
  ordering; regression: `V2_Unit_GraphExecutorCollective` plus the focused
  draft-3 graph-stream stress parity test.
- Added stable verifier graph lifetime handling for all-position logits and
  budget-aware draft-depth clamping.
- Added batched ROCm verifier-row argmax through declared graph workspace.
- Added graph-native M=2/3/4 small-M VNNI routes for Q/K/IQ codebooks, fused
  QKV/GateUp/GDN projections, fused SwiGLU/down, and tiny FP32 alpha/beta.
- Added GDN verifier-row rollback restore for short-conv and recurrence state.
- Moved touched graph scratch paths onto declared workspace buffers.

## Next Work

ROCm dense is speed-positive but not at the 2x Phase 14 target. The next dense
work is reducing captured verifier replay cost and improving acceptance on
general prompts. After that, move to Qwen3.6 MoE MTP tuning on ROCm, then back
through CUDA, CPU, and the multi-participant matrix.
