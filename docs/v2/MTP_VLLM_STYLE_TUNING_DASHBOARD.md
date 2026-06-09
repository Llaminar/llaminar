# vLLM-Style MTP Tuning Dashboard

Scope: SingleDevice Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU. Keep under 5KB;
update every iteration.

Iteration contract: refresh CUDA/ROCm/CPU with no-MTP, fixed d1/d2/d3, and
dynamic. Do not report dynamic without same-run fixed neighbors. Dynamic starts
at d1, may demote to d0, and may probe/promote to d3. Before commit, run broad
units plus relevant parity cells.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or not fully measured. Red = fails or lacks required correctness.

## Matrix

| Backend | Model | Sampling | RAG | Latest speed evidence | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 53.8/53.7/66.7/50.8 vs 44.6 | dynamic too shallow |
| CUDA | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 41.9/33.3/29.6/35.5 vs 44.7 | low acceptance |
| ROCm | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 43.9/33.3/41.8/34.3 vs 31.4 | dynamic conservative |
| ROCm | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 24.8/20.3/22.9/23.9 vs 31.2 | speed-negative |
| CPU | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 5.5/5.7/9.1/5.6 vs 4.6 | dynamic too shallow |
| CPU | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 3.6/3.5/3.6/3.5 vs 4.7 | rollback path slow |
| CUDA | MoE 35B | greedy | Amber | 16tok d3 70.3 vs 109.8; handoff ok | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 48.9/45.0/44.0/53.4 vs 109.7 | zero acceptance |
| ROCm | MoE 35B | greedy | Amber | 16tok d3 42.4 vs 64.7; handoff ok | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 30.4/25.7/23.3/30.6 vs 64.2 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 11.3/11.5/12.7/11.8 vs 17.9 | CPU verifier cost |
| CPU | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 11.9/12.3/12.5/12.3 vs 17.5 | host verifier overhead |

## Latest Evidence

- Fresh full bounded matrix:
  `benchmark_results/mtp_vllm_style/20260609T061226Z-iteration-matrix-6753b5e7/`.
  It covers CUDA/ROCm/CPU, dense/MoE, greedy/stochastic, baseline plus fixed
  d1/d2/d3/dynamic at 16 decode tokens. All rows completed.
- ROCm shared-expert grouped prefill preparation is implemented and regression
  covered. It makes the path graph-native but does not yet make MoE MTP
  speed-positive.
- Dynamic controller tuning: d0 bypass is supported, shifted MTP state is
  maintained during d0 normal decode, demotion is stepwise, d1 only demotes to
  d0 on an all-zero window, and perfect probes can promote early. This removed
  the bad greedy d0 cliff seen in the previous bounded matrix, but dynamic still
  often trails the best fixed depth on short runs.
- Regression/perf: `V2_Unit_MTPDepthController` covers d0 hysteresis.
  `V2_Perf_MTPDepthController` shows policy bookkeeping is tiny: 11-25 ns/op.
- Matrix runner supports bounded sweeps:
  `scripts/run_mtp_iteration_benchmark_matrix.sh --decode-tokens 16 --perfstats`.
  Summary includes verifier/condition/correction timing and replay health.
- CPU dense Prefix/MTP parity harness removed duplicate no-MTP baseline runner
  passes; focused cells now land around 41-43s instead of prior 80-200s costs.
- CPU dynamic dense greedy profile
  `20260609T122350Z-cpu-dense-dynamic-verifier-profile`: 6.56 tok/s; verifier
  4.20s, condition 1.48s, publication 56.8ms. Decode stage time is mostly
  GEMM_FUSED_GATE_UP 30.7%, GEMM 27.2%, GDN_PROJECTION 13.9%, LM_HEAD 12.2%.
  The controller is not the bottleneck.
- CPU dense long-window token parity stops before the known step-114 near-tie:
  `1473` beats PyTorch `48567` by 0.0315 logit; old sampler matches.
- Runner guard: dynamic evidence now requires same-run baseline plus fixed
  d1/d2/d3 unless `--allow-partial-variants` is used for diagnostics.
- Versioned replay plus verifier-stream handoff guards passed focused units and
  CUDA/ROCm MoE d3+stochastic parity. Release d3 check
  `20260609T085703Z-gpu-moe-d3-verifier-stream-handoff`: CUDA 70.3 tok/s,
  ROCm 42.4 tok/s, both 72.7% acceptance, zero rollback/correction replay.
  Handoff counters fired, but verifier/condition time still dominates.
- CPU MoE replay false failure fixed: verifier-base restore and batched LM-head
  all-position rows match serial decode (replay 37.0s, row 27.3s).

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Run the bounded CUDA/ROCm/CPU matrix every iteration: dense/MoE,
   greedy/stochastic, baseline plus fixed d1/d2/d3 and dynamic. Split CPU only
   when needed; record crashes/timeouts explicitly.
2. Use full default matrix for acceptance checkpoints after bounded lanes are
   stable.
3. Attack MoE verifier/catch-up cost; CUDA, ROCm, and CPU MoE are functional
   but speed-negative.
