# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

Iteration contract: refresh the same CUDA/ROCm/CPU matrix with no-MTP baseline,
fixed d1, fixed d2, fixed d3, and dynamic depth. Do not report dynamic without
the fixed-depth neighbors from the same run. Dynamic starts at d1, may demote to
d0 per-step bypass, and may probe/promote back up to d3.

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
| CUDA | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 62.6/66.6/69.5/56.5 vs 109.8 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 48.9/45.0/44.0/53.4 vs 109.7 | zero acceptance |
| ROCm | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 38.1/37.3/41.4/38.6 vs 64.7 | verifier dominates |
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
- Regression:
  `V2_Unit_MTPDepthController` covers d0 probe/cooldown, perfect-probe
  promotion, stepwise demotion, and d1-to-d0 all-zero demotion.
- Matrix runner supports bounded sweeps:
  `scripts/run_mtp_iteration_benchmark_matrix.sh --decode-tokens 16 --perfstats`.
  `summary.tsv` now includes verifier/condition/correction timing,
  main-verifier warmup/capture/replay counts, and replay reset/preserve counts.
  CPU lanes are still minutes-long even when bounded; full default-length
  matrix remains the acceptance capture.
- Runner guard: dynamic evidence now requires same-run baseline plus fixed
  d1/d2/d3 unless `--allow-partial-variants` is used for diagnostics.
- MoE verifier finding: fixed d3 greedy spends 378/220 ms verifier/condition
  time on CUDA and 711/313 ms on ROCm; correction replay is 0 ms, so the next
  target is the 524-stage all-position verifier graph and condition-forward
  frequency.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Run the bounded device matrix every iteration: CUDA/ROCm/CPU, dense/MoE,
   greedy/stochastic, baseline plus fixed d1/d2/d3 and dynamic. Schedule CPU
   separately only when runtime forces it; keep the same git/config, use
   verifier/capture columns to explain speed deltas, and record crashes/timeouts
   explicitly instead of leaving stale numbers.
2. Use full default matrix for acceptance checkpoints after bounded lanes are
   stable.
3. Attack MoE verifier/catch-up cost; CUDA, ROCm, and CPU MoE are functional
   but speed-negative.
