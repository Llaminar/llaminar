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
| CUDA | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 61.1/53.8/59.5/49.4 vs 44.6 | dynamic not best depth |
| CUDA | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 38.3/33.1/29.6/36.8 vs 44.6 | low acceptance |
| ROCm | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 38.6/33.2/41.7/34.4 vs 31.4 | dynamic conservative |
| ROCm | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 24.9/20.5/24.4/23.5 vs 31.3 | speed-negative |
| CPU | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 5.7/5.7/9.1/5.6 vs 4.5 | dynamic too shallow |
| CPU | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 3.4/3.6/3.6/3.6 vs 4.4 | rollback path slow |
| CUDA | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 61.8/67.1/70.2/57.0 vs 109.7 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 50.7/47.1/43.0/57.6 vs 110.2 | near-zero acceptance |
| ROCm | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 38.8/34.8/41.7/33.3 vs 64.4 | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 30.0/25.9/24.1/28.7 vs 64.8 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 11.5/11.4/12.9/11.4 vs 18.2 | CPU verifier cost |
| CPU | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 12.4/12.7/12.2/12.6 vs 18.1 | host verifier overhead |

## Latest Evidence

- Fresh full bounded matrix:
  `benchmark_results/mtp_vllm_style/20260609T-post-hysteresis-tune-matrix/`.
  It covers CUDA/ROCm/CPU, dense/MoE, greedy/stochastic, baseline plus fixed
  d1/d2/d3/dynamic at 16 decode tokens. All rows completed.
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
- MoE verifier finding: fixed d3 greedy spends 379/214 ms verifier/condition
  time on CUDA and 684/312 ms on ROCm; correction replay is 0 ms, so the next
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
