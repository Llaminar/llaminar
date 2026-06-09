# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or not fully measured. Red = fails or lacks required correctness.

## Matrix

| Backend | Model | Sampling | RAG | Latest speed evidence | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 60.09/51.93/89.13/60.28 vs 44.60 | dynamic conservative |
| CUDA | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 40.69/42.66/37.15/40.67 vs 44.59 | speed-negative short lane |
| ROCm | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 45.23/31.33/38.70/45.45 vs 31.28 | dynamic stays d1 |
| ROCm | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 26.94/27.31/25.17/26.88 vs 31.26 | speed-negative |
| CPU | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 5.02/5.40/8.51/5.16 vs 4.55 | dynamic conservative |
| CPU | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 3.52/3.62/3.62/3.56 vs 4.62 | rollback path slow |
| CUDA | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 62.02/67.35/70.75/63.23 vs 109.90 | verifier/correction cost |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 67.64/55.92/49.62/62.72 vs 110.51 | reject-heavy d2/d3 cost |
| ROCm | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 35.39/36.13/38.21/35.68 vs 64.29 | verifier/catch-up cost |
| ROCm | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 32.69/28.76/29.78/32.35 vs 64.26 | sampler/verifier overhead |
| CPU | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 10.81/11.57/12.85/10.85 vs 17.55 | CPU publication/catch-up |
| CPU | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 12.30/12.25/12.19/12.28 vs 18.14 | host verifier overhead |

## Latest Evidence

- Bounded dense device matrix:
  `benchmark_results/mtp_vllm_style/20260608T231927Z-bounded-dense-device-matrix/`.
  Completed CUDA/ROCm/CPU dense greedy and stochastic with baseline, fixed
  d1/d2/d3, and dynamic at 16 decode tokens. Greedy is speed-positive across
  all three backends; stochastic is speed-negative across all three.
- Bounded MoE GPU matrix:
  `benchmark_results/mtp_vllm_style/20260609T003150Z-bounded-gpu-moe-matrix-after-cuda-fix/`.
  CUDA/ROCm greedy and stochastic baseline, d1/d2/d3, and dynamic all completed.
  CUDA graph-capture crash is fixed, but every MoE MTP lane is speed-negative.
- CUDA MoE scoped replay diagnostics:
  `benchmark_results/mtp_vllm_style/20260609T011555Z-cuda-moe-greedy-scoped-reset/`
  and `20260609T011753Z-cuda-moe-stochastic-scoped-reset/`. Verifier replay now
  occurs in MoE MTP lanes, improving d1/d2/dynamic versus the prior bounded
  greedy matrix, but verifier plus correction time still keeps MoE speed-negative.
- Bounded MoE CPU matrix:
  `benchmark_results/mtp_vllm_style/20260609T003744Z-bounded-cpu-moe-matrix-after-cuda-fix/`.
  CPU greedy/stochastic lanes completed. MTP is also speed-negative there.
- Regression:
  `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1CorrectionReplayResetsCapturedStateBoundary`
  proves rejected-token accepted-state publication resets captured replay and
  kernel dynamic state before correction main-decode replay.
- Matrix runner now supports bounded sweeps:
  `scripts/run_mtp_iteration_benchmark_matrix.sh --decode-tokens 16 --perfstats`.
  `summary.tsv` now includes verifier/correction timing, main-verifier
  warmup/capture/replay counts, and replay reset/preserve counts. CPU lanes are
  still minutes-long even when bounded; full default-length matrix remains the
  acceptance capture.

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
   separately if needed; use verifier/capture columns to explain speed deltas,
   and record crashes/timeouts explicitly instead of leaving stale numbers.
2. Use full default matrix for acceptance checkpoints after bounded lanes are
   stable.
3. Attack MoE verifier/catch-up cost; CUDA, ROCm, and CPU MoE are functional
   but speed-negative.
