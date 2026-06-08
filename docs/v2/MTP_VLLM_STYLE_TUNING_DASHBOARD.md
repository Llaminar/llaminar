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
| CUDA | MoE 35B | greedy | Amber | fixed d1 crash fixed; standalone d1 80.58 tok/s; old baseline 133.78 | verifier/catch-up cost |
| CUDA | MoE 35B | stochastic | Amber | old best dynamic 97.86 vs 133.51 tok/s | verifier cost; depth policy |
| ROCm | MoE 35B | greedy | Amber | old d1 68.09 vs 76.23 tok/s | speed-negative |
| ROCm | MoE 35B | stochastic | Amber | old d1 52.57 vs 76.23 tok/s | sampler/verifier overhead |
| CPU | MoE 35B | greedy | Amber | not refreshed | CPU publication/bench gap |
| CPU | MoE 35B | stochastic | Amber | not refreshed | CPU publication/bench gap |

## Latest Evidence

- Bounded dense device matrix:
  `benchmark_results/mtp_vllm_style/20260608T231927Z-bounded-dense-device-matrix/`.
  Completed CUDA/ROCm/CPU dense greedy and stochastic with baseline, fixed
  d1/d2/d3, and dynamic at 16 decode tokens. Greedy is speed-positive across
  all three backends; stochastic is speed-negative across all three.
- CUDA MoE fixed-d1 correction replay lane:
  `benchmark_results/mtp_vllm_style/20260608T-cuda-moe-fixed-d1-correction-boundary/`.
  The former graph-capture crash is fixed; d1 completed at 80.58 tok/s with
  85.94% acceptance.
- Regression:
  `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1CorrectionReplayResetsCapturedStateBoundary`
  proves rejected-token accepted-state publication resets captured replay and
  kernel dynamic state before correction main-decode replay.
- Matrix runner now supports bounded sweeps:
  `scripts/run_mtp_iteration_benchmark_matrix.sh --decode-tokens 16 --perfstats`.
  CPU lanes are still minutes-long even when bounded; full default-length
  matrix remains the acceptance capture.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Run bounded full device matrix each iteration: CUDA/ROCm/CPU, dense/MoE,
   greedy/stochastic, baseline plus fixed d1/d2/d3 and dynamic. Schedule CPU
   separately if needed, but do not omit it from dashboard evidence.
2. Use full default matrix for acceptance checkpoints after bounded lanes are
   stable.
3. Attack MoE verifier/catch-up cost; both CUDA and ROCm MoE remain
   speed-negative.
