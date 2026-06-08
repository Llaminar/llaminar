# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or not fully measured. Red = fails or lacks required correctness.

## Matrix

| Backend | Model | Sampling | RAG | Latest speed evidence | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | d1/d2/d3/dyn 57.25/70.60/79.28/60.73 vs 43.83 tok/s | dynamic conservative |
| CUDA | Dense 27B | stochastic | Green | d1/d2/d3/dyn 50.97/45.53/45.02/49.61 vs 43.77 tok/s | d2/d3 acceptance cliff |
| ROCm | Dense 27B | greedy | Green | d1/d2/d3/dyn 42.91/35.01/35.41/40.62 vs 30.07 tok/s | d1 best on default prompt |
| ROCm | Dense 27B | stochastic | Amber | d1/d2/d3/dyn 31.21/26.96/24.59/31.69 vs 30.22 tok/s | small d1 win; deeper slow |
| CPU | Dense 27B | greedy | Amber | partial: d1 4.96 vs 4.28 tok/s | full CPU matrix slow |
| CPU | Dense 27B | stochastic | Amber | not refreshed | host verifier speed gap |
| CUDA | MoE 35B | greedy | Amber | fixed d1 crash fixed; standalone d1 80.58 tok/s; old baseline 133.78 | verifier/catch-up cost |
| CUDA | MoE 35B | stochastic | Amber | old best dynamic 97.86 vs 133.51 tok/s | verifier cost; depth policy |
| ROCm | MoE 35B | greedy | Amber | old d1 68.09 vs 76.23 tok/s | speed-negative |
| ROCm | MoE 35B | stochastic | Amber | old d1 52.57 vs 76.23 tok/s | sampler/verifier overhead |
| CPU | MoE 35B | greedy | Amber | not refreshed | CPU publication/bench gap |
| CPU | MoE 35B | stochastic | Amber | not refreshed | CPU publication/bench gap |

## Latest Evidence

- Partial full matrix:
  `benchmark_results/mtp_vllm_style/20260608T223800Z-full-iteration-matrix-correction-boundary/`.
  Completed CUDA dense, ROCm dense, CPU dense baseline and d1 before CPU d2 was
  stopped for runtime. Use this for current dense dashboard cells.
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
  Full default-length matrix remains the acceptance capture.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next

1. Run bounded full device matrix each iteration: CUDA/ROCm/CPU, dense/MoE,
   greedy/stochastic, baseline plus fixed d1/d2/d3 and dynamic.
2. Use full default matrix for acceptance checkpoints after bounded lanes are
   stable.
3. Attack MoE verifier/catch-up cost; both CUDA and ROCm MoE remain
   speed-negative.
