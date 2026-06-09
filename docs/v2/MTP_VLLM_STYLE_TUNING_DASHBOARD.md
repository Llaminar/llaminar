# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or not fully measured. Red = fails or lacks required correctness.

## Matrix

| Backend | Model | Sampling | RAG | Latest speed evidence | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 53.09/55.24/75.48/52.96 vs 44.66 | dynamic conservative |
| CUDA | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 44.58/42.42/37.04/44.74 vs 44.67 | deeper drafts reject |
| ROCm | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 38.27/33.08/41.40/40.81 vs 31.34 | dynamic stays d1 |
| ROCm | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 29.30/27.09/25.24/28.92 vs 31.23 | speed-negative |
| CPU | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 5.02/5.40/8.51/5.16 vs 4.55 | dynamic conservative |
| CPU | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 3.52/3.62/3.62/3.56 vs 4.62 | rollback path slow |
| CUDA | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 61.42/67.31/69.58/62.83 vs 110.30 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 71.62/53.14/60.62/71.61 vs 110.61 | verifier dominates |
| ROCm | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 38.58/37.36/40.24/38.61 vs 64.65 | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 34.20/30.02/29.40/32.75 vs 64.71 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 10.81/11.57/12.85/10.85 vs 17.55 | CPU publication/catch-up |
| CPU | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 12.30/12.25/12.19/12.28 vs 18.14 | host verifier overhead |

## Latest Evidence

- Fresh GPU dense matrix:
  `benchmark_results/mtp_vllm_style/20260609T014847Z-bounded-device-matrix-deferred-correction/`.
  CUDA/ROCm dense baseline, fixed d1/d2/d3, and dynamic completed for
  greedy/stochastic. CPU baseline finished at 4.57 tok/s but fixed CPU lanes
  were split out after the combined run spent about five minutes on CPU.
- Fresh CUDA MoE matrix:
  `benchmark_results/mtp_vllm_style/20260609T014540Z-cuda-moe-deferred-correction-fix/`.
  Rejected corrections now show `correction_ms=0`, `correction_count=0`, and
  nonzero `deferred_corrections`; MoE remains speed-negative.
- Fresh ROCm MoE matrix:
  `benchmark_results/mtp_vllm_style/20260609T020327Z-rocm-moe-deferred-correction-fix/`.
  Same fixed/dynamic variant shape completed for greedy/stochastic with zero
  rollback and zero correction-forward time; MoE remains speed-negative.
- Last complete CPU matrices:
  dense `20260608T231927Z-bounded-dense-device-matrix/`, MoE
  `20260609T003744Z-bounded-cpu-moe-matrix-after-cuda-fix/`.
- Regression:
  `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1RejectedCorrectionDefersToConditionToken`
  proves rejected-token publication commits shifted MTP state but defers the
  expensive correction main forward.
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
