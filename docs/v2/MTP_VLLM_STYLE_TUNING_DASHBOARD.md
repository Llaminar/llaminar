# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or not fully measured. Red = fails or lacks required correctness.

## Matrix

| Backend | Model | Sampling | RAG | Latest speed evidence | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 51.80/55.15/59.50/60.85 vs 44.63 | dynamic now OK in this lane |
| CUDA | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 39.03/34.81/29.58/38.95 vs 44.63 | low acceptance |
| ROCm | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 45.61/33.03/41.91/39.17 vs 31.19 | d2 acceptance dip |
| ROCm | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 24.85/20.58/24.27/24.81 vs 31.30 | speed-negative |
| CPU | Dense 27B | greedy | Green | 16tok d1/d2/d3/dyn 5.49/5.69/8.89/5.50 vs 4.59 | dynamic conservative |
| CPU | Dense 27B | stochastic | Amber | 16tok d1/d2/d3/dyn 3.45/3.56/3.58/3.50 vs 4.58 | rollback path slow |
| CUDA | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 64.23/66.72/69.42/61.26 vs 110.49 | verifier dominates |
| CUDA | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 50.28/47.31/41.69/49.12 vs 111.24 | near-zero acceptance |
| ROCm | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 38.64/37.46/43.57/38.64 vs 64.81 | verifier dominates |
| ROCm | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 31.82/24.64/24.20/29.73 vs 65.08 | verifier dominates |
| CPU | MoE 35B | greedy | Amber | 16tok d1/d2/d3/dyn 11.97/11.50/12.69/11.37 vs 17.48 | CPU verifier cost |
| CPU | MoE 35B | stochastic | Amber | 16tok d1/d2/d3/dyn 11.91/12.08/12.68/12.22 vs 17.53 | host verifier overhead |

## Latest Evidence

- Fresh full bounded matrix:
  `benchmark_results/mtp_vllm_style/20260609T022015Z-bounded-matrix-skip-allpos-sidecar-checkpoint/`.
  It covers CUDA/ROCm/CPU, dense/MoE, greedy/stochastic, baseline plus fixed
  d1/d2/d3/dynamic at 16 decode tokens. All rows completed. GPU all-position
  lanes now skip the unused post-sidecar checkpoint (`capture_post_sidecar` is
  absent; `post_sidecar_checkpoint_skipped_all_position_publication` is present).
- Regression:
  `Qwen36MoECUDASingleDevicePrefixMTPPathGuards.Depth1RejectedCorrectionDefersToConditionToken`
  proves rejected-token publication commits shifted MTP state but defers the
  expensive correction main forward.
- Matrix runner supports bounded sweeps:
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
