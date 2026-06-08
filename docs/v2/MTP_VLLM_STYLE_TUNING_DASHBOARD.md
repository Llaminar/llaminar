# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on SingleDevice CUDA, ROCm, and CPU. Keep this
file under 5KB and update it after every tuning iteration.

RAG rules:

- Green: parity/functional gates pass and MTP is speed-positive versus current
  no-MTP baseline, near the llama.cpp/vLLM-style target for that backend.
- Amber: functional evidence exists, but speed is slow, unmeasured, or not yet
  accepted.
- Red: current lane fails, lacks a required correctness lane, or cannot run.

## Matrix

| Backend | Model | Sampling | RAG | Correctness | Speed evidence | Main blocker |
|---|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | yes | 56.91 vs 43.82 tok/s, 1.30x | accepted-count path works |
| CUDA | Dense 27B | stochastic | Green | yes | seed123 51.88 vs 43.82 tok/s, 1.18x | monitor acceptance |
| ROCm | Dense 27B | greedy | Green | yes | 41.44 vs 30.19 tok/s, 1.37x | accepted-count path works |
| ROCm | Dense 27B | stochastic | Amber | yes | seed123 31.31 vs 30.19 tok/s, 1.04x | small win; stochastic stream drift |
| CPU | Dense 27B | greedy | Amber | yes | not refreshed | CPU state publication/bench gap |
| CPU | Dense 27B | stochastic | Amber | yes | not refreshed | host verifier is correct, speed unmeasured |
| CUDA | MoE 35B | greedy | Amber | yes | best d3 115.94 vs 133.78 tok/s, 0.87x | verifier cost; dynamic stuck d1 |
| CUDA | MoE 35B | stochastic | Amber | yes | best dynamic seed123 97.86 vs 133.51 tok/s, 0.73x | verifier cost; depth policy |
| ROCm | MoE 35B | greedy | Amber | yes | 68.09 vs 76.23 tok/s, 0.89x | speed-negative; verifier/sync cost |
| ROCm | MoE 35B | stochastic | Amber | yes | 52.57 vs 76.23 tok/s, 0.69x | speed-negative; sampler/sync cost |
| CPU | MoE 35B | greedy | Amber | partial | not refreshed | vLLM-style CPU publication/bench gap |
| CPU | MoE 35B | stochastic | Amber | yes | not refreshed | benchmark lane missing |

## Latest Evidence

- Phase A accepted-count publication materializes shifted MTP KV rows before
  state publication; focused unit and GPU stochastic parity gates pass.
- Fresh dense publication evidence:
  `benchmark_results/mtp_vllm_style/20260608T-dense-publication-refresh/`.
  CUDA greedy d1 is 56.91 tok/s, CUDA stochastic d1 seed123 is 51.29 tok/s,
  ROCm greedy d1 is 41.44 tok/s, ROCm stochastic d1 seed123 is 31.31 tok/s.
  Long seeded stochastic acceptance matches on CUDA/ROCm at 52 accepted and 12
  rejected, but generated streams still differ at a few whitespace-token choices.
- CUDA MoE stable matrix:
  `benchmark_results/mtp_vllm_style/20260608T-cuda-moe-prod-greedy-iteration-matrix/`.
  Greedy uses `--temperature 0 --seed 123`: baseline 133.78 tok/s;
  fixed d1/d2/d3/dynamic 88.49/94.39/115.94/88.75 tok/s. Stochastic uses
  `--seed 123`: baseline 133.51 tok/s; fixed d1/d2/d3/dynamic
  91.26/93.72/86.47/97.86 tok/s. Dynamic records no promotions on this
  128-token default run, so depth-policy tuning remains open.
- Fresh ROCm MoE evidence:
  `benchmark_results/mtp_vllm_style/20260608T170802Z-rocm-moe/`.
  Baseline 76.23 tok/s; greedy d1 68.09 tok/s at 84.38% acceptance;
  stochastic d1 52.57 tok/s at 96.88% acceptance.
- ROCm MoE profiler: greedy `verifier_forward` 5.28s/192 and main-decode
  segmented stream sync 4.09s; stochastic forward/catch-up 5.08s/384 plus
  stochastic sampler 0.61s/192. State publication is correct but not fast.
- ROCm MoE stage-breakdown parity passed in isolation but took 340.57s; track
  as a test-duration/perf anomaly, not a correctness failure.
- `V2_Integration_GPUSamplingKernels` passes, so the remaining stochastic gap is
  real-model stream parity and ROCm sampler/verifier overhead, not the controlled
  sampler-kernel math. The suite now includes Qwen3.6 real-logit-style seeded
  rows with close whitespace/code-token probabilities for both CUDA and ROCm.
- Dense/MoE parity surfaces remain symmetric across CPU/CUDA/ROCm shared
  behavior; backend-specific path guards stay separate.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next Dashboard Updates

1. Use `scripts/run_mtp_iteration_benchmark_matrix.sh` to refresh baseline,
   fixed d1/d2/d3, and dynamic rows for touched CUDA/ROCm/CPU lanes.
2. Refresh dense CPU and CPU MoE stochastic benchmark cells.
3. Re-run the full iteration gate from `MTP_VLLM_STYLE_PROJECT_PLAN.md` before
   any WiP commit.
