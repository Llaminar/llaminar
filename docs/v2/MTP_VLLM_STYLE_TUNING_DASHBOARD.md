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
| CUDA | Dense 27B | greedy | Amber | yes | 39.93 vs 43.74 tok/s, 0.91x | needs refresh after publication path |
| CUDA | Dense 27B | stochastic | Amber | yes | 39.60 vs 43.74 tok/s, 0.91x | needs bench refresh after publication path |
| ROCm | Dense 27B | greedy | Amber | yes | 28.20 vs 30.21 tok/s, 0.93x | needs refresh after publication path |
| ROCm | Dense 27B | stochastic | Amber | yes | 24.79 vs 30.21 tok/s, 0.82x | needs bench refresh after publication path |
| CPU | Dense 27B | greedy | Amber | yes | not refreshed | CPU state publication/bench gap |
| CPU | Dense 27B | stochastic | Amber | yes | not refreshed | host verifier is correct, speed unmeasured |
| CUDA | MoE 35B | greedy | Amber | yes | 118.07 vs 131.92 tok/s, 0.90x | low acceptance, verifier replay |
| CUDA | MoE 35B | stochastic | Amber | yes | not refreshed | benchmark lane missing |
| ROCm | MoE 35B | greedy | Amber | yes | 68.09 vs 76.23 tok/s, 0.89x | speed-negative; verifier/sync cost |
| ROCm | MoE 35B | stochastic | Amber | yes | 52.57 vs 76.23 tok/s, 0.69x | speed-negative; sampler/sync cost |
| CPU | MoE 35B | greedy | Amber | partial | not refreshed | vLLM-style CPU publication/bench gap |
| CPU | MoE 35B | stochastic | Amber | yes | not refreshed | benchmark lane missing |

## Latest Evidence

- CUDA no-MTP determinism fixed by invalidating request-scoped prefill graph
  captures on `SessionReset`. Evidence: `V2_Unit_` 497/497 and CUDA
  no-MTP fresh-run determinism parity passed.
- Phase A all-position publication now explicitly materializes shifted MTP KV
  rows before state publication. Focused unit gate passed, and GPU stochastic
  parity passed for dense+MoE on CUDA+ROCm in one command.
- Fresh dense evidence:
  `benchmark_results/dense_phase138/20260608T_dashboard_dense_greedy/` and
  `benchmark_results/dense_phase138/20260608T124752Z-postcleanup-cuda-rocm-assessment/`.
- Fresh CUDA MoE greedy evidence:
  `benchmark_results/moe_phase138/20260608T_dashboard_moe_greedy/`.
- Fresh ROCm MoE evidence:
  `benchmark_results/mtp_vllm_style/20260608T170802Z-rocm-moe/`.
  Baseline 76.23 tok/s; greedy d1 68.09 tok/s at 84.38% acceptance;
  stochastic d1 52.57 tok/s at 96.88% acceptance.
- ROCm MoE profiler: greedy `verifier_forward` 5.28s/192 and main-decode
  segmented stream sync 4.09s; stochastic forward/catch-up 5.08s/384 plus
  stochastic sampler 0.61s/192. State publication is correct but not fast.
- ROCm MoE stage-breakdown parity passed in isolation but took 340.57s; track
  as a test-duration/perf anomaly, not a correctness failure.
- Dense stochastic profiler: CUDA `decode_equivalent_stochastic_forward_one`
  23.06 ms/call; ROCm 33.39 ms/call. Verifier replay remains the dense blocker.
- CUDA MoE profiler: `verifier_forward` 16.10 ms/call, sidecar only 0.88
  ms/call. The verifier path, not draft generation, dominates.
- Dense/MoE parity surfaces remain symmetric across CPU/CUDA/ROCm shared
  behavior. Latest GPU stochastic verifier gate passed: dense ROCm 27.21s,
  dense CUDA 26.41s, MoE ROCm 22.27s, MoE CUDA 18.33s.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next Dashboard Updates

1. Refresh dense greedy/stochastic cells on CUDA/ROCm/CPU with the vLLM-style
   publication path and host/device stochastic verifier coverage.
2. Refresh CUDA/CPU MoE stochastic benchmark cells.
3. Re-run the full iteration gate from `MTP_VLLM_STYLE_PROJECT_PLAN.md` before
   any WiP commit.
