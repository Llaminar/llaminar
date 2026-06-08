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
| CUDA | Dense 27B | stochastic | Amber | yes | 39.60 vs 43.74 tok/s, 0.91x | verifier replay path |
| ROCm | Dense 27B | greedy | Amber | yes | 28.20 vs 30.21 tok/s, 0.93x | needs refresh after publication path |
| ROCm | Dense 27B | stochastic | Amber | yes | 24.79 vs 30.21 tok/s, 0.82x | verifier replay plus sampler cost |
| CPU | Dense 27B | greedy | Amber | yes | not refreshed | CPU state publication/bench gap |
| CPU | Dense 27B | stochastic | Amber | yes | not refreshed | host verifier is correct, speed unmeasured |
| CUDA | MoE 35B | greedy | Amber | yes | 118.07 vs 131.92 tok/s, 0.90x | low acceptance, verifier replay |
| CUDA | MoE 35B | stochastic | Red | missing | none | no MoE stochastic parity/bench lane |
| ROCm | MoE 35B | greedy | Amber | yes | not refreshed | needs full parity/benchmark refresh |
| ROCm | MoE 35B | stochastic | Red | fails | none | ROCm MoE prefill plus no stochastic lane |
| CPU | MoE 35B | greedy | Amber | partial | not refreshed | vLLM-style CPU publication/bench gap |
| CPU | MoE 35B | stochastic | Red | missing | not refreshed | no accepted stochastic MoE CPU lane |

## Latest Evidence

- CUDA dense no-MTP determinism regression fixed: `clearCache()` now invalidates
  request-scoped prefill graph captures with `SessionReset`, preventing stale
  prefill replay without logits gather. Evidence:
  `V2_Unit_` 497/497 and
  `V2_Integration_Parity_Qwen36_CUDA_SingleDevice_Qwen36CUDASingleDevicePrefixMTPParity_NoMTPBenchmarkStyleFreshRunnerDeterminism`
  passed in 155s.
- Phase A publication path: focused MTP unit gate passed after adding the
  shared all-position greedy verifier contract, accepted-count KV publication,
  graph-order state publication, DGO publication capability, and runner coverage
  for greedy/stochastic accept-all plus rejected-correction replay. Real-model
  dense parity/benchmark refresh is next.
- Dense fresh runs:
  `benchmark_results/dense_phase138/20260608T_dashboard_dense_greedy/` and
  `benchmark_results/dense_phase138/20260608T124752Z-postcleanup-cuda-rocm-assessment/`.
- CUDA MoE fresh runs:
  `benchmark_results/moe_phase138/20260608T_dashboard_moe_greedy/`.
- ROCm MoE greedy moved off the workspace hard-fail: focused
  `Qwen36MoEROCmSingleDevicePrefixMTPParity_MTPGreedyMatchesPyTorchDecodeTokens`
  passed in 210.03s after grouped-prefill workspace sizing was fixed.
- Dense stochastic profiler: CUDA `decode_equivalent_stochastic_forward_one`
  23.06 ms/call; ROCm 33.39 ms/call. This is the shared reason dense MTP is
  not fast.
- CUDA MoE profiler: `verifier_forward` 16.10 ms/call, sidecar only 0.88
  ms/call. The verifier path, not draft generation, dominates.
- Dense parity surface is now symmetric across pinned CPU/CUDA/ROCm targets:
  each backend exposes the same 18 Qwen3.6 Prefix+MTP tests. Latest stochastic
  verifier gate passed for all three:
  ROCm 26.83s, CUDA 25.74s, CPU 83.40s.
- MoE Prefix+MTP parity discovery is also symmetric for shared behavior:
  CPU/CUDA/ROCm each expose the same 14 Qwen3.6 MoE SingleDevice cases.
  CUDA keeps 2 extra path guards for CUDA-specific fused/grouped kernels.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`:

| Lane | Dense 27B decode | MoE 35B decode |
|---|---:|---:|
| no-MTP | 41.83 tok/s | 118.26 tok/s |
| MTP d1 | 54.9 tok/s | 142.0 tok/s |
| MTP d3 | 52.5 tok/s | 132.8 tok/s |

## Next Dashboard Updates

1. Fix ROCm MoE workspace binding and refresh ROCm MoE no-MTP/greedy cells.
2. Add accepted MoE stochastic parity/benchmark cells for CUDA, ROCm, and CPU.
3. Refresh dense greedy/stochastic cells on CUDA/ROCm/CPU with the vLLM-style
   publication path and host/device stochastic verifier coverage.
4. Re-run the full iteration gate from `MTP_VLLM_STYLE_PROJECT_PLAN.md` before
   any WiP commit.
