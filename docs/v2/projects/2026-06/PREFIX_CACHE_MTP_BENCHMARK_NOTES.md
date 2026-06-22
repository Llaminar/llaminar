# Prefix Cache And MTP Benchmark Notes

Phase 14 dashboard for Qwen3.6 MTP and prefix-cache tuning. Last refreshed
2026-06-08 after the Phase 13.8 shortcut cleanup.

`Correct?` means the lane has an applicable parity/smoke gate and the current
benchmark path does not fail. `Speed` is relative to the matching current
no-MTP baseline; near-baseline or negative MTP is marked slow.

## Status Matrix

| Backend | Model | Sampling | Correct? | Speed | Current evidence |
|---|---|---|---|---|---|
| CUDA | Dense 27B Q4_K_S | greedy d1 | yes | slow | 39.93 tok/s vs 43.74 baseline, 0.91x, 85.94% accept |
| CUDA | Dense 27B Q4_K_S | stochastic d1 | yes | slow | 39.60 tok/s vs 43.74 baseline, 0.91x, 89.06% accept |
| ROCm | Dense 27B Q4_K_S | greedy d1 | yes | slow | 28.20 tok/s vs 30.21 baseline, 0.93x, 96.88% accept |
| ROCm | Dense 27B Q4_K_S | stochastic d1 | yes | slow | 24.79 tok/s vs 30.21 baseline, 0.82x, 95.31% accept |
| CUDA | MoE 35B A3B | greedy d1 | yes | slow | 118.07 tok/s vs 131.92 baseline, 0.90x, 46.88% accept; profiled 113.40 tok/s, 65.62% accept |
| CUDA | MoE 35B A3B | stochastic | no | n/a | no accepted MoE stochastic parity or benchmark lane yet |
| ROCm | MoE 35B A3B | greedy d1 | no | n/a | current no-MTP and MTP benchmarks fail during grouped prefill workspace binding |
| ROCm | MoE 35B A3B | stochastic | no | n/a | blocked by ROCm MoE prefill failure and missing stochastic MoE parity |

## Evidence

- Fresh dense greedy:
  `benchmark_results/dense_phase138/20260608T_dashboard_dense_greedy/`.
- Fresh dense stochastic:
  `benchmark_results/dense_phase138/20260608T124752Z-postcleanup-cuda-rocm-assessment/`.
- Fresh CUDA MoE:
  `benchmark_results/moe_phase138/20260608T_dashboard_moe_greedy/`.
- ROCm MoE current failure:
  `ROCmMoEKernel::prepareExpertGroupsAsync(group_active_expert_ids)` reports
  `moe_group_active_expert_ids` needs 19200 bytes but the bound workspace has
  1024 bytes. This happens for no-MTP and MTP, before decode.
- Dense stochastic profiling shows the shared verifier replay bottleneck:
  CUDA `decode_equivalent_stochastic_forward_one` is 384 calls,
  8853.94 ms total, 23.06 ms/call; ROCm is 384 calls, 12822.52 ms total,
  33.39 ms/call. ROCm stochastic sampling itself costs 661.02 ms total,
  3.44 ms/call.
- Current CUDA MoE greedy profiling shows verifier replay dominates:
  `verifier_forward` is 192 calls, 3091.48 ms total, 16.10 ms/call;
  `decode_equivalent_catchup_forward_one` is 384 calls, 3005.61 ms total,
  7.83 ms/call. The sidecar is only 169.04 ms total.
- Dense parity/smoke coverage exists for greedy and stochastic on CUDA and
  ROCm (`Qwen36*PrefixMTPParity`, GPU graph stochastic smokes). CUDA MoE greedy
  has PyTorch/style parity coverage. MoE stochastic does not.

## Interpretation

- The current answer is not "CUDA fast, ROCm slow." Dense MTP is slow on both
  CUDA and ROCm because both pay the decode-equivalent verifier replay path.
- CUDA MoE is functionally alive but no longer speed-positive in the fresh
  benchmark. The old CUDA MoE 143 tok/s row had much higher acceptance and is
  now a stale ratchet, not current acceptance evidence.
- ROCm MoE is currently broken before decode due to workspace sizing/binding for
  grouped prefill. Fix that before drawing ROCm MoE speed conclusions.
- Stochastic MoE is not accepted on either backend until it has parity,
  benchmark, and profiler evidence.

## Next Gates

1. Fix ROCm MoE grouped-prefill workspace binding and add a regression test for
   the required `moe_group_active_expert_ids` size.
2. Re-run ROCm MoE no-MTP and greedy d1 after the workspace fix.
3. Investigate the CUDA MoE acceptance drop versus the old 90%+ captures.
4. Add MoE stochastic parity and benchmark lanes before claiming production
   stochastic MTP support.
5. Keep dense MTP default-off until the shared verifier path is speed-positive
   on both CUDA and ROCm.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`; MTP-off uses `llama-bench`, MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |
