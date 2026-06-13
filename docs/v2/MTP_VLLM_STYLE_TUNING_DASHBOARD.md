# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU over SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep this file under 6KB.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or policy-sensitive. Red = failing, speed-negative, or unproven.

Iteration rule: refresh no-MTP, fixed d1/d2/d3, and dynamic for active lanes.
Before a WiP commit, broad units plus touched parity must pass.

## Latest Evidence

- SingleDevice GPU stochastic matrix:
  `benchmark_results/mtp_vllm_style/20260612T170149Z-gpu-stochastic-vllm-greedyq-c4096-post-moe-workspace/`.
  Dense is speed-positive: CUDA d2 52.6 vs 44.7 tok/s (1.18x), ROCm d1
  37.0 vs 31.3 tok/s (1.18x). MoE stochastic is speed-red: CUDA best 81.6
  vs 114.4 tok/s, ROCm best 51.5 vs 68.7 tok/s in
  `20260612T225841Z-moe-stochastic-gpu-stage-timing`.
- vLLM check: bonus sampling is upfront there too; our processed-target,
  one-hot-q path is aligned. The remaining GPU MoE stochastic
  blocker is target/condition transaction cost, not compact
  sampler table math or bonus deferral.
- Phase 8 request batching: padded metadata, scheduler/owner, batched
  publication, variable-length hidden gather, true batched sidecar drafts,
  greedy d1/d2/d3, and bounded stochastic continuation are unit-proven.
  20260613 fix: stochastic batched prefill now samples compact rows with the
  same logical-position thresholds as scalar MTP. CUDA/RB=2 and ROCm/RB=2
  MoE stochastic `-n16` both pass with 12 accepted, 78 rejected, zero rollbacks
  at 75.90/52.12 tok/s. Same-run scalar d1 is 88.18/56.47 and no-MTP is
  115.32/69.79. RoPE explicit row metadata is now graph-capture safe on
  CUDA/ROCm; ROCm MoE RB=2 greedy `-n16` passes without segmented-capture
  fallback at 69.29 tok/s. Phase 8 is correctness-green but perf-red.
- Phase 10 MoE stochastic remains Red. Short RB=2 is rejected; long RB=1 is
  closer but not accepted: CUDA d3 136.4 vs 138.6 tok/s, ROCm d2 77.0 vs 77.0
  while dynamic stays 62.0. A generated-policy refresh probe
  `20260613T_phase10_depth_policy_refresh_rocm_moe_stoch` was rejected too:
  ROCm dynamic 60.3 vs 77.7 and demoted back to d1. Keep the table unchanged.
- Fresh Phase 9 ROCm dense greedy topology matrix:
  `benchmark_results/mtp_vllm_style/20260612T234446Z-iteration-matrix-3ed9c37e/`.
  LocalTP best d3 55.4 vs 34.1 tok/s (1.62x), dynamic 54.2 (1.59x). LocalPP
  best dynamic 62.9 vs 30.3 tok/s (2.08x), fixed d3 55.5 (1.83x). All MTP
  lanes completed with zero rollbacks.
- ExpertOverlay mixed parity is green for ROCm2TP-hot plus CPU2LocalTP-cold.
  Gate: `^V2_Integration_Parity_Qwen36MoE_ExpertOverlay_` passed 6/6 after
  the ROCm MoE local-expert workspace and stream-handoff fixes.
- NodeLocalTP dense parity is green: full `^V2_Integration_Parity_Qwen36_NodeLocalTP_`
  passed 5/5 real-model tests plus fixture. LocalPP dense parity is green:
  full `^V2_Integration_Parity_Qwen36_LocalPP_` passed 7/7.

## Topology Matrix

| Topology | Impl | Parity | Bench/Tuning |
|---|---|---|---|
| SingleDevice | Green dense/MoE greedy+stochastic on CPU/CUDA/ROCm | Green broad device matrix | Dense accepted; MoE stochastic Red |
| LocalTP | Green dense fixed d1/d2/d3 + dynamic; rank-wide depth/stream handoff | Green Qwen3.6 dense parity | ROCm dense d3 1.62x Green; stochastic/MoE unproven |
| LocalPP | Green dense final-stage sidecar + all-stage publication | Green prefix+d1/d3/dyn/stoch+prefix-MTP restore | ROCm dense dynamic 2.08x Green; stochastic/MoE unproven |
| NodeLocalTP | Green dense fixed d1/d2/d3 + dynamic scalar broadcast | Green full dense NodeLocalTP parity | Benchmark preset ready; MoE unproven |
| ExpertOverlay | Green MoE hot/mixed correctness path | Green ROCm2TP-hot + CPU2LocalTP-cold parity | Speed Amber/Red; preset ready |

## Device Matrix

| Backend | Model | Sampling | RAG | Latest decode tok/s | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | base 44.0; d1/d2/d3/dyn 63.9/76.1/91.4/87.8 | dynamic lag |
| CUDA | Dense 27B | stochastic | Green | base 44.7; 52.6/52.6/47.7/47.7 | short-lane d3 regress |
| ROCm | Dense 27B | greedy | Green | base 30.3; 42.3/48.1/65.0/52.9 | absolute CUDA gap |
| ROCm | Dense 27B | stochastic | Green | base 31.3; 37.0/28.7/27.0/36.9 | d2/d3 cost |
| CPU | Dense 27B | greedy | Green | base 4.7; 5.9/6.0/9.3/6.1 | dynamic shallow |
| CPU | Dense 27B | stochastic | Green | base 4.46; 5.06/5.78/4.85/5.41 | verifier/condition |
| CUDA | MoE 35B | greedy | Amber | base 139.2; d1/d2/d3/dyn 129.6/136.6/139.1/146.1 | weak win |
| CUDA | MoE 35B | stochastic | Red | long base 138.6; 113.2/124.8/136.4/134.1; RB2 best 70.2 | verifier+condition cost |
| ROCm | MoE 35B | greedy | Amber | base 77.0; 79.0/91.5/90.3/86.1 | below dense-class win |
| ROCm | MoE 35B | stochastic | Red | long base 77.0; 59.9/77.0/74.5/62.0; refresh dyn 60.3 | verifier+condition cost |
| CPU | MoE 35B | greedy | Amber | base 17.7; 13.5/13.9/12.4/13.3 | host verifier cost |
| CPU | MoE 35B | stochastic | Amber | base 17.6; 14.5/13.3/14.1/14.0 | host verifier cost |

## Current Read

Dense MTP is accepted on CUDA/ROCm SingleDevice and ROCm LocalTP/LocalPP. MoE
stochastic is the active Phase 10 blocker: verifier/condition work outweighs
recovered tokens, and RB=2 worsens acceptance. Next target: cheaper MoE target
verification plus condition-token replay, then selective batching.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`: dense no-MTP 41.83,
d1 54.9, d3 52.5 tok/s; MoE no-MTP 118.26, d1 142.0, d3 132.8 tok/s.

## Next

1. Profile MoE stochastic RB=1 verifier/condition cost on CUDA/ROCm and remove
   avoidable target-graph or condition-token replay work.
2. Re-test RB=2 only after scalar MoE stochastic is speed-positive.
3. Run Phase 9 NodeLocalTP CPU and ExpertOverlay benchmark presets when the
   active slice touches those topologies.
