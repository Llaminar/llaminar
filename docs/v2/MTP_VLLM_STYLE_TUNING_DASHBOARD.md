# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CUDA/ROCm/CPU across SingleDevice, LocalTP,
LocalPP, NodeLocalTP, and ExpertOverlay. Keep this file under 6KB.

RAG: Green = correct and speed-positive near target. Amber = correct but slow,
partial, or policy-sensitive. Red = failing, speed-negative, or unproven.

Iteration contract: refresh no-MTP, fixed d1/d2/d3, and dynamic for the active
device/model/mode. Before a WiP commit, broad units plus touched parity must pass.

## Latest Evidence

- GPU stochastic matrix:
  `benchmark_results/mtp_vllm_style/20260612T170149Z-gpu-stochastic-vllm-greedyq-c4096-post-moe-workspace/`.
  CUDA/ROCm dense+MoE rows ran with zero rollbacks.
- Dense stochastic remains speed-positive in the short `-c 4096` lane: CUDA best
  d2 52.6 vs 44.7 tok/s (1.18x), ROCm best d1 37.0 vs 31.3 tok/s (1.18x).
  Dynamic starts at d3 on CUDA and d1 on ROCm; `n=16` is too short for
  hysteresis updates.
- MoE stochastic is functionally green but speed-red on both GPUs. Same-run
  baselines beat every MTP depth: CUDA best d3 79.0 vs 115.3 tok/s; ROCm best
  d1 51.8 vs 69.1 tok/s.
- `rocprof` on ROCm MoE d1 shows true GPU time dominated by GDN, routed/shared
  MoE GEMMs, native GEMM, and LM head. Sampling kernels are secondary.
- Phase 9 ExpertOverlay mixed parity is now green for ROCm2TP-hot plus
  CPU2LocalTP-cold. Fixes: ROCm MoE local-expert nested workspace declarations,
  LocalTP all-position verifier stream consumption, and deterministic ROCm MoE
  parity mode. Guards: `V2_Unit_RankOrchestrator`,
  `V2_Unit_MoELocalExpertStage_PreparedWeights`, and full
  `^V2_Integration_Parity_Qwen36MoE_ExpertOverlay_`.
- Fresh gate: full `^V2_Unit_` passed 502/502 after rebuilding, then
  `^V2_Integration_ROCmMoEKernel$|^V2_Integration_Parity_Qwen36MoE_ExpertOverlay_`
  passed 6/6.
- NodeLocalTP dynamic-depth scalar coordination is green in the dense parity
  lane: full `^V2_Integration_Parity_Qwen36_NodeLocalTP_` passed 5/5
  real-model tests plus fixture.
- LocalPP dense MTP is green on ROCm for fixed d1/d3, dynamic depth, prefix
  restore, and prefix+MTP restore. Stochastic and all-position publication
  remain gated.
- Stage-owned CUDA side-stream workspace declarations still hold the dense VRAM
  win: one-token d3 stochastic graph workspace is about 784 MB instead of the
  stale LM-head-sized 1827 MB plan.

## Topology Matrix

| Topology | Impl | Parity | Bench/Tuning |
|---|---|---|---|
| SingleDevice | Green dense/MoE greedy+stochastic on CPU/CUDA/ROCm | Green broad device matrix | Dense accepted; MoE speed weak |
| LocalTP | Green dense fixed d1/d2/d3 + dynamic; rank-wide depth and stream handoff wired | Green Qwen3.6 dense parity | MoE bench sparse; keep in matrix |
| LocalPP | Amber: final-stage sidecar; stochastic gated | Green dense prefix+d1/d3/dyn+prefix-MTP restore | Bench/all-position pending |
| NodeLocalTP | Green dense fixed d1/d2/d3 + dynamic scalar broadcast | Green full dense NodeLocalTP parity | MoE still unproven |
| ExpertOverlay | Green MoE hot/mixed correctness path; dense not a separate target | Green ROCm2TP-hot + CPU2LocalTP-cold parity | Speed Amber/Red until MoE economics improve |

## Device Matrix

| Backend | Model | Sampling | RAG | Latest decode tok/s | Main blocker |
|---|---|---|---|---|---|
| CUDA | Dense 27B | greedy | Green | base 44.0; d1/d2/d3/dyn 63.9/76.1/91.4/87.8 | dynamic lag |
| CUDA | Dense 27B | stochastic | Green | base 44.7; 52.6/52.6/47.7/47.7 | short-lane d3 regress |
| ROCm | Dense 27B | greedy | Green | base 30.3; 42.3/48.1/65.0/52.9 | absolute CUDA gap |
| ROCm | Dense 27B | stochastic | Green | base 31.3; 37.0/28.7/27.0/36.9 | d2/d3 acceptance/cost |
| CPU | Dense 27B | greedy | Green | base 4.7; 5.9/6.0/9.3/6.1 | dynamic shallow |
| CPU | Dense 27B | stochastic | Green | base 4.46; 5.06/5.78/4.85/5.41 | verifier/condition |
| CUDA | MoE 35B | greedy | Amber | base 139.2; d1/d2/d3/dyn 129.6/136.6/139.1/146.1 | weak win |
| CUDA | MoE 35B | stochastic | Red | base 115.3; 78.5/76.0/79.0/78.3 | verifier+condition cost |
| ROCm | MoE 35B | greedy | Amber | base 77.0; 79.0/91.5/90.3/86.1 | below dense-class win |
| ROCm | MoE 35B | stochastic | Red | base 69.1; 51.8/48.5/40.7/51.6 | verifier/model drain |
| CPU | MoE 35B | greedy | Amber | base 17.7; 13.5/13.9/12.4/13.3 | host verifier cost |
| CPU | MoE 35B | stochastic | Amber | base 17.6; 14.5/13.3/14.1/14.0 | host verifier cost |

## Current Read

Dense relative speedup is accepted on CUDA/ROCm. GPU MoE greedy is correct and
sometimes positive but still below the dense-class win. GPU MoE stochastic is
not benchmark-accepted: the vLLM-shaped path is correct, but verifier,
condition, and sidecar transaction work overwhelm recovered tokens.

## Target Anchors

llama.cpp CUDA anchors from `ggml-org/llama.cpp@6ddc943`: dense no-MTP 41.83,
d1 54.9, d3 52.5 tok/s; MoE no-MTP 118.26, d1 142.0, d3 132.8 tok/s.

## Next

1. Continue Phase 9 multi-device hardening, keeping LocalTP, LocalPP,
   ExpertOverlay, and NodeLocalTP status updated beside SingleDevice.
2. Continue Phase 8 transaction-level vLLM alignment for MoE; reduce verifier,
   condition, and sidecar graph economics before deeper sampler/kernel tuning.
3. Re-run the bounded GPU MoE stochastic matrix after the next verifier/condition
   tuning slice, then refresh this table from same-run baselines.
