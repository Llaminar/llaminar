# MoE Expert Overlay Phase 0 Baseline

**Date:** 2026-05-09
**Scope:** Phase 0 of [docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): rebaseline current state and lock the audit harness.

## CTest Surface

The discovered `V2_Integration_Parity_Qwen35MoEExpertOverlay_*` names are concise and stable:

- `V2_Integration_Parity_Qwen35MoEExpertOverlay_OverlayPlanTopology_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
- `V2_Integration_Parity_Qwen35MoEExpertOverlay_OverlayPlanTopology_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- `V2_Integration_Parity_Qwen35MoEExpertOverlay_PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
- `V2_Integration_Parity_Qwen35MoEExpertOverlay_DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
- `V2_Integration_Parity_Qwen35MoEExpertOverlay_PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- `V2_Integration_Parity_Qwen35MoEExpertOverlay_DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`

CTest also includes `V2_FetchModelsFixture` when the regex selects these parity tests.

## Current Baseline

Required parity selection:

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result on 2026-05-09: 7/7 CTests passed, 0 failed. This includes the model fixture, two topology-only tests, and four real parity CTests whose GTest bodies currently skip intentionally.

Verbose real-parity sweep:

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_(PrefillParity|DecodeParity)_" --verbose --parallel
```

Result on 2026-05-09: 5/5 CTests passed, 0 failed. The model fixture passed. All four real `PrefillParity_*` / `DecodeParity_*` bodies reported GTest skips after constructing the real runner and detecting current production blockers.

Required unit selection:

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Parallel|Overlay)" --output-on-failure --parallel
```

Result on 2026-05-09: 6/6 CTests passed, 0 failed. This includes `V2_FetchModelsFixture` plus `V2_Unit_MoEExpertParallelPlan`, `V2_Unit_MoEExpertParallelPlanner`, `V2_Unit_MoEExpertParallelReduceStage`, `V2_Unit_MoEExpertOverlayConfig`, and `V2_Unit_Qwen35MoEExpertOverlayGraph`.

## Topology vs Real Parity

`OverlayPlanTopology_*` tests are audit harness checks only. They assert that planned same-layer routed tiers are non-empty for the requested ROCm/CPU and CUDA/ROCm/CPU layouts, but they do not prove production inference parity.

`PrefillParity_*` and `DecodeParity_*` are the real V2 parity bodies. In Phase 0 they must either run `runPrefillParity()` / `runDecodeParity()` or report an explicit GTest skip reason from `overlayRuntimeBlockers()`.

## Intentional Real-Parity Skips

The ROCm shared/hot plus CPU cold layout currently skips both prefill and decode because:

- `rocm_shared_hot` requests `TensorParallelExperts` across two ROCm participants, but the production `DeviceGraphExecutor` overlay path still lowers that domain to the primary participant only.
- The `shared_hot` tier lowers to `ROCm:0`, but `ExpertGemmRegistry` does not contain all active gate/up/down expert engines for that tier.
- `cpu_cold` requests `TensorParallelExperts` across two CPU participants, but the production overlay path still lowers that domain to the primary participant only.

The CUDA shared/hottest plus ROCm hot plus CPU cold layout currently skips both prefill and decode because:

- The `shared_hottest` tier lowers to `CUDA:0`, but `ExpertGemmRegistry` does not contain all active gate/up/down expert engines for that tier.
- `rocm_hot` requests `TensorParallelExperts` across two ROCm participants, but the production overlay path still lowers that domain to the primary participant only.
- The `hot` tier lowers to `ROCm:0`, but `ExpertGemmRegistry` does not contain all active gate/up/down expert engines for that tier.
- `cpu_cold` requests `TensorParallelExperts` across two CPU participants, but the production overlay path still lowers that domain to the primary participant only.

These skips do not mean production overlay inference is complete. Later phases must remove the runtime blockers before the real parity bodies can count as passing inference parity.