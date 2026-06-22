# MoE Expert Overlay Phase 5A Audit

**Date:** 2026-05-09
**Scope:** Bridge Phase 5A from [docs/v2/projects/2026-06/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): audit and stabilize the current multi-domain lowering checkpoint.

## Verdict

Phase 5A is accepted with caveats. The current checkpoint has useful support code from later bridge phases, but real Qwen3.5 overlay parity is still intentionally blocked. The important Phase 5A contract is now explicit: CPU NodeLocalTP fallback can be graph-integrated, while active accelerator LocalTP routed tiers fail fast and are not lowered to their primary participant.

## Accepted as Phase 5 Support Code

- Runtime domain resolution in `MoEExpertOverlayRuntimePlan` records participants, primary devices, continuation/shared domains, and pending multi-participant execution diagnostics.
- Qwen3.5 overlay graph lowering creates `MoEExpertDispatchStage`, per-tier partials, graph-integrated CPU fallback stages for CPU NodeLocalTP domains, and a host dense reduction bridge.
- CPU fallback helper/stage coverage is accepted for CPU fallback domains only. It proves the CPU NodeLocalTP bridge shape and TensorParallelExperts math for CPU fallback, not accelerator LocalTP support.
- Accelerator LocalTP routed tiers are explicitly guarded. If an active routed tier resolves to a multi-participant non-CPU domain, graph construction throws instead of using `participants.front()`.
- Parity topology tests remain topology/audit tests only. Real prefill/decode parity bodies either run real parity or report an explicit bridge-phase blocker.

## Deferred Bridge-Phase Work

- Bridge Phase 5B: production accelerator residency and `ExpertGemmRegistry` closure. Current preparation diagnostics and registry subset helpers are support code, not proof that all planned accelerator experts are resident.
- Bridge Phase 5C: accelerator LocalTP TensorParallelExperts runtime. ROCm LocalTP domains still do not execute sharded expert GEMM/allreduce.
- Bridge Phase 5D: Qwen graph integration for accelerator multi-participant routed tiers. The graph currently fails fast for active ROCm LocalTP tier work.
- Bridge Phase 6A: production dispatch descriptor consumption and sparse transfer. Dispatch and token-row transfer helpers are model-light/bridge helpers until wired into production tier compute.
- Bridge Phase 7A: cross-domain reduce back to continuation domain. The current dense reducer is a host correctness bridge, not the final continuation-domain reduce.
- Bridge Phase 8A: real V2 overlay parity unskip. The parity suite must not be counted as final inference proof while these bridge blockers remain.
- Bridge Phases 9A-11A: profiling, release benchmark, and final cleanup remain out of scope for Phase 5A.

## Parity Blocker Mapping

The real `PrefillParity_*` and `DecodeParity_*` bodies currently skip with Bridge Phase 5C/5D accelerator LocalTP blocker messages:

- Layout A (`ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`): `rocm_shared_hot` requests TensorParallelExperts across two ROCm participants.
- Layout B (`CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`): `rocm_hot` requests TensorParallelExperts across two ROCm participants.

These skips are intentional for Phase 5A. They prevent topology-only coverage, model-light dense reduction, or synthetic fallback from being mistaken for real overlay parity.

## Test Results

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(Qwen35MoEExpertOverlayGraph|MoEExpertOverlayRuntimePlan|MoEExpertDispatchStage|MoEExpertParallelReduceStage)" --output-on-failure --parallel
```

Result: 5/5 CTests passed, 0 failed. This includes `V2_FetchModelsFixture` plus the four targeted unit suites.

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts|MultiAcceleratorTiers)" --output-on-failure --parallel
```

Result: 4/4 CTests passed, 0 failed. CPU fallback and CPU TensorParallelExperts MPI integration are accepted for the CPU fallback bridge surface.

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result: 7/7 CTests passed, 0 failed. The topology tests passed; the four real parity bodies are still GTest-skipped by explicit Bridge Phase 5C/5D accelerator LocalTP blockers.

Additional verbose real-parity audit:

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_(PrefillParity|DecodeParity)_" --verbose --parallel
```

Result: 5/5 CTests passed, 0 failed. Root-rank real parity bodies reported the expected Bridge Phase 5C/5D accelerator LocalTP blocker messages.

## Next Phase

Proceed to Bridge Phase 5B: production accelerator residency and registry closure. Do not advance to accelerator LocalTP runtime until the planned accelerator expert handles are prepared and diagnosable per tier/domain/device.