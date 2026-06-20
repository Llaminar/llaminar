# MoE Expert Overlay Phase 5B Audit

**Date:** 2026-05-09
**Scope:** Bridge Phase 5B from [docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): production accelerator residency and `ExpertGemmRegistry` closure.

## Verdict

Phase 5B is accepted. Planned accelerator routed experts are now prepared through the production runner/model-load path and registered under domain-scoped `ExpertGemmRegistry` keys. The real Qwen3.5 overlay parity bodies still skip, but the skip boundary has advanced to Bridge Phase 5C/5D accelerator LocalTP execution and Qwen graph integration. No real parity skip is currently caused by missing accelerator registry entries.

## Accepted Implementation Surface

- `InferenceRunnerFactory` resolves requested overlay plans into model-aware planned placements before graph construction, then resolves runtime domains from that planned plan.
- Production weight preparation calls `WeightManager::prepareMoEExpertOverlayWeights()` whenever a resolved overlay runtime plan is present, while disabling broad all-expert preparation for the normal device pass.
- `MoEExpertOverlayPreparationPlan` builds per-layer, per-expert, per-role, per-domain, per-device preparation requests from the planned overlay placement.
- The GPU weight pipeline filters MoE expert slots through the overlay preparation plan and registers accelerator handles with `ExpertGemmRegistry::registerEngineForDomain()`.
- CPU fallback routed experts remain host-owned and are not inserted into the accelerator expert registry.
- `ExpertGemmRegistry` supports domain-scoped lookup, completeness checks, counts, replacement, removal, and population so logical domains sharing a physical device cannot overwrite each other.
- Qwen3.5 overlay graph construction looks up active GPU routed expert handles by logical domain and emits placement-specific diagnostics including tier, domain, participant, device, expert, and role.

## Remaining Deferred Work

- Bridge Phase 5C: real accelerator `LocalTP` / `TensorParallelExperts` execution for multi-participant ROCm domains.
- Bridge Phase 5D: Qwen graph integration for active accelerator multi-participant routed tiers.
- Later bridge phases still own production sparse transfer, cross-domain reduce, parity unskip, profiling, release benchmark, and final cleanup.

## Test Results

```bash
cmake --build build_v2_integration --parallel
```

Result: completed successfully, 513/513 build steps.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(WeightManager|ExpertGemmRegistry|MoEExpertOverlayPreparation|InferenceRunnerFactory|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

Result: 16/16 CTests passed, 0 failed. This includes `V2_FetchModelsFixture`, the domain-scoped registry suite, overlay preparation planning, production factory planning, and Qwen3.5 overlay graph coverage.

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result: 7/7 CTests passed, 0 failed. The topology CTests pass as audit coverage. The four real prefill/decode parity bodies still report GTest skips, now due to Bridge Phase 5C/5D accelerator LocalTP blockers only.

Additional log audit:

```bash
rg "Bridge Phase 5B|accelerator residency blocker|missing registry|does not contain all active gate/up/down|missing active masked expert|incomplete ExpertGemmRegistry" build_v2_integration/Testing/Temporary/LastTest.log
```

Result: no matches. The parity log contains `WeightManager` registrations for planned MoE expert GEMM kernels and real-parity skip messages only for Bridge Phase 5C/5D accelerator LocalTP execution.

## Next Phase

Proceed to Bridge Phase 5C: accelerator `LocalTP` `TensorParallelExperts` runtime. The next implementation should consume the domain-scoped accelerator handles already prepared by Phase 5B and replace the current active accelerator multi-participant fail-fast boundary with a real sharded expert executor.