# MoE Expert Overlay Phase 6A Audit

**Date:** 2026-05-09
**Scope:** Bridge Phase 6A from [docs/v2/projects/2026-06/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): production dispatch descriptor consumption and sparse token-row transfer.

## Verdict

Phase 6A is accepted. The production Qwen3.5 MoE overlay graph now shares the `MoEExpertDispatchStage` descriptor output with non-continuation routed tiers and lowers those tiers through descriptor-aware CPU fallback and accelerator `LocalTP` stages. Sparse token-row transfer is the default production path, while dense full-sequence transfer remains available only through an explicit debug compatibility knob.

## Accepted Implementation Surface

- Qwen3.5 MoE graph construction keeps a shared `MoEExpertDispatchOutput` lifetime for each overlay layer and passes it into CPU fallback and accelerator `LocalTP` routed tier stages.
- `MoEExpertOverlayCPUFallbackStage` consumes per-tier dispatch descriptors, selected token rows, transfer mode, and routing metadata instead of independently re-filtering the full sequence.
- `MoEExpertOverlayLocalTPStage` consumes the same dispatch descriptors, gathers compact selected rows for sparse modes, executes the domain-scoped `MoEExpertOverlayLocalTPExecutor`, and scatter-adds compact partials back to the full output rows.
- `MoEExpertTokenRowTransfer` now accepts capacity-sized tensors whose first dimension is at least the live selected-row count, which matches the compact transfer buffers used by the production graph path.
- Diagnostics include transfer mode, selected row count, transfer volume, and routed-entry information; `LLAMINAR_MOE_EP_TRANSFER_TRACE=1` enables explicit transfer tracing.
- `LLAMINAR_MOE_EP_DENSE_TRANSFER=1` selects dense full-sequence compatibility mode for audit and debugging. The default path remains sparse/auto.

## Remaining Deferred Work

- Bridge Phase 7A: add the production cross-domain reducer that returns all shared, accelerator, and CPU fallback partials to the continuation domain.
- Bridge Phase 8A: unskip real V2 overlay parity once the reducer closes the remaining production return-path gap.
- Optimized continuation-domain reduction and release-performance gates remain later work; host-staged correctness mode must be reported clearly when used.

## Test Results

```bash
cmake --build build_v2_integration --parallel
```

Result: passed. The build relinked the affected parity executable and rediscovered the overlay tests.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Dispatch|TokenRowTransfer|ParallelReduce)" --output-on-failure --parallel
```

Result: 3/3 CTests passed, 0 failed. There is no separate `TokenRowTransfer` CTest target in this build; token-row transfer coverage is exercised through the sparse overlay integration tests below.

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(Sparse|Transfer|CPUFallback|MultiAcceleratorTiers)" --output-on-failure --parallel
```

Result: 5/5 CTests passed, 0 failed.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
```

Result: 2/2 CTests passed, 0 failed.

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result: 7/7 CTests passed, 0 failed. Real prefill/decode parity bodies still report GTest skips, now due to the Bridge Phase 7A/8 reducer blocker rather than missing sparse dispatch/transfer.

Additional log audit:

```bash
rg "Bridge Phase 7A|Bridge Phase 8|sparse dispatch/transfer|production sparse dispatch|final cross-domain reducer|accelerator residency blocker|Bridge Phase 5D|missing registry" build_v2_integration/Testing/Temporary/LastTest.log
```

Result: the only real-parity blocker is the expected Bridge Phase 7A/8 message: production sparse dispatch now lowers into accelerator `LocalTP`, but real parity remains skipped until the final cross-domain reducer returns all tier partials to the continuation domain. No stale sparse-dispatch, accelerator-residency, missing-registry, or Phase 5D graph-lowering blocker remains.

## Next Phase

Proceed to Bridge Phase 7A: add the production cross-domain reducer that returns shared, accelerator, and CPU fallback partials to the configured continuation domain.