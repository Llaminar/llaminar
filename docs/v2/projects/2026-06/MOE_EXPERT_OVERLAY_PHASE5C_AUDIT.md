# MoE Expert Overlay Phase 5C Audit

**Date:** 2026-05-09
**Scope:** Bridge Phase 5C from [docs/v2/projects/2026-06/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): accelerator `LocalTP` `TensorParallelExperts` runtime.

## Verdict

Phase 5C is accepted as a graph-independent accelerator `LocalTP` runtime checkpoint. The branch now has a domain-scoped `MoEExpertOverlayLocalTPExecutor` that validates a resolved `LocalTP` `TensorParallelExperts` runtime domain, shards each selected expert across participants by intermediate range, runs per-participant partial math, reduces the partials through `ILocalTPContext`, and records per-participant routed-entry diagnostics.

The production Qwen3.5 MoE graph still intentionally fails fast for active accelerator `LocalTP` routed tiers. That is the expected Bridge Phase 5D boundary.

## Accepted Implementation Surface

- `MoEExpertOverlayLocalTPExecutor` validates the runtime domain against the supplied `ILocalTPContext`, including participant count, backend, local addressability, and device ordering.
- The executor uses the same `TensorParallelExperts` math contract as the CPU fallback reference: gate/up intermediate sharding, SwiGLU on the local shard, down input-parallel partial output, domain allreduce, and route-weight application after reduction.
- The executor records domain name, backend, degree, per-participant device, intermediate shard range, executed expert IDs, routed-entry counts, and allreduce counts.
- `MoEExpertOverlayRuntimePlan` now marks accelerator `LocalTP` `TensorParallelExperts` domains as domain-context-ready instead of leaving them in the generic multi-participant pending state.
- The Qwen graph fail-fast remains in place and now points to Bridge Phase 5D graph wiring after the Phase 5C executor.

## Caveats

- The Phase 5C executor is intentionally graph-independent and synthetic-FP32 oriented. It proves sharded expert math and real accelerator-domain collective behavior, but it does not yet consume production Qwen graph dispatch descriptors or prepared quantized expert GEMM handles.
- The verbose ROCm test currently logs a backend-level warning/error when per-device async allreduce is attempted without registered compute streams, then falls back to the synchronous barrier allreduce path and succeeds. This is noisy but did not indicate a failed collective.
- Real Qwen overlay parity still skips because production graph wiring for active accelerator `LocalTP` tiers is deferred to Bridge Phase 5D.

## Test Results

```bash
cmake --build build_v2_integration --parallel
```

Result: passed, no work needed after the subagent build.

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

Result: 2/2 CTests passed, 0 failed.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlayRuntimePlan|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

Result: 3/3 CTests passed, 0 failed.

Additional verbose audit:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --verbose
```

Result: `RocmLocalTPTensorParallelExpertsExecutesBothParticipantsAndMatchesReference` ran on visible ROCm hardware and passed. The log shows two ROCm participants entering the `LocalTPContext` barrier allreduce path for the selected routed experts.

## Next Phase

Proceed to Bridge Phase 5D: wire the Phase 5C accelerator `LocalTP` executor into the real Qwen3.5 MoE overlay graph so active ROCm routed tiers are first-class runtime domains instead of graph-construction fail-fast blockers.