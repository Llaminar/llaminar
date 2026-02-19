# ROCm Kernel Lifecycle & Forward Graph Reuse Plan

## Goal
Eliminate per-forward kernel recreation so ROCm kernels can safely keep mutable state as instance members instead of process-global static workarounds.

## Problem Summary
- Today, many stages/kernels are effectively rebuilt outside decode-cache hits.
- Static state in ROCm kernels was introduced as a lifecycle workaround.
- Static mutable caches increase cross-device/thread interference risk in LOCAL TP.

## Scope
- Runtime path: `DeviceGraphOrchestrator` forward execution lifecycle.
- Primary target: stage/kernel reuse across forwards.
- Secondary target: migrate ROCm static mutable caches to instance-owned state once lifetime is stable.

## Non-Goals (Phase 1)
- No broad redesign of execution APIs.
- No behavior changes for PP/TP scheduling semantics.
- No performance tuning beyond preventing obvious redundant rebuilds.

## Phases

### Phase 1 — Forward Graph Signature Cache (Start Here)
Objective: make full forward graphs reusable by signature, not only one decode case.

Tasks:
1. Add `GraphSignature` key type in `DeviceGraphOrchestrator` (mode + shape + path flags).
2. Replace single `forward_cache_` usage path with signature-based lookup.
3. On cache miss, build graph once, then store by signature.
4. Keep dynamic updates per-forward (`token_ids`, `position_ids`, `updateDynamicParams`) exactly as today.
5. Preserve current invalidation behavior when mode/path changes.

Exit Criteria:
- Repeated forwards with same signature do not rebuild stages.
- Existing decode fast path behavior remains intact.

### Phase 2 — Session Lifecycle Hooks
Objective: explicit lifecycle boundaries without tearing down reusable stage objects.

Tasks:
1. Add orchestrator-level hooks (`onSessionReset`, `onTopologyChange`) and connect invalidation.
2. Use reset hooks for sequence/KV-position state while keeping cached graphs where valid.
3. Restrict hard invalidation to topology/device/layout changes.

Exit Criteria:
- Session reset no longer forces unnecessary graph/stage destruction.
- Topology changes safely invalidate affected cache entries.

### Phase 3 — ROCm Static State Removal
Objective: migrate mutable static ROCm kernel state to instance members.

Tasks:
1. Inventory per-kernel static mutable caches (Embedding, RoPE, etc.).
2. Move ownership into kernel/stage instances.
3. Ensure thread-safe instance usage where kernels are shared across worker threads.

Exit Criteria:
- Target ROCm kernels have no mutable process-global static cache state.
- LOCAL TP race surface is reduced and observable in logs/tests.

## Tracking Checklist

### Phase 1 Checklist
- [x] Add graph signature structure and hashing.
- [x] Add cache map keyed by signature.
- [x] Refactor execute path to lookup by signature.
- [x] Insert/store built graphs by signature.
- [x] Validate build + smoke test.
- [x] Expand signature for partial PP single-device path.

### Phase 2 Checklist
- [ ] Add lifecycle reset hooks.
- [ ] Wire invalidation policy by reset cause.
- [ ] Validate cache retention across session reset.

### Phase 3 Checklist
- [x] Refactor Embedding static state.
- [x] Refactor RoPE static state.
- [ ] Remove remaining mutable static ROCm caches.
- [ ] Run targeted LOCAL TP regression tests.

## Risks
- Cache key too coarse may replay invalid graph; too fine may reduce reuse.
- Unified PP / nested TP paths may require conservative cache eligibility.
- Additional cached graphs may increase memory pressure.

## Validation Plan
1. Build: `cmake --build build_v2_release --parallel`
2. Targeted run(s): LOCAL TP repro scenario with logging enabled.
3. Compare before/after for:
   - Graph rebuild frequency
   - Stage/kernel construction frequency
   - Correctness/no new crashes

## Notes
- Keep changes incremental and guarded by existing path checks.
- Prefer preserving current behavior first, then broadening reuse safely.
