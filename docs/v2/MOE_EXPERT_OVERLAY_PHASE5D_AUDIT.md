# MoE Expert Overlay Phase 5D Audit

**Date:** 2026-05-09
**Scope:** Bridge Phase 5D from [docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md](MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md): Qwen graph integration for accelerator multi-participant routed tiers.

## Verdict

Phase 5D is accepted. Active accelerator `LocalTP` `TensorParallelExperts` routed tiers no longer fail graph construction solely because they have multiple participants. The Qwen3.5 MoE overlay graph now lowers those tiers through a domain-scoped LocalTP stage backed by the Phase 5C executor, while preserving the CPU `NodeLocalTP` fallback stage and the existing single-device accelerator routed tier path.

## Accepted Implementation Surface

- `MoEExpertOverlayLocalTPStage` wraps `MoEExpertOverlayLocalTPExecutor` as a graph stage for active accelerator `LocalTP` overlay tiers.
- Qwen3.5 MoE graph construction requires `GraphConfig::domain_tp_contexts[tier.domain]` for accelerator `LocalTP` tiers and fails clearly when the context is missing or mismatched.
- The old primary-participant fail-fast remains only for unsupported multi-participant domain shapes outside the Phase 5D accelerator `LocalTP` bridge.
- The new graph path carries per-tier diagnostics through the stage lifetime and supports host-backed mock LocalTP contexts for unit tests.
- Real parity skip wording now points past 5D to production sparse dispatch/transfer and final cross-domain reduce work.

## Remaining Deferred Work

- Bridge Phase 6A: consume production dispatch descriptors and sparse token-row transfer in the real Qwen overlay path.
- Bridge Phase 7A: replace helper-only dense reduction with production cross-domain reduction back to the continuation domain.
- Bridge Phase 8A: unskip real V2 overlay parity once dispatch/transfer/reduce blockers are closed.
- Later profiling, benchmark, hardening, cleanup, and documentation phases remain open.

## Test Results

```bash
cmake --build build_v2_integration --parallel
```

Result: passed, no work needed after the subagent build.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
```

Result: 2/2 CTests passed, 0 failed.

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

Result: 2/2 CTests passed, 0 failed.

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result: 7/7 CTests passed, 0 failed. Real prefill/decode parity bodies still report GTest skips, now due to Bridge Phase 7A/8 production overlay blockers: sparse dispatch/transfer and final cross-domain reducer are not yet production graph paths.

Additional log audit:

```bash
rg "Bridge Phase 5D|multi-participant|primary participant|Bridge Phase 7A/8|sparse dispatch|cross-domain reducer|accelerator residency blocker" build_v2_integration/Testing/Temporary/LastTest.log
```

Result: the only real-parity blockers are Bridge Phase 7A/8 dense compatibility messages for `rocm_hot` and `rocm_shared_hot`. No remaining skip names missing accelerator registry entries or accelerator multi-participant graph lowering.

## Next Phase

Proceed to Bridge Phase 6A: make the production Qwen overlay path consume dispatch descriptors and sparse token-row transfer for non-continuation domains, leaving the final cross-domain reducer for Bridge Phase 7A.