# Model Weight Lifetime Redesign Audit

**Date**: 2026-05-06
**Branch**: `feat/qwen35-moe`
**Plan audited**: `docs/v2/cleanup/MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md`

## Executive Summary

The redesign is materially advanced but not complete. The core identity, planning, frozen binding, and `PreparedWeightStore` ownership scaffolding exists, and the old `KernelFactory` prepared-GEMM registry API has been removed from production/test source. However, the final target contract is not fully satisfied for multi-device modes.

The most important remaining work is to make LocalTP, LocalPP, Global/NodeLocalTP, and Hybrid PP+TP use explicit plan/materialize/prepare/freeze flows before graph construction. Today several paths still rely on live `WeightManager` callbacks or stage-local preparation sequencing rather than one immutable model-wide weight plan.

## Phase Status

| Phase | Status | Evidence |
|---|---|---|
| 0: Baseline audit and lifecycle trace | Partial/implemented | `WeightLifecycleTrace` exists and is wired in loader/prepared-store paths. The trace does not yet prove every mode has a complete binding audit. |
| 1: Weight identity metadata | Implemented | `WeightIdentity`, `WeightSliceSpec`, `WeightResidency`, `WeightLifecycleState`, and `WeightMetadataRegistry` exist under `src/v2/loaders/`. |
| 2: WeightPlan and strategy planning | Partial | `WeightPlan` and `InferenceStrategy` exist, but dedicated LocalTP, LocalPP, GlobalTP, and Hybrid PP+TP plan builders are incomplete or not consistently used. |
| 3: Frozen weight bindings | Partial | `FrozenModelWeightSet` exists and SingleDevice uses it. PP and nested TP paths still use compatibility `ModelWeights` callbacks before/around freezing. |
| 4: PreparedWeightStore compatibility layer | Implemented | `PreparedWeightStore` owns prepared GEMM, embedding, fused, sliced, and expert-slab state. |
| 5: GPU pipeline registration in store | Partial | GPU pipeline registers store-owned prepared handles, but Hybrid PP+TP can still reach graph construction with missing refs. |
| 6: Graph binding API migration | Partial | `ModelWeightBindings` and `PreparedWeightRef` stage params exist. Legacy `buildWeights(WeightAccessor)` adapters remain for PP/new-model compatibility. |
| 7: Stage execution through prepared refs | Mostly implemented | GEMM-bearing graph stages resolve through `PreparedWeightStore`. The remaining risk is missing refs in multi-device graph construction, not stage fallback to `KernelFactory`. |
| 8: Slim KernelFactory | Mostly implemented | Deleted prepared-GEMM registry APIs are absent. Device-scoped non-weight kernel caches remain. Low-level caller-owned creators remain for `PreparedWeightStore`. |
| 9: Host release/freeze semantics | Partial | Lifecycle gates exist, but release is still coarse and partly tensor-marker based (`has_prepared_device_state_`) rather than fully binding-policy based. |
| 10: Remove compatibility APIs | Partial | The old prepared-GEMM factory fallback is gone. Callback-based weight access, TensorBase prepared-state markers, and compatibility direct `KernelFactory::clearCacheFor()` test calls remain. |

## Remaining Blocking Gaps

1. **Mode-specific weight plans are incomplete.**
   SingleDevice is the cleanest path. LocalTP/GlobalTP need TP-domain-aware `WeightPlan` builders. LocalPP needs a plan that includes stage ownership for embeddings, layer ranges, final norm, and LM head. Hybrid PP+TP needs both PP stage id and nested TP device/rank identity on every binding.

2. **PP and unified pipeline still use live weight access callbacks.**
   `configurePPStageWeightsImpl()` and unified pipeline setup still construct `ModelWeights` through callbacks built from `WeightManager`/`ModelContext` accessors. This is a compatibility bridge, not the final contract.

3. **Hybrid PP+TP prepared refs are not complete.**
   The current failing pre-commit path is `Qwen35MoEHybridPPTPParityTest.PrefillParityWithGpuExpertCache`, which reaches `layer20_gdn_proj` on `ROCm:1` without a required prepared ref. This should be fixed by model-wide Hybrid PP+TP materialization/preparation, not by graph-time fallback preparation.

4. **Host release remains too coarse.**
   `WeightResidency::host_policy` and lifecycle gates exist, but host release decisions still depend partly on tensor-level flags and broad lifecycle sequencing. The final design requires binding-level host-policy transitions.

5. **Compatibility markers remain.**
   `TensorBase::has_prepared_device_state_` remains in use for host release. It is acceptable as a temporary compatibility marker but should not be the final truth for model-owned prepared state.

## Work Completed During This Audit

- Removed the experimental graph-time prepared-weight fallback (`prepareGraphBindingForDevice`) because it contradicted the final contract and did not fix Hybrid PP+TP.
- Removed the canonical-name prepared-ref fallback (`preparedRefForName`) so missing binding identity cannot be hidden by name-only lookup.
- Restored strict `PreparedWeightStore` model-id validation.
- Removed the temporary prepared-GEMM grep guard after it served its diagnostic cleanup purpose; binding-first prepared-weight behavior is now covered by focused unit/regression tests.

## Current Verification

Passing:

```bash
cmake --build build_v2_integration --target v2_unit_prepared_weight_store v2_test_device_graph_orchestrator --parallel
ctest --test-dir build_v2_integration --output-on-failure -R 'V2_Unit_(PreparedWeightStore|DeviceGraphOrchestrator)$'
```

Known failing before completion:

```bash
ctest --test-dir build_v2_integration --output-on-failure -R 'Qwen35MoEHybridPPTPParityTest_PrefillParityWithGpuExpertCache'
```

The failure is still missing `PreparedWeightRef` metadata for `layer20_gdn_proj` on the ROCm TP worker.

## Completion Plan

1. Add mode-specific `WeightPlan` builders for LocalTP and LocalPP first. These should populate `pp_stage`, `tp_domain`, `tp_rank_or_device_index`, device residency, and expected prepared kind.
2. Convert `configurePPStageWeightsImpl()` to materialize/freeze from a PP-stage `WeightPlan` before calling `setFrozenWeightSet()`.
3. Convert Hybrid PP+TP runner construction to materialize the nested PP-stage/TP-device binding set before building any worker graph.
4. Make `PreparedWeightStore` registration consume those frozen bindings only; do not prepare missing GPU weights from graph build.
5. Replace host release checks based on tensor markers with binding-level host policy transitions.
6. Remove or demote `TensorBase::has_prepared_device_state_` once host release no longer depends on it.

The project is not fully complete until the Hybrid PP+TP parity gate above passes without graph-time preparation and the full pre-commit hook is green.