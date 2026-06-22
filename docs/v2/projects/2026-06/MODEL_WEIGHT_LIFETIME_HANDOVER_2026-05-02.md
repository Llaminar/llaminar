# Model Weight Lifetime Redesign Handover

**Date**: 2026-05-02  
**Branch**: `feat/qwen35-moe`  
**Latest pushed commit**: `d1156d05 Implement model weight lifetime redesign`  
**Parent plan**: `docs/v2/projects/2026-06/MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md`

## Current State

The first implementation pass for the Model Weight Lifetime Redesign has landed and was pushed to `llaminar/feat/qwen35-moe` as `d1156d05`. It introduced the additive lifecycle scaffolding and compatibility hooks needed for the next phases:

- `WeightIdentity`, `WeightSliceSpec`, `WeightResidency`, and prepared-ref metadata in `src/v2/loaders/WeightIdentity.*`.
- `WeightMetadataRegistry` sidecar metadata storage in `src/v2/loaders/WeightMetadataRegistry.*`.
- `WeightLifecycleTrace` instrumentation in `src/v2/loaders/WeightLifecycleTrace.*`.
- `WeightPlan`, `WeightRequirement`, `FrozenModelWeightSet`, and binding builder scaffolding in `src/v2/loaders/WeightPlan.*`.
- `PreparedWeightStore` compatibility wrapper over existing `KernelFactory` prepared-GEMM APIs in `src/v2/loaders/PreparedWeightStore.*`.
- `WeightManager::materialize(const WeightPlan&)` compatibility materialization path.
- Unit tests in:
  - `tests/v2/unit/loaders/Test__WeightIdentity.cpp`
  - `tests/v2/unit/loaders/Test__WeightPlan.cpp`
  - `tests/v2/unit/loaders/Test__PreparedWeightStore.cpp`

The implementation is still a compatibility layer. Production graph builders still mostly bind raw `TensorBase*` through existing `ModelWeights` callbacks, and stage execution still resolves prepared GEMM through `KernelFactory` in many paths. The new lifecycle types are present, tested, and partially integrated, but Phases 6-10 of the parent plan remain unfinished.

## Validation Already Run

Before commit `d1156d05`, the following gates were run successfully except for the noted benchmark variance:

- Full Integration build: passed.
- Full V2 unit suite: passed, including 429 unit tests at the time of the run.
- Requested PyTorch parity gate across SingleDevice, LocalTP, LocalPP, NodeLocalTP, and HybridPPTP Prefill/Decode cases: passed 157/157 after fixing a deterministic MoE LocalTP ROCm decode crash.
- E2E server tests in the precommit hook: passed 50/50.
- Precommit performance gate was bypassed with `--no-verify` by explicit human approval because Qwen2.5 7B CPU prefill was noisy around the 10% threshold. Other benchmark metrics were within bounds.

Relevant benchmark blocker at the time:

```text
Qwen 2.5 7B Instruct Q8_0 cpu prefill: 337.99 -> 300.60 tok/s (-11.1%, threshold 10%)
```

## Workspace Caveat

At the time this handover was written, there was an unrelated dirty file in the workspace:

```text
M tests/v2/e2e/server/test_server_e2e.sh
```

Do not assume it belongs to the model weight lifetime work unless you inspect it and intentionally include it.

## Known Shortcut: Raw MoE Expert Host Data Retention

The last implementation pass fixed a deterministic crash in Qwen35MoE LocalTP ROCm decode by taking a deliberately conservative shortcut: raw host data for MoE expert tensors is retained under dynamic MoE expert rebalancing.

The shortcut lives in `src/v2/loaders/WeightManager.cpp`:

```cpp
bool shouldRetainRawForLazyMoE(const std::string &key)
{
    const auto &env = debugEnv();
    return env.moe_rebalance.mode == "dynamic" &&
           !env.moe_rebalance.release_raw_weights &&
           key.find("_exps.weight") != std::string::npos;
}
```

It is consulted by:

- `WeightManager::releaseAllHostWeightData()`
- `WeightManager::releaseHostResidentWeightData()`

This prevents crashes when `MoEExpertComputeStage::ensureGemmEnginesForExperts()` lazily requests an expert that was not prepared yet and calls:

```cpp
MoEExpertWeightService::registerAndPrepareNewExperts(ctx, needed, nullptr)
```

For GPU devices, that falls into `MoEExpertWeightService::registerAndPrepareNewExpertsGPU()`. If no transferred packed expert blob is provided, it falls back to `view->raw_data()` and `quantizedViewRawBytes(*view)`. Without retained raw host data, `view->raw_data()` is null and the decode path fails.

### Why This Is Bad

For Qwen3.5 MoE models, `_exps.weight` tensors are huge. Retaining all raw expert host tensors severely bloats CPU RAM and defeats one of the main goals of the lifetime redesign: explicit, minimal host retention based on declared consumers.

This shortcut should be removed as soon as the proper expert transfer/materialization path is in place.

## Proper Fix Plan For Unwinding The Shortcut

The correct fix is not to keep every raw MoE expert tensor alive. The correct fix is to make dynamic expert migration/rebalancing use explicit prepared or transferable expert payloads, with host retention scoped only to the small set of experts that are not yet prepared and cannot be transferred from another owner.

### Goal

Remove the unconditional `_exps.weight` retention in `shouldRetainRawForLazyMoE()` while preserving:

- Qwen35MoE LocalTP ROCm decode parity.
- HybridPPTP GPU expert cache migration.
- Dynamic MoE expert rebalancing.
- Host RAM release after graph/preparation boundaries.

### Recommended Implementation Sequence

1. **Make missing GPU experts require a source payload**

   Update `MoEExpertComputeStage::ensureGemmEnginesForExperts()` so GPU lazy preparation cannot silently fall back to raw source tensors after host release. Instead of calling `registerAndPrepareNewExperts(ctx, needed, nullptr)`, route through an explicit provider:

   ```cpp
   const std::unordered_map<int, ExpertWeightBlobs>* received = ...;
   MoEExpertWeightService::registerAndPrepareNewExperts(ctx, needed, received);
   ```

   If no prepared handle, transferred blob, or declared retained raw source exists, fail during graph/materialization with a diagnostic that includes layer, expert id, device, and binding identity.

2. **Add an ExpertWeightPayloadProvider owned by the runtime/model context**

   Introduce a small service, likely near `src/v2/execution/moe/` or `src/v2/loaders/`, that can answer:

   ```cpp
   std::optional<ExpertWeightBlobs> payloadFor(layer, expert, target_device);
   bool hasPreparedExpert(layer, expert, target_device);
   ```

   It should use the existing serialization path in `MoEExpertWeightService::serializeExpert()` / `detachAndSerializeExpert()` and the existing `ExpertWeightBlobs` type. The provider should be model-context or orchestration-owned, not stage-local.

3. **Pre-materialize transferable expert blobs at rebalance boundaries**

   `DeviceGraphOrchestrator` already has a dynamic rebalance path around `registerAndPrepareNewExperts(masks[layer], layer_received)`. Extend this so expert movement always supplies `layer_received` blobs for experts that are newly assigned to a GPU device.

   Start from:

   - `DeviceGraphOrchestrator.cpp`, around the dynamic MoE rebalance logic that calls `stage->registerAndPrepareNewExperts(...)`.
   - `MoEExpertWeightService::registerAndPrepareNewExpertsGPU()`.
   - `MoEExpertWeightService::makeTransferSource()` inside the GPU rebalance path.

4. **Represent MoE expert host policy per binding**

   Extend the current `WeightPlan`/`WeightBinding` usage for MoE expert tensors so expert bindings can declare one of these policies:

   - `RequiredForCPUExecution`: CPU-owned or CPU-executed experts.
   - `RequiredUntilGraphMaterialized`: temporary raw bytes needed only while graph/stage views are created.
   - `RequiredUntilPreparedOrTransferred`: new policy or equivalent state for expert tensors whose packed transferable blob has not been created yet.
   - `ReleasableAfterPreparation`: GPU experts once prepared refs or transferable blobs exist.

   The parent plan already introduced `WeightHostPolicy`; use it instead of string matching on `_exps.weight`.

5. **Move raw retention decision from string matching to metadata**

   Replace `shouldRetainRawForLazyMoE(key)` with a binding/metadata-aware query, for example:

   ```cpp
   bool WeightManager::hostDataRequired(const TensorBase* tensor) const;
   ```

   It should inspect `WeightMetadataRegistry` / `WeightBinding` state, not the cache key string. It must be able to say exactly why host data is retained.

6. **Add a release-after-transfer gate**

   Once an expert has either:

   - a model-owned prepared ref in `PreparedWeightStore`, or
   - a serialized transferable packed blob registered with the payload provider,

   mark its source tensor host data releasable. Then `releaseAllHostWeightData()` can free it normally.

7. **Add strict diagnostics before removing fallback**

   Before deleting the raw fallback entirely, add an Integration-only assertion/log path in `registerAndPrepareNewExpertsGPU()`:

   - If `received_weights == nullptr` and `view->raw_data() == nullptr`, log a structured error with model/layer/expert/weight role/device/binding id.
   - If `view->raw_data() != nullptr` after graph freeze, warn that the old fallback path is still in use.

8. **Remove the shortcut**

   Delete or narrow `shouldRetainRawForLazyMoE()` so it no longer retains all `_exps.weight` tensors. The only allowed retained raw expert data should be explicitly represented by binding host policy.

### Acceptance Tests For The Proper Fix

Run these before and after removing the shortcut:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R '^V2_Unit_' --output-on-failure --parallel
```

Then run the MoE parity tests that exercise this path:

```bash
ctest --test-dir build_v2_integration -R 'Qwen35MoE.*LocalTP.*DecodeParity' --output-on-failure
ctest --test-dir build_v2_integration -R 'Qwen35MoE.*HybridPPTP.*DecodeParity' --output-on-failure
ctest --test-dir build_v2_integration -R 'Qwen35MoE.*NodeLocalTP.*DecodeParity' --output-on-failure
```

Also rerun the full requested parity gate from the parent task:

```bash
ctest --test-dir build_v2_integration -R 'V2_Integration_Parity_.*(SingleDevice|LocalTP|LocalPP|NodeLocalTP|HybridPPTP).*(PrefillParity|DecodeParity)' --output-on-failure
```

For RAM validation, compare resident memory on Qwen35MoE LocalTP/HybridPPTP before and after removing the shortcut. The expected result is that raw `_exps.weight` host data drops after graph/preparation/transfer materialization while decode parity remains stable.

Suggested diagnostic runs:

```bash
LLAMINAR_WEIGHT_LIFECYCLE_TRACE=1 \
LLAMINAR_MOE_REBALANCE=dynamic \
./build_v2_integration/tests/v2/v2_integration_parity_qwen35moe_local_tp \
  --gtest_filter='*DecodeParity*'

LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1000000 \
./build_v2_release/llaminar2 benchmark -d rocm:0 \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

## Remaining Parent-Plan Work

The landed implementation mostly covers M1/M2 scaffolding and a Phase 4 compatibility shim. The next agent should not treat the plan as complete. Remaining high-value work:

1. Build mode-complete `WeightPlan` generation from schema + execution strategy, not ad hoc requirements in tests.
2. Make graph construction consume `FrozenModelWeightSet` / `ModelWeightBindings` rather than `WeightManager::getWeightForDevice()` callbacks.
3. Thread `PreparedWeightStore` into `ModelContext` ownership and GPU pipeline registration instead of using `KernelFactory` as primary truth.
4. Convert GEMM-bearing stages to carry `PreparedWeightRef` / binding ids, then assert graph replay does not call preparation APIs.
5. Replace host release timing flags with binding lifecycle states and completion gates.
6. Slim `KernelFactory` into kernel selection plus compatibility shims after prepared ownership is model-context-owned.

## Suggested Next Agent Starting Point

Start with the host raw expert release shortcut, because it is a correctness/performance debt introduced to unblock parity:

1. Reproduce current behavior with Qwen35MoE LocalTP decode and memory tracing.
2. Add payload-provider plumbing so GPU expert preparation receives `ExpertWeightBlobs` rather than using raw tensor fallback.
3. Add binding host policies for MoE expert tensors.
4. Replace `shouldRetainRawForLazyMoE()` with metadata-based retention.
5. Validate memory reduction and parity.

Only after that should the agent proceed to deeper graph binding migration.