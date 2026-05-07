# KernelFactory Prepared GEMM Fallback Removal Plan

**Date**: 2026-05-05  
**Status**: Implemented 2026-05-06  
**Scope**: Remove model-stage and workspace-consumer compatibility paths that lazily prepare model GEMM weights through `KernelFactory`, then delete `KernelFactory::getOrCreatePreparedGemmWeights()` itself.  
**Parent plan**: `docs/v2/cleanup/MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md`  
**Primary files**: `src/v2/loaders/PreparedWeightStore.*`, `src/v2/kernels/KernelFactory.*`, `src/v2/execution/compute_stages/`, `src/v2/models/`, `tests/v2/`

---

## Executive Summary

**Implementation outcome (2026-05-06)**: The migration is code-complete. `KernelFactory::getOrCreatePreparedGemmWeights()` has been deleted from `KernelFactory.h/.cpp`; no C++ production or test code under `src/v2` or `tests/v2` references it. Model GEMM stages validate `PreparedWeightStore`/`PreparedWeightRef` availability before execution, workspace consumer paths resolve through the same store-backed refs, and standalone stage tests use explicit prepared-weight fixtures.

The remaining `KernelFactory` prepared APIs are intentionally narrower low-level helpers: pipeline registration/lookup for already-prepared transferred weights, caller-owned local handle creation for non-model tests, sliced CPU kernel creation, fused adapter creation from prepared handles, and embedding preparation while embedding store migration remains separate from this GEMM fallback removal.

Phase 6 and Phase 7 of the model weight lifetime redesign have moved graph-built model paths toward frozen `WeightBinding` and `PreparedWeightRef` use. The remaining cleanup is to remove the compatibility behavior that still lets standalone tests, direct stage construction, and workspace planning silently fall back to `KernelFactory::getOrCreatePreparedGemmWeights()`.

The migration target is strict:

1. Model graph construction passes prepared refs for every model-weight-backed GEMM stage.
2. Stage execution and workspace planning resolve model GEMM kernels only through `PreparedWeightStore`.
3. Standalone and direct stage tests use explicit prepared-store test fixtures instead of relying on lazy factory fallback.
4. Sliced GEMM, fused gate/up GEMM, embeddings, shared experts, and MoE expert slabs are all reachable through store-owned APIs.
5. The final mandatory gate deletes `KernelFactory::getOrCreatePreparedGemmWeights()` from `KernelFactory.h` and `KernelFactory.cpp`, and the full required test set still passes because nobody calls it anymore.

This plan intentionally does not remove every `KernelFactory` responsibility. Kernel selection, low-level non-model kernels, device helpers, and direct low-level kernel tests can continue to use a renamed or slimmed registry/factory. What must disappear is model-weight lifetime ownership and all lazy prepared-GEMM creation through the global factory method.

---

## Completed Compatibility Surface

These areas were migrated before the factory method was deleted.

### Phase 0 Baseline Inventory

The initial inventory was captured on 2026-05-05 with:

```bash
rg "getOrCreatePreparedGemmWeights|getOrCreateGemmSliced|getKernelAsWorkspaceConsumer" src/v2 tests/v2
```

The Phase 0 static guard originally ratcheted production stage fallback calls to `KernelFactory::getOrCreatePreparedGemmWeights()`. That grep-based guard was intentionally temporary and has been removed after the cleanup; the durable contract is enforced by binding-first prepared-weight wiring and regression tests.

Final stage fallback baseline:

| File | Allowed calls | Category |
|------|---------------|----------|
| all stage files | 0 | deleted API is forbidden |

Final classification:

- `PreparedWeightStore.cpp`: uses caller-owned local preparation helpers and owns GEMM/fused/sliced caches.
- `WeightManager.cpp`: uses caller-owned local GEMM handles for compatibility preload paths.
- `KernelFactory.*`: no deleted method declaration/definition remains; local helper APIs remain for low-level use.
- `IWorkspaceConsumerStage.h`, `TensorSlice.h`, ROCm/CUDA kernel headers: comments now point to `PreparedWeightStore` or local handle helpers.
- `tests/v2/`: model-stage tests use `PreparedWeightTestHarness`; old global prepared-key collision tests were removed or rewritten.

### Model Stage Execution Fallbacks

Current GEMM-bearing stages require store/ref-backed resolution for model weights:

- `src/v2/execution/compute_stages/stages/GEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/LMHeadStage.cpp`
- `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/GDNProjectionStage.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`

### Workspace-Consumer Fallbacks

Workspace planning now shares the execution resolver and cannot independently prepare model GEMM kernels through `KernelFactory`. The migrated methods include:

- `getKernelAsWorkspaceConsumer()`
- `getWorkspaceRequirements()`
- `bindWorkspace()`

These paths must share the same prepared-store resolver used by `execute()`. Workspace planning must not be a hidden preparation path.

### Store Coverage Gaps

`PreparedWeightStore` now owns the required GEMM surface:

- Plain GEMM lookup by `PreparedWeightRef`.
- Sliced GEMM lookup for tensor-parallel row/output ranges.
- Fused gate/up kernel lookup by prepared refs.
- Prepared embedding lookup and registration.
- Shared expert prepared refs.
- MoE expert slab lookup and rebalanced expert registration.
- Test-only explicit registration that does not depend on deleted production APIs.

### Tests and Workspace Consumers

Unit/integration tests now distinguish between:

- **Allowed**: low-level tests that exercise intentionally retained `KernelFactory` helpers such as transfer registration, sliced kernel creation, embedding preparation, or caller-owned local handle creation.
- **Migrated**: direct stage tests, graph tests, loader tests, MPI utility tests, and workspace-consumer tests no longer call the deleted factory method.
- **Removed or rewritten**: tests whose only subject was the deleted global prepared-GEMM registry behavior.

---

## Non-Negotiable Final Acceptance Gate

The migration is not complete until all of the following are true:

1. `KernelFactory::getOrCreatePreparedGemmWeights()` declaration is removed from `src/v2/kernels/KernelFactory.h`.
2. `KernelFactory::getOrCreatePreparedGemmWeights()` definition is removed from `src/v2/kernels/KernelFactory.cpp`.
3. No production or test C++ code calls the removed method.
4. All model-weight-backed GEMM stages require prepared refs/store access and fail initialization or execution if missing.
5. Workspace planning cannot lazily prepare model GEMM weights.
6. The required unit and integration gates pass after the method is deleted.

Final code-search gate:

```bash
rg "getOrCreatePreparedGemmWeights" src/v2 tests/v2
```

Expected result at final acceptance: no C++ declarations, definitions, calls, or comments in `src/v2` or `tests/v2` that depend on this removed API. Documentation may keep historical references only when clearly marked as obsolete.

Final build and test gate:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_"
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Integration_"
```

If hardware or model availability makes the full integration gate impossible in a local environment, the implementation handoff must list the skipped tests and run the strongest available targeted subset. The final project acceptance still requires the full gate in CI or an equivalent hardware environment.

---

## Phase 0: Baseline Inventory and Guardrails

**Goal**: Freeze the current call surface and add checks that prevent new lazy factory usage from sneaking in during the migration.

### Implementation Tasks

- Run and record the current call inventory:

```bash
rg "getOrCreatePreparedGemmWeights|getOrCreateGemmSliced|getKernelAsWorkspaceConsumer" src/v2 tests/v2
```

- Classify every match as one of:
  - production fallback to remove
  - workspace fallback to remove
  - store-internal temporary bridge
  - low-level factory test to rewrite or delete
  - documentation/comment to update
- Add a lightweight test or script target that fails if new stage files call `KernelFactory::getOrCreatePreparedGemmWeights()` directly.
- Add a short comment in `KernelFactory.h` marking the method as scheduled for deletion by this plan, not merely deprecated.
- Confirm the Phase 7 graph-built paths still pass existing focused tests before deeper changes.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_unit_prepared_weight_store v2_test_device_graph_orchestrator
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "V2_Unit_(PreparedWeightStore|DeviceGraphOrchestrator)$"
```

### Exit Criteria

- The current compatibility surface is documented in the implementation notes or handoff.
- No new production stage fallback can be added without tripping a search-based guard.
- Existing prepared-store and graph-binding tests pass.

---

## Phase 1: PreparedWeightStore API Completion

**Goal**: Make `PreparedWeightStore` capable of serving every model GEMM stage and workspace consumer without reaching back into the global factory method.

### Implementation Tasks

- Ensure `PreparedWeightStore::gemmKernel(const PreparedWeightRef&)` is the only plain GEMM resolver used by model stages.
- Complete or harden `PreparedWeightStore::slicedGemmKernel(...)` for tensor-parallel row/output ranges.
- Prefer a ref-based sliced API so stage code does not have to key on raw tensors:

```cpp
ITensorGemm* slicedGemmKernel(const PreparedWeightRef& ref,
                              size_t row_start,
                              size_t row_end);
```

- Keep a temporary tensor-based overload only as a transition helper, with tests proving it maps to a known prepared binding.
- Ensure `PreparedWeightStore::fusedGateUpKernel(gate_ref, up_ref)` can replace all fused gate/up fallback paths.
- Ensure embedding preparation and lookup are store-owned enough for follow-up removal of embedding prepared registry APIs.
- Ensure shared-expert and MoE expert code can get store-owned prepared kernels for all required gate/up/down weights.
- Add strict diagnostics for missing refs:
  - binding id
  - canonical name
  - device
  - expected prepared kind
  - stage name

### Unit Coverage

Add or extend `tests/v2/unit/loaders/Test__PreparedWeightStore.cpp` for:

- `gemmKernel(ref)` success and missing-ref failure.
- `preparedRefForBinding(binding_id, device)` success and wrong-device/wrong-binding failures.
- `slicedGemmKernel(ref, start, end)` success, cache reuse, invalid ranges.
- fused gate/up lookup by refs.
- store cleanup releases owned handles and fused/sliced caches.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_unit_prepared_weight_store
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_PreparedWeightStore$"
```

### Exit Criteria

- All model-stage kernel shapes have a prepared-store API.
- Store APIs are covered by focused unit tests.
- Stage files do not need new `KernelFactory` calls to support workspace planning.

---

## Phase 2: Prepared-Weight Test Harness for Standalone Stage Tests

**Goal**: Replace test reliance on implicit factory fallback with explicit prepared context construction.

### Implementation Tasks

- Add a reusable test helper, likely under `tests/v2/utils/PreparedWeightTestHarness.h`.
- The harness should create:
  - `ModelContextId`
  - `WeightBinding`
  - `PreparedWeightStore`
  - `PreparedWeightRef`
  - stage params populated with both tensor pointers and refs
- Provide helpers for common cases:

```cpp
PreparedGemmFixture makePreparedGemmFixture(TensorBase* tensor,
                                            DeviceId device,
                                            std::string canonical_name);

PreparedGateUpFixture makePreparedGateUpFixture(TensorBase* gate,
                                                TensorBase* up,
                                                DeviceId device,
                                                int layer);
```

- Migrate direct stage tests to use the harness instead of raw tensor-only params.
- Remove any direct stage-test expectation that a missing prepared ref lazily succeeds.
- Keep low-level kernel tests separate. If a test exists only to validate the old factory method, mark it for deletion or rewrite against `PreparedWeightStore`.

### Unit Coverage

Add focused harness tests proving:

- A direct `GEMMStage` can execute with a test-prepared ref.
- Missing prepared ref fails clearly once strict mode is enabled in later phases.
- Fused QKV and fused gate/up fixtures can build params without direct factory calls in the stage.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_unit_prepared_weight_store
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "PreparedWeightStore|GEMMStage|FusedQKV|FusedGateUp"
```

Use the exact target/test names found during implementation; do not limit CTest parallelism.

### Exit Criteria

- Direct model-stage tests no longer depend on lazy `KernelFactory` preparation.
- The harness is the standard path for standalone model-weight-backed stage tests.
- Test code distinguishes low-level kernel tests from model-stage tests.

---

## Phase 3: Stage Execution Strict Prepared-Ref Resolution

**Goal**: Remove execution-time compatibility fallbacks from model GEMM stages.

### Implementation Tasks

- Add a shared resolver helper per stage or per stage family:

```cpp
ITensorGemm* resolvePreparedGemmOrFail(const char* stage_name,
                                       PreparedWeightStore* store,
                                       const std::optional<PreparedWeightRef>& ref,
                                       const TensorBase* tensor_for_diagnostics);
```

- Refactor `GEMMStage::execute()`:
  - Use `PreparedWeightStore::gemmKernel(ref)` for normal GEMM.
  - Use `PreparedWeightStore::slicedGemmKernel(ref, start, end)` for sliced TP ranges.
  - Remove direct `KernelFactory::getOrCreatePreparedGemmWeights()` fallback.
- Refactor `LMHeadStage::execute()` the same way.
- Refactor `FusedQKVGEMMStage::execute()` so q/k/v must all resolve through the store.
- Refactor `FusedGateUpGEMMStage::execute()` so gate/up must resolve through the store.
- Refactor `GDNProjectionStage::execute()` so qkv/z/a/b must resolve through the store.
- Refactor shared expert paths in `MoEExpertComputeStage.cpp` so gate/up/down resolve through store-owned refs or expert slab refs.
- Add explicit failure messages when refs are absent. A missing ref is no longer a cue to lazily prepare.
- Preserve legitimate non-model CPU/floating fallback only behind an explicit non-model path, if such a path is still required by a low-level test. Do not allow graph-built model stages to take it.

### Unit Coverage

- Positive tests for each migrated stage with prepared refs.
- Negative tests for each migrated stage with missing refs/store.
- Negative tests for store miss with a provided ref.
- Sliced GEMM stage test using store-owned sliced resolver.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_unit_prepared_weight_store v2_test_device_graph_orchestrator
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "V2_Unit_(PreparedWeightStore|DeviceGraphOrchestrator)$"
```

Then run the most relevant stage/unit tests discovered in Phase 0.

### Exit Criteria

- `rg "getOrCreatePreparedGemmWeights" src/v2/execution/compute_stages` returns no stage execution calls.
- Missing prepared refs produce deterministic test failures, not lazy preparation.
- Existing graph-built unit tests continue to pass.

---

## Phase 4: Workspace-Consumer Migration

**Goal**: Make workspace planning and binding use the same prepared-store kernels as execution.

### Implementation Tasks

- Refactor `getKernelAsWorkspaceConsumer()` implementations for GEMM-bearing stages to call the strict prepared-store resolver.
- Refactor `getWorkspaceRequirements()` and `bindWorkspace()` paths that currently create or fetch kernels independently.
- Cache resolved workspace consumers only after store-backed resolution succeeds.
- For multi-weight stages, define deterministic workspace-consumer selection:
  - Fused QKV: either a fused consumer or the max/sum requirement from q/k/v consumers, matching current semantics.
  - Fused gate/up: fused consumer from store-owned fused gate/up kernel.
  - GDN projection: consumer for each projection or a combined plan, matching the stage implementation.
  - Shared experts: consumers from store-owned shared expert refs.
- Update `IWorkspaceConsumerStage.h` examples/comments so they no longer instruct authors to call `KernelFactory::getOrCreatePreparedGemmWeights()`.
- Add assertions that workspace planning did not increase any global factory prepared-GEMM counter while compatibility counters still exist.

### Unit Coverage

- Workspace consumer returns non-null for prepared-ref stages that need workspace.
- Workspace consumer returns null or fails clearly for missing refs, without preparing anything.
- `bindWorkspace()` works after store-backed consumer resolution.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_test_device_graph_orchestrator
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "V2_Unit_DeviceGraphOrchestrator"
```

Also run any memory-planner or workspace-specific test targets found in Phase 0.

### Exit Criteria

- `rg "getOrCreatePreparedGemmWeights" src/v2/execution/compute_stages` returns no workspace-path calls.
- Workspace planning cannot create prepared model weights.
- Graph capture/workspace-related tests pass.

---

## Phase 5: Graph and Orchestrator Validation Hardening

**Goal**: Catch missing prepared refs at graph materialization instead of during first execution.

### Implementation Tasks

- Add a validation hook to compute stages, for example:

```cpp
virtual bool validatePreparedWeights(std::string* error) const;
```

- Default implementation returns true for non-weight or non-GEMM stages.
- GEMM-bearing stages validate:
  - required refs are present
  - store pointer is present
  - ref device matches stage device
  - store contains each ref
  - sliced ranges have a valid ref-backed sliced kernel path
- Run validation after graph build and before execution/workspace planning in `DeviceGraphOrchestrator` or `DeviceGraphExecutor`.
- Include stage name, layer, binding id, canonical name, and device in failures.
- Add a debug/integration assertion that graph-built model stages never contain raw tensor-only model weights.

### Unit Coverage

- A graph with complete refs validates successfully.
- A graph with one missing ref fails before execution.
- A ref-device mismatch fails before execution.
- A stale binding id or missing store entry fails before execution.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_test_device_graph_orchestrator v2_test_device_graph_orchestrator_phase_aware_weights
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "V2_Unit_DeviceGraphOrchestrator|V2_Unit_DeviceGraphOrchestrator_PhaseAwareWeights|V2_Unit_PhaseAwareExecution"
```

### Exit Criteria

- Prepared-ref failures are initialization/materialization errors.
- Graph replay cannot encounter a missing prepared weight unless validation was explicitly bypassed.

---

## Phase 6: Migrate Loader, MPI Utility, and Integration Consumers

**Goal**: Remove non-stage test and utility dependencies on the old factory method where they are testing model-weight behavior rather than factory internals.

### Implementation Tasks

- Migrate loader/integration tests that call the factory method only to verify prepared GEMM availability.
- Replace direct calls with `PreparedWeightStore::prepareGemm()`, `registerPreparedGemmFromPipeline()`, or test-harness helpers.
- Migrate MPI utility tests that use sliced GEMM to the store-owned sliced API.
- Rewrite prepared-key collision tests as prepared-store identity tests, or delete them if the global registry behavior is no longer relevant.
- Rewrite prepared-embedding lifecycle tests against `PreparedWeightStore` if they are model-weight lifecycle tests.
- Keep or rewrite low-level kernel parity tests to use lower-level kernel creation APIs that do not own model prepared state.

### Likely Test Areas

- `tests/v2/unit/kernels/`
- `tests/v2/unit/loaders/`
- `tests/v2/unit/moe/`
- `tests/v2/integration/loaders/`
- `tests/v2/integration/utils/mpi/`
- `tests/v2/integration/kernels/`

### Validation

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_"
```

Run focused integration tests for the touched areas:

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "UnifiedGPUPipeline|SlicedGEMM|ColumnParallel|WeightManagerMemoryEfficient|MoE"
```

### Exit Criteria

- Non-factory tests no longer call `KernelFactory::getOrCreatePreparedGemmWeights()`.
- Tests that remain tied to the old method are explicitly marked for deletion in Phase 8.
- Unit test suite passes.

---

## Phase 7: Store-Internal KernelFactory Dependency Removal

**Goal**: Stop `PreparedWeightStore` itself from relying on the soon-to-be-deleted factory method.

### Implementation Tasks

- Move the preparation logic currently inside `KernelFactory::getOrCreatePreparedGemmWeights()` behind a non-global, ownership-neutral helper.
- Candidate shape:

```cpp
class KernelRegistry {
public:
    static std::shared_ptr<PreparedGemmHandle> prepareGemmHandle(
        const TensorBase* tensor,
        DeviceId target_device,
        GemmPreparationKind prep_kind);

    static ITensorGemm* resolveGemmEngine(const PreparedGemmHandle* prepared);
};
```

- The new helper must not insert into a global prepared-GEMM registry.
- `PreparedWeightStore::prepareGemm()` owns the returned prepared handle.
- `PreparedWeightStore::slicedGemmKernel(ref, start, end)` owns or indexes sliced kernels locally.
- `PreparedWeightStore::fusedGateUpKernel()` owns fused kernels locally.
- Preserve pipeline registration by transferring ownership or borrowing only from explicitly model-owned pipeline handles. Avoid hidden static registry ownership.
- Update cleanup so `PreparedWeightStore::releaseAllPreparedState()` is the owner-side teardown path.
- Remove `TensorBase` destructor dependence on prepared-GEMM cache invalidation for model weights if it still exists.

### Unit Coverage

- Preparing via store does not increase a global factory prepared-GEMM registry size if such a counter still exists.
- Destroying temporary tensors does not affect store-owned prepared entries.
- Destroying the store releases all owned handles and fused/sliced caches exactly once.
- Pipeline-registered handles remain alive for the model context lifetime and are released through the store or owning pipeline object.

### Validation

```bash
cmake --build build_v2_integration --parallel --target v2_unit_prepared_weight_store
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_PreparedWeightStore$"
```

Then run the loader/GPU pipeline focused integration gate:

```bash
cmake --build build_v2_integration --parallel --target v2_integration_unified_gpu_pipeline
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Integration_UnifiedGPUPipeline$|^V2_Integration_UnifiedGPUPipelineFP$"
```

### Exit Criteria

- `PreparedWeightStore` can prepare and own model GEMM state without calling `KernelFactory::getOrCreatePreparedGemmWeights()`.
- The old factory method is now unused except by tests/comments scheduled for final deletion.

---

## Phase 8: Delete the KernelFactory Prepared GEMM Method

**Goal**: Remove the old API completely and prove nothing needs it.

### Implementation Tasks

- Delete the declaration of `KernelFactory::getOrCreatePreparedGemmWeights()` from `KernelFactory.h`.
- Delete the definition from `KernelFactory.cpp`.
- Delete or rewrite tests whose only subject was that API's global registry behavior.
- Update comments that recommend the old method.
- Update any deprecated ROCm/CUDA headers that cite the old method as replacement guidance.
- Remove or rename global registry counters and cleanup helpers that were only needed by the deleted method.
- Remove static prepared-GEMM registry storage if no other code owns it legitimately.
- Compile and fix every resulting error by routing callers through `PreparedWeightStore`, `KernelRegistry`, or a low-level non-owning kernel creation helper.

### Validation

Search gate:

```bash
rg "getOrCreatePreparedGemmWeights" src/v2 tests/v2
```

Build gate:

```bash
cmake --build build_v2_integration --parallel
```

Unit gate:

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_"
```

Integration gate:

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Integration_"
```

Targeted GPU/lifetime gate when hardware is available:

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "UnifiedGPUPipeline|Parity|MoE|SlicedGEMM|ColumnParallel"
```

### Exit Criteria

- The old method no longer exists.
- The search gate is clean.
- Full required unit and integration gates pass.
- No model stage, workspace planner, loader, MPI utility, or test has a hidden lazy prepared-GEMM path.

---

## Phase 9: Cleanup, Documentation, and Regression Guards

**Goal**: Leave the codebase in a state where the old pattern cannot return accidentally.

### Implementation Tasks

- Update `MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md` to mark this migration complete.
- Update new-model and graph-builder guidance to require `WeightBinding` and `PreparedWeightRef` population for model GEMM stages.
- Add a CI/search guard for forbidden stage-level prepared factory calls, adjusted now that the method is deleted.
- Ensure diagnostic messages say `PreparedWeightStore` or `KernelRegistry`, not deleted `KernelFactory` APIs.
- Remove temporary transition helpers that key model prepared state by raw tensor pointer if no longer needed.
- Keep only deliberately low-level kernel creation APIs in `KernelFactory` or the renamed `KernelRegistry`.

### Validation

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Unit_"
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Integration_"
```

### Exit Criteria

- Documentation matches the implemented contract.
- CI/search guard prevents reintroduction.
- Handoff notes include any hardware-specific tests that were deferred locally.

---

## Required Test Matrix by Phase

Use exact targets where available; discover names with `ctest --test-dir build_v2_integration -N` as needed.

| Phase | Minimum validation |
|-------|--------------------|
| 0 | PreparedWeightStore and DeviceGraphOrchestrator unit tests |
| 1 | PreparedWeightStore unit tests |
| 2 | PreparedWeightStore plus direct migrated stage tests |
| 3 | PreparedWeightStore, DeviceGraphOrchestrator, migrated stage tests |
| 4 | DeviceGraphOrchestrator plus workspace/memory-planner tests |
| 5 | DeviceGraphOrchestrator, phase-aware execution, graph validation tests |
| 6 | Full V2 unit suite plus focused loader/MPI/MoE integration tests |
| 7 | PreparedWeightStore plus UnifiedGPUPipeline integration tests |
| 8 | Full V2 unit and integration suites, with search gate clean |
| 9 | Full V2 unit and integration suites plus CI/search guard |

For every phase that touches GPU-prepared ownership, run the unified GPU pipeline gate when hardware is available:

```bash
cmake --build build_v2_integration --parallel --target v2_integration_unified_gpu_pipeline
ctest --test-dir build_v2_integration --output-on-failure --parallel -R "^V2_Integration_UnifiedGPUPipeline$|^V2_Integration_UnifiedGPUPipelineFP$"
```

---

## Regression Risks and Mitigations

| Risk | Failure mode | Mitigation |
|------|--------------|------------|
| Direct stage tests lose easy setup | Tests become noisy or skip coverage | Add `PreparedWeightTestHarness` before making refs mandatory. |
| Workspace planning resolves different kernels than execution | Workspace size mismatch or graph-capture failure | Use one shared store-backed resolver for execution and workspace. |
| Sliced TP path still keys by raw tensor | Host-release or alias bugs return | Add ref-based `slicedGemmKernel` API and tests. |
| Store internally depends on deleted factory method | Final deletion blocked late | Move non-global preparation helper before deleting the method. |
| MoE expert code has dual registry assumptions | Missing experts or stale engines | Migrate expert slabs and shared experts before final deletion. |
| Low-level kernel tests hide production dependencies | False confidence from partial search | Classify tests in Phase 0 and enforce search gates at Phase 8. |
| Full integration is too slow for every slice | Local iteration stalls | Run focused gates per phase, but final acceptance requires full unit/integration gate. |

---

## Done Definition

This project is done only when the deleted-method gate passes. A partial migration that leaves the method present as a compatibility shim is not complete.

Final state:

- `PreparedWeightStore` owns model prepared GEMM lifetime.
- Graph-built stages carry prepared refs for all model GEMM weights.
- Stage execution and workspace planning use store-backed resolvers.
- Standalone model-stage tests construct explicit prepared context.
- `KernelFactory::getOrCreatePreparedGemmWeights()` no longer exists.
- All required unit and integration tests pass after deletion.
