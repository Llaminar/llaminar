# Unified GPU Weight Pipeline Remediation Plan

## Purpose

This document captures the follow-up work needed to bring the unified GPU weight pipeline implementation in line with `UNIFIED_GPU_WEIGHT_PIPELINE_PLAN.md`.

The current implementation has a working core: dense GEMM weights and MoE expert weights can flow through `WeightManager::packGemmWeightsViaPipeline()`, and the quantized unified pipeline integration test passes. The remaining work is primarily contract hardening:

- make `ExpertGemmRegistry` report complete readiness, not partial presence
- make GPU graph construction fail early when required expert engines are missing
- make MoE host-data release tensor-specific and device/layer/role-aware
- close dynamic rebalancing gaps around raw GGUF fallback
- add the missing VRAM budget preflight
- add focused tests for these contracts

## Current Audit Summary

### Confirmed Working

- `ExpertGemmRegistry` exists and is wired into `WeightManager`.
- `WeightManager::packGemmWeightsViaPipeline()` collects dense GEMM weights and MoE `_exps.weight` tensors into one GPU pipeline invocation.
- MoE expert views are registered in `ExpertGemmRegistry` after pipeline load.
- Unit tests for registry, MoE expert collection, and FP pipeline support pass.
- `V2_Integration_UnifiedGPUPipeline` passes for the Q4_K unified pipeline path.

### Gaps To Remediate

1. GPU graph construction is not fail-closed when registry entries are missing.
2. `ExpertGemmRegistry::populateExpertEngines()` returns true if it finds any engine, not all required gate/up/down engines.
3. `_exps.weight` host-data release uses `expert_gemm_registry_.size() > 0`, which is too broad.
4. GPU graph build still calls `MoEExpertComputeStage::prepareExpertGemmEngines()` in masked expert-cache paths.
5. GPU dynamic rebalance paths still contain raw GGUF fallback behavior.
6. VRAM budget validation before `LoadOrchestrator::allocate()` is absent.
7. The planned `GraphSchema::isMoEExpertWeight()` helper is absent or needs to be removed from the original plan as unnecessary.
8. `.githooks/benchmark_baseline.json` was modified outside the plan and should be reverted or explicitly approved.

## Phase 0: Guardrails And Cleanup

### Goals

Keep remediation focused and prevent unrelated changes from being mixed into the patch.

### Tasks

1. Revert or isolate the unrelated `.githooks/benchmark_baseline.json` threshold change unless a human explicitly approves it.
2. Avoid changes unrelated to the unified GPU weight pipeline remediation.
3. Rebuild focused targets after each implementation phase:

```bash
cmake --build build_v2_integration --parallel --target \
  v2_unit_expert_gemm_registry \
  v2_unit_unified_pipeline_moe \
  v2_test_fp_weight_pipeline \
  v2_integration_unified_gpu_pipeline
```

### Acceptance Criteria

- No benchmark baseline edits are included without explicit approval.
- Focused targets build cleanly before deeper test runs.

## Phase 1: Make ExpertGemmRegistry Complete-Aware

### Problem

The registry currently answers "did I find anything?" when graph construction needs "do I have every required expert role?" A partially populated layer can be treated as usable.

### Tasks

1. Add strict completeness APIs to `src/v2/loaders/ExpertGemmRegistry.h` and `.cpp`:

```cpp
bool hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const;
bool hasCompleteLayer(DeviceId device, int layer, int num_experts) const;
```

2. Change `populateExpertEngines()` semantics so it returns true only when all `num_experts * 3` engines exist.
3. If permissive behavior is still needed, add a clearly named helper such as `populateAvailableExpertEngines()`.
4. Keep `removeEngine()` and `replaceEngine()` consistent with completeness checks.
5. Extend `tests/v2/unit/loaders/Test__ExpertGemmRegistry.cpp` with:
   - complete layer returns true
   - missing one gate engine returns false
   - missing one up/down role returns false
   - removal makes a previously complete layer incomplete
   - replacement restores completeness
   - `populateExpertEngines()` returning true guarantees no null entries

### Acceptance Criteria

- Partial registry population cannot be reported as graph-ready.
- Existing and new registry tests pass.

## Phase 2: Fail Closed In Qwen35MoEGraph

### Problem

`Qwen35MoEGraph` can still build GPU MoE stages with null expert GEMM engines and defer failure until execution. It also still calls the old preparation path for GPU masked expert-cache paths.

### Tasks

1. In `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`, make all GPU graph construction consume `ExpertGemmRegistry`.
2. Remove the GPU null-vector fallback:

```cpp
prepared_gate_gemm.assign(num_experts, nullptr);
prepared_up_gemm.assign(num_experts, nullptr);
prepared_down_gemm.assign(num_experts, nullptr);
```

3. Remove GPU graph-builder calls to `MoEExpertComputeStage::prepareExpertGemmEngines()`.
4. For normal full-resident GPU mode:
   - require `registry.hasCompleteLayer(device, layer_idx, num_experts)`
   - call `populateExpertEngines()`
   - fail graph build with a clear log message if incomplete
5. For GPU expert-cache or masked mode:
   - require all active masked experts to have gate/up/down engines in the registry, or define a deliberate dynamic-rebalance bootstrap path that does not rely on raw host data after release
   - fail early if required active engines are absent
6. CPU graph construction should continue to use `MoEExpertComputeStage::prepareExpertGemmEngines()`.

### Acceptance Criteria

- No GPU path in `Qwen35MoEGraph` calls `prepareExpertGemmEngines()`.
- Missing GPU expert engines fail during graph construction with layer, device, expert, and role context.
- Normal full-resident GPU mode has fully populated `prepared_*_gemm` vectors.

## Phase 3: Make MoE Host Release Tensor-Specific

### Problem

`releaseAllHostWeightData()` treats a 3D MoE parent tensor as releasable when `expert_gemm_registry_.size() > 0`. One registry entry can therefore allow unrelated `_exps.weight` tensors to be released.

### Tasks

1. Replace the global registry-size check in `src/v2/loaders/WeightManager.cpp` with a tensor-specific proof.
2. Parse MoE parent tensor identity from names such as:
   - `blk.N.ffn_gate_exps.weight`
   - `blk.N.ffn_up_exps.weight`
   - `blk.N.ffn_down_exps.weight`
3. Infer `num_experts` from `tensor->shape()[2]`.
4. Release only when the relevant role is complete for the relevant device:

```cpp
expert_gemm_registry_.hasCompleteRole(device, layer, num_experts, role)
```

5. If the relevant device cannot be determined for a shared cache tensor, retain host data conservatively.
6. Prefer adding explicit prepared-state metadata for the 3D parent tensor during unified pipeline registration if that is cleaner than name parsing.
7. Add regression coverage:
   - one layer registered does not release another layer's `_exps.weight`
   - one role registered does not release other roles
   - partial expert registration retains the parent
   - complete role registration allows release

### Acceptance Criteria

- MoE parent host data is released only when its own layer/role/expert set is fully prepared.
- No global registry-size release shortcut remains.

## Phase 4: Close Dynamic Rebalance Gaps

### Problem

The design says runtime GPU dynamic rebalancing uses serialized `ExpertWeightBlobs`. Current GPU rebalance code still contains raw `view->raw_data()` fallback paths.

### Tasks

1. In `src/v2/execution/moe/MoEExpertWeightService.cpp`, separate initial materialization from runtime rebalance.
2. Runtime GPU expert arrivals must require either:
   - `received_weights` with serialized expert blobs, or
   - `ExpertWeightPayloadProvider` blobs
3. After host release is allowed, runtime GPU rebalance must not fall back to raw GGUF tensor data.
4. Preserve one-off `LoadOrchestrator` use for arrived experts.
5. Keep registry updates on arrival/departure, but add focused tests:
   - arriving GPU expert inserts gate/up/down in `ExpertGemmRegistry`
   - departing expert removes gate/up/down
   - missing payload after host release fails cleanly
   - registry state and stage `prepared_*_gemm` state remain consistent
6. Extend or mirror `tests/v2/integration/moe/Test__CrossDomainExpertTransferIntegration.cpp` for GPU when a backend is available.

### Acceptance Criteria

- Dynamic GPU rebalance does not depend on retained raw 3D tensor data.
- Arrivals and departures keep stage state, registry state, and ownership lifetimes synchronized.
- Focused dynamic rebalance tests pass.

## Phase 5: Add VRAM Budget Preflight

### Problem

The original plan requires a pre-allocation memory budget check, but the implementation currently allocates directly.

### Tasks

1. Before `orchestrator->allocate(max_raw_bytes, repack_streams)` in `WeightManager::packGemmWeightsViaPipeline()`, compute required bytes:
   - total planned pool bytes
   - staging bytes: `repack_streams * max_raw_bytes`
   - safety margin
2. Query backend free/total device memory. Prefer adding a backend API if one does not already exist, instead of embedding CUDA/ROCm-specific calls in `WeightManager`.
3. Account for reserved KV cache and activation estimates if available. If exact estimates are not available yet, use a conservative safety margin and document it.
4. Fail before allocation when required memory exceeds available budget.
5. Error message should include:
   - device
   - required bytes
   - available bytes
   - planned weight bytes
   - staging bytes
   - suggested mitigations: `LLAMINAR_WEIGHT_STREAMING=1`, smaller model, smaller context, or fewer resident experts

### Acceptance Criteria

- Oversized unified pipeline requests fail before VRAM allocation.
- Failure is actionable and not a generic allocation error.
- Existing successful unified pipeline integration still passes.

## Phase 6: Remove Stale Dead Code And Comments

### Problem

Some comments and code paths still describe lazy graph packing as normal behavior, which conflicts with the unified pipeline contract.

### Tasks

1. Update stale lazy-graph comments in:
   - `src/v2/loaders/WeightManager.cpp`
   - `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
   - `src/v2/execution/moe/MoEExpertWeightService.cpp`
2. Decide whether to add `GraphSchema::isMoEExpertWeight(name)`:
   - if useful, add it and use it from `WeightManager`
   - if unnecessary, update `UNIFIED_GPU_WEIGHT_PIPELINE_PLAN.md` to remove it from required files
3. Remove or rename fallback helpers so remaining names reflect intentional behavior.

### Acceptance Criteria

- Comments match the actual lifecycle.
- No code path documents lazy GPU expert packing as the normal path.
- Original plan and implementation plan no longer disagree on the `GraphSchema` helper.

## Phase 7: Verification Gates

Run these in order.

### 1. Fast Unit Gates

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  -R "V2_Unit_ExpertGemmRegistry|V2_Unit_UnifiedPipelineMoECollection|V2_Unit_FPWeightPipeline"
```

### 2. Core Unified Pipeline Integration

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  -R "^V2_Integration_UnifiedGPUPipeline$"
```

### 3. Dynamic Rebalance Tests

Run the new focused dynamic rebalance registry tests and the existing cross-domain expert transfer tests that apply to the available backend.

### 4. Full MoE Parity

Run at least:

```bash
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  -R "V2_Integration_Parity_Qwen35MoE_SingleDevice.*ROCm"
```

If CUDA hardware is available, also run the CUDA variant.

For masked/dynamic paths, run the Hybrid PP/TP GPU expert cache parity tests.

### 5. Log Checks

Successful full Qwen35MoE GPU runs should show:

- full registry load count for full model: `30720` engines for 40 layers x 256 experts x 3 roles
- no `Missing GEMM engines` during normal GPU execution
- no `RETAINED host data for blk.*.ffn_*_exps.weight` after successful full preparation
- no runtime raw GGUF fallback during GPU dynamic rebalance after host release

## Implementation Order

Recommended order:

1. Phase 0: guardrails
2. Phase 1: registry completeness
3. Phase 2: graph fail-closed behavior
4. Phase 3: tensor-specific host release
5. Phase 4: dynamic rebalance closure
6. Phase 5: VRAM preflight
7. Phase 6: stale comments and plan sync
8. Phase 7: verification

Phases 1 through 3 are the critical correctness path. Phase 4 depends on the release contract being precise. Phase 5 improves failure mode quality and can be implemented after the core behavior is strict.
