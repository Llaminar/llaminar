# MoE Expert Residency Budget Plan

**Date**: 2026-05-07
**Status**: Proposed
**Scope**: Orchestrator-owned VRAM budgets for shared experts and dynamic hot routed expert caches in mixed CPU/GPU MoE execution.

---

## Summary

Llaminar should keep dynamic at-inference-time hot expert rebalancing and migration between domains based on decode histograms. That mechanism is useful and should remain part of the target architecture.

The missing contract is not dynamic migration. The missing contract is an orchestrator-owned VRAM budget for GPU-resident MoE experts.

The target behavior is:

1. The orchestrator computes a per-domain expert residency budget before graph/materialization.
2. Shared experts are placed first because they are always active and should have priority over opportunistic routed expert caching.
3. Remaining budget is assigned to a hot routed expert cache.
4. Histogram-driven rebalancing can change which routed experts are cached, but it must never exceed the planned hot-cache budget.
5. If the shared experts do not fit, the plan should fail early or degrade explicitly according to a user-selected policy; it should not silently steal unbounded VRAM from the hot cache or general model allocation.

---

## Current Gap

The active runtime already has several pieces of the desired dynamic system:

- `DecodeExpertHistogram` records expert usage.
- `MoERebalanceController` computes new expert masks from histogram data.
- `RankOrchestrator::applyMoEExpertMasksForAllDevices()` applies masks to local device runners.
- `DeviceGraphOrchestrator::applyExpertMasks()` mutates existing `MoEExpertComputeStage` residency.
- `MoEExpertWeightService` can release departed experts and prepare arrived experts.

Those pieces are compatible with dynamic migration and should not be removed.

The gap is that the GPU hot expert cache is currently described mostly as a count of experts per layer, not as an orchestrator-owned byte budget that competes with shared expert residency and the rest of the model allocation.

That makes deployments fragile because the runtime can decide that `N` hot experts should be cached without knowing whether the orchestrator intended to reserve enough VRAM for those experts after shared experts, dense weights, KV cache, activation workspace, and safety margins are accounted for.

---

## Design Principles

### Dynamic Rebalancing Is In Scope

Dynamic rebalancing is a core feature, not a workaround.

The system should continue to support:

- decode histogram collection,
- periodic hot expert selection,
- expert migration between CPU and GPU domains,
- expert migration between GPU domains where supported,
- release of cold experts,
- preparation of newly hot experts,
- stable execution while the hot set changes over time.

The budget plan constrains **how much** can be resident, not **which** experts can become resident.

### Shared Experts Have First Claim

Shared experts should be budgeted before routed expert cache capacity.

Reasoning:

- shared experts are always used when the MoE block has them,
- they are not speculative cache entries,
- evicting them to make room for routed experts creates poor latency behavior,
- their residency is architectural, while hot routed expert cache residency is adaptive.

### Hot Expert Cache Is a Budgeted Pool

The routed expert GPU cache should be represented as a byte pool owned by an expert residency plan.

The runtime may swap experts in and out of that pool based on histograms, but the pool size is fixed unless the orchestrator deliberately replans.

### Byte Budgets Beat Expert Counts

A user-facing or debug count such as `experts_per_layer` can be retained as a convenience, but the authoritative contract should be bytes.

Expert counts are insufficient when:

- shared experts and routed experts have different sizes,
- layers have heterogeneous expert dimensions,
- quantization formats differ,
- devices have different free memory,
- a domain has a fixed safety margin,
- PP/TP sharding changes per-device resident bytes.

---

## Proposed Runtime Contract

Introduce an expert residency plan produced by orchestration and consumed by graph/materialization and dynamic rebalance code.

### `MoEExpertResidencyPlan`

Suggested fields:

```cpp
struct MoEExpertResidencyPlan {
    DeviceId device;
    size_t total_expert_vram_budget_bytes = 0;
    size_t shared_expert_budget_bytes = 0;
    size_t hot_routed_expert_cache_budget_bytes = 0;
    size_t safety_margin_bytes = 0;

    std::vector<int> shared_expert_layers;
    std::vector<int> shared_expert_ids;

    int max_hot_experts_per_layer_hint = 0;
    bool shared_experts_required_on_gpu = true;
    bool allow_partial_shared_expert_gpu_residency = false;
};
```

The exact shape can evolve, but the plan must answer these questions unambiguously:

- Which shared expert weights are intended to be GPU-resident?
- How many bytes do those shared experts consume on each device/domain?
- How many bytes remain for hot routed expert caching?
- What is the maximum cache occupancy allowed at inference time?
- What happens if the budget cannot fit all requested shared experts?

### `MoEExpertResidencyState`

Runtime state should be separate from the plan:

```cpp
struct MoEExpertResidencyState {
    DeviceId device;
    size_t hot_cache_bytes_used = 0;
    std::vector<std::vector<bool>> hot_expert_mask_by_layer;
    std::vector<std::vector<size_t>> hot_expert_bytes_by_layer;
};
```

The state changes during inference. The plan is the budget contract it must obey.

---

## Placement Algorithm

The orchestrator should compute the budget in this order.

### Step 1: Compute Available Expert VRAM

For each GPU domain/device:

1. Start from user-provided or probed free VRAM.
2. Subtract dense model weights resident on that device.
3. Subtract KV cache reservation.
4. Subtract activation/workspace reservation.
5. Subtract transfer/repack staging reservation.
6. Subtract a safety margin.
7. The remainder is the maximum expert residency budget.

The user should be able to cap this explicitly so the orchestrator does not consume every spare byte.

### Step 2: Place Shared Experts First

For each MoE layer owned by the domain:

1. Estimate prepared shared expert bytes for gate/up/down and any associated gate tensors.
2. Reserve those bytes before considering routed expert cache entries.
3. Materialize and prepare shared experts under ordinary prepared-weight ownership.

If shared experts do not fit, behavior must be explicit:

- fail validation by default, or
- use a configured fallback such as placing shared experts on CPU.

The runtime should not silently reduce dense model safety margins or overcommit the hot cache to make shared experts fit.

### Step 3: Allocate Hot Routed Expert Cache Budget

After shared experts are reserved:

```text
hot_cache_budget = total_expert_vram_budget - shared_expert_budget
```

If the result is zero, dynamic hot caching is disabled for that GPU domain, but CPU/GPU expert routing and CPU execution can still work.

### Step 4: Convert Budget to Per-Layer Capacity

The rebalance controller can still work with masks, but those masks must be derived from byte capacity.

Options:

- global hot cache pool across all MoE layers on a device,
- equal per-layer hot cache budget,
- weighted per-layer hot cache budget based on histogram pressure,
- user-provided per-layer budget overrides.

The first implementation should use a simple deterministic policy:

1. Compute bytes per routed expert for each layer.
2. Allocate an equal byte budget per MoE layer, or a proportional budget based on historical activation counts once histograms are available.
3. Select hot experts by descending histogram count until the layer or global byte budget is exhausted.
4. Never admit an expert if it would exceed the planned budget.

---

## Rebalance Rules

Dynamic migration remains histogram-driven, but admission is budget-gated.

For each rebalance interval:

1. Snapshot the decode histogram.
2. Rank candidate routed experts by benefit.
3. Keep already-resident hot experts if they remain high-value and fit.
4. Evict cold experts needed to make room.
5. Admit new hot experts only if their prepared bytes fit inside `hot_routed_expert_cache_budget_bytes`.
6. Apply masks and migrations.
7. Update residency state and metrics.

The rebalance controller should report:

- requested hot experts,
- admitted hot experts,
- rejected hot experts due to budget,
- cache bytes used,
- cache bytes available,
- migration bytes moved,
- preparation time for arrivals.

This makes budget pressure visible instead of silently flattening into a smaller expert count.

---

## CLI And Config Surface

The current legacy memory flags are not enough. We need explicit MoE expert residency knobs that feed the active orchestrator path.

Suggested YAML-first fields:

```yaml
moe:
  expert_residency:
    gpu_budget_mb: 4096
    shared_experts:
      placement: gpu_required   # gpu_required | gpu_preferred | cpu
    hot_cache:
      budget_mb: 2048           # optional; if omitted, use leftover after shared experts
      policy: histogram_lru     # fixed_topk | histogram_lru | histogram_weighted
      rebalance_interval_tokens: 128
```

Suggested CLI equivalents:

```bash
--moe-expert-gpu-budget-mb 4096
--moe-shared-experts gpu-required
--moe-hot-cache-budget-mb 2048
--moe-hot-cache-policy histogram-lru
--moe-rebalance-interval-tokens 128
```

Debug environment variables can remain useful for experiments, but production placement should come from parsed orchestration config so dry-run, validation, and explain-placement can all see the same budget.

---

## Integration Points

### `OrchestrationConfig`

Add active config fields for expert residency budget and policy.

These fields should not be marked as legacy/inactive. They must participate in validation and dry-run output.

### `ExecutionPlanBuilder` / `RankExecutionPlan`

Carry per-domain expert residency plan data alongside primary and auxiliary domain participation.

The plan should distinguish:

- shared expert residency requirements,
- routed expert hot-cache budget,
- dynamic rebalance policy,
- CPU expert fallback domain.

### `WeightPlan` / `WeightManager`

Shared experts should be planned and prepared as first-priority resident weights.

Routed hot experts should be prepared according to the cache plan, not as an unbounded side effect of model loading.

### `MoERebalanceController`

Replace count-only GPU cache selection with budget-aware admission.

The controller can still expose count hints for logs and debug, but byte limits should be authoritative.

### `RankOrchestrator` / `DeviceGraphOrchestrator`

Mask application should validate that the incoming mask does not exceed the device's expert residency budget before mutating live stages.

If a mask exceeds budget, the caller should either:

- reject it and keep the previous mask, or
- trim it deterministically according to the configured policy.

### `MoEExpertWeightService`

Expert arrival preparation should report actual bytes allocated/prepared per expert so residency state can stay accurate.

Departures should decrement the state before or atomically with release.

---

## Failure Policy

The default policy should favor explicit errors over surprising performance cliffs.

Recommended defaults:

- shared experts: `gpu_required` for GPU expert residency mode,
- hot cache: best-effort within budget,
- over-budget hot candidate: reject candidate, keep running,
- over-budget shared expert placement: fail validation or require explicit CPU fallback,
- unknown expert byte size: fail validation.

This preserves correctness and makes memory behavior inspectable.

---

## Validation Plan

### Unit Tests

- Budget calculator reserves shared expert bytes before hot cache bytes.
- Shared experts that exceed budget fail validation by default.
- Hot cache admission rejects experts that exceed byte budget.
- Rebalance keeps total resident hot expert bytes below budget across repeated histogram updates.
- Equal-count and byte-budget policies differ correctly when expert byte sizes differ.

### Planner Tests

- Config with `gpu_budget_mb` produces a per-device `MoEExpertResidencyPlan`.
- Config with explicit `hot_cache.budget_mb` never allocates more than that to routed experts.
- Config without `hot_cache.budget_mb` allocates leftover expert budget after shared experts.
- CPU fallback for shared experts is represented explicitly in the plan.

### Integration Tests

- Shared experts remain resident while hot routed experts migrate.
- Histogram-driven hot expert selection changes masks over decode steps.
- Cache bytes used never exceed the planned hot-cache budget.
- Over-budget hot candidates are logged and skipped without corrupting outputs.
- Dry-run and explain-placement show shared expert bytes, hot cache bytes, and safety margin.

---

## Open Questions

1. Should the first hot-cache budget be global per device or split equally per layer?
2. Should shared experts be allowed to spill to CPU automatically when only a subset fits, or should that require an explicit `gpu_preferred` policy?
3. Should histogram-based budget allocation consider expert preparation/migration cost, not only activation frequency?
4. Should the hot cache include a hysteresis threshold to avoid churn when two experts have near-equal counts?
5. Should budget enforcement live entirely in `MoERebalanceController`, or should `DeviceGraphOrchestrator` also reject over-budget masks as a final guard?

Recommended initial answers:

- Use a global per-device hot cache budget with deterministic per-layer admission for phase one.
- Make shared expert CPU fallback explicit.
- Use activation frequency first, then add migration cost once metrics exist.
- Add simple hysteresis before enabling frequent migrations.
- Enforce budget in both planner/controller and orchestrator guard paths.

---

## Definition Of Done

This work is complete when:

- Dynamic histogram-driven expert rebalancing still works.
- Shared experts are budgeted and placed before routed hot expert cache entries.
- Hot routed expert residency is constrained by an orchestrator-provided byte budget.
- Runtime mask application cannot exceed the planned hot-cache budget.
- Dry-run/explain-placement reports expert budget, shared expert bytes, hot cache bytes, and policy.
- Tests prove repeated rebalancing respects the budget while changing the hot expert set.