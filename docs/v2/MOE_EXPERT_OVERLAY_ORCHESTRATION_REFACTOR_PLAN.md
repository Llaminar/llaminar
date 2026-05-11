# MoE Expert Overlay Orchestration Refactor Plan

**Date:** 2026-05-11
**Status:** Architecture refactor plan
**Branch context:** `feat/qwen35-moe`
**Scope:** Make MoE expert overlay a first-class orchestration and graph execution mode for `oneshot`, `serve`, parity, and benchmark paths.

## Summary

The Qwen3.5 MoE expert overlay parity suite now proves that the same-layer overlay math can be made correct across CUDA, ROCm, and CPU tiers. The next attempted step, release benchmarking, surfaced a deeper architecture mismatch rather than a benchmark-only bug.

The current runtime can express a single primary execution path per rank, with local TP, local PP, or named-domain global PP as variants. MoE expert overlay needs a composite execution path: one continuation/root domain owns activation flow, while auxiliary expert domains contribute work inside the same logical layer. Some auxiliary domains are local accelerator groups, and some are cross-rank worker groups such as CPU `NodeLocalTP` fallback.

The existing parity fixture proves this by manually making rank 0 run the real graph while non-root ranks run a hand-written CPU fallback participant pass. That is a useful proof bridge, but it is not a production architecture. This plan replaces that fixture-only split with a clean orchestration layer that can run the same topology in `llaminar2 oneshot`, `llaminar2 serve`, parity tests, and release benchmarks.

## Triggering Symptom

Layout A release validation currently fails on rank 1 with a continuation-domain reachability error:

```text
MoE expert overlay continuation_domain selects root device ROCm:0 instead of requested/default CPU
failed to resolve MoE expert overlay plan: MoE expert overlay continuation domain 'rocm_shared_hot' is not locally reachable for the current DeviceGraphExecutor MVP: domain 'rocm_shared_hot' primary participant rocm:0 (ROCm:0) rank=0; current_rank=1, primary_is_local=true, primary_owned_by_current_rank=false
```

This error is correct under today's assumptions: every MPI rank tries to create a normal root inference runner, and a rank-local `DeviceGraphExecutor` cannot use a continuation device owned by another rank. But for overlay, rank 1 should not create the continuation graph at all. It should run as an auxiliary CPU fallback participant in the `cpu_cold` domain.

## Current Architecture Friction

### 1. `-d` Conflates Several Meanings

For overlay validation, `-d cpu` is being used as a workaround to avoid base-model memory validation against CUDA/ROCm before the overlay continuation domain takes over. That is a symptom of a larger problem: the CLI has no separate fields for root continuation, base/non-expert residency, and auxiliary expert domains.

In overlay mode, `continuation_domain` should choose the root activation/logits owner. Base model placement and expert residency should be separate concepts. `-d` should remain a simple single-device shorthand, not an overlay root escape hatch.

### 2. `RankExecutionPlan` Is Primary-Path Oriented

`RankExecutionPlan` currently describes what one MPI rank owns in a layer/TP/PP execution path. Overlay needs a rank-role contract:

- continuation root rank,
- local accelerator participant,
- CPU fallback participant,
- remote expert worker,
- relay/idle rank.

The current plan cannot say "rank 1 owns no root graph but participates in `cpu_cold` expert collectives for every MoE layer."

### 3. App Initialization Creates A Full Runner On Every Rank

`RuntimeInitPhase` creates and initializes an `IOrchestrationRunner` on all MPI ranks. That is appropriate when all ranks participate in one homogeneous runner family. It is wrong for overlay, where only the continuation owner should build the root graph and auxiliary ranks should build domain workers.

### 4. DeviceGraph Reachability Guards Are Doing The Wrong Job

`MoEExpertOverlayRuntimePlan` correctly refuses to let a rank-local root graph use a continuation primary owned by another rank. The fix is not to loosen that guard. The fix is to avoid constructing the root graph on ranks that are not continuation owners.

### 5. Named Domains Are Still Mostly PP-Shaped

`NamedDomainGlobalRunner` understands multi-rank named domains, but its topology is organized around pipeline stages and layer ownership. MoE overlay domains contribute to the same layer, then reduce back to the continuation domain. Treating overlay as PP would violate the same-layer expert overlay design.

### 6. Parity Uses Production Math But Not Production Orchestration

The parity fixture's non-root participant pass is intentionally narrow and test-local. It should be promoted into production runtime roles so parity tests stop proving a special harness behavior.

## Target Architecture

### Composite Overlay Runtime

Add a new orchestration shape, tentatively named `MoEOverlayCompositeRunner` or `CompositeDomainRunner`.

This runner owns one logical inference request across multiple rank roles:

- **Continuation root actor**: builds the normal model graph on the continuation domain, owns routing, residual flow, logits, sampling, and final output.
- **Domain worker actors**: long-lived rank-local workers for auxiliary expert domains. They own role-specific weight residency, prepared handles, domain collectives, and per-domain profiling.
- **Coordinator**: broadcasts prefill/decode/control commands, coordinates cross-rank worker participation, aggregates errors, and gathers profiling summaries.

The continuation actor may still use `DeviceGraphOrchestrator` or `RankOrchestrator` internally. The key change is that non-continuation ranks do not pretend to be root graph owners.

### Overlay Execution Plan

Introduce an overlay-specific plan derived from `MoEExpertParallelPlan`, model metadata, CLI/YAML topology, and MPI inventory.

Sketch:

```cpp
enum class OverlayRankRole {
    ContinuationRoot,
    LocalAcceleratorParticipant,
    CpuFallbackParticipant,
    RemoteExpertParticipant,
    RelayOnly,
};

struct OverlayRankPlan {
    int world_rank;
    OverlayRankRole role;
    std::vector<std::string> owned_domains;
    std::vector<DeviceId> local_devices;
    bool builds_root_graph;
    bool loads_tokenizer;
    bool loads_full_model_metadata;
    bool loads_root_weights;
    bool loads_expert_weights;
};

struct OverlayExecutionPlan {
    std::string continuation_domain;
    std::string shared_expert_domain;
    std::vector<MoEOverlayRuntimeDomain> domains;
    std::vector<OverlayRankPlan> rank_plans;
};
```

The exact types can change, but the contract should be explicit: each rank knows whether it builds a root graph, which domains it serves, and which weights it must load or prepare.

### Domain Worker Protocol

Domain workers should be command-driven and long-lived. They must not rebuild model state for every layer or token.

Initial commands:

- `InitializeDomain`: prepare role-specific weights and contexts.
- `ClearCache`: clear per-request state.
- `PrefillLayerWork`: receive or access selected hidden rows and routing metadata for a layer.
- `DecodeLayerWork`: one-row decode fast path.
- `ReturnPartial`: reduce or transfer compact row partials back to continuation.
- `FlushMetrics`: send profiling counters to the coordinator.
- `Shutdown`: exit worker loop cleanly.

The first production worker can target CPU `NodeLocalTP` fallback because that is the exact role currently implemented by parity fixture code. Later phases can generalize the same protocol for remote accelerator workers.

### Graph Boundary

The model graph should remain declarative. Overlay graph stages should not manually know which MPI ranks are running which workers. They should call a small runtime service, for example `IOverlayDomainRuntime`, to submit domain work and obtain partial output handles.

The continuation graph remains responsible for:

- dispatch descriptor creation,
- local/shared domain work when the domain is continuation-local,
- cross-domain reduce input ordering,
- final output residency on the continuation domain.

The domain runtime is responsible for:

- worker command transport,
- sparse row payload transport,
- domain-local allreduce,
- return transport,
- metrics aggregation.

### CLI And YAML Model

Overlay mode needs explicit concepts rather than overloading `-d`.

Keep current overlay domain syntax, but evolve the top-level semantics:

```bash
--moe-expert-overlay tiered
--moe-expert-overlay-continuation rocm_shared_hot
--moe-expert-overlay-shared-domain rocm_shared_hot
--moe-expert-overlay-domain "rocm_shared_hot=rocm:0,rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0"
--moe-expert-overlay-domain "cpu_cold=cpu:0,cpu:1;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1"
```

Add or refine explicit placement concepts:

- root activation/logits owner: already `--moe-expert-overlay-continuation`.
- base/non-expert model residency: new explicit option or YAML field, for example `base_model_domain` or `non_expert_domain`.
- auxiliary expert residency: already in overlay tiers and planned placements.

Acceptance target: final Layout A and Layout B commands do not require `-d cpu` as a workaround. If `-d` is supplied in overlay mode, it should either be rejected as ambiguous or interpreted only as a legacy/simple default when no explicit overlay root/base placement is configured.

## Definition Of Done

The refactor is complete when all of the following are true:

1. `llaminar2 oneshot` can run Layout A and Layout B using the composite overlay runtime.
2. `llaminar2 serve` can initialize the same composite runtime and keep auxiliary workers alive across requests.
3. `llaminar2 benchmark` uses the same runtime path as `oneshot` and parity.
4. The Qwen3.5 MoE ExpertOverlay parity fixture no longer manually runs non-root CPU fallback participant passes.
5. Non-root ranks in overlay mode do not try to construct a continuation-root `DeviceGraphExecutor`.
6. The `-d cpu` workaround is removed from accepted overlay commands.
7. Profiling reports per-domain work and timing across continuation and auxiliary ranks.
8. No test passes solely because a parity-only auxiliary path is still present.

## Implementation Guardrails

- Do not loosen continuation-domain reachability checks to hide invalid root graph construction.
- Do not express same-layer overlay as sequential PP layer ownership.
- Do not keep parity-only worker behavior as the production proof path.
- Do not add global all-expert accelerator caches. Prepared-weight ownership must remain domain-scoped.
- Do not make benchmark mode special. `benchmark`, `oneshot`, and `serve` must share the same overlay runtime construction.
- Do not artificially limit build or test parallelism.

## Phased Plan

### Phase 0: Architecture Lock And Failure Harness

#### Goal

Record the current mismatch and add tests that make the desired rank-role behavior explicit before implementation begins.

#### Deliverables

- Add an architecture decision note or comments near overlay runtime planning explaining why non-continuation ranks must not build root graphs.
- Add unit tests for resolving overlay rank roles from Layout A and Layout B.
- Add a regression test for the current failure shape: rank 1 in Layout A is an auxiliary CPU fallback participant, not a continuation-root runner.
- Mark the parity fixture non-root participant helper as temporary with a reference to this plan.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlayRuntimePlan|MoEExpertOverlay|Orchestration)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout A rank 0 resolves as continuation root and ROCm LocalTP owner.
- Layout A rank 1 resolves as CPU fallback participant only.
- Layout B rank roles distinguish CUDA continuation, ROCm LocalTP participation, and CPU fallback participation.
- The expected current production runner limitation is documented as an orchestration gap, not a graph math gap.

### Phase 1: Overlay Execution Plan And Rank Roles

#### Goal

Introduce a first-class `OverlayExecutionPlan` that maps the planned overlay topology to rank-local runtime roles.

#### Deliverables

- Add overlay rank-role value types under `src/v2/execution/moe/` or `src/v2/execution/runner/`.
- Build role plans from `MoEExpertParallelPlan`, `MoEExpertOverlayRuntimePlan`, model metadata, MPI world size, and domain definitions.
- Distinguish root graph, local accelerator participant, CPU fallback participant, and relay-only roles.
- Provide diagnostics that render the full role plan in `--explain-placement` and `LLAMINAR_MOE_EP_TRACE=1` output.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlayExecutionPlan|MoEExpertOverlayRuntimePlan|MoEExpertOverlayConfig)" --output-on-failure --parallel
```

#### Acceptance Criteria

- Role planning is deterministic and independent of test fixtures.
- Invalid topology fails before model load with rank/domain-specific diagnostics.
- A non-root auxiliary rank never receives `builds_root_graph=true` unless it owns the continuation domain.

### Phase 2: Composite Overlay Runner Skeleton

#### Goal

Add the top-level orchestration runner that chooses between continuation-root actor and domain-worker actor per rank.

#### Deliverables

- Add `MoEOverlayCompositeRunner` or equivalent implementing `IOrchestrationRunner`.
- Update `IOrchestrationRunnerFactory` selection so tiered overlay with multi-rank auxiliary domains uses the composite runner.
- On continuation ranks, delegate to the existing `OrchestrationRunner` or directly create the existing root `IInferenceRunner` path.
- On auxiliary ranks, initialize a domain worker instead of a root graph.
- Add collective init/shutdown synchronization so failed initialization does not strand worker ranks.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEOverlayCompositeRunner|OrchestrationRunnerFactory|MPICoordinatedMode)" --output-on-failure --parallel
```

#### Acceptance Criteria

- Rank 1 in Layout A does not call `createInferenceRunner()` for the ROCm continuation graph.
- Existing non-overlay single-device, LocalTP, LocalPP, and named-domain PP paths keep their current factory behavior.
- Composite runner can initialize and shut down cleanly in a mock MPI test.

### Phase 3: Production CPU Fallback Domain Worker

#### Goal

Move the parity fixture's non-root CPU fallback participant behavior into a production domain worker.

#### Deliverables

- Add `MoEOverlayDomainWorker` interface.
- Implement `CpuNodeLocalTPFallbackWorker` using `MoEExpertOverlayCPUFallback` and the existing CPU fallback stage logic.
- Keep worker state long-lived: model metadata, expert weight access, prepared host expert handles, domain context, and profiling counters.
- Support prefill and decode commands with sparse token-row payloads.
- Preserve exact routing weights, top-k semantics, and domain-local allreduce behavior.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEOverlayDomainWorker" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts)" --output-on-failure --parallel
```

#### Acceptance Criteria

- CPU fallback worker output matches the existing helper/stage behavior.
- Worker ranks participate in collectives without constructing a root graph.
- Sparse and dense compatibility payloads produce matching output.

### Phase 4: Overlay Runtime Service For Graph Stages

#### Goal

Decouple Qwen graph stages from direct cross-rank helper calls by introducing a runtime service for domain work submission.

#### Deliverables

- Add `IOverlayDomainRuntime` or equivalent service reachable from graph config/stage params.
- Root graph stages submit non-continuation tier work through the service.
- The service chooses local execution, domain-worker command, or compatibility fallback based on the overlay execution plan.
- Preserve current local accelerator LocalTP and shared expert paths where the domain is continuation-local.
- Return explicit partial descriptors for cross-domain reduce.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(Qwen35MoEExpertOverlayGraph|MoEExpertDispatchStage|MoEExpertParallelReduceStage|MoEOverlayDomainRuntime)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|MultiAcceleratorTiers)" --output-on-failure --parallel
```

#### Acceptance Criteria

- The continuation graph no longer needs parity-fixture code to activate CPU fallback participants.
- Remote or cross-rank tier work is represented as explicit domain runtime work, not as PP layer ownership.
- The graph still exposes clear dependencies for dispatch, expert compute, and reduce.

### Phase 5: Role-Aware Model Loading And Weight Residency

#### Goal

Load and prepare only the weights required by each rank role.

#### Deliverables

- Extend weight planning to understand root weights, shared expert weights, accelerator routed experts, and CPU fallback expert subsets separately.
- Continuation root ranks load root graph weights plus local-domain expert weights.
- CPU fallback worker ranks load only CPU fallback expert weights and required metadata/tokenizer state.
- Preserve prepared-weight ownership by domain, participant, layer, expert id, and role.
- Make memory validation operate per role and per domain.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(WeightManager|WeightPlan|ExpertGemmRegistry|MoEExpertOverlayPreparation|MoEOverlayExecutionPlan)" --output-on-failure --parallel
```

#### Acceptance Criteria

- Overlay commands no longer need `-d cpu` to bypass root-device memory validation.
- CUDA/ROCm tiers do not prepare CPU fallback-only experts.
- CPU fallback workers do not prepare accelerator-only experts.
- Memory validation reports root, shared, routed tier, fallback, and worker memory separately.

### Phase 6: Overlay CLI And YAML Semantics

#### Goal

Make the user-facing config express overlay intent directly and remove ambiguous `-d` behavior from accepted overlay paths.

#### Deliverables

- Add explicit config fields for base/non-expert model residency if needed by the final architecture.
- Validate that `continuation_domain` owns the root activation/logits flow.
- Validate that auxiliary domains have rank roles and worker implementations.
- Update CLI help and YAML parsing examples.
- Emit a diagnostic or validation error when `-d` conflicts with overlay root/base placement.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(OrchestrationConfig|ConfigValidator|MoEExpertOverlayConfig|OrchestrationConfigParser)" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout A and Layout B can be expressed without `-d cpu`.
- Config validation names the exact conflicting flags when legacy single-device placement is mixed with overlay root placement.
- YAML and CLI produce identical overlay execution plans.

### Phase 7: Parity Fixture Migration

#### Goal

Make the accepted Qwen3.5 MoE overlay parity suite use the same composite runtime as production.

#### Deliverables

- Remove manual non-root CPU fallback participant passes from `Test__Qwen35MoE_ExpertOverlay_Parity.cpp`.
- Instantiate the composite overlay runner in the parity harness.
- Keep hardware/model precondition skips only.
- Preserve existing stage comparisons for `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD`.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- The four real overlay parity tests still pass.
- Non-root ranks participate through production domain workers.
- No parity-only helper remains responsible for the final accepted path.

### Phase 8: Oneshot And Generate Integration

#### Goal

Run overlay inference through `llaminar2 oneshot` using the composite runtime.

#### Deliverables

- Ensure `prefill`, `decodeStep`, `generate`, cache clear, and sampling commands coordinate all domain workers.
- Ensure worker failure propagates to the root rank and exits the MPI world cleanly.
- Add concise `--explain-placement` output for root and worker roles.
- Add smoke commands for Layout A and Layout B.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(Oneshot|Composite|Runtime)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- `llaminar2 oneshot` runs Layout A and Layout B without parity-only worker code.
- Decode workers remain in lockstep for at least one generated token.
- Shutdown is clean after success and after injected worker failure.

### Phase 9: Serve Mode Integration

#### Goal

Keep overlay domain workers alive across server requests and make request lifecycle explicit.

#### Deliverables

- Integrate composite runner with `serve` and chat completion paths.
- Add request-boundary commands for cache reset, sampling params, and worker state reset.
- Ensure auxiliary workers survive multiple requests and cleanly shutdown on server stop.
- Add health/status diagnostics for domain workers.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(ServerMode|MoEOverlayCompositeRunner|MPICoordinatedMode)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*Serve" --output-on-failure --parallel
```

#### Acceptance Criteria

- Server mode does not strand non-root MPI ranks.
- Consecutive requests do not reuse stale worker routing or partial output state.
- Worker metrics are request-scoped and can be reset.

### Phase 10: Benchmark And Profiling Integration

#### Goal

Run release benchmarks through the same composite runtime and surface per-domain performance data.

#### Deliverables

- Make `BenchmarkMode` use the composite runner automatically for overlay topology.
- Aggregate profiling summaries from continuation root and all domain workers.
- Record per-domain assigned experts, resident experts, routed entries, selected rows, transfer bytes, GEMM/GEMV time, domain allreduce time, final reduce time, and idle/stall time.
- Emit CSV through `LLAMINAR_MOE_EP_PROFILE_CSV` without forcing extra D2H transfers when disabled.
- Add concrete Layout A and Layout B benchmark commands.

#### Required Commands

```bash
cmake --build build_v2_release --parallel

LLAMINAR_PROFILING=1 \
LLAMINAR_MOE_EP_TRACE=1 \
./build_v2_release/llaminar2 benchmark \
  -m models/<qwen35-moe-model>.gguf \
  --moe-expert-overlay tiered \
  --moe-expert-overlay-continuation <domain> \
  --moe-expert-overlay-shared-domain <domain> \
  --moe-expert-overlay-domain "..." \
  --moe-expert-overlay-tier "..."
```

#### Acceptance Criteria

- Benchmark uses the same production runtime as `oneshot` and parity.
- Layout B benchmark uses CUDA-owned final accumulation when configured for CUDA continuation.
- Profiling identifies every configured domain by name and rank role.
- GEMM/GEMV, transfer, local collective, final reduce, and idle/stall categories are distinguishable.

### Phase 11: Performance Optimization Pass

#### Goal

Tune the now-correct composite runtime without changing its architecture.

#### Deliverables

- Identify dominant costs from Phase 10 profiling.
- Optimize sparse row transfer, continuation-device reduce, domain-local allreduce, and worker idle time as measured.
- File explicit follow-up issues for blockers outside this phase.
- Compare against single-domain accelerator baseline and CPU-only fallback baseline where hardware allows.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_" --output-on-failure --verbose
```

#### Acceptance Criteria

- Correctness remains within the accepted Qwen3.5 MoE overlay parity envelope.
- After warmup, GEMM/GEMV is the dominant active compute category per participating compute domain unless an explicit measured follow-up is filed.
- Transfer, allreduce, final reduce, or idle/stall time is not silently accepted as the dominant cost.

### Phase 12: Cleanup And Documentation

#### Goal

Remove bridge scaffolding and leave overlay orchestration maintainable.

#### Deliverables

- Remove parity-only worker helper code and obsolete skip messages.
- Remove accepted-command reliance on `-d cpu` for overlay layouts.
- Update `MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md` with the final architecture result and benchmark/parity status.
- Update CLI/YAML docs with Layout A and Layout B examples.
- Add a changelog entry with final tests, benchmark results, and known follow-ups.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert|V2_Integration_.*MoEExpert|^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_" --output-on-failure --verbose
```

#### Acceptance Criteria

- Docs match shipped behavior.
- No test passes solely because the real production path was skipped or fixture-emulated.
- Final overlay Layout A and Layout B commands are reproducible without machine-specific paths in committed docs.
- Prepared-weight ownership remains domain-scoped.

## Suggested Dispatch Order

Use the phases in order. Do not jump to kernel tuning until Phase 10 profiling runs on the composite runtime.

```text
0. Architecture lock and failure harness
1. Overlay execution plan and rank roles
2. Composite overlay runner skeleton
3. Production CPU fallback domain worker
4. Overlay runtime service for graph stages
5. Role-aware model loading and weight residency
6. Overlay CLI and YAML semantics
7. Parity fixture migration
8. Oneshot and generate integration
9. Serve mode integration
10. Benchmark and profiling integration
11. Performance optimization pass
12. Cleanup and documentation
```

Stop advancing phases whenever the latest implementation fails its acceptance gate. Fix the current phase first, then continue.
