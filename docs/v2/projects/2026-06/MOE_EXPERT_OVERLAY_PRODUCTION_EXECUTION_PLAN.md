# MoE Expert Overlay Production Execution Plan

**Date:** 2026-05-09
**Status:** Productionization implementation plan
**Branch context:** `feat/qwen35-moe`
**Scope:** Remaining work required for real Qwen3.5 MoE Expert Overlay inference across CUDA, ROCm, and CPU tiers, with parity and release-performance gates.

## Summary

The earlier plan in `docs/v2/projects/2026-06/MOE_EXPERT_PARALLEL_GPU_HOT_CPU_COLD_PROJECT_PLAN.md` defines the target architecture: same-layer MoE expert overlay, where ordered expert tiers contribute partial expert output for the same logical MoE layer and reduce back to the continuation domain.

The current implementation has important scaffolding in place: value types, parser/config support, planner tests, model-light dispatch/reduce helpers, CPU fallback helpers, sparse transfer helpers, multi-accelerator synthetic tests, and V2 parity test names. The remaining production gap is that the real Qwen3.5 MoE graph still does not execute all overlay domains as first-class runtime domains. Several paths are still model-light, primary-participant-only, host-staged, or test-preplanned.

**2026-05-12 graph-native dispatch checkpoint:** Release benchmarking exposed a deeper mismatch in the attempted worker/coordinator layering: side-effectful fallback-domain command streams were being driven from per-device continuation graph workers, which breaks the original one-graph-per-device execution model for accelerator `LocalTP` and risks similar drift for `LocalPP`. The next direction is to make MoE overlay a graph-native sparse collective. Continuation devices/ranks and fallback-domain participants meet at explicit dispatch points, just like existing collective points, and participants with no routed work enter with an empty/no-op request. Phase 10C performance benchmarking should resume only after Phase 10A restores this graph-native dispatch contract, Phase 10B hardens the collective ownership/ordering contract, and the domain/placement UX has been reconciled, so benchmark, parity, `oneshot`, and `serve` all exercise the same production graph path with the same final configuration model.

This document is the implementation plan for closing those gaps. Each phase is intended to be dispatched to a coding subagent for implementation, then audited before the next phase begins.

## Final Gate

The feature is complete only when all of the following are true.

1. **Real three-tier MoE inference runs in V2 parity tests.**
   - The `V2_Integration_Parity_Qwen35MoEExpertOverlay_*` tests must run the real Qwen3.5 MoE inference path.
   - The tests must not pass by skipping due to missing `ExpertGemmRegistry` entries, missing multi-participant lowering, missing hardware-domain execution, or synthetic/model-light fallback.
   - At least these layouts must be covered:

```text
Layout A: ROCm shared/hot + CPU cold
  continuation_domain = rocm_hot
   base_model_domain = rocm_hot
  shared_expert_domain = rocm_hot
  routed_tier_0 = rocm_hot, devices 0:rocm:0,0:rocm:1, scope local, backend rccl, compute tensor_parallel_experts
  routed_fallback = cpu_cold, devices 0:cpu:0,1:cpu:0, scope node_local, backend upi, compute tensor_parallel_experts

Layout B: CUDA shared/hottest + ROCm hot + CPU cold
  continuation_domain = cuda_fast
   base_model_domain = cuda_fast
  shared_expert_domain = cuda_fast
  routed_tier_0 = cuda_fast, devices 0:cuda:0, scope single, backend auto, compute replicated_experts
  routed_tier_1 = rocm_hot, devices 0:rocm:0,0:rocm:1, scope local, backend rccl, compute tensor_parallel_experts
  routed_fallback = cpu_cold, devices 0:cpu:0,1:cpu:0, scope node_local, backend upi, compute tensor_parallel_experts
```

Phase 9C finalizes the placement language used by parity, oneshot, serve, and benchmark: define hardware domains once with `--define-domain` or top-level YAML `domains`, then reference those names from PP layer placement and MoE overlay placement. `--moe-expert-overlay-domain` remains accepted only as a strict compatibility alias into the same inventory; duplicating a logical domain with different devices, scope, backend, ranks, weights, or compute kind is invalid.

Layout A YAML shape:

```yaml
domains:
   - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0"
   - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1"
moe_expert_parallel:
   enabled: true
   execution_kind: tiered
   continuation_domain: rocm_hot
   base_model_domain: rocm_hot
   shared_expert_domain: rocm_hot
   residency:
      mode: histogram
   routed_tiers:
      - "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto"
      - "cold@cpu_cold;priority=1;fallback=true"
```

Layout B YAML shape:

```yaml
domains:
   - "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0"
   - "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=1"
   - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,2"
moe_expert_parallel:
   enabled: true
   execution_kind: tiered
   continuation_domain: cuda_fast
   base_model_domain: cuda_fast
   shared_expert_domain: cuda_fast
   residency:
      mode: static-by-id
   routed_tiers:
      - "hottest@cuda_fast;priority=0;max-experts-per-layer=4;memory-mb=512"
      - "warm@rocm_warm;priority=1;max-experts-per-layer=8;memory-mb=auto"
      - "cold@cpu_cold;priority=2;fallback=true"
```

2. **All three tiers must actually participate.**
   - The test topology must enforce non-empty work on each configured routed tier.
   - Do not rely on natural model fit. The model may fit entirely on the CUDA tier, which would turn Layout B into a CUDA-only test.
   - The test plan must force tier participation with explicit budgets or caps, for example:
     - CUDA tier: all shared experts when configured and fitting, plus a bounded hot routed cache such as 2 GiB or a strict expert-count cap.
     - ROCm tier: a bounded hot/warm routed cache such as 4 GiB or a strict expert-count cap.
     - CPU tier: the fallback complement, with at least one cold routed expert per tested layer.
   - Parity tests must assert that each active tier has non-zero assigned experts and non-zero executed routed work for the exercised layers/tokens.

3. **Numerical parity must be comparable to existing Qwen35 MoE parity suites.**
   - Thresholds must not be diagnostically loose just to make the overlay pass.
   - The overlay suite should use the same `BackendThresholds` conventions as the existing Qwen35 MoE parity tests.
   - Current target envelope:
     - `LM_HEAD` KL divergence no worse than the existing Qwen35 MoE threshold, currently expected around `<= 0.05` for the overlay suite.
     - Top-1 and Top-5 agreement comparable to existing Qwen35 MoE parity, currently expected around `>= 0.80` for both in the overlay suite.
     - Stage cosine thresholds comparable to current Qwen35 MoE parity, currently expected around `prefill >= 0.90`, `decode >= 0.80`, with early-layer requirements at least `5/6` unless the baseline suite is stricter.
   - `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD` must be checked. Hot-only sparsity in `MOE_EXPERT_OUTPUT` is a failure.

4. **Release-build inference must be performant and diagnosable.**
   - A release benchmark must run the real overlay topology, not a synthetic graph.
   - Devices should spend most active time in GEMM/GEMV and related expert compute, not in allreduces, host-staged transfers, queue stalls, or synchronization waits.
   - For CUDA/ROCm continuation layouts, final cross-domain partial accumulation must be owned by the continuation device. CPU dense summing is allowed only as a correctness bridge and cannot satisfy the release-performance gate.
   - Non-continuation tiers should return compact routed-row partials where possible. Dense full-sequence partial transfers are a compatibility/debug fallback, not the optimized path.
   - Initial acceptance target: steady-state decode profiles should show GEMM/GEMV as the dominant per-domain active compute category. Any case where allreduce, transfer, or idle/stall time is the largest category must be treated as a performance blocker or explicitly documented with a follow-up issue.
   - `LLAMINAR_PROFILING=1` must break down kernel/stage time across all configured domains and report MoE-specific diagnostics such as cache residency, tier assignment, routed token counts, fallback rows, rebalancing decisions, transfer volume, and per-domain reduction time.

## Current Production Gaps

The current codebase contains useful pieces, but the real production graph is not yet a true three-tier Expert Overlay execution path.

1. **Production config does not run the overlay planner.**
   - Parser/config can create a requested `MoEExpertParallelPlan`.
   - Tests can manually call `MoEExpertParallelPlanner`.
   - Production factory paths currently copy the requested plan into graph config without guaranteeing that model-aware placements were planned.

2. **Continuation domain is not the runtime root.**
   - The graph still follows the externally selected `DeviceId` as default device.
   - `continuation_domain` is metadata for dispatch, not the authoritative owner of routing, combine, logits continuation, arena placement, and return reductions.

3. **Multi-participant domains collapse to one participant.**
   - LocalTP and NodeLocalTP overlay domains are currently lowered to the first participant in graph construction.
   - This blocks real `2x ROCm` and `2 CPU socket rank` execution.

4. **Expert weight residency is not tier-aware.**
   - GPU expert engines are consumed through `ExpertGemmRegistry`.
   - Existing preparation is per selected device and can prepare all experts for that device.
   - There is no production pass that prepares only the experts assigned to each overlay tier according to the planned placement and budgets.

5. **Shared expert placement is not independently resolved.**
   - The shared expert still runs on the graph/default device.
   - `shared_expert_domain` must control where shared expert weights are resident and where shared expert compute executes.

6. **Dispatch descriptors are not consumed by tier compute.**
   - `MoEExpertDispatchStage` can produce per-tier work descriptors.
   - Current tier compute still receives full hidden/routing tensors and filters with masks.
   - Sparse token-row transfer helpers are not yet part of the production graph.

7. **CPU fallback is a helper, not a graph-integrated domain.**
   - `MoEExpertOverlayCPUFallback` proves NodeLocalTP and `TensorParallelExperts` behavior in integration tests.
   - Qwen graph execution does not yet call it as a real fallback domain path.

8. **Cross-domain reduce is CPU-only and host-staged.**
   - `MoEExpertParallelReduceStage` currently supports CPU and sums host-visible dense tensors.
   - That is acceptable only as an explicit correctness bridge, not as the final performant continuation-domain reduction.
   - The production target is a continuation-domain accumulator: CUDA continuation sums on CUDA, ROCm continuation sums on ROCm, and host staging is used only as a transport fallback when direct peer transfer is unavailable.

9. **Parity tests are not yet final proof.**
   - The V2 parity test names exist and topology checks can pass.
   - Parity bodies may still skip when production runtime blockers are detected.

## Subagent Operating Model

Each implementation phase should be handled as follows.

1. Dispatch exactly one phase to a non-readonly coding subagent.
2. Tell the subagent to keep changes scoped to that phase and to run the required tests.
3. After the subagent returns, audit the diff for:
   - phase deliverables and acceptance criteria,
   - code quality and consistency with V2 graph/runtime patterns,
   - ownership and lifetime safety for prepared weights and domain contexts,
   - whether tests are meaningful rather than synthetic or skip-only,
   - whether new failures are unrelated or introduced by the phase.
4. Remediate any issues directly or dispatch a focused fix subagent.
5. Do not begin the next phase until the current phase's required tests pass or a blocker is explicitly documented.

Do not use broad refactors as phase implementation. The fastest route is a sequence of small, auditable changes that progressively converts model-light overlay pieces into production graph/runtime behavior.

## Phase 0: Rebaseline Current State and Lock the Audit Harness

### Goal

Capture the current state before more production work begins, so future subagents cannot accidentally turn skipped parity or synthetic tests into false progress.

### Deliverables

- Document current passing, failing, and skipped overlay tests in the subagent handoff or changelog.
- Confirm that `V2_Integration_Parity_Qwen35MoEExpertOverlay_*` CTest names are concise and stable.
- Confirm that V2 parity bodies clearly distinguish topology checks from real inference parity.
- Add or update comments in the parity fixture explaining every remaining skip condition.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Parallel|Overlay)" --output-on-failure --parallel
```

### Acceptance Criteria

- Current skips are intentional and explain the exact missing production capability.
- No parity body can silently pass without either running real inference or reporting a clear skip reason.
- Topology tests assert non-empty tier assignment for CUDA, ROCm, and CPU when those tiers are part of the layout.

## Phase 1: Production Overlay Planning in InferenceRunnerFactory

### Goal

Ensure production factory paths convert a requested overlay plan into a model-aware planned overlay before graph construction.

### Deliverables

- Add a single production helper that derives `MoEExpertModelMetadata` from loaded model metadata/context.
- If `moe_expert_parallel.enabled` is true and placements are empty, run `MoEExpertParallelPlanner` during runner setup.
- Use the same path for CLI and YAML configurations.
- Preserve manually supplied explicit placements when present.
- Surface validation errors before model execution.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertParallelPlanner|MoEExpertOverlayConfig|InferenceRunnerFactory|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

### Acceptance Criteria

- A production runner config with only domains, tiers, budgets, and residency policy produces complete per-layer expert placements.
- The planner respects explicit tier caps/budgets and assigns fallback experts to CPU.
- Production graph construction no longer falls back to legacy MoE solely because placements were omitted from CLI/YAML.

## Phase 2: Runtime Domain Resolution and Continuation Semantics

### Goal

Make overlay domains resolve to explicit runtime domain descriptors, and make `continuation_domain` authoritative for root activation flow.

### Deliverables

- Add a runtime domain resolution step that maps every overlay domain to:
  - participants,
  - backend,
  - compute kind,
  - primary/root participant when needed,
  - rank/device ownership,
  - domain-scoped collective context.
- Validate that `continuation_domain` and `shared_expert_domain` exist and are reachable.
- Make the continuation domain choose the graph continuation/root device for routing, combine, residual, and logits flow.
- Add diagnostics that print the resolved overlay topology when tracing is enabled.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertOverlay" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(CPUFallback|MultiAccelerator)" --output-on-failure --parallel
```

### Acceptance Criteria

- Layout A resolves ROCm as continuation and CPU as fallback.
- Layout B resolves CUDA as continuation, ROCm as warm/hot accelerator tier, and CPU as fallback.
- Invalid or unreachable domains fail before graph execution.
- No code path treats overlay domains as PP layer ownership.

## Phase 3: Tier-Aware Expert Weight Residency and Preparation

### Goal

Prepare and register expert weights per overlay tier/domain according to planned placement, instead of preparing all experts on the selected runner device.

### Deliverables

- Add an overlay-aware expert preparation request, keyed by layer, expert id, role, device/domain, and residency policy.
- Extend `WeightManager`/prepared-weight plumbing to prepare only assigned routed experts for each domain.
- Populate `ExpertGemmRegistry` for accelerator tiers only for experts assigned to those tiers.
- Preserve CPU fallback ownership for experts assigned to CPU.
- Track per-domain memory budget usage and expert counts.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(WeightManager|ExpertGemmRegistry|MoEExpertParallelPlanner|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

### Acceptance Criteria

- CUDA and ROCm tiers receive only their planned expert subsets.
- CPU fallback receives the complement.
- Shared expert residency is prepared on the configured shared domain.
- Tests prove that a small model cannot accidentally make the CUDA tier own all routed experts unless explicitly configured that way.
- Missing registry entries fail with a placement-specific diagnostic, not a generic graph exception.

## Phase 4: Shared Expert Domain Execution

### Goal

Move shared expert compute from the graph/default device to `shared_expert_domain`.

### Deliverables

- Resolve shared expert stages through the overlay domain descriptor.
- Prepare shared expert weights for the shared domain.
- Transfer or alias normalized hidden state as needed for shared expert execution.
- Return shared partial output to the continuation domain for final combine/reduce.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

### Acceptance Criteria

- Layout A runs shared expert on ROCm continuation tier.
- Layout B runs shared expert on CUDA continuation tier.
- If shared expert does not fit on the configured domain, planning/validation fails or chooses an explicitly configured fallback policy. It must not silently run somewhere else.

## Phase 5: Multi-Participant Domain Lowering

### Goal

Replace primary-participant-only lowering with real LocalTP and NodeLocalTP domain execution.

### Deliverables

- Remove or retire graph lowering that uses only `participants.front()` for multi-participant domains.
- Implement LocalTP execution for accelerator routed tiers.
- Implement NodeLocalTP domain execution for CPU fallback tiers.
- Ensure `ExpertDomainComputeKind` is honored:
  - `ReplicatedExperts` selects exactly one owner per replicated expert contribution.
  - `ExpertIdSharded` maps to stage-local expert-id ownership.
  - `TensorParallelExperts` uses domain-scoped tensor parallel expert GEMM.
- Ensure collectives are domain-scoped and deterministic.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_CPUFallback" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_CPUTensorParallelExperts" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

### Acceptance Criteria

- 2x ROCm tiers use both ROCm participants for tier work.
- 2-rank CPU NodeLocalTP fallback uses both CPU participants.
- Tests fail fast on insufficient MPI ranks or missing accelerator hardware.
- Prepared handles from one domain cannot overwrite handles from another domain.

## Phase 6: Graph-Integrated CPU Fallback TensorParallelExperts

### Goal

Promote `MoEExpertOverlayCPUFallback` from helper/test utility into a production graph-integrated CPU fallback path.

### Deliverables

- Add a compute stage or composite graph component that invokes CPU fallback domain execution.
- Support `TensorParallelExperts` for gate/up column-parallel and down input-parallel fallback expert GEMMs.
- Add domain allreduce for fallback partials.
- Preserve exact routing weights and top-k semantics.
- Keep CPU fallback deterministic for prefill and decode.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_CPUTensorParallelExperts" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
```

### Acceptance Criteria

- CPU fallback computes real cold expert contributions for the same MoE layer as accelerator tiers.
- Both CPU participants contribute to the same selected fallback experts in `TensorParallelExperts` mode.
- Output matches replicated fallback reference within existing tolerances.

## Phase 7: Production Dispatch Descriptors and Sparse Transfer

### Goal

Make tier compute consume `MoEExpertDispatchStage` descriptors, and transfer only the data each non-continuation domain needs.

### Deliverables

- Use dispatch descriptors to identify per-tier selected token rows and expert entries.
- For prefill, transfer only fallback/warm token rows to non-continuation domains.
- For decode, transfer one hidden row plus top-k metadata when fallback tiers are active.
- Preserve token row indices for scatter-add or continuation-domain reduce.
- Keep a dense-transfer compatibility mode behind a debug/config option.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(Sparse|Transfer)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Dispatch|ParallelReduce)Stage" --output-on-failure --parallel
```

### Acceptance Criteria

- Prefill transfer volume scales with routed fallback rows, not full sequence length.
- Decode fallback transfer is one hidden row plus compact routing metadata.
- Sparse and dense compatibility modes produce matching output.
- Dispatch tracing can report selected rows, routed entries, and bytes by domain.

## Phase 8: Cross-Domain Reduce to Continuation Domain

### Goal

Replace the CPU-only dense reduce with a production cross-domain reducer that returns all partials to the continuation domain.

### Deliverables

- Add an explicit cross-domain reduce component that accepts shared, CUDA, ROCm, and CPU partials.
- Support host-staged correctness mode and continuation-domain optimized mode.
- Make coherence transitions explicit through `TransferEngine` or graph coherence APIs.
- Track per-domain transfer bytes and reduce time.
- Preserve input partials unless the plan explicitly permits destructive reduction.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertParallelReduceStage" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

### Acceptance Criteria

- Final MoE partial sum is resident on `continuation_domain`.
- Layout B can reduce ROCm and CPU partials back to CUDA continuation.
- Profiling separates expert compute, transfers, domain-local reductions, and cross-domain final reduce.
- Host-staged reduction is allowed only as a measured correctness fallback.

## Phase 9: Real V2 Parity Inference

### Goal

Turn the existing V2 parity tests into true real-inference proof for the production overlay path.

### Deliverables

- Remove runtime skips for missing `ExpertGemmRegistry` and multi-participant lowering once those blockers are fixed.
- Keep hardware-precondition skips for unavailable CUDA, ROCm, or MPI resources.
- Compare at least `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD` against PyTorch snapshots.
- Assert tier participation for every layout and phase.
- Use thresholds comparable to existing Qwen35 MoE parity.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

### Acceptance Criteria

- The following CTest targets run real inference and pass:
  - `V2_Integration_Parity_Qwen35MoEExpertOverlay_PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
  - `V2_Integration_Parity_Qwen35MoEExpertOverlay_DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
  - `V2_Integration_Parity_Qwen35MoEExpertOverlay_PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
  - `V2_Integration_Parity_Qwen35MoEExpertOverlay_DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- Hot-only sparsity in `MOE_EXPERT_OUTPUT` disappears.
- LM head KL/top-k metrics match the existing Qwen35 MoE parity envelope.
- Decode passes for at least one generated token per final layout.

## Phase 10: MoE Overlay Profiling and Observability

### Goal

Make `LLAMINAR_PROFILING=1` explain where time and memory go across all overlay domains.

### Deliverables

- Add MoE overlay profiling categories for:
  - planner and residency setup,
  - per-domain expert GEMM/GEMV,
  - domain-local allreduce/reduce,
  - cross-domain transfers,
  - cross-domain final reduce,
  - dispatch and sparse packing/scatter,
  - cache hits/misses and expert residency,
  - rebalancing decisions when enabled.
- Add optional trace environment variables:

```text
LLAMINAR_MOE_EP_TRACE=1
LLAMINAR_MOE_EP_DUMP_PLACEMENT=1
LLAMINAR_MOE_EP_TRANSFER_TRACE=1
LLAMINAR_MOE_EP_PROFILE_CSV=1
```

- Emit per-layer/per-domain metrics in a machine-readable form when CSV tracing is enabled.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(Profiling|MoEExpertOverlay)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

### Acceptance Criteria

- Profiling output identifies all configured domains by name.
- Each domain reports assigned experts, executed routed entries, GEMM/GEMV time, transfer time, and reduction time.
- Cache metrics show hot-cache capacity, used bytes, resident expert count, hits, misses, and fallback count.
- Profiling does not force unnecessary D2H transfers in non-profiled runs.

## Phase 11: Release Benchmark and Performance Tuning

### Goal

Prove the production overlay path is not merely correct, but practical.

### Deliverables

- Add or document a release benchmark invocation for Layout A and Layout B.
- Benchmark both prefill and decode.
- Capture `LLAMINAR_PROFILING=1` output for each layout.
- Include per-domain throughput and time breakdown in benchmark output.
- Add regression guards where practical, without making CI depend on unavailable local hardware.

### Required Commands

```bash
cmake --build build_v2_release --parallel

LLAMINAR_PROFILING=1 \
./build_v2_release/llaminar2 benchmark \
  -m models/<qwen35-moe-model>.gguf \
  --moe-expert-overlay tiered \
  --moe-expert-overlay-continuation <domain> \
  --moe-expert-overlay-shared-domain <domain> \
  --moe-expert-overlay-domain "..." \
  --moe-expert-overlay-tier "..."
```

Concrete commands should be added once the production CLI is stable and the model path is known.

### Acceptance Criteria

- Release benchmark runs the same production overlay path as parity.
- Profiling proves each hardware tier is active.
- For steady-state decode, GEMM/GEMV is the dominant active compute category per participating device/domain.
- Allreduce, cross-domain reduce, transfer, or stall time is not the dominant cost after warmup. If it is, the phase is not accepted until the root cause is addressed or an explicit performance follow-up is filed with measurements.
- Benchmark output includes enough data to compare overlay performance against a single-domain baseline and a CPU-only fallback baseline.

## Phase 12: Hardening, Cleanup, and Documentation

### Goal

Remove temporary scaffolding and leave the feature maintainable for future MoE work.

### Deliverables

- Remove obsolete skip messages, TODOs, and primary-participant-only warnings that are no longer true.
- Update CLI/YAML documentation with final examples for Layout A and Layout B.
- Update any handover docs that claimed sequential PP was the target.
- Add a changelog entry summarizing implementation, parity results, and benchmark results.
- Store no secrets and no machine-specific absolute paths in committed docs.

### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert|V2_Integration_.*MoEExpert|^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_" --output-on-failure --verbose
```

### Acceptance Criteria

- Docs match shipped behavior.
- Final parity and benchmark commands are recorded with results.
- No test passes solely because the real production path was skipped.
- The final code path is compatible with existing prepared-weight ownership rules and does not reintroduce global model-weight GEMM caches.

## Implementation Guardrails

- Do not fix production gaps only inside the parity fixture.
- Do not express same-layer expert overlay as sequential PP layer ownership.
- Do not prepare all routed experts on CUDA/ROCm unless the configured residency plan says to do so.
- Do not let synthetic/model-light integration tests substitute for real V2 parity.
- Do not relax parity thresholds without comparing against the existing Qwen35 MoE suite and documenting the reason.
- Do not introduce domain-global prepared-weight caches that can be overwritten by another domain.
- Do not artificially limit build or test parallelism.

## Current Checkpoint Bridge Plan

This section is the dispatch plan from the current checkpoint, rather than the original coarse roadmap from project start.

Current checkpoint assumptions:

- Phases 0 through 5B have been accepted.
- Phase 5C now provides a graph-independent accelerator `LocalTP` `TensorParallelExperts` executor that can be tested with synthetic FP32 tensors and a domain-scoped `LocalTPContext`.
- These pieces are not enough for the final gate. The real Qwen3.5 overlay path can still be blocked by Phase 5D accelerator `LocalTP` graph wiring, helper-only sparse transfer/reduce paths, cross-domain reduce gaps, or parity skips.

For the remaining work, dispatch the bridge phases below. Each bridge phase should remove one explicit production blocker or convert one helper/model-light component into a real Qwen graph/runtime path. Do not count model-light tests, topology-only tests, or intentionally skipped parity bodies as acceptance for a bridge phase.

### Phase 5A: Audit and Stabilize Current Multi-Domain Lowering Checkpoint

#### Goal

Close the current Phase 5 audit before new feature work continues.

#### Deliverables

- Audit the current diff for phase bleed and document which pieces are accepted as Phase 5 support code versus deferred bridge-phase work.
- Confirm CPU `NodeLocalTP` fallback paths are graph-integrated only for CPU fallback domains, not used to claim accelerator `LocalTP` support.
- Confirm active accelerator `LocalTP` routed tiers fail fast with explicit diagnostics instead of lowering to `participants.front()`.
- Confirm every remaining real-parity skip maps to a named bridge phase below.
- Fix or revert any changes that make synthetic/model-light tests look like final parity.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(Qwen35MoEExpertOverlayGraph|MoEExpertOverlayRuntimePlan|MoEExpertDispatchStage|MoEExpertParallelReduceStage)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts|MultiAcceleratorTiers)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- CPU fallback integration is either accepted with passing tests or explicitly fixed before proceeding.
- No active accelerator `LocalTP` domain is silently executed on only its primary participant.
- The parity suite still cannot pass real overlay parity by topology-only coverage or synthetic fallback.
- The next blocking message in each real parity body references either accelerator residency, accelerator `LocalTP`, dispatch/sparse transfer, cross-domain reduce, or hardware availability.

### Phase 5B: Production Accelerator Residency and Registry Closure

#### Goal

Eliminate missing `ExpertGemmRegistry` blockers for planned accelerator tiers by hydrating only the experts assigned to each overlay domain/participant.

#### Deliverables

- Wire the overlay preparation plan into the real model load / runner setup path for accelerator domains, not just model-light tests.
- Populate accelerator expert GEMM handles for every planned routed expert role on every participant that owns or tensor-parallel-shards that expert.
- Keep CUDA/ROCm residency bounded by the planned placement, tier caps, and memory budgets.
- Prepare shared expert weights on `shared_expert_domain` through the same production residency path.
- Make prepared handle lookup diagnostics include layer, expert id, role, tier, domain, participant, and device.
- Preserve domain-scoped ownership if two logical domains use the same physical device; one domain's prepared handles must not overwrite another domain's handles.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(WeightManager|ExpertGemmRegistry|MoEExpertOverlayPreparation|InferenceRunnerFactory|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout B's CUDA routed tier has complete gate/up/down registry entries for its planned hot experts and no entries for CPU fallback-only experts.
- ROCm participants have complete registry entries for the experts assigned to the ROCm tier according to the chosen `TensorParallelExperts` ownership contract.
- CPU fallback remains the complement for cold routed experts.
- Real parity no longer skips solely because a planned accelerator tier is missing registry entries.

### Phase 5C: Accelerator LocalTP TensorParallelExperts Runtime

#### Goal

Implement real accelerator-domain `TensorParallelExperts` execution for multi-participant `LocalTP` domains, starting with ROCm/RCCL because both final layouts require a two-ROCm routed tier.

Current status: implemented as an independent runtime helper for synthetic/reference coverage. Production Qwen graph wiring remains Phase 5D.

#### Deliverables

- Add a domain-scoped accelerator expert executor or compute stage for `ExpertDomainKind::LocalTP` plus `ExpertDomainComputeKind::TensorParallelExperts`.
- Shard each selected expert's gate/up/down work across domain participants using the same math as the CPU tensor-parallel fallback:
   - gate/up column or intermediate split,
   - down input/intermediate split,
   - domain allreduce of the expert output partial,
   - exact route-weight application and top-k semantics.
- Use the resolved runtime domain descriptor for participants, rank/device ownership, and collective context.
- Keep the executor independent enough to test with synthetic FP32 tensors before the full Qwen graph path is enabled.
- Record per-participant executed expert ids and routed-entry counts for tests/profiling.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlayRuntimePlan|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

Hardware-specific tests may skip only when the build lacks ROCm/RCCL support or fewer than two ROCm devices are visible.

#### Acceptance Criteria

- A 2x ROCm `TensorParallelExperts` synthetic/reference test executes both ROCm participants and matches a replicated full-expert reference within existing MoE tolerances.
- The runtime no longer uses `participants.front()` for active ROCm routed tier work.
- Domain-scoped collective failures are reported as accelerator `LocalTP` execution errors, not generic graph construction failures.

### Phase 5D: Qwen Graph Integration for Accelerator Multi-Participant Routed Tiers

#### Goal

Wire the accelerator `LocalTP` executor into the real Qwen3.5 MoE overlay graph so routed accelerator tiers are first-class runtime domains.

#### Deliverables

- Replace the current active-accelerator-`LocalTP` fail-fast graph path with the Phase 5C executor for planned routed tiers.
- Ensure the Qwen graph creates one logical tier contribution per routed tier and keeps partial output lifetimes/domain ownership distinct.
- Ensure Layout A's ROCm shared/hot domain and Layout B's ROCm hot domain both use two ROCm participants for routed work.
- Keep CUDA `ReplicatedExperts` routed tier execution as a single-domain path for Layout B.
- Keep CPU fallback tier execution through the graph-integrated CPU fallback path.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*Qwen35MoEExpertOverlayGraph" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout A can build the real graph with an active ROCm `TensorParallelExperts` tier without primary-participant lowering.
- Layout B can build the real graph with active CUDA, ROCm, and CPU routed tiers.
- Parity skips, if any remain, are no longer due to accelerator multi-participant lowering.

### Phase 6A: Production Dispatch Descriptor Consumption and Sparse Transfer

#### Goal

Move dispatch descriptors and sparse token-row transfer from helper/model-light status into the production Qwen overlay path.

#### Deliverables

- Make each non-continuation tier consume `MoEExpertDispatchStage` descriptors rather than re-filtering full hidden/routing tensors independently.
- Transfer only selected token rows and compact top-k metadata to non-continuation domains in prefill.
- Use the one-hidden-row decode fast path when `seq_len == 1`.
- Preserve original token-row indices for return scatter-add and final reduce.
- Keep dense full-sequence transfer available only as an explicit debug/compatibility mode.
- Add tracing hooks for selected rows, routed entries, and bytes by domain.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert(Dispatch|TokenRowTransfer|ParallelReduce)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay.*(Sparse|Transfer|CPUFallback|MultiAcceleratorTiers)" --output-on-failure --parallel
```

#### Acceptance Criteria

- Prefill transfer volume scales with selected routed rows for ROCm/CPU fallback tiers, not full sequence length.
- Decode fallback transfer is one hidden row plus compact routing metadata.
- Sparse and dense compatibility modes produce matching MoE combined output.
- Tests assert non-zero executed work and non-zero selected rows for every active non-continuation tier.

### Phase 7A: Cross-Domain Reduce Back to Continuation Domain

#### Goal

Replace helper-only host dense reduction with an explicit production reducer that returns shared, CUDA, ROCm, and CPU tier partials to `continuation_domain`, with the continuation device owning final accumulation for optimized CUDA/ROCm layouts.

#### Deliverables

- Add a production cross-domain reduce component with explicit `HostStagedCorrectness` and `ContinuationDeviceOptimized` modes.
- Keep `HostStagedCorrectness` clearly measured and diagnosable as a bridge path.
- Make `ContinuationDeviceOptimized` the production path for CUDA/ROCm continuation. This mode streams partials to the continuation device and performs the final sum/scatter-add on that device, not in a CPU loop.
- Prefer compact row partial payloads of `(row_id, partial_vector)` for non-continuation tiers. Dense `[seq_len, d_model]` partial transfers remain available only for debug/compatibility or when a tier genuinely produced dense output.
- Support transport fallbacks independently from accumulation ownership: ROCm-to-CUDA may be direct peer copy where available or ROCm-to-pinned-host-to-CUDA where required, but accumulation still happens on CUDA for Layout B.
- Make all coherence transitions explicit through `TransferEngine` or graph coherence APIs.
- Reduce shared expert and routed tier partials into the final MoE output resident on the continuation domain.
- Keep partial tensors immutable unless the execution plan explicitly permits destructive reduction.
- Report transfer bytes, reduce time, and final output residency.
- Report whether each partial was accumulated on continuation device, host-staged and then device-accumulated, or fully host-summed correctness fallback.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpertParallelReduceStage" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout A reduces CPU fallback partials back to ROCm continuation.
- Layout B reduces ROCm and CPU partials back to CUDA continuation.
- Layout B's optimized path performs final accumulation on CUDA. CPU summing may remain tested as `HostStagedCorrectness`, but it is not accepted as the production/performance path.
- Sparse routed-row partial return is used for non-continuation tiers when dispatch descriptors identify a strict subset of token rows.
- `MOE_EXPERT_OUTPUT` no longer contains hot-only sparsity caused by missing cold/warm tier return paths.
- Host-staged transport and host-summed correctness fallback are separately reported and cannot be mistaken for continuation-owned optimized reduction.

### Phase 8A: Real V2 Overlay Parity Unskip

#### Goal

Turn the overlay parity suite into real Qwen3.5 inference proof for both final layouts.

#### Deliverables

- Remove runtime skips for missing accelerator registry entries, missing accelerator multi-participant lowering, helper-only dispatch, and helper-only reduction.
- Keep only hardware/build/model precondition skips.
- Assert non-zero assigned experts and non-zero executed routed entries for CUDA, ROCm, and CPU tiers in Layout B, and ROCm/CPU tiers in Layout A.
- Compare `MOE_EXPERT_OUTPUT`, `MOE_COMBINED_OUTPUT`, and `LM_HEAD` against PyTorch snapshots.
- Exercise the same continuation-domain reduce mode intended for production when CUDA/ROCm continuation hardware is available. Host-summed correctness fallback may have dedicated tests, but it must not be the only path proving Layout B parity.
- Use the existing Qwen35 MoE `BackendThresholds` envelope unless a separately documented baseline comparison justifies a change.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- The four real prefill/decode overlay parity CTests run inference and pass for available hardware.
- No parity body passes by topology-only execution or synthetic/model-light fallback.
- Tier participation counters prove that the exercised tokens route work to all configured active tiers.
- Continuation-owned reduce diagnostics prove final accumulation happened on CUDA for Layout B and ROCm for Layout A when the optimized path is available.
- LM head KL/top-k metrics match the existing Qwen35 MoE parity envelope.

### Phase 9A: Overlay Profiling and Diagnostics Closure

#### Goal

Make profiling explain the real production overlay execution across all domains.

#### Deliverables

- Emit per-domain/per-layer metrics for assigned experts, resident experts, routed entries, selected token rows, transfer bytes, GEMM/GEMV time, domain allreduce time, and final reduce time.
- Add or finalize `LLAMINAR_MOE_EP_TRACE`, `LLAMINAR_MOE_EP_DUMP_PLACEMENT`, `LLAMINAR_MOE_EP_TRANSFER_TRACE`, and `LLAMINAR_MOE_EP_PROFILE_CSV`.
- Ensure profiling does not add unnecessary device-to-host transfers when disabled.
- Surface final blocker categories clearly: compute, transfer, local collective, cross-domain reduce, or idle/stall.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(Profiling|MoEExpertOverlay)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- `LLAMINAR_PROFILING=1` identifies every configured domain by name in Layout A and Layout B.
- Every active domain reports non-zero routed work and a complete compute/transfer/reduce breakdown.
- Profiling output can distinguish direct peer transfer, host-staged transport, continuation-device accumulation, and host-summed correctness fallback.

### Phase 9B: Unified Execution Domain Contract

#### Goal

Converge the older named-domain TP/PP system and the MoE overlay domain system on one canonical execution-domain value type and resolver before the release benchmark locks in the public configuration model.

#### Deliverables

- Close `DomainDefinition` and `ExpertComputeDomain` into a single canonical domain contract, or into one canonical type plus thin compatibility aliases during migration.
- Preserve all fields required by both systems: name, participants/devices, scope, backend, owner/ranks, weights where meaningful, and domain-internal compute kind.
- Move role-specific meaning out of the domain type. PP layer ownership, TP sharding, continuation ownership, shared expert placement, and routed expert tiers should be placements over domains, not separate definitions of what a domain is.
- Replace duplicated parser, validation, normalization, and diagnostics for named domains and MoE overlay domains with shared code or a shared conversion path.
- Ensure domain identity is stable enough for domain-scoped prepared weights, `ExpertGemmRegistry` entries, collective contexts, profiling, and trace output. Two logical domains that share one physical device must still have separate logical ownership where configured.
- Keep existing CLI/YAML syntax working during the transition, but route both old named-domain and overlay-domain inputs through the unified representation.
- Add migration comments near the compatibility aliases so future phases do not extend the old split types.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(OrchestrationConfigParser|ConfigValidator|ExecutionPlanBuilder|MoEExpertOverlayRuntimePlan|MoEExpertOverlayExecutionPlan|MoEExpertParallelPlanner)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- `DomainDefinition` and `ExpertComputeDomain` can no longer drift in supported fields, parser behavior, validation rules, or diagnostics.
- The same resolved domain inventory feeds legacy simple/named-domain TP/PP planning and MoE overlay runtime planning.
- Overlay domains are still rejected as PP layer ownership; same-layer expert overlay remains a MoE placement over domains.
- Domain-scoped prepared weights and collective contexts remain isolated by logical domain name even when domains share physical participants.
- Existing accepted Layout A and Layout B parity still pass after the type convergence.

### Phase 9C: Unified Placement UX and Configuration Model

#### Goal

Make TP, PP, and MoE overlay feel like one Llaminar orchestration language: define hardware domains once, then place root flow, layer ranges, tensor-parallel work, and same-layer MoE expert roles onto those domains.

#### Deliverables

- Define the final user-facing model for advanced heterogeneous runs:
   - domains: hardware/collective groups,
   - root or continuation placement: activation/logits owner,
   - base/non-expert model placement: dense path and non-expert weights,
   - PP layer placement: layer ranges over domains,
   - MoE placement: shared expert domain plus ordered routed tiers over domains.
- Add or finalize an explicit base/non-expert model domain option for overlay runs. Do not require `-d cpu` or any other fake root-device workaround to pass validation.
- Decide the final CLI compatibility story:
   - keep `-d`, `-tp`, `--tp-devices`, and `--pp-stage` as concise simple-mode shorthands,
   - prefer unified named domains for advanced heterogeneous configs,
   - either deprecate `--moe-expert-overlay-domain` in favor of shared domain definitions or make it a strict alias that populates the same canonical domain inventory.
- Update YAML examples so Layout A and Layout B define each hardware domain once and reference those domains from overlay placements.
- Update validation so ambiguous combinations fail early with user-facing guidance. In particular, `-d` should not silently override or conflict with `continuation_domain`, base model domain, or overlay tier domains.
- Make `--explain-placement` render one coherent domain/placement tree for simple TP/PP, named-domain PP, and MoE overlay.
- Ensure `benchmark`, `oneshot`, `serve`, and parity all consume the same final config shape rather than mode-specific wrappers.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(OrchestrationConfigParser|ConfigValidator|ExecutionPlanBuilder|OrchestrationRunnerFactory|MoEExpertOverlayExecutionPlan|MoEExpertOverlayRuntimePlan)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- Layout A and Layout B can be expressed without `-d cpu` or duplicated domain definitions.
- Existing simple commands for single-device, simple TP, explicit TP, and PP remain valid or receive clear migration diagnostics.
- `--pp-stage` remains layer placement only and cannot be used to model same-layer hot/warm/cold expert overlay.
- `--explain-placement` makes the continuity between legacy TP/PP and MoE overlay visible: one domain inventory, multiple placement kinds.
- The Phase 10C benchmark commands use this final CLI/YAML shape.

### Phase 10A: Graph-Native Overlay Dispatch Collectives

#### Goal

Bring MoE overlay back into the original graph execution model by treating overlay tier dispatch as a graph-level sparse collective, not as a separate rank-local coordinator or ad hoc worker command layer.

Each continuation graph keeps its normal structure: single-device, accelerator `LocalTP`, `LocalPP`, and future hybrids all reach explicit overlay dispatch points in graph order. Every participant in the dispatch group enters the stage for the same logical layer/token/microbatch. Participants with routed work submit compact dispatch descriptors and selected token rows; participants with no routed work submit an empty/no-op request. CPU fallback and other remote overlay domains are graph participants/endpoints for that dispatch group, not a second orchestration plane.

#### Deliverables

- Replace the abandoned rank-local worker-coordinator direction with a graph-native overlay dispatch collective.
- Keep one graph per continuation device/rank participant wherever the original system already does so; do not introduce a separate scheduler that owns layer progress outside the graph.
- Define an `OverlayDispatchGroup`-style contract, analogous to a collective communicator, with cached integer identity for domain id, layer id, dispatch group id, participant count, participant index, owner/executor participant(s), and stage/microbatch/decode sequence.
- Make overlay tier stages call a backend-shaped API such as `MoEOverlayBackend::dispatch(group, request, output)` rather than directly sending fallback-domain commands from arbitrary per-device graph workers.
- Require all participants in a dispatch group to enter the dispatch point in graph order:
   - single-device continuation: one continuation participant plus the fallback endpoint when needed;
   - accelerator `LocalTP`: all local TP graph participants enter, even when one participant has no cold routed rows;
   - `LocalPP`: only participants that own the pipeline stage/layer for that request enter the dispatch group; unrelated local devices are not forced into every fallback call.
- Make empty/no-op dispatch a first-class request type with the same request identity as real work. No-op participants must be cheap and must still participate in ordering, cancellation, and tracing.
- Have CPU fallback / remote-domain participants consume dispatch-point messages keyed by dispatch group and sequence, not duplicate same-rank command streams from individual local device workers.
- Preserve exact route weights, selected token rows, sparse transfer metadata, output scatter semantics, and domain-local `TensorParallelExperts` behavior already implemented by prior phases.
- Keep cross-domain reduce graph-native: fallback partials return as explicit dispatch results/partials and feed the existing `MoEExpertParallelReduceStage` or its successor, with continuation-domain accumulation semantics preserved.
- Integrate failure/cancel semantics at the dispatch collective boundary: a failure from any participant wakes all local participants, sends cancel/no-op completion to remote participants as needed, and prevents remote fallback ranks from waiting for the full collective timeout.
- Add trace/profiling hooks that can explain dispatch wait time, no-op count, routed request count, selected row count, remote endpoint work count, transfer bytes, and cancel events without forcing D2H transfers or hot-path string formatting.

#### Performance Contract

Overlay dispatch is part of the decode hot path. The graph-native path must preserve the low-overhead feel of existing TP collectives.

- Preallocate dispatch request/result slots during graph/runtime initialization; no heap allocation in the per-token path.
- Use cached integer ids for dispatch groups, domains, layers, participants, and request slots; no hot-path string lookups, YAML/config maps, dynamic casts for protocol identity, or formatted logging unless tracing is enabled.
- Use fixed-size local rendezvous structures for accelerator `LocalTP` participants and avoid OS blocking in the normal all-participants-ready path.
- Keep no-op dispatch close to a branch plus slot publish in the uncontended path.
- Keep single-device continuation close to the existing direct stage call when no remote/fallback participant is active.
- Measure dispatch-collective overhead separately from remote expert compute, transfers, local allreduce, and final reduce when `LLAMINAR_PROFILING=1` is enabled.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEOverlay.*Dispatch|MoEExpertDispatchStage|MoEOverlayDomainRuntime|Qwen35MoEExpertOverlayGraph|TPWorkerPool)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts|MultiAcceleratorTiers)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

#### Acceptance Criteria

- The production overlay path remains graph-native: one graph per continuation device/rank participant, with overlay work represented by explicit graph dispatch stages.
- Single-device continuation can call CPU fallback through the dispatch collective without fake TP participation or a separate rank-local coordinator.
- Accelerator `LocalTP` participants all rendezvous at the same dispatch point for a logical layer/token/microbatch request, and participants with no routed work submit no-op requests rather than skipping the stage.
- Accelerator `LocalTP` emits one coherent fallback-domain dispatch sequence for the logical request; remote fallback workers no longer observe duplicate same-rank command streams from multiple local device workers.
- `LocalTP` participants all receive the correct returned fallback partials and cannot advance independently past a failed, canceled, or incomplete dispatch point.
- `LocalPP` dispatch groups are scoped to the owning pipeline stage/layer/microbatch; unrelated local devices are not required to enter every fallback call.
- CPU fallback and remote overlay-domain participants consume ordered dispatch-point messages and can distinguish real work, no-op, cancel, and shutdown without relying on accidental rank-global side effects.
- Abort/cancel tests prove no local worker or remote fallback rank waits for the full collective timeout after a dispatch-side failure.
- Profiling can report graph-native dispatch overhead, and release-mode no-op/single-device overhead is below 1 us per decode fallback request while `LocalTP` dispatch overhead remains a small fraction of remote expert compute.

### Phase 10B: Collective Contract Hardening

#### Goal

Make dense TP, overlay tier TP, and remote fallback dispatch collectives impossible to confuse. A release benchmark is not accepted while a collective hang can only be diagnosed by waiting for `LLAMINAR_TP_COLLECT_TIMEOUT_MS`.

The hard rule is: every collective belongs to exactly one resolved execution domain, uses exactly one owner-created context for that domain, and advances through a participant-visible ordered sequence. If a participant skips, duplicates, interleaves, or uses a sibling context for the same logical sequence, debug/integration builds must fail before NCCL/RCCL can hang.

#### Deliverables

- Add a lightweight collective trace/contract mode for LocalTP and overlay dispatch. It should stamp each collective with domain name, context identity, backend identity, stage/dispatch group name, sequence number, device/participant index, element count, dtype, stream pointer, and tensor pointer.
- Trace dense continuation LocalTP separately from overlay expert-domain LocalTP and CPU fallback NodeLocalTP. The logs must make it obvious whether `GraphConfig::tp_ctx`, named-domain `domain_tp_contexts`, and overlay dispatch runtimes are sharing or separating contexts intentionally.
- Add first-pass sequence validation for LocalTP on-stream allreduce: all participants in one domain/context must issue the same stage/sequence/count/dtype tuple in the same order. A mismatch should return a clear error and request collective abort rather than waiting for the generic TP timeout.
- Ensure `RankOrchestrator` worker enter/leave logs can distinguish a worker stuck inside `runner->forward(...)` from a worker that returned but is waiting on GPU completion or result collection.
- Audit root continuation graph construction so dense base-model collectives use the root `RankOrchestrator` LocalTP context, while overlay-domain collectives use explicit domain-scoped contexts from the resolved runtime registry. Graph builders must not independently create duplicate RCCL/NCCL contexts for the same logical domain.
- Add context-identity diagnostics at `LocalTPContext` construction, compute stream registration, `RCCLBackend`/`NCCLBackend` coordinator creation, and on-stream allreduce launch.
- Make timeout handling abort the affected LocalTP context/communicator when a TP worker does not complete, so failed benchmark attempts do not leave live child ranks holding accelerator memory.
- Document the reduced-context Layout A triage sequence: first run with allreduce enabled and contract tracing, then run once with `LLAMINAR_SKIP_ALLREDUCE=1` only to prove or disprove dense LocalTP allreduce as the hang site.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(LocalTPContext|TPWorkerPool|MoEOverlay.*Dispatch|MoEOverlayDomainRuntime|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts|MultiAcceleratorTiers)" --output-on-failure --parallel
```

Hardware-specific reduced-context triage, when the Qwen3.5 MoE model and two ROCm devices are available:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
   --config configs/qwen35-moe-overlay-layout-a.yaml \
   -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   -p Hello -n 0 -t 0 -c 512

LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE=1 \
LLAMINAR_SKIP_ALLREDUCE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
   --config configs/qwen35-moe-overlay-layout-a.yaml \
   -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   -p Hello -n 0 -t 0 -c 512
```

#### Acceptance Criteria

- Layout A reduced-context benchmark no longer hangs silently: it either completes prefill, fails with a collective contract diagnostic that names the mismatched domain/context/sequence, or fails at a later named stage with enough worker-boundary evidence to continue the investigation.
- Dense continuation LocalTP allreduces are proven to use one shared root context/coordinator across ROCm:0 and ROCm:1 for the root forward.
- Overlay-domain LocalTP contexts are either intentionally aliased to the root context or intentionally separate by logical domain, and the trace output makes that choice explicit.
- If two logical domains share physical devices, their prepared weights and collective contexts remain logically isolated unless the domain resolver explicitly aliases the collective context.
- `LLAMINAR_SKIP_ALLREDUCE=1` is retained only as a diagnostic switch; no production or parity acceptance depends on skipped allreduce.
- A timed-out TP worker triggers local context abort/communicator abort so follow-up benchmark runs do not require manual cleanup of live accelerator-owning children.
- The investigation result is recorded before Phase 10C begins: root cause fixed, or a precise blocker documented with context/sequence evidence.

### Phase 10C: Release Benchmark and Performance Gate

#### Goal

Prove the real overlay path is practical in release builds and identify performance blockers before final cleanup.

#### Deliverables

- Add concrete Layout A and Layout B benchmark commands with the final CLI/YAML syntax.
- Benchmark prefill and steady-state decode with `LLAMINAR_PROFILING=1`.
- Report per-domain throughput, GEMM/GEMV time, allreduce time, transfer time, final reduce time, and residency/cache metrics.
- Compare against a single-domain accelerator baseline and CPU-only fallback baseline where hardware allows.
- File explicit follow-up issues for any measured performance blocker that cannot be fixed in this phase.

#### Required Commands

```bash
cmake --build build_v2_release --parallel

LLAMINAR_PROFILING=1 \
./build_v2_release/llaminar2 benchmark \
   -m models/<qwen35-moe-model>.gguf \
   --config configs/<layout-a-or-b-overlay>.yaml
```

#### Acceptance Criteria

- Release benchmark uses the same production graph path as parity.
- Layout B benchmark uses CUDA-owned final accumulation. A benchmark that falls back to CPU dense summing is a correctness bridge measurement, not release-performance acceptance.
- GEMM/GEMV is the dominant active compute category per participating device/domain after warmup.
- Transfer, allreduce, final reduce, or idle/stall time is not the dominant cost unless an explicit measured follow-up issue is attached.
- Benchmark output is sufficient to reproduce the run and compare against baselines without machine-specific paths in committed docs.

### Phase 11A: Final Hardening and Documentation

#### Goal

Remove temporary bridge scaffolding and leave the overlay feature maintainable.

#### Deliverables

- Remove obsolete skip messages, TODOs, and fail-fast text that referenced now-completed bridge phases.
- Update CLI/YAML docs with final Layout A and Layout B examples.
- Update this execution plan or a changelog with accepted parity and benchmark results.
- Ensure no committed docs contain secrets or machine-specific absolute paths.
- Confirm the code path preserves prepared-weight ownership rules and does not reintroduce global all-expert accelerator caches.

#### Required Tests

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*MoEExpert|V2_Integration_.*MoEExpert|^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_" --output-on-failure --verbose
```

#### Acceptance Criteria

- Docs match shipped behavior.
- No test passes solely because the real production path was skipped.
- Final parity and benchmark results are recorded.
- The final implementation is compatible with existing prepared-weight ownership and lifetime rules.

## Suggested Phase Dispatch Order

The original coarse sequence remains useful as architectural context:

```text
0. Rebaseline and lock audit harness
1. Production overlay planning in InferenceRunnerFactory
2. Runtime domain resolution and continuation semantics
3. Tier-aware expert weight residency and preparation
4. Shared expert domain execution
5. Multi-participant domain lowering
6. Graph-integrated CPU fallback TensorParallelExperts
7. Production dispatch descriptors and sparse transfer
8. Cross-domain reduce to continuation domain
9. Real V2 parity inference
10. MoE overlay profiling and observability
11. Release benchmark and performance tuning
12. Hardening, cleanup, and documentation
```

From the current checkpoint, use this bridge dispatch order instead:

```text
5A. Audit and stabilize current multi-domain lowering checkpoint
5B. Production accelerator residency and registry closure
5C. Accelerator LocalTP TensorParallelExperts runtime
5D. Qwen graph integration for accelerator multi-participant routed tiers
6A. Production dispatch descriptor consumption and sparse transfer
7A. Cross-domain reduce back to continuation domain
8A. Real V2 overlay parity unskip
9A. Overlay profiling and diagnostics closure
9B. Unified execution domain contract
9C. Unified placement UX and configuration model
10A. Graph-native overlay dispatch collectives
10B. Collective contract hardening
10C. Release benchmark and performance gate
11A. Final hardening and documentation
```

The project should stop advancing phases whenever the latest subagent work fails its acceptance gate. Fix the current phase first, then continue.