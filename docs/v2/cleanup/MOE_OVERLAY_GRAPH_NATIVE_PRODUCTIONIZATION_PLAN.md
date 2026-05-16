# MoE Overlay Graph-Native Productionization Plan

**Date:** 2026-05-15  
**Status:** Follow-up implementation plan  
**Predecessor:** `docs/v2/cleanup/MOE_OVERLAY_GRAPH_NATIVE_RESET_PLAN.md`  
**Scope:** Qwen3.5 MoE graph-native overlay production wiring, deletion of the failed shadow LocalTP overlay runtime, expert rebalancing / hot expert caching, real-model PyTorch parity, profiling metrics, and Release benchmark/tuning.

## Executive Summary

The graph-native MoE overlay reset established the correct execution shape: every overlay participant enters explicit sparse graph collectives, local expert stages own only local work, and routed expert ownership is whole-expert expert parallelism rather than hidden tensor parallelism. The next step is to make that path the production path for Qwen3.5 MoE and then prove it with real 35B model parity and Release benchmarks.

This plan has four outcomes:

1. **Production lock-in:** Qwen3.5 MoE overlay inference must use graph-native sparse dispatch, participant-local expert compute, and sparse return/reduce. The old overlay domain runtime and shadow LocalTP expert execution must be impossible to ship accidentally.
2. **Dead-code removal:** Legacy bridge/runtime code should be deleted or quarantined behind explicit legacy-only tests until the production path no longer depends on it.
3. **Hot expert placement:** The routed-tier planner should support measured expert rebalancing and GPU hot-cache policies for both all-GPU and mixed GPU/CPU tier configurations.
4. **Measured performance:** Real Qwen3.5 35B MoE PyTorch parity must pass on the same production path that benchmark mode uses, and Release benchmarks with `LLAMINAR_PROFILING=1` must expose enough MoE overlay metrics to diagnose tier placement, sparse transport, and expert kernel behavior.

## Operating Model

We will continue the same implementation workflow used for the graph-native reset:

1. Gather context for exactly one phase.
2. Dispatch a subagent with a bounded coding/test task for that phase.
3. Audit the returned diff against the phase acceptance criteria.
4. Run the phase gate and relevant prior gates.
5. Only then advance to the next phase.

Subagents may implement code and tests, but the main agent must audit for architecture drift, dead-code leaks, weak tests, and accidental resurrection of the legacy runtime.

## Non-Negotiable Invariants

1. **Graph-native only in production.** Production Qwen3.5 MoE overlay graphs must not lower to `MoEOverlayDomainRuntimeStage`, `MoEExpertOverlayLocalTPStage`, `MoEOverlayCPUFallbackParticipantRunner`, or hidden endpoint receive pumps.
2. **No shadow LocalTP expert runtime.** Routed expert tiers with multiple devices mean disjoint whole-expert owners by participant unless a future explicit `TensorParallelExperts` feature is enabled with its own reduction stage.
3. **One canonical owner per routed expert.** Every routed expert has exactly one execution owner per layer/step, even if several devices keep cached resident copies for future use.
4. **CPU fallback is optional.** All-GPU configurations are valid when accelerator tiers cover every routed expert. Mixed GPU/CPU configurations use CPU as capacity fallback, not as the default path.
5. **Parity before tuning.** Release benchmarking and tuning start only after real Qwen3.5 35B MoE parity passes for the production path on the project hardware lane.
6. **Measure before optimizing transport.** Host-staged sparse payload transport remains acceptable until profiling shows it is the limiter.
7. **No full hidden-state transfers by accident.** One-token decode should move compact routed payloads, not repeatedly transfer dense activations, outside explicit diagnostics.

## Target Production Runtime

The production overlay path should look like this for each MoE layer:

```text
Continuation Dense Graph
  -> Router / routing indices and weights
  -> MoESparseDispatchStage          keyed by generation/step/layer/tier
  -> MoELocalExpertStage             local whole-expert compute only
  -> MoESparseReturnReduceStage      compact return + root scatter-add
  -> Continuation import             TP broadcast/materialization when needed
  -> Continue dense graph
```

For dense TP continuation domains:

- replicated-hidden first implementation: only the logical continuation root packs routed rows;
- other continuation participants publish no-op sparse dispatch payloads;
- return/reduce targets the logical continuation root;
- continuation import materializes `MOE_COMBINED_OUTPUT` into the layout required by the next dense stage.

## Expert Rebalancing And Hot Expert Cache Design

### Inputs

The rebalancer should consume measured or sampled data rather than hardcoded names:

- prefill expert activation histogram by layer/expert;
- decode expert activation histogram by layer/expert;
- per-device expert GEMM throughput for prefill;
- per-device expert GEMV throughput for decode;
- sparse dispatch and return transport cost by source/target participant pair;
- VRAM budget and resident expert bytes per tier/device;
- current owner map and current resident cache map;
- expert move/repack cost;
- user policy: static, histogram, explicit masks, or adaptive rebalancing.

### Scoring Model

The first policy can be deterministic and simple:

```text
benefit(layer, expert, tier) =
    expected_hits(layer, expert)
  * (fallback_compute_cost - tier_compute_cost - sparse_transport_cost)
  - residency_bytes_penalty
  - rebalance_churn_penalty
```

Decode and prefill should be scored separately because decode is expected to be GEMV-like and latency sensitive, while prefill is expected to be GEMM-like and throughput oriented.

### All-GPU Tier Considerations

Example: `cuda_hot / rocm_warm`, no CPU fallback.

- All routed experts must be covered by accelerator tiers.
- Tier capacity is a hard coverage constraint, not a cache hint.
- Hottest decode experts should prefer the fastest and nearest continuation tier.
- Warm GPU tiers should own full experts, not shards, unless true `TensorParallelExperts` is explicitly introduced later.
- Cross-vendor sparse transport cost must be modeled. CUDA-to-ROCm host-staged payloads may dominate decode if too many hot experts are placed on the remote tier.
- Rebalancing should have hysteresis so experts do not churn between GPUs every few decode steps.
- Benchmark expectations should compare static-by-id placement against histogram placement.

### Mixed GPU/CPU Tier Considerations

Example: `cuda_hot / rocm_warm / cpu_cold`.

- CPU is the complete capacity fallback and correctness anchor.
- GPU tiers are caches/owners for high-benefit routed experts.
- The rebalancer should minimize CPU fallback rows, especially in decode.
- Promotion requires sustained heat and sufficient expected benefit over CPU compute plus transport.
- Demotion requires sustained coldness or memory pressure.
- CPU resident weights should remain eager and complete enough to satisfy any expert not covered by GPU tiers.
- Metrics must distinguish `cpu_fallback_rows`, `gpu_cached_rows`, and `gpu_coverage_ratio`.

### Rebalancing Safety Rules

- Rebalancing may update future owner maps only at safe step boundaries.
- A generation/step/layer collective key must never observe owner-map changes mid-collective.
- Existing in-flight sparse payloads must complete or abort before owner slots are reused.
- A cached copy is not an execution owner unless the planner selects it as the canonical owner for that step.
- The first production implementation should use deterministic histogram snapshots, not asynchronous live migration during a decode step.

## Required Profiling Metrics

When `LLAMINAR_PROFILING=1`, benchmark mode should emit a MoE overlay profiling section with at least:

| Metric | Why |
|--------|-----|
| `moe.prefill.rows_by_tier` | Shows routed-row distribution during GEMM-heavy prefill |
| `moe.decode.rows_by_tier` | Shows routed-row distribution during GEMV-heavy decode |
| `moe.entries_by_tier` | Tracks top-k fanout and sparse entry volume |
| `moe.unique_experts_by_tier` | Reveals expert locality and cache pressure |
| `moe.gpu_cached_rows` | Measures GPU cache usefulness |
| `moe.cpu_fallback_rows` | Explains mixed-tier decode latency |
| `moe.compact_dispatch_bytes` | Verifies sparse dispatch volume |
| `moe.compact_return_bytes` | Verifies sparse return volume |
| `moe.dense_bytes_avoided` | Confirms dense activation transfer avoidance |
| `moe.dispatch_wait_ms` | Finds sparse collective stalls |
| `moe.return_wait_ms` | Finds return/reduce stalls |
| `moe.local_expert_compute_ms` | Separates compute from transport |
| `moe.import_broadcast_ms` | Measures continuation TP materialization cost |
| `moe.owner_map_summary` | Records tier/device/residency/fallback coverage |

Stage timing should make it possible to test these expectations:

1. Most prefill time is GEMM-like expert/dense compute.
2. Most decode time is GEMV-like expert/dense compute plus sparse transport overhead.
3. `cuda_hot / rocm_warm` is faster than `cuda_hot / rocm_warm / cpu_cold` when GPU coverage is complete.
4. Inference improves as more experts are placed on GPUs, up to the point where transport or memory pressure dominates.

## Phased Implementation Plan

### Phase 10: Production Graph Lock-In

**Goal:** Make graph-native overlay lowering the production Qwen3.5 MoE path.

Tasks:

- Wire overlay plans through the production `OrchestrationRunner` / `InferenceRunnerFactory` path into `MoEGraphRoleRunner` and participant graph construction.
- Ensure Qwen3.5 MoE overlay graph lowering emits graph-native sparse stages for production configurations.
- Add graph construction diagnostics that print continuation domain, continuation root, routed tiers, owner participants, and fallback coverage.
- Keep legacy overlay runtime behind an explicit legacy-only environment flag for tests only.

Acceptance:

- A production-style Qwen3.5 MoE overlay config lowers without `MoEOverlayDomainRuntimeStage`.
- The role runner does not move tensor payloads directly.
- Sparse stages receive keyed workspaces and owner-map-derived participant ids.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_Qwen35MoEGraphNativeProductionLowering|Integration_MoEGraphNative_ProductionConfigSmoke)" \
  --output-on-failure --parallel
```

### Phase 11: Legacy Runtime Quarantine And Deletion Prep

**Goal:** Stop building or registering dead shadow-domain execution paths in production.

Tasks:

- Inventory all production references to old overlay runtime files.
- Mark legacy-only tests and remove them from production-lowering coverage.
- Add a forbidden dependency scan for production sources and graph-native tests.
- Split reusable math/transport helpers from runtime/scheduler code where needed.

Acceptance:

- Production code cannot instantiate the old domain runtime, LocalTP executor, or CPU sidecar runner.
- Legacy files, if still present, are not linked into production graph construction paths.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoEGraphNative_ForbiddenDependencyScan|Unit_MoEOverlayLegacyRuntimeQuarantined)" \
  --output-on-failure --parallel
```

### Phase 12: Production Expert Weight Residency

**Goal:** Connect graph-native local expert stages to production prepared expert weights and residency metadata.

Tasks:

- Replace toy/CPU-only local expert wiring in production graph lowering with prepared routed expert weights from `PreparedWeightStore` or the expert GEMM registry.
- Make owner-map participant ids resolve to the current participant's resident full experts.
- Ensure no peer prepared participant vectors are exposed to graph-native local expert params.
- Keep CPU expert weights eager/resident for CPU fallback tiers.

Acceptance:

- Each participant graph sees only its own expert weights.
- GPU participants launch only local-device expert kernels for owned expert ids.
- CPU participants run as normal graph participants, not sidecars.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoELocalExpertStage_PreparedWeights|Integration_MoEGraphNative_PreparedExpertWeights_MVP)" \
  --output-on-failure --parallel
```

### Phase 13: Hot Expert Rebalancer Planner

**Goal:** Add deterministic expert rebalancing and hot-cache placement over arbitrary routed tiers.

Tasks:

- Extend `MoEExpertParallelPlanner` with a rebalancing policy input using decode/prefill histograms and per-tier capacities.
- Add all-GPU coverage validation: no fallback is valid only when all experts are covered.
- Add mixed GPU/CPU policy: CPU fallback owns uncovered experts; GPU tiers own high-benefit experts.
- Add hysteresis knobs for promotion/demotion thresholds.
- Add diagnostics for per-layer placements, expected GPU hit rate, expected CPU fallback rows, and estimated memory use.

Acceptance:

- All-GPU plans produce complete owner maps without CPU fallback.
- Mixed plans reduce CPU fallback assignment as GPU budgets increase.
- Rebalancing does not assign duplicate execution owners.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoERoutedTierRebalancer_(AllGPU|MixedGpuCpu)|Integration_MoEGraphNative_RebalancedOwnerMap_MVP)" \
  --output-on-failure --parallel
```

### Phase 14: Graph-Native MoE Profiling Metrics

**Goal:** Emit enough profiling data to explain overlay performance in benchmark mode.

Tasks:

- Add graph-native MoE metrics collection around sparse dispatch, local expert compute, return/reduce, and continuation import.
- Integrate with `LLAMINAR_PROFILING=1` and benchmark output.
- Use `libfort` for any tabular profiling summaries.
- Include per-phase prefill/decode splits.
- Export machine-readable CSV or JSON metrics for benchmark sweeps.

Acceptance:

- Benchmark output reports row counts, entry counts, bytes moved, dense bytes avoided, tier hit rates, wait times, and compute times.
- Metrics are emitted only when profiling is enabled or explicit diagnostics request them.
- Hot-path overhead is negligible when profiling is disabled.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoEGraphNativeProfilingMetrics|Integration_MoEGraphNative_ProfilingMetrics_MVP)" \
  --output-on-failure --parallel
```

### Phase 15: Real Qwen3.5 35B Parity - `rocm_hot / cpu_cold`

**Goal:** Prove the production graph-native overlay path matches PyTorch for a heterogeneous accelerator plus CPU fallback plan.

Tasks:

- Add a config-driven parity test under `tests/v2/integration/parity/qwen35moe/`.
- Use real Qwen3.5 35B MoE GGUF weights and the existing PyTorch snapshot generator.
- Run through production `OrchestrationConfig`, not a hand-built toy stage harness.
- Assert production graph contains no legacy overlay runtime stages.
- Compare prefill and decode logits/stages against PyTorch snapshots.

Acceptance:

- `rocm_hot / cpu_cold` parity passes on the project hardware lane.
- CPU fallback rows are visible in MoE profiling/parity logs.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_GraphNative_RocmHotCpuCold" \
  --output-on-failure --parallel
```

### Phase 16: Real Qwen3.5 35B Parity - `cuda_hot / rocm_warm`

**Goal:** Prove all-GPU graph-native overlay parity without CPU fallback.

Tasks:

- Add a production-path parity test for a CUDA continuation/hot tier and ROCm warm tier.
- Validate that no routed tier has `fallback=true` and all experts are accelerator-owned.
- Assert compact sparse dispatch row counts are not multiplied by continuation TP degree.
- Compare against the same PyTorch reference snapshots.

Acceptance:

- `cuda_hot / rocm_warm` parity passes on the project hardware lane.
- No CPU expert participant or sidecar appears in the production graph.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_GraphNative_CudaHotRocmWarm" \
  --output-on-failure --parallel
```

### Phase 17: Real Qwen3.5 35B Parity - `cuda_hot / rocm_warm / cpu_cold`

**Goal:** Prove mixed accelerator plus CPU fallback parity through the same production graph-native path.

Tasks:

- Add a parity test with CUDA hot tier, ROCm warm tier, and CPU cold fallback.
- Validate owner map contains CUDA, ROCm, and CPU owners.
- Validate only the configured cold tier is fallback.
- Compare PyTorch prefill and decode snapshots.
- Export routing/fallback metrics for later benchmark correlation.

Acceptance:

- Mixed-tier parity passes on the project hardware lane.
- CPU fallback contribution is mathematically indistinguishable from the single-domain PyTorch reference within established thresholds.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold" \
  --output-on-failure --parallel
```

### Phase 18: Delete Legacy Shadow Runtime

**Goal:** Remove dead code from the failed overlay system once production parity is established.

Tasks:

- Delete production registration and source inclusion for old runtime stages/executors.
- Delete or archive tests that only validate the old shadow runtime.
- Remove obsolete config compatibility paths that imply routed `TensorParallelExperts` by default.
- Keep reusable code only if it is independent of side schedulers and peer-device stage ownership.

Expected deletion candidates:

- `MoEOverlayDomainRuntimeStage.*`
- `IOverlayDomainRuntime.*`
- `MoEOverlayDomainRuntime.*`
- `MoEExpertOverlayLocalTPStage.*`
- `MoEExpertOverlayLocalTPExecutor.*`
- `MoEOverlayCPUFallbackParticipantRunner.*`
- endpoint receive-pump production dependencies
- old bridge tests that cannot be rewritten to graph-native stages

Acceptance:

- Searching production source for old runtime symbols finds no active references.
- Graph-native production and parity gates remain green.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoEGraphNative_ForbiddenDependencyScan|Integration_Parity_Qwen35MoE_GraphNative_)" \
  --output-on-failure --parallel
```

### Phase 19: Release Benchmark Harness

**Goal:** Create repeatable Release benchmark sweeps for overlay tier configurations.

Tasks:

- Add benchmark YAML configs for:
  - `cuda_hot / rocm_warm`
  - `cuda_hot / rocm_warm / cpu_cold`
  - all-GPU increasing expert capacity
  - mixed GPU/CPU increasing GPU hot-cache budget
  - static-by-id placement
  - histogram-rebalanced placement
- Add scripts or documented commands for Release benchmark runs.
- Save profiler outputs with config name, git hash, hardware inventory, and model path.

Build and run pattern:

```bash
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel

LLAMINAR_PROFILING=1 \
./build_v2_release/llaminar2 benchmark \
  --config configs/moe_overlay/<config>.yaml \
  -m models/<qwen35-moe-35b>.gguf
```

Acceptance:

- Benchmark mode completes for each required config on the project hardware lane.
- Profiling output includes MoE overlay metrics and kernel/stage timing summaries.

Benchmark gate:

```bash
ctest --test-dir build_v2_release \
  -R "V2_Perf_MoEGraphNativeOverlay" \
  --verbose
```

### Phase 20: Benchmark Analysis And Tuning

**Goal:** Tune placement, transport, and kernels using measured Release data.

Tasks:

- Compare all-GPU versus mixed GPU/CPU tier latency and throughput.
- Sweep GPU expert budgets and plot throughput against GPU coverage.
- Separate prefill and decode timing.
- Verify prefill is GEMM-like dominated and decode is GEMV-like dominated.
- Identify sparse transport bottlenecks before implementing GPU-native pack/unpack.
- Tune expert placement hysteresis and rebalancing thresholds.
- Tune CUDA and ROCm expert kernels independently where profiling points to compute bottlenecks.

Acceptance:

- `cuda_hot / rocm_warm` is faster than `cuda_hot / rocm_warm / cpu_cold` when all experts fit on GPUs.
- Inference improves as more experts are allocated to GPUs until transport or memory pressure dominates.
- Any deviation has profiler evidence explaining why.

Output:

- Benchmark report under `benchmark_results/` or `docs/v2/perf/` with hardware, configs, profiler tables, and recommendations.

### Phase 21: Production Hardening

**Goal:** Make the feature safe enough for normal inference modes.

Tasks:

- Add CLI/config docs for graph-native MoE overlay tiers.
- Add dry-run/explain-placement output for continuation domain, routed tiers, owner map, fallback coverage, and estimated memory.
- Add failure diagnostics for missing hardware, incomplete coverage, unsupported tensor-parallel experts, and insufficient GPU capacity.
- Add transfer tracing guidance specific to MoE overlay.
- Add abort propagation tests for sparse collectives and participant graph failures.

Acceptance:

- Users can understand why a tier plan was accepted or rejected.
- Production failures fail loud before inference starts when coverage or hardware is invalid.
- One-token decode avoids repeated full dense activation transfers outside explicit debug dumps.

Test gate:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_(Unit_MoEGraphNativeProductionHardening|Integration_MoEGraphNativeAbortPropagation)" \
  --output-on-failure --parallel
```

## Parity Test Requirements

All real-model parity tests must:

- use real Qwen3.5 35B MoE GGUF weights;
- regenerate or consume PyTorch snapshots from the same GGUF;
- exercise the production runner/config path;
- verify graph-native stage lowering;
- emit parity CSVs for prefill and decode;
- compare final logits and relevant stage snapshots;
- include routing agreement diagnostics for MoE layers;
- be allowed to skip locally when hardware/model files are absent;
- be mandatory and non-skipped on the project hardware parity lane.

Required parity cases:

| CTest | Tier Shape | Purpose |
|-------|------------|---------|
| `V2_Integration_Parity_Qwen35MoE_GraphNative_RocmHotCpuCold` | ROCm hot + CPU cold | Heterogeneous accelerator plus CPU fallback correctness |
| `V2_Integration_Parity_Qwen35MoE_GraphNative_CudaHotRocmWarm` | CUDA hot + ROCm warm | All-GPU no-fallback correctness |
| `V2_Integration_Parity_Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold` | CUDA hot + ROCm warm + CPU cold | Mixed-tier fallback correctness |

## Benchmark Matrix

Minimum benchmark matrix after parity:

| Axis | Values |
|------|--------|
| Build | `Release` only |
| Profiling | `LLAMINAR_PROFILING=1` |
| Prompt shape | short-prefill/long-decode, long-prefill/short-decode, balanced |
| Tier shape | `cuda_hot/rocm_warm`, `cuda_hot/rocm_warm/cpu_cold` |
| Placement policy | static-by-id, histogram, adaptive rebalance |
| GPU expert budget | low, medium, high, all-fit |
| CPU fallback | absent, present |

Expected measurements:

1. Prefill should show most time in GEMM-like kernels.
2. Decode should show most time in GEMV-like kernels and sparse transport/import overhead.
3. All-GPU `cuda_hot / rocm_warm` should beat mixed `cuda_hot / rocm_warm / cpu_cold` when all experts fit on GPUs.
4. Increasing GPU expert residency should improve inference until transport or memory pressure dominates.
5. CPU fallback rows should correlate with mixed-tier decode latency.

## Definition Of Done

This follow-up plan is complete when:

1. Qwen3.5 MoE overlay production graphs use graph-native sparse stages only.
2. Old shadow LocalTP domain runtime code is removed from production builds.
3. Expert owner maps are produced by flexible tier planning and optional rebalancing.
4. All-GPU and mixed GPU/CPU plans both support real model parity.
5. The three required Qwen3.5 35B MoE PyTorch parity tests pass on the hardware lane.
6. `LLAMINAR_PROFILING=1` emits actionable MoE overlay metrics.
7. Release benchmarks validate or explain the expected performance ordering.
8. Increasing GPU expert allocation improves measured throughput until a documented bottleneck dominates.
9. Users can inspect placement, fallback coverage, and profiling output without reading source code.

## First Implementation Step

Begin with **Phase 10: Production Graph Lock-In**. The first subagent should gather current Qwen3.5 MoE graph lowering, runner factory, and legacy runtime registration context, then implement a production-lowering smoke gate that proves a graph-native overlay config cannot lower to the old shadow runtime.