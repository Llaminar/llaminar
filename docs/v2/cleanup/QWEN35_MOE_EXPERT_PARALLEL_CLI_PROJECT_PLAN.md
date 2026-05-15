# Qwen3.5 MoE Expert Parallel CLI Project Plan

**Date**: 2026-05-14  
**Status**: Proposed  
**Scope**: Make Qwen3.5 MoE routed expert execution explicitly selectable from the CLI, defaulting to expert-id parallelism across TP participants, with bounded hot-expert replication.  
**First implementation target**: Correct, memory-bounded expert parallelism for CPU NodeLocalTP / GlobalTP. Tensor-parallel experts are exposed as a CLI option but fail fast with `Not Implemented` until the execution path is production-ready.

---

## 1. Executive Summary

Qwen3.5 MoE currently has the right low-level ingredients for expert-parallel execution, but the default graph does not assemble them into the desired runtime contract.

The current observed CPU TP behavior is:

1. `-d cpu` starts two MPI ranks and creates a TP context.
2. Dense/shared weights use existing TP machinery where supported.
3. Routed MoE expert weights are still treated as replicated by the Qwen3.5 MoE schema and graph.
4. Each rank prepares all routed expert GEMM engines.
5. The first request can retain another large set of prepared expert allocations through cached decode graph state.

That behavior explains the measured memory blow-up: the model file is around 21 GB, but each rank reached roughly 47 GB RSS at server ready and roughly 63 GB RSS after the first short request.

The target behavior is:

- **Default MoE mode**: expert parallelism by expert id.
- **Startup in EP mode**: each TP participant owns roughly `num_experts / tp_domain_num_participants` routed experts.
- **Runtime in EP mode**: routed expert outputs are partial and reduced through the existing MoE output allreduce.
- **Hot expert cache**: a bounded number of hottest remote experts may be replicated on each rank/device, defaulting to 10% of routed expert count.
- **Tensor-parallel experts mode**: selectable from CLI/YAML, but currently fails with a clear `Not Implemented` error.

---

## 2. User-Facing Contract

### 2.1 New CLI Options

Add first-class MoE expert execution options:

```bash
--moe-expert-mode <mode>
```

Valid values:

| Value | Meaning | Initial behavior |
|---|---|---|
| `expert-parallel` | Split routed expert ids across TP participants. | Default and implemented first. |
| `tensor-parallel` | Shard every selected expert's gate/up/down GEMMs across TP participants. | Parse and validate, then fail with `Not Implemented`. |
| `replicated` | Keep current full routed expert replication behavior. | Optional compatibility/debug mode. If retained, it should be explicit, not the MoE TP default. |

Add bounded hot-cache controls:

```bash
--moe-hot-expert-cache <count|percent|off>
```

Examples:

```bash
--moe-hot-expert-cache 10%
--moe-hot-expert-cache 24
--moe-hot-expert-cache off
```

Default:

```text
10% of routed expert count per rank/device, rounded down with a minimum of 1 when dynamic MoE rebalancing is enabled.
```

Add CLI equivalents for the existing environment-driven rebalance controls:

```bash
--moe-rebalance <off|observe|dynamic>
--moe-rebalance-window <tokens>
--moe-rebalance-max-window <tokens>
--moe-rebalance-window-growth <factor>
--moe-release-raw-expert-weights
```

Existing environment variables can remain as overrides or debug hooks, but the production contract should flow through `OrchestrationConfig`.

### 2.2 YAML Shape

Add matching YAML support:

```yaml
moe:
  expert_mode: expert-parallel
  rebalance: dynamic
  hot_expert_cache: 10%
  rebalance_window: 256
  rebalance_max_window: 4096
  rebalance_window_growth: 1.5
  release_raw_expert_weights: false
```

### 2.3 Not-Implemented Behavior

If a user selects tensor-parallel experts before the production path is ready:

```bash
./llaminar2 oneshot -d cpu --moe-expert-mode tensor-parallel -m model.gguf -p test
```

the runtime should fail during configuration/plan validation with a message like:

```text
MoE expert mode 'tensor-parallel' is recognized but not implemented for the standard Qwen3.5 MoE execution path yet. Use --moe-expert-mode expert-parallel.
```

This failure must happen before model weight materialization.

---

## 3. Current Implementation Inventory

### 3.1 Pieces Already Present

| Capability | Current location | Status |
|---|---|---|
| Expert-id filtering in MoE execution | `MoEExpertComputeStage::Params::local_expert_start/local_expert_count` | Present |
| Expert masks for dynamic ownership/replicas | `MoEExpertComputeStage::Params::expert_mask` | Present |
| MoE output allreduce when output is partial | `Qwen35MoEGraph.cpp` | Present |
| 3D expert tensor expert-slice loading | `WeightManager.cpp` `ShardingMode::EXPERT_PARALLEL` | Present |
| Hot expert histogram controller | `MoERebalanceController` / `DecodeExpertHistogram` | Present |
| Replica set representation | `ExpertReplicaSet` | Present |
| Apply expert masks to graph stages | `DeviceGraphOrchestrator::applyExpertMasks` | Present |
| Prepared expert slab ownership | `PreparedWeightStore` expert slab API | Present, recently expanded |
| Tensor-parallel-experts vocabulary | `ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS` | Present |

### 3.2 Gaps Blocking the Target Contract

| Gap | Impact |
|---|---|
| Qwen3.5 MoE schema marks routed expert tensors as replicated. | Loader does not default to expert slices. |
| Dynamic rebalance path forces full expert tensors when `LLAMINAR_MOE_REBALANCE=dynamic`. | EP memory savings are defeated. |
| Qwen35 MoE graph does not set default local expert range for CPU TP. | Every rank computes/prepares every routed expert. |
| Normal server/generation path does not propose/apply hot replicas the same way benchmark mode does. | Hot cache is not a bounded production feature. |
| Hot-cache default is currently `0 => 2 * top_k`, not 10% of routed expert count. | Cache capacity is not aligned with desired memory budget. |
| Decode graph cache can retain duplicate prepared expert engines. | First request can grow RSS even after startup. |
| Tensor-parallel-experts exists mainly through overlay/fallback code. | Not ready as a simple standard Qwen35 MoE mode. |

---

## 4. Target Architecture

### 4.1 Expert Parallel Mode

Expert parallelism means each TP participant owns whole experts by expert id.

For `num_experts = 256` and two CPU ranks:

```text
rank 0: experts [0, 128)
rank 1: experts [128, 256)
```

For uneven splits, use balanced contiguous ranges:

```cpp
base = num_experts / participants;
rem = num_experts % participants;
count(rank) = base + (rank < rem ? 1 : 0);
start(rank) = rank * base + min(rank, rem);
```

Execution contract:

1. Router computes top-k expert ids on all ranks.
2. Each rank only computes selected experts it owns or has as hot replicas.
3. Each rank writes a partial `[seq_len, d_model]` MoE output.
4. Existing TP allreduce sums partial outputs.
5. Shared expert path remains dense TP where applicable.

### 4.2 Tensor Parallel Experts Mode

Tensor-parallel experts means every selected expert is internally sharded across all TP participants.

Expected future contract:

1. Every rank owns a shard of every expert.
2. Expert gate/up weights are column-parallel over expert intermediate dimension.
3. Expert down weights are input/row-parallel over expert intermediate dimension.
4. All ranks participate for every selected expert.
5. Per-expert partials are reduced before weighted combine or are accumulated into a reduced MoE output.

Initial project behavior:

- Add CLI/config enum.
- Validate the choice.
- Reject with `Not Implemented` before model load for the standard Qwen3.5 MoE path.
- Do not silently fall back to replicated experts.

### 4.3 Hot Expert Cache

In EP mode, each rank may keep a bounded cache of remote hot experts.

Definitions:

- **Owned expert**: belongs to the rank's static EP range or current rebalance placement.
- **Replica expert**: not owned by the rank, but prepared locally because it is hot.
- **Hot-cache cap**: maximum number of replica experts per rank/device.

Initial default:

```text
max_replicas_per_rank = floor(num_experts * 0.10)
```

For Qwen3.5 with 256 routed experts, this is 25 remote experts per rank/device by default.

Runtime rules:

1. Histogram records routed expert usage during decode.
2. At rebalance boundaries, controller selects the hottest remote experts for each rank up to the cap.
3. New replicas are prepared or transferred through `MoEExpertWeightService` / `PreparedWeightStore`.
4. Cold replicas beyond the cap are evicted through `releaseDepartedExperts`.
5. `ExpertReplicaSet` is applied to stages so per-token dispatch remains deterministic.
6. The number of resident experts per rank is bounded by `owned_count + hot_cache_cap`.

---

## 5. Phase Plan

### Phase 0: Config Contract and Fail-Fast Modes

Goal: Add user-facing configuration without changing execution yet.

Tasks:

1. Add `MoEExpertMode` enum:

   ```cpp
   enum class MoEExpertMode {
       ExpertParallel,
       TensorParallel,
       Replicated,
   };
   ```

2. Add fields to `OrchestrationConfig` and downstream runtime config:

   ```cpp
   MoEExpertMode moe_expert_mode = MoEExpertMode::ExpertParallel;
   MoEHotExpertCacheConfig moe_hot_expert_cache;
   MoERebalanceRuntimeConfig moe_rebalance;
   ```

3. Add parser support for CLI and YAML.
4. Add validation:
   - tensor-parallel mode is recognized but rejected as not implemented.
   - hot cache supports count, percent, and off.
   - hot cache percent must be in `[0, 100]`.
5. Add dry-run/explain-placement output showing:
   - selected MoE expert mode,
   - static expert range per rank,
   - hot-cache cap per rank.

Exit criteria:

- CLI tests prove defaults and parsing.
- Selecting tensor-parallel mode fails before model load.
- Existing env variables still work or are explicitly documented as lower-priority debug overrides.

### Phase 1: Static Expert Parallel Startup

Goal: Make EP the default resident/execution mode for Qwen3.5 MoE under TP.

Tasks:

1. Change Qwen3.5 MoE routed expert weight sharding to EP when `moe_expert_mode == ExpertParallel`.
2. Preserve explicit `replicated` mode for compatibility/debug if desired.
3. Wire the static expert range into `MoEExpertComputeStage::Params` during graph build.
4. Ensure `MoEExpertWeightService::extractExpertViews` handles pre-sliced 3D expert tensors correctly.
5. Ensure `prepareGemmEngines` prepares only local experts.
6. Keep MoE output allreduce active whenever output is partial.
7. Remove or gate the current dynamic-rebalance full-expert-tensor shortcut so it does not override EP.

Exit criteria:

- With two CPU ranks and 256 experts, each rank prepares roughly 128 experts per MoE layer.
- Server-ready RSS drops substantially relative to the replicated baseline.
- Prefill/decode numerics match replicated mode within existing parity thresholds.
- `--explain-placement` reports the expert range per rank.

### Phase 2: Prepared Weight Reuse Across Cached Graphs

Goal: Prevent prefill/decode graph cache entries from owning duplicate packed expert engines.

Tasks:

1. Ensure the `PreparedWeightStore` is attached before the first graph build that prepares MoE experts.
2. Make startup MoE expert engines register as expert slabs.
3. On subsequent cached graph construction, resolve engines from expert slabs instead of repacking.
4. Add diagnostics for expert slab hits/misses by layer and role.
5. Add a guardrail warning or assertion if a cached graph tries to prepare an already-prepared expert slab again.

Exit criteria:

- First request does not materially increase resident prepared expert memory.
- `PreparedWeightStore::expertSlabCount()` and populated expert counts remain stable across prefill/decode graph cache creation.
- `malloc_info` / smaps no longer show a second wave of large live expert allocations after the first request.

### Phase 3: Bounded Hot Expert Cache in Server/Generation Path

Goal: Make hot expert replicas a bounded production feature, not just benchmark plumbing.

Tasks:

1. Move the benchmark-mode replica workflow into shared orchestration logic:
   - histogram summary,
   - `proposeReplicas(max_replicas_per_rank)`,
   - transfer/preparation of replicas,
   - expert mask application,
   - `setExpertReplicaSet`,
   - optional raw weight release.
2. Change the default cap from `2 * top_k` to `10%` of routed experts.
3. Enforce cap on every rebalance cycle.
4. Evict departed replicas through `releaseDepartedExperts`.
5. Keep owned experts resident and non-evictable.
6. Add metrics:
   - owned experts per rank,
   - replica experts per rank,
   - cache hits/misses,
   - evictions,
   - bytes or estimated bytes resident.

Exit criteria:

- Memory growth from dynamic hot expert caching is bounded by the configured cap.
- Repeated requests do not grow unbounded RSS.
- Hot replica dispatch is deterministic and preserves numerics.
- Cache cap can be disabled with `--moe-hot-expert-cache off`.

### Phase 4: Tensor-Parallel Experts Design Hardening

Goal: Keep tensor-parallel mode visible but blocked until it is a product-quality execution mode.

Tasks:

1. Document the future tensor-parallel expert contract.
2. Decide whether to lower it through the existing overlay runtime or a standard graph path.
3. Audit CPU fallback host-dequant bridge performance and memory behavior.
4. Define prepared-weight representation for 3D expert tensor row/column shards.
5. Add parity and performance gates before enabling.

Exit criteria:

- Tensor-parallel mode remains fail-fast in production CLI.
- A separate implementation plan exists before removing the `Not Implemented` guard.

---

## 6. Implementation Touchpoints

### Configuration

- `src/v2/config/OrchestrationConfig.h`
- `src/v2/config/OrchestrationConfigParser.cpp`
- `src/v2/config/OrchestrationConfig.cpp`
- `src/v2/config/ConfigValidator.cpp`
- `src/v2/execution/config/RuntimeConfig.h`
- `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`

### Graph and Model Wiring

- `src/v2/models/GraphTypes.h`
- `src/v2/models/qwen35moe/Qwen35MoESchema.h`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/models/qwen35moe/Qwen35MoEGraphConfigBuilder.cpp`
- `src/v2/models/qwen/QwenGraphBase.cpp`

### Weight Loading and Prepared State

- `src/v2/loaders/WeightManager.cpp`
- `src/v2/loaders/PreparedWeightStore.h`
- `src/v2/loaders/PreparedWeightStore.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/kernels/KernelFactory.cpp`

### Dynamic Rebalance and Cache

- `src/v2/execution/moe/MoERebalanceController.h`
- `src/v2/execution/moe/MoERebalanceController.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/app/modes/BenchmarkMode.cpp`

---

## 7. Testing Strategy

### Unit Tests

Add or extend tests for:

1. CLI parsing:
   - default mode is `expert-parallel`,
   - `tensor-parallel` parses but validation/runtime rejects,
   - `--moe-hot-expert-cache 10%`, `24`, and `off` parse correctly.
2. Expert range calculation:
   - even splits,
   - uneven splits,
   - invalid participant counts.
3. Qwen35 MoE schema:
   - routed expert weights are EP in EP mode,
   - routed expert weights are not silently replicated in EP mode.
4. MoE graph construction:
   - `local_expert_start/count` are set in EP mode,
   - MoE output allreduce is inserted for partial routed expert output,
   - tensor-parallel mode throws before graph materialization.
5. Rebalance cache cap:
   - `10%` resolves to expected count,
   - replicas never exceed cap,
   - departed replicas are released.

### Integration Tests

Focused tests:

```bash
ctest --test-dir build_v2_integration \
  -R "Qwen35.*MoE.*CPU.*TP|MoE.*ExpertParallel" \
  --output-on-failure --parallel
```

Add a model-light MPI test that asserts:

- rank 0 and rank 1 get disjoint expert ranges,
- each rank prepares only its range,
- allreduce restores the replicated-mode output.

Add a server memory smoke test, if feasible:

1. Start Qwen3.5 MoE 35B with `-d cpu`.
2. Capture per-rank RSS/PSS at server ready.
3. Send one short request.
4. Capture post-request RSS/PSS.
5. Assert post-request growth is bounded by a configurable threshold.

### Manual Verification Commands

EP default:

```bash
LLAMINAR_LOG_LEVEL=INFO \
./build_v2_integration/llaminar2 serve \
  -d cpu \
  --moe-expert-mode expert-parallel \
  --moe-hot-expert-cache 10% \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Tensor-parallel fail-fast:

```bash
./build_v2_integration/llaminar2 oneshot \
  -d cpu \
  --moe-expert-mode tensor-parallel \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p test
```

Expected result:

```text
Not Implemented: MoE tensor-parallel expert mode is not implemented for the standard Qwen3.5 MoE path.
```

Memory inspection:

```bash
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2|mpirun'
sudo gdb -q -batch -p <rank-pid> -ex 'call malloc_info(0, stdout)'
awk '/^(Rss|Pss|Private_Dirty|Anonymous|RssFile):/ {print}' /proc/<rank-pid>/smaps_rollup
```

---

## 8. Acceptance Criteria

### Functional

- `--moe-expert-mode expert-parallel` is the default for Qwen3.5 MoE under TP.
- `--moe-expert-mode tensor-parallel` fails clearly before model load.
- `--moe-hot-expert-cache` is configurable from CLI and YAML.
- Dynamic hot replicas are bounded by the configured cap.
- Replicated mode is explicit if kept.

### Correctness

- EP output matches replicated routed expert output after MoE output allreduce.
- Hot replica dispatch does not change generated tokens under deterministic sampling.
- Rebalance boundaries do not race with active forward execution.

### Memory

- Server-ready resident memory reflects expert slicing rather than full expert replication on every rank.
- First request does not retain a second full set of prepared expert engines.
- Repeated requests do not grow without bound.
- Resident expert count per rank is bounded by:

```text
owned_experts + configured_hot_expert_cache
```

### Observability

- Logs report selected MoE expert mode.
- Logs report per-rank expert ranges.
- Logs report hot-cache cap and current replica count.
- `--dry-run` / `--explain-placement` prints MoE expert placement decisions.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Expert-sliced tensors break dynamic rebalance because raw remote experts are unavailable. | Replicas must come from prepared transfer or explicit retained source policy; do not force full raw tensor replication by default. |
| Cached decode graphs repack experts independently. | Make `PreparedWeightStore` expert slabs the only long-lived owner and require slab reuse. |
| Tensor-parallel mode accidentally falls back to replicated mode. | Fail fast until implemented; add tests for the error path. |
| Hot-cache cap is counted by experts but actual bytes differ. | Use count cap first for CPU; later migrate to byte-budgeted residency using `MOE_EXPERT_RESIDENCY_BUDGET_PLAN.md`. |
| Dynamic rebalance moves too many experts and spikes latency. | Enforce cap, apply at decode boundaries, and prewarm arrivals before switching masks. |
| EP changes numerical behavior due to missing reduction. | Keep MoE output allreduce mandatory whenever output is partial; add parity tests. |

---

## 10. Immediate Next Steps

1. Add `MoEExpertMode` and hot-cache config to `OrchestrationConfig` and parser.
2. Add validation that tensor-parallel mode is recognized but not implemented.
3. Change Qwen3.5 MoE graph/schema to honor EP mode and set per-rank expert ranges.
4. Gate/remove the dynamic full-expert-tensor loading shortcut in EP mode.
5. Verify startup memory and parity on Qwen3.5 MoE CPU TP.
6. Move benchmark-mode hot-replica workflow into normal generation/server orchestration.
7. Fix prepared expert slab reuse across cached graph signatures.
