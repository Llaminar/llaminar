# Handover: MoE Overlay Phase 10A benchmark runnability

Date: 2026-05-13  
Branch: `feat/qwen35-moe`  
Primary goal: get Qwen3.5 MoE overlay far enough through initialization and `llaminar2 benchmark` that we can start measuring real prefill/decode behavior, then move on to Phase 11.

## Executive summary

Phase 10A is no longer blocked at MPI/bootstrap, root/non-root graph selection, or CPU fallback participant construction. The current reduced-context benchmark reaches warmup prefill with:

- rank 0 running the root continuation graph on the existing local ROCm tensor-parallel model (`rocm_hot`, ROCm:0 + ROCm:1), and
- rank 1 initialized as a native graph-mode CPU fallback participant endpoint.

The benchmark still fails: both TP workers enter the first forward pass and launch early GPU work, then `RankOrchestrator::forwardTP` times out waiting for device completion. This looks less like a missing CPU fallback endpoint now and more like a deeper local TP / RCCL / stream synchronization issue exposed by the overlay benchmark path.

The next agent should treat the current hang as an architectural investigation, not just a missing `return true` somewhere.

## User direction that shaped the implementation

The user explicitly did **not** want one `DeviceGraphOrchestrator` per CPU device for fallback. The CPU fallback path should respect the already-proven `GlobalTPContext` model and work natively without inventing a coordinator/worker protocol for benchmark mode.

That direction is important because `benchmark` already drives `runner_->forward(...)` on all MPI ranks and handles its own token broadcasts. Routing non-root ranks into the Completion/SingleShot MPI worker loop would conflict with benchmark semantics.

## What was implemented

### 1. Graph-native CPU fallback participant runner

New files:

- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.h`
- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp`

The new runner implements `IInferenceRunner` and lets a non-root CPU fallback rank participate as a benchmark-native endpoint. It uses the existing CPU fallback execution machinery rather than a new worker protocol:

- `MoEExpertOverlayCPUFallbackStage`
- `MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(...)`
- `GlobalTPContext`

The runner currently allocates placeholder FP32 hidden/routing/output buffers, loads CPU fallback expert weights per layer, and iterates MoE layers / fallback tiers when its `forward()` is called.

### 2. Overlay-aware orchestration plan routing

Main file:

- `src/v2/execution/runner/OrchestrationRunner.cpp`

Important additions/changes:

- `requiresOverlayMPIWorld()` causes MoE overlay benchmark runs with node-local/world participation to launch under MPI.
- `resolveOverlayExecutionPlanForRunner()` resolves the plan early enough for rank role decisions.
- `applyOverlayRankRoleToExecutionPlan()` now does two key things:
  - non-root `CpuFallbackParticipant` ranks are narrowed to endpoint plans, not root graph plans;
  - the root rank is rebound to the continuation/base model domain, e.g. `rocm_hot`, before local TP setup.
- `buildComputeGraph()` routes non-root `CpuFallbackParticipant` plans to `MoEOverlayCPUFallbackParticipantRunner`.
- memory validation now accounts for local TP shards/devices, avoiding charging the full model to one primary GPU.
- nested runner configs carry the overlay MPI context.

This is the change that moved the benchmark past the earlier error where rank 1 tried to build a root ROCm graph it could not own.

### 3. CPU fallback runtime/dispatch support

Main files:

- `src/v2/execution/moe/MoEOverlayDomainRuntime.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.cpp`

Notable changes:

- added a native CPU fallback graph dispatch path for `GraphDispatchCollective` without relying on a generic backend;
- cached CPU fallback domain context is used for fallback execution;
- if an expert mask has zero active assignments, `MoEExpertOverlayCPUFallbackStage` clears output and returns success.

The zero-active behavior is needed by the current local TP bridge so non-executing participants can pass through fallback stages without duplicating fallback MPI work.

### 4. Qwen35MoE graph integration

Main file:

- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`

Relevant changes:

- overlay tier work is wrapped in `MoEOverlayDomainRuntimeStage`;
- dispatch group identity helpers were added;
- for CPU `NodeLocalTP` fallback, only local TP local-rank 0 currently executes the actual CPU fallback dispatch path, while nonzero local TP participants get a zero expert mask.

This local-rank-0 gating is a bridge, not the final architecture. The long-term shape should be that all continuation participants enter a graph-native dispatch collective and publish either real work or an ordered no-op/cancel record.

### 5. Graph-native overlay runtime primitives

New or heavily changed files include:

- `src/v2/execution/moe/IOverlayDomainRuntime.h`
- `src/v2/execution/moe/MoEOverlayDomainRuntime.h`
- `src/v2/execution/moe/MoEOverlayDomainRuntime.cpp`
- `src/v2/execution/moe/MoEOverlayDispatchCollective.h`
- `src/v2/execution/moe/MoEOverlayDispatchCollective.cpp`
- `src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.h`
- `src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.cpp`

These establish the graph-stage-level overlay dispatch/runtime layer used by the Qwen35MoE graph.

### 6. Config/domain/weight residency work in the dirty tree

Other relevant Phase 10A-era work present in the current diff:

- unified execution domain definition:
  - `src/v2/orchestration/ExecutionDomainDefinition.h`
  - `src/v2/orchestration/ExecutionDomainDefinition.cpp`
- overlay configs:
  - `configs/qwen35-moe-overlay-layout-a.yaml`
  - `configs/qwen35-moe-overlay-layout-b.yaml`
- `base_model_domain` on `MoEExpertParallelPlan`;
- participant-aware expert residency / registry / preparation logic in the loader path.

Be careful when reviewing the diff: there are also older or adjacent changes in the worktree, such as benchmark/Completion/SingleShot MPI test coverage and Qwen35 graph fixes. Do not revert unrelated dirty changes.

## Verification already performed

### Unit/integration gates

The following relevant gates have passed during this phase of work:

```bash
cmake --build build_v2_integration \
  --target v2_unit_moe_overlay_domain_runtime v2_unit_moe_overlay_dispatch_collective \
  --parallel

ctest --test-dir build_v2_integration \
  -R "V2_Unit_MoEOverlay(DomainRuntime|DispatchCollective)" \
  --output-on-failure --parallel
```

CPU fallback integration coverage also passed after building the missing executable target:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Integration_MoEExpertOverlay_CPUFallback_MPI" \
  --output-on-failure --parallel

cmake --build build_v2_integration \
  --target v2_integration_moe_expert_overlay_cpu_tensor_parallel_experts_mpi \
  --parallel

ctest --test-dir build_v2_integration \
  -R "V2_Integration_MoEExpertOverlay_CPUTensorParallelExperts_MPI" \
  --output-on-failure --parallel
```

Release build passed:

```bash
cmake --build build_v2_release --parallel
```

### Benchmark progress

The benchmark now gets substantially farther than before.

#### Earlier failure mode: non-root rank tried to build a root graph

Before the participant runner/root-domain rebinding changes, this failed during initialization:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=300000 \
./build_v2_release/llaminar2 benchmark \
  --config configs/qwen35-moe-overlay-layout-a.yaml \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p Hello -n 0 -t 0
```

Rank 1 attempted to build a graph for `rocm_hot` and failed because the continuation devices were not locally reachable from the CPU fallback rank. Rank 0 then hit a `pure virtual method called` abort while the MPI job unwound.

#### Current default-context failure mode: memory validation

After the role routing changes, the same command fails cleanly at default context length (`-c 4096`) because ROCm:0 is short on available memory:

```text
ROCm:0: need 25.2 GB but only 23.2 GB available (deficit: 2.0 GB)
```

This is actually progress: memory validation is now charging roughly half the model to each local TP shard instead of charging the whole model to the primary GPU.

#### Current reduced-context failure mode: TP forward hang

With a reduced context, initialization and graph build pass:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
  --config configs/qwen35-moe-overlay-layout-a.yaml \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p Hello -n 0 -t 0 -c 512
```

Important success signals:

```text
[OrchestrationRunner] Execution strategy: MoE overlay CPU fallback participant endpoint rank=1 domains=1
[MoEOverlayCPUFallbackParticipantRunner] Rank 1 initialized as graph-native CPU fallback participant d_model=2048 num_experts=256 top_k=8 layers=40
[OrchestrationRunner] Execution strategy: MULTI-DEVICE (LOCAL TP)
[OrchestrationRunner]   Device 0: localhost:0:rocm:0
[OrchestrationRunner]   Device 1: localhost:0:rocm:1
[OrchestrationRunner] Multi-device compute graph built successfully
```

Then warmup prefill starts. Both ROCm workers enter early graph work:

```text
[TPWorkerPool] Created 2 persistent worker threads
[GDNRecurrenceStage] GPU launch pointers layer=0 ... device ROCm:0 ... seq=1
[GDNRecurrenceStage] GPU launch pointers layer=0 ... device ROCm:1 ... seq=1
[RCCLCoordinator] Compute streams registered for 2 devices — using stream-level pre-sync
```

After the configured timeout:

```text
RankOrchestrator::forwardTP: Device 0 did not complete (stuck)
Forward pass failed during prefill
Warmup prefill failed
```

The command exits with benchmark failure.

## Operational caveat: failed benchmark runs can leave live MPI children

After a TP timeout, the run may leave live `mpirun` / `llaminar2` children holding ROCm memory. Before rerunning, clean them up:

```bash
ps -eo pid,ppid,stat,comm,args | grep -E 'llaminar2|mpirun|orted|prte' | grep -v grep
kill <mpirun_pid> <rank_pids> || true
```

Defunct `llaminar2` zombies may remain under PID 1; those do not hold ROCm memory, but live `Sl` / `Rl` children do.

This matters because a subsequent rerun failed memory validation with ROCm:1 reporting only 3.7 GB free until the stale job was killed.

## Architectural analysis

### Benchmark mode should remain graph-native on all ranks

`benchmark` is not Completion or SingleShot mode. It already calls `runner_->forward(...)` on each MPI rank and coordinates prompt/decode token broadcasts itself. Therefore the right shape for overlay benchmark is:

- rank 0: root continuation graph runner, using established local TP over ROCm devices;
- rank 1+: endpoint runners for overlay participant domains, also implementing `IInferenceRunner`;
- no benchmark rank should be sent into a separate coordinator/worker command loop.

The new CPU fallback participant runner follows this shape.

### CPU fallback should stay aligned with `GlobalTPContext`

The fallback domain is not supposed to be modeled as one DGO per CPU device. The current implementation correctly leans on the existing CPU TP design:

- one logical CPU fallback domain;
- `GlobalTPContext` for node-local CPU tensor parallelism;
- CPU fallback stage/domain context for expert execution.

This matches the user's explicit direction and should be preserved.

### The current hang appears to happen before meaningful CPU fallback work

The reduced-context benchmark logs show the root local TP workers starting layer 0 GPU work and stream-level RCCL setup, then timing out. There is no clear evidence yet that the CPU fallback endpoint is the thing blocking the forward pass.

The strongest current hypothesis is that the hang is in the continuation-domain local TP path, especially around RCCL allreduce or stream synchronization.

### Repeated TP/RCCL context creation is suspicious

The logs show `RCCLCoordinator` and `LocalTPContext` initialization multiple times:

- once during root orchestration setup;
- again while creating nested/local TP graph runners for each ROCm device.

Representative log pattern:

```text
[RCCLCoordinator] Initialized with 2 ROCm GPU(s)
[RCCLBackend] Initialized multi-GPU single-process (via RCCLCoordinator) with 2 GPU(s), local_rank=0
[LocalTPContext] Backend RCCL initialized for 2 devices
[InferenceRunner] LOCAL TP enabled: degree=2 device_idx=0 ...
[RCCLCoordinator] Initialized with 2 ROCm GPU(s)
[LocalTPContext] Backend RCCL initialized for 2 devices
[InferenceRunner] LOCAL TP enabled: degree=2 device_idx=1 ...
```

If each nested device graph runner constructs or owns a separate local TP context/coordinator while `RankOrchestrator` expects workers to issue matching collectives through one shared communicator group, the workers can deadlock. The next investigation should verify whether both device runners are using the pre-existing `LocalTPContext` supplied by `RankOrchestrator` or independently initialized contexts.

Continuation graph collectives and overlay dispatch collectives must remain separate:

- continuation dense graph allreduces should use the established shared local TP context/RCCL coordinator owned by the root `RankOrchestrator`;
- CPU fallback `NodeLocalTP` should use `GlobalTPContext` across CPU fallback ranks;
- overlay dispatch should coordinate logical tier work/no-op/cancel records, not replace either of the above.

### Duplicate expert preparation is another risk

The current benchmark initialization logs also show full expert loading on both ROCm devices before overlay-filtered preparation:

```text
GPU pipeline: loading 311 GEMM weights + 30720 MoE expert slots for ROCm:0
GPU pipeline: loading 311 GEMM weights + 30720 MoE expert slots for ROCm:1
```

Later, overlay-specific preparation loads only the expected accelerator tier:

```text
domain rocm_hot device=ROCm:0 ... assigned_experts=320 planned_engines=960
domain rocm_hot device=ROCm:1 ... assigned_experts=320 planned_engines=960
GPU pipeline: loading 0 GEMM weights + 960 MoE expert slots for ROCm:0
GPU pipeline: loading 0 GEMM weights + 960 MoE expert slots for ROCm:1
```

This suggests `RankOrchestrator` / `WeightManager::finalizeForDevices` may still be doing pre-overlay all-expert GPU preparation before the overlay filtered plan is applied. That is probably not the direct cause of the hang, but it bloats memory, increases initialization time, and will block default-context benchmark runs.

## Recommended next steps

### 1. Clean the environment before every benchmark rerun

Use the process check above. Do not trust ROCm free memory after a timed-out benchmark until stale MPI children are gone.

### 2. Prove or disprove local TP allreduce as the hang

After cleanup, rerun the reduced-context benchmark with allreduce skipped:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_SKIP_ALLREDUCE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
  --config configs/qwen35-moe-overlay-layout-a.yaml \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p Hello -n 0 -t 0 -c 512
```

Interpretation:

- if this completes prefill, focus on local TP allreduce/RCCL stream sync;
- if it still hangs, instrument the graph stages after GDN recurrence to find the first non-returning stage.

The first attempt to run this diagnostic did not get a clean read because a stale prior job held GPU memory and caused memory validation failure.

### 3. Instrument the shared TP context and RCCL call path

Recommended trace points:

- `RankOrchestrator::forwardTP` / TP worker task boundaries:
  - log when each worker enters and leaves `runner->forward(...)`;
  - log whether the worker thread is stuck inside the graph or only waiting on GPU completion.
- `LocalTPContext::allreduceOnStream`:
  - before/after backend call;
  - device index, tensor size, dtype, stream pointer, sequence number.
- `RCCLCoordinator::allreduceSingleDeviceOnStream`:
  - before/after `ncclAllReduce`/RCCL call;
  - communicator identity and registered stream identity.
- any stream-level pre-sync/post-sync points in `RCCLCoordinator`.

If `LLAMINAR_VALIDATE_GPU_PTRS=1` or existing RCCL pointer diagnostics are available, run once with those enabled as well.

### 4. Audit whether nested graph runners share the root TP context

Inspect the local TP branch in `InferenceRunnerFactory.cpp` and `RankOrchestrator.cpp`. The key question:

> Are both ROCm device graph runners using the same pre-existing `LocalTPContext` owned by `RankOrchestrator`, or does each one create its own `LocalTPContext` / `RCCLCoordinator`?

If separate contexts are being created, fix the factory/config plumbing so nested local TP graph runners receive and use the shared root context. The log line `RankOrchestrator: Creating with pre-existing TP context` is good, but the repeated nested `LocalTPContext` initialization logs suggest that this may not be fully true inside each DGO runner.

### 5. Compare against a non-overlay local TP benchmark

Run a baseline without MoE overlay if a fitting config/model path exists. The purpose is to determine whether the hang is overlay-specific or a pre-existing ROCm local TP/GDN allreduce issue exposed by this model.

Possible direction:

```bash
./build_v2_release/llaminar2 benchmark \
  <non-overlay local-tp config> \
  -m <model> \
  -p Hello -n 0 -t 0 -c 512
```

If non-overlay local TP also hangs at the same point, prioritize RCCL/local TP fixes before overlay dispatch work.

### 6. Fix duplicate/full expert preparation for overlay root devices

Once the hang is understood, wire overlay-filtered expert preparation through the root local TP finalization path so ROCm devices do not load all 30,720 MoE expert engines before loading the 960 engines assigned by the overlay plan.

Files likely involved:

- `src/v2/execution/runner/RankOrchestrator.cpp`
- `src/v2/loaders/WeightManager.cpp`
- `src/v2/loaders/WeightManager.h`
- overlay preparation plan/registry files under `src/v2/loaders/`

This is required for default-context benchmark viability and probably for realistic Phase 11 performance.

### 7. Replace the temporary CPU fallback gating with full graph-native dispatch

Current local-rank-0 CPU fallback execution is an acceptable bridge for getting to a runnable benchmark. The final Phase 10A/Phase 11 architecture should allow all local TP continuation participants to enter the same logical overlay dispatch group and publish ordered records:

- real fallback work where applicable;
- no-op for participants with no assigned fallback work;
- cancel/error propagation if a participant fails.

The CPU fallback endpoint runner should consume these ordered dispatch points rather than assuming a simple per-layer loop forever.

### 8. Benchmark acceptance sequence

Once prefill completes at `-c 512`:

1. run with `-n 0` and confirm benchmark reports success;
2. run with `-n 1` to include a decode step;
3. only then increase context length or tune memory/workspace residency toward the default context.

Example next acceptance commands:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=300000 \
./build_v2_release/llaminar2 benchmark \
  --config configs/qwen35-moe-overlay-layout-a.yaml \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p Hello -n 0 -t 0 -c 512

LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=300000 \
./build_v2_release/llaminar2 benchmark \
  --config configs/qwen35-moe-overlay-layout-a.yaml \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p Hello -n 1 -t 0 -c 512
```

## Files most likely relevant for the next agent

Core orchestration/runner:

- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/execution/runner/RankOrchestrator.cpp`
- `src/v2/execution/runner/InferenceRunnerFactory.cpp`
- `src/v2/execution/runner/BenchmarkRunner.cpp`

Local TP / RCCL:

- `src/v2/parallel/LocalTPContext.cpp`
- `src/v2/parallel/RCCLBackend.cpp`
- `src/v2/parallel/RCCLCoordinator.cpp`

Overlay runtime and fallback:

- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.h`
- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp`
- `src/v2/execution/moe/MoEOverlayDomainRuntime.h`
- `src/v2/execution/moe/MoEOverlayDomainRuntime.cpp`
- `src/v2/execution/moe/MoEOverlayDispatchCollective.h`
- `src/v2/execution/moe/MoEOverlayDispatchCollective.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.cpp`
- `src/v2/execution/compute_stages/stages/MoEOverlayDomainRuntimeStage.cpp`

Qwen35 graph:

- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/models/qwen35moe/Qwen35GraphConfigBuilder.cpp`

Weight preparation/residency:

- `src/v2/loaders/WeightManager.cpp`
- `src/v2/loaders/ExpertGemmRegistry.*`
- `src/v2/loaders/MoEExpertOverlayPreparationPlan.*`

Config/tests/docs:

- `configs/qwen35-moe-overlay-layout-a.yaml`
- `configs/qwen35-moe-overlay-layout-b.yaml`
- `tests/v2/unit/execution/moe/Test__MoEOverlayDispatchCollective.cpp`
- `tests/v2/unit/execution/moe/Test__MoEOverlayDomainRuntime.cpp`
- `tests/v2/integration/Test__MoEExpertOverlay_CPUFallback_MPI.cpp`
- `tests/v2/integration/Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI.cpp`
- `docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md`

## Bottom line

The Phase 10A work has moved from “benchmark cannot construct the right rank roles” to “benchmark constructs the right rank roles and hangs inside the first root local TP forward.” The CPU fallback endpoint is now architecturally aligned with the `GlobalTPContext` model, but the root continuation graph likely has a shared-context/RCCL synchronization problem or a closely related local TP issue that must be solved before meaningful Phase 11 benchmark work can begin.