# MoE Graph-Native Sparse Collectives Project Plan

**Date:** 2026-05-13
**Status:** Architecture reset project plan
**Branch context:** `feat/qwen35-moe`
**Scope:** Restore Llaminar's original per-device graph model for MoE overlay execution, make CPUs first-class graph participants, and replace nested multi-device MoE stages with sparse graph collectives.

## Design North Star

Llaminar graphs remain per-device and symmetric. MoE does not introduce nested multi-device subgraphs. Sparse MoE routing is represented as graph collectives. Every compute stage is participant-local. Devices with no work no-op until the next collective.

## Summary

The current Qwen3.5 MoE overlay implementation proved that tiered hot/cold expert placement can be made mathematically correct, but the production execution path has drifted away from Llaminar's simplest and strongest architectural invariant: one graph runner per participating device, with symmetric graph structure and explicit collective rendezvous points.

The current overlay path embeds a multi-device expert-domain executor inside a single graph stage. That shape conflicts with normal LocalTP execution. In a normal LocalTP graph, every participant executes only its own local shard, then meets peers at collective stages. In the current MoE overlay graph, a participant-local stage can execute an entire accelerator expert domain, including peer-device kernels, peer scratch, and domain collectives. When that same stage is present in more than one device graph, the whole domain execution can run in duplicate.

This plan moves MoE back into the graph system instead of building a second scheduler inside a stage. CPUs become full graph participants. Sparse dispatch and sparse return/reduce become graph collective stages. Expert compute becomes participant-local on every device type.

## Triggering Symptoms

The reduced Layout A ROCm integration probe exposed two related failures:

1. Both ROCm graph participants run the same `rocm_hot` LocalTP expert domain in duplicate.
2. A one-token run performs repeated large device-to-host transfers of full arena-sized tensors, suggesting dense synchronization where sparse row movement should dominate.

The duplicate execution is visible in traces where `rocm:1` enters the layer 8 MoE LocalTP executor, launches participant 0 and participant 1 work, completes an allreduce sequence, and then `rocm:0` enters the same executor and repeats the same domain work with a new sequence number. RCCL is being asked to participate in an execution pattern that violates the graph contract. Disabling RCCL or routing the allreduce through host staging would hide the symptom, not fix the architecture.

## Current Architecture Mismatch

### Original LocalTP Shape

Bog-standard LocalTP over `rocm:0,rocm:1` has symmetric per-device graphs:

```text
ROCm:0 graph                         ROCm:1 graph
------------                         ------------
local q/k/v shard                    local q/k/v shard
local attention output shard         local attention output shard
collective rendezvous                collective rendezvous
local FFN shard                      local FFN shard
collective rendezvous                collective rendezvous
```

The graph is duplicated, but each stage means "do my participant's local work." Domain-wide coordination appears only at collective stages.

### Current MoE Overlay Shape

The current MoE overlay path in [Qwen35MoEGraph.cpp](../../../../src/v2/models/qwen35moe/Qwen35MoEGraph.cpp) can attach a `MoEOverlayDomainRuntimeStage` or `MoEExpertOverlayLocalTPStage` whose params include all participants for a runtime domain. The executor in [MoEExpertOverlayLocalTPExecutor.cpp](../../src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp) then loops over the domain's prepared participants, binds their device streams, runs their experts, allreduces the partials, and scatters results.

That is a domain-level subgraph hidden inside a participant-local stage:

```text
ROCm:0 graph stage:
  run ROCm:0 expert work
  run ROCm:1 expert work
  run rocm_hot collective

ROCm:1 graph stage:
  run ROCm:0 expert work
  run ROCm:1 expert work
  run rocm_hot collective
```

This is the root architectural flaw.

### Current CPU Fallback Shape

CPU fallback ranks are not yet normal graph participants. They run through [MoEOverlayCPUFallbackParticipantRunner.cpp](../../src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp), which behaves like a sidecar endpoint for MoE work instead of a graph runner that participates in the same stage sequence as accelerators.

That creates two execution models:

- GPU and continuation devices run normal graphs.
- CPU fallback devices run a worker protocol adjacent to the graph.

The long-term model should have one execution abstraction: graph participants plus collectives.

## Target Architecture

Every device that can contribute to a MoE layer has its own graph runner. That includes CPU sockets/ranks.

For each MoE layer, the graph contains sparse MoE stages with the same logical sequence on every participant:

```text
MoERouteStage
MoESparseDispatchStage
MoELocalExpertStage
MoESparseReturnReduceStage
```

Participants that do not own routed work for a stage no-op locally but still enter the next collective rendezvous.

Example Layout A shape:

```text
ROCm:0 graph              ROCm:1 graph              CPU graph participant
------------              ------------              ---------------------
route or receive route    route or receive route    no-op or receive route
sparse dispatch           sparse dispatch           sparse dispatch
hot expert shard          hot expert shard          cold expert shard
sparse return/reduce      sparse return/reduce      sparse return/reduce
continue layer            no-op as needed           no-op as needed
```

The important invariant is that `hot expert shard` on `rocm:0` only launches `rocm:0` work. `hot expert shard` on `rocm:1` only launches `rocm:1` work. CPU expert stages only launch CPU-owned work.

## Composite Graph-Role Runner Boundary

Graph-native MoE still needs an outer runner for ranks that are not simple single-device continuation roots. That runner must be a step clock and lifecycle coordinator, not a hidden execution domain.

The composite runner owns:

- Rank role lifecycle: continuation root, local accelerator participant, CPU expert participant, relay/no-op participant.
- Participant graph construction and ownership.
- Prefill/decode step headers, position state, shutdown, cancellation, and failure propagation.
- Starting local participant graph runners for a step and waiting for their completion.
- Lightweight scalar control traffic between ranks, such as `BeginStep`, `EndStep`, `AbortStep`, and `Shutdown`.

The composite runner must not own:

- Routed tensor rows, hidden-state payloads, partial output rows, or expert weights as runtime work items.
- Peer-device kernel launch, peer-device stream binding, or peer scratch buffers.
- Domain-level allreduce/allgather/scatter logic.
- A `submitDomainWork()`-style API that lets a graph stage call back into a domain scheduler.

### Proposed Runner Shape

The implementation target is a `MoEGraphRoleRunner` (name provisional) that implements `IInferenceRunner` on ranks participating in an overlay plan.

```cpp
struct MoEGraphParticipantSpec {
    int participant_id = -1;
    int world_rank = -1;
    GlobalDeviceAddress address;
    DeviceId local_device = DeviceId::invalid();
    std::string domain_name;
    bool continuation_root = false;
    bool owns_dense_model_path = false;
    bool owns_routed_experts = false;
    bool owns_shared_expert = false;
};

struct MoEGraphStepHeader {
    uint64_t generation_id = 0;
    uint64_t step_id = 0;
    ExecutionPhase phase = ExecutionPhase::PREFILL;
    int seq_len = 0;
    int position_offset = 0;
    bool final_step = false;
};

class MoEGraphRoleRunner final : public IInferenceRunner {
public:
    bool forward(const int *tokens, int seq_len) override;
    void clear_cache() override;
    const float *logits() const override;

private:
    bool broadcastStepHeader(const MoEGraphStepHeader &header);
    bool runLocalParticipantGraphs(const MoEGraphStepHeader &header,
                                   const int *root_tokens);
    bool collectParticipantStatus(const MoEGraphStepHeader &header);
};
```

This runner may own one or more participant graph runners, but each participant graph runner still executes a normal `ComputeGraph`. The runner can decide that a CPU-only overlay rank should wake up and run its CPU participant graph for decode step 42. It cannot decide which token rows are sent to that CPU participant; that decision belongs to the `MoESparseDispatchStage` collective inside the graph.

### Allowed Step Flow

```text
MoEGraphRoleRunner::forward(tokens, seq_len)
  build MoEGraphStepHeader
  broadcast/receive scalar step header
  run local participant graph(s)
    graph stage: MoERouteStage or route-receive no-op
    graph stage: MoESparseDispatchStage enters dispatch collective
    graph stage: MoELocalExpertStage runs local-device experts only
    graph stage: MoESparseReturnReduceStage enters return/reduce collective
  collect graph completion status
  expose logits only on continuation root
```

### Forbidden Step Flow

```text
MoEGraphRoleRunner::forward(...)
  dispatch routed rows to rocm_hot
  launch rocm:0 and rocm:1 expert kernels
  allreduce rocm_hot partials
  scatter rows back to continuation output
```

That forbidden flow is just the current domain runtime moved one layer up. It must remain impossible by interface design.

### Interface Litmus Tests

The composite runner design is acceptable only if these checks hold:

1. Compute-stage params never contain `MoEGraphRoleRunner*`, `IOverlayDomainRuntime*`, or a domain-work request object.
2. Compute-stage params for local expert compute contain only the current participant's prepared expert refs and scratch, never a vector of peer prepared participants.
3. Tensor payload movement appears only in graph collective stages such as `MoESparseDispatchStage` and `MoESparseReturnReduceStage`.
4. The runner's cross-rank messages are scalar control messages only. They do not include hidden rows, routing rows, expert partials, or tensor pointers.
5. Every participant graph enters MoE collective stages in the same logical order for a given `(generation_id, step_id, layer, tier)`, even when its local payload is empty.
6. A non-root CPU participant can make progress by executing its graph after receiving a step header; it does not need a sidecar work-loop API.
7. The continuation root is the only role that exposes logits, sampling, and tokenizer-facing state.

### Collective Context Boundary

Collective stages may hold an `IMoESparseCollectiveContext*` or equivalent backend handle, but that handle is not the composite runner. Its API should be limited to named collective participation:

```cpp
class IMoESparseCollectiveContext {
public:
  virtual MoESparseDispatchCompletion beginDispatch(
    const MoESparseDispatchCall &call,
    IDeviceContext *ctx) = 0;
  virtual MoESparseDispatchResult waitDispatch(
    MoESparseDispatchCompletion &completion) = 0;
  virtual MoESparseReturnCompletion beginReturnReduce(
    const MoESparseReturnCall &call,
    IDeviceContext *ctx) = 0;
  virtual MoESparseReturnResult waitReturnReduce(
    MoESparseReturnCompletion &completion) = 0;
    virtual void requestAbort(uint64_t generation_id, uint64_t step_id) = 0;
};
```

The context can use MPI, host staging, shared memory, NCCL, or RCCL internally. It must not expose `submitDomainWork`, `runDomain`, `executeTier`, or peer-kernel APIs.

## CPU Participant Identity Boundary

CPU participant identity should be rank-qualified, not modeled as multiple local CPU devices inside one rank.

Llaminar's normal launch model binds MPI ranks to sockets: `MPIBootstrap` maps and binds by socket, `DeviceManager` initializes with the detected local NUMA node, and CPU backend allocation is already NUMA-local to the process. Under that invariant, each MPI rank has exactly one local CPU execution context: "this rank's socket-bound CPU." A bare local `DeviceId::cpu()` is therefore acceptable as a rank-local kernel/context identifier.

The plan should not require `DeviceId::cpu(1)` for graph-native MoE unless Llaminar later supports multiple CPU graph participants inside one MPI rank. That would be a separate feature, because it would require explicit thread partitioning, per-participant OpenMP placement, separate CPU arenas, and NUMA-bound allocation inside a single process. It is not required for the MoE overlay reset.

The canonical CPU participant identity for this project is:

```cpp
struct CpuGraphParticipantIdentity {
  int participant_id;
  int world_rank;
  GlobalDeviceAddress address;  // hostname + NUMA node + CPU + local ordinal 0
  DeviceId local_device;        // always DeviceId::cpu() for CPU participants
};
```

`GlobalDeviceAddress::cpu(numa, host)` should continue to identify the socket/NUMA domain globally, while `DeviceId::cpu()` identifies the local CPU backend inside the current rank. If a remote CPU participant must be distinguished from another CPU participant, the key must include `world_rank` or `participant_id`; it must not rely on local CPU ordinals.

This changes the interpretation of the earlier CPU identity concern:

- `GlobalDeviceAddress::cpu(...)` setting `device_ordinal = 0` is not a bug under one-CPU-per-rank semantics; the NUMA node and rank carry the socket identity.
- `GlobalDeviceAddress::toLocalDeviceId()` collapsing CPUs to `DeviceId::cpu()` is expected when converting to a rank-local execution device.
- `DeviceGraphExecutor::executeMultiDevice()` using the default context for CPU nodes is acceptable if each rank has one CPU context. It should assert or diagnose if a graph attempts to schedule multiple distinct local CPU participants in one rank.
- `CPUDeviceContext` does not need to bind threads by CPU ordinal. Binding is owned by MPI bootstrap / process affinity / OpenMP placement. CPU scratch allocation should still use rank-local NUMA-aware allocation or first-touch paths.
- `DeviceManager::deviceExists(DeviceId::cpu())` treating CPU as available is fine for local execution. Global placement validation must use rank-qualified `GlobalDeviceAddress`/participant identity.

### Required CPU Identity Changes

1. Keep `DeviceId::cpu()` as the canonical rank-local CPU device. Do not make Phase 1 depend on `DeviceId::cpu(n)`.
2. Make `MoEGraphParticipantSpec` carry the rank-qualified CPU participant identity: `participant_id`, `world_rank`, `GlobalDeviceAddress`, and local `DeviceId::cpu()`.
3. Update overlay planning, graph assignment, and expert placement so CPU participants are keyed by participant/global identity, never by bare `DeviceId` when the identity may cross rank boundaries.
4. Preserve the current one-local-CPU-per-rank invariant with validation. If a plan asks for two CPU participants on the same rank, fail with a clear error unless an explicit future multi-CPU-context mode is implemented.
5. Leave `GlobalDeviceAddress::cpu(numa, host)` with `device_ordinal = 0` unless/until multiple CPU devices per rank exist. Add comments/tests documenting that CPU socket identity is `hostname + numa_node + world_rank`, not `device_ordinal`.
6. Treat `GlobalDeviceAddress::toLocalDeviceId()` returning `DeviceId::cpu()` for CPU as correct rank-local conversion. Audit only call sites that incorrectly use the returned `DeviceId` as a global CPU participant key.
7. Audit weight placement, prepared-weight stores, graph assignments, and MoE domain planning for cross-rank CPU keys. Rank-local caches may still key CPU weights by `DeviceId::cpu()` because each rank has only one CPU participant.
8. Audit CPU graph-participant scratch allocation paths. They must inherit rank/socket affinity through existing MPI/OpenMP/NUMA mechanisms and must not introduce an ordinal-based CPU binding scheme.

## Sparse Buffer And Live-Row Contract

Sparse MoE cannot be implemented as a thin wrapper around full arena tensors.

Today [StageBufferContract.h](../../../../src/v2/memory/StageBufferContract.h) declares only whole `BufferId` reads/writes/inouts. [BufferArena.cpp](../../../../src/v2/memory/BufferArena.cpp) prepares and marks whole tensors. [StageBoundBuffers.h](../../../../src/v2/memory/StageBoundBuffers.h) exposes full `rows` and `cols`. The verification and dump paths still operate through full [StageDumpInfo](../../../../src/v2/execution/compute_stages/IComputeStage.h): [StageVerifier.cpp](../../../../src/v2/execution/local_execution/graph/StageVerifier.cpp) calls `ensureOutputsOnHost()` and validates full output shapes, while [StageDumper.h](../../../../src/v2/execution/debug/StageDumper.h) dumps full declared input/output buffers.

For sparse MoE, live-row shape is part of correctness. A one-token dispatch must carry compact row metadata and compact hidden rows through coherence, validation, dumping, and transport. If sparse stages expose only dense `[max_seq_len, d_model]` tensors, debug/integration builds will keep forcing full-tensor synchronization and validation. That makes sparse dispatch unusable exactly when the project needs integration assertions most.

### Sparse Payload Model

Add a sparse payload abstraction parallel to dense arena buffers:

```cpp
enum class MoESparseCollectiveDirection : uint8_t {
  Dispatch,
  LocalTPPartialReduce,
  ReturnReduce,
};

struct MoESparseCollectiveKey {
  uint64_t generation_id = 0;
  uint64_t step_id = 0;
  ExecutionPhase phase = ExecutionPhase::PREFILL;
  int layer_idx = -1;
  int tier_index = -1;
  int domain_id = -1;
  int collective_group_id = -1;
  uint64_t stage_sequence = 0;
  int microbatch_id = 0;
  uint64_t decode_sequence = 0;
  MoESparseCollectiveDirection direction = MoESparseCollectiveDirection::Dispatch;
};

enum class SparsePayloadKind : uint8_t {
  DispatchRows,
  LocalTPPartials,
  ReturnRows,
};

struct SparsePayloadId {
  SparsePayloadKind kind = SparsePayloadKind::DispatchRows;
  int layer_idx = -1;
  int tier_index = -1;
  int domain_id = -1;
  int slot = 0;
};

struct SparseLiveRows {
  MoESparseCollectiveKey key;
  size_t row_count = 0;
  size_t row_capacity = 0;
  size_t cols = 0;
  size_t routed_entry_count = 0;
  size_t routed_entry_capacity = 0;
  bool empty() const { return row_count == 0 && routed_entry_count == 0; }
};

struct SparsePayloadBuffers {
  TensorBase *values = nullptr;          // compact [row_capacity, cols]
  TensorBase *row_ids = nullptr;         // INT32 [row_capacity]
  TensorBase *entry_offsets = nullptr;   // INT32 [row_capacity + 1], optional
  TensorBase *expert_ids = nullptr;      // INT32 [routed_entry_capacity], optional
  TensorBase *routing_weights = nullptr; // FP32 [routed_entry_capacity], optional
  SparseLiveRows live;
};
```

`row_count` is not a debug scalar. It is part of the buffer state. It must travel with the payload from route/dispatch through local expert compute and return/reduce.

### Sparse Stage Contract Extensions

Extend `StageBufferContract` with sparse bindings rather than overloading dense `BufferBinding`:

```cpp
enum class SparseBindingRole : uint8_t {
  DenseRowsRead,     // read selected rows from a dense BufferId
  SparseRead,        // read compact sparse payload
  SparseWrite,       // write compact sparse payload
  SparseReadWrite,
};

struct SparseBufferBinding {
  SparsePayloadId id;
  SparseBindingRole role;
  const char *dtype = "FP32";
  BufferId dense_source = BufferId::_COUNT; // valid for DenseRowsRead
  BufferId row_selector = BufferId::_COUNT; // optional top-k/router output source
};

struct StageBufferContract {
  // existing dense inputs/outputs/inouts remain
  std::vector<SparseBufferBinding> sparse_bindings;

  StageBufferContract &addSparseInput(SparsePayloadId id, const char *dtype = "FP32");
  StageBufferContract &addSparseOutput(SparsePayloadId id, const char *dtype = "FP32");
  StageBufferContract &addSparseInOut(SparsePayloadId id, const char *dtype = "FP32");
  StageBufferContract &addDenseRowsRead(BufferId dense_source,
                      SparsePayloadId sparse_output,
                      BufferId row_selector = BufferId::_COUNT,
                      const char *dtype = "FP32");
};
```

The important distinction is `DenseRowsRead`: it prepares the dense source on the stage device, but downstream work reads only selected rows into a compact sparse output. It must not imply host access to the whole dense tensor.

`StageBoundBuffers` should gain sparse accessors:

```cpp
template <typename T, BufferAccess Access>
class SparseBufferView {
public:
  T *values = nullptr;
  int32_t *row_ids = nullptr;
  int32_t *entry_offsets = nullptr;
  int32_t *expert_ids = nullptr;
  float *routing_weights = nullptr;
  SparseLiveRows live;
  DeviceId device;
  TensorBase *values_tensor = nullptr;
};

template <typename T>
SparseBufferView<const T, BufferAccess::READ> sparseInput(SparsePayloadId id) const;

template <typename T>
SparseBufferView<T, BufferAccess::WRITE> sparseOutput(SparsePayloadId id) const;
```

Sparse stages should use these views, not raw `TensorBase::data()` or full dense `StageDumpInfo` buffers.

### Sparse Arena Coherence

`BufferArena` should either own a `SparseBufferArena` member or be extended with a separate sparse-payload registry. The sparse registry owns compact payload tensors and their live-row metadata.

Required operations:

```cpp
bool registerSparsePayload(SparsePayloadId id,
               size_t row_capacity,
               size_t cols,
               size_t routed_entry_capacity,
               const char *dtype,
               DeviceId home_device);

bool prepareSparseForRead(SparsePayloadId id, DeviceId target, void *stream = nullptr);
bool prepareSparseForWrite(SparsePayloadId id,
               const SparseLiveRows &shape_hint,
               DeviceId target,
               void *stream = nullptr);
void markSparseWritten(SparsePayloadId id,
             const SparseLiveRows &live,
             DeviceId device,
             void *stream = nullptr);
SparsePayloadBuffers *getSparsePayload(SparsePayloadId id);
```

Rules:

- Sparse coherence transfers only compact payload buffers up to live row/entry counts.
- `prepareSparseForRead` must not cohere or download the dense source buffer unless the binding explicitly asks for dense rows and the source is not already readable on the target device.
- `prepareSparseForWrite` allocates capacity and resets live metadata for the key being written.
- `markSparseWritten` records the authoritative device and the exact live row/entry counts for that key.
- Zero-live-row payloads are valid. They still carry the collective key and participate in rendezvous, but validation and dumping must not treat zero values as a failed tensor output.
- Sparse payload capacities are preallocated from graph/schema maxima. Decode should not allocate compact tensors on the hot path.

### Sparse Gather And Scatter Semantics

The first sparse dispatch stage must distinguish between dense source coherence and sparse row movement:

```text
dense HIDDEN_STATE on target device
selected row ids / routing CSR metadata
device or CPU gather into SparsePayloadBuffers.values
transport compact payload
```

For CPU stages, host-side gather/scatter can reuse [MoEExpertTokenRowTransfer.h](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.h). For GPU stages, production paths need device-side gather/scatter kernels or backend copy helpers. Calling `TensorBase::data()` on a GPU-resident dense hidden state is bridge-only because it can force full D2H synchronization.

### Existing Sparse Helper Reuse

The sparse reset should reuse existing code, but only at the layer where that code is valid.

#### `MoEExpertTokenRowTransfer`

[MoEExpertTokenRowTransfer.h](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.h) already provides useful sparse row utilities:

- `estimateVolume(...)` for dense-vs-sparse byte accounting.
- `validateTokenRows(...)` for row id sanity checks.
- `ensureBuffers(...)` / `allocateBuffers(...)` for host-side compact FP32 buffers.
- `gatherRows(...)` and `scatterAddRows(...)` for correctness-oriented compaction/scatter.

However, [MoEExpertTokenRowTransfer.cpp](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.cpp) is host-facing today. `gatherRows(...)` calls `hidden->data()`, `routing_indices->data()`, and `routing_weights->data()`, and `scatterAddRows(...)` calls `sparse_output->data()` plus `full_output->mutable_data()`. On GPU-resident activations, those calls can force full D2H/H2D synchronization and defeat sparse execution.

Reuse decision:

- Keep `MoEExpertTokenRowTransfer` as the CPU / host-staged correctness implementation and test oracle.
- Move its shape/volume/row-validation pieces into a backend-neutral utility if needed.
- Do not use its host `gatherRows(...)` or `scatterAddRows(...)` in production GPU sparse stages.
- Add a device-side equivalent, for example `MoESparseRowPacker`, with CPU and GPU implementations:

```cpp
class IMoESparseRowPacker {
public:
    virtual bool gatherDispatchRows(const DenseRowsReadBinding &source,
                                    SparsePayloadBuffers &dst,
                                    IDeviceContext *ctx) = 0;
    virtual bool scatterAddReturnRows(const SparsePayloadBuffers &src,
                                      BufferId dense_output,
                                      IDeviceContext *ctx) = 0;
};
```

The CPU implementation may delegate to `MoEExpertTokenRowTransfer`. GPU implementations must gather/scatter on device or through explicit compact staging controlled by the sparse transport backend.

#### `MoEExpertParallelReduceStage`

[MoEExpertParallelReduceStage.h](../../../../src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.h) already models sparse partial metadata through `MoEExpertParallelReducePartialInfo::selected_rows`. It also supports sparse partial diagnostics and scatter-adds compact partials into the continuation output.

That stage should not be discarded, but it should not become the cross-rank transport either. Its current responsibility is local accumulation/reduction over tensors already visible to the continuation process. Graph-native sparse return needs two separable responsibilities:

1. Transport compact return payloads from expert participants to the continuation participant.
2. Accumulate those compact payloads into the dense continuation output by row id.

Reuse decision:

- Split `MoEExpertParallelReduceStage` into a reusable accumulation core plus compatibility wrapper.
- The accumulation core should accept `SparsePayloadBuffers`/`SparseLiveRows` and preserve the existing `selected_rows` semantics.
- `MoESparseReturnReduceStage` should own graph collective participation and transport completion, then call the accumulation core on the continuation participant.
- The existing `MoEExpertParallelReduceStage` may remain as a dense/runtime-service compatibility stage during migration, but it should be reimplemented on top of the same accumulation core where practical.
- Runtime-service-only fields such as `runtime_partial_results` should not appear in graph-native sparse return stage params.

Proposed split:

```text
MoESparseTransport
  moves compact return payloads and validates collective keys

MoESparseRowAccumulator
  local helper extracted from MoEExpertParallelReduceStage sparse path
  scatter-adds compact rows into continuation output

MoESparseReturnReduceStage
  graph collective stage: wait transport, then accumulate on continuation participant

MoEExpertParallelReduceStage
  compatibility wrapper for dense/runtime partials during migration
```

This avoids inventing return/reduce from scratch while also avoiding a transport-shaped overload of `MoEExpertParallelReduceStage`.

### Sparse Dump And Verification

`StageDumpInfo` needs sparse entries separate from dense inputs/outputs:

```cpp
struct SparseDumpBuffer {
  const char *name = nullptr;
  SparsePayloadBuffers *payload = nullptr;
  SparseLiveRows live;
  const char *dtype = "FP32";
};

std::vector<SparseDumpBuffer> sparse_inputs;
std::vector<SparseDumpBuffer> sparse_outputs;
```

Verification and dumping rules:

- Entry/exit verification validates only `live.row_count * live.cols` values for sparse payload values.
- Routing metadata validation checks `row_ids`, `entry_offsets`, `expert_ids`, and `routing_weights` up to live counts.
- Output zero checks are disabled for zero-live sparse payloads and use compact live rows for non-empty payloads.
- `ensureSparseOutputsOnHost()` syncs only compact payload tensors, not the dense source tensor.
- Stage dumps write compact payload files plus metadata: `row_count`, `row_capacity`, `routed_entry_count`, `key`, and original `row_ids`.
- Snapshot/parity callbacks must opt into sparse payload materialization explicitly; they should not call `ensureOutputsOnHost()` on dense buffers just to inspect sparse rows.

This sparse dump/verification support belongs before Phase 3, not in Phase 7. Phase 7 can optimize GPU-native sparse validation and dumps, but Phase 3 must already avoid full activation D2H for sparse stages under Integration assertions.

## Sparse MoE Data-Plane Transport Contract

Sparse MoE dispatch must be a real data-plane collective, not a sidecar request to a domain runtime.

[MoEOverlayDispatchCollective.h](../../src/v2/execution/moe/MoEOverlayDispatchCollective.h) is a useful seed for naming and rendezvous identity, but it is currently control-plane only: request kind, counts, pointers, metrics, and completion. [MoEOverlayMPIDispatchBackend.cpp](../../src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp) broadcasts a fixed header; it does not move hidden rows, routing rows, partial rows, or receive buffers. [MoEExpertTokenRowTransfer.cpp](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.cpp) can compact and scatter host-visible sparse rows, but it is not a cross-rank or cross-device transport. The graph-native plan must add that missing transport layer explicitly.

At minimum, the graph needs these data-plane stages:

```text
MoESparseDispatchStage
MoELocalExpertStage
MoELocalTPPartialReduceStage   # only for TP expert domains; no-op for degree 1
MoESparseReturnReduceStage
```

The TP partial reduce is listed here because accelerator LocalTP cannot become truly participant-local while expert partial allreduce remains hidden inside `MoELocalExpertStage`.

### Collective Key

Every sparse collective call uses `MoESparseCollectiveKey` from the sparse payload model. The key must be present in headers, trace logs, receive slots, sparse live-row metadata, and completion records.

Rules:

- `stage_sequence` is monotonic within `(generation_id, step_id, layer_idx, tier_index, direction)`.
- A ring slot is keyed by the full key, not by `stage_sequence` alone. Slot reuse is illegal until all participants have completed or aborted that key.
- Receivers must reject stale payloads whose key does not exactly match the expected graph stage key.
- No-op participants still enter the collective with zero counts for the same key.

### Dispatch Payload Layout

`MoESparseDispatchStage` sends compact selected token rows plus routing entries from the routing/continuation participant to expert-domain participants.

The canonical dispatch payload is row-major FP32 for hidden rows in the first implementation:

```cpp
struct MoESparseDispatchPayloadView {
  MoESparseCollectiveKey key;
  int source_participant_id = -1;
  int target_participant_id = -1;
  int d_model = 0;
  int top_k = 0;

  // row_ids are original token rows in the continuation hidden state.
  const int32_t *row_ids = nullptr;          // [row_count]
  size_t row_count = 0;

  // Hidden rows are compact [row_count, d_model].
  const float *hidden_rows = nullptr;
  size_t hidden_row_stride = 0;

  // Routing entries are CSR over row_ids. Multiple entries per row are allowed.
  const int32_t *entry_offsets = nullptr;    // [row_count + 1]
  const int32_t *expert_ids = nullptr;       // [routed_entry_count]
  const float *routing_weights = nullptr;    // [routed_entry_count]
  size_t routed_entry_count = 0;
};
```

Destination participants receive a compact local view with the same logical fields. For a LocalTP expert domain, every TP participant that contributes to the expert receives the same compact row set for its shard. For non-TP expert placement, only the participant that owns the target expert receives that expert's entries.

### LocalTP Partial Reduce Payload

`MoELocalTPPartialReduceStage` is an explicit graph collective for tensor-parallel expert domains. It replaces the hidden `allreduceExpertPartials(...)` inside [MoEExpertOverlayLocalTPExecutor.cpp](../../src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp).

Input per participant:

```text
key(direction=LocalTPPartialReduce)
expert_id
row_ids[]
partial_hidden_rows[row_count, d_model]
```

The stage reduces matching compact partial rows across the TP group. The first implementation may allreduce one expert at a time, mirroring the current executor sequence, but the collective must be a named graph stage. It must not be launched from `MoELocalExpertStage`.

Output policy:

- Degree 1 domains pass through without transport.
- TP domains either materialize the reduced compact output on every TP participant or on a designated return owner. The policy must be explicit in the stage params and trace output.
- The return owner is the only participant that sends a reduced TP result into `MoESparseReturnReduceStage` unless the return/reduce stage is explicitly configured to deduplicate replicas.

### Sparse Return/Reduce Payload

`MoESparseReturnReduceStage` returns expert outputs keyed by original token row and reduces them into the continuation hidden state.

The local expert stage should apply routing weights before return so return/reduce only sums row partials:

```cpp
struct MoESparseReturnPayloadView {
  MoESparseCollectiveKey key;
  int source_participant_id = -1;
  int target_participant_id = -1;
  int d_model = 0;
  const int32_t *row_ids = nullptr;       // [row_count]
  size_t row_count = 0;
  const float *partial_rows = nullptr;    // [row_count, d_model]
  size_t partial_row_stride = 0;
};
```

The continuation participant receives zero or more compact partial sets and scatter-adds them into its dense continuation output. This should reduce sparse rows, not synchronize full `[max_seq_len, d_model]` slabs when only one or a few rows are live.

### Buffer Ownership And Lifetimes

The transport contract must be explicit about ownership:

- Source graph stages own source tensor views only until `beginDispatch(...)` or `beginReturnReduce(...)` has copied or retained them according to the backend contract.
- The sparse transport owns staging buffers, MPI receive buffers, pinned host buffers, and local ring slots until the completion handle is waited and released.
- Destination graph stages receive arena-backed or stage-owned compact output buffers. They do not receive raw pointers into another participant's tensors.
- Completion handles are mandatory even for a blocking MVP. A stage may not read receive buffers until `waitDispatch(...)` or `waitReturnReduce(...)` succeeds.
- Slot acquisition provides backpressure. If all slots are still live, the backend blocks within the stage or returns a typed error; it must not silently overwrite a slot.
- Cancellation marks the key aborted, wakes all waiters, and prevents later stale payloads for that key from being consumed.

### Transport Backend Responsibilities

The transport backend is responsible for counts, payload movement, device placement, and completion. It is not allowed to run expert kernels.

Minimum API shape:

```cpp
struct MoESparseDispatchCall {
  MoESparseCollectiveKey key;
  int participant_id = -1;
  DeviceId local_device = DeviceId::invalid();
  std::vector<MoESparseDispatchPayloadView> outbound;
  MoESparseReceiveBufferSet *receive_buffers = nullptr;
};

struct MoESparseReturnCall {
  MoESparseCollectiveKey key;
  int participant_id = -1;
  DeviceId local_device = DeviceId::invalid();
  std::vector<MoESparseReturnPayloadView> outbound;
  MoESparseReceiveBufferSet *receive_buffers = nullptr;
};
```

Implementation requirements:

- All participants call the same collective key in the same graph order.
- The backend first exchanges per-target counts, including zero counts for no-op participants.
- Inter-rank CPU and remote participant payloads use MPI transport. The MVP can use host-staged `MPI_Isend`/`MPI_Irecv` or `MPI_Alltoallv`; either way the key must be encoded into tags or payload headers and verified on receipt.
- Same-rank CPU payloads can copy through host memory into the destination receive buffers.
- Same-rank accelerator payloads initially use host-staged compact buffers plus `TransferEngine` H2D/D2H/P2P as needed. A future NCCL/RCCL all-to-all path can replace the movement backend without changing graph-stage semantics.
- Remote accelerator participants receive MPI payloads into host or pinned host buffers, then upload compact rows to device-resident receive buffers before `waitDispatch(...)` returns.
- Metrics report selected rows, routed entries, return rows, payload bytes, staging bytes, H2D/D2H bytes, wait time, no-op participants, and stale/aborted payload rejects.

### Relationship To Existing Code

- [MoEOverlayDispatchCollective.h](../../src/v2/execution/moe/MoEOverlayDispatchCollective.h) should either be renamed or wrapped as the sparse collective control header layer. It is not sufficient as the payload transport.
- [MoEOverlayMPIDispatchBackend.h](../../src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h) can seed the fixed-header exchange, but it needs a payload phase and receive-slot lifetime management before graph-native sparse dispatch can depend on it.
- [MoEExpertTokenRowTransfer.h](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.h) remains the CPU/host-staged correctness implementation and test oracle for sparse row pack/scatter. Production GPU paths must use a device-side row packer and avoid unconditional `TensorBase::data()` calls in hot decode.
- [MoEExpertParallelReduceStage.h](../../../../src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.h) should seed the local sparse row accumulation core. `MoESparseReturnReduceStage` should split transport from accumulation instead of replacing all existing reduce logic wholesale.

## Phased Implementation

### Phase 0: Lock The Graph Invariants

**Goal:** Make the architecture contract explicit before more MoE code lands.

Tasks:

- Document the invariant that compute stages are participant-local.
- Add debug assertions or trace checks that detect a graph stage trying to execute peer-device prepared participants.
- Add a focused regression test that would fail if two ROCm graph participants both execute the full `rocm_hot` LocalTP domain for the same layer/expert sequence.
- Keep the current CPU fallback runner temporarily, but mark it as a bridge.

Acceptance:

- Integration traces can identify whether a MoE stage is participant-local or domain-level.
- The duplicate ROCm LocalTP execution pattern is captured by a test or diagnostic assertion.

### Phase 0.5: Add The Composite Graph-Role Runner Skeleton

**Goal:** Replace sidecar endpoint control flow with an overlay runner that clocks participant graphs without owning domain execution.

Tasks:

- Add `MoEGraphRoleRunner` as the overlay-aware `IInferenceRunner` for ranks with continuation, accelerator participant, CPU participant, or relay roles.
- Define `MoEGraphParticipantSpec` and `MoEGraphStepHeader` as scalar-only contracts. These types must not contain tensor pointers, hidden-row payloads, expert partials, or prepared peer participants.
- Teach `OrchestrationRunner` to create `MoEGraphRoleRunner` for non-root overlay ranks instead of immediately selecting `MoEOverlayCPUFallbackParticipantRunner`.
- Create placeholder participant graph runners that can execute no-op graphs and report status for a step.
- Keep existing CPU fallback work behind the compatibility bridge until graph-native CPU participant stages exist, but do not expose it through the new runner interface.
- Add compile-time or unit-test checks that new graph-native MoE stages cannot depend on `MoEGraphRoleRunner` or `IOverlayDomainRuntime`.

Acceptance:

- A non-root overlay rank can receive a scalar step header, execute a no-op participant graph, report success/failure, and shut down cleanly.
- `MoEGraphRoleRunner` has no API that accepts routed rows, expert partials, domain work requests, or peer-device prepared participant vectors.
- No graph-native MoE stage introduced for this runner has a direct reference to `MoEGraphRoleRunner`.
- The legacy CPU fallback runner remains available only as a bridge and is not the interface new graph-native stages target.

### Phase 1: Make CPU Participants Rank-Qualified

**Goal:** CPU graph participants can be uniquely addressed, scheduled, and keyed without inventing multiple local CPU devices inside one MPI rank.

Tasks:

- Keep `DeviceId::cpu()` as the rank-local CPU execution device.
- Define CPU participant identity as `(participant_id, world_rank, GlobalDeviceAddress, DeviceId::cpu())` in `MoEGraphParticipantSpec` or a closely related type.
- Update overlay planning and MoE graph assignment so CPU participants are distinguished by participant/global identity, not by bare `DeviceId`.
- Validate the one-local-CPU-per-rank invariant. Reject any overlay plan that asks for two local CPU participants in one rank unless a future explicit multi-CPU-context mode is added.
- Document that `GlobalDeviceAddress::cpu(numa, host)` uses `numa_node` plus rank for socket identity and keeps local `device_ordinal = 0`.
- Audit call sites that convert CPU `GlobalDeviceAddress` to `DeviceId`. The conversion is fine for local execution, but the result must not be reused as a global CPU participant key.
- Audit weight placement, prepared-weight stores, graph assignments, and TP assignments for cross-rank CPU keys. Rank-local CPU caches can remain keyed by `DeviceId::cpu()`.
- Audit CPU scratch allocation used by graph participants and keep it on existing rank-local NUMA-aware paths.

Acceptance:

- CPU participants on different MPI ranks have distinct participant identities even though their local device is `DeviceId::cpu()`.
- Overlay planning cannot accidentally merge two remote CPU participants because both convert to `DeviceId::cpu()` locally.
- A plan requesting multiple CPU participants in one rank fails loudly with a clear explanation.
- Existing `DeviceId::cpu()` call sites do not need broad churn for this phase.

### Phase 2: Introduce CPU Graph Participants

**Goal:** CPU ranks run graph runners instead of sidecar fallback loops.

Tasks:

- Extend overlay execution planning so CPU participants receive graph assignments rather than `CpuFallbackParticipant` sidecar roles only.
- Build CPU graph runners with the same logical layer/stage sequence as accelerator participants.
- Add no-op implementations or no-op stage policies for dense stages that a CPU participant does not own.
- Ensure CPU graph participants enter MoE sparse collectives at the same logical layer points as accelerators.
- Keep tokenization, sampling, logits, and continuation ownership on the configured continuation domain.

Acceptance:

- A CPU-only MoE participant can initialize a graph without building a continuation-root graph.
- CPU participants can no-op non-owned dense stages and still rendezvous at MoE sparse collective stages.
- Existing CPU fallback behavior remains available through a compatibility bridge while graph-native CPU participation is developed.

### Phase 2.5: Add Sparse Buffer Views And Live-Row Contracts

**Goal:** Make sparse row payloads first-class in stage contracts, arena coherence, validation, and dumps before sparse transport stages depend on them.

Tasks:

- Add `SparsePayloadId`, `SparseLiveRows`, `SparsePayloadBuffers`, and a sparse payload registry owned by or adjacent to `BufferArena`.
- Extend `StageBufferContract` with sparse bindings: sparse input/output/inout and dense-selected-rows read bindings.
- Extend `StageBoundBuffers` with typed `SparseBufferView` accessors carrying compact value pointers, row ids, routing CSR metadata, live counts, and device identity.
- Add sparse coherence APIs: `prepareSparseForRead`, `prepareSparseForWrite`, `markSparseWritten`, and sparse payload lookup.
- Ensure sparse coherence transfers and validates only live compact rows and routed entries, not full dense arena capacity.
- Add sparse entries to `StageDumpInfo` and implement sparse-aware verifier/dumper paths.
- Add bridge-only guardrails around host helper use: `MoEExpertTokenRowTransfer` is allowed for CPU/correctness tests, but GPU sparse stages must not call `TensorBase::data()` on dense activation tensors in hot paths.
- Introduce `IMoESparseRowPacker` (or equivalent) with a CPU implementation backed by `MoEExpertTokenRowTransfer` and a GPU implementation path that does not require full dense host materialization.
- Extract a `MoESparseRowAccumulator` helper from the sparse path in `MoEExpertParallelReduceStage` so graph-native return/reduce and compatibility reduction share row-id accumulation semantics.

Acceptance:

- A sparse stage can write a compact payload with `row_count=1` and Integration validation reads only one compact row.
- A zero-live sparse payload is valid, carries its collective key, and does not fail all-zero output validation.
- Stage dumps for sparse payloads include compact values, row ids, routing metadata, live counts, capacity, and collective key.
- Transfer tracing for a GPU one-token sparse stage does not show full dense activation D2H caused by verifier/dumper plumbing.
- CPU sparse pack/scatter tests reuse `MoEExpertTokenRowTransfer`; GPU sparse pack/scatter tests fail if the implementation calls full dense `TensorBase::data()`/`mutable_data()` on activation buffers.
- `MoESparseRowAccumulator` matches the existing `MoEExpertParallelReduceStage` sparse `selected_rows` behavior on a shared fixture.
- Dense stages continue to use existing whole-buffer contracts without behavior changes.

### Phase 3: Add Graph-Native Sparse Dispatch Stages

**Goal:** Move MoE routing transport into graph collectives using the Phase 2.5 sparse buffer contract.

Tasks:

- Define `MoESparseCollectiveKey`, dispatch payload views, return payload views, receive buffer sets, and completion handles.
- Add a sparse transport backend with count exchange, payload movement, receive-slot lifetime management, abort, and no-op participation.
- Add `MoESparseDispatchStage` as the graph collective from continuation/routing participants to expert-domain participants.
- Add `MoELocalTPPartialReduceStage` as an explicit named collective for TP expert-domain partials.
- Add `MoESparseReturnReduceStage` as the graph collective from expert-domain participants back to continuation participants; it should wait transport completion, then call the shared sparse row accumulation core on the continuation participant.
- Use sparse payload bindings for dispatch, TP partial reduce, and return/reduce. These stages must not expose their compact payloads as dense `[max_seq_len, d_model]` outputs.
- Add trace counters for selected rows, routed entries, transferred bytes, no-op participants, reduced rows, stale payload rejects, and slot wait time.

Acceptance:

- One-token decode moves one live row per routed expert path, not full max-sequence activation slabs.
- Devices with no routed rows still enter collective stages, exchange zero counts, and complete the same rendezvous key.
- Dispatch, LocalTP partial reduce, and return/reduce can be unit-tested without launching expert kernels.
- Stale or out-of-order payloads are rejected by key validation rather than read from a reused slot.
- Receive buffers are not visible to downstream stages until the corresponding completion wait succeeds.

### Phase 4A: Contain The Accelerator LocalTP Bridge

**Goal:** Stop duplicate full-domain accelerator execution before the full sparse data plane lands.

This is a bridge phase, not graph-native execution. The current `MoEExpertOverlayLocalTPStage` stores all prepared participants in one stage param and [MoEExpertOverlayLocalTPExecutor.cpp](../../src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp) loops over every participant, refreshes every participant scratch, and allreduces partials inside the stage. That shape may be kept temporarily only if it is explicitly marked as a bridge and only one graph participant owns the bridge execution for a given domain/layer/tier sequence.

Tasks:

- Rename or tag the current all-participant accelerator stage path as a bridge, for example `MoELocalTPDomainBridgeStage` in diagnostics and comments.
- Add an owner-participant guard so exactly one graph participant executes the bridge for `(generation_id, step_id, layer_idx, tier_index, domain)`.
- Make all non-owner participants enter a matching no-op bridge rendezvous or skip marker so traces can prove they did not execute peer-device kernels.
- Add diagnostics that print `bridge_executes_peer_devices=true`, owner participant, domain, local graph device, and collective sequence.
- Forbid new graph-native stages from depending on the bridge stage params, especially `prepared_participants` and `prepared_partial_views` vectors.
- Keep RCCL enabled in the bridge only if the owner-only execution sequence is single-shot and not duplicated by peer graphs.

Acceptance:

- The reduced Layout A ROCm repro no longer shows both ROCm graph participants executing the entire `rocm_hot` domain.
- The bridge path is visibly labeled as non-graph-native in traces and diagnostics.
- Non-owner participants do not bind peer streams, launch peer kernels, refresh peer scratch, or call the domain allreduce.
- The plan has a test or assertion that fails if the bridge is inserted as an executable stage in more than one participant graph for the same key.

### Phase 4B: Refactor Accelerator MoE LocalTP To Participant-Local

**Goal:** Remove the nested multi-device subgraph from accelerator MoE expert compute.

This phase depends on Phase 3. Without `MoESparseDispatchStage`, `MoELocalTPPartialReduceStage`, and `MoESparseReturnReduceStage`, moving accelerator LocalTP to participant-local compute would only relocate hidden dynamic collectives rather than remove them.

Tasks:

- Replace `MoEExpertOverlayLocalTPStage` params that contain all domain participants with params for the current graph participant only.
- Split `MoEExpertOverlayLocalTPExecutor` into local expert compute and explicit graph collective participation.
- Ensure ROCm:0 only launches ROCm:0 expert kernels, ROCm:1 only launches ROCm:1 expert kernels, and RCCL/RCCL-backed collectives are entered exactly once per logical collective sequence per participant.
- Move participant partial allreduce/reduce-scatter behavior into `MoELocalTPPartialReduceStage`.
- Make sparse return use `MoESparseReturnReduceStage` rather than scatter-add inside the local expert compute stage.
- Remove peer stream binding and peer scratch ownership from participant-local stages.

Acceptance:

- The layer 8 reduced ROCm repro no longer shows duplicate full-domain execution.
- RCCL remains enabled for ROCm LocalTP collectives.
- The memory access fault does not reproduce under integration assertions for the reduced Layout A probe.
- `MoELocalExpertStage` params contain only the current participant's prepared expert refs and scratch.
- No accelerator expert compute stage calls LocalTP allreduce directly.

### Phase 5: Convert CPU Expert Domains To The Same Sparse Graph Path

**Goal:** Retire CPU fallback as a special execution model.

Tasks:

- Implement CPU `MoELocalExpertStage` using CPU-prepared expert shards and NodeLocalTP collectives where applicable.
- Replace `MoEOverlayCPUFallbackParticipantRunner` work loops with CPU graph participant stages.
- Preserve the existing CPU expert math and NodeLocalTP domain context behavior, but move invocation under the graph executor.
- Make CPU cold domains consume `MoESparseDispatchStage` output and produce `MoESparseReturnReduceStage` input.

Acceptance:

- CPU cold experts run as graph stages.
- No non-root CPU participant needs a custom MoE fallback runner to participate in inference.
- CPU participants and ROCm participants share the same MoE layer stage sequence.

### Phase 6: Remove `MoEOverlayDomainRuntimeStage`

**Goal:** Delete the domain-runtime-in-a-stage abstraction from the hot path.

Tasks:

- Replace `MoEOverlayDomainRuntimeStage` graph insertion in Qwen3.5 MoE with the sparse stage sequence.
- Retain small helper services only where they do not own graph control flow or peer-device execution.
- Remove owner/executor participant concepts that imply a single participant executes a full domain.
- Update tests and parity fixtures to use graph-native CPU and accelerator participants.

Acceptance:

- MoE graph construction does not add a stage that can execute peer-device kernels.
- Overlay parity, benchmark, and oneshot paths use the same graph-native sparse collective path.
- The old sidecar fallback path is either deleted or guarded behind a temporary legacy flag with no production use.

### Phase 7: Performance And Coherence Cleanup

**Goal:** Make sparse graph execution fast enough to validate real MoE layouts.

Tasks:

- Audit all MoE stages for `TensorBase::data()` or `mutable_data()` calls that force full D2H transfers.
- Optimize sparse dump/verification paths with GPU-side live-row validation where useful.
- Add per-stage transfer tracing for sparse dispatch and sparse return/reduce.
- Ensure decode fast paths allocate no hot-path scratch beyond arena or preallocated stage-owned buffers.
- Add benchmark counters for sparse transfer bytes versus dense tensor bytes.

Acceptance:

- One-token integration probes no longer spend minutes per stage on dense synchronization.
- Transfer traces show sparse row payload movement for MoE dispatch/return.
- Release benchmark and integration assertion paths both remain usable for Layout A and Layout B.

## Migration Strategy

The safest migration is to fix the accelerator duplicate-execution fault before fully retiring CPU fallback.

Recommended order:

1. Phase 0 establishes the participant-local invariant and catches duplicate domain execution.
2. Phase 0.5 introduces the composite graph-role runner skeleton with no tensor payload or domain-work APIs.
3. Phase 1 establishes rank-qualified CPU participant identity.
4. Phase 4A contains the current accelerator LocalTP bridge with owner-only execution to address the duplicate ROCm domain fault.
5. Phase 2.5 adds sparse buffer views and live-row-aware validation/dump/coherence plumbing.
6. Phase 3 introduces sparse graph data-plane collectives and the explicit LocalTP partial reduce stage.
7. Phase 4B makes accelerator LocalTP truly participant-local on top of the Phase 2.5/Phase 3 sparse path.
8. Phase 2 and Phase 5 promote CPU participants into graph runners.
9. Phase 6 removes the domain runtime stage once all active paths are graph-native.
10. Phase 7 removes residual dense transfer/coherence costs and optimizes sparse validation/dump paths.

This order keeps the immediate ROCm RCCL investigation honest while opening the door to the larger graph-native CPU design.

## File Impact Map

Likely high-impact areas:

- [GlobalDeviceAddress.h](../../../../src/v2/backends/GlobalDeviceAddress.h) and [GlobalDeviceAddress.cpp](../../../../src/v2/backends/GlobalDeviceAddress.cpp): document CPU local ordinal semantics and avoid treating `device_ordinal` as CPU socket identity.
- [DeviceRegistry.cpp](../../../../src/v2/backends/DeviceRegistry.cpp): keep CPU NUMA/socket discovery globally visible, but do not require multiple local CPU `DeviceId` ordinals.
- [StageBufferContract.h](../../../../src/v2/memory/StageBufferContract.h): add sparse payload bindings and dense-selected-row read bindings.
- [StageBoundBuffers.h](../../../../src/v2/memory/StageBoundBuffers.h): add typed `SparseBufferView` accessors with live-row metadata.
- [BufferArena.h](../../../../src/v2/memory/BufferArena.h) and [BufferArena.cpp](../../../../src/v2/memory/BufferArena.cpp): add sparse payload registration, compact coherence, sparse writes, and sparse live metadata.
- [IComputeStage.h](../../../../src/v2/execution/compute_stages/IComputeStage.h) and [ComputeStageBase.cpp](../../../../src/v2/execution/compute_stages/ComputeStageBase.cpp): extend `StageDumpInfo` with sparse input/output entries and sparse host-sync helpers.
- [StageVerifier.cpp](../../../../src/v2/execution/local_execution/graph/StageVerifier.cpp): validate sparse live rows and metadata without forcing full dense output sync.
- [StageDumper.h](../../../../src/v2/execution/debug/StageDumper.h) and [AsyncStageDumper.h](../../../../src/v2/execution/debug/AsyncStageDumper.h): dump compact sparse payloads and metadata rather than full capacities.
- [DeviceGraphExecutor.cpp](../../../../src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp): keep default CPU context behavior, but add diagnostics if a graph plan claims multiple local CPU participants in one rank.
- [QwenGraphBase.cpp](../../../../src/v2/models/qwen/QwenGraphBase.cpp): replace singleton-CPU workarounds with rank-qualified participant lookup where CPU assignment crosses ranks.
- [Qwen35MoEGraph.cpp](../../../../src/v2/models/qwen35moe/Qwen35MoEGraph.cpp): replace domain runtime nodes with sparse graph stages.
- [OrchestrationRunner.cpp](../../../../src/v2/execution/runner/OrchestrationRunner.cpp): route overlay ranks through the composite graph-role runner instead of sidecar endpoint-only execution.
- `src/v2/execution/moe/MoEGraphRoleRunner.h/.cpp` (new): own overlay participant graph lifecycle, scalar step headers, shutdown, and failure propagation.
- [MoEOverlayDispatchCollective.h](../../src/v2/execution/moe/MoEOverlayDispatchCollective.h): keep or wrap as the sparse collective header/rendezvous layer, not the payload transport itself.
- [MoEOverlayMPIDispatchBackend.h](../../src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h) and [MoEOverlayMPIDispatchBackend.cpp](../../src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp): extend beyond header broadcast into payload movement or split into a new transport backend.
- `src/v2/execution/moe/MoESparseTransport.h/.cpp` (new): own sparse payload count exchange, receive slots, completion handles, backpressure, stale-key rejection, and backend selection.
- `src/v2/execution/moe/MoESparseRowPacker.h/.cpp` (new or extracted): backend-neutral interface plus CPU/GPU implementations for compact row gather/scatter. CPU can delegate to `MoEExpertTokenRowTransfer`; GPU must avoid full dense host materialization.
- `src/v2/execution/moe/MoESparseRowAccumulator.h/.cpp` (new or extracted): local row-id accumulation helper extracted from `MoEExpertParallelReduceStage` sparse selected-row behavior.
- `src/v2/execution/compute_stages/stages/MoESparseDispatchStage.h/.cpp` (new): graph collective stage for dispatch payloads.
- `src/v2/execution/compute_stages/stages/MoELocalTPPartialReduceStage.h/.cpp` (new): explicit TP expert partial collective stage.
- `src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.h/.cpp` (new): graph collective stage for sparse return transport plus continuation accumulation via `MoESparseRowAccumulator`.
- [MoEExpertOverlayLocalTPStage.h](../../src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.h): remove all-participant params from participant-local stage execution.
- [MoEExpertOverlayLocalTPExecutor.cpp](../../src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp): split local compute from collective participation.
- [MoEExpertTokenRowTransfer.h](../../../../src/v2/execution/moe/MoEExpertTokenRowTransfer.h): reuse as the host-side correctness implementation for compact row pack/scatter, but do not treat it as the production GPU transport.
- [MoEExpertParallelReduceStage.h](../../../../src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.h): keep as a compatibility wrapper and extract/reuse its sparse selected-row accumulation semantics.
- [MoEOverlayCPUFallbackParticipantRunner.cpp](../../src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp): retire once CPU graph participants own CPU expert work.

## Testing Plan

### Unit Tests

- CPU `MoEGraphParticipantSpec` values on different ranks remain distinct even when `local_device == DeviceId::cpu()` for both.
- `GlobalDeviceAddress::cpu(...)` conversion to local `DeviceId::cpu()` is documented and tested as rank-local behavior.
- Overlay participant maps distinguish cross-rank CPU participants without using bare `DeviceId` as the global key.
- Overlay planning rejects multiple CPU participants assigned to the same rank.
- Sparse buffer bindings preserve live row counts through `StageBufferContract`, `StageBoundBuffers`, and `BufferArena` mark-written paths.
- Sparse verifier checks only live compact rows and treats zero-live payloads as valid no-op outputs.
- Sparse stage dump writes compact live rows and row metadata without syncing the dense source tensor.
- Dense-selected-row bindings prepare the dense source on the target device without implying full host access.
- CPU `IMoESparseRowPacker` output matches `MoEExpertTokenRowTransfer::gatherRows`/`scatterAddRows` on host fixtures.
- GPU `IMoESparseRowPacker` tests assert no full dense activation `TensorBase::data()`/`mutable_data()` calls are used for sparse pack/scatter.
- `MoESparseRowAccumulator` matches `MoEExpertParallelReduceStage` sparse `selected_rows` behavior for compact partials.
- `MoESparseCollectiveKey` rejects stale receive-slot reuse and mismatched direction/layer/tier/sequence keys.
- Sparse transport count exchange includes zero-count no-op participants and completes only after every participant enters.
- Sparse transport completion handles prevent downstream buffer reads before `wait*()` succeeds.
- Sparse dispatch payload packing preserves row ids, CSR routing entries, expert ids, routing weights, and hidden rows.
- LocalTP partial reduce can reduce compact partial rows by key without launching expert kernels.
- Sparse dispatch request construction handles routed work, no-op work, cancel, and shutdown.
- Sparse return/reduce combines partial rows by original row id through the shared sparse row accumulation core.
- `MoEGraphStepHeader` and `MoEGraphParticipantSpec` are scalar/control-plane-only and cannot carry tensor payloads or prepared peer participants.
- `MoEGraphRoleRunner` exposes no `submitDomainWork`, `runDomain`, `dispatchRows`, or peer-device execution API.

### Integration Tests

- NodeLocalTP CPU participants build graph runners and no-op non-owned dense stages.
- Non-root overlay ranks can execute a no-op participant graph under `MoEGraphRoleRunner` and report completion to the continuation root.
- The Phase 4A bridge executes an accelerator LocalTP domain only once per collective key; peer graph participants report no-op/skip rather than executing peer devices.
- With Integration validation and stage dumps enabled, a one-token sparse dispatch stage syncs/dumps only the compact sparse payload and does not trigger full hidden-state D2H.
- MPI host-staged sparse dispatch moves compact hidden/routing rows from a continuation rank to a CPU participant rank and rejects a deliberately stale key.
- Remote accelerator sparse dispatch receives compact rows into host/pinned buffers and uploads them to device-resident compact receive buffers before completion.
- ROCm LocalTP MoE layer with two participants executes one local shard per graph participant and one collective sequence per participant.
- Layout A reduced probe passes in `build_v2_integration` with RCCL enabled.
- CPU cold domain receives sparse dispatch rows and returns partials through graph collectives.

### Diagnostics

- Add a trace line for each MoE graph stage with `participant_device`, `domain`, `selected_rows`, `routed_entries`, `live_rows`, `live_capacity`, `collective_key`, `bridge_executes_peer_devices`, and `executes_peer_devices=false` for graph-native stages.
- Add transfer counters for dense bytes versus sparse bytes, staging bytes, H2D/D2H bytes, slot wait time, no-op participant count, and stale-key reject count.
- Add validation/dump counters for sparse live bytes versus dense tensor bytes that would have been synced by the legacy path.
- Keep `LLAMINAR_TRACE_TRANSFERS` useful by making sparse payload transfers identifiable in logs.

## Definition Of Done

This architecture reset is complete when:

1. CPU graph participants are uniquely identified by rank-qualified participant/global identity, while `DeviceId::cpu()` remains rank-local.
2. CPU ranks can own graph runners and participate in MoE sparse collectives.
3. MoE expert compute stages are participant-local on CPU, CUDA, and ROCm.
4. No graph stage executes peer-device kernels or owns a nested multi-device subgraph.
5. Sparse payloads are first-class buffer contract objects with live-row metadata, compact coherence, sparse dumps, and sparse validation.
6. Sparse MoE dispatch, LocalTP partial reduce, and sparse return/reduce are explicit graph collective stages with payload ownership, completion, and stale-key validation.
7. Existing sparse helpers are reused at the right layer: `MoEExpertTokenRowTransfer` for CPU/host correctness, and `MoEExpertParallelReduceStage` sparse selected-row logic as the shared local accumulation core.
8. Overlay rank lifecycle is owned by `MoEGraphRoleRunner`, whose public API is scalar-control-only and cannot submit domain work.
9. The Qwen3.5 MoE overlay path does not require `MoEOverlayCPUFallbackParticipantRunner` for production inference.
10. RCCL remains enabled for ROCm LocalTP, and the reduced Layout A memory fault no longer reproduces.
11. One-token MoE decode does not perform repeated full activation D2H transfers outside explicitly requested diagnostics.

## Open Questions

1. Do we ever need to support multiple CPU graph participants inside one MPI rank, or should that remain explicitly unsupported?
2. Should sparse dispatch initially support only continuation-to-expert and expert-to-continuation flows, or should it start as a general sparse all-to-all?
3. How much of the existing `MoEOverlayDispatchCollective` can be kept once payload transport becomes data-plane rather than request-plane?
4. Should `SparsePayloadId` be MoE-specific for the first implementation, or should it be a general arena concept available to future sparse attention / sequence compaction stages?
