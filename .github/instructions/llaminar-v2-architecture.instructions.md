# Llaminar V2 Architecture

This document maps the production architecture implemented under `src/v2/`.
It is intentionally a map of ownership boundaries and execution flow, not an
API catalogue. When this document and the code disagree, the code plus the
non-negotiable maxims in `AGENTS.md` are authoritative; update this document in
the same change.

## 1. Architectural shape

Llaminar V2 is a kernel-centric inference engine. Model code declares a graph
of typed compute stages, orchestration assigns each rank and device a portion
of that graph, and backend kernels execute the resolved graph. There is no
production operator layer between a stage and its kernel interface.

The control flow is:

```text
Main / subcommand
  -> AppLifecycle + RuntimeInitPhase
  -> IOrchestrationRunner
       -> OrchestrationRunner                         ordinary path
       -> NamedDomainGlobalRunner                    cross-rank named domains
  -> ExecutionPlanBuilder -> RankExecutionPlan
  -> runner construction
       -> GlobalOrchestrator                         cross-rank PP / global TP
          -> RankOrchestrator                        rank-local TP or local PP
             -> DeviceGraphOrchestrator              one device
  -> ForwardExecutionEngine
  -> model IGraphBuilder + GraphSchema
  -> ComputeGraph of IComputeStage nodes
  -> DeviceGraphExecutor + DeviceGraphCaptureController
  -> KernelFactory interfaces
  -> CPU / CUDA / ROCm backend kernels
```

The data plane follows a stricter path:

```text
model weights -> prepared weight stores -> stage bindings
request state -> BufferArena / KV cache / persistent device state
stage inputs  -> exact device + exact stream
stage output  -> producer event publication
consumer      -> waits on the published event
terminal data -> compact response result
```

The graph is participant-local. Cross-device or cross-rank movement is an
explicit collective or pipeline transfer. A compute stage must not conceal a
nested multi-device graph.

## 2. Application and configuration

### 2.1 Application lifecycle

`src/v2/Main.cpp` enters `app/AppLifecycle`. `SubcommandRouter` selects the
user-facing command (`oneshot`, `benchmark`, `serve`, `plan`, or `describe`),
while `RuntimeInitPhase` owns common runtime initialization. MPI bootstrap,
device discovery, topology inventory, configuration validation, and runner
creation happen before an execution mode starts inference.

The executable normally self-bootstraps MPI. `MPIBootstrapPhase` establishes
rank placement and the CPU/OpenMP environment used by production execution.
Before inventory or the already-in-MPI early return, it derives device intent
from the parsed typed declarations and installs CPU-only intent in the existing
backend startup authority. Named CPU domains must not initialize unused GPU
contexts. Both dense and routed accelerator endpoints prevent CPU-only
classification; rank count and collective backend do not imply a compute type.
Direct `--no-mpi-bootstrap` launches are reserved for tools that must attach to
the actual process, such as a debugger or backend profiler.

### 2.2 User configuration

`config/OrchestrationConfigParser` parses CLI or YAML into
`OrchestrationConfig`. `ConfigValidator` owns cross-option rules and detects
these selection forms:

- no explicit selection (automatic placement);
- one explicit device;
- an explicit rank-to-device map;
- simple tensor parallelism;
- an explicit tensor-parallel device list;
- named execution domains and pipeline stages;
- a parsed parallelism topology tree.

Do not duplicate the full flag matrix in architecture documents. The parser,
`CliSpec`, generated `--help`, and `ConfigValidator::createStandard()` are the
sources of truth.

`execution/config/RuntimeConfig` contains already-parsed runtime policy such as
activation precision, KV-cache precision, graph row capacity, prefix-cache
configuration, MTP configuration, and routed-MoE policy. It is copied into the
rank plan and then into `InferenceRunnerConfig`; downstream execution should
not reparse user strings.

MTP automatic thresholds retain explicit/automatic intent until admission.
`ExecutionPlanBuilder` selects an immutable `MTPDepthDefaultsProfile` from all
participants of the continuation domain, using the gathered card inventory and
exact rank ownership. A heterogeneous expert tier does not change that domain's
profile; a mixed or uncharacterized continuation uses the portable defaults.
The retained runtime owns the hardware profile, while `MTPRequestPolicy` owns
only request intent. `MTPDeviceGenerationPolicy` seals their effective values
into the existing integer device ABI; no host-side depth controller or hardware
query is introduced into GPU inference.

### 2.3 Topology status

Named domains are the installed interface for cross-rank PP and global or
NodeTP. `ExecutionDomainDefinition` describes device/rank membership,
scope, collective backend, and optional proportional work. `PPStageDefinition`
assigns a layer interval to a domain.

The parallelism-tree parser and validator exist, and a parsed tree is stored in
`OrchestrationConfig`. Runner creation does not currently compile that tree:
`OrchestrationRunnerFactory` logs it and continues through the standard plan
path. Treat topology-tree execution as incomplete until the factory constructs
the runner produced by `TreeToRunnerCompiler` after model metadata is known.

Likewise, simple non-named global PP/TP is detected but not installed. The
factory emits a diagnostic directing callers to named domains. Documentation
and capability reporting must not present either path as production-ready.

## 3. Planning and runner ownership

### 3.1 RankExecutionPlan

`execution/mpi_orchestration/ExecutionPlanBuilder` deterministically converts
the user configuration, model metadata, and `ClusterInventory` into one
`RankExecutionPlan` per MPI rank. That plan is the boundary between cluster
placement and rank-local execution. It contains:

- rank, host, and NUMA identity;
- the pipeline stage and owned layer interval;
- embedding and LM-head ownership;
- local and global TP participation;
- concrete devices, proportional work, and collective backends;
- weight-shard identity;
- parsed runtime configuration.

Rank-local code should execute this plan without rediscovering cluster-wide
placement. `ExecutionPlanBuilder` has separate resolution paths for named
domains and simple local plans; global named-domain execution additionally
builds `GlobalPPTopology` and a rank-specific `GlobalPPRankPlan`.

### 3.2 Public orchestration runner

`execution/runner/IOrchestrationRunner` is the application-facing lifecycle:
initialize, prefill, decode, request-batched decode, generation, cache reset,
snapshot access, and execution statistics.

The ordinary implementation, `OrchestrationRunner`, owns one rank's model
context, tokenizer, plan, local collective contexts, `IInferenceRunner`, prefix
cache policy, MTP controller, sampling policy, and request lifecycle. Its
initialization order is:

1. acquire MPI context and build the rank plan;
2. create local TP/PP contexts;
3. load and shard weights;
4. freeze model-aware routed-MoE placement;
5. validate TP/PP, context length, and the memory plan;
6. construct the selected inference runner and its graphs;
7. install GPU-resident logits and request-state policy.

`NamedDomainGlobalRunner` is selected when named PP domains span ranks. It
builds global topology and communicator state, stage-local runners, a
`GlobalOrchestrator`, and an adapter implementing `IOrchestrationRunner`.

### 3.3 Three execution tiers

`IInferenceRunner` is the common rank/device execution contract.

| Tier | Owner | Responsibility |
|---|---|---|
| Device | `DeviceGraphOrchestrator` | Model graph construction, arena and persistent state, graph caching/capture, one device's kernels |
| Rank | `RankOrchestrator` | One child runner per local participant, local TP or local PP coordination, logits and MTP coordination |
| Global | `GlobalOrchestrator` | Rank-specific global PP actions, cross-rank activation transfers, global TP domains, tail sampling/token coordination |

`InferenceRunnerFactory` constructs a `DeviceGraphOrchestrator` for a single
participant and a `RankOrchestrator` when a local domain has multiple devices
or pipeline stages. `StageRunnerFactory` applies the same rule for every local
stage owned by a `GlobalOrchestrator` and keeps each stage's collective context
and prepared-weight store alive with its runner.

`MultiDomainOrchestrator`, `ModelExecutor`, and
`HeterogeneousLayerExecutor` remain in the tree, but they are not the primary
production entry path. `LayerExecutor` and `ILayerExecutor` are compatibility
aliases for `DeviceGraphExecutor` and `IGraphExecutor`. New architecture work
should use the three tiers above rather than extending those older surfaces.

## 4. Declarative model graphs

### 4.1 Registration and model-specific policy

`models/ModelRegistrations.cpp` is the registration point for built-in model
families. It currently registers graph builders and schema factories for
`qwen2`, `qwen3`, `qwen35`, and `qwen35moe`.

`GraphBuilderRegistry` maps the model architecture string to an
`IGraphBuilder`. `SchemaFactoryRegistry` provides the matching `GraphSchema`,
weight-sharding rules, snapshot sharding rules, recommended sampling policy,
and model-specific metadata. Generic orchestration code must use these
registries instead of including model headers or switching on architecture
names.

The Qwen family shares reusable machinery in `models/qwen/QwenGraphBase`.
`QwenStandardGraph`, `Qwen35Graph`, and `Qwen35MoEGraph` specialize layer
composition. Qwen3.5 adds Gated DeltaNet/short-convolution stages; the MoE
variant replaces dense FFN work with explicit routing, dispatch, expert
compute, and return/reduction stages.

### 4.2 Schema and resolution

`GraphSchema` declares:

- stage templates and their tensor references;
- buffer semantics, types, shapes, and alias groups;
- TP behavior and weight-sharding patterns;
- stage-output sharding metadata used by diagnostics.

`GraphResolver` combines a schema with concrete sequence geometry and graph
configuration. The resulting descriptors are used to allocate buffers and
bind stages. Model graph builders then produce a `ComputeGraph` for a complete
forward role: full forward, partial PP stage, MTP sidecar, all-position
verifier, or a maintenance graph.

Model files declare placement and communication through the graph interfaces.
They do not directly run other ranks, synchronize devices, or move tensors.

### 4.3 ComputeGraph and stages

`ComputeGraph` is a DAG of named `ComputeNode` objects. A node owns one
`IComputeStage` and dependency names. The graph records a terminal node and can
precompute a fast schedule after validation.

An `IComputeStage` declares its:

- stage type and target `DeviceId`;
- input/output/workspace contract;
- prepared-weight requirements;
- backend support and graph-capture properties;
- dynamic replay parameters;
- exact GPU stream once bound.

Compute stages are thin orchestration units. They validate/bind tensors and
invoke interfaces such as `ITensorGemm`, `ITensorAttention`, `IKVCache`, or
backend-specific primitives obtained through `KernelFactory`. Collective and
pipeline operations are stages or explicit global-plan actions, not hidden
side effects of a GEMM or attention stage.

Important stage families live under
`execution/compute_stages/stages/`:

- embedding, normalization, QKV/GEMM, RoPE, attention, residual, FFN, LM head;
- KV-cache append/gather and GDN state localization/all-gather;
- TP allreduce/allgather and local/global pipeline transfers;
- MoE routing, sparse dispatch, expert execution, return reduction, and
  rebalance maintenance;
- MTP target/draft publication, verifier outcome, and speculative-state
  publication.

## 5. Device graph execution

### 5.1 DeviceGraphOrchestrator

`DeviceGraphOrchestrator` is the owner of a participant's executable model
state. It owns or binds:

- the model graph builder and graph configuration;
- `BufferArena`, activation buffers, prepared weights, and device workspace;
- device context and explicit streams/events;
- main, MTP, and shifted KV caches plus GDN live state;
- prefix-cache storage and request checkpoints;
- forward graph caches and `DeviceGraphExecutor`;
- device-side logits, sampling, MTP verifier, and response-state buffers.

It implements `IForwardExecutionHost`, letting `ForwardExecutionEngine` manage
generic graph lookup and replay while the orchestrator supplies model graph
construction, contexts, workspace, live-state preparation/publication, and
terminal result handling.

### 5.2 ForwardExecutionEngine

`ForwardExecutionEngine` owns forward dispatch and cache selection. A
`ForwardGraphSignature` distinguishes graph role, phase, geometry, policy, and
state that affects embedded pointers or launch behavior. Separate cache state
covers ordinary decode, MTP and verifier roles, and bounded/bucketed prefill.

On a cache miss the engine asks the orchestrator to build and prepare a graph,
executes warmup/capture as required, and installs the cache entry. On a hit it
updates permitted dynamic parameters, joins required ordering edges, and
replays the proven graph. Request reset clears request data and replay-session
state without discarding valid captured topology.

### 5.3 DeviceGraphExecutor and capture

`DeviceGraphExecutor` validates the graph, binds stage contracts through the
arena, verifies prepared weights and pointers, and executes the fast schedule.
`StageRunPolicy` selects the explicitly supported behavior for full execution,
fast decode, capture, or debugging; it is not permission to repair missing
state.

For GPU graphs, `DeviceGraphCaptureController` owns warmup, capture-stream
ordering, captured executable state, replay timing events, and any declared
manual segment boundary. A homogeneous GPU production forward must resolve to
one complete captured graph. Segmentation is reserved for a declared boundary
that cannot be represented in one native graph; it must never become an eager
replay fallback.

The executor also owns structured `GraphExecutorStats`, stage timelines,
snapshot callbacks, cancellation, and stage-failure publication. Diagnostics
may materialize tensors on the host, but that behavior is outside the normal
inference path.

Serving logs consume `IOrchestrationRunner::requestRuntimeSummary()`: a value
projected from existing request outcomes and the validated terminal ledger.
Obtaining it must not inspect child runners, query device state, transfer data,
or join maintenance. `prefixStateProbe()` is a separate explicit diagnostic
that may synchronize live caches; ordinary HTTP/SSE logging must never invoke
it, even when the corresponding log level is enabled.

## 6. Memory, coherence, and weights

### 6.1 Activation and workspace ownership

`BufferArena` is the activation/scratch/workspace authority for a device graph.
Schemas and graph setup register buffers by `BufferId`, including explicit
aliases and externally owned bindings. The arena allocates stable storage
before capture and gives the executor typed `StageBoundBuffers` views.

`StageBufferContract` describes every buffer access and prepared-weight
binding. A stage must not obtain an unrelated raw tensor and silently bypass
that contract. Concurrent graphs receive exclusive regions; graphs proven not
to overlap may share capacity according to the memory plan.

`planning/MemoryPlanner` estimates model, cache, activation, graph, and runtime
state before runner construction. Initialization fails when the selected plan
does not fit; allocation is not deferred to the hot path.

`PhysicalMemoryAuthority` owns the canonical physical admission, reservation,
materialization, and release ledger. Planners contribute typed BOMs, not live
capacity arithmetic. Opaque native graph pools are admitted as complete retained
families through `GPUGraphMemoryContract`; an individual pool-growth observation
is diagnostic evidence, not bytes owned by the graph that triggered it.

`MTPGraphOwnerPlan` supplies the same general/bounded-helper owner partition to
ordinary and ExpertOverlay admission. `ComputeGraph` declares the executable
memory class as captured topology. Before native instantiation, CUDA/HIP inspect
the helper's actual node count and kinds; nested/control graphs cannot claim a
small flat-helper reservation. Request reset preserves this class and graph
identity. The guard adds no replay-time queries or ordering edges.

### 6.2 Weight lifecycle

`ModelLoader` and `ModelContext` own GGUF metadata and raw model tensors.
`WeightManager` applies schema-provided sharding and placement. Backend-specific
packing or conversion is completed before executable graphs depend on it, and
`PreparedWeightStore` gives stages stable typed handles. Global PP keeps a
separate prepared store per local stage so constructing another stage cannot
replace live bindings.

MoE expert overlays add explicit owner maps, preparation/runtime plans, payload
providers, device slot pools, and transfer services. Rebalance changes are
published through graph-visible maintenance/state rather than by rewriting
ordinary stage ownership behind the graph.

The device overlay controller separates the open transaction's phase intent
from its last sealed command. Opening snapshot N+1 must retain command N and
its publication word: an empty decision can finish before a remote transport
worker acquires it. Every worker consumes N before joining snapshot N+1, so the
existing topology-wide snapshot fan-in is the sole command-buffer reuse edge.
Do not add a second acknowledgement, clear the command at transaction-open,
or reconstruct a missed command from host-side policy state.

### 6.3 Coherence and transfers

`TransferEngine` is the public authority for tensor movement and coherence
publication. Placement owners use its preparation APIs before execution;
stages require already-resident inputs, prepare output storage, and publish the
exact producer stream/event through `StageGPUExecution`.

The executor may join a consumer stream to a published event. It may not
allocate, upload, download, migrate, or guess a stream to repair a stage input
while a graph is running. CPU consumers explicitly materialize host data;
cross-device and cross-vendor transfers use a declared transfer plan.

Background mapped expert copies use prepared `PersistentTransferExecutionLane`
leases and `TransferEngine::enqueueBackgroundMappedCopy`. Its progress contract
is symmetric; native mechanisms differ: CUDA uses bounded byte-copy kernels,
whereas HIP explicitly requests no-compute DMA on registered device-visible
aliases; an ordinary small HIP copy may silently use a compute blit. Exact host
addresses and device aliases are retained for both directions; no backend
retries another mechanism after failure. Function preparation belongs to lane setup: lazy module resolution
must not synchronize a peer-held graph during submission. CUDA DMA queues can
serialize otherwise independent streams behind a captured copy; HIP compute
queues can do the same behind a held kernel. The backend owns this native
distinction. Native-event completion remains the contract for those direct
stream operations.

`MappedTransferProgressEpoch` separates permanent topology slots from a bounded
physical execution inbox. On CUDA, independently queued kernels can also be
starved by future inference-event consumers. The epoch therefore supplies an
`IGraphCaptureAuxiliaryBranch` to each retained inference graph: a private GPU
word opens at the root and closes at the inference terminal. The copy worker
retires at a bounded byte quantum before the graph joins it; it never waits for
a whole maintenance command or a host acknowledgment. Device-only claims and
partial byte cursors survive intervals and are shared with finite idle passes.
A host enqueue cannot claim GPU execution. Host maintenance owns immutable IO
commands and acquires exact generation receipts, not a shadow of GPU cursors.
Graph caches retain private interval words, graph-only branch sources and the
shared epoch authority.
`IForwardExecutionHost` supplies that device-lifetime authority for both ordinary
forward and hosted retained MTP replay. `ForwardExecutionEngine` resolves it
directly on every submission; a hosted caller cannot supply a second factory
or erase the captured branch with an empty descriptor. The cache still rejects
changed/removed authority rather than recapturing or trusting its old owner.
Resource construction and the small Open/worker/Close graph-only recordings run
on the device-context worker. One typed attachment decorates the final sealed,
uninstantiated native graph: direct capture and retained CPU-ticket composition
use the same operation. The original body is preserved in place, including
CUDA conditional-handle ownership. Open precedes every original root; every
original terminal precedes Close; only the final join awaits both Close and the
worker. The worker therefore progresses throughout external CPU waits, without
per-child branches, paired capture-event state or per-replay host submissions.
ROCm retains the independently progressing native SDMA implementation. Model
teardown joins the final exact idle/setup event before releasing service storage;
there is no model-lifetime persistent kernel that can obstruct reclamation.
CUDA contiguous CPU expert weights and remote raw GPU blobs bind to that same
epoch through `BackgroundTransferProgressBinding`; their exact generation
receipt owns completion. Native conversion/copy lanes and HIP use
`enqueueBackgroundStagingCopy` and exact native events. Their
`PersistentTransferStagingSlice` retains
one exclusive region of a shared mapped host slab plus its device scratch slab.
No client sharing the maintenance stream pool may bypass native background
submission with an ordinary DMA copy: that would reintroduce a physical queue
dependency ahead of every otherwise-independent client on the same stream.
Repack kernels keep their exact stream and terminal event; only TransferEngine
selects the physical copy mechanism, independently of expert tensor format.

CPU canonical-route ingress is a captured acquire/materialize/acknowledge DAG.
One thread acquires the CPU publication, then parallel row tiles copy its
immutable contributions, and the exact-stream terminal acknowledges reuse.
Never park a payload-sized grid awaiting a CPU publication: independent
maintenance and its event markers must remain runnable while the producer works.
Route metadata is shared per row rather than re-read from mapped host memory
for each contribution element. CPU ownership lasts until publication; the
ticket and payload remain immutable until the GPU acknowledgement.

## 7. Parallel execution

### 7.1 Local tensor parallelism

`RankOrchestrator` owns one `IInferenceRunner` child per local TP participant
and an `ILocalTPContext`. Each child graph is symmetric and participant-local.
Weights are sharded according to schema policy; `TPAllreduceStage`,
`AllGatherStage`, `AllGatherVStage`, and fused collective stages represent the
cross-participant edges.

The rank layer coordinates terminal logits, sampling, KV/GDN state, prefix
operations, MTP verifier inputs/outcomes, and failure cancellation without
turning one child into the hidden authority for all other children.

### 7.2 Local pipeline parallelism

Local PP also uses `RankOrchestrator`, configured with disjoint stage runners
and an `ILocalPPContext`. `FactoryPPStageConfig` records layer range plus
embedding/LM-head ownership. Activation movement appears as local pipeline
transfer stages. Local TP may be nested inside one PP stage through that
stage's own rank-local runner.

### 7.3 Global pipeline and tensor parallelism

For installed cross-rank named-domain execution:

1. `ExecutionPlanBuilder` builds `GlobalPPTopology`.
2. `GlobalPPRankPlanBuilder` derives this rank's ordered stage and transfer
   actions.
3. `DomainCommunicatorRegistry` creates domain-specific global TP contexts.
4. `StageRunnerFactory` constructs each stage's single-device or local-TP
   runner and stage-local prepared weights.
5. `GlobalOrchestrator` executes local actions, activation transfers, global
   collectives, and tail-token coordination.

Every MPI rank has a `GlobalOrchestrator`, but only executes actions present in
its rank plan. Global TP stages use `IGlobalTPContext`; PP transfers carry
activations between stage owners. No rank-local compute graph contains another
rank's graph.

## 8. Request state, prefix cache, and MTP

`OrchestrationRunner` owns the high-level request transaction; the concrete
runner owns tensor state. Prefill may use bounded graph buckets and chunk
schedules. Decode uses resident logits and, on GPU, device-side sampling so the
ordinary host boundary is a compact token result.

Prefix caching is integrated with live KV/GDN state. Lookup, restore, truncate,
harvest, promotion/demotion, and device rehydration are explicit lifecycle
operations. Cache fingerprints include model/graph policy needed to reject an
incompatible state image.

MTP is not a separate eager model loop. The main graph, sidecar drafts,
all-position verifier, stochastic/greedy outcome graphs, shifted caches,
request-batched state, and accepted-state publication share typed transaction
objects and device events. `MTPDepthController` selects a permitted draft depth;
`MTPVerifierForwardExecutor` and the runner interfaces coordinate verifier
forwards. Production grouped verification must preserve serial-row byte
equivalence while publishing accepted state without row replay.

Request reset uses typed reset transactions and KV-cache reset boundaries. It
joins outstanding device work, resets data and live-state generations, then
publishes readiness for the next request. It does not rebuild topology solely
to clear request state.

## 9. Kernels and backends

`KernelFactory` is the model/stage-facing dispatch point. Kernel interfaces are
typed by operation (`ITensorGemm`, attention, norms, RoPE, KV cache, sampling,
GDN, MoE, and collectives). CPU, CUDA, and ROCm implementations live below
`kernels/<backend>/`; backend runtime services live below `backends/`.

GPU kernels receive an exact non-null stream from their stage. Persistent
workspace and pointer tables are allocated and bound before capture. CPU
kernels use runtime ISA dispatch where applicable and must remain valid for
every positive worker count. Backend-specific profiling and tuning procedure
lives in the project skills under `.agents/` rather than in this architecture
map.

## 10. Observability and tests

`PerfStatsCollector`, `GraphExecutorStats`, and stage timelines are the normal
path evidence. They report graph capture/replay, transfer and collective
counts, prefix/MTP decisions, routing, and selected kernel paths. Profiler
overhead is kept outside canonical timing samples.

Snapshot capture, stage dumps, stage-output printing, and tensor verification
are diagnostic features configured through `utils/DebugEnv.h`. They may add
host copies or synchronization and therefore cannot certify production
residency or performance.

The test hierarchy mirrors ownership:

- unit tests cover policy, parsers, state machines, graph contracts, and source
  architecture rules without requiring devices;
- CUDA/ROCm integration tests exercise real capture, streams, kernels, and
  collectives;
- parity suites compare model-stage outputs to the reference implementation;
- performance targets isolate concrete kernels and representative geometry;
- static source tests reject default streams, blocking synchronization, direct
  coherence transitions, and other forbidden structure.

## 11. Extension points

### Add a model family

1. Implement an `ISchemaFactory` and `IGraphBuilder` under `models/<family>/`.
2. Keep reusable graph wiring in a base builder; keep the model file
   declarative.
3. Register both factories in `models/ModelRegistrations.cpp`.
4. Add schema, graph-construction, parity, backend, and integration coverage.

### Add a compute stage

1. Implement `IComputeStage` with a complete buffer/weight contract.
2. Give it an explicit device and exact GPU stream behavior.
3. Add construction through `ComputeStageFactory` or the relevant graph
   builder.
4. Prove shape, coherence, capture, reset, and backend behavior with focused
   tests.

### Add or tune a kernel

1. Extend the typed kernel interface only when the operation contract changes.
2. Register backend dispatch through `KernelFactory`.
3. Preserve exact arithmetic and backend symmetry where supported.
4. Use the matching CPU, CUDA, ROCm, or NativeVNNI tuning skill for correctness
   and economy gates.

## 12. Source map

| Concern | Primary source |
|---|---|
| Application lifecycle | `src/v2/app/`, `src/v2/Main.cpp` |
| CLI and validation | `src/v2/config/OrchestrationConfigParser.*`, `ConfigValidator.*`, `CliSpec.*` |
| Rank planning | `src/v2/execution/mpi_orchestration/` |
| Public inference lifecycle | `src/v2/execution/runner/` |
| Global PP/TP | `src/v2/execution/global/`, `global_pp/` |
| Per-rank orchestration | `src/v2/execution/local_execution/orchestrators/RankOrchestrator.*` |
| Per-device orchestration | `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.*` |
| Forward cache and dispatch | `src/v2/execution/local_execution/engine/` |
| Graph representation/execution | `src/v2/execution/local_execution/graph/` |
| Stages | `src/v2/execution/compute_stages/` |
| Model graph declarations | `src/v2/models/` |
| Activation memory | `src/v2/memory/`, `src/v2/planning/` |
| Weight lifecycle | `src/v2/loaders/` |
| Coherence and movement | `src/v2/transfer/` |
| Prefix cache / MTP / MoE | `src/v2/execution/prefix_cache/`, `mtp/`, `moe/` |
| Kernel dispatch and implementations | `src/v2/kernels/` |
| Runtime backends | `src/v2/backends/` |
| Runtime diagnostics | `src/v2/utils/DebugEnv.h`, `src/v2/execution/debug/` |
| Tests and registered CTest names | `tests/v2/CMakeLists.txt`, `tests/v2/` |
