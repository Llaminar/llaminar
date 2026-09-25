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
       -> OrchestrationRunner                         one initialization owner
  -> ResolvedRankOrchestration -> ExecutionPlanBuilder -> RankExecutionPlan
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

Hostfile launches delegate slot admission and per-host socket mapping to MPI,
without applying the initiating node's CPU set or worker count to other hosts.
Each child derives its physical worker budget from its actual binding before
inventory discovery. The hostfile remains in child configuration as launch
provenance. `NodeDetection` derives physical node identity from MPI shared-memory
membership, never from hostfile aliases or processor-name equality.
Serving's CPU shorthand likewise resolves to the observed NUMA endpoint;
it never equates local MPI rank order with physical CPU locality.
Automatic serving requests host-wide accelerator visibility when its hard
constraints permit GPUs: one hostfile slot must not hide devices on another
socket. CPU-only hard filters enter the existing CPU-only startup authority
before enumeration; a preference alone does not exclude other backends.

`DeviceManager` publishes the `HardwareInventory` observation even during quiet
startup. `ClusterInventoryGatherer` projects and exchanges those observations;
requested TP devices cannot replace them or renumber their backend ordinals.
`MPITopology` consumes that same gatherer and derives `RankPlacement` from its
installed snapshot. Its inventory accessor neither discovers hardware nor
enters a collective. Hostname aliases are labels, CPU NUMA ownership follows
actual affinity, and P2P matrix indices are explicitly mapped to device ordinals.
CUDA and ROCm discovery preserve driver UUIDs. Node summaries union repeated
UUID/NUMA observations rather than summing rank visibility; rebuilding a summary
is idempotent. These physical facts are not an allocation ledger. Collective
membership-only records carry no observed capacity and cannot price a candidate.

CPU physical capacity and execution workshare are different inventory facts.
`RankInventory::cpu_cores` describes physical cores in the observed locality;
`cpu_worker_threads` records the fixed OpenMP budget after startup policy.
Both `plan` and `serve` use `MPIBootstrapPhase::configureRequestedCPUThreads`
before discovery, preserving a bootstrap-selected team when no override exists.
The inventory wire preserves each rank's own budget. Measurement authentication
and ordinary/ExpertOverlay CPU workspace BOM inputs use `cpuWorkerThreads()`,
which rejects a missing observation instead of substituting physical cores.
GPU SM/CU counts and node-level physical-core aggregation remain hardware facts.

Physical-machine membership and MPI process membership are distinct facts.
`ClusterInventory::connectionBetweenRanks` projects a checked
`RankConnectionTopology` with both endpoint ranks, both node IDs and a typed
same-rank / same-node / cross-node classification. `MPITopology::same_node`
delegates to that authority and rejects absent ranks. Latency, bandwidth,
hostname spelling, GPU ordinal and NUMA index never establish locality. Node
labels are scoped to their inventory; selected execution ranks use the selected
inventory, whose projection preserves physical relationships after renumbering.
Domain-based TP context creation also requires the canonical resolved execution
scope. It cannot choose rank-local versus cross-rank construction from a UPI/MPI
backend label or differing hostname strings. A cross-rank context may still be
entirely node-local; its actual communicator membership establishes that fact.

Production pipeline construction binds each activation edge to the same
immutable `RankConnectionTopology`. Both send and receive actions retain its
original direction and physical IDs. Their checked accessors reject missing or
edited endpoint identity; structural unit-test plans cannot supply transport
membership. GlobalPP owns no parallel hostname/locality table and does not
reclassify connections from measured bandwidth.

Cluster publication uses the complete driver-visible hardware observation,
not DeviceManager's socket-local execution view. CPU affinity binds the CPU
endpoint but never removes GPUs on another socket; the planner uses GPU NUMA
identity to prefer an owner later. This applies equally to plan, describe and
serve, including hostfiles with just one discovery rank on a multi-socket host.
Projecting another view reuses that observation without driver rediscovery.

Full-backend process loading does not require an NVIDIA driver on CPU-only
cluster members. `CUDADriverApi` owns one immutable table of exact SDK-declared
Driver API symbols. CUDA preparation binds it before graph recording; missing
symbols fail that preparation. CPU/ROCm discovery does not bind the table.
The native library stays pinned through graph and runtime-context retirement;
there is no model-owned loader, alternate ABI, toolkit-stub deployment or
per-replay resolution. CUDA still uses its native graph queries, stream memory
operations and context checks. Driver-free startup and native binding belong
to the canonical model-free integration preflight.

Each real `MPIContext` publishes that observation once for its exact communicator.
Startup, runner admission and topology share the immutable publication; topology
does not own a refresh API. Failed discovery and failed topology construction
retain their original exceptions instead of retrying a partially entered MPI
protocol. Actual communicator membership sizes the exchange, so a malformed
wrapper cannot masquerade as local discovery. Separate/reordered communicators
own separate rank namespaces; live memory availability remains PMA admission
input, never a refresh of this startup hardware snapshot.

`ExecutionRankMembership` is the immutable discovery-to-execution mapping.
`MPIContextFactory::selectRanks` collectively authenticates the ordered subset
before splitting, returning an owned active context or a typed inactive rank.
Selected contexts retain their parent and project its inventory without a new
hardware exchange. Derived topology retires before its owned communicator.
`ExecutionRankSelection` is the validated ordered value retained by config
documents and the membership projector. Candidate admission seals it together
with the enclosing discovery process count. The public frontend authenticates
that selection before runner construction; excluded ranks return completed and
retire their process session while the selected communicator continues. CPU
maps and accelerator visibility use execution indices, not discovery indices.
Saved local plans use the same unnarrowed discovery-launch contract as hostfiles;
an explicit selected CPU pin cannot shrink or rebind the discovery namespace.
The runner's explicit-context factory and constructor preserve that communicator through
initialization consensus and collective construction; it cannot substitute
WORLD or SELF. Automatic startup publishes root's complete selected apply
document before entering this same admission boundary. Runner factories accept
only apply intent; they cannot search or change their caller's membership.

`MPIProcessSession` is the frontend's move-only process-lifetime owner.
`AppContext` declares it before dependent fields so it finalizes MPI only after
request-mode adapters, runners and context owners have retired. Command-only
discovery uses the same owner. Context factories keep weak lookups, never
process-static strong ownership of derived communicators. Execution modes own
request termination, not MPI finalization. Startup returns a typed ready,
completed or failed result; configuration flags are not an exit-status channel.

GlobalTP membership IDs refer to the supplied parent communicator. Its local
CPU endpoint is the observed hostname/NUMA address supplied at creation, never
CPU zero or a rank-derived hostname. Domain compilation preserves endpoint,
parent rank and proportional work as one tuple when sorting participants.
The same tuple supplies named global-stage construction and rank-local plans.

`RankHardwareOwnership` is the shared CPU ownership query used by ordinary
placement and ExpertOverlay binding. Bound ranks own their observed NUMA
endpoint; whole-host observations require explicit physical-node detail.
Neither MPI local-rank order nor NUMA-node count establishes a physical ID.
GPU owner preference compares its observed locality with observed CPU affinity,
so reordered communicators and moved cards retain the correct physical owner.

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

`plan` and `serve` parse the same complete inference request. Plan adds only
output presentation options through checked `CliSpec` composition; an extension
cannot shadow a shared flag. MTP capacity, KV precision, prefix storage and
movement economics therefore enter planning exactly as they enter serving.
Command discovery borrows this same request for MPI bootstrap and backend
intent. The selected document is apply-only: consumers do not replay automatic
search constraints over it or silently change its memory policy afterward.
Use the shared `--only-strategies` and `--kv-cache-precision` names; plan has no
private strategy aliases or precision table.

`AutomaticHostParticipation` is a hard search constraint, separate from backend
filters and performance preferences. Omission permits a best host subset;
`--auto-hosts all` (YAML `planning.hosts: all`) requires every discovered physical
node to own a compute pool. The streaming candidate publisher checks this using
the authenticated inventory and selected membership before admission or ranking.
Multiple ranks on one host do not force all of those ranks to execute. Applying
a saved selection retains that exact membership, not a new automatic search.

`ExpertTierDefinition` is the compact declaration adapter for
`--expert-tier name=devices;priority=N`. It produces the existing routed-domain
and tier types, not another topology authority. Shared normalization derives
omitted continuation/shared roles from integer priority. Scope, collective,
ownership and physical capacity remain inventory/admission decisions; expanded
declarations and explicit overrides use the same contracts.

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
`OrchestrationRunnerFactory` rejects it before model admission. It must never
ignore the requested tree and select a different ordinary plan. Tree execution
remains incomplete until lowering uses the canonical model-aware admission.

Simple non-named global/NodeTP and PP use one ordinary `OrchestrationRunner`
per participating rank; the rank plan carries their cross-rank edges. They are
not rejected by the factory merely for using cross-rank communication. Named
domains additionally express per-stage device sets and TP-inside-PP composition.
Do not confuse either installed path with the incomplete topology-tree compiler.

## 3. Planning and runner ownership

### 3.1 RankExecutionPlan

`PlanningModelMetadata` carries the exact GGUF tensor inventory and the
learned-MTP-aware main-layer boundary. Ordinary runner admission exchanges this
descriptor from root; automatic startup retains one root-owned source and
publishes metadata before gathering evidence and selecting an apply document.
Fallible read, receive allocation
and decode phases use `RankInitializationLifecycle`; a missing GGUF is collectively
rejected before any follower waits for a payload. The ordinary and ExpertOverlay
BOM builders reuse that descriptor. Explicit pre-built model contexts project
their already parsed loader through the same descriptor type without reopening
the GGUF. This does not replace rank-local weight loading or tokenizer setup.

`PlanningPublication` is that shared root-only immutable-artifact transport.
Metadata and selected apply configurations use the same three-phase consensus
and authenticate distinct artifact identities before transferring payloads.
Native communicator membership is checked even when a wrapper claims one rank;
root and peers all decode and validate the same bytes before returning a value.
Selected configurations retain complete policy and ordered discovery selection,
but publication is not a hardware or physical-memory certificate. Both `plan`
and `RuntimeInitPhase` invoke `AutomaticPlanningStartup::run` on every discovery
rank. It publishes the complete automatic request, retains and publishes model
metadata, gathers evidence, authenticates evaluator readiness, then invokes the
pure selector only on root and publishes its apply document. Sample collectors
own their internal failure consensus; final local evaluator-construction errors
reach a common completion consensus before root starts searching. The shared
composition owns metadata lifetime, capture-memory policy and workload policy;
followers neither reopen the planning source nor evaluate candidates.
`gatherClusterInventory` and command sessions retain the same immutable shared
inventory as the MPI context, not detached copies that measurement admission
cannot authenticate. Serving admits the published rank selection
before constructing an apply-only runner. Its default preparation is
`PlanningRequestCostModel::prepare`: collect one source-family/FP32/streaming/link
basis on discovery membership, then price candidates without any more device
work. There is no topology-only constant-score implementation or private
frontend estimator.

`exchangePlanningSamples` scatters one root-described control envelope per
discovery rank and gathers completed local observations in authenticated rank
order. Preparation, measurement and validation failures share the existing
initialization consensus; callbacks cannot introduce nested collectives.
`PlanningMPITransferMeasurement` separately measures prepared host-MPI
request/reply traffic at exact sizes on a private communicator. Its payloads
require rank-bound PMA claims before allocation. Timed observations retain the
inventory's physical connection identity; they neither classify locality nor
claim one-way bandwidth, native GPU collective costs or remote compute service.
One excluded warmup and three completed exchanges bound sampling work. Untimed
pair rendezvous prevents uninvolved ranks from contaminating the next sample;
pending MPI failure uses the standard fatal 30-second transport deadline.

`PlanningModelSource` also owns explicit bounded native-weight sampling. A
`PlanningModelSampleRequest` selects an ordinary matrix, a source row/column
interval, or one routed expert. The retained GGUF directory validates exact
N/K, codebook alignment and byte extents before PMA admission or I/O. Loading
reuses `ModelLoader`; it does not map the whole model, create another loader,
or convert weight formats. Source storage and the temporary reader buffer are
separate typed BOM contributions. `PlanningLoadedModelSample` retains the
source/view/lease transaction and exposes a const tensor borrow for the
sampling operation, not a tensor pointer with independent lease ownership.
It must outlive any prepared kernel that borrows it. Role-specific runtime
representation, prepared weights, workspace, graph capture, and completed
timing remain subsequent explicit steps; source loading alone is no service
observation or whole-model throughput estimate.

`PlanningMatrixSamplePlan` seals an ordinary whole matrix, source-axis shard,
or individual expert selection for publication. Source coordinates remain part
of identity even when two shards have identical dimensions. Its loaded owner
keeps the plan and final PMA-backed payload together. Matrix and full-expert
publications share one strict native-format codec and one fixed-batch native
MPI completion lifecycle; adding an artifact has one registration/validation
decision. The matrix adapter changes neither the production GEMM numerical
policy nor its workspace contract. It supplies native bytes, not a surrogate
expert-service timing for dense execution.

Projection preparation derives its executed type from
`PreparedWeightRepresentationContract`, just as the model loader does. Native
publication bytes and runtime FP32 promotion remain distinct identities and
physical allocations. Promotion is admitted before conversion and retained
until all prepared-engine borrows retire. Observation receipts authenticate
both the native source and executed format; borrowing CPU floating kernels
still incur the converted representation's memory traffic even when they own
no separate packed weights.

`PlanningFP32ArithmeticPlan` is a separately typed, source-free arithmetic
proxy. It reuses projection preparation, PMA admission, native capture, timing
and retirement rather than maintaining another benchmark lifecycle. Its fixed
geometry cannot be relabeled as a model source, quantized GOPS, vendor peak, or
measured attention/GDN latency. Catalog receipts preserve that distinction and
bind the same physical observer and CPU workshare as streaming evidence.

`PlanningExpertSampleRequest` and `PlanningExpertSampleDescription` are the
shared backend-independent full-expert source contract. They reject partial,
mixed-layer, mixed-expert and incompatible projections before payload I/O.
`PlanningExpertSamplePlan` seals that source identity and native block geometry
into a control envelope. `PlanningExpertSamplePublication` first authenticates
the plan on all discovery ranks, then admits/allocates each follower's final
source tensors before posting the payload broadcasts. Root alone opens GGUF;
followers consume the published `PlanningLoadedExpertSample` without filesystem
access. There is no extra receive staging copy. All native requests complete
before source ownership is published; failure after posting is fatal under the
standard collective deadline, never unwinding a live receive buffer. A loaded
sample retains the same native source/claim lifetime on CPU and GPU. Source
origin is explicit in its BOM: only local GGUF reading needs reader staging.

`PlanningKernelServiceCatalog` collects those observations before root-only
candidate pricing. Reporter selection uses physical node, backend and GPU UUID,
deduplicating ordinal-remapped visibility and preferring observed NUMA affinity.
CPU observations remain qualified by discovery rank, observed NUMA scope and
the measured worker/ISA identity; a whole-host measurement is not a socket
measurement. Neither receipt contents nor elapsed time can assign rank or
physical-node membership. MPI supplies reporting rank; inventory supplies
physical identity. Incomplete, foreign-source or non-captured GPU evidence is
rejected as a batch rather than treated as a slow endpoint.

The collector authenticates the full assignment before PMA admission and native
source publication. Each physical node measures one reporting rank at a time;
different nodes may measure concurrently. Rank-local CPU/GPU samplers execute
sequentially and retire temporary owners between devices. Their complete BOMs
compose through `PhysicalMemoryPlanBuilder::addMutuallyExclusive`, which owns
the per-resource/per-owner admission envelope. Concurrent contributions remain
additive; residency credits cannot be merged by a maximum. One source triplet
stays live across that sequence. No sampler-local capacity arithmetic, live
allocation ledger, CPU worker pool or whole-model warmup is introduced.

Native quantized weight views report the encoded extent of their borrowed
shape, not the size of their empty owning vector. The common checked
`NativeTensorExtent` geometry covers every native codebook; FP views retain
their corresponding element-size contracts. These are logical tensor byte
extents, not independent allocation or capacity accounting.

`PlanningCPUExpertMeasurement` consumes one complete same-layer source expert
triplet and prepares it through `KernelFactory` with the ordinary GPU-aligned
expert arithmetic policy. It reuses
`MoEOverlayCPUServiceMeasurement::measurePrepared`, which also serves the
published-bank certifier: one complete production grouped FFN, private execution
bindings, explicit CPU workspace, one warmup and three measured invocations.
There is no second router, runner or synthetic full-model calibration. The
rank's existing fixed OpenMP/NUMA execution policy is inherited, not replaced by
a sampling pool. Source and prepared matrices overlap; serial reader/packing
scratch and phase workspaces reuse their maxima through the canonical PMA BOM.
`CPUWeightPreparationMemory` contributes the unrotated native packer's retained
payload and temporary allocations, not its smaller consumed cache footprint.
Floating prepared engines report their actual owned bytes; non-allocating
execution views report zero instead of counting the source twice.

`PlanningGPUExpertMeasurement` drives the admitted `LoadOrchestrator` and
`KernelFactory` pool binding, then captures the ordinary `MoEExpertComputeStage`
on the device's owning worker. CUDA scratch composition calls the same
quantized projection requirement implementation used by prepared kernels;
floating admission and runtime share the complete wrapper plus nested cuBLAS
named-buffer contract. Actual-card CUDA launch-policy queries stay on that
device's worker, never on rank zero using a remote ordinal. CPU source, upload
rings, native weights, execution buffers and the native graph family all belong
to PMA. Phase graphs and workspace bindings retire before their storage; one
family reservation spans their driver-pool observations. Timings use retained
graph replay and exact-stream native events, with one warmup and three timed
FFNs per phase. Functional observation tests use one process per device: two
independent observers of the same GPU would corrupt pool-growth attribution.

These CPU observations retain source tensor/expert identity, exact dimensions,
formats, worker count, compiled/runtime ISA and repeated-weight cache regime.
They do not claim streaming bandwidth, GPU service, remote transport cost or
whole-model latency. GPU observations carry the corresponding device, source,
format, shape and repeated-expert cache qualification; they are not transfer
bandwidth measurements. Their rank/node membership must come from the surrounding
authenticated discovery publication. Collecting these observations across
discovery ranks and connecting CPU/GPU service to automatic critical-path
selection remain separate responsibilities; having a sample producer does not
make the current startup topology score a measured cost model.

`PlanningWeightServiceModel` composes the authenticated source-kernel catalogs
with independent streaming-memory observations. One representative is selected
per ordinary executed format or complete expert format triplet, weighted by the
aggregate arithmetic of its source shape. Main-layer classification is shared
with `PlanningForwardWeightWork`; retained MTP sidecars do not multiply setup
work. All candidates reuse those observations. Exact CPU workshare/geometry and
physical endpoint identity must agree across component catalogs.

This is explicitly a bounded weight-work estimate: same-family shape scaling,
invocation-time interpolation between measured row counts, and a separate
streaming-traffic limit. Routed work uses admitted execution shares and declared
uniform top-k expectations for rows and nonempty expert groups; replication is
not an automatic speedup. It owns no allocation ledger and must not be presented
as complete request timing without the remaining execution and communication
costs. GPU native collectives, host MPI and node-local mapped activation channels
are different protocols; observations of one cannot silently price another.

`PlanningForwardStateWork` projects causal attention and equivalent recurrent
algebra from the compiled continuation role, TP heads, PP interval and actual
invocation. It excludes expert-only followers and retained MTP sidecar capacity.
Decode context can be analytically averaged over the requested horizon; spare
KV capacity never becomes traffic. `KVCacheMemoryEstimator::logicalPayload`
owns CPU/GPU native payload geometry, excluding allocator metadata, scratch and
linearization replicas. GDN work uses `HybridGDNStateGeometry` and local live
banks, not complete prefix-serialization banks. Pricing is explicitly a bounded
FP32/streaming roofline proxy with ideal intra-invocation reuse, not an exact
kernel trace or complete request cost. Dependent weights, non-state operations
and communication remain the request composer's responsibility.

`PlanningRequestCostModel` is that shared request composer. It uses admitted
row capacities and the common overlay prefill contract, compiled main-forward
weights/state, indexed embedding traffic, terminal-only vocabulary work, and
explicit scalar-operation proxies. Full prefill chunks are priced at their
mean preceding context plus one tail; decode uses the mean requested horizon.
Work on one endpoint adds, independent endpoint estimates join by maximum,
and dependent layers/PP boundaries add. No speculative communication overlap,
future expert migration gain, or MTP acceptance is credited without evidence.

Single-device candidates have zero interconnect demand. Ordinary TP replicas
are not sparse-overlay participants; only an admitted overlay uses its bound
logical root and live route packet ABI. Node-local mapped-byte service stays
distinct from remote MPI plus GPU DMA staging. Native groups price homogeneous
GPU reductions. Where the production protocol differs (point-to-point volume,
canonical-order reduction, or CPU collective), evidence explicitly labels the
primitive-service proxy; it is not measured model or collective latency. Missing
required evidence is fatal. This bounded model ranks predicted request latency,
not guaranteed fastest throughput, and never changes inference protocol or
precision to match its samples.

`PlanningCommunicationService` collects native collective, GPU/host and host-MPI evidence
through the existing failure-atomic discovery publication. Its sample basis is
physical, not per candidate: maximal homogeneous rank-visible GPU groups are
deduplicated by node/backend/UUID set, with one common affinity-preferred rank
observing each group. The communicator itself keeps that rank's production
ordinal order; UUID sorting is only a deduplication key, never a replacement
ring order. Receipts retain the actual ordered group, reporting rank, native
wire precision, payload and both completed graph phases. They do not certify
unmeasured smaller subsets.

Host-MPI observations remain directed, complete request/reply round trips at
control, outbound-heavy and return-heavy geometries. Vary one payload at a time:
symmetric payloads alone cannot identify asymmetric dispatch service. Physical
locality comes from inventory, not the measured speed, and round trips do not
imply symmetric one-way links.
An explicitly single-device search has no communication probes. Native, GPU/host
and MPI storage share a mutually exclusive PMA envelope; node-local reporter rounds
avoid self-contention. `planningObservedResource` supplies the same observed
allocator projection to compute and communication contributors without another
capacity query or live ledger.

`PlanningHostDeviceMeasurement` uses that same collection transaction for DMA
and mapped-kernel byte copies in both directions. Each observation binds the
physical GPU to the reporting rank's host first-touch scope; GPU affinity does
not relabel those pages. TransferEngine supplies the prepared exact-stream lane
and canonical mapped backing extent. Retained graph timing excludes setup,
fresh host publication and exact-byte readback verification. These are byte
primitives, not complete activation-packet/wait-kernel service; full request
dependency/cost composition remains a separate responsibility.

`PlanningCommunicationCost` is the pure query boundary over those immutable
receipts. `PlanningPayloadServiceCurve` interpolates complete invocation time
with a positive startup floor and positive effective bulk extrapolation; source
observations remain unchanged when a monotone envelope handles measurement
noise. A single validated GPU has no collective cost. Native aliases resolve
through physical node/backend/UUID, with exact order preferred. Smaller or
reordered groups retain a typed containing-group proxy without an invented
degree discount, never a claim of measured candidate performance or a guaranteed
upper bound. Missing protocol/precision/physical membership is fatal. Host MPI
composes separate outbound/return payload increments with one whole control RTT;
it never supplies mapped-GPU or DMA costs. GPU/host queries require their own
explicit mechanism, direction and same-node first-touch observation. A GPU alias
may change rank-local ordinal, but cannot substitute a different host-page scope
or cross a physical-machine boundary. Queries perform no device work, probe,
transport selection or live physical admission.

`OrchestrationPlanningWorkload` in the shared configuration policy owns the
positive prefill/generation horizon, context validation and context-bounded
default. CLI `--plan-workload` and YAML `planning.workload` produce the same
optional typed value. Selection checks that every cost names that exact
objective; saved apply configurations discard search hints. This expected
workload neither limits inference nor constitutes measured performance evidence.

`planning/ResolvedRankOrchestration` composes the existing rank compiler with
ExpertOverlay authority normalization and continuation/follower role projection.
It copies the request, resolves against observed inventory and real model
geometry, validates, then publishes configuration and rank plan together. It
does not discover hardware, allocate memory or reserve physical capacity.
Runtime startup adopts the runner's resolved config instead of rebinding an
overlay. Role diagnostics consume this same result. Automatic selection must
use this compiler rather than duplicate these rules in a frontend.

`AutomaticOrchestrationCandidates` supplies declarative proposals from the
observed inventory, preserving discovery membership separately from compact
execution ranks. It visits candidates without loading models or retaining
compiled Cartesian products. GPU subsets, CPU pools, main-layer pipeline splits
and ordered expert tiers are construction choices, not performance rankings.
Candidate enumeration grants neither graph support nor physical capacity; each
candidate still requires model-aware compilation and complete BOM/cost admission.
The current catalog's boundaries and unmeasured performance ranking are recorded
in the automatic-orchestration project plan, not advertised as execution or
fastest-placement certificates.

`OrchestrationCandidateAdmission` composes that proposal with the shared rank
compiler and ordinary/overlay BOM builders for every selected rank. A scoped
`PlanningModelSource` retains the root's metadata-only GGUF loader through the
search, supplying the existing MTP and routed-weight manifests without another
wire directory or reopening the model per candidate. Successful admission
retains the immutable physical certificate and exact graph/quota geometry; it
does not reserve live resources, prove execution, or claim a performance rank.
Applying a saved choice still requires fresh runtime admission. A whole-model
rank-local TP proposal uses ordinary TP intent, not a one-stage pipeline, so
the shared compiler can install MoE's sole ExpertOverlay authority.

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

`DeviceGraphOrchestrator` and `ForwardExecutionEngine` accept participant-local
full-model or PP-shard graphs, never an embedded multi-device pipeline. DGO
retains constructor-injected named TP contexts but cannot lazily manufacture
TP/PP domains or own another device's main KV cache. Rank/global orchestration
owns pipeline composition. Legacy graph-builder-only pipeline construction is
not a supported DGO input and is rejected before setup.

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

Named PP domains use the same `OrchestrationRunner` initialization, physical
memory admission, snapshot setup, and serving preparation as local execution.
`NamedDomainGraphBuilder` constructs the explicit `GlobalOrchestrator` and its
stage-local runners at the ordinary graph-build phase; it owns no parallel
model-loading or request lifecycle. The global graph's immutable
`requestAuthorityRank()` names its vocabulary-head domain leader. Before
readiness, the outer runner binds command, sampling and frontend ownership to
that same rank. A pipeline head cannot sample another rank's absent logits;
the tail is selected by topology, not by rank numbering. Local graphs leave
the enclosing plan's existing continuation authority unchanged.

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

Cache-backed attention always consumes the configured native post-append KV
cache, on CPU as well as GPU. The graph orders append before attention. Cold
prefill, restored suffixes, decode and grouped verification cannot switch to
transient FP32 projections based on row count or an independent read-mode flag.
An explicitly cacheless attention stage consumes its supplied K/V operands.
Missing publication is an error, not permission to change precision or source.

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

`IWorkspaceConsumer::workspaceBindingPolicy()` distinguishes capture-bound
engines from immutable CPU engines receiving invocation-owned scratch. Stages
retain and pass the latter's `DeviceWorkspaceManager`; binding a participant
must not mutate a shared prepared CPU kernel. CPU SwiGLU/down declares its
transform separately from a plain projection, using the same named-buffer
contract in metadata admission, graph stages, sparse followers and service
sampling. Missing or undersized scratch is fatal, not a request to grow a
thread-local buffer. The sampling stage retires before its admitted workspace.

`planning/MemoryPlanner` estimates model, cache, activation, graph, and runtime
state before runner construction. Initialization fails when the selected plan
does not fit; allocation is not deferred to the hot path.

`RankMemoryPlanInputs` assembles ordinary execution's complete per-device BOM
inputs from its compiled rank plan and canonical observation. Local TP binds
the same proportional slice as graph construction; PP preserves stage layer
intervals. MTP, prefix, diagnostics, upload rings and retained graph families
enter that shared boundary. Authenticated retained-weight/workspace credits
stay at the runner's existing lifetime boundary; frontends cannot invent them.
ExpertOverlay keeps its topology-wide BOM resolver and is not charged again by
the ordinary builder. `MoEOverlayMemoryPlanInputs` assembles its exact retained
main/MTP graph families, snapshot variants, upload policy, activation-channel
rows and distributed CPU histogram owners. Prefill segment rows cannot shrink
retained verifier capacity. Candidate assembly and admission run inside the
common typed rank-initialization consensus before physical budgets are
exchanged; a phase mismatch is fatal, not another rejected memory shape.
Neither input builder owns live admission arithmetic or chooses a topology.

KV admission requires both `KVCacheFamily` and storage precision. The main
hybrid cache retains its family through FA-only pipeline slices; shifted MTP
caches are independently attention-only. `KVCacheMemoryEstimator` supplies
their distinct ring and serialized-block BOMs to both planning and factory
allocation. Precision alone must never select those byte formulas: hybrid GPU
Q8 uses linear K/V blocks, while attention-only Q8 uses anchored keys. Native
cache layout versus whole-slot prefix admission is covered on both GPUs in
the model-free preflight suite.

`MTPStateRole` distinguishes disabled capacity, a main-model follower, and a
predictor owner. Cache construction and persistent-state admission consume the
same role. Disjoint PP followers retain rollback state for their own main
layers but never allocate shifted predictor caches; the terminal vocabulary
owner retains those caches. A same-layer TP replica is not a PP follower.

`PhysicalMemoryAuthority` owns the canonical physical admission, reservation,
materialization, and release ledger. Planners contribute typed BOMs, not live
capacity arithmetic. Opaque native graph pools are admitted as complete retained
families through `GPUGraphMemoryContract`; an individual pool-growth observation
is diagnostic evidence, not bytes owned by the graph that triggered it.

Explicit RAM/VRAM ceilings enter ordinary and overlay admission through the
same `PhysicalMemoryAuthority::admissionCapacity` operation, including a GPU's
associated host allocations. Configuration only converts checked MiB intent to
bytes. Raw observations are validated before a ceiling is applied; an exhausted
resource is distinct from a missing or inconsistent observation. Only
`PhysicalMemoryCapacityExhausted` identifies a valid BOM that does not fit.
Candidate searches must not catch malformed topology, unsupported geometry,
overflow or lifecycle failures as permission to choose a different shape.

Native homogeneous-GPU ExpertOverlay treats its replica-cache request as an
upper bound. `MoEOverlayCapacityAdmission` selects the largest positive cache
that fits alongside complete model coverage and all named graph/runtime BOMs.
Only a typed physical-capacity exhaustion permits another candidate; invalid
topology or format still fails. The immutable replica-count grant travels with
the resolved placement plan into graph setup and retained-runner reuse. Rolling
transfer lanes remain unchanged, and admission never switches Dynamic or an
enabled cache off. The grant is geometry, not a parallel physical-byte ledger.

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
`PreparedWeightStore` gives stages stable typed handles. Global PP shares the
model-owned store across its local stages; complete preparation identity and
stage-scoped bindings keep disjoint layer/device preparations distinct. A stage
must not introduce another store or replace another stage's live bindings.

MoE expert overlays add explicit owner maps, preparation/runtime plans, payload
providers, device slot pools, and transfer services. Rebalance changes are
published through graph-visible maintenance/state rather than by rewriting
ordinary stage ownership behind the graph.

Terminal model disposal and prepared-context reuse have different obligations.
`modelContextOverlayDrainIntent` projects the exported reuse authority into the
local device-controller drain intent. Ordinary server disposal drains admitted
transactions but never restores a discarded model's initial placement. The
existing shutdown rendezvous joins the rank-local obligations; if any rank
retains a Dynamic prepared context, all physical peers participate in its
restoration. Only that retained branch can certify prepared-context reuse.
This exchanges host-owned resource lifetime, not GPU placement or inference
state, and adds no inference-time collective.

Host maintenance shutdown closes proposal admission, not physical-wave ownership.
The retained transaction's `ReadyToStage` admission boundary may discard an
unstaged process-local proposal; distributed publication obligations still
finish. An `Active` wave keeps the ordinary authority poller through preparation,
selector publication or asynchronous abort. Old-reader retirement remains a
separate authority-owned drain obligation. Shutdown does not clear active intent
or invent a second transfer/completion path.

Host-authoritative placement keeps its smoothed demand forecast private to
`MoEOverlayResidencyAuthority`. Executable transactions retain the original
observed window and derive movement activity from it. Repeated generations
authenticate complete routing identity, including retained batch boundaries,
before reusing a forecast. Only the coordinator's policy fingerprint includes
the forecast identity; followers adopt the selected plan without reconstructing
history or receiving a second histogram.

Host service economics uses one observed-transaction objective for participant
search, bounded tier selection and final cycle/cohort admission. It sums the
slowest participant's work in each actual routed invocation; taking a maximum
after summing the entire window loses co-occurrence and can reverse the payoff.
Private forecasts still rank placement, but cannot supply service evidence or
its token denominator. Missing invocation evidence is a hard error when pricing
service. Production setup must admit the bounded routing banks and publication
storage before enabling this contract.

`MoEOverlayHostDemandMemoryPlan` composes those payload owners into the host
`ExecutionWorkspace` BOM before automatic expert filling. Its geometry comes
from model-resolved routed layers, the maximum adaptive window and retained
invocation bounds, not a fixed model or topology cap. The plan covers both RCU
banks, overlapping immutable observations and one reused MPI mailbox. Static
and device-resident authorities cannot request this host evidence allocation.
The retained-weight capacity identity includes both window bounds; it cannot
reuse an admission made for smaller routing storage.

The admitted host-demand plan is retained with the resolved capacity result and
binds live histogram storage to the same PMA and model/window identity. CPU
routers and declared heterogeneous dispatch-ticket boundaries publish complete
executed invocations, including MTP candidates later rejected. Phase is explicit:
a one-row predictor is MTP work, not ordinary decode. Histograms measure routed
work at the named main-layer boundary, not accepted response-token progress.
KV/GDN, sampler and accepted-token publication remain separate authorities;
CPU accepted-state publication does not walk routers or republish their history.
The ticket path adds no readback beyond its existing heterogeneous boundary.
Native all-GPU accepted-row ledgers remain device-owned; their transaction-cost
port is a separate implementation requirement, not a host-policy substitution.

The device overlay controller separates the open transaction's phase intent
from its last sealed command. Opening snapshot N+1 must retain command N and
its publication word: an empty decision can finish before a remote transport
worker acquires it. Every worker consumes N before joining snapshot N+1, so the
existing topology-wide snapshot fan-in is the sole command-buffer reuse edge.
Do not add a second acknowledgement, clear the command at transaction-open,
or reconstruct a missed command from host-side policy state.

Native homogeneous GPU maintenance likewise retains the exact command wave
through durable publication. Applying it prepares the reserved RCU bank and
publishes `PreparedForPublication`, not `Applied`. The existing all-layer
finalizer authenticates the apply's wave/epoch/count against the retained
headers, switches the selector, and only then recycles the commands. Invalid
publication poisons that controller and preserves the failed identity. This
uses the same captured stream DAG; no host acknowledgement or extra launch is
part of retirement.

The native finalizer also seals a bounded movement receipt after selector
publication. At the existing terminal request boundary, the configured root
exports only the populated journal ranges and authenticates request generation,
workspace identity and monotonically increasing model epochs.
`NativeMoEMovementArchive` retains these immutable diagnostic facts across
request resets; it is never consulted by inference or placement policy. Native
load-spread proofs retain their routed-work units and completed copy receipts
retain actual payload bytes. Optional PerfStats mirrors use the common
completed transaction/edge/byte vocabulary. The public runner forwards the
unique publication root, not the first participant or the dormant setup-time
residency object. Device-owned activity is explicitly opaque to this terminal
observer; an old receipt cannot certify current between-wave quiescence.

`moeOptimizationStatus()` projects passive admission headroom from the actual
owner. Host policy exposes its active RCU routing bank; a mapped all-GPU
controller exposes the fuller of its independent prefill/decode submission
windows, including retired progress queued for the next command. The typed
scope distinguishes cadence from GPU routing histograms and names the traffic
that can close the window. This diagnostic is not a reservation or a host
placement mirror. Only an exclusively admitting observer may use it to place
an evidence cohort between waves; ordinary inference retains lock-free progress
notification and device-authored policy/receipts.

### 6.3 Coherence and transfers

`TransferEngine` is the public authority for tensor movement and coherence
publication. Placement owners use its preparation APIs before execution;
stages require already-resident inputs, prepare output storage, and publish the
exact producer stream/event through `StageGPUExecution`.

The executor may join a consumer stream to a published event. It may not
allocate, upload, download, migrate, or guess a stream to repair a stage input
while a graph is running. CPU consumers explicitly materialize host data;
cross-device and cross-vendor transfers use a declared transfer plan.

Same-vendor activation copies acquire the source tensor event and submit
NCCL/RCCL through `copyOnStreams` with exact sending and receiving streams.
Coordinators may serialize host submission but cannot replace those streams or
wait for GPU completion. TransferEngine publishes the receiving event and
extends the source lifetime through its read. Retained PP replay imports its
single newly written ingress buffer before launch; cached residency is not
proof of fresh-byte readiness. CPU-visible and cross-vendor host boundaries
likewise acquire the source event before observing its bytes.

Every successful forward publishes its graph-declared result tensor after
launch, including deferred decode and verifier execution. A private sampler
stream handoff is not that public tensor event: nonterminal PP stages publish
hidden state for transfer consumers, while terminal stages publish logits.
Deferring completion never suppresses result publication; publication records
an event without waiting or making host data visible.

`CapturedTransferChannel` is TransferEngine's rank-local captured byte-message
primitive. It owns one mapped payload slot and a private GPU cursor at each
endpoint, all claimed through PMA. `memoryFor()` contributes allocation geometry;
it performs no independent capacity admission. Setup joins the exact cursor
initialization events before returning. Immutable bindings retain physical
storage, endpoint, nonce, semantic message key and exact extent through capture
retirement. The primitive cannot address another process or physical host.

For workspace-backed metadata, `DeviceWorkspaceManager::retainBuffer` returns
an immutable bounded `WorkspaceBufferLease`. It retains the original physical
block and its original PMA claim, not a manager or a name that can later move.
TransferEngine binds that region directly; it does not create a second mailbox
or copy state to prolong its lifetime. Dropping a manager's names is distinct
from releasing physical capacity: the last allocation lease frees the block
and only then returns its claim. Whole-block reuse is rejected while any
retained region survives. The enclosing graph still owns producer/consumer
ordering and must retire submitted work before its last storage lease.

The captured lifecycle is acquire → parallel payload copy → release. Each GPU
owns its completion word; producer reuse requires consumer acknowledgement.
Epochs continue across requests and retained graph families, without a host
shadow or reset round. Wrong identity/extent/order or the bounded peer deadline
aborts the channel and raises a native failure. A fixed-rate device timer keeps
that deadline independent of shader DVFS. This byte primitive does not itself
compose pipeline domains, choose MTP branches or certify a complete inference
topology; those responsibilities remain with the existing graph/rank owners.

Captured cross-vendor GPU pipelines preserve two distinct topology levels.
`PipelineDeviceGeneration` owns the ordered layer domains, while each domain's
existing runner owns its native TP members. `PipelineNativeDomain` freezes and
validates that membership independently of physical GPU ordinals. Adjacent
domain leaders exchange complete hidden rows forward and typed input/commit
metadata backward; a receiving leader then broadcasts inside its own NCCL/RCCL
domain. The final domain retains its existing sampler, mirrored MTP sidecars,
ticket authentication and result authority. Earlier domains own only their
local layer execution and accepted-state publication.

`PipelineForwardGraphEdges::PublicationOrder` expresses that publication as
one of three complete topologies: local commit only, commit then exchange, or
exchange then commit. Terminal TP siblings need only their mirrored local
commit; the leader alone publishes the committed words to earlier domains.
Those domains receive/broadcast before restoring their own state. Readiness
means a retained topology/storage lease with the exact bank identity, not
necessarily a nonempty transport executable. Policy-specific commit parents
compose the frozen transport; they never re-enter collective capture. Lease
retirement makes preparation unavailable and cannot resurrect an old arena.

`PipelineTransferMemory` supplies one shared geometry calculation to planning
and materialization. Two opposite-direction channels belong to each physical
boundary; TP width and the retained graph-family size do not multiply them.
Only boundary leaders claim GPU cursor storage, and one endpoint's host resource
claims the mapped backing. Arena activation storage is materialized before its
immutable channel binding is frozen. Metadata bindings retain the original
INT32 arena regions or workspace leases, never newly invented state mailboxes.

Role-asymmetric participant graphs declare their shared logical forward through
the existing `GraphCaptureWaveContract`; matching capture boundaries must not
depend on a leader-only egress node's name. This is setup identity, not a new
runtime rendezvous. Every native TP domain remains separate, and same-vendor
domain-to-domain transport cannot silently use the cross-vendor mapped channel.

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
`IGraphCaptureAuxiliaryBranch` to each retained inference graph. A private
mapped wake word is armed at the root. The branch performs one eager finite
inbox scan and remains as exactly one co-resident CTA until the inference
terminal publishes `InferenceComplete`. That bounded service interval observes
incremental CPU preparation without a host wake protocol and avoids the former
four-CTA SM tax. Device-only claims are shared with independently queued finite
passes, so exactly one executor completes a generation. A host enqueue cannot
claim GPU execution. Host maintenance owns immutable IO commands and acquires
exact generation receipts, not a shadow of GPU claims. Graph caches retain
private lifetime words, graph-only branch sources and the shared epoch authority.
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
worker. The worker therefore accepts work during external CPU waits without
per-child branches, paired capture-event state or per-replay host submissions.
Its one-CTA geometry is invariant across topology size; independently queued
finite passes may use wider grids only when no captured lifetime is retained.
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

Captured GPU-to-CPU dispatch publishes the request's pinned placement epoch
with its route and activation payload through the existing mapped release
edge. CPU dispatch acquires exactly that residency snapshot, not the latest
global publication. A background wave may publish a newer bank while the
forward still holds its old GPU reader. Device-authored ticket epochs are
immutable inputs; only explicitly host-admitted transactions may author their
epoch during host admission. A missing captured epoch or conflict with a
sequence lease is fatal.

CPU canonical-route ingress is a captured acquire/materialize/acknowledge DAG.
One thread acquires the CPU publication, then parallel row tiles copy its
immutable contributions, and the exact-stream terminal acknowledges reuse.
Never park a payload-sized grid awaiting a CPU publication: independent
maintenance and its event markers must remain runnable while the producer works.
Route metadata is shared per row rather than re-read from mapped host memory
for each contribution element. CPU ownership lasts until publication; the
ticket and payload remain immutable until the GPU acknowledgement.
Canonical GPU expert publication writes the complete original route bank,
including zeroes for non-local slots. CPU ticket materialization therefore
depends on the local GPU publication before replacing those slots. Only CPU
compute overlaps the GPU writer; joining both writers at the final reducer is
insufficient to order their overlapping stores.

Portable sparse rank batches own independent persistent MPI send/receive
storage per rank pair. `wireMoEOverlayRankBatchForkJoin` places every peer
dispatch before the first blocking return within a tier. Returns retain their
canonical accumulation order because they share the FP32 destination; transport
completion order cannot choose the arithmetic order. The helper adds only
graph dependencies, with no extra runtime barrier or host worker. Previous-tier
shared-buffer ownership and the final ticket publication remain explicit edges.

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

`PipelineGraphExecutionPlan` freezes the complete ordered local execution
membership of each domain, not just its leader. Setup and replay validation
compare every nested TP runner with the original collective membership without
allocating or entering a collective. A domain's GPU members are unique and
homogeneous; mixed backends belong in separate domains. Physical node/NUMA
identity remains with the canonical rank plan and context. Preserving a wider
domain in this declaration is not proof that its captured pipeline composition
has been installed.

PP construction is a two-phase setup transaction. Participant factories prepare
weights and stable state banks but do not declare a PP workspace family: the
pipeline composer must first bind the ingress owner and freeze its transport
edges. It then invokes the same serving-family materializer on every stage,
including CPU stages, before request admission. Whole-model factories retain
their independent eager-family policy. Missing PP ingress remains a hard error;
neither a dummy input nor first-request preparation substitutes for composition.

The pipeline chunk distinguishes logical token rows from physical transfer
capacity. GPU children retain captured bucket geometry and mask padding; CPU
children execute only logical rows. A nonterminal CPU child zeroes unused
outgoing activation rows for the next captured participant, without evaluating
fake tokens or appending them to KV. Terminal logits, snapshots and prefix
state describe the logical frontier, never the transport bucket tail.

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

`StageRunnerRegistry` contains only explicitly identified stage entries. Its
constructor authenticates every local entry against the compiled rank action,
including the layer interval, domain and vocabulary/embedding endpoints. Head
or tail lookup returns no owner when that endpoint is remote. An arbitrary
rank-local runner must never stand in for a missing stage or remote endpoint.

## 8. Request state, prefix cache, and MTP

`OrchestrationRunner` owns the high-level request transaction; the concrete
runner owns tensor state. Prefill may use bounded graph buckets and chunk
schedules. Decode uses resident logits and, on GPU, device-side sampling so the
ordinary host boundary is a compact token result.

Prefix caching is integrated with live KV/GDN state. Lookup, restore, truncate,
harvest, promotion/demotion, and device rehydration are explicit lifecycle
operations. Cache fingerprints include model/graph policy needed to reject an
incompatible state image.

Prefix payload ownership follows the participant's actual cache role. A
pipeline follower archives its main-model state, not the tail's shifted MTP
cache. `PrefixPayloadLayout` distinguishes attention-block chains from complete
recurrent checkpoints. An all-GDN slice hashes the full prompt ancestry but
stores only genuine checkpoint records; it owns no dummy K/V payload or empty
ancestor allocations. Coordination counts logical token coverage, not the
number of physical records. `PhysicalMemoryAuthority` admission includes the
recurrent archive staging and bounded RAM/device tiers even when the slice has
zero full-attention layers.

Coordinated prefix admission retains a typed placement-epoch span, not just
the newest participant epoch. Local and MPI nesting preserve both endpoints;
request completion samples the placement authority after harvest. This makes
publication during child lookups or harvest observable without changing cache
keys, inventing a host placement mirror, or synchronizing inference with
maintenance. A stale participant admission is discarded by the existing
participant-local harvest contract, never relabeled with a newer fingerprint.

MTP is not a separate eager model loop. The main graph, sidecar drafts,
all-position verifier, stochastic/greedy outcome graphs, shifted caches,
request-batched state, and accepted-state publication share typed transaction
objects and device events. `MTPDepthController` selects a permitted draft depth;
`MTPVerifierForwardExecutor` and the runner interfaces coordinate verifier
forwards. Production grouped verification must preserve serial-row byte
equivalence while publishing accepted state without row replay.

Terminal-head authoring defaults to `auto`. `ExecutionPlanBuilder` seals it
before admission: CPU continuation domains use vocabulary shards; accelerator
continuation domains retain full-vocabulary mirrors. Expert-only tiers do not
participate in that decision, and pipelines use the final domain. Explicit
policies survive unchanged. `ResolvedRankOrchestration` publishes the same
concrete policy to the saved configuration and rank runtime, so graph layout,
costing and `PhysicalMemoryAuthority` weight-set inputs cannot select separate
defaults. Lower-level graph and weight-accounting consumers reject unresolved
automatic intent.

Stochastic admission distinguishes full-model collective peers from expert-only
transaction followers through `MTPRankParticipation`. An installed overlay
coordinator owns the continuation's sampling transaction; remote expert services
do not acquire a vocabulary or sampler merely because they share its MPI world.
`mtpStochasticVerificationOwnerFailure` checks the actual runner's distribution
and publication contract: gathered CPU GlobalTP or resident GPU outcomes, with
mirrored full heads additionally required for local TP. It adds no new sampler,
live-state mirror, transport or synchronization boundary.

Logical rollback checkpoint admission carries the scheduler cursor and explicit
main/shifted append bounds for the admitted transaction. Resolve its draft
extent before checkpointing and use the same extent for execution. Retained
graph/workspace capacity is not live KV demand; every later transaction obtains
its own checkpoint. Participants reject append extents that could wrap over
archived prefix bytes. Device-owned positive-width transactions retain their
selected geometry and clip accepted publication on device.

Shifted-KV commit metadata is an observer, not the next payload writer. It
borrows the producer event without consuming the pending reader publication.
The actual terminal-hidden mailbox writer or shifted-KV sidecar acquires that
publication on its own exact execution stream. A wait on a metadata/setup
stream cannot authorize a later write on an unrelated stream; first-use capture
ordering is not proof of this dependency during retained replay.

Scalar condition forwards also carry explicit commit ownership. A speculative
continuation has already been counted by accepted-result publication; a
budget-one or forced-token forward owns a new serial commit. The latter is a
distinct `ForwardStateTransaction` in graph-cache identity, setup capture, and
canonical memory admission. HIP embeds the existing serial cadence publisher
at that graph's terminal; CUDA retains its native conditional maintenance
transaction. Never infer this distinction from shape or add a host cadence
counter to compensate for a missing captured publication.

Request reset uses typed reset transactions and KV-cache reset boundaries. It
joins outstanding device work, resets data and live-state generations, then
publishes readiness for the next request. It does not rebuild topology solely
to clear request state.

Ordinary decode snapshots its canonical device KV position inside the captured
forward graph. `DeviceDecodePositionBinding` names the cache count and the
distinct arena position row; every model root depends on
`DecodePositionSnapshotStage`. The full binding belongs to graph-cache identity.
Request admission still joins earlier readers, but must not perform this
snapshot outside the graph: a generation parent can replay its child without
another host prelude, and attention may advance the count on each iteration.

`DeviceGenerationGraphProgram` lowers the admitted generation policy and its
ordered transaction fragments into a native parent, or captures the isolated
immutable ticket publisher for a declared hosted policy. It borrows controller
storage and graph owners; it neither owns a second request ledger nor selects a
different execution policy after failure. Recorded child graphs need not own
independent executables when only the composed parent executes them. Nested
recording and implicit replacement of a retained parent are rejected before
changing capture ownership.

Its `DeviceControlledLoopProgram` separates admitted local initialization,
unconditional collective arrival and the repeated transaction. A pipeline tail
initializes under its own health/completion guard; followers have no local
initializer. Arrival always runs, even when the initializer just terminated the
request. A follower therefore receives the current tail command before reading
any predicate, and a terminal tail cannot strand a peer in its receive. Only
then is continuation evaluated before any WHILE iteration. Sampling prefill logits
therefore consumes no model-state row, and budget-one/EOS initialization must
admit no dummy forward. Initialization may use explicit device-word predicates,
but not an iteration-depth selector. Native IF and WHILE scheduling handles
are distinct; both derive their decision from the same resident controller.
Empty arrival adds no nodes; single-device programs retain their original
initialization and loop shape. This compiler
contract is shared by the device owner's ordinary and speculative programs.

Ordinary GPU `decodeStep()` admits a complete response budget and immutable
sampling law, then consumes the device owner's terminal ledger. It never
returns tokens to a host-driven per-token forward/sampler loop. A budget-only
terminal retains its already-emitted condition in the existing resident
mailbox; the next admission consumes it exactly once. Ordinary execution owns
no verifier or speculative-depth statistics. CPU generation remains host-owned.
The participant compiler requires a complete local model and sampler; a
pipeline shard or overlay boundary needs its explicit composed transfer and
maintenance protocol, and is rejected until that composition is installed.

`PipelineDeviceGeneration` supplies the rank-local captured composition for
ordinary and speculative homogeneous GPU pipelines. It borrows each DGO's captured forward,
arena and request-event handoff. Only the tail admits a generation budget or
records a sampler. Earlier stages receive exactly health, completion and the
next condition token over the existing native collective; they do not initialize
or materialize response ledgers. Activation receives target each stage's own
stable hidden bank, with a captured send after the preceding stage's forward.
Every parent is prepared before any is submitted. The sole tail response is
returned after terminal validation of independently advanced stage KV positions.
The host submission lifecycle is Idle → Admitted → Materialized → Submitted →
Idle; partially failed transitions are absorbing.

Completed ordinary parents publish forward-replay observations through the
original `ForwardExecutionEngine` cache owner. The observer checks the exact
signature and borrowed child identity in constant time, then reports completed
model forwards from the authenticated terminal count. It neither walks stages
nor advances cache/execution state. The prefill sample contributes zero
forwards. Pipeline followers publish only after independent terminal KV checks;
PerfStats never supplies a count or controls retirement.

CUDA parents receive an unconditional command before their first predicate,
then exchange the next/terminal command at the end of each device-owned WHILE
iteration. HIP instead records command arrival at the start of each complete
transaction. Only the tail publishes authenticated immutable scheduler tickets;
the host submits every follower transaction before the tail's selected branch
and observes the next ticket only after all submissions. A pending prefill
sample is a tail-only zero-forward transaction. A continued response starts
with a complete all-stage transaction, consuming the already-emitted frontier
exactly once. Terminal tickets retire follower stream handoffs without forging
terminal controller words or launching an extra stop-broadcast graph. Followers
own the explicit `HostedPipelineTransaction` executable kind, never ticket
publishers or independent response controllers. Request reset and continuation
reuse these retained executables and their existing graph memory inventory.

Speculative followers retain their main-layer grouped verifier and local
accepted-state publication. Only the tail owns the sidecar, outcome decision
and depth selector. CUDA composes native continuation commands; HIP selects
complete retained transactions using the tail's authenticated immutable ticket.
This composition does not yet provide heterogeneous or nested-TP pipeline
programs: those require complete domain-preserving transaction/follower
composition, not treating a TP group as one DGO or introducing a host-token loop.

Captured speculative state publication has an explicit immutable authority:
`CompactOutcome` derives accepted metadata, `BoundedGeneration` also commits
the sole response ledger, and `PipelineFollower` consumes that committed
metadata to advance only its own main-model KV/recurrent/routing state.
`Unbound` is invalid. A follower cannot bind outcome reduction, response,
penalty-history or shifted-predictor storage; all such mixtures fail before
enqueue. The role participates in graph identity. This participant-local
operation alone does not supply the pipeline command/activation transport or
certify a complete speculative pipeline program.

Verifier preparation has the matching explicit `VerifierInputOwner` and
`PipelineFollower` roles (`Unbound` cannot execute). The owner assembles tokens,
positions and transaction width; the follower checkpoints only its own opaque
main-KV frontier before append. A follower rejects every token, controller,
maintenance and geometry-publication binding before enqueue. Both roles use the
same captured checkpoint operation, with role/address/physical geometry in
capture identity. The follower declares no sampler/token arena buffers. This
stage contract does not itself install the complete pipeline MTP scheduler.

Homogeneous rank-local pipeline setup freezes `PipelineForwardGraphEdges`
before creating any forward engine. It encloses each local model DAG in native
adjacent-stage receive/send nodes; the receive targets the participant's own
hidden bank and the send depends on every local model leaf. The canonical
collective classifier keeps these edges inside the complete captured graph.
The chunk materializer owns real lengths and KV position; transport moves
exactly the selected physical bucket, not arena capacity. Every stage receives
the same immutable root schedule and request input once, then submits its
retained chunks without a host stage-by-stage activation handoff. Capture-only
rendezvous uses the native communicator and the standard bounded protocol.
Its one persistent fence word per GPU is admitted by the shared rank physical
memory authority via `CollectiveMemoryEstimator::nativePipelineBoundaryBytes`;
pipeline transport does not allocate TP's FP16 conversion scratch.
The immutable edge owner is part of forward-cache identity and outlives its
borrowed captures. Late installation is rejected rather than invalidating an
already prepared graph family. No additional prefill executable or activation
allocation is introduced by composition.

`PipelineActivationExchange` is the explicit activation collective used by
these native edges. Its separate `CapturedDomain` binding describes a
rank-local heterogeneous TP-domain boundary: only member zero publishes or
receives the TransferEngine channel, then destination members use their native
NCCL/RCCL broadcast. Member zero is communicator order, not GPU ordinal zero.
The frozen binding rejects membership drift, nonleader publication, wrong
endpoint ownership and same-vendor channel substitution. Banks stay
participant-local and the physical message extent excludes unused capacity.
Captured receive-to-send dependencies use the normal graph dependency ledger,
not an external event or a host assertion that recording produced live bytes.
This activation primitive alone does not install mixed-domain MTP metadata
transport, memory admission or complete generation composition.

The same owner encloses a `GroupedMTPVerifier` forward using its explicit
decode phase, resident token/position/length rows, and exact physical verifier
width. It does not require a prefill binding or derive acceptance. Contradictory
roles and foreign banks fail before graph mutation. Receive precedes every
model root and send follows every model leaf; unused arena rows never cross
the edge. The forward signature includes the frozen edge owner and existing
role/geometry/address identities. Ordinary decode remains parent-composed and
must not acquire duplicate child edges. This transport support is not, by
itself, a complete MTP pipeline controller.

Scalar `MTPCondition` forwards use that same edge owner with exactly one
resident row. They retain ordinary, committed-condition or restored-prefix
transaction identity rather than posing as a grouped verifier. Only the
terminal participant may carry the complete runtime shifted-cache binding for
a restored-prefix bridge; earlier stages carry none. Unknown transactions,
host rows and declaration-only bridge bindings fail before graph mutation.

For both scalar conditions and grouped verification, the terminal broadcasts
three independently addressed resident banks (tokens, positions and logical
width) inside that forward graph. The follower's local main-KV checkpoint
follows the row receive and precedes every verifier model root. No bank is
assumed contiguous with another, and only physical active graph rows cross the
native collective. The DGO prelude validates those exact local arena/mailbox
bindings against the frozen contributor; it neither manufactures an owner
token plan nor launches another preparation producer. Grouped checkpoint
ownership must match the invocation's local KV cache. Nonterminal participants
bind neither vocabulary outputs nor shifted predictor/terminal-hidden state.
These graph-local contracts do not yet compose the full speculative parent.

Accepted-state composition uses that same frozen collective owner. Serving setup
captures its immutable native broadcast once. The local publication authority
captures its computation separately; `composePublication` clones both recordings
into one complete native parent: the terminal first derives/commits acceptance,
then sends; each follower first receives, then restores its local KV, recurrent
state and accepted routing history. A changed terminal commit limit must not
make unchanged followers enter another collective capture wave. The four
independently bound INT32 banks are restore
row, target cached count, accepted count and publication health. Never copy a
whole workspace or presume contiguous allocation. Follower checkpoint identity
must match the preceding verifier, and only the terminal may bind outcome,
response or predictor authority. Validation occurs before graph mutation or
enqueue. Each DGO owns the transport capture lease and releases it after enclosing
parents but before its arena; the longer-lived rank topology holds only a weak
reference and cannot resurrect an expired participant lease. The existing
graph-cache exporter returns the complete composed parent,
never its graph-only local child, and rejects parents with host-serviced boundaries.
This publication subgraph does not itself authorize a complete MTP
generation policy; the enclosing parent/ticket compiler must still join it to
verification and the next condition.

Pipeline prefix restoration remains rank-owned admission over independently
owned stage caches. A full hit restores every stage's state and the tail's
terminal output; a partial hit submits only the unprocessed suffix, with its
absolute start in the shared chunk schedule. Reset retires logical state, not
necessarily old physical KV bytes. Restore/import events must therefore order
the next captured prefill/generation without a diagnostic host observation.
The model-free preflight checks exact restored KV payloads after generation as
well as token output; neither check substitutes for public real-model proof.

Forward export readiness comes from the cache's typed executable-submission
state, not the semantic node-reset optimization flag. A materialized but
unlaunched exact child is composable: preparation must not execute a synthetic
inference transaction just to make that child exportable. Request admission
joins the existing reset publication on its exact stream before overwriting
resident control banks. Ordinary admission does not require speculative graph
capacity; this storage contract alone does not certify a composed model
generation loop or a pipeline follower protocol.

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

Snapshot bytes, shape, lifetime, and publication completeness share one
immutable owner. A named finalizer may publish a complete value under a key
whose earlier producer held a TP partial. `SnapshotPublication` carries that
distinction to collectors: completed copies must agree and are never summed
with earlier partials; partial-only captures retain ordinary schema assembly.
Prefill chunk joins preserve the same completeness contract across all rows.

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
