# Kubernetes and OpenMPI Multi-Cluster Execution Project Proposal

- **Date**: 2026-07-14
- **Status**: Proposed
- **Scope**: Production deployment of Llaminar V2 inference worlds on Kubernetes, OpenMPI launch and control, GPU-native intra-world communication, topology-aware scheduling, world-level recovery, and multi-cluster fleet placement
- **Related projects**: [Automatic Device Placement V2](../2026-06/AUTO_DEVICE_PLACEMENT_V2_PROJECT_PLAN.md) and [Multi-Domain Pipeline Execution](../2026-06/MULTI_DOMAIN_PIPELINE_EXECUTION_PLAN.md)
- **Normative language**: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` are requirements levels.

---

## 1. Executive Decision

Llaminar will treat one Kubernetes-hosted OpenMPI job as one immutable,
gang-scheduled **Llaminar execution world**. A world is a fixed set of MPI ranks
that loads one model revision, constructs one compatible orchestration plan, and
serves as one failure and readiness unit.

The production multi-cluster architecture will be a fleet of independent
execution worlds behind a global request router:

```text
                               client traffic
                                     |
                                     v
                         global / regional router
                         +-----------+-----------+
                         |                       |
                         v                       v
                 Kubernetes cluster A   Kubernetes cluster B
                 +-------------------+   +-------------------+
                 | Llaminar world A1 |   | Llaminar world B1 |
                 | MPI_COMM_WORLD    |   | MPI_COMM_WORLD    |
                 | ranks 0..N-1      |   | ranks 0..M-1      |
                 +-------------------+   +-------------------+
                         |                       |
                 latency-local fabric   latency-local fabric
```

The default production design will **not** stretch one `MPI_COMM_WORLD` across
multiple Kubernetes clusters or regions. Tensor parallelism performs
latency-sensitive collectives repeatedly within every transformer layer. A
wide-area link, cross-cluster network policy, transient route, or failed rank
would therefore stall or abort the entire model instance. Cross-cluster scaling
will add or remove whole worlds and route requests between them; it will not add
ranks to a live world.

The execution hierarchy is:

```text
fleet
  -> cluster-local execution world (one static MPI job)
       -> topology domain (node, NVLink/xGMI island, rack, fabric block)
            -> MPI rank
                 -> one or more participant-local DeviceGraphOrchestrators
```

This preserves the V2 design north star: graphs remain per-device and
symmetric, compute stages remain participant-local, and distributed behavior is
represented by explicit collectives or pipeline-transfer stages.

OpenMPI will remain the process-launch, rank-identity, control, and fallback
communication substrate. Homogeneous CUDA and ROCm tensor collectives will use
NCCL and RCCL respectively. Cross-rank pipeline transfers will become
device-native and asynchronous before they are promoted as a production GPU
path. Host-staged MPI remains a correctness fallback, not the target GPU data
plane.

The first production integration will use the Kubeflow MPI Operator's
`MPIJob` API. Kueue will provide gang admission and topology-aware placement.
MultiKueue may place a complete `MPIJob` onto one selected worker cluster; it
does not partition one MPI world across clusters. A Llaminar-specific
`LlaminarWorld` controller is a later lifecycle layer, not a prerequisite for
the first single-cluster milestone.

## 2. Goals, Use Cases, and Non-Goals

### 2.1 Primary goals

This project will make the following scenarios supported and reproducible:

1. Run single-rank and multi-rank Llaminar `oneshot`, `benchmark`, and `serve`
   commands in Kubernetes.
2. Run one MPI rank per physical GPU island or Kubernetes worker node, with
   local TP over all GPUs allocated to that rank.
3. Run one MPI rank per GPU for cross-process NCCL or RCCL tensor parallelism on
   a low-latency cluster fabric.
4. Run pipeline stages across ranks and nodes without implicit GPU-to-host
   staging on the promoted GPU path.
5. Expose only rank 0 as the HTTP ingress endpoint while all other ranks remain
   coordinated inference workers.
6. Schedule all ranks atomically and place them according to GPU, NUMA, rack,
   and fabric locality constraints.
7. Treat a failed or replaced rank as a failed world and restart the complete
   world with a new epoch.
8. Operate multiple independent worlds across clusters and regions behind a
   common service-routing layer.
9. Produce topology, launch, readiness, collective, and request telemetry that
   makes a distributed failure diagnosable without entering the pods.
10. Preserve parity between local bare-metal OpenMPI execution and Kubernetes
    execution for the same model, graph, and placement plan.

### 2.2 Representative use cases

The target deployment patterns are:

- a model replica on one 8-GPU NVLink or xGMI node;
- a model replica spanning several nodes in one RDMA-capable cluster;
- multiple replicas placed into different failure zones for availability;
- a central queue choosing among several GPU clusters with different GPU
  models or available capacity;
- a heterogeneous local world whose named domains combine CUDA, ROCm, or CPU
  pipeline stages where the existing orchestration model permits it; and
- an explicitly experimental federated pipeline for a model that cannot fit in
  any one cluster, using separate cluster-local worlds and a purpose-built
  stage bridge.

### 2.3 Non-goals

The following are not production goals for this project:

- elastic membership inside an active MPI world;
- replacing failed ranks in place;
- stretching tensor parallelism across regions or ordinary WAN links;
- relying on Kubernetes Service load balancing between MPI ranks;
- using MPI as the global request-routing protocol;
- making Multi-Cluster Services API discovery create a cross-cluster network
  fabric;
- depending on `--no-mpi-bootstrap` in Kubernetes;
- introducing nested multi-device subgraphs;
- changing model numerics or kernel algorithms merely to support deployment;
- building a training scheduler; or
- promising cross-cluster pipeline performance before device-native transfer,
  batching, and latency-hiding gates pass.

MPI fault-tolerance extensions such as ULFM may be investigated separately,
but this proposal assumes conventional static OpenMPI job semantics.

## 3. Terminology

| Term | Meaning |
|---|---|
| **Execution world** | One immutable set of Llaminar MPI ranks with one `MPI_COMM_WORLD`, model identity, orchestration plan, and world epoch. |
| **World epoch** | A unique identifier for one creation of a world. Any full restart creates a new epoch, even if the Kubernetes object name is reused. |
| **Fleet** | A routable collection of independent execution worlds, potentially spread across clusters and regions. |
| **Launcher pod** | The MPI Operator pod that runs `mpirun` and supervises the job. It does not serve inference and normally requests no GPU. |
| **Worker pod** | A pod running `sshd` and one or more launched Llaminar MPI processes. The proposed default is one MPI slot per worker pod. |
| **Rank pod** | A worker pod configured with one slot, so its single Llaminar process has an unambiguous rank-to-pod identity. |
| **Service rank** | MPI rank 0, the only rank that binds the public Llaminar HTTP port. |
| **GPU island** | A set of GPUs with a high-bandwidth local interconnect such as NVLink/NVSwitch or xGMI. |
| **Topology domain** | A placement and communication boundary such as GPU island, NUMA node, physical node, rack, or fabric block. |
| **Cluster-local world** | A world whose ranks are admitted into one Kubernetes cluster and connected by one qualified low-latency network profile. |
| **Federated pipeline** | Separate cluster-local worlds connected through an explicit stage-transfer protocol, not one cross-cluster `MPI_COMM_WORLD`. |
| **World fingerprint** | Digest of the build, model, runtime libraries, rank layout, orchestration config, transport profile, and topology identity. |

Kubernetes pod names and OpenMPI processor names are not, by themselves,
physical-node identities. The runtime must keep cluster, zone, rack, physical
node, pod, rank, device UUID, and NIC identities distinct.

## 4. Non-Negotiable Invariants

### 4.1 Graph and execution invariants

1. Every compute graph MUST remain per-device and symmetric.
2. Multi-rank work MUST remain an explicit collective or transfer boundary.
3. A device with no participant-local work MUST no-op until the next
   collective; it MUST NOT be hidden inside a nested remote subgraph.
4. Domain communicators MUST include only the ranks that participate in that
   domain.
5. Rank-local and domain-local execution MUST produce the same model results in
   Kubernetes as in the equivalent direct OpenMPI launch.

### 4.2 World membership invariants

1. World membership, rank order, and world size MUST be fixed before model
   initialization.
2. All rank pods MUST be admitted before `mpirun` starts Llaminar.
3. A rank restart, pod replacement, unrecoverable collective timeout, or network
   partition MUST fail the world.
4. Kubernetes MUST restart the complete world with a new epoch. A replacement
   process MUST NOT attempt to join an existing communicator.
5. A world MUST NOT receive inference traffic until all ranks have confirmed
   their model, graph, device, and communicator readiness.

### 4.3 Launch invariants

1. Kubernetes launches MUST use external `mpirun` through the MPI Operator.
2. `--no-mpi-bootstrap` MUST NOT appear in a Kubernetes workload manifest.
3. Llaminar MUST detect `OMPI_COMM_WORLD_*` and continue into normal
   `MPI_Init_thread` rather than recursively invoking `mpirun`.
4. Because external MPI detection bypasses Llaminar's self-bootstrap command
   construction, the launcher MUST supply the required OpenMP, BLAS, binding,
   and MPI environment explicitly.
5. Every rank MUST run the same Llaminar image digest, OpenMPI build, collective
   library ABI, model digest, and orchestration configuration.
6. Llaminar MUST fail startup when `MPI_Init_thread` provides less thread
   support than the selected runtime features require.

### 4.4 Networking and collective invariants

1. Every rank MUST have direct, bidirectional connectivity to every peer needed
   by the selected OpenMPI and GPU-collective transports.
2. Service NAT or ingress proxies MUST NOT sit on an MPI collective path.
3. The launcher host list is process-placement information; it MUST NOT be
   treated as transport-interface selection.
4. Network profiles MUST explicitly select the routable data and OpenMPI
   control interfaces when pods expose multiple interfaces.
5. Homogeneous CUDA global TP SHOULD use NCCL; homogeneous ROCm global TP SHOULD
   use RCCL.
6. A production GPU path MUST NOT silently fall back to host staging. Such a
   fallback MUST be visible in startup status and telemetry and MUST fail closed
   when the deployment profile requires device-native transport.
7. Tensor parallelism MUST remain inside a qualified low-latency topology
   domain. Cross-region TP is prohibited.

### 4.5 Service invariants

1. Only rank 0 may bind or receive traffic from the inference Service.
2. A Service MUST NOT select every MPI worker pod.
3. Rank 0 readiness MUST represent whole-world readiness, not merely a listening
   HTTP socket.
4. Liveness probes MUST NOT execute collectives or introduce a second command
   stream into the coordinated worker loop.
5. Initial serving remains one serialized inference stream per world. Horizontal
   concurrency comes from multiple worlds until a separate batching/concurrency
   project changes that runtime contract.

## 5. Current Llaminar Baseline Audit

### 5.1 External OpenMPI launch already enters the correct path

`src/v2/app/MPIBootstrapPhase.cpp` detects an existing OpenMPI environment via
`OMPI_COMM_WORLD_*`. When a process is already an MPI child, the bootstrap phase
returns `CONTINUE` and does not invoke Llaminar's self-generated `mpirun`
command. `src/v2/app/commands/CommandMPI.cpp` and
`src/v2/app/RuntimeInitPhase.cpp` then initialize MPI with
`MPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)`.

This makes MPI Operator launch structurally compatible today. It also means the
external launcher inherits responsibility for environment and affinity values
normally generated by `MPIBootstrap::buildMPIRunCommand()` in
`src/v2/utils/MPIBootstrap.cpp`.

Two launch gaps remain:

- the returned MPI thread level is recorded but not validated; and
- there is no immutable startup record proving that every rank received the
  same launch contract.

### 5.2 Rank 0 is already the service process

`src/v2/app/modes/ServerMode.cpp` sends non-root ranks into
`runMPIWorkerLoop()`. Rank 0 alone creates the HTTP server and broadcasts work
to its peers. This is the correct basic service shape for Kubernetes.

The current `/health` endpoint is process-local and does not prove that the
whole world is ready. Inference is also intentionally serialized through a
single `httplib` task-queue worker. The first Kubernetes release therefore
scales throughput by adding worlds, not by routing concurrent requests into one
world.

### 5.3 The ordinary coordinated runner has a worker protocol

`src/v2/execution/runner/OrchestrationRunner.cpp` implements rank-0 command
broadcast, non-root `runMPIWorkerLoop()`, cache clearing, prefill, decode,
sampling updates, shutdown, and related coordinated actions. This is the
baseline serving path for the first multi-rank Kubernetes milestone.

### 5.4 Global/named-domain serving is incomplete

`src/v2/execution/global/GlobalOrchestratorRunner.cpp` still has unimplemented
`runMPIWorkerLoop()` and `shutdownMPIWorkers()` methods. A named-domain or global
pipeline configuration that selects this runner cannot yet be considered a
production multi-rank server even if one-shot inference succeeds.

Completing the global runner command protocol is a hard prerequisite for
promoting named-domain PP+TP serving on Kubernetes.

### 5.5 Native multi-process GPU collective foundations exist

`src/v2/collective/BackendRouter.cpp` prefers NCCL for homogeneous CUDA groups
and RCCL for homogeneous ROCm groups, including global-scope groups.
`src/v2/collective/backends/NCCLBackend.cpp` uses MPI to broadcast a native
collective unique ID and initializes a communicator with `ncclCommInitRank`.
The RCCL path follows the same design.

`src/v2/execution/local_execution/collective/CollectiveContext.cpp` uses a
device pointer directly for native backends. It host-stages a GPU tensor only
when the selected backend is MPI. These are the correct primitives for a
Kubernetes data plane, but every promoted global domain still needs tests that
prove it receives the appropriate domain-scoped `MPIContext` and native backend
rather than an accidental `MPI_COMM_WORLD` or MPI fallback.

### 5.6 Global pipeline transfer is currently host-staged and blocking

`src/v2/execution/compute_stages/stages/GlobalPPTransferStage.cpp` calls
`tensor->data()` before a blocking MPI send, forcing a GPU-to-host coherence
transition when the tensor is device-dirty. Its receive path calls
`mutable_data()`, performs a blocking MPI receive into host memory, and marks
the host copy authoritative.

This is a correctness implementation, not a production GPU pipeline transport.
It serializes synchronization, transfer, and downstream execution and prevents
useful latency hiding across nodes.

### 5.7 Cluster inventory conflates pod identity with physical topology

`src/v2/execution/mpi_orchestration/DeviceInventory.h` represents ranks,
hostnames, nodes, NUMA information, devices, UUIDs, PCIe data, and P2P matrices.
`src/v2/planning/ClusterInventoryGatherer.cpp` obtains a processor hostname and
derives node IDs from hostname values. `src/v2/utils/MPITopology.cpp` also uses
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` plus hostname-based node detection.

In Kubernetes, separate worker pods on the same physical node normally have
different hostnames and separate IPC namespaces. The runtime can therefore
mistake pods for physical nodes and lose the actual rack, zone, node UID, GPU
resource, and network-device relationships. Kubernetes topology identity must
be injected explicitly and kept separate from the OpenMPI processor name.

### 5.8 Failure semantics already imply whole-world recovery

Several collective timeout and fatal paths call `MPI_Abort(MPI_COMM_WORLD, 1)`.
That is consistent with a static-world deployment: a rank-desynchronized
inference process is not repaired in place. Kubernetes supervision must surface
the world failure and replace the whole gang.

### 5.9 Baseline blocker matrix

| Surface | Current state | Required before production |
|---|---|---|
| External MPI detection | Implemented | Add launch-contract validation and integration tests. |
| MPI thread support | Requested but not checked | Fail or disable incompatible features based on `provided`. |
| Ordinary coordinated serve | Implemented | Add world readiness and Kubernetes lifecycle tests. |
| Global/named-domain serve | Worker loop and shutdown are TODO | Implement full epoch-checked command protocol. |
| CUDA/ROCm global collectives | Native foundations exist | Prove domain scoping, no fallback, parity, and performance. |
| Global PP transfer | Blocking host-staged MPI | Add device-native asynchronous transfer backend. |
| Topology inventory | Hardware-rich but hostname-centric | Add Kubernetes cluster/zone/rack/node/pod/NIC identity. |
| Failure recovery | Process abort semantics | Add world controller policy and full-gang restart. |
| HTTP readiness | Local `/health` only | Add `/livez` and whole-world `/readyz`. |
| Request concurrency | Serialized | Scale whole worlds first; batch/concurrency is separate. |

## 6. Target System Architecture

### 6.1 Control plane and data plane

The deployment separates fleet control, world control, request ingress, and
tensor data:

```text
management cluster / GitOps
  |
  +-- Kueue / MultiKueue admission
  +-- world template and model revision
  +-- global service discovery and traffic policy
  |
  v
selected workload cluster
  |
  +-- MPI Operator
  |     +-- launcher pod: mpirun, hostfile, job supervision
  |     +-- worker-0: Llaminar rank 0, HTTP ingress
  |     +-- worker-1..N: coordinated Llaminar workers
  |
  +-- rank-0 Service / EndpointSlice
  +-- world status, metrics, logs, events
  |
  +-- OpenMPI launch/control plane: SSH + PRRTE/OOB + MPI bootstrap
  +-- tensor data plane: NCCL / RCCL / qualified MPI-UCX fallback
```

The launcher pod MUST remain alive for the lifetime of the world because its
`mpirun` process is the Kubernetes-visible supervisor for the remote ranks.
The launcher does not sit in the inference request or tensor data path.

### 6.2 Locality hierarchy

The placement compiler and scheduler will use this order from strongest to
weakest locality:

| Locality | Preferred execution | Communication policy |
|---|---|---|
| Same GPU island | Local TP within one rank when practical | NCCL/RCCL over NVLink/NVSwitch/xGMI. |
| Same physical node | Local TP or rank-per-GPU global TP | Native GPU collective; PCIe/NUMA aware. |
| Same rack/fabric block | Cross-rank TP or PP after qualification | RDMA-capable NCCL/RCCL or device-aware transport. |
| Same cluster, different block | PP preferred; TP only with measured admission | Explicit network profile and performance gate. |
| Same region, different cluster | Independent worlds | Route complete requests; no TP. |
| Different region | Independent replicas | Route for locality and availability; no TP or token-synchronous PP. |

The planner MUST fail closed when a requested topology cannot be placed inside
the maximum allowed locality class.

### 6.3 Rank layout profiles

The first release will support two explicit rank layouts.

#### Profile A: one rank per GPU island

Each worker pod requests all GPUs in one island and has `slotsPerWorker: 1`.
Llaminar uses `RankOrchestrator` for local TP across the devices visible to that
rank.

```text
physical node 0 / worker pod 0 / MPI rank 0
  -> cuda:0..7 local TP

physical node 1 / worker pod 1 / MPI rank 1
  -> cuda:0..7 local TP
```

This is the preferred shape when local TP is mature for the selected model and
inter-node communication is limited to PP or a coarser global domain. It avoids
confusing pod hostnames with multiple ranks on one physical node.

#### Profile B: one rank per GPU

Each worker pod requests one GPU and has one MPI slot. Global TP spans rank-local
devices through NCCL or RCCL.

```text
worker pod 0 / rank 0 -> one GPU
worker pod 1 / rank 1 -> one GPU
worker pod 2 / rank 2 -> one GPU
worker pod 3 / rank 3 -> one GPU
```

This shape gives direct rank-to-device identity and is appropriate when the
native global-TP path has been certified. The scheduler must still co-locate
pods according to node, rack, and fabric requirements.

Multiple MPI slots in one worker pod are not part of the initial support
contract because they complicate rank-to-pod readiness, CPU allocation, GPU
visibility, failure attribution, and Service endpoint selection.

## 7. Execution World Contract

### 7.1 Immutable world specification

Before launch, the controller will freeze a `WorldSpec` containing at least:

```text
world_id
world_epoch
llaminar_image_digest
llaminar_build_id
model_uri
model_digest
tokenizer_and_chat_template_digest
world_size
rank_layout
orchestration_config_digest
expected_device_classes
expected_topology_class
collective_profile
openmpi_version_and_build_id
nccl_or_rccl_version
network_profile
security_profile
```

Every rank will compute the same canonical digest. Startup will use a collective
comparison to prove all digests match. A disagreement fails startup before the
world becomes ready.

### 7.2 Rank startup phases

World startup is a coordinated state machine:

```text
PendingAdmission
  -> PodsScheduled
  -> MPIReachable
  -> RankInitialized
  -> InventoryGathered
  -> PlanValidated
  -> ModelLoaded
  -> CollectivesQualified
  -> WorldReady
  -> Serving
  -> Draining
  -> Terminated

Any non-retryable phase -> WorldFailed -> replacement world with new epoch
```

Rank 0 MUST publish `WorldReady` only after every rank has reported success for
the same epoch and fingerprint. Startup timeouts MUST identify the last
completed phase per rank.

### 7.3 Readiness and liveness

The HTTP surface will evolve as follows:

- `/livez`: rank-0 process-local liveness only; no MPI calls;
- `/readyz`: success only when the current world epoch is `WorldReady` or
  `Serving`, all rank initialization gates passed, and the process is accepting
  requests;
- `/health`: compatibility alias whose exact behavior is documented; and
- an internal world-status endpoint or metrics surface containing the
  fingerprint, epoch, rank count, model state, and degraded transport flags.

Non-root rank health is supervised by `mpirun` and world status. Worker-pod
probes MUST NOT open an independent inference command channel. If the remote
Llaminar child exits, `mpirun` must exit nonzero so the MPIJob becomes failed.

### 7.4 Rank-0 endpoint ownership

The inference Service MUST resolve to the pod whose live Llaminar process has
`OMPI_COMM_WORLD_RANK=0`. Pod ordinal alone is not sufficient proof.

The initial MPIJob template may use one slot per ordered worker and a dedicated
rank-0 label/endpoint. The production controller must validate rank identity at
runtime before publishing the EndpointSlice. A stale endpoint from an older
world epoch MUST be removed before a replacement becomes ready.

## 8. Kubernetes Workload Model

### 8.1 MPI Operator integration

The Kubeflow MPI Operator is the initial world substrate:

- one `Launcher` replica runs `mpirun`;
- `Worker` replicas run non-interactive `sshd`;
- `slotsPerWorker` is `1` for the supported initial profiles;
- the operator-generated hostfile addresses worker pods;
- a per-world SSH authentication mount is shared with launcher and workers;
- the same immutable image runs in both roles; and
- the launcher command invokes `llaminar2` directly on workers.

Llaminar's automatic MPI bootstrap stays enabled in configuration. It simply
detects that each remote process is already under OpenMPI and does not recurse.
The profiling-only `--no-mpi-bootstrap` flag is forbidden.

### 8.2 Resource requests

Worker pods MUST request and limit all placement-sensitive resources:

- the exact count and vendor resource name for GPUs;
- CPU requests appropriate for the chosen OpenMP thread count;
- memory sufficient for host weights, activation staging allowed by the
  profile, runtime metadata, page cache, and safety margin;
- hugepages when a measured CPU or communication path requires them;
- RDMA devices or network attachments when selected by the network profile; and
- ephemeral storage for logs or model staging when not using a mounted volume.

For latency-sensitive CPU execution, the cluster SHOULD use the static CPU
Manager policy and an appropriate Topology Manager policy. The launcher may use
burstable CPU; rank pods SHOULD receive exclusive integer CPU allocations when
the platform supports them.

GPU allocation MUST use the vendor device plugin or operator appropriate to the
cluster. CUDA and ROCm world templates remain separate unless a deliberately
heterogeneous Llaminar topology is selected.

### 8.3 Model distribution

Supported model-delivery profiles are:

1. read-only shared filesystem or CSI volume;
2. node-local cached volume populated by a controlled prefetch job;
3. object-store download in an init container; or
4. image-layer embedding for small, immutable test models.

Every path MUST verify the model digest before `mpirun` declares model loading
complete. A URI match is insufficient. Concurrent downloads from all ranks
SHOULD use a node-aware cache to avoid creating a startup traffic storm.

The world MUST record whether weights are shared, duplicated per rank, or
materialized per stage/domain. Kubernetes restarts must not expose partially
written model artifacts as valid cache entries.

### 8.4 Scheduling and admission

Kueue will provide all-or-nothing workload admission. Topology Aware Scheduling
will be used to constrain the gang to an approved node, rack, block, or other
cluster-defined topology level.

The admission request will distinguish:

- hard topology requirements needed for correctness or an accepted performance
  contract;
- soft preferences such as a particular zone or GPU generation; and
- fungible capacity classes that the planner may re-resolve before world
  creation.

The orchestration plan MUST be frozen only after the actual admitted topology
is known. If Kueue admits a different resource flavor than the request was
planned for, Llaminar must recompute and validate the plan before model load.

### 8.5 Disruption policy

A world SHOULD use a PodDisruptionBudget that prevents voluntary partial
eviction while serving. Cluster upgrades and consolidation must drain traffic,
terminate the complete world, and recreate it elsewhere. Priority and
preemption policy apply to the whole admitted workload.

Kubernetes `restartPolicy` and MPIJob cleanup settings must be chosen so that a
failed remote rank does not restart independently inside the old world. The
world supervisor owns the retry budget and always advances the epoch.

## 9. OpenMPI Launch Contract

### 9.1 Launcher responsibilities

The launcher command MUST provide:

- hostfile and exact process count;
- one slot per worker for initial profiles;
- explicit OpenMP thread count and binding variables;
- explicit BLAS thread variables;
- required Llaminar model and orchestration arguments;
- release-appropriate OpenMPI control and data interface selection;
- transport diagnostics required by the network profile;
- a world ID, epoch, and expected fingerprint; and
- a path to the immutable world-spec artifact.

An illustrative command is:

```bash
mpirun \
  -np "${LLAMINAR_WORLD_SIZE}" \
  --hostfile /etc/mpi/hostfile \
  -x OMP_NUM_THREADS \
  -x OMP_PLACES \
  -x OMP_PROC_BIND \
  -x OPENBLAS_NUM_THREADS \
  -x MKL_NUM_THREADS \
  -x LLAMINAR_WORLD_ID \
  -x LLAMINAR_WORLD_EPOCH \
  -x LLAMINAR_WORLD_SPEC \
  /opt/llaminar/bin/llaminar2 serve \
    --config /etc/llaminar/orchestration.yaml \
    --model /models/model.gguf
```

The final binding and MCA arguments are network-profile data, not hard-coded
universal defaults. Container CPU sets, OpenMPI/hwloc behavior, rank layout,
and the selected CNI must be validated together.

### 9.2 SSH and executable symmetry

OpenMPI's SSH launch requires non-interactive access among the hosts involved
in its launch tree. The per-world SSH key MUST be scoped to worker pods in that
world, mounted read-only where practical, and removed when the world is deleted.

OpenMPI binaries, helper executables, shared libraries, Llaminar, NCCL/RCCL,
and device runtime libraries MUST resolve compatibly on launcher and workers.
Using one image digest for all roles is the default way to satisfy this rule.

### 9.3 Thread support and affinity validation

After `MPI_Init_thread`, Llaminar will compare `provided` with the features
enabled by the world profile. A profile that uses the server worker protocol,
background load progress, or concurrent MPI status handling and requires
`MPI_THREAD_MULTIPLE` MUST fail when the implementation supplies less.

At startup, every rank will report:

- MPI version and thread level;
- visible CPU set;
- OpenMP places and binding;
- rank, local rank, pod, and physical-node identity;
- visible GPU UUIDs and device ordinals;
- selected NICs and transport backend; and
- device-to-NUMA relationships.

The launcher will reject duplicate GPU UUID assignment, an unexpected rank
count, missing CPU affinity, or a topology mismatch.

## 10. Network Architecture

### 10.1 Intra-cluster requirements

The Kubernetes network must provide direct pod-to-pod reachability for all MPI
peers in the world. NetworkPolicy must allow:

- launcher-to-worker and worker-to-worker SSH required by OpenMPI launch;
- OpenMPI runtime control traffic;
- MPI TCP traffic when selected;
- NCCL or RCCL socket traffic;
- RDMA traffic and device access when selected;
- DNS resolution for worker endpoints; and
- metrics/logging egress without exposing worker control ports publicly.

The deployment SHOULD constrain dynamic communication ports to an audited range
where supported by the selected OpenMPI and native collective versions. The
exact parameters are part of the versioned network profile because OpenMPI 4
and OpenMPI 5 / PRRTE parameter names and behavior differ.

### 10.2 Multiple interfaces

GPU worker pods may see loopback, default CNI, service, storage, host, and RDMA
interfaces. OpenMPI assumes that selected interfaces in the same address family
are mutually routable. It may attempt unreachable paths when virtual or
cross-cluster interfaces are exposed without constraints.

Each network profile MUST declare:

- control-plane interface or subnet;
- tensor data-plane interface or subnet;
- excluded interfaces;
- IP address family;
- MTU;
- RDMA device and GID selection where applicable;
- NCCL/RCCL interface selection; and
- whether `hostNetwork` or a secondary CNI attachment is required.

Interface selection MUST be verified with a startup connectivity matrix, not
inferred only from interface names.

### 10.3 RDMA and GPUDirect

Clusters promoting multi-node GPU TP SHOULD provide a qualified RDMA stack,
including the relevant GPU and network operators, device plugins, kernel
modules, container permissions, and topology placement.

Qualification MUST prove the actual selected path. The presence of an RDMA
device in the pod is not sufficient evidence that NCCL/RCCL is using RDMA or
that GPUDirect is active. Startup and benchmark artifacts must record the
transport selected by the library.

### 10.4 Cross-cluster networking

A true cross-cluster MPI world would require all of the following:

- stable, direct, bidirectional L3 routing between every participating pod;
- no address overlap or hidden NAT on peer paths;
- cross-cluster DNS or an explicit global hostfile;
- worker-to-worker non-interactive SSH across clusters;
- compatible MTU, OpenMPI, runtime, and device libraries;
- compatible security and network policies;
- release-specific interface and port constraints; and
- failure handling for cluster-level partitions.

Multi-Cluster Services API can provide service discovery when an implementation
is installed, but it does not itself create this network fabric. A CNI cluster
mesh or VPN may create reachability, but reachability alone does not make WAN
collectives suitable for inference.

Consequently, a stretched MPI world remains an explicit research profile with
no production fallback from the fleet-of-worlds architecture.

## 11. Collective and Pipeline Transport Policy

### 11.1 Transport matrix

| Participants | Tensor location | Preferred backend | Fallback policy |
|---|---|---|---|
| One device | Device | No collective | No-op. |
| CUDA devices in one rank | Device | NCCL | Host only for explicit debug/correctness profile. |
| ROCm devices in one rank | Device | RCCL | Host only for explicit debug/correctness profile. |
| CUDA ranks in qualified domain | Device | NCCL multi-process | Fail closed if native transport is required. |
| ROCm ranks in qualified domain | Device | RCCL multi-process | Fail closed if native transport is required. |
| CPU ranks on one node | Host | UPI/shared-memory/MPI domain backend | Profile-selected. |
| CPU ranks across nodes | Host | MPI over qualified transport | No silent network-class downgrade. |
| Heterogeneous domain | Host/device mix | Explicit staged backend | Never selected accidentally by `AUTO`. |
| GPU PP edge | Device | NCCL/RCCL send/receive or qualified device-aware transport | Host MPI only in correctness tier. |

### 11.2 Domain-scoped native communicators

Every TP domain will own a communicator registry entry with:

```cpp
struct DomainCommunicationIdentity {
    int domain_id;
    std::vector<int> world_ranks;
    MPI_Comm mpi_comm;
    CollectiveBackendType backend;
    std::string topology_class;
    std::string transport_profile;
};
```

All ranks must call communicator construction in the same deterministic order.
The NCCL/RCCL unique-ID broadcast and native rank numbering must use the domain
communicator's size and rank, not an unrelated global context.

Startup will run a small, bounded collective qualification for each domain and
record backend, pointer location, bytes, latency, and success. This is a
correctness gate, not a replacement for offline performance qualification.

### 11.3 Device-native pipeline transfer

Introduce an inter-rank transfer abstraction instead of embedding host MPI
semantics in `GlobalPPTransferStage`:

```cpp
class IInterRankTransferBackend {
public:
    virtual TransferHandle sendAsync(
        ITensor &tensor,
        int peer,
        int tag,
        IDeviceContext &device_ctx) = 0;

    virtual TransferHandle recvAsync(
        ITensor &tensor,
        int peer,
        int tag,
        IDeviceContext &device_ctx) = 0;

    virtual bool wait(TransferHandle &handle) = 0;
};
```

Initial implementations are:

- `MPIHostTransferBackend`, preserving the current correctness path;
- `NCCLTransferBackend` for CUDA point-to-point transfer;
- `RCCLTransferBackend` for ROCm point-to-point transfer; and
- an optional qualified device-aware MPI/UCX backend after capability tests.

The graph must express transfer start, dependency, and completion so device work
can overlap communication without violating tensor coherence. A production GPU
transfer must use the active device pointer and update device-side coherence
state without first making the host copy authoritative.

### 11.4 Wide-area pipeline policy

If no single cluster can hold a model, the preferred experimental architecture
is:

```text
cluster A: static MPI world for stages 0..K
                 |
                 | versioned federated activation protocol
                 v
cluster B: static MPI world for stages K+1..L
```

The bridge is an explicit graph transfer boundary with request ID, model
fingerprint, tensor schema, sequence position, deadline, retry classification,
and backpressure. It does not expose one cluster's ranks as members of the
other's MPI world.

Promotion requires microbatching or another latency-hiding mechanism. A
token-at-a-time synchronous WAN PP implementation is expected to have poor
decode latency and is not a default deployment target.

## 12. Multi-Cluster Fleet Architecture

### 12.1 MultiKueue placement semantics

MultiKueue may accept an MPIJob on a management cluster and create its mirror
on one selected worker cluster. The actual launcher and worker pods all run in
that selected cluster. This is the desired capacity-brokering behavior.

The fleet scheduler will select a cluster using:

- GPU vendor, model, count, and memory;
- required rack or fabric topology;
- model-data availability;
- runtime image availability;
- queue time and quota;
- regional request locality;
- qualified network/collective profile; and
- current healthy world count.

The scheduler MUST NOT describe this as splitting an MPIJob across clusters.

### 12.2 Service routing

Every ready world publishes a service record containing:

```text
world_id and epoch
cluster and region
model and tokenizer fingerprint
supported API/features
context and batch limits
health and draining state
queue depth
observed latency/throughput class
cost/capacity metadata
```

The global or regional router sends a complete request to one compatible world
and keeps streaming responses pinned to that world. It must stop assigning new
requests before a world drains. Retrying a partially streamed request on a
different world is an API-level policy and must not be mistaken for transparent
MPI recovery.

### 12.3 Fleet autoscaling

Autoscaling changes the number of complete worlds. It does not change rank
count inside a world.

Scale-out inputs may include queue depth, admission latency, token throughput,
time-to-first-token, active streams, and reserved capacity. Scale-in selects a
whole world, marks it draining, waits for its request policy, removes the
endpoint, and deletes the MPIJob.

Cold model-load time must be included in capacity policy. The fleet may keep
warm spare worlds or node-local model caches, but a pod is not ready merely
because its GPU allocation exists.

### 12.4 Availability model

High availability is achieved with at least two independent worlds in separate
failure domains. A single large world has the availability of its least
reliable rank and network path. The router, not MPI, provides failover between
worlds.

Worlds in different regions SHOULD be independently routable so a management
cluster or inter-cluster control-plane outage does not immediately terminate
already serving local traffic.

## 13. Lifecycle, Failure, and Upgrade Semantics

### 13.1 Failure classes

| Failure | Required response |
|---|---|
| Rank process exits | Fail `mpirun`, mark world failed, remove endpoint, recreate whole world. |
| Worker pod evicted or lost | Same as rank exit; no in-place communicator repair. |
| Launcher exits | Mark world failed even if remote processes appear alive; terminate remnants. |
| Collective timeout | Abort world, preserve diagnostics, recreate with new epoch subject to retry policy. |
| Rank fingerprint mismatch | Fail startup permanently until spec/image/model is corrected. |
| Network qualification failure | Fail or return to pending on a different admitted topology; never serve degraded silently. |
| Rank 0 HTTP failure | Remove endpoint and fail world; workers must not continue indefinitely. |
| Cluster partition | Each affected world fails locally; global router removes unreachable endpoints. |
| Model corruption | Quarantine cache artifact and fail startup. |

### 13.2 Retry policy

The controller distinguishes transient infrastructure failures from immutable
specification failures. Transient failures use bounded exponential backoff and
a fresh epoch. Model digest mismatch, incompatible runtime ABI, invalid
orchestration config, or repeated topology qualification failure enters a
terminal condition that requires an updated spec or operator action.

World status must retain the previous epoch's exit reason, rank, phase, pod,
node, and log reference through replacement.

### 13.3 Graceful drain

Drain order is:

1. mark world `Draining`;
2. remove it from new-request routing;
3. finish or cancel active requests according to the configured deadline;
4. send coordinated shutdown to MPI workers;
5. flush telemetry;
6. allow `mpirun` to exit;
7. remove rank-0 endpoint; and
8. delete pods and per-world credentials.

A forced deadline may abort the world. Shutdown must be idempotent because
Kubernetes termination signals can race with an administrative drain.

### 13.4 Rolling upgrades

Build, model, configuration, OpenMPI, NCCL/RCCL, driver-sensitive network
profile, and graph-plan changes create a new world fingerprint. Upgrades use
blue/green world replacement:

```text
old world serving
  -> create new epoch/fingerprint
  -> qualify and mark new world ready
  -> shift traffic
  -> drain old world
  -> delete old world
```

Ranks within one world are never rolled independently.

## 14. Security Model

### 14.1 Pod identity and privilege

Launcher and workers SHOULD run as a non-root UID with a read-only root
filesystem where compatible with GPU and MPI requirements. Capabilities and
host mounts must be minimal and declared by the network/GPU profile.
`hostNetwork`, host PID, privileged containers, and broad `/dev` mounts are not
defaults; any requirement must be justified and isolated to a qualified
profile.

### 14.2 SSH credentials

MPI launch credentials are per-world ephemeral secrets. They MUST NOT be shared
across namespaces or long-lived fleet-wide. `authorized_keys` should restrict
use to the worker image and network policy boundary. Secret material must not
appear in logs, world fingerprints, or status objects.

### 14.3 Network policy

Worker control and collective ports are reachable only from pods in the same
world and required monitoring agents. Only the rank-0 inference port is exposed
through the serving path. Administrative shutdown endpoints remain disabled or
protected by a separate authenticated management plane.

Cross-cluster routing uses authenticated service-to-service transport. It does
not expose OpenMPI worker ports to public ingress.

### 14.4 Supply chain and model integrity

World templates SHOULD pin image digests and record SBOM/provenance references.
Model objects and orchestration configs MUST be digest verified. Runtime status
must identify digests, not mutable tags or bucket paths alone.

## 15. Observability and Operations

### 15.1 Required identity on every signal

Logs, metrics, traces, events, and benchmark artifacts MUST be joinable by:

- world ID and epoch;
- world fingerprint;
- cluster, region, zone, rack/block, and physical node UID;
- namespace, MPIJob, pod UID, and pod name;
- MPI world rank, domain rank, and local rank;
- model/build/config digests; and
- device UUID and selected communication backend where relevant.

High-cardinality identifiers belong in logs and traces or carefully controlled
metric labels; Prometheus cardinality limits remain explicit.

### 15.2 Startup telemetry

Startup status will expose duration and outcome for:

- scheduling and gang admission;
- image pull and model staging;
- SSH reachability;
- MPI initialization;
- inventory gather;
- topology/placement validation;
- weight loading and sharding;
- communicator creation;
- collective qualification;
- graph construction/capture; and
- world readiness.

The slowest rank and phase must be visible.

### 15.3 Runtime telemetry

At minimum, collect:

- request count, active request, queue time, time to first token, inter-token
  latency, tokens per second, and error class;
- command sequence and coordinated-worker wait time;
- per-domain collective operation, bytes, duration, timeout, and backend;
- PP transfer bytes, duration, queueing, overlap, and backend;
- GPU-to-host and host-to-GPU staging bytes;
- GPU memory, host memory, KV cache, prefix cache, and model-residency state;
- world ready, draining, failed, and restart counters; and
- unexpected transport fallback count.

A device-native-required world must alert and fail its readiness gate if staging
telemetry becomes nonzero on a promoted TP or PP edge.

### 15.4 Diagnostic bundle

Every failed world should produce a bounded diagnostic bundle with:

- frozen world spec and fingerprint;
- MPI and collective library versions;
- sanitized environment relevant to affinity and networking;
- rank-to-pod/node/device/NIC mapping;
- per-rank last completed lifecycle phase;
- launcher and rank logs;
- Kubernetes events and termination reasons;
- recent collective/transfer timeline; and
- selected transport diagnostics.

Secrets, tokens, and model contents are excluded.

## 16. Required Llaminar Runtime Work

### 16.1 Kubernetes-aware deployment identity

Extend the inventory contract with a deployment identity that remains separate
from hardware discovery:

```cpp
struct DeploymentIdentity {
    std::string cluster_id;
    std::string region;
    std::string zone;
    std::string rack_or_block;
    std::string kubernetes_node_name;
    std::string kubernetes_node_uid;
    std::string namespace_name;
    std::string pod_name;
    std::string pod_uid;
    std::string world_id;
    std::string world_epoch;
};
```

Values will be injected through the Downward API, trusted node labels copied by
an admission/controller layer, and an immutable ConfigMap or projected volume.
The planner must use physical node UID for node grouping when it is present,
while retaining the MPI processor name for connection diagnostics.

Inventory serialization, equality, printer output, dry-run output, plan
digests, and tests must include the new topology fields.

### 16.2 External launch contract validation

Add a startup component that:

1. recognizes an externally managed MPI world;
2. loads the expected world spec;
3. verifies rank count and epoch;
4. validates MPI thread support;
5. validates OpenMP/BLAS affinity;
6. gathers runtime and model fingerprints;
7. detects duplicate or missing device UUIDs;
8. verifies network profile identity; and
9. emits one deterministic launch manifest.

Dry-run and `--explain-placement` output should show both Kubernetes admission
identity and the final Llaminar plan.

### 16.3 Coordinated command protocol hardening

Promote the server worker commands from implicit ordered broadcasts to a
versioned envelope:

```cpp
struct MPICommandEnvelope {
    uint32_t protocol_version;
    uint64_t world_epoch_hash;
    uint64_t request_sequence;
    uint32_t command;
    uint32_t payload_bytes;
    uint64_t payload_checksum;
};
```

All ranks validate epoch, monotonic sequence, command, and payload shape before
executing graph work. A mismatch aborts the world rather than risking a
different collective order.

Implement equivalent prefill, decode, cache, sampling, error, and shutdown
semantics in `GlobalOrchestratorRunner`. Factor shared protocol code so ordinary
and global runners cannot drift.

### 16.4 Whole-world readiness

Add a lifecycle coordinator owned by the runner/application boundary. It
collects per-rank phase results and exposes immutable failure details. Rank 0
may bind its socket before the last phase for probe visibility, but `/readyz`
must remain false and the Service endpoint must remain unpublished.

### 16.5 Domain-native GPU collectives

Audit every global/named-domain path from topology compilation through
`DomainCommunicatorRegistry`, `GlobalTPContext`, `CollectiveContext`, and
`BackendRouter`. Add fail-closed backend requirements and route telemetry.

The acceptance test must prove that a device-resident buffer remains
device-resident across the collective and that the selected NCCL/RCCL
communicator uses the intended domain size and rank order.

### 16.6 Asynchronous inter-rank transfer

Refactor `GlobalPPTransferStage` around `IInterRankTransferBackend`. Preserve the
host implementation for CPU and correctness tests. Add native CUDA and ROCm
implementations, explicit graph dependencies, stream/event integration,
timeout/error propagation, and staging telemetry.

### 16.7 Request concurrency and pipeline utilization

The initial server remains serialized. Before multi-stage PP can claim
throughput scaling across slower links, add a separate, measured design for
microbatching or multiple in-flight sequences. That design must preserve KV
cache ownership, sampling order, MTP state, prefix-cache identity, coordinated
command order, and streaming response correctness.

This work is not required for the first MPIJob serving milestone, but it is a
promotion dependency for federated PP.

## 17. Packaging and API Surface

### 17.1 Initial repository artifacts

The first implementation should add:

```text
deploy/kubernetes/
  base/
    mpijob.yaml
    rank0-service.yaml
    orchestration-config.yaml
    network-policy.yaml
  overlays/
    cuda-single-node/
    cuda-rdma/
    rocm-single-node/
    rocm-rdma/
  tests/
    kind/
    conformance/
```

Helm may be added if parameterization becomes unwieldy, but rendered manifests
must remain reviewable and conformance tested.

### 17.2 Optional `LlaminarWorld` API

Once world-level lifecycle behavior is stable on raw MPIJobs, introduce a thin
custom resource that compiles into MPIJob, Service/EndpointSlice, ConfigMap,
Secret, Kueue labels, and status:

```yaml
apiVersion: inference.llaminar.ai/v1alpha1
kind: LlaminarWorld
metadata:
  name: qwen-world-a
spec:
  replicas: 4
  slotsPerWorker: 1
  rankLayout: one-rank-per-island
  model:
    uri: pvc://models/qwen/model.gguf
    digest: sha256:<model-digest>
  runtime:
    image: registry.example/llaminar@sha256:<image-digest>
    configRef: qwen-orchestration
  resources:
    worker:
      gpuResource: nvidia.com/gpu
      gpus: 8
      cpu: "32"
      memory: 256Gi
  topology:
    requiredLevel: kubernetes.io/hostname
  communication:
    collective: nccl
    networkProfile: cuda-rdma-v1
    requireDeviceNative: true
  service:
    port: 8080
  queueName: gpu-production
```

This is a conceptual API, not a committed schema. The controller must not
duplicate MPI Operator or Kueue scheduling logic. Its value is world
fingerprinting, rank-0 endpoint publication, readiness aggregation, full-world
restart, drain, and status.

### 17.3 Status contract

Suggested status fields are:

```yaml
status:
  observedGeneration: 7
  worldID: qwen-world-a
  epoch: 4f9c...
  fingerprint: sha256:...
  phase: Serving
  readyRanks: 4
  expectedRanks: 4
  rank0Endpoint: 10.42.7.18:8080
  admittedCluster: gpu-west-1
  admittedTopology: block-17
  transport:
    tp: nccl
    pp: nccl-send-recv
    degraded: false
  conditions: []
```

Status updates must be reconstructable after controller restart from MPIJob,
pod, endpoint, and rank-0 world-status evidence.

## 18. Testing and Qualification Plan

### 18.1 Unit tests

Add unit coverage for:

- external MPI environment detection without recursive bootstrap;
- world-spec canonicalization and fingerprint equality;
- MPI thread-level acceptance;
- Kubernetes deployment-identity parsing;
- physical-node grouping independent of pod hostname;
- rank/pod/device UUID uniqueness;
- topology requirement validation;
- domain communicator membership and deterministic creation order;
- native-required backend fail-closed behavior;
- command-envelope epoch, sequence, length, and checksum rejection;
- lifecycle state transitions and terminal error classification;
- rank-0 endpoint eligibility; and
- drain/retry/epoch rules.

### 18.2 Local multi-process tests

Using a normal Ninja Integration build, run two- and four-rank tests under
OpenMPI for:

- startup fingerprint agreement and mismatch;
- coordinated prefill/decode/shutdown;
- ordinary and global runner command parity;
- rank loss and launcher nonzero exit;
- communicator isolation across multiple named domains;
- CPU MPI PP transfer parity; and
- injected command-order mismatch causing bounded world abort.

Normal tests and production runs must not use `--no-mpi-bootstrap`.

### 18.3 Kubernetes control-plane tests

A CPU-only kind or equivalent test cluster can verify:

- MPIJob rendering and admission;
- launcher/worker SSH and hostfile behavior;
- one slot per worker;
- rank-0 Service endpoint isolation;
- readiness staying false until all ranks report;
- complete-world restart after worker deletion;
- endpoint removal during failure/drain;
- NetworkPolicy shape;
- Kueue suspension/admission; and
- immutable epoch/fingerprint status.

Mock GPU resources may validate templates, but they do not qualify GPU
communication.

### 18.4 Single-cluster GPU conformance

Run conformance on both CUDA and ROCm where supported:

1. one rank, one GPU;
2. one rank, multiple local GPUs;
3. multiple ranks on one physical node;
4. two physical nodes on the standard pod network;
5. two physical nodes on the qualified RDMA profile;
6. global TP with native collectives;
7. global PP with device-native transfer; and
8. named-domain PP+TP serving after the global worker loop is complete.

Each case covers one-shot parity before server parity. The test records device
residency, backend route, transport diagnostics, and rank placement.

### 18.5 Fault injection

Inject at least:

- worker process exit;
- worker pod deletion;
- launcher deletion;
- rank-0 server termination;
- collective timeout;
- blocked peer port;
- DNS failure;
- model digest mismatch on one rank;
- duplicate GPU assignment;
- node drain;
- network partition; and
- stale endpoint from a previous epoch.

No test may pass because an individual rank silently rejoins or because traffic
continues to a partially alive world.

### 18.6 Multi-cluster tests

Use at least two workload clusters to prove:

- MultiKueue places the complete MPIJob in exactly one worker cluster;
- both clusters can host independent worlds for the same model;
- the router only selects ready compatible fingerprints;
- cluster loss removes only affected worlds;
- traffic remains pinned for streaming requests;
- scale-out creates whole worlds; and
- blue/green model or runtime updates do not mix rank versions.

The optional federated-pipeline experiment has a separate test and performance
report and cannot weaken the independent-world gates.

### 18.7 Performance gates

On identical hardware and topology, compare direct OpenMPI launch with the
Kubernetes MPIJob:

- model output parity must be unchanged;
- median prefill and decode throughput regression should be no more than 5%;
- p99 time-to-first-token and inter-token latency regression should be no more
  than 10%, with a documented variance method;
- native collective bandwidth should retain at least 90% of the qualified
  direct-container baseline;
- a native-required GPU TP/PP case must report zero host-staging bytes on the
  measured edge; and
- no unplanned backend or network fallback is permitted.

Platform-specific thresholds may be tightened. Any exception requires a
recorded cause and cannot be hidden by comparing different rank placement,
power state, model cache state, or transport.

## 19. Phased Implementation Plan

### Phase 0: Freeze contracts and baseline evidence

Implementation:

- commit the world, epoch, launch, readiness, failure, and topology contracts;
- add direct OpenMPI baseline scripts and result schema;
- record current ordinary and global runner behavior;
- create the initial deployment directory and manifest validation; and
- define CUDA and ROCm network-profile inputs without claiming qualification.

Exit criteria:

- the two initial rank layouts and failure semantics are unambiguous;
- a direct two-rank ordinary runner baseline is reproducible; and
- all Kubernetes artifacts render without using `--no-mpi-bootstrap`.

### Phase 1: Single-cluster MPIJob correctness

Implementation:

- build one immutable launcher/worker image;
- run CPU and one-GPU `oneshot` through `MPIJob`;
- run two-rank ordinary inference;
- explicitly pass OpenMP/BLAS/MPI environment;
- expose rank 0 only; and
- add basic MPIJob logs and status collection.

Exit criteria:

- MPI Operator launches Llaminar without recursive bootstrap;
- one-shot outputs match direct execution;
- the Service never targets non-root ranks; and
- launcher failure accurately reflects remote-rank failure.

### Phase 2: World readiness and recovery

Implementation:

- add world spec, epoch, fingerprint, and lifecycle coordinator;
- validate MPI thread level and runtime symmetry;
- add `/livez` and whole-world `/readyz`;
- publish/remove the rank-0 endpoint based on epoch-aware readiness;
- implement full-world retry and drain behavior; and
- add fault-injection tests.

Exit criteria:

- no partially initialized world receives traffic;
- any rank loss removes the endpoint and replaces the whole world; and
- diagnostics identify the failing rank and phase.

### Phase 3: Kubernetes topology and gang scheduling

Implementation:

- extend `ClusterInventory` with Kubernetes identity;
- integrate Kueue gang admission;
- integrate Topology Aware Scheduling constraints;
- validate CPU sets, NUMA, physical node, GPU UUIDs, and NIC placement; and
- freeze orchestration plans after admission.

Exit criteria:

- pod hostnames are never treated as authoritative physical-node IDs;
- requested topology constraints are proved by runtime inventory; and
- a nonconforming placement fails before model serving.

### Phase 4: Native multi-rank GPU tensor parallelism

Implementation:

- audit domain communicator propagation;
- enforce NCCL/RCCL for native-required homogeneous global TP;
- add collective route and staging telemetry;
- qualify pod-network and RDMA profiles; and
- run model parity and direct-versus-Kubernetes performance gates.

Exit criteria:

- CUDA and/or ROCm global TP, according to available hardware, uses a
  domain-scoped native communicator;
- no host staging occurs on the promoted collective path; and
- performance gates pass on at least two physical nodes.

### Phase 5: Device-native pipeline transport and global serve

Implementation:

- add `IInterRankTransferBackend`;
- implement CUDA/ROCm device-native asynchronous PP transfer;
- make tensor coherence and graph dependencies explicit;
- complete `GlobalOrchestratorRunner` worker and shutdown protocols;
- share the hardened command envelope across runners; and
- certify named-domain PP+TP server parity.

Exit criteria:

- promoted GPU PP edges do not call host `data()`/`mutable_data()` for transfer;
- ordinary and global coordinated serving pass the same lifecycle suite; and
- shutdown and failure cannot leave workers blocked in an old epoch.

### Phase 6: Multi-cluster fleet placement

Implementation:

- integrate MultiKueue for complete-world placement;
- publish world capability records;
- add regional/global routing and drain behavior;
- autoscale complete worlds; and
- test cluster outage and blue/green replacement.

Exit criteria:

- a world always resides in one selected workload cluster;
- two clusters can serve compatible independent worlds;
- cluster loss does not corrupt or block healthy worlds elsewhere; and
- fleet scale changes world count rather than live rank membership.

### Phase 7: Optional `LlaminarWorld` controller

Implementation:

- introduce the smallest CRD needed for world lifecycle;
- compile it into MPIJob and standard Kubernetes resources;
- own epoch, fingerprint, rank-0 endpoint, drain, retry, and status;
- preserve Kueue and MPI Operator authority; and
- add controller restart/reconciliation tests.

Exit criteria:

- raw MPIJob remains a documented escape hatch;
- reconciliation never creates mixed epochs or duplicate endpoints; and
- deleting the CR cleans up world-scoped credentials and remnants.

### Phase 8: Federated pipeline research gate

Implementation:

- define a versioned activation bridge between independent worlds;
- add backpressure, deadline, cancellation, and tensor-schema validation;
- add microbatching/latency hiding as required; and
- measure same-cluster, cross-cluster same-region, and cross-region behavior.

Exit criteria for any production promotion:

- model correctness and failure isolation pass;
- partial request failure has an explicit API behavior;
- throughput or capacity benefit justifies latency and operational cost; and
- independent-world routing remains the default.

## 20. Acceptance Matrix

| Capability | Correctness | Placement | Failure | Performance | Production state |
|---|---|---|---|---|---|
| Single-rank MPIJob | Required | GPU/CPU identity | Pod/launcher exit | Direct parity | Phase 1 |
| Multi-rank ordinary serve | Required | One slot per worker | Any rank fails world | Direct parity | Phase 2 |
| Topology-aware world | Required | Node/rack/block verified | Admission retry | Placement baseline | Phase 3 |
| Global CUDA TP | Model parity | Qualified domain | Collective abort | NCCL/no staging | Phase 4 |
| Global ROCm TP | Model parity | Qualified domain | Collective abort | RCCL/no staging | Phase 4 |
| GPU global PP | Model parity | Qualified edge | Transfer timeout | Native async/no staging | Phase 5 |
| Named-domain serve | Request parity | Domain-specific | Coordinated shutdown | Profile-specific | Phase 5 |
| Multi-cluster fleet | API parity | Whole job in one cluster | Cluster isolation | Routing/SLO | Phase 6 |
| Federated PP | Full model parity | Explicit bridge | Independent-world isolation | Separate research gate | Phase 8 only |
| Stretched MPI world | Research only | Global routability | Whole-world abort | No production target | Unsupported default |

## 21. Risks and Mitigations

### 21.1 World size reduces availability

Every added rank and network path increases the chance that the whole world
fails. Keep TP inside the smallest useful domain, prefer PP or more independent
worlds when practical, and measure failure rate by world size.

### 21.2 Kubernetes topology labels can be incomplete or untrusted

Do not trust arbitrary pod-provided labels for physical placement. Copy an
allowlist through a controller/admission path, compare with runtime device and
NUMA discovery, and retain node UID as the stable physical identity.

### 21.3 CNI abstraction can hide the actual data path

Require startup diagnostics and offline collective benchmarks for every network
profile. Profile names are immutable contracts tied to CNI, driver, firmware,
OpenMPI, NCCL/RCCL, MTU, and node image versions.

### 21.4 MPI Operator is training/job oriented

Long-running inference can still use its launcher supervision and worker model,
but readiness, rank-0 Service ownership, drain, and world replacement require
additional lifecycle logic. Add that logic incrementally rather than forking
the MPI Operator.

### 21.5 Host-staging fallback hides performance defects

Make staging bytes and backend choice observable. Native-required profiles fail
closed. Keep host MPI only as an explicit correctness tier.

### 21.6 Serialized serving underutilizes pipeline stages

Scale replicas first. Treat microbatching and multi-request state as a distinct
runtime project with parity and cache-ownership gates before relying on PP for
throughput over slower links.

### 21.7 Model loading can dominate recovery

Use immutable node-local caches, prefetching, and warm capacity. Do not weaken
digest verification or reuse a half-written cache merely to reduce restart
time.

### 21.8 Multi-cluster control can become a single point of failure

Keep serving worlds cluster-local and capable of continuing already admitted
traffic during management-plane interruption. Replicate routers and define
bounded stale-record behavior.

## 22. Open Decisions

The project must resolve these with measured prototypes:

1. Whether one rank per GPU island or one rank per GPU is the default for each
   supported model and platform.
2. Whether rank-0 endpoint publication is handled by a small sidecar, an
   EndpointSlice controller, or the later `LlaminarWorld` controller.
3. Which Kueue topology levels map to GPU island, rack, and network fabric on
   each target cluster.
4. Whether native PP should use NCCL/RCCL point-to-point first or a common
   device-aware MPI/UCX abstraction first.
5. How to represent transport-profile compatibility in the orchestration plan
   digest.
6. Whether background world-health communication requires a dedicated MPI
   communicator/thread or should rely solely on command-bound acknowledgements
   and `mpirun` process supervision.
7. The exact request-drain policy for long streaming responses.
8. Whether model cache population is owned by a DaemonSet, CSI driver, prefetch
   Job, or platform-specific cache service.
9. Which first target environment supplies both multi-node CUDA and ROCm
   conformance hardware.
10. Whether the federated pipeline has a real capacity use case that justifies
    its complexity after cluster-local placement and quantization options are
    exhausted.

None of these decisions changes the central architecture: static cluster-local
worlds are the unit of execution, failure, and scale.

## 23. Definition of Done

This project is complete when:

1. Llaminar has supported, versioned CUDA and ROCm Kubernetes deployment
   profiles, subject to available qualification hardware.
2. An MPIJob can run one-shot and server inference with externally launched
   OpenMPI and without `--no-mpi-bootstrap`.
3. Rank 0 is the only routable HTTP endpoint and its readiness represents all
   ranks in the current epoch.
4. Kubernetes, MPI, physical-node, device, and NIC identities are distinct and
   validated in `ClusterInventory`.
5. Kueue admits the full rank gang and topology placement is verified at
   runtime.
6. Any rank or launcher failure removes the world from service and recreates
   the entire world with a new epoch.
7. Global CUDA and ROCm TP use domain-scoped NCCL/RCCL without hidden host
   staging on their promoted profiles.
8. Global GPU PP has a device-native asynchronous path and the host path is an
   explicit fallback tier.
9. `GlobalOrchestratorRunner` implements coordinated server work and shutdown
   with the same protocol guarantees as the ordinary runner.
10. Direct OpenMPI and Kubernetes parity, fault, and performance gates pass.
11. MultiKueue can place whole worlds into at least two workload clusters, and
    a router can serve and drain those independent worlds safely.
12. Operational dashboards and diagnostic bundles identify topology, backend,
    staging, rank failure, and world epoch without pod inspection.
13. Documentation clearly labels stretched cross-cluster MPI as unsupported by
    default and federated PP as experimental until its independent gate passes.

The durable design rule is:

> Keep fast, token-synchronous collectives inside a qualified topology domain;
> make broader communication an explicit graph transfer or whole-request
> routing boundary.

## Appendix A: Illustrative MPIJob

This manifest is intentionally a design example. Resource names, labels,
security context, MPI arguments, ports, and topology constraints must be
rendered from a qualified platform profile.

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: llaminar-qwen
  labels:
    kueue.x-k8s.io/queue-name: gpu-production
spec:
  slotsPerWorker: 1
  sshAuthMountPath: /home/llaminar/.ssh
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: mpi-launcher
              image: registry.example/llaminar@sha256:<image-digest>
              securityContext:
                runAsNonRoot: true
                runAsUser: 1000
              command: ["mpirun"]
              args:
                - "-np"
                - "4"
                - "--hostfile"
                - "/etc/mpi/hostfile"
                - "-x"
                - "OMP_NUM_THREADS"
                - "-x"
                - "OMP_PLACES"
                - "-x"
                - "OMP_PROC_BIND"
                - "-x"
                - "LLAMINAR_WORLD_ID"
                - "-x"
                - "LLAMINAR_WORLD_EPOCH"
                - "/opt/llaminar/bin/llaminar2"
                - "serve"
                - "--config"
                - "/etc/llaminar/orchestration.yaml"
                - "--model"
                - "/models/model.gguf"
              envFrom:
                - configMapRef:
                    name: llaminar-qwen-world
              resources:
                requests:
                  cpu: "2"
                  memory: 2Gi
                limits:
                  cpu: "2"
                  memory: 2Gi
    Worker:
      replicas: 4
      template:
        metadata:
          labels:
            app.kubernetes.io/name: llaminar
            inference.llaminar.ai/world: llaminar-qwen
        spec:
          restartPolicy: Never
          terminationGracePeriodSeconds: 120
          containers:
            - name: mpi-worker
              image: registry.example/llaminar@sha256:<image-digest>
              securityContext:
                runAsNonRoot: true
                runAsUser: 1000
              command: ["/usr/sbin/sshd"]
              args: ["-De", "-f", "/home/llaminar/.sshd_config"]
              envFrom:
                - configMapRef:
                    name: llaminar-qwen-world
              resources:
                requests:
                  cpu: "32"
                  memory: 256Gi
                  nvidia.com/gpu: "8"
                limits:
                  cpu: "32"
                  memory: 256Gi
                  nvidia.com/gpu: "8"
              volumeMounts:
                - name: models
                  mountPath: /models
                  readOnly: true
                - name: orchestration
                  mountPath: /etc/llaminar
                  readOnly: true
          volumes:
            - name: models
              persistentVolumeClaim:
                claimName: model-store
            - name: orchestration
              configMap:
                name: llaminar-qwen-orchestration
```

The rank-0 Service is omitted because endpoint ownership must be tied to proven
runtime rank identity. A platform-specific prototype may use the worker replica
index only after verifying the MPI Operator's host ordering, but the production
contract remains runtime-validated rank 0.

## Appendix B: Example World Fingerprint Input

```json
{
  "schema_version": 1,
  "world_id": "llaminar-qwen",
  "epoch": "4f9c3d10-1d73-4ce8-aac8-8c7cce3c7647",
  "image_digest": "sha256:<image>",
  "build_id": "<llaminar-build>",
  "model_digest": "sha256:<model>",
  "config_digest": "sha256:<config>",
  "world_size": 4,
  "rank_layout": "one-rank-per-island",
  "required_topology": "fabric-block",
  "collective_profile": "cuda-nccl-rdma-v1",
  "openmpi": "5.0.x:<build-id>",
  "nccl": "<version>",
  "ranks": [
    {
      "rank": 0,
      "cluster_id": "gpu-west-1",
      "node_uid": "<node-uid>",
      "pod_uid": "<pod-uid>",
      "device_uuids": ["GPU-..."]
    }
  ]
}
```

Pod UIDs are observed evidence and should not be included in the desired-spec
digest if that would prevent deterministic pre-launch planning. The final
runtime fingerprint may contain both a desired-spec digest and an observed
placement digest.

## Appendix C: Initial Code Surface Inventory

| Area | Primary files |
|---|---|
| Bootstrap and external MPI detection | `src/v2/app/MPIBootstrapPhase.cpp`, `src/v2/utils/MPIBootstrap.cpp` |
| MPI initialization and shutdown | `src/v2/app/RuntimeInitPhase.cpp`, `src/v2/app/commands/CommandMPI.cpp`, `src/v2/app/MPIShutdown.cpp` |
| Server rank behavior and health | `src/v2/app/modes/ServerMode.cpp` |
| Ordinary coordinated worker protocol | `src/v2/execution/runner/OrchestrationRunner.cpp` |
| Global coordinated runner gap | `src/v2/execution/global/GlobalOrchestratorRunner.cpp` |
| Inventory and topology | `src/v2/execution/mpi_orchestration/DeviceInventory.h`, `src/v2/planning/ClusterInventoryGatherer.cpp`, `src/v2/utils/MPITopology.cpp`, `src/v2/utils/NodeDetection.cpp` |
| Domain communicators | `src/v2/execution/global/DomainCommunicatorRegistry.cpp`, `src/v2/collective/GlobalTPContext.cpp`, `src/v2/utils/MPIContext.h` |
| Collective routing | `src/v2/collective/BackendRouter.cpp`, `src/v2/execution/local_execution/collective/CollectiveContext.cpp` |
| CUDA native collectives | `src/v2/collective/backends/NCCLBackend.cpp`, `src/v2/collective/backends/NCCLBackendCUDA.cu` |
| ROCm native collectives | `src/v2/collective/backends/RCCLBackendHIP.cpp` and RCCL dynamic-loader files |
| Global pipeline transfer | `src/v2/execution/compute_stages/stages/GlobalPPTransferStage.cpp` |
| Placement planning | `src/v2/planning/`, `src/v2/config/OrchestrationConfig.*` |

## Appendix D: External References

These references describe the external behavior assumed by this proposal. The
repository should pin tested component versions in deployment profiles rather
than treating a moving documentation page as the runtime contract.

- [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator)
- [MPI Operator `v2beta1` example](https://raw.githubusercontent.com/kubeflow/mpi-operator/master/examples/v2beta1/pi/pi.yaml)
- [OpenMPI SSH launch requirements](https://docs.open-mpi.org/en/v5.0.x/launching-apps/ssh.html)
- [OpenMPI TCP and interface selection](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/tcp.html)
- [OpenMPI InfiniBand and RoCE guidance](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/ib-and-roce.html)
- [Kubernetes Services, Load Balancing, and Networking](https://kubernetes.io/docs/concepts/services-networking/)
- [Kubernetes DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)
- [Kubernetes device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage/device-plugins/)
- [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)
- [Kueue Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/tasks/run/topology_aware_scheduling/)
- [Kueue MultiKueue MPIJob behavior](https://kueue.sigs.k8s.io/docs/tasks/run/multikueue/mpijob/)
- [Multi-Cluster Services API concepts](https://multicluster.sigs.k8s.io/concepts/multicluster-services-api/)
- [Multi-Cluster Services API implementations](https://multicluster.sigs.k8s.io/implementations/mcs-implementations/)
- [NVIDIA Network Operator deployment](https://docs.nvidia.com/networking/display/kubernetes25100/deployment-guide-kubernetes.html)
- [NVIDIA GPU Operator RDMA guidance](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
- [AMD Kubernetes device plugin](https://instinct.docs.amd.com/projects/k8s-device-plugin/en/latest/index.html)
- [AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/)
