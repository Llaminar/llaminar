# QFN pooled ngrams across continuation topologies

Implementation contract, updated 2026-11-01. The isolated integration tree implements ordinary rank-local cross-vendor storage behind a singleton GPU continuation and CPU expert domain. All four static-expert, MTP-Off retained15 source controls pass loading and final retirement in r65. The four corresponding frozen-history panels now pass their required groups, actual captured inference, exact reset and final retirement; see the [panel result and explicit numerical-policy decision](QFN_POOLED_NGRAM_PANEL_2026-11-01.md). Dynamic balancing, active MTP, performance and complete preflight remain unqualified for this new path. This note records the complete target; source admission alone does not qualify inference.

The isolated integration tree includes the [source/fill component](QFN_POOLED_SOURCE_FILL_PLAN.md),
mixed-local Pin/Free/Victim, publication/acquisition/Ready, original-order hit
delivery, and terminal release/retirement bodies. Twelve focused host checks
pass 3,329 case executions; original worker and graph producers are exercised
with device API doubles. CUDA SM80/86/89/90 and HIP gfx906/gfx1100 leaf
compilation passes the canonical spill guard. This is component evidence,
not native capture or serving qualification.
[Completion plan](QFN_POOLED_COMPLETION_PLAN.md).

Two subsequent focused corrections also pass in the joined tree: public remote
row packing rejects local owners without an ingress/payload family, while the
original local read remains valid; zero-slot release import ignores inactive
serial count storage and still requires actual physical retirement. Both have
original-code failing controls and explicit production-preflight regressions.
Evidence: `/tmp/qfn-ngram-read-schema-joined-r1/` and
`/tmp/qfn-ngram-terminal-c0-joined-r1/`.

The isolated integration tree combines all-remote and mixed local/remote
owner components. Direct local edges retain actual producer and receipt
dependencies without a self-channel. The reader retains private source misses
and transfers only live hit rows; a single remote owner is covered independently
of the coordinator. Terminal graph wiring preserves the actual last-reader
edge before releasing pins. The ordinary cross-vendor path retains original
remote native programs and passes full source admission. Host-component and
source passes alone do not qualify serving, active MTP or multi-consumer capture.

The existing memory composer now has a private multi-owner overload, charging
each physical GPU separately and the coordinator/source once through PMA.
Focused tests cover 1/2/4/8 owners, unequal/C0 capacity, mixed backend metadata,
query buckets through M2048, exact fit and one-byte-short rejection. The
original single-owner composer agrees in 120 metadata combinations, including
the merged publication version; all four production consumers compile. Directed
transport still supplies its own BOM; planner/factory eligibility stays guarded.
Evidence: `/tmp/qfn-ngram-cohort-proof-r1/`,
`/tmp/qfn-ngram-terminal-joined-proof-r1/`,
`/tmp/qfn-ngram-read-terminal-device-compile-r1/`, `/tmp/qfn-ngram-rollout-r1/`.

## User-visible completion criteria

The end goal is fast, numerically qualified QFN inference on the available CPU,
CUDA and ROCm hardware, including composed topologies and flexible ngram tiers.
Admission accounting is supporting infrastructure. A successful load does not
qualify a configuration for users, and another accounting refinement is not a
substitute for running the admitted configuration.

The immediate functional/performance campaign is the user-selected
[six-row hardware target](QFN_FUNCTIONAL_PERFORMANCE_TARGET_2026-11-02.md):
MI50 counts 1/2/4, RTX 3090 counts 1/2, and RTX 3090×2 continuation with all four
MI50s doing follower expert work. Cross these with RAM/SSD backing and bounded
RAM/VRAM row-cache policies, including observed promotion/demotion. The detailed
note distinguishes cheap full-product metadata checks from complementary real
execution, numerical and performance projections. Broader PP/cross-rank/mixed-TP
capabilities below remain future contract coverage, not extra rows in this
immediate campaign. Existing CPU/backend regressions remain in force.

Use the existing canonical typed matrix as the configuration authority. For each
configuration being qualified, keep these results distinct and traceable to its
exact model, topology, placement, arithmetic and runtime policy:

| Check | Required evidence |
|---|---|
| Loading | Real model admission, declared compute/storage owners, fit or an intentional capacity rejection, bounded resources and complete cleanup. |
| Serving | Actual public inference, captured execution, repeated requests, reset and prefix restoration; ngram hits, misses, promotions and evictions where selected. |
| Numerical accuracy | Frozen-history panels with separate prefill/decode mean KL, top-1 agreement and top-5 overlap; retain exact-history layer/stage replay for diagnosis. |
| Dynamic experts | Actual migration decisions, published owner generations and live transfer extents, with outputs and resources correct during lookup work. Merely enabling the option is insufficient. |
| MTP | Off, fixed1, fixed3 and Dynamic execution; observe proposed/accepted work and depth decisions, and preserve serial/grouped byte equivalence and reset behavior. Retained graph capacity is not proof of MTP execution. |
| Performance | Matched Release prefill/decode and complete-request timings, plus lookup latency and movement/communication costs. Compare the relevant dashboard baseline and the best existing supported placement under identical workload intent. |

Static and dynamic expert balancing are independent of the four MTP policies;
the canonical functional matrix must exercise their supported combinations.
Performance measurements must also expose the cost and benefit of both features,
including contention with ngram read-ahead and tier movement. Select defaults
from the fastest accurate measured policy for that topology, not from an isolated
kernel result or from adding devices alone. Keep slower explicit configurations
usable and accurately described when they provide capacity or placement value.

Run cheap policy/admission and focused replay coverage broadly; use the existing
model, generation, numerical and benchmark projections for their respective
proofs. Every configuration advertised as qualified needs the applicable real
execution and accuracy evidence. Missing hardware, unimplemented combinations,
and unmeasured performance remain explicit gaps in the dashboard. Do not turn
them into successful skips or add a separate manually maintained topology list.

The four earlier opposite-vendor ordinary controls and the 24 completed
ConsumerLocal movement/MTP cells provide scoped baselines. They do not qualify
one shared adaptive table across continuation and follower devices. The current
priority is completing that owner integration and the native live-payload path,
then running the six-row campaign with its numerical and Release economy pairs.
A scoped recorder cleanup may remove duplicate setup; a graph-builder rewrite
is outside this slice.

## Existing target and installed boundary

The existing [tiered storage plan](QFN_NGRAM_TIERED_STORAGE_PLAN.md), especially “Next distributed runtime slice”, fixed-phase protocol, directed transport, and qualification sections. It explicitly says pooled adaptive TP and native variable-size collectives remain uninstalled. Historical component passes are not current full-model/economy qualification.

Installed `gpu-adaptive-vram-ram-ssd` is one bounded GPU hot cache over bounded CPU RAM and original SSD source. The production provision rejects multiple consumers, shards or partial main models in [ImmutableRowTablePlacementPlanner.cpp:199](../../../../src/v2/planning/ImmutableRowTablePlacementPlanner.cpp). Removing that check alone cannot implement distributed ownership.

Existing reusable boundaries:

- [Qwen4ExpFixedHeadShards](../../../../src/v2/models/qwen4exp/Qwen4ExpFixedHeadShards.h): checked original 16-head ranges for homogeneous TP2/4/8, including unequal vocabulary lengths, gaps and source tail.
- [Qwen4ExpReplicatedPLECohort](../../../../src/v2/models/qwen4exp/Qwen4ExpReplicatedPLECohort.h) and query/history declarations authenticate actual TP consumers. No expert-only devices join merely because they own VRAM.
- [Qwen4ExpFixedHeadLookupStage](../../../../src/v2/execution/compute_stages/stages/qwen4exp/Qwen4ExpFixedHeadLookupStage.cpp) authenticates native range and cohort, frozen query/control and per-device kernel geometry. It currently requires resident range storage.
- [PooledImmutableLocalLookupGraph](../../../../src/v2/execution/qwen4exp/PooledImmutableLocalLookupGraph.h), [ImmutableLookupExecutionBoundary](../../../../src/v2/execution/qwen4exp/ImmutableLookupExecutionBoundary.h), [native owners](../../../../src/v2/kernels/qwen4exp/PooledImmutableNativeStorage.h) and [source storage](../../../../src/v2/kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h) now separate the coordinator directory/policy, owner-local payload/slots and original source service. The combined adaptive implementation has been retired. Ordinary activation roots are declared during graph construction and sealed after the executor's cold materialization; external cache/source roots retain their original PMA owners. This local graph does not implement directed multi-device transport.
- [HostImmutableRowProvider](../../../../src/v2/loaders/HostImmutableRowProvider.h) and CPU adaptive cache supply bounded RAM/SSD service; native bytes retain source identity. This infrastructure exception must not become a host mirror of GPU cache decisions.
- [ILocalTPContext](../../../../src/v2/collective/ILocalTPContext.h) exposes explicit-stream grouped P2P and raw collectives, but that API does not prove a device-counted, captured variable-length directed protocol. [DeviceCountedAllGather](../../../../src/v2/collective/DeviceCountedAllGather.h) is a distinct mapped-channel transport for proven absent-P2P cohorts, not a reason to replace the native transport or send capacity.

## Selected implementation target

Implement the original flexible pooled owner/consumer design. Homogeneous TP is the first runtime qualification slice, not a product restriction. Fixed-head partition is one explicit distribution policy inside the common declaration; no separate head-local tiering product is proposed. Preserve existing fixed single-tier modes. The topology-neutral declaration, protocol, BOM and transport seams cover independent CPU/CUDA/ROCm owners and consumers from the first implementation.

## Pooled ownership extension already required by original plan

Separate coordinator device row-directory/frequency/recency policy from owner-local payload/slot/pin/generation roots. One authoritative coordinator issues a monotonically advancing lookup-frame identity even for empty/all-hit frames; source-miss sequence is not the frame. Owners issue availability leases; use a fixed bounded phase graph for initial pin, free allocation plan, grants, original-order policy replay, distinct victim plan, source fills and ACK/publication. Global promotion tokens and local slot generations remain distinct. Cancellation drains accepted immutable plans and closes unobserved grants; poisoned/stale frames fail fatally.

Mint one cold-admitted transport operation in the original LocalTPContext binding ordered participants and common bank/chunk geometry. Commands travel only to their addressed owner. Replies may be ragged AllGather only if every actual PLE consumer needs those exact rows; otherwise route only to real consumers. Every call sends device-live rows/counts, never admitted capacity. Empty owners exchange required protocol metadata only. Native NCCL/RCCL capture/connectors and device-counted extents are a feature gap requiring implementation/proof, not a permissive API capability check.

## Independent consumers and storage owners

### Ownership and caching clarification — 2026-11-02

Keep these concepts distinct in the implementation, tests and user-facing policy:

| Concept | Contract |
|---|---|
| Backing table | A complete immutable row source supplied by RAM or SSD. |
| Resident shard | An explicitly placed portion retained for the admitted configuration. |
| Row cache | A bounded allocation containing replaceable copies of useful rows. |
| Consumer | A participant that actually needs lookup output under its declared PLE layout. |

The current pooled adaptive coordinator can allocate cached rows across owners;
it does not impose a permanent source-row home partition. Here an owner usually
owns physical cache slots and their lifetime. Do not describe that as exclusive
logical ownership of a table range. Fixed-head resident partitioning is a
separate existing policy. Immutable replicas may have independent physical
owners without inventing a second authority for any allocation.

Current source misses enter the coordinator's private response storage directly.
Remote cache fills and remote-hit returns are separate transactions. Shared
readers currently receive full replicated PLE lookup embeddings; downstream TP
does not imply that this lookup has already been sharded. Expert-only followers
are storage participants only when declared, and are not automatically PLE
consumers.

The user has authorized implementing and comparing the following policy
variants. They are not installed defaults or qualified performance results.
Prefer local hits, compare shared VRAM capacity
against independent consumer caches, and consider bounded hot-row replication
only when measured saved delivery cost repays the space and fill traffic.
The six-row campaign should measure consumer-local, remote-GPU, RAM and actual
SSD service separately. On this host without direct P2P, remote VRAM is not
assumed faster than RAM. Placement priority remains VRAM, RAM, SSD; routing
should consider actual delivery costs and consumers rather than require an
extra visit to an arbitrary owner GPU. Optional promotion should not extend
the demand critical path when its lifetime can be retained independently.

For the mixed CUDA2 continuation/ROCm4 follower row, the CUDA domain consumes
PLE output. The ROCm devices may contribute resident shards or cache capacity
while doing genuine expert work; their expert role alone does not justify
receiving every lookup result. Ngram placement and expert movement are separate
policies, sharing the same admitted physical resources and transport authority.

Cache contents may change within stable admitted pools. Evicting a row does not
itself release the pool's VRAM allocation, and ordinary promotion/eviction must
not cause readmission or graph reconstruction. Ngram caches remain lowest-priority
VRAM admission consumers. A complete RAM backing table does not automatically
need an additional RAM row cache; SSD-backed configurations require bounded RAM
cache/read-ahead accounting. Immutable cache eviction requires no row writeback
to SSD. Implementation proceeds through the existing typed declarations and
original caller/transport ownership, with required accuracy and matched Release
performance evidence before any production default is promoted. Compare resident
shards, independent consumer caches, a shared dynamic pool, and a shared pool
with bounded consumer-local hot replicas. Preserve the same expert residency
when attributing a gain to cache policy; separately measure joint capacity tuning.

### Placement priority clarified 2026-11-02

The primary user configuration shares physical GPUs between ngram storage and
ExpertOverlay compute. Shared adaptive ownership must be qualified on
continuation domains, follower expert domains, and both simultaneously; a GPU
dedicated to ngrams is a supported control, not the representative completion
target. Fixed-head shards on TP compute GPUs and independent per-consumer
caches are existing partial coverage, not substitutes for this shared pool.

The current first shared-source candidate still requires TP2 compute plus one
distinct storage GPU (optionally the coordinator also owns cache). Native
qualification of that path establishes transport and source/ACK lifetimes; it
does not certify colocation. The next canonical cells must exercise:

| Ngram owners | Required compute activity |
|---|---|
| Continuation-domain GPUs only | Every declared continuation participant executes its model shard and services its own ngram share. |
| Follower-domain GPUs only | Every ngram owner also executes positively routed expert work. |
| Continuation and follower domains together | Both roles serve ngram rows during the same inference session and retain their actual compute obligations. |

Cover homogeneous and cross-vendor domains, single and multiple physical owners,
zero/positive VRAM and RAM quotas with the original SSD source, hot-row promotion
and eviction, static and dynamic expert residency, and MTP off/fixed1/fixed3/
dynamic. Reuse the typed topology definitions and their ordinary serial controls;
do not introduce separate runner-owned topology lists. An idle expert domain
cannot count as successful follower colocation.

Each case needs source admission and final-owner retirement, captured inference,
frozen-history final-logit scoring with layer/stage replay retained, exact MTP,
observed expert movement when requested, and a matched Release benchmark.
Assert one original context/PMA per physical GPU, admitted concurrent storage
and compute bytes, exact live communication, all-reader pin retention and reset/
cancellation. Scheduling must prove absence of same-stream compute/storage wait
cycles; merely deleting topology restrictions is insufficient. Ngram storage
remains the lowest-priority VRAM admission consumer.

Performance records must separate lookup latency, transfer volume, prefill and
decode throughput, and quantify interference with actual expert compute and
rebalance. Retain dedicated-storage and cache-disabled controls to attribute
contention. Numerical, admission and economy outcomes remain separate; declare
the unavailable hardware configurations without treating them as passes.

CPU, CUDA and ROCm storage resources must be independent of where PLE computes and independent of expert placement. Include single-device, TP, PP and composed named domains, cross-rank ownership and heterogeneous CUDA/ROCm/CPU. On this host, both CUDA cards and all four ROCm cards may contribute storage while continuation runs on any supported subset; CPU is optional compute but RAM/SSD service remains an explicitly admitted storage owner. A storage-only GPU is not silently inserted into a compute collective. A PP stage without PLE is not made a consumer merely because it has a device.

Cross-vendor continuation compute is part of the target, as well as cross-vendor
storage. Qualification must include CUDA→ROCm and ROCm→CUDA pipelines,
`PP(TP(2×CUDA), TP(4×ROCm))` and its reverse on this host, and mixed-vendor TP
with independently admitted head/row partitions. Cover `TP(CUDA0, ROCm0)` and
`TP(CUDA0, CUDA1, ROCm0, ROCm1)` as initial mixed TP controls; storage owners may
include additional GPUs outside each compute cohort. Reuse the existing typed
HOST/HETEROGENEOUS collective and graph boundaries where their production
contracts apply. Missing grouped-MTP, source, head-layout or native transport
support is explicit implementation work with admission and execution regressions,
not grounds to change the requested continuation to another topology. Prove
both source admission and actual captured/declared-boundary execution before
qualifying a cross-vendor continuation.

### One topology-neutral declaration

Declare a source identity plus immutable row/range ownership and replication policy; separately declare consumers with exact row/query producers, result format and ordering. Declare storage owners by world rank, physical allocator, tier and capacity, with optional fixed head/range constraints. Declare policy coordinator and CPU source-service authority explicitly. GPU continuation control stays device-owned; a CPU continuation may use host-owned control. Neither form creates a second shadow directory for the same state. Derive directed demand/response edges from those manifests; never infer owner identity from consumer TP coordinate or current continuation primary. Range maps support fixed ownership, bounded adaptive slot ownership and intentional replicas; replicas require explicit placement/publication rules and bytes. Metadata can name original SSD ranges without retaining a full RAM/VRAM table.

Use the existing ResolvedImmutableRowTablePlacement and ImmutableRowTableLayoutGeometry as source-owned roots for this vocabulary rather than a parallel planning registry. Extend the existing provision/wire serialization with owner+consumer+edge identities and versioned policy. ModelContext retains immutable source/service/cache owners; graph instances borrow admitted frames and exact owner generations. Rank/global orchestration owns cross-device protocol membership; ordinary stages remain single-participant. The completed declaration should be valid for arbitrary owner/consumer relations even when the first materializer supports a smaller explicit relation. Resolved placement needs an actual pooled arm; audit callers of worldRank(), which currently projects its first allocator, rather than treating that accessor as a multi-rank domain authority.

### Transport and execution contracts

The starting transport policy is NCCL for CUDA-to-CUDA edges, RCCL for
ROCm-to-ROCm edges, and the existing optimized cross-vendor transport for
CUDA-to-ROCm edges in either direction. The native libraries own their physical
P2P or host-staged choice. This development host has no direct GPU P2P; that is
an expected topology, not a failed device configuration. Keep strict native
P2P tests for eligible hardware and prove absent-peer admission on this host.
Do not select a bespoke same-vendor peer channel merely because the inventory
reports complete coverage. These are measured starting policies: qualify any
subsequent topology/shape specialization against matched transfer and full
lookup performance, with identical ownership, capture and numerical contracts.

Ngram payload lengths remain device-owned. Ordinary captured NCCL/RCCL sends
take a host count, and the installed device-row extension describes equal-prefix
collectives rather than ragged directed ownership. Neither sending capacity nor
reading counts back to the host closes this gap. Extend the existing native
transport with an exact directed payload contract before enabling that route.
A fixed-size native transport floor is useful performance evidence but cannot
certify the variable-length lookup protocol. Preserve the actual source count,
required header traffic, empty-participant progress and payload byte receipts.

Partition edges into native homogeneous local domains, native homogeneous cross-rank domains, and explicit heterogeneous boundaries. CUDA NCCL and ROCm RCCL cannot form one invented mixed-vendor communicator. Cross-vendor storage edges use the public TransferEngine/captured transfer-channel authority with explicit source/destination endpoints and CPU mapped/pinned staging where the actual transport requires it. Cross-rank edges use existing MPI/global transfer infrastructure only after proving its native capture/live-count protocol; existing MPI availability is not proof of GPU-resident control. Unsupported edges fail admission with the exact missing transport capability.

For a homogeneous complete GPU generation domain, lookup phases must be retained inside its complete native parent. A genuine heterogeneous storage/collective edge is a declared graph boundary with authenticated immutable protocol identity; use the existing heterogeneous orchestration boundary and exact producer event/consumer wait. It must not create eager homogeneous execution, host-selected GPU cache policy, per-row submission or recapture. HIP's fixed complete transaction graph selection follows the existing authenticated scheduler-ticket contract; lookup completion cannot introduce mutable host loop decisions.

Every lookup frame has one coordinator-issued ordinal, immutable source/admission generation, per-owner availability grants, owner-local slot generations and original-position permutation. Duplicate misses coalesce in observation order; replies carry exactly required original native rows plus protocol metadata. Multiple consumers may share one reply only when authenticated demand and ordering match. Routing commands are directed; an allgather is justified only when all communicator consumers need every row. Send only live data and required protocol metadata. Retain complete replicated tables only under an explicitly requested, admitted replication policy. GPU directory state has no CPU shadow and device counts are never read back for dispatch. Header/control bounds are frozen; active command/row extents remain device-owned. CPU consumers use host-owned request state and event-aware typed transfer publication at the declared boundary, not a shadow of GPU controller state.

Piggyback compact control sidebands on existing ILocalTPContext/allreduceWithSidebandsOnStream or global collective only when membership, ordering, datatype, producer readiness and consumer demand are identical. A MoE expert collective or activation TP sum generally has different owners and dependencies; do not reuse it merely because a communicator exists. Extra lookup exchanges must be explicit and measured. Cold setup must establish actual connections, fixed phase graph and operation identity before capture; inference must neither allocate connectors nor choose chunk geometry independently at endpoints.

### Full PMA BOM and lifecycle

One common declaration prices every actual owner's native payload, directory/policy state, slots/pins/generations, frames, source read scratch, CPU RAM cache, pinned/mapped channels, per-edge device banks, native connector/graph owners and publication objects. Place host resources on the exact world-rank/CPU0 physical allocator, with NUMA policy recorded separately as placement metadata; distinct NUMA workers do not create additional host capacity ledgers. GPU resources name their exact world-rank/vendor/ordinal. Native reservation units remain certified; multiplicity follows real retained graph owners and concurrency. Tier capacity is lowest-priority optional storage after mandatory execution and protocol roots. PMA performs all admission, materialization, shared-root lifetime and release; caches may track logical occupancy, not a second physical budget.

The [native preparation plan](QFN_NATIVE_ADMISSION_PREPARATION.md) closes the
early CPU-storage dependency through a one-way Preparing→Admitted transition
in that same PMA state. Its exact SDK declarations must precede optional fill;
the plan is approved implementation work, not current native qualification.

Prove simultaneous graphs' concurrency explicitly. One serial family may borrow max-sized frame banks; concurrent consumers require exclusive frames and safe shared cache pin/generation protocol. Accepted allocation plans survive cancellation until all grants, fills and ACKs retire. Host source worker and network/device channel closures drain accepted operations before source/cache roots disappear. Reuse authenticates source files/ranges, owner/coordinator mappings, policy, capacities, graph generations and native transport identity. Faults poison the exact frame and fail fatally; no silent alternate tier or topology outside the declared policy.

### Dynamic expert balancing is part of continuation acceptance

QFN continuation domains retain the existing Qwen35MoE/Qwen 3.6 MoE dynamic
expert-balancing implementation. Reuse its controller, movement journal, owner
generations, quotas and publication ordering; ngram placement must not replace
them with fixed expert ownership or an independent accounting/transport path.
Expert movement and hot-ngram promotion remain distinct typed operations on the
same admitted physical resources, with only their actual live payloads sent.

For every supported homogeneous and cross-vendor continuation policy, include
an execution case that observes real expert movement while serving ngram hits,
misses and promotions. Assert the original movement records/generations and
quotas, source and destination byte extents, correct outputs and bounded memory.
Cover reset, skewed ownership, empty participants and cancellation with both
movement and lookup work outstanding. Exercise Off, fixed1, fixed3 and Dynamic
MTP through the normal generation policy. Source admission, no-movement runs
and resource-owner replays alone cannot certify this requirement; unsupported
execution combinations remain explicitly unqualified.

### Implementable sequence toward the complete feature

1. **Complete the existing topology-neutral contract and planner proof.** Reuse ImmutableRowTableLayoutGeometry: it already checks independent CPU/CUDA/ROCm/rank owners and consumers, pooled/fixed ranges, tiers, replicas, zero capacity and overflow, with 1/2/4/8-owner tests. The separate coordinator-location, owner-slot, lease/grant/ACK and frame records now exist in the pooled native protocol. Their physical owner, coordinator and source geometry supplies the local production BOM, including slot-dependent frame growth as well as cache payload. The remaining production seam is distributed materialization and exact directed edges. Extend the existing AdaptiveImmutableLookupGeometry composer and execution-provision serialization to price and authenticate every actual root and edge through PMA; do not add another manifest or capacity resolver. Device-free tests cover the eventual full matrix, exact capacity, replicas/zero tiers, foreign locality, impossible connectivity and concurrency. Runtime capability admission remains explicit, with no positive claim from declarations alone.
2. **First pooled production slice.** One rank-local homogeneous TP2 domain, actual GPU owners including zero-capacity members, explicit device coordinator and one original CPU RAM/SSD provider. Implement the original plan's separated coordinator/owner authority and bounded fixed-phase protocol; this is real pooled capacity, not head-local imitation. Keep interfaces owner/consumer independent from day one. TP4/8 follows the same declaration and protocol, with unequal capacities and storage-only/consumer-only members required before claiming general domain support.
3. **Local heterogeneous storage edges.** Same coordinator/owner protocol, explicit CUDA↔ROCm and GPU↔CPU endpoints through TransferEngine at declared heterogeneous boundaries. Qualify this host's 2CUDA+4ROCm(+CPU), including continuation on CUDA TP2 with ROCm storage, ROCm TP4 with CUDA storage, and CPU consumer with GPU caches where admitted. Consumer mappings follow the actual admitted continuation graph. Mixed-vendor TP continuation uses an explicit heterogeneous collective boundary and requires its own qualification; separate CUDA and ROCm domains never become an invented NCCL/RCCL communicator. Initial explicit mappings may be fixed by policy; automatic cross-owner migration is independently declared and qualified.
4. **PP/composed consumer integration.** Connect only real PLE-consuming stages and named compute domains, retaining one source/service ownership authority. Add per-graph query/history bindings, precise fanout/routing, and serial/concurrent frame plans. Reuse domain exchanges only under the sideband proof above. Existing missing homogeneous PP(TP,TP) native generation capability must remain a separate explicit execution prerequisite, never bypassed to qualify storage.
5. **Cross-rank pooling.** Extend exact owner/consumer edges with world-rank membership and existing global transport/event publication; authenticate distributed admission and compact traffic. Qualify source placement, remote service failure, empty ranks and process teardown. Do not duplicate source/cache roots per rank for convenience.
6. **Automatic placement/economy and promotion.** Once functional declarations and native paths are complete, select capacity/owner mappings with measured demand/service cost through the existing planner policy. Promotion/eviction remains bounded and GPU-owned for GPU state. Publish automatic eligibility only after the full functional/economy gates; explicit supported mappings remain first-class rather than fallback destinations.

### Qualification product, not a hand-picked topology list

Expand the existing typed execution/storage-owner inventory through the supported product. Axes: CPU/CUDA/ROCm consumers and owner tiers; single/TP2/4/8/PP/composed/cross-rank; same- and cross-vendor edges; storage-only and consumer-only devices; full/partial/head-range/pooled ownership; replicas where explicit; C0/tiny/unequal/full capacities; cold SSD/RAM/VRAM, skew/churn/working-set overflow; one/multiple logical requests and concurrent graphs; activation/KV precision and both CPU ISAs. MTP policies are exactly Off, fixed1, fixed3 and Dynamic, preserving normal production stochastic verifier and dynamic automatic initial-depth intent.

Admission matrix proves complete typed BOM/actual lease multiplicity and exact source identity without claiming execution. Native functional/parity matrix exercises actual retained homogeneous parents and explicit heterogeneous boundaries, public request reset, prefix restore, large→small/empty/repeated frames, accepted/unaccepted cancellation, stale identities, missing ACKs and retirement; assert each wire extent and per-tier occupancy, cache/program decisions and original row bytes. Controller execution and dynamic depth must be observed through actual public generation, separate from source setup. Existing all-format native lookup decode coverage is reused, with source IQ4_NL production rows and byte-exact serial reference ordering.

Performance gate uses Release complete requests and independent profile runs: source-only, RAM-only, full VRAM, bounded pooled tiers and declared cross-device mappings under matched tokens/geometry/policies. Report prefill/decode/end-to-end latency, all communication and promotion bytes, remote/source I/O time, hit/miss/eviction counts, SM/core contention, overlap, NUMA/network costs, physical memory peak and retirement. Measure best existing non-tiered supported placement as the control; selecting more hardware is not itself an economy pass. Canonical Unit+ProductionTestPreflight, backend integration, numerical source/reference gates and full HTTP/image gates retain their existing thresholds. Every new defect receives focused original-code negative/replay registration.

Completing and qualifying these slices fulfills the original flexible distributed target. Homogeneous TP is the first implementation proof, not a topology assumption in the contract and not the final advertised capability.
