# QFN sharded-expert dynamic residency plan

## Status, prerequisite and first target

Dynamic hot/cold promotion and demotion of sharded experts remain unresolved on CPU and GPU. This plan is an implementation proposal, not a support or qualification claim. It replaces a CPU-only design: all layouts must use the same placement, source, transport, accounting and publication authorities.

The common bounded loader and matched Static numerical/RSS slice is checkpointed at `3312168c7`; both physical one- and two-socket panels passed their required groups. See [CPU expert tensor-sharding qualification](QFN_CPU_EXPERT_TENSOR_SHARDING_QUALIFICATION.md) for scope, measured RSS and the reviewed single-row exception. Preserve the model, arithmetic, frozen histories and all aggregate limits. Source admission alone does not certify accuracy, RSS, movement or economy.

The placement metadata slice replaces scalar-only migration vertices with a checked whole-owner or complete-cohort target derived from the existing owner map. The existing cycle builder, physical slot-flow analysis and shadow-slot builder consume that target. Every cohort member contributes one physical fragment slot while the tier records one logical expert. Focused tests cover two through eight members, whole/cohort and cohort/cohort cycles, incomplete members, changed cuts, foreign coordinates and repeated large/empty/small waves. Compact original-coordinate transfer primitives pass their focused CPU/GPU controls. Source-authenticated live joint candidates, exact projection completion and actual CPU-fragment epoch-bank replacement/reader retirement now pass component contracts. A complete initial-bank prepass fixes multi-layer graph startup without retaining old engine roots. The physical fabric now derives actual source/destination intersections, global remote piece identities and matching exact lane demand. Genuine two-socket staging executes exact destination weights; successful staging, failed-piece drain and setup rejection pass 20 repetitions on each actual rank with original source-reader retention and reclamation. Native CUDA/ROCm packet tests now resolve live cohort targets from the acquired placement bank while retaining the original captured executable. These foundations do not enable dynamic joint execution: global publication, controller economy and complete inference/retirement remain required before the Static-only guards can be removed. [Current qualification](QFN_COHORT_MIGRATION_AND_MTP_ADMISSION_QUALIFICATION.md#wholecohort-physical-transfer-and-exact-lane-demand) records the focused transfer checks and retained real-model dynamic-MTP control.

The prepared-bank prerequisite now binds the existing residency registry to its actual original `ModelContext`, concrete `WeightManager`, engine registry and preparation ownership. Expected model IDs alone cannot authenticate a caller-supplied engine registry. This association is installed once at the existing cold producer boundary, with weak retained owners and short-lived source leases; it adds no per-payload provenance table, accounting ledger or retained chain of old roots. All six focused source-binding/lifetime registrations pass, including repeated genuine nonempty-bank controls. Fixed-1, fixed-3 and dynamic native model proofs also pass with this binding, exact serial-token witnesses and complete retirement; dynamic execution covers three controller windows. Complete inference-time candidate publication remains a separate obligation.

The first concrete target is one MI50 continuation GPU and two physical CPU sockets, with rank zero owning the GPU continuation and CPU root, and rank one owning the other CPU fragment. Each CPU expert is tensor-sharded across those sockets. Dynamic movement must promote/demote complete logical experts between the GPU tier and that CPU cohort without assembling a whole CPU expert.

The existing histogram proposal now shares sealed candidate construction for
whole and fixed-cohort tier choices, unchanged prerequisites and bounded cycle
subsets. All 74 owning authority tests and five focused registrations pass;
production cadence retains the original economy lock. The cohort case proves
source-bound derivation from recorded demand, not enabled dynamic execution.
The first genuine ROCm/CPU2 prepared compute/rooted-return/reset case also
passes with exact communication and retirement. Dense whole/cohort proposals now
round-trip through the existing-size wire and source-bound reconstruction;
75 authority cases, 30 protocol cases and actual two-rank consensus controls pass.
Both CUDA and ROCm now pass 20 retained MainPrefill compute/rooted-return/reset
invocations with real empty/small cohort work and final retirement. Decode and grouped-verifier return now also pass twenty invocations per backend,
with exact-row retained captures and same-input serial-row byte equivalence.
Complete cohort service/transfer prices and changed-epoch publication remain
required; public Dynamic/adoption guards are unchanged.

Exact movement observations now retain one immutable original geometry per
logical move and the explicit coordinate of every physical piece. Completion
checks independent bound payload quotes and hands owned observations to the
journal without allocation. Twenty focused registrations pass, followed by the
actual public-factory/composite/journal case repeated twenty times on two
physical CPU sockets: twelve transfers retain two six-piece logical rows, exact
bytes and clean source/memory retirement. The common economy exchange now
transports complete whole/cohort observations over exact-live MPI payloads,
with original local geometry retained through distributed abort/drain.
Complete-target calibration and service prices, including rooted combine and
return, remain prerequisites to enabling dynamic sharded placement.

## Scope: fixed admitted layouts

A placement target may be one whole-expert owner or a complete already-admitted shard cohort. Keep each target's membership, physical endpoints, root/reduction policy, cuts, native contexts, captured graph topology and capacity fixed. Migration between two pre-admitted targets may change the expert's physical partition because the targets have different cuts. That is redistribution between existing layouts, not live cohort or graph repartitioning.

Changing an admitted cohort, inventing a new cut, resizing graph storage, adding a domain or changing a reduction tree while serving is outside this slice. CPU singleton placement remains ordinary whole ownership. Generic metadata covers populations through eight; native tests require genuine corresponding sockets/devices and never emulate eight sockets on a two-socket host.

## Multiple tiers and controller movement graph

The required configuration includes two CUDA GPUs as a hot tier, four ROCm GPUs as a warm tier, and two CPU sockets as a cold-capacity tier. Each tier independently declares apportioned whole experts or tensor-sharded experts. The configuration/test dimension includes all eight apportioned/sharded assignments across these three tiers, wherever each selected compute layout is implemented and qualified. An intermediate tier may be apportioned while either endpoint is sharded. Accepting a configuration document is not evidence that its execution layouts work.

The controller constructs one layout-aware movement graph from the original source and destination **tier expert configurations**, including their projection ownership, replicas, cuts, reductions, formats and capacities. Tier order expresses residency policy, not a mandatory transfer itinerary. Support adjacent moves, direct hot-to-cold/cold-to-hot moves that skip the warm residency tier, and capacity-preserving multi-tier cycles. A transport may relay bytes through host staging or another admitted transfer endpoint without installing an expert there. Such a relay is not a residency hop, must not change authoritative placement, and must never require a whole expert to materialize in an apportioned intermediate tier.

Plan each logical migration as the intersections of the actual source and destination layouts. Stage every member/projection required by the destination while retaining old source leases. Treat a multi-tier cycle as one capacity-admitted candidate: no partial tier, shard or replica publication. Commit one exact candidate epoch and retire all replaced roots only after the existing all-rank selector/publication and reader-drain protocol completes. Direct and multi-tier routes retain the same original expert identity throughout; no per-hop competing controller or ownership path is introduced.

Controller economy must combine existing hotness/histogram evidence and hysteresis with the full logical-expert/cohort execution critical path, transfer and reshard cost, topology-dependent transport/relay cost, every member's final/shadow/staging capacity, and actual available overlap. A scalar representative GPU or CPU cannot stand in for a sharded tier. A candidate must preserve per-member capacities across the entire cycle and offer the existing required amortized benefit; tier labels alone do not justify movement. Transport route selection must not silently override the controller's residency decision.

## Grounded support and gaps

| Area | Existing foundation | Unresolved boundary |
|---|---|---|
| Shared residency machinery | Typed whole-owner/cohort proposals, prepared participant banks, physical shadow slots and source-bound distributed reconstruction | Complete cohort service/transfer pricing and changed-epoch publication/retirement still need qualification before Dynamic can admit sharded experts |
| CPU intermediate sharding | Source-authenticated Gate/Up F-row stripes and Down K stripes, original cohort/root, compact native gather; live placement candidates and fragment epoch banks pass component contracts | Production remains Static-only; two-socket physical staging/abort and native GPU live-target packet dispatch pass. Repeated CUDA/ROCm prefill, decode and grouped-verifier rooted returns pass at a fixed epoch; global publication remains open |
| GPU projection layouts | `MoEExpertProjectionOwnership` describes complete experts and gate/up-owned down-output-column placement; preparation retains source projection geometry | This projection layout differs from CPU F sharding. It does not establish arbitrary GPU tensor-sharded expert execution or dynamic migration |
| General GPU expert tensor sharding | Domain grammar and typed projection metadata are reusable | Standard factory explicitly rejects tensor-sharded MoE; whole-owner map rejects it by default; the joint cohort factory authenticates CPU members only. Each advertised GPU layout still needs real execution/capture/admission support |
| CPU/GPU and CPU/CPU movement | Existing lane/manifests now carry independent original-coordinate rectangles and compact live encoded planes; focused all-format host/native proofs pass, floating edges retain representation | Physical fabric wires each intersection and requires all actual completions; genuine CPU/CPU staging executes exact destinations. Actual CUDA/ROCm FP32 whole/cohort staging and source retirement now pass. Global cohort publication and performance remain unqualified |
| Same-vendor GPU movement | Physical fabric selects actual peer access or an explicitly admitted no-peer host relay | Whole-region peer/blob contracts cannot silently become shard redistribution |
| Cross-vendor GPU movement | Existing CrossBackend blob relay and live physical-format provenance | Cohort/range mapping, compatible destination representation and all-member publication need proof; a relay is not GPU shard-compute support |

`GateUpOwnedDownColumns` keeps Gate/Up owner-local and partitions Down output rows while preserving K and its reduction. CPU intermediate sharding instead partitions Gate/Up F and Down K, requiring a rooted sum. The shared design must retain this distinction. Shape metadata is not proof that either graph or movement implementation supports a layout.

### Logical demand and physical work are separate evidence

Count each logical routed expert use exactly once at the original authenticated routing transaction, independent of shard count, replica count or transport relays. Per-shard copies of the same selected route are execution observations, not additional demand. Do not obtain hotness by summing duplicated participant routing histograms. Bind logical demand to original layer/expert, request/transaction and placement epoch so reset and publication cannot count a route twice or attribute it to a foreign epoch.

Retain physical member work, communication volume, overlap and cohort critical-path measurements separately for economy. These may vary with TP degree or replication even when logical demand is identical. Relay-only transfers contribute transport cost and bytes, but leave logical hotness and authoritative residency unchanged.

GPU routing/controller state and candidate publication remain device-owned and event-ordered through the existing native capture/publication contracts. Do not introduce a mutable host histogram mirror, count readback, blocking synchronization, eager execution or recapture to implement movement. Any required heterogeneous boundary must remain explicit in the admitted graph and retain its existing authenticated ticket/epoch authority.

## One placement and source contract

Extend the existing `MoEOverlayResidencyAuthority` proposal/snapshot with a typed whole-owner-or-cohort target. Preserve original model, layer, expert, projection, source N/K and native provenance. Every member's typed projection layout defines owned original coordinates and execution/reduction obligations. Do not select a cohort's first member as a scalar owner, rebuild a competing source map, or create a CPU-only movement registry.

A migration plan intersects source and destination projection ownership in original coordinates. Each transfer rectangle carries both source and destination origins, physical strides, logical/native block extents, epoch, participant and source-layout identity. Validate all intersections before writes: exact required destination coverage, no unintended overlap, no holes, native alignment and compatible representation. Explicit replicas need typed copy semantics; they must not accidentally contribute twice to a sum.

Use actual prepared storage as the source. CPU engines retain NativeVNNI or floating roots; GPU engines retain separated arrays and live format descriptors plus original provenance. Common loading releases raw source parents. Passive source geometry cannot supply raw GGUF bytes, and movement should not introduce a new GGUF reread protocol.

## Existing transport and accounting boundaries

Extend existing lane/manifests/kernel contracts rather than adding a transport:

- Whole-to-sharded selects only destination-owned original rectangles. Sharded-to-whole writes disjoint rectangles into one inactive destination root, with all-member readiness before publication.
- Sharded-to-sharded transfers the original-coordinate intersections between the two admitted cuts, including different source/destination shard counts. No whole CPU assembly, packed-span concatenation or unnecessary full-expert fanout.
- CPU/GPU edges use the existing WeightTransferLane and TransferEngine staging/streams. Same-vendor GPU edges use admitted peer regions where available, otherwise the existing explicit relay. Cross-vendor edges use the existing CrossBackend blob transport. Nonlocal endpoints use the physical fabric's existing remote projection protocol, not activation gather or a new communicator.
- Native movement preserves codebook, scale, minimum and grid-factor semantics. GPU physical formats may have changed after a previous promotion; authenticate the live descriptor with original provenance. Reformat without requantization. Unsupported format/layout combinations fail before staging. Floating K stripes require explicit strided gather/scatter, not contiguous multirow memcpy.

`PhysicalMemoryAuthority` remains the sole ledger. Extend the original capacity/fabric BOM for simultaneous old/new final roots, per-member placement, inactive destinations and actual concurrent lane demand. Reuse `ExpertMigrationStaging`, `ExpertShadowSlots` and existing TransferEngine persistent staging/execution slices. Source engines and transfer readiness remain retained until actual completion; abort cannot free pending DMA or synchronize a destructor. No blanket reserve or second byte authority is permitted.

## Implementation sequence

1. **Make layout capability explicit.** Describe supported whole and sharded targets using original projection coordinates and actual compute/reduction contracts. Retain current guards where GPU execution or movement is absent. Establish the MI50/CPU-cohort target first without inventing general GPU shard support.
2. **Extend shared proposal and epoch banks.** Add cohort placement to existing transactions, wire validation, participant banks and source leases. Stage every required member/projection under the same candidate epoch and PMA admission. Partial readiness cannot publish.
3. **Implement rectangular movement.** Extend WeightTransferLane, peer/blob/remote manifests and native kernels with the shared source/destination range contract. Whole projections use identity rectangles. Add bounded intersection iteration for different admitted cuts; preserve native bytes and fixed arithmetic.
4. **Publish versioned execution placement.** Extend existing versioned placement data to carry cohort fanout and root/reduction roles. Captured pointers/topology stay stable. Publish prepared banks and packet epoch through one selector; never mutate captured host membership bits and assume graphs changed.
5. **Reuse distributed commit and retirement.** Keep existing reserve, stage, inactive-bank prepare, selector and authority consensus. Pin old epochs across dispatch, compute, reduction, sparse return and ACK. Close old admission, drain leases and retire all old roots through existing consensus. Keep native domain/session lifetime unchanged.
6. **Integrate economy and qualify.** Existing histogram and rebalance policy must price complete target critical paths and all transfer edges/member storage. Enable a layout/mode only after its functional, numerical, communication and native gates pass. Observe also needs a complete supported evidence path.

### Qualified common cost-evidence exchange

Use the existing calibration controller and MPI evidence lane. The actual
`beginStage` result retains a measurement-only context containing the
transaction identity and the same immutable geometry owners emitted by its
prepared factory. The controller keeps that context through wave drain and
evidence exchange. It holds no engines, source leases or publication authority.

Whole and cohort operations retain their independent payload quotes in one
piece-observation representation. The existing scalar-price consumer may
explicitly project a checked singleton whole-to-whole geometry onto Gate/Up/Down;
it must keep rejecting cohorts until complete-target pricing is implemented.
Both migration-profile and attempt exchange require the local checked context.
Retire their old fixed payload together rather than keeping an unbound wire
route for old test fixtures.

The same MPI request, communicator and original deadline cover a fixed
identity/count-header exchange followed by `MPI_Iallgatherv` of only live
quote/sample records. Empty observers send required metadata and no payload.
Decode resolves ordinals against local checked geometry and verifies endpoint
rank, original coordinates, formats and exact extents; sender layouts cannot
create authority. Tests cover malformed and incomplete records, empty observers,
large-to-small reuse, actual transmitted byte counts and abort/drain. Readiness
and service exchanges keep their existing contracts. The owning Unit suites,
actual two-rank exchanges and CPU/GPU physical controls pass all 23 focused
registrations in `focused219`, with no skips and clean driver/retirement checks.
This is not an enabled cohort-calibration or Dynamic execution claim.

The same calibration planner now authors genuine nonpublishable whole/cohort
shadow jobs from one canonical target list. Its typed ledger preserves complete
piece geometry/quotes with allocation-free recording and robust wave-maximum
pricing. The actual two-socket Fabric can prepare these authenticated shadows,
complete/drain them and reuse slots without changing live ownership. All 32
focused registrations pass after the coherent build, including native MPI,
CPU/GPU transfer controls and twenty-repeat contracts. See the
[qualification scope](QFN_COHORT_MIGRATION_AND_MTP_ADMISSION_QUALIFICATION.md#complete-target-shadow-calibration-and-robust-measurements).

The native controller witness now joins the existing distributed transport,
actual warmup/measurement waves and typed ledger: twenty repetitions complete
eighty shadow waves with independent rank progress, exact live bytes and final
reader/PMA retirement. All three selected native registrations pass; production
code is unchanged from the preceding green slice. Whole-only certification and
unresolved cohort layer expansion still reject cohort prices explicitly. Service
measurement must cover the actual fragment computation through rooted combine
and publication; local fragment durations omit that tail. Preserve one logical
demand count and charge shared invocation work once. Complete service pricing,
certification and publication remain required before enabling movement.

## Required tests

### Functional and ownership

Extend existing authority, proposal/protocol, registry, Store and physical-fabric suites. Cover whole-to-sharded, sharded-to-whole and sharded-to-sharded transactions, differing admitted cuts, exact original identities, replicas, empty intersections, stale/foreign epochs, duplicate/missing members, failure at every phase, abort and delayed reader retirement. Metadata sweeps cover one through eight participants without native-hardware claims. Preserve whole-expert controls.

Use real prepared handles/source leases and exact allocator geometry. Prove exact quota, one-byte-short failure, simultaneous old/new roots, bounded staging, partial preparation cleanup and retirement after the last actual reader. No fake readiness or raw-parent substitutes.

### Tier configurations and movement graph

Exercise all eight hot/warm/cold apportioned-versus-sharded assignments as explicit configuration cases, including an apportioned intermediate tier between sharded endpoints. Positive execution cases require actual qualified compute layouts; missing GPU shard implementations remain explicit gaps. Validate adjacent and direct tier-skipping moves, hot/warm/cold exchanges and three-tier capacity-preserving cycles, with failures at each member and publication phase. Prove that a relay leaves the intermediate tier's residency and final-root ownership unchanged and never creates a full intermediate expert. Include asymmetric capacities, unavailable destination members, hysteresis/no-benefit rejection and repeated hotness reversal without placement thrashing.

Mixed projection and replicated targets are additional typed layout dimensions. Preserve the distinction between owner-local Gate/Up with distributed Down output stripes, CPU F stripes with Down K cuts, and explicit full or partial replicas. Validate required coverage per projection and replica role, and reduce each actual contribution exactly once. Do not coerce one layout into another because its tensor byte count happens to match.

Logical demand regressions must feed equivalent original routes through apportioned, TP1, TP2 and TP4 layouts and assert identical per-expert logical counts and hotness decisions. Check physical work counters independently rather than requiring identical participant costs. Cover partial and empty work, repeated captured replay, request reset, epoch transitions and old readers; every genuine route counts once, while replay of execution for the same authenticated transaction cannot duplicate demand. Relay-only hops must change only transport evidence, leaving both residency and hotness unchanged. Run device-free controller contracts where possible and native event/capture proofs on each qualified backend; register focused regressions explicitly in ProductionTestPreflight.

### Accuracy and native formats

Sweep all supported native codebooks and floating formats across nonzero row/K origins and unequal aligned cuts. Compare whole-to-sharded-to-whole and sharded-to-differently-sharded results against independent original representation and production execution oracles. Include previously promoted GPU physical formats and both Down-output and Down-K layouts. Preserve fixed reductions and native quantization; do not replace unsupported movement with requantization.

Keep three separate proof obligations: exact native-format/roundtrip preservation proves representation transport; cross-layout model accuracy proves the changed partition/reduction configuration against its canonical numerical contract; fixed-layout MTP grouped decode must retain serial-row byte equivalence. One proof does not substitute for another, and exact weight movement alone does not establish identical arithmetic across different reduction layouts.

Run matched QFN histories, full-vocabulary metrics and required full-stage checkpoints before and after repeated promotion/demotion and reset. Compare Static and Dynamic with identical source/arithmetic. Existing source or static results cannot certify movement accuracy.

### Communication and economy

Assert actual transferred and collective bytes for partial, empty, skewed and large-to-small work. Only required intersection data and protocol metadata may cross each edge. Allocation padding, unused expert slots, full replicated matrices and duplicate root outputs are not communication extents. Record exact staging high water, PMA ownership, transfer/capture path counters, retirement and movement critical path. Measure production overlap and useful benefit after correctness; a functional but uneconomical path is unfinished.

### Hardware-realistic native matrix

The first required cell is MI50 continuation plus two physical CPU sockets, one MPI rank per socket, with the GPU/CPU root colocated. Exercise existing ShmemSpin and MPI CPU domains, actual promotion/demotion and in-flight old readers. Then cover same-vendor GPU whole/shard edges with real peer and admitted no-peer topology, and CUDA/ROCm mixed-vendor tiers through the existing relay, only where actual supported compute layouts and hardware are available. The concrete two-CUDA/four-ROCm/two-CPU tier configuration requires the corresponding real devices and physical CPU socket/rank binding. Preserve explicit original inventory and GPU UUID visibility masks, including the actual MI50 UUID masks recorded by the qualification owner; never guess ordinal identity or replace an unavailable topology with extra artificial ranks. Record the exact selected masks with each qualification invocation. Do not manufacture extra ranks or advertise unavailable GPU-shard layouts as qualified. Four/eight CPU socket cells require matching physical hardware.

Every production defect needs a focused `ProductionTestPreflight` entry. Reuse existing residency-consensus, tier-migration, CPU shard lifecycle, peer/blob, native repack and backend capture suites. Extend CUDA/ROCm resource/spill compilation and runtime evidence for changed kernels. These constituent tests currently do not certify dynamic sharded-expert residency.
