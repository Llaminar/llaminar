# ExpertOverlay Tier Migration and Heterogeneous Ticket Design

Created: 2026-08-11  
Last implementation audit: 2026-08-14

## Document status

This is the normative target architecture for arbitrary-tier ExpertOverlay
migration. It is also a status map: the table below distinguishes code that is
installed in the production graph from protocol and device components that are
implemented but not yet composed into the live runner. In the design sections,
“must” describes the required production contract; it does not by itself claim
that the corresponding row is already certified.

| Slice | Current implementation evidence | Status |
|---|---|---|
| Immutable residency epochs, capacity-preserving cycles, shadow requirements, ticket leases, deferred retirement, and fail-closed abort | `MoEOverlayResidencyAuthority.*`; `V2_Unit_MoEOverlayResidencyAuthority` | Implemented and device-free proven |
| Exact old/current host epoch lookup for delayed captured tickets | `MoEOverlayResidencyAuthority::tryAcquireTicketSnapshot(epoch)`, the lock-free retiring-epoch slot, dispatch ticket ABI v3, and focused authority/dispatch regressions | Implemented and device-free proven; the device producer does not yet populate the epoch |
| Device-owned captured epoch admission and two-bank grace period | `DeviceMoEOverlayEpochABI.h`, `MoEOverlayDeviceEpochProtocol.*`, and `V2_Unit_MoEOverlayDeviceEpochProtocol` define one model-role selector, acquisition guard, per-bank readers, candidate readiness, publication, abort, and reuse | CPU implementation/oracle and adversarial 20-epoch/concurrent proof are green; CUDA/HIP kernels, runtime-table composition, and graph acquire/release stages remain |
| Arbitrary-tier composite prepare/commit/abort polling | `MoEOverlayTierMigrationTransport.*`; `V2_Unit_MoEOverlayTierMigrationTransport` | Implemented and device-free proven |
| Non-blocking histogram generations | `DecodeExpertHistogram::freezeAndRotateWindow()` and its concurrent-writer regressions | Implemented and device-free proven |
| GPU/CPU stream manifest and layout conversion | `ExpertTierWeightStream.*`; its unit sweep covers all 21 catalogued NativeVNNI source formats | Implemented and device-free byte-layout proven |
| CUDA and ROCm GPU↔CPU conversion kernels | `CUDAExpertTierWeightKernels.cu`, `ROCmExpertTierWeightKernels.hip`, and the corresponding `V2_Integration_*_ExpertTierWeightKernels` suites | Implemented; real-device, all-format, byte-exact proof |
| Persistent asynchronous GPU↔CPU and GPU↔GPU lanes | `ExpertTierWeightTransferLane.*`, `ExpertTierGpuBlobTransferLane.*`, and `ExpertTierMigrationOperations.*` | Implemented; real-device overlap proof with deterministic test payloads |
| CUDA-hot/CPU-cold and CUDA-hot/ROCm-warm/CPU-cold cycles | `V2_Integration_MoEOverlayCudaHotCpuColdMigration` and `V2_Integration_MoEOverlayThreeTierMigration` | Real-device protocol proof; no model is loaded |
| Captured heterogeneous ticket publication/consumption | `MoEOverlayTicketPublishStage`, `MoEOverlayTicketConsumeStage`, conditional lowering in `Qwen35MoEGraph`, and authority propagation through `InferenceRunnerFactory` | The segmented host boundary is installed and real-model parity proven. It still copies the complete routing and hidden bucket to host, acquires the current host epoch there, and has no captured continuation-local expert branch |
| Production authority, participant banks, physical fabric, and background maintenance ownership | `OrchestrationRunner::initializeMoEExpertOverlayResidencyAuthority()`, `initializeMoEExpertOverlayResidencyMaintenance()`, `MoEOverlayParticipantResidencyRegistry`, and `MoEOverlayResidencyMaintenanceService` | Installed for process-local and distributed tier sets; real-model three-tier migration certification proven |
| Live service and movement-economy certification | `MoEOverlayEconomyCalibrationController`, `MoEOverlayMigrationMeasurementLedger`, `MoEOverlayEconomyProfileComposer`, `MoEOverlayEconomyCertificationController`, and the private MPI evidence lane measure real non-publishable waves plus exact live prepared-expert service, merge owner-authenticated evidence, and seal the authority before proposals | Installed for local and distributed dynamic overlays; device-free lifecycle, real two-rank physical-fabric calibration, and the complete real-weight two-/three-tier backend matrix are proven |
| Live histogram-to-tier proposal, physical transfer, inactive-bank publication, and retirement | `MoEOverlayResidencyMaintenanceService` drives coordinator-frozen histogram generations through the local or distributed migration transport and `MoEOverlayPhysicalResidencyFabric` | Installed for local and distributed dynamic overlays; real-weight three-tier production parity proves repeated publication and retirement, while the wider topology/performance matrix remains |
| Cross-rank prepare/commit/abort/retire consensus and remote projection data plane | `MoEOverlayDistributedResidencyProtocol.*`, `MoEOverlayMPIResidencyConsensus.*`, `MoEOverlayMPIRemoteProjectionTransport.*`, `MoEOverlayGpuRemoteProjectionEndpoint.*`, `MoEOverlayPhysicalResidencyFabric.*`, and the two-rank heterogeneous residency integrations | Implemented and composed in the production runner; CPU, GPU/CPU conversion, same-packed GPU, and cross-vendor GPU blob paths are proven below model level |
| Exact model- and topology-aware capacity admission | `MoEOverlayCapacityResolver.*`, `MoEOverlayCapacityAdmission.*`, `MoEOverlayLocalCapacityPlanner.*`, and `OrchestrationRunner::freezeMoEExpertOverlayPlanForLoadedModel()` build the GGUF manifest, charge fixed/live/shadow/staging/reserve bytes per rank/device, search resident captured-prefill shapes, and install layer quotas before placement | Installed in production and device-free proven across all 21 NativeVNNI formats; real-weight CUDA/ROCm/CPU admission is proven, while the remaining target topology matrix must still be certified |
| Universal multi-device MoE placement authority | Every multi-device MoE topology, including one domain with one tier, is normalized to an ExpertOverlay plan and uses one RCU authority for durable placement, same-tier skew correction, replicas, and request-scoped LLEP leases | **Target architecture; not yet installed for legacy non-overlay LocalTP/NodeLocalTP cells.** The tiered-overlay runner already owns the RCU authority, but its planner does not yet deliberately optimize participant load within an unchanged tier |
| Homogeneous single-tier execution locus | ExpertOverlay remains the sole logical authority, while a backend-native controller owns policy, epoch selection, and generation-loop decisions without a host mirror | The registered real-weight CUDA2 and ROCm2 fresh/prefix-restore by fixed-depth-2, fixed-depth-3, and dynamic-depth matrices are numerically and structurally green in 332.09 and 365.53 seconds respectively. CUDA proves one native conditional parent. ROCm uses the first-class authenticated-ticket policy described below: HIP retains captured transactions, the device publishes only the immutable branch decision, and LocalTP submits every participant branch before the next ticket wait |
| Real-weight two- and three-tier model parity | The Qwen 3.5 production graph-native overlay cells cover CUDA/CPU, ROCm/CPU, CUDA/ROCm, and CUDA/ROCm/CPU dynamic layouts, segmented heterogeneous prefill, exact checkpoints/CSV artifacts, and a static zero-movement control | Numerical/path evidence is green. The new convergence-speed assertion is red because migrated execution is currently about 36--50% slower rather than at least 2% faster |
| Repack/stream throughput, occupancy, register/VGPR, and spill certification | `V2_Perf_CUDA_ExpertTierWeightStreaming`, `V2_Perf_ROCm_ExpertTierWeightStreaming`, Nsight Compute, `rocprofv3`, and static gfx906 code-object inspection | Implemented and certified for all formats and real projection shapes |
| End-to-end migration interference and staging backpressure during real inference | Deterministic device integrations prove asynchronous overlap, including concurrent FusedQKV stream pools. The real-weight two-rank CPU NodeTP campaign now collects exact paired baseline/concurrent inference intervals while a non-publishable reciprocal expert wave runs. | CPU NodeTP overlap is measured and gated; equivalent real-model CUDA/ROCm/heterogeneous contention and observed-speed certification remain |

The historical non-overlay Dynamic machinery and its
`DeviceMoERebalanceController` still own a separate durable table in legacy
multi-device paths. That is an implementation gap, not the target architecture.
Useful participant-load planning, device kernels, transfer slots, and LLEP
mechanics must move underneath the ExpertOverlay authority; their independent
publication path must then be removed. Existing legacy Dynamic/LLEP counters
must not be cited as proof that the ExpertOverlay protocol ran. Completion still
requires universal one-authority composition, the captured device-local
execution bank, observed performance certification, and the later
MTP/prefix/target-model matrix through that authority itself.

## Objective

The completed ExpertOverlay system keeps the most frequently selected routed experts in the tiers
with the greatest compute capacity and demotes colder experts when residency is
rebalanced. The live tiers are the source and destination of every movement;
the model loader is not a migration fallback and the runtime does not retain a
second host copy of accelerator-owned weights.

A rebalancing integration is evidence only when it proves all of the following:

- routing histogram evidence changed the selected residency;
- complete gate/up/down weights moved directly from the old tier to the new;
- an immutable residency epoch was published only after destination prepare;
- inference continued issuing tickets while the candidate epoch was prepared;
- no inference stream synchronized with or joined a migration stream;
- dispatch tickets used exactly one epoch from dispatch through final return;
- old owners were retired only after publication and that old epoch's leases
  drained;
- all required domain, rank, backend, conversion, and byte counters appeared in
  `PerfStats`;
- Static residency ran the same maintenance check and produced zero movement;
- logits and stage checkpoints retained the canonical parity CSV contract.

The protocol applies to decode and bucketed captured prefill, plus grouped MTP
verification when MTP is enabled. Decode and prefill are universal certified
phases. Grouped-verifier service and interference evidence is required only for
an MTP-enabled instance; an MTP-disabled certificate retains exact zero in that
column, and later grouped demand is fatal. Ordinary CUDA-only and ROCm-only
production cells, plus CUDA MTP, remain one complete captured graph. HIP MTP
retains complete captured transaction graphs and crosses only the authenticated
fixed-size scheduler-ticket boundary; this is neither eager replay nor
segmentation. Segmentation is permitted only at declared heterogeneous
ExpertOverlay boundaries.

## Universal multi-device MoE authority

ExpertOverlay is not a synonym for heterogeneous execution or for movement
between differently prioritized tiers. It is the placement, residency, and
publication authority for **every** MoE execution cell with more than one
physical participant. A one-domain, one-tier cell is the simplest valid
ExpertOverlay topology: every expert's tier index remains unchanged, so it has
no promotions or demotions, while participant ownership, resident replicas,
and current-batch work assignment may still change.

The production rule is structural:

```text
one physical MoE participant  -> direct single-device residency
two or more MoE participants  -> exactly one ExpertOverlay authority
```

No graph in the second row may construct a legacy domain controller, publish a
second ownership epoch, or infer placement from graph-construction order. CLI,
YAML, and programmatic configurations that name one multi-participant domain
are normalized during topology planning to one tier with an explicit integer
priority and exact physical budgets. This normalization is canonical topology
resolution, not a runtime fallback and not a hidden compatibility mode.

The authority owns four related but distinct decisions:

1. **Tier membership:** which priority tier contains each logical expert.
   One-tier cells keep this vector constant.
2. **Durable participant residency:** the exact owner and any bounded durable
   replicas inside the selected tier/domain. Histogram-driven skew correction
   changes this state through the ordinary `E -> E+1` RCU transaction.
3. **Request-scoped LLEP residency and assignment:** temporary complete expert
   copies and routed-row destinations for one ordinary-prefill transaction.
   These leases are based on one durable epoch and never become the next durable
   owner map implicitly.
4. **Execution publication:** one immutable epoch identity consumed by main
   decode, ordinary prefill, grouped verification, MTP sidecars, and prefix
   restore. Every physical transfer finishes before the relevant durable or
   request-scoped publication becomes visible.

`Off`, `Observe`, and `Dynamic` are maintenance policies of this authority,
not selectors for different controller implementations. Static/Off still
constructs the authority and runs its immobility check. Observe records the
same evidence and candidate diagnostics without moving bytes. Dynamic may
produce cross-tier cycles, same-tier participant cycles, replica changes, or
any combination admitted by the fixed capacity and economy contracts.

### Authority execution locus

The topology chooses where the one authority executes; it never chooses a
second authority. A homogeneous, single-backend, single-tier GPU cell uses a
backend-native device controller. Tier membership is constant, but that
controller may still perform Dynamic same-tier ownership cycles and LLEP row
assignment. Histogram windows, policy decisions, epoch publication, and all
mutable MTP/decode state remain device-owned. The graph API determines only how
the selected branch is submitted: CUDA composes a native conditional parent;
HIP publishes one immutable 48-byte scheduler ticket and the host submits the
already-captured transaction named by that ticket.

A topology with multiple integer-priority tiers or heterogeneous participant
types uses the host-authoritative ExpertOverlay RCU coordinator. It composes
capacity budgets, GPU/CPU format conversion, arbitrary transfer domains, and
cross-rank consensus while inference consumes immutable device-published epoch
tickets. This is an explicit topology choice, not a fallback from a failed
device controller. There is no concurrent legacy Dynamic/LLEP controller and
no host shadow of the homogeneous controller's live state.

The backend implementations are intentionally asymmetric at the graph-control
boundary. CUDA has a native conditional MTP parent. The installed HIP graph
surface has no conditional nodes, so ROCm uses a first-class ticket-selected
captured-transaction policy. Certification requires graph-family
materialization, the exact 48-byte `immutable_scheduler_snapshot` with
`state_payload=false`, authenticated decisions, per-device transaction and
terminal submissions, controller/compact-reducer ledger equality, no
intermediate host state materialization, and either a standalone device launch
or a rank launch proving every participant was submitted before the next ticket
wait. `production_path.csv` keeps `full_graph_*` false for this policy while
requiring `forward_full_graph_*`, `hosted_ticket_boundary_certified`, and
`generation_loop_certified`; it never relabels the captured transactions as a
native parent.

### Same-tier skew objective

Tier-only optimization is insufficient for an apportioned domain. Equal expert
counts can leave one participant with nearly all routed work when its experts
are disproportionately popular. Candidate construction must therefore solve
for exact physical participants, not derive owners afterward from ordinal or
random expert order.

For each layer, the planner assigns experts to participant vertices. Each
vertex inherits its tier membership, exact resident-slot/byte quota, measured
phase service costs, and compute-capacity weight. The objective first respects
the fixed tier quotas and then minimizes projected service time and normalized
participant load using the frozen decode/prefill/grouped histogram. Incumbent
ownership is the deterministic movement-cost tie-break. A capacity-preserving
pair or cycle within one tier is a `SamePriority` migration: it increments no
promotion or demotion counter but must increment explicit same-tier and
same-domain movement evidence.

Owner swaps address divisible aggregate skew without increasing residency.
When one indivisible expert dominates a participant, an optional bounded
durable replica is the only way to distribute that expert's rows. Replica
slots are part of the immutable epoch, charged by the physical capacity plan,
and published by the same transaction. An independent hot-cache table is not
another authority.

Replicated domains already retain every complete expert and ordinarily need
only row reassignment. Tensor-sharded domains retain one collective shard group
per expert; they participate in the authority and epoch protocol, but a whole
expert owner swap is inapplicable unless the declarative topology first changes
the residency representation. These distinctions are typed compute policies,
not reasons to bypass the authority.

### LLEP interaction

Current-batch LLEP is an authority-owned child transaction rather than another
placement authority. It pins durable epoch `E`, plans only ordinary-prefill
rows, reserves separately budgeted temporary slots, transfers missing complete
experts through concurrent transport lanes, publishes its request-local
assignment, executes, and then releases those leases. The current request
cannot consume a missing expert until that payload is complete, so its LLEP
materialization remains on the prefill dependency chain even though the MPI or
GPU operations within the wave are non-blocking and concurrent. Decode and
grouped MTP verification continue to use the durable epoch. An `E+1`
maintenance wave may prepare and publish while the LLEP transaction executes,
but retirement and slot recycling for `E` wait for both durable inference
readers and LLEP leases.

This gives one unambiguous resource arbiter. Durable shadow arrivals,
request-scoped LLEP arrivals, and retiring epochs cannot claim the same physical
slot, while their independent transfer streams may overlap inference whenever
their event DAG permits it.

## Capacity, apportionment, and hotness destination

### Tier identity and ordering invariant

ExpertOverlay has no built-in `hot`, `warm`, or `cold` tier classes. A tier
name is an opaque user label; it is used only for configuration identity,
diagnostics, distributed-plan hashing, and human-readable output. Production
must never branch on a tier name, derive a backend from it, or use declaration
order as a preference tie-break.

Every tier instead has one unique signed integer `priority`. Smaller values are
more preferred, values need not start at zero or be contiguous, and tier index
is stable identity rather than rank. Equal priorities are rejected because
otherwise some second field would silently become the ordering authority. The
optional `fallback` flag is orthogonal coverage responsibility, not a thermal
class: when present it must be on the greatest numeric priority, and it implies
neither CPU placement nor unbounded storage. Terms such as "hot tier" in old
test names or examples are shorthand for a particular configured priority,
never schema values or recognized labels.

### The intended answer

Yes: after setup has resolved safe physical capacities, live experts are
apportioned by filling the lowest-numeric-priority tier first, then the next
priority, and finally the coverage tier. The word "fill" refers to the tier's
resolved **live expert quota**, not every byte of RAM or VRAM. Migration shadow
slots, old-epoch leases, transfer staging, graph workspaces, KV cache, and a
safety reserve consume physical memory but are never routable live capacity.

For each routed layer `l`, let `Q[t,l]` be tier `t`'s fixed live logical-expert
quota and `E[l]` the model expert count. Setup must establish:

```text
sum over tiers t of Q[t,l] = E[l]
```

Every expert consequently has exactly one tier, including the least-preferred
tier. A
fallback tier is the final coverage tier; it is not an infinite-memory escape
hatch. Its resolved physical budget must be able to hold the remainder. A plan
whose priority-ordered capacities cannot cover all experts fails before any
graph is built.

Example: if a 256-expert layer resolves to 96 slots at priority `-20`, 64 slots
at priority `7`, and 96 coverage slots at priority `90`, cold start assigns 96
experts to priority `-20`, the next 64 to priority `7`, and the remaining 96 to
priority `90`. The domains could be CUDA, ROCm, CPU, or another supported
combination; the labels do not decide. After routing evidence exists, the 96
most demanded experts belong at priority `-20`, the next 64 at priority `7`,
and the remainder at priority `90`. A newly high-demand expert is promoted
through a capacity-preserving cycle that demotes a lower-demand occupant;
capacities do not change while inference is live.

### Physical memory budget and live quota resolution

Memory is owned by physical participants, not by a tier name. A tier spanning
two GPUs therefore needs one budget and one exact byte bill for each GPU. Two
tiers sharing a physical device must be admitted against one combined device
budget. CPU budgets are NUMA-endpoint budgets; treating all host RAM as freely
fungible would defeat the NodeLocalTP placement contract.

For every physical participant `p`, setup must prove an inequality of this
form using the actual model manifest and destination packing:

```text
fixed[p]
  + transfer_and_shadow[p]
  + sum over layers l of resident_copies[p,l] * expert_bytes[p,l]
  <= usable_budget[p]
```

The terms have exact meanings:

- `usable_budget[p]` is the smaller of the inventory-reported allocatable
  memory and any explicit user limit, less a configured safety reserve;
- `fixed[p]` includes continuation/non-expert weights, captured graph arenas,
  activations, KV and prefix-cache reservations, libraries, descriptors, and
  other model-lifetime allocations owned by that participant;
- `transfer_and_shadow[p]` includes the inactive arrival slots needed for RCU
  publication, old/new epoch overlap, bounded conversion buffers, network
  staging, streams, and events;
- `expert_bytes[p,l]` is the exact sum of gate/up/down prepared bytes for one
  expert in layer `l` on that backend. It comes from the GGUF weight manifest
  and the real GPU or CPU packed layout, including scales, minima, extended
  minima, compensation, alignment, and codebook-specific sections;
- `resident_copies[p,l]` follows the domain's compute policy. Apportioned
  domains divide logical experts between participants, replicated/LLEP decode
  domains charge every replica, and tensor-sharded domains charge their exact
  shards rather than pretending a logical expert has one copy.

The capacity resolver chooses the integer vectors `Q[t,l]` once, before weight
preparation and graph materialization. In automatic mode it lexicographically
maximizes live capacity in strict integer-priority order subject to every
physical inequality. Thus it uses all safely available capacity at each
successive priority and assigns the remainder to the coverage tier. An explicit expert cap is an additional
upper bound, not a substitute for byte admission. An explicitly fixed quota is
either admitted exactly or rejected; setup must not silently shrink it.

The resolved quota may vary by layer when projection geometry or codebook
changes. Runtime migration preserves each `(tier, layer)` quota and each
participant's derived share exactly. Migration headroom is planned separately
so increasing shadow concurrency cannot silently evict live experts. A tier
with zero resolved live slots is inactive for residency; tests expecting
movement through that tier must require a positive quota instead of relying on
its name being present.

### Cold-start selection and ordinal/random ownership

Capacity and membership are distinct from participant ownership:

1. Capacity resolution decides how many logical experts each tier can hold.
2. Cold-start ordering decides which expert IDs initially occupy those quotas.
3. `RoutedExpertOwnerOrder` distributes a tier's selected experts over that
   tier's physical participants.

Without a certified prior histogram/profile, cold start is deterministic. The
default orders expert IDs ordinally across priority-ordered tier quotas. A
seeded-random cold-start policy may randomize tier membership when requested,
but its seed and permutation are part of placement identity. Independently,
ordinal and seeded-random owner order control which participant in a tier owns
each selected expert. Random owner order must never randomize tier priority or
make a cold expert outrank a hot one once histogram evidence exists.

A future persisted routing profile may seed cold start, provided its model,
quantization, topology, phase mix, and profile generation are authenticated.
It is an optimization only: absence or mismatch returns to deterministic cold
start, not to an unproved placement fallback.

### Runtime placement objective

Raw frequency is a useful baseline, but the production objective is expected
inference time saved. The histogram authority must retain separate evidence for
ordinary decode, grouped MTP verification, and real (unpadded) prefill rows.
For an expert `e`, layer `l`, phase `f`, and candidate tier `t`, the planner
uses the measured service cost of that tier for the model's projection shapes
and codebook:

```text
expected_cost[e,l,t] = sum over phases f of
    demand[e,l,f] * service_time[t,l,f]
```

Assignment minimizes total expected cost subject to the fixed quotas. When
tier service costs are monotonically ordered and experts in a layer have equal
size, this reduces to the intuitive rule: sort by phase-weighted demand and
fill `Q[t,l]` in ascending integer-priority order. The explicit priority
remains the capacity-fill authority and final deterministic assignment tie.
It is not an assertion that one backend wins every phase and sparse geometry:
CPU/GPU and GPU/GPU kernels can have legitimate batch-size crossovers. Exact
measured phase service time therefore remains the primary runtime objective.
The engine must preserve and expose a crossover rather than reject it, clamp it
to the configured order, or infer a cost from a device-type label. Operators
who require a different residency preference express it through capacities and
priorities; the optimizer never rewrites those quotas.

Movement itself has a cost. Counts are smoothed across immutable histogram
generations, and an incumbent remains preferred on a deterministic tie. A
candidate cycle is admitted only when its projected service-time gain over the
configured horizon exceeds transfer/repack cost, measured interference, and a
hysteresis margin. Minimum residency generations and a deterministic
expert-ID final tie break prevents occupants from oscillating on noisy
windows. Bounded waves may approach the global target over several epochs;
every admitted wave must have positive predicted benefit and preserve all
capacities. The number of independent closed cycles admitted to one wave is a
configuration policy, not an architectural constant.

### Measured economy and asynchronous-overlap contract

The economy certificate prices the production protocol, not an estimated link
bandwidth. Setup builds capacity-preserving reciprocal expert swaps for
representative manifest-equivalent layers. Each attempt first records an
ordinary production inference interval. It then reserves the same physical
lanes used by maintenance, arms an exact-workload concurrent probe, and releases
the transfer only after that live inference ticket reports `Running`. The
candidate bank is never published. All ranks accept a sample only when the
concurrent inference interval wholly contains the transfer stage interval;
otherwise the wave is aborted, cleaned up, and retried with a fresh baseline.
Warmups are discarded and the sealed profile uses robust medians.

For a histogram window containing `W` routed tokens and a configured payoff
horizon `H`, the policy computes:

```text
window_gain = phase-aware service savings - phase-aware service penalties
projected_service_gain = floor(window_gain * H / W)
projected_net_benefit = max(
    0,
    projected_service_gain - transfer_and_repack - interference)
```

A same-priority participant swap uses the reduction in the domain's maximum
participant makespan rather than aggregate tier work, which is unchanged by the
swap. Cross-priority movement uses the demand-weighted difference between the
source and destination phase service costs. The strict admission gate also
requires `projected_net_benefit` to exceed the configured minimum and every
expert to have satisfied its minimum-residency generations.

Cost composition follows the calibration shape. Gate, up, and down projections
run concurrently and one reciprocal two-expert swap is calibrated as one wave.
Its two directed rows therefore contain the same critical path and the cycle is
charged the slower row once, not once per expert. This corrects the former
double charge. A longer cycle, or multiple separately calibrated cycles admitted
to one wave, is conservatively additive today. The destination design is a
typed physical contention graph: each conversion engine, PCIe root, GPU peer
fabric, NUMA memory controller, UPI link, NIC, and maintenance CPU executor is a
resource group; calibrated edge intervals may take a maximum only when their
groups are disjoint or the exact composite wave shape was measured. This is
required before a multi-cycle scheduler may claim a less conservative price.

The homogeneous GPU controller and current-batch LLEP use the same concurrency
rule for their row-based admission floors. CUDA and ROCm compact-transfer waves
price `payload_bucket_requested_slots`, the maximum source-lane payload ordinal,
rather than the global command count. A reciprocal ownership swap therefore
publishes two logical arrivals but has one serialized payload slot. The shared
CPU/CUDA/ROCm LLEP planner now prices the maximum outgoing or incoming payload
count incident on any participant: reciprocal and disjoint edges overlap,
while two sends or receives on one participant remain additive. The offline
trace evaluator uses the identical multiplier and reports both logical
`weight_transfer_count` and `critical_path_transfer_slots`; PerfStats likewise
keeps logical movement separate from critical-path payload width.

This closes the duplicate-edge accounting defect, but the two controller
families still have different evidence classes. Durable ExpertOverlay movement
uses measured nanoseconds and an amortization horizon. Homogeneous device-side
maintenance and request-scoped LLEP currently use configured row/spread floors;
they contain no reservation-to-dispatch clock that can contaminate cost, but
the floors are policy proxies rather than measured time. The destination is a
setup-certified, device-resident GPU cost table and a request-local LLEP
break-even model that compares transfer preparation against savings in that
prefill batch. Neither may borrow the durable 65,536-token horizon.

Asynchronous means inference never waits on reservation, transfer, commit, or
retirement. It does not mean migration consumes no shared hardware. The
certificate retains interference as a separate term precisely because CPU
scheduler time, memory bandwidth, UPI/PCIe bandwidth, MPI progress, and GPU
execution resources can slow an otherwise independent inference stream. The
maintenance worker has its own `std::jthread` and event-polls every operation;
the inference worker neither polls nor joins it. Admission-to-dispatch delay is
reported separately from active transfer wall time so waiting for an exact
calibration ticket or distributed reservation vote can never be mislabeled as
wire cost.

The 2026-08-14 Qwen 3.5 35B-A3B CPU NodeTP ordinal campaign on this two-socket
host provides the first audited example after those accounting fixes:

| Quantity | Observed value | Interpretation |
|---|---:|---|
| Reciprocal pair-wave transfer/repack price | 17.091 ms | Complete concurrent two-expert wave; not a per-expert cost |
| Worst robust inference interference | 5.168 ms | Added latency charged separately from movement |
| Projected service gain over 65,536 routed tokens | 2,603.981 ms | Horizon projection, not elapsed time for the short parity request |
| Projected net benefit | 2,581.722 ms | Service gain minus the two measured cost terms |
| Calibration admission-to-dispatch delay | 33.8--34.7 ms mean by projection lane | Intentional wait for the exact live ticket; excluded from movement economics |
| Active remote projection wall | 12.7--19.6 ms mean by projection lane | Manifest, header, payload, endpoint work, and observed MPI progress |
| Endpoint host work | 1.0--2.1 ms mean by projection lane | Checksum, validation, final CPU copy, and publication work |
| MPI request-visible time | 11.7--18.0 ms mean by projection lane | Includes time until maintenance observes completion; it is not raw UPI latency |

The representative expert payload is 2,686,976 bytes at layer 0 and 3,407,872
bytes at layer 10 in each direction. The remaining cost is therefore dominated
by authenticated endpoint work and the three-step MPI request protocol rather
than a single bulk-copy limit. Follow-up optimization must benchmark checksum,
copy, request progression, maintenance affinity, and scheduler interference as
separate production-shaped candidates. Any checksum replacement is a protocol
ABI change and needs cross-backend byte-corruption tests; any affinity or
priority policy must be topology-derived and configurable, never hard-coded to
a socket or rank.

### Current implementation audit

The current code installs setup-time RAM/VRAM capacity resolution and the
complete measured-economy lifecycle in the production runner. It has the
phase-separated histogram, exact fixed-quota service-cost optimizer,
incumbent-preserving tie break, smoothed payoff gate, minimum-residency
hysteresis, real non-publishable movement calibration, live prepared-expert
service accumulation, owner-authenticated MPI evidence exchange, and atomic
certificate installation inside the residency authority. Maintenance cannot
request its first proposal until that certificate is installed. The real-model
topology matrix is now certified; the remaining gaps are listed explicitly:

| Behavior | What the code does now | Gap to the destination |
|---|---|---|
| Priority fill | `OrchestrationRunner` gathers physical rank/device budgets, `MoEOverlayCapacityResolver` resolves layer-varying live quotas in ascending signed `priority`, installs them into the frozen plan, and `MoERoutedExpertPlacementPlanner` consumes those exact quotas. Shared validation rejects duplicate priorities and a coverage tier that is not numerically last. | Installed, name/declaration independent, and real-model topology proven. Complete placement-explanation output remains. |
| Fixed expert cap | `max_experts_per_layer` is an upper bound on every resolved layer quota. `resolved_live_experts_per_layer` represents the immutable fixed result consumed by placement, distributed plan hashing, and graph diagnostics. | The user-facing scalar cap is still uniform. If explicit layer-varying fixed quotas are to be user selectable, configuration and CLI need a typed spelling rather than exposing the setup-owned resolved field. |
| Byte cap | The GGUF manifest supplies exact projection geometry and codebook identity. For every one of the 21 catalogued formats, the resolver prices a reusable GPU allocation as the union of the loader's compact representation and the migration-stable CPU-promotion representation, including every required scale, minimum, and embedded-minimum region. `memory_budget_bytes` remains only a legacy tier-local upper bound; zero/`auto` no longer means unbounded because physical admission still applies. | Retire or rename the ambiguous tier-local `memory-mb` spelling and expose physical participant limits/reserves clearly in configuration, CLI help, and `--explain-placement`. |
| Fallback capacity | Automatic preferred tiers consume safely admitted slots first; the fallback tier must admit the exact remainder under its physical budgets and upper bounds. A one-expert shortfall fails setup before graph construction. | Installed and real-weight target-topology proven. Campaign-visible failure diagnostics for deliberate one-slot shortfalls remain. |
| Hardware admission | `MoEOverlayLocalCapacityPlanner` uses the production `MemoryPlanner` to charge dense weights, graph arenas, KV cache, and headroom, joins tier participants to unique physical rank/device authorities, and adds transfer staging and shadow slots. The runner gathers those budgets across ranks and searches resident captured-prefill row candidates before freezing placement. GPU admission uses reported free memory and optional limits. CPU rank inventory publishes both exact owned-NUMA total bytes and current available bytes; admission uses the smaller authority, so another socket or the parity ramdisk cannot be counted as local free capacity. | Installed and proven with both socket-owned CPU endpoints while the inventory binder remains free to place either GPU backend on either MPI rank. Placement-explanation detail remains. |
| Initial membership | Absent histogram evidence, expert IDs are filled ordinally into the priority tiers. `owner_order` then distributes members ordinally or by a deterministic random permutation within each tier. | There is no separate typed random cold-start membership policy or persisted-profile seed. |
| Histogram membership | The histogram rotates immutable decode, real-prefill, and accepted grouped-verifier source banks. `HistogramTieredCache` and `RoutedTierRebalanced` consume a total certified per-tier/layer/active-phase service profile and solve the exact fixed-quota assignment, minimizing measured service cost first, incumbent movement second, and expert id last. The residency authority smooths generations and applies measured transfer/interference payoff plus minimum-residency hysteresis before admitting a cycle. The production runner now calibrates representative exact-weight layers with real abort-only transport waves, expands only manifest-equivalent layers, gathers owner-authenticated rows over a private non-blocking MPI lane, and installs the composed certificate before proposal admission. | Installed and local/distributed protocol proven. Real-weight two- and three-tier tests prove complete certification, positive projected service/net benefit, and improving residency scores on every rank. The observed speed gate currently rejects the resulting path. |
| Physical migration headroom | Production preallocates one inactive RCU shadow slot per endpoint/layer and a 4 MiB staging lane, retains old/new residency banks, and adopts every loader-owned initial live slot into the recyclable physical arena. `migration_max_cycles_per_wave` is a positive scheduling cap (default `1`) and may admit multiple independent cycles when their aggregate endpoint/layer shadow requirements fit the installed BOM. The same slot policy is charged during admission, including the union of compact loader bytes and the largest migration-stable arrival representation plus a 128 MiB GPU safety reserve, then reused during physical-fabric construction. | Slot recycling and the public cycle cap are installed. A real-weight three-tier run with cap `2` proves full two-cycle waves, loader-slot adoption, and post-retirement reuse. The staging chunk size and per-endpoint shadow depth remain production constants; changing either requires model-aware readmission plus real-model backpressure/interference certification. |
| Production proof | Dynamic real-weight CUDA/CPU, ROCm/CPU, CUDA/ROCm, and CUDA/ROCm/CPU inference proves seeded-random/adversarial residency, histogram-driven promotions/demotions across domains, ranks, and backend types, repeated improving epochs, tunable cap-`2` waves, non-blocking background preparation, numerical parity, and the canonical CSV artifact set. Static CUDA/ROCm proves zero movement; segmented tri-tier prefill proves its declared heterogeneous capture boundary. | The observed inference gate is now present and red. Endpoint traces show small continuation-GPU packets around 0.62--0.67 ms versus roughly 0.29--0.51 ms for CPU packets because every layer still pays full-bucket host publication and constructs a fresh GPU packet stage. Install the captured local branch, then rerun the same gate. |

Therefore the answer to "does it work that way now?" is: **priority-ordered
capacity budgeting, phase-aware economic planning, and live two-/three-tier
migration are implemented and real-model correctness certified, but the
current segmented execution path fails the observed performance requirement**.
The implementation determines exact layer quotas from the model and
per-participant physical budgets, then admits only cycles whose measured phase
benefit pays for measured transfer and interference. For MTP-disabled
instances the typed active-phase mask requires decode and prefill only;
grouped verification stays zero instead of stalling certification or borrowing
a decode estimate. The remaining proof must show that accepted movement
improves measured inference latency on the target hardware.

### Configuration and observability destination

The configuration model must represent these concepts independently with typed
fields: strict tier priority, fixed-versus-automatic live capacity, per-physical
participant memory limit/reserve, optional per-layer expert caps, cold-start
membership policy, participant owner order, shadow concurrency, and migration
payoff/hysteresis policy. `fallback=true` means final coverage responsibility;
it must not imply unlimited memory. Before retaining the existing CLI spelling,
the parser must stop treating `memory-mb=auto` as the same value as "no cap".

`--validate-only`, `--dry-run`, and `--explain-placement` must print, for every
participant and layer, the budget source, fixed bytes, exact expert bytes,
resolved live quota, replica/shard multiplier, shadow/staging bytes, reserve,
and remaining headroom. The same resolved-plan identity and BOM are included in
graph-cache identity and distributed consensus. Setup and every migration
proposal publish `PerfStats` for quota occupancy, phase-specific hit rate,
expected service-time gain, transfer cost, hysteresis rejection, and the
hottest expert assigned to each tier. Tests assert these values rather than
inferring success from a tier name or a nonzero copy counter.

## Authorities and ownership

### Live tier residency

One live participant owns each resident expert. Its prepared gate/up/down
engines are the only transfer source for a promotion or demotion. GPU weights
use the common separated accelerator packing consumed by CUDA and ROCm. CPU
weights use the CPU NativeVNNI interleaved packing. Migration reorders an
already-quantized representation and copies its metadata; it never dequantizes,
requantizes, or asks the model loader to recreate an evicted expert.

Each `(domain, participant, layer)` endpoint owns persistent destination slots
and conversion staging sized before captured execution. GPU engine pointers and
runtime placement banks remain stable for the graph lifetime. CPU engines are
host-owned and NUMA-placed according to the endpoint's typed policy.

The physical arena is the endpoint's resolved live allocations plus its
separately charged RCU shadow allocations. Initial live allocations are
prepared by the loader before the migration fabric exists, so the fabric
adopts their exact storage as initially occupied slots instead of copying the
weights into a second arena. The first migration normally arrives in a shadow
slot. Only after the old epoch's final inference ticket and distributed retire
fence have drained may the departed loader slot become a free destination for
a later epoch.

Every assignment made after bootstrap carries an aliasing lease token through
the candidate bank, published bank, transfer operations, and retirement work.
Destroying the final alias after retirement returns that exact physical slot
to the endpoint arena. Consequently one shadow slot is overlap capacity, not a
lifetime limit of one migration: live and shadow allocations alternate roles
across arbitrarily many epochs without allocation, rereading the model file,
or overwriting a pointer reachable by an old ticket. Retirement publishes
`bootstrap_live_slots_recycled`, `adopted_live_slot_acquired`, and ordinary
pool-release evidence through `PerfStats`.

#### Live accelerator format and recyclable allocation capacity

The execution descriptor and the physical allocation descriptor are distinct
typed contracts. The execution descriptor names the bytes that the current
CUDA or ROCm GEMM consumes. The allocation descriptor names every byte that
the slot can safely hold after any permitted tier round trip. A loader-owned
slot may therefore keep its compact initial representation without an eager
normalization copy, provided that its allocation was admitted and created for
the union of:

- the source format's compact payload, scales, minima, and embedded minima; and
- the migration-stable GPU representation produced by CPU-to-GPU promotion.

Nibble codebooks and Q6_K keep their native 16- or 24-byte payloads across the
CPU boundary. Other compact formats may return as signed INT8 with a 32-byte
payload and, for asymmetric sources, an FP16 minimum under execution codebook
23. For example, a Q5_K loader slot executes its original 20-byte payload but
owns 32 bytes of payload capacity per logical block, so a later expanded
promotion can reuse that exact address. Q2_K retains its source embedded-minimum
region even though the normalized arrival does not need it. The exhaustive
catalog contract, rather than a hand-picked format list, defines this union for
all 21 supported source formats.

Coalesced expert preparation preserves ordinary GEMM layout: bytes remain
tightly packed within each expert, and only the base of the next independently
addressable expert advances by the wider allocation capacity. There is no
per-block padding in the live matrix. This prevents a future expanded arrival
from overwriting its neighbor without changing the kernel's arithmetic or
requiring a second loader copy. Capacity admission, loader allocation, prepared
engine construction, physical-fabric adoption, and transfer destination
validation all consume the same typed allocation descriptor and fail if they
disagree.

Every cached prefill bucket and decode graph for an endpoint resolves the same
prepared-store slab identity. A transport never keeps an untracked raw pointer
to a graph-cache stage. Stages consume a participant residency object whose
lifetime exceeds every graph-cache entry.

### Residency publication

`MoEOverlayResidencyAuthority` is the only owner of the live expert owner map.
It publishes immutable snapshots with monotonically increasing epochs.
The authority combines one immutable routing-demand generation with the
setup-certified capacity and service profile; no mutable runtime table or
loader state may influence a proposal. Tier priority is compute-capacity order:
lower numeric priority is more preferred. Candidate planning minimizes the documented
phase-weighted service objective, uses incumbent residency and expert ID as
deterministic tie breaks, and preserves each tier/layer and participant capacity
exactly.

### Dispatch tickets and multi-version retirement

One request-lifetime device ticket selects `{epoch, bank, generation}` before
the first captured MoE boundary. Every continuation-local layer reads the bank
from that ticket rather than the mutable published selector. Every segmented
boundary copies the same epoch into dispatch ticket ABI v3; CPU and remote
participants acquire that exact epoch from the host authority. The final
captured continuation consume releases the device reader and the final host
return releases its exact host lease. Ticket admission never closes for weight
preparation or transfer.

Publication is read-copy-update (RCU), not a stop-the-world maintenance gate.
Tickets already bound to epoch `E` continue using `E` while an inactive bank for
`E+1` is prepared. After the candidate bank's ready event completes, one atomic
publication makes new tickets acquire `E+1`. The immutable snapshot, prepared
engines, runtime descriptor bank, and slot leases for `E` remain owned by an
epoch retirement record until the last `E` ticket releases. Retirement is
performed by the maintenance worker; a final-return thread only decrements its
epoch lease counter and never frees weights or waits for a stream.

Host-current ticket acquisition increments a per-epoch counter and rechecks the
published epoch. Exact acquisition instead searches only the current and single
retiring generation, increments its counter, and rechecks both addressability
and the retirement gate. This lets a device ticket for `E` arrive after host
publication of `E+1` without pairing its packets with the wrong owner map.

Device admission uses one 64-bit packed selector as its linearization point.
Before reading it, the captured acquire kernel increments
`acquisitions_in_flight`; it then increments the selected physical bank's
reader count before dropping that guard. Publication can select the ready peer
bank immediately. Reuse of the old bank requires both its reader count and the
global acquisition guard to be zero, closing the otherwise possible delayed
reader-increment race. The selector generation prevents ABA when physical bank
indices alternate over many epochs. Bucket selection changes ticket data,
never graph topology or embedded addresses.

The control block is device-owned. CUDA and ROCm use equivalent captured atomic
acquire/release kernels and maintenance-stream publication kernels over the
shared ABI; the host does not download it to decide inference. Maintenance
polls a bounded retirement-ready result and exact event, just as it polls
transfer readiness. CPU execution operates directly on the same ABI through
`MoEOverlayDeviceEpochProtocol`.

Current implementation boundary: the CPU protocol/oracle and exact host lookup
are installed and green. The existing GPU ticket publisher still copies only
logical rows, routes, weights, and hidden data; host dispatch chooses the
current epoch and writes it into the header. Until the CUDA/HIP acquire kernel,
epoch D2H field, bank-selected local kernels, and final release kernel are
composed, this is correctness evidence for the segmented path rather than the
target no-bounce execution path.

### Arbitrary-tier cycle scheduling

A full tier is never updated by overwriting one expert at a time. For each
layer, the planner represents the old-to-target delta as a directed multigraph
whose vertices are exact physical participants and whose edges are complete
expert moves. Fixed participant capacity makes every vertex balanced, so the
graph decomposes into closed directed cycles. A same-participant descriptor
replacement is a one-edge cycle; a same-tier move between two participants is
part of the corresponding participant cycle. A hot/warm/cold rotation is
therefore one closed
`cold -> hot -> warm -> cold` transaction, not three independently publishable
copies.

`MoEOverlayResidencyTransaction` carries the deterministic cycle decomposition
and an exact inactive-slot requirement for each
`(layer, destination tier, destination participant)`. Before any byte moves,
the transport atomically reserves every requirement and pins every source in
the selected cycle set. A simple cycle needs one inactive expert slot in every
participating endpoint. Multiple independent cycles may share one background
wave only when the sum of their endpoint requirements fits the currently free
adopted-plus-shadow arena and the configured
`migration_max_cycles_per_wave` scheduling cap. Otherwise the planner emits a
smaller capacity-preserving cycle wave or defers it; it never exposes a
path-shaped partial placement.

This generalizes to any positive tier count. Conversion and transport edges in
one or more cycles execute concurrently subject to per-lane bandwidth budgets,
while publication of residency epochs remains serialized. At least one
preallocated inactive slot per participating endpoint permits a single simple
cycle to make progress. The cycle cap is independently tunable and is never
assumed to be one by the protocol; actual admission is still bounded by the
resolved per-endpoint/layer arena BOM. Raising it can exploit disjoint cycles
without increasing memory, while a topology requiring multiple simultaneous
arrivals at one endpoint must first provision and readmit deeper shadow
capacity. Runtime allocation is forbidden.

### Histogram generation rotation

Planning freezes one complete routing-evidence generation and immediately
rotates inference to a fresh preallocated histogram bank. The frozen generation
is retained by the candidate transaction, so its expert temperatures and
placement decision cannot change while transfer is in flight. Decode, grouped
MTP, and prefill routes continue accumulating in the new bank throughout that
wave. Commit updates ownership but does not clear the new bank. Abort also
leaves it intact. This avoids losing potentially thousands of observations when
a cross-domain migration lasts longer than one histogram window.

CPU histogram writers use the same RCU acquire/recheck pattern as residency
tickets. Rotation switches banks atomically and the maintenance worker waits
only for writers already pinned to the frozen bank; inference writers never
wait for maintenance. Device histogram sources rotate their persistent device
banks on their exact maintenance event and merge the frozen bank only after its
ready event is queryable. There is no inference-stream synchronization or host
shadow of a live device counter bank.

## Lossless transfer formats

### GPU to GPU

CUDA and ROCm share `GpuExpertPackedDescriptor`: separated quantized payload,
FP16 scales, optional FP16 minima, and optional extended minima. Movement is a
length-checked blob transfer with no format conversion. Local peer, staged
host/network, and cross-rank transports all preserve the same arrays byte for
byte. Completion is published from the exact destination stream/event.

### GPU to CPU

The source GPU transforms its separated accelerator arrays into final CPU
NativeVNNI interleaved bytes while producing bounded stream chunks. Payload
bytes are transposed into the 64-row CPU grouping and scale/min/compensation
metadata is written at its final inline offsets. The destination CPU receives
those chunks directly into its final NUMA-owned buffer; it performs no repack.

Conversion is representation-exact. Compensation values are derived with the
same fixed integer order as the canonical CPU packer. Formats whose CPU engine
uses additional eager sections must have those sections produced by the source
GPU protocol too; an unsupported codebook is a fatal planning error.

### CPU to GPU

The source CPU streams its final NativeVNNI interleaved bytes. The destination
GPU receives bounded chunks into persistent staging and launches a conversion
kernel on its exact maintenance stream as chunks become ready. That kernel
deinterleaves payload bytes and extracts scale/min/extended-min metadata into
the common accelerator arrays. The destination does not require raw GGUF tensor
views and does not retain the CPU representation after commit.

### Background execution and streaming invariants

The wire manifest fixes layer, expert, projection, geometry, codebook,
asymmetry, source packing, destination packing, total bytes, chunk geometry,
residency epoch, and checksum before data moves. Chunks have monotonic sequence
numbers and exact final ranges. Persistent double-buffered staging slots use
producer/consumer events; there are no null streams, blocking synchronizations,
hot-path allocations, or host callbacks. A missing, duplicate, stale, or
out-of-range chunk aborts the candidate epoch.

All repack kernels and DMA execute in a background migration execution domain
owned by the rebalance service. It reuses the device context's persistent
auxiliary-stream machinery, but it does not share event handles with captured
inference stages. The implementation exposes two typed lanes:

- a low-priority conversion lane for GPU source/destination repack kernels;
- a low-priority transport lane for peer copies, H2D/D2H, and collectives.

One-lane execution is permitted only when the isolated performance corpus proves
it is more economical. In the two-lane pipeline, conversion records a per-slot
ready event, transport waits on that exact event, and conversion waits on the
slot's transport-consumed event before reuse. This permits conversion of chunk
`n+1` to overlap transport of chunk `n` without sharing a staging address.

The ordinary inference/capture stream never waits on either migration lane.
Event queries and network progress run on a maintenance worker, not an inference
worker. GPU stream priority and bounded chunks reduce compute interference;
backpressure defers additional migrations when staging, bandwidth, or inactive
slot capacity is exhausted. Deferral leaves the current epoch live and is not a
fallback placement.

## Transaction protocol

For a candidate epoch `E+1` planned from live epoch `E`:

1. Acquire a maintenance-wave lease and validate that the proposal references
   exact published snapshot `E`; ticket admission remains open.
2. Pin every source engine/slot named by the wave so epoch `E` and its live
   tickets cannot lose a source during background preparation.
3. Resolve each movement to typed source and destination participant endpoints.
4. Reserve inactive destination engines, final slots, and persistent
   streaming/conversion slots. If shadow capacity is unavailable, defer the
   wave without evicting a live slot or stalling inference.
5. Enqueue all three projections directly from each old owner on the background
   lanes and return to inference immediately.
6. The maintenance worker advances chunk events, checks manifest checksums, and
   verifies every candidate expert has three complete prepared engines.
7. Build and upload the inactive runtime descriptor/mask bank for `E+1` on each
   participant's exact migration stream. No pointer reachable through epoch `E`
   is overwritten. Candidate readiness is an event, not a stream wait.
8. After every destination and descriptor-ready event is queryable, atomically
   compare/exchange the host residency authority from exact `E` to `E+1`. A
   stale compare aborts the unpublished candidate.
9. Enqueue each GPU's tiny device-selector publication after its candidate-ready
   event. Host-first ordering is intentional: a lagging device may still emit an
   exact `E` ticket, which the host retiring slot accepts; no device may emit
   `E+1` before the host can resolve it. No inference stream waits on this work.
10. New device tickets use `E+1`; existing `E` tickets continue through their
    retained bank. Move `E` and its old slot/engine leases to the retirement
    queue.
11. Once the host `E` lease count, every device-bank reader, every acquisition
    guard, and the distributed retirement fence are clear, the maintenance
    worker retires excluded engines, returns bootstrap assignments or later
    aliasing leases to the physical arena, and publishes final `PerfStats`
    evidence. The next histogram generation has already been live since
    proposal time and is never reset by this retirement.

Failure before publication aborts destination staging and leaves epoch `E` and
all old owners live. A failure after any endpoint exposes a candidate bank is
fatal; there is no eager replay, recapture, archive reload, or alternate
placement, or synchronous inference-thread cleanup.

## Multi-domain and cross-rank movement

A three-tier CUDA-hot/ROCm-warm/CPU-cold wave may contain multiple independent
edges. Each migration still has exactly one live source, one destination, and
one conversion authority:

- accelerator-to-accelerator edges transfer common packed blobs;
- accelerator-to-CPU edges convert on the source accelerator;
- CPU-to-accelerator edges convert on the destination accelerator.

For cross-rank edges, metadata and bounded chunks traverse the overlay
communicator. Rank-local endpoint services stage their destinations and join a
two-phase transaction vote. All ranks acknowledge prepare before the root
publishes `E+1`; epoch and manifest checksum are included in every command. The
standard 30-second collective timeout applies.

The remote service accepts only typed `StageWave`, `CommitWave`, `AbortWave`,
and `RetireWave` commands. Repetition is idempotent only for an identical epoch
and manifest checksum. A stale epoch, unknown participant, missing projection,
null GPU stream, or incomplete mask is fatal.

## PerfStats proof contract

Dynamic and LLEP integrations assert positive applicable counters for checks,
committed waves, committed migrations, promotions, demotions, cross-domain,
cross-rank and cross-backend migrations, exact source/destination bytes,
conversion chunks/bytes, dispatch leases, final-return releases, captured
ticket publication/consumption, heterogeneous segments, background waves,
inactive-bank publications, old-epoch retirements, bootstrap-slot recycling,
and subsequent adopted-slot acquisitions. They also assert zero
inference-stream migration waits, zero migration stream synchronizations, zero
active-slot overwrites, and at least one interval where inference epoch `E`
completed while preparation of `E+1` remained pending.

The CUDA-hot/ROCm-warm/CPU-cold integration must move experts across at least
two distinct domain edges and prove both promotion and demotion. The
CUDA-hot/CPU-cold integration must prove movement in both directions. Static
integrations assert the maintenance check ran while all movement, transfer,
conversion, promotion, and demotion counters remain exactly zero.

## Verification layers

1. Device-free units exhaust manifest validation, chunk ordering, byte-layout
   maps, transaction races, stale epochs, rollback, capacity selection,
   ordinal/random placement, ticket-acquire/publication races, overlapping old
   and new epoch leases, deferred retirement, shadow-capacity deferral, and
   repeated byte-exact migration through at least three waves with one shadow
   slot per endpoint.
2. CPU-only protocol integrations emulate both accelerator layout endpoints,
   exercise the exact streaming state machine and reference conversions, and
   compare every resulting packed section byte for byte.
3. Backend integrations execute real CUDA/ROCm conversion kernels, stream/event
   publication, prepared-engine replacement, and same-packed GPU transfers.
4. Real-weight model parity campaigns run production capture and compare live
   checkpoints to the Hugging Face reference with canonical CSV artifacts.
5. Campaign orchestration enforces one aggregate wall-clock budget below one
   hour across every backend, precision, model, and test type.

## Real-weight migration proof (2026-08-14)

The isolated Qwen 3.5 MoE graph-native campaigns now certify every backend
pairing plus the three-tier topology under real weights and two production MPI
instances:

| Residency cell | Backends | Wall time (seconds) | Result |
|---|---|---:|---|
| Dynamic seeded-random | CUDA + CPU | 71.550 | Numerical/path pass; convergence-speed gate red |
| Dynamic seeded-random | ROCm + CPU | 73.593 | Numerical/path pass; convergence-speed gate red |
| Dynamic seeded-random | CUDA + ROCm | 22.904 | Numerical/path pass; convergence-speed gate red |
| Dynamic seeded-random | CUDA + ROCm + CPU | 124.245 | Numerical/path pass; convergence-speed gate red |
| Static seeded-random control | CUDA + ROCm | 11.015 | Pass, zero movement |
| Segmented prefill | CUDA + ROCm + CPU | 14.072 | Pass |

Every cell stayed within the shared campaign wall-time target and emitted the
seven required CSV artifacts: decode layers, decode stages, decode steps,
prefill layers, prefill stages, prefill summary, and production path. Every
live checkpoint remained within the authenticated Hugging Face tolerances.
That is not an economy pass: once the campaign added a required observed
convergence improvement of at least two percent, the current dynamic path was
roughly 36--50% slower after movement. The segmented cell still certifies its
ordered heterogeneous capture boundary and full-prompt reconstruction; the
captured continuation-local branch is the next implementation dependency.

The dynamic cells used real CUDA, ROCm, and/or NodeLocalTP CPU participants as
declared. `PerfStats` proved seeded-random or authenticated adversarial initial
layouts, at least two improving residency epochs per rank, positive projected
service gain and net benefit, balanced promotions and demotions, all applicable
cross-domain/rank/backend traffic, background physical preparation,
inactive-bank publication, old-epoch retirement, loader-slot adoption, and
subsequent bootstrap-slot recycling. The public
`migration_max_cycles_per_wave` value was `2`; every rank published that value
and no wave exceeded it. The three-tier cell admitted a full two-cycle wave on
every rank. There were no inference-stream migration waits, blocking migration
synchronizations, active-slot overwrites, or migration failures.

Transport capacity follows that public knob rather than assuming one cycle.
`MoEOverlayRemoteProjectionLaneBudget` materializes exactly
`maximum_participants_per_cycle * maximum_concurrent_cycles * 3` projection
lanes per rank, with checked zero/overflow rejection. The tri-tier proof thus
used 24 preallocated remote projection lanes per rank (four participants, two
cycles, three projections). Config parser tests cover other positive values,
and changing the cap requires model-aware slot admission rather than a source
change.

Separately, `V2_Integration_CUDAVnniUnpackKernels` and
`V2_Integration_ROCmVnniUnpackKernels` sweep all 21 source formats through
three coalesced recyclable experts and compare the entire capacity-sized output
against independently prepared experts byte for byte, including unused compact
tails and all metadata regions. This is the device proof that the loader layout
used by the real model and the later migration layout share one safe physical
allocation contract.

### Homogeneous GPU MTP proof (2026-08-14)

The registered Qwen3.6 CUDA2 and ROCm2 ExpertOverlay campaigns now pass all 12
fresh/prefix-restore combinations at fixed depth 2, fixed depth 3, and dynamic
depth in 332.09 and 365.53 seconds respectively. The combined CSV audit covered
8,544 prefill-stage rows, 832 decode-stage rows, 107,648 sidecar checkpoint
rows, and 64 transaction-trace rows, with every aggregate pass bit true and no
non-finite production or reference value.

HIP's retained graph reuse is certified as production evidence, not inferred
from requested topology. A reuse record is accepted only when it authenticates
the HIP backend, hosted execution policy, request and depth geometry, sampling
mode, fragment geometry, and positive workspace generation. The request-local
ticket/submission/reduction/terminal ledgers must still agree exactly on every
participant. This matters for prefix restore: the seed request may materialize
the executable, while the restored request truthfully reports an exact reuse
after request-local counters have been reset.

## Performance certification baseline (2026-08-13)

The registered Release targets exercise all 21 NativeVNNI source formats, both
Qwen MoE projection geometries (`N=512,K=2048` gate/up and
`N=2048,K=512` down), both conversion directions, and kernel-only plus
kernel-and-pinned-DMA modes. Every fixture first performs a byte-exact full
round trip. Each timing row is the median of three independent 20-iteration
event batches after five warmups and records its economy floor and pass bit in
CSV. PCI sysfs selects the GPU's actual NUMA node before pinned memory is
allocated, so moving a card between sockets does not invalidate the test.

The canonical registered invocation completed both backends in 6.46 seconds
of aggregate wall time. It emitted 168 rows per backend, 42 rows per
backend/mode/direction combination, with zero malformed, unvalidated, or
uneconomical rows. The minimum and arithmetic-mean semantic throughput in the
retained evidence was:

| Backend | Mode | Direction | Minimum GB/s | Mean GB/s |
|---|---|---|---:|---:|
| CUDA | kernel | GPU to CPU | 69.189 | 206.945 |
| CUDA | kernel | CPU to GPU | 43.554 | 62.595 |
| CUDA | streaming | GPU to CPU | 7.237 | 8.280 |
| CUDA | streaming | CPU to GPU | 6.615 | 6.797 |
| ROCm | kernel | GPU to CPU | 64.962 | 114.382 |
| ROCm | kernel | CPU to GPU | 116.125 | 210.974 |
| ROCm | streaming | GPU to CPU | 8.532 | 9.667 |
| ROCm | streaming | CPU to GPU | 8.592 | 10.040 |

The production 4 MiB staging lane can hold all 512 complete CPU packing units
for either model shape. A 64/128/256/512-unit sweep confirmed that 512 units is
the economical production point on both backends, so the benchmark default
matches the installed lane rather than measuring an artificial tiny chunk.

Nsight Compute isolated every distinct CUDA conversion specialization with
`--single-launch`. Across 16 demotion and five promotion specializations it
reported zero local-memory spilling requests. Demotion used 32--56 registers
per thread and promotion used 40; achieved occupancy was 17.7--23.9 percent
against 66.67 percent theoretical occupancy. The complete 512-block workload
is only 0.39 waves per SM on an RTX 3090, so the occupancy delta reflects a
short bounded migration kernel rather than register or spill pressure.

On gfx906, `rocprofv3` counter passes were isolated into compatible SQ and
memory groups. Seven physical kernel-family representatives reported zero
scratch, 12--48 runtime VGPRs, 32 SGPRs, and 512 wavefronts. Static metadata
and ISA inspection covered every emitted specialization: private segment size,
VGPR spill count, and SGPR spill count were all zero; VGPR use was 11--58 and
SGPR use was 18--28; no scratch, flat spill, or LDS instructions were present.
Raw profiler captures are intentionally temporary and are not repository
artifacts. This dated summary and the reproducible Release targets are the
durable evidence.

## Current completion sequence

The remaining implementation should be completed in this order so no test-only
composition becomes an accidental second production path. The model-aware
capacity resolver, quota installation, and setup-time physical BOM are complete
foundations rather than remaining design work.

1. Finish the capacity system's user-facing boundary: replace the legacy
   tier-local byte-cap wording with typed physical participant limits/reserves,
   expose the installed per-layer quotas and complete BOM in
   `--explain-placement`. Retain the now-green per-NUMA available-memory
   admission cells for the real two- and three-tier model topologies.
2. Retain and certify the installed production economy profiler/composer. It
   measures exact prepared gate/up/down service for every runtime-active phase,
   every directed participant transfer/repack edge, and paired overlap
   interference; authenticates and distributes the rows; and installs the
   profile before maintenance proposal admission. Add real-model assertions
   for complete phase evidence, identical identities, abort-only calibration
   waves, and identical proposals on every rank.
3. Retain the installed all-codebook capacity, apportioned/replicated, shared
   resource, fallback-failure, shadow-headroom, and captured-graph admission
   sweeps. Extend policy coverage with ordinal and seeded-random membership and
   ownership, varying NUMA/VRAM availability, strict integer-priority fill, and
   noisy-window non-oscillation. Add focused configuration/parser and
   placement-explanation regressions for `auto` versus explicit limits.
4. Certify the installed shadow/staging materialization and remote inactive
   prepared-engine banks on their exact background streams. Prove decode,
   grouped MTP, prefix restore, and bucketed prefill tickets retain one epoch
   while a new epoch is prepared, with no inference-stream join.
5. Retain the green real-weight production parity slice for histogram-forced
   dynamic promotion/demotion. CUDA/CPU, ROCm/CPU, CUDA/ROCm, and
   CUDA/ROCm/CPU cells use explicit integer priorities, canonical checkpoint
   CSVs, and the complete capacity/benefit/movement `PerfStats` contract. Keep
   static zero-movement and segmented-prefill cells as controls.
6. Measure inference-only versus inference-plus-migration on real models,
   including concurrent internal stream pools, and gate overlap loss, staging
   backpressure, migration deferral, quota hit rate, projected-versus-observed
   benefit, and zero inference-stream waits through `PerfStats`.
7. Fold those cells into the aggregate parity campaign, skill, precommit, and
   CI gate while preserving the sub-one-hour whole-matrix performance target.

Remote endpoint and distributed-runner composition are no longer remaining
design work. The two-/three-tier real-weight matrix now certifies correct live
production migration. Until the remaining steps are green, it does not
demonstrate that convergence delivers the required observed inference speedup
or certify the later MTP/prefix/target-model matrix.
