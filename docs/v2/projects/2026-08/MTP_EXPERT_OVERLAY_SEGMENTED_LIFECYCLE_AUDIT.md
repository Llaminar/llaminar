# MTP ExpertOverlay segmented lifecycle audit

Date: 2026-08-25

Scope: the Qwen3.5 122B `CUDA1 + CPU1` ExpertOverlay Depth-15 production
parity failure reached a successful 97-unit grouped-verifier forward, then
rejected it as non-monolithic while pairing the verifier with its preparation
graph. The subsequent error path had already consumed
`DeviceGenerationStateReady`, so request reset could not join the controller
frontier.

This is a dated investigation record. The source-owned contracts are
`DeviceGraphOrchestrator`, `ForwardExecutionEngine`,
`HostedDeviceGenerationLifecycle`, and `DeviceExecutionTimeline`.

## Lifecycle before the correction

```mermaid
flowchart TD
    A[Admit device generation] --> B[Publish DeviceGenerationStateReady]
    B --> C[Capture verifier preparation]
    C --> D[Consume ready event and clear valid bool]
    D --> E[Run grouped verifier production graph]
    E --> F{Pairing chooses replay representation}
    F -->|Coordinator present or ROCm| G[Retained decode plan]
    F -->|Otherwise| H[Monolithic device-loop template]
    H --> I{Graph has one replay unit?}
    I -->|No: local CUDA plus CPU| J[Return failure]
    J --> K[Request reset sees active request]
    K --> L{ready.valid?}
    L -->|No: consumed already| M[Teardown abort]
    G --> N[Assemble hosted transaction]
    I -->|Yes| O[Assemble native CUDA parent]
```

Two unrelated facts had been collapsed:

1. The presence of a rank transaction coordinator was being used as a proxy
   for heterogeneous execution. A process-local CUDA/CPU overlay has no remote
   coordinator but still has an explicit sparse boundary and segmented replay.
2. `ready.valid == false` represented both “no request owns this state” and
   “one exact stream borrowed it and has not republished yet.” The latter is a
   live frontier that reset must join.

## Simplified authoritative lifecycle

`deviceGenerationExecutionPolicy()` is the sole outer-loop execution-policy
authority. It consumes the same typed topology proof as decode capture:

```mermaid
flowchart LR
    T[Declared execution topology] --> P{Mixed device types or remote sparse participant?}
    P -->|Yes| H[HostScheduledCapturedTransactions]
    P -->|No| C{Native conditional graph complete?}
    C -->|Yes| N[NativeConditionalGraph]
    C -->|No and ROCm| H
    C -->|No otherwise| U[Unsupported: fatal]
```

This is not a fallback. The policy is chosen before parent materialization and
both verifier pairing and branch assembly consume it. Hosted execution retains
the exact production graph, including every declared sparse boundary; the host
submits only the device-authenticated transaction named by the ticket.

Controller ordering now has one explicit state machine:

```mermaid
stateDiagram-v2
    [*] --> Inactive
    Inactive --> Published: admission event recorded
    Published --> Borrowed: exact consumer waits on event
    Borrowed --> Published: committed or terminal event recorded
    Published --> Inactive: ordered request reset
    Borrowed --> Inactive: record event after borrower, then reset waits
    Borrowed --> Inactive: terminal D2H completion event observed
```

The backend callback must succeed before a transition commits. A reset of a
borrowed frontier re-records the preallocated controller event after the exact
borrower stream and enqueues a wait on the reset stream. It does not synchronize
a stream/device, claim that partial controller contents are committed, or replay
work.

## Remaining proof sequence

1. Device-free adversarial transition tests for publish, borrow, republish,
   backend-edge failure, published reset, and borrowed reset.
2. Source-policy proof that pairing and branch assembly consume the same typed
   topology policy and that reset contains no blocking synchronization.
3. Exact real-weight `CUDA1 + CPU1`, Dynamic, Ordinal, FP16 activation/KV,
   Depth-15 production parity cell.
4. Affected CUDA/ROCm ExpertOverlay matrix, then the complete canonical parity
   campaign.

## Follow-up: sparse request identity audit

The complete production campaign subsequently isolated a second lifecycle
proxy in hosted HIP MTP replay.  `resolveMTPMoEOverlayCollectiveRuntimeParams()`
read only the ExpertOverlay root-generation field.  Dense Qwen3.6 and ordinary
single-device MoE do not install that field, so dynamic-depth replay rejected a
valid local request after all numerical checkpoints had passed.

```mermaid
flowchart LR
    A[Retained MTP branch] --> B{Old: overlay generation nonzero?}
    B -->|yes| C[Stamp runtime stages]
    B -->|no| D[Reject hosted replay]
    E[Dense local request] --> D
    F[Single-device MoE request] --> D
```

The simplified flow selects one typed authority before capture or replay.  A
backend, coordinator pointer, and presence of sparse stages are no longer
authority proxies.

```mermaid
flowchart LR
    A[Retained graph transaction] --> P{Declared ExpertOverlay authority?}
    P -->|yes| R[ExpertOverlayRoot generation]
    P -->|no| L[LocalRequestState session]
    R --> V{Nonzero and admitted?}
    L --> V
    V -->|yes| C[One runtime identity for capture and replay]
    V -->|no| F[Fail before graph submission]
```

`MoESparseRequestIdentity` makes zero invalid at construction and preserves the
authority kind even when both sources happen to carry the same integer.  The
orchestrator owns the sole selection method.  Distributed graphs still verify
that the selected root generation equals the transaction coordinator binding;
local graphs never populate or impersonate that distributed field.

## Follow-up: LocalTP capture-wave collective cycle

The `ROCm2 + CPU2` static production cell exposed a third defect during the
first real prefill transaction.  Both ROCm participants captured valid native
segments, but the authority participant placed the next rooted shared-expert
reduction in its pre-ticket unit while its sibling placed the same collective
in the post-ticket unit.  This was a graph dependency error, not a GEMM or
ticket-polling defect.

The resulting wait-for graph was cyclic:

```mermaid
flowchart TD
    A[Authority launches pre-ticket captured unit] --> B[Unmatched rooted RCCL reduction becomes runnable]
    B --> C[RCCL waits for sibling and occupies ROCm compute units]
    C --> D[Earlier mapped ticket publication cannot make progress]
    D --> E[Authority manual CPU transaction waits for ticket]
    E --> F[Authority cannot enter next capture wave]
    G[Sibling finishes its pre-ticket unit] --> H[Sibling requests next capture wave]
    H --> I[Wave rendezvous waits for authority]
    I --> J[Sibling cannot launch matching rooted reduction]
    J --> B
```

Textual stage order did not make this safe.  The rooted collective and mapped
publication are separate GPU-stream branches inside one native graph, so a
later collective can run before an independent copy.  Correctness therefore
has to follow typed transaction boundaries and collective ordinals rather than
assuming insertion order.

The simplified per-layer lifecycle is now:

```mermaid
stateDiagram-v2
    [*] --> PreTicketCapturedUnit
    PreTicketCapturedUnit --> TicketPublished: router + shared partial + local routed experts
    TicketPublished --> HostSparseTransaction: authenticated fixed-capacity ticket
    HostSparseTransaction --> PostTicketCapturedUnit: canonical return publication
    PostTicketCapturedUnit --> RoutedFoldComplete: ordered canonical route fold
    RoutedFoldComplete --> SharedReductionComplete: matching rooted collective on every LocalTP participant
    SharedReductionComplete --> CombinedPublished: shared gate + one combined broadcast
    CombinedPublished --> PreTicketCapturedUnit: next layer
```

`GraphHeterogeneousTicketUnitContract` remains the sole cutpoint authority.  A
rooted shared reduction now depends on the post-ticket routed-unit terminal;
the reverse dependency is forbidden.  This one edge gives all participants the
same collective sequence in the same capture wave without a stream
synchronization, timeout extension, eager replay, or topology-specific branch.

The focused graph-lowering proof covers both mapped-sparse and native-collective
LocalTP transport.  The exact real-weight `ROCm2 + CPU2`, static, ordinal,
FP16-activation/KV production cell then passed graph-captured prefill, every
CSV checkpoint comparison, incremental decode, and automatic prefix restore.

## Follow-up: reusable ModelContext physical seal

The next `ROCm1 + CPU2` Dynamic cell restored its initial logical owner map and
successfully rebound the model registry, but the following MTP cell rejected
one GPU resident.  Its execution payload matched the source format while its
allocation still included the generic shadow arena's optional `mins` and
`emins` regions.  Logical restoration and physical canonicalization had been
incorrectly treated as the same lifecycle edge.

The old teardown path was therefore incomplete:

```mermaid
flowchart LR
    A[Drain inference tickets and maintenance] --> B[Restore initial owner map]
    B --> C[Acquire published participant banks]
    C --> D[Rebind model registry]
    D --> E[Destroy physical fabric]
    C -. shadow-slot engine alias .-> F[Old shadow arena remains pinned]
    F --> G[Next runner charges loader-format allocation]
    G --> H[Reuse adoption fails or memory drifts]
```

A closed migration cycle cannot write directly into a departing expert's
loader allocation: the old epoch may still read that slot.  Staging through a
shadow slot is correct.  Once the terminal ticket and maintenance barriers have
drained, however, the newly free loader allocation must become authoritative
again before the model context is reusable.

The simplified lifecycle has one typed terminal transition and no second
logical publication:

```mermaid
stateDiagram-v2
    [*] --> Open
    Open --> Sealing: terminal context seal owns zero-ticket fabric
    Sealing --> Failed: owner map, capacity, lane, or event invariant fails
    Sealing --> Sealed: every resident aliases an adopted loader allocation
    Sealed --> Sealed: return immutable cached seal
    Sealed --> [*]: atomically rebind registry, then destroy fabric
    Failed --> [*]: invalidate reusable context
```

```mermaid
flowchart TD
    A[Restore initial prepared owner map] --> B[Preflight every local bank]
    B --> C{Resident already owns adopted slot?}
    C -->|yes| D[Retain exact triplet]
    C -->|no| E[Reserve free adopted slot]
    E --> F{CPU or GPU endpoint?}
    F -->|CPU| G[Background NUMA copy worker]
    F -->|GPU| H[Same-device transfer stream and completion event]
    G --> I[Canonical immutable seal bank]
    H --> I
    D --> I
    I --> J[Atomic model-registry replacement]
    J --> K[Release published banks and shadow arenas]
```

The fabric materializes same-endpoint lanes during setup even for a single
logical participant.  Sealing polls worker state or exact GPU completion events
and never uses stream/device synchronization.  After `Sealing` begins, ordinary
physical movement is structurally rejected.  A device-free regression rotates
one shadow slot through three closed cycles, leaves exactly one expert in the
shadow arena, and proves byte-exact canonicalization for all 20 quantized
codebooks plus FP16, BF16, and FP32.

## Follow-up: prefix/MTP proof crossed a movement fingerprint

The first reused `ROCm1 + CPU2` Depth-1 cell initialized successfully after the
physical seal, then failed three apparently separate assertions: the serial MTP
oracle observed a prefix miss, the partial restore's cached KV bytes differed,
and no typed Depth-1 boundary was published.  All three came from one stale
proof edge.  Decode parity captured its step-zero oracle at movement epoch 2,
then ordinary dynamic maintenance published epoch 3.  The cell deliberately
uses `InvalidateOnRebalance` while measuring convergence, so epoch 3 correctly
has a new cache fingerprint.

The old proof composition silently crossed those epochs:

```mermaid
flowchart TD
    A[Fresh prefix seed at movement epoch 2] --> B[Complete restore at epoch 2]
    B --> C[Decode step-zero oracle at epoch 2]
    C --> D[Dynamic maintenance publishes epoch 3]
    D --> E[Serial MTP oracle assumes epoch-2 entry is still current]
    E --> F[Correct cache miss and epoch-3 recomputation]
    F --> G[Partial proof compares epoch-3 KV with epoch-2 oracle]
    G --> H[KV mismatch and no declared-depth boundary]
```

The simplified proof owns one cache fingerprint from admission through
comparison.  It admits at most one seed request; the immediately following
request must be a complete hit at the same movement epoch.  A second miss,
cache bypass, epoch change, or missing MTP payload is fatal.

```mermaid
stateDiagram-v2
    [*] --> AdmitCurrentFingerprint
    AdmitCurrentFingerprint --> CompleteHit: entry already exists
    AdmitCurrentFingerprint --> Seeded: miss or partial hit computes full prompt
    Seeded --> CompleteHit: exactly one complete restore at same movement epoch
    Seeded --> Failed: second miss, bypass, incomplete payload, or epoch change
    CompleteHit --> SerialMTPProof: MTP payload requested
    CompleteHit --> DecodeOracle: partial-prefix proof requested
    DecodeOracle --> PartialRestore: append one authenticated token
    PartialRestore --> Compared: cached prefix exact, suffix numerically certified
    SerialMTPProof --> Compared: declared-depth graph transaction certified
    Failed --> [*]
    Compared --> [*]
```

The canonical prefill parity phase remains the graph-capture proof.  Cache-only
admissions do not require compute snapshots because a complete hit launches no
model graph by construction; the following decode or MTP transaction restores
the ordinary production snapshot policy.  This removes cross-phase state reuse
without changing production cache policy, numerical thresholds, movement
cadence, or execution mode.

## Follow-up: Dynamic economy evidence multiplied by MTP depth

The first post-seal CPU/ROCm process campaign reached only nine of 96 cells in
12 minutes.  Every Dynamic MTP policy was independently repeating the same
256-token initial/converged timing cohorts.  The MTP axis must still prove its
own captured numerical path and real movement, but logical draft depth does not
create a new hypothesis about whether an adversarial expert placement becomes
faster after one economical residency wave.

The old product accidentally made an evidence role look like an execution
axis:

```mermaid
flowchart LR
    A[Topology + precision + owner order] --> B[Dynamic MTP off]
    A --> C[Dynamic depth 1]
    A --> D[Dynamic depth 2]
    A --> E[Dynamic depth 3]
    A --> F[Dynamic depth 15]
    A --> G[Dynamic adaptive depth]
    B --> H[Certify + initial timing + move + converged timing]
    C --> I[Certify + initial timing + move + converged timing]
    D --> J[Certify + initial timing + move + converged timing]
    E --> K[Certify + initial timing + move + converged timing]
    F --> L[Certify + initial timing + move + converged timing]
    G --> M[Certify + initial timing + move + converged timing]
```

`ModelParityDynamicEvidence` makes the two proof obligations explicit. Every
Dynamic cell owns `EconomicMovement`: it runs the real production service
certification, publishes physical expert bytes, settles at a typed between-wave
boundary, proves its exact graph/MTP/prefix path, and emits the normal CSV
evidence. The first expander version incorrectly multiplied
`EconomicMovementAndObservedSpeedup` over every CPU-bearing topology,
activation/KV precision pair, and owner order. That was redundant economy work,
not additional mathematical coverage.

`ModelParityFeatureMatrix::dynamic_speedup_witness` is now the typed selection
authority. A definition selects ordinal, random, both owner orders, or disabled;
the expander assigns the matched timing cohort only to the first activation/KV
pair, MTP-off policy, and first declared prefill schedule. The representative
set covers CUDA/CPU, ROCm/CPU, and CUDA/ROCm/CPU transport economies through the
122B CUDA2+CPU2 random cell, 122B ROCm4+CPU2 random cell, and 35B
CUDA1+ROCm1+CPU2 random cell respectively. Every other Dynamic cell still moves
and proves real expert payload bytes. No fixture infers this role from a test
name.

```mermaid
stateDiagram-v2
    [*] --> EconomyCertified: measured production service profile active
    EconomyCertified --> InitialCohortMeasured: observed-speedup witness only
    EconomyCertified --> MovementTargetSatisfied: ordinary Dynamic cell
    InitialCohortMeasured --> MovementTargetSatisfied: profitable bytes published
    MovementTargetSatisfied --> MovementBoundarySettled: no admitted wave remains
    MovementBoundarySettled --> ConvergedCohortMeasured: witness only
    MovementBoundarySettled --> GraphParity: ordinary Dynamic cell
    ConvergedCohortMeasured --> GraphParity: positive prefill and decode gates
    GraphParity --> PrefixRestore: fresh, complete, and partial proof
    PrefixRestore --> [*]
```

The first real `ROCm1 + CPU2`, Dynamic, ordinal, FP16 activation/KV, MTP-depth-2
non-witness passed on both ranks in 135.6 seconds from a cold process.  It
published 144 movement rows, retained all three prefix-restore rows, certified
full captured prefill/decode/MTP execution, and intentionally emitted no
convergence-timing artifact.  The previously identical depth cell took roughly
four minutes when it also repeated the matched timing proof.  This is an
evidence-lifecycle reduction only; production movement, graph capture,
numerical thresholds, and output artifacts remain mandatory.

## Follow-up: phase-collapsed movement economy

The retained `ROCm1 + CPU2` speedup witness then isolated a policy defect rather
than another execution-lifecycle defect.  A longer matched sample made the
signal unambiguous: the accepted placement improved decode by 9.83% while
regressing prefill by 6.75%.  The controller's projected net benefit remained
strongly positive because it had summed decode, prefill, and grouped-verifier
critical paths before asking whether service improved.

That scalar was an invalid state: a large decode win could cross-subsidize a
prefill loss even though the production campaign independently gates both
phases.  The same collapse existed in the heterogeneous host authority, the
deterministic CPU oracle for device policy, and the shared CUDA/HIP controller
kernel.

```mermaid
flowchart LR
    D0[Decode critical path before/after] --> S[Sum every phase]
    P0[Prefill critical path before/after] --> S
    G0[Grouped-verifier critical path before/after] --> S
    S --> R{Aggregate reduction positive?}
    R -->|Yes| C{Projected gain exceeds transfer and interference?}
    C -->|Yes| A[Admit movement]
    P0 -. Regression hidden by decode gain .-> A
```

The simplified economy lifecycle preserves phase identity until the independent
production objectives have been checked.  There is still one movement payoff
calculation and one publication authority; this change does not introduce a
second controller, calibration pass, or phase-specific placement map.

```mermaid
stateDiagram-v2
    [*] --> PhasePureMakespans: score exact owner map per layer and participant
    PhasePureMakespans --> Rejected: any measured phase becomes slower
    PhasePureMakespans --> AggregatePayoff: every phase improves or is unchanged
    AggregatePayoff --> Rejected: aggregate reduction is not strict
    AggregatePayoff --> Rejected: gain does not repay movement plus interference
    AggregatePayoff --> Admitted: positive net benefit clears policy floor
    Admitted --> Prepared: asynchronous parallel transfer and repack
    Prepared --> Certified: destination payload and event complete
    Certified --> Published: one atomic owner-map epoch
    Rejected --> [*]
    Published --> [*]
```

Equality for an individual phase is deliberate.  It permits a cycle that helps
decode without changing prefill, or vice versa, while forbidding cross-phase
harm.  The aggregate objective must still decrease strictly and repay measured
transfer/repack and inference-interference costs.  Host, CUDA, and HIP consume
the same no-regression rule; focused regressions construct the formerly accepted
decode-win/prefill-loss case and a positive control in which both phases favor
the move.

## Follow-up: invalidation changed identity but retained stale capacity

The phase-pure controller admitted a placement whose captured forward work and
CPU expert service both decreased, yet the exact production witness reported a
2.4x prefill slowdown: 865.1 ms became 2075.6 ms while decode improved 5.78%.
Request-matched `forward_graph` timers ruled out the graph. The missing latency
was in prefix harvest after the forward returned.

The cell uses one-token prefix blocks to prove exact partial restore. Its
terminal hybrid block is roughly 157 MB and its RAM tier is 4 GiB.
`InvalidateOnRebalance` correctly incorporated the movement epoch into the
request fingerprint, but publication only replaced `prefix_fingerprint_`.
RAM/device-hot records from every old epoch remained capacity-resident. Once
ordinary convergence traffic filled RAM, each new harvest synchronously
checksummed, appended, and fsynced a stale 157 MB block into the disk tier.

```mermaid
flowchart TD
    A[Movement authority publishes epoch N+1] --> B[Build new prefix fingerprint]
    B --> C[Replace active lookup fingerprint only]
    C --> D[Epoch-N RAM and device-hot entries remain charged]
    D --> E[New-epoch prefix harvest needs RAM]
    E --> F[LRU demotes stale entry]
    F --> G[Wait payload, checksum, write, fsync disk archive]
    G --> H[Prefill returns after unrelated storage maintenance]
```

The corrected lifecycle has one post-initialization fingerprint publisher.
For `InvalidateOnRebalance`, it first performs a typed cache rebase. The rebase
preflights legacy explicit leases, drops stale RAM and device-hot cache owners,
and removes stale disk records from the current runtime index. Shared handles
already held by an admitted request remain valid. Durable records stay in the
append-only archive and are reclaimed by its normal bounded LRU, so the
request-boundary transition performs no archive write, delete, compaction, or
fsync.

```mermaid
stateDiagram-v2
    [*] --> FingerprintStable
    FingerprintStable --> CandidateBuilt: movement epoch or live payload shape changes
    CandidateBuilt --> Unchanged: candidate equals active fingerprint
    CandidateBuilt --> RebasePreflight: InvalidateOnRebalance and identity changed
    CandidateBuilt --> Published: compatible retention policy and identity changed
    RebasePreflight --> Failed: zero identity or retained explicit lease
    RebasePreflight --> VolatileRetired: stale RAM/device-hot owners dropped; stale disk keys unindexed
    VolatileRetired --> Published: active fingerprint replaced atomically
    Unchanged --> FingerprintStable
    Published --> FingerprintStable
    Failed --> [*]
```

Two device-free regressions lock down the transition. One proves that RAM
capacity is recovered without changing durable archive bytes or adding another
RAM-to-disk demotion; the other proves a busy explicit lease rejects the whole
transition before any tier is mutated. A source-policy regression requires live
hybrid layout refresh to use the same publisher, eliminating the former direct
assignment exception.

## Follow-up: convergence workload identity and demand-window admission

After volatile prefix rebasing removed the storage-maintenance latency, the
same production witness improved decode by 2.01% but prefill by only 0.58%.
Captured-forward timers were flat and the prefix archive remained an empty
80-byte header. The remaining weakness was in the test lifecycle: an 18-row
prefill was split into a 16-row captured segment and a noisy two-row tail, while
movement training repeatedly reused the eight measurement cache identities.
After the first pass, most training windows therefore represented prefix-hit
decode traffic instead of the prefill/decode mix judged by the final gate.

The identity reuse also created an avoidable race model. Because a movement
publication could retire between observation and prefix admission, the fixture
kept a ninth baseline candidate, remembered an optional overlapping identity,
and omitted that identity from both cohorts.

```mermaid
flowchart TD
    A[Measure IDs 0 through 8] --> B[Retain one publication-overlap reserve]
    B --> C[Reuse IDs 0 through 7 as movement traffic]
    C --> D[Prefix hits make later histograms decode-heavy]
    D --> E{Publication observed before or after request?}
    E --> F[Remember optional overlapping ID]
    F --> G[Omit that ID from both cohorts]
    G --> H[Compare six short segmented prefills]
```

The first simplification removed the ninth candidate and made every observed
prefill 64 rows, exactly divisible into four retained 16-row segments on this
memory-constrained 122B topology. It also assigned every training request a new
identity outside measurement IDs 0 through 7. That fixed the prefill witness,
but the identity is a real first token, not merely a cache salt. The nominally
stationary training workload therefore optimized different autoregressive
decode trajectories. One exact run improved prefill by 9.35% but decode by only
1.02%, and the request-paired samples showed that the prompts whose routes were
not represented by the moving training interval had regressed.

The final lifecycle distinguishes two traffic purposes instead of making cache
identity carry both meanings. Normal optimization cycles the exact eight A/B
prompt identities, including genuine production prefix hits. Every accepted
placement publication changes the `InvalidateOnRebalance` fingerprint, so each
new epoch still sees a full-compute copy of those prompts before reuse begins.
Only after the movement target is satisfied may a partially occupied demand
bank receive cache-distinct rotation traffic; those IDs can never seed a hit in
the subsequent measured cohort.

The 640-row histogram window admits the complete timing cohort because
`8 * (64 + 5) = 552`. Quiescence alone is insufficient: a partial active bank
can have fewer than 552 rows remaining and publish a new epoch during timing.
The host authority now exposes a typed `MoEOptimizationDemandWindow` containing
the RCU bank generation, exact collected row count, and capacity. The snapshot
pins the active bank and rechecks its complete monotonic epoch after reading, so
it cannot return an ABA-stale bank during rotation. Timing admission requires
both reconciled quiescence and strictly more than 552 rows of headroom; equality
is rejected because the cohort's last row would complete the window.

```mermaid
stateDiagram-v2
    [*] --> EconomyCertified
    EconomyCertified --> InitialCohortMeasured: IDs 0..7; one immutable epoch
    InitialCohortMeasured --> MovementTraining: cycle exact IDs 0..7
    MovementTraining --> MovementTargetSatisfied: required waves and axes complete
    MovementTargetSatisfied --> ReconcileProgress: stop admitting requests
    ReconcileProgress --> MovementTraining: publication or physical work remains
    ReconcileProgress --> InspectDemandHeadroom: typed authority is quiescent
    InspectDemandHeadroom --> MovementBoundarySettled: remaining rows > 552
    InspectDemandHeadroom --> DemandWindowRotation: insufficient headroom
    DemandWindowRotation --> ReconcileProgress: ordinary disjoint IDs 8+n rotate bank
    MovementBoundarySettled --> ConvergedCohortMeasured: replay exact IDs 0..7
    ConvergedCohortMeasured --> GraphParity: strict prefill and decode economy gates
    GraphParity --> [*]
```

No measurement request can overlap movement because movement does not begin
until the initial cohort completes, and the converged cohort does not begin
until the typed reconciled-and-headroom boundary. Prefix rebasing makes the
final replay a full-compute miss under the new placement, while the disjoint
rotation mode prevents a later no-movement bank rotation from repopulating its
keys. The reserve count, optional overlap identity, omission branch, short-tail
timing geometry, and guessed fixed-wave headroom are removed. The production
controller, asynchronous publication, cache policy, and strict 2% economy
thresholds are unchanged. The focused real-weight ROCm1+CPU2 run settled after
four publications at placement epoch 5, improved prefill by 12.35% and decode
by 5.04%, passed both MPI ranks, and left the prefix archive at its empty 80-byte
header.

### Movement-only cells must not reserve the timing cohort

The first canonical wrapper run exposed one more leaked lifecycle assumption.
The demand-headroom check above was applied to every Dynamic matrix cell, even
though only the centrally assigned MTP-off speed witness executes the matched
A/B cohort. Movement-only MTP cells use the normal short production histogram
window. Asking those cells to reserve 552 unused rows was therefore impossible:
each quiescent observation requested another bank rotation, ordinary inference
closed that small bank, and maintenance kept publishing new epochs instead of
crossing into graph parity.

Settlement now owns a typed `ConvergenceBoundaryPurpose`. A
`MovementProof` ends at the first quiescent topology-valid publication;
`ObservedSpeedupCohort` additionally requires the authoritative demand window
to admit all 552 routed rows. This keeps bank geometry out of cells that do not
consume it while preserving the stronger no-publication interval around the
speed witness.

```mermaid
stateDiagram-v2
    MovementTargetSatisfied --> ReconcileProgress
    ReconcileProgress --> MovementBoundarySettled: MovementProof and quiescent
    ReconcileProgress --> InspectDemandHeadroom: ObservedSpeedupCohort and quiescent
    InspectDemandHeadroom --> MovementBoundarySettled: remaining rows > 552
    InspectDemandHeadroom --> DemandWindowRotation: insufficient headroom
    DemandWindowRotation --> ReconcileProgress
    MovementBoundarySettled --> GraphParity: movement-only cell
    MovementBoundarySettled --> ConvergedCohortMeasured: speed witness
```

The focused real-weight ROCm1+CPU2 Dynamic/Ordinal/MTP-depth-1 cell then passed
both MPI ranks in 122 seconds, including physical movement, captured prefill,
prefix restore, decode, MTP/Hugging Face checkpoints, promoted-expert execution,
and teardown. The pre-fix wrapper instance had remained in repeated rotations
for more than 18 minutes without producing its first parity artifact.

## Follow-up: `Waiting` raced an unreconciled inference notification

The first 64-row run reached the required durable movement axes, then exposed
one remaining publication race. The continuation observed three committed
waves and a `CollectingDemand` activity left by the worker's previous empty
poll. Its final training request filled generation four and woke maintenance,
but the status reader ran before the worker consumed that wake. It therefore
treated the stale activity as quiescent. Wave four became visible inside the
first converged prefill (`expected=3`, `before=3`, `after=4`), and the typed
epoch guard correctly rejected the cohort.

```mermaid
sequenceDiagram
    participant I as Inference
    participant S as Public status
    participant W as Maintenance worker
    I->>W: Complete request and set wake boolean
    S->>S: Read stale CollectingDemand
    S-->>I: quiescent=true
    I->>I: Begin converged prefill at epoch 3
    W->>W: Drain newly full histogram
    W->>I: Publish epoch 4
    I->>I: Reject epoch crossing
```

A boolean wake cannot identify which progress a worker has actually
reconciled. The host authority now publishes a monotonic progress generation
before every inference/event wake. Immediately before the authoritative
histogram poll, the worker captures that generation. It may publish
`CollectingDemand` only together with the captured generation after the poll
reports no complete window. Public quiescence requires equality between the
published and reconciled generations as well as an idle activity state.

```mermaid
stateDiagram-v2
    [*] --> Reconciled: published_generation == reconciled_generation
    Reconciled --> ProgressPublished: inference/event increments published_generation
    ProgressPublished --> ReconcilingDemand: worker observes wake
    ReconcilingDemand --> MovingWeights: complete window exists
    ReconcilingDemand --> Reconciled: no complete window; publish reconciled_generation
    MovingWeights --> PublishingResidency: prepared payload events complete
    PublishingResidency --> ReconcilingDemand: new epoch is selectable
    Reconciled --> TimingEligible: activity is CollectingDemand and generations match
    ProgressPublished --> TimingRejected: generations differ even if activity is stale
```

The generation is host-owned lifecycle state, not PerfStats and not a shadow of
device routing data. A distributed follower retains a normalized zero/zero pair
because its preposted proposal mailbox owns no policy decision; the arbitrary
coordinator alone reconciles topology demand. This strengthens the production
status surface used by diagnostics and tests without adding a synchronization,
host-side inference wait, or test-only controller call.

## Follow-up: logical graph identities underpriced native executables

The reused `ROCm1 + CPU2` process campaign passed MTP-off and Depth-1, then
failed while constructing the Depth-2 runner with only 36 MiB free. The next
4 MiB physical-fabric allocation was merely where exhaustion became visible.
The capacity plan had reserved driver storage for logical graph-cache
identities, while a host-authority ExpertOverlay graph is lowered into one
native HIP executable before every authenticated CPU boundary plus one terminal
unit. Retained MTP controller fragments add a second, independent executable
family.

The old lifecycle both priced the wrong object and ordered topology too late:

```mermaid
flowchart TD
    A[Declarative ExpertOverlay plan] --> B[Capacity admission]
    B --> C[Count logical graph-cache identities]
    C --> D[Fill GPU expert quota from apparent remainder]
    D --> E[Freeze host or device authority]
    E --> F[Lower each logical graph into native units]
    F --> G[Instantiate retained MTP fragments]
    G --> H[Driver pool exhausts during reused runner setup]
```

For the 48-layer host-authority model, every complete logical model graph owns
49 native executables. Dynamic MTP depth `D` additionally retains the exact
semantic fragment upper bound `D * (D + 10)`. Debug driver evidence at depth 15
observed hundreds of physical HIP instantiations and roughly 928 MiB of positive
driver high-water growth, while the former logical estimate reserved only about
185 MiB. Increasing a generic safety margin would not make those two meanings
coherent.

The simplified setup lifecycle freezes one topology authority first and prices
one typed physical inventory before any expert quota is resolved:

```mermaid
stateDiagram-v2
    [*] --> DeclarativeTopology
    DeclarativeTopology --> AuthorityFrozen: normalize explicit or synthesized plan
    AuthorityFrozen --> InventoryResolved: model identities x native segments plus auxiliary executables
    InventoryResolved --> CapacityAdmitted: fixed graph bytes charged on each GPU
    CapacityAdmitted --> PlacementFrozen: fill tier quotas from the true remainder
    PlacementFrozen --> GraphsMaterialized: exact retained executable family
    GraphsMaterialized --> InferenceReady: driver and physical-fabric admission complete
    AuthorityFrozen --> Failed: declared authority conflicts with topology
    InventoryResolved --> Failed: invalid geometry or arithmetic overflow
    CapacityAdmitted --> Failed: complete fallback coverage cannot fit
```

`CapturedGraphExecutableInventory` now distinguishes model graph identities,
native segments per model graph, and auxiliary native executables. Host-owned
heterogeneous overlays resolve `layers + 1` segments; all-GPU device-owned
execution remains one complete native executable per logical graph. The same
`resolveMTPRetainedDeviceGenerationFragmentCapacity()` function feeds both
memory admission and `DeviceGraphOrchestrator`, so requested depth cannot be
priced with one formula and materialized with another.

Authority normalization was already the intended one-way transition, but the
runner installed its result only for synthesized single-tier plans. Explicit
plans therefore reached capacity with `Unresolved` authority even though the
normalizer had produced the correct immutable clone. The normalized result is
now installed for both dispositions before capacity admission; later
model-aware placement freezing consumes it instead of reconstructing topology.
Focused capacity, memory-planner, and orchestration-plan unit gates cover the
typed inventory and authority transition. The reused real-weight MTP sequence
remains the production proof of driver-pool sufficiency.

## Follow-up: participant search breadth was confused with admission capacity

After native-executable admission was corrected, the reused `ROCm1 + CPU2`
Dynamic sequence reached live inference but Depth-1 published tier movements
only. The histogram and planner both reported participant skew, yet no
participant-placement edge survived economy admission. The controller had
applied the two-entry participant transfer budget while *generating*
candidates, so only the raw-count swap from layer zero existed. Phase-weighted
service accounting correctly assigned that swap zero projected gain; useful
independent swaps in later layers were never visible to the economy authority.

The test driver amplified the defect. Movement-only cells repeatedly used a
small set of prefix identities and sized their horizon as though each request
would execute five routed rows. Once those prefixes were cached, a request
executed only its decode continuation, so the nominal 520-request horizon made
little histogram progress while spending minutes proving cache hits.

```mermaid
flowchart TD
    H[Complete typed histogram] --> G[Generate at most two participant entries]
    G --> L[Only layer-zero swap exists]
    L --> E{Phase-weighted economy positive?}
    E -->|No| T[Tier-only publication]
    T --> P[Reuse cached movement prompt]
    P --> D[One decode row instead of full prefill]
    D --> P
```

Search breadth and physical admission are now separate typed stages. Candidate
discovery exposes one self-contained closed swap per layer, avoiding dependent
same-layer swap chains while allowing economy to inspect the entire model.
The selector may skip a phase-regressive or over-budget candidate and continue
to later layers. Only after selection does it enforce the two participant
objective entries and the configured concurrent physical-cycle capacity.

```mermaid
flowchart TD
    H[Complete typed histogram] --> S[One independent participant swap per layer]
    S --> E[Score exact phase and participant service makespan]
    E --> O[Order economically eligible closed cycles]
    O --> A{Fits participant-objective and physical-cycle budgets?}
    A -->|No| R[Record typed rejection and inspect next candidate]
    A -->|Yes| B[Build bounded authoritative transaction]
    B --> C[Recompose and certify physical closed cycles]
    C --> P[Publish immutable placement epoch]
```

Admission evidence is a conservation ledger over disjoint outcomes:
individual-policy rejection, dependent-payoff rejection,
participant-axis-budget rejection, physical-capacity rejection, or admitted
candidate. `policy_bounded` includes the first three; `capacity_bounded`
includes only physical capacity. The production parity gate parses every
category and requires the totals and flags to agree, so adding a rejection
reason without extending the conservation equation fails closed.

Movement-only traffic now uses cache-distinct, reference-shaped 64-row
prefills. Ten such requests are sufficient to close a 640-row histogram, with
eight additional requests reserved only for asynchronous publication overlap.
The observed-speedup witness retains its fixed paired corpus and stronger
demand-headroom lifecycle; the two proof purposes no longer share a guessed
request multiplier. A focused authority regression makes layer zero
phase-regressive and proves that a profitable layer-one participant swap is
selected without exceeding the two-entry admission budget.

## Follow-up: native executables outlived the arena they borrowed

With physical executable admission fixed, the retained Depth-1 runner passed
and released its public buffers, but Depth-2 still exhausted ROCm memory while
allocating a final 4 MiB physical-fabric block. The explicit allocation ledger
showed roughly 7.8 GiB free and the HIP graph ledger showed that all 562 native
executables were eventually destroyed. Neither observation described the
ordering edge that controlled physical reclamation.

Line-by-line VRAM accounting identified one 4,369,981,188-byte activation and
workspace arena. `BufferArena` returned that allocation to the backend while
all 562 HIP graph executables still embedded addresses inside it. The backend
removed the pointer from Llaminar's allocation registry, but HIP retained the
physical pages until the executable family was destroyed. The following runner
therefore allocated a second complete arena against the same device. A failed
runner whose capture had not begun returned the identical arena immediately,
which rules out capacity estimation and generic allocator caching as the
cause.

The old teardown order made member destruction, rather than the lifecycle API,
the accidental owner of graph topology:

```mermaid
flowchart TD
    A[Retire selected publication events] --> B[Release BufferArena]
    B --> C[Backend registry forgets 4.07 GiB allocation]
    C --> D[Live native executables still borrow arena addresses]
    D --> E[HIP defers physical reclamation]
    E --> F[Destructor eventually destroys child graphs]
    F --> G[Next runner has already admitted and allocated a second arena]
    G --> H[Depth-2 setup OOM]
```

There is now one destructive topology transition shared by object teardown,
public buffer release, and explicit cache invalidation. It is idempotent and
owns the complete graph inventory. The MTP generation parent is destroyed
before the children whose capture identities it embeds; forward, verifier,
publication, stochastic, overlay-epoch, and maintenance graphs are destroyed
before their exact device contexts; the arena remains live until all borrowers
are gone.

```mermaid
stateDiagram-v2
    [*] --> Captured: arena addresses embedded in native executables
    Captured --> EventsRetired: wait exact producer events
    EventsRetired --> ParentDestroyed: release MTP generation parent and stream
    ParentDestroyed --> ChildrenDestroyed: release every forward, MTP, publication, stochastic, and overlay graph
    ChildrenDestroyed --> ContextsDestroyed: release graph streams and device contexts
    ContextsDestroyed --> ArenaReleased: return activation/workspace allocation
    ArenaReleased --> [*]
```

No device or stream synchronization is introduced. Event retirement preserves
the producer/consumer DAG, and graph destruction happens only at a destructive
runner boundary where no new inference can be admitted. A source-policy
regression inventories every retained graph owner and requires parent-before-
child, graph-before-context, and topology-before-arena ordering. The formerly
failing reused real-weight Depth-1 to Depth-2 sequence is the required physical
reclamation proof.

## Follow-up: retained MTP capacity was mistaken for an active transaction

The first cell in a reusable production-matrix process deliberately executes
with MTP off while retaining capacity for the largest later matrix member. The
memory admission path charged that retained depth, but several graph-resource
and participant declarations were still guarded by `mtp.enabled`. The MTP-off
runner therefore sealed a 1,864,785,668-byte serial-family arena; the following
Depth-1 runner required the complete 4,369,981,188-byte family and correctly
rejected reuse. A safety reserve would only have hidden the disagreement
between two authorities.

The underlying lifecycle error was a category collapse:

```mermaid
flowchart TD
    A[Matrix declares retained MTP depth 15] --> B[Memory admission charges retained capacity]
    B --> C{Old graph setup checks MTP enabled?}
    C -->|No in first MTP-off cell| D[Omit verifier, sidecar, KV, and workspace participants]
    D --> E[Seal undersized reusable arena]
    E --> F[Later Depth-1 cell requests declared family]
    F --> G[Exact reuse check rejects insufficient arena]
```

Retained capacity and transaction activation now have separate meanings.
`retainsMTPGraphCapacity()` is the sole setup-time authority for resources that
must exist in a retained graph family: decode-equivalent row geometry, terminal
hidden storage, shifted KV state, grouped-verifier and condition manifests,
sidecar logits participation, and the eager workspace-family declaration.
`mtp.enabled` remains the execution-time authority: it alone permits a request
to bind and launch an MTP transaction. The
`WorkspaceFamilyDeclaration` binding mode can describe the dormant family
without pretending that MTP executed.

```mermaid
stateDiagram-v2
    [*] --> DeclarativeCapacity: standard matrix publishes maximum retained depth
    DeclarativeCapacity --> ResourceInventory: resolve all retained graph participants
    ResourceInventory --> ExactWorkspacePlan: combine complete serial-family manifests
    ExactWorkspacePlan --> ArenaSealed: allocate the exact maximum block once
    ArenaSealed --> MTPDisabled: request policy selects MTP off
    ArenaSealed --> MTPEnabled: request policy selects fixed or dynamic depth
    MTPDisabled --> ArenaSealed: no MTP branch binds or launches
    MTPEnabled --> ArenaSealed: captured transaction completes
    ArenaSealed --> [*]: graphs retire before arena release
```

This is capacity accounting, not speculative headroom. Every byte is produced
by the same declared graph geometry later consumed by capture; an insufficient
block remains a fatal error. Focused unit gates require the capacity authority
on every affected participant, and a reused real-weight `ROCm1 + CPU2`
MTP-off-to-Depth-1 sequence passed with full graph-capture, prefix-restore, MTP,
and CSV evidence in one model context.

## Follow-up: parallel selector fan-out exposed a prepared candidate

The mixed `CUDA1 + ROCm1 + CPU2` Dynamic cell intermittently failed its first
segmented prefill with `InvalidControl`. Failure-only device status identified
an acquire for placement epoch 2 while the local selector still published
epoch 1. Epoch 2 was present in the other bank with `Ready` state, and a peer's
authenticated activation descriptor already named it. The former acquire
contract accepted only `Published` or `Retiring`, so it rejected a legitimate
intermediate state in the host-authoritative two-phase publication protocol.

The bug was not missing preparation or a stale descriptor. Global preparation
had completed before the candidate epoch became externally nameable. Selector
publication then deliberately fanned out across endpoints in parallel; making
that last step serial would remove the race only by adding inference-visible
latency and would contradict the movement architecture.

```mermaid
sequenceDiagram
    participant H as Host overlay authority
    participant A as Participant A
    participant B as Participant B
    H->>A: Prepare inactive bank E+1
    H->>B: Prepare inactive bank E+1
    A-->>H: Ready(E+1)
    B-->>H: Ready(E+1)
    H->>H: Open exact candidate epoch E+1
    par Parallel selector publication
        H->>A: Publish selector E+1
        H->>B: Publish selector E+1
    end
    A->>B: Authenticated activation descriptor E+1
    Note over B: Local selector may still name E while bank E+1 is Ready
    B--xB: Old acquire rejects Ready as InvalidControl
```

Serializing the selector fan-out, prematurely changing every participant's
publication floor, or accepting arbitrary prepared banks would each obscure a
different ownership error. Admission is instead represented by one scoped
policy with two explicit meanings:

```mermaid
stateDiagram-v2
    [*] --> Ready: global inactive-bank preparation consensus
    Ready --> PinnedPrepared: authenticated exact peer descriptor names E+1
    Ready --> Published: local selector atomically advances to E+1
    PinnedPrepared --> Published: local selector advances while reader remains pinned
    Published --> PinnedPublished: ordinary or exact acquisition
    PinnedPrepared --> Ready: exact reader releases before local publication
    PinnedPublished --> Published: reader releases
    Published --> Retiring: later selector publication
    Retiring --> Reusable: final reader releases
    Reusable --> Ready: prepare a later epoch
```

`PublishedFloor` admission continues to accept only `Published` or `Retiring`
banks. `ExactPreparedPeer` additionally accepts `Ready`, but only after the
existing peer descriptor authentication resolves one positive exact epoch.
It pins that bank with the anticipated selector generation and does not mutate
the local selector, publication floor, or host authority. CUDA and ROCm use the
same rule, and the host protocol exposes the same typed transition for
device-free adversarial testing. Generation overflow and every descriptor,
epoch, bank, and state mismatch remain fatal.

The focused proof comprises the device-free admission state machine, real CUDA
and ROCm epoch tests that hold E+1 in `Ready`, and the exact real-weight mixed
backend production cell. After the fix, that cell passed 20 fresh MPI worlds:
20 correctness passes, 20 artifact-contract passes, 160 validated parity
artifacts, and a 106.33--108.23 second runtime range under the unchanged
600-second exact-cell watchdog.

## Follow-up: parity teardown isolation was only partially adopted

The first broader campaign after the prepared-peer race proof found a harness
lifecycle defect. Qwen2 single-device, LocalTP, and LocalPP cells completed
their numerical comparisons and emitted passing CSV artifacts, then every cell
failed from `TearDown` with `Parity teardown has no isolated lifecycle channel`.
The common base had begun using a teardown-only communicator, but only the
heterogeneous GraphNative fixture constructed and retired that communicator.
The exact 20-run race gate therefore passed while unrelated fixture families
could not complete teardown.

```mermaid
flowchart TD
    B[Common ParityTestBase TearDown] --> R[Require teardown-only barrier]
    G[GraphNative fixture SetUp] --> C[Create control and teardown channels]
    G --> P[GraphNative cells pass]
    O[Every other parity fixture] --> N[No lifecycle channel]
    N --> R
    R --> F[Throw before runner and cache cleanup]
```

Channel ownership is now centralized rather than conditionally repaired in
each model fixture. `ParityCellLifecycle` is one typed object with three states.
Its active payload contains both communicator owners and both non-owning
`MPIContext` facades, so a control-only or teardown-only partial state cannot
exist. `ParityTestBase::SetUp` is the sole entry authority and
`ParityTestBase::TearDown` is the sole retirement authority. A derived setup
that skips before invoking the base remains explicitly `NotEntered`; every
entered lifecycle must cross both teardown barriers and retire exactly once.

```mermaid
stateDiagram-v2
    [*] --> NotEntered
    NotEntered --> Active: common base SetUp duplicates both lanes
    NotEntered --> [*]: derived prerequisite skips before base entry
    Active --> Active: setup and evidence use control lane
    Active --> Active: teardown entry barrier on teardown-only lane
    Active --> Active: destroy runner, graphs, caches, and model state
    Active --> Active: teardown exit barrier on teardown-only lane
    Active --> Retired: common base releases both lanes
    Retired --> [*]
```

The protocol now has two complementary gates. A source-policy unit requires
the common base to own the only entry/retirement calls and forbids GraphNative
or another topology fixture from reclaiming that authority. A model-free MPI
integration binary executes communicator isolation, ordered barriers, and all
invalid state transitions at both one and two ranks. The surrounding MPI
topology, rank initialization, orchestration, heterogeneous ticket, prepared
ExpertOverlay weight, CUDA/ROCm graph-capture, prefill-cache, and cached-replay
integration gates run alongside it before another real-weight campaign.

## Follow-up: one epoch deadline incorrectly bounded 48 healthy rendezvous

The ROCm2/CPU2 Static predecessor of the first Dynamic production cell failed
at routed layer 45 with dispatch timeline 22 while expecting 23. The terminal
snapshot was only 10 microseconds beyond its deadline and showed that the CPU
follower had successfully consumed every preceding layer. This localized the
fault to timeout scope rather than graph admission, ordering, or payload
publication: one deadline was created while the epoch was armed, then reused
for every dispatch and the final endpoint-completion wait.

The as-built lifecycle therefore treated a complete multi-layer inference
transaction as though it were one collective rendezvous:

```mermaid
flowchart LR
    A[Arm epoch and set T plus 30 seconds] --> S0[Wait for layer 0 dispatch]
    S0 --> C0[CPU expert compute and return]
    C0 --> SN[Repeat healthy layer rendezvous]
    SN --> S45[Wait for layer 45 using original deadline]
    S45 --> X{Wall clock beyond T plus 30 seconds?}
    X -- yes despite continuous progress --> F[Abort epoch as NotReady]
    X -- no --> Z[Wait for endpoint completion using same deadline]
```

That deadline was redundant with the typed activation state machine and had
the wrong semantic owner. A transaction may contain any admitted number of
layers and expensive CPU expert work; the standard 30-second collective
timeout bounds one absent peer publication, not the sum of all compute and
collectives in the request.

The simplified lifecycle retains one immutable epoch identity and creates a
fresh typed deadline only when a peer edge is actually expected:

```mermaid
flowchart TB
    A[Arm immutable epoch identity] --> L{Next routed layer?}
    L -- yes --> D[Begin DispatchPublication rendezvous]
    D --> W{Expected timeline published within 30 seconds?}
    W -- no --> X[Terminal abort with exact layer and timeline]
    W -- yes --> C[Acquire descriptor]
    C --> E[Run CPU expert compute outside wait budget]
    E --> R[Publish exact return]
    R --> L
    L -- no --> G[Acquire submitted GPU terminal events]
    G --> T[Begin EndpointCompletion rendezvous]
    T --> Q{Both typed endpoints Complete within 30 seconds?}
    Q -- no --> X
    Q -- yes --> Z[Validate traffic and reset epoch]
```

No progress flag or host shadow was added. The activation ABI now calls its
timestamp `timeout_not_before_ns`: it is merely the earliest legal terminal
watchdog observation, not a transaction budget. The immutable
`MoEOverlayActivationRendezvousDeadline` distinguishes dispatch publication
from terminal completion, is recreated at each exact edge, and can emit
timeout evidence only after that edge expires. A model-free integration proof
drives all 48 ordered round trips with synthetic elapsed time far greater than
one rendezvous interval, then proves that a genuinely stalled next generation
still fails terminally.

## Follow-up: accepted MTP publication was a late histogram producer

The first `ROCm1 + CPU2` Dynamic depth-two production cell completed prefill,
ordinary decode, and the grouped verifier, then failed while first
materializing the accepted-state publication graph. The background histogram
drain had already sealed its exact producer set. Publication nevertheless
created a cache-local stream at that late edge, so the runtime table correctly
rejected it as a foreign writer. The earlier stream-preservation fix covered
recapture after a publication graph already existed; it could not make the
first lazy stream exist before maintenance.

```mermaid
sequenceDiagram
    participant T as Runtime-table histogram authority
    participant V as Grouped verifier graphs
    participant M as Async maintenance
    participant P as Accepted-state publication cache
    T->>V: Admit already-materialized verifier streams
    V->>T: Publish grouped service and route evidence
    M->>T: Begin first bank rotation
    T->>T: Producer topology Open to Sealed
    P->>P: Lazily create cache-local stream
    P->>T: Attempt producer admission
    T--xP: Fatal foreign producer after seal
```

The simplified design removes that timing dependency. A mirrored runtime table
whose typed source mask includes `GroupedVerifier` creates one dedicated
accepted-publication stream while allocating its model-lifetime histogram
banks. It admits the stream immediately through the same initialization event
edge as every other producer. Every routed verifier publisher exposes that
single identity, the publication stage requires all layers to agree on it, and
the graph cache borrows it instead of owning another stream.

```mermaid
stateDiagram-v2
    [*] --> ResourcesAllocated: allocate two banks and maintenance stream
    ResourcesAllocated --> PublicationStreamAdmitted: create and register exact grouped-verifier stream
    PublicationStreamAdmitted --> ProducerTopologyOpen: materialize remaining graph families
    ProducerTopologyOpen --> ProducerTopologySealed: first asynchronous bank rotation
    ProducerTopologySealed --> PublicationGraphBound: lazy graph borrows pre-admitted stream
    PublicationGraphBound --> PublicationGraphBound: replay or replace capture identity on same stream
    PublicationGraphBound --> Retired: terminal graph fence and table teardown
    ProducerTopologySealed --> Fatal: any foreign stream requests admission
```

There is no host shadow, recapture fallback, or synchronization in inference.
The stream is part of complete capture identity and outlives every mutually
exclusive publication graph. Only terminal model teardown may fence and
destroy it. CUDA and ROCm integration tests now force the first rotation before
capturing publication work, prove replay on the reserved stream, and prove
that a genuinely late stream remains fatal. Those tests stay in the model-free
`ProductionParityPreflight` gate so this lifecycle is certified before any
real-weight campaign starts.

## Follow-up: static graphs falsely claimed deferred histogram ownership

The next production cell exposed a second ambiguity in the old boolean
contract. Grouped-verifier geometry alone made every routed MoE verifier stage
claim deferred histogram publication, even when Static placement deliberately
disabled runtime histogram collection. Graph setup then demanded a publication
stream that did not—and must not—exist in Static mode.

The correction is a typed, graph-wide boundary lifecycle. Graph lowering now
selects one of three roles from two independent facts: whether the stage owns a
routed-layer boundary, and whether the graph collects runtime history. The
orchestrator collects those boundaries once and resolves them atomically; it no
longer reconstructs intent from several booleans while wiring the publication
stage.

```mermaid
flowchart TD
    L[Declarative graph lowering] --> O{Owns routed-layer boundary?}
    O -- no --> N[NotOwner]
    O -- yes --> H{Runtime history enabled?}
    H -- no --> S[StaticNoPublication]
    H -- yes --> D[DeferredAcceptedRows]

    N --> I[Ignore during boundary discovery]
    S --> B[Register exactly one boundary per routed layer]
    D --> B
    B --> R{Resolve complete boundary set}
    R -- missing, duplicate, mixed role, or foreign stream --> F[Faulted: terminal setup error]
    R -- all StaticNoPublication --> P[Resolved: no publisher and no stream]
    R -- all DeferredAcceptedRows on one exact stream --> A[Resolved: ordered publishers plus publication stream]
    P --> K[Publish ordinary accepted KV/GDN/model state only]
    A --> C[Publish accepted rows into runtime histogram on exact stream]
```

```mermaid
stateDiagram-v2
    [*] --> Collecting
    Collecting --> Collecting: register routed layer or candidate stage
    Collecting --> Resolved: complete and internally consistent
    Collecting --> Faulted: missing/duplicate boundary, role mixture, or stream mismatch
    Resolved --> Faulted: any late mutation
    Faulted --> Faulted: fail closed
```

This is the minimum state machine required by the production distinction:
Static still owns an explicit verifier-history boundary, so graph validation
can prove that every routed layer was considered, but it has neither a
histogram publisher nor a synthetic stream. Dynamic/Observe owns deferred
accepted-row publication and must name the exact model-lifetime stream shared
by every layer. Dense graphs resolve as an empty boundary set. A model-free
integration test adversarially covers all valid and invalid transitions, and
the formerly failing real-weight Static/ordinal/FP16/MTP-depth-one cell proves
the resolved Static branch through captured production inference and its CSV
artifact contract.

## Follow-up: Dynamic teardown outlived borrowed HIP producer streams

The amortized Static-to-Dynamic campaign completed two real histogram
generations, then crashed in `ROCmBackend::recordEvent` during runner teardown.
The runtime table retained raw producer-stream identities, but ordinary C++
member destruction first cleared the graph caches and `DeviceContext` objects
that owned those streams. The table destructor subsequently tried to create a
terminal event edge on a dead HIP handle. A single-cell Static process never
registered those Dynamic producers, which is why the invalid lifetime remained
hidden until process-resident campaign reuse exercised the transition.

The old lifecycle conflated two independent operations in one destructor:

```mermaid
flowchart LR
    A[Inference stops] --> G[Destroy captured graph caches]
    G --> C[Destroy DeviceContexts and their streams]
    C --> T[Destroy graph builder and runtime tables]
    T --> E[Record terminal event on remembered producer pointer]
    E --> X[Use-after-destroy in CUDA or HIP runtime]
```

The corrected lifecycle gives producer references an explicit ownership type
and a terminal state transition. Graph/context streams are borrowed; the
grouped-verifier publication stream is table-owned. The device orchestrator
closes every model-internal producer DAG while the borrowed owners are still
alive, and only then destroys executable topology:

```mermaid
flowchart TD
    S[Stop inference admission] --> P[Retire exact published inference events]
    P --> R[Graph builder retires borrowed execution streams for this device]
    R --> A[Record one arrival on every live producer]
    A --> J[Maintenance stream joins all arrivals]
    J --> F[One terminal maintenance-stream fence]
    F --> B[Erase borrowed stream identities; retain table-owned handles]
    B --> G[Destroy captured graph families]
    G --> C[Destroy DeviceContexts and external streams]
    C --> T[Destroy runtime tables]
    T --> O[Destroy table-owned events, maintenance stream, and publication stream]
```

```mermaid
stateDiagram-v2
    [*] --> CollectingProducers
    CollectingProducers --> CollectingProducers: admit exact borrowed or table-owned stream
    CollectingProducers --> ProducersRetired: terminal producer-DAG join while owners are live
    ProducersRetired --> ProducersRetired: idempotent retirement
    ProducersRetired --> ResourcesReleased: destroy only table-owned resources
    CollectingProducers --> Fatal: borrowed producer reaches resource destruction
    ProducersRetired --> Fatal: producer admission or histogram progress resumes
```

There is no per-request fence, replacement stream, backend fallback, or host
shadow. The one blocking fence is permitted only after inference admission has
stopped and covers the complete producer family. CUDA and ROCm model-free
integration regressions deliberately retire the DAG, destroy the external
producer stream, and only then destroy the runtime table. Both registrations
are part of `ProductionParityPreflight`, and a unit source-policy assertion
locks the retirement call ahead of graph/context destruction.

## Follow-up: snapshot manifest construction observed device-owned KV state

Removing the redundant device-wide teardown drains exposed one remaining
phase-skew defect in the process-resident ROCm campaign.  While one LocalTP
participant was beginning native graph capture, another participant assembled
the immutable diagnostic manifest for the same transaction.  The attention
manifest builder called the scalar KV-cache accessor, which synchronously
downloaded the GPU-owned cache head/count.  HIP correctly rejected the
resulting legacy-stream dependency on a peer's capturing blocking stream.

The failing lifecycle mixed descriptor construction with execution-state
observation:

```mermaid
sequenceDiagram
    participant A as LocalTP participant A
    participant B as LocalTP participant B
    participant K as GPU KV-cache authority
    A->>A: Begin native graph capture
    B->>B: Build immutable snapshot manifest
    B->>K: Read scalar cached-token count on host
    K->>K: Device-wide observation and synchronous D2H
    K--xA: Legacy stream depends on capturing stream
    A--xA: Capture invalidated
```

Snapshot manifest construction now consumes only graph-stable tensors already
published by the attention execution stage.  GPU sequence state remains
device-owned throughout preparation and capture; the CPU branch retains its
ordinary host-owned cache inspection.  The manifest describes addresses and
maximum logical geometry, while captured D2D copy nodes and their exact
completion event publish the requested diagnostic contents.

```mermaid
flowchart TD
    E[Attention execution resolves exact cache-owned K/V views] --> P[Publish graph-stable diagnostic tensor views]
    P --> M[Every participant builds immutable snapshot manifest]
    M --> C[Every participant captures D2D snapshot nodes]
    C --> L[Launch retained production graph]
    L --> V[Record exact snapshot completion event]
    V --> H[Diagnostic host reader waits on that event]
    CPU[CPU cache is host-owned] --> CH[CPU-only scalar descriptor branch]
    CH --> M
```

The simplified invariant is structural: a GPU snapshot descriptor may never
call `get_cached_tokens()` or `get_kv()`; only the execute-published tensor
view is authoritative.  A source-policy regression locks that boundary.  The
exact two-ROCm Qwen3.6 campaign then passed all 24 process-resident cells—both
owner orders, Static and Dynamic placement, and MTP off, depths 1/2/3/15, and
dynamic depth—with 212 CSV artifacts validated.  Those 24 ordered transitions
also provide the required greater-than-20 stress proof for the original race.

## Follow-up: GDN linked-state handoff had two contradictory layouts

The Qwen3.5 TP=3 production proof exposed an older GDN assumption: fused Q/K/V
weights were sliced as though Q and K were replicated while V alone followed
the participant. The actual recurrence dependency is modular—each value head
consumes one key head—so a participant must own one contiguous Q/K interval and
every value-head interval linked to it. Once the weight layout was corrected,
the equal-count TP=2/4/8 live-state allgather still concatenated those local
banks rank-major, while mirrored decode consumes semantic groups globally.

```mermaid
flowchart LR
    W[Global Q, K, and repeated V groups] --> S[Dependency-closed TP slicer]
    S --> R0[Rank 0: Q0, K0, V0 repeat 0..N]
    S --> R1[Rank 1: Q1, K1, V1 repeat 0..N]
    R0 --> G[Raw equal-count allgather]
    R1 --> G
    G --> X[Rank-major bytes: rank 0 groups, rank 1 groups]
    X --> P[Captured device-side permutation]
    P --> F[Full semantic order: all Q, all K, each complete V repeat]
    F --> D[Mirrored dense decode]
```

`GDNLinkedLiveStateGeometry` is now the sole accounting and layout authority.
It derives convolution-history and recurrence sizes from global key/value head
counts, per-head widths, history length, and TP degree. CUDA and ROCm launch one
explicit-stream permutation into a distinct persistent destination, so raw
collective input cannot be overwritten while it is still being read. TP=3 is
valid for dependency-closed projection slicing but cannot use an equal-count
state collective when 16 key heads divide unevenly; it stays on the TP-local
state path instead of pretending that incompatible counts form a full bank.

The verifier lifecycle had retained a second, unreachable mechanism from the
old layout. A `GDNLiveStateLocalizeStage` was supposed to convert mirrored
state back to local state for verification, after which the generic MTP
publisher could invoke a non-capturing post-restore allgather. Its construction
predicate simultaneously required the allgather to be available and
unavailable. Production verifier graphs already use full mirrored GDN kernels,
whose short-convolution and recurrence stages capture and restore their own
full state directly.

The lifecycle is therefore one-way and has one accepted-state authority:

```mermaid
stateDiagram-v2
    [*] --> LocalPrefillState: dependency-closed local kernels
    LocalPrefillState --> FullDecodeState: captured raw allgather plus device permutation
    FullDecodeState --> SpeculativeFullState: grouped verifier uses isolated full-state scratch
    SpeculativeFullState --> FullDecodeState: accepted row restored by each exact state owner
    SpeculativeFullState --> FullDecodeState: rejected transaction leaves live bank unchanged
    FullDecodeState --> FullDecodeState: ordinary captured decode
```

`GDNLiveStateLocalizeStage`, its stage type/factory path, and the generic
`requiresPostVerifierStatePublication()` / `publishPostVerifierStateRestore()`
hooks have been removed. MTP publication now has only two graph-node outcomes:
a verifier-capturing state owner restores its selected row and detaches scratch,
or a non-state stage is skipped. CPU-only protocol tests prove the layout and
accounting through TP=8; captured CUDA and ROCm integration tests prove both
state kinds byte-exact through TP=8 and run in `ProductionParityPreflight`.

## Follow-up: device-owned restoration crossed split teardown authorities

The exact 122B `CUDA2 + ROCm4`, Dynamic, ordinal, FP16 activation/KV,
MTP-off cell eventually passed every Hugging Face checkpoint, movement proof,
and prefix-restore comparison but failed while retiring its reusable model
context. This was not a collection of unrelated numerical defects. It was the
first test to traverse the complete process-resident Dynamic lifecycle, and
each earlier failure prevented the next terminal edge from executing.

The masked sequence was:

1. An empty post-calibration transaction advanced the recurring device cadence
   despite observing zero routed rows.
2. Movement-proof traffic populated the production prefix cache with the parity
   prompt, so the subsequent parity request restored all rows and produced no
   fresh graph snapshots.
3. Device RCU restored the runtime owner table, but terminal physical sealing
   tried to acquire the deliberately stale host participant-bank epoch.
4. Once physical sealing used the device slot ledger, registry rebinding still
   re-read the stale host residency snapshot and compared its setup epoch with
   the later restored durable epoch. The physical seal succeeded, but the model
   lifecycle remained in `Sealing` and exact final-owner retirement rejected it.

That explains the serial discovery pattern: every defect was deterministic,
but a failed predecessor made its successor unreachable. Unit tests covered
the individual cadence, prefix, RCU, fabric, and retirement components; the
real-weight process campaign was the missing composition proof.

The audited production lifecycle now has one explicit forward path:

```mermaid
flowchart TD
    A[Initial prepared placement and owner map] --> B[Bounded transport profile]
    B --> C{Observed routed rows?}
    C -->|No| D[Keep recurring cadence unchanged]
    C -->|Yes| E[Advance evidence-derived cadence]
    D --> F[Certify live service economy]
    E --> F
    F --> G[Dynamic movement proof]
    G --> H[MovementBoundarySettled]
    H --> I[Coordinated production prefix purge]
    I --> J[NumericalParityReady]
    J --> K[Captured prefill and decode parity]
    K --> L[Fresh, complete, and partial prefix restore]
    L --> M[TerminalContextSeal]
    M --> N[Close admission and drain device service]
    N --> O[Device authors PreparedContextRestore]
    O --> P[Publish and retire exact durable restore epoch]
    P --> Q[Snapshot quiescent physical slot ledger]
    Q --> R[Self-contained physical seal: canonical owner map plus local banks]
    R --> S[Atomically replace model ExpertGemmRegistry bindings]
    S --> T[Seal exact retained allocation BOM]
    T --> U[Sealing to Reusable]
    U --> V{Another campaign cell?}
    V -->|Yes| W[Reusable to RunnerExclusive]
    V -->|No| X[Exact final-owner device retirement]
    W --> F
```

The ownership split is deliberate and non-overlapping:

| Live value | Sole authority |
|---|---|
| Dynamic logical placement and inference selector | device controller/runtime table |
| Process-local prepared-engine lifetimes | device physical slot ledger |
| Terminal restored placement and canonical engine banks | `MoEOverlayReusableContextSeal` |
| Model-visible routed GEMM bindings after sealing | `ExpertGemmRegistry` |
| Runner admission/reuse state and retained allocation BOM | `ModelContextReuseAuthority` |

The physical seal now carries the exact prepared `MoEExpertOwnerMap` represented
by its immutable local banks. Its validity check proves epoch, participant,
device, layer/expert geometry, owner masks, and triplet completeness together.
Consequently, registry rebind cannot consult a host residency snapshot or infer
that two different epoch numbers should match. Host-owned RCU and device-owned
RCU still use different inventory sources, selected by the typed
`MoEOverlayPhysicalInventoryAuthority`, but both converge to the same terminal
seal type and the same atomic registry replacement.

Focused proof includes the quiescent device physical-inventory state machine,
all 20 quantized codebooks plus FP16/BF16/FP32 terminal sealing, and the real
CUDA/ROCm prepared-context restoration integration. The exact 122B cell then
passed both MPI ranks through clean suite teardown in 296.5 seconds, emitted
all numerical and prefix CSVs, certified one restoration wave to durable epoch
3, rebound 2,058 CUDA and 10,486 ROCm resident experts, and retired all six
device runtime generations.

## Follow-up: logical transfer edges multiplied physical HIP resources

The `ROCm1 + CPU2` 122B Dynamic cell exposed a different composition defect
before inference began. The physical residency fabric had correctly retained
one logical lane for every directed edge, projection, remote role, and admitted
cycle. It had incorrectly made every logical lane allocate its own 4 MiB pinned
buffer, device buffer, and HIP stream. A 49-cycle policy therefore expanded
into hundreds of host registrations and stream creations even though the GPU
could execute only the bounded cycle cohort concurrently. On this host, HIP
registration and queue setup made model admission exceed the cell watchdog.

Logical command identity, storage concurrency, and execution concurrency are
now three explicit quantities:

```mermaid
flowchart TD
    A[Typed topology and integer-priority tier plan] --> B[Resolve maximum concurrent physical cycles]
    B --> C[Build complete logical command directory]
    B --> D[Allocate one pinned slab and one device slab per compatible lane pool]
    D --> E[Issue disjoint typed staging slices]
    B --> F[Materialize one background stream per local GPU participant and cycle]
    F --> G[Issue typed execution-lane handles]
    C --> H[Construct projection, edge, and remote-role state machines]
    E --> H
    G --> H
    H --> I[InferenceReady]
    I --> J[Authority admits a bounded movement wave]
    J --> K[Compatible logical operations share their participant/cycle stream]
    K --> L[Independent completion events publish each operation]
    L --> M[Atomic placement publication at the wave boundary]
```

The important separation is that logical lanes still own disjoint staging
regions and completion events. Only compatible operations share an execution
queue. Different admitted cycles retain different queues, so movement can
progress in parallel up to the typed physical-cycle budget without creating a
queue for every graph edge. `TransferEngine` is the sole allocation and stream-
pool authority; weight, peer, and cross-rank endpoint lanes reject a missing or
wrong-device handle and never create a stream themselves.

Shared slabs have one RAII owner. Lane cleanup destroys only its own events and
clears borrowed slice views; it cannot free an interior pinned/device address.
The stream pool is setup-owned by the worker GPU context and uses the
`BackgroundMaintenance` scheduling class. Live movement performs no allocation,
registration, stream construction, or blocking synchronization.

```mermaid
stateDiagram-v2
    [*] --> GeometryValidated
    GeometryValidated --> SlabsBound: allocate shared host/device slabs
    SlabsBound --> StreamsBound: materialize bounded background pool
    StreamsBound --> LanesMaterialized: bind slices, typed streams, and private events
    LanesMaterialized --> Pending: authority submits disjoint operation
    Pending --> Pending: event query completes one chunk and enqueues the next
    Pending --> Ready: exact lane completion event observed
    Ready --> LanesMaterialized: release logical reservation
    LanesMaterialized --> Retired: all lanes quiescent before context retirement
    GeometryValidated --> Fatal: zero/overflowing geometry or wrong device
    StreamsBound --> Fatal: null/missing/mismatched execution handle
    Pending --> Fatal: event or transfer failure
```

The regression surface is deliberately layered. Device-free tests prove one
allocation per slab and invalid typed geometry. CUDA and ROCm integration tests
prove exact bounded stream identity and idempotent reuse. A second adversarial
test races two multi-chunk FP16/BF16 transfers on one physical stream while
retaining independent events and byte-exact output; it passed 20 consecutive
runs on each backend. The production parity fixture additionally requires the
real Dynamic model to publish `transfer` path evidence whose lane count is a
positive multiple of its configured migration-cycle capacity. That PerfStats
record is observability only: completed movement remains authoritative solely
through the typed optimization movement ledger.

## Follow-up: partial prefix restore crossed a valid movement epoch

The CPU NodeTP Dynamic/Random production cell eventually passed every
Hugging Face checkpoint, KV comparison, terminal-payload comparison, and
movement assertion, but rejected the layer-4 GDN recurrence hash. The serial
oracle ran at movement epoch 8 and the independent partial-restore request ran
at epoch 9. Dynamic maintenance had validly published between those requests;
it did not mutate either admitted request in place.

The cached prefix was still byte exact. Only the one-token suffix was
recomputed through a different expert placement, so its CPU participant and
reduction order introduced a small bounded floating-point change that then
propagated into the recurrent state. Main KV and terminal payloads already had
a placement-aware comparison contract. GDN retained two booleans that always
selected byte hashes, making it the lone state family that interpreted a
legitimate request-boundary publication as corruption.

```mermaid
sequenceDiagram
    participant O as Serial oracle request
    participant A as ExpertOverlay authority
    participant C as Prefix cache
    participant R as Partial-restore request
    O->>A: Admit and pin placement epoch 8
    O->>C: Seed exact prefix and retain full diagnostic state
    O-->>A: Complete and release epoch lease
    A->>A: Prepare, certify, and atomically publish epoch 9
    R->>A: Admit and pin placement epoch 9
    C-->>R: Restore byte-exact cached prefix
    R->>R: Recompute one-token suffix under epoch 9
    R->>R: Compare authenticated epoch-8 and epoch-9 states
```

The comparison lifecycle now has one typed policy rather than independent
hash/value flags:

```mermaid
stateDiagram-v2
    [*] --> MetadataValidated
    MetadataValidated --> ExactComparison: ExactBytes
    MetadataValidated --> NumericalComparison: NumericalValues
    MetadataValidated --> ExactComparison: ExactUnlessMoEPlacementChanged and epochs equal
    MetadataValidated --> NumericalComparison: ExactUnlessMoEPlacementChanged and epochs differ
    ExactComparison --> Passed: every authoritative byte hash matches
    ExactComparison --> Failed: any byte hash differs
    NumericalComparison --> Failed: authority is asymmetric or a full bank is missing
    NumericalComparison --> Failed: NaN, Inf, cosine, or scale-aware relative-L2 gate fails
    NumericalComparison --> Passed: every recurrence and short-conv value is certified
    Passed --> [*]
    Failed --> [*]
```

CPU probes copy the complete host-owned GDN bank. GPU probes reuse the exact
stream export used for hashing and select the full replicated decode bank from
the serialized local/full layout; they never compare a stale host mirror or a
range spanning two banks. The CSV contract records policy, payload and element
counts, minimum cosine, maximum relative L2, maximum absolute difference, and
the aggregate decision. Thus an epoch change cannot weaken or skip evidence:
it changes exact bytes into a complete numerical proof.

A separate ambiguity appeared only during the CPU-only cell's teardown. An
empty device-retirement BOM meant “host-only context,” but the caller passed it
to the strict GPU retirement API, which correctly rejected an empty batch.
`PendingExclusiveModelRetirement` now represents `HostOnly` and `DeviceBatch`
as disjoint kinds. The host-only transition consumes its plan exactly once and
never fabricates a GPU transaction.

The focused production cell passed 20 fresh two-rank MPI processes. Two runs
exercised genuinely non-byte-identical GDN banks and passed complete numerical
comparisons; the remaining schedules happened to stay byte exact. The
canonical four-cell CPU NodeTP campaign then passed Static/Dynamic and
Ordinal/Random placement, validated all 32 CSV artifacts, and completed in
168.3 seconds including its model-free preflight. A dedicated model-free
CPU-hybrid integration regression now runs in `ProductionParityPreflight` so
the cross-epoch contract is checked before any model campaign starts.

## Follow-up: rank-local fail-fast crossed parity-cell identity

The canonical `ROCm1 + CPU2`, Dynamic/Ordinal, Depth-15 cell produced its full
graph-path and numerical CSV set in about 90 seconds, yet the campaign watchdog
expired at 600 seconds. A faithful two-cell process reproduction showed the
cause: the comparison-owning rank rejected one recursive LM-head KL value while
its peer passed. GoogleTest fail-fast is process-local, so the failing rank
stopped after teardown and the passing rank entered the next generated cell.
Its next-cell collectives could never match the peer's suite-retirement work.

The old test lifecycle had two indistinguishable barriers and no result
authority spanning ranks:

```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant B0 as Rank 0 GoogleTest
    participant R1 as Rank 1
    participant B1 as Rank 1 GoogleTest
    R0->>R1: Cell teardown entry barrier
    R0->>B0: Local numerical failure
    R1->>B1: Local pass
    R0->>R1: Cell teardown exit barrier
    B0-->>R0: Fail-fast ends suite
    B1-->>R1: Advance to next generated cell
    R1->>R0: Next-cell MPI collective waits forever
```

`ParityCellLifecycle` now makes that invalid interleaving unrepresentable. The
teardown communicator has one typed transaction, and a rank-wide reduction
creates an immutable outcome receipt before either process may inspect
fail-fast or retire the communicator:

```mermaid
stateDiagram-v2
    [*] --> NotEntered
    NotEntered --> Active: enter and duplicate control/teardown lanes
    Active --> TeardownEntered: beginTeardown entry rendezvous
    TeardownEntered --> OutcomeConverged: convergeOutcome all-reduce
    OutcomeConverged --> Retired: cleanup complete + finishTeardown rendezvous
    Retired --> [*]
```

```mermaid
sequenceDiagram
    participant R0 as Comparing rank
    participant T as Isolated teardown channel
    participant R1 as Peer rank
    R0->>T: Failed
    R1->>T: Passed
    T-->>R0: AtLeastOneRankFailed, failed ranks = 1
    T-->>R1: AtLeastOneRankFailed, failed ranks = 1
    R1->>R1: Publish aggregate failure into local GoogleTest state
    R0->>T: Finish teardown
    R1->>T: Finish teardown
    T-->>R0: Both ranks retire the same cell
    T-->>R1: Both ranks retire the same cell
```

The model-free MPI preflight supplies a synthetic one-rank failure as data and
proves that every participant receives the same aggregate result. It also
adversarially rejects every skipped or repeated transition. This does not make
test evidence a production authority: it only prevents the campaign harness
from losing MPI cell identity when a valid production comparison is red.

The numerical rejection itself was independent and deterministic. Repeated
artifacts measured a maximum KL of 0.06695 at the fourteenth recursive FP16
round trip, while cosine and mutual top-3 evidence remained within their strict
contracts. Per the explicit campaign decision to budget this fixed-depth
quantized drift rather than switch activations to FP32, only Depth-15 receives
a 0.07 KL threshold; ordinary MTP and dynamic-depth remain at 0.05.

## Follow-up: heterogeneous main inference had a split epoch transaction

The `ROCm4 + CPU2` Static/Ordinal cell completed prefill but its first decode
timed out before layer-zero sparse dispatch. Native graph inspection proved the
retained parent had one root and one fully chained DAG, and the production-
shaped TP4/RCCL analogue completed under the same graph and VRAM pressure. The
remaining difference was the residency envelope: heterogeneous main inference
submitted a separate one-node acquire graph, made the parent stream wait on its
event, and then immediately submitted the much larger retained parent. All four
input edges were complete, but HIP scheduled the tiny producer on only one
participant while the other three output events remained pending. The parent
could not begin because it consumed those events.

The old topology-dependent lifecycle was:

```mermaid
flowchart LR
    P[Prior release receipt] --> S[Submission stream]
    S --> A[Separate captured Acquire graph]
    A --> E[Acquire completion event]
    E --> M[Retained main parent waits]
    M --> F[Captured model and sparse collectives]
    F --> R[Separate captured Release graph]
    R --> N[Next release receipt]
```

This represented one logical forward as three native graph launches and two
cross-stream event handoffs. Placement authority did not require that split;
it only chooses which published bank the acquire kernel observes. The ordinary
main graph was already a complete captured transaction under either host- or
device-authoritative placement.

Main inference now has one transaction shape for every GPU topology:

```mermaid
flowchart LR
    P[Prior immutable release receipt] --> W[Exact parent stream event wait]
    subgraph MAIN[One retained captured main epoch transaction]
        A[Acquire root] --> F[Model, collectives, and heterogeneous tickets]
        F --> R[Release terminal]
    end
    W --> A
    R --> N[Record immutable release receipt]
```

The typed policy depends only on two facts: whether the participant has an
epoch binding and whether the graph is ordinary main inference. Controller
location, tier count, backend mix, and maintenance timing are not lifecycle
proxies. Auxiliary MTP graphs remain deliberately different because several
captured children share one external sequence or hosted-parent reader:

```mermaid
stateDiagram-v2
    [*] --> Unbound: no epoch binding
    [*] --> CapturedMain: main inference
    [*] --> ExternalSequence: auxiliary or MTP child
    CapturedMain --> ReleasePublished: Acquire + forward + Release in one graph
    ExternalSequence --> ExternalSequence: child borrows enclosing reader
    ExternalSequence --> ReleasePublished: sequence terminal releases once
    ReleasePublished --> CapturedMain: next parent waits on receipt
```

The focused preflight reproduces the four-participant, 49-fragment retained
RCCL parent, mapped CPU ticket service, approximately 3,800 nodes, and model-
like VRAM pressure. It embeds the real epoch acquire as the first node and the
real release as the terminal, replays two retained families, and proves every
reader count and request ticket is retired. This locks down the simplified
production transaction before any real-weight campaign begins.

## Follow-up: promoted MTP evidence was consumed before its producer

The `CUDA2 + ROCm4`, Dynamic/Random, Depth-1 cell completed graph-captured
prefill, prefix restore, decode, and its movement epoch. That epoch promoted
two experts from model layer 48, the Qwen3.5 MTP sidecar. The fixture then
rejected an empty promoted-expert witness set even though it already owned an
exact primary-MTP checkpoint observer capable of producing those witnesses.

The failure was an ordering bug in the proof lifecycle, not a numerical or
physical-movement failure:

```mermaid
flowchart LR
    A[Movement epoch publishes layer-48 promotions] --> B[Main prefill parity]
    B --> C[Main decode and prefix-restore parity]
    C --> D[Old promotion epilogue consumes witnesses]
    D --> E[Failure: MTP witness set is still empty]
    E -. unreachable evidence producer .-> F[Primary Hugging Face MTP checkpoint]
```

Main and MTP checkpoints are independent evidence producers with disjoint
snapshot namespaces. A main-model promotion may be witnessed by prefill or
decode, while an MTP-layer promotion can only be witnessed after the primary
sidecar checkpoint. The fixture now encodes that distinction in the existing
Dynamic proof state machine:

```mermaid
stateDiagram-v2
    [*] --> NumericalParityReady: movement settled and proof prefix purged
    NumericalParityReady --> NumericalEvidenceComplete: main parity, MTP disabled
    NumericalParityReady --> AwaitingMTPParity: main parity, MTP enabled
    AwaitingMTPParity --> NumericalEvidenceComplete: primary MTP checkpoint compared
    NumericalEvidenceComplete --> PromotionEvidenceEpilogue: join movement, placement, and numerical witnesses
    PromotionEvidenceEpilogue --> [*]
```

There is no retry, synthetic route, forced token, or extra inference phase.
The same production checkpoints already required by the canonical cell now run
before their single evidence consumer. A model-free preflight regression
adversarially rejects MTP-before-main, repeated-main, and MTP-when-disabled
transitions, and proves both the MTP and non-MTP terminal paths.

## Follow-up: topology-valid movement was not necessarily numerically witnessable

After fixing the MTP evidence ordering, the same Dynamic/Random/Depth-1 cell
still reached a valid movement boundary without producing a promoted-expert
numerical witness. The first two production promotions were layer 0 expert 226
and layer 1 expert 69. Neither expert appeared in any authenticated Hugging Face
prefill or decode route for this fixed corpus. Later production waves promoted
layer 6 expert 120 and layer 9 expert 235, which appeared five and four times,
respectively, in the authenticated prefill routes.

This exposed two distinct convergence predicates that had been collapsed into
one:

1. A service-economical movement is topology-valid and may improve residency
   for production traffic.
2. A parity-proof movement is also selected by the immutable authenticated
   workload, so the captured checkpoint can execute its destination and compare
   its contribution with the Hugging Face reference.

The proof lifecycle now keeps those predicates explicit:

```mermaid
flowchart TD
    S[Ordinary service-economy traffic] --> L[Typed production movement ledger]
    A[Authenticated Hugging Face prefill traffic] --> L
    L --> T{Required topology publications and movement axes satisfied?}
    T -->|No| D[Continue ordinary production traffic]
    T -->|Yes| W{Ledger suffix contains an authenticated-workload promotion?}
    W -->|No| D
    W -->|Yes| B[Settle at a between-wave boundary]
    B --> C[Cache only numerically witnessable promotion identities]
    C --> P[Captured prefill, decode, prefix-restore, and MTP comparisons]
    P --> J[Join movement, selected destination, and per-route numerical evidence]
```

The authenticated route counts are a test-side convergence requirement only.
They never enter the ExpertOverlay controller, alter its histogram, force a
route, or choose a placement. The production controller continues making its
normal device-authoritative decisions, and every committed movement remains in
`expert_movement.csv`; the numerical witness set merely selects the ledger
suffix entries that the fixed reference corpus can independently exercise.

The corresponding model-free preflight proves that a topology-valid promotion
absent from the authenticated workload remains pending, that a later routed
promotion satisfies the proof, and that malformed ledger geometry fails
fatally. This makes the evidence dependency explicit without using PerfStats as
an authority or weakening the production movement path.

## Follow-up: finite traffic horizon was not a demand-bank lifecycle

The 35B heterogeneous Dynamic/Ordinal cell completed an 82-migration physical
transaction and published generation two, then waited for movement with that
successor bank empty. The test driver had spent its global request allowance
while generation one was frozen for asynchronous transfer. Those requests
correctly kept inference live, but they could not populate a bank that did not
yet exist. Treating submitted request count as bank occupancy therefore created
a passive state with no remaining producer.

The broken control flow was:

```mermaid
flowchart LR
    O[Spend global request horizon] --> C[Close generation 1]
    C --> M[Move and publish asynchronously]
    M --> E[Generation 2 is empty and quiescent]
    E --> W[Classify as AwaitMovement]
    W --> W
```

Settlement now follows the authority's generation and occupancy rather than
trying to infer either from request count:

```mermaid
stateDiagram-v2
    [*] --> AdmitOrdinaryTraffic
    AdmitOrdinaryTraffic --> AwaitMovement: bank full or movement in flight
    AdmitOrdinaryTraffic --> Settle: finite ordinary horizon exhausted
    AwaitMovement --> Settle: publication and retirement complete
    Settle --> SeedEmptyBank: target incomplete and occupancy = 0
    SeedEmptyBank --> AwaitDemandPublication: authenticated prefill + boundary submitted
    AwaitDemandPublication --> AwaitDemandPublication: generation and occupancy unchanged
    AwaitDemandPublication --> ClosePartialBank: occupancy increased
    ClosePartialBank --> AwaitDemandPublication: exact remaining rows submitted
    AwaitDemandPublication --> AwaitMovement: bank completed
    Settle --> CloseProtectedBoundary: target complete but cohort lacks headroom
    CloseProtectedBoundary --> AwaitDemandPublication
    Settle --> Ready: target complete and boundary quiescent
    Ready --> [*]
```

One typed admission receipt carries the generation and pre-submit occupancy for
both seeding and exact closure. The next admission is impossible until the sole
optimization authority either increases occupancy or rotates to a newer
generation; regression is fatal. Movement work is bounded by the number of
authority generations needed for the publication target plus its authenticated
promotion witness, while post-target cohort closure remains a separate state.
This removes the former optional-boolean/pending-closure control flow and keeps
the 30-second watchdog attached to one unchanged authoritative frontier.
