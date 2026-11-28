# Cache-centric continuous batching with tiered, pooled KV

**Revision 5 — owner-approved direction, 2026-11-25.** Maximize useful GPU KV
residency within complete PMA admission; provide independently configurable
RAM and disk retention, a configurable disk directory, and UMA-aware physical
accounting. Retain and restore valuable computation. Request identity and
continuation survive changes in residency.

This revision supersedes revision 4's resident-only independent-maximum
admission as the initial product policy and its deferral of complete suspension
to a later offload project. Pooled native access, sharing and COW remain core.
Complete bounded pressure handling and suspension/restore now belong to the
main delivery path. The owner approved this architecture, not an implemented
capability, a numerical size/default table, or a gate bypass.

**Amendment, 2026-11-26 (owner-requested).** §8.4 adds client retention intent
through the existing standard request fields only, and the tier ordering that
intent selects. It amends §§4.3 and 14. It is a documented contract, not an
implemented capability, a new public option table or a numeric default.

This is a detailed design, not a dashboard or capability certificate. The
[implementation plan](plan_continuous_batching_implementation_20261125.md) owns
sequencing, implementation status and evidence. Earlier
[research](research_continuous_batching_20261105.md) and
[reviews](review_continuous_batching_20261105.md) retain their historical,
source-pinned meaning. Historical seals and failed attempts remain intact.
The independent revision-4 findings PKV-01–09 inform the explicit contracts
below; documenting their resolution does not prove their implementations.

The adopted G1 serving decisions remain: immutable ingress, request-intrinsic
bounded windows, authenticated terminal/ticket receipts, and separate
execution/response lifetimes. GPU model decisions remain device-owned.
Intentional tier I/O adds neither a host controller nor permission to inspect
mutable GPU execution state. Full refactoring is permitted; project maxims and
native correctness/economy gates remain mandatory.

## 1. Decision and scope

Independent requests and idle reusable prefixes use common physical page pools,
without fixed request partitions or a fixed active-versus-cache split. Logical
tokens map to versioned page references. Work rows, request records and resident
frame addresses are different coordinates. Pages are assigned incrementally;
immutable content can be shared; every writer owns an exclusive tail or COW
destination.

| Concern | Selected target |
|---|---|
| Objective | Completed useful output within latency/fairness targets; not batch size, occupied bytes or cache-hit percentage alone |
| GPU pool | Largest legal shared KV pool after all other admitted obligations, with an adjustable explicit capacity cap |
| Lower tiers | Bounded configurable RAM and disk; explicit disk location; valuable state can move without ending its request |
| UMA / CPU | One physical memory budget for shared CPU/GPU memory; omit a redundant retained RAM copy; CPU RAM is already its resident tier |
| Pressure | Bounded optimistic execution admission backed by an executable all-resource progress witness, not independent full-context reservations for every request |
| Preservation | Retain/restore first; complete live continuation is protected, while disposable idle cache can be evicted |
| Scheduling | Cache-aware bounded lookahead, restore preparation, intrinsic chunked prefill, aging and bounded bypass |
| GPU execution | Device-owned selection, page tables, live extents and complete retained transactions; no mutable host mirror or recapture |
| Authority | Runner-owned `ServingSession`, PMA physical memory/page ownership, `TransferEngine` movement, one disk-extent authority |
| Delivery | Detached bounded response storage whose credit survives the final transport borrower |

GPU → RAM → disk is the normal discrete-memory hierarchy when those tiers are
configured and admitted. It is not an obligation to copy every object through
every tier. A valid unchanged backing replica avoids another write; a low-value
idle prefix can be discarded. On UMA the useful hierarchy can be GPU-visible
unified memory → disk. Host I/O staging is still real, charged memory.

Stopping computation, removing GPU residency and discarding saved state are
different operations. Ordinary cache eviction must not destroy a live request's
only continuation. Merely parking requests does not free pages.

Recomputation is not the default pressure escape. A future explicitly selected
recomputation policy would require its own preservation, numerical and matched
economy contract; this design enables no hidden recompute, scalar replay,
context reduction or MTP disablement. Cache size alone cannot determine whether
restore is economical: compare the matched end-to-end restoration critical path
with actual avoided prefill for the model, format, hardware and context.
Reported MI50, H100 RAM and DGX Spark/TensorFold disk benefits motivate the
retain/restore direction; they are workload observations, not universal results
or independent qualification of Llaminar.

CPU, CUDA and ROCm; ordinary decode and MTP; dense/MoE and recurrent hybrids;
and valid admitted TP/EP/PP compositions are the target. Missing combinations
remain explicitly unqualified, never silently redirected. Capacity one uses
the same native paged seam. Existing supported behavior must survive qualified
cutover; there is no fixed-slot compatibility mode.

## 2. Logical identities, not physical lanes

Use generation-qualified identities with distinct meanings:

| Identity | Meaning |
|---|---|
| Request handle | Nonreused external response incarnation, including queued, suspended and completed delivery states |
| Request binding | Serving-instance identity, request-record index and monotonic generation |
| Work row | Temporary compact coordinate for one operation; maps back to a request binding and logical token position |
| Logical KV block | A token interval in a request's ordered history, with its cache-family/role identity |
| Physical page handle | Pool, participant, storage class, frame index and non-reused frame generation |
| Operation-readable view | Exact content/range and table version, registered before enqueue and pinned through completion |
| Committed publication | Accepted history, visible length and coherent checkpoint/terminal version, distinct from tentative readability |
| Archive/replica identity | Immutable content/version and placement/format identity, plus a distinct storage incarnation |

An index retained for request-local sampler or recurrent state is not a KV
partition. No address formula may recover KV as `slot * full_context_bytes`.
One long request can use more than `pool_pages / request_capacity` pages while
other requests use fewer, subject to admission. Repacking live work must not
move request identity or copy an entire KV history.

Request, frame, grant and table generations never wrap or silently reuse an
incarnation. Exhaustion fails before admission. Duplicate live external IDs
are rejected before mutation. Valid late external cancellation is idempotent
through the old response lifetime; it cannot become a command against a reused
record. Stale internal handles fail before accessing storage.

## 3. Ownership and interfaces

```text
frontends -> bounded immutable preparation -> OrchestrationRunner / ServingSession
                                               |
                                      execution authority
                                       |               |
                               paged request views   live work selection
                              |                          |
                 PMA-owned page arena       participant-local model graphs
                              |                          |
                    TransferEngine <---- exact producer/consumer frontiers
                              |
                 detached publications -> independent response consumers
```

Use deep ownership modules: each interface encapsulates its conservation,
visibility, retirement and failure contract rather than exposing free lists,
reference counters or event edits to model files. Backend adapters implement
that interface without moving policy into callers. Interface depth is judged
by the ownership behavior callers obtain, not by wrapper or class count.
Names below describe required semantics, not installed classes.

| Owner/module | Responsibility and interface |
|---|---|
| `PhysicalMemoryAuthority` | Sole physical admission/reservation/materialization/release ledger; owns typed page-pool resources and their logical grant authority |
| Paged-KV arena | PMA-owned suballocation implementation: finite grants, conservation, write/read/commit visibility and deferred retirement; no independent byte budget |
| Continuation/residency module | One typed request-state and progress contract: suspend/restore manifests, safe phase/resource witness and protected live references; consumes PMA/storage grants, never another ledger |
| Tier directory / persistence module | Versioned replica and effective-use ordering, finite disk-extent ownership and I/O receipts; no independent memory admission |
| `ServingSession` | Request lifecycle, request-local policy and operation ordering; asks the page authority for grants, never allocates KV itself |
| KV kernel interface | Reads/writes typed paged views and live extents; cannot manufacture page ownership |
| Prefix coordinator | Compatibility lookup and retention policy; obtains immutable page/checkpoint leases, not writable aliases |
| `TransferEngine` | Public movement/coherence authority; consumes source/destination leases and exact stream/event dependencies |
| Rank/global orchestration | One topology-wide logical decision and participant-local grants; no remote GPU ownership or nested multi-device compute stage |
| HTTP/other frontend | Immutable input, output credit and external cancellation; no execution controller, page ledger or live GPU-state mirror |

CPU and GPU adapters implement the same ownership semantics. GPU page metadata,
free-page selection, finite grants, progress decisions, live lengths and table mutations are
device-owned after admission. PMA's host-side physical allocation records do
not mirror that device state: they retain the preallocated backing and immutable
geometry. Its device-resident arena is the canonical suballocation authority.
Host admission receipts report acceptance/ownership outcomes, not free-page
counts used to reconstruct allocation decisions.

Preparation consumes an immutable model-policy/template/tokenizer snapshot and
cannot mutate runner sampling, stops or cache state. Prove concurrent-read
safety or use a bounded preparation owner separate from inference submission.
Validate incompatible requests and acquire their storage before copying input
or committing a successful streaming response.

Existing PMA allocation leases retain actual allocations through free. They
are not already a logical-page allocator. Extend the authority explicitly;
relabeling its present `claimAllocation` calls is insufficient. Free lists and
reference counts inside that authority describe ownership within charged
backing, not a second subtraction from device capacity.

## 4. Pool geometry, tier controls and complete physical admission

### 4.1 Logical lifetimes are not residency slots

The resolved plan carries distinct typed limits:

- `S >= 1`: execution-admitted request records, including running, suspending,
  suspended, restoring and execution-retiring generations.
- `Q >= 0`: preparing/queued requests not yet execution-admitted.
- `R >= S + Q`: response lifetimes, including detached delivery after execution
  retirement; total live response leases never exceeds R.
- Per-request context/output bounds, intrinsic prefill chunks/window W and
  immutable model/sampling policy.
- Per-participant/class pool geometry, native page layout and bounded table,
  reference, checkpoint, replica and operation metadata.
- Actual concurrent compute, transfer, persistence and delivery envelopes.

Suspension keeps its original S record and response lease. It does not consume
a new Q credit, free a request identity, or create an unbounded hidden suspended
queue. Compact compute rows and physical frames may serve other records.
Retiring borrowers retain the old generation even if a logical record can
subsequently be reused. Detached deliveries overlap later executions only
within R and their complete physical envelope.

Authored positive logical limits are exact or rejected, never silently clamped.
A byte cap is an upper bound, not a promise to allocate exactly that many usable
payload bytes: native alignment and required metadata are reported separately.
S does not promise that every set of S maximum histories fits concurrently.
Automatic sizing derives from admitted topology and PMA, not development
machine counts.

### 4.2 Maximize the useful resident pool

For each actual physical memory domain, PMA admits the complete non-pool BOM
and then resolves the largest legal pool satisfying the selected caps and
native geometry. The conceptual budget is:

```text
physical memory admitted for this instance
  - weights and prepared/native context ownership
  - activations, graphs, libraries and execution workspaces
  - recurrent, sampler/history, main/shifted MTP and terminal state
  - table/reference/controller metadata, communication and lifecycle storage
  - admitted tier replicas, I/O staging and response/preparation obligations
  = capacity available to the shared resident KV pool
```

This is an explanation of one PMA transaction, not permission for the scheduler
to reproduce the subtraction or invent a percentage reserve. Pool-dependent
metadata must enter the same checked sizing calculation; solve/round to legal
native geometry and validate the complete candidate. GPU `auto` maximizes the
admitted pool after these obligations; an explicit cap bounds it. Increasing a
tier or staging cap cannot silently steal protected execution memory.

Backing and hot metadata are materialized before serving/capture. Assignment
inside that backing performs no native allocation or graph rebinding. Capacity
controls configure an admitted instance; online resize/replacement is not
implied. Changing them requires renewed admission and orderly instance
retirement unless a separately proved resize contract is implemented.

### 4.3 Configuration contract

The following are required typed intents, **not new installed CLI spellings**.
Resolve them through the canonical parser, lossless configuration, MPI/startup
publication and PMA path; no frontend-private defaults or duplicate budget.

| Intent | Supported meaning |
|---|---|
| Resident GPU pool | `auto` or explicit byte cap; permit per-device caps using actual participant identities |
| Host retained tier | `off`, `auto` or explicit cap; bounded per real host/NUMA ownership, not duplicated per rank |
| Disk tier | `off` or enabled with an explicit finite capacity and configured directory |
| UMA placement | Canonical observed physical-memory-domain identity; automatic omission of a redundant retained RAM tier |
| Retention lifetime classes | Code-owned map from each accepted standard request value (§8.4) to an ordered typed class and minimum lifetime; the class for requests without a value; optional operator ceiling on accepted classes |
| Resolved publication | Requested and effective caps, payload versus metadata/staging, physical domains, actual tier routes and refusal reason |

The owner selected the hierarchy and controls, not arbitrary numeric RAM/disk
sizes, worker counts, watermarks, or a universal disk directory. P0 maps existing
`prefix_cache` budget/directory intent into this single resource model and
settles code-owned defaults explicitly; it must not infer a new default from
this document. An old prefix-only VRAM budget is not silently reinterpreted as
the entire live KV pool cap. Remove replaced names with a documented migration,
rather than retaining a second independent prefix-payload allocator.

Disk enablement validates the configured location, ownership, finite quota and
admitted persistence route before readiness. Never silently substitute a system,
network or another volume. The engine owns only its namespace/files and never
purges unrelated files. Disk quota includes metadata, allocation rounding,
pending replacements and old extents still borrowed—not only visible payload.
An impossible configured tier fails admission with a precise diagnostic.
Disabling a tier selects a fully qualified configuration of this same engine,
not an automatic alternate path after failure.

### 4.4 UMA, NUMA and native storage classes

Shared CPU/GPU visibility does not create two independent physical capacities.
On UMA, resident GPU and CPU allocations compete in the same canonical PMA
domain; omit the redundant retained RAM copy. Distinct allocations in that
domain still each cost bytes. Device mappings/aliases of the same allocation
do not duplicate its physical charge. Bounded host I/O, pinning/registration,
transfer descriptors and OS-facing storage staging remain charged even when
the retained RAM tier is off. CPU-only serving uses its RAM pool as residency;
an additional RAM label cannot manufacture another cache capacity.

This requires a real observed physical-domain contract, not a device-name
heuristic, pointer equality, or an assumption that all GPU-accessible host memory
is UMA. If identity cannot be established, report it as unqualified rather than
double-admit capacity. An explicit request for a redundant independent retained
copy on UMA is diagnosed, not silently instantiated. Preserve actual NUMA/rank
ownership, local GPU eligibility and cross-rank transport; shared host or disk
capacity must have one owner, not one full allowance per participant.

Derive page classes from model layers, true TP shards, PP intervals, cache
family and numerical layout. Include K/V, scales/anchors, codebooks,
TurboQuant metadata, alignment and guards. A logical interval can require
several physical classes. Hybrid main and shifted predictor roles stay distinct;
an all-recurrent participant receives no dummy attention pages.

P0/P1 must model AQ8 request-origin basis lifetime, asymmetric V codecs, QFN
compressed/indexer history and raw partial tails, selected-read metadata and
model-owned shifted-MTP origins explicitly. A second page is not a new AQ8
sequence; shifted origin is not universally main minus one. Paging dense K/V
while retaining an unaccounted full-context auxiliary bank is incomplete.

### 4.5 Complete resource vector

PMA charges each memory allocation once: all materialized pool frames, native
metadata/tables, registered-access records and events, request continuation
state, rollback/snapshot banks, graph/work/communication arenas, replica and
staging storage, and preparation/parser/tokenizer/response overlap. Disk extent
leases belong to one storage-capacity authority; it owns no parallel RAM/VRAM
ledger. Progress references these authorities' typed leases, not new balances.

Metadata may cover the bounded context without private context-sized KV per
record. Sharing aliases consume metadata but no duplicate frame; genuinely
distinct replicas and COW destinations are distinct occupied storage. Suspended
live content and idle prefix content can share immutable backing, but live
references forbid ordinary cache disposal. Physical pool bytes remain charged
when frames become reusable; do not report that as returned system memory.

## 5. Admission, conservation and bounded pressure progress

### 5.1 Admission outcomes and promises

The target overcommits **future residency**, never physical allocations. Output
estimates influence scheduling/economy, not safety; the unspecified output limit
remains the actual remaining context unless its public contract is explicitly
changed. No estimator silently caps generated output. For an explicitly larger
output request, the witness prices the smaller of that requested bound and the
remaining legal context, including the model's terminal-token/feed and shifted-
append rules. It does not promise output beyond context or allocate speculative
bytes past the legal envelope. Preserve the existing visible length/context
terminal; this is not a new pressure-triggered output cap.

Preparation first obtains bounded intake ownership (§12). Acceptance then has
one atomic outcome:

1. **Execute-admit:** obtain the S record, original response incarnation,
   compatible pinned state, immediate transaction resources and a valid
   all-resource progress witness for the resulting admitted set.
2. **Queue/defer:** own a Q credit and bounded immutable input/response state,
   without promising resident pages. Spare S records alone do not imply fit.
3. **Refuse:** distinguish an intrinsically impossible declared request from
   current pressure/no proved plan and queue/response exhaustion. Do not label
   heuristic failure as hardware impossibility.

Direct admission with Q = 0 is valid when the full execution transaction fits;
otherwise refuse, not an unbounded hidden wait. Provisional prefix lookups
have bounded metadata/lease lifetimes and are revalidated at attachment.
A queued request cannot indefinitely pin the very cache needed for progress.
Cancellation binds the nonreused external response incarnation before a record
exists and after one is recycled, not just a recyclable numeric request ID.
GPU acceptance/defer/refusal arrives through immutable receipts; the host must
not infer it from downloaded occupancy.

Execution admission does **not** reserve every request's maximum independently
in GPU or RAM. It promises a safe continuation/progress plan under the admitted
transport and delivery contracts, not simultaneous maximal residency. Prefix
retention remains revocable; a live request's sole continuation does not.

### 5.2 Conservation within charged backing

For every real participant and storage class, let:

- P be the materialized number of frames;
- O be unique assigned frames, including private, tentative, shared, cache-only,
  copying and retiring frames;
- G be unconverted finite-operation grants;
- T be additional unconverted simultaneous transient grants not already in G;
- F be unassigned frames, so `P = O + F`.

The authority enforces `G + T <= F`, equivalently `O + G + T <= P`.
References are aliases, not extra O. These are suballocation facts under PMA's
charged backing, not another physical-byte ledger. A progress witness (§5.3)
is a proof about legal future transitions, not an additional fictitious frame
count. Its protected leases/grants are charged exactly once in the applicable
authority. Memory, disk extents, table versions, checkpoints, staging, events,
response/control credit and topology resources have matching finite bounds.

| Transition | Conservation and publication rule |
|---|---|
| Acquire a finite grant | Increase G/T only after all-class/resource feasibility; no frame or readable history yet |
| Assign a destination | Convert exactly one G/T unit to O; burn a fresh incarnation before exposure |
| Share immutable content | O unchanged; acquire bounded references and pin all native referents |
| Seal a private partial tail | Before alias publication, obtain any immediate COW allowance and revalidate the complete progress witness for every affected writer |
| COW / restore / snapshot | Destination is a distinct occupied owner; source/old version remains occupied until actual retirement |
| Commit tentative bytes | Reclassify occupied bytes; do not debit another frame or expose rejected suffixes |
| Truncate / reject / cancel | Withdraw visibility/new work; release unused grants, not still-borrowed occupied frames |
| Last owner/access retirement | O decreases only at proven retirement; return capacity once, to the original eligible grant incarnation or general pool |
| Reused/retired grant | Never credit a new request occupying the old record; duplicate/stale return cannot mint capacity |
| Abort unpublished prepare | Return only new unpublished resources; preserve existing owners; issued versions/nonces never roll back |
| Move between tiers | Debit the destination before copying; credit source only after preservation and its complete access frontier retire |

A lease-bound private-frame return may replenish its original still-live
finite grant; it cannot resurrect a cancelled future-growth promise. Freeing a
request does not imply freeing its frames, and cache unpinning does not imply
eviction. Multi-owner prepare/refusal must preserve its original published
owners and grants. Separately committed eligible-cache reclamation need not be
undone; record it as its own operation, not hidden partial admission.

**Exact-fit COW contract:** a 16-token frame containing nine sealed tokens has
an exporter and two writers that must both obtain private tails. To issue both
destinations concurrently requires three frames: initially O=1,G=2; after the
first grant converts O=2,G=1; after the second O=3,G=0. A two-frame pool must
refuse that simultaneous plan before publication. It may use a different
proved sequential preservation plan, never silently issue the impossible
grants. Cancelling a writer with an outstanding reader leaves O unchanged
until that reader actually retires. This constructed example is a contract
test, not recorded replay.

### 5.3 An executable all-resource completion witness

Checking only the next token is insufficient. Every execution admission and
state-growing transaction must preserve a bounded, authority-validated path by
which the admitted requests can finish without discarding live state. The
normal scheduler may run more useful interleaved work; the witness is its
safety/progress constraint, not an instruction to run serial rows in production.

A witness binds the admitted request generations and declared maxima, all
participant/class geometry, content/table versions, enabled tier routes,
outstanding borrower frontiers, and a finite ordered sequence of:

- retire already-submitted work and eligible idle cache;
- preserve and suspend selected live requests using actual destination,
  checkpoint, staging and persistence leases;
- restore a chosen request into its complete resident working set;
- allow that request to reach its declared bound with protected transaction
  peaks, then retire its private state and releasable references;
- repeat for remaining requests, retaining their current complete continuations.

For each phase, verify simultaneous occupancy, shared-page unions, all COW and
main/shifted speculative outcomes, old/new copy and table overlap, checkpoints,
transfer/control resources and participant agreement. Count only storage that
is truly reclaimable after preservation **and** real retirement. Do not count
the same shared page as independently reclaimable by two victims. A resource
required to evacuate an owner cannot itself be obtainable only by evacuating
that owner. Future phases can reuse proved retired storage; unrelated future
arrivals, hoped-for cache hits/EOS/compression or an unreserved filesystem
allocation are not resources.

This is completion-order reasoning across the hierarchy, not the sum of each
request's independent maximum in a lower tier. While one request grows, the
witness may preserve other requests at their **current** continuation extent.
It must protect the finite resources/right to take its next progress step,
and invalidate/revalidate the witness before any new work can consume those
rights. Decisions and resource transactions belong to the canonical authority;
the scheduler receives an opaque versioned result, not a copied free balance.

A bounded deterministic planner produces a concrete witness or defers/refuses.
Define its search/work bound from admitted membership and geometry. A heuristic
ranking score is not the witness. Before each allowed model transaction,
validate the worst possible resulting state and all concurrent transitions;
obtain all immediate grants atomically. If it would destroy the witness,
perform an already-proved progress step first. Completion of dependencies,
not repeated speculative retry, wakes deferred work.

Safe-order is especially useful when tiers are off/full: protect an order in
which an eligible request can finish and release resources. It is not the main
throughput strategy and cannot make an impossible state safe. With a 32K pool,
two 32K-max requests holding 1K each cannot both be admitted by resident-only
safe-order: either needs 31K but only 30K is free. With a 40K pool and two such
requests holding 4K each, 32K is free and one needs 28K, so an order exists.
These token-equivalent examples exclude sharing/transients; the real witness
uses the complete resource vector. Valid lower-tier preservation can admit
the former pair only if the actual move/restore/completion path fits.

An individual full-attention request's required resident history and transaction
peak must itself fit its execution domain. Disk retention does not implement
streaming attention over nonresident pages or expand that supported context.
Such an execution mode would require its own native correctness/economy proof.

### 5.4 Pressure action, failure and fairness

Trigger preservation before the next legal transaction would consume the
resources needed to evacuate/resume. Prefer dropping eligible low-value idle
retention, reclaiming redundant valid GPU replicas, and preserving valuable
displaced state in admitted RAM/disk. Select a live victim only at a complete
transaction boundary with the entire suspension contract (§11). Pausing a
request without relinquishing frames is not a reclamation action.

Victim value includes actual reclaimable bytes after sharing, existing backing,
restoration/preservation cost, saved computation, remaining work, request age,
latency obligation and prior suspension cost. Protect a chosen progress owner
against repeated displacement until its promised progress point; bounded bypass
and drain prevent an endless stream of hot newcomers from defeating older work.
Bounds are policy-versioned operation counts/credits, not fabricated wall-clock
guarantees about slow disks or clients.

When every tier is full or disabled, reject/defer new work before exhaustion
unless a valid resident completion order remains. Existing accepted work retains
its witness; never discover normal OOM after spending its last escape resources.
Output blockage has bounded cancellable delivery policy and independent wakes,
not an assumption that a client eventually drains. A missing event, corrupt
publication, invalidated sole continuation or indeterminate submitted work is
a fatal invariant/storage failure with precise diagnosis and safe retirement,
not permission to recompute or retry another backend. Hardware/storage failure
is not a successful completion guarantee; do not promise fault tolerance that
the admitted durability contract does not provide.

## 6. Page lifecycle and three visibility contracts

Frame ownership and content visibility are distinct state machines. A frame
incarnation can hold several successive content versions; rolling back and
reappending within the same frame does not make an old reader safe.

```text
Free -> Exclusive destination -> Written content -> Published content
                                      |                    |
                           operation-readable       committed history
                                                           |
                                               immutable shared content
                                                           |
                        close new accesses -> Retiring -> actual reuse
```

Require three typed interfaces, not a generic mutable “page view”:

1. **Exclusive write plan:** original request/content version, exact write range,
   destination owner, retained old readable extent and native format/basis
   referents, and complete prior-access frontier. It exposes no new history.
2. **Operation-readable view:** immutable table snapshot, operation identity,
   exact producer ordering and per-query causal/tentative visibility, with a
   registered access lease. A verifier can read its ordered tentative data
   without publishing it as committed prefix state.
3. **Committed publication:** accepted content/table/visible-length and coherent
   recurrent/terminal checkpoint version. Subsequent ordinary work and prefix
   lookup/export acquire only its valid committed referents.

Reserve → write → operation-readable publication → accepted commit are distinct
transitions. Reserve is not readability; a ready tentative suffix is not
committed history. Per-query masks/lengths cannot expose unwritten or rejected
positions. Every table snapshot retains its payload, AQ8 basis, QFN auxiliary
history and checkpoint dependencies—not just a copied header.

Access registration is atomic against closing the old view. Its lifetime starts
**before enqueue**: a registered but not yet submitted reader remains a borrower.
It must attach exact native completion or prove abandonment before closure can
finish. Enqueue/event ambiguity seals or quarantines the resources; a destructor
cannot assert completion of unknown native work. Join all actual streams/tasks,
not just the last writer or a convenient main-stream event.

A shared/sealed page is immutable while any cache, request, graph, transfer or
archive reader can reach it. Extending a partial shared tail obtains COW
resources and a valid progress witness before publishing the alias/write plan.
Copy only its valid native payload and required metadata, then publish the new
request view; peers retain unchanged bytes. Private append also respects old
readers' exact ranges. Truncate/reject withdraws visibility but does not permit
overwriting a same-frame suffix still borrowed by an old content version.

Request cancellation/finish revokes new submissions, not queued native work.
Zero logical references alone is insufficient for reuse. Frame, content, table,
grant and receipt identities remain generation-bound through all borrowers.
Version/nonce issuance capacity is itself an admitted finite resource. Protect
sufficient issuance for the active requests' bounded transactions, maintenance
and suspension/progress obligations; stop new admissions and drain before it
can run out. Refusing only a later window after accepting a request is too late.
Aborted publication burns versions rather than recycling them; duplicate
external intents must not create unbounded issuance demand.

Stable table banks, registered-access records and retirement queues have bounded
capacity for the real concurrency. Acquire these alongside payload grants.
Publish prepared versions atomically; no consumer sees half an append, partial
K/V mapping or a bank reused under queued work. Exhaustion defers before
submission; it never allocates an extra hot-path bank.

## 7. Native paged access and capture identity

Append, attention, rollback, export/import and diagnostics all consume the same
typed page geometry. Attention traverses a request's logical token order across
arbitrary frame order and attends only its valid tokens. Other requests' pages,
unwritten tails and alignment padding are inaccessible through the view.

CPU kernels must retain useful vectorization, work sharing and physical-core
scaling. CUDA/ROCm kernels use native page indirection in their optimized
attention/append paths, with tensor-core prefill where beneficial as required
by project policy. A full-history gather into a private contiguous scratch
buffer is not the implementation. Bounded tile staging intrinsic to a kernel
is permitted when admitted, live-sized and measured; it cannot reconstruct
fixed partitions or duplicate all retained KV.

Page boundaries must preserve quantization/anchor/codebook semantics. FP32,
FP16/BF16 where supported, quantized KV and TurboQuant each need native layout,
partial-tail, COW, restore and rollback coverage. A convenient floating-format
adapter cannot qualify another format. Preserve fixed arithmetic order across
page-frame permutations, request row compaction, batching and supported live
M/N/K geometry; page size changes need their own numerical qualification.

Captured identity includes pool and table storage addresses, devices, storage
and workspace generations, formats, page geometry, schema/version, immutable
policy scalars and every embedded launch mode. Logical page IDs, valid lengths,
request laws and current table contents are runtime device data, not reasons
to recapture. If an implementation embeds any such value, it must instead
complete the runtime-data implementation before supporting changing requests.

Before freezing irreversible page/layout/catalog choices, prove a small native
vertical path on **both CUDA and ROCm**: device-selected finite grants and live
shape, noncontiguous attention/append, registered-reader overwrite ordering,
large-to-small captured reuse, and an immutable storage completion waking only
its dependent work. Critically, change live work **inside** a complete transaction:
one row reaches EOS while a peer continues, verifier width changes, and routed
expert work becomes known only after routing. An entry selector cannot predict
those later shapes. Specify each vendor's native lowering and what it factors
rather than enumerates; neither inactive grids nor host tickets per layer are
permitted. Measure retained catalog cardinality/bytes, actual internal submitted
grids and every receipt. A device-free model or CPU table test cannot establish this feasibility;
keep unavailable native cells unqualified while the design is being built.

Every launch covers actual live rows, page spans, tiles or experts. No
capacity-sized grids with wholly inactive blocks, no empty compute families,
no padding communications, no host live-count readback and no per-request row
replay disguised as batching. Device-selected retained branches must cover the
supported shape space economically; catalog size and capture bytes are proof
obligations, not assumptions.

## 8. Shared prefix caching and computation value

### 8.1 Compatibility and real sharing

Prefix identity includes exact token ancestry, model/weights and graph policy,
positions/RoPE, cache family/layout/precision/codebook, producer numerical
boundary/version, required placement history, checkpoint semantics and an
authenticated trust domain. A hash accelerates lookup; exact ancestry or an
equivalent collision-resolving proof is required before sharing. Trust comes
from admitted authentication/isolation policy, not an untrusted caller-chosen
cache key. Same tokens alone do not prove identical KV across different
prefill/restore boundaries.

Resident hits attach actual immutable pages. A nonresident archive imports its
valid native content into exclusive pooled destinations and publishes readiness.
Concurrent imports join one incarnation or retire their own unpublished
candidate; they do not double-spend capacity. A provisional queued lookup becomes
a hit only when compatibility and the exact content incarnation are pinned at
attachment. Evicting retention cannot invalidate existing live aliases.

Versioned archives contain only explicitly classified portable sections.
Validate shape, offsets, lengths, checksum, native format/basis and numerical
provenance before payload access. Never import an opaque model blob that rewinds
domain-wide expert placement, epochs or histograms. QFN append-only compressed
payload versus endpoint/rollback state and model-specific shifted origins need
their own typed sections. Incompatible archives are explicitly rejected, not
interpreted as another format.

A **new request** reconstructs its own sampler/history, seed, penalties/DRY,
output and depth learner. It adopts only compatible KV and required coherent
recurrent/hidden/logit boundary state; it does not inherit another request's
sampled condition. A **resumed same request** instead restores its complete
continuation (§11), including its own prior controller state. These operations
must not share an ambiguous “restore state” interface.

Full-hit terminal logits, the model's last-token rule and recurrent checkpoint
must describe one usable logical boundary. GDN/short-conv snapshots can be shared
immutably; mutable active recurrence is request-owned. Never rewind a live peer
or replay missing history to fabricate a checkpoint. Across participants, the
selected boundary must actually be restorable everywhere (§13), not merely the
minimum token count or an equal epoch label.

### 8.2 Retention value and no-copy hits

Retain/restore is a normal serving mechanism, not emergency spill. Prefer GPU
for active/imminently reusable content; RAM for valuable displaced state on
discrete systems; disk for valuable less latency-sensitive content. Do not
persist cold garbage just to maximize occupancy or force every page through
all tiers. A valid unchanged lower-tier replica makes GPU reclamation cheaper.

Retention policy combines recency, frequency, reuse likelihood, avoided compute,
resident footprint and measured restore critical-path cost. Account for copy
contention, decompression/validation, required staging and overlap, not nominal
bandwidth alone. Unknown cost is reported as unknown; it is not evidence that
large prefixes should be recomputed. Policy scores never override live-state
protection or accounting.

Define **effective use** as successful compatible attachment/adoption, not a
provisional lookup or every attention read. Bind that event to the sealed lookup,
content version and replica incarnations. The retention owner assigns/accepts
an ordered use sequence and applies or watermarks all accepted touches before
an eviction decision. No-copy GPU hits refresh valid backing usefulness through
a bounded immutable adoption receipt; they do not download page tables.

An attachment whose touch has not been acknowledged retains a bounded retention
pin/fence so eviction cannot race ahead of that accepted use. Once the ordering
point is acknowledged, ordinary recency—not a permanent pin—governs idle
retention. Duplicate touches are idempotent; obsolete touches cannot refresh a
replacement under the same key. Queue capacity and delayed receipts are part
of admission; backpressure must not silently drop effective-use ordering.
This ordering is policy correctness; it does not replace overwrite/read fences.

### 8.3 Actual observations

Publish actual lookup, selected usable boundary, matched logical blocks/tokens,
restored coverage, source tier(s), checkpoint identity, bypass/discard reason,
adoption and harvest outcomes. Logical block coverage need not equal physical
record count on all-recurrent participants. Multi-tier restoration is not
mislabelled simply “RAM.”

Carry original observations through request summaries and HTTP. Neither a
constant zero nor a ceiling inferred from restored token count proves actual
matched blocks. Use clamped/partial and mixed-participant cases where inference
cannot reproduce the lookup result. Passive observation is never another
admission ledger or live GPU probe.

### 8.4 Client retention intent and tier ordering

Clients express retention only through the standard fields of an API surface
that Llaminar actually serves. No vendor extension, header or private endpoint
is added for this purpose.

| Served API | Standard field | Meaning here |
|---|---|---|
| Chat Completions | `prompt_cache_key` | Cache namespace within the authenticated trust domain |
| Chat Completions | `prompt_cache_retention` (`in_memory`, `24h`) | Requested minimum-lifetime class |
| Responses (only if served) | `prompt_cache_options.ttl`, `prompt_cache_breakpoint`, `prompt_cache_options.prewarm` | Class, explicit cache boundary, prefill-only cache write |
| Anthropic Messages (only if served) | `cache_control` with `ttl`, `max_tokens: 0` | Class at a block boundary, prefill-only cache write |

Fields of an unserved API are not interpreted. A recognized field with an
unsupported value fails the request with a precise `invalid_request_error`; it
is never silently mapped to another class. An absent value selects the
code-owned default class (§4.3). P0 settles the value-to-class map, minimum
lifetimes and default; this design fixes only the semantics below.

**Namespace is not trust.** A cache key partitions retention and accounting
and keeps related requests on one owner. It never makes incompatible content
shareable or grants access across trust domains (§8.1). Distinct live
namespaces and per-entry intent records are bounded admitted metadata,
priced in the BOM; no unbounded key map exists.

**Lifetime is a minimum, not a pin.** An entry's class expires at its minimum
lifetime, measured from the start of the request that wrote or last effectively
used it. Effective use (§8.2) refreshes the expiry under the same ordered
receipt; a provisional lookup does not. An expired entry keeps its value score
but loses its class. Intent never protects idle content against live demand,
never adds capacity and never authorizes an allocation: running requests and
protected live continuations (§11) always outrank all idle retention, and PMA
admits every byte.

**Victim order.** Within one physical tier, revocable idle pages are ranked by:

1. unexpired lifetime class, longest first;
2. the §8.2 value score (recency, frequency, reuse likelihood, avoided compute,
   footprint and measured restore cost);
3. a stable tie-break by older effective-use sequence.

Pressure in a tier selects the lowest-ranked pages. They move to the next
enabled tier with an admitted route (GPU → RAM → disk); from the lowest enabled
tier they are discarded. Move only the deficit, rounded to native page and
extent granularity. Cascade into the next tier only when that tier actually
overflows. Never move content merely because a lower tier has free space.

**Tail-first partial demotion.** A hit always starts at token zero, so a
partially demoted prefix keeps its leading pages in the faster tier and demotes
its tail. A later hit attaches the resident head and imports the tail before
any dependent read (§§8.1, 11.2). A hybrid prefix is usable at a shorter
resident boundary only where a coherent recurrent/terminal checkpoint exists
there; otherwise the hit imports the tail.

**Shared pages.** A page referenced by several entries takes the highest
unexpired class of any referencing entry, and moves once. Demoting one entry
never moves a page that another entry still ranks higher.

**Prewarm** writes a prefix cache entry without sampling. It is offered only
where the served API defines it; Chat Completions has no such field, so no
zero-token or other workaround is accepted there. A prewarm is an ordinary
request for S/Q/R, PMA, fairness and response custody, returns standard
usage, and has no priority over live work.

**Reporting** uses only fields the served API defines, from the actual §8.3
observations: `usage.prompt_tokens_details.cached_tokens` on Chat Completions,
and write-token fields only where that API has them. The engine additionally
reports per-tier occupancy by class, demotions, discards, expiries and
refreshes (§14).

Worked example (illustrative only). Units are tokens; GPU pool 1.1M, RAM 2M,
disk 50M. Each request's prefix becomes idle after it finishes. The class
labels 1h > 30m > no value are illustrative; P0 maps real request values.

| Arrival | Needed on GPU | Moves | GPU | RAM | Disk |
|---|---|---|---|---|---|
| A, 30m | 1.0M | none | A 1.0 | — | — |
| B, 1h | 1.0M | A tail 0.9 → RAM | A 0.1, B 1.0 | A 0.9 | — |
| C, 30m | 1.0M | A head 0.1, then B tail 0.9 → RAM | B 0.1, C 1.0 | A 1.0, B 0.9 | — |
| D, no value | 1.0M | C → RAM; RAM overflows by 0.9, so A tail 0.9 → disk | B 0.1, D 1.0 | A 0.1, B 0.9, C 1.0 | A 0.9 |

B stays above C although it is older, because its class is longer. A and C
share a class, so value decides; with equal use, A is less recent. Only A's
0.9 tail moves because that is the RAM deficit; A's head stays in RAM and C is
not touched. If B's class had expired, B would compete with A and C on value
alone. If A were still running when D arrived, A could not be demoted; D would
wait or a separately justified suspension (§11) would apply.

## 9. Request state, MTP and numerical behavior

Paging separates attention history from request-local mutable state; it does
not eliminate that state. Classify every field:

| Class | Examples | Lifetime |
|---|---|---|
| Request | Sampler/history/DRY, stop/think transducer, GDN/conv, MTP learner, terminal logits/hidden, pending correction/sidecar | Request generation and exact borrowers |
| Shared immutable content | Sealed KV pages, compatible recurrent/terminal checkpoints | Last content reference and access frontier |
| Transaction | Compact work rows, temporary speculative chain, scratch and collective operands | Exact submitted operation frontier |
| Domain | Prepared weights, durable expert placement, communicators and maintenance authority | Admitted instance/domain; never restored from a request prefix |

Terminal prefill logits and hidden state survive intervening requests. A typed
handoff distinguishes unsampled logits, an already sampled/recorded condition,
and pending/completed predictor work. Replace runner-global ready/pending flags
and slot-address arithmetic with generation-qualified ownership. Price actual
FP32 vocabulary rows, hidden state, history and events; no per-request state is
hidden in nominal KV savings.

MTP uses explicit main and shifted-predictor page roles. Proposals/verification
reserve bounded tentative pages before work. Accepted history and visible
lengths commit together; rejection retires unpublished suffixes after their
read/write frontiers. COW protects shared prefixes during speculative append,
truncate and correction. A correction updates KV, history and output once;
reusing published token output as a host shadow of device condition is invalid.
Per-request depths, seeds, laws and learner observations remain independent in
one grouped verifier operation. Suspension preserves that same request state;
resumption does not initialize a fresh learner or reinterpret shifted origins.
A full-request manifest and its replay include all relevant auxiliary histories.

For the same admitted execution/numerical policy, resolved seed, prefix boundary,
external control trace and placement epochs, a request must produce the same
committed bytes alone or co-scheduled. Physical frame assignment, page reuse,
request-record index, batch row and queue timing do not alter its arithmetic,
RNG coordinates or learner. Prefix-hit versus cold-prefill at different restore
boundaries remains a separately proved numerical question, not an excuse to
change the comparison boundary after a failure.

The stochastic key remains `(resolved seed, logical output position, draw
purpose)`. Seed zero resolves once; explicit seeds are preserved. Sampler laws
are per-request data, including model-default penalties and supported DRY,
not graph-specialization/cohort keys used to serialize incompatible laws.

Prefill chunk/checkpoint grids and positive output-window quantum W are
request/configuration-intrinsic. The next window ends at the next W-aligned
committed-output boundary, bounded by the real response limit. Co-scheduling,
page pressure and output timing may defer an operation but cannot shorten its
window, change its prefill chunk or clip MTP learning using spare batch space.
Placement switches and external cancellation have explicit logical trace
positions. LLEP assignment is request-local or its eligible replicas must be
byte-equivalent under the selected arithmetic.

Reasoning phase and budget enforcement have one execution authority. The
versioned token-piece matcher handles alternate delimiter tokenizations and
markers spanning pieces/publications. It publishes bounded close/removal and
forced-injection coordinates for presentation. Host text parsing cannot force
model tokens. The declared rule excludes the marker-closing token from sampled
reasoning counts; forced tokens update state once without consuming an ordinary
sample, and clipped/forced speculative transactions skip depth learning under
the same solo/batch policy. Any change from existing frontend semantics is
versioned and qualified, never silently rebaselined.

## 10. GPU control and finite complete transactions

The host owns immutable request preparation, external lifetime intents,
destination-backed output grants and delivery. The GPU execution authority
owns active phases, page grants/tables, lengths, sampling, MTP verification,
stop/think decisions and selection. No mutable host copy is maintained.

| Permitted exchange | Meaning |
|---|---|
| Initial admission/policy ingress | Immutable prompt, generation-qualified binding, policy and admitted envelope; acceptance may defer under page pressure, not a host-authored live table update |
| Cancellation ingress | External intent for the original request lifetime; never immediate free |
| Window grant ingress | Unique permission backed by response destination storage; no positions, live counts, depths or selected rows |
| Terminal publication | Committed live output, bounded framing/accounting, typed outcomes and ownership receipts |
| HIP scheduler ticket | Fixed-size authenticated identity naming an already retained complete transaction; host submits only that named branch |
| Explicit KV/prefix tier I/O | TransferEngine-owned storage movement and identity-bound completion; not a general GPU state readback channel |

Page allocation is not an added host ingress kind. A storage receipt cannot be
repurposed to choose model work from downloaded page/row occupancy. Source
sanitizers enforce the closed interfaces and exact non-null streams. P0 must
bind the recorded G1 decision to these exact permitted exchanges and reconcile
canonical architecture/source-policy declarations before installing the changed
serving path. An adoption date, stub or this documentation rewrite alone is not
authorization for a broader host boundary.

Operation kinds remain typed: lifecycle/selection, one canonical prefill chunk,
complete decode window, and terminal idle. Lifecycle-only progress never calls
an empty generation commit. Initially allow one model transaction in flight per
domain and per request; independent disjoint lifecycle I/O may overlap. Two
publication banks do not authorize two speculative model submissions.

Conditional-capable execution retains selection and complete chosen work in
its parent. HIP uses selector-after-ingress, validates only host-known instance,
ingress, binding-snapshot and retained-catalog identities, then submits the
named complete operation. Every selector/branch ticket binds a single operation
ordinal, sealed ingress/binding snapshot, catalog and workspace incarnation
through branch completion. Duplicate/stale tickets fail even for successive
prefill chunks using the same output grant. The selector receipt is an additional observation;
native MTP branch tickets may add more. Enumerate and measure every receipt,
graph and retained byte rather than claiming one wait per window. Neither
backend may host-step layers, poll live counts, synchronize whole streams,
recapture, allocate or rebind during execution.

Ingress is sealed for the operation using it. Request/table generations are
validated on device before access. Terminal/retirement receipts may refer to
bindings still retiring in the immutable ingress snapshot; the host cannot
replace that snapshot with its latest set. GPU readiness gates only the work
that needs the associated pages, not unrelated ready requests. Residency policy
and pressure-victim selection remain in that execution authority. Host tier
workers service immutable I/O commands and return authenticated completion;
they do not interpret suspended model state to make execution decisions.

## 11. First-class tiering and complete suspension

These are required target contracts, not claims that current prefix archival or
the CPU page prototype already implements live continuation. Suspension and
pressure-driven residency changes are part of the main implementation sequence.

### 11.1 Request lifetime versus residency

```text
Queued -> Running -> ExecutionTerminal -> DetachedDelivery -> Released
             |
             +-> Suspending -> Suspended -> Restoring -> Running
```

Suspension begins only at a complete transaction boundary, after accepted
publication/rollback and with new model submissions revoked. A typed continuation
manifest binds the serving instance, original request generation, immutable
model/policy, logical position and content versions, original topology/roles,
and **every** state item needed to resume:

- committed attention and auxiliary history, scales/anchors and native layout;
- GDN/conv and other recurrent state plus the matching checkpoint;
- main/shifted MTP history, learner, sidecar/condition and terminal handoff;
- RNG resolution/coordinates, sampling/penalty/DRY history, stop/think state,
  forced/correction state and accepted-output accounting;
- outstanding output-grant/publication identities and exact ownership receipts.

A legal suspension point has no unresolved speculative model transaction.
Any allowed pending handoff is represented explicitly and retained; unsupported
states cannot be silently dropped. Preserve history, policy and identity, not
merely token IDs or a prefix archive. Response delivery may continue separately;
suspension neither redelivers output nor resets the response lifetime.

Prepare all backing, metadata and control resources before freezing state.
The sole continuation is protected from idle-cache eviction. GPU frames become
reclaimable only after the preservation receipt and all relevant old accesses
retire. If a clean, authenticated backing replica already contains the identical
version, reuse it instead of rewriting it. Dirty mutable state needs a new
coherent snapshot before its old physical owner can be released.

Restoration obtains the complete resident working set and next-transaction
envelope, imports/pins the coherent continuation and publishes all-participant
readiness before model execution. The request keeps its original numerical
coordinates, learner, grant and terminal arbitration. Repeated suspend/restore
must be observationally equivalent to the same request without movement under
the declared numerical/control trace. Suspension is not a fresh admission or
permission to sample its last logits twice.

### 11.2 Prior readers, overwrite, subsequent consumers

For every destination content/incarnation:

```text
close admission of old-view accesses
  -> retire all registered/submitted old readers and writers
  -> exclusive destination / exact prior-access frontier
  -> restore, COW, clear or overwrite on the actual transfer stream
  -> generation-bound completion and coherent table/content publication
  -> subsequent consumers wait on that event on their actual streams
```

Both overwrite edges are necessary. A copy-stream FIFO does not order another
stream's queued reader; disk latency is not synchronization. Graph, verifier,
sidecar, collective and exporter accesses survive logical cancellation. Join
every relevant event, including registered-before-enqueue consumers; retain
event incarnations until all waiters finish.

`TransferEngine` consumes a destination-overwrite lease plus pinned
source-ready ownership and returns the exact completion publication. No caller
fabricates coherence with dirty flags or guesses a stream. Delayed completion
cannot attach bytes to a newer frame/table/request. On failure withdraw
unpublished destinations and retain all native borrowers; indeterminate native
work seals/quarantines, never returns usable credit.

GPU-chosen frame addresses remain device data. Declared pack/scatter operations
use device page tables and bounded admitted block/tile staging; host persistence
consumes identity-bound staging extents, not a downloaded live frame enumeration.
Storage commands/receipts expose only the immutable I/O identity and extent
needed by that operation. They cannot become a host row/depth/phase selector.
Do not gather a full history as an attention adapter.

### 11.3 Storage ownership, durability and bounded overlap

Pre-admit copy lanes, events, registered-access records, table/snapshot banks,
staging and persistence queues. RAM retained capacity and I/O staging are
different resources: turning the former off cannot remove the latter's charge.
Reuse a valid source replica where possible; otherwise persistence retains its
source/leased staging until real completion. Bounded chunked storage I/O may
overlap independent ready model work without changing request-intrinsic prefill
or decode transaction boundaries.

Disk extents have checked finite quota, exact namespace/file identity, generation,
content integrity and replacement/retirement ownership. Account for old and new
extents, in-flight writes, metadata and allocation rounding concurrently.
The configured directory selects the actual storage route. Atomic publication
of complete valid content precedes exposing it to readers; preserve the old
valid replica until replacement is acknowledged. A required sole disk
continuation requires the admitted durable publication, including the specified
native write/fsync completion—not merely queue acceptance or a filename.
Partial/corrupt content is never a valid continuation.

A process restart is not transparent continuation of old live requests unless
a separate crash-recovery/authentication contract is qualified. Durable reusable
prefixes and live-request resumption are different promises. Engine-owned
cleanup/purge cannot unlink storage borrowed by an active request or another
instance, and must never scan/delete unrelated files.

All host buffers, pinned registrations and I/O metadata stay under PMA. Disk
space is owned by the typed storage authority; a logical quota alone is not
proof that a required future extent is physically available. A progress witness
must hold actual admitted capacity/extent rights for its preservation steps.
No optimistic compression ratio or assumed filesystem free space supplies a
safety guarantee.

If bounded staging cannot be obtained, defer at the still-safe checkpoint;
never allocate secretly, miss it or replay backwards. Required storage
progress uses reserved control capacity independent of output grants. A slow
restore blocks only dependent model work. Cancellation revokes activation but
not submitted DMA/filesystem access; keep source, destination, staging and
receipts alive until actual retirement. Shutdown/cancellable transport follows
the same ownership; a timeout alone releases nothing.

Native I/O faults report precise failure, invalidate uncommitted destinations
and preserve proven replicas/borrowers. Do not automatically change tier/backend,
reduce context, or recompute to hide a missing or failed implementation.
All-full normal pressure is handled by the prior progress witness; an external
storage failure is an explicit failed request/session according to its affected
ownership, not a falsely successful cache miss.

### 11.4 Replica directory and progress protection

The authoritative directory distinguishes immutable content version from each
physical replica incarnation: valid clean copies, private dirty state, pending
copies, invalid/retiring copies and durable receipts. A stale disk/RAM copy cannot
replace newer committed state. Bind content, checkpoint and terminal metadata
together; partial mixtures are invalid.

Idle reusable prefixes may lose every replica under value/quota policy.
A live suspended continuation may not. Shared backing remains live-protected
until its last live reference and actual access retire, even if its cache key
is evicted. Protected progress resources are explicit typed leases, not an
anonymous emergency reserve or full-context reservation copied into another
tier. Validate real GPU/RAM/disk routes and UMA memory→disk routes separately;
alternative RAM and disk experiments do not prove a composed promotion chain.

## 12. Publication, liveness and terminal outcomes

Request state and KV retire independently from response delivery:

```text
execution terminal staging -> event-ordered transfer -> bounded host demux
    -> detached immutable request publications -> framer -> socket/result sink
```

GPU staging retires after its exact copy/source-consumer edge. Host batch
staging retires after demux's CPU readers. Detached publications do not pin
shared staging, KV pages or an execution record. A response lease can outlive
and overlap several uses of that record. Transfer its ownership through the
actual HTTP provider, retirement callback and final result/socket borrower.
Producer-thread exit is not response retirement; a later transport error cannot
release bytes still borrowed by an outstanding write.

One output grant reserves the maximum complete request-local window before
eligible model work. It binds serving instance, request generation, non-reused
grant ID, immutable window policy and destination ownership. At most one is
outstanding per binding. A grant can remain unspent across prefill chunks;
selection can defer it but cannot spend it twice or resize the window. Only its
terminal completion or retirement receipt releases unused reservation or
transfers committed payload to detached publications. Credit return permits a
new grant, never a host write to device phase, position or depth.

Credit covers simultaneous producer, queued and consumer
payloads, escaping/framing, retained UTF-8/reasoning/tool state, metadata and
non-streaming accumulation. Popping a queue transfers its byte ownership to
the consumer; it does not release credit while `sink.write` still borrows it.
Lazy fragmentation is necessary but does not bound arbitrary metadata or all
producer transients. Remove admit-arbitrarily-large-when-empty policies from
bounded production channels. Derive a maximum from admitted geometry or reject
unsupported input before success; preserve semantic and framing boundaries.

Non-streaming results need an admitted complete-response bound or a checked
accumulation cap with defined failure semantics. A fixed nominal price per
response is not proof. Preparation/parser/tokenizer buffers and simultaneous
copies belong in the same complete host-memory account. Bound HTTP body/intake
and parsing **before** their copies, using response admission or a separately
admitted bounded intake owner; rejecting a request afterward does not undo its
peak. Price final serialization and the simultaneously retained HTTP body,
repeated fragment metadata and optional access tracing. A trace option cannot
introduce unbounded accumulation. A promised maximum output window must actually
fit its complete producer/framer/consumer envelope.

Lifecycle operations have reserved control/terminal storage independent of
output grants. All-output-blocked cancellation and retirement still progress.
Idle is valid only with a concrete pending wake source or genuinely absent
eligible work; it cannot discard an outstanding grant. Delivery deadlines
resolve abandoned consumers through cancellable transport, cleanup and an
independent lifecycle wake, without inventing model deadlines or killing
arbitrary kernels. A deadline alone does not retire an existing borrower.

Execution outcome and delivery outcome are distinct. The first execution
terminal committed in operation order wins; a later disconnect changes only
delivery. Completed peers keep their real outcomes if another request's reset
fails. A shared execution invariant failure seals the session; service health
and inference fail closed until supervisor restart. Do not continue on an
unproven subset of state.

## 13. Cache-aware scheduling and topology-wide execution

### 13.1 Useful work with bounded fairness

Use bounded queue lookahead to compare compatible prefix reuse, actual avoided
prefill, restoration readiness/cost, decode service, long-context attention,
routing cost and age. Coordinate requests sharing missing prefix work so they
can consume a published prefix rather than perform duplicate cold prefills;
do not attach incomplete speculative state. Preparing an attractive restore
requires real grants and a witness, not unbounded speculative prefetch.

Interleave request-intrinsic cold-prefill chunks with decode/ready work. Pressure
or co-scheduling can defer a chunk/window but cannot shrink it, reduce output,
clip learner observations or silently lower authored MTP depth. Request-local
dynamic MTP retains its admitted policy and complete speculative envelope.
Aggregated/fused prefill is a separately qualified optimization, not a substitute
for native paged execution or a reason to serialize differing sampler laws.

Aging uses bounded/saturating operation counts and stable admission-order ties.
Specify finite lookahead, maximum bypass and a drain/protected-progress rule
so a long cold request or suspended victim cannot be bypassed forever by hot
short arrivals. Waiting on exact I/O/output dependencies has an independent
wake and applicable cancellation/failure contract; it is not eligible compute.
Do not promise a wall-clock latency bound from operation fairness alone.
Record suspension count, movement cost and useful progress to detect thrashing;
protect chosen progress points rather than evicting the same request repeatedly.

The completion witness constrains legal transitions; value ranking chooses
among them. Ordinary grouped decode remains genuinely grouped. A safe-order
witness is not a production serial-row replay algorithm and cannot excuse
uneconomical grouped kernels.

### 13.2 Prepare, agree, commit or seal

The admitted continuation authority chooses one logical decision for the exact
operation participant set. Each participant retains only its local shards;
page IDs need not match. Proposals bind request generation, decision/epoch,
content and numerical versions, **usable** prefix/terminal boundary, native
resource classes, prepared grants and readiness/retirement dependencies.
Equal epoch numbers or successful local “fits” votes do not establish compatible
state.

Prepare resources without exposing them. Agree the common usable boundary and
all-owner decision, then commit coherently or abort every unpublished prepare.
Model consumers cannot run while a required participant can still refuse.
Apply this protocol to initial attachment, subsequent tentative grants,
accept/reject publication, preservation, suspension and restoration—not only
initial admission.

A known pre-publication refusal releases only new resources. Indeterminate or
post-publication failure seals/drains the affected session; independent local
retry/rollback cannot undo a view already consumed elsewhere. Exact decision
identity authenticates late/doubled messages. Empty and expert-only followers
exchange only necessary protocol metadata, but still make required progress.
No host-selected GPU row map or downloaded mutable state substitutes for native
agreement.

Preserve original MPI ranks and NUMA ownership. Native GPU collectives remain
within real homogeneous rank-local domains; CPU/cross-rank movement uses existing
MPI/shmemspin contracts. Only the declared continuation samples/controls.
Expert-only and all-recurrent participants acquire no dummy attention history.

Communicate live rows/spans plus protocol metadata, never table capacity,
unowned shards or inactive padding. PP has an explicit microbatch schedule,
common publication points and exclusive simultaneously active arenas; one global
in-flight transaction does not prove pipeline utilization. Prove singleton,
odd/non-power-of-two and larger memberships, unequal widths, more than two
stages and valid CPU/GPU compositions. Local hardware limits evidence, not the
typed topology model.

## 14. Observability and qualification

Report materialized pool/metadata bytes separately from occupied frames and
unconverted finite grants. Define disjoint frame ownership categories plus
orthogonal alias counts: adding shared references must not multiply physical
occupancy. Report valid/dirty/retiring replicas per physical tier, protected
live continuation versus revocable idle retention, actual reclaimable bytes,
COW/import/export bytes, staging and disk-extent high water, progress-witness
outcomes, suspension/restore cycles and admission reasons. These are immutable
permitted observations, not live GPU probes or a copied allocation ledger.
A complete physical BOM and actual custody remain separate proof obligations.

Per-request observations include actual matched/restored coverage and tiers,
avoided prefill, committed output, draft/verifier work, suspension waits and
terminal/delivery outcomes. Shared time/bytes cannot be charged in full to every
request and then summed. Record actual submitted launch dimensions, transfer
and collective extents, graph reuse and forbidden-path counters. Stats epoch
reset cannot clear caches, cancel requests or accept old terminal counters into
the new epoch. A path counter alone proves neither correctness nor economy.

These are qualification contracts, not claims that named tests already exist:

| Contract | Required adversarial evidence |
|---|---|
| Pool sizing / controls | Auto maximization after complete BOM; explicit caps and alignment; independent S/context; limiting shard; all-class refusal; restart/re-admission on changed capacity |
| UMA / NUMA / tiers | Shared physical-domain single accounting, genuine replicas charged, no redundant UMA retained copy, bounded staging with RAM off, actual locality and per-host quota ownership |
| Conservation | Every debit/convert/return through shared partial tails, COW, speculation, cancellation, retired/reused grants and actual final borrower retirement; exact-fit negative controls |
| Progress | All live requests need growth; tiers off/full; finite staging/event/table shortage; safe-order and displacement witnesses; no circular evacuation dependency or independent maxima simply moved into RAM |
| Admission / fairness | Pressure with spare records, Q=0 direct admit, queue full, stale pre-record cancel, bounded bypass/drain, cold long requests versus hot arrivals, repeated suspension and actual wakes |
| Sharing / provenance | Identical physical frames, divergent tails, surviving aliases, deduplicated import, AQ8 second page, QFN selected history, model-specific shifted origin, incompatible/trust/archive rejection before payload access |
| Three visibility states | Unwritten/tentative/committed distinction, registered-not-enqueued readers, late borrower refusal, rollback/reappend within one frame, abort/version burn, table/native referent lifetime |
| Native arithmetic | Noncontiguous/permuted frames; partial pages; all supported formats/codebooks, live M/N/K and CPU ISA/thread geometry; optimized ordinary/MTP paths |
| Continuation | Full recurrent/MTP/sampler/DRY/handoff/grant state; same-request restore versus new-prefix attachment; zero/some/all acceptance and exactly-once correction; repeated movement without duplicate output |
| Tier ordering / recency | Prior-read → overwrite → subsequent-read edges with missing-edge controls; cancelled/failed I/O and stale receipts; no-copy hit refresh followed by unpin and forced competing eviction; delayed/duplicate/obsolete touches |
| Storage | Configured directory and quota, rounded/metadata/replacement peaks, full/unavailable disk, partial/corrupt writes and exact durability receipts, protected live continuation, engine-only cleanup |
| Retention intent | The §8.4 worked trace through the real authority; class expiry and refresh ordering; shared pages under mixed classes; tail-first partial hits; deficit-only cascade; live demand overriding every class; unsupported values rejected; key isolation across trust domains; bounded namespace metadata |
| Capture | Early CUDA and ROCm feasibility, then complete captured serving/tier policy; large-to-small replay, exact live grids/traffic, bounded catalog/receipts, no recapture/readbacks/host state shadow |
| Response custody | Oversized event with prior slow consumer; last blocked write after producer exit, R refill attempts, concurrent rejected parses, stopped connected sink, non-streaming body/serialization and cleanup |
| Topology | Incompatible successful local proposals; failure before/after commit; empty/expert followers; singleton/odd/larger/unequal TP/PP/EP, more than two PP stages and valid CPU/GPU compositions |

Replay recorded inputs and initial state through the actual production
controller/kernel, preserving policy/version, numerical boundary, geometry,
formats, ordering, sampler laws, page/table/content identities, grants and
completions. Expected observations include decisions, state transitions and
outputs. Reject incomplete/incompatible traces. Constructed accounting examples
and repeated synthetic fixtures are contract/lifecycle tests, not recorded replay.
Every changed controller/kernel and discovered production defect receives a
focused explicit `ProductionTestPreflight` registration. Unit stays fast and
device-free; native kernels/capture run in explicit integration suites.

Complete Unit/preflight, source-to-loaded-code custody, native execution and
canonical multi-prompt frozen-history final-logit panels remain separate gates.
Focused passes or old fixed-slot traces do not certify paging/tiering. Large
certified corpora belong in `Llaminar/corpora`.

Economy uses matched dedicated-host Release workloads with production defaults
intact and separately identified controls. Include capacity one and mixed
concurrency, long/short contexts, prefix fanout and misses, bursts and sustained
offered load, streaming/non-streaming, slow clients, cancellation/refill,
pressure, no-copy backing hits, actual RAM/disk restore and UMA routes.

Measure queue-inclusive TTFT, inter-token gaps, completion/tail latency, useful
output tokens/s and SLO goodput together with unfinished work, starvation,
avoided prefill, tier I/O, movement/recompute cost, thrash and resource high water.
A completed-only percentile, busy GPU or large batch with a growing queue is not
a win. Compare restoration's matched end-to-end critical path with the actual
avoided prefill; total cache size or isolated bandwidth cannot choose the policy.
No superiority over another engine is assumed.

Profile every new CPU/CUDA/ROCm kernel, physical-core/NUMA scaling, launch
geometry and transport extent. Compile every shipping GPU target with spill
guards; retain isolated candidate profiling separately from canonical timing.
Correctness and economy must pass simultaneously.

## 15. Replacement and completion rule

Refactor or replace existing slot-addressed cache, graph-binding, prefix,
controller, estimator and test machinery wherever necessary. Accepted old
source is reusable only where its contract survives; acceptance does not freeze
an unsuitable interface. Retain useful numerical or lifecycle oracles while
replacing tests that assert the retired physical layout.

The finished implementation has one paged KV path for supported serving,
including S = 1, and one authority for each live value and physical byte.
Fixed full-context request partitions, their capacity arithmetic, copy-per-hit
resident-prefix restoration and alternate serving launchers retire at cutover.
Historical source remains in Git and sealed evidence, not as a runtime fallback.

The implementation plan gives the dependency order and exit evidence. No gate
is closed by rewriting this design; no new runtime support, capacity increase,
PR readiness, release or image publication is asserted here.
