# Continuous batching with pooled, paged KV

**Revision 4 — replacement design, 2026-11-25.** The owner-directed target is
pooled, paged KV from the first replacement implementation. This revision
supersedes revision 3 in full, including its fixed-slot/full-context first mode
and its deferral of live prefix sharing. Logical request records remain;
dedicated full-context KV partitions do not. Full refactoring is permitted.

This is a detailed design, not a dashboard or a capability certificate. The
companion [implementation plan](plan_continuous_batching_implementation_20261125.md)
owns sequencing, current gaps and evidence requirements. Earlier
[research](research_continuous_batching_20261105.md) and
[reviews](review_continuous_batching_20261105.md) retain their historical,
source-pinned meaning; their fixed-slot recommendations no longer select the
target architecture. Historical review seals and failed attempts remain intact.

The previously adopted serving-boundary decisions remain the starting policy:
immutable ingress, request-local bounded generation windows, authenticated
terminal/ticket receipts, and separate execution/response lifetimes. Paging
adds storage ownership, not another host permission to inspect or control GPU
execution. Project maxims, native correctness/economy gates and release policy
still apply. Documentation publication neither implements paging nor qualifies
an increased serving capacity.

## 1. Decision and scope

Independent requests share model work and a common pool of physical KV pages.
A request's logical token positions map to page references; its compute row,
request-record index and physical page addresses are different coordinates.
Requests grow, finish and release references independently. Compatible prefixes
share immutable pages; every append has exclusive writable ownership.

| Concern | Selected target |
|---|---|
| Lifecycle | One runner-owned `ServingSession`, through `OrchestrationRunner`; CPU authority on host, GPU authority on device |
| Physical KV | PMA-admitted pooled pages, consumed directly by append/attention kernels; no per-request full-context backing |
| Logical capacity | Request records, response leases, queue bounds and pool page geometry are distinct admitted resources |
| Prefix reuse | Reference compatible resident immutable pages; import nonresident archives once into pooled backing; copy-on-write for a shared tail |
| Initial pressure policy | Resident-only, guaranteed-completion admission using pooled growth commitments; no implicit overcommit, preemption or offload |
| Work | One request-intrinsic prefill chunk or a genuinely grouped decode/verifier operation; live extents, never capacity-sized compute |
| GPU execution | Complete retained transactions; device-owned page selection, tables and lifecycle; no count readbacks or recapture |
| Future offload | Versioned page identity, source/destination leases, exact read/write frontiers, coherent replicas and effective-use recency designed now |
| Delivery | Detached, bounded response storage; slow sockets retain neither shared staging nor execution ownership |

The resident-only pressure choice preserves the existing guarantee that an
execution-admitted request can reach its declared limit. It is **not** a promise
of vLLM-style overcommit. Dynamic assignment and shared prefixes are required
now; elastic admission relying on live-request offload is a later, explicitly
qualified policy over the same page implementation (§5 and §11).

CPU, CUDA and ROCm, ordinary decode and MTP, dense and MoE, hybrid recurrent
models and admitted TP/EP/PP topologies all belong to the final target. Work may
land in qualified slices, but missing combinations remain unqualified rather
than silently using fixed partitions, scalar serving, disabled prefix caching,
disabled MTP or a weaker sampling law. Capacity one uses the same paged seam.
Existing supported behavior must be preserved before replacement cutover.

## 2. Logical identities, not physical lanes

Use generation-qualified identities with distinct meanings:

| Identity | Meaning |
|---|---|
| Request handle | External request lifetime, including queued and completed delivery states |
| Request binding | Serving-instance identity, request-record index and monotonic generation |
| Work row | Temporary compact coordinate for one operation; maps back to a request binding and logical token position |
| Logical KV block | A token interval in a request's ordered history, with its cache-family/role identity |
| Physical page handle | Pool, participant, storage class, frame index and non-reused frame generation |
| Published page view | Page handles, valid extents and table version pinned for specified consumers |
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

The implementation should concentrate page behavior behind a small, typed
interface rather than expose free lists, raw reference counts and event edits
to model files. Names below describe required semantics, not installed classes.

| Owner/module | Responsibility and interface |
|---|---|
| `PhysicalMemoryAuthority` | Sole physical admission/reservation/materialization/release ledger; owns typed page-pool resources and their logical grant authority |
| Paged-KV arena | PMA-owned suballocation implementation: atomic admit/append/COW/commit/retire operations and immutable typed views; no independent byte budget |
| `ServingSession` | Request lifecycle, request-local policy and operation ordering; asks the page authority for grants, never allocates KV itself |
| KV kernel interface | Reads/writes typed paged views and live extents; cannot manufacture page ownership |
| Prefix coordinator | Compatibility lookup and retention policy; obtains immutable page/checkpoint leases, not writable aliases |
| `TransferEngine` | Public movement/coherence authority; consumes source/destination leases and exact stream/event dependencies |
| Rank/global orchestration | One topology-wide logical decision and participant-local grants; no remote GPU ownership or nested multi-device compute stage |
| HTTP/other frontend | Immutable input, output credit and external cancellation; no execution controller, page ledger or live GPU-state mirror |

CPU and GPU adapters implement the same ownership semantics. GPU page metadata,
free-page selection, logical commitments, live lengths and table mutations are
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

## 4. Pool geometry and complete physical admission

A resolved plan contains independent, typed dimensions:

- `S`: maximum execution-attached request records; `Q`: preparing/queued limit;
  `R`: total response lifetimes, including detached deliveries.
- Per-request context/output bounds and request-local window policy.
- Per-participant, per-storage-class page size, page count, native layout,
  precision/codebook and alignment; bounded page-table/reference capacity.
- Prefill token-row and decode/verifier work envelopes; recurrent, history,
  sampler and terminal-handoff capacity.
- Capture families, metadata/table banks, transfer lanes, staging and response
  storage, each sized by its real concurrency.

`S >= 1`, `Q >= 0`, `R >= S + Q`; the total preparing, queued,
execution-attached and detached-delivering responses never exceeds `R`.
An explicit positive authored limit is exact or rejected, not clamped.
Automatic capacity comes from admitted topology and PMA, not local test-device
counts. `S` does not promise that every possible set of S maximum-length
requests can enter execution simultaneously; §5 owns memory admission.

Page geometry is derived from model layers, actual TP shards, PP intervals,
KV-head dimensions, cache family and numerical format. K/V, quantization
scales/anchors, TurboQuant metadata, alignment, guards and page-management
storage are all priced. A logical block may comprise multiple local physical
pages; incompatible layer/format classes have explicit separate pools.
Attention-only, hybrid main and shifted predictor caches retain their actual
roles. An all-recurrent participant allocates no dummy attention pages.

The BOM includes, once per physical owner:

1. Complete pool backing, including unused frames and alignment.
2. Frame descriptors, generations, references, retirement queues, growth
   commitments, table versions and bounded concurrent table snapshots.
3. Request state: GDN/short-conv, history/DRY, RNG/controller state, rollback,
   terminal logits/hidden state and sidecar handoffs.
4. Live-work maps, graph/workspace families and collective metadata/buffers.
5. Prefix metadata/checkpoints and each additional RAM/GPU/archive staging
   allocation; shared live pages are not charged again as cache copies.
6. Preparation, input/parser/tokenizer storage, response queues, framers,
   non-streaming accumulation and producer/consumer overlap.

Pool backing and hot-path metadata are materialized before serving/capture.
Logical page assignment during a transaction does not call CPU/GPU allocation
or rebind graph pointers. Growing or replacing physical pools requires a new
admitted instance and retirement of the old one, not hot-path arena growth.
Metadata may cover a worst-case request context without reserving a matching
private KV payload for each record.

For storage class k with B tokens per logical block, unshared live history
uses approximately `ceil(live_tokens / B)` blocks, plus explicitly priced
speculative/COW transients. Several readers of one immutable block consume one
physical copy in that participant's pool. This logical occupancy is distinct
from the already materialized pool's bytes and from promised growth (§5).
Do not advertise unused-but-reserved pool bytes as returned system memory.

## 5. Admission, growth and memory pressure

### 5.1 Resident-only completion guarantee

Queue acceptance reserves bounded preparation/response ownership; it does not
promise a live KV binding. Execution admission atomically obtains a request
record, compatible prefix references, initial page grants and a **growth
commitment** sufficient for that request's declared maximum lifetime.

Commitments are fungible capacity within the PMA-owned page arena, not assigned
frames, full-context per-slot regions or another physical allocation. Actual
pages are assigned only when append/COW needs them. For every storage class,
admission protects the union of pinned live/shared pages plus all admitted
requests' worst-case remaining private growth and simultaneous speculative,
COW, copy and retirement overlap. A shared prefix is counted once; each future
private tail is counted separately. Pending-retirement pages are unavailable
until their actual consumers finish. Retained cache-only pages may be evicted
under their policy, but are not free until references and events permit reuse.

The authority derives the checked geometry and performs this decision once.
Callers cannot estimate free bytes or decrement a mirrored credit counter.
Converting a growth commitment into a page, releasing an unused commitment,
or rolling back an unpublished multi-class reservation is one typed operation.
It must leave every participant unchanged on refusal.

The unspecified output limit remains the real remaining-context limit unless
the public request policy is explicitly changed. Consequently, several requests
that each reserve almost the entire pool may queue even if their current live
KV is small. Paging alone cannot guarantee all of those futures simultaneously.
Publish committed headroom separately from live occupancy; do not present a
full-context commitment as merely next-token admission.

This policy still removes physical partitioning. For example, a 32K-token-equivalent
pool with four request records can admit maximum retained histories of 20K,
4K, 4K and 4K when format rounding and separately priced transients fit. A 20K
request is not constrained by an 8K partition. These are illustrative units,
not an actual model's byte estimate. Shared prefix blocks further reduce the
union of required physical histories without changing the per-request context
contract.

### 5.2 Progress and refusal

Before each finite model operation, reserve every required append/COW/tentative
page and table-publication resource, then publish the new view. No kernel may
write beyond its grant. Current work may await an exact pending retirement or
storage completion, but it cannot wait for unrelated future generation to
free capacity needed to honor its own admitted guarantee.

On new admission pressure: reclaim eligible cache-only references, wait for
specific already-submitted retirements where appropriate, otherwise retain the
request in the bounded queue or return explicit overload. Never partially
activate it, evict live request history, truncate context, silently disable MTP,
or enter a scalar/fixed-slot path. An inability to honor an existing valid
commitment is an invariant failure, not normal mid-response OOM handling.
Output-blocked requests retain their commitments until completion/cancellation;
bounded delivery policy ensures they cannot retain them forever.

### 5.3 Future elastic admission

Admitting only current/next-operation pages can increase concurrency, but needs
a proved pressure escape when all live requests require growth. Merely parking
all of them deadlocks. Future live-request offload must provide bounded
suspend/restore, backing-tier capacity, progress/fairness and full recurrent/MTP
state preservation before such an admission policy is enabled. It must not
silently replace the resident-only guarantee. The same paged substrate serves
that versioned policy; fixed-slot storage is not retained as an alternative.

## 6. Page lifecycle and immutable views

A frame's lifecycle is conceptually:

```text
Free(g) -> GrantedPrivate(g) -> Writing(g) -> Published(g, valid_extent)
                                           |                  |
                                     private append      ImmutableShared
                                           |                  |
                                           +-----> Retiring(g)
                                                        |
                                      all references and access frontiers retire
                                                        |
                                                    Free(g + 1)
```

This is a typed state machine, not permission for callers to set enums. A
private append can mutate only its exclusive valid tail. A sealed shared page
never becomes writable while a cache, request, graph, transfer or archive
reader can still reach it. Appending to a shared partial page requires COW:
grant a new private frame, copy only the valid extent with its format metadata,
publish readiness, then replace that request's table entry. Other readers keep
the old page unchanged. Whole immutable pages can remain shared indefinitely
subject to references and retention policy.

A publication carries valid-token extents, content/version identity, table
version and exact readiness. Reserved unwritten bytes are not KV. Truncate or
speculative rejection changes logical visibility first; discarded pages become
reusable only after queued work using the old view retires. Partial accepted
tails retain only valid committed history; rejected tokens cannot enter prefix
keys, exports, attention lengths or later request histories.

Logical reference count zero is necessary but not sufficient for reuse.
Submitted readers/writers retain access leases through their completion
frontiers. A request cancelling, finishing, or releasing its table does not
cancel already queued native work. Freeze admission of old-view readers before
closing a generation's frontier; otherwise an overwrite can race a reader
submitted after the purported last-reader event.

Stable table storage has bounded banks/versions for the actual pipeline depth.
Publish a new version only after its preparation; retain the old version until
all readers finish. A graph cannot observe half an append, partially updated
K/V mappings or a table reused while its previous consumer remains queued.
Table exhaustion is handled before submission, not by allocating another bank.

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

Every launch covers actual live rows, page spans, tiles or experts. No
capacity-sized grids with wholly inactive blocks, no empty compute families,
no padding communications, no host live-count readback and no per-request row
replay disguised as batching. Device-selected retained branches must cover the
supported shape space economically; catalog size and capture bytes are proof
obligations, not assumptions.

## 8. Shared prefix caching

Prefix identity binds exact token ancestry, model/weights and graph policy,
positions/RoPE, KV family/layout/precision/codebook, required placement history,
checkpoint semantics and trust domain. A hash match alone does not authorize
sharing incompatible content or crossing a tenant isolation boundary.

Resident hits acquire immutable page references instead of restoring full
private copies. A nonresident archive imports its valid payload into exclusive
pooled destinations, then publishes a reusable immutable representation. A
concurrent duplicate import joins the same incarnation or retires its own
unpublished candidate; it must not create unaccounted duplicate ownership.
Cache retention and a request's live reference refer to the same resident
physical bytes when they really share backing. Separate device/RAM replicas
are distinct charged allocations even when their content keys agree.

Complete blocks are natural sharing units. A partial sealed block records its
exact valid extent and compatibility; extending it requires COW. Prefix lookup
while queued is provisional; execution admission atomically validates and pins
the selected incarnation. Eviction removes retention eligibility, not live
readers. One request finishing drops its reference without invalidating peers.

Full-hit terminal logits, the last-token recomputation rule and any recurrent
checkpoint must describe the same logical prefix boundary. Reuse does not
import another request's sampled condition, RNG, penalties, DRY history, depth
learner, output counters or delivery state. New requests initialize their own
policy/history. Domain placement and rebalance state are not request snapshots.

Hybrid models require their GDN/short-conv checkpoints alongside the matching
paged attention prefix. Recurrent state is not interchangeable with K/V pages:
immutable snapshots may be shared, but active mutable recurrent state is
request-owned. Restore, terminal state and page tables activate coherently
across all required families. Never rewind a live peer or replay a missed
checkpoint to manufacture a hit.

Record actual per-request matched blocks, matched/restored tokens, tier,
checkpoint, bypass/discard reason and harvest outcomes. Logical matched blocks
are not necessarily physical record counts for all-recurrent participants.
Carry those observations through terminal summaries to HTTP; reconstructing a
summary with `matched_blocks = 0` loses evidence.

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
one grouped verifier operation.

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
| Initial admission/policy ingress | Immutable prompt, generation-qualified binding, policy and admitted envelope; not a live table update |
| Cancellation ingress | External intent for the original request lifetime; never immediate free |
| Window grant ingress | Unique permission backed by response destination storage; no positions, live counts, depths or selected rows |
| Terminal publication | Committed live output, bounded framing/accounting, typed outcomes and ownership receipts |
| HIP scheduler ticket | Fixed-size authenticated identity naming an already retained complete transaction; host submits only that named branch |
| Explicit KV/prefix tier I/O | TransferEngine-owned storage movement and identity-bound completion; not a general GPU state readback channel |

Page allocation is not an added host ingress kind. A storage receipt cannot be
repurposed to choose model work from downloaded page/row occupancy. Source
sanitizers enforce the closed interfaces and exact non-null streams.

Operation kinds remain typed: lifecycle/selection, one canonical prefill chunk,
complete decode window, and terminal idle. Lifecycle-only progress never calls
an empty generation commit. Initially allow one model transaction in flight per
domain and per request; independent disjoint lifecycle I/O may overlap. Two
publication banks do not authorize two speculative model submissions.

Conditional-capable execution retains selection and complete chosen work in
its parent. HIP uses selector-after-ingress, validates only host-known instance,
ingress, binding-snapshot and retained-catalog identities, then submits the
named complete operation. The selector receipt is an additional observation;
native MTP branch tickets may add more. Enumerate and measure every receipt,
graph and retained byte rather than claiming one wait per window. Neither
backend may host-step layers, poll live counts, synchronize whole streams,
recapture, allocate or rebind during execution.

Ingress is sealed for the operation using it. Request/table generations are
validated on device before access. Terminal/retirement receipts may refer to
bindings still retiring in the immutable ingress snapshot; the host cannot
replace that snapshot with its latest set. GPU readiness gates only the work
that needs the associated pages, not unrelated ready requests.

## 11. Future offload: ordering and tier semantics

The following contracts are mandatory in the page substrate now. Existing
prefix-tier import/export must obey them. They do not claim that transparent
live-request offload, SSD paging or elastic admission is implemented.

### 11.1 Prior readers, overwrite, subsequent consumers

For every destination incarnation, the required dependency chain is:

```text
all prior readers/writers of the destination's old view
    -> exclusive destination ownership / retired old-view frontier
    -> restore, COW copy, clear or overwrite on its exact transfer stream
    -> exact completion publication and coherent new table/view
    -> subsequent consumers on their actual execution streams
```

The transfer waits for the destination's **prior access frontier**, not merely
its last writer. The new consumer waits for transfer completion. Both edges
are necessary. A copy-stream FIFO does not order a queued compute reader on
another stream, and slow SSD I/O is not synchronization. A submitted graph,
sidecar, verifier, collective or exporter remains a reader even if the request
was logically cancelled meanwhile.

Close the old view to new submissions before recording/joining its frontier.
For several compute streams, join all relevant access events; a guessed main
stream event cannot stand in for independent readers. Event handles and their
generations remain owned until their waiters retire. CPU adapters express the
same happens-before relation through actual task/reader completion.

TransferEngine consumes a typed destination-overwrite lease with that frontier
and a pinned source-ready lease. It returns a generation-bound completion
publication; callers cannot fabricate readiness by changing dirty flags.
A new frame/table incarnation is visible only after successful publication.
Delayed completion for an obsolete incarnation cannot attach bytes to a reused
frame. I/O failure withdraws the unpublished destination and preserves native
borrowers until retirement; it never exposes a partially restored page or
silently recomputes through another execution path.

### 11.2 Export, staging and selective overlap

Export waits for the source producer and keeps its immutable source reference
until transfer completion. Mutable recurrent checkpoints freeze at their
canonical boundary and copy into admitted exclusive snapshot storage before
that request resumes mutation. Slower persistence owns detached bytes rather
than a mutable request buffer. A different ready request continues throughout.

Pre-admit copy lanes, events, staging, table banks and retirement queues. If a
required snapshot cannot obtain bounded staging, defer that request at the
checkpoint rather than allocate secretly, miss the checkpoint, or rewind it.
Storage progress is independent of normal output credit and can wake the
scheduler through immutable ownership receipts. Its dependencies must not
block unrelated model roots behind an unselected restore.

Cancellation during load/store revokes future activation, not submitted DMA or
filesystem I/O. The old source, destination, staging and completion records
remain owned until real consumers finish. Shutdown joins actual owners before
free; timeout does not establish retirement.

### 11.3 Replica identity, recency and later pressure policy

A logical content version can have several separately charged physical
replicas. A tier directory distinguishes clean immutable copies, private dirty
state, pending copies, invalidated incarnations and durable archive receipts.
An old backing copy cannot restore over newer committed content. Disk extent
ownership and capacity are explicit storage contracts; memory allocations and
pinned staging remain in PMA, never a second RAM/VRAM reserve.

Eviction recency follows **effective logical use**, not just transfers into a
tier. Reusing a GPU-resident prefix refreshes the retention value of its valid
backing replica even when no RAM/SSD load occurs. Keep a versioned monotonic
use observation, propagate it through the declared cache/storage protocol and
apply a defined eviction ordering before reclaiming. This must not require
per-token host polling or a D2H shadow of page state. Test hot-GPU/cold-backing
competition separately from overwrite safety: recency is policy, events are
correctness, and neither proves the other.

Any future GPU/RAM/SSD chain declares its real stages. RAM and SSD alternative
backends do not prove a measured SSD-to-RAM-to-GPU promotion path. Before live
request offload is enabled, qualify complete page/recurrent/MTP suspend and
resume, backing and staging admission, bounded pressure progress, dirty-version
publication, faults/cancellation and native capture/transport behavior. No
pressure-triggered offload is entered automatically to hide a failed resident
implementation.

## 12. Publication, liveness and terminal outcomes

Request state and KV retire independently from response delivery:

```text
execution terminal staging -> event-ordered transfer -> bounded host demux
    -> detached immutable request publications -> framer -> socket/result sink
```

GPU staging retires after its exact copy/source-consumer edge. Host batch
staging retires after demux's CPU readers. Detached publications do not pin
shared staging, KV pages or an execution record. A response lease can outlive
and overlap several uses of that record.

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
copies belong in the same complete host-memory account.

Lifecycle operations have reserved control/terminal storage independent of
output grants. All-output-blocked cancellation and retirement still progress.
Idle is valid only with a concrete pending wake source or genuinely absent
eligible work; it cannot discard an outstanding grant. Delivery deadlines
resolve abandoned consumers without inventing model deadlines or killing
arbitrary kernels.

Execution outcome and delivery outcome are distinct. The first execution
terminal committed in operation order wins; a later disconnect changes only
delivery. Completed peers keep their real outcomes if another request's reset
fails. A shared execution invariant failure seals the session; service health
and inference fail closed until supervisor restart. Do not continue on an
unproven subset of state.

## 13. Scheduling and multi-participant execution

Decode service and single-request prefill chunks use a bounded anti-starvation
policy. Execution age is a saturating count of completed operations with a
stable admission-order tie break, not host timing. Output- or storage-blocked
requests do not prevent other eligible work or lifecycle progress. Page growth
is secured before selection; headroom does not alter request arithmetic.
Measure long-context attention and routing cost, not token counts alone.
Aggregated prefill/fused prefill-decode are future separately qualified
optimizations, not automatic alternate implementations.

One request binding and ordered admission/commit epoch spans the admitted
participant set. Each participant retains only its actual shard's pages and
state; physical page IDs need not match between participants. A global logical
block maps to typed participant-local handles. Multi-pool/multi-rank admission
is atomic: prepare all real owners, agree the decision, publish coherently or
roll back every unpublished grant. Independent rank-local grant reordering
cannot create divergent histories.

Preserve original MPI rank and NUMA ownership. Native GPU collectives remain
within real homogeneous rank-local domains; CPU/cross-rank communication uses
the existing MPI/shmemspin machinery. Only the declared continuation owns
sampling/control. Expert-only participants do not acquire dummy KV histories.

Collectives carry only actual live rows/spans plus necessary protocol metadata.
Page-table capacity, unowned shards and inactive sequences are not wire extents.
PP requires an explicit microbatch schedule and exclusive concurrent arenas;
its publication/retirement joins every actual participant. One global in-flight
transaction alone does not certify pipeline utilization. Test singleton, odd
and larger participant counts, unequal widths, more than two stages and valid
CPU/GPU compositions; local test hardware does not set support limits.

## 14. Observability and qualification

Report separately: materialized pool bytes; free, private, shared, cache-only,
tentative and retiring pages; logical references; protected growth commitments;
COW bytes; valid imported/exported bytes; staging occupancy; admission refusals
and reasons. Define non-overlapping occupancy categories and orthogonal aliases
rather than summing references as physical pages. Passive reports come from
permitted immutable observations, not live GPU probes or a host admission
ledger. Expose a complete physical BOM and high-water evidence independently.

Per-request summaries preserve actual prefix observations, committed tokens,
draft/verifier work, terminal causes and delivery outcome. Shared kernel time
and shared bytes cannot be charged in full to every request and then summed.
Record real submitted launch dimensions, transfer/collective extents, graph
reuse and forbidden-path counters. A path counter is not an economy receipt.
Stats epoch reset cannot cancel requests, clear caches or let old requests
repopulate new counters.

The following are qualification obligations, not existing test names:

| Contract | Required adversarial evidence |
|---|---|
| Pooled geometry/admission | One request exceeds an equal-share partition; mixed lengths, exact fit/refusal, overflow, all-class rollback, S independent of pool size, full-default-output commitments |
| Shared prefixes/COW | Multiple readers of actual identical frames, private divergence, partial tails, one reader cancellation, eviction with surviving aliases, no private full-prefix clones |
| Lifetime/reuse | Delayed queued readers, multiple streams, stale generations, cancel/restore/retire races, repeated reuse and reset; no early reuse after reference count zero |
| Native paged arithmetic | Noncontiguous/permuted frames, page-edge and partial extents, supported formats/codebooks, all live M and relevant N/K, ordinary/MTP and optimized dispatch |
| MTP/hybrid | Tentative pages, accept/reject/correction, shifted caches, per-request recurrence, checkpoint/hidden/logit handoff under interleaving |
| Offload-ready protocol | Prior-reader -> overwrite -> new-consumer edges; missing-edge negative controls; hot GPU hits refresh valid backing recency; obsolete/failing completions cannot publish |
| Response memory/liveness | Oversized semantic event plus slow consumer retaining a prior payload; exact reconstruction and simultaneous byte bound; no-grant cancellation and bounded shutdown |
| Distributed execution | Atomic participant admission, real rank/NUMA ownership, ragged live communication, arithmetic order and topology-generic lifecycle |

Replay recorded inputs and initial state through the actual production
controller/kernel, preserving schema/policy, geometry, formats, ordering,
sampler laws, page/table identities and relevant completions. Expected outcomes
must include decisions, ownership transitions and outputs. Reject incomplete
traces; synthetic repetitions are lifecycle/contract tests, not recorded replay.
Every changed controller/kernel and discovered production defect receives an
explicit focused `ProductionTestPreflight` registration. Fast Unit tests remain
device-free; native kernels/capture use explicit integration suites.

Complete Unit/preflight, source-to-loaded-code identity and canonical multi-prompt
frozen-history final-logit panels remain separate gates. Focused passes or
legacy fixed-slot traces cannot certify the new allocator or paged kernels.
Large corpora belong in `Llaminar/corpora`, not this source repository.

Economy uses matched Release binaries and production defaults: solo and mixed
lengths, prefix-sharing and miss-heavy traffic, burst/steady offered load,
streaming and non-streaming, slow consumers, long/short coexistence and
cancellation/replacement. Measure queue-inclusive TTFT/E2E, output gaps, SLO
goodput, unfinished work, pool/commitment high water, copy/attention overhead
and pressure behavior. Profile new CPU/CUDA/ROCm kernels and compile all shipping
GPU targets with spill guards. Numerical correctness and economical execution
must pass together.

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
