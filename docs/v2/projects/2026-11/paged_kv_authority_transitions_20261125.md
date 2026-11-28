# P0/P1 authority transition specification — working draft

**Revision-5 reconcile (2026-11-25):** PKV-01 is now owner-selected in the
rewritten design/plan (`tiered-paged-design-rewrite.L7ICwq`): bounded
optimistic retain/restore with a concrete all-resource/topology completion
witness (design §§5.1–5.3, §10, §13.2). The summed-maximum O+G+T≤P form below
remains a conservation sub-contract inside charged backing, not the
completion policy; the selected policy replaces it as the admission witness.

This specification responds to PKV-02/03, not an accepted P1 exit. The initial
unpublished `PhysicalPagePool` prototype implements only the private-grant
subset below; it exposes no payload address or readable view and is not used
by inference.

## Conservation domain

Every count belongs to one real participant and native storage class. Let:

- O = unique assigned physical frames, including private, tentative, immutable,
  cache-only and retiring frames. References are aliases, not extra occupancy.
- G = unconverted protected private-growth promises.
- T = unconverted simultaneous transient allowance not already included in G.
- P = materialized pool frame count.

In revision 5, G and T are outstanding finite-operation and progress
obligations: each is the concrete work of an admitted complete transaction
(one exclusive write, one COW destination, one queued progress grant), not an
independently reserved eventual request maximum. The accounting invariant
O+G+T≤P is a current conservation requirement for any admitted transaction
state, not a historical or replaced inequality. This is logical ownership
inside already PMA-charged backing, not a second physical-byte ledger.
Resource vectors also include bounded table snapshots, access/retirement
records, descriptors, staging and completion objects. A page transaction that
fits payload but cannot acquire its publication resources cannot submit. No
resource count is copied into a second scheduler balance.

Revision 5 selects exactly that common topology-wide resource-vector witness
(non-simultaneous finite grants plus all-resource/participant completion
witness); it is not implemented by merely changing this inequality or accepting
every next-token fit, and the witness algorithm itself remains an open
implementation obligation, not an installed default. Summed-maximum wording
above is historical pre-selection context.

## Required logical transitions (finite-operation form)

The current requirement is the finite-operation form below: each row protects
or releases one outstanding admitted transaction, and the completion witness
is the authority for admitting further work. The earlier prototype API reserved
per-request eventual maximum pages; those rows are retained as prototype
history under [Prototype maximum-reservation subset history](#prototype-maximum-reservation-subset-history)
and are not the revision-5 current requirement.

| Operation | Occupancy / promise transition | Publication / refusal obligation |
|---|---|---|
| Protect one outstanding finite private-operation grant | G += one transaction's concrete pages, after all-class feasibility | No frame assignment; refuse without changing existing owners |
| Assign one protected private frame | O += 1; G -= 1 | Increment non-reused frame incarnation; no readable bytes yet |
| Reserve extra bounded transient | T += allowance only if not part of G | Derive from allowed overlap, not sum mutually exclusive peaks |
| Assign transient destination | O += 1; T -= 1 | Exclusive write plan; no consumer sees unwritten bytes |
| Share a complete immutable frame | O unchanged; logical reference added | Authenticate content and boundary; pin referents through accesses |
| Seal/share a partial private tail | Recompute every writer's COW liability and top up G/T **before** alias publication | If top-up cannot fit, defer/refuse sealing; do not silently underpromise existing writers |
| Allocate COW destination | Convert the appropriate G or T exactly once into O | Copy only valid native extent/metadata; source readers remain valid |
| Publish COW replacement | O unchanged; new reference replaces this writer's old reference | Source may remain occupied for peers/cache/queued readers; table-version publication is atomic |
| Accept tentative retained frames | Reclassify occupied tentative frames as committed; no extra O debit | Commit table/content/checkpoint/visible length together |
| Reject tentative suffix | Remove logical visibility, but keep O for every borrowed frame | Queue retirement; never credit cancellation/length rollback as physical reuse |
| Finish/cancel outstanding finite private-operation grant | Release unused G/T; assigned private frames release only if there are provably no consumers | Unpublished rollback is legal because the unpublished API cannot publish or register access |
| Finish/cancel published request | Release future liability after revoking new work; drop request references | Existing queued work and access receipts keep the old incarnation alive |
| Real last-reference/access retirement | O -= 1, only when closed frontier actually completes | Frame becomes assignable under a fresh incarnation; old completion cannot publish into it |
| Aborted all-class prepare | Release only newly acquired unpublished ownership | Existing published owners/promises unchanged; issued nonces/generations are not rolled back |

### Prototype maximum-reservation subset history

The initial unpublished `PhysicalPagePool` API instead reserved a request's
eventual maximum pages and released them as unpublished private maximum. It
implemented only the private-grant subset, exposed no payload address or
readable view, and was never used by inference. Under revision 5 those rows
are superseded: maximum holding periods are not admitted as fungible
reservations, and only finite-operation/progress obligations are protected
under a complete witness.

No transition converts the same destination from both G and T. Retained cache
frames remain in O until real eviction retirement. A change in reference
classification cannot manufacture a free frame.

## Shared-tail exact-fit witness to implement

Use 16-token pages. One sealed page contains nine tokens and is pinned by an
exporter. Two writers may each reach token 16. Neither may append into the
shared page. Occupancy is initially O=1 and each writer requires one private
replacement, so G=2 while both one-page private-operation grants are
outstanding together, total 3 frames simultaneously. A three-frame pool can
protect the simultaneous state; a two-frame pool cannot grant both together
and must refuse the second new liability before publishing the alias. This
refusal is scoped to simultaneous protected grants: a separately proved
complete witness may permit ordering such that the first operation actually
finishes and retires its frame before the second grant is admitted, while the
exporter's source remains pinned. No serial or ordered completion is claimed
by this example or proven here.

After writer A receives its destination: O=2,G=1. After writer B receives its
destination: O=3,G=0. Accepting either retained private tail does not add another
frame. Cancelling A while an operation still borrows its destination keeps O=3;
only its real retirement reduces O. The exporter's source reference can retire
independently when its actual transfer completes. Late sealing of a formerly
exclusive tail must obtain these replacement liabilities, not assume the
original admission already contained them.

This is a constructed contract example, **not recorded replay evidence**, and
does not prove any ordering policy.

## Three distinct typed objects required before native access

1. **ExclusiveWritePlan**: original request/content version, pinned old readable
   extent, exact newly granted write range, format/basis referents, exclusive
   destination owner and joined prior-access frontier. It does not expose new
   bytes as history.
2. **OperationReadableView**: immutable table snapshot and operation identity,
   producer ordering, per-query causal/tentative visibility, registered access
   lease and eventual exact completion. Its lifetime includes registration
   before native enqueue. Admission to the old view closes atomically with
   access registration; a registered reader must attach real completion or abort
   before closure can finish. Indeterminate enqueue/event failure seals or
   quarantines, never fabricates a completed access.
3. **CommittedPublication**: accepted table/content/visible-length/checkpoint
   version. Prefix lookup/export/subsequent model operations can acquire only
   these published committed referents, not a writable plan or verifier suffix.

Frame incarnation alone is insufficient: a rejected suffix and correction can
use the same physical frame. An old operation-readable content/range version
must retain its bytes until its last borrower completes, even if frame index and
incarnation do not change. Table-header copying alone does not pin payload,
AQ8 basis, QFN compressed history or recurrent checkpoint referents.

## Current executable subset and limits

The private CPU prototype materializes all payload and bounded metadata through
PMA `CPUStorageRoot`, issues process-unique pool incarnations, and keeps frame
and request generations monotonic. `tryAdmitTogether` validates exact empty
output slots and same-PMA distinct owners before issuing requests. It rolls back
new unpublished grants on pressure/exception. CPU ownership is single runner
thread; no multi-thread/native-reader publication semantics are claimed.

Focused contract tests exercise two-class refusal after a first prepare,
original peer frame preservation, subsequent exact-fit success, distinct pool
identities, generation reuse, exact BOM/release, invalid geometry and partial
construction rollback. Packet `CB-checkpoints/paged-p1-multiclass.6dWosx` passes
three tests in the explicit production preflight registration. These constructed
cases are not recorded replay, complete P1, native paged attention, a pressure
policy, or a GPU authority implementation. The disclosed full PMA Unit gate
failure remains open. Access/publication, sharing/COW and policy-dependent
conservation remain to implement before any cutover.

Follow-up private lifetime subset: an original grant can pin an assigned frame
without obtaining any payload pointer or readability claim. Grant retirement
returns unused growth immediately, but retains assigned occupied capacity until
the final private pin is destroyed. A recycled request record cannot reclaim
that frame early; the pin also retains backing/metadata PMA claims through pool
handle destruction. New admission sees that occupied retirement debt. This is
not yet a native access registration/enqueue/completion protocol or published
view: no submitted device/task operation can use this interface.

The four-test focused registration passes at
`CB-checkpoints/paged-p1-private-pins.dwZywI`. The new lifecycle test holds two
pins across logical retirement, verifies refusal until both release, then checks
real frame reuse with a newer generation and rejects the old identity. It also
proves the last retained pin owns physical storage after external pool release.
These are lifecycle/contract tests, not recorded replay or complete P1 acceptance.

A subsequent native diagnostic capture recorded actual commands and decisions
from this unpublished authority to
`tests/v2/fixtures/paged_kv/private_ownership_v1.txt`. Initial state is explicitly
empty with zero generations; replay binds the new process pool incarnation once,
then compares every decision, physical frame index and frame generation. It
preserves the pinned-retirement refusal and later reuse at generation two.
Capture: `CB-checkpoints/paged-p1-recording.rcAtQ6`; ordinary replay never rewrites
expectations. Two explicit preflight registrations now pass (four contracts plus
two replay/validation tests) at `CB-checkpoints/paged-p1-replay.lM1duq`.
This is recorded controller-input replay of the unpublished subset, not native
KV execution, access-frontier replay or complete P1 acceptance.

Latest CPU byte-access subset adds `ReadAccess`: registration pins the original
content before submission, `closeReaders` prevents later registration, and an
already registered CPU reader can begin after closure/cancellation. Executing
access destruction is fatal unless exact synchronous CPU completion was
reported; abandoning an unsubmitted registration releases its pin. Append and
frame reuse remain blocked until the final original registration retires. This
is single-host-authority synchronous CPU semantics, not device-event ordering,
parallel metadata mutation or a native KV-table operation view.

The added registration lifecycle case passes at
`CB-checkpoints/paged-p1-registered-readers.xzekDw`. The complete PMA Unit
registration subsequently passes **147/147**, plus six page contracts and two
replay tests, at `CB-checkpoints/paged-p1-pma-family.Wnqug6`. The earlier Unit
failure is preserved: its prepared-lookup test demanded explicit online NUMA
before exercising aggregate `cpu` payload ownership. That test now initializes
the aggregate CPU backend; no explicit-placement assertion or hardware test was
removed. This passing family is not complete Unit/production preflight, P1 exit,
or any accelerator qualification.
