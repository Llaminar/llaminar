# Native preparation before QFN source admission

The canonical PMA lifecycle and exact descriptor permits below are implemented
and pass the focused qualification recorded below. Complete SDK preparation,
application wiring and native capture remain under implementation. The existing
complete preflight checkpoint and source-cell results remain in
[the admission matrix](QFN_SOURCE_ADMISSION_MATRIX.md).

## Problem and target

Native communicator topology preparation needs real CPU storage before it can
produce the exact GPU/CPU buffer declarations used by final placement. Today
`OrchestrationRunner::publishPhysicalMemoryAuthority` creates PMA after the
optional-filled source certificate. An independent preparation certificate
would leave live native roots outside the final authority.

The target is one `PhysicalMemoryMaterializationState`, one immutable resource
observation registry and one set of live allocation counters throughout startup:

1. **Preparing:** native infrastructure claims exact source-declared CPU0
   preparation roots before creating backing. Read-only device/granularity
   queries may produce allocator geometry. Model/optional allocations and GPU
   backing are unavailable in this phase.
2. **Frozen preparation:** close early growth after every preparation root is
   published and temporary probes/scratch retire. Retained program containers
   and their original leases remain live. A typed snapshot authenticates their
   actual root declarations, epoch, allocator identities and bytes.
3. **Final admission:** combine that retained preparation footprint with the
   SDK's mandatory native root declarations, then place optional experts and
   ngrams using the existing planners. Seal the same state once with the final
   certificate. No new capacity sample, residency relabel or lease migration.
4. **Materialization and retirement:** the actual SDK consumes its prepared
   program once, creates the declared roots, records/captures using admitted
   storage and releases each original physical owner after its final alias.
   The same session retirement gate includes the preparation containers.

Existing construction from a complete certificate remains direct initialization
of an Admitted state. Serving cannot reopen preparation or reseal admission.

## Authority and interfaces

The immutable registry contains all physical allocator identities that the
final plan may use. Resource indices never change while leases exist. CPU0
owns host capacity; NUMA remains allocation-placement metadata. New resources,
conflicting observations and duplicate initial observations are errors.

`NativeStoragePreparationPmaBinding` is a restricted infrastructure entrypoint,
not an engine-wide reserve API. Its early callback admits only exact SDK-issued
CPU0 preparation roots. The original PMA mutex protects preclaim, publication,
cancellation, snapshot close and final seal. Preparation counters remain in
that same state and become source-BOM materialization exactly once after seal.
They are neither additional setup slack nor adopted residency.

The native preparation program owns frozen membership/configuration, actual
topology/channel/transport decisions and source-allocator geometry. Its visitor
enumerates physical creator roots once; alias edges identify retained source
owners without adding chargeable bytes. Actual program/root owners retain the
lifetime tokens. Numeric pointers or anchor IDs alone are not lifetime proof.
Materialization consumes the same immutable geometry rather than calculating
another allocation plan.

The SDK program may record immutable creator facts from its actual metadata
allocators. Its record pages must themselves be preclaimed and self-declared
without recursively allocating another record page. Retained and scoped
factory allocations use distinct typed authorities: only retained containers
enter the creator visitor, while scoped scratch retires before planning. If
record insertion fails, release the pending canonical claim before the requesting
allocator reaches Create. A later Create or constructor failure invalidates the
whole factory. Only the actual successful preparation jobs' return/join proof
may freeze it. Actual container owners release their allocations; declaration
records neither free described roots nor become an allocation lookup registry.

Final callbacks consume an exact source descriptor once, under the existing
PMA mutex. The permit retains its original root anchor, allocator, placement,
geometry, family, lifetime and phase; an aggregate byte reservation alone is
insufficient. Changed or duplicate roots cannot spend another descriptor's
capacity. Failed creation returns its pending bytes but does not recycle the
consumed program slot. Failed physical cleanup retains both the actual token
and its charge. These permits carry lifecycle state, not a second byte ledger.
The prepared-program path replaces the preliminary setup-slack callback path;
CPU0 preparation probes use Preparing, and other selected roots require exact
source declarations before final admission.

Temporary lifetime sharing requires an SDK-authored concurrency relation that
the actual native schedule enforces. A numeric interval ID is insufficient.
Retained recording metadata derives from the actual typed operation program
and capture-family multiplicity; a caller-supplied byte allowance is not a
declaration. Unknown selected native roots fail precisely.

## Failure and publication

Before final seal, construct every potentially throwing object, including the
returned final binding, in a disarmed state. Under the original mutex, recheck
the snapshot/root floor and phase, then publish the certificate and transfer
the existing epoch using only nonthrowing operations. Failed admission leaves
all live tokens, counters, indices, snapshot and epoch unchanged and available
for ordered cancellation/cleanup. The old preparation binding cannot close an
epoch transferred to the final native session.

Ordinary PMA allocation/reservation/adoption and source-certificate queries
must reject Preparing under that same mutex. Once published, the certificate
is immutable. Late callbacks from the old preparation context cannot allocate
against Admitted state. Native cleanup failure retains the claim until physical
retirement actually succeeds.

## Implementation and tests

The first combined vertical slice uses the actual native abort-owner subtree:
CPU0 preparation container, exact host-VMM descriptor, one-shot final consume,
actual communicator abort fields/refcount and final alias retirement. Its
existing private callback-double tests must be replaced by the real PMA
Preparing state before it qualifies this integration boundary. Full communicator,
channel, proxy, recording and graph ownership remains subsequent work.

Opaque driver resources remain a separate, unqualified integration boundary.
The current CUDA `prepareNativeExecutionContextStorage` probes stack growth and
restores it before requesting a measured lease; the current
`claimMeasuredRuntimeContextStorage` creates a separate authority. Neither is a
pre-creation certificate for new SDK streams, events, mempools or capture
resources. The exact-root transition does not extend or certify that route.
The selected SDK source must enumerate those creation sites and lifetimes.
Any prepared backend handle it borrows needs a real retained owner and original
admission proof; a numeric context ID or handle is insufficient. Newly created
resources cannot be relabeled as pre-existing occupancy. Resolve that backend
contract before qualifying the complete native program.

Focused regressions must be explicit in Unit and `ProductionTestPreflight`:

- Competing last-capacity preparation claims; malformed identity, NUMA,
  geometry, epoch and purpose; no backing creation after a rejected preclaim.
- Claim/close races, pending publication, cancellation before/after close and
  surviving aliases after source/binding destruction.
- Allocation failure immediately before seal; duplicate seal; foreign snapshot;
  changed/missing resource; retained-root release during planning; missing native
  mandatory footprint and attempted resident-credit relabel.
- Actual optional placement with exact preparation/native demands present before
  fill: CPU metadata can move optional RAM rows to mmap, and GPU native roots
  can reduce optional VRAM placement. Each physical root is charged once.
- Actual SDK preparation/consume/retire through the same authority, including
  rejected final admission, source-first destruction and physical-cleanup faults.
- Exact descriptor consumption with spare aggregate capacity: changed root
  geometry, duplicate anchors and simultaneous claims must reject before Create.
  Failed Create cannot make the consumed descriptor available again; recording
  metadata cannot be claimed during preparation or executable replay.
- Source-owned metadata recording: self-declared page growth, insertion-failure
  rollback before Create, failed constructor preventing factory freeze, and
  child-container retirement before the declaration pages disappear.
- Existing direct-Admitted behavior, source guards and the complete Unit suite.

After the host-owned lifecycle is proven, run the native selected-root/captured
regressions on CUDA and ROCm, then affected real QFN source cells with MTP Off,
fixed1, fixed3 and Dynamic while preserving the declared retained capacity.
Keep CPU, CUDA, ROCm and composed expert/ngram placements in the canonical
matrix. Complete production preflight and every hardware-eligible source cell
remain required before declaring the expanded admission work qualified.

## Core qualification, 2026-10-30

The main Integration/AVX512 build passes all **794 Unit registrations**, with
zero skips. Its unchanged target set is CUDA 80/86/89/90 and HIP gfx906. The
existing physical-authority executable includes the original direct-Admitted
tests, preparation/permit tests and actual optional-placement regression.

Five explicit `ProductionTestPreflight` replay registrations pass **20 repetitions
each**: preparation, exact permits, optional placement, resource programs and
cancelled execution.
The source policy and its negative controls also pass. Compile-time access
checks reject public preparation construction, aggregate-only final callbacks
and direct native-root claims. A fully instrumented actual PMA implementation
and its 45 core tests pass **20 repetitions under ThreadSanitizer**; only that
diagnostic process disables ASLR, with no host or production setting changes.
The original core evidence is `/tmp/qfn-native-preparation-main-r1/`; the expanded
Unit/replay run is `/tmp/qfn-native-resource-programs-main-r1/`.

The resource-program replay covers all seven native allocator kinds across
12 device-free inventories: CPU-only; CUDA and ROCm with 1/2/4/8 GPUs; and mixed
CUDA/ROCm with 2/4/8 GPUs. Reversed discovery order preserves canonical resource
indices. NUMA placement shares CPU0 accounting; a changed vendor or CPU ordinal
cannot spend another descriptor's capacity. Every program proves materialized
bytes per resource and final retirement. These synthetic resource declarations
do not qualify those hardware topologies or their transports.

Review found that cancellation originally shared the successful executable-seal
state. A fully published session could consequently cancel and then seal for
execution. `Abandoned` now permanently rejects recording and executable
publication while preserving physical retirement. The preserved original
implementation fails the new cancellation regression. Earlier aggregate-only
permits also fail changed-root and duplicate-root controls despite spare quota.
Those controls and the frozen import are retained under
`/tmp/qfn-native-pma-preparing-transition-r1/`.

This is a core lifecycle checkpoint, not a fresh complete preflight receipt or
qualification of a whole SDK communicator. CUDA/ROCm owner subtrees and the
source-owned metadata recorder have separate private CPU-double/ASan proofs;
their actual selected transport, backend-resource, captured-operation and
application re-admission boundaries must still be integrated and qualified.
