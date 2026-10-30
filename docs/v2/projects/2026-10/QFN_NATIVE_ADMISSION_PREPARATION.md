# Native preparation before QFN source admission

Implementation plan. The interfaces below are approved for the private QFN
admission work; they are not yet installed or qualified. The existing complete
preflight checkpoint and source-cell results remain in
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
- Existing direct-Admitted behavior, source guards and the complete Unit suite.

After the host-owned lifecycle is proven, run the native selected-root/captured
regressions on CUDA and ROCm, then affected real QFN source cells with MTP Off,
fixed1, fixed3 and Dynamic while preserving the declared retained capacity.
Keep CPU, CUDA, ROCm and composed expert/ngram placements in the canonical
matrix. Complete production preflight and every hardware-eligible source cell
remain required before declaring the expanded admission work qualified.
