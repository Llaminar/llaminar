# QFN cross-rank preparation and production-preflight repair

This is a qualification investigation, not a tuning dashboard. It records the
implementation boundary and the evidence required before claiming cross-rank
inference. The current feature branch has not qualified real QFN generation
across ranks. The last pushed checkpoint is `2948b1e49`.

## Scope and qualification boundary

The user requested a complete green production preflight before further QFN
performance tuning, and explicitly authorized implementing cross-rank pipeline
preparation. Original MPI ranks must own only GPUs on their observed NUMA node.
Current inventory places both RTX 3090s on NUMA 0 and all nine AMD
GPUs on NUMA 1 (MI50 ordinals 0–3 and 5–8, RX 7900 XTX ordinal 4).
The first eight ROCm devices therefore include the XTX; eight-device coverage
is not evidence for an eight-MI50 cohort. Earlier evidence
recorded a different XTX placement; current native discovery and PCI sysfs take
precedence. A local rank cannot supply a two-CUDA plus two-ROCm pipeline. Its
two GPU domains must live on the two original MPI ranks.

CUDA-only cross-rank execution is not hardware-qualified on this host. Adding
another rank to the CUDA socket does not qualify the requested one-rank-per-socket
deployment. Device-free tests can still cover arbitrary rank labels, disjoint
stage ownership, and more than two ranks.

Qualification has four distinct levels:

1. Exact original stage ownership, prepared weights and one canonical PMA per
   physical rank.
2. Native participant-local captured activation transport, exact live extents,
   retirement and original TP fanout.
3. Complete ordinary generation, request reset and fixed/dynamic MTP through the
   actual runner. The terminal stage alone owns sampling and outcomes.
4. Real GGUF QFN generation and PyTorch parity, including final logits and
   retained layer/stage diagnostics.

Passing a level does not certify a later one. Existing success obligations must
not be converted into expected rejection or skipped tests.

## Closed baseline and focused repairs

The complete baseline prerequisite run closed red after 8,084.45 seconds:

| Lane | Tests | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Unit | 1,021 | 1,004 | 17 | 0 |
| Host integration | 2,355 | 2,327 | 28 | 0 |
| CUDA integration | 312 | 309 | 3 | 0 |
| ROCm integration | 431 | 412 | 17 | 2 |
| Exclusive integration | 714 | 592 | 95 | 27 |

Many aliases share a failure owner. Repairs have been collected and tested in
focused batches rather than repeatedly stopping a full run at its first error.
The subsequent complete Unit run passed 1,017/1,021; the four remaining owners
then passed focused reruns. A new complete Unit/preflight transaction is still
required after the current implementation changes.

The four native planning MPI2 timeouts had a concrete collective-order defect.
`SharedHostDevicePages` called lazy `MPIContext::topology()` only on the GPU and
host-allocation roles. Constructing that topology enters WORLD collectives. An
idle peer entered the enclosing phase's completion collective instead, so the
collective sequences disagreed. All ranks now initialize the original topology
at the beginning of the existing second phase, after the first phase has
collectively accepted identity, ownership and PMA validation. This avoids both
the idle-peer deadlock and an asymmetric-validation deadlock.

The original four CUDA/ROCm planning aliases pass in 36.02 seconds total. Their
new native regression first rejects a missing PMA on one peer, then performs two
actual collections where one rank owns both roles and the other is idle. The
collective timeout is unchanged.

## Cross-rank implementation under qualification

The current source adds typed global stage projection, rank-local prepared
weight claims, and shared stage bindings. Original stage indices and physical
rank ownership survive projection; remote stages remain metadata, never remote
GPU pointers or contexts.

Cold CPU discovery records the join between OS NUMA/CPU identities and hwloc
logical core indices. Native NUMA mapping uses that immutable original
observation rather than assuming OS CPU numbers, socket numbers and MPI rank
numbers are interchangeable. The corrected launcher writes a native rankfile
through the existing MPI session owner. Each rank receives the complete logical
core range of its observed NUMA node. This avoids combining OpenMPI's conflicting
CPU-set and PPR mapping mechanisms. The rankfile retires with the original MPI
session; no second launcher or session is introduced.

The captured rank-channel primitive reuses `TransferEngine`, original shared
backing/session machinery, existing native kernels and `PhysicalMemoryAuthority`.
Each process owns one local endpoint. An enclosing session retains shared pages
and the allocator's physical charge until every local graph/alias has retired
and the original ranks have completed explicit unmap consensus. Destructors do
not enter MPI. Forward and reverse channels declare their physical host owner
explicitly.

Earlier two-rank native evidence passed CUDA→ROCm and ROCm→ROCm before the
reboot. The current focused native batch passes the actual rank-local CUDA and
ROCm pairs plus CUDA→ROCm rank transport. The ROCm-only distinct-rank case is now
hardware-ineligible: all AMD cards belong to the same NUMA node. Canonical
qualification records that absence without treating it as success or weakening
required local ROCm coverage. Sparse ordinals and unknown locality are covered
by device-free eligibility and receipt regressions.

The current startup rerun also exposed two defects after replacing conflicting
CPU-set/PPR mapping: relative `+n0` rankfiles require an explicit local host
allocation, and independent frontend children under one guardian reused one
fixed rankfile name. A native rerun proved that `--host localhost` does not establish allocation
early enough for OpenMPI4.1’s relative rankfile parser. The final launcher names
`localhost` directly in its local-only rankfile and creates a unique, inode-owned
file per launch through the existing session authority. The ancestor retains files through native exec and retires them
after descendants exit. Focused coexistence/retirement and native affinity validation now pass; no
timeout or affinity requirement was relaxed.

Live INT32 row inputs lower to checked byte extents in the existing acquire
operation. The receiver checks its own expected rows against the producer's
publication. No row-count readback, extra byte-count authority or recapture is
introduced. Native tests cover both directions, 65→3→1→0→65 rows, poison guards,
read-only row inputs, rejected ownership and explicit retirement.

The first focused host batch passed six of ten aliases. New preparation fixtures
omitted the existing model-specific weight-load contract; they now reuse that
complete source policy. A new alias regression also found that batch validation
could import producer events before rejecting a later overlapping row input.
The correction checks all immutable batch geometry before importing those
events. The focused binding and stage-preparation rerun now passes; no admission
or alias guard was relaxed.

Receiving-domain TP fanout is implemented and undergoing native qualification.
The first native rerun exposed missing external-input event joins in the new
fixture; those now match the production executor’s pre-capture frontier pass.
The original two-rank TP2 activation regression now uses only each rank’s local
workers and explicit rank channels between the domains. Each member retains a PMA-owned metadata bank. The original
channel's 24-byte publication header precedes the completed native broadcast of
only the live activation bytes. Each member validates the receipt and its own
fresh row count before model roots run. Zero rows retain only protocol traffic.
The actual graph residual width determines storage, including QFN's wider
residual representation; embedding width cannot substitute for it.

The latest eight-case focused batch is green in 46.81 seconds. It includes
CPU startup/placement, CUDA→ROCm rank transport, both CUDA/ROCm native fanout
components, and both TP2 cross-rank activation directions. Each activation-domain
case completes 100 retained round trips across 257/17/129/1/0 live rows, with
independent padding guards and original graph identity.

The subsequent prepared-fixture batch passes all three aliases in 15.29
seconds: the full device-free binding suite plus both native metadata directions.
All seven metadata schemas now share one original source/PMA and native domain,
while each retained definition keeps its proper opaque reservation. Both
directions preserve 20 replays per schema and the retained-workspace guards.
The remaining 27 previously failing host aliases pass in 28.21 seconds.

Complete generation composition remains pending.
The intended reuse is the existing MPI transaction publisher/channel/follower:
the actual terminal rank runs the decode algorithm and other ranks execute the
named retained bodies. Explicit scalar MTP condition roles distinguish
speculative continuation from a newly committed serial token. These roles keep
the fixed ticket layout but require transaction ABI version 6; older or contradictory tickets
are rejected before publication. The full-generation implementation is under focused qualification. The new
original-rank service retains cold-allocated native event slots and progresses
immutable tickets without waiting for each transaction. Receipts require actual
native completion; only bounded ring reuse and command terminals may wait.
The transaction unit suite, command replay, original-MPI overlapping-completion
replay and failed-fence teardown replay pass. The first native service replay
exposed an ACK ordering defect: an immediately completed submission overtook
an earlier pending native completion. Both now retire through the same retained
FIFO. The original failure and repaired replay are recorded separately. It has no native
generation or real-model correctness certificate yet.

## Peer investigation and realistic cohort correction

The strict original-inventory peer test reproduces an old byte on the return
from an MI50 to the 7900 XTX. Both channel producer/consumer epochs reach the
expected value. Terminal observations show the current byte in the source,
forward slab, intermediate buffer and return slab, but the previous byte in the
destination. A separate GPU load of the destination also sees the old value;
this is not only a DMA observation problem. A post-terminal GPU read of the
remote slab sees the new value. This fixture combined the XTX on NUMA 0 and an
MI50 on NUMA 1 in one process. It does not represent the requested inference
ownership and cannot establish a defect in rank-local MI50 communication.

The user clarified that homogeneous NCCL/RCCL qualification must use each
original rank's actual local GPU cohort. Tests are being corrected accordingly.
On this machine the CUDA pair belongs to rank 0 and the MI50 cohort belongs to
rank 1. The explicit two-rank captured transport remains valid cross-socket
coverage. No GPU visibility masks, fabricated additional ranks on one socket,
or lowered byte/economy assertions are permitted to manufacture a pass.

The original ordinary peer allocation was coarse. HIP's `Finegrained` request
also selected a coarse pool on these devices. Requesting the extended fine-grain
pool with `hipDeviceMallocUncached` and checking the actual allocation mode fixes
that concrete admission mismatch. It does not by itself fix the stale byte.
The pinned runtime's pool name does not establish that all relevant hardware
caches bypass memory: its allocation flags differ by architecture.

Changing consumer load policies, producer streaming stores, or adding a sole
publication system fence did not repair the strict byte test. Those unsuccessful
kernel candidates have been removed. Their source/ISA and native failures remain
in the local evidence directory. A native HDP publication candidate also failed
the strict byte test and has been fully reversed. Its successful compilation
and zero-spill ISA do not establish a coherence fix. Further tuning of that
artificial cross-socket homogeneous fixture is stopped; the actual rank-local
cohorts retain their full native correctness obligations.

## Current focused qualification

The corrected original-rank projection, routing, ownership and peer batch
completed 20 aliases in 228.96 seconds: 19 passed and one failed. The failure is
`V2_Integration_CapturedPeerPayload_ROCm2_OriginalInventory`, now using MI50
ordinals 0 and 1 on the same original NUMA-local rank. Thus the stale payload
is also reproduced on the intended homogeneous cohort; the earlier artificial
cross-socket fixture was not sufficient evidence either way.

At replay 4 of the 257-byte, offset-1 case, source, forward slab, peer buffer
and backward slab contain 217 while the returned device byte contains 15.
A subsequent GPU read of the remote slab sees 217. Both channel epochs reach
143 because they span preceding geometries; the diagnostic's expected value 5
is only the local geometry replay index. The original byte oracle remains red.
A terminal observation cannot establish when the remote data became visible.
Per-writer system fences did not repair the actual MI50-pair failure. Separating
acquisition and copying into native kernels also failed, at replay 34 of the
257-byte, offset-1 case. Consumer-only full-L1 invalidation and consumer-only
GLC/SLC payload loads also fail on the actual pair, at replays 3 and 6. Their
emitted gfx906 instructions were checked independently. All four experiments
are removed. The next isolated trial publishes through the slab producer's
SDK-owned HDP register; its earlier failure on the artificial mixed-architecture
cohort cannot establish the result for this actual MI50 pair. Actual peer page
mapping and root cause remain unproved. No production fix or performance result
is claimed for these experiments.

The apparent ordinary-generation reset stall was asymmetric exception teardown.
The tail fixture queried a CPU allocator with no admitted host allocation; its
destructor then entered collective retirement while the other rank entered
request reset. The fixture now reports the original failure before aborting
the original communicator, and tests only resources actually admitted by PMA.
The focused host-authority lifecycle and LocalTP maintenance-ownership tests
pass. The latter removes a lexical hostname test that rejected a valid original
rank-local GPU cohort; exact scope, membership and native capture requirements
remain. Mixed MoE initialization now completes in both native domain orders.

Remaining generation failures are distinct: the preparation exporter rejects
named domains using an obsolete model-context cardinality rule, while ordinary
generation reaches prefill but rejects a valid zero placement epoch. Their
focused authority fixes pass the device-free tests. `rank-generation-and-preparation5`
also passes the complete transaction Unit executable and both original MPI
completion/failure-teardown regressions: 6/10 selected aliases pass, with no skips.
The four native cases expose the next lifecycle boundaries: the global parent
must publish shared weight graph completion after all nested stages are ready,
and a follower must acquire its admission event before closing a response that
emits only the prefill sample. Those repairs are under native qualification;
this does not establish a generation pass.

`rank-preparation-and-native-control1` closes the mixed preparation cases:
CUDA-TP2→ROCm-TP2 and the reverse pass with MTP disabled, fixed and dynamic.
The final fixture correction supplies its actual original world rank during
prepared-stage inspection; it does not relax locality validation. All six
cases retain the original export, layer/participant, PMA and retirement checks.
`pipeline-preparation-local-and-rejection1` also passes all nine local pipeline
and explicit rejection cases in 164.05 seconds, without skips. Together these
batches cover fifteen preparation aliases; they do not certify generation.

`pipeline-generation-native-modes1` closes ordinary singleton generation in
both directions. Setup-only recording no longer increments executed decode
rows, and a completion retains its first actual native terminal proof across
later request reset. A new recording resets that proof before publication.
The two focused host regressions and both ordinary native cases pass, including
two resets, budgets 1/65/1/17, exactly 166 executed rows, and clean retirement.
The next batch exposes two separate gaps: MTP mistakes original pipeline stages
for vocabulary peers, and the TP2 fixture lacks original native prepared workers
required by receiving activation fanout. The installed role now comes from the
actual pipeline service, with strict resident verifier/publication requirements
on its vocabulary tail. The fixture now creates and finally admits its original
native source before constructing collectives, retaining that same PMA.

`pipeline-generation-native-modes2` passes the role regression and both ordinary
singleton cases. `pipeline-phase-accounting-focused1` additionally passes
ordinary TP2 generation in both directions: the shared graph executor now uses
its signature-owned phase so prefill replays do not increment decode work.
The new focused unit exposed a separate cache-hit prelude bug: a valid
device-timeline envelope selected a monolithic prefill stream before execution
selected its common transaction stream. Both now use the same envelope-derived
cache selection, retaining the failing configuration and strict stream identity.
The complete ForwardExecutionEngine unit owner and explicit prefill/decode
accounting regression pass in `pipeline-generation-native-modes3`.

The MTP async ACK incorrectly tried to consume accepted model state as a
host-result bridge. The correction records
only the original producer's native completion event; the actual controller
inherits accepted-state publication through the existing timeline edge. The
focused identity/negative-edge checks and full DeviceExecutionTimeline unit
owner pass. All eight ordinary/dynamic generation cells pass across both vendor
directions and singleton/TP2 stages. The 18-case batch was 13 passed, five failed,
with no skips: four fixed-depth cells rejected the next ContinueResponse because
the follower guard incorrectly required Dynamic, and MemoryPlanner had three
stale test expectations omitting receiving-TP fanout storage. The next scoped
batch, `pipeline-generation-native-modes4`, passes **23/23 with no skips** in
114.5 seconds. Fixed followers authenticate the same pinned depth range used
when preparing their retained verifier family; the source's externally scheduled
first verifier is unchanged. The shared depth-range contract has an explicit
preflight regression. Exact fanout test quotes now include the existing receiving
owners. All twelve ordinary/fixed/dynamic singleton/TP2 generation cells and four
complete relevant Unit owners pass. This is focused evidence, not a complete
preflight certificate.

Production loops are not capped at two pipeline stages. Adjacent same-backend
captured metadata boundaries remain explicitly unsupported by this interface,
so the proposed three-stage native case is CUDA→ROCm→CUDA across the two original
NUMA-owned ranks. It exercises entry and tail on one rank, the middle stage on
the other, and two distinct cross-rank boundary identities. All three new
ordinary/fixed/dynamic cells pass in `pipeline-generation-extended1`. This is
native three-stage component evidence; it does not imply cross-host GPU transport
support or real-model qualification. The complete 47-case extended batch passes
46 cases in 272.7 seconds with no skips. Its one failure is the singleton
CUDA→ROCm cold Dynamic Initial3 case: HIP aborts at the patched
`rocvirtual.cpp:509` check, `Profiled SDMA timing marker lacks valid native timestamps`.
The isolated cell is being repeated and the SDK timing-marker path investigated;
the failure remains open. It reproduces on the third repetition after two passes
in `pipeline-cold-initial3-repeat20`. All penalty and direct-history cases pass.

The resumed host has eight MI50s plus one 7900 XTX on NUMA 1 and two RTX 3090s
on NUMA 0 (boot `14af7953-33ac-4488-aad9-c69f525ddd7f`). The unmasked canonical
ROCm inventory initializes successfully. A manual diagnostic initially retained
`HSA_OVERRIDE_GFX_VERSION=9.0.6`, which incorrectly rejects the XTX; the test
wrapper already removes it. Eight-device ROCm coverage can now be selected by
the canonical hardware admission. This does not change the 2×ROCm + 2×CPU
performance target or manufacture extra socket owners.

The prior 27 repaired host aliases, original-rank activation fanout in both
vendor directions, and prepared binding/metadata cases retain their separate
focused passes. A fresh complete Unit and ProductionTestPreflight remains
required after all current defects close. No throughput result has changed.

## Real-model gate still required

The existing matrix did not contain the exact cross-rank CUDA-TP2→ROCm-TP2 QFN
cell. Adjacent rank-local PP and source-admission cases cannot substitute for it.
A proposed canonical cell uses the same four-part QFN `UD-Q4_K_XL` GGUF and
separate shared-Q8 MTP file, the existing independent reference, and unchanged
numerical limits. It must authenticate actual original rank-local device
ownership before preparation. Required executions are ordinary full-stage
parity, native generation/reset, and fixed depths 1/3 plus dynamic MTP.

Review found two fixture integration gaps. The previous summary required both
embedding and final-logit evidence on every rank, whereas a split pipeline owns
them on different ranks. The diagnostic framework now aggregates completed
scalar observations and indexed actual layer records through existing MPI
collectives. It rejects duplicate, missing, foreign or mismatched observations
and retains every local stage CSV. Empty ranks send no layer payload, only
bounded identity/count metadata. No model/reference hashing is introduced. The generation fixture must
also enter the original root/service protocol instead of starting independent
public request controllers on every rank. Neither gap permits changing the
reference, numerical limits or required layer coverage.

No real-model accuracy, generation or performance result is claimed for that
cell yet. Performance tuning resumes after the complete preflight is green.

## Local evidence and next checks

Root-owned build/run artifacts are under
`/tmp/qfn-full-preflight-after-cpu-next/`; the complete baseline is `run-1/`.
`planning-topology-native1.{json,xml,log}` records the closed planning repair.
`peer-local-gpu-observer-native1.log` separates GPU-local and DMA byte views;
`peer-system-publish-native1.log` records the rejected extra-fence candidate.
`/tmp/qfn-full-preflight-peer-payload-next/` retains exact ISA and investigation
evidence. These generated payloads are not committed to this repository.

`applied-batch-3.json` identifies the reviewed source packets for the cross-rank
foundation. `cross-rank-foundation-build2` passed after three compile-owner
repairs; `cross-rank-foundation-{pure1,native1}` preserve the results above.
`applied-batch-4.json` records the next corrections and the subsequently rejected
HDP candidate. `peer-hdp-build1` compiled and `peer-hdp-native1` failed; that
candidate is no longer installed. `cross-rank-foundation-pure2` records the
focused binding/preparation passes. The first fanout build was interrupted; its
compiler diagnostics also exposed a missing device-pass guard in the shared
validation kernel. The corrected incremental build is `cross-rank-fanout-build2`.
The later `peer-hdp-actual-pair-build1` passed the complete shipped architecture
set. `peer-hdp-actual-pair-diagnostic1` then rejected producer-owned HDP flushing
on the real same-NUMA MI50 pair: replay 5, 257 bytes, offset 1, expected 15 but
returned 217. The two device-free binding checks passed. The entire HDP candidate
was removed; this negative result does not justify retaining additional fences
or changing transport. Fixture ordering and protocol lifetime remain under
independent review alongside physical cache visibility.
`peer-load-observation-native1` supplies direct causal evidence at epoch 143,
replay 4, 257 bytes, offset 1. All four source/destination addresses and epochs
match their original bindings. Forward producer, forward consumer and backward
producer each record byte 217; the backward consumer's actual load records the
previous byte 15 and writes 15. A later terminal remote read observes 217.
The emitted gfx906 instructions prove the same loaded register feeds both the
payload store and the diagnostic record, with zero spills. Thus the observed
failure is a stale peer input read, not a later output overwrite. The temporary
cursor instrumentation was removed after collection; its source/ISA evidence is
under `/tmp/qfn-peer-payload-load-observation-next/`.
The independently registered native control
`V2_Integration_CapturedNativePeerVisibility_ROCm2_257Offset1` passes five warmups
and 64 measured replays with two alternating immutable inputs, the same 257-byte
extent and offset 1, and the original full-capacity byte/guard oracle. The log
records actual MI50 `8c000`↔`91000` **P2P/direct pointer** transport in both
directions. This proves the existing RCCL path on that geometry, not the custom
pull implementation or a new counted-gather composition. The scoped next repair
uses the original homogeneous native domain for the remaining counted gather;
it must retain device-owned live extents, capture, cancellation, lifecycle and
economy regressions before obsolete direct-peer machinery can retire. The
seven-case batch, including this control, passes without skips in 82.55 seconds.
Builds retain the
canonical spill policy and target CUDA 80/86/89/90 plus ROCm gfx906/gfx1100.

The native counted replacement is installed for qualification. Each original
homogeneous domain uses ordered native broadcasts with device-owned live byte
counts, existing native completion receipts, and no host count observation.
Its exact participant metadata quote is `32 + 144 × degree` device bytes and
zero mapped host bytes; packet backing remains with its existing arena owner.
Revocable original-context borrowing protects retained bindings, and direct
receive publication follows validation on the exact stream. An acquired
auxiliary transfer publishes only through its enclosing completion-event join.
`native-counted-and-three-stage-build3` passes after correcting a HIP function
pointer cast and a test-only attempt to call a private publication hook. The
test now observes real event state without overriding publication.
`native-counted-host-contracts1` passes all five selected contracts with no skips;
the synchronization policy and its negative controls separately pass.

Compiled-object inspection is in
`/tmp/qfn-native-counted-allgather-qualified-isa-r1/`. Both new scalar controls
have zero spills, private scratch and shared memory on every configured target.
On gfx906 they use 16/18 VGPRs and 30/28 SGPRs (prepare/validate); on gfx1100,
16/22 VGPRs and 28/26 SGPRs. On sm86 they use 40/23 registers. The native graph
regression also requires an actual one-block, one-thread launch for each control.
These are compilation/resource facts, not completed runtime or throughput
qualification. The obsolete custom peer path remains pending retirement until
its native byte, cancellation, reader and lifecycle coverage is proved.
The first actual ROCm2 replacement replay reaches final binding reset after all
64 payload/count/receipt/guard iterations and direct/fork publication checks.
It then aborts because compiler-generated binding assignment releases an old
tensor owner before its capture lease. The native allocation guard correctly
rejects this order. Whole-binding replacement and an expanded final-owner
teardown regression correct this ordering without relaxing the allocation guard.
`native-counted-binding-lifecycle-build1` passes, followed by all four actual
native cohorts in `native-counted-all-cohorts1`: CUDA2 6.15 s, ROCm2 6.16 s,
ROCm4 9.65 s, ROCm8 19.98 s, with no skips. Each checks retained capture, 64
payload/count/receipt/guard replays, direct/fork publication, revoked context,
self-assignment and final-owner replacement/reset. ROCm8 describes the actual
eight-member selected cohort, including gfx1100; it is not eight-MI50 evidence.
These durations are correctness-suite runtimes, not throughput benchmarks.

Next: close peer payload and original-rank generation/MTP defects, run a fresh
complete Unit/ProductionTestPreflight transaction against the expanded hardware,
then qualify the exact real-model cell before returning to performance tuning. Only
a meaningful green checkpoint is eligible for the authorized feature-branch
commit/push with `--no-verify`.

## Deferred SDMA profiling and consumer qualification

The extended pipeline batch completed 46 of 47 cases, including genuine
three-stage CUDA→ROCm→CUDA ordinary, fixed and dynamic generation. Its single
failure was the cold CUDA1→ROCm1 dynamic-MTP initial-depth-three cell. A fresh
until-fail replay passed twice and failed on its third execution at the same
strict native HIP timestamp check. These results do not qualify real-model QFN.

Pinned ROCr defers gfx906 SDMA packet construction when dependencies are pending.
Its existing completion callback then rereads mutable agent-wide profiling state.
The original HIP timing scope can have ended by then: the copy completes but
its requested timestamp packets are absent. The reverse transition can also add
timestamps to a command that did not request them. The scoped repair retains
the original profiling fact with the already-retained command and uses it when
the existing callback resumes submission. It changes no copy extent, engine,
signal lifetime or HIP timestamp guard. An actual-body host replay preserves
the original failing code as a negative control; only the two policy-transition
cases fail there, while the repaired body passes. The canonical minimal ROCr build and both dependency proofs pass in 38.84
seconds. All SDMA timing/ordering and installed-loader proof cases pass. The
original cold-MTP reproducer now passes all 20 repetitions in 96.06 seconds.
A neighboring mapped-progress executable initially failed with an invalid
device ordinal. Rebuilding its stale backend ABI resolves it: the current
mapped-progress regression passes in 21.04 seconds.

The subsequent consumer/lifecycle batch completed 29 cases: 16 passed and 13
failed. The seven ROCm router/projection failures occur during inventory
selection because old MPI1 registrations omit the genuine ROCm-owning NUMA
rank. Those aliases now launch the two original ranks and use the existing
owner selector; the other rank does not acquire remote GPU ownership.
The mixed MoE lifecycle failures have two separate owners. The two MTP-off
cases incorrectly label their restored-prefix scalar bridge as an MTP condition,
requiring unplanned MTP logits and follower ownership. The four fixed/dynamic
cases instead reject initial shifted-commit snapshots because an original
rank-level collection is compared against local TP-participant cardinality.
Both remain under focused repair; MTP-off must not gain artificial MTP buffers. All three local ROCm1→ROCm1 lifecycle modes
passed. Both vendors' native completed wire and reader-ACK replays passed,
as did ROCm eight-device addressed source-zero and source-seven replays.

Evidence: `/tmp/qfn-full-preflight-after-cpu-next/` contains
`pipeline-generation-extended1`, `pipeline-cold-initial3-repeat20`,
`native-counted-all-cohorts1`, and
`native-counted-consumers-and-pipeline-lifecycle1` logs, receipts and JUnit.
The runtime diagnosis and patch handoff are under
`/tmp/qfn-rocr-deferred-profile/`. A fresh complete Unit/preflight transaction
is still required after these repairs.

The corrected original-owner consumer rerun passes all eight selected aliases
without skips: CUDA2 router, ROCm2/4 routers, ROCm2/4 counted projections,
ROCm shared full rows, and ROCm2/4 phase scheduling. This closes all seven
consumer registration failures from the preceding batch. The largest four-GPU
projection case takes 76.65 seconds; the registration timeout is unchanged.
Evidence: `native-consumer-original-owner-registration1.{log,json,xml}`.

## Applied final focused repair batch

The MTP-off bridge now rejects a missing graph authority, preserves the exact
restored-history check, and submits its admitted ordinary decode graph with
ordinary logits. The global initial shifted commit borrows the checkpoint of
the actual terminal stage from the outer local-stage aggregate. Its nested TP
children and their independent validation remain intact. The host regression
executes actual Global capture/commit for 20 repetitions at widths 1, 2 and 4;
six explicit native aliases cover the actual off/fixed/dynamic lifecycle in both
TP2 domain directions. No MTP-off capacity or additional checkpoint storage
was introduced. The host checkpoint regression and both native MTP-off
directions pass in the first rebuilt batch. Enabled MTP reaches a separate
prefix-frontier defect, described below; its subsequent focused repair is now green.

The custom homogeneous peer-payload implementation and its selector/allocation
API have been removed. Production homogeneous counted communication now uses
its original native NCCL/RCCL cohort. The existing cross-vendor and original-rank
mapped channels remain. The native regression extensions preserve malformed
reader metadata and ACK handling, add 4,099-byte source slices and guards, and
prove cancellation after a real positive payload publication with an absent
reader ACK. Late original endpoints replay after cancellation. Failure cleanup
only aborts and joins; it does not launch additional test protocol work. The
original causal-red custom-peer evidence is retained outside the repository.
The current header/interface changes require rebuilding every affected owner;
`native-retirement-and-pipeline-owner-build1` records the focused build.

The first rebuilt batch (`retirement-and-prefix-focused1`) completes all 35
selected aliases: 22 pass, 13 fail, none skip. Eight failures are a new native
fixture defect: it supplied sample labels as retained-executable submission
ordinals, violating the existing consecutive-per-owner contract. The fixture
now tracks consecutive submissions independently for each retained graph,
including metadata probes and late cancellation replays. One global unit
assertion still expected an obsolete unsupported-topology message. Both repairs
pass their focused rerun, with production guards unchanged. All ten selected
aliases pass in 48.88 seconds. The four additional original-owner controller
publication, arrival and cold-source aliases pass in 14.65 seconds. Evidence:
`native-retirement-ordinal-fixture-focused1` and
`controller-original-owner-followups1`.

The four enabled-MTP failures were **MessageMismatch (status 6)**,
not timeouts. After reset, the upstream stage restores 32 tokens and prefills
64; the vocabulary tail restores all 96. The next scalar activation therefore
encounters different protocol epochs. Original pipeline prefix lookup omitted
its communicator from the existing coordinated lookup, and the upstream stage
was incorrectly required to own the tail's terminal hidden state. The deliberate
protocol-failure trap explains the HSA exception. The repair uses the actual
pipeline communicator and typed required-owner completeness. Physical terminal
state remains local; all stages agree on a locally restorable frontier before
populating it. Missing required tail state remains a rejection. The seven-case
focused rerun passes in 41.14 seconds: full prefix-flow Unit, required-owner MPI
replay, terminal-state contract and both TP2 directions with fixed/dynamic MTP.
Evidence: `original-pipeline-prefix-frontier-focused1.{log,json,xml}`.

## Eight-MI50 pipeline and automatic peer grouping

The requested additional topology is `PP(EP(4×MI50), EP(4×MI50))`. Both local
domains must use their genuine original NUMA owner and native RCCL transport;
the 7900 XTX is excluded by its observed ISA/identity, not by assuming the first
eight ROCm ordinals are MI50s. Qualification remains pending.

Current hardware evidence under `/tmp/amd-p2p-matrix-20261110/` reports all 72
directed AMD paths enabled and validates 64 MiB copies. The MI50s nevertheless
form two four-card PCI-switch groups: BDFs 8c/91/97/9a below PEX86 and
ba/bd/c2/c5 below PEXaf. RCCL uses direct peer transport within each switch and
shared memory between them. Access eligibility alone cannot rank that locality.
Automatic planner work retains ancestor paths from its existing PCI discovery
authority and uses observed mutual access and PCI locality to break otherwise
equivalent admitted-cost choices. It must not invent access for missing facts,
override a cheaper measured plan, or depend on fixed ordinals or group sizes.

The planner implementation builds across the complete configured GPU architecture
set. Its two explicit replay aliases and full inventory unit suite pass. The
original memory-planner run passed 457 of 458 cases; its failure exposed a
distinct pooled ngram storage-only membership gap, not a peer-choice error.
After the native storage-union repair, all 458 cases pass in 45.85 seconds.
Evidence: `automatic-observed-peer-build1`, `automatic-observed-peer-focused1`
and `pooled-native-storage-union-memory-planner1`.
Its ordering retains admitted cost, explicit preferences and compute footprint
ahead of observed mutual connectivity and PCI distance. The new inventory wire
version preserves ancestry and native ISA from the existing discovery authority.
The focused planner regression exercises actual automatic dense PP selection
with sparse, shuffled ordinals and equal proxy costs. It does not yet certify
automatic routed-expert PP or the requested eight-MI50 inference topology.

Both MI50 groups have one physical NUMA owner. Their qualification should start
with the normal two-rank discovery and use the existing execution-rank selection
to retain that owner. Production already excludes inactive discovery ranks before
runner construction; this case needs neither fabricated CPU computation nor a
new idle-pipeline role. Physical UUID and NUMA facts survive the namespace
projection. Native nested-domain transport remains a separate implementation
and qualification requirement.


## Flexible stage composition and remaining qualification

The native pipeline contract is arbitrary positive TP/EP stage widths, unequal
widths and more than two stages, with original rank/NUMA/UUID ownership. NCCL and
RCCL remain inside one original GPU-owning rank. Cross-rank boundaries and CPU
socket domains use the existing MPI/shmemspin authorities. Single-socket ordinary
CPU stages, multi-socket CPU TP/EP and mixed CPU/GPU composition are required;
CPU support is not yet qualified and must not be inferred from GPU tests.

A real policy defect was fixed: pipeline normalization discarded an authored
stage's expert compute/phase/assignment policy, and explicit parent scalar policy
could survive after a child domain was normalized. Both paths now project the
actual child policy consistently. The complete execution-plan unit suite and two
explicit stage-policy/ownership aliases pass after the scalar follow-up, in
5.67 seconds. Evidence: `pipeline-authored-policy-scalar-focused1`.

The installed, not yet qualified native GPU panel uses observed gfx906 selection for `4+4`,
`1+2`, `2+1` and `1+2+4`; off, fixed 1, fixed 3 and dynamic MTP; ordinal/random
ownership; and static/dynamic movement. Process-resident groups amortize runtime
startup while retaining independent admissions, executions and retirements for
every named body. The device-free geometry tests cover larger arbitrary widths.
This host has only two CUDA GPUs: a genuinely nested CUDA GPU pipeline requires
at least three and remains explicitly hardware-unqualified. CUDA endpoint/kernel
and ordinary singleton-pipeline evidence cannot certify that missing full parent.

Pooled ngram transport is being repaired independently. Its native cohort was
incorrectly tied to model-reader membership and its invocation binding required
exactly two members. Storage-only owners need an actual native transport union,
without becoming model compute participants or receiving reader-result payloads.
Completion storage is sized to the actual population through the same PMA;
C0 owners retain required fixed protocol only. The three focused host aliases
pass in 7.16 seconds, including exact native metadata accounting, larger/odd
owner populations and the original planner pressure regression. Evidence:
`pooled-native-storage-union-focused2`; the complete memory-planner result above
precedes the new native pipeline additions. Actual pooled native lifecycle
qualification remains pending.

The native pipeline implementation and fixture built together in
`native-pipeline-domain-and-ep-fixture-build5` (307.11 seconds), across CUDA
80/86/89/90 and ROCm gfx906/gfx1100. Addressed transfers retain original communicator ownership
and exact per-endpoint PMA frames, without mapped payloads or storage-only model
readers. Upstream dynamic expert maintenance is an explicit unfinished mode:
its cadence must use actual committed output rounds, which differ from accepted
KV rows on carry/rejection and response clipping. The positive Dynamic test
cells remain unqualified until that service is implemented and executes.
Full Unit/preflight and real-model inference remain open until all applied
changes rebuild and pass their required gates.

The four focused synchronization/native-allocation source-policy and negative
control aliases pass in 34.48 seconds (`native-pipeline-source-authority-focused1`).
Static resource evidence for the new prepare/validate scalar nodes is retained
under `/tmp/qfn-native-pipeline-kernel-resources-20261110-r1/`. Both CUDA kernels
use 24 registers with zero stack/local/shared bytes on sm80/86/89/90. ROCm
prepare/validate use respectively 5/20 VGPRs and 26/30 SGPRs on gfx906, and
18/19 VGPRs and 22/26 SGPRs on gfx1100; both targets have zero spills, private
bytes and LDS. This is compiled resource/ISA evidence, not runtime throughput
or full native pipeline qualification.

## First component qualification and CPU-stage implementation

`native-pipeline-component-focused1` collected all 12 selected CTest results in
81.20 seconds: five passed and seven failed. The five passing aliases cover the
native-domain contract, observed ISA placement, original ownership, parity
binding and source receipts. The two full host suites exposed four stale
expectations: package labels were being treated as canonical NUMA inventory,
discovery rank count was expected to disappear from the public configuration,
and two planner tests still expected nested GPU pipelines to have no native
transport. Repairs retain the ownership negatives and now check exact endpoint
counts for unequal and three-stage domains.

The native endpoint fixtures incorrectly observed terminal state and released
retained definitions on GPU workers. Failure cleanup then masked the initial
capture error. These operations now run on the original host owner after jobs
join, while native teardown still dispatches to the exact worker. The original
capture diagnostic is preserved. Pooled union fixtures also used a non-MPI test
entrypoint; they now have a dedicated owner using the canonical integration MPI
main. Both repairs await native rerun. The continuous driver observer completed
with zero findings for this batch; these failures do not establish a driver
fault.

The CPU ordinary-stage implementation is applied for focused qualification. It
retains the original CPU rank/NUMA domain and existing MPI/shmemspin transport,
uses a cold-admitted stable host activation bank, and returns unweighted expert
fragments to the existing ordered combine. Metadata/arithmetic tests cover
larger populations; native fixtures use this machine's two physical sockets
plus a singleton tail. Distributed dynamic CPU maintenance, CPU-stage MTP and
mixed CPU/GPU composition remain open. No native CPU-stage result or throughput
claim is made yet. `native-pipeline-cpu-stage-build6` rebuilds these changes with
the component repairs; the previously qualified per-stage scalar policy
projection remains intact.

The corrected combined build passes in `native-pipeline-cpu-stage-build7`
(302.70 seconds). The dedicated pooled fixture also rebuilds successfully after
its constructor/partial-retirement correction in
`pooled-native-owner-retirement-build8` (36.61 seconds). Source synchronization
and allocation-policy checks pass again in
`cpu-pipeline-source-authority-focused2` (four aliases, 18.93 seconds).

`native-pipeline-cpu-host-focused3` passes 15 of 17 aliases in 57.19 seconds,
including complete MemoryPlanner, ModelParityDefinition, ExecutionPlanBuilder
and GlobalOrchestrator owners plus the new CPU-stage metadata checks. Its two
failing aliases expose the same real arbitrary-count defect: after admission's
eight-participant cap was removed, the compact CPU reducer still wrote a fixed
eight-entry stack array, and its common numerical reducer retained another
eight-member rejection. The larger-population regression correctly detects the
first failure. Repair must use the existing numerical reducer with an
allocation-free bank view; no new arithmetic path or per-call allocation is
authorized by this result. Native component qualification has resumed in
`native-pipeline-component-focused4` with the original driver observer.

That native batch completes all six cases in 43.92 seconds, with zero driver
findings and the same boot identity. None passes yet. Both native byte fixtures
now preserve their real initial failure: cold external input events were not
joined to the exact consumer stream before recording. The repair joins only
actual inputs before any parallel capture begins; destination payloads remain
invalid until the native transfer publishes them. The CPU TP stage reaches the
model validator, which still assumes all named domains are rank-local. Its
repair must validate the original cross-rank CPU stage rather than reinterpret
its participants as local devices. All three pooled storage-union cases reach
the same strict owner recipe check: actual native collective stages were being
compared against a mapped-channel recipe. Native library graphs must retain
General executable admission and cannot acquire a fabricated flat kernel-count
certificate. These are separate focused repairs, not evidence of native
pipeline inference qualification.

The arbitrary-count CPU reducer repair replaces its fixed descriptor array
with an indexed borrow of the existing admitted records, ranges and route maps.
Both descriptor and indexed callers execute the same ordered arithmetic and
tile scheduling. The regression now covers populations 2 through 29, raw and
fused output, compact holes, cancellation-sensitive addition, reset/repeat and
a malformed final member. Related placement and cold calibration storage also
derive their sizes from original membership. Build and qualification are in
progress under `native-pipeline-component-repairs-build9`.

## Current focused qualification

The arbitrary-count CPU reducer and placement owners now pass, including the
2–29-member ordered numerical cases. An additional alias regression rejects
indexed source-range metadata overlapping either output before writing. The
complete execution-plan suite and original CPU stage validation also pass;
device-free stage validation covers TP/EP populations through 12 sockets. This
does not claim native qualification on hardware beyond the two installed
sockets.

Both native pipeline byte components now pass: CUDA2 in 7.15 seconds and ROCm4
in 6.52 seconds (`native-pipeline-component-focused6`). The corrected fixture
distinguishes internal recorded publication from the global tensor publication
made only after the actual retained graph is submitted. It preserves strict
invalid-input checks, exact-stream event joins, live byte extents and repeated
empty/partial/reset execution. These component successes do not yet certify
complete PP generation.

The three native pooled storage components now pass their original owner recipe
check but time out during cold recording. Inspection identifies serial
`submitAndWait` recording: the first native collective participant enters before
the remaining participants have been dispatched. Cold collective recording and
launch submission need the original participant group to make concurrent host
progress. The mapped-channel regression remains green, with 20 retained
repeats in 17.06 seconds. Native-library graph admission is being kept separate
from mapped flat-graph certificates.

The real dual-socket CPU test reaches memory admission and exposes another
rank-local assumption (`cpu-stage-original-inference-focused7`). The input
adapter rejects global CPU TP before the caller attaches its already-validated
original CPU programme. The focused repair supplies that same borrowed typed
programme at the input boundary, verifies original rank/NUMA/member/shard and
stage scope, and preserves the generic rejection of unauthenticated global
domains. Native CPU generation remains unqualified until the repaired test
actually reaches and completes inference.

All scoped driver observations through `pooled-native-cuda-stall-driver7`
complete with zero findings and the same boot identity. The current evidence
identifies host composition and admission failures; it does not establish a
new driver fault. Complete Unit/preflight and real QFN generation remain open.

The next focused build and host batch pass: build16 takes 170.50 seconds;
`pipeline-maintenance-owner-host-focused9` passes all 16 selected aliases in
79.44 seconds, including complete MemoryPlanner, ForwardGraphTypes,
MTPSpecStateContract, GlobalOrchestrator and pooled remote-owner host suites.
The actual two-socket cold-failure child preserves its original cause before
aborting the same parent communicator, and its session guardian proves owned
process/file retirement. These results qualify the host contracts, not native
maintenance execution.

The first real eight-MI50 static group reaches graph materialization and fails
there (`native-mi50-pipeline-static-focused10`, 62.06 seconds, driver observation
complete with zero findings). Native tail TP siblings receive the primary's
hidden-state bank because serving-plan reprojection was restricted to
heterogeneous boundaries. The existing strict participant-local bank check
detects this. The repair must project each original participant's own bank for
every typed domain boundary. Later failures in the same process follow the
failed initial admission; independently, the dynamic-MTP fixture requests a
retained capacity of three without limiting its adaptive maximum to three.
Neither issue permits weakening ownership or retained-capacity validation.

## Native isolation and current failures

The focused eight-MI50 serving-bank regression now passes
(`native-mi50-serving-banks-focused12`, 35.43 seconds including the driver
observer). It uses the actual PP(EP4, EP4) topology and ordinary admission;
it does not certify decode or MTP. The driver observation is complete with
zero findings and the unchanged boot identity.

Concurrent original-worker capture removes the earlier pooled-storage stall
on CUDA2, ROCm2 and ROCm4. In `native-pooled-union-after-cold-fix-focused14`
all three complete the first, all-zero-capacity layout's fifteen captured
lookup trajectories, cancellation/failure handling and physical retirement.
The second independently prepared source is then correctly rejected by the
process-retained PMA root. Each immutable positive/empty layout now has its
own exact MPI2 CTest process lifetime. Both layouts still require qualification;
successful execution of the first layout did not make these aliases green.
The device-free nine-member capture fixture likewise passed its assertions but
exposed a fixed eight-ordinal teardown sweep. It now retires its declared
source roster, including the ninth participant, through the existing reset
contract.

The original MI50 wildcard process-resident groups had the same independent
source-lifetime error after their first passing body. They are retired rather
than weakening source ownership. All 128 existing exact preparation/lifecycle
bodies are registered separately in production preflight, using their existing
MPI session guardian. Exact `NativeISA-gfx906`, device-count and ordered-stage
labels feed the existing observed-hardware qualification. Seven MI50s plus a
7900 XTX cannot certify an eight-MI50 case; eligible cases must run and absent
hardware remains explicitly unqualified. The focused bank alias is retained.

Running seven of those exact PP(EP4, EP4) static cases in independent process
lifetimes (`native-mi50-static-original-lifetimes-focused16`, 179.14 seconds)
exposes two production failures rather than secondary source-adoption errors:

- Ordinary decode reaches the complete follower parent, then attempts both
  acquisition and release through the same one-use epoch `Submission`.
  The lifecycle rejects the second publication. The repair must retain strict
  one-use state transitions and publish the actual captured terminal event.
- Fixed depths one/three and dynamic MTP materialization require an empty
  committed-state transport graph on terminal-domain nonleaders. They have no
  actual outgoing metadata work. Their cold original-cohort participation must
  remain ordered, but no empty GPU graph or dummy operation may be created.

All seven cases were allowed to finish; none is certified yet. The continuous
driver observation again completed with zero findings. The dual-socket CPU
capacity repair is separately under build: it exchanges only actual original
rank contributions through the shared CAP4 codec, solves one complete stage
plan, and projects local ownership. Its negative tests distinguish certified
capacity pressure from malformed input, which must not select a smaller row
candidate. Complete Unit/preflight and real QFN generation remain open.

`original-capacity-and-native-lifetimes-build19` completes successfully in
131.21 seconds. The complete MemoryPlanner owner and all five focused host/MPI
capacity, cold-capture and maintenance aliases pass in
`cpu-capacity-cold-capture-host-focused18` (45.77 seconds). The isolated native
pooled components are now all green in
`native-pooled-isolated-lifetimes-focused19` (49.16 seconds including driver
observation): CUDA2, ROCm2 and ROCm4 each pass both positive/mixed-owner and
all-zero-capacity layouts. The observer reports zero findings. This qualifies
the component's repeated captured lookup and retirement, not reuse of its
native recordings across the full retained model-source family. Actual CPU
pipeline inference and the two native pipeline fixes are the next focused gates.

The actual dual-socket CPU run first exposed two original-owner assumptions:
request publication rejected an embedding stage shared across sockets, and
stage binding compared a tensor-sharded owner map against the deliberately
empty scalar ownership arm. Request publication now uses the same original
stage leader as GlobalPP transfers, accepting nonzero leaders and unsorted
unique memberships. Joint fragment banks retain exact original map identity;
banks from a second setup are rejected.

The hosted-follower repair uses separate one-shot acquire/release submissions
around the complete captured parent and its actual terminal event. Its typed
claim rejects child/release interleavings while the native launch runs without
the host mutex. Publication preparation now gives terminal siblings a local
commit role while they join the original native begin/end phases on their
existing setup streams. They retain their own banks without an empty graph.

`original-owner-publication-and-lease-build21` passes in 45.10 seconds after
updating two test helpers to supply their existing explicit setup stream.
`original-owner-publication-host-focused22` passes all six selected aliases in
44.90 seconds, including complete MemoryPlanner and epoch lifecycle suites.
The four actual CPU cases (`cpu-stage-original-owner-inference-focused23`,
9.20 seconds) now pass these admission/ownership checks but fail at the factory's
rank-local graph communicator guard. The CPU global-stage graph-context wiring
remains under repair; this is not a native inference certificate.

`native-mi50-publication-and-lease-focused24` passes all seven eight-MI50
static cases in 173.63 seconds including observation: ordinary, fixed-one,
fixed-three and dynamic MTP lifecycles, fixed-three/dynamic preparation, and the
focused fixed-one publication preparation regression. The observer is complete
with zero findings and the same boot identity. This closes these two failures
on the actual PP(EP4, EP4) fixture. Random ownership, dynamic balancing, the
remaining topology matrix and full real-model QFN qualification remain open.

The CPU follow-up repair preserves the authored dense domain and borrows the
already admitted stage split for the dense graph. Sparse/request coordination
retains its original parent. The graph-context guard remains unchanged; no
new communicator or hardware discovery is introduced. A separate bounded
CUDA/HIP proof now imports one original graph-only transfer child into two
distinct source parents and one receiver parent, with exact input/output,
partial/zero-byte, receipt and lifetime oracles. Build22 is qualifying these
changes. Full model-source archive integration follows only after that native
proof succeeds; this is not yet a complete preflight or real QFN certificate.
The [native wire archive design](QFN_NATIVE_WIRE_ARCHIVE_DESIGN_2026-11-10.md)
records the scoped production integration and its separate qualification steps.

Build22 succeeds for CUDA 80/86/89/90 and ROCm gfx906/gfx1100 with the spill
guards intact (685.34 seconds). The focused CPU domain and shared-child lifetime
host replays pass in `cpu-dense-and-native-child-host-focused27` (6.44 seconds).
The actual split-view/cold-abort regression also passes in
`cpu-dense-communicator-inference-focused26`; the four ordinary CPU generation
cases expose two later failures: apportioned weight materialization exceeds its
declared primary-weight plan, and the legacy static-range factory rejects the
already-authored tensor-sharded fragment path. Those are separate repairs; no
capacity reserve or generic rejection is being weakened.

`native-shared-child-frontier-focused28` passes the original direct CUDA2 and
ROCm4 byte replays but fails both shared-child fixtures (2/4, 33.32 seconds).
The fixture's added consumer copy calls an API that publishes an external event
inside capture. The correct external-event guard rejects it on both backends.
The fixture must use the existing capture-aware transfer/dependency path before
this can establish shared-child runtime correctness. The continuous observer
again records zero driver findings on the unchanged boot. The complete CI
registration owners also pass (`native-child-registration-python-focused25`,
2/2 in 9.15 seconds).

The corrected capture-aware fixture passes both native shared-child cases in
`native-shared-child-fixed-focused29` (18.31 seconds with observation): CUDA2
7.02 seconds, ROCm2 5.50 seconds, zero driver findings. Distinct source parents,
partial/empty payloads, post-child reads and retained-definition retirement all
execute. The CPU graph-scope replay initially retained the fixture's old rank0
owner after changing its declared members to {1,3,4}; that inconsistent fixture
is corrected to name its actual first member. ConfigValidator remains green.

The user replaced the 2×MI50+2×CPU performance goal with QFN
PP(EP(4×MI50), EP(4×MI50)), ngrams sharded in VRAM across all eight cards.
Full Unit/preflight remains a prerequisite. The existing Q7 real-model source
admission covers RAM/mmap only; fixed GPU sharding currently requires a single
full-model TP cohort. QFN's sole main PLE layer means only one pipeline stage
queries the table. The other stage may hold source shards but must not receive
unused PLE results. A fixed-source service on the existing captured pipeline
parents is under design; the larger adaptive wire archive is deferred for this
target. Real-model generation, numerical and benchmark cells must be added to
the canonical matrix before reporting Q7 accuracy or throughput.

## CPU original source and host-boundary follow-up

The original CPU TP source-window repair passed its full WeightManager sharding
owner plus explicit original-source, final-cache and graph-scope preflight
aliases: host31, 4/4, 2.90 seconds. It retains the planner's admitted slices
and native quantized bytes; no capacity reserve was added. The subsequent
actual two-socket ordinary cases (UPI/MPI, apportioned/tensor-sharded) moved
past that allocation failure and exposed two further boundaries:

- The CPU fixture entered OrchestrationRunner directly without RuntimeInitPhase's
  explicit CPU backend initialization. It now binds and asserts the actual
  original inventory NUMA node before construction. Workspace access still
  refuses to invent an aggregate backend.
- Dense-only preparation of already committed CPU joint fragments was entering
  a GPU-only guard. The CPU arm now authenticates the complete original bank;
  missing or partial banks cannot be repacked by the dense pass. The existing
  CPUJointBoundedSourceLifecycle preflight regression verifies unchanged
  engines, source reads and PMA claims.

After CPU backend initialization, both apportioned cases reached actual prefill
and rejected a flattened host activation tensor. GlobalOrchestrator had the
correct physical capacity but exposed a one-dimensional shape. It now retains
`[admitted_rows, residual_width]`. HostedPipelineResidualMatrixLifecycle tests
actual parent materialization, remote receive and local handoff, ordinary and
widened residuals, and stable identity across shrinking requests. Build26 passed;
host34 passed all three selected tests, including the complete GlobalOrchestrator
unit owner (2.18 seconds).

The next actual CPU run exposed hosted decode ordering incorrectly forwarded
through GPU vocabulary-tail authority, stale TP assignment inherited by a
singleton stage, and joint CPU root/producer declaration mismatches. Hosted
ordering now reaches every original local stage, singleton stages publish their
complete one-participant assignment before weight materialization, and canonical
joint storage retains its admitted root layout. Build28 passed; host36 passed
all six selected ownership/lifecycle and complete GlobalOrchestrator and
StageRunnerFactory tests (4.20 seconds). Native37 then passed both apportioned
UPI/MPI ordinary request/reset cases. Both tensor-sharded variants still fail
at the first rank-batch dispatch because the original joint routes deliberately
name a cohort rather than a scalar destination. That dispatch repair is pending.
The four-case result is 2/4, not a full CPU or preflight certificate.

A nearby audit also found that hosted greedy sampling with penalties requests
remote tail authority on workers before the required token broadcast. The
repair leaves logits mutation with the real tail and keeps non-tail CPU ranks
in the existing broadcast. Its device-free ownership regression and an actual
greedy-with-penalties request in each original CPU case preserve stochastic
replay. Build29 and focused38 pass all three selected cases (5.81 seconds).

The rank-batch dispatch repair authenticates the exact retained CPU shard
cohort, layer, tier, epoch and target rank before broadcasting each route to its
actual fragment owners. Only nonzero routes and their compact hidden rows are
packed. Missing leases, foreign target ranks, and mixed whole/cohort routes
remain errors. Build30 passed; focused39 passed all five tests in 10.45 seconds:
the 20-repeat ownership/compact-packet regression and all four actual two-socket
UPI/MPI × apportioned/tensor-sharded ordinary cases. The latter each complete
four requests and 20 generated tokens, including stochastic repeatability,
greedy penalties, partial prompts, resets and final model/native retirement.
These close the focused CPU blockers. The next run is the complete canonical
Unit/ProductionTestPreflight transaction; no real Q7 numerical or performance
claim follows from these small-model fixtures.

The active eight-MI50 GPU-sharded goal uses the
[approved fixed-resident plan](QFN_FIXED_VRAM_PIPELINE_PLAN_2026-11-10.md) and
[service/forward-role map](QFN_FIXED_RESIDENT_PIPELINE_SERVICE_2026-11-10.md).
Planning/loader, captured transport, and canonical model registration are being
implemented in private parallel patches. They are kept out of the prerequisite
source until this repair slice completes its gate.
