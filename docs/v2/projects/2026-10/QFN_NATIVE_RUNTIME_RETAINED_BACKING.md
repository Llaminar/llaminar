# Native runtime backing during QFN source admission

This is an ownership investigation and implementation plan, not a tuning dashboard.
It records the boundary that must be qualified before the new GPU ngram source
admission path can receive production opaque-resource limits.

## Finding

Successful native handle destruction and a quiescent device-reset receipt do not
prove that all physical backing was released. On the tested ROCm 10.0 / MI50
closure, runtime-owned queue and allocator backing survives `hipDeviceReset`.
The original PMA obligation must survive with that backing. CUDA and ROCm must
use the same typed ownership model, with source-qualified physical lifetimes.

The diagnostic observes the actual original worker constructor: one primary
stream, BLAS and BLASLt handles, and 32 ordering events. The original mapped
worker stack and object remain separately charged exact roots. Its interposers
record native calls but do not admit a production resource or supply a limit.
External NVML / amdgpu sysfs observations do not load CUDA/HIP in the observer.
After reset the child makes no device or memory query that could recreate a
context. Process RSS, GTT and VRAM are observations, not a second memory ledger.

## Initial measurements

Values are physical-device MiB, rounded here; retained JSON preserves bytes.
These are diagnostic observations, not installed budget constants.

| Boundary | RTX 3090 CUDA | MI50 ROCm |
|---|---:|---:|
| Before backend lookup | 452.750 | 10.363 |
| Startup metadata ready | 455.625 | 10.406 |
| Original worker ready | 728.938 | 176.496 |
| Original worker destroyed | 718.938 | 150.496 |
| Exact host root released | 718.938 | 150.496 |
| Physical runtime reset complete | 455.625 | 150.496 |

The ROCm residue stayed unchanged during a 3.6-second passive observation.
This rules out an immediately delayed telemetry update in that experiment; it
does not attribute every remaining byte to a specific private runtime object.

The independent read-ahead experiment separates explicit payload backing from
side-stream overhead:

| ROCm boundary | VRAM MiB |
|---|---:|
| Original worker ready | 176.500 |
| Explicit payload allocated | 178.500 |
| Side stream created | 178.512 |
| Ordering events created | 178.512 |
| Graph executable instantiated | 178.645 |
| Graph executable destroyed | 178.516 |
| Side resources and payload destroyed | 178.516 |
| Runtime reset complete | 152.516 |
| Control rejected before side-stream creation, after reset | 152.500 |

The 2 MiB increment belongs to payload allocator backing in this experiment,
not to the side stream. The side-resource increment is about 12–16 KiB. Both
remain after reset. Keep original worker activation, explicit allocator backing,
side resources and graph executables distinct; do not move unexplained bytes
into whichever family has spare budget.

## Source evidence

The configured ROCm source pin in `scripts/docker/rocm-release.env` is
`6b0e43f341195e203754e08f850e437ff2fc09f9`; the installed TheRock manifest names
the same rocm-systems revision. The local extracted source is not a Git checkout.
Native diagnostic records additionally authenticate the actual loaded ELF files
and compiled creator source, including project changes. Optional ELF build-ID
notes may be absent; complete regular-file content identity remains required.

In `projects/clr/` of that source:

- `hipamd/src/hip_device_runtime.cpp`, `hipDeviceReset`, delegates to the HIP
  device reset implementation.
- `hipamd/src/hip_device.cpp`, `Device::Reset`, releases HIP pools and streams,
  purges objects, and recreates pools. The underlying ROCclr device survives.
- `rocclr/device/rocm/rocdevice.cpp`, ordinary queue release, decrements pooled
  queue references. CU-mask and cooperative queues have distinct destruction.
- The ROCclr device destructor destroys the ordinary queue pool at runtime
  teardown, after the application's lexical GPU cleanup boundary.

Default queue population, priorities, CU-mask/cooperative behavior, system/device
queue placement, CU count, kernarg and staging policy affect the qualified class.
A device UUID identifies the live owner; it is not a portability allowlist.
The default gfx906 AQL ring resides in system memory, so its logical size cannot
be used to explain the observed 2 MiB VRAM increment. An older HSA source tree
does not authenticate the current loaded HSA allocator implementation.

## Target ownership

1. The closed, source-qualified opaque certificate declares either context-
   generation or process-runtime physical lifetime. Callers cannot select an
   arbitrary lifetime or budget.
2. Before native creation, the existing process authority retains the original
   PMA obligation for process-runtime backing. No second balance, free-memory
   subtraction, copied charge or replacement PMA is introduced.
3. Reset retires native handles and releases context-generation obligations only
   after the existing validated retirement pipeline. It closes generation use
   of a process obligation without releasing that obligation.
4. Where an actual SDK root supplies an authenticated destruction callback, its
   existing `ProcessLifetimeV1` token releases at that callback. Where opaque
   backing exposes no such callback, the original obligation remains owned until
   OS process termination. Returning from the application's main-frame cleanup
   cannot publish a false zero-accounting receipt.
5. Rebirth may borrow a qualified bounded process population only through the
   same original PMA/controlblock, source/runtime class and physical endpoint.
   Execution generation and handle identity are new. Concurrent use must obey
   the declared population, and an independent fresh PMA cannot adopt old roots.
6. Original capacity observations and mandatory source BOM precede optional
   expert/ngram quotas. Loader success must prove actual consumption of the
   prepared worker; a merely bound source is insufficient.

Explicit payload allocator retention remains a separate lifetime/accounting
question. It must not be silently reclassified as read-ahead stream storage.

## Focused validation and remaining gates

The raw diagnostic now has an explicit reset-between-lifetimes mode. It retains
the original Preparing PMA, destroys each actual worker/root, executes the
original reset API, pauses for external observation, and only then creates the
next requested worker. Its ordinary warm-lifetime mode is unchanged. Protocol
replays reject omitted resets, reordered boundaries, incorrect generations,
intervening probes, incomplete teardown and silently changed policy.

The ROCm campaign passed twenty reset/recreate cycles with twenty uses per
lifetime, plus one cold-process control, in 300 seconds with a clean driver
window. Reset VRAM was 150.500, 150.511719 and 150.523438 MiB in the first three
cycles, then remained byte-identical through cycle twenty. GTT was 72.925781,
92.957031 and 112.988281 MiB, then also remained byte-identical. The plateau
supports a bounded retained population in this tested closure.

After the child process exited, VRAM returned to 10.363281 MiB and GTT to
12.824219 MiB, exactly the pre-backend baseline. Process exit and device reset
therefore have measurably different physical retirement boundaries.

The matched CUDA campaign also passed twenty reset/recreate cycles and one cold
control, in 304 seconds with a clean driver window. Every reset returned VRAM to
the 455.625 MiB startup value, and process exit returned it to the 452.750 MiB
pre-backend value. These experiments are not full ngram latency benchmarks or
model admission passes, and do not by themselves install production limits.

Device-free ownership regressions must cover original-token survival through
reset and lexical shutdown, context-only release, same-PMA borrowing without
duplicate charge, wrong identity/authority, partial construction and repeated
retirement. Each defect regression belongs explicitly to production preflight.

After those gates: install reviewed class limits; run the existing CUDA+CPU and
ROCm+CPU adaptive zero-cache and bounded-cache source admission cases; then
extend through the authored MTP and expert-placement matrix. Actual dynamic
expert movement still requires the original Qwen3.6MoE controller, movement
journal, compact-transfer and parity witnesses. Admission alone does not prove
multi-device inference or its performance.

## Integrated implementation and qualification boundary

The joined source now retains attempted process-runtime context births in the
existing process owner, including failures before final model admission. It
keeps the same source binding, PMA pointer/controlblock and original opaque
commitment. Exact mapped stack/object roots still retire at the authenticated
reset boundary. Independent replacement authorities, duplicate publication and
foreign live owners reject before native creation. Source identity captures the
original environment before metadata discovery; physical visibility selectors
remain in live identity without making a tested resource class UUID-specific.

The joined Release build passed for CUDA 80/86/89/90 and ROCm gfx906/gfx1100.
Fresh joined CPU replay passed 776 tests: 196 QFN/read-ahead, 72 PMA, 26 resource
lifetime, 31 channel binding, 401 MemoryPlanner and 50 source/placement/discovery
tests. The three delayed/fatal channel cleanup cases also pass 20 repeats. The planner
population includes its eight translation units added by later CMake
`target_sources` calls, including the actual graph-arena admission check. No
MPI sessions remained. Earlier 329-planner-test evidence covers only the base
executable declaration; it is not a complete planner target pass. Process-owner
publication/borrowing and original source/loader-policy regressions also passed
20 repeats on the preceding coherent core. Full current Unit/preflight and
real-model source qualification remain open.

The public declared-cache policy keeps storage GPUs separate from model compute
and expert domains. Source preparation includes their original worker roots;
per-owner ceilings and authored order enter layout, wire identity and the final
PMA plan. Those declarations do not yet provide recurring remote execution.
The directed two-channel owner and its exact mandatory/slot-growth BOM are
integrated. Retained participant transactions still need to be joined to the
original generation controller. Original dynamic expert balancing,
movement journals, generations and live payload extents remain required.

The raw measurement executable exports native creator interposers. A public
birth-key query inside that executable can therefore identify the diagnostic
executable as the BLAS symbol provider. Such a key cannot qualify the ordinary
production provider. The proposed direct raw-key projection was withheld before
native use. The uninstrumented key/source-lifecycle target now authenticates
actual configured BLAS/Lt providers and their ELF contents on both backends.
Both metadata-only native probes pass with clean driver windows. Interposed raw
observations remain separately labeled and bind original provider/source evidence. No production lookup may
search past an unexpected provider to hide that mismatch.

A process-retained source also does not currently authorize loading a fresh
independent model authority in the same process after retirement. Supporting
that requires an explicit new model epoch using the original PMA and retained
obligations. Rejecting an independent replacement prevents lost charges but is
not a completed model-reload implementation. Bootstrap runtime inventory and
explicit payload allocator residue remain separate qualification boundaries.

## Current native and lookup-event work (2026-11-01)

The coherent Release r17 build is green. The probe initially lacked direct
ROCm BLAS links; linking the exact configured providers fixed that diagnostic
executable without changing its production source key. The source-key probes
passed on CUDA and ROCm, proving that no diagnostic interposer or existing pool
worker supplied the observed production key. Protocol replay and the narrow
native-preparation source guard also pass.

Fresh CUDA and ROCm creation/use/reset each pass 20 warm lifetimes plus a cold
control, with 20 uses per lifetime and clean driver evidence. CUDA's maximum
observed VRAM/RSS increments were 286,588,928/127,041,536 bytes. ROCm's maximum
VRAM/GTT/RSS increments were 180,477,952/105,021,440/35,610,624 bytes. Physical
device usage returned to its external baseline after process exit. These
observations do not automatically install limits.

The reviewed conservative class limits are 256 MiB host on each backend,
384 MiB device on CUDA, and 256 MiB device on ROCm. The explicit worker stack,
guard and object remain separate exact roots. RSS and GTT can overlap; their
sum is a conservative envelope, not an asserted disjoint allocation census.
These limits are installed against the final public source identity
`e52b5725d5a8c5a60b0141cbaceb6eb5690257ed33ce19c1c35b51f236b3f4dd`.
The whole-source lifetime is `ProcessRuntime` on both backends. CUDA VRAM
return alone cannot prove opaque host reclamation, particularly with raw-probe
bookkeeping and reusable malloc arenas present. The actual uninstrumented
source-lifecycle probes now pass on CUDA and ROCm: original preparation,
same-PMA/certificate admission, immutable original-key alias, 20 event-family
replays, authenticated native reset and process exit. Exact preparation storage
is zero after reset; original process charges remain 640 MiB on CUDA and
512 MiB on ROCm, including host and device commitments. Selected-device VRAM
returns exactly to the external process baseline, with zero driver findings.
This qualifies the selected-worker population, not lookup/graph/payload or
real-model admission. The first probe exposed an invalid diagnostic read of
`reservedBytes` during Preparing. It now projects canonical `rankAttestation`
in both phases; the actual source/worker/admission regression passes 20 repeats
on both backend identities without weakening the production phase guard.

Event destruction also had a concrete accounting defect: four native cleanup
methods could return after a failed device selection or destruction call.
They now terminate before the caller can retire the associated commitment.
The exact production bodies have a device-free replay and original-code
negative controls, explicitly registered in production preflight. Joined
native lifecycle checks pass. Graph executable/graph destruction had the same
early-return risk and now also fails fatally before releasing ownership.
The 21 event and four graph cleanup tests each pass 20 repetitions against the
extracted current production bodies.

The baseline lookup census also found events outside the existing payload BOM:
one setup event per pooled owner/coordinator/source group, one lazy canonical
publication event for each nonempty tensor in that group, and one submission
boundary event. The exact root bitmap determines the count. The legacy complete
cohort additionally creates one remote submission-frontier event per remote;
the new recurring companion must not add that event. The focused repair prices
the existing closed initialization programs and keeps their original ordering
and tensor coherence semantics. No table or graph bytes may pay these costs.

Current evidence is under `parity-results/qwen-flash-next/ngram-lookup-perf-20261101/`
(`production-key-{cuda,rocm}-r1`, `context-birth-{cuda,rocm}-reset20-r1`) and
the final `original-source-{cuda,rocm}-r17` probes plus
`original-source-qualification-r17.json`. Joined evidence is in
`/tmp/qfn-read-ahead-root-integration-r1/` (`coherent-release-build-r17`,
`linked-cpu-r16`, `native-cleanup-replays-r16`, `source-probe-pma-replay-r17`). Full current Unit/preflight,
real-model admission, remote inference and economy remain unqualified.

## Lookup program qualification and precreator cancellation

The r19 joined Release build passes across CUDA80/86/89/90 and
HIPgfx906/gfx1100 (108 seconds). Fresh isolated original-source, ordering-group,
root, geometry and actual-program owner replay passes **19 tests ×20** with no
MPI session residue. The complete canonical planner target freshly linked to
r19 passes **406/406** with zero skips and clean MPI session cleanup. The last
fixture now acquires/freezes the actual recorded original worker and same
Preparing PMA before asserting the specifically unqualified side-family program;
it also verifies the unchanged original key alias. The prior joined failure is
retained as a negative control; the repaired case passes twenty repetitions.
This is focused evidence, not a current full Unit or production-preflight pass.

All fourteen raw ordering runs pass: **33 lifetimes and 660 program uses per
backend**, including C0, positive and mixed layouts, twenty repeated resets,
and successful partial-construction prefixes. The original 32-event worker
baseline remains separate. A positive local program adds exactly twelve events
(owner4, coordinator4, source3, boundary1). Maximum measured extra heap use is
9,936 bytes CUDA and 19,248 bytes ROCm; extra VRAM/GTT is zero. CUDA RSS/arena
increments are 12,288/16,384 bytes; ROCm RSS is 4,096 bytes. Native reset, process
exit and driver windows pass, with zero process-exit VRAM deltas. These observed
increments include diagnostic bookkeeping; they are neither exact allocations
nor automatically fitted limits.

The reviewed catalog now has twenty-four lookup rows and four selected-worker
classes, covering the original and clean loader environments independently.
Actual original-source setup passes **4/4**: CUDA/ROCm × C0/positive VRAM caches,
with the original RAM/SSD source, two initialization lifetimes each, same-PMA
admission, actual byte checks and validated native reset. Every process exits
with zero GPU memory delta on all observed devices and no MPI session residue;
the driver window is clean. Receipt: `lookup-ordering-qualification-r21.json`.
This certifies those original source and fixed event programs, not complete
process accounting, captured lookup, model inference or throughput.

The r22/r23 joined slice installs owner-controlled graph submission and authenticates
every recording mutation against the original worker, generation and PMA. A failed
native recording cannot be retried, and a retained capture reference cannot submit,
reset or recapture behind its owner. Local ProviderReplay programs now declare their
complete immutable cases before the one original source admission; the former fake
capacity and independent-PMA constructor is removed. Closed factory fixtures exercise
the same materializer with sealed original claims rather than bypassing public
installed-source selection.

The complete shipped-architecture Release build passes (r22: 129.56 seconds), followed
by the catalog-only r23 rebuild (6.71 seconds). Fresh production-core tests pass
Owner32, focused owner2×20, Read6×20 (19.23 seconds), and ProviderPlan1×20, with no MPI
residue. The graph-owner replay exposed a test-economy defect: its 6+4 fault loops
re-exec the canonical threadsafe death-test process separately for each case, taking
about 64 seconds against the unchanged 30-second registration. Assertions pass before
the timeout. Consolidating the actual fault prefixes into one child per case retains
all assertions and passes **9×20** against the joined production core in **24.66 s**.
Each case now has its own 30-second, 20-repeat, threadsafe preflight registration;
the executable/CTest inventory checker rejects missing or duplicate cases, weakened
repeats and the retired broad gate. Its ten negative/positive controls pass.
This is not a green complete preflight claim.

The actual diagnostic target builds coherently in r24 (**38.69 s**, unchanged
production core `4939fce6...`). A read-ahead terminal-order regression exposed a
newly queued marker being observed after an earlier join: publication now precedes
the same existing join, followed by storage completion and readiness observation.
No synchronization was added. The consumed helper passes **3×20** fresh replays,
including the old-order pending-marker negative control and storage failure.

Initial CUDA ordinary lookup completes all byte checks and its real reset with
zero process-exit VRAM delta on all seven cards, clean driver diagnostics and no
session residue. Its resource report remains red: combined stdout/stderr corrupts
buffered JSON, fixed by preserving separate streams; then the inspector correctly
rejects timing-event destruction whose plain native creation was not yet observed.
The initial missing API observations were repaired in the r25 successor.
Preserved artifacts: `actual-program-r24-cuda-ordinary` and
`actual-program-r24-cuda-ordinary-separated` under the 20261101 ngram results.

The r25 observer forwards the exact plain event, graph-create and graph-clone
APIs and records thread/call ancestry. Nested SDK forwarding cannot double-count
one handle or authenticate an unrelated alias. Its 13 recorded controls pass,
including the original failing inspector; the coherent diagnostic rebuild passes
in **5.42 s** with unchanged production core. Canonical observer and QFN admission
CI-policy Unit/preflight entries pass **4/4**, and the selected registration,
source-discovery and ownership-policy entries pass **5/5**.

Fresh original-source CUDA/ROCm diagnostics pass **18/18**: ordinary/read-ahead
M1 controls; read-ahead C0, M32 partial cache and M512 all-hit cases; and injected
failures at each of the four executable-instantiation positions. The ten successful
executions validate all native output bytes. All eight injected failures stop before lookup,
destroy exactly the successful native prefix and publish the real source reset.
Every case has zero post-exit VRAM delta on all seven physical devices, a clean
driver window and no temporary/session residue. Ordinary M1 creates one stream,
65 events and four definitions/executables; read-ahead adds exactly one stream
and three events on both vendors. No failed r24 observation was relabeled.

Receipt: `actual-program-r25-complete-prefix-campaign/result.json` in the 20261101
ngram results; its original 14-case predecessor remains unchanged.
These are **instrumented component observations**, not installed opaque limits,
clean throughput, all-format/full-model accuracy or whole-process admission.
They cannot qualify full-forward/conditional/MTP host graph populations from a
small lookup graph. CPU worker and cross-vendor original-source integration remain
open; all observed host/TLS instrumentation overhead stays in the raw evidence.

The r28 successor closes the actual flat graph inventory gap. Interposed runtime
queries alone missed CUDA's driver-level type inspection and several smaller
roles. The explicit diagnostic now queries each immutable definition immediately
before its original instantiation and emits every node kind and default edge.
The offline validator authenticates graph, process, thread and instantiate
ordinal, requires a complete DAG, and rejects missing/duplicate/foreign nodes,
cycles and child/control/host/allocation kinds. It adds no production query or
new resource allowance. Cold SDK queries and temporary inspection storage remain
included in the diagnostic's observations.

Both vendors have identical observed kernel-only flat shapes:

| Geometry | Lookup nodes/edges | Terminal | Bound-rejection | Small lookup |
|---|---:|---:|---:|---:|
| M1 ordinary/read-ahead; M32 partial | 58/57 | 1/0 | 3/2 | 58/57 |
| M1 C0 | 54/53 | 1/0 | 3/2 | 54/53 |
| M512 | 93/92 | 1/0 | 38/37 | 58/57 |

The full **18/18** actual-program sweep passes again with these inventories,
including every failed instantiate prefix; bytes, source reset, driver findings,
all-device post-exit VRAM and owned-file retirement remain green. Receipt:
`actual-program-r28-inventory-campaign/result.json`. The template's four compiled
device-free tests pass twenty replays; the offline observer has **22** controls,
and three selected canonical Python gates pass. Exact C++ Unit/preflight entries
exist in the full generated inventory; focused C++ bodies were linked against the
unchanged Release core, not the pending complete Integration core. The successful
diagnostic rebuild took 2.20 seconds. Its first build command requested a
Unit-only target from the Release performance selection and was rejected before
compilation; that failed r26 receipt remains separate.

These shapes still do not price pre-API C++ graph-owner allocations or certify
arbitrary General/full-forward/MTP programs. Those known objects and opaque SDK
backing need their actual original creator population and separate attribution.

Lookup event counts, flags and native creator/retirement bodies remain unchanged;
the independent equivalence review is `/tmp/qfn-legacy16-equivalence-review.json`.
The 24 existing rows retain their numeric limits and ProcessRuntime obligations
under reviewed auxiliary source `2a94731e...`; the public worker source remains
`e52b5725...`. Fresh native CUDA/ROCm C0/positive setup passes **4/4** in 8.97 seconds,
with two actual initialization/reset lifetimes per case, zero process-exit VRAM
delta on all seven observed devices, clean driver findings and no MPI residue.
Receipt: `lookup-ordering-qualification-r23.json`. General graph host backing,
side/parent families and CPU service stack/libc/TLS remain separately unqualified.

The bounded CPU repair is specified in
[Retained CPU service accounting](QFN_RETAINED_CPU_SERVICE_ACCOUNTING.md).
Both consumers can declare their exact immutable program before quota allocation
and materialize it after the same authority admits it. They do not need a CPU
thread to obtain GPU source observations, so no Preparing capability, phase change
or preparation ABI extension is required. Exact CPU0 stack/object storage and
separately qualified libc/TLS obligations keep distinct lifetimes; concurrent
source-I/O and executor-ticket workers remain separate.

The first r19 attempt correctly rejected a different runtime policy: its runner
cleared inherited `LD_LIBRARY_PATH`, while earlier qualification retained it.
Clean-policy key comparison found identical source, hardware, loaded providers
and worker geometry; only environment policy/live identity changed. Both clean
worker measurements remain inside the existing reviewed ceilings. The clean
lookup extension passes **21 lifetimes and 420 uses per backend**, including
twenty warm resets. Largest extra heap use is 9,840 bytes CUDA and 19,328 bytes
ROCm, with zero extra VRAM/GTT and zero process-exit VRAM delta. No key field or
ownership check was weakened, and no limit was fitted or raised. Immutable
reviews are `257e8322...` (worker) and `07d665e1...` (ordering). The r21 Release
rebuild passes with unchanged public worker/auxiliary source identities; fresh
class-selection, read-ahead-family and ordering replay passes **39 tests ×20**.

An additional integration-build defect was found and repaired: the event-cleanup
replay registered a creator source using `tests/v2/../../src/v2/...`, while the
core fingerprint used its normalized path. Ninja rejected duplicate phony
outputs before compilation. Normalize the event module's root once. The actual
CMake module prefix now has a compiler-free Ninja replay, source-freshness check,
and original-code negative control. Five focused tests pass and have explicit
Unit/preflight registration. The actual complete Integration catalog configures
and its process guardian builds; inventory, NUMA isolation and both dependency
registrations pass. This does not claim a complete Integration binary build.

A remaining CPU population is explicit: `ImmutableLookupService::hostBytes`
currently declares the mapped channel, while its persistent
`RetainedParentTicketServiceWorker` starts a separate `std::thread`. Its stack,
compiled owner and libc/TLS backing require their own source-owned CPU0 plan;
GPU-worker and graph allowances cannot cover them. The bounded repair must
preserve the existing ticket state machine and affinity, claim exact mapped
stack/object pages before thread creation, and join/unmap before release.
Opaque libc backing remains a distinct version-bound reviewed contract. The
same worker's `DeviceGraphExecutor` consumer must be included. This remains open
and prevents a claim of whole-process admission completeness.

The joined cancellation repair holds original PMA reservations pending until
`beginNativeAttempt` transfers those same tokens into the original retirement
slot. Discarding an uninitialized actual root returns capacity immediately;
a failed native creator retains its commitment until certified physical
retirement. The old production body fails the focused actual-root regression;
the joined three root and five group/source tests pass twenty times. Both
focused defects have explicit preflight entries. Immutable old-code/candidate
evidence is `ordering-unattempted-reservation-r1/` under the current evidence
root; joined replay is `/tmp/qfn-read-ahead-root-integration-r1/ordering-replays-r19/`.

The CPU1 audit distinguishes placement from physical allocation. `CPUBackend`
allocates through CPU0 per rank; NUMA placement remains a separate policy.
Canonical planning now rejects independent CPU1 physical banks while preserving
NUMA1 placement. Adaptive replanning forwards the original typed ordering-event
contribution exactly once alongside table/staging costs. Source Preparing and
PMA code remain unchanged; the original successor control fails all three
focused cases, and repaired original-source cases pass twenty repetitions.

## Local evidence

Native evidence root:
`parity-results/qwen-flash-next/ngram-lookup-perf-20261031/`.

- `context-birth-cuda-smoke-r4`: native creation/use/reset and driver window pass.
- `context-birth-rocm-smoke-r1`: same API gate passes; retained backing observed.
- `context-birth-rocm-postreset-observation-r1`: passive reset hold.
- `read-ahead-resource-external-rocm-r1`: payload/side-stream attribution and
  partial-creation controls.
- `context-birth-rocm-reset20-r1`: repeated reset/rebirth experiment; its
  `physical-memory-summary.json` retains exact-byte plateau and process-exit
  observations.
- `context-birth-cuda-reset20-r1`: matched CUDA reset/rebirth experiment with
  the same compact physical-memory summary.

Coherent Release build and focused replay receipts are retained under
`/tmp/qfn-read-ahead-root-integration-r1/`; these are local diagnostic evidence,
not a published complete Unit, preflight or full-model qualification.
