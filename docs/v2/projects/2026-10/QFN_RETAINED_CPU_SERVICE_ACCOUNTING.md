# Source-owned retained CPU service worker proposal

## Scope correction

The user confirmed on 2026-11-01 that the accounting investigation had expanded
too far. The delivery priority is working QFN continuations, native lookup
correctness and latency. Keep exact PMA ownership for explicit buffers, mapped
worker stacks and objects. Use conservative, typed **tested
allowances** for opaque libc/driver metadata, with representative actual-consumer
measurements and bounded-growth/cleanup regressions. Do not require exhaustive
TLS allocation, loader-path or proprietary allocator audits before delivering
the feature. Do not add another ledger or invent an unmeasured allowance.

Application-source hashes must not select or invalidate admission allowances.
Remove their generated CMake fingerprints and runtime matching. Routine graph
edits do not require native-memory requalification. Preserve the existing tested
resource families, actual declared geometry and owner lifetimes; use source
revisions only to identify test evidence. Native ABI support belongs to the
configured loader, not exact provider hashes in an allowance table. Earlier
source/compiler/build-bound proposals below are superseded by this instruction.

The follow-up correction removes the empirical sixteen-worker, 64-module and
1 MiB TLS ceilings. They describe measured coverage, not supported population.
Each declared service receives the conservative planning allowance
`max(2 MiB, round_up(declared_TLS_bytes + 1 MiB, 2 MiB))`, using existing cheap
declaration-time TLS metadata with checked arithmetic. Current mapped stack,
guard, page and compiled object sizes travel through the existing rank-pricing
descriptor and remain exact. A 184-byte to 192-byte object change therefore
does not invalidate an opaque family; crossing a page boundary correctly adds
one page to the explicit BOM. Local creation authenticates the original
declaration, and PMA funds the declared population without a second count cap.
This policy passes the r53–r55 focused family/geometry regressions and both
actual installed consumers in Release and Integration. Full-model continuation
admission has progressed past the corrected transfer-buffer BOM and failed-start
GPU retirement, flat remote graph preparation and cold worker binding. Native
transfer capture classification remains open, tracked in
[the allowance review](QFN_NATIVE_RESOURCE_ALLOWANCE_REVIEW_2026-11-01.md).
Complete Unit/preflight and model qualification remain open. The historical
measurements and earlier rejected designs below are not the implementation.

The detailed investigations below preserve earlier evidence and design context;
they are not an instruction to expand opaque-memory certification. Further
changes must fix a concrete admission, execution or retirement failure. In
particular, finish only affinity work required by the existing real consumer,
and reuse the current diagnostic targets rather than building another framework.

## Historical investigation and superseded proposal

The r41 focused refresh passed all 19 selected Unit/preflight registrations, including the actual IO worker and retained-parent ticket consumer bodies, portable family/live-change rejection and diagnostic selector controls. Six fresh CPU processes also passed: two identity checks and one 21-lifetime constructor/work/join run per role, with no exact claims or session files left behind. Earlier r39 coverage was 48 Unit and 46 preflight registrations; it is historical broader evidence, not a current complete gate. The r40 mixed two/sixteen-worker growth observations remain representative population evidence.

The superseded r41 experiment hashed four runtime-provider files (5,306,472 bytes), down from 26 libraries (1,236,161,256 bytes), and reduced cold evidence construction from 2.86 seconds to 23–25 milliseconds. Ordinary declaration now performs no provider-file hashing and has no empirical module-count or TLS-byte eligibility ceiling. The historical timings are diagnostic setup observations, not model throughput benchmarks. Evidence: `/tmp/qfn-read-ahead-root-integration-r1/focused-r41/result.json` and `/tmp/qfn-read-ahead-root-integration-r1/host-service-native-r41/result.json`.

## Actual consumers and unchanged behavior

* `execution/qwen4exp/ImmutableLookupService.h` retains one `PreparedRetainedHostService` and a reference to its worker. `ImmutableLookupService::create` preserves `ScopedOpenMPPartitionAffinity` and consumes the original admitted worker token. `hostBytes(layout)` still prices mapped channel storage; the separate pre-quota declaration prices the worker's stack, object and runtime. The worker remains present for C0/all-hit service intervals.
* `execution/local_execution/graph/DeviceGraphExecutor.h` also owns a `PreparedRetainedHostService`. Setup installs its admitted executor-ticket role before capture. `prepareRetainedParentTicketServiceWorker` now authenticates presence and idle state; it cannot create a late worker. Ordinary policies create no executor-ticket worker unless their actual submission path requires one.
* Preserve `Idle -> Queued -> Running -> Complete -> Idle`, exact monotone Ticket, task/result/error semantics, stale-ticket rejection and destructor drain. Keep the actual `run`, `dispatch`, `await` decisions unchanged. Change only birth/storage ownership and its cold wiring.

The I/O and executor-ticket roles must remain separate when both can run concurrently. A GPU parent can wait for source I/O while another CPU service handles a ticket; one shared thread would change progress and risk deadlock. Counts come from actual service/executor construction declarations, never GPU storage-owner count or number of MTP graph variants.

## One authority and concrete API seam

Declare an immutable `RetainedHostServiceProgram` and its typed mandatory BOM before freeze/optional quota allocation. Materialize exact CPU0 storage and the thread only AFTER final same-PMA admission, before capture. Neither current consumer needs a CPU thread to collect native GPU observations; there is no identified pre-admission creation dependency. No new `OriginalSourcePreparation` capability, Preparing claim, NativeStoragePreparation ABI arm, or phase widening is needed.

Proposed narrow types:

* `RetainedHostServiceProgram`: closed roles `ImmutableLookupIO` and `ExecutorParentTicket`, compiled object geometry, original pthread defaults, creator-affinity policy, immutable source/runtime/TLS identity and declared concurrency. No arbitrary byte/count constructor.
* `RetainedHostServiceProgram::bom(originalCPU0)`: exact mapped object/stack contribution plus separately qualified libc/TLS contribution, authored once before quotas. The same immutable declaration is retained through admission; materialization does not reprice it.
* `PreparedRetainedHostService::materialize(originalAdmittedPMA, program, constructorArgs)`: consumes a noncopyable admitted use token, authenticates CPU0/program/original authority, reserves original typed BOM commitments, maps/prefaults exact storage, constructs the actual worker and starts its pthread. Missing opaque qualification rejects before native creation.
* `RetainedHostServiceUseToken`: sealed move-only per-instance ownership prevents duplicate concurrent materialization. Request reset reuses the retained thread/task slot.

The actual lookup planner declares its I/O role and executor planning declares its conditional ticket role BEFORE final admission. Both consumers receive those exact declarations and the same admitted authority. No GPU context/provenance API, fake GPU generation, late free-memory probe, existing-worker adoption or second mandatory ledger participates. A CPU-only admitted fixture can exercise the same materialization API.

## Exact storage versus libc/TLS

For OS page size P, original stack S, original guard G and concrete object size/alignment O/A:

* Checked mapping extent is `G + S + round_up(O,P)` (or separate exact object and stack roots if that keeps the existing object owner cleaner).
* Guard G is an exact virtual PROT_NONE range, not claimed as writable physical backing.
* Physical reservation/materialization after admission is `S + round_up(O,P)` on original CPU0, prefaulted and owned before `pthread_create`.
* `pthread_attr_setstack` binds that root; original scheduling/inherit policy, affinity, detach/joinability and supported signal-mask defaults remain preserved. No use of `pthread_attr_init` that drops process defaults.
* After actual successful join, destroy the compiled object, unmap exact roots, then release their original claims. Join/unmap failure is fatal before release. Allocate all bookkeeping before native creation; publish the successful handle with no throwing allocation afterward. An escaping entry exception terminates.

Both consumers now retain a typed prepared owner handle. The actual worker object remains the same compiled class/state machine. This avoids mapping an entire DeviceGraphExecutor or changing unrelated thread families.

Libc may allocate TCB/DTV/TLS and retain allocator/native backing after join. Price that as a DISTINCT source/version-bound opaque host requirement, not stack padding, a GPU SDK allowance, graph allowance or a free-memory subtraction. Start with an empty reviewed catalog. A missing class rejects before pthread creation, not a fallback to std::thread. Root may conservatively choose ProcessRuntime after measured retention; exact mapped S/O still retires at join/unmap.

Static TLS/TCB that libc places inside the already mapped user stack is part of exact S, not a second opaque charge. Prefault/map S/O before the calibration baseline, then observe additional creator/use/join backing. Do not infer proprietary/private pthread field sizes or subtract a late free-memory sample. The reviewed opaque scope covers only backing outside the already priced root (including any retained libc allocator/TLS structures).

The class binds libc/libpthread provider identity/version, source/compiler/build, actual loaded PT_TLS module closure and `p_memsz/p_align`, stack/object geometry, native defaults and affinity-policy geometry, declared simultaneous thread population and creation/use program. Snapshot loader metadata under `dl_iterate_phdr` and hash selected immutable files afterward. Preserve exact live policy validation; portable class fields must not become per-host CPU-index allowlists accidentally. No numeric limit is proposed here.

The opaque CPU process obligation must use a distinct CPU0 host-root arm of the original process-entry lifetime owner. `NativeContextRetirementSlot` is GPU-generation-specific and must NOT receive a fabricated GPU/device/generation for this thread. The process arm stores ORIGINAL reservation/source tokens only (no byte ledger), survives every GPU reset, and uses the same global original-PMA/control-block guard. Independent-source re-admission remains rejected while retained backing lives unless an explicit supported same-authority epoch is implemented. No process-owner rename or wholesale lifetime redesign is necessary.

## Reuse without GPU impersonation

Factor only the proven low-level pthread mechanics used by PreparedGPUWorkerThread: deep original-attribute snapshot, exact mapped stack binding, create/join wrapper and fatal cleanup. A private generic `PreparedOwnedPthread` accepts a sealed mapped-thread-storage capability; the existing GPU wrapper and new CPU service wrapper use it. Preserve existing ordinary GPU-context behavior explicitly. Do not route CPU work through PreparedGPUContextBirth, fabricate a DeviceId::cuda/rocm or create a fake pool generation.

Exact mapped roots use normal admitted CPU0 typed claims and the existing mapped allocation/publication authority. Add only a distinct named host-service owner if required to distinguish this BOM; do not add a Preparing creator-family or NativeStoragePreparation ABI arm. CPU0 remains the only physical host bank; NUMA is execution association. The additions must reject CPU1 physical-bank declarations, GPU allocators, forged anchors, duplicate roots and phase/lifetime mismatches. Do not relax general geometry or phase checks.

If low-level extraction or appended owner/ABI changes a public birth-source fingerprint input, list the exact source diff and rekey/recheck the actual source lifecycle. Do not relabel e52 evidence. Source callback/ABI tests and original GPU prepared-thread tests must remain green.

## Consumed call paths

1. Runner/source planning derives immutable host service declarations from actual selected lookup arms and executor parent policy. This happens before the mandatory BOM is frozen and optional quotas are allocated; it creates no thread or mapped root.
2. Actual ModelContext/loader consumes the I/O role when materializing ImmutableLookupService and retains its owner. Existing source/channel/cache/table authority remains unchanged.
3. DeviceGraphExecutor configuration receives its admitted executor-ticket declaration and materialized owner before capture materialization; `prepareRetainedParentTicketServiceWorker` becomes an authentication/idle check, not a thread creator after admission.
4. Both use the same original source CPU0 observation/PMA. The two prepared owners are distinct when concurrent. MTP variants share their proven persistent executor owner instead of multiplying by graph count.
5. Teardown retains task/context until actual await/drain, joins the original worker, destroys/unmaps its exact storage, then releases the original exact claim. Only the original process owner releases a ProcessRuntime libc/TLS commitment at actual process retirement.

For the first bounded native fixture, declare one I/O owner per actual service and no executor-ticket worker unless its real policy creates one. Preserve the current ScopedOpenMPPartitionAffinity by retaining its immutable execution-affinity policy at planning, and authenticate the actual creator/use mask. Do not temporarily rebind unrelated threads just to collect observations. Serving keeps the source I/O worker alive across requests; no per-request pthread creation.

## Focused tests and qualification

* Actual unchanged ticket-state replay with prepared storage: dispatch/await, stale/duplicate/outstanding tickets, reset/repeated20, exception task result, destructor drain with borrowed context.
* Exact CPU0 stack/object root, guard inaccessibility, complete default attrs and modified supported affinity/scheduling control with restoration; no CPU1 physical relabel.
* Partial native prefix: mmap/prefault/object construction failure, pthread_create failure (including possible TLS backing), creation success then later constructor failure, failed join death, failed unmap death. Original tokens remain retained after an attempted opaque creator; never release merely because no handle was returned.
* Same-PMA/control-block/program/compiled-object/affinity/source identity negatives, duplicate slot materialization, missing CPU0, Preparing materialization rejection, missing mandatory contribution rejection and unchanged final resource set.
* Both ACTUAL consumers exercise the owner: service source/cache/root creation, and executor RequireRetainedParentWithConcurrentTicketService. Assert ordinary policies create zero executor-ticket workers; assert concurrent I/O+ticket roles use distinct stacks/owners.
* Source/BOM test before quota and final same-PMA admission, exact allocation deltas and join/unmap retirement; no opaque bytes reported as exact mapped storage.
* Explicit Unit and focused ProductionTestPreflight registrations. Device-free replay uses sealed recorded source/opaque input only; actual native calibration has no installed numeric override.
* Process-isolated libc/TLS calibration using the same real constructor/task/join program, external pre-API host/process snapshots, exact mapped S/O separated, first/lazy/repeated20 creation/dispatch/join and process-exit observations. Provider full-program observer independently labels this population from General graph, activation/capture and Timing resources.

## Implementation order

A. Ownership/API/BOM scope reviewed; freeze the exact creator inventory with the implementation.
B. Implement immutable CPU service program/BOM and admitted-only owner with empty libc/TLS selector. Reuse existing low-level pthread code without extraction if practical; factor only when necessary, preserving original GPU behavior and explicitly rekeying affected source.
C. Consume in both service and executor through the original planning/admitted authority; remove unprepared production creation paths after all callers/tests migrate.
D. Freeze code/ABI/source, run focused replay/preflight and shipped-architecture compile, then root measures the exact CPU program and reviews conservative opaque limits.
E. Run the actual prepared local C0/positive and concurrent executor+I/O native gates, then pair real model source admission with numerical/movement qualification. Dynamic Qwen35MoE/Qwen3.6 policy/controller/journal remains unchanged; this thread slice cannot certify dynamic continuation by itself.

Affinity implementation detail: preserve the DEEP original pthread attributes and perform creation inside the same admitted ScopedOpenMPPartitionAffinity scope as today's service factory. An explicit process-default pthread affinity can override ordinary creator inheritance; do not silently replace that precedence with a computed intersection or force the IO mask. The immutable plan records the original defaults and the declared creator-scope policy, and regression compares actual child behavior with an original std::thread control under both default and modified supported attributes. No physical CPU1 bank is introduced by an execution mask.

## Joined implementation and current evidence

The original service declarations now pass through ordinary, ExpertOverlay and
pipeline planning before optional quota selection. Automatic discovery exchanges
bounded role and identity receipts for each original rank; execution-rank
compaction preserves that mapping. Candidate metadata does not create a worker.
Actual local binding authenticates the selected declaration against the same
admitted PMA, and both existing consumers require its move-only use token.

The earlier r32 Release build completed in 254.1 seconds with an unchanged source
snapshot and the complete configured target set: CUDA 80/86/89/90 and ROCm
gfx906/gfx1100. Its core SHA-256 is
`ac924b0ea2a6e2188bfad877079f80bf8e73aab8df3bcb1cfb14b79f109bdcfa`.
This is historical compile evidence, not qualification of the current source or
native qualification of all those architectures.

Six fresh CPU diagnostic processes passed: two identity observations and one
21-lifetime constructor/work/join run per role. Each lifetime dispatched and
completed 20 tasks through the production ticket worker. Each role retained a
stable portable class across its fresh processes. Its exact 8,388,608-byte
stack and 4,096-byte object mapping were fully released after join; the virtual
guard remained separate. The process guardian reported no leftover sessions.
The artifact deliberately reports `qualified=false` and
`whole_process_bytes_complete=false`: allocator samples alone do not certify
opaque libc/TLS backing. The installed CPU class catalog remains empty for both
roles.

Local evidence is retained under
`/tmp/qfn-read-ahead-root-integration-r1/build-r32/result.json` and
`/tmp/qfn-read-ahead-root-integration-r1/host-service-native-r33/result.json`.
The r38 coherent Integration refresh passed in 36.8 seconds. All 41 selected
Unit registrations passed in 25.1 seconds; all 33 remaining focused preflight
registrations passed in 179.2 seconds with a clean driver observation. No selected
test was missing or skipped. The read-lifecycle registration preserves twenty
replays per case, partitioned into five-replay jobs under the unchanged 30-second
timeout. Earlier green cohort coverage includes 1/2/4/8 participants. These are
focused results, not the complete prerequisite gate. Current records are
`integration-build-r38`, `focused-unit-r38`, `focused-preflight-r38`, and
`focused-preflight-r38-window` under the same local evidence root.

The explicit-VRAM planner now omits unreachable hosted-mode contributions while
retaining its actual upload-geometry admission checks. A separate confirmed
duplicate-binding defect now uses a typed PMA instance lease: two bindings cannot
spend the same worker declaration against another role's aggregate budget. The
lease follows the worker through join/unmap and byte-claim retirement. Its new
focused preflight replay covers concurrent bindings, actual separate ModelContexts,
failed claims and independent shared-owner control blocks, and is green in r38.
Mixed participant-policy validation now precedes optional pricing, preserving a
precise rejection before any unreachable hosted-mode lookup. The final expected
CPU total includes the original executor worker's declared storage and runtime
BOM rather than a separately invented allowance.

The joined correction authenticates the actual `__tls_get_addr` provider.
The loader has no PT_TLS segment, so the previous passive-module census omitted
the code that allocates DTV/TLS storage. Its full immutable provider identity and
cheap current-provider validation now participate in the same original evidence.
The existing diagnostic target also supports concurrent populations and a
zero-native claim control. The coherent r39 build and all 48 Unit/46 preflight
registrations passed; the r40 observer-unwind fix and receipt replay also passed.
No numeric limit was installed. The small native batch completed the mixed-two,
mixed-sixteen and zero-native-control cases in 3.34, 5.10 and 3.39 seconds
respectively. Sixteen simultaneous workers claimed exactly 134,283,264 bytes;
every join/unmap checkpoint returned exact claims to zero. Typical warm retained
heap increments were 208 bytes per created worker, consistent with the retained
process bookkeeping; the allowance must include this rather than subtracting it.
Evidence is in `host-service-growth-r40/result.json`. These Integration diagnostics
are not Release performance or full-model qualification.

The earlier diagnostic exposed a cold evidence cost: full-file passive TLS closure
authentication read 1,235,924,640 bytes from 25 shared libraries per fresh process.
The narrowed plan removes that incidental closure from the portable admission
class. The existing creator/source version, relevant runtime providers, explicit
geometry and tested policy limits remain the contract. The optional OpenMP Inputs
refactor is preserved privately and is not required: no real-consumer affinity
rejection was demonstrated, and existing creator/restoration behavior passes.
