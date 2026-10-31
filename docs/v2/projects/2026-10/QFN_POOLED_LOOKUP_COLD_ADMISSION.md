# QFN pooled lookup: cold admission and ownership regressions

Investigation and implementation record, 2026-10-31. Current matrix status belongs
in [the bounded admission dashboard](QFN_SOURCE_ADMISSION_MATRIX.md). This record
does not qualify distributed tiering, inference accuracy or throughput.

## Installed path

Single-GPU adaptive lookup now uses the separate pooled owner, coordinator and
source stages in the ordinary QFN graph. The former combined adaptive kernel,
root owner, fixture and performance entrypoint have been removed. Public
VRAM/RAM/SSD placement, original UInt64 row identities and the production
controller remain authoritative. The planner and materializer consume the same
eight physical roots; a zero-sized VRAM cache still has mandatory protocol
storage. Optional cache slots add only their declared physical storage.

The model graph contains participant-local stages. This change does not enable
distributed ownership or native PP by replacing their missing implementation
with a local path. Those remain separate positive admission obligations.

## Cold arena failure

The first integrated source run attempted all 16 original GPU-adaptive cases:
CUDA/ROCm, zero and positive VRAM/RAM cache sizes over SSD, and MTP
Off/fixed1/fixed3/Dynamic. Every case failed with:

> QFN storage seal requires ordinary prepared native addresses

The phase-stage constructor sealed its ID operand before the ordinary arena
had materialized its GPU allocation. The source-read constructor also required
an already resident output and declared that output preallocated. Existing
component fixtures uploaded both operands first, concealing the cold-start
ordering defect.

Constructors now validate declarations, geometry and original ownership.
`DeviceGraphExecutor` retains its existing allocation boundary, and the stages
seal physical addresses and retain original allocation leases at their actual
`CaptureOnly` preparation boundary. Source-read output is an ordinary graph
output. No allocation, transfer, synchronization or alternative executor was
added to stage preparation. Once sealed, signatures authenticate the original
backend, address and runtime generation; capture and execution require those
same retained bindings.

The new native replay leaves the ID and output tensors without GPU addresses
through graph construction and contract/signature queries. It then uses the
production executor to materialize and capture them. It covers 16/64/256/1024
IDs, zero and positive cache capacity, partial/empty work, repeated preparation,
request resets and retained allocation lifetime. An original-stage control
fails on the 3090, MI50 and 7900 XTX at the former seal, while the corrected
stage passes on all three.

Explicit production-preflight registrations:

- `V2_Integration_PooledImmutableColdOperandDeclarationReplay`
- `V2_Integration_CUDA_PooledImmutableProductionColdArenaSealReplay`
- `V2_Integration_ROCm_PooledImmutableProductionColdArenaSealReplay`

Original integrated failures: `parity-results/qwen-flash-next/`
`source-admission-pooled-original-full15-r21/`. Private actual-stage positive
and original-code controls: `/tmp/qfn-pooled-admission-seal-r1/native-r1/`
(CUDA) and `native-r2/` (MI50/XTX). The first private ROCm attempt retained a
CUDA UUID mask and exposed no HIP devices; it is retained as invalid diagnostic
evidence, not a kernel failure or a qualification result.

## Cancellation and terminal observations

An acknowledged source response can be followed by cancellation before the
owner publishes Ready. The source graph now reauthenticates the original
device controller after acknowledgement/fill and before that publication edge.
Cancellation preserves the original source-consumption witness and drains the
existing grants; it never publishes the reserved row as a usable promotion.

The native regression substitutes the actual old unchecked authorization at
that one boundary. The old control publishes Ready and a promotion, while the
production path rejects it. Reporting snapshots actual work before grants
retire and accumulates the terminal counters only after original
source/owner/coordinator retirement. These observations do not decide policy.

Both backend `PooledImmutableACKReadyCancellationReplay` registrations and
the provider/source replay suite belong explicitly to production preflight.
The suite covers all 23 supported GGUF formats and original live byte counts.

## Supporting lifetime and metadata corrections

- `DeviceGraphOrchestrator` is a fixed-address owner. Its forward engine retains
  a reference to the original executor, and its cold-family lifetime has an
  ordered teardown. Moving the object could invalidate those relationships.
  Object copy/move operations are now forbidden; moving its `unique_ptr`
  preserves the address. `DeviceGraphOrchestratorStableOwnerLifetime` tests
  both contracts through the actual type and owner.
- `PhysicalMemoryPlan::resources()` is an lvalue-only borrowed view. Returning
  the view from a temporary plan previously allowed a dangling iteration in
  tests. The actual interface rejects that expression, and callers retain the
  plan. `PhysicalMemoryPlanResourceViewLifetime` is a focused preflight entry.
- An empty compiled GPU kernel may legitimately use zero registers. Native
  graph inspection accepts that result while still rejecting failed queries
  and negative signed register counts. `GPUGraphKernelMetadataReplay` checks
  this contract; the actual ROCm original-code cancellation control supplies
  the zero-register native case.

## Evidence boundary

The combined Integration rebuild passes with CUDA SM80/86/89/90 and HIP
gfx906/gfx1100. Static extraction of the actual source-kernel objects reports
zero register spills and zero stack/private/local memory on every target.
The two new metadata nodes use one block of one thread: BeforeReady uses
16 CUDA registers and 3 AMD VGPRs; ObserveRetired uses 40 CUDA registers,
5 gfx906 VGPRs and 19 gfx1100 VGPRs. These are resource measurements, not
achieved occupancy or model-throughput claims.

Separate MI50 profiler invocations measured the first cold C0/N64 dispatch:
BeforeReady took 1.44 microseconds and ObserveRetired took 8.00 microseconds.
Each executed one wave with zero scratch and LDS; actual register allocation
was 4/32 and 8/80 VGPR/SGPR respectively. Both driver windows were clean.
These single-dispatch diagnostics do not establish model throughput.
Evidence is `resource-profile-r5/qualified-scope-summary.json` beneath the
integration directory below. CUDA counter and activity instrumentation produced
NVIDIA `pSmIssueThrottleCtrl` assertions and remains unqualified; subsequent
unprofiled native replays passed with clean driver windows. No driver setting,
kernel arithmetic or test-health allowance was changed to accommodate profiling.

`/tmp/qfn-pooled-production-integration-r1/` retains build, patch and exact
source evidence. Its `component-gates-r3-result.json` records 20 device-free,
28 CUDA/MI50 and 14 XTX focused regressions passing without skips, with clean
driver checks, unchanged source and final-owner/MPI retirement. The complete
845-test Unit run found only a missing `DeviceFree` label on the new fixed-
address ownership registration. After correcting that label, both canonical
inventory checks and the affected lifetime registration pass in
`/tmp/qfn-admission-integration-r19/pooled-production-registration-r6/`.
The metadata-only rebuild performed no compilation or linking.

The original 16-case source rerun passes **16/16**, with no skips or failures,
in `parity-results/qwen-flash-next/source-admission-pooled-original-full15-r22/`.
The final `batch-result.json` records 1,814.64 seconds, unchanged runtime/source
files, a clean driver window and complete retirement. It retains all 48 layers,
the full active MTP capacity of 15 and the original source/topology identities.
Admission does not execute the controller or a numerical request.

The complete integrated r30 Unit/model-free preflight passes: 3,263 passes, zero failures and 29 unchanged hardware-unqualified cases.
Exact membership, source/runtime identity and the final clean driver window
are authenticated in `/tmp/qfn-admission-failure-integration-r30/r30-audit.json`.
Distributed source transport, native-domain composition and the complete
source-owned NCCL/RCCL preparation factory remain open.
