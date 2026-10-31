# QFN admission: native readiness and discovery repairs

Investigation record, 2026-10-31. Current overall qualification belongs in the
[bounded admission dashboard](QFN_SOURCE_ADMISSION_MATRIX.md). These repairs
do not qualify distributed ngram execution or complete native preparation.

## Coordinator readiness

The complete r29 Unit/preflight transaction finished with 3,240 passes, one
failure and 29 unchanged hardware-unqualified cases. The failure was
`V2_Integration_RCCLUnusedRetirement_DirectOwnerRetiresTwentyUnusedCliques_2Devices`,
which timed out during repeated unused-communicator retirement. Source remained
unchanged, and the passive driver window recorded no faults.

Both original NCCL and RCCL worker loops contained two host races:

- They published initialization completion before setting `running`. An owner
  could return from initialization, request shutdown, and have that stop
  overwritten by the worker's later unconditional `running = true`.
- They changed the condition-variable initialization predicate outside its
  mutex. Notification could pass between the waiting owner's predicate check
  and its transition into the wait.

The existing queue mutex now protects publication of `running = init_success`
followed by initialization completion. Notification follows the unlock. The
later unconditional running-state write is removed. Shutdown uses that same
mutex. No retry, timeout increase or alternate communicator path is added.

Device-free regressions execute extracted, unmodified production lifecycle
methods and actual state declarations under deterministic mutex/wait schedules.
They cover stop-before-worker-continuation, both success/failure notification
gaps and repeated lifecycle reuse on both backends. Eight Unit and eight explicit
`ProductionTestPreflight` registrations execute precompiled replays; extraction
and compilation happen only during the build. Original-source controls fail the
intended schedules. The native CUDA twenty-lifetime test complements the
existing ROCm direct and pooled-owner tests.

The races are reproduced defects. No native thread backtrace was retained at
the r29 timeout, so the evidence does not exclude a different native cleanup
stall as a contributor to that particular timeout.

## Discovery failures cannot become capacity facts

ROCm enumeration previously ignored errors from device selection, memory and
primary-context observations, restoration/release, and peer queries. Some
failure paths substituted total VRAM for available capacity or represented a
failed peer observation as an absent edge.

Enumeration now checks every original native result and preserves the exact
saved device and initially active-context observations. Its local scope records
only successfully selected devices, attempts ordered restoration and release
once, and reports failures precisely. Cleanup failure during unwinding is fatal.
Legitimate zero free memory and asymmetric peer access remain valid results.

The focused replay includes the actual production body with isolated native
API doubles and an archived original-body control. It covers every observation
and cleanup failure boundary and is explicitly registered in preflight. A
successful primary-context release call proves a checked reference-release
request; it does not certify context destruction, returned VRAM or the complete
opaque-resource admission boundary.

## Discovered model jobs retain the original MPI owner

Parity discovery emitted raw MPI launches for both focused source cases and
numerical campaigns. The fast-gate registration validator also excluded model
labels. The original inventory contained 1,201 affected registrations.

Both discovery branches now use the existing process/session helper, with its
original native module, termination grace and retirement authority. Logical
test names, model prerequisites, ranks and filters remain unchanged. A bounded
metadata-derived directory name accommodates long test identities. Resolved
helper location and configuration participate in registration regeneration;
unchanged builds retain their timestamps. The validator covers all actual MPI
and process-owned jobs, including model registrations.

`V2_Integration_ParityDiscoveryMPISessionOwnership` explicitly protects this in
preflight. Device-free tests cover original unowned launches, missing inputs,
long/spaced paths, helper relocation and multiple build configurations. The
actual CPU MPI probe uses the existing owner and confirms empty session storage
after initialize/reduce/finalize. Broader timeout/failure coverage remains in
the original owner lifecycle suites.

After rebuilding all 36 existing parity targets, the actual inventory has zero
raw MPI launches. All 12 focused registrations pass. One CUDA and one ROCm
dynamic-MTP source case also pass through the new owner, with clean driver and
retirement evidence in `source-admission-owned-launcher-smoke-r25/`. Candidate,
original controls and independent review are retained in
`/tmp/qfn-parity-mpi-session-ownership-r1/` and its sibling review directory.

## Focused qualification

The coherent Integration rebuild passed with CUDA SM80/86/89/90 and HIP
gfx906/gfx1100. All 22 focused host registrations passed. Real hardware
discovery and five native CUDA/ROCm direct/pooled retirement registrations
passed. The previously failing two-device ROCm case then passed 20 complete
CTest executions, each containing 20 immediate lifetimes: 400 additional
lifetimes without a timeout. The driver window was clean and the owned MPI
session directories were empty afterward.

The twelve unchanged CUDA16/CPU32 source-admission cases also passed: mmap,
zero-capacity host-adaptive cache and positive host-adaptive cache, each with
MTP Off/fixed1/fixed3/Dynamic. All 48 layers and the original active retained
capacity15 were preserved. Every case proved native owner retirement and
removed its fixture-owned prefix files. Runtime identity stayed unchanged;
the complete ten-minute driver window was clean. The symmetric twelve original
MI50/CPU cases then passed in 590.74 seconds, also with complete retirement,
unchanged runtime and a clean driver window. Earlier RAM evidence remains
attached to its earlier binary and is not relabeled as a fresh execution.

Evidence: `/tmp/qfn-admission-failure-integration-r30/` contains the preserved
r29 audit, reviewed patch identities, coherent build, focused host/native
receipts and source-batch audit. The real source receipts are under
`parity-results/qwen-flash-next/source-admission-cuda16-remaining-host-tiers-r23/`
and `source-admission-rocm16-remaining-host-tiers-r24/` in that same parent.
The complete r30 Unit/preflight transaction is now green: **3,263 passes, zero failures and 29 unchanged hardware-unqualified cases**.
The actual phase receipts report Unit: 855 pass, 0 hardware-unqualified; host: 1330 pass, 0 hardware-unqualified; CUDA: 273 pass, 0 hardware-unqualified; ROCm: 378 pass, 2 hardware-unqualified; mixed/exclusive: 427 pass, 27 hardware-unqualified.
The final audit authenticates exact canonical membership, the unchanged
hardware skip set, source/runtime identity and a complete clean driver
window. All 2,985 completed process-owned jobs have empty declared MPI session
bases afterward. The driver window contains 251 new records and zero findings.
Evidence is `full-gate/`, `r30-audit.json` and `r30-session-retirement.json`
under the r30 directory.
This qualifies the integrated model-free slice; the complete real-model
source matrix and distributed continuation implementation remain separate.

The next focused source rerun (`source-admission-implemented-topology-recheck-r26/`)
passes all six previously failing implemented-topology cases in 705.38 seconds:
mixed CUDA/ROCm/CPU PP fixed3/Dynamic, ROCm TP4 random dynamic expert ownership
with sharded VRAM ngrams fixed1/fixed3, and CPU NUMA PP2 positive host-adaptive
RAM/SSD fixed1/fixed3. All retain the original 48 layers and active MTP capacity15.
Native owners retire, fixture prefix files are removed and all six declared
MPI session bases are empty afterward. Runtime identity is unchanged; the
driver window has 25 new records and zero findings. `completion-audit.json`
joins the exact CTest registrations, native receipts and cleanup observations.
This closes those six historical failures without changing their policies.
The four original ROCm PP(TP2, TP2) capacity/composition failures remain open.
