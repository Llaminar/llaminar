# QFN joined GPU and CPU-cohort service calibration

This records a focused native component qualification on 2026-11-05. It does
not renew complete preflight, qualify inference-time expert publication or
provide a Release performance result. The preceding feature checkpoint is
`155e254b9` on `feat/qfn-ngram-continuation`.

## One policy, complete execution targets

Whole experts and sharded experts use the existing demand forecast, tier
priorities, minimum residency, phase non-regression and migration-payback
criteria. Sharding changes the physical execution and movement cost, not the
promotion/demotion heuristic. A cohort is priced as one complete execution
target; individual shards are not independent competing experts.

The joined fixture now measures a whole expert on one GPU and a tensor-sharded
expert on two actual CPU sockets. Both targets borrow the original model,
prepared engines, projection cuts and final PhysicalMemoryAuthority. GPU
service uses the original native context, admitted retained calibration graph
and native event timing. CPU service uses its original admitted collective
domain and root-observed complete execution interval. Shared probe storage is
sized for the largest serially executed probe, rather than their sum.

The existing exact-live evidence exchange sends 240 payload bytes from rank
zero and zero payload bytes from the other rank: two complete targets, three
phases and one 40-byte record per target/phase. Required protocol metadata is
separate. Prefill, decode and verifier prices join four real warmup/measured
bidirectional movement waves through the existing profile composer. All waves
drain without installing a candidate; unchanged placement is asserted.

The shared MPI integration entrypoint now uses the production
`MPIProcessSession`, whose lifetime owns the native CPU process resources.
Raw MPI initialization did not supply that owner. Canonical shutdown retires
collectives and GPU workers before MPI finalization. The joined fixture also
uses the complete-placement accessor; its former scalar-owner accessor
correctly rejected a sharded expert.

## Failure diagnostics and regression

Prepared-service failures must still abort their authenticated original
domain so peers cannot wait indefinitely. Both production measurement
boundaries now log the original exception before that fatal abort. Previously
the destructor-triggered abort terminated the process before the reason was
visible. Successful execution and foreign-domain rejection are unchanged.

The regression launches two native failed jobs with real prepared sources,
PMA owners and one MPI rank per physical socket. One rejects an invalid
initial epoch in the nested CPU producer; the other rejects a missing PMA
argument in the outer target sequence while the real owners remain alive.
It requires the exact original diagnostic before the first abort, the native
nonzero status and complete process/file retirement. It does not substitute a
mock communicator or permit. Each child retains the standard 30-second
progress boundary.

## Focused evidence

All runs use the Integration build and canonical driver observer. Selected
tests have no skips and clean driver windows. Small model-free native fixtures
use one worker per socket; they make no full-socket throughput claim.

| Evidence | Result |
|---|---|
| `focused375`: CUDA and ROCm `CPUJointPreparedServiceMovementCalibration` | 2/2 pass; 9.34 s including observation |
| `focused378`: `CPUJointPreparedComputeService` | Pass, retaining 20 original source/domain lifetimes |
| `focused378`: `CPUPreparedServiceOriginalAbortDiagnostic` | Both native failure children pass; complete selection 2/2 in 15.44 s |
| `focused379`: preflight inventory ownership and device isolation | 2/2 pass; 10.67 s including observation |

These names carry the `V2_Integration_` prefix, with the backend before the
joined calibration name. The three new regressions are explicitly registered
in `ProductionTestPreflight`. The owning native builds are `build370`,
`build372`, `build374` and `build377`; the latter corrects only the observer's
CTest registration. Original failed runs remain preserved, including the
missing MPI process owner and wrong scalar-placement accessor.

Receipts and complete native logs are retained under
`/tmp/qfn-cpu-joint-qualified-r224/`. Isolated checkpoint translation-unit and
Python syntax checks are recorded in `/tmp/qfn-joined-checkpoint-syntax-r451/`.
The shared MPI main has 49 configured consumers; focused validation does not
claim all those binaries have been rebuilt or the complete suite is green.

## Remaining qualification

Complete target service and movement pricing now have joined native coverage
on both GPU backends. The next boundary is committed dynamic placement in
real QFN inference: incoming-engine use on every member, old-bank retirement,
numerical accuracy and useful performance. Those observations cannot be
inferred from calibration waves. Fresh complete prerequisites are still
required before the canonical numerical campaign; complete67 remains stale.
The six-row functional/performance matrix and shared ngram ownership beyond
the qualified scoped cases also remain open.

See [the dynamic residency plan](QFN_SHARDED_EXPERT_DYNAMIC_RESIDENCY_PLAN.md),
[cohort qualification](QFN_COHORT_MIGRATION_AND_MTP_ADMISSION_QUALIFICATION.md)
and [native read-ahead qualification](QFN_READ_AHEAD_NATIVE_CAPTURE_QUALIFICATION.md).
