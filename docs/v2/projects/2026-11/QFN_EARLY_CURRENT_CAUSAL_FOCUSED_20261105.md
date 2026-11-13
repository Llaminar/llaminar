# QFN early-current read-ahead: focused causal regression

The ordinary pooled read-ahead frame passes focused native component tests on
CUDA and ROCm with genuine RAM and SSD backing. This iteration deliberately
omits the full Unit and ProductionTestPreflight suites at the user's request.
It does not certify full-model overlap, lookup latency, or a release image.

## What the regression establishes

The fixture delays the existing immutable lookup service's actual CPU worker
before its original task body. The original captured QFN PLE frame publishes
its demand and executes independent native hash work. A mapped completion
marker must become visible while that same source worker remains held.
Only then does the fixture release the worker and allow the original response,
read, acknowledgement and retirement graph to complete.

The independent hash work is component work, not an artificial claim about
useful model-stage overlap. There is no alternate source processor, raw
diagnostic parent, GPU host callback, or test-selected transfer implementation.
Four production files extract the existing private admission/task bodies;
public calls continue dispatching the same original task and context.

Each backend/backing combination uses two request admissions and replays
4 → 1 → 0 → 4 live token rows through retained capture. Assertions check:

- Exact native source bytes and decoded outputs, including poisoned inactive rows.
- Issued/consumed protocol sequences, live ID/native-byte totals and the existing
  24-byte terminal message; allocation capacity is never the payload extent.
- Idle source/owner frames, zero held pins and reservations, and exact physical
  payload/native-context retirement.
- Gate release and original worker retirement on cancellation, exceptions and
  owner destruction, without a strong service/worker ownership cycle.

SSD cases include the bounded RAM row cache. Fully RAM-backed cases do not
allocate a redundant RAM cache. All existing source, PMA, channel, generation,
stream and retained-cohort checks remain in force.

## Focused results

Normal owning CMake build: **254.155 seconds**, unrestricted parallelism,
canonical CUDA 80/86/89/90 and HIP gfx906/gfx1100 target settings, unchanged
spill guards. No partial object replacement was used.

Eight focused CTest registrations pass with no skipped cases and a clean
driver window in **16.575 seconds** including the driver observer:

| Registration | Result |
|---|---|
| `PooledEarlyCurrentCausal_CUDA_SSD` | Pass |
| `PooledEarlyCurrentCausal_CUDA_RAM` | Pass |
| `PooledEarlyCurrentCausal_ROCm_SSD` | Pass |
| `PooledEarlyCurrentCausal_ROCm_RAM` | Pass |
| `ImmutableLookupClosedWorkerGateLifecycle` | Pass |
| `PooledCurrentDemandGraphLifecycle` | Pass |
| `RetainedHostServiceOriginalConsumers` | Pass |
| `ReadAheadWorkspaceDeclarationReplay` | Pass |

All names have the `V2_Integration_` prefix. The four new native regressions
and new device-free worker-lifetime regression are explicitly registered in
ProductionTestPreflight; registration does not imply that suite ran here.
The existing graph and workspace checks complement native ordering and
retirement but do not supply model-performance evidence.

Evidence: `/tmp/qfn-early-current-integration-r638/build.json`,
`/tmp/qfn-early-current-integration-r638/focused-results.json`, and
`/tmp/qfn-cpu-joint-qualified-r224/focused638-early-current-causal/` contain
the exact commands, CTest output, JUnit and driver observations. The frozen
implementation package and private host controls are in
`/tmp/qfn-pooled-early-current-causal-r620/`.

Remaining work is useful full-model overlap and measured serving benefit,
plus qualification of shared/adaptive and grouped-MTP read-ahead roles.
This component result changes no inference default and broadens no capability
advertisement.
