# Native Dynamic maintenance completion and MTP layer scope

This is a focused correctness record, not a full preflight or real-model
certificate. The affected model-free pipeline is MI50 EP1 → EP2 on its original
MPI/NUMA owner. QFN performance tuning remains behind complete Unit and
ProductionTestPreflight qualification.

## Defects and fixes

Native maintenance did not expose completed policy decisions through the public
GPU status. A launch counter cannot supply that evidence: a valid no-move
decision has no weight transfer, and a retained moving transaction may finish
after a newer planner pass reports an incomplete window.

The existing device controller now retains a bounded completed-decision receipt.
The finalizer records successful no-move completion after candidate abort, or
moving completion after selector and journal publication. Moving provenance
comes from the retained command proof, not mutable planner scratch. The terminal
archive preserves cumulative counts across request reset and rejects duplicate,
regressing or changed receipts. An unavailable rejection count is reported as
unavailable. No additional inference readback or host decision authority was
introduced.

The first native lifecycle rerun then isolated a separate MTP defect. Dedicated
post-commit maintenance inherited an inline-before-current-layer observation
window. A final stage with one main layer and one predictor layer selected only
the predictor histogram. Predictor execution intentionally does not author
committed routing statistics, so that window could never become ready. Both
MTP-off placement cases passed, while all six MTP cases still lacked decisions.

Dedicated maintenance now selects the original stage's main transformer
interval, translated into runtime-table storage coordinates. The complete table
capacity and predictor placement descriptors remain intact. The checked cold
scope helper is separate from the device controller header. Existing terminal
DEBUG output now includes the already copied decision, cadence and readiness
fields; logging adds no device observation.

## Focused qualification

All of these entries passed on the installed source:

| Coverage | Result |
|---|---|
| MoE runtime-table and rebalance-controller Unit owners | 2/2 passed |
| Completed-decision replay and main-layer scope contract | 2/2 passed |
| Original CUDA2 and ROCm2 captured completion replay | 2/2 passed; complete clean driver observation |
| MI50 EP1 → EP2 Dynamic request/reset/prefix lifecycle | 8/8 passed; complete clean driver observation |

The eight lifecycle cases cross ordinal/random expert placement with MTP off,
fixed depth 1, fixed depth 3 and dynamic depth. They use the existing registered
runner, captured execution, original device ownership and unchanged assertions.

The native completion replay executes 20 recorded transactions for each of two
requests on both backends. It covers no-move completion, incomplete windows,
failed apply, repeated finalization, non-root behavior, retained moving proof
after a busy reservation and actual controller reset. It is publication-protocol
coverage; recorded apply inputs do not certify physical expert copies. The
separate pipeline lifecycle cases exercise the actual maintenance path.

Two new-fixture defects were corrected during qualification: the external input
producer event must be joined before capture, and an old reader must genuinely
pin a published bank before testing retirement. The replay now acquires/releases
that recorded reader through the production CPU epoch protocol around
consecutive native publications. It does not manufacture a reader on an empty
bank or alter production epoch semantics.

The changed native publication kernel compiled with mandatory spill guards for
CUDA sm80/sm86/sm89/sm90 and HIP gfx906/gfx1100. CUDA register counts were
74/68/68/66, with zero stack/local storage; HIP used 66 VGPRs and 76/74 SGPRs,
with zero private scratch and zero recorded spills. Both implementations retain
one 256-thread block and 280 bytes of shared/LDS storage. These are compiler
resource facts, not measured throughput or occupancy gains.

Local evidence is retained under
`/tmp/qfn-native-main-scope-install-r2-20261111/focused-qualification.json`.
The exact command/result logs are
`/tmp/qfn-full-preflight-after-cpu-next/q7-native-main-scope-r2-{host1,native1,lifecycle1}.*`.
Compiler resource details are in
`/tmp/qfn-native-completed-decisions-resources-private/NOTE.md`.

## Remaining qualification

The broader repaired-preflight panel, the mapped-packet fixture migration and
its Batch resource accounting still need qualification. The original mapped
stall was reproduced with two mutually waiting complete parents on the same
GPU, a topology production rejects; that does not establish the cause of a
stall on a valid distinct-device topology. The replacement must preserve compact
byte, changed-publication, poisoned-suffix and repeated-capture checks on real
participants. Complete Unit/preflight and real QFN source, generation, numerical
and Release performance gates remain open.
