# Pipeline follower maintenance ordering

This records focused model-free qualification on 2026-11-11. It is not a full
Unit/preflight, real-model accuracy or performance certificate. The active
performance target remains QFN PP(EP(4×MI50), EP(4×MI50)), with ngrams sharded
across all eight GPUs.

## Reproduced failure and evidence

The existing Dynamic EP4→EP4 request lifecycle stalled in six consecutive fresh
processes. It exercises fresh prefill, full prefix reuse, a partial-prefix
continuation, request reset and generation budgets of 1, 17 and 1 tokens.
Failures occurred during both the first and subsequent request, so prefix reuse
alone did not explain the defect.

Bounded passive host stacks showed upstream participants waiting for maintenance
ticket events while the tail had not completed the first transaction. Native
queue inspection showed all four upstream GPUs executing RCCL AllGather, but
their actual work descriptors disagreed: participants 0/2 were in the 104-byte
LL status gather, while participants 1/3 were in the 111,104-byte Simple payload
gather. The communicator had four actual participants. Descriptor layouts came
from DWARF built with the exact RCCL source, compiler and definitions; no model
payload or mutable device state was downloaded or changed.

Retained child preparation can submit expert-placement maintenance on its own
stream. The child's preparation dependency orders the child's recording stream.
The follower subsequently launches the enclosing captured generation parent on
a different retained stream. That parent did not join the durable maintenance
publication, allowing generation and maintenance to reuse the same communicator
without the required order. The tail already had this join.

## Installed change

`PipelineDeviceGeneration::submitHostedFollowerCapture` now consumes the
existing maintenance publication event on the actual parent launch stream,
before ordinary or MTP parent submission. `DeviceExecutionTimeline` admits
`DeviceGenerationController` as a consumer of that existing event. The change
adds no host synchronization, count readback, allocation, recapture or new
transport. Original rank/NUMA ownership and the retained captured policy remain
unchanged.

The lifecycle fixture now checks the production evidence for each actual
upstream multi-device domain: every captured follower transaction has exactly
one matching maintenance-event wait on its device. The device-free event replay
checks exact event/stream/device identity and zero H2D, D2H and synchronization
calls for both backend classes across 20 repetitions.

## Focused results

| Coverage | Result |
|---|---|
| Full DeviceExecutionTimeline Unit owner and dedicated event regression | 2/2 passed |
| Fresh Dynamic EP4→EP4 ordinal lifecycle processes | 20/20 passed; approximately 19–21 seconds each |
| TP2→TP2 ordinary request lifecycle | Passed |
| Dynamic EP4→EP4 randomized lifecycle | Passed |
| Dynamic EP1→EP2→EP4 ordinary lifecycle, ordinal and randomized | 2/2 passed |

All native intervals have complete clean driver observations. The repetitions
run the original complete request lifecycle in fresh processes, not just a warm
graph loop. The new registrations are
`V2_Integration_PipelineParentMaintenanceEventReplay` and
`V2_Integration_PipelineFollowerMaintenanceOrdering_ROCm8`; both explicitly join
ProductionTestPreflight. The native alias uses the existing exact lifecycle
case and hardware eligibility policy, with its unchanged 120-second deadline.

Local patch, before/after sources, JUnit, discovery and qualification receipts:
`/tmp/qfn-follower-maintenance-ordering-20261111/`. Exact commands and continuous
driver records are under
`/tmp/qfn-full-preflight-after-cpu-next/q7-follower-maintenance-{host1,native1,repeat19,adjacent1}*`.
The valid passive RCCL work observation is retained under
`/tmp/qfn-root-off-stall-work-r6/761275/`. Earlier debugger attempts with failed
runtime symbol admission are incomplete observations, not evidence of empty
GPU queues.

## Other repaired prerequisites and remaining work

Sixteen CUDA/ROCm planning contracts pass after retaining each existing service
test as an independent registered process. Each retains its original 30-second
deadline; no test body or geometry was removed. The previous aggregate exceeded
that deadline after its earlier bodies had passed. Receipt:
`/tmp/qfn-planning-per-case-registration-20261111/qualification.json`.

Original domain-address preservation, CPU-tail generation ownership, duplicate
activation-download prevention and invalid TP candidate rejection have their
separate focused passes in
`/tmp/qfn-pipeline-focused-root-install-20261111/focused-results.json`.
The earlier completed-maintenance and main-layer scope results are recorded in
[the maintenance note](QFN_NATIVE_MAINTENANCE_SCOPE_20261111.md).

The retained preceding-child metadata proof is installed. Subsequent unequal-
stage MTP failures exposed an independent RCCL MIN/MAX ABI inversion; all eight
original failing lifecycle cells pass after its correction. Original-rank
mapped-packet tests exposed a current-generation dispatch ordering defect.
The next focused qualification and remaining admission work are recorded in
[the reduction/publication note](QFN_RCCL_MINMAX_AND_MAPPED_PUBLICATION_20261111.md).
Complete Unit/preflight, exact QFN source/generation/parity and matched Release
timings remain open. Model-free lifecycle timings are not inference throughput.
