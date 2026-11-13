# QFN shared-reader source admission: focused qualification

The user explicitly requested focused tests and repairs for this iteration.
No complete Unit/preflight receipt or release qualification is produced.

## Confirmed defect and repair

A shared TP query cohort may have several compute readers while only its
coordinator owns cached ngram rows. ModelContext treated the absence of remote
storage owners as the absence of remote participants. The coordinator therefore
omitted retained result transport, and the first query-only reader failed cold
source preparation. A second check rejected a materialized reader-only transport;
the PLE graph builder separately repeated the owner-only predicate.

`PooledImmutableParticipantExecutionPolicy::requiresRetainedParticipants()` now
projects both independent roles from the original checked source geometry.
Model loading and PLE composition consume that same decision. Retained creation
requires actual remote-owner or compute-reader transport; original source,
cohort, allocator, worker, stream and native membership validation remains.
No cache replicas, implicit transfer route, enlarged timeout, or precision change
was introduced.

The focused regression crosses CUDA/ROCm metadata, 1/2/4/8 owners and readers,
zero/positive cache capacities, and coordinator-owned/remote storage. It is
registered as Unit and explicitly in ProductionTestPreflight. Six selected
registrations pass in both focused624 and focused626, including twenty repeated
runs of the new invariant and existing original-cohort failure/abort coverage.
These are device-free role and lifecycle proofs; they do not certify all native
hardware populations.

## Complete local lookup with shared readers

The second defect was in graph composition: the directed builder attempted a
mixed multi-owner binding for one local storage owner. That owner correctly has
no distributed ingress frontier. The repaired builder reuses the existing local
lookup graph and appends the original result/ACK transfers before ReadersRetired,
release and retirement. It preserves the original layout and mixed-P1 rejection.
A closed local-or-distributed completion descriptor carries the actual graph
through the existing SourcePlan. The local descriptor authenticates phase, source,
controller and output identities, internal edges, the real hash producer and the
existing read-ahead trunk dependencies. No arithmetic or kernel changes are made.

The new explicit preflight lifecycle regression and 32 existing owning graph
registrations pass. Actual source admission, snapshot admission and the complete
unchanged numerical panel now pass for the originally failing ROCm TP2 SSD/C64
cell. Missing/foreign shared-ACK mutation coverage requiring the actual native
cohort remains a separate model-free fixture gap; metadata doubles never receive
native authority. Eight-device execution remains unqualified on this host.

## Evidence and exact scope

| Evidence | Result |
|---|---|
| r611 coherent ordinary build | Passed; canonical native dependency selections and shipped compiler targets retained |
| focused621 merged host registrations | 47/47 pass |
| focused622 addressed CUDA2/ROCm2/ROCm4 wire cases | 8/8 pass; 320 source-bound samples, 288 progress observations; component scope only |
| focused623 ROCm TP2, coordinator-only C64, SSD backing, MTP off | Reproduced missing reader transport; clean driver and complete resource retirement |
| r624 / focused624 | Five-path repair, normal incremental build; 6/6 selected host registrations pass |
| focused625 same model cell | Passed source preparation; exposed duplicate PLE-builder predicate; complete retirement |
| r626 / focused626 | PLE builder uses the same tested role decision; normal incremental build and 6/6 host registrations pass |
| focused627 same model cell | Exposed sole-local-owner/shared-reader graph composition gap; complete retirement, clean driver |
| focused628 complementary ROCm TP2 cell | RAM backing, owner1 C0, query-only coordinator, MTP off: source admission passes in 140.226 s; one 28,800,138,240-byte source, zero cache payload, complete retirement and clean driver |
| focused629 same topology, R3 snapshot admission | Passes in 112.949 s with unchanged reference/snapshot setup; both device contexts retire; owned 80-byte cache header removed; numerical inference has not executed |
| r631 RAM/C0 frozen-history numerical panel | All 25 histories / 178 rows collected in 263.948 s; all four required groups pass, exact reset replay, clean drivers and complete native retirement |
| r634 normal owning rebuild / focused634 | Complete incremental core/owner/matrix/canary build passes in 252.788 s; 33/33 focused ownership/graph registrations pass in 40.479 s |
| focused635 original SSD/C64 source cell | Passes in 114.611 s; complete source/resource retirement and clean drivers |
| focused636 SSD/C64 R3 snapshot admission | Passes in 88.125 s; both GPU contexts retire and owned 80-byte cache header is removed |
| r634 SSD/C64 frozen-history numerical panel | All 25 histories / 178 rows collected in 242.111 s; unchanged required limits pass, reset byte-exact, clean drivers and complete retirement |

## Owner-subset numerical results

Both ROCm TP2 owner-subset cells execute the original frozen-history panel
through the actual production runner. It keeps the exact GGUF, all six
arithmetic selectors, teacher histories and numerical limits. The four selected
framework regressions pass before the model run. Their explicit diagnostic
receipt cannot satisfy complete preflight or image/release consumers.

| Required group | Phase | Mean KL | Top-1 agreement | Mean top-5 overlap |
|---|---|---:|---:|---:|
| Short, 16 histories | Prefill | 0.029528 | 87.50% | 90.00% |
| Short, 16 histories | Decode | 0.028153 | 95.83% | 91.67% |
| Held-out, 8 histories | Prefill | 0.019763 | 100% | 90.00% |
| Held-out, 8 histories | Decode | 0.004590 | 100% | 92.92% |

The table applies to both runs: every one of their 178 row-metric records is
identical. This is metric identity, not an unretained raw-logit byte comparison.
The RAM/C0 run precedes the local-graph delta; the SSD/C64 run qualifies the final
repaired source. All required row, offset, tail and agreement guards pass without adjustment.
The remaining QSA-boundary history also executes (prefill/decode mean KL
0.0000805/0.0001762); its existing diagnostic-only acceptance role is unchanged.
The reset probe is byte-exact. Both original GPU contexts and all native process
owners retire, and the canonical runner removes its raw logit payloads after
scoring. These are scoped Static/Off source-placement results, not active MTP,
movement, every owner mask, performance, or complete model/image certification.

The report is
`/tmp/qfn-owner-subset-integration-refresh-r611/diagnostic-panel-rocm2-mask2-ram-c0-r612/report.json`;
the compact metrics/closure record is
`/tmp/qfn-panel-integration-r631/accuracy-summary.json`. The repaired SSD/C64
report and summary are `/tmp/qfn-sole-local-integration-r634/mask1-numerical-panel/report.json`
and `/tmp/qfn-sole-local-integration-r634/accuracy-summary.json`.

Local evidence roots are `/tmp/qfn-owner-subset-integration-refresh-r611`,
`/tmp/qfn-shared-reader-materialization-r624`,
`/tmp/qfn-shared-reader-graph-r626`, and the named focused directories under
`/tmp/qfn-cpu-joint-qualified-r224`. Earlier r611 artifact provenance remains
bound to its original binary; r624/r626 record their own normal incremental
builds and artifacts. No original model, reference history or accuracy threshold
was changed. Further owner-subset admission and their original frozen-history
numerical panels remain required. Active shared-GPU MTP ordering, remaining
topologies and cache/read-ahead performance remain open.
