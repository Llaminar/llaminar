# Frozen binding lookup and QFN admission setup

## Scope and implementation

`FrozenModelWeightSet::optionalLayerForDevice` scanned every binding on each
layer-field lookup. A cold QFN CUDA0 participant retained 57,699 bindings,
including original CPU expert sources; CUDA1 retained 1,251. Retained-source
runner reconstruction reduced both sets to 1,104. Attention and FFN construction
repeat the same lookup, so the cost multiplied across MTP graph specializations.

The existing immutable layer/name index now retains all original binding
positions. Device-qualified lookup filters that small candidate range against
the original home/resident device. It preserves ambiguity rejection, exact
vendor/ordinal identity and original binding addresses. Unqualified lookup
explicitly retains its previous last-in-plan selection. Copy/move resolves
indices against the destination's own binding vector. No separate device cache,
weight copy, placement decision or graph-specialization policy is introduced.

Optional `forward_graph` diagnostics report disjoint host construction time for
binding, attention, FFN, merge and remaining work. Conditional transfer-manifest
timing is nested within FFN, never additive. CPU inference does not sample these
GPU setup timers. Fixed category labels and one summary per completed graph keep
observation size bounded.

## Focused validation

The development composition rebuilds the core and five owning targets in
`/tmp/qfn-cpu-joint-qualified-r224/build360-result.json`. Eleven selected Unit
and integration registrations pass in 10.45 seconds, including the complete
WeightPlan suite, graph-template/PerfStats checks, source/owner lifecycle
contracts and preflight inventory ownership. CUDA and ROCm native weight-plan
preparation both pass in 3.16 seconds. Neither run skips tests or reports driver
errors. Evidence: `focused361` and `focused362` under the same qualification root.

`V2_Integration_FrozenDeviceBindingLookupContract` explicitly joins
`ProductionTestPreflight`. The device-free tests cover 1/2/4/8 metadata endpoint
widths, reversed source order, mixed vendors with equal ordinals, home/resident
aliases, missing fields/devices, ambiguous ownership, global and PP layer
coordinates, copy/move and 1,104/57,699-entry populations. Those metadata widths
do not claim native device qualification. Four affected production/test
translation units also pass syntax checking against the isolated feature
checkpoint, rather than borrowing the development composition's source headers.

Complete Unit/preflight and numerical generation are not renewed by these
focused checks. Other pending native-transport and cohort-migration changes are
outside this checkpoint.

## Matched metadata measurement

The diagnostic calls the actual production class through the current core,
recompiled against matching headers for each variant. One sweep uses three
passes over 49 layers, 18 fields and two vendor identities: 5,292 lookups,
1,323 hits and 3,969 misses. Three process samples per population return the
same ordered binding identities. Table values are medians in milliseconds.

| Original bindings | Scan lookup | Indexed lookup | Scan construction | Indexed construction |
|---:|---:|---:|---:|---:|
| 57,699 | 1,305.35 | 0.593 | 90.83 | 87.03 |
| 1,251 | 13.611 | 0.491 | 1.97 | 1.95 |
| 1,104 | 12.163 | 0.541 | 1.82 | 1.68 |

These are metadata measurements, not inference throughput. The scan measurements
overlap a dependency compiler; the indexed samples run after that compiler
finishes. No wall-time limit is embedded in Unit. Source, exact compiler/link
arguments, outputs and result comparison are retained in
`/tmp/qfn-frozen-binding-index-r445/`.

## Real-model source admission

The matched canonical cell is Qwen Flash Next UD-Q4_K_XL, CUDA0/CUDA1 TP2
ExpertOverlay with CPU0 capacity, GPU128/CPU384 expert ceilings, both continuation
GPUs owning bounded adaptive ngram caches (256 MiB ceilings), original RAM/SSD
backing, Static/Ordinal, FP32 activations, FP16 KV and fixed MTP depth three.
It retains the full depth-15 graph family. Both runner lifetimes must prepare
their original source and completely retire; this fixture does not execute a
numerical generation panel.

The original measured run passes in 506.43 seconds (`source359`). First-runner
CUDA0 graph construction across 64 MTP specializations consumes 232.27 seconds
(binding 42.38, attention 93.41, FFN 95.01); CUDA1 takes 16.13 seconds.
Replacement runners take 20.43/20.02 seconds. Participant times overlap and
must not be summed as elapsed time.

The indexed run passes in **251.77 seconds** (`source363`), with clean drivers
and complete retirement. First-runner construction takes **15.13/15.05 seconds**
for CUDA0/CUDA1; replacement runners take 15.61/15.39 seconds. CUDA0's first
binding/attention/FFN components are 0.10/0.32/13.44 seconds. Both variants retain
64 MTP graph specializations per participant/lifetime, 3,072 prepared predictor
records, two serving participants and retained depth 15. The original binding
populations also remain 57,699/1,251, then 1,104/1,104 on reconstruction. The
improvement does not reduce admitted work. This is one matched diagnostic
admission pair, not a Release inference-throughput result.

The existing one-MI50 plus two-socket CPU tensor-sharded source cell also passes
after the change in **171.07 seconds** (`source364`), compared with 300.05 seconds
in `source348`. It uses one MPI rank/socket, 28 physical workers/rank, original
GPU64/CPU448 expert ceilings, RAM ngrams, Static/Ordinal, FP32 activations,
FP16 KV and fixed depth three. Both ranks retire with clean drivers; their
original predictor-record counts remain 1,728/1,344 and retained depth remains
three. Only rank zero owns the serving GPU participant and RAM ngram table.
Its 16 MTP graph constructions total 1.97 seconds. This two-process source check
is distinct from the CUDA cell's in-process runner reconstruction.

CPU evidence: `/tmp/qfn-cpu-joint-mtp3-source-r348/` and
`/tmp/qfn-cpu-joint-mtp3-source-r364/`. The indexed CPU/MI50 setup categories are
in `cpu2-fixed3-indexed-attribution.md` beside the CUDA category reports.

Exact cell identity, build receipts, commands, canonical source reports, driver
windows and complete logs are in `/tmp/qfn-cuda-shared-mtp3-source-r359/` and
`/tmp/qfn-cuda-shared-mtp3-source-r363/`. Reproduce through
`scripts/ci/run_qfn_source_admission.py --build-dir BUILD --output NEW_DIR
--exact-cell CELL`, using the full registered identity in either invocation
record. The cold-build category report is under
`/tmp/qfn-cold-build-parser-r444/`.
