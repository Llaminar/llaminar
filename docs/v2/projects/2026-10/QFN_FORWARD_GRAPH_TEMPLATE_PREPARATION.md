# QFN forward graph template preparation

Date: 2026-10-31. Status: mechanical extraction and immutable template compile
for all shipped GPU architectures and pass focused device-free, matched native
startup and request-lifecycle replay checks. Two further profile-led preparation
rounds are authorized; broader joined admission remains open.
This is a project plan and evidence index, not a tuning dashboard.

## Problem and measured target

Serving prepares distinct captured executables for both sparse-query programs,
two MTP condition purposes, every retained verifier width, and both terminal
outcomes. The current maximum-width family needs 64 MTP forward constructions
per participant. Those executable identities are intentional, but rebuilding
unchanged ownership metadata inside every layer of every construction is not.

A host profile of actual retained MTP preparation found 88.33% inclusive samples
in FFN construction and 72.53% in canonical expert-owner-map construction.
Continuation resolution accounts for 37.98% and sorting for 25.42%; these
inclusive percentages overlap and must not be added. Native capture accounted
for 5.75%. These are sampled CPU-cycle shares, not elapsed-time savings.

Evidence: `/tmp/qfn-admission-graph-build-profile-r2/` records exact process,
configuration, sampling settings and reports. The sampled source-admission cell
is explicitly profiler-perturbed and cannot be used as an unprofiled benchmark.
The earlier mixed setup observation is in the sibling `-r1` directory.

## Implementation boundary

The existing [incremental source split](DGO_INCREMENTAL_SOURCE_SPLIT_PLAN.md)
provides the physical boundary. `DeviceGraphOrchestratorServingGraphFamily.cpp`
contains cold family materialization; the original class continues to own all
state. `MTPGraphPolicyScopes` supplies one implementation shared with live and
workspace-declaration paths. The first extraction moved three complete methods
and the setup phase timer without changing their bodies. Policy scope bodies
retain identical tokens. The class declaration and member layout did not change.

An immutable builder recipe is prepared once at the original cold setup boundary
before workspace-family declaration. Its typed specializations expand through
the existing production Qwen attention, FFN and PLE producers. Each resulting
graph owns fresh mutable stages. The current original `ForwardInput`, exact
`ForwardGraphSignature`, workspace generation, snapshot identity, device and
embedded pointer identity remain authoritative.

The recipe consumes original immutable authorities:

- Frozen model and decode-replicated weight bindings, with their actual owners.
- The resolved source/routed plan and exact participant/TP/PP ordering.
- The participant residency's frozen initial expert map when its source and
  map-option contract match the graph's required map.
- A continuation descriptor derived once from that same immutable topology.

Where the current map has no provenance receipt, one cold canonical validation
may compare the complete original participants and ownership before aliasing.
This transient build must appear in preparation timing/counters. Preferred or
random initial maps and option-dependent validation must retain their original
semantics. Any genuinely distinct required map is prepared once and retained
under its existing metadata owner. There is no new live placement ledger.

Layer descriptors use original frozen source views rather than duplicating
retained weight payloads or adding unpriced CPU storage. Static-source replacement
after publication is rejected at actual setters. Arena/buffer binding, sparse
policy, row geometry, terminal policy, invocation data and current expert
placement remain specialization inputs. Mutable stages and request state cannot
be shared across family members.

The facility is reusable by other model builders without routing QFN through
an incomplete older schema or cloning captured native graphs. No device query,
new transfer, arithmetic change, executable launch or alternate execution policy
is introduced by template preparation.

## Functional and economy gates

Use the actual production builder and small admitted CPU fixtures for the fast
construction tests. Initialization and tensor preparation are outside timing.
Exercise N = 1, 2, 4, 8, 16, 32 and 64 family members with distinct row, role,
sparse-query and terminal policies. Check:

1. Preparation/map-validation counts are independent of family size; every
   requested specialization is represented.
2. Original nodes, dependency edges, tensor roles and binding identities match
   direct production construction, including full and PP-partial graphs.
3. Every mutable stage instance is distinct; changing one member cannot mutate
   another. Original immutable backing stays retained until its final consumer.
4. Foreign device/source/cohort, static replacement and incomplete preparation
   fail before specialization. Random/preferred ownership and different map
   option contracts preserve their original validation.
5. Current placement and invocation state are bound separately from the frozen
   initial topology. Reset and repeated specialization preserve that separation.

Register the deterministic count/lifecycle regression explicitly in
`ProductionTestPreflight` and Unit. The construction benchmark reports initial
preparation, specialization and total family time separately, with exact model
geometry, backend, compiler and CPU/NUMA/thread configuration. Use repeated
unprofiled measurements; CI should assert work counts/complexity rather than a
fragile shared-host wall-clock threshold.

Before promotion, rerun a matched real source-admission cell and compare its
unprofiled phase timings and preparation counters. Native CUDA/HIP qualification
must retain the entire family, serial/grouped numerical contract, reset/replay,
exact live communication extents and the declared backend topology. Keep CPU,
CUDA, ROCm, MTP off/1/3/dynamic and the existing admission matrix authoritative.
No device-free test certifies native capture, numerical parity or model speed.

## Current extraction evidence

`/tmp/qfn-dgo-serving-split-r1/` contains the isolated eight-file extraction,
before/after sources, body comparison, build log and test reports. In the same
Integration build the new family file compiled in 30.526 seconds; remaining DGO
compiled in 143.8 seconds. This establishes the cost of their individual build
steps under that host load, not a paired whole-build speedup.

The complete DGO and workspace-policy test binaries pass their two CTest entries:
357 GTest cases passed; three existing GPU-dependent cases remain skipped in the
device-free DGO suite, and its ten pre-existing disabled cases remain disabled.
All ten selected synchronization, publication, KV-lifecycle, transition and CPU
allocation source-policy CTests pass, including their negative controls.
The main running source-admission tree is unchanged. Full candidate preflight
and native template qualification remain pending.

## Implemented template evidence

`/tmp/qfn-forward-template-implementation-r1/` retains source deltas and the
focused test/build receipts. The original IGraphBuilder, DGO, Qwen35MoEGraph and
GPU workspace-policy Unit registrations pass. The new eight-case
`V2_Unit_PreparedForwardGraphTemplate` and explicit
`V2_Integration_PreparedForwardGraphTemplateReplay` also pass. They cover cold
failure, immutable publication, independent mutable stages, actual canonical
map-construction counts, original random/preferred-map semantics and map lifetime
after builder teardown through the actual general overlay stage. Initial test
fixture failures are preserved; their corrections changed no production bytes.

The separate construction benchmark preserves five raw samples for each of seven
family sizes for both dense forward and real routed FFN construction. All sizes
perform exactly one canonical owner-map construction per routed builder. The
64-member median specialization times are 0.940 ms for the tiny dense fixture and
1.310 ms for the two-layer routed fixture. These measurements overlapped the
source-admission campaign; they are diagnostic construction timings, not an
isolated speedup or a measurement of full-model/native startup. Full raw output
is retained alongside the JUnit receipt rather than relying on its truncated
successful-test stdout.

The bounded independent source review is
`/tmp/qfn-forward-template-review-r1/review.md`; it found no concrete blocker and
records the untested native boundary explicitly. A narrow comparison worktree at
`/tmp/qfn-forward-template-native-r1/` contains the current main feature HEAD plus
only the template and source split. Use that binary and the unchanged baseline
with the same original source-admission cell and retained family to measure
preparation phases without attributing unrelated pooled-ngram work to this slice.

The narrow build passes the six focused Unit/preflight registrations in 4.02 s.
Its core, real QFN source-admission executable and native lifecycle replay target
compile for CUDA SM80/86/89/90 and HIP gfx906/gfx1100 with the spill guard enabled.
The four selected existing native lifecycle registrations exercise fixed-depth
same-vendor PP and dynamic mixed-vendor PP(TP2,TP2), with reset and partial work;
all four now pass with no skips. These tiny-model tests complement full QFN
startup timing and do not certify model logits.

The earlier independent admission batch r34 is closed red: 110/111 model source
cells pass, with one MPI-root retirement failure and six driver findings. Its
results cannot certify the template. A fresh canonical recovery cell must close
with its own clean driver and retirement receipt before the matched timing runs.
Retain both the failed batch and the subsequent outcome separately.

## Matched native startup result

The original failed r34 source cell passed its separate recovery invocation,
including native retirement and a clean driver window. Its historical MPI-root
metadata is archived unchanged under r34's `retained-failure-witnesses`; this
manual archival does not turn the old failed cleanup into a successful one.

The first unprofiled ROCm TP4 pair uses the original four MI50s, QFN UD-Q4_K_XL
and shared-Q8 predictor, GPU experts8, RAM ngrams, Static/Ordinal overlay,
active MTP depth1 and retained capacity15. Both Integration `-O3` binaries use
the same selected SDKs and native dependencies, with no competing compiler job.
Both original source-admission validators pass, sources/runtimes remain stable,
and both driver windows are clean.

| Observation | Baseline | Immutable template |
|---|---:|---:|
| MTP forward-family preparation, each GPU's combined scopes | 122.271–122.314s | 58.722–58.739s |
| Speedup for that phase | — | 2.082–2.083× |
| Full diagnostic driver elapsed, including loading/teardown | 239.746s | 145.566s |
| Actual retained family record identities | 292 | 292, exactly equal |
| Immutable template / owner-map / continuation preparations | unavailable before instrumentation | one of each per GPU |

Per-device setup intervals overlap; do not sum them as elapsed startup time.
This is one matched setup pair, not Release throughput or a statistical timing
distribution. The full-driver difference also includes unrelated setup and
teardown variation. Other cold source setup performs genuine owner-map builds;
the global constructor counter is not claimed to be one for the whole process.

Raw logs, original validations, exact family records and counter projections:
`/tmp/qfn-forward-template-native-r1/runs/comparison-rocm4-r1.json` and its two
named invocation directories.

The matched CUDA TP2 pair also passes all original validators with a clean driver
interval and an exactly equal146-record family inventory. MTP preparation drops
from94.753–94.755s to26.111–26.123s per GPU (**3.627–3.629×**). Full diagnostic
elapsed falls from194.529s to103.903s; the same single-pair/setup-only limits
apply. Each GPU reports one immutable template, owner-map and continuation
preparation. Raw comparison: `comparison-cuda2-r1.json` beside the ROCm result.

All four selected native lifecycle registrations pass in41.325s, with no skips,
empty exact session bases and a clean driver interval. They exercise the actual
shared builder through CUDA and ROCm fixed-MTP PP plus both mixed-vendor orders
of PP(TP2,TP2) with dynamic MTP, request reset and partial work. Evidence:
`/tmp/qfn-forward-template-native-r1/native-replay-r1/result.json` and its complete
per-test logs. This is behavioral graph qualification, not independent model KL.

The remaining roughly59s in ROCm MTP preparation has not yet been attributed.
The first optimization does not eliminate all repeated construction. A separate
profiled invocation will select the next immutable work to prepare once; its
elapsed times must not be mixed into the unprofiled comparison above.

## Next consolidation: pooled owner recording

Scope: the user explicitly limited this to "tidy up as we go", not a graph
builder rewrite. Make small, behavior-preserving consolidations alongside the
current admission and serving fixes. Reuse existing APIs and owners; broader
template work requires a concrete measured problem and separate scope.

The 2026-11-01 continuation review found a concrete duplication in cold graph
recording. `PooledImmutableRetainedParticipants::prepareOriginalOwners` performs
storage binding, stage preparation, publication preflight, dependency planning,
scoped capture and stage recording itself. The executor's existing
`captureRetainedGraphFragment` / `recordGraphCaptureBody` already implements this
sequence for other retained graphs. The current collective-classification fix
must first distinguish participant-local captured transfer endpoints from
runtime-communicator collectives across all those entrypoints.

Consolidate recording through that existing executor with a typed retained
setup-only purpose. Preserve the exact original worker's software capture flag,
the existing `setupGraphMaterialization` policy and `SetupAddressesOnly`
authority. Setup must not perform ordinary request coherence, mark outputs dirty
or mark the graph as executed. Common readiness, dependency-ledger, capture
activity, read-ahead and failure handling belong in the shared recorder.
General executable-class validation, same-PMA runtime preparation, instantiation
and submission stay with the existing `RetainedParticipantGraphExecutable`
owner. Replace the old caller loops rather than running both preparations.

This is a bounded consolidation of recording protocols, not another template
or executable cache. Remote owner graphs are already built and captured once
per original cohort and reused for subsequent requests. Their native handles,
streams, source generations, arena roots and mutable stage instances remain
participant-specific. Existing `PreparedGraphTemplate` recipes already share
model bindings, expert-owner maps and continuation metadata across forward
specializations.

Add actual-recorder regressions for setup versus normal recording, exact worker
and stream identity, unchanged setup coherence/completion, one preparation and
dependency-planning pass, partial-failure retirement, and native endpoint versus
runtime-collective classification. Register the focused replay in production
preflight. Re-run the four opposite-vendor source-admission controls and the
applicable request/reset tests after migration; a setup-only pass cannot qualify
generation or numerical output.

For further source-graph templating, first count and profile repeated pooled
topology construction across the existing family-size benchmark (1–64 members).
Only repeated immutable phase/edge descriptions belong in the existing builder
recipe; each specialization still constructs its own bound stages. Extend the
existing preparation/specialization counters and benchmark, with matched native
startup timings when the affected production configuration is supported. No
speedup for this consolidation or pooled source templating has been measured.
