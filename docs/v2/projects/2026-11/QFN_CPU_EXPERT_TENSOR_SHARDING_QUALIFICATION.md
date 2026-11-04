# QFN CPU expert tensor sharding: qualification boundary

This is a qualification note, not a performance dashboard or a release certificate.
The source slice is based on `f0227e068a5f7460728842b72085a41c8d61c78e`.

## Current qualification boundary

The common bounded CPU loader has completed the matched one-socket and physical
two-socket runs: the same original QFN UD-Q4_K_XL, one UUID-selected MI50,
GPU64/CPU448 expert placement, RAM ngrams, Static residency, MTP off, 25 frozen
histories, 178 scored rows and a separate byte-exact reset control. The
one-socket panel and the latest two-socket panel both report
`REQUIRED_GROUPS_GREEN`. The two-socket run finishes in 269.40 seconds with
178 scored rows, complete source/path/publication evidence, byte-exact reset,
both rank reports, clean driver evidence and native retirement. All 26 retained
logit files remain byte-identical to the preceding two-socket runs.

| Required group | One-socket mean KL | Two-socket mean KL | Two-socket top-1 | Two-socket top-5 overlap |
|---|---:|---:|---:|---:|
| Short prefill | 0.0289344 | 0.0178018 | 87.500% | 91.250% |
| Short decode | 0.0402378 | 0.0379831 | 96.875% | 91.667% |
| Held-out prefill | 0.00845677 | 0.00948526 | 100.000% | 100.000% |
| Held-out decode | 0.00359046 | 0.00384224 | 100.000% | 95.000% |

All four required groups above pass. The QSA boundary history remains
diagnostic in both policies and is not accepted by the required short groups.
These local pilots are not complete corpus, topology, model or image
certification. Short decode's worst per-offset mean is **0.09490395**, still
below the unchanged **0.1** gate; its tail fraction is **0.0520833**.

The user explicitly approved a single-row exception for two-socket short decode:
`bounded_row_outliers: {maximum_rows: 1, maximum_kl_nats: 1.5}`. The ordinary
row ceiling remains **1.2**, and every remaining row must satisfy it. Mean KL
and per-offset mean KL remain **0.1**; tail threshold **0.1**, maximum tail
fraction **0.125**, total variation **0.075**, top-1 **0.875** and top-5 **0.8**
remain unchanged. Every row, including the outlier, remains in every aggregate.
Prefill, held-out and one-socket policies are unchanged. The one observed
exception is `original-nine:decode:2`, KL **1.382641146**. This is a reviewed
numerical trade-off, not proof that a particular stage or quantization operation
caused the difference. Same-input fragment tests and final-logit attribution
remain useful diagnostics; no reference, arithmetic or history was replaced.

The optional count/cap contract has focused tests for zero/one/two outliers,
an exceeded cap, malformed policy, nonfinite rows, unchanged mean/offset/tail/
variation/agreement gates and paired KL delta checks. The full scorer suite
passes 25 tests. The observer now distinguishes a source-authenticated CPU
expert-only follower from a full-model global-TP/logit authority, while keeping
the original source, Store, PMA and per-layer prepared-cohort checks.

The last reporting defect inferred CPU follower phase from reusable buffer
capacity. Gather counters now use the existing authenticated sparse key and
match the original invocation graph role. Standalone dense-route execution
requires an explicit typed phase; declaring both phase authorities is rejected.
The actual-stage contract covers padded decode, one-row prefill, grouped MTP,
empty work and malformed phase. All six phase-focused75 registrations pass,
including the native two-rank collective. In the successful model run both
ranks report **1,392 prefill** and **7,632 decode** gathers, including setup/reset
work. Computation, payload extents, reference logits and numerical policy did
not change with this reporting fix.

Peak sampled process RSS is **115.44 GiB** for the current one-socket run,
versus **178.53 GiB** before the common bounded loader: about **63.09 GiB** less
source/final-storage overlap. Its anonymous peak remains about 114.64 GiB;
file-backed peak falls from 63.65 GiB to 0.88 GiB. The latest two-socket worker
peaks are **73.27 GiB** and **44.39 GiB**, with zero observed process swap.
These are individual process maxima, not summed physical memory or matched
throughput measurements. Native model loading now uses the same bounded
source/consumer/PMA path for ordinary and sharded CPU components; final weights
and the RAM ngram table are intentionally not bounded by the temporary staging
cap. The read-only coverage audit maps 22 focused preflight registrations in
`/tmp/qfn-common-cpu-coverage-audit-r298/`.

The complete67 Unit/preflight transaction executes 4,209 registrations: 4,178
pass, two inventory-classification checks fail, and 29 require unavailable
hardware. Those two checks pass in focused69. After the observer and policy
changes, all six focused72 registrations pass, including both inventory checks.
The original complete67 receipt remains red; no replacement full-suite receipt
is manufactured. The user explicitly requested focused verification and no new
full preflight this turn. Four-device ROCm coverage in complete67 used the
7900 XTX plus three MI50s; these model panels use the selected MI50 UUID.

Current evidence: `/tmp/qfn-matched-rss-rerun-r278/tp1/run/`,
`/tmp/qfn-current-panel-numerics-r299/`,
`/tmp/qfn-cpu-joint-row-allowance-r303/`,
`/tmp/qfn-cpu-joint-phase-panel-r305/tp2/run/`,
`/tmp/qfn-cpu-joint-qualified-r224/common-cpu-observer-focused72/`, and
`/tmp/qfn-cpu-joint-qualified-r224/common-cpu-phase-focused75/`.
The canonical two-socket source-admission case previously passed in r266,
including original source, physical-socket ownership and final retirement.
That pass does not substitute for numerical or dynamic-movement qualification.

Dynamic migration between whole and sharded CPU/GPU expert tiers, active MTP
and the broader placement matrix remain unqualified by this Static MTP-off
slice. Their shared target design is in
[the sharded-expert dynamic residency plan](QFN_SHARDED_EXPERT_DYNAMIC_RESIDENCY_PLAN.md).
The sections below retain earlier implementation and investigation evidence;
this section owns the current outcome.

## Current common-loader validation

The canonical CPU factory now admits the final plans before reading payloads.
It no longer eagerly builds whole CPU source tensors. The optional layer
streamer inspects source metadata, while genuine GPU preparation keeps its
actual target device and graph rank. Role-first precision resolution preserves
FP32 storage for immutable operators and shared-expert input gates without
changing their activation arithmetic.

The joined source/Writer/Store path also fixes final-weight accounting and
lifetime defects. Admission prices exact padded native layouts, independently
rounded expert allocations and the actual typed allocator placement. Native
embedding lookup and packed tied-head layouts remain separate; compatible
floating layouts share their final owner. Cache validation runs outside the
Manager mutex, retains the original Store and logical TP identity, and rejects
retired final storage. Clearing lookup indexes does not reload completed weights.
Decode-cache keys retain exact finite fraction bits rather than rounded strings.

`common-cpu-source-build53` compiles 26 affected targets, including the model
canary and MPI launcher, in 84 Ninja actions. Its focused run executes 60 CTest
registrations in 68.70 seconds: 57 pass, with no skips or driver findings. All
new final-cache, exact-footprint, source-bound, ordered-assembly, precision-policy,
prepared-engine, Registry and RowSlice contracts pass. The remaining failures
were in the complete WeightPlan, WeightMemoryEstimator and WeightManagerSharding
Unit suites.

`common-cpu-source-build54-final-expectations` changes only the WeightPlan and
WeightMemoryEstimator test executables. Their complete owning suites pass in
`common-cpu-source-focused54-final-expectations` (1.83 seconds, no skips, clean
driver evidence). The corrections request the existing input-parallel column
slice explicitly and distinguish useful native bytes from per-root allocator
rounding. Neither correction changes production arithmetic or capacity.

`common-cpu-source-build55-sharding` compiles the joined sharding fixtures.
`common-cpu-source-focused55-sharding` passes the complete owning Unit suite and
the explicit expert-selection and final-cache preflight registrations: three
registrations in 2.44 seconds, no skips, clean driver evidence. The fixtures
declare original model/PMA ownership before CPU reads and retain independent
source-byte, slice, cache and failure-lifetime assertions. Together with the
unchanged passing build53 selection and build54 corrections, every affected
registration in that selection is green. This focused selection did not cover
all legacy CPU preparation fixtures.

The complete transaction `common-cpu-complete55/` has retired after 8,671.64
seconds with clean driver diagnostics. Its unchanged source/binary builds
successfully in 1,770 Ninja actions. Complete Unit executes 990 registrations:
972 pass and 18 fail. Two registrations also contain embedded GPU skips that
the staged fixture split removes from Unit. Host preflight executes 1,929
registrations: 1,912 pass, 17 fail and none skip. CUDA executes 286: 282 pass,
four fail and none skip. ROCm executes 393: 387 pass, four fail and two report
unavailable hardware. The final exclusive phase executes 594: 561 pass, six
fail and 27 report unavailable hardware. The complete preflight inventory is
3,202 registrations; together with Unit, there are 49 failing registrations.

All 29 native unavailable rows have explicit source-owned hardware contracts:
eight-device ROCm, four/eight-device CUDA and eight-GPU mixed extensions exceed
this host's inventory. Required available two- and four-device cases executed;
unavailable rows provide no native qualification. The final MPI planning reds
reproduce the same duplicate CPU scratch claims, including GPU-named cells
with real CPU observers. Both mixed-vendor CPU-tail reds reproduce the same
primary-versus-routed owner defect. The full sweep added no other defect class.
Its source and shared core remained immutable throughout; none of the staged
corrections is represented as a result from that run.

The failures expose both old fixtures and production defects. Fixture migration
now uses actual original source plans, admitted final Store owners and real
prepared engines, preserving source-byte and lifetime oracles. Native
DeviceContext creation moves into explicit CUDA/ROCm integration registrations;
cold backend-tagged MoE cache contracts execute device-free for both vendors.
The source-policy checks pass on the staged composition, including 34 allocation
negative controls and 22 blocking-synchronization controls.

Private compatibility checks compile the migrated Qwen graph, CPU expert stage
and routing fixtures against the unchanged production headers/core: all 129
GTests pass without skips or GPU startup. This validates those fixture repairs,
not the staged production changes. The four native sparse-ticket registrations
also expose one fixture omission: passive route buffers have no original packet
key. The repair supplies the same complete identity as the production dispatch
producer and tests that a foreign layer cannot publish or modify ticket payload.
The production identity guard and all numerical/live-traffic assertions remain
unchanged. Native verification awaits the coherent successor build.

The other two CUDA failures occur in hosted expert and GPU/CPU pipeline
initialization. Combining the continuation and overlay source plans loses the
overlay allocation owner: CPU expert materialization incorrectly charges
`PrimaryModelWeights`, although its storage is admitted as
`RoutedExpertWeights`. The staged repair preserves that original typed ownership
without enlarging either budget. Its device-free original-body control reproduces
the missing routed owner, spare-primary masking and malformed-source failures;
the ordinary Primary/Additional owner control passes. The new
`CPUOverlaySourceOwnerContract` preflight registration and existing CUDA/ROCm
hosted/pipeline registrations will qualify the correction after the rebuild.

Two production corrections are staged with focused preflight registrations.
Final readable CPU weight roots must retain `RequiredForCPUExecution`, including
tied aliases; the original implementation fails three actual post-cleanup
lifetime tests. Planning measurements must not claim scratch already owned by
the native packer; the existing BOM and exact capacity limits remain unchanged.
A third correction is staged: an already sealed/migrated CPU expert bank
must be classified before original source materialization. Complete banks retain
their actual participant/domain engines, partial banks reject before sibling
mutation, and only empty groups populate through the common loader. Strict
Store source validation and joint cohort identity remain in force.

The planning-sample review additionally finds allocator rounding omitted from
source and converted-root declarations. A permitted nine-row FP16 sample has
4,608 logical bytes but an 8,192-byte allocation on this host. Source samples,
converted weights and projection host inputs/outputs now use existing typed
storage roots with independently rounded BOM entries. Logical and wire extents
stay unchanged. The new exact-quota regression covers FP32-to-FP16 and
BF16-to-FP32 conversion, one-byte-short admission, nonfinite input and failure
retirement. Its production and owning test translation units compile, while
runtime proof awaits the coherent build. The separate GPU expert measurement's
legacy host phase allocation remains a recorded follow-up, not a qualified
part of this correction.

The coherent `common-cpu-repair-build58` succeeds after correcting one test
target's missing fixture include directory. Its focused successor executes all
126 registrations in 180.80 seconds: 109 pass, 17 fail, no CTest skips and clean
driver diagnostics. Both MPI planning sample backends, two-/three-rank planning,
all four mixed CPU-tail pipelines and ordinary native/hosted retention now pass.
The remaining registrations share a small set of fixture/lifetime defects;
the broad Unit MTP body also contains four optional native cases, which are being
moved into explicit backend integration registrations rather than retained as
Unit skips.

The tied-alias validator still compares the committed final CPU storage policy
with the original temporary-source intent. Its correction validates the actual
required execution lifetime, with both native/floating sources, alias orders,
source policies and repeated materialization covered. CPU expert fixtures need
the same original routed owner and bounded source scratch as production. Host
retirement fixtures must publish a genuine prepared handle through the Store's
registration API; merely preparing an engine does not publish tensor readiness.
Private fixture checks remain diagnostic until the shared successor runs.

The four all-format CUDA/ROCm sparse-ticket tests now reach their numerical
oracle and expose a production indexing defect: packet top-k remains eight,
while CPU local execution narrows to two routes per row. Publication reads row
one at source slot eight instead of slot two. The fix carries a typed source
layout with the exact execution stride while preserving packet coordinates,
live-only publication and the original per-element multiply. Device-free
regressions sweep widths one through eight, partial/empty work, dirty repeated
replay and invalid complete prefixes. Existing native all-format captured
consumer tests remain the end-to-end regression. No numerical tolerance changes.

The coherent build59 plus build60's native test-environment registration now
passes 132 of 134 focused registrations in 206.32 seconds, with no reported
CTest/GTest skips and clean driver diagnostics. All four CUDA/ROCm sparse-ticket
all-format captured-consumer tests pass on AVX2/AVX512, as do the source-stride
replays, tied-owner lifetime tests, native GPU prepared-cache reuse and all
previously failing ownership/planning/retirement registrations. The remaining
two registrations are the newly exposed native MTP bodies on both backends:
two omit a real main-forward terminal publication, one authors nine tokens
against an eight-token context, and one omits its schema-owned MTP logits
capacity. The build61 fixture successor uses the real GPU weight pipeline and
one admitted native setup. Its focused nine-registration run passes all seven
CPU/driver registrations, with no skips and clean driver evidence. The native
bodies then expose the omitted production workspace-finalization call and stale
result-mailbox widths: a depth-three transaction emits four committed tokens,
and the current metadata ABI has twelve words. The batch-capacity case passes
on both backends in focused62. The other three then reject the fixture's zero
local vocabulary width during graph construction. The successor resolves all
single-participant dimensions before schema construction, and updates stale
native telemetry assertions to the retained integrated prefill path. Its oracle
requires device-observed main/shifted KV counts, current main terminal-hidden
publication and a captured prefill graph; it never selects the old host sidecar
path. Focused63 observes the correct main-nine/shifted-eight counts and terminal
publication on both backends, but the old first two bodies still invoke retired
GPU host-token APIs and the third omits retained serving-family preparation.
The successor explicitly rejects those host-token entrypoints, samples actual
device Target/Draft banks, and uses the public no-launch serving-family setup
for the prefill proof. Focused64 captures and executes the real target/draft
sidecars on both backends, then rejects the fixtures' obsolete host-flush calls.
The prefill setup also correctly rejects an unadmitted default 4 GiB prefix
tier. The successor observes each independent replay through the existing
event-aware logits diagnostic bridge and asserts its handoff counters. It
explicitly turns off prefix cache outside the shifted-prefill fixture's scope;
the separate native prefix suites retain that coverage. Actual captured prefill,
main-nine/shifted-eight counts and current terminal publication remain required.
The final `common-cpu-mtp-fixture-build66` and focused66 successor pass all four
owning CPU/native registrations in 6.37 seconds, with no skips and clean driver
diagnostics. CUDA and ROCm both prove captured target/draft reuse, event-aware
terminal observation, integrated shifted-prefill and batch capacity. No
production arithmetic or admission changes accompany these fixture corrections.

The complete67 transaction finishes in 7,870.52 seconds with immutable source
and binaries and clean driver diagnostics. It executes the full 990 Unit and
3,219 preflight inventory: 4,178 pass, 29 require unavailable hardware, and two
registrations of the same inventory check fail. Unit passes 989/990; host
preflight passes 1,939/1,940. CUDA passes 288/288, ROCm passes 393 with two
unavailable eight-device cases, and the exclusive lane passes 569 with 27
unavailable cases. Every hardware-eligible runtime test passes. Unavailable
device counts remain unqualified. The remaining inventory
test still classifies every selection from the mixed MTP executable as
device-free. A test-only successor classifies the actual GTest selection and
requires the exact native backend/filter/startup/ownership scope, while preserving
CPU isolation. Its seven private metadata tests pass against the complete67
inventory, with leakage and missing/malformed native-registration negatives.
The same tests now copy only the row each negative control mutates, preserving
the complete original inventory. Private runtime falls from 19.99 to 7.73 seconds;
the original document remains unchanged after all checks. This is a test-data
copy optimization, not a change to validation, native scheduling or coverage.
The successor is not represented as a result from complete67. Activation68
changes only the inventory test and documentation; all four owning build targets
report no work. At the user's request, the broader Unit follow-up is cancelled
cleanly and its partial run is not qualification evidence. Focused69 then runs
exactly the repaired Unit and preflight registrations: both pass in 10.48 seconds,
with no skips and clean driver diagnostics. The completed sweep's unchanged
runtime results remain separate; no replacement canonical green receipt for
complete67 is manufactured and no full preflight is repeated this turn.

The unchanged one-/two-socket RSS panel descriptors validate against the current
public parser, retaining 25 histories and 178 positions each. No model inference
was run during this metadata check. The one-socket current-loader panel is now
running; its original two-socket counterpart follows after native retirement.
Both retain the same model, precision, 25 histories and 178 scored positions.
These native panels remain prerequisites for a checkpoint; no new real-model
RSS improvement or numerical qualification is claimed before their completion.

The original-nine full-stage diagnostic metadata now also binds the actual
failing panel's HC-up `two_term_q8` policy. An older full-stage descriptor used
`single_q8` and is retained only as historical reference/history provenance;
it is not a matched native control. The independent FP32 reference and all
seven teacher-scored rows remain unchanged. Metadata-only launch validation
exposes a separate driver defect: the frozen-history adapter rejects the
canonical MPI session guardian and the MPI application delimiter. Its repair
preserves both the cleanup owner and exact original rank population, and its
owning Unit/preflight plus focused guardian registrations pass in the build61
selection. The former implementation fails the retained negative control.
Evidence is in `/tmp/qfn-world2-outlier-metadata-r295/`; no model run is implied.

The last explicit native component run, `common-cpu-source-native52`, passes
all four CUDA/ROCm preparation and CPU-joint padded-prefill registrations in
5.48 seconds, with no skips or driver findings. This predates the final cache
lifetime successor and is not a current full preflight certificate. No new
real-model RSS or numerical result is claimed until the complete gates and
unchanged one-/two-socket panels run against the coherent replacement.

A component audit confirms that CPU projections, embeddings/tied heads,
routers, GDN, norms/scalars, predictor weights and full/fragment experts reach
the common source/Store writer. RAM ngrams retain their intentional direct
reads into final admitted storage with at most 16 MiB per read and no full
temporary mirror. This is a CPU population contract: GPU preparation still
retains legacy whole host tensors and floating overrides before upload. The
next mixed-run RSS report must distinguish those remaining GPU host copies
from the bounded CPU path. Detailed mappings are retained in
`/tmp/qfn-cpu-population-coverage-r289/NOTE.md`.
The GPU follow-on must also verify the raw non-GEMM allocation boundary:
`uploadFrozenNonGemmWeights` reaches the canonical TransferEngine upload,
whose TensorBase allocation path does not itself retain a PMA lease. Native
MTP graph fixtures exercise that existing production path but do not certify
complete GPU-weight accounting. This boundary is separate from the common
CPU writer's exact final-storage and staging leases.

## Canonical bounded CPU loading contract

There is one source-to-final-storage pipeline. Placement and source selection
are inputs to it, so ordinary CPU inference, whole-expert ownership and tensor
shards cannot select different memory-lifetime implementations. Original GGUF
metadata and native source blocks remain authoritative. Quantized sources are
not requantized as a consequence of loading or sharding.

The loading overhead is a configured bounded workspace, using the existing
512 MiB default. Its bound includes native input and every simultaneously live
conversion or packing scratch region. PMA owns both that workspace and the
separate final model storage. The serialization lifetime extends from reading
through consumer commit and source-window release; independent GPU upload rings
retain their existing event ordering and additive storage declarations.

The final tensor or prepared matrix may exceed the loading cap. The common
reader divides it at valid native block/row boundaries and populates its final
owner incrementally. A complete-matrix fit requirement would merely move the
failure to large embeddings or LM heads and is not acceptable. Packed GEMM
storage is allocated once, while resident tensor and RAM-table consumers write
into their admitted final owner. Lazy SSD tables retain their explicit storage
policy. Source windows retire promptly; final tensor bytes remain live when
they are the actual inference representation.

The focused regressions must exercise the real reader and consumers, not a
fixture-only copy routine. They cover native format preservation, full and
selected source rectangles, tensors larger than the cap, conversion/packing
scratch, partial last windows, ordinary and routed ownership, failed reads,
failed preparation, and release while prepared handles remain usable. A common
planner quote must bound the same allocations observed at runtime. Source and
prepared owners must retire on failure as well as success.

Real-model verification repeats the unchanged one-socket and physical
two-socket workloads, retaining per-rank peak RSS and final admitted storage.
Loading overhead and final resident model size are reported separately; the
178.53 GiB observation is the original one-socket comparison. Numerical parity
must stay unchanged by window size. The shared window geometry currently passes
its focused tests, but the complete common pipeline and its RSS improvement
are not yet qualified.

The current caller audit traces all non-ngram CPU WeightPlan requirements through
`WeightManager::materializeCPURequirement` and the common reader/Writer/Store.
Replicated and sliced CPU getters use that same path, GDN noncontiguous operands
use declared source stripes, and expert loading publishes committed Store-owned
engines. Dynamic whole-expert movement consumes existing prepared banks rather
than rebuilding GGUF parents. CPU tensor-sharded dynamic cohort movement remains
explicitly unsupported and is not certified by these loading tests.

RAM ngram storage has a separate final-table consumer: `ImmutableRowTable` claims
its final PMA root once and reads bounded 16 MiB chunks directly into that root.
It has no additional whole-table source parent. Legacy whole-tensor APIs still
exist for other callers; canonical CPU model loading bypasses or rejects them.
GPU-target floating-source preparation remains a separate bounded-loading gap,
so this CPU work is not a claim that every backend's source path is closed.

Floating final-storage adoption now has a named constructor that retains the
same genuine CPU tensor and original allocation lease, without cloning it.
It rejects source views and incomplete storage. Its source-retirement contract
distinguishes a copied source from an aliased final tensor: the latter remains
live, including through service views. Focused tests cover FP32/FP16/BF16,
serial/grouped use after external handles retire, one fully occupied PMA root,
and final release. The owning runtime checks pass: four registrations, no skips,
clean driver evidence in `final-floating-build47-focused-r33/summary.json`.
The integrated loader and real-model RSS qualification remain pending.

The actual common native reader now passes 46 cases repeated 20 times: all 23
GGUF formats, non-square two-dimensional matrices and selected three-dimensional
expert ranges, both full-row and below-row native-group windows. The original
default mapping faults 28 KiB into the small fixture's source VMA; the corrected
demand-paged mapping leaves it at zero. Source byte order, unused destination
tails and the original PMA claim remain checked. Evidence is retained in
`/tmp/qfn-native-weight-staging-r264/reader-positive-result.json` and its
matching original-prefault negative control. These are component tests, not a
new real-model RSS measurement.

Planning calibration samples also use the common reader directly into their
one admitted final sample root. Their former full-source staging reservations
are removed while real packing scratch and GPU upload rings remain declared.
The original matching reader/PMA/sample object closure passes 13 cases and the four
focused lifecycle cases repeated 20 times in
`/tmp/qfn-planning-source-window-r267/component-test-r4/` and
`component-test-r5/`. Its successor preserves an explicit nonzero CPU allocator
for every native sample format; that successor awaits the coherent common-loader
build and is not covered by the earlier component receipt.

Final CPU embedding preparation now retains the same native tensor and original
allocation lease, without a second backend copy. Its real lookup/lifetime sweep
covers all 21 runtime quantized formats. An adversarial check exposed ten native
tensor classes that kept an internal view flag but inherited the owning-storage
query; their `is_view()` overrides now report borrowing correctly. The original
Q4_0 view-adoption test fails, while all-format view/nested-view rejection passes
with the correction. The complete embedding component has 25 passing cases;
the five focused cases pass 20 repetitions. Evidence is in
`/tmp/qfn-final-cpu-embedding-r268/component-test-r{1,2,3}/`.

GPU floating overrides have a bounded source interface that writes exact output
ranges into the existing pinned upload slots. Complete-row splitting preserves
source offsets, partial final transfers and the original event dependencies.
Failed conversion never uploads its partial output, and pending jobs retain the
source until lane retirement. A later-window failure exposed a separate cleanup
defect: the original implementation destroyed four ordering resources while an
earlier upload/repack was still pending. The corrected terminal failure path
drains accepted work before releasing its events and slots. The actual
pipeline/orchestrator with a tracked device-free backend passes all 28 owning
cases and four focused cases repeated 20 times in
`/tmp/qfn-bounded-gpu-source-r269/component-test-r{3,4,5}/`; the original-body
negative control fails in `component-negative-r1/` with clean driver evidence.
The new interface is joined into the source tree; the model-loader conversion
producer, native GPU loading and full-model RSS comparison remain pending.
`CPUFinalEmbeddingStorage`, `BoundedGPUWeightSource`, and
`GPUWeightSourceFailureRetirement` have explicit production preflight
registrations. Exact CPU allocator ownership, borrowed Q8_1 source lifetime and
immutable final Q8_1 adoption are assembled for the common build. These add
runtime object state, so their qualification requires a coherent rebuild of
every owning producer rather than mixed old/new component linking. No numerical
tolerance or model acceptance changed.

## Execution contract

`expert_compute=apportioned` assigns complete experts to participants.
`expert_compute=tensor-sharded` partitions every selected expert's intermediate
axis across the original ordered CPU participants. Gate/up own rows; down owns
matching columns. The loader copies native source blocks without requantizing
weights. Source formats determine legal cuts, including unequal block-aligned
partitions. Degree one uses ordinary complete-matrix ownership.

The implementation represents two through eight members without a two-rank
special case. Each member retains its actual world rank and NUMA endpoint. A
logical expert is present only when all of its original fragments are present.
Per-participant storage uses local intermediate width and the original expert
axis, rather than treating the tier's expert quota as a source tensor dimension.

Communication remains in `GlobalTPContext` and the existing ShmemSpin/MPI
backends. Members produce compact unweighted route records. A rooted gather
returns only live records; the root sums original participants in fixed order,
then applies the original route weight once. Other members publish metadata-only
returns. No new communicator implementation, per-layer communicator, or padded
payload is introduced. The declared root receive limit is identical on every
participant, independently of the smaller contributor backing. This keeps the
existing transport policy consistent without sending any unused capacity.

When the root socket also owns the GPU continuation, its local expert stage
borrows the immutable return ticket and produces unweighted fragments. A
separate rooted collective owns the ticket publication, combines the original
CPU participants, and applies weights into the root's mapped return bank.
Graph dependencies place this collective before remote return waits. The
memory plan counts that mapped bank once, alongside the distinct CPU combine
bank; contributors cannot acquire a duplicate canonical payload bank.

PMA owns explicit buffers and the existing shared arena. Its creator owns the
physical backing; other mappings borrow it. Typed conservative domain/process
commitments cover opaque native runtime storage. The process commitment survives
model retirement until successful MPI finalization. Request completion does not
stand in for context or mapping retirement.

## Defects exposed by the focused tests

- GPU source routing previously represented CPU fragment ownership only as an
  external scalar. The packet producer therefore omitted those routes. An
  immutable projection of the original cohort now identifies every actual
  consumer without manufacturing a whole-expert owner.
- Shared mapped dispatch exported captured capacity. Its producer now writes
  the union of live rows needed by the original consumer group. Physical shared
  bytes are counted separately from logical per-lane packet bytes; unused rows
  remain untouched.
- Explicit CPU tensor degree one attempted to construct a multi-member cohort
  during capacity admission. It now uses the existing full-matrix accounting
  semantics while preserving the authored compute policy.
- A fast MPI contributor could publish the next gather before a slower source
  completed the current gather. `MPI_ANY_SOURCE` could consume that next message
  and call it a duplicate. Probing only unmatched original sources preserves
  MPI's per-source ordering without adding a barrier, tag protocol or transport.
- The distributed numerical observer accepted the CPU joint intent, then rejected
  the same cross-rank domains in a later generic guard. Both typed domain views
  now retain the original approved membership; unrelated or changed domains
  still fail. The regression parses the actual two-socket model arguments.
- A CPU follower rebuilding the global ngram plan tried to obtain local GPU
  worker provenance for the remote continuation's cold resource BOM. Cold
  allowance projection now uses the existing installed typed limits without a
  local GPU context. Actual materialization still requires the original live
  worker, source and generation. Public request-wire round trips cover both GPU
  backends and RAM/SSD sources without fabricating native GPU readiness.
- CPU route gathering incorrectly bounded the root receive limit by each
  caller's own backing. For the model's one-row/top-eight/2,560-column decode,
  the contributor's 81,952 bytes and root's 163,904 bytes chose opposite sides
  of the 96 KiB ShmemSpin/MPI threshold. All ranks now pass the same declared
  root extent; the constructor separately validates root and contributor banks.
  Actual-stage tests check populations two through eight and live counts
  eight, one and zero. The native two-socket fixture uses the threshold-crossing
  shape. The original stage fails the scripted regression before transport;
  no deliberately deadlocked native negative control is required.
- Native preparation initially declared only the local rank's resources, while
  final admission correctly supplied the full topology. Preparation now declares
  the original global registry before creating local workers. Remote entries
  remain metadata-only; rank-local planning rejects remote nonzero charges and
  contributes each local native commitment once. PMA's exact registry validation
  remains unchanged.
- The graph factory tried to collapse the GPU's whole-expert policy and the
  CPU cohort's tensor-fragment policy into one scalar. The CPU joint cohort now
  retains its own original policy while the GPU graph receives its declared
  scalar policy. Unsupported mixtures within GPU lanes still fail.
- CPU grouped workspace reconstruction omitted the admitted NUMA placement and
  initialization policy. It now preserves those declarations while independently
  checking all extents; no storage equality or capacity check was relaxed.
- CPU joint preparation inherited the aggregate model's primary-weight owner,
  although admission correctly priced the fragments as routed-expert weights.
  Only the validated joint bindings now use that routed owner. The real model's
  565,248-byte claim matches one local Q4_K gate/up pack exactly: 400 prepared
  units of 1,408 bytes, rounded to a 4,096-byte allocation boundary. The fix does
  not increase the budget or alter the source-native cuts.

Each defect has a focused production-preflight regression. The literal old
capacity and gather implementations reproduce their failures. The old shared
copy also fails the native untouched-row assertion on CUDA and ROCm, while the
current producer passes. Dynamic symbol bindings confirm the negative controls
execute the old producer. Its performance comparison remains pending.

## Evidence completed on the development host

- Coherent core build: CUDA SM80/86/89/90 and HIP gfx906/gfx1100, with canonical
  spill checks enabled.
- Focused device-free gate: 34/34, no skips. This covers source formats, source
  ownership, local storage/BOM geometry, route folding and counter publication.
- Complete Unit registration: 989/989 passed after fixing the four initially
  failing suites. The existing `LayerWeightStreamer` fixture still contains
  unconditional skipped methods; the aggregate is a CTest registration count.
  This was the earlier integrated slice, not a receipt for every later edit.
  A subsequent complete Unit run passed 988/989 registrations and exposed an
  outdated graph source assertion; the assertion now recognizes the explicit
  joint-source guard and passes in the focused rerun. A fresh complete Unit
  receipt remains required before checkpointing.
  The changed CPU preparation keeps the allocation scanner unchanged and adds
  negative controls against passing an unaccounted or rebound placement.
- Affected production-preflight registrations: 65/65, no skips, clean driver
  window. This focused receipt is not the complete production-preflight gate.
  The existing remote-only three-process fork/join test was outside this
  two-socket selection and was not executed.
- Native CPU deployment gate: both two-socket ShmemSpin/MPI domain lifecycle
  cases pass with distinct physical-socket evidence. Each repeats 20 independent
  admissions and four large/empty/skewed/small gathers, verifies explicit
  retirement and retains the original process-pool admission until MPI finalize.
  Rooted variable gathers use the domain's original MPI publication communicator.
- The canonical packet suite and its five focused registrations pass with no
  skipped cases. MI50-only dispatch and shared-row tests also pass. Both driver
  observation windows are clean. These tests validate kernel/transfer behavior;
  their synthetic membership geometries do not qualify CPU model deployments.
- Native shared-row union: CUDA and ROCm pass. Two targets sharing one mapped
  matrix require 49 unique rows at the tested geometry, producing exactly 1,568
  bytes instead of 2,080 capacity bytes or 2,112 duplicated consumer bytes.
  Empty work writes zero hidden bytes and leaves the sentinel matrix unchanged.

The host has two physical CPU sockets. The earlier three/four/eight-process
communication diagnostics exposed the gather ordering defect, but do **not**
qualify deployment topologies. They are no longer acceptance targets on this
host. Native acceptance uses one MPI rank per actual socket; four/eight-socket
cases require matching physical hardware. Generic participant-count planning
and arithmetic remain covered by device-free tests. Complete production-preflight
qualification remains pending the colocated path. Native joint-routing tests now
pass with the corrected cold placement fixture; the earlier failure occurred
before dispatch.

Local evidence is retained under `/tmp/qfn-cpu-joint-qualified-r224/`, with
component provenance in the composition records. Those generated results must
not be committed to the source repository.

## Real-model acceptance next

The first model diagnostic uses the ordinary two-socket layout: world rank zero
owns the MI50 continuation and its socket-local CPU expert fragment; world rank
one owns the other socket's CPU fragment. Each rank is bound to its actual
socket. There is no extra GPU-only MPI rank or overlapping-rank workaround.
The logical per-layer quotas remain 64 GPU experts and 448 CPU experts; each
CPU member owns a native tensor fragment of the same 448 experts.

The existing frozen-history QFN observer is extended through the production
coordinated MPI worker loop. Only the authority supplies histories and publishes
full final logits. Every rank must provide its original source/configuration,
execution counters and final-owner retirement evidence. Numerical scoring uses
the unchanged reference pack and limits, with separate prefill/decode mean KL,
top-one agreement and top-five overlap. Reset equality remains required.

The initial two-socket model slice is Static with MTP off and RAM ngrams. Its
GPU-colocated CPU fragment publication and mapped-root declaration are now
implemented and built together. Native model qualification remains pending.
Dynamic expert balancing, active MTP, additional continuation
and placement combinations, CPU-joint PP and retained second-Runner reuse
remain explicit work. No result here promotes a performance default or
certifies those configurations. Previously prepared three-rank diagnostic
metadata was never executed and is superseded by this two-rank acceptance
target.

The native model diagnostic has progressed through global admission and GPU/CPU
weight loading. The corrected loader passes its owning Unit registration and
explicit production-preflight regression. Its literal old implementation fails the
same routed-only admission; the private repeated run passed 100 source cases.
An earlier attempt exposed two graph-construction mismatches:
the colocated CPU arena requests full intermediate width 640 instead of its
declared local width 320, and the follower creates its retained executor before
installing its original prepared engines. Both ranks retired cleanly, without
producing logits or new driver errors. Those defects now have passing actual
constructor lifecycle tests, including 180 sparse invocations after clearing the mutable
store and registry. A further regression exposed the same lost-NUMA-policy error in
the nested workspace binding; both declaration and binding now retain the
original backing's placement and initialization. Separate original-body controls
reproduce the engine and workspace failures. This remains a red model-admission
result until the combined native model run succeeds.

Additional focused receipts cover the global registry (three registrations),
rank-local BOM projection (four), graph policy and NUMA workspace (four), cold
resource projection (19, including native CUDA/ROCm execution), and common gather
extent (14, including the physical two-socket stage and repeated UPI/MPI domain
lifecycles). All have zero skipped cases and clean driver windows. These receipts
overlap and must not be added into a unique full-suite count. The current retry
uses the unchanged 25 histories and 178 reference rows. Startup failures do not
constitute numerical observations or justify changing the reference limits.

The corresponding source/build and focused evidence are
`/tmp/qfn-cpu-joint-qualified-r224/retained-arena-build22-result.json` and
`/tmp/qfn-cpu-joint-qualified-r224/retained-arena-focused-r12/summary.json`
(seven registrations, no skips, clean driver window).
That diagnostic is `/tmp/qfn-cpu-joint-world2-r244/`; it is explicitly
not eligible for release certification without fresh complete prerequisites.

The coherent pre-window Unit refresh passes all 990 CTest registrations in
138 seconds (`/tmp/qfn-cpu-joint-qualified-r224/unit-current-r15/summary.json`).
There are no CTest skips or new driver errors. Two existing internal
`LayerWeightStreamer` cases skip because their fixture lacks a valid
`WeightManager`; this receipt does not claim those cases executed. The cold
planning contract test now verifies that pricing an absent worker neither creates nor
qualifies its native resources. These results precede the bounded-source ABI
change below and do not certify that change or the complete preflight gate.

The r244 run's memory growth is consistent with the actual quantized tensor
directory and preparation policy. Each CPU socket owns 42.394775 GiB of prepared
fragments plus up to 31.381226 GiB of compact selected raw source parents. The
latter were retained by the original projection-source lifetime. The bounded
source implementation below removes that retention without increasing the final
prepared-weight budget. File-backed RSS is
shared and must not be summed as private physical storage. Metadata-only totals
and the source-lifetime audit are in
`/tmp/qfn-cpu-joint-footprint-audit-r244/REVIEW.md`.

## Bounded source preparation

The loader reuses the configured startup weight-loading cap (512 MiB by default) for one serialized
CPU source window per rank. The final routed-weight budget remains unchanged.
Its explicit `WeightLoadStaging` allocation is separate from GPU upload rings,
which may load concurrently; all claims remain in the same PMA.

Each temporary native batch retains the complete original CPU source selection
plus its actual ordinal subrange. Only its real compact tensor geometry and
addresses are published. Load a bounded batch of complete expert matrices,
prepare their final native-format engines, then release the actual source
backing before loading another batch. Retain immutable source/window metadata
and prepared engines, not the raw payload. Final publication still requires all
original selected gate/up/down matrices; an incomplete layer cannot become
ready. Logical source and expert identities must be independent of batch size.
An individual matrix larger than the configured bound must reject precisely
before allocation; silently raising the bound is forbidden.

Regressions force several batches, preserve sparse original expert IDs and
source-native cuts across all 23 loader formats, check actual staging-lease
high-water and final release, and prove identical prepared output under different
valid batch budgets. They exercise partial preparation failure and continued
engine lifetime after source release. The real two-socket run checks admission, full-logit accuracy,
resource observations and final retirement. OS file-cache RSS is recorded
separately from exact anonymous source backing, not treated as a second ledger.
The coherent build is `bounded-release-build27-result.json`; a test-fixture-only
correction is in `bounded-format-test-build28-result.json`. The r17 focused
window passes source release, staging, split-file/error cleanup, read economy,
and native CUDA/two-MI50 registration cleanup. Its two failed registrations
shared an omitted immutable flag in the new native-format fixture; both pass
in `bounded-format-focused-r18/summary.json`. Neither run has skipped cases or
new driver errors. These focused receipts do not replace complete Unit/preflight.

The first execution exposed actual view-retirement defects: floating-point views
formed offsets from released storage, while quantized views did not mark their
own borrowed access retired. The common tensor contract now uses its existing
release state and parent lineage before raw pointer arithmetic. A sibling remains
valid until its own or its parent's retirement; independently prepared engines
and decoded caches keep their own lifetimes. All-format lifecycle tests, repeated private
tests and original-body controls cover that correction. Native host-registration
cleanup remains green on CUDA and ROCm.

CPU windows use the loader's existing buffered file streams, with one seek per
selected expert and buffered traversal of strided source rows. Genuine split-file
tests show source payload mappings remain unfaulted, and truncated-read failures
release their exact staging claims. There is no added scratch allocation or broad
page eviction. The real QFN down-projection geometry uses 156 kernel reads for
2,560 rows, with 614,400 useful bytes and 614,160 intervening source bytes. Its
warm-file diagnostic measured 1.015 ms buffered versus 0.438 ms mmap; that is an
isolated startup-reader observation, not an inference or cold-SSD performance
claim.

The unchanged physical world2 retry in `/tmp/qfn-cpu-joint-world2-r248/`
confirms the memory reduction. Sampled process peak RSS falls from 150.526 and
121.809 GiB in r244 to 72.233 and 44.245 GiB. Anonymous peaks fall from 102.784
and 74.516 GiB to 71.339 and 43.799 GiB; file-backed peaks fall from 47.661 and
47.264 GiB to 0.813 and 0.417 GiB. The larger process also owns the full RAM
ngram table. These are per-process observations, not additive physical-memory
accounting or a claim that the sampling captured every transient allocation.

That run reaches graph construction and fails the strict original-source/cohort
lease check. Cold residency setup retains one preparation authority, while the
loading factory reconstructs another equal-valued owner map. Their cohort
objects are distinct. Loading and preparation must borrow the original retained
authority; weakening the identity check would conceal the ownership defect.
The run exits after 124.84 seconds, releases all live PMA allocations,
retires its native context, and has no driver findings. It produces no logits;
model admission and numerical qualification remain red.

Factory loading and public weight preparation now borrow the preparation retained
by cold residency setup. The source plan may be copied by value; its original
cohort objects and prepared-source leases may not be reconstructed. The actual
cold-load-to-graph-adoption lifecycle regression rejects a complete competing
lease with the same model and selection but a different cohort, then proves that
the original source survives loading, preparation and graph adoption without an
owner-map rebuild. Missing source authority and foreign model publication reject.
The strict production identity check is unchanged.

The coherent build is `source-authority-build29-result.json`. The scoped check
contains 25 registrations: 22 passed immediately; three registrations included
the same new fixture, whose staging declaration initially replaced its admission
certificate too late. The fixture now prices staging in the initial BOM and
retains the one final certificate. Its owning Unit registration and both lifecycle
registrations pass in `source-authority-focused-r20/summary.json`; no skips or
driver findings occurred. The unchanged real-model retry in
`/tmp/qfn-cpu-joint-world2-r251/` passes the original source check and completes
retained GPU serving-graph capture. It then fails the diagnostic observer's
local-only physical-address predicate for the distributed topology. Follower
failure cleanup additionally calls a root-only shutdown operation. This is still a red model result
without logits. The 188.33-second run has no driver findings, both model
processes have exited, and sampled peak RSS remains bounded at 73.468 and
44.242 GiB. This does not certify successful final-owner retirement on both
ranks or numerical accuracy.

The observer now validates the actual MPI rank and accepts only local aliases
or the original rank plan's exact canonical hostname. It does not rediscover
hardware or rewrite topology. A passive follower-record failure after the
worker loop returns now retires locally; an earlier follower failure invokes
the original fatal worker-command path. Authority cancellation retains its
exact-once shutdown behavior. The coherent build is
`observer-authority-build33-result.json`, with the observer test targets refreshed
in `observer-lifecycle-build34-result.json`. The complete coordinated-MPI Unit
registration plus its focused abort lifecycle pass in r21; the two observer
owning Unit registrations and two explicit preflight registrations pass in r22.
All six registrations execute without skips or driver findings. The five new
observer cases also pass 20 private repetitions each. The unchanged model retry
is `/tmp/qfn-cpu-joint-world2-r253/`. It passes observer admission and reaches
the first actual prefill, where the rooted CPU fragment-combine stage rejects
cold padded preflight for a 9-token prompt in a 64-row bucket. The stage already
consumes and publishes compact live sparse routes, but did not declare that
padded real-length contract. The run exits after 187.99 seconds with no driver
findings; neither logits nor complete successful retirement are qualified.

The narrow correction declares padded-prefill support only for the stage's
exact original canonical ticket binding. Native graph capturability remains
false, and the declared heterogeneous host-service boundary remains mandatory.
No capture policy, transport extent, reduction order or execution code changes.
The frozen production patch is `/tmp/qfn-rooted-padded-host-service-r254/`;
actual stage/cache and compact live-row regressions are being added before the
next unchanged model retry.

The rebuilt canary and three prefill Unit registrations plus bucket economy
pass in `rooted-padded-build35-result.json` and `rooted-padded-focused-r23/`.
The next unchanged diagnostic, `/tmp/qfn-cpu-joint-world2-r254/`, passes padded
preflight and reaches actual dispatch. It fails because host dispatch reads the
whole-expert arm of a joint-fragment residency snapshot, which intentionally has
no scalar owner for CPU experts. The original source map and cohort are already
retained by that snapshot. Dispatch must consume that map, and sparse dispatch
must authenticate each required cohort member through the same retained lease.
The narrow producer/consumer fix is preserved in
`/tmp/qfn-cpu-joint-dispatch-ownership-r255/`; focused tests and native
qualification remain pending. The failed run exits in 184.40 seconds with no
driver findings and no logits; it does not qualify final-owner retirement.

Dispatch now borrows `originalOwnerMap()` from its retained residency lease;
sparse delivery authenticates the actual cohort members from that same lease.
A separate zero-weight defect was reduced before another model retry: the
host descriptor retained zero entries while the CPU producer omitted them,
violating exact packed counts and original-slot alignment. Joint sparse delivery
now omits those inactive entries before compaction. Whole-owner delivery and
router evidence retain their existing behavior. The two focused cases pass 20
repetitions each, and the identical tests fail against the original ownership
body and the pre-filter sparse body. Canonical registry/dispatch/source
registrations pass in `joint-dispatch-contract-focused-r26/`.

`joint-native-boundary-focused-r25/` also passes the actual CUDA and MI50
mapped-ticket/cache contracts, two-socket compact gather economy, compact
arithmetic and ordinary dispatch checks: five registrations, no skips or driver
findings. The native gather observes actual receive counts, live-byte counters
and poisoned unused tails under capacities one and 64. Both capacities select
the original MPI payload lane; this is not a new shared-memory-path claim.

The unchanged model retry `/tmp/qfn-cpu-joint-world2-r255/` aborts during first
prefill after 186.51 seconds, before final logits. It reports only the original
collective cancellation. Its driver window is clean, both model processes exit,
and the owned MPI session directory is empty; successful final-owner retirement
remains unqualified. Sampled peak RSS is 73.466 and 44.243 GiB.

Source tracing exposes a scheduling-context mismatch: the captured GPU parent
passes its existing GPU context into the CPU host service, whose old guard
requires a CPU context even after authenticating the original CPU domain.
The narrow fix accepts CPU scheduling or the exact GPU named by the original
bound ticket; native CPU source, borrower and epoch checks still precede work.
It also logs each specific rooted-publication rejection before cancellation can
terminate MPI. The cumulative patch is
`/tmp/qfn-cpu-rooted-host-service-context-r258/`, built in
`joint-context-build39-result.json`. The next model run completes inference;
the remaining observer and numerical results are recorded above.

The raw-view test is named `V2_Integration_TensorRawViewLifecycle`. Nearby new
CPU source, arena, ownership and geometry tests likewise use lifecycle or
contract names. Registration comparison preserves all 21 affected entries and
adds `V2_Integration_CPUJointSourceAuthorityLifecycle`; obsolete names are removed
and the preflight labels retained. Repetition alone is not recorded-input replay.

The exact two-rank configuration and unchanged reference metadata have been
prepared under `/tmp/qfn-cpu-joint-world2-diagnostic-r232/`. Native model
qualification remains pending. Its explicit MI50 UUID selection avoids relying on
the host's current ROCm ordinal zero, which is the RX 7900 XTX. A cold launch
check also exposed a missing session-base directory in the diagnostic driver;
the driver now creates that fresh owned directory before invoking the unchanged
strict MPI cleanup wrapper, rejects reused/symlink bases, and has no
oversubscription diagnostic option. All 57 corpus/driver tests pass. Focused
regressions cover the rooted collective's ordering, immutable producer ticket,
exact-once route weighting and dirty/empty row reuse. These component checks
do not substitute for the combined native model run.
