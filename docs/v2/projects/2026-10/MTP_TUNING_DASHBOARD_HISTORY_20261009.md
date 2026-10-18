# MTP dashboard history archived on 2026-10-09

Historical evidence only. The current qualification status is in
[the MTP dashboard](../2026-06/MTP_VLLM_STYLE_TUNING_DASHBOARD.md).

# vLLM-Style MTP Tuning Dashboard

## 2026-10-08 r22: prepared-context retirement and configuration codec

Terminal reset failures now stop before retirement/reuse publication. The focused
regression rejects the previous swallowed-exception behavior. Native transfer
directories retain immutable prepared sources, so their reuse seal follows
participant destruction and verifies exact original engine handles in each stage.
Mutable physical residency still restores through its live fabric. This checks
metadata and shared ownership only; it does not read or hash weight/cache bytes.

The full configure caught 39 new NO_MPI Integration registrations inheriting a
two-rank default; they now declare one process explicitly. The complete build
also caught the missing first_model_layer field in the strict saved-configuration
codec. That field now round-trips with nonzero stage origins and terminal MTP
rows. Four stale source checks now enforce the compact stage representation.

Validation: 240 focused C++ cases, 200 repeated case executions, four explicit
preflight gates and the failing old-reset negative control. All 290 qualified
ACTIVE production bindings remain unchanged. The consistent Integration build is
tracked separately at `/workspaces/llaminar/parity-results/opencode-tool-calling/resource-growth-work/hybrid-candidate-build-20261008-r22c/controller.json`; it is not yet a native serving
certificate. Focused receipt: `/workspaces/llaminar/parity-results/opencode-tool-calling/resource-growth-work/hybrid-stage-ownership-20261008/result-r22.json`.

Remaining: full candidate prerequisites, native reuse/retirement and prefix
restore/resource proofs, setup-scratch lifetime, intrusive/global epoch audit,
matching Release overlay and both hybrid app sessions. No image rebuild, commit
or push occurred.


## 2026-10-08 r21: stage-scoped completed prefix observations

Completed rank-local PP request summaries now preserve each stage's actual
admission interval and completion epoch. Stable stages at epochs 100 and 1
previously looked like movement under a flattened MIN/MAX comparison. A maximum
also hid movement in the smaller-epoch stage. Scoped comparisons now distinguish
unchanged placement, publication during a request and publication before the
next admission. Missing, changed or stale scopes fail. Actual cache restoration
already validates each participant's own fingerprint and retained lookup; these
observations remain passive metadata and perform no payload hashing or reading.

Root harvest captures the scoped values once. Cached request summaries and the
HTTP response reuse that observation without a live probe. Pipeline HTTP summaries
use schema 2 with null unscoped epoch fields; benchmark prefix metadata uses the
same serializer. Python validates exact unsigned integers and independent scopes
and still requires actual fresh/full prefix outcomes. Both runtime-summary
schemas retain movement-observer coverage.

Validation: 24 pipeline, 43 prefix-flow and 169 HTTP C++ cases; 225 production
Python and 32 movement Python cases pass. Seven focused canonical preflight
registrations pass, including three new explicit entries. Four new pipeline
cases pass twenty repeats under partial ASan (80 case executions). The C++/Python
bridge covers both declared four/eight widths and both vendor orders, with
72 rejected metadata corruptions plus fresh/full restore validation. These are
device-free tests; no GPU execution or model serving was run in this slice.

The broader HTTP check exposed an old topology object in the standalone diagnostic
link. Explicit current topology and HTTP-stats producers fix that diagnostic
composition; the full 169-case HTTP run is green. All 290 qualified
ACTIVE production source bindings remain intact. The candidate changes public
value layouts and IInferenceRunner's vtable, so a consistently rebuilt serving
binary remains required. Evidence: `/workspaces/llaminar/parity-results/opencode-tool-calling/resource-growth-work/hybrid-stage-ownership-20261008/result-r21.json`.

Remaining: intrusive/global scalar-epoch consumers; native maintenance and
prepared-context physical sealing/teardown; setup-scratch lifetime; complete
candidate qualification; then both hybrid app sessions and branch publication.
No image rebuild, commit or push occurred.


## 2026-10-08 r20: independent pipeline movement publications

The candidate now retains one publication namespace per PP stage and elects one
root inside each native TP domain. Completed work totals add; transaction IDs,
policy receipts, demand windows and progress generations stay with their owner.
Native archive exports translate compact journal rows into global layer IDs.
Counter keys include the fixed layer interval, including when stages reuse a
device. HTTP and terminal JSON, the Python observer, and frozen topology reports
retain identical stage identities. Shutdown publication marks each already-retired
stage drained while preserving failures. This is passive reporting, not a proof
of physical prepared-context restoration.

Validation: 20 pipeline C++ cases, 18 counter cases, 32 movement Python cases and
151 server-policy cases pass. Twenty repeats pass normally and under partial
ASan (400 case executions each). Six negative-control cases reproduce the old
rank selection and archive layer defects. Four new focused preflight gates pass,
including production C++ JSON/counter exports consumed by Python for declared
four/eight-device topologies. Registration auditing resolves 58 selected C++
gates / 94 case selections; 14 optional native eight-GPU skip properties remain.
These tests are device-free. No real model or GPU serving ran in this slice.

ASan also caught a temporary-view lifetime mistake in the new fixture. Retaining
the owner fixes it, and the metadata API now rejects such rvalue views at compile
time. All 290 qualified ACTIVE production source bindings remain intact.
The candidate's publication values and orchestration virtual interface changed;
real serving requires a consistent rebuild, not an old ABI diagnostic executable.

Evidence: `/workspaces/llaminar/parity-results/opencode-tool-calling/resource-growth-work/hybrid-stage-ownership-20261008/result-r20.json`.

Next work: audit scalar movement-epoch consumers (the old max reduction remains),
prove native maintenance/prepared-context sealing and teardown, then prefix
restore, bounded resources, complete candidate qualification and the two hybrid
OpenCode app sessions. No Docker image rebuild or branch publication occurred.


## October 8, 17:54 UTC: pipeline startup and retained placement handoff

The live root now admits all MoE PP stages jointly, freezes each global layer
interval with the model-aware placement resolver, and carries those immutable
placements through the prepared-context contract. Runtime construction creates
fresh checked stage authorities and registries from the original PMA. Source
identity is tied to the actual ModelContext without extending its lifetime;
metadata checks compare projection shapes/formats, never model/cache payloads.
The first native dry run exposed the production loader adapter, and the focused
fixture now exercises that exact adapter rather than a concrete-loader mock.

A focused negative control found that retained MTP reuse omitted predictor
replication versus TP sharding. The shared dense/MoE gate and pipeline metadata
identity now reject that ownership change. Active off/fixed/dynamic policies
remain legal inside unchanged retained capacity and weight ownership.

**1,038 focused cases in 34 groups pass**, including 94 planning/setup cases.
Seven new cases pass twenty normal and twenty partial-ASan iterations
(**140 executions each**). The prepared handoff sweeps 23 source formats,
both vendor orders, four MTP storage/execution lanes and all three maintenance
modes at four/eight participants: **1,104 combinations**, with two fresh runtime
lifetimes and two stages per combination. The canonical registration audit
finds **56 relevant entries selecting 74 cases**, including explicit regressions
for the predictor ownership defect and both pipeline widths.

**All 16 real-model dry runs pass**: Qwen 3.8 dense and Qwen 3.6 MoE, both
ROCm/CUDA stage orders, default dynamic MTP, fixed depth 7 with retained 15,
MTP disabled, and disabled with retained capacity. Each admits **262,144 context**,
16 GiB RAM prefix capacity per participant and a shared 32 GiB disk tier. The
MoE cases prove one aggregate authority across both stages. Native process and
driver cleanup pass. These are current-root admission checks against the local
qualified support library; they do not certify serving or captured model graphs.
The earlier 18 native component passes/14 hardware skips retain their unchanged
source bindings. Eight-GPU native hybrid coverage still needs unavailable GPUs.

All 290 qualified production source files remain unchanged; free workspace
space is 218.2 GiB. Evidence:
`hybrid-stage-ownership-20261008/result-r19.json` and
`pipeline-model-admission-20261008-r19c/controller.json`.

Remaining: stage-aware movement/status aggregation, native maintenance and
prepared-context sealing/teardown, actual prefix restore, sequential CPU scratch
lifetime, complete candidate qualification, and both hybrid OpenCode app sessions.
The stage handoff is wired and component-tested; live serving is not yet green.
No Docker image build, commit or push was performed.

## October 8, 17:16 UTC: shared stage-owned residency setup

The ordinary live runner now uses one checked `MoEOverlayResidencySetup`
constructor for histogram, residency authority, Dynamic economy catalog and
empty prepared-bank endpoints. The replaced inline constructor is removed.
Compact histogram and endpoint storage retain the exact model-global origin;
terminal MTP banks do not move the main-token histogram boundary. The factory
rejects foreign certificates, sibling-stage physical resources, unfinished or
changed quotas and mismatched maintenance storage before publishing owners.

**1,031 focused cases in 34 groups pass**, including 87 planning/setup cases.
The four/eight-participant matrix composes actual admission, placement freezing,
runtime construction and the sealed pipeline binding: **23 source formats**,
both vendor orders, disabled/fixed/dynamic MTP, Off/Observe/Dynamic maintenance
and apportioned/projection compute (**1,656 combinations; 3,312 stage sets**).
The ordinary control covers CPU and CUDA/ROCm widths 1/2/4/8. Host demand claims
retire on normal destruction and on failure after histogram construction.
Thirty foreign-owner rejections per pass cover both pipeline widths.

The five new cases pass twenty repetitions (**100 executions**) and twenty
partial-ASan repetitions (**100 executions**). The canonical registration audit
resolves 53 relevant explicit entries to 67 cases. The previous **18 native
passes and 14 hardware skips** retain exact unchanged native source bindings;
the new runtime constructor has component coverage, not a native model claim.
All 290 qualified production source files remain unchanged. Free workspace
space is 218.4 GiB. Evidence:
`hybrid-stage-ownership-20261008/result-r18.json`.

The initial ordinary control needed a valid CPU NUMA observation and its
rank-local demand descriptor, which live setup retains separately from discovery
capacity. Its original small Qwen query geometry correctly rejected eight TP
slices; the final source supplies enough real directory geometry.

The live MoE PP guard remains: startup still needs to retain stage admission
through prepared-weight reuse, install every stage binding, and compose native
maintenance, movement/statistics and teardown. CPU setup lifetime, actual prefix
payload and full-model cleanup proofs, complete candidate qualification and
both hybrid app sessions remain. No image, commit or push was produced.

## October 8, 16:49 UTC: joint MoE pipeline admission

The candidate planner now uses `AdmittedMoEPipelineMemory` to derive exact
stage GGUF manifests, assemble all fixed owners, and resolve every stage's
expert capacity in one PMA transaction. Every immutable stage result retains
that same aggregate certificate. Automatic quotas remain adaptive. Request
costing resolves each layer's own expert authority, including the later PP
root, instead of consulting a model-wide parent plan.

The shared adapter accepts the current physical observation and performs no
model payload reads or device allocations. It retains verifier geometry on
nonterminal stages while assigning predictor sources only to the terminal
stage. Only typed capacity exhaustion advances the declared graph-row search;
malformed input remains fatal.

**1,026 focused cases in 34 groups pass**, including 82 admission/planning/cost
cases. Four/eight-device metadata coverage exercises **23 loadable GGUF source
formats**, off/fixed/dynamic MTP, auto/apportioned/projection compute and both
vendor orders (**828 combinations**). Shared host exhaustion, GPU exhaustion,
failed-transaction reuse and missing transfer evidence have focused checks.
The eight new cases pass twenty repetitions (**160 executions**); the six
admission cases pass twenty partial-ASan repetitions (**120 executions**).
The registration audit now resolves 50 explicit entries to 62 cases.

The previous **18 native passes and 14 hardware skips** retain exact unchanged
native source bindings; no native run was repeated for this metadata change.
An initial incremental diagnostic mixed the new candidate value with the old
automatic-startup caller ABI; rebuilding the complete caller/producer closure
resolved the tooling abort. All 290 qualified production source files remain
unchanged, and 133 candidate files are bound. Free workspace space is
218.5 GiB. Evidence is `hybrid-stage-ownership-20261008/result-r17.json`.

This is component evidence, not a complete candidate or Release certificate.
The live runner still rejects MoE PP before weights because stage runtime
owners have not yet been constructed. Remaining work is that runtime producer,
CPU setup lifetime sharing, actual prefix payload and admission-only cleanup
proofs, complete qualification, and both hybrid Python-app sessions. No image,
commit or push was produced.

## October 8, 16:25 UTC: unused RCCL retirement

The hybrid admission-error process-exit crash is reproduced and repaired in the
candidate. An unused pooled RCCL coordinator launched a synthetic allreduce
during process teardown, which crashed inside HIP. The focused regression
passes its test body on the original implementation, then exits with SIGSEGV;
the process exit code is essential evidence.

Cleanup now requires native communicator finalization, retires unused and used
owners through that same path, and performs no priming allocation/collective.
Missing finalization is rejected at initialization. Native teardown errors are
fatal with a precise diagnostic; handles are not silently abandoned.

**18 current native cases pass**: two/four-ROCm unused teardown and pooled exit,
runtime-generation reset/reuse, rejected hybrid admission and all 12 four-GPU
ordinary/fixed/dynamic generation cases. The direct cases include **20 cycles at
each device count (40 total)**. Fourteen native eight-device cases are skipped
for missing hardware. Driver and owner-retirement evidence is clean. KFD can
publish process removal shortly after waitpid; the diagnostic records that
latency and waits for the same retirement boundary without retrying a test.

The prior **944 focused cases in 33 groups**, 400 repetitions and 640 ASan
executions remain bound to unchanged implementation sources. Registration was
refreshed: 48 explicit registrations select 54 cases, with six new unused-RCCL
preflights. Evidence is `hybrid-stage-ownership-20261008/result-r16.json` and
`unused-rccl-retirement-native-20261008-r2/controller.json` under the work root.
All 290 qualified production source files remain unchanged;
126 candidate files are bound. Workspace free space is
218.9 GiB. No image, commit or push was produced.

This remains focused evidence. The MoE pipeline startup producer, CPU setup
lifetime sharing, actual prefix payload movement, full-model admission-only
cleanup, complete candidate qualification and both hybrid Python-app sessions
remain outstanding before committing/pushing.

## October 8, 16:05 UTC: four/eight-device preflight and retained forwards

Preflight now projects four and eight devices through middle PP stages, uneven
TP groups, both vendor orders, MTP off/fixed/observe/dynamic policies, exact
channel owners and joint memory admission. Shared disk archive scratch is priced
once across the pipeline; RAM tiers remain independent per participant. The
stage planner feeds all fixed contributions into one capacity transaction.

Native four-GPU tests found and fixed a retained-width defect: active fixed
depth 7 selected eight verifier rows, while followers assumed the retained
16-row capacity. Setup now captures widths 16/8/4/2, both outcome variants,
and admission prices them all. Every supported retained depth 1–15 and legal
request depth is covered. A capture-reuse assertion also exposed the ordinary
device-token forward being deferred until first decode; that already-admitted
executable is now captured during serving setup.

All **944 focused cases in 33 groups**, **400 repeated case executions** and
**640 additional ASan executions** pass. ASan covers changed implementations,
not the complete supporting dependency DSO. All **12 native four-GPU cases**
pass: ordinary/fixed/dynamic, both vendor orders, rows 128/256, exact token/KV
oracles, verifier inventory, main-forward reuse from startup and all-helper
reuse after request reset. Driver and retirement evidence is clean. Twelve
eight-GPU cases are explicitly skipped: this host has two CUDA and four ROCm
devices; the extension requires four of each. Device-free eight-device cases
passed. The canonical registration audit resolves 42 registrations to 48 cases.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r15.json`
and `four-eight-gpu-preflight-native-20261008-r5/controller.json`. The receipt
binds 124 candidate source files and verifies all 290
qualified production files remain unchanged. This is focused component/native
evidence, not full candidate preflight or a hybrid model certificate. No new
image, commit or push occurred. Workspace free space: 219.1 GiB.

The MoE startup producer still must install these jointly admitted stage owners;
the precise fail-closed checks remain. CPU setup lifetime sharing, actual prefix
payload movement, admission-only cleanup and the earlier failure-unwind crash
still need closure, followed by complete candidate qualification and both hybrid
Python-app sessions on the locally rebuilt Release overlay.

## October 8, 14:24 UTC: authored pipeline and fixed-memory scopes

The isolated compiler now retains the authored MoE PP parent and immutable
stage projections, with actual devices, collective policy, global main-layer
bounds and terminal roles. Child configuration removes parent placement
selectors. Repeated normalization preserves each stage's compute policy;
automatic/off/fixed/dynamic MTP keeps the parent serving policy. These are
metadata projections, not installed runtime owners.

Fixed-memory assembly now requires the exact retained graph family. Main
capture frontiers use owned main layers; CPU setup sizing uses the owned routed
interval, including terminal sidecars when retained. Full parent source metadata
stays intact. Foreign scopes, mismatched continuation/histogram/controller
geometry and nonterminal sidecars fail before physical admission.

All **822 focused cases in 29 groups** pass. The **nine new cases pass 20
repetitions (180 executions)** and **180 additional ASan executions with leak
detection**. The changed compiler, normalizer and memory adapters are instrumented;
supporting objects/DSO are not a whole-program sanitizer build. Three previous
implementations fail the exact regressions: PP rejection, a raw-block capture
overcount and foreign-stage CPU descriptor access. Scope coverage includes both
GPU orders, widths 1/2 and all 24 CPU source formats. The combined diagnostic
startup consumers compile. Three explicit preflights register topology, memory
projection and CPU setup scope.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r14.json`,
`all-focused-r27-full.log`, and the `projection-*-r14` logs. The receipt binds
**112 candidate files**, revalidates **290 unchanged
qualified production files**, and retains historical native evidence separately.
This is not complete candidate preflight or a hybrid server certificate. No new
native/model run, image, commit or push occurred; **220.0 GiB** remains free.

Startup still rejects missing stage owners before weights or ordinary dense
admission. The next producer must retain parent pipeline-channel buffers and
price shared archive scratch once across stage BOMs, while preserving independent
RAM tiers and correctly typed serial setup ownership. Then submit one joint
capacity transaction, construct/seal the runtime bindings, resolve native cleanup
and failure unwinding, qualify the local Release overlay and complete both hybrid
app sessions. Details are in `startup-next-r14.txt`. The preceding goal turn
completed joint capacity; this turn advances its topology and fixed-scope inputs.

## October 8, 13:40 UTC: joint pipeline capacity and replica grants

The isolated candidate now admits one or several owned routed-layer intervals
through the same production capacity implementation. Distinct fixed allocations
on shared CPU/GPU resources contribute to one typed BOM. Every stage retains the
same final `PhysicalMemoryPlanAdmissionCertificate`, while its quota and copy
counts remain compact and retain global layer identity. All fixed banks, exact
quotas and migration sources precede optional placement. Conflicting observations
and alternate names for one physical allocator are rejected before publication.

`MoEOverlayCapacityAdmission::resolvePipelineCapacity` searches complete pipeline
candidates. It first tries all requested cache maxima, proves every enabled
minimum together when bounded selection is needed, and then maximizes grants in
authored stage order. Retained grants remain exact and transfer lanes retain
their admitted geometry. Explicit movement-off and replica-cache-off remain
separate supported policies. Ordinary single-stage callers use this same path;
there is no alternate byte ledger or automatic execution-mode change.

All **794 focused cases in 27 groups** pass. The **13 new regressions pass 20
repetitions (260 case executions)**, and another **260 case executions pass
AddressSanitizer with leak detection enabled**. The resolver, policy adapter,
transfer-directory geometry, PMA and fixture are instrumented; the supporting
DSO is not a whole-program sanitizer build. Coverage includes all **21 quantized
source formats plus FP16/BF16/FP32**, CPU/CUDA/ROCm, both GPU orderings, TP widths
1/2, compact nonzero intervals, shared RAM/GPU overcommit, exact byte boundaries,
later-stage fixed/minimum/retained grants and independent ledger-lease retirement.
The previous resolver fails the exact new resource-alias rejection test. The
changed startup, factory, graph and orchestrator consumers compile together.
Unit registration and the explicit
`V2_Integration_MoEPipelineAggregateCapacity` preflight entry own the regressions.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r13.json`,
`all-focused-r26-full.log`, and the `pipeline-capacity-*-r13` logs. The receipt
binds **101 candidate files** and revalidates **290 unchanged qualified-source
files**. This remains component evidence, not a complete candidate preflight or
hybrid server certificate. No model/cache payload hashing, native GPU execution,
new Docker image, commit or push occurred in this slice; about **221 GiB** is free.

Startup does **not yet produce the stage bindings**. Preserve authored PP during
normalization, construct each stage's fixed BOM with its own captured-graph and
CPU-service geometry, submit the contributions to the joint API, publish one PMA,
and construct/seal the runtime owners. The implicit MoE PP guard remains until
that producer is complete. Then validate actual prefix payload movement,
admission-only cleanup and failure unwinding, build the local Release overlay,
and finish both dense/MoE hybrid app sessions before review and commit/push.
The preceding goal turn was status-only and was revalidated; this turn changed
production admission and established the evidence above.

## October 8, 13:01 UTC: PP children retain their own expert runtime

The production PP child projection now selects a sealed
`MoEOverlayPipelineStageBinding`. It authenticates the owned main interval,
terminal routed MTP banks, ordered continuation devices, TP weights/backend,
maintenance mode, initial expert owner map, histogram lifetime and prepared-bank
registry. Equal dimensions and devices cannot conceal a different expert map.
Single-device PP and nested TP use the same projection. The old model-wide
runtime broadcast into every stage has been removed. Every child handoff is
checked before the first stage prepares weights, and stage owners retire after
child graph destruction. This adds no payload hashing or physical-memory ledger.

The pure configuration methods now live in `RankOrchestratorConfig.cpp`, letting
focused regressions execute the production projection without initializing GPU
runners. Three existing nested-TP tests now call that projection instead of
copying its implementation. The new binding tests are registered in both Unit
and the explicit `V2_Integration_MoEPipelineStageRuntimeBinding` preflight entry.

All **781 focused cases in 27 groups** pass. The **nine new cases pass 20
repetitions (180 case executions)** and cover CPU/CUDA/ROCm topology metadata,
single-device/TP widths, Off/Observe/Dynamic residency, nonzero stage origins,
terminal auxiliary banks, foreign owners, changed destinations and incomplete
installation. Exact original configuration validation fails two focused negative
controls: it accepts model-wide expert inheritance and missing/swapped stage
bindings. The current startup, factory and graph consumers compile. Build/link
failures from the diagnostic harness's initial dependency/include setup are
retained; the final combined build and test transaction passed.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r12.json`
and `all-focused-r25-full.log`. The receipt binds **99 candidate files** and
revalidates all **290 unchanged qualified-source files**. No native/model test or
new sanitizer execution is claimed by this slice. This remains focused evidence,
not complete candidate preflight or hybrid inference certification. No commit or
push was made; about **221 GiB** remains free.

The caller audit redirects the earlier controller/replica task: the legacy host
rebalance controller has no production construction site. The remaining target
is the device-owned production startup path. Preserve authored PP through
normalization, admit the complete pipeline through one PMA, construct each
stage's runtime and publish `moe_pipeline_stage_bindings_`. **That producer is
not implemented yet**, and the implicit MoE PP guard remains. Then qualify
actual prefix payload movement and admission-only/failure cleanup, build the
local Release overlay, complete both requested hybrid app sessions, and
review/commit/push. The preceding goal turn was status-only; this turn produced
production changes and reproducible regression evidence.

## October 8, 12:21 UTC: residency planning and proposals authenticate their pipeline stage

The residency authority now owns one exact global layer interval. Snapshot
construction, frozen-window intake, transaction validation and authoritative
export/adoption reject equal-shaped evidence from a different stage. Participant
ownership and service/movement cost tables retain only compact owned rows;
public plans, routing observations and migration records retain global model
identities. Forecast updates and publication hysteresis use the same checked
translation. Off, Observe and Dynamic maintenance all retain stage ownership.

Distributed proposals derive their origin from the existing authenticated
histogram metadata. Receivers require their admitted origin, without adding
payload rows or packet bytes. The MPI publisher validates that origin at setup,
send and receive; its consumer and startup call site compile. This turn does not
claim live MPI proposal transport or complete hybrid inference.

All **735 focused cases in 24 groups** pass. Five new cases each pass **20
repetitions**, covering origins 0, 32, 40 and near the integer limit; actual
observed decode/prefill/verifier batches; foreign/ragged cost evidence; temporal
smoothing; and publication/follower adoption. The four authority cases also
pass **20 AddressSanitizer repetitions (80 case executions)** without reported
memory errors. The authority and fixture are instrumented; this is not a
whole-program sanitizer build. Exact prior authority and decoder source fail
three regressions normally. Two explicit ProductionTestPreflight registrations
cover these defects. The first fixture build referenced a nonexistent histogram
config field; ownership already owns its origin. The first economy oracle summed
overlapping endpoint transfers; it now checks their actual critical path. Those
failed attempts are retained alongside the corrected evidence.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r11.json`
and `all-focused-r24-full.log`. All **91 candidate source bindings** and all
**290 qualified-source bindings** were checked. The earlier physical inbox and
transport receipt remains source-identical. The broad r10 native prefill receipt
has five changed source bindings and is historical evidence, not qualification
of this combined candidate. No new native/model run occurred, and no commit or
push was made.

Remaining: stage-aware controller/replica consumers, authored PP preservation
through startup normalization, and per-stage PMA-owned plan/authority/runtime
bundles. The MoE PP guard remains until that implementation is complete. Then
qualify actual prefix payload movement and admission/failure cleanup, build the
local Release overlay, complete dense/MoE hybrid app sessions, and review/commit/
push. The preceding goal turn was status-only; this turn produced code and
reproducible regression evidence.

## October 8, 11:51 UTC: stage routing and captured rebalance use compact runtime rows

The local routing, rebalance and prefill consumers now obtain the runtime base
from its first owned global model layer. Kernel commands retain compact storage
rows; graph parameters, public runtime lookups and per-layer identities retain
global model IDs. Routing preserves the all-layer selector. Rebalance preparation
and capture admission reject foreign apply selectors or mismatched table geometry.
Current-batch prefill additionally authenticates layer, expert and top-k geometry.
Prefix rehydration uses the same checked row translation; real payload execution
of that path still needs the combined Release/model qualification.

The new device-free routing regression executes the production stage with a
recording kernel and CPU-owned storage for both backend launch identities. It
covers default/current/all/explicit selectors, nonzero and near-integer-limit
origins, foreign rows and malformed geometry. It passes twenty repetitions; the
original implementation fails by looking up model layer zero in a later stage.

Two explicit native regressions cover CUDA and ROCm at stage origins 0, 32 and 40.
Real prefill graph construction and mirrored-table preparation accept every owned
row and reject foreign layer or shape bindings. A retained production rebalance
graph collects exact changing demand at scales 1, 1000, 0 and 7; the same graph
survives large-to-zero replay without stale counters. Both cases pass twenty
repetitions (40 case executions), with a clean driver window and complete owner
retirement. The original prefill and rebalance stage implementations fail both
new regressions normally, without a native crash.

Focused validation is **671 cases in 23 groups**, including the existing prefill
capture-contract suite. The new routing, prefill-preparation and captured-rebalance
regressions have explicit `ProductionTestPreflight` registrations. Evidence is
`resource-growth-work/hybrid-stage-ownership-20261008/result-r10.json`,
`hybrid-stage-prefill-native-20261008-r4/` and
`hybrid-stage-prefill-original-20261008-r2/`. The 86 candidate source bindings and
all 290 qualified-source bindings were checked. Earlier transport/inbox native
proof remains source-identical. This is focused evidence, not combined preflight
or a Release hybrid-model certificate.

Remaining work is stage-aware residency/controller planning, authored PP startup
normalization, per-stage PMA-owned plan/authority/runtime installation, combined
qualification including prefix payload movement and failure cleanup, then the
dense and MoE hybrid app runs. The MoE PP guard remains until its implementation
is complete. No commit, push, Docker rebuild or new model stress run occurred.

## October 8, 11:14 UTC: physical movement and mapped inboxes bind exact pipeline stages

The immutable command batch now retains the fabric's stage origin. Acquisition,
read-only scheduler predicates and every physical completion edge authenticate
origin, row count, expert count and participant count before acknowledging work.
Device commands keep compact rows. Physical owner endpoints, durable cycles and
shadow-slot keys translate those rows once into global model layers; transfer
bytes, command counts and device command ABI remain unchanged. Execution identity
includes only the additional setup metadata, never model or cache payloads.

Physical allocation-lifetime tracking now requires explicit immutable stage
geometry, including ranks with no local expert allocations. Initial enrollment,
wave admission, source lookup, staging, publication, abort and retirement reject
foreign stages. The production fabric supplies that geometry from its admitted
ownership. Mapped destination inboxes also reject foreign geometry before they
claim a wave or publish descriptors, and stage identity remains required for reuse.

All **566 focused cases across 21 groups** pass. Ten new device-free cases pass
**20 repetitions** each, covering Dynamic, prepared-context restore, LLEP,
no-movement receipts, near-limit origins, empty ranks and overlapping allocation
lifetimes. Exact original physical-movement and lifetime implementations both
fail the new regressions; their copied source bindings and failing results are
retained. Three focused entries explicitly register these defects in preflight.

Six native CUDA/ROCm cases pass **20 repetitions (120 executions)**, including
two new stage-origin inbox cases, the two existing inbox controls and both
captured service-publication cases. A fourth preflight entry registers the new
native inbox proof. Complete driver windows and final GPU-owner retirement pass.
Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r9.json`
(**80 candidate files**, **290 qualified source bindings unchanged**) and
`resource-growth-work/hybrid-stage-device-boundary-native-20261008-r1/controller.json`.
These are focused device-free/native Integration results; combined canonical
preflight and Release/model hybrid inference remain unverified.

Remaining: local rebalance/routing/prefill row coordinates, stage-aware residency
planning, authored PP preservation and per-stage PMA-owned orchestration bundles.
Then verify native admission/failure cleanup, qualify the combined binary, run
dense/MoE hybrid Release app sessions and review/commit/push. The prior status
turn made no implementation progress; this turn produced code and native proof.

## October 8, 10:53 UTC: stage-bound controller metadata and captured service publication proven

The follower and activation transport now retain model-global layer identities
while storing only the owned stage rows. Main-only, terminal NextN and nested
stage families reject foreign or incomplete intervals; topology fingerprints
include the origin. The exact original activation-manifest function fails the
new regression because it emits zero-based layer IDs for a later stage.

Controller runtime bindings authenticate the same interval as residency and
runtime tables. Host bank publication translates compact recipe rows at the
physical boundary. Controller fabric ABI 18 and service publication ABI 2 carry
the origin in existing reserved metadata, without increasing record size,
row counts or communication extents. Shared channel names, profile admission
and telemetry decode distinguish equal-shaped stages. The current controller,
follower, DGO, rank and orchestration consumers compile with the new interfaces.

All **525 focused cases across 18 groups** pass. Ten new device-free cases pass
**20 repetitions** each, covering depths 0–15, retained sidecar geometry,
near-limit origins, foreign stages, malformed headers and unchanged storage
extents. Graph-family, controller-fabric, service-decoder and native-publication
regressions have explicit ProductionTestPreflight registrations; the existing
stage-runtime entry also covers the two controller-binding regressions.

The real CUDA and ROCm service publication kernels now have a captured replay
proof for origins 0 and 32. Both cases pass **20 repetitions (40 executions)**;
each case exercises both vendors and two advancing publications from the same
retained graph. Driver diagnostics and final native owner retirement pass.
The local sm_86/gfx906 compiler spill checks pass. This is scoped native
Integration evidence; full shipped-architecture and combined Release/model
qualification remain pending. No model or prefix payload hashing was added.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r8.json`
(**69 candidate files**, **290 qualified source bindings unchanged**) and
`resource-growth-work/hybrid-stage-service-native-20261008-r1/controller.json`.
The negative-control build initially mixed headers from two worktrees; its
preserved r35 failure was resolved by compiling the exact original pure helper
against the current descriptor. The expected failing helper regression is
retained. A subsequent Doxygen-only header delta is recorded explicitly beside
the native test's unchanged compiled-source receipt.

Remaining: translate compact device movement commands into authenticated global
physical layer identities; finish local rebalance/routing/prefill coordinates
and stage-aware residency planning; preserve authored PP and install per-stage
PMA-owned runtime bundles. Combined qualification, admission/failure cleanup,
dense/MoE hybrid Release app sessions and feature-branch commit/push remain open.
The last turn was a status-only no-progress turn; this turn produced code,
focused regressions and native evidence. No image or model app was started.

## October 8, 10:14 UTC: dense pipeline admission fixed natively; MoE runtime construction binds its stage

The first dense hybrid Release startup failed before readiness: its captured
transport recomputed the global 512-row ceiling after PMA had admitted only
128 resident rows. It requested 10,489,856 host staging bytes against 2,629,632
admitted. The candidate now uses the initialized arenas' common admitted
capacity and rejects unequal stage geometry. No reserve or timeout was enlarged.

The native negative control reproduces the same PMA failure. The corrected
production pipeline passes all **16** captured cross-vendor numerical cases:
both vendor orders, one/two participants per domain, ordinary/dynamic MTP, and
256/128 admitted rows. Requests exercise chunked prefill, exact stochastic token
oracles, continuation, reset, KV payloads and exact physical-memory claims. All
eight added cases are explicitly registered in ProductionTestPreflight. The
first regression fixture omitted the root plan's admitted row count; production
already publishes it. That failed receipt remains beside the corrected proof.
The original negative control exits with SIGSEGV after reporting its expected
admission failure; the repaired cases retire normally and driver checks pass.
Evidence: `resource-growth-work/hybrid-channel-capacity-native-20261008-r2/proof.json`.

Qwen MoE runtime construction now derives its global origin from the authored
PP stage and uses compact storage counts for main, transient prefill and MTP
tables. Durable parents cover the complete owned placement manifest, including
terminal NextN banks; children cannot exceed it. Foreign-stage plans fail before
scratch allocation. Runtime keys and graph rebalance identities include stage
geometry, reused tables must match it, and fixed projection bindings and prepared
transfer-format discovery iterate only owned global layers.

All fifteen focused groups pass **467** tests. Four new device-free construction
cases pass **20 repetitions** each, covering both inert GPU identities, nonzero
and near-limit origins, parent/child NextN geometry, invalid/foreign intervals,
and unique device/stage/role/depth identities. Their production recipe is tested
without allocating mirrored GPU owners. An initially incomplete fixture plan
was rejected by ordinary plan validation; the final fixture supplies the complete
production-shaped plan. The focused construction cases have an explicit preflight
entry. Source-bound evidence is `resource-growth-work/hybrid-stage-ownership-20261008/result-r7.json`
(**52 candidate files**); all **290** qualified c20 source bindings remain unchanged.

This is focused Integration/CPU evidence, not native MoE or a new Release-model
certificate. Stage-aware controller/epoch/service and follower wiring, residency
planning, per-stage orchestration bundles, native dry-run cleanup, actual dense
and MoE hybrid app sessions, complete candidate qualification and commit/push
remain open. No new model session or Docker image build was started in this slice.

## October 8, 09:23 UTC: compact runtime tables and stage-bound prefix archives proven locally

The isolated hybrid candidate now binds runtime tables to immutable global layer
intervals while keeping their arrays compact. Placement access, fixed down banks,
initial-state replay, request reset and histogram capture/restore use checked
stage coordinates. Async histogram admission rejects foreign geometry before
querying or rotating any device bank. Child telemetry views address their owned
interval in the canonical parent's storage; native GPU proof remains pending.

Portable prefix runtime archive v6 carries global layer IDs, and its schema version
is part of immutable cache compatibility metadata. The real Qwen serializer now
checks the entire archive's syntax, table identities and layer geometry before
restoring tables. Duplicate tables, foreign later rows, obsolete versions,
truncated suffixes and trailing bytes leave live tables unchanged. Runtime-level
restore also validates all row identities and shapes before mutating an earlier
row. No model or prefix payload hashing was introduced.

All fifteen focused groups pass **463 tests**, including eleven new cases. The
seven stage-runtime tests, three real graph codec tests and one fixed-projection
binding test each pass **20 repetitions**. CPU execution and inert CUDA/ROCm
metadata cover ordinary, nonzero and near-limit stage origins; this does not
certify native GPU telemetry or hybrid inference. Two explicit preflight entries
register runtime and prefix-archive coverage; fixed projection coverage joins the
existing stage-residency entry. The source-bound receipt is
`resource-growth-work/hybrid-stage-ownership-20261008/result-r6.json`, with
`all-focused-r16-full.log`; it binds **50 candidate files**. All 290 qualified
c20 production source bindings remain unchanged.

Remaining work is graph-owned table construction, device controller/epoch and
service bindings, stage-aware residency controller planning and orchestration
bundles. MoE PP admission remains guarded. Native admission-only cleanup,
dense/MoE hybrid inference, complete candidate qualification and commit/push are
still open. This slice started no model run, image rebuild or remote publication.

## October 8, 08:43 UTC: compact stage residency and physical CPU waves proven locally

Participant banks now bind an immutable global layer interval and retain only
owned rows. Registration, missing-layer diagnostics, service imports, migration
preparation, epoch publication, reusable seals and prepared-context rebinds use
checked global-layer lookups. Same-sized foreign owner maps, snapshots, manifests
and banks are rejected; physical admission rejects a foreign stage before any
shadow memory is claimed. The CPU service measurement consumer and the local
expert stage use the same bank access contract.

All thirteen focused groups pass **337 tests**, including ten new residency and
physical-fabric cases. The eight stage bank/registry/migration cases and the two
physical admission/transfer cases each pass **20 repetitions**. Real CPU copies
cover every **21 quantized formats plus FP16/BF16/FP32**, origins 32 and 40, two
owned layers and one shared shadow slot per participant. Complete move/restore
cycles preserve exact prepared bytes, retire through both admission fences,
retain bounded PMA claims and produce valid reusable-context seals. GPU bank
identities and both movable projection families are exercised as device-free
metadata; this does not certify native GPU publication.

The new CPU fixture initially polled individual operations serially, starving
the whole-wave launch barrier. It now uses the production composite transport.
Its next attempt correctly hit the fatal retirement guard because the fixture
omitted the two-phase grace period; the test now proves active-reader deferral
and both fences before retirement. Both failed logs and the confirming backtrace
remain in the evidence directory. No production timeout or execution mode was
changed to make those tests pass.

The production local-expert stage, factory and orchestrator compile with the
candidate interfaces. Two explicit ProductionTestPreflight registrations cover
stage residency and physical residency. Evidence is bound to **44 candidate
source files** in `resource-growth-work/hybrid-stage-ownership-20261008/result-r5.json`
and `all-focused-r14-full.log`. All 290 qualified c20 production source bindings
remain unchanged. This remains focused local evidence, not complete canonical
preflight or native hybrid certification.

Next work is device runtime tables, controller/epoch state, stage-aware residency
planning and orchestration bundles. MoE PP admission remains guarded. Native
dry-run cleanup verification, actual dense/MoE hybrid inference, final candidate
qualification and commit/push remain open. No model run, image rebuild, commit,
push or remote CI was started in this slice.

## October 8, 07:54 UTC: stage-local model admission and calibration proven locally

The isolated hybrid candidate now admits compact nonzero-origin GGUF manifests,
prepared-weight BOMs and reusable transfer directories. Capacity installation
rejects a different stage even when its row count and tier geometry match.
Factory metadata and placement resolution retain only the requested main-layer
interval; routed NextN banks belong solely to the terminal stage. Serial, retained
MTP-off, fixed depths 1/15 and dynamic policies cover fifteen stage/policy cases.
Calibration classes, prepared evidence, rank merges and certified service/migration
profiles retain global layer IDs while their storage contains only owned rows.

All twelve focused groups pass **294 tests**, including thirteen newly added
stage regressions. Capacity tests sweep the source-format catalog across CPU,
CUDA and ROCm, both movable projection families, and ordinary/nonzero/near-limit
origins. Equal-sized foreign-stage evidence, missing rows and invalid intervals
are rejected. Five additional focused ProductionTestPreflight entries register
capacity, GGUF manifest, MTP admission, calibration and economy coverage. The
production factory and orchestrator compile against the candidate interfaces.

This remains focused model-free evidence, not full canonical preflight or native
hybrid certification. The first local calibration run linked an old evidence-
merger object; rebuilding that dependency clears both failures. A subsequent
new directory test incorrectly supplied changing matrix shapes to a directory
whose contract requires fixed shapes; its corrected fixture preserves that
contract while retaining heterogeneous-shape BOM tests. Both failed logs remain.
The final complete native output and all source bindings are retained in
`resource-growth-work/hybrid-stage-ownership-20261008/result-r4.json` and
`all-focused-r8-full.log`. Qualified c20 production source bindings still match.

Next work is stage-local runtime tables, participant banks, controller/epoch
state and orchestration bundles. MoE PP admission remains guarded until those
production owners exist. Native dry-run cleanup verification, actual dense/MoE
hybrid inference, complete candidate qualification and commit/push remain open.
No model run, image rebuild, commit, push or remote CI was started in this slice.

## October 8, 07:15 UTC: compact stage planning and histogram transport proven locally

The isolated hybrid candidate now preserves model-global layer identities through
compact expert-placement quotas, incumbent/cost-driven planning, histogram RCU
banks and retained MTP phase topology. Frozen demand and transaction packets bind
the stage origin; histogram ABI v4 rejects equal-sized foreign-stage data and old
packets. No preceding-stage rows are allocated or communicated by these paths.
The serving candidate is still the qualified c20 local overlay; its bound source
files remain unchanged. Hybrid MoE runtime admission is not enabled yet.

Seven focused local groups pass 161 tests, including ten new stage cases. A
nonzero-origin concurrent routing/rotation test passes 20 repetitions. The test
build recompiles changed implementations and their residency/calibration callers
together, using unchanged qualified-core dependencies. This is focused evidence,
not full canonical preflight or native hybrid certification. Early mixed-object
and diagnostic-message failures are retained in the result directory; the final
seven-group CTest receipt is green. OrchestrationRunner also compiles against the
candidate headers; the dry-run lifecycle fix still needs rebuilt-native evidence.

The work exposed a separate quota-validator defect: a ragged tier's missing row
was read even after geometry validation recorded an error. AddressSanitizer
reproduces a heap-buffer-overflow in the original validator and observes a clean
rejection in the repaired candidate. The regression covers both whole-model and
stage-offset plans and has its own ProductionTestPreflight registration,
`V2_Integration_MoERaggedQuotaValidation`, alongside the stage planning/histogram
entries.

Evidence: `resource-growth-work/hybrid-stage-ownership-20261008/result-r3.json`,
`all-focused-r4.xml`, `quota-asan-proof.json`, and
`resource-growth-work/hybrid-progress-20261008.json`. Remaining work is the stage
mapping in model catalogs, capacity resolution, device runtime/controller state
and orchestration bundles, followed by complete candidate qualification and real
dense/MoE mixed ROCm/CUDA inference. No commit, push or remote CI action occurred.

## October 8, 06:40 UTC: bounded telemetry qualified; hybrid admission and lifecycle followup

The incremental local qualification completed all 723 Unit and 839 preflight
registrations, retaining unchanged passes for the CMake-only repair. Release
was rebuilt incrementally and copied into the existing develop-image overlay
`c20f470baa4019dddb05a9c5097f7fa230771abf3a35f0e6453fe5f7a570aeb0`.
Both exact Qwen3.6 MoE whole-expert dynamic-MTP HTTP cells pass 45/45 checks.
Their bounded native movement counters join the complete terminal histories:
46 CUDA and 50 ROCm publications. These cells exercise native load-spread
movement, not a positive topology-wide controller journal. Original failed
local-wrapper evidence is retained: the wrapper initially omitted the canonical
private model-tmpfs publication step. The corrected run uses that existing
publication authority and has clean native retirement and driver evidence.

Dense hybrid admission now preserves one MPI rank owning both authored TP
stages. The local launcher previously inherited the host's two-rank bootstrap,
so an extra rank attempted an unrelated full-model admission on CUDA. Its
explicit `--mpi-procs 1` contract and seven local regressions now pass. The exact
ROCm-then-CUDA pipeline admits all 262,144 context tokens with default dynamic
MTP, 16 GiB RAM prefix storage per participant and a shared 32 GiB disk budget.
This is admission evidence only; no hybrid app session has run yet.

The successful dry run exposed a production cleanup defect: admission-only
initialization shared the inference-ready boolean and then attempted to seal
unprepared model weights. The isolated candidate now uses one typed completion
authority for admission versus inference, rejects implicit promotion, and gates
prepared retention and inference shutdown on actual inference readiness. Five
focused lifecycle cases pass and the changed production orchestrator compiles
with the qualified build's options. Native verification of this new fix is
still pending; the earlier admission receipt's error log remains intact.

Hybrid MoE work also now has compact stage-scoped ownership tables that retain
global model-layer IDs without padding earlier stages. All 16 owner-map unit
cases pass, including five new cases and 54 backend/order/layer combinations.
The two new invariants are registered explicitly in ProductionTestPreflight.
The candidate still needs stage-scoped planner/catalog/histogram/controller
integration and runtime bundles; the MoE admission guard remains in place.
These candidate changes are isolated from the qualified source. No commit or
push has occurred.

Evidence: `controller-counter-incremental-coverage-20261008-r2/`,
`bounded-counter-native-release-20261008-r2/`, `dense-hybrid-admission-20261008-r3/`,
and `resource-growth-work/hybrid-stage-ownership-20261008/result-r2.json`.
`resource-growth-work/hybrid-progress-20261008.json` records outstanding work.

## October 8, 05:38 UTC: fixture dependency repaired; remaining native gates running

The current build passed all 723 Unit registrations and 417 of 418 host
preflight registrations. The new C++/Python movement transport check failed
before execution because its direct Python CTest registration did not make the
canonical preflight build depend on its native fixture. It now uses the shared
`add_v2_test` registration. Building only the canonical gates produced the
previously absent fixture; the focused regression passes. Executable metadata
and semantic CTest comparisons prove that every retained passing test is
unchanged. No completed long app session or unaffected host test is repeated
for this registration repair.

The original failed transaction remains intact. Separate incremental local
coverage retains those 1,140 passes, adds the repaired test, and runs the complete
unrun CUDA, ROCm and exclusive lanes. One local observer initially classified
regenerated `compile_commands.json` as a binary; it stopped before any GPU work.
Its failure is preserved, and the corrected observer tracks executable files.
This evidence is explicitly local qualification, not a fabricated canonical
full-transaction or published-image certificate.

The owner then incrementally builds Release and creates a copy-only overlay
from the existing develop image. A live-process-authenticated queue admits two
exact canonical Qwen3.6 MoE whole-expert HTTP cells only after qualification.
Their short-request policies require real economical movement; the independent
retirement audit checks each rank's full movement sidecar and all bounded
counter families. The four completed OpenCode app receipts remain unchanged.
The hybrid MoE stage ownership implementation and commit/push remain pending.

Evidence: `controller-counter-qualification-20261008-r1/`,
`controller-counter-incremental-coverage-20261008-r2/`, and
`bounded-counter-native-release-queue-20261008-r1/`. The reviewed source delta is
bound by `resource-growth-work/ci-followup-candidate-20261008-r8.json`.

## October 8, 04:58 UTC: controller movement evidence bounded and independently verified

The remaining topology-wide controller counter conversion is complete. An
unchanged original publisher produced 15,360 wave/capacity rows for 512 waves
across three participant labels, before its per-edge rows. The replacement
retains all twelve wave/edge families in 36 fixed rows. The typed completed
receipt preserves actual bytes, base/candidate epochs, policy economics,
physical capacity certification and exact edge identity. Both the native C++
parity fixture and the Python HTTP consumer check the complete ordered history.
Followers retain physical receipts without acquiring leader-owned economics.
Policy cycles and physical circuits are counted separately because physical
projection can merge policy cycles sharing participants.

Validation passes: 64 focused native tests, 729 Python tests across 26 modules,
and two cross-language tests consuming the actual C++ counter/JSON export.
Negative cases cover missing interior waves with equal totals/extrema, adjacent
uint64 IDs, wrong actual bytes, invalid capacity/economy, missing or duplicated
rank evidence, and later valid terminal waves beyond the HTTP cutoff. The
changed controller, server and native publisher translation units and the real
node-overlay parity fixture compile with the qualified build's options. New
focused preflight entries register this coverage explicitly. These are
metadata-only diagnostics; a complete rebuilt binary still needs qualification.
Evidence: `resource-growth-work/controller-movement-bounded-20261008/result-r5.json`
and `resource-growth-work/ci-integration-focused-tests-20261008-r8/result.json`.

The four completed app runs remain green on `fc98`. Their exact qualified source
is preserved at `resource-growth-work/fc98-qualified-source-20261008` before
promoting the reviewed followups into the feature worktree. No original app,
resource, driver or archive evidence is removed. The next action is the local
incremental build and complete Unit/preflight refresh, then a focused native
movement/terminal-export proof. The hybrid launch remains held pending real
stage-scoped MoE ownership. No commit, push or remote CI action has occurred.

## October 8, 04:09 UTC: all four app cells passed; native movement counters bounded

All four requested homogeneous OpenCode app lifetimes passed the independent
completion audit on the locally qualified sparse-checkpoint Release overlay
`fc98a6799ad78a62159fae003c3d894025e073060be6ea7b00163a1e819f1010`.
Each completed all ten phases with default dynamic MTP, PMA-admitted maximum
context, 32,768 output tokens, 16 GiB RAM prefix cache per participant and a
shared 32 GiB disk tier. The exact client requests were preserved. All native
parsing, graph/MTP, resource, driver, archive and native-shutdown checks pass.
This is local qualification, not a published-image certificate.

| Model / TP backend | Responses | Tool errors / attempts | Token reuse | Mean / median TTFT | Workload |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 dense / CUDA | 58 | 0 / 112 | 92.30% | 12.55 / 3.61 s | 56.6 min |
| Qwen 3.8 dense / ROCm | 95 | 0 / 129 | 96.29% | 39.28 / 20.14 s | 146.7 min |
| Qwen 3.6 MoE / CUDA | 220 | 3 / 226 | 97.08% | 5.44 / 3.44 s | 49.4 min |
| Qwen 3.6 MoE / ROCm | 236 | 4 / 235 | 98.47% | 21.30 / 14.60 s | 193.3 min |

The final MoE ROCm app independently passes all 78 web-app acceptance checks;
its authored suite also reports 114 passing tests. Its four model edit mistakes
are 1.70%, under the strict 5% gate. The original report's strict task verdict
remains false because it retains those errors; the authorized engine verdict is
true. Dense ROCm and MoE CUDA app-quality failures remain visible. Across all
cells, 609 HTTP responses and 702 tool attempts completed, with no engine/parser
failures. The independent audit replays every resource journal and rechecks the
exact exited Docker IDs. Evidence:
`resource-growth-work/four-cell-completion-audit-20261008-r3/result.json`.

The separate candidate now also fixes the native-load-spread transaction-key
cardinality defect: 512 waves over three participant labels retain **12 rows**,
versus **6,144** in the preserved original-publisher negative control. The
canonical movement ledger retains actual physical receipts independently of
leader economics, including transport followers. Clean server retirement emits
a rank-owned terminal metadata sidecar. Bounded wave/edge witnesses are joined
to this full history before an earlier HTTP prefix is accepted. There is no
cache/model payload I/O or hashing. Topology-wide controller positive
movement/economy/capacity/edge families still need their corresponding conversion;
the native change does not weaken those existing per-wave consumers.

All **60 focused native tests** pass (six receipt/export, forty collector and
fourteen bounded-counter cases). Five changed production translation units
compile with the qualified build's options. A metadata-only C++ probe using the
production journal/archive/publisher is independently checked by Python: four
bounded rows cover three final waves while the HTTP cutoff contains two. Wrong
actual bytes, a missing middle wave, a missing edge and a changed backend all
fail. The candidate also passes **726 Python tests across 26 modules** in 15.05
seconds. Explicit focused preflight registrations are added, but not yet
configured/promoted into the serving build. Evidence:
`resource-growth-work/completed-movement-publication-20261008/result-r4.json`
and `resource-growth-work/ci-integration-focused-tests-20261008-r6/result.json`.

The completed serving image/source and original receipts remain unchanged.
Remaining work is the controller-specific positive telemetry conversion,
reviewed candidate promotion and local rebuild/qualification, followed by the
requested hybrid PP(TP(2xROCm), TP(2xCUDA)) cells after real stage-scoped MoE
ownership exists. The old hybrid launch stays held. No feature commit/push,
remote workflow, ruleset or release action has occurred.

## October 8: three app cells passed; canonical CI candidate prepared

The sparse-checkpoint Qwen 3.6 MoE 2xCUDA app completed all ten phases in
49.4 minutes: 220 responses, 226 tool calls and three model edit errors (1.33%).
Its engine/protocol, native-response, graph/MTP, resource, archive and continuous
driver audits pass, with native exit zero and no surviving inference process.
The generated app failed its independent quality checks; that remains visible
and does not fail the authorized engine gate. Dense ROCm has also completed at
262,144 context in 146.7 minutes: 95 responses, 129 tool calls, zero tool errors
or native alterations, and clean runtime/resource/archive/driver retirement.
Its authored-test discovery check failed because the model placed the tests
under a package; that app-quality failure remains separate. Three of four new
local Release-overlay cells are complete. MoE ROCm began after dense retirement.
The serving source remains frozen.

MoE CUDA reuses 97.08% of all prompt tokens (97.78% for ordinary turns), with
mean TTFT 5.441 s and median 3.442 s. Its first post-compaction request restores
4,096 of 17,772 tokens (23.05%) with 15.807 s TTFT, after both tiers reached
capacity and disk evictions totaled 47.47 GB. The final metadata audit reports
6,092 live payload files, no orphans and zero payload bytes read. Resource
observations cover 553 samples; anonymous plus swapped memory peaks at
2,043,412,480 bytes and each CUDA GPU at 22,273,851,392 bytes, within admission.
The continuous driver window has 2,788 snapshots and no faults. This is live
reuse evidence under churn; the matched replay owns the causal speed comparison.

Dense ROCm's request 78 legitimately performed a 1,176.77-second prefill.
At UTC midnight the real OpenCode client changed only the date in its first
system message, invalidating the later causal state. The sparse 4,096-token
checkpoint survived; 99,507 prompt tokens needed prefill. The turn completed
naturally and later requests resumed cache reuse. Its exact prompt-mutation
receipt is `qwen38-rocm-sparse/live-midnight-prefix-change.json`. No generation
timeout, prompt rewrite or special client rule was introduced.

The separate CI candidate now wires master-PR-only coding after both benchmark
lanes. It derives 27 blessed canonical cells across two ISAs (54 jobs), preserves
production placement and PMA context, independently validates every app/native,
runtime/resource/driver/archive receipt, and rejects missing, stale or cancelled
jobs. Release promotion requires the same complete coding proof. No workflow,
ruleset, release or published-image matrix has been executed or changed remotely.
The candidate passes 678 focused device-free regressions in 12.25 seconds.
CMake contains explicit Unit/preflight entries for the new lifecycle/evidence
checks; these candidate registrations have not yet been promoted/configured in
the frozen serving build.

Real model-free Docker probes additionally prove init signal forwarding and
outer parent-before-child retirement. The first outer probe exposed case-varying
Docker missing-object diagnostics; the final implementation uses exact daemon
inventory instead of error-text parsing. The repaired native probe retires both
containers in 1.53 seconds while preserving the interrupted job's failed verdict.
A failed-create reply retains ownership, and worker cancellation is propagated
before executor join. The original failures and their negative controls remain
available. Every production device lease also checks native coding-owner
inventory, so a cancelled runner cannot admit a successor beside a survivor.

A second real Docker defect dropped fast helper output from attached streams
despite exit zero. The helper now reads retained stdout/stderr after native
retirement. Twenty real Docker repetitions preserve stdin, emoji text and both
output streams. The exact serving image decoder re-audits the completed dense
CUDA session: all 58 responses and 112 calls match. A small control-image build
installs pinned OpenCode 1.18.34 over an existing local runtime; the non-root
offline client probe passes with its complete XDG environment. No inference
image was rebuilt. No cache/model payload hashing is used.

Evidence is under `resource-growth-work/ci-integration-focused-tests-20261008-r5/`,
`coding-outer-lifetime-proof-20261008-r2/`, and
`prefix-checkpoint-app-20261007-r3/qwen36-moe-cuda-sparse/`.
`resource-growth-work/ci-followup-candidate-20261008-r4.json` binds the reviewed
candidate source deltas. The final MoE ROCm app, promotion/registration of the CI
candidate, and hybrid MoE stage ownership remain outstanding. Nothing has been
committed or pushed.

An additional device-free production-coordinator regression exposes unbounded
cross-rank transaction telemetry: 256 commands grow 13 records to 2,818 in the
frozen implementation. The isolated candidate folds command/sequence identities
into fixed-width ordered evidence and preserves exact retired work. All 34
native coordinator tests pass; the HTTP graph consumer validates bounded
families against an owner-wide contiguous retirement span, including large
uint64 IDs after reset. Both producer and consumer have explicit preflight
entries. The serving source and image remain unmodified; this focused native
diagnostic links the changed publisher against the qualified shared core and
does not constitute a rebuilt-image certificate. Evidence:
`resource-growth-work/bounded-overlay-telemetry-20261008/`.

The physical-fabric follow-up now bounds six lifecycle counter families; the
no-movement controller publisher also retains numeric decision evidence without
per-transaction keys. Negative controls extracted from the original publishing
bodies retain 9,216 versus 18 physical rows and 12,288 versus six idle-controller
rows for identical workloads. Ten native counter tests and seven production CPU
fabric tests pass, including the complete weight-format sweep. The changed
fabric and controller translation units compile against the qualified shared
core. New focused preflight entries cover both fixes. This is device-free local
evidence, not a native GPU movement run or a rebuilt-image certificate. The
positive movement/economy/edge families and their strict per-wave consumers
still need a coherent bounded conversion; other controller diagnostic keys are
also under review. Evidence:
`resource-growth-work/bounded-physical-telemetry-20261008/result-r2.json`.

The independent four-cell completion audit revalidates all three retired cells,
including native Docker identity/exit, exact app counts, full resource journals,
MTP/graph policy, archive ownership and HTTP/driver observations. It binds the
three test-only image-receipt differences to the subsequent green incremental
qualification. Original cell receipts remain untouched. MoE ROCm remains pending;
at 02:09 UTC it has completed 130 responses and six phases, with zero failed HTTP
requests and 97.07% prefix-token reuse. Its live client events report three model
edit mistakes in 130 tool calls, below the authorized 5% threshold; final wire
and native validation still awaits retirement. Evidence:
`resource-growth-work/four-cell-completion-audit-20261008-r2/result.json`.

The next isolated follow-up also removes varying cadence sizes, service-snapshot
counts/generations and physical polling counts from controller map keys. Completed
prepared-context restoration remains distinct from useful optimization work.
Twelve counter tests and seven real CPU fabric tests pass. Both changed host
translation units compile, and the corresponding maintenance regression is
explicitly registered for preflight. The positive movement/economy/capacity/edge
publication families still require their coherent bounded consumer conversion;
that proof has not been weakened. Evidence:
`resource-growth-work/bounded-physical-telemetry-20261008/result-r3.json`.

The final ROCm MoE run completed phase seven at 02:45 UTC (177 responses,
175 tool calls, three model edit errors). Its next user phase appended messages
without changing earlier client history, yet the Qwen template legitimately
removed the previous turn's reasoning. Request 178 restored 93,568 of 151,227
tokens and finished a 545.148-second prefill with 548.375-second TTFT. A separate
metadata-only native-tokenizer probe reproduces all three observed prompt
lengths and proves 93,591 common leading tokens: the cache restored the longest
complete 64-token block, leaving only 23 reusable tokens. It therefore did not
miss a reusable long prefix. No debugger or GPU instrumentation was used.
Evidence: `qwen36-moe-rocm-sparse/live-phase-reasoning-prefix-change.json` and
`phase-seven-prefix-diagnostic/` in the current app root.

The movement-audit dependency review also found native-load-spread mirrors with
transaction keys, and a canonical generation join requiring exact earlier edge
identities even when shutdown records later legitimate movement. Three new
adversarial regressions preserve that contract: equal totals/ranges cannot hide
a missing middle wave, later terminal waves cannot replace earlier HTTP history,
and adjacent uint64 identities beyond double precision cannot alias. All 21
movement-ledger tests and the explicitly registered four-case
`V2_Integration_MovementTransportHistoryEvidence` command pass. Production
consumers remain unchanged pending a coherent bounded conversion; actual copied
bytes must not be inferred from estimated edge sizes. Evidence:
`resource-growth-work/movement-transport-history-contract-20261008/result.json`.

A metadata-only negative control now also reproduces the native-load-spread
publisher's growth: 512 waves and three participant labels create 6,144 retained
rows from unmodified publishing bodies. The cardinality assertion fails as
intended; no model or GPU path ran. This additional publisher must participate
in the same bounded-history repair. The reproduction is preserved at
`resource-growth-work/positive-movement-metadata-growth-20261008/result.json`.
At 03:17 UTC the final ROCm MoE app has completed eight phases and 201 requests,
with no failed/disconnected HTTP requests and four model tool errors in 198
calls (2.02%). Its 92 authored tests passed at the end of the persistence phase;
final independent engine/native/resource audits remain pending.

## October 7: sparse checkpoint reuse and app-only measurements

All four earlier app workloads have retired. Their strict native audit covers
675 responses and 693 tool calls without argument or reasoning changes. Both
dense apps passed acceptance; MoE app-quality failures remain separate from
engine/protocol results. The last ROCm MoE driver's bookend log cursor was lost,
so that old cell is not a complete certificate. The collector now journals the
whole interval continuously and its focused lifecycle/rotation tests pass.

A captured compaction transition retained 6,212 leading coding-prompt tokens
but restored none; the summary request itself shares only three. The general
recurrent checkpoint schedule now preserves aligned 4K, 8K, 16K and subsequent
doubling frontiers, plus the existing pre-tail checkpoint. It adds at most six
images at 256K context, uses existing PMA tier budgets and never reexecutes
restored history. No OpenCode-specific engine logic or payload hashing is added.
Metadata-only inspection measured approximately 95 MiB per MoE checkpoint and
225 MiB per dense checkpoint per TP participant, making sparse writes necessary.

The current incremental Integration build passes 14 focused CTest registrations,
including restored-history RAM/disk churn, MTP off/fixed/dynamic scheduling,
request measurement accuracy and TP/PP propagation. Native CPU/CUDA/ROCm prefix
state and MPI coordination pass four further registrations with a complete,
clean continuous driver interval. These are focused proofs, not renewed full
candidate certification. Both paired incremental Release overlays are built and
authenticated. A standalone decoder regression also passes after exporting its
public JSON dependency; complete Unit/preflight targets are built.
The new process-stats test needed its mock include directory registered.
The first complete Unit run found three stale dynamic-tag assertions in two
MTP suites. They now assert exact bounded sequence coordinates; both suites
and focused preflight registrations pass. These test-only repairs preserve
both serving image identities. The subsequent canonical transaction passed
713 Unit, 394 host, 127 CUDA and 92 completed ROCm registrations before two
more obsolete tag assertions failed in the shared native prefill test header.
All six affected CUDA/ROCm registrations now pass with exact bounded-coordinate
assertions. The original canonical receipt stays red. Separate incremental
coverage now passes all 713 Unit and 815 preflight registrations, retaining
unchanged completed cases and freshly running the six affected registrations,
70 remaining ROCm and 129 exclusive registrations. Its continuous driver window
is clean. No serving binary changed for these test-only repairs.

OpenCode's optional `--measure-prefix-reuse` mode binds every sequential exchange
to exact `/stats` request counters and wire usage. It reports ordinary,
compaction and post-compaction TTFT/reuse separately, bounded tier occupancy and
committed write/churn traffic including tool-execution intervals. Its real
HTTP/SSE regression preserves request/response bytes. The matched Qwen 3.6 MoE
2xCUDA replay now passes both policies with identical 4,313 committed token IDs.
After compaction, sparse checkpoints restore 4,096 of 20,391 prompt tokens
(20.09%) versus zero for the near-tail control. TTFT falls from 23.208 to 18.962
seconds (4.246 seconds, 18.3%) in this single trace-enabled comparison. Cold
prefill is effectively unchanged. Request-boundary RAM-cache peaks rise from
14,771,847,168 to 17,366,568,960 bytes across both participants (2.42 GiB more),
within the aggregate 32 GiB RAM capacity. This short replay does not exercise disk
churn. Both lifetimes pass native-output, captured runtime/MTP, resource, archive
metadata and continuous-driver audits with clean administrative shutdown.

The current Qwen 3.8 2xCUDA app has completed all ten phases and all 78
independent app checks in 56.6 minutes. Its 58 responses and 112 tool calls have
zero tool errors or native argument/reasoning changes. Runtime graph/MTP,
resource-growth, archive ownership and continuous-driver audits all pass, with
clean administrative shutdown. This is one exact local Release-overlay cell,
not a published image certificate or completion of the other model/topology cells.

The app's real post-compaction request restores 4,096 of 22,164 tokens (18.48%),
with 30.521 s TTFT, after the 32 GiB disk tier exercised eviction at capacity.
The separate 96,490-token summary prompt restores zero and takes 240.906 s to
its first token. Ordinary requests reuse 95.83% of prompt tokens, with mean TTFT
8.149 s and median 3.533 s. All 58 requests together reuse 92.30%, with mean
TTFT 12.548 s. The live transition proves retained reuse under churn; the earlier
matched MoE replay owns the causal timing comparison. The final metadata audit
finds 2,088 live payload files, no orphans and zero payload bytes read.

Dense ROCm remains active at the model maximum of 262,144 tokens. MoE CUDA has
started after dense CUDA's complete retirement, at its PMA-proven 195,328-token
context; MoE ROCm is queued behind dense ROCm. Every app retains default dynamic
MTP, 16 GiB RAM cache per participant and 32 GiB shared disk cache, and observes
all requests without a generation deadline. Concurrent vendor-pool observations
are stress evidence, not isolated throughput benchmarks.

The observer's unrelated-process probes, unconditional NVIDIA telemetry on
CPU/ROCm cells, failed collector retirement and suite-driver Docker COPY context
have staged fixes with negative controls. A scoped one-shot observer authenticates
both live native PIDs without errors. Failed-collector retirement passes 20
repetitions (100 tests). A separate `resource-growth-work/ci-integration-source`
checkout now holds these changes and the shared benchmark/app receipt validators;
all 158 focused device-free harness/CI regressions pass there. The app validator
also independently admits the completed CUDA app's full native coverage.
Seven new Unit/preflight registration pairs are prepared; the active serving
source stays frozen while the remaining cells run. Earlier local-driver mistakes
remain failed artifacts. The three remaining fresh app cells, master-PR
post-benchmark workflow wiring and hybrid MoE ownership remain outstanding.
Nothing has been committed or pushed.

Evidence: `parity-results/opencode-tool-calling/resource-growth-work/build/`
(`sparse-checkpoint-all-focused.xml`, `sparse-native-checkpoint-20261007/`) and
`app-only-scope-20261007/compaction-prefix-transition.json`.

## October 7 13:44 UTC Release registration repair and new image

The full HTTP-fix qualification passed all 703 Unit and 791 production-preflight
registrations (371 host, 128 CUDA, 164 ROCm and 128 shared-device), without
failures, skips or new driver records. Release then exposed two unconditional
HTTP-affinity registrations referencing its intentionally omitted test target.
The production registration module now follows target selection. A configure-only
regression proves both native policies remain intact in Integration and are absent
in Release; the old declaration fails the new Release control.

Per the user's explicit instruction, the complete gate was retained rather than
repeated for this test-only correction. Five focused CTest registrations pass,
including both native affinity modes. Only three test/CMake files differ from
the full-gate snapshot. The incremental Release build and local overlay
`sha256:ff9bbf2ece7404474e6f392196ed45f89129e6cdcce830695f3382343dc10292`
pass exact binary, native-loader and rebuilt prompt/parser probe checks.
The Release core is `aa824b75b0924b5fd43f68143064cf24a8642ec226e6765a08115df91725f1f3`.

`mtp-seed-http-registration-fix/` retains the old failures and names the inherited
1,494-test gate separately from its focused recheck. Campaign PID 2948268 has
started new-image context admission, followed by all four ten-phase app plus
240-session OpenCode cells. Source remains frozen at 193 bound files. Hybrid MoE
ownership remains unresolved; no commit or push has occurred.

## October 7 12:45 UTC complete original cells and fresh HTTP qualification

All four original server lifetimes and their parent controllers have retired.
MoE CUDA completed ten app phases, all seven required tools, all 78 app
acceptance checks, and 240 primitive sessions. Its original primitive result
remains 239 protocol passes / one failure, with 149 task passes. The final
native audit verified 1,204 responses and 1,043 tool calls without argument or
reasoning changes; stats, runtime policies, archive storage, shutdown and GPU
driver checks passed.

The single failure was the harness treating OpenCode's completed Bash error
for a model-authored missing working directory as a protocol defect. The actual
client exited normally. Three real OpenCode 1.18.34 loopback reproductions cover
an absent internal tool-output directory, an absolute workdir and a relative
Unicode workdir. Commands never executed. The narrow classifier requires an
exact authenticated workdir and the exact FileSystem.access NotFound diagnostic;
changed input, identity, schema, continuation and unknown errors still fail.
All 36 harness units and 20 repetitions of its focused preflight pass; the old
implementation fails all six positive path/transport controls. A separate
reassessment of retained evidence passes protocol while keeping failed tasks
and unsuccessful Bash coverage. Original reports are unchanged.

The first HTTP repair gate applied its source, then failed six 30-second Unit
timeouts. The queued campaign was retired before any serving work. A complete
unchanged Unit diagnostic subsequently passed, while process observations
exposed under-reserved CPU work: the kernel sweep used 56 workers against a
four-core reservation, and Python references created 182 threads against one
core. CMake now shares one width with the compiled sweep and reserves its full
physical demand; native numerical libraries use one worker for tiny reference
fixtures. Four inventory/runtime regressions and their negative controls pass.
No generation or test deadline was increased, and global concurrency remains
unrestricted.

Fresh owner PID 2702735 at `mtp-seed-http-requalified/run.py` authenticates 191
source files and runs the complete 703 Unit + 791 preflight inventory before an
incremental Release build, local image overlay and native probe qualification.
Campaign PID 2821112 waits on that exact owner and then runs all four fresh
cells with the explicit glob/grep/read final review. Six campaign-admission
negative controls pass. Hybrid MoE admission remains unresolved. No commit,
push, full dependency image build or new GPU profiler attachment occurred.

MoE CUDA's terminal stats show 89.40% prompt-token reuse, 79.82% request hit
rate, 83.38% MTP acceptance and 1,754 depth updates. Mean TTFT is 10.87 s,
including 7.89 s queueing; weighted decode is 82.65 tokens/s and uncached prefill
762.07 tokens/s. RAM is 92.71% full, disk 99.86%; no disk-to-RAM restores occurred
in this MoE cell. All 7,987 HTTP responses at the final reset-epoch frontier are
200. These are traced stress observations, not benchmark certification.

Evidence: `opencode-bash-workdir-reproduction/`,
`mtp-seed-http-followup/unit-contention/`, and `mtp-seed-http-requalified/`.

## October 7 11:42 UTC complete ROCm MoE cell

Qwen 3.6 MoE on two ROCm devices completed the full current-image cell and
retired normally with Docker exit code zero. All ten app phases and all seven
required tools were exercised; all 240 primitive sessions passed protocol and
aggregate successful-tool coverage. The model's wrong-workspace app failure
and 160 primitive task-quality failures remain recorded separately. They have
not been relabelled as successful tasks.

The complete native audit matched 1,268 responses and 1,133 tool calls with no
argument changes or reasoning mismatches. Full workload statistics, captured
execution/MTP/transfer policies, prefix ownership and capacity, metadata-only
archive storage/compaction checks and the driver window all passed. The actual
cell process has retired. This complete audit supersedes its partial frontiers.

At the terminal stats frontier, prompt-token reuse was 91.23%, request hit rate
80.99%, MTP acceptance 83.05%, and mean TTFT 16.19 seconds including 11.07 seconds
mean queue time. Weighted decode was 25.26 tokens/s and uncached prefill 544.53
tokens/s. The workload generated 270,420 tokens, with 1,951 dynamic-depth
updates. RAM occupancy was 93.44% and disk 99.79%; disk-to-RAM restore remained
zero for this MoE workload. The 16,754 completed HTTP responses at this reset
epoch's final stats frontier were all 200. These are traced stress observations,
not benchmark certification.

MoE CUDA remains live in its primitive sweep (93/240 at observation). The HTTP
repair/build controller PID 2340509 and next full campaign PID 2381938 still
wait for its actual retirement and the original parent owners. Dense failures
remain unchanged; the new source/image/full HTTP qualification and hybrid MoE
support are still outstanding. Nothing has been committed or pushed.

## October 7 11:33 UTC MoE app completion and queued requalification

| Active cell | Completed app | Primitive sessions | Protocol failures so far |
| --- | --- | ---: | ---: |
| Qwen 3.6 MoE / ROCm2 | 10 phases, all seven tools; wrong-workspace task failure retained | 223 / 240 | 0 |
| Qwen 3.6 MoE / CUDA2 | 10 phases, all seven tools, 78 independent app checks passed; ambiguous-edit/result marker task failures retained | 49 / 240 | 0 |

Partial native audits match 1,140 ROCm responses / 1,020 calls and 336 CUDA
responses / 304 calls, with zero argument or reasoning changes. These counts
supersede earlier partial audits. Both active driver windows remain clean.

| Live cumulative statistic | MoE ROCm2 | MoE CUDA2 |
| --- | ---: | ---: |
| Completed requests at stats frontier | 1,168 | 374 |
| Prompt-token reuse | 91.64% | 94.79% |
| MTP acceptance | 83.06% | 86.22% |
| Mean TTFT, including queue | 16.49 s | 10.51 s |
| Mean queue time | 11.11 s | 5.91 s |
| RAM tier used | 93.20% | 91.91% |
| Disk tier used | 99.75% | 99.81% |

The next full four-cell HTTP campaign is queued as PID 2381938 at
`mtp-seed-http-followup/campaign/run.py`. It requires build handoff PID 2340509
to finish and retire, then reauthenticates source/image/probes and reruns native
CUDA context admission. Six admission negative-control tests and its live
read-only handoff check passed. Both controllers are currently waiting;
source is still unchanged. GPU implementation evidence is explicitly inherited
from unchanged code/native libraries, not relabelled as fresh profiling.
New-image Unit/preflight, Release build and complete HTTP stress remain pending.
Neither controller commits/pushes or resolves the hybrid MoE support hold.

## October 7 11:07 UTC live proof and next gate

The HTTP follow-up controller is live as PID 2340509, waiting on actual MoE
process/container retirement. It has not applied source changes or started GPU
work. Its source-bound transition will preserve the preceding code/cores, run
the complete Unit/preflight gate, build Release incrementally, and authenticate
a small local overlay plus native prompt/parser probes. Failure ends that
handoff; HTTP stress must still qualify the resulting image. No commit/push or
hybrid-hold resolution is automated by it.

| Partial native audit | Responses | Tool calls | Argument / reasoning changes |
| --- | ---: | ---: | ---: |
| Qwen 3.6 MoE / ROCm2 | 826 | 746 | 0 / 0 |
| Qwen 3.6 MoE / CUDA2 | 175 | 167 | 0 / 0 |

These cumulative frontiers supersede earlier partial counts. ROCm had completed
143/240 primitive sessions without protocol failures. CUDA was in its tenth
app phase, adding edge-case tests at approximately 161,000 prompt tokens. Both
servers and their parent controllers are still live, without GPU driver
findings. The handoff's host-PID CUDA observation was independently matched to
the live serving process on both cards, so an empty devcontainer NVML view
cannot admit the next GPU gate.

## October 7 10:41 UTC current qualification frontier

| Cell | App evidence | Primitive sweep | Qualification status |
| --- | --- | --- | --- |
| Qwen 3.8 / ROCm2 | 10 phases, 78 acceptance checks; final native audit 103 responses / 116 calls unchanged | Not started after stats-observer failure | Retired normally; failed timeout and required `grep` coverage |
| Qwen 3.8 / CUDA2 | 10 phases, 78 acceptance checks; required `grep` omitted | 240/240 complete, zero protocol failures | Retired normally; app coverage failure retained |
| Qwen 3.6 MoE / ROCm2 | 10 phases, all seven required tools; generated app in wrong workspace | 69/240 complete, zero protocol failures | Protocol/coverage pass; task failure retained |
| Qwen 3.6 MoE / CUDA2 | Five phases complete | Pending | Ordinary/dynamic 512-token and stats controls passed |

Dense CUDA's final native audit matched 979 responses and 860 tool calls with
no changes. Its complete workload stats validation passed across 979 requests:
84.82% prompt-token reuse, 81.38% MTP acceptance, 22.41 s mean TTFT including
17.55 s mean queue time, and 49.29 decode tokens/s. These are traced stress-run
measurements, not benchmark certification. Both active MoE driver windows have
no GPU findings.

The completed MoE app's partial native audit matched 215 responses and 218
calls, including its incorrectly authored workspace path. No argument or
reasoning changes were found. This remains partial cell evidence while its
primitive sweep runs; the model's `No module named taskboard` acceptance failure
is not waived or relabeled as a successful application.

Dense ROCm's original observer error is confirmed as a socket timeout. A
supplementary observer retained another timeout during decode, then continued
to 3,523 successful observations while preserving its failed result. Native
handlers were fast after the delayed arrivals. The underlying delay is not yet
attributed. All 720 subsequent paired workspace/host-loopback reads passed in
at most 7.2 ms, after dense ROCm's retirement. Full records and the Docker-output
capture limitation are retained in `mtp-seed-http-timeout-transport/`.

The isolated HTTP affinity fix passes 24 native focused executions, the
30-test combined server suite and twenty production-pool loopback repetitions.
Its production-call-site negative control fails exactly the worker-mask check.
The observer fix passes ten tests and twenty real socket-stall repetitions.
Both patches remain outside the bound campaign; a new full gate and serving
image qualification are pending. All 185 source and 18 helper bindings remain
unchanged. The hybrid MoE admission hold remains unresolved. Nothing is
committed or pushed.

`mtp-seed-http-followup/transition.json` binds the exact eleven-file update
and its 189 expected source files, without applying it to the live campaign.
Future Docker calls select the direct daemon socket, whose canonical delayed
output/nonzero-exit probe passed; the default socket's attach truncation is a
separate diagnostic transport issue. Cleanup reclaimed 63.8 GiB from the two
retired dense caches on the model-cache filesystem, retaining four metadata
journals and all results without reading payloads. Workspace availability is
approximately 258 GiB.

## October 7 09:02 UTC HTTP observation and CPU-affinity follow-up

The unchanged `ab29f1b5fb52` candidate remains under the live TP2 campaign.
Qwen 3.8 CUDA completed all ten coding phases and all 78 independent app
acceptance checks. Its recorded tool transport passes, but it omitted `grep`,
so the app coverage gate remains failed. Its 240-session primitive sweep is
continuing. Both ROCm apps have completed eight phases; MoE CUDA still follows
normal dense-CUDA retirement. The latest partial native audit joined 480
responses and 476 tool calls with zero argument or reasoning changes. These
observations do not replace the unfinished full-cell and retirement gates.

An independent read-only stats audit matched native/SSE counters for 249
completed requests. CUDA dense and ROCm MoE also matched all 192 retained
per-request timing rows at those frontiers. Dense ROCm's original stats polling
thread stopped after 1,801 observations at 07:57:34 UTC while inference kept
progressing. Its actual exception remains pending until the original workload
context exits. The nearby access log reports a 0.3 ms handler after a longer
pre-handler gap; the original cause is not yet proven. A separate, read-only
observer now retains later observations without changing the original result,
resetting statistics, or interrupting generation.

The old observer silently stopped on its first polling exception. The isolated
replacement immediately journals failed polls, retains its failed qualification
status, and continues scheduled reads of the same endpoint after transport
errors. Invalid statistics still terminate observation. Ten device-free tests
pass, including a real socket stall repeated twenty times; the old behavior
fails that regression. Both Unit and focused ProductionTestPreflight entries
are prepared. Evidence and the pending local helper replacement are in
`mtp-seed-stats-observer-repair/`.

Native scheduler masks also exposed an independent HTTP placement defect:
ordinary listener/HTTP children inherited the OpenMP initial thread's first
physical core despite a 28-place rank partition. Both ROCm servers shared that
core. A scoped repair uses the admitted OpenMP partition for the HTTP listener
and newly created HTTP children, preserves the existing inference worker, and
restores the caller before shutdown. Eighteen native checks pass across bound
and unbound modes, direct execution, and MPI placement on either CPU socket.
The previous inheritance behavior fails the focused mask regression; the
updated server translation unit compiles, and the combined 29-test HTTP server
suite passes with that object. This has not yet been qualified in a new serving image and does not prove the cause of the original delayed poll.

These follow-up changes remain isolated from the running candidate. The six-file
`mtp-seed-http-affinity-repair/http-observer-fix.patch` applies cleanly to the
active source and excludes the still-red MoE PP admission regression. All 185
active source bindings and 18 helper bindings remain unchanged. Apply and
qualify the HTTP follow-up only after the bound campaign owners retire. Keep the
original dense ROCm observer failure and dense CUDA tool-coverage failure in the
campaign evidence. Current runs now report actual SSD-to-RAM restores for both
dense cells; no prefix payload hash or checksum pass was added. The live driver
windows have no GPU findings; three unrelated AppArmor records remain visible.
The next app-review prompt specifies concrete `glob`, `grep`, and `read`
operations. All 34 existing harness regressions pass with that isolated prompt
change; the current missing-`grep` result is retained.
No commit or push has been made.

## October 7 08:05 UTC live campaign and hybrid admission findings

The repaired `ab29f1b5fb52` Release overlay remains under the three active app
runs: Qwen 3.8 on CUDA2 and ROCm2, and Qwen 3.6 MoE on ROCm2. MoE CUDA follows
normal dense-CUDA retirement. All three passed ordinary/dynamic 512-token and
HTTP stats controls. The latest partial native-token audits matched 171
responses and 185 tool calls with no changed arguments or reasoning. These
partial audits do not certify the unfinished apps or primitive sweeps. The
serving driver intervals remain clean; long tool arguments continue advancing
in native token logs even while the client waits for a complete tool call.

A device-free check of the queued hybrid command found two launcher defects:
`--pp-stage` accepts inclusive ends, while the helper emitted half-open ends;
and GPU declarations accidentally pinned NUMA node zero. An isolated helper
corrects both, with nine passing regressions including real GGUF metadata and
the actual parser/rank compiler at NUMA IDs 0, 1 and 7. The original homogeneous
campaign's 185 source bindings and 18 helper bindings remain byte-for-byte
unchanged. The repaired helper is not installed into those active bindings.

After correcting the command, MoE PP still fails before inference because the
current normalization cannot represent its pipeline expert ownership. The
isolated `hybrid-admission-source` worktree now contains a focused
`V2_Integration_AuthoredPipelineMTPAdmission` registration: 160 device-free
cases cover both vendor orders, homogeneous pipelines, widths one/two, sparse
NUMA IDs, MTP off, retained-capacity off, dynamic, fixed one and fixed fifteen.
All 80 dense cases pass; all 80 MoE cases reproduce that exact ownership error
in 0.076 seconds, with no skips. This is an expected-red regression, not a
completed implementation or a new qualified binary.

The idle hybrid handoff was retired through its authenticated PID handle after
proving it had no children or GPU work. The original image waiter remains held.
Homogeneous stress and completed-fixture disk cleanup continue. Hybrid execution
now also requires the complete stage-scoped expert authority and physical-memory
admission implementation; bypassing its current rejection would not satisfy
that requirement. Evidence is in `mtp-seed-hybrid-admission-audit/`, with the
explicit pending hold in `mtp-seed-hybrid-admission-hold.json`. No commit or push
has been made.

## October 7 completed baseline and capture-repair qualification

All four original TP2 servers retired normally with clean driver intervals.
The 960 primitive sessions passed protocol validation. Full native-token audits
joined 4,363 responses and 3,822 tool calls with zero argument or reasoning
changes. Dense CUDA and ROCm completed all ten app phases and all 78 independent
app checks. Both MoE apps passed 77 independent checks but retained authored-test
discovery failures; MoE CUDA also lacked a successful `glob` call. The original
four-cell campaign therefore remains failed. No coverage requirement is waived.

| Model / backend | Audited responses / calls | Primitive task passes / failures | Reused prompt tokens | Mean TTFT, including queue |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.8 dense / CUDA2 | 949 / 843 | 220 / 20 | 83.16% | 20.89 s |
| Qwen 3.8 dense / ROCm2 | 938 / 839 | 217 / 23 | 81.26% | 44.30 s |
| Qwen 3.6 MoE / CUDA2 | 1,203 / 1,043 | 132 / 108 | 88.85% | 16.32 s |
| Qwen 3.6 MoE / ROCm2 | 1,273 / 1,097 | 86 / 154 | 92.00% | 20.70 s |

These figures describe the preceding binary, not the pending capture repair.
Per-cell final receipts are `prefix-economy-*-cell-checkpoint.json`. Disk tiers
filled and evicted metadata-backed entries, but these workloads recorded no
SSD-to-RAM restore. They do not certify that separate restoration path.

Dense ROCm's zero depth updates were an evaluated dynamic-policy hold at depth
one: terminal snapshots show evaluated windows and learned-rule matches under
bounds 1–15. Snapshot counts are observations, not independent policy windows.
Its long delimiter task emitted incorrect literals in native text; the wire and
client preserved those bytes. Neither observation establishes a parser defect.

The publication-limit defect now reproduces on both backends and passes with
the isolated repair. The changing-request-seed regression also reproduces on
both backends against the preceding core. The combined fixes and narrow grep
classifier change are applied to the feature worktree. Initial canonical
qualification stopped during CMake configuration because two new device-free
Integration registrations lacked `NO_MODELS`. That failed receipt is preserved;
the declarations are corrected, and generated CTest metadata confirms the
focused registrations are model-free preflight members. The next full gate
passed all 701 Unit and 365 host-preflight registrations, then stopped on three
CUDA registrations: stochastic MTP resolved request seeds but omitted them from
scalar and batched device admission. The new verifier correctly rejected its
zero-initialized seed bank with `InvalidRequestSeed` (error 17). ROCm's lane was
interrupted by the CUDA failure; that interruption is not a separate failure.

Four focused admission regressions reproduced the omission on the preceding
core. Both callers now pass the already resolved seeds, and stochastic graph
preparation checks descriptors against immutable admission metadata before
capture. The four regressions and all 164 prefill/decode unit cases pass on the
repair. CUDA and ROCm fixed/dynamic request-reset and retained seed-replay
native gates also pass, with no skipped cases and clean driver intervals. The new focused registration is
`V2_Integration_MTPRequestSeedAdmission` in ProductionTestPreflight. The failed
canonical receipt and exact core remain under
`mtp-seed-followup/admission-failure-20261007/`; focused evidence is under
`mtp-seed-followup/admission-repair/`. The canonical rerun completed at
06:50 UTC: 701 Unit and 786 ProductionTestPreflight registrations passed
(366 host, 128 CUDA, 164 ROCm, and 128 shared-device/MPI). Its driver window
contains no new records. All ten focused registrations retained their full
execution logs: 20 test executions and zero skipped cases. These focused counts
include intentionally shared fixtures, not 20 distinct tests. The incremental
Release rebuild, local overlay `ab29f1b5fb52`, exact loader proof and parser/prompt
probes have completed. All 96 paired native timing/oracle samples passed with a
clean driver window. The first CUDA NCU attachment completed its oracle but
raised six driver assertions in the driver's unsupported SM-throttle metadata
query; that diagnostic remains failed, and further CUDA attachment is suspended.
All 16 post-attachment unprofiled health cases passed with a clean driver window.
The CUDA review uses native graph inventories and the final linked SASS/resource
inspection, as required by the CUDA skill after an attachment failure. It does
not claim clean CUDA hardware-counter or achieved-occupancy evidence.

After correcting profiler-only visibility aliases, all 40 isolated ROCm profiles
passed with clean driver windows. Native graph IDs, dispatch correlation, code
object paths and candidate-only counters authenticate the rebuilt core, with no
copies inside those captured graphs. Register counts, launch geometry, native
occupancy capacity and intentional private storage are unchanged on both
backends. The three paired samples per shape show at most 1.413 microseconds
(1.307%) added to the rejection verifier; serial-equivalent medians are at or
below baseline. Overlapping CUDA ranges do not prove a speedup. The reviewed
kernel receipt explicitly retains the failed NCU attempt separately from the
clean execution/ROCm windows. A stale duplicate kernel-field check in the local per-cell wrapper initially
stopped three cells before any native work; its exact failed wrapper and reports
are preserved. The cell now calls the shared kernel validator. Three focused
caller tests reproduce the old failure, and all 37 local admission, cell,
resume and handoff test executions pass after the fix.

The completed six native CUDA admission cases prove dense context 155840 and
MoE context 195328; each next 64-token bucket and model-max 262144 are rejected
by capacity. The restarted HTTP owner explicitly authenticates and reuses that
completed receipt. Dense CUDA and dense ROCm have passed their ordinary/dynamic
512-token controls and entered the webapp phase. MoE ROCm is in its dynamic
control; MoE CUDA follows dense CUDA retirement. The hybrid owner waits for all
four complete HTTP cells, and duplicate dependency cleanup follows its lifetime.
All four fresh HTTP cells and both `PP(TP(2xROCm), TP(2xCUDA))` models remain
required. No full image certificate, commit, or push has been issued. The continuation keeps native-model context,
16 GiB RAM per participant and 32 GiB shared disk prefix capacity, with no
coding-turn deadline. Retired npm dependency cleanup recovered 13.03 GiB while
preserving logs, generated apps, client databases and unauthenticated installs.

## October 7 MoE app completion and authenticated grep failure

Both MoE sessions completed all ten app phases and passed the 77 independent
HTTP/Unicode/concurrency/persistence checks. Their final authored-test discovery
failed because they placed tests outside the requested `tests/` directory. ROCm
retained passing protocol and all seven successful tools; CUDA omitted `glob`,
so its app remains a failed coverage gate. These task failures are retained.

CUDA also produced an invalid regex escape in a `grep` call. Two native-token
snapshot audits matched 594 responses and 554 calls with no changed arguments
or reasoning. The failing regex is byte-identical in native text, HTTP and
OpenCode input; standalone ripgrep reproduces its exact parse error. The
isolated harness fix accepts only an authenticated full-pattern diagnostic and
caret annotation, keeps it as a task failure, and cannot satisfy successful
search coverage. Old-code negative controls fail as expected; both focused
regressions and all 34 harness tests pass. The dedicated preflight registration
is `V2_Integration_OpenCodeGrepRegexEvidence`. Saved original results are intact;
read-only reanalysis still rejects CUDA's missing `glob` coverage.

The waiting MTP refresh now includes this four-file harness patch after its
22-file capture repair (25 unique files; 185 production/test inventory entries).
The original native red/green owners and active binaries remain unchanged.
Fresh homogeneous and hybrid drivers are queued against the exact next source
and require a separate kernel qualification before HTTP admission. The local
captured-verifier probe compiles against both preceding and repaired interfaces,
with no GPU symbol interposition; 96 counterbalanced native timing/oracle
samples are queued after the rebuilt Release prerequisite gate. Profiler
attachment, economy review, all fresh HTTP cells and hybrid inference remain
pending. No commit or push has occurred. See `mtp-seed-followup/checkpoint.json`
and `opencode-grep-followup/` for retained local evidence.


## October 7 request-seed capture reuse follow-up

The completed dense CUDA inventory also contains 950 stochastic-outcome graph
materializations per participant, alongside 949 workload responses and the
earlier dynamic control. Source inspection identifies the second request-varying
capture scalar: the outcome stage stores each resolved request seed by value.
The existing admission path already publishes an arena-owned
`SAMPLING_REQUEST_SEEDS` bank. The isolated `mtp-seed-source` change connects
both verifier laws to that bank; its address participates in graph identity,
while changing seed contents does not. Zero seed data fails on device, and
ambiguous scalar/device seed bindings are rejected. No extra transfer or
allocation is introduced into captured execution.

Two device-free recording-backend tests pass. The native sweep now changes both
seed words, rejects zero seed data, recovers on the next replay, and retains the
same graph over both laws, five top-k widths and every depth 1–15. The public
pipeline proof now retains publication, outcome and parent counts across twenty
changing-seed request resets, as well as within streaming chunks. Five focused
seed/reset preflight registrations complement the original publication gates.
All eleven isolated translation-unit checks pass, including CUDA sm80/86/89/90
and ROCm gfx906 with the canonical spill policies. Compiler/ISA evidence reports
zero register spills; runtime profiling and native execution remain pending.

The original homogeneous cells and publication red/green owner are unchanged.
Four idle downstream publication waiters were retired before doing work.
`run-mtp-seed-native-red.py` follows the original native proof and uses the old
core plus only the publication fix, isolating the new cross-request regression.
`run-mtp-seed-refresh.py` then admits the combined 22-file patch, complete
Unit/preflight and incremental Release overlay against a 185-file source
inventory. Its terminal state requires kernel profiling before a fresh full
HTTP campaign. The earlier HTTP/hybrid waiters are superseded, and the original
hybrid qualification hold remains unresolved. See `mtp-seed-followup/` and
`mtp-seed-refresh-intent.json`. No new image or model pass is claimed yet,
and nothing has been committed or pushed.


## October 7 dense CUDA complete on the archive-economy candidate

The complete dense CUDA cell on image `76a30339ee73` passes its engine and
protocol gate: ten app phases, all seven required tools, 78 independent app
checks, and 240/240 primitive protocol sessions with successful coverage.
Twenty primitive sessions retain model task failures; 220/240 pass their
task checks. The terminal native audit joins 949 responses and 843 calls
with unchanged arguments and reasoning. Stats accounting, the 512-token
ordinary/dynamic controls, runtime checks, archive storage, driver checks
and normal container retirement all pass.

Terminal prefix token reuse is 83.155%, request hit rate is 74.289%, and MTP
acceptance is 82.368%. TTFT averages 20.891 seconds including 16.257 seconds
of queue time during the concurrent primitive sweep; the last TTFT is
1.871 seconds. RAM and shared disk occupancy are 98.303% and 99.794%.
All 39 metadata compactions copy zero payload bytes. The retired archive
audit reads no payload bytes and finds no orphan payload files.
See `prefix-economy-dense-cuda-cell-checkpoint.json`.

MoE CUDA has started on the cleanly released pair. The original dense ROCm
primitive sweep and MoE ROCm final app review remain active. This completed
cell does not qualify the isolated MTP publication capture repair; its native
proof, full prerequisites, rebuilt Release workload and hybrid workloads
remain pending. Nothing has been committed or pushed.


## October 7 dense ROCm app milestone

Dense ROCm has passed all ten app phases, all seven required tools and all
78 independent app checks. Its complete app history joins 68 responses and
109 tool calls to native tokens with unchanged arguments and reasoning.
No OpenCode compaction was needed. Dense CUDA's earlier full app pass remains
valid for this same image; both 240-session primitive sweeps remain incomplete.
See `prefix-economy-dense-rocm-app-checkpoint.json`. MoE ROCm is still working
on persistence, and MoE CUDA has not started.

The saved observer history also proves that the stats endpoint remained
responsive throughout MoE's long request: 406 observations, 2.102 ms mean,
2.360 ms p95 and 4.395 ms maximum HTTP latency. Request age increased
monotonically across setup, prefill and decode; the largest gap between polls
was 1.009 seconds. No additional requests or GPU instrumentation were needed
for this audit. See `long-prefill-0146-diagnostic/stats-responsiveness.json`
under the MoE ROCm cell. These are live observations of image `76a30339ee73`,
not qualification of the pending MTP capture repair.

## October 7 long-prefill diagnosis

MoE ROCm request 146 completed normally after a 388.951-second prefill.
The native renderer/tokenizer probe found 139,434 prompt tokens and a
97,491-token common prefix with request 145. Its 64-token checkpoint boundary,
97,472, exactly matches the production cache restore. The HTTP history was
append-only, but the Qwen template removes earlier reasoning after a later
user message; that render change starts at the prior app phase. The remaining
41,962 tokens required prefill. This proves reuse of the checkpointable prefix
shared with the immediately preceding request, not optimal reuse across all
historical branches.

The terminal stats observation reports 391.827-second TTFT, 0.023-second queue
time and a normal HTTP 200. Request 147 then restored all 139,434 tokens from
the preceding prompt and prefilled in 12.590 seconds. No debugger, profiler,
restart, generation deadline or cache-payload scan was used. Evidence is under
`native-opencode-qwen36-moe-rocm-prefix-economy-shared-archive/long-prefill-0146-diagnostic/`.
This is expected template behavior, not a newly established cache defect.

## October 7 follow-up: MTP capture reuse still requires qualification

The three active coding sessions continue, with MoE CUDA queued on the same
CUDA pair. Live emoji preservation passes native/wire/client equality, and all
three sessions now demote RAM entries to disk. Dense CUDA has also reached
bounded disk eviction. These remain incomplete model runs.

A newly discovered publication graph identity defect carries an unused host
commit-row limit into device-controller capture identity. The isolated fix
removes that conflicting authority. The focused old-library regression is
red; all 85 device-free capture cases pass with the changed stage, and both
production translation units compile. CUDA/ROCm native regressions now cover
fixed and adaptive modes, greedy and stochastic sampling, request reset, exact
serial tokens and graph reuse across streaming budgets. They are prepared,
not yet runtime-qualified. See `mtp-publication-cpu-proof/` and
`prefix-economy-mtp-publication-hold.json`.

The active 176-file source/image binding is unchanged. The hybrid waiter has
been replaced with an explicit qualification hold (five local admission tests
pass). It must consume the corrected candidate's native and full prerequisite
evidence before either requested hybrid model starts. No commit/push.

The native diagnostic executables have now compiled and are queued behind
clean retirement of all four current homogeneous cells. Their controller and
365-file execution binding are retained under `mtp-publication-cpu-proof/`.
They do not clear the Release qualification hold. The additional three-cell
Unicode audit authenticates 178 responses and 196 native calls; all requested
emoji remain exact through the wire and client arguments. MoE's first Unicode
edit reported the known exact-match-not-found outcome, then its subsequent
edit completed with the same preserved Unicode inventory. These are partial
session audits, not final workload passes.

The incremental refresh is now queued as `run-mtp-publication-refresh.py`.
Its 13-input plan authenticates the eight-file patch and existing Ninja trees
without starting GPU work. Only a successful four-case native red/green proof
and clean retirement of every current owner admit source promotion. It then
preserves the old Integration core, runs the complete canonical prerequisites,
builds Release and copies a small local image layer. Release HTTP validation
and the hybrid hold remain outstanding; this queue cannot resolve that hold.
The later partial audit in `prefix-economy-review-native-checkpoint.json`
authenticates 249 responses and 280 native calls with unchanged arguments and
reasoning. MoE's identical-old/new edit is independently authenticated as the
existing `edit_no_change` task outcome, not reclassified as a success.

Dense CUDA has now passed all ten app phases, all seven required tools and
all 78 independent app checks. Its 100 responses and 116 native tool calls
match exactly, including the one OpenCode context-compaction response. The
240-session primitive sweep has started. Dense ROCm remains at seven completed
app phases and MoE ROCm at six; their sessions are still active. These results
belong to image `76a30339ee73`; the isolated capture repair is not yet certified.
See `prefix-economy-dense-cuda-app-checkpoint.json` for the bounded milestone.

The full repaired-image HTTP campaign is also queued behind that refresh as
`run-mtp-publication-stages.py`. Its plan binds the expected 182-file source
and 15 local driver/helper inputs. Seven device-free admission cases reject
stale source/binaries, another image, incomplete refresh and missing focused
MTP gates. The workload remains all four requested homogeneous cells with
512-token controls, ten app phases and 240 primitive sessions per cell; the
same context, cache budgets, native parsing, stats, storage and driver gates
remain mandatory. It does not resolve the hybrid hold or reuse the preceding
image's model passes.

The dense CUDA primitive sweep's first 21 completed sessions have valid
protocol evidence; one delimiter session retains a task failure. Native token
decoding confirms the model omitted the requested literal closing tool-call
tag. The parser preserved that generated value exactly, and only the terminal
token was excluded from emitted text. See
`prefix-economy-dense-cuda-delimiter-task-evidence.json`; this failure is
retained rather than counted as task success.

The repaired-image hybrid handoff is now queued as
`run-mtp-publication-hybrid-handoff.py`, with 23 bound local inputs. Nine
device-free handoff tests and five tests of the actual hybrid driver pass.
The handoff requires the exact native red/green cases, complete prerequisites
and all four repaired-image HTTP cells. Only then does it preserve and retire
the old image's idle waiter using a native process handle, resolve this
specific capture hold for the qualified image, and run both requested hybrid
models sequentially. The present hold is still unresolved and no hybrid GPU
work has started. A serial cleanup-observer handoff is queued behind the
existing observer; these workers cannot prune the same dependencies
concurrently. See `mtp-publication-hybrid-handoff-intent.json` and the two
`mtp-publication-hybrid-*-tests.log` artifacts. The read-only summary now
includes the complete repair, homogeneous and hybrid queue.



## Qwen/OpenCode regression work (2026-10-06)

**Latest — complete prerequisites passed; fresh candidate workloads started.**
The unchanged local Release image `76a30339ee73` passes all 700 Unit and 776
production-preflight registrations (362 host, 125 CUDA, 161 ROCm and 128
exclusive). All eight mixed-backend ordinary/dynamic pipeline generation
fixtures pass, and the complete driver interval is clean. The qualification
binding now includes 176 source/test files after the hybrid-controller evidence
and authenticated OpenCode glob-error regressions. The serving binary is
unchanged. The exact registration audit lives in
`prefix-economy-prerequisite-registration-audit.json`.

These counts describe CTest registrations, not universal nested-case coverage:
three unchanged legacy loader fixtures have 22 skipped and 12 disabled GTest
cases. The verbose device-free diagnostic and exact list are retained in
`prefix-economy-unit-loader-coverage-observation.json`; the new feature
regressions are separate. No model or image certificate is claimed.

Fresh dense/MoE ROCm and dense CUDA cells passed exact 512-token ordinary/
dynamic/streaming controls and stats/reset checks, and started their coding
sessions. MoE CUDA follows on the shared pair. CUDA admission reconfirms
155,840 dense and 195,328 MoE tokens
with the next 64-token bucket rejected. ROCm retains 262,144 tokens. Every cell
uses 16 GiB RAM per participant, one shared 32 GiB active disk payload budget,
default dynamic MTP and the full ten-phase app plus 240-session tool sweep.
The two requested ROCm-first/CUDA-terminal hybrid cells remain queued behind
the four fresh passes and clean retirement. The last two frozen ROCm caches
reclaimed another 619 GiB without reading payloads. Nothing is committed/pushed.

**Earlier — bounded payload storage replaces whole-archive compaction.**
The live v3 archive compactor could not catch continuous payload churn: one
metadata-only observation found a 227.4 GB journal, a 108.3 GB temporary copy,
and only 34.2 GB of live payload. No payload bytes were read by that inspection.
Format 3 stores immutable payload files beside a metadata-only journal and
unlinks retired generations after their journal mutation becomes durable.
Selected readers retain only their selected inode; maintenance never copies
payloads. Restart authenticates file extents and reclaims uncommitted orphans
without payload checksums, hashes or scans. Legacy formats are rejected intact.
The two economy regressions fail against the old implementation. The durable
journal publication regression also reproduced a writer-lease gap before the
directory fsync; publication now retains its lease through that durable edge.
Twenty-four focused registrations pass, and all 700 Unit registrations pass
in 87.58 seconds. Bounded storage, selected reads and publication ordering pass
twenty repetitions each. The background concurrency gate timed out once while
large cache deletion and full relinking overlapped; retain that failed receipt.
Its subsequent unchanged twenty-repetition run passes in 36.49 seconds. Full
native preflight and four-cell qualification remain required.
The complete CMake-owned host preflight subset also passes 361/361 with no
skips; it remains a diagnostic subset of the 774-registration native gate.
The small local Release image is
`76a30339ee730f7f6666d89effcf5c197c4531a56f223baa99d6f9d3da81064f`,
with 175 modified source/test identities, matching copied binaries and native
loader closure. Fresh prompt/parser probes match both saved 512-token controls.
`prefix-economy-stages.json` now waits for both frozen ROCm owners to retire,
then runs complete prerequisites, context admission and four new native cells.
The new cells additionally require metadata-only compaction counters and an
independent retired-storage ownership audit with zero payload reads.
The unstarted `prefix-stats-stages.json` queue was
explicitly superseded, with its original binding and cancellation receipt
retained; it must not qualify the newly changed source or binaries.

The second v3 MoE ROCm app completed all ten phases. Its 154-request history
passes the candidate protocol recheck, retaining all editing, missing-coverage
and app-acceptance failures. A separate partial native audit joins 388 responses
and 349 calls with no argument or reasoning changes. OpenCode's authenticated
identical-edit rejection is now a task error; 31 harness tests pass, including
negative controls for changed arguments and unrelated errors. The focused
`V2_Integration_OpenCodeUnchangedEditEvidence` is registered. Original app
reports remain unchanged; these diagnostics do not certify the live cell.

Cleanup removed 149.9 GiB from two stopped task containers and 356.5 GiB of
derived payload files from six authenticated retired cache namespaces. All
source, model weights, raw exchanges, generated apps and qualification receipts
remain intact. `workspace-cleanup-20261006/second-cleanup-summary.json` and its
per-owner receipts record the deletions. After MoE CUDA retired normally,
another 444.9 GiB of its derived cache files were reclaimed; its evidence lives
in `workspace-cleanup-20261006/retired-moe-cuda-v3-cache/`. The completed native
audit joins all 1,174 responses and 1,047 tool calls without argument/reasoning
changes. All 240 primitive sessions pass protocol and coverage; task quality is
133/240 and the original app remains failed. The dependency cleanup observer now
follows the frozen v3 pipeline; active cache growth remains under observation.

**Earlier candidate snapshot — superseded before native qualification.**
The active feature worktree is now `prefix-stats-source/`. The existing `source/`
worktree is detached, with all 165 qualified production/test hashes intact.
`prefix_cache.storage.tiers` publishes RAM/disk payload utilization and exact
shared-archive capacity. `storage.churn` separates demotions, hydrations,
evictions, physical payload writes and metadata-only backing reuse. Epoch
resets preserve occupancy and cannot split a demotion from its RAM eviction.
No HTTP poll reads cache payloads or enters GPU/maintenance/MPI execution.
Eighteen focused registrations pass (eight complete Unit registrations and ten
explicit preflights), with real GET/PUT, concurrent freshness, nested TP/PP,
archive aliases, restart and actual selected read-byte coverage. The OpenCode
ambiguous-edit diagnostic now counts as a task error only when its tool identity
and native argument bytes are exact; all 28 harness tests pass. Full candidate
gates and rollout remain pending; no commit or push has occurred.
The complete candidate Unit recheck passes 700/700 in 80 seconds. The first
run's sole failure was the absent exported compile database; the CMake export
setting is now enabled. Keep that failed setup run and the passing recheck as
separate evidence. Full native preflight waits for serving-device ownership.
The complete host preflight subset now passes 357/357, with zero skips. Its
local controller caught a successful `SystemExit(0)`; retain that failed wrapper
report beside the independently validated CTest receipt. Release and binary/
loader verification pass for local image
`123bbca769ebccdc0536a0e5a853fbcd7ff728682ac042c8c8a75ba7633c55fd`
(174 modified source/test identities). The queued `prefix-stats-stages.json`
controller will run full prerequisites and all four native cells only after
the existing campaign has cleanly retired its devices. Its real HTTP observer
also checks occupancy arithmetic, one shared disk budget and monotonic churn.
The candidate's fresh metadata-only prompt/parser probes match both models'
preserved controls. Partial audits authenticate 800 completed dense responses
and 739 native tool calls with zero changed arguments or reasoning mismatches;
keep those observations separate from the final whole-session audit. The CUDA
MoE lifetime has its own passive hardware observer.
Additional partial candidate-parser audits pass for 157 fresh MoE responses
and 159 tool calls across CUDA and the second ROCm session. They show no
argument changes or reasoning mismatches; whole-session completion/audits
remain outstanding.

MoE CUDA has now finished ten app phases, including real automatic compaction:
173,812-token coding context, a 96,844-token summary request (190.76 seconds),
then a 10,105-token resumed prompt. The native audit's missing-`tools` assumption
and the harness's immediate-history check both rejected this valid auxiliary
request. Five local audit regressions and all 30 harness tests now pass, with
explicit preflight `V2_Integration_OpenCodeCompactionContinuation` registered
and passing. It defers the pending join to the resumed coding request and still
rejects missing/changed continuation evidence. Both parser probes match all
198 completed responses/199 calls in the fresh partial audit; summary/resumed
content bytes match as well. The original app remains failed for missing
`grep` coverage and final sentinel, plus a retained editing task mistake.
The candidate replay clears only the erroneous protocol diagnosis. Preserve
`opencode-compaction-*` receipts and the previous binding beside the refreshed
174-file queued qualification binding. Fresh complete qualification now owns
771 preflight registrations. Release binaries
and all frozen v3 source identities are unchanged.

Fresh MoE ROCm exchange 117 contains an incomplete native `read` call: `filePath`
was never generated. Its 121 content and 2,666 reasoning characters survive
unchanged, with no published executable call. The existing required-argument
regression and its `V2_Integration_QwenMixedParameterFraming` registration pass.
A new partial audit verifies 118 responses/113 calls; keep
`opencode-moe-rocm-incomplete-call-*` as parser evidence, without interpreting it
as numerical MTP equivalence. The ongoing app has advanced to phase six.

The v3 dense CUDA cell now passes its complete engine/protocol gate: ten app
phases, all 240 tool sessions and required coverage, 972 independently reconciled
HTTP requests, and 850 exact native tool calls. Capture/MTP, transfers, prefix
ownership, capacity, normal shutdown and driver reports all pass. Primitive
task quality remains 221/240; the app passes final acceptance but retains one
recovered edit-target error. MoE CUDA has started on the cleanly retired pair.
This is v3 evidence, not a completed stats-candidate or four-cell campaign.
A separate `qwen36-moe-rocm-metadata-v3-second-session` now runs on the idle
ROCm pair with the same complete controls, coding workload and 240-session
sweep. Keep its result independent of the interrupted original. Stats
qualification waits for the additional controller's authenticated clean
retirement, with six local scheduling regressions passing. The manifest and
evidence live in `prefix-metadata-v3-additional-cells.json` and its named cell;
the candidate's complete qualification/four-cell run remains required.

Both dense backends completed the ten-phase webapp. Dense ROCm continues its
240-session tool batch; its app passes protocol and task checks.
MoE ROCm was interrupted after a repeated
same-command loop and retired normally with clean driver evidence; its app is
incomplete. The exact preserved looping response is token-identical with cache
and MTP disabled independently. Loop-onset controls are now complete and pass
all eight prompt/completion-ID comparisons for exchanges 84 and 90. The earlier
history finishes normally under the diagnostic seed, while the later history
repeats the test command under every configuration. This points to model
behavior in that conversation; its deeper trigger remains unproven. Keep
`opencode-moe-repetition-reproduction/`,
`opencode-moe-repetition-onset/`, `prefix-stats-final-tests.xml` and original
native artifacts distinct; none turns an interrupted app into a passing cell.
Five local controller exit-status regressions now pass, including previously
failing workload, observer and driver negative controls. The queued scripts
propagate these failures to their parent pipeline. Preserve the before/after
evidence in `prefix-stats-controller-tests.json`; source/image bindings and
running workloads are unchanged.
Workspace cleanup reclaimed 64.7 GiB from 805 build artifacts no longer named
by their existing Ninja target inventories. Active builds, caches, source and
test evidence remain intact; `workspace-cleanup-20261006/` records the exact
inventory, active-file checks and recovered filesystem space.
Completed tool fixtures were also retaining two npm installations apiece.
Removing 975 generated duplicates recovered another 59.4 GiB (about 171 GiB
free immediately afterward). Preserve one installation per dependency lock and
all prompts, raw exchanges, client databases, generated apps and test evidence.
Five cleanup regressions pass. `opencode-dependency-retirement/` retains each
plan/deletion journal and observes the queued pipeline, cleaning only fixtures
whose client and artifact publication have already completed.
The original cleanup observer stopped on an inaccessible procfs environment.
Its replacement defers deletion on an unavailable ownership sample, then
requires fresh proof at the next scheduled sweep. Eight regressions pass,
including both observation boundaries and recovery freshness. Preserve the
original failure and the separate observer/test recheck receipts.

**Current update — metadata-only payload I/O passes complete prerequisites.**
The user requested removal of all RAM/disk cache-payload checksums and hashes.
The disk writer previously scanned RAM payloads for publication/reuse and cold
restores verified bytes before reading them again. The replacement uses opaque
version metadata, bounds selected sections by the retained inode and hydrates
them directly once. Metadata checks and prompt keys remain small. Archive format
2 rejects legacy format 1 intact. Foreground verification scratch is removed,
reducing the shared archive reservation from 8 MiB to 4 MiB for compaction only.
Focused tests observe actual native read extents, forbid RAM payload reads on
reuse, cover version replacement/restart/compaction and retain selected LRU order
before capacity admission. The `prefix-readset` candidate was superseded before
GPU fixtures started; its complete driver interval is clean, but it is not a
completed qualification. New full gates and native sessions remain required.
The seven affected device-free executables pass (271 tests, three existing
hardware-dependent skips); six explicit preflight registrations pass all twenty
repetitions in 40.59 seconds. Preserve `prefix-metadata-final-suites.json` and
`prefix-metadata-final-focused.json`. The new Release build and full canonical
qualification use fresh `prefix-metadata` evidence and archive directories.
That candidate's complete Unit gate stopped on five registrations (695 pass).
The failures exposed stale eager-lookup assertions, a missing device-free label,
two GPU-dependent checks admitted to Unit, and a real host-resident activation
bug: its no-op path allocated/promoted an uninitialized GPU buffer. The transfer
fix preserves host authority for CPU/CUDA/ROCm destinations and has an explicit
preflight entry. The scratch lifecycle test now runs symmetrically in CUDA and
ROCm integration. All five affected Unit registrations now pass, as do eight
host and three native preflight registrations repeated twenty times; the native
driver interval is clean. Preserve `prefix-metadata-prerequisites/` as failed
and `prefix-metadata-v2-final-focused.json` as the new focused evidence. Complete
qualification and native coding sessions now use the fresh `prefix-metadata-v2`
candidate; no preceding result is rewritten or promoted.
The v2 candidate passes complete Unit (700/700, 78.28 seconds). Host preflight
stops at 348/350 because the pipeline-prefix fixture still inspected an eager
terminal pointer and attempted to select two frontiers from one consumed lookup.
The fixture now materializes its selected terminal and proves another frontier
needs a fresh lookup. All six CPU/CUDA/ROCm registrations pass twenty repetitions
(24.70 seconds host, 94.00 seconds native) with clean driver evidence. No further
production-library change was needed. Preserve the failed v2 host gate and
`prefix-metadata-v3-final-focused.json`; fresh full qualification proceeds under
`prefix-metadata-v3` before any native coding session.

The fresh v3 gate passes **700/700 Unit and 763/763 production preflight** in
2,513.28 seconds; its complete driver interval has no findings. Local Release
overlay `f8baf85b31ab` and its 165 changed production/test files are bound by the
v3 receipts. CUDA dry runs prove the 155,840-token dense and 195,328-token MoE
capacity boundaries, including rejection of each next 64-token bucket. ROCm
uses 262,144 tokens. All three initial cells pass 512-token MTP-off/dynamic and
stream/nonstream controls plus live stats checks, and are executing fresh
ten-phase webapp sessions. Each retains raw token output, 16 GiB RAM per
participant and a shared 32 GiB SSD archive. The four full native workloads,
independent parser audits and clean driver retirement remain required;
MoE/CUDA follows dense CUDA retirement.

**R — complete cache/startup prerequisites pass; native qualification is pending.**
The user-requested normal reboot restored all four MI50 links after the PCIe
outage. An added fan is in place; thermal cutoff is suspected, not confirmed.
The interrupted runs, original driver windows, complete logs and generated apps
remain preserved in `reboot-pause-20261006T142549Z/`. No commit or push has occurred.
The dense cache reproduction improves from 34/48 to 48/48 cached tokens with
exact KV bytes in the same RAM budget. All 38 cache ownership tests pass; three
focused cache preflight registrations pass 20 repetitions. A separately
reproduced lazy-backend startup-policy defect has a native-entry trap regression
and also passes 20 preflight repetitions after its fix. The full device-free
DGO suite now completes. Exact read-set byte counters are covered, and a
reproduced archive-open failure now retains its native storage diagnostic.
The fresh Release overlay and complete gates above now pass; long-session
archive-capacity/ancestor behavior remains under native observation.

The preceding candidate passed complete 700/700 Unit and 753/753 preflight in
2,564.66 seconds, with clean driver evidence. Qwen3.8 on CUDA and ROCm, and
Qwen3.6 MoE on ROCm, pass same-image 512-token MTP-off/dynamic and stream/nonstream
controls. All three have entered their fresh OpenCode webapp sessions. The
fourth cell, MoE/CUDA, follows dense CUDA retirement. Earlier failed campaigns
remain failed; those gates apply only to the preceding source/image.
The interrupted native lifecycle windows contain a new AMDGPU
`amdgpu_amdkfd_restore_userptr_worker` CPU-hog warning at 12:18:26 UTC,
during the MTP-off controls, before any main coding server started. Those
windows therefore cannot receive a clean driver certificate. The preceding
complete Unit/preflight pass remains valid. The three coding sessions were
preserved and stopped for the reboot;
the CUDA sequence will reject its next admission when the failed driver window
retires. Preserve `runtime-stats-native-workqueue-warning-first-observation.json`
and the original armed checkpoints. Read-only mapping observations confirm both
16 GiB ROCm RAM-cache buffers use native driver-owned backing. No log is cleared,
checkpoint re-armed, driver warning waived, or timeout substituted.

The new local Release overlay `ef21a8aaa1e02` includes passive `GET /stats`,
epoch-resetting `PUT /stats`, the HIP event-producer lifetime repair and mixed
Qwen parameter framing, and DeviceRegistry startup-policy enforcement. Its four
copied binaries, native installation receipt
and actual HIP loader path are authenticated. This is a local diagnostic
overlay, not a newly certified image.

The user explicitly accepts less efficient MoE work and model mistakes while
requiring engine/parser correctness. The native cells now use the explicit
protocol gate and retain every strict task failure, app acceptance failure and
recognized tool-domain outcome. Required arguments, SDK validation, unchanged
wire/client inputs, complete histories and stream framing remain fatal checks.
Every selected case must also exercise all its requested tools successfully.
`V2_Integration_OpenCodeQualityProtocolSeparation` registers the focused positive
and negative controls in preflight. Runtime and exact-generation controls remain
independent prerequisites; a protocol result alone does not prove model math.
The new image receipt binds all 155 changed production/test files and its exact
Release binaries. `runtime-stats-qualification-binding.json` binds the fresh
gate to that same source; prior image and cancelled-run evidence remain retained.

The MoE app completed 166 requests, 81,887 completion tokens and a maximum
215,693-token prompt over 9,886.7 seconds. Native IDs prove two writes contained
both `content` and `filePath`; a missing newline immediately before the content
terminator made the parser swallow the path into content. Line framing now
accepts a newline on either side of the terminator, retains literal inline tags,
and requires all declared required arguments before publishing a call. Both
focused tests fail on the old implementation. Parser and HTTP tests cover both
argument orders, LF/CRLF, emoji, empty and whitespace strings and every SSE
split point; four focused preflight registrations pass twenty repetitions.
Preserve `mixed-qwen-framing-evidence/`.
The final read-only audit joins all 1,278 OpenCode responses to their native IDs
and exact prompt/completion counts. An independent inverse renderer verifies
1,086 of 1,090 wire calls against complete native argument bytes. Four fail:
the two swallowed paths above and two model-generated calls that genuinely
omitted a required path. The repaired parser corrects the former and keeps the
latter inert; all reasoning bytes agree. Preserve the preceding partial audits
and `moe-final-native-audit/result.json`. The four native candidate
cells require this independent argument check as well as production replay.

Independent app checks also found an overly broad task-list route and an
unmatched brace in the HTML formatter. Emoji POST/PATCH round trips pass;
the two emoji GET failures return the wrong response shape because of routing.
These are generated-app defects, separate from the native parser defect. The
original app remains failed. All 240 primitive/workflow sessions completed:
81 pass the strict task gate and 239 pass the separate protocol assessment.
Every case exercised its required tools. The sole protocol failure contains
the two model-generated calls missing a required path; the old parser published
those calls. Preserve `moe-final-quality-protocol/result.json` and every original
task failure, including wrong paths, altered content and tool-domain outcomes.

An operator incorrectly called the driver observer's finalization operation
during the live old-image run. Its original final report remains failed because
the checkpoint was already closed. A separate read-only retained-cursor audit
covers all 600 kernel-log records through normal process retirement and finds no
GPU fault; it does not certify or re-arm the old cell. Fresh qualification uses
its own driver checkpoint. The first two attempts stopped on launcher environment
errors (lost Python path, then root-owned ccache configuration). Restoring the
actual account identity, pinned PATH and compiler-cache settings passes all three
affected Unit registrations in the exact private namespace. The complete
699-Unit/751-preflight transaction then passed Unit and all 341 host preflight
registrations. Its native lanes exposed another launcher error: HIP uses
`CUDA_VISIBLE_DEVICES` when `HIP_VISIBLE_DEVICES` is absent, narrowing the four
ROCr devices to two. The RCCL four-device cases consequently fail admission.
The corrected invocation explicitly binds all three masks and authenticates
native runtime counts (four ROCm, two CUDA). All six formerly masked regressions
then pass in 188.56 seconds with clean driver evidence. Preserve
`four-rocm-mask-proof/` and the failed complete run under `.hip-mask-rejection`.

The ownership audit found full-machine discovery in backend-filtered router,
projection, CPU-reference CUDA/GDN and cross-vendor transfer fixtures. Their
targets now declare `FullInventory`; actual generated registrations are covered
by `V2_Integration_PreflightInventoryOwnership`. Device-free MTP and graph
construction preflights now inherit the existing Unit CPU-only startup guard.
The broader audit cancelled the preceding gate after all 700 Unit and 342 host
registrations passed; that run did not complete GPU qualification.

The Unit logs also exposed a production defect: `DeviceRegistry::discover()`
ignored the typed backend-startup policy. It now checks that authority before
either vendor enumeration, memory query or P2P probe. New regressions fail before
the fix for CPU-only startup and both selective exclusions. Native checks prove
excluded CUDA primary contexts stay inactive and excluded ROCm never opens KFD.
All 20 focused registrations pass in 23.85 seconds with clean driver evidence.
`V2_Integration_DeviceRegistryStartupPolicy` explicitly locks the defect into
preflight. The refreshed gate has 700 Unit and 753 preflight registrations:
342 host, 124 CUDA, 160 ROCm and 127 exclusive. Preserve
`device-registry-startup-evidence/` and the preceding image/gate artifacts under
`.before-startup-policy`. Complete Unit/preflight and the new driver window now pass. Preserve
`runtime-stats-prerequisites/` and `runtime-stats-qualification-controller.json`.
The three initial native cells also pass live GET/PUT epoch, configuration,
weighted-control metrics and responsive-observer checks; sustained coding and
whole-workload counter validation remain in progress.

A separate flaky ROCm cleanup failure was reduced to a native HIP event
dereferencing its retired producer stream. ASan proves the use-after-free;
real stream-address reuse also makes the old runtime incorrectly join an
unrelated capture. A registry-locked lease authenticates the original device
and monotonic stream identity without retaining a retired queue. Twelve native
cases plus the complete 23-case prefill lifecycle suite pass twenty fresh-process
repetitions; the canonically installed artifact passes all seventeen HIP
preflight registrations with clean driver evidence. The full gate still needs
all four ROCm devices; the earlier two-device-masked run failed six inventory
requirements and the now-reduced lifecycle case. Preserve
`hip-event-lifetime-{diagnostic,proof,installed-proof}/` and the original failures.

The stats regressions prove weighted rates, lifecycle freshness, active/reset
epoch boundaries, disconnect accounting, HTTP error counts and responsiveness
while inference is queued. Idle reads/resets call no runner methods. The fresh
native cells additionally compare the endpoint with exact HTTP token usage and
observe it during generation. After the full workload they reconcile all saved
SSE usage and native prefix/MTP terminal counters with the cumulative HTTP
snapshot, including weighted rates and TTFT sample counts derived from actual
nonterminal token traces. `runtime-stats-qualification-controller.json`
owns the queued complete gate; `runtime-stats-opencode-campaign.json` owns the
four dependent cells. No feature commit or push has occurred.

The final old-image prefix ledger covers 1,280 requests (including the two
controls): 23,052,546 of 25,253,804 prompt tokens reused (91.28%), with reuse
on 1,038 requests (81.09%). The final 100 requests reuse 79.88% of tokens.
These are retired log observations, not live results from the new HTTP route;
preserve `moe-prefix-reuse-20261006-final.json`.

The preceding shared-archive image passed all 697 Unit and 744 preflight
registrations before these additional changes.
Local Release overlay `6d8342114e86` binds 79 changed production files. The
512-token MoE stream/nonstream control passes exact prompt/completion IDs,
uses default dynamic MTP depths 1–15 and records twelve depth updates.
ROCm retains the model maximum context of 262,144. All cells retain a 32,768
output budget, 16 GiB RAM prefix storage per participant and one shared 32 GiB
disk tier. No generation, turn or inactivity deadline is introduced.

The dense ROCm startup crashed in the shipping ROCm SMI dependency's mutex
unlock, before serving any request. A CPU-only two-container reduction proves
that host IPC with private PID namespaces lets unrelated threads with TID 1
enter the same recursive robust mutex. Matching PID namespaces excludes the
contender in all twenty repetitions. Launchers now use private IPC by default;
explicit host/peer IPC joins the corresponding PID namespace. Retirement checks
the owning mount namespace before signalling a process. Nine device-free
regressions pass and `V2_Integration_DockerIPCIdentity` explicitly registers the
contract in preflight. The development container now requests its own 32 GiB
shared memory. The currently running container is not recreated.

The CUDA model maximum does not fit the two 24 GiB 3090s. Exact production
`--dry-run` admission finds the largest 64-token-aligned context at 155,840 for
Qwen3.8 dense and 195,328 for Qwen3.6 MoE; the next 64-token step fails each
complete physical BOM. These are admission observations, not successful native
inference or an image certificate. The first attached diagnostic returned no
logs, so the completed probes use explicit create/start/wait/log collection.
Their raw failures remain retained. The same boundaries are re-authenticated
on overlay `9f23be8362bf`, including model-maximum rejection and clean driver
evidence. The first launcher qualification used only physical ROCm 0/1 and
failed the full gate's four-device inventory requirements; its lifecycle failure
is reduced above. Preserve `ipc-identity-*`,
`rocm-smi-startup-diagnostic/` and both detached CUDA admission directories.

The delimiter generation failures reproduce in an explicit llama.cpp MTP-off
control with the same GGUF and exact 6,390 prompt IDs: early EOS while quoting
a tag, inserted line breaks, and the same `_kernel` substitution. All three
reference seeds fail exact content. The CUDA layer-split control establishes
that these symptoms occur outside Llaminar/MTP, without claiming cross-engine
numerical parity. The three controlled native raw write payloads equal their
wire arguments; this does not exclude other parser failures in the longer run.
Preserve `template-delimiter-reference-comparison.json` and its linked raw
evidence; strict stress failures remain red. A separately attempted reference
tensor-split load crashed in NCCL and supplies no inference evidence.
A separate pinned Unsloth Q8_0 artifact also preserves only one of three exact
writes in that same reference configuration; the other two alter literal tags
in raw generated tokens. Its 6,390 prompt IDs match, all requests finish on EOS
without truncation, and shutdown/driver checks pass. Higher precision alone
does not eliminate the observed generation failure. Preserve
`template-delimiter-q8-reference-comparison.json`; artifact revision and
SHA-256 are retained without asserting a quantization-only comparison.
The live trace omission of terminal IDs is repaired, with twelve focused
cases and the full chat-handler suite passing. The new preflight entry is
`V2_Integration_HTTPGeneratedTokenTrace`.

The longer delimiter session subsequently exposed a separate parser defect:
four native calls became three wire calls. A file string containing
`A</parameter><parameter=x>KEEP</parameter>B` was misread as argument syntax,
so the valid write appeared as inert assistant content. The reduced old-code
regression is red. The fix authenticates newline framing and schema-owned
parameter boundaries, retaining literal bytes, compact calls and fail-closed
unknown/duplicate arguments. Parser split-point and HTTP JSON/SSE tests include
emoji beside the tags. Eight focused CTest registrations pass, including both
complete parser and chat-handler suites. Preserve
`template-delimiter-long-generation-observation/parser-defect.json` and
`qwen-adjacent-parameter-{red,focused-ctest}.log`.

Release overlay `74576c0cd143` includes both repairs with authenticated binaries.
Its fresh complete 697 Unit/742 preflight transaction waits for both current
native servers to retire. The earlier trace-only queued controllers were
superseded before execution; their evidence remains. No new image qualification
or native parser-fix result is claimed yet.

The harness now requires matching completed write/read calls to the original
workspace's result file and exact content in both wire and client write inputs,
with the completed verification read following that write.
Correct final bytes cannot hide a wrong call followed by another operation.
Six false-positive negatives fail the old file validators. A second gap allowed
a workflow with an unrelated or failed shell command to pass. The harness now
authenticates its exact unittest command, workspace, client arguments, zero
native exit and nonempty discovery summary. Eight old-code false positives
are retained; nine negatives plus absolute/default/relative workspace positives
pass after the repair. All 21 Python tests pass directly. The exact-file and
workflow focused CTest entries pass; the first combined Unit invocation exposed a
two-second socket timeout in the existing transport test under live disk I/O.
Twenty unchanged focused repetitions pass, and the subsequent combined three
CTest entries pass in 1.26 seconds; the original failure remains retained.
Read-only revalidation of all 98 completed native sessions retains 91 passes
and seven failures without rewriting their original evidence. The explicit
preflight entry is `V2_Integration_OpenCodeExactFileToolEvidence`.
The workflow defect has its own `V2_Integration_OpenCodeWorkflowTestEvidence`
entry and `opencode-workflow-test-proof-{red,green,focused-ctest}.log` evidence.

The native sweep also retains a model-emitted missing `content` parameter on
an empty write; raw IDs prove the omission and its later correction. Strict
session failure remains recorded. Storage economy is unqualified: the tool
archive and compaction copy exceeded 262 GiB despite a 32 GiB live-payload
budget. A recorded 212.44-second primitive-client pause let compaction publish;
the app and in-flight inference continued. This intervention preserves session
evidence but does not certify continuous pressure or repair archive growth.

The superseded primitive campaign was retired after 98 completed sessions
(91 pass, seven fail), retaining four interrupted client workspaces separately.
Its server processed the queued administrative shutdown and exited zero; all
runtime policy and driver checks pass. The admin observer's five-second timeout
is retained rather than treated as proof that shutdown was rejected. No second
shutdown request or server signal was needed. Preserve
`template-primitives-parser-fix-retirement.json` and
`qwen38-template-mtp-tools-server.retirement.json`.
The queued `final-tools-controller.json` was superseded before execution by the
shared-archive repair below. Fresh sweeps retain the model, topology, MTP/context
and cache budgets, with task-owned archive directories on the model volume for
append/compaction headroom. Extra filesystem space is not an archive economy fix.

The application completed nine of ten phases and 47 authored tests. Its
105,154-token request matched 104,883 cached tokens but reported 777.287 seconds
of prefill before producing a coherent 159-token answer. Three saved requests
around 105K tokens match independent HF rendering, token IDs and decoded text;
successive requests preserve the entire preceding prompt prefix. Preserve
`opencode-long-app-prompt-observation/`. Storage activity was observed during
the delay. The following 105,451-token request missed the entire prefix and
completed after 1,220.369 seconds of prefill, producing coherent tool output.
Brief host-only debugger attachments sampled both TP workers in HIP's queue-slot
wait during batched graph submission while archive workers were idle. The
request later completed; those transient stacks do not establish a deadlock.
The attachments and exact live wait-loop instructions remain diagnostic evidence
in `app-wait-stack-file-observation.json` and `app-hip-queue-wait-proof.json`.

Read-only committed archive metadata then showed 24.767 GiB of recurrent
checkpoint state and 6.950 GiB of attention/MTP rows in the shared 32 GiB tier.
One participant's root prefix blocks had been evicted. A reduced test proves
that a participant could trust its stale disk index and release the remaining
RAM copy after a peer evicted that record. The initial local-residency repair
still fails with an independent archive writer. The final implementation instead
requires the existing ordered archive writer to authenticate backing for each
selected RAM victim and consumes that receipt at source retirement. Identical
backing appends only a metadata touch. All 53 focused cache tests pass, including
retains arriving during publication, independent archive writers, and every
snapshot section. Twenty repetitions of the eviction cases and both explicit
preflight registrations pass. Preserve
`prefix-shared-archive-red3.log`, `prefix-shared-archive-candidate-tests.log`
and `app-archive-metadata-observation.json`.

The four queued prerequisite/native/sweep controllers were superseded before
this production change. The subsequent turn interruption terminated the app's
client during phase ten. Its workspace is retained; an independent snapshot
passes all 78 HTTP acceptance checks and 47 authored tests, but the ten-phase
session remains incomplete. The server then accepted its normal shutdown and
exited zero with passing graph, transfer, MTP, prefix, capacity and driver checks.
Both incremental Integration and Release builds pass. The shared-archive
overlay subsequently passed all 697 Unit and 744 preflight registrations with
clean driver evidence. Native startup findings and the next qualification
transaction are recorded at the top of this section. The goal requires OpenCode
stress for both Qwen3.8 dense and Qwen3.6 MoE on both two-device ROCm TP and
two-device CUDA TP; a pass on one combination cannot close that goal.

| Defect or contract | Retained proof |
|---|---|
| Captured capacity confused with live draft depth/bonus row | 29,160 captured replays per GPU backend across depths 1–15, policies, sampling laws and transaction boundaries |
| Streaming publication reset depth learning | CPU state-machine negatives and 46,080 short captured response windows per GPU backend; stale request/arena identities rejected |
| Prefix restore retained unread sections/completed sources | Focused CPU/native lifecycle gates, including 80 held-read cases per GPU backend and fixed-capacity fragmented placement |
| Full-context MTP workspace and host logits capacity | TP 1/2/4/8 and context 4K/32K/256K controls; main host logits now one 993,280-byte row |
| Qwen sampling defaults and reasoning history | 25 pinned official cards, mode/override tests, native Qwen2.5 repetition controls and captured GPU sampling oracles |
| Tool-string types, literal delimiters, emoji and SSE usage/errors | Explicit parser, HTTP, SDK-accounting and streaming-fragment preflight entries |
| Missing declared pre-tokenization and NFC | All 7,302 real-GGUF comparisons match pinned HF token IDs and decoded text; focused boundary/normalization preflight |
| Harness changed quoted prompts or denied ordinary shell checks | Stdin plus actual wire-text authentication; 22 real-client prompt positives and one deliberate argv-escaping negative |
| OpenCode's implicit five-minute provider deadlines | Total/header/chunk deadlines disabled; real-client header and SSE probes finish after 310 quiet seconds, one request each, no retry |
| Qwen3.5 architecture override erased Qwen3.8 template defaults | Focused red proves missing xhigh instruction and historical reasoning; revision-owned ModelGenerationPolicy replaces the graph-builder hook |
| Artifact-copy failure deleted the unfinished app | ENOSPC negative control and explicit workspace lifetime; included in the focused Python suite |
| Live token traces omitted the stopping token | Streaming/nonstreaming, tokenizer/runner termination and disabled-trace controls; explicit terminal disposition without extra device observations |
| Adjacent literal parameter tags discarded a native write | Retained four-call generation versus three-call wire; reduced red, split-point/emoji/HTTP regressions and explicit preflight |
| Final file bytes could hide incorrect write/read arguments | Five negative controls, exact original-workspace binding and explicit preflight; offline retained-session revalidation |
| Completed shell tool could hide missing or failed workflow tests | Exact command/workspace/input join, native exit and nonempty unittest summary; focused negatives and explicit preflight |
| Peer archive eviction made a local disk hint discard the final RAM owner | Shared and independent writers, retained-victim race, exact section checks and metadata-only reuse; 53 focused tests and two explicit preflight entries |

The preceding **697 Unit + 736 preflight** transaction passes in 2,260.446 seconds,
including all 327 host, 147 CUDA, 173 ROCm and 89 exclusive entries. The fresh
driver interval passes without findings; preserve `stdin-mtp-prerequisites/` and
`stdin-mtp-prerequisites.driver-report.json`. Added entries are `OpenCodePromptTransport` and
`OpenCodeClientDeadlines`. All eighteen focused harness tests pass, with retained
red controls. The complete grouped-verifier selection passed 81/81 registrations.

Preceding-image native controls pass: 220 HTTP requests; default dynamic, fixed
1/3/15 and observe-3 policy cells; twenty MTP-off requests in each of single,
TP, PP and multi-device MoE; Qwen2.5 implicit/explicit card-default equivalence.
Every completed lifetime has clean graph, transfer, MTP, prefix, capacity and
driver checks. The default-dynamic control evaluates 53 depth windows per GPU.

The preceding app is **interrupted, not accepted**: six phases, 38 authored tests
and 66 tool executions completed without a client tool error. A 51,231-token
cache miss took 347.558 seconds; OpenCode retried after 302.582 seconds and the
first connection reset. Preserve `dynamic-opencode-webapp-qwen38-unicode-mtp/`
and `opencode-provider-timeout-defect.json`. The server retired normally;
each GPU completed 24,251 controller transactions and evaluated 1,422 depth
windows, retaining depth one with no update. This proves continued evaluation,
not a speculative speedup. The earlier primitive sweep remains interrupted at
six of seven passes after the quoted-input defect; active workspaces are saved.

The stdin/deadline-corrected runs are also **incomplete**. Four app phases and
19 authored tests completed; the primitive sweep's last intact observation is
28/29 passes. Its delimiter failure is generated text: all three raw-token
decodes match the wire arguments byte for byte. The 32 GiB disk option limits
active payload, while append history and compaction consumed about 83 GiB in
the tool server. Insufficient filesystem headroom caused ENOSPC in both clients'
artifact writers. Both servers exited zero, with clean recovered driver and
graph/transfer/MTP/prefix/capacity evidence. Preserve the failed run directories
and `post-enospc` reports. Reclaimed caches belong only to normally exited task
containers, with logs and inspection retained in `retired-cache-cleanup/`.

The fresh policy-aware renderer matches all eleven saved requests against the
independent HF renderer and tokenizer, including the default xhigh instruction.
Preserve `opencode-revision-prompt-observation/`; live server prompt-count
authentication remains required. The fresh transaction passes all 697 Unit and 738 preflight entries in
2,262.232 seconds, with no driver findings: `template-mtp-prerequisites/` and
`template-mtp-prerequisites.driver-report.json`. The native controller starts
both workloads only after those receipts pass; see `template-native-controller.json`
and `template-native-observation.json`. Fresh native completion remains required.
Commit and push after all checks and normal retirement pass. Paged KV/prefix
storage remains a strategic follow-up; no physical disk bound is inferred from
the active-payload budget.

Compiler-resource proofs cover gfx906 and shipping CUDA sm80/86/89/90, with no
off-chip spills. A CUPTI attachment caused NVIDIA driver findings and remains
failed; no further attachments were attempted. Six later unattached native
sampling observations pass their output and fresh driver checks. Their event
brackets include submission/transfer/scheduling costs; achieved occupancy,
isolated kernel throughput and model speedup remain unqualified. Traced model
timings are diagnostic. Detailed history and evidence paths remain in the
[MTP project plan](MTP_VLLM_STYLE_PROJECT_PLAN.md).

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm. Keep implementation
detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Dense TP2 follow-up (2026-10-04)

**A — ROCm exceeds the revised 1.7x prefill target; CUDA tuning continues.**
The last qualified global-FP16/dirty-workspace slice passes **693 Unit + 667
preflight** registrations, both complete architecture builds and **45/45** default
Release HTTP checks on each topology. ROCm defaults reach **431.498/46.141
tok/s**, or **1.739015x** prefill scaling. CUDA cap 448 reaches **1005.358/66.362**,
or **1.038109x**, and passes its own 45/45 HTTP cell; it is not promoted.
CUDA's fast/slow process variation remains unresolved. Automatic head sharding
and global FP16 sums are installed; dynamic depth defaults remain unchanged.

Named CUDA GPU stage intervals and the interactive HTML/JSON/CSV report pass
paired model/ownership checks. Skills document event overhead and opaque
conditional-body limits. Raw projection scaling is **1.832x**, excluding
recurrence, attention and other non-projection work. Main matrices are correctly
sharded and each chunk executes once per participant. The launch audit finds
96 repeated GDN quantizers per prompt and fifteen physical first-transaction
sidecar forwards per admission despite selected depth three. Removing only
duplicate quantizers preserves twelve token/MTP ledgers but shows no macro win.
The installed eight-channel NCCL policy remains preferred; new sealed-dispatch
and protocol/thread screens do not justify promotion.

**A — shifted-MTP communication correction is implemented, qualification pending.**
Independent request-prefix banks replace capacity-sized embedding traffic;
dense/MoE declaration and captured CUDA/ROCm payload/poison regressions are added.
The old 1360-check/HTTP receipts do not certify this new source. First-transaction
capacity preparation remains an economy defect. Read-only PCIe checks and
frozen-plan authentication find matching CUDA/ROCm x16/x8 paths and Gen3 x16 CPU
uplinks; ROCm GPU 2's endpoint reports x16 above an x8 switch link. Loaded CUDA
widths are stable and AER counters do not increase. Bus width alone does not
explain the scaling gap. See the
[dense TP2 report](../2026-10/QWEN38_27B_TP2_TUNING.md) for exact artifacts and
workload limits. These are native binary qualifications, not image certificates.

## Latest verification (2026-10-03)

**A — checkpoint correctness and corrected-runtime benchmarks are green;
automatic ownership defaults are being qualified.** Checkpoint
`9491bd370c27a50b0fcb1f9f0e170aaac81c0dcc` passed its
normal commit gate: **687/687 Unit** and **606/606 ProductionTestPreflight**.
Its frozen **24-cell native Release HTTP cohort passed 24/24**, with
**1,080/1,080 checks**, clean driver intervals and clean retirement. This is
native qualification, not a Docker/image certificate. The earlier intermittent
CPU maintenance-drain trace remains causally unresolved; the CPU hammer was
stopped at the user's request and must not be restarted by this handoff.

The same checkpoint completed all 24 production-default benchmark cells.
One formal high-water failure remained: single-CUDA Qwen3.8 dense prefill,
**614.26 vs 718.36 tok/s**. No high-water file was changed. Large apparent
CPU/hybrid drops were independently traced to this local Release tree's
compiler-private ROCm LLVM library directory shadowing GNU `libgomp.so.1`.
Same-binary GNU-runtime controls retained identical token streams and MTP
work: Qwen CPU2 **332.93/40.29**, Ornith CPU2 **411.73/44.48**, 122B ROCm2+CPU2
**256.08/23.69**, and 122B ROCm4+CPU2 **311.71/27.59** prefill/decode tok/s.
The local RPATH is now corrected; preload controls are diagnostic evidence,
not substitute release configuration. Native loader identity and shadow
negative controls now have explicit Unit/production-preflight entries.

The corrected-runtime 24-cell benchmark sweep passes the recorded ratchet gate
in **1,387.50 seconds**, with a clean driver interval and no high-water edits.
Qwen CPU2 measures **332.37/41.00**, Ornith CPU2 **409.97/45.78**, and the CUDA
dense prefill regression is resolved at **900.68/61.56** prefill/decode tok/s.
The retained-family fix passes **20/20 native lifetimes on each GPU backend**;
same-model CUDA controls retain identical tokens and MTP work, remove all
1,600 padded rows across five prefills, and retain four graph captures.

Gate/up-owned, down-column-sharded Qwen execution reproduces at CUDA2
**2,068.31/203.20** versus whole experts **1,275.66/133.16**; ROCm2 measures
**1,422.30/130.77** versus **1,132.32/103.84**. Tokens still match on all measured
requests. CUDA MTP work matches exactly; ROCm projection uses one fewer verifier
round per request. The earlier ROCm whole-expert decode baseline was 61.82;
its now-higher speed has exactly the same tokens and verifier work. Conversely,
Ornith Q4 ROCm2 whole-expert decode falls from 114.61 to 65.47 with unchanged
tokens, verifier work and zero movement. That same-cell variability requires
targeted remeasurement despite the formal ratchet's passing identity keys.
Do not describe the previous +107.9% ROCm projection gain as reproduced.

The requested topology-aware default is now staged: automatic intent selects
projection ownership only in one homogeneous native multi-GPU tier. CPU,
heterogeneous and multi-tier plans retain whole experts; explicit modes remain
immutable. Automatic-default native launches are qualified below; the renewed
Unit/preflight gates remain pending. This does not prove cross-tier projection
support or certify a Docker image.

The five automatic-default native benchmark probes subsequently pass, with no
compute override: Qwen CUDA2 **2,074.82/204.06**, Ornith Q4 CUDA2
**1,915.28/222.09**, Qwen ROCm2 **1,423.45/130.80**, Ornith Q4 ROCm2
**1,429.62/143.93**, and Ornith Q8 ROCm4 **1,207.82/98.75** prefill/decode
tok/s. The four-participant probe first exposed a graph-builder contract gap:
shared decode uses canonical rank-order arithmetic, while overlap admitted only
native sums. The repair shares the ordinary stage's fixed-rank fold and admitted
scratch through its existing event pair. Fifteen focused overlap tests pass,
including real CUDA2/ROCm2/ROCm4 replay and live prefixes. The ROCm4 benchmark
retains all three control token streams and MTP work counts byte-for-byte.
The affected eleven-cell HTTP cohort completes **10/11 green**. Its first
Qwen CUDA2 launch misses readiness during the gate build; the preserved red
result is followed by an idle **45/45** retry in **80.39 seconds**, with the
same 60-second readiness limit and unchanged runtime. All eleven affected
configurations are therefore individually green, including six dense CUDA/
ROCm single/TP/PP cells. The 32K single-ROCm cell certifies 30,205/32,768-token
admission, 2,048-token generation and clean retirement. This is a collect-all
cohort plus a focused retry, not a rewritten all-green aggregate receipt.
Canonical source coverage additionally derives Ornith Q4 CUDA2/ROCm2 and Q8
ROCm4 projection variants beside their whole-expert controls. Fresh canonical
discovery confirms **27 HTTP cells**; all three additive cells pass **135/135
checks**, with clean drivers and retirement. Their five-entry focused contract
gate passes. The source-derived three-cell canonical benchmark projection also
passes in **153.98 seconds**: Ornith Q4 CUDA2 **1,915.07/222.07**, Q4 ROCm2
**1,427.09/143.90**, and Q8 ROCm4 **1,205.46/99.11** prefill/decode tok/s.
A retrospective extension of the preceding HTTP driver cursor contains no
new kernel records, not a separately armed benchmark-driver certificate.
Final aggregate gates and image-bound qualification remain pending. See
`docs/v2/projects/2026-10/MOE_AUTOMATIC_COMPUTE_DEFAULT.md` for the event map.

The first final normal hook completes **688/688 Unit** and **611/615 preflight**,
with clean driver evidence. The commit remains blocked. Its only reds are the
CUDA/ROCm pipeline-MTP predecessor byte proof and CUDA/ROCm MPI bounded-cost
preparation. The audit identifies a request-device hint incorrectly controlling
a physical-owner event edge, plus nonintegral projection proposals and a stale
zero-communication expectation for replicated dense decode. The staged repair
uses the participant's canonical owner and the arena's shared output-partition
invariant. All nine focused Unit/integration entries pass in **80.32 seconds**,
including both complete pipeline-MTP and real-MPI sampling suites. The explicit
predecessor-ordering regression passes **20/20 CTest executions on CUDA and
20/20 on ROCm** in **54.85 seconds**, with clean authenticated driver evidence.
Each execution contains twenty held-predecessor byte checks. The three affected
canonical pipeline HTTP lifetimes pass **135/135 checks** in **533.86 seconds**:
CUDA PP, mixed CUDA/ROCm TP+PP and ROCm PP, with clean driver intervals and full
memory retirement. The fresh complete normal hook and image-bound PR gate are
still required before publication.

Safe native graph/host-boundary observers localized the remaining CUDA dense
gap: the 512-live-token workload executed **832 physical rows**, because a
64-row nonterminal remainder reused a 384-row graph. The working-tree fix
preserves the admitted request bank and device KV cursor while sealing each
chunk's exact retained materializer **and shifted-MTP** graph identity. Its
stronger native regression exercises device inputs, two retained widths,
repeated request resets and rejection of incomplete/foreign ownership before
any execution. Complete incoming native event ordering is also required; the
earlier unjoined trial's Xid 43 is retained as red evidence rather than hidden.
Focused regression, 20-run backend loops and corrected unprofiled timing pass.
Source/runtime changes after the checkpoint do not inherit its full-cohort pass.

Evidence lives under
`parity-results/qwen36-rocm2-prefill/native-live-extent/`:
`checkpoint-full-http.json`, `checkpoint-full-benchmarks.json`,
`checkpoint-cpu-gnu-openmp-ab.json`,
`checkpoint-gnu-runtime-affected-benchmarks.json`,
`checkpoint-gnu-hybrid-benchmarks.json`,
`checkpoint-corrected-full-benchmarks.json`,
`checkpoint-corrected-full-benchmarks-driver-report.json`,
`checkpoint-canonical-ornith-projection-http.json`,
`checkpoint-canonical-ornith-projection-benchmarks.json`,
`checkpoint-prefill-entry-ordering-stress20.log`, and
`dense-cuda-prefill-attribution/`. These local artifacts are ignored and must
not be staged as source or published corpus payloads.

## Earlier investigation record

2026-10-03: **R — aggregate exposes a CPU maintenance-drain failure;
performance still pending.** The frozen-runtime prerequisite receipt records
**687 Unit + 601 production preflight entries** passing. The six-GPU 122B dynamic-MTP cell passes
**20/20 fresh full HTTP lifetimes / 900 checks**, with clean driver windows
and zero retained VRAM growth. The unchanged Release completes its **frozen
21-cell HTTP manifest: twenty green, one red**, in 7,039.33 seconds. The final
single-ROCm Qwen3.8 dense cell at 32K context passes **45/45**, including
30,205/32,768-token admission.
Every green cell passes **45/45**, including all eight long-context checks,
shutdown, retirement and independently checked driver evidence. This includes
both selectable CUDA/ROCm `GateUpOwnedDownColumns` modes, the 122B GPU/CPU
topologies, dense CUDA TP/PP and mixed CUDA/ROCm TP+PP.

The first Qwen3.6 CPU2 cell passes all eight long-context checks but hits its
900-second watchdog during coordinator maintenance drain; both rank stacks
are retained. The exact retained frontier remains unresolved. No runtime,
harness or timeout changes were made during the collect-all aggregate. The
driver exits one with `correctness_passed: false`; diagnosis now narrows to the
exact CPU2 cell rather than repeating the aggregate.
The unchanged-runtime exact retries are **20/20 green**, each **45/45** in
669.75–681.00 seconds. A separate read-only replay of the saved HTTP, all eight
long-context, tool, automatic-selection and final driver validators also passes
20/20; its log is `cpu-drain-20-independent-evidence-validation.log` under the
native-live-extent result root. Every completed attempt exits
cleanly, returns VRAM 42→42 MiB and retains an independently checked complete,
passing driver report with zero new records or findings. The first ten have no
native debugger/probe attachment. Attempt eleven has a bounded passive probe
with no stopping-worker samples and no debugger attachment; it also retires
cleanly. These passes do not establish
the intermittent stall's cause or repair the original red aggregate.
The loop stops at the first failure. A diagnostic sidecar observes only a
retained shutdown after all eight long checks finish, authenticates original
process births before its delay, revalidates them before attachment, and
preserves bounded passive/stack snapshots. Its former 25-second threshold
proved too early for attempt eleven's healthy tail; the observer alone is
stopped, syntax-checked and restarted with a 60-second diagnostic delay.
No native/cell timeout changes. The canonical driver remains the
sole request, timeout and retirement owner; runtime, harness and timeouts are
unchanged. Per-attempt evidence is tabulated in the linked investigation.
Production-default benchmarks, projection-owned A/B results, full aggregate
pass and image certification remain unproven.
Test-only coverage edits now add matching Qwen3.6 whole-expert CUDA2/ROCm2
HTTP controls and the reported Ornith Q8 ROCm4 HTTP/benchmark selector. They
preserve existing mathematical identities. The rebuilt discovery exports
**24 HTTP cells**, adding exactly those three tags with no removed cells and
unchanged local HTTP/benchmark/runtime/model profiles for the original 21.
The small device-free coverage regression has an explicit preflight entry.
All four focused coverage, selector, host-lease-terminal and hammer preflight
entries pass in 2.02 seconds. The renewed **687/687 Unit** gate passes in
77.722 seconds; complete **604/604 preflight** passes in 1,842.281 seconds.
The canonical **1,291-entry** combined receipt passes in 1,921.011 seconds.
The fresh-prefix hammer starts only after canonical unchanged-build receipt
validation, reusing its one staged GGUF without copying any weights.
The new real HTTP/benchmark selectors remain unrun. The completed CPU cohort kept its
original binary/manifest, unaffected by these metadata edits. The next E2E
projection is expected to gain three cells, not reuse the old aggregate pass.
The existing HTTP hammer also has a staged fresh-lifetime option: finite
request/prefix cycles followed by normal harness retirement, repeated under
one model-staging lease until the first failure. Separate lifetime journals,
final driver validation and interruption/fail-first tests join its existing
Unit/preflight registrations. Its 25 device-free Python regressions now pass
in 0.543 seconds, without launching a model or touching the active cohort.
Real fresh-lifetime execution is now running: two nonce-qualified cold/full/
partial-prefix cycles, followed by normal retirement, then another fresh
server. Its first real lifetime is **40/40 green** in **252.008 seconds**,
with two fresh prefills, eight full restores, two partial restores, normal
retirement and complete clean driver evidence. An independent read-only audit
confirms all forty ordered result records and the next lifetime's distinct
nonce. Lifetime two is running; the observer makes no native attachment after
the first original rank pair retires before its threshold. This tooling is not
a native shutdown fix or a certificate.
The following read-only update confirms three fresh-prefix lifetimes at
40/40 requests each (252.008, 254.726 and 250.045 seconds), with lifetime four
live. The second lifetime commits all 89 started movement waves and completes
1,958,477,824 transfer payload bytes before clean retirement, so the diagnostic
is not passing with maintenance inactive. These finalized counters are passive
path evidence, not policy authority or a performance certificate. The already
green real-MPI preflight includes repeated worker lifetimes, pending proposal
acknowledgements and old-reader retirement. No missing terminal edge or native
repair is yet demonstrated by this additional audit.
A benchmark admission review also finds ROCm's `atoi` CSV helper discarding
bad entries or selecting defaults after malformed explicit input. Both vendor
harnesses now delegate to one strict device-free geometry parser; a small new
test shard joins the existing trainer Unit target and explicit
`V2_Integration_BenchmarkGeometrySelection` preflight entry. This is also
compiled and its focused preflight passes, with no measured result or inference
policy change. Its labels
declare host-only ownership: shared vendor callers do not make this pure
selector proof an accelerator workload.
A read-only host epoch audit confirms that both sparse-return transports
already release their dispatch lease at the declared final ordered return.
The existing device-free graph regression was Unit-only; its explicit
`V2_Integration_MoEOverlayHostDispatchLeaseTerminal` preflight registration
is now compiled and passes its focused entry. This closes an inventory gap,
not the original runtime stall. Together with the two new coverage/parser
entries, preflight discovery grows from 601 to 604; complete execution is green.

The next October 3 slice closes a narrower real-MPI reader-coverage gap. The
existing sparse-return proof checked arithmetic across sixteen placements;
exact final-reader release had only a single-participant Unit proof. The
two-rank proof now pins a real authority lease, including an empty root or
follower, and requires the native final return to clear it before retirement.
Its explicit `V2_Integration_MoEOverlayHostDispatchEpochLease_MPI` preflight
entry passes **20/20 fresh process repetitions in 24.98 seconds**. A test-only
negative control selecting Retain instead of Release fails on all sixteen
placements with one active reader; the restored source passes. The original
sparse-transport group also passes. No production/runtime change is made.
The gate build targets are current, and discovery is now **687 Unit + 605
preflight entries**. The earlier 1,291-entry receipt is correctly rejected as
stale after this build/inventory change; the complete new 1,292-entry gate is
not yet qualified. The still-unbounded prefix hammer independently reaches
**nine clean fresh lifetimes / 360 completed requests**, with lifetime ten
active and no native attachment. This coverage does not repair or explain the
original intermittent shutdown stall. GPU HTTP and uncontended performance
work still await retirement of that CPU diagnostic.
See the [native-runtime and prefix-maintenance evidence](../2026-10/2026-10-01-rocm-captured-packet-publication.md).

The reporter's exact Nail weights now pass fixed-3 and dynamic stochastic HTTP
matrices, plus the 950-record cold/exact/changed-prefix reproduction with genuine
cache hits. Missing learned predictors are rejected before allocation; GPU DRY
remains an explicit pre-inference rejection, not a generated-text certificate.
For issue #16, the latest matched native means remain **143.92 fixed / 128.64
dynamic tok/s** (10.61% gap). Source audit identifies dormant tiled-MoE launches
retained by the wide dynamic envelope; no new route policy or timing result is
claimed. The ROCm communication-discounted scaling target also remains open
(1.524x versus 1.6x). See the
[dated slice, source lifecycle and issue inventory](../2026-09/2026-09-27-qwen36-rocm2-prefill.md).

A read-only verifier geometry trace confirms that the same retained row owner
sizes input/output strides and preparation key space, while serving capture
chooses the maximum envelope for Dynamic. The existing preparation registry
admits bounded scalar bucket keys, but that is not a narrower complete serving
graph family or permission to change one stride. The staged identical-live-row
probe must establish the cost before any runtime/graph-family redesign.

A further read-only bridge audit confirms the generic eight-row crossover is
an **economic route policy**, not a fixed kernel row limit:
`launch_moe_grouped_gate_up_route_owned_geometry()` sizes its grid from
`admission.maximumSlotsFor(RouteOwned)`. Existing measured Q8_0 keys admit
M16/M32 through that same template. For untrained M16 capacity, however, M9–M16
still select expert tiling, so capture must retain both families. Changing one
endpoint alone cannot prune the tiled family. The existing
`v2_perf_moe_verifier_prefill` harness includes all-format and deep-row economy
cases and the new speedometer now builds in both Integration and Release.
No policy, depth bound, arena,
kernel or measurement has changed; a broader crossover needs an isolated
all-format/geometry/route-distribution economy proof before adoption.

A test-only `ROCm_AdaptiveVerifierCapacityTax` speedometer now compiles in
that existing performance target. It compares identical live routes/weights
at different retained capacities (by default four live rows in M4 and M16),
with reused/uniform expert profiles, all canonical quantized source formats
and optional explicit mixed down format. Seven captured-event samples and a
serial-byte oracle accompany each complete key. It is opt-in, remains outside
preflight. Its Release target now builds successfully in seven commands;
the active serving executable and core-library inode, size and modification
time remain unchanged. Loader inspection selects the same repaired HIP and
matched BLAS closure. A first GPU-only diagnostic now runs while the CPU hammer
continues. All four identical-live-row cases pass seven samples with **zero
byte mismatches and nonfinite values**. Capacity-4/16 medians are **209.664 /
232.378 microseconds** for reused experts and **221.201 / 253.318 microseconds**
for uniform routes: 10.8% and 14.5% extra cost. CPU contention makes these
timings provisional, not a production performance certificate. Separate exact
native graph traces authenticate **five versus sixteen dispatches per launch**,
eight launches each, with no captured copies or driver findings. Gate/up and
down dot durations are nearly unchanged; wider grouping and dormant tiled
dispatches account for this generic-entrypoint probe's extra work. A following
source-fidelity check finds the real grouped main verifier binds runtime-owned
grouping and its deferred route ledger, including single-GPU Qwen execution.
That planner has a 256-slot compact boundary, not this probe's 64-slot boundary.
The six additional generic grouping dispatches are therefore **not established
as production overhead**. Actual runtime-entrypoint or whole-model attribution
is required; changing the generic threshold is not yet a justified server fix.
The profiler changes queue interception,
so its durations are attribution only. No route policy, MTP bound, arena or
default changes. `rocm-capacity-iq2s-iq4xs-live4-analysis.json` retains the joined
inventory and unprofiled samples under the ignored result root. A read-only audit
of its first eleven completed lifetimes independently revalidates all **440
saved responses**, nonce-qualified requests, SSE/needle answers, exact-prefix
tokens, partial GDN/MTP restore evidence and complete clean driver bookends.
`cpu-fresh-prefix-independent-evidence-validation.log` retains the result;
it is diagnostic evidence, not a shutdown repair or an image certificate.

The saved **real-model** trace reconstruction now authenticates every device
budget/prepare/commit/ticket transition and the declared last-40 measured cohort.
It independently reproduces **56,483 fixed / 72,483 dynamic dispatches**,
identical 128-token outputs, 88 accepted / 15 rejected drafts and depth three.
Runtime grouping has **1,600 dispatches in both**. The entire 16,000-launch delta
is dormant routed/shared projection-family work: 6,400 tile directories, 3,200
tiled gate/up, 3,200 tiled down and 3,200 partial publishers. Their 68.805 ms
summed service is profiler attribution under the original binary/SDK, not a
current Core 10 speedup certificate. The other three actual Nail expert pairs
also pass nine component byte checks at 4/9/16 live rows, with zero mismatches
or nonfinite values and clean driver bookends. Reproducible attribution is saved
as `issue16-real-mtp-dispatch-attribution.json` under the ignored result root.

The October 3 probe-fidelity audit confirms that production routing publishes
`-1` expert IDs and zero weights for inactive verifier rows, matching the
probe's retained-capacity suffix. The existing device-owned group admission
already selects route-owned work for four live rows inside M16; M16 does not
force those rows to execute tiled dots. Both reachable families still have
captured nodes, so dormant launch/grouping overhead remains a measurement
question. This routed-only probe does not include shared-expert live-count
publication, router Q8 reuse, or the rest of the model transaction.

The transaction-consolidation audit now separates two changes: retaining one
complete homogeneous HIP body per selected branch, and narrowing the paired
verifier geometry. The existing ordered-timeline composer and strict
self-contained graph export supply the first contract; declared sparse
boundaries must keep their exact epoch/follower publication edges. The second
cannot be a row-count tweak: preparation, forward, state publication and
stochastic captures currently share the widest immutable bindings, and every
additional owner must appear in the canonical memory BOM. Neither change is
implemented or measured yet. Ten focused controller/preparation/outcome/depth/
maintenance entries pass on CUDA and ROCm in 81.02 seconds, with a complete
clean driver window (`mtp-transaction-functional.tDKNJT`). The earlier
runtime-grouping/live-row checks also pass both vendors and all 24 expert
formats. These are focused functional passes, not a refreshed full gate or
model certificate. The unbounded CPU hammer reaches 29 clean fresh lifetimes /
1,160 requests with lifetime 30 active; the frozen serving binary/core remain
unchanged and the original intermittent shutdown cause is still unresolved.

2026-09-27: **A — six-GPU 122B tuning.** Unchanged dynamic-MTP defaults reach
**351.49 prefill / 42.38 decode tok/s**; the new targets are 345.58 / 43.80.
An explicit sharded-head comparison exposed missing local-TP gathered-logits
capacity before inference. The shared schema/admission fix passes five owning
Unit/preflight groups. Model setup now passes that boundary, then rejects the
unimplemented participant-local GPU verifier over sharded heads. No GPU head
default changed; tuning stays on the mirrored path. The combined full gate is
not yet refreshed. See the
[six-GPU report](../../../../changelog/2026-09-27-six-gpu-expert-overlay-return-packing.md).

2026-09-26: **G — stochastic-default performance target reached.** Matched
Qwen3.6 MoE IQ3_S Release medians on the 512/256 workload are **278.33 CUDA /
146.82 ROCm / 44.94 dual-socket CPU tok/s**, above the old-default controls
of **254.93 / 86.68 / 41.34**. Both the 75% first pass and 100% second pass
are complete. Learned startup/adaptation remains authoritative, with bounds
1–15. Corrected CPU training proves 30,720 serial-matching MTP token IDs;
after-policy CPU checks add 6,144 and fresh GPU prose checks add 1,536.
Non-MTP benchmark controls now honor the same requested sampling law.

Optimized CUDA/HIP builds reject memory spills, including cached objects and
HIP device callees, while reporting proven register-only moves. Spill cleanup
preserves all-format byte equivalence and measured model speed; isolated CUDA
attention improves 14.4%. The live Ornith Q8_0 CPU2 service also exposed a prefix
archive defect: a gathered row was copied from token zero on every vocabulary
shard. A checked logical/physical slice now owns copy and diagnostic geometry;
the persisted cache ABI is v3. CPU/CUDA/ROCm TP1–8 regressions pass 20 runs each
(8,640 surface cases). The complete gate after that fix passes **673/673 Unit
and 419/419 preflight**, with no skips and no GPU driver-warning findings.
Both direct HTTP and OpenWebUI pass 20 exact cached replays each, non-greedy
stochastic MTP, separate streaming reasoning, and the 1–15 adaptive contract.

The matched service check additionally found idle HTTP keep-alive connections
holding the sole stable worker for five seconds. One complete response per
connection preserves the worker/OpenMP team and releases it promptly; ordinary
and SSE regressions pass 20 repetitions and join production preflight.
Normal pre-commit and image-bound PR gates cover the complete resulting tree.
Remaining independent limitations: ROCm dynamic is 76.8–87.2% of best fixed
depth on the earlier holdout set, and the earlier Nsight-associated NVIDIA
assertions remain unresolved. Neither is hidden by the old-default target win.
See the [current experiment](../2026-09/2026-09-26-stochastic-mtp-default.md).
The older CPU wrap-up below remains historical, not this goal's completion.

2026-09-25 wrap-up: **A — accepted as good enough; tuning stopped.** Both CPU
sockets sustain approximately **42 tok/s dynamic-MTP decode** and **350-class
prefill**. The requested 50 tok/s was not reached. The last candidate was
byte-exact but had no production benefit (41.43 versus 41.50 tok/s), so it was
removed rather than counted as a speedup. The restored executable/core are
byte-identical to the control. Final gates pass **670/670 Unit**, **103/103
CPU preflight** and **6/6 true-AVX2 focused checks**, with a complete clean
driver interval. No tests or tuning runs remain active. Existing cross-backend
limitations below remain unrecertified. See the
[CPU wrap-up evidence](../2026-09/2026-09-24-qwen36-dual-cpu-mtp-decode.md).

2026-09-25, current CPU follow-up: **A — approximately 42 tok/s dynamic decode;
50 tok/s remains open.** Setup-owned expert workspace reuse passed 670 Unit,
99 CPU-tagged preflight and true-AVX2 focused checks. Reusing the existing SIMD
greedy selector cuts an isolated 124,160-logit scan from 135 to 9 us, but its
paired whole-model means are 41.70 (control) and 41.80 (change), inside noise.
All twelve 256-token streams match; default depth remains 1–15, acceptance
87.05%, and movement remains active. Matched MTP-off prefill is 349.00 tok/s.
Final gates, including mixed-format expert proofs, pass 670/670 Unit and
103/103 CPU preflight; the driver interval has zero new records/findings.
A byte-exact compact-codebook lookup-table experiment had no meaningful
mixed-format FFN win and was removed.
Do not report the scan's microbenchmark gain as a decode speedup or claim the
unrefreshed full cross-backend gate. See the
[current CPU evidence](../2026-09/2026-09-24-qwen36-dual-cpu-mtp-decode.md).

2026-09-25, latest: **A — 350-class prefill retained; 50 tok/s decode open.**
Medium expert-input publication uses the active team's total work budget.
Clean default dynamic decode is **39.17/39.44 tok/s**, matched MTP-off prefill
**350.52 tok/s**, with unchanged output streams. Gates: **670/670 Unit**,
**96/96 CPU-tagged ProductionTestPreflight**, clean fresh driver logs, and
focused AVX2-only numerical tests. The full cross-backend gate is not refreshed;
the GPU-only findings below remain open. Wider movement budgets and starting
dynamic MTP at depth 3 fail to improve this workload, so defaults are unchanged.
The common fused-projection completion simplification passes all 17 affected
numerical groups and three true-AVX2 gates. Release decode is **39.33/39.59
tok/s**, versus a paired 38.88 control; matched prefill is **352.93 tok/s**,
with exact tokens. A finer ordinary-head GDN partition was byte-exact but
slower and is removed. Final gates pass **670/670 Unit** and **98/98 CPU
preflight**, with zero new driver records. Both completion ISA registrations
pass twenty consecutive processes; the 50 tok/s target remains open.

2026-09-25: **A/R — CPU prefill preserved; GPU driver warning under diagnosis.**
An exact compile-time half-scale reciprocal lookup and direct AVX2/AVX512 byte
packing reduce isolated expert input publication by another 28%/40%, without
changing activation precision, formats or arithmetic results. Whole-model
dynamic decode is **37.51/38.99/38.43 tok/s**: the spread is larger than the
small median improvement, so no new decode win is certified. All twelve
compared 256-token streams match, with the same draft acceptance and movement
counts. The matched MTP-off prefill is **351.84 tok/s**, preserving the earlier
350-class result. The rebuilt full Unit gate passes **669/669 in 76.17s**;
the full preflight passes **375/375 assertions in 1006.90s**, but emits fresh
AMDGPU IH-ring-overflow warnings. A driver-guarded split reproduces them in
the all-device source-free planning probes; individual GPU probes are clean
in three repeats. This is not a clean driver-health certificate. The earlier
mixed-GPU capture setup/cleanup failure below remains unexplained despite
forty passing isolated repeats and its passing latest full-gate rerun.
The 50 tok/s dynamic-decode goal remains unmet.

The subsequent shared-descriptor experiment was removed: neither its original
nor copy-free form improved the production-aligned expert timing. A smaller
fix removes **25 dormant-diagnostic C++ allocations per expert FFN**, verified
red-before/green-after across all 21 formats and three ISA regimes; six focused
preflight entries pass. Latest unprofiled Release results are **38.62 tok/s
dynamic decode** and **348.79 tok/s matched MTP-off prefill**, with unchanged
tokens. This preserves the roughly 350 tok/s prefill result; it is not a new
whole-model speedup. Rebuilt gates pass **669/669 Unit in 76.01s** and all ten
affected CPU integration entries plus their fixture in 57.49s; two additional
AVX2-only entries also pass. The full 377-entry cross-backend preflight is not
recertified by this focused refresh; the GPU issues above remain open.

2026-09-24: **A — CPU correctness gates green; decode target remains open.**
Removing an M=2 single-worker bypass improves fixed-depth-1 Release decode
from 17.78 to 26.13 tok/s with unchanged tokens. Default dynamic MTP remains
32.92 tok/s versus the initial 32.71; an explicit sharded terminal-head
experiment measures 35.72–36.57 tok/s. The matched workload has 512 prompt
and 256 generated tokens, real movement, and the default 1–15 depth range.
The active 50 tok/s target remains unmet. Q6 instruction scheduling is under
evaluation without changing precision or generated dispatch. All 668 Unit
tests and all 372 production-preflight tests pass for the worker/Q6 slice.
The CPU-only sharded-head default now measures 36.11/36.38 tok/s versus a
32.45 mirrored control (+11–12%), with identical tokens. Five focused groups
and the rebuilt 668-Unit/373-preflight gates pass. Its latest
333.79 tok/s prefill is comparable to the prior dynamic-MTP baseline of 335.49,
not the 350 tok/s MTP-off/16-output-token checkpoint. The exact recheck exposed
a non-MTP sampler deadlock: CPU sampling inferred a shard from its allocation
instead of consuming the graph's published full row. A two-rank focused
regression reproduces the wrong row, extra candidate collective, and misplaced
penalties; CPU sampler/penalty/participation consumers now share the existing
typed publication. The repaired focused regression passes 20/20 fresh MPI
runs, and the rebuilt Unit gate passes 668/668. Matched MTP-off prefills are
348.50/348.91 tok/s versus the committed 349.57/351.44; dynamic decode repeats
at 36.89 tok/s with identical tokens/acceptance and live movement. The full
post-fix gates pass: 668 Unit tests in 75.92 seconds and 374 production-preflight
tests in 1007.23 seconds. See the
[CPU decode investigation](../2026-09/2026-09-24-qwen36-dual-cpu-mtp-decode.md).
Before the register-rounding change below, the fixed-depth-3 control reaches
40.14 tok/s with identical tokens; dynamic at 36.89 is 91.9% of it. A separate
restricted CPU-stage profile places about
80% of transaction time in verification, with routed experts the largest
stage family. The next isolated fix replaces volatile scalar FP32 rounding
stores with opaque x86 register dependencies, retaining the exact Newton tree
and GPU arithmetic. Exhaustive half-scale, general FP32 and Q8-tail oracles
pass in AVX2 and AVX512 builds. Quantization micro-latency falls about 30–32%;
default dynamic decode repeats at **38.02/38.29 tok/s** with identical token
streams and acceptance/movement counts. Matched MTP-off prefill remains
**348.14 tok/s**. The rebuilt Unit gate passes **669/669 in 76.06 seconds**;
the full production preflight finishes 374/375 and has one unresolved ordinary
ROCm2→CUDA2 pipeline setup timeout (RCCL reports an invalid HIP launch
configuration before capture, followed by a cleanup stall). Its previous
full-gate run passed; causality is not established. No
depth policy, precision, weight format or topology change was made, and
50 tok/s remains open.

2026-09-15, 05:27 UTC: **A — cross-host speedup measured; long E2E pending.**
Real full-image Azure requests improve 5.35x/6.43x for 32/64 output tokens,
with identical request/response messages. Both rank files pass remote expert
and graph-completion checks; 1,678 empty replies are avoided. The native
655-Unit/186-preflight gate and ten focused installed groups per ISA pass.
The full remote E2E runner is active; the diagnostic is not an image certificate.

Earlier 2026-09-15: **A — cross-host empty-return elision locally verified.**
Typed numerical obligations remove empty return RPCs while exact graph receipts
retain follower lifetime ownership. The complete MPI regression suite passes
20 fresh processes; 22 observer tests distinguish real bytes, empty outcomes
and graph completion. Full rebuilt prerequisites and both immutable-image/Azure
measurement remain pending. No changed MTP depth, precision or numerical gate.

Earlier 2026-09-15: **A — cross-host completion prerequisite under verification.**
Root MPI return waits consumed 60.843 of 66.833 seconds in a bounded full-image
HTTP probe; at least 53.455 seconds belonged to provably empty exchanges.
The working tree now has explicit, exact follower graph-completion receipts
before transaction retirement. The focused regression demonstrated the old
premature-retirement behavior. Empty numerical returns have not yet been
removed, and no speedup or new image certification is claimed. See
[the measured lifecycle audit](../2026-09/cross-host-empty-route-lifecycle.md).

2026-09-14: **510/510 native generation cells individually green**: 175/175
serial controls plus 335/335 MTP comparisons. The final unseen batch passes all
49 cells without another runtime change or per-cell prerequisite run. The
auditor confirms zero unseen and zero unresolved failed cells. Acquisition is
complete but remains unapproved, mixed-revision native evidence, not Docker
certification. Coordinated orchestration rebuild and Unit/preflight are next;
automatic planning and both ISA image certificates remain in progress.

Previous 2026-09-13 checkpoint: all 175 native serial controls pass; **304/335 MTP cells pass**,
and 31 remain. Qwen36 dual-ROCm Dynamic/Ordinal depth 1 now passes its
original four 384-token HTTP requests in 68.998s with clean shutdown.
The continuation selector now retires the completed tail before the next
verifier; terminal tickets retain outstanding maintenance. Both backend
captured probes (20 replays per depth/terminal shape) and all 28 controller
units pass. The rebuilt canonical gate passes 651 Units and 170 preflight
tests. All twenty fresh-process repetitions pass; unseen-only acquisition has
resumed for the remaining 49 cells. Depths 2/3/15/adaptive pass in
70.048/69.147/107.390/90.373s, completing Dynamic/Ordinal. Static/Random
and Dynamic/Random also complete every depth. Ornith dual-ROCm Static/Ordinal
depths 1/2/3/15 pass; dynamic depth is running. No further runtime fix or
per-cell prerequisite run was needed. See
[the lifecycle audit](../2026-09/hosted-mtp-maintenance-continuation.md).

Earlier acquisition history:
Both Qwen122B CUDA1/CPU2 and CUDA2/CPU2 complete their 20-cell MTP matrices:
Static/Dynamic × Ordinal/Random × depths 1/2/3/15/adaptive. The newly completed
CUDA2/CPU2 Dynamic/Random group takes 210.810/204.967/214.114/382.661/233.881s.
Every original response is serial-token-exact; prefix, captured-graph, the
declared movement/no-movement contract and teardown checks pass. Adaptive
bounds remain 1–15 with 163/152 policy updates while both movement objectives
execute. CUDA1/CPU1 rank-local Static/Ordinal also passes all five policies in
180.666/195.358/207.304/436.957/229.373s, with all original exact streams,
prefix/capture proof, empty movement journals and clean teardown. Adaptive
bounds remain 1–15 with 162/153 updates. No additional runtime fix or prerequisite
rerun was needed. CUDA1/CPU1 Dynamic/Ordinal depths 1 and 2 also pass. Depth 2
previously completed all exact streams but failed strict shutdown-log validation.
Local stop discarded Active wave ownership; the consolidated admission-only
cancellation fix passes its focused regression, existing maintenance Unit suite,
and 20 focused preflight repetitions. Its exact HTTP retry passes all eight
checks in 207.942s. The next unseen depth-3 cell passes all original exact
streams and complete harness checks in 219.823s. Depth 15 then passes in
439.805s and adaptive depth in 238.194s, completing the Dynamic/Ordinal MTP
family. All original responses remain serial-exact; captured execution,
prefix, physical movement and clean teardown pass. Adaptive bounds remain
1–15 with real policy updates. Static/Random depths 1/2/3/15/adaptive then pass
in 186.485/198.318/208.200/442.751/233.111s. All original streams are exact and authoritative
movement journals remain empty; the full harness passes each cell.
Adaptive bounds remain 1–15 with 162/153 updates on the two workloads.
Dynamic/Random depths 1/2/3/15/adaptive then pass in
196.827/213.845/227.335/451.655/245.368s, with all original exact streams and
completed tier movement. The complete harness passes each cell. All 20
CUDA1/CPU1 MTP cells are individually green, matching CUDA1/CPU2 and CUDA2/CPU2.
ROCm1/CPU2 Static/Ordinal depths 1/2/3/15/adaptive then pass in
148.082/155.219/164.075/336.245/170.512s, with all original exact streams,
empty movement journals, prefix/capture proof and clean two-rank teardown.
Adaptive bounds remain 1–15 with 164/157 policy updates. The five-cell family
is individually green. Dynamic/Ordinal depths 1/2/3/15/adaptive then pass in
164.900/171.482/181.239/352.470/186.031s, with exact original streams, both
completed movement objectives and full harness checks. Adaptive bounds remain
1–15 with 164/157 updates. All ten Ordinal MTP cells are individually green;
Static/Random depths 1/2/3/15/adaptive then pass in
154.864/158.659/168.618/337.743/170.689s, with exact original streams,
empty movement journals and full harness checks. Adaptive bounds remain 1–15
with 164/157 updates. Dynamic/Random depths 1/2/3/15/adaptive then pass in
172.719/174.058/183.639/355.958/188.444s. All original streams are exact;
both movement objectives and full harness checks pass. Adaptive bounds remain
1–15 with 164/157 updates. All 20 ROCm1/CPU2 MTP cells are individually green,
and ROCm2/CPU2 Static/Ordinal depths 1/2/3/15/adaptive then pass in
176.736/180.290/186.327/312.773/194.906s. All original streams are exact,
movement journals remain empty, and every complete harness passes. Adaptive
bounds remain 1–15 with 177/157 updates. ROCm2/CPU2 Dynamic/Ordinal depth 1
then passes in 205.818s with all original exact streams, the full harness and
1,026/66/568 tier/participant/combined movement edges. Depths 2/3/15/adaptive
then pass in 211.033/216.909/341.155/222.380s, with all original exact streams,
both movement objectives and full harness checks. Adaptive retains bounds
1–15 with 177/157 updates. Static/Random depths 1/2 pass in 179.491/182.147s,
with all original exact responses, empty movement journals and full harness
checks. Depths 3/15/adaptive then pass in 187.117/311.856/194.399s with the
same complete proof. Adaptive bounds remain 1–15 with 177/157 updates;
Dynamic/Random depths 1/2 then pass in 209.427/212.771s, with all original
exact responses, complete harness checks and 1,136/28/620 and 1,124/34/596
tier/participant/combined edges. Depths 3/15/adaptive then pass in
218.342/344.114/228.956s, with all original exact responses, complete harness
checks and both movement objectives. Adaptive retains bounds 1–15 with
177/157 policy updates. All 20 ROCm2/CPU2 MTP cells are individually green;
ROCm3/CPU2 Static/Ordinal depth 1 passes in 187.570s with all original exact
streams, empty movement and complete harness/teardown checks. Depths 2/3 then
pass in 191.153/194.513s with the same complete proof. Depth 15 then passes
in 303.296s, retaining every exact stream and full harness checks. Adaptive
then passes in 204.985s, with bounds 1–15 and 164/166 policy updates.
The complete Static/Ordinal group is green; Dynamic/Ordinal depth 1 then
passes in 229.968s with all original exact responses, complete harness checks,
and 992/120/401 tier/participant/combined movement edges. Depths 2/3 then
pass in 229.984/233.690s with all exact streams, both movement objectives and
complete harness checks. Depths 15/adaptive then pass in 350.839/245.420s,
with all exact streams, both movement axes and complete harness checks.
Adaptive bounds remain 1–15 with 164/166 policy updates. All ten Ordinal
MTP cells are green. Static/Random depths 1/2 then pass in 198.714/196.421s
with all original exact responses, empty movement and complete harness checks.
Depths 3/15 then pass in 200.319/305.105s with all original exact responses,
empty movement and complete harness checks. Adaptive then passes in 213.251s
with bounds 1–15 and 164/166 updates, completing Static/Random.
Dynamic/Random depth 1 passes in 234.066s with every original exact stream,
full harness checks and 1,244/36/435 tier/participant/combined movement edges.
Depths 2/3 then pass in 236.122/239.801s with all original exact responses,
both movement axes and complete harness checks. Depths 15/adaptive then pass
in 350.789/251.770s with the same full proof. Adaptive bounds remain 1–15
with 164/166 updates. All 20 ROCm3/CPU2 MTP cells are individually green;
ROCm4/CPU2 Static/Ordinal depths 1/2/3/15/adaptive then pass in
200.682/198.210/199.563/282.448/212.993s with all original exact streams,
empty movement and complete harness checks. Adaptive retains bounds 1–15
and 153/148 updates. Dynamic/Ordinal depth 1 then passes in 262.293s with
all original exact streams, complete harness checks and 1,046/74/107
tier/participant/combined movement edges. Depths 2/3 then pass in
261.630/251.378s with all original exact streams, both movement objectives
and full harness checks. Depths 15/adaptive then pass in 342.316/267.864s
with all original exact streams, both movement objectives and full harness
checks. Adaptive retains bounds 1–15 with 153/148 updates. All ten Ordinal
MTP cells are green. Static/Random depths 1/2 then pass in 212.640/215.307s
with all original exact streams, empty movement and full harness checks.
Depths 3/15 then pass in 224.841/284.762s with all original exact streams,
empty movement and full harness checks. Adaptive then passes in 227.177s,
with every original response exact, no movement, bounds 1–15 and 153/148
updates. Static/Random is complete; Dynamic/Random depths 1/2 then pass in
277.379/265.607s with all original exact streams, both movement objectives
and complete harness checks. Depths 3/15 then pass in 269.783/344.945s
with all original exact streams, both movement objectives and full harness
checks. Adaptive then passes in 282.797s with all original exact streams,
both movement objectives and complete harness checks. Bounds remain 1–15
with 153/148 updates. All 20 ROCm4/CPU2 MTP cells are individually green;
ROCm1/CPU1 rank-local Static/Ordinal depths 1/2 then pass in 191.276/203.472s
with all original exact streams, empty movement and complete harness checks.
Depths 3/15 then pass in 220.130/472.126s with the same full proof;
adaptive then passes in 217.918s with all exact streams, no movement,
bounds 1–15 and 164/157 updates. Static/Ordinal is complete;
Dynamic/Ordinal depths 1/2 then pass in 206.421/224.080s with all original
exact streams, complete harness checks and 768/864 tier-residency edges.
Only tier migration is available on this topology. Depth 3 then passes in
237.466s with all original exact streams, 864 final tier edges and full
harness checks. Depths 15/adaptive then pass in 490.823/244.203s with all
original exact streams, complete harness checks and 1,638/768 tier edges.
Adaptive retains bounds 1–15 with 164/157 updates. All ten Ordinal MTP
cells are green. Static/Random depths 1/2/3 then pass in
189.673/205.602/222.891s with all original exact streams, empty movement
and complete harness checks. Depths 15/adaptive then pass in 473.716/220.650s
with all original exact streams, no movement and full harness checks.
Adaptive retains bounds 1–15 with 164/157 updates. Static/Random is complete.
Dynamic/Random depths 1/2/3 then pass in 209.247/222.772/239.261s with all
original exact streams, full harness checks and 576/768/768 tier edges.
Depths 15/adaptive then pass in 499.928/245.152s with all original exact
streams, full harness checks and 1,416/768 tier edges. Adaptive retains
bounds 1–15 with 164/157 updates. All twenty ROCm1/CPU1 MTP cells are green.
Qwen 3.6 35B IQ3S dual-CUDA Static/Ordinal depth 1 passes in 41.165s with
all original exact streams, no movement and full harness checks; depth 2
stalled after two exact responses. An unchanged retry passes three then stalls
on the fourth; read-only host stacks locate the terminal generation event wait.
The first process later crashed after a failed CUDA-GDB attach, so that crash
is not attributed to the original hang. This was the sole current red.
The NCCL native-WHILE/request-reset regression is added to preflight. Its expanded
all-reduce/reduce/broadcast version passes twenty process runs (400 request
resets) in 65.26s. Disposable markers locate different full-sidecar collective
positions on the GPUs despite matched generation/epoch state and captured
collective geometry; source graph submission/reuse remains under investigation.
Rebuilt prefill-default/parser
Unit suites pass too. The clean retry reaches its unchanged 600s watchdog.
Event tracing finds metadata consuming shifted-KV readiness on a setup stream,
leaving the actual mailbox writer unordered. All seven metadata paths now use
the existing const observation API. Symmetric held-reader regressions pass
twenty runs each. Rebuilt Release completes all four exact streams and clean
teardown; the direct diagnostic omitted the movement-policy environment and
cannot count as a ledger pass. The shared refresh now passes all 650 Unit and
168 preflight tests after updating the long-prefill mock to use production
chunk plans. Twenty canonical depth-2 HTTP runs then pass all eight checks and
eighty exact original responses (median cell time 38.648s). The saved-evidence
audit confirms no current red and 123 unseen. The unseen-only batch resumes
at depth 3, reusing the same receipt, and passes eighteen new cells to finish
all twenty Qwen 3.6 dual-CUDA MTP cases. All original responses and independent
harness checks pass; every Dynamic final journal has 108 device-owned
participant-placement edges, Static journals are empty, and adaptive bounds
remain 1–15. Ornith's entire twenty-cell dual-CUDA matrix then passes with all
original responses and lifecycle checks. Each Dynamic final journal has 108
participant-placement edges; Static remains empty and adaptive bounds remain
1–15. Qwen 3.6 single-CUDA then passes all five MTP policies in
22.348/22.246/22.948/38.648/25.300s with every original response and harness
check. No current failure remains; 80 cells are unseen and Qwen 3.8 dense
single-CUDA depth 1 is running. All previously completed coverage stands.

The read-only acquisition auditor rechecks full saved responses against original
controls, preserving mixed-source provenance without issuing approval. The
three newly approved Qwen2 Q4_0 HF allowances are declared and rebuilt;
phase-isolation Unit tests pass and all 18 runtime/generation declarations are
unchanged. The full refreshed gate passes 650 Unit and 167 preflight tests;
all three fresh HF retries pass with eight CSVs each in 2.900/3.817/3.448s
for CPU-Q8/CUDA-Q8/ROCm-TQ KV. Existing decode cosine-or-KL limits remain
unchanged. The remaining 185-cell HTTP batch resumes at ROCm3/CPU2 depth 1
after an interruption, reusing the fresh receipt without replaying completed
cells. Corpus approval and both image certificates remain pending.

An independent outer CI fix rejects escaping/noncanonical model paths before
admission. The 126-test pipeline suite and both registered pipeline/E2E Unit
tests pass; the ongoing runtime, inventory and control corpus are unchanged.

The CPU dense depth-15 timeout, stochastic expert-follower admission guard and
first-use MTP capture-evidence defect are closed. Both native first-submission
policies pass 20 repetitions per GPU backend. The shutdown fix passes the full
650-Unit/167-preflight gate, including the newly registered local-drain regression.
Unseen-only comparisons resume at depth 3 with that refreshed receipt.
Neither an approved corpus nor either Docker
image is certified. See the [current evidence and lifecycle audit](../2026-09/production-ci-generation-regression.md#latest-continuation--september-13-unseen-mtp-proof).

2026-09-12 lossless-grid follow-on: IQ3_S, IQ3_XXS, IQ2_XXS and IQ1_S now
preserve native grids/scales through CPU preparation and CUDA/ROCm movement.
Both backend/ISA sweeps reduce the failing format set from nine to five;
all four CPU sparse endpoint/ticket tests pass. Rebuilt Unit gate is
648/648 green (74.32 s), and 64 Release repack/streaming economy rows pass
with zero CUDA/ROCm spills. Full production preflight completes in 583.62 s:
149/154 pass; its stale CPU packing oracle is then corrected and passes on
focused rerun, leaving 150/154 individually green. The four remaining
backend/ISA registrations fail only dual-scale Q2_K/Q3_K/IQ2_S/IQ2_XS/IQ1_M.
A clean, uncontended Release rerun also passes all 64 economy rows. These
multi-scale formats and the separate 122B HTTP drift
remain unresolved. No model/image certificate advances. See the
[repair and remaining representation contract](../2026-09/generation-qwen122-overlay-control-coverage.md#lossless-single-scale-grid-repair-2026-09-12).

2026-09-12 focused follow-on: corrected three empty floating-CPU preflight
filters and made all CTest/GTest registrations fail on zero executed tests.
Removed the obsolete sole-disabled attention registration; all remaining 648
Unit tests pass before the subsequent kernel fix. A real CPU sparse endpoint
feeding captured CUDA/ROCm tickets exposed an M=1 quantization-policy omission;
the all-format fix and both native/forced-AVX2 endpoint sweeps pass. Release
rebuild succeeds; all 648 rebuilt Unit tests pass in 74.49 seconds. Extending
the transfer arithmetic proof to original resident GPU descriptors then
exposes lossy CPU INT8 normalization in nine Q2/Q3/IQ formats on both vendors.
Those new regressions are red in preflight and need lossless representation
work, not threshold changes. Q8_0 is not among them, and the current 122B HTTP
Dynamic repeatability failure remains unresolved: nine of the original 36
controls green, one red, 26 unseen. See the
[focused evidence and arithmetic map](../2026-09/generation-qwen122-overlay-control-coverage.md).

122B acquisition (2026-09-11): all eight CUDA1/CPU2 and CUDA2/CPU2 MTP-off
controls pass on their first attempt (Static/Dynamic, Ordinal/Random), with
32 complete 384-token HTTP responses and exact cross-policy/placement streams
within each topology. Dynamic commits both tier and participant movement;
Static remains movement-free. The existing 797-test receipt and four-shard
tmpfs cache were reused unchanged. CUDA1/CPU1 Static/Ordinal then passes tokens
and teardown but exposes missing rank-local ticket-boundary evidence in the
HTTP graph observer. The sealed parent/service inventory now feeds that proof;
126 evidence-policy unit tests and both GPU cached-graph preflight suites pass.
The refreshed 649 Unit / 148 preflight gate passes, and the exact HTTP retry
passes all eight checks in 202.094 s with unchanged tokens. The 27 untouched
controls resumed with that receipt, then stopped on CUDA1/CPU1 Dynamic/Ordinal:
the fresh 384-token stream matches Static, but the full-prefix repeat drifts
at index 291 and at index 198 in an unchanged reproduction. Graph, memory and
teardown checks pass. Nine original unseen 122B controls are individually
green; 26 remain unadmitted. Focused CUDA/ROCm lowering regressions prove a
missing local-expert producer edge at the single-GPU canonical fold. The fix
and all 649 Unit / 150 preflight checks pass, but the unchanged HTTP retry
still drifts at index 216: that dependency defect is not the resolved root
cause. An uncertified no-prefix probe also drifts (index 309), excluding
restore as a necessary trigger. A four-request repeat probe drifts on requests
2/3, then returns to the original 384-token stream on request 4 (admitted at
epoch 8). Transient movement/publication is the next focus; these ad-hoc probes
did not certify graceful teardown or export normal terminal PerfStats.
These remain unapproved
controls, not MTP/HF or Docker certificates. See the
[122B control coverage record](../2026-09/generation-qwen122-overlay-control-coverage.md).

Dense 27B follow-on (2026-09-11): four new CPU controls pass their four
384-token HTTP requests. The first CUDA LocalPP control exposed a non-monotonic
attention workspace peak at an intermediate captured prefill bucket. A shared
CUDA policy envelope now feeds stage binding and metadata admission; both GPU
backends bound it to admitted query rows rather than the full KV horizon.
Five focused registrations and the full 649 Unit / 148 preflight gate pass.
The failed CUDA cell is green on exact retry (96.892 s); the two unseen
CUDA/ROCm and dual-ROCm pipeline controls pass (131.015 / 132.462 s), reusing
the same receipt. All seven dense controls are individually green, with 28
successful 384-token requests. The remaining 36 unseen controls are all 122B
overlays; prior failures and MTP comparisons remain. No token corpus or image
is certified. The current receipt is `dense27b-workspace-prerequisites-02`.
See the [dense coverage and lifecycle audit](../2026-09/generation-dense27b-control-coverage.md).

Ornith acquisition (2026-09-11): all twelve unseen CPU2/CUDA2/ROCm2 overlay
MTP-off controls and the unseen single-ROCm control pass first try. All 52
requests reach 384 tokens; policy/placement variants have identical streams
within each overlay topology while retaining movement/no-movement checks.
No implementation change or repeated prerequisite run was needed. The shared
797-test receipt remains valid; 43 MTP-off controls remain unseen. These are
unapproved controls, not model/image certificates. See the
[Ornith coverage record](../2026-09/generation-ornith15-control-coverage.md).

Native GPU evidence follow-on (2026-09-11): fixed a final-empty-wave false
negative and excluded resident-only assignments from physical movement proof.
CUDA/ROCm lifecycle entries pass 20 repetitions each; the full 649 Unit +
148 preflight gate passes. The exact CUDA2 Dynamic/Ordinal control and six
paused unseen controls pass; all eight Qwen3.6-35B homogeneous two-GPU
MTP-off controls are individually green. There are now 56 unseen controls.
No baseline or Docker certificate is approved. See the
[native movement lifecycle audit](../2026-09/native-gpu-movement-generation-evidence.md).

Host-overlay follow-on (2026-09-11): executed routing work is separated from
MTP acceptance, removing the CPU accepted-histogram graph walks. Explicit phase
tags and retained PMA-bound invocation storage are installed; seven focused
registrations pass 20 repetitions (53.80 s), followed by all 649 Unit and
146 preflight tests (609.268 s). Release and matrices rebuilt; the preserved
HTTP control passes in 85.688 s with four 384-token requests and 125 completed
expert moves (previously zero). All six resumed unseen controls also pass,
including both device-owned CUDA/ROCm Dynamic placements. Four further unseen
Qwen3.6 MoE CPU NodeTP controls pass, with 520/510 Dynamic moves but no speedup
over Static. These ten new greens leave 64 unattempted MTP-off controls. This is an
unapproved MTP-off control, not an MTP or model/image certificate. See the
[producer lifecycle audit](../2026-09/generation-rocm-cpu-random-economy.md#live-host-producer-and-mtp-workacceptance-separation).

Latest GPU follow-on (2026-09-11): compressed-cache cold/partial restore drift
was reproduced on CUDA and traced to chunk-mean AQ8 bases on both GPU backends.
CUDA and ROCm now use the first input key in every phase; the redundant policy
enum is removed and legacy disk fingerprints are invalidated. Both Release
request pairs match all 384 tokens, and the full 647 Unit + 135 preflight gate
passes. The canonical CUDA Q4_0/Q8-KV four-request generation cell is green in
17.022 s. Its HF prefill Top-5 is still 4/5 (KL 0.00317625 passes), so numerical
certification remains red. The next unseen CUDA Q4_0/TQ generation cell stops
naturally at 76 tokens on its initial cold request and remains unproven at the
required 384-token horizon. Follow-up: the cache regression passes 20 unprofiled
runs on each GPU backend; seven of eight further unseen generation controls
pass in 14.6–43.8 s. ROCm Q8_0/Q8-KV ends naturally at 182 tokens. All six
affected GPU HF cells have now run: four pass, CUDA Q4_0/Q8-KV remains Top-5
red, and ROCm Q4_0/TQ is red on KL 0.007492 versus 0.005 and Top-5 4/5.
Each has eight validated CSVs; prerequisites were reused throughout. Native
339-tensor banks now attribute both GPU failures: norms/projections closely
match their quantized same-input equations, and independently reconstructed
attention has mean relative-L2 residual 8.91e-7 (CUDA) / 3.59e-6 (ROCm).
The actual cells select post-RoPE keys; the older pre-RoPE sensitivity diagnostic
is not their live-path explanation. Final-head isolation places drift upstream,
strongly supporting accumulated quantization error; both gates remain red.
The optional effective-K/V cold-capture manifest fix is now installed: the
cache describes prepared storage before capture, and reset preserves immutable
stage descriptors. The CUDA native BF16 and FP32/BF16 request-batch gaps
are now closed with typed native kernels, without conversion workspaces.
Both graph integration binaries pass 60/60 tests; the 43-case snapshot/native
math subset passes 20 repetitions on each backend. All 99 native CUDA
specializations have zero stack/local memory, with focused NCU evidence.
Integration and Release rebuilds pass. The refreshed shared gate passes
647 Unit + 135 preflight CTests (584.434 s). Its exact ROCm Q4_0/TQ effective-K/V
diagnostic first stopped before numerical comparison: full-context snapshot
banks were absent from the reference-only memory declaration. That accounting
fix is now installed and both builds pass. Fresh 647 Unit / 135 preflight tests
pass (77.20 / 504.89 s); a driver interruption required receipt recovery from
complete CTest evidence. The exact model diagnostic now reaches all eight CSVs
in 9.206 s, retaining the same KL/Top-5 red. Actual effective-cache snapshots
give a stronger independent attention residual: mean relative L2 4.30e-7,
worst 3.42e-6 across 24 layers. No thresholds were waived. Subsequent unchanged
cells reuse this shared gate; resumed coverage prioritizes unseen controls.
Two more unseen Qwen3 controls ran: ROCm/FP32-KV passes all four 384-token
requests in 41.851 s; CUDA/Q8_1-KV finishes its partial request at 372 tokens
and fails the minimum in 16.370 s. An independent cold request matches all
372 tokens exactly, excluding restore-induced drift in this reproduction.
The next unseen Qwen3 CUDA/ROCm FP16-KV controls end their partial requests at
301/260 tokens; independent cold servers match every token exactly, ruling out
restore-induced drift in those reproductions but not satisfying the 384 minimum.
No thresholds or precisions changed. See the
[GPU prefix-basis audit](../2026-09/production-ci-gpu-aq8-prefix-basis.md).

Latest follow-on (2026-09-11): CPU RoPE's history-dependent angles and ignored
implicit offsets are fixed; Q16 no longer has a four-row position buffer.
Release cold/partial generation matches 384/384 tokens. All 28 functional
RoPE tests pass 20 repetitions on actual AVX2 and AVX512 Release builds. The
RoPE full gate passes 647 Unit + 133 preflight tests. Q16 block-local encoding
and complete-reduction range safety are now installed: three focused checks
pass 20 repetitions on both Release ISAs. Q16 HF KL improves to 0.00246115
against 0.006, but prefill Top-5 remains 4/5, like the separate Q8 cell.
The all-layer audit explains 169 linears, 24 attention operations and 97
residual-linked norm/RoPE operations; accumulated approximation remains at the
Top-5 boundary. A reproduced fused-GEMM dump-width defect is fixed and passes
20 focused repeats. A second diagnostic admission defect (async work after the
dump budget was exhausted) is also fixed; all 25 dump tests pass 20 repetitions.
Final 647 Unit + 133 preflight checks pass (589.649 s including build). The
5.201 s post-fix cell reuses the receipt, proves 24 complete gate dumps and 48
byte-identical projection outputs without dump errors, but remains Top-5 red.
No threshold was relaxed; no aggregate, corpus or image certificate is claimed.
See the [position-identity audit](../2026-09/production-ci-cpu-rope-position-identity.md).

Current slice: CPU Q8/TQ caches now use native anchored keys, with one
first-token basis across cold prefill, grouped verification and prefix restore.
All-format native runtime/storage checks pass (14 tests), including 20 complete
repetitions and the AVX2 runtime route. Fresh **647 Unit / 132 production
preflight** tests pass (572.852 s). The original outlier regression improves
from cosine 0.915217 to 0.999970. Qwen2 CPU/Q8_1 HF KL falls from 0.613263 to
0.00138603; all five decode tokens and prefix checks pass. That cell is still
red solely on prefill Top-5 (4/5), now narrowed to a 0.001010 fifth/sixth gap.
Independent terminal Q8-operand arithmetic reproduces native logits to
relative L2 2.21e-7; upstream approximation and the gate boundary remain under
investigation. No tolerance has been waived. Release runtime economy and the
continuous-generation recheck remain before claiming this CPU slice certified.
Historical generation coverage is 34 individual cells; CPU evidence
must be refreshed after this arithmetic-source change. Docker certificates and
routine-generation CI cutover remain incomplete.

Earlier generation follow-up: **33 individually green cells**. The thirty
Qwen3.6/Qwen3.8 pilots are joined by focused Qwen2 CPU/CUDA/ROCm controls,
each with four 384-token requests, exact repeats, real prefix restores,
production graph evidence and clean shutdown (34.278/13.562/29.412 s).
Cold/restored comparisons ruled out cache corruption for the short journal
answers. The model now declares a field-guide workload; the unsuccessful
follow-up-turn prototype is removed. No EOS, precision or mathematical gate
changes. All 79 script-policy tests, C++ contracts and 510-cell discovery pass;
only the 30 Qwen2 request bodies change. The refreshed shared prerequisite
gate and canonical Qwen2 family run are in progress. These focused results
are not corpus or image certificates. See the
[workload audit](../2026-09/generation-qwen2-continuous-workload.md).

September 10: **509/510 historical individual numerical greens**, no unseen
cells; Qwen2 CPU Q16 Top-5 remains red. Ornith ROCm dynamic depth now passes
with an independently authenticated HF-only conditional suffix, preserving
original scores and unchanged gates (26.894s, ten CSVs, 614 byte-exact rows).
The new Release generation pilot is **12/12 green** (Qwen3.6 MoE, CUDA/ROCm,
Off/1/2/3/15/dynamic), four continuous 384-token requests each with exact seeded
Off-control tokens and real fresh/full/partial prefix outcomes. No approved
token corpus or certified Docker image yet. Common native byte-assertion
hardening passes 647 Unit / 128 preflight and all twelve Qwen3.6 single-device
numerical policies. Ornith's 614 native verifier rows are exact; its remaining
HF route discontinuity is handled by the installed bounded suffix proof.
Its adversarial C++/Python preflight suites pass twenty repetitions each;
fresh full 647 Unit / 128 preflight gates also pass.
The complete shared single-device family passes 24/24 (Qwen3.6 CPU/CUDA/ROCm
and Ornith ROCm), with 232 canonical CSVs and 12,280 finite byte-exact verifier
rows. Dense/multi-participant live proof and the full image gate remain pending.
Follow-up closes representative dense/multi-rank proof: Qwen3.8 CPU/CUDA/ROCm
passes 18/18, and 122B CUDA1/CPU2 MPI2 Static/Dynamic ordinal passes 12/12.
The symmetric 122B ROCm1/CPU2 twelve-cell slice also passes. Combined fresh
common-gate coverage is 66 distinct cells, 638 canonical CSVs
and 34,980 byte-exact checkpoint rows. Other overlay topologies and the full
image gate remain pending; historical mathematical coverage is still 509/510.
Generation admission hardening passes 74 policy tests and a fresh 647-test
Unit gate. The new CPU control's NUMA/bootstrap failure is fixed and its
preflight regression passes twenty repeats; fresh 647 Unit / 129 preflight
also pass. CPU Off passes all four 384-token requests (118.892s), but the first
depth-1 comparison exposes a seeded RNG mismatch: ordinary CPU decode advanced
a call-count RNG while MTP keyed draws by output position. Identical-logit
regressions reproduce eighteen wrong ordinary/batched draws. The shared
position-keyed fix passes the runner suite, twenty focused preflight repeats,
and fresh 647 Unit / 130 preflight gates. The regenerated unapproved CPU control
passes (120.586s), but the exact depth-1 retry now diverges at token 54 instead
of token 2. The residual is double application of CPU stochastic history
penalties, reproduced at the same token despite 22,784 exact model checkpoint
rows. The single-owner fix passes its model-free regression, twenty focused
preflight repeats, and the short model probe (26,344 exact rows). Fresh complete
647 Unit / 130 preflight gates pass; corrected Off (119.881s) and depth 1
(133.171s) pass all four 384-token requests with exact serial equality and real
prefix restores. The four remaining unseen CPU policies pass on that build:
depth 2 (134.204s), depth 3 (144.147s), depth 15 (340.522s), and dynamic
(175.802s). All six CPU modes are green; the Qwen3.6 CPU/CUDA/ROCm generation
pilot now has 18 distinct individual greens. Every CPU cell still misses the
60-second economy target. The broader generation matrix and both image
certificates remain pending. See the
[sampling audit](../2026-09/production-ci-cpu-seeded-sampling.md).
See [verifier audit](../2026-09/production-ci-mtp-verifier-evidence-audit.md).

Earlier entries below describe prior checkpoints, not current queue totals.

Local unseen-first parity is at **429/510**. The last build passed 645 Unit and
128 production preflight registrations. Six-GPU 122B Dynamic/Ordinal MTP off
and fixed depths 1/2/3/15 pass after bounded prepared-context restoration.
Dynamic depth now also passes in 402.843s with all nine CSVs, depth-15 serial
and acceptance witnesses, prefix restore, and clean retirement. Checkpoint
admission uses explicit transaction append bounds instead of retained graph
width; three focused tests pass twenty repeats each, and fresh full gates pass.
Static/Random MTP-off passes in 42.820s; unseen-first pass 17 continues.
No kernel, format, cache size, or numerical gate changes. Docker and the known
Qwen2 Q16 Top-5 red remain pending. See the
[append-admission audit](../2026-09/production-ci-mtp-checkpoint-append-admission.md).

September 9 proof 22 built/discovered both ISA images and passed AVX512's
645 Unit / 120 preflight gates before 122B CUDA2/CPU2 Static depth-1 checkpoint
admission failed. The single-request oracle fix passes that exact cell and the
complete six-policy CUDA2/CPU2 Static sequence (116.47s, 53 CSVs). Qwen36 two-CUDA
Static also passes all six (108.47s, 53 CSVs). Full Unit passes 645/645 (76.23s);
53 pipeline-policy tests also pass after closing a fine-tune metadata ignore gap.
ROCm2/CPU2 off/depth 1/depth 2 pass, but depth 3 fails recursive MoE tensor parity
at reference row three, both in sequence and alone. Tokens and draft acceptance
remain exact. The snapshot scale/publication audit is open; no gate relaxation,
new full pipeline, E2E/benchmark certificate, or published image is claimed.
See the [oracle audit](../2026-09/production-ci-mtp-oracle-authority.md).

September 9 Docker proof 21 built the AVX512 builder and Release runtime, but
failed before discovery: Docker attach returned empty successful device-probe
output. The completed-artifact fix passes 20 real checks per backend and the
exact original discovery step (13 canonical E2E cells). No numerical model cell
ran in proof 21. Full dual-ISA certification requires new source-frozen proof
22; neither image is certified. See the
[metadata authority audit](../2026-09/production-ci-device-metadata.md).

September 9 proof 20 completed seventeen campaigns after 645/645 Unit and
118/118 preflight, then stopped on Qwen36 IQ3_S two-ROCm Dynamic/Ordinal D1.
The exact failure reproduces alone: initial maintenance cadence splits the
checkpoint response. Correcting test admission restores all nine numerical/
prefix/path CSVs but exposes missing movement. A continuous budget-one trace
proves the MTP condition graph never advanced HIP's maintenance clock. Explicit
serial-versus-speculative condition ownership now passes that exact ROCm cell
(48.626s, nine CSVs), twenty-repeat focused gates, and 645/645 Unit (73.23s).
Bounded replica-cache admission now passes all-codebook/backend Units and
admits 21/25 CUDA replicas; ROCm depth 1/15 pass at the canonical 25-slot grant.
CUDA Static and Dynamic expose a separate same-prefix oracle mismatch before
MTP starts. The same-prefix fix passes six focused CUDA/ROCm cells (54 CSVs),
including deep/adaptive cases. Fresh prerequisites pass 645/645 Unit (74.42s)
and 120/120 preflight (469.25s). Both four-cell process-level teardown/re-entry
sequences pass (ROCm 190.311s, CUDA 126.699s, 72 more CSVs). The slice is ready
for source-frozen dual-ISA Docker proof 21; neither ISA image is certified. See the
[commit lifecycle audit](../2026-09/production-ci-mtp-maintenance-checkpoint.md).

September 9 follow-up: Docker proof 19 passed fourteen complete campaigns and
all six Static/Ordinal 122B cells, then failed Dynamic/MTP-off prefix restore.
The missing headroom branch is removed. Seven targeted runs now pass with
60 fresh CSVs, including three-cell CUDA and ROCm reuse/transition sequences.
The finished slice passes 645/645 Unit (72.80s), 118/118 production preflight
(453.87s), and twenty repetitions of focused lifecycle regressions. Docker
proof 20 is the next full dual-ISA run; neither image is certified or published.
See the [MTP-off prefix admission audit](../2026-09/production-ci-non-mtp-prefix-boundary.md).

September 9: short-sequence testing fixed retained setup-evidence loss and
snapshot capture sizing from the first shallow request instead of retained
capacity. Both full Static sequences pass (24 cells, 212 validated CSVs): CUDA
154.20 seconds, ROCm 151.85 seconds. The rebuilt Unit gate passes 645/645 (73.76
seconds), and focused snapshot regressions pass twenty repeats per backend.
The full production-preflight gate passes 118/118 (454.65 seconds); Docker proof
19 is the next source-frozen attempt for both ISAs. See the
[retained-runner audit](../2026-09/production-ci-retained-runner-evidence.md).

Container certification follow-up: the native-token oracle correction passes
seven isolated 122B CUDA1/CPU2 and ROCm1/CPU2 Static MTP cells (63 required
CSVs). The next Dynamic depth-1 cell exposed premature prefix/serial-proof
admission after movement. Both CUDA and ROCm focused cells now pass with
stationary settlement and the allocated parallel transfer fabric (153.318 /
202.259 seconds, nine CSVs each). This is not an end-to-end speedup claim.
The first Dynamic depth-15 follow-up exposed a bounded-cycle admission defect
before MTP verification. Its device-free multi-wave reproduction is fixed;
the authority and preflight gates pass 20 repetitions each. The exact CUDA
depth-15 retry passes in 176.951 seconds with nine required CSVs and clean
retirement, after 644/644 Unit tests passed. CUDA dynamic-depth, ROCm depth 15
and ROCm dynamic-depth also pass (203.743 / 211.162 / 215.225 seconds; 27 more
CSVs). CUDA Random depth 15 and ROCm Random dynamic-depth pass as well
(203.705 / 193.205 seconds): six deep checks, 54 required CSVs in total.
The complete production-preflight gate passes 118/118 in 456.21 seconds;
the prerequisites for the next source-frozen container run are green. See the
[oracle audit](../2026-09/production-ci-mtp-oracle-authority.md) and
[cycle admission audit](../2026-09/production-ci-overlay-cycle-admission.md).
Whole production/image certification remains incomplete.

September 8 hardware-defaults follow-on is verified: RTX3090 automatically
retains 0.30 and MI50 selects 0.45, based on complete continuation membership,
with explicit request overrides preserved. Fresh clean Release decode is
**68.091/41.103 tok/s CUDA/ROCm**, with all five token arrays per backend
matching the accepted receipts. **639 Unit + 115 preflight + twelve parity
cells + 106 CSV artifacts** pass in **600.305 seconds**. An auxiliary two-card
dry-run check exposed an existing planning-only retirement seal error; normal
inference retirement passes, and that separate issue remains open. See the
[hardware-defaults handoff](../2026-09/2026-09-08-mtp-hardware-defaults.md).

September 8 accepted dynamic-depth result: **CUDA 68.166 tok/s (97.39% of
best fixed)** and **ROCm 41.013 tok/s (92.77% of best tested fixed)**, with
capacity 15, initial depth 2, real depth updates and identical tokens across
all five measured requests. CUDA completed fixed depths 1–15; ROCm completed
1–9 before the user stopped the slower deeper sweep (10 interrupted; 11–15
not measured). Depth 2 wins both completed inventories at 69.991/44.208 tok/s.
At that earlier tuning checkpoint, ROCm required
`--mtp-depth-demote-zero-accept 0.45` and global defaults had not yet changed;
the automatic-profile follow-on above supersedes that requirement. Its affected
gate passed **638 Unit + 115
preflight + twelve parity cells + 106 CSV artifacts** in **613.462 seconds**.
This tuning goal is accepted, not a claim that the entire production or HTTP
E2E campaign is green. No benchmark remains running. Reproduction and evidence:
[September 8 handoff](../2026-09/2026-09-08-dynamic-mtp-device-row-range.md).

### Investigation history

September 8 dynamic-depth follow-up: refreshed fixed-2/dynamic-15 measurements
are **69.555/24.318 CUDA** and **43.724/12.684 ROCm tok/s**, with all output IDs
identical within each backend. The typed row contract now reaches raw CUDA
grouped kernels and ROCm single/fused/mixed-decoder kernels. Captured regressions
pass 21 CUDA / 63 ROCm format cases, wider grouped/float checks pass, and all
638 rebuilt Unit tests pass. CUDA's public verifier scope now owns its raw
arithmetic selector too. Both new registrations pass twenty repeats (1,680
format/launch cases); the rebuilt 115-registration preflight passes in 444.75s.
The subsequent ROCm mixed-decoder register-lifetime cleanup passes focused checks:
it retains eight scalar-to-vector spills, with no off-chip scratch/spills.
The row contract now reaches public tensor adapters and Qwen FFN, QKV/GDN,
attention-output and identity-layout LM-head stages. Captured tests pass 64
CUDA and 106 ROCm cases, including FP16/BF16/FP32 projection and SwiGLU.
The gates are rebuilt for this interface revision. The first Unit pass was
637/638: a competitive collective timing assertion ran during compilation.
Its unchanged isolated Perf registration now passes; Unit keeps full-sequence
arithmetic coverage. The canonical gate now passes **638 Unit, 115 preflight,
twelve Qwen3.8 CUDA/ROCm cells, and 106 CSV artifacts** in 596.888 seconds:
`/tmp/qwen38-device-rows-proof-v2.{json,log}`. End-to-end economy certification
remains pending. Clean Release fixed-2/dynamic-15 is now **69.842/56.839 CUDA**
and **45.319/34.135 ROCm tok/s**: dynamic improved 2.337x/2.691x but reaches only
81.38%/75.32% of fixed 2. All five repeats preserve output IDs. A controlled
same-decision capacity pair confirms a remaining verifier-width cost; an
explicit Perf-only FFN reuse probe is measuring CUDA's capacity-selected
register footprint. The occupancy-aware all-format CUDA exact refresh now
measures **69.991 fixed-2 / 68.166 dynamic-15 tok/s**, dynamic +19.93% and
97.39% of fixed 2, with identical tokens and controller counters. All 1,050
counted-row observations pass; the rebuilt affected gate passes **638 Unit +
115 preflight + twelve cells + 106 CSV artifacts** in 594.741 seconds:
`/tmp/qwen38-counted-policy-proof.{json,log}`. CUDA's full fixed-depth 1–15
inventory now passes: depth 2 is fastest, confirming dynamic reaches **97.39%
of the best fixed depth**. The latest ROCm fused-row candidate reaches 36.073
dynamic versus 44.277 fixed-2 tok/s (previously 34.135/45.319). All 72 fused
specializations are spill-free and 128 focused captured-row cases pass, including
all-format column tails/bias. The rebuilt affected gate passes **638 Unit + 115
preflight + twelve model cells + 106 CSV artifacts** in **613.462 seconds**:
`/tmp/qwen38-rocm-grid-v21-proof.{json,log}`. Default ROCm dynamic is
still below the performance target, and the ROCm fixed-depth inventory
and remaining capacity-cost tuning are open. Isolated probes identify excess
inactive fused-projection workgroups, not GDN recurrence, as a material cost.
A bounded device-side row grid removes most of that microbenchmark capacity tax;
the remaining work separates controller demotion cost from verifier overhead.
Three-repeat screening identifies the zero-accept demotion threshold: 45%
instead of 30% keeps dynamic near depth 2, with 113 rather than 139 verifier
passes/request and about 41.15 tok/s. Compilation overlapped the screen, so a
quiet confirmation was required before acceptance. It subsequently passed as
recorded above; the user ended the deeper ROCm inventory after depth 9. No
default changed. Activation quantization
and ragged/compacted row layouts are not claimed to be count-admitted. See the
[bounded implementation record](../2026-09/2026-09-08-dynamic-mtp-device-row-range.md).

September 8 WIP checkpoint: additive packed-prefill exact dispatch finishes its
clean bracket at **1186.524 prefill / 46.485 decode tok/s**, with unchanged
tokens and weight/workspace bytes. Pinned llama.cpp confirmation is
**1204.301 / 45.757**: decode leads, prefill still trails. All 147 format/shape
cells and 45 isolated zero-spill launch profiles pass. The affected gate passes
**638 Unit + 112 preflight + twelve model cells + 106 CSV artifacts** in
581.568 seconds: `/tmp/qwen38-staged-prefill-proof.{json,log}`.
This is not the entire production campaign. Dynamic's physical verifier-width fix and its per-backend >=90% target
remain pending. See the
[phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md).

September 8 floating-prefill follow-up: shared-operand projection dispatch
improves clean CUDA prefill by about 2.3% in a control/retest bracket, ending at
**1146.532 prefill / 46.495 decode tok/s**, with identical tokens and unchanged
weight/workspace bytes. A refreshed same-pinned llama.cpp run gives
**1208.976 / 45.787**: decode still leads, prefill still trails. All native
floating formats have byte/resource and boundary-economy evidence; the complete
affected gate passes **638 Unit + 112 preflight + twelve model cells + 106 CSVs**
in 576.680 seconds: `/tmp/qwen38-tiny-shared-proof.{json,log}`.
This does not change dynamic-depth execution; its maximum-width verifier
remains the pending target. See the September 8 section of the
[phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md).

September 8 parallel-attention checkpoint: deterministic mode now retains the
normal ordered KV-split policy on CUDA and ROCm, including removal of HIP's
device-side single-split override. Clean CUDA MTP-off decode is **46.576 tok/s**
(43.977 before), ahead of the pinned llama.cpp 45.689 comparison; prefill is
1126.644 and still trails. ROCm's refreshed MTP-off baseline is **30.804 tok/s**.
No weight/activation precision or attention workspace increase. Deterministic
and normal parallel serial/grouped outputs are byte-identical in 288 captured
configurations per vendor; old single-split vs new model tokens first differ
at index 231, so do not claim old/new output byte identity.

The consolidated ROCm request-cache tests pass 20 repeats and now join the
preflight label alongside ROCm grouped/context attention. The expanded gate
passes **638 Unit + 112 preflight + all twelve Qwen3.8 CUDA/ROCm cells**, with
106 CSV artifacts in 574.176 seconds:
`/tmp/qwen38-parallel-deterministic-proof.{json,log}`.
The full ROCm attention binary is 59/61: two legacy real-Qwen2 fixture setup
failures lack a PhysicalMemoryAuthority and remain explicitly open outside
this model-free preflight. Dynamic physical verifier width is **not fixed yet**.
Re-establish fixed-depth winners after the attention change before certifying
the dynamic >=90% objective; the measurements below predate that change.

September 8: fixed-depth 1/2/3 now measures **64.193/67.226/65.640 tok/s CUDA**
and **33.436/38.418/38.041 ROCm**, with identical tokens within each backend.
Depth 2 leads this neighborhood; the complete depth-through-15 search is pending.
The startup parser now refreshes the canonical kernel-policy snapshot after
`--deterministic`: a reproducing Unit passes twenty repeats and all 181 parser
tests pass. Corrected direct profiles match canonical output tokens. They expose
the next ordinary decode target: deterministic mode forces attention to one KV
split on both vendors. CUDA attention measures 1.956 ms/token in the matched
trace, not the earlier unmatched 0.428 ms. Full Unit/preflight/model validation
of startup publication passed: 638 Unit registrations, 109 preflight integrations,
and all twelve CUDA/ROCm model cells with 106 validated CSV artifacts in 491.887 s.
At that earlier checkpoint, neither attention parallelism nor dynamic
execution-width selection had changed.

**Starting after the green attention decode slice:** tune dynamic-depth MTP on
CUDA and ROCm to **at least 90% of each backend's best fixed-depth decode** for
the same Qwen3.8-27B / exact 512-token prompt. Initial fixed-2/dynamic-15
measurements are **67.226/23.950 tok/s CUDA** and **38.418/13.990 ROCm**:
dynamic delivers only 35.6% and 36.4% respectively. A diagnostic ceiling of 2
recovers 65.367 CUDA / 37.176 ROCm with unchanged tokens and adaptive counters.
This isolates a large capacity-dependent execution tax, not a finished fix;
the production solution must retain depth 15. September 8 normal-bootstrap
captured-event evidence localizes 99.6% of the ROCm capacity-dependent loop
increase to its verifier: 121 identical-count replays take 51.506 ms each at
physical M3 versus 144.917 ms at M16. Outputs and adaptive counters match.
Best-fixed-depth inventory, per-kernel attribution and implementation remain
pending; do not tune controller thresholds around this execution-width tax.
The [project plan](MTP_VLLM_STYLE_PROJECT_PLAN.md#dynamic-depth-economy-cuda-and-rocm)
defines the matched Release bracket, fixed-depth search through 15, unchanged
precision/memory constraints, controller-cost evidence and correctness gates.

September 7 23:40 UTC, attention decode spill cleanup: **638 Unit,
109 preflight, twelve CUDA/ROCm Qwen3.8 cells and 106 CSVs pass** in
489.473 seconds. The final linked FP16 decoder has zero stack/local bytes and
zero measured spills; Release model speed is unchanged within noise at
1123.878/43.977 tok/s. The ordinary full-generation binding and original
llama.cpp throughput target remain open. The maximum-width dynamic verifier
tail is the source-audit lead for the measured capacity tax; exact kernel
attribution and the production fix are still pending.

September 7 22:28 UTC, ordinary live-frontier publication: **638 Unit,
108 preflight and all twelve CUDA/ROCm Qwen3.8 cells pass**, with 106 validated
CSVs, in 827.206 seconds including the Unit rebuild. Both backends share the
response/position/next-condition transition and pass captured aliasing and
twenty-reset proofs; isolated profiles show zero scratch/spills. The ordinary
model loop is not yet connected and both off cells still report no complete
generation-loop certificate. Reuse the existing sampler/logical-state buffers
for the pending production binding. Last retained Release throughput remains
1117.762/43.930 tok/s against pinned llama.cpp's 1152.615/45.689; the goal stays
open. See the [current phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md)
for the fresh receipt, per-cell timings, scope limits and lifecycle map.

September 7, ordinary entry/terminal composition: CUDA can now place its
captured prefill sample before the first WHILE predicate. One immutable
admission and shared validator authenticate ordinary/MTP terminal accounting;
the seven pure controller tests and focused native-prologue proof pass. The
fresh selected gate is **635/635 Unit, 96/96 preflight, 6/6 CUDA Qwen3.8 cells
and 53 CSV artifacts**, in 379.455 seconds. One stale source-count assertion
in the first prerequisite run was replaced by explicit lowering boundaries and
the prologue-to-predicate edge. Ordinary model-loop wiring and a new throughput
measurement are not complete; the off cell still truthfully reports no
generation-loop certificate. See `qwen38-ordinary-contract-proof-v2` in the
tuning receipt for the durable run identity and per-cell timings.

September 7, ordinary controller foundation: a typed ordinary generation policy
now shares the existing 44-byte admission ABI and 46-word resident ledger.
All **635 Unit, 96 preflight and six CUDA Qwen3.8 cells pass**, with 53 validated
CSV artifacts, in 397.622 seconds. Captured CUDA/ROCm publication/reset tests
pass; isolated publication profiles show zero spills on both backends.
This does **not** yet replace the model's host-per-token loop or
change the last measured throughput. See the current tuning receipt below for
the exact first-sample, pending-condition and EOS lifecycle.

September 7, MTP-off follow-up: retained source confirms **1075.182 / 42.475
tok/s** prefill/after-prefill decode, unchanged output bytes and VRAM. The
exhaustive Q5 unpack primitive joins Unit; **634/634 Unit, 96/96 preflight,
6/6 CUDA Qwen3.8 cells and 53 CSV artifacts pass** in 392.645 seconds. Slower
CTA-local reduction and balanced-unpack prototypes were removed. Attribution
also exposed a structural gap: ordinary decode captures each forward but
still submits/samples through the host per token; complete device-owned
generation is wired only for MTP. Existing MTP-off math/capture certificates
do not prove that stricter full-generation contract. The
[tuning receipt](../2026-09/2026-09-06-cuda-qwen38-mtp-off-tuning.md)
maps the current/target lifecycle and records rejected experiments. The
llama.cpp target remains unmet; no precision, weight, workspace or generated
dispatch-policy change was used to improve the numbers.

September 6, MTP-off focus: the user has made non-speculative CUDA Qwen3.8
prefill/decode the active comparison target. A BK64 partition-boundary cursor
improves unprofiled Release prefill **957.946 -> 1065.861 tok/s (+11.27%)**;
after-prefill decode is unchanged at **42.426 tok/s**, versus llama.cpp
**1175.520 / 45.664 tok/s**. All 256 output IDs and physical workspace bytes
are unchanged. IQ1_M retains its original economical register schedule after
rejecting a spill-free but slower one-CTA variant. **Unit 633/633, preflight
96/96, and all six CUDA Qwen3.8 cells with 53 validated CSV artifacts pass**;
the selected aggregate completes in 380.84 seconds using cached model/reference
data. This is a green correctness slice, not a completed performance goal.
Next attribute Q5 projection instructions and ordered-reducer publication;
do not change arithmetic, precision, weights or workspace to close the gap.
See the
[MTP-off tuning receipt](../2026-09/2026-09-06-cuda-qwen38-mtp-off-tuning.md).

September 6, CUDA Qwen3.8 tuning: small floating projections now preserve the
fixed reduction with one block barrier for decode and no block barriers for
prefill. The first Release sample improves 512-token prefill **930.700 ->
957.651 tok/s**, while MTP3 decode is effectively unchanged at **63.701 tok/s**.
Final-source confirmation gives 948.547/63.507 tok/s; the prefill gain is 1.9–2.9%,
not a fixed best-case result. Outputs and workspace bytes are unchanged. All
1,076 isolated byte-oracle cases, Unit 633/633 and preflight 94/94 pass. Fresh
llama.cpp measures 981.901/70.600 tok/s. Its output diverges at token 77 and it
accepts more drafts, so the remaining decode gap needs transaction/acceptance
attribution as well as kernel timing. The external goal is still open. See the
[focused tuning receipt](../2026-09/2026-09-06-cuda-qwen38-tiny-projection-tuning.md)
for profiles, exact commands, rejected candidates and remaining work. The fresh
canonical CUDA parity matrix passes all six MTP cells and all 53 CSV artifacts;
MTP3 draft-head cosines are 0.999910/0.999922. A separate Release 512-prompt /
256-output serial run matches every MTP3 output token in all repetitions:
42.587 tok/s serial versus 63.507 with MTP3. No mathematical defect was exposed;
the short-prompt HF comparisons and longer token witness remain distinct proofs.
Earlier
campaign progress below is historical; the September certification dashboard
records thirteen individually green E2E cells with their repeat gate paused.

September 6, 13:40 UTC: **A, final-parent attachment; R, aggregate.** CUDA2/CPU2
fails setup because its one retained parent cannot use the former direct-capture
fork/join interface. The shared CUDA/HIP native-DAG attachment removes those
paired events and recording flags. Focused tests pass on both GPUs; 22 transfer,
87 engine and three CPU-only DAG tests pass. At 13:51 twenty fresh-process
stress rounds pass (520 test executions), Unit is 633/633, preflight is 93/93
and Release is rebuilt. The CPU-tier retry passes all behavior and clean
shutdown but times out during its 740281-record artifact handling. Offline
validation passes. Single-owner collection/validation preserves byte-identical
evidence and improves isolated processing from 18.929 s to 10.819 s; 106 focused
tests and Unit 633/633 pass. The fresh online retry passes all 43/43 checks and
740078 evidence records in 597.234 s, including clean logs/shutdown and zero
residual GPU memory. The old timeout remains red. Its 2.766-second runtime
margin is narrow; this is not a full repeat/economy certificate. Ornith 1.5
LocalTP 2xCUDA passes 43/43 checks in 109.319 s, including all eight long checks
and 34439 evidence records, with clean shutdown and zero residual VRAM.
Qwen3.6 MoE NodeTP 2xCPU passes 43/43 in 572.911 s, all eight long checks and
877663 evidence records, with clean shutdown and no GPU use. Ornith NodeTP
2xCPU also passes 43/43 in 572.858 s, all eight long checks and 838794 records,
with clean shutdown and no GPU use. Four of thirteen have fresh passing
receipts. The remaining-nine run's first cell, 122B ROCm2/CPU2, passes 43/43 in
552.792 s with all eight long checks, 700812 records, clean shutdown/logs and
full VRAM release. ROCm4/CPU2 then passes 43/43 in 512.371 s with 558140
records, and Qwen3.6 MoE CUDA1 passes 43/43 in 55.196 s with 7692 records.
Both retain all eight long checks, clean shutdown/logs and exact VRAM return.
Qwen3.8 dense CUDA1 then passes 43/43 in 158.479 s, with all eight long checks,
5757 records, clean shutdown/logs and exact VRAM return. Eight of thirteen are
fresh green; mixed 122B CUDA2/ROCm4 runs next.

After all thirteen individual cells are green, pause the twenty-repeat gate for
a fixed-depth-three Release comparison against then-current upstream llama.cpp:
Qwen3.6 MoE 35B and Qwen3.8 dense 27B on CUDA1 and ROCm1, identical inputs,
with separate prefill/decode wins required in every comparable cell.

Mixed 122B CUDA2/ROCm4 passes 43/43 in 389.070 s with all eight long checks,
107654 records, clean shutdown/logs, exact VRAM return and no recurrence of the
long transfer warnings. Nine of thirteen are fresh green; Ornith RCCL ROCm2
runs next.

Ornith RCCL ROCm2 passes 43/43 in 159.072 s with all eight long checks, 52878
records, clean shutdown/logs and exact VRAM return. Ten of thirteen are fresh
green; Qwen3.6 MoE ROCm1 runs next.

September 6, 13:03 UTC: **G, targeted mixed 122B E2E; R, full aggregate.** The bounded
transfer service and TP-worker capture fix pass Unit 633/633, preflight 93/93,
and 20 symmetric fresh-process repetitions. The latest retry passes readiness
and two needles, then fails at MTP depth 14: hosted verifier replay explicitly
omits the branch authority installed at capture. A device-free reproducer is
red-before. The engine now resolves the policy from the same runner on capture
and replay, retaining strict rejection of changed ownership. The 200-replay
regression, all 87 engine tests, both builds, Unit 633/633 and preflight 93/93
pass. The exact retry passes 43/43 checks and all eight long checks in 394.907 s,
with clean shutdown, zero residual VRAM and no warnings. It commits 170 moves
(36 promotions/36 demotions/98 same-priority), 3.209 GB; remote endpoint p95 is
410.561 ms versus the prior 26.870 s. Other cells, CPU-quantized progress,
service accounting and full-suite repeat gates remain open. No end-to-end
throughput speedup is claimed. The September handoff owns diagrams/receipts.

September 6: **R, full HTTP aggregate not certified.** The HTTP/SSE delimiter
fix and two event-ordering fixes pass Unit 633/633, preflight 92/92, and twenty
symmetric GPU process repetitions. Arithmetic now requires natural EOS, not a
correct final number inside a length-exhausted loop. MTP-on/off agree after the
event fixes. The remaining arithmetic-B loop also reproduces in the independent
CPU/FP32 reference on the exact GGUF and native forced prefix. Our model schemas
omitted Qwen's paragraph boundary before the forced stop phrase. Correcting
only those bytes changes the reference to `14` plus EOS. A shared model policy
and three-schema red-before regression are installed; Unit 633/633 and preflight
92/92 pass. Twenty full CUDA E2E repetitions pass 20/20, each 43/43 checks in
54.49–55.60 s. The fresh aggregate passes both CUDA single-device cells, then
mixed 122B CUDA2/ROCm4 times out preparing transaction-36 physical transfers
during near-boundary prefill. Earlier needles/long generation pass; missing
PerfStats and shutdown errors are consequences of MPI abort. Fatal-only
projection diagnostics are building; root cause is not yet established.
Detailed receipts and the lifecycle map live in the September E2E handoff.

Historical follow-up: the 2048-operation CUDA KV archive pressure regression passes
20/20; it does not reproduce the mixed-model stall. A topology-wide diagnostic
rerun instead exposes an earlier lost empty-command race: CUDA opens snapshot
17 and clears command 16 before ROCm acquires it. The shared CUDA/HIP and CPU
fix retains sealed commands until the existing next-snapshot fan-in and keeps
new phase intent separate. CPU regression is red-before/green-after; a retained
real-GPU completion/open pair passes 20 delayed-follower replays with each
vendor as authority. Full gates are rebuilding. No new E2E cell is certified;
the original archive stall and CUDA2/CPU2 economy timeout remain open. See the
September certification handoff for exact receipts and the lifecycle diagram.

Latest 2026-09-05: NCCL resumed-capture regression is fixed (20/20), installed
under the canonical dependency SONAME, and gated by Unit 632/632 and preflight
90/90. Paired Release collective replay ratio is 0.99907 fixed/control.
CUDA2+CPU2 122B now passes startup and shared-prefix requests but fails forced
SSE decode: placement floor 19 versus an acquired epoch-18 reader. KV-only MTP
now publishes KV readiness without acquiring expert residency. Typed ownership
and observer-only maintenance waits pass CUDA/HIP 20/20 each, Unit 632/632,
preflight 90/90. The E2E retry clears SSE and four needle/JSON checks, then hits
the 600-second watchdog in structured generation. No stale epoch through 127;
throughput and pending-DMA diagnostic attribution remain open. Not an E2E pass.
Command-local DMA age/throttle now passes Unit 633/633 and preflight 90/90;
warnings retain severity and expose their exact command age. CPU sampling shows
substantial OpenMP waiting and NativeVNNI projection work, not a proven tuning
win. 122B ROCm4+CPU2 passes all eight behavioral checks unchanged, including
2048-token generation and 7595/8192 context, but is not certified: file-size RAM
heuristics and shutdown/rank-export defects lose final evidence. Canonical
memory attestation, rank-qualified collection and zero-only clean shutdown are
implemented; 71 focused harness regressions, Unit 633/633 and preflight 90/90
pass. The exact ROCm4+CPU2 retry fails its first MTP answer after readiness:
catch-up terminal refresh consumes prefill lengths instead of verifier geometry.
The redundant refreshes are now removed: typed scratch retirement precedes the
existing captured accepted-state publication. The initial API also drops its
irrelevant row count. Two-stream captured regressions pass 20/20 per backend;
both builds, Unit 633/633 and preflight 90/90 pass. Exact ROCm4/CPU2 passes all
behavior and clean teardown (514.54 s). Its original receipt rejects two valid
evidence types: retained parents and mapped activation collectives. Corrected
validators pass 77 focused tests and revalidate all 550186 saved records.
The fresh end-to-end receipt is green in 523.24 s: all eight long checks,
39/39 harness assertions, 555186 validated all-rank records, clean shutdown,
and zero post-teardown VRAM delta. ROCm2/CPU2 also passes all eight long checks
and 39 assertions in 597.62 s, with 705832 validated records and clean teardown.
Its 2.38-second watchdog margin is fragile. Four historical passes still need
refresh, and the remaining failing cells are not certified.

Small-helper admission evidence: 128 retained 64-kernel graphs consume 80 MiB
cold / 0 MiB warm on CUDA and 256 MiB per lifetime on ROCm. Focused certificates
and both full graph suites pass. Typed owner classification and capture-shape
enforcement are now implemented: depth-15/request-one has 82 bounded helpers
and 25 general auxiliaries, with the same 107 total owners. CUDA admission
decreases 1640 MiB per runner; ROCm admission is unchanged. Both builds, four
focused graph suites, Unit 633/633 and preflight 90/90 pass. Dense Qwen3.8 CUDA
now passes full E2E in 151.16 s: 39 assertions, all eight long checks, 2048
generated tokens, clean shutdown and zero VRAM delta. Seven of nine cells have
successful receipts; earlier shared-change refreshes are still due. The
CUDA2/ROCm4 122B retry reproduces its prior prefix-progress stall in 282.98 s:
first needle passes, then ROCm prepared transaction 28 waits on CUDA at 27;
the CUDA request thread is blocked in KV archive launch. CUDA debugger attach
fails internally and terminates a separate diagnostic, providing no device
kernel evidence. No new runtime workaround was installed. Model-free
archive/controller queue-ordering reduction is next.

2026-09-05 E2E update: CPU2 Qwen3.6 MoE reaches server readiness after the
distributed-TP memory fix, but its first MTP request exposed a constant sparse
operation ID. A host-owner sequence shared by retained sidecar variants is
under validation; the production-runner regression reproduced step `0` on all
20 executions before the fix. Runner and real-MPI regressions now pass 20/20
repeats each, with Unit 632/632 and preflight 89/89. CPU E2E advances through
the first long needle but exposes a separate grouped-policy cache collision
(M=258 aliases M=2); its full-width identity fix passes 20 focused repeats,
CPU all-format grouped-verifier integration, and refreshed Unit 632/632.
Refreshed preflight passes 89/89. CPU2 E2E is now fully green: 40/40 checks,
all eight full-context proofs, 2,048 completion tokens, clean shutdown in
572.2 s. Economy remains close to the 600-second cell watchdog.
Mixed122B remains red in prefix KV archive submission.
122B CUDA2+CPU2 now clears CPU expert preparation after first-touch allocation
isolation and fault-resolving NUMA certification fixes. Large concurrent NUMA
and guard/accounting regressions pass 20 repeats; refreshed Unit is 632/632 and
preflight 89/89. The rejected prefill fragment contains a non-clonable CUDA
conditional. Direct recording into the eventual parent passes 20-repeat graph
tests, refreshed Unit 632/632, and preflight 89/89, but real E2E now fails during
the second fragment's NCCL rooted reduction. Repeated capture-to-graph sessions
keep the same CUDA capture ID; NCCL 2.28.9's ID-only strong-stream cache reuses
a stream that stopped capturing. Add this missing NCCL/direct-recording
intersection to the integration gate before completing the fix. No fresh
full-cell pass is claimed for this topology yet.
See [current evidence and lifecycle maps](../2026-09/2026-09-05-model-parity-e2e-certification.md).

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 97% | refresh d1 economy and stochastic matrix |
| LocalTP fully device-resident MTP | 99% | remote participants and economy matrix |
| ExpertParallel fully device-resident MTP | 97% | mirrored-head batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  regimes inside one immutable capture; attention never requests recapture.
- GPU KV append, TurboQuant, verifier publication, and collectives use
  persistent workspace and explicit events. Hot paths have no allocation,
  blocking sync, segmentation, or intermediate host observation.
- Prefix restore preserves graph addresses through producer events. GPU
  FFN/MTP APIs reject null publication streams. CPU and GPU MTP terminal-hidden
  mailboxes are capacity-complete arena owners whose addresses remain fixed for
  the runner lifetime.

## Canonical CUDA2/ROCm2 LLEP Target

`LLEP` is the ordinary large-prefill routed-row assignment axis, not a
shorthand for an entire MoE execution mode. The active tuning target is this
explicit tuple:

| Axis | Required policy |
|---|---|
| Dense/shared trunk | tensor parallel |
| Routed expert storage/compute | apportioned whole experts |
| Routed phase | uniform |
| Grouped verifier/decode assignment | static owner |
| Ordinary large-prefill assignment | least-loaded resident (LLEP) |
| Small-prefill regime | static owner below the explicit routed-row threshold |
| MTP terminal norm/head | mirrored full vocabulary |
| Durable residency maintenance | off |
| Hot expert replica cache | off |
| Transport | one fully captured NCCL/RCCL graph |

The production large-prefill boundary is currently `M * top_k >= 8192` routed
rows. Setting it to zero is reserved for focused transfer-path proof. This is a
declared work-regime switch: grouped verifier/decode remains static-owner and
must never inherit least-loaded prefill assignment. Dynamic whole-expert
residency maintenance and hot replicas are independent experiments and are not
part of the canonical LLEP economy row.

Canonical tests must prove the tuple through PerfStats: static-owner grouped
verifier calls, least-loaded current-batch prefill when the threshold is met,
mirrored terminal-head execution, graph-captured NCCL/RCCL, no segmented
execution, and no durable maintenance or hot-cache activity. Performance
tuning starts with stochastic fixed depth 3 so graph/communication economics
are isolated; dynamic depth is tuned only after that baseline is sound.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf active |
| LocalTP | ROCm2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | A/A | MoE long-context green; economy red |
| ExpertParallel | GPU+CPU | A/R | G/R | explicit Dynamic/LLEP greedy+prefix green |

## Correctness Proof

- The 2026-08-12 Qwen3.6 MoE SingleDevice production-checkpoint campaign is
  green on CPU, CUDA, and ROCm for fixed depths 1, 2, and 3 plus dynamic depth.
  The three `ALL_PRECISIONS` campaign targets take `138.14 s`, `104.14 s`, and
  `112.85 s` respectively from a warm authenticated reference pack: `355.13 s`
  (`5 min 55.13 s`) for the complete three-backend slice. A forced reference
  refresh followed by the same CUDA and ROCm runs takes `511.04 s`
  (`8 min 31.04 s`). The previous CUDA campaign took `1524.20 s`; retaining one
  graph-native serial oracle and one plan-certified immutable
  `ModelContext`/`PreparedWeightStore` per campaign removes repeated model
  loading while every depth still owns fresh request, arena, stream, graph,
  controller, and prefix state. CUDA and ROCm CSV evidence records full prefill
  and decode graph capture plus replay, zero segmentation, and prepared-weight
  reuse only after the first depth. Token traces prove actual depths 1/2/3;
  dynamic runs begin at depth 3, evaluate and update the device-owned policy,
  demote once, and finish at depth 2.
- The same campaign now authenticates `moe_router_snapshot_schema: 1` and MTP
  sidecar schema 5, so `MOE_ROUTER_OUTPUT` always means the complete
  post-softmax distribution. Probability checkpoints use direct row-wise
  symmetric KL rather than an erroneous second softmax. Every active MTP graph
  context also writes a `counterfactual_vs_mtp_router` row to the existing
  `mtp_sidecar_snapshot_breakdown.csv`: an independent double-accumulation
  projection loads the real GGUF gate and evaluates it on the exact live
  production `FFN_NORM` input. Across all three backends the worst causal
  relative L2 is below `3.3e-7` and worst symmetric KL below `6.9e-14`, proving
  the router kernel itself when recurrent upstream drift is larger. The
  device-free `V2_Unit_SnapshotCapture` gate rejects malformed distributions,
  proves worst-row behavior, and prevents reintroducing a second softmax.
- All-format grouped attention, TurboQuant, MoE routing, GDN, short-conv,
  stochastic target preparation, and draft publication sweeps are serial-row
  byte exact on their production backends and M ranges.
- The 2026-08-07 partial-terminal prefix/MTP failure was a CPU graph-lifetime
  defect. Scalar MTP first allocated a one-row terminal-hidden mailbox; later
  grouped publication replaced it with a wider tensor while cached CPU stages
  retained the freed address. Every backend now reserves
  `max(batch capacity, request_count * (depth + 1))` rows in one arena owner
  before graph construction, and runtime validation forbids allocation,
  rebinding, or reassignment. A focused scalar-to-four-row pointer-stability
  regression, typed CPU grouped-verifier routing tests, retained-producer role
  tests, and all six CPU/CUDA/ROCm ordinary-partial plus MTP-partial restore
  E2Es pass (`6/6`, `86.83 s` wall time). The KV lifecycle source sanitizer
  rejects reintroduction of the mutable-owner pattern. `GlobalOrchestrator`
  now preserves the typed grouped-verifier request across every execute and
  transfer step in GlobalTP/GlobalPP rank plans instead of erasing that role
  through generic `forward()` dispatch.
- The follow-on device-free gate exposed a deterministic MPI/OpenMP Q8 embedding
  repack defect: a ceil-divided 256-element grouped unpack crossed row boundaries
  whenever `d_model` contained fewer than eight Q8 blocks or had a block tail.
  Full-table and vocabulary-range repacks now share one implementation that uses
  grouped unpack only for complete eight-block groups and the bounded block API
  for every tail. Instrumented regressions prove short rows never enter grouped
  unpack while complete 256-element rows retain it. The canonical MPI-wrapped
  embedding suite passed 20 consecutive runs, the complete Integration tree
  rebuilt cleanly, and the unit/source gate passed `591/591` in `135.43 s`.
- CUDA MoE grouped verifier passed every native format at M=2..16 and M=31
  through the real router/expert path after the small-M grouping fusion.
- Release CUDA2/ROCm2 passed all eight Dynamic/LLEP cells and `166/166` checks
  through 2048 tokens with full capture and clean VRAM release.
- Release CUDA2 LLEP + RAM prefix + stochastic dynamic d4..15 passed `23/23`
  canonical server checks at context 4096 and 1024 output tokens.
  This includes deterministic prefix replay, forced movement, long recall,
  prefill replay, no segmented execution, clean shutdown, and PerfStats path
  assertions.
- The 2026-08-04 canonical grouped-verifier gate passed `89/89` explicit CPU,
  CUDA, CUDA2, ROCm, and ROCm2 cells. CUDA/ROCm KV publication is isolated by
  all 14 cache/source format pairs plus converted-read, logical-restore,
  adversarial, and TurboQuant lifecycle cells; each process uses the production
  graph-capture transaction and exact stream/event ownership.
- TurboQuant codebooks are uploaded once per cache/device on its construction
  stream and covered by the cache constructor's initialization fence. Launch
  wrappers cannot perform codebook upload, and no process-global ready flag can
  incorrectly alias initialization across devices. The complete Integration
  tree rebuilt cleanly and the device-free unit/source gate passed `585/585`.
- Expert-overlay policy is now represented as independent typed axes throughout
  config, graph lowering, fixtures, and E2E registration. The model-free
  canonical-tuple regression proves dense TP, apportioned/uniform routed work,
  static-owner grouped verification, least-loaded current-batch prefill,
  mirrored MTP head, and both durable maintenance and hot replicas Off. The
  complete Integration tree rebuilt all `1413/1413` edges and the device-free
  unit/source gate passed `585/585` on 2026-08-04.
- LocalTP routed-plus-shared publication now has one typed canonical transaction:
  every participant publishes router slots plus one rank-indexed shared bank,
  one rooted sum transports that payload, the root finalizes routes, shared
  banks, sigmoid gate, and residual in fixed serial order, and one broadcast
  publishes the terminal hidden rows. The previous routed reduce/broadcast plus
  independent shared allreduce/gate/combine transaction is absent from this
  policy. The production graph-lowering regression proves one reduce and one
  broadcast, exact producer dependencies, root-only finalization, and no old
  shared collective or epilogue nodes.
- The canonical publication kernels are byte exact against repeated production
  M=1 arithmetic for every `M=1..16,31`, full Qwen width 2048, and ragged/vector
  boundary widths `1,3,4,5,255,256,257,512,513`. Captured M=16 graphs replay at
  live M=11 without changing inactive rows. The complete CUDA and ROCm MoE
  grouped-verifier suites, including all native expert formats and real-path
  graph capture, passed in `189.83 s` and `109.47 s`, respectively. The full
  Integration tree then rebuilt all 971 affected targets and the complete
  device-free unit/source gate passed `585/585` in `143.32 s`. The strengthened
  source policy also forbids compute stages from inspecting raw tensor
  coherence after exact-stream publication; graph capture records that edge in
  its dependency ledger and replay owns the authority transition.
- The 2026-08-05 CurrentBatchLLEP long-context failure was a runtime-table
  identity defect, not a collective timeout. Ordinary prefill and durable
  static decode had aliased one placement table, allowing transient
  least-loaded assignment to leak into the next decode layer. Qwen3.5 MoE now
  declares typed `MainDecodeDurablePlacement`, `CurrentBatchLLEPPrefill`, and
  `MTPDepth` table roles; graph keys, prefix runtime state v4, bindings, and
  source/unit tests preserve those identities independently. CUDA2 and ROCm2
  CurrentBatchLLEP long-context cells passed after the change in `239.50 s` and
  `455.36 s`, respectively, with their canonical full-capture and PerfStats
  assertions enabled.
- A full CUDA FlashAttention fixture then exposed a second device-state defect:
  the cached-token parameter writer treated one shared FA2 parameter record as
  one logical query row, publishing `kv_len=1` for ordinary multirow prefill.
  CUDA and ROCm now distinguish shared prefill/M=1 records from row-local
  grouped-verifier records. Focused device-memory regressions cover initial and
  continuation prefill, active-row padded prefill, M=1 decode, and M=4 grouped
  verification. The complete CUDA and ROCm FlashAttention fixtures passed in
  `69.60 s` and `64.05 s`; CUDA captured cache growth/reset and ROCm all-format
  captured request-cache proofs remain byte exact. GPU cache views also require
  an explicit backend-qualified `DeviceId`, preventing a HIP allocation from
  being silently labeled as CUDA storage. The final Integration tree rebuilt
  all 784 affected targets and the device-free unit/source gate passed
  `585/585` in `135.90 s`.
- Long-context prefill now has an explicit `M=262145` totality gate on CPU,
  CUDA, and ROCm. The CPU cell uses Qwen2.5-0.5B geometry and one native Q8_1
  KV layer, processes exactly `64 * 4096 + 1` real rows, and proves native-byte
  first/final-row identity in `0.52 s`. Each GPU cell captures one 4096-row
  graph and replays it 64 times plus the one-row tail without per-chunk host
  slicing, transfer, allocation, or synchronization. A profiler-amplified
  final-tail miss exposed an unordered diagnostic D2H; the harness now records
  the terminal graph producer event and consumes it on the explicit observation
  stream before the sole terminal readback. CUDA and ROCm focused gates pass,
  including the exact `262145` device KV count. The CUDA materializer winner is
  32 threads at approximately `3.55 us`, 35 registers, and zero spills. The
  ROCm winner is 32 threads at `2.56 us` median and `2.734 us` pooled mean versus
  `2.72/2.890 us` for 256 threads; ISA evidence shows vectorized dwordx4
  loads/stores, 16 VGPRs, 52 logical SGPRs, and zero scratch, spills, LDS,
  barriers, or atomics.
- The CUDA NativeVNNI grouped WIDE/DIRECT path now publishes its only K
  partition directly from the weight-reuse CTA. Explicit RN multiply/add
  operations preserve the former store/load publication boundary byte for byte,
  while the obsolete publication kernel and partials-workspace dependency are
  gone. The canonical all-format grouped-verifier integration gate passed in
  `115.86 s`, covering native formats, FP32/FP16/BF16, LM-head M16, large-K
  KPAR, MoE projections, and fused SwiGLU against serial rows. On the
  Qwen3.6-35B IQ3_S terminal head (`248320 x 2048`, M4), Nsight reports one
  kernel instead of two, 48 registers/thread, zero local loads/stores, and an
  unchanged `464.0 us` counter-replay producer versus `464.5 us` before fusion;
  canonical latency remains approximately `414 us` and is therefore neutral.
- The shared release geometry catalog now exposes 12 unique MTP-specific
  matrices: `H x 2H` plus `248320 x H` for every distinct Qwen3.5/3.6 hidden
  width. The turnkey `qwen-mtp-head` profile sweeps all formats and M=1..16,31.
  A focused Qwen3.6-35B IQ3_S tournament proved all grouped M=2..16 rows byte
  exact and retained the installed R2/R4/R8 policy. The terminal M1 KPAR
  challenger was rejected despite a `1.22%` isolated win because changing the
  serial family would reintroduce grouped reduction overhead; the existing
  WIDE serial route remains the economical whole-transaction choice.

## CUDA LLEP Economy

- Fixed-d3 control: Qwen3.6-35B-A3B IQ3_S, stochastic sampling, fixed prompt,
  425 prefill + 256 decode tokens, three iterations after one warmup.
- Stable M=4 grouping removed four launches/layer and moved verifier compute
  nodes `1536 -> 1376` per device. Direct FP32 route publication then removed
  one conversion kernel plus one memcpy/layer: compute nodes are now `1336`,
  captured prefill graphs are 80 total nodes/device smaller, and the native hot
  transaction is `1403` kernels including 67 NCCL/control sidecars.
- Runtime expert publication now uses one deterministic two-level warp/wave
  scan for counts, offsets, and active ranks instead of the former quadratic
  per-expert scan and second publication phase. Across M=2..31 the fused plan is
  `1.14x..1.26x` faster on CUDA and `1.16x..1.72x` faster on ROCm; at M=4 it
  moved `12.16 -> 9.76 us` and `30.28 -> 19.43 us`, respectively.
- The canonical routed-plus-shared transport transaction is `1.075x..1.151x`
  faster than the former three-collective transaction for M=2..16 in the
  isolated CUDA2 NCCL harness. Its fixed-order finalizer moved from `12.86` to
  `9.82 us` on CUDA with a 256-column tile and from `19.84` to `15.36 us` on
  ROCm with a 128-column tile. CUDA uses 30 registers/thread, 1.02 KiB shared
  memory, and zero spills; ROCm uses 24 VGPR, 48 SGPR, 512 bytes LDS, and zero
  scratch/spills. Narrower CUDA and wider ROCm alternatives were measured and
  rejected.
- The matched Release CUDA2 fixed-d3 benchmark validates the complete graph
  win: prefill improved `583.34 -> 654.43 tok/s` (`1.122x`) and decode improved
  `172.99 -> 180.43 tok/s` (`1.043x`). Generated token IDs, generated text
  bytes, and `79.25%` stochastic acceptance are identical to the baseline;
  verifier transaction validation failures remain zero. Both participants
  captured one complete prefill graph (`6387/6347` nodes), replayed it without
  recapture or segmentation, and all decode contexts report full-graph replay
  with deferred event completion. The mirrored LocalTP PerfStats records show
  participant-local outcomes with no outcome collective and no host shadow;
  the only D2H boundary is terminal response materialization.
- The M=4 CUDA fused kernel uses 40 registers/thread and 4.23 KiB shared memory,
  has zero spills, and retains 100% theoretical occupancy. The ROCm wave64
  kernel uses 45 VGPR, 89 SGPR, and 4,232 bytes LDS with zero private segment
  and zero spills. The post-change Integration tree rebuilt cleanly, the
  canonical grouped-verifier gate passed `89/89` in 834.31 seconds, and the
  device-free unit/source gate passed `585/585` in 134.99 seconds.
- Isolated M=4 grouping is `5.99 us`: 38 registers/thread, 4.10 KiB shared,
  zero spills, and 100% theoretical per-SM
  occupancy. Achieved whole-GPU occupancy is intentionally low for this single
  dependency block; splitting it would restore launch/dependency overhead.
- Matched decode improved `156.97 -> 164.93 -> 165.92 tok/s`; acceptance stayed
  `61.57%` with zero validation failures. Prefill is `179.51 tok/s`; decode is
  `3.43%` below the historical `171.81 tok/s` promotion target.
- Direct CUDA router publication is 40 registers, zero spills, 100% theoretical
  and 83% achieved occupancy. ROCm is 18 VGPR with no scratch or spills.
- Cooperative CUDA Top-K 40 cut target/draft distribution to `0.238/0.214 ms`;
  it is spill-free and byte exact through M=16.
- Qwen `N=512,K=2048` grouped projection is byte exact across all 21 CUDA
  formats at M=2..31 and gains `1.69x..2.21x`; its eight-CTA geometry remains a target.
- CUDA GDN is byte exact through M=31; at `d_k=d_v=128` it reaches `17.73 us`,
  541 GB/s, and zero spills.
- This RTX 3090 pair has no peer access; mirrored verification avoids its
  measured `941 us/layer` rooted M=5 NCCL collective.
- The post-fix long-context parity harness remains unsuitable as an economy
  result: CUDA2 CurrentBatchLLEP takes `239.50 s` and ROCm2 takes `455.36 s`,
  versus `31.72 s` for the earlier CUDA2 dynamic-maintenance control. The ROCm
  trace attributes roughly `194 s` to CPU-bound model preparation/loading and
  includes one `131.6 s` mapped-download wait plus per-step snapshot downloads.
  The next profile must separate test-only parity observation from the Release
  production graph before attributing those waits to LLEP itself; neither cost
  is accepted as part of the target device-resident inference transaction.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

### Reproducible CUDA1 SingleDevice Reference

The active CUDA1 control uses the Release binary, a 434-token production-valid
Qwen chat prompt, stochastic fixed-depth-3 verification, one capture/warmup,
and three measured replays. MTP prefill population is inside the measured
prefill transaction; model load, graph construction, and warmup are outside it.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d cuda:0 --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-cuda1-qwen36-35b-mtp-d3.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed_chat.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --moe-residency-maintenance off
```

The 2026-08-07 pre-router-tuning baseline was `2271.66 tok/s` prefill
(`191.05 ms`) and `189.60 tok/s` decode. A captured, byte-exact tournament over
every production bucket from M=64 through M=4096 replaced the underfilled
Qwen-35B `64x64` FP32 router tile with a typed bucket policy. At the canonical
M=512 bucket, `32x32` reduced isolated replay from approximately `426 us` to
`267 us`; all candidate outputs matched the installed arithmetic byte for byte.
Two post-install Release runs measured `2336.31/2335.26 tok/s` prefill
(`185.76/185.85 ms`) and `189.52/189.68 tok/s` decode. Both retained exactly
`450/672` stochastic accepts (`66.96%`), zero transaction-validation failures,
and one complete 1611-node captured graph.

Nsight Compute confirms the physical improvement. The former `64x64` node used
40 registers, zero spills, 32 blocks, 16.67% achieved occupancy, and `466.53 us`.
The installed `32x32` node uses 39 registers, zero spills, 4.10 KiB shared
memory, 128 blocks, 25.24% achieved occupancy, and `280.29 us`; measured SM and
memory throughput both reach 42.25%. The next dominant target is grouped IMMA:
gate/up/SwiGLU plus down projection account for approximately `78.45 ms` of the
complete prefill replay, versus roughly `11.2 ms` for the tuned router.

The follow-on all-format grouped-IMMA tournament covered the three concrete
Qwen3.6 35B MoE gate/up and down codebook tuples over every production bucket:
63 cells, 24 physical candidates per cell, and 1,512 authenticated records.
Every candidate matched the serial arithmetic byte for byte. The installed
policy uses paired gate/up projection with 32-column tiles through the smaller
buckets, 64-column tiles where they win at larger M, and 32-column down tiles.
Nsight Compute reports 64 registers, zero spills, and `64.19%/65.42%` achieved
occupancy for the representative M=512/M=1024 winners. The resulting Release
control measures `2445.92 tok/s` prefill (`177.44 ms`) and `189.69 tok/s`
decode with unchanged `450/672` stochastic acceptance (`66.96%`), zero
transaction-validation failures, and one complete captured verifier graph.

The first full-context HTTP gate then exposed a separate graph-family workspace
defect at Q8 `M=256,N=512,K=2048`: exact dispatch is non-monotonic in M, but
planning had assumed the largest bucket represented every smaller bucket. The
M=4096 direct winner therefore declared zero canonical-reducer scratch even
though M=64/128/256 use the 16-partition public-M1 tree. Planning now computes
a typed envelope over every installed cell through the graph family's maximum,
caches it with the complete mutable policy identity, and fails during planning
if the envelope cannot be proven. A focused CUDA integration regression covers
all 16 execution codebooks and every captured prefill bucket; the Q8 case proves
the 32 MiB four-slot allocation and the exact-overlay cache-key transition.

The corrected Release live-server cell passed `19/19`: repeated prefill replay,
three-position and strict-JSON needle recall, 1,024-token stochastic generation,
cache reset, `3827/4096` near-boundary context, oversized rejection, clean exit,
and complete VRAM release. PerfStats captured 2,702 records with zero segmented
execution, 240 paired-IMMA exact-overlay calls, all 40 grouped-verifier layers,
436 depth-three verifier transactions over 1,744 rows, and eight real executions
of the formerly failing Q8 canonical bucket. The only D2H publications were the
14 terminal request results. Peak process VRAM was 21,710 MiB on the 24 GiB
RTX 3090.

The final linked Release binary repeated the same `19/19` full-tier result and
2,702-record PerfStats contract after the fail-fast planner change. The complete
Integration tree rebuilt successfully (`792/792` targets), and the device-free
unit gate passed `591/591` in 139.32 seconds.

The next CUDA1 prefill slice pipelined the causal GDN recurrence instead of
changing its arithmetic. Two 16-byte-vectorized `cp.async` stages overlap row
`t+1` Q/K publication with the exact row-`t` recurrence; the measured launch is
64 threads with a ten-block launch bound. At the production Qwen M=425 shape,
latency moved from `568.893 us` to `449.331 us` (`1.266x`). Nsight reports 96
registers/thread, zero spill requests, 41.67% theoretical and 25.94% achieved
occupancy, 90.91% L2 hit rate, and `8.39` warp cycles per issued instruction.
The D_K=64/D_K=128 M-totality, captured M=3, M=2/M=4 snapshot, runtime-M
publication, and unequal-request continuation tests all remain byte exact. The
matched Release model replay consequently moved `2749.13 -> 2806.76 tok/s`
prefill (`+2.10%`) while decode remained `190.36 tok/s`; stochastic acceptance
remained exactly `450/672` (`66.96%`) with zero transaction-validation failures.

CUDA FA2 has a shared capture-time physical K/V tile policy backed by a
complete 288-cell Release tournament: every released Qwen attention geometry,
TP=1/2/4/8 where the head split is legal, M=64/128/512, and KV horizons from 64
through 131072. `TILE_KV=64` won `0/288`; HD128/HD256 remain within `3.149%`
with the 16-row tile, while HD64 query groups of width three or six select 32.
The installed generic rule passed all 288 cells with `2.481%` p95 and `3.149%`
maximum regret. Its device-free totality gate covers 48/64/100/164 KiB shared
memory ceilings and fail-closed profiling overrides.

The follow-on slice now implements the real byte-exact K/V-context mode and
its fixed-order device merge. One immutable captured transaction contains the
direct query-sequence root and an IF-only context body; the live device-owned
K/V count publishes the CUDA conditional predicate, so neither mode selection
nor intermediate attention state crosses the host. The capture plan owns the
maximum persistent partial-output and `(m,l)` workspace envelope, and invalid
or incomplete geometry fails closed instead of selecting another launch path.

The certified device-adaptive tournament covers 720 model/TP/M/KV domains:
all listed Qwen 2.5 and Qwen 3.5/3.6 geometries, legal TP=1/2/4/8 splits,
HD=64/128/256, M=17/32/64/128, and live K/V lengths
256/512/1024/8192/131072. Mean regret is `0.574%`, p95 is `3.108%`, and maximum
regret is `5.511%`. The five domains above 5% are all Qwen2.5-0.5B, KV=256
launch-floor cases; the worst is `62.935 us` versus `59.648 us`, an absolute
`3.287 us` difference within the explicit `3.5 us` native-conditional budget.
An HD256 PV8 challenger was also measured and retired: at Qwen3.6-35B TP8,
M=64, KV=131072 it regressed context execution by approximately `77%` and
direct execution by approximately `33%` versus PV4.

Nsight Compute on the retained PV4 context kernel reports `4.51 ms`, 128
registers/thread, 43.78 KiB dynamic shared memory, zero local-memory spilling
requests, `24.86%` achieved versus `25%` theoretical occupancy, and 11.93
active warps/SM. Compute and memory-pipe throughput are both `32.91%`; the
remaining limiter is fixed-tree barrier latency rather than spills or an
underfilled grid. The policy unit gate passed `15/15`, and the CUDA attention
integration binary passed `54/54`, including adaptive replay, fixed direct and
context byte equality, every native K/V format, grouped M=2..16, TP boundaries,
and TurboQuant-adjacent cache paths.

The linked Release Qwen3.6-35B CUDA1 server cell then passed `20/20` with
dynamic stochastic MTP depth 1..15 and the full long-context tier. It proved
seeded stochastic prefix replay, repeated captured prefill replay, three needle
positions, strict multi-needle JSON, 1,024-token generation, cache reset,
3,827/4,096 near-boundary context, oversized rejection, clean exit, and full
VRAM release. Its 3,365 PerfStats records prove complete homogeneous CUDA graph
capture with no segmented execution and no forbidden intermediate D2H. The
run also exposed and fixed a harness-only asymmetry: CUDA terminal-ledger
validation had rejected valid mixed greedy/stochastic compact-outcome records,
while the ROCm validator already accepted both audited sources and required at
least one stochastic record. The focused policy regression now enforces that
same fail-closed rule for CUDA.

The post-FA2 fixed-depth-3 Release control measures `2785.83 tok/s` prefill and
`214.25 tok/s` decode with `73.18%` stochastic acceptance and zero transaction
validation failures. The matched no-MTP control is `2805.78 tok/s` prefill, so
population of the shifted MTP KV state costs only `0.71%`; main-model prefill,
not the MTP sidecar, owns the remaining throughput gap.

```bash
env LLAMINAR_E2E_LONG_CONTEXT=1 \
  LLAMINAR_E2E_LONG_CONTEXT_TIER=full \
  LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
  LLAMINAR_E2E_LONG_MAX_TOKENS=1024 \
  LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
  LLAMINAR_E2E_PERF_STATS=1 \
  LLAMINAR_E2E_PERF_STATS_GPU_STAGE_TIMING=1 \
  tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_release/llaminar2 \
  --suite '/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf|cuda:0|64|--moe-residency-maintenance off --mtp --mtp-draft-tokens 4 --mtp-min-draft-tokens 1 --mtp-initial-draft-tokens 4 --mtp-max-draft-tokens 15 --mtp-depth-policy dynamic --mtp-depth-window 4 --mtp-depth-min-samples 4 --mtp-depth-promote-windows 1 --mtp-verify-mode speculative-sampling|qwen36-moe-cuda1-dynamic-mtp-long-context|prefill-graph-probe,require-prefill-graph-capture,stochastic-mtp-probe,non-thinking-only'
```

### Reproducible ROCm1 SingleDevice Reference

The active ROCm SingleDevice tuning control uses the Qwen3.6-35B-A3B MoE
model, a byte-stable 434-token Qwen assistant-generation prompt, stochastic
sampling, and MTP depth 3. Llaminar measures three steady-state graph replays
after one warmup; model loading, arena construction, graph capture, and the
warmup are outside the timing sample. MTP prefill is enabled, so each measured
prefill includes population of both the main and MTP KV caches.

`qwen36_mtp_fixed_chat.txt` is the production-valid benchmark prompt. Its
2,511 bytes have SHA-256
`63d628982074c2785953dfde6f327e4f0146162a5c51396a48b1f59a239a97ae` and
contain the exact Qwen user/assistant wrapper generated from the model's GGUF
chat template. The unwrapped `qwen36_mtp_fixed.txt` remains useful as a raw
tokenizer/continuation diagnostic, but the model closes that incomplete turn
with `<|im_end|>` as its first greedy token. Acceptance and throughput from
that malformed framing are not promotion evidence.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d rocm:0 --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-rocm1-qwen36-35b-mtp-d3.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed_chat.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --moe-residency-maintenance off
```

The 2026-08-07 post-exact-GDN/grouped-verifier stochastic baseline is
`1301.15 tok/s` prefill and `120.39 tok/s` decode, with `72.27%` draft
acceptance and zero transaction-validation failures. Relative to the active
`1400/125 tok/s` targets, the remaining shortfall is `7.06%` for prefill and
`3.69%` for stochastic decode. On the same prompt at temperature zero,
Llaminar measured `1302.46 tok/s` prefill and `145.36 tok/s` decode with
`82.94%` acceptance.

The temperature-zero correctness matrix compares 256 generated token IDs from
serial and fixed-depth-3 MTP. Llaminar is identical for all 256 tokens on both
the production-valid prompt and the older raw diagnostic prompt. On the same
434-token production-valid input, llama.cpp MTP first diverges from its own
serial decode at generated token 147; it is therefore a useful performance
reference but not a byte-exact grouped-verifier oracle. The two engines' serial
lanes first differ at generated token 26, which is an ordinary cross-engine
kernel-math branch and is independent of Llaminar's internal MTP equivalence.

The focused Release ROCm1 fixed-depth-3 live-server gate passed `19/19`. It
proved repeated captured-prefill replay; beginning, middle, and end needle
recall; strict multi-needle JSON; a non-degenerate 1024-token completion; cache
reset; valid use of `3827/4096` context tokens; oversized-context rejection;
clean shutdown/VRAM release; and `8,405` PerfStats records. The direct grouped
verifier operation-equivalence matrix remains green on both CUDA and ROCm for
M=1..4, the first production transaction/publication, and the M=6
resident-sidecar device target.

The 2026-08-08 ROCm FA2 policy slice installs the same explicit
query-sequence/K/V-context capture-time choice used by CUDA. Its complete
2,040-domain Qwen catalog covers every released dense/MoE attention geometry,
legal TP=1/2/4/8 split, tuned context horizons through 128K, and the production
head-dimension envelope. Mean regret is `0.371%`, maximum regret is `2.663%`,
and no catalog domain exceeds 5%. The independent phase-grid tournament
selected 60 resident phase blocks and retained a `3.387%` maximum regret.
`rocprof`/ISA evidence for
the installed kernels reports zero scratch: HD128 direct and phase use 68
VGPRs, HD256 phase uses 80 VGPRs, and the ordered reducer uses 32 VGPRs.

Timing exploration remains capped at 128K, but correctness and capacity do
not. Device-free CUDA/ROCm policy regressions exhaust every positive capacity
through one million and prove extremal dispatch totality through `INT_MAX`.
Real captured 256K transactions then prove CUDA direct/context byte equality
for both compiled native K/V types (FP16 and FP32), and ROCm direct/context
equality for FP32, FP16, BF16, and Q8_1 K/V storage. This exposed a
non-monotonic arena-sizing defect:
the largest M selected direct query execution and declared no partial
workspace, while an intermediate M selected context execution and needed the
family maximum. The ROCm launch policy now computes the complete family
workspace envelope before capture; runtime rebinding or allocation is not used.

The linked Release Qwen3.6-35B ROCm1 server cell subsequently passed `19/19`
again at fixed MTP depth 3 and the full long-context tier, producing `3,842`
PerfStats records with clean shutdown and complete VRAM release. Every FA2
capture requested `geometry_selected`: M=256 selected the actual context graph
with 60 phase blocks and 1,024 reducer blocks, while M=1,536/2,048/4,096 selected
query-sequence execution. The server gate now rejects a legacy query-only ROCm
request policy or malformed/missing per-backend FA2 capture evidence.
The complete Integration target set rebuilt cleanly, and the final device-free
unit/source-policy checkpoint passed `593/593` after synchronizing the stale
CUDA MoE router boundary regression with its installed measured overlay.

The 2026-08-09 CPU FA2 slice replaced cache-percentage magic numbers with a
typed, empirically certified K/V tile policy. Code-generation ISA and runtime
dispatch ISA are independent policy axes: native AVX2 passed 81 domains at
`2.859%` p95 regret (`4.120%` maximum), native AVX-512 passed at `3.371%`
(`5.124%` maximum), and an AVX-512 build forced through AVX2 runtime dispatch
passed at `2.733%` (`10.079%` maximum). The mixed-profile maximum is one
Q16_1, HD128, M=1 domain; it remains explicit evidence rather than being
hidden by an alias to the native AVX2 policy. CPU certification runs must be
isolated: simultaneous socket-wide tournaments measurably perturb shared
power and memory behavior and are not valid policy evidence.

The production byte-totality gate independently covers 34 distinct
participant geometries induced by every listed Qwen dense/MoE geometry and
legal TP=1/2/4/8 split, all nine native K/V formats, decode M=1, grouped
M=2/4/8/15, and prefill M=32/128/257 over a 512-row prefix. Every one of the
2,448 domains executes all seven compiled physical tiles through the optimized
kernel, for 17,136 candidate executions, and is byte identical to the fixed
canonical arithmetic. The complete gate passed in `326.68 s`; the focused
policy, Q16, tournament-driver, and production-kernel checks pass in `1.46 s`.

### CPU Qwen3.6-35B MoE Release Matrix

The 2026-08-09 control uses the 434-token canonical chat prompt
(`sha256:63d628982074c2785953dfde6f327e4f0146162a5c51396a48b1f59a239a97ae`),
256 stochastic output tokens, seed 123, temperature 0.8, top-k 40, top-p 0.9,
one warmup, and three measured iterations. Single-socket rows use 28 physical
cores. Dual-socket rows use two MPI ranks pinned one per socket with 28 physical
cores each. PSS is the observed warm process-tree value, not model-file size.

| ISA | Topology | MTP | Prefill tok/s | Decode tok/s | Acceptance | MTP decode / baseline | PSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| AVX-512 | 1 socket | off | 110.84 | 15.69 | - | 1.000x | 42,239 |
| AVX-512 | 1 socket | fixed d3 | 108.77 | 10.71 | 79.91% | 0.683x | 43,680 |
| AVX-512 | 2 sockets | off | 32.07 | 20.09 | - | 1.000x | 44,904 |
| AVX-512 | 2 sockets | fixed d3 | 31.81 | 14.77 | 78.77% | 0.735x | 46,355 |
| AVX2 | 1 socket | off | 71.21 | 14.28 | - | 1.000x | 42,640 |
| AVX2 | 1 socket | fixed d3 | 69.90 | 9.08 | 74.09% | 0.636x | 44,084 |
| AVX2 | 2 sockets | off | 27.05 | 19.83 | - | 1.000x | 45,585 |
| AVX2 | 2 sockets | fixed d3 | 26.92 | 14.36 | 79.81% | 0.724x | 46,433 |

All eight cells completed with zero transaction rollbacks and zero transaction
validation failures. AVX-512 improves single-socket prefill by `1.557x` and
decode by `1.099x` over AVX2. Dual-socket baseline decode scales by `1.281x`
on AVX-512 and `1.389x` on AVX2, but dual-socket prefill regresses to only
`0.289x` and `0.380x` of one socket. MTP is also unequivocally uneconomical in
this control despite high proposal acceptance; its loss is verifier/draft
execution cost, not rejection or rollback pathology.

The first structural dual-prefill defect is explicit in the collective code.
Same-node CPU TP selects `ShmemSpinBackend`, whose native allreduce is capped at
8,192 elements. A 434-row by 2,048-hidden activation has 888,832 elements, so
large prefill allreduces leave the shared-memory implementation and enter its
MPI fallback, while decode-sized 2,048-element reductions remain native. This
matches the measured phase split but still requires per-stage timing before
attributing the entire deficit. The fix gate is an economical, total native
same-node allreduce for all positive payload sizes; retaining the fallback is
not an acceptable final architecture.

Both AVX-512 and AVX2 Release binaries then passed the full 4,096-context CPU2
MoE server tier `20/20`, including 2,048-token generation, boundary handling,
RAM prefix restore with MTP state, and CPU FA2 policy PerfStats. The complete
Integration target set rebuilt cleanly, and the device-free unit/source gate
passed `592/592` in `159.17 s`. Runtime ISA reporting now reads
`activeISALevel()` rather than claiming every CPU run is AVX-512, and the
canonical matrix wrapper admits CPU2 MoE through the UPI same-node policy
instead of rejecting the lane or forcing MPI-only transport.

The llama.cpp CLI command below intentionally receives the unwrapped source
file because `--conversation` applies the GGUF chat template itself. Its
effective 434-token prompt is byte-equivalent to Llaminar's checked-in
`qwen36_mtp_fixed_chat.txt`; passing the preformatted file here would wrap it
twice.

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-rocm/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

On current llama.cpp master `f9e832c10e94`, three runs measured prefill at
`408.7`, `405.3`, and `403.0 tok/s`, and generation at `78.4`, `77.7`, and
`78.4 tok/s`. The medians are therefore `405.3 tok/s` prefill and
`78.4 tok/s` generation.

### Reproducible Llaminar CUDA2 LLEP Reference

This is the canonical clean-throughput command for the explicit apportioned
LLEP policy tuple. It uses stochastic fixed depth 3 and the same prompt and
sampling controls as the llama.cpp reference below. The 425-token prompt is
below the production `M * top_k >= 8192` LLEP-prefill boundary, so this row
isolates grouped decode and two-device communication economy; use a longer
prompt to measure least-loaded current-batch expert movement itself.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-cuda2-llep-d3-clean.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --mtp-terminal-head-policy mirrored-full-vocabulary \
  --moe-release-raw-expert-weights \
  --moe-residency-maintenance off --moe-hot-expert-cache off \
  --moe-routed-expert-placement tiered-overlay \
  --moe-routed-expert-continuation-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-base-model-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-shared-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-residency static-by-id \
  --moe-continuation-dense-policy tensor-parallel \
  --moe-routed-expert-domain \
    'qwen36_moe_cuda_hot=cuda:0,cuda:1;scope=local;backend=nccl;routed_compute=apportioned;routed_phase=uniform;routed_decode_assignment=static-owner;routed_prefill_assignment=least-loaded-resident;owner=0' \
  --moe-routed-expert-tier \
    'hot@qwen36_moe_cuda_hot;priority=0;max-experts-per-layer=256;memory-mb=8192'
```

The 2026-08-05 post-correctness refresh measured `682.13 tok/s` prefill and
`124.07 tok/s` decode, with `52.89%` stochastic draft acceptance and zero
transaction-validation failures. Both devices captured and replayed one full
prefill graph (`5947/5907` nodes). This is the active economy baseline; the
older `180.43 tok/s` row had `79.25%` acceptance and must not be treated as
current until the acceptance regression is explained.

### Reproducible llama.cpp CUDA Reference

```bash
CUDA_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-cuda/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

The CUDA reference was rebuilt from current llama.cpp master `f9e832c10e94`
before measurement. Three serial runs produced
`1488.2/1411.4/1460.4 tok/s` prefill, for a `1460.4 tok/s` median. Three
fixed-depth-3 MTP runs produced `1250.5/1290.8/1304.8 tok/s`, for a
`1290.8 tok/s` median; llama.cpp therefore pays an `11.61%` MTP prefill tax on
this control. Llaminar is `1.92x` faster than the serial control and `2.16x`
faster than the MTP control while paying only a `0.71%` MTP prefill tax.
Promotion retains the stricter historical `171.81 tok/s` decode target and
requires at least `2245 tok/s` prefill.

### AIME25 HTTP Math And Multi-Turn Gate

`scripts/benchmarks/aime25_http_benchmark.py` owns a Release HTTP server and
runs the pinned 30-problem `math-ai/aime25` corpus as two requests per problem:
an initial solution followed by an independent review carrying the complete
message history. The runner authenticates the dataset, model, executable,
prompts, sampling policy, and server arguments; proves history with a nonce
canary; and durably checkpoints each first turn and reviewed result.

The first ROCm1 live probe passed server startup, model loading, and the
two-request history canary. Problem 0's first turn completed, but its review
exposed a production chunked-prefill admission defect: a 4096-row chunk followed
by a 156-row terminal tail was admitted into the fixed 4096 graph bucket, then
`ForwardExecutionEngine` incorrectly reapplied the 256-row raw-prompt minimum to
the already admitted tail. The staged fix makes positive `bucket_seq_len` an
explicit scheduler-admission contract. Focused device-free regressions, the
shared CUDA/ROCm 256K+1 graph-cache test, and both complete 12-case backend graph
cache suites are green. Each backend captures one 4096-row graph, replays the 64
full chunks, and admits the final one-row tail without eager execution or
recapture.

The follow-on audit found that the same minimum also made ordinary short GPU
prompts permanently eager. The staged structural fix redefines it as a minimum
*physical padded bucket* for raw prompts: short prompts are coalesced into that
bucket and remain graph-cache owned, exact smaller graphs remain capturable,
and scheduler-admitted tails retain their fixed transaction bucket. The old
parity-only graph-disable escape hatch is removed. The AIME runner now stops the
owned server before publishing success and authenticates PerfStats evidence for
prefill and decode capture plus replay, zero segmented/manual graph execution,
and zero intermediate D2H. The workflow is backend-neutral across `cpu:N`,
`cuda:N`, and `rocm:N`; GPU-only execution invariants are required for CUDA and
ROCm without imposing them on CPU. Its 15 device-free workflow regressions and
registered CTest gate are green. The full Release tree rebuilt cleanly; the
fresh 30-problem ROCm run is the remaining live gate.

The exact replay canary also certifies shifted MTP KV restore rather than only
ordinary prompt KV reuse: owned-server arguments force depth-three MTP plus RAM
prefix storage, the replay must preserve deterministic response bytes, and the
completed PerfStats ledger must contain a restore tagged
`includes_mtp_state=true`. The underlying partial-terminal restore path is now
green on CPU, CUDA, and ROCm as described in the correctness ledger above.

The CUDA dense-prefill corpus is complete across all 21 source formats:
`33,075/33,075` authenticated cells produced 25,200 exact runtime overlays from
Release scorer evidence. The installed generated include compiled in both the
complete Integration and Release trees. Mixed-format MoE overlay collection
follows the AIME live proof.

## Next Gates

1. Profile the complete canonical CUDA2 d3 graph with `nsys`, then use `ncu` on
   its hottest non-collective kernels. Attribute NCCL primitives, verifier
   compute, prefill expert movement, and tiny control launches separately;
   retain full graph capture and exact output bytes throughout.
2. Close the prefill gap from `654.43 tok/s` to at least llama.cpp's current
   `1122.5 tok/s` median, then pursue the `2245 tok/s` two-times target with
   longer prompts that expose LLEP's theoretical crossover.
3. Keep native geometry for large kernels and fuse adjacent tiny/control work.
   `V2_Perf_CUDAPersistentVerifierGeometry` found a one-kernel CTA proxy `1.25x`
   faster, but full-lane/ALU8 proxies only `0.964x/0.952x`; ncu attributes
   `63-68%` of issue stalls to cooperative barriers. Zero spills.
4. Tune dynamic depth/hysteresis through depth 15 under deterministic prompt
   scenarios without regressing the fixed-d3 `180.43 tok/s` reference.
5. Repeat the economy pass for ROCm and the remaining SingleDevice/EP lanes.
6. After ROCm SingleDevice MTP is green, replace the current MTP KV-only
   prefill lowering with a dedicated captured K/V-only stage. Preserve the
   exact hidden/embedding norms, MTP projection, attention-input norm, K/V
   projection, K norm/RoPE, and cache-format append, while eliminating Q
   projection/gating, Q split/norm/RoPE, Q scratch, and unshiftable or padded
   row work. Promotion requires byte-identical MTP KV payloads across cache
   formats and prefix restore, plus CUDA/ROCm profiler evidence for occupancy,
   registers/VGPRs, zero spills, and end-to-end prefill gain.
