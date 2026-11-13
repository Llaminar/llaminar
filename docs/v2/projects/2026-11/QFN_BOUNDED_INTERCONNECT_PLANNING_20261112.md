# Bounded interconnect sampling for automatic placement

This is a focused implementation and qualification note, not a model throughput
certificate. The target is automatic TP4 and PP(EP4, EP4) QFN placement under
the original MPI/NUMA ownership. Startup must remain a short measurement phase;
the longer fabric investigation is an offline diagnostic.

## Evidence and decision

The 2026-11-12 native study used eight MI50s, two observed four-card PCIe
switch groups and two mixed groups. Original MPI2 membership was retained.
Thirty balanced blocks completed with exact byte checks and clean driver and
process retirement. Evidence is under
`/tmp/qfn-qualification-d836-20261112/fabric-balanced-r1-run/`, particularly
`analysis.json`, `raw.jsonl`, `source-geometry.json` and `process-retirement.json`;
root receipts are `q297` and `q298` under
`/tmp/qfn-full-preflight-after-cpu-next/`.

Representative native-event medians, in microseconds:

| Operation and actual live payload | Same switch | Across switches |
|---|---:|---:|
| Directed P2P, 16 bytes | 29.12 | 29.99 |
| Directed P2P, 10,240 bytes | 30.14 | 32.48 |
| Directed P2P, 655,360 bytes | 158.26 | 180.81 |
| FP32 allreduce, one 2,560-column row | 69.80 | 98.10 |
| FP32 allreduce, 64 rows | 420.16 | 533.32 |
| Allgather, 64-row branch payload | 463.56 | 694.95 |

Same-switch grouping was useful on this host, but pairwise penalties did not
predict collective penalties. One affine latency/bandwidth fit missed the
medium payload by 32–44% despite apparently strong R-squared. Use bounded
piecewise observations and preserve direction. Neither these measurements nor
the startup matrix prove exact collective latency for an unmeasured subgroup.

## Implementation

`PlanningCommunicationService` obtains one complete directed pair matrix per
original rank and homogeneous GPU population, once across requested wire
precisions. Each pair uses three payload anchors: bounded protocol metadata,
one FP32 model row and the bounded prefill sample. Candidate count does not
multiply native work. Existing worker streams, PMA-owned buffers and a constant
number of executable slots are reused. The physical peer-copy probe is separate
from the existing native collective calibration. Direct edges use the shared
TransferEngine byte-copy implementation and one destination-owned graph, without
creating a communicator per pair. Original discovery selects the directed peer
edges; unavailable edges use explicitly identified, measured same-rank host DMA
legs in the cost model. They do not acquire invented raw peer samples. Only
active endpoints receive preparation work; there are no empty GPU launches.

`PlanningInterconnectCostMatrix` binds original rank/node/backend/ordered UUIDs,
retains all three raw samples and rejects missing or substituted evidence. Each
anchor uses their median so one scheduling outlier cannot dominate placement;
no extra samples or retries are introduced. The request cost model consumes
this matrix. Known directed request/reply edges retain
their actual bytes and direction. For unknown native collective schedules,
the worst directed service within the candidate is an explicitly labeled
placement proxy, on the same basis for every candidate. It does not assume a
ring, a degree discount or exact collective contention. Larger payload queries
use explicitly qualified measured-size chunks and the exact live remainder;
the query does not alter the inference transport's packetization.

Admitted compute and communication estimates rank candidates. Observed peer
access and PCIe proximity resolve equal-cost ties. Synthetic full-planner
regressions prove that faster measured cross-switch groups can win, and that
the preferred pair can change between decode-heavy and prefill-heavy workloads.
No machine GPU ordinals or fixed domain degree enter the production policy.

## Startup economy and focused qualification

The first complete Release automatic run (`q313`, nine eligible ROCm GPUs)
spent **5.099 seconds** in pair setup and **5.882 seconds** in replay. Its total
planning transaction was 113.58 seconds, versus the earlier 102.77-second
transaction. These are single observations; the pair timers directly identify
the added work. Eleven seconds of new work is excessive for this purpose.

Reducing repetitions and retaining events did not resolve ROCm's overhead:

| Diagnostic | Pair setup | Pair replay | Scope |
|---|---:|---:|---|
| Initial complete Auto (`q313`) | 5.099 s | 5.882 s | Release, nine ROCm devices |
| Short batch (`q320`) | 4.954 s | 4.550 s | Integration, same nine devices |
| Queue all three intervals (`q325`) | 4.820 s | 3.401 s | Integration, same nine devices |
| Short batch CUDA (`q321`) | 0.01160 s | 0.00185 s | Integration, two RTX 3090s |

These are short diagnostic observations, not matched inference benchmarks.
Raw samples and identities are retained in `startup-pair-fast-r1/` and
`startup-pair-queued-r2/` under the qualification directory. All four native
transactions and their driver/retirement checks passed. The reduced operations
still leave excessive ROCm startup cost and timing contamination.

The separate `q326` rocprof trace identifies the host-side start-event record
as a major contributor: across timed pair endpoints its API median is 6.392 ms,
while the two-kernel graph span median is 156 microseconds. These are instrumented
intervals, not unprofiled throughput; endpoint/thread sums overlap. The profiler
also changes queue instrumentation, so its timings cannot replace the ordinary
measurement. Trace and reproduction script: `startup-profile-r1/trace/` and
`analyze-q326.py` under the same qualification directory.

The captured-event candidate (`q334` CUDA, `q335` ROCm) combined all three
payloads and their raw windows into one graph per directed pair. CUDA added
13.97 ms (11.89 ms setup, 2.07 ms replay), but nine-device ROCm regressed to
12.57 seconds (4.36 s setup, 8.22 s replay). Its small-message windows retained
a systematic approximately 1 ms floor; taking a median cannot repair that
instrumentation error. These receipts are in `startup-pair-captured-r3/`.

The separate captured-event trace (`q337`, `startup-profile-captured-r2/`)
shows median small-message two-kernel spans of about 52 microseconds, separated
by roughly 3–4 ms gaps between windows. Profiled graph-launch API calls take
about 37.8 ms median. This identifies substantial scheduling between measured
operations; it does not identify an exact upstream HIP code defect. As before,
the instrumented timings are diagnostic, not throughput evidence.

The current candidate replaces captured event timestamps with one-thread
fixed-clock writes: CUDA `%globaltimer`, HIP `wall_clock64()` using the admitted
constant-frequency metadata. Native events now supply only final Ordering
completion. Each measured destination reads back a 144-byte timestamp bank after its
measured graph; no sampled payload or model state is downloaded. PMA owns the
exact device bank and page-rounded mapped host backing. Only same-device tick
differences are converted; no cross-device clock alignment is assumed. Completed
graph destruction is folded into the next setup on the same original worker,
removing the reset-only dispatch round.

Distinct guarded payload regions preserve each anchor's byte-extent proof.
One warmup and one measured graph execute; three raw windows per anchor remain.
Wire protocol v10 rejects the superseded event-timing and grouped-peer methods. ISA evidence in
`startup-clock-marker-isa-r1/` covers CUDA sm80/86/89/90 and HIP gfx906/gfx1100;
the marker has no stack, shared-memory use or spills.

The fixed-clock grouped-peer probe passed native byte-range, reset/replay and
retirement checks on CUDA and ROCm (`q344`), and the complete planning Unit owner
passed (`q345`). On nine ROCm GPUs its setup took 4.273 s and measured replay
0.228 s (`q342`); on two CUDA GPUs setup took 0.712 s and replay 0.00186 s
(`q343`). These single startup observations are in `startup-pair-clock-r4/`.
They show the instrumentation improvement, but do not qualify the latest
physical-copy producer.

The grouped-peer observations exposed a different bias: for all nine ROCm
devices, the previous-rank edge in the original communicator had approximately
238-microsecond small-message latency, independent of PCIe switch. Source
inspection maps that schedule class to RCCL's P2P channel selection; it does
not prove the precise cause of the delay. Treating that parent communicator's
schedule as physical link cost would wrongly favor mixed switch groups. The
current physical-copy producer removes this communicator dependency. Its
rankings remain placement proxies, not predictions of exact subgroup NCCL/RCCL
latency. The following focused runs qualify its correctness and startup cost;
no inference performance claim is made.

The first physical-copy attempt is **rejected** (`q349`, `q350`,
`startup-pair-physical-r5/assessment.json`). CUDA rejected the peer-specific API
inside capture. HIP accepted it but recorded no copy nodes: all 18 graph nodes
were clock markers, and even 655,360-byte samples reported about 0.74 us. The
former functional check inspected bytes already copied during recording and
therefore did not establish captured execution. The focused regression now
poisons destinations after recording and before replay, and the sampler requires
the expected copy-node population before accepting timestamps. Prepared peer
access will use capture-aware asynchronous D2D submission; ordinary expert
migration retains its existing submission policy. The reported 0.283 s setup
plus replay from this attempt is not a valid calibration-cost result.

The corrected captured-copy matrix passes CUDA and ROCm byte-range/replay tests
(`q354`). The complete four-case matrix/capability slice subsequently passes
on both backends (`q361`), including corrected fixture setup for deliberately
unpublished storage and publication events retired by terminal reads.
Unprofiled startup diagnostics
`q355`/`q356` retain 36 nodes per graph: 18 clock markers and 18 copies. All
transaction PMA claims and driver/process owners retire normally.

| Backend population | Matrix preparation | Matrix replay | Combined |
|---|---:|---:|---:|
| Nine original rank-owned ROCm GPUs | 0.30016 s | 0.09279 s | **0.39294 s** |
| Two original rank-owned RTX 3090s | 0.73655 s | 0.00180 s | **0.73835 s** |

These are single cold-process observations, not stress-test medians or total
server startup times. Complete native calibration transactions, including the
pre-existing collective sample, took 7.182 s and 1.385 s respectively. Receipt
directory: `startup-pair-physical-r6/`. Large-payload four-GPU worst-edge means
were about 30–35 us within the two MI50 switch groups versus roughly 80 us
across switches; this is physical-copy evidence, not a collective benchmark.
The short CUDA samples contain an isolated scheduling spike, motivating the
median-of-three placement estimator while preserving every raw observation.

The separate trace attempt (`q363`, `startup-profile-physical-r3/`) aborts in the
SDK with `Profiled SDMA timing marker lacks valid native timestamps` when memory
copy tracing is enabled. It supplies no usable timing/profile qualification.
The external driver observer retires cleanly. Preserve this instrumentation
limitation separately from the passing unprofiled copy gates; the marker's
all-shipped-architecture ISA/resource evidence above remains available.

Focused gates:

- `V2_Integration_PlanningInterconnectMatrixReplay`: immutable evidence,
  codecs, event lifecycle and actual planner selection, device-free.
- `V2_Integration_PlanningDirectedPairMatrixReplay_CUDA/ROCm`: real capture,
  exact live prefix and untouched tail bytes, repeated execution and retirement.
- `V2_Performance_PlanningInterconnectStartup_CUDA/ROCm`: optional startup
  diagnostic, one production-size sample without functional readbacks. GTest
  XML retains raw directed samples, original UUIDs and wall-time breakdown.
  It has no hardware-dependent throughput threshold and is outside preflight.

The initial matrix/selection gate passed (`q312`); initial CUDA and nine-GPU
ROCm native regressions passed with clean driver windows (`q306`, `q308`).
The earlier broader planning owner (`q314`) exposed two failures in native
sample deduplication across eligible original MPI ranks. Sampling now retains
each original rank's matrix, including a smaller visible group that another
rank's group contains. The original failing tests remain. A separate regression
also requires a supported GPU/CPU expert overlay with nonzero expert ownership
on both tiers to use its actual mapped-host service. The complete planning Unit
owner and both focused device-free preflight entries passed in `q329` (44.93 s).
The later physical-copy producer is qualified by the final focused gates below.

Final focused qualification: the complete 523-test planning Unit owner and both
device-free matrix/native-context preflight entries pass (`q368`, 48.45 s).
CUDA/ROCm captured-copy matrix and capability lifecycle gates pass (`q361`).
The estimator retains all samples and uses the median of three; its outlier
replay and no-P2P missing-host-evidence regression belong to the explicit
matrix preflight entry. This is a focused slice, not a full preflight or model
certificate. Release and Integration serving/test owners are rebuilt (`q366`,
`q367`). A fresh Release Auto request (`q372`) then measures the nine-device
matrix in **0.32973 s**: 0.23661 s preparation and 0.09313 s sampling. It reuses
the 216 directed payload observations while ranking candidates and selects
ROCm devices 1–4, all behind the observed `0000:86:00.0` switch under original
discovery rank 1 / NUMA node 1. No visibility mask or requested GPU ordinal
determines that selection. The complete planning owner takes 119.10 s, including
the pre-existing model/kernel calibration and candidate evaluation; that total
must not be attributed to the new matrix.

The unchanged saved Apply plan passes the real PMA dry-run (`q376`, 15.46 s),
including native context accounting, context 45,056, all-GPU
GateUpOwnedDownColumns experts, dynamic MTP capacity 15, RAM ngrams and the
requested prefix-cache budgets. Both native owners and their continuous driver
observers retire cleanly. Artifacts are in
`auto-resume-physical-copy-prep/tp4-context-45056/`, `tp4-q372-join/final/` and
`tp4-q372-context-r1/` under the qualification directory. This proves agreement
between Auto and Apply for that plan; it is not a maximum-context, full model
materialization or generation result.

The QFN OpenCode runs remain pending. A separate Auto-to-Apply admission gap
omitted the typed native context contribution during Auto selection. Its owner
breakdown is retained in `tp4-auto49152-apply-debug-r3/native.log` under the
qualification directory. The corrected contribution passes focused real-BOM and serialized Auto/Apply
regressions. A separate historical-topology capacity control (`q330`) admits
45,056 context tokens with the complete native contribution and unchanged cache
budgets, dynamic MTP and expert policy. This control is not a fresh automatic
selection, maximum-context proof or server qualification. Fresh Auto and its
fixed-plan validation now pass as above; maximum-context qualification and the
actual server workloads remain outstanding.

The subsequent fixed-topology PMA search (`q378`) admits **47,168** tokens and
rejects the adjacent aligned **47,232**-token request with the same policies.
The search varies only public context demand on the saved selected topology;
it does not rerun calibration for each capacity probe or claim a maximum over
other automatic candidates. All 14 probes and the continuous driver owner
retire normally. A fresh Auto request at the admitted bound is prepared for
the final source build.

The first fresh PP8 Auto request (`q377`) keeps matrix preparation plus sampling
at 0.39508 s but fails before selection: `Immutable lookup cost requires final
admitted source placement`. Pipeline admission had published its genuine final
placement to rank runtimes but omitted it from escaping stage device plans.
The scoped repair publishes that same placement object to its admitted device
owners before the stage inputs move; costing retains its strict missing-source
rejection. The focused
`PlanningImmutableLookupWork.AutomaticPipelinePublishesFinalFixedHeadPlacementToStageDevices`
regression exercises actual automatic 4+4 GPU admission on both backend metadata
paths, original rank remapping, later-stage source reads and exact live byte
counts. It belongs to the existing explicit
`V2_Integration_QFNImmutableLookupWorkReplay` preflight entry. The failed PP8
request is not a capacity rejection and cannot justify reducing its context.

The repaired source builds in Integration and Release (`q379`, `q380`). The
complete planning Unit owner and all three focused preflight entries pass
(`q381`, 44.92 s), including the new PP regression. Fresh TP4 Auto at **47,168**
tokens passes (`q386`, 103.45 s) and again selects devices 1–4 behind the same
observed switch. Matrix preparation plus sampling is **0.40909 s**. The adaptive
local ngram cache fits about 2.03 MiB per participant at this maximum-context
candidate, within its requested 256 MiB ceiling; that ceiling is not a guaranteed
reservation. The following exact Apply and runtime checks must retain the
actual admitted cache size rather than claim the full ceiling was resident.

## Whole-planner latency target

The user requires **under five seconds for planning including measurements**,
ideally less. The sub-second pair matrix alone does not meet this requirement.
The PP8 retry (`q388`) was stopped after 662.44 seconds through its exact MPI
session owner, with clean driver and session retirement and no selected plan.
It is an interrupted diagnostic, not a failed capacity admission or a pass.

Two five-second CPU profiles distinguish roles: original rank 1 waits in MPI;
the rank-0 coordinator actively constructs candidate admissions. The latter
profile has 502 samples and zero lost samples. Its inclusive call graph places
87.85% in `MemoryPlanner::planOwnerGroup`, with 56.77% in workspace estimation,
15.14% in weight estimation and 11.95% in floating-point workspace sizing.
Inclusive percentages overlap and must not be added. Flat samples show repeated
weight-name matching and allocation. Evidence is in
`pp8-planning-cpu-sample-r3/`; `r2` contains the waiting peer. The ordinary `/usr/bin/perf`
wrapper lacks this custom kernel's packaged tool, so the retained diagnostic
uses `/usr/lib/linux-tools-6.8.0-138/perf` without changing host settings.

The exhaustive PP request has 29,610 proposals. Even TP4's 252 proposals spend
about 76.3 seconds between the completed matrix log and final selection in
`q386`; source-owned phase timers are being added to measure exact boundaries.
The repair keeps every candidate's physical admission and reuses only immutable
source classification and pure sizing work. Automatic calibration is also being
changed to request physical peer samples directly: its cost model no longer
uses the native allreduce observations, so creating a communicator and sampling
those collectives is unnecessary startup work. Explicit native collective
measurement remains a separately typed operation.

The first latency batch builds in both configurations (`q389`–`q393`). The
focused run (`q394`) passes the complete planner, metadata and weight owners and
the five selected preflight entries. Its workspace owner catches two tests that
mutated a global precision selector after constructing the newly frozen geometry;
their explicit-policy repairs preserve the byte and rejection assertions.

The host-only real-model diagnostic compares eight balanced PP placements twice,
using original saved hardware facts without initializing a GPU runtime. Prepared
admission takes 1.536 s versus 4.561 s without per-search reuse (`q395`, `q396`),
but this is **not an accepted speedup**: comparison of the complete resource
breakdowns catches an indexed-weight undercount of 115,286,016 bytes per final-
stage GPU. The new index incorrectly excluded appended MTP tensors beyond the
nominal main-layer count. Its repair retains all original layer keys and checks
the caller's explicit interval. A cross-backend sidecar regression is added;
the real-model byte comparison must pass after rebuilding. Both diagnostics
have two accepted and fourteen rejected proposals; equal verdicts alone are
insufficient evidence of equivalent admission.

The separate profiled diagnostic (`q397`, 898 CPU samples, zero lost samples)
still spends substantial time constructing repeated projection-arena, activation
and prepared-expert workspace declarations, and advancing mandatory expert
quotas one slot at a time. Follow-up changes reuse only complete immutable quote
inputs and batch exact mandatory quota increments while retaining canonical PMA
checks and the original first-failure boundary. The host probe, comparison and
profile receipts are in `host-planning-scaling-prep/`. The full 29,610-candidate
request has not been rerun, and the five-second requirement remains unqualified.

The corrected second batch passes all fourteen focused CTest entries (`q400`),
including the complete planning, metadata, weight, workspace, physical-root and
MoE-capacity owners. Integration and Release binaries are rebuilt (`q398`,
`q399`). The real-model sample now has **identical complete resource breakdowns,
normalized configurations and rejection diagnostics** to both the current
uncached reference and the earlier canonical reference. Sixteen admissions take
0.575 s prepared versus 4.270 s uncached, a **7.43x** reduction in this sample
(`q402`, `q403`). Both have two accepted and fourteen rejected proposals. The
full command takes 0.971 s, including 0.107 s of metadata preparation. Evidence
is in `host-planning-scaling-r2/`; this remains a host-only sample, not a
whole-planner latency certificate.

### Reduced search and CPU parallelism

Automatic PP now assigns one deterministic contiguous layer partition per
physical roster, proportional to stage participant counts and with at least one
layer per stage. Integer remainder assignment is deterministic; explicit applied
plans retain their own layer ranges. This rule covers arbitrary stage counts and
unequal widths. For the current 48-layer request with two four-device stages,
the 47 boundary variants disappear: 630 physical rosters replace 29,610
proposals. Device selection still uses original observed ownership and measured
link costs. No machine-specific ordinal list or candidate cutoff is introduced.

Independent CPU admissions are being parallelized with the configured OpenMP
team. Each worker owns its preparation and immutable-quote reuse scope; each
candidate still receives its own physical admission. Only a team-sized batch is
retained. Cost evaluation, rejection callbacks and final selection remain ordered
on the original caller. New focused equivalence and lifecycle tests and fresh
whole-planner timings are required before qualifying this change.

Physical-peer-only capture/replay passes on CUDA and ROCm (`q406`), with zero
driver findings. Explicit startup samples (`q407`, XML receipts) include setup,
sampling and PMA/native retirement: **0.594 s** for two CUDA GPUs and **1.335 s**
for nine ROCm GPUs. Their pair capture/replay portions are 0.580/0.00176 s and
0.245/0.108 s respectively. Both record zero native collective samples and
complete authority retirement. These receipts exclude prior discovery/backend
bootstrap; they must not be called full planner wall time. The cardinal and
OpenMP source then builds successfully in Integration and Release (`q408`,
`q409`); focused validation and complete Auto timing follow.

The full-selector host diagnostic (`q412`, `q413`) now processes every one of
the 630 proposals. One worker takes **16.841 s**; 28 configured workers take
**3.632 s**, a **4.64x** reduction. Both produce 70 accepted and 560 rejected
admissions, 1,120 participant weight-work records and 26,880 state-layer records;
all callbacks retain the original caller. Source loading adds about 0.108 s.
This diagnostic uses a clearly labeled constant tie-cost evaluator plus actual
forward-work compilation. It does not measure hardware calibration, production
cost evaluation or CLI lifecycle and cannot qualify the five-second requirement.
Receipts and the exact harness are in `host-planning-scaling-r3-prep/`.

The scoped gate (`q411`) passes fourteen of sixteen CTest entries, including
the entire planning and execution-plan owners and the explicit cardinal/parallel
regressions. Both failures name one newly added tensor-sharded capacity fixture:
its independent byte oracle applied cache-line alignment where production uses
4 KiB per prepared projection. The test is corrected and expanded to a geometry
with unequal physical fragment sizes; no production validator is weakened.

The corrected capacity owner and its explicit preflight regression both pass
(`q414`, `q415`). Together with the unchanged passing entries from `q411`, this
completes the focused host gate. It is not a full production-preflight run.

Fresh Release PP8 Auto now completes successfully (`q416`, clean driver/session
retirement): **21.047 s CLI wall**, **17.530 s startup API**. The phase breakdown
is metadata 0.115 s, hardware calibration 13.895 s, budget metadata 0.013 s and
candidate admission/selection/publication 3.506 s. Calibration comprises
weights/streaming 7.121 s, FP32 arithmetic 3.896 s and communication 2.877 s.
Every proposal is considered: 70 admitted and 560 capacity-rejected.

The actual chosen order is the observed MI50 bank behind PEX `af` followed by
the bank behind PEX `86`, retaining original discovery rank 1 / NUMA node 1.
The saved plan has contiguous layers 0–23 and 24–47, 262,144 context tokens,
GPU-only GateUpOwnedDownColumns experts, FixedHeadSharded VRAM ngrams and
dynamic MTP capacity 15. Device ordinals were not supplied as a planner hint.
This proves automatic selection and planned physical admission, not model
materialization or OpenCode completion. The five-second target remains unmet.

The next scoped pass targets calibration, which remains serial across devices.
The host-device byte-proof path unnecessarily invokes a complete timing helper
despite discarding its timing; a single ordered readback preserves the proof.
Passive projection phase timing will identify conversion, host fill, engine
preparation, native execution and retirement costs before further optimization.

The byte-proof change passes focused host replay (`q421`) and four actual
CUDA/ROCm service/NUMA-role integration entries (`q422`), with clean driver
retirement. It preserves all measured service coordinates and replaces each
discarded timing family with one ordered proof launch. The subsequent PP8 run
(`q423`) passes in 19.848 s CLI / 17.391 s API. Communication takes 2.573 s;
selection takes 3.954 s and hardware calibration 13.308 s. These single-run
timings are not an isolated attribution of every observed difference.

The new phase observations locate the FP32 proxy cost: operand fill takes about
4–9 ms and engine preparation 19–27 ms per GPU. M64 preparation/measurement/
retirement takes 219–375 ms, while its three native timed replays take only
0.2–2.45 ms. A separate CPU profile of the canonical source-free arithmetic
integration (`q424`) passes with 1,993 samples and zero lost samples. Its DEBUG
timeline puts the delay **before graph instantiation**; instantiation itself is
about 0.15 ms. Samples include Tensile library deserialization, HIP code-object
loading and LLVM work. MPI waiting dominates the aggregate profile, so aggregate
sample percentages are not a direct breakdown of active GPU-worker time.

A minimal workspace change lazily hoists the source-only sharding configuration
out of the QFN correction loop; its Release and Integration builds and focused
equivalence checks pass (`q425`–`q427`).

The `rocblas_initialize()` experiment is **rejected and removed**. It built in
both configurations and passed focused lifecycle and native service checks
(`q428`–`q431`), but moved or increased the cold cost. Standalone ROCm FP32
collection took 8.904 s, including about seven seconds of eager library setup.
Fresh full Auto (`q432`) took 20.208 s CLI / 16.710 s API, with 13.243 s hardware
calibration and 3.343 s selection. FP32 collection remained 3.698 s: earlier
families had already initialized the library, and the shape-specific recording
cost persisted. Both native runs retired cleanly with zero driver findings.
Fast post-initialization captures are not a whole-startup improvement.

The next scoped candidate prepares the actual FP32 projection on each original
GPU worker concurrently, then measures each ready graph in isolation. It reuses
the existing production projection and timing bodies, retains one live row
phase per device, and admits simultaneously retained host staging through PMA.
CPU sampling remains on its original workshare. Preparation, completion,
measurement and retirement must preserve original worker ownership and drain
every started task on failure. This is pending implementation and qualification;
the under-five-second whole-planner target and both model app runs remain open.

The first prepared-FP32 candidate builds in both configurations (`q433`, `q434`)
and passes all thirteen focused host entries (`q435`), including the complete
planner owner. Native `q436` passes the individual CUDA/ROCm producer lifecycle
tests. Its expanded CUDA collection catches a new fixture mistake: CPU floating
engines borrow their source and correctly report zero separate prepared bytes.
The fixture now distinguishes that from GPU prepared storage.

The same run exposes a real ordering defect in the new concurrent candidate:
one ROCm worker starts capture while another is still finalizing its upload
pool. The latter's cold device synchronization invalidates the active capture.
This is an application abort, with clean driver observation and no host reboot.
The fix separates payload construction, phase allocation/upload, capture, and
post-capture instantiation/warmup into completed worker cohorts. Timed samples
remain isolated. The all-device collection is the reproducer; the prepared-owner
regressions also check rejection of reordered phase operations. No synchronization
error or capture failure is suppressed, and this candidate is not yet qualified.

The staged repair builds in Integration and Release (`q437`, `q438`), passes the
three focused host entries (`q439`) and all four native CUDA/ROCm collection and
lifecycle cells (`q440`). Five additional fresh-process ROCm collections pass
(`q441`), each including repeated complete collection; every driver window is
clean. The native traces show all payload/storage/capture/completion futures
retiring before the next cohort and the isolated measurements. Test coverage
also includes abandoning a warmed phase and checking private PMA retirement
before creating another producer.

Two fresh matched Release PP8 Auto runs (`q442`, `q443`) take **17.635/17.752 s
CLI** and **14.417/15.309 s API**. FP32 collection is **1.629/1.639 s**, compared
with 3.565 s in the accepted serial-preparation control `q423`; this is a
measurable reduction while retaining exactly one warmup and three timed calls.
Candidate selection takes 3.095/3.354 s. Both runs consider 630 proposals, admit
70, reject 560 for capacity, and select the same observed MI50 switch groups.
Both have clean driver and session retirement. Total startup still misses the
five-second requirement; weights/streaming (6.911/7.555 s) and communication
(2.653/2.633 s) remain separate costs.

The source-backed extension reuses this staged producer for ordinary projection
samples. It retains one shared source per rank, sums each concurrent GPU's
private preparation, and deletes the duplicate former GPU path. Integration
and Release builds pass (`q444`, `q445`), as do the complete planner unit owner
and four focused host entries (`q447`). All six CUDA/ROCm native entries pass
(`q448`), including repeated source publication for Q8_0, F16, BF16, F32 and
the canonical F16-to-FP32 conversion. Source storage, private claims, abandoned
phases and repeated collection retire correctly; the driver window is clean.

Two matched full Auto runs (`q449`, `q451`) take **18.564/18.413 s CLI** and
**15.309/15.171 s API**. They retain the same 630/70/560 proposal/admission/
rejection counts and the same physical switch groups. This is **not a proven
whole-startup speedup** over `q442`/`q443`. FP32 collection remains 1.61–1.67 s;
weights/streaming takes 7.47–7.64 s and communication 2.62–2.65 s. Both driver
windows are clean. Expert and streaming preparation have not changed.

The host-only canonical visitor diagnostic (`q446`, `q450`) takes
**33.1–34.4 ms** for each of five independent 630-proposal enumeration passes,
with the same original discovery-rank union `{1}`. Model metadata loading is
reported separately (110 ms). This is enumeration evidence, not an admission
or winner certificate, and it does not justify dropping host/storage evidence.

The workspace schema/vector hoists preserve quote identity, original geometry
and lazy validation boundaries. Their Release/Integration builds (`q454`,
`q455`) and complete selected host owners/equivalence/policy checks (`q456`)
pass. Fresh Auto `q457` completes in 18.766 s CLI; this is not evidence that the
small CPU hoists materially improve whole startup.

Startup expert sampling now records two required phases (decode and prefill),
while live model execution retains all three expert phases, including MTP.
This reduces calibration work only; it does not remove MTP execution, alter its
arithmetic, or relabel prefill observations as verifier measurements. Builds
`q458`–`q460` pass. Host checks `q461` and `q463` exposed stale phase-coverage
fixture expectations; their failures remain retained. The repaired fixtures and
final Integration/Release rebuilds pass (`q467`, `q468`). Final host `q469`
passes both complete `Unit_MemoryPlanner` and the explicit
`PlanningStartupExpertPhaseCoverage` preflight entry (42.284 s). Native `q464`
passes all ten selected CPU/CUDA/ROCm execution, MPI expert/service and prepared
producer entries (39.618 s), with clean external driver/retirement evidence.

Fresh matched Auto `q465` completes in **18.358 s CLI / 15.0639 s API**:
weights/streaming **7.43642 s**, FP32 **1.70327 s**, communication **2.61562 s**,
and selection **3.18033 s**. The actual saved selected plan chooses stage 0
ROCm ordinals **[1,4,7,8]** and stage 1 **[2,3,6,9]**: both cross the two
observed PEX banks. This is the measured-cost winner of that run, not a
same-switch grouping result or demonstrated application-performance optimum.
The earlier `q442`/`q443` grouping observations are separate receipts. Fresh
Auto will select again; no roster is forced. The under-five-second whole-startup requirement
remains unmet. The separate instrumented trace `q466` is diagnostic, not an
unprofiled startup measurement. Remaining cold-library, endpoint-demand and
phase-timing questions were parked after the focused-green slice. The user
subsequently identified mixed-switch selection as a bug and blocks OpenCode
until it is fixed. The earlier freeze does not certify this selection behavior.

Fresh `q474` TP4 and `q475` PP8 plans admit but select mixed groups, taking
**17.44/17.57 s CLI**. DEBUG `q477` compares all 70 admitted PP proposals from
the same immutable observations. Its mixed winner is [1,2,4,9]/[3,6,7,8].
The bank-local [1,2,3,4]/[6,7,8,9] partition has the **lowest predicted request
communication**, **0.201158 s** versus the winner's **0.245429 s**. However,
its predicted compute is **310.68028 s** versus **308.00886 s**, so the additive
objective favors mixed placement by **2.627722 s** total. The reverse bank order
predicts 312.87483 s compute. These are cost predictions, not inference timings;
component evidence is rounded. All 70 records and both orders are retained in
`/tmp/qfn-qualification-d836-20261112/planner-switch-selection-debug-q476/comparison.json`.
This isolates compute calibration as the dominant ranking component rather
than showing that measured communication prefers mixed banks.

The source-backed `q478` trace finds a concrete calibration defect: an external
event interval is **2,413 us** around roughly **130 us** of projection kernels;
expert timing is **2,087 us** around roughly **172 us** of kernels. Host enqueue
and idle gaps contaminate the compute service. The diagnostic driver windows
are clean (zero findings); these traced intervals are not unprofiled throughput.
The implemented repair gives startup projection, expert and streaming samples
one fixed-device-clock parent enclosing exactly three original body instances,
after one original-body warmup. Compact terminal timestamp readback replaces
host-enqueue-spanning event timing for this prepared path. Arithmetic, live MTP
phases and placement policy are unchanged; no switch rule or roster override
is installed. The final contiguous warmup/clock parent passes Integration and
Release builds (`q486`, `q487`), the complete planner Unit owner and focused
host entries (`q488`), and ten native producer/replay entries (`q489`). The
driver window is clean. Fresh Auto `q490` still selects mixed banks: its winner
predicts 12.154592 s versus 12.192201 s for bank-local placement. Removing
instrumentation error alone does not make short per-card samples stable enough
to rank equivalent compute hardware independently.

## Actual communication and shared PCIe cost

The current scoped repair retains raw per-device samples but uses median
complete compute predictions within authenticated identical hardware classes.
Physical admission and measured directed link services are never pooled. The
request model retains the existing compute/collective/stage-transfer breakdown;
there is no new execution simulator or candidate search.

Startup now records one bounded native allreduce basis per original
rank/backend, using the same captured-clock producer. Its smallest positive
payload supplies a native latency floor. Only the payload-dependent excess is
adjusted by the candidate/reference pair service and observed shared PCIe cut
pressure. This is an explicitly qualified subgroup estimate, not a measured
subgroup benchmark or a guessed ring/tree algorithm. Native fixed latency
cannot disappear because a pair copy is faster. The exact reference group
recovers its native curve. Contradictory timing below a known wire-capacity
floor is rejected. No communicator is created for each candidate.

Observed negotiated link speed, width and ancestry supply PCIe cut floors.
Same-switch traffic does not consume a shared upstream link it never crosses.
Allreduce requires one live payload in each cut direction; allgather charges
each unique owner's fragment once across a needed cut, not once per consumer.
Opposite directions are full duplex. Unknown fabric capacity stays explicitly
unqualified. Expert counted-gather costing follows the actual ordered native
broadcasts and uint64 live extents; the obsolete mapped-peer description is
removed. Immutable gathers use actual consumer-assembled head fragments.
QFN PP transfers now use the residual-trunk width rather than the four-times
smaller compute width; ordinary attention/FFN reductions remain compute-width.

Release `q493` and Integration `q495` build. New cut/composition/immutable
regressions pass in `q496`; four older/new-fixture assertions need the recorded
fixture correction before the whole planner owner is green. All eight native
CUDA/ROCm collective, pair and captured-clock regressions pass (`q497`,
63.50 s), with clean driver and resource retirement.

Actual Auto PP8 `q498` selects **[6,7,8,9] / [1,2,3,4]**, the two intact
observed switch groups, from all 70 admitted candidates without a forced
roster. Both bank orders have the lowest predicted communication cost. The
selected 256/256 request prediction is 12.068229 s; this is not inference
throughput. Evidence is in `planner-class-service-pp/comparison.json` under
the qualification root. Startup is **27.677 s CLI / 23.8125 s API**, including
7.456 s weight/streaming, 1.541 s FP32, 8.468 s communication and 6.222 s
admission/selection/publication. The measurement phase is bounded, but the
under-five-second requirement remains unmet and this slice is not a startup
speedup. Driver and process retirement are clean. Final TP4 selection and
application qualification remain separate gates.

### Current qualification: native gather calibration (2026-11-13)

The single `q498` grouping success did not persist. After the host-transfer
clock repair, `q507` TP4 selects `[1,4,7,8]` and `q508` PP8 selects
`[1,2,4,7] / [3,6,8,9]`. API planning takes **21.546 / 23.015 s**;
both driver windows are clean. These are the latest completed automatic plans,
and neither qualifies the requested switch-local application layout.

The host-transfer repair uses the same prepared captured clock transaction as
compute and native collectives, with fresh CPU publication before each timed
invocation. It replaces external event timestamps that included host gaps.
`q502/q503` Release/Integration build; `q505` eight native clock/transfer checks
and `q506` four original-MPI-role checks pass. `q509` repairs a flat synthetic
host-service curve in the PP-width regression; `q511` passes all ten selected
host entries, including the complete planner Unit owner. Those receipts precede
the new native gather implementation below.

The remaining `q507` ranking error is attributable to the old expert-gather
proxy. It summed worst tiny directed peer-copy samples as if each was a native
broadcast. Multiplication across 48 layers prices projection communication at
4.443 ms for bank A, 3.973 ms for bank B, and 3.117 ms for the mixed winner per
decode step. This explains essentially the entire 1.327/0.856 ms disadvantage
assigned to intact banks. The repaired four-consumer RAM-lookup host service is
only about 35 microseconds, with roughly one microsecond between candidates.
Raw pair observations are retained; no fastest-sample selection or forced
device roster is used to erase this discrepancy.

Source inspection identifies the actual operations:

- Expert intermediate records use `DeviceCountedAllGather`: frame advance,
  ascending-root uint64 extent publication, completion-aware live payload,
  validation and stream publication. Empty owners still take part.
- Completed expert columns use a regular native allgather, not the counted
  root loop. Fixed and live-row forms retain the same logical equal fragments.
- Fixed-head ngrams first gather an 8-byte validation header per consumer.
  TP4 and the requested PP4+4 then use an equal native-row payload gather.
  Unequal assembled PP fragments instead use the counted protocol.

The scoped replacement records two anchors for each real gather family on the
existing original communicator: fixed 1-byte/bulk and counted zero-byte/bulk.
The counted observation calls the existing production frame/binding rather
than reconstructing its protocol. Distinct source/receive banks, frame storage
and four concurrently retained timing graphs are admitted through PMA. Native
functional tests use late unique source publication, untouched-tail guards,
zero payload and repeated collection; ordinary planning downloads no payload.

Costing keeps the native control floor and adjusts only payload-dependent
service using the measured bulk interconnect and known PCIe cuts. It does not
use tiny peer-copy launch latency as native control. Counted costs include
extent and completion once; their measured root mean estimates unequal owners
and changed degree, with the frame overhead explicitly amortized. Fixed gather
cost retains the containing group's control floor. Device-row fixed-head
cost uses that fixed payload-service estimate; its extra count binding remains
unmeasured. These are qualified predictions, not per-candidate benchmarks.
There is no new communicator per candidate, topology search or runtime simulator.

Release `q519` and Integration `q520` build. `q521` passes all **12** selected
host entries, including the **569-test planner Unit owner**. `q522` passes all
six CUDA/ROCm native gather, collective and directed-pair replay entries in
55.22 s, with clean driver closure. The first native diagnostic run `q518`
found that late oracle source publication needed its producer event joined
again before the next capture. Per-phase preparation now joins that exact
stream through `TransferEngine`; production input bytes and arithmetic are
unchanged. The ordinary validation-disabled path already passed in `q518`.

Fresh actual Auto `q523` selects TP4 **[1,2,3,4]**; `q524` selects PP8
**[1,2,3,4] / [6,7,8,9]**. Both choose intact observed switch groups from all
70 admitted proposals without a forced roster. Both bank-local orders beat
every mixed group on predicted communication. TP4's closest mixed competitor
is 84.86 ms slower for the modeled 256/256 request. That is a cost-model
difference, not measured inference throughput. The saved plans and complete
comparisons are `planner-native-gather-{tp,pp}-q512/` under the qualification
root. Driver and process retirement are clean for both.

TP4 startup is **24.797 s CLI / 22.0603 s API**; PP8 is **27.386 / 23.8253 s**.
The under-five-second target remains open. The user accepted pausing further
startup tuning to proceed to the application goal once placement is correct.
OpenCode has not started. Before final admission, `q525` tests TP4 at the
262,144-token model maximum and receives a genuine complete-BOM capacity
rejection, with clean driver closure. The 47,168-token TP4 plan is accepted,
but the current largest fitting context still requires qualification. PP8 is
already planned at 262,144 tokens and awaits native Apply admission.

Historical PP accuracy receipts in the dashboard used **Apportioned** expert
placement and do not certify current GateUpOwnedDownColumns PP8. TP4 has a
separate existing **green local bounded-row pilot** (`q299`), following the
user-authorized one-row allowance and matched q293/q296 quantization evidence.
Its prefill/decode mean KL remains 0.0402656/0.0340420; all means, agreement
limits and original observations are unchanged. The original red report is
preserved. See `tp4-q257-one-row-review/EVIDENCE.md` and its separate numerical
summary under the qualification root. This supersedes the older incomplete
causal assessment; it is not a calibrated release or dynamic-MTP certificate.
Current PP GateUp numerical qualification and both complete app runs remain
open; planner/lifecycle passes do not establish those results.

### Applying the selected plans

`q527` refreshes the original MPI2 inventory with clean retirement. `q528`
uses the existing PMA dry-run context selector on the unchanged TP4 Apply
configuration: **47,168 accepted / 47,232 rejected**, with all 14 probes and
the driver observer clean. This equals q523's actual selected context, so no
redundant Auto calibration is needed. Exact fixed Apply `q529` also passes.

PP8 fixed Apply `q530` rejects its saved model-maximum plan before weight
loading. The complete BOM differs from Auto only in `NativeGraphExecutable`: Auto
570,425,344 bytes versus Apply 891,289,600 bytes, a 153-executable / 306 MiB
difference per stage-two GPU. The selected fixed-head ranges then exceed the
budget by 58,104–88,180 bytes. No context reduction or artificial reserve is
used to hide the disagreement. A focused quote repair is in progress.

TP4 smoke `q531` reaches original four-device setup but exits before HTTP
readiness: prepared-service calibration requires a complete expert, while the
selected GateUpOwnedDownColumns bank correctly owns movable gate/up payloads.
Production telemetry prices gate/up only; fixed down-column work and exchanges
are outside that movement-service boundary. The calibration repair must use
the same boundary, not fabricate a whole expert or disable maintenance.
Both failed native lifetimes retire cleanly with clean driver reports. No
OpenCode app, smoke response or benchmark has completed for these final plans.

The OpenCode admission classifier also lacked the frozen fixed-head BOM
diagnostic. Its focused regression requires complete positive deficit arithmetic
and rejects crashes, truncated or inconsistent quotes. This only classifies a
controlled capacity rejection; exact selected-plan validation still fails and
never silently lowers context. The 18-test runtime owner and four context
selector tests pass; explicit preflight registration is added.

### Focused admission and calibration repairs

The source-free/source-bound pipeline regression reproduces the PP discrepancy
without devices. The original preparation source contributes birth rows for
every local stage GPU. The first stage incorrectly assigned its captured model
family to those rows even when the device was only a member of a later stage.
Real Auto/Apply diagnostics `q536/q537` confirm the extra **153 executable slots /
306 MiB** on each second-stage GPU: 92 model identities plus 61 auxiliary graphs.
The real owner's 272 executable slots are unchanged. Both native lifetimes
retire cleanly.

`completeFixedOwners` now contributes captured model executables only for an
actual stage continuation or routed endpoint. Original native-source bytes
remain in the same BOM and authority. The focused CUDA/ROCm device-free test
compares source-free and original-source Apply at graph capacities 0 and 15;
it failed before the fix and now passes, including 20 repeated executions.

Prepared gate/up calibration now calls the existing production gate/up kernel
with the authentic fixed-down format descriptor and original owner-domain
dispatch metadata. It measures only the movable projections, matching production
service telemetry; it does not reconstruct or execute a missing whole expert.
Complete-expert measurement retains its existing resident-source contract.
Floating and Q4_K/Q4_K/Q5_1 native fixtures cover all three service phases,
repeated capture/replay, source isolation and retirement on both GPU backends.
Each independent original source runs in its own process, as required by the
process-lifetime PMA authority.

Release `q538` and Integration `q539` build successfully. `q540` passes all six
focused host entries, including the **570-test planner Unit owner**. `q541`
passes all six CUDA/ROCm prepared-service entries in 16.07 seconds with clean
driver closure. These are focused mechanism gates; real Apply, numerical and
HTTP/application qualification remain separate.

Real fixed Apply `q542` now passes the unchanged PP8 plan at **262,144 tokens**,
with native-source-only rows contributing zero model graphs and the true tail
retaining exactly 570,425,344 graph bytes. Driver retirement is clean. Public
request metadata and original MPI2 hardware inventory are refreshed against
the q538 Release binary (`q543`–`q545`). Exact TP4 Apply `q546` again passes
**47,168 tokens**; the prior same-policy maximum-context bisection is retained.
The fresh TP4 smoke is in progress (`q547`); no application pass is claimed.

### First final-policy inference and measurement

TP4 `q547` returns exactly `ready` with a normal stop and two completion tokens.
The original report remains red because its graph checker required the legacy
executed-capture counter even for a setup-materialized executable. The corrected
canonical checker joins physical rank/device/context/executable-family identity
to the actual replay and completed decode phase; it retains all ordinary
lifecycle obligations and rejects missing, empty, foreign or stale evidence.
All 158 owner tests pass (`q550`). Separate immutable-artifact revalidation of
q547 passes; resources, driver and graceful retirement were already green.
This is a short response smoke, not stochastic MTP or application qualification.

The retained physical observation also proves all four GPUs share observed
upstream switch `86:00.0` and mutual P2P. The task-local checker had equated
individually nearest leaf switches, which is too narrow for the nested fabric.
Its generic common observed type-5 ancestor check passes 14 pure tests and the
saved TP4 evidence. Required switch grouping is now explicit; ancestry alone
does not establish measured link performance.

TP4 benchmark `q549` completes one warmup and three measured samples with
512 prompt tokens and 256 output tokens, dynamic MTP and prefix cache off.
Canonical medians are **616.270 prefill / 93.320 decode-after-prefill tok/s**
(the latter prices 255 decode tokens). Sample ranges are 615.172–616.584 and
93.240–93.381 respectively. This is the public synthetic benchmark, not a
coding-workload or matched external-engine measurement. Driver and native
retirement are clean. Its original wrapper report remains red: it mixed
explicitly requested MTP/graph evidence into the narrower passive-lookup
validator, and the passive initial-bank validator did not recognize uniform
gate/up ownership. Separate revalidation retains every observed record and
checks gate/up banks with the existing exact nonreplicated population rule.
The timing is valid as a local diagnostic; full-lifetime graph/MTP certification
is not inferred from counters reset at the measured epoch. The 40 benchmark
policy and 11 wrapper tests pass.

PP numerical `q548` produces no logits. The canary incorrectly treats the
excluded original discovery rank as an output owner while the selected rank is
still initializing. The native process also reports SIG11; there is no stack
and its relationship to the abort is unclassified. The first lifetime repair
passes six focused gates (`q553`), but its fresh frontend run `q554` exposes a
missing MPI initialization before obtaining the original context. That second
regression is repaired using the existing sole MPI process-session owner and
an explicit borrowed runtime-initialization entry. The fresh-process regression
starts before MPI initialization, rejects an unowned borrow, retains the original
discovery context through inactive-rank retirement and verifies finalization only
after its sole owner retires. Release `q555`, Integration `q556` and all 14 focused
entries in `q557` pass. Public metadata and original MPI2 inventory are refreshed
against that binary (`q558`–`q560`).

PP numerical `q561` still exits with rank-1 SIG11 after all eight devices finish
prefill graph preparation and enter `history_bearing_serial_decode`. There is
no longer an inactive-rank artifact or pre-initialization MPI error. This proves
a separate native preparation defect remains; it has not produced logits.
All three failed numerical driver windows are clean. A fresh diagnostic retains
the exact workload, original MPI2 launch and selected plan while collecting a
GDB backtrace. No PP numerical verdict or full app pass is claimed.

### PP decode preparation: cyclic HIP capture membership

Diagnostics `q563` and `q564` retain the same original MPI2 launch, numerical
workload and selected PP plan. The failing worker reaches normal
`HIPGraphCapture::endNativeCapture` / `hipStreamEndCapture`, then exhausts its
host stack in roughly 125,800 recursive calls to `hip::Stream::EndCapture`.
Neither observation reports a driver fault. This occurs before model arithmetic
or final logits and is independent of the earlier MPI lifetime defects.

The installed HIP binary matches the pinned
[ROCm 10.1 stream implementation](https://github.com/ROCm/rocm-systems/blob/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/hipamd/src/hip_stream.cpp#L51).
It recursively retires parallel capture streams before clearing their membership.
A legal origin→A→B→C→A sequence among three auxiliary streams can create cyclic
stream membership even though the captured operation graph remains acyclic. The
existing RCCL direct-event-wait path can expose that distinction. Removing
valid waits or serializing model execution would change the required execution
contract; the intended repair makes native capture bookkeeping retirement
idempotent by detaching membership before descending.

The first three-stream fixture (`q566`) passes the original SDK: HIP explicitly
protects the origin from membership insertion. It is not a reproducer of this
defect. The revised fixture rejoins a non-origin auxiliary before returning to
the origin, which targets the unprotected membership cycle.

Qualification will combine a small native cyclic-membership reproducer, repeated
capture/replay/reset, and source-bound lifecycle controls, then rerun the actual
PP numerical workload. The first diagnostic wrapper (`q562`) did not launch MPI
because its fresh session directory was absent; `q563/q564` created it explicitly.
Debugger observations are diagnostic only and are not performance evidence.

The revised four-stream native regression reproduces SIG11 on the original
runtime (`q568`, 4.97 seconds, clean driver closure). The canonical user-space
HIP installer then builds and installs the membership-retirement repair (`q569`,
47.40 seconds). Actual-body controls retain 12 failing cyclic cases in the
original implementation and pass all 20 candidate cases, including invalidated
retirement and repeated reentry. The patch preserves captured operation edges;
it changes only cleanup order. The host kernel and GPU drivers are unchanged.

Release `q570b` and Integration `q571` relink successfully. (`q570` itself is a
failed pre-build command with a misspelled tool target, retained separately.)
All five native capture/argument-geometry entries pass in `q572`, including the
unchanged four-stream reproducer on ROCm and the matching CUDA case. Its driver
window is clean. All seven installed-runtime, packaging and Unit entries pass
in `q573`. Focused native aliases and the installed actual-body receipt check
are explicitly registered in production preflight. The full suite is not rerun.
Fresh q570 runtime and numerical recipes retain the original selected plans and
frozen references; no application or new PP numerical pass is inferred.

The real PP rerun `q578` clears both serial decode and ordinary resident forward
materialization on all eight GPUs, confirming the native crash repair in the
original model path. It then rejects the retained restored-prefix MTP bridge:
capacity 15 requires its materialization even while execution MTP is off, but
the terminal binding helper receives the ordinary execution purpose and omits
the required shifted-MTP binding. The original pipeline ownership guard detects
the incomplete input. No logits are produced; native retirement and driver
closure are clean. The next scoped fix must use the existing retained-graph
binding contract, preserving the guard and the off/fixed/dynamic distinctions.

Both DGO transaction binders now use the explicit submission lifecycle for
retained MTP roles. Materialization binds a complete executable transaction
without launching it; ordinary off-mode prefill/decode retains its original
binding policy, and workspace declaration remains nonexecutable. Release
`q579` and Integration `q580` build successfully. `q581` passes both complete
affected Unit owners and both explicit preflight regressions. The real pipeline
edge test covers terminal/follower roles, both backend orders and domain widths
1/2/4/8, including missing-capacity, off-execution and declaration-only rejection
before graph mutation. Fresh public metadata is retained in `q582`; `q583`
reruns the unchanged PP numerical inputs. Application qualification remains open.

`q583` materializes the restored-prefix bridge on all eight GPUs, then rejects
the first stage's 16-row verifier because its original follower checkpoint is
absent. The run retires cleanly with no driver findings. Auditing the retained
family together identifies the same execution/capacity conflation in both
pipeline follower installers, checkpoint-pool initialization, publication
transport, and terminal publication preparation. These are preparation owners;
the live verifier still requires MTP execution to be enabled.

The scoped repair retains the original PMA checkpoint roots and common
participant input/event initialization whenever capacity is retained. Predictor
publication graphs still belong only to the terminal stage. The regression
calls the actual cold checkpoint entry and exercises strict pipeline verifier
edges across disabled/enabled execution, fixed/dynamic policy, two and sixteen
rows, and domain widths 1/2/4/8. Missing and foreign checkpoints fail before
graph mutation with a named invariant. Both contracts have explicit preflight
registrations. Release `q584` passes. Integration `q585` catches a new fixture's
incorrect `ModelConfig` type; changing it to the graph's actual `GraphConfig`
interface yields green `q585b`. All seven focused entries pass in `q586`,
including the three complete affected Unit owners. `q587` reruns the unchanged
q579 frozen numerical intent against the fresh Release binary, retaining new
native/binary evidence in `opencode-final-q584-prep/pp8-numerical-q584`.
No numerical or application pass is inferred from the construction tests.

`q587` completes in 145.85 seconds and passes the unchanged local absolute
pilot: 16 prefill rows and 96 decode rows from the original 16 histories.
Mean final-logit KL is **0.0185063 / 0.0204195** for prefill/decode; top-1
agreement is **93.75% / 96.875%**, and top-5 overlap is **96.25% / 92.0833%**.
The largest row KL is 0.157671 prefill and 0.464968 decode, with no threshold
changes. Reset replay is byte-identical, the production numerical path is
authenticated, and native retirement plus the driver window are clean.
All eight retained graph families prepare successfully; the overlapping
participant setup takes approximately 61.7 seconds. This is a context-1024,
MTP-off accuracy control with capacity 15 retained, not full-context dynamic-MTP,
application, calibrated-corpus, or image certification. The next runtime gate
uses the original selected full-context plan and dynamic MTP policy.

Fresh metadata and inventory (`q588`–`q592`) retain the same two observed switch
groups. Full-context apply `q593` passes at 262,144 tokens. The real dynamic-MTP
server smoke `q594` returns exactly `ready` (18 input, two output tokens), then
retires normally with no surviving descendants or driver findings. The resource
growth gate passes and cleanup reclaims approximately 601 MB of owned prefix
archive storage without reading payload bytes.

Its original verdict remains red because maintenance replay changes the cache's
PerfStats context from `pipeline_original_stage_maintenance_source`, which owns
the 19-node prepared executable, to `moe_rebalance_maintenance_atomic`. The
strict lifecycle checker correctly cannot join these identities. An in-memory
diagnostic changing only those context tags clears the failure, but is not
accepted evidence. The repair must preserve the original materialized context
at the producer and pass a fresh native smoke; the checker is not weakened.

The producer now uses a narrow maintenance-context binding helper: an unbound
cache receives its atomic context, while a materialized source retains its
original context and executable family. Missing or foreign ownership is fatal.
Lifecycle tests exercise the actual capture-plan family across repeated use,
reset, move and rematerialization; the focused alias is registered in preflight.
Release `q595`, Integration `q596` and all four selected checks in `q597` pass.
Fresh full-context PP smoke `q604` passes in 186.80 seconds, including the
unchanged strict graph checker, resource bounds, clean driver window, graceful
retirement, and complete owned-archive reclamation.

The first benchmark launcher attempt `q605` stops before native work: its local
fixed-plan validator incorrectly requires maximum-context qualification to be
false even when the saved plan exactly reaches the authenticated model maximum.
The local validator now accepts the matching true/true maximum-context flags
only at that exact boundary, retaining every other admission check. All twelve
local wrapper tests pass; no production code or numerical policy changes.
`q605b` uses a fresh output directory. TP4's current-binary fixed-context apply
also passes at 47,168 tokens (`q606`).

PP8 benchmark `q605b` completes with clean native retirement and driver evidence.
The three 512-input/256-output samples yield median **1082.630 prefill /
58.409 decode-after-prefill tok/s** (255 decode tokens per timed sample).
Prefill ranges 1081.393–1083.751; decode ranges 58.350–58.616. Dynamic MTP is
enabled with retained depth 15; the greedy benchmark's measured controller is
at depth 3 and reports 12 windows, 576 accepted drafts, and no verifier failures.
This is a short-prompt diagnostic at 262K allocated capacity, not a long-context
throughput or stochastic-generation certificate. The unchanged ten-phase
stochastic OpenCode app lifetime starts in `q607`, with the original full-context
Apply plan and bounded prefix tiers.

### Full PP8 application qualification and TP4 handoff

`q607` passes the complete ten-phase Python/SQLite web-app session on the same
selected Auto PP8 plan: all 512 experts on GPU with
`GateUpOwnedDownColumns`, ngrams sharded across the eight GPUs, context 262,144,
and stochastic dynamic MTP with retained capacity 15. OpenCode 1.18.34 completes
111 inference exchanges and **124 tool attempts with zero tool errors** in
4,869.61 seconds. All seven required tools execute successfully. The generated
app's final test invocation passes 76 tests, and the independent web-app and
final-sentinel acceptance checks pass. A model-authored query-handling defect
is caught by its own tests and repaired during the session; no engine or
tool-protocol failure is observed. There are no generation/session deadlines
and no compaction requests in this run. The final prompt contains 134,671 tokens.

The terminal `/stats` snapshot reports 111 completed requests, no failed or
disconnected requests, and 92,420 completion tokens. Prefix reuse is
7,279,479 / 7,414,150 requested prompt tokens (**98.1836%**), with 110 partial
hits and one miss. MTP performs 1,126 depth updates and accepts 55,266 of 95,440
draft tokens; stochastic acceptance is 76.0067% across 72,712 tests. Session
decode is **27.057 tok/s**, and mean TTFT is **12.553 seconds**. These aggregate
app rates include growing contexts and prefix restoration; they do not replace
the matched short-prompt benchmark. All 6,156 completed HTTP responses,
including observation requests, are HTTP 200.

The strict production runtime contract passes full captured execution,
participant-local sparse attention, maintenance identity, and stochastic MTP
activity. The resource-growth gate passes: peak serving-rank anonymous memory
plus swap is 24,509,546,496 bytes, and observed device allocations remain within
the original admitted bounds. The inactive original MPI rank remains a separately
identified observer. No driver findings occur in 565,731 new records collected
through 2,051 snapshots. Native retirement is graceful, with no forced signals
or surviving descendants.

During serving, RAM prefix storage reaches 137,151,754,720 / 137,438,953,472
bytes, and disk reaches 14,206,323,360 / 34,359,738,368 bytes. Shutdown flushes
retained RAM entries to disk: the final metadata audit sees 34,359,719,424 active
payload bytes, within the 32-GiB budget, with zero orphan payload files. The
harness then removes all 6,660 owned files and reclaims 34,373,722,112 allocated
bytes. Audit and reclamation read **zero cache-payload bytes**. Total adapter
lifetime, including startup and retirement, is 5,159.14 seconds.

Authoritative evidence is under
`/tmp/qfn-qualification-d836-20261112/opencode-final-q595-prep/pp8-runtime-q595/app/`:
`result.json`, `app-validation.json`, `app/stress.json`, `stats-final.json`,
`runtime-contract.json`, `resource-growth.json`, `driver-report.json`,
`native-retirement.json`, `child-custody.json`, `archive-audit.json`, and
`archive-reclamation.json`. The root command receipt is
`/tmp/qfn-full-preflight-after-cpu-next/q607.json`. This is a local Release
application qualification, not a newly minted full-preflight or image certificate.

After complete PP8 retirement, the durable sequence starts TP4 on its unchanged
selected Auto plan and original two-rank NUMA roster. `q608` passes the dynamic
MTP smoke and cleanup in 185.04 seconds. `q609` passes the diagnostic benchmark
with median **616.185 prefill / 93.413 decode-after-prefill tok/s** at context
47,168, using the same 512/256, one-warmup/three-sample, prefix-off greedy method.
Prefill samples are 615.872, 616.638, 616.185; decode samples are 93.430, 93.413,
93.255. The controller reports nine depth updates across 27 windows, 642
accepted drafts, and no verifier failures. Native retirement and driver closure
are clean. The full stochastic TP4 application starts as `q610`; its result
remains pending and no final checkpoint is inferred from smoke or throughput.

### TP4 client terminal completion preempted by compaction

`q610` reproduces an OpenCode 1.18.34 client lifecycle defect, rather than an
inference or parser stall. The second app phase implements the HTML increment,
passes all eight current app tests, and returns a normal tool-free `stop` in
exchange 25. Its total usage is 15,239 tokens. OpenCode caps the advertised
32,768 output budget to 32,000, leaving a compaction threshold of
47,168 - 32,000 = 15,168 tokens. The upstream processor gives proactive
compaction precedence over terminal completion. Exchange 26 therefore compacts;
exchange 27 receives the client's synthetic continuation instruction and starts
unrequested completion-toggle work instead of returning control to the next
harness phase. Repeated later compactions have the same cause.

The harness correctly waits for the actual OpenCode process lifetime. Counting
an intermediate `stop` as phase completion would bypass the client lifecycle
and could mistake a summary for a finished phase. Lowering the output budget or
disabling compaction would change the intended workload. The scoped repair is
to honor a completed, tool-free terminal turn while preserving compaction for
ongoing tool work, real overflow, and the next user turn.

At 05:37:05 UTC, only the authenticated owned OpenCode child receives SIGTERM;
`operator-client-retirement.json` records its PID, start ticks, parent, reason,
and exact reproducing exchanges. The ordinary harness/server retirement path
remains responsible for native cleanup. This run is deliberately unsuccessful
and is not promoted to an application pass. A focused client regression must
fail on the original pin and pass on the repair before a fresh TP4 app run.
The passing smoke, benchmark, and PP8 application evidence remain independent.

The retired run contains 36 completed requests, zero inference failures, and one
operator-caused disconnect. Native retirement, child custody, resources and
driver checks pass. Its archive audit stays within 32 GiB with zero payload
reads; separate operator reclamation removes the retained failed-run cache
(34,367,676,416 allocated bytes), preserving all diagnostic artifacts.

The strict final runtime check also finds an independent observation defect:
the TP4 maintenance family has a real 19-node initial capture and 36 replay
observations, but no initial `decode_graph_phase` capture observation. Its cache
does not bind `replay_workload`, so the executor suppresses that phase while the
direct replay producer emits subsequent phases. The repair binds the singleton
maintenance transaction geometry before actual capture, preserving the original
family/context and rejecting changed sealed identity. This geometry describes
one controller transaction; it does not change live expert payload rows or
launch dimensions. PP source preparation retains its no-launch lifecycle.

Release `q611` and Integration `q612` build successfully. Four affected host
registrations pass in `q613`; twelve OpenCode Unit registrations pass in `q620`.
The native executor regression builds in `q619` and passes on both CUDA and ROCm
in `q621` with a clean driver window. Its original missing-workload control
genuinely captures but lacks the phase; the fixed control observes the actual
initial capture and three byte-identical reset/replays, with unchanged identity
and exact counters. Both native aliases are explicitly registered in production
preflight. The checker is unchanged, and no capture counter is synthesized.

OpenCode's ten-line production repair is pinned as `1.18.34-llaminar.1`, from
upstream commit `aec0b9a6d8898f68f923aaf08b7306d931fd9d76`. The suite installer
builds authenticated source with pinned Bun and the source-owned model-catalog
fixture; only the explicitly configured local provider is qualified. Build tools
stay in a separate Docker stage. Actual native CLI controls demonstrate the
original terminal-stop failure and the repaired exit, next-user-turn compaction,
tool-call/stop-with-tools compaction, and preserved genuine overflow error/exit.
Requests retain `max_tokens: 32000` under the configured 32,768-token maximum.
The original upstream production-loop regression fails; the patched prompt suite
passes 61 tests, with its existing disabled-projector skip separately retained.

The final installer qualification and atomic publication pass in `q622`.
The source-fixture-built binary is
`/tmp/qfn-qualification-d836-20261112/opencode-terminal-stop-client/opencode`,
with provenance and all four native controls in its sibling `receipt.json`.
Original and patched controls remain under
`/tmp/opencode-terminal-stop-candidate/`; the first experimental build used a
different catalog snapshot and is not the final installed client. The retry
uses the same selected Auto topology and numerical controls, freshly bound to
the `q611` Release binary. No full preflight or new image certificate is claimed.

Fresh fixed-context admission `q624` passes at 47,168 tokens in 15.57 seconds.
The full ten-phase TP4 retry starts as `q625` under
`opencode-final-q611-prep/tp4-runtime-q611/app/`. It uses the same all-GPU expert
placement, RAM ngrams/local GPU caches, original MPI roster, dynamic MTP and
unbounded generation/session lifetime. The current smoke and benchmark remain
retained controls; this observation-only engine repair changes no inference
arithmetic or live payload policy. Its final outcome is recorded below.

### Leading phase compaction exposes a harness authentication assumption

`q625` retires normally after four successful client phase exits, but its app
verdict remains red. The fourth phase begins with a valid context-summary
request (exchange 32), followed by the resumed coding request (exchange 33).
That coding request contains the exact phase prompt as its penultimate user
message, followed by OpenCode's pinned synthetic continuation message. The
harness authenticates only the first request's latest user text and therefore
compares the intended coding prompt with the summary request instead. It stops
after the phase despite the model having completed the intended HTML changes.
This is a harness defect; the failed run is not reclassified as a ten-phase pass.

The narrow repair will inspect the first ordinary coding request after
structurally recognized leading compactions. Exact prompt identity remains
mandatory; only the pinned client's adjacent synthetic continuation suffix is
permitted. Arbitrary history search, later-request matches, missing or changed
prompts, and compaction-only sequences must fail. Existing protocol auditing
continues to validate every summary and tool exchange independently.

The native server completes all 51 requests with zero failures or disconnects.
All 46 attempted tools succeed. The strict maintenance capture, graph, transfer
and stochastic MTP audit now passes on all four participants. Resource, driver,
process retirement and native-output checks also pass. Session decode averages
48.198 tok/s, uncached prefill 321.256 tok/s, mean TTFT 8.067 seconds, and prefix
token reuse is 455,742 / 563,661 (80.8539%). These partial-session observations
are not substituted for full app qualification or matched benchmark evidence.

At 06:23 UTC, after independently verifying the saved native and child custody
identities have retired, operator reclamation removes only the owned failed-run
prefix archive: 3,578 files and 34,366,038,016 allocated bytes. Its metadata
audit passed the 32-GiB active-payload bound with no orphan blocks, and neither
audit nor reclamation reads cache-payload bytes. The original result, complete
transcripts, generated app, runtime audit and cleanup receipt remain under
`opencode-final-q611-prep/tp4-runtime-q611/app/`. No engine binary or arithmetic
changes are needed for this harness repair.

The repaired authenticator passes all 42 harness Python tests, including strict
direct transport and adversarial compaction-prefix/suffix cases. An independent
review confirms the exact retained transcript shape and unchanged final phase
coverage rules. `/tmp/opencode-q625-phase-prompt-replay.json` preserves the
original rejection and repaired exact-prompt result; altered phase bytes fail.
The new `V2_Integration_OpenCodePhasePromptCompaction` regression is explicitly
registered in production preflight. Configuration `q626` and all 16 selected
Unit/preflight registrations in `q627` pass (three seconds of test wall time).

Because context provenance also pins the harness input, a fresh fixed-context
probe runs after this edit: `q629` passes in 14.88 seconds at the same 47,168
tokens, with no source rebuild, replanning or benchmark repetition. `q628` joins
the actual successful prerequisite receipts, preserving numerical and benchmark
controls as historical observations. A fresh complete app starts as `q630` in
`opencode-final-q611-prep/tp4-runtime-q630/app/`, with the same selected Auto
plan, native Release binary, original two-rank NUMA roster, dynamic MTP and
full output/cache budgets. No generation or session deadline is added.

During `q630`, the UI phase repeats file reads and design work across several
compactions before its first edit. Read-only diagnosis finds no additional
usage-accounting defect: wire requests contain `reasoning_content`, and the
actual Qwen3.8 GGUF template preserves historical thinking when
`preserve_thinking` is unset, as it is in this path. Summary requests also include
assistant reasoning. Exchange 53 totals 11,922 input + 3,323 output = 15,245
tokens; exchange 57 totals 14,464 + 6,027 = 20,491. Both exceed the client's
15,168-token input threshold after reserving its 32,000-token output maximum.
The model eventually performs the UI-module write in exchange 61 (8,575 output
tokens, with 7,445 characters of tool arguments). This is evidence of conservative
context management and slow model planning, not a missing engine response or
miscount of discarded thoughts. No further patch, reduced output budget,
synthetic phase completion or time limit is introduced. The diagnostic receipt
is `/tmp/qfn-q630-compaction-accounting.json`; app qualification remains pending.

A second read-only check diagnoses phase 5's prolonged lack of edits. Resumed
inputs after summaries are only 9,010–10,699 tokens, often with no old tool
outputs at all. The model then repeatedly rereads seven files: one observed
batch adds 11,199 input tokens (10,699 → 21,898), including 32,866 characters
of fresh tool results. This exceeds the 15,168-token input allowance before
additional reasoning. The recent-tail budget is adaptive (3,792 tokens); fixed
40K/20K pruning constants are inactive because pruning defaults to disabled.
No oversized retained-history or token-accounting defect is demonstrated.
`/tmp/qfn-q630-compaction-retention.json` retains these exact observations.

The run remains live and unchanged with four completed phases and no native
request failures. Its phase-progress gate is not green. A concrete, unapplied
option in `/tmp/qfn-opencode-context-split-proposal-20261113.json` uses OpenCode's
existing `limit.input=47168` and `compaction.reserved=16384`, giving 30,784 tokens
of input room. The configured 32,768 output maximum (32,000 on the wire) remains,
but only 16,384 generation slots are guaranteed near that input threshold.
The largest completed response observed when proposing this option was 8,575
tokens; that does not prove future replies fit. Because the option changes the
guaranteed reasoning allowance, the user has been asked before applying it.
No workload, context, client binary, native source, or test verdict is changed.

### Retiring the nonprogressing TP4 attempt and auditing summary delivery

At 07:45:59 UTC, only the exact owned OpenCode CLI (PID/start identity recorded
in `operator-client-retirement.json`) receives SIGTERM after phase 5 repeats
64 file reads, six task-list updates and four searches without any write/edit.
This is an observed progress loop, not a generation deadline. Four phases
completed normally in 119.37, 795.33, 480.78 and 1,192.63 seconds; phase 5 was
interrupted after 2,022.35 seconds and is not certified. The full app lifetime
is 4,626.87 seconds and root adapter `q630` returns 1 after 5,078.06 seconds.

The final server counters show 113 completed requests, zero failures, and one
operator-caused disconnect. All 176 tool attempts report successful execution;
22 compactions occurred. There are 132,184 completion tokens; session decode
is 45.489 tok/s, uncached prefill 301.134 tok/s, and mean TTFT 14.900 seconds.
Prefix reuse is 874,947 / 1,347,794 tokens (64.917%). Dynamic MTP records 1,485
depth updates and 87,542 accepted drafts; stochastic acceptance is 76.834%.
These are partial-session diagnostic observations, not a completed app gate.

Native capture/transfer/dynamic-MTP, bounded resource growth, driver and exact
process-retirement checks all pass. No native process is forcibly killed.
The final native-output audit remains incomplete because the last HTTP response
was intentionally interrupted; it is not rewritten as a passing verdict.
Shutdown persistence stays within the 32-GiB active-payload bound. At 07:54 UTC,
separate operator reclamation removes only the retired archive's 5,886 files,
reclaiming 34,381,742,080 allocated bytes. Audit and reclamation read zero
cache-payload bytes; all transcripts, client work and original red receipts remain.

The complete harness audit also reveals 22 false missing-tool-history failures
at earlier compaction boundaries. Each emitted call has an exact completed
client outcome; the compaction request contains its actual call/result
serialization, and the resumed coding request contains the actual summary.
The checker incorrectly insists that the structured call ID survive afterward.
`/tmp/qfn-q630-compaction-tool-join-diagnosis.json` binds all 22 IDs to those
requests. A narrow repair must authenticate that entire summary-delivery chain,
keep direct structured joins strict, and preserve the final seven unfinished
call obligations caused by operator interruption. No whole-session pass may be
inferred by correcting these earlier false positives.

### Authenticated tool-summary delivery repair

The repair binds each pending call to a unique native completed/error outcome,
matches its typed JSON arguments and exact serialized result in the actual
compaction input, and requires the exact resulting summary in the next coding
request's canonical leading summary pair. Consecutive summaries must carry the
authenticated previous summary. Direct structured history joins remain strict;
a partially present or changed call ID cannot use the summary path. Distinct
identical calls consume distinct transcript occurrences.

The pinned client truncates tool outputs at 2,000 UTF-16 code units, including
possible surrogate boundaries. Regressions cover that behavior, emoji, numeric
JSON equivalence without boolean aliases, nested key ordering, error results,
attachment filenames, failed summaries, reset of pending summary chains and
interrupted continuations. The client's transcript does not escape role labels:
the bridge proves independently authenticated content reached the summarizer,
not original role provenance from those unescaped labels.

All 49 harness Python tests pass. Configuration `q632` and all 17 selected
Unit/preflight registrations in `q633` pass in 2.97 seconds. The focused
`V2_Integration_OpenCodeCompactionToolSummaryBridge` entry is explicitly in
production preflight. Actual q630 replay removes exactly its 22 false failures,
leaving all seven pending calls plus the proxy/operator-exit errors red.
`/tmp/qfn-q630-compaction-tool-join-replay.json` records that comparison;
original q630 artifacts and its incomplete phase verdict remain unchanged.

### Rechecking TP4 context capacity before reducing reasoning allowance

The user questioned the context limit instead of approving the proposed smaller
reply reserve. No reserve, precision, placement, MTP or cache policy is changed.
The context was 47,168 total tokens, not 32K; OpenCode's 32,000-token reply
reserve leaves 15,168 tokens before compaction. The q528 search had proved
47,168 accepted / 47,232 rejected. Later fixed-context checks alone do not
re-establish a maximum after subsequent code changes, so `q634` runs the existing
PMA context selector again against the q611 Release binary and current source.

The original selected Auto BOM in `q523.log` exposes this per-MI50 breakdown:

| Physical owner group | GiB per device |
|---|---:|
| Model, expert and MTP weights | 25.6890 |
| Execution workspaces | 2.1227 |
| Main and shifted KV caches | 1.1696 |
| Recurrent live/rollback state and sequence metadata | 0.7965 |
| Captured/preparation graphs | 0.4922 |
| Weight-loading and expert-movement staging | 0.6820 |
| Prefix GPU staging and hot tier | 0.2772 |
| Activations and collectives | 0.4181 |
| Native runtime and immutable lookup | 0.2610 |
| **Total / admission-available** | **31.9082 / 31.9082** |

These are original admission owners, not independent live allocation readings.
The small adaptive ngram cache consumes the remaining admitted capacity. q630's
live resource peaks were about 31.79–31.87 GiB per device; its retirement-time
free bytes cannot establish inference headroom because topology cleanup has
already released owners. No duplicate main/MTP KV charge has been demonstrated:
committed main and shifted predictor caches are separate runtime objects.
Detailed read-only extracts are `/tmp/qfn-context-memory-review-20261113.md` and
`/tmp/qfn-context-memory-components-20261113.json`. The refreshed boundary and
any concrete reclaim opportunity must be established before changing budgets.

`q634` completes all 14 current-binary probes in 138.36 seconds, confirming the
same 47,168 accepted / 47,232 rejected boundary. Its native and driver owners
retire cleanly, with no findings. A narrow read-only workspace audit identifies
unproven candidates involving FP32 attention temporaries (184.25 MiB) and
request-major gather scratch (up to 92.125 MiB). Those figures are inferred
geometry, not proven reclaimable allocations; runtime consumers and planner
contracts would need coordinated qualification. No memory or arithmetic policy
is changed, and that investigation is deferred in favor of the app milestone.

After the confirmed capacity result, the user approves the proposed 16K reply
reserve. The harness gains optional `--reply-reserve-tokens`; an explicit value
sets both `model.limit.input` and `compaction.reserved`. Default CI calls retain
the native client policy. The output maximum remains 32,768 configured / 32,000
on the wire. Near the 30,784-token input allowance, only 16,384 context slots
remain for generation; the larger maximum is available with shorter inputs.
Campaign/session artifacts publish the exact policy. The independent app
validator requires the lifecycle owner's explicit matching reserve and rejects
an override supplied to an unchanged default-policy CI call.

All 52 harness tests and nine certification-evidence tests pass. `q635`
configures the new explicit preflight entry; all 19 selected Unit/preflight
registrations pass in `q636` (3.19 seconds). Twenty-nine local adapter tests
also pass. The unchanged installed native client is exercised against a scripted
loopback provider: usage 22,000 and 30,783 causes no next-phase compaction;
30,784 and 30,785 causes exactly one. All eight CLI turns exit normally, all
ten wire requests retain `max_tokens=32000`, and no client errors occur.
This client-only evidence is in
`/tmp/qfn-qualification-d836-20261112/opencode-reserve16384-diagnostic/receipt.json`;
it is not a real-model pass.

`q637` joins the actual successful receipts; `q638` refreshes fixed-context
admission and current harness bindings in 15.56 seconds. Fresh app `q639`
starts at 08:19 UTC under `opencode-final-q611-prep/tp4-runtime-q639/app/`.
It retains the same selected Auto topology, all-GPU expert placement, original
MPI roster, dynamic MTP, arithmetic and prefix-cache budgets. The only workload
policy change is the approved client reserve; there is no generation/session
deadline. The earlier q630 failure remains intact. At launch, commit/push was
held for the complete TP4 app milestone; the subsequent user-directed stopping
point and its narrower evidence are recorded below.

### TP4 user-accepted stopping point and normal retirement — 2026-11-13

The user explicitly accepted the observed TP4 coverage as sufficient for now
and requested wrapping up the OpenCode test. Only the authenticated client
PID/start identity received SIGTERM, through a pidfd, at 09:12:32 UTC. The
existing harness retained responsibility for evidence collection, administrative
server shutdown, prefix persistence and process retirement. No generation
deadline was introduced, and no native process was signalled or forcibly retired.

Eight phases completed normally: bootstrap, task API, CRUD, HTML, Unicode/emoji,
input validation, filtering/pagination and persistence. The ninth, concurrency,
was in progress; the final verification phase did not run. The app's existing
65 tests passed after adapting their callers to its revised SQLite connection
API. Newly added concurrency tests were still being repaired at interruption,
so this is not a final all-tests-passing application snapshot.

| Recorded observation | TP4 q639 |
|---|---:|
| Client session elapsed | 3,057.37 s |
| Root lifetime including admission and retirement | 3,414.37 s |
| Normally completed phases | 8 / 10 |
| Completed / failed server requests | 137 / 0 |
| Disconnects caused by the requested stop | 1 |
| Completed tool attempts / tool errors | 131 / 1 (0.763%) |
| Completed / interrupted compactions | 4 / 1 |
| Prefix matched / requested tokens | 2,613,447 / 2,830,180 (92.342%) |
| Aggregate decode | 48.373 tok/s |
| Uncached prefill / effective prompt processing | 204.870 / 2,675.264 tok/s |
| Mean TTFT, all terminal request observations | 8.682 s |
| Dynamic MTP depth updates | 704 |
| Accepted / drafted tokens | 51,463 / 97,148 |
| Stochastic accepts / acceptance tests | 51,463 / 62,462 (82.391%) |
| Driver records inspected / findings | 308,558 / 0 |

These session aggregates include the requested interruption and changing prompt
lengths. They are distinct from the matched 512/256 benchmark medians above.
The one tool error was an ordinary README edit-match miss from which the model
recovered. The four complete summaries were followed by resumed coding work;
the fifth summary request was interrupted by the user-directed stop.

Native runtime/capture/MTP, physical-fabric selection, resource growth, driver
observation, graceful native retirement and child custody all pass. The resource
gate has 601 samples with no failures; serving RSS ended near 40.08 GiB, with
bounded VRAM and prefix-cache use. HTTP responsiveness also passes with 3,046
snapshots, 2,595 during active inference, zero failed polls and a maximum observed
HTTP duration of 0.1262 s. The topology remains the same genuine selected Auto
plan on ROCm 1–4 behind their common switch, with all-GPU
GateUpOwnedDownColumns experts, RAM ngrams, dynamic MTP and the approved explicit
16K reply reserve. No placement, precision or native execution policy changed.

Strict original receipts are preserved: `q639` returns 1, the ten-phase stress
gate is red, and the native-output audit stops at interrupted
`exchange-0137.json`. The report also retains the unfinished continuation,
missing final sentinel and `glob` coverage, and unfinished authored concurrency
tests. User acceptance of this stopping point does not relabel those results
as a full application, image, preflight or native-output certificate.

The native owner exits 0 after normal shutdown and RAM-prefix persistence, with
no forced cleanup or survivors. The archive audit finds 2,918 active payloads,
34,359,707,168 active bytes within its 34,359,738,368-byte budget, and no orphan
payloads. After authenticated retirement, the existing metadata-only reclaimer
removes 2,921 files and frees 34,369,908,736 allocated bytes. Audit and reclamation
read zero payload bytes. Logs, transcripts, generated application and original
red receipts remain available.

Evidence is under
`/tmp/qfn-qualification-d836-20261112/opencode-final-q611-prep/tp4-runtime-q639/app/`:
`accepted-stop-summary.json`, `operator-client-retirement.json`,
`runtime-contract.json`, `selected-fabric.json`, `resource-growth.json`,
`driver-report.json`, `native-retirement.json`, `child-custody.json`,
`archive-audit.json` and `operator-archive-reclamation.json`. The retained PP8
`q607` full ten-phase app remains green. Both retained short numerical panels
remain accepted in their recorded scopes. No further TP4 run is scheduled;
source changes are not committed or pushed by this wrap-up.
