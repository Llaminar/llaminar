# QFN participant-local hot caches ahead of shared ngram capacity

Implementation and qualification plan, 2026-11-06. This is a project plan, not a
tuning dashboard. The scope extends the existing pooled lookup implementation;
it does not replace the cache policy, graph executor, transport or memory
authority. The base checkpoint is `f0d4ba9a8`; the focused current-demand extension is
recorded in [its qualification note](QFN_POOLED_CURRENT_READAHEAD_FOCUSED_20261106.md).

## Outcome and current boundary

The requested hierarchy is **local VRAM cache → shared GPU pool → RAM → SSD**.
Each actual PLE consumer can retain immutable original-format rows locally,
while the shared pool owns miss service and its independently distributed cache
capacity. Expert-only followers may own shared capacity without becoming PLE
consumers. An immutable replica owns its own physical bytes, not another
authority over the source table.

`ConsumerLocal` and shared-only `DeclaredRankLocal` remain comparison policies.
The shared-only replicated-reader contract delivers expanded FP32 lookup output;
the combined extension delivers compact native misses and retains local hits.
Current-demand early overlap is implemented. The future-source components now
separate speculative lookup from actual demand credit, but complete model
future scheduling remains disabled until its retained source transactions and
graph families are wired and qualified. Existing owner/source/MTP passes alone
do not qualify model future read-ahead.

The current-demand slice is checkpointed on `feat/qfn-ngram-continuation` at
`6ebdb5019`. Its four fresh numerical panels pass the required groups. The first
natural-history Release comparison stopped before inference because the declared
diagnostic benchmark omitted the paired numerical profile's disabled prefix-cache
policy and opened the default disk archive. That failed run is retained at
`/tmp/qfn-declared-benchmark-command-prep-r785/shared-varied4-execute-r1`;
it provides no timing result. The paired diagnostic adapter now carries through
the authenticated observer's explicit disabled-prefix setting and checks the
complete prefix policy before launch. The original exports and accuracy panels
are unchanged. All three short natural-history campaigns now complete eight
workloads each with identical generated tokens and clean native/driver retirement.
The saved shared run required a separate postvalidation receipt after fixing
stale passive telemetry assumptions; its original failed validator report is
preserved. The owning Python suite passes all 250 tests. The repair validates
physical expert-bank populations according to apportioned, tensor-sharded,
replicated and phase-aware domain policies, including zero-capacity banks and
registry-global participant IDs. Evidence and the matched rates are in
`/tmp/qfn-declared-benchmark-command-prep-r785/comparison.json` and the sibling
Markdown table. This short ROCm TP2 plus CPU-capacity workload favors
ConsumerLocal; combined hierarchy overhead currently outweighs its smaller
native-source traffic. It is not a cold-SSD or lookup-latency measurement, and
no default policy is promoted. Longer prompt/scan coverage and phase attribution
remain performance obligations.

The [coordination and fusion investigation](QFN_LOCAL_HOT_CACHE_FUSION_20261106.md)
records the matched rates, actual traffic, compact coordinator bound repair and
separate local-probe/shared-union fusion candidates. Internal kernel boundaries
may be replaced by block synchronization where the original policy and ordering
remain exact; actual source, transfer and publication dependencies must remain.

Future read-ahead remains a separate implementation slice. The source
lookup-only, complete-demand adoption, producer/consumer lifecycle and borrowed
next-prefill-window components are now integrated for a coherent focused build;
actual model scheduling is still disabled. Speculation uses the existing frame
and response banks and earns no replacement-policy credit before real demand.
The 41 focused component cases pass, including real source delivery after
adoption, pressure/discard/reset safety and large-to-small demand replay on CUDA
and ROCm. The component receipt is
`/tmp/qfn-future-component-integration-r789/component-qualification.json`.
The next-window requirement is an immutable retained-family contract checked at
every request ingress; a proof about one request cannot authorize a later request
that reuses the same pointers. No new token/count cursor is introduced.

Model enabling still requires ordered finite auxiliary bodies and one original
source plan covering their genuine source occurrences. First-current plus next-
future executes two source lookups; adopted-current plus next-future executes
one. The original remote owner must perform that exact count inside one selected
complete retained transaction, with canonical admission for any additional
retained executable. Future Hash/Probe must join both prior local retirement and
accepted PLE history, while transformer work can proceed independently. This is
a bounded extension of the existing graph/source owners. Native component
qualification, actual model overlap, numerical panels and Release benefit remain
separate obligations; component passes do not advertise model read-ahead support.

| Slice | Implementation | Functional qualification | Performance qualification |
|---|---|---|---|
| Existing ConsumerLocal | Installed | Existing scoped evidence | Existing measurements; remeasure matched controls |
| Existing shared pool | Installed; hosted verifier and ordinary retained identity repairs | ROCm TP2 fixed3/fixed1 exact replay plus current-demand fixed1 and ordinary reset/reuse pass; see focused note | No new matched comparison in this slice |
| Local-cache configuration and PMA roots | Implemented in the integration tree; independent ceilings, C0, overrides, exact ready-page ownership | Focused parser/document/admission/ownership and runtime-projection registrations pass | Not a runtime measurement |
| Combined local hot cache and shared pool | Native compact delivery, current-probe parent dependency, model wiring, terminal counters and once-request admission integrated | CUDA/ROCm native coverage spans 23 formats; focused ordinary/fixed1/fixed3/dynamic checks pass. Real TP2 held-source cases prove local-hit completion independently of a missing peer's owner service on both backends. Six-GPU C0 and positive-cache configurations pass ordinary generation with expert movement and fixed1 exact replay. Six-GPU C0 source recreation and final retirement pass. Fresh CUDA combined and ROCm refined-combined/shared-only/ConsumerLocal numerical panels pass every required group | Pending; no default promotion |
| Native compact responses and unequal consumer miss sets | Compact request/union and native completion bindings implemented; actual delivery graph under integration | Compact-consumer, request-publication and native-request-binding CTests pass; native transport qualification pending | Pending |
| Small local probe fusion | Integrated; four launches become one for at most 512 original IDs | 32 distinct focused native cases pass, including every small token count, both key widths and C0; compiler spill guards pass for all six shipped GPU targets. Fresh CUDA/ROCm numerical panels retain all 178 baseline row metrics exactly | Matched QFN-sized warm graph latency improves about 3.7% on CUDA; ROCm changes are near run variation. Combined probe/bound/union Release run passes; +1.34% prefill/+0.85% decode is too small to establish a firm gain from one historical pair |
| Shared query launch bound and small union fusion | Integrated as separate changes after the local-probe component pass; existing storage and transport contracts retained | Bound-only 11-case and union 10-case host/CUDA/MI50 selections pass; gfx1100 adds 14 passing native cases; no skips or new driver errors. Fresh numerical panels pass | Eight-request matching MI50 Release campaign passes with exact generated tokens; no standalone union speedup or default promotion claim |
| Real token-trace measurement | Exact GGUF/PLE trace exporter and metric contract implemented; native IQ4_NL source extent corrected | Parser and existing trace regressions pass, with the old parser failing the same regression. Two saved histories now validate their actual 90-byte native rows; oversized 320-byte exports reject | Two short histories provide limited warm/history inputs. Long-scan and speculative captures, actual combined-runtime timing and matched controls remain required |

The first new model projections are the existing ROCm TP2 RAM-backed,
StaticOrdinal owner-subset cell with an explicitly engaged **zero-capacity local
cache**, in ordinary and fixed-depth-1 modes. They keep the original source,
native transports, expert placement and exact accounting witnesses. This tests
the actual compact-miss path before optional local hit behavior; registration
alone is not a passing model result. The first real C0 admission rejected an
unsealed compact query during graph declaration. Physical producer addresses
are sealed during launch preparation; cold declaration checks must validate
ownership and geometry without requiring that later seal. Native execution
continues to reject unsealed or changed addresses. The native CUDA and ROCm
regressions now pass with original prepared source workers and clean retirement.
The next model run passed that boundary and exposed an independent child-export
coverage error: the main child graphs were incorrectly required to contain the
separately recorded auxiliary body. Export now validates the exact main-stage
projection and the original auxiliary declaration; the complete parent must
still attach that body before becoming executable. The exporter and the following
MoE parent composer now use this same declared projection. Focused regressions
reproduce both original coverage errors and pass with the fixes, including
missing, duplicate, reordered, foreign and prematurely executable child rejection.
The clean source-admission run loaded all 48 layers, retained the original
28,800,138,240-byte table, and proved runner/model retirement, prefix-file removal
and retirement of both GPU generations. It executed no numerical inference.
One earlier setup result is explicitly excluded because a private compile
command contaminated shared objects during linking; the affected objects were
rebuilt normally and six focused regressions rerun before the accepted admission.

The five cache-policy profiles and lossless public configuration export are now
integrated: 10 source-admission, five ordinary-generation and five fixed1 exact
cases are discovered. Three focused configuration/budget gates and all 31 Python
inventory tests pass. Their optional caps total 128 MiB, but this is not a claim
of matched complete physical footprints. The benchmark helper obtains those
footprints only through the canonical planner; its current matched result is
fixture evidence and still needs the original runtime planning inputs.
A downstream executable-template export check now uses the same main/auxiliary
projection while retaining the full stage identities and actual body attachment;
the normal build and all seven affected graph/cache gates pass. Fixed-depth-1
source admission also passes in 221 s with clean driver state, no skips and
complete retirement. The separate fixed1 native exact run passes in 176 s:
1,208 stage/row comparisons across the original checkpoint and three prefix
positions are byte exact, and the real verifier and acceptance witness match
serial tokens. This is native serial/grouped equivalence, not HF KL scoring.
Combined equal and unequal positive-cache ordinary runs also pass two public
requests with reset/reuse in 122 s and 124 s respectively. Those runs predate
the new local terminal counters and do not yet prove a hit-rate or speed gain.

Terminal reporting now adds fifteen local hit, consumption, churn and actual
receive-byte totals to the existing whole-request metadata copy. The original
local storage owns cumulative counters and the completed-observation baseline;
runner replacement or request reset cannot double-count the previous window.
The controller layout is unchanged. A separate exact arena buffer and complete
pinned destination are admitted by PMA. Focused model assertions require every
original consumer's complete, correctly attributed tuple, including zero-capacity
and native-byte bounds. The normal build compiles all shipped CUDA and HIP
architectures. All eight native cache gates pass on CUDA and ROCm, including
23 row formats, C0/positive capacity, wide row identities and terminal reporting.
The first instrumented unequal model run completed both requests with clean
driver state but was red: its retained-source assertion incorrectly equated
source/ready leases with the whole CPU staging category, omitting the separate
runner terminal leases. The corrected assertion authenticates the source subset
and requires exact complete backing of the original final PMA owner; it does not
relax equality or substitute a planned reservation for live bytes. Four focused
accounting/configuration gates pass. The corrected real model run passes in
129 s, with clean driver state and no skips. Its complete terminal proof shows
all 16 repeated-request row IDs served locally by the positive-cache consumer,
with zero native row payload and bounded control traffic; the C0 participant
still services 16 shared-pool hits. These are functional and traffic results,
not a Release timing comparison. The legacy source/pool counters here describe
the final request, while local counters explicitly distinguish initialization
and subsequent terminal windows; they must not be added across unlike scopes.
Six additional original R3 requests cover C0/equal/unequal with fixed3 and dynamic
MTP; their configuration/discovery checks pass. Equal-cache ROCm fixed3 now
passes in 196 s: 2,416 verifier stage/row comparisons across four checkpoint
positions are byte exact at four physical rows, and the actual depth-3 verifier
and acceptance witness match serial tokens. Unequal-cache ROCm dynamic MTP also
passes in 207 s with 2,416 exact stage/row checks, three controller-window
increments, nine attempted draft tokens and four verifier transactions in the
serial-token-exact dynamic witness. Both runs have clean driver state and no skips.
Source admission now also exports the original final PMA's exact resource and
owner costs as separate CSVs. These are passive observations, not another planner
or a claim that the functional profiles already have equal total footprints.
Four original CUDA TP2/RAM requests now provide C0/equal-capacity source,
ordinary and fixed1 candidates (eight discovered registrations). Their source,
CPU expert-capacity tier and predictor envelope are unchanged; the device-free
projection gate passes. Equal-cache CUDA ordinary generation/reset/reuse passes
in 130 s with clean driver state and all terminal assertions. On the repeated
request both original CUDA consumers serve 16/16 local hits without row payload
transfer. CUDA fixed1 also passes in 189 s with 1,208 exact stage/row checks.

The rank-local all-GPU parent extension now retains the original immutable
ExpertOverlay descriptor and authenticates graph-only Probe/main fragments
before composing one executable with its finite auxiliary body. No CPU expert
tier or independently replayed child is introduced. The ordinary enum-only
capture constraint remains strict; the existing subsequent graph-aware builder
installs the authenticated composition. A coherent build and 53 focused
ownership/graph/native gates pass, including CUDA and ROCm delayed-auxiliary
diagnostics with the actual controller/composer. The subsequent real singleton
source test uses the original compact/coalescer/provider/native delivery and
actual Probe/LocalRead stages. CUDA and ROCm each pass forty held-owner all-hit
replays over two request admissions, with mixed-miss exact output, native extents,
source-service totals and complete retirement. The whole focused selection passes
21/21 with clean drivers and no skips in
`/tmp/qfn-goal-continuation-r729/local-hot-real-source-native-r1`.
This does not yet prove unequal multi-consumer delayed-owner behavior. The
stronger original-source counter checks now pass on CUDA and ROCm in 5.13 s,
covering exact hit/consumed IDs, promotions, zero evictions and zero speculative
credit. Four real TP2 held-source candidates invert the C0 and positive local
consumers. Their first CUDA run exposed a test-only use of forbidden host logits;
the corrected test compares the already-enabled canonical LM_HEAD snapshots.
All four CUDA/ROCm TP2 candidates now pass with either consumer owning the
positive cache (140–143 s per model case), clean drivers and full retirement.
Across eight held-source repeats, the positive consumer reaches
LocalReadComplete before source release while its C0 peer and whole request
stay pending; the final original LM_HEAD bytes match exactly after both resets.
This establishes original Rank/cohort causality, not a throughput or numerical
panel result. Evidence: `held-source-four-native-qualified-r779.json` under the
root focused result directory.

The native leaf fixture also has a focused 1024-ID candidate with live sizes
512, 528, 1024, 16 and zero, byte-exact inverse/miss extents, dirty tails and a
four-slot consumed scan. CUDA and ROCm both pass the original FP32/IQ4_NL scan trace (7.92 s
and 6.67 s). A separate full-width-key companion passes both native backends
(3.04 s CUDA, 1.02 s ROCm) with the same 1024-ID capacity and two request
admissions. These use the production cache kernels and a recorded response
producer; they do not certify the shared transport or whole-model performance. An additional tiny sparse-key pressure
trace has a documented approximate-frequency collision/protected-victim
limitation. It is not a key-truncation fault, and does not justify claiming
universal hot-row retention or changing the existing policy without realistic
capacity measurements. Detailed evidence is retained under
`/tmp/qfn-local-hot-sparse-policy-trajectory-r774`.

Six-GPU admission exposed two setup defects. Both CUDA continuation participants
had prepared engines but no published banks because graph declaration was
deferred. The follower constructor now uses the existing original-context-bound
registry finalizer before its unchanged cohort readiness check. The focused
two/six/eight-participant lifecycle test passes twenty repeats, and actual source
admission passes that boundary. It next exposed role-specific implicit capture
labels (`pooled/local_ready` versus `pooled_reader/local_ready`). Split units now
retain explicit transaction identities or shared semantic frontier names. The
new asymmetric-name regression fails against the old controller and passes with
the fix. The next actual six-GPU source run passes both repaired boundaries,
then rejected an associated reader during finite-source projection: it has a real
coordinator attachment but is neither the source owner nor source-free. The
repair now authenticates the reader's original sealed attachment and preserves
the single original source owner; both focused source-family regressions pass.
The next actual run passes that boundary and exposes a retained-capacity mismatch:
with MTP initially off and three drafts retained, followers prepare the admitted
row family while the source prepares only active-policy rows. The repair now
materializes the admitted source family and original eligible Full/Chained
predictor owners without launching model work. Live MTP remains disabled for an
Off request. Focused original-code controls reproduce the missing association.
The next native six-GPU run passes initial admission, but runner recreation
exhausts the existing CPU opaque-graph reservation because new source-owner
definitions obtain new process-lifetime slots. The repair is to reuse the
original source's bounded, PMA-priced owner slots after complete native
retirement; opaque allowances remain unchanged. Ordinary six-GPU generation
also completes both requests, but the three-/two-token fixture cancels expert
movement before completion. Dynamic cells now select their existing sustained
384-token workload; the original movement assertions remain mandatory.
The source-slot repair now retains the same native wrapper and opaque owner
through twenty proven cold definition resets; its focused host lifecycle and
original-code failure control pass. The actual rerun passes that boundary, then
exhausts the original follower capture-terminal event population on recreation.
An enriched diagnostic confirms program kind 5, one event, ROCm 0. The installed repair retains the original physical event population while
each runner receives fresh protocol/receipt state; opaque allowances stay
unchanged. Six normal owning/capacity registrations pass with clean drivers
in 7.84 s (`terminal-recreation-owning-focused-r1`), including twenty source
recreations per backend and partial-creator cleanup. The actual six-GPU rerun
passes that event boundary, then finds duplicate preparation-calibration graph
reservations during runner recreation (`source-combined-all-gpu-q6-c0-event-reuse-r9`).
Its drivers remain clean and all six native contexts retire. The repair now
retains the original per-device calibration wrapper and its timing events
through the existing prepared source, without increasing opaque allowances.
Its 28-test focused owner run passes; the literal per-runner factory control
reproduces the reservation failure on CUDA and ROCm. Final model retirement
also closes those retained calibration handles before consuming the existing
device-retirement tickets. The literal old completion order fails the direct
pending-boundary regression on both backends. Normal Integration and Release
builds pass, followed by all 18 selected lifecycle/registration gates with clean
drivers and no skips in 65 s. Actual six-GPU C0 source admission, runner
recreation and final retirement then pass in 130 s
(`source-combined-all-gpu-q6-c0-calibration-reuse-r10`), with all six original
native generations retired, clean driver diagnostics and no skipped tests.
This does not expand the existing cold source-object bookkeeping contract; the
new independently allocated holder has an exact original-PMA CPU root.

The sustained six-GPU request exposed a separate migration-progress failure at
logical step 179. Bidirectional expert destination copies between CUDA 1 and
ROCm 2 stopped completing while generation initially continued. The existing
30-second transfer deadline fired; driver diagnostics stayed clean and the
process fully retired. Evidence is retained in
`/tmp/qfn-goal-continuation-r729/ordinary-combined-all-gpu-q6-c0-sustained-r2`.
The repeat preserves the underlying error and identifies a maintenance
service-telemetry snapshot waiting on the ordinary logical-state writer, which
was held for the whole request. The repair publishes each original hosted transaction through the existing
mailbox event and joins prior readers before the next writer; it changes neither
arithmetic nor device-state ownership. The actual 20-reset native fixture now
passes on CUDA/ROCm hosted and native modes. The identical corrected fixture
fails against the retained old library on all 20 CUDA hosted requests. An initial
fixture run omitted the `mtp` statistics domain and is not valid negative
evidence; the corrected control is `ordinary-writer-original-control-r2`.
The first sustained rerun completes both 384-token requests, but its first-request
movement assertion runs before cold transport calibration finishes. The test now
marks that request as intermediate and requires the full existing positive movement
evidence after the second request. Static zero-movement and all generation/graph/ngram
checks remain per request. Both focused evidence registrations pass; their replay
includes empty/proposal-only/stale-cell failures. The actual six-GPU rerun now passes
in 259 s (`ordinary-combined-all-gpu-q6-c0-cohort-r5`), completing both 384-token
requests and 46 dynamic migrations totaling 2,444,288,000 physical bytes. Drivers
stay clean and all six native contexts retire. No transfer timeout, movement policy
or correctness requirement is relaxed. This qualifies that ordinary MTP-off
functional slice; six-GPU source recreation and matched Release performance remain
separate pending gates. Two additional native hybrid checks pass: fixed-depth-1
exact serial/grouped MTP replay in 274 s
(`exact-combined-all-gpu-q6-c0-fixed1-r1`), including five completed expert
migrations, and ordinary generation with positive local/shared caches in 312 s
(`ordinary-combined-all-gpu-q6-equal-r1`), including both 384-token requests and
72 completed migrations totaling 3,708,313,600 physical bytes. Both retain clean
driver evidence, zero skips and all six final native-context retirements. These
are functional checks; their Integration wall times are not performance results.
The positive-cache six-GPU fixed1 companion also passes in 258 s
(`exact-combined-all-gpu-q6-equal-fixed1-r1`), with clean driver diagnostics,
no skipped tests and all six original native-context generations retired.
Both six-GPU cache variants now have ordinary and fixed1 functional evidence;
this does not substitute for the separate runner-recreation admission gate.

A fresh ROCm ConsumerLocal recreation run exposed a separate cold-preparation
fault after admitted MTP families were materialized: shared-expert launch setup
consumed an unpublished live-count pointer. Cold setup now initializes only
fixed-capacity metadata; captured execution still consumes the controller-owned
count. The focused CUDA/ROCm regression poisons that pointer before preparation
and then checks empty, partial, full and reset/repeated captured execution across
all supported formats. Both native backend gates pass (2.65 s CUDA, 3.43 s
ROCm); together with the three affected host gates, the selection passes 5/5
with clean driver diagnostics and no skips. The original real-model recreation
rerun now passes in 155 s with clean drivers, no skips and complete retirement
(`source-measured-rocm-tp2-consumer-off-fixed-r2`). The failed model run recorded
a driver workqueue warning and remains explicitly excluded.

The eight new six-GPU registrations preserve the original 2×CUDA continuation,
4×ROCm follower, DynamicOrdinal expert placement and R3 envelope. Shared ngram
owners are the original ROCm follower coordinates 0 and 2; only the two actual
CUDA PLE consumers receive local caches. C0/equal profiles each have Off/fixed1
source admission, with corresponding ordinary and native-exact execution.

Release binaries now build successfully for CUDA SM80/86/89/90 and HIP
gfx906/gfx1100. The existing paired benchmark driver can validate local setup
lineage and terminal windows and preserve unequal budgets through the exact
public configuration document; all 108 affected Python tests pass. It still
requires a genuine same-policy frozen-history numerical panel and original
final-PMA physical costs. The functional passes above do not substitute for
those prerequisites or establish a performance winner.

Six numerical cells are now registered through the original paired framework:
ROCm TP2 and CUDA TP2, each with shared-only, consumer-local and combined equal
profiles. They retain the original 25 histories and 178 rows (168 required,
10 diagnostic), references, arithmetic and limits. Actual public-parser and
source-binding checks pass. The ROCm TP2 combined-equal panel completed all 178
rows with unchanged limits, byte-exact reset, authenticated production execution,
clean drivers and complete native retirement. All four required groups pass:

| ROCm TP2 combined-equal group | Mean KL, nats | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|
| Short prefill, 16 histories | 0.029528 | 87.50% | 90.00% |
| Short decode, 16 histories × 6 offsets | 0.028153 | 95.83% | 91.67% |
| Held-out prefill, 8 histories | 0.019763 | 100.00% | 90.00% |
| Held-out decode, 8 histories × 6 offsets | 0.004590 | 100.00% | 92.92% |

Evidence: `/tmp/qfn-goal-continuation-r729/numerical-rocm-tp2-combined-r1/report.json`.
The additional QSA-boundary history remains diagnostic, as in the original panel;
its prefill/decode means are 0.0000805/0.0001762 nats. This is focused numerical
qualification of the selected topology's required groups, not full preflight or
image certification. The first CUDA panel collected all 178 rows and its raw
numerical groups pass, but its enclosing gate failed source freshness because
this project document was edited during collection. Preserve that failed report
and rerun against a fresh prerequisite receipt with the tree frozen; the raw
scores alone did not qualify that CUDA attempt. The subsequent frozen-tree
rerun now passes, as do fresh ROCm refined-combined, shared-only and measured
ConsumerLocal panels. Each completes all 178 rows, with unchanged required-group
limits, byte-exact reset, clean driver evidence and complete native retirement.
The three ROCm policies have identical required-group metrics to the table above.
The fresh CUDA combined results are:

| CUDA TP2 combined group | Mean KL, nats | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|
| Short prefill | 0.018523 | 93.75% | 91.25% |
| Short decode | 0.034786 | 93.75% | 90.83% |
| Held-out prefill | 0.020159 | 100.00% | 92.50% |
| Held-out decode | 0.006049 | 100.00% | 95.42% |

Evidence: `local-hot-current-accuracy-qualified-r786.json` under the root focused
result directory links all four fresh reports. Their four host prerequisite
checks are diagnostic subsets, not complete preflight or release certification.
The 128 MiB optional ceilings
are not a claim of equal complete physical footprints.

Actual final-PMA exports are now available for all three original ROCm policies.
Shared-only and combined source admission pass; ConsumerLocal reached the actual
exports before its recreation harness incorrectly required a remote participant
program for a purely local source. The harness now tests the original policy's
presence or absence of that program while retaining source, cache and PMA lifetime
checks. Eight focused regressions pass. Actual ConsumerLocal admission and cold
recreation now pass in 134 seconds with clean drivers, no skips and full native
retirement.
Two explicit-budget candidates adjust ConsumerLocal and combined ceilings using
those measured costs. They remain candidates until fresh admission measures the
actual residual and their original numerical panels pass. ConsumerLocal's fresh
final-PMA export measures a +5,248-byte residual against the shared-only control;
the complete non-cache owner base is unchanged. Combined's fresh run passes in
202 s with clean drivers and no skips, but measures +2,323,328 bytes against
shared-only. Its larger cache geometry costs more than the linear forecast.
A refined candidate selected through the canonical planner now passes actual
source admission and recreation in 201 s, with clean drivers and no skips
(`source-refined-rocm-tp2-combined-off-r1`). Its shared ceiling is 70,432,384 bytes
and each participant-local ceiling is 35,216,192 bytes. The complete admitted
plan totals 165,169,709,232 bytes: a 64-byte residual against the shared-only
control. The fresh refined numerical panel passes all required groups; matched
Release comparisons remain pending.
The original 128 MiB
functional profiles and their identities are unchanged. SSD-backed ordinary CUDA
TP2 passes in 104 seconds, and ROCm TP2 fixed1 native exact replay passes in 160
seconds, both with unequal C0/positive local capacities, clean drivers and no
skips. These runs qualify the backing path and request/MTP behavior; they do not
establish cold physical SSD latency because the OS page cache may already be warm.

The existing public benchmark campaign now accepts authenticated natural corpus
prompts and expected token IDs, preserving complete per-iteration terminal
counter records with their scopes and sequence windows. Fourteen trace/metadata
tests pass. Projection now rejects an unspecified seed instead of emitting an
unrunnable campaign: an explicit positive seed must also be supplied to the
public benchmark. The new regression fails against the original projection and
the complete fifteen-test metadata suite passes. Fresh varied four-history and
complete 25-history declarations use seed 17 for shared-only, measured
ConsumerLocal and the refined combined candidate
(`/tmp/qfn-matched-native-campaigns-r782`). Execution and Release timing
remain pending the actual admission and numerical gates; warm campaign results
will not be labeled cold-prefill or cold-SSD measurements.

Keep these columns separate as implementation lands. A policy/parser, source
admission, CPU replay, native component test, real-model run and Release benchmark
prove different things. None can be substituted for a later qualification.

## Configuration and admission

Add an optional `AdaptiveGPULocalHotCacheRequest local_hot_cache` to the existing
`AdaptiveVRAMRAMSSDRequest`. Its common row and physical-byte ceilings apply to
each actual consumer; exact typed consumer overrides may replace those ceilings.
It reuses `CpuImmutableRowCachePolicy`. Unknown, duplicate or foreign endpoints
reject before materialization. Only `DeclaredRankLocal` accepts this component:
absence means existing shared-only, and an engaged zero-capacity component means
the combined path with no optional local slots. Existing `ConsumerLocal` remains
a first-class comparison, without stacking a second local cache on it.

Use distinct `--ngram-local-cache-*` controls and lossless document/wire fields.
Existing source/pool and RAM cache controls retain their meaning. Resolve one
checked local geometry per actual consumer alongside the unchanged pool layout;
do not reinterpret resident ownership, a replica-group declaration, or expert
placement as permission to allocate local cache bytes.

`PhysicalMemoryAuthority` remains the sole admission/materialization ledger:

1. Freeze all existing higher-priority model, expert, KV and runtime obligations.
2. Include the full mandatory local transaction, pool and source floors. Price
   actual cache metadata, compact miss/scatter arrays and transport controls,
   including C0, with the existing exact geometry/root BOM machinery.
3. Admit optional local native slots up to their explicit ceilings, then shared
   GPU slots, then the existing optional RAM row cache. This order is authored,
   not inferred from observed free memory or a duplicated reserve calculation.
4. Materialize each root once through its existing PMA claim and allocator.
   A cache ceiling includes its payload and slot-dependent metadata; mandatory
   frames remain independently visible. Compare total physical cost in timing.

All ngram capacity remains lower priority than frozen model/expert/KV resources.
Changing cache contents does not release its admitted pool or reconstruct a
graph. A RAM-backed source does not acquire a redundant RAM row cache. Full
original source identity and storage generation bind local roots at setup;
request reset preserves valid immutable rows, while model/table replacement
requires a new binding. No payload checksum or hash is needed for that identity.

## Captured lookup transaction

Extend participant-local stages and existing prepared roots; model graph files
continue to declare the policy. The native local cache uses the existing
`AdaptiveImmutableRowCache` operations, radix directory, slot generations,
pin-before-reserve discipline, aged frequency and protected/probation streams.
Do not copy those algorithms into a second cache implementation.

1. **Probe locally.** Validate the original query/count through its existing PLE
   authority, deduplicate keys and pin usable local rows. Publish local readiness
   immediately for an all-hit frame. Record a compact miss roster and stable
   inverse/scatter map on device.
   Original sixteen-head token extent and arbitrary compact miss extent are
   distinct typed values; never round the latter up to a token or capacity.
2. **Issue only misses.** Address compact row keys to the shared authority through
   the existing counted NCCL/RCCL or cross-vendor TransferEngine edges. All-hit
   participants publish only necessary bounded progress metadata. Local output
   readiness must have no dependency on an owner result or acknowledgement;
   an empty request/receipt round trip is not a local-hit implementation.
3. **Coalesce shared work.** Form a device-owned stable union where consumers
   request the same source rows. Preserve per-consumer inverse maps. Reuse the
   existing pool coordinator and RAM/SSD source rather than starting an extra
   host service for each local cache. Native response delivery may repeat a row
   for its actual consumers; source lookup/fetch need not repeat it.
4. **Deliver native rows.** Return exactly each consumer's required original row
   bytes, authenticated by table/generation/interval, format, ordered roster and
   byte extent. One main-stream LocalRead stage acquires the frame's ready
   frontier, decodes pinned local slots or native responses with the existing
   format implementation, and writes all original output rows. There is one
   model-embedding producer, with no expanded hit staging or cross-stream
   partial tensor publication. Shared-only remains an unchanged control until
   a separately measured change is justified.
5. **Publish and retire.** Keep Reserved slots unreadable until their payload is
   present and validated; Ready publication follows the actual producer. Demand
   consumption applies the existing promotion policy once, and pins retire
   only after actual readers. Cancellation/error drains original
   accepted work through existing lifetime authorities. Immutable eviction has
   no writeback or coherence exchange.

Static capture retains admitted worst-case storage but communicates live extents
only. Device counts and cache decisions stay on device. There is no host count
readback, per-invocation allocation, host callback, eager path or recapture.
Exact streams/events and original captured transport ownership remain mandatory.
An all-hit consumer still participates in required protocol progress without
forcing the model consumer to wait for unrelated owner service. Prove this
distinction with a delayed owner, not only a zero-payload counter.

The finite service body starts at an explicit native parent dependency after
the current main Probe. Its completion joins the complete parent, not the probe
compilation unit: LocalRead may belong to a later unit. This avoids both a
previous-replay event satisfying readiness and a service/read dependency cycle.
Compiler partitioning, native parent composition and actual repeated capture
each need their own focused regression. No extra mutable readiness ledger or
independent service executable is introduced.

Request admission acquires the canonical controller's existing initialization or
reset event before publishing each participant's original source grant and local
reset. The same interval submits once; later entries acquire the existing frame
publication. Original whole-request terminal completion retires only that enqueue
duty, so a continuing response can open a new source interval in the same session.
Reset preserves hot rows and does not run per lookup. The generation-admission loop must submit all participants before
joining enqueue results: native peer setup can otherwise block the first sender
before its receiver exists, including a request whose prefill is skipped by a
full prefix hit. The once-request lifecycle has six focused cases passing 20
repeats. The actual Rank fan-out regression passes for one, two, four and eight
participants, including rejection and exception draining; the same tests fail
against the original sequential loop. Native request publication remains a
separate gate. Combined PP and homogeneous full-parent integration remain
unqualified until their own complete execution paths pass.

Prefill's original controller state is `Ok=0,error=0`, while generation admits
`Ok=1,error=0`. The local kernels now accept both valid forward states and reject
sticky errors or malformed flags; a focused shared-predicate regression and
native format replay cover this distinction. The host must not manufacture a
generation-admitted state merely to make a prefill cache operation succeed.

Ordinary decode, prefill and fixed/dynamic MTP use the same typed transaction.
Grouped verifier and scalar MTP-condition graph identities/count semantics stay
distinct. Cache frequency reflects consumed demand, not unconsumed speculative
read-ahead. Request reset cannot invalidate stable topology or rebind captured
roots. Dynamic expert movement may contend for bandwidth but cannot invalidate
immutable ngram rows or silently reclaim their PMA-owned physical pools.

## Read-ahead and bounded local retention

Current production implements early issue for the current invocation. Genuine
future tier prefetch remains an implementation gap: shared and RAM owners
currently credit reads as demand, and the complete current parent waits for
LocalRead before retirement. A future producer cannot wait for a later parent's
consumer. The [scoped future-prefetch design](QFN_LOCAL_HOT_READAHEAD_DESIGN_20261106.md)
records the required lookup-only operation, actual demand-ID authentication and
producer/consumer lifetime split. The contract below is the target state.
New focused host regressions prove mixed speculative discard leaves hot bytes
and policy unchanged, late adoption earns credit only after actual read, and a
finite consumed scan preserves a warmed row, for both key widths. They do not
claim a future native fetch, GPU event ordering or a performance gain.

Read-ahead and caching share the same lookup identity and native row lifecycle.
Distinguish early issue of already consumed/current demand from speculative
future work with a typed access intent. A later demand for an outstanding exact
table/generation/row request adopts that result once; it must not issue a second
owner/source fetch or credit the same access twice. Probe Ready local rows before
transport and share exact in-flight requests where their lifetime permits it.

Use existing bounded native response/request storage for speculative staging.
An unconsumed prefetched row does not earn demand-frequency/protection credit,
and prefetch does not reserve the whole local cache or evict protected demand
rows. Consumption applies the existing cache policy; a response can supply its
consumer without acquiring an optional cache slot. Prefill scan length therefore
does not dictate local-cache occupancy or retained pin count. A future read may
be omitted when the admitted lookahead/staging window is full; required demand
still has its first-class counted transaction. No background unbounded queue or
second RAM/SSD provider is introduced. Cold prefill favors compact batching and
useful chunk overlap; decode favors low fixed overhead and the local-hit path.
The same fixed lookahead depth is not presumed economical for both phases.

Known prompt chunks and already published MTP candidates are valid inputs;
unknown future sampled tokens are not. The existing producer frontier determines
how early work can start. Rejecting an MTP proposal or cancelling/resetting a
request retires speculative staging through the original transaction, without
invalidating immutable hot rows or leaving reserved slots pinned. Short decode
and all-hit workloads must not pay an owner round trip merely to run a prefetch
controller. The capture-safe local-ready frontier belongs to the actual cache
frame; independent auxiliary progress joins at required retirement, not before
the model's local-hit consumer.

Measure read-ahead disabled and enabled at the same total admitted bytes and
cache partition. Include local-hot decode, shared-hit/RAM/SSD misses, cold long
prefill, skewed consumers, rejected MTP candidates and sudden hot-set changes.
Report useful/wasted native fetches, duplicate suppression, consumer wait saved,
time-to-use, staging/pin high-water marks, protected-row survival and full-model
throughput. Earlier submission alone is not a performance win. Keep any adaptive
lookahead decisions in the existing device controller and test recorded inputs;
do not add a host policy that reads live counts or cache contents.

Completion requires effective fetch/compute overlap across the six hardware rows
below, not merely native component passes or a higher future-ready counter.
The finite producer parent joins its future body before completing, so
`future_on_time_native_rows` means ready at adoption even if that join stalled.
Use the existing native graph/stage timing machinery to measure the independent
main-tail and future-body completion intervals on the same device, plus the
complete parent critical path. Report exposed fetch delay separately from
useful native bytes. Never subtract timestamps from different GPU clocks or
label host enqueue duration as device work. Keep instrumented attribution
separate from interleaved, unprofiled Release comparisons. A benefit must survive
the complete request timing, including required control traffic and retirement.

## Functional qualification

Extend existing cache/provider/pooled graph fixtures and the canonical typed QFN
matrix; do not create a second topology list or a synthetic serving path.
Register focused production regressions explicitly in `ProductionTestPreflight`.
During this iteration run those focused registrations, not the full preflight.

| Level | Required cases and evidence |
|---|---|
| Fast device-free | Real cache operations and root geometry: C0/positive/unequal capacities; all-hit/all-miss/mixed; duplicate/disjoint/overlapping consumer rosters; missing/foreign identities; exact fit and one-byte-short admission; every supported native row format and 32/64-bit keys. |
| Lifecycle replay | Pin/eviction interleavings, scan pollution, hot-set changes, stale/reserved slots, partial and empty frames, large-to-small replay, reset, table replacement, error/cancel and repeated execution. Test source, shared and local generations separately. |
| Native CUDA/ROCm | Actual captured producer/consumer kernels and transports, poisoned inactive tails, output byte agreement, exact per-edge request/native-result/metadata bytes, repeated counts including zero. Hold owner service and prove all-hit output is independently ready. Compile shipped CUDA SM80/86/89/90 and HIP gfx906/gfx1100 with the normal spill guard. |
| Placement | Original RAM and bounded mmap SSD sources; RAM cache on/off; shared owners colocated with continuation, follower expert, or both roles. Empty owners, non-owner coordinator, reader-only participants and unequal local caches remain valid. |
| Real model | Project the existing six hardware rows: MI50×1/2/4, RTX3090×1/2, and RTX3090×2 continuation plus MI50×4 expert followers. Use original GGUF, admitted source and immutable table lifetime. Run actual ordinary generation and Off/fixed1/fixed3/Dynamic MTP witnesses with repeated requests/reset/prefix restoration. |
| Accuracy and experts | Existing frozen-history numerical panels report prefill/decode mean KL and top-1/top-5; diagnose drift with canonical layer/stage PyTorch CSVs. Preserve serial/grouped byte equivalence. Dynamic balancing must demonstrate actual published expert migration and live payload while ngram caches remain correct. |

Use cheap metadata/ops coverage for the full policy product, with complementary
native/model projections for real transports, lifetimes and arithmetic. Each
advertised topology/mode needs its actual execution witness; unavailable or
unfinished cells remain unqualified, not successful skips. CPU-backed sources
are required for all GPU configurations; existing CPU lookup/cache tests remain
the source-format and host-policy controls.

## Matched performance qualification

Compare **ConsumerLocal**, **shared-only**, and **combined** at matched total
physical VRAM cache budgets, counting local plus shared metadata and payload.
Keep the RAM budget, original source, expert residency, arithmetic, MTP intent,
token IDs and context fixed. Record admitted bytes per participant and tier;
nominal row count alone does not establish a fair comparison. Use several
explicit local/shared splits rather than spending unlimited capacity on replicas.

Extend the existing lookup measurement and canonical Release benchmark drivers.
Record realistic QFN token/ngram traces from cold prefill, sustained decode,
repeated coding requests, partial-prefix reuse and changed conversations. Retain
original request/phase boundaries and speculative/consumed status. Include a
held-out trace, sequential scan pollution and working sets above each tier's
capacity. Synthetic IDs remain adversarial controls, not the sole economy proof.

For every policy report:

- End-to-end prefill and committed decode tokens/s with MTP work/acceptance, plus
  complete lookup consumer-stall latency p50/p95/p99 and separate submit/event
  scopes. Timing runs are unprofiled and interleaved with matched controls.
- Local hits, shared hits, RAM service and SSD service, by original demand and
  unique native rows. Specify denominators; a window mapping hit is not an SSD
  cache hit. Preserve per-participant and total views.
- Actual request/native-payload/control bytes per edge, dedup savings, necessary
  empty progress metadata and useful/wasted read-ahead. Do not infer wire bytes
  from admitted capacity or count all replicas as unique source rows.
- Promotions, evictions, protected/probation retention, pin occupancy, turnover
  per useful row, resource peaks and bounded steady-state allocations. OS page
  cache residency/fault/read-byte evidence remains separate from explicit RAM.

Collect cold-source evidence using only fixture-owned files and the existing
page-cache observation procedure; never clear unrelated host cache. Include warm
repeated and warm changed-hot-set trajectories. Profile new kernels with the
backend skill's isolated shape/ISA procedure, documenting registers, spills,
occupancy and bandwidth separately from canonical timing. Profile transport
waits to explain gains or regressions; no direct P2P is assumed on this machine.
NCCL/RCCL may use P2P when available without a new application-level transport.

Promotion is per proven topology/workload policy. Correctness and economical
native traffic are prerequisites; an isolated hit-rate increase does not justify
a default if complete throughput or tail latency regresses. Preserve explicit
capacity-oriented modes even where another mode is faster. Dashboard rows link
the exact source/settings/evidence and keep unqualified columns visible.

## Delivery sequence

1. Add typed policy/geometry/PMA, production local cache operations and compact
   request/response contracts in parallel, with focused host regressions.
2. Join their native roots, participant-local stages and original model binding;
   prove coherent builds and actual captured all-hit/miss/mixed transactions.
3. Qualify the real model on representative CUDA, ROCm and hybrid slices with
   ordinary/MTP and dynamic experts, then cover the remaining canonical rows.
4. Run matched Release trace and model benchmarks. Tune measured bottlenecks,
   preserve numerical gates and promote only the combinations with evidence.

Checkpoint meaningful green slices. Implementation progress never changes a
pending functional or performance column to green automatically.
