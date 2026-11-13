# QFN eight-MI50 inference after the passing checkpoint

## Scope and qualification boundary

Checkpoint `d1fd0a077` is published on `feat/qfn-ngram-continuation`. Its
completed CTest lanes contain 5,256 passing selected Unit/preflight cases,
zero skips, and 39 hardware-ineligible cases explicitly unqualified. The
interrupted continuous driver observer prevented a complete whole-run receipt.
The user accepted checkpointing that passing slice and canceled its redundant
repeat. Neither that decision nor the focused work below creates an image,
release, or complete preflight certificate.

The next workload remains the original observed QFN
`PP(EP(4×MI50), EP(4×MI50))`: discovery MPI rank 1 owns both stages on NUMA 1;
stage 0 uses ROCm 7/6/8/9 and layers 0–23, stage 1 uses ROCm 1/3/4/2 and
layers 24–47. Ngrams retain disjoint fixed-head VRAM ownership across all
eight GPUs. The original nine prompt tokens, six teacher-forced observations,
GGUF weights, reference snapshots, precision policy, and context are preserved.
MTP is off; prefix caching remains enabled by the frozen configuration.

## Defects isolated by the real-model diagnostic

1. **Snapshot admission charged both PP stages for all layers.** The aggregate
   diagnostic payload had lost layer ownership. Preserve each main-layer
   contribution and project it through the planner's admitted layer interval;
   retain global payloads. The materializer was already participant-local.
   No capacity, bucket, model, or snapshot obligation is reduced. Planner BOM
   regressions cover uneven intervals, whole-model sizing, malformed ownership,
   and CUDA/ROCm resource descriptions. The next native attempt passed admission.

2. **A storage-only pipeline participant never materialized its ngram range.**
   The factory wrapper filtered participants by the PLE layer before reaching
   the existing fixed-head storage-owner implementation. A shared typed role
   selector now preserves those owners and uses their original loader,
   PhysicalMemoryAuthority, source, staging, and publication path. The focused
   factory test exercises both PLE-stage placements and the original eight-GPU
   order. The next native attempt reached model forward execution.

3. **Repeated retirement used borrowed stages after their destruction.**
   Rank retirement correctly retired the pipeline once, but the pipeline's
   later destructor revisited its borrowed device-stage pointers. Successful
   retirement now revokes those borrows through a typed lifecycle. Strict
   device-generation epoch checks remain unchanged. The production-body
   replay covers 48 cold/submitted/maintenance combinations; the subsequent
   failing native diagnostic exited normally instead of aborting in teardown.

4. **Parity required an intentionally omitted short-context query projection.**
   QSA's proven-no-ranking prefill program omits raw/prepared query producers.
   Decode already had structural omission evidence; prefill did not. A fresh
   successful exact-retained request observation now records its original
   logical range, actual physical width, phase, selected program, and admitted
   participants. The C++ and Python consumers independently verify the empty
   ranking interval and every participant's counter delta. Missing work gains
   no numerical samples or equality verdict. Published prefill values keep
   their existing numerical comparison contract. Stale, padded, partial,
   foreign-device, and wrong-phase witnesses fail. Actual C++ CSV output is
   round-tripped through the Python consumer for both phases.

5. **Supporting state publication was mistaken for generation-loop execution.**
   Zero-output model forwards share event/publication infrastructure with
   generation. The diagnostic interpreter now distinguishes those support
   records and exact ForwardOnly admissions from actual generation requests.
   Every original admitted device still needs complete capture and an
   authenticated model-forward terminal observation. Setup capture alone is
   insufficient; forward-only evidence never certifies full generation.
   Ordinary, speculative, malformed, and unknown generation records retain
   their existing fail-closed requirements. Replay tests also reject missing
   completions, foreign devices, invalid counter extents, and overflow.

6. **Nested PP diagnostic participant discovery expected only its leader.**
   The first repaired native retry (`q53`) still failed the strict omission
   proof. The top-level TP list is intentionally empty for local PP with
   per-stage TP; the diagnostic interpreted that as one device. Participant
   projection must instead retain the original prepared admission stages and
   every stage's full member list. The focused nested-PP regression preserves
   the original eight-device order and rejects duplicate, foreign-rank,
   foreign-host, and conflicting memberships. Missing-query failures also
   retain their actual counter observations for diagnosis. No participants
   are inferred from the observed counters and no padded geometry is accepted.

7. **Composed decode bypassed the direct-forward query observation.** A source
   audit before the next costly native retry found that successful pipeline
   ForwardOnly transactions never reached the direct DGO observation emitter.
   Both paths now use one typed production recorder; the composed path records
   each original child only after every terminal and output publication succeeds.
   Completion retains its original executed family through admission retirement,
   so clearing active request state cannot select a different family's signature
   for cleanup or observation. The production recorder replay covers six rows,
   participant counts through eleven, reset, partial completion, stale counters,
   and role/program/width mismatches. It is a metadata observation, not another
   device-state readback or a change to query computation.

8. **The fixed pipeline hid its final lexical gather from snapshots.** Native
   `q59` proved the original nine physical rows and all eight query-program
   witnesses, then correctly rejected the missing `layer1_PLE_GATHER` output.
   The terminal decoder already produced the full FP32 embedding; its stage
   returned an empty dump description alongside the intermediate protocol nodes.
   Only that terminal stage now exposes the original tensor and exact physical
   subview as `gathered_rows`, using the existing semantic mapping and ordered
   snapshot authority. Intermediate packets remain private. A device-free
   regression exercises the actual production descriptor and snapshot extractor,
   offset/extent validation, repeated/reset geometries, and all three PLE keys.
   The diagnostic now reports every missing required key in one phase together.

9. **Completed composed Decode did not publish its captured snapshots.** The
   `q62` prefill gate passed, and its Decode parent completed, but the composed
   path never invoked the ordinary executor snapshot-publication callbacks.
   `ForwardExecutionEngine` now authenticates the exact cached child/capture,
   producer stream and snapshot-configuration epoch, then publishes that child's
   already-written manifest through the existing acquisition path. It records
   the original last-executed source only after success. The pipeline uses its
   retained completed family for every participant; no graph is re-executed or
   recaptured. An actual-engine mock replay exercises repeated geometries and
   rejects stale identities, epochs, default streams and reset-invalidated views.

10. **Failed numerics suppressed the authenticated owner configuration.** The
    original join previously conflated work success and communicator-borrower
    retirement. They are now separate typed outcomes in the same compact
    terminal exchange. Only the authenticated original active owner may export
    after every borrower retires; numerical failures remain failed in receipts.
    This preserves useful diagnostic configuration without qualifying failed work
    or inventing an execution/discovery-rank mapping.

11. **Snapshot publication consulted a withdrawn child stream.** A composed
    parent supplies its exact producer stream, but the executor queried the
    child's unavailable standalone stream before consulting that override.
    The explicit producer now owns publication directly; missing, null and
    default stream validation remains strict. Actual-executor replay covers
    six observations, exact events and no additional input transfers. `q68`
    passed all four focused executor/engine aliases. The next native run
    published all six Decode steps and all eight canonical CSV artifacts.

12. **A one-row prefill tail left replicated GDN state one token behind.**
    `q69` first diverged at layer 0 short convolution, with relative L2
    0.134/0.067/0.042 on the first three Decode rows, then approximately
    0.001. The original prompt uses exact retained chunks of 8+1 rows.
    Its final chunk is explicitly Prefill and updates TP-local convolution
    and recurrence banks, but a `rows > 1` graph guard omitted publication
    into the full banks used by replicated Decode. Prefix serialization
    faithfully preserved both banks, including the stale full state.
    Graph and workspace admission now share a typed producer-layout rule:
    every positive linked-local producer publishes; full replicas do not
    perform a redundant gather. Regressions distinguish Prefill 8/1/9 from
    Decode 1/2 and replay retained native handoffs with dirty full banks.
    The graph/accounting/telemetry selection passed all seven aliases in
    `q71`; native CUDA and ROCm retained handoff replays both passed in `q74`,
    with zero driver findings. `q75` confirmed the fix: all six layer-0
    short-convolution relative errors are 0.00077–0.00110, and recurrence
    errors 0.00053–0.00130. The old three-token history-error signature is gone.

13. **Outer PP position telemetry reported its scheduler cursor.** The
    completed child Decode observations advanced from 10 through 15, while
    the outer probe retained 9. The projection now uses the original terminal
    stage's already-acquired scalar/vector state, preserving independent
    positions of other stages and leaving the host scheduler unchanged.
    Focused replay covers reset, restore, repeated/partial work and a missing
    terminal observation; its Unit and explicit preflight aliases passed in
    `q71`; `q75` has no remaining logical-position failure. This observation
    fix changes no model arithmetic.

14. **The source-admission assertion conflated membership and wire order.**
    Canonical ngram layout geometry sorts allocator identities; the fixed
    pipeline intentionally retains the original PLE-stage participant order.
    Requiring the two orders to match rejected valid unsorted original domains.
    The test now proves unique exact membership independently of exact
    original stage/wire order. Per-owner ranges, participant indices, UUIDs,
    budgets and communication obligations remain unchanged. A focused
    device-free regression exercises both backend descriptions, both PLE-stage
    placements, and missing/duplicate/foreign/reordered negative cases.

15. **The HTTP/canary observer rejected the effective default prefill cap.**
    The unpaired full-corpus collector supplies the normal omitted cap, which
    the canonical configuration resolves to M512. The observer compared the
    optional authored value directly with 512 and rejected that valid intent
    before admission. It now consumes the existing canonical normalization;
    other caps still fail, and the independently authored ExpertOverlay segment
    geometry is preserved. The new default-cap regression runs through the
    existing explicit preflight contract. Build `q88` and all five focused
    checks in `q89` pass. This repair is published as `9b970ad61`.

16. **The public local-PP frontend initialized a test-only global router.**
    The public OrchestrationRunner called `GlobalBackendRouter::initForTests()`.
    That helper fabricates a no-MPI rank-zero inventory from every visible
    device. The `q90` observation consequently initialized unrelated CUDA and
    ROCm groups before the actual selected eight-MI50 pipeline. A typed PP
    admission now authenticates the original active communicator, immutable
    inventory, ordered physical participants, and actual graph execution plan.
    Captured pipeline edges keep their own transport authority and create no
    additional router communicator. Generic boundaries admit only their actual
    directed initial copy endpoints; destination TP publication remains separate.
    Admission and runtime share allocation-free endpoint selection, preserving
    source-TP priority. Device-free tests exercise original rank projection,
    spare and foreign GPUs, unequal/mixed-vendor boundaries, snapshot identity,
    and forbidden lazy group creation. A source-policy negative control restores
    the original production test-helper call and proves its rejection.

17. **Driver diagnostics overlooked informational AMD/KFD wait-stack frames.**
    The old severity/word filter accepted `q90` despite kernel frames showing
    GPUVM mapping and clock-counter waits. Exact symbol/module/frame matching
    now catches those records without flagging ordinary startup prose or
    unrelated waits. All 21 direct Python tests pass, including original-code
    failure controls and supplied stale-summary rejection. Reanalysis finds
    20 findings in the preserved `q90` journal; its original clean summary is
    not accepted. No original evidence was overwritten. The existing explicit
    HTTP driver-diagnostics preflight entry covers this classifier.

## Public-frontend startup observation

Build `q94` passes. After correcting the new fixture's original one-host
rank metadata, `q95` passes all three repaired router/source aliases; the four
unchanged RankOrchestrator, source-negative-control and driver aliases passed
in `q93`. These are focused checks, not a full preflight rerun.

The unpaired `q96` collector uses the unchanged 16-history corpus, original MPI
rankfile, SingleQ8 arithmetic and selected eight-MI50 layout. Its observer
contract intentionally has MTP and prefix caching off, unlike the prefix-enabled
short detailed fixture above. It remains explicitly **UNCALIBRATED**, unscored
and ineligible for certification. No failed detailed receipt was rewritten.

The unrelated CUDA communicator is absent after the frontend repair. RCCL setup
is nevertheless very slow: loading at 03:54:06 UTC, the first intended group
7/6/8/9 finishes setup at 04:00:41, and group 1/3/4/2 at 04:04:12. Read-only host
GDB observations confirm the first original group contains exactly four devices
7/6/8/9 and show HSA GPUVM mapping and RCCL buffer/stream preparation waits.
Those attachments perturb the observation; it is not an uninstrumented timing
or stability result. The process subsequently materializes approximately
19.4–19.7 GB on each selected MI50, so it was progressing rather than remaining
permanently stuck at the first mapping call.

The canonical 15-minute cell budget expires before logits are emitted; the
wrapper retires in 908.4 seconds with zero completed cells. Every native process
retires, VRAM returns to baseline, and the complete `q96` driver window contains
no findings. Sampled active-rank peak RSS is about 20.5 GiB; this is startup
resource evidence, not a leak qualification. The immediate blocker is startup
latency, not an accepted corpus or performance result. No timeout was enlarged.
Evidence is in `/tmp/qfn-postcheckpoint-unscored-single-q8-corpus-q96/` and
`/tmp/qfn-postcheckpoint-focused-launch/q96-progress.jsonl`.

The fresh `q97` original short-history control completes native execution in
101.1 seconds on the same boot, with a clean complete driver window and normal
retirement. Its prefill and all six decode CSV metrics repeat `q78` exactly;
the unchanged decode-quality failure remains visible. The first intended RCCL
group returns about 2.8 seconds after loading the library, and the second route
is published 15.3 seconds later, including intervening setup work. This shows
the failure is intermittent; it does not rule out a kernel scheduling defect.

Source review corrects the earlier startup hypothesis: both the detailed parity
fixture and the public canary use the same OrchestrationRunner factory and
native-worker preparation order. The public configuration also explicitly
requests 28 threads, applied before RCCL startup; the launcher's initial
56-thread environment does not establish inference oversubscription. HWLOC
module policy differs (`-gl,-opencl` in the short control, omitted in `q96`),
as does optional CUDA discovery. These are experiment candidates, not proven
causes. A proposed standalone clique constructor was not applied because it
would introduce a second admission route. Continue through the existing public
observer, preserving its configuration and native retirement authority.

After the user reboot, `q99` reaches M512 capture but times out at
`layer24_gdn_wo_allreduce` with one of four participants present. Other threads
wait inside KFD mapping. `q100` and `q101` then isolate the same wait during
initial communicator construction, before capture. The blocked allocation is
the original RCCL uncached P2P buffer, not a newly created late communicator.

The decisive `q101` observation finds a pending AMD TLB-flush software fence
whose dependency is already signaled. Its CPU work remains queued while both
workers in the CPU28 `events` pool are idle; one is runnable but receives no
CPU time for minutes. Four normal-priority native initialization threads spin
on that CPU inside `ncclProxyCallBlocking`. All ten GPUs' sampled ring fences
are retired. A brief diagnostic host attachment releases progress. These
observations strongly implicate host scheduler starvation; they do not establish
a GPU arithmetic fault or justify changing peer visibility, capture, or timeouts.

The running patched kernel contains a matching upstream CFS reweight defect.
An isolated repair candidate preserves its existing IOMMU/DRM fixes and keeps
the installed kernel as rollback. No host installation or reboot has occurred.
See [the scheduler investigation](QFN_HOST_CFS_TLB_STARVATION_20261112.md) for
source/boot identity, upstream repair dependencies, passive evidence, and the
required post-repair native validation. The diagnostic runs remain unscored;
the original detailed Decode quality gate remained red at that point, and
commits were held pending the completed corpus below.

## Current startup and observation result

The [broader host-affinity audit](QFN_HOST_THREAD_AFFINITY_AUDIT_20261112.md)
repairs master-mask leakage in MPI/OpenMP bootstrap and independent control
workers inheriting one OpenMP place. Its completed combined Integration
selection passes 47/47 with zero skips, including the loading, mmap reclaim
and NUMA regressions. Both disk-worker lifetime aliases additionally each
pass 20 consecutive invocations.

The unchanged 16-history public observer, `q117`, completed in **82.621 seconds**
on the current ABI1000 kernel without debugger interruption or a stall. Actual
control-thread masks retain the original rank CPU partition, while existing
OpenMP workers retain individual placement. All 112 observations completed,
reset bytes matched, the driver window was clean, all native owners retired and
the eight selected GPUs returned to their exact baseline VRAM usage. Frozen
source/build and retirement proof live under
`/tmp/qfn-postcheckpoint-focused-launch/q117-prep/`.

This is **not an accuracy pass**. All six Decode rows in every history, including
the reset history, are byte-identical to that history's Prefill row. The public
observation path is exposing stale logits; its Decode metrics cannot measure
quantization accuracy. The separately captured detailed snapshots and their
mean KL 0.1982729 remain unchanged evidence. Repair the original pipeline result
publication, add a focused freshness regression and rerun the unchanged corpus
before interpreting its Decode quality. Its report remains `UNCALIBRATED`, with
`passed: null`; no tolerance or policy was changed.

The separately built ABI1001 scheduler candidate remains uninstalled. Twelve
paired KVM trials reproduced starvation on neither image, so that experiment
is inconclusive and does not establish the upstream defect's causality here.

### Fresh publication and completed corpus

The composed parent had already published each terminal child's fresh logits
on its exact stream. `RankOrchestrator::retirePipelineForwardOnlyDomain` then
advanced metadata without updating the terminal domain's retained host result.
The outer pipeline copied that old Prefill row. Completion now invalidates the
old host result and, only when host observation is enabled, uses the original
child-consuming gather before publishing rank metadata. Device-only and hidden
domains perform no host gather. Failed publication leaves no valid host row and
does not advance rank metadata. No model row is replayed and no arithmetic changes.

The existing LogitsGatherer owner exercises this actual method with CUDA/ROCm
metadata and participant widths 1/2/3/8, exact consuming streams, repeated
rows/reset, disabled observation, non-head stages and failed copies. All three
new lifecycle cases pass; all fail against the saved original core. The `q124`
selection passes 8/8, including the full LogitsGatherer Unit owner and six
strengthened loader aliases. The first fixture attempt (`q122`) incorrectly
read child-owned position to check rank bookkeeping; its assertion was repaired
without changing production position ownership. The contaminated negative
control remains retained separately from the valid `r2` proof.

`q125` repeats the unchanged original model/topology/precision corpus and
completes in **83.44 seconds** including setup, observation, driver checks and
scoring. Each history now has seven distinct full-vocabulary rows. All 16
histories/112 observations complete, reset bytes match, the driver window is
clean, all native owners retire and all eight GPUs return to baseline VRAM.

| Phase | Rows | Mean KL | Top-1 agreement | Mean top-5 overlap |
|---|---:|---:|---:|---:|
| Prefill | 16 | 0.0260335 | 87.5% | 92.5% |
| Decode | 96 | 0.0225803 | 96.875% | 91.875% |

The unchanged original pilot's short-context Prefill/Decode limits and all six
Decode-offset limits pass in a separate comparison. Five Decode rows exceed
0.1, below the existing 12.5% tail allowance; maximum KL is 0.525417, below its
1.0 bound. Three are the original nine-token history's steps 3/4/6, matching the
earlier detailed CSV values to rounding precision. Their same-input diagnosis
supports intended Q8 error amplified by closely ranked expert selection. The
other two are Unicode/accent step 1 (0.224736) and prose/cause step 5 (0.133051);
those have not been individually isolated. Four of the five preserve top-1.
Unicode/accent Prefill is the single Prefill row above 0.1 (0.246579).

The user accepted these quantisation-sensitive spikes and authorized this
checkpoint; they do not block performance work. No arithmetic policy or limit
was changed. The canonical unscored report remains **UNCALIBRATED / passed:null**:
the separate passing numerical comparison does not manufacture an approved
baseline, calibration, full preflight receipt or release certificate.

Peak observed native RSS was 21.99 GB on the active rank and 0.787 GB on the
other rank, measured separately. GPU peaks were 21.31–21.70 GB per participant;
native swap was zero. All histories and original native resources retired.
These finite diagnostic lifetimes do not certify long-session leak behavior.
The result and unchanged-input/physical-identity/freshness/retirement receipts
are under `/tmp/qfn-postcheckpoint-unscored-single-q8-corpus-q125/` and
`/tmp/qfn-postcheckpoint-focused-launch/q125-prep/`. Release throughput and
stage timings are next; diagnostic wall intervals include full-vocabulary D2H
and must not be presented as production performance.

## Focused validation and next measurement

The `q69` real-model retry retained the green prefill final-logit KL
**0.012907**, top-1 agreement **1.0**, and top-5 agreement **1.0** on the
 original frozen prompt. All six Decode observations are now published:
KL **0.811063, 0.329030, 0.890677, 0.713227, 0.311029, 3.1799**;
top-1 **4/6**, mean top-5 **0.5667**. This is an actual red numerical result,
not missing evidence. The state-handoff fix above must resolve it without
changing the reference, precision or acceptance gates. All eight original
participant identities and completed terminal joins remain authenticated.

After the handoff and position fixes, `q75` Decode KL is **0.0580601,
0.0431418, 0.203313, 0.525417, 0.0538835, 0.305822** (mean **0.1982729**),
top-1 **5/6**, mean top-5 **0.7333**. Prefill is unchanged. All six layer-0
GDN convolution/recurrence comparisons now agree near 0.1% relative L2.
The next large discontinuity occurs when the selected expert set changes:
first affected layers **17, 31, 10, 12, 2, 2** respectively. Small observed
router errors exceed the independent reference's narrow tenth/eleventh-expert
probability margins. No selector defect or same-input arithmetic equivalence
is claimed from that observation. The original SingleQ8 GDN output projection
is the earliest consistent amplification and needs a same-input comparison
before assigning the remaining drift to intended quantization. The existing
parity-owned bounded raw snapshot exporter supplies those operands; the older
generic SnapshotConfig dump flags have no live call sites in this tree.

`q78` removed the source-admission assertion failures and repeated all six
`q75` KL measurements unchanged, with a clean driver window and retired native
owners. The existing bounded exporter retained 2.50 MiB of operands. Offline
same-input calculations read only the original layer-0 `ssm_out` Q8_0 tensor
and five original router tensors; no weights were requantized or payloads
hashed. Native GDN projection versus the independently reconstructed SingleQ8
operand agrees within **9.42e-8** relative L2 on all six Decode observations;
the intended Q8 activation effect itself is **0.00833–0.01053** versus the
same original FP32 input. The nine Prefill rows show the same behavior.

At all six first expert-set changes, independent FP16-operand/FP32-accumulator
router calculations agree with native probabilities within **1.34e-7** relative
L2 and reproduce the selected expert sets. Normalized selected weights agree
within **1.49e-8** maximum absolute error. Original-reference-input FP32
calculations also reconstruct the reference. Using FP32 operands on the
already-perturbed native input retains five of the six changed sets: promoting
the router alone would not remove the main upstream quantization effect.
These numerical comparisons establish the cause at the checked sites; they
do not waive the failed final-logit gate or certify every kernel/native byte.
Reports and offline scripts are under `/tmp/qfn-q75-same-input-diagnosis/`.

The next controlled option is the already-supported TwoTermQ8 GDN output
operand. It must have its own declared arithmetic binding and frozen replay
descriptor, with the original source, histories, references and thresholds
unchanged. Runtime environment changes alone cannot silently alter the
SingleQ8 case's authenticated policy. Production defaults remain unchanged.

That controlled candidate was prepared under
`/tmp/qfn-postcheckpoint-focused-panel-prep/gdn-output-two-term/`.
Only its declared GDN-output selector changed; all document-identity updates
followed from that change. The canonical freezer accepted it, the fixture-only
build `q82` passed, and all six binding/history checks passed in `q83`.
`q84` then stopped before model forwards with the explicit **unqualified TP
GDN-output source-interval** diagnostic. It has no numerical or performance
result; native teardown and its driver window completed cleanly.

The existing operator supports local TwoTerm arithmetic, but GDN prefill packs
potentially noncontiguous value-head column spans. Its current source contract
requires a complete dense interval. Extending support must authenticate that
actual packed layout, retain the original native codebook and existing rank
reduction, and qualify same-topology serial/grouped replay. Removing the guard
is insufficient. There is also no existing phase-aware operand policy that
can honestly select SingleQ8 Prefill and TwoTerm Decode under this scalar
setting. This extension remains separate from the verified production fixes.
The original SingleQ8 binding was restored in `q85`/`q86`. Next characterize the complete
frozen corpus through an explicitly unscored diagnostic route, retaining every
short-history numerical failure and all original acceptance limits.

The original seven admission/factory/retirement registrations passed. The
extended selection has 17 registrations: 13 passed in `q50`; two new fixture
mistakes caused the other four aliases to fail. Correcting canonical device
spellings and separating numerical verdicts from invalid provenance made all
four pass in `q52`. The existing 146-test campaign-driver suite and native
collector suite passed. Nested participant regressions passed in `q55`, seven
composed-observation/retirement/collector checks in `q58`, five PLE/snapshot
checks in `q61`, eight composed-snapshot/export checks in `q65`, and four
explicit-stream checks in `q68`. These are focused results, not a full rerun.

The GDN/PP candidate built in `q70` (350.2 s), followed by seven passing
focused aliases in `q71` (45.1 s). `q72` passed ROCm but exposed an error in
the revised CUDA regression's selection predicate: CUDA publishes its original
SM major/minor and leaves the ROCm-style `native_isa` string empty. The fixture
now checks the existing SM observation for its selected endpoints; no device
alias, ownership override or extra probe was added. Its single-owner rebuild
`q73` took 29.1 s. Both native aliases passed in `q74` (11.5 s CTest), with a
complete clean driver window. The original model retry is `q75`, in
`numerical-run-9`.

Local evidence is under `/tmp/qfn-full-preflight-after-cpu-next/` and
`/tmp/qfn-postcheckpoint-focused-launch/`: builds `q48`/`q51`, focused JUnit
files `q50-focused-evidence-regressions.xml` and
`q52-focused-evidence-regressions.xml`, and the original failing exact-cell
artifacts `numerical-run-1` through `numerical-run-8`. The `q69` driver window
completed with zero findings; native processes retired, all nine MI50s returned
to approximately 10.4 MiB driver usage, and the 114 GB run-scoped source staging
was removed. Peak sampled active-rank RSS was approximately 49.8 GiB with full
diagnostic snapshots; this is not a long-session leak qualification.

The functional repairs have focused model-free and native coverage; the model
now supplies complete snapshots and fresh public logits. The completed unchanged
16-history observation passes its existing numerical limits in the separate
pilot comparison. Accepted isolated quantisation spikes remain visible in the
raw detailed evidence. The canonical scored panel still requires its original
successful binding; unscored collection retains its non-certifying status.
The first Release throughput observation is complete below. Stage profiling is
next; no image qualification follows from these focused observations.

## Release accuracy and first throughput baseline

The runtime repairs are checkpointed as `60d9a2f7d` on
`feat/qfn-ngram-continuation`. The subsequent selective Release configure found
two test-registration errors: linker options were applied to omitted mmap-reclaim
and loader-affinity fixtures. Existing `if(TARGET ...)` guards fix both without
changing runtime code or complete Integration registration. Three focused
target-selection tests pass. The actual Release rebuild (`q127`) passes in
178.0 seconds with CUDA 80/86/89/90 and ROCm gfx906/gfx1100 unchanged, including
the mandatory spill guards.

The Release canary then repeats all 16 histories and reset. All 112 final-logit
observations are byte-identical to the accepted Integration `q125` observation;
the maximum change in KL is zero. The unscored report remains `UNCALIBRATED`.
Both this run and the following generation benchmark retire every native and
guardian process, leave a clean driver window, and return all eight GPUs to
their exact 10,866,688-byte idle baseline. No host kernel or driver changed.

The unprofiled public `llaminar2 benchmark` uses the original physical domain
order, fixed-head GPU ngram ranges and precision policy. Context is 4096,
prefill uses the default M512 policy, expert placement is Static/Ordinal,
and MTP and prefix caching are off. The production workload is 512 raw prompt
tokens (`" test"` repeated 512 times), 256 generated tokens, greedy sampling,
seed 42, one warmup and three measured iterations. Each request supplies one
output from prefill logits and 255 during the timed decode phase.

| Phase | Three measured rates, tok/s | Median tok/s |
|---|---|---:|
| Prefill | 733.0481, 732.9226, 732.1396 | **732.9226** |
| After-prefill decode | 37.40243, 37.25986, 37.07988 | **37.25986** |

All three requests complete their full output budget. Retained capture/replay,
original participant/owner bindings, native fixed-head terminal observations,
reset and passive counters validate. Stage clocks and profiler instrumentation
are off. These rates establish a local diagnostic baseline; they do not certify
an image, a full preflight run, a performance improvement, or proximity to a
hardware limit. The next step is separate stage/kernel attribution of the same
workload. Capture-construction durations are not inference-stage timings.

The ordinary benchmark exports exact raw prompt identity, actual token count
and all generated token IDs, but no prompt-token-ID array. No cross-engine
token equality is claimed. A first wrapper attempt rejected an inherited
`HSA_OVERRIDE_GFX_VERSION` before native launch; the completed R2 invocation
uses the canonical sanitized environment with that override absent. No ISA
spoofing or visibility masks enter either completed native run.

Local evidence:

- `/tmp/qfn-full-preflight-after-cpu-next/q127-affinity-publication-release-build.json`
- `/tmp/qfn-release-final-logit16-smoke-20261112/report.json`
- `/tmp/qfn-release-final-logit16-smoke-prep-20261112/post-retirement-and-q125-audit.json`
- `/tmp/qfn-release-pp-ep4ep4-baseline-r2-20261112/report.json` and `benchmark.json`
- `/tmp/qfn-release-pp-ep4ep4-baseline-r2-prep-20261112/post-retirement-audit.json`

These retain the exact commands, Release identities, source delta (only the two
CMake guards), actual resources, driver observations and cleanup. Run artifacts
remain outside the source repository.
