# QFN continuation follow-up qualification

This records the follow-up to feature checkpoint `c4cad15348ef70b15ccfef5ac94ad317edae73f7`
on `feat/qfn-ngram-continuation`. The joined follow-up passed complete
model-free prerequisites at r199; the later source-identity correction has
focused validation and fresh numerical diagnostics as scoped below. The earlier ConsumerLocal model
panels and Release rates remain in
[the original qualification note](QFN_CONSUMER_LOCAL_QUALIFICATION_2026-11-02.md).

## Current qualification boundary, 2026-11-03

The canonical r199 prerequisite collection completed with **4,042 actual passes,
zero failures and 29 explicit hardware-unavailable skips**: 981 Unit, 1,827 host,
282 CUDA, 387 ROCm and 565 exclusive registrations passed. The absent CUDA4/8
and ROCm8 cases remain unqualified. Native process retirement and the passive
driver window passed. The original receipt is
`/tmp/qfn-read-ahead-root-integration-r1/canonical-prerequisites-r199/gates/prerequisites.json`.
It certifies that source/build boundary, not subsequent changes or an image.

After the host reboot, Linux renumbered the unchanged model filesystem from
device 259:2 to 259:1. Frozen-reference admission now compares the durable file
identity (path, inode, size and nanosecond modification/change times), then pins
the complete current device identity for strict before/after checks during the
run. It does not read or hash model payloads, rewrite the reference, or permit
mid-run changes. The actual Python/reference/native regressions pass, including
old-code negative controls; the rebuilt native canary and ten focused CTest
registrations pass. Full prerequisites have not been refreshed after this
narrow correction, so r199 is retained as historical baseline evidence.

Six fresh ROCm TP4 ConsumerLocal diagnostics passed the existing absolute pilot:
RAM, SSD mmap, SSD plus 256 MiB/1 GiB per-card caches, and full RAM backing plus
the same two cache grants. **All six have identical metrics for all 178 scored
rows**, exact reset replay, original-source/production-path/publication proof,
successful retirement, and clean driver windows. Raw logits were removed.

| Required panel/phase | Rows | Mean KL (nats) | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|---:|
| Short prefill | 16 | 0.043370 | 87.50% | 88.75% |
| Short decode | 96 | 0.027889 | 98.96% | 92.08% |
| Heldout-short prefill | 8 | 0.016847 | 100% | 92.50% |
| Heldout-short decode | 48 | 0.005527 | 100% | 92.92% |

The remaining ten QSA-boundary rows are diagnostic. These runs retain the
original CPU1 apportioned-expert configuration and MTP off with retained
capacity 15. They do not qualify CPU2 tensor-sharded experts, shared GPU owners,
hot replicas, or long-context behavior. They are local diagnostics, not a
completed baseline/calibration approval or a production certificate. Evidence:
`/tmp/qfn-read-ahead-root-integration-r1/tp4-numerical-diagnostics-r205/summary.json`
and `row-comparison.json` in that directory. No new Release speed is claimed.

The six historical runs exposed a resource-observation gap: their extra RSS
sample followed the outer MPI wrapper rather than the engine descendant. The
local canary observer now follows actual child links from its original launched
PID and pins each descendant to kernel start ticks. It samples memory per
process, follows children created by helper threads, retains known descendants
after parent exit, and rejects PID reuse. Shared RSS is never summed into a
second physical-memory total. Fourteen focused tests pass, including an actual
spawned child with its own allocation; six canonical related CTest registrations
pass. Evidence is in `/tmp/qfn-local-process-observer-r215/`.

This improves subsequent observations; it does not retroactively supply the
missing engine RSS in the six earlier runs, nor qualify a memory-growth envelope
without the corresponding admission baseline and workload observations.

Checkpoint `31449ce1a` contains the qualified continuation follow-up and source
identity repair; `726049c38` adds the explicit `expert_*` CLI keys. Both are
pushed to `feat/qfn-ngram-continuation`. The CLI slice passed all 981 Unit
registrations in its latest complete run. Earlier attempts exposed intermittent
CPU packed-weight retirement and MPI root-publication failures that remain
under investigation; passing isolated repeats is not a fix. Full production
prerequisites after r199 still need refreshing.

## Current correctness boundary

The joined changes repair duplicate TP routing-history publication, preserve
the original failed source group through native retirement, and distinguish a
request failure after proven retirement from missing native retirement proof.
Only the authenticated TP root publishes the logical routing history; every
participant still executes its expert work. The actual-stage replay covers
TP2/4/8 and nonzero roots. Unknown native failure remains fatal and takes
precedence over a safely retired semantic failure.

The joined immutable-record optimization cooperates within the existing owner
kernel to validate immutable fields. It preserves the original ordered commit,
including partial commits on late failure. No new live ledger or allocation is
introduced. The independent CPU scalar implementation remains its oracle.

The rebuilt complete Unit run passed all 975 registrations. The eleven new
host preflight regressions pass across the original run and the scoped repair
of one incorrectly registered target. That target originally ran zero tests and
failed; its corrected owning executable ran four tests twenty times. This is
not a complete preflight result.

Native validation completed with no skips and clean driver windows:

- 131 CUDA/ROCm owner-record and existing pooled regressions passed.
- Six actual captured-channel cancellation cases passed, covering cancellation
  before acquire and absent/late peers after publication.
- CUDA NoPeer source controls passed with zero and positive cache capacity.
- On the actual RX 7900 XTX/MI50 pair, the observed NoPeer source controls passed;
  another 26 owner-record/identity checks exercised gfx1100.

These are 167 native executions across 141 distinct registrations. Success
coverage of the NoPeer transport and standalone channel cancellation does not
yet certify failed multi-consumer model retirement. Shared-owner compute
continuations remain guarded until their full production wiring is complete.

The first cancellation fixture incorrectly tried to record an external event
inside capture. The backend rejected it before launch. The corrected fixture
submits its captured operation, records the exact external progress event on
that stream, and submits the second operation before host observation. It
retains the original two producer attempts and twenty absorbing replays; this
proves two retained submissions, not two operations inside one captured parent.

## GPU calibration unblocks the initial movement cases

The 24-case matrix completed: CUDA/ROCm, zero/64-row GPU ngram cache,
ordinal/random initial expert placement, and fixed1/fixed3/dynamic MTP.
Every case failed only the four required physical-movement observations,
each checked twice. There were no skipped cases or other exceptions. The
original model inputs remained unchanged, native/process retirement completed,
and the driver window passed with no findings. Final metadata-only cleanup
released 34,186,058,909 bytes in 240 prefix-cache files.

The failure is genuine missing migration, not a mislabeled copy counter. The
economy certifier remains in `AwaitingServiceEvidence`: some GPU expert
class/phase measurements are never produced by the short native workload. The
existing prepared CPU measurement explicitly skips GPU participants. Transport
calibration submits real transfers and then aborts their staged placements;
those bytes do not prove a committed migration. The movement assertions remain
unchanged. A joined candidate extends the existing prepared-expert measurement
owner to the missing GPU classes, using original weights, PMA declarations,
exact streams, captured work and event timing. One cold calibration owner and
timing pair per GPU are reused across completed definitions; class/phase counts
do not multiply the conservative General reservation. Timed submissions are
enqueued in one worker closure so control-thread handoffs do not inflate GPU
service time. The coherent build and direct CUDA/ROCm native regressions must
pass before retrying the original movement cases. Those focused prerequisites
now pass. The original CUDA/ROCm fixed1, ordinal, 64-row-cache pilot then
failed during startup, before movement or numerical execution: the serving
owner had reserved the complete `NativeGraphExecutable` budget, including the
new calibration allowance. Calibration attempted to reserve that allowance a
second time. CUDA requested 23,068,672 bytes after all 3,957,325,824 admitted
bytes were reserved; ROCm requested 2,097,152 after all 517,996,544 bytes were
reserved. Both processes retired cleanly and the driver window remained clean.

The narrow repair assigns calibration a distinct typed
`NativePreparationGraphExecutable` owner inside the same physical authority.
It preserves the existing General allowance and total planned bytes. Purpose
is fixed before allocation, and every known and opaque calibration claim uses
that owner. Tests hold the entire serving reservation while calibration runs,
so the original double reservation fails deterministically. The reviewed nine-file repair is joined and built coherently in Integration
and Release. Complete Unit passes all 975 registrations, and the scoped set of
23 host and three native calibration checks is green across CUDA, MI50 and RX
7900 XTX. One existing twenty-process host replay exceeded its unchanged
30-second deadline while the root overlapped compilation and testing; all its
completed repetitions passed. Its unchanged quiet rerun passed all twenty in
18.58 seconds. The original failed receipt remains preserved.

Both original model pilots now pass: CUDA in 223.43 seconds and ROCm in 232.45
seconds. They retain all four actual physical-movement assertions, exact serial
MTP tokens, verifier rows and accepted-draft witnesses. Each fixed1 witness
accepted one draft token and matched its serial oracle. Native retirement and
the complete driver window passed; original GGUF metadata is unchanged. The
harness removed twenty owned prefix-cache files totaling 2,848,838,241 bytes
without reading their payloads. The remaining 22 original dynamic-expert cases
also pass from that same source. All 24 retain the four physical-movement
assertions, serial token equality and accepted-draft witnesses; all eight
dynamic cases retain their controller witnesses. The bounded CSVs contain
120,800 verifier-row comparisons, all byte exact. Both driver windows and
original GGUF metadata checks pass. Across both runs, cleanup removed 240 owned
prefix-cache files totaling 34,186,058,923 bytes without reading their payloads.
This is the scoped movement/exact-MTP gate, not a new numerical panel or full
production preflight.

The current Integration and Release builds are green with the shipped CUDA
80/86/89/90 and HIP gfx906/gfx1100 targets and mandatory spill guards. Complete
Unit passes all 975 registrations; another 22 focused host checks pass. Native
calibration and related regressions pass in 21 executions: seven each on CUDA,
MI50 and RX 7900 XTX. All completed native driver windows are clean. This remains
scoped qualification, not complete production preflight or model accuracy.

The full Unit run exposed an incomplete channel-planning fixture. Its original
16-column geometry now declares FP32 expert source metadata explicitly; it
cannot represent a complete Q8 block. Production admission remains strict. The
corrected 63-case planner executable and subsequent complete Unit run pass.
Three XTX cancellation tests initially stopped before execution because the
local device mask exposed only one GPU. Their unchanged two-device registrations
pass with the actual XTX/MI50 pair. No timeout or assertion was relaxed.

## Isolated owner-record timing

A candidate/baseline/candidate sequence held the other
joined fixes constant and changed only the owner-validation header. The final
candidate was restored before the current model matrix. All cases use the
original complete group timing interval, ten measured samples, two retained
request intervals, and OS-hot source pages. They are component measurements,
not SSD-cold latency or full-model throughput.

| Workload per GPU | Backend | Previous record validation | Cooperative record validation | Speedup |
|---|---|---:|---:|---:|
| 128 live rows, GPU hits | CUDA TP2 | 1.929 ms | 1.691 ms | 1.141x |
| 128 live rows, GPU hits | ROCm TP2 | 6.476 ms | 5.971 ms | 1.085x |
| 16 live rows, GPU hits | CUDA TP2 | 0.589 ms | 0.556 ms | 1.060x |
| 16 live rows, GPU hits | ROCm TP2 | 2.548 ms | 2.449 ms | 1.040x |

The repeated candidate measurements support the cache-hit improvement on both
backends. Cache-disabled controls vary and do not establish a general gain.
An approximately 0.7 ms ROCm difference against older package timings was not
reproduced in the profiled comparison: its complete group difference was about
49 microseconds. Overlapping GPU and host intervals must not be summed.

Offline inspection found a separate concrete compiler issue: outlining the
small source error helper caused `retireResponse` to copy its entire 1,808-byte
argument into private storage, even on success. A private force-inline
candidate removes the copy on gfx906 and gfx1100. Private storage drops from
1,824 bytes to zero, outlined calls from two to zero, and VGPRs from 29/44 to
16/16 respectively. Both versions have zero register spills. CUDA retirement
resources are unchanged on sm80/sm86/sm89/sm90. The candidate passes focused
object compilation with the original spill guards, canonical rebuild and native
regressions on all three local GPU architectures. Quiet timing remains pending.

## Additional exact-identity candidate held

A CTA-local exact-identity candidate passed 127 focused CUDA/MI50/host checks
and another 61 checks on the RX 7900 XTX, with clean driver windows. Two matched
original/candidate timing pairs show modest gains for 128-row GPU-cache hits,
but a repeatable 3–4% regression in the ROCm cache-disabled workload. Small-row
measurements vary. The five candidate changes were removed from the shared
qualification tree while a narrow small-count refinement is evaluated. This
keeps the calibration repair independent. Detailed measurements and isolated
kernel counters are retained in
`/tmp/qfn-read-ahead-root-integration-r1/cta-identities-evaluation-r148.md`.

## Evidence and next work

The following are local evidence directories beneath
`/tmp/qfn-read-ahead-root-integration-r1/`:

- `reviewed-continuation-integration-build-r134` and
  `owner-record-candidate-release-build-r136`: current coherent binaries.
- `complete-unit-and-new-host-r131` and
  `retained-failure-registration-repair-r133`: complete Unit and scoped host checks.
- `cooperative-records-native-complete-r133`,
  `native-channel-and-nopeer-cuda-r134`, and
  `gfx1100-records-and-nopeer-r134`: native receipts and driver windows.
- `owner-record-baseline-controls-r135`,
  `owner-record-baseline-one-token-controls-r135`,
  `owner-record-candidate-controls-r136`, and
  `owner-record-candidate-one-token-controls-r136`: matched timing samples.
- `source-publish-rocm16-trace-r137` and
  `source-publish-rocm128-trace-r137`: profiled attribution, not canonical timing.
- `consumer-local-dynamic-experts-r138`: complete red matrix, failure summary,
  clean driver report and owned prefix-cache cleanup.
- `join-prepared-service-r141`, `prepared-service-integration-build-r143` and
  `prepared-service-release-build-r143`: reviewed calibration/inline candidate
  and current coherent builds, including the test-only fixture correction.
- `prepared-service-unit-native-r142`, `prepared-service-unit-gfx1100-r143`
  and `prepared-service-gfx1100-pair-r143`: scoped checks and original outcomes;
  the last two establish the green complete Unit/host and XTX native slice.
- `prepared-service-movement-pilot-r143`: original two-cell startup reservation failure.
- `join-calibration-owner-r148`, `calibration-owner-integration-build-r148`
  and `calibration-owner-release-build-r148`: exact typed-owner repair and
  source-identical complete builds.
- `calibration-owner-unit-native-r148` and
  `calibration-owner-quiet-host-gfx1100-r148`: complete Unit, host regressions,
  native calibration, original timeout and unchanged quiet rerun.
- `calibration-owner-movement-pilot-r148`: both model pilots green, exact MTP,
  original movement assertions, complete driver window and owned-cache cleanup.
- `calibration-owner-movement-remaining-r148`: remaining 22 original cases
  pass; clean driver window, unchanged source models and owned-cache cleanup.
- `calibration-owner-movement-summary-r153.json`: the completed original24
  matrix, verifier rows, accepted drafts and dynamic-depth witnesses.

The isolated compiler candidate and before/after ISA are in
`/tmp/qfn-retire-reject-inline-compile-r139/`. The profile interpretation is in
`/tmp/qfn-source-publish-comparison-r137/`.

## Shared-source candidate and next coverage

The reviewed r153 candidate is joined and built coherently. It retains one
original source/cohort, exact result and all-reader ACK ownership, cancellation
after native producer proof, and explicit peer-VRAM or mapped transport from
the original directed inventory. The first native cases use ROCm TP2 with GPU2
storage, and CUDA TP2 with ROCm0 storage. CUDA admission retains the same source,
PMA and channel protocol; only two duplicated ROCm planning restrictions were
removed. Host replays cover original and new roles. Native/model qualification
is still pending.

Inspection of the completed dynamic runs also found a stale evidence consumer:
it treated response-continuation admission tickets as committed transactions.
The original four ROCm dynamic CSVs truthfully retain that reporting failure.
The joined test-side correction authenticates admission, observation and first
submission separately, including identical lifecycle words and ordered counts.
Its 19 evidence tests pass twenty repeats, including repeated same-depth epochs
and missing/reordered witnesses. The old consumer fails the new positive test.
Runtime tickets, controller behavior, arithmetic and numerical limits are unchanged.

The user clarified that the primary placement is ngram shards on GPUs already
performing continuation or follower expert work, across either or both domain
types. Fixed-head shard admission and independent ConsumerLocal caches provide
partial coverage, but do not qualify shared adaptive ownership across those
domains. The first shared-source topology still requires a distinct storage
GPU. The next implementation/testing priority is continuation-only,
follower-only and combined ownership; dedicated storage remains a control.
See the [pooled placement contract](QFN_POOLED_NGRAM_CONTINUATIONS.md).

The joined shared-reader Integration and Release builds passed. The corrected
focused batch then passed all 1,011 selected checks, including the complete
975-test Unit suite, shared-reader/native channel coverage and the explicit
fixture replays. The driver interval contained 56 new records and no findings.
The prior batch's five strict peer-payload failures were ineligible hardware
selection: this host intentionally has no direct P2P. Those tests remain
registered for capable hardware; the actual-inventory admission gates prove
the absent-peer case here. The other failures were fixed test fixtures and
exact terminal/reset synchronization-policy registration. No production guard
was relaxed. Evidence: `/tmp/qfn-read-ahead-root-integration-r1/shared-reader-unit-native-r154/`.

The transport baseline is now explicitly NCCL within CUDA domains, RCCL within
ROCm domains, and the optimized cross-vendor method between vendors. This
supersedes the proposed same-vendor mapped/custom-peer selection for ngrams.
Native-library handling of P2P remains transparent to lookup ownership. The
existing equal-prefix native row API does not yet supply ragged directed
payloads, so the exact device-counted native operation must be closed before
this revised path is qualified. Matched fixed-live-payload benchmarks establish
a transport floor; full lookup and inference benchmarks decide useful tuning.

Next: complete the native payload and colocated graph integration, then extend
actual source/execution cells to those continuation/follower roles.
Keep dynamic expert balancing, off/fixed1/fixed3/dynamic MTP, tier promotion and
eviction, exact live communication, frozen-history numerical panels and matched
Release timing. Refresh complete preflight after focused qualification. Earlier
model accuracy does not certify new production changes, and missing hardware
lanes remain explicitly unqualified.

## Six-hardware source controls and current defects

The requested six hardware rows now live in the existing typed definitions.
Eight owner configurations cover the mixed row's continuation-only,
follower-only and combined ownership. The cheap metadata sweep covers seven
storage/tier policies and MTP off/1/3/dynamic; it grants no execution result.
Twelve separate whole-RAM/SSD source controls use production expert residency
planning, MTP off and no retained predictor. They share their exact case producer
with the numerical/performance projection. Correctness quotas and retained15
are not inherited by performance. Fresh numerical panels remain required.

The first native batch failed all twelve before loading: the resource BOM
required original prepared GPU workers, but the runner used that path only for
adaptive GPU ngrams. RAM/SSD went through legacy worker creation and a late free
memory sample. The shared repair prepares every selected GPU through the existing
`NativeContextPreparation` and PMA before sizing; adaptive storage only adds its
authored owner endpoints. Reuse retains the same source, observed capacities and
worker generation. CPU-only preparation remains a no-op.

Four actual-method regressions cover all storage policies, source/PMA membership,
legacy-worker rejection and CPU-only behavior. All four fail against the original
method. Each has an explicit production-preflight entry. Coherent production
builds and the refreshed complete Unit suite plus seven focused preflight checks
pass **982/982**. This is not a full production-preflight result.

The next twelve-source batch passed all eight RAM/SSD cells on CUDA1/2 and
ROCm1/2. Both ROCm4 cells fail because an all-GPU expert assignment leaves an
empty CPU tier while graph construction still requires a remote sparse return.
Both mixed CUDA2+ROCm4 cells fail because rank-local follower GPU work reaches
the continuation graph's foreign-device guard. Those failures remain positive
execution obligations; no guard is removed to turn them green.

Every cell proved runner/native retirement, but the batch driver window is red:
one `amdgpu 0000:8c:00.0: ih2 ring buffer overflow` warning. Passive inspection
still reports a live MI50 with a 16 GT/s, x16 link. This is not evidence of
another lost-link event, nor permission to ignore the warning. Native
qualification requires a subsequent complete clean window.

Evidence under `/tmp/qfn-read-ahead-root-integration-r1/`:

- `six-hardware-source-r156/`: original twelve common setup failures.
- `six-hardware-source-r157/`: eight source passes, four graph failures and
  the failed driver window, with complete retirement receipts.
- `all-storage-preparation-regression-build-r158-r2/` and
  `all-storage-unit-build-r158/`: coherent production/test build results.
- `all-storage-unit-preflight-r158/`: all 982 selected checks passed.

The separately measured fixed-payload transport floors and CUDA native-memory
observation remain described in
[the transfer investigation](QFN_NGRAM_NATIVE_TRANSFER_FLOORS_2026-11-02.md).
Neither source admission nor those component timings certifies adaptive
colocation, full lookup latency, final-logit accuracy or full-model speed.

## Empty expert tiers and continuation-colocated storage

The TP4 source failures exposed three uses of authored CPU capacity as though it
were admitted CPU execution. The original graph now omits the external sparse
return only when the frozen quota and complete owner map prove no external
contribution. The same admitted participant projection selects a complete native
capture envelope and avoids constructing unused CPU dispatch stages. An
unresolved quota or any positive quota retains the external endpoint; a zero
quota contradicted by actual ownership is fatal. PP layer origins are explicit.
The conservative pre-admission service BOM remains unchanged.

The refreshed Unit suite and explicit affected preflight entries pass 984/984.
Both TP4 RAM/SSD configurations now initialize with the complete native parent,
but their source-admission checks still fail at model-context reuse: the
prepared-weight authority list includes the CPU despite its frozen zero expert
quota. That weight-ownership projection is being corrected without deleting
physical runtime or ngram owners. This is not yet a source-admission pass.
The latest run retires all resources, and its driver window has 31 records and
zero findings. Earlier failed runs and the earlier AMD warning remain retained.

The continuation-colocated storage composition is now joined. The original
LocalTP capture-end boundary shares the immutable source definition; each native
parent cache owns its finite storage-body borrow before SDK mutation. CUDA
joins the storage terminal before closing mapped progress. ROCm uses native DAG
roots and terminals. Source sealing follows the original executable creation,
and the enclosing request parent retains ownership until physical completion.
No extra source registry, live-count readback, fake completion or graph recapture
is introduced. Expert-follower attachment and the native counted-collective
recipe remain separate, unfinished obligations.

The package's device-free adapter, native-DAG and lifetime replays pass, including
an original-code negative control and zero-allocation checks of the added warm
validation. Ten explicit CUDA/ROCm native replay registrations cover empty and
partial live payloads, repeated parents and foreign-device rejection. All ten
pass in the coherent r164 native run; these component tests do not
replace the original-model colocated admission and numerical panels.

Additional evidence under `/tmp/qfn-read-ahead-root-integration-r1/`:

- `tp4-source-r159/` and `tp4-source-r160/`: the preserved capture-policy and
  unused manual-dispatch failures, with clean driver/retirement windows.
- `no-empty-host-unit-r161/`: 984/984 selected checks pass.
- `tp4-source-r161/`: successful initialization, reuse check failures, complete
  retirement and clean driver window.
- `join-colocated-composition-r162/`: exact 45-path reviewed integration receipt.
- `join-fixed-floor-original-owner-r162/`: original-source collective fixture
  repair; no production memory allowance was raised.

## TP4 source admission closure

The prepared-weight owner repair now consumes the original admitted tier
projection. Frozen-zero external endpoints retain their physical runtime and
storage ownership, while reuse stops demanding prepared weight records that
they never load. Dense/shared roles and unresolved or positive expert quotas
remain required. The existing final-retirement regression proves that a
storage-only original GPU source is retained and retired correctly.

The next native run exposed a test-only assumption: CPU activation/workspace
capacity reserved before quota resolution was treated as a requirement for live
CPU compute storage after all experts had been assigned to GPUs. The source
observer now consults that same production owner projection using the exported
original frozen plan. It still rejects absent claims on active CPU paths and
all missing CPU weight claims. Its explicit preflight replay covers unresolved,
incomplete, contradicted, later-positive and shared-expert ownership, different
CPU/rank identity, PP layer offsets, and repeated projection.

Both actual TP4 source controls, whole-RAM and SSD-mmap, now pass. They prove
initialization, exported reusable weight authority, source identity, exact live
claims and complete runner/native retirement. The driver window contains 31
records and no findings. This does not certify numerical inference or speed.
The refreshed complete Unit suite (981 registrations) and 41 affected checks
pass **1,022/1,022**; complete production preflight remains a separate gate.

Evidence under `/tmp/qfn-read-ahead-root-integration-r1/`:

- `join-prepared-weight-owner-r163/`: seven-path shared owner projection.
- `cpu-materialization-observer-build-r167/`: coherent test/canary build.
- `cpu-materialization-unit-r167/`: 1,022 executed, zero skips or failures.
- `tp4-source-r166/`: preserved false-positive CPU materialization assertions.
- `tp4-source-r167/`: both source controls pass with complete clean retirement.

## Matched cache policy admission

The new callable `Qwen4ExpHardwareMatchedCacheEconomy` family compares CUDA TP2
and ROCm TP2 with the same explicit GPU128/CPU384 expert ceilings. Each backend
has whole-RAM and whole-SSD controls, ConsumerLocal caches at 256 MiB and 1 GiB
per GPU, and shared two-owner caches at the same byte limits. The limits are
requests to the existing authority; they never stand in for actual grants.
The source tests use MTP Off with retained capacity zero. The reusable typed
definition retains the full functional MTP and expert-movement axes.

All twelve source cases executed. **Eight pass**: both whole-source controls
and both ConsumerLocal capacities on each backend. All twelve retire their
runner and native resources, and the driver window is clean. The successful
ConsumerLocal cases materialize 265,076,608 or 1,070,382,976 bytes of GPU table
storage on each participant. All eight successful cases materialize identical
routed expert weights: 12,025,856,000 bytes per GPU and 77,817,446,400 bytes on
CPU. This establishes useful positive cache capacities and a matched weight
workload; it is not a numerical or throughput result.

The **four shared-cache cases fail** in the original planning restriction that
permits only the coordinator plus exactly one separate ROCm storage device.
The authored cases instead place owners on both continuation GPUs. Completing
that graph/source-token/parent ownership is the next shared-policy requirement;
removing the guard alone would not implement it. Full-RAM backing for adaptive
caches is also being implemented as a genuine shared source, not an oversized
RAM row cache. No cache policy has been promoted to a production default.

All 32 configured numerical bindings pass actual compiled registry/export and
CLI-parser validation. This proves metadata only; fresh accuracy panels and
Release timing remain pending. Evidence:

- `join-matched-cache-r169/` and `matched-host-diagnostic-build-r169/`.
- `matched-host-diagnostic-focused-r169/`: both new focused registrations pass.
- `/tmp/qfn-matched-cache-configured-r169/`: 32 actual configured bindings.
- `matched-cache-source-r170/`: complete twelve-case source result, clean
  driver/retirement evidence and `positive-capacity-observations.json`.

## Host-service metadata rejection

The refreshed original eight CUDA/ROCm single/TP2 source controls passed six
cases. Both TP2 RAM cases rejected changed CPU-service metadata; all resources
retired, but AMD user-pointer workqueue warnings also failed that run's driver
window. Both cases then passed a fresh diagnostic rerun with a clean window.
Their exact transient metadata change was not captured, so that original
failure's cause remains unproven.

An independent actual-loader regression established an overly broad guard:
loading or unloading an unrelated non-TLS library invalidated a CPU service
declaration despite unchanged resource requirements. An unlinked but still
loaded non-TLS library also failed unnecessary file canonicalization. The
correction compares current TLS geometry and selected source/provider/policy
inputs, while retaining unrelated loader paths and counters for diagnostics
only. It neither raises a quota nor redeclares an existing service. Genuine
TLS or provider changes remain errors. The five focused loader/policy tests
pass twenty repetitions against the corrected body; original-code controls
fail. They are explicitly registered in production preflight. The complete
Unit refresh passes all 981 registrations. Thirteen of fourteen affected checks
pass in that run; the remaining check used a stale executable. Rebuilding its
actual owning target and rerunning that unchanged check passes. The original
failed receipt remains preserved; native refresh is still required.

Evidence: `homogeneous-eight-source-r168/`, `host-evidence-source-r169/`,
`join-host-current-resources-r171/` and `join-host-path-only-r171/` under the
same root evidence directory. Private actual-body regression packages are
`/tmp/qfn-host-evidence-churn-candidate-r169/` and
`/tmp/qfn-host-evidence-path-only-r170/`.

## Joined mixed followers and explicit RAM backing

The r173 composition adds an original rank-owned auxiliary runner for actual
ROCm expert followers behind CUDA TP2. These GPUs retain their original expert
roles and are not inserted into the continuation TP membership. Existing
mapped activation channels carry live expert traffic. Original workers submit
the follower graphs asynchronously; immediate scheduler cancellation preserves
the first endpoint error, and every queued submission requires its original
join and native retirement before reset. Device-free replays cover partial
enqueue, delayed failure, stale identity and attempted reset before join. Native
mixed execution, MTP and dynamic expert movement still need fresh qualification.

Adaptive GPU caches now support an explicit full-RAM original source through
`--ngram-source-backing ram`. One original table is shared by the consumers;
redundant RAM row-cache ceilings reject. SSD backing remains an explicit mode.
The existing PMA prices full-RAM source storage once and retains separate live
channel scratch and GPU cache grants. Eight RAM-backed counterparts derive
from the existing matched cache cases without changing expert budgets. The
wire parser also rejects full-width enum values before narrowing, including
256/257 aliases for both backing and placement; original-body negative controls
prove the regression.

All four packages are joined. The coherent build passes in 662.96 seconds, and
all 40 actual compiled case exports match the retained canonical configuration
and CLI-parser projections. Old numerical evidence is not relabeled to the new
config schema. The independent
native completed-transfer dependency and shared-pool caller adapter remain
unjoined; no new default or performance result is established by this work.

The combined Unit/affected run executes all 1,010 registrations: 1,006 pass,
four fail. The complete Unit portion passes 979/981. The new full-RAM metadata
case exposes a second SSD-only predicate in physical-owner geometry (also
failing its explicit admission replay). A malformed-schema test hardcodes 12,
which is now the current schema; it must derive invalid neighboring versions.
The new follower BOM registration names the wrong suite, and the zero-test guard
correctly rejects it. Both test-only repairs are prepared independently of the
production geometry fix. Audit also finds the terminal stats emitter and
GPU-adaptive source observer still assume SSD backing; those fixes and actual
emitter regressions are required before full-RAM cache qualification.

The actual mapped-activation native suite passes all 42 cases, and the explicit
scheduler-cancellation replay passes two more executions. The driver window is
clean. Both original six-GPU source controls then pass in 127.68 seconds:
CUDA TP2 continuations plus four real ROCm expert followers, with RAM or SSD
ngram backing. Original expert allocations materialize on all six GPUs (about
16.9 GB per CUDA device and 15.6 GB per ROCm device), with no CPU expert weights.
Runner/native retirement and driver checks pass. These tests perform source
admission, not numerical inference, MTP replay or performance measurement.

Evidence under the same root directory: `host-resource-unit-r171/`,
`host-consumer-target-build-r171/`, `host-consumer-rerun-r171/`,
`join-mixed-follower-r173/`, `join-adaptive-full-ram-r173/`,
`join-adaptive-wire-width-r173/`, `join-adaptive-ram-cases-composed-r173/`, and
`configure-ram-follower-r173/`, `ram-follower-build-r173/`,
`ram-follower-unit-r173/`, `follower-native-r173/`, and
`mixed-follower-source-r173/`. The actual 40-case configuration proof is in
`/tmp/qfn-adaptive-full-ram-configured-r172/`. The original adjacent CMake context conflict
made no source changes; the composed case patch preserves both independent
regression filters.

## RAM correction and benchmark lookup evidence

The physical-owner geometry now accepts the original full-RAM source as well
as bounded SSD mapping. The terminal observation accumulator and emitter accept
the explicit full-RAM mode while rejecting RAM-cache or mapping-window activity
under that mode. The source-admission observer derives its expected mode from
the requested placement. Focused regressions exercise the actual emitter,
geometry and service; original-code controls reproduce both defects. The two
test registration/schema defects above are also corrected.

The benchmark JSON whitelist previously discarded `immutable_lookup` and
`ngram_lookup` despite their terminal counters being collected. It now retains
both domains. Actual-emitter replay proves warmup is excluded, measured values
and units survive, and disabled collection remains disabled. Two tests pass
twenty repetitions; the original exporter fails the same observation test.
The regression is explicitly registered in production preflight. This preserves
existing aggregate lookup evidence; it does not invent local/remote attribution
or prove cache economy.

The coherent r175 build passes in 114.78 seconds. Complete Unit (981
registrations) plus 32 affected integration checks passes **1,013/1,013**, with
no skipped or missing cases, in 180.33 seconds. This closes all four failures
in the earlier combined run. Complete production preflight and new numerical
qualification remain pending.

Evidence: `join-full-ram-publication-r174/`, `join-root-test-repairs-r174/`,
`full-ram-repair-build-r174/`, `join-benchmark-lookup-r175/`,
`lookup-evidence-build-r175/`, and `lookup-evidence-unit-r175/` under the same
root evidence directory. The isolated exporter controls are in
`/tmp/qfn-benchmark-lookup-evidence-r175/run-with-original-guardian-r3/`.

## Actual RAM cache grants and incidental file metadata

The four RAM-backed ConsumerLocal source cases finish in 308.11 seconds:
both CUDA cases and the 1 GiB ROCm case pass. Actual GPU table allocations
are 265,076,608 bytes at the 256 MiB ceiling and 1,070,382,976 bytes at the
1 GiB ceiling, independently on each consumer. The shared CPU original table
is materialized once, at 28,800,139,264 bytes including its existing alignment.
Source admission executes no model logits. Every process/native owner retires;
the driver window records no findings.

The 256 MiB ROCm case rejects a changed `libllaminar2_core.so` file fact while
TLS geometry is unchanged. A concurrent private validation helper created a
hardlink to that same inode. Its modification time remained unchanged, but
hardlink creation changed inode ctime. That incidental metadata cannot change
the loaded library's TLS storage; using it in resource equality and live
identity made admission unnecessarily fragile. Validation helpers now use
copies instead of altering the original inode's link metadata.

The correction removes ctime from that resource identity and keeps device,
inode, size, modification time, selected providers and actual TLS geometry.
An actual loaded private TLS fixture replays link creation/removal and chmod,
while independent controls still reject modification and byte-identical file
replacement. All 14 host-service family tests pass twenty repetitions; the
original body fails the new metadata-only regression. Its explicit preflight
registration is `V2_Integration_RetainedHostServiceFileMetadataReplay`.

The next composition also records an immutable, original-owner projection of
admitted ngram slots, cache payload/metadata and source/staging allocations.
It is historical setup evidence, not live occupancy or another admission
ledger. Shared RAM backing appears once. Source-admission CSV and benchmark
setup export consume the same observation; the benchmark keeps only that
exact setup family across reset. Existing lookup work remains measured-only.
These changes are joined and both Integration and Release rebuilds are active.
Native source refresh and complete preflight are still required.

Evidence: `ram-consumer-local-source-r175/`, `join-immutable-setup-r177/`,
`join-benchmark-admitted-setup-r177/`, `join-host-file-metadata-r177/` and
`admitted-setup-metadata-build-r177/` under the root evidence directory.
Private controls are `/tmp/qfn-host-file-metadata-r177/` and
`/tmp/qfn-benchmark-admitted-setup-r176/`; the snapshot owner tests are in
`/tmp/qfn-immutable-setup-evidence-r175/`. Refreshed actual compiled exports
and public-parser projections pass all 40 cases in
`/tmp/qfn-adaptive-full-ram-configured-r175/`.

## Clean rebuild and passive benchmark comparison

Private validation helpers accidentally followed symlinks into three shared
source files: an unjoined reader declaration and two setup-evidence files.
The latter two were immediately restored; the reader declaration was restored
after its missing implementation caused the r177 builds to fail at link time.
No GPU run used those builds. The earlier r175 test results describe their
actual binaries, but are not a clean current-source qualification. All failed
builds and the isolation audit are retained.

The shared source and test trees are now write-protected between reviewed
root-owned joins. Writable private views use physical copies. Ninja's dependency
records identified ten affected Integration objects and five Release objects;
those outputs were invalidated explicitly. The corrected r178 builds pass:
Integration 213.99 seconds, Release 29.41 seconds. A subsequent registration
fix moves the file-metadata regression below its target declaration so it is
actually included in the preflight inventory. Native refresh and complete
current Unit/preflight remain pending.

The existing declared benchmark owner now enables a versioned passive policy
before startup, with CPU/GPU stage clocks disabled. It retains actual admitted
owner setup, measured terminal lookup counters, native submission observations
and the original expert-selection setup. Native source byte validation consumes
the original row-byte width directly. Full-RAM source bytes appear once and
cannot also claim a RAM cache. Aggregate cache hits do not identify local versus
remote hits, submission counts are not wire bytes, and mmap window misses are
not SSD reads. Those limitations remain explicit in each report.

The final five-file package passes all 32 Python benchmark-policy tests, with
actual-owner controls for collection setup and cleanup on validation failure.
The row-width emitter passes its focused replay; the original emitter fails.
Current-source CTest execution is still required; isolated results do not grant
numerical or runtime performance acceptance.

Evidence under `/tmp/qfn-read-ahead-root-integration-r1/`:
`source-isolation-incident-r177/`, `source-isolation-restore-r178/`,
`join-declared-passive-r178/`, `build-lookup-evidence-r178/`,
`build-release-lookup-evidence-r178/`, and
`join-host-metadata-registration-r179/`. The private package is
`/tmp/qfn-declared-passive-evidence-r177-r2/`; its handoff links the original
controls and preserved isolation audit.

The r179 registration refresh passes both builds and all seven focused CTest
registrations (14.84 seconds). Actual RAM-backed ConsumerLocal source admission
then passes **4/4** in 317.99 seconds, including the formerly rejected ROCm
256 MiB case. All native/process resources retire and the driver window is
clean. Both backends admit 1,118,462 slots with 265,076,608 bytes of cache plus
policy storage at the 256 MiB ceiling, and 4,526,334 slots with 1,070,382,976
bytes at 1 GiB. Per-GPU lookup staging is a separate 9,934,336 bytes. The one
shared RAM source remains 28,800,139,264 bytes with 3,514,368 bytes of host
staging; it has no redundant RAM row-cache slots.

Receipts: `lookup-evidence-focused-r179/`, `ram-consumer-local-source-r179/`
and its `admitted-capacity-summary.json`. The complete canonical prerequisite
run is next, followed by matched numerical panels and Release measurements.
No shared-pool or hot-replica policy inherits these ConsumerLocal results.

## Complete regression collection and scoped repairs

The original canonical r179 transaction completed Unit981, host1820, CUDA282
and ROCm389, then refused an old-boot orphan before its592exclusive cases.
The orphan was verified against boot time, absent original owners, shared
backing and process mappings before exact-path removal. The unexecuted original
lane was then collected separately at r184; it is not a synthesized green
canonical receipt. Combined original registrations: **3,988pass,47fail,29
unqualified**. The29 require absent CUDA4/8 or ROCm8 hardware. Both passive driver
windows pass and all native test backing retires. Evidence is retained under
`/tmp/qfn-read-ahead-root-integration-r1/canonical-prerequisites-r179/`,
`remaining-preflight-exclusive-r184/` and `original-complete-collection-r185.json`.

The47 failures reduce to31 pipeline source-publication registrations, two
readonly-fixture copies, one diagnostic-helper boundary, one heterogeneous
MPI idle-reporter registration, and12 hosted-lookup fixture registrations.
Pipeline planning used immutable topology copies made before the original
native source existed. The fix carries that same source into runtime planning
copies before the existing shared BOM deduplication; PMA rules remain unchanged.
Focused actual admission replays pass20times, and the original body fails.
The lookup fixtures now use the existing original prepared source, pre-quota
CPU-service declaration and original PMA; their capture, numerical, failure,
reset and teardown assertions remain intact. Their native reruns are pending.

Eight reviewed packages are joined (35unique source paths); no native SDK or
shared-owner caller candidate is included. `join-reviewed-fixes-r185/` and
`join-hosted-native-preparation-r185/` retain exact edits. The fixture/private
checks pass; coherent Integration/Release builds and focused native checks are
next, followed by a new complete canonical prerequisite transaction.

ConsumerLocal admission now supports homogeneous2/4/8participants through the
existing source and worker machinery. Six new TP4 Off comparisons bring the
Integration paired binding set from40 to46 without changing original histories,
reference rows, arithmetic or acceptance limits. Actual configured exports and
parser validation await rebuild. Paired bindings belong to Integration; the
ordinary Release engine retains its separate build configuration.

The separate TP4 expert-movement workload covers SSD/full-RAM backing, ordinal
and random placement, and Off/1/3/Dynamic MTP. All16 recorded-input canonical
metadata admissions retain384GPU/128CPU experts on every49main/sidecar layers.
All six AutoExperts Off comparisons separately retain512GPU/0CPU experts; thus
their recorded-input policy comparison has no expert-residency confound. Fresh
native setup remains authoritative, especially for the tighter1GiB cache.
The new movement definitions preserve existing physical-movement and observed
speed obligations. They have no inherited model accuracy or economy result.
Evidence: `/tmp/qfn-consumer-local-tp4-movement-r184/`.

Shared native ownership on both TP2 continuation GPUs remains an unjoined
candidate awaiting coherent completed-transfer dependencies and native
qualification. Review found concurrent shared-error-string writes in the new
transfer error path; an owned per-call result and actual-method concurrent
replay are being implemented before qualification. The hot-replica candidate
now has tested native-row operations; symmetric consumer-demand projection,
original pin/ACK lifetime, admitted budget sharing and production wiring remain
in progress. Neither is selectable as a proven hybrid policy or promoted as
a default. Release policy comparisons are still pending.


## 2026-11-03 — TP4 source controls and remaining fixture lifetimes

The coherent r185 Release and Integration rebuilds pass (316 s and 676 s).
All 46 exact canonical paired exports and public configuration projections agree;
the Release parser matches the Integration parser for every one. This is metadata
qualification, not numerical acceptance.

The focused r185 run collected all 56 selected registrations in 470 s: 34 pass,
22 fail, no skips, with complete native retirement and a clean driver window.
The admission propagation correction now admits the original PP plans; matching
request lifecycle cases and the new TP4/TP8 declaration/replay cases pass. The
remaining reds expose three fixture ownership errors:

- Twelve PP preparation and four NoPeerHostTail cases export reusable model
  ownership for inspection but omit canonical final-owner retirement. The fix
  releases inspection/runner borrows and calls the existing retirement API,
  checking actual retired-device receipts. An existing source-final-retirement
  replay now asserts the original missing-reset failure before proper cleanup.
- Two foreign-current reset cases create independent PMAs for persistent native
  process roots. They now predeclare both devices under one original source,
  execute both unchanged device trajectories and retire the source once.
- Four adaptive-owner/source-binding aliases consume a one-shot CPU service for
  C0 and then try to bind a new C4 ModelContext to it. The same complete assertion
  body now runs C0 and C4 through separate existing native process registrations;
  process-root ownership and service policy remain strict.

The three reviewed fixture packages are joined in r186 (five test/registration
paths, no production changes). Syntax, actual device-free replay and registration
checks pass; native rerun and a fresh complete canonical prerequisite gate remain
pending. All original failing logs and discarded drafts remain diagnostic evidence.

Six actual ROCm TP4 source-admission controls pass in 295 s, with clean native
retirement and driver checks: full RAM, mmap, and ConsumerLocal 256 MiB/1 GiB over
both backings. Each GPU receives exactly 1,118,462 rows / 265,076,608 cache+policy
bytes at 256 MiB, or 4,526,334 rows / 1,070,382,976 bytes at 1 GiB; staging adds
9,934,336 bytes per GPU. RAM backing has one original host source. These are real
source grants; full inference/expert residency and model accuracy still need the
paired panel and passive Release observations.

The four separate 384-GPU/128-CPU expert-movement source cases also pass their
admission and retirement checks in 287 s, but their outer driver report is RED.
It records two `amdgpu_amdkfd_restore_userptr_worker` CPU-hog workqueue warnings
(131 and 259 occurrences). No cell failure or reboot occurred. Preserve that red
result and diagnose the observed warnings before qualification; do not weaken the
driver classifier or infer movement/throughput from source admission.

Local receipts: `/tmp/qfn-read-ahead-root-integration-r1/` contains
`reviewed-fixes-focused-r185`, `tp4-source-off-r185` (including the cache-capacity
audit), `tp4-movement-source-off-r185`, and the three `join-*-r186` fixture receipts.
The current 46 parser comparison is in `release-public-configurations-r185`.
Shared native ownership, per-call enqueue diagnostics and local hot replicas
remain separate private candidates pending coherent SDK/native qualification;
no ownership or transport default is promoted by this source-only slice.

The r186 coherent fixture rebuild passed in 106 s (Release remained unchanged).
The actual follow-up native run passes **57/57** registrations in 450 s, with no
skips, no driver findings and complete original resource retirement. This includes
every original failed registration, the new TP4/TP8 admission obligations and the
source-final-retirement replay. It does not replace a complete prerequisite gate.
The earlier movement-window USERPTR warning remains a separate red record; its
bounded review is `/tmp/qfn-userptr-warning-review-r186/REVIEW.md`. The strongest
observed phase is raw non-GEMM weight upload/host registration. Random interval
placement is not proof of causality: workqueue reports use global accumulated
counts with backoff. No unsolicited GPU fault/reset or incomplete retirement was
found. A minimal diagnostic is being prepared before performance acceptance.
