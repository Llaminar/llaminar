# QFN TP4 collective attribution and selector repair

Updated: 2026-10-22. The historical trace below uses L69; selector repair is
checkpoint `7bfca677d`. The later independent short-K checkpoint `e0d2e73b3`
(L74) now owns the accepted **785.397 prefill / 33.882 decode tok/s** observation.
The new component attribution below does not replace that model measurement.

## Fresh accepted-runtime trace

The [L69 diagnostic](../../../../parity-results/qwen-flash-next/a1-control69-rocm-profile/result.json)
completed with unchanged source/runtime identities, clean drivers, normal
retirement and the exact control's 256 generated IDs. It retains the A1
four-MI50 `TensorParallel + GateUpOwnedDownColumns` placement, RAM ngrams,
512-token prompt, 256 emitted tokens, M512, MTP off and prefix caching off.
One warmup and one measured request were profiled. Instrumented throughput is
excluded from the dashboard.

Original JSON graph identities authenticate 7,085,044 captured dispatches and
4,108 graph-launch records across all four agents. Each decode model graph has
3,441 kernels; the measured phase contains 255 model graphs and 256 scheduler
graphs per GPU. All eight measured prefill/decode interval windows close with
no unclassified kernel names. Local graph joins and overlap-aware interval
reports are retained in `/tmp/qfn-a1-control69-profile-joined-r1/` and
`/tmp/qfn-a1-control69-profile-intervals-r1/`.

On GPU 0, the **profiled** decode envelope averages 88.856 ms/token. Collective
union is 39.886 ms/token, including 97 SIMPLE all-gathers and 48 LL all-gathers
per model graph. The first group serves canonical ascending-rank reductions;
the latter includes the routed output column publication. These are waits and
execution inside an instrumented timeline, not independently removable transport
latency. Profiling increases host model-graph submission substantially: its
median API duration is 75.089 ms. One retained request has a 1.779-second first
all-gather while the other GPUs arrive late and finish almost together. Neither
that outlier nor overlap is discarded to manufacture a speedup estimate.

The corrected compact-import experiment remains parked. Compared with the
older retained timeline, the fresh accepted graph has 192 fewer kernels per
decode graph, attributable to already accepted captured-transfer scheduling;
this is not evidence that the parked compact-import code is active.

## Exact component fixture

`V2_Perf_RCCLCanonicalRankOrder_ROCm4_A1` exercises the actual fixed-row
`nativeRows(AllGather)` and production ascending-rank FP32 fold. Four ranks
send **10,240 bytes each** (H2560, M1). Each retained graph contains 97 complete
transactions, using immutable inputs, four receive banks and a separate sum
output. Five warmups precede 31 event-timed samples. Every rank's original
input, every gathered bank and the cancellation-sensitive ordered sum are
checked byte for byte.

Native GPU events measure the slowest participant, including amortized arrival
skew from host submissions. This fixture has no interlayer computation or
optional sidebands; its total cannot simply be subtracted from a model's time.
Protocol selection stays in the native RCCL planner. A fresh filtered process
can explicitly load an isolated dependency through
`LLAMINAR_RCCL_PERF_LIBRARY`; ordinary tests retain CMake's exact selection.

The [complete component record](../../../../parity-results/qwen-flash-next/canonical70-analysis/report.json)
retains all nine unprofiled observations and three separate kernel traces.
The initial protocol trials exposed a correctness defect:

| Requested protocol | Actual captured all-gather | Median transaction, µs |
|---|---|---:|
| Default | SIMPLE | 44.442 |
| `AllGather:LL` | **SIMPLE, incorrect selection** | 44.724 |
| Global `LL` diagnostic | LL | 43.321 |

Each separate trace authenticates exactly 13,968 captured all-gathers and
13,968 ordered folds, 3,492 of each per GPU. Native selection logs agree with
the dispatched kernel symbols. The global LL control also affects the fixture's
setup all-reduces; it is not a proposed production override.

Additional unprofiled component trials request fixed minimum/maximum channel
counts. Every trial passes the exact output and driver/retirement checks. Their
channel settings are recorded requests; separate kernel geometry authentication
has not been performed for every channel arm.

| Requested channels | SIMPLE median, µs | LL median, µs |
|---:|---:|---:|
| 1 | 44.213 | 50.166 |
| 2 | 45.630 | 45.953 |
| 4 | 46.070 | 43.959 |

These single component trials show no large latency improvement. The initial
authenticated LL/default difference is about 2.5%; it does not establish a
whole-model win. Keep automatic production protocol/channel defaults.

## Confirmed native selector defect

The selected RCCL source's `ncclFunc_t` starts with `Broadcast`, `Reduce`,
`AllGather`, `ReduceScatter`, `AllReduce`. Its `ncclFuncStr` instead starts
with `AllGather`, `AllReduce`, `AllToAllPivot`, `Broadcast`, `Reduce`.
The actual `parseList` treats table indices as enum indices. Consequently
`AllGather:LL` changes broadcast, while the real all-gather remains eligible
for SIMPLE. The same mismatch affects per-collective algorithm expressions
and makes tuning-table diagnostic labels misleading.

The repair derives the existing table from the independent canonical
`ncclFuncToString` switch during cold library initialization. It preserves the
table ABI, kernel arithmetic, native work descriptors and protocol defaults.
The regression must execute the actual native parser for every supported
collective prefix in both protocol and algorithm expressions, and require the
original reversed-patch implementation to fail the same oracle. Production
preflight needs the source/DSO-bound proof, and a native captured trace must
show that the repaired `AllGather:LL` request actually dispatches LL.

The repair is implemented. A fresh Release gfx906 dependency build preserves
all 129 native compile commands and ordered link inputs/flags after path
relocation. Its library SHA256 is
`2c01a288945d7c2c4b06c482244de16822612442e6f1eb4c5a0199eca7f7a9c1`.
The [build record](../../../../parity-results/qwen-flash-next/selector70-vendor-build/result.json)
confirms successful completion and unchanged accepted r35 source/build inputs.
Integration and Release application rebuilds also pass; the accepted r35 DSO,
paths, receipts and earlier observations remain intact.

The [functional parser proof](../../../../parity-results/qwen-flash-next/selector70-native-proofs-build-r3/result.json)
executes **341 checks with zero failures** against the repair. The exact
reverse-patched implementation completes the same 341 checks with **121
failures**, including four expected missing-`ReduceScatter` diagnostics.
The probe captures those diagnostics through a local host logger, avoiding
the native logger's HIP device query. Eight device-free verifier tests pass.
`V2_Integration_RCCLCollectiveSelectors` and the packaging verifier are
explicitly registered in production preflight.

The existing protocol-defaults proof also passes against the new DSO, including
1,728,000 concurrent checks; its original-code controls retain their expected
failures. Two initial negative-control setup failures (quoted sibling-header
lookup and a source-directory/output-binary name collision) are retained in the
failed build records and corrected in the final proof driver.

The [complete Unit run](../../../../parity-results/qwen-flash-next/selector70-unit/result.json)
passes **756/756** registrations. The [focused native and packaging slice](../../../../parity-results/qwen-flash-next/selector70-focused/result.json)
passes **9/9**, including captured host transport storage, runtime-generation
retirement and rooted collective dependencies. Driver windows and process
retirement are clean.

The [new dependency's captured traces](../../../../parity-results/qwen-flash-next/selector70-analysis/report.json)
prove the repaired selector: `AllGather:LL` dispatches exactly 13,968 LL
all-gathers, while the automatic arm dispatches exactly 13,968 SIMPLE
all-gathers. Each arm also executes 13,968 ordered folds, with 3,492 of each
operation on each of four GPUs. All 97 native selection messages agree with
the actual kernel symbols. Both captures retain byte-exact inputs, gathered
banks and cancellation-sensitive output.

Separate unprofiled component observations give **45.684 µs** default versus
**42.740 µs** per-collective LL (31 samples each). The request now selects the
intended implementation; this isolated observation is not a whole-model gain
or justification to override automatic protocols.

The [focused A1 Release accuracy slice](../../../../parity-results/qwen-flash-next/selector70-release-accuracy/report.json)
passes every required group, production-path, driver, reset and retirement check.
All [26 files / 185 retained final-logit rows](../../../../parity-results/qwen-flash-next/selector70-release-accuracy/ngram69-logit-byte-comparison.json)
are byte-identical to accepted L69, including reset and diagnostic rows. Short
prefill/decode mean KL remains **0.033535869 / 0.032070057**, top-1 agreement
**93.75% / 97.92%**, and top-5 overlap **87.50% / 92.29%**. Heldout mean KL is
**0.009691368 / 0.003837619**, with **100%** top-1 agreement in both phases.
The loaded native process mapped the isolated repaired DSO. This is the frozen
local numerical slice; complete current-source production preflight and new
model throughput qualification remain pending.


## Gather, fold and arrival attribution after L74

The additive `A1CanonicalRankOrderAttributionP4H2560M1` performance fixture
preserves the original complete control and adds separate gather-only and
fold-only graphs at both one and 97 transactions. Each mode has five warmups
and 31 measured samples, rotating the first submitted rank. Every replay checks
the original input, all four source-indexed banks, independent ascending-FP32
sum and suffix guards. All four stop events complete before any diagnostic
readback. Graph inspection verifies the exact kernel counts. The ordinary
120-second fixture budget and 30-second collective timeout remain unchanged.

The [three unprofiled processes](../../../../parity-results/qwen-flash-next/collective75-protocol-comparison.json)
run automatic selection, `AllGather:LL`, then automatic selection again on the
same repaired RCCL DSO and Release binary. Each passes 24 graph records, 864
rank sample records and six summaries; driver, native retirement and runtime
closure checks are clean. The two performance aliases retain explicit protocol
selection, rather than changing production defaults.

| Protocol request / order | Transactions per captured graph | Complete median µs/transaction | Gather-only median µs/transaction | Fold-only median µs/transaction |
|---|---:|---:|---:|---:|
| Automatic, opening | 1 | 95.520 | 88.960 | 14.400 |
| AllGather:LL | 1 | 79.360 | 72.000 | 14.400 |
| Automatic, closing | 1 | 96.000 | 87.360 | 14.240 |
| Automatic, opening | 97 | 45.615 | 41.367 | 2.293 |
| AllGather:LL | 97 | 42.553 | 38.936 | 2.291 |
| Automatic, closing | 97 | 44.691 | 40.627 | 2.293 |

These are slowest-participant event intervals. At 97 transactions, opening-auto
complete medians by launch position are 45.615, 43.830, 42.134 and 40.435 µs;
gather-only positions are 41.367, 40.393, 39.406 and 38.423 µs. Rotating the
physical first rank changes the complete median only from 45.501 to 45.667 µs,
and gather-only from 41.360 to 41.499 µs. Thus launch-arrival skew is visible,
without evidence here for one uniquely slow GPU. The one-transaction controls
have much more startup/arrival overhead per transaction.

Gather-only clearly dominates isolated fold-only cost. Complete minus gather
is not a pure fold measurement: removing the fold changes dependencies,
submission duration and cache behavior. These measurements cannot be subtracted
from model wall time. The LL request gives a modest component improvement; a
whole-model observation is still needed before considering any policy change.
The earlier separate kernel traces authenticate this repaired selector, while
these unprofiled records themselves prove the requested protocol only.

Every gather still passes exactly 2,560 FP32 elements: **10,240 send bytes per
rank / 40,960 receive bytes per transaction**. The backing allocation is 64 MiB
and is never used as a communication extent. The fixed-native API provides no
physical-link byte receipt; the test records that limitation instead of
switching to the distinct device-counted implementation for instrumentation.

The next private component candidate uses existing TransferEngine captured
channels to gather only the three required peer banks. Its proposed local-aware
fold would consume the original local contribution at its explicit communicator
rank and preserve ascending-rank FP32 additions, avoiding a redundant self-copy.
No production transport, arithmetic or runtime selection has changed. Ownership,
exactness, measured economy and symmetric backend coverage must pass before
that candidate can be considered for production.


## Whole-model LL pilot: decode gain, prefill regression

The [single LL model pilot](../../../../parity-results/qwen-flash-next/collective75-ll-model-pilot-comparison.json)
uses the unchanged archived L74 Release runtime and all original A1 settings.
Only `NCCL_PROTO=AllGather:LL` changes. One warmup and three measured requests
pass with clean drivers, exact runtime/source metadata, bounded resources and
normal retirement. All three pilot output sequences equal the six retained
L74 automatic-protocol sequences, each containing 256 tokens.

| Selection | Prefill median tok/s | Decode median tok/s | Prefill sample range | Decode sample range |
|---|---:|---:|---:|---:|
| Retained L74 automatic, six samples | 785.397376 | 33.881558 | 784.584272–786.028355 | 33.853745–33.926483 |
| LL pilot, three samples | 702.850344 | 34.852762 | 702.579134–703.020254 | 34.778360–34.863189 |

The descriptive change is **−10.510% prefill / +2.866% decode**. This is one
pilot against earlier controls, not a new paired accepted win or a full accuracy
gate. It establishes an unfavorable prefill trade-off for the broad protocol
request. Automatic selection remains the production baseline. Further tuning
targets small decode messages explicitly while retaining an economical policy
for large prefill messages. No environment selector or global native protocol
default was installed.

## Fixed-message channel component 76: exact, slower, not promoted

The component compares the original native fixed-row AllGather plus full-bank
ascending-rank FP32 fold with existing TransferEngine fixed-message channels
plus a local-aware fold. The latter reads the original local contribution at its
communicator position and gathers only the other participants' banks; it does
not alter the order or rounding of additions. Permuted membership distinguishes
communicator position from device ordinal. Functional checks cover independent
native full-bank output bits, signed zeros/nonfinites, cancellation, in-place
output, tails, poisoned own slots and retained large/small/large replay. This
remains a separately named component, with no production transport attachment.

The [CUDA P2 results](../../../../parity-results/qwen-flash-next/fixed-rank76-cuda-timing/analysis.json)
and [ROCm P4 results on the isolated repaired HIP runtime](../../../../parity-results/qwen-flash-next/fixed-rank76-rocm-repaired-runtime-timing/analysis.json)
retain five warmups and 31 paired measured samples per path/count, alternating
path order and rotating first host rank. Timing uses identical packed H2560
bank stride, aligned original source/bank/output offsets and separate immutable
input/output. Per-participant event intervals include waits and host-arrival
skew. They are not pure link latency or a model speedup prediction.

| Backend / participants | Transactions | Native median µs/transaction | Channel median µs/transaction | Native range | Channel range |
|---|---:|---:|---:|---:|---:|
| CUDA / 2 | 1 | 49.792 | 51.872 | 46.240–66.720 | 49.312–67.712 |
| CUDA / 2 | 97 | 15.101 | 18.994 | 14.812–15.327 | 18.793–19.906 |
| ROCm / 4 | 1 | 170.560 | 164.800 | 127.199–201.279 | 122.400–204.800 |
| ROCm / 4 | 97 | 41.519 | 45.209 | 41.122–43.771 | 44.282–46.431 |

At 97 transactions the channel component is **25.8% slower on CUDA / 8.89%
slower on ROCm**. The count-1 spread and different arrival sensitivity do not
override that result. Both processes pass exactness, normal native retirement,
unchanged source/runtime metadata and clean driver windows. The fixed messages
are exactly 10,240 logical bytes per directed peer edge; physical-link receipts
are unavailable. CUDA P2 is not an A1 TP4 topology result. Native communicator
storage remains owned by its original context and is not falsely included in
the small fixture's PMA BOM. No production integration or whole-model channel
benchmark is warranted by this component outcome. Accepted A1 remains L74:
**785.397 prefill / 33.882 decode tok/s**, with automatic native protocol.

## HIP argument-geometry defect exposed by long retained families

The initial long ROCm component exceeded its unchanged general graph-family
admission. The cause was native HIP setup metadata: original kernel-node
construction/cloning could leave `alignedKernArgSize` at its default 256 bytes before
parent binding. Graph sizing summed those defaults while packet capture used
the actual kernel/device argument width, forcing repeated pool rollovers. This
was not evidence that the model needed a larger reserve. The
[scoped vendor patch](../../../../scripts/docker/patches/rocm-hip-graph-kernarg-metadata.patch)
resolves actual original node/device metadata before recursive sizing, retains
original device identity across cloning, and propagates metadata/overflow
errors. It changes neither arithmetic, public ABI nor PMA budgets.

The focused `HIPGraphArgumentGeometry.OriginalCloneUpdateChildFamily` proof
uses 97 nodes with a declared 4096-byte trivial argument packet, four retained
original/clone/updated-clone/child variants, and three unchanged replays per
variant. Host argument storage is overwritten after native construction;
original copied arguments, exact outputs and suffix guards must survive.
Terminal events, final original device and zero PMA claims after retirement are
checked. The compiled gfx906 code object reports a 4096-byte argument segment with
8-byte compiler alignment; the runtime applies 128-byte alignment. Its kernel
uses 6 VGPRs and 11 SGPRs with no spills, private storage or shared memory.
The fixture consumes the declared packet directly, without a private copy.

The [meaningful original-runtime negative](../../../../parity-results/qwen-flash-next/hip-kernarg76-original-native-negative/native.log)
records **6 MiB per variant / 24 MiB total** against the existing **8 MiB** family
budget, then fails that exact admission check with native exit 1. The failed
receipt remains failed, with no kill/survivors, unchanged source/runtime and a
clean driver window. The
[isolated repaired-runtime positive](../../../../parity-results/qwen-flash-next/hip-kernarg76-repaired-native-positive/native.log)
records **2 MiB per variant / 8 MiB total**, passes the same budget and all
four-variant/three-replay exact-output checks, and retires with native exit 0 cleanly.
Total-device-memory identity is unchanged throughout both setup observations.
The earlier `hip-kernarg76-original-negative` attempt only lacked a built
executable after compilation failed; it is preserved as setup failure and is
**not** a numerical or runtime-budget negative.

The isolated dependency's source/patch receipt is retained at
`/tmp/qfn-hip-kernarg76-runtime/share/llaminar/hip-graph-runtime.txt`; selected DSO
and loader facts are attached to the original negative, repaired positive and
ROCm component receipts. These do not imply the installed runtime has been
promoted or that complete preflight/model qualification passed. The surrounding [24 model-free graph/lifetime/packaging tests](../../../../parity-results/qwen-flash-next/hip-kernarg76-related-gates/result.json)
passed with no skips, normal retirement and a clean driver window. The complete
[Unit inventory](../../../../parity-results/qwen-flash-next/hip-kernarg76-complete-unit/result.json)
also passed **757/757**. The frozen [A1 Release accuracy slice](../../../../parity-results/qwen-flash-next/hip-kernarg76-release-accuracy/report.json)
is green with exact reset replay, normal native retirement and a clean driver
window. All **26 files / 185 logit rows** are byte-identical to accepted L74,
including required and diagnostic rows. The observer/core are unchanged copies
of the real Release files; passive loader receipts prove the isolated repaired
HIP DSO was selected. No model throughput improvement is claimed by this runtime
repair. All-model and full production preflight qualification remain pending.

Focused registration is explicit: `V2_Integration_HIPGraphArgumentGeometry`,
`V2_Integration_TPRankOrderedLocalSumContract`, and
`V2_Integration_CapturedFixedRankGather_CUDA2` / `_ROCm4` belong to
`ProductionTestPreflight`. Unit geometry and opt-in performance aliases are
separate; timing does not substitute for those correctness entries.

## Batch-only 16 KiB candidate 77: small ROCm component gain

The batched transfer selector now keeps each endpoint in one acquire/copy/publish
CTA through 16 KiB, or 64 bytes per thread. Individual non-batched transfers retain
their separate 8 KiB boundary. The original kernel bodies, channel protocol and
exact live-copy extents are unchanged. A packet fuses only when every endpoint
fits; a larger entry selects the existing parallel implementation for the whole
packet. Device-counted capacity selects geometry, while the acquired live extent
still owns actual communication.

All [24 focused checks](../../../../parity-results/qwen-flash-next/captured-batch77-focused-gates/result.json)
pass, including the four backend directions, partial packets, boundary/tail and
large/small replay cases, and actual successful host-submission counters with
failed-enqueue controls. Eight new explicit preflight entries cover the batch
boundary, fixed-rank components and counter observation. These are focused results,
not complete production preflight qualification.

The complete [Unit suite](../../../../parity-results/qwen-flash-next/captured-batch77-complete-unit/result.json)
passes **757/757**, with no skips. The frozen [A1 Release accuracy slice](../../../../parity-results/qwen-flash-next/captured-batch77-release-accuracy/report.json)
also passes, including exact reset replay, source/runtime identities, driver
windows and ordinary retirement. All **26 files / 185 logit rows** remain
[byte-identical to accepted L74](../../../../parity-results/qwen-flash-next/captured-batch77-release-accuracy/shortk74-logit-byte-comparison.json).
The isolated canonical HIP argument-metadata repair is selected explicitly;
the installed SDK remains unchanged.

The matched Release component retains the original H2560 geometry, exactly
10,240 logical bytes per directed peer edge, five warmups and 31 paired measured
samples per path/count. Path order alternates and the first host rank rotates.
Actual captured templates contain two fused transfer nodes and one ordered-fold
node per transaction. All original native exactness, runtime/source identity,
driver-window and retirement checks pass.

| Backend / observation | Transactions | Native median µs/transaction | Channel median µs/transaction | Native range | Channel range | Channel wins / pairs |
|---|---:|---:|---:|---:|---:|---:|
| ROCm P4, first | 1 | 271.199 | 250.239 | 250.719–311.998 | 222.879–273.919 | 29/31 |
| ROCm P4, first | 97 | 43.744 | 42.200 | 42.990–45.809 | 40.965–45.171 | 31/31 |
| ROCm P4, confirmation | 1 | 198.080 | 176.159 | 155.040–237.119 | 137.600–213.279 | 27/31 |
| ROCm P4, confirmation | 97 | 44.366 | 42.553 | 43.437–45.438 | 41.488–43.261 | 31/31 |
| CUDA P2 | 1 | 41.184 | 41.248 | 37.600–47.488 | 38.816–47.904 | 14/31 |
| CUDA P2 | 97 | 14.730 | 18.401 | 14.425–14.890 | 18.281–18.546 | 0/31 |

The [first ROCm run](../../../../parity-results/qwen-flash-next/captured-batch77-rocm-timing/analysis.json)
and [confirmation](../../../../parity-results/qwen-flash-next/captured-batch77-rocm-confirmation/analysis.json)
give **3.53% / 4.09% lower latency** than their paired native controls at 97
transactions. The [CUDA run](../../../../parity-results/qwen-flash-next/captured-batch77-cuda-timing/analysis.json)
is still **24.93% slower** than native. Count-1 observations remain particularly
sensitive to host-arrival timing. Cross-run changes from candidate 76 are not a
matched threshold-only speedup, and none of these component medians measures a
whole-model improvement. The ROCm advantage corresponds to only about 0.15–0.18
ms across 97 synthetic transactions if fully exposed.

The first private CUDA analysis rejected the driver's 32-byte local-memory
attribute as scratch traffic. That classification was too broad: all four
compiled targets (SM80/86/89/90) report a 32-byte stack and zero compiler-local
bytes for these batch kernels, with no local loads. Every local store leads
directly to a diagnostic call and trap. SM86 ELF relocations identify every
such call as `vprintf`. The corrected analysis retains the raw driver values
and attaches this exact compiled-code evidence; the native run already passed.
No compiler spill guard, failure diagnostic or production arithmetic changed.
The fused CUDA kernel uses 54 registers and a 256-thread block; this static
evidence does not claim achieved occupancy or a profiler-measured bandwidth.
The fresh gfx906 fused kernel uses 51 VGPRs, 80 SGPRs, a 576-byte argument
segment and 256 threads, with zero spills, private storage or LDS. The separate
acquire and copy/publish kernels likewise retain zero spills/private storage.

There is still no production TP attachment. Accepted A1 remains L74 at
**785.397 prefill / 33.882 decode tok/s**. Candidate 78 below tests the proposed
direct peer-payload fold while preserving ascending-rank arithmetic and original
acknowledgment ordering. Its possible saving was one launch plus local peer-bank
write/read traffic, with unchanged interconnect bytes.


## Direct consumer/fold candidate 78: exact, slower, retired

The bounded prototype kept the original producers and consumed acquired mapped
peer payloads directly into the original ascending-rank FP32 sum. Acknowledgments
followed all readers. Its captured component used two nodes per transaction,
removing one launch and the intermediate peer-bank write/read traffic. For P4,
that removes 61,440 local GPU bytes per endpoint, while the twelve directed
10,240-byte messages remain unchanged. This prototype was never attached to
production TP stages or their sidebands.

All [32 focused registrations](../../../../parity-results/qwen-flash-next/consumer-fold78-focused-gates-corrected/result.json)
passed with no skips and clean driver/retirement checks. The new component
checks covered CUDA2 and ROCm3/4 with permuted participants, tails through
16 KiB, changed input generations, large/small replay, exact local/output alias,
97 retained transactions, untouched poisoned peer banks and complete captured
node inventories. Issuance/ownership and failed-submission checks were included.
The first local CTest filter used unsupported noncapturing-group syntax; that
setup failure occurred before native execution and remains recorded separately.

The [Release build](../../../../parity-results/qwen-flash-next/consumer-fold78-release-build/result.json)
passed for CUDA SM80/86/89/90 and gfx906 with the spill guards enabled. Each
comparison below uses H2560, five warmups and all 31 alternating measured pairs,
rotating the first host rank. Values are the maximum original per-rank native
event interval, amortized over 97 retained transactions.

| Backend | Paired control | Control median µs | Direct median µs | Control range | Direct range | Direct wins |
|---|---|---:|---:|---:|---:|---:|
| ROCm P4 | Previous mapped copy + fold | 42.149 | 47.017 | 41.036–43.144 | 45.885–48.536 | 0/31 |
| ROCm P4 | Native gather + original fold | 43.187 | 46.573 | 42.096–44.541 | 45.179–49.735 | 0/31 |
| CUDA P2 | Previous mapped copy + fold | 19.112 | 24.129 | 18.939–19.296 | 23.346–24.401 | 0/31 |
| CUDA P2 | Native gather + original fold | 14.950 | 22.829 | 14.544–15.340 | 22.563–23.447 | 0/31 |

Both [ROCm](../../../../parity-results/qwen-flash-next/consumer-fold78-rocm-timing/analysis.json)
and [CUDA](../../../../parity-results/qwen-flash-next/consumer-fold78-cuda-timing/analysis.json)
pass every original byte, source/runtime, driver and retirement check. Direct
consumption is **7.84% slower than native on ROCm and 52.70% slower on CUDA**;
it also loses against the previous mapped implementation. Separate one-transaction
observations and every raw pair remain in those reports. No model speed or new
model accuracy result is inferred from this synthetic component.

Fresh final-object inspection reports 50 CUDA registers, a 32-byte diagnostic
stack and zero compiler-local bytes on all four targets. Every local store
terminates in a relocation-proven `vprintf` call and trap; there are no local
loads. SM90's 1,024-byte compiler shared allocation remains explicit. The gfx906
leaf has 65 VGPRs, 78 SGPRs, a 552-byte argument segment, alignment 8, block size
256, and zero spills/private storage/LDS. These are static resource facts, not
achieved occupancy or memory-throughput measurements. The initial artifact
parser assumed REL relocations; its corrected version also handles SM90 RELA.

CUDA memory-checker qualification is **not green**. The
[first diagnostic](../../../../parity-results/qwen-flash-next/consumer-fold78-cuda-memcheck/result.json)
timed out at the existing 120-second test limit. It reported NCCL's explicit
unsupported-kernel attribute probes, which the pinned dependency drains during
initialization. A [targeted diagnostic](../../../../parity-results/qwen-flash-next/consumer-fold78-cuda-memcheck-targeted/result.json)
then instrumented the transfer-batch and direct-consumer kernels, excluding API
probe reports. Native gather completed, but the existing mapped consumer hit
its original 30-second peer deadline at epoch zero before the new direct path
ran. The trap produced NVIDIA Xid43 and a host driver fault. Neither diagnostic
passes; further sanitizer attachments stopped. The later ordinary CUDA timing
process retired normally with a fresh clean driver window. No larger timeout,
production protocol change or numerical tolerance change was introduced.

The losing prototype and its test-only registrations have been removed from
the working source. Its complete patch, source identities and tested Release
build products are retained at `/tmp/qfn-consumer-fold78-rejected-snapshot`;
resource evidence is under `/tmp/qfn-consumer-fold78-release-{cuda,hip}-resource-audit`.
Accepted A1 remains **L74 785.397 / 33.882 tok/s**. Candidate 79 below tests the
existing native RCCL tuner interface for small per-rank LL messages while
preserving the larger-message choices.

## Size-aware native tuner 79: paired decode gain, production integration pending

The isolated v4 plugin selects the lowest original eligible LL cost for positive
AllGather messages of at most 16 KiB **per rank**. RCCL supplies a rank-expanded
byte count: the P4 H2560 case is 40,960 callback bytes and 10,240 bytes per rank.
Ignored costs and the native channel output remain unchanged. Missing eligible
LL costs, malformed tables and unsupported communicator contracts fail before
changing the table. Larger and other collective cost tables remain byte-identical.
This does not change the installed RCCL, its source, or production defaults.

The private package at `/tmp/qfn-size-aware-ll-tuner79-next` binds the original
v4 ABI and behavioral source inputs. Its host-only Release build and all three
CPU tests pass, covering cost-table decisions, concurrent actual-plugin context
lifetimes and strict retired-receipt validation. These are experimental checks,
not new production-preflight registrations. The plugin observes a bounded set of
native planning/capture calls; its logs do not count retained graph executions.

The [boundary control/plugin/control comparison](../../../../parity-results/qwen-flash-next/tuner79-boundary-analysis.json)
passes actual mapped-library/export checks, all byte oracles, original graph
memory bounds, normal retirement and clean driver windows. Each process retains
56 complete captured templates on the same four physical participants. All 24
target native templates use LL; the 16,388-byte control retains the exact original
SIMPLE kernel, grid and block. Mapped-channel templates remain identical. The
automatic controls load no external tuner or network plugin. Two earlier local
capture-log analyses failed because merged stdout/stderr interleaved JSON; those
raw native outcomes remain retained. Separating the streams produced complete
saved records without changing native execution or suppressing diagnostics.

The [Release timing controls](../../../../parity-results/qwen-flash-next/tuner79-timing-analysis.json)
retain H2560, five warmups and every one of 31 alternating path pairs, with
rotating first-rank submission and original 1/97 transaction counts. The following
are maximum-rank native-event intervals amortized over 97 transactions, not model
latencies or sums of independent kernel times.

| Observation | Native gather + fold median µs | Unchanged mapped control median µs |
|---|---:|---:|
| Automatic before | 45.493 | 43.258 |
| Size-aware LL | 41.407 | 42.035 |
| Automatic after | 44.760 | 42.435 |

The native component improves against both bracketing controls; the mapped
control also varies, so the complete model comparison supplies the relevant gain.
Both model arms use the unchanged L74 runtime, RAM ngrams, original six arithmetic
selectors, 512 fresh prompt tokens, 256 emitted tokens, one warmup and three
measured requests per process on the original 28 physical NUMA1 cores. Actual
plugin mapping and retired planning receipts are required separately from the
native benchmark's output and lifecycle checks.

All four [A/B/B/A model arms](../../../../parity-results/qwen-flash-next/tuner79-model-abba-comparison-loader-metadata-v2.json)
pass their original benchmark, source/runtime, resource, driver and retirement
checks. The independent terminal proof also confirms actual original RCCL
mapping, absent external tuners in both automatic controls, and the exact plugin
with all four normally retired communicator contexts in both candidate arms.
Every one of the twelve measured 256-token sequences is identical.

| Policy | Prefill median [min, max] tok/s | Decode median [min, max] tok/s |
|---|---:|---:|
| Automatic, six samples | 785.103 [783.061, 786.170] | 33.947 [33.862, 34.043] |
| Size-aware LL, six samples | 785.609 [784.228, 786.455] | 34.697 [34.645, 34.793] |

The measured change is **+0.064% prefill / +2.209% decode**, with separated decode
ranges. This avoids the earlier broad-LL prefill loss. The plugin's bounded cold
planning observations remain explicit instrumentation; this is a diagnostic
policy comparison, not a new canonical uninstrumented baseline or a full-logit
accuracy certificate. Castagna's 42.959 decode target remains ahead.

The first automatic arm's terminal library audit rejected `/etc/ld.so.cache`
as non-ELF because the passive collector includes that loader metadata. Its
original native result passed. A separately sealed validator now classifies
only that exact path as metadata, preserving both stat checks and rejecting
other unexpected non-ELF mappings. Four device-free controls pass. The original
failure remains intact, and a distinct corrected proof revalidates the saved
run without repeating inference. All four comparison arms explicitly consume
that corrected proof format.

Production integration is still pending. The proposed native dependency policy
keeps the original collectives and no-plugin selection/postpass machinery; it
does not install the experimental plugin. Its eligibility must agree across
every rank and preserve explicit user selectors. A proposed rank-local transport
check was rejected in review because mixed SHM/P2P connections could make ranks
choose different protocols. The implementation must use the original native
clique's fully published connection metadata and prove its ordering and lifetime.
Then the native candidate needs focused regressions, captured selection and
byte checks, the frozen full-logit accuracy slice and matched Release timing.
Accepted A1 remains L74 until those gates complete.


## Native small-AllGather policy 80: accepted focused checkpoint

The native implementation retains the original RCCL selector and postpasses.
After the outermost blocking `ncclCommInitAll` completes successfully, one
control thread validates the entire original clique before publishing the
communicator-local preference. Eligibility requires four gfx906 participants
on one host/process, default CTA policy, no explicit algorithm/protocol selector
or external tuner, four channels, and completed direct SHM ring endpoints on
every owner. Mixed, incomplete, lazy, proxy/CE and other-architecture arrangements
retain their existing native policy. No extra connection, barrier, collective,
host count readback or hot-path allocation is added. The preference changes only
eligible positive AllGather messages up to 16 KiB per rank; channel selection and
native postpasses remain intact.

The isolated [native build receipt](../../../../parity-results/qwen-flash-next/native-small-allgather80-release-build/result.json)
and dependency receipt at `/tmp/qfn-native-small-allgather80-v3-build-r1/dependency-build-receipt.json`
bind the new dependency to the original compile configuration. The complete
gfx906 dependency rebuild took 771.37 seconds. All 132 original compile commands
retain identical normalized flags; the sole additional command builds the new
host policy. Production Release and Integration products select the explicit
new DSO. Paired controls preserve untouched old products; both arms use the same
isolated HIP kernel-argument repair, with no external tuning plugin.

Validation completed so far:

- All three [supporting host tests](../../../../parity-results/qwen-flash-next/native-small-allgather80-host-test-r4/result.json)
  pass, including exhaustive policy equivalence with candidate79 and complete
  clique publication/exclusion cases.
- The [source/compiler/DSO-bound native proof](../../../../parity-results/qwen-flash-next/native-small-allgather80-proof-build-r3/result.json)
  runs 29 scope and 17 selection cases against the actual native implementations.
  The original enqueue negative fails exactly the nine intended selection/error
  assertions, while excluded cases and native postpasses remain equal. Explicit
  policy and proof-packaging registrations belong to production preflight.
- [Complete Unit](../../../../parity-results/qwen-flash-next/native-small-allgather80-unit/result.json)
  passes **759/759**; all [17 focused registrations](../../../../parity-results/qwen-flash-next/native-small-allgather80-focused/result.json)
  pass, including storage, capture, generation retirement and twenty-clique
  lifecycle loops. Driver windows and native retirement are clean.
- The [real captured boundary comparison](../../../../parity-results/qwen-flash-next/native-small-allgather80-boundary-analysis.json)
  retains all 56 templates per arm and every exact byte oracle. All 24 small
  native captures match candidate79's complete kernel/resource/geometry records.
  Larger-message SIMPLE and mapped-channel captures remain unchanged. Native
  live bank stride and mapped storage stride are distinct; both communicate
  exactly the live payload extent.
- [Frozen full-logit accuracy](../../../../parity-results/qwen-flash-next/native-small-allgather80-release-accuracy/report.json)
  passes every required group. All **185 rows in 26 files are byte-identical to
  L74**, including reset and diagnostic rows. This is the original accuracy
  workload and scoring policy, with unchanged arithmetic.

The source-bound host proof tests the actual selection and publication helper;
it does not claim to initialize a real communicator. The separate native
captured comparison above proves actual publication and LL execution on this
host. A portable permanent captured-policy assertion would additionally need
an independent transport-eligibility observation; the existing public storage
receipt does not expose the complete guard. Do not infer eligibility from the
flag or LL result being tested, or require LL on every generic ROCm4 cell.

The [matched Release component timings](../../../../parity-results/qwen-flash-next/native-small-allgather80-timing-analysis.json)
retain all 31 pairs per process. At 97 transactions, native gather/fold medians
are **42.169 / 41.961 / 44.086 µs** for control/candidate/control; unchanged
mapped medians are **41.270 / 42.456 / 42.722 µs**. These ranges reinforce the
need for the full model comparison instead of attributing all component
movement to the policy. The final uninstrumented model comparison below decides
acceptance; complete production preflight remains pending.

The [complete native-policy A/B/B/A](../../../../parity-results/qwen-flash-next/native-small-allgather80-model-abba-comparison.json)
passes all four original benchmark, archive/source, resource, driver and normal
retirement gates. All twelve measured 256-token sequences match. Both arms use
real archived Release binaries with their actual selected RCCL dependency and
the same isolated repaired HIP runtime; neither loads a tuning plugin.

| Native policy | Prefill median [min, max] tok/s | Decode median [min, max] tok/s |
|---|---:|---:|
| Original automatic, six samples | 785.911 [785.104, 786.302] | 33.939 [33.886, 34.115] |
| Small-AllGather LL, six samples | 785.303 [783.835, 785.682] | 34.700 [34.663, 34.746] |

The matched change is **−0.077% prefill / +2.240% decode**, with separated
decode ranges. Together with the frozen byte-identical accuracy slice and
focused regressions, this qualifies **L80 as the new accepted A1 tuning
checkpoint**. Castagna's 42.959 decode tok/s remains the target.

The workload is unchanged: four MI50s, `TensorParallel` dense computation plus
`GateUpOwnedDownColumns` routed experts, all experts on GPU, RAM ngrams,
512 prompt tokens / 256 output tokens / M512 / context4096, FP32 activation
storage and FP16 KV, MTP and prefix cache off, greedy seed42, 28 physical NUMA1
workers and ordinary MPI bootstrap. Each process runs one warmup and three
measured requests; decode throughput counts the 255 steps after prefill. All
six established precision selectors remain unchanged (HC-up TwoTermQ8,
FP16 router and MoE alpha/beta, other selected projections SingleQ8).
PerfStats is disabled for this throughput comparison. No full-preflight or
other-topology qualification is inferred from these focused results.

Earlier supporting test builds exposed incomplete header/ABI linkage and
CMake option grouping; their failed logs are retained. The canonical native
proof also exposed lost RDC mode and loss of the C++ compiler driver name
through symlink resolution. The corrected proof preserves and authenticates
the original RDC flags and invokes CMake's declared C++ entrypoint. Stale and
interrupted proof publication now has explicit device-free regressions. None
of these fixture corrections changes the native policy or model arithmetic.
