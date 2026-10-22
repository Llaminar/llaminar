# QFN TP4 collective attribution and selector repair

Updated: 2026-10-22. Accepted inference source: `545192c8` (L69); documentation
checkpoint `24ac6ab0`. This investigation does not change the dashboard's
accepted **785.182 prefill / 33.238 decode tok/s** observation.

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
