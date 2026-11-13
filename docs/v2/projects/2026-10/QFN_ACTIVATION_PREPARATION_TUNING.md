# QFN activation preparation tuning

## Existing ROCm V2 versus V4 quantizers

The bounded screen finds a small component win for V2, **18.10995 µs versus
18.42740 µs** for V4: **1.72% less complete-projection latency**. V2 wins all
12 alternating pairs; median paired saving is 0.3426 µs. No default changes
or model-throughput claims follow from this synthetic cell. Checkpoint179
remains the qualified model result at 819.648 prefill / 40.656 decode tok/s.

This compares the public ordinary M1 projection at N320/K10240, native codebook
19, with the unchanged 16-part producer and ordered reducer. Both captures use
the same prepared weights, tensor/workspace owners, pointers, scalar arguments
and coherence topology. Only the existing explicit activation-quantizer
selector differs. The seeded Q8_0 weights have the model's shape, but are not
the GGUF's weight values. Neither graph enters a verifier work scope.

Release on MI50 gfx906, CPUs28–55: 32 complete projections per graph, five
paired warmups and twelve alternating-order samples. Event intervals include
quantization, projection and reduction. Setup and checks stay outside timing.
Scratch/output bytes, complete partials, Q8 values, scale words, immutable input
and guards match before and after timing under two dirty patterns. This format
does not consume activation sums; its sum bank remains poisoned. This is not
coverage for asymmetric formats, tails or grouped rows.

## Separate profiler and compiler evidence

Six isolated rocprof processes select one exact quantizer occurrence and
counter each. The actual trace authenticates 42 complete graph submissions,
21 per variant, 673 quantizer calls per variant including its eager primer,
and the unchanged projection/reducer inventory. Occurrence226 is the first
timed quantizer after the primer, dirty checks and warmups. Profiled durations
are excluded from canonical timing.

| Observation | V4 | V2 |
|---|---:|---:|
| Workgroups × threads | 320 × 32 | 40 × 256 |
| Wavefronts | 320 | 160 |
| VALU instructions per wave | 93.8875 | 152.3125 |
| Estimated total vector instruction issues | 30,044 | 24,370 |
| Fetched KiB | 40.0625 | 40.0625 |
| VGPRs used / allocated | 8 / 8 | 14 / 16 |
| SGPRs used | 36 | 28 |
| LDS, private bytes, register spills | 0 | 0 |

V2 fills wave64 lanes with independent 32-value groups. It executes more
instructions per wave but approximately **18.9% fewer total vector instruction
issues**, with unchanged observed fetch traffic. This supports a packing
benefit; it does not establish achieved occupancy, bandwidth efficiency or an
additive model-speed prediction. The resource report inspects the exact
profiler-observed code-object slice from the loaded core, without hashing the
whole library or model.

The aggregate is
`parity-results/qwen-flash-next/quant-v2-v4-screen-summary-r1/result.json`.
It indexes functional, event, profile and ISA evidence. A missing diagnostic
graph accessor and missing post-check marker were repaired in the private
fixture; the initial failed build/collector receipts remain preserved.
The V2/V4 screen changes no production kernel or selector. Subsequent private
GDN trial entrypoints below do not change production dispatch.

## GDN normalization and SingleQ8 fusion

The complete captured normalization/projection boundary improves from
**15.325 to 13.2925 µs (13.26%)** with the aligned128 integer specialization.
Every normalized FP32 value, Q8 byte, scale word, projection partial and output
matches the original two-kernel preparation before and after timing. This is
still a private synthetic M1/N2560/K1536 native19 cell, not a new model score
or production default. It retains the original 16-part projection/reducer.

| Candidate | Separate norm + quantization + projection, µs | Fused preparation + projection, µs | Reduction |
|---|---:|---:|---:|
| Runtime integer geometry, first build | 15.3175 | 13.8025 | 9.89% |
| Runtime integer geometry, final build | 15.3100 | 13.8000 | 9.86% |
| Aligned128 integer geometry, final build | 15.3250 | 13.2925 | 13.26% |

Each row is its own unprofiled paired run: 32 complete operations per graph,
five paired warmups and twelve alternating-order samples, on MI50 gfx906 with
CPUs28–55. All twelve pairs favor fusion in each run. The final specialized
transaction interval, including the fixture's identical 64 publication events,
is 24.12255 versus 22.085 µs. These additional synthetic publication events are
reported separately from the native normalization/projection boundary; neither
interval predicts whole-model throughput by addition.

The fixture retains the actual original graph and prepared weight/workspace
owners. A cold clone replaces normalization with the fused function and removes
only its canonical activation quantizer. The existing producer/reducer arguments,
events and dependency edges remain authenticated. Every replay restores its
32 distinct in-place inputs before timing; it never repeatedly normalizes the
previous output. Dirty checks cover complete banks, guards and immutable gate/
gamma inputs. Sums are unused and remain poisoned in this cell. The fused kernel
retains the normalized FP32 store, original gate/RMS reduction order and
half-rounded SingleQ8 scales.

The specialization changes integer indexing for norm/gamma width128 and aligned
head strides. It retains the original floating divisor, both barriers and the
same floating reduction tree. The compiled bodies and resources of all
24 pre-existing functions, including the first fused variant, remain unchanged.
This establishes an integer-geometry optimization, not a precision trade-off.

## GDN profiler evidence

Nine singleton profiler runs cover the original norm, quantizer and first fused
kernel. Three further singleton runs cover the specialized fused kernel. Each
process selects one exact dispatch and one counter; profiler durations are not
used for the timing table. The actual loaded bounded code-object bytes match the
compiled resource report. Existing norm/quantizer counters below are from the
first build, whose original function bodies remain byte-identical in the final
build; they are not relabeled as freshly profiled final-build counters.

| Observation | Original norm | Original quantizer | Runtime fused | Aligned128 fused |
|---|---:|---:|---:|---:|
| Wavefronts | 24 | 48 | 24 | 24 |
| VALU instructions per wave | 139 | 94.75 | 236.5 | 201 |
| Estimated vector instruction issues | 3,336 | 4,548 | 5,676 | 4,824 |
| Fetched KiB | 12.625 | 6.0625 | 12.625 | 12.625 |
| VGPRs used / allocated | 17 / 20 | 8 / 8 | 24 / 24 | 16 / 16 |
| SGPRs used | 29 | 36 | 64 | 26 |
| LDS bytes | 16 | 0 | 16 | 16 |
| Private bytes / register spills | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The final fused candidate removes the quantizer launch and its 6.0625 KiB read,
with **38.8% fewer estimated vector instruction issues** than the two original
kernels combined. This is dispatch-level attribution, not measured achieved
occupancy or an additive model latency claim.

The first profile collector rejected HIP runtime copy/fill kernels used outside
the measured graphs for input restoration and poison checks. Its failed receipt
is retained. The corrected offline validator authenticates their successful HIP
API correlations, exact source-owned extents and completion outside every native
graph interval. Five negative controls reject an in-graph builtin, unknown
kernel, wrong reset extent, overlap or missing completion. No native run was
silently replaced or production path altered to satisfy this collector.

The aggregate evidence index is
`parity-results/qwen-flash-next/gdn-single-q8-screen-summary-r1/result.json`.
It retains all paired timing, functional, profiler and bounded ISA/resource
reports, including the original failed collector and separate adjudication.

## Expanded rows and recorded-value correctness

The explicit **122-case matrix passes 244 captures and 976 retained replays**.
It covers M1–16,31,128,512,1024,2048; head counts1,3,6,12,24,48; all three
gate activations; optional row-major sums; full, partial and empty prefixes;
and request holes with both one-row and four-row request capacities. Directed
inputs include scale1 rounding ties, distinct adjacent32-value maxima, signed
zeros, subnormals and finite inputs producing nonfinite intermediates. All
physical normalized FP32 words and guards match. Quantized inactive banks stay
poisoned, and a separate host sum of the returned signed codes checks each
active row-major sum.

The control invokes the actual canonical quantizer: V4 for M1, V2 for no-sum
small rows, V4-with-sums for small rows with sums, and register-block preparation
above M16. Captured function identities, parameters, row descriptors and launch
geometry are checked. The eight new row-policy specializations use16VGPRs,
16B LDS and no private storage or spills. These are correctness/resource results;
they are not measured grouped/prefill performance or all-format consumer coverage.

The first helper extraction changed the two older private trial kernels'
compiled instructions. Its failed comparison is retained in
`/tmp/qfn-gdn-single-q8-isa-r3/result.json`. Restoring the frozen trial bodies
resolves this: the r4 comparison proves all25 prior functions and their resources
unchanged. The new row helper remains a separate private experiment. No
production kernel was changed by this correction.

A separate **12-case recorded-value check passes96 replays**: layers0,24,46,
prefill-row0 and decode-step0, each with and without sums. The original current
normalization first reproduces every saved FP32 output word before the fused
comparison runs. Inputs are12 complete128-value heads sliced from historical
single-device ROCm snapshots, with authenticated original F32 gamma vectors.
They are model-derived value witnesses, not actual TP4 activations. The original
historical model diagnostic failed and remains labeled as such; successful
boundary ingestion does not turn it into a model-accuracy certificate.

Final local receipts are
`gdn-single-q8-rows-functional-r2/result.json` and
`gdn-single-q8-recorded-functional-r1/result.json` under
`parity-results/qwen-flash-next/`. Both verify the loaded runtime, unchanged
sources/assets, clean driver window and native retirement. The exact final
prototype source and Release runtime are archived in
`/tmp/qfn-gdn-single-q8-component-green-r1/`; its private source edits have been
removed from the shared production tree while the independent scalar MoE
column-publication change is qualified.

## Promotion boundary and next work

Keep V2 available through its existing explicit diagnostic selector. A
production policy still needs representative geometry/format coverage and a
matched model comparison; this one shape does not justify overriding V4 for
all single-row projections.

Original row ownership and optional sums now have the bounded correctness proof
above. Normalization still stores every physical row; only Q8 preparation follows
the original projection row authority. This does not yet qualify a production
composite invocation or its complete projection consumers across all formats.

For eventual integration, prefer a typed composite invocation owned by the
existing prepared GEMM engine and its admitted workspace, following the HC
input-transform precedent. It needs no new allocation ledger or public
cross-stage prepared-activation lifecycle. Backend/all-format/row tests,
snapshot/capture equivalence and matched model accuracy/performance are still
required. Floating weights must remain floating; other activation policies
cannot silently consume this SingleQ8 program. Checkpoint179 remains the latest
qualified model result, and G60 remains the latest complete production preflight.
