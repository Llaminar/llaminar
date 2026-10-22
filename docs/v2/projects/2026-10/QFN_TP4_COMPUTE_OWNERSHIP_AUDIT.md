# QFN TP4 computation ownership audit

Initial audit reviewed 2026-10-22 against source `77e17645b`, the retained L69
per-device trace, and NativeVNNI82's construction observations. A1 uses one MPI
rank with four MI50 participants, dense `TensorParallel`, routed
`GateUpOwnedDownColumns`, RAM ngrams, 48 main layers and MTP disabled. Its
initial accepted unprofiled baseline was **L80 785.303 prefill / 34.700 decode
tok/s**. The measured follow-up below advances the explicit vocabulary-sharded
configuration to **L84 786.235 / 35.056 tok/s**, with unchanged accuracy.

The transformer is not replicated wholesale. Several explicit replicas do
repeat substantial work, and two narrower opportunities avoid computation
whose result the current consumer does not need.

## Actual ownership

| Component | Local work on each TP4 participant | Assessment |
|---|---|---|
| GDN, 36 layers | QKV 2560×2560, gate 1536×2560, output 2560×1536, four key heads and twelve value heads | Distinct head/state slices. Full output N does not imply a full input K. |
| Sparse/full attention, 12 layers | Six of 24 Q heads; paired Q/gate 3072×2560; output 2560×1536 | Q, attention and output work are sharded. |
| K/V projections and cache | Both global KV heads on every GPU; each projection 512×2560 | Explicit replication includes one head unused by each local six-query-head interval. |
| Shared FFN, 48 layers | Gate/up 160×2560; down 2560×160 | Distinct intermediate slices, with the original ordered reduction. |
| Routed experts, 48 layers | Gate/up assigned by expert ID; disjoint 640-column down slices | Each selected gate/up runs once, and down owners compute distinct output columns. Importing selected intermediates is necessary for those consumers. |
| Hyperconnections | 97 full branch reads and 96 injections | Explicit replicated computation and branch state. |
| PLE/ngram frontend | Full key/value projections and per-participant lookup responses | Explicit replicated compute; four response channels do not mean four physical RAM tables. |
| Sparse indexer | Full BF16 query/key projections, pool history and selection | Replicated indexer; its subsequent attention consumes distinct query heads. |
| Terminal vocabulary head, mirrored control | 248320×2560 on each GPU | Full projection repeated four times. Arithmetic-policy N62080 does not shard physical work. The qualified explicit sharded follow-up below uses physical N62080. |
| Router/norms/activation quantization | Router and some full-row operations repeat; head-local norms are sharded | Inspect actual inputs and consumers. Repeated kernel names or replicated norm weights alone do not establish duplicate work. |

The declarations and bindings are in
[Qwen4ExpSchema.h](../../../../src/v2/models/qwen4exp/Qwen4ExpSchema.h),
[Qwen35MoESchema.h](../../../../src/v2/models/qwen35moe/Qwen35MoESchema.h),
[QwenGraphBase.cpp](../../../../src/v2/models/qwen/QwenGraphBase.cpp) and
[MoEProjectionPipeline.cpp](../../../../src/v2/execution/compute_stages/stages/MoEProjectionPipeline.cpp).
Original source dimensions/formats are retained in the
[bounded descriptor inventory](../../../../parity-results/qwen-flash-next/nativevnni81-source-descriptors.json),
which reads metadata without scanning or hashing tensor payloads.

The [L69 trace](../../../../parity-results/qwen-flash-next/a1-control69-rocm-profile/result.json)
authenticates 255 measured model-forward graphs on each of four native agents.
All four execute 255 full-vocabulary producers and reducers. Current
[construction observations](../../../../parity-results/qwen-flash-next/nativevnni82-release-accuracy/native-q8-physical-route-observation.json)
independently retain physical N248320, policy N62080 and K2560. Construction
counters alone are not per-device execution proof; the retained per-agent trace
establishes membership. Padded HC-down and K/V launches can share grid geometry,
so source-bound dimensions must not be described as directly decoded trace
arguments where that join is absent.

## Measured cost and limits

The following are additive GPU dispatch durations from instrumented **L69**,
per model-forward, with ranges across the four GPUs. They are neither current
unprofiled L80 times nor independently removable critical-path costs.

| Subset | Calls per GPU/token | Profiled milliseconds per GPU/token |
|---|---:|---:|
| Full head producer and reducer | 2 | 1.093–1.209 |
| HC-up TwoTerm projection | 97 | 1.630–1.710 |
| HC-up activation preparation | 97 | 0.587–0.606 |
| HC injection | 96 | 1.864–1.933 |
| Indexer query projection | 12 | 0.629–0.642 |
| Indexer query preparation | 12 | About 0.10 |
| Router logits | 48 | 0.420–0.449 |

HC injection can overlap block computation; shared-expert reduction can
overlap routed down computation. L69 also predates the accepted short-K and
RCCL improvements. Summing these durations or multiplying savings by four
would overstate an end-to-end gain. See the
[collective attribution](QFN_TP4_COLLECTIVE_ATTRIBUTION.md) for overlap and
profiling overhead. Detailed source estimates and extracted per-agent subsets
are retained locally in `/tmp/qfn-tp4-duplicated-work-audit-next/`.

One full head represents 1.271 billion dot multiply/add operations and a
675,430,400-byte original Q8_0 source footprint per GPU. These are useful work
and source-size estimates, not measured HBM traffic or admission accounting.
Ideal column sharding removes three globally repeated copies of that head
arithmetic; it cannot remove 75% of model latency. A roughly one-millisecond
head is only part of the accepted roughly 28.8-millisecond decode step.

## First candidate: ordinary vocabulary sharding

The explicit `vocabulary-sharded` terminal policy already selects local head
weights and an explicit native logits gather. Auto currently mirrors QFN;
the measured automatic sharding exception is a different dense TP2/MTP family.
An explicit comparison must not change Auto before it is qualified.

The original audit found a concrete ordinary-mode gap: native local vocabulary
publication in [GraphTypes.h](../../../../src/v2/models/GraphTypes.h) and its
terminal publication fact in
[MemoryPlanner.h](../../../../src/v2/planning/MemoryPlanner.h) depended on retained
MTP capacity. With MTP disabled and zero capacity, the complete ordinary
sampler correctly rejects incomplete sharded logits. Removing that rejection
would hide the defect.

The existing one-row native stage gathers directly from `LOGITS_LOCAL` into
full `LOGITS`; it needs no rank-major transpose scratch or MTP sidecar. The
repair must declare this ordinary producer and its full-output BOM/prefix
ownership independently of MTP capacity. Preserve the existing local native
communicator, explicit stream, terminal owner and complete captured-generation
checks. Reserving otherwise unnecessary MTP capacity is not the solution.

For this vocabulary, each GPU projects 62080 columns and contributes exactly
248,320 bytes of FP32 logits to a 993,280-byte completed row. That send exceeds
the accepted small-AllGather threshold. Benchmark the complete projection,
gather and sampling transaction, followed by the unchanged 512-prompt/256-output
Release workload. Require exact output bytes, live communication extents,
zero-capacity admission, reset/prefix lifecycle and the canonical 185-row model
accuracy slice.

### Experiment84: measured ordinary M1 head sharding

The two publication predicates now admit an explicit native local sharded
head independently of retained MTP capacity. The focused old-source regression
fails in all three ownership/admission cases; the repaired source passes
**11/11 device-free** and **9/9 native** registrations, including CUDA TP2,
ROCm TP2 and the actual 248320-column ROCm TP4 vocabulary. Existing counted,
serial-family and fixed-serial native cases remain green. Ordinary M1 retains
no transpose bank, predictor or verifier count. These are focused results,
not a complete production-preflight certificate.

Both explicit policies pass the unchanged canonical QFN accuracy panel.
The [sharded byte comparison](../../../../parity-results/qwen-flash-next/vocabulary84-release-accuracy-sharded/logit-byte-comparison.json)
retains **all 26 files / 185 final-logit rows byte-identically** to both L74
and the newly measured mirrored control. This includes prefill, decode and
reset rows, each with one projected terminal row. Construction receipts show
physical N62080 instead of N248320, with the same arithmetic-policy N62080,
K2560 and native-Q8 kernel geometry. Those receipts are not replay counters.

The [complete A/B/B/A comparison](../../../../parity-results/qwen-flash-next/vocabulary84-model-abba-comparison.json)
uses one archived Release binary, HIP76 and RCCL80. Each arm has one warmup
and three measured 512-prompt/256-output iterations; all twelve generated
sequences match. Dense/routed placement, 28 physical NUMA1 workers, RAM ngrams,
C4096/M512, precision, seed, disabled prefix cache and MTP-off/capacity0 are
identical. Only the explicit terminal-head policy changes. PerfStats timing
instrumentation is disabled, and all four arms retire normally with clean
driver windows and unchanged runtime/source identities.

| Explicit terminal policy | Prefill median tok/s | Decode median tok/s | Decode sample range tok/s |
|---|---:|---:|---:|
| Mirrored full vocabulary | 785.752 | 34.627 | 34.514–34.656 |
| Vocabulary sharded | 786.235 | 35.056 | 34.970–35.158 |

The combined six-sample medians give **+0.061% prefill / +1.237% decode**.
This removes roughly 0.35 milliseconds per decoded token in this workload;
it does not close the remaining gap to Castagna's 42.959 tok/s. Five-second
passive resource samples also show lower per-card VRAM peaks, by approximately
0.5–0.65 GiB in the first three arms. These are sampled whole-process/device
observations, not a measurement of the projection's individual allocation.

### Follow-up85: ordinary projected-row admission and traffic regressions

`OrdinaryTerminalProjectionCapacity` now carries setup-time output demand
through the existing rank, participant and graph configuration. Selected-final
execution retains one row per request; an explicit larger declaration is
bounded by the existing flattened activation capacity. Admission prices the
same `VocabularyGatherWorkspaceContract` used by the native stage. Its exact
bank is published before capture, and later output selection cannot enlarge
it. Retained MTP rows join that capacity only when an MTP family actually
exists; inactive placeholders do not authorize storage or execution.

The [19 device-free registrations](../../../../parity-results/qwen-flash-next/ordinary-vocabulary84-capacity-devicefree/result.json)
pass declaration/overflow, pipeline propagation, implicit MoE recompilation,
ownership and workspace checks. Planning coverage includes two, four and
eight participants. The [12 native registrations](../../../../parity-results/qwen-flash-next/ordinary-vocabulary84-capacity-native-r2/result.json)
pass on CUDA TP2 and ROCm TP2/TP4, including 20 poisoned large/partial/empty/small
replays, exact admitted ordinary banks, stable workspace identity and the
existing MTP serial/counting families. The rebuilt
[complete Unit suite](../../../../parity-results/qwen-flash-next/ordinary-vocabulary84-unit/result.json)
passes **763/763**, with no failures or skips, in **116.43 seconds**.

The device-free ABI tests call the real NCCL/RCCL loader wrappers against an
explicit recording dependency. Fixed M1 forwards exactly **62080 FP32 elements
/ 248320 bytes**, preserving its original stream and buffer identities. It
rejects a passive counter rather than fabricating a traffic receipt. A counted
positive control proves that the distinct extension is actually observed.
Native captured fixtures independently prove output and untouched tails.

The first TP4 counted-fixture run exposed a test expectation inherited from
two participants. Native receipts count primitive sends, including necessary
ring forwarding: aggregate payload is `P * (P - 1) * live_rows * local_columns
* sizeof(float)`. The old assertion omitted `P - 1`; outputs and tails were
already correct. The corrected assertion passes without a runtime change or
tolerance. Original failed build/test receipts remain retained.

The [final-source Release build](../../../../parity-results/qwen-flash-next/vocabulary85-capacity-release-build/result.json)
passes in **215.09 seconds**. Its [canonical M1 panel](../../../../parity-results/qwen-flash-next/vocabulary85-capacity-release-accuracy-sharded/runtime-observation.json)
passes in **100.64 seconds**, with all required numerical groups, reset,
driver, source/runtime identity and normal retirement checks green. All
**26 files / 185 rows** remain [byte-identical](../../../../parity-results/qwen-flash-next/vocabulary85-capacity-release-accuracy-sharded/logit-byte-comparison.json)
to both L74 and accepted84 sharded. The archived source is parent
`27b6006f5ec630ac46b74a98e5555019dc654ddb` plus scoped source delta
`8e258dd5e55238d4df610c6212d4e70332f8699d9450ef28e151588e1e0bc792`.
This follow-up validates the completed capacity implementation; throughput
numbers remain the actual four-arm Release84 measurements above.

All discovered contracts have explicit `ProductionTestPreflight` registrations.
The model evidence establishes M1 correctness within the retained panel:
native multi-row fixture coverage does not claim complete all-position model
qualification or qualify the existing dynamic ordinary-logits allocation
owner. Auto policy is unchanged. Complete production preflight remains a
separate qualification.

## Consumer-minimal KV ownership

The current Q intervals are `[0,6)`, `[6,12)`, `[12,18)` and `[18,24)`.
QFN's sparse kernel maps global Q heads to KV heads with a ratio of twelve,
so those participants consume KV0, KV0, KV1 and KV1 respectively. Computing,
appending and storing the other head provides no local consumer benefit.

At context4096 with FP16 KV, removing that unused head would save per GPU:

- 31.457 million K/V dot operations across twelve layers per decoded token.
- A 16,711,680-byte original Q8_0 projection-source footprint.
- 12,288 bytes of KV append per token and **48 MiB** of KV capacity.

Attention already reads the needed head; do not also count half of attention
computation as removable. No new communication is mathematically required.
This ownership is not currently selectable: the assignment authority accepts
disjoint ranges or identical replicas, and some cache constructors reconstruct
the start from participant index rather than the original typed interval.

For arbitrary Q intervals, derive the covering contiguous KV range from the
global Q/KV ratio. Unequal intervals may straddle a KV group and need both
heads. Update the shared ownership/union validation and carry the original
start/count through slicing, cache/BOM geometry, backend consumers and
snapshots. Do not hardcode TP4 pairs or assume generic attention consumers
already support partial replication. Test equal, straddling, empty, reordered
and asymmetric TP4/TP8 assignments, original source bytes, cache append/reset,
supported codecs and actual per-device work/storage. Implementation remains
under source review. QFN's sparse CPU/CUDA/ROCm consumers already subtract the
global KV origin; the generic dense attention mapping currently has no such
field. Changing its assignments without extending that mapping would be
incorrect for intervals crossing a KV group. Keep the existing generic policy
until its complete consumer contract is implemented and tested.

## Short-context indexer query work

The current benchmark sees 512–767 visible tokens. Its sparse selector returns
every visible token and does not consume query scores, but the BF16 query
projection and query preparation still execute on all GPUs. The all-token
condition extends through **2051** visible tokens; scoring begins at **2052**,
including the four-token pooling tail convention.

A device-owned guard could avoid unused query computation without a new
collective. Raw-key projection, pool preparation and history commit must
continue for later context growth. Any candidate must preserve finite/error
contracts, capture identity, acceptance/reset and crossings at
2047/2048/2049/2051/2052/2053, including long-to-short replay. The old profile's
roughly 0.73–0.75 milliseconds per GPU is a hypothesis to measure, not an
accepted saving.

HC/router/PLE sharing would require earlier explicit publication to current
consumers. Their existing downstream collectives occur too late to carry those
inputs. There is no demonstrated free sideband optimization here, and spare
allocation capacity never authorizes extra communication. These larger graph
changes remain separate from the bounded candidates above.
