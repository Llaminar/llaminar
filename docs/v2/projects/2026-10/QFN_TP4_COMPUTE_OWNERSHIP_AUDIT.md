# QFN TP4 computation ownership audit

Reviewed 2026-10-22 against source `77e17645b`, the retained L69 per-device
trace, and NativeVNNI82's current construction observations. A1 uses one MPI
rank with four MI50 participants, dense `TensorParallel`, routed
`GateUpOwnedDownColumns`, RAM ngrams, 48 main layers and MTP disabled. Its
accepted unprofiled result remains **L80 785.303 prefill / 34.700 decode tok/s**.
This audit changes neither placement nor accuracy policy.

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
| Terminal vocabulary head | 248320×2560 on each GPU | Full projection repeated four times. Arithmetic-policy N62080 does not shard physical work. |
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

The audit found a concrete ordinary-mode gap: native local vocabulary
publication in [GraphTypes.h](../../../../src/v2/models/GraphTypes.h) and its
terminal publication fact in
[MemoryPlanner.h](../../../../src/v2/planning/MemoryPlanner.h) depend on retained
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
accuracy slice. No QFN TP4 sharded-head performance claim is established yet.

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
