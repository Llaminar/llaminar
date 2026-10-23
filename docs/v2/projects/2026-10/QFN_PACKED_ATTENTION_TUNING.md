# QFN packed selected-attention tuning

Current disposition: the original packed107 experiment was rejected and
removed. The later shared-tile FP16 program123 passes the accepted model
accuracy gate; candidate124 preserves its bytes and improves matched model
decode by0.925%. It remains explicit while the corrected post-RoPE M1/64K
component regression is tuned. The source-derived QK unroll16 winner128
reduces that regression to1.71% and preserves component bytes. Its matched
A/B/A reaches818.838 prefill /39.987 decode tok/s versus816.165 /39.283 for
FP32, with185 exact model-logit rows and825 passing Unit/device registrations.
Decode optimization is the priority; prefill remains a regression guard, with
separate physical schedules to be selected only after their own qualification.
See the final sections
and the [actual fork source comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md).
The bounded selector, tie correction and observation economies remain retained.
The sections below record chronology; complete preflight still needs a refresh
beyond G60.

## Starting checkpoint and acceptance

L106 is committed as `68450fa3e` on `feat/qwen-flash-next`. Its matched four-MI50
A1 measurement is 815.992 prefill / 39.274 decode tok/s, versus HC103 at
815.330 / 37.520. The historical Castagna decode target is 42.959 tok/s.
See the [dashboard](QFN_TUNING_DASHBOARD.md) and
[L106 qualification](QFN_FLOATING_REDUCTION_TUNING.md#l106-overlap-loads-while-preserving-each-original-dot-product).

Attention107 is an explicit experimental component. It is not yet selected by
the model graph or installed as a default. Native component correctness,
performance and canonical model accuracy must pass before promotion. The
existing FP32 arithmetic and every native codec remain unchanged.

This candidate changes precision. Its acceptance uses the existing independent
QFN model-reference gates, including per-row diagnostics and mean final-logit
KL/top-1/top-5 results. Old FP32 logit byte equality is not a suitable precision
gate. Within the new arithmetic program, serial/grouped rows and full/sharded
head intervals must still produce identical bytes. Scheduling optimizations
must also preserve the initial packed program's bytes directly.

## Arithmetic and physical execution

The core attention has D256, 24 original Q heads and two original KV heads.
The first prototype targets native FP16 K/V published after RoPE on gfx906.
It does not convert a whole cache or reinterpret another cache format as FP16.

The new program uses 33 fixed partitions of 64 selected-token ranks. Every
partition uses scaled-half Q, native half K and `v_dot2_f32_f16` with FP32
accumulation, a fixed width-32 QK tree, and a fixed 64-leaf softmax tree.
Probabilities are rounded to half after scaling by exactly 1/128. PV uses one
ascending packed-half FMA chain; 33 partials merge in ascending FP32 order.
The probability scale bounds the unnormalized half accumulator even when V is
near the largest finite half value. Its underflow/accuracy cost still needs
native and model measurement.

Canonical pairs of original query heads share one K/V tile. A participant's
head interval intersects those pairs; odd tails retain their original KV
owner and arithmetic. The initial physical layout is G2/T32/256 threads.
Decode and prefill may later select different physical schedules from their
immutable geometry, while retaining the same arithmetic program.

The original eight-part FP32 scratch bank is unchanged. The new typed packed
record is 528 bytes; the partial bank is `M * local_heads * 33 * 528` bytes.
At Q6/M512 this is 53,526,528 bytes, versus the old 25,559,040 bytes. This is one
shared sequential scratch bank, not a per-layer allocation. The checked layout
contributes a BOM to the existing arena/PMA authority; it is not another ledger.
Logical per-head consumption and actual group-owned native loads have distinct
observation types.

## Initial build and static evidence

The first compile exposed a HIP half-header include-order error; its corrected
Release build passes the mandatory spill guard. The actual code object targets
gfx906 and contains native `v_dot2_f32_f16` and `v_pk_fma_f16` instructions.

| Initial component | Compiler-used VGPR | Compiler-used SGPR | LDS bytes | Private bytes | Register spills |
|---|---:|---:|---:|---:|---:|
| G2/T32 producer | 19 | 92 | 18,064 | 0 | 0 |
| Ordered 33-part merge | 73 | 49 | 276 | 0 | 0 |

These are static compiler resources, not achieved occupancy or runtime speed.
Evidence: `parity-results/qwen-flash-next/attention107-component-release-build-r2`
and `attention107-g2-isa-r1`.

The five device-free component tests pass through explicit preflight entry
`V2_Integration_QwenSparseAttentionPlanAdmission`, including the follow-up
row-parameter alignment regression. They cover exact capacity/overflow,
aliasing, head groups, codec admission and immutable recording identity.
The complete Unit suite passes 776/776 in
`attention107-component-unit-r1`. These tests do not certify native attention.

## Focused experiment sequence

1. Record the first packed producer/merge baseline with independent flat-softmax
   half-range witnesses, serial versus M2..16 rows, compact head shards for
   degrees 1..8, ring mapping and dirty retained replays. Export deterministic
   partial/output bytes for later same-arithmetic comparisons.
2. Time only attention plus merge against the current FP32 pair, excluding the
   indexer, diagnostic exports and profiler overhead. Cover Q6/M1 decode and
   larger prefill buckets, including M1024/M2048 as subsequent scaling points.
3. Test wave-local max/sum reductions with the exact original 64-leaf operand
   tree and required cross-wave phase barriers. Separately test merge-loop
   scheduling to reduce register pressure. Compare both directly with saved
   baseline bytes, then collect fresh ISA, isolated counters and event timing.
4. Only after component accuracy/economy are demonstrated, carry the typed
   program through graph policy, buffer geometry, request leases, stage/native
   bindings and complete capture identity. Publish actual grouped-load counters
   with their correct semantics. Keep graph declarations participant-local.
5. Run unchanged canonical model numerical gates and matched unprofiled A1
   Release measurements before choosing a default and checkpointing a win.

Full production preflight remains G60. No attention speedup or model numerical
result is claimed yet.

## Initial native measurements and rejected economics

The initial component passes all four explicit ROCm tests plus the admission
entry (`attention107-baseline-native-r1`, five tests). The native inventory
includes M1..16 serial/grouped comparisons, actual compact head intervals at
degrees 1..8, ring addressing, 20 dirty full/partial/empty retained replays and
an independent flat-softmax/max-half range oracle. It exports 246 named cases
and 492 output/partial banks, including deterministic empty/inactive poison.
This is component evidence; the initial prototype has no model qualification.

Timing repeats complete producer/merge pairs inside one captured graph, so one
host submission is amortized over 256 pairs for M1 or 16 pairs for M512. The
fixture authenticates the exact pair inventory and ordered native dependency
chain. Five warmups precede 12 alternating old/new pairs. Q6, native FP16 KV,
context 512, no gate, indexer or observations are shared by both sides.

| Packed scheduling candidate | M1 packed / existing FP32, us | M512 packed / existing FP32, us | Same-program bytes |
|---|---:|---:|---|
| Initial G2/T32, CTA softmax tree | 64.628 / 56.787 | 2261.780 / 665.932 | Initial saved program |
| G2/T32, wave64 softmax tree | 61.692 / 56.835 | 2154.295 / 665.933 | All 246 cases / 492 banks exact |
| G2/T16, wave64 softmax tree | 62.648 / 56.930 | 1480.190 / 665.969 | All 246 cases / 492 banks exact |

The wave tree preserves each original guarded 64-leaf reduction edge and its
operand order while removing unnecessary CTA barriers. It passes the same five
focused tests (`attention107-wave-native-r1`) and the direct saved-byte comparison
(`attention107-wave-native-byte-comparison-r1.json`). Its measured improvement
over the first prototype is about 4.5% for M1 and 4.8% for M512. Both prototypes
remain slower than the existing kernel; neither is a production candidate yet.
The event receipts are `attention107-{baseline,wave}-m{1,512}-c512-batched-r1`.
Earlier single-pair event receipts remain diagnostic evidence of host-submission
overhead and are not used as isolated decode-kernel latency.

The baseline isolated trace's final warm dispatch attributes 1929.436 us to the
producer and 333.759 us to the merge. Those instrumented times identify where
to investigate, not a subtractable model speedup. The wave object uses
18 VGPR/90 SGPR for its producer, retains 18064 bytes LDS, and reports zero
register spills/private scratch. The profiler reports descriptor register
allocations rather than compiler-used counts. Pinned LLVM deliberately pads the
former to the occupancy ceiling imposed by LDS: T32 allocates 68 VGPR/112 SGPR,
while T16 allocates 40 VGPR/112 SGPR despite using 18 VGPR/90 SGPR. The final
loaded ELF metadata and descriptor independently confirm the T16 distinction
(`attention107-wave-t16-final-code-isa-r1`). These are resource ceilings, not
measured occupancy.

The G2/T16 experiment retains all
64-token arithmetic partitions and ordered FMA chains, reduces source LDS from
18064 to 9808 bytes, and adds tile barriers. It improves M512 by 31.3% over T32,
but is about 1.5% slower at M1. Its five focused tests and all 492 byte banks
pass (`attention107-wave-t16-native-r1` and
`attention107-wave-t16-native-byte-comparison-r1.json`). The isolated M512
producer reports 98.125% VALU lane utilization, no LDS bank conflicts and zero
scratch; these counters do not establish elapsed-time utilization or model gain.
Decode and prefill may choose
different proven physical schedules; capture identity and timing markers must
name the actual selected layout. No precision or finite-input check is removed
by this experiment.

At M512/context512, the static partition grid contains 50,688 producer CTAs but
only 6,912 have selected data. A separate prefill experiment assigns all fixed
partitions of a row/head-group to one CTA, retaining every independent partial
and its arithmetic. Decode retains the existing partition-parallel option.
Both schedules must compare output and complete packed-record bytes directly;
no automatic dispatch is installed before measuring the crossover.

## Required scaling qualification through 64K

The acceptance sweep must extend through actual 65,536-token native cache
capacity, including decode and M512/M1024/M2048 prefill blocks. Fixed-M tests
vary the causal prefix independently of the bucket. Long-context membership
must include selected four-token groups spread across that prefix, rather than
reading only its first 2K tokens. Timing records bind the membership policy,
physical schedule, tile, live rows, context, actual cache capacity and exact
captured kernel inventory.

Measure selected attention and the indexer separately. The former consumes at
most 2,051 tokens per row and should level off after that limit; the latter
still scans growing context and should be evaluated against its O(M * context)
work. Report full-model prefill/decode results as a separate acceptance step,
including 64K numerical checks. A capped attention-only result cannot certify
full-context indexer or end-to-end scaling, and fixed-M scaling cannot be
substituted for cumulative whole-prompt prefill cost.

## Explicit schedules, 64K byte gates and first scaling results

`attention107-schedules-native-r1` passes seven focused registrations. The
partition-parallel and row/group-loop schedules match complete output and
partial-record bytes over every M1..16, compact head slices through degree 8,
dirty retained replays and actual 64K row-parameter rings with spread selection.
The separate `attention107-highm-64k-native-r1` passes M257/M512/M1024/M2048
at actual 64K capacity, full and M-1 live rows, including independently checked
inactive poison. The existing 246 exported cases remain byte-identical.

The first row/group-loop is slower: M512/context512 takes 1956.650 us versus
1483.680 us for the same binary's partition-parallel control. The compiler
reports 164 scalar spill slots lowered to register-lane moves, with zero
private scratch and zero VGPR spills. This is permitted register-only storage,
but not free work. The loop is an explicit experiment, not an installed policy.

A one-helper trial makes each fatal predicate wave-uniform without removing
checks. All eight focused registrations pass through M2048/64K in
`attention107-uniform-fatal-native-r1`, and all 492 saved banks remain exact.
It only reduces M512/context512 to 1923.525 us for the loop and 1459.885 us for
partition-parallel. The loop still emits 178 scalar register-lane spill slots;
this trial does not solve that pressure. Malformed-input process tests remain
a separate qualification item before adopting the changed fatal edge.

The complete Unit run `attention107-schedules-unit-r1` passed 775/776 entries.
Its sole failure was the closed expected target inventory missing the new
opt-in ROCm attention benchmark. The inventory was corrected and that entry
passes in `attention107-unit-target-selection-r1`; the original failure is
retained. Existing explicit preflight entries cover this inventory regression.
This is the complete run plus its scoped repair, not a new full preflight.

All 31 context-sweep cells pass in
`attention107-uniform-parallel-64k-sweep-r1`, using the frozen Release runtime,
full native cache capacities and authored selections spread across each causal
prefix. Per-cell receipts authenticate capture geometry, source/runtime pins,
driver windows and ordinary retirement. Values below are producer plus merge,
not indexer or end-to-end model timings.

| M / total context | Packed parallel T16, us | Existing FP32, us |
|---|---:|---:|
| 1 / 2048 | 71.175 | 57.534 |
| 1 / 65536 | 70.924 | 57.360 |
| 512 / 4096 | 8291.110 | 4830.565 |
| 512 / 65536 | 8762.095 | 5117.705 |
| 1024 / 65536 | 17504.250 | 10191.400 |
| 2048 / 65536 | 34854.150 | 20405.450 |

Selected attention levels off once its selection cap is reached; the measured
M512 growth from 4K to 64K is about 5.7%. Absolute latency still loses to the
existing kernel. M1024/M2048 completed the same immutable sweep contract without
repeating accepted cells. At 64K, doubling M nearly doubles latency. No model
numerical or full-prefill scaling result is inferred from these component records.

The register-direct schedule uses 128 producer threads and no staged K/V tile;
each native K or V pair serves two original query heads while their arithmetic
chains remain separate. `attention107-register-native-r1` passes all nine
focused registrations, including independent flat-half witnesses, same-mode
serial/grouped bytes, compact shards and M2048/64K dirty replay. The original
246 cases / 492 exported banks remain byte-identical in
`attention107-register-native-byte-comparison-r1.json`. These are component
correctness results; timing, native resource profiling and model acceptance
remain separate requirements.

The register-direct unprofiled pair measures 79.109 us at M1/context512,
985.346 us at M512/context512, and 5334.305 us at M512/context65536. The same
binary's FP32 controls are 56.892, 665.820 and 5086.675 us respectively.
Removing the staged K/V tile substantially improves prefill over packed T16,
but still does not beat the installed kernel; no default is changed.

## DeepInfer adoption in the current tuning pass

The user requested incorporating the existing
[DeepInfer research](DEEPINFER_INFERENCE_RESEARCH.md). Its first recommendation
is DeepSelect's bounded-candidate top-k algorithm for the sparse indexer.
The branch already contains CPU/CUDA/ROCm candidates with 256/512, 512/512 and
1024/1024 block/slack geometries; serving still uses `Installed` selection.
The new work must qualify and tune those existing implementations rather than
introduce another selector.

The pinned upstream source is available locally at DeepSelect revision
`bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`. Retained focused qualification
matched the seven audited candidate source files before this pass's coverage
expansion. The complete indexer benchmark must
explicitly name the selector, authenticate all four captured kernel roles,
compare selected membership and tie order, and time each policy on matching
scores/causal geometry. Scoring precision and the downstream attention arithmetic
remain separate controls.

The current adaptation scans blocks in ascending order and preserves selected
IDs through stable compaction. Upstream's randomized block order has an expected
work guarantee that cannot be claimed for that deterministic traversal. Any
traversal improvement must retain a total score/index ordering and restore
canonical ascending output IDs. Native cluster features from newer NVIDIA
architectures do not provide an implementation for MI50 or RTX 3090.

The frozen Release archive `/tmp/qfn-indexer108-register107-runtime-r1` now has
40 passing complete-indexer event cases. Each receipt binds all four native
kernel roles, exact explicit selector geometry, actual prepared scores, all-row
numeric-sort membership before/after replay, and ordinary native retirement.
The measured scope excludes projection GEMMs, selected attention and model
inference. These timings do not independently validate indexer score arithmetic.

| Rows / context | Current selector, us | Best measured bounded geometry | Bounded, us |
|---|---:|---|---:|
| 1 / 512 | 13.217 | 1024/1024 | 13.244 |
| 1 / 4096 | 51.062 | 512/512 | 57.746 |
| 1 / 8192 | 72.836 | 1024/1024 | 72.119 |
| 1 / 16384 | 152.827 | 1024/1024 | 138.281 |
| 1 / 65536 | 600.075 | 1024/1024 | 272.937 |
| 512 / 512 | 47.065 | 512/512 | 54.020 |
| 512 / 4096 | 1051.565 | 512/512 | 1029.525 |
| 512 / 8192 | 2540.605 | 512/512 | 2198.805 |
| 512 / 16384 | 5208.740 | 512/512 | 4251.425 |
| 512 / 65536 | 22146.600 | 512/512 | 17351.150 |
| 1024 / 65536 | 38600.950 | 512/512 | 34347.200 |
| 2048 / 65536 | 71644.150 | 512/512 | 67904.900 |

Only 4K and 64K M1/M512 compare all three bounded geometries; other rows compare
the named candidate against the current selector. At 64K M1, 256/512 and 512/512
take 356.041 and 337.938 us. At 64K M512, 256/512 and 1024/1024 take 17369.850
and 17680.350 us. The best measured complete-indexer reductions are 54.5% for
M1 and 21.7% for M512. A blanket replacement would regress short contexts.

Explicit typed builder/stage forwarding and selector-bound capture identity
are implemented for qualification, with the existing default preserved. A
separate decode-only contextual candidate is being prepared: its GPU-owned live
prefix chooses the original short-row algorithm or the bounded long-row
algorithm inside one retained launch. Admission capacity is not a proxy for
live context. This candidate, expanded adversarial/backend regressions and
native profiling must pass before any production policy is promoted.

`indexer108-policy-coverage-focused-r1` passes all 15 focused registrations
across CPU/CUDA/ROCm. Added coverage includes every grouped M1..16, true 64K
stored-score selection, signed zeros, a one-ULP cutoff, full/small/empty/full
replays and actual downstream attention bytes. Typed construction and capture
identity pass independently. This is a focused slice, not full preflight G60
replacement or real-model qualification.

Review found that the two existing GPU Installed selectors compared raw zero
sign bits. They now canonicalize only signed zero before integer comparison;
negative nonzero and non-finite encodings are not repaired. Actual indexer
scores already produce positive zero. Dedicated native Installed regressions
are registered; their fresh native result is pending.

The explicit `DecodeContextual` candidate admits M1..16. It uses the original
wide selector through 2048 complete pools (8195 visible tokens including a
three-token tail), then bounded 1024/1024 from token 8196. One CTA owns each
row and reads the device-owned prefix; the same retained graph can cross the
boundary. A shared union reserves 42224 bytes and uses one CTA barrier to
publish its active member. Static LDS remains reserved on the short path, so
its economy must be measured rather than assumed. Its Release build passes for
gfx906 and CUDA targets 80/86/89/90; serving defaults remain unchanged.

The first isolated B1024 profile retired cleanly and authenticated its dispatch
and counter, but its postprocessor incorrectly rejected all SGPR spill metadata.
The selected final kernel has three scalar values stored in vector-register
lanes, zero VGPR spills and zero private storage. Direct disassembly finds three
matching lane writes and eight reads. A separate successor observer reports
these values without replacing the canonical compiler allocation certificate.
The original failed profile remains failed; no receipt is retroactively green.

`indexer108-contextual-repeat-focused-r1` passes all 25 focused registrations
across CPU/CUDA/ROCm in 50.43 seconds, with clean driver observations and normal
process retirement. The actual compute-stage test now authenticates all five
selector functions at M1/M16 and the four prefill selectors at M33. It replays
visible-token boundaries 8195/8196, mixed grouped rows, and long/partial/empty/
short/reset states with unchanged captures. The signed-zero regressions pass on
both GPUs. Serving still uses `Installed`; this is not full-model qualification.

The first stage-inspection run passed 22/23 entries. Its new CUDA check had
incorrectly required zero private storage in the two unchanged rotary producers.
The linked sm86 `preparePools` disassembly and resources match the retained
pre-contextual control exactly: 32 bytes of stack storage, zero local allocation
in the code-object report. The test now leaves deliberate producer stack storage
to the canonical compiler spill gate while requiring unchanged producer function
identities and zero private storage in scoring and selection. The original failed
receipt and selected disassemblies are retained in
`indexer108-cuda-preparation-resource-diagnostic-r1`.

The existing stored-score benchmark now records repeated selection nodes inside
one captured graph. It validates every node and the full linear dependency chain,
submits one graph per event interval, and divides by the captured count once.
Focused regressions cover repeat limits, removed-edge rejection, and retained
full/partial/empty rows on CUDA and ROCm. This removes host launch-loop timing
from the seven-distribution selector comparison; whole-indexer timing continues
to use its separate four-stage captured transaction. The new Release measurements
are pending and cannot replace the earlier 40 receipts until completed.

## Mathematical selector economics

The provisional 2048-pool contextual crossover is not a promotion rule. Fresh
same-runtime full-indexer measurements in
`indexer108-contextual-v3-event-summary-r1.json` show 636.320 -> 275.989 us at
64K for M1, but 82.460 -> 118.906 us at 8196 tokens. All 22 component cells
pass membership/replay checks. The contextual profile
`indexer108-v3-contextual-m1-c65536-valu-r1` authenticates the exact native entry,
42224-byte LDS bank, zero private storage and zero observed memory spills; four
scalar values occupy vector-register lanes. These are component results, not
model throughput or production-default qualification.

Let N be the device-live complete-pool count, K=512, and B=S=1024 the bounded
scan block and compaction slack. For N>K, the existing wide selector performs
33N logical score reads and 64+3*ceil(N/256) CTA barriers. These reads may hit
cache; 33N is not a claim about HBM traffic. The bounded algorithm reads N
scores, but its shared-memory selection work depends on their ordering.

For this exact geometry its tight worst-case number of compactions is

```
c_max(N) = max(1, ceil(N/1024) - 1)
W_max(N) = N + 512 * (c_max(N) - 1)
```

Here W is the sum of the input bank sizes over compactions. The conservation
identity is W=A+512*(c-1), where A<=N is the number of accepted keys. Ascending
unique scores attain both bounds. At N=2048, c_max=1 and W_max=2048; at
N=2049 they become 2 and 2561. This explains a structural cost discontinuity,
without treating a particular model activation or token boundary as universal.

Fourteen stored-score Release cells now pass on MI50 and RTX 3090, using all
seven authored score distributions at 64K. Each process verifies selected IDs
and source bytes, times one repeated captured graph per event, and reports
separate untimed work counts. For the current 1024/1024 candidate:

| Backend / distribution | Wide, us | Bounded, us | Compactions | Radix passes |
|---|---:|---:|---:|---:|
| MI50 / hash | 542.11 | 160.98 | 3 | 7 |
| MI50 / ascending | 533.27 | 824.39 | 15 | 76 |
| MI50 / equal | 533.25 | 59.67 | 1 | 1 |
| RTX 3090 / hash | 81.01 | 73.83 | 3 | 7 |
| RTX 3090 / ascending | 80.38 | 330.68 | 15 | 76 |
| RTX 3090 / equal | 79.72 | 32.18 | 1 | 1 |

The same context has opposite winners, so context length alone is insufficient
for a distribution-independent speed claim. The current implementation also
spends radix passes on original-ID bits. A score-only radix candidate with
canonical stable tie handling is under review; it must keep exact membership,
true retained thresholds and ascending output IDs while reducing the pass bound
from eight to four. Its extra tie-publication work must be measured too.

The intended dispatch rule is a calibrated cost inequality:

```
choose bounded when upper_bounded_cost(N, live_rows, resources) + margin
                    < lower_wide_cost(N, live_rows, resources)
```

Use the proven work bounds, nonnegative hardware cost coefficients and a
heldout residual envelope. Include score scans, compaction key reads, histogram
passes, barriers and the actual retained kernel's occupancy/resource regime.
The current observation counters do not determine histogram key reads:
sum(bank_size * passes) is required, not sum(bank_size) * sum(passes).
Calibration and heldout contexts/distributions must be separate, and grouped
latency cannot be inferred by multiplying M1 latency. Finite calibration provides
a validated cost prediction, not a formal universal hardware-time guarantee.
If the conservative inequality admits no winning region, improve the kernel;
do not substitute an optimistic token threshold.

Upstream's logarithmic expected candidate-work bound assumes an input-independent
uniform random block permutation. The current deterministic traversal does not
satisfy that assumption. A fixed affine permutation is also not equivalent to
a uniform choice among all block permutations. No such expected-work guarantee
is claimed for this implementation.


## Score-only radix and parallel histogram qualification (110)

The score-only candidate preserves full FP32 scores, canonical lower-ID ties,
ascending selected IDs and the true retained 64-bit threshold. It visits at most
four score bytes; equal-score banks visit none. Stable tie handling adds one CTA
publication barrier per 256-key output round. The subsequent one-warp pivot
adapts DeepSelect's five-step suffix over 32 groups of eight histogram bins;
only the unique owner probes its eight local bins. It adds no CTA barrier and
changes no score arithmetic or selected membership.

Fresh Integration passed 31 focused CPU/CUDA/ROCm registrations, including every
histogram owner/bin, exact serial/grouped bytes, retained full/partial/empty
replay, capture/stage forwarding and mathematical work bounds. The complete
freshly built Unit suite passed 776/776. Receipts are
`indexer110-parallel-histogram-focused-r1` and `indexer110-unit-r1`.
Release build r2 also removes NVCC's host-only numeric_limits constexpr warning
by using the compile-time INT_MAX macro and admits timing rows M1..16. CUDA
compilation covers 80/86/89/90; HIP covers gfx906 with the required spill guard.
The frozen component runtime is `/tmp/qfn-indexer110-parallel-histogram-runtime-r1`.
These gates do not replace full production preflight or real-model qualification.

All 14 stored-score cells (seven distributions on each backend) pass exact IDs,
source immutability, captured repetition and native retirement checks. For
B1024/S1024 at M1 and 64K tokens, unprofiled median microseconds are:

| Backend / score distribution | Installed | Score-only serial pivot | One-warp pivot |
|---|---:|---:|---:|
| MI50 / hash | 541.473 | 165.60 | 99.237 |
| MI50 / ascending | 533.293 | 674.97 | 256.368 |
| MI50 / equal | 531.682 | 58.53 | 59.800 |
| RTX 3090 / hash | 81.007 | 77.02 | 53.528 |
| RTX 3090 / ascending | 79.760 | 273.14 | 120.212 |
| RTX 3090 / equal | 79.696 | 34.45 | 32.636 |

The Installed column is the current one-warp receipt's paired control. The
serial-pivot column is the earlier separately recorded 109 candidate, not a
simultaneous pair. All three bounded geometries and all distributions remain in
the receipts; `indexer110-one-warp-stored-summary-r1.json` summarizes B1024.
Ascending now wins on MI50 but still loses on CUDA. A new stress distribution
will retain 511 permanent high keys while forcing repeated four-score-digit
frontiers, so calibration does not assume ascending ordinary floats maximize
histogram work or contention.

The full four-stage indexer also passed 28 component cells with actual produced
scores and independent pre/post numeric-sort membership checks. The one-warp
contextual candidate removes the previously measured 8196-token regression.
Selected unprofiled transaction medians (microseconds) are:

| M | Context | Installed | Contextual |
|---:|---:|---:|---:|
| 1 | 8196 | 82.577 | 55.408 |
| 1 | 12288 | 101.745 | 60.343 |
| 1 | 16384 | 161.817 | 99.059 |
| 1 | 65536 | 636.273 | 201.813 |
| 8 | 65536 | 803.671 | 377.106 |
| 16 | 65536 | 1052.715 | 630.060 |

M8/M16 at 8196 include rows on both sides of the live-context branch under one
retained graph. At M1/context512 the contextual wrapper costs 13.663 versus
13.214 us, so its short path also needs an economy check. These transactions
exclude query projections, attention, commit and whole-model inference. They
are not a new tok/s claim. Full details are in
`indexer110-one-warp-full-indexer-summary-r1.json`.

The first exact contextual profiler run has valid native/trace/counter evidence
but failed its private ISA observer: LLVM reports 31 scalar register-lane stores
using 29 physical VGPR lane slots, while the observer wrongly required 31
unique slots. Root inspected all 31 writes and 77 reads: lanes14/15 are reused
through a loop. Final metadata has VGPR spills0, private bytes0, dynamic stack
false, 71 used VGPRs and 100 used SGPRs; the loaded descriptor reserves132/112.
The canonical compiler spill guard passed. The failed receipt is retained;
a corrected static observer must distinguish write instructions from physical
slots and cannot claim CFG/lifetime proof or replace the compiler certificate.
No accepted profile is claimed for this candidate until that observer rerun.

For the score-only implementation the exact bounded barrier count is
`1 + 3*ceil(N/256) + 5*C + 3*D + 4*Q`, with D<=4*C and
Q=sum(ceil(bank_size/256)). The typed WorkBound conservatively bounds Q by
floor((W_max+255*C_max)/256). At N2049/B1024 this yields C2, W2561,
Dmax8, Qmax11 and at most106 body barriers. Output copying, inactive retirement
and the contextual lifetime barrier are explicitly outside that bound.
A five-coefficient calibrated envelope is the next screening model:

```
U_bounded = aB + bB*ceil(N/256) + cB*C_max(N)
L_wide   = aW + bW*ceil(N/256)
choose bounded only when U_bounded + margin < L_wide
```

Coefficients must be nonnegative and backend/compiler/resource-specific;
calibration and heldout sets stay separate. One conservative envelope across
M1..16 avoids extrapolating single-row latency. The compaction coefficient must
cover the bounded bank, four digits, histogram contention and tie work. Work
bounds are proved; hardware timing envelopes remain measured assumptions.
Standalone Installed measurements screen against the old production control;
qualification of an in-kernel rule must additionally measure its final physical
closure. Serving still uses Installed. No selector default has been promoted.


## Persistent four-digit stress and frozen calibration (111)

The appended test distribution ID7 uses 511 permanent FP32-max scores followed
by strictly increasing score words `0x01010101 + (id-511)`. It attains maximum
candidate arrivals/compactions and four score digits at aligned block frontiers.
An unaligned final pool count congruent to255 modulo256 ends after three digits;
tests and diagnostic expectations account for that carry case explicitly.
At 64K tokens B1024 executes15 compactions,23552 total bank-key visits and60
radix passes. These are observed native counters as well as the independently
derived workload identity. Membership remains checked by numeric FP32 sorting.

The expanded focused slice passed34/34 CPU/CUDA/ROCm registrations, including
all eight distributions and captured four-digit/carry/tail/empty replay. The
first build caught multiple SCOPED_TRACE macros on one source line; separate
lines fixed the test declarations. Build and failed/successful receipts remain
separate (`indexer111-persistent-frontier-integration-build-r1/r2`). Release
fixture rebuild passed, and the frozen111 core has exactly the same source
file stat identity as the110 native core; only the test fixture changed.

The corrected profiler observer also passed four fresh exact-selector launches:
VALUUtilization65.451062% (active vector lanes, not occupancy), LDSBankConflict
0.0314992944% (SDK metric), FetchSize64.3125KiB and four wavefronts. Final private
storage and observed memory spills are zero. There are31 static scalar-to-vector
lane writes using29 physical slots, with77 static reads; this observation is not
an allocation/lifetime proof and the compiler guard remains authoritative.
Installed Wide fetched67.0KiB in its separate matched profile. The33 logical
scan passes do not imply33-times HBM traffic; cache absorbs most repeated reads.
Accepted profile receipts use `indexer110-contextual-m1-c65536-*-r2`, with the
Installed control `indexer110-installed-m1-c65536-fetchsize-r1`.

The first private ID7 event parser accidentally shadowed its diagnostic-row list
with one row's expected-work dictionary. Native timing/oracle/lifecycle passed,
but that observer failed with KeyError; the r1 result stays failed. The v2
collector uses a distinct variable and schema, and calibration uses fresh r2
outputs. No failed result is relabeled or included in a fit.

All60 predeclared calibration cells now pass (30 per GPU), using one frozen
Release111 runtime, M1/M16 and selected complete-pool counts1024/2048/2049/4096/
16384. The matrix combines ordinary ascending/byte-boundary/hash/equal/extreme
scores with the four-digit frontier. At64K the harsh frontier's B1024 median
is341.376us versus533.251us on MI50. RTX3090 gives148.78us versus80.37us, while
its4096-token frontier gives9.64us versus15.47us. This is measured evidence for
backend-specific regions, not a universal increasing-context cutoff.

`indexer111-calibration-plan-r1/manifest.json` freezes the30 calibration and44
heldout cells for each backend before fitting. The collector contract pins its
source and that manifest. The reserved set includes every M1..16, unseen
3072/3073/8192-pool frontiers, all original distributions and nine additional
four-digit cases. Calibration and validation remain separate. The declared
empirical guard is max(2% of the sample,0.25us), applied above the maximum
bounded sample and below the minimum Wide sample. The decision screen is
`U_bounded + 0.5us < 0.99*L_wide`. No coefficient fit or heldout acceptance is
claimed by this calibration receipt alone. Production still uses Installed.


### First frozen timing model: rejected

Both fits were frozen before running the88 heldout native cells; all88 passed
native correctness and event/lifecycle checks. The timing envelopes themselves
failed and are retained as rejected, with no refit. Artifacts are
`indexer111-envelope-{rocm,cuda}-r1/{fit,heldout}.json`.

With t=ceil(N/256), C=C_max(N), the fitted microsecond coefficients were:

| Backend | Upper bounded | Lower Wide | Predicted complete-pool interval |
|---|---|---|---|
| MI50 | 6.63323 + 2.473402857*t + 12.252847143*C | 6.51473375*t | 1025..16384 |
| RTX3090 | 1.57981 + 1.20938*t + 4.88053*C | 10.923733333 + 1.059706667*t | 513..2048 |

The decision included the separately declared `UB+0.5 < 0.99*LW` guard. These
numbers are rejected experiment results, not installed policy constants.

MI50 had four bound misses. Three ascending cases at N3073 had guarded Wide
lower observations84.456..84.535us versus predicted84.692us. The four-digit
M16/N3072 upper observation was64.319us versus predicted60.820us. All44 chosen
MI50 switches nevertheless passed the guarded pairwise speed criterion; that
does not rescue the failed upper/lower-bound claim. CUDA had six bound misses,
mostly the four-digit cases, and its44 original heldouts were all outside its
predicted short winning region. Thus zero CUDA switches were validated by that
set; an empty no-regret check cannot qualify its short region. Any future short
validation needs a separately frozen prospective set. Neither backend's timing
model is promoted or quietly padded/refitted to these heldouts.

A separate24-cell complete-indexer check of direct B1024 at short contexts
passed. At M1/context512 it costs13.341 versus13.213us, a0.128us overhead;
at M16/context512 it costs13.738 versus13.617us. Unlike the contextual union,
the direct bounded all-visible path does not add a union-lifetime barrier.
When selection first becomes necessary at2052tokens, M1 improves47.043->33.199us
and M16 improves49.019->35.188us. M1/context4096 improves53.498->33.708us and
M16/context8192 improves133.641->100.346us. These component results remain
separate from the frozen stored-score timing fit and from model tok/s.
Full data are `indexer111-short-full-summary-r1.json`.

## Simple policy and CUDA crossover diagnosis (113–114)

The user stopped the timing-envelope refinement in favor of a simple rule with
at most 5% selection regret, then requested a focused investigation of CUDA's
input-dependent crossover. The prepared second LP matrix is retained privately
but was not run. The rejected first fits remain rejected.

For MI50 decode M1..16, direct B1024 wins all 81 retained selector comparisons
against Installed Wide. Eighteen fresh boundary/contention cells also pass:
M1/M16, maximum complete-pool counts513/769/16384, and ascending, persistent
four-digit and hot-histogram scores. The largest fresh B1024/Wide ratio is
0.676602, so the new check has zero regret between these two algorithms. This
is not a claim that B1024 is within5% of every alternative bounded geometry:
other geometries can win on particular synthetic score patterns. The separate
24-cell complete-indexer check covers the all-visible short path, whose overhead
is about0.12us and below5% in every measured case.

The simple production declaration now selects B1024 for typed ROCm Decode
capacities1..16. Its existing native all-visible branch returns all IDs through
512 complete pools; actual top-k work starts at513 pools (2052 visible tokens).
No host context inspection or additional dispatch kernel is introduced.
One-row Prefill, CPU, CUDA and other capacities retain Installed. The helper
and its sole model caller have device-free backend/phase/capacity regressions,
explicitly registered in production preflight. The declaration has passed the
focused real-model qualification below; a new production checkpoint is still
pending. The expanded pre-policy CPU/CUDA/ROCm slice passed37/37. After wiring
the production phase policy, the complete Unit suite plus focused backend
regressions passed814/814 in130seconds, with a clean driver window and normal
retirement (`indexer114-unit-and-focused-r1`). This is not a refreshed complete
production-preflight or full-model certificate.

The Release real-model accuracy run `indexer114-release-accuracy-r1` passed
with the same frozen short/held-out histories and threshold-crossing diagnostic
as L106. All185 final-logit rows across26 files are byte-identical to
`injection106-split-release-accuracy-r1`, including the history crossing2052
visible tokens where bounded selection becomes active. The canonical runner
retired normally with a clean driver window and unchanged source/runtime.
Mean KL/top1/top5 for short prefill are0.033535869/0.9375/0.875, short decode
0.032070057/0.979166667/0.922916667, held-out prefill
0.009691368/1/0.95, and held-out decode0.003837619/1/0.941666667.
The boundary history remains a byte-equivalence diagnostic rather than a new
absolute-KL acceptance threshold. This is the local required-group accuracy
slice, not a refreshed complete production-preflight certificate.

The matched L106/control-selector114 A/B/B/A Release run completed with normal
retirement, clean driver windows and the correct frozen core, HIP, RCCL and GNU
OpenMP closure in all four arms. Each arm has one warmup and three measured
N512/D256 iterations. All12 measured256-token sequences are identical.
Aggregating the six samples per policy gives control815.582691 prefill and
39.245717 after-prefill decode tok/s, versus candidate816.056225 and39.221305:
+0.0581% prefill and-0.0622% decode, inside the observed noise. The workload
stays below the ranking boundary, so this checks short-context overhead rather
than establishing a long-context selector gain. Receipts are
`indexer114-model-{a1,b1,b2,a2}-r1`; the independent aggregate is
`indexer114-model-summary-r1/summary.json`.

Fresh artifacts:

- `indexer113-simple-threshold-plan-r1/{manifest,validation}.json`:18/18 MI50.
- `indexer113-cuda-mechanism-r1/{manifest,summary}.json`:15/15 CUDA.
- Runtime `/tmp/qfn-indexer113-hot-histogram-runtime-r1`, Release with
  CUDA80/86/89/90 and HIPgfx906 compilation and spill guards.

### CUDA work explains the reversal

The same source, binary, score rows, capture geometry and exact numeric-sort
oracle reproduce the following M1/64K selector-only event medians. Diagnostic
work is collected separately from timing; times exclude score production,
attention and the rest of the model.

| Score distribution | Compactions | Compacted keys | Radix passes | B1024 us | Wide us |
|---|---:|---:|---:|---:|---:|
| Descending | 1 | 2048 | 3 | 36.648 | 80.334 |
| Hash | 3 | 4975 | 7 | 53.940 | 81.634 |
| Ascending | 15 | 23552 | 36 | 121.087 | 80.319 |
| Persistent four-digit | 15 | 23552 | 60 | 148.787 | 80.368 |
| Hot histogram | 15 | 23552 | 60 | 150.288 | 80.344 |

The bounded algorithm traverses original IDs. When the early top512 already
dominate later scores, it filters nearly all later arrivals. Increasing scores
continually replace the retained frontier, requiring repeated bank compaction.
Each compaction adds min/max reductions, histogram digits and stable compaction
with CTA publication barriers. The old Wide path performs a fixed31-bit
threshold search regardless of score ordering, followed by tie counting and
stable output. Its regular repeated reads are largely cached; logical read
counts must not be presented as HBM traffic.

The persistent four-digit CUDA series is a monotone staircase, rather than an
unexplained decrease in absolute latency. N2048->2049 adds one final compaction
and changes15.344->21.665us; N3072->3073 similarly changes25.504->31.840us.
The ratios versus Wide cross1 because the algorithms have different work
growth. At64K, the source has816 CTA barriers for the four-digit bounded case
versus256 for Wide; the descending bounded case has239. These are exact
source-program counts for those authored M1 rows, not profiler stall cycles.

Increasing histogram contention while keeping compactions, compacted keys and
radix digits fixed raises eligible atomic updates from55356 to74466 (+34.52%),
but latency changes148.787->150.288us (+1.01%, or+1.04% normalized to each
paired Wide control). This intervention points to repeated compaction/radix
work as the major cause; it does not establish that histogram contention is
free. The same pattern holds atM16. All actual CUDA candidate resources are
unchanged:56 registers/thread,42224 shared bytes, zero local bytes,256 threads
and one CTA per row. Four paired samples in the M1 four-digit cell span only
148.784..148.800us. Exact membership, causal tails, retained replay and driver
checks pass.

MI50 executes identical C/W/digit work, but its old Wide path costs about533us
at64K, leaving enough room for the341us difficult bounded case to win. CUDA's
old Wide path is about6.6times faster than MI50's, whereas that bounded case is
about2.3times faster. This backend ratio, together with data-dependent work,
explains the different winners. No context-only rule can be within5% of the
faster algorithm for every tested CUDA input: the same M1/64K shape favors
opposite algorithms by far more than5%.

CUDA production selection remains unchanged. A future optimization should
reduce repeated compaction/publication or use a predictable exact byte-radix
selection algorithm, rather than fit another context-only formula. No such
candidate is implemented by this diagnostic. CUDA hardware stall/occupancy
counters remain unavailable while profiler attachment is suspended following
the earlier driver fault; native events, source accounting and static resources
establish the workload explanation, not an instruction-cycle attribution.

## Observation counting and compiler scheduling (115)

The production QFN graph binds attention observations on every invocation.
`selectedAttentionPartitions` originally counted logical decoded coordinates
per thread and reduced those integers through FP32 shared storage, with20
diagnostic CTA barriers. Each nonempty partition instead has exact totals
`count * (256 + rotary_dimension)` for pre-RoPE keys, `count *256` for
post-RoPE keys, and `count *256` for values. These count logical decoder calls,
not HBM transactions or compressed bytes. Empty/inactive ownership and all
QK/softmax/PV arithmetic remain unchanged.

New explicit CUDA/ROCm preflight regressions passed first on the unchanged
kernel, then on the count-only candidate. They compare observation ON/OFF
partial and output bytes for every M1..16, dirty full/partial/empty/reset
replays, all native codecs/Q16 block widths and pre-RoPE keys. The expanded
Unit plus selected backend slice passed787/787 with a clean driver window.
This slice also calibrates a new literal historical GDN oracle against the
unchanged GDN production kernel. A performance-only Release configuration
exposed a test-registration error: target properties referenced its omitted
GDN integration target. The HIP source now belongs to the original target's
source list, with target-specific properties guarded by actual target existence.
Release rebuilt successfully; this was test wiring, not a numerical failure.

The first count-only implementation is **rejected for performance** despite
its exact-byte correctness. Paired native-event producer+merge medians include:

| M/context | Original ON us | Count-only ON us | Original OFF us | Count-only OFF us |
|---|---:|---:|---:|---:|
| 1/512 | 59.1736 | 96.52175 | 56.9339 | 96.02335 |
| 1/65536 | 59.5561 | 95.0265 | 57.3533 | 95.5262 |
| 16/512 | 73.0023 | 106.122 | 70.62825 | 105.5705 |
| 16/65536 | 190.7895 | 324.2185 | 187.434 | 323.7955 |

All eight measured M/context cells regress. Replaying the frozen original
afterward reproduces59.1862us ON and56.9296us OFF at M1/512, excluding
ordinary run-to-run drift as an explanation for this large change. The OFF
regression also excludes the runtime cost of the removed optional barriers
as its cause.

Exact gfx906 profiler/code-object evidence locates the code-generation change.
Both versions have64 SGPRs,3088 LDS bytes, no private scratch and192 launched
wavefronts for the M1/Q6/eight-partition producer. VGPRs drop32->20. Original
QK coordinates are fully unrolled: eight half loads at byte offsets0..448,
preloaded query registers and ordered FP32 FMAs. The count-only version rolls
that loop, loading one key and one shared query coordinate, waiting for both,
and executing EXEC loop control on every coordinate. PV remains four-way
unrolled. Fewer static instructions and registers therefore accompany much
worse latency. Hardware stall attribution is not inferred from the instruction
census alone.

The explicit fixed eight-coordinate lane schedule passed the same CUDA/ROCm
all-codec, serial/grouped, observation and retained-reset byte regressions.
It preserves ascending dependent FMA order while allowing independent loads
to move ahead. The speculative tail-barrier/liveness probes were not run once
the actual QK loop difference was identified. Release/Integration compile the
shipped CUDA80/86/89/90 and HIPgfx906 set with mandatory spill guards. The
actual revised MI50 producer uses32 VGPRs,64 SGPRs,3088 LDS bytes and no private
scratch; its launch still contains192 wavefronts at M1/Q6/eight partitions.

The revised ON medians are56.9600/57.09005/57.5166/57.49315us at M1 and
contexts512/768/2052/65536, versus59.1736/59.4117/59.89455/59.5561us originally.
At M16 they are70.96555/81.39345/175.886/187.8015us, versus
73.0023/83.7835/178.605/190.7895us. All eight cells improve by1.52–3.97%.
These are captured native-event producer+merge component timings, not model
throughput. All original/rejected/revised results remain retained.

`observation115-unrolled-release-accuracy-r1` passed the unchanged canonical
short/held-out and boundary corpus in93.23seconds. All185 logit rows in26 files
are byte-identical to selector114, including reset and the2052-token boundary;
KL/top1/top5 are unchanged. The application retired normally, the driver window
was clean, and source/runtime identities stayed fixed. The original-workload
model A/B/A completed against the frozen selector114 control. Across six control
and three candidate samples, the medians are815.694004/39.238692 versus
814.978841/39.207662 prefill/decode tok/s. The changes of-0.0877%/-0.0791% lie
inside the observed noise; all nine256-token sequences are identical. Every
arm retired normally with a clean driver window and authenticated source/runtime
closure. This is unchanged whole-model throughput, not a model-level win.
`observation115-model-summary-r1/summary.json` retains the comparison and the
exact frozen scripts/source. No new complete-preflight certificate follows.

After that source-frozen run, cleanup removes the rejected packed107 experiment
and unused DecodeContextual selector. The active bounded policies, production
selector114, fixed-order attention and observation-neutrality regressions remain.
The observation-only Release fixture now has its own target and identifiers,
without packed-kernel dependencies. Old frozen runtimes and evidence remain
unchanged; their names describe the binaries that actually produced them.

`observation115-cleanup-unit-native-r1` passes all821 selected tests after
rebuilding complete Unit and the affected native fixtures. This covers CPU,
CUDA and ROCm bounded selection, full indexer/attention, stage forwarding and
capture identity, observation neutrality, compact KV ownership, retained
grouped rollback, and the unchanged historical GDN oracle. Runtime retirement
and the driver window pass. The complete production-preflight inventory is
still outside this focused run.

Receipts: `observation115-unchanged-kernel-regression-r1`,
`observation115-unit-native-gdn-baseline-r1`, the eight
`observation115-{baseline,candidate}-m{1,16}-c{512,768,2052,65536}-events-r1`
cells, `observation115-baseline-after-m1-c512-events-r1`, paired
`observation115-{baseline,candidate}-m1-c512-wavefronts-r1` and selected
`observation115-{baseline,candidate}-producer-isa-r1` code-object extracts.
The winning component cells are `observation115-unrolled-m*-c*-events-r1`, with
resource proof `observation115-unrolled-m1-c512-wavefronts-r1` and focused native
gate `observation115-unrolled-focused-r1`.

### Cross-backend observation qualification

The shared native-event fixture now runs on both CUDA and ROCm, with thin
backend adapters for graph inspection and native function metadata. The same
fresh harness runs against frozen old/current cores after source-owned ABI
checks. All eight cells (M1/M16 at contexts512/65536 on both backends) produce
exactly identical output and FP32 partial bytes across versions. Each native
run also checks observation ON/OFF byte neutrality. No CUDA profiler was
attached; attachment remains suspended after the earlier driver fault.

| Backend | M/context | Original ON µs | Revised ON µs | Revised/original |
|---|---:|---:|---:|---:|
| CUDA | 1/512 | 45.7220 | 36.2080 | 0.7919 |
| CUDA | 1/65536 | 46.8780 | 35.96985 | 0.7673 |
| CUDA | 16/512 | 63.6520 | 67.6040 | 1.0621 |
| CUDA | 16/65536 | 184.0900 | 157.9120 | 0.8578 |
| ROCm | 1/512 | 58.9672 | 56.7534 | 0.9625 |
| ROCm | 1/65536 | 59.61535 | 57.4484 | 0.9637 |
| ROCm | 16/512 | 72.97815 | 70.96225 | 0.9724 |
| ROCm | 16/65536 | 190.5565 | 187.8555 | 0.9858 |

The CUDA grouped short-context regression is not accepted as an invisible
trade-off. It persists with observations OFF (62.11195→67.456µs), implicating
common-path code generation. Producer registers fall64→48, theoretical
resident-block limits rise4→5, and shared/local storage stays3076/0B. These
limits do not measure achieved occupancy. The merge stays32 registers,36B
shared and no local storage. C512 also leaves six of eight partitions empty,
unlike the long-context selected workload; instruction scheduling must be
measured across both geometries. A fixed-order PV scheduling probe is pending.

Receipts are `observation115-cross-{cuda,rocm}-{baseline,candidate}-m*-c*-r1`.
The private controller retains all commands, four ABI proofs and bounded
terminal exports in `/tmp/qfn-observation115-cross-qualification-r1/`.

### CUDA query reuse: rejected PV probe and retained-query candidate

Making the PV loop's four-token unroll factor explicit did not repair the
short grouped regression. The exact sm86 producer instruction listings for
the prior and PV4 candidates are identical (984 instructions); both use 48
registers. All eight byte comparisons pass, but CUDA M16/512 remains
67.674 µs versus 63.860 for its fresh original control. The redundant pragma
was removed. Static instruction counts describe the compiled program, not
executed work or hardware stall rates.

The instruction dump instead identifies eight shared query loads inside the
selected-token loop. Candidate120 retains those same eight FP32 values in
per-lane registers before the loop. The resulting sm86 steady-state loop has
no query shared loads and pipelines two independent tokens. Every token still
uses the same ordered eight FMAs and five shuffle/add steps. CUDA uses 56
registers, 3076 shared bytes, zero local bytes and a theoretical resident-block
limit of four. The source retains the same rule for all row counts, contexts
and native KV formats; there is no measured-shape dispatch threshold.

| Backend | M/context | Original ON µs | Retained query ON µs | Time reduction |
|---|---:|---:|---:|---:|
| CUDA | 1/512 | 45.954 | 34.908 | 24.04% |
| CUDA | 1/65536 | 46.814 | 37.492 | 19.91% |
| CUDA | 16/512 | 63.792 | 55.634 | 12.79% |
| CUDA | 16/65536 | 183.186 | 141.244 | 22.90% |
| ROCm | 1/512 | 59.1445 | 56.9001 | 3.79% |
| ROCm | 1/65536 | 59.58385 | 57.48455 | 3.52% |
| ROCm | 16/512 | 72.98195 | 70.9732 | 2.75% |
| ROCm | 16/65536 | 190.608 | 187.800 | 1.47% |

All eight old/current output and FP32 partial comparisons pass, along with
ON/OFF neutrality. The ROCm results preserve the earlier observation115 gains;
no additional ROCm improvement is claimed from query retention. CUDA M1/64K
samples span 35.460–37.528 µs, so the table retains the observed median rather
than selecting its minimum. These are component timings, not model throughput.

The larger prefill fixture now admits M1..2048 with M×repeats bounded at 8192,
checks output guards and empty-partition poison independently, and limits
terminal exports to 128 MiB. All twelve additional old/current comparisons
pass on both GPUs at M512/1024/2048, with zero-prefix context equal to M and
with context65536. Every terminal FP32 output and partial byte matches. Each
event graph records exactly two complete attention invocations; the reported
time is already normalized per invocation.

| Backend | M/context | Original ON µs | Retained query ON µs |
|---|---:|---:|---:|
| CUDA | 512/512 | 647.168 | 529.408 |
| CUDA | 512/65536 | 5075.710 | 4271.355 |
| CUDA | 1024/1024 | 2444.800 | 2097.410 |
| CUDA | 1024/65536 | 10207.750 | 8534.530 |
| CUDA | 2048/2048 | 9605.120 | 8055.295 |
| CUDA | 2048/65536 | 20432.350 | 17158.400 |
| ROCm | 512/512 | 678.959 | 671.846 |
| ROCm | 512/65536 | 5145.750 | 5091.690 |
| ROCm | 1024/1024 | 2530.840 | 2505.320 |
| ROCm | 1024/65536 | 10269.300 | 10137.900 |
| ROCm | 2048/2048 | 9842.110 | 9763.790 |
| ROCm | 2048/65536 | 20448.450 | 20324.500 |

CUDA time falls14.21–18.20% in these cells. The modest ROCm changes are
consistent with removing optional observation reductions; OFF timings stay
effectively unchanged. At fixed64K context the selected-token budget is
approximately constant, and doubling rows approximately doubles time. That
does not claim linear zero-prefix prefill versus prompt length, where later
rows also attend to more tokens. The rebuilt complete-Unit and affected
all-codec/native gate passes all 823 CTest registrations (776 Unit, 46 Integration
and one setup) in169.92 seconds, with no CTest failures or skips and clean
driver/retirement checks. The subsequent97.10-second canonical model panel
passes its required groups; all185 final-logit rows in26 files remain
byte-identical to the accepted GDN118 control. Threshold/reset rows also match,
without adding an absolute KL gate to the diagnostic threshold groups.

Evidence: `observation119-pv4-cross-*`, `observation120-query-registers-cross-*`,
the corresponding all-architecture Release builds, and exact sm86 producer
dumps in `/tmp/qfn-observation{119,120}-cuda-sass-r1/`. The old, rejected and
current binaries remain separate. No CUDA profiler attachment was attempted.
The larger-shape receipts are `observation120-prefill-cross-*`, with the
commands, ABI proofs and exact exported comparisons in
`/tmp/qfn-observation120-prefill-qualification-r1/`.

The next candidate targets matched complete attention-stage time below
Castagna. FP16 arithmetic is explicitly allowed for this experiment; its
mean final-logit KL and top-1/top-5 agreement must be reported separately for
prefill and decode. Serial/grouped byte equivalence remains an invariant
within each policy, not a requirement that FP16 reproduce FP32 bytes.
Candidate120 remains the FP32 comparison control. No new FP16 speed or
accuracy result is claimed yet.

## Candidate122: shared selected-KV tiles with bounded half accumulators

The next component uses four canonical global query heads per CTA, one wave
per head and one lane per complete token score. Each score executes128 ordered
packed-half dots with FP32 accumulation. Eight fixed streams consume64-token
tiles at stride512; neither M nor the compact head shard changes a head's
arithmetic. The half-wave PV chains accumulate even/odd tokens separately in
FP16, widen into independent FP32 online totals, and merge the two parities
once in a fixed order. Exact1/128 probability scaling bounds the tile's half
accumulators even at maximum finite FP16 V. The FP32 denominator carries the
same scale. Default serving is unchanged.

Unlike the earlier post-RoPE component fixture, this comparison explicitly
includes selected-key transformation: native pre-RoPE FP16 keys,
canonical-engine factors,64 rotary coordinates and theta10000000 read from
the same model's GGUF metadata. Both arms borrow identical queries, KV,
selection and mappings. This is a supported component mode, **not the matched
model's native-FP16 production mode**: `QwenGraphBase` selects `PostRotary`
for native floating caches. The earlier "production-read" fixture name was
misleading; the component correction below keeps both modes explicit.
The first build's real register spills were rejected. Bounded loop unrolling
removes them without relaxing the guard. The final producer reports83 used
VGPRs,19968B LDS,zero private storage and a theoretical limit of three resident
blocks; these are resources, not measured achieved occupancy.

The first native groupedM1..16/serial comparison passes retained dirty replays
and guard checks. Initial alternating event medians for the complete producer
plus reducer are:

| M / context | Current FP32, µs | Candidate122, µs |
|---|---:|---:|
| 1 / 512 | 71.121 | 29.720 |
| 16 / 512 | 137.346 | 74.731 |
| 512 / 512 | 1385.970 | 869.488 |
| 1 / 65536 | 71.308 | 93.624 |
| 512 / 65536 | 11731.600 | 5896.460 |

The long-context one-row regression remains open. These synthetic component
gains do not establish a complete attention-stage win against Castagna or a
model-throughput gain. Model KL/top-1/top-5, broader compact-shard/extreme-value
tests and isolated hardware profiling remain pending. Evidence is retained in
`attention122-fp16-*`; the r1 spill failure and r2 fixture compile failure remain
separate from the successful r4 build. The initial native binary/core/source
archive is `/tmp/qfn-attention122-r4-runtime`.

### Candidate123: rotate each native split-half pair once

The key loader now owns both outputs of each64-wide split-half rotary pair.
It loads each raw coordinate once, evaluates each factor once, and retains the
original separately half-rounded coefficients, products and sums. Native key
reads fall from320 to256 coordinates per selected token/group. The query dot,
PV chains, partitions and barriers remain unchanged. Across five old/new cells,
every output and all258 numerical FP32 words per partial match byte for byte;
read counters are intentionally compared separately because their meaning is
actual shared coordinate reads. The expanded tests also pass compact degrees
1–8, both pre/post-RoPE banks, read-only operands, observation neutrality and
maximum-finite FP16 values with an independent scalar half-rounding oracle.

| M / context | Current FP32, µs | Paired-loader FP16, µs |
|---|---:|---:|
| 1 / 512 | 71.173 | 23.868 |
| 16 / 512 | 137.120 | 47.840 |
| 512 / 512 | 1386.560 | 597.480 |
| 1 / 65536 | 71.375 | 71.853 |
| 512 / 65536 | 11730.900 | 3983.325 |

The long one-row case is now near the FP32 control, still0.67% slower in this
sample. The other measured cases reduce attention-pair time by56.9–66.5%.
Isolated sixth-occurrence ROCm profiles authenticate the actual producer:
68 allocated VGPRs,112 SGPRs,19968B LDS and zero scratch. Measured wavefronts
are64 at M1 and32768 at M512, matching the declared grid. LDS bank-conflict
counters report zero in the three profiled shapes. VALUUtilization at M1/512
reports83.81%; it is not achieved device occupancy. Profiled durations are
excluded from the timing table.

`LLAMINAR_QFN_SPARSE_ATTENTION=fp32|fp16_tile` is wired as a setup-only
explicit experiment. Its typed policy is frozen into the model graph, stage
capture identity and native arguments. FP32 remains the default; the FP16
experiment requires a ROCm attention participant and native FP16 KV, with
precise rejection elsewhere. Both local model panels use unchanged canonical
histories, full-vocabulary scoring and required-group gates, while recording
the experimental policy separately. This is not a default promotion or
full-preflight claim.

Evidence: `attention123-paired-rope-functional-r1`,
`attention123-paired-rope-byte-comparison-r1.json`, the matched
`attention123-{old,paired}-m*-c*-r1` event receipts, and
`attention123-isolated-profile-summary-r1.json` with eight isolated launches.

### Same-build final-logit precision comparison

The fresh FP32 control and explicit FP16 candidate both pass every required
canonical group. Each panel retains 178 scored rows and seven reset rows.
Actual sealed attention policy is observed on all four ROCm participants;
source/runtime identity, driver checks and native retirement pass. The table
reports mean final-logit KL in nats, top-1 agreement and mean top-5 overlap
against the same reference. It does not pool prefill and decode or hide the
held-out group.

| Group / phase | Rows | FP32 KL | FP16 KL | Top-1 FP32 / FP16 | Top-5 FP32 / FP16 |
|---|---:|---:|---:|---:|---:|
| Short / prefill | 16 | 0.033536 | 0.025614 | 93.75% / 93.75% | 87.50% / 92.50% |
| Short / decode | 96 | 0.032070 | 0.028276 | 97.92% / 95.83% | 92.29% / 92.29% |
| Held-out short / prefill | 8 | 0.009691 | 0.014911 | 100% / 100% | 95.00% / 95.00% |
| Held-out short / decode | 48 | 0.003838 | 0.004553 | 100% / 100% | 94.17% / 91.67% |
| Selection threshold / prefill, diagnostic | 1 | 0.000202 | 0.000168 | 100% / 100% | 80.00% / 100% |
| Selection threshold / decode, diagnostic | 9 | 0.000211 | 0.000368 | 100% / 100% | 86.67% / 86.67% |

The user accepted this precision trade-off on 2026-10-23. Promotion still needs
the matched unprofiled whole-model benchmark and refreshed Unit/affected native
tests. The 64K component measurements above do not establish 64K model KL.
The current functional Release sweep passes all 15 selected native tests;
the two environment-dependent profiler cases are excluded explicitly from
that functional invocation. The earlier invocation that included them is
retained as a wrapper failure because it correctly rejected their skips.

Evidence: `attention123-model-{fp32,fp16}-r1`,
`attention123-model-precision-comparison-r1.json`,
`attention123-typed-native-r2`, and the all-shipped-architecture
`attention123-policy-release-build-r1` build. The typed-policy wiring recompiles
the producer at 62 used VGPRs, still with 19968B LDS and zero private storage;
final-build component timing/profile refresh remains pending.

### Candidate124: remove redundant synchronization and unowned arithmetic

The first key-publication barrier now also publishes the disjoint query bank.
The final value-panel retirement barrier executes only when another value panel
or tile will reuse that storage. This removes two redundant barriers per
nonempty CTA. A wave owns one complete global query head, so unowned waves now
skip QK, softmax, PV and final arithmetic while retaining all cooperative loads
and required CTA barriers. Neither the arithmetic of owned heads nor the
canonical global groups/eight tile streams changes.

The Release build passes the spill guard and all 15 selected functional native
tests pass. Seven old/new event cells preserve every output and numerical
partial byte, covering M1,16,512,1024,2048 and contexts512/65536. Both old and
new arms use the same final fixture and compatible frozen argument ABI.

| M / context | FP32 control, µs | Prior123 FP16, µs | New124 FP16, µs |
|---|---:|---:|---:|
| 1 / 512 | 71.135 | 23.850 | 23.950 |
| 16 / 512 | 136.290 | 47.365 | 46.720 |
| 512 / 512 | 1383.220 | 592.229 | 584.239 |
| 1 / 65536 | 71.320 | 71.807 | 71.732 |
| 512 / 65536 | 11828.750 | 3998.435 | 3965.850 |
| 1024 / 65536 | 23732.600 | 7931.930 | 7857.630 |
| 2048 / 65536 | 47469.300 | 15803.850 | 15660.600 |

The one-row change is effectively neutral; larger rows improve0.8–1.4%.
At64K, doubling M512→1024→2048 costs1.981× then1.993×, retaining the fast
path beyond the default bucket. These are complete producer-plus-merge
component timings with selected-key RoPE, not whole-model throughput.
The refreshed Unit/native gate, exact isolated profile and final model panel
were initially pending. The refreshed gate now passes all **825 registrations**
(complete Unit and affected CUDA/ROCm integration), and the final FP16 model
panel passes the unchanged required groups. All **26 logit files /185 rows**
are byte-identical to the accepted123 FP16 panel, including reset replay.
Evidence: `attention124-release-build-r1`,
`attention124-typed-native-r1`, `attention124-component-comparison-r1.json`
and all14 individual old/new receipts, `attention124-unit-native-r1`, and
`attention124-model-fp16-r1/logit-byte-comparison.json`.

#### Component key-encoding correction

The current model graph selects post-RoPE K for FP16/FP32/AUTO native caches
in `QwenGraphBase::resolveAttentionExecutionPolicy`; no QFN override changes
that decision. The prior timing fixture forced the supported pre-RoPE mode.
Its timings and `<true>` profiles remain valid for that explicit mode, but
they cannot authenticate the matched model's `<false>` specialization or rank
its production attention stage against Castagna. The canonical model accuracy
panels used the real model graph throughout and are unaffected.

The fixture now requires an explicit key encoding, retains the same native
mapping in both modes, reports it in every timing record and verifies the exact
captured `<true>`/`<false>` specialization. Its functional regression covers
both encodings at M1..16 through dirty full/partial/empty/long retained replays;
all15 selected native tests pass. The explicit preflight registration names
the renamed `FP16NativeKeyEncodingsGroupedSerialDirtyReplay` test. The rebuilt
Integration binary and that explicit registration pass in
`attention128-key-encoding-integration-r1`. Whole-model A/B/A uses the actual
model policy and the identical frozen124 runtime for both arithmetic choices.

The seven corrected post-RoPE complete-pair measurements are:

| M / context | FP32 control, µs | Candidate124 FP16, µs |
|---|---:|---:|
|1 /512 |56.88435 |21.31625 |
|16 /512 |70.92745 |42.84090 |
|512 /512 |670.329 |505.569 |
|1 /65536 |57.65870 |64.43840 |
|512 /65536 |5057.055 |3370.245 |
|1024 /65536 |10073.100 |6707.760 |
|2048 /65536 |20103.100 |13343.900 |

The M1/64K cell is11.76% slower than FP32. Default promotion is held while this
regression is tuned; the pre-RoPE near-tie cannot stand in for this result.
At64K, larger-M time still scales about1.99× per doubling. Evidence is in
`attention124-post-m*-c*-r1` and `attention124-key-encoding-native-r1`.

Twelve isolated producer profiles (six counters at each M1 context) report
64 wavefronts at both contexts, no LDS conflicts and zero private scratch.
VALU instructions/wave rise from885.25 to3253 at long context; VALU Busy is
2.256%/2.924% and memory-unit Busy10.050%/12.030%. These are not achieved
occupancy or HBM bandwidth percentages. Exact `<false>` ISA already has wide
LDS loads feeding four-dot QK windows. Evidence:
`attention124-post-isolated-profile-summary-r1.json`. The subsequent bounded
unroll8 results and source-derived opportunities are recorded in
[the structural comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md).

### Matched124 unprofiled model result

One immutable Release executable/core serves both explicit policies, in
FP32→FP16→FP32 order. Each arm has one warmup and three measured requests:
512 original token1228 prompt IDs,256 emitted tokens/255 after-prefill decode
steps, context4096, native FP16 KV, RAM ngrams, MTP/prefix cache off, full TP4
on the original four MI50s, vocabulary-sharded output and28 physical NUMA1
host workers. The six other arithmetic choices and native loader closure are
unchanged. Actual model cache encoding is post-RoPE.

| Phase | FP32 median (six samples) | FP16 median (three samples) | Change |
|---|---:|---:|---:|
| Prefill tok/s | 816.091874 | 817.871298 | +0.218% |
| After-prefill decode tok/s | 39.390114 | 39.754473 | +0.925% |

Decode ranges do not overlap: FP32 **39.248462–39.508997**, FP16
**39.725820–39.838323**. Prefill ranges overlap. Every one of the nine natural
256-token sequences matches, so the timing comparison also retains the same
generated trajectory. All three arms pass source/runtime/loaded-library,
driver, passive resource and native retirement checks. This is a modest
whole-model improvement; it does not close the42.959 tok/s Castagna target.

Evidence: `attention124-model-summary-r1/summary.json` and the three
`attention124-model-throughput-{a1,b1,a2}-r1` receipts. The summary directory
retains exact settings, commands, source patch and untracked source pins.
The accepted precision change and refreshed825-registration/185-row gates
qualify this explicit candidate. Promotion remains held for the corrected
post-RoPE M1/64K component regression above; full preflight remains G60.

The contemporaneous pinned Castagna repeat measures **462.165607 prefill /
42.656949 decode tok/s**; its three decode samples span **42.625311–42.670483**.
All generated IDs match the nine Llaminar samples, and the reference passes
runtime, driver, resource and retirement checks. Llaminar's candidate is
**1.76965×** the reference prefill rate and **6.804% behind** its decode rate.
The historical42.959 target remains recorded; the fresh reference does not
replace it with a selectively lower historical result. Evidence:
`attention124-castagna-baseline-r1` and
`attention124-fresh-engine-comparison-r1.json`.

The first fresh Castagna profiler collection, `attention124-castagna-trace-r1`,
completed inference and export but is **rejected**: the driver observer recorded
an `amdgpu_amdkfd_restore_userptr_worker` CPU-hog warning. It is not admitted as
clean stage-timing evidence. Passive inspection found all four GPUs idle with
normal PCIe links and temperatures, with no later GPU fault/reset message.
A separately retained repeat uses the unchanged health gate and workload;
it cannot erase the rejected collection.

That repeat, `attention124-castagna-trace-r2`, and the frozen Llaminar trace,
`attention124-stage-trace-r1`, both pass. Actual Llaminar execution confirms
post-RoPE `selectedAttentionFP16Tiles<false>` on all four GPUs. The source-bound
complete-core comparison still favors Castagna: median per-GPU prefill is
**12.2255 versus3.9578µs per prompt row**, and decode is **0.581918 versus
0.162666ms per step**. Whole-model improvement is not an attention-core win.
These instrumented different-arithmetic timings exclude external wrapper work
and cannot be converted into predicted removable latency. Exact counts,
physical head work, overlap accounting and retained inputs are in
`attention124-core-comparison-r1`; see `QFN_CASTAGNA_STAGE_COMPARISON.md`.
