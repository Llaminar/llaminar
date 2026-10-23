# QFN attention: actual Castagna and Llaminar execution structures

This comparison uses Castagna revision
`3483d462715d61f21ed5877836be35a724446359`, the kernels authenticated by the
current four-MI50 traces, and Llaminar candidate124. It distinguishes source
structure from measured causation. The source findings inform subsequent
arithmetic-preserving candidates; they do not establish a new production
default or an end-to-end speedup by themselves.

## Matched execution and current gap

The actual Castagna decode specialization is
`flash_attn_tile<256,256,1,4,false,false,false>`: D256, four query heads per
group, no indexed tile reads and persistent half PV accumulation. Its large
prefill chunks use `fdg_prep<12,true>`, `fdg_attn<12>` and
`fdg_combine<12>` (default dense variant0), followed by the generic four-row
tile kernel for the tail. The complete prompt is320+188+4 rows.

Llaminar uses native FP16 keys stored **after RoPE**, executing
`selectedAttentionFP16Tiles<false>`, followed by its eight-part reducer.
Each GPU owns six query heads on twelve attention layers; Castagna owns twelve
heads on six layers. Both therefore execute72 query-head/layer instances per
GPU. Per-layer launch counts alone are not an equivalent-work comparison.

The clean complete-core traces report median per-GPU decode time of
0.581918ms/step for Llaminar versus0.162666ms for Castagna. Prefill is
12.2255 versus3.9578µs/prompt row. These are instrumented, different-arithmetic
core measurements, excluding external projections, RoPE/cache publication,
gates and indexer work. They are not predicted removable request latency.
Matched unprofiled throughput is817.871/39.754 prefill/decode tok/s versus
462.166/42.657 for Castagna. See
[the complete attribution](QFN_CASTAGNA_STAGE_COMPARISON.md).

## Decode: what already matches and what differs

| Property | Actual Castagna decode | Llaminar124 |
|---|---|---|
| Score ownership | One physical wave64 per query head; one score/lane | Same |
| QK arithmetic |128 ascending packed-half dot2 operations with FP32 accumulation | Same operation count and dimensional order |
| Tile and transfers |64 keys, two128-coordinate K panels,16-byte vector loads and padded LDS | Same basic organization |
| QK loop scheduling | Fully unrolled source | Bounded unroll4; native ISA already uses two `ds_read_b128` followed by four dependent dot2 updates |
| Partitions | Shape/occupancy-selected;12 in the traced decode | Eight immutable interleaved tile streams |
| PV state between tiles | Four persistent half2 totals, rescaled in half | Four tile-local half2 totals widened into four float2 online totals |
| Denominator reduction | Per-lane FP32 accumulation, reduced after the KV loop | Fixed64-lane reduction for every tile, then ordered FP32 online accumulation |
| Addressing | Contiguous rows in this tile specialization | Selected indices and canonical ring mapping |
| Allocated resources |84 VGPR,112 SGPR,20000B LDS, zero scratch |68 VGPR,112 SGPR,19968B LDS, zero scratch |

The packed-half dot instruction and wide LDS loads are already present in our
kernel. Adding another vector wrapper would repeat existing compiler output.
Wider bounded unrolling can instead expose more independent loads ahead of the
same ordered score chain and reduce loop control. It does not shorten that
chain or reduce the128 dot operations.

The fork has36 producer CTAs per active layer in the traced decode
(three head groups × twelve partitions), versus our16 (two groups × eight).
This offers more per-launch parallelism. Its dynamic partition policy also
changes the reduction tree; copying it directly would violate our current
serial/grouped arithmetic contract. A physical scheduling change must retain
the same eight logical streams and merge order.

Persistent half PV and delayed denominator reduction remove work, but they
are **arithmetic changes**. Our exact1/128 probability scale bounds each
32-term half parity chain for finite logits/exponentials, even for maximum
finite FP16 V values; widening after each64-token tile prevents long-stream
half overflow. This does not make nonfinite QK or an overflowing half-cast Q
safe. The fork's approximate
ln8 maximum offset and persistent-half carry do not provide that same bound.
Any candidate borrowing those choices needs a declared arithmetic policy,
finite-range witnesses and fresh canonical final-logit KL qualification.

Source counts do not support a fewer-barriers explanation for the fork's
advantage. For a nonempty stream of T tiles, its source has8T+2 phase barriers;
current Llaminar has8T−1 after
candidate124 removed redundant query publication and final V retirement.
The remaining barriers protect real cross-wave LDS dependencies; native wait
placement and scheduling can still affect their elapsed cost. Similarly,
our ring mapping is conditional add/subtract, not integer division. Indexed
address work remains a possible cost, but its source repetition is not proof
of repeated HBM transactions. Long-context fork compaction must be included
when comparing the broader attention stage.

## Prefill: much more independent work per loaded tile

Castagna's dense prefill assigns **four tokens × twelve heads =48 query rows**
to one256-thread CTA. Each lane maintains twelve independent FP32 score
chains: three rows × four cells. Q and K register operands serve several
scores, and each V operand serves twelve output rows. Llaminar currently
handles one token × up to four heads, with one score chain/lane.

That is a clear source-level reuse opportunity: reuse the loaded KV data across
more independent rows, and overlap their score dependencies without changing
the operation order within any score. Our local six-head TP shard cannot
simply copy the fork's twelve-head ownership, and arbitrary sparse selections
need not match between adjacent query rows. A new layout must preserve each
row's selected membership, causal bounds and ring mapping.

Other relevant differences:

* Castagna prepares scaled-half Q separately, then loads global vectors into
  registers before retiring the old LDS panel. Padded LDS and an explicit
  LDS-only barrier permit overlap with independent VMEM work. A barrier change
  needs an actual dependency proof; no producer/consumer edge may disappear.
* Dense prefill uses FP32 PV accumulation and a denominator derived from
  rounded half probabilities. Our tile policy uses bounded half PV chains and
  a denominator derived from FP32 exponentials. Copying its prefill arithmetic
  would create another numerical program, despite the shared dot2 QK path.
* Dense prefill merges two or three contiguous segments, versus our eight
  fixed streams. Lower partial traffic partly follows from a different
  arithmetic partition policy, not just a better buffer layout.
* Its merge packs four output rows per256-thread CTA: one wave64 per row and
  four output dimensions per lane. Our reducer uses one CTA per row/head and
  one dimension per lane. Packing independent rows can preserve every original
  eight-part max, scale, denominator, FMA and division operation. It does not
  reduce the necessary numerator bytes by itself.

The current Llaminar reducer already computes max/scales/denominator once in
lane0 and publishes them; it does not repeat all that metadata arithmetic in
every thread. A replacement must be measured against that actual code.

## First measured source-derived scheduling change

Candidate127 changes only the two QK loop unroll factors from4 to8. KV loaders,
PV loops, grouping, eight streams, partial ABI and all arithmetic remain intact.
The corrected post-RoPE fixture measures the full producer-plus-merge pair,
with twelve alternating FP32-control/FP16-candidate event samples per cell.

| M / context | Original124 FP16, µs | Candidate127 FP16, µs | Reduction |
|---|---:|---:|---:|
|1 /512 |21.30435 |20.42310 |4.14% |
|1 /65536 |64.42590 |60.84805 |5.55% |
|512 /65536 |3370.050 |3233.395 |4.05% |

All outputs and258 numerical partial words per live partial are byte-identical
in the three old/new comparisons. All15 focused native tests pass, including
both pre-RoPE and post-RoPE M1..16 grouped/serial dirty replay. This is component
evidence, not a new model KL or throughput claim. At M1/65536 the candidate
still takes5.37% longer than its57.74650µs FP32 control, so default promotion
remains held while the scheduling candidates are measured.

Evidence: `attention127-qk8-release-build-r1`,
`attention127-qk8-native-r1` and
`attention127-qk8-{old,new}-m*-c*-r1` under the ignored local QFN results root.
The subsequent16-wide window is the measured winner;32-wide is slower on the
three trial cells and is not retained. The refreshed seven-cell comparison of
original124 against128 uses the same final fixture and frozen original core:

| M / context | Original124 FP16, µs | Candidate128 FP16, µs | Reduction |
|---|---:|---:|---:|
|1 /512 |21.6175 |20.5025 |5.16% |
|16 /512 |43.1999 |40.6874 |5.82% |
|512 /512 |510.3595 |488.0695 |4.37% |
|1 /65536 |65.00495 |59.49990 |8.47% |
|512 /65536 |3367.440 |3183.930 |5.45% |
|1024 /65536 |6709.130 |6319.410 |5.81% |
|2048 /65536 |13343.900 |12524.900 |6.14% |

Every output and numerical partial byte remains identical in all seven pairs.
At long context, time scales1.985× and1.982× for the two M doublings.
The M1/64K FP32 control is58.49995µs: the remaining regression is1.71%.
The15-test native sweep passes, including both key encodings.
These are QFN sparse-attention cells: at most512 complete four-token groups
plus the causal tail are selected (2051 tokens maximum). A64K cache is not
65536 attended keys, and this timing excludes indexer selection work.

The actual post-RoPE ISA now issues eight128-bit LDS reads ahead of sixteen
ordered dot2 instructions, using progressive `lgkmcnt` waits instead of a wait
after every four-dot window. Compiler-used VGPRs rise56→75; actual profiler
allocation rises68→76. LDS remains19968B and scratch/spills remain zero.
Twelve separately isolated counter launches retain64 wavefronts and zero LDS
conflicts at both contexts; fetch volume stays520.4375/2062.4375KiB. VALU
instructions/wave fall885.25→849.5 at512 and3253→3120.5 at65536. Resource
limits permit three resident CTAs/CU; this is not measured occupancy.

Evidence: `attention128-qk16-summary-r2.json`,
`attention128-qk16-native-r1`,
`attention128-post-isolated-profile-summary-r1.json` and the exact gfx906 ISA
retained locally in `/tmp/qfn-attention128-native-isa-r1`.
The retained128 model refresh passes every required canonical group. All26
final-logit files /185 rows, including reset rows, are byte-identical to the
accepted124 FP16 result, so KL/top-1/top-5 remain unchanged. The refreshed
complete Unit plus affected CUDA/ROCm gate passes825 registrations with no
failures or skips and clean driver/native retirement checks. Evidence:
`attention128-model-fp16-r1/logit-byte-comparison.json` and
`attention128-unit-native-r1`. The subsequent source-frozen unprofiled A/B/A
passes: FP16 measures818.837751 prefill /39.986885 decode tok/s, versus pooled
FP32 controls816.164783 /39.283088. All nine256-token sequences are identical;
decode ranges39.882062–40.070589 and39.195133–39.365111 do not overlap. The
workload is512 prompt tokens,256 emitted tokens/255 decode forwards, context4096,
fullTP4 with sharded vocabulary,28 physical NUMA-local workers, RAM ngrams,
native post-RoPE FP16 KV, MTP/prefix cache disabled; each arm has one warmup and
three measured iterations. Evidence: `attention128-model-throughput-summary-r2`
and the three `attention128-model-throughput-{a1,b1,a2}-r2` receipts. The first
batch invocation lacked its required outer CPU affinity and failed before
inference; the corrected invocation retains the startup guard. Complete
preflight remains G60. Decode is now the tuning priority; retain prefill as
a regression guard and qualify phase-specific physical schedules without
changing grouped-versus-serial arithmetic.

## Packed merge: accurate, rejected for decode economy

Candidates130/131 implemented four wave-owned output tasks per CTA without
changing any original max, scale, denominator, FMA, division or gate expression.
They passed16 focused native tests and all seven saved component output/partial
comparisons. A literal test-only historical reducer independently checked all
eight active partials, extreme gates, head tails, dirty inactive/empty replays
and read-only inputs. Neither candidate is retained in production.

| Complete pair | Original reducer + unroll16 | Initial packed130 | Wave-uniform packed131 |
|---|---:|---:|---:|
|M1 /512, µs |20.5025 |22.9850 |21.23995 |
|M16 /512, µs |40.6874 |43.2298 |41.4324 |
|M512 /512, µs |488.0695 |483.7390 |478.7695 |
|M1 /65536, µs |59.4999 |61.9249 |60.1774 |

Candidate131 explicitly broadcasts the wave owner before division/addressing,
as the fork does for its row owner. Passing the admitted X grid extent directly
also avoids an implicit HIP grid-dimension argument block. Actual kernarg size
falls584→332 bytes, compiler-used VGPRs46→40, and static global-load instructions
41→17 (scalar-load instructions9→33). The M1 merge counter drops303.25→253
VALU instructions/wave. All sampled profiles report zero scratch and zero LDS
bank conflicts. These facts describe the new compiled schedule, not an isolated
causal attribution between its two changes.

The revised packed layout remains about0.7µs slower at decode; larger-M changes
are small and do not justify installing this as the default merge. Both
experimental kernels, launch changes and their dedicated test closure were
removed, leaving the simpler original reducer and unroll16 producer. Exact
patches, historical-oracle source, ISA, profiles and rejected measurements are
retained locally. Evidence: `attention130-packed-merge-summary-r1.json`,
`attention131-wave-owner-summary-r1.json`, their native/profile receipts and
`/tmp/qfn-attention125-packed-reducer-private` plus
`/tmp/qfn-attention131-wave-uniform/final-source`.

## Primary source map

All Castagna references below are pinned to the compared revision:

* [Tile QK, softmax, PV, final reduction and dispatch](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/fattn-tile.cuh)
* [Dense GCN preparation,48-row core and packed merge](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/fattn-dense-gcn.cu)
* [Partition launch and merge machinery](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/fattn-common.cuh)
* [gfx906 packed-half dot and copy-width definitions](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/common.cuh)

Llaminar ownership remains in
`src/v2/kernels/rocm/attention/ROCmQwenSparseFP16Attention.cuh`,
`ROCmQwenSparseAttention.hip`, and the reducer in
`src/v2/kernels/qwen4exp/QwenSparseGPU.cuh`. Native key-encoding and arithmetic
regressions live in `QwenSparseFP16ROCmTest.h`; their explicit preflight entry
is `V2_Integration_ROCm_QwenSparseFP16TileArithmetic`.
