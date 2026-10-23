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

## Smaller decode head groups: exact, rejected for economy

Diagnostic132 retained one wave per head and the original eight numeric streams,
but shared each native tile across one or two heads instead of four. Cooperative
loader strides, native capture geometry, source-read counters and occupancy
queries were updated together. G1 passes15 native tests; G2 passes the three
focused FP16 arithmetic tests, including both encodings, head shards1..8,
serial/grouped dirty replay and finite-range witnesses. Both preserve every
output and258-word numeric partial byte at the two decode contexts.

| Complete M1/Q6 pair | G4 retained | G1 rejected | G2 rejected |
|---|---:|---:|---:|
|Context512, µs |20.1573 |36.2935 |25.62385 |
|Context65536, µs |59.03335 |121.4355 |80.9233 |

G1's compiler removes all hardware CTA barriers automatically, while preserving
source ordering. This alone does not make the kernel faster. Isolated short-
context counter records show849.5 /1947 /1316 vector instructions per wave for
G4/G1/G2. FetchSize stays520.4375KiB for all three: duplicated native group-copy
work must not be described as tripled external-memory traffic. Native G1
metadata reports81 used VGPRs,18048B LDS and zero private scratch; rocprof's
132 VGPR field describes padded descriptor allocation, not132 live registers.
The SDK rounds LDS allocation to18432B. Detailed dispatch joins remain separate
from these raw counter records; profiler durations are not timing scores.

Neither smaller group is retained. Their complete fixture/core/source archives,
exact numeric exports and native/event/profile receipts remain local under
`attention132-*`; summary: `attention132-groups-summary-r1.json`. Production
source returns to checkpoint128/G4. Decode tuning remains the priority, with
prefill retained as a regression guard. The next comparison profiles the actual
pinned Castagna producer and merge independently through their real launcher.

## Direct decode profiles and partition bottleneck

The isolated133 target calls the pinned fork's actual tile launcher through a
bridge compiled inside its HIP backend. The only fork edit is the opt-in CMake
hook; producer/merge source and launch selection are unchanged. Captured native
functions, shapes, input/scratch pointers and pair edges are checked. Release
uses the same explicit gfx906 SDK/header/bitcode roots and qualified HIP runtime.
No model files are loaded or hashed. The target's real setup pair remains outside
timing, followed by five warmups and12 unprofiled repeated-pair samples.

| Visible context | Llaminar H6 pair, µs | Castagna H12 pair, µs | Llaminar H6×12 estimate, µs | Castagna H12×6 estimate, µs |
|---|---:|---:|---:|---:|
|512 |20.1573 |14.42435 |241.8876 |86.5461 |
|640 |32.4813 |15.22065 |389.7756 |91.3239 |
|768 |32.4638 |15.22130 |389.5656 |91.3278 |

Both observed model layouts perform72 head-layer instances per GPU, but these
are different per-call shapes and original arithmetic. The normalization above
is a repeated-bank component estimate, not a full4GPU attention-stage result.
The640 fork cell has physical768 KV with128 explicitly masked tail entries;
its real G4 dispatch requires256-token physical alignment. Llaminar's fixed
logical view has640 physical entries. Original FP32 coordinate generators match;
each engine performs its own native half conversion, so cross-engine input/output
byte equality is not claimed.

Twenty isolated counter launches authenticate the exact SDK counter callback,
kernel descriptor, dispatch/correlation, graph pair, occurrence and physical
MI50. The fork selects occurrence7 after its real eager setup and five graph
warmups; Llaminar selects6 after five graph warmups. Each profiler process records
one role/counter/shape. Counter durations are excluded from throughput scores.

| Producer fact | Llaminar512 | Castagna512 | Llaminar768 | Castagna768 |
|---|---:|---:|---:|---:|
| Physical partitions |8 |8 |8 |12 |
| Dispatched waves |64 |96 |64 |144 |
| VALU instructions/wave |849.5 |642 |1228 |642 |
| Reported FetchSize, KiB |520.4375 |525.25 |not sampled |not sampled |
| LDS bank conflicts |0 |0 |not sampled |not sampled |
| Used / allocated VGPR |75 /76 |83 /84 |75 /76 |83 /84 |

The512 producer has16 vs24 CTAs; at768 it remains16 while the fork launches36.
Our fixed eight streams process a second64-token tile once512 tokens are crossed;
the fork keeps one tile per physical partition at these contexts. This is a
measured decode scheduling bottleneck. The paired merge is also profiled:
Llaminar reports111 vector instructions/wave and52.0625KiB FetchSize versus
213.5 and96.8125KiB for the fork's twice-as-many output heads. Neither reports
private scratch. Raw per-wave figures must not be compared as equal total work.

The SDK defines VALUInsts as total vector instructions divided by wave count;
FetchSize already sums memory read-request bytes and must not be multiplied by
waves. Combining separately isolated, matched source/shape runs reconstructs
about1.76× and1.70× more producer vector work for Llaminar after the observed
layer multiplicities at512 and768. FetchSize includes cache and extra-fetch
effects; it does not measure source-level gather operations or LDS copy count.

Evidence: `attention133-events-comparison.json`,
`attention133-profile-comparison.json`, `attention133-castagna-build-r1` and the
individual133 native/event/profile receipts. The decoder-focused successor
uses33 logical64-token parts as a new arithmetic program, independently from
future physical scheduling. It first enters a test-only exact-span fixture;
production remains8 until numerical, economy and typed scratch/capture gates
qualify it. Prefill optimization remains parked.

A separate G6 sharing probe134 preserves old8 arithmetic and passes the three
focused native tests. Its M1 pair is19.7606 /31.2469 /58.79005µs at512 /768 /65536;
short/long saved output/partial bytes match G4 exactly. This modest scheduling
candidate is archived, not promoted; it does not remove the partition cliff.

## Fixed logical decode tiles: component qualification

Prototype135 assigns each of the at most 33 selected 64-token tiles its own
arithmetic part. The test-only entry receives an exact span of 33 partials per
row/head; production dispatch, graph planning and prefill still use eight.
This changes the long-context reduction program and therefore requires fresh
model KL qualification before production use. Serial/grouped execution within
the new program must remain byte-identical.

All 18 native tests pass, including both key encodings, every M from 1 to 16,
compact head shards for device degrees 1 through 8, dirty replays, inactive
rows, finite extremes and signed zeros. Eight saved component exports cover
post-RoPE M1 at 512/640/768/2048/65536, M16 at 65536, and pre-RoPE M1 at
512/65536. Subsequent candidates compare every output and numerical partial
byte against those exports; this does not assert old-eight/new-33 equivalence.

| Visible context | Same-build eight-part pair, µs | 33-part pair, µs |
|---|---:|---:|
| 512 | 20.3175 | 25.24625 |
| 640 | 32.65875 | 25.3575 |
| 768 | 32.6475 | 25.48875 |
| 2048 | 57.6038 | 29.28 |
| 65536 | 59.975 | 29.575 |

The long-context cliff is reduced, but short-context merge overhead prevents
promotion. Authenticated isolated profiles show 69 used / 72 allocated VGPRs
for the 33-part reducer, versus 19 / 20 for eight. The 33-part producer launches
264 waves, including empty parts; its 222.606 instructions/wave at 512 must not
be compared with the old producer's 849.5 without accounting for those waves.
All sampled kernels have zero scratch.

Candidate136 bounds only metadata loops to live parts and keeps all 33 weighted
FMAs. It passes all native tests and export comparisons, but is rejected:
512 rises to 27.02605 µs and 65536 to 39.4422 µs. Fewer short-context vector
instructions (238.25 to 194 per reducer wave) do not imply lower latency;
register allocation stays unchanged and the dynamic loops lose performance.
The original reducer is restored.

Candidate137 exposes the producer's proven single iteration while preserving
its literal first recurrence. All 18 native tests and eight export comparisons
pass. Used VGPRs fall from 75 to 48, descriptor allocation from 76 to 68, and
the long-context vector count from 824.424242 to 815.939394 per wave. LDS still
limits residence to three blocks, and event times remain effectively unchanged
(25.1748 / 29.3461 µs at 512 / 65536). It provides register headroom for the
next load-scheduling experiment, not a claimed throughput win.

Evidence: `attention135-parts33-*`, `attention136-live-metadata-events-r1.json`,
`attention137-one-pass-events-r1.json`, and the independently joined
`attention{135,136,137}-sdk-join-r1.json` reports. Each join authenticates the
sixth retained producer/merge transaction, native symbol/geometry, SDK callback,
counter, code-object provenance and physical MI50. No profiled duration enters
the event timing table. Full-model KL, phase-aware scratch admission and actual
TP4 attention-stage qualification remain open for this new program.

Further isolated trials retain the same 33-part numerical exports:

| Candidate and change | M1 / 512 pair, µs | M1 / 65536 pair, µs | M16 / 65536 pair, µs |
|---|---:|---:|---:|
| 139: four independent vector loads exposed per thread | 25.03615 | 29.1674 | 130.6055 |
| 138: dynamic weighted prefix, over producer139 | 23.94115 | 30.30365 | 131.47 |
| 140: identity addresses for fully visible selections, over139 | 23.45365 | 29.39615 | 131.431 |
| 142: cooperative metadata loads, over producer140 | 22.61115 | 28.5787 | 127.682 |

Each trial passes all 18 native tests and all eight output/partial export
comparisons. Candidate138 is removed because its dynamic loop exchanges a
short-context gain for a long-context loss. It preserves at most two trailing
zero-product FMAs, sufficient to reach the finite RN fixed point even with
output-only denormal flushing. Static inspection of copied final code objects
binds both exact reducer descriptors to their function entries: RN arithmetic,
preserved FP32 subnormals, and no mode-changing instructions or indirect calls
in the inspected bodies. Those observations supplement the native byte checks.

Candidate140 validates identity membership before deriving contiguous logical
addresses; sparse selections retain their original selected IDs and ring map.
Candidate142 adds one necessary publication barrier after cooperative max,
denominator and optional observation-counter loads, then retains every original
ordered fold. Its shared allocation is 708 bytes, with 71 used / 72 allocated
VGPRs and zero scratch. The old-eight specialization still uses its original
metadata path. These trials do not introduce a new model default. In142,
the same-build old-eight pair remains faster at512 (20.3262 µs), so short decode
still needs work despite the long-context improvement over old-eight's60.0299 µs.

Artifacts: `attention{138,139,140,142}-*-events-r1.json`, the corresponding
native/profile receipts and SDK joins, and
`attention138-weighted-prefix-fp-mode-r2`. The142 runtime is retained for paired
follow-ups. Structural fixed-bound folds are the next independent merge trial.

Candidate141's four fixed fold bounds recover the short pair (20.24495 µs)
but regress65536 to34.55615 µs. Exact disassembly shows both long loops already
unrolled; the new kernel repeats coordinate shifts and grows from9444 to25664
code bytes. This supports investigating generated addressing and code layout,
not assuming a missing-unroll problem.

Combining the fixed folds with cooperative metadata in143 retains every saved
numerical byte and recovers the long-context performance. M1 pairs at
512/640/768/2048/65536 are19.0698 /20.34605 /20.4998 /26.4635 /28.8222 µs;
the same-build old-eight controls are20.30105 /32.6559 /32.63345 /57.63825 /
60.0845 µs. M16/65536 is127.91 versus138.251 µs. The reducer reports71 used /
72 allocated VGPRs,708 bytes static LDS and zero scratch. These are component
gains, still short of a Castagna win and not yet model-KL qualification.

All **19 native tests** pass, including the new explicitly registered
`FP16Virtual33EmptyTailPreservesSignedZero`. It authors an exactly representable
score of−103 and a half numerator that makes the final weighted FMA underflow
to−0. A real old-eight control proves that sensitivity on the native device.
The 33-part program must produce+0 when empty tail parts remain, and−0 when
the final33rd part is live. The test covers512/1024/2048/2051 selected tokens,
M1/M16, identity/sparse addressing and independent captured serial replays.

Evidence: `attention141-fixed-fold-*`, `attention143-fixed-cooperative-*`,
their SDK joins, and the retained143 runtime.

The two-bound merge145 preserves all bytes and passes19 native tests, but loses
about12% at640/768 (22.86105 /23.05105 µs) without a meaningful long-context
gain. It is rejected and the four-bound143 merge restored. Full-panel copy
specialization144 also passes19 native tests and all eight export comparisons.
It removes count guards only when a complete64-key or32-value panel is live;
partial panels, signed-zero-sensitive PV guards and the old-eight program stay
unchanged. Its pairs are19.2699 /20.5774 /20.73115 /26.7974 /28.6149 µs at
512/640/768/2048/65536, with M16/65536 at127.3595 µs: effectively flat overall,
not an additional qualified throughput win.

Exact producer profiles for144 report190.909091 /701.515152 vector
instructions per wave at512 /65536, compared with204.727273 /756.787879 for140.
Used / allocated VGPRs are53 /68, static LDS19968 bytes, and private scratch0.
Fewer instructions again do not establish a latency gain by themselves. The
sixth captured transaction, native function, geometry, SDK callback, code
object and MI50 identity are joined in `attention144-full-panel-profile-joined-r1.json`.
The unchanged33 logical program is next tested with six-head cooperative groups;
that changes physical copy ownership, not numerical partial layout or order.

Candidate146 uses one six-wave block per canonical six-head group. All19 native
tests and eight numerical export comparisons pass; metadata explicitly admits
only the physical group-width change4→6. M1 pairs at512/640/768/2048/65536 are
18.7737 /20.0549 /20.2128 /25.2349 /26.98865 µs, and M16/65536 is109.726 µs.
The native occupancy query now uses the actual captured block size (384 for
this producer), while independent fixture assertions still require the declared
geometry. CUDA's observation adapter receives the same actual-block correction.
Exact profiles report52 used/allocated VGPRs,21248 bytes static LDS, zero private
scratch and zero LDS bank conflicts. The198-wave producer's vector counts are
214.010101 /784.919192 per wave at512 /65536; these averages use fewer waves
than G4 and are not direct total-work ratios. Reported512 FetchSize remains
520.4375 KiB, so fewer source-level copy owners do not establish fewer HBM bytes.

Candidate147 adds a shared full-tile PV specialization: only an explicitly
admitted33-part tile with64 live tokens can omit the original per-token guard.
Partial tiles and Parts8 retain it; half-FMA order and signed-zero boundaries
stay unchanged. All19 native tests and eight byte comparisons pass. Pairs fall
to17.2374 /18.50865 /18.6474 /23.6874 /25.43365 µs at512/640/768/2048/65536,
with M16/65536 at101.9045 µs. Used/allocated registers and LDS stay52/52 and
21248 bytes, with zero scratch; vector counts fall to190.979798 /692.797980
per wave. Evidence is in `attention{146,147}-*-events-r1.json` and the exact
SDK/profile joins. Both remain component experiments, without new model KL or
a Castagna performance win. The next QK scheduling candidates preserve literal
dot order while measuring whether the reduced register footprint permits wider
load scheduling.

Both148 QK schedules pass19 native tests and all eight numerical exports.
Unroll64 wins over32 in every measured pair:512 /65536 /M16-65536 are
17.03865 /25.2574 /101.1335 µs, versus17.3174 /25.54365 /101.7245 µs for32.
The original production8 still uses unroll16. Exact profiles give51 used /
52 allocated VGPRs for64, versus54 /56 for32; both have21248 bytes LDS and
zero scratch. The64 vector counts are186.373737 /674.373737 per wave at
512 /65536, below32's189.040404 /685.040404. This is a small component gain;
Castagna remains ahead at short context and full-model qualification is open.

The proposed149 one-wave index validation is rejected before implementation.
Other waves could read invalid sparse indices before the existing post-copy
barrier. Keeping every required pre-load validation edge takes precedence over
removing that duplicated work. No native result or performance claim exists
for149.

Candidate150 parallelizes the merge's per-part scale calculation while retaining
the exact ascending maximum and denominator FMA order. A logical32-lane group
computes scales, broadcasts each original ordered contribution, and handles the
33rd term explicitly. It adds no reduction tree or barrier; the weighted fold,
signed-zero-sensitive empty tail and original eight-part program are unchanged.
All19 native tests and eight saved numerical output/partial comparisons pass.

| Unprofiled captured pair, microseconds | 512 | 640 | 768 | 2048 | 65536 | M16 /65536 |
|---|---:|---:|---:|---:|---:|---:|
| Retained148 unroll64 | 17.03865 | 18.3349 | 18.49365 | 23.51615 | 25.2574 | 101.1335 |
| Parallel scales150 | 16.53615 | 17.24745 | 17.28365 | 20.29615 | 22.04615 | 97.60345 |
| Same-build original eight parts | 20.19115 | 32.3949 | 32.38115 | 57.10845 | 59.4473 | 137.4435 |

Exact isolated merge profiles report70 used /72 allocated VGPRs,708 bytes LDS,
zero private scratch, zero LDS bank conflicts at64K, and95.25 /350.25 vector
instructions per wave at512 /64K. The64K memory-unit-busy counter is7.938428%.
Each counter joins the sixth retained transaction, native captured function,
SDK symbol/code object and physical MI50. Profiler durations are excluded from
timing scores. The producer retains148's51 used /52 allocated VGPRs and21248
bytes LDS. Evidence: `attention150-parallel-scales-events-r1.json` and
`attention150-parallel-scales-joined-profile-r1.json`; source/runtime150 is frozen.

Castagna's pinned short-context pairs remain14.42435 /15.22065 /15.22130 µs
at512 /640 /768. These are different local head geometries, not an equivalent
request-latency comparison.150 has not beaten that component baseline. The typed
experimental schedule now has focused model qualification: eight parts
for the original Prefill phase,33 for ordinary or retained verifier Decode.
Its shared scratch uses `heads * max(8 * resident_rows, 33 * max(1, retained_rows))`
records; Uniform retains its original byte count. The production default remains
FP32; the measured FP16 schedule is an explicit experiment.

Integration151 now routes33 through the ordinary ROCm dispatcher and retires
the test-only launcher. Its setup-only opt-in is
`LLAMINAR_QFN_SPARSE_ATTENTION=fp16_tile` with
`LLAMINAR_QFN_SPARSE_ATTENTION_PHASE_SCHEDULE=prefill8_decode33`.
The schedule is snapshotted once into runtime configuration, copied through
admission and wire7, and resolved from the original typed execution phase.
Unsupported backends or Decode rows beyond the admitted retained family reject
before graph publication. Uniform keeps its previous scratch byte count.

The complete Unit namespace and affected native/admission registrations pass
**832/832**, with a clean driver window and unchanged source during execution
(`attention151-unit-native-r1`). This includes exact estimator/allocator deltas,
phase/program capture identity, immutable setup propagation, wire validation,
and prefix identity. The initial build's three test-only compile defects were
corrected without removing assertions; the successful rebuild is retained as
`attention151-phase-integration-build-r2`. Full production preflight is not
refreshed by this focused run.

Review also found that the older Uniform FP16 experiment lacked a distinct base
attention prefix identity. The existing model metadata hook now publishes the
graph's frozen base arithmetic, preserving parent MoE fields and historical
FP32 identity. Actual graph-hook and pure-field regressions prove that later
debug reloads cannot relabel cached state. No payload hashing, planner ledger,
new live setting read or additional transfer was introduced.

### Phase schedule151: focused model qualification

The Release build passes the spill guards for CUDA80/86/89/90 and HIPgfx906.
All19 native functional tests pass through the ordinary production dispatcher;
all eight output/partial exports remain byte-identical to the original33-part
program, including both key encodings, short boundaries,64K and groupedM16.
The native pair measures16.503µs at512, below the old-eight20.198µs control,
but still above Castagna's14.424µs at its different local head geometry.

The canonical required QFN groups pass with exact reset replay, unchanged
source/runtime, clean drivers and normal retirement. Actual phase/program
counters are verified on all four devices in this accuracy run. Against
uniform8 checkpoint128,176 of185 final-logit rows are byte-identical; the only
changed rows are the nine boundary-diagnostic decode observations. That
diagnostic is not a calibrated required acceptance group.

| Panel / phase | Rows | Uniform8 mean KL | Prefill8/Decode33 mean KL | Split top-1 | Split top-5 overlap |
|---|---:|---:|---:|---:|---:|
| Main prefill | 16 | 0.025613694 | 0.025613694 | 93.75% | 92.50% |
| Main decode | 96 | 0.028276396 | 0.028276396 | 95.83% | 92.29% |
| Held-out prefill | 8 | 0.014910553 | 0.014910553 | 100% | 95.00% |
| Held-out decode | 48 | 0.004552875 | 0.004552875 | 100% | 91.67% |
| Boundary diagnostic prefill | 1 | 0.000168082 | 0.000168082 | 100% | 100% |
| Boundary diagnostic decode | 9 | 0.000367672 | 0.000209866 | 100% | 93.33% |

Matched unprofiled Release A/B/A uses the same binary: fullTP4 on fourMI50s,
`TensorParallel + GateUpOwnedDownColumns`, sharded vocabulary, RAM ngrams,
28 physical NUMA1 workers,512 prompt tokens and256 emitted tokens, context4096,
native FP16 KV, MTP/prefix cache off. Each arm has one warmup and three measured
iterations. PerfStats is disabled for these timings; the same-binary accuracy
run above supplies actual phase/program proof. All nine generated256-token
sequences match, and source/runtime/driver/retirement checks pass.

| Throughput, tok/s | Uniform8, six control samples | Prefill8/Decode33, three samples | Change |
|---|---:|---:|---:|
| Prefill median | 818.788 | 818.579 | -0.026% |
| Decode median | 39.884 | 40.110 | +0.566% |
| Decode range | 39.741–39.971 | 40.052–40.124 | Nonoverlapping |

The closing control drifted lower; this is a modest descriptive gain, not a
general confidence-bound claim. Castagna's42.959tok/s target remains open.
Evidence: `attention151-phase-model-fp16-r1`,
`attention151-phase-mean-metrics-r1.json`,
`attention151-phase-model-throughput-comparison-r2.json`,
`attention151-phase-native-functional-r1` and
`attention151-phase-native-events-r1.json`. The first benchmark wrapper rejected
an omitted outer CPU-affinity binding before launching inference; its failure
is retained, and the successful retry uses the unchanged workload and required
CPU mask. This is a focused checkpoint, not a refreshed complete preflight.

The per-GPU normalization above is a **work comparison**, not a request-latency
identity. The fork rounds attention tensor slices to whole 12-query-head GQA
groups and rotates ownership by attention-layer ordinal. Each attention layer
therefore has two active GPUs, whereas Llaminar has four six-head participants.
All-device interval unions and authenticated per-layer spans must remain
separate from per-GPU sums/medians when assessing the actual stage latency.

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
