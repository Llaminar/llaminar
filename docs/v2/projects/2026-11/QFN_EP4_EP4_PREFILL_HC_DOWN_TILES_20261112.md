# QFN HC-down prefill tiles on MI50 and 7900 XTX

## Result and scope

The installed M512 HC-down dispatch improves matched eight-MI50 full-model
prefill from **789.433 to 793.724 tokens/s (+0.544%)**. Together with the
[publication repair](QFN_EP4_EP4_PREFILL_PUBLICATION_TUNING_20261112.md), this
pass improves the original 734.870 tokens/s by **8.009%**. Component gains are
larger than the model gain because this projection occupies only part of prefill.

Production changes are two generated exact-table entries, preserving all 580
previous entries and the existing generic dispatch. Both select existing native
Q8 kernels for M512/N320/K10240, MB1/U2, checked loads:

| Actual architecture | Previous tile | New tile | Captured corpus speedup |
|---|---|---|---:|
| gfx906 | NT128/MT32 | NT64/MT16 |1.11045×|
| gfx1100 | NT128/MT32 | NT64/MT32 |1.13032×|

The physical execution codebook is 19, with independent original Q8_0, Q8_1 and
Q8_K source evidence. No source is relabeled or requantized for the experiment.
The corpus speedup is the existing emitter's geometric mean across source
aliases against actual Auto. It is not an end-to-end throughput claim for the
7900 XTX. Other M/N/K keys keep their existing policies; architecture proximity
does not select a nearby row count.

## Full-model comparison

The workload and lifecycle owners are exactly the prior publication comparison:
original four-shard Qwen3.8-Flash-Next-UD-Q4_K_XL; PP(EP(4×MI50), EP(4×MI50));
fixed-head ngrams sharded in VRAM across all eight cards; Static/Ordinal expert
placement; MTP and prefix cache off. Discovery rank 1 owns both stages on NUMA1,
with stage0 devices7/6/8/9 and tail1/3/4/2. Original UUIDs, PCI ancestry, rankfile,
model bytes and arithmetic settings remain authenticated by the same owners.

Release, exact raw `" test" * 512` prompt, context4096, M512 capture, greedy
sampling, seed42, one warmup and three measured requests. Each emits one token,
with zero after-prefill decode steps. No profiler, stage clocks or competing
build/native workload overlaps the model clocks. The public benchmark's prefill
phase ends at durable inference completion, before first-token emission.

| Observation | Publication checkpoint, q163 | New tiles, q183 |
|---|---:|---:|
| Request1 prefill, ms |649.368066|646.835414|
| Request2 prefill, ms |648.566754|644.336694|
| Request3 prefill, ms |647.960964|645.060744|
| Median prefill, tokens/s |789.433003|793.723699|
| First output ID, every request |1228|1228|

All eight owners' terminal ngram calls, live IDs, native payload bytes and
retirement tuples match the baseline. Drivers remain clean, native processes
and descriptors retire, and every selected GPU returns to10,866,688B idle VRAM.
The two processes' separately observed VmHWM values are750,284,800B and
21,852,577,792B, with no swap. Twelve whole-lifetime samples span55.34s;
sampled VRAM peaks range20.914–21.326GB/card. These sparse samples are not
phase-specific peaks or a long-duration leak certificate.

This is a focused prefill diagnostic, not a full preflight/release certificate
or a new long-context/decode result. The previous512/256 decode measurement
remains37.260 tokens/s. Numerical qualification here is exact kernel-byte
equivalence and the matched model observations; no KL threshold changes.

## Why the tiles help

The retained full-model trace contains48 HC-down calls per stage0 participant
and49 per tail participant, including the final HC mixer:388 calls total.
They occupy20.142–20.547ms/GPU with no overlap against another selected kernel
on the same GPU. These costs are not added across devices to invent wall time.

NT128 covers384 columns for live N320. Its last column tile takes the checked
boundary-load loop for all160 K64 iterations. NT64 divides320 exactly and uses
the existing safe-load loop while retaining the checked template and identical
ordered K32 arithmetic. M512 divides each tested row tile.

| gfx906 resource | NT128/MT32 | NT64/MT16 | NT64/MT32 control |
|---|---:|---:|---:|
| CTAs,256 threads each |48|160|80|
| Raw compiler VGPR |114|71|83|
| LDS, bytes |21,504|10,752|12,800|
| Private bytes / spills |0 /0|0 /0|0 /0|

The smaller MI50 tile offers more independent blocks with less per-thread
state. It also revisits weight tiles more often, so the benefit is not universal:
gfx1100 wins with32-row weight reuse. Its raw compiler VGPR counts are114/70/83
for the same three variants, with identical LDS sizes and zero private/spills.
All kernels remain wave64. No new device kernel or arithmetic mode is added.

Separate sealed ROCprof runs cover winner and Auto for all three sources on
both architectures. On gfx906, launched waves192→640 match the grid change;
reported VALUBusy rises approximately31–37%→46–49%, and L2 hit88.12%→93.31–93.34%.
Reported fetch size is nearly unchanged despite additional tile-level weight
visits. Achieved occupancy is unavailable; GPUBusy100% is not full utilization.
LDS conflict rises rather than falling, so this is not a barrier/bank-conflict
elimination claim. Each entry retains its existing barriers.

gfx1100 confirms192→320 launched waves and resource reductions, but several
utilization/traffic counters are unavailable or return unusable zeros. Its
isolated bank-conflict readings are inconsistent. Those fields do not support
an efficiency claim. Profiler durations never replace native event clocks.

## Timing regimes and regressions

The exact corpus uses five warmups and30 event samples, with16 captured
complete projections per sample. Every candidate passes original-source byte
and resource checks. Separate31-pair AB/BA tests time one retained complete
projection per event interval. The independent forced NT128/MT32 baseline
survives Auto promotion; NT64/MT32 is also retained as a control.

The single-projection events confirm approximately1.096–1.105× on MI50. On
gfx1100 they show about1.031× for Q8_1/Q8_K and1.001× for Q8_0. Their absolute
clocks are much higher than the16-projection corpus. Both regimes are retained;
neither is relabeled as the other, and the cause of that difference is not
inferred from these observations. The full-model MI50 run decides its gain.

Focused qualification:

- Both Integration and Release rebuilds pass with the configured shipping
  targets CUDA80/86/89/90 and ROCm gfx906/gfx1100. The existing compiled kernels
  retain mandatory spill proof; the dispatch change creates no new GPU body.
- Twelve focused CTests pass: complete dense-prefill policy and evidence owners,
  their explicit preflight aliases, four existing tile regressions and three
  new M512 source-specific byte/reset aliases.
- New `V2_Integration_ROCmNativeQ8N320M512K10240TileByteResetQ8_{0,1,K}` cases
  are explicitly registered in ProductionTestPreflight. They use captured
  production stages, two input histories,20 dirty/reset/large-to-small replays,
  independent baseline output, inactive-row guards and unchanged source bytes.
- Separate rebuilt Release checks on native ordinals1 and5 prove all six actual
  M512 Auto captures hit the measured entries. Their captured function pointers
  match the corresponding forced controls, and complete bytes match the old tile.
- `NativeQ8MeasuredM512SelectionAndActualRowIsolation` checks both installed
  architecture tuples and unchanged physical metadata. Neighbor-row absence is
  checked against an authored table, allowing future independent measurements.

## Reproduction and next work

The existing `production_dense_prefill_sweep` scope is three sources, native
Q8 family, `Qwen4Exp_320x10240`, M512, architecture gfx906 or gfx1100,5 warmups,
30 samples. Each architecture owns a separate corpus. The canonical profiler
and additive emitter validate winner/Auto evidence before composition580→581→582;
only the final two-architecture table is installed. No generic fit or unrelated
resweep is introduced.

Local evidence, excluded from the source repository:

- `/tmp/qfn-m512-hc-down-native-q8-exact-20261112`: MI50 corpus, six sealed
  profiles, winner summary and first retention receipt.
- `/tmp/qfn-m512-hc-down-native-q8-gfx1100-exact-20261112`: independent companion,
  profiles, final generated include and second retention receipt.
- `/tmp/qfn-m512-hc-down-paired-20261112`: alternating events and installed-Auto
  validation; q170–q182 receipts are in `/tmp/qfn-full-preflight-after-cpu-next`.
- `/tmp/qfn-m512-hc-down-q8-isa-audit-20261112`: exact ISA/resource extraction
  and separate counter reductions, including explicit unavailable fields.
- `/tmp/qfn-ep4ep4-prefill-tile-20261112`: q183 model samples, immutable launch
  inputs, source delta, memory observations and driver/retirement receipts.

The next larger candidate is the separate HC two-term **up** projection,
M512/N10240/K320. Retained timing is34.3–35.1ms/GPU; its already-fused publisher
is only about0.5ms/GPU. A larger cooperative row tile could reuse weights across
more rows while preserving both ordered term folds. This is a hypothesis,
not an implemented or measured win; register pressure and full-chain byte
equivalence must be qualified before promotion.
