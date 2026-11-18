# MI50 QFN decode kernel tuning

Status dashboard for 15 physical decoder specialisations. Limit 15,000 UTF-8 bytes.

## Full-model throughput on four MI50s

Both Release engines used the same four model shards, the same four cards
behind one switch, an empty prompt cache, and the same exact input IDs.
Each cell has three measured generations of 256 tokens following a warmup.
Decode measures the 255 tokens after the first.

| Mode and engine | Review prefill / decode tok/s | Code prefill / decode tok/s | Matches own ordinary history |
| --- | ---: | ---: | --- |
| Llaminar, MTP off | 1082.65 / 88.71 | 1051.46 / 82.49 | 3 identical repeats per prompt |
| Castagna, MTP off | 870.45 / 85.59 | 1003.97 / 75.03 | 3 identical repeats per prompt |
| Llaminar, fixed 3 | 1042.02 / 134.08 | 1009.10 / 132.14 | 6/6 |
| Castagna, fixed 3 | 785.74 / 117.62 | 892.87 / 108.39 | 0/6 |
| Llaminar, dynamic (1–15) | 1040.70 / 120.95 | 1009.24 / 134.75 | 6/6 |
| Castagna, adaptive (1–15) | 672.70 / 70.71 | 794.71 / 79.81 | 0/6 |

Llaminar’s equal-width experiment placed 160 routed and 160 shared columns on
each GPU across all 48 layers, with retained ordinary HC/FFN and a counted
four-row FFN. Both frozen-history accuracy panels passed. Equal-width mean
KL was 0.03150 prefill and 0.01928 decode; the original split measured
0.01432 and 0.02211. Regrouping the FP32 partials changed the generated
histories. Fixed-depth code used 73 verification rounds with equal widths
versus 86 with the original split. On review those counts were 76 and 74.
Throughput describes each engine's actual completed work.

The shipping Castagna local HC/FFN kernel took 156.27 µs for four rows on
the wide layer-1 slice. Our matched full body took 256.42 µs. Bypassing
route grouping helped three rows but cost 5.2% at four. Routed gate/up took
77.36 versus 27.12 µs on precisely matched expert IDs. Equal-width code2048
reduced the full profiled verifier from 26.39 to 22.94 ms per round.
Castagna measured 18.22 ms. Each profiler run matched its unprofiled
control's tokens and work counts.
[Phase measurements, run identities and commands](QFN_MI50_STAGE_COMPARISON_20261118.md)

The counted four-row FFN can reuse exact K32 activation sums on the two
measured affine format combinations that benefit. On the same MI50 and
Release ELF, Q4_K/Q5_1 measured 150.23 → 135.36 µs, Q5_K/Q8_0
150.14 → 145.89 µs, and Q4_K/Q8_0 kept its original path (135.02 →
135.12 µs in that bracket). These labels are
routed gate/up and down; the immutable shared expert used Q8_0 in all three.
The four-MI50 full-model fixed-depth gate gained 2.35% and 2.65%
on review512 and code2048. Dynamic MTP changed by less than 0.1%.
On code2048, the matched fixed-depth verifier dispatch union fell from
25.08 to 23.45 ms/round. The full four-card span fell from 26.64 to
25.04 ms. The dominant gate/down leaves reached 59%/62% of the same-card
geometry-matched read probe. Broader source formats remain under tuning.
[Matched runs and read calibration](QFN_MI50_STAGE_COMPARISON_20261118.md#counted-verifier-format-qualification)

The counted four-row reader passed static and deferred replay across all
15 generated format families. Median local time improved 9.23% over staged.
On original 128- and 256-column QFN shards, paired gate/up reads saved
10%–13% on Q4_K; Q5_K was 4.5%–7.9% slower, so its reads stay separate.
On original TP4 shards, the checkpointed Release build reached 130.72 tok/s on
review512 and 107.28 on code2048 at fixed depth three. At dynamic depth
it reached 116.08 and 94.74. Four-row dedicated tiles now bring the wide
Q4 gate to 87.5% of its matched physical-read reference. The complete
four-row router/FFN fell from 178.30 to 151.10 µs on the wide TP4 shard
and from 119.20 to 114.12 µs on the narrow shard. The smallest TP8
shard took 0.42 µs longer. Nine focused preflight tests passed.
The counted six-row FFN reduced the captured local router/FFN time by
35.5% on the narrow shard and 25.0% on the wide shard.
On four MI50s the six-row change alone raised dynamic decode from 115.92 to
126.91 tok/s on review and 94.43 to 103.81 on code, with matching tokens
and verifier work. The combined Release passed 13 focused preflight cases.
At four rows and F256/S256, all 15 generated format families improved,
with a 22.01% median reduction in local time. Full-model timing is in progress.
[Six-row model trial](QFN_MI50_STAGE_COMPARISON_20261118.md#six-row-dynamic-verifier-on-the-original-tp4-shards)
[Full-model measurements](QFN_MI50_STAGE_COMPARISON_20261118.md#transferring-the-single-row-loop-to-grouped-verification)
[Verifier tiles and packet scheduling](QFN_MI50_STAGE_COMPARISON_20261118.md#verifier-tile-assignment-and-packet-scheduling)

The current ordinary decode trace covers all 255 post-prefill review tokens.
Values are mean active milliseconds per GPU per token. The projection group
includes the fused GDN input projection.

| Stage | Llaminar | Castagna |
| --- | ---: | ---: |
| Hyperconnections, projections, router and experts | 8.219 | 6.747 |
| Collectives and ordered sum | 1.969 | 2.439 |
| Attention, indexer and RoPE | 0.650 | 0.204 |
| GDN and convolution state | 0.745 | 0.600 |

A second Release compared the original gather and ascending FP32 fold with a
direct peer push that preserves the same sum order. The only change between
arms was the transport selector. All six 256-token outputs matched. Each
ordinary graph lost 97 native nodes per card.

| Prompt | Gather/fold decode | Direct peer decode | Castagna decode | Direct peer prefill |
| --- | ---: | ---: | ---: | ---: |
| Review 512 | 68.28 | 78.30 | 86.17 | 1038.48 |
| Code 2048 | 64.68 | 74.05 | 75.93 | 1009.12 |

Direct peer saved 1.87 and 1.96 ms per generated token. The isolated 10 KiB
reduction took 12.15 µs against 26.81 µs for gather and fold.

Native partial, empty, abort and same-process owner-replacement replays passed.
Two sequential model owners reused one real communicator and each released its
host and device reservations. The final lifecycle build reproduced all six
histories at 78.23 and 74.04 decode tok/s. Peer push remains opt-in. Wider deployment needs
native coverage of additional cohorts and their serving lifecycles.

An independent publication fusion removed 98 decode nodes per device and
matched six ordinary histories. It regressed review from 67.69 to 66.17 tok/s
and code from 64.39 to 62.69 tok/s. The publication fusion stays out of the selected policy.

The review fixed-3 trace reproduced all 256 output IDs and its unprofiled
222 draft steps, 182 accepts and 74 verifier calls. Each MI50 ran a 2,392-node
verifier graph 74 times (296 target rows). It accounted for 88.4–88.6% of
post-prefill summed kernel time, measured separately on each card. On the
first card routed expert projection/activation took 6.80 ms per verifier
call, other projections 6.32 ms and HC normalization/residual 5.96 ms.
The 320-wide Q8 residual projection alone took 3.54 ms across 97 dispatches
per verifier call. The trace used the original gather/fold transport.

## Real-model shapes

The matched Q4_K/Q5_1 operation at F256/S192 passed in three fresh Release
processes on PCI97. Llaminar measured **69.62–69.72 µs**, Castagna **75.09–75.10 µs**.
The composed Release reconfirmed the checkpoint at 69.58, 69.48 and 69.52 µs
versus 75.06, 75.11 and 75.07 µs. Both timings end before the collective. The linked checkpoint specifies the
operator boundary, arithmetic tests, model, external revision and run command.
[Checkpoint](QFN_MI50_RETAINED_HC_FFN.md)

The ordinary TP4 policy splits shared width 640 evenly. In the follow-up build,
six local M1 cases used actual model weights and the same 20-history staged
byte oracle. The four-member TP4 movement replay reused its captures and
verified new expert bytes, changed outputs and exact collective extents.
The table shows retained medians in microseconds on PCI91, with S160 on each
participant. The large routed shard rotates across layers.

| Routed gate/up | Routed down | Layers | F128 | F256 |
| --- | --- | ---: | ---: | ---: |
| Q4_K | Q5_1 | 43 | 61.45 | 69.09 |
| Q4_K | Q8_0 | 4 | 63.17 | 72.87 |
| Q5_K | Q8_0 | 1 | 66.98 | 75.91 |

The Q5_K F256 control took 96.09 µs. Its gate/up workers now share the
prepared affine coefficients. The unchanged fork requires shared widths
divisible by 64. Compare each engine's own sharding in whole-model runs.
In a source-matched TP2 comparison, the common Q4_K/Q5_1 layer took
73.02 vs 76.05 µs (F256/S320) and 80.55 vs 80.91 µs (F384/S320).
The Q8-down follow-up reduced Q5_K/Q8_0 to 74.03 vs 74.77 µs and
84.37 vs 80.81 µs at those widths. Its TP4 window F256/S192 took
71.79 vs 73.79 µs. Q4_K/Q8_0 measured 71.15 vs 71.04, 80.59 vs 78.56
and 69.62 vs 69.94 µs at those same three widths. F384 remains behind for
both Q8-down tuples.
These timings are Llaminar versus the unchanged fork. All 20 histories
selected the same experts for these two layers. On Q4_K/Q8_0 layer 4 the
routers selected nine common experts out of ten on history 16. The Llaminar
stage-byte oracle passed on all histories. The separate timing schedule uses
four inputs fixed before either router runs.

These real-model format variants all use the affine/byte decoder group. The
other families below exercise additional user-selectable weight formats.

## Independent format measurements

All 15 families passed both native publication modes on the final Release
binary. Each mode replayed 20 histories against the original staged bytes,
including reset, empty work and the next active request. The two Q8-down
variants also passed grouped decode at capacities 3 and 4 and six-GPU
expert movement across three two-member RCCL domains.

Each row uses H2560/F256/S256/E512/top-10 and one active decode row. The
staged graph and fused graph receive the same source encodings and the same
publication contract. Each arm retains all 64 observations from 32 balanced
quartets. Ordinary model GGUFs need to be able to encode their own complete
rows. Generated format coverage is reported separately from measured model
layer widths. A source-compatible external fork kernel is required for a
direct speed claim against Castagna.

| Group | Routed gate/up | Routed down | Shared | Previous fused µs¹ | Current control µs | Candidate µs |
| --- | --- | --- | --- | ---: | ---: | ---: |
| G01 | Q4_K | Q5_1 | Q8_0 | 157.56 | 132.10 | see real weights above |
| G02 | Q4_0 | Q4_0 | Q4_0 | 152.03 | 125.48 | keep control |
| G03 | Q4_0 | Q4_0 | Q8_0 | 157.64 | 132.64 | 130.89 |
| G04 | IQ4_NL | IQ4_NL | IQ4_NL | 156.95 | 128.50 | keep control |
| G05 | IQ4_NL | IQ4_NL | Q8_0 | 158.85 | 134.08 | keep control |
| G06 | IQ4_NL | Q4_0 | Q4_0 | 156.07 | 127.93 | keep control |
| G07 | IQ4_NL | Q4_0 | Q8_0 | 160.21 | 129.66 | keep control |
| G08 | IQ2_S | IQ2_S | IQ2_S | 190.03 | 160.10 | 153.56 |
| G09 | IQ2_S | IQ2_S | Q8_0 | 182.30 | 153.08 | 145.72 |
| G10 | IQ2_S | Q4_0 | Q4_0 | 170.92 | 144.92 | 132.21 |
| G11 | IQ2_S | Q6_K | Q8_0 | 181.20 | 150.10 | 138.20 |
| G12 | IQ3_S | IQ2_S | IQ4_NL | 182.36 | 153.35 | 148.27 |
| G13 | IQ3_XXS | Q5_1 | IQ2_S | 192.59 | 163.26 | 157.84 |
| G14 | IQ3_S | Q6_K | IQ2_S | 198.54 | 167.35 | 163.21 |
| G15 | IQ3_XXS | Q6_K | IQ2_S, Q8_0 down² | 192.91 | 156.60–156.71 | 149.33 |

¹ Earlier common S256 Release build on PCI49. Current control and tuned
columns must use the same card, source formats, extent, output contract and
profiling-free timing procedure within each row. Keep a measured control
when a tested candidate loses. G01 used PCI97, G02–G07 used PCI9a, G08–G11 used PCI8c,
G12–G14 used PCI97 and G15 used PCIc5. The IQ2 candidate combines
workgroup-local lookup with whole-wave down projection. The IQ3 candidate
reuses workgroup-local lookup across gate and up. The table gives unprofiled medians of the selected builds. The bounded
IQ2/IQ3 layout preserves the existing coefficient panel and uses complete
tables that fit the remaining space. Sharing gate/up activation reads helped
G03 in two independent runs. Applying it to G04–G06 regressed their
staged-adjusted time by 1.40, 1.51 and 3.25 µs respectively.
G07 showed why the scope matters: compiling the IQ3 changes into its
IQ4 path cost 5.10 µs after adjustment for the paired staged control.

² The historical G15 recipe relied on the expanded INT8 format reachable from
a CPU expert tier. A GPU-only roster with those same three encodings selects
G14. The current G15 fixture supplies a genuine Q8_0 shared-down matrix to
exercise all four decoder groups on GPU.

The per-family memory reference uses actual fetched bytes and the issuing
card's sustained streaming bandwidth. The vector-issue reference uses the
observed vector instructions and same-card DOT4 calibration at the actual
launch geometry. For G02–G07 the two references account for about 19–20%
and 25–27% of elapsed time respectively. Each models an independent resource.
[Per-family measurements and formulas](QFN_MI50_EMPIRICAL_COMPUTE_MEMORY_REFERENCE.md)

The real four-card ordinary decode graph contains all 192 expected
layer/participant bindings. Each device captured 192 retained nodes across
the graph variants. The server completed three measured 32-token generations
after warmup. The same capture census found no retained M1 nodes in prefill
or verifier graphs. A short-K projection and branch-mix fusion removed 49
nodes per ordinary decode graph. The matched six histories stayed exact.
Review improved from 68.03 to 68.22 tok/s, code from 64.67 to 64.89. The
fusion remains experimental. The wider TP2 Q8-down shard remains a separate tuning target.
