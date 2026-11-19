# MI50 QFN decode kernel tuning

Status dashboard for 15 physical decoder specialisations. Limit 15,000 UTF-8 bytes.

## Shipping RCCL, four MI50s, automatic planning

On code2048 the automatic serving defaults measured **918.20 prefill / 79.84
decode tok/s**. Dynamic MTP and residency used one fully resident tensor-sharded
tier. Both generations matched the maintenance-off control (**920.03 / 80.37**)
in tokens and verifier work, with zero expert transfers. The earlier
GateUpOwnedDownColumns control measured **939.02 / 64.48**.
[Settings and results](QFN_MI50_STAGE_COMPARISON_20261118.md#serving-defaults-with-dynamic-residency-enabled)

## Earlier measurements on the original four-MI50 split

Checkpoint `7001788d8` used routed expert widths of 128/128/128/256,
160 shared columns per GPU and the measured verifier switches. Both engines
ran the same input IDs on the same four cards. Entries are **prefill / decode
tokens per second**, medians of three 256-token generations after warmup.

| Mode | Review 512, Llaminar | Review 512, Castagna | Code 2048, Llaminar | Code 2048, Castagna |
| --- | ---: | ---: | ---: | ---: |
| Off | 1040.78 / 86.80 | 874.62 / 85.76 | 1008.53 / 81.16 | 922.66 / 73.64 |
| Fixed 3 | 1001.09 / 137.38 | 785.74 / 117.62 | 970.47 / 111.88 | 892.87 / 108.39 |
| Dynamic / adaptive | 999.25 / 131.68 | 672.70 / 70.71 | 970.17 / 107.59 | 794.71 / 79.81 |

All 18 Llaminar histories matched the qualified control. Both MTP modes
also matched their own ordinary decode and all five verifier-work counters.
Castagna generated different continuations and its throughput includes its
own draft acceptance. The largest remaining fixed-depth stage deficit on
code2048 is 3.08 ms/round in HC, projections, routing and experts.
On code2048 the HC-down reducer reached 113.54 tok/s with identical work.
The 256-column shard finished last in 4,117/4,128 FFN comparisons. Tuning
the 128-column selector and wide Gate together gave 111.95 → 111.80 tok/s.
[Matched workload, numerical checks and phase timings](QFN_MI50_STAGE_COMPARISON_20261118.md#fixed-depth-verifier-after-the-counted-routing-and-hc-changes)

## Full-model throughput on four MI50s with equal-width Llaminar shards

These historical Llaminar timings used an experimental RCCL build. Both
engines used identical inputs, model shards and GPUs, an empty prompt cache,
and three timed 256-token generations after warmup. Decode covers the last
255 tokens.

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

The four-row FFN reuses K32 activation sums for Q4_K/Q5_1 and Q5_K/Q8_0.
The fixed-depth code2048 verifier fell from 25.08 to 23.45 ms/round. Gate
and down reached 59% and 62% of the same-card read reference.
[Matched runs and read calibration](QFN_MI50_STAGE_COMPARISON_20261118.md#counted-verifier-format-qualification)

The counted four-row reader passed static and deferred replay across all
15 generated format families. Median local time improved 9.23% over staged.
On original 128- and 256-column QFN shards, paired gate/up reads saved
10%–13% on Q4_K; Q5_K was 4.5%–7.9% slower, so its reads stay separate.
On the original TP4 layout, the combined four- and six-row Release and its
predecessor generated identical tokens and performed the same verifier work.
Fixed-depth-3 review improved from 130.46 to 136.83 tok/s and code from
107.48 to 112.71. The fork measured 117.62 and 108.39 on the same inputs
with different output histories. Ordinary decode reached 86.41 and 81.18
tok/s versus the fork’s 85.76 and 73.64. Dynamic reached 129.52 and 106.30
tok/s, 11.76% and 12.69% faster than the same-boot staged control. Thirteen focused preflight tests
passed. The four-row, F256/S256 timings improved in all 15 generated
families, with a median reduction of 22.01%.
On the original six-row F256/S160 shard, gate/up reached 85.0% and down
73.3% of the measured physical read rate for their respective packet layouts.
Register-based routing at intermediate row counts raised dynamic decode to
131.75 and 108.37 tok/s on the same histories. In the six-row format sweep,
reused workgroups recovered 3.3%–5.4% on four indexed formats. Batched
IQ3_XXS codebook reads took the G15 local verifier from 403.71 to 324.75 µs
at six rows. Its gate reaches 27.6% of the corresponding read reference.
G13 shares those gate operands and improved from 437.70 to 359.86 µs.
The original TP4 code2048 fixed-3 target now measures 23.23 ms/round versus
18.22 for Castagna. HC, projections, routing and experts account for 3.08 ms
of the 5.01 ms gap. The HC, selector and indexed-reader changes kept all 18
generations exact. Fixed-depth decode measured 137.38 and 111.88 tok/s on
review512 and code2048. Aggregate performance was within 0.5% of the predecessor.
[Latest four-card stage timings](QFN_MI50_STAGE_COMPARISON_20261118.md#fixed-depth-verifier-after-the-counted-routing-and-hc-changes)
[Indexed-format measurements](QFN_MI50_STAGE_COMPARISON_20261118.md#indexed-weight-formats-on-mi50)
[Architecture coverage](QFN_MI50_STAGE_COMPARISON_20261118.md#verifier-schedules-by-gpu)
[Six-row read calibration](QFN_MI50_STAGE_COMPARISON_20261118.md#physical-read-limits-on-the-six-row-tp4-verifier)
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

Direct peer push saved 1.87–1.96 ms per ordinary token over the earlier
gather and ordered fold, removed 97 nodes per GPU and matched all six
histories. Its isolated 10 KiB reduction took 12.15 versus 26.81 µs.
The stationary four-MI50 layout selects direct peer push on supported links.
Native replays cover partial and empty work,
abort and communicator reuse. A separate publication fusion reduced node
count but slowed both prompts and remains outside the selected policy.
[Transport and lifecycle measurements](QFN_MI50_STAGE_COMPARISON_20261118.md)

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
The fusion remains experimental. The wider TP2 Q8-down shard remains a tuning target.
