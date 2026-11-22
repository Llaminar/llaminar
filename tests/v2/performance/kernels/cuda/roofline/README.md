# CUDA decode hardware references

Build the explicit probe in a CUDA Release configuration. Run each mode on an
idle card. The first record identifies the physical PCI device, ISA, SM count,
and L2 size. Keep raw samples and the final checksum result. Repeat in fresh
processes with reversed card and mode order.

```sh
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_cuda_decode_roofline
PROBE="$CMAKE_BUILD/tests/v2/v2_perf_cuda_decode_roofline"
"$PROBE" 0 dram > dram.csv
"$PROBE" 0 l2 > l2.csv
"$PROBE" 0 dp4a > dp4a.csv
"$PROBE" 0 prmt > prmt.csv
"$PROBE" 0 imma > imma.csv
```

Every sample times 16 consecutive launches in a retained CUDA graph and reports
the duration per launch. The DRAM probe traverses a working set at least eight
times the observable L2 capacity once per launch. The L2 probe repeatedly
traverses a set that fits in half of L2. Both use the same coalesced 16-byte
loads and checksum every output lane. The packed signed DP4A probe uses eight
independent accumulator chains. Its work column counts DP4A instructions across
all threads. Each instruction performs four integer multiply-accumulates.
The byte-permutation probe issues eight independent PRMT chains and counts
one instruction per active lane. Its host oracle checks the resulting byte
rotations for every lane. This is an independent permutation-issue ceiling.
The dense INT8 tensor-core probe counts warp-level `m16n8k32` operations, each
consisting of 4,096 integer multiply-accumulates.

Keep physical traffic separate from requested bytes. In a distinct profiler
process, use `--kernel-name 'regex:decodeReadCeiling' --launch-skip 79
--launch-count 1` to select the last warmed read dispatch. Use
`dram__bytes_read.sum` as the DRAM read denominator and record physical writes
separately. Use device-available L2 read sectors or bytes for the L2 counter
reference. Bind the native counters and static resources to the exact PCI
identity, clocks, executable, grid, and mode. Use quiet event samples for
duration. Follow the CUDA tuning skill's driver diagnostic window and
attachment procedure.

Use the [grouped verifier analyzer](../../qwen4exp/verifier-roofline.md) to
combine these references with the actual routed, shared, and drafter dispatches.
The streaming and issue probes describe independent hardware resources. A
format-specific efficiency result requires replay of its original route order,
live work, prepared weight format and layout, and graph geometry. The measured
parent must also include required publication, ordered reductions, and any
collectives on its critical path.

## RTX 3090 reference measurements

Each entry is the median of 63 retained-launch samples from three fresh
processes. The PCI endpoints are the two sm86 cards in the QFN CUDA/CPU
qualification host.

| PCI endpoint | Streamed reads, GB/s | L2-resident reads, GB/s | DP4A, TOPS | PRMT, trillion lane instructions/s | INT8 tensor core, TOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0000:3f:00.0` | 879.53 | 2349.78 | 73.30 | 8.03 | 317.08 |
| `0000:42:00.0` | 879.53 | 2378.91 | 73.35 | 8.22 | 319.36 |

Read rates count requested bytes. Dot-product rates count two operations per
multiply-accumulate. The driver logged a graphics-engine assertion during
counter attachment on this host, so these references use quiet CUDA events.

The captured M1 MoE regression can retain independent native-leaf event
samples for its IQ2_S gate/up, IQ4_NL Down and 320-column SwiGLU geometry.
Run this in a separate process from the model throughput benchmark.

```sh
LLAMINAR_TEST_CUDA_MOE_M1_PROBE_JSONL="$PWD/moe-m1-leaves.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeGroupedDecodeSwiGLUPublicationReplay
```

The replay verifies nonzero outputs from the exact prepared weights before
measuring the captured leaves. For the 512-column IQ4_NL Down layer, the
logical footprint is 737,280 bytes across eight routes and ten K32 packets
per output. The direct K32 publication took 4.416 µs on both adapters, with
40 registers per thread and zero local bytes. Its requested-weight rate
was 167 GB/s, or 7.0–7.1% of the warm L2 reference. The earlier split-K
projection and reducer took 2.752 and 1.600 µs in separate graphs.
Activation and output traffic are outside the requested-weight calculation.

Set `LLAMINAR_TEST_CUDA_MOE_M1_SOURCE=main`, `main_q5`, or `drafter` to
measure the QFN geometry with hidden width 2560, local intermediate width
320, and eight selected local routes. These cases prepare Q4_K and Q5_1,
Q5_K and Q8_0, or Q8_0 and Q8_0 for Gate/Up and Down respectively. The probe
records the dependent projection and publication nodes in the same timed
sequence, with each source and its requested byte count in the JSONL.

| M1 Down source and case | Split-K stage | Direct stage | Direct requested weight |
| --- | ---: | ---: | ---: |
| Q5_1 main | 8.704 µs | 5.248–5.312 µs | 925–937 GB/s |
| Q8_0 main | 12.736 µs | 8.384 µs | 831 GB/s |
| Q8_0 drafter | 13.056 µs | 8.192 µs | 850 GB/s |

Both RTX 3090s returned the same Q8_0 medians across 62 samples per arm and
card. The final Q5_1 medians use 31 samples per card. Its Down weight
footprint fits in 6 MiB L2 and reached 39.4% of the warm L2 reference.
The drafter's Q8_0 Down footprint is larger than L2 and its requested-weight
rate equals 96.6% of the streaming reference.
Physical DRAM and L2 traffic for these dispatched kernels require hardware
counters. The captured output matched the staged arithmetic at each shape.

Sharing the original K32 activation traversal between Gate and Up saved
1.3–2.5% at the same geometry. On both SM86 cards the paired Q4_K, Q5_K and
Q8_0 stages measured 15.168, 17.408 and 21.504 µs respectively. Each role
retains its separate ordered FP32 accumulator.

The four-row verifier regression can also time the captured Down publication
for each available source codebook. Its JSONL records include the physical
codebook identity, logical weight requests and the number of adjacent
same-expert pairs after sorting the actual routes by expert.

```sh
LLAMINAR_TEST_CUDA_MOE_VERIFIER_DOWN_PROBE_JSONL="$PWD/moe-verifier-down.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeVerifierSmallMPrefillAllNativeFormatsMatchRuntimeDecodeRows
```

The four-row sweep covered 21 source formats at N=512 and K=256, with
31 samples per format and adapter. Requested-weight rates spanned
98–553 GB/s, equivalent to 4.1–23.5% of the warm L2 reference. The
Q5_1 and Q8_0 cases measured 356 GB/s and 544–553 GB/s respectively.
Every measured Down kernel reported zero local bytes and matched the
ordered serial output after retained replay.

Set `LLAMINAR_TEST_CUDA_MOE_M4_DOWN_GEOMETRY=qfn` for N=2560, K=320,
Q5_1 and Q8_0 Down with 32 local routes. Production preflight checks this
geometry in `V2_Integration_CUDA_MoEVerifierQFNWidthPublicationReplay`.
The same probe accepts `iq4` for N=2560, K=256 and both IQ4 source formats.
Use `wide` to sweep all 21 source formats at N=2560 and K=256. The
`V2_Integration_CUDA_MoEVerifierWideNativeFormatsPublicationReplay` preflight
test checks those captured publications against the serial row bytes.

On the QFN geometry, grouping adjacent same-expert routes to share one
weight decode cut logical requests by 28.1%. Median Down latency rose from
14.784 to 17.536 µs for Q5_1 and from 19.200 to 20.224–20.288 µs for Q8_0.
Assigning two lanes to each output column raised the medians to
26.112–26.176 and 27.008 µs respectively. Two-packet unrolling improved both
IQ4 source formats at N=512 but slowed them at N=2560. All three alternatives
passed retained byte comparisons on both RTX 3090 cards.

For the measured SM86 Down formats, loading the activation in two 16-byte
vectors saves 5–22% at M=4. The Q5_1 case drops from 4.864 to 4.416 µs and
Q6_K drops from 7.552–7.616 to 5.888–5.952 µs. The QFN M1 Q5_1 case drops
from 5.440 to 5.312 µs. The vector loads preserve the original K32 packet
order in both paths.

On SM86, the Q5_1 Down path also uses DP4A against packed signed ones to
compute the asymmetric activation correction. The two exact integer half-dots
and sums are joined before applying the original FP32 scale and accumulation
order. For QFN M1, this lowers Down to 4.928 µs. At M=4 it lowers Down from
4.416 to 4.032 µs at N=512 and from 14.784 to 14.656 µs at N=2560.
The compiled M4 loop shrinks from 147 to 104 instructions and uses 37
registers per thread with zero local bytes. The M1 kernel uses 40 registers
and zero local bytes. The model-free preflight runs the captured QFN M1
case in `V2_Integration_CUDA_MoEDecodeQFNWidthPublicationReplay`.

The SM86 grouped verifier uses packed activation sums for the Q4_1 and Q4_K
physical reader and for the Q2_K reader. Across the two cards at N=2560
and K=256, median latency drops from 9.088 to 8.768 µs for Q4_1, from
9.280 to 8.896 µs for Q4_K, and from 10.784–10.880 to 8.832–8.896 µs
for Q2_K.
Both compiled readers use 37 registers per thread and zero local bytes.

Unpacking whole weight words cuts the SM86 verifier's Q2_K loop from 132
to 75 compiled instructions. The Q6_K reader uses aligned eight-byte loads
and expands the high two-bit planes across packed bytes before centering.
At the same wide geometry, Q2_K runs in 7.168–7.232 µs and Q6_K drops from
21.120–21.248 to 14.016–14.080 µs. The readers use 38 and 37 registers per
thread respectively, with zero local bytes. All 21 source formats retain
byte equality with serial decode under capture.

The SM86 Q3_K and Q5_0 verifier readers form signed bytes directly from
the low planes and the inverted high-bit plane. A packed mask supplies the
sign-extension bits for all four lanes together. At N=2560 and K=256,
Q3_K drops from 11.328–11.392 to 7.360 µs and Q5_0 drops from 11.776 to
11.648 µs on both cards. At N=512, Q3_K and Q5_0 measure 3.904 and
4.032 µs. Their compiled readers use 36 and 39 registers respectively,
with zero local bytes.

The wide SM86 verifier uses the canonical 4 KiB ternary IQ1 grid and
expands each entry into signed bytes in registers. IQ2_XXS loads the
index and sign planes of each packet together. IQ3_XXS reads three
aligned words at the wide geometry. The wide reader is selected during
capture from the physical shape and codebook. Both reader variants
retain the original K32 rounding order.

| M=4, N=2560, K=256 | Control, µs | Tuned, µs | Registers per thread |
| --- | ---: | ---: | ---: |
| IQ1_S | 12.736 | 10.752–10.816 | 38 |
| IQ1_M | 13.632 | 12.224 | 40 |
| IQ2_XXS | 11.712 | 10.048 | 39 |
| IQ3_XXS | 11.584 | 10.944–11.008 | 40 |

The main QFN M1 Q5_1 Down reader fixes the ten-packet traversal for
N=2560 and K=320. Time falls from 4.928 to 4.672 µs on both cards.
It uses 44 registers per thread. The four grouped readers and serial
Q5_1 use zero local bytes. Measurements use 31 samples per arm and card.
The wide-format replay verifies both the captured reader selection and
byte equality with serial decode.

The fixed-pitch SM86 reader resolves the wide N and K before entering
the packet traversal. IQ2 uses the canonical magnitude grid encoded
as packed selectors and applies each packet's sign bits in registers.
IQ2_S, IQ2_XS and IQ3_S reassemble their odd-width payloads from aligned
32-bit words. The captured column pitch keeps those loads within each
prepared weight plane.

| M=4, N=2560, K=256 | Previous reader, µs | Fixed-pitch reader, µs |
| --- | ---: | ---: |
| IQ1_S | 10.752–10.816 | 10.496–10.560 |
| IQ1_M | 12.160–12.224 | 11.712–11.776 |
| IQ2_S | 11.904–11.968 | 10.624–10.688 |
| IQ2_XS | 13.376–13.440 | 10.496–10.560 |
| IQ2_XXS | 10.048 | 9.152 |
| IQ3_S | 14.336–14.400 | 13.376–13.440 |

IQ3_XXS varies by one 0.064 µs timing step at this width. All seven
wide readers use 36–40 registers per thread and zero local bytes.

## Route-matched decode weight reference

The single-token QFN fixture can compare its captured Gate/Up and Down
stages with aligned weight readers at the same geometry. The readers use
the original prepared addresses, route slots, projection grid, and output
stores. Gate/Up reads both weight sets and writes the original split-K
partial buffers. Its native stage measurement also includes the ordered
reduction and SwiGLU publication. The fixture checks both packet readers
against a byte-wise checksum and verifies the staged output again after
replaying the complete capture.

```sh
CUDA_VISIBLE_DEVICES=0 \
LLAMINAR_TEST_CUDA_MOE_M1_SOURCE=drafter \
LLAMINAR_TEST_CUDA_MOE_M1_WEIGHT_REFERENCE_JSONL="$PWD/decode-weight-reference.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeGroupedDecodeSwiGLUPublicationReplay
```

Repeat with `main` and `main_q5` for the other QFN format pairs. For each
stage, divide the median time for `decode_gate_up_weight_planes` or
`decode_down_weight_planes` by its native stage median in the same JSONL.
The result estimates the fraction of the matched requested-weight rate
sustained through publication.

| QFN stage, eight live routes | Weight replay, µs | Native stage, µs | Matched weight rate |
| --- | ---: | ---: | ---: |
| Main Q4_K Gate/Up | 12.160 | 15.168 | 80.2% |
| Main Q5_1 Down | 3.904 | 4.608–4.672 | 83.6–84.7% |
| Main Q5_K Gate/Up | 14.272 | 17.408 | 82.0% |
| Main Q8_0 Down | 8.000–8.064 | 8.384 | 95.4–96.2% |
| Drafter Q8_0 Gate/Up | 18.752 | 21.504 | 87.2% |
| Drafter Q8_0 Down | 7.680 | 8.192 | 93.8% |

Each measurement contains 31 samples per card. Both original projections
and the aligned readers report zero local bytes. These rates use logical
requested bytes from an isolated decode fixture. The full mixed-topology
graph and physical memory-traffic rates need their own measurements.

The SM86 capture can resolve the five K32 packets per ordered Gate/Up
partition before recording the node. Unrolling the fixed pitch and sharing
the two aligned activation loads gives 14.848 µs for Q4_K, 17.024 µs for
Q5_K, and 21.312 µs for Q8_0 at the eight-route geometry. The selected
production entries use 48–52 registers per thread and zero local bytes.
In a matched QFN run on two CUDA devices and two CPU participants, the
candidate moved dynamic MTP median decode from 47.87 to 47.96 tokens/s.
All six 256-token histories and per-iteration MTP work records matched
with expert movement disabled during timing.

Set `LLAMINAR_TEST_CUDA_MOE_QFN_ROUTE_COUNT=10` for the QFN model's full
selected-expert count. The width-specific production preflight replays use
that setting. The reference fixture keeps every selected route local so it
can compare a known work volume with the identical captured launch.

| QFN stage, ten live routes | Native stage, µs | Matched weight rate |
| --- | ---: | ---: |
| Main Q4_K Gate/Up | 17.344 | 84.1% |
| Main Q5_1 Down | 6.208 | 86.6% |
| Main Q5_K Gate/Up | 19.904 | 85.9% |
| Main Q8_0 Down | 12.224 | 96.9% |
| Drafter Q8_0 Gate/Up | 25.408 | 89.4% |
| Drafter Q8_0 Down | 12.608–12.672 | 96.5–97.0% |

## Route-matched verifier Down weight reference

The verifier fixture can reuse the original capture arguments to measure a
weight-only reference. The reader retains the prepared addresses, original
route order, grid, and output stores. It checks an aligned packet checksum
against an independent byte-wise implementation before taking timing samples.
The final verifier replay checks the original output bytes again after the
diagnostic overwrites its scratch.

```sh
CUDA_VISIBLE_DEVICES=0 \
LLAMINAR_TEST_CUDA_MOE_M4_DOWN_GEOMETRY=wide \
LLAMINAR_TEST_CUDA_MOE_VERIFIER_DOWN_WEIGHT_REFERENCE_JSONL="$PWD/weight-reference.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeVerifierSmallMPrefillAllNativeFormatsMatchRuntimeDecodeRows
```

Compare the medians for `verifier_routed_down_weight_planes` and
`verifier_routed_down` from the same run. Both request the same logical
weight extent. The reference time divided by the Down time estimates
how much of the measured weight-only rate the complete leaf sustains.
The reference includes descriptor reads and checksum stores. Kernel
arithmetic, activation traffic, and physical DRAM and L2 counters must
be measured separately for a whole-leaf roofline.

With 31 samples per format and card, 12 of 21 source formats sustain at
least 80% of the weight-only rate at M=4, N=2560, K=256 on both RTX 3090s.
At N=512 and K=256 the highest rate is 76.2%. At the QFN-shaped N=2560
and K=320, Q5_1 and Q8_0 sustain 94.3–94.8% and 96.3% respectively.

## Route-matched verifier Gate/Up weight reference

The four-row verifier fixture also measures the original Gate/Up scatter
and its ordered SwiGLU publication in the same retained sequence. The
weight-only reader has the scatter's grid and addresses and overwrites both
partial planes. The fixture verifies exact serial-row bytes after replaying
the original parent over the diagnostic scratch.

```sh
CUDA_VISIBLE_DEVICES=0 \
LLAMINAR_TEST_CUDA_MOE_M4_DOWN_GEOMETRY=qfn-main \
LLAMINAR_TEST_CUDA_MOE_VERIFIER_GATE_UP_WEIGHT_REFERENCE_JSONL="$PWD/verifier-gate-up.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeVerifierSmallMPrefillAllNativeFormatsMatchRuntimeDecodeRows
```

Use `qfn-main` for Q4_K Gate/Up and Q5_1 Down, `qfn-main-q5` for Q5_K
Gate/Up and Q8_0 Down, and `qfn` for the same-format Q8_0 and Q5_1 cases.
`wide` measures all 21 source formats with Gate/Up N=256 and K=2560.
Omit the geometry variable for N=256 and K=512. Divide the median
`verifier_routed_gate_up_weight_planes` time by the median
`verifier_gate_up_native_graph` time to include ordered publication in the
requested-weight comparison. The `verifier_routed_gate_up` record isolates
the scatter and reports the original launch and compiler resources.

| QFN Gate/Up source, 32 live routes | Weight replay, µs | Native stage, µs | Matched weight rate |
| --- | ---: | ---: | ---: |
| Q4_K | 23.936–24.000 | 29.056 | 82.4–82.6% |
| Q5_K | 27.840–27.904 | 34.112–34.176 | 81.6% |
| Q8_0 | 37.184–37.248 | 44.672–44.928 | 82.9–83.2% |

With ten routes selected for each of four verifier rows, the fixed packet
schedule lowers Q4_K Gate/Up from 34.624–34.688 to 33.408 µs and Q5_K from
42.432–42.560 to 40.256 µs. Q5_K rises from 79.5–79.8% to 84.3% of its
matched weight rate. Across both cards, the QFN Gate/Up cases sustain
84.2–88.3% through publication. The QFN-width Q8_0 producer retains the
general schedule after the comparison showed similar stage times.

Unrolling the packed wide packets and the fixed 16-part ordered publication
brings 12 of 21 wide formats to at least 80% in both Gate/Up and Down on
both cards. The 31-sample Gate/Up medians are 19.584 µs for Q2_K (86.3%),
20.544 µs for Q3_K (81.3%), 25.408 µs for Q5_0 (81.1%) and
30.912–30.976 µs for Q6_K (87.2–87.4%). IQ2_XS falls from the earlier
27.264 µs control to 20.864 µs (74.5%). The low-bit IQ1, IQ2 and IQ3
formats range from 49.8% to 77.4% at the wide geometry. The smaller
geometry still falls short of 80%. These measurements retain the tested
prepared formats and fully local route tape. The same Release build
passes ten focused CUDA preflight tests and preserves generated tokens
in the real 2×CUDA/2×CPU QFN movement regression.

The SM86 wide IQ4 schedule uses adjacent-nibble lookup results directly.
Gate and Up share the reordered activation and independently resolve their
physical scales before the packet loop. Down uses the same integer program
with the canonical ordered K32 fold. The measured Gate/Up launch has 52–54
registers and room for nine CTAs per SM. Down has 44 registers and room
for sixteen CTAs per SM. Both report zero stack and local bytes.

| Source | Gate/Up with publication, µs | Matched rate | Down, µs | Matched rate |
| --- | ---: | ---: | ---: | ---: |
| IQ4_NL | 22.272 | 81.0% | 9.024–9.088 | 78.2–78.7% |
| IQ4_XS | 24.448 | 79.8% | 9.024 | 85.1% |

The mixed IQ4 preflight executes both Gate/Up scale orderings against the
serial-row result. Before and after the shared lookup extraction, the
compiled ROCm residual reader instruction bytes match for gfx906 and gfx1100.

The mixed 2×CUDA/2×CPU QFN dynamic MTP benchmark produced identical
256-token histories and per-iteration MTP work in both run orders. With the
tuned verifier first, median decode was 47.29 tokens/s versus 47.54 for the
control. With the control first, medians were 47.57 and 48.02 tokens/s for
control and tuned verifier respectively. Each arm has three measured runs.
The throughput difference changed direction with run order.

## QFN mirrored drafter vocabulary projection

The matched topology uses a mirrored full-vocabulary head on both CUDA
participants. The shared MTP sidecar binds the target model's `output.weight`,
which is Q8_0 at N=248320 and K=2560 in the measured UD-Q4_K_XL model.
The isolated Release fixture loads that tensor through the native model
loader and captures the installed M=1 projection with the TP2 serial-policy
width of 124160. It measures the retained quantize and projection graph and
checks the full logits for byte-exact repeatability.

| RTX 3090 PCI endpoint | Graph median, µs | GEMV median, µs | Requested weight GB/s |
| --- | ---: | ---: | ---: |
| `0000:3f:00.0` | 765.952–766.976 | 763.008–763.072 | 880.64–881.82 |
| `0000:42:00.0` | 765.952 | 763.008–763.072 | 881.82 |

Each range spans three process medians of 31 retained-graph samples. The
projection records 7760 CTAs of 32 threads, 38 registers and zero local
bytes. Its graph requests 675,430,400 packed weight bytes, which puts the
measured rate within 0.3% of the independent 879.53 GB/s streaming reference.
This comparison uses requested bytes. The production trace measures a
larger captured fragment that also contains the routed fold and other work.

## QFN grouped vocabulary verification

The fixed-width grouped verifier uses the same original Q8_0 tensor and the
same serial output-partition policy. Its SM86 single-partition specialization
loads the prepared Q8 operands directly into MMA registers, applies the
original K32 scales in ascending order, and folds the one ordered partition
into the final logits store. It keeps one weight traversal for up to 16 live
rows and omits the intermediate FP32 partial plane.

The production-preflight replay covers five and sixteen live rows. It
checks exact serial bytes after poisoning the retained output and inspects
the original captured SM86 kernel, including its grid and compiler
resources. The same fixture can be exercised with the Release target.

```sh
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_cuda_device_counted_verifier_rows
CUDA_VISIBLE_DEVICES=0 "$CMAKE_BUILD/tests/v2/v2_perf_cuda_device_counted_verifier_rows" \
  --gtest_filter=CUDAQFNVocabularyVerifier.OrderedSinglePartitionPublicationReplay
```

The realized Release path produced exact serial logits on both RTX 3090s.
Each result is the median of 31 retained-graph samples with the real
675,430,400-byte vocabulary weight. The efficiency column divides that
unique weight size by captured stage time and then by 879.53 GB/s.
The original schedule traverses the weights twice at 8 and 12 rows and
three times at 16 rows.

| Rows | Original stage, µs | Fused Release stage, µs | Streaming reference |
| ---: | ---: | ---: | ---: |
| 5 | 1015.808–1016.832 | 780.288 | 98.4% |
| 6 | 1025.024–1030.144 | 782.336 | 98.2% |
| 8 | 1991.680–1995.776 | 787.456 | 97.5% |
| 12 | 2057.216–2064.384 | 801.792 | 95.8% |
| 16 | 3061.760–3077.120 | 814.080 | 94.3% |

The fused kernel records 7760 CTAs of 128 threads, 46 registers and room
for ten CTAs per SM. The compiled body contains integer tensor-core
instructions, zero shared and local bytes, and no warp synchronization.

On the matched 2×CUDA/2×CPU QFN model at fixed draft depth seven, the
control-first comparison measured median decode throughput of 44.42 tokens/s
for the control and 44.89 for the fused prototype. Each arm ran three
measured requests after warmup. All six requests produced the same 256
tokens and the same 57 verifier transactions, 456 verifier rows, and
399 drafts.
