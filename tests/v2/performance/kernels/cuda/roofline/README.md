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
"$PROBE" 0 lop3 > lop3.csv
"$PROBE" 0 int-mix > int-mix.csv
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
The `lop3` and `int-mix` probes count scalar-lane instructions from independent
chains. The mixed loop issues two LOP3, one signed DP4A and one IMAD per
chain. It measures that instruction schedule separately from the weight
reader and the end-to-end projection.

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

### Resident HC, router, and FFN

The [MI50 calibration](../../../../../../docs/v2/projects/2026-11/QFN_MI50_EMPIRICAL_COMPUTE_MEMORY_REFERENCE.md)
defines 15 original source tuples at M1, H2560, HC rank 320, routed/shared
widths 256/256, 512 experts and ten selected routes. The CUDA probe reads that
same source roster. It also emits the other QFN main-model and Q8_0 drafter
tuples. Run the estimator with the prepared widths and the actual number of
local routes for each participant. An incompatible original K block is
reported without a score.

The table uses 63 checksum-verified samples per cell, collected in three
fresh processes with reversed device and probe order. Each process records
the native PCI ID and launches one exact mode. CTA width is 256 threads.

| PCI | CTAs | DRAM GB/s | L2 GB/s | DP4A Ginst/s | FP32 FMA Ginst/s | PRMT Ginst/s | Mixed INT Ginst/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0000:3f:00.0` | 82 | 886.17 | 2,304.44 | 8,223.34 | 8,788.15 | 8,016.80 | 12,548.03 |
| `0000:42:00.0` | 82 | 886.17 | 2,348.79 | 8,344.65 | 8,919.42 | 8,138.17 | 12,726.28 |
| `0000:3f:00.0` | 164 | 885.66 | 2,350.81 | 9,028.06 | 12,151.39 | 7,984.05 | 13,046.76 |
| `0000:42:00.0` | 164 | 885.66 | 2,387.36 | 9,127.73 | 12,248.32 | 8,147.43 | 13,427.67 |

Pass an explicit CTA count as the final hardware probe argument. For example,
record 82- and 164-CTA captures on each card for `dram`, `l2`, `dp4a`, `ffma`,
`prmt` and `int-mix`, then pass the receipts to the scorer.

```sh
"$PROBE" 0 ffma 82 > device0-82-ffma.csv
"$PROBE" hc-ffn-work 2560 320 256 256 512 10 10 > family-work.csv
"$PROBE" hc-ffn-work 2560 320 320 320 512 10 4 > overlay-work.csv
python3 tests/v2/performance/kernels/cuda/roofline/score_hc_ffn.py \
    family-work.csv "$CALIBRATIONS"/*.csv > family-reference.csv
```

The scorer takes the dominant read or useful arithmetic time for each ordered
phase and adds the phases. HC Down and the paired HC Up count their Q8_0
weights once. The paired Up counts both integer MACs. Routing counts the
prepared FP16 weight and scalar FP32 MACs. The FFN counts both Gate/Up roles
and one Down role for each live local route and the always-active shared
expert. Integer MACs use four per DP4A. The named mixed and PRMT probes can
explain instruction pressure once the exact reader instruction mix is known.

The native rendezvous probe captures otherwise identical cooperative grids
with zero, one, and six joins. Each CTA publishes one checked word. Each
table cell is the median of 63 samples from three fresh processes with
reversed probe order. The probe checks residency against the loaded CUDA
entry before recording the graph.

| PCI | CTAs | Zero joins, µs | One join, µs | Six joins, µs | Increment for six joins, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0000:3f:00.0` | 82 | 1.408 | 2.112 | 5.184 | 3.776 |
| `0000:42:00.0` | 82 | 1.408 | 2.112 | 5.184 | 3.776 |
| `0000:3f:00.0` | 164 | 1.472 | 2.176 | 5.184 | 3.712 |
| `0000:42:00.0` | 164 | 1.536 | 2.112 | 5.184 | 3.648 |

Keep these latency samples alongside the hardware rate calibration. The
increment estimates rendezvous overhead in an otherwise idle cooperative
grid. Work imbalance and memory visibility in the model can add to that
time.

```sh
for mode in grid-0 grid-1 grid-6; do
    "$PROBE" 0 "$mode" 164 > "$RENDEZVOUS/device0-164-$mode.csv"
done
```

The 15 common tuples span 7.97–10.75 µs for ideal L2 delivery on PCI3f at
82 CTAs, and 20.73–27.96 µs for one DRAM stream of every compulsory weight
request. At F640/S320, the Q4_K/Q5_1/Q8_0 model tuple's streaming reference
is 13.76, 32.25, or 50.74 µs with 0, 5, or 10 local routed experts.
The L2 estimate is an optimistic reference. The DRAM estimate assumes every
compulsory weight request reaches DRAM once. These estimates exclude packet
decode and correction, activation and output traffic, routing metadata,
barriers, launch costs and the collective. Use graph event times for the
complete transaction and capture contents for launch accounting. A lower
bound or a ratio using requested bytes cannot establish physical bandwidth
or end-to-end throughput.

The staged CUDA comparator below used native graph events from the attention
write through the local routed slots and weighted shared output. It ran the
15 original source combinations at F256/S256 with ten local routes, using
deterministic synthetic prepared planes. Each 21-node graph repeated two
passes per card with reversed source order and 31 event samples per pass.
Five independent initial states repeated byte-exactly. The explicit
collective and the model loader were outside this diagnostic.

| Family | Packed FFN bytes | 82-CTA DRAM reference, µs | Staged CUDA median range, µs |
| --- | ---: | ---: | ---: |
| G01 | 15,196,160 | 27.96 | 96.78–98.07 |
| G02 | 12,165,120 | 24.54 | 91.94–93.27 |
| G03 | 13,148,160 | 25.65 | 93.06–93.98 |
| G04 | 12,165,120 | 24.54 | 96.72–97.64 |
| G05 | 13,148,160 | 25.65 | 94.44–95.30 |
| G06 | 12,165,120 | 24.54 | 93.20–94.17 |
| G07 | 13,148,160 | 25.65 | 94.23–95.14 |
| G08 | 8,785,920 | 20.73 | 105.20–106.42 |
| G09 | 10,076,160 | 22.19 | 98.85–99.89 |
| G10 | 10,117,120 | 22.23 | 95.49–96.50 |
| G11 | 13,148,160 | 25.65 | 100.98–102.01 |
| G12 | 10,731,520 | 22.93 | 101.86–102.94 |
| G13 | 11,448,320 | 23.73 | 102.51–103.59 |
| G14 | 13,496,320 | 26.05 | 107.11–108.20 |
| G15 | 12,697,600 | 25.14 | 104.18–105.57 |

The real 2×CUDA/2×CPU tensor-sharded graph uses F320/S320. At four live
local routes, 164 CTAs, and 256 threads per CTA, the standalone resident
comparison produced these native event medians on the two RTX 3090s.

| Routed Gate/Up and Down | Ideal L2 time | One DRAM stream | Staged | Resident |
| --- | ---: | ---: | ---: | ---: |
| Q4_K and Q5_1 | 7.85–7.98 µs | 21.17 µs | 87.37–87.42 µs | 70.88–70.90 µs |
| Q4_K and Q8_0 | 8.28–8.41 µs | 22.33 µs | 85.02–85.10 µs | 67.82–68.16 µs |
| Q5_K and Q8_0 | 8.63–8.76 µs | 23.25 µs | 86.18–86.19 µs | 69.64–69.66 µs |

The positive-row prototype reduced 20 native nodes to one and matched the
staged bytes in 18 states per layout and GPU, including all, partial and
zero local routed selections. Gate/Up uses CTA-local partials and emits
the Q8 operands for Down. All CTAs own an HC Up tile on a live row. The three
timed entries each used 96 registers and zero local bytes per thread. The
single-stream DRAM reference is 29.9–33.4% of the resident time. The fixture
omits the mixed graph's CPU dispatch and collective dependencies. The
device-counted variant passed zero/full reset and output guard checks and
took 77.10–79.60 µs with the original formats.

A pinned-bank diagnostic changed the physical Down decoder between packed
Q5_1 and its lossless expanded representation. It compared the staged path
and the original output bytes through 36 captured transitions across both
cards and both launch contracts, including mixed physical formats in one
row. The mixed-format positive-row timings were 70.74–72.86 µs in the
resident program and 87.89–90.77 µs staged. The native counted graph
contains a 1-CTA predicate and an IF body with the 164×256 cooperative grid.

The mapped CUDA candidate also admits lossless Q5_K Gate/Up promotion.
The device packs compact and expanded descriptors into separate ranges before
the arithmetic. Twelve native comparisons cover the three original source
pairs on both cards with both row contracts. The two promoted pairs preserved
the staged bytes across 72 pinned-bank changes. Every entry used 96 registers
and zero local bytes. `V2_Integration_CUDA_RetainedHCFFNCountedReplay`
replays the recorded staged output digests through the native counted graph.

The Release candidate selected by
`LLAMINAR_QFN_CUDA_MAPPED_RETAINED_M1_HC_FFN_CANDIDATE=1` captures the
complete original QFN serial decode graph. At F320/S320 it reduces the
CUDA root from 3,536 to 2,576 native nodes and the follower from 2,911 to
1,951. Both retain 49 compilation units and one native executable each.

The real Release comparison used the same prompt, admitted tensor sharding
and one warmup followed by three measured 256-token iterations per cell.
All token IDs and per-iteration MTP work match in both run orders.

| Run order | Ordinary control → candidate | Fixed-depth-7 control → candidate |
| --- | ---: | ---: |
| Candidate first | 29.385 → 27.538 tok/s (−6.28%) | 43.034 → 43.698 tok/s (+1.54%) |
| Control first | 30.071 → 27.533 tok/s (−8.44%) | 44.394 → 43.918 tok/s (−1.07%) |

The candidate stays opt-in. The fused producer publishes the router outputs
in the original benchmark only after its local FFN completes. This places
the packet later on the ordinary decode path than the staged release, which
publishes it before local FFN math.

The early-packet revision releases the mixed hidden row and router outputs
before expert arithmetic. The root records its original packet stages as a
sibling of the resident kernel and joins both native completions before the
ticket boundary. The root captures 2,672 nodes, including one reset and one
acquire for each of the 48 layers. The follower captures 1,951 nodes. Both
retain one executable and 49 compilation units per GPU. The real Release
smoke matched the first 12 reference tokens on the 2×CUDA/2×CPU plan.

`V2_Integration_CUDA_RetainedHCFFNEarlyPacketReplay` checks the actual
native fork, both packet edges and the retirement join. It replays the
recorded serial bytes across the three source pairs and pinned placement
changes. Both RTX 3090s passed this replay and the original counted replay.
The same-device acquire uses 10 registers and no local memory. The resident
entries retain their 96–108 register allocation and zero local bytes. The
early fork admits fully active rows. The counted replay exercises the
separate native conditional entry, including empty rows.

### Grouped Gate/Up packets and direct Down on SM86

The grouped native program uses one 256-thread CTA per live 32-column
Gate/Up packet. It folds all 16 serial partitions in shared memory, applies
SwiGLU and writes the Down Q8 codes and scales. The ordered 128-thread Down
tiles read that publication directly. A device plan sets each grid in the
original native executable and disables both arithmetic nodes when the live
count is zero. There are no global Gate/Up partials or separate SwiGLU/Q8
launches in the isolated capture.

The retained packet and grid preflight tests passed on each RTX 3090. The
packet replay cycles through live counts `6,3,0,5,1,0,6` and checks recorded
serial CUDA Q8 and Down bytes, including both placements of promoted operands
and poisoned inactive output. `cuobjdump` reports 64–72 registers and 4,096
shared bytes for Gate/Up, and 46–63 registers and no shared memory for Down.
All 16 entries use zero stack and local bytes. The smaller Down tile improved
single-row Q8_0 timings by 3.7–6.9% across the two widths and roles. At six
rows the change was within 1% for every measured format and role.

Both cards timed captured control and math together with six independent
prepared rows, including their format-specific weights. Percentages use
one compulsory read of the packed weights, scales and minima at 886.17 GB/s.

| Gate/Up and Down physical format | Role | F320, µs | F320 weight reference | F640, µs | F640 weight reference |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_K / Q5_1 | Routed | 21.728–21.760 | 51.0–51.1% | 35.744–35.872 | 61.9–62.1% |
| Q4_K / expanded Q5_1 | Routed | 23.904 | 55.1% | 37.344–37.408 | 70.4–70.6% |
| Q4_K / Q8_0 | Routed | 23.840–23.872 | 53.7–53.8% | 36.608 | 70.1% |
| Q5_K / Q8_0 | Routed | 25.248–25.344 | 56.1–56.3% | 39.200–39.232 | 72.5–72.5% |
| Expanded Q5_K / Q8_0 | Routed | 29.056 | 63.2% | 47.584–47.680 | 77.1–77.2% |
| Q8_0 / Q8_0 | Routed | 27.584–27.872 | 63.4–64.1% | 46.077–46.144 | 76.6–76.7% |
| Q8_0 / Q8_0 | Shared | 28.128–28.416 | 62.2–62.9% | 45.984–46.080 | 76.7–76.9% |

The isolated replay supplies prepared operands. Production verifier and
drafter qualification also requires the original runtime route publication,
shared-gate and collective edges, and a paired complete transaction.

### CUDA peer transfers at QFN decode sizes

The two QFN RTX 3090s report direct access in both directions over the
PCIe PXB link. CUDA reports no native atomics across that link. The
diagnostic therefore joins the original CUDA streams with native graph
events. The NCCL controls use the same configured provider as the Release
runtime. Each captured transaction times 32 successive operations on two
already-prepared FP32 rows and checks the complete receiver publication
after three input resets. The table spans both directions and two passes
with reversed operation order, each with 43 samples per arm.

| Payload per direction | Direct store | Peer DMA | Direct rooted publication | Root pulls peer | NCCL Reduce |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,240 B | 2.27 µs | 2.85–3.05 µs | 3.68 µs | 3.84 µs | 9.86–10.08 µs |
| 102,400 B | 7.97–8.10 µs | 7.74–8.87 µs | 11.10–12.51 µs | 10.75 µs | 24.83–30.69 µs |
| 409,600 B | 31.23–31.33 µs | 31.00–32.14 µs | 55.68–55.78 µs | 33.50–33.82 µs | 56.51–60.42 µs |

The direct rooted publisher reads both original inputs and writes the sum
from the source GPU into the receiver's local VRAM in one kernel. The
root-pull variant reads the peer input from the receiving GPU and writes
the sum locally. A separate push and receiver fold costs 10.94–11.62 µs
at 10,240 B. The published and pulled sums use the original FP32 add order.
The fixture starts with ready inputs. Production graphs must also preserve
the input-producer events and ownership before reusing either participant's
buffers. The table measures isolated transport and reduction, before any
whole-model comparison.

The 32-, 64-, 128- and 256-thread sweep exercised both PCI directions. The
32-thread direct rooted publisher measured 3.62–3.65 µs at 10,240 B. At
102,400 B the 256-thread root pull measured 10.75 µs in both directions.
At 409,600 B all four root-pull widths measured 33.50–33.82 µs. Forcing
`LL`, `LL128` or `Simple` gave no repeatable NCCL Reduce improvement at
10,240 B or 102,400 B. The CUDA kernels use 14 registers for the direct
store and 18 for the two-input add, with zero private bytes.

```sh
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_cuda_peer_decode_latency
PEER="$CMAKE_BUILD/tests/v2/v2_perf_cuda_peer_decode_latency"
"$PEER" 0 direct 10240 128
"$PEER" 0 publish-root 10240 128
"$PEER" 1 pull-root 102400 128
"$PEER" 1 nccl-root 102400 128
```

### Independent projection and issue references

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

The mixed instruction probes used the same 63-sample procedure.

| PCI endpoint | LOP3, trillion lane instructions/s | Integer mix, trillion lane instructions/s |
| --- | ---: | ---: |
| `0000:3f:00.0` | 7.816 | 13.041 |
| `0000:42:00.0` | 7.569 | 13.262 |

The compiled loops use 26 and 49 registers respectively, with zero stack,
shared, and local bytes. The mixed body contains 16 LOP3, eight DP4A and
eight IMAD instructions per iteration, counted independently of loop
control. The route-matched weight comparisons below retain their measured
weight reference as the denominator.

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
In that sweep at N=512 and K=256, the highest rate was 76.2%. At the
QFN-shaped N=2560 and K=320, Q5_1 and Q8_0 sustained 94.3–94.8% and 96.3%
respectively.

For the short N=512, K=256 Q8 Down projection, SM86 uses a fixed eight-packet
schedule. Each lane has two independent signed dot chains per K32 packet
and the original ordered FP32 fold. A paired sweep of unroll factors one,
two, four, and eight selected two packets per loop iteration. All three Q8
source layouts passed the byte-exact replay.

| Short Q8 source | Release Down, µs | Route-matched weight rate |
| --- | ---: | ---: |
| Q8_0 | 3.456 | 90.7% |
| Q8_1 | 3.456 | 88.9–90.7% |
| Q8_K | 3.392 | 90.6% |

The 64-column CTA uses 50 registers, zero shared and local bytes, and no
warp synchronization. The captured replay verifies the short kernel's
launch against the admitted output geometry.

### QFN-width source coverage

Set `LLAMINAR_TEST_CUDA_MOE_QFN_ROUTE_COUNT=10` for both QFN-width sweeps.
Use `LLAMINAR_TEST_CUDA_MOE_M4_DOWN_GEOMETRY=qfn-all-gate` with
`LLAMINAR_TEST_CUDA_MOE_VERIFIER_GATE_UP_WEIGHT_REFERENCE_JSONL` to measure
all 21 source formats at Gate/Up N=320 and K=2560. The paired Down weights
use QFN's Q8_0 source. Use `qfn-all-down` with
`LLAMINAR_TEST_CUDA_MOE_VERIFIER_DOWN_WEIGHT_REFERENCE_JSONL` to measure
the seven source formats with 32-value source blocks at Down N=2560 and
K=320. The `wide` sweep exercises the 256-value-block Down formats at K=256.
Both QFN-width sweeps retain all 40 routes and compare captured M=4 output
with the serial-row oracle, including the replay after diagnostic timing.

At baseline, 10 of 21 Gate/Up formats and six of seven Down formats
reached 80% of their matched weight-only rate on both RTX 3090s. The
remaining Down format, IQ4_NL, achieved 76.9–78.3% at 18.560–18.880 µs.
All source formats in both sweeps matched the serial oracle byte for byte.

### Fixed QFN-width Gate/Up packets

The SM86 packed Gate/Up reader fixes each partition at five K32 packets.
It loads the original activation once per packet for both projections and
keeps the original FP32 accumulation order. For M=4 and ten routes per
row, Q2_K and Q6_K use 128-column CTAs. The other packed readers use
64-column CTAs, which divide the 320-column output evenly. After the
change, 14 of 21 QFN-width Gate/Up formats reached 80% on both cards.

| Source | Gate/Up with publication, µs | Matched weight rate |
| --- | ---: | ---: |
| Q2_K | 28.480 | 88.5% |
| Q3_K | 29.248–29.312 | 81.4–81.6% |
| Q6_K | 45.952–46.016 | 83.7–83.8% |
| IQ3_S | 33.984–34.048 | 85.3–85.5% |
| IQ3_XXS | 26.816–26.880 | 78.1–78.3% |
| IQ2_S | 29.760–29.888 | 72.8–74.4% |
| IQ2_XS | 28.544 | 75.3–76.5% |
| IQ2_XXS | 23.488–23.552 | 68.5–68.7% |
| IQ1_S | 26.816–26.880 | 61.2–62.1% |
| IQ1_M | 28.480–28.608 | 58.2–58.6% |

The six sources below 80% use complete 64-column tiles. Capture fixes
the grid and K-partition bounds, leaving each CTA to resolve its routed
operands before the dot. IQ1_S also uses independent integer dot and
activation-sum chains for each half, then applies the original FP32 scale
and minimum correction. The compiled readers use 40–62 registers and
zero shared, stack and local bytes. Their launch geometry allows 9 or
16 CTAs per SM. The all-format replay verifies the captured tile and
serial-row output bytes.

At the same QFN geometry the SwiGLU publication uses a fixed 320-column
pitch. Each submitted warp owns a complete K32 output group and folds
the original 16 Gate and Up partitions in order. The captured kernel
uses 40 registers, zero shared and local bytes, and no CTA barrier. In
three paired runs per card, the IQ3_XXS stage dropped from 26.752 µs to
26.560–26.624 µs. The full Release sweep measured 26.688 µs on both
cards, or 79.4–79.6% of the matched weight reference. The 80% counts
remain 15 of 21 for Gate/Up and seven of seven for Down.

The complete QFN warp now reduces unsigned FP32 magnitude bits with one
native warp maximum and converts the bounded scale with binary16 RNE.
The publisher retains the original ordered FP32 fold and Q8 quantizer.
The focused replay checks output bytes for all-zero, minimum-scale,
subnormal-scale, and saturated-scale packets. The IQ3_XXS stage went
from 26.688 to 26.432 µs in the paired runs. The Release sweep measured
26.432 µs on both cards, or 79.4–80.4% of the matched weight reference.

The fixed-width publisher uses its positive binary16 scale range to
apply the original Newton reciprocal with one FP32 exponent adjustment.
Every positive finite half value matched the original reciprocal bits
on both cards. The IQ2_XS tile also deposits the even and odd prepared
sign bits independently, using 48 registers and zero shared or local
bytes. Across six focused Release runs, Gate/Up took 27.712–27.904 µs
for IQ2_XS and 26.304–26.368 µs for IQ3_XXS. IQ3_XXS reached 80.05–81.27%
in those runs and 79.6–80.4% in the full-format sweep.

For the six low-bit 64-column readers, the complete runtime plan now checks
both prepared descriptor planes before publishing them. Each projection
uses the validated publication for its exact stream and workspace. The
compiled tiles remove six load instructions and 16–20 comparisons, with
the same 40–62 registers and zero shared, stack, or local bytes. One full
sweep reached 80% on both RTX 3090s for 18 of 21 Gate/Up formats. A second
sweep reached 80% on both cards for 16 formats. IQ2_S and IQ2_XS varied
around 80% because the matched reader timings varied between runs. All 21
formats matched serial output bytes in both sweeps. The paired complete
captured verifier transactions changed by less than 0.6% in either
direction, including the plan checks.

| Source | Release Gate/Up with publication, µs | Matched weight rate |
| --- | ---: | ---: |
| IQ1_M | 27.456–27.584 | 60.1–61.3% |
| IQ1_S | 24.832–25.024 | 65.7–67.3% |
| IQ2_S | 27.648–27.840 | 79.6–80.3% |
| IQ2_XS | 27.200–27.392 | 79.0–80.9% |
| IQ2_XXS | 22.080–22.208 | 72.9–73.7% |
| IQ3_XXS | 26.240 | 80.2–81.0% |

The four plan-validated IQ1_M, IQ1_S, IQ2_S, and IQ2_XS tiles use a
compiler target of sixteen 64-thread CTAs per SM. They use 64 registers
and zero shared, stack, or local bytes. The exact launch geometry and
ordered partition arithmetic are the same in both timing arms.

| Source | Release Gate/Up with publication, µs | Matched weight rate | Complete capture time reduction |
| --- | ---: | ---: | ---: |
| IQ1_M | 26.688–26.752 | 62.1–63.2% | 0.82–0.89% |
| IQ1_S | 24.448–24.576 | 67.7–68.3% | 0.61% |
| IQ2_S | 26.880–27.008 | 80.7–82.0% | 0.89–0.97% |
| IQ2_XS | 27.072–27.200 | 79.7–80.9% | 0.15% |

This sweep matched serial bytes for all 21 Gate/Up and seven K=320 Down
formats on both RTX 3090s. Seventeen Gate/Up and all seven Down formats
reached 80% of their matched reference on both cards.

The six plan-validated low-bit tiles share the rounded exponential and
reciprocal sequence for positive and negative gates. Independent device
functions returned identical SiLU words for all 4,294,967,296 binary32
input patterns on each RTX 3090. The captured reducer uses 216 compiled
instructions, down from 264. The native replay checks mixed signs and the
exp cutoff around −87 alongside the binary16 scale boundaries.

Five of the six formats reduced complete capture time by 0.064–0.256 µs
in the paired runs. IQ2_XS varied from a 0.064 µs gain to a 0.192 µs loss.
The Release sweep measured IQ3_XXS at 26.112 µs on both cards and IQ2_S
at 26.752–26.944 µs. Both cleared 80% of the matched reference on each card.

The validated 64-column IQ2_XS tile extracts the high index byte and
gathers the four sign bytes directly from the aligned words. At byte
offset three the sign word is the entire next word, so the clamped shift
must select that word. The prepared weight planes and the integer dot
order are unchanged. The Release tile uses 64 registers with zero local
or shared memory. The same route-matched reference gave 82.30% and 81.77%
on the two RTX 3090s at 26.752 and 26.688 µs, respectively. In two paired
orders the complete capture improved by 0.192–0.448 µs. The full sweep
matched serial output bytes and reached 80% on both cards for 18 of 21
Gate/Up formats and all seven K=320 Down formats. The remaining Gate/Up
rates were 62.5–62.7% for IQ1_M, 67.5% for IQ1_S, and 73.3–73.5% for
IQ2_XXS.

Forcing the existing grouped INT8 tensor-core engine at four rows
preserved serial output bytes on the six low-bit sources. The complete
captured transaction, including routing and publication, ranged from
169.5 to 254.1 µs across both cards. The selected packed path took
81.0–87.4 µs on the same sources and inputs.

At M=4 with ten routes per token, IQ1_S and IQ2_XXS address the original
activation row directly from the route slot. The grouped producer
validates the inverse map before the projection. The selected
64-column tiles use 52 and 40 registers respectively and zero shared
or local bytes.

| Source | Paired control, µs | Release Gate/Up with publication, µs | Matched weight rate |
| --- | ---: | ---: | ---: |
| IQ1_S | 26.688–26.816 | 25.600–25.728 | 64.9–65.0% |
| IQ2_XXS | 23.360–23.488 | 22.976–23.040 | 70.2–71.1% |

### QFN-width IQ4 schedule

The interleaved IQ4 reader also uses the fixed 320-column Gate/Up pitch
and ten-packet Down traversal. Each Gate/Up descriptor resolves its native
scale program before the packet body. The replay exercises both pairings
of IQ4_NL and IQ4_XS scale programs at the QFN width.

| Source and projection | Release stage, µs | Matched weight rate |
| --- | ---: | ---: |
| IQ4_NL Gate/Up | 31.872 | 81.7% |
| IQ4_XS Gate/Up | 34.496–34.560 | 86.1–86.3% |
| IQ4_NL Down | 16.192 | 89.3% |

The Gate/Up readers use 52–54 registers and the Down reader uses 44.
All three use zero stack, local and shared bytes. This QFN-width
sweep passed byte parity and measured 15 of 21 Gate/Up and all seven
admitted Down formats at 80% or above on both RTX 3090s.

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

The wide IQ4_NL Down reader now prepares the reordered codes and original
scales once per CTA before entering its ordered eight-packet dot. Its
Release medians are 8.448–8.512 µs and 80.3–80.5% on both cards. The
64-column CTA uses 43 registers, 288 shared bytes, and one setup barrier.
Static inspection found zero stack and local bytes.

The mixed IQ4 preflight executes both Gate/Up scale orderings against the
serial-row result. Before and after the shared lookup extraction, the
compiled ROCm residual reader instruction bytes match for gfx906 and gfx1100.

The wide Down program stages the canonical IQ3 lookup table once per
64-column tile. The 256- and 512-entry tables consume 1 and 2 KiB of shared
memory respectively, and the two kernels require 38 registers with zero
local bytes. IQ1_S and IQ2_XXS instead unroll two consecutive K32 packets,
keeping the serial accumulation order. The Release replay gives these
31-sample median ranges across the two RTX 3090s.

| Source | Wide Down, µs | Route-matched weight rate |
| --- | ---: | ---: |
| IQ1_S | 9.792–10.176 | 46.5–47.1% |
| IQ2_XXS | 8.448–8.768 | 52.6–53.0% |
| IQ3_XXS | 10.112–10.304 | 57.0–57.1% |
| IQ3_S | 12.928–13.120 | 49.8–52.0% |

The same sweep measured 13 of 21 formats at 80% or above in both wide
leaf families on both cards. IQ4_XS is close to the Gate/Up threshold,
and its classification varies between runs. The wide all-format
production replay checks the two IQ3 shared-memory sizes and exact output
bytes across repeated launches.

The IQ1_M Down tile prepares the four signed activation correction
combinations for each of the two K16 halves before the K32 loop. The
prepared values occupy 256 bytes of shared memory per CTA and retain the
original rounded operations. The IQ3 hot loop uses a packed sign mask and
carry-free byte addition. A compile-time check compares every original
IQ3_S and IQ3_XXS grid entry under all 16 sign patterns with scalar bytes.

| Source | Gate/Up with publication, µs | Matched rate | Down, µs | Matched rate |
| --- | ---: | ---: | ---: | ---: |
| IQ1_M | 20.416–20.480 | 50.0–50.2% | 10.368–10.496 | 44.5–45.1% |
| IQ3_S | 24.512–24.576 | 82.6–82.8% | 11.200 | 60.6% |
| IQ3_XXS | 19.648–20.224 | 73.7–75.6% | 8.832–9.024 | 60.9–61.0% |

These are the Release medians of 31 samples on each card. Both cards passed
the 21-format serial-byte replay. The three Down variants use 38–40 registers
and zero local bytes. In this sweep, 12 formats cleared 80% in both stages
on both cards. The Release build also passed the 11 focused CUDA preflight
tests and the mixed CUDA/CPU movement token regression.

At M=4, top-K=8, N=2560 and K=256 on SM86, four Down formats use
256-column CTAs. Each CTA publishes one complete route tile and retains
the original K32 accumulation order. IQ3_S loads its 2 KiB table once
per 256 columns. The other three tiles use zero shared bytes. All four
use 36–40 registers with zero local memory. Both RTX 3090s passed the
full 21-format serial-byte replay with a native grid of 10 × 32 × 1 for
these tiles.

| Source | Release Down, µs | Route-matched weight rate | Complete capture gain, µs |
| --- | ---: | ---: | ---: |
| IQ1_S | 9.664–9.728 | 46.7–47.0% | 0.324–0.448 |
| IQ2_S | 10.240–10.304 | 50.6–50.9% | 0.320–0.576 |
| IQ2_XXS | 8.384 | 51.9% | 0.128–0.448 |
| IQ3_S | 10.048–10.112 | 64.6–65.0% | 0.064–0.192 |

The complete capture differences use both paired run orders on both cards.
Fourteen of 21 formats reached 80% on both cards in each of the wide Gate/Up
and Down sweeps from this build.

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

The tuned-first comparison used the narrower Release specialization. Its
median was 43.73 tokens/s, followed by 44.20 for the archived control.
All six histories and MTP work counts match the prototype comparison.
The real verifier captures in these runs borrow a device-owned row count.
Their graph inventory shows the original split projection and reduction on
both participants, so the fixed-row leaf measurements stand on their own.

### Device-counted vocabulary projection

The QFN main verifier records directly into the final retained timeline.
Its device controller publishes the live row prefix. A native CUDA IF node
submits one vocabulary tile per 32 columns for a positive count and skips
the projection for an empty count. The projection resolves operands before
the packet loop. Unused MMA lanes borrow a live activation row and the
epilogue publishes only the live prefix.

The vocabulary replay tests full, partial, empty, and resumed counts through
the same executable. It also checks the split publication of independently
recorded fragments that are eligible for child graph import.

The real-weight probe kept the same 16-row capacity across these replays
and timed 31 consecutive launches per publication. Both recordings use the
same Release executable and original 675,430,400-byte Q8_0 weight. All live
logits matched serial bytes and the poisoned inactive output bytes survived.
The reference is 879.53 GB/s on each card.

| Live rows | Original counted stage, µs | Arithmetic-only counted stage, µs | Streaming reference |
| ---: | ---: | ---: | ---: |
| 0 | 34.155–35.163 | 6.706–7.102 | |
| 1 | 966.028–968.638 | 775.763–776.192 | 98.9–99.0% |
| 5 | 1036.650–1044.480 | 787.456–787.852 | 97.5% |
| 8 | 2021.890–2027.900 | 797.035–798.389 | 96.2–96.4% |
| 16 | 3072.905–3090.595 | 819.712–824.270 | 93.2–93.7% |

The compiled counted MMA kernel uses 52 registers, zero shared and local
bytes, and no warp synchronization. The empty stage timing includes the
activation producer and conditional control.

The matched 2×CUDA/2×CPU QFN Release workload used fixed draft depth seven.
Each arm measured three 256-token requests after warmup. Both orders matched
every generated token and the MTP work per request, including 57 verifier
transactions, 456 verifier rows, and 399 drafts.

| Run order | Previous verifier, median tokens/s | Fused verifier, median tokens/s | Gain |
| --- | ---: | ---: | ---: |
| Fused first | 43.547 | 44.552 | 2.31% |
| Control first | 43.488 | 44.155 | 1.53% |

The control-first Release build also passed the mixed-tier movement
regression with greedy and stochastic sampling and all 14 focused CUDA
preflight tests.
