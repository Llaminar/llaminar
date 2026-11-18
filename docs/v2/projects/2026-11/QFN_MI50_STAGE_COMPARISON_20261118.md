# Four-MI50 QFN inference phases against Castagna Veloce

With 160 routed and 160 shared expert columns per GPU, Llaminar leads the
pinned Castagna build in whole-model prefill and decode on both prompts
with MTP off, at fixed depth three and with adaptive MTP. Llaminar also
passed the MTP own-history check in both modes on both prompts. The local four-row HC/FFN body
remains slower than Castagna. The comparisons below distinguish work per
verification from output tokens per second.

## Matched workload

Both engines ran the original four UD-Q4_K_XL shards of Qwen3.8-Flash-Next
and, for speculation, the shared Q8_0 predictor. They ran tensor sharding on
the same four 60-CU gfx906 cards behind one switch (PCI buses 8c, 91, 97, 9a)
with FP16 KV, RAM ngrams, an empty prompt cache, greedy generation and exact
matched input IDs. Reported throughput is the median of three unprofiled
256-token generations after warmup. Decode counts the 255 tokens after the
first. Native timings come from separate rocprofv3 traces.

The initial Llaminar core SHA256 is
`731498df6d777a77372aa68dcff49b9ad67fb26e5f3822744057e2c45f4c800b`.
Castagna is revision `3483d462715d61f21ed5877836be35a724446359`.
Each profile reproduced its own unprofiled token history and draft work.
The two engines generate different histories even without MTP. At fixed
depth Llaminar matched all six of its ordinary histories. Castagna diverged
from its own ordinary decoded history at review token 224 and code token 14
(zero-based). Routed work can therefore differ between engines.

## Fixed depth three on review

The table gives mean per-device native time in milliseconds. For the two
phase totals, overlapping dispatch intervals on one device count once. Stage
breakdowns sum named kernel durations. Each number uses its own physical
per-round or per-draft count. All four devices contribute equally.

| Boundary | Llaminar | Castagna | Llaminar minus Castagna |
| --- | ---: | ---: | ---: |
| Complete verifier, per four target rows | 25.056 | 16.970 | +8.086 |
| HC, projections, activation preparation, routing and experts | 19.221 | 12.616 | +6.605 |
| Target peer reductions | 2.886 | 2.539 | +0.347 |
| Gated delta net and convolution | 1.357 | 0.962 | +0.395 |
| Attention, indexer and RoPE | 0.707 | 0.234 | +0.473 |
| Target vocabulary projection | 0.284 | 0.313 | −0.028 |
| Draft prediction, per proposed token | 0.988 | 0.977 | +0.010 |
| Accepted-state publication, per verification | 1.133 | — | — |

Llaminar completed 74 verifier rounds, proposed 222 tokens and accepted 182.
Castagna completed 76 rounds, proposed 228 and accepted 181. All 76 native
target head boundaries contained exactly 96 peer reductions on each card.
Castagna issues those reductions outside its captured compute graphs.
Counting only the repeated captured portion gives 14.35 ms for the target
and 0.61 ms for the draft, substantially understating its cost. The complete
target column includes the two eager rounds. The complete draft column
includes all work on the fork's draft queue between target boundaries.
Llaminar's retained graph markers identify verification, draft and accepted
state publication separately. The fork trace has no independently verified
accepted-state publication interval.

The previous Llaminar verifier took 28.75 ms. The selected group of changes
saved 3.70 ms (12.9%). Its HC/projection/router/expert subtotal dropped from
23.21 to 19.22 ms. Ordinary projection kernels, routed experts, and grouped
routing still use 4.92, 5.10 and 4.14 ms respectively per verifier. Castagna
fuses these roles, so comparison below the combined subtotal requires a matched
operator replay. Each Llaminar captured verifier issues 2,295 dispatches
versus 485 in Castagna’s captured target compute. The complete Castagna target
queue issues 588 dispatches per round on PCI 8c.

For the 512-token prefill, the first-to-last observed device work spanned
510.04 ms in Llaminar and 607.08 ms in Castagna. Mean dispatch interval
unions were 508.61 and 349.04 ms. Castagna used physical input batches of
320, 188 and 4. Llaminar used one 512-row prefill. The profiler perturbs host
submission and device overlap. Use the unprofiled throughput for end-to-end
speed comparisons and separate native intervals for attribution.

## Ordinary decode on review

Each trace has all 255 post-prefill vocabulary boundaries on all four devices.
The table gives mean device-active milliseconds per token for each stage.
The 36 fused GDN input projections per token are charged to the
HC/projection group. Their native symbol contains the word “attention”.

| Stage | Llaminar | Castagna | Llaminar minus Castagna |
| --- | ---: | ---: | ---: |
| HC, projections, activation preparation, routing and experts | 8.219 | 6.747 | +1.472 |
| Attention, indexer and RoPE | 0.650 | 0.204 | +0.446 |
| Gated delta net and convolution | 0.745 | 0.600 | +0.145 |
| Vocabulary projection | 0.247 | 0.232 | +0.015 |
| Communication | 1.969 | 2.439 | −0.470 |

This identifies two distinct tuning opportunities. The verifier and ordinary
decode both spend more GPU time on HC/projection/routed expert computation.
Ordinary decode already leads on interconnect kernel time. For the 512-token
prefill, 49 gather kernels, 48 ordered folds and 49 FP16 reductions account
for most of Llaminar's 81.57 ms of communication kernel activity. Castagna's
communication kernels take 10.43 ms. The pinned fork defaults to 16-bit staging for large reductions and enables
peer DMA. Both ordinary kernel traces omit copy records. Compare actual wire
formats, copy events and payloads before interpreting the 71 ms gap as
communication latency.

## Unprofiled comparison

Each pair of numbers is prefill / decode in tokens per second.

| Mode and prompt | Llaminar | Castagna first control | Castagna quiet repeat |
| --- | ---: | ---: | ---: |
| Off, review 512 | 1038.42 / 86.92 | 878.32 / 86.17 | 876.82 / 86.38 |
| Off, code 2048 | 1008.12 / 80.63 | 1003.65 / 75.93 | 1003.40 / 75.48 |
| Fixed 3, review 512 | 999.16 / 118.68 | 774.28 / 114.89 | 785.74 / 117.62 |
| Fixed 3, code 2048 | 970.95 / 97.16 | 888.35 / 106.31 | 892.87 / 108.39 |
| Dynamic, review 512 | 1002.89 / 116.99 | — | — |
| Dynamic, code 2048 | 970.22 / 95.08 | — | — |

The first Llaminar fixed-depth run measured 108.91 and 95.05 decode tok/s.
Its code prefill samples of 683, 643 and 968 tok/s overlapped the Release
build. After the build finished, the same executable and settings produced
970.17, 970.95 and 971.00 tok/s with identical histories and draft work.
The table uses this repeat. A separate Castagna run during the build measured 85.71 and 82.16 decode
tok/s. The quiet fork repeat recovered to 117.62 and 108.39. Against that
repeat Llaminar leads by 0.90% on review decode and trails by 10.36% on
code decode. With the verifier improvements disabled on the same quiet
Llaminar ELF, decode measured 105.96 and 87.72 tok/s. The two
ordinary Castagna controls agree within 0.6% on all four rates. The ordinary
code prefill lead is about 0.5%, so a distribution over independent runs is
especially important there. The fork's fixed-depth code requests required
78–81 verification rounds versus Llaminar's 86. Report both per-round
kernel cost and total rounds for this prompt.

## Fixed-depth follow-up

A second Release image compares three additional changes on top of the complete
previous verifier policy. It uses register-based router top-K, a fused short-context
identity-indexer preparation, and vector GDN accepted-state publication. The
two arms loaded the same executable and core
`23fc618630bad1dc69b7b816e67248295ecf5c600504cebd43a3ecc5460f16e0`.
All six measured outputs matched the ordinary same-image generations. All
five draft and verifier work counters matched the fixed-depth control in each
iteration. Both arms passed driver and retirement checks before the host reboot.

| Fixed 3, unprofiled | Control prefill / decode | Candidate prefill / decode | Castagna prefill / decode |
| --- | ---: | ---: | ---: |
| Review 512 | 1002.48 / 119.20 | 1002.74 / 122.94 | 785.74 / 117.62 |
| Code 2048 | 970.54 / 97.96 | 970.94 / 100.20 | 892.87 / 108.39 |

The candidate gained 3.14% and 2.29% in decode over its control. Review is
4.53% above the quiet fork median. Code is 7.55% below it and needs another
8.17% throughput gain to reach parity. The control and treatment executed
74 review and 86 code verifications per generation. After reboot an independent, unprofiled, same-boot pair measured
119.56 → 123.22 decode tok/s on review and 97.65 → 100.13 on code.
Its prefill rates were 1001.64 → 1000.68 and 969.79 → 970.06 tok/s.
Both fixed-depth arms reproduced all six same-layout serial histories and
all five MTP-work counters per generation.

The post-reboot dynamic comparison also retired cleanly. Review prefill/decode
moved from 1000.95/115.85 to 1001.90/119.30 tok/s. Code moved from
969.63/94.23 to 969.79/97.02 tok/s. All six generated histories and all five
work counters matched. Review executed 58 verifications and 294 draft steps,
and code executed 69 verifications and 345 draft steps. The corresponding
unprofiled evidence is in `boot3-r1-dynamic-{control,phase}` beside the fixed
evidence. A matched dynamic fork measurement remains unavailable.

Four fresh ROCProfiler captures passed admission, token-history, verifier-work,
driver and retirement checks. The profiler accepted both original input lengths.
All timings below are milliseconds, averaged over four physical participants.
The verifier and accepted-state columns each count dispatch overlap once per
device per round. Draft time uses each engine’s actual number of proposals.

| Prompt and engine | Verifier | HC, projections, router and experts | Draft per proposal | Accepted-state publication |
| --- | ---: | ---: | ---: | ---: |
| Review control | 24.961 | 19.119 | 0.990 | 1.132 |
| Review candidate | 24.782 | 18.881 | 0.997 | 0.562 |
| Review Castagna | 16.970 | 12.616 | 0.977 | — |
| Code control | 26.545 | 19.439 | 1.036 | 1.073 |
| Code candidate | 26.391 | 19.171 | 1.039 | 0.546 |
| Code Castagna | 18.222 | 13.033 | 1.208 | — |

Every Llaminar participant executed 48 router top-K kernels and 36 GDN
accepted-state updates per round in both arms. Register top-K reduced native
time for those 48 calls from 1.037 to 0.779 ms on review and 1.029 to
0.783 ms on code. Vector GDN publication reduced the 36 updates from 0.901
to 0.331 ms on review and 0.843 to 0.317 ms on code. The candidate launched
12 fused short-context indexer preparations per review round and zero on code.
The two Llaminar arms executed 74 review rounds and 86 code rounds. The
corresponding Castagna traces contain 76 and 81 rounds. The largest
remaining verifier subtotal on code is HC, projection, router and experts,
with a 6.14 ms gap.

The 2048-token prefill dispatch unions were 2109.4 ms for the candidate
and 1249.2 ms for Castagna. These are GPU-active times under instrumentation.
The independent unprofiled comparison above measures prefill as users
experience it. The fork’s 672 large reduce-and-distribute operations on
each card include 4,032 runtime copy kernels. Counting those kernels gives
35.32 ms of visible collective work per card. The elapsed spans of its large
transactions averaged 329.63 ms per card. Llaminar’s visible collective
kernels took 330.27 ms per card. Kernel duration and transaction spans
measure different things. Neither trace records copy-engine counters.
The source algorithms imply 15.426 GB of directed collective payload for
Llaminar and 6.052 GB for Castagna. Llaminar transports exact FP32 rank
inputs for its ordered expert fold. Castagna rounds its large reduction
inputs and final owner shards to BF16. These byte counts exclude transport
overhead and additional host legs.

On ordinary review decode each Llaminar GPU executes 12 attention producers
per token, while each Castagna GPU executes six. The pinned fork rotates pairs
of whole GQA-group owners. Llaminar divides query heads across four participants
in each layer. Both perform 72 query-head/layer instances per GPU. Mean
device-active time therefore answers a different question from elapsed time
across the active participants. [Source comparison and ownership](../2026-10/QFN_CASTAGNA_STAGE_COMPARISON.md)

## Which TP participant finishes each FFN layer last

The retained review trace joins each layer's last FFN producer on the common
rocprof clock across all four cards. A verifier marker closes each round. The
join authenticated 48 FFN completions per card in each of 74 rounds. The
physical GGUF shard inventory identifies one 256-column routed shard and three
128-column shards per layer. The wide owner rotates between cards.

**The 256-column owner published last in all 3,552 comparisons, with no ties.**
For each layer we took the median difference between its completion and the
latest 128-column completion across the 74 rounds. Those medians range from
20.80 to 40.59 µs. Their median is 38.01 µs. Subtracting each participant's
own preceding peer-reduction end before comparing widths gives 38.04 µs.
This identifies the participant to accelerate before the following collective.
It does not establish the savings a particular kernel will deliver.

The narrower FFN fusion reached 31, 34, 32 and 32 layers on the four cards.
The other unselected 128-column sites use different weight formats. On a
separate same-ELF whole-model test with only the grouped HC-up and convolution
policies already enabled, the fusion removed 217–238 native graph nodes per
M4 verification. Review decode changed from 107.32 to 107.19 tok/s and code
from 88.47 to 88.79 tok/s. All six token histories and work counters matched.

A paired native review trace measured 1.76–1.91 ms less work per verification
on each card in the selected narrow FFN suffixes, a 38.1–38.8% reduction.
Before fusion the wide participant was last in 3,546 of 3,552 layer-rounds;
afterwards it was last in 3,550. The typical wide-to-last-narrow gap grew
from 17.25 to 74.43 µs. This is the same narrow-only A/B and has its own
control, distinct from the later 38.01 µs baseline above. Cutting work on
the earlier arrivals explains how an isolated speedup can leave request
throughput nearly unchanged.

A fusion on the 256-column Q4_K/Q5_1 slice measured 227.85 µs staged and
227.73–227.80 µs fused in the complete local FFN graph. It left the widest
participant with the same amount of work. All three original
expert format pairs in this model use Q5_1 or Q8_0 down weights with
32-column source groups. Four 160-column
intervals therefore cover the original 640 columns exactly.

The same MI50 measured complete local operations on the original wide
participant and on its proposed 160-column interval. Each layout passed
serial-row byte equivalence on its own original weights and routes.

| Original layer and routed gate/up, down | Retained ordinary, F256 → F160 (µs) | Staged four-row verifier, F256 → F160 (µs) |
| --- | ---: | ---: |
| L1 Q4_K, Q5_1 | 69.854 → 62.822 | 212.350 → 178.703 |
| L2 Q5_K, Q8_0 | 71.419 → 65.522 | 256.133 → 231.298 |
| L4 Q4_K, Q8_0 | 68.779 → 60.322 | 209.851 → 174.223 |

The other three participants gain 32 columns each under the proposed layout.
The complete four-card measurement must account for those increases, peer
waits, and any change in routing caused by regrouping the FP32 partial sums.
Each layout needs an independent frozen-history accuracy panel and an ordinary
generation control for its own MTP output.

The native-byte source test checked 401,100,800 bytes at each of TP1, TP2,
TP4 and TP8 for both layouts. It compared routed and shared gate, up and
down slices with the original GGUF, sampling one routed expert per layer.
Both layouts passed. Candidate TP2 uses two 320-column intervals, TP4 uses
four 160-column intervals, and TP8 retains rotating 64/96-column intervals
to preserve whole native blocks.

The four-MI50 Release PyTorch panels also passed independently on the same
16 frozen histories, covering 16 prefill and 96 teacher-forced decode rows.
Each row compares all 248,320 final logits. All 48 layers reported the
expected routed column widths on all four physical participants. The
candidate includes modified grouped verifier kernels, which teacher-forced
ordinary decode does not exercise.

| Layout and phase | Mean KL, nats | Top-1 agreement | Mean top-5 overlap |
| --- | ---: | ---: | ---: |
| Original, prefill | 0.01432 | 93.75% | 96.25% |
| Equal width, prefill | 0.03150 | 93.75% | 92.50% |
| Original, decode | 0.02211 | 94.79% | 91.04% |
| Equal width, decode | 0.01928 | 95.83% | 91.67% |

Ordinary review attention also illustrates the importance of measuring across
participants. The union across GPUs is 0.738 ms/token for Llaminar and 0.598
for Castagna. Attention was the only active kernel category for 0.565 and
0.008 ms/token respectively. In Castagna, attention overlapped peer pushes
alone for 0.337 ms/token. GPU intervals cannot distinguish a peer kernel
moving useful data from one waiting for an arrival.

## Equal-width Release qualification

The same Llaminar Release ran the original and equal-width policies on the
same quartet. Both passed the two accuracy gates above. Prefill and decode
throughput below are in tokens per second. The fixed and adaptive Llaminar
arms each matched all three 256-token histories from their own layout's
ordinary decode on both prompts. Equal-width code fixed-3 took 73 rounds
with 183 accepted draft tokens. The original split took 86 rounds with 170
accepts. Review used 76 versus 74 rounds with 180 versus 182 accepts.
The different work explains part of the change in delivered throughput.

| Engine and policy | Review prefill / decode | Code prefill / decode |
| --- | ---: | ---: |
| Llaminar original, off | 1039.52 / 86.79 | 1008.07 / 81.47 |
| Llaminar equal, off | 1082.65 / 88.71 | 1051.46 / 82.49 |
| Castagna off, fresh quiet control | 870.45 / 85.59 | 1003.97 / 75.03 |
| Llaminar original, fixed 3 | 998.76 / 121.50 | 969.40 / 99.85 |
| Llaminar equal, fixed 3 | 1042.02 / 134.08 | 1009.10 / 132.14 |
| Castagna fixed 3 | 785.74 / 117.62 | 892.87 / 108.39 |
| Llaminar original, dynamic | 999.96 / 119.15 | 968.52 / 96.82 |
| Llaminar equal, dynamic | 1040.70 / 120.95 | 1009.24 / 134.75 |
| Castagna adaptive | 672.70 / 70.71 | 794.71 / 79.81 |

Both adaptive controllers were allowed 1–15 draft tokens. Castagna used
its native time-based depth policy. Its six measured adaptive outputs
differed from its own MTP-off outputs. Each controller's timing includes
its actual selected depths, proposed tokens and accepted work.
[Adaptive comparator and per-request first difference](/tmp/qfn-mi50-fork-dynamic/quiet-r1.json)

The equal-width reference core is
`baa2b17af74a2711a37b16c97c6a30c11c60922ff4b262e40310bbac194a72ce`.
It includes the experimental native gate/up, route, router and identity-indexer
policies in its admission receipt. This is the tested binary. A clean build
of the feature branch needs its own throughput result. The original accepted
files, MPI teardown, driver window and physical VRAM retirement passed for
each timing arm. The first fixed-3 candidate run failed the comparator artifact
check; only its clean repeat contributes to the table.

The standalone cohort runner takes a fresh run name and its frozen Release
directory. It holds the physical roster for the numerical panel, own-Off
controls and both MTP modes. On this benchmark host the full invocation is

```bash
cohort=/tmp/qfn-mi50-f160-frontier-balanced-sol62
lock=/tmp/qfn-mi50-retained-real-tp4/r7-all-tp-live/full-model/grouped-mtp-suite.lock
cd "$cohort/parity"
flock -x "$lock" bash run-layout-cohort.sh execute checkpoint-r2 "$cohort/frozen-release"
```

[Matched controls and exact sample histories](/tmp/qfn-mi50-f160-frontier-balanced-sol62/parity/combined-boot3-r1-summary.json) ·
[Feature switches and reproduction commands](/tmp/qfn-mi50-f160-frontier-balanced-sol62/DISPATCH.md)

## Four-row HC and FFN on one MI50

The shipping Castagna kernel can be measured independently of its source
rebuild. On the wide slice of layer 1 it used 60 blocks of 1,024 threads,
64 VGPRs per lane and 51,240 bytes of shared memory per block, with zero
private memory. It executed the local HC and FFN together. Across all 81
verifier rounds of code2048, layers 1–47 used that fused path followed by
a separate peer reduction. Layer 0 issued separate routed kernels. Using
layer 0 as the timing reference would omit expert work.

The harness used original QFN layer-1 weights, 256 routed columns, 192
shared columns and identical seeded input rows. Times are microseconds per
complete captured operation. Each timing arm had 16 samples of 32 operations.

| Implementation | Three rows | Four rows |
| --- | ---: | ---: |
| Castagna deployed gfx906 object | — | 156.27 |
| Castagna source rebuild in the paired harness | 126.36 | 156.71 |
| Llaminar, reuse a loaded weight for two rows selecting the same expert | 221.72 | 256.42 |
| Llaminar, assign work directly to the selected route slots | 206.25 | 269.66 |
| Llaminar, load 16 adjacent weight columns and transpose partials before the ordered fold | 234.22 | 268.19 |

The two Castagna entries agreed byte-for-byte on all five output planes.
Each Llaminar variant passed 40 comparisons against its own serial oracle
across 12 output planes. The paired fixture selected 26 distinct experts in
40 slots. The separate 160-column native fixtures selected 35–39 across three
layers. A second matched input came from the canonical PyTorch model at layer 1,
using four teacher-forced rows following the original nine-token prompt.
Castagna selected 30 distinct experts across 40 slots. Its deployed body
took 161.45 µs and Llaminar took 263.61 µs. Packing affine scale and minimum
together in advance increased Llaminar time to 284.35 µs. All 40 native
comparisons passed on each arm. [Reference input planes and provenance](/tmp/qfn-mi50-real-m4-activations/manifest.json)

Both microbenchmarks have a single resident layer.
The wide serving rank took 188.48 µs for the Castagna operation in its trace.
Actual model traffic and layer transitions matter. The previously attained
68 µs result belongs to single-row decode.

With equal routed and shared widths of 160, Llaminar's captured four-row
FFN took 150.02 µs for Q4_K/Q5_1, 138.17 µs for Q4_K/Q8_0, and 152.79 µs
for Q5_K/Q8_0. The corresponding staged controls took 179.00, 178.43 and
235.64 µs. These intervals include the router and FFN. They exclude the HC
prefix and four-device communication. Castagna's original program requires
widths divisible by 64. Its 256-column result measures a different operation
from these 160-column intervals.

The routed gate/up phase on matched IDs took 77.36 µs in Llaminar versus
27.12 µs in the fork on the original wide fixture. The next grid join took
12.56 and 12.96 µs. Llaminar stages native affine coefficients through LDS and
synchronizes the participating threads before using them. The next layout
experiment moves lossless coefficient packing into weight preparation and
keeps the ordered FP32 reduction. Its admission contract must preserve the
single authoritative expert bank when weights migrate.

[Shipping binary, source slices and paired replays](/tmp/qfn-mi50-fork-m4-native/REPORT.md) ·
[Matched routed-phase census](/tmp/qfn-castagna-family-sprints-r61/G08/experiments/12-retained-body-grouped/e15-matched-gu/RESULT.md) ·
[Native 160-column timings and traffic counters](/tmp/qfn-mi50-f160-roofline-c2-sol62/REPORT.md)

## Counted verifier format qualification

The code2048 fixed-three trace of the equal-width Release above completed
73 verifier rounds with 183 accepted tokens and reproduced all 256 generated
IDs from both of its own ordinary and fixed-depth controls. Per GPU the
complete target occupied 22.94 ms per four-row round. The earlier layout
took 26.39 ms. The fork occupied 18.22 ms across its 81 rounds. The
equal-width draft occupied 1.026 ms per proposed token versus 1.208 ms
for the fork. The original source inventory and native launch geometry
confirmed 43 Q4_K/Q5_1, four Q4_K/Q8_0 and one Q5_K/Q8_0 leaf pair per
participant per round.
[Native target/draft timing](/tmp/qfn-mi50-current-stage-pair/qualified-f160-code-candidate.json)
[Per-layer native census](/tmp/qfn-mi50-f160-fixed3-profile/boot3-r3-candidate-census.json)

The fixed-row FP16 router selects its M4, M7 or M8 program during graph
preparation. The kernel consumes the original device-owned live count, with
the same half rounding and FP32 reduction tree as counted M1. On one MI50
with the original layer-1 weights and F160/S160, the local M4 graph measured
147.6635, 139.4655, 139.4585 and 147.6790 µs across control, candidate,
candidate and control. The projection fell from 26.08 to 17.76 µs at the
same 512×128 launch geometry. Both arms used the same register top-K kernel.
The production preflight replays passed with partial and zero live rows,
deferred publication and dirty inactive buffers on the exact Release core.
These measurements cover one participant's graph. The switch
`LLAMINAR_QFN_ROCM_GROUPED_FP16_ROUTER_FIXED_ROWS=1` selects the candidate.
[Native source, paired windows and run commands](/tmp/qfn-mi50-fixed-router-canonical-sol62/canonical/REPORT.md)

The exact K32 reuse experiment used original-source four-row F160/S160
fixtures on MI50 PCI c2. Each row below is a median in microseconds.
Its two control windows bracket the treatment in the same Release.
Each replay compared the live results with serial M1, including deferred
routing and replacement of the expert bank. This candidate does not alter
quantized weights, scales or floating-point partial order.

| Original routed gate/up and down | Control 1 | Selected path | Control 2 |
| --- | ---: | ---: | ---: |
| Q4_K and Q5_1 | 150.230 | 135.361 | 150.294 |
| Q5_K and Q8_0 | 150.139 | 145.894 | 150.081 |
| Q4_K and Q8_0, ordinary sum path | 135.024 | 135.119 | 134.821 |

The third case uses the original producer. Its separate wave-register
candidate rose from 136.36 to 138.76 µs. The immutable experiment core
was `6f9f21a8a59ef5eddb2110b37b2cb9265ea090f862b61b0fab90ff028882b1d4`.
The full-model and benchmark-only source combinations are distinct scopes.
[Native fixture qualification](/tmp/qfn-mi50-f160-affine-sums-sol62/frozen-release-narrow-c2/qualification.json)

The four-MI50 full-model A/B changed only
`LLAMINAR_QFN_ROCM_GROUPED_FFN_AFFINE_SUMS` in the same Release. Each
cell contains the median decode throughput after prefill across three
256-token generations, in tokens/s. All six arms retired cleanly. Both
policies produced the same tokens on both prompts and matched the verifier
work counters at each MTP setting.

| MTP | Review, 512 tokens, off → on | Code, 2,048 tokens, off → on |
| --- | ---: | ---: |
| Off | 87.824 → 87.810 | 82.399 → 82.440 |
| Fixed depth 3 | 133.935 → 137.055 | 131.859 → 135.338 |
| Dynamic | 121.091 → 121.158 | 134.684 → 134.592 |

The same-ELF timestamp pair on code2048 fixed depth 3 contains 73 verifier
rounds per arm, with 183 accepted drafts and identical 256-token histories.
The dispatch union per device fell from 25.077 to 23.450 ms per round.
The span from the earliest dispatch to the last completion across the four
cards fell from 26.640 to 25.040 ms. Gate/up fell from 2.816 to 2.779 ms
and down/fold from 2.582 to 2.025 ms. Of the 0.928 ms decrease in peer
kernel residence, 0.922 ms occurred before the last participating GPU
entered the matching transfer. These observations share a timeline and
must not be added to the independently measured end-to-end improvement.
Castagna's earlier code2048 dispatch-union reference was 18.222 ms.
[Matched timestamp pair and per-PCI attribution](/tmp/qfn-mi50-k32-affine-model-gate/timestamp-trace/c2-code2048-same-elf-pair-r1.json)

The fixed-depth improvement measures 2.35% and 2.65% on paired trials.
Dynamic MTP changed by less than 0.1% on both prompts. The standalone
branch regression exercised all three source combinations, both static
and deferred routing, and the expert bank epoch change in 108.11 s.
[Matched full-model results](/tmp/qfn-mi50-k32-affine-model-gate/affine-k32-c2-r1-summary.json)
[Registered replay result](/tmp/qfn-mi50-verifier-roofline/k32-branch-replay.json)

On the dominant Q4_K/Q5_1 layer the gate and down leaves measured 56.48
and 38.40 µs. Separate counter runs on the same card measured 19,938.0
and 11,876.625 KiB of physical fetches. A four-bank packet read using
the same geometry sustained 611.78 and 514.79 GB/s. Applying those rates
to the measured fetches gives 33.37 and 23.62 µs, or 59.1% and 61.5% of
the read-only probe's bandwidth. The full operations also perform format
decode, ordered arithmetic and publication. A 256 MiB linear stream
sustained 873.24 GB/s on this card. Cache-resident throughput and physical
fetch throughput use different byte counts.
[Physical counter calibration](/tmp/qfn-mi50-generic-m4-route-groups-c2/roofline-counters/counter-roofline-c2.json)

The Q5_K/Q8_0 gate and down measured 64.32 and 41.12 µs.
Their physical fetch rates reached 63.6% and 67.1% of their respective
packet probes. For Q4_K/Q5_1 the 120- and 160-workgroup issue probes used
the native 256 threads per workgroup and sustained 85.71 and 80.19 billion
vector wave instructions per second. At those rates the measured gate
and down instruction counts take 10.50 and 14.10 µs. The probes issue
independent packed dots. Real work also includes dependent floating-point
operations, address generation and synchronization.
[Measurement procedure](../../../../tests/v2/performance/kernels/qwen4exp/verifier-roofline.md)
[Per-resource results](/tmp/qfn-mi50-verifier-roofline/physical-resources.csv)

### Original shard widths and verifier preparation

The counted four-row reader resolves six serial-M1 reduction schedules from
the original weights during graph preparation. The shared gate, up and down
keep independent schedules, as do the three routed projections. A published
expert can change address or physical encoding when it moves. Its source
identity and reduction order remain fixed for the captured program.

The table gives median microseconds per captured local operation on MI50
`0000:ba:00.0`. Each run timed 12 windows of 32 replays. Each row used
the original QFN GGUF shard, four rows and 160 shared columns. Both readers
passed the serial byte comparison. The run order was separate, paired,
paired, separate; each table cell is the median of its two runs.

| Original routed gate/up, down | Routed columns | Separate reads (µs) | Paired gate/up (µs) |
| --- | ---: | ---: | ---: |
| L1 Q4_K, Q5_1 | 128 | 164.25 | 146.53 |
| L1 Q4_K, Q5_1 | 256 | 238.78 | 214.34 |
| L2 Q5_K, Q8_0 | 128 | 165.58 | 173.10 |
| L2 Q5_K, Q8_0 | 256 | 244.60 | 263.92 |
| L4 Q4_K, Q8_0 | 128 | 155.96 | 139.78 |
| L4 Q4_K, Q8_0 | 256 | 229.46 | 200.48 |

The graph selects paired reads for the measured physical Q4_1 gate/up and
Q8_0 shared projections when each pair has equal partition counts. The Q5_K
shards retain separate reads. Moving partition lookup to preparation alone
changed the L1 wide parent by less than 1 µs in a separate bracket.
[Timing receipts](/tmp/qfn-mi50-verifier-roofline)
[Source-window fixture](../../../../tests/v2/integration/loaders/GroupedFFNM4SourceMatrix.inc)

A separate same-ELF comparison covered 15 generated source families. The
direct reader reduced local parent time against the staged graph by a median
9.23%, with individual gains of 0.81%–26.10%. G02 (plain Q4_0) and G13
(IQ3_XXS/Q5_1 with IQ2_S shared) each gained less than 1%. G15 measured
326.94 → 271.18 µs. Static and deferred captures matched serial bytes for
all 15. These are local verifier timings; the four-device model measurements
above describe the separate equal-width, K32-sum experiment.
[Family timing CSV](/tmp/qfn-mi50-m4-source-matrix/parent-readers-families-abba-r1/result.csv)
[Captured replay results](/tmp/qfn-mi50-m4-source-matrix/parent-readers-families-parity-bd-r1/result.json)

The same reader passed 80 independent static captures, varying 20 encodable
original GGUF formats across routed gate/up, routed down, shared gate/up and
shared down. A separate prepared Q8_1 shared-down fixture retained source ID
20 with execution ID 19, and passed static and deferred replay. The fixture
also checks the existing rejection of Q8_1 as an original routed GGUF tensor.
[Complete independent source and role results](/tmp/qfn-mi50-m4-source-matrix/parent-readers-qualified-matrix.json)

Timestamp-only wide-layer traces attributed the format split to gate/up.
For Q4_K it took 126.88 → 105.44 µs, and for Q5_K it took 141.28 → 164.16
µs. All four recorded gate symbols used zero private scratch. The paired Q5_K
kernel used 80 VGPRs versus 84 separately, despite taking longer.
[Captured stage medians](/tmp/qfn-mi50-verifier-roofline/prepared-graph-stage-traces.json)

The packet probe also covered both original TP widths and each calibrated
format's secondary weight planes. On PCI c2, the separate gate/up reader
reached 48.5%–52.9% of its matched physical-read rate across the six original
shards. Q5_1 down reached 48.0%–54.4% and Q8_0 down reached 69.9%–85.2%.
In G15, IQ3_XXS routed gate/up with IQ2_S shared reached 36.9%, and Q6_K
down with Q8_0 shared reached 49.0%. For L1 wide gate/up the read-only probe
fetched 32,056.6 KiB versus 32,156.9 KiB in the numeric kernel. The probe
omits arithmetic, activation reads and publication. The raw cache-resident
control and the 256 MiB streaming control used independent physical counters.
[Packet benchmark procedure](../../../../tests/v2/performance/kernels/rocm/roofline/README.md)
[Measurements by PCI, geometry and source format](/tmp/qfn-mi50-parent-roofline-c2/score.json)

### Full-model fixed-depth result on the original TP4 shards

Release checkpoint `8e11d8880` used four MI50s, RAM ngrams and the original
128/128/128/256 routed split with 160 shared columns. The control and reader
used the same executable and model. Each performed one warmup followed by
three measured 256-token generations per prompt. Only
`LLAMINAR_QFN_ROCM_GROUPED_FFN_FRONTIER` changed between arms.

| Fixed depth 3 | Staged decode (tok/s) | Prepared reader decode (tok/s) | Paired change | Prepared reader prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| Review 512 | 114.892 | 124.581 | +8.44% | 1,002.452 |
| Code 2048 | 94.655 | 102.513 | +8.34% | 970.566 |

All six generated histories matched their own ordinary controls and both arms
agreed on all five verifier counters. Review took 74 rounds with 182 accepted
tokens. Code took 86 rounds with 170 accepted tokens. With speculation off,
the paired decode changes were −0.17% and +0.61%, respectively. The fork ran
the same input IDs but generated different histories and verifier work.
[Original per-trial counts and timings](/tmp/qfn-mi50-w-readers-model-gate/w-readers-r1-fixed3-summary.json)
[Cross-engine work comparison](/tmp/qfn-mi50-w-readers-model-gate/w-readers-r1-staged-historical-fork.json)

Dynamic MTP in the same two Release arms measured 116.077 → 116.221 tok/s on
review and 94.817 → 94.641 on code. Paired prefill changed by less than 0.3%.
The arms generated identical tokens and performed identical verifier work on
each prompt. At fixed depth, a separate native capture found 192 gate/up and
192 down reader nodes per GPU across four verifier graphs. The wider and
narrower source shards both executed the selected reader. The paired captures
retained the same peer-push count. [Dynamic per-trial results](/tmp/qfn-mi50-w-readers-model-gate/w-readers-r1-dynamic-summary.json)
[Native node and width inventory](/tmp/qfn-mi50-w-readers-model-gate/w-readers-r1-qualified-native-pair.json)

### Transferring the single-row loop to grouped verification

The single-row F256/S192 HC/FFN checkpoint measured 69.62–69.72 µs on three
runs. Its gate/up wave computes four output columns and sixteen independent
K partitions. Each lane appends five K32 blocks in order. Only the eight
packed dot operations *within* a K32 block are unrolled per projection. The
fold obtains later partition values as it consumes them. The verifier instead
works on sixteen columns and four partitions at a time across each wave,
with four visits to cover the same sixteen partitions. Both layouts can
produce identical serial sums. They differ in contiguous weight access,
exchange instructions and register lifetimes.
[Single-row contract and timings](QFN_MI50_RETAINED_HC_FFN.md)

The gfx906 scalar and vector instructions explain an opportunity beyond
integer dot throughput. In the closed single-row Q4 path, the gate role
borrows its original descriptor fields before the K loop. The up role still
has scalar field loads in the loop. Hidden codes and coefficients come from
the block's already populated LDS panels. In the four-row Q4 path, vector
loads fetch the routed plane pointers in the loop before their dependent
payload and scale loads. Each active wave uses one acquired expert slot.
Its plane pointers can therefore be captured in registers for the duration
of that job. The acquisition must repeat when the launched graph sees a
new bank epoch. Its immutable shared-expert descriptors arrive by value.

| Complete captured four-row parent on MI50, µs | F128/S160 | F256/S160 |
| --- | ---: | ---: |
| Typed reader at `d50b74c9a` | 140.26 | 207.65 |
| Physical codebook checks moved to publication | 142.77 | 210.84 |
| Prepared width/partition arithmetic plus preceding change | 141.98 | 210.92 |
| Paired packed gate/up words plus publication-time checks | 140.29 | 207.45 |

Each value is the average of the two process medians from a mirrored run
order on PCI ba. Each process timed 12 batches of 32 complete operations
from the original QFN L1 source. All arms passed the native byte checks. In
the routed Q4 inline-sum K32 loop, the paired reader reduces 32 static
`v_dot4_i32_i8` instructions to 24, including eight for the shared activation
sum. The same loop grows from seven to eleven static `s_waitcnt`
instructions. These counts describe compiler output; they do not measure
cycles spent stalled. The complete gate entry uses 62 VGPRs in the typed
and paired variants, versus 84 in the width/partition variant. All three
have zero private scratch. The paired reader showed no repeatable advantage
over the typed program across both widths.
[Mirrored timings and native receipt paths](/tmp/qfn-mi50-verifier-roofline/typed-verifier-abba.json)

Borrowing both routed descriptors' payload, scale and minimum pointers once
per wave was a larger improvement. On PCI 8c the complete Q4 parent took
141.35 → 133.03 µs at F128 and 207.88 → 191.12 µs at F256. The treatment
moved four scalar memory instructions before the K loop. In the inline-sum
backedge it removed two vector four-word and two vector two-word loads and
reduced static waits from eleven to eight. It allocated 70 VGPRs versus 62,
with 104 SGPRs and zero private scratch in both. Static and deferred native
replays passed on the original weights. This experimental build is separate
from the four-card throughput run.
An ordinary C++ field copy hoisted the same loads and used 86 VGPRs. It
improved the parent by only 0.49% at F128 and 0.54% at F256.
[Scalar and ordinary C++ field-borrow brackets](/tmp/qfn-mi50-q4-wave-field-borrow/RUN.md)

The initial hand-written address helper sign-extended the low 32 bits
returned by `readfirstlane`. The first experimental GPU run faulted and was
excluded. Explicit unsigned halves preserve all 64 pointer bits. A fresh
original control passed after the fault. The published bank can move between
executions, so the wave must obtain its pointers again after each acquire.
At F128 and F256, native replay relocated six actual planes over epochs 2,
3 and 4, changing the selected bank 0 → 1 → 0. Poisoning the retired
fixture allocations left serial byte equivalence intact.
[F128 relocation receipt](/tmp/qfn-mi50-q4-wave-field-borrow/moved-scalar-r2-f128/result.json)
[F256 relocation receipt](/tmp/qfn-mi50-q4-wave-field-borrow/moved-scalar-r2-f256/result.json)

Changing who owns a Q8 output block produced opposite results at the two
TP widths. Giving one wave all 32 columns kept SwiGLU values in registers
through quantization. It removed the two per-tile CTA barriers, reduced LDS
from 320 to 64 bytes and retained 62 VGPRs. The complete parent went from
206.98 to 199.13 µs at F256 and from 140.31 to 162.13 µs at F128. For
four live rows the new mapping supplies 85 and 45 workgroups respectively
on the 60-CU MI50. That smaller grid is a possible contributor to the
narrow-shard regression. A simpler owner-relative fold removed six static
cross-lane permutes and seven static waits, but changed full time by less
than 0.2% on either shard. Both experiments passed their serial byte checks.
[Wave-local publication bracket](/tmp/qfn-mi50-verifier-roofline/wave-retry-abba.json)
[Output-owner fold bracket](/tmp/qfn-mi50-verifier-roofline/owner-fold-pinned-abba.json)

The matched Q4 experiment also tested computing the exact signed K32 sums
in the existing router quantizer and publishing intermediate sums alongside
the routed activation. That helps the previously measured equal-width fixed
program. In this source-format program the complete F128/S160 parent took
140.31 → 141.44 µs, and F256/S160 took 207.30 → 210.62 µs. The trace shows
both arms executed five nodes and used their intended quantizer. In the wide
shard, Down took 56.48 → 59.84 µs despite the cheaper dot expression. The
source-format program continues to compute its sums inline.
[Same-binary timing arms](/tmp/qfn-mi50-verifier-roofline/uneven-activation-abba.json)
[Per-stage kernel inventory](/tmp/qfn-mi50-verifier-roofline/affine-c2/symbols.json)

The fixed-format Down kernel provided a stronger transfer from single-row
decode. Capture preparation certifies the original format, width and planes.
The current grouped route publication still authenticates the acquired bank
and epoch. The projection can then enter its native one-block-per-partition
loop directly. On real F160 shards the isolated change took 146.33 → 143.89
µs on L1, 162.25 → 148.28 µs on L2 and 151.07 → 136.69 µs on L4 for the
whole captured operation. L2's Down dispatch alone fell from 48.0 to 34.4
µs. All arms checked serial bytes and original expert routing.
[Fixed-pitch measurements and stage attribution](/tmp/qfn-mi50-fixed-bank-proof/REPORT.md)

With the separate Q5_K gate/up reader, specializing the pitch and original
five-block partition increased register use from 70 to 110–111 VGPRs and
still improved the complete parent at F128, F160 and F256. In the wide
case, physical fetches stayed within 0.01% and VALU wave instructions fell
3.75%. Both shipped architectures compiled with zero private scratch.
These measurements make register lifetime and dependent loads worth tracking
alongside allocation count. [Independent Q5_K receipts](/tmp/qfn-mi50-q5-fixed-span-c2/typed-w/REPORT.md)

A Release combining the fixed Down change with the separate Q5_K pitch
reader replayed both paths against the actual QFN weights on PCI c2.
F128/S160 L1 in the fixed program measured 149.78 → 136.50 µs. F256/S160
L2 in the source-format program measured 232.25 → 223.28 µs. Each bracket
used fresh processes in control, treatment, treatment, control order. Source
intervals were checked on every arm because the wide shard rotates by layer.
The control was `d50b74c9a` and the combined core SHA256 was
`2f2a80d37cf62422425c7380a1f0b7f3bbefe58a32c7c241a42697c36d254376`.
All ten focused production-preflight registrations passed on this combined
core. The two 160-column replays used private tmpfs for generated fixtures
and the registered 120-second limit. They took 97.52 and 101.06 seconds.
[Complete operation and byte-replay receipts](/tmp/qfn-mi50-verifier-roofline/combined-original-abba.json)
[Registered native gates and ELF identities](/tmp/qfn-mi50-fixed-bank-proof/canonical-w/combined-r1/qualified.json)

## Reproduction

The local evidence root for native phase attribution is
`/tmp/qfn-mi50-current-stage-pair`. It contains the read-only SQLite readers,
per-device classified results, native queue bounds and timing receipts. The
two original model trace roots retain the device mapping, loader, source,
token and GPU retirement receipts. Re-run analysis on those receipts with:

```bash
P=/tmp/qfn-mi50-current-stage-pair
D=/tmp/qfn-mi50-grouped-stack-final-r1
M=/tmp/qfn-mi50-retained-real-tp4/r7-all-tp-live/full-model
python3 "$P/phase-queues.py" review512 "$D/quartet-r1-fixed3-trace" \
  "$M/trace-fork-fixed3-r1" "$P/review-complete-replay.json"
```

The three-change unprofiled reproduction uses the frozen source and archive
members recorded beside the Release image. A fresh cohort name is required.

```bash
X=/tmp/qfn-mi50-phase-tuning-r2
flock -x "$M/grouped-mtp-suite.lock" bash "$X/run-phases.sh" execute replay-01
```

The clean pre-reboot evidence is under `$X/quartet-r1-fixed3-{control,phase}`.
The matching ordinary arm is `$X/quartet-r1-off`. The resumed dynamic
comparison used `$X/boot3-r1-{off,dynamic-control,dynamic-phase}`.
The independent fixed-depth pair used `$X/boot3-r1-fixed3-{control,phase}`.
The six measured generations and five verifier-work counters matched their
controls. The four new ROCProfiler runs can be reanalyzed with the same
per-round denominator and fork inputs used in the table.

```bash
for arm in baseline treatment; do
  for prompt in review512 code2048; do
    fork="$M/trace-fork-fixed3-r1"
    if [[ "$prompt" == code2048 ]]; then fork="$P/code-fork-r1"; fi
    python3 "$P/phase-queues.py" "$prompt" "$X/phase-native-boot3-$arm-$prompt" \
      "$fork" "$P/replay-$arm-$prompt.json"
  done
done
```

The repository analyzer reads the original ROCProfiler database, measured
verification count and authenticated physical PCI admission from each window.
It derives the final prefill and target graph boundaries directly from native
dispatches. Supply the matching per-PCI ordered source inventory for the run.

```bash
F=/tmp/qfn-mi50-grouped-ffn-frontier/quartet
layout=/tmp/qfn-mi50-ffn-other-sol62/per-pci-mapping.json
analyze=tests/v2/performance/kernels/qwen4exp/ffn_arrivals.py
python3 "$analyze" "$D/quartet-r1-fixed3-trace" "$layout"
python3 "$analyze" "$F/native-k4-control-review512" "$layout"
python3 "$analyze" "$F/native-k4-treatment-review512" "$layout"
```

The F160 component results, matched F256 measurements and harness commands
are in `/tmp/qfn-mi50-f160-sol62/REPORT.md`. The complete balanced Release,
source ownership contract and run commands are recorded in
`/tmp/qfn-mi50-f160-frontier-balanced-sol62/DISPATCH.md`. Its
`parity/combined-boot3-r1-panel-gate.json` ties both full-vocabulary panels
to the same core, measured widths, original GGUF and PyTorch references.
The prefill transport census and read-only trace queries are in
`/tmp/qfn-mi50-code2048-comm-attribution/REPORT.md`.
