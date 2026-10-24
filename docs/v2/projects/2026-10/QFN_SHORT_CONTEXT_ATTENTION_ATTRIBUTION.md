# QFN short-context attention attribution, 2026-10-24

## Empty-header coalescing191: rejected

Publishing the six neutral partition headers through six lanes of the first
wave leaves all 396 producer waves intact. The matched Q12/M1/post-RoPE
complete producer-plus-merge A/B/A screen measures **14.7900 versus14.756875
µs at512 (+0.224%)** and **15.5100 versus15.428125 µs at768 (+0.531%)**.
All outputs and numerical partials remain byte-identical. Four isolated
counter launches and their strict joins pass: scalar instructions per wave
fall from106.10101 to105.59596, only0.476%, while wave count remains396.
This does not isolate the cause of the slight slowdown. The candidate is
rejected without an unnecessary long-context or model rerun.

The qualified190 production source is restored. Rebuilt Release ISA proves
all six affected kernel bodies and resource tuples equal190, with spill
guards intact. The useful partial-head regression remains: both key encodings,
M1/M16, non-six-aligned head shards and long/short/inactive/partial/long dirty
replays check neutral metadata, untouched payload and exact outputs.
The canonical `V2_Integration_ROCm_QwenSparseFP16TileArithmetic` registration
passes all ten GTests in8.279seconds on restored190. Its private wrapper
initially misparsed ANSI color codes in case names; an independent audit of
the unchanged log proves the exact ten-case set, clean driver window and
normal retirement. The failed wrapper receipt remains preserved.

Evidence: `attention191-empty-header-r1/events-short-screen.json`, the four
`profile-*-producer-c512-*-retry-r2-joined.json` reports under that directory,
`attention191-retained-tests-integration-r1/ansi-normalized-audit.json`, and
`/tmp/qfn-attention191-restored-isa-r1/comparison.json`. This is focused native
regression coverage; it does not claim another complete preflight run.

## Remaining matched-width gap

The final matched Q12 producer-plus-merge measurement is **14.749183 versus
14.555000 µs at 512**, a reproducible **0.194183 µs / 1.334%** gap in this retained
window. At 768 it is **15.446250 versus 15.380963 µs**, or **0.065287 µs / 0.424%**;
that difference is comparable to the candidate's 0.0675 µs process-median span.
These spans are descriptive observations, not confidence intervals.

Both actual input owners use compact 512-byte KV rows, both have two kernel
nodes, and neither spills or records LDS bank conflicts. Different layout pitch
or launch count therefore does not explain this small remainder. Llaminar has
66 producer CTAs versus the fork's 24/36 at 512/768, with 50/42 empty-header CTAs.
That proves extra control work exists, but does not establish its elapsed cost.
The instruction counters cannot assign the remaining 0.194 µs among header
scheduling, merge metadata and instruction scheduling.
The dominant large discrepancy was local Q6 versus Q12 batching; the demonstrated
Fold12 improvement below removes most of the matched 768 gap. No further layout
rewrite is justified by these observations. Detailed follow-up:
`/tmp/qfn-attention-residual-followup-private/REPORT.md`.

Four fresh singleton profiler launches now close the unequal-width counter gap:
at matched Q12/context 512, Llaminar issues **396 wavefronts** versus the fork's
**96**, with the same 96 useful waves and 300 additional empty-header waves.
Scalar instruction issues per query head are **3,501.33 versus 1,536**, or
**128.0% more**. These source/runtime/occurrence-bound observations confirm that
the extra scalar work persists at equal width. They still do not measure its
contribution to the 0.194 µs residual or justify changing reduction/layout
arithmetic. All four native runs and strict joins pass; the record is
`attention190-q12-producer-work-r1/matched-q12-c512-producer-work.json`.

The combined production 190 candidate, including nested Fold12 and the GDN
layout change, now passes its canonical selected accuracy panel and matched
model benchmark at 820.871265 prefill / 41.536547 decode tok/s. See the
[GDN qualification record](QFN_GDN_RECURRENCE_TUNING.md). The component results
below do not independently attribute that model gain to attention.

## Nested selector190: long-context check

The 12-record selection now nests under the existing through-1024 branch.
Fresh unprofiled A/B/A complete-call measurements retain the short-context gain
and reduce the earlier pre-rotary 64K cost. Each graph contains 64 complete
producer/merge operations, with five warmups and twelve alternating samples;
the pre-rotary 64K cell additionally repeats B/A. Every exported output and
numerical partial is byte-identical to the frozen control.

| Q6 rows / context / key encoding | Control, µs | Nested Fold12, µs | Time change |
|---|---:|---:|---:|
| M1 / 768 / post-rotary | 15.59750 | 15.28000 | −2.036% |
| M1 / 64K / pre-rotary | 20.45042 | 20.53438 | +0.411% |
| M1 / 64K / post-rotary | 18.53813 | 18.51625 | −0.118% |
| M16 / 64K / post-rotary | 81.61200 | 81.45765 | −0.189% |

All fourteen native processes pass source/runtime, driver and normal-retirement
checks. The nine exact registered FP16 arithmetic tests also pass in the Release
fixture. The remaining 0.411% pre-rotary cost is measured and retained; it is
not relabeled as a speedup or dismissed as noise. This component result does not
establish a whole-model gain or change the qualified checkpoint180 score.
Evidence: `attention190-nested-fold12-long-summary-r1/result.json` and
`attention190-nested-fold12-functional-r1/{result,exact-filter-audit}.json`.

This compares qualified Llaminar180 with pinned Castagna
`3483d462715d61f21ed5877836be35a724446359` (retained165 recording), on the
matched four-MI50, 512-token prompt and 255 after-prefill forwards. The
[bounded trace analysis](../../../../parity-results/qwen-flash-next/attention188-short-context-attribution-r1/attribution.json)
authenticates all 24,480 Llaminar and 12,240 fork attention dispatches.
The qualified model policy and score remain checkpoint180. Source candidates
below are not production promotions.

## Matched-width follow-up: the large gap is mostly local batch size

The fixture-only Q6/Q12 extension is now measured with the **same unchanged
native producer and merge bodies/resources**, authenticated against the archived
pre-change core. Both engines receive Q12/KV1, identical visible/physical length
at these cells, and the same deterministic semantic input coordinates. All
fresh build/runtime/driver/retirement checks pass. These are unprofiled complete
captured producer-plus-merge calls, with 64 operations per graph, five warmup
pairs and 12 samples. The original eight-part control is not the score shown here.

| Context | Llaminar Q6, µs | Llaminar Q12, µs | Castagna Q12, µs | Q12 gap |
|---|---:|---:|---:|---:|
|512|14.74490|14.78365|14.62995|+1.051%|
|768|15.59615|15.72865|15.42490|+1.969%|

Doubling our useful query heads increases call time by only **0.263% at 512**
and **0.850% at 768**. The Q6 heads' output and all numerical partial bytes match
exactly inside Q12. A repeat at 768 gives 15.63365 µs for Llaminar Q6 and
15.42750 µs for the fork Q12, supporting the same conclusion. Matching width removes roughly
98–99% of the isolated per-head gap under this normalization; it does **not**
establish that much token-latency improvement from a placement change.

This identifies narrow local head batching as the dominant cause of the apparent
per-head discrepancy at these short shapes. It does not support a twofold
intrinsic attention-kernel speed deficit or a need to eliminate shared memory.
The remaining equal-width difference is about 1–2%. These original188 width measurements preceded the separate Fold12 component experiments below. Whole-model placement/capture/communication
and timing need separate qualification before changing attention participation.

Evidence: [matched-width summary](../../../../parity-results/qwen-flash-next/attention188-short-context-attribution-r1/matched-width.json),
`attention188-width-{llaminar,fork}-q{6,12}-c{512,768}-r1` receipts, and archive
`/tmp/qfn-attention188-width-green-r1`.

## What the measurements establish

Castagna processes twelve local query heads in approximately the same producer
call time as Llaminar's six. Llaminar runs all four cards on each of twelve
attention layers; Castagna uses two rotating cards per layer. Each card still
performs 72 useful head-layer instances per step. The difference combines local
batch width, geometry, arithmetic and participant scheduling. It does not prove
that the fork's memory layout alone is twice as fast, or that dropping half the
calls would preserve useful work.

| Arithmetic mean per card per decode step | Llaminar180 | Castagna165 |
|---|---:|---:|
| Producer service, µs | 258.285 | 129.814 |
| Merge service, µs | 98.268 | 33.382 |
| Producer and merge calls, each | 12 | 6 |
| Query heads per call | 6 | 12 |
| Producer service per call, µs | 21.524 | 21.636 |
| Merge service per call, µs | 8.189 | 5.564 |

The producer accounts for **66.44%** of the additive per-card service gap;
merge accounts for **33.56%**. These are service contributions, not removable
request latency. Both engines still execute twelve global attention layers.

## Concrete merge overhead

The model run uses selected counts 513–767, derived from the request's causal
progression. Llaminar therefore selects its **16-record fold**, despite allocating
33 records. Castagna folds 12 partitions in the recorded model geometry.

| Work per output coordinate | Llaminar | Castagna |
|---|---:|---:|
| Denominator FMA steps | 16 | 12 |
| Numerator FMA steps, including required neutral tail | 18 | 12 |
| Metadata publication barriers | 2 | 1 |

The 33–50% extra FMA work is consistent with the 47.2% longer merge call; it is
not an isolated causal timing proof. Llaminar reads weighted payloads only from
9–12 nonempty records, about 9–12 KiB per head, rather than all allocated slots.
Castagna reads 12 KiB including masked payload. Observation counters add metadata
when enabled. An isolated 512-token profile selects Llaminar's eight-record fold
and must not be reported as evidence for this model's sixteen-record fold.

A focused next candidate is an intermediate **12-record fold through 768 selected
tokens**. The producer proves later records empty. This removes four denominator
and metadata terms and changes numerator steps from 18 to 14, retaining the two
neutral FMAs that preserve finite rounding and flush-to-zero behavior. It leaves
weighted payload traffic, the producer, and all 33 storage slots unchanged.
Signed-zero, boundary, actual-counter, grouped/serial and complete-pair economics
must pass before promotion. No speedup is assumed from operation counts.

## Producer costs and register limits

Retained isolated 512-token profiles, with the current producer source, show:

| Producer metric | Llaminar Q6 | Castagna Q12 |
|---|---:|---:|
| Vector instructions per useful head | 6,289 | 5,136 |
| Scalar instructions per useful head | 3,501.33 | 1,536 |
| Fetched KiB | 528.813 | 525.938 |
| Static shared memory, bytes | 21,248 | 20,000 |
| Used / allocated vector registers | 63 / 64 | 83 / 84 |
| Required one-tile CTA barriers | 7 | 10 |
| Spills and measured LDS bank conflicts | 0 | 0 |

Our producer executes **22.4% more vector work** and **128.0% more scalar work
per useful head**. At 512 tokens, only eight of our 33 CTAs perform heavy
arithmetic; the others publish empty headers. The isolated fork has 24 useful
CTAs. More useful blocks and wider local head batches are plausible scheduling
advantages, but these counts do not measure achieved occupancy or assign cycles.
The earlier G2/128 experiment already increased distribution and lost performance
because it added instructions; underfilling alone is not a sufficient diagnosis.

Both kernels keep scores, softmax state and output accumulators in registers.
Both use a 17,408-byte shared KV panel. Moving our 768-byte probability bank into
registers remains a separate hypothesis: it saves no barriers and exchanges LDS
reads for shuffles. The fork's no-LDS GDN recurrence is unrelated to its attention
kernel. Bandwidth saturation, bank conflicts, spills, fewer barriers and lower
register use are not supported explanations for this attention result.

## Limits and next measurement

The measured attention-only union over all four cards is 1,140.421 µs per step
for Llaminar and 652.763 µs for the fork. Fork attention intervals have almost no
cross-card overlap in this trace; Llaminar has substantial overlap. The record
has no authenticated graph-node-to-layer map, so these unions are active-time
diagnostics, not layer critical paths. Observer effects remain unresolved.

The matched Q12/KV1 follow-up above resolves the width effect at512and768.
A640comparison still needs its physical-padding difference disclosed. A Q6 fork
run changes its GQA dispatch and cannot certify the same selected kernel. Keep
canonical accuracy and unprofiled whole-model Release throughput as the
promotion gates; the width-only188 experiment changed no native attention body or placement. Later189/190 component trials below change the merge and remain separate from whole-model qualification.

The detailed retained report is
[here](../../../../parity-results/qwen-flash-next/attention188-short-context-attribution-r1/REPORT.md).


## Fold12 component189: a short win with a measured long-context cost

This section records the first, linear-selector189 experiment. The later nested-selector190 short-context comparison is recorded below; these historical189 timings and counters are retained unchanged.

The first implementation adds a 12-record structural prefix for selected counts
513 through 768. It keeps the 33-record storage, producer, weighted-read guards,
ordered FP32 chains and two neutral numerator FMAs. The additional boundary is
read from the existing device count; no host readback or new graph node exists.

All nine tests from the registered FP16 arithmetic filter pass in the Release
fixture. The extended signed-zero test covers 512/513 and 768/769 with M1/M16,
verifies actual empty headers/counters and untouched weighted payloads, and
compares grouped rows to serial capture. The wide-counter test retains its old
cases and adds authored 513/705/768/769 cases. Twelve frozen/current exported
output and numerical-partial pairs are byte-identical, including long context
and both key encodings. This is focused qualification, not a refreshed complete
ProductionTestPreflight or model accuracy result.

Matched unprofiled complete producer-plus-merge events use five warmups,
64 operations per captured graph and twelve alternating samples. Frozen169
provides the old merge; its numerical body is also the qualified180 body.
A/B/A controls bracket every cell; 768 and the two 64K M1 encodings additionally
have B/A repeats. Representative arithmetic means of run medians are:

| Rows / context / key encoding | Control, µs | Fold12, µs | Time change |
|---|---:|---:|---:|
| M1 / 513 / post-rotary | 15.52498 | 15.17000 | −2.286% |
| M1 / 640 / post-rotary | 15.57750 | 15.22500 | −2.263% |
| M1 / 768 / post-rotary | 15.61667 | 15.23683 | −2.432% |
| M1 / 64K / post-rotary | 18.60250 | 18.66875 | +0.356% |
| M16 / 64K / post-rotary | 81.57230 | 81.47105 | −0.124% |
| M1 / 64K / pre-rotary | 20.59708 | 20.91250 | +1.531% |

Fresh build guards pass CUDA SM80/86/89/90 and ROCm gfx906. Bounded ISA
inspection proves both post-rotary producers and the eight-part merge unchanged.
The 33-part merge retains 124 VGPRs, 708 bytes of static LDS and zero spills or
private storage, but grows from 11,132 to 12,736 code bytes. At context768,
independent exact-dispatch profiles show VALUInsts 173.25→152.75 (−11.83%) and
LDSInsts 21.50→16.75 (−22.09%), with 24 waves unchanged. Native graph, sixth
retained occurrence, SDK callback, counter CSV and loaded code object join
successfully. Profile durations are not benchmark scores.

The long pre-rotary cost repeats and is retained as a regression, not dismissed
as noise. A source follow-up nests the 12-record choice inside the existing
16-record range to avoid adding selector depth to long contexts. At the189 checkpoint that follow-up
was not yet built or measured; its subsequent190 short-context result appears below. No model-level improvement or default promotion
is claimed from component189.

Evidence: [complete timing and profiler summary](../../../../parity-results/qwen-flash-next/attention189-fold12-summary-r1/result.json),
`attention189-fold12-functional-r1`, `attention189-fold12-release-build-r1`, and
frozen archive `/tmp/qfn-attention189-fold12-component-green-r1`.


## Nested Fold12 component190: near parity at matched Q12

Fresh matched-width runs compare the frozen188 attention pair, the current nested
Fold12 pair, and the same pinned fork at M1/Q12/KV1/D256. Both512 and768 use
identical visible and physical lengths, original deterministic semantic inputs,
64 complete producer-plus-merge operations per captured graph, five warmups and
12 retained samples/process. The order is frozen188 / nested / fork / nested /
fork / nested / frozen188. The 768 block was repeated to assess its small residual.

| Context / block | Frozen188, µs | Nested190, µs | Fork, µs | Nested vs188 | Nested vsfork |
|---|---:|---:|---:|---:|---:|
|512|14.775625|14.749183|14.555000|−0.179%|+1.334%|
|768 first|15.765650|15.438333|15.381250|−2.076%|+0.371%|
|768 repeat|15.778775|15.454167|15.380675|−2.057%|+0.478%|

Values are arithmetic means of process medians. The combined 768 mean is
15.446250µs versus15.380963µs for the fork, a0.065287µs (+0.424%) residual.
Nested 768 process medians span0.0675µs, versus0.0100µs for the fork; within-process
MAD is0.02125–0.03245µs versus0.00745–0.01375µs. The residual stays positive in
these epochs but its exact sub-percent magnitude is not a stable causal estimate.
At 512 the0.194183µs residual is steadier: nested process medians span0.00125µs
and fork medians0.00750µs. These are descriptive spans/MAD, not independent-sample
confidence intervals from the correlated retained replays.

All 21 native processes exit0 and retire with clean driver windows, no signals,
forced kills or survivors. Every frozen/nested FP32 output and numerical-partial
export is byte-identical over all12 heads. Current admission explicitly permits
only changed merge33, binds its fresh12,732-byte function/124 VGPR/708 B LDS/private0,
and proves both producer bodies/resources and merge8 byte-identical to188.
Used merge SGPR changes 29→30 while encoded allocation remains32. No GDN kernel,
model or profiler runs during this matched sequence; it certifies neither the
current GDN implementation nor model throughput.

Evidence: [512 summary](../../../../parity-results/qwen-flash-next/attention190-matched-width-r1/matched-q12-c512-summary.json),
[768 first](../../../../parity-results/qwen-flash-next/attention190-matched-width-r1/matched-q12-c768-summary.json),
[768 repeat](../../../../parity-results/qwen-flash-next/attention190-matched-width-r2/matched-q12-c768-summary.json),
`attention190-nested-fold12-functional-r1`, and the bounded current attention ISA
at `/tmp/qfn-attention190-fold12-root-private/isa-r1`.

### Layout is not the demonstrated dominant cause

Both matched inputs already have compact 512-byte FP16 K/V token strides at KV1.
The common fixture's larger two-head source bank is not the captured input:
`Inputs` explicitly repacks original head bytes into compact storage and overrides
the view. The fork uses the same token/head/coordinate order here. Both keep a
17,408-byte padded shared K/V panel. Their remaining differences are physical
scheduling, shared query/probability storage, metadata and arithmetic—not a
missing global KV transpose or a no-LDS attention kernel.

| Matched Q12 producer fact | Llaminar | Fork |
|---|---:|---:|
|Threads / wave64s per CTA|384 /6|256 /4|
|Heavy CTAs at 512 /768|16 /24|24 /36|
|Useful waves at 512 /768|96 /144|96 /144|
|Total launched waves at 512 /768|396 /396|96 /144|
|Used / allocated VGPR|63 /64|83 /84|
|Static LDS|21248B|20000B|
|Runtime theoretical resident CTAs/CU|2|3|
|Corresponding theoretical resident waves/CU|12|12|
|Spills / measured LDS conflicts|0 /0|0 /0|
|Required terminal one-tile barriers|7|10|

Thus the fork distributes the same useful waves over more heavy CTAs/CUs while
Llaminar launches many empty-header waves. This is a scheduling hypothesis;
achieved occupancy is not measured by the theoretical resource API. Lower
Llaminar VGPR use does not yield a higher producer wave-residency limit.
The merge's theoretical limit differs 8 versus 32 waves/CU, but only 12 merge CTAs
are launched, so that limit alone cannot establish elapsed-time loss either.

The retained512 producer counters show22.4% more vector and128.0% more scalar
work/head in Llaminar, but nearly equal call-level FetchSize (528.8125 versus
525.9375KiB), low MemUnitBusy and zero bank conflicts/spills. Bounded compiled
bodies contain20 versus11 permutation sites and128-bit native loads in both;
static sites include branches/loops and are not dynamic instruction counts or
memory transactions. Q12 producer counter passes and actual transaction/achieved-
occupancy metrics remain uncollected. Those missing measurements must not be
invented from source-copy counts or used to call the small residual bandwidth-
limited.

Intermediate storage does differ: Llaminar uses 1040-byte partial records with
max/denominator/counters beside weighted values; the fork separates float2
metadata and contiguous1024-byte payloads. Empty-record weighted reads are guarded.
Retained512 merge FetchSize per head is6.78% higher for Llaminar, but observation
metadata and Q6/Q12 cache behavior are confounded; this does not assign a latency
cost to that layout.

The strongest demonstrated implementation cause at 768 is the old Fold16 tail:
component189 removed four empty denominator/metadata terms and four numerator
FMAs while preserving two neutral RN/FTZ operations; fresh Q12 now independently
confirms about2.07% complete-boundary gain. The remaining 512 margin could involve
header/control/address work, partial AoS metadata, or fixed dispatch costs, but
none has a measured cycle attribution. A matched Q12 role/counter pass is the
smallest next diagnostic if this residual remains worth pursuing. A layout
rewrite or new attention placement policy is not justified by these results.

Private detailed source/ISA/counter map, copied owned collectors, negative tests
and exact commands: `/tmp/qfn-attention-residual-attribution-private/REPORT.md`.
Whole-model placement, accuracy, overlap-aware phase attribution and Release
throughput remain the production gates; root records long-context190 guards
separately.
