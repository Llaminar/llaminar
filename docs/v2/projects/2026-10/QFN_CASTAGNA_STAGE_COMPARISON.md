# QFN: Castagna and Llaminar stage comparison

## Fresh checkpoint190 attribution, 2026-10-24

The qualified ROCm ValueMajor default now has a fresh complete-request profile.
The source/runtime, natural 256-token trajectory, driver and retirement checks
pass. All 6,461,320 dispatches join, including 6,348,052 captured dispatches
across 4,108 graph launches. Actual decode uses the ValueMajor DK128 vector
endpoint recurrence on all four cards, with 36 calls per step per card.
Both layout overrides remain absent. The comparison reauthenticates the same
retained Castagna165 trace; it does not run a new fork benchmark.

These are median per-card **instrumented kernel-union milliseconds per decode
step**, not request latency or an estimate of removable time. Every measured
request retains prompt512, emitted256 and 255 after-prefill steps. The report
keeps per-card sums, unions, exclusive intervals and overlap separately.

| Source-identified decode portion | Llaminar190 ms/step | Castagna ms/step |
|---|---:|---:|
| Guarded short-context indexer query projection/preparation nodes | 0.280 | No corresponding work |
| Main attention core, producer and merge | 0.349 | 0.163 |
| GDN convolution, recurrence and output norm | 0.814 | 0.690 |
| Vocabulary projection and reducer | 0.300 | 0.260 |
| HC injection projection | 0.657 | 0.759 |

The paired GDN union gap falls from approximately 0.459 ms at180 to **0.126 ms**
at190. The new largest selected cohort is the guarded query work at0.280 ms,
followed by the attention gap at0.186 ms. The short-query nodes already skip
matrix arithmetic and weight reads; the remaining measurement includes their
captured dispatch and guards. Omitting them altogether requires a proved
captured-graph policy with an authoritative demand bound, not a host live-count
readback or hot recapture.

Attention still uses Q6 across12 layers/card versus the fork's Q12 across6.
The separate matched-Q12 component comparison remains the evidence about
intrinsic kernel efficiency. GDN excludes projections/gates, and the fork's
output norm also quantizes. Generic and fused kernels retain their native-family
classification; the report does not claim complete semantic model-stage naming.
The whole512-row prefill remains separately normalized, including every fork
physical chunk. None of these category differences may be added into a
predicted end-to-end speedup.

Unprofiled production performance remains **820.777 prefill /41.562 decode
tok/s**. The **42.959 decode target remains open**. Evidence:
`gdn190-default-stage-trace-r1`, `gdn190-default-stage-join-r1`,
`gdn190-default-stage-intervals-r1`,
`gdn190-default-attention-core-comparison-r1`, and
`gdn190-default-stage-comparison-r1/stages.json` under the QFN result root.
The archived profile source reconstructs the qualified pre-commit tree exactly;
it does not substitute later attention experiments for checkpoint190.

## Fresh checkpoint180 attribution, 2026-10-24

The fresh four-MI50 capture of qualified checkpoint180 passes native exit,
driver, runtime/source identity, exact work and cleanup checks. The measured
request has 512 prompt tokens, 256 emitted tokens and 255 after-prefill forwards;
all emitted IDs match both the qualified180 control and retained Castagna165.
All 6,461,320 dispatches join successfully, including 6,348,052 captured
dispatches across 4,108 graph launches. No measured kernel name is unclassified
at the operation-family level. Complete semantic model-stage attribution is
still unavailable for some fused and generic kernels.

The table reports the median across four physical GPU pairs of each engine's
recorded kernel-union time per decode step. It is **instrumented diagnostic
time**, not request latency or predicted removable time. The output retains
per-card sums, unions, exclusive intervals and the full overlap partition.

| Source-identified decode portion | Llaminar ms/step | Castagna ms/step |
|---|---:|---:|
| GDN convolution, recurrence and output norm | 1.154 | 0.690 |
| Short-context indexer query projection/preparation nodes | 0.271 | No corresponding work |
| Main attention core, producer and merge | 0.356 | 0.163 |
| Vocabulary projection and reducer | 0.290 | 0.260 |
| HC injection projection | 0.653 | 0.759 |

**GDN is the largest remaining directly comparable gap.** Its paired-card
median difference is 0.459 ms/step, followed by the guarded indexer nodes at
0.271 ms and attention at 0.193 ms. Paired medians need not equal the difference
between separately rounded engine medians. These gaps must not be added into
an expected throughput gain.

Checkpoint175 skips short-context query arithmetic and weight reads, but
retains its captured projection and preparation nodes. The fresh trace proves
12 guarded projection and 12 preparation calls per decode step/GPU. This is
remaining guarded-dispatch cost; it does not mean the unused query matrix
product was reintroduced. Prefill still uses its original AllLive query policy.
The existing measured-work boundary requires ranking first at visible length
2052; this workload remains below that boundary.

The GDN comparison includes the selected three physical operations, rather
than every projection or gate in a full GDN layer. Castagna moves some
normalization into convolution and also quantizes in its output norm. Its
persistent HC extensions remain indivisible in the full family report.
The attention layouts retain Q6 across 12 layers per Llaminar GPU versus Q12
across 6 active Castagna layers per GPU. Arithmetic and grouping differ.
HC injection is now ahead on all four cards; the separate HC-down/up fusion
screen182 concerns a different operation.

The [three-way GDN structure map](QFN_GDN_STRUCTURE_COMPARISON.md) compares
Llaminar ROCm/CUDA with the fork's observed HIP program. Independent component
service sums locate the main deficit in recurrence: 682.274 versus 221.521
µs/step/GPU. These are sums of kernel durations, not the union values above.
The fork uses FP32 recurrence with a different state layout, more lanes per
column, fewer registers and no recurrence LDS/barriers. It also computes Q/K
normalization at convolution once per key head; our current recurrence repeats
it 12 times. The map records arithmetic differences and required tests before
transferring these structures.

The complete 512-row prefill is retained separately, including Castagna's
320+188+4 physical chunks. Attention-core time is 11.670 versus 4.028 µs per
prompt row/GPU. Prefill tuning remains secondary to the decode target.
The four-device report also retains active-kernel unions and concurrent-GPU
intervals. Kernel-free intervals are not identified as host idle. Large
instrumented collective waits are not unprofiled communication costs.

Unprofiled production evidence remains checkpoint180's 819.594 prefill /
41.030 decode tok/s; this capture supplies no new throughput score. The
42.959 Castagna decode target remains open. The source-only next-step reviews
focus on the GDN components and a proven short-context graph variant that
could omit the guarded query launches entirely, without host live-state mirrors
or hot-path recapture.

[Fresh trace](../../../../parity-results/qwen-flash-next/direct180-stage-trace-r1/result.json),
[complete graph join](../../../../parity-results/qwen-flash-next/direct180-stage-join-r1/joined-dispatches.json),
[native-family intervals](../../../../parity-results/qwen-flash-next/direct180-stage-intervals-r1/intervals.json),
[paired core admission](../../../../parity-results/qwen-flash-next/direct180-attention-core-comparison-r1/attention-core.json)
and [full comparison](../../../../parity-results/qwen-flash-next/direct180-stage-comparison-r1/stages.json)
retain the exact evidence. The Castagna side is the admitted retained165 trace,
not a new run; its shared GGUF identities, source/runtime receipts, complete
trajectory and four physical PCI pairings were reauthenticated. The source
view proves committed 819496597 code equals qualified180's 1309a0fe plus its
original source patch. Private experiments181/182 were archived and removed
before the production capture.

## Historical attention124 matched core comparison

The same four-MI50 workload now measures **817.871 prefill /39.754 decode
tok/s** for Llaminar's explicit FP16 tile policy, versus **816.092 /39.390**
for its same-build FP32 A/B/A control. The fresh pinned Castagna baseline is
**462.166 /42.657**. All natural 256-token trajectories match. These are
unprofiled Release results; Llaminar's +0.925% decode improvement does not
establish an attention-stage win against the fork.

Both fresh admitted traces pass driver, runtime, source, workload and retirement
checks. Llaminar's executed producer is `selectedAttentionFP16Tiles<false>` on
all four GPUs: this native FP16 cache stores post-RoPE keys. The earlier
pre-RoPE component timings are not production-stage timings. Actual allocation
metadata reports68 vector registers,19968 bytes LDS and zero private scratch.

| Complete attention core, median of four per-GPU observations | Llaminar | Castagna |
|---|---:|---:|
| Prefill, µs per prompt row | 12.2255 | 3.9578 |
| Decode, ms per after-prefill step | 0.581918 | 0.162666 |

Llaminar still spends **3.089×** the recorded prefill core time and **3.577×**
the decode core time. Decode ranges are0.563831–0.614159ms and
0.161527–0.162802ms respectively. These ratios compare different arithmetic
programs and physical layouts; they are diagnostic, not equal-arithmetic kernel
benchmarks or predictions of removable request latency.

The boundary contains each complete producer and ordered merge, including the
fork's dense prefill preparation/compute/merge. It excludes external RoPE,
cache publication, projections, output gating and indexer work. All other
recorded kernels participate in the same overlap sweep. The report retains
sums, unions and exclusive intervals separately; no GPU times are added.
Broader wrapper and complete attention-block attribution remain unresolved.

The subsequent source audit explains the different participation: the fork
rounds attention slices to whole 12-query-head GQA groups and rotates their
owners by full-attention-layer ordinal. With four equal devices and two KV
heads, only two devices own attention slices in each layer; the active pairs
alternate. Llaminar uses four six-query-head slices in every attention layer.
The table above measures GPU work and must not be read as a request-latency
ratio.

A read-only follow-up accounts for the union of core kernel intervals across
all four devices, counting overlapping intervals once. It authenticates the
same dispatch inventory and all 255 decode steps, using the original graph
windows for Llaminar and revalidated vocabulary-head endpoints for Castagna.

| Recorded all-device core-active union | Llaminar124 | Castagna |
|---|---:|---:|
| Prefill, µs per prompt row | 15.403832 | 15.715459 |
| Decode, ms per after-prefill step | 1.660075 | 0.649616 |

These are additional **instrumented active-time diagnostics**, not unprofiled
attention latency or removable request time. Instrumentation changes scheduling
and has different overhead in the two engines. First-to-last spans include
intervening non-core work and are retained separately. Exact per-layer spans
remain unavailable because the retained traces lack an executed node-to-layer
identity map; layer IDs are not inferred from timestamp clusters.

[All-device accounting](../../../../parity-results/qwen-flash-next/attention124-all-device-core-accounting-r1/result.json)
retains the separate GPU-work, union, span and per-step results. The bounded
reader makes one core-family scan per original immutable database, creates no
index and does not parse the full JSON traces. This is a refinement of the124
diagnosis, not a new trace or qualification of the later component candidates.

Both engines execute72 query-head/layer instances per GPU: Llaminar Q6×12,
Castagna Q12×6. The complete measured prefill is512 rows, split as320+188+4
by the fork, and decode contains255 forwards. Strict geometry/count checks
authenticate24/48 prefill core launches and6120/3060 decode core launches per
GPU respectively. All original graph-to-dispatch counts match.

[Core comparison](../../../../parity-results/qwen-flash-next/attention124-core-comparison-r1/attention-core.json)
retains the derived joins, intervals, classifier outputs and comparison source.
[Unprofiled comparison](../../../../parity-results/qwen-flash-next/attention124-fresh-engine-comparison-r1.json)
retains the distinct throughput authority. Castagna's first fresh trace remains
rejected for an AMD restore-userptr workqueue warning; the admitted repeat is
`attention124-castagna-trace-r2`, with no new driver diagnostics. Llaminar's
admitted trace is `attention124-stage-trace-r1`.

The [actual kernel source comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md)
separates the fork's wider prefill KV reuse, independent score chains, loop
scheduling and packed merge from its different partition/PV/denominator
arithmetic. The first arithmetic-preserving QK unroll8 trial improves measured
post-RoPE component time by4–6%; no new whole-model speed is claimed for it.

## Historical HC103 workload and throughput

The earlier comparison uses the HC103 Llaminar Release archive and Castagna
commit `3483d462715d61f21ed5877836be35a724446359`, on the same four MI50s.
The GGUF is Qwen3.8-Flash-Next UD-Q4_K_XL, with FP16 KV, RAM ngrams, context
4096, 28 physical NUMA1 workers, 512 repetitions of token 1228, and 256 emitted
tokens. The first emission belongs to prefill; decode has 255 forwards. MTP and
prefix reuse are off. Both engines generate the **same 256 IDs**. Their expert
partitioning, arithmetic and host control are still different implementations.

Llaminar uses **TensorParallel + GateUpOwnedDownColumns**, static GPU experts,
a vocabulary-sharded head and Automatic resolved to Four workers. Its precision
policy remains FP16 MoE GDN alpha/beta, FP16 routing, SingleQ8 for GDN QKV and
GDN/attention output activations, and TwoTermQ8 for HC up. Castagna uses its
tensor split with within-expert sharding and the original F62 switches. The
linked invocation receipts retain exact CLI and environment settings.

| Unprofiled Release observation | Prefill tok/s | After-prefill decode tok/s |
|---|---:|---:|
| Llaminar L104 control, six ABBA samples | 814.723 | 37.439 |
| Llaminar HC103, six ABBA samples | 815.461 | 37.576 |
| Castagna fresh repeat1, three samples | 465.370 | 38.192 |
| Castagna fresh repeat2, three samples | 453.233 | 42.468 |
| Historical Castagna F62 target | 467.821 | 42.959 |

HC103's +0.091%/+0.367% changes have overlapping sample ranges and are small
descriptive gains. The second Castagna repeat returns close to its historical
decode result. Retain the first repeat's variation; do not lower the target to
manufacture a win. Against the second repeat, Llaminar has about 1.80× prefill
throughput and 11.5% lower decode throughput.

[ABBA](../../../../parity-results/qwen-flash-next/hc103-model-comparison.json),
[Castagna repeat1](../../../../parity-results/qwen-flash-next/castagna-f62-repeat-r1/result.json),
and [repeat2](../../../../parity-results/qwen-flash-next/castagna-f62-repeat-r3/result.json)
retain the samples, original runtime/source identities and clean retirement.
HC103 passes 776 Unit tests, the final 23 focused registrations, and all 185
final-logit rows across 26 files remain byte-identical to L104. This is focused
qualification; complete ProductionTestPreflight remains G60.

## Diagnostic timing and attribution

Both fresh traces pass workload, driver and retirement checks. Original
graph/correlation/agent/count joins authenticate 6,632,680 Llaminar dispatches
and 2,248,086 Castagna dispatches across setup, warmup and measurement. Those
whole-run counts are not per-token comparisons. All 129 measured Llaminar names
and 91 Castagna names have explicit operation-family classification. Generic
projections, elementwise kernels and persistent runtime extensions still have
unresolved model callers; source-stage completeness is **not** claimed.

The [comparison evidence](../../../../parity-results/qwen-flash-next/hc103-castagna-stage-comparison-r1/strong-cohorts.json)
contains all four physical PCI pairings, actual symbols/geometries/call counts,
kernel sums, unions, exclusive intervals and overlap partitions. The retained
`llaminar/`, `castagna/` and `source-notes/` directories preserve the broader
sweeps and source mappings. No raw model or native-library payload hashing was
used. Profiler throughput is 10.355/13.821 decode tok/s for Llaminar/Castagna,
so instrumentation affects the engines differently. These are diagnostic
intervals, not predictions of removable unprofiled latency.

The following values are milliseconds of recorded kernel work per decode step,
shown as the median across the four per-GPU observations. They are not sums of
GPU time. Each source-identified cohort's union is equal to its summed duration
in this capture; the artifact also retains its overlap with other work.

| Source-identified physical cohort | Llaminar ms/step | Castagna ms/step | Interpretation |
|---|---:|---:|---|
| HC injection projection, FP32, N4/K10240, 96 calls | 1.927 | 0.763 | Strong same-shape FP32 target; reduction order differs. |
| Main attention core, partition producer + merge | 1.039 | 0.163 | Different arithmetic, partitioning and query-head grouping; external gates excluded. |
| GDN convolution + recurrence + output norm | 1.148 | 0.686 | Preconditioning/fusion boundaries differ; not the complete GDN pipeline. |
| Short-context indexer query projection + preparation | 0.746 | 0 | Llaminar performs work unused below visible 2052; retained key/state work is separate. |
| Vocabulary head producer + ordered reducer | 0.300 | 0.270 | Same N62080/K2560 per GPU; common activation preparation is excluded. |

The differences hold on all four cards. Ranges are 1.865–1.984 versus 0.760–0.771ms
for injection, 1.004–1.096 versus 0.162–0.164ms for attention, and 1.088–1.292
versus 0.674–0.690ms for the selected GDN boundary. The head is a smaller gap:
0.289–0.306 versus 0.258–0.283ms. Do not add these diagnostic differences into
a predicted request speedup.

## What explains the differences

**HC injection is the strongest first target.** Both engines use four 256-thread
blocks for the same FP32 projection. Llaminar accumulates 40 strided scalar
products per lane, then reduces the original 128/64/32…1 tree with three CTA
barriers and 1024 B LDS. Castagna loads ten contiguous float4 groups per lane,
groups products differently, and uses a wave-first reduction with one barrier
and 16 B LDS. Its measured specialization reports 16 VGPRs, 32 SGPRs and zero
private scratch; Llaminar's trace proves zero scratch/LDS 1024 but does not
provide its register counts. These are architectural differences, not proof
that any single one causes the whole timing gap.

The earlier [one-barrier experiment](QFN_FLOATING_REDUCTION_TUNING.md#deferred-one-barrier-experiment)
already tested those original pairs in wave 0. It needed explicit native
operand ordering to preserve NaN payloads, and its HC timing was inconclusive.
That history supersedes the initial recommendation to try it again. The current
load-scheduling work instead retains all three barriers and the original FMA
chain, while overlapping independent operand loads. Its scalar, four-load and
eight-load candidates are measured separately for decode and prefill.

**Attention buys speed partly with lower precision.** Llaminar retains FP32
queries, eight fixed 257-index partitions, fixed FP32 QK trees and ascending
FP32 FMA accumulation for PV. On gfx906, Castagna rounds scaled Q to half2,
uses packed FP16 dot products with FP32 QK accumulation, rounds softmax weights
to half, and uses half2 PV accumulation/rescaling for the observed `false`
accumulator specialization. Its tiled layout groups four query heads and uses
twelve KV partitions. It is not a precision-equivalent replacement.

Llaminar processes six query heads in each of twelve layers per GPU. Castagna
assigns whole twelve-head GQA groups to two active participants per layer,
rotating ownership: six layers per GPU. Both have 72 query-head/layer instances
per card, but KV replication, launch count and layout differ. The current
selected-attention observation counters also perform extra shared-memory
reductions; simplifying proven coordinate counts and removing unused indexer
queries are candidates that need not change floating arithmetic. Any FP16
attention experiment must pass the canonical QFN numerical regime and grouped
decode obligations before promotion.

**GDN needs a whole-boundary comparison.** The isolated recurrence is much
shorter in Castagna, but Llaminar owns Q/K normalization and alpha/beta nonlinear
work there. Castagna moves Q/K normalization into convolution and some gate
work into persistent HC extensions. Its output norm also publishes Q8_1 for the
next projection. The table preserves convolution, recurrence and norm together,
while explicitly excluding gate projections/other preconditioning. This is a
promising tuning area, not a justified claim that its recurrence can replace
ours unchanged.

**Short-context indexer queries are unused.** Castagna omits them on this
workload. Llaminar already skips expensive score/ranking math, but still projects
N512/K2560 indexer queries and prepares them at every full-attention layer.
The first row requiring ranking is visible 2052. An optimization must preserve
raw keys, incremental pools, accepted tails and selected indices, use a typed
device-owned row predicate, and handle mixed grouped rows and boundary crossings.
It must also describe diagnostic query liveness honestly.

**Whole persistent HC is not yet a clean scalar ratio.** Castagna fuses
normalization/down/activation/up/read and usually preceding residual write;
runtime extensions can include further projections/experts. Its injection
projection remains separate with HCP_INJ=0. Llaminar's current fusion is only
activation/publication. The fork also uses fixed-point accumulation and a
different up-projection arithmetic program. Preserve the complete native
boundary and unresolved extensions rather than assigning estimated child costs.

## Prefill findings

Fresh native/source evidence corrects the earlier 320+192 assumption. Castagna
stops its first server batch at 508 tokens for a prompt checkpoint, executes
320+188 physical rows, then finishes a four-row batch: **320+188+4=512**.
The native progress log, scheduler checkpoint branch, large-row geometries,
T4 specializations and completion receipt agree. Llaminar uses oneM512 trunk.
Both use an M1 vocabulary head after selecting the final requested row.

| Recorded prefill diagnostic, median per GPU | Llaminar ms | Castagna ms |
|---|---:|---:|
| GPU activity envelope | 661.449 | 1265.654 |
| Union of recorded kernel work | 656.200 | 672.917 |
| No recorded kernel in that envelope | 5.083 | 592.737 |
| Cross-category overlap | 59.475 | 1.221 |

These GPU envelopes are not exact HTTP wall endpoints. Castagna's gaps may
include copies, scheduling and profiler effects; they are not proven host cost
or removable latency. Its largest provisional kernel cohort is copy/layout
work, with 340.033ms median union per GPU; symbol names alone do not establish
transport direction or live bytes. Llaminar's largest cohort is collectives
(202.178ms median union), which may include waiting. The evidence retains copy
endpoint bytes, intersections and boundary-crossing records separately. Neither
cohort should be subtracted from request latency or summed with overlapping
projection/expert work.

## Next measured candidates

The subsequent attention165 checkpoint refreshes the bounded numerical-core
comparison. Both complete-model collectors, shared four-shard model identities,
graph joins and measured phase/count checks pass. Llaminar's median per-GPU
decode-core work is386.339µs/step versus Castagna's163.230µs; core-active unions
are identical to those sums, with exclusive times386.339/163.223µs. Earlier124
Llaminar core work was581.918µs. These instrumented measures retain H6×12 versus
H12×6 local ownership and are not critical-path or removable-latency estimates.

The per-GPU mean call-duration breakdown, taking the median across four GPUs,
is22.635/21.625µs for Llaminar/Castagna producers and9.560/5.573µs for their
merges. There are3060 versus1530 measured calls per role/GPU. This makes the
ordered merge the next attention target. The matched unprofiled165 A/B/A is
818.073/40.190 prefill/decode tok/s versus818.437/40.151 for frozen151 controls,
with overlapping ranges and identical natural outputs. See the
[complete qualification and evidence](QFN_ATTENTION_STRUCTURE_COMPARISON.md#retained-key-rows165-and-refreshed-paired-profiles).
Broader wrapper attribution and the other candidates below remain open.

### Qualified169 merge component

HIP33 now folds the same ordered denominator beside each output numerator,
removing the register-broadcast banks. All eight unprofiled component cells
improve0.95–3.58% versus165; Post-RoPE M1 pairs are14.745µs at512 and18.521µs
at64K, with M16/64K81.506µs. All21 native tests, including the retained actual-
merge uint64 carry regression, and eight independent numerical exports pass.
Canonical model groups/counters pass and all185 logit rows remain byte-identical
to151. All832 Unit/affected registrations pass. Matched unprofiled A/B/A measures
**819.319 prefill /40.175 decode tok/s**, versus frozen165's **818.872 /40.112**.
The +0.055% /+0.159% differences have overlapping ranges, with all nine natural
sequences identical: whole-model throughput is effectively unchanged.

Exact isolated merge evidence is124/124 VGPRs,708B LDS, zero scratch/spills and
preserved FP32 chains/division. Lower-register168/170/171/172 alternatives are
rejected for economy; no achieved-occupancy claim follows from VALU utilization.
See [the169 qualification and benchmark settings](QFN_ATTENTION_STRUCTURE_COMPARISON.md#ordered-per-output-normalization169).
The full-model/core comparison above remains165: component timing is not a new
whole-stage result or request-latency ratio. Defaults/full preflight are unchanged;
the approximately43tok/s goal and broader attribution remain open.

1. Preserve the HC injection FP32 reduction order while reducing barriers;
   validate the real optimized CPU/CUDA/ROCm paths as applicable.
2. Avoid unused short-context indexer Q work with explicit row liveness and
   crossing/reset/grouped regressions; retain the complete persistent key state.
3. Profile GDN and attention candidates on their exact input/geometry, including
   the selected-attention observation reductions. Separate precision-preserving
   changes from explicitly measured FP16 trade-offs.
4. Extend actual stage/native ownership observations before claiming complete
   attribution of generic projection callers or persistent HC extensions.

Every candidate needs focused regressions, numerical acceptance, isolated
profiler evidence and a matched unprofiled model decision. Current source
inspection is not acceptance of an unimplemented optimization.


### Short-context attention attribution, 2026-10-24

The [quantified follow-up](QFN_SHORT_CONTEXT_ATTENTION_ATTRIBUTION.md) separates
local Q6/Q12 geometry and card duty cycle from kernel work. The measured
per-card service gap is 66.44% producer and 33.56% merge. The actual model
merge uses16 records versus the fork's12, not all33 allocated records; an
intermediate12-record fold is the next bounded candidate. No register-only
attention or request-latency saving is inferred from per-card sums.
