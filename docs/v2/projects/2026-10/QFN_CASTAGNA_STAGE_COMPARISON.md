# QFN: Castagna and Llaminar stage comparison

## Attention124: current matched core comparison

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
