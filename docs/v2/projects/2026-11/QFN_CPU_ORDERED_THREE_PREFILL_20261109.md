# QFN CPU three-row Prefill candidate

Status: explicit diagnostic, **not selected by Auto or installed exact tables**.
The last qualified model checkpoint is `84f29461a`:285.983 prefill /40.013
benchmark decode tok/s (39.857 after-prefill) on2×MI50+2×CPU socket TP. Nothing
below is a new full-model gain. The500/40 goal remains open.

## Arithmetic and implementation

The candidate reuses the existing ordered three-row AVX512 body for regular
Prefill tiles, rather than only a final three-row fragment. Each output keeps
its original GPUAlignedExpert nibble preparation, fixed K partitions and
ordered FP32 fold. Existing two/one-row bodies handle short tails. The explicit
`OrderedThreeRows` enum is distinct from WideRows; old ordinals and exact-table
meanings stay unchanged. Other numerical domains, formats and runtime ISAs do
not acquire this candidate implicitly. Explicit M<3 requests reject before
writes, including the batched wrapper that otherwise normalizes legacy M1
verifier overrides to Auto. Legacy singleton behavior remains tested.

No graph, allocation, transfer, model precision or admission policy changes.
The existing FFN performance owner now accepts independent gate/up and down
policies, can time one numerical domain, suppresses counter construction during
clocks through the existing immutable collector scope, and emits chronological
native samples after clocks. A fresh actual complete-FFN call authenticates
selected physical tiles and unchanged down policy outside timing. Missing
numerical-domain timings are NaN, not zero.

## Focused correctness and compiled evidence

Thirteen original-owner registrations pass with no skips: OrderedNibbleReplay,
OrderedThreeRowNibble, PrefillOrderedThreeTile, SamePanelPairRunReplay,
LivePlanFill and ExecutionExactPolicy on runtime AVX2/AVX512, plus the FFN
policy/timing contract. These are registered in production preflight; the full
preflight is deliberately outside this iteration. The build is Release AVX512;
runtime AVX2 checks are not an independent AVX2 compilation certificate.

The new lifecycle covers all registry admissions/rejections, M73→38→39→36→32→7→5→3→2→1
reuse, second-half source intervals, N65 tails, signed Q8 extrema, original
serial bytes, guards, existing teams and workers1/4/7. The seven-worker M3 case
explicitly executes SharedBank. Later invalid M1/M2 descriptors cannot modify
any earlier output. The canonical ordered arithmetic helpers remain unchanged.

Read-only disassembly finds no new hot-loop vector spill in the inspected
ordered3 LUT/identity helpers or inlined linear path. The first inspection
accidentally used objcopy without an output-object argument, rewriting the
current and frozen benchmark objects. The current object was removed and
regenerated through its original CMake target, followed by coherent relinking;
read-only objdump then confirmed the inspected helper instruction bytes. The
frozen linked executable/core were untouched and remain the baseline authority.
The touched frozen object is not evidence of whole-object preservation.

Evidence: `/tmp/qfn-cpu-large-prefill-three-row-r1127/focused*`, `final-build.log`
and `isa/`. No model/cache payloads were read or hashed for these checks.

## Bounded geometry comparison before the scheduling correction

Original exact-batch owner, Q4_K gate/up N320/K2560, GPUAlignedExpert,28 disjoint
physical workers on one socket, retained team,1024 warmup calls,30 samples×8
repeats. Each shape has two fresh lifetimes with reversed Auto/candidate order.
These are authored homogeneous bundles, not recorded joint model route vectors.
All outputs are byte-identical to serial; actual selected tiles and empty timed
counter collection are authenticated. P is the number of gate/up projections.

| M / P | Auto median μs, two lifetimes | Ordered3 median μs | Candidate latency change |
|---|---:|---:|---:|
|72 /70|3682.398 /3680.711|3219.809 /3223.464|−12.56% /−12.42%|
|38 /70|2041.235 /2063.090|2140.763 /2146.963|+4.88% /+4.07%|
|38 /168|5386.781 /5347.123|6064.679 /6034.082|+12.58% /+12.85%|
|72 /168|9858.184 /9849.546|8514.852 /8520.435|−13.63% /−13.49%|
|36 /168|5116.956 /5088.961|4525.900 /4515.785|−11.55% /−11.26%|
|39 /168|5622.794 /5615.864|6020.955 /6050.839|+7.08% /+7.75%|

The existing full-FFN owner also passes bytes for authored E84/R3208
(38/39rows), Q4_K gate/up and Q5_1 downAuto. Its gate/up slows from about5.12ms
to5.87–5.90ms. Complete persistent timings vary substantially between controls
(10.48/11.49ms versus9.56/9.53ms candidate), so they do not establish a model
win. All raw six-phase samples and actual role-policy records are retained.

The first full-FFN configuration mistakenly interpreted1024 warmup batches as
calls: with8 repeats it executed8192 calls per phase. One Auto control completed;
the next candidate was intentionally stopped and its original helper retired
cleanly. The bounded matched comparison uses32 warmup batches,30samples and
one repeat: millisecond-scale calls do not need short-kernel repeat amortization.
The earlier long-warm Auto control is diagnostic, not paired with those runs.

Evidence: `m72/`, `m38-p70/`, `m38-p168/`, `m72-p168/`, `m36-p168/`,
`m39-p168/`, `geometry-comparison.json`, `adjacent-comparison.json`, and
`full-ffn-bounded/` under the same r1127 root.

## Identified worker-assignment problem

The existing cyclic `schedule(static,1)` gives successive jobs to the same
worker28 jobs apart. M38 ordered3 has13 physical tiles grouped into four
same-panel jobs with live row counts[12,12,12,2]. Because28 is divisible by4,
seven workers repeatedly receive only two-row tails and21 receive twelve-row
jobs. More homogeneous descriptors preserve that imbalance. Pairwise M38 has
five runs[8,8,8,8,6], so its workers rotate across the short run.

This is not a changed tail arithmetic or a partial-column copy: N320 has five
full64-column panels; M38 uses the original two-row helper, same K tree and
zero shared partial storage. Workers do not alternate helper families in the
resonant homogeneous case. M39 loses even though every physical tile uses the
three-row helper, while M36 and M72 have equal-sized runs and win. Those
measurements support the ownership explanation without assigning it a measured
percentage of the total loss yet.

Five isolated existing LinuxPerfControl observations per shape/policy, separate
from canonical timing, show the same direction:

| M / P | Auto instructions /cycles /IPC | Ordered3 instructions /cycles /IPC |
|---|---:|---:|
|38 /168|568.6M /293.2M /1.94|534.1M /350.8M /1.52|
|72 /168|1072.8M /528.0M /2.03|989.0M /474.5M /2.08|

The profiler aggregates original per-worker counters; it does not retain
individual worker samples. Ordered3 lowers L1 load/miss counts at both shapes,
but M38 LLC misses rise. These counters do not alone prove the causal share of
imbalance. `profiles/` retains the original five raw observations, launch scopes
and compiled-owner receipts; profiler durations are not performance clocks.

## Contiguous ownership correction

The explicit three-row bundle now receives contiguous static job chunks, using
ceil(jobs / actual team width). Every legacy bundle retains cyclic chunk one.
The flattened job order, panel mapping, arithmetic and output writers remain
unchanged. This breaks the repeated short-tail assignment; it is not a claim
of universally optimal weighted scheduling for arbitrary mixed bundles.

The original 13 focused registrations pass again with no skips in 8.24 seconds.
An actual OpenMP ownership regression checks unique job ownership, partial grids,
the original sixfold M38 imbalance, and the corrected distribution. Read-only
final disassembly confirms no hot vector spills in the three ordered helpers;
schedule lowering uses integer operations. The combined caller's code/frame
grows, so the whole caller is not described as stack-free.

Matched original-owner timings retain the same 28 physical workers, 1,024
warmup calls and 30 samples of eight repetitions. Each row has two fresh
lifetimes with reversed candidate order:

| M / P | Auto median μs | Corrected ordered3 median μs | Latency reduction |
|---|---:|---:|---:|
|38 /168|5327.383 /5263.662|3763.869 /3720.054|29.35% /29.33%|
|39 /168|5620.939 /5596.382|3770.838 /3765.392|32.91% /32.72%|
|72 /168|9415.997 /9433.023|6585.834 /6578.479|30.06% /30.26%|
|38 /70|1999.707 /2000.724|1417.474 /1411.438|29.12% /29.45%|

All byte, effective-route, timing-collection and clean-retirement checks pass.
The complete E84/R3208 FFN also passes in four fresh lifetimes (Auto, candidate,
candidate, Auto). Gate/up falls from 5.013/4.981 ms to 3.808/3.793 ms. Persistent
complete FFN falls from 9.734/10.092 ms to 7.280/7.281 ms, a 25–28% latency
reduction. The Q5_1 down policy remains Auto; these authored mixed-row bundles
are component evidence, not model throughput or recorded joint route vectors.

Five separate profiler observations per shape/policy corroborate improved
execution efficiency, without contributing clocks to the timing comparisons:

| M / P | Auto instructions /cycles /IPC | Corrected ordered3 instructions /cycles /IPC |
|---|---:|---:|
|38 /168|568.9M /294.1M /1.93|528.3M /223.0M /2.37|
|72 /168|1072.9M /527.6M /2.03|988.9M /371.5M /2.66|

Both shapes reduce L1 misses and LLC misses. These are aggregate per-worker
counters; no individual-worker stall percentage is inferred.

Evidence: `/tmp/qfn-cpu-ordered-three-balanced-r1128/` contains the coherent
build, focused JUnit, reversed geometry clocks, full-FFN samples, isolated
profiles and final ISA review. Auto and installed exact tables remain unchanged.
Next is fresh exact-choice qualification and installed-Auto verification before
any matched model performance claim.

## Collection and promotion boundary

No broad sweep has restarted. Completed compatible modern-family measurements
remain retained; Qwen2.5/original-Qwen3-only geometries stay excluded. New
candidate evidence cannot be relabeled as an old WideRows measurement.
The existing exact-key merge can add a new choice without discarding unrelated
installed rows or forcing historical receipts to contain it. These diagnostic
records are explicitly non-promotable and stay so.

The retained root histogram is diffuse: M64,63,445 and72 together represent
only3.23% of gate row-events, not measured kernel-time coverage. Installing a
few lucky exact points cannot establish a material model speedup. Any eventual
policy needs appropriate geometry coverage, installed-Auto verification and a
matched full-model result; none is claimed here.
