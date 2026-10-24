# QFN NativeQ8 producer/reducer fusion trial 178

## Disposition

**Rejected on 2026-10-24.** Every tested variant is slower than the qualified
177 control for the ordinary decode used by the target model. The private
entrypoints and fixtures were archived and
removed; production dispatch was never changed. The current model checkpoint
remains **819.876 prefill / 40.328 decode tok/s**, with the Castagna target of
42.959 decode tok/s still open. No model benchmark or full-preflight claim is
made for 178.

The retained 175 model trace identifies the HC read-down projection as
Native19/Q8_0, N320/K10240, M1, sixteen ordered K partitions, **ordinary work**,
97 calls per decode step per GPU. The experiment keeps the original Q8
quantizer, packed weights, FP32 block commits and partition order. It changes
CTA ownership so each block owns four or eight output columns and all sixteen
of their original partials. Each partial is still written to its original
global address, then the existing ascending FP32 fold runs inside the CTA.
This removes the separate reducer launch without removing its arithmetic or
changing workspace admission.

## Matched component results

Release, gfx906 MI50 device 0, physical CPUs 28–55, original production tensor
and workspace owners. Each sample measures a retained complete projection:
original quantizer + producer + reducer versus the same quantizer + fused
producer. There are 32 complete operations per graph, five paired warmups and
twelve alternating event samples. Preparation, byte checks and terminal
observations are outside timing. All samples are retained, without filtering.

| Candidate | Control µs | Candidate µs | Latency increase |
|---|---:|---:|---:|
| Eight columns, 128 threads |18.445|24.615|33.45%|
| Four columns, 64 threads |18.440|24.535|33.05%|
| Eight columns, prefetched fold |18.440|24.170|31.07%|
| Four columns, prefetched fold |18.445|24.093|30.62%|

Device-counted active work is roughly flat in the initial variants, while
empty work benefits from removing a launch. Those results cannot qualify the
ordinary path used by the measured model. No dispatch policy was installed
from a favorable inactive case.

## What profiling established

Sixteen isolated rocprof singleton passes authenticate exact native functions,
graph arguments, geometry, two setup primers, verification, warmups and the
selected first timed operation. Profiling durations never enter the event
comparison. The original producer and reducer execute 96 and 6 waves;
the fused variants execute 80 waves.

For the eight-column baseline, the producer records 291.166667 VALU
instructions per wave and the reducer 64.5, versus 538 for the fused kernel.
Together with the wave counts, that estimates **51.9% more SQ vector instruction
issues** across the fused producer versus the original producer and reducer.
This excludes the unchanged quantizer and counts wave instruction issues,
not per-lane operations or FLOPs. Candidate and original producer fetch approximately 3411 KiB
each; the original reducer additionally fetches 20.0625 KiB. LDS bank conflicts
are 0% in the measured eight-column baseline singleton pass; the other variants
were not measured for this counter. These observations support an instruction/scheduling cost, rather
than increased total fetched bytes, as a cause of the slowdown; they are not
an additive latency model or an achieved-occupancy measurement.

ISA inspection shows the original producer broadcasts activation loads through
scalar registers. Making K partitions lane-dependent moves those loads and
loop/address work into vector instructions. The initial fold also serializes
sixteen LDS-load/wait/add sequences. The final trial uses tied read/write
register operands to materialize all leaves first: emitted code has eight
grouped LDS reads before the exact ascending sixteen-add fold. This reduces
latency slightly but leaves the overall candidate uneconomical.

All four final specializations use 28 VGPRs and zero private storage/spills.
Static LDS is 512 B for eight columns and 256 B for four. The compiler removes the
hardware barrier for the single-wave four-column case; that alone does not
recover performance. All 356 nontrial function bodies are byte-identical
between the column-layout and prefetched-fold builds. Loaded gfx906 code
objects were extracted by their exact profiler URI and match the inspected
compiled objects.

## Correctness and evidence

Both column geometries and ordinary/device-counted work pass captured replay,
including full→empty→full transitions after the work scope expires. Checks
compare complete output and original partial bytes, quantized activations,
scales, unchanged inputs, poisoned unused capacity and output guards. Every
repeated control and candidate pointer/scalar/work tuple is authenticated;
captured copies, clears, callbacks and extra compute roles are rejected.

The first functional attempt exposed a fixture setup omission: its count
publication event had not been joined before capture. The production
TransferEngine guard correctly rejected it. Joining every declared external
input on the exact consumer stream fixed the fixture without weakening the
guard or adding a blocking synchronization. The failed receipt is preserved.

The local aggregate is
`parity-results/qwen-flash-next/native178-rejected-r1/result.json`; it indexes
the event/profile receipts and archives all four private source files and
their patch. Individual failed attempts remain separate. Compiler, loaded
code and independent review evidence live under the corresponding
`/tmp/qfn-native178-*` directories. These local diagnostic artifacts are not
published corpora or image certificates.

Next: test the existing parallel integer prefix scan inside legacy
512-expert grouping. Preserve its stable route order, metadata ownership and
literal FP32 weight copies; first measure complete captured grouping, then
the real ImportDown boundary and matched model throughput if it wins.
