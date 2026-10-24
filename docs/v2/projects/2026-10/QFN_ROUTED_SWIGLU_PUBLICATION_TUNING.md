# QFN routed SwiGLU publication screen

The private ROCm experiment 181 is **parked**, with production remaining at
checkpoint 180. It preserves exact bytes but yields only modest component gains
for the two or three local routes relevant to the four-MI50 workload. It has
not received all-format qualification or a real-model throughput comparison.
The experimental source was removed from the working tree after archiving it.

## Complete-boundary measurements

The Release screen retains hidden quantization, the original Q4_K-source
gate/up arithmetic, routed SwiGLU/Q8 publication, and the original asymmetric
Q5_1 down consumer. Dimensions are M1, H2560, F640 and local down width 640, with source
down-policy width 2560 and ten physical route slots. Device-owned route counts
select useful work; neither physical storage nor launch capacity is reduced.
Weights and activations are synthetic, not captured real-model values.

| Active local routes | Original median µs | Paired median µs | Time reduction |
|---|---:|---:|---:|
| 10, full-route control | 200.4225 | 198.1575 | 1.130% |
| 2, participant 0 of 4 | 56.5323 | 56.2673 | 0.469% |
| 3, participant 2 of 4 | 73.9473 | 73.1398 | 1.092% |

Each observation uses 32 complete operations per retained graph, five warmup
pairs and 12 alternating measured pairs. Full/empty/owned transitions, original
FP32 gate/up words, Q8 codes/scales, final local down output and inactive guards
pass byte checks before and after timing. The subset cases retain the same
physical graph and publish their authored ownership before warmup. These are
captured component intervals, not decode tok/s or predicted model gains.

The new leaf combines both projection roles in a 512-thread CTA, retaining
four partition workers per role, the original source-selected K partitions,
ascending RN folds and deterministic routed SwiGLU/Q8 helpers. Two disjoint
partial banks use 8192B dynamic LDS. The original FP32 gate/up stores remain;
the separate quantizer launch and its input reads are removed. Production
selectors were never changed.

## Separate resource and profiler evidence

All 12 isolated singleton counter joins pass for the original producer, original
quantizer and paired leaf. Each selected dispatch is occurrence 289, authenticated
against the complete 50-graph replay schedule and 800 occurrences per role.
Profiled event times are excluded from the table above.

For ten active routes, estimated vector-wave instruction issues decrease from
877,252 to 848,152 (**3.317%**); fetched traffic decreases from 20,054.5625 to
20,004.375 KiB (**0.250%**). All three kernels report zero LDS bank conflicts.
The new leaf uses 25 VGPRs (28 allocated) and 90 SGPRs (96 allocated), with zero
private storage or spills. Its original producer, quantizer and down consumer
have byte-identical compiled bodies/resources to the frozen 180 control.
Achieved occupancy is unavailable in the installed gfx906 metric catalog;
wavefront counts and static residency limits are not occupancy measurements.

The initial fixture failed before candidate execution because HIP's
`hipGraphKernelNodeSetParams` validates the replacement block size against the
old function's launch bound. Cold graph construction now adds the new node,
authenticates all original arguments and reconnects the existing consumer.
The failed receipt remains retained; no HIP runtime change or production
execution workaround was made.

## Retained evidence

- `parity-results/qwen-flash-next/paired181-functional-r{2,3}`: exact-byte gates.
- `paired181-events-r1`, `paired181-two-owned-events-r1` and
  `paired181-three-owned-events-r1` under that root: unprofiled raw samples.
- `paired181-{control,paired,quantizer}-{wavefronts,valuinsts,fetchsize,ldsbankconflict}-r1`:
  independent profiler receipts and strict native joins.
- `/tmp/qfn-paired181-complete-screen-r1`: final private source, bounded source
  identities, frozen Release assets and result manifest.
- `/tmp/qfn-routed181-isa-private`: bounded compiled-code comparison.

Checkpoint 180's 819.594 prefill / 41.030 decode tok/s remains the production
measurement. Castagna's 42.959 decode target is still open. The next component
screen combines the HC ordered reduction with its activation/TwoTermQ8
publication while preserving the original down producer.
