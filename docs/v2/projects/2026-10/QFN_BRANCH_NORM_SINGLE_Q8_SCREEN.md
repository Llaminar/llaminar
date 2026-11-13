# Branch normalization and HC-down Q8 fusion screen

**Rejected, 2026-10-25.** The synthetic complete-operation screen preserved every
checked byte but was 17.74% slower. The test-only candidate was archived and
removed; production code and dispatch were never changed. No model rerun was
needed to reject it.

The actual HC read-down boundary uses SingleQ8, independently of the later
HC-up TwoTerm policy. This screen retained Q8_0/Native19 weights, ordinary M1,
four branches of H2560, K10240/N320, and sixteen ordered projection partitions.
It combined the original 256-lane branch RMSNorm with Q8 preparation, preserving
the full normalized FP32 output required by ResidualRead. The original projection,
reducer, workspace, weights and publication events were unchanged.

| Complete operation, µs | Original | Fused |
|---|---:|---:|
| Retained native graph |30.534953 |35.952453 |
| Including equal output publication |30.969968 |36.389936 |

Release/gfx906 timings use 32 complete operations per captured graph, five paired
warmups and twelve alternating A/B pairs. Inputs are immutable and restored
outside timing. Four dirty states and every actual timed bank pass exact checks
for normalized values, projection outputs, workspace, immutable operands and
guards. Both lifetimes pass the quiet-host, runtime-loader, driver and normal
retirement checks. This is a synthetic component result, not model accuracy or
whole-model throughput.

The linked test-owned kernel uses 16 VGPRs, 56 SGPRs, 1 KiB LDS and no private
storage or spills. It launches four 256-thread blocks. Fusion moves quantization
from 320 independent blocks into those four norm blocks, with ten iterations per
lane. That concentration is a plausible cost; it is not a measured critical-path
attribution. The experiment does not revive the separately rejected NativeQ8
producer/reducer fusion.

The [retained result](../../../../parity-results/qwen-flash-next/branch-norm-single-q8-rejected-r1/result.json)
indexes the patch, source identities, complete event samples, strict byte proof
and linked ISA. The serving core's file identity remained unchanged throughout.
Next investigate the guarded indexer query projection's empty-block launch cost.
