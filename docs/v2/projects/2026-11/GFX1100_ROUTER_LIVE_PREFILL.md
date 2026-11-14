# Router WMMA retained live-prefix qualification

Full-live M512 gains alone do not qualify automatic selection for a retained
graph. The production bucket planner normally chooses the smallest fitting tail,
but a supported ladder containing only 512 also admits small tails in that graph.
The existing native router therefore needs qualification across live counts.

## Evidence

The Release component retains explicit Dot4 and WMMA graphs at capacity512.
It publishes live counts 0,1,16,256,511,512 without recapture, for four model
width/expert geometries and both original and SignedHadamard32 input bases.
Each cell has 31 alternating paired samples of 16 complete operations, including
quantization, logits and restoration of the original expert-input Q8 bank.

An immutable full-live Dot4 snapshot owns the expected logits, Q8 codes and
scales. Each candidate receives fresh poison before its start event; every timed
pair checks all active bytes against that snapshot and all inactive/guard bytes
against poison. Transfers, observations and setup remain outside event timing.
The lifetime guard drains before host staging, graphs or tensors are destroyed.

All eight cases and 48 cells pass. The timing receipt reports no compiler overlap
and unchanged producer hashes. Ratios are Dot4 median latency divided by native
median latency; larger is better.

| K / experts | Basis | Live1 ratio | Live16 ratio | Live512 ratio |
|---|---|---:|---:|---:|
| 2048 / 256 | Original | 1.001 | 1.001 | 1.417 |
| 2048 / 256 | Hadamard32 | 0.858 | 1.022 | 1.369 |
| 2560 / 512 | Original | 0.789 | 1.036 | 2.520 |
| 2560 / 512 | Hadamard32 | 0.813 | 1.033 | 2.306 |
| 3072 / 256 | Original | 0.740 | 0.986 | 1.838 |
| 3072 / 256 | Hadamard32 | 0.769 | 0.987 | 1.697 |
| 8192 / 512 | Original | 0.629 | 0.849 | 0.910 |
| 8192 / 512 | Hadamard32 | 0.684 | 0.869 | 0.919 |

The single-row regressions are outside the paired-median bootstrap intervals
around parity, except K2048/original which is approximately tied. These results
retain Automatic on Dot4. The next experiment increases cooperating K waves to
reduce the native entry's serial subtree work while preserving its arithmetic.

Local evidence is under `/tmp/llaminar-gfx1100-wmma-prefill-evidence/`:
`router-live-prefix-receipt.json`, `router-live-prefix-summary.json`,
`router-live-prefix-tournament.log` and
`router-live-prefix-final-release-build.log`. The registered performance owner
is `V2_Perf_ROCmRouterQ8LivePrefix`. No full preflight or model-E2E result is
claimed by this component run.

## Production bucket domain

The default ladder has 448 and512, so ordinary non-owned M512 work has 449–512
live rows. `PrefillBucketUtils` selects the smallest fitting remainder; prefix
restoration and chunk tails are reselected. TP forwards the policy, and PP
preserves the selected physical width and live extent.

This is not a universal kernel invariant. Both
`LLAMINAR_PREFILL_GRAPH_MIN_SEQ=512` and
`LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES=512` are supported. With resident cap 512, a
prompt of 512+r tokens executes two M512 graphs, the second with live=r. The
ROCm row materializer computes `min(bucket,total-source_start)`. The serial
decode exception for a complete one-token suffix does not apply to a one-token
tail inside a longer prefill.

Code owners are `utils/PrefillBucketUtils.cpp`, `utils/PrefillGraphBuckets.def`,
`execution/runner/OrchestrationRunner.cpp`, `utils/DebugEnv.h` and
`kernels/rocm/ops/ROCmRowSelectKernels.hip`. Capture identity intentionally omits
live scalar values and does not carry a minimum-live/predecessor-bucket proof.
Capacity-only router admission cannot infer that default-ladder lower bound.
