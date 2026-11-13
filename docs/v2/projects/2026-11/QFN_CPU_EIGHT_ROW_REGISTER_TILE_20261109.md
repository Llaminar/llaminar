# Eight-row CPU expert tile: closed without promotion

The proposed AVX-512 eight-row tile passes its focused arithmetic and allocation
checks, but does not show a reproducible Q4_K component speedup. Production keeps
its existing paired-row implementation. The diagnostic helper, scale-loader
experiments and temporary test registration have been removed from the source;
the proposed production integration was never applied.

## Scope and arithmetic

Eight independent rows shared the decoding of one N16 weight vector inside the
original N64 panel and fixed ordered K partitions. The experiment retained Q8
activation bytes, source weight bytes, separate FP32 contribution operations,
partition order and output ownership. It covered Q4_0, Q4_1, Q4_K, IQ4_NL and
IQ4_XS, without changing Auto, exact dispatch entries or any public row policy.
This was a single-calling-worker component, not a complete expert FFN, model
route replay or production inference benchmark.

The first LUT body spilled two reduced accumulators. Removing unreachable
expanded-IQ3_S scale handling reduced this to one. Exact low-six-bit sign
extension removed a vector mask, but GCC still retained a bias vector and
spilled one reduction. The final candidate forced the immutable four-byte -32
bias into an AVX-512 memory-broadcast operand. Inspection of the original
Release owner's compiled object then found zero arithmetic vector stack traffic
in all three decoder families. Remaining scalar stack values held metadata;
the explicit read-only bias was not saved arithmetic state. Exception calls
were on invalid-bank paths, not inside valid dot-product work.

The original native owner passed 80 format/geometry/bank/repetition cases,
including signed Q8 extremes, N tails, malformed-input no-write checks and
the retained asymmetric NaN-payload case. All 65,536 raw coefficient words
matched both the generic SIMD scale entry and the scalar source authority.
Six existing ordered-nibble, unsigned-dot and complete fused-completion CTest
registrations passed across AVX2 and AVX-512 runtime dispatch. These are focused
checks, not full production preflight or a new AVX2 eight-row implementation.

## Matched component clocks

Release, GCC 14, actual AVX-512, Q4_K, M8/N320/K2560, original 16 K partitions,
one physical worker on NUMA1 CPU28. Each arm retained 128 warm intervals of
eight calls and 40 measured samples of eight calls. PerfStats recording was
disabled inside clocks. GNU waiting variables were unset. Both fresh process
lifetimes used the same coherent executable/core/session closure and reversed
candidate order. All raw samples remain retained.

| Order | Existing four pair calls, µs | One eight-row call, µs | Pair/eight speedup |
|---|---:|---:|---:|
| Pair, then eight | 62.55175 | 62.00225 | 1.0089× |
| Eight, then pair | 59.09975 | 61.87850 | 0.9551× |

The first order is approximately flat; the reversed order makes the candidate
4.70% slower. Removing spills and reducing source-level weight loads did not
establish an economic improvement. No full-FFN comparison, isolated hardware
counter profile, exact-table installation or model rerun is warranted for this
candidate. No model gain is claimed toward the 500/40 target.

The first timing launch used unnecessary `sudo`, preventing its unprivileged
observer from reading `/proc` descriptor metadata. Its native test and driver
bracket passed and all processes retired, but its observation receipt remains
failed and its timings are excluded above. The two accepted lifetimes retain
the ordinary user identity, complete observation, unchanged binary metadata,
clean driver brackets and empty owned session directories.

## Evidence and next direction

`/tmp/qfn-cpu-eight-row-rocm101-root/` retains the final diagnostic source patch,
compiled-object disassembly, build logs, focused tests and `conclusion.json`.
`/tmp/qfn-cpu-eight-row-rocm101-next/economy/{forward-user,reverse-user}/`
retains the accepted original-owner clocks, raw samples and native retirement.
The rejected observation is preserved separately under `economy/forward/`.
The parked full-pair production patch remains private and unapplied.

The [OpenMP investigation](QFN_CPU_OPENMP_WAIT_ATTRIBUTION_20261109.md) also
keeps the existing waiting policy. Gate/up, SwiGLU and down already share one
team; their input/output publication scopes cannot justify a large prefill
gain. The next bounded GPU target is the previously measured prepared-Q8
NativeCooperative row tile, whose four physical cells still require exact
producer-bound qualification and post-install model evidence.
