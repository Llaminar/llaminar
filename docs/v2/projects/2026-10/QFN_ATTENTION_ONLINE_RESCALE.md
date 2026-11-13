# QFN FP16 attention online-max rescaling

The user-proposed ROCm change removes redundant online-rescale exponentials while retaining the original FP16/FP32 arithmetic program. Focused native, capture, compiler/resource and canonical accuracy gates are green. Component gains are small; the matched whole-model run establishes **no throughput gain**. This is not a full production-preflight certificate or a replacement for checkpoint 196's dashboard rate.

## Formula and exact scope

For running maximum `m`, tile maximum `t` and next maximum `m′ = max(m,t)`, the original scales are `a = exp(m−m′)` and `b = exp(t−m′)`, except that the original denominator-zero guard sets `a = +0` without evaluating it.

| Online state | Original exp evaluations | Candidate |
|---|---:|---:|
| Finite unequal maxima, nonzero denominator | 2 | 1: losing maximum only |
| Finite equal maxima, nonzero denominator | 2 | 0: both scales 1 |
| First finite tile: `m=−Inf`, denominator 0 | 1 | 0: scales (0,1) |

Counts describe these scale expressions, excluding probability and RoPE evaluation. The denominator-zero guard remains literal in every finite branch, including signed zero. Other nonfinite maximum cases retain the original expressions: no infinity/NaN arithmetic is replaced by a finite shortcut. The actual compiled runtime-input evaluator must return exactly 1 for both `exp(+0)` and `exp(−0)`.

At the maximum 2,051 selected tokens per query/head, the finite-input rescale budget falls from **58 to at most 25 exponentials for Prefill8**, and **33 to zero for Decode33**. Per-token probability exponentials remain unchanged.

The source retains the original `fmaxf`, tile multiplies, numerator/denominator FMAs, half conversions, reductions, barriers, partial stores and publication boundaries. The helper changes no selected IDs, rows, head ownership, storage, graph identity, launch count or allocation. Existing Prefill8/G4/256-thread and Decode33/G6/384-thread programs retain their independent arithmetic and pre/post-RoPE paths. Empty waves still launch and publish their required neutral metadata.

The implementation is in [ROCmQwenSparseOnlineRescale.cuh](../../../../src/v2/kernels/rocm/attention/ROCmQwenSparseOnlineRescale.cuh), consumed by [the FP16 producer](../../../../src/v2/kernels/rocm/attention/ROCmQwenSparseFP16Attention.cuh). The phase builder is unchanged.

## Compiler finding and strict regression

Control ISA already removed the previous-scale exponential from the 33-part program's sole iteration, but retained the online tile-scale evaluator after `t−max(−Inf,t)`. Eight-part control retains probability, tile-scale and denominator-guarded previous-scale sites; pre-RoPE also has theta-factor sites. Candidate CFG bypasses the online evaluator on the admitted finite branches. Static sites can remain or increase because the literal nonfinite paths remain compiled; static counts are **not** dynamic call counts or a latency attribution.

A test-oracle issue was retained and repaired rather than hidden. All 208 scale pairs and numerator updates matched, but four cases over three replays differed in denominator NaN payloads. Probe ISA showed independently lowered multiplies with reversed operand order: old `tileScale × tileDenominator` versus new `tileDenominator × tileScale`. With two distinct NaNs this selected `0xffc00000` versus authored `0x7fc12345`.

The test-only repair uses two threads per input, selects old/new scales, then executes one shared compiled multiply/FMA suffix and dynamically indexed typed `SideResult sides[2]` stores. No NaN comparison is weakened. The 512-thread capture authenticates its exact function/three arguments; input/output guards and all result words remain strict. A terminal pass marker is emitted only after accumulated nonfatal comparisons pass. Fresh probe ISA has the shared suffix at 0x1C5C–0x1C88 and device-loaded zero evaluation at 0x1CC4. The serving core exports no diagnostic mode or probe API.

Native proof covers 208 cases × 3 retained replays: finite/equal/first-tile, adjacent/subnormal/underflow differences, signed zeros, infinities and distinct NaNs against the original device evaluator. Complete 8/33 tests cover M1/M16, partial head shards, pre/post-RoPE, retained serial/grouped output and partial bytes, guards, immutable inputs and long→short→empty→partial→long reuse. Nonfinite score/denominator states stay in the independent probe because complete merge requires a finite positive denominator; admitted nonfinite V/numerator cases exercise the complete chain separately.

[Release functional evidence](../../../../parity-results/qwen-flash-next/attention-online-rescale-release-functional-r2/result.json) passes 13/13. The [focused Integration gate](../../../../parity-results/qwen-flash-next/attention-online-rescale-focused-gate-r1/result.json) passes the original FP16 arithmetic alias and new evaluator/replay aliases. The [canonical model panel](../../../../parity-results/qwen-flash-next/attention-online-rescale-model-fp16-r1/report.json) completes in 101.3 seconds: all 185 rows are byte-identical against five retained references.

## Measured component and counter results

Unprofiled complete producer+merge A/B/A uses the mean of the two control-arm medians. All exported output, partial and shape bytes match. These are component microseconds, not model throughput.

| Program / rows / context / key policy | Control µs | Candidate µs | Time change |
|---|---:|---:|---:|
| 33 / M1 /2051 / post-RoPE |17.860625 |17.833750 |−0.150%: neutral |
| 33 / M1 /768 / post-RoPE |15.283750 |15.191250 |−0.605% |
| 33 / M1 /65536 / pre-RoPE |20.164375 |20.153750 |−0.053%: neutral |
| 33 / M16 /65536 / post-RoPE |81.620525 |81.464900 |−0.191%: near neutral |
| 8 / M512 /65536 / post-RoPE |3172.830 |3149.070 |−0.7489% |

The first 2,051-token delta is smaller than its control-bracket spread; it should not be sold as an isolated speed win. See the [initial component summary](/tmp/qfn-attention-online-rescale-root-private/initial-component-summary.json) and [additional boundary evidence](../../../../parity-results/qwen-flash-next/attention-online-rescale-boundary-events-r1/result.json).

Separate singleton producer profiles at M1/C2051, Q6/KV1, preserve exact geometry. Wave-average `VALUInsts` changes **705.656566→692.656566 (−1.842%)** for 33 parts and **3261.968750→3235.734375 (−0.804%)** for 8 parts. These are diagnostic work counters, not profiled timing or achieved occupancy. [Strict profile comparison](/tmp/qfn-attention-online-rescale-root-private/profile-comparison.json).

Producer VGPR and LDS geometry remain unchanged. The changed attention producers and test probe have zero private storage and spills; canonical gfx906 builds pass the unchanged global spill guard. Eight-part used SGPR rises by 9 post-RoPE and 5 pre-RoPE; this resource cost is retained in the evidence. Allocation limits do not prove achieved occupancy. Fresh source/ISA and mathematical flags are recorded under [/tmp/qfn-attention-online-rescale-root-private](/tmp/qfn-attention-online-rescale-root-private), including `compiler-math-flags.json`, `isa-control`, `isa-candidate-r2`, and `probe-isa-r1/r2`. Exact control CFG/site provenance is in the [independent review](/tmp/qfn-attention-online-rescale-independent-review-private/REVIEW.md).

## Matched whole-model check

The final unprofiled Release A/B/A isolates this attention change against the immediately preceding qualified norm-fusion runtime. Both arms retain TP4 on four MI50s, RAM ngrams, sharded vocabulary, FP16 KV, M512, context 4096, the identical 512-token prompt, 256 emitted tokens, greedy seed 42, MTP/prefix cache off, and 28 physical workers on CPUs 28–55/NUMA 1. All diagnostic selectors and PerfStats timing are absent. Each arm has one warmup and three measurements; decode rates exclude the prefill-produced token.

| Runtime | Prefill median (range), tok/s | Decode median (range), tok/s |
|---|---:|---:|
| Control, six measurements |820.343 (820.081–821.397) |42.416 (42.195–42.519) |
| Candidate, three measurements |820.812 (820.096–821.114) |42.211 (42.189–42.522) |

Candidate/control changes are **+0.057% prefill and −0.482% decode**. The overlapping sample ranges do not establish either an overall improvement or a causal slowdown; retain the unfavorable median instead of selecting the fastest candidate sample. All nine natural 256-token sequences match exactly. The complete quiet/driver/runtime/retirement observer passes. See [aggregate and individual samples](/tmp/qfn-attention-online-rescale-model-benchmark-r1/summary.json) and [host-window result](../../../../parity-results/qwen-flash-next/attention-online-rescale-model-host-window-r1/result.json).

The user accepted the small component gain. This slice retains that measured work reduction without claiming progress in whole-model speed. Broader topology economy and full preflight remain pending; the prior 828-registration gate belongs to parent checkpoint `f52896b8`, with the changed attention covered by the fresh focused gates above.
