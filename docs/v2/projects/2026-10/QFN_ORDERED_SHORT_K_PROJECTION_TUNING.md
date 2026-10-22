# QFN ordered short-K projection experiment

2026-10-22. Base checkpoint: `7bfca677d`. The 32-column production dispatch
passes complete Unit, focused native tests, the frozen A1 Release accuracy slice
and the matched model throughput comparison. Accepted L74 medians are
**785.397 prefill / 33.882 decode tok/s**: **+0.089% / +1.797%** against the
interleaved control. All retained final logits are byte-identical. This focused
checkpoint does not replace the complete production-preflight qualification.

## Candidate and unchanged arithmetic

For a physical M1, signed-Q8 projection with complete K32 groups, the original
kernel assigns one thread to an output column and processes K groups serially.
The selected candidate assigns a 32-column cohort to each K32 group. Each thread computes
the original primary and residual integer dot products and their original FP32
scaled contributions. After one workgroup barrier, group zero folds those
contributions in the original ascending-K order and adds the two completed sums
once. Publication, quantization, prepared weight bytes and activation buffers
are unchanged. There are no global partials or extra communication.

The dispatch domain is physical M1, execution codebook 19, K64–512 and N≥4K
on both backends. K320/N10240 uses 320 threads per block, 320 blocks and
2,560 bytes of shared memory. The geometry header owns the column extent used
by dispatch, launch bounds and LDS indexing. Redistributing work adds two
contribution planes and a barrier; the complete publication/projection
transaction is measured below. The original-column diagnostic launcher remains
independent of normal dispatch. Public experimental launchers are removed.

## Functional and compiler evidence

The initial 64-column [Integration build](../../../../parity-results/qwen-flash-next/shortk71-integration-build/result.json)
passes for CUDA SM80/86/89/90 and HIP gfx906 with spill guards enabled.
The [three focused registrations](../../../../parity-results/qwen-flash-next/shortk71-focused/result.json)
pass: every K32 group count from 2 through 16, output tails, original Q8_1/Q8_K
sources, captured live→empty→live replay, source/buffer identity and actual
native launch resources. Outputs and all four publication buffers equal the
untouched production serial path byte for byte. All
[19 surrounding all-format registrations](../../../../parity-results/qwen-flash-next/shortk71-all-formats/result.json)
also pass, including CPU ISA/thread cases and both GPU backends. All driver
windows and process retirement checks are clean.

Final HIP code-object metadata gives 24–32 VGPRs, 26–28 SGPRs, 1–8 KiB LDS and
zero private storage or memory spills across the 15 specializations. K320 uses
24 VGPRs and 26 SGPRs. The final CUDA device-link object has all 60 expected
architecture/specialization combinations. SM86 K320 uses 40 registers, 5 KiB
shared memory and zero stack/local storage. Relocatable CUDA object metadata
reports unresolved shared allocation as zero; the final link and actual native
graph attributes establish the real allocation. Local static evidence is in
`/tmp/qfn-shortk71-integration-resources/`; these are resource bounds, not
measured occupancy or hardware throughput.

## Component timing and tile comparison

Both Release builds and the same-input comparisons pass. The retained
[component and profiler report](../../../../parity-results/qwen-flash-next/shortk72-tile-component-analysis/report.json)
keeps all eight alternating AB/BA batches, 512 graph submissions per sample and
64 warmups. Native-event intervals include the complete captured publication
and projection transaction, including submission gaps. Every batch verifies
exact outputs and all four activation buffers. The historical K320 HC source
and input are a fixed component workload, not a fresh accepted-model snapshot.

| Backend | Columns | Original median µs | Candidate median µs | Speed ratio |
|---|---:|---:|---:|---:|
| MI50 | 64 | 27.00140 | 21.53215 | 1.254× |
| MI50 | 32 | 27.00030 | 21.57175 | 1.252× |
| RTX 3090 | 64 | 5.83128 | 5.40228 | 1.079× |
| RTX 3090 | 32 | 5.83035 | 5.38747 | 1.082× |

The 32-column successor halves block threads and shared storage and doubles
the block count. It passes the same three focused registrations. Across all
15 HIP specializations and 60 CUDA architecture/specialization combinations,
static inspection reports zero private/local storage or memory spills. K320
uses 320 threads and 2,560 shared bytes; gfx906 retains 24 VGPRs/26 SGPRs and
SM86 retains 40 registers.

The subsequent [shape comparison](../../../../parity-results/qwen-flash-next/shortk73-shape-comparison/report.json)
passes all **180 backend/tile/shape cases**. Each tile covers K64–512 in K32
steps, the N=4K boundary, a one-column tail and a wider projection, for 45 shapes
per backend. Every measured candidate median beats its original-column control.
The 32-column tile beats 64 columns on 43/45 CUDA and 29/45 ROCm shape medians;
the QFN K320/N10240 case is effectively tied. This supports one shared
32-column policy, without fitting a separate tile to individual shapes. It is
not a claim of statistical superiority at every shape. K above 320 uses
explicitly labeled cyclic copies of original quantized groups; no weights are
requantized, and these synthetic shapes do not replace model accuracy checks.

Isolated MI50 counter passes identify one exact K320/N10240 candidate dispatch
per launch. Both tiles have zero LDS bank conflicts and less than 0.011% LDS
stalling. Their HBM fetch measurements are 3,420.94/3,413.88 KiB (64/32 columns),
close to the 3,400 KiB logical weight extent. VALU lane utilization is
99.982%/94.991%. These counters characterize the observed dispatch; they do not
establish whole-model latency or a single exclusive bottleneck. HBM write
counters can be zero when output remains cached and do not mean no stores ran.

Two initial oversized counter sets were rejected by the profiler and are
retained as failed attempts. Smaller sets completed normally with clean driver
windows. `MeanOccupancyPerCU` is unavailable on this gfx906 profiler, so the
occupancy collection remains incomplete. CUDA profiler attachment remains
suspended after the earlier driver failure; only static final-link resources
and unprofiled native-event timing are available. Profiled timings are excluded
from the table above.

## Promotion requirements

The working-tree promotion uses actual GEMMStage capture in the native fixture,
with the independent original-column launcher as its serial oracle. It adds
matching grouped M2/3/15/32/64/65/512 live→empty→live cases and exhausts device-free
dispatch eligibility over M1–4096. A narrow PerfStats counter records accepted
host submissions/capture with the actual immutable geometry; retained replay
does not increment that counter. Focused registrations remain explicit in
production preflight.

Final production promotion requires fresh builds, focused native/all-format
checks, complete Unit, the frozen model accuracy slice and a matched Release
comparison. The previously profiled kernel arithmetic is retained, but the
rebuilt artifact must still pass resource checks. A component win alone does
not change the dashboard's accepted model rates. At 97 affected projections per
GPU/token, fully exposed savings would improve A1 decode by roughly 1.8%; that
estimate is not an observed model gain.


## Completed focused production qualification

The final production-dispatch [focused registrations](../../../../parity-results/qwen-flash-next/shortk74-focused/result.json)
pass 3/3, including independent original-column comparisons, actual GEMMStage
capture, grouped serial equivalence and counter metadata/replay semantics. The
[surrounding all-format registrations](../../../../parity-results/qwen-flash-next/shortk74-all-formats/result.json)
pass 19/19. Driver windows are clean and every native process retired normally.
Final Integration static inspection in
`/tmp/qfn-shortk74-integration-resources-checked/` retains all 15 gfx906 and 60
SM80/86/89/90 specializations with zero private/local storage or memory spills;
CUDA resources exactly match the previously measured 32-column candidate.

Two initial builds exposed test-fixture include problems: a stale DeviceManager
header path and namespace pollution from a broad CUDA test utility. The final
fixture owns only its two collector environment settings and no longer imports
that utility. The failed build artifacts are retained. The first static-resource
helper also incorrectly equated Hopper's compiled total with declared shared
arrays; its corrected check preserves raw resources and compares the full CUDA
inventory with the prior measured candidate. None of these corrections changes
production arithmetic. Complete Unit, final Release accuracy and the
whole-model throughput comparison are now green.

The [complete Unit gate](../../../../parity-results/qwen-flash-next/shortk74-unit/result.json)
passes **756/756 registrations** in 116.21 seconds. The final
[Release build](../../../../parity-results/qwen-flash-next/shortk74-release-build/result.json)
passes, including all shipped GPU targets. Static Release resource inspection
also passes in `/tmp/qfn-shortk74-release-resources/`. Fresh
[production-stage component timing](../../../../parity-results/qwen-flash-next/shortk74-component-analysis/report.json)
retains the same gains with an independent original-column reference:

| Backend | Original median µs | Production median µs | Speed ratio |
|---|---:|---:|---:|
| MI50 | 26.98125 | 21.57695 | 1.2505× |
| RTX 3090 | 5.832345 | 5.387845 | 1.0825× |

The [frozen A1 Release accuracy slice](../../../../parity-results/qwen-flash-next/shortk74-release-accuracy/report.json)
passes every required group, reset, driver and native-retirement check. All
[26 retained final-logit files / 185 rows](../../../../parity-results/qwen-flash-next/shortk74-release-accuracy/selector70-logit-byte-comparison.json)
are byte-identical to selector70. Thus the measured speed improvement adds no
numerical change on this slice. The diagnostic N2047 rows are included in the
byte comparison but remain outside the required acceptance groups.

| Required panel | Phase | Mean KL (nats) | Top-1 agreement | Mean top-5 overlap |
|---|---|---:|---:|---:|
| Short | Prefill | 0.033535869 | 93.75% | 87.50% |
| Short | Decode | 0.032070057 | 97.92% | 92.29% |
| Heldout short | Prefill | 0.009691368 | 100% | 95.00% |
| Heldout short | Decode | 0.003837619 | 100% | 94.17% |

The [model path record](../../../../parity-results/qwen-flash-next/shortk74-release-accuracy/ordered-short-k-path.json)
contains 776 accepted host submissions/captures for physical M1/N10240/K320,
execution codebook 19, 320 threads, 320 blocks and 2,560 declared shared bytes.
This confirms selection in the real model; it is not a per-token, per-device or
graph replay counter. The frozen accuracy scope remains a local diagnostic with
required groups green, rather than a new complete production preflight or
whole-corpus certification.


## Matched model result and checkpoint

The [complete A/B/B/A comparison](../../../../parity-results/qwen-flash-next/shortk74-abba-comparison.json)
passes all four original process arms, with one warmup and three measured
requests per arm. No samples were excluded or replaced. Each runtime uses the
same repaired RCCL dependency and automatic protocol selection. Runtime/source
identity, driver windows, native retirement, zero-cache phase counts and all
**twelve 256-token output sequences** pass their checks.

| Phase | Control median tok/s | Production median tok/s | Change | Control range | Production range |
|---|---:|---:|---:|---:|---:|
| Prefill | 784.698992 | 785.397376 | +0.089% | 784.324663–785.842941 | 784.584272–786.028355 |
| Decode after prefill | 33.283484 | 33.881558 | +1.797% | 33.206288–33.449608 | 33.853745–33.926483 |

Each role contributes six measured samples. Prefill ranges overlap; decode
ranges separate in this observation. This is a measured improvement on the
frozen workload, not a statistical confidence bound or a long-context result.

The contract is the original four `Qwen3.8-Flash-Next-UD-Q4_K_XL` shards on four
32 GiB MI50s, `TensorParallel + GateUpOwnedDownColumns`, all experts on GPU and
ngrams in RAM. The shared Q8 MTP GGUF is supplied but MTP is disabled. Release,
FP32 activation storage, FP16 KV, M512, context capacity 4096, prefix cache off,
PerfStats off, 28 physical NUMA1 workers (`taskset -c 28-55`) and ordinary MPI
bootstrap remain fixed. The prompt is `b" test" * 512` (token 1228 repeated 512
times; SHA-256 `d17afaa1069c1800ba2db70c90cc82ad85c213ec245d07e6916bf089fe06fe21`).
Greedy generation uses temperature 0, top-k 1, top-p 1, seed 42 and 256 output
tokens, giving 255 after-prefill decode steps. The six arithmetic selectors are
unchanged:

```text
LLAMINAR_QFN_FA_OUTPUT_ACTIVATION=single_q8
LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION=single_q8
LLAMINAR_GDN_PROJECTION_ARITHMETIC=fp16_moe_alpha_beta
LLAMINAR_QFN_GDN_QKV_ACTIVATION=single_q8
LLAMINAR_QFN_HC_UP_ACTIVATION=two_term_q8
LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16
```

The control runtime receipt is `ebfa3973d8352101d6b8d0de274ab6153efb01052cdfee547a740f643a302682`;
the production receipt is `9d9f832116c3423a93e0df986203d8f6ab28a2ac31dda740cab22e9d652b2707`.
The archived executables load their own immutable core and the same selected
external dependencies. Model payloads were authenticated by metadata, without
hashing the GGUF or cache contents.

Castagna's retained 42.959 decode tok/s remains ahead. The next focused probe
separates the existing fixed-row rank gather and ascending-rank fold, with
single-transaction and 97-transaction captured controls. It will preserve rank
order, exact live communication extents and the production transport branch;
component timings alone cannot establish exposed model time.
