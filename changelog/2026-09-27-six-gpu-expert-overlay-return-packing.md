# Six-GPU ExpertOverlay return packing

This tuning session targets Qwen 3.5 122B on two RTX 3090s and four MI50s,
with CUDA continuation, automatic expert capacity, dynamic residency and
dynamic-depth MTP. It builds on the earlier mapped-return consumer changes;
weights, FP32 activations, FP16 KV, MTP policy and benchmark intent are unchanged.

## Baseline and measured result

The comparison is release `2026-09-27.1`, whose AVX512 runtime source is
`557f0f97d15d447850b77b7ce4e91bb0663331b3`. The exact immutable image was
remeasured on the same host before tuning. The canonical benchmark has a
512-token prompt, 256 generated tokens, one warmup and three measured requests;
reported throughput is the median. Context is 8192. PerfStats/profilers are off
for these measurements.

| Runtime | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Published release | 153.59 | 25.03 |
| Same release image, remeasured | 155.72 | 25.26 |
| Earlier consumer-only optimization | 244.17 | 37.15 |
| Cooperative descriptor read | 244.02 | 37.58 |
| Row-owned return packing | 321.92 | 39.75 |
| Independent repeat of row-owned packing | 320.88 | 39.67 |
| Root-local cache reads and vector route staging | 322.56 | 40.35 |
| Row-owned single-row draft returns | 321.20 | 41.17 |
| Floating SwiGLU computed once | 330.73 | 41.23 |
| Floating empty-slot reduction elision | 331.59 | 41.40 |
| Native warp/wave floating projections | 343.16 | 42.33 |
| Unified cooperative return validation | 343.90 | 42.88 |
| Larger bulk return-copy blocks | 349.96 | 42.48 |
| Paired independent CUDA return loads | 351.49 | 42.38 |
| Continuation-owned return-index reads | 351.60 | 42.62 |
| Skip absent-route split-K scratch | 352.87 | 43.13 |
| Bounded runtime counts and one stable-position search | 351.75 | 43.66 |
| Final independent process 1 (three measured requests) | 351.89 | 43.37 |
| Final independent process 2 (three measured requests) | 352.28 | 43.63 |
| Final independent process 3 (three measured requests) | 351.77 | 43.27 |
| Final cohort (all nine measured requests) | 351.89 | 43.37 |

The requested goal is 125% faster prefill and 75% faster decode: at least
345.58 and 43.80 tok/s respectively. The final nine-request cohort improves
prefill by **129.11%** and decode by **73.29%**; decode remains below its target.
All three independent processes are retained rather than selecting the fastest
repeat. Their decode samples are 42.83/45.85/43.37, 42.81/45.38/43.63 and
42.57/45.83/43.27 tok/s. The final reports are `final-repeat-{1,2,3}.json`.
Paired loads have not established a decode benefit. The complete generated
token streams and MTP counters in all nine requests match the release exactly.
Each measured request
also completes physical tier promotion/demotion and within-tier moves;
movement was not disabled to obtain the result.

The CUDA canonical quantized down producer also wrote split-K zeros for absent
routes, and its reducer read them before discarding them. Both now reject those
routes before touching private scratch; public collective contributions are
still explicitly overwritten with positive zero. ROCm already follows this
contract. CUDA Q8_0 captured M16 pipeline times with 0/1/3 live rows fell from
109.57/137.22/173.06 to 75.78/106.46/138.24 microseconds. The fully populated
case remained near 1.14 milliseconds. All-format CUDA/ROCm preflight replay
tests now poison private partials and both public outputs between transactions.
The real-model token streams and MTP counters remain unchanged.

The runtime grouping kernel also scanned every route separately for every
expert. Both GPU backends now build exact bounded integer counts in the
existing shared-memory offset bank, then recycle it after the prefix scan.
Integer increment order cannot affect these counts (at most 256 routes); the
serial original-route order and all floating weight bits are unchanged. ROCm
also retains each route's expected destination instead of recomputing its
stable position during validation. New symmetric preflight tests cycle a
retained graph through empty, all-to-one, distributed and sparse routing for
20 replays, with 8/17/256 experts and 1/2/16/32 rows. The focused six-test gate
includes all-format expert arithmetic and both compiler-resource checks.

For the original row-owned return-packing slice, the full gate passed: 673 Unit tests and all 424 production-preflight
tests (1097 total). The canonical six-GPU HTTP cell passed all 45 checks,
including eight long-context checks and 2,048-token structured generation.
Shutdown returned GPU VRAM to the exact initial 42 MiB and recorded no new
AMDGPU/NVIDIA warnings. This is local tuning evidence, not a new image
certificate or an official high-water update.

The combined retained candidate now passes a fresh complete gate: 673 Unit
and 428 production-preflight tests, 1,101 total, in 1,110.4 seconds including
the full test-target rebuild. Its receipt is `gates-final/prerequisites.json`.
The fresh combined-candidate HTTP run (`e2e-final.json`) is **red: 44/45**.
All eight long-context checks, 2,048-token structured generation, prefix-cache
checks, streaming and tool calling pass. Shutdown is clean and GPU memory
returns to 42 MiB, but the driver observer catches a new
`amdgpu_amdkfd_restore_userptr_worker` CPU-hog warning at 13:59:36 UTC.
This is not a clean server/image certificate. Its reported count of 131 is
host-wide, exponentially rate-limited history for that worker; earlier counts
through 67 were logged the previous night. The cause of the current
invalidation/restoration work is under investigation; the warning is not
suppressed or reclassified as a pass.

A subsequent unchanged full HTTP run under passive kernel tracing passes
**45/45 checks in 260.81 seconds**, with zero USERPTR restore-worker executions
and no new driver warnings (`e2e-userptr-traced.json`). The tracer's positive
control observes 20/20 restore executions when reclaiming an unregistered
neighbour sharing a huge page; the boundary-protected control observes 0/20.
Thus an empty trace is meaningful, but this clean rerun does not identify the
allocation behind the original intermittent failure. The loaded driver still
queues USERPTR restoration on the CPU-bound `system_freezable_wq`; its
`WQ_UNBOUND` hint concerns driver scheduling and would not itself prevent
mapping invalidation or the associated process-wide GPU queue pause. The
checkpoint retains this unresolved caveat rather than claiming a driver fix.

## Mechanism

Subsequent dispatch-side experiments were rejected. Cooperative dispatch
descriptor reads plus entry-parallel CSR validation reduced a ROCm 16-row
microbenchmark from 61.60/71.84 to 44.48/55.20 microseconds (sparse/full), but
the real model remained at 320.34/39.67 tok/s. Row-owned dispatch
materialization further improved isolated CUDA copying but regressed the real
model to 313.49/39.53 tok/s. Both production changes were removed; their
ready-packet economy fixture and wide-payload regression coverage are retained.
The original return-packing improvements below remain installed.

A native-subgroup integer dispatch scan removed fourteen block barriers per
chunk but produced no full-model gain (343.85/42.82 versus 343.90/42.88).
It was removed. Profiling continues on the retained candidate; barriers alone
are not evidence of a critical-path bottleneck.

A CUDA down-projection experiment compacted live routes in groups of 32.
It substantially improved the sparse isolated pipeline but left real-model
decode at 43.16 tok/s versus 43.13 before the change. It was removed; the
adversarial noncontiguous live-route coverage remains. The new
`GPUExpertPipeline` performance fixture covers every registered quantized
expert format plus FP16/BF16/FP32 on both GPUs and remains outside preflight.

Two further trials did not establish an end-to-end win and were removed:
256-thread CUDA down-reduction blocks (351.49/43.74 tok/s), and a shared
integer membership bitset replacing the stable-position scan
(351.52/43.60 tok/s). The bitset passed the byte-exact replay gates and had no
memory spills, but the extra representation was not justified by model speed.
The retained grouping change uses the existing offset bank for integer counts
and removes ROCm's duplicate position search, without additional shared storage.
The fresh whole-model ROCm trace confirms its complete-plan specialization
fell from 30.37 to 19.55 microseconds per dispatch in a decode-only interval
(about 36%). This is kernel attribution, not an unprofiled throughput score.

The shared CUDA/HIP packet kernels unnecessarily reread mapped route metadata
for every FP32 output element. These are fresh system-memory reads, so ordinary
GPU caching cannot amortize them. The producer now assigns one complete route
row to a block: two lanes fetch original/compact identities once, then the block
stripes the independent payload stores. A bounded grid strides over live rows.
The same shared-memory lifetime barrier covers unaligned widths and repeated
grid-stride iterations. No extra persistent buffer, event, graph node or host
operation is added.

Return validation also loads its acquired 64-byte descriptor cooperatively,
instead of issuing eight serial lane-zero mapped reads. The prior system
acquire and subsequent acknowledgement still own visibility and lifetime.
Every existing identity, geometry and ordering validation is retained; invalid
grants exit uniformly before any cooperative barrier.

## Isolated evidence and regression coverage

The `MappedReturnPackEconomy` performance fixture captures the production pack
entrypoint with a ready interior-stage grant. Setup/reset and source upload are
outside GPU event timing. It verifies all returned and untouched bytes. Its
finite timing proof is separate from the full peer protocol integration tests.

| GPU, 3072-wide payload | Capacity/live routes | Before µs | After µs |
|---|---:|---:|---:|
| CUDA | 128 / 32 | 2313.22 | 47.10 |
| CUDA | 128 / 128 | 8153.09 | 139.26 |
| CUDA | 4096 / 512 | 18936.83 | 607.23 |
| CUDA | 4096 / 4096 | 145425.42 | 6321.15 |
| ROCm | 128 / 32 | 279.20 | 54.56 |
| ROCm | 128 / 128 | 1053.44 | 148.00 |
| ROCm | 4096 / 512 | 3348.64 | 501.28 |
| ROCm | 4096 / 4096 | 25751.00 | 3847.35 |

CUDA Nsight Compute reports 22 registers/thread, zero local spilling requests,
100% theoretical occupancy and 17.46% achieved occupancy for the sparse
128-block/256-thread pack; only 32 blocks carry payload. The diagnostic memory
throughput is 9.51 GB/s. ROCm has 15 VGPRs, 33 SGPRs, zero private scratch and
zero register spills in the code object; rocprof reports 16 allocated VGPRs and
zero scratch, with a 31.36 µs median for the payload kernel alone. The event
timings above include its metadata kernel too.

The cooperative CUDA descriptor validator measures 7.41 µs median versus
11.66 µs in the same captured round-trip fixture. CUDA reports 44 registers
and zero spills; ROCm has 22 VGPRs, 42 SGPRs and zero scratch/spills. Its ROCm
captured trace records a 13.44 µs average, not a cross-vendor speed claim.

The complete mapped-packet preflight entry passes with new twenty-generation
replay tests in both continuation/follower role directions. Additional
257-row, width-67 cases cross the bounded 256-block route grid and check tails
and sparse original/compact mapping over twenty generations. These run in
`V2_Integration_MappedActivationPacket_CUDA_ROCm`, already explicitly included
in `ProductionTestPreflight`. Performance tests remain outside preflight.

The first whole-model traces collected only startup probes because the profiler
followed a discovery rank instead of the selected execution owner. The saved
plan had membership `[1,0]`: CUDA execution rank 0 lived in discovery process 1.
After resolving that mapping, Nsight Systems recorded 1,273,952 real CUDA kernel
executions and rocprofv3 recorded 824,365 HIP dispatches, including retained
inference. The diagnostic applies an unprofiled saved plan to prevent profiler
overhead from changing auto selection, and enables `LLAMINAR_PROFILER_NORMAL_EXIT`
so normal-exit handlers flush trace buffers. CUDA/ROCm skills now document this
procedure. Canonical timing continues to use auto and no profiler flags.

Whole-model traces include startup, calibration and overlapping persistent
service/wait kernels; their total busy times must not be presented as inference
critical-path latency. An eight-pass isolated NCU validator run exceeded the
fixture's two-second host-submission assertion because of profiler replay;
the unprofiled functional suite passes, and no timeout was relaxed.

## Follow-on measured hotspots

Single-row MTP returns repeated original/compact route metadata reads per
payload element too. Their fused consumer and canonical materializer now give
one route to a block/subgroup, retaining the same acquire/acknowledgement
protocol. The four-lane, 3072-wide captured transaction fell from 97.28 to
27.65 microseconds on CUDA for four live routes, and from 171.01 to 31.74 for
eight. ROCm's full eight-route case fell from 78.56 to 56.80 microseconds; the
four-route case stayed near 53 microseconds. A twenty-generation regression
exercises both backend directions, aligned and unaligned widths.

The real model also contains a BF16 routed-expert layer. Its floating down
projection recomputed SwiGLU separately for every output column. Computing the
same FP32 activation once in the existing intermediate arena removes that
redundancy without another buffer or graph node. FP16, BF16 and FP32 all use
the same implementation. Absent routes still perform their explicit rounded
zero contribution but no longer reduce 256 zeros. All benchmark token streams
and MTP counters remain identical to the return-packing baseline.

An isolated BF16 M16 CUDA profile of that intermediate candidate exposes
393,216 down-projection blocks, 72 registers/thread, 45.18% achieved occupancy,
321.45 GB/s diagnostic bandwidth and zero spills. It spends substantial work
on block-local reductions. The next candidate maps the unchanged 256 logical
partitions to registers per native warp/wave, producing eight CUDA or four HIP
columns per block without shared reduction scratch or block barriers. This is
an arithmetic-preserving scheduling change, not a new reduction tree. Its
HIP wave64 code-object audit reports 25 VGPRs for both projections, zero LDS,
zero private scratch and zero spills. The CUDA BF16 M16 down projection reaches
91.33% achieved occupancy and 879.88 GB/s diagnostic bandwidth with 39 registers
and zero spills; gate/up reaches 76.89%, 887.41 GB/s, 48 registers and zero spills.
Its isolated captured BF16 pipeline improves from 4.41 to 2.75 ms (CUDA M16),
and 7.74 to 6.37 ms (ROCm M16). CUDA M1 improves from 290.82 to 185.34 us and
ROCm M1 from 606.72 to 575.04 us. A width-32 HIP experiment regressed M1 and
was replaced by the native wave64 mapping.

The extended all-floating byte oracle exposed unrelated quantized preparation
dependencies: floating table decode bound unused Q8 intermediates and therefore
rejected odd widths. CUDA now skips those quantized bindings; HIP pointer-array
publication owns its own preallocated arena binding instead of depending on
quantized scratch setup. HIP's shared prefill binding also used floor division
where the canonical workspace BOM and CUDA use rounded-up block counts. The
new regression exercises 3072/1024, 259/263 and 31/7 hidden/intermediate shapes,
FP16/BF16/FP32, routed/shared experts, direct/canonical outputs, all live M=0..16,
and poisoned retained replays against the unchanged serial arithmetic oracle.
Both preflight entries pass (12.34 s). The native-wave full-model median reaches
343.16/42.33 tok/s with every generated token and MTP counter still unchanged.
Those are intermediate measurements; the final cohort and refreshed gate
status are recorded above.

The return validators now share one cooperative implementation for single-row,
single-lane and batched prefill/verifier entrypoints. This removes the old
serial mapped descriptor/route walk from draft returns while preserving every
identity, order, capacity and route check. The owning packet preflight also
contains eight new captured adversarial cases (four entrypoints on CUDA and
ROCm). Each alternates malformed descriptors/routes with valid replays of the
same graph, requiring the exact terminal error and zero payload publication
for rejected returns. The complete mapped-packet suite passes in 9.08 seconds.

Isolated ROCm floating BF16 M16 counter passes authenticate the wave64 mapping:
gate/up has 91.22% VALU utilization and 131,072 wavefronts; down has 95.18% and
393,216 wavefronts. Both report 28 allocated VGPRs, zero scratch and zero LDS
(the code object requests 25 VGPRs). FetchSize is approximately 1,573,350 KiB
and 787,000 KiB respectively. The separate kernel trace measures about 4.05 ms
and 2.50 ms; counter-instrumented timings never enter the canonical benchmark.

An explicit vocabulary-sharded MTP head experiment exposed a storage defect:
local TP reserved a 1x1 gathered-logits placeholder but the restored-prefix
condition graph requires full-vocabulary rows. `MTPTerminalGatherGeometry`
now owns the logical shape for both schema resolution and the admission BOM.
Two new preflight entries exercise CPU/CUDA/ROCm, TP2/4/8, retained depth 1/3/15
and disabled requests, plus exact admission deltas through 31 target rows.
All five owning Unit/preflight groups pass. The real model passes that setup
boundary but then rejects the separate unimplemented participant-local GPU
verifier over sharded heads. The experiment has no timing result and does not
change GPU defaults. Further speed work retains mirrored ownership.

Local artifacts are under the ignored
`parity-results/cuda2-rocm4-122b-tuning/` directory. Retain the benchmark JSON,
raw sample logs, profiler reports and final gate receipts there; do not commit
those payloads or update official benchmark marks from this diagnostic run.
