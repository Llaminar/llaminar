# Qwen 122B CUDA2 + CPU2 tuning

## Objective and fixed comparison contract

Improve both prefill and decode by at least 50% over release `2026-09-27.1`
for its canonical Qwen3.5 122B CUDA2 + CPU2 ExpertOverlay benchmark. Work starts
on `feature/qwen122b-cuda2-cpu2-tuning`, based on develop `4673ed148`.

The authority is the release's exported typed cell, not a new hand-written
topology: `Qwen35_122B_CUDA2_CPU2_2xMPI_NodeExpertOverlay_Dynamic_Ordinal_ActFP32_KVFP16_MTPDynamicDepth`.
Its benchmark projection requests auto selection with two CUDA GPUs and two CPU
NUMA endpoints, two MPI ranks, an 8192-token context, ordinal initial ownership,
dynamic residency maintenance and dynamic-depth MTP. Weights remain the four
`Qwen3.5-122B-A10B-UD-Q8_K_XL` shards; activations stay FP32 and KV stays FP16.

The primary tuning target is the AVX512 release series on this AVX512 host.
The AVX2 release values below are recorded for comparison, not a claim that its
end-to-end performance has already been remeasured.

The canonical workload is `benchmarks/production/workload.json`: 512 prompt
tokens, 256 generated tokens, seed 42, greedy sampling, one warmup and three
measured requests. Compare phase medians from the JSON, with profiling disabled.

| ISA | Release prefill tok/s | Release decode tok/s | Required prefill tok/s | Required decode tok/s |
|---|---:|---:|---:|---:|
| AVX512 | 124.92698 | 14.04954 | 187.39046 | 21.07431 |
| AVX2 | 79.71677 | 13.87499 | 119.57515 | 20.81248 |

## Baseline and first attribution

The exact released AVX512 image, local immutable ID
`sha256:872c2dd9c7bd2d3e1023bf297152d1a527ce9d4873842621f4f744ab3c2a7c78`,
reproduces **134.81831 tok/s prefill and 14.04510 tok/s decode**. Prefill samples
are 123.43735, 134.81831 and 138.11002; decode samples are 13.92994, 14.04510
and 14.28200. This baseline uses the unmodified release. The existing
Ornith CPU service was left running; its roughly one-core background activity
must be accounted for in controlled follow-up measurements.

MTP is live, with 133 accepted drafts and 123 verifier invocations per measured
request. All model shards were reused from the persistent tmpfs cache without
copying. There were no active CI hardware jobs.

A separate instrumented local Release run has phase medians 123.83965/12.91078
tok/s. These are attribution measurements, not candidate performance scores.
The remote CPU follower's four-request cohort (including warmup) records about
25.82 s of grouped-verifier expert service versus 48.36 s waiting for successive
dispatches. Those waits include predecessor GPU/peer work: they are not proven
transfer overhead. Its prefill totals are 7.15 s expert service and 11.91 s
dispatch wait across ten chunks. Rank-zero and follower counter reset scopes
differ, so their totals must not be compared without their invocation counts.

A separate 15-second CPU `perf` observation of that diagnostic run puts roughly
73% of sampled user cycles in libgomp wait loops and about 20% in expert
projection kernels. Idle/barrier worker cycles do not establish the critical
path. The next measurement is a CUDA node-level timeline to distinguish useful
GPU computation from collective/ticket waiting before choosing an optimization.
The profiler first passed the captured MoE routing smoke test and reported
executed CUDA kernel records.

## First implementation: fresh mapped-return reads

The CUDA timeline identified canonical-return materialization, not GEMM, as a
large critical-path cost. Its summed co-resident maintenance and collective wait
durations are **not** additive wall time. An isolated ready-payload benchmark
confirmed the transfer issue without any CPU producer wait: a 128-route captured
bank carrying 32 live routes took 614.4 us on CUDA. Eight live routes in an
eight-route bank took 68.6 us. Nsight Compute found 65.5% barrier stalls and
32.6% long-scoreboard stalls for the latter, with no memory spills.

The shared CUDA/HIP implementation now coalesces the acquired ticket's metadata
reads across a block instead of serializing them on lane zero. Return kernels
read a route index once per row instead of once per FP32 element. CUDA uses
aligned 16-byte cache-volatile payload loads, with exact scalar tails. HIP keeps
its coalesced scalar wavefront payload reads: applying per-lane vectors there
reduced large-transfer throughput, so that candidate was rejected. Neither
backend changes the release/acquire/acknowledgement protocol, FP32 fold order,
graph node count, physical buffer allocation, or weight format.

The refined microbenchmark reports CUDA sparse-verifier time **154.6 us**,
512-route time **784.4 us** (previously 1640.4 us), and large-prefill throughput
**11.47 GB/s** (previously 7.80). HIP large-prefill throughput is preserved at
**13.37 GB/s** (previously 13.41). HIP materializers use 12/16 VGPRs and zero
private memory or register spills; CUDA ticket materialization uses 24 registers
and zero measured spill requests. Canonical timing is unprofiled and separate
from these profiler records.

The first whole-model candidate, with the user-authorized pause of the Ornith
CPU server, reached median **188.05560 tok/s prefill / 22.67624 tok/s decode**:
**50.53% / 61.40% above the published release**. Prefill samples were 179.01,
188.06, 196.17; decode 22.52, 22.68, 23.23. Each request retained the same 133
accepted MTP drafts, 123 verifier runs and 396 verifier tokens. All 256 token
IDs match the released-image replay exactly in each of the three requests.
Prefill graph capacities, node counts, capture counts and replay counts are also
unchanged. Counters prove
both tier promotion/demotion and within-priority movement remained active.
Prefill has little margin over the target, so repeatability is still required.

The full captured CUDA/ROCm mapped-packet integration suite passed. Its existing
preflight ticket tests now additionally retain one graph while alternating
empty, sparse and full payloads, changing route permutations and arbitrary
payload bits, and preserving untouched destination slots across twenty replays.
The final refined implementation passed all **673 Unit tests** in 76.62 seconds
and all **424 ProductionTestPreflight tests** in 1078.61 seconds. This includes
the complete mapped-packet suite, CPU-expert ticket and cross-tier arithmetic
sweeps across formats and both CPU ISAs, capture lifetime, prefix restore and
MTP checks. The matched, unprofiled AVX512 repeat is:

| Run | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Published release | 124.92698 | 14.04954 |
| Released image, isolated rerun | 138.52205 | 14.98963 |
| Refined candidate, isolated rerun | **189.32796** | **23.40156** |

The refined candidate is **51.55% / 66.56%** above the published benchmark and
**36.68% / 56.12%** above today's isolated released-image rerun. Do not attribute
the difference between the published and rerun baseline to this code change.
Candidate prefill samples are 189.94, 179.01, 189.33; decode samples 23.18, 23.40,
23.61. Token streams and MTP evidence again match exactly in all three requests.
Each request completed 21–23 movement commands spanning both movement axes.
Both complete candidate runs exceed the two published AVX512 targets; prefill
still has relatively little margin. A freshly exported current typed manifest
agrees exactly with the released cell's E2E and benchmark profiles.

The public HTTP suite passed **45/45 checks** in 483.91 seconds, including all
**eight full long-context checks** at an 8192-token context: beginning/middle/end
needle recall, multi-needle strict JSON, structured generation, cache reset,
valid near-boundary context and oversized-request rejection. Structured
generation produced 2048 tokens with no sequence resets or duplicate lines;
the valid boundary request used 7595 tokens. The same run passed thinking and
non-thinking chat, prefix reuse, streaming and all four tool-call round trips.
The canonical runner accepted the required runtime evidence. Shutdown was
clean, GPU memory returned from the workload to its original 42 MiB baseline,
and there were no new AMDGPU or NVIDIA driver warnings.

After measurements and E2E completed, the original `llaminar-ornith-cpu`
container was restarted without changing its image or configuration. Its
health endpoint and `Ornith-1.5-35B-Q8_0` model listing were verified from the
still-running OpenWebUI container. This remains local branch validation, not
an image certification or a published high-water update.

## Evidence and acceptance

Local, untracked evidence lives under
`parity-results/cuda2-cpu2-122b-tuning/`: release attachments,
`baseline-image.json`, `baseline-image-isolated.json`, `final-returns.json`,
`http-e2e.json`, the full HTTP artifacts in `e2e-1790496862507407758/1/`,
`attribution.json`, per-rank PerfStats, CPU `perf` data, and CUDA/ROCm profiling
artifacts. Published high-water marks remain untouched.

For each implementation slice: prove the targeted invariant in a focused
regression, register production-defect regressions in `ProductionTestPreflight`,
and remeasure the unchanged public benchmark. Amortize the full Unit/preflight
gate across a built slice. Kernel changes must retain all-format, grouped/serial
byte equivalence and AVX2/AVX512 dispatch. Final acceptance needs unprofiled
measurements meeting both targets plus the affected public HTTP long-context,
prefix, MTP and movement checks.
