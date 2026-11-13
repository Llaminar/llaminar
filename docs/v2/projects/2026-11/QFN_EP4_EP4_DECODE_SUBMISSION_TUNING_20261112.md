# QFN eight-MI50 decode submission experiment

## Result and retained scope

Concurrent host submission of the first pipeline stage's captured graphs did
**not** improve unprofiled throughput. The candidate is set aside. Its correctness
checks passed, but profiler observations alone do not justify installing it.

The experiment independently reproduced an exception-safety defect in
`TPWorkerPool::collectAll()`: allocating the result snapshot could throw after
collector admission was incremented, leaving subsequent dispatch permanently
excluded. The retained repair releases collector admission through RAII while
leaving the unsuccessful snapshot's generation uncollected. Callers can collect
that same generation again. It does not change graph submission, model math,
worker placement, streams, or GPU memory ownership.

## Matched workload

- Release baseline: `533415222`, whose inference implementation is `60d9a2f7d`.
- Model: `Qwen3.8-Flash-Next-UD-Q4_K_XL`, the original four GGUF shards.
- Topology: `PP(EP(4×MI50), EP(4×MI50))`, with disjoint fixed-head ngram
  ownership in VRAM across all eight GPUs. Discovery MPI rank 1 owns both
  stages on NUMA 1. Stage 0 retains GPUs 7/6/8/9 and layers 0–23; the tail
  retains GPUs 1/3/4/2 and layers 24–47.
- Static expert placement with ordinal ownership; original precision policy;
  MTP and prefix caching disabled for the throughput comparison.
- Canonical raw 512-token prompt, context capacity 4096, M512 prefill, 256 generated
  tokens, greedy sampling, seed 42; one warmup and three measured requests. Decode throughput
  counts the 255 tokens generated after prefill.
- No profiler or GPU stage clocks in either throughput measurement. No driver,
  kernel, clock, visibility-mask, or device-identity changes.

| Measurement | Baseline | Concurrent submission candidate | Median change |
|---|---:|---:|---:|
| Prefill, median tokens/s | 732.9226 | 733.1893 | +0.036% |
| Decode after prefill, median tokens/s | 37.25986 | 37.12449 | −0.363% |
| Decode request times, ms | 6817.739 / 6843.825 / 6877.045 | 6831.260 / 6868.782 / 6886.493 | Within observed variation |

All three candidate requests produced exactly the baseline's 256 token IDs.
The separate 16-history ForwardOnly diagnostic retained byte-identical final
logits for all 112 rows and the reset payload, with zero KL change. That
diagnostic alone does not exercise ordinary generation submission; the token
comparison supplies that separate evidence. The existing uncalibrated diagnostic
verdict remains unchanged. This experiment is not full preflight or release
certification.

Both native ranks, their guardian, and GPU descriptors retired. All eight cards
returned to 10,866,688 bytes of used VRAM. Candidate peak VRAM was 21.086–21.579
GB. The two processes' high-water RSS values were 748,974,080 and 21,933,498,368
bytes, reported separately; neither used swap. Canonical resource observation
and the driver window were clean. Additional passive-monitor permission errors
belonged only to diagnostic `sudo`/`dmesg` descendants.

## What the trace establishes

The original first-stage host thread submits four captured graphs sequentially;
the tail uses concurrent persistent workers. GPU execution nevertheless overlaps.
Full ROCprof tracing and nominal API-only tracing both increased decode from
approximately 6.84 to 10.5 seconds. The API-only invocation still enabled graph
and event tracking and intercepted queues. Its timing is not an independent
low-overhead control.

`rocpd_graph_launch` intervals describe host submission, not GPU completion.
Attribution instead selected the last 255 occurrences of each stable captured
main-graph node, verifying all 510 occurrences and every measured frame's node
count. This preserves work that continues after `hipGraphLaunch` returns.

Long RCCL receive-header intervals mostly expose peer arrival and progress.
They are not measurements of PCIe payload latency. Each addressed transfer
records an eight-byte extent header and a separate live payload on both
endpoints. Per decoded token, the topology has:

- Four next-token transfers, four ngram-ID requests, four native-row replies,
  and four residual-activation deliveries: 16 directed edges, 32 endpoint
  invocations, and 64 LocalP2p kernel launches across the eight GPUs.
- Four-byte token payloads, 16-byte ID requests, 180-byte IQ4_NL row replies,
  and 40,960-byte residual payloads per corresponding consumer.
- 24 routed-expert allreduces per GPU, each publishing the original ten-route
  FP32 bank, plus two PLE allgathers per first-stage reader.

Domain leaders execute 14 LocalP2p kernels per token; the other GPUs execute
six. Summing their overlapping wait intervals would exaggerate token latency.
The tail already services its ngram rows before waiting for the upstream
residual. Its subsequent model work cannot run without that residual.

Next tuning should target measured device work and collective latency. Possible
overlap or fusion changes need explicit consumer, scratch, and arithmetic-order
proofs. This investigation does not authorize removing protocol headers, release
events, receipts, or progress edges.

## Next focused kernel experiments

The source audit identifies two separate HC candidates, neither implemented or
performance-qualified by this slice:

1. Fuse residual write with the following branch normalization, initially at
   the intra-layer attention-to-FFN boundary. The existing
   [write and norm kernels](../../../../src/v2/kernels/qwen4exp/GPUQwen4ExpKernels.cuh)
   are separate. Preserve the updated trunk, normalized output, explicit write
   rounding, stride-256 FMA chain, reduction tree, gain order, and alias rules.
   Do not cross a PP communication boundary. Measure the change in parallelism:
   write currently has several CTAs per branch, while norm has one.
2. Fuse branch normalization with HC-down SingleQ8 activation publication.
   Keep FP32 normalized values because the other HC consumers still need them.
   Preserve K32 groups, Q8 scales and rounding, live rows, and the exact producer
   stream. Consume the publication through existing prepared-GEMM ownership;
   do not introduce a second quantization cache.

Existing HC SiLU/two-term publication, router-Q8 reuse, grouped shared gate/up,
ordered SwiGLU-Q8 publication, and eligible shared-gate/final-combine fusion
already exist. Reimplementing them or replicating HC activation publication in
every projection tile would add work.

Extend the actual-capture primitive fixture and builder contracts. Compare
original intermediate and final bytes; cover CPU/CUDA/ROCm, original aliasing,
partial/live rows, grouped decode, dirty large-to-small replay, and reset.
Retain all shipping-architecture spill checks and profile each candidate before
the unchanged numerical panel and unprofiled Release comparison. Filtered
built-in ROCm stage timing currently measures graph/parent scopes rather than
these leaf operations; enabling it alone does not supply leaf attribution.

## Evidence

Local artifacts are intentionally outside the source repository:

- Baseline: `/tmp/qfn-release-pp-ep4ep4-baseline-r2-20261112/` and its adjacent
  `-prep-20261112` directory.
- Candidate ordinary generation: `/tmp/qfn-release-host-fanout-candidate-benchmark-20261112/`;
  its adjacent prep directory contains `CANDIDATE-RESULT.md`, resource summary,
  and independent retirement audit.
- Candidate numerical comparison:
  `/tmp/qfn-release-host-fanout-candidate-smoke-prep-20261112/post-retirement-and-byte-comparison.json`.
- Captured GPU attribution:
  `/tmp/qfn-release-pp-ep4ep4-rocprof-full-prep-20261112/GPU-ATTRIBUTION.md`,
  `gpu-timeline-attribution.json`, and `gpu-execution-and-unions.svg`.
- Communication mapping: `/tmp/qfn-native-collectives-source-map-20261112.md`
  and `/tmp/qfn-native-collective-purpose-map.json`.
- Set-aside implementation and original-code controls:
  `/tmp/qfn-hosted-submission-candidate/`.

The broader candidate passed canonical focused CTest 5/5 and its new lifecycle
test for 20 repetitions. The retained worker-pool repair separately passed its
canonical Integration and Release builds (`q133`, `q135`), five focused CTest
registrations (`q134`), and 20 repetitions of
`V2_Integration_TPWorkerPoolCollectionLifecycle` (`q136`).
The direct-header fixture contains four cases; all four fail against the original
collector body. Receipts are in `/tmp/qfn-full-preflight-after-cpu-next/` and
`/tmp/qfn-tp-collection-lifecycle/`. No full preflight repeat is claimed.
