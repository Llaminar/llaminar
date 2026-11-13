# QFN ngram CPU path investigation

Date: 2026-10-21. Source reviewed at `96edf3492d18f75c53c9f40cb281400a288f8362`.

Initial investigation: source and existing Release-object inspection only.
The recommendations below were candidates to measure. The measured follow-up
and candidate69 validation sections record subsequent work separately.

## Finding and relevance to the current tuning cell

CPU ngram support is only partly specialized. The CPU row decoder has canonical
runtime ISA selection and OpenMP worksharing for its optimized formats. The
host RAM gather used by the current four-MI50 run is serial and includes
synchronization and metadata-check overhead.

The current A1 configuration is `TensorParallel + GateUpOwnedDownColumns`, with
all transformer and expert computation on GPU, ngrams in RAM, and an AVX512
Release binary. Its last recorded checkpoint reached 784.382 prefill / 33.376
decode tokens/s on the dashboard's 512-input/256-output workload. Those numbers
include host lookup service costs but do not attribute time to them. See the
[tuning dashboard](QFN_TUNING_DASHBOARD.md) for the complete benchmark contract.

There are two distinct execution paths:

- **GPU execution with RAM ngrams:** GPU hashes produce row IDs; the host
  service gathers original native row bytes into the retained response channel;
  GPU kernels consume and decode those bytes. CPU hash/dequantization kernels
  are not on this path.
- **CPU execution:** CPU hashing, the same host provider, and CPU row decoding
  supply the CPU PLE computation.

[ImmutableLookupStage::execute](../../../../src/v2/execution/compute_stages/stages/qwen4exp/ImmutableLookupStage.cpp)
owns this distinction. Optimization of the CPU decoder cannot by itself improve
the current all-GPU compute cell.

## Verified implementation

| Component | ISA/vectorization | Parallelism and observations |
|---|---|---|
| Whole-RAM row gather | Per-row `std::memcpy`; no custom Llaminar SIMD dispatch. The copy implementation is delegated to libc. | Sequential loop under one table mutex; no OpenMP gather. |
| CPU IQ4_NL row decoder | `activeISALevel()` selects explicit AVX512/AVX2 code. Decodes complete 32-value blocks and amortizes scale conversion. | Canonical `OMP_WORKSHARE_REGION`, static worksharing over blocks. |
| CPU FP16/BF16 row decoder | Explicit AVX512/AVX2 widening with exact FP16 NaN-bit handling. FP32 uses byte copies. | Canonical nested-safe worksharing over contiguous 256-element blocks. |
| Other CPU native quantized row decoders | Generic per-element decoding, without equivalent explicit ISA-specialized implementations. | Canonical OpenMP worksharing, but repeated per-element row/block addressing remains. |
| CPU ngram hash | Scalar integer arithmetic; the inspected Release object contains scalar division for modulo and no YMM/ZMM vector arithmetic in the hash implementation. | Workshared in 256-result blocks. One decode token has 16 head hashes, hence one useful work block. Outside an existing team, the unconditional worksharing wrapper can still create a full team. |

Sources:

- [ImmutableRowTable::gather and Impl::copyRow](../../../../src/v2/loaders/ImmutableRowTable.cpp)
  validate IDs, acquire the table mutex, authenticate the source, copy each row,
  and publish counters. The RAM branch copies existing bytes without decoding
  or requantization.
- [CPUImmutableLookupKernels.cpp](../../../../src/v2/kernels/cpu/qwen4exp/CPUImmutableLookupKernels.cpp)
  contains the specialized floating/IQ4_NL decoders and generic quantized path.
- [CPUQwen4ExpKernels.cpp](../../../../src/v2/kernels/cpu/qwen4exp/CPUQwen4ExpKernels.cpp)
  contains `ngramHash` and its 256-element `pointwise` worksharing helper.
- [CPUFeatures.h](../../../../src/v2/utils/CPUFeatures.h) and
  [OpenMPUtils.h](../../../../src/v2/utils/OpenMPUtils.h) are the canonical ISA
  and nested-safe worksharing authorities used by those CPU kernels.

Read-only `objdump -drC -Mintel` inspection of the existing
`build_v2_qfn_release/CMakeFiles/llaminar2_core.dir/` objects confirmed YMM/ZMM
instructions in the specialized CPU row decoders, OpenMP calls, and the scalar
hash implementation. This establishes emitted instructions, not the selected
branch or its measured efficiency during a live workload. Forced AVX2 dispatch
inside an AVX512 build is also distinct from an AVX2-compiled binary.

Two host-service costs deserve direct attribution:

1. `ImmutableRowTable::gather` calls `requireUnchanged()` once per ordinary
   gather, issuing **one `fstat` and one `stat`**, including for a RAM-resident
   table. These are metadata checks, not payload hashes or checksums.
2. [ImmutableLookupService::run](../../../../src/v2/execution/qwen4exp/ImmutableLookupService.cpp)
   requests a **20 microsecond sleep** after each poll when the service remains
   open, including after successfully servicing a request. Scheduler wakeup
   latency can exceed the requested sleep. This is not evidence that every
   request incurs exactly 20 microseconds of added latency.

## Specific improvement candidates

| Priority | Improvement | Required boundaries and proof |
|---|---|---|
| P1 | **Reduce lookup-service wakeup latency.** Evaluate a bounded active-poll interval after progress, with an explicit idle/backoff policy, instead of sleeping unconditionally after a serviced request. | Keep the existing persistent worker and request/ACK/close protocol. Measure CPU utilization, tail latency and interference with CPU expert workloads; do not replace sleeping with unlimited spinning. |
| P1 | **Permit concurrent reads of the immutable RAM table.** Move RAM byte copying out of the table-wide exclusive critical section while retaining the original owner through each read. Keep counter publication short and per-call observations exact. | Preserve retirement safety and the sole physical allocation owner. Mmap window replacement and adaptive-cache policy require their own synchronization; do not remove their locks by analogy. Stress concurrent consumers, stats reads and retirement. |
| P1 | **Separate RAM snapshot validation from live file-backed access.** Consider authenticating an owned RAM snapshot at admission/reuse boundaries rather than issuing two file-metadata syscalls for every lookup. | This requires an explicit source-lifetime contract: current code detects source-file changes on a later gather even in RAM mode. Define and test the intended failure timing before changing it. Preserve mmap/truncation protection and fail-fast ownership checks; do not silently delete validation or introduce payload hashing. |
| P2 | **Parallelize sufficiently large RAM gathers.** Partition disjoint destination rows using canonical worksharing and a measured threshold based on useful bytes/rows. Evaluate coalescing genuinely adjacent source/destination spans. | Keep small decode gathers economical. Use the admitted host worker budget across service lanes; do not create a full OpenMP team independently for every GPU service or change global thread settings. Preserve exact row order, duplicate outputs, native bytes and live extents. |
| P2 | **Tune CPU hashing for CPU execution.** Compute the shared two-token and three-token hash numerators once per row rather than separately for every head. Evaluate exact precomputed modulo parameters and ISA-specific batched arithmetic where profitable. | Preserve unsigned wraparound, exact remainder, EOS handling and prior-token history. Use the canonical runtime ISA authority. Add a measured small-work policy so one decode token does not pay for an otherwise idle team. This is not a current A1 GPU-path optimization. |
| P3 | **Extend CPU native-format decode specialization.** Decode complete quantization blocks, reusing scale/codebook work and eliminating repeated per-element division/address construction; add AVX2/AVX512 variants where they win. | Preserve original quantized weights and exact decoded values, including special floating values. Cover every supported source format and tail. Prioritize formats actually present in CPU-serving workloads; this is not a current A1 GPU-path optimization. |

Retain libc `memcpy` as the measured copy baseline. Absence of handwritten AVX
intrinsics is not by itself a defect. For the current GPU cell, establish
polling, mutex and metadata costs before investing in a custom copy kernel.

Also authenticate the actual lookup-service worker TIDs, CPU masks and NUMA
placement before attributing a locality problem. The earlier GPU-control
affinity improvement does not prove placement of every independent host service.
Thread names alone are insufficient evidence.

## Validation and measurement plan

Keep this work in the existing Llaminar test framework. Reuse the immutable
table/provider, protocol and stage fixtures; extend their coverage instead of
creating a separate imitation of the production path.

Functional regressions should cover:

- RAM, mmap and adaptive ownership separately; unchanged native row bytes and
  output order; duplicate/random/adjacent IDs; empty and partial requests;
  invalid IDs and aliased destinations; large-to-small replay and poisoned tails.
- Concurrent readers and service lanes, retirement during an admitted read,
  exact per-call/cumulative counters, and metadata mutation/replacement/truncation
  at every boundary permitted by the chosen source-lifetime contract.
- Protocol sequence, generation, ACK and close ordering; cancellation and I/O
  failure while the worker is polling or sleeping; retained GPU graph replay
  on CUDA and ROCm without callbacks, recapture or new mutable host mirrors.
- CPU hashes and decoders against independent exact references, serial/grouped
  equivalence, EOS/history boundaries, every supported format, vector/block tails,
  both compiled ISAs and runtime dispatch controls. Exercise positive worker
  counts through the physical-core budget and nested worksharing entry.

Every concrete defect fixed needs a focused regression registered explicitly
in `ProductionTestPreflight`. Existing `Qwen4ExpCPUContracts`, immutable-table,
lookup-protocol and stage tests provide correctness scaffolding; their passing
status alone does not prove thread scaling or performance.

Performance experiments should separate:

1. **RAM gather:** use the actual model's row format/width, live decode and
   prefill sizes, representative larger buckets, random and duplicate-heavy
   IDs, and 1/2/4/8 concurrent consumers. Sweep the admitted physical-core
   budget on one NUMA node; label cross-NUMA experiments separately. The
   dashboard's repeated-token prompt must not be the only access distribution.
2. **Complete host lookup service:** measure publication-to-response latency,
   lock wait, gather/copy time, metadata calls, wakeups, CPU time and useful
   bytes. Report median and tail latency. Include actual bound service-worker
   placement and CPU-expert contention where that topology is supported.
3. **CPU arithmetic:** isolate hash and native-row decoding by format, shape,
   compiled/runtime ISA and thread count. Keep these results separate from GPU
   lookup-service results.
4. **End-to-end inference:** compare control/candidate unprofiled Release runs
   with the dashboard's frozen topology/model/prompt/precision/settings, retain
   all paired samples, and rerun the corresponding accuracy gate. Require a
   repeatable improvement beyond noise without an unacceptable prefill,
   decode-tail or host-CPU regression.

Collect `perf`/counter evidence separately from canonical timing. Observability
must remain diagnostic or CPU-owned terminal data: no live GPU state readbacks,
hot-path allocation, second admission ledger or full-table checksum. Coordinate
all native experiments with the main tuning run rather than competing with it.

The next useful experiment is attribution of the **current RAM gather/service**
cost, followed by one isolated change among polling, read serialization and
source-validation placement. No performance benefit or implementation completion
is claimed by this note.

## Measured follow-up: host-worker placement, 2026-10-21

The source investigation above is retained as the initial proposal. A subsequent
diagnostic ran the archived L66 Release runtime on the unchanged A1 workload:
four MI50s, RAM ngrams, 512 input tokens, 256 output tokens, one warmup and three
measured requests. The existing benchmark also performs one explicit prefill
preparation before those requests. Function entry/return probes observed the
actual host service and table; production code and policies were unchanged.

The [completed diagnostic](../../../../parity-results/qwen-flash-next/ngram68-host-service-diagnostic-global/result.json)
passed source/runtime identity, driver, output and ordinary retirement checks.
All three measured output sequences match the L66 control. Its instrumented
throughput is excluded from the dashboard. The local probe source, attachment
proof and component analysis are retained under
`/tmp/qfn-ngram68-host-attribution-global/`.

The actual four service-worker TIDs were joined to the runner's passive
`/proc` observations. **Every worker retained affinity `{28,84}`**, the two SMT
threads of one physical core, across 12–13 observations per worker. Both
prefill and decode used those same retained workers. L66 broadened GPU-control
worker eligibility; it did not cover these independently constructed I/O workers.

The trace accounts for **4,100 gathers in 20 service intervals**: per GPU,
one prefill preparation, then four requests containing one prefill and 255
decode forwards each. Prefill requests contain 8,192 native rows; decode
requests contain 16. Original IQ4_NL rows are 90 bytes, so the complete trace
observed 20,620,800 native response bytes. These are live response extents,
not staging allocation capacity.

The following component observations exclude preparation and warmup. Times are
microseconds and **include probe overhead**. Validation plus lock acquisition
is a combined interval; copy plus publication includes the copy loop, counters
and mutex release. Summing worker durations does not produce removable model
wall time.

| Host gather component | Decode median / p95, 3,060 gathers | Prefill median / p95, 12 gathers |
|---|---:|---:|
| Complete gather | 45.006 / 146.648 | 299.602 / 1,006.941 |
| Validation plus lock acquisition | 13.201 / 120.965 | 54.224 / 758.744 |
| File metadata authentication | 17.255 / 25.106 | 16.871 / 25.232 |
| Native copying plus counter publication | 4.707 / 17.958 | 222.558 / 521.959 |

Instrumented `clock_nanosleep` intervals averaged about 86.7 microseconds per
worker. This includes kernel execution, instrumentation and scheduler delay;
it is neither the requested 20 microseconds nor a per-token latency penalty.
CPU scheduled-residency measurements likewise include tracing overhead.

The first attempted probe attachment followed only its launcher and recorded no
service events; that run also failed its driver check with a userptr workqueue
warning. It is retained as rejected evidence. The corrected attachment was
first proven with a separate CPU-only parent/child test and explicit host/container
TID correspondence. Zero events were never interpreted as zero service cost.

**Candidate selected for measurement:** construct the lookup service's own persistent
worker within the existing admitted OpenMP partition, restoring its caller's
exact mask afterward. Keep generic CPU expert workers, polling, table locking,
source validation and native bytes unchanged. Prove actual service-worker
placement and lifecycle on CUDA and ROCm, then compare uninstrumented A/B/B/A
throughput and the frozen numerical gate. No speed benefit is established yet.

## Candidate69 implementation and validation

`ImmutableLookupService::create` now scopes construction of its own retained
worker with the existing `ScopedOpenMPPartitionAffinity`. Admission and mapped
storage creation precede that scope; the original caller mask is restored on
exit. Generic retained workers, CPU expert work, polling, table locking, file
metadata validation and native response bytes are unchanged.

The [old-core negative control](../../../../parity-results/qwen-flash-next/ngram69-old-core-negative/negative-control-proof.json)
failed the actual-worker affinity assertion on both CUDA and ROCm with bound
OpenMP, while both unbound controls passed. After the fix, all
[eight focused registrations](../../../../parity-results/qwen-flash-next/ngram69-focused/result.json)
passed, including the original captured primitives and adaptive CPU storage
protocols. The four explicit bound/unbound CUDA/ROCm affinity regressions are
registered in `ProductionTestPreflight`. They observe the actual retained
service thread during RAM and mmap requests, preserve native byte/ACK/close
checks, and verify creator-mask restoration and fresh worker observations
through live counts 1025, 15, 1, 0, 512, 1024, 2 and 1025.

The observer delegates the real libc `fstat`, preserving its result and errno.
A test-only mutex protects exact-inode filtering and fixed observation storage
against unrelated in-flight libc calls; it never rebinds the service child.
No instrumentation is installed in the production runtime.

The complete [Unit suite](../../../../parity-results/qwen-flash-next/ngram69-unit/result.json)
passed **754/754**, and the [Release build](../../../../parity-results/qwen-flash-next/ngram69-release-build/result.json)
completed. The [frozen A1 numerical panel](../../../../parity-results/qwen-flash-next/ngram69-release-accuracy-diagnostic/report.json)
passed every required group; all **185 retained logit rows in 26 files** are
[byte-identical to L66](../../../../parity-results/qwen-flash-next/ngram69-release-accuracy-diagnostic/affinity66-logit-byte-comparison.json),
including reset and diagnostic rows. Accuracy uses its unchanged unbound
56-physical-core contract. Bound ordinary-bootstrap throughput and generated
tokens are measured separately. Complete current-source production preflight
remains pending.

The [unprofiled A/B/B/A comparison](../../../../parity-results/qwen-flash-next/ngram69-abba-comparison.json)
retained all twelve samples and identical generated outputs. Across six samples
per runtime, control medians were **784.599102 prefill / 33.326504 decode tok/s**;
candidate medians were **785.181665 / 33.237744** (**+0.07425% / −0.26634%**).
The ranges overlap: control **784.229200–785.816661 / 33.285583–33.402426**,
candidate **784.420225–786.116590 / 33.226563–33.389609**. Every arm passed
driver, resource, source/runtime and normal-retirement checks. Retain the fix
for correct independent-worker placement; no throughput improvement is claimed.
Other CPU-path proposals remain unmeasured. Resume the prepared GPU compact
import candidate to pursue the remaining decode gap.
