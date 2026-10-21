# Additive CPU GEMM participation experiment

Date: 2026-10-20. Status: proposal; implementation and measurements pending.
This document defines an experiment, not a new supported mode or a performance
claim. It does not change the active QFN qualification sequence.

## Question and hypothesis

Can spare CPU compute improve inference throughput when the model's experts
already fit on GPUs, including configurations whose ngrams also fit in VRAM?

The hypothesis is that a small amount of concurrent CPU expert work can shorten
the GPU critical path. Full GPU residency should remain intact. CPU execution
uses separately admitted immutable RAM weight replicas and exchanges only live
activations, route metadata and results.

For one assisted expert phase, an approximate planning condition is:

```text
T_assisted = max(T_GPU_remaining, T_to_CPU + T_CPU + T_from_CPU) + T_join
T_assisted < T_GPU_only
```

The times must include observed contention. CPU GEMM throughput in isolation
cannot establish this inequality: CPU workers, memory traffic, PCIe transfers
and communication progress can all affect GPU completion. A universally zero
slowdown guarantee is unrealistic. The experiment must establish a useful
speedup and bounded latency effects on explicitly qualified workloads.

## Initial scope

Start with routed MoE work: a batch of `(token row, expert ID)` pairs whose
complete gate/up/activation/down computation runs on one CPU participant.
The GPU domain omits exactly those routes. The router's expert choices and
weights remain unchanged, and the final combination retains canonical route
order. Each route has exactly one execution owner.

Test prefill and decode separately. Larger routed batches may amortize transfer
and dispatch costs; small-batch decode may offer too little work to repay them.
These are hypotheses to measure. Grouped MTP verification remains decode work.

Dense GEMM output-column sharing is a later experiment if the MoE result is
promising. It needs its own physical layout, communication and arithmetic proof.
Splitting the reduction dimension, racing duplicate CPU/GPU calculations, and
retrying late CPU work on GPUs are outside this initial experiment.

Ngram lookup, attention, routing and shared-expert placement remain identical
within each comparison. Ngram work is not credited as a GEMM acceleration.

## Existing foundations and implementation boundary

The current source separates weight representation, phase policy and assignment
among eligible complete expert residents in
[RoutedExpertPolicy.h](../../../../src/v2/execution/config/RoutedExpertPolicy.h).
It also defines compact publication of CPU expert rows and fixed route-order
combination. The measured request evaluator in
[PlanningRequestCostModel.h](../../../../src/v2/planning/PlanningRequestCostModel.h)
already distinguishes overlapping compute endpoints from communication costs.
These are reusable contracts; they do not establish this experiment's runtime
support or economics.

There is an important comparison trap. Adding a CPU domain can change automatic
GPU expert ownership from projection sharding to whole-expert apportionment.
See [automatic compute ownership](MOE_AUTOMATIC_COMPUTE_DEFAULT.md) and the
[projection ownership expansion plan](MOE_PROJECTION_OWNERSHIP_EXPANSION_PLAN.md).
A measured difference after that change cannot be attributed solely to spare
CPU compute. Freeze the concrete GPU compute policy, membership and shard
geometry in each A/B pair. If preserving them requires an unimplemented
composition, record that prerequisite and implement it explicitly before timing.

In particular, a GPU TP domain may collectively hold an expert without any GPU
holding a complete local copy. CPU execution needs its own complete prepared
expert. All GPU participants must agree that the CPU owns the selected route,
omit its GPU projection work, and accept its complete result at the existing
logical combination boundary. Merely inserting a CPU into a native GPU TP
communicator is not a valid implementation.

## Proposed implementation sequence

1. **Admit immutable CPU replicas and workspaces.** Select a declared bounded
   expert set, prepare it from the same original GGUF, and preserve existing
   quantized weight formats and Q8 GEMM activation policy. Do not requantize
   quantized source weights to create a CPU copy. Charge replicas, preparation,
   pinned buffers, queues and retained graphs through `PhysicalMemoryAuthority`.
   Keep GPU residency and prefix/ngram cache budgets fixed. No hot-path weight
   download, page fault into an unprepared expert, or allocation is permitted.
2. **Prove fixed work shares.** Use an explicit, bounded assignment among
   eligible replicas. Prebuild the production graphs and persistent CPU worker
   pool. Transfer compact live inputs/results through `TransferEngine` and the
   declared heterogeneous graph boundary. Keep GPU routing and execution state
   device-owned; CPU workers own only their assigned CPU work and completion.
   Preserve exact streams, event ordering, generation identity and reset reuse.
3. **Measure completion and transfer costs.** Record actual worker count, NUMA
   placement, CPU service time, GPU exposed time, transfer bytes/time and join
   wait. Preserve cores needed for orchestration and communication progress;
   sweep physical-core budgets rather than automatically occupying every core.
4. **Add adaptive assignment only after fixed-share evidence.** Extend the
   existing assignment authority with a bounded completion-cost policy. Use
   conservative estimates and hysteresis, with a safety margin for CPU work to
   finish before the GPU branch. The first version can update its cost policy
   between requests from terminal observations. It must not introduce per-token
   GPU count readbacks, a host shadow of GPU state, graph recapture, or a second
   placement/accounting controller.

An enabled scheduling policy may assign zero CPU routes when no eligible work
is economical. This is an explicit policy outcome. Missing required state,
failed CPU work or broken transport must fail; they must not trigger an
unadvertised execution mode or duplicate GPU retry.

CPU-only-prefill assistance is a legitimate outcome. Its decode phase must
retain its explicitly declared GPU execution policy and be measured for any
remaining setup, graph or resource overhead.

## Controlled comparison

Derive all cells from `ModelParityDefinition` and its existing runtime exporter.
Pair the accuracy workload and benchmark with the same concrete topology,
prepared formats, arithmetic, ngram placement and CPU participation policy.
Do not create a second Python model/topology registry or imply E2E eligibility
merely by adding a diagnostic cell.

| Variant | Purpose |
|---|---|
| A: GPU-only | Existing qualified GPU execution and residency baseline. |
| B0: CPU assistance admitted, zero assigned routes | Isolate admission, graph, polling and idle-participant overhead. Empty participants exchange only required metadata. |
| B1: Fixed CPU share | Establish whether real additive work repays communication and joining. |
| B2: Adaptive CPU share | Test whether measured cost selection retains B1's gains as routing and system load change. |

Start with proposed fixed shares of 1%, 2%, 5%, 10% and 20% of eligible routed
work, alongside zero. These are search points, not an installed dispatch table.
Report actual route counts, GEMM shapes and estimated work; a percentage of
expert IDs does not imply the same percentage of computation. Include enough
independent routes to represent a nonzero share without fabricating work.

| Axis | Initial coverage and extension |
|---|---|
| Model | An existing fully GPU-admitted MoE cell first; QFN after its numerical/topology prerequisites pass. A second MoE model checks generality. |
| GPU topology | CUDA TP2 and ROCm TP2/TP4 where the exact GGUF fits. Extend to qualified PP-of-TP and mixed-vendor cells separately. |
| CPU | One worker through physical cores per socket, both NUMA sockets, explicit local/remote memory placement; AVX2 and AVX-512 where supported. |
| Prefill | Representative live rows 64, 512, 1024 and 2048, plus partial buckets; retain the actual supported bucket family. |
| Decode | Serial decode at short, 16K and 64K retained context where admitted; grouped verification only after serial correctness. |
| Routing | Balanced, skewed/hot experts, repeated routes, CPU-ineligible experts, empty CPU assignment and changing prompt distributions. |
| Ngrams | A fully VRAM-resident cell when admitted; separate matched RAM and SSD/adaptive cells to expose host-memory and I/O contention. |
| Cache and load | Cold and reused prefixes, bounded cache growth, idle host and reproducible bounded CPU/memory pressure. |

Availability is not admission. A smaller GGUF or different topology creates a
separate experiment identity. Do not reduce GPU expert residency or change
ngram placement in the assisted arm to make it fit. Do not describe a RAM-ngram
QFN result as proof of an entirely VRAM-resident QFN configuration.

## Correctness and functional gates

Use the [Llaminar testing workflow](../../../../.agents/llaminar-testing/SKILL.md)
and [canonical parity framework](../../../../tests/v2/integration/parity/README.md).

- Device-free tests cover eligibility, exact-once ownership, bounded assignment,
  zero work, stale estimates/epochs, partial batches, hysteresis and complete
  PMA capacity declarations. Sweep supported tensor formats and legal shape
  boundaries. Reject unsupported physical compositions.
- CUDA/ROCm integration tests use actual retained graphs and CPU execution.
  Cover delayed CPU completion, skew, empty participants, large-to-small replay,
  reset, prefix restore, shutdown and failures during publication. Prove actual
  communication extents and stable admitted allocations, including when CPU
  assistance contributes nothing.
- Preserve the same-GGUF CPU/FP32 PyTorch oracle. Compare the fixed histories
  with identical teacher tokens, live rows, bucket geometry and worker settings.
  Report final-logit mean KL by prefill and decode step, top-1 and literal top-5
  agreement, and every per-row outlier. Use all-layer/stage CSVs to locate new
  divergence. In addition, compare assisted and GPU-only runs directly to
  separate existing quantization error from the participation change.
- Keep the approved numerical limits unchanged. An uncalibrated baseline can
  produce diagnostic observations but cannot qualify a candidate. Preserve
  within-policy grouped-versus-serial byte equivalence before enabling MTP;
  do not confuse that obligation with cross-backend floating-point equality.
- Every discovered defect gets a focused regression explicitly registered in
  `ProductionTestPreflight`. Run focused checks during implementation, then
  complete Unit/preflight for each coherent source batch. Reuse the genuine
  unchanged prerequisite receipt across subsequent numerical cells.

## Performance method and observability

Measure unprofiled **Release** runs using identical original model sources,
token IDs, prompt lengths, context, physical buckets, KV precision, sampling,
MTP work, cache state, GPU clocks and CPU affinity. Keep the standard
512-token prefill/256-token generation workload as one anchor; retain separate
long-context cases. GPU-only Llaminar is the primary causal control. A matched
external-engine comparison may provide context after this question is settled.

Use interleaved A/B/B/A trials and identical warmup policy. Separate cold setup
cost from steady inference. Begin with focused repetitions; qualify winners
with enough independent request repetitions to report confidence intervals and
tail latency. Treat p99 estimates as exploratory until sample support is
adequate, and preserve slow runs. Freeze acceptance thresholds before collecting
qualification data, and use held-out prompts/load patterns for adaptive policy.

Proposed observability fields belong in existing PerfStats and terminal stats
publication; names below describe evidence, not already installed counters:

- Eligible/assigned/completed CPU route rows, actual CPU worker count and
  per-phase work share; each route's ownership epoch.
- CPU GEMM service time, GPU branch completion, CPU-induced join wait,
  assignment-budget misses, and measured overlap versus exposed time.
- Actual H2D/D2H payload and metadata bytes, transfer latency, and GPU collective
  traffic. CPU participation must not enlarge unrelated communication.
- Retained graph capture/replay and segmentation boundaries; allocations,
  prepared RAM replica bytes, VRAM residency, RSS and prefix/ngram tier usage.
- End-to-end prefill/decode tok/s, TTFT, inter-token p50/p95/p99 and request
  completion time, with the same phase definitions in both arms.

Aggregate ordinary telemetry at the existing publication boundaries. Detailed
per-route traces belong to separate diagnostic runs; they must not introduce
per-row host readbacks or contaminate canonical timing.

Profile separately from timing. Use CPU `perf` and NUMA/cache evidence, CUDA
Nsight or ROCm profiling for exact kernels and the full timeline. Distinguish
overlapping interval unions from sums. Any new GPU kernel needs register,
occupancy, spill and memory-traffic evidence; retain the existing spill guards.
Bracket native lifetimes with the standard driver-log checks. Keep large raw
results ignored and publish only concise findings in this project document.

## Decision criteria and deliverables

Proposed economic thresholds, to freeze before qualification:

1. At least **3% end-to-end improvement** in an explicitly targeted phase, with
   a paired 95% confidence interval excluding zero benefit.
2. At most **1% overhead** for B0 and other enabled phases, with the upper
   bound of the paired 95% confidence interval at or below that limit.
3. No more than **2% tail-latency regression** at the agreed quantiles under
   idle and bounded-contention workloads. Inadequate tail evidence remains an
   open gate. A different tolerated trade-off requires a separate decision.
4. All numerical, capture, ownership, exact live-traffic and bounded-memory
   gates pass. CPU assistance never earns a pass merely by occupying the CPU.

Retain a policy only for the backends, phases and geometry regions supported
by the evidence. If useful gains require a different GPU layout, measure that
layout separately against its own GPU-only control. If no share clears these
gates, record the result and keep GPU-only execution for that region.

Deliverables are the typed experimental policy and canonical paired cells;
focused Unit/backend/preflight regressions; admission and same-GGUF numerical
reports; Release A/B tables and timeline evidence; and a decision listing
winning regions, extra RAM cost, latency effects and remaining limitations.
Production default changes and CI certification follow only after that review.
