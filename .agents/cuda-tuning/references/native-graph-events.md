# Native CUDA graph diagnostics without CUPTI

Use this procedure when auditing a retained graph's structure or selected
intervals without profiler attachment. It does not replace hardware counters,
compiler spill evidence, or unprofiled Release benchmarks. Keep current driver
incidents and timing results in the dated investigation, not in this reference.

## Evidence and safety boundaries

- First resolve the production plan without profiling. Reuse its model,
  topology, precision, prompt, seed, prefix-cache policy and MTP settings.
- Open/close a GPU-driver diagnostic window. Stop on new driver failures; do
  not repeatedly attach CUPTI after even a trivial probe triggers an assertion.
- `gpu_graph_inventory/kernel_nodes` records describe retained graphs. Their
  counts are capture occurrences, not actual execution counts. A graph cache
  hit also does not prove that a particular conditional body ran.
- Native event intervals include graph scheduling, event-record overhead and
  contention. A long before/after bracket does **not** prove that the enclosed
  kernel was executing for the entire interval. This is especially important
  for small kernels on branches concurrent with GEMMs or collective waits.
- Native memset nodes need the same caution. A long event bracket around a
  short clear is not a bandwidth measurement: engine scheduling and concurrent
  collective progress may dominate it. Preserve byte/sentinel semantics when
  testing a fused initialization kernel, and require an uninstrumented model
  A/B before attributing a win to fewer nodes or a different execution engine.
- Adding events can change graph scheduling even when all original dependency
  edges are preserved. Instrument a narrow family first and compare against an
  uninstrumented run. Whole-graph instrumentation can materially perturb cost
  and overlap. Do not subtract a universal per-event constant to "correct" it.
- No CUPTI-free event technique supplies achieved occupancy, stall reasons or
  memory-transaction counters. Say those measurements are unavailable when
  attachment is unsafe; do not invent them from static launch geometry.

## Standalone observer

### Audit launch counts and native collective extents

Keep three separate quantities: captured kernel inventory, actual executable
submissions, and useful device work. Join submissions to an executable
generation, device, exact stream, instantiation flags and retained topology.
Native handles can be reused after retirement; a handle or node count alone
is not an identity. Unused captured buckets contribute no replay work.

Reconcile the request boundary before multiplying inventory by replay counts.
Benchmark graph-readiness preparation, warmup, measured requests and serving
reset are distinct phases. Preserve their source-owned counters or boundary
records. A preparation prefill outside measurement is not a second measured
prefill. Conversely, a terminal MTP ledger that counts selected drafts does
not prove that capacity preparation avoided extra physical sidecar forwards.
Audit the first transaction and retained subsequent requests explicitly.

Record the admitted bucket inventory separately from the configured ladder.
Prefix checkpoints can split a prompt into several retained parents even when
its length matches a configured bucket. Measure every parent submission in a
request; the last completed event interval multiplied by a chunk count is not
a full-request measurement. Report each TP participant and the slower
participant's complete span alongside any participant mean.

For communication-elision diagnostics, enumerate every removed and retained
collective, including vocabulary gathers and embedding sums. Matching compute
inventories does not prove representative inputs: removing a vocabulary-shard
embedding sum can leave an all-zero participant. Preserve an input-producing
collective in a separately labelled communication-included control, or bind
immutable replicated embeddings before capture. Subsequent elided layer sums
still invalidate model arithmetic; neither control certifies inference
correctness. Never promote the intervention into a production execution mode.

For a metadata-only diagnostic, read native kernel names, geometry, edges and
compiled parameter extents during setup. Copy only immutable host launch
argument storage; retain device pointer identities without following their
pointees. Matching names or grids do not prove duplicate work. Identical
arguments across stages may reflect legitimate reuse of a mutable arena bank.
To establish redundancy within one producer lifetime, identify the consumed
input, every output, intervening writers and required ordering edges. An
optional block-sum output also distinguishes two otherwise identical activation
quantizers; retain the producer that supplies every required consumer value.

A separate read-only structural observer may enumerate child graphs and retain
CUDA-owned conditional body handles at their original node creation. It must
forward creation/launch arguments unchanged, retire metadata with the owning
graph, and never add events, flatten a conditional or expose device state.
This is broader structural evidence than the event observer below supplies.
Conditional bodies in the event timing report remain opaque. Body inventory
is potential work; actual IF/WHILE/SWITCH multiplicity requires the production
device controller's execution evidence. Mutually exclusive attention branches
can contain similarly named kernels while only one performs arithmetic.

For NCCL, bind evidence to the loaded project DSO and its exact patched source.
Record the selected algorithm, protocol, channels, block size, node/stream
priorities and transport. Authenticate any native work decoder against a
compiled source layout and the actual kernel parameter ABI. Decode RING/TREE
work unions according to their selected operation, rather than assuming a
layout because its fields fit. A row-parallel projection halves K and still
needs a sum over its full output width. Capacity in a graph BOM is not actual
wire traffic: prove the device live-count owner and use native passive outgoing
payload receipts for partial, empty and large-to-small replays. MTP shifted
prefill needs its own row authority; the main prompt count cannot substitute
for it, and ragged request banks cannot use the first request's prefix count.

Host submission skew does not establish GPU compute overlap, and raw event
clocks on different devices do not share an origin. Compiled local-memory size
can include device call frames; static stack instructions alone do not prove
executed spill traffic. Attribute resources to one exact loaded symbol and SM.
Pinned SHM mappings absent from `/proc/PID/numa_maps` do not prove NUMA locality.
Retain these limits alongside any isolated protocol or dispatch experiment;
promote defaults only after matched unobserved Release/model and HTTP gates.

Experimental GPU DSOs must match the production architecture, math flags and
relocatable-device-code policy. Authenticate those options and loaded core
symbols before attributing register or timing differences to a kernel change.
Use one shared CUDA runtime with the application's ABI and link driver APIs to
the real driver DSO; duplicate static runtimes or driver stubs do not establish
a valid comparison.

`.agents/cuda-tuning/scripts/native_graph_event_trace.cpp` is an opt-in
`LD_PRELOAD` observer for CUDA 13's runtime graph entrypoints. It is deliberately
not linked into Llaminar. It intercepts `cudaGraphInstantiateWithFlags`,
`cudaGraphLaunch`, `cudaGraphDestroy` and `cudaGraphExecDestroy`, and uses driver graph queries so
opaque NCCL kernel functions can be named without runtime-symbol assumptions.

The observer retains every original node and dependency. Before/after event
nodes bracket selected operations. Event handles stay alive through executable
retirement, with their owning CUDA context selected. Export requires the
ordinary terminal completion to have happened; it never adds a per-node host
wait. Each JSON describes the **last completed replay** of one executable, not
all launches or a full-request timeline. Files are written only for selected
graphs that actually launched and retired normally.

Supported parent node types are kernel, memcpy, memset, empty, and conditional.
Conditional bodies are **opaque**: no event nodes are added inside them. CUDA
places stricter restrictions on those bodies. Do not flatten, replace or
disable the production conditional execution to get more convenient evidence.
Unsupported parent node types are explicitly logged and skipped. A benchmark
exit code of zero without trace files is not successful instrumentation.

Controls belong to the observer, not to production CLI configuration:

| Environment variable | Meaning |
| --- | --- |
| `LLAMINAR_NATIVE_EVENT_TRACE_DIR` | Existing output directory; unset disables observation. |
| `LLAMINAR_NATIVE_EVENT_TRACE_GRAPH_CONTAINS` | Match a parent kernel symbol; default `gdn_chunk_forward_kernel` selects hybrid Qwen prefill parents. Choose an actual prefill symbol for another model. |
| `LLAMINAR_NATIVE_EVENT_TRACE_MIN_NODES` | Minimum parent size, default 1000; use 1 only for a focused smoke test. |
| `LLAMINAR_NATIVE_EVENT_TRACE_KERNEL_CONTAINS` | Comma-separated symbol substrings to bracket; omitted measures every supported parent node. Unmatched filters produce a skip diagnostic. |
| `LLAMINAR_NATIVE_EVENT_TRACE_ALL_FLAT` | Bypass the parent symbol selector for a model-free smoke. Do not apply indiscriminately to fragments later imported into conditional bodies. |
| `LLAMINAR_NATIVE_EVENT_TRACE_REQUIRE_STAGE_NAMES` | Require completed canonical stage capture scopes on each selected parent; missing or stale provenance is fatal. |
| `LLAMINAR_PROFILER_NORMAL_EXIT` | Engine diagnostic exit mode; set to 1 so normal retirement writes evidence rather than `_exit` discarding it. |

## Build and run

Run from the source root. The host needs the real CUDA driver and the same CUDA
toolkit ABI as the binary. The helper only uses host C++; it does not compile a
new GPU kernel. Neither the helper nor the SDK belongs in the serving image.

A CUDA-enabled build also produces the standalone CMake target
`v2_cuda_native_graph_event_observer` as
`tests/v2/libv2_cuda_native_graph_event_observer.so`. Prefer that build-bound
artifact when validating a qualified tree; its C++20 requirement and real
capture fixture are owned by CMake. The manual command below is useful for an
isolated diagnostic helper and must retain its source/compiler/DSO identity.

```bash
LLAMINAR_TRACE_DIR="$(mktemp -d /tmp/llaminar-cuda-events.XXXXXX)"

g++ -std=c++20 -O2 -shared -fPIC \
  -I/usr/local/cuda/include -Iexternal/vendor \
  .agents/cuda-tuning/scripts/native_graph_event_trace.cpp \
  -o "$LLAMINAR_TRACE_DIR/native-event-trace.so" \
  -Wl,-l:libcuda.so.1 -ldl -pthread

python3 tests/v2/e2e/server/gpu_driver_diagnostics.py begin \
  --state "$LLAMINAR_TRACE_DIR/driver.state.json"

# Substitute existing, unprofiled plan/prompt paths and the benchmark's exact
# sampling flags. A short diagnostic is not a replacement benchmark score.
env -u LLAMINAR_PROFILING -u LLAMINAR_PERF_STATS_GPU_STAGE_TIMING \
  LD_PRELOAD="$LLAMINAR_TRACE_DIR/native-event-trace.so" \
  LLAMINAR_NATIVE_EVENT_TRACE_DIR="$LLAMINAR_TRACE_DIR" \
  LLAMINAR_NATIVE_EVENT_TRACE_KERNEL_CONTAINS=exclusive_scan_kernel,prefill_group_scan_scatter_deterministic_runtime_kernel \
  LLAMINAR_PROFILER_NORMAL_EXIT=1 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_BENCHMARK_ITERATIONS=1 \
  build_v2_release/llaminar2 benchmark \
    --config /path/to/saved-plan.json \
    --prompt-file /path/to/fixed-prompt.txt \
    --temperature 0 --seed 42 -n 16 \
    --benchmark-json-output "$LLAMINAR_TRACE_DIR/benchmark.json" \
  > "$LLAMINAR_TRACE_DIR/run.log" 2>&1

# Run this even after a failed benchmark; use an EXIT trap in an automated run.
python3 tests/v2/e2e/server/gpu_driver_diagnostics.py finish \
  --state "$LLAMINAR_TRACE_DIR/driver.state.json" \
  --report "$LLAMINAR_TRACE_DIR/driver.report.json"

rg 'native event trace' "$LLAMINAR_TRACE_DIR/run.log"
jq -e '.diagnostic_only and .launches > 0 and
  ([.nodes[] | select(.instrumented)] | length > 0)' \
  "$LLAMINAR_TRACE_DIR"/cuda-event-*.json
```

The normal frontend can bootstrap MPI. Preserve the observer environment on
the participating ranks and use process/device IDs in the files to verify
ownership; no `--no-mpi-bootstrap` is necessary for this helper. A remote
rank would need the same shared-library path. Do not preload it into unrelated
services. For a standalone smoke test, keep the same driver window but use
the model-free captured replay test below; it has no frontend MPI bootstrap:

```bash
env LD_PRELOAD="$LLAMINAR_TRACE_DIR/native-event-trace.so" \
  LLAMINAR_NATIVE_EVENT_TRACE_DIR="$LLAMINAR_TRACE_DIR" \
  LLAMINAR_NATIVE_EVENT_TRACE_ALL_FLAT=1 \
  LLAMINAR_NATIVE_EVENT_TRACE_MIN_NODES=1 \
  build_v2_integration/tests/v2/v2_integration_cuda_moe_kernel \
    --gtest_filter=Test__CUDAMoEKernel.RouteWithTensorsTiledPrefillCapturesAfterWarmup
```

Afterward require identical generated token IDs, live row counts and MTP
draft/accept/reject work against the same uninstrumented diagnostic. Verify
prefix-cache hits and model inputs as well. Report instrumentation overhead
separately. Leave all observer variables and `LD_PRELOAD` unset for canonical
timing. A filter or parent-selector miss is an investigation failure, not zero
kernel cost.

### Captured model stage timing

The engine's optional `llaminar_native_graph_stage_annotation_v1` ABI publishes
immutable stage identity only during setup recording. The observer snapshots
native node frontiers on that stage's exact device/stream and joins them to the
event trace before instantiation. No timing callback executes during replay.
Absent observation introduces no serving dependency, GPU event or readback.
The model-free `V2_Integration_CUDANativeGraphStageTiming` preflight proves that
named RMSNorm/residual stages retain captured snapshot byte equality with this
instrumentation. Scope lifecycle and report interpretation also have
device-free Unit and explicit preflight gates.

For a per-stage report, omit the kernel filter, set
`LLAMINAR_NATIVE_EVENT_TRACE_REQUIRE_STAGE_NAMES=1`, and retain the same
unprofiled plan and paired control. Render only that exact trace cohort:

```bash
python3 tests/v2/performance/kernels/native_graph_stage_report.py \
  --trace-dir "$LLAMINAR_TRACE_DIR" \
  --output "$LLAMINAR_TRACE_DIR/stages"
```

`stage-timing.html` is a standalone interactive table/timeline, accompanied by
JSON and CSV. It reports each stage's GPU span, covered interval union, owned
native-node interval sum, nesting and measurement completeness. Stage spans
include gaps between their native operations. Concurrent and nested stages
overlap; summing their spans or node intervals does not give graph elapsed
time. Unowned native plumbing remains explicitly unattributed. Zero-node
capture scopes are empty, while filtered or missing events are unmeasured.
Conditional bodies remain opaque, so this observer does not claim per-stage
attribution inside a conditional generation loop. Only the final completed
replay of each retained executable is shown, not a whole-request sum.

### Distinguish queue delay from kernel execution

When a tiny kernel acquires an implausibly large multi-device event bracket,
narrow the observer to that exact family first. If it remains large, inspect
the retained kernel's arguments with `cuGraphKernelNodeGetParams` **after
capture**. Decode only a verified ABI; pointer-sized arguments are not tensor
values. `cuPointerGetAttribute` with `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` and
`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` can establish local VRAM versus mapped
host or peer storage without downloading tensor data. Also record the actual
logical extents: identical grids alone do not prove equal work.

If that does not resolve the attribution, an isolated, numerically equivalent
diagnostic kernel may record per-block `%globaltimer` start/end timestamps.
Observe them only at the normal terminal/retirement boundary, using a separate
explicit diagnostic stream and one terminal join. Keep timestamp storage owned
through every replay and retire it with the graph. Never add this machinery to
ordinary inference, call a substituted diagnostic kernel an unchanged
production benchmark, or blindly add block barriers around a kernel whose
early exits are not converged. Authenticate the replacement symbol, inspect
its ISA/spills, test captured replay first, and require matching model tokens.

The earliest block start to latest block end measures an on-device work span;
the surrounding event interval can additionally include scheduling, contention
and retirement. Timestamp stores and any added barriers perturb the probe.
Inspect clock granularity; a zero tick difference is below resolution, not
zero execution cost. Do not subtract event and device-clock absolute timestamps
or compare absolute clocks across GPUs. This distinction is especially
important for zero-communication DAG estimates: zeroing collective nodes does
not remove their queue delay or interference already charged to compute-node
brackets. Such a DAG must remain labeled a contaminated diagnostic bound,
not a communication-free compute-scaling certificate.

## Reading the evidence

Keep `start_ms`/`end_ms` in the per-graph device-clock domain. Different GPUs'
event clocks cannot be aligned by comparing raw timestamps. Join the exact
kernel name **and geometry**, preserve original parent edges, and label opaque
nodes rather than attributing their complete cost to a guessed operation.
Example selected-interval summary:

```bash
jq '[.nodes[] | select(.instrumented) |
      {name, grid, block, interval_ms: (.end_ms - .start_ms)}] |
    group_by([.name, .grid, .block]) |
    map({name: .[0].name, grid: .[0].grid, block: .[0].block,
         calls: length, inclusive_interval_ms: (map(.interval_ms) | add)})' \
  "$LLAMINAR_TRACE_DIR"/cuda-event-*.json
```

For a communication-free model, re-time a complete dependency DAG with the
communication durations set to zero. Preserve a separate estimate with
observed queue order/gaps when that evidence exists. Do not sum overlapping
times, assume unknown peer readiness, infer physical queues from CUDA stream
counts, or call a subgraph estimate full-model tok/s. Selected-only intervals
are insufficient for this calculation; broad event brackets are still
perturbed intervals, not pure service costs.

When tuning compute scaling on a communication-bound topology, use this
communication-discounted full-model comparison alongside communication-free
local kernel probes. Keep the ordinary communication-enabled model run as the
numerical/driver-health and end-to-end check; a neutral wall-time result alone
does not reject a real compute-critical-path improvement. Compare control and
candidate with the same binary, exact workload, observer, topology, and graph
phases. Include the main and suffix prefill chunks on every participant, not
only the changed kernel or the leading GPU. Report absolute compute time and
single-to-multi-device scaling separately, because improving the single-device
case more can reduce the ratio despite a faster multi-device case.

A maximum of participant-local critical paths is only a lower-bound envelope
when cross-device producer/consumer readiness is missing. Say so explicitly.
Do not manufacture collective matching from repeated symbol names or compare
different device clocks. A joint DAG requires authenticated collective identity
and peer-ready edges. Preserve packing/unpacking in the main zero-communication
estimate; removing it too is a separate idealized bound. Include host/prefix
work separately before calling anything a whole-request estimate. Never report
the counterfactual as measured communication-disabled model throughput.

For a single-thread audit, distinguish literal grid×block size 1 from a wider
kernel whose leader performs data-sized work. Scalar publication after a
parallel reduction and one lane per independent request are not bulk-serial
defects. Integer expert-count scans can use parallel prefixes without changing
the answer when their validated range cannot overflow. That does not authorize
reassociating floating-point reductions or changing stable route order.

## Account for host time outside captured graphs

A main/tail graph pair is not the entire timed prefill. Request admission,
prefix checkpoint allocation/export and the final completion-event wait also
belong to the production path. Preserve prefix caching and its chunk boundaries
while measuring that remainder; disabling them is not an optimization proof.

GDB entry/return breakpoints can locate these boundaries, but all-stop pauses
can severely perturb multi-GPU scheduling. Pending breakpoints must actually
resolve and fire in the compute process. If the frontend bootstraps MPI, launch
the debugger **inside** the same explicit `mpirun` placement instead of assuming
it follows a re-executed child. Do not interpret a breakpoint's inclusive wait
as CPU compute or use debugger throughput as the model score.

For less intrusive attribution,
`scripts/host_prefill_boundary_trace.cpp` interposes the actual exported C++
host methods without calling a CUDA/HIP API. It records prefill, harvest,
pinned allocation, Qwen MoE runtime archive and terminal-wait intervals into
fixed host storage. It works with CUDA or ROCm plans. This is an optional
diagnostic shared library, never linked into the engine or serving image.

The helper assumes the current x86-64 ELF/C++ ABI. Confirm the core still
exports the exact measured method symbols before using it; a successful run
with no trace records is an instrumentation failure. Its bounded record table
fails rather than silently truncating. The output path has one process owner:
use this procedure for a single MPI compute rank, including rank-local GPU TP.
It does not trace independent remote ranks merely because root prefill waits
for them. Only calls nested inside benchmark prefill are recorded; startup,
decode and teardown are outside the timed domain.

```bash
LLAMINAR_HOST_TRACE_DIR="$(mktemp -d /tmp/llaminar-host-prefill.XXXXXX)"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -shared -fPIC \
  .agents/cuda-tuning/scripts/host_prefill_boundary_trace.cpp \
  -o "$LLAMINAR_HOST_TRACE_DIR/host-boundary.so" -ldl

nm -D --defined-only build_v2_release/libllaminar2_core.so | \
  rg 'BenchmarkRunner10runPrefill|OrchestrationRunner7prefill|Backend14allocatePinned'

# Use the same driver begin/finish window and exact baseline environment as
# above, including any separately measured native-collective settings.
env -u LLAMINAR_PROFILING -u LLAMINAR_PERF_STATS_JSON \
  -u LLAMINAR_PERF_STATS_CSV -u LLAMINAR_PERF_STATS_SUMMARY \
  -u LLAMINAR_PERF_STATS_GPU_STAGE_TIMING -u LLAMINAR_GPU_STAGE_TIMING \
  -u LLAMINAR_GPU_STAGE_TIMING_DETAIL \
  LD_PRELOAD="$LLAMINAR_HOST_TRACE_DIR/host-boundary.so" \
  LLAMINAR_HOST_BOUNDARY_OUTPUT="$LLAMINAR_HOST_TRACE_DIR/spans.jsonl" \
  LLAMINAR_PROFILER_NORMAL_EXIT=1 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 LLAMINAR_BENCHMARK_ITERATIONS=3 \
  build_v2_release/llaminar2 benchmark \
    --config /path/to/saved-plan.json --prompt-file /path/to/fixed-prompt.txt \
    --temperature 0 --seed 42 -n 16 \
    --benchmark-json-output "$LLAMINAR_HOST_TRACE_DIR/benchmark.json" \
  > "$LLAMINAR_HOST_TRACE_DIR/run.log" 2>&1

jq -se 'any(.[]; .name == "benchmark_prefill") and
  any(.[]; .name == "device_harvest")' \
  "$LLAMINAR_HOST_TRACE_DIR/spans.jsonl"
jq -s 'group_by(.request) | map({request: .[0].request,
  spans: (group_by(.name) | map({name: .[0].name, calls: length,
    inclusive_ms: (map(.duration_ms) | add)}))})' \
  "$LLAMINAR_HOST_TRACE_DIR/spans.jsonl"
```

Require a clean driver report and identical per-iteration token IDs/MTP work
against an otherwise identical invocation with `LD_PRELOAD` and
`LLAMINAR_HOST_BOUNDARY_OUTPUT` unset. Keep normal retirement and all other
settings the same in that control. Measure timer overhead before using the
attribution; the diagnostic's short generation is not the canonical benchmark.
Trace request IDs also include readiness/warmup prefills. Identify the final
measured invocations from the benchmark iteration count rather than silently
averaging initialization into steady state.

These intervals are **nested**: allocation is inside harvest, which is inside
prefill. Do not add their totals together. GPU work can run while the host
allocates archives; a long allocation total may be almost completely hidden.
Conversely, a runtime-state export may wait for earlier GPU work and must not
be labeled serialization CPU time. The final terminal-event wait is already
inside benchmark prefill, not an additional latency charge. Cross-reference
the captured graph DAG before choosing which dependency to change.

For process-to-process timing swings, bracket only the complete parent first
and correlate each measured replay with the exact host request. A final-replay
event value cannot explain an earlier iteration. If needed, trace the native
host-allocation call separately from its enclosing backend method; this separates
runtime page pinning from engine lock/bookkeeping. Preserve token IDs, chunk
boundaries, MTP work and actual loaded-core identity throughout the comparison.
An allocation-only diagnostic intervention can establish causality, but the
production fix for repeated payload materialization is persistent admitted
backing with event-aware range reuse. Do not promote a faster hot-path allocator.
The GPU RAM prefix tier should report one `ram_arena_materializations` per tier
at setup and `ram_arena_payload_leases` during harvest, with zero native pinned
allocations inside timed prefill. Zero allocation trace rows are therefore an
expected success state; require harvest/request provenance rather than inventing
an allocation event to make the observer's authentication pass.

## Compiler resources without attachment

For graph families with an engine inventory publication point, query native
launch resources during setup without inserting timing-event brackets:

```bash
env -u LD_PRELOAD -u LLAMINAR_PROFILING \
  LLAMINAR_GPU_GRAPH_KERNEL_INVENTORY=1 \
  LLAMINAR_PERF_STATS_FILTER=gpu_graph_inventory,forward_graph \
  LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=0 \
  LLAMINAR_GPU_STAGE_TIMING=0 LLAMINAR_GPU_STAGE_TIMING_DETAIL=0 \
  LLAMINAR_PERF_STATS_JSON=/tmp/cuda-inventory-rank{rank}.json \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 LLAMINAR_BENCHMARK_ITERATIONS=1 \
  build_v2_release/llaminar2 benchmark \
    --config /path/to/saved-plan.json --prompt-file /path/to/fixed-prompt.txt \
    --temperature 0 --seed 42 -n 16

jq '[.records[] |
  select(.domain == "gpu_graph_inventory" and .name == "kernel_nodes") |
  {device, captured_occurrences: .count, kernel: .tags.kernel,
   fragment: .tags.fragment, grid: .tags.grid, block: .tags.block,
   registers: .tags.registers_per_thread, local_bytes: .tags.local_bytes_per_thread,
   static_smem: .tags.static_smem_bytes, dynamic_smem: .tags.dynamic_smem_bytes,
   theoretical_blocks_per_sm: .tags.max_active_blocks_per_sm}]' \
  /tmp/cuda-inventory-rank0.json
```

Keep the exact sampling settings and saved topology of the original diagnostic.
The native occupancy query reports a resource ceiling, not achieved occupancy.
Inventory publication covers registered transaction/fragment callsites; it does
not currently promise every ordinary prefill parent. A missing family is not
proof of absent execution. Use the native parent observer for that structure,
and the compiled object for its resource evidence. None of these counts should
be relabeled as executed launch counts.

Inspect the actual compiled object, not an isolated recompilation with different
flags. `cuobjdump` needs no profiler attachment or GPU execution:

```bash
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  build_v2_integration/CMakeFiles/cuda_backend.dir/kernels/cuda/moe/CUDAMoEKernels.cu.o \
  > /tmp/llaminar-moe-cuda-resources.txt
rg -n -A 1 'Function .*exclusive_scan_kernel|Function .*prefill_group_scan_scatter_deterministic_runtime_kernel' \
  /tmp/llaminar-moe-cuda-resources.txt
```

Keep the full output's architecture headings with the evidence: registers and
local-memory allocation can differ across SM targets. `REG`, `SHARED`, `LOCAL`
and `STACK` establish compiler resource use, not achieved occupancy or dynamic
spill traffic. Join them to the actual launch block and any dynamic shared
memory from the graph inventory. Retain the normal spill-guard build proof for
every shipped architecture; absence of local storage in one native-device
object cannot certify other targets.

When runtime `local_bytes_per_thread` disagrees with the object, inspect the
**final linked core** as well:

```bash
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  build_v2_release/libllaminar2_core.so > /tmp/llaminar-linked-cuda-resources.txt
rg -n -A 1 'Function .*capturedTransferFusedKernel|Function .*publish_router_owned_rows_kernel' \
  /tmp/llaminar-linked-cuda-resources.txt
```

Device linking can add a diagnostic `printf` call frame to a kernel that had
zero `STACK` in its object. Runtime local-memory attributes include that frame;
they do not identify register spills or prove hot-path scratch traffic. Retain
the final symbol/architecture evidence, inspect the call path, and distinguish
it from arithmetic-kernel private storage. Do not weaken the compiler spill
guard or classify an unexplained nonzero allocation as harmless.

For a small routing change, the existing model-free public grouping harness
provides an uninstrumented native-event comparison before a full-model run:

```bash
cmake --build build_v2_release --target v2_perf_gpu_expert_pipeline --parallel
mpirun -np 1 --bind-to socket --map-by socket \
  build_v2_release/tests/v2/v2_perf_gpu_expert_pipeline \
  '--gtest_filter=Grouping/GPURuntimeGroupingPerf.CapturedStableGrouping/CUDA_Rows512'
```

It captures the production clear/count/scatter transaction, checks stable route
ordering, and emits `MOE_GROUPING_ECONOMY` latency in microseconds. Measure the
same exact shape/backend on both versions in interleaved A/B and B/A order.
An isolated grouping win does not establish a whole-model or multi-GPU win.

The same binary can isolate the public expert-pipeline boundaries at the local
geometry corresponding to a particular ownership degree:

```bash
env -u LD_PRELOAD -u LLAMINAR_PROFILING \
  -u LLAMINAR_PERF_STATS_JSON -u LLAMINAR_PERF_STATS_GPU_STAGE_TIMING \
  CUDA_VISIBLE_DEVICES=0 \
  build_v2_release/tests/v2/v2_perf_gpu_expert_pipeline \
    '--gtest_filter=ProductionShape/GPUExpertPipeline.CapturedPipeline/CUDA_Qwen36WholeExpertShard2Rows512'
```

`moe_phase_sample` retains individual timings for grouping, owner gate/up,
down projection and fold. `moe_phase_resource` supplies the actual native
kernel/launch identity and compiler resource ceilings. The `whole_expert` and
`down_columns` records are distinct workloads: the latter consumes all routes
but owns only a column slice. These are local-shape probes on one physical GPU;
degree two in the name does not measure two-device communication or inference.
Do not add complete-pipeline times to their component phases. Preserve format,
live rows, physical capacity and routing distribution when comparing revisions.

The shared harness also has `_Route_hotset` and `_Route_power_law` cases,
generated from `NativeVNNIMoERoutingProfiles`, not a second fixture-local
distribution. `moe_route_profile` records the selected distribution and actual
owned live route count. Their phase identities include the profile suffix to
prevent accidentally combining unlike samples. Ordinal expert ownership under
skew need not halve that count; these are participant-zero local-cost probes,
not maximum-participant load measurements. The complete skewed inventory is
registered separately as `V2_Perf_ExpertShardRouting_CUDA` / `_ROCm`, outside
production preflight. For a quick two-way A/B, select
`*CapturedPipeline/CUDA_Qwen36WholeExpertShard2Rows*` (six main/suffix cases).

For an A/B against a frozen shared core, verify the performance caller still
matches that core's public ABI. A stale caller that fails admission is not a
kernel regression. Reject incorrect, spilling or slower local candidates before
paying full-model setup costs; a local winner still needs model A/B and B/A,
identical token/MTP work, and the applicable production tests.

## GDN geometry and local shard scaling

Use the retained-graph GDN harness to measure head ownership and shared-input
layout changes without attaching a profiler:

```bash
cmake --build build_v2_release --target v2_perf_cuda_gdn_verifier_rows --parallel
env -u LD_PRELOAD -u LLAMINAR_PROFILING \
  -u LLAMINAR_PERF_STATS_JSON -u LLAMINAR_PERF_STATS_GPU_STAGE_TIMING \
  CUDA_VISIBLE_DEVICES=0 \
  build_v2_release/tests/v2/v2_perf_cuda_gdn_verifier_rows \
    '--gtest_filter=*.PrefillHeadShardCapturedSweep:*.PrefillInputLayoutCapturedSweep'
```

Use the driver-diagnostic begin/finish window above. The input-layout sweep
covers key widths 64/128, value widths 17/128, live rows 64/448 and head counts
3/16/32. `gdn_geometry_sample` identifies M, heads, key width, value width,
sample index and microseconds; preserve every field when joining timings.
`gdn_resource` reports native launch/compiler ceilings, not hardware counters.
Head-count ratios on one device represent local shard shapes, not multi-GPU
inference scaling.

The timing fixture uses a zero fixed point to permit repeated captured replay.
It is not a numerical oracle for a recurrence change. Run the nonzero,
byte-exact GDN integration proofs, including grouped/serial and padded-live-row
cases, before a whole-model A/B. Shared-memory layout improvements can move the
local microbenchmark while leaving the full graph unchanged: reject candidates
whose full-model gains reverse between A/B and B/A order. Preserve the original
prefix-cache policy and compare equal token streams and MTP work.

## Rank local shard shapes without communication

Separate two questions: how well a compute operation scales when its local
ownership shrinks, and how much of that operation is on the real model's
critical path. A poor ratio on a tiny kernel is not necessarily the first
tuning target. Conversely, a near-flat expensive recurrence can remain a major
bottleneck even when dense projections scale well.

For the first question, capture the production entrypoint at degrees 1, 2 and
4 **on the same physical GPU and linked Release core**. Freeze source formats,
activation/KV precision, installed dispatch and persistent workspaces. Degree
here names a local shape, not a physical multi-card measurement. Run GPU probes
serially, clear profiler/inventory/timing variables, retain all raw samples and
close the driver-diagnostic window even on failure. A skipped test or missing
completion marker is not evidence.

Communication is absent from these local timed graphs. Do not replace a full
model's collectives with no-ops: that changes downstream inputs and routing,
so a speed gain can simply reflect invalid work. Prepare valid imported data
outside timing and retain every necessary local producer/consumer dependency.
Normal communication-enabled inference remains the correctness and end-to-end
check. A topology-bound neutral model result alone does not disprove a local
gain; separately report absolute latency, per-doubling ratios, real-route
sensitivity and any suffix regression. A faster candidate may have a worse
scaling ratio if it improves the single-device operation more.

The existing Qwen-oriented probes cover GDN, attention, dense projections and
the grouped expert pipeline. The exact registered filters remain in source:

```bash
cmake --build build_v2_release --parallel --target \
  v2_perf_cuda_gdn_verifier_rows v2_perf_cuda_flash_attention \
  v2_perf_cuda_native_vnni_gemm v2_perf_gpu_expert_pipeline

env CUDA_VISIBLE_DEVICES=0 build_v2_release/tests/v2/v2_perf_cuda_flash_attention \
  '--gtest_filter=*.Prefill_Qwen36MoE_TPShardScaling'

env CUDA_VISIBLE_DEVICES=0 build_v2_release/tests/v2/v2_perf_gpu_expert_pipeline \
  '--gtest_filter=*CapturedPipeline/CUDA_Qwen36WholeExpertShard*'
```

Use the GDN command above and the dense harness's `CapturedProductionShapes`
selector with explicit production formats/shapes/M. Do not silently replace a
main graph's physical bucket with its live count: dense kernels may execute the
physical M while attention/GDN execute live rows. Prefix harvesting can split
one prompt into distinct main and suffix graphs. Measure both using the actual
KV position and length, not an unrelated full-bucket attention call.

Normalize raw records to the backend-neutral schema consumed by
`tests/v2/performance/kernels/plot_kernel_shard_scaling.py`:
`operation,workload,degree,sample,latency_us,disposition,calls`. The optional
positive `calls` is the number of occurrences in the measured workload, not a
universal model constant. `disposition` is `partitioned`, `replicated` or
`collective`; only partitioned work has an ideal halving expectation. Keep
backend, model/shape, binary and precision provenance alongside the input CSV.

```bash
python3 tests/v2/performance/kernels/plot_kernel_shard_scaling.py \
  --samples /path/to/normalized-captured-samples.csv \
  --output-prefix /path/to/results/scaling \
  --threshold 1.9 --rank-degree 2
python3 tests/v2/unit/scripts/test_kernel_shard_scaling.py
```

The report rejects incomplete degree sets, repeated sample identities and
mixed ownership semantics. It uses medians and checks each doubling separately.
Its priority score is `calls * max(0, T_after - T_before/2)`. This is local-time
excess, **not additive achievable model savings**. Do not rank a complete
pipeline alongside its constituent kernels, count alternative whole-expert and
column-owned down modes together, or mix a GEMM producer-only interval with a
quantize/GEMM/reducer pipeline. Preserve main and suffix curves even if a
family-level summary combines their scores. Investigate apparent superlinear
cliffs rather than presenting them as general scaling promises.

For the second question, authenticate the same operation in the executed
production graph. Use exact symbols, geometry, dependencies and, on ROCm,
unambiguous physical queue binding. Captures with the same node count may have
different dispatch choices, especially suffix graphs; node count alone is not
identity. Recompute the dependency critical path with communication durations
zero, reporting queue order, scheduler gaps and payload-layout removal as
separate assumptions. Do not subtract inclusive communication bars or add
overlapping compute families. Keep host/prefix work outside the graph visible.
Profiling throughput is diagnostic, not the unprofiled model score.

Compiler resource ceilings and grid counts can establish launch underfill;
they cannot supply achieved occupancy or prove a cache/stall explanation.
Preserve safety restrictions on profiler attachment. GDN's zero-fixed-point
timing fixture and balanced synthetic MoE routes need the nonzero correctness
oracles and real-prompt model evidence before any tuned candidate is retained.

## Captured compute/native-collective overlap

Before adding row-pipeline graph nodes, isolate whether overlapping a real
output projection with the existing native sum is economical:

```bash
cmake --build build_v2_release --target v2_perf_projection_collective_pipeline --parallel
env -u LD_PRELOAD -u LLAMINAR_PROFILING \
  -u LLAMINAR_PERF_STATS_JSON -u LLAMINAR_PERF_STATS_GPU_STAGE_TIMING \
  build_v2_release/tests/v2/v2_perf_projection_collective_pipeline \
    '--gtest_filter=Captured/ProjectionCollectivePipelinePerf.CUDA/Rows448'
```

Use the driver window above and preserve the baseline's native-library/channel
settings and NUMA placement. The same binary has `ROCm` cases and rows 64, 448
and 512. It requires two visible devices of the selected backend. Each sample
reports the maximum participant interval for eight captured transactions;
`projection_collective_sample` retains the raw observations, and
`PROJECTION_COLLECTIVE` reports microseconds per transaction. No CUPTI attachment
or per-node timing events are involved.

Compare **all three** schedules: untiled serial, tiled serial and tiled overlap.
Beating tiled serial only proves that the fork helps that decomposition; it
does not prove the decomposition beats the existing whole projection. Test
both bulk and tail shapes, since native-call overhead and GEMM occupancy can
reverse the winner. Preserve the full-row byte oracle and untouched output
guards, and require the ordinary all-format functional gates plus whole-model
A/B before installing any scheduling policy. The probe's Q6_K shape is not an
all-format or model certificate.

These graphs use fixed **live** row extents, not allocation capacity. They do
not implement device-variable native message counts or certify partial-bucket
communication. Internal captured fork/join edges use
`IWorkerGPUContext::recordEventChecked`/`waitEventChecked`; `IBackend` events
publish externally completed work and reject capture-time publication.
Ordinary FP32 host views are not GPU storage aliases. The probe explicitly
retains an output-only device slice, whereas production lowering must use its
admitted arena/coherence contract. Keep the performance cells outside preflight.

## Native collective protocol probes

A direct NCCL/RCCL probe must reproduce the engine's communicator admission,
not just its device list. For node-local NCCL, follow `NCCLNetworkPolicy` and
`NCCLCoordinator`: concurrent `ncclCommInitRankConfig` calls with the configured
Socket network module and the same pre-initialization graph-ordering policy.
That network choice preserves native GPU P2P and SHM selection. `ncclCommInitAll`
does not reproduce the per-communicator policy and can enter unrelated network
plugin discovery. Do not diagnose such a startup failure as a graph/kernel
regression; run the unchanged library under the same probe before attributing it.

Keep experimental DSOs isolated and explicitly loaded. Match source revision,
canonical capture patches, compiler, GPU targets and generated function
inventory. RCCL's shipping inventory is `scripts/docker/rccl-functions.txt`;
compare it with the installed build's `ONLY_FUNCS` before configuring. An
allgather-only build can prove that protocol but not whole-library resources;
an unrestricted upstream build can likewise differ materially from the shipped
inventory. Inspect linked resources, not just an individual object. Preserve
failed receipts even when the GPU-driver report is clean: a host crash is still
a failed experiment. New startup-policy choices are not authorized implicitly.

If a prototype changes the native work descriptor, first compile a device-free
geometry probe for its actual size, alignment, batch byte budget and work items
per batch. A larger descriptor can leave a byte-budget-derived item count at
zero, breaking algorithm costing even while a selected ring still executes
correctly. Preserve the native batching policy with type-derived geometry and
compile-time bounds; do not tune around an invalid denominator. A resource
control with the same changed ABI isolates compiler effects, but is not proof
that the ABI still preserves the installed library's algorithm selection.

For device-variable payloads, distinguish storage stride, protocol progress
and useful bytes. Correct live output alone cannot detect padded traffic. Use
retained large-to-small/empty/growing replay, passive primitive-byte receipts,
and poisoned remote tails; cover in-place and separate-buffer contracts. Disable
observation writes for matched timing. Primitive bytes exclude protocol framing
and do not measure PCIe transactions. Native algorithm/transport changes,
unequal-owner packets and cross-host protocols require separate evidence.

Run the same numerical and poison-tail checks with the observer **absent**.
An observer can deliberately select a counted implementation and thereby miss
the unobserved production specialization. Keep payload-count receipts for the
observed run; the unobserved run proves outputs and guards, not measured traffic.

Exercise native **work batching**, not only one collective per kernel. Mix
ordinary fixed work with independently changing row extents and unequal row
geometries in one group. Cross the inline argument-bank limit and authenticate
the retained storage kind from the actual native node's host-side launch
metadata. A captured graph must retain persistent work storage, not borrow the
ordinary transient FIFO. Keep the small inline-argument case as well. Count
replay sets separately from the banks/participants checked within each replay.
Vary reduction operators inside the same group too: function identity and
per-work reduction arguments must survive batching, including SUM/MIN/MAX.
One typed kernel entry can still call other native device functions for later
work items; its entry-only resource report is not a complete call-path proof.

For NCCL nodes recorded through the CUDA Driver API,
`cudaGraphKernelNodeGetParams` can return `cudaErrorInvalidDeviceFunction`
even when the retained kernel is valid. Use the real driver's
`cuGraphKernelNodeGetParams` and `cuFuncGetName`; if resolving with `dlsym`,
match the exact versioned symbol to the header's structure ABI (for example
`cuGraphKernelNodeGetParams_v2` with `CUDA_KERNEL_NODE_PARAMS_v2`). Bind those
symbols before capture, never from a toolkit stub. Authenticate the function
before decoding arguments, and validate packed `extra` tags/size or the actual
`kernelParams` representation. Derive private work layouts from the exact
native source/header revision. A failed query is missing diagnostic evidence,
not proof of absent work or an inference/driver failure. Do not attach CUPTI
merely to repair a graph-query ABI mismatch.

If extent resolution moves into the native work loader, preserve distinct
parameter-space and global-space load branches. A generic pointer to an
argument bank can make CUDA spill the whole bank into local memory. Keep
producer-pointer and resolved-extent states explicit, verify their concrete
sizes/pack offsets, and publish resolution through the loader's existing
barrier. Counts must be ready and immutable for their native batch; a later
producer inside the same native kernel is not ordered by that early load.
For ring-backed work metadata, mask **each** addressed word independently;
adding to an already-masked pack pointer can escape the FIFO at its last pack.
Prove ring wrap, virtual-offset rollover and linear persistent storage with a
device-free check of the same offset helper, then retained native replay.

For reductions, compare live elements byte-for-byte with the unchanged native
capacity schedule, including cancellation-sensitive FP32/FP16/BF16 and integer
inputs. Keep channel partitions, reduction order and protocol acknowledgement
counts fixed while clipping useful primitive arguments. An empty payload does
not authorize omitting a step that an existing peer/proxy expects. Test in-place
reduce-scatter's entire unowned input banks as well as output tails. A sum proof
does not certify other reduction operators or an unselected native algorithm.

Keep the full-live device-count case in timing alongside the fixed-capacity
control and partial-live case; it exposes the mechanism's overhead. Use frozen
DSO copies and alternate timing order without concurrent builds, GPU probes or
profiler writes. Never relink a DSO that a running probe has mapped. Require
complete driver windows and matched canonical-library inventories, not just a
successful process exit.

A work-uniform full-span specialization can remove per-chunk bounds overhead,
but cloning large primitive loops inside one entry can increase spilling.
Compare separate helper bodies against inlining with identical protocol and
register limits. Attribute any gain to the measured call path, not merely to
the source-level branch removal. Preserve the fixed-full control so a useful
partial-payload result cannot conceal an ordinary full-transfer regression.

For CUDA resource changes, compare ptxas spill-store/load and frame reports at
the **same native register ceiling** and inspect final linked code. Unchanged
register count or zero `LOCAL` metadata does not establish no memory spilling:
stack allocation can contain spills, and device helpers matter too. Preserve
the unchanged vendor baseline separately from the candidate's added pressure;
do not weaken the project spill guard or claim a microbenchmark is a production
model certificate. Compressed HIP/RDC cache experiments additionally require
the [header-invalidation proof](../../rocm-tuning/references/rocprof-isolated-kernel.md#timing-trace).

For a direct typed native entrypoint, compare the complete executed caller and
helper path against the original representative-kernel/indirect-helper path.
An absent call boundary can move spill allocation into the entry; neither one
entry's frame nor a sum of unrelated symbols proves runtime traffic. Keep the
native algorithm, protocol, channels and register ceiling unchanged during the
paired experiment. Record any added code-image footprint separately and run
fresh alternating timings; historical receipts are not an interleaved A/B.

A fully uniform native plan can eliminate the indirect branch rather than
merely make one function a fast case. Admission must inspect **every batch on
every channel**, derive function IDs from the generated inventory, and retain
the original grouped-work semantics. Test uniform multi-work groups and mixed
collective functions in the same group; authenticate the actual selected entry
and argument/persistent storage from captured launch metadata. A direct entry
must reject a foreign function, not reinterpret it as its specialized body.
Do not infer spill freedom from that source-level specialization: inspect the
final reachable ISA, and separate normal execution from assertion-only call
frames. Keep ordinary full-extent timing as an independent regression check.

If the native entry's launch bound appears broader than the executed plan,
query the retained node with `cuGraphKernelNodeGetParams_v2` and its function
with `cuFuncGetAttribute` (`NUM_REGS`, `LOCAL_SIZE_BYTES` and
`MAX_THREADS_PER_BLOCK`). Resolve the real driver through the existing explicit
driver boundary; no profiler attachment or device-state readback is needed.
NCCL uses driver function handles, so do not assume a runtime registration
query can resolve them. Record block width, grid and dynamic shared memory
alongside compiler resources. A whole-register-file division is not a complete
admission/occupancy proof.

### Distinguish native graph storage from context-owned kernel storage

Before treating a `cudaMemGetInfo` delta around graph instantiation as an
executable footprint, record the owning context's `cudaLimitStackSize` with
`cudaDeviceGetLimit` before and after that boundary. Join the retained kernel
inventory's `LOCAL_SIZE_BYTES` and function identity to those records. The
runtime may increase kernel-private/stack storage when a captured library
kernel first becomes executable; that storage belongs to the device context,
not exclusively to that graph. Destroy the complete executable family, then
the communicator, and observe each retirement boundary separately. Persistent
context bytes must not be relabeled as executable bytes or silently paid for
by increasing a graph allowance. Physical-memory contributions and resident
credits must preserve the actual owner and lifetime.

These setup-only native queries do not require CUPTI, graph replay, or model
state readback. Do not use a global stack-limit environment override to move
the allocation earlier and then claim it disappeared. If a standalone probe
links the core, use the exact target's conditional ABI definitions as well as
its ISA flags, includes and runtime libraries; options such as `USE_PROF_API`
can change tensor object layout. Prefer a CMake-owned focused test. A probe
with incompatible headers does not establish a production failure.

To measure a context-growth obligation without guessing CUDA's allocation
granules, use a cold, sealed captured family on its exact device/stream. Record
the original stack limit and free/total memory; set the required captured-kernel
limit, record the driver-aligned limit and footprint, then restore the original
limit. Require exact restoration of both the limit and free/total bytes. Only
then admit the measured growth through `PhysicalMemoryAuthority`, protecting
the family's still-unmaterialized allocations, and materialize the same
aligned limit under the returned context-lifetime lease. Require the second
physical observation to reproduce the first. Failed restoration or changed
footprint is a fatal preparation failure, not permission to add a reserve.

`prepareRuntimeContextStorage()` is the code-owned pre-instantiation boundary
for planning probes. CUDA retains the native-context PMA lease until successful
exclusive context reset; graph or communicator destruction must not release
it. HIP authenticates the same physical scope but has no CUDA-style
instance-time stack-limit expansion. Never run a stack-limit footprint probe
in a timed inference region or while recording a graph: the driver can order
prior work internally. This procedure changes accounting/preparation, not
kernel arithmetic, stream selection, precision, or native collective dispatch.

Build and run the independently launched cold-context and device-free admission
regressions, with the usual complete driver-diagnostic bookends:

```bash
cmake --build build_v2_integration --parallel --target \
  v2_test_physical_memory_authority v2_integration_planning_execution_measurement
ctest --test-dir build_v2_integration --parallel --output-on-failure \
  -R '^V2_Integration_(NativeExecutionContextMemoryAdmission|CUDA_NativeLocalTPContextStorageLifetime)$'
```

Run the affected public-auto HTTP cell afterward; the native primitive alone
does not certify model startup, placement costing, long-context inference, or
prefix reuse. Keep these correctness probes separate from performance timing.

Evaluate a narrower compiled-entry budget as a separate, explicitly identified
experiment after the unchanged-budget control. Host admission must check the
**complete plan's** block width as well as every function identity before
selecting that entry; test wider and mixed groups too. Do not reduce native
workers/progress geometry merely to fit the entry, globally raise the register
ceiling, or weaken spill rejection. Fewer spill instructions are not a finished
resource fix, and a spill-free compile is not proof of legal launch geometry or
better end-to-end latency.

If the task owner explicitly permits a measured spill exception, keep it narrow
and evidence-bound: exact native entry, source/DSO identity, architecture,
admitted block geometry, final spill/frame evidence, and matched full/partial
timings. Prefer the least invasive proven candidate; changing shared primitive
retirement to save a few spill bytes also expands the lifecycle proof to every
caller. An exception does not disable the project-wide compiler guard or make
unmeasured entries acceptable. Resource reduction alone is not the win: require
correct retained replay and throughput before accepting the trade-off.

## Duplicated computation across participants

Establish duplication from the actual input, weight and output ownership, not
from matching kernel symbols. A projection on distinct head or column shards
is different work even when both participants launch the same symbol. Compare
its local geometry and native dependency edges before proposing a shared owner.

For independent row transforms, compare replicated work against disjoint
**complete rows** followed by the real publication operation. Include packing,
native communication and any reconstruction needed by existing consumers in
the timed captured transaction. Preserve the arithmetic within each row;
splitting a floating-point reduction across devices is a separate numerical
contract. Authenticate every participant's complete result against the original
full-row producer over changed inputs and retained replay. Use A/B/B/A ordering
and maximum participant intervals, not the faster participant alone.

Retain producer side products that downstream consumers still need. For
example, row-owned router logits do not remove the need for full local hidden-Q8
data when each expert owner consumes tokens from the entire prompt. Omitting
that quantization or silently transferring the hidden tensor understates cost.
Sending selected IDs and weights is much smaller than exchanging all logits.
Measure short tails and dispatch crossovers as well as bulk prefill; reducing
duplicate arithmetic can still lose to extra native-call latency.

A fixed, fully-live geometry probe certifies neither device-variable counts nor
padded-bucket traffic. Production lowering must retain the canonical count
authority, communicate exact live extents, preserve native P2P selection, and
publish routing/histogram state exactly once at its existing semantic boundary.
Do not install capacity-sized collectives, host count readbacks or recapture to
make the experiment fit production. Keep the ordinary optimized control as the
benchmark comparator and require real-model correctness and end-to-end timing
before claiming an inference improvement.

The counted router transaction has a separate Release economy harness:

```bash
cmake --build build_v2_release --parallel --target v2_perf_moe_router_exchange
ctest --test-dir build_v2_release -V -R '^V2_Perf_MoERouterOwnedExchange_CUDA2$'
```

Use `ROCm2` or `ROCm4` for those actual physical memberships. The fixture
requires canonical no-P2P inventory and never disables native P2P to qualify.
It captures the shared router, one exact-count ID/weight packet exchange and
complete selection publication. It covers fully-live buckets and partial 512-row buckets;
`ROUTER_EXCHANGE` reports median max-participant intervals, and
`ROUTER_RESOURCE` reports native launch/resource ceilings. Include both small
and bulk live prefixes when selecting a policy. Its model-free, synthetic
prepared weights do not replace HF or whole-model A/B. Functional byte/count/
tail and replay checks are separately registered in preflight; timing stays
outside that gate.
