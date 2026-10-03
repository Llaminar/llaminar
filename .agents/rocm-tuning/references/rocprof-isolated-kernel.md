# Isolated ROCm Kernel Profiling

Use this workflow after PerfStats has identified an expensive captured graph or
fragment. Profile a standalone harness that invokes the exact production kernel
entrypoint with the same geometry, tensor format, stream contract, and
persistent workspace. Do not disable production graph capture to make rocprof
attach to whole-model inference.

## Timing trace

Keep the canonical repeated timing run unprofiled. Use a separate one-iteration
profiler run to identify dispatch indices and kernel symbols:

```bash
RESULT=/tmp/llaminar-rocm-kernel-profile
mkdir -p "$RESULT"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
rocprof --stats --timestamp on --basenames off \
  -d "$RESULT" -o "$RESULT/dispatch.csv" \
  ./build_v2_release/tests/v2/<isolated-perf-binary> \
  --gtest_filter='Perf__ExactSuite.ExactProductionShape'

column -t -s, < "$RESULT/dispatch.stats.csv"
```

Repeat the unprofiled harness enough times to establish a stable median. Treat
rocprof timing as attribution evidence, not as the canonical benchmark sample.

If a wrapper changes directory to collect native HIP graph DOT files, resolve
the executable, saved plan and prompt to absolute paths **before** that change.
Validate the executable exists first. Keep a launch/setup failure distinct from
a GPU failure; an empty trace does not establish zero kernel cost.

For a whole-model dependency audit, set `DEBUG_HIP_GRAPH_DOT_PRINT=1` inside
the resolved execution-rank wrapper before starting the profiled process. HIP
writes DOT files relative to that process's working directory. Match the
retained graph to an executed window using the complete kernel multiset, native
edges, device identity and an unambiguous stream/queue binding; graph filenames
or shared stream IDs alone do not authenticate that match. Keep admitted bucket
rows separate from actual live rows. Report any parent/child timestamp overlap
instead of silently adjusting clocks or reordering nodes to make a graph fit.

A communication-free estimate must recompute the dependency critical path with
the selected communication nodes assigned zero duration. Report separately
whether queue order, scheduler gaps and packing/unpacking are retained. This
holds measured compute service times fixed: removing communication may also
change resource contention, so the estimate is neither a benchmark nor proof
of achievable speed. Keep prefix harvesting, sidecars and host work outside the
selected graph visible rather than silently discounting them too.

For compute-scaling tuning, compare control/candidate full-model DAGs with that
same zero-communication rule, in addition to the isolated communication-free
shape curves. Include both main and suffix chunks and every participant.
Communication-included throughput remains an end-to-end/correctness check, not
the sole compute acceptance gate on a link-bound host. The maximum of separate
participant critical paths is only a bound unless exact cross-device readiness
edges are available; repeated RCCL kernel names do not identify those edges.
See the shared [full-model counterfactual limitations](../../cuda-tuning/references/native-graph-events.md#reading-the-evidence).

For a diagnostic phase ablation, first authenticate an unchanged interposed
control. Keep the measured phase's results observable and inspect the resulting
ISA/resources: an early return can otherwise delete the work or change register
allocation. An intentionally incomplete kernel may finish a timing harness but
does not pass inference correctness. Never install that ablation or label its
timing-harness success as a numerical certificate.

For an isolated HIP source interposer, keep **all** device entrypoints and
module-owned constant tables distinct from the core DSO, not only the changed
kernel. ELF/host-stub interposition can otherwise mix one module's launch with
another module's IQ-table initialization. First run an unchanged interposed
control through the same all-format oracle. A failed control invalidates the
experiment; it is not evidence of a production arithmetic regression. Use a
fresh driver-observation checkpoint for each attempt, preserving failed logs.

For isolated RCCL changes, follow the shared
[native collective protocol-probe procedure](../../cuda-tuning/references/native-graph-events.md#native-collective-protocol-probes).
Use the canonical shipping function inventory, existing HIP capture repair,
explicit host-memory startup ABI, and actual native transport. A narrow
protocol-only build is not a comparable full-library timing/resource result;
nor should a broader upstream inventory accidentally replace the release list.

Native work-loader experiments additionally need mixed fixed/live work,
independent counts, inline and persistent work banks, and actual captured-node
storage identity. Follow the shared protocol procedure for FIFO word masking
and batch-lifetime rules; a pure offset proof does not replace captured GPU
replay. Resolving metadata at an existing publication barrier is preferable to
another event/barrier only when every producer is already ordered before that
native batch. Do not infer that ordering from passing single-work tests.
Repeat functional replay with passive counters absent, as well as enabled:
instrumentation may select a different full-span specialization. Include mixed
operators within native groups, and compare complete caller/helper resource
paths when testing inlining against separate collective bodies. Keep the
installed fixed-full case alongside full-live and partial-live timings; fewer
useful bytes alone do not certify that full transfers have retained their speed.

For a compressed HIP/RDC build using ccache, prove header invalidation before
trusting incremental native-library experiments. With affected ccache versions,
the compressed preprocessor result can yield a direct-cache manifest containing
**zero header dependencies**: editing a device header then returns the old
object. A compiler command reappearing in Ninja output is not proof of a cache
miss. Use a private cache and a tiny header-defined constant consumed by both
host and device code; compile, change only that header, and inspect both the
cache result and emitted constant. Keep an uncompressed control to establish
the affected boundary rather than declaring every historical build invalid.

For a confirmed affected build, compiler-emitted dependency mode (`-MD`/`-MMD`
plus `CCACHE_DEPEND=1`) retains header dependencies. Use a **fresh, scoped cache
namespace** as well: enabling dependency mode alone can still hit an earlier
invalid direct-cache manifest. For an isolated CMake dependency build, the
launcher can be `env;CCACHE_DEPEND=1;CCACHE_NAMESPACE=<validated-new-namespace>;ccache`.
Prove both a header-change miss and a subsequent unchanged hit. Do not purge the
shared host cache or globally change unrelated build policy to repair one
experiment. Carry the regression into the build gate before promoting a
production compiler-cache policy change.

For duplicated cross-GPU work, use the shared
[ownership and complete-transaction comparison procedure](../../cuda-tuning/references/native-graph-events.md#duplicated-computation-across-participants).
It applies to both backends: preserve the actual ROCm producer, its reusable
side products and native communication policy rather than borrowing a CUDA
router merely because the visible output has the same shape.

For communication-free scaling, use the shared
[local-shape ranking and executed-DAG procedure](../../cuda-tuning/references/native-graph-events.md#rank-local-shard-shapes-without-communication).
ROCm counterparts are `v2_perf_rocm_gdn_verifier_rows`
(`PrefillHeadShardCapturedSweep`), `v2_perf_rocm_flash_attention_prefill`
(`CapturedPrefillShardScaling`), `v2_perf_native_vnni_sweep`
(`CapturedProductionShapes`) and the shared `v2_perf_gpu_expert_pipeline`
(`ROCm_Qwen36WholeExpertShard*`). Pin one physical ROCm device for the entire
1/2/4 curve and retain the logical-to-physical mapping. The dense sweep's
`LLAMINAR_ROCM_NVNNI_PROBE_OPERATION=producer` and `pipeline` modes isolate
different work: collect both separately before blaming poor GEMM scaling on
replicated quantization. Timing remains unprofiled; native DOT plus a separate
kernel trace corroborates actual graph identity and critical-path relevance.

These local probes execute **no communication**, rather than subtracting a
transport estimate from their timer. Do not implement the experiment by making
full-model collectives no-ops: missing reductions or intermediate exchanges
change downstream activations and expert routes. A faster invalid workload is
not a compute speedup. Prepare valid fixed inputs and complete imported
intermediates outside timing, then capture the same public production phase
at each local ownership geometry. Keep the independent serial-row oracle and
normal communication-enabled model checks.

Report absolute candidate latency and its 1→2/2→4 ratios separately. Improving
every shard can still worsen the ratio if the unsharded operation improves more.
Uniform routes are only a balanced compute bound; validate skewed profiles and
the actual prompt before installing a dispatch change. A neutral topology-bound
whole-model result alone does not disprove a local gain, but a real-route
regression or suffix loss must not be hidden by a favorable uniform main bucket.

## Matched whole-model rocprofv3 transactions

Use an unprofiled, counterbalanced benchmark first. Apply one saved public-auto
plan to both cohorts so that endpoint selection, NUMA affinity, weights,
prompt, context, precision and sampling do not drift. Compare actual executed
MTP depth, acceptance and exact output as well as throughput: fixed depth 3
versus dynamic 1..15 is not an acceptance-policy experiment if both execute
depth 3 throughout the selected window. Keep the complete dynamic range;
narrowing it hides reserved-envelope overhead rather than fixing it.

Launch the separate trace on the same physical-core mask as the accepted
benchmark. An outer `taskset` with `mpirun --bind-to none` avoids conflicting
OpenMPI `--cpu-set`, socket mapping and `--bind-to core` policies. The mask and
thread budget must come from the actual selected endpoint's physical NUMA
cores, not a hard-coded socket number. For one execution rank:

```bash
# Set these to the exact saved plan, prompt, binary and accepted CPU affinity.
LLAMINAR_PROFILE_BIN=/absolute/path/build_v2_release/llaminar2
LLAMINAR_PROFILE_PLAN=/absolute/path/plan.json
LLAMINAR_PROFILE_PROMPT=/absolute/path/prompt.txt
LLAMINAR_PROFILE_RESULT=/absolute/path/separate-trace
LLAMINAR_PROFILE_CORES='<physical-core-list>'
LLAMINAR_PROFILE_THREADS='<matching-physical-core-count>'

timeout --signal=TERM --kill-after=15s 180s \
env LLAMINAR_PROFILER_NORMAL_EXIT=1 \
  LLAMINAR_BENCHMARK_ITERATIONS=1 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=0 \
  OMP_NUM_THREADS="$LLAMINAR_PROFILE_THREADS" OMP_PLACES=cores OMP_PROC_BIND=close \
  taskset -c "$LLAMINAR_PROFILE_CORES" \
  mpirun -np 1 --bind-to none \
    --mca mpi_leave_pinned 1 --mca btl_vader_single_copy_mechanism none \
  rocprofv3 --kernel-trace --stats --output-directory "$LLAMINAR_PROFILE_RESULT" \
  -- "$LLAMINAR_PROFILE_BIN" benchmark \
    --config "$LLAMINAR_PROFILE_PLAN" --no-mpi-bootstrap \
    --prompt-file "$LLAMINAR_PROFILE_PROMPT" -n 128 \
    --temperature 0 --seed 42 --mtp --mtp-depth-policy dynamic \
    --benchmark-json-output "$LLAMINAR_PROFILE_RESULT.json"
```

Use `--mtp-depth-policy fixed --mtp-draft-tokens 3` for the corresponding
fixed control. These are explicit diagnostic comparisons, not instructions to
replace production defaults. Adapt output length and watchdog to the bounded
workload; do not accept a timed-out or incomplete profiler as evidence. The
normal-exit flag allows trace flushing, not a different inference algorithm.
Do not remap GPU visibility after applying a plan with physical ordinals.
Bracket the full process lifetime with the canonical driver-diagnostics helper.

The v3 SQLite database's `kernels` view contains dispatch timestamps,
duration, geometry, queue/stream identity, registers and static scratch.
Inspect its schema rather than assuming a `kernel_dispatches` table. Durations
in `kernels` are nanoseconds; derived `top_kernels` reports may use microseconds.
The `regions` view can be empty when only kernel tracing was requested.

Exclude setup and warmup by authenticating an executed request window. HIP
transaction anchors are the prepare-budget and publish-ticket kernels:
`rocm_prepare_device_generation_transaction_budget_kernel` and
`rocm_publish_device_generation_dispatch_ticket_kernel`. Match their ordered
counts to the benchmark JSON's verifier/transaction counts; do not select a
fixed number of transactions copied from another model. Restrict per-kernel
aggregation to the chosen measured request and actual GPU participant.

Report dispatched-kernel count as well as useful arithmetic. A device-count
guard can suppress every load in an inactive branch yet still incur a kernel
dispatch cost. Compare native launch grids and scratch strides to separate
live-row work from retained allocation capacity. Summed durations are service
attribution, not wall time or an overlap-aware critical path. Profiler timings
never replace the unprofiled counterbalanced throughput certificate.

## Candidate-only counters

Counter evidence belongs to one exact kernel/ISA/shape/M candidate. First read
the timing CSV and identify the candidate's dispatch range. On the rocprof v1
shipped with ROCm 7.1.1, `range: 1 : 3` selects dispatches 1 and 2. Confirm the
semantics against the emitted CSV whenever the profiler version changes.

Create a counter request such as `/tmp/rocm-counters.txt`:

```text
pmc : Wavefronts VALUUtilization VALUBusy SALUBusy LDSInsts LDSBankConflict ALUStalledByLDS
range: 1 : 3
pmc : FetchSize MemUnitBusy MemUnitStalled
range: 1 : 3
pmc : WriteSize
range: 1 : 3
pmc : L2CacheHit
range: 1 : 3
```

Then run only the deterministic isolated harness under that request:

```bash
COUNTERS=/tmp/rocm-counters.txt
RESULT=/tmp/llaminar-rocm-kernel-counters
mkdir -p "$RESULT"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
rocprof -i "$COUNTERS" --timestamp on --basenames off \
  -d "$RESULT" -o "$RESULT/counters.csv" \
  ./build_v2_release/tests/v2/<isolated-perf-binary> \
  --gtest_filter='Perf__ExactSuite.ExactProductionShape'
```

MI50 cannot collect every derived memory metric in one hardware pass. If
rocprof reports `Input metrics out of HW limit`, retain its valid grouping or
split the metrics into separate `pmc` records. rocprof reruns the executable for
each record, so inputs and dispatch ordering must remain deterministic.

For `rocprofv3`, use separate invocations for these memory groups as well:
`--pmc FetchSize MemUnitBusy MemUnitStalled`, then `--pmc WriteSize`.
Combining all four on gfx906 can fail with counter-config error 38 and leave
the profiler's signal handler waiting on its own queue. Put diagnostic launches
under a bounded process timeout; confirm the owned MPI/profiler process tree
has exited before any benchmark. Terminate only that diagnostic if necessary.
This is failed instrumentation, not a successful zero-traffic sample. Never
reuse an unprofiled timing sample that overlaps the lingering helper.

Resolve the profiler from the SDK used to compile the binary, not an unrelated
`rocprofv3` earlier on `PATH`. Check `${ROCM_PATH}/bin/rocprofv3 --version` and
retain its SDK/source identity. For an isolated SDK upgrade, keep the same
explicit HIP/RCCL library selection as the unprofiled process and include that
SDK's `lib` and `lib/llvm/lib` in its loader path. Inspect the actual dependency
closure: a matching SONAME alone does not prove a matching SDK, and a missing
host dependency can silently resolve from an older system installation.

Audit the compile driver as well as the runtime loader. A relocated Clang can
find the new device bitcode relative to its executable while still searching
`/opt/rocm/include` for HIP headers. CMake can suppress the intended include
directory as supposedly implicit. Use the selected `ROCM_PATH` for both
`--hip-path` and `--rocm-path` on HIP compile/link commands; pass those options
to standalone repro builds too. Inspect `clang++ -###` to corroborate the actual
header and bitcode roots. The canonical `ROCmSDKCompilerBinding` Unit/preflight
proof checks all HIP commands, including HIP-compiled `.cpp` bridges. A compiler
version or matching runtime SONAME alone cannot certify this closure.

### Authenticate retained HIP graph dispatches

When the selected SDK's `rocprofv3 --help` advertises `--hip-graph-trace`, use
its JSON graph records to distinguish captured replay from eager preparation
and serial diagnostic oracles. The graph records are not exported to CSV;
request JSON even if kernel/API CSVs are also useful. For one exact candidate,
with its geometry/format selectors already set:

```bash
"${ROCM_PATH}/bin/rocprofv3" \
  --hip-runtime-trace --hip-graph-trace --kernel-trace --memory-copy-trace \
  --output-format json csv --output-directory "$LLAMINAR_PROFILE_RESULT" \
  -- "$LLAMINAR_PROBE_BINARY" --gtest_filter="$LLAMINAR_EXACT_PROBE"
```

In `rocprofiler-sdk-tool[0].buffer_records`, join each `hip_graph` record to
`kernel_dispatch` records by `correlation_id.internal`, and require the kernel
records' `graph_exec_id` to match the graph's `graph_exec_id.handle`. The number
of joined dispatches must equal `kernel_dispatch_count` for every launch.
Resolve each dispatch's `dispatch_info.kernel_id` through `kernel_symbols`;
preserve its launch geometry, resource counts and `code_object_id`. Join that
last field to `code_objects` to identify the actual loaded device image.
Graph IDs and kernel IDs are process-local: never join two runs by those IDs.

Check graph-associated `memory_copy` records separately. Eager weight loading,
output readback and the serial oracle are outside the captured interval and
must not inflate the inference transfer or kernel inventory. Do not classify
capture from a timestamp window or kernel name alone when explicit graph
association is available. Likewise, a `hipGraphLaunch` API duration is host
submission time, not the graph's GPU critical path. Use the joined kernel
intervals for attribution and separate unprofiled event samples for economics.
Include inactive retained branches: an early-exiting kernel still costs a
dispatch, even though it performs no expert dot products.

Authenticate the **production entrypoint**, not only the shape and projection
specialization. A component probe can call generic grouping while the model's
grouped verifier publishes a runtime-owned plan and deferred accepted-row route
ledger. Those planners can have different compact-group boundaries and launch
inventories even when their inputs and output bytes agree. Trace the graph's
typed bindings into the stage's selected grouping/pipeline call and confirm the
same entry families in the actual server trace before proposing a server fix.
Keep each source/SDK/binary cohort separate; an older whole-model trace may
identify a structural lead, but cannot certify a change in the current runtime.

If an older trace lacks native graph records, reconstruct complete device
transaction boundaries from its budget publication through its terminal
dispatch ticket. Authenticate ordered prepare/commit/ticket markers for every
transaction and match the benchmark's declared warmup/measurement cohorts.
Exclude warmup, prefix prefill, output readback and teardown by those boundaries,
not by counting the last arbitrary number of kernels. Preserve the trace's
limitations; this is not equivalent to explicit graph-executable association.

Counter interception can report that a device-memory queue ring was replaced
with a system-memory profiling ring, without preserving priority or the CU
mask. This is a profiler-only change, not evidence that production changed its
queue policy. Keep the warning with the counter evidence and never use this
process's latency as the unprofiled throughput certificate. Trace-only and
counter-enabled runs are separate observations; neither substitutes for
canonical timing. Also distinguish the compiler's logical VGPR count from the
counter CSV's hardware-rounded allocation count; small allocation-granularity
differences are not spills. `Scratch_Size` and final code-object private storage
remain separate, required evidence.

The v3 selector `--kernel-include-regex '<exact-family>'
--kernel-iteration-range 2` selects one captured replay after the eager warmup
in the public runtime-grouping harness. With `--output-format csv`, require
all counter rows to name that family and one dispatch ID. The iteration
selector bounds **counter collection**, not the trace: `--kernel-trace` may
still emit all replay dispatches. Verify both inventories instead of assuming
the output contains only the selected launch.

Build the regex from the timing CSV's actual `Kernel_Name`. Rocprofv3 normally
reports demangled C++ template names, not the mangled symbols printed by LLVM;
a mangled-only regex can finish successfully with no counter file. Require a
nonempty `*counter_collection.csv`, exactly one kernel identity and exactly one
dispatch ID before accepting the evidence. For retained adaptive graphs, select
the specialization doing live work: a simultaneously captured alternative can
legitimately exit immediately. Template family alone does not identify the
active row tile. Check its runtime duration and geometry in the timing trace.

Interpret related metrics together:

- `DurationNs` is dispatch latency; use it to attribute an unprofiled timing
  regression or win.
- `scr` and `.private_segment_fixed_size` expose private scratch. Both should be
  zero unless a measured exception is justified.
- `arch_vgpr`, `sgpr`, `lds`, and `Wavefronts` constrain occupancy and reveal
  whether the grid supplies enough waves for all CUs.
- `VALUUtilization` estimates active lanes in issued vector instructions. Very
  low utilization often indicates divergence or lane-zero serial work.
- `VALUBusy` and `SALUBusy` distinguish vector and scalar pipeline pressure.
- Interpret `LDSBankConflict` with `ALUStalledByLDS`. Bank conflicts are not the
  limiting factor when LDS stall is near zero and a conflict-reducing A/B
  variant is slower.
- `FetchSize`, `WriteSize`, `MemUnitBusy`, `MemUnitStalled`, and `L2CacheHit`
  distinguish bandwidth/cache pressure from comparison, reduction, and launch
  latency. Normalize traffic against logical bytes for the invocation.

## Captured dense activation quantizer

`v2_perf_blockwise_quant_kernel` calls the installed public sum/no-sum producer,
with explicit streams and persistent storage. Its shape sweep covers decode,
grouped verification, main prefill and tail widths; the profiler entry runs
only one selected shape. Both check exact bytes, scale words and optional INT32
sums. They are performance diagnostics, not model certificates or preflight
entries. The separate `V2_Integration_ROCmActivationQuantization` preflight
test changes inputs over retained replays and checks unaligned views and guards.

```bash
cmake --build build_v2_release --parallel --target v2_perf_blockwise_quant_kernel
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
  ./build_v2_release/tests/v2/v2_perf_blockwise_quant_kernel \
  --gtest_filter=BlockwiseQuantPerfTest.CapturedProductionShapeSweep

RESULT="$(mktemp -d /tmp/llaminar-quantizer-profile.XXXXXX)"
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
LLAMINAR_QUANT_PROFILE_M=512 LLAMINAR_QUANT_PROFILE_K=2048 \
LLAMINAR_QUANT_PROFILE_SUMS=0 \
timeout --signal=TERM --kill-after=5s 120s \
rocprofv3 --kernel-trace --stats --output-format csv \
  --kernel-include-regex 'quantizeActivationsQ8_register_blocks_kernel' \
  --kernel-iteration-range 17 \
  --pmc Wavefronts VALUUtilization VALUBusy SALUBusy LDSInsts LDSBankConflict \
  --output-directory "$RESULT" \
  -- ./build_v2_release/tests/v2/v2_perf_blockwise_quant_kernel \
  --gtest_filter=BlockwiseQuantPerfTest.ExactProductionShapeProfilerLaunch
```

Choose the available physical device before running, and bracket accelerator
work with the testing skill's driver diagnostics. The harness captures sixteen
producer nodes; dispatch 17 is the first node of the second warm replay. Verify
that the counter CSV contains exactly that dispatch and the requested variant.
Repeat separately with `LLAMINAR_QUANT_PROFILE_SUMS=1`, and with `K=33` for the
scalar-tail variant. Run fetch and write counter groups in separate launches
as above; never report profiled latency as the unprofiled sweep's median.
Unset `LLAMINAR_ROCM_BLOCKWISE_QUANT_VARIANT` for production auto dispatch.

The dense ROCm scale and sum layout are not interchangeable with CUDA's or the
movable-expert producer's ABI. Compare against the contract of the actual
consumer. Shuffles can appear as LDS instructions despite zero LDS allocation;
that does not establish a shared-memory spill or bank-conflict bottleneck.

## Isolated candidate libraries

An isolated HIP translation unit can shorten iteration by interposing its public
host entrypoint in an existing harness. It is diagnostic evidence only: an
installed production build still needs functional gates and whole-model timing.
Compile with the production flags and mandatory spill guard, and retain the
matching source, object, code object and resource metadata for both candidates.

Give the probe's GPU kernels **and device constants/lookup tables** distinct
symbol names from the installed library; leave only the intended public host
entrypoints interposable. Two copies of a HIP translation unit can otherwise
collide during kernel or device-variable registration. Do not assume that
different shared-library filenames isolate those symbols. First run an unchanged
interposed control through the same all-format numerical oracle. A failing
control invalidates the probe; it is not evidence of a production regression.

Confirm the captured graph names the probe kernels, then collect unprofiled
control/candidate/candidate/control timings in separate processes. Linker binding
diagnostics belong in a separate inspection run, never the timing process.

For fixed-size accumulator arrays, inspect the generated ISA rather than trusting
`#pragma unroll`. A runtime `break` can prevent static row expansion, leaving
indirect VGPR indexing (`s_set_gpr_idx_on` / `v_movrel`) in the inner loop. An
equivalent bounded row guard can expose constant indices without changing the
per-row floating-point order. Prove partial/empty tiles and serial-row byte
equivalence, and recheck SGPR/VGPR pressure, spills, instruction size and all
supported formats: unrolling is a candidate, not an unconditional optimization.
Measure sparse tails separately from full row tiles. Independent guards still
evaluate absent rows; a compile-time-indexed, short-circuiting visitor can retain
constant registers while ending at the first absent row. Validate that emitted
code really removes indirect indexing rather than trusting the source pattern.

## Wave64 vectorization and reductions

When removing idle physical waves, keep the logical reduction partition
separate from the launch width. Changing a K loop from its canonical stride to
the smaller `blockDim.x` can append different products to each accumulator and
break grouped/serial byte equivalence. Admit the smaller geometry only where
the removed owners have no inputs; retain their mathematical zero contribution
and check signed zeros. Test widths immediately below, at and above that
boundary, as well as widths that require multiple canonical K iterations.
Wave32 and wave64 need separate reasoning about the final inter-wave tree.

Keep the numerical oracle's scalar bridge unchanged. Inspect the captured
kernel's block geometry and scratch usage as well as its output: an older,
slower kernel can remain perfectly numerically correct. A focused regression
can therefore assert both exact output bits and the intended physical mapping,
without placing a noisy timing threshold in production preflight. Retained
replays must include empty/partial/full counts and untouched output guards.

Do not diagnose scalar execution solely from one source value per thread or an
ISA `global_load_dword`. VMEM executes across active wave lanes: 64 contiguous
lane addresses make that instruction a coalesced 256-byte wave access. Verify
lane address progression, active-lane utilization, transaction bytes, and
throughput before changing it to per-lane `dwordx4` loads.

For cooperative reductions, confirm HIP shuffles lower to wave operations such
as `ds_bpermute_b32`. Compare against lane-zero loops. An exact Top-K kernel can
give every lane one sorted list, reduce current heads with the canonical total
comparator, broadcast the winning lane, and advance only its cursor. This
changes comparison topology without changing floating-point arithmetic, but it
must still pass the all-M grouped-versus-serial byte sweep.

## ROCm 7 code-object extraction

ROCm 7 HIP objects commonly store a bundle in `.hip_fatbin`. Extract the exact
target from the compiled object with the installed ROCm LLVM tools:

```bash
OBJ=build_v2_release/CMakeFiles/<target>.dir/path/Kernel.hip.o

/opt/rocm/llvm/bin/llvm-objcopy \
  --dump-section .hip_fatbin=/tmp/kernel.hip_fatbin "$OBJ" /dev/null
/opt/rocm/llvm/bin/clang-offload-bundler \
  --list --type=o --input=/tmp/kernel.hip_fatbin
/opt/rocm/llvm/bin/clang-offload-bundler \
  --unbundle --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx906 \
  --input=/tmp/kernel.hip_fatbin \
  --output=/tmp/kernel.gfx906.co

/opt/rocm/llvm/bin/llvm-readelf --notes /tmp/kernel.gfx906.co
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx906 \
  --disassemble-symbols='<mangled-kernel-symbol>' \
  /tmp/kernel.gfx906.co
```

Use the exact target printed by `clang-offload-bundler --list`; do not assume
all builds use `gfx906`. For each retained specialization, record
`.private_segment_fixed_size`, `vgpr_count`, `sgpr_count`, both spill counts,
and wavefront size. Census VMEM, LDS, shuffle/permute, and store instructions in
that exact symbol.

For an RDC library with non-inlined device helpers, entry-kernel metadata is
not the complete spill proof. Request final link-time frame classifications
alongside resource remarks. The isolated RCCL CMake build accepts
`"-DCMAKE_SHARED_LINKER_FLAGS=-Rpass-analysis='stack-frame-layout|kernel-resource-usage'"`
as one shell argument (retain the inner quotes in the stored value), with
`REPORT_KERNEL_RESOURCE_USE=OFF` so its additional remark selector does not
override the combined expression. Inspect every emitted helper's `Type: Spill`
locations as well as generic entrypoints; zero VGPR spills on the entry does
not exclude helper memory spills. Preserve multiple emitted instances of a
name rather than overwriting them in an analyzer. Distinct frame offsets are
static allocation evidence, not dynamic scratch traffic. Authenticate the
reported relink against the frozen timed DSO's embedded device image before
joining the evidence; source names and unchanged register totals are insufficient.

A sealed uniform-function entry can remove device-call save/restore traffic
that survives ordinary inlining into an indirectly called collective. Follow
the shared native-plan admission and mixed-function proof above. Use
`hipKernelNameRefByPtr` with the participant's explicit stream to identify the
retained host-registered kernel after capture. A nonzero private allocation may
come solely from `__assert_fail`; prove this from the linked call targets and
frame/ISA records before distinguishing it from hot-path spilling. Zero entry
VGPR-spill metadata alone still cannot establish that distinction.

A linked vendor DSO can contain several `NT_AMDGPU_METADATA` notes. Use
`llvm-readobj --notes --elf-output-style=JSON` on its exact extracted code
object, iterate every note's `AMDGPU Metadata`, and enforce unique kernel names
across notes. The project guard's single-object admission is not a reason to
ignore the other notes or relax that guard for ordinary project objects.
Join actually captured entry names to this complete inventory, then inspect
each final symbol with `llvm-objdump --disassemble-symbols=...`. Resolve
PC-relative `s_getpc_b64` / `s_add_u32` / `s_addc_u32` / `s_swappc_b64` call
targets against `llvm-nm -n`; multiple linked copies of `__assert_fail` can have
different addresses. Fail classification on unknown call dataflow. Preserve
SGPR register-lane moves separately from memory loads/stores, and distinguish
the executed-entry audit from the entire generated inventory.

When changing inlining or device-function boundaries, also preserve the
caller-to-helper mapping. A disappearing helper is not evidence that all its
storage disappeared: some can move into its caller. Conversely, a larger
caller frame alone does not prove a regression when a large callee frame was
eliminated. Compare complete reachable call paths, retain unmatched functions
explicitly, and inspect the final entrypoint's private allocation and occupancy.
The sum of static frame records is not dynamic traffic or a spill-free
certificate. Keep this resource experiment separate from uncontended timings
and revalidate every affected scalar/operator/protocol before promotion.

For a completed rocprofv3 SQLite trace, the kernel-symbol table links to a
code-object table whose `uri` may identify the exact image loaded from a DSO:
`file:///.../libllaminar2_core.so#offset=<byte-offset>&size=<byte-count>`.
Reading that bounded ELF image avoids guessing which rebuilt object supplied
the profiled kernel. Preserve the matching binary before rebuilding. Comparing
named ELF function bodies is stronger than comparing only register counts;
PC-relative relocations can explain raw differences, while identical bodies
do not prove equal runtime scheduling or data-address placement. A
process-memory URI is unavailable after that process exits: report that gap,
never claim those kernels were also compared.
