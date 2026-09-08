# Paired inference attribution

## Capture comparable execution

First obtain unprofiled Release measurements using [the fixed contract](benchmarks.md).
Then profile both binaries on that workload, with a warmup and a measured
request. Longer decode budgets should use the same context positions; an old
64-token trace is not a replacement for the other engine's 256-token run.

On CUDA, a useful Systems capture is:

```bash
COMPARISON_NSYS_BIN="$(command -v nsys)"
sudo -n env OMP_NUM_THREADS=1 OMP_PROC_BIND=true \
  LLAMINAR_BENCHMARK_ITERATIONS=1 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PROFILER_NORMAL_EXIT=1 \
  "$COMPARISON_NSYS_BIN" profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true --output=/tmp/comparison-llaminar \
  ./build_v2_release/llaminar2 benchmark --no-mpi-bootstrap \
  -m "$COMPARISON_MODEL" -d cuda:0 -c 4096 \
  --prompt-file "$COMPARISON_PROMPT" -n 256 --deterministic \
  --benchmark-json-output /tmp/comparison-llaminar.json
```

Resolve the installed profiler executable first. `--no-mpi-bootstrap` and the
explicit OMP settings above are attachment settings, **not canonical benchmark
settings**. Normal-exit mode allows profiler/interposer receipts to flush;
the production executable may otherwise use `_exit`. Wrap the pinned llama.cpp
server with the same Systems options, submit the same warmup/measured requests,
then stop its owned process cleanly so Nsight exports its report.

Check that actual `CUPTI_ACTIVITY_KIND_KERNEL` records exist. Graph-construction
metadata alone is not execution evidence. Privilege and tool-version support
matter. Some conditional graphs omit activities or reject NCU child-node
profiling. Do not disable production capture to work around that limitation.
Use a graph-preserving supported profiler, or profile the exact production
kernel in a focused captured integration/performance fixture and state its
scope. A fixture is not whole-model performance evidence.

Export the trace read-only for analysis:

```bash
"$COMPARISON_NSYS_BIN" export --type sqlite --force-overwrite=true \
  --output=/tmp/comparison-llaminar.sqlite /tmp/comparison-llaminar.nsys-rep
```

## Establish phase windows from evidence

Inspect SQLite schemas with Python's `sqlite3` if the CLI is unavailable.
Join `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` to `StringIds.id`. Use exact
kernel names **and grid/block geometry** to identify the vocabulary head, then
authenticate the number of requests and emitted tokens against benchmark/HTTP
receipts. Do not hardcode a weight codebook as “the vocabulary head” globally.

The first measured vocabulary head ends prefill. Subsequent heads delimit
after-prefill decode cycles. Associate a retained graph's kernel records through
`correlationId`, but also include overlapping uncorrelated activity: conditional
attention nodes may have correlation zero. Grouping only by the vocabulary
head's correlation can silently drop attention. Prefill must include every
physical batch through its first head, excluding initialization and warmup.

Report both complete decode cycles and exact captured graph intervals. A useful
steady comparison averages the last twenty graphs at the same context positions,
while retaining the all-token view to expose first-token/cache effects. Explain
why each boundary is valid; never select a fast tail and hide the full request.

The supplied analyzer takes explicit windows (all timestamps in Nsight ns):

```json
[{"phase":"prefill","start_ns":100,"end_ns":200},
 {"phase":"decode_graph","start_ns":300,"end_ns":400}]
```

Its ordered rules are JSON objects with `category` and a case-insensitive
`pattern` regex. Put vocabulary-head and ordered-reducer rules **before** broad
matrix rules. Derive the actual rules from each engine's current source/trace.
Invoke separately for each engine:

```bash
python3 .agents/llama-cpp-comparison/scripts/attribute_nsight.py \
  --sqlite /tmp/comparison-llaminar.sqlite \
  --windows /tmp/llaminar-windows.json --rules /tmp/llaminar-rules.json \
  --output-prefix /tmp/llaminar-attribution
```

Unknown kernels remain `UNCLASSIFIED` and cause a nonzero exit *after* artifacts
are written. Review their source and fix classification; do not discard them.
Match semantic phases across engines, not just similarly named functions.
The helper measures one device at a time; use `--device-id` for a multi-device
trace. Multiple samples in one phase must be disjoint. Different phase views
may intentionally overlap, but their totals must not be added together.
Its device-free checks run with:

```bash
python3 .agents/llama-cpp-comparison/scripts/test_attribute_nsight.py
```

## Classify without double counting

Keep at least these distinctions where present:

- Quantized projections, floating projections, and vocabulary head.
- Ordered projection reducers and activation quantization/preparation.
- Attention/RoPE/output gating; GDN/recurrent update/convolution.
- Normalization/elementwise work; embedding; sampling/controller work.
- KV/state/layout work. Generic gathers/copies are not all provably KV traffic.

Fused operations remain indivisible. To compare a combined projection/reduction
pipeline, rerun the interval sweep with both rules assigned the same category;
do not add their unions or exclusive times after the fact.

For each phase, retain raw summed kernel time, category union and category-
exclusive time. Use a timestamp event sweep: simultaneous same-category kernels
count once; simultaneous different categories occupy one explicit overlap row.
Exclusive rows, overlap and no-kernel intervals must sum to the entire window.
A summed 1.5-ms reducer is not a removable 1.5-ms penalty if it overlaps producers.

Inspect `CUPTI_ACTIVITY_KIND_MEMCPY` independently: direction, byte count,
duration, stream and position in the request. Separate tiny control transfers,
full vocabulary logits, cache snapshots and per-layer tensors. A no-kernel
interval may contain DMA; it is not automatically GPU idle or host overhead.
DMA may overlap kernels, so do not add all copy durations as serial penalties.
Zero GPU sampler time in llama.cpp can mean host sampling, not free sampling.

## Deep profiles and decisions

Select one exact kernel, shape and schedule per NCU launch. Nsight's demangled
names can contain `(int)256` and `(bool)1` rather than C++ source spellings.
Confirm a report exists and contains the intended grid; launch-skip counts only
matching launches. Keep occupancy/register/spill, memory-width/traffic, tensor-
core and stall counters separate from unprofiled timing. See the CUDA skill for
current section/metric names and graph-support limitations.

Check both engines' operand and accumulator precision. A half P×V tensor-core
path and ordered FP32 P×V path have different costs and correctness contracts.
Likewise a codebook name alone does not establish the activated GEMM operand
format or reduction order. Inspect actual specialization, shape and source.

If timeline data suggests host overhead, quantify it separately with a bounded
low-overhead CPU API interposer or equivalent tool. Nsight interception can
inflate graph-launch API time drastically. Such instrumentation must not add
CUDA operations, synchronization or unbounded per-token logging; record drops
and errors and flush at terminal exit. Do not subtract unrelated profiled and
unprofiled runs and call the remainder an exact host tax.
