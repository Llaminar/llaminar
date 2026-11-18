# Measuring the grouped verifier roofline

Start with a byte-exact captured verifier and its serial decode control. Record
the physical GPU, clocks, source weight formats, TP shard widths, live rows,
expert IDs, the number of distinct experts, and the captured kernel symbols.
Retain the actual graph and resource lifetimes when changing a kernel. Measure
the parent with device events in an unprofiled Release run. Use a separate
timestamp-only trace to attribute time to leaves. Isolate hardware counters
by symbol and dispatch geometry in separate runs of the same captured work.

Resolve the original gate, up and down reduction schedules during graph
preparation, separately for routed and shared weights. Verify shared source
planes during preparation and incoming expert descriptors before publishing
an inactive bank. Captured replay still acquires the current bank generation
and reads the device-owned live row count, routes and weights. Use a changed
bank and a partial verifier replay to check this distinction. Compare compiled
ISA for the recorded kernel symbol to verify that per-tile descriptor checks
actually disappeared. Time the unprofiled parent in both run orders.

Measure memory service on the same card with the native weights' access
pattern, extent, route ordering and launch geometry. Include both the
working set of an ordinary warm request and a working set larger than L2.
Use the same counter on the probe and on the native kernel. For example,
use ROCProfiler `FetchSize` in KiB for physical reads on gfx906. Application
bytes requested in a cache-resident loop can exceed physical memory throughput
many times over. Record writes, cache hits, VALU issue and occupancy as
additional evidence for a proposed bottleneck.

The [ROCm probe instructions](../rocm/roofline/README.md) give the native
measurement commands for MI50. Run the analyzer on one evidence document
and retain the resource-level CSV.

```sh
python3 tests/v2/performance/kernels/qwen4exp/verifier_roofline.py \
    verifier-evidence.json --csv verifier-resources.csv
```

An evidence document contains a `references` list and a `leaves` list.
Each reference has an `id`, `device`, `isa`, `clock_policy`, `resource`,
`unit`, `counter`, `regime` and repeated `samples` containing `count` and
`us`. A leaf names the same device, ISA and clock policy, along with its
`case`, `role`, `symbol`, `variant`, separate `timing_us` samples and
`work` records. Each work record names the `reference`, the same resource,
unit, counter and regime, and the measured or explicitly derived `count`
per native dispatch. Put operand shape, format and row count in the case
identity when they vary. Retain the raw traces beside the evidence.

The resource time is work per native dispatch divided by the median
independently measured service rate. Memory and arithmetic can overlap, so the
largest resource time determines the reported fraction. Report the underlying
rates, sample counts and interquartile ranges. Fractions above one call for
checking the calibration and accounting. A standalone dot-product issue test
measures a different instruction mix from a real decoder. Likewise a
routed-only read omits shared weights and final publication. Name those
references explicitly and use them to locate further work.

To qualify performance, repeat the best leaf in alternating control and
candidate process order on the same idle card. Confirm the expected graph
symbols, per-row bytes and unprofiled parent time. Then measure verifier
round time and produced tokens per second on the real model with matched
prompts and acceptance histories. A leaf belongs near a practical roofline
only when the identified resource explains its measured time within the
observed variation. A whole-verifier claim also requires accounting for the
other leaves, collective waits and graph boundaries on its critical path.
