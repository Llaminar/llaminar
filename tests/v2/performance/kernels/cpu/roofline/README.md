# QFN hybrid CPU decode hardware references

Build the explicit Release target on Linux x86. Select one hardware thread per
physical core from the same observed NUMA node and shared-cache domain. Use the
actual worker placements of the model runs for the full-team measurements.

```sh
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_qfn_cpu_decode_roofline
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
PROBE="$CMAKE_BUILD/tests/v2/v2_perf_qfn_cpu_decode_roofline"
for mode in l2 shared dram vnni; do
    taskset -c "$PHYSICAL_CORES" "$PROBE" "$mode" > "node-${NODE}-${mode}.csv"
done
```

Keep nine samples per process and repeat in fresh processes with reversed node
and mode order. To measure socket interference, launch the same mode on both
separately pinned teams at the same time. Keep the probe off the host during
canonical model timing. The probe checks all read data and VNNI results and
verifies the physical placement of every live read page after sampling.

The L2 test reads half of each private L2. The shared-cache test uses seven
eighths of the shared cache across the selected team and requires every worker's
segment to exceed its private L2. This is a working-set reference for the cache
hierarchy. The DRAM test reads four times the combined private and shared cache
footprint. All byte rates count requested cache lines. Collect memory-controller
and cache events in separate profiler passes when inferring physical traffic.

The VNNI test runs 12 independent `vpdpbusd` chains on each worker. A ZMM
instruction performs 64 byte multiply-accumulates. The probe reports vector
instructions per second. For QFN CPU FFN, use the prepared stride from
`preparedFootprintForFormat()`, the admitted shard widths and actual live
routes. Compute separate read and dot times for Gate/Up and Down, take the
maximum per phase and add the ordered phases. Use the shared-cache reference
for qualified cache residency and the DRAM reference for a one-pass stream.
Account for synchronization, routed activation quantization, publication and
cross-device communication in the complete transaction measurements.

## Xeon Gold 6238R reference

Two sockets with 28 workers each, one per physical core, 1 MiB private L2 and
38.5 MiB shared cache per socket. The full-team rows are medians of 27 samples
from three fresh processes per mode. The timed VNNI loop has 12 register-only
ZMM `vpdpbusd` instructions with no vector spills.

| Node | Workload | L2 GB/s | Shared cache GB/s | DRAM GB/s | VNNI Ginst/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Independent | 2822.94 | 578.47 | 104.02 | 45.41 |
| 1 | Independent | 2855.17 | 584.80 | 110.64 | 44.94 |
| 0 | Concurrent sockets | 2977.60 | 572.00 | 102.73 | 45.28 |
| 1 | Concurrent sockets | 2706.08 | 573.50 | 110.00 | 44.62 |

At 7, 14 and 21 workers the independent DRAM medians were 73.24, 103.87 and
107.98 GB/s on node 0 and 76.55, 106.64 and 115.24 GB/s on node 1. Those are
screening runs with one process per width. This host approaches its saturated
streaming rate before using all 28 physical cores. Any proposed worker count
still needs to qualify the real projection imbalance and hybrid model timing.

## Hybrid target and starting point

The arithmetic and memory targets are 80% of the matching CPU and
[CUDA hardware references](../../cuda/roofline/README.md). Compare end-to-end
Release ordinary decode and fixed-depth seven MTP with warmup and both arm
orders. Require exact prompt and generated token histories, exact MTP work and
no repeatable model regression. Record grouped verifier byte equivalence and
test production capture/collectives independently.

The measured 2×CUDA/2×CPU tensor-sharded plan has H2560/F320 on the remote CPU
participant. Its real M1 roster is Q4_K/Q5_1 for 43 layers, Q4_K/Q8_0 for four,
and Q5_K/Q8_0 for one. Accounting for each original routed expert's two
Gate/Up matrices and one Down matrix gives 2,099,200, 2,048,000, and 2,867,200
prepared weight bytes per live route, respectively. There is no shared-expert
projection on this CPU endpoint. In one matched ordinary decode diagnostic,
1,488 local calls processed 4–10 routes each. Their mean request was 16.59 MB
of prepared weights. A single ideal DRAM pass on node 1 takes about 151 µs;
the 80% target is about 189 µs for that workload. The useful VNNI phase times
are smaller than the corresponding read times.

| CUDA parent | CPU expert execution, µs per call | Effective prepared-weight GB/s | Fraction of one-pass DRAM reference |
| --- | ---: | ---: | ---: |
| Staged | 305.81 | 54.2 | 49.3% |
| Resident with independent packet readiness | 317.55 | 52.2 | 47.5% |

These CPU times include local expert execution and its team/activation work.
The rates assume one requested read per prepared byte. Establish physical
traffic and cache residency with separate counter measurements. The sampled decode
window on this CPU rank spent 74.18% of user cycles at identified `libgomp`
wait-loop addresses, 12.22% in the fused projection dispatcher and 11.20%
in the AVX-512 non-nibble GEMV entry. Attribute critical-path time with separate
elapsed measurements of the CPU leaves and actual thread scaling alongside the
CUDA dispatch timeline.
