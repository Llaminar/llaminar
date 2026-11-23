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

A separate DRAM probe run sampled all six memory-controller channels per
socket. Over three complete 100 ms windows after first touch, the active
node read 96.7–98.9 GB/s on node 0 and 95.0–96.7 GB/s on node 1. Each counter
ran for the full window. The inactive peer read at most 0.30 GB/s. These
figures describe physical traffic in the profiled run. The quiet medians in
the table define the requested-byte reference.

A Release model screen used the same 2×CUDA/2×CPU staged binary and requested
21 workers per socket. The live ranks showed 21 physical worker placements
each. With one warmup and three 256-token requests per cell, the screen had
identical token histories and MTP work to the nearby 28-worker control.
Ordinary and fixed-depth-seven decode fell 2.24% and 2.26%. Prefill fell 4.51%
and 4.35%.

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
1,488 local calls processed 4–10 routes each. Their mean prepared weight
extent was 16.59 MB. The M1 Q4_K identity-nibble dot skips the 128-byte
compensation plane in each of its Gate and Up N64/K32 units. The
compulsory operand count is therefore 15.80 MB per call. A single ideal DRAM
pass on node 1 takes about 144 µs and the 80% target is about 180 µs. The
useful VNNI phase times are smaller than the corresponding read times.

| CUDA parent | CPU expert execution, µs per call | Compulsory operand GB/s | Fraction of one-pass DRAM reference |
| --- | ---: | ---: | ---: |
| Staged | 305.81 | 51.7 | 47.0% |
| Resident with independent packet readiness | 317.55 | 49.8 | 45.2% |

These CPU times include local expert execution and its team/activation work.
The rates assume one requested read per compulsory operand byte. Establish
physical traffic and cache residency with separate counter measurements. The
sampled decode window on this CPU rank spent 74.18% of user cycles at identified
`libgomp` wait-loop addresses, 12.22% in the fused projection workshare body
and 11.20% in the AVX-512 non-nibble GEMV entry. The most sampled instructions
in those two projection symbols were AVX-512 arithmetic and weight operands.
Attribute critical-path time with separate elapsed measurements of the CPU
leaves and actual thread scaling alongside the CUDA dispatch timeline.

## AVX-512 unsigned Q5 M1

The Q5_1/Q5_K single-row path uses canonical unsigned prepared weights and the
original signed Q8 activations as the VNNI operands. The dot skips activation
rebiasing and the 128-byte compensation plane per N64/K32 unit. The kernel
uses the original FP16 scale and minimum values, FP32 contribution policy
and K reduction order. The linked Release code has 32 `vpdpbusd` instructions
per K block in each bank path and no vector stack accesses. The existing
AVX2 and AVX-512 unsigned-weight preflight cases passed, including the
independent original biased-dot oracle for serial and fused one-row Q5.

The Release comparison fixed the CUDA parent to its staged path and used
the same 2×CUDA/2×CPU F320/S320 plan in both arms. Each cell used one warmup
and three measured 256-token requests. Every generated history and all MTP
work matched, including `[1197, 594, 132, 171, 1368]` draft steps, accepted
tokens, rejected tokens, verifier runs and verifier tokens in each fixed-depth
arm. Driver diagnostics were unchanged during the comparison.

| Arm order | Cell | Original decode tokens/s | Unsigned Q5 decode tokens/s | Decode change | Combined prefill and decode latency change |
| --- | --- | ---: | ---: | ---: | ---: |
| Candidate first | Ordinary | 29.600 | 29.829 | +0.77% | −0.81% |
| Staged control first | Ordinary | 29.570 | 30.443 | +2.95% | −2.60% |
| Candidate first | Fixed depth seven | 43.750 | 44.174 | +0.97% | −0.74% |
| Staged control first | Fixed depth seven | 43.546 | 45.048 | +3.45% | −2.96% |

Median prefill rate improved by 0.30–1.94% across these cells. The revised
compulsory footprint is 1,945,600 bytes per route for Q4_K/Q5_1 and
Q4_K/Q8_0, and 2,764,800 for Q5_K/Q8_0.

A separate 32-token service trace matched all 1,488 remote CPU calls by layer
and live route count. Local expert execution fell from 305.81 to 294.15 µs
per call, with a decrease across all 48 layers. The new kernel requests
15.43 MB per measured call and delivers 52.44 GB/s of compulsory operands,
47.7% of the 110 GB/s reference. In the same trace, the remote joint
publication interval rose from 40.33 to 65.68 µs, while the GPU rank's
rooted gather interval fell from 1.716 to 1.479 ms per token. These intervals
include peer arrival and reflect a separate instrumented request cohort.

The native rooted gather screen compared the actual MPI matched-envelope
gather and native shared-memory gather with the same admitted NUMA record
banks. At 2,561 floats per record and 4–10 live records per socket, MPI had
the lower root median in both transport orders, with ready peers and with
15 or 30 µs of root delay. In the ready-peer trials at four live routes,
MPI took 11.60–14.14 µs and shared memory took 18.57–21.03 µs. At ten live
routes the corresponding ranges were 24.94–29.54 and 41.18–46.69 µs. Both
transports matched every output byte and live extent, including unequal and
empty peers. The MPI path is the control for further overlap work.

### Compact five-bit CPU operands

The experimental AVX-512 reader stores the four-bit payload and a separate
plane containing the high bits. It reads 1,536 bytes per N64/K32 tile,
including the original scales and minima. The expanded unsigned reader
requests 2,304 bytes for the same arithmetic. Both paths preserve the
ordered K partitions and produce identical FP32 bytes.

The fixture alternated both readers over 48 distinct weight frames of
eight routes, using 28 pinned physical workers per socket. The compact
weight set totals 236 MB in each geometry. The measured interval includes
the workshare completion edge between frames. A single-frame comparison
with cache reuse took 29–39% longer with the initial compact reader.

The register-only revision retains all 32 `vpdpbusd` instructions per
K32 iteration and removes vector stack accesses. It passed the exact
comparison on both sockets in both orders. The reference fraction uses
one read of the compact operands and the independently measured NUMA
DRAM bandwidth for that socket.

| Source and projection | NUMA node | Compact latency, µs | Latency reduction | Requested-byte reference fraction |
| --- | ---: | ---: | ---: | ---: |
| Q5_1 Down N2560/K320 | 0 | 50.28–51.49 | 35.1–37.3% | 91.8–94.0% |
| Q5_1 Down N2560/K320 | 1 | 52.33–53.76 | 36.6–39.1% | 82.6–84.9% |
| Q5_K Gate/Up N320/K2560 | 0 | 54.88–55.77 | 31.4–33.1% | 84.7–86.1% |
| Q5_K Gate/Up N320/K2560 | 1 | 57.99–58.99 | 31.8–33.7% | 75.3–76.6% |

The two-row program reuses each reconstructed weight vector for both
original activation rows. Its ordered products and outputs matched the
expanded row pair on both sockets. Across the eight 48-frame cells the
compact pair reduced latency by 32.9–39.5%. Seven of the eight single-frame
cells were slower. The inspected AVX-512 entries have no vector stack
accesses.

A single-row revision folds the original K partitions in registers.
It improved single-frame compact Q5_K by 6.1–8.1% and changed the
48-frame medians by less than 0.5% in either direction.

The measurements cover the original five-bit projection arithmetic with
experimental packing. The complete hybrid model uses the expanded format.

## NativeQuantised source matrix

Use the Release `v2_perf_cpu_moe_expert_verifier_rows` fixture pinned to
one thread per physical core of the selected socket. Its production
GPU-aligned path compares Gate/Up, SwiGLU Q8 publication and Down against
independent serial rows both before and after its samples. Set
`LLAMINAR_PERF_STATS_SUMMARY=1` and `LLAMINAR_PERF_STATS_FILTER=kernel` for
the route and ownership proof. The fixture suppresses these records inside
the timed regions and collects a fresh proof afterward.

```sh
LLAMINAR_ISA_LEVEL=avx512 OMP_NUM_THREADS="$PHYSICAL_WORKERS" \
OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false \
LLAMINAR_PERF_STATS_SUMMARY=1 LLAMINAR_PERF_STATS_FILTER=kernel \
LLAMINAR_CPU_MOE_EXPERT_MEASURE_POLICY=aligned \
LLAMINAR_CPU_MOE_EXPERT_GATE_UP_OWNERSHIP=Auto \
LLAMINAR_CPU_MOE_EXPERT_DOWN_OWNERSHIP=Auto \
LLAMINAR_CPU_MOE_EXPERT_D_MODEL=2560 LLAMINAR_CPU_MOE_EXPERT_INTERMEDIATE=320 \
LLAMINAR_CPU_MOE_EXPERT_ACTIVE_EXPERTS=8 LLAMINAR_CPU_MOE_EXPERT_ROUTE_ROWS=8 \
LLAMINAR_CPU_MOE_EXPERT_PURPOSE=decode LLAMINAR_CPU_MOE_EXPERT_FORMATS=all \
LLAMINAR_CPU_MOE_EXPERT_DOWN_FORMAT=Q8_0 \
LLAMINAR_CPU_MOE_EXPERT_POLICY_CSV="$PWD/gate-up-m1.csv" \
    taskset -c "$PHYSICAL_CORES" "$CMAKE_BUILD/tests/v2/v2_perf_cpu_moe_expert_verifier_rows" \
    --gtest_filter=Perf_CPUMoEExpertVerifierRows.ProductionGeometryByteExactAndEconomical
```

At H=2560, use intermediate width 256 for all 21 original sources in both
roles. At width 320, sweep all Gate/Up sources with Q8_0 Down. A standalone
K=320 Down source admits seven 32-value-block formats. Sweep one, two, four
and eight rows per expert, with eight distinct expert weights and the
admitted ownership policy. Record the effective runtime ISA and the
original labels even when two formats share a prepared codebook.
For multiple rows per expert, set the purpose to `mtp` and the total route
rows to the number of experts times the desired rows per expert.

The AVX-512 sweep passed all 434 source/geometry observations on the two
28-core sockets, including the 64-expert, width-256 M=1 cells.

The first single-row pass on both 28-core sockets measured these Gate/Up
intervals with Q8_0 Down and eight live experts. Each row contains the
range of socket medians.

| Original Gate/Up source | Gate/Up, µs | Whole persistent FFN, µs |
| --- | ---: | ---: |
| Q4_K | 30.75–31.35 | 63.43–69.07 |
| Q5_K | 31.53–32.01 | 70.30–77.37 |
| IQ3_S | 31.25–31.65 | 64.03–64.50 |
| Q3_K | 80.19–80.86 | 122.28–125.01 |
| IQ2_XS | 98.29–109.05 | 134.04–134.96 |
| IQ2_S | 111.90–112.60 | 146.85–148.33 |
| IQ1_M | 119.22–120.12 | 155.65–156.50 |

The fixed expert set is reused across samples. Report these as prepared
projection and FFN latencies. A comparison run with 64 distinct experts
has a larger working set and still reuses its weights across samples.
Quantify its physical reads with the memory-controller counters before
interpreting the ratio of prepared bytes to latency as DRAM efficiency.

### Ordered AVX-512 five-bit pairs

At Gate/Up N=320 and K=2560, the two-row GPU-aligned tile uses the same
unsigned Q5 weights and original signed Q8 activations as the single-row
path. The tile preserves all 16 FP32 partitions and each partial's five
K32 contributions. It skips the prepared compensation plane for Q5_1
and Q5_K. Other formats retain their original signed operand program.

The isolated projection test alternated arms on each physical socket and
checked full and partial K trees against the original pair. The two-row
leaf was 15.6–19.1% faster with eight distinct experts and 4.5–9.3% faster
with 64. The production unsigned-weight lifecycle gate passed under both
runtime ISA selections. It exercises the actual output-tile pair at
N=320/K=2560 and checks both banks, dirty output tails, and serial bytes.
The linked unsigned AVX-512 entry contains 64 static `vpdpbusd`
instructions and no vector stack operands.

The 2×CUDA/2×CPU Release comparison combined the ordered CPU pairs and the
[packed SM86 decode readers](../../cuda/roofline/README.md#route-matched-decode-weight-reference).

## Gate/Up ownership and planning experiments

A 28-worker AVX-512 experiment on both sockets compared the original staged
Gate/Up and SwiGLU publication against assigning the same worker the two
projections and its output Q8 packet. All 84 observations across the 21
source formats matched the original Q8 bytes. The 64-column owner reduced
the eight-expert workshare from 80 projection tasks to 40 fused tasks.
Median latency rose by 16.8% at eight experts and 0.8% at 64 experts.
A 32-column owner restored 80 jobs by splitting the two existing 16-column
SIMD segments per original weight panel. All 20 observations for Q4_0,
Q4_1, Q4_K, IQ4_NL and IQ4_XS also matched. Latency rose by a median 10.9%
at eight experts and 13.3% at 64 experts. Production uses the staged
ownership measured in these comparisons.

A separate plan-only probe checked the same live NativeQuantised schedule
for ten representative sources at M=1 and M=4 on both sockets. A single
planner inside the retained team added a publication barrier and took
3.7–22.9% longer than the current per-worker plan. Planning once before
starting the team and adding the measured pre-admitted worker entry cost
reduced the estimated planning phase by 17.2–39.0% across the 40 cells.
The native plan checks cover the actual worker count, prepared codebook,
K partitioning and output ownership.

The complete-transaction probe includes both plans, each team completion
edge and the shared SwiGLU publication. It passed the serial-byte oracle
before and after timing in all 672 observations. Each width and row count
used eight experts, 21 original Gate/Up sources, and both arm orders per
socket. Down used the same source at width 256 and Q8_0 at width 320.

| Expert width | Rows per expert | Median latency change, socket 0 | Median latency change, socket 1 |
| --- | ---: | ---: | ---: |
| 256 | 1 | −5.24% | +1.81% |
| 256 | 4 | +0.18% | +0.09% |
| 320 | 1 | −1.45% | −5.30% |
| 320 | 4 | −0.30% | −1.09% |

The width-320, one-row Q4_K Gate/Up and Q8_0 Down case improved 7.8–16.1%
across sockets and arm orders. Other sources and work shapes gave mixed
results in that comparison.

The follow-up alternated the complete original and prepared transactions
inside the same process. Both arms reused the same prepared weights,
activations, worker team and partial bank. All 48 runs checked the serial
bytes before and after sampling. Across the three original QFN source pairs
and four, six, eight and ten live experts on both sockets, all 24 cells
improved in both orders. The median paired improvement was 8.35%, with
order medians from 5.25% to 14.69%.

At width 320 with Q8_0 Down, 20 of the 21 Gate/Up sources improved in
each order on both sockets. Across all sources, one-row medians improved
5.84–6.32%. The repeated orders added 0.79% and 0.93% to IQ1_M. The four-row
medians changed by less than 0.6%. At width 256, one-row results varied by
source and four-row results were essentially flat.

The Release candidate for owner-side planning uses AVX-512 VNNI,
H=2560, F=320, a standalone fixed worker team, and one live decode row per
expert. The candidate's focused preflight case compares all 21 original Gate/Up
sources and both Q5_1 and Q8_0 Down sources with the serial path. It checks
Gate, Up, the SwiGLU Q8 publication, and Down, including untouched tails
after the live expert count shrinks. It also verifies admission with
1, 3, and 7 workers. IQ1_M retains the per-worker schedule based on the
repeated-order measurements.

The rerun used the staged CUDA graph, identical F320/S320 sharding, and
three measured 256-token requests per mode and arm after warmup. Every
prompt, token history, and per-request MTP work count matched. The complete
column measures prefill and decode time together. The opposite arm orders
gave opposite throughput changes, so owner-side planning remains experimental.
The rank-by-rank dispatch trace counted grouped prefill transactions. The
one-token decode service enters `MoEExpertComputeStage::executeSingleToken`
and consumes its own prepared Gate/Up and Down paths. The paired figures
therefore measure the effect of the linked refactor on the complete model.

| Arm order | Mode | Decode, control → candidate tok/s | Decode change | Complete throughput change |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 29.412 → 29.618 | +0.70% | +0.52% |
| Candidate first | Fixed depth seven | 43.409 → 43.673 | +0.61% | +0.43% |
| Control first | Ordinary | 29.988 → 29.566 | −1.41% | −1.32% |
| Control first | Fixed depth seven | 44.137 → 43.430 | −1.60% | −1.50% |

A follow-up binds the existing SIMD tile entries on the actual one-token
service path. A single team projects Gate/Up, publishes the GPU-aligned
SwiGLU Q8 blocks and finishes Down into the original route rows. The
32-token Release diagnostic matched the control and recorded 1,488 such
transactions in its measured decode interval. The paired Release runs
retained the original F320/S320 sharding and used three measured 256-token
requests per mode after warmup. Prompts, generated tokens, MTP work and
driver diagnostics matched in both orders. The expanded focused contract
matched all 21 Gate/Up source formats and all seven source-legal K320 Down
formats on AVX-512 and AVX2, including smaller live route prefixes in
reused storage.

| Arm order | Mode | Decode, control → candidate tok/s | Decode change | Complete throughput change |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 29.716 → 31.267 | +5.22% | +3.82% |
| Control first | Ordinary | 29.946 → 31.149 | +4.02% | +3.05% |
| Candidate first | Fixed depth seven | 44.026 → 44.148 | +0.28% | +0.12% |
| Control first | Fixed depth seven | 44.119 → 43.777 | −0.78% | −0.79% |

The revision that checks ISA, worker-team eligibility and native Down
ownership before binding scratch passed the same focused format checks.
One additional matched Release pair measured 30.001 → 31.255 tok/s for
ordinary decode and 44.152 → 43.926 tok/s at fixed depth seven. Complete
throughput changed by +3.05% and −0.74%, respectively. All generated
tokens and MTP work matched.

The corresponding CPU service trace includes one warmup and one measured
32-token request on each arm. Rank 1 reported 2,976 local calls in each
arm with the same layer and live route counts. The candidate authenticated
two prepared phase plans for each call. Local execution decreased from
331.95 to 277.73 µs per call. Counting 15.426 MB of compulsory operands
per call gives 46.47 → 55.54 GB/s, or 42.2% → 50.5% of the measured
110 GB/s simultaneous socket reference. The dispatch wait was practically
flat, and joint publication decreased from 57.40 to 51.51 µs per call.

A separate fixed-depth-seven trace matched the first 32 staged tokens.
Rank 1 observed 182 prepared calls alongside 182 one-row drafter calls and
1,248 eight-row verifier calls across warmup and measurement. This identifies
the drafter as the fixed-depth workload that exercised the specialization.

The next candidate binds the prepared schedule only during ordinary decode.
The 32-token service probe observed 2,976 prepared ordinary calls on rank 1
and the original schedule on all one-row drafter and eight-row verifier
calls. The complete 256-token comparisons preserved every generated token
and the per-request MTP counts.

| Arm order | Mode | Decode, control → candidate tok/s | Decode change | Complete throughput change |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 30.916 → 31.213 | +0.96% | +0.80% |
| Control first | Ordinary | 29.580 → 31.166 | +5.37% | +4.25% |
| Candidate first | Fixed depth seven | 45.981 → 43.875 | −4.58% | −3.36% |
| Control first | Fixed depth seven | 43.759 → 43.378 | −0.87% | −0.65% |

The unchanged fixed-depth control varied by 5.08% between arm orders.
Five of 36 matched grouped CPU symbols changed size in the ordinary-only
candidate. The revision that compiles the standalone implementation
separately restores the linked sizes of all 36 grouped symbols. The
instruction comparison found differences in the diagnostic sections of two
symbols, including the compiled source path. The isolated revision passes
the AVX-512 and AVX2 all-format contract.

The isolated revision's Release probe selected the prepared schedule on
all 2,976 ordinary calls observed on rank 1. The 182 one-row drafter calls
and the 1,248 eight-row verifier calls used their original schedules. Both
full comparisons matched the staged control's 256 generated tokens and
per-iteration MTP work. Driver diagnostics were unchanged.

The ordinary probe's rank 1 local execution averaged 264.19 µs.
Its 15.426 MB of compulsory operands per call gives 58.39 GB/s, or 53.1%
of the 110 GB/s simultaneous socket reference.

A separate phase trace timed the same 2,976 ordinary calls with counters
inside the prepared CPU worker team. Gate/Up averaged 94.07 µs for 8.19 MB
per call, reaching 87.02 GB/s or 79.1% of the socket reference. Down
averaged 76.41 µs for 7.24 MB per call, reaching 94.74 GB/s or 86.1%. The
two arithmetic phases together reached 82.3%. The complete prepared FFN,
including plan construction, team entry, both completion edges, SwiGLU Q8,
and retirement, averaged 184.88 µs and reached 83.43 GB/s or 75.8%. The
outer service figure above is from the separate, minimally instrumented
probe and includes the endpoint's other work.

A distinct endpoint trace kept the same Release arithmetic and matched the
32-token control. Rank 1 recorded 2,976 packets with mean 4.11 µs for route
validation and compaction, 23.50 µs for stage setup, and 257.91 µs inside the
compute stage. Output materialization and return validation took 0.70 and
0.71 µs. The endpoint averaged 286.93 µs with detail collection enabled. Rank
0 recorded 1,488 measured packets with 21.21 µs for stage setup and 225.46 µs
inside the compute stage. These endpoint timings include instrumentation and
serve to localize the work around the prepared FFN.

| Arm order | Mode | Decode, control → isolated tok/s | Decode change | Complete throughput change |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 29.598 → 32.069 | +8.35% | +7.04% |
| Control first | Ordinary | 29.421 → 32.040 | +8.90% | +7.31% |
| Candidate first | Fixed depth seven | 43.714 → 44.998 | +2.94% | +2.63% |
| Control first | Fixed depth seven | 43.476 → 45.044 | +3.61% | +3.05% |

The isolated Release fixture compared the complete FFN with four and ten
live routes, 28 workers on each socket, and both arm orders. It checked the
Gate/Up, SwiGLU Q8 and Down bytes before and after timing. All 216 cells
improved, covering every Gate/Up source with Q8_0 Down and every other
legal K320 Down source with Q4_K Gate/Up. Complete FFN latency fell by
4.3–39.1% across the cells. Q4_K Gate/Up with Q8_0 Down improved by
26.4–36.2%. Runtime selection uses the original shape, worker, numerical
and execution-phase contracts for the prepared schedule.

A separate route-local publication experiment lets Down start after its
own route's SwiGLU Q8 packets are complete. It matched all 216 cells and
improved 201, with a median 5.0% lower complete FFN latency. Individual
cells ranged from a 30.9% decrease to a 19.0% increase. Q4_K Gate/Up improved
in every measured cell with either Q5_1 or Q8_0 Down.

The paired Release model passed exact 256-token history and MTP work checks
in both orders. Ordinary decode changed from 31.998 to 31.741 tok/s with the
candidate first and from 31.771 to 31.479 tok/s with the control first.
Fixed-depth-seven decode measured 45.075 → 44.190 and 44.683 → 43.579 tok/s
with the same staged MTP schedule in both arms. The prepared CPU production
schedule uses the team completion edges.

A separate CPU trial bound only live fragment triplets from the original
packet's acquired bank and skipped building the GPU routing mask on the CPU.
It retained the existing bank release edge and CPU arithmetic. The matched
32-token traces authenticated all 2,976 candidate bindings on rank 1. Mean
stage setup fell from 23.50 to 9.11 µs and packet service fell from 286.93 to
251.21 µs. On rank 0, setup fell from 21.21 to 8.73 µs and packet service
fell from 250.68 to 246.84 µs over 1,488 observed packets.

The same-library Release comparison used one warmup and three measured
256-token runs in each mode. Both orders matched every generated token and
MTP count. The experiment remains gated while the fixed-depth result is
investigated.

| Order | Mode | Decode, original → live binding tok/s | Decode change | Complete throughput change |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 32.309 → 32.871 | +1.74% | +1.19% |
| Control first | Ordinary | 31.270 → 32.269 | +3.19% | +2.33% |
| Candidate first | Fixed depth seven | 45.492 → 44.672 | −1.80% | −1.36% |
| Control first | Fixed depth seven | 43.884 → 43.926 | +0.09% | −0.06% |

A separate 32-token fixed-depth-seven trace matched the saved prompt, token
sequence, and MTP work. Both arms recorded zero live bindings and zero
ordinary prepared FFN plans. On rank 1, the mean endpoint time for 182
capacity-one packets was 419.92 → 400.65 µs and the mean for 1,248
capacity-eight packets was 1,115.84 → 1,081.18 µs. Rank 0 changed
420.33 → 417.55 µs and 1,111.78 → 1,114.81 µs across the same capacities.
These instrumented endpoint timings leave the quiet end-to-end difference
unexplained. The live-binding experiment remains gated.

The sealed IQ4 scale probe resolved the scale program at output-tile entry.
The IQ4_XS packet loop consumed coefficients validated at weight publication.
Both sockets, one and two rows per expert, eight and 64 experts, and both
orders produced 96 byte-exact observations with Q4_K as the timing control.
IQ4_XS Gate/Up improved 2.7–21.4% at eight experts and gave mixed results
at 64 experts. The Q4_K control varied by up to 14.0% in the small workloads,
so these clocks did not qualify a production dispatch change.
Both arms used the staged CUDA graph and identical F320/S320 sharding.
Every arm used one warmup and three measured 256-token requests per mode.
All generated IDs, prompts, and per-iteration MTP counts matched.

| Arm order | Mode | Control → candidate, tok/s | Decode change |
| --- | --- | ---: | ---: |
| Candidate first | Ordinary | 30.836 → 30.342 | −1.60% |
| Candidate first | Fixed depth seven | 45.589 → 45.074 | −1.13% |
| Control first | Ordinary | 29.812 → 30.450 | +2.14% |
| Control first | Fixed depth seven | 44.077 → 45.052 | +2.21% |

The throughput change reversed with run order. These samples qualify the
available QFN weight file's output parity. The other source formats were
qualified in captured projection fixtures.

The AVX-512 compact-grid experiment moved the integer center into the
read-only IQ1_M, IQ2_S, and IQ2_XS lookup tables. All 30 original geometry,
source, and socket cells remained byte-exact. IQ1_M improved in several
grouped cells and added 4.0–5.8% to the eight-expert M=1 Gate/Up latency.
The 64-expert M=1 Gate/Up cells also slowed for IQ2_S and IQ2_XS. The
production grid arithmetic stays with the original tables.

### Direct packed Down publication

The standalone CPU Down candidate wrote each prepared native result into the
existing packed route record. Only the payload destination changed. The route
identity occupied the same trailer word, and the consumer kept the original
top-k fold. Both AVX2 and AVX-512 passed the native publication contract. All
216 matched format, socket, route-count, and order cells were byte-identical.
Their complete isolated transaction latency fell 5.3–25.5%, with a median
reduction of 17.6%.

The Release A/B used the local QFN UD-Q4_K_XL model with the Q8 sidecar,
2×RTX 3090 and 2×Xeon 6238R, and the existing F320/S320 placement. Each arm
ran one warmup and three measured 256-token requests in each mode. The generated
tokens and per-iteration MTP work matched in both orders.

| Order | Mode | Control → direct decode, tok/s | Decode | Complete |
| --- | --- | ---: | ---: | ---: |
| Candidate first | Ordinary | 31.036 → 32.806 | +5.70% | +4.76% |
| Control first | Ordinary | 31.094 → 32.526 | +4.61% | +3.78% |
| Candidate first | Fixed depth seven | 43.507 → 44.831 | +3.04% | +2.40% |
| Control first | Fixed depth seven | 43.476 → 44.442 | +2.22% | +1.75% |

The counter-enabled 32-token run observed 2,976 direct Down writes. Raw
samples and the matched comparison are in
`benchmark_results/qfn-cuda-2x2-95943372/cpu-direct-canonical-down/`.
These measurements cover the isolated standalone candidate.
