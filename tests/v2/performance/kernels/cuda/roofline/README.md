# CUDA decode hardware references

Build the explicit probe in a CUDA Release configuration. Run each mode on an
idle card. The first record identifies the physical PCI device, ISA, SM count,
and L2 size. Keep raw samples and the final checksum result. Repeat in fresh
processes with reversed card and mode order.

```sh
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_cuda_decode_roofline
PROBE="$CMAKE_BUILD/tests/v2/v2_perf_cuda_decode_roofline"
"$PROBE" 0 dram > dram.csv
"$PROBE" 0 l2 > l2.csv
"$PROBE" 0 dp4a > dp4a.csv
"$PROBE" 0 imma > imma.csv
```

Every sample times 16 consecutive launches in a retained CUDA graph and reports
the duration per launch. The DRAM probe traverses a working set at least eight
times the observable L2 capacity once per launch. The L2 probe repeatedly
traverses a set that fits in half of L2. Both use the same coalesced 16-byte
loads and checksum every output lane. The packed signed DP4A probe uses eight
independent accumulator chains. Its work column counts DP4A instructions across
all threads. Each instruction performs four integer multiply-accumulates.
The dense INT8 tensor-core probe counts warp-level `m16n8k32` operations, each
consisting of 4,096 integer multiply-accumulates.

Keep physical traffic separate from requested bytes. In a distinct profiler
process, use `--kernel-name 'regex:decodeReadCeiling' --launch-skip 79
--launch-count 1` to select the last warmed read dispatch. Use
`dram__bytes_read.sum` as the DRAM read denominator and record physical writes
separately. Use device-available L2 read sectors or bytes for the L2 counter
reference. Bind the native counters and static resources to the exact PCI
identity, clocks, executable, grid, and mode. Use quiet event samples for
duration. Follow the CUDA tuning skill's driver diagnostic window and
attachment procedure.

Use the [grouped verifier analyzer](../../qwen4exp/verifier-roofline.md) to
combine these references with the actual routed, shared, and drafter dispatches.
The streaming and issue probes describe independent hardware resources. A
format-specific efficiency result requires replay of its original route order,
live work, prepared weight format and layout, and graph geometry. The measured
parent must also include required publication, ordered reductions, and any
collectives on its critical path.

## RTX 3090 reference measurements

Each entry is the median of 63 retained-launch samples from three fresh
processes. The PCI endpoints are the two sm86 cards in the QFN CUDA/CPU
qualification host.

| PCI endpoint | Streamed reads, GB/s | L2-resident reads, GB/s | DP4A, TOPS | INT8 tensor core, TOPS |
| --- | ---: | ---: | ---: | ---: |
| `0000:3f:00.0` | 879.53 | 2349.78 | 73.30 | 317.08 |
| `0000:42:00.0` | 879.53 | 2378.91 | 73.35 | 319.36 |

Read rates count requested bytes. Integer rates count two operations per
multiply-accumulate. The driver logged a graphics-engine assertion during
counter attachment on this host, so these references use quiet CUDA events.

The captured M1 MoE regression can retain independent native-leaf event
samples for its IQ2_S gate/up, IQ4_NL Down and 320-column SwiGLU geometry.
Run this in a separate process from the model throughput benchmark.

```sh
LLAMINAR_TEST_CUDA_MOE_M1_PROBE_JSONL="$PWD/moe-m1-leaves.jsonl" \
    "$CMAKE_BUILD/tests/v2/v2_integration_cuda_moe_kernel" \
    --gtest_filter=Test__CUDAMoEKernel.RuntimeGroupedDecodeSwiGLUPublicationReplay
```

For the captured 512-column IQ4_NL Down layer, the fixed 16-partition
projection measured 2.05 µs and its separate reducer measured 1.60 µs on
each card. The direct K32 publication measured 1.79 µs on `3f:00.0` and
1.73 µs on `42:00.0`, at 40 registers per lane and zero local bytes.
Its logical weight footprint is 737,280 bytes across eight routes and ten
K32 packets per output. Dividing that footprint by the leaf time gives
411–427 GB/s, or 17.5–17.9% of the warm L2 requested-read reference. This
calculation excludes activation and output traffic. The isolated leaf
times describe separate native graphs, each with 62 samples per card.
