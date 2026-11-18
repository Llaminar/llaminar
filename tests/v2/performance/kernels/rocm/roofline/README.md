# MI50 read and vector issue references

These standalone diagnostics measure physical read and write counters, useful read requests, and issued vector instructions independently. They allocate synthetic checksum data for the probes and do not load a model or modify production dispatch.

Set ROCM_PATH to the repository-pinned SDK. Use an exclusive device cgroup exposing the one MI50 at the explicitly assigned PCI address. The runner checks the visible system render node and KFD device, authenticates the native HIP device and ISA, monitors the kernel driver and checks physical retirement. GPU visibility environment variables alone do not establish isolation.

Build the explicit target from an ordinary out-of-tree Release configuration
with ROCm and Llaminar tests enabled. It uses the configured architecture set
and repository HIP spill guard. Then, from this directory, use fresh output
paths and point the runner at the diagnostic executable directory.

```bash
cmake --build "$CMAKE_BUILD" --parallel --target v2_perf_rocm_mi50_roofline
BUILD="$CMAKE_BUILD/tests/v2/mi50-roofline"
python capture.py read --pci "$PCI" --build "$BUILD" --output "$RESULTS/read-quiet"
python capture.py issue --pci "$PCI" --build "$BUILD" --output "$RESULTS/issue-gate-quiet" \
  --issue-grid 120 --issue-block 256
python capture.py issue --pci "$PCI" --build "$BUILD" --output "$RESULTS/issue-gate-counter" \
  --issue-grid 120 --issue-block 256 --kernel 'issue<\(Operation\)2, 256u>' \
  --iteration 48 --metric vector
```

The down geometry uses 160 workgroups of 256 threads, corresponding to a native 40×4 workgroup grid. The three issue probes execute an independent signed packed DOT4, INT ADD or FP32 FMA instruction schedule. To report total vector instructions, multiply VALUInsts by Wavefronts. Divide by a quiet duration from the corresponding op, grid and block. Use the resulting issue rates as empirical references for the application's mixed VALU work. They establish neither an actual mix of its instructions nor a combined compute-and-memory bound.

## Cache-resident and DRAM controls

```bash
python capture.py read --pci "$PCI" --build "$BUILD" --output "$RESULTS/read-resident-fetch" \
  --kernel 'readOnly<4u>' --iteration 34 --metric fetch
python capture.py read --pci "$PCI" --build "$BUILD" --output "$RESULTS/read-stream-fetch" \
  --kernel 'readOnly<4u>' --iteration 994 --metric fetch
```

Once warmed, the 2 MiB working set is read 64 times per dispatch at 128 workgroups and 256 threads. The 256 MiB DRAM control uses one pass with the same workgroup geometry. Iterations 34 and 994 name one warm native dispatch of each shape in round zero. Both verify every output lane. The cache-resident case is the negative control for reading requested bytes as DRAM traffic. On one measured MI50 its 1.446 TB/s request rate corresponded to just 27.18 GB/s of aggregate FetchSize. The large stream on the same card measured 873.24 GB/s of physical FetchSize. These observations depend on the actual hardware and software setting.

## QFN routed read pattern

Supply one whitespace-separated history of 40 expert IDs observed on the original M4 graph. Preserve row and selected-slot order, including repeated experts across different rows. Each row must contain ten distinct local expert IDs. Record the source model, participant, native shard extents, bank history and graph settings with the route file.

Pass the tuple number, routed and shared intermediate widths, and both gate K partition counts as `--packet-shape TUPLE F S ROUTED_PARTS SHARED_PARTS`. For the original H2560 source policies the gate reductions use 16 partitions. The native K32 blocks use these independent layouts.

| Tuple | Routed gate/up | Routed down | Shared gate/up | Shared down |
| --- | --- | --- | --- | --- |
| 0 | Q4_K (16 + 2 + 2 B) | Q5_1 (20 + 2 + 2 B) | Q8_0 (32 + 2 B) | Q8_0 |
| 1 | Q4_K (16 + 2 + 2 B) | Q8_0 (32 + 2 B) | Q8_0 | Q8_0 |
| 2 | Q5_K (20 + 2 + 2 B) | Q8_0 (32 + 2 B) | Q8_0 | Q8_0 |
| 3 | IQ3_XXS (12 + 2 B) | Q6_K (24 + 2 + 2 B) | IQ2_S (9 + 2 + 2 B) | Q8_0 |

The two-byte planes hold scale and, where present, an affine minimum or second coefficient. This probe keeps the parent direct reader's 128-thread gate tile, per-wave K stripes, grid-stride routed and shared tiles, and down CTA's original slot and full-column traversal. The four routed banks occupy disjoint ranges. Shared matrices are reused across rows and banks. Every payload byte and coefficient varies with its address. The host reconstructs the expected checksum for every lane on all four banks using logical matrix coordinates.

This is a weight-only traffic reference. Production descriptor indirection, address placement across independent allocation owners, activation loads, codebook LUT fetches, publication stores, dequantization arithmetic and reductions have separate cost. For odd-byte packets the calibration reads the complete physical payload and coefficients. It does not reproduce the decoder's exact instruction stream. Warm one-bank and rotating four-bank results describe the stated route working set. Neither mode guarantees cache residency or a cache flush.

```bash
# QFN layer 1, original F256/S160 participant
ARGS=(packet --pci "$PCI" --build "$BUILD" --routes "$ROUTES" --packet-shape 0 256 160 16 16)
python capture.py "${ARGS[@]}" --output "$RESULTS/packet-quiet"
python capture.py "${ARGS[@]}" --output "$RESULTS/packet-gate-fetch" \
  --kernel 'packetRead<16, true, true, 32, false>' --iteration 257 --metric fetch
python capture.py "${ARGS[@]}" --output "$RESULTS/packet-down-fetch" \
  --kernel 'packetRead<20, true, false, 32, false>' --iteration 257 --metric fetch
```

Iteration 257 selects a warm rotating-bank dispatch for an explicit one-tuple invocation. Record WriteSize in a separate invocation using `--metric write`. The original three-history F160/S160 invocation is still accepted without `--packet-shape`. The later same-symbol dispatch in that invocation uses iteration 691. Capture rejects absent or mismatched counter triggers and metric sets.

Use the independent quiet median for a scored shape and warm case. The physical rate is FetchSize or WriteSize in KiB multiplied by 1024, divided by elapsed microseconds and then by 1000 for decimal GB/s. The application's requested packed bytes and useful unique bytes retain their own denominators. Report observed rates, source binding, resource geometry and service-time comparisons separately. Keep native binaries, raw counters and histories outside the source tree.

The Q4 packet probe can also calibrate a direct single-tile gate launch.
Pass `--single-tile --packet-shape 0 F S ROUTED_GATE_PARTS SHARED_GATE_PARTS`
with the original route packet. This mode emits gate samples only. It launches
one 256-thread workgroup per pair of 32-column tiles, with no grid-stride reuse.
Use `packetRead<16,\s*true,\s*true,\s*32,\s*false,\s*true>` as the counter
kernel filter. Match the physical launch, original ordered routes and prepared
K32 partition policy before comparing it with an inference dispatch.

The same gate-only switch accepts all explicit tuples. Tuple 1 uses the
same Q4_K routed and Q8_0 shared gate format as tuple 0. Tuple 2 uses
the original Q5_K prepared affine reader and Q8_0 shared reader, with kernel
filter `packetRead<20,\s*true,\s*true,\s*32,\s*false,\s*true>`. Tuple 3
uses the G15 IQ3_XXS routed and IQ2_S shared prepared readers, with filter
`packetRead<12,\s*false,\s*true,\s*9,\s*true,\s*true>`.
Keep the original route order and K32 partitions for each measured shard.
