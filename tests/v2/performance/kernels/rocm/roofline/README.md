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

Supply a whitespace-separated route file with three groups of 40 original routed expert IDs. Group order is Q4_K/Q5_1, Q4_K/Q8_0, Q5_K/Q8_0. In each group the 40 IDs hold the four captured rows in original selected-slot order, with all ten routes local to the original participant. Use IDs observed on the actual fixed shape, rather than generated uniform indices. The file may be recorded with its source model revision, participant, original bank/epoch, graph settings and live route census outside this repository. The packet diagnostic rejects missing, extra and out-of-range IDs and repeated IDs in one row. It rotates four physically distinct device allocations and checks bank-tagged lane checksums. The kernel uses the original role payload and coefficient spacings. It omits shared weights and arithmetic.

```bash
python capture.py packet --pci "$PCI" --build "$BUILD" --output "$RESULTS/packet-quiet" \
  --routes "$ROUTES"
python capture.py packet --pci "$PCI" --build "$BUILD" --output "$RESULTS/packet-gate-fetch" \
  --routes "$ROUTES" --kernel 'packetRead<4, true, true>' --iteration 257 --metric fetch
python capture.py packet --pci "$PCI" --build "$BUILD" --output "$RESULTS/packet-down-fetch" \
  --routes "$ROUTES" --kernel 'packetRead<5, true, false>' --iteration 257 --metric fetch
```

These select the first source tuple's warm four-bank dispatches. For the second Q4_K gate or the third symmetric Q8_0 down with the same respective signature, iteration 689 selects its later window. The third source gate uses `packetRead<5, true, true>` at 257. Both Q8_0 downs use `packetRead<8, false, false>`. Record WriteSize in a separate invocation using `--metric write`. Capture fails if an exact counter trigger is missing or if the profiler selects another metric set.

Use the independent quiet median for a scored shape and warm case. The physical rate is FetchSize or WriteSize in KiB multiplied by 1024, divided by elapsed microseconds and then by 1000 for decimal GB/s. The application's requested packed bytes and useful unique bytes retain their own denominators. Report observed rates, source binding, resource geometry and service-time comparisons separately. Keep native binaries, raw counters and histories outside the source tree.
