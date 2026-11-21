# ROCm grouped verifier epilogue repair

This prerequisite to [grouped MTP WMMA](GFX1100_MTP_GROUPED_WMMA_PLAN.md)
repairs standalone NativeQuantised verifier projections with nonzero beta or
mapped output. It does not install an MTP WMMA producer or establish an MTP
speedup. Source base: `4854c84dd`; evidence root:
`/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

## Reproduced defect and implementation

Previously, grouped admission required beta zero and an unmapped destination.
A fully active M3 call with beta or mapped output recorded three serial M1
producers. With an explicit device-counted M3 descriptor, the first one-row
quantizer instead threw `Verifier launch row geometry disagrees with physical
matrix`. The authenticated original-source run reproduced both behaviors;
neither met the grouped verifier contract.

The repaired path quantizes and submits one grouped producer, retaining the
original physical geometry and device-owned row authority. Beta/mapped output
uses the existing persistent `TEMP_C_FP32` workspace and one common combine
kernel. The adapter rejects absent, aliased or undersized scratch. The obsolete
production serial-row loop has been removed.

The combine boundary forwards the verifier prefix and inherits independent
`ProjectionRowWork`, including request-local holes and subview origins. Its
counted specializations reject inactive rows before reading any operand or
writing output. The count-free specialization retains the ordinary path.
Pure copies preserve signed zero, subnormals and NaN payload bits. Unsigned
flat indexing and overflow-safe grid rounding cover the admitted INT32 extent.
No host live-count readback, new hot allocation or recapture is introduced.

## Replay and feature results

`V2_Integration_GroupedVerifierRows_ROCm_EpilogueReplay` is registered explicitly
in `ProductionTestPreflight` and the grouped-verifier feature family. Its four
cases pass in Release and Integration on gfx1100:

| Case | Contract exercised |
|---|---|
| `DeviceOutput` | Q4_0 M3/N17/K32, alpha 0.75, beta -0.25, bias; fixed and device-counted rows; one grouped producer and zero serial M1 producers. |
| `MappedOutput` | The same independent production-M1 byte oracle with mapped destination, including beta-zero copy/combine. |
| `RequestHolesAndSubview` | An M4 view beginning at original row 1 of two capacity-3 requests; mixed request counts, holes and both destination placements. |
| `MappedCopyBits` | Exact signed-zero, subnormal and NaN payload words through repeated counted mapped copies. |

Counted and request-work graphs replay full, small, empty and regrown work;
fixed-row graphs repeat fully active work. Consumed scratch is poisoned between
projection replays, and output is reset to authored nonzero beta inputs.
The copy-only case poisons inactive destination bytes. Inactive output remains
untouched in every case. Projection-case node inspection checks the actual
grouped producer and the exact row descriptor/capture identity of the combine.
Stable host count
storage and a fatal-on-failure exact-stream drain protect asynchronous fixture
lifetimes, including assertion exits.

The affected feature run passed all three registrations without skips:
`V2_Integration_ROCm_DeviceCountedVerifierRows`, the new epilogue replay, and
`V2_Integration_GroupedVerifierRows_ROCm_AllFormats`. That run preceded the
test-only addition of the request-hole/subview case; all four final cases then
passed in both builds against the unchanged production cores. Full preflight
and model-level MTP gates were not run, per the user's requested scope.

## Compiler resources and profiling

Release and Integration compiled the complete shipping HIP set, gfx906 and
gfx1100, with spill guards intact. All three combine specializations have zero
private memory, zero VGPR/SGPR spills and zero LDS in both builds.

| Specialization | gfx906 VGPR / SGPR | gfx1100 VGPR / SGPR |
|---|---:|---:|
| Fixed rows | 7 / 24 | 8 / 22 |
| Device-counted prefix | 8 / 26 | 7 / 24 |
| Request work | 11 / 40 | 13 / 38 |

All **28 exact captured-dispatch counter passes** completed: fixed, counted,
request-work and mapped-copy paths, each with seven metrics. Every dispatch
uses one 256-thread CTA (four wave64 waves), with zero scratch instructions or
LDS bank conflicts. This grid provides only four waves globally; these small
correctness cases cannot assess device-wide sustained occupancy or throughput.

| Path | Memory-unit busy | L2 hit | Fetch size (KiB) | VALU instructions | Profile duration (µs) |
|---|---:|---:|---:|---:|---:|
| Fixed | 6.472% | 25.000% | 1.500 | 10.50 | 2.64–3.36 |
| Counted | 0.000% | 21.053% | 1.875 | 11.75 | 2.80–3.16 |
| Request work | 5.958% | 13.889% | 3.875 | 69.00 | 5.48–6.04 |
| Mapped copy | 0.000% | 0.000% | 1.750 | 9.00 | 3.76–4.12 |

These are independent instrumented counter passes, not unprofiled latency
measurements. Zero sampled memory-unit activity does not mean no memory access;
the kernels are too short to infer bandwidth utilization. `GPUBusy` reports
100% for these selected intervals and likewise does not establish global GPU
utilization. No MTP performance improvement is claimed.

The profiler's iteration argument accepts a single integer; the rejected
`1:1` probe is retained as a tooling failure. Corrected probes select one actual
captured combine and authenticate its loaded code object, PCI/UUID, nondefault
stream and graph identity. Each restores the original device power policy after
its owned process retires. No gfx1100 isolation receipts or monitoring were used.

## Evidence identities

| Evidence | Relative path / identity |
|---|---|
| Original-source failure | `mtp-verifier-epilogue-red-1794734704972619354/receipt.json`; saved original test and frozen core reference |
| Final Integration replay | `mtp-verifier-epilogue-four-integration-replay-1794735624882833235/receipt.json` |
| Final Release replay | `mtp-verifier-epilogue-four-release-replay-1794735636763869769/receipt.json` |
| Affected feature run | `mtp-verifier-epilogue-focused-1794735230258753666/receipt.json` |
| Counter campaign | `mtp-verifier-epilogue-counter-campaign-1794735982009893610/campaign.json` |
| Counter summary SHA256 | `mtp-verifier-epilogue-profile-summary.json`: `70fbaa4efd08451ed5616ec21cf9ed0142090c18993de1300cae20caebab7e79` |
| Frozen producer/evidence | `mtp-verifier-epilogue-qualified-snapshot/snapshot.json` |
| Full-target allocation | `mtp-verifier-epilogue-{release,integration}-allocation/allocation.json` |
| Integration core SHA256 | `c8e365b64eed7bde511a480f1dddffa94cd4f6dfd1265205a8506401f22594e3` |
| Release core SHA256 | `550212e81b7660fb85bd9d744bee630604bf867276bf9932ee257ccd8b606706` |
| Release replay binary SHA256 | `79ae2e7daf72c26f62efd4659bf32267587ddd5ed29e5388173ba0e03a8abba6` |

Read-only reviews found the final row-coordinate oracle, captured descriptor,
stream retirement and inactive-byte checks sound. The next implementation is
the actual two-wave M16 WMMA verifier producer and its standalone/fused,
all-format byte and economy qualification.
