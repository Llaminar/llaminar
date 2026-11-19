# QFN MI50 empirical compute and memory reference

The 15 generated original-source families exercise the Release full native
HC/router/FFN capture on MI50 gfx906. Every row uses M1, H2560, routed F256,
shared S256, 512 experts and topK10. The 60-CTA × 1024-thread wave64 dispatch
issues 960 waves. LDS per CTA is 47,712 B for G01 and 36,224 B for G02–G15.
Observed private bytes are zero. Timing covers the complete local capture
from the previous write through final FFN output before the collective.
The staged/full fixture retains 32 ABBA/BAAB quartets and 64 samples per arm.
It checks original output, bank, route, histogram, reset and empty state before
timing.

## Measured families

Reader order is **routed gate/up, routed down, shared gate/up, shared down**.
The names follow the physical execution masks recorded by each capture.
G01 reads Q4_1 from prepared GGUF Q4_K weights. The other source names match
the readers shown. G15 has separate IQ2_S shared gate/up and Q8_0 shared down.
The PCI column identifies the device-specific calibration.

| Family | PCI bus | Native readers | Full median (µs) | FetchSize (KiB) | VALU wave instructions | Memory ref. | Dot4 issue ref. | GGUF FFN (B) | Packed FFN (B) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| G01 | 97 | `Q4_1,Q5_1,Q8_0,Q8_0` | 132.104 | 24,382.4375 | 2,960,593 | 21.53% | 24.59% | 14,376,960 | 15,196,160 |
| G02 | 9a | `Q4_0,Q4_0,Q4_0,Q4_0` | 125.483 | 21,419.6250 | 2,970,741 | 19.88% | 25.50% | 12,165,120 | 12,165,120 |
| G03 | 9a | `Q4_0,Q4_0,Q8_0,Q8_0` | 130.886 | 22,382.6875 | 2,988,340 | 19.91% | 24.59% | 13,148,160 | 13,148,160 |
| G04 | 9a | `IQ4_NL,IQ4_NL,IQ4_NL,IQ4_NL` | 128.501 | 21,418.6875 | 3,220,895 | 19.41% | 26.99% | 12,165,120 | 12,165,120 |
| G05 | 9a | `IQ4_NL,IQ4_NL,Q8_0,Q8_0` | 134.082 | 22,386.8750 | 3,338,641 | 19.44% | 26.82% | 13,148,160 | 13,148,160 |
| G06 | 9a | `IQ4_NL,Q4_0,Q4_0,Q4_0` | 127.928 | 21,422.1250 | 3,159,792 | 19.50% | 26.60% | 12,165,120 | 12,165,120 |
| G07 | 9a | `IQ4_NL,Q4_0,Q8_0,Q8_0` | 129.661 | 22,386.5000 | 3,223,608 | 20.10% | 26.77% | 13,148,160 | 13,148,160 |
| G08 | 8c | `IQ2_S,IQ2_S,IQ2_S,IQ2_S` | 153.555 | 18,139.5000 | 3,572,982 | 13.81% | 25.31% | 6,927,360 | 8,785,920 |
| G09 | 8c | `IQ2_S,IQ2_S,Q8_0,Q8_0` | 145.717 | 19,400.4375 | 3,582,044 | 15.57% | 26.74% | 8,386,560 | 10,076,160 |
| G10 | 8c | `IQ2_S,Q4_0,Q4_0,Q4_0` | 132.207 | 19,435.1875 | 3,357,518 | 17.19% | 27.62% | 8,990,720 | 10,117,120 |
| G11 | 8c | `IQ2_S,Q6_K,Q8_0,Q8_0` | 138.200 | 22,395.7500 | 3,552,571 | 18.95% | 27.96% | 11,663,360 | 13,148,160 |
| G12 | 97 | `IQ3_S,IQ2_S,IQ4_NL,IQ4_NL` | 148.272 | 20,035.5000 | 3,855,125 | 15.76% | 28.53% | 8,837,120 | 10,731,520 |
| G13 | 97 | `IQ3_XXS,Q5_1,IQ2_S,IQ2_S` | 157.841 | 20,738.0000 | 3,563,267 | 15.33% | 24.77% | 10,562,560 | 11,448,320 |
| G14 | 97 | `IQ3_S,Q6_K,IQ2_S,IQ2_S` | 163.211 | 22,737.3125 | 3,843,751 | 16.25% | 25.84% | 11,637,760 | 13,496,320 |
| G15 | c5 | `IQ3_XXS,Q6_K,IQ2_S,Q8_0` | 149.334 | 21,953.7500 | 3,685,259 | 17.22% | 26.98% | 11,509,760 | 12,697,600 |

## Units and normalization

| PCI bus | Streaming read (decimal GB/s) | 60-CTA dot4 (wave instructions/ns) |
|---|---:|---:|
| 97 | 877.8626 | 91.13311 |
| 9a | 879.3964 | 92.85473 |
| 8c | 875.7149 | 91.93747 |
| c5 | 874.0526 | 91.47244 |

Each same-card read rate is the byte-checked median for a 256 MiB stream at
128 CTAs × 256 threads. Each dot4 rate uses 60 CTAs × 1024 threads, matching
the full kernel's launch geometry, and the calibration's independent dot4
chains. The profiler samples the named kernel's captured dispatch ordinal 386.
FetchSize alone and VALUInsts plus Wavefronts were collected in separate runs.
Instrumented durations are excluded.

Let `T` be the unprofiled median in µs, `F` the observed FetchSize in KiB,
`B` the same-card streaming rate in decimal GB/s, and `D` the same-card dot4 rate
in wave instructions/ns.

- Physical fetched bytes = `1024 × F`. The gfx906 SDK metric expresses physical
  memory reads in KiB, including extra fetches and cache effects. It excludes
  writes.
- Issued vector wave instructions `I = VALUInsts × Wavefronts`. The SDK defines
  `VALUInsts` as `SQ_INSTS_VALU / SQ_WAVES`. `I` covers the full vector instruction
  mix, including lane-masked work.
- Memory reference time = `1024 × F / (1000 × B)` µs. Memory reference percentage
  = `100 × memory reference time / T`.
- Dot4 issue reference time = `I / (1000 × D)` µs. Dot4 issue reference percentage
  = `100 × dot4 issue reference time / T`. The numerator scales the whole vector
  instruction mix and is independent of the actual number of dot4 instructions.

Each row has `H × (3 × topK × F_routed + 3 × S)` logical FFN matvec MACs, or
21,626,880. The FFN byte columns count each selected projection once. That
includes three matrices per selected expert and three shared matrices. GGUF
bytes sum original blocks times bytes per block. Packed bytes sum the canonical
native K32 payload, scale and secondary planes from
`nativeQuantisedPackedRegionSizes(...).logicalBytes()`. These logical storage
counts exclude HC, routing and activations. FetchSize and VALU count observed
work across the whole capture. The byte columns follow the fixture's
`reportFullNativeFamilyWork` formula and agree with the receipts that emit it.

The two reference times can overlap. Instruction mix, dependencies, active
lanes, scalar work, LDS, memory latency and synchronization all affect the
elapsed time. These measurements describe component work at S256 and do not
establish performance for the dedicated S192 path, whole-model serving,
grouped MTP or cross-device execution.

## Reproduce and inspect the evidence

The source roster and logical work formula are in
`tests/v2/integration/loaders/FullNativeHCFFNFamilyReplay.inc`. Build the
Release owner `v2_integration_moe_intermediate_ffn_finalization` with the canonical
ROCm SDK and explicit Release native target selection. The unprofiled
component test is:

```bash
LLAMINAR_TEST_RETAINED_HC_FFN_PCI="$PCI" \
  ctest --test-dir "$BUILD" -C Release --no-tests=error \
  -R "^V2_Perf_FullNativeHCFFN_G${GROUP}_ROCm$" -V
```

Use the physically scoped runner and the exact executable and DSO from the
receipts. On the lab host:

- G01 and G12–G15 use
  `/tmp/qfn-mi50-perf-checkpoint-evidence/iq3-family/format-bounded/Gxx`.
  G01 uses `profile-bounded-clean-fetch-G1` and `profile-bounded-r2-vector-G1`.
  Both identify the same native kernel code on PCI97. G12–G15 use
  `profile-bounded-{fetch,vector}-Gxx`.
- G02–G07 use `/tmp/qfn-mi50-families-g02-g07/normalized-efficiency.json`.
  It identifies the timing and profile receipts. G03 uses scoped core
  `cd36072f`. The other five use `94defe98`.
- G08–G11 use `bounded-isolation` and `bounded-profile` under
  `/tmp/qfn-castagna-production-contract-r62/iq2-family-campaign`.
  The scoped core is `3dfa2627`.
- Calibration replays use `iq3-family/{bandwidth,issue}/isolated-native.py`
  with `LLAMINAR_ISOLATION_PCI="$PCI" python3 <runner> numerics <new-receipt>`.
  PCI9a and PCI8c read receipts are in their family evidence.

Recover the exact launch from `command.json` and the counter configuration from
`plan.json`. Verify the loaded object, geometry, named graph dispatch, driver and
physical retirement in the corresponding `result.json` and
`profile-result.json` records. Raw telemetry and binaries remain on the lab host.
