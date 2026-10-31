# Mixed-ISA MoE reduction diagnosis — 2026-10-30

## Result and scope

The eight mixed MI50/7900 XTX numerical failures have one demonstrated cause:
different FP32 sum trees in the common MoE wave reduction. This is an arithmetic
ordering defect, not evidence that FP16 precision or quantized expert weights
need to change. All four router and all four projection registrations pass with
the corrected tree and their original byte-exact assertions.

The qualification here is model-free native captured replay. The subsequent
complete production preflight also passes; neither certifies whole-model
accuracy or performance.

## First divergent values

The original router diagnostic recorded all three source observations: the
transmitted packet, the source's full oracle and the receiver's output. They
were identical. The receiver's locally computed oracle differed by one ULP;
one example was `0x3e001809` versus `0x3e00180a`. Thus the exchange preserved its
input bytes, but the two architectures computed different probabilities.

The original projection diagnostics retained byte-identical Q8 intermediates and
scales. On the first Q4_0 case, final output differed by five or six ULPs:

| Receiver | Observed | Local oracle | Distance |
|---|---|---|---:|
| MI50, column 128 | `0x3e29a0b5` | `0x3e29a0bb` | 6 ULP |
| 7900 XTX, column 0 | `0xbe52faf8` | `0xbe52faf3` | 5 ULP |

Q4_0 was the first format tested, not an isolated faulty format. Both counted
and shared-column tests select `SharedColumnCompletion`. Their full and column
paths compute a shared-expert sigmoid gate through the same wave-sum helper
used by the router. Local routed and shared columns already matched their local
oracles; assembling columns gated on another architecture exposed the drift.

## Arithmetic cause and correction

`ROCmMoEWaveReduction.h` preserves gfx906's established DPP implementation. Its
FP32 tree pairs adjacent lanes first, then combines groups of 4, 8, 16, 32 and
64: ascending XOR distances `1,2,4,8,16,32`.

The portable wave64 branch previously used descending down-shuffle distances
`32,16,8,4,2,1`. Both produce a mathematical sum, but they associate FP32
additions differently. Rounding differences survive softmax normalization and
the shared-expert sigmoid, then propagate into assembled projection output.

The portable sum now uses the same ascending XOR tree. The gfx906 DPP body,
maximum reduction, per-lane dot products, quantization, sigmoid, publication
boundaries and numerical acceptance limits remain unchanged. These kernels
still use the explicit compiled wave64 contract on gfx1100; this result does
not qualify a future wave32 implementation.

## Causal controls

1. With the original code, all eight cases passed on four MI50s and all eight
   failed with the mixed MI50/7900 inventory.
2. The corrected sum passed the actual-helper captured replay and all four
   mixed-device router registrations.
3. After the separate ownership repairs were installed, reverting **only** the
   portable sum loop reproduced the exact two projection bit patterns above.
4. Restoring the corrected loop passed all four projection registrations,
   covering native/counted publication, two/four participants, changing live
   work, retained captures and the existing expert-format sweep.
5. The native Q4 scale/contribution replay also passed all five AMD devices.
   Its 256 arithmetic cases and explicit-widening diagnostic did not identify
   an FP16 conversion/multiply fault. No production precision change was made.
6. The focused full/column shared-gate regression was then qualified against
   the original loop: it fails on the first width-256, one-row trace with a
   three-ULP gate mismatch across cards. The corrected loop passes that test
   and the independent actual-helper replay.

Each native run retained clean driver diagnostics and confirmed process/MPI
retirement. The control restoration also rebuilt the fixed binary, rather
than leaving an original-order DSO behind.

## Durable regressions

`V2_Integration_ROCmMoEWaveReductionNativeReplay` executes the actual production
helper with recorded lane inputs and an independent fixed-tree host expression.
It retains a frozen original down-tree counterexample and checks full, partial,
empty and repeated captured replay on every visible ROCm device.

`V2_Integration_ROCmSharedGateCrossDeviceCapturedReplay` runs the actual full and
column gate APIs with the original projection input construction. Widths
256/512, rows 1/16/129 and partition degrees 2/4 each receive twenty retained
replays with poisoned outputs and changing full/empty/partial counts. The test
checks every output word locally and compares the compact observed gate values
across physical devices. It includes guard suffixes and exact inactive zeros.
Both regressions are explicitly registered in `ProductionTestPreflight`.

The existing all-format projection and router integration tests remain the
broader consumer checks. Compact first-mismatch and source-provenance diagnostics
remain; the temporary full Q4 payload dump has been removed.

## Evidence

Artifacts are local, ignored results under `/tmp/qfn-admission-integration-r19/`:

- `metadata-numerics-original-comparison-r20.json`: original homogeneous/mixed controls.
- `metadata-router-tree-mixed-r21/`: actual helper plus four router passes.
- `metadata-wave-control-r23.json`: isolated original-order projection failure
  and four corrected projection passes, with source hashes and rebuild receipts.
- `metadata-q4-contribution-r22/`: actual Q4 arithmetic replay on all five cards.
- `metadata-shared-gate-focused-r24/`: cross-device gate, column sweep and
  subnormal-publication regressions, three passes with clean retirement.
- `metadata-wave-control-r25.json`: original-order failure and corrected pass
  of the new focused shared-gate regression. Its overlapping timing probes are
  explicitly excluded from economy evidence; isolated probes use revision 27.

`/tmp/qfn-router-isa-r21/` binds the compiler, object hashes, disassembly and
resources for both shipping HIP targets. The corrected portable ISA contains
ascending XOR steps. All 34 inspected sum consumers compile as wave64 with no
VGPR spills. The 21 gate-logit/row-softmax specializations have zero private
storage and no SGPR spill metadata; larger runtime families require the final
spill guard's classification of intentional storage and register-lane moves.
The complete CUDA SM80/86/89/90 and HIP gfx906/gfx1100 build passed its mandatory
spill guards. Native economy qualification is separate from these ISA facts.

## Bounded economy check

The existing retained-graph gate microbenchmark was run one registration at a
time, with three fresh process repetitions per variant, separate from profiler
work. Integration uses optimized code; these are component timings, not a
Release server benchmark. Geometry is 512 rows, 448 live, width 2048, forty
launches per graph and 41 event samples after eight warmups.

| Device / output | Original median µs | Fixed median µs |
|---|---:|---:|
| MI50 / full width | 45.548 | 45.712 |
| MI50 / half columns | 45.784 | 45.644 |
| 7900 XTX / full width | 10.768 | 10.739 |
| 7900 XTX / half columns | 7.603 | 8.099 |

The MI50 and full-width XTX changes are within 0.4%. XTX half-column samples
overlap substantially (original 7.587–8.065 µs; fixed 7.594–8.124 µs), with a
6.5% median increase in this small sample. This does not establish a throughput
win or stable regression. Further tuning is deferred per the bounded XTX
qualification scope; the correct arithmetic remains installed. Results and
individual sample records are in `metadata-wave-economy-summary-r27.json` and
`metadata-gate-perf-{original,fixed}-{gfx906,gfx1100}-{full,columns}-r27/`.

Complete Unit (844/844) and the separate focused repair batch (29/29) pass with
clean native driver and process-retirement evidence at revision 26. The complete
frozen revision-28 prerequisite gate subsequently passes: 3,239 registrations,
zero failures and 29 explicitly unavailable hardware cases, including all eight
original mixed-ISA failures. Its driver report has zero findings and all 516
owned files remained unchanged. See `prerequisites-metadata-r28/` and
`metadata-frozen-source-r28-validation.json` under the same evidence root.
