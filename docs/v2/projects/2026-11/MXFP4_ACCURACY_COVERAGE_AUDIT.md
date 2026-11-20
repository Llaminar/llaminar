# MXFP4 accuracy-coverage audit (branch feat/mxfp4-accuracy-coverage)

Base: `079ecdfe9a9bdfecf076911a62916d9cea1cb91c` (PR #38).

Method: inventory-driven greps over `tests/v2`. A file is "auto-covered" when
its format list comes from one of the canonical inventories; "hardcoded" when
it carries a hand-maintained label list. Everything below is read-only audit;
fixes follow in later commits on this branch.

## Auto-covered (no change needed)

The canonical inventories already include MXFP4:

- `tests/v2/utils/QuantizedVerifierFormats.h`: `quantizedVerifierFormats()`
  has `{"MXFP4", TensorType::MXFP4, kNativeQuantisedMXFP4Codebook, ...,
  createMXFP4Random}` (line 125).
- `quantizedMoEVerifierFormats()` derives from it (overrides only the IQ4_XS
  and IQ3_S creators), so MXFP4 is present in the MoE verifier inventory.
- `kAllSourceFormats` (`src/v2/tensors/NativeVnniFormatInfo.h`) includes
  MXFP4 (codebook 25), and PR #38 wires it into the shared all-format suites
  and immutable-lookup inventories.

Files that iterate the inventories therefore already exercise MXFP4 with the
same depth as peer formats. Confirmed users (spot-checked, 11 files):

- `Test__CUDAMoEKernel.cpp` (CapturedMixedFormatTransfers...,
  `runMixedFormatExpertTransferPublication`)
- `Test__ROCmMoEKernel.cpp` (same, plus 5 direct MXFP4 mentions)
- `Test__ROCmMoEProjectionBoundary.cpp` (+
  `ROCmMoEProjectionBoundaryFixture.h` with the 1-byte plane fix from PR #38)
- `MixedFormatExpertTransferPublication.cpp`
- `NativeQuantisedExpertTransferParityTest.h`
- `Test__CPUNativeQuantisedVerifierWorkers.cpp`
  (`MixedFormatExpertTransactionPreservesEveryProjection` cycles the whole
  MoE registry through gate/up/down roles)
- `Test__CPUNativeQuantised_GEMV.cpp`,
  `Test__CPUNativeNEONExecution.cpp`, `Test__CPUNativeNEONProjectionBatch.cpp`
- `Test__DeviceMoEExpertDescriptorBuilder.cpp`,
  `Test__MoEVerifierFixtures.cpp`, `MoEProjectionPipelineFixture.cpp`
- `Perf__GPUExpertPipeline.cpp`, `Perf__CPUMoEExpertVerifierRows.cpp`,
  `Perf__ROCmMoEVerifierPrefill.cpp` (inventory loops).

## Task-named items: status

| Item | Status |
|---|---|
| Dense GEMV decode M1, grouped MTP M2..16/M31, prefill (CPU x86 AVX2/512, CUDA, ROCm) | Covered where the suite iterates `quantizedVerifierFormats()` (e.g. `Test__CPUNativeQuantised_GEMV`, `CPUNativeVNNI*` post-#38). Perf gates below still miss MXFP4. |
| Serial-row byte equivalence of grouped rows | Covered by `Test__CPUNativeQuantised_GEMV` inventory sweeps and the MoE verifier perf tests that use the inventory. |
| MoE routed FFN, shared expert, paired gate/up, combined top-k | `Perf__MoEVerifierPrefill` CUDA/ROCm `M1234_*`, `M4M9M31_CombinedTop9*` use the hardcoded `sharedExpertPreparedFormatCases()` which has NO MXFP4 -> **GAP**. The `CUDA_ProductionCandidatesAreSerialRowByteInvariant` path uses the inventory loops (covered). |
| CPU fused verifier / fused expert members | `Test__CPUNativeQuantisedVerifierWorkers` covered. `Test__FusedProjectionParity` hardcoded [Q4_0, Q8_0, IQ4_NL] -> **GAP**. |
| QuantizedVerifierFormats, kAllSourceFormats users | Covered (above). |
| Mixed-format expert tuples as in real GGUFs (MXFP4 gate/up + Q5_K/Q6_K down) | `MixedFormatExpertTransactionPreservesEveryProjection` (CPU, registry cycle) and `runMixedFormatExpertTransferPublication` (CUDA/ROCm, registry-derived specs) cover all pairings including MXFP4-mixed. The patterns manifest `qwen_moe_gguf_codebook_patterns_v1.json` has MXFP4 variants; per instruction the tuning corpus is untouched. |
| Reference correctness (OCP E2M1 x 2^(e-128), E8M0 edges, zero marker, max normal, reject 0xFF; kernel vs reference per backend/ISA) | `V2_Unit_MXFP4Tensor` covers exponent totality (V2_Unit) and the E8M0 plane edges from PR #38. Kernel-vs-independent-reference for the NativeQuantised CPU path is covered by inventory sweeps; need a focused independent-witness test on parity with `IQ2SOriginalSourceEquationInFusedExpertTransaction` style oracles. |

## Hardcoded format lists without MXFP4 (23 files, three classes)

Class A: accuracy/isolation test bodies, add MXFP4.

- `tests/v2/unit/kernels/Test__FusedProjectionParity.cpp` [Q4_0, Q8_0, IQ4_NL]
- `tests/v2/unit/moe/Test__GPUExpertTransfer.cpp` [Q4_0, Q4_1, IQ3_S, Q2_K]
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp` [Q4_0, Q4_1, IQ4_NL, Q2_K, Q8_0]
- `tests/v2/integration/Test__VnniUnpackKernels.cpp` (all-format) - MXFP4 repack exists (`RepackFormat::MXFP4`, kernel 1b; `codebookIdToRepackFormat(25) -> MXFP4`); check `isReversibleFormat` before adding unpack cases.
- `tests/v2/integration/Test__IQ4XSSourceEquations.cpp` - IQ4_XS-specific, not applicable.
- `tests/v2/integration/loaders/Test__FPWeightPipeline.cpp` - floating pipeline, not applicable (no quantized codebook).

Class B: perf/tuning trainers, add MXFP4 (owner runs them; not accuracy gates but peer parity for economy evidence):

- `Perf__MoEVerifierPrefill.cpp` -> `sharedExpertPreparedFormatCases()` (primary gap for the task-named CUDA/ROCm M1234/M4M9M31 tests)
- `Perf__CPUMoEExpertVerifierRows.cpp`, `Perf__GPUExpertPipeline.cpp`, `Perf__ROCmMoEVerifierPrefill.cpp` (inventory loops already; verify shared cases lists)
- `Perf__AVX2vsAVX512.cpp`, `Perf__CPUNativeQuantised_Sweep.cpp`, `Perf__CPUNativeQuantised_GEMV.cpp` (MTP_SMALL_M_FORMATS), `Perf__CPUNativeQuantised_DispatchSweep.cpp`, `Perf__OneDNN_Reference.cpp`
- `Perf__CUDABlockwiseTensorCoreGemm.cpp` (+Sweep), `Perf__CUDANativeQuantisedDecodeTrainer.cpp`
- `Perf__BlockwiseNativeQuantised.cpp`, `Perf__NativeQuantised_GEMM.cpp`, `Microbench__IQDecodeLab.hip`, `Perf__IINT8Unpackable.cpp`

Class C: representativeness samples, not coverage claims; leave as-is (note only):

- `Test__PlacementStrategy_MPI_Integration.cpp`, `Test__BufferRole.cpp`, `Test__CollectiveContext.cpp`, `Test__KernelSnapshotInfo.cpp`, `Perf__CpuGemmHeuristicValidation.cpp`

## Fix plan (this branch)

1. Add MXFP4 to `sharedExpertPreparedFormatCases()` in `Perf__MoEVerifierPrefill.cpp`
   (and any other shared-case lists in the MoE verifier perf file).
2. Add MXFP4 to `Test__FusedProjectionParity` and `Test__GPUExpertTransfer`
   (unit, device-free runs here).
3. Add MXFP4 to the CPU perf trainers' label lists (factory resolver check per
   file; `TestTensorFactory::createMXFP4Random` exists).
4. Add MXFP4 to the ROCm/CUDA quantised-GEMM and blockwise trainer lists; the
   CUDA/ROCm runs happen on the Sparks/josh lanes.
5. Verify `isReversibleFormat`/unpack applicability for MXFP4; add unpack and
   ROCm quantised-GEMM cases if supported.
6. Add the focused independent-reference test (OCP E2M1 x 2^(e-128) oracle,
   E8M0 edges: zero marker, e=2, max normal e=252, reject 0xFF) for the
   NativeQuantised CPU kernel and register it in ProductionTestPreflight
   (repo maxim 22) once the harness shape is settled.
7. Build Integration CPU AVX512 + AVX2, run the touched suites with
   `ctest -FA '.*'`, then full Unit + ProductionTestPreflight label; classify
   against develop `0cc2c355c`.


## Accuracy-branch gate results (Integration, CPU-only AVX512, SHARED_CORE=ON, c477999c7/e09b8ce04)

- V2_Unit_*: 995/1005 passed, 10 failed.
- ProductionTestPreflight: 2619/2661 passed, 42 failed.
- Classification (all environment or pre-existing; the only branch-relevant defect found and fixed):
  - Container NUMA/seccomp (mbind/set_mempolicy EPERM, "failed to bind memory", MPI bind): CPUBackend, NUMAAllocator, ExpertGemmRegistry, Qwen35MoEGraph, PreparedEmbeddingWeightsLifecycle, VocabParallelEmbeddingSharding, CPUStartupAffinity, Qwen4ExpCPUContracts, CPU_NativeQuantisedFusedCompletion_*, CPU_NativeQuantisedWeightPackingAllFormats, CPUFlashAttentionAQ8NativeEntries/CausalRowOwnership, GroupedVerifierRows_CPU_Attention_NonQ16NativeFormats, FrontendAutomaticSelection_*_MPI, PlanCommandPublication_MPI, MTPSpecCPUBatchedVerifierPlans, NativeEmbeddingPreparedOwnership, TensorFactoryAggregate/BoundNUMAPolicy, CPUJoint* (first-touch placement), CPUFinalEmbeddingStorage, CPURootedGatherDegree8.
  - CUDA-only tree, tests needing CUDA/multi-GPU: CUDAEventSynchronization_MultiDevice, HostReleaseAfterGpuUpload_MultiDevice (+ GpuWorkspaceAllocationPolicy).
  - /proc/self/io absent (WSL kernel): ModelLoaderRowSlice.
  - Pre-existing base (#38/develop) failure: Qwen4ExpPrimitives (NaN/Inf verifier), ShmemSpinBackend_NP8 (MPI slots).
  - **Fixed in e09b8ce04:** CPU_NativeQuantisedNibbleDotILP_AVX2/AVX512 (nibble factory count 5 -> 6; MXFP4 joined quantizedVerifierFormats in #38 and the lifecycle loops all six formats; the count expectation was stale).
