# ARM CPU port: x86 SIMD source-surface inventory

Date: 2026-11-04. Base commit: `2cba3ca1d`; includes the uncommitted ARM foundation on
`codex/qfn-ngram-continuation-arm64`.

This is a **lexical candidate inventory, not a port-completeness certificate**.
It finds 85 production source files with x86 intrinsic headers/types/calls or ISA
selection markers. Guarded code, diagnostic or legacy helpers and comments may
match. Transitive includes, generated ISA policies, allocation/runtime code and
operation-level performance require separate compile/runtime proof.

The work packages and acceptance criteria are in
[the ARM CPU implementation plan](ARM_CPU_INFERENCE_KERNEL_PLAN.md).

## Reproduce the candidate inventory

```bash
rg -l '(#\s*include\s*[<"](?:[a-z0-9]*intrin|cpuid)\.h|_mm(?:512|256)?_[A-Za-z0-9_]+|__m(?:128|256|512)[id]?|__mmask[0-9]+|__builtin_cpu_(?:supports|init)|__AVX(?:2|512[A-Z0-9_]*)?__|_?__rdtsc|_xgetbv|target\("(?:avx|sse))' src/v2 \
  --glob '*.{h,hpp,cpp,inl,inc,cc,c}' | sort
```

## Source groups

| Production group | Matching files |
|---|---:|
| `src/v2/collective/backends/` | 2 |
| `src/v2/execution/` | 5 |
| `src/v2/kernels/cpu/attention/` | 4 |
| `src/v2/kernels/cpu/gdn/` | 4 |
| `src/v2/kernels/cpu/gemm/` | 13 |
| `src/v2/kernels/cpu/moe/` | 1 |
| `src/v2/kernels/cpu/ops/` | 1 |
| `src/v2/kernels/cpu/primitives/` | 12 |
| `src/v2/kernels/cpu/qwen4exp/` | 3 |
| `src/v2/kernels/cpu/rotation/` | 1 |
| `src/v2/kernels/cpu/sampling/` | 1 |
| `src/v2/kernels/cpu/simd/` | 1 |
| `src/v2/kernels/cpu/turboquant/` | 5 |
| `src/v2/tensors/` | 31 |
| `src/v2/utils/` | 1 |
| **Total** | **85** |

### `src/v2/collective/backends/`

Port spin/yield and publication fences without weakening timeout, ordering or retirement; this is transport infrastructure, not a compute-kernel shortcut.

- [`ShmemSpinBackend.cpp`](../../../../src/v2/collective/backends/ShmemSpinBackend.cpp)
- [`ShmemSpinBackend.h`](../../../../src/v2/collective/backends/ShmemSpinBackend.h)

### `src/v2/execution/`

Replace direct stage/transport intrinsics with typed ARM implementations of the existing operations. Preserve fused residual/norm, gating and KV publication; model graphs stay declarative.

- [`compute_stages/stages/AttentionOutputGateStage.cpp`](../../../../src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp)
- [`compute_stages/stages/FusedResidualNormStage.cpp`](../../../../src/v2/execution/compute_stages/stages/FusedResidualNormStage.cpp)
- [`compute_stages/stages/GatedRMSNormStage.cpp`](../../../../src/v2/execution/compute_stages/stages/GatedRMSNormStage.cpp)
- [`compute_stages/stages/KVCacheAppendStage.cpp`](../../../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp)
- [`moe/MoEOverlayNodeLocalRankBatchTransport.cpp`](../../../../src/v2/execution/moe/MoEOverlayNodeLocalRankBatchTransport.cpp)

### `src/v2/kernels/cpu/attention/`

Native Q/K dots and V accumulation for all KV formats, canonical 256-row summaries/merge order and sparse selection remain in scope. Anchored AQ8 encode/decode has native exact rational/reconstruction programs. TQ4/TQ8 centroid score/numerator leaves now use checked original borrows and native lookup/FMA, with independent byte/role/alias/worker proofs; eight unavailable-ISA fallback stubs were retired. Forward/transposed rotations now reuse native grouped-dot/AXPY with checked original operands and general fringe tests; these do not certify full attention/quantize binding or cache/request lifecycle. TQ/rotation resource/economy and matrix-content identity across standard-library ABIs remain open.

- [`CPUAttentionKeyQ8.h`](../../../../src/v2/kernels/cpu/attention/CPUAttentionKeyQ8.h)
- [`CPUFlashAttentionKernelT.h`](../../../../src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h)
- [`CPUQwenSparseAttention.cpp`](../../../../src/v2/kernels/cpu/attention/CPUQwenSparseAttention.cpp)
- [`TQFusedAttentionPrimitives.h`](../../../../src/v2/kernels/cpu/attention/TQFusedAttentionPrimitives.h)

### `src/v2/kernels/cpu/gdn/`

Native independent-value/query SIMD over the fixed eight-position checkpoint/raw-history program, plus short convolution and accepted-state publication.

- [`CPUFixedEpochGDN.h`](../../../../src/v2/kernels/cpu/gdn/CPUFixedEpochGDN.h)
- [`CPUFixedEpochQueryBlock.h`](../../../../src/v2/kernels/cpu/gdn/CPUFixedEpochQueryBlock.h)
- [`CPUGatedDeltaNet.cpp`](../../../../src/v2/kernels/cpu/gdn/CPUGatedDeltaNet.cpp)
- [`CPUShortConvolution.cpp`](../../../../src/v2/kernels/cpu/gdn/CPUShortConvolution.cpp)

### `src/v2/kernels/cpu/gemm/`

Quantized and floating decode/prefill, activation publication, six prepared encodings, source-factor reconstruction, rounded contribution boundaries, fused multi-projections and two-term Q8 residual correction.

- [`CPUNativeAVX2Gemv.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeAVX2Gemv.h)
- [`CPUNativeQuantisedActivationSums.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedActivationSums.h)
- [`CPUNativeQuantisedContributionContract.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedContributionContract.h)
- [`CPUNativeQuantisedFP16.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedFP16.h)
- [`CPUNativeQuantisedGemmKernel.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h)
- [`CPUNativeQuantisedGemv.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedGemv.h)
- [`CPUNativeQuantisedMultiScaleKernels.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h)
- [`CPUNativeQuantisedQ6Kernels.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedQ6Kernels.h)
- [`CPUNativeQuantisedWeightPacker.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h)
- [`CPUPreparedWeightWriter.cpp`](../../../../src/v2/kernels/cpu/gemm/CPUPreparedWeightWriter.cpp)
- [`CPUQ8ResidualCorrection.h`](../../../../src/v2/kernels/cpu/gemm/CPUQ8ResidualCorrection.h)
- [`FloatingPointGemmKernel.h`](../../../../src/v2/kernels/cpu/gemm/FloatingPointGemmKernel.h)
- [`VNNIEmulation.h`](../../../../src/v2/kernels/cpu/gemm/VNNIEmulation.h)

### `src/v2/kernels/cpu/moe/`

Preserve router-to-expert Q8 reuse, canonical route ordering and batched gate/up/SwiGLU/down transactions; no independently repeated per-row expert calls.

- [`CPUMoEKernel.cpp`](../../../../src/v2/kernels/cpu/moe/CPUMoEKernel.cpp)

### `src/v2/kernels/cpu/ops/`

Port the direct RMSNorm implementation as well as reusable primitives; a primitive-only pass cannot certify stage dispatch.

- [`CPURMSNormKernelT.cpp`](../../../../src/v2/kernels/cpu/ops/CPURMSNormKernelT.cpp)

### `src/v2/kernels/cpu/primitives/`

Native SIMD traits, FP math, normalization, softmax, RoPE and SwiGLU, including low-precision/internal paths. VectorPrimitives.cpp and the production SwiGLU translation unit have focused native NEON execution/byte checks (all SwiGLU storage routes and exact experts). RoPE floating pairs/inline fragments, Q8/Q16 in-place cycles, integer normalized/legacy/fixed/dynamic conversions and streaming hybrid FP32 now execute natively with independent arithmetic and original-row/ownership regressions; x86 equivalents and full stage/arena qualification remain open. Native four-storage softmax now has checked original row/batch geometry, fixed FP64 summaries, bounded SIMD publication, independent exact/accuracy/worker proofs and focused FP16 metadata/range/coordinate/alignment/retained-team regressions. Legacy x86 softmax cleanup, typed kernel/factory and economy remain open. SIMD traits and normalization are unfinished; this is not completion of the family or a serving/performance certificate.

- [`GPUAlignedExpertQ8Primitives.h`](../../../../src/v2/kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h)
- [`RMSNormPrimitives_detail.h`](../../../../src/v2/kernels/cpu/primitives/RMSNormPrimitives_detail.h)
- [`RMSNormPrimitives.cpp`](../../../../src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp)
- [`RoPEInline.h`](../../../../src/v2/kernels/cpu/primitives/RoPEInline.h)
- [`RoPEPrimitives_detail.h`](../../../../src/v2/kernels/cpu/primitives/RoPEPrimitives_detail.h)
- [`RoPEPrimitives.cpp`](../../../../src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp)
- [`RoPEPrimitives.h`](../../../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h)
- [`SimdTraits.h`](../../../../src/v2/kernels/cpu/primitives/SimdTraits.h)
- [`SoftmaxPrimitives_New.h`](../../../../src/v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h)
- [`SoftmaxPrimitivesImpl.h`](../../../../src/v2/kernels/cpu/primitives/SoftmaxPrimitivesImpl.h)
- [`SwiGLUPrimitives.cpp`](../../../../src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp)
- [`VectorPrimitives.cpp`](../../../../src/v2/kernels/cpu/primitives/VectorPrimitives.cpp)

### `src/v2/kernels/cpu/qwen4exp/`

Immutable codebook/source lookup, exact half/BF16 expansion, QFN-specific vector/control operations and their tails.

- [`CPUImmutableLookupKernels.cpp`](../../../../src/v2/kernels/cpu/qwen4exp/CPUImmutableLookupKernels.cpp)
- [`CPUQwen4ExpKernels.cpp`](../../../../src/v2/kernels/cpu/qwen4exp/CPUQwen4ExpKernels.cpp)
- [`CPUQwen4ExpSIMD.h`](../../../../src/v2/kernels/cpu/qwen4exp/CPUQwen4ExpSIMD.h)

### `src/v2/kernels/cpu/rotation/`

Fixed-order Hadamard/activation rotation and source preparation; preserve the explicitly selected rotation policy.

- [`ActivationRotation.h`](../../../../src/v2/kernels/cpu/rotation/ActivationRotation.h)

### `src/v2/kernels/cpu/sampling/`

Top-k/vector filtering and deterministic tie/scan order; retain stochastic sampler state and draw ownership.

- [`CPUSamplerPrimitives.cpp`](../../../../src/v2/kernels/cpu/sampling/CPUSamplerPrimitives.cpp)

### `src/v2/kernels/cpu/simd/`

Isolate genuine x86 helpers; provide native ARM equivalents only behind the operation contracts that consume them.

- [`AVX2Helpers.h`](../../../../src/v2/kernels/cpu/simd/AVX2Helpers.h)

### `src/v2/kernels/cpu/turboquant/`

Quantize/dequantize, sign/index/table handling and fixed-order rotations; certify codebook/tie behavior and persistent cache bytes.

- [`TurboQuantDequantizeTQ4.h`](../../../../src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h)
- [`TurboQuantDequantizeTQ8.h`](../../../../src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h)
- [`TurboQuantQuantizeTQ4.h`](../../../../src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h)
- [`TurboQuantQuantizeTQ8.h`](../../../../src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h)
- [`TurboQuantRotation.h`](../../../../src/v2/kernels/cpu/turboquant/TurboQuantRotation.h)

### `src/v2/tensors/`

Native decode/pack/conversion for every format and source-factor tuple. Audit AlignedVector and include-graph leaks too; removing intrinsics does not prove allocator or tensor portability.

- [`AlignedVector.h`](../../../../src/v2/tensors/AlignedVector.h)
- [`BF16Tensor.cpp`](../../../../src/v2/tensors/BF16Tensor.cpp)
- [`cpu/CPUTensors.h`](../../../../src/v2/tensors/cpu/CPUTensors.h)
- [`FP32Tensor.cpp`](../../../../src/v2/tensors/FP32Tensor.cpp)
- [`IQ1_MTensor.cpp`](../../../../src/v2/tensors/IQ1_MTensor.cpp)
- [`IQ1_STensor.cpp`](../../../../src/v2/tensors/IQ1_STensor.cpp)
- [`IQ2_STensor.cpp`](../../../../src/v2/tensors/IQ2_STensor.cpp)
- [`IQ2_XSTensor.cpp`](../../../../src/v2/tensors/IQ2_XSTensor.cpp)
- [`IQ2_XXSTensor.cpp`](../../../../src/v2/tensors/IQ2_XXSTensor.cpp)
- [`IQ3_STensor.cpp`](../../../../src/v2/tensors/IQ3_STensor.cpp)
- [`IQ3_XXSTensor.cpp`](../../../../src/v2/tensors/IQ3_XXSTensor.cpp)
- [`IQ4_NLTensor.cpp`](../../../../src/v2/tensors/IQ4_NLTensor.cpp)
- [`IQ4_XSTensor.cpp`](../../../../src/v2/tensors/IQ4_XSTensor.cpp)
- [`Q16_1Tensor.cpp`](../../../../src/v2/tensors/Q16_1Tensor.cpp)
- [`Q2_KTensor.cpp`](../../../../src/v2/tensors/Q2_KTensor.cpp)
- [`Q3_KTensor.cpp`](../../../../src/v2/tensors/Q3_KTensor.cpp)
- [`Q4_0Tensor.cpp`](../../../../src/v2/tensors/Q4_0Tensor.cpp)
- [`Q4_1Tensor.cpp`](../../../../src/v2/tensors/Q4_1Tensor.cpp)
- [`Q4_KTensor.cpp`](../../../../src/v2/tensors/Q4_KTensor.cpp)
- [`Q5_0Tensor.cpp`](../../../../src/v2/tensors/Q5_0Tensor.cpp)
- [`Q5_1Tensor.cpp`](../../../../src/v2/tensors/Q5_1Tensor.cpp)
- [`Q5_KTensor.cpp`](../../../../src/v2/tensors/Q5_KTensor.cpp)
- [`Q6_KTensor.cpp`](../../../../src/v2/tensors/Q6_KTensor.cpp)
- [`Q8_0Tensor_detail.h`](../../../../src/v2/tensors/Q8_0Tensor_detail.h)
- [`Q8_0Tensor.cpp`](../../../../src/v2/tensors/Q8_0Tensor.cpp)
- [`Q8_1Tensor_detail.h`](../../../../src/v2/tensors/Q8_1Tensor_detail.h)
- [`Q8_1Tensor.cpp`](../../../../src/v2/tensors/Q8_1Tensor.cpp)
- [`Q8_KTensor.cpp`](../../../../src/v2/tensors/Q8_KTensor.cpp)
- [`QuantizationUtils.h`](../../../../src/v2/tensors/QuantizationUtils.h)
- [`SIMDHelpers.h`](../../../../src/v2/tensors/SIMDHelpers.h)
- [`TensorClasses.h`](../../../../src/v2/tensors/TensorClasses.h)

### `src/v2/utils/`

Separate compile/runtime ISA identity from CPU architecture; replace CPUID/cache assumptions with explicit AArch64 and OS observations.

- [`CPUFeatures.h`](../../../../src/v2/utils/CPUFeatures.h)

## Important non-matches and transitive obligations

- `KernelFactory.cpp`, prepared weight/workspace contracts, `CPUPackedWeights.h`,
  `CPUNativeQuantisedPrefillSchedule.h` and the generated CPU decode/verifier tables
  are binding/dispatch dependencies even where they have no intrinsic calls.
- `CPUQ16AttentionMath.h`, `CPUAdjustedGDNReadout.h`,
  `Q16HeadNormalization.cpp`, shared numerical-contract headers and immutable
  source-scale helpers remain normative arithmetic dependencies.
- `NativeQuantisedFormatInfo.h` owns 21 source formats; the CPU Python registry has
  18 runtime codebook identities. Neither count equals six physical prepared
  encodings, and none is interchangeable with the other.
- Linux NUMA/sysfs, `MmapRegion.h`, MPI lifecycle/internal ABI, ELF loader tests,
  `NativeQuantisedProfilerControl.h`, CMake ISA loops and fixed x86 test flags require
  the separate OS/build/testing work packages. This list deliberately does not
  pretend that SIMD replacement alone makes a server build.
- Completion requires a target-native translation-unit/header closure build,
  supported-operation inventory and canonical tests. A zero-result source scan
  alone is neither a numerical proof nor a supported-backend admission API.
