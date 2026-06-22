# CUDA Dual-Format Quantised GEMM Project Plan

## Goal
Give CUDA the same high-level two-family setup that ROCm already has:

- INT8-expanded GEMM/GEMV for 8-bit weight formats
- native-payload GEMM/GEMV for Q6 and lower formats, without forcing full INT8 expansion at rest

This plan is structural. It defines the packing split, cache shape, dispatch model, kernel families, and rollout phases needed to make CUDA match the ROCm dual-format story.

## Naming
At the project level this mirrors the ROCm split that has historically been described as:

- INT8-VNNI
- native-VNNI

For CUDA, the code should use CUDA-native names because the backend is not literally using VNNI instructions:

- `Int8Expanded`
- `NativePayload`

The conceptual split is the same as ROCm. The execution details are CUDA-specific.

## Current CUDA State

Today CUDA is structurally one-path-only.

- All quantized weight formats supported by `CUDAQuantisedGemmKernel` are funneled through a single `packWeightsToCUDA()` path.
- That path materializes FP32 via `tensor->data()`, then requantizes every column to symmetric INT8 with one FP32 scale per output column.
- The tensor-core path then repacks that INT8 matrix into a blocked tensor-core layout.

Relevant code:

- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.h`
- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel_CUTLASS.cu`
- `src/v2/kernels/cuda/CUDATensorCoreBlockwiseGemm.cu`

This means CUDA currently does not preserve the native on-device bitwidth of Q6/Q5/Q4/Q3/Q2/IQ* weights.

## Target CUDA State

CUDA should be split into two persistent packed-weight families.

### Family A: INT8-expanded

Use for:

- `Q8_0`
- `Q8_1`
- `Q8_K`
- any future weight family already natively 8-bit or intentionally promoted to 8-bit

Properties:

- weights are stored on device as INT8 plus per-column scales
- GEMV uses the existing DP4A-oriented decode path
- GEMM uses the existing tensor-core INT8 path
- host packing should avoid the current FP32 round-trip when an `IINT8Unpackable` fast path exists

### Family B: NativePayload

Use for:

- `Q6_K`
- `Q5_0`, `Q5_1`, `Q5_K`
- `Q4_0`, `Q4_1`, `Q4_K`, `IQ4_NL`, `IQ4_XS`
- `Q3_K`, `Q2_K`
- `IQ3_*`, `IQ2_*`, `IQ1_*`

Properties:

- weights remain in a compact native payload format on device
- per-block metadata stays separate on device: scales, mins, emins, codebook id, or equivalent format-specific side channels
- GEMV decodes native payload directly in-kernel
- GEMM decodes native payload tilewise during execution without requiring a permanent full-INT8 device copy

## Core Structural Decision

The CUDA split should happen before kernel selection, at the packed-weight/cache layer.

That is the same structural choice ROCm makes today.

The dispatch order should become:

1. Weight family selection: `Int8Expanded` vs `NativePayload`
2. Operation class: GEMV (`M=1`) vs GEMM (`M>1`)
3. Shape dispatch within that family

This is the correct level to split because it preserves the VRAM and bandwidth properties of the native formats. If we split only at kernel launch time but keep all CUDA weights cached as INT8, we would still lose the main benefit.

## Proposed Packed-Weight Model

`CUDAPackedWeights` should become a tagged union in spirit, even if implemented as one struct.

### Shared fields

- `K`, `N`
- `family`
- per-device upload cache
- source tensor back-reference

### INT8-expanded fields

- host `int8_data`
- host `scales`
- device `d_int8_data`
- device `d_scales`
- device `d_int8_data_tc_blocked`

### NativePayload fields

- host `native_payload`
- host `native_scales`
- host optional `native_mins`
- host optional `native_emins`
- host `native_codebook_id`
- host `native_blocks_per_row`
- device `d_native_payload`
- device `d_native_scales`
- device optional `d_native_mins`
- device optional `d_native_emins`

This gives CUDA the same persistent at-rest split ROCm already has.

## Proposed Packing Split

### New top-level packer behavior

`packWeightsToCUDA()` should stop being a single universal INT8 conversion path.

Instead it should do:

1. detect tensor type
2. route to `packInt8ExpandedCUDA()` for 8-bit families
3. route to `packNativePayloadCUDA()` for Q6-and-lower families

### INT8-expanded packing

Target behavior:

- match ROCm's INT8 family structurally
- prefer `IINT8Unpackable::requantizeRowToInt8()` or equivalent direct-format extraction
- avoid `tensor->data()` when a direct quantized-to-INT8 path exists

This still ends with:

- row or column-major INT8 storage suitable for current CUDA GEMM/GEMV kernels
- tensor-core blocked repack for GEMM

### Native-payload packing

Target behavior:

- use `vnniFormatInfo()` and `packVnniBlock()` style metadata already present in tensor classes
- produce a CUDA-native coalesced payload layout, not necessarily byte-identical to ROCm's layout
- keep the original payload density of the source format
- upload payload and side-channel arrays separately

Important constraint:

- the native CUDA layout should be designed around coalesced global loads and simple per-block decode on NVIDIA hardware, not copied blindly from the ROCm layout if CUDA wants a different interleave

## Proposed CUDA Kernel Families

### A. INT8-expanded GEMV

Status:

- mostly already exists today
- current DP4A family remains the implementation base

Role:

- decode path for 8-bit weights

### B. INT8-expanded GEMM

Status:

- already exists today
- current CUTLASS tensor-core family remains the implementation base

Role:

- prefill path for 8-bit weights

### C. NativePayload GEMV

New file family:

- `src/v2/kernels/cuda/CUDANativePayloadGemvKernels.cu`

Role:

- decode `M=1` path for Q6 and lower

Execution sketch:

- read native payload + per-block metadata from global memory in a coalesced layout
- decode one quant block at a time into register-resident signed INT8 fragments
- consume those fragments with DP4A
- apply per-block scale and min corrections inline into FP32 accumulation

This is the closest CUDA analogue to ROCm native-VNNI GEMV.

### D. NativePayload GEMM

New file family:

- `src/v2/kernels/cuda/CUDANativePayloadGemmKernels.cu`

Role:

- prefill `M>1` path for Q6 and lower

Execution sketch, phase 1:

- cooperative load native payload tiles into shared memory
- decode payload into register or shared-memory INT8 fragments on a block boundary
- compute with DP4A microkernels
- apply per-block scale and min corrections during accumulation

Execution sketch, phase 2:

- explore decode-to-fragment or decode-to-shared-memory staging that can feed tensor-core MMA efficiently
- only keep a tensor-core native-payload GEMM path if it beats the DP4A native-payload path without requiring a permanent full-INT8 weight copy

Important design rule:

- native-payload GEMM must preserve the at-rest memory advantage
- a temporary execution-time decode tile is acceptable
- a persistent full-INT8 mirror of the entire matrix is not

## Dispatch Matrix

The top-level CUDA GEMM/GEMV dispatch should become:

| Weight family | M=1 | M>1 |
|---|---|---|
| `Int8Expanded` | DP4A GEMV | tensor-core GEMM |
| `NativePayload` | native decode GEMV | native decode GEMM |

Fallback rules:

- unsupported native formats fall back to the existing INT8-expanded path behind a feature flag during rollout
- unsupported shapes within a native family fall back to the safe INT8-expanded path or CUTLASS path

## Proposed File Layout

### New or reshaped CUDA files

- `src/v2/kernels/cuda/CUDAWeightPacker.h`
- `src/v2/kernels/cuda/CUDAWeightPacker.cpp`
- `src/v2/kernels/cuda/CUDANativePayloadGemvKernels.cu`
- `src/v2/kernels/cuda/CUDANativePayloadGemmKernels.cu`

### Existing files to refactor

- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.h`
- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/CUDATensorCoreGemvKernels.cu`
- `src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu`
- `src/v2/kernels/cuda/CUDATensorCoreBlockwiseGemm.cu`

## Rollout Phases

### Phase 1: Cache split only

- add `Int8Expanded` and `NativePayload` families to `CUDAPackedWeights`
- move CUDA packing logic out of `CUDAQuantisedGemmKernel.cpp`
- preserve current execution behavior by allowing all families to fall back to the existing INT8-expanded kernels initially

Deliverable:

- CUDA can store native payloads separately, even before native CUDA kernels exist

### Phase 2: INT8-expanded cleanup

- replace FP32-roundtrip packing for 8-bit families with direct per-format extraction where possible
- keep existing DP4A GEMV and tensor-core GEMM execution

Deliverable:

- CUDA 8-bit path structurally matches ROCm's INT8 family

### Phase 3: NativePayload GEMV

- implement native decode GEMV for Q4_0 and IQ4_NL first
- extend to Q5/Q6/K-quant/IQ families through format traits and decode helpers

Deliverable:

- decode path no longer forces permanent INT8 expansion for native formats

### Phase 4: NativePayload GEMM

- implement native decode GEMM for prefill
- start with a correctness-first DP4A tile kernel
- add aspect/work dispatch within the native family only after baseline correctness and bandwidth wins are stable

Deliverable:

- CUDA prefill matches the ROCm dual-format architecture at a structural level

### Phase 5: Policy unification

- keep one family-level split shared by GEMV and GEMM
- allow each family to keep its own internal shape classifier
- expose tuning overrides for native-payload GEMM the same way the current specialized INT8 GEMM path does

## Validation Gates

### Correctness

- parity vs current CUDA INT8-expanded path for `Q8_*`
- parity vs CPU reference for `Q6_K` and lower
- layer-level cosine checks on decode and prefill
- no regression in bias handling, fused projections, or packed-weight cache reuse

### Performance

- verify native-payload GEMV reduces weight-byte traffic for Q6 and lower
- verify native-payload GEMM does not require a persistent full-INT8 weight mirror
- compare native-payload prefill against current INT8-expanded CUDA path and ROCm native path on representative Qwen shape classes

### Memory

- report at-rest VRAM for both families
- confirm native-payload weights keep their expected byte advantage over INT8-expanded storage

## Non-Goals

- forcing native-payload GEMM to use tensor cores in the first slice
- making CUDA's native layout byte-identical to ROCm's if a different CUDA interleave is better
- changing the activation-quantization contract; this plan is about weight families, not activation format redesign

## Recommended First Implementation Slice

The lowest-risk path is:

1. split `CUDAPackedWeights` into `Int8Expanded` vs `NativePayload`
2. move CUDA packing into a dedicated CUDA weight packer module
3. implement direct INT8-expanded packing for `Q8_*` without FP32 materialization
4. implement native-payload GEMV for `Q4_0` and `IQ4_NL`
5. add native-payload GEMM after the cache split and GEMV path are stable

That gets CUDA onto the same structural architecture as ROCm quickly, while keeping the harder native-payload GEMM work isolated behind the correct data model.