# ROCm Blockwise Activation Quantization Sprint Plan

**Status**: Planned  
**Date**: 2026-03-06  
**Scope**: Replace row-wise activation quantization with blockwise activation quantization across ROCm GEMV/GEMM paths  
**Priority**: High — this is now a correctness and parity issue, not a cleanup-only improvement

---

## Summary

The current ROCm quantized GEMM/GEMV stack quantizes activations with **one symmetric scale per row** before dispatching decode or prefill kernels. That contract is shared by decode GEMV, prefill GEMM, fused multi-projection paths, and the scaling epilogues.

Recent parity debugging established that this contract is the source of the current Qwen2 ROCm prefill drift. The native ROCm `down_proj` kernel matched an exact host reference built from ROCm-quantized activations and native Q4_0 weight blocks with cosine `1.0`, while CPU only matched at `0.998601`. When the same host reference was switched to **32-element blockwise activation quantization**, CPU returned to cosine `1.0`.

That result changes the framing of the work:

- This is not a narrow arithmetic bug in `ROCmGemmKernel_native_VNNI.hip`.
- This is a **contract problem** in how activations are quantized and represented.
- The correct fix is a coordinated redesign of the ROCm activation-quantization pipeline, not more kernel-local patching.

---

## Findings

### Proven Root Cause

From `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp`:

- `rocm_vs_quantized_ref = 1.0`
- `cpu_vs_quantized_ref = 0.998601`
- `cpu_vs_blockwise_quantized_ref = 1.0`

Interpretation:

- ROCm native-VNNI GEMM is correctly implementing the current quantized contract.
- The failing contract is **row-wise activation quantization granularity**.
- At least for the investigated layer-0 Qwen2 `down_proj` prefill case, **blockwise activation quantization removes the observed parity loss**.

### Non-Findings That Matter

- The layer-0 `FFN_DOWN` defect is not explained by incorrect native Q4_0 decode.
- The ROCm kernel path is not silently using the fused QKV logic; `FFN_DOWN` goes through normal `GEMMStage -> ROCmQuantisedGemmKernel::multiply_tensor(...)`.
- `LLAMINAR_ROCM_FORCE_CK=1` is currently not a useful discriminator for this native-VNNI route because CK fallback expects VNNI source weights and later hits an unimplemented branch.

### Why This Needs a Sprint

Blockwise activation quantization is not a one-line kernel change. It alters:

- activation quantization kernel output layout
- activation scale semantics
- workspace sizing and buffer naming assumptions
- GEMV and GEMM kernel interfaces
- fused multi-projection reuse paths
- debug/profiling assumptions that currently describe `d_scales_A` as `[M]`
- CPU-side reference/testing utilities that assume row-wise quantization

---

## Current Architecture

### Current Contract

Today ROCm activation quantization is defined as:

- input: FP32 `A [M x K]`
- output: INT8 `A_q [M x K]`
- metadata: FP32 `d_scales_A [M]`
- policy: one symmetric scale per row

This contract is explicitly documented and implemented in:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/execution/local_execution/device/WorkspaceDescriptor.h`
- `src/v2/tensors/TensorClasses.h`

### Main ROCm Entry Points Affected

The ROCm quantized GEMM stack is orchestrated from:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

The main execution surfaces are:

- `multiply_tensor(...)`
  - M=1 decode path
  - M>1 prefill path
  - CK debug/fallback path
- `multiply_fused_tensor(...)`
  - reused activation quantization for QKV and Gate/Up multi-projection execution

Kernel-side quantization and scaling live in:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip`
  - `quantizeActivationsQ8_kernel_t`
  - `rocmQuantGemm_quantizeActivations(...)`
  - scaling epilogues that consume `d_scale_A`

Native decode kernels that currently consume row-scale activation metadata:

- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmGemmKernel_native_VNNI.hip`

INT8/VNNI fallback or alternate routes that also depend on row-scale activation metadata:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_INT8_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_CK.hip`

---

## Blast Radius

### 1. Activation Quantization Kernel Layer

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

Current assumption to remove:

- `d_scales_A` is a single FP32 scale per row

Required change:

- quantize activations per fixed K-block, likely 32 elements initially to match proven parity behavior
- output scale metadata as `[M x blocks_per_row]`
- update quantization launch heuristics and shared-memory staging to support blockwise writes efficiently

Likely new abstraction:

- `ActivationQuantLayout` or equivalent metadata object that describes:
  - block size
  - blocks per row
  - scale dtype
  - packed activation layout

### 2. GEMV Native-VNNI Decode Path

Primary file:

- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`

Current assumption to remove:

- one `scale_A[row]` is applied across all K blocks

Required change:

- consume per-block activation scales in the same block loop that already handles per-block weight scales
- define the activation block contract for all supported `NativeVNNIFormat` variants
- decide whether activation blocks are always 32 elements or format-dependent

Why this matters:

- decode is already block-aware on the weight side; activation quantization must become equally block-aware or parity will remain path-dependent

### 3. GEMM Native-VNNI Prefill Path

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/kernels/rocm/gemm/ROCmGemmKernel_native_VNNI.hip`

Current assumption to remove:

- prefill kernels receive `d_scales_A [M]`

Required change:

- rework the native GEMM inner contract so activation scaling is applied per K-block, not once per row
- preserve tile efficiency while loading scale metadata for multiple rows and multiple K-blocks
- ensure the K-tile/block geometry is compatible with activation quant block boundaries

This is the highest-risk technical area in the sprint.

### 4. INT8 VNNI and CK Compatibility Paths

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_INT8_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_CK.hip`

Current assumption to remove:

- epilogues accept row-scale activation metadata

Decision needed:

- either upgrade these paths to full blockwise activation support
- or explicitly treat them as debug/fallback paths with a temporary row-wise compatibility adapter

Recommendation:

- do not block the sprint on making CK elegant
- but do make fallback behavior explicit and testable, because today the CK debug path is incomplete for native-VNNI-prepared weights

### 5. Fused Multi-Projection Paths

Primary file:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

Affected entry point:

- `multiply_fused_tensor(...)`

Current assumption to remove:

- quantize activations once into shared `d_A_int8 + d_scales_A[row]`, then reuse that row-scale metadata for QKV / Gate / Up projections

Required change:

- quantize once into `d_A_int8 + d_scales_A_blockwise`
- allow every projection path to consume the new blockwise metadata without requantizing
- keep batched QKV and Gate/Up launches aligned with the new metadata layout

This is likely where the best performance recovery will come from, because the fused paths amortize quantization overhead across multiple projections.

### 6. Workspace and Buffer Contracts

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/execution/local_execution/device/WorkspaceDescriptor.h`

Current assumption to remove:

- `SCALES_A` is sized as `[M]`

Required change:

- resize activation-scale workspace as `[M x blocks_per_row]`
- audit `getWorkspaceRequirements(...)`, `bindWorkspace(...)`, and all workspace validation/logging
- rename or document buffers clearly enough to avoid stale “per-row scale” assumptions in future work

Expected side effect:

- workspace pressure increases with sequence length and K, especially in prefill

### 7. Graph and Stage-Level Surfaces

Primary files:

- `src/v2/execution/compute_stages/stages/GEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`
- `src/v2/execution/compute_stages/stages/LMHeadStage.cpp`
- `src/v2/models/qwen/Qwen2Graph.cpp`

Stage code does not directly perform quantization math, but it does encode expectations around:

- which kernel entrypoint is used
- whether multi-projection quantization can be shared
- which shapes are passed for prefill vs decode

The sprint should keep stage APIs stable if possible, but the implementation plan must test all stage families that rely on the ROCm quantized GEMM engine.

### 8. Tensor/CPU Reference Utilities and Documentation

Primary files:

- `src/v2/tensors/TensorClasses.h`
- `src/v2/tensors/FP32Tensor.cpp`
- `src/v2/tensors/TensorBase.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp`

Current assumption to remove:

- the canonical activation quantized representation is per-row scaled INT8

Required change:

- add reference helpers for blockwise activation quantization
- update comments/docs that currently describe activation quantization as row-wise
- keep CPU references capable of reproducing both old and new contracts during migration

---

## Proposed Technical Direction

### Activation Contract V2

Introduce a new ROCm activation quantization contract:

- activation values remain INT8
- activation scales become blockwise
- initial block size: `32` K-elements
- scale tensor layout: row-major over logical activation blocks

Proposed logical representation:

- `A_q`: `[M x K]` INT8
- `A_scale_blocks`: `[M x ceil(K / 32)]` FP32 or FP16

Initial recommendation:

- use block size 32 first because:
  - it already fixed the investigated parity case
  - it aligns with existing native-VNNI weight block granularity for common formats
  - it minimizes redesign ambiguity for decode/application logic

### Interface Goal

Prefer a named activation-quant metadata structure over further raw pointer growth.

Candidate direction:

```cpp
struct QuantizedActivationView {
    const int8_t* values;
    const void* scale_blocks;
    int rows;
    int cols;
    int block_size;
    int blocks_per_row;
    ActivationScaleType scale_type;
};
```

Even if the first sprint does not expose this fully through public interfaces, the internal ROCm code should converge toward this model rather than continuing to thread special-purpose `d_scales_A` pointers everywhere.

### Scale DType

Recommendation:

- start with FP32 scale blocks for correctness and migration simplicity
- evaluate FP16 scale blocks as a second step once parity and performance are stable

Reason:

- the current defect is correctness-sensitive
- introducing blockwise scale layout and scale-dtype compression simultaneously would make validation unnecessarily ambiguous

---

## Sprint Workstreams

### Workstream 1: Contract and Workspace Refactor

Deliverables:

- define the blockwise activation quantization contract
- update workspace sizing and buffer binding
- remove hard-coded documentation/comments that say “per-row scales” for ROCm activation paths

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/execution/local_execution/device/WorkspaceDescriptor.h`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.h`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip`

### Workstream 2: Quantization Kernel Upgrade

Deliverables:

- new blockwise quantization kernel or kernel family
- verified layout for activation scales
- perf-safe launch heuristics for prefill and decode shapes

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip`

### Workstream 3: Native GEMV Upgrade

Deliverables:

- native GEMV decode path consumes blockwise activation scales
- parity tests across supported native formats
- fused decode projections reuse blockwise metadata without requantization

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

### Workstream 4: Native GEMM Upgrade

Deliverables:

- native prefill GEMM consumes blockwise activation scales
- tile/block geometry reconciled with activation block boundaries
- layer-0 `down_proj` parity regression eliminated under strict checks

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmGemmKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

### Workstream 5: Compatibility and Fallbacks

Deliverables:

- explicit migration stance for CK and INT8-VNNI fallback routes
- either blockwise support or clearly isolated compatibility shims
- fix or document the unusable `LLAMINAR_ROCM_FORCE_CK` debug path for native-VNNI weights

Primary files:

- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_CK.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_INT8_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

### Workstream 6: Validation and Perf

Deliverables:

- new direct reference tests for blockwise activation quantization
- updated parity tests for Qwen2 prefill and decode
- benchmark comparison against row-wise baseline

Primary files:

- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp`
- parity and benchmark test suites as needed

---

## Milestone Plan

### Milestone 1: Contract Skeleton

Goal:

- introduce blockwise activation scale storage and workspace sizing without changing all compute paths yet

Exit criteria:

- code compiles
- row-wise path still works behind a compatibility shim or temporary adapter
- new unit/reference helpers exist for blockwise quantization

### Milestone 2: Decode Path First

Goal:

- convert native GEMV and fused projection decode paths to blockwise activation scales

Why first:

- smaller surface than prefill GEMM
- highest leverage for validating shared fused quantization semantics

Exit criteria:

- M=1 native-VNNI decode passes parity against blockwise reference
- fused QKV / Gate-Up paths still reuse a single quantization pass

### Milestone 3: Prefill GEMM Conversion

Goal:

- convert native M>1 prefill GEMM to blockwise activation scales

Exit criteria:

- investigated Qwen2 layer-0 `down_proj` regression is eliminated without loosening thresholds
- representative Qwen2 prefill parity suite improves or clears

### Milestone 4: Compatibility Closure

Goal:

- resolve or isolate CK and INT8 compatibility paths

Exit criteria:

- debug/fallback behavior is explicit
- no path silently assumes row-wise activation scales once blockwise mode is enabled

---

## Risks

### Risk 1: Prefill Performance Regression

Blockwise activation scales increase metadata traffic and inner-loop complexity.

Mitigation:

- start with correctness-first FP32 scale blocks
- benchmark prefill separately from decode
- use fused quantize-once paths to recover overhead for QKV and Gate/Up

### Risk 2: Workspace Growth

`SCALES_A` grows from `M` to `M * blocks_per_row`.

Mitigation:

- measure worst-case prefill memory use on target Qwen2 shapes
- consider FP16 scale blocks only after correctness is stable

### Risk 3: Partial Migration Bugs

The biggest danger is a mixed state where one path uses row-wise scales and another uses blockwise scales.

Mitigation:

- centralize activation quantization metadata construction
- make incompatible paths fail loudly during transition rather than silently reinterpreting metadata

### Risk 4: CK/Fallback Drift

The fallback path is already not a reliable debug discriminator.

Mitigation:

- explicitly decide whether CK remains a first-class parity path in this sprint
- if not, mark it as non-authoritative for native-VNNI debugging until compatibility work lands

---

## Validation Plan

### Correctness

Required:

- preserve the new direct host reference tests in `Test__ROCmQuantisedGemmKernel.cpp`
- add explicit blockwise activation reference helpers for decode and prefill
- keep a row-wise reference only as a migration comparator, not as the target contract

Must-pass scenarios:

- Qwen2 layer-0 `down_proj` prefill snapshot case
- fused QKV decode on real model inputs
- fused Gate/Up decode on real model inputs
- LM head / standalone GEMM stage sanity for `M=1` and `M>1`

### Parity

Required:

- rerun the failing Qwen2 ROCm parity scenarios under strict thresholds
- compare before/after cosine at stage and model level

Success condition:

- parity improves without loosening assertions

### Performance

Required:

- measure decode tok/s and prefill tok/s before and after
- break out quantization time separately from kernel time where possible

Guardrail:

- do not accept large prefill regressions just to fix one parity case without understanding the trade

---

## Recommended Scope Boundaries

### In Scope

- ROCm quantized GEMV and GEMM activation quantization contract
- native-VNNI decode and prefill paths
- fused ROCm multi-projection paths
- workspace and validation tooling required to support the new contract

### Out of Scope

- broad CPU activation quantization redesign
- CUDA parity work unless interface changes force a synchronized update
- unrelated native-VNNI format expansion work

Note:

Public interfaces shared across backends may still need mechanical updates. That is acceptable, but the sprint should remain ROCm-led and correctness-driven.

---

## Open Questions

1. Should activation block size be fixed at `32`, or should it vary by kernel family or quant format?
2. Should blockwise activation scales remain FP32 in v1 of the sprint, or is FP16 required to stay within workspace budgets?
3. Do we want CK and INT8-VNNI paths to become fully blockwise-aware in the same sprint, or do we explicitly demote them to compatibility/debug status first?
4. Should the long-term internal interface move to a named activation-quant view object rather than continuing to thread raw pointers through GEMM entrypoints?

---

## Recommendation

Approve this as a dedicated sprint and treat it as a **contract migration** with kernel work, not as a local parity patch.

The key reason is simple: the investigated failure already demonstrated that the current row-wise contract is materially less faithful than blockwise quantization, and ROCm’s native kernels are accurately implementing that weaker contract. Further local debugging will not fix the underlying issue.