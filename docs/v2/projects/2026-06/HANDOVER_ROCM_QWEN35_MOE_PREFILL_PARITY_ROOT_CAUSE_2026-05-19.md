# Handover: ROCm Qwen35 MoE Prefill Parity Root-Cause Investigation

Date: 2026-05-19
Workspace: `/workspaces/llaminar`
Primary model: `/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Primary failing path: single-device ROCm Qwen3.5 35B MoE prefill parity
Previous handover: `docs/v2/projects/2026-06/HANDOVER_ROCM_MOE_NATIVE_VNNI_FORMAT_SUPPORT_2026-05-19.md`

## Executive Summary

This document continues the investigation after the native-VNNI format-support handover. The earlier handover fixed MoE native-VNNI format admission and mixed `Q4_K` gate/up plus `Q5_K` down dispatch, but the real Qwen35 ROCm prefill parity failure remains.

The new stage-dump evidence sharply narrows the failure:

- CPU Qwen35 MoE prefill parity passes.
- ROCm Qwen35 MoE prefill parity fails at layer 0 `MOE_EXPERT_OUTPUT`.
- CPU and ROCm routed MoE inputs are close to PyTorch before the expert FFN.
- CPU routed expert output matches PyTorch.
- ROCm routed expert output is essentially uncorrelated with both CPU and PyTorch.
- The expert stage receives exactly the router stage outputs for routing indices/weights.
- Row permutations, sign flips, reversed columns, and half swaps do not explain the ROCm output.

Current best conclusion: the root cause is inside the ROCm routed `MOE_EXPERT_FFN` execution for real prefill. The next split is between:

1. the new ROCm grouped prefill pipeline being wrong for the real model path, and
2. ROCm packed expert GEMMs or real-model expert weight preparation being wrong before the grouped pipeline sees them.

The fastest next diagnostic is to temporarily re-enable the old per-expert ROCm prefill path as an A/B oracle. If old per-expert ROCm prefill matches CPU/PyTorch, the grouped prefill pipeline is the root. If it also fails, investigate real-model ROCm packed expert weights/GEMMs.

## Current Worktree Warning

The worktree is dirty from this and adjacent graph-capture work. Do not reset or revert unrelated files.

Relevant dirty files observed in this area include:

- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.h`
- `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip`
- `src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/ROCmWeightPacker.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp`

Also present are unrelated graph-capture and orchestration changes. Always check:

```bash
git status --short
```

## Stage Dumps Captured

ROCm layer-0 routed MoE dumps were captured with:

```bash
env LLAMINAR_STAGE_DUMP_ENABLED=1 \
    LLAMINAR_STAGE_DUMP_DIR=/tmp/llaminar_stage_dumps_qwen35_rocm_l0 \
    LLAMINAR_STAGE_DUMP_NAMES=moe_routing,moe_expert_ffn \
    LLAMINAR_STAGE_DUMP_LAYERS=0 \
    LLAMINAR_STAGE_DUMP_INPUTS=1 \
    LLAMINAR_STAGE_DUMP_OUTPUTS=1 \
    LLAMINAR_STAGE_DUMP_WEIGHTS=0 \
    LLAMINAR_STAGE_DUMP_ASYNC=0 \
    LLAMINAR_STAGE_DUMP_MAX=8 \
    ctest --test-dir build_v2_integration \
    -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
    --output-on-failure --parallel
```

Result: failed as expected in about 47 seconds.

CPU layer-0 routed MoE dumps were captured with:

```bash
env LLAMINAR_STAGE_DUMP_ENABLED=1 \
    LLAMINAR_STAGE_DUMP_DIR=/tmp/llaminar_stage_dumps_qwen35_cpu_l0 \
    LLAMINAR_STAGE_DUMP_NAMES=moe_routing,moe_expert_ffn \
    LLAMINAR_STAGE_DUMP_LAYERS=0 \
    LLAMINAR_STAGE_DUMP_INPUTS=1 \
    LLAMINAR_STAGE_DUMP_OUTPUTS=1 \
    LLAMINAR_STAGE_DUMP_WEIGHTS=0 \
    LLAMINAR_STAGE_DUMP_ASYNC=0 \
    LLAMINAR_STAGE_DUMP_MAX=8 \
    ctest --test-dir build_v2_integration \
    -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_CPU_KV_FP16$' \
    --output-on-failure --parallel
```

Result: passed in `367.09 sec`.

The relevant dump folders still existed at handover time:

```text
/tmp/llaminar_stage_dumps_qwen35_rocm_l0/
/tmp/llaminar_stage_dumps_qwen35_cpu_l0/
./pytorch_qwen35_moe_snapshots/
```

Important dump subdirectories:

```text
stage_0000_MOE_ROUTER_layer0_moe_routing_rank0/
stage_0000_MOE_EXPERT_FFN_layer0_moe_expert_ffn_rank0/
```

`LLAMINAR_STAGE_DUMP_NAMES` uses substring matching. Exact graph stage names for Qwen35 layer 0 are:

```text
layer0_moe_routing
layer0_moe_expert_ffn
```

`MoEExpertComputeStage::buildDumpInfoImpl()` dumps:

- `input`
- `routing_indices`
- `routing_weights`
- `output`
- scalar params

The dump framework syncs stage inputs to host through `ensureOnHost()` and outputs through `ensureOutputsOnHost()`, so these dumps are useful for CPU/ROCm/PyTorch comparisons.

## Numeric Evidence

### ROCm Dump vs PyTorch Snapshots

Layer-0 ROCm pre-expert tensors are close:

```text
expert input vs PyTorch FFN_NORM:
  cosine=0.999714780 rel_l2=0.023886429 max_abs=0.222826

router logits vs PyTorch:
  cosine=0.999771480 rel_l2=0.022065100 max_abs=0.00252018

routing weights vs PyTorch:
  cosine=0.999829540 rel_l2=0.018551732 max_abs=0.0113036
```

Layer-0 ROCm routed expert output is not close:

```text
expert output vs PyTorch:
  cosine=0.029062917 rel_l2=1.322280419 max_abs=0.0760929
  rocm_mean=-0.0027869 pytorch_mean=0.0000782
```

The expert stage routing inputs equal the router stage outputs exactly:

```text
routing_indices: np.array_equal == true
routing_weights: max diff == 0
```

ROCm routing exactness for this dump:

```text
exact slots vs PyTorch: 59/72
top1 exact vs PyTorch: 9/9
```

### CPU Dump vs PyTorch And ROCm

CPU reaches the expert stage with close inputs and produces the correct output:

```text
CPU expert input vs PyTorch:
  cosine=0.999863082 rel_l2=0.016547725 max_abs=0.0814465

CPU expert output vs PyTorch:
  cosine=0.999876722 rel_l2=0.015760074 max_abs=0.000637734
  cpu_mean=0.00007699 pytorch_mean=0.00007823

CPU routing weights vs PyTorch:
  cosine=0.999954852 rel_l2=0.009565727
```

ROCm and CPU inputs/routes are also close, but outputs are not:

```text
ROCm expert input vs CPU:
  cosine=0.999849270 rel_l2=0.017365505

ROCm routing weights vs CPU:
  cosine=0.999872521 rel_l2=0.015982680

ROCm expert output vs CPU:
  cosine=0.028990544 rel_l2=1.323197437
```

Routing exactness:

```text
CPU exact slots vs PyTorch: 66/72
ROCm exact slots vs PyTorch: 59/72
ROCm exact slots vs CPU:     64/72

CPU top1 vs PyTorch: 9/9
ROCm top1 vs PyTorch: 9/9
ROCm top1 vs CPU:     9/9
```

Interpretation: routing drift is real but too small to explain the complete routed expert-output collapse, especially because CPU routes are not perfectly identical to PyTorch either and still pass.

### Layer-0 Routing Tables

ROCm layer-0 routing indices:

```text
[[112 106 107 238 181  57 200  43]
 [109  41  65  24  86  29 136   2]
 [134 115 220 159  41   2 224  28]
 [ 13 131 139  47 151 159 192 207]
 [ 13 234 242 188 144 186 217 159]
 [ 96 187 242 117  52 133  27 250]
 [181 107  82 251  80  68  59 131]
 [109  41   2  24 177 131   5 225]
 [139  13 159  47 116  55 220 131]]
```

CPU layer-0 routing indices:

```text
[[112 106 107 238 181  57 200  43]
 [109  41  65  24  86  29 136   2]
 [134 115 220 159  41   2 224  28]
 [ 13 131 139  47 151 159 192 207]
 [ 13 234 242 188 144 186 217 159]
 [ 96 187 242 117  52 133  27 250]
 [181 107 251  82  68  59 255  99]
 [109  41   2  24 177 131   5 225]
 [139  13 159  47 220  55 116 131]]
```

PyTorch layer-0 routing indices:

```text
[[112 106 107 238  57 181 200  43]
 [109  41  65  24  86  29 136   2]
 [134 115 220 159  41   2  28 224]
 [ 13 131 139  47 151 159 192 207]
 [ 13 234 242 188 144 186 217 159]
 [ 96 187 242 117  52 133  27 250]
 [181 107 251  82  68  59 255  99]
 [109  41   2  24 177 131   5 225]
 [139  13 159  47  55 220 116 131]]
```

### Permutation Checks

Extra checks against CPU output did not find a simple layout/permutation bug:

- ROCm output rows have low cosine to all CPU rows.
- Best row-to-row cosine maxima are only about `0.03` to `0.08`.
- Reversed columns, swapped halves, and negation do not fix the output:

```text
reverse columns cosine ~= -0.0185
swap halves cosine      ~= -0.0089
negation cosine         ~= -0.0290
```

## Closed Or Lower-Priority Hypotheses

### Snapshot/Stage-Key Mismatch

The stage dumps directly compare the expert stage input/output tensors, not only parity CSV summaries. CPU dump output matches PyTorch, ROCm dump output does not. This strongly reduces the odds that the failure is just a parity snapshot naming issue.

### Router Output Feeding

The expert stage receives exactly the router stage outputs for indices and weights in the ROCm dump. There is no evidence that the stage is reading stale routing buffers.

### Routing Drift Alone

Routing slots differ slightly across CPU/ROCm/PyTorch, but:

- top-1 routes match all tokens,
- routing weights are close,
- CPU routes also differ from PyTorch in some slots and still pass,
- ROCm output is uncorrelated rather than mildly degraded.

Routing may still affect final numbers, but it is not the main cliff.

### Generic Native-VNNI Q4_K/Q5_K GEMV/GEMM Math

Broad ROCm native-VNNI tests passed against CPU references:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCm_NativeVNNI_GEMV$' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCm_NativeVNNI_GEMM$' \
  --output-on-failure --parallel
```

Results:

```text
V2_Integration_ROCm_NativeVNNI_GEMV passed
V2_Integration_ROCm_NativeVNNI_GEMM passed
```

This does not rule out real MoE packed-weight preparation, descriptor wiring, or grouped-pipeline bugs, but it makes a generic Q4_K/Q5_K native-VNNI math bug less likely.

## Existing Synthetic Coverage And Its Gap

The focused ROCm MoE suite passes:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCmMoEKernel$' \
  --output-on-failure --parallel
```

Relevant test:

```text
tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp
TEST(Test__ROCmMoEKernel, GroupedPrefill_Q4KGateUp_Q5KDownMatchesSequentialGemm)
```

That test uses:

- `seq_len=4`
- `d_model=2048`
- `intermediate=512`
- `num_experts=4`
- `top_k=2`
- `Q4_K` gate/up
- `Q5_K` down

It compares grouped prefill against sequential ROCm GEMMs. This is valuable, but it cannot catch:

- a bug common to both ROCm grouped and ROCm sequential paths,
- real GGUF weight-packing edge cases,
- full-model expert view metadata issues,
- high expert-id or 256-expert route-table issues unless expanded,
- differences between ROCm outputs and CPU/PyTorch references.

A temporary diagnostic expanded this synthetic path to 256 experts and still passed, which reduces but does not eliminate high-expert-id concerns.

## Key Code Map

### Routed Expert Stage

File:

```text
src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp
```

Important points:

- For ROCm prefill with `seq_len > 1`, `canUseFixedTopologyGroupedPrefill()` selects the grouped path when enabled.
- `executeFixedTopologyGroupedPrefill()` calls `prepareExpertGroupsAsync()` and then `executeGroupedPrefillPipeline()`.
- There is an old GPU per-expert prefill path below the grouped path.
- The old path uses `prepareExpertGroups`, `gatherExpertBatch`, per-expert `multiply_fused_tensor`, `fusedSwigluDown`, and `scatterExpertResults`.
- That old path is currently unreachable for ROCm prefill because the stage hard-fails if grouped prefill is unavailable:

```cpp
if (params_.device_id.is_rocm() && params_.seq_len > 1)
    return false;
```

This hard-fail is why `LLAMINAR_ROCM_MOE_GROUPED_PREFILL=0` alone does not provide an A/B oracle today.

### ROCm MoE Kernel

File:

```text
src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp
```

Important points:

- `prepareExpertGroupsAsync()` converts float indices to int, groups tokens by expert on device, and computes max expert count.
- `groupTokensByExpertDevice()` count/scan/scatter carries token index and routing weight together; ordering should not affect final weighted scatter.
- `executeGroupedPrefillPipeline()` validates descriptor tables, ensures scratch, zeros output, and calls `rocmMoE_grouped_prefill_pipeline(...)`.

### Grouped Prefill HIP Pipeline

File:

```text
src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip
```

Pipeline:

1. gather and quantize hidden states,
2. grouped gate/up GEMM,
3. SwiGLU and quantize,
4. down GEMM,
5. weighted scatter.

Prior inspection found no obvious linear-indexing mismatch. The GEMM indexing uses `linear = b * N + n`, matching the non-grouped native-VNNI GEMM layout.

### Expert Weight Service

File:

```text
src/v2/execution/moe/MoEExpertWeightService.cpp
```

Important points:

- `extractExpertViews()` handles GGUF 3D expert tensors with shape `[cols, rows, experts]`.
- It creates views of shape `{rows, cols}` with element offset `tensor_idx * rows * cols`.
- `prepareGemmEnginesGPU()` builds GPU-packed engines from each view via `view->raw_data()`, `quantizedViewRawBytes(*view)`, `LoadOrchestrator`, and `ROCmQuantisedGemmKernel`.
- CPU parity passes through the same expert view extraction, so raw slicing is less suspicious than GPU packing or grouped execution.

## Recommended Immediate Next Step

Add a diagnostic-only path to let ROCm `seq_len > 1` use the old per-expert GPU prefill implementation. Suggested shape:

- Add a temporary env-gated bypass in `MoEExpertComputeStage.cpp`, for example `LLAMINAR_ROCM_MOE_PREFILL_LEGACY_AB=1`.
- When set, skip `executeFixedTopologyGroupedPrefill()` and allow the existing old per-expert GPU prefill path to run.
- Do not permanently change production behavior without deciding whether the old path is safe and intended.

Then run ROCm parity again with layer-0 dumps:

```bash
env LLAMINAR_ROCM_MOE_PREFILL_LEGACY_AB=1 \
    LLAMINAR_STAGE_DUMP_ENABLED=1 \
    LLAMINAR_STAGE_DUMP_DIR=/tmp/llaminar_stage_dumps_qwen35_rocm_l0_legacy_ab \
    LLAMINAR_STAGE_DUMP_NAMES=moe_routing,moe_expert_ffn \
    LLAMINAR_STAGE_DUMP_LAYERS=0 \
    LLAMINAR_STAGE_DUMP_INPUTS=1 \
    LLAMINAR_STAGE_DUMP_OUTPUTS=1 \
    LLAMINAR_STAGE_DUMP_WEIGHTS=0 \
    LLAMINAR_STAGE_DUMP_ASYNC=0 \
    LLAMINAR_STAGE_DUMP_MAX=8 \
    ctest --test-dir build_v2_integration \
    -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
    --output-on-failure --parallel
```

Interpretation:

- If legacy ROCm prefill matches CPU/PyTorch, focus on the grouped pipeline internals.
- If legacy ROCm prefill also fails, focus on real-model ROCm packed expert weights or per-expert ROCm GEMM behavior.

## If Grouped Pipeline Is Confirmed Root

Instrument or test these in order:

1. Group arrays produced by `prepareExpertGroupsAsync()`:

```text
d_group_counts
d_group_offsets
d_group_token_indices
d_group_weights
```

2. Grouped pipeline internal scratch:

```text
d_prefill_hidden_q8_
d_prefill_gate_
d_prefill_up_
d_prefill_swiglu_q8_
d_prefill_down_out_
```

3. Per-expert slot counts and offsets for the exact failing layer-0 route table.

4. Weighted scatter accumulation, especially repeated experts and high expert IDs.

5. Gate/up/down descriptor-table selection for mixed `Q4_K` gate/up and `Q5_K` down.

A useful regression would be a stronger grouped-prefill test using:

- `num_experts=256`
- `top_k=8`
- `seq_len=9`
- the exact failing layer-0 route IDs above
- random or real-like `Q4_K`/`Q5_K` blocks
- CPU/PyTorch-style reference, not only sequential ROCm reference

## If Legacy Per-Expert ROCm Prefill Also Fails

Prioritize real-model weight and GEMM diagnostics:

1. Build a layer-0 expert replay that loads the actual GGUF expert tensors and routes from the dump.
2. Compare CPU reference, ROCm sequential expert GEMM path, and grouped prefill path.
3. Byte-compare CPU-packed vs GPU-packed native-VNNI payload/scales/mins for selected real layer-0 experts.
4. Start with experts from token 0 and token 6 because they include both common and divergent route slots:

```text
token 0: 112, 106, 107, 238, 181, 57, 200, 43
token 6 ROCm: 181, 107, 82, 251, 80, 68, 59, 131
token 6 CPU:  181, 107, 251, 82, 68, 59, 255, 99
```

Potential files:

- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp`
- `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_SingleDevice_Parity.cpp`
- `src/v2/loaders/ModelLoader.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/kernels/rocm/ROCmWeightPacker.cpp`
- `src/v2/kernels/rocm/repack/VnniRepackKernels.hip`

## Commands Worth Reusing

List Qwen35 parity tests:

```bash
ctest --test-dir build_v2_integration -N -R 'Qwen35MoE'
```

Run focused ROCm MoE suite:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCmMoEKernel$' \
  --output-on-failure --parallel
```

Run ROCm Qwen35 prefill parity:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure --parallel
```

Run CPU Qwen35 prefill parity:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_CPU_KV_FP16$' \
  --output-on-failure --parallel
```

Build the focused ROCm MoE target after HIP/template changes:

```bash
cmake --build build_v2_integration \
  --target v2_integration_rocm_moe_kernel \
  --parallel
```

## Final Mental Model For The Next Agent

Do not spend the next cycle debating whether layer 0 diverges before MoE. It does not, in any meaningful way. The cliff is inside ROCm routed expert computation.

Do not rely on the existing synthetic grouped-prefill test as proof of correctness against PyTorch. It proves grouped ROCm matches sequential ROCm for friendly synthetic inputs. The real failure needs either a legacy-path A/B or a real-model replay with CPU/PyTorch reference.

The most valuable next bit of information is binary:

```text
Does old per-expert ROCm prefill produce a correct layer-0 MOE_EXPERT_OUTPUT?
```

Answer that first. It will cut the search space roughly in half.
