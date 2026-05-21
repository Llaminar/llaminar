# Handover: Qwen35 MoE ROCm Decode Parity Continuation

Date: 2026-05-20
Workspace: `/workspaces/llaminar`
Branch/HEAD: `feat/qwen35-moe` at `ed903cf2`
Primary decode target: `DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
CTest decode target: `V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
Model: `/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

This document continues from:

```text
docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
```

## Current Status

The decode parity target still passes, but layer/stage cosine shows real remaining drift. The final decode tokens match, yet step 0 logit cosine is still only about `0.980268`.

Latest `decode_steps.csv`:

```text
step 0: cosine=0.980268 kl=0.0671053 token_match=true passed=true
step 1: cosine=0.994339 kl=0.0725424 token_match=true passed=true
step 2: cosine=0.990065 kl=0.0622723 token_match=true passed=true
step 3: cosine=0.986558 kl=0.00976627 token_match=true passed=true
step 4: cosine=0.994731 kl=0.00156709 token_match=true passed=true
```

The matching prefill parity target is clean:

```text
lm_head_cosine=0.994735 lm_head_kl=0.0191821 lm_head_top1=1 lm_head_top5=1 overall_passed=true
layer0 avg_cosine=0.999923 min_cosine=0.999811 worst_stage=MOE_SHARED_EXPERT_OUTPUT passed=true
```

## Changes Made In This Continuation

### 1. Native-VNNI GEMV split-K regression coverage

File changed:

```text
tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp
```

Added/changed:

- Added Q8_0 to the native-VNNI GEMV format sweep by using direct `packNativeVNNI()` for Q8_0 instead of the default INT8-VNNI packing path.
- Added `NativeVNNIGEMVTest.SplitK_KB8MatchesKB1_AllNativeCodebooks`.
- The new regression forces `KB=1`, then `KB=8`, on the same packed native-VNNI weights and compares outputs for all native codebooks.

Important result:

```text
Standalone ROCm native-VNNI GEMV split-K does not reproduce the decode corruption.
KB=8 vs KB=1 passed for Q4/Q5/K/IQ/Q8 native codebooks with cosine=1.0 and tiny rel_l2.
```

This means a future agent should not keep chasing the bare `rocmGemv_native_vnni_fp32()` split-K reduce path as the current layer-0 decode-cosine root cause unless new evidence appears.

### 2. GDN all-codebook projection regression coverage

File changed:

```text
tests/v2/unit/execution/Test__GDNKernels.cpp
```

Added/changed:

- Added `ROCmProjectionDecodeAllNativeCodebooksMatchesReference`.
- The test builds GDN-shaped fused decode projections (`qkv`, `z`, `alpha`, `beta`) for every supported native-VNNI codebook, including Q8_0.
- It forces `LLAMINAR_ROCM_NVNNI_GEMV_KB=8` and compares ROCm fused projection output against CPU dequantized reference.
- Uses cosine, relative L2, and scale-normalized max absolute error (`max_abs_ratio`) so low-bit formats with large-magnitude outputs do not fail spuriously.

Important result:

```text
All GDN-shaped projection codebook cases pass under forced KB=8.
Worst observed max_abs_ratio was below 0.007.
```

This covers the concern that the GDN path only worked for the specific Q4_K/Q5_K quantization used by the current model file.

### 3. Strengthened ROCm GDN prefill-to-decode state handoff regression

File changed:

```text
tests/v2/unit/execution/Test__GDNKernels.cpp
```

Updated existing test:

```text
Test__GDNKernels.ROCmPrefillMatchesSequentialDecodeQwen35Shape
```

The test now compares:

1. One ROCm kernel instance that runs prefill history and then one decode token.
2. A separate ROCm kernel instance that runs every token via sequential recurrent decode.

Important result:

```text
The synthetic ROCm prefill-history plus one decode-token handoff passes at Qwen35 dimensions.
```

So the remaining full-model decode issue is not a generic ROCm GDN hidden-state writeback failure in isolation.

## Focused Verification Already Run

Builds:

```bash
cmake --build build_v2_integration --config Integration --target v2_integration_rocm_native_vnni_gemv v2_test_gdn_kernels --parallel
cmake --build build_v2_integration --config Integration --target v2_test_gdn_kernels --parallel
```

Focused gtests:

```bash
./build_v2_integration/tests/v2/v2_integration_rocm_native_vnni_gemv \
  --gtest_filter='NativeVNNIGEMVTest.SplitK_KB8MatchesKB1_AllNativeCodebooks' \
  --gtest_brief=1

./build_v2_integration/tests/v2/v2_test_gdn_kernels \
  --gtest_filter='Test__GDNKernels.ROCmProjectionDecodeAllNativeCodebooksMatchesReference' \
  --gtest_brief=1

./build_v2_integration/tests/v2/v2_test_gdn_kernels \
  --gtest_filter='Test__GDNKernels.ROCmPrefillMatchesSequentialDecodeQwen35Shape' \
  --gtest_brief=1
```

Focused CTest:

```bash
ctest --test-dir build_v2_integration \
  -R '^(V2_Integration_ROCm_NativeVNNI_GEMV|V2_Unit_GDNKernels)$' \
  --output-on-failure --parallel
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

Full parity checks:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

Both passed.

Editor diagnostics:

```text
get_errors reported no errors for:
tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp
tests/v2/unit/execution/Test__GDNKernels.cpp
```

## Remaining Decode-Cosine Lead

Layer 0 decode step 0 now points away from projection GEMV and toward full-model decode state/context.

Fresh layer-0 decode-stage metrics:

```text
ATTENTION_NORM           cos=1.000000 drop=0.000000 rel_l2=0.000000 max_abs=7.62939e-06
QKV_PROJECTION           cos=0.999997 drop=0.000003 rel_l2=0.002553 max_abs=0.051897
GDN_Z_PROJECTION         cos=0.999997 drop=-0.000000 rel_l2=0.002361 max_abs=0.0293584
GDN_DELTA_RULE_OUTPUT    cos=0.983706 drop=0.016292 rel_l2=0.312449 max_abs=0.336776
GDN_NORM_GATE_OUTPUT     cos=0.981886 drop=0.001819 rel_l2=0.193637 max_abs=0.167211
ATTENTION_OUTPUT         cos=0.918513 drop=0.063374 rel_l2=0.453615 max_abs=0.0402975
FFN_NORM                 cos=0.891478 drop=0.027035 rel_l2=0.478803 max_abs=2.03131
MOE_ROUTER_OUTPUT        cos=0.952972 drop=-0.061494 rel_l2=0.319988 max_abs=0.0295928
MOE_ROUTING_INDICES      routing_overlap=0.875000 top1=0.000000
MOE_ROUTING_WEIGHTS      cos=0.989887 rel_l2=0.142273 max_abs=0.0462891
MOE_EXPERT_OUTPUT        cos=0.619589 rel_l2=0.871965 max_abs=0.0182032
MOE_SHARED_EXPERT_OUTPUT cos=0.831999 rel_l2=0.702424 max_abs=0.230407
MOE_SHARED_GATE_OUTPUT   cos=0.831999 rel_l2=0.705534 max_abs=0.0136185
MOE_COMBINED_OUTPUT      cos=0.702070 drop=0.129929 rel_l2=0.786980 max_abs=0.0193941
```

Interpretation:

- Projection accuracy is not the immediate issue: QKV and Z are both `0.999997`.
- The first meaningful decode drop is `GDN_DELTA_RULE_OUTPUT`.
- `ATTENTION_OUTPUT` then amplifies the drift.
- MoE routing diverges (`routing_overlap=0.875`, `top1=0`), after which expert output cosine becomes much worse.
- Prefill layer 0 does not show this problem: `GDN_DELTA_RULE_OUTPUT` and `GDN_NORM_GATE_OUTPUT` are essentially perfect during prefill.

## Recommended Next Steps

1. Investigate full-model decode state/context rather than standalone GEMV.

   The synthetic GDN prefill-to-decode handoff passes, but full-model decode step 0 drops at `GDN_DELTA_RULE_OUTPUT`. The next likely task is to compare the actual full-model recurrence state consumed by layer 0 at decode step 0 against the PyTorch/reference state or against a CPU Llaminar state produced by the same prefill.

2. Add a state dump or checksum around GDN recurrence state.

   Useful places:

   ```text
   src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp
   src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h
   src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip
   ```

   Compare state immediately after prefill and immediately before decode step 0. The stage output dumps show outputs, but not necessarily the hidden recurrence state.

3. Check whether parity harness prefill and decode use identical GDN layer state lifecycle on ROCm vs CPU/PyTorch.

   The full-model result suggests a difference that the isolated kernel test does not cover. Possibilities include layer-state object reuse, decode runner reset behavior, graph executor fast-decode coherence, or a mismatch in which Q/K/V/head offset path feeds decode.

4. Inspect attention after the GDN drop.

   `ATTENTION_OUTPUT` has the largest early layer-0 cosine drop after GDN. It may be amplifying a small state issue, or it may have its own decode-only issue once GDN output is slightly perturbed.

5. Treat MoE expert cosine as downstream until routing is stable.

   Layer-0 MoE expert output looks bad, but the router already selected a different top-1 expert. Fixing upstream GDN/attention/routing agreement should come before digging into expert math again.

## Current Worktree Notes

The worktree is intentionally dirty and contains staged WIP from adjacent ROCm MoE/native-VNNI/prefill graph-capture work. Do not reset, checkout, stash, or revert unrelated changes unless explicitly asked.

Focused status at handover time included:

```text
MM src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp
M  src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip
MM src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip
 M tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp
 M tests/v2/unit/execution/Test__GDNKernels.cpp
?? docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
```

Additional staged docs unrelated to this continuation were also present:

```text
A  docs/v2/HANDOVER_ROCM_MOE_NATIVE_VNNI_FORMAT_SUPPORT_2026-05-19.md
A  docs/v2/HANDOVER_ROCM_QWEN35_MOE_PREFILL_PARITY_ROOT_CAUSE_2026-05-19.md
A  docs/v2/PREFILL_HIP_GRAPH_CAPTURE_PLAN.md
```

Diff stat for this continuation's two regression test files before adding this handover document:

```text
tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp | 114 +++++-
tests/v2/unit/execution/Test__GDNKernels.cpp               | 393 +++++++++++++++++++++
2 files changed, 505 insertions(+), 2 deletions(-)
```

## Useful Memory Note

A repository memory was saved at:

```text
/memories/repo/qwen35-moe-rocm-decode-layer0.md
```

It records the current invariant: standalone native-VNNI GEMV split-K now passes all-codebook coverage, while the remaining decode loss starts in full-model layer-0 decode after clean prefill.