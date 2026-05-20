# Handover: Qwen35 MoE ROCm Decode Parity Fix

Date: 2026-05-20
Workspace: `/workspaces/llaminar`
Branch/HEAD: `feat/qwen35-moe` at `ed903cf2` (`WiP: ROCm MoE tuning: prefill up to 1204tok/s on 35b moe model`)
Primary target: `DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
CTest target: `V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
Model: `/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

## Current Status

The decode parity failure is fixed in the current working tree. The exact default target now passes without isolation environment flags.

Final verified decode metrics from `tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16/decode_steps.csv`:

```text
step 0: cosine=0.982379 kl=0.057873 token_match=true passed=true
step 1: cosine=0.994668 kl=0.051543 token_match=true passed=true
step 2: cosine=0.992034 kl=0.040494 token_match=true passed=true
step 3: cosine=0.988236 kl=0.003233 token_match=true passed=true
step 4: cosine=0.994785 kl=0.005501 token_match=true passed=true
avg_cosine=0.990420
passed=5/5
```

Final passing CSV snapshot was preserved here:

```text
/tmp/qwen35_decode_parity_snapshots/final_passing_20260520_062736
```

## Root Causes Fixed

### 1. Grouped prefill mixed native-VNNI codebooks

The grouped prefill WIP must be kept, but it had a real mixed-format correctness bug. Qwen35 MoE can use different native-VNNI codebooks for gate/up and down, for example Q4_K gate/up with Q5_K down.

`ROCmMoEKernel.cpp` already declared and called `rocmMoE_grouped_prefill_pipeline()` with separate `gateup_codebook_id` and `down_codebook_id`, but the current unstaged HIP implementation in `ROCmMoEGroupedPrefillKernels.hip` had collapsed them into one `codebook_id`. That created an ABI/signature mismatch and also forced the down GEMM through the wrong codebook dispatch.

Fix applied:

- `src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip`
- Keep separate parameters: `gateup_codebook_id`, `down_codebook_id`
- Dispatch gate/up with `gateup_codebook_id`
- Dispatch down with `down_codebook_id`

Focused signal before fix:

```text
Test__ROCmMoEKernel.GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference
cosine=0.999670 rel_l2=0.543691 max_abs=36460.324219
FAILED
```

Focused signal after fix:

```text
Test__ROCmMoEKernel.GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference
cosine=0.999998 rel_l2=0.002108 max_abs=98.359375
PASSED
```

The full `V2_Integration_ROCmMoEKernel` target also passes now.

### 2. Snapshot/integration device-routed decode table consumption

Default Integration decode originally hit a hard all-zero verifier failure at `layer0_moe_expert_ffn`. The dumped tensors showed nonzero input, nonzero routing indices/weights, and all-zero output.

Cause: in snapshot-enabled Integration builds, `MoERoutingStage` keeps the legacy routing tensors authoritative for parity dumps and does not populate the ROCm runtime top-k table. `MoEExpertComputeStage` still tried the device-routed grouped decode path, consuming stale/zero runtime table data.

Fix applied:

- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`
- Under `ENABLE_PIPELINE_SNAPSHOTS`, force `can_try_device_routed_decode = false`
- Non-snapshot builds still keep the runtime-table device-routed decode path enabled when all existing conditions are satisfied

This removed the all-zero Integration failure and allowed the corrected grouped prefill mixed-codebook path to restore full decode parity.

## Regression/Isolation Added

Added a focused ROCm GDN recurrence check even though GDN was ultimately not the root cause:

- `tests/v2/unit/execution/Test__GDNKernels.cpp`
- Test: `Test__GDNKernels.ROCmPrefillMatchesSequentialDecodeQwen35Shape`
- It directly compares ROCm `chunk_forward()` against repeated ROCm `recurrent_step()` at Qwen35 dimensions (`n_heads=16`, `d_k=128`, `d_v=128`) to lock down prefill/decode recurrence state compatibility.

This test passes and documents that the suspected GDN prefill-state path was not the active parity regression.

## Commands Already Run

Builds:

```bash
cmake --build build_v2_integration --config Integration --target v2_integration_rocm_moe_kernel --parallel
cmake --build build_v2_integration --config Integration --parallel
cmake --build build_v2_integration --config Integration --target v2_test_gdn_kernels v2_integration_rocm_moe_kernel v2_integration_parity_qwen35moe_single_device --parallel
cmake --build build_v2_integration --config Integration --target v2_integration_parity_qwen35moe_single_device --parallel
```

Focused tests:

```bash
./build_v2_integration/tests/v2/v2_integration_rocm_moe_kernel \
  --gtest_filter='Test__ROCmMoEKernel.GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference'

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCmMoEKernel$' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Unit_GDNKernels$' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R '^(V2_Unit_GDNKernels|V2_Integration_ROCmMoEKernel)$' \
  --output-on-failure --parallel
```

Final parity verification:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 2
Test #232 ... Passed 88.69 sec
```

## Current Worktree State

The worktree is intentionally dirty and contains large staged WIP from adjacent ROCm MoE, prefill graph-capture, native-VNNI, and loader work. Do not reset, checkout, stash, or revert unrelated changes unless the user explicitly asks.

Current `git status --short` has many staged files. Files touched in this continuation are mixed staged/unstaged (`MM`) or unstaged:

```text
MM src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp
MM src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip
 M tests/v2/unit/execution/Test__GDNKernels.cpp
?? docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
```

There is also an unrelated unstaged edit in:

```text
MM tests/v2/integration/loaders/Test__UnifiedGPUPipeline.cpp
```

Do not attribute that loader diff to the decode parity fix without inspecting it. It predates or is adjacent to this work.

Unstaged diff stat at handover time:

```text
src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp   |   7 ++
src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip          |   7 +-
tests/v2/integration/loaders/Test__UnifiedGPUPipeline.cpp          |  34 ++++--
tests/v2/unit/execution/Test__GDNKernels.cpp                       | 132 +++++++++++++++++++++
4 files changed, 165 insertions(+), 15 deletions(-)
```

Staged diff stat remains large, about:

```text
37 files changed, 7748 insertions(+), 386 deletions(-)
```

## Important Files

Functional fixes from this continuation:

- `src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip`
  - Keep separate grouped prefill codebooks for gate/up and down.
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`
  - Disable device-routed decode table consumption in snapshot-enabled Integration builds.
- `tests/v2/unit/execution/Test__GDNKernels.cpp`
  - Adds ROCm prefill-vs-sequential-decode regression at Qwen35 dimensions.

Existing focused mixed-codebook regression that now passes:

- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp`
  - `GroupedPrefill_Qwen35RouteTable_Q4KQ5KMatchesCpuDequantReference`

## Recommended Next Steps

1. Review the three continuation diffs above and decide whether to stage them with the existing WIP or split them into a focused fix commit.

2. Re-run the final exact parity target if any further WIP edits touch ROCm MoE, native-VNNI packing, routing, or snapshot decode:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

3. Keep `V2_Integration_ROCmMoEKernel` as the fast guard for mixed native-VNNI codebook correctness:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCmMoEKernel$' \
  --output-on-failure --parallel
```

4. If continuing production/runtime graph-capture work, test the non-snapshot device-routed decode path separately. The Integration guard is correct because snapshot builds do not populate the runtime table, but production still relies on that runtime-table path.

5. Preserve CSV result directories before more parity experiments; `tests/v2/integration/parity/results/ed903cf2/...` is overwritten by every run.

## Watchouts

- The grouped prefill optimization is WIP and must be kept; fix correctness issues inside it rather than reverting it wholesale.
- Never collapse grouped prefill back to one `codebook_id`. Qwen35 MoE needs mixed gate/up and down native-VNNI formats.
- Do not confuse intra-run `cosine_drop` with current-versus-good regression deltas when comparing CSVs.
- The final decode parity fix depends on both parts: mixed-codebook grouped prefill dispatch and the snapshot-build device-routed decode guard.
- The build still emits a pre-existing GCC `StageBufferContract` inlining warning through `MoEExpertComputeStage::bufferContract()`. `get_errors` reported no editor diagnostics for the touched files.
- Follow project rules: use Ninja/full parallelism, and do not reset or revert user/WIP changes.

## Useful Memory Note

A repository memory was saved at:

```text
/memories/repo/qwen35-moe-rocm-mixed-codebooks.md
```

It records the invariant that Qwen35 MoE grouped prefill must keep separate gate/up and down codebook IDs, and that snapshot parity builds must not consume the runtime top-k decode table.
