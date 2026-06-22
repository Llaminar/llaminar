# Handover: ROCm MoE Native-VNNI Format Support and Qwen35 Prefill Parity

Date: 2026-05-19
Workspace: `/workspaces/llaminar`
Primary model: `/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Primary failing path: single-device ROCm Qwen3.5 35B MoE prefill parity

## Executive Summary

The user's request was to continue debugging ROCm MoE correctness, with an explicit requirement that MoE kernels must support every quantized format supported by the underlying GEMM/GEMV kernels. The artificial format restrictions were removed from the ROCm MoE grouped native-VNNI paths, and focused ROCm MoE tests now pass for mixed `Q4_K` gate/up plus `Q5_K` down, including GPU-repacked 3D expert views.

The original real-model parity failure is not fully fixed. The remaining failure is still sharply localized to routed MoE expert output in the Qwen3.5 ROCm prefill path:

- CPU Qwen35 MoE prefill parity passes.
- ROCm Qwen35 MoE prefill parity fails from layer 0 at `MOE_EXPERT_OUTPUT`.
- Router, routing indices/weights, and shared expert output are close before the routed expert collapse.
- Focused synthetic ROCm MoE kernel regressions pass, including a temporary 256-expert diagnostic run.

## Current Worktree Warning

The worktree is very dirty from this and adjacent graph-capture work. Do not reset or revert unrelated files.

Files directly involved in this investigation:

- `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.h`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmMoEGroupedPrefillKernels.hip`
- `src/v2/kernels/rocm/ROCmWeightPacker.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp`

There are many other dirty files in the workspace, including graph-capture files and stage/orchestrator changes. Check `git status --short` before editing and keep changes scoped.

## Format-Support Changes Made

### Host Descriptor Admission

`ROCmMoEKernel.cpp` no longer limits grouped native-VNNI descriptor tables to the old subset. It should admit the native-VNNI codebook set used by GEMM/GEMV:

```text
0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19
```

Descriptor validation now needs to preserve these rules:

- `mins` required for codebooks `5,7,8,9,10,13,14,16,17`.
- `emins` required for `Q2_K`, codebook `10`.
- `Qwen3.5-35B-A3B-UD-Q4_K_XL` routed MoE uses `Q4_K` gate/up and `Q5_K` down.

### Decode-Side MoE Kernels

`ROCmGemvKernel_native_VNNI.hip` had grouped MoE decode entry points with six-format static assertions and switch cases. Those were broadened to match generic native-VNNI GEMV dispatch.

Keep an eye on IQ1 formats if later tests cover them. The generic GEMM path has special IQ1_M grid-base handling, while the MoE prefill path currently uses the shared IQ1 grid table plus correction logic. This did not affect the Qwen35 Q4_K/Q5_K model, but it remains worth a future direct IQ1_M regression.

### Prefill Grouped Pipeline Dispatch

`ROCmMoEGroupedPrefillKernels.hip` already had broad format machinery, but the host bridge was dispatching the down projection with the gate/up table codebook. That was fixed so gate/up and down dispatch independently.

This is important for Qwen35 MoE:

```text
blk.0.ffn_gate_exps.weight  Q4_K  [256, 512, 2048]
blk.0.ffn_up_exps.weight    Q4_K  [256, 512, 2048]
blk.0.ffn_down_exps.weight  Q5_K  [256, 2048, 512]

blk.0.ffn_gate_shexp.weight Q8_0  [512, 2048]
blk.0.ffn_up_shexp.weight   Q8_0  [512, 2048]
blk.0.ffn_down_shexp.weight Q8_0  [2048, 512]
```

## Tests And Commands Already Run

Use full parallelism. Do not add `-j` limits.

### Focused ROCm MoE Build/Test

```bash
cmake --build build_v2_integration \
  --config Integration \
  --parallel \
  --target v2_integration_rocm_moe_kernel

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_ROCmMoEKernel$' \
  --output-on-failure --parallel
```

Latest result: passed.

```text
V2_FetchModelsFixture ............ Passed
V2_Integration_ROCmMoEKernel ..... Passed
Total Test time (real) = 2.40 sec
```

The focused suite now covers:

- all-codebook descriptor admission,
- rejection when `Q2_K` descriptors lack `emins`,
- Q4_K grouped gate/up decode,
- Q5_K grouped down decode,
- mixed Q4_K gate/up plus Q5_K down grouped prefill,
- nonzero `dmin` K-quant min correction,
- GPU-repacked 3D expert views compared against host-packed ROCm GEMM reference.

A temporary diagnostic changed the mixed prefill regression from 4 experts to 256 experts; it passed in about 22 seconds and was then restored to 4 experts so the committed/final focused suite remains fast.

### CPU Qwen35 MoE Prefill Parity

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_CPU_KV_FP16$' \
  --output-on-failure --parallel \
  -O /tmp/qwen35moe_cpu_prefill_ctest.log
```

Latest result: passed.

```text
V2_FetchModelsFixture ... Passed
Qwen35MoE_35B_CPU_KV_FP16 prefill parity ... Passed
Total Test time (real) = 28.97 sec
```

### ROCm Qwen35 MoE Prefill Parity

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure --parallel \
  -O /tmp/qwen35moe_rocm_prefill_ctest_final.log
```

Latest result: failed.

Key layer-0 parity lines from the final log:

```text
Layer 0 ATTENTION_NORM cosine=1.000000
Layer 0 QKV_PROJECTION cosine=0.999997
Layer 0 GDN_Z_PROJECTION cosine=0.999996
Layer 0 GDN_DELTA_RULE_OUTPUT cosine=0.999929
Layer 0 GDN_NORM_GATE_OUTPUT cosine=0.999913
Layer 0 ATTENTION_OUTPUT cosine=0.999715
Layer 0 FFN_NORM cosine=0.999641
Layer 0 MOE_ROUTER_OUTPUT cosine=0.999551
Layer 0 MOE_ROUTING_INDICES routing_overlap=0.972222 top1_match=0.888889
Layer 0 MOE_ROUTING_WEIGHTS routing_overlap=0.994934 weight_l1=0.040023
Layer 0 MOE_EXPERT_OUTPUT cosine=0.028888
Layer 0 MOE_SHARED_EXPERT_OUTPUT cosine=0.999306
Layer 0 MOE_SHARED_GATE_OUTPUT cosine=0.999239
Layer 0 MOE_COMBINED_OUTPUT cosine=0.302463
```

Summary from the failed parity run:

```text
Early layers passed: 1/6
LM_HEAD KL divergence: 1.45346153 (threshold: 0.09)
LM_HEAD cosine: 0.554839
LM_HEAD Top-5: 40.0%
```

CSV output from that run:

```text
tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16/prefill_layers.csv
tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16/prefill_summary.csv
tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16/prefill_stages.csv
```

Layer-0 MoE stats from `prefill_stages.csv`:

```text
MOE_ROUTER_OUTPUT        cosine=0.999551 rel_l2=0.030816
MOE_EXPERT_OUTPUT        cosine=0.028888 rel_l2=1.323838 llaminar_mean=-0.002723 pytorch_mean=0.000078
MOE_SHARED_EXPERT_OUTPUT cosine=0.999306 rel_l2=0.037288
MOE_COMBINED_OUTPUT      cosine=0.302463 rel_l2=1.102897
```

## Important Observations

1. The old artificial format bottleneck is real and has been addressed. Do not reintroduce a smaller supported format set in MoE just because the failing model uses Q4_K/Q5_K.
2. CPU Qwen35 prefill parity passes, so the issue is ROCm-specific.
3. Shared expert output passes on ROCm. The collapse is in routed expert output only.
4. Focused synthetic ROCm MoE tests pass for the mixed Q4_K/Q5_K format combination, even through GPU-repacked 3D expert views and a 256-expert diagnostic.
5. Routing is close but not perfect: layer 0 top-1 routing match is `0.888889`, routing overlap is `0.972222`. The `MOE_EXPERT_OUTPUT` failure is much larger than that routing drift alone would normally explain.
6. `TestTensorFactory::createQ4_KRandom()` and `createQ5_KRandom()` default to `dmin = 0`, so they miss K-quant min-correction bugs unless tests explicitly inject nonzero `dmin`.

## Best Next Investigation Targets

### 1. Real-Model Layer-0 Expert Replay

Build a direct replay that loads layer-0 routed MoE expert weights from the actual GGUF and compares:

- CPU/preexisting reference path,
- ROCm sequential expert GEMM path,
- ROCm grouped prefill path.

The synthetic tests may not reproduce real GGUF K-quant scale/min distributions. A direct layer-0 replay should use the exact expert IDs and routing weights from the failing prefill run.

Useful files:

- `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_SingleDevice_Parity.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp`
- `src/v2/loaders/ModelLoader.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`

### 2. Stage Dump / Stage Output Print Around Layer 0 MoE

Use the stage dump or stage output print framework to capture layer-0 routed MoE inputs and outputs for CPU vs ROCm:

```bash
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=layer0_moe \
LLAMINAR_STAGE_DUMP_LAYERS=0 \
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure --parallel
```

Adjust `LLAMINAR_STAGE_DUMP_NAMES` after confirming exact stage names in the graph. `LLAMINAR_STAGE_OUTPUT_PRINT_STAGES` may be useful for quick row samples, but binary dumps are better for replay.

### 3. Compare Input Tensor Used By Grouped Gather

The synthetic grouped prefill test constructs its own input and routes. The real path uses `MoEExpertComputeStage` buffers. Verify that the grouped gather kernel receives the same FFN-norm hidden state that PyTorch and CPU parity use.

Targets:

- `MoEExpertComputeStage::execute()`
- `MoEExpertComputeStage::executeFixedTopologyGroupedPrefill()`
- `ROCmMoEKernel::prepareExpertGroupsAsync()`
- `moe_grouped_gather_quantize_prefill_kernel` in `ROCmMoEGroupedPrefillKernels.hip`

Hypothesis: a real graph buffer/coherence issue could feed stale or misaligned token rows into grouped prefill while synthetic tests remain clean.

### 4. Inspect Routing Index Semantics

Layer 0 routing is close but not exact. Confirm whether the ROCm routed expert output is compared against PyTorch routes or Llaminar routes in the parity harness. If PyTorch routes are used for reference while ROCm executes slightly different expert IDs, the expert output cosine can collapse even with correct kernels.

Targets:

- parity snapshot collection around `MOE_ROUTING_INDICES`, `MOE_ROUTING_WEIGHTS`, `MOE_EXPERT_OUTPUT`,
- `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp`,
- `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip`.

Suggested experiment: force the ROCm stage to consume PyTorch routing indices/weights for layer 0 in a replay-style test, or compute a reference expert output using the exact ROCm routing table from the run.

### 5. Real GGUF K-Quant Distribution Edge Cases

The synthetic K-quant tests now inject nonzero mins but still use friendly random blocks. Real GGUF K-quants may contain scale/min edge cases not covered by the factory.

Targets:

- `src/v2/kernels/rocm/repack/VnniRepackKernels.hip`
- `src/v2/kernels/rocm/ROCmWeightPacker.cpp`
- `src/v2/tensors/Q4_KTensor.cpp`
- `src/v2/tensors/Q5_KTensor.cpp`

Suggested experiment: add a small test that reads the first few real layer-0 expert blocks from the GGUF, packs them via CPU and ROCm GPU repack, and byte-compares payload/scales/mins.

## Hypotheses Ranked

1. Routing-reference mismatch in the parity harness or replay comparison. The routing top-1 match is not perfect, and `MOE_EXPERT_OUTPUT` is a sparse expert sum where even a small expert-ID mismatch can destroy cosine.
2. Real graph buffer/coherence issue in the ROCm grouped prefill path. Focused kernels pass, so the failure may be the tensor passed into gather/quantize or the output buffer lifecycle.
3. Real GGUF K-quant packing edge case. Synthetic blocks pass; direct real-block pack/decode comparison has not been done yet.
4. Expert view slicing/order mismatch only present in full model loading. A 256-expert synthetic 3D view test passed, but it did not use the actual GGUF tensor byte order or model loader slice metadata.
5. Remaining grouped prefill math issue for a route pattern not covered by synthetic tests. Less likely after the 256-expert diagnostic, but still possible if the real route histogram has repeated experts/token patterns not tested.

## Useful CTest Names

List relevant Qwen35 MoE tests:

```bash
ctest --test-dir build_v2_integration -N -R 'Qwen35MoE'
```

Known IDs from this workspace:

- `229`: CPU single-device prefill parity, currently passing.
- `230`: ROCm single-device prefill parity, currently failing.
- `231`: CPU single-device decode parity.
- `232`: ROCm single-device decode parity.
- `235-237`: local TP RCCL ROCm Qwen35 MoE parity/snapshot tests.

## Debugging Tips

- The terminal wrapper sometimes failed to retrieve long build/CTest output. Re-run the same build command or use `ctest -O /tmp/some.log` and inspect with `tail`/`rg`.
- Use `LLAMINAR_LOG_LEVEL=INFO` for parity unless you need detailed stage logs. Debug logs become large quickly.
- Keep `--no-mpi-bootstrap` out of normal parity/benchmark commands; use it only for direct profiling/debugging.
- If touching HIP template dispatch, rebuild `v2_integration_rocm_moe_kernel` to force the relevant instantiations.
- If adding tests around K-quants, make `dmin` nonzero for Q4_K/Q5_K synthetic tensors.

## Suggested Immediate Resume Plan

1. Re-run focused ROCm MoE suite to confirm the local state still passes.
2. Dump or replay layer-0 real-model MoE routed expert inputs/routes/outputs.
3. Determine whether `MOE_EXPERT_OUTPUT` is being compared under identical routing indices/weights.
4. If routing is identical, byte-compare CPU vs GPU native-VNNI packed payload/scales/mins for real layer-0 Q4_K/Q5_K expert weights.
5. If packing is identical, instrument grouped gather/quantize input and per-expert slot counts in `ROCmMoEGroupedPrefillKernels.hip`.
