# Qwen3.5 MoE Expert Overlay Phase 8A Audit

Date: 2026-05-10  
Branch: `feat/qwen35-moe`  
Verdict: Phase 5E implemented; Phase 8A blocked by current-host ROCm P2P precondition

## Scope

Phase 8A is the real V2 overlay parity unskip gate. It must prove real Qwen3.5 MoE overlay inference through the production graph path, not topology-only coverage, synthetic/model-light helpers, or CTest green with GTest skips.

Required gate:

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

## Gate Result

CTest reports the overlay parity filter as passing, but the four real prefill/decode parity bodies are still GTest-skipped on the current host. This is not Phase 8A acceptance.

Skipped parity bodies observed in `build_v2_integration/Testing/Temporary/LastTest.log`:

- `PrefillParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- `PrefillParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`
- `DecodeParity_CUDA1_SharedHot_ROCm2TP_Hot_CPU2NodeLocalTP_Cold`
- `DecodeParity_ROCm2TP_SharedHot_CPU2NodeLocalTP_Cold`

The current skip reason is an explicit hardware precondition emitted by `overlayRuntimeBlockers()` in `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ExpertOverlay_Parity.cpp`: prepared ROCm LocalTP execution requires bidirectional ROCm P2P among all LocalTP participants, and this host does not provide that for the tested `rocm_hot` / `rocm_shared_hot` domains.

Topology tests in this filter can pass, and non-root MPI ranks can legitimately skip topology assertion bodies, but that does not establish Phase 8A inference parity. No accepted LM head, MoE expert output, or combined-output parity metrics were produced by the skipped parity bodies.

## Historical Blocker Evidence

Before the Phase 5E update below, the skip guard was preventing Phase 8A from accidentally accepting the bridge executor as production parity.

- `Test__Qwen35MoE_ExpertOverlay_Parity.cpp`
  - `overlayPlanOnlyRuntimeBlockers()` reported accelerator LocalTP `TensorParallelExperts` tiers as blocked.
  - `setupPipeline()` recorded this blocker before creating the production inference runner.
  - `runOverlayPrefillParityBody()` and `runOverlayDecodeParityBody()` call `GTEST_SKIP()` when the synchronized blocker is present.
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
  - Accelerator LocalTP expert tiers lower to `MoEExpertOverlayLocalTPStage`.
  - The Phase 7A reducer is wired after overlay partials, and currently receives dense full-shape partial outputs from the graph path.
- `src/v2/execution/moe/MoEExpertOverlayLocalTPStage.cpp`
  - The stage delegates accelerator LocalTP expert execution to `MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts()`.
  - Sparse dispatch mode uses `MoEExpertTokenRowTransfer::gatherRows()` / `scatterAddRows()` bridge helpers.
- `src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp`
  - The bridge executor materialized non-FP32 expert weights through host-readable dequantization paths using tensor host data/views.
  - It built expert token lists from host routing tensors.
  - It computed expert partials with CPU loops, created per-participant FP32 partial tensors, uploaded those partials to participant devices, then used LocalTP allreduce and host-side accumulation.

That bridge behavior is now reserved for explicit HOST/mock fallback. Phase 8A still cannot be accepted on this machine because the hardware P2P precondition prevents running the prepared ROCm LocalTP parity bodies.

## Phase 7A Reducer Status

The Phase 7A reducer change is not the reason for the current skip. `MoEExpertParallelReduceStage` now requires caller-provisioned `Params::sparse_expansion_scratch` for sparse optimized partials, and the current Qwen3.5 graph path feeds dense partial outputs without `selected_rows` into the reducer.

If accelerator LocalTP execution is changed later to return compact sparse partials, the graph must also provision `sparse_expansion_scratch` through arena/stage-owned lifetime before enabling sparse selected-row partial infos in the continuation-device optimized reducer.

## Next Required Work

Phase 8A remains blocked on validation preconditions, not the old unconditional Phase 5E bridge guard:

- Rerun the parity gate on hardware with bidirectional ROCm P2P for the LocalTP participants.
- Keep routing, row transfer, expert partial buffers, and reduce scratch caller-provisioned or arena-backed; no hot-path allocations.
- Emit and inspect diagnostics proving non-zero assigned/executed expert work per active tier and per participant.
- Do not accept Phase 8A from topology-only CTests or GTest skips.

## Code Quality Risks Noticed

- The production prepared LocalTP path now reuses stage-owned gate/up/partial scratch and a caller-owned partial-view list; it does not create per-expert FP32 partial tensors in the hot path.
- The explicit HOST/mock bridge path still contains allocation-heavy host-dequant behavior and must not be used as real accelerator parity proof.
- Sparse row transfer still uses host-side gather/scatter compatibility helpers over caller-owned buffers; this remains a D2H/data-path risk for a later device-resident sparse transfer phase.
- CTest pass with GTest skip is currently easy to misread as acceptance.
- Future compact sparse partial outputs must wire `sparse_expansion_scratch` explicitly before entering `ContinuationDeviceOptimized` reduce.

## Phase 5E Implementation Update

The LocalTP overlay stage now carries domain-scoped prepared participant GEMM handles and stage-owned scratch buffers into `MoEExpertOverlayLocalTPExecutor`. Non-HOST accelerator LocalTP graph construction populates those handles from `ExpertGemmRegistry` for each active planned expert, participant device, layer, and role, and fails before execution if any gate/up/down engine is missing. The host-dequant bridge remains available for explicit HOST/mock fallback only.

The prepared path executes routed expert math through `ITensorGemm` gate/up/down engines and reuses stage-owned batch/gate/up/partial tensors, so it no longer creates per-expert FP32 partial tensors in the production hot path. Sparse row transfer now accepts caller-owned buffers; compatibility gather/scatter still performs host-side sparse compaction and should be replaced by a device-resident sparse transfer path in a later phase. A follow-up audit cleanup also moved participant partial-view scratch and GPU scratch allocation out of the per-expert loop and into stage construction.

Current remaining risk: the registry identifies engines by domain, device, layer, expert, and role, but it does not yet encode per-participant row/column shard metadata. The prepared path therefore requires complete engines for every participant and scales participant outputs before allreduce to avoid duplicate accumulation with the current full-view preparation. True gate/up intermediate-shard and down input-shard prepared handles remain a follow-up code-quality/performance item, not a reason to accept host-dequant bridge parity.

## Verified Test Results After Phase 5E Update

```bash
cmake --build build_v2_integration --parallel
```

Result: passed, including relink of `v2_test_qwen35moe_expert_overlay_graph`.

```bash
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlay|ExpertGemmRegistry|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
```

Result: 6/6 CTests passed.

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

Result: 2/2 CTests passed. The direct prepared ROCm LocalTP body was GTest-skipped because this host reports ROCm P2P access for 0 pairs.

```bash
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Result: 7/7 CTests passed. The four real prefill/decode parity bodies were still GTest-skipped with the hardware availability message requiring bidirectional ROCm P2P among LocalTP participants. Phase 8A is therefore resumed but not accepted on this host.