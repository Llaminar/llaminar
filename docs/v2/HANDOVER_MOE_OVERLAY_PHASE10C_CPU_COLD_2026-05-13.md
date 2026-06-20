# Handover: MoE Overlay Phase 10C CPU-Cold Dispatch

Date: 2026-05-13
Branch context: `feat/qwen35-moe`
Primary goal for next agent: close Phase 10C in one clean pass so Layout A can run end-to-end with real `cpu_cold` fallback contribution, then move to Phase 10D release benchmarking.

## Executive Summary

The current Phase 10C slice has converted the old CPU fallback boundary from "backend missing" into a graph-native MPI dispatch envelope path:

- root/continuation graph stages can publish ordered CPU-cold dispatch envelopes through `MoEOverlayMPIDispatchBackend`,
- non-root CPU fallback ranks consume those envelopes through an endpoint receive pump,
- routed envelopes now enter the existing `MoEExpertOverlayCPUFallbackStage` / `MoEExpertOverlayCPUFallback::runExpertFallback` path instead of failing at the envelope boundary,
- root forward success/failure sends `ForwardDone` / `Cancel` envelopes so endpoint ranks do not wait for a timeout.

The important architectural constraint remains: the endpoint receive pump is transport for a sparse collective. It must not own layer progress, walk MoE layers independently, or infer work outside dispatch envelopes emitted by continuation graph stages.

The next agent should finish the end-to-end CPU-cold bridge, remove the temporary compatibility/dead-code paths, then run the reduced Layout A probe. Do not move to Phase 10D benchmarking until `cpu_cold` either computes a real contribution or fails at a named compute/transfer bug after the backend is entered.

## Current Dirty Tree

Current relevant modified/untracked files from this Phase 10C slice:

- `docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md`
- `src/v2/CMakeLists.txt`
- `src/v2/execution/factory/InferenceRunnerFactory.h`
- `src/v2/execution/factory/InferenceRunnerFactory.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h`
- `src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp`
- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.h`
- `src/v2/execution/moe/MoEOverlayCPUFallbackParticipantRunner.cpp`
- `src/v2/execution/moe/MoEOverlayDomainRuntime.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `tests/v2/unit/execution/moe/Test__MoEOverlayDispatchCollective.cpp`

There may be unrelated dirty files from user or previous work. Do not revert unrelated changes.

## What Is Already Implemented

### Phase Plan Update

`docs/v2/MOE_EXPERT_OVERLAY_PRODUCTION_EXECUTION_PLAN.md` now contains Phase 10C: CPU-Cold Graph-Native Dispatch Backend. Existing release benchmarking was pushed to Phase 10D.

The language was tightened from imperative "commands" to dispatch messages/envelopes and endpoint receive pump terminology. Keep that wording. It reflects the desired graph-native sparse collective shape.

### MPI Dispatch Backend

New files:

- `src/v2/execution/moe/MoEOverlayMPIDispatchBackend.h`
- `src/v2/execution/moe/MoEOverlayMPIDispatchBackend.cpp`

Key pieces:

- `MoEOverlayMPIMessageKind`: `RoutedWork`, `NoOp`, `Cancel`, `ForwardDone`, `Shutdown`
- `MoEOverlayMPIDispatchHeader`: versioned fixed-size `int32_t` envelope with dispatch identity fields
- `MoEOverlayMPIDispatchBackend::dispatch(...)`: root-only publish path
- `receiveHeader(...)`: endpoint receive path
- `beginForward()`, `cancelBroadcastSinceForwardBegin()`: prevents double cancel envelopes when a dispatch-stage failure already broadcast cancel

Important: the backend currently transports the control envelope only. Tensor payload movement still goes through the existing CPU fallback helper (`MoEExpertOverlayCPUFallback`) after the routed envelope is published. That is intentional for the bridge.

### Backend Wiring

The backend is propagated through:

- `InferenceRunnerConfig::moe_overlay_dispatch_backend`
- `RankOrchestrator::Config::moe_overlay_dispatch_backend`
- `MoEOverlayDomainRuntime::Config::dispatch_backend`
- `MoEOverlayCPUFallbackParticipantRunner::Config::dispatch_backend`

`OrchestrationRunner::ensureMoEOverlayDispatchBackend(...)` creates the backend only when the resolved overlay execution plan includes non-root `CpuFallbackParticipant` endpoint ranks. This avoids stray dispatch envelopes for overlay topologies with no remote CPU fallback endpoint.

### Root Forward Lifecycle Signals

`OrchestrationRunner` now calls:

- `beginMoEOverlayForwardDispatch()` before root `runner_->forward(...)`,
- `signalMoEOverlayForwardDone()` after successful root forward,
- `signalMoEOverlayForwardCancel(...)` on root forward failure/exception unless a cancel was already broadcast by the dispatch backend.

This is required so endpoint ranks leave their per-forward receive pump without waiting for the full TP collective timeout.

### Endpoint Receive Pump

`MoEOverlayCPUFallbackParticipantRunner::runDispatchEndpointPump(...)` consumes dispatch envelopes during endpoint `forward(...)`.

Current behavior:

- `NoOp`: cheap log/continue
- `RoutedWork`: validate dispatch identity, resolve placement/domain/weights, then call `executeLayerFallback(...)`
- `Cancel`: return failure quickly
- `ForwardDone`: advance endpoint position and return success
- `Shutdown`: return success

`RoutedWork` does not walk all layers. It executes only the layer/tier named by the envelope.

### Root Runtime Dispatch + Compute

`MoEOverlayDomainRuntime::runDispatchBackend(...)` now:

1. builds the dispatch request from the descriptor,
2. publishes the routed/no-op/cancel envelope through the backend,
3. if the request was routed CPU fallback work, runs `runCPUFallbackGraphDispatch(...)` on the root side.

That root-side stage uses the existing CPU fallback domain context and transfer helper to broadcast transfer plans, transfer tensors, run `TensorParallelExperts`, reduce the CPU partial, and return data to the root rank.

## Verified So Far

These commands passed in the current slice:

```bash
cmake --build build_v2_integration --target v2_unit_moe_overlay_dispatch_collective --parallel

ctest --test-dir build_v2_integration \
  -R 'V2_Unit_.*(MoEOverlay.*Dispatch|MoEOverlayDomainRuntime|MoEOverlayCPUFallbackParticipantRunner)' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R 'V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts)' \
  --output-on-failure --parallel

git diff --check
```

VS Code diagnostics also reported no errors in the touched Phase 10C files.

## Remaining Phase 10C Gaps

### 1. Prove Reduced Layout A End-to-End

Run the reduced-context real benchmark probe before claiming Phase 10C acceptance:

```bash
cmake --build build_v2_release --parallel

LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_MOE_EP_TRACE=1 \
LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
   --config configs/qwen35-moe-overlay-layout-a.yaml \
   -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   -p Hello -n 0 -t 0 -c 512
```

Acceptance for this probe:

- not acceptable: `graph-native CPU fallback dispatch requires a dispatch backend`
- not acceptable: `overlay MPI routed-work payload transfer is not implemented yet`
- not acceptable: endpoint rank waits for `LLAMINAR_TP_COLLECT_TIMEOUT_MS` because no envelope arrived
- acceptable interim failure: named compute/transfer/coherence bug after `RoutedWork` envelope is logged and the CPU fallback stage is entered
- ideal: prefill completes with a real CPU-cold contribution and no endpoint hang

If the probe hangs in dense ROCm LocalTP allreduce before CPU-cold dispatch, use the Phase 10B diagnostic path once only:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_MOE_EP_TRACE=1 \
LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE=1 \
LLAMINAR_SKIP_ALLREDUCE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
   --config configs/qwen35-moe-overlay-layout-a.yaml \
   -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   -p Hello -n 0 -t 0 -c 512
```

Do not accept skipped allreduce as production behavior. It is only a site-isolation probe.

### 2. Replace the LocalTP Primary-Participant Bridge

Current bridge in `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`:

```cpp
const bool execute_cpu_fallback_on_this_local_tp_participant =
    !config_.tp_ctx || config_.tp_ctx->degree() <= 1 || config_.local_rank == 0;
```

This prevents duplicate same-rank MPI envelopes by letting only local TP rank 0 execute CPU fallback dispatch and giving other local TP participants a zero expert mask/no-work path. It is useful as a bridge but not the final graph-native contract.

Clean cut target:

- all local TP graph participants should enter the overlay dispatch stage in graph order,
- participants with no CPU-cold routed rows publish no-op locally,
- exactly one coherent remote CPU-cold envelope is emitted for the logical dispatch request,
- no local TP participant advances past failed/canceled/incomplete dispatch.

Likely implementation shape:

1. Add a small local rendezvous/aggregator wrapper around `MoEOverlayMPIDispatchBackend` for root local TP domains.
2. All local TP participants publish routed/no-op to that local rendezvous.
3. The executor/owner participant sends one MPI envelope to remote CPU-cold endpoints after the local participants have rendezvoused.
4. Non-owner local participants get the same collective result without touching MPI broadcast directly.
5. Remove the `execute_cpu_fallback_on_this_local_tp_participant` bridge and related zero-mask workaround once the aggregator is in place.

Be careful: if every local TP participant calls the MPI backend directly, remote endpoints will see duplicate same-rank envelopes. That was the architectural bug Phase 10A was designed to avoid.

### 3. Remove Legacy Endpoint All-Layer Fallback

The endpoint runner still has a compatibility branch guarded by `enable_native_compatibility_fallback`:

- `MoEOverlayCPUFallbackParticipantRunner::Config::enable_native_compatibility_fallback`
- the `forward(...)` path that loops all MoE layers independently when no dispatch backend exists

This is the old non-graph-native shape. Once dispatch endpoint execution is proven, remove this branch and the config flag. Endpoint ranks should require the dispatch backend.

Keep helper methods that are still needed by envelope execution:

- `placementForLayer(...)`
- `domainForName(...)`
- `expertMaskForTier(...)`
- `cachedLayerWeights(...)`
- `executeLayerFallback(...)`

Remove only the independent all-layer scheduler behavior.

### 4. Remove or Constrain Routed-Work Disabled Mode

`MoEOverlayMPIDispatchBackend::Config::routed_work_enabled` currently supports a test/protocol-only failure mode. Production `OrchestrationRunner` sets it to true.

After end-to-end routed execution is proven, either:

- remove `routed_work_enabled` and the `kUnsupportedRoutedWork` path, or
- keep it only as an explicit test-only constructor mode with no production caller.

Do not leave a production path that can regress to `overlay MPI routed-work payload transfer is not implemented yet`.

### 5. Make Metrics Honest

Phase 10C acceptance requires metrics for:

- no-op count,
- routed request count,
- selected row count,
- routed entry count,
- transfer bytes,
- remote endpoint work count,
- cancel count.

The dispatch backend currently records request-level metrics from the root envelope. Confirm that `remote_endpoint_work_count` is incremented when endpoint participants actually execute routed CPU fallback work, not merely when the root creates a local CPU domain context. If needed, add a root-visible status/ack path or fold the existing CPU fallback tensor-parallel stats into the runtime result.

### 6. Unblock Real Overlay Parity After Runtime Probe

The parity fixture still has Phase 10A-era runtime skips around real overlay dispatch, for example in `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ExpertOverlay_Parity.cpp`.

After the reduced Layout A probe works, update those skip messages or remove them if real inference can run. Do not make parity pass by topology-only assertions.

## Clean-Cut Implementation Plan

Recommended order for the next agent:

1. Build the current focused targets and rerun the two passing regexes to confirm the starting point.
2. Run the reduced Layout A probe and capture the first real failure.
3. If the failure is duplicate/missing endpoint envelopes, implement the local TP rendezvous/aggregator and remove the local-rank-0 bridge in `Qwen35MoEGraph.cpp`.
4. If the failure is tensor transfer/compute, fix the exact `MoEExpertOverlayCPUFallback` / `MoEExpertOverlayCPUFallbackStage` issue without adding a parallel scheduler.
5. Remove the endpoint all-layer compatibility branch and `enable_native_compatibility_fallback`.
6. Remove or test-gate `routed_work_enabled=false` and `kUnsupportedRoutedWork`.
7. Add/update tests proving routed envelope delivery, endpoint execution, cancellation, and no duplicate same-rank endpoint envelopes.
8. Rerun the full Phase 10C tests and the reduced Layout A probe.
9. Only then proceed to Phase 10D benchmarking.

## Required Validation Before Phase 10D

Minimum validation for closing Phase 10C:

```bash
cmake --build build_v2_integration --parallel

ctest --test-dir build_v2_integration \
  -R "V2_Unit_.*(MoEOverlay.*Dispatch|MoEOverlayDomainRuntime|MoEOverlayCPUFallbackParticipantRunner)" \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R "V2_Integration_.*MoEExpertOverlay_(CPUFallback|CPUTensorParallelExperts)" \
  --output-on-failure --parallel

cmake --build build_v2_release --parallel

LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_MOE_EP_TRACE=1 \
LLAMINAR_TP_COLLECTIVE_CONTRACT_TRACE=1 \
LLAMINAR_TP_COLLECT_TIMEOUT_MS=60000 \
./build_v2_release/llaminar2 benchmark \
   --config configs/qwen35-moe-overlay-layout-a.yaml \
   -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   -p Hello -n 0 -t 0 -c 512
```

Also run:

```bash
git diff --check
```

## Do Not Do These

- Do not revive a CPU fallback participant loop that independently walks all MoE layers.
- Do not express same-layer hot/warm/cold overlay as PP layer ownership.
- Do not let every local TP participant broadcast MPI dispatch envelopes directly to endpoint ranks.
- Do not accept `LLAMINAR_SKIP_ALLREDUCE=1` as production behavior.
- Do not claim Phase 10C acceptance from synthetic/model-light tests alone.
- Do not relax Qwen35 MoE parity thresholds to hide missing CPU-cold contribution.
- Do not remove endpoint cancellation/forward-completion signaling; without it, endpoint ranks can wait for the timeout after root failure.

## Acceptance Definition For This Handover

Phase 10C is done when:

- CPU-cold endpoint ranks consume only graph dispatch envelopes,
- routed CPU-cold work executes through the existing domain-scoped NodeLocalTP fallback compute path,
- root and endpoint ranks exchange ordered message identities without duplicates,
- endpoint ranks exit promptly on cancel and forward completion,
- the independent all-layer compatibility loop is gone,
- the reduced Layout A benchmark no longer fails at missing backend/payload/not-entered-dispatch boundaries,
- the remaining failure, if any, is a named compute/transfer issue with enough trace evidence to fix before Phase 10D.