# MoE Graph-Capture Grouped Prefill And Decode Design

Date: 2026-05-17

Scope: Qwen3.5 MoE on ROCm first, with an interface shape that can later map to CUDA and multi-participant expert-parallel overlay execution.

## Executive Summary

The path to llama.cpp-class MoE performance is not another small loop optimization inside `MoEExpertComputeStage`. The current system still treats routing, expert selection, and expert execution as host-visible stage decisions. Graph capture can only replay fixed launch topology, while expert rebalancing needs expert placement to remain mutable. The design target is therefore:

```text
stable captured graph topology
    + device-resident routing/grouping/workspace state
    + double-buffered device expert placement tables
    + ROCm kernels that read placement and descriptor tables at execution time
```

Graph-captured kernels must receive stable pointers. Rebalancing must update the contents behind those stable pointers between graph replays, not change graph nodes, kernel argument shapes, or per-expert host dispatch.

The single-device ROCm fast path should converge to:

```text
Decode MoE layer:
  FusedRouteSelectKernel -> FusedDecodeExpertKernel

Prefill MoE layer:
  RouteLogits/TopK -> DeviceGroupRoutes -> GroupedPrefillGateUp -> GroupedPrefillDownAccumulate
```

The multi-device overlay path uses the same routing/grouping and placement-table concepts, but sparse dispatch/return collectives remain explicit graph stages. Participants with no work no-op inside fixed topology.

## Current State And Observations

Recent measured state on Qwen3.5-35B UD-Q4_K_XL, ROCm:0:

- Release benchmark after grouped decode and router tuning: about `266.9 tok/s` prefill and `27.9 tok/s` decode.
- Decode profiling remains dominated by `MOE_ROUTER` at about 34% and `MOE_EXPERT_FFN` at about 23%.
- The hipBLAS decode-router experiment was slower than the custom single-token router path.
- The custom router path is still structurally expensive: it materializes router logits, emits top-k tensors, and uses separate follow-on expert kernels.
- Prefill already groups tokens on device, but `prepareExpertGroups()` D2H-copies counts/offsets and `MoEExpertComputeStage` loops experts on the host.
- `MoERoutingStage` and `MoEExpertComputeStage` currently return `isGraphCapturable() == false`.
- The current rebalancing controller can move or replicate experts, but updates are reflected through host masks, stage cached GEMM engines, and prepared weight stores, not through a stable device runtime table that captured kernels can query.

## Design Rules

1. Kernel launch topology must be fixed for a captured graph instance.
2. Expert activity, ownership, and residency must be device data, not host control flow.
3. Rebalancing may update descriptor-table contents between graph replays, but must not require graph node shape changes.
4. Kernel arguments captured by HIP graphs must point to stable allocation addresses.
5. Expert movement must be represented as a placement epoch update, not stage reconstruction in the hot path.
6. Debug/snapshot builds may perform D2H copies, but Release graph capture must not.
7. Kernels must fail loudly or no-op based on explicit invalid descriptors; they must never silently compute with stale expert data.

## Core Runtime Object: Device MoE Layer State

Add a model-owned, per-device, per-MoE-layer runtime state object. Graph stages and kernels receive only stable pointers to these objects.

```cpp
struct DeviceMoEExpertDescriptor {
    DeviceNativeVNNIMatrixDesc gate;
    DeviceNativeVNNIMatrixDesc up;
    DeviceNativeVNNIMatrixDesc down;
    int logical_expert_id;
    int owner_participant;
    int local_slot;
    uint32_t flags; // resident, replicated, preferred_owner, valid
};

struct DeviceMoEPlacementBank {
    DeviceMoEExpertDescriptor experts[MAX_EXPERTS];
    uint8_t local_compute_mask[MAX_EXPERTS];
    uint8_t replica_role[MAX_EXPERTS];
    uint32_t epoch;
};

struct DeviceMoELayerRuntime {
    uint32_t active_bank;
    uint32_t active_epoch;
    DeviceMoEPlacementBank banks[2];

    // Decode workspace, fixed by model top_k/dims.
    int topk_expert_ids[MAX_TOP_K];
    float topk_weights[MAX_TOP_K];
    uint64_t decode_histogram[MAX_EXPERTS];

    // Prefill workspace, fixed by current graph seq_len capacity.
    int *route_expert_ids;       // [seq_len * top_k]
    float *route_weights;        // [seq_len * top_k]
    int *expert_counts;          // [num_experts]
    int *expert_offsets;         // [num_experts]
    int *grouped_token_ids;      // [seq_len * top_k]
    float *grouped_route_weights;// [seq_len * top_k]
};
```

The exact storage can be C++ wrappers over HIP allocations, but the important contract is stable pointer identity. Captured kernels get `DeviceMoELayerRuntime*`. Rebalance updates inactive bank contents and flips `active_bank` on stream after all new descriptors are ready.

## Graph-Compatible Rebalancing Protocol

Rebalancing must happen between decode graph replays or between request phases.

1. The device router records decode histograms into `DeviceMoELayerRuntime::decode_histogram` during captured execution.
2. At the controller window boundary, host copies histograms D2H outside the captured graph segment.
3. `MoERebalanceController` computes new owner/replica placement.
4. `ExpertWeightTransfer` moves prepacked weights as today, but all arrivals are prepared before the epoch flip.
5. New gate/up/down descriptors are exported into the inactive placement bank.
6. A table-update command copies masks, owner metadata, and `active_bank/active_epoch` on the same stream used by the next graph replay, or records an event that the replay stream waits on.
7. The old bank and departed expert device allocations are retired only after a graph-replay completion event proves no captured kernel still references the previous epoch.

This allows experts to move without recapture because the graph captures a pointer to `DeviceMoELayerRuntime`, not individual expert pointers. The values behind that pointer change by epoch.

### Required Rebalance Invariants

- `num_experts`, `top_k`, `d_model`, and `intermediate` are graph-shape constants.
- Descriptor banks are preallocated for maximum experts, not resized during serving.
- Weight device buffers for arrivals are allocated and populated outside graph capture.
- A descriptor is valid only if all gate/up/down payload pointers and codebook metadata are ready.
- Kernels check `local_compute_mask[expert]` and descriptor validity before computing.
- A rebalance epoch flip is atomic with respect to graph replay ordering.

## Decode Structural Change

The current decode path has separate router and expert stages. The target is a new Release fast path stage, tentatively `MoEDecodeFusedStage`, with fixed launch count per MoE layer.

### Kernel 1: `moe_decode_route_select`

Responsibilities:

- Compute router logits for one token.
- Compute max/sum/top-k in one kernel without writing full router logits in Release.
- Emit top-k expert ids and weights into `DeviceMoELayerRuntime`.
- Apply the active placement bank to produce local compute decisions.
- Update device histogram counters.

Performance direction:

- Stop treating router weights as a generic FP32 raw pointer path.
- Export router weights through the same descriptor/prepared-weight mechanism where possible.
- Add a ROCm native-VNNI or FP16 router GEMV path tuned for `num_experts x d_model` with one-token input.
- Fuse top-k and weight renormalization so there is no separate int-to-float conversion or D2D routing-copy kernel in Release.

Debug/snapshot mode may still materialize router logits and routing tensors through an explicit slow path.

### Kernel 2: `moe_decode_expert_fused`

Responsibilities:

- Read top-k ids/weights from `DeviceMoELayerRuntime`.
- Resolve each logical expert through the active placement bank.
- Compute gate/up/down for the routed experts and accumulate in routing order into `[d_model]` output.

First production target:

- Merge the current grouped gate/up and grouped down decode into one stage with no host routing data and no pointer staging in the hot path.
- Keep two internal kernels if needed: grouped gate/up followed by fused SwiGLU+down+accumulate.
- Remove external scratch pointer arrays from captured arguments by storing scratch base pointers in `DeviceMoELayerRuntime`.

Parity target:

- Deterministic routing-order accumulation for the first implementation.
- Optional faster unordered/atomic path can be added behind a perf flag after parity is stable.

Performance target:

- Decode MoE should move from roughly 26 ms/token across router+expert to less than 10 ms/token across all MoE layers.
- This requires both router specialization and expert-kernel fusion; graph capture alone cannot deliver the remaining gap.

## Grouped Prefill Kernel Design

Prefill needs graph compatibility first, then performance. Graph compatibility means fixed launch topology for a given `seq_len`, with all per-expert variability read from device counts/masks.

### Stage Shape

Replace the current host-driven GPU prefill branch with a graph-capturable stage sequence:

```text
MoERoutePrefillStage
MoEGroupRoutesStage
MoEGroupedPrefillExpertStage
```

For the first single-device ROCm implementation these may be collapsed into `MoEGroupedPrefillExpertStage` after routing tensors exist, but the kernel boundary should remain clear.

### `MoEGroupRoutesStage`

Inputs:

- routing indices `[seq_len, top_k]`
- routing weights `[seq_len, top_k]`
- active placement bank

Outputs in `DeviceMoELayerRuntime`:

- `expert_counts[num_experts]`
- `expert_offsets[num_experts]`
- `grouped_token_ids[seq_len * top_k]`
- `grouped_route_weights[seq_len * top_k]`

Implementation:

1. Convert FP32 expert ids to int, or avoid FP32 route ids entirely in the new router.
2. Count local/resident routed entries per expert on device.
3. Prefix-scan counts to offsets on device.
4. Scatter `(token_id, route_weight)` pairs into grouped route slots.

No D2H count/offset copy is allowed in Release.

### `moe_prefill_gate_up_grouped`

Fixed grid:

```text
grid = (num_experts, ceil(seq_len / TILE_M), ceil(intermediate / TILE_N))
```

Each block:

- reads the active bank and no-ops if the expert is not local/resident;
- reads `count = expert_counts[expert]` from device;
- no-ops if the tile is outside `count`;
- maps local grouped row to original token id through `grouped_token_ids[offset + row]`;
- computes gate and up tiles into preallocated grouped scratch `[seq_len * top_k, intermediate]`.

The launch grid depends on graph shape (`seq_len`, dims), not runtime counts.

### `moe_prefill_down_accumulate_grouped`

Fixed grid:

```text
grid = (num_experts, ceil(seq_len / TILE_M), ceil(d_model / TILE_N))
```

Each block:

- reads grouped rows for one expert;
- applies SwiGLU from grouped gate/up scratch;
- runs down projection with the active bank descriptor;
- accumulates `route_weight * down_output` into output at original token ids.

Initial accumulation options:

1. Atomic add into output: fastest first ROCm path, acceptable if parity tolerances remain stable.
2. Route-slot partials plus fixed token reduce: deterministic path for parity/debug, one extra kernel.

The debug deterministic path should be available even if the default is atomic.

### Scratch And Memory

Prefill scratch is preallocated per graph shape:

- grouped gate scratch: `[seq_len * top_k, intermediate]`
- grouped up scratch: `[seq_len * top_k, intermediate]`
- optional partial output scratch: `[seq_len * top_k, d_model]` for deterministic reduce

This is large but bounded by the captured prefill graph shape. For long prompts, add a tiled prefill mode that processes route rows in fixed-size chunks; each chunk is a fixed graph segment.

## Multi-Participant Overlay Compatibility

The single-device ROCm runtime table is also the local compute table for graph-native overlay participants.

- Continuation/root participant runs router and sparse dispatch stages.
- Expert participants run `MoEGroupRoutesStage` or receive already grouped sparse payloads.
- Local expert kernels use the same active placement bank and descriptor validity rules.
- Participants with no rows still execute the same kernel launches; counts are zero and masks no-op.
- Sparse dispatch/return collectives remain manual or graph-collective segments until the backend supports captured collectives.

This keeps the V2 invariant: every compute stage is participant-local, and all cross-device/rank coordination is explicit collective staging.

## Interface Changes

Add a MoE runtime-table abstraction rather than extending every `ITensorGemm` call site.

```cpp
class IMoERuntimeTable {
public:
    virtual DeviceMoELayerRuntime* deviceLayerState(int layer_idx) = 0;
    virtual bool prepareInactiveBank(int layer_idx, const MoEPlacementUpdate&) = 0;
    virtual bool flipActiveBank(int layer_idx, uint32_t epoch, void* stream) = 0;
};

class IMoEKernel {
public:
    virtual bool decodeRouteSelect(DeviceMoELayerRuntime*, ITensor* hidden, ITensor* router_weight) = 0;
    virtual bool decodeExpertFused(DeviceMoELayerRuntime*, ITensor* hidden, ITensor* output) = 0;
    virtual bool groupPrefillRoutes(DeviceMoELayerRuntime*, ITensor* route_ids, ITensor* route_weights) = 0;
    virtual bool groupedPrefillExpert(DeviceMoELayerRuntime*, ITensor* hidden, ITensor* output) = 0;
};
```

`ITensorGemm` remains responsible for weight preparation and descriptor export. The MoE runtime table owns device-readable descriptor tables and placement state.

## Graph Capture Requirements

Stages become graph-capturable only when all of these are true:

- no host reads of routing, counts, offsets, or histograms;
- no host-side per-expert loops that change launches;
- no allocations in `execute()`;
- no stream sync in `execute()`;
- kernel launch grids are fixed by graph shape, not route contents;
- kernel arguments are stable pointers to tensors and runtime tables;
- rebalancing updates table contents outside capture/replay, then flips epoch on stream.

Expected stage changes:

- `MoERoutingStage`: split Release graph path from snapshot path; Release path becomes capturable for decode and prefill routing/grouping.
- `MoEExpertComputeStage`: keep CPU/fallback path; add ROCm graph-native grouped decode and prefill path; return `true` from `isGraphCapturable()` only when the graph-native path is active.
- Overlay sparse stages: remain non-capturable until collectives are graph-capturable, but local expert kernels inside participants use the same device runtime table.

## Implementation Phases

### Phase A: Runtime Table And Rebalance Epochs

- Add `DeviceMoELayerRuntime` and host owner classes.
- Export existing grouped gate/up/down descriptor tables into double-buffered banks.
- Wire `MoERebalanceController` updates to inactive-bank preparation and epoch flip.
- Add tests that move an expert between banks and verify captured kernels read the new descriptor without graph recapture.

### Phase B: Decode Release Graph Path

- Replace `MoERoutingStage` decode Release path with `decodeRouteSelect()`.
- Replace stage-level active expert construction with `decodeExpertFused()`.
- Remove decode host routing reads from Release entirely.
- Mark decode MoE stages capturable when runtime table path is active.

### Phase C: Router Performance

- Add prepared/quantized router weights for ROCm.
- Implement fused logits/top-k/renorm/histogram kernel that does not materialize full logits in Release.
- Compare FP32, FP16, and native-VNNI router paths with low-level tests and model parity.

### Phase D: Graph-Compatible Prefill

- Replace `prepareExpertGroups()` host count read with `groupPrefillRoutes()`.
- Implement grouped prefill gate/up and down/accumulate kernels with fixed all-expert launch grids.
- Add deterministic reduce mode for parity and atomic mode for performance.
- Mark prefill MoE expert stages capturable for fixed-shape prefill graph segments.

### Phase E: Overlay Integration

- Use the runtime table inside `MoELocalExpertStage`.
- Keep sparse dispatch/return collectives explicit.
- Ensure expert participants no-op by device counts/masks, not host branching.

### Phase F: Parity And Performance Closure

- Gate each phase with focused ROCm kernel tests, Qwen35 decode/prefill parity, graph-capture tracing, and Release benchmark.
- Target decode parity with llama.cpp first, then prefill.

## Acceptance Gates

Correctness:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen35MoE_SingleDevice" --output-on-failure --parallel
```

Graph capture:

```bash
LLAMINAR_LOG_LEVEL=DEBUG LLAMINAR_GPU_GRAPHS=1 LLAMINAR_GPU_GRAPH_MAX_STAGES=1 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Rebalance compatibility:

```bash
LLAMINAR_MOE_REBALANCE=dynamic LLAMINAR_MOE_REBALANCE_WINDOW=32 \
LLAMINAR_GPU_GRAPHS=1 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Performance:

```bash
LLAMINAR_LOG_LEVEL=WARN LLAMINAR_GPU_GRAPHS=1 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

## Risks

Risk: atomic prefill accumulation changes parity.

Mitigation: keep deterministic route-slot partial plus reduce path for parity and debugging.

Risk: captured graph sees partially updated placement.

Mitigation: double-buffer descriptor banks and flip active bank only after a stream-ordered update event.

Risk: old weights freed while a replay still references the old bank.

Mitigation: retire departed weights after replay completion event for the old epoch.

Risk: router quantization changes top-k decisions.

Mitigation: start with FP32 fused router, then add quantized router behind parity thresholds and top-k overlap gates.

Risk: prefill scratch is too large for long prompts.

Mitigation: fixed-size chunked prefill segments with graph shape determined by chunk size.

## Bottom Line

The route to parity is a runtime-table architecture, not a bigger host loop. Rebalancing and graph capture are compatible if expert placement is a device-resident epoch table with stable pointer identity. ROCm MoE kernels must become native consumers of that table for both decode and prefill. Once routing/top-k, grouping, and expert descriptors are all device-side, the remaining performance work becomes kernel tuning rather than execution-model repair.