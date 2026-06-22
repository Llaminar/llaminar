# MoE Device-Side Grouped Kernel Project Plan

Date: 2026-05-17
Scope: Qwen35 MoE single-device ROCm first, with CUDA/CPU-compatible interfaces where practical.

## Goals

Close the Qwen35 MoE single-device performance gap by replacing host-driven per-expert execution with graph-compatible device-side grouped kernels.

Primary assumptions:

1. Decode will use a single-kernel SwiGLU + down + weighted accumulate path.
2. Prefill and decode will move toward a graph-compatible grouped-kernel design that avoids host-side per-expert control.
3. `ITensorGemm` interface changes are allowed when they keep the design cleaner than pushing MoE-specific behavior through generic GEMM entry points.

## Current Bottlenecks

`MOE_EXPERT_FFN` is slow because active experts are still executed as a sequence of small operations. Decode already batches gate/up projection descriptors, but the GPU down path still loops over active experts and performs per expert:

1. SwiGLU into temporary activation storage.
2. Activation quantization and down GEMV.
3. Weighted add into the MoE output.

Prefill has device-side routing/grouping, but the host still reads expert counts/offsets and drives a per-expert C++ loop. That keeps MoE stages non-capturable and leaves CPU control flow between routing and expert execution.

The performance target requires eliminating both problems: fewer tiny expert kernels and no host-side dynamic expert dispatch in the hot path.

## Architecture Direction

Add a MoE-specific grouped expert execution API rather than stretching the generic `multiply_fused_tensor()` interface too far.

Recommended interface shape:

```cpp
struct GroupedMoEDecodeDesc {
    const float* hidden;              // [d_model]
    const float* routing_indices;     // [top_k], FP32 expert ids or int buffer variant
    const float* routing_weights;     // [top_k]
    float* output;                    // [d_model]
    int top_k;
    int d_model;
    int intermediate;
    int layer_idx;
};

struct GroupedMoEExpertWeights {
    DeviceExpertMatrixTable gate;
    DeviceExpertMatrixTable up;
    DeviceExpertMatrixTable down;
};

class IMoEGroupedExpertKernel {
public:
    virtual bool executeDecode(const GroupedMoEDecodeDesc& desc,
                               const GroupedMoEExpertWeights& weights) = 0;
    virtual bool executePrefill(const GroupedMoEPrefillDesc& desc,
                                const GroupedMoEExpertWeights& weights) = 0;
};
```

`ITensorGemm` should still own common weight preparation and native-VNNI metadata, but MoE grouped execution needs a compact device descriptor table that kernels can index by expert id. That table should be produced from prepared GEMM engines once at graph build/preload time.

## Phase 0: Alpha/Beta Correctness Gate

Status: implemented as a prerequisite.

Deliverables:

- Ensure ROCm native-VNNI decode GEMM supports `alpha != 1` and `beta != 0` safely.
- Add tests for plain decode GEMV and fused SwiGLU/down decode accumulation.

Acceptance:

- Focused ROCm native-VNNI alpha/beta tests pass.
- Qwen35 decode parity remains clean once direct accumulation is enabled in MoE.

## Phase 1: Decode Single-Kernel SwiGLU + Down + Accumulate

Objective: replace the per-active-expert down loop with one MoE decode kernel launch per MoE layer.

Current decode structure:

- Router produces `top_k` indices and weights.
- Gate/up projections are batched at stage level.
- Down path loops over active experts.

Target decode structure:

1. Keep the existing batched gate/up path initially.
2. Add a ROCm kernel that consumes:
   - `gate_scratch[top_k, intermediate]`
   - `up_scratch[top_k, intermediate]`
   - `expert_ids[top_k]`
   - `expert_weights[top_k]`
   - a device expert down descriptor table
3. In one launch, for each active expert slot:
   - compute `silu(gate) * up`
   - run the down projection for that expert
   - accumulate `routing_weight * down_output` into the final output

Implementation notes:

- Start with deterministic ordered accumulation. Since `d_model=2048` and `top_k` is small, one block/tile per output column range can loop over `top_k` in routing order and avoid atomics.
- Use existing native-VNNI decode routines as a reference, but expect a new kernel because the down projection must select expert-specific weight descriptors by device-side expert id.
- Keep a debug fallback to the current per-expert path under an env flag until parity and perf settle.

Acceptance:

- Focused unit/integration test compares grouped decode output to the current sequential expert loop for random top-k expert sets.
- Qwen35 ROCm decode parity passes.
- `MOE_EXPERT_FFN` decode time drops materially in `LLAMINAR_PROFILING=1` runs.

## Phase 2: Device Expert Descriptor Table

Objective: make expert selection device-readable instead of host `ITensorGemm*` dispatch.

Deliverables:

- Add a compact `DeviceExpertMatrixTable` for ROCm native-VNNI weights:
  - payload pointers
  - scale/min/extra-min pointers
  - codebook ids
  - `N`, `K`, format metadata
- Build tables per layer for gate/up/down during graph construction or prepared-weight preload.
- Store table handles in MoE stage params or a model-level prepared expert store.

Design choice:

- Prefer a MoE-specific table over adding expert-id indexing to every `ITensorGemm` method. `ITensorGemm` can expose descriptor export hooks, while grouped MoE kernels consume the exported table.

Acceptance:

- Table export validates all experts in a layer have compatible dimensions and supported formats.
- Existing non-MoE GEMM paths are unchanged.
- Missing or unsupported expert formats fail loud during graph build/preload.

## Phase 3: Fully Device-Side Decode MoE

Objective: remove host routing reads and host active-expert construction from decode.

Target decode flow:

1. Router writes device-resident top-k indices and weights.
2. Gate/up grouped kernel reads routing indices directly and computes all active expert gate/up outputs.
3. Single-kernel SwiGLU + down + accumulate reads routing indices/weights directly.
4. Host does not call `routing_indices->data()` in the hot path.

Graph compatibility requirements:

- Fixed launch count per MoE layer.
- Kernel bodies branch/no-op based on device routing data.
- No host-side `num_active`-dependent loop.
- No D2H routing read except debug/snapshot paths.

Acceptance:

- Decode MoE stages can be marked graph-capturable on ROCm for Release execution.
- Graph capture trace shows MoE decode in captured segments, not manual segments.
- Decode parity passes with graph replay enabled.

## Phase 4: Graph-Compatible Prefill Grouped Expert Execution

Objective: remove host-side per-expert control from prefill.

Current prefill already does device grouping, but it D2H's counts/offsets and the host loops over experts with work.

Target prefill options:

1. Fixed all-expert launch pattern:
   - launch one grouped kernel over all experts
   - each expert reads `count` from device and no-ops if zero
   - graph-capturable because launch topology is fixed
2. Bucketed grouped launch pattern:
   - fixed number of expert buckets
   - device compaction fills buckets
   - kernels process buckets with fixed upper bounds

Recommended first implementation:

- Use fixed all-expert launches for simplicity and correctness.
- Optimize later if launching all experts with zero-count checks is too expensive.

Acceptance:

- No host read of `host_expert_counts_`/`host_expert_offsets_` in Release prefill.
- Prefill MoE expert stages become graph-capturable.
- Prefill parity passes for Qwen35 ROCm.
- Prefill throughput improves against the current graph-enabled baseline.

## Phase 5: Cross-Backend and Interface Cleanup

Objective: keep the architecture maintainable after ROCm proves the design.

Deliverables:

- CUDA implementation of the same grouped MoE interfaces, or explicit unsupported-path diagnostics.
- CPU fallback remains the existing sequential/batched CPU path.
- Common tests for grouped expert output equivalence across backends where available.
- Remove temporary env flags once parity/perf are stable.

## Phase 6: Performance Gates

Use these as recurring gates after each phase:

Correctness:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen35MoE_SingleDevice" --output-on-failure --parallel
```

ROCm benchmark:

```bash
LLAMINAR_LOG_LEVEL=WARN LLAMINAR_GPU_GRAPHS=1 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Profiling:

```bash
LLAMINAR_LOG_LEVEL=WARN LLAMINAR_PROFILING=1 LLAMINAR_GPU_GRAPHS=1 \
LLAMINAR_VALIDATE_BUFFERS=0 LLAMINAR_VALIDATE_INPUTS=0 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Targets:

- Decode: reach or exceed llama.cpp ROCm baseline after device-side decode MoE.
- Prefill: reach or exceed llama.cpp ROCm baseline after graph-compatible prefill grouped kernels.

## Risks and Mitigations

Risk: accumulation order changes affect decode parity.

Mitigation: keep routing-order accumulation in the first grouped kernel and compare directly against the current sequential path.

Risk: descriptor table lifetime bugs.

Mitigation: own tables in the prepared-weight store or stage params with explicit graph lifetime; validate pointers/devices before execution in Debug/Integration.

Risk: graph capture fails due hidden stream or host sync.

Mitigation: keep `LLAMINAR_GPU_GRAPH_MAX_STAGES=1` capture tracing as a gate for each newly capturable stage type.

Risk: single-kernel implementation gets too complex.

Mitigation: land a two-kernel grouped version first if needed: grouped SwiGLU+down outputs, then ordered weighted reduce. Upgrade to one kernel after parity is locked.