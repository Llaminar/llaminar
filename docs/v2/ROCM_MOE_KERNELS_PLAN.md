# ROCm MoE Kernels: GPU-Native MoE Ops + Dynamic Rebalancing

**Status:** Project plan — phased implementation  
**Branch:** `feat/qwen35-moe`  
**Prerequisite:** MoE stage decomposition (Phases 1-3 of `MOE_STAGE_DECOMPOSITION_PLAN.md`) complete  
**Related:**
- `docs/v2/MOE_STAGE_DECOMPOSITION_PLAN.md` — Stage decomposition (complete)
- `docs/v2/QWEN35_MOE_EXPERT_REBALANCING_PLAN.md` — Rebalancing architecture (Milestones 1-3 complete)
- `src/v2/kernels/rocm/` — Existing ROCm kernel infrastructure

---

## Problem

MoE stages currently run **all non-GEMM operations on CPU** even when the model executes on ROCm GPUs:

| Operation | Current | Bottleneck |
|-----------|---------|------------|
| Routing (gate logits + softmax + top-k) | CPU `CPUMoEKernel::route()` | 256 dot products → 256×2048 gate matrix D2H, then softmax+top-k on CPU |
| Token gather | CPU `memcpy`-based row collection | D2H hidden states, gather, H2D batch buffer |
| Scatter-add weighted | CPU `vec_axpy` | D2H expert output, accumulate, H2D combined output |
| SwiGLU fallback | CPU ISA-dispatched | Only needed when GEMM doesn't fuse SwiGLU |
| Shared expert gate | CPU `vec_dot` + sigmoid + `vec_scale` | D2H for dot product, H2D gated output |
| Histogram recording | CPU atomic increments | Requires routing indices on host (D2H) |
| Expert mask filtering | CPU loop | Requires routing results on host |
| Replica dispatch | CPU `assignForToken()` | Requires routing results on host |

For **prefill** (seq_len > 1), this creates O(seq_len) data movement between D2H and H2D around every non-GEMM operation. For **decode** (seq_len = 1), the data movement is small but latency dominates — each D2H sync is ~5-15μs on PCIe, and we do ~6 per MoE layer × 40 layers = 240 round-trips.

### Measured overhead (Qwen3.5-35B-A3B MoE, ROCm MI50)

| Phase | Current (CPU MoE ops) | Projected (GPU MoE ops) | Savings |
|-------|----------------------|------------------------|---------|
| Prefill (596 tok) | ~512ms | ~490ms | ~4% (GEMM-dominated) |
| Decode per-token | ~38ms | ~30ms | ~21% (latency-dominated) |

Decode savings come from eliminating 6 D2H/H2D syncs per MoE layer:
- `route()` requires hidden state on host → gate matmul on host → routing result on host
- `gatherTokenBatch()` requires hidden state on host
- `scatterAddWeighted()` requires expert output on host → combined output H2D

With GPU-native MoE ops, the entire MoE layer stays device-resident: hidden state → routing → gather → expert GEMM → scatter → combined output, all on-device. Only the histogram recording (decode counter, ~16 bytes per layer) needs D2H.

---

## Goals

1. **`ROCmMoEKernel`** — GPU-native `IMoEKernel` implementation for routing, gather/scatter, shared expert gate, SwiGLU fallback
2. **GPU-side histogram** — Device-resident expert activation counters with periodic host sync
3. **GPU-side expert mask** — Device-resident mask buffer updated on rebalance events (not per-token)
4. **Async weight repack for rebalancing** — Stream-ordered expert weight migration on ROCm
5. **Zero behavioral change** — Parity tests pass; routing produces identical results to CPU reference

## Non-Goals

- CUDA MoE kernel (parallel effort; same design, different backend)
- Changing the LPT rebalancing algorithm (stays CPU-side)
- GPU-side expert replica dispatch (requires cross-socket GPU communication — future work)
- Fused routing+gather kernel (premature optimization; measure first)

---

## Architecture Overview

### Current Data Flow (CPU MoE ops on ROCm device)

```
Device (ROCm)                          Host (CPU)
─────────────                          ──────────
hidden_state [device]
        │
        ├──── D2H sync ──────────→  hidden_state [host]
        │                                  │
        │                           CPUMoEKernel::route()
        │                                  │
        │                           routing_indices [host]
        │                           routing_weights [host]
        │                                  │
        │                           CPUMoEKernel::gatherTokenBatch()
        │                                  │
        │  ←──── H2D ────────────  batch_buffer [host → device]
        │
  expert GEMM (ROCm)
        │
        ├──── D2H sync ──────────→  expert_output [host]
        │                                  │
        │                           CPUMoEKernel::scatterAddWeighted()
        │                                  │
        │  ←──── H2D ────────────  combined_output [host → device]
        ▼
```

### Target Data Flow (GPU MoE ops)

```
Device (ROCm)                          Host (CPU)
─────────────                          ──────────
hidden_state [device]
        │
  ROCmMoEKernel::route()               (nothing)
        │
  routing_indices [device]
  routing_weights [device]
        │
  ROCmMoEKernel::gatherTokenBatch()    (nothing)
        │
  batch_buffer [device]
        │
  expert GEMM (ROCm)
        │
  ROCmMoEKernel::scatterAddWeighted()  (nothing)
        │
  combined_output [device]              histogram_sync (periodic, async)
        ▼
```

---

## Kernel Design

### 1. `ROCmMoEKernel` — HIP Implementation of `IMoEKernel`

**File:** `src/v2/kernels/rocm/moe/ROCmMoEKernel.h`  
**HIP kernels:** `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip`

```cpp
class ROCmMoEKernel : public IMoEKernel, public ROCmKernelBase
{
public:
    explicit ROCmMoEKernel(int device_ordinal);
    ROCmMoEKernel(IWorkerGPUContext* ctx);  // Phase 4 device context
    ~ROCmMoEKernel() override;

    // --- IMoEKernel interface (device-resident) ---
    bool route(const float* hidden, const float* gate_weights,
               int seq_len, int d_model, int num_experts, int top_k,
               bool normalize_weights, MoERoutingResult& result) override;

    void gatherTokenBatch(const float* hidden, float* batch_buffer,
                         const int* token_indices,
                         int num_tokens, int d_model) override;

    void scatterAddWeighted(float* output, const float* expert_output,
                           const int* token_indices, const float* weights,
                           int num_tokens, int d_model) override;

    void sharedExpertGate(const float* input, const float* gate_inp,
                         float* shared_output, int seq_len, int d_model) override;

    void swiGLU(float* gate, const float* up, int count) override;

    // --- Device-resident extensions ---

    /// Route on device, return device pointers (avoids D2H for routing results)
    /// Called by MoERoutingStage when device_id is ROCm
    bool routeDevice(const float* d_hidden, const float* d_gate_weights,
                     int seq_len, int d_model, int num_experts, int top_k,
                     bool normalize_weights,
                     int* d_routing_indices,    // [seq_len * top_k] device output
                     float* d_routing_weights,  // [seq_len * top_k] device output
                     float* d_router_logits);   // [seq_len * num_experts] device output

    /// Gather with device-resident index buffer
    void gatherTokenBatchDevice(const float* d_hidden, float* d_batch_buffer,
                                const int* d_token_indices,
                                int num_tokens, int d_model);

    /// Scatter-add with device-resident routing
    void scatterAddWeightedDevice(float* d_output, const float* d_expert_output,
                                  const int* d_token_indices,
                                  const float* d_weights,
                                  int num_tokens, int d_model);

    /// Shared expert gate on device (sigmoid dot + scale)
    void sharedExpertGateDevice(const float* d_input, const float* d_gate_inp,
                                float* d_shared_output,
                                int seq_len, int d_model);

    // --- Histogram support ---

    /// Record routing to device-side histogram (lock-free atomicAdd)
    void recordHistogramDevice(const int* d_routing_indices, int seq_len,
                               int top_k, int layer_idx);

    /// Sync device histogram → host DecodeExpertHistogram (async D2H)
    void syncHistogramToHost(DecodeExpertHistogram* host_histogram,
                             int layer_idx, int num_experts);

    /// Reset device histogram counters (called at window boundaries)
    void resetHistogramDevice(int num_experts);

    // --- Expert mask support ---

    /// Upload expert mask to device (called on rebalance, not per-token)
    void updateExpertMaskDevice(const std::vector<bool>& mask);

    /// Apply device-resident expert mask to routing weights (zero non-local)
    void applyExpertMaskDevice(float* d_routing_weights,
                               const int* d_routing_indices,
                               int seq_len, int top_k);

    // --- ITensorKernel ---
    bool supports_device(int device_idx) const override;
    KernelSnapshotInfo getKernelSnapshotInfo() const override;

private:
    int device_ordinal_;

    // Device-resident buffers (allocated once, reused)
    float* d_logits_scratch_ = nullptr;     // [max_seq_len * num_experts]
    uint64_t* d_histogram_ = nullptr;       // [num_layers * num_experts]
    bool* d_expert_mask_ = nullptr;         // [num_experts]

    int max_seq_len_ = 0;                   // For scratch sizing
    int num_experts_ = 0;
    int num_layers_ = 0;

    void ensureScratch(int seq_len, int num_experts, int num_layers);
};
```

### 2. HIP Kernel Implementations

**File:** `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip`

#### 2a. Routing: Gate Logits + Softmax + Top-K

```
Kernel: rocm_moe_gate_logits_fp32
  Grid:  [num_experts, (d_model+255)/256]
  Block: [256]
  
  Per expert, per thread-block chunk:
    partial_dot = Σ hidden[t][d] * gate[e][d]   (float4 vectorized)
    Block reduce → logits[t][e]
  
  Uses warp shuffle + shared mem for reduction.
  For decode (seq_len=1): single token → 256 experts fits in 1 kernel launch.
  For prefill: tile over tokens.

Kernel: rocm_moe_softmax_topk_fp32
  Grid:  [seq_len]
  Block: [32]  (single warp per token, matches sampling kernel pattern)
  
  Per token:
    Phase 1: Thread-parallel scan of num_experts logits → local max
    Phase 2: Warp reduce max → subtract → exp → warp reduce sum → normalize
    Phase 3: Thread-parallel insertion sort top-k (k ≤ 16 → registers)
    Phase 4: Write top-k indices + weights to output
    
  If normalize_weights: renormalize top-k weights to sum to 1.
  
  Reuses existing rocmOps_topk_f32 pattern from ROCmSamplingKernels.hip
  but operates on per-token softmax probabilities rather than global logits.
```

**Design note:** For Qwen3.5-35B (256 experts, top_k=8), the gate logits kernel does 256 dot products of dimension 2048. This is a (256, 2048) × (2048, 1) GEMV — we could alternatively delegate to `ROCmQuantisedGemmKernel` for the gate GEMV and only write a custom softmax+top-k kernel. Decision: **custom kernel for the full route pipeline** because:
1. Gate weights are FP32 (not quantized) — GEMM engine overhead for FP32 GEMV is wasteful
2. Fusing logits+softmax+top-k in 2 kernels saves 2 kernel launches vs 3 separate ops
3. For decode (seq_len=1), the entire route is ~10μs on GPU vs ~50μs on CPU

#### 2b. Token Gather

```
Kernel: rocm_moe_gather_tokens_fp32
  Grid:  [num_tokens, (d_model+255)/256]
  Block: [256]
  
  Per token-in-batch:
    batch_buffer[i * d_model + d] = hidden[token_indices[i] * d_model + d]
  
  float4 vectorized for coalesced reads (d_model is always multiple of 4).
  For decode (seq_len=1): gather is a no-op (single token IS the batch).
```

#### 2c. Scatter-Add Weighted

```
Kernel: rocm_moe_scatter_add_fp32
  Grid:  [num_tokens, (d_model+255)/256]  
  Block: [256]
  
  Per token-in-batch:
    output[token_indices[i] * d_model + d] += weights[i] * expert_output[i * d_model + d]
  
  atomicAdd for conflict-free accumulation (tokens may map to same output row
  when multiple experts serve the same token via top-k).
  
  float4 vectorized where possible; scalar atomicAdd fallback for conflicts.
  
  For decode (seq_len=1, top_k=8): 8 weighted accumulations into 1 output row.
  No atomics needed (sequential expert execution within single token).
```

**Optimization for decode:** When `seq_len == 1`, the scatter-add degenerates to `output[d] += w * expert_out[d]` with no index indirection. A specialized single-token kernel avoids index loads and atomic overhead.

#### 2d. Shared Expert Gate

```
Kernel: rocm_moe_shared_expert_gate_fp32
  Grid:  [(seq_len * d_model + 255) / 256]
  Block: [256]
  
  Two-phase:
    Phase 1: dot = Σ gate_inp[d] * input[t * d_model + d]  (parallel reduce)
    Phase 2: gate = 1 / (1 + exp(-dot))                     (fast sigmoid)
             shared_output[t * d_model + d] *= gate          (elementwise scale)
  
  For decode: single-token dot product + broadcast scale → 1 block.
  For prefill: tile over tokens, 1 block per token.
```

#### 2e. SwiGLU Fallback

Delegates to existing `hipOps_swiglu_fp32()` in `ROCmSwiGLUKernels.hip`. No new kernel needed — the existing implementation already handles arbitrary-length float4-vectorized SwiGLU.

#### 2f. Histogram Recording

```
Kernel: rocm_moe_histogram_record
  Grid:  [seq_len]
  Block: [top_k]  (or 32, whichever is larger for occupancy)
  
  Per token:
    for s in 0..top_k:
      expert_id = routing_indices[t * top_k + s]
      atomicAdd(&d_histogram[layer_idx * num_experts + expert_id], 1)
  
  Lock-free uint64 atomicAdd. No ordering requirements.
  Only writes to device-resident histogram buffer.
```

#### 2g. Expert Mask Application

```
Kernel: rocm_moe_apply_expert_mask
  Grid:  [seq_len, (top_k + 31) / 32]
  Block: [32]
  
  Per token, per top-k slot:
    expert_id = routing_indices[t * top_k + k]
    if (!d_expert_mask[expert_id]):
      routing_weights[t * top_k + k] = 0.0f
  
  Simple conditional zero — no atomics, no reductions.
  Called after softmax+top-k, before gather.
```

---

## Dynamic Rebalancing: GPU-Side Operations

### Current Rebalancing Lifecycle (CPU-only, Milestones 1-3)

```
DecodeExpertHistogram::record()                  ← per-token, per-layer
    ↓
MoERebalanceController::shouldRebalance()        ← window boundary check
    ↓
MoERebalanceController::rebalanceLPT()           ← CPU-side LPT algorithm
    ↓
MoEExpertWeightService::releaseDepartedExperts() ← free outgoing expert GEMM engines
    ↓
MPI weight transfer (inter-socket)               ← serialized expert blobs
    ↓
MoEExpertWeightService::registerAndPrepareNewExperts() ← pack + prep incoming experts
    ↓
MoEExpertComputeStage::updateExpertMask()        ← new mask applied
```

### Extended Lifecycle with ROCm MoE Ops

The key insight: with GPU-native routing, the **histogram** and **expert mask** live on-device. The rebalance controller still runs on CPU, but mask/histogram synchronization becomes explicit:

```
[Device-side, per-token]
ROCmMoEKernel::routeDevice()                    ← gate logits + softmax + top-k
    ↓
ROCmMoEKernel::recordHistogramDevice()           ← atomicAdd to device histogram
    ↓
ROCmMoEKernel::applyExpertMaskDevice()           ← zero non-local expert weights

[Host-side, at window boundary]
ROCmMoEKernel::syncHistogramToHost()             ← async D2H of histogram counters
    ↓
MoERebalanceController::shouldRebalance()        ← same CPU logic
    ↓
MoERebalanceController::rebalanceLPT()           ← same CPU LPT
    ↓
MoEExpertWeightService::releaseDepartedExperts() ← free device GEMM engines
    ↓
MPI weight transfer (inter-socket)               ← serialized expert blobs
    ↓
asyncPrepareIncomingExperts()                     ← stream-ordered H2D + repack (NEW)
    ↓
ROCmMoEKernel::updateExpertMaskDevice()          ← H2D new mask (1 small copy)
    ↓
MoEExpertComputeStage::updateExpertMask()        ← host mask for dump/diagnostics
```

### Async Expert Weight Preparation (New)

**Problem:** `registerAndPrepareNewExperts()` currently blocks inference while repacking incoming expert weights (VNNI packing + device upload). For 8 experts × 2048×14336 weights = ~940MB, this takes ~200ms on PCIe Gen3.

**Solution:** Stream-ordered preparation on a secondary HIP stream:

```cpp
// In MoEExpertWeightService (extended for ROCm async prep)
struct AsyncPrepContext {
    hipStream_t prep_stream;          // Secondary stream for async H2D
    hipEvent_t prep_done_event;       // Signaled when all experts ready
    std::vector<int> pending_experts; // Expert IDs being prepared
    bool in_flight = false;
};

bool MoEExpertWeightService::asyncPrepareIncomingExpertsROCm(
    MoEWeightContext& ctx,
    const std::vector<bool>& new_mask,
    const std::unordered_map<int, ExpertWeightBlobs>* received_weights,
    AsyncPrepContext& async_ctx)
{
    // 1. VNNI repack on host (CPU-parallel, same as before)
    packMoEExpertsROCm(new_expert_views, ...);
    
    // 2. Upload to device on secondary stream (non-blocking)
    hipMemcpyAsync(d_vnni, h_vnni, bytes, hipMemcpyHostToDevice, async_ctx.prep_stream);
    hipMemcpyAsync(d_scales, h_scales, bytes, hipMemcpyHostToDevice, async_ctx.prep_stream);
    
    // 3. Record event on prep stream
    hipEventRecord(async_ctx.prep_done_event, async_ctx.prep_stream);
    
    // 4. On compute stream: wait for prep to finish before using new engines
    hipStreamWaitEvent(compute_stream, async_ctx.prep_done_event, 0);
    
    return true;
}
```

**Key design:** The compute stream continues processing with the old expert mask while the prep stream uploads new weights. Once `prep_done_event` fires, the compute stream atomically switches to the new expert mask + new GEMM engines. No inference stall.

**Rollback:** If prep fails (OOM, hipError), the old expert mask remains active. The controller retries at the next window boundary. No partial state.

---

## Integration with MoE Stages

### MoERoutingStage Changes

The routing stage already delegates to `IMoEKernel`. For ROCm devices, `KernelFactory::getOrCreateMoEKernel()` returns `ROCmMoEKernel` instead of `CPUMoEKernel`. The stage must be extended to support device-resident routing outputs:

```cpp
// MoERoutingStage::execute() — extended for GPU kernels
bool MoERoutingStage::execute(IDeviceContext* ctx) {
    const auto& env = debugEnv();
    
    if (auto* rocm_kernel = dynamic_cast<ROCmMoEKernel*>(moe_kernel_)) {
        // GPU path: route directly on device, outputs stay on device
        rocm_kernel->routeDevice(
            params_.input->device_data(),       // device pointer
            params_.gate_weights->device_data(), // device pointer
            seq_len, d_model, num_experts, top_k,
            params_.norm_topk_prob,
            params_.routing_indices->mutable_device_data(),  // device output
            params_.routing_weights->mutable_device_data(),  // device output
            d_router_logits_scratch_);
        
        // Expert mask application (device-resident)
        if (hasExpertParallelism()) {
            rocm_kernel->applyExpertMaskDevice(
                params_.routing_weights->mutable_device_data(),
                params_.routing_indices->device_data(),
                seq_len, top_k);
        }
        
        // Histogram recording (device-resident, periodic host sync)
        if (params_.decode_histogram) {
            rocm_kernel->recordHistogramDevice(
                params_.routing_indices->device_data(),
                seq_len, top_k, params_.layer_idx);
            
            // Periodic sync (every N tokens, configured by rebalance controller)
            if (shouldSyncHistogram()) {
                rocm_kernel->syncHistogramToHost(
                    params_.decode_histogram, params_.layer_idx, num_experts);
            }
        }
        
        // Mark outputs as device-authoritative
        params_.routing_indices->mark_device_dirty();
        params_.routing_weights->mark_device_dirty();
    } else {
        // CPU path: existing logic (unchanged)
        // ...
    }
    return true;
}
```

### MoEExpertComputeStage Changes

The expert compute stage already reads routing indices/weights from tensors. If those tensors are device-authoritative (GPU routing), the stage's gather/scatter calls must also use device pointers:

```cpp
// MoEExpertComputeStage::execute() — GPU gather/scatter path
if (auto* rocm_kernel = dynamic_cast<ROCmMoEKernel*>(moe_kernel_)) {
    // Prefill: group tokens by expert on device
    // (routing indices are device-resident, need D2H for grouping logic)
    // Option A: D2H routing indices for CPU grouping, then device gather/scatter
    // Option B: Device-side grouping kernel (future optimization)
    
    // Phase 1 (this plan): Option A — D2H routing for grouping only
    const int* h_indices = params_.routing_indices->data();  // triggers D2H
    const float* h_weights = params_.routing_weights->data();
    
    // Build expert token lists (CPU, same as before)
    buildExpertTokenLists(h_indices, h_weights, seq_len, top_k);
    
    // For each expert: device-resident gather → GEMM → scatter
    for (int e = local_start; e < local_end; ++e) {
        if (token_lists[e].empty()) continue;
        
        // Upload token indices for this expert (small: top_k ints)
        uploadTokenIndices(token_lists[e], d_token_indices_scratch_);
        
        rocm_kernel->gatherTokenBatchDevice(
            d_hidden, d_batch_buffer, d_token_indices_scratch_,
            num_tokens, d_model);
        
        // Expert GEMM (already device-resident)
        gate_gemm->multiply_fused_tensor(...);
        
        rocm_kernel->scatterAddWeightedDevice(
            d_output, d_expert_out, d_token_indices_scratch_,
            d_token_weights_scratch_, num_tokens, d_model);
    }
} else {
    // CPU path (unchanged)
}
```

**Phase 1 compromise:** Token grouping (building expert→token lists) still happens on CPU because it's a complex irregular operation. The routing indices D2H is small (~seq_len × top_k × 4 bytes = ~19KB for 596 tokens × 8 top-k). The net savings come from eliminating the **large** D2H/H2D transfers (hidden states, expert outputs). A fully device-side grouping kernel is deferred to Phase 3.

### Decode Fast-Path (seq_len = 1)

For decode, **no grouping is needed** — there's exactly 1 token with top-k experts. The entire flow stays on-device:

```cpp
// MoEExpertComputeStage::executeSingleToken() — GPU decode path
if (auto* rocm_kernel = dynamic_cast<ROCmMoEKernel*>(moe_kernel_)) {
    // Routing results already on device from MoERoutingStage
    // No gather needed — single token IS the input
    
    // D2H only the top-k indices (8 ints = 32 bytes) for expert dispatch
    int h_experts[MAX_TOP_K];
    float h_weights[MAX_TOP_K];
    hipMemcpy(h_experts, d_routing_indices, top_k * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_weights, d_routing_weights, top_k * sizeof(float), hipMemcpyDeviceToHost);
    
    // Zero output on device
    hipMemsetAsync(d_output, 0, d_model * sizeof(float), stream);
    
    for (int k = 0; k < top_k; ++k) {
        int eid = h_experts[k];
        if (!isLocalExpert(eid)) continue;
        
        // Gate+Up GEMM (device-resident, single token)
        gate_gemm[eid]->multiply_fused_tensor(d_hidden, {d_gate_scratch, d_up_scratch},
                                               1, d_model, ...);
        
        // SwiGLU (device-resident, reuse existing ROCm SwiGLU kernel)
        rocm_kernel->swiGLU(d_gate_scratch, d_up_scratch, intermediate);
        
        // Down GEMM (device-resident)
        down_gemm[eid]->multiply_tensor(d_gate_scratch, d_down_scratch,
                                         1, d_model, intermediate, ...);
        
        // Weighted accumulate (device-resident, no scatter needed for 1 token)
        // kernel: output[d] += weight * down_out[d]
        rocm_kernel->weightedAccumulateDevice(d_output, d_down_scratch,
                                               h_weights[k], d_model);
    }
}
```

**Decode overhead reduction:** The only D2H is 32 bytes (8 expert IDs) + 32 bytes (8 weights) = 64 bytes total. Everything else stays on device.

---

## KernelFactory Integration

### `getOrCreateMoEKernel()` Extension

```cpp
// KernelFactory.cpp — extended dispatch
llaminar2::IMoEKernel* KernelFactory::getOrCreateMoEKernel(DeviceId device_id) {
    // ... existing cache lookup ...
    
    std::unique_ptr<llaminar2::IMoEKernel> kernel;
    
    if (device_id.is_rocm()) {
#ifdef HAVE_ROCM
        // Check for Phase 4 device context
        auto* ctx = getWorkerGPUContext(device_id);
        if (ctx) {
            kernel = std::make_unique<llaminar2::rocm::ROCmMoEKernel>(ctx);
        } else {
            kernel = std::make_unique<llaminar2::rocm::ROCmMoEKernel>(
                device_id.ordinal);
        }
#else
        kernel = std::make_unique<llaminar2::CPUMoEKernel>();
#endif
    } else if (device_id.is_cuda()) {
        // CUDA MoE kernel (future — parallel effort)
        kernel = std::make_unique<llaminar2::CPUMoEKernel>();
    } else {
        kernel = std::make_unique<llaminar2::CPUMoEKernel>();
    }
    
    // ... cache and return ...
}
```

### Workspace Requirements

`ROCmMoEKernel` needs scratch buffers for routing logits and intermediate results. These are declared via `IWorkspaceConsumer` (inherited from `ROCmKernelBase`):

```cpp
WorkspaceDescriptor ROCmMoEKernel::describeWorkspace() const {
    WorkspaceDescriptor desc;
    
    // Routing logits scratch: [max_seq_len * num_experts] floats
    desc.add("MOE_ROUTE_LOGITS",
             max_seq_len_ * num_experts_ * sizeof(float),
             64);  // 64-byte alignment for float4
    
    // Device histogram: [num_layers * num_experts] uint64
    desc.add("MOE_HISTOGRAM",
             num_layers_ * num_experts_ * sizeof(uint64_t),
             64);
    
    // Expert mask: [num_experts] bool
    desc.add("MOE_EXPERT_MASK",
             num_experts_ * sizeof(bool),
             64);
    
    // Token index scratch (for gather/scatter): [max_seq_len * top_k] int32
    desc.add("MOE_TOKEN_INDICES",
             max_seq_len_ * 16 * sizeof(int32_t),  // top_k ≤ 16
             64);
    
    // Token weight scratch (for scatter): [max_seq_len * top_k] float
    desc.add("MOE_TOKEN_WEIGHTS",
             max_seq_len_ * 16 * sizeof(float),
             64);
    
    return desc;
}
```

**Memory budget** (Qwen3.5-35B, max_seq_len=2048, 256 experts, 40 layers):
- Logits scratch: 2048 × 256 × 4 = 2MB
- Histogram: 40 × 256 × 8 = 80KB
- Expert mask: 256 × 1 = 256B
- Token indices: 2048 × 16 × 4 = 128KB
- Token weights: 2048 × 16 × 4 = 128KB
- **Total: ~2.3MB** (negligible vs model weights)

---

## Phased Implementation Plan

### Phase 1: Core ROCm MoE Kernel — Route + Gather/Scatter

**Goal:** GPU-native routing, gather, scatter. Decode fast-path stays fully on-device. Histogram/mask remain on CPU (D2H routing indices for histogram + mask check).

**Files to create:**
| File | Purpose |
|------|---------|
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.h` | Kernel header |
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp` | Host-side implementation |
| `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip` | HIP device kernels |
| `tests/v2/unit/kernels/rocm/moe/Test__ROCmMoEKernel.cpp` | Unit tests |

**Files to modify:**
| File | Change |
|------|--------|
| `src/v2/kernels/KernelFactory.cpp` | Return `ROCmMoEKernel` for ROCm devices |
| `src/v2/CMakeLists.txt` | Add new source files |
| `tests/v2/CMakeLists.txt` | Add new test files |

**HIP kernels to implement:**
1. `rocm_moe_gate_logits_fp32` — gate weight × hidden state dot products
2. `rocm_moe_softmax_topk_fp32` — per-token softmax + top-k selection
3. `rocm_moe_gather_tokens_fp32` — batched row collection
4. `rocm_moe_scatter_add_fp32` — weighted scatter-add accumulation
5. `rocm_moe_weighted_accumulate_fp32` — single-token weighted add (decode fast-path)
6. `rocm_moe_shared_expert_gate_fp32` — sigmoid dot + elementwise scale

**IMoEKernel contract compliance:** The host-side `route()` etc. methods perform D2H of results to populate `MoERoutingResult` (host struct). The new `routeDevice()` etc. methods keep results on-device. Stage code uses the device variants when the kernel is `ROCmMoEKernel`.

**Tests:**
- `Test__ROCmMoEKernel_Route` — Compare GPU routing output with CPU reference (identical top-k selections)
- `Test__ROCmMoEKernel_GatherScatter` — Round-trip: gather tokens → identity transform → scatter = weighted sum of input
- `Test__ROCmMoEKernel_SharedExpertGate` — Compare GPU sigmoid-gate with CPU reference
- `Test__ROCmMoEKernel_DecodePathFullyOnDevice` — Single-token route+gather+scatter with no D2H except final output

**Exit criteria:**
- All parity tests pass (GPU routing produces identical expert selections)
- Decode path eliminates large D2H/H2D transfers
- `KernelFactory::getOrCreateMoEKernel()` returns `ROCmMoEKernel` for ROCm devices

---

### Phase 2: Device-Resident Histogram + Expert Mask

**Goal:** Histogram recording and expert mask filtering run entirely on-device. Only periodic histogram sync to host (at window boundaries).

**Files to modify:**
| File | Change |
|------|--------|
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.h` | Add histogram/mask device buffers |
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp` | Implement histogram sync + mask upload |
| `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip` | Add histogram + mask kernels |
| `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp` | Use device histogram/mask APIs |
| `tests/v2/unit/kernels/rocm/moe/Test__ROCmMoEKernel.cpp` | Add histogram/mask tests |

**HIP kernels to implement:**
1. `rocm_moe_histogram_record` — atomicAdd to device histogram buffer
2. `rocm_moe_apply_expert_mask` — conditional zero of routing weights
3. `rocm_moe_histogram_reset` — zero histogram counters (at window boundary)

**Histogram sync protocol:**
```
[Every token — device-side]
  recordHistogramDevice(d_routing_indices, seq_len, top_k, layer_idx)

[Every window_size tokens — host-side trigger]
  syncHistogramToHost(host_histogram, layer_idx, num_experts)
    → hipMemcpyAsync(h_counts, d_counts, ..., hipMemcpyDeviceToHost, stream)
    → host_histogram->mergeDeviceCounts(h_counts, layer_idx)
  resetHistogramDevice(num_experts)

[At rebalance boundary — host-side]
  MoERebalanceController::shouldRebalance()  // uses merged host histogram
  MoERebalanceController::rebalanceLPT()
  updateExpertMaskDevice(new_mask)  // H2D: 256 bytes
```

**Tests:**
- `Test__ROCmMoEKernel_Histogram` — Record N tokens, sync, verify counts match CPU reference
- `Test__ROCmMoEKernel_ExpertMask` — Upload mask, apply to routing, verify zeroed weights
- `Test__ROCmMoEKernel_HistogramReset` — Reset, verify all-zero after reset
- `Test__ROCmMoEKernel_MaskUpdateAfterRebalance` — Simulate rebalance cycle with mask update

**Exit criteria:**
- Histogram sync produces identical results to CPU-side recording
- Expert mask application matches CPU mask logic
- Decode path: zero D2H for histogram (only periodic sync)

---

### Phase 3: Device-Side Token Grouping (Prefill Optimization)

**Goal:** Eliminate the remaining D2H of routing indices during prefill by grouping tokens by expert on-device.

**Problem:** Building `expert_token_lists[expert_id] = [(token_idx, weight), ...]` is an irregular scatter operation. On CPU, it's a simple loop. On GPU, it requires either:
- **Option A:** Radix sort by expert ID → segmented scan for offsets
- **Option B:** Atomic-based histogram + scatter (simpler, sufficient for 256 experts × 596 tokens)

**Design decision:** Option B — atomic histogram + exclusive scan + scatter:

```
Kernel 1: rocm_moe_count_per_expert
  Grid: [seq_len]  Block: [top_k or 32]
  atomicAdd(&expert_counts[routing_indices[t*k + s]], 1)

Kernel 2: rocm_moe_exclusive_scan_expert_offsets
  Grid: [1]  Block: [num_experts]  (single block, 256 threads)
  Blelloch exclusive scan on expert_counts → expert_offsets

Kernel 3: rocm_moe_scatter_token_indices
  Grid: [seq_len]  Block: [top_k or 32]
  offset = atomicAdd(&expert_write_heads[expert_id], 1)
  d_grouped_indices[expert_offsets[expert_id] + offset] = t
  d_grouped_weights[expert_offsets[expert_id] + offset] = weight
```

**Result:** Three small kernels produce device-resident grouped token lists. Expert GEMM can then iterate by expert using `expert_offsets[e]` and `expert_counts[e]` — no D2H.

**Trade-off:** For decode (seq_len=1), this is slower than the 32-byte D2H. Only use device-side grouping when `seq_len > grouping_threshold` (default: 16).

**Files to modify:**
| File | Change |
|------|--------|
| `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip` | Add 3 grouping kernels |
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.h` | Add `groupTokensByExpertDevice()` |
| `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp` | Use device grouping for prefill |

**Tests:**
- `Test__ROCmMoEKernel_GroupTokens` — Verify grouped indices match CPU reference
- `Test__ROCmMoEKernel_PrefillFullyOnDevice` — Full prefill with no D2H except final output

**Exit criteria:**
- Prefill path eliminates routing D2H for token grouping
- Decode path still uses minimal D2H (32 bytes)
- All parity tests pass

---

### Phase 4: Cross-Device Expert Weight Transfer + Async Migration

**Goal:** Efficient expert weight transfer across device boundaries during dynamic rebalancing, with three transfer modes: ROCm↔ROCm (direct memcpy), CPU→ROCm (format conversion + upload), and ROCm→CPU (download + format conversion). Async preparation overlaps with inference.

#### Background: Packing Format Landscape

Expert weights are pre-packed into device-specific VNNI layouts at graph-build time. The formats share the same underlying `packVnniBlock()` call from `IINT8Unpackable`, but store the results differently:

| Format | Owner | Layout | Key Arrays |
|--------|-------|--------|------------|
| **CPU Native VNNI** (`CPUNativeVNNIPackedWeights`) | `CPUNativeVNNIGemmKernel` | Interleaved: `[block][1024B data + 256B inline metadata]` | `native_interleaved`, `payload`, `int8_flat` |
| **ROCm Batch VNNI** (`MoEBatchPackedWeightsROCm`) | `ROCmQuantisedGemmKernel` | Separated: `payload[]`, `scales[]`, `mins[]`, `emins[]` per expert slab | `all_native_vnni`, `all_native_scales`, `all_native_mins`, `all_native_emins` |
| **CUDA Batch VNNI** (`MoEBatchPackedWeightsCUDA`) | `CUDAQuantisedGemmKernel` | Separated (identical structure to ROCm) | `all_vnni`, `all_scales`, `all_mins`, `all_emins` |

**Key insight:** CUDA and ROCm batch-packed formats are **layout-identical** — both call `packVnniBlock()` into separated `{payload, scales, mins, emins}` arrays with the same offsets. Only the naming convention differs (`all_vnni` vs `all_native_vnni`). This means **any GPU can directly consume any other GPU's packed weights with zero conversion** — CUDA↔CUDA, ROCm↔ROCm, and CUDA↔ROCm are all plain memcpy. CPU packed format uses a fundamentally different interleaved layout with inline metadata, requiring a format translation step — **performed on the GPU side** (faster memory bandwidth, massively parallel scatter/gather).

**Design convention:** All format conversion between CPU-interleaved and GPU-separated layouts is performed by GPU kernels (HIP or CUDA). The CPU never runs conversion code — it only sends/receives raw bytes. This keeps the CPU free for other work during rebalancing and exploits GPU memory bandwidth (~900 GB/s MI50, ~2 TB/s A100) for the scatter/gather operations.

#### Transfer Mode 1: GPU ↔ GPU (Same-Format Direct Transfer)

When transferring experts between **any two GPUs** — ROCm↔ROCm, CUDA↔CUDA, or CUDA↔ROCm — the batch-packed VNNI layout is byte-identical. **No repacking or conversion is needed.** The transfer is a pure memcpy of 4 arrays (vnni, scales, mins, emins):

```cpp
/// Transfer a single expert's packed weights between GPU devices.
/// Works across same-vendor (D2D/peer) and cross-vendor (host-staged).
struct GPUExpertTransfer {
    /// Same-vendor D2D: hipMemcpyPeerAsync or cudaMemcpyPeerAsync
    /// Cross-vendor or no peer access: host-staged (D2H + H2D)
    static bool transferExpert(
        const GPUBatchExpertPointers& src_ptrs,
        GPUBatchExpertPointers& dst_ptrs,
        DeviceId src_device, DeviceId dst_device,
        size_t vnni_bytes, size_t scales_bytes,
        size_t mins_bytes, size_t emins_bytes,
        void* stream);  // hipStream_t or cudaStream_t
};
```

**Implementation strategy:**
1. **Same vendor + peer access** (ROCm↔ROCm on xGMI, CUDA↔CUDA on NVLink/PCIe): use `hipMemcpyPeerAsync()` / `cudaMemcpyPeerAsync()` — direct DMA, no host involvement
2. **Same vendor, no peer access**: allocate pinned host bounce buffer, D2H → H2D on prep stream
3. **Cross-vendor** (CUDA↔ROCm): always host-staged — D2H from source vendor runtime → H2D via target vendor runtime. The pinned host buffer is the bridge between HIP and CUDA memory spaces
4. Per-expert transfer: 4 memcpy calls (vnni, scales, mins, emins) — total ~4.7MB per expert for Qwen3.5-35B (14336×2048 Q4_0)
5. After transfer: create the target vendor's GEMM kernel instance pointing to new device pointers

**No repacking overhead:** The per-expert batch-packed VNNI layout is byte-identical across all GPU vendors. This is analogous to how CPU→CPU socket transfer uses `CPUPackedWeights::clone()` (memcpy of interleaved data) without re-running `packVnniBlock()`.

**Cross-vendor heterogeneous TP:** This enables heterogeneous TP domains mixing CUDA and ROCm GPUs. When the dynamic rebalancing controller decides to migrate an expert from a CUDA device to a ROCm device (or vice versa), the transfer is a trivial host-staged memcpy with no format translation. The `ExpertWeightTransfer` MPI protocol already treats blobs as opaque bytes — a CUDA-serialized blob can be uploaded directly to a ROCm device.

**Cost:**

| Path | Time per Expert (4.7MB) | Notes |
|------|-------------------------|-------|
| Same-vendor peer DMA (xGMI/NVLink) | ~0.06ms | Direct hardware DMA |
| Same-vendor peer DMA (PCIe) | ~0.5ms | PCIe Gen3 ×16 |
| Same-vendor host-staged (no peer) | ~1.0ms | D2H + H2D via pinned host |
| Cross-vendor host-staged (CUDA↔ROCm) | ~1.2ms | D2H + H2D across runtime boundaries |
| Full repack from raw tensors (current) | ~25ms | `packVnniBlock()` × all blocks + H2D |

For 8 experts: ~4ms worst-case (host-staged) vs ~200ms (full repack) — **50× faster**.

#### Transfer Mode 2: CPU → GPU (GPU-Side Deinterleave)

When experts migrate from a CPU socket to a GPU (ROCm or CUDA), the CPU's `native_interleaved` blob must be converted to GPU separated layout. **The GPU performs this conversion** — the CPU simply uploads the raw interleaved bytes:

**Pipeline:**
```
1. CPU: memcpy raw native_interleaved blob to pinned host buffer (plain copy, no conversion)
2. GPU: hipMemcpyAsync(d_interleaved_buf, h_pinned, interleaved_bytes, H2D, prep_stream)
3. GPU: launch deinterleave_vnni_weights kernel on prep_stream
   → reads d_interleaved_buf, writes to d_vnni, d_scales, d_mins, d_emins (separated)
4. GPU: hipEventRecord(prep_done, prep_stream)
```

**GPU deinterleave kernel** (`ROCmMoEFormatConversion.hip` / `CUDAMoEFormatConversion.cu`):

```cpp
/// GPU kernel: deinterleave CPU-packed VNNI blocks into separated GPU arrays.
/// One thread per quantization block — massively parallel scatter.
///
/// CPU interleaved layout per block (stride = interleaved_block_stride):
///   [payload_bytes data | FP16 scale | FP16 min (if asymmetric) | padding]
///
/// GPU separated layout:
///   payload_array[block_idx * payload_bytes .. (block_idx+1) * payload_bytes]
///   scales_array[block_idx]
///   mins_array[block_idx]
__global__ void deinterleave_vnni_weights(
    const uint8_t* __restrict__ interleaved,    // Raw CPU interleaved blob (on device)
    uint8_t* __restrict__ out_vnni,             // Separated payload array
    uint16_t* __restrict__ out_scales,          // Separated scales array
    uint16_t* __restrict__ out_mins,            // Separated mins array (nullptr if symmetric)
    uint32_t* __restrict__ out_emins,           // Separated emins array (nullptr if none)
    int total_blocks,                           // blocks_per_row × rows
    int interleaved_block_stride,               // Bytes per interleaved block
    int payload_bytes,                          // Payload size per block
    bool has_mins,                              // Asymmetric quantization?
    bool has_emins)                             // Extended mins?
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= total_blocks) return;

    const uint8_t* src_block = interleaved + bid * interleaved_block_stride;

    // Extract payload (bulk of the data)
    memcpy(out_vnni + bid * payload_bytes, src_block, payload_bytes);

    // Extract scale (FP16, immediately after payload)
    out_scales[bid] = *reinterpret_cast<const uint16_t*>(src_block + payload_bytes);

    // Extract min if asymmetric
    if (has_mins && out_mins)
        out_mins[bid] = *reinterpret_cast<const uint16_t*>(src_block + payload_bytes + 2);

    // Extract emins if present
    if (has_emins && out_emins)
        out_emins[bid] = *reinterpret_cast<const uint32_t*>(src_block + payload_bytes + 4);
}
```

**Why GPU-side conversion?**
- GPU memory bandwidth (~900 GB/s MI50) vs CPU (~40 GB/s DDR4) — **22× faster** for the scatter operation
- The deinterleave is embarrassingly parallel (one thread per block, no dependencies)
- CPU stays free for MPI communication, tokenization, or other work during rebalancing
- The raw interleaved blob is already the CPU's native wire format — no CPU-side transformation needed

**Cost:** ~0.15ms kernel time (4.7MB at ~30 GB/s effective scatter throughput) + ~0.5ms H2D upload = **~0.65ms total** per expert. Compare to CPU-side deinterleave: ~2ms (scatter on DDR4) + ~0.5ms H2D = ~2.5ms. GPU-side is **~4× faster end-to-end**.

#### Transfer Mode 3: GPU → CPU (GPU-Side Re-interleave + D2H)

When experts migrate from GPU back to a CPU socket, the GPU first re-interleaves its separated arrays into CPU interleaved format, then sends the result:

**Pipeline:**
```
1. GPU: launch interleave_vnni_weights kernel on prep_stream
   → reads d_vnni, d_scales, d_mins, d_emins; writes to d_interleaved_buf
2. GPU: hipMemcpyAsync(h_pinned, d_interleaved_buf, interleaved_bytes, D2H, prep_stream)
3. GPU: hipStreamSynchronize(prep_stream)
4. CPU: memcpy from pinned host buffer → CPUNativeVNNIPackedWeights.native_interleaved
5. CPU: Create CPUNativeVNNIGemmKernel from received interleaved data
```

**GPU re-interleave kernel:**

```cpp
/// GPU kernel: re-interleave separated GPU arrays into CPU interleaved format.
/// Inverse of deinterleave_vnni_weights — one thread per block.
__global__ void interleave_vnni_weights(
    const uint8_t* __restrict__ in_vnni,
    const uint16_t* __restrict__ in_scales,
    const uint16_t* __restrict__ in_mins,
    const uint32_t* __restrict__ in_emins,
    uint8_t* __restrict__ out_interleaved,
    int total_blocks,
    int interleaved_block_stride,
    int payload_bytes,
    bool has_mins,
    bool has_emins)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= total_blocks) return;

    uint8_t* dst_block = out_interleaved + bid * interleaved_block_stride;

    // Write payload
    memcpy(dst_block, in_vnni + bid * payload_bytes, payload_bytes);

    // Write scale
    *reinterpret_cast<uint16_t*>(dst_block + payload_bytes) = in_scales[bid];

    // Write min if asymmetric
    if (has_mins && in_mins)
        *reinterpret_cast<uint16_t*>(dst_block + payload_bytes + 2) = in_mins[bid];

    // Write emins if present
    if (has_emins && in_emins)
        *reinterpret_cast<uint32_t*>(dst_block + payload_bytes + 4) = in_emins[bid];

    // Zero padding (if stride > payload + metadata)
    // Not strictly necessary but keeps wire format clean
}
```

**Cost:** ~0.15ms kernel + ~0.5ms D2H = **~0.65ms total** per expert.

#### Async Transfer Pipeline

All transfer modes use the same async pipeline to overlap with inference:

```cpp
struct AsyncExpertTransferContext {
    void* prep_stream;                    // hipStream_t or cudaStream_t
    void* prep_done_event;                // hipEvent_t or cudaEvent_t
    std::vector<int> pending_experts;     // Expert IDs being transferred
    DeviceId target_device;               // Destination device
    bool in_flight = false;
};

bool MoEExpertWeightService::asyncTransferExperts(
    MoEWeightContext& ctx,
    const std::vector<bool>& new_mask,
    const TransferManifest& manifest,     // which experts, from where
    AsyncExpertTransferContext& async_ctx)
{
    for (const auto& [expert_id, source] : manifest.incoming) {
        if (source.device_type.is_gpu() && ctx.device_id.is_gpu()) {
            // GPU → GPU: direct memcpy (same or cross-vendor, no conversion)
            GPUExpertTransfer::transferExpert(
                source.device_ptrs, dst_ptrs,
                source.device_id, ctx.device_id,
                vnni_bytes, scales_bytes, mins_bytes, emins_bytes,
                async_ctx.prep_stream);
        }
        else if (source.device_type == DeviceType::CPU && ctx.device_id.is_gpu()) {
            // CPU → GPU: upload raw interleaved blob, GPU deinterleaves
            hipMemcpyAsync(d_interleaved_tmp, source.interleaved_blob,
                          interleaved_bytes, hipMemcpyHostToDevice,
                          async_ctx.prep_stream);
            launch_deinterleave_vnni_weights(
                d_interleaved_tmp,
                d_vnni, d_scales, d_mins, d_emins,
                total_blocks, interleaved_block_stride, payload_bytes,
                has_mins, has_emins, async_ctx.prep_stream);
        }
        else if (source.device_type.is_gpu() && ctx.device_id.is_cpu()) {
            // GPU → CPU: GPU re-interleaves, then D2H
            launch_interleave_vnni_weights(
                source.device_ptrs.d_vnni,
                source.device_ptrs.d_scales,
                source.device_ptrs.d_mins,
                source.device_ptrs.d_emins,
                d_interleaved_tmp,
                total_blocks, interleaved_block_stride, payload_bytes,
                has_mins, has_emins, source.stream);
            hipMemcpyAsync(h_pinned, d_interleaved_tmp, interleaved_bytes,
                          hipMemcpyDeviceToHost, source.stream);
            hipStreamSynchronize(source.stream);
            // CPU: wrap received interleaved blob as CPUNativeVNNIPackedWeights
            buildCPUPackedWeightsFromInterleaved(h_pinned, ctx, expert_id);
        }
    }

    // Record completion event (for GPU→GPU and CPU→GPU paths)
    hipEventRecord(async_ctx.prep_done_event, async_ctx.prep_stream);

    // Compute stream waits for prep before using new experts
    hipStreamWaitEvent(compute_stream, async_ctx.prep_done_event, 0);

    // Create GEMM kernel instances pointing to new device buffers
    for (const auto& [expert_id, _] : manifest.incoming) {
        createGemmEngineForTransferredExpert(ctx, expert_id);
    }

    // Update expert mask on device
    moe_kernel->updateExpertMaskDevice(new_mask);

    return true;
}
```

#### Serialization Format Extension

The current `PackedWeightsSerialization` only handles `CPU_NATIVE_VNNI`. For cross-device transfer, we extend it to support `GPU_BATCH_VNNI`:

```cpp
enum class PackedWeightsFormat : uint32_t {
    CPU_NATIVE_VNNI  = 1,       // Existing: interleaved blocks
    GPU_BATCH_VNNI   = 2,       // NEW: separated {payload, scales, mins, emins}
                                // Used by BOTH CUDA and ROCm (identical layout)
};
```

Since CUDA and ROCm batch formats are byte-identical, there is a single `GPU_BATCH_VNNI` wire format — no need for separate CUDA and ROCm format IDs. A blob serialized from a CUDA device can be uploaded directly to a ROCm device and vice versa.

**Wire format for `GPU_BATCH_VNNI`:**
```
[PackedWeightsHeader — 64 bytes]
  format = GPU_BATCH_VNNI
  N, K, blocks_per_row, codebook_id, payload_bytes, is_asymmetric, has_emins
[PackedWeightsSectionTable — 32 bytes]
  interleaved_size = 0 (not used)
  payload_size = vnni_bytes
  int8_flat_size = scales_bytes + mins_bytes + emins_bytes
  native_blocks_size = 0
[payload — vnni_bytes]
[scales — scales_bytes]
[mins — mins_bytes (0 if symmetric)]
[emins — emins_bytes (0 if no emins)]
```

This enables MPI transfer of GPU-packed weights between ranks of any vendor without re-serializing through the CPU format. A CUDA device sends its batch-packed blob; a receiving ROCm device uploads it directly.

#### ExpertWeightTransfer Protocol Extension

The existing `ExpertWeightTransfer::transferAllLayers()` MPI protocol sends `ExpertWeightBlobs` (opaque byte vectors). The only change needed is that `serializeExpert()` and `registerTransferredExpert()` now dispatch based on source/target device type:

```cpp
ExpertWeightBlobs MoEExpertWeightService::serializeExpert(
    const MoEWeightContext& ctx, int expert_id)
{
    if (ctx.device_id.is_gpu()) {
        if (/* target is CPU */) {
            // GPU → CPU: re-interleave on GPU, D2H, wrap as CPU_NATIVE_VNNI
            return serializeExpertGPU_ForCPU(ctx, expert_id);
        } else {
            // GPU → GPU: D2H the separated arrays, wrap as GPU_BATCH_VNNI
            return serializeExpertGPU(ctx, expert_id);
        }
    } else {
        // CPU → any: send raw interleaved blob as CPU_NATIVE_VNNI
        // Receiving GPU will deinterleave on its side
        return serializeExpertCPU(ctx, expert_id);
    }
}

bool MoEExpertWeightService::registerTransferredExpert(
    MoEWeightContext& ctx, int expert_id, const ExpertWeightBlobs& blobs)
{
    auto format = packed_weights_serialization::detectFormat(blobs.gate);

    if (ctx.device_id.is_gpu()) {
        if (format == PackedWeightsFormat::GPU_BATCH_VNNI) {
            // GPU blob → GPU target: direct H2D upload (no conversion)
            return registerExpertGPU_DirectUpload(ctx, expert_id, blobs);
        } else if (format == PackedWeightsFormat::CPU_NATIVE_VNNI) {
            // CPU blob → GPU target: H2D raw interleaved, deinterleave on GPU
            return registerExpertGPU_FromCPU(ctx, expert_id, blobs);
        }
    } else { // CPU target
        if (format == PackedWeightsFormat::CPU_NATIVE_VNNI) {
            // CPU blob → CPU target: existing deserialization path
            return registerTransferredExpertCPU(ctx, expert_id, blobs);
        } else if (format == PackedWeightsFormat::GPU_BATCH_VNNI) {
            // GPU blob → CPU target: this shouldn't normally happen
            // (serializeExpert for CPU targets produces CPU_NATIVE_VNNI)
            // Fallback: upload to temporary GPU buffer, re-interleave, D2H
            return registerExpertCPU_FromGPUBlob(ctx, expert_id, blobs);
        }
    }
    return false;
}
```

**Note on serialize dispatch:** When a GPU serializes an expert destined for a CPU target, it runs the `interleave_vnni_weights` kernel first and produces a `CPU_NATIVE_VNNI` blob. When destined for another GPU (any vendor), it produces a `GPU_BATCH_VNNI` blob. The `TransferManifest` provides the target device type so the serializer knows which format to produce. This means the `GPU_BATCH_VNNI → CPU` path in `registerTransferredExpert` is only a defensive fallback — it should not occur in normal operation.

#### Files to Create

| File | Purpose |
|------|---------|
| `src/v2/kernels/rocm/moe/ROCmMoEFormatConversion.h` | GPU deinterleave/re-interleave kernel API (ROCm) |
| `src/v2/kernels/rocm/moe/ROCmMoEFormatConversion.hip` | HIP kernels: `deinterleave_vnni_weights`, `interleave_vnni_weights` |
| `src/v2/kernels/cuda/moe/CUDAMoEFormatConversion.h` | GPU deinterleave/re-interleave kernel API (CUDA) |
| `src/v2/kernels/cuda/moe/CUDAMoEFormatConversion.cu` | CUDA kernels (identical logic to HIP) |
| `src/v2/execution/moe/GPUExpertTransfer.h` | Unified GPU↔GPU expert memcpy (any vendor) |
| `src/v2/execution/moe/GPUExpertTransfer.cpp` | Implementation (peer DMA, host-staged, cross-vendor) |
| `src/v2/execution/moe/AsyncExpertTransfer.h` | Async transfer context + pipeline |
| `src/v2/execution/moe/AsyncExpertTransfer.cpp` | Async transfer implementation |
| `tests/v2/unit/kernels/rocm/moe/Test__ROCmMoEFormatConversion.cpp` | Deinterleave/re-interleave round-trip |
| `tests/v2/unit/kernels/cuda/moe/Test__CUDAMoEFormatConversion.cpp` | CUDA format conversion tests |
| `tests/v2/unit/moe/Test__GPUExpertTransfer.cpp` | GPU↔GPU transfer tests (D2D, staged, cross-vendor) |
| `tests/v2/unit/moe/Test__AsyncExpertTransfer.cpp` | Async transfer pipeline tests |

#### Files to Modify

| File | Change |
|------|--------|
| `src/v2/execution/moe/MoEExpertWeightService.h` | Add `asyncTransferExperts()`, cross-device serialize/register |
| `src/v2/execution/moe/MoEExpertWeightService.cpp` | Implement dispatch by source/target device type |
| `src/v2/kernels/PackedWeightsSerialization.h` | Add `GPU_BATCH_VNNI` format, `detectFormat()` |
| `src/v2/kernels/rocm/ROCmWeightPacker.h` | Add `uploadToDeviceAsync()`, per-expert upload |
| `src/v2/kernels/cuda/CUDAWeightPacker.h` | Add `uploadToDeviceAsync()`, per-expert upload |
| `src/v2/execution/moe/ExpertWeightTransfer.h` | Extend blob format metadata, target device hint |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Use async transfer path |
| `src/v2/CMakeLists.txt` | New sources (.hip, .cu) |
| `tests/v2/CMakeLists.txt` | New tests |

#### Rollback

- If D2D memcpy fails: fall back to host-staged transfer
- If host-staged transfer fails (OOM): keep old mask, retry at next window
- If GPU format conversion kernel fails: fall back to full repack from raw tensor views
- No partial state — old experts remain functional until full replacement succeeds

#### Transfer Cost Summary

| Transfer Mode | Conversion? | Where? | Time per Expert (14336×2048 Q4_0) | Notes |
|---------------|------------|--------|-----------------------------------|-------|
| GPU → GPU (peer DMA) | No | — | ~0.06ms (xGMI/NVLink) / ~0.5ms (PCIe) | `hipMemcpyPeerAsync` / `cudaMemcpyPeerAsync` |
| GPU → GPU (host-staged, same vendor) | No | — | ~1.0ms | D2H + H2D via pinned host |
| GPU → GPU (host-staged, cross-vendor) | No | — | ~1.2ms | D2H (HIP) + H2D (CUDA) or vice versa |
| CPU → GPU | Deinterleave | GPU | ~0.65ms (H2D + GPU kernel) | `deinterleave_vnni_weights` |
| GPU → CPU | Re-interleave | GPU | ~0.65ms (GPU kernel + D2H) | `interleave_vnni_weights` |
| Full repack (current) | Yes | CPU | ~25ms | `packVnniBlock()` × all blocks + H2D |

#### Tests

- `Test__ROCmMoEFormatConversion_Deinterleave` — Upload CPU interleaved blob, deinterleave on GPU, verify byte-exact match with `packMoEExpertsROCm()` output
- `Test__ROCmMoEFormatConversion_Interleave` — Re-interleave GPU separated → interleaved, D2H, verify matches original CPU `native_interleaved`
- `Test__ROCmMoEFormatConversion_RoundTrip` — CPU → deinterleave → re-interleave → CPU produces identical bytes
- `Test__CUDAMoEFormatConversion_Deinterleave` — Same as ROCm but on CUDA device
- `Test__CUDAMoEFormatConversion_RoundTrip` — Same round-trip on CUDA
- `Test__GPUExpertTransfer_D2D_SameVendor_PeerAccess` — ROCm↔ROCm with peer DMA
- `Test__GPUExpertTransfer_D2D_SameVendor_Staged` — ROCm↔ROCm with host bounce (peer disabled)
- `Test__GPUExpertTransfer_CrossVendor_HostStaged` — CUDA→ROCm via pinned host
- `Test__AsyncExpertTransfer_GPUToGPU` — Async D2D transfer overlaps with inference
- `Test__AsyncExpertTransfer_CPUToGPU_Deinterleave` — Async CPU→GPU with GPU-side conversion
- `Test__AsyncExpertTransfer_GPUToCPU_Interleave` — Async GPU→CPU with GPU-side conversion
- `Test__AsyncExpertTransfer_Rollback` — Simulate OOM, verify old engines still work
- `Test__PackedWeightsSerialization_GPUBatchFormat` — Serialize/deserialize `GPU_BATCH_VNNI` round-trip
- `Test__MoEExpertWeightService_CrossDeviceRegister` — Register CPU-serialized expert on GPU and vice versa

#### Exit Criteria

- GPU↔GPU transfer (any vendor): ~50× faster than full repack (0.5-1.2ms vs 25ms per expert)
- CPU↔GPU transfer with GPU-side conversion: ~38× faster than full repack (0.65ms vs 25ms per expert)
- CUDA↔ROCm cross-vendor transfer works for heterogeneous TP domains
- Format conversion round-trip produces byte-identical results
- GPU deinterleave output matches `packMoEExpertsROCm()` reference
- Async pipeline overlaps transfer with inference (no stall)
- Rollback tested for all failure modes
- Existing parity tests pass (rebalance cycle produces correct inference)

---

## File Inventory

### New Files

| File | Phase | Purpose |
|------|-------|---------|
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.h` | 1 | Kernel header |
| `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp` | 1 | Host-side dispatch |
| `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip` | 1-3 | HIP device kernels |
| `src/v2/kernels/rocm/moe/ROCmMoEFormatConversion.h` | 4 | GPU deinterleave/re-interleave API (ROCm) |
| `src/v2/kernels/rocm/moe/ROCmMoEFormatConversion.hip` | 4 | HIP kernels: deinterleave + re-interleave |
| `src/v2/kernels/cuda/moe/CUDAMoEFormatConversion.h` | 4 | GPU deinterleave/re-interleave API (CUDA) |
| `src/v2/kernels/cuda/moe/CUDAMoEFormatConversion.cu` | 4 | CUDA kernels: deinterleave + re-interleave |
| `src/v2/execution/moe/GPUExpertTransfer.h` | 4 | Unified GPU↔GPU expert memcpy (any vendor) |
| `src/v2/execution/moe/GPUExpertTransfer.cpp` | 4 | Peer DMA, host-staged, cross-vendor impl |
| `src/v2/execution/moe/AsyncExpertTransfer.h` | 4 | Async transfer context + pipeline |
| `src/v2/execution/moe/AsyncExpertTransfer.cpp` | 4 | Async transfer implementation |
| `tests/v2/unit/kernels/rocm/moe/Test__ROCmMoEKernel.cpp` | 1-3 | Kernel unit tests |
| `tests/v2/unit/kernels/rocm/moe/Test__ROCmMoEFormatConversion.cpp` | 4 | ROCm deinterleave/re-interleave round-trip |
| `tests/v2/unit/kernels/cuda/moe/Test__CUDAMoEFormatConversion.cpp` | 4 | CUDA deinterleave/re-interleave round-trip |
| `tests/v2/unit/moe/Test__GPUExpertTransfer.cpp` | 4 | GPU↔GPU transfer tests (D2D, staged, cross-vendor) |
| `tests/v2/unit/moe/Test__AsyncExpertTransfer.cpp` | 4 | Async transfer pipeline tests |
| `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ROCm_SingleDevice_Parity.cpp` | Exit Gate | ROCm single-device parity test |
| `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ROCm_LocalTP_Parity.cpp` | Exit Gate | ROCm LocalTP=2/4 parity tests |

### Modified Files

| File | Phase | Change |
|------|-------|--------|
| `src/v2/kernels/KernelFactory.cpp` | 1 | ROCm MoE kernel dispatch |
| `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp` | 1-2 | Device routing + histogram |
| `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp` | 1, 3 | Device gather/scatter |
| `src/v2/execution/moe/MoEExpertWeightService.h` | 4 | `asyncTransferExperts()`, cross-device serialize/register |
| `src/v2/execution/moe/MoEExpertWeightService.cpp` | 4 | Dispatch by source/target device type |
| `src/v2/kernels/PackedWeightsSerialization.h` | 4 | `GPU_BATCH_VNNI` format, `detectFormat()` |
| `src/v2/kernels/rocm/ROCmWeightPacker.h` | 4 | `uploadToDeviceAsync()`, per-expert upload |
| `src/v2/kernels/cuda/CUDAWeightPacker.h` | 4 | `uploadToDeviceAsync()`, per-expert upload |
| `src/v2/execution/moe/ExpertWeightTransfer.h` | 4 | Extend blob format metadata, target device hint |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | 4 | Async transfer path |
| `src/v2/CMakeLists.txt` | 1, 4 | New sources (.hip, .cu) |
| `tests/v2/CMakeLists.txt` | 1, 4 | New tests |

---

## Performance Projections

### Decode (seq_len = 1, per MoE layer)

| Operation | Current (CPU) | Phase 1 (GPU route) | Phase 2 (GPU hist/mask) |
|-----------|--------------|--------------------|-----------------------|
| Route (gate logits + softmax + top-k) | ~50μs | ~10μs | ~10μs |
| D2H routing indices | ~10μs | 32 bytes ~1μs | 0 (mask on device) |
| Histogram recording | ~2μs (CPU) | ~2μs (CPU, D2H indices) | ~1μs (GPU atomic) |
| Expert mask check | ~1μs (CPU) | ~1μs (CPU) | ~1μs (GPU kernel) |
| Gather (no-op for 1 token) | 0 | 0 | 0 |
| Expert GEMM ×8 | ~800μs | ~800μs | ~800μs |
| Scatter-add ×8 | ~15μs (CPU + D2H) | ~3μs (GPU) | ~3μs (GPU) |
| **Per-layer total** | ~878μs | ~816μs | ~815μs |
| **40-layer total** | ~35.1ms | ~32.6ms | ~32.6ms |

### Prefill (seq_len = 596, per MoE layer)

| Operation | Current (CPU) | Phase 1 | Phase 3 (device grouping) |
|-----------|--------------|---------|--------------------------|
| Route | ~3ms (CPU GEMV) | ~0.5ms (GPU) | ~0.5ms |
| D2H hidden states (for CPU route) | ~5ms | 0 | 0 |
| D2H routing indices (for grouping) | ~0.02ms | ~0.02ms | 0 |
| Grouping | ~0.1ms (CPU) | ~0.1ms (CPU) | ~0.05ms (GPU) |
| Gather | ~0.5ms (CPU memcpy) | ~0.1ms (GPU) | ~0.1ms |
| Expert GEMM | ~8ms | ~8ms | ~8ms |
| Scatter-add | ~0.5ms (CPU) | ~0.1ms (GPU) | ~0.1ms |
| **Per-layer total** | ~17.1ms | ~8.8ms | ~8.75ms |
| **40-layer total** | ~684ms | ~352ms | ~350ms |

**Key insight:** Prefill savings are dominated by eliminating the D2H of hidden states for CPU routing. The gate logits computation on CPU requires the full hidden state on host (~596 × 2048 × 4 = ~4.9MB D2H per layer). GPU routing eliminates this entirely.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Softmax numerical divergence (GPU vs CPU) | Use Kahan summation in GPU softmax; parity test with tolerance ≤ 1e-6 |
| Top-k selection order differs (ties) | Break ties by expert ID (deterministic); parity test exact match |
| atomicAdd contention in histogram | 256 experts × single atomic each — no contention (non-conflicting addresses) |
| Async transfer race with compute stream | hipStreamWaitEvent provides ordering; no manual synchronization needed |
| OOM during async weight upload | Rollback to old mask; no partial state exposed |
| Grouping kernel correctness (Phase 3) | Atomic scatter is well-understood; parity test against CPU grouping |
| Performance regression in prefill | Gate logits kernel must match or beat hipBLAS GEMV for (256, 2048) — benchmark both |
| GPU deinterleave produces wrong data | Round-trip test: CPU → GPU deinterleave → GPU re-interleave → CPU must produce byte-identical `native_interleaved` |
| GPU D2D fails without peer access | Automatic fallback to staged transfer via pinned host; no user intervention |
| Cross-vendor transfer (CUDA↔ROCm) fails | Always host-staged — no peer access assumption; pinned host buffer bridges HIP/CUDA runtimes |
| Mixed-format MPI transfer (CPU sends to GPU rank) | `detectFormat()` on wire header dispatches correct path; GPU-side deinterleave for CPU blobs |
| Interleaved block stride mismatch in GPU kernel | Assert stride matches `FormatInfo::interleavedBlockStride()` at kernel launch; CPU sends stride in blob header |

## Validation Strategy

### Intermediate Phase Gates (Phases 1–4)

After completing each phase, the gate to proceed is:

```bash
# All unit tests pass (fast, targeted)
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
```

That's it. No parity tests, no integration tests between phases. Unit tests are the sole intermediate gate — they run in seconds and catch regressions without burning an hour on full model inference. Parity and integration testing are deferred to the final project exit gate below.

### Final Project Exit Gate: PyTorch Parity Tests

The project is declared **functionally complete** when all of the following parity tests pass. These compare Llaminar ROCm MoE inference layer-by-layer against PyTorch FP32 reference snapshots.

**Existing CPU parity tests** (must continue to pass — no regressions):

| CTest Name | Config | Description |
|------------|--------|-------------|
| `V2_Integration_Parity_Qwen35MoE_SingleDevice.*` | `Qwen35MoE_35B_CPU_KV_FP16` | Single-socket CPU, MPI_PROCS=1 |
| `V2_Integration_Parity_Qwen35MoE_NodeLocalTP.*` | `NodeLocalTP_2xMPI_CPU_35B_MoE` | 2-way cross-socket CPU via MPI, MPI_PROCS=2 |

**New ROCm parity tests** (to be created as part of this project):

| CTest Name | Config | Devices | Collective | Description |
|------------|--------|---------|------------|-------------|
| `V2_Integration_Parity_Qwen35MoE_ROCm_SingleDevice.*` | `Qwen35MoE_35B_ROCm_SingleDevice` | `rocm:0` | — | Single MI50/MI100, MPI_PROCS=1 |
| `V2_Integration_Parity_Qwen35MoE_ROCm_LocalTP.*` | `Qwen35MoE_35B_ROCm_LocalTP2` | `rocm:0, rocm:1` | RCCL | 2-way Local TP, MPI_PROCS=1 |
| `V2_Integration_Parity_Qwen35MoE_ROCm_LocalTP.*` | `Qwen35MoE_35B_ROCm_LocalTP4` | `rocm:0..3` | RCCL | 4-way Local TP, MPI_PROCS=1 |

**Note:** All ROCm tests use **LocalTP** (not NodeLocalTP) because all ROCm cards are on a single NUMA node / MPI rank — no cross-socket MPI needed.

**Thresholds for ROCm tests** — match the existing CPU SingleDevice thresholds:

| Threshold | SingleDevice | LocalTP (2-way) | LocalTP (4-way) |
|-----------|-------------|-----------------|-----------------|
| `cosine_threshold` | 0.96 | 0.90 | 0.90 |
| `decode_cosine_threshold` | 0.98 | 0.80 | 0.80 |
| `early_layers_count` | 6 | 6 | 6 |
| `min_early_layers_passed` | 5 | 5 | 5 |
| `kl_threshold` | 0.03 | 0.03 | 0.03 |
| `min_top1_accuracy` | 0.80 | 0.80 | 0.80 |
| `min_top5_accuracy` | 0.60 | 0.80 | 0.80 |
| `pytorch_top1_in_topk` | 3 | 4 | 4 |

SingleDevice ROCm thresholds match `Qwen35MoE_35B_CPU_KV_FP16` exactly. LocalTP thresholds match `NodeLocalTP_2xMPI_CPU_35B_MoE` (relaxed cosine due to sharded intermediates, tighter top-5 since TP shouldn't degrade token predictions).

**LocalTP excluded stages** — same as the existing `kTPExcludedStages` list from the dense Qwen3.5 LocalTP tests, extended with MoE-specific sharded stages:

```cpp
static const std::vector<std::string> kMoETPExcludedStages = {
    // Standard attention projections (from kTPExcludedStages)
    "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_ROPE", "K_ROPE", "ATTENTION_CONTEXT",
    "FFN_GATE", "FFN_UP", "FFN_SWIGLU",
    "QKV_PROJECTION",
    "GDN_DELTA_RULE_OUTPUT", "GDN_NORM_GATE_OUTPUT",
    // MoE-specific: expert outputs are per-shard before allreduce
    "MOE_EXPERT_FFN_GATE", "MOE_EXPERT_FFN_UP", "MOE_EXPERT_FFN_SWIGLU",
    "MOE_EXPERT_FFN_DOWN",
    "MOE_SCATTER_ADD",  // partial scatter before EP allreduce
};
```

**Files to create:**

| File | Purpose |
|------|---------|
| `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ROCm_SingleDevice_Parity.cpp` | Single-device ROCm parity test |
| `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ROCm_LocalTP_Parity.cpp` | LocalTP=2 and LocalTP=4 ROCm parity tests |

**Running the exit gate:**

```bash
# Build integration tests
cmake --build build_v2_integration --parallel

# Run all Qwen3.5 MoE parity tests (CPU + ROCm)
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoE" --output-on-failure --verbose

# Or run ROCm-only parity tests
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoE_ROCm" --output-on-failure --verbose
```

**Success criteria:** All PrefillParity and DecodeParity tests pass for every config. This declares functional correctness — performance tuning follows in Phase 5.

---

## Phase 5: Performance Tuning Sprint

**Prerequisite:** All parity tests pass (project declared functionally correct).

**Goal:** Measure, compare, and tune ROCm MoE kernel/stage performance against the CPU baseline. Ensure GPU stages achieve proportional speedup and no individual stage is a bottleneck.

### Step 1: CPU Baseline Profiling

Establish per-kernel/per-stage timing baselines on CPU. Ensure all MoE stages appear in the profiling breakdown — fix any missing entries.

```bash
# Single-socket CPU baseline
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark \
  -d cpu:0 -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

# All-socket CPU baseline (2-socket TP)
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark \
  -d cpu -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

**Expected output:** Per-stage timing breakdown showing every MoE stage individually:
- `MOE_ROUTING` (gate logits + softmax + top-k)
- `MOE_GATHER` (token collection by expert)
- `MOE_EXPERT_GEMM` (expert FFN ×8 per layer)
- `MOE_SCATTER_ADD` (weighted accumulation)
- `MOE_SHARED_EXPERT` (shared expert FFN)
- `MOE_SHARED_EXPERT_GATE` (sigmoid gate + scale)

**Action item:** If any MoE stage is missing from the profiling output or is aggregated into a parent stage, add the missing `LLAMINAR_PROFILE_SCOPE()` markers to the stage implementations. All MoE stages must be individually visible for meaningful A/B comparison.

**Deliverables:**
- `benchmark_results/cpu_single_socket_moe_baseline.txt` — Single-socket timing breakdown
- `benchmark_results/cpu_dual_socket_moe_baseline.txt` — Dual-socket TP timing breakdown

### Step 2: ROCm Profiling

Run the same benchmarks on ROCm. Ensure all MoE stages appear in the GPU profiling breakdown — add `LLAMINAR_PROFILE_SCOPE()` markers to ROCm kernel dispatch paths if needed.

```bash
# Single-device ROCm
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark \
  -d rocm:0 -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

# 2-way LocalTP ROCm
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark \
  --tp-devices "rocm:0,rocm:1" -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

**Action item:** Verify every MoE stage is profiled. The ROCm kernel dispatch paths (new in this project) may need explicit profiling annotations. Fix any missing stages before proceeding to comparison.

**Deliverables:**
- `benchmark_results/rocm_single_device_moe.txt` — Single-device ROCm timing breakdown
- `benchmark_results/rocm_tp2_moe.txt` — 2-way LocalTP ROCm timing breakdown

### Step 3: CPU vs ROCm Stage Comparison

Compare the **percentage of total inference time** each MoE stage consumes on CPU vs ROCm. The distribution should be roughly similar — if a ROCm stage takes a disproportionately larger share of time, it needs tuning.

**Analysis framework:**

| Stage | CPU % (prefill) | CPU % (decode) | ROCm % (prefill) | ROCm % (decode) | Delta | Action |
|-------|----------------|---------------|-------------------|-----------------|-------|--------|
| MOE_ROUTING | X% | X% | Y% | Y% | ΔZ% | Tune if ΔZ > 5% |
| MOE_GATHER | ... | ... | ... | ... | ... | ... |
| MOE_EXPERT_GEMM | ... | ... | ... | ... | ... | ... |
| MOE_SCATTER_ADD | ... | ... | ... | ... | ... | ... |
| MOE_SHARED_EXPERT | ... | ... | ... | ... | ... | ... |
| MOE_SHARED_EXPERT_GATE | ... | ... | ... | ... | ... | ... |

**Tuning criteria:**
- If a ROCm stage takes >5 percentage points more of total time than its CPU counterpart, it is a **tuning candidate**
- Focus on **decode** first (latency-sensitive path), then **prefill**
- The expert GEMM stage should dominate both CPU and ROCm — if it doesn't on ROCm, something is wrong with the dispatch overhead

**Common tuning targets:**
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| MOE_ROUTING high on ROCm | Gate logits GEMV not using hipBLAS/rocBLAS | Replace custom kernel with rocBLAS GEMV for large vocab |
| MOE_GATHER/SCATTER high on ROCm | Excess kernel launch overhead for 8 experts | Fuse gather into expert GEMM launch |
| MOE_EXPERT_GEMM lower % on ROCm than CPU | Other stages inflated — good GEMM perf | Investigate inflated stages |
| High D2H/H2D in profiling | Stage still using host-side path | Verify device-resident path is active |

### Step 4: Targeted Kernel Tuning

For any stage identified in Step 3 as a bottleneck:

1. **nsys timeline** — Verify kernel launch counts and durations:
   ```bash
   sudo /usr/local/cuda/bin/nsys profile -t cuda --stats=true \
     -o /tmp/moe_trace -f true \
     ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d rocm:0 \
     -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf -p "test" -n 10
   ```

2. **ncu deep analysis** — For specific kernels:
   ```bash
   sudo -E /usr/local/cuda/bin/ncu \
     --kernel-name "rocm_moe_gate_logits" \
     --launch-skip 1 --launch-count 1 \
     --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
     -o /tmp/moe_kernel_ncu -f \
     ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d rocm:0 \
     -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf -p "test" -n 1
   ```

3. **Iterate:** Fix → re-benchmark → compare → repeat until stage timing distribution matches CPU baseline.

### Performance Exit Criteria

| Metric | Target |
|--------|--------|
| Decode tok/s (single ROCm) | ≥ 1.5× single-socket CPU baseline |
| Prefill tok/s (single ROCm) | ≥ 2× single-socket CPU baseline |
| Decode tok/s (2-way ROCm TP) | ≥ 2× single-socket CPU baseline |
| No ROCm MoE stage > 5% above CPU % share | All stages proportional |
| No unexpected D2H/H2D in decode path | Zero large transfers during decode |
