# MoE Expert Placement & Dynamic Offloading — Design Sketch

**Date**: April 2, 2026
**Author**: David Sanftenberg
**Status**: Sketch / RFC

---

## 1. Problem Statement

Mixture-of-Experts (MoE) models (DeepSeek-V3, Mixtral, DBRX, Qwen-MoE) have 64–256 experts per layer, of which only 2–8 are active per token. The total expert weight footprint exceeds any single GPU's VRAM, but per-token compute is sparse.

**Requirements:**

1. **Expert offloading to CPU** — store cold experts on host, transfer on demand
2. **Expert offloading to TP domains** — distribute experts across multi-socket CPU TP or mixed GPU domains (e.g., attention on 3090s, experts on Mi50s)
3. **Dynamic rebalancing** — migrate frequently-used experts to faster devices based on usage histograms

**Design constraint**: fit naturally into Llaminar's existing graph system (GraphSchema → GraphResolver → ComputeGraph → DeviceGraphExecutor) with minimal core changes.

---

## 2. Architecture Overview

```
                         ┌─────────────────────────────┐
                         │    OrchestrationConfig       │
                         │  --expert-domain "gpu_fast"  │
                         │  --expert-offload cpu        │
                         │  --expert-rebalance adaptive │
                         └─────────────┬───────────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │      ExpertPlacementMap      │
                         │  expert_id → PlacementEntry  │
                         │  { device, domain, state }   │
                         └─────────────┬───────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
   ┌────────▼────────┐    ┌───────────▼──────────┐    ┌─────────▼─────────┐
   │  MoERouterStage  │    │  MoEDispatchStage    │    │ MoECombineStage   │
   │  (unchanged)     │    │  (NEW: scatter tokens │    │ (weighted merge)  │
   │                  │    │   to expert devices)  │    │                   │
   └──────────────────┘    └───────────┬──────────┘    └───────────────────┘
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                    ┌─────▼────┐ ┌────▼─────┐ ┌───▼──────┐
                    │Expert(0) │ │Expert(1) │ │Expert(N) │
                    │cuda:0    │ │cpu:0     │ │rocm:1    │
                    └──────────┘ └──────────┘ └──────────┘
                                       │
                         ┌─────────────▼───────────────┐
                         │   ExpertUsageTracker         │
                         │   histograms → rebalance     │
                         └─────────────────────────────┘
```

---

## 3. Core Data Structures

### 3.1 ExpertPlacementMap

Central authority for where every expert lives. Consulted by the graph builder and by the dynamic rebalancer.

```cpp
// src/v2/execution/moe/ExpertPlacementMap.h

enum class ExpertResidency {
    GPU_HOT,        // Weights resident on GPU VRAM — no transfer needed
    CPU_WARM,       // Weights on host — H2D on demand (pinned memory)
    CPU_COLD,       // Weights on host, unpinned — slower path
    DOMAIN_REMOTE,  // Weights on another device in a TP domain
};

struct ExpertPlacement {
    int expert_id;
    int layer_idx;
    DeviceId home_device;           // Where weights currently reside
    DeviceId execution_device;      // Where this expert will execute
    ExpertResidency residency;
    std::string domain_name;        // Empty if no domain — "expert_tp", "mi50_pool", etc.
    float weight_bytes;             // Memory footprint of expert weights
};

class ExpertPlacementMap {
public:
    // Query
    const ExpertPlacement& getPlacement(int layer_idx, int expert_id) const;
    std::vector<ExpertPlacement> getExpertsOnDevice(DeviceId device) const;
    std::vector<ExpertPlacement> getExpertsInDomain(const std::string& domain) const;

    // Batch query for graph building
    DeviceId executionDeviceFor(int layer_idx, int expert_id) const;
    bool needsTransfer(int layer_idx, int expert_id, DeviceId target) const;

    // Mutation (used by rebalancer)
    void migrateExpert(int layer_idx, int expert_id, DeviceId new_device,
                       ExpertResidency new_residency);

    // Initialization
    static ExpertPlacementMap createUniform(int n_layers, int n_experts,
                                            DeviceId device);
    static ExpertPlacementMap createFromPolicy(
        const ExpertOffloadPolicy& policy,
        const ModelArchConfig& arch,
        const std::vector<DeviceId>& available_devices);

private:
    // [layer_idx][expert_id] → placement
    std::vector<std::vector<ExpertPlacement>> placements_;
};
```

### 3.2 ExpertOffloadPolicy

User-facing configuration for expert placement strategy. Parsed from CLI/YAML.

```cpp
// src/v2/execution/moe/ExpertOffloadPolicy.h

enum class ExpertOffloadMode {
    NONE,           // All experts on primary device
    CPU,            // Overflow experts to CPU
    DOMAIN,         // Distribute across named domain
    SPLIT,          // Top-K hot experts on GPU, rest on CPU
    ADAPTIVE,       // Dynamic rebalancing based on histograms
};

struct ExpertOffloadPolicy {
    ExpertOffloadMode mode = ExpertOffloadMode::NONE;

    // For CPU offloading
    int gpu_expert_budget = -1;         // Max experts on GPU (-1 = auto based on VRAM)
    bool pin_cpu_expert_memory = true;  // Use pinned memory for faster H2D

    // For domain offloading
    std::string expert_domain_name;     // Domain from --define-domain
    std::string attention_domain_name;  // Attention runs on a different domain

    // For adaptive rebalancing
    int rebalance_interval_tokens = 1024;  // Re-evaluate placement every N tokens
    float promotion_threshold = 0.15;      // Promote expert if usage > threshold
    float demotion_threshold = 0.02;       // Demote expert if usage < threshold
    int min_samples_before_rebalance = 256;
};
```

### 3.3 ExpertUsageTracker

Maintains per-expert usage histograms for dynamic rebalancing.

```cpp
// src/v2/execution/moe/ExpertUsageTracker.h

class ExpertUsageTracker {
public:
    ExpertUsageTracker(int n_layers, int n_experts, int window_size = 4096);

    // Called by MoERouterStage after computing routing decisions
    void recordRouting(int layer_idx, const std::vector<int>& selected_experts,
                       int token_count);

    // Query
    float getUsageRate(int layer_idx, int expert_id) const;
    std::vector<std::pair<int, float>> getHotExperts(int layer_idx,
                                                      float threshold) const;
    std::vector<std::pair<int, float>> getColdExperts(int layer_idx,
                                                      float threshold) const;

    // Snapshot for rebalancer
    struct LayerUsageSnapshot {
        std::vector<float> usage_rates;  // [n_experts]
        int total_tokens;
    };
    LayerUsageSnapshot snapshot(int layer_idx) const;

    // Reset (e.g., after rebalance)
    void resetWindow();

private:
    // Sliding window of token counts per expert
    // [layer_idx][expert_id] → count within current window
    std::vector<std::vector<std::atomic<int>>> counts_;
    std::vector<std::atomic<int>> total_tokens_;
    int window_size_;
};
```

---

## 4. Graph System Integration

### 4.1 GraphSchema Extensions

The MoE layer template replaces the dense FFN block with a router → dispatch → experts → combine sequence.

```cpp
// Extension to LayerTemplate in GraphSchema.h

struct LayerTemplate {
    std::vector<StageSpec> attention_stages;
    std::vector<StageSpec> ffn_stages;           // Dense FFN (Qwen2, Llama)

    // MoE-specific (mutually exclusive with ffn_stages for MoE layers)
    std::optional<MoELayerTemplate> moe_template;
};

struct MoELayerTemplate {
    StageSpec router;                       // Gate GEMM → top-k selection
    StageSpec dispatch;                     // Scatter tokens to expert devices (NEW)
    StageSpec expert_ffn;                   // Template for per-expert FFN
    StageSpec combine;                      // Gather + weighted sum
    StageSpec shared_expert_ffn;            // Optional shared expert (DeepSeek-V3)

    int num_experts;
    int top_k;
    bool has_shared_expert = false;
    float router_aux_loss_coeff = 0.0f;    // For training (unused inference-side)

    // Expert weights reference pattern
    // "weights.expert_{expert_id}.gate" → expands to "blk.{layer}.ffn_gate_exps.weight"
    std::string expert_weight_pattern;
};
```

### 4.2 New StageTypes

```cpp
enum class StageType {
    // ... existing ...
    MoERouter,      // Already exists — computes gate logits + top-k
    MoEFFN,         // Already exists — single expert FFN
    MoEDispatch,    // NEW: scatter tokens to expert devices, handle H2D
    MoECombine,     // Already exists — weighted merge of expert outputs
    MoEGather,      // NEW: collect results from remote expert devices
};
```

### 4.3 New TPMode

```cpp
enum class TPMode {
    None,
    ColumnParallel,
    RowParallel,
    ExpertParallel,     // Already reserved — now used
    ExpertOffload,      // NEW: expert may live on different device/domain
};
```

### 4.4 Schema Example: DeepSeek-V3 MoE Layer

```cpp
GraphSchema DeepSeekV3Schema::createSchema() const {
    GraphSchema schema;
    schema.name = "deepseek_v3";

    // Attention stages (same as dense transformer)
    schema.layer_template.attention_stages = {
        {.name = "attn_norm", .type = StageType::RMSNorm, ...},
        {.name = "qkv_proj",  .type = StageType::FusedQKVGEMM, ...},
        // ... RoPE, KV cache, attention, Wo, residual ...
    };

    // MoE FFN replaces dense FFN
    schema.layer_template.moe_template = MoELayerTemplate{
        .router = {
            .name = "moe_router",
            .type = StageType::MoERouter,
            .inputs = {"hidden"},
            .outputs = {"router_logits", "expert_indices", "expert_weights"},
        },
        .dispatch = {
            .name = "moe_dispatch",
            .type = StageType::MoEDispatch,
            .inputs = {"hidden", "expert_indices"},
            .outputs = {"dispatched_tokens"},
            // MoEDispatch reads ExpertPlacementMap to scatter tokens
            // to the correct device for each expert
        },
        .expert_ffn = {
            .name = "expert_ffn",
            .type = StageType::MoEFFN,
            .tp_mode = TPMode::ExpertParallel,
            // Template — graph builder instantiates per expert
        },
        .combine = {
            .name = "moe_combine",
            .type = StageType::MoECombine,
            .inputs = {"expert_outputs", "expert_weights"},
            .outputs = {"moe_output"},
        },
        .num_experts = 256,
        .top_k = 8,
        .has_shared_expert = true,
        .shared_expert_ffn = {
            .name = "shared_expert",
            .type = StageType::MoEFFN,
            .inputs = {"hidden"},
            .outputs = {"shared_output"},
        },
    };

    return schema;
}
```

### 4.5 Graph Resolution: ExpertPlacementMap → ComputeGraph

The `GraphResolver` uses `ExpertPlacementMap` to decide how to instantiate the MoE layer template:

```
GraphResolver::resolveMoELayer(layer_idx, moe_template, placement_map):

  1. Emit MoERouterStage (always on primary device)

  2. Group experts by execution device:
     device_experts = {
       cuda:0 → [0, 3, 7, ...],     // hot experts on primary GPU
       cpu:0  → [1, 2, 4, 5, 6, ...], // cold experts on CPU
       rocm:0 → [8, 9, ...],         // experts on secondary GPU domain
     }

  3. For each device group:
     a. If device != primary: emit MoEDispatchStage (scatter tokens)
        - H2D transfer for CPU→GPU expert weights (via TransferEngine)
        - Cross-device token scatter for domain-remote experts
     b. Emit N parallel MoEExpertStage nodes (one per active expert)
        - Each node has device = expert's execution_device
        - Dependencies: dispatch stage (if cross-device) or router stage
     c. If device != primary: emit MoEGatherStage (collect results)

  4. Emit MoECombineStage (weighted merge on primary device)
     - Dependencies: all gather/expert stages
     - If shared_expert: also depends on shared_expert_ffn
```

**Key insight**: experts on different devices become **separate ComputeNodes with different `device` fields** in the ComputeGraph. The existing `DeviceGraphExecutor::executeMultiDevice()` already handles dispatching nodes to different devices. No new execution machinery needed.

---

## 5. Expert Offloading Strategies

### 5.1 Strategy A: CPU Offloading (Simple)

All experts start on host. Top-K experts' weights are transferred to GPU on demand.

```
[User]
./llaminar2 -m deepseek-v3.gguf --expert-offload cpu --gpu-expert-budget 16

[Runtime behavior per MoE layer]
1. Router runs on GPU → selects top-8 experts
2. ExpertWeightPrefetcher checks which of the 8 are already on GPU
   - Hit: expert weights already in GPU cache (hot path)
   - Miss: async H2D of expert weights from pinned host memory
3. Execute 8 expert FFNs on GPU
4. Evict least-recently-used experts if GPU budget exceeded
```

**Implementation**: `ExpertWeightCache` manages a fixed-size LRU cache of expert weights on GPU, backed by pinned host memory. This is analogous to `LayerWeightStreamer` but at expert granularity.

```cpp
// src/v2/execution/moe/ExpertWeightCache.h

class ExpertWeightCache {
public:
    ExpertWeightCache(DeviceId gpu_device, size_t budget_bytes,
                      EvictionPolicy policy = EvictionPolicy::LRU);

    // Ensure expert weights are on GPU (returns immediately if cached)
    // Async: begins transfer, returns future
    std::future<ExpertWeights*> prefetch(int layer_idx, int expert_id);

    // Blocking: wait for weights to be ready
    ExpertWeights* get(int layer_idx, int expert_id);

    // Stats
    float hitRate() const;
    size_t residencyCount() const;
    size_t evictionCount() const;

private:
    struct CacheEntry {
        int layer_idx, expert_id;
        std::shared_ptr<ExpertWeights> weights;  // GPU-resident copy
        uint64_t last_access_token;
        uint64_t access_count;
    };

    DeviceId gpu_device_;
    size_t budget_bytes_;
    size_t used_bytes_ = 0;
    std::unordered_map<uint64_t, CacheEntry> cache_;  // key = pack(layer, expert)
    EvictionPolicy policy_;

    void evictIfNeeded(size_t required_bytes);
};
```

### 5.2 Strategy B: Domain Offloading (Heterogeneous)

Attention runs on one domain (e.g., 3090s), expert FFNs run on another (e.g., Mi50s or multi-socket CPUs).

```
[User]
./llaminar2 -m deepseek-v3.gguf \
  --define-domain "attn=cuda:0,cuda:1;backend=nccl" \
  --define-domain "experts=rocm:0,rocm:1,rocm:2,rocm:3;backend=rccl" \
  --expert-domain experts \
  --attention-domain attn

[Runtime behavior per layer]
1. Attention runs on {cuda:0, cuda:1} with NCCL allreduce
2. Hidden state D2D transfer: cuda:0 → rocm:0 (via PCIeBAR or host staging)
3. Router runs on rocm:0 → selects top-8 experts
4. 256 experts distributed across 4 ROCm GPUs (64 each)
   - Top-8 selected experts execute in parallel on their home GPUs
   - ExpertParallel allreduce within ROCm domain (RCCL)
5. Hidden state D2D transfer back: rocm:0 → cuda:0
6. Residual add on cuda:0
```

**Implementation**: This maps directly onto the existing **Named Domains** infrastructure (`--define-domain` + `--pp-stage`), extended with expert-domain awareness:

```cpp
// In OrchestrationConfig, new fields:
struct OrchestrationConfig {
    // ... existing ...
    std::string expert_domain;      // Domain name for MoE expert execution
    std::string attention_domain;   // Domain name for attention (default: primary)
};
```

The `GraphResolver` emits cross-domain transfer stages (already supported via `TransferEngine`) between the attention and expert domains. Expert FFN stages get `device` set to devices within the expert domain, with expert-parallel distribution computed by `ExpertPlacementMap::createFromPolicy()`.

### 5.3 Strategy C: TP-Domain Expert Parallelism

Experts are sharded across devices within a TP domain, with allreduce after the combine.

```
[User]
./llaminar2 -m mixtral-8x7b.gguf \
  --tp-devices "cuda:0,cuda:1,cuda:2,cuda:3" \
  --expert-parallelism tp

[Runtime behavior]
- 8 experts distributed: 2 per GPU
- Router on all GPUs (replicated)
- Each GPU executes its local experts
- AllGather of expert outputs across TP domain
- Combine on each GPU (replicated)
```

**Implementation**: `TPMode::ExpertParallel` annotation on the MoE stages. The `GraphResolver` inserts allgather after expert execution, analogous to how `ColumnParallel` inserts allgather for LM head vocab sharding.

### 5.4 Strategy D: Adaptive Rebalancing

Combines CPU offloading with dynamic migration based on usage patterns.

```
[User]
./llaminar2 -m deepseek-v3.gguf \
  --expert-offload adaptive \
  --gpu-expert-budget 32 \
  --rebalance-interval 1024

[Runtime behavior]
1. Initially: 32 experts on GPU (chosen by first-pass profiling), 224 on CPU
2. ExpertUsageTracker records which experts are selected per token
3. Every 1024 tokens, ExpertRebalancer runs:
   - Experts with usage_rate > 15%: promote CPU → GPU (if budget allows)
   - Experts with usage_rate < 2%: demote GPU → CPU
   - ExpertPlacementMap is updated
   - WeightCache pre-fetches promoted experts
4. Graph is NOT rebuilt — MoEDispatchStage reads placement map dynamically
```

**Key design decision**: the `ExpertPlacementMap` is a **runtime-mutable** structure that `MoEDispatchStage` reads at execution time. This avoids rebuilding the compute graph on every rebalance. The graph structure (router → dispatch → experts → combine) stays fixed; only the dispatch routing table changes.

```cpp
// src/v2/execution/moe/ExpertRebalancer.h

class ExpertRebalancer {
public:
    ExpertRebalancer(ExpertPlacementMap& placement_map,
                     ExpertWeightCache& weight_cache,
                     const ExpertOffloadPolicy& policy);

    // Called periodically from the decode loop
    // Returns true if any migrations occurred
    bool maybeRebalance(const ExpertUsageTracker& tracker,
                        uint64_t current_token_position);

    struct RebalanceStats {
        int promotions = 0;       // CPU → GPU
        int demotions = 0;        // GPU → CPU
        int domain_transfers = 0; // Between devices
        float estimated_speedup;  // Based on transfer time saved
    };
    RebalanceStats lastStats() const;

private:
    ExpertPlacementMap& placement_map_;
    ExpertWeightCache& weight_cache_;
    ExpertOffloadPolicy policy_;
    uint64_t last_rebalance_token_ = 0;
};
```

---

## 6. New Compute Stages

### 6.1 MoEDispatchStage

Scatters tokens to expert devices. This is the key new stage.

```cpp
// src/v2/execution/compute_stages/stages/MoEDispatchStage.h

class MoEDispatchStage : public IComputeStage {
public:
    struct Params {
        STAGE_PARAMS_COMMON_FIELDS;

        const ITensor* hidden = nullptr;         // [seq_len, d_model]
        const ITensor* expert_indices = nullptr;  // [seq_len, top_k] — selected experts
        const ITensor* expert_weights = nullptr;  // [seq_len, top_k] — routing weights

        // Output: per-expert token buffers (scattered)
        std::vector<ITensor*> expert_inputs;      // [num_active_experts][tokens_for_expert, d_model]
        std::vector<std::vector<int>>* token_maps; // Which tokens go to which expert

        const ExpertPlacementMap* placement_map = nullptr;
        ExpertWeightCache* weight_cache = nullptr;  // For CPU-offloaded experts
        int layer_idx = 0;
        int num_experts = 0;
        int top_k = 0;
    };

    bool execute(IDeviceContext* ctx) override;

    // Execute flow:
    // 1. Read router decisions (expert_indices)
    // 2. For each selected expert:
    //    a. Look up placement in ExpertPlacementMap
    //    b. If expert is GPU-resident: direct scatter
    //    c. If expert is CPU-resident: trigger weight prefetch via ExpertWeightCache
    //    d. If expert is on remote domain device: scatter tokens via TransferEngine
    // 3. Build token_maps for MoECombineStage
};
```

### 6.2 MoEGatherStage

Collects expert outputs from remote devices back to the primary device.

```cpp
class MoEGatherStage : public IComputeStage {
    // Symmetric to MoEDispatchStage
    // Collects expert FFN outputs from remote devices
    // Uses TransferEngine for D2D / D2H transfers
};
```

---

## 7. CLI Integration

### 7.1 New CLI Flags

```
Expert Offloading:
  --expert-offload <mode>        Expert offload mode: none, cpu, domain, adaptive
  --gpu-expert-budget <n>        Max experts resident on GPU (-1 = auto)
  --expert-domain <name>         Named domain for expert execution
  --attention-domain <name>      Named domain for attention (default: primary)
  --expert-parallelism <mode>    Expert distribution: none, tp, ep (expert-parallel)
  --pin-expert-memory            Pin CPU expert weights for faster H2D (default: on)

Adaptive Rebalancing:
  --rebalance-interval <n>       Re-evaluate placement every N tokens (default: 1024)
  --promotion-threshold <f>      Promote to GPU if usage > threshold (default: 0.15)
  --demotion-threshold <f>       Demote to CPU if usage < threshold (default: 0.02)
  --expert-cache-policy <p>      Eviction policy: lru, lfu, frequency (default: lru)

Introspection:
  --explain-expert-placement     Show expert placement decisions and exit
  --expert-usage-log             Log per-layer expert usage histograms
```

### 7.2 YAML Configuration

```yaml
# orchestration.yaml
expert_offload:
  mode: adaptive
  gpu_expert_budget: 32
  pin_memory: true
  rebalance:
    interval_tokens: 1024
    promotion_threshold: 0.15
    demotion_threshold: 0.02
    cache_policy: lru
  domains:
    attention: "attn_gpus"
    experts: "expert_pool"
```

---

## 8. Integration with Existing Systems

### 8.1 Weight Loading (WeightManager)

MoE expert weights follow a predictable naming pattern:

```
blk.{layer}.ffn_gate_exps.weight   → [num_experts, d_ff, d_model]
blk.{layer}.ffn_up_exps.weight     → [num_experts, d_ff, d_model]
blk.{layer}.ffn_down_exps.weight   → [num_experts, d_model, d_ff]
```

`WeightManager` already supports per-weight sharding. For MoE, add `ExpertWeight` shard mode that slices along the expert dimension:

```cpp
enum class WeightShardingMode {
    Replicate,
    ColumnParallel,
    RowParallel,
    InputParallel,
    ExpertParallel,   // NEW: shard along expert dimension
};
```

`ExpertParallel` sharding distributes experts evenly (or proportionally with `--tp-weights`) across devices in the expert domain.

### 8.2 BufferArena Extensions

MoE layers need per-expert activation buffers. Rather than statically registering `N_experts` buffers, use a **dynamic scratch pool**:

```cpp
// New BufferIds for MoE
enum class BufferId : uint32_t {
    // ... existing ...
    MOE_ROUTER_LOGITS,      // [seq_len, num_experts]
    MOE_EXPERT_INDICES,     // [seq_len, top_k]
    MOE_EXPERT_WEIGHTS,     // [seq_len, top_k]
    MOE_DISPATCH_SCRATCH,   // Reusable scratch for token scatter
    MOE_GATHER_SCRATCH,     // Reusable scratch for result gather
    MOE_OUTPUT,             // [seq_len, d_model] — combined expert output
    MOE_SHARED_OUTPUT,      // [seq_len, d_model] — shared expert output (DeepSeek)
};
```

Expert-internal activations (gate, up, down) reuse the existing `GATE_PROJ`, `UP_PROJ`, `FFN_OUTPUT` buffers since experts execute sequentially (or in a bounded parallel pool).

### 8.3 TransferEngine

Already supports all needed transfer methods:
- `HOST_TO_DEVICE` — CPU expert weights → GPU
- `DEVICE_TO_DEVICE_SAME_BACKEND` — GPU↔GPU within NCCL/RCCL domain
- `BAR_HOST_BOUNCE` — cross-vendor GPU↔GPU (CUDA↔ROCm)

No changes needed to `TransferEngine` itself.

### 8.4 CollectiveContext

Expert-parallel allreduce/allgather after expert execution uses the same `ICollectiveBackend` infrastructure. The expert domain gets its own `ILocalTPContext` with the appropriate backend (NCCL, RCCL, or PCIeBAR).

### 8.5 Profiling Integration

```bash
# Per-expert timing
LLAMINAR_PROFILING=1 ./llaminar2 --expert-offload adaptive -m model.gguf

# Expert usage histograms
LLAMINAR_EXPERT_USAGE_LOG=1 ./llaminar2 -m model.gguf -n 100

# Stage dump for MoE debugging
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_TYPES=MOE_ROUTER,MOE_DISPATCH,MOE_EXPERT_FFN,MOE_COMBINE \
./llaminar2 -m model.gguf
```

---

## 9. Execution Flows

### 9.1 Decode Step (CPU Offloading, Adaptive)

```
decode_step(token):
  ... attention layers (unchanged) ...

  // MoE FFN layer
  MoERouterStage:
    gate_logits = hidden @ gate_weights    // [1, num_experts]
    top_k_experts, top_k_weights = top_k(softmax(gate_logits), k=8)
    usage_tracker.recordRouting(layer_idx, top_k_experts, 1)

  MoEDispatchStage:
    for expert_id in top_k_experts:
      placement = placement_map.getPlacement(layer_idx, expert_id)
      if placement.residency == GPU_HOT:
        expert_inputs[expert_id] = hidden  // zero-copy
      elif placement.residency == CPU_WARM:
        weight_cache.prefetch(layer_idx, expert_id)  // async H2D
        expert_inputs[expert_id] = hidden  // token stays on GPU

  MoEExpertStage × top_k:  (sequential or parallel on GPU)
    for expert_id in top_k_experts:
      weights = weight_cache.get(layer_idx, expert_id)  // blocks on prefetch
      expert_output = FFN(expert_inputs[expert_id], weights)

  MoECombineStage:
    output = sum(expert_weights[i] * expert_outputs[i] for i in top_k)
    hidden = hidden + output  // residual

  // Periodic rebalance check
  if token_position % rebalance_interval == 0:
    rebalancer.maybeRebalance(usage_tracker, token_position)
```

### 9.2 Decode Step (Cross-Domain)

```
decode_step(token):
  // Attention on domain "attn" (cuda:0, cuda:1)
  ... attention with NCCL allreduce ...

  // Transfer hidden to expert domain
  TransferEngine::execute(DEVICE_TO_DEVICE, hidden, cuda:0 → rocm:0)

  // MoE on domain "experts" (rocm:0, rocm:1, rocm:2, rocm:3)
  MoERouterStage on rocm:0
  MoEDispatchStage:
    scatter tokens to expert home devices within ROCm domain
  MoEExpertStage × top_k:
    parallel execution across ROCm GPUs
  MoEGatherStage:
    allgather results back to rocm:0
  MoECombineStage on rocm:0

  // Transfer hidden back to attention domain
  TransferEngine::execute(DEVICE_TO_DEVICE, hidden, rocm:0 → cuda:0)
  ResidualAdd on cuda:0
```

---

## 10. Implementation Phases

### Phase 1: Basic MoE Support (Dense Graph)
- [ ] Implement `MoEDispatchStage` and `MoEGatherStage`
- [ ] Add `MoELayerTemplate` to `GraphSchema`
- [ ] Implement `DeepSeekV3Schema` / `MixtralSchema` (or generic `MoESchema`)
- [ ] All experts on primary GPU, sequential execution
- [ ] Unit tests for router, dispatch, expert FFN, combine

### Phase 2: CPU Offloading
- [ ] Implement `ExpertPlacementMap` and `ExpertWeightCache`
- [ ] `ExpertParallel` sharding mode in `WeightManager`
- [ ] `--expert-offload cpu` CLI flag
- [ ] Pinned memory for CPU expert weights
- [ ] Async H2D prefetch pipeline
- [ ] Integration test: DeepSeek-V3 Q4_0 with 32 GPU + 224 CPU experts

### Phase 3: Domain Offloading
- [ ] `--expert-domain` / `--attention-domain` CLI flags
- [ ] Cross-domain transfer stages in graph resolution
- [ ] Expert distribution across domain devices
- [ ] Test: attention on CUDA, experts on ROCm via PCIeBAR

### Phase 4: Adaptive Rebalancing
- [ ] `ExpertUsageTracker` with sliding window histograms
- [ ] `ExpertRebalancer` with promotion/demotion logic
- [ ] `--expert-offload adaptive` CLI flag
- [ ] Weight migration (CPU↔GPU) without graph rebuild
- [ ] Rebalance logging and `--explain-expert-placement`

### Phase 5: Performance Optimization
- [ ] Expert batching (group tokens for same expert in one GEMM)
- [ ] Pipelined expert execution (overlap H2D with compute)
- [ ] CUDA graph capture for hot expert paths
- [ ] Expert-parallel NCCL allgather for TP domains

---

## 11. Why This Fits Naturally

| Existing Concept | MoE Extension |
|---|---|
| `ComputeNode.device` | Each expert node gets its own `device` field |
| `TPMode::ExpertParallel` | Already reserved in `GraphSchema.h` |
| `MoERouterStage`, `MoEExpertStage`, `MoECombineStage` | Already implemented in `MoEStages.h` |
| `DomainDefinition` + `--define-domain` | Expert domain is just another named domain |
| `TransferEngine` | H2D/D2D expert weight and activation transfers |
| `LayerWeightStreamer` | `ExpertWeightCache` follows same LRU pattern |
| `ExpertPlacementMap` | Analogous to `WeightPlacementMap` for TP |
| `GraphResolver` | Reads `ExpertPlacementMap` to set per-node devices |
| `DeviceGraphExecutor::executeMultiDevice()` | Handles mixed-device graphs already |
| `BufferArena` aliasing | Expert activations alias with dense FFN buffers |

The entire design avoids new execution primitives. Every MoE-specific behavior reduces to: *"set the `device` field on existing `ComputeNode`s based on a placement table, and let the existing multi-device executor handle the rest."*

---

## 12. Open Questions

1. **Expert batching during prefill**: With seq_len=512, many tokens may route to the same expert. Should we batch them into a single large GEMM? (Likely yes — significant throughput win.)

2. **Shared expert interaction with TP**: DeepSeek-V3's shared expert runs on every token. If expert domain differs from attention domain, shared expert needs its own placement decision.

3. **Router on which device?**: If attention is on domain A and experts on domain B, the router GEMM is cheap but its output (expert indices) drives the dispatch. Running on expert domain avoids one transfer.

4. **Graph rebuild vs dynamic dispatch**: Current design uses dynamic dispatch (placement map read at runtime). Alternative: rebuild graph on rebalance. Dynamic is simpler and avoids cache invalidation, but prevents CUDA graph capture of the MoE section.

5. **Multi-rank MoE**: For GLOBAL TP with MoE, experts could be distributed across MPI ranks. This requires MPI-based expert dispatch (send tokens to the rank owning the expert). Not addressed in this sketch — focus is single-rank / local first.
