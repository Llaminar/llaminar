# Unified Multi-Device Orchestration Architecture

## Phase 3: Unifying TP and PP in MultiDeviceOrchestrator

**Date**: February 2026  
**Status**: Design Document  
**Author**: Design Collaboration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Analysis](#2-problem-analysis)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Interface Design](#4-interface-design)
5. [Forward Pass Flow](#5-forward-pass-flow)
6. [Configuration Design](#6-configuration-design)
7. [KV Cache Strategy](#7-kv-cache-strategy)
8. [Weight Distribution Strategy](#8-weight-distribution-strategy)
9. [Graph Building Changes](#9-graph-building-changes)
10. [Activation Transfer Mechanism](#10-activation-transfer-mechanism)
11. [Test Infrastructure Changes](#11-test-infrastructure-changes)
12. [Migration Path](#12-migration-path)
13. [Code Sketches](#13-code-sketches)

---

## 1. Executive Summary

### Current State

- **LOCAL TP** (working): `MultiDeviceOrchestrator` manages N `DeviceGraphOrchestrator` instances. Each has its own `InferenceState` with sharded KV cache. Forward runs all in parallel via `std::async`, followed by allgather for logits.

- **LOCAL PP** (broken): Uses single `DeviceGraphOrchestrator` with a unified graph spanning all devices. This causes:
  - KV cache device mismatch (crash when KV cache on device A but layer runs on device B)
  - Buffer allocation confusion
  - No clear ownership of per-device state

### Key Insight

**PP should use the same multi-orchestrator pattern as TP**, but with:
- **Vertical sharding** (different layers per device) instead of horizontal (different heads)
- **Sequential forward execution** with explicit activation transfer instead of parallel with allreduce

### Proposed Solution

Extend `MultiDeviceOrchestrator` to handle both TP and PP through a **unified execution model**:
- Each device gets its own `DeviceGraphOrchestrator` (whether for TP or PP)
- Coordination logic differs: parallel + allreduce (TP) vs sequential + transfer (PP)
- Hybrid TP+PP composes: each PP stage IS a TP domain with its own sub-orchestrator

---

## 2. Problem Analysis

### 2.1 Why Current PP Architecture Fails

```
CURRENT (BROKEN) PP ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│          Single DeviceGraphOrchestrator                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Unified ComputeGraph (spans devices)                        │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │ │
│  │  │Layer 0-5│────│Transfer │────│Layer 6-11│                │ │
│  │  │ cuda:0  │    │  Stage  │    │  cuda:1  │                │ │
│  │  └─────────┘    └─────────┘    └─────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Single InferenceState                                       │ │
│  │  • hidden: device ???                                       │ │
│  │  • KV cache: device ??? (ONE cache for ALL layers!)        │ │ ← BUG!
│  │  • activations: device ???                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Problems:
1. KV cache lives on ONE device but layers run on DIFFERENT devices
2. KV cache.update() called from cuda:1 stage but cache is on cuda:0 → crash
3. Buffer ownership unclear (who owns hidden buffer across stages?)
4. No natural place for per-device state
```

### 2.2 Why TP Architecture Works

```
CURRENT (WORKING) TP ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│               MultiDeviceOrchestrator                            │
│  ┌──────────────────┐        ┌──────────────────┐               │
│  │DeviceGraphOrch #0│        │DeviceGraphOrch #1│               │
│  │ (cuda:0)         │        │ (cuda:1)         │               │
│  │                  │        │                  │               │
│  │ InferenceState:  │        │ InferenceState:  │               │
│  │ • KV cache       │        │ • KV cache       │               │
│  │   (heads 0-7)    │        │   (heads 8-15)   │               │
│  │ • hidden         │        │ • hidden         │               │
│  │ • Q,K,V (local)  │        │ • Q,K,V (local)  │               │
│  └────────┬─────────┘        └────────┬─────────┘               │
│           │                           │                          │
│           └──────────┬────────────────┘                          │
│                      ▼                                           │
│           ┌──────────────────┐                                   │
│           │  forward() flow  │                                   │
│           │  1. parallel exec│                                   │
│           │  2. allreduce    │                                   │
│           │  3. gather logits│                                   │
│           └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘

Why it works:
1. Each device has its OWN InferenceState with its OWN KV cache
2. Each device runs SAME layers with DIFFERENT heads (horizontal shard)
3. Clear ownership: device 0 owns its buffers, device 1 owns its buffers
4. Coordination happens in MultiDeviceOrchestrator, not in graph
```

### 2.3 Applying TP Pattern to PP

```
PROPOSED PP ARCHITECTURE (same pattern as TP):
┌─────────────────────────────────────────────────────────────────┐
│               MultiDeviceOrchestrator                            │
│  ┌──────────────────┐        ┌──────────────────┐               │
│  │DeviceGraphOrch #0│        │DeviceGraphOrch #1│               │
│  │ (cuda:0, stage 0)│        │ (cuda:1, stage 1)│               │
│  │                  │        │                  │               │
│  │ InferenceState:  │        │ InferenceState:  │               │
│  │ • KV cache       │        │ • KV cache       │               │
│  │   (layers 0-11)  │        │   (layers 12-23) │               │
│  │ • hidden         │        │ • hidden         │               │
│  │ • Q,K,V (full)   │        │ • Q,K,V (full)   │               │
│  └────────┬─────────┘        └────────┬─────────┘               │
│           │                           │                          │
│           └──────────┬────────────────┘                          │
│                      ▼                                           │
│           ┌──────────────────┐                                   │
│           │  forward() flow  │                                   │
│           │  1. stage 0 exec │                                   │
│           │  2. transfer act │                                   │
│           │  3. stage 1 exec │                                   │
│           └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘

Key differences from TP:
- Layers are DIFFERENT per device (vertical shard) instead of same
- Execution is SEQUENTIAL instead of parallel
- Coordination is TRANSFER instead of allreduce
- KV cache has DIFFERENT layers instead of different heads
```

---

## 3. Proposed Architecture

### 3.1 Should MultiDeviceOrchestrator Handle Both?

**Answer: YES**, with a clear internal strategy pattern.

The fundamental pattern is identical:
- N devices → N `DeviceGraphOrchestrator` instances
- Each instance has its own `InferenceState` (including KV cache)
- Coordination happens in the orchestrator, not in the graph

What differs is **coordination strategy**:

| Aspect | TP Mode | PP Mode | TP+PP Hybrid |
|--------|---------|---------|--------------|
| Layer assignment | Same layers, all devices | Different layers per device | Per-PP-stage layers |
| Head assignment | Sharded by device | Full heads | Sharded within TP domain |
| Forward execution | Parallel (`std::async`) | Sequential | Sequential PP, parallel within stage |
| Coordination op | AllReduce (row-parallel) | Transfer (activation) | Transfer between stages, allreduce within |
| KV cache | Sharded by heads | Sharded by layers | Both (layers per stage, heads within stage) |

### 3.2 Class Hierarchy

```
┌───────────────────────────────────────────────────────────────────┐
│                    IInferenceRunner                                │
│  (forward, logits, clear_cache, get_position, vocab_size, ...)    │
└───────────────────────────────────────────────────────────────────┘
                               ▲
                               │
         ┌─────────────────────┴─────────────────────┐
         │                                           │
┌────────────────────────┐              ┌────────────────────────┐
│IMultiDeviceOrchestrator│              │DeviceGraphOrchestrator │
│  device_count()        │              │  (single device)        │
│  deviceRunner(idx)     │              │                         │
│  localTPContext()      │              │                         │
│  allDevicesReady()     │              │                         │
└────────────────────────┘              └────────────────────────┘
            ▲
            │
┌────────────────────────────────────────────────────────────────────┐
│              MultiDeviceOrchestrator                                │
│                                                                     │
│  Config:                                                            │
│  - ParallelismMode: TP_ONLY, PP_ONLY, TP_PLUS_PP                   │
│  - ExecutionStrategy (interface) for coordination                  │
│                                                                     │
│  Members:                                                           │
│  - device_runners_: vector<unique_ptr<DeviceGraphOrchestrator>>    │
│  - tp_ctx_: unique_ptr<ILocalTPContext> (for TP coordination)      │
│  - pp_ctx_: unique_ptr<ILocalPPContext> (for PP coordination)      │
│  - strategy_: unique_ptr<IExecutionStrategy>                       │
│                                                                     │
│  Derived state per mode:                                           │
│  - TP: combined_logits_ (AllGather result)                         │
│  - PP: transfer_buffer_ (activation staging)                       │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 Execution Strategy Interface

```cpp
/**
 * @brief Strategy interface for multi-device execution coordination
 *
 * Encapsulates the coordination logic that differs between TP and PP:
 * - TP: parallel execution + allreduce
 * - PP: sequential execution + transfer
 * - TP+PP: hybrid (sequential stages, parallel within stage)
 */
class IExecutionStrategy {
public:
    virtual ~IExecutionStrategy() = default;
    
    /**
     * @brief Execute forward pass across all devices
     *
     * @param runners Device runners to coordinate
     * @param tokens Input tokens
     * @param seq_len Sequence length
     * @return true on success
     */
    virtual bool executeForward(
        std::vector<DeviceGraphOrchestrator*>& runners,
        const int* tokens,
        int seq_len) = 0;
    
    /**
     * @brief Gather final logits after forward pass
     *
     * @param runners Device runners
     * @param output_buffer Pre-allocated output buffer
     * @param seq_len Sequence length
     * @return true on success
     */
    virtual bool gatherLogits(
        std::vector<DeviceGraphOrchestrator*>& runners,
        TensorBase* output_buffer,
        size_t seq_len) = 0;
    
    /**
     * @brief Clear caches on all devices
     */
    virtual void clearCaches(
        std::vector<DeviceGraphOrchestrator*>& runners) = 0;
    
    /**
     * @brief Get parallelism mode
     */
    virtual ParallelismMode mode() const = 0;
};
```

### 3.4 Concrete Strategy Implementations

```cpp
/**
 * @brief TP execution: parallel forward + allreduce + gather
 */
class TPExecutionStrategy : public IExecutionStrategy {
public:
    TPExecutionStrategy(ILocalTPContext* tp_ctx);
    
    bool executeForward(...) override {
        // 1. Launch all device runners in parallel via std::async
        // 2. Wait for all to complete
        // 3. (AllReduce happens inside each runner's graph for row-parallel ops)
        // 4. Return combined success
    }
    
    bool gatherLogits(...) override {
        // AllGather partial logits from column-parallel LM head
    }
    
    ParallelismMode mode() const override { return ParallelismMode::TP_ONLY; }

private:
    ILocalTPContext* tp_ctx_;
};

/**
 * @brief PP execution: sequential forward + transfer
 */
class PPExecutionStrategy : public IExecutionStrategy {
public:
    PPExecutionStrategy(ILocalPPContext* pp_ctx);
    
    bool executeForward(...) override {
        // For each stage in order:
        //   1. Execute stage's device runner
        //   2. Transfer activations to next stage (if not last)
    }
    
    bool gatherLogits(...) override {
        // Logits come from last stage only - direct copy
    }
    
    ParallelismMode mode() const override { return ParallelismMode::PP_ONLY; }

private:
    ILocalPPContext* pp_ctx_;
};

/**
 * @brief TP+PP execution: sequential stages, parallel within stage
 */
class TPPPExecutionStrategy : public IExecutionStrategy {
public:
    TPPPExecutionStrategy(
        std::vector<std::unique_ptr<ILocalTPContext>> stage_tp_contexts,
        ILocalPPContext* pp_ctx);
    
    bool executeForward(...) override {
        // For each PP stage in order:
        //   1. Execute all TP devices for this stage in parallel
        //   2. (AllReduce happens inside each graph for row-parallel ops)
        //   3. Transfer activations to next stage's TP domain
    }
    
    bool gatherLogits(...) override {
        // AllGather from last stage's TP devices
    }
    
    ParallelismMode mode() const override { return ParallelismMode::TP_PLUS_PP; }

private:
    std::vector<std::unique_ptr<ILocalTPContext>> stage_tp_contexts_; // One per PP stage
    ILocalPPContext* pp_ctx_;
};
```

---

## 4. Interface Design

### 4.1 New Interfaces

```cpp
// ============================================================================
// Parallelism Mode Enum
// ============================================================================

enum class ParallelismMode {
    SINGLE_DEVICE,   // One device, no parallelism
    TP_ONLY,         // Tensor parallelism only (horizontal sharding)
    PP_ONLY,         // Pipeline parallelism only (vertical sharding)
    TP_PLUS_PP,      // Both TP and PP (hybrid)
};

// ============================================================================
// Device Assignment for PP
// ============================================================================

/**
 * @brief Assignment of layers to a device (PP mode)
 */
struct PPDeviceAssignment {
    int device_index;        // Index in MultiDeviceOrchestrator's device list
    DeviceId device_id;      // The device
    int first_layer;         // First layer (inclusive)
    int last_layer;          // Last layer (exclusive)
    bool has_embedding;      // Does this device handle embedding?
    bool has_lm_head;        // Does this device handle LM head?
    
    int layerCount() const { return last_layer - first_layer; }
};

// ============================================================================
// Extended IMultiDeviceOrchestrator
// ============================================================================

class IMultiDeviceOrchestrator : public IInferenceRunner {
public:
    // Existing methods
    virtual int device_count() const = 0;
    virtual IInferenceRunner* deviceRunner(int device_idx) = 0;
    virtual ILocalTPContext* localTPContext() = 0;
    virtual bool allDevicesReady() const = 0;
    virtual void synchronizeDevices() = 0;
    
    // New methods for unified TP/PP
    virtual ParallelismMode parallelismMode() const = 0;
    virtual ILocalPPContext* localPPContext() = 0;
    virtual const PPDeviceAssignment* ppAssignment(int device_idx) const = 0;
};
```

### 4.2 Extended DeviceGraphOrchestrator

```cpp
/**
 * @brief Extended configuration for DeviceGraphOrchestrator
 *
 * Now supports both TP and PP layer range specification.
 */
struct DeviceGraphConfig {
    // Existing TP fields
    ILocalTPContext* local_tp_ctx = nullptr;
    int local_tp_device_index = 0;
    
    // New PP fields
    std::optional<int> pp_first_layer;    // First layer this device handles
    std::optional<int> pp_last_layer;     // Last layer (exclusive)
    bool pp_has_embedding = true;          // Handle embedding lookup?
    bool pp_has_lm_head = true;            // Handle LM head projection?
    
    // Inferred
    bool isPP() const { 
        return pp_first_layer.has_value() && pp_last_layer.has_value(); 
    }
    int ppLayerCount() const {
        return isPP() ? (*pp_last_layer - *pp_first_layer) : -1;
    }
};
```

---

## 5. Forward Pass Flow

### 5.1 Pure LOCAL TP (Reference - Already Working)

```
forward(tokens, seq_len):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. PARALLEL EXECUTION                                            │
    │    ┌──────────────────┐  ┌──────────────────┐                   │
    │    │ Device 0         │  │ Device 1         │                   │
    │    │ embedding        │  │ embedding        │  (replicated)     │
    │    │ for layer 0..N:  │  │ for layer 0..N:  │                   │
    │    │   attn (heads 0-7)  │   attn (heads 8-15)  (sharded)       │
    │    │   allreduce(Wo)  │  │   allreduce(Wo)  │  (sync point)     │
    │    │   FFN (cols 0-2k)│  │   FFN (cols 2k-4k)│ (sharded)        │
    │    │   allreduce(down)│  │   allreduce(down)│  (sync point)     │
    │    │ lm_head (cols 0-V/2) │ lm_head (cols V/2-V) (sharded)      │
    │    └──────────────────┘  └──────────────────┘                   │
    │                                                                  │
    │ 2. ALLGATHER LOGITS                                             │
    │    [0:V/2] from dev0, [V/2:V] from dev1 → [0:V] combined        │
    └─────────────────────────────────────────────────────────────────┘
```

### 5.2 Pure LOCAL PP (Proposed)

```
forward(tokens, seq_len):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. STAGE 0 EXECUTION (cuda:0, layers 0-11)                      │
    │    │ embedding(tokens) → hidden                                 │
    │    │ for layer in 0..11:                                        │
    │    │   attn(hidden, kv_cache[layer]) → hidden                  │
    │    │   FFN(hidden) → hidden                                     │
    │    │ output: hidden [seq_len, d_model]                         │
    │    └──────────────────────────────────────────────────────────  │
    │                                                                  │
    │ 2. ACTIVATION TRANSFER (cuda:0 → cuda:1)                        │
    │    pp_ctx->transfer(hidden, stage_from=0, stage_to=1)          │
    │    └──────────────────────────────────────────────────────────  │
    │                                                                  │
    │ 3. STAGE 1 EXECUTION (cuda:1, layers 12-23)                     │
    │    │ (no embedding - receives hidden from stage 0)              │
    │    │ for layer in 12..23:                                       │
    │    │   attn(hidden, kv_cache[layer]) → hidden                  │
    │    │   FFN(hidden) → hidden                                     │
    │    │ lm_head(hidden) → logits                                  │
    │    └──────────────────────────────────────────────────────────  │
    │                                                                  │
    │ 4. RETURN LOGITS (from last stage only)                         │
    └─────────────────────────────────────────────────────────────────┘
```

### 5.3 LOCAL TP + PP Hybrid (Proposed)

```
Example: PP(TP(cuda:0,cuda:1), TP(rocm:0,rocm:1))
         Stage 0: layers 0-11 on 2-way CUDA TP
         Stage 1: layers 12-23 on 2-way ROCm TP

forward(tokens, seq_len):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. STAGE 0 EXECUTION (TP domain: cuda:0, cuda:1)                │
    │    ┌──────────────────┐  ┌──────────────────┐                   │
    │    │ cuda:0           │  │ cuda:1           │  PARALLEL         │
    │    │ embedding        │  │ embedding        │                   │
    │    │ layers 0-11      │  │ layers 0-11      │                   │
    │    │ (heads 0-7)      │  │ (heads 8-15)     │                   │
    │    │ allreduce at Wo  │  │ allreduce at Wo  │                   │
    │    └────────┬─────────┘  └────────┬─────────┘                   │
    │             └──────────┬──────────┘                              │
    │                        ▼                                         │
    │    After allreduce: cuda:0 and cuda:1 have identical hidden     │
    │                                                                  │
    │ 2. ACTIVATION TRANSFER (TP domain 0 → TP domain 1)              │
    │    • Pick representative device from stage 0 (cuda:0)           │
    │    • Transfer hidden to representative device of stage 1 (rocm:0)│
    │    • Broadcast within stage 1's TP domain if needed             │
    │    (Note: for row-parallel, all TP devices need same activation)│
    │                                                                  │
    │ 3. STAGE 1 EXECUTION (TP domain: rocm:0, rocm:1)                │
    │    ┌──────────────────┐  ┌──────────────────┐                   │
    │    │ rocm:0           │  │ rocm:1           │  PARALLEL         │
    │    │ layers 12-23     │  │ layers 12-23     │                   │
    │    │ (heads 0-7)      │  │ (heads 8-15)     │                   │
    │    │ allreduce at Wo  │  │ allreduce at Wo  │                   │
    │    │ lm_head (col 0-V/2)│ │ lm_head (col V/2-V)│                │
    │    └────────┬─────────┘  └────────┬─────────┘                   │
    │             └──────────┬──────────┘                              │
    │                        ▼                                         │
    │ 4. ALLGATHER LOGITS (within stage 1's TP domain)                │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Design

### 6.1 Extended MultiDeviceOrchestrator::Config

```cpp
struct MultiDeviceOrchestrator::Config {
    // =========================================================================
    // Parallelism Mode (inferred or explicit)
    // =========================================================================
    
    /**
     * @brief Parallelism mode (AUTO = infer from config)
     *
     * AUTO inference:
     * - If pp_config present with >1 stage → PP or TP+PP
     * - Else if devices.size() > 1 → TP_ONLY
     * - Else → SINGLE_DEVICE
     */
    ParallelismMode mode = ParallelismMode::AUTO;
    
    // =========================================================================
    // Common Fields (all modes)
    // =========================================================================
    
    size_t max_seq_len = 4096;
    int batch_size = 1;
    ActivationPrecision activation_precision = ActivationPrecision::FP32;
    float kv_cache_scale = 1.0f;
    bool use_mapped_memory = false;
    
    // =========================================================================
    // TP Configuration (TP_ONLY or TP+PP modes)
    // =========================================================================
    
    /// Devices for TP (when mode is TP_ONLY)
    std::vector<GlobalDeviceAddress> devices;
    
    /// Proportional weights for TP work distribution
    std::vector<float> weights;
    
    /// Backend for TP collective operations
    CollectiveBackendType tp_backend = CollectiveBackendType::AUTO;
    
    // =========================================================================
    // PP Configuration (PP_ONLY or TP+PP modes)
    // =========================================================================
    
    /// Pipeline configuration (when mode involves PP)
    /// If set, overrides simple TP-only mode
    std::shared_ptr<PipelineConfig> pipeline_config;
    
    // =========================================================================
    // Validation and Inference
    // =========================================================================
    
    /**
     * @brief Validate and infer mode
     */
    bool validate() const;
    
    /**
     * @brief Get inferred parallelism mode
     */
    ParallelismMode inferMode() const {
        if (mode != ParallelismMode::AUTO) return mode;
        
        if (pipeline_config && pipeline_config->hasPP()) {
            return pipeline_config->hasTP() 
                ? ParallelismMode::TP_PLUS_PP 
                : ParallelismMode::PP_ONLY;
        }
        
        if (devices.size() > 1) {
            return ParallelismMode::TP_ONLY;
        }
        
        return ParallelismMode::SINGLE_DEVICE;
    }
};
```

### 6.2 Unified Configuration via PipelineConfig

The existing `PipelineConfig` structure already supports PP+TP composition:

```cpp
// From PipelineConfig.h - already supports all modes

// Single device (no parallelism)
auto config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));

// TP only (existing TP pattern)
auto config = PipelineConfig::tensorParallel(24, 
    {DeviceId::cuda(0), DeviceId::cuda(1)}, 
    CollectiveBackendType::NCCL);

// PP only (new pattern)
auto config = PipelineConfig::pipelineParallel2Stage(24,
    DeviceId::cuda(0), 12,  // Stage 0: layers 0-11
    DeviceId::cuda(1),      // Stage 1: layers 12-23
    CollectiveBackendType::PCIE_BAR);

// PP + TP (hybrid)
PipelineConfig config;
config.total_layers = 24;
config.tp_domains = {
    {"stage0_tp", {DeviceId::cuda(0), DeviceId::cuda(1)}, NCCL},
    {"stage1_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, RCCL},
};
config.pp_stages = {
    PPStageConfig::firstStage(0, "stage0_tp", 0, 12),   // Has embedding
    PPStageConfig::lastStage(1, "stage1_tp", 12, 24),   // Has LM head
};
config.pp_transfer_backends[{0, 1}] = CollectiveBackendType::PCIE_BAR;
```

---

## 7. KV Cache Strategy

### 7.1 Per-Mode KV Cache Organization

| Mode | KV Cache Organization | Example (24 layers, 16 heads) |
|------|----------------------|------------------------------|
| **TP_ONLY** | Each device: all layers, subset of heads | Dev0: L0-23, H0-7; Dev1: L0-23, H8-15 |
| **PP_ONLY** | Each device: subset of layers, all heads | Dev0: L0-11, H0-15; Dev1: L12-23, H0-15 |
| **TP+PP** | Each device: subset of layers, subset of heads | Stage0-Dev0: L0-11, H0-7; Stage0-Dev1: L0-11, H8-15; etc. |

### 7.2 Implementation in InferenceState

```cpp
// Current InferenceState (works for TP, needs extension for PP)
struct InferenceState {
    // Single KV cache - works for TP (same layers, different heads)
    std::unique_ptr<IKVCache> kv_cache;
    
    // PP extension: per-device KV caches when PP is enabled
    // Key: local layer index (0 to pp_layer_count-1)
    // Only used when DeviceGraphConfig::isPP() == true
    std::unordered_map<DeviceId, std::unique_ptr<IKVCache>> pp_kv_caches;
    
    // ... other fields ...
};
```

### 7.3 KV Cache Factory Changes

```cpp
/**
 * @brief Create KV cache for a device based on parallelism mode
 *
 * @param config KV cache config
 * @param tp_config TP configuration (for head sharding)
 * @param pp_assignment PP assignment (for layer range) 
 * @param device Target device
 */
std::unique_ptr<IKVCache> createKVCacheForDevice(
    const KVCacheConfig& config,
    const TensorParallelConfig* tp_config,
    const PPDeviceAssignment* pp_assignment,
    DeviceId device)
{
    int n_layers, n_kv_heads;
    
    // Determine layer count
    if (pp_assignment) {
        // PP mode: only layers for this stage
        n_layers = pp_assignment->layerCount();
    } else {
        // TP-only or single: all layers
        n_layers = config.n_layers;
    }
    
    // Determine head count
    if (tp_config && tp_config->hasAssignment(device)) {
        // TP mode: sharded heads
        n_kv_heads = tp_config->forDevice(device).kv_head_count;
    } else {
        // PP-only or single: all heads
        n_kv_heads = config.n_kv_heads;
    }
    
    return KernelFactory::createKVCache({
        .n_layers = n_layers,
        .n_kv_heads = n_kv_heads,
        .head_dim = config.head_dim,
        .max_seq_len = config.max_seq_len,
        .dtype = config.dtype,
        .device = device,
    });
}
```

---

## 8. Weight Distribution Strategy

### 8.1 Weight Sharding Modes by Parallelism

| Weight | TP Sharding | PP Handling |
|--------|-------------|-------------|
| `token_embd` | REPLICATED | Only on embedding stage |
| `attn_q/k/v` | COLUMN (by heads) | Only layers for this stage |
| `attn_o` (Wo) | ROW (row-parallel) | Only layers for this stage |
| `ffn_gate/up` | COLUMN | Only layers for this stage |
| `ffn_down` | ROW (input-parallel) | Only layers for this stage |
| `output` (LM head) | COLUMN (by vocab) | Only on LM head stage |
| `*_norm` | REPLICATED | Only layers for this stage |

### 8.2 Extended WeightManager API

```cpp
class WeightManager {
public:
    // Existing TP method
    void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> tp_config);
    
    // New PP method
    void setPipelineConfig(std::shared_ptr<PipelineConfig> pp_config);
    
    /**
     * @brief Get weight tensor for a specific device
     *
     * Applies both TP sharding and PP layer filtering.
     *
     * @param weight_name Full weight name (e.g., "blk.5.attn_q.weight")
     * @param device Target device
     * @return Sharded/filtered tensor, or nullptr if weight not on this device
     */
    ITensor* getWeightForDevice(const std::string& weight_name, DeviceId device);
    
    /**
     * @brief Check if a layer's weights should be on a device
     *
     * Uses PipelineConfig to determine if layer is in device's PP stage.
     *
     * @param layer_idx Layer index
     * @param device Target device
     */
    bool layerOnDevice(int layer_idx, DeviceId device) const;
    
    /**
     * @brief Pre-load weights for PP stage
     *
     * Loads only the weights needed for a specific PP stage's layers.
     *
     * @param stage_id PP stage ID
     * @param device Device to load to
     */
    bool preloadForPPStage(int stage_id, DeviceId device);
};
```

### 8.3 Weight Loading Flow for PP

```cpp
// In MultiDeviceOrchestrator::initializeDeviceRunners()

if (pipeline_config_->hasPP()) {
    auto* weight_mgr = model_ctx_->weightManager();
    weight_mgr->setPipelineConfig(pipeline_config_);
    
    // Pre-load weights per PP stage
    for (const auto& stage : pipeline_config_->pp_stages) {
        const auto* domain = pipeline_config_->getDomain(stage.domain_name);
        
        // For each device in the stage's TP domain
        for (const auto& device : domain->devices) {
            LOG_INFO("Pre-loading weights for PP stage " << stage.stage_id
                     << " (layers " << stage.first_layer << "-" << stage.last_layer
                     << ") on device " << device.to_string());
            
            if (!weight_mgr->preloadForPPStage(stage.stage_id, device)) {
                throw std::runtime_error("Failed to preload weights for PP stage");
            }
        }
    }
}
```

---

## 9. Graph Building Changes

### 9.1 DeviceGraphOrchestrator Graph Building

Each `DeviceGraphOrchestrator` builds a graph for **its layers only** based on PP assignment:

```cpp
// In DeviceGraphOrchestrator::buildComputeGraph()

ComputeGraph DeviceGraphOrchestrator::buildComputeGraph(
    const ModelContext& model,
    int seq_len,
    InferencePhase phase)
{
    auto& config = getDeviceGraphConfig();
    
    // Determine layer range
    int first_layer = config.pp_first_layer.value_or(0);
    int last_layer = config.pp_last_layer.value_or(model.layerCount());
    bool has_embedding = config.pp_has_embedding;
    bool has_lm_head = config.pp_has_lm_head;
    
    ComputeGraph graph;
    
    // Embedding (only if this device handles it)
    if (has_embedding) {
        graph.add(buildEmbeddingStage(model, seq_len));
    }
    
    // Transformer layers (only layers assigned to this device)
    for (int layer_idx = first_layer; layer_idx < last_layer; ++layer_idx) {
        // Map to local KV cache layer index
        int local_layer = layer_idx - first_layer;
        
        graph.add(buildAttentionStage(model, layer_idx, local_layer, seq_len));
        graph.add(buildFFNStage(model, layer_idx, seq_len));
        
        // TP AllReduce stages (if in TP mode)
        if (local_tp_ctx_) {
            graph.add(buildAllReduceStage("wo_allreduce", layer_idx));
            graph.add(buildAllReduceStage("ffn_down_allreduce", layer_idx));
        }
    }
    
    // LM Head (only if this device handles it)
    if (has_lm_head) {
        graph.add(buildLMHeadStage(model, seq_len));
        
        // TP AllGather for column-parallel LM head (if in TP mode)
        if (local_tp_ctx_) {
            // Note: AllGather for logits is handled by MultiDeviceOrchestrator
            // to gather across all TP devices, not within the graph
        }
    }
    
    return graph;
}
```

### 9.2 Special Cases for Embedding and LM Head

| Component | PP Stage 0 (has_embedding=true) | PP Stage N-1 (has_lm_head=true) | Middle Stages |
|-----------|-------------------------------|--------------------------------|---------------|
| Input | tokens (int) | hidden from prev stage | hidden from prev stage |
| Output | hidden | logits | hidden |
| Embedding | YES | NO | NO |
| LM Head | NO | YES | NO |

### 9.3 Graph Builder Interface Changes

```cpp
class Qwen2Graph : public IGraphBuilder {
public:
    // Extended config
    struct Config {
        // Existing fields...
        
        // PP layer range (optional - defaults to all layers)
        std::optional<int> first_layer;
        std::optional<int> last_layer;
        bool has_embedding = true;
        bool has_lm_head = true;
    };
    
    /**
     * @brief Build graph for assigned layer range
     *
     * @param phase Inference phase (PREFILL or DECODE)
     * @param seq_len Sequence length
     * @return ComputeGraph covering [first_layer, last_layer) + embedding/lm_head
     */
    ComputeGraph build(InferencePhase phase, int seq_len) override;
};
```

---

## 10. Activation Transfer Mechanism

### 10.1 Transfer Location: Coordinator vs Graph

**Recommendation**: **Handle transfers in the coordinator (MultiDeviceOrchestrator)**, not in the graph.

Rationale:
- Graphs are per-device; transfers are cross-device coordination
- Cleaner separation of concerns
- Easier to implement TP+PP (transfer between TP domains)
- Avoids dual-graph complexity of having LocalPPTransferStage span devices

```cpp
// PPExecutionStrategy::executeForward()

bool PPExecutionStrategy::executeForward(
    std::vector<DeviceGraphOrchestrator*>& runners,
    const int* tokens,
    int seq_len)
{
    const int num_stages = static_cast<int>(runners.size());
    
    for (int stage = 0; stage < num_stages; ++stage) {
        auto* runner = runners[stage];
        
        // 1. Execute this stage's graph
        if (stage == 0) {
            // First stage: process tokens
            if (!runner->forward(tokens, seq_len)) {
                LOG_ERROR("PP stage " << stage << " forward failed");
                return false;
            }
        } else {
            // Middle/last stages: process hidden from previous stage
            // (hidden was transferred in step 2 of previous iteration)
            if (!runner->forwardFromHidden(seq_len)) {
                LOG_ERROR("PP stage " << stage << " forward failed");
                return false;
            }
        }
        
        // 2. Transfer activations to next stage (if not last)
        if (stage < num_stages - 1) {
            TensorBase* hidden = runner->inferenceState().hidden.get();
            
            if (!pp_ctx_->transfer(hidden, stage, stage + 1)) {
                LOG_ERROR("PP transfer from stage " << stage << " to " << (stage + 1) << " failed");
                return false;
            }
            
            // Update next stage's hidden buffer pointer
            // (transfer may have changed the buffer location)
            runners[stage + 1]->setInputHidden(hidden);
        }
    }
    
    return true;
}
```

### 10.2 TP+PP Transfer: TP Domain → TP Domain

```cpp
// TPPPExecutionStrategy::executeForward()

bool TPPPExecutionStrategy::executeForward(
    std::vector<DeviceGraphOrchestrator*>& runners,
    const int* tokens,
    int seq_len)
{
    // Group runners by PP stage
    std::vector<std::vector<DeviceGraphOrchestrator*>> stage_runners;
    // ... group by stage ...
    
    for (int stage = 0; stage < num_pp_stages; ++stage) {
        auto& tp_runners = stage_runners[stage];
        auto* tp_ctx = stage_tp_contexts_[stage].get();
        
        // 1. Execute all TP devices for this stage IN PARALLEL
        std::vector<std::future<bool>> futures;
        for (auto* runner : tp_runners) {
            futures.push_back(std::async(std::launch::async, [=]() {
                if (stage == 0) {
                    return runner->forward(tokens, seq_len);
                } else {
                    return runner->forwardFromHidden(seq_len);
                }
            }));
        }
        
        // Wait for all TP devices to complete
        for (auto& f : futures) {
            if (!f.get()) {
                LOG_ERROR("PP stage " << stage << " TP execution failed");
                return false;
            }
        }
        
        // 2. Transfer to next stage's TP domain (if not last)
        if (stage < num_pp_stages - 1) {
            // After TP allreduce, all devices in this TP domain have identical hidden
            // Pick representative device (device 0) as transfer source
            TensorBase* hidden = tp_runners[0]->inferenceState().hidden.get();
            
            // Transfer to next stage's representative device
            if (!pp_ctx_->transfer(hidden, stage, stage + 1)) {
                LOG_ERROR("PP transfer failed");
                return false;
            }
            
            // The next stage's TP domain needs the same hidden on all devices
            // Option A: ILocalPPContext handles broadcast internally
            // Option B: Explicit broadcast in next stage's TP context
            auto* next_tp_ctx = stage_tp_contexts_[stage + 1].get();
            if (next_tp_ctx->degree() > 1) {
                // Broadcast to all devices in next TP domain
                next_tp_ctx->broadcast(hidden, 0);  // From device 0
            }
            
            // Update all next stage runners' hidden pointers
            for (auto* runner : stage_runners[stage + 1]) {
                runner->setInputHidden(hidden);
            }
        }
    }
    
    return true;
}
```

---

## 11. Test Infrastructure Changes

### 11.1 ParityTestBase Changes

```cpp
// tests/v2/integration/parity/ParityTestBase.h

class ParityTestBase : public ::testing::Test {
protected:
    // New factory method for PP tests
    std::unique_ptr<IInferenceRunner> createPPRunner(
        const std::string& model_path,
        const std::vector<std::pair<DeviceId, int>>& stage_layers)
    {
        // stage_layers: [(device, num_layers), ...]
        // e.g., [(cuda:0, 12), (cuda:1, 12)] for 24-layer model
        
        PipelineConfig pp_config;
        pp_config.total_layers = 0;
        int layer_offset = 0;
        
        for (size_t i = 0; i < stage_layers.size(); ++i) {
            const auto& [device, num_layers] = stage_layers[i];
            
            // Create single-device TP domain for this stage
            std::string domain_name = "stage" + std::to_string(i) + "_domain";
            pp_config.tp_domains.push_back({
                domain_name, {device}, CollectiveBackendType::AUTO
            });
            
            // Create PP stage
            PPStageConfig stage;
            stage.stage_id = static_cast<int>(i);
            stage.domain_name = domain_name;
            stage.first_layer = layer_offset;
            stage.last_layer = layer_offset + num_layers;
            stage.has_embedding = (i == 0);
            stage.has_lm_head = (i == stage_layers.size() - 1);
            pp_config.pp_stages.push_back(stage);
            
            layer_offset += num_layers;
        }
        
        pp_config.total_layers = layer_offset;
        
        // Set transfer backends
        for (size_t i = 0; i < stage_layers.size() - 1; ++i) {
            pp_config.pp_transfer_backends[{i, i + 1}] = CollectiveBackendType::AUTO;
        }
        
        // Create orchestrator
        MultiDeviceOrchestrator::Config config;
        config.pipeline_config = std::make_shared<PipelineConfig>(pp_config);
        config.max_seq_len = max_seq_len_;
        
        return std::make_unique<MultiDeviceOrchestrator>(
            loadModel(model_path), config);
    }
    
    // New factory for TP+PP tests
    std::unique_ptr<IInferenceRunner> createTPPPRunner(
        const std::string& model_path,
        const std::vector<TPDomainConfig>& tp_domains,
        const std::vector<PPStageConfig>& pp_stages)
    {
        PipelineConfig pp_config;
        pp_config.tp_domains = tp_domains;
        pp_config.pp_stages = pp_stages;
        pp_config.total_layers = /* compute from stages */;
        pp_config.autoSelectBackends();
        
        MultiDeviceOrchestrator::Config config;
        config.pipeline_config = std::make_shared<PipelineConfig>(pp_config);
        
        return std::make_unique<MultiDeviceOrchestrator>(
            loadModel(model_path), config);
    }
};
```

### 11.2 New Test Cases

```cpp
// tests/v2/integration/parity/Test__PPParityQwen2.cpp

TEST_F(PPParityTest, TwoStagePP_CUDA_CUDA) {
    // 2-stage PP: layers 0-13 on cuda:0, layers 14-27 on cuda:1
    auto runner = createPPRunner("qwen2-0.5b.gguf", {
        {DeviceId::cuda(0), 14},
        {DeviceId::cuda(1), 14},
    });
    
    auto ref = createSingleDeviceRunner("qwen2-0.5b.gguf", DeviceId::cuda(0));
    
    runParityTest(runner.get(), ref.get(), test_prompts_);
}

TEST_F(PPParityTest, ThreeStagePP_Heterogeneous) {
    // 3-stage PP: cuda:0, rocm:0, cpu
    auto runner = createPPRunner("qwen2-0.5b.gguf", {
        {DeviceId::cuda(0), 10},
        {DeviceId::rocm(0), 10},
        {DeviceId::cpu(), 8},
    });
    
    // ... parity test ...
}

TEST_F(TPPPParityTest, TwoStage_TwoWayTP) {
    // PP(TP(cuda:0,cuda:1), TP(rocm:0,rocm:1))
    auto runner = createTPPPRunner("qwen2-0.5b.gguf",
        /* tp_domains */ {
            {"stage0_tp", {DeviceId::cuda(0), DeviceId::cuda(1)}, NCCL},
            {"stage1_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, RCCL},
        },
        /* pp_stages */ {
            PPStageConfig::firstStage(0, "stage0_tp", 0, 14),
            PPStageConfig::lastStage(1, "stage1_tp", 14, 28),
        });
    
    // ... parity test ...
}
```

---

## 12. Migration Path

### Phase 3.1: Core Infrastructure (Week 1)

1. **Add ParallelismMode enum** to `OrchestrationConfig.h`
2. **Add IExecutionStrategy interface** to `execution/strategies/`
3. **Implement TPExecutionStrategy** (extract from current MultiDeviceOrchestrator)
4. **Add pp_assignment to DeviceGraphConfig**
5. **Unit tests for strategy interface**

### Phase 3.2: PP Execution Strategy (Week 2)

1. **Implement PPExecutionStrategy**
   - Sequential stage execution
   - Activation transfer via ILocalPPContext
2. **Extend DeviceGraphOrchestrator** for PP layer range
   - `forwardFromHidden()` method
   - Layer-range-aware graph building
3. **Extend InferenceState** for PP KV cache
4. **Unit tests for PPExecutionStrategy** (mock runners)

### Phase 3.3: Weight Distribution for PP (Week 2-3)

1. **Extend WeightManager** for PP layer filtering
   - `setPipelineConfig()`
   - `layerOnDevice()`
   - `preloadForPPStage()`
2. **Update weight preloading** in MultiDeviceOrchestrator
3. **Integration tests** with real model weights

### Phase 3.4: MultiDeviceOrchestrator Integration (Week 3)

1. **Extend Config** with `pipeline_config`
2. **Add strategy selection** in constructor
3. **Implement PP mode initialization**
   - Create PPExecutionStrategy
   - Create per-stage DeviceGraphOrchestrators
   - Setup ILocalPPContext
4. **Integration tests** for 2-stage PP

### Phase 3.5: TP+PP Hybrid (Week 4)

1. **Implement TPPPExecutionStrategy**
2. **Handle TP domain → TP domain transfers**
3. **Integration tests** for hybrid mode
4. **Parity tests** against single-device reference

### Phase 3.6: Test Infrastructure (Week 4-5)

1. **Update ParityTestBase** with PP factory methods
2. **Add PP parity test suite**
3. **Add TP+PP parity test suite**
4. **Performance benchmarks** for PP vs single-device

---

## 13. Code Sketches

### 13.1 Extended MultiDeviceOrchestrator Header

```cpp
// src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h

#pragma once

#include "IMultiDeviceOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include "../strategies/IExecutionStrategy.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/ILocalPPContext.h"
#include "../../../config/PipelineConfig.h"
#include <memory>
#include <vector>

namespace llaminar2 {

enum class ParallelismMode {
    AUTO,           // Infer from config
    SINGLE_DEVICE,
    TP_ONLY,
    PP_ONLY,
    TP_PLUS_PP,
};

class MultiDeviceOrchestrator : public IMultiDeviceOrchestrator {
public:
    struct Config {
        // Mode (AUTO = infer)
        ParallelismMode mode = ParallelismMode::AUTO;
        
        // Common
        size_t max_seq_len = 4096;
        int batch_size = 1;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        
        // TP-only config (simple mode)
        std::vector<GlobalDeviceAddress> devices;
        std::vector<float> weights;
        CollectiveBackendType tp_backend = CollectiveBackendType::AUTO;
        
        // Full pipeline config (PP or TP+PP)
        std::shared_ptr<PipelineConfig> pipeline_config;
        
        bool validate() const;
        ParallelismMode inferMode() const;
    };
    
    // Constructors
    MultiDeviceOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config& config);
    
    ~MultiDeviceOrchestrator() override;
    
    // IInferenceRunner
    bool forward(const int* tokens, int seq_len) override;
    const float* logits() const override;
    void clear_cache() override;
    int get_position() const override;
    int vocab_size() const override;
    const char* architecture() const override;
    
    // IMultiDeviceOrchestrator
    int device_count() const override;
    IInferenceRunner* deviceRunner(int device_idx) override;
    ILocalTPContext* localTPContext() override;
    ILocalPPContext* localPPContext() override;
    ParallelismMode parallelismMode() const override;
    const PPDeviceAssignment* ppAssignment(int device_idx) const override;
    bool allDevicesReady() const override;
    void synchronizeDevices() override;
    
private:
    void initializeForTP();
    void initializeForPP();
    void initializeForTPPP();
    void createStrategy();
    
    std::shared_ptr<IModelContext> model_ctx_;
    Config config_;
    ParallelismMode mode_;
    
    // Device runners (one per device regardless of TP/PP mode)
    std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners_;
    
    // PP assignments (only set in PP/TP+PP modes)
    std::vector<PPDeviceAssignment> pp_assignments_;
    
    // Contexts
    std::unique_ptr<ILocalTPContext> tp_ctx_;           // TP-only mode
    std::unique_ptr<ILocalPPContext> pp_ctx_;           // PP modes
    std::vector<std::unique_ptr<ILocalTPContext>> stage_tp_contexts_; // TP+PP mode
    
    // Execution strategy
    std::unique_ptr<IExecutionStrategy> strategy_;
    
    // Output buffer
    std::unique_ptr<TensorBase> combined_logits_;
};

} // namespace llaminar2
```

### 13.2 IExecutionStrategy Interface

```cpp
// src/v2/execution/local_execution/strategies/IExecutionStrategy.h

#pragma once

#include "../orchestrators/DeviceGraphOrchestrator.h"
#include <vector>

namespace llaminar2 {

enum class ParallelismMode;

class IExecutionStrategy {
public:
    virtual ~IExecutionStrategy() = default;
    
    /**
     * @brief Execute forward pass across all devices
     */
    virtual bool executeForward(
        std::vector<DeviceGraphOrchestrator*>& runners,
        const int* tokens,
        int seq_len) = 0;
    
    /**
     * @brief Gather final logits
     */
    virtual bool gatherLogits(
        std::vector<DeviceGraphOrchestrator*>& runners,
        TensorBase* output_buffer,
        size_t seq_len) = 0;
    
    /**
     * @brief Clear caches on all devices
     */
    virtual void clearCaches(
        std::vector<DeviceGraphOrchestrator*>& runners) = 0;
    
    /**
     * @brief Get current position (max across all devices)
     */
    virtual int getPosition(
        const std::vector<DeviceGraphOrchestrator*>& runners) const = 0;
    
    /**
     * @brief Get parallelism mode
     */
    virtual ParallelismMode mode() const = 0;
};

} // namespace llaminar2
```

### 13.3 PPExecutionStrategy Implementation Sketch

```cpp
// src/v2/execution/local_execution/strategies/PPExecutionStrategy.cpp

#include "PPExecutionStrategy.h"

namespace llaminar2 {

PPExecutionStrategy::PPExecutionStrategy(ILocalPPContext* pp_ctx)
    : pp_ctx_(pp_ctx)
{
    if (!pp_ctx_) {
        throw std::invalid_argument("pp_ctx cannot be null");
    }
}

bool PPExecutionStrategy::executeForward(
    std::vector<DeviceGraphOrchestrator*>& runners,
    const int* tokens,
    int seq_len)
{
    const int num_stages = static_cast<int>(runners.size());
    
    for (int stage = 0; stage < num_stages; ++stage) {
        auto* runner = runners[stage];
        
        LOG_DEBUG("PPExecutionStrategy: Executing stage " << stage 
                  << " on device " << runner->device().to_string());
        
        // Execute this stage
        bool success;
        if (stage == 0) {
            // First stage: embed tokens
            success = runner->forward(tokens, seq_len);
        } else {
            // Later stages: continue from hidden state
            // Hidden was set by previous transfer
            success = runner->forwardFromHidden(seq_len);
        }
        
        if (!success) {
            LOG_ERROR("PPExecutionStrategy: Stage " << stage << " failed");
            return false;
        }
        
        // Transfer to next stage (if not last)
        if (stage < num_stages - 1) {
            auto* hidden = runner->inferenceState().hidden.get();
            
            LOG_DEBUG("PPExecutionStrategy: Transferring activations "
                      << "from stage " << stage << " to " << (stage + 1));
            
            if (!pp_ctx_->transfer(hidden, stage, stage + 1)) {
                LOG_ERROR("PPExecutionStrategy: Transfer failed");
                return false;
            }
            
            // Point next stage's input to transferred hidden
            runners[stage + 1]->setInputHidden(hidden);
        }
    }
    
    return true;
}

bool PPExecutionStrategy::gatherLogits(
    std::vector<DeviceGraphOrchestrator*>& runners,
    TensorBase* output_buffer,
    size_t seq_len)
{
    // Logits come from last stage only
    auto* last_runner = runners.back();
    const float* stage_logits = last_runner->logits();
    
    if (!stage_logits) {
        LOG_ERROR("PPExecutionStrategy: Last stage has no logits");
        return false;
    }
    
    // Copy to output buffer
    size_t vocab = last_runner->vocab_size();
    size_t copy_size = seq_len * vocab;
    std::memcpy(output_buffer->mutable_data(), stage_logits, 
                copy_size * sizeof(float));
    
    return true;
}

void PPExecutionStrategy::clearCaches(
    std::vector<DeviceGraphOrchestrator*>& runners)
{
    for (auto* runner : runners) {
        runner->clear_cache();
    }
}

int PPExecutionStrategy::getPosition(
    const std::vector<DeviceGraphOrchestrator*>& runners) const
{
    // All stages should have same position
    return runners.empty() ? 0 : runners[0]->get_position();
}

} // namespace llaminar2
```

### 13.4 DeviceGraphOrchestrator PP Extensions

```cpp
// In DeviceGraphOrchestrator.h

struct DeviceGraphConfig {
    // Existing TP fields...
    ILocalTPContext* local_tp_ctx = nullptr;
    int local_tp_device_index = 0;
    
    // PP fields
    std::optional<int> pp_first_layer;
    std::optional<int> pp_last_layer;
    bool pp_has_embedding = true;
    bool pp_has_lm_head = true;
    
    bool isPP() const {
        return pp_first_layer.has_value() && pp_last_layer.has_value();
    }
    
    int ppLayerCount() const {
        return isPP() ? (*pp_last_layer - *pp_first_layer) : -1;
    }
};

class DeviceGraphOrchestrator : public IInferenceRunner {
public:
    // ... existing methods ...
    
    /**
     * @brief Forward pass starting from pre-set hidden state
     *
     * Used in PP mode for stages that receive hidden from previous stage
     * instead of embedding tokens.
     *
     * @param seq_len Sequence length
     * @return true on success
     */
    bool forwardFromHidden(int seq_len);
    
    /**
     * @brief Set input hidden state for PP forward
     *
     * Called by coordinator after PP transfer to point this stage's
     * input to the transferred hidden buffer.
     *
     * @param hidden Hidden state tensor (may be on different device)
     */
    void setInputHidden(TensorBase* hidden);
    
    /**
     * @brief Get mutable reference to inference state
     */
    InferenceState& inferenceState() { return state_; }
    
private:
    DeviceGraphConfig device_config_;
};
```

---

## Summary

This design unifies TP and PP handling in `MultiDeviceOrchestrator` through:

1. **Same fundamental pattern**: N devices → N `DeviceGraphOrchestrator` instances, each with own state
2. **Strategy pattern for coordination**: `IExecutionStrategy` encapsulates TP vs PP vs hybrid execution
3. **Per-device state**: Each device owns its KV cache (sharded by heads for TP, by layers for PP)
4. **Coordinator-level transfers**: PP activation transfers happen in the strategy, not in device graphs
5. **Composable TP+PP**: Each PP stage can be a TP domain; transfer happens between domains

The migration path is incremental, allowing us to build and test PP support without disrupting working TP functionality.
