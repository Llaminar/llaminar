# Project Plan: PP+TP Infrastructure Improvements

**Date**: February 2026  
**Author**: David Sanftenberg  
**Scope**: Per-stage buffer pools + automatic backend selection for complex PP+TP compositions

---

## Overview

This plan addresses two critical infrastructure gaps for supporting complex pipeline parallel + tensor parallel (PP+TP) compositions like:

```
PipelineParallel(TensorParallel(rocm:0, rocm:1), rocm:2, cuda:0)
```

| Gap | Current State | Impact |
|-----|---------------|--------|
| **Single Buffer Pool** | `Qwen2ActivationBuffers` assumes one device | Cannot run PP stages on different devices |
| **Manual Backend Selection** | `pp_transfer_backends` must be manually specified | User must understand device topology |

---

## Part 1: Per-Stage Buffer Pools

### 1.1 Problem Statement

The current `Qwen2ActivationBuffers` is a flat structure that assumes all buffers reside on a single device. In a multi-stage PP setup, each stage runs on a different device and needs its own buffer allocation:

```
PP Stage 0 (rocm:0+rocm:1) → buffers on rocm:0 (primary)
PP Stage 1 (rocm:2)        → buffers on rocm:2
PP Stage 2 (cuda:0)        → buffers on cuda:0
```

### 1.2 Design

#### Option A: Per-Stage Buffer Map (Recommended)

Introduce a `PerStageBufferPool` that wraps `Qwen2ActivationBuffers` with per-stage allocation:

```cpp
// src/v2/execution/local_execution/device/PerStageBufferPool.h

/**
 * @brief Per-PP-stage buffer allocation
 *
 * Each PP stage gets its own Qwen2ActivationBuffers allocated on the
 * stage's primary device. Enables heterogeneous PP execution.
 */
class PerStageBufferPool {
public:
    /**
     * @brief Initialize buffer pools for all PP stages
     * @param config Pipeline configuration with stage→device mapping
     * @param buffer_spec Buffer specification (shapes, dtypes)
     * @return true if allocation succeeded on all devices
     */
    bool initialize(const PipelineConfig& config, const BufferSpec& buffer_spec);

    /**
     * @brief Get buffers for a specific PP stage
     * @param stage_id PP stage index (0, 1, 2, ...)
     * @return Reference to stage's activation buffers
     * @throws std::out_of_range if stage_id invalid
     */
    Qwen2ActivationBuffers& forStage(int stage_id);
    const Qwen2ActivationBuffers& forStage(int stage_id) const;

    /**
     * @brief Get buffers for the stage that owns a layer
     * @param layer_idx Layer index (0 to n_layers-1)
     * @return Reference to owning stage's buffers
     */
    Qwen2ActivationBuffers& forLayer(int layer_idx);

    /**
     * @brief Release all allocated buffers
     */
    void release();

    /**
     * @brief Get allocation statistics
     */
    const DomainAllocationStats& stats() const;

private:
    /// Stage ID → allocated buffers
    std::map<int, std::unique_ptr<Qwen2ActivationBuffers>> stage_buffers_;
    
    /// Stage ID → owning tensor storage (keeps tensors alive)
    std::map<int, std::vector<std::unique_ptr<TensorBase>>> tensor_storage_;
    
    /// Reference to pipeline config for stage→device lookup
    const PipelineConfig* config_ = nullptr;
    
    /// Allocation statistics
    DomainAllocationStats stats_;
};
```

#### BufferSpec Structure

```cpp
// src/v2/execution/local_execution/device/BufferSpec.h

/**
 * @brief Specification for buffer allocation sizes
 *
 * Captures the shapes needed for each buffer type, allowing
 * allocation without knowing the concrete tensor types.
 */
struct BufferSpec {
    size_t batch_size = 1;
    size_t seq_len = 512;
    size_t d_model = 896;
    size_t n_heads = 14;
    size_t n_kv_heads = 2;
    size_t head_dim = 64;
    size_t intermediate_size = 4864;
    size_t vocab_size = 151936;

    /// Activation precision (FP32, HybridQ16, etc.)
    ActivationPrecision precision = ActivationPrecision::FP32;

    /// Whether to allocate snapshot buffers
    bool enable_snapshots = false;

    // Derived dimensions
    size_t kv_dim() const { return n_kv_heads * head_dim; }
    size_t hidden_dim() const { return d_model; }
};
```

### 1.3 Integration Points

| Component | Change Required |
|-----------|-----------------|
| `DeviceGraphOrchestrator` | Store `PerStageBufferPool` instead of single `Qwen2ActivationBuffers` |
| `Qwen2Graph::buildUnifiedPipelineGraph()` | Accept `PerStageBufferPool&` or query buffers via stage accessor |
| `GraphBuildSession` | Add `.withBufferPool(PerStageBufferPool&)` method |
| `executeLayer()` | Query `buffer_pool_.forLayer(layer_idx)` instead of single buffer |

### 1.4 Implementation Tasks

| Task | File(s) | Estimate |
|------|---------|----------|
| 1.4.1 Create `BufferSpec` struct | `src/v2/execution/local_execution/device/BufferSpec.h` | 0.5h |
| 1.4.2 Create `PerStageBufferPool` class | `src/v2/execution/local_execution/device/PerStageBufferPool.h/cpp` | 2h |
| 1.4.3 Add `PerStageBufferPool` to `DeviceGraphOrchestrator` | `DeviceGraphOrchestrator.h/cpp` | 1h |
| 1.4.4 Update `GraphBuildSession` with `.withBufferPool()` | `DeviceGraphOrchestrator.h/cpp` | 0.5h |
| 1.4.5 Update `Qwen2Graph::buildUnifiedPipelineGraph()` | `Qwen2Graph.cpp` | 2h |
| 1.4.6 Update `executeLayer()` to use pool | `DeviceGraphOrchestrator.cpp` | 1h |
| 1.4.7 Unit tests for `PerStageBufferPool` | `tests/v2/unit/execution/device/Test__PerStageBufferPool.cpp` | 2h |
| 1.4.8 Integration test: 2-stage PP on different devices | `tests/v2/integration/Test__PP_MultiDevice.cpp` | 2h |

**Part 1 Total: ~11 hours**

---

## Part 2: Automatic Backend Selection

### 2.1 Problem Statement

Currently, `PipelineConfig::pp_transfer_backends` must be manually populated:

```cpp
config.pp_transfer_backends = {
    {{0, 1}, CollectiveBackendType::PCIE_BAR},  // rocm→rocm (manual)
    {{1, 2}, CollectiveBackendType::PCIE_BAR}   // rocm→cuda (manual)
};
```

Users shouldn't need to understand the device topology to get correct backends.

### 2.2 Existing Infrastructure

The `LocalPPContext.cpp` already has a `selectBackendForTransfer()` helper:

```cpp
static CollectiveBackendType selectBackendForTransfer(DeviceType src, DeviceType dst)
{
    // CUDA → CUDA: NCCL
    // ROCm → ROCm: RCCL
    // CUDA ↔ ROCm: PCIE_BAR
    // CPU involved: HOST
}
```

This logic needs to be:
1. Exposed as a public utility
2. Integrated into `PipelineConfig` validation/completion
3. Used by `ILocalPPContext::backendForTransfer()` when `AUTO`

### 2.3 Design

#### Auto-Selection Utility

```cpp
// src/v2/config/BackendSelector.h

/**
 * @brief Automatic backend selection based on device topology
 *
 * Encapsulates the rules for selecting optimal collective backends
 * for different device-pair combinations.
 */
class BackendSelector {
public:
    /**
     * @brief Select optimal backend for PP activation transfer
     *
     * Rules:
     * - CUDA → CUDA: NCCL (low latency, high bandwidth)
     * - ROCm → ROCm: RCCL (AMD equivalent)
     * - CUDA ↔ ROCm: PCIE_BAR (cross-vendor GPU)
     * - GPU ↔ CPU: HOST (staging through host memory)
     * - CPU → CPU: HOST (or UPI if available)
     *
     * @param src Source device
     * @param dst Destination device
     * @return Selected backend type
     */
    static CollectiveBackendType selectForTransfer(DeviceId src, DeviceId dst);

    /**
     * @brief Select optimal backend for TP allreduce within a domain
     *
     * Rules:
     * - All CUDA: NCCL
     * - All ROCm: RCCL
     * - Mixed CUDA+ROCm: PCIE_BAR (2 devices) or HETEROGENEOUS (3+)
     * - All CPU: MPI or UPI
     *
     * @param devices Devices in the TP domain
     * @return Selected backend type
     */
    static CollectiveBackendType selectForTPDomain(const std::vector<DeviceId>& devices);

    /**
     * @brief Check if a backend is available at runtime
     *
     * Verifies that the required library (NCCL, RCCL) is linked
     * and the devices support the backend.
     *
     * @param backend Backend type to check
     * @param devices Devices that would use this backend
     * @return true if backend can be used
     */
    static bool isBackendAvailable(CollectiveBackendType backend,
                                   const std::vector<DeviceId>& devices);
};
```

#### PipelineConfig Auto-Completion

Add a method to `PipelineConfig` that fills in missing backend selections:

```cpp
// In PipelineConfig

/**
 * @brief Auto-complete missing backend selections
 *
 * For any {from_stage, to_stage} pair not in pp_transfer_backends,
 * uses BackendSelector to determine the optimal backend based on
 * the stage devices.
 *
 * Also sets tp_backend=AUTO domains to their optimal backends.
 */
void autoSelectBackends();

/**
 * @brief Validate and auto-complete (convenience method)
 *
 * Calls autoSelectBackends() then validate().
 *
 * @param error_msg Output: error message if invalid
 * @return true if valid after auto-completion
 */
bool completeAndValidate(std::string* error_msg = nullptr);
```

### 2.4 Backend Selection Matrix

| Source | Destination | Selected Backend | Rationale |
|--------|-------------|------------------|-----------|
| CUDA | CUDA | NCCL | Native NVIDIA collective |
| ROCm | ROCm | RCCL | Native AMD collective |
| CUDA | ROCm | PCIE_BAR | Cross-vendor via PCIe P2P |
| ROCm | CUDA | PCIE_BAR | Cross-vendor via PCIe P2P |
| GPU | CPU | HOST | Stage through host memory |
| CPU | GPU | HOST | Stage through host memory |
| CPU | CPU | HOST (or UPI) | No GPU involvement |

### 2.5 Implementation Tasks

| Task | File(s) | Estimate |
|------|---------|----------|
| 2.5.1 Create `BackendSelector` utility | `src/v2/config/BackendSelector.h/cpp` | 1.5h |
| 2.5.2 Move `selectBackendForTransfer()` from `LocalPPContext.cpp` | Refactor existing code | 0.5h |
| 2.5.3 Add `BackendSelector::selectForTPDomain()` | `BackendSelector.cpp` | 1h |
| 2.5.4 Add `BackendSelector::isBackendAvailable()` | `BackendSelector.cpp` | 1h |
| 2.5.5 Add `PipelineConfig::autoSelectBackends()` | `PipelineConfig.cpp` | 1h |
| 2.5.6 Add `PipelineConfig::completeAndValidate()` | `PipelineConfig.cpp` | 0.5h |
| 2.5.7 Update `TPDomainConfig` to use `BackendSelector` when `AUTO` | `TPDomainConfig.cpp` | 0.5h |
| 2.5.8 Unit tests for `BackendSelector` | `tests/v2/unit/config/Test__BackendSelector.cpp` | 2h |
| 2.5.9 Update `PipelineConfig` tests | `tests/v2/unit/config/Test__PipelineConfig.cpp` | 1h |
| 2.5.10 Integration test: auto-backend with mixed devices | `tests/v2/integration/Test__AutoBackendSelection.cpp` | 1.5h |

**Part 2 Total: ~10.5 hours**

---

## Part 3: Integration

### 3.1 DeviceGraphOrchestrator Updates

The orchestrator needs to:
1. Create `PerStageBufferPool` when `PipelineConfig` is set
2. Call `PipelineConfig::completeAndValidate()` before use
3. Pass buffer pool to graph building

```cpp
// DeviceGraphOrchestrator.cpp

void DeviceGraphOrchestrator::setPipelineConfig(std::shared_ptr<PipelineConfig> config)
{
    // Auto-complete backend selections
    std::string error;
    if (!config->completeAndValidate(&error)) {
        throw std::invalid_argument("Invalid pipeline config: " + error);
    }

    pipeline_config_ = std::move(config);

    // Allocate per-stage buffers
    if (pipeline_config_->hasPP()) {
        BufferSpec spec = createBufferSpecFromConfig(graph_builder_->config());
        if (!buffer_pool_.initialize(*pipeline_config_, spec)) {
            throw std::runtime_error("Failed to allocate per-stage buffers");
        }
    }
}
```

### 3.2 Qwen2Graph Updates

The graph builder needs to accept the buffer pool:

```cpp
// Option A: Pass pool to buildUnifiedPipelineGraph
ComputeGraph Qwen2Graph::buildUnifiedPipelineGraph(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output,
    PerStageBufferPool& buffer_pool);  // NEW parameter

// Option B: Store buffer pool in Qwen2GraphConfig
struct Qwen2GraphConfig {
    // ... existing fields ...
    PerStageBufferPool* buffer_pool = nullptr;  // For PP mode
};
```

**Recommendation**: Option B (config field) keeps the API cleaner.

### 3.3 Implementation Tasks

| Task | File(s) | Estimate |
|------|---------|----------|
| 3.3.1 Update `DeviceGraphOrchestrator::setPipelineConfig()` | `DeviceGraphOrchestrator.cpp` | 1h |
| 3.3.2 Add `buffer_pool_` member to orchestrator | `DeviceGraphOrchestrator.h/cpp` | 0.5h |
| 3.3.3 Add `PerStageBufferPool*` to `Qwen2GraphConfig` | `Qwen2Graph.h` | 0.5h |
| 3.3.4 Update `buildUnifiedPipelineGraph()` to use pool | `Qwen2Graph.cpp` | 1.5h |
| 3.3.5 Update `GraphBuildSession::buildUnified()` | `DeviceGraphOrchestrator.cpp` | 1h |
| 3.3.6 End-to-end integration test: 3-stage PP | `tests/v2/integration/Test__ThreeStagePP.cpp` | 3h |

**Part 3 Total: ~7.5 hours**

---

## Summary

### Total Effort

| Part | Description | Estimate |
|------|-------------|----------|
| Part 1 | Per-Stage Buffer Pools | 11h |
| Part 2 | Automatic Backend Selection | 10.5h |
| Part 3 | Integration | 7.5h |
| **Total** | | **29 hours** |

### Dependency Graph

```
Part 1: Per-Stage Buffer Pools
    └── 1.4.1 BufferSpec
    └── 1.4.2 PerStageBufferPool ──────────────────────┐
                                                       │
Part 2: Auto Backend Selection                         │
    └── 2.5.1 BackendSelector                          │
    └── 2.5.5 PipelineConfig::autoSelectBackends() ───┐│
                                                      ││
Part 3: Integration                                   ││
    └── 3.3.1 setPipelineConfig() ◄────────────────────┘│
    └── 3.3.4 buildUnifiedPipelineGraph() ◄────────────┘
    └── 3.3.6 End-to-end test
```

### Files to Create

| File | Purpose |
|------|---------|
| `src/v2/execution/local_execution/device/BufferSpec.h` | Buffer allocation specification |
| `src/v2/execution/local_execution/device/PerStageBufferPool.h` | Per-stage buffer pool header |
| `src/v2/execution/local_execution/device/PerStageBufferPool.cpp` | Per-stage buffer pool impl |
| `src/v2/config/BackendSelector.h` | Backend selection utility header |
| `src/v2/config/BackendSelector.cpp` | Backend selection utility impl |
| `tests/v2/unit/execution/device/Test__PerStageBufferPool.cpp` | Unit tests |
| `tests/v2/unit/config/Test__BackendSelector.cpp` | Unit tests |
| `tests/v2/integration/Test__PP_MultiDevice.cpp` | Integration tests |
| `tests/v2/integration/Test__ThreeStagePP.cpp` | End-to-end tests |

### Files to Modify

| File | Changes |
|------|---------|
| `DeviceGraphOrchestrator.h` | Add `buffer_pool_` member, `PerStageBufferPool` include |
| `DeviceGraphOrchestrator.cpp` | Update `setPipelineConfig()`, `executeLayer()` |
| `Qwen2Graph.h` | Add `PerStageBufferPool*` to config |
| `Qwen2Graph.cpp` | Update `buildUnifiedPipelineGraph()` |
| `PipelineConfig.h` | Add `autoSelectBackends()`, `completeAndValidate()` |
| `PipelineConfig.cpp` | Implement auto-completion |
| `LocalPPContext.cpp` | Refactor to use `BackendSelector` |
| `CMakeLists.txt` | Add new source files |

---

## Success Criteria

1. **Per-Stage Buffers**: A 3-stage PP config with `[rocm:0+rocm:1, rocm:2, cuda:0]` allocates separate buffer pools on each primary device
2. **Auto Backend Selection**: `PipelineConfig` with `pp_transfer_backends={}` (empty) auto-selects RCCL for rocm→rocm, PCIE_BAR for rocm→cuda
3. **Integration**: `./llaminar2 -m model.gguf --pp 3 --tp-devices "rocm:0,rocm:1" "rocm:2" "cuda:0"` works without explicit backend flags
4. **No Regressions**: All existing 336 unit tests pass

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU memory fragmentation with multiple buffer pools | Medium | High | Pre-allocate pools at init, reuse across forward passes |
| Backend availability detection false positives | Low | Medium | Test `isBackendAvailable()` with actual hardware in CI |
| Complex PP+TP configurations untested | Medium | High | Add matrix of integration tests (Part 3.3.6) |

---

## Next Steps After Completion

After Parts 1-3 are complete, the next phase will be:

**CLI Syntax Parsing**: Implement parsing for the nested topology syntax:
```bash
./llaminar2 --topology "PP(TP(rocm:0, rocm:1), rocm:2, cuda:0)" -m model.gguf
```

This will be covered in a separate project plan once the infrastructure is in place.
