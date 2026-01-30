# GPU Workspace Buffer Management - Code Proposals

**Author:** David Sanftenberg  
**Created:** January 14, 2026  
**Status:** Code Proposals  
**Related:** [GPU_WORKSPACE_BUFFER_MANAGEMENT_PROJECT_PLAN.md](GPU_WORKSPACE_BUFFER_MANAGEMENT_PROJECT_PLAN.md)

This document contains concrete, copy-pasteable code proposals for implementing the GPU Workspace Buffer Management project.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Phase 0: Generic Buffer Declaration System](#phase-0-generic-buffer-declaration-system)
3. [Phase 1: GPU Workspace Manager](#phase-1-gpu-workspace-manager)
4. [Phase 2: Kernel Integration](#phase-2-kernel-integration)
5. [Phase 3: Slab-Based FP16 GEMM](#phase-3-slab-based-fp16-gemm)
6. [Phase 4: Memory Budget Configuration](#phase-4-memory-budget-configuration)
7. [Implementation Order](#implementation-order)

---

## Design Philosophy

### Core Principles

1. **Stage-Centric Declaration**: Stages declare what buffers they need. The graph doesn't interpret buffer semantics.
2. **No Hardcoded Semantics**: No enums like `GEMM_QUANTIZE` or `ATTENTION_SCORES` in the graph/buffer system.
3. **Type + Size Only**: The graph only needs to know data type, size, and alignment - not purpose.
4. **Dynamic Sizing**: Buffer sizes computed by stages based on workload parameters (M, K, N, seq_len, etc.).
5. **Declarative Graph**: Buffers are just another declaration stages make, like inputs/outputs.

### What the Graph Knows vs. What Stages Know

| Concern | Graph/Buffer Manager | Stage |
|---------|---------------------|-------|
| Buffer name | Opaque string key | Meaningful internal name |
| Buffer purpose | Don't care | Knows exactly what it's for |
| Data type | `DataType::FP32`, etc. | Knows why that type is needed |
| Size | Bytes (computed) | Knows formula: `M * K * sizeof(int8_t)` |
| Aliasability | Hint from stage | Knows if buffer can be shared |
| Lifetime | Tracks via liveness | Knows when it's needed |

### Anti-Pattern (What We're Avoiding)

```cpp
// ❌ BAD: Graph knows about GEMM-specific buffer types
enum class GpuWorkspaceCategory {
    GEMM_QUANTIZE,      // Graph shouldn't know about GEMM internals
    ATTENTION_SCORES,   // Graph shouldn't know about attention internals
    // ...semantic coupling to specific algorithms
};

// ❌ BAD: Graph has semantic knowledge
void GraphBuilder::allocateGemmBuffers(GEMMStage* stage) {
    allocate("quant_a", stage->M * stage->K);  // Graph knows GEMM formulas!
}
```

### Correct Pattern (What We Want)

```cpp
// ✅ GOOD: Stage declares what it needs, graph is agnostic
WorkspaceRequirements GEMMStage::getWorkspaceRequirements() const {
    WorkspaceRequirements reqs;
    // Stage knows its own formulas
    reqs.add("quant_a", params_.M * params_.K, DataType::INT8);
    reqs.add("scales", params_.M, DataType::FP32);
    reqs.add("acc", params_.M * params_.N, DataType::INT32);
    return reqs;
}

// ✅ GOOD: Graph just collects and allocates
void GraphBuilder::build() {
    for (auto& stage : stages_) {
        auto gpu_reqs = stage->getWorkspaceRequirements();
        workspace_manager_.registerBuffers(stage->name(), gpu_reqs);
    }
}
```

---

## Phase 0: Generic Buffer Declaration System

This phase establishes a **semantics-free** buffer declaration system where stages declare their needs and the graph just allocates.

### 0.1 WorkspaceDescriptor.h (Generic Design)

```cpp
/**
 * @file WorkspaceDescriptor.h
 * @brief GPU workspace buffer requirement descriptors
 * @author David Sanftenberg
 * @date January 2026
 * 
 * DESIGN: The graph/buffer system is buffer-agnostic. It doesn't know or care
 * what a buffer is used for. Stages declare what they need; the graph allocates.
 */

#pragma once

#include "tensors/DataType.h"  // Existing: FP32, FP16, INT8, INT32, etc.

#include <cstddef>
#include <string>
#include <vector>
#include <functional>

namespace llaminar2
{

/**
 * @brief Describes a single GPU workspace buffer requirement
 * 
 * NOTE: No semantic fields like "category" or "purpose". The graph doesn't
 * need to know what a buffer is for - just its type, size, and constraints.
 */
struct WorkspaceDescriptor
{
    std::string name;           ///< Stage-internal name (opaque to graph)
    size_t size_bytes;          ///< Required size in bytes
    DataType dtype;             ///< Data type (FP32, INT8, INT32, FP16, etc.)
    size_t alignment = 256;     ///< Alignment requirement (default 256 for GPU)
    bool aliasable = false;     ///< Can this buffer be shared with non-overlapping stages?
    bool required = true;       ///< If false, allocation failure is not fatal

    WorkspaceDescriptor() = default;
    
    WorkspaceDescriptor(std::string name_, size_t size_, DataType dtype_,
                           size_t align_ = 256, bool aliasable_ = false, 
                           bool required_ = true)
        : name(std::move(name_))
        , size_bytes(size_)
        , dtype(dtype_)
        , alignment(align_)
        , aliasable(aliasable_)
        , required(required_)
    {}
    
    /**
     * @brief Get element count based on dtype
     */
    size_t elementCount() const
    {
        return size_bytes / dataTypeSize(dtype);
    }
};

/**
 * @brief Collection of workspace buffer requirements for a stage
 * 
 * Stages populate this with their buffer needs. The graph collects these
 * from all stages and passes them to the workspace manager for allocation.
 */
struct WorkspaceRequirements
{
    std::vector<WorkspaceDescriptor> buffers;

    /**
     * @brief Add a buffer requirement
     * 
     * @param name Stage-internal buffer name
     * @param size_bytes Buffer size in bytes
     * @param dtype Data type
     * @param aliasable Can be shared with non-overlapping stages
     */
    void add(const std::string& name, size_t size_bytes, DataType dtype,
             bool aliasable = false)
    {
        buffers.emplace_back(name, size_bytes, dtype, 256, aliasable, true);
    }
    
    /**
     * @brief Add a buffer with explicit element count
     * 
     * @param name Stage-internal buffer name
     * @param count Number of elements
     * @param dtype Data type (determines element size)
     * @param aliasable Can be shared with non-overlapping stages
     */
    void addElements(const std::string& name, size_t count, DataType dtype,
                     bool aliasable = false)
    {
        size_t size = count * dataTypeSize(dtype);
        buffers.emplace_back(name, size, dtype, 256, aliasable, true);
    }
    
    /**
     * @brief Add an optional buffer (allocation failure is not fatal)
     */
    void addOptional(const std::string& name, size_t size_bytes, DataType dtype)
    {
        buffers.emplace_back(name, size_bytes, dtype, 256, false, false);
    }

    /**
     * @brief Calculate total bytes needed for all buffers
     */
    size_t totalBytes() const
    {
        size_t total = 0;
        for (const auto& buf : buffers)
        {
            size_t aligned = (buf.size_bytes + buf.alignment - 1) & ~(buf.alignment - 1);
            total += aligned;
        }
        return total;
    }
    
    /**
     * @brief Calculate total bytes for aliasable buffers only
     */
    size_t aliasableBytes() const
    {
        size_t total = 0;
        for (const auto& buf : buffers)
        {
            if (buf.aliasable)
            {
                size_t aligned = (buf.size_bytes + buf.alignment - 1) & ~(buf.alignment - 1);
                total += aligned;
            }
        }
        return total;
    }

    bool empty() const { return buffers.empty(); }
    size_t count() const { return buffers.size(); }
};

/**
 * @brief Helper to get size in bytes for a DataType
 */
inline size_t dataTypeSize(DataType dtype)
{
    switch (dtype)
    {
        case DataType::FP32:  return 4;
        case DataType::FP16:  return 2;
        case DataType::BF16:  return 2;
        case DataType::INT32: return 4;
        case DataType::INT8:  return 1;
        case DataType::INT16: return 2;
        default: return 1;
    }
}

} // namespace llaminar2
```

### 0.2 IComputeStage.h Extension

Add the virtual method to the base class:

```cpp
// In IComputeStage.h, add include:
#include "execution/local_execution/device/WorkspaceDescriptor.h"

// In the IComputeStage class, add:
class IComputeStage {
public:
    // ... existing methods ...
    
    /**
     * @brief Get GPU workspace requirements for this stage
     * 
     * Override in stages that need GPU workspace buffers. The graph builder
     * collects these from all stages during construction.
     * 
     * Buffer sizes should be computed based on stage parameters (M, K, N,
     * seq_len, etc.) - not hardcoded.
     * 
     * @return GPU workspace buffer requirements (empty by default)
     */
    virtual WorkspaceRequirements getWorkspaceRequirements() const
    {
        return {};
    }
    
    /**
     * @brief Receive allocated workspace buffers
     * 
     * Called by graph builder after allocation. Maps buffer names to pointers.
     * Stages store these pointers for use during execute().
     * 
     * @param buffers Map of buffer name -> device pointer
     */
    virtual void setGpuWorkspaceBuffers(
        const std::unordered_map<std::string, void*>& buffers)
    {
        // Default: no-op for stages without GPU workspace
    }
    
    // ... existing methods ...
};
```

### 0.3 Example: GEMMStage Declaration

Stage knows its own sizing formulas:

```cpp
class GEMMStage : public IComputeStage {
public:
    WorkspaceRequirements getWorkspaceRequirements() const override
    {
        WorkspaceRequirements reqs;
        
        // Only need GPU workspace when executing on GPU
        if (!isGpuDevice(params_.device)) return reqs;
        
        const size_t M = params_.M;
        const size_t K = params_.K;
        const size_t N = params_.N;
        
        // INT8 quantization path
        reqs.addElements("quant_a", M * K, DataType::INT8, /*aliasable=*/true);
        reqs.addElements("scales_a", M, DataType::FP32, /*aliasable=*/true);
        reqs.addElements("acc", M * N, DataType::INT32, /*aliasable=*/true);
        
        // FP16 slab path (if using adaptive FP16 on gfx906)
        if (useFp16SlabPath())
        {
            auto cfg = getSlabConfig();  // Stage computes its own slab sizes
            reqs.addElements("slab_a", cfg.slab_m * cfg.slab_k, DataType::FP16);
            reqs.addElements("slab_b", cfg.slab_k * cfg.slab_n, DataType::FP16);
            reqs.addElements("slab_c", cfg.slab_m * cfg.slab_n, DataType::FP16);
        }
        
        return reqs;
    }
    
    void setGpuWorkspaceBuffers(
        const std::unordered_map<std::string, void*>& buffers) override
    {
        // Store pointers for use in execute()
        if (auto it = buffers.find("quant_a"); it != buffers.end())
            d_quant_a_ = static_cast<int8_t*>(it->second);
        if (auto it = buffers.find("scales_a"); it != buffers.end())
            d_scales_a_ = static_cast<float*>(it->second);
        if (auto it = buffers.find("acc"); it != buffers.end())
            d_acc_ = static_cast<int32_t*>(it->second);
        // ... etc
    }
    
private:
    int8_t* d_quant_a_ = nullptr;
    float* d_scales_a_ = nullptr;
    int32_t* d_acc_ = nullptr;
    // ...
};
```

### 0.4 Example: AttentionComputeStage Declaration

```cpp
class AttentionComputeStage : public IComputeStage {
public:
    WorkspaceRequirements getWorkspaceRequirements() const override
    {
        WorkspaceRequirements reqs;
        
        if (!isGpuDevice(params_.device)) return reqs;
        
        const size_t n_heads = params_.n_heads;
        const size_t max_seq = params_.max_seq_len;
        const size_t head_dim = params_.head_dim;
        
        // Score matrices: n_heads × max_seq × max_seq
        reqs.addElements("scores", n_heads * max_seq * max_seq, DataType::FP32);
        
        // Softmax intermediate (can be aliased - only needed during softmax)
        reqs.addElements("softmax_tmp", n_heads * max_seq, DataType::FP32, 
                         /*aliasable=*/true);
        
        // Context accumulation
        reqs.addElements("context", max_seq * head_dim * n_heads, DataType::FP32);
        
        return reqs;
    }
    
    void setGpuWorkspaceBuffers(
        const std::unordered_map<std::string, void*>& buffers) override
    {
        if (auto it = buffers.find("scores"); it != buffers.end())
            d_scores_ = static_cast<float*>(it->second);
        if (auto it = buffers.find("softmax_tmp"); it != buffers.end())
            d_softmax_tmp_ = static_cast<float*>(it->second);
        if (auto it = buffers.find("context"); it != buffers.end())
            d_context_ = static_cast<float*>(it->second);
    }
};
```

### 0.5 Example: AllreduceStage Declaration

```cpp
class AllreduceStage : public IComputeStage {
public:
    WorkspaceRequirements getWorkspaceRequirements() const override
    {
        WorkspaceRequirements reqs;
        
        if (!isGpuDevice(params_.device)) return reqs;
        
        // Staging buffer size = input tensor size
        // Stage knows its own sizing: tensor_elements from params
        reqs.addElements("staging", params_.tensor_elements, DataType::FP32);
        
        return reqs;
    }
};
```

### 0.6 Graph Builder Integration

The graph builder is completely buffer-agnostic:

```cpp
class GraphBuilder {
public:
    void build()
    {
        // Phase 1: Collect requirements from all stages
        for (auto& stage : stages_)
        {
            // CPU buffers (existing)
            auto cpu_reqs = stage->getBufferRequirements();
            buffer_manager_.registerCpuBuffers(stage->name(), cpu_reqs);
            
            // GPU workspace (new) - graph doesn't interpret these
            auto gpu_reqs = stage->getWorkspaceRequirements();
            if (!gpu_reqs.empty())
            {
                workspace_manager_.registerBuffers(stage->name(), gpu_reqs);
            }
        }
        
        // Phase 2: Allocate all buffers
        buffer_manager_.allocate();          // CPU buffers
        workspace_manager_.allocate();       // GPU workspace
        
        // Phase 3: Distribute allocated buffers back to stages
        for (auto& stage : stages_)
        {
            // CPU buffers (existing)
            stage->setBuffers(buffer_manager_.getBuffersFor(stage->name()));
            
            // GPU workspace (new)
            auto gpu_buffers = workspace_manager_.getBuffersFor(stage->name());
            if (!gpu_buffers.empty())
            {
                stage->setGpuWorkspaceBuffers(gpu_buffers);
            }
        }
    }
};
```

### 0.7 Phase 0 Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/v2/execution/WorkspaceDescriptor.h` | CREATE | Generic descriptor (no semantic enums) |
| `src/v2/execution/compute_stages/IComputeStage.h` | MODIFY | Add `getWorkspaceRequirements()` + `setGpuWorkspaceBuffers()` |
| `src/v2/execution/DeviceWorkspaceManager.h` | CREATE | Collects requirements, allocates, tracks (Phase 1 detail) |

### 0.8 Phase 0 Acceptance Criteria

- [ ] `WorkspaceDescriptor` has NO semantic/category fields
- [ ] Stages can declare arbitrary buffers with name/size/type
- [ ] Graph builder is completely buffer-agnostic
- [ ] Buffer sizing formulas live in stages, not graph
- [ ] Existing tests pass (no behavior change)

---

## Phase 1: GPU Workspace Manager

Phase 1 implements the centralized workspace manager. Note that this builds on the **generic** descriptor from Phase 0 - no semantic factory methods here.

### 1.1 DeviceWorkspaceManager.h

```cpp
/**
 * @file DeviceWorkspaceManager.h
 * @brief Per-device GPU workspace buffer management
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "WorkspaceDescriptor.h"
#include "../utils/DeviceId.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace llaminar2
{

// Forward declarations
class IBackend;

/**
 * @brief Manages GPU workspace buffers for a single device
 *
 * Allocates workspace buffers upfront within a memory budget.
 * Buffers are reused across kernel calls - no hot-path allocations.
 * 
 * DESIGN: This class is buffer-agnostic. It allocates bytes, not semantics.
 * It doesn't know or care what buffers are used for.
 *
 * ## Usage
 *
 * @code
 * // Create manager with 256MB budget
 * DeviceWorkspaceManager workspace(DeviceId::cuda(0), 256 * 1024 * 1024);
 *
 * // Allocate from stage-provided requirements
 * auto reqs = stage->getWorkspaceRequirements();
 * workspace.allocate(reqs);
 *
 * // Get buffer in kernel (stage knows what the buffer is called)
 * void* buf = workspace.getBuffer("quant_a");
 * @endcode
 */
class DeviceWorkspaceManager
{
public:
    /**
     * @brief Construct workspace manager for a device
     * @param device Target GPU device
     * @param budget_bytes Maximum GPU memory to use for workspace (0 = unlimited)
     */
    DeviceWorkspaceManager(DeviceId device, size_t budget_bytes = 0);
    
    /**
     * @brief Destructor - releases all buffers
     */
    ~DeviceWorkspaceManager();

    // Non-copyable, movable
    DeviceWorkspaceManager(const DeviceWorkspaceManager&) = delete;
    DeviceWorkspaceManager& operator=(const DeviceWorkspaceManager&) = delete;
    DeviceWorkspaceManager(DeviceWorkspaceManager&&) noexcept;
    DeviceWorkspaceManager& operator=(DeviceWorkspaceManager&&) noexcept;

    // =========================================================================
    // Allocation
    // =========================================================================

    /**
     * @brief Allocate workspace from requirements
     * @param requirements Workspace requirements to satisfy
     * @return true if all required buffers were allocated
     */
    bool allocate(const WorkspaceRequirements& requirements);

    /**
     * @brief Allocate a single named buffer
     * @param name Buffer identifier
     * @param size_bytes Buffer size
     * @param alignment Alignment requirement
     * @return true if allocation succeeded
     */
    bool allocateBuffer(const std::string& name, size_t size_bytes, size_t alignment = 256);

    /**
     * @brief Release all workspace buffers
     */
    void release();

    /**
     * @brief Release a single buffer
     * @param name Buffer to release
     */
    void releaseBuffer(const std::string& name);

    // =========================================================================
    // Buffer Access
    // =========================================================================

    /**
     * @brief Get a workspace buffer by name
     * @param name Buffer name from requirements
     * @return Device pointer (nullptr if not found)
     */
    void* getBuffer(const std::string& name);

    /**
     * @brief Get a workspace buffer by name (const version)
     */
    const void* getBuffer(const std::string& name) const;

    /**
     * @brief Get buffer size
     * @param name Buffer name
     * @return Size in bytes (0 if not found)
     */
    size_t getBufferSize(const std::string& name) const;

    /**
     * @brief Check if a buffer exists
     */
    bool hasBuffer(const std::string& name) const;

    // =========================================================================
    // Query
    // =========================================================================

    DeviceId device() const { return device_; }
    size_t budgetBytes() const { return budget_bytes_; }
    size_t allocatedBytes() const { return allocated_bytes_; }
    size_t availableBytes() const { return budget_bytes_ - allocated_bytes_; }
    
    /**
     * @brief Get list of allocated buffer names
     */
    std::vector<std::string> bufferNames() const;

    /**
     * @brief Log workspace summary
     */
    void logSummary() const;

private:
    DeviceId device_;
    size_t budget_bytes_;
    size_t allocated_bytes_ = 0;
    IBackend* backend_ = nullptr;  // Not owned

    struct BufferInfo
    {
        void* ptr = nullptr;
        size_t size = 0;
        size_t alignment = 256;
    };
    std::unordered_map<std::string, BufferInfo> buffers_;

    // Get backend for device
    IBackend* getBackend();
};

} // namespace llaminar2
```

### 1.3 DeviceWorkspaceManager.cpp

```cpp
/**
 * @file DeviceWorkspaceManager.cpp
 * @brief GPU workspace manager implementation
 * @author David Sanftenberg
 */

#include "DeviceWorkspaceManager.h"
#include "../backends/IBackend.h"
#include "../backends/BackendRegistry.h"
#include "../utils/Logger.h"
#include <algorithm>

namespace llaminar2
{

// =========================================================================
// Construction / Destruction
// =========================================================================

DeviceWorkspaceManager::DeviceWorkspaceManager(DeviceId device, size_t budget_bytes)
    : device_(device)
    , budget_bytes_(budget_bytes)
{
    backend_ = getBackend();
    if (!backend_)
    {
        LOG_WARN("[DeviceWorkspaceManager] No backend available for device " 
                 << device_.toString());
    }
}

DeviceWorkspaceManager::~DeviceWorkspaceManager()
{
    release();
}

DeviceWorkspaceManager::DeviceWorkspaceManager(DeviceWorkspaceManager&& other) noexcept
    : device_(other.device_)
    , budget_bytes_(other.budget_bytes_)
    , allocated_bytes_(other.allocated_bytes_)
    , backend_(other.backend_)
    , buffers_(std::move(other.buffers_))
{
    other.allocated_bytes_ = 0;
    other.backend_ = nullptr;
}

DeviceWorkspaceManager& DeviceWorkspaceManager::operator=(DeviceWorkspaceManager&& other) noexcept
{
    if (this != &other)
    {
        release();
        device_ = other.device_;
        budget_bytes_ = other.budget_bytes_;
        allocated_bytes_ = other.allocated_bytes_;
        backend_ = other.backend_;
        buffers_ = std::move(other.buffers_);
        other.allocated_bytes_ = 0;
        other.backend_ = nullptr;
    }
    return *this;
}

// =========================================================================
// Allocation
// =========================================================================

bool DeviceWorkspaceManager::allocate(const WorkspaceRequirements& requirements)
{
    if (!backend_)
    {
        LOG_ERROR("[DeviceWorkspaceManager] Cannot allocate: no backend");
        return false;
    }

    // Check total requirements against budget
    size_t total_required = requirements.totalBytes();
    if (budget_bytes_ > 0 && total_required > budget_bytes_)
    {
        LOG_ERROR("[DeviceWorkspaceManager] Requirements (" << total_required / (1024*1024) 
                  << " MB) exceed budget (" << budget_bytes_ / (1024*1024) << " MB)");
        return false;
    }

    // Allocate each buffer
    for (const auto& desc : requirements.buffers)
    {
        if (!allocateBuffer(desc.name, desc.size_bytes, desc.alignment))
        {
            if (desc.required)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Failed to allocate required buffer: " 
                          << desc.name);
                return false;
            }
            else
            {
                LOG_WARN("[DeviceWorkspaceManager] Failed to allocate optional buffer: " 
                         << desc.name);
            }
        }
    }

    LOG_INFO("[DeviceWorkspaceManager] Allocated " << buffers_.size() << " buffers, "
             << allocated_bytes_ / (1024*1024) << " MB total on " << device_.toString());
    return true;
}

bool DeviceWorkspaceManager::allocateBuffer(const std::string& name, size_t size_bytes, 
                                          size_t alignment)
{
    if (!backend_)
    {
        return false;
    }

    // Check if buffer already exists
    if (buffers_.count(name))
    {
        LOG_WARN("[DeviceWorkspaceManager] Buffer '" << name << "' already exists, skipping");
        return true;
    }

    // Calculate aligned size
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);

    // Check budget
    if (budget_bytes_ > 0 && (allocated_bytes_ + aligned_size) > budget_bytes_)
    {
        LOG_ERROR("[DeviceWorkspaceManager] Allocation of '" << name << "' (" 
                  << aligned_size / (1024*1024) << " MB) would exceed budget");
        return false;
    }

    // Allocate via backend
    int device_idx = device_.is_cuda() ? device_.cuda_index() : device_.rocm_index();
    void* ptr = backend_->allocate(aligned_size, device_idx);
    
    if (!ptr)
    {
        LOG_ERROR("[DeviceWorkspaceManager] Backend allocation failed for '" << name 
                  << "' (" << aligned_size / (1024*1024) << " MB)");
        return false;
    }

    // Record allocation
    BufferInfo info;
    info.ptr = ptr;
    info.size = aligned_size;
    info.alignment = alignment;
    buffers_[name] = info;
    allocated_bytes_ += aligned_size;

    LOG_DEBUG("[DeviceWorkspaceManager] Allocated '" << name << "': " 
              << aligned_size / 1024 << " KB at " << ptr);
    return true;
}

void DeviceWorkspaceManager::release()
{
    if (!backend_)
    {
        buffers_.clear();
        allocated_bytes_ = 0;
        return;
    }

    int device_idx = device_.is_cuda() ? device_.cuda_index() : device_.rocm_index();
    
    for (auto& [name, info] : buffers_)
    {
        if (info.ptr)
        {
            backend_->free(info.ptr, device_idx);
            LOG_DEBUG("[DeviceWorkspaceManager] Released '" << name << "'");
        }
    }
    
    buffers_.clear();
    allocated_bytes_ = 0;
    LOG_DEBUG("[DeviceWorkspaceManager] Released all workspace buffers");
}

void DeviceWorkspaceManager::releaseBuffer(const std::string& name)
{
    auto it = buffers_.find(name);
    if (it == buffers_.end())
    {
        return;
    }

    if (backend_ && it->second.ptr)
    {
        int device_idx = device_.is_cuda() ? device_.cuda_index() : device_.rocm_index();
        backend_->free(it->second.ptr, device_idx);
        allocated_bytes_ -= it->second.size;
    }
    
    buffers_.erase(it);
}

// =========================================================================
// Buffer Access
// =========================================================================

void* DeviceWorkspaceManager::getBuffer(const std::string& name)
{
    auto it = buffers_.find(name);
    if (it != buffers_.end())
    {
        return it->second.ptr;
    }
    return nullptr;
}

const void* DeviceWorkspaceManager::getBuffer(const std::string& name) const
{
    auto it = buffers_.find(name);
    if (it != buffers_.end())
    {
        return it->second.ptr;
    }
    return nullptr;
}

size_t DeviceWorkspaceManager::getBufferSize(const std::string& name) const
{
    auto it = buffers_.find(name);
    if (it != buffers_.end())
    {
        return it->second.size;
    }
    return 0;
}

bool DeviceWorkspaceManager::hasBuffer(const std::string& name) const
{
    return buffers_.count(name) > 0;
}

std::vector<std::string> DeviceWorkspaceManager::bufferNames() const
{
    std::vector<std::string> names;
    names.reserve(buffers_.size());
    for (const auto& [name, info] : buffers_)
    {
        names.push_back(name);
    }
    return names;
}

void DeviceWorkspaceManager::logSummary() const
{
    LOG_INFO("[DeviceWorkspaceManager] Workspace summary for " << device_.toString() << ":");
    LOG_INFO("  Budget: " << budget_bytes_ / (1024*1024) << " MB");
    LOG_INFO("  Allocated: " << allocated_bytes_ / (1024*1024) << " MB");
    LOG_INFO("  Available: " << availableBytes() / (1024*1024) << " MB");
    LOG_INFO("  Buffers: " << buffers_.size());
    
    for (const auto& [name, info] : buffers_)
    {
        LOG_INFO("    " << name << ": " << info.size / 1024 << " KB");
    }
}

// =========================================================================
// Private
// =========================================================================

IBackend* DeviceWorkspaceManager::getBackend()
{
    if (device_.is_cuda())
    {
        return BackendRegistry::getCUDABackend();
    }
    else if (device_.is_rocm())
    {
        return BackendRegistry::getROCmBackend();
    }
    return nullptr;
}

} // namespace llaminar2
```

### 1.4 GraphBufferManager Extensions

Add to `GraphBufferManager.h`:

```cpp
// In private section:
    std::unordered_map<DeviceId, std::unique_ptr<DeviceWorkspaceManager>, DeviceIdHash> device_workspaces_;
    std::unordered_map<DeviceId, size_t, DeviceIdHash> device_workspace_budgets_;

// In public section:
    /**
     * @brief Set GPU workspace budget for a device
     * @param device Target device
     * @param budget_bytes Maximum workspace memory (0 = auto-detect)
     */
    void setGpuWorkspaceBudget(DeviceId device, size_t budget_bytes);

    /**
     * @brief Get workspace manager for a device
     * @return Workspace manager (creates if needed)
     */
    DeviceWorkspaceManager* getDeviceWorkspace(DeviceId device);

    /**
     * @brief Allocate GPU workspace for devices in graph
     * @param requirements Per-device requirements
     * @return true if allocation succeeded
     */
    bool allocateDeviceWorkspace(
        const std::unordered_map<DeviceId, WorkspaceRequirements, DeviceIdHash>& requirements);

    /**
     * @brief Release all GPU workspace
     */
    void releaseDeviceWorkspace();
```

---

## Phase 2: Kernel Integration

### 2.1 IWorkspaceConsumer.h

```cpp
/**
 * @file IWorkspaceConsumer.h
 * @brief Interface for kernels that consume GPU workspace
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/WorkspaceDescriptor.h"

namespace llaminar2
{

// Forward declaration
class DeviceWorkspaceManager;

/**
 * @brief Interface for kernels that consume GPU workspace
 *
 * Kernels implementing this interface declare their workspace
 * requirements and receive allocated buffers before execution.
 *
 * ## Usage Pattern
 *
 * @code
 * class MyGpuKernel : public IWorkspaceConsumer {
 * public:
 *     WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override {
 *         WorkspaceRequirements reqs;
 *         reqs.addBuffer("temp", m * k * sizeof(float));
 *         return reqs;
 *     }
 *
 *     void bindWorkspace(DeviceWorkspaceManager* workspace) override {
 *         workspace_ = workspace;
 *     }
 *
 *     void compute() {
 *         if (workspace_) {
 *             float* temp = static_cast<float*>(workspace_->getBuffer("temp"));
 *             // Use workspace buffer
 *         }
 *     }
 * };
 * @endcode
 */
class IWorkspaceConsumer
{
public:
    virtual ~IWorkspaceConsumer() = default;

    /**
     * @brief Get workspace requirements for given GEMM dimensions
     * @param m Number of activation rows
     * @param n Number of output columns
     * @param k Inner dimension
     * @return Workspace buffer requirements
     */
    virtual WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const = 0;

    /**
     * @brief Bind workspace manager to this kernel
     *
     * Called before kernel execution to provide access to allocated workspace.
     * The workspace pointer is NOT owned by the kernel.
     *
     * @param workspace Workspace manager with allocated buffers (may be nullptr to unbind)
     */
    virtual void bindWorkspace(DeviceWorkspaceManager* workspace) = 0;

    /**
     * @brief Check if workspace is currently bound
     * @return true if workspace has been bound
     */
    virtual bool hasWorkspace() const = 0;

    /**
     * @brief Get the bound workspace manager
     * @return Workspace manager (nullptr if not bound)
     */
    virtual DeviceWorkspaceManager* getWorkspace() const = 0;
};

// NOTE: No GemmWorkspaceBuffers namespace here!
// Buffer names are defined by each kernel/stage privately. The graph doesn't
// know or care what buffers are called - it just allocates bytes.

} // namespace llaminar2
```

### 2.2 ROCmQuantisedGemmKernel Modifications

**Header additions** (`ROCmQuantisedGemmKernel.h`):

```cpp
// Add include
#include "../interfaces/IWorkspaceConsumer.h"

// Change class declaration
class ROCmQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer
{
    // ... existing declarations ...

    // Add IWorkspaceConsumer implementation
    WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override;
    void bindWorkspace(DeviceWorkspaceManager* workspace) override;
    bool hasWorkspace() const override;
    DeviceWorkspaceManager* getWorkspace() const override;

private:
    DeviceWorkspaceManager* workspace_ = nullptr;  // NOT owned
    
    // Kernel-private buffer names (NOT exposed to graph)
    static constexpr const char* kQuantA = "quant_a";
    static constexpr const char* kScalesA = "scales_a";
    static constexpr const char* kAccInt32 = "acc_int32";
    static constexpr const char* kSlabA = "slab_a";
    static constexpr const char* kSlabB = "slab_b";
    static constexpr const char* kSlabC = "slab_c";
};
```

**Implementation additions** (`ROCmQuantisedGemmKernel.cpp`):

```cpp
// IWorkspaceConsumer implementation
// NOTE: Buffer names are kernel-private. The graph just sees opaque strings.
WorkspaceRequirements ROCmQuantisedGemmKernel::getWorkspaceRequirements(int m, int n, int k) const
{
    WorkspaceRequirements reqs;

    // Core quantization buffers (always needed)
    // Kernel uses its private constants; graph sees opaque strings
    reqs.addBuffer(kQuantA, static_cast<size_t>(m) * k * sizeof(int8_t));
    reqs.addBuffer(kScalesA, static_cast<size_t>(m) * sizeof(float));
    reqs.addBuffer(kAccInt32, static_cast<size_t>(m) * n * sizeof(int32_t));

    // FP16 slab buffers (for slab GEMM path)
    const auto& env = debugEnv().gpu_workspace;
    reqs.addBuffer(kSlabA, static_cast<size_t>(env.slab_m) * env.slab_k * sizeof(__half), 256, false);
    reqs.addBuffer(kSlabB, static_cast<size_t>(env.slab_k) * env.slab_n * sizeof(__half), 256, false);
    reqs.addBuffer(kSlabC, static_cast<size_t>(env.slab_m) * env.slab_n * sizeof(__half), 256, false);

    return reqs;
}

void ROCmQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager* workspace)
{
    workspace_ = workspace;
    if (workspace_)
    {
        LOG_DEBUG("[ROCmQuantisedGemmKernel] Workspace bound with " 
                  << workspace_->allocatedBytes() / (1024*1024) << " MB");
    }
}

bool ROCmQuantisedGemmKernel::hasWorkspace() const
{
    return workspace_ != nullptr;
}

DeviceWorkspaceManager* ROCmQuantisedGemmKernel::getWorkspace() const
{
    return workspace_;
}

// Modify multiply_tensor() to use workspace when available
bool ROCmQuantisedGemmKernel::multiply_tensor(/* ... */)
{
    // ... existing code ...

    // Get workspace buffers if available, otherwise use legacy Impl buffers
    int8_t* d_A_int8 = nullptr;
    float* d_scales_A = nullptr;
    int32_t* d_C_int32 = nullptr;

    if (workspace_)
    {
        // Use kernel-private buffer name constants
        d_A_int8 = static_cast<int8_t*>(workspace_->getBuffer(kQuantA));
        d_scales_A = static_cast<float*>(workspace_->getBuffer(kScalesA));
        d_C_int32 = static_cast<int32_t*>(workspace_->getBuffer(kAccInt32));
    }

    // Fallback to legacy if workspace buffers not available
    if (!d_A_int8 || !d_scales_A || !d_C_int32)
    {
        if (workspace_)
        {
            LOG_WARN("[ROCmQuantisedGemmKernel] Workspace missing buffers, falling back to legacy");
        }
        ensureWorkBuffers(m);
        d_A_int8 = impl_->d_A_int8;
        d_scales_A = impl_->d_scales_A;
        d_C_int32 = impl_->d_C_int32;
    }

    // ... rest of implementation using these buffers ...
}
```

---

## Phase 3: Slab-Based FP16 GEMM

### 3.1 SlabGemmConfig.h

```cpp
/**
 * @file SlabGemmConfig.h
 * @brief Configuration for slab-based GEMM execution
 * @author David Sanftenberg
 * @date January 2026
 * 
 * NOTE: This is a helper struct for STAGES that need slab GEMM.
 * The graph doesn't use this - stages call workspaceRequirements()
 * which returns a generic WorkspaceRequirements.
 */

#pragma once

#include "../execution/WorkspaceDescriptor.h"
#include <cstddef>
#include <algorithm>

namespace llaminar2
{

/**
 * @brief Configuration for slab-based GEMM execution
 *
 * Controls how large GEMM operations are chunked into smaller slabs
 * to fit within a memory budget.
 */
struct SlabGemmConfig
{
    int slab_m = 256;   ///< Rows per slab
    int slab_n = 256;   ///< Columns per slab
    int slab_k = 512;   ///< Inner dimension per slab

    /**
     * @brief Calculate workspace bytes for FP16 slab buffers
     */
    size_t workspaceBytes() const
    {
        constexpr size_t FP16_SIZE = 2;
        size_t slab_a = static_cast<size_t>(slab_m) * slab_k * FP16_SIZE;
        size_t slab_b = static_cast<size_t>(slab_k) * slab_n * FP16_SIZE;
        size_t slab_c = static_cast<size_t>(slab_m) * slab_n * FP16_SIZE;
        return slab_a + slab_b + slab_c;
    }

    /**
     * @brief Calculate number of slab iterations for full GEMM
     */
    int estimateIterations(int m, int n, int k) const
    {
        int m_iters = (m + slab_m - 1) / slab_m;
        int n_iters = (n + slab_n - 1) / slab_n;
        int k_iters = (k + slab_k - 1) / slab_k;
        return m_iters * n_iters * k_iters;
    }

    /**
     * @brief Create config from memory budget
     *
     * Determines optimal slab sizes given available memory.
     *
     * @param budget_bytes Available workspace memory
     * @param m Full GEMM M dimension
     * @param n Full GEMM N dimension
     * @param k Full GEMM K dimension
     */
    static SlabGemmConfig fromBudget(size_t budget_bytes, int m, int n, int k)
    {
        SlabGemmConfig config;
        constexpr size_t FP16_SIZE = 2;
        
        // Start with defaults
        config.slab_m = 256;
        config.slab_n = 256;
        config.slab_k = 512;

        // Check if defaults fit
        if (config.workspaceBytes() <= budget_bytes)
        {
            // Try to increase slab sizes while staying in budget
            // Prioritize slab_k for better GEMM efficiency
            while (config.workspaceBytes() <= budget_bytes)
            {
                size_t current = config.workspaceBytes();
                
                // Try doubling k first
                config.slab_k *= 2;
                if (config.workspaceBytes() > budget_bytes || config.slab_k > k)
                {
                    config.slab_k /= 2;
                    break;
                }
            }
        }
        else
        {
            // Shrink slab sizes to fit budget
            while (config.workspaceBytes() > budget_bytes)
            {
                // Reduce the largest dimension first
                if (config.slab_k >= config.slab_m && config.slab_k >= config.slab_n)
                {
                    config.slab_k /= 2;
                }
                else if (config.slab_n >= config.slab_m)
                {
                    config.slab_n /= 2;
                }
                else
                {
                    config.slab_m /= 2;
                }

                // Enforce minimums
                if (config.slab_m < 32 || config.slab_n < 32 || config.slab_k < 32)
                {
                    break;
                }
            }
        }

        // Clamp to actual dimensions
        config.slab_m = std::min(config.slab_m, m);
        config.slab_n = std::min(config.slab_n, n);
        config.slab_k = std::min(config.slab_k, k);

        return config;
    }

    /**
     * @brief Get workspace requirements for this slab config
     * 
     * Returns generic requirements - caller (stage/kernel) uses this
     * to declare buffers. Buffer names are opaque to the graph.
     * 
     * @param slab_a_name Stage-chosen name for slab A buffer
     * @param slab_b_name Stage-chosen name for slab B buffer
     * @param slab_c_name Stage-chosen name for slab C buffer
     */
    WorkspaceRequirements workspaceRequirements(
        const std::string& slab_a_name = "slab_a",
        const std::string& slab_b_name = "slab_b",
        const std::string& slab_c_name = "slab_c") const
    {
        WorkspaceRequirements reqs;
        constexpr size_t FP16_SIZE = 2;
        
        reqs.addBuffer(slab_a_name, static_cast<size_t>(slab_m) * slab_k * FP16_SIZE);
        reqs.addBuffer(slab_b_name, static_cast<size_t>(slab_k) * slab_n * FP16_SIZE);
        reqs.addBuffer(slab_c_name, static_cast<size_t>(slab_m) * slab_n * FP16_SIZE);
        
        return reqs;
    }
};

} // namespace llaminar2
```

### 3.2 Slab FP16 HIP Kernel

Add to `ROCmQuantisedGemmKernel_FP16.hip`:

```cpp
/**
 * @brief Extract A slab from INT8 to FP16 with scale application
 *
 * Extracts [actual_m × actual_k] sub-matrix from A[M × K] starting at (m_start, k_start)
 * and converts to FP16 with per-row scale application.
 */
__global__ void extractSlabA_int8ToFP16_kernel(
    const int8_t* __restrict__ A_int8,  // [M × K]
    const float* __restrict__ scaleA,    // [M]
    __half* __restrict__ slab_a,         // [slab_m × slab_k]
    int M, int K,
    int m_start, int k_start,
    int actual_m, int actual_k,
    int slab_k)  // stride for output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = actual_m * actual_k;
    
    if (idx < total)
    {
        int local_m = idx / actual_k;
        int local_k = idx % actual_k;
        int global_m = m_start + local_m;
        int global_k = k_start + local_k;
        
        // Read INT8 value
        int8_t val = A_int8[global_m * K + global_k];
        
        // Apply scale and convert to FP16
        float scaled = static_cast<float>(val) * scaleA[global_m];
        slab_a[local_m * slab_k + local_k] = __float2half(scaled);
    }
}

/**
 * @brief Extract B slab from INT8 to FP16 with scale application
 *
 * Extracts [actual_k × actual_n] sub-matrix from B[K × N] starting at (k_start, n_start)
 * and converts to FP16 with per-column scale application.
 */
__global__ void extractSlabB_int8ToFP16_kernel(
    const int8_t* __restrict__ B_int8,  // [K × N]
    const float* __restrict__ scaleB,    // [N]
    __half* __restrict__ slab_b,         // [slab_k × slab_n]
    int K, int N,
    int k_start, int n_start,
    int actual_k, int actual_n,
    int slab_n)  // stride for output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = actual_k * actual_n;
    
    if (idx < total)
    {
        int local_k = idx / actual_n;
        int local_n = idx % actual_n;
        int global_k = k_start + local_k;
        int global_n = n_start + local_n;
        
        // Read INT8 value
        int8_t val = B_int8[global_k * N + global_n];
        
        // Apply scale and convert to FP16
        float scaled = static_cast<float>(val) * scaleB[global_n];
        slab_b[local_k * slab_n + local_n] = __float2half(scaled);
    }
}

/**
 * @brief Accumulate FP16 slab result to FP32 output
 *
 * Adds [actual_m × actual_n] FP16 slab to FP32 output at (m_start, n_start).
 */
__global__ void accumulateSlabC_fp16ToFP32_kernel(
    const __half* __restrict__ slab_c,   // [slab_m × slab_n]
    float* __restrict__ C_fp32,           // [M × N]
    int M, int N,
    int m_start, int n_start,
    int actual_m, int actual_n,
    int slab_n)  // stride for input
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = actual_m * actual_n;
    
    if (idx < total)
    {
        int local_m = idx / actual_n;
        int local_n = idx % actual_n;
        int global_m = m_start + local_m;
        int global_n = n_start + local_n;
        
        // Read FP16 value and accumulate to FP32
        float val = __half2float(slab_c[local_m * slab_n + local_n]);
        C_fp32[global_m * N + global_n] += val;
    }
}

/**
 * @brief Execute slab-based INT8→FP16 GEMM with fixed workspace
 */
extern "C" bool rocmQuantGemm_executeSlabFP16(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8,
    float* d_C_fp32,
    const float* d_scaleA,
    const float* d_scaleB,
    int M, int N, int K,
    void* slab_a_fp16,
    void* slab_b_fp16,
    void* slab_c_fp16,
    int slab_m, int slab_n, int slab_k,
    hipblasHandle_t handle,
    hipStream_t stream)
{
    __half* d_slab_a = static_cast<__half*>(slab_a_fp16);
    __half* d_slab_b = static_cast<__half*>(slab_b_fp16);
    __half* d_slab_c = static_cast<__half*>(slab_c_fp16);

    // Zero-initialize output
    hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), stream);

    constexpr int BLOCK_SIZE = 256;
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);

    // Loop order: K (outer) → N → M (inner) for maximum B slab reuse
    for (int k_start = 0; k_start < K; k_start += slab_k)
    {
        int actual_k = std::min(slab_k, K - k_start);

        for (int n_start = 0; n_start < N; n_start += slab_n)
        {
            int actual_n = std::min(slab_n, N - n_start);

            // Extract B slab
            int b_elems = actual_k * actual_n;
            int b_blocks = (b_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
            extractSlabB_int8ToFP16_kernel<<<b_blocks, BLOCK_SIZE, 0, stream>>>(
                d_B_int8, d_scaleB, d_slab_b,
                K, N, k_start, n_start, actual_k, actual_n, slab_n);

            for (int m_start = 0; m_start < M; m_start += slab_m)
            {
                int actual_m = std::min(slab_m, M - m_start);

                // Extract A slab
                int a_elems = actual_m * actual_k;
                int a_blocks = (a_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                extractSlabA_int8ToFP16_kernel<<<a_blocks, BLOCK_SIZE, 0, stream>>>(
                    d_A_int8, d_scaleA, d_slab_a,
                    M, K, m_start, k_start, actual_m, actual_k, slab_k);

                // FP16 GEMM: slab_c = slab_a @ slab_b
                // Note: hipBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
                hipblasHgemm(handle,
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    actual_n, actual_m, actual_k,
                    &alpha_h,
                    d_slab_b, slab_n,  // B is [actual_k × actual_n], stride slab_n
                    d_slab_a, slab_k,  // A is [actual_m × actual_k], stride slab_k
                    &beta_h,
                    d_slab_c, slab_n); // C is [actual_m × actual_n], stride slab_n

                // Accumulate to FP32 output
                int c_elems = actual_m * actual_n;
                int c_blocks = (c_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                accumulateSlabC_fp16ToFP32_kernel<<<c_blocks, BLOCK_SIZE, 0, stream>>>(
                    d_slab_c, d_C_fp32,
                    M, N, m_start, n_start, actual_m, actual_n, slab_n);
            }
        }
    }

    hipStreamSynchronize(stream);
    return true;
}
```

---

## Phase 4: Memory Budget Configuration

### 4.1 GpuMemoryConfig.h

```cpp
/**
 * @file GpuMemoryConfig.h
 * @brief GPU memory budget and slab configuration
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../utils/DeviceId.h"
#include <cstddef>
#include <string>

namespace llaminar2
{

class IBackend;

/**
 * @brief GPU memory budget and slab configuration
 */
struct GpuMemoryConfig
{
    size_t workspace_budget_bytes = 0;  ///< 0 = auto (10% of VRAM)
    size_t weight_budget_bytes = 0;     ///< 0 = unlimited
    size_t kv_cache_budget_bytes = 0;   ///< 0 = auto
    
    int slab_m = 256;
    int slab_n = 256;
    int slab_k = 512;
    bool force_slab_gemm = false;

    /**
     * @brief Auto-detect configuration for a device
     */
    static GpuMemoryConfig autoDetect(DeviceId device);

    /**
     * @brief Load from environment variables
     */
    static GpuMemoryConfig fromEnvironment();

    /**
     * @brief Check if slab GEMM should be used
     */
    bool shouldUseSlabGemm(int m, int n, int k) const;

    /**
     * @brief Calculate full FP16 GEMM memory
     */
    size_t fullFp16GemmBytes(int m, int n, int k) const;

    /**
     * @brief Calculate slab workspace memory
     */
    size_t slabWorkspaceBytes() const;

    /**
     * @brief Validate configuration
     */
    std::string validate() const;

    /**
     * @brief Log configuration
     */
    void logConfiguration() const;
};

} // namespace llaminar2
```

### 4.2 DebugEnv.h Additions

Add `GpuWorkspaceConfig` struct:

```cpp
struct GpuWorkspaceConfig
{
    size_t workspace_budget_mb = 0;  ///< 0 = auto-detect 10% VRAM
    int slab_m = 256;
    int slab_n = 256;
    int slab_k = 512;
    bool force_slab_gemm = false;

    GpuWorkspaceConfig() { reload(); }

    void reload()
    {
        workspace_budget_mb = 0;
        slab_m = 256;
        slab_n = 256;
        slab_k = 512;
        force_slab_gemm = false;

        if (const char* env = std::getenv("LLAMINAR_GPU_WORKSPACE_MB"))
            if (int val = std::atoi(env); val >= 0)
                workspace_budget_mb = static_cast<size_t>(val);

        if (const char* env = std::getenv("LLAMINAR_GPU_SLAB_M"))
            if (int val = std::atoi(env); val > 0)
                slab_m = val;

        if (const char* env = std::getenv("LLAMINAR_GPU_SLAB_N"))
            if (int val = std::atoi(env); val > 0)
                slab_n = val;

        if (const char* env = std::getenv("LLAMINAR_GPU_SLAB_K"))
            if (int val = std::atoi(env); val > 0)
                slab_k = val;

        if (const char* env = std::getenv("LLAMINAR_GPU_FORCE_SLAB"))
            force_slab_gemm = (std::atoi(env) != 0);
    }

    size_t workspaceBudgetBytes() const { return workspace_budget_mb * 1024 * 1024; }
    bool isAutoDetect() const { return workspace_budget_mb == 0; }
};
```

Add to `DebugEnv` struct:

```cpp
GpuWorkspaceConfig gpu_workspace;
```

Add to `reload()`:

```cpp
gpu_workspace.reload();
```

---

## Implementation Order

```
Phase 1 (Week 1)
├── WorkspaceDescriptor.h ─────┐
├── DeviceWorkspaceManager.h/cpp ────┼── Can be done in parallel
└── GraphBufferManager extensions ┘

Phase 2 (Week 2, depends on Phase 1)
├── IWorkspaceConsumer.h
├── ROCmQuantisedGemmKernel modifications
└── CUDAQuantisedGemmKernel modifications

Phase 3 (Week 3, depends on Phases 1 & 2)
├── SlabGemmConfig.h/cpp
├── ROCmQuantisedGemmKernel_FP16.hip slab kernels
└── Integration tests

Phase 4 (Week 4, can partially parallel with Phase 3)
├── GpuMemoryConfig.h/cpp
├── DebugEnv.h additions
└── Documentation updates
```

---

## Environment Variables Summary

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_GPU_WORKSPACE_MB` | Workspace budget per GPU (MB) | Auto (10% VRAM) |
| `LLAMINAR_GPU_SLAB_M` | Slab M dimension | 256 |
| `LLAMINAR_GPU_SLAB_N` | Slab N dimension | 256 |
| `LLAMINAR_GPU_SLAB_K` | Slab K dimension | 512 |
| `LLAMINAR_GPU_FORCE_SLAB` | Force slab GEMM | 0 |

---

## Memory Analysis: 7B Model FFN

For `ffn_down` (M=512, K=18944, N=3584):

| Approach | Memory | Iterations |
|----------|--------|------------|
| Full FP16 | 159 MB | 1 |
| Slab (64MB) | 6 MB | 74 |

**Memory reduction: 26.5× at cost of ~2× throughput**
