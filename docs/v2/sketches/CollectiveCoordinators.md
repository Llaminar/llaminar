# Collective Coordinators Design Sketch

**Date**: February 3, 2026  
**Status**: Design Sketch for Phase 3

## Overview

Each collective backend gets a dedicated **coordinator thread** that owns all communication state and serializes operations. This solves the threading mismatch where comms were created on worker threads but used from the caller thread.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Collective Coordinator Layer                          │
├─────────────────┬──────────────────┬────────────────────────────────────┤
│ NCCLCoordinator │ RCCLCoordinator  │      PCIeBARCoordinator            │
│ (CUDA↔CUDA)     │ (ROCm↔ROCm)      │      (CUDA↔ROCm heterogeneous)     │
├─────────────────┴──────────────────┴────────────────────────────────────┤
│                    Device Worker Threads (per-GPU)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ CUDA GPU 0 │  │ CUDA GPU 1 │  │ ROCm GPU 0 │  │ ROCm GPU 1 │        │
│  │ (compute)  │  │ (compute)  │  │ (compute)  │  │ (compute)  │        │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Common Base: ICollectiveCoordinator

```cpp
// src/v2/collective/coordinators/ICollectiveCoordinator.h

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace llaminar2 {

/**
 * @brief Abstract base for collective communication coordinators
 * 
 * Each coordinator owns a dedicated thread that serializes all collective
 * operations. This ensures proper threading semantics for NCCL/RCCL group
 * operations and prevents context corruption.
 * 
 * Thread safety: All public methods are thread-safe (work is queued).
 */
class ICollectiveCoordinator {
public:
    virtual ~ICollectiveCoordinator() = default;
    
    // =========================================================================
    // Lifecycle
    // =========================================================================
    
    virtual bool initialize(const std::vector<int>& device_ordinals) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;
    
    // =========================================================================
    // Synchronization with Device Workers
    // =========================================================================
    
    /**
     * @brief Get completion event for a device after last collective
     * 
     * Device workers call cudaStreamWaitEvent(stream, getCompletionEvent(device))
     * before launching kernels that depend on collective results.
     * 
     * @param device_idx Local device index (0 to num_devices-1)
     * @return Opaque event handle (cudaEvent_t or hipEvent_t)
     */
    virtual void* getCompletionEvent(int device_idx) const = 0;
    
    /**
     * @brief Wait for device worker's stream before collective
     * 
     * Coordinator calls this to ensure all prior device work is done
     * before starting a collective that reads from device buffers.
     * 
     * @param device_idx Local device index
     * @param worker_event Event recorded after device worker's last kernel
     */
    virtual void waitForDeviceEvent(int device_idx, void* worker_event) = 0;
    
    // =========================================================================
    // Submission API
    // =========================================================================
    
    /**
     * @brief Submit work and wait for completion
     */
    template<typename F>
    auto submitAndWait(F&& work) -> decltype(work()) {
        std::promise<decltype(work())> promise;
        auto future = promise.get_future();
        enqueueWork([&promise, work = std::forward<F>(work)]() mutable {
            if constexpr (std::is_void_v<decltype(work())>) {
                work();
                promise.set_value();
            } else {
                promise.set_value(work());
            }
        });
        return future.get();
    }
    
    /**
     * @brief Submit work without waiting (returns future)
     */
    template<typename F>
    auto submitAsync(F&& work) -> std::future<decltype(work())> {
        using ReturnType = decltype(work());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(work));
        auto future = task->get_future();
        enqueueWork([task]() { (*task)(); });
        return future;
    }
    
protected:
    virtual void enqueueWork(std::function<void()> work) = 0;
};

} // namespace llaminar2
```

---

## NCCLCoordinator

```cpp
// src/v2/collective/coordinators/NCCLCoordinator.h

#pragma once

#include "ICollectiveCoordinator.h"
#include "../ICollectiveBackend.h"
#include <cuda_runtime.h>

namespace llaminar2 {

/**
 * @brief Dedicated coordinator thread for NCCL collective operations
 * 
 * Owns all ncclComm_t communicators, CUDA streams for collectives, and
 * completion events. All NCCL operations (including ncclGroupStart/End)
 * execute on this single thread, ensuring proper threading semantics.
 * 
 * Usage:
 *   NCCLCoordinator coord;
 *   coord.initialize({0, 1});  // GPUs 0 and 1
 *   
 *   // From any thread:
 *   coord.allreduceMulti(buffers, count, dtype, op);
 *   
 *   // Device worker synchronization:
 *   cudaStreamWaitEvent(compute_stream, coord.getCompletionEvent(gpu_idx));
 */
class NCCLCoordinator : public ICollectiveCoordinator {
public:
    NCCLCoordinator();
    ~NCCLCoordinator() override;
    
    // Non-copyable, non-movable
    NCCLCoordinator(const NCCLCoordinator&) = delete;
    NCCLCoordinator& operator=(const NCCLCoordinator&) = delete;
    
    // =========================================================================
    // ICollectiveCoordinator interface
    // =========================================================================
    
    bool initialize(const std::vector<int>& device_ordinals) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }
    
    void* getCompletionEvent(int device_idx) const override;
    void waitForDeviceEvent(int device_idx, void* worker_event) override;
    
    // =========================================================================
    // NCCL Collective Operations (thread-safe, queued to coordinator)
    // =========================================================================
    
    /**
     * @brief In-place allreduce across all local GPUs
     * 
     * @param buffers One buffer per device (size = num_devices)
     * @param count Elements per buffer
     * @param dtype Data type
     * @param op Reduction operation
     * @return true on success
     */
    bool allreduceMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op);
    
    /**
     * @brief Allgather across all local GPUs
     */
    bool allgatherMulti(const std::vector<const void*>& send_buffers,
                        const std::vector<void*>& recv_buffers,
                        size_t send_count, CollectiveDataType dtype);
    
    /**
     * @brief Broadcast from root to all local GPUs
     */
    bool broadcastMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, int root);
    
    /**
     * @brief Reduce-scatter across all local GPUs
     */
    bool reduceScatterMulti(const std::vector<const void*>& send_buffers,
                            const std::vector<void*>& recv_buffers,
                            size_t recv_count, CollectiveDataType dtype,
                            CollectiveOp op);
    
    /**
     * @brief Single-device allreduce (uses comm_[device_idx])
     */
    bool allreduce(void* buffer, size_t count, CollectiveDataType dtype,
                   CollectiveOp op, int device_idx);
    
    /**
     * @brief Synchronize all NCCL streams (blocking)
     */
    bool synchronize();
    
    /**
     * @brief Get last error message
     */
    const std::string& lastError() const { return last_error_; }
    
protected:
    void enqueueWork(std::function<void()> work) override;
    
private:
    // Worker thread
    void coordinatorLoop();
    void initializeOnThread();
    void cleanupOnThread();
    
    // Internal implementations (called ON coordinator thread)
    bool doAllreduceMulti(const std::vector<void*>& buffers, size_t count,
                          int dtype_int, int op_int);
    bool doAllgatherMulti(const std::vector<const void*>& send_buffers,
                          const std::vector<void*>& recv_buffers,
                          size_t send_count, int dtype_int);
    bool doBroadcastMulti(const std::vector<void*>& buffers, size_t count,
                          int dtype_int, int root);
    
    // State
    std::vector<int> device_ordinals_;
    int num_devices_ = 0;
    std::atomic<bool> initialized_{false};
    std::string last_error_;
    
    // NCCL state (owned by coordinator thread)
    std::vector<void*> comms_;           // ncclComm_t[]
    std::vector<cudaStream_t> streams_;  // One per device
    std::vector<cudaEvent_t> completion_events_;  // Signaled after each collective
    
    // Worker thread
    std::thread coordinator_thread_;
    std::atomic<bool> running_{false};
    
    // Work queue
    std::queue<std::function<void()>> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

} // namespace llaminar2
```

### NCCLCoordinator Implementation (Key Parts)

```cpp
// src/v2/collective/coordinators/NCCLCoordinator.cpp

bool NCCLCoordinator::initialize(const std::vector<int>& device_ordinals) {
    if (initialized_.load()) return true;
    
    device_ordinals_ = device_ordinals;
    num_devices_ = static_cast<int>(device_ordinals.size());
    
    // Start coordinator thread
    running_.store(true);
    coordinator_thread_ = std::thread(&NCCLCoordinator::coordinatorLoop, this);
    
    // Wait for initialization to complete on coordinator thread
    return submitAndWait([this]() {
        initializeOnThread();
        return initialized_.load();
    });
}

void NCCLCoordinator::initializeOnThread() {
    // Allocate state
    comms_.resize(num_devices_, nullptr);
    streams_.resize(num_devices_, nullptr);
    completion_events_.resize(num_devices_, nullptr);
    
    // Create streams and events for each device
    for (int i = 0; i < num_devices_; ++i) {
        cudaSetDevice(device_ordinals_[i]);
        cudaStreamCreate(&streams_[i]);
        cudaEventCreateWithFlags(&completion_events_[i], cudaEventDisableTiming);
    }
    
    // Initialize NCCL communicators
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    
    // Single-process multi-GPU: all comms init together
    ncclGroupStart();
    for (int i = 0; i < num_devices_; ++i) {
        cudaSetDevice(device_ordinals_[i]);
        ncclCommInitRank(reinterpret_cast<ncclComm_t*>(&comms_[i]),
                         num_devices_, id, i);
    }
    ncclGroupEnd();
    
    initialized_.store(true);
}

bool NCCLCoordinator::allreduceMulti(const std::vector<void*>& buffers,
                                      size_t count, CollectiveDataType dtype,
                                      CollectiveOp op) {
    return submitAndWait([&, this]() {
        return doAllreduceMulti(buffers, count, toNcclDataType(dtype), toNcclOp(op));
    });
}

bool NCCLCoordinator::doAllreduceMulti(const std::vector<void*>& buffers,
                                        size_t count, int dtype_int, int op_int) {
    // All of this runs on coordinator thread - groups work correctly!
    
    ncclGroupStart();
    for (int i = 0; i < num_devices_; ++i) {
        cudaSetDevice(device_ordinals_[i]);
        ncclAllReduce(buffers[i], buffers[i], count,
                      static_cast<ncclDataType_t>(dtype_int),
                      static_cast<ncclRedOp_t>(op_int),
                      static_cast<ncclComm_t>(comms_[i]),
                      streams_[i]);
    }
    ncclGroupEnd();
    
    // Record completion events for device workers
    for (int i = 0; i < num_devices_; ++i) {
        cudaSetDevice(device_ordinals_[i]);
        cudaEventRecord(completion_events_[i], streams_[i]);
    }
    
    return true;
}

void NCCLCoordinator::coordinatorLoop() {
    while (running_.load()) {
        std::function<void()> work;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !work_queue_.empty() || !running_.load();
            });
            
            if (!running_.load() && work_queue_.empty()) break;
            
            work = std::move(work_queue_.front());
            work_queue_.pop();
        }
        
        // Execute work on coordinator thread
        work();
    }
}
```

---

## RCCLCoordinator

```cpp
// src/v2/collective/coordinators/RCCLCoordinator.h

#pragma once

#include "ICollectiveCoordinator.h"
#include "../ICollectiveBackend.h"
#include <hip/hip_runtime.h>

namespace llaminar2 {

/**
 * @brief Dedicated coordinator thread for RCCL collective operations
 * 
 * Mirror of NCCLCoordinator for AMD ROCm GPUs. Uses HIP runtime and RCCL
 * (which has the same API as NCCL).
 * 
 * RCCL is loaded dynamically to avoid symbol conflicts with NCCL in
 * heterogeneous builds.
 */
class RCCLCoordinator : public ICollectiveCoordinator {
public:
    RCCLCoordinator();
    ~RCCLCoordinator() override;
    
    // Non-copyable, non-movable
    RCCLCoordinator(const RCCLCoordinator&) = delete;
    RCCLCoordinator& operator=(const RCCLCoordinator&) = delete;
    
    // =========================================================================
    // ICollectiveCoordinator interface
    // =========================================================================
    
    bool initialize(const std::vector<int>& device_ordinals) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }
    
    void* getCompletionEvent(int device_idx) const override;
    void waitForDeviceEvent(int device_idx, void* worker_event) override;
    
    // =========================================================================
    // RCCL Collective Operations (thread-safe, queued to coordinator)
    // =========================================================================
    
    bool allreduceMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op);
    
    bool allgatherMulti(const std::vector<const void*>& send_buffers,
                        const std::vector<void*>& recv_buffers,
                        size_t send_count, CollectiveDataType dtype);
    
    bool broadcastMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, int root);
    
    bool reduceScatterMulti(const std::vector<const void*>& send_buffers,
                            const std::vector<void*>& recv_buffers,
                            size_t recv_count, CollectiveDataType dtype,
                            CollectiveOp op);
    
    bool allreduce(void* buffer, size_t count, CollectiveDataType dtype,
                   CollectiveOp op, int device_idx);
    
    bool synchronize();
    
    const std::string& lastError() const { return last_error_; }
    
protected:
    void enqueueWork(std::function<void()> work) override;
    
private:
    void coordinatorLoop();
    void initializeOnThread();
    void cleanupOnThread();
    
    // Internal implementations (ON coordinator thread)
    bool doAllreduceMulti(const std::vector<void*>& buffers, size_t count,
                          int dtype_int, int op_int);
    
    // State
    std::vector<int> device_ordinals_;
    int num_devices_ = 0;
    std::atomic<bool> initialized_{false};
    std::string last_error_;
    
    // RCCL state (owned by coordinator thread)
    // Note: RCCL uses ncclComm_t but we store as void* for dynamic loading
    std::vector<void*> comms_;
    std::vector<hipStream_t> streams_;
    std::vector<hipEvent_t> completion_events_;
    
    // Dynamic loader for RCCL symbols
    void* rccl_lib_handle_ = nullptr;
    
    // Worker thread
    std::thread coordinator_thread_;
    std::atomic<bool> running_{false};
    
    // Work queue
    std::queue<std::function<void()>> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

} // namespace llaminar2
```

---

## PCIeBARCoordinator

```cpp
// src/v2/collective/coordinators/PCIeBARCoordinator.h

#pragma once

#include "ICollectiveCoordinator.h"
#include "../ICollectiveBackend.h"

namespace llaminar2 {

/**
 * @brief Coordinator for PCIe BAR heterogeneous transfers (CUDA↔ROCm)
 * 
 * Unlike NCCLCoordinator/RCCLCoordinator which handle homogeneous collectives,
 * PCIeBARCoordinator manages cross-vendor GPU communication via PCIe BAR
 * memory mapping.
 * 
 * Architecture:
 * - Owns BAR memory mappings (cudaDeviceEnablePeerAccess / hipDeviceEnablePeerAccess)
 * - Coordinates with BOTH CUDA and HIP runtimes
 * - Uses staging buffers for cross-vendor copies
 * - Records events on BOTH sides for synchronization
 * 
 * Cross-vendor synchronization:
 *   1. CUDA worker: cudaEventRecord(cuda_done, compute_stream)
 *   2. PCIeBAR coordinator: waitForCudaEvent(cuda_done)
 *   3. PCIeBAR coordinator: perform BAR transfer
 *   4. PCIeBAR coordinator: hipEventRecord(hip_ready, transfer_stream)
 *   5. HIP worker: hipStreamWaitEvent(compute_stream, hip_ready)
 */
class PCIeBARCoordinator : public ICollectiveCoordinator {
public:
    /**
     * @brief Device specification for heterogeneous setup
     */
    struct DeviceSpec {
        enum class Type { CUDA, ROCM };
        Type type;
        int ordinal;  // Device ordinal within that vendor
    };
    
    PCIeBARCoordinator();
    ~PCIeBARCoordinator() override;
    
    // Non-copyable, non-movable
    PCIeBARCoordinator(const PCIeBARCoordinator&) = delete;
    PCIeBARCoordinator& operator=(const PCIeBARCoordinator&) = delete;
    
    // =========================================================================
    // ICollectiveCoordinator interface
    // =========================================================================
    
    /**
     * @brief Initialize with device ordinals
     * 
     * For PCIeBAR, use initializeHeterogeneous() instead which takes
     * DeviceSpec to distinguish CUDA vs ROCm devices.
     */
    bool initialize(const std::vector<int>& device_ordinals) override;
    
    /**
     * @brief Initialize with heterogeneous device specs
     * 
     * @param devices Vector of CUDA and ROCm device specifications
     * @return true on success
     */
    bool initializeHeterogeneous(const std::vector<DeviceSpec>& devices);
    
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }
    
    void* getCompletionEvent(int device_idx) const override;
    void waitForDeviceEvent(int device_idx, void* worker_event) override;
    
    // =========================================================================
    // Heterogeneous Collective Operations
    // =========================================================================
    
    /**
     * @brief Allreduce across mixed CUDA+ROCm GPUs
     * 
     * Algorithm:
     * 1. Reduce within each vendor group (NCCL for CUDA, RCCL for ROCm)
     * 2. Cross-vendor transfer via PCIe BAR (one direction)
     * 3. Reduce the partial results
     * 4. Broadcast back to all GPUs
     * 
     * @param buffers One buffer per device (mixed CUDA/ROCm)
     * @param count Elements per buffer
     * @param dtype Data type
     * @param op Reduction operation
     * @return true on success
     */
    bool allreduceMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op);
    
    /**
     * @brief Direct memory copy between CUDA and ROCm GPUs via BAR
     * 
     * @param dst Destination pointer (on dst_device)
     * @param src Source pointer (on src_device)
     * @param bytes Number of bytes
     * @param src_device_idx Source device index
     * @param dst_device_idx Destination device index
     * @return true on success
     */
    bool crossVendorCopy(void* dst, const void* src, size_t bytes,
                         int src_device_idx, int dst_device_idx);
    
    /**
     * @brief Scatter data from one GPU to all others (cross-vendor)
     */
    bool scatter(const void* src, const std::vector<void*>& dst_buffers,
                 size_t count_per_dest, CollectiveDataType dtype,
                 int src_device_idx);
    
    /**
     * @brief Gather data from all GPUs to one (cross-vendor)
     */
    bool gather(const std::vector<const void*>& src_buffers, void* dst,
                size_t count_per_src, CollectiveDataType dtype,
                int dst_device_idx);
    
    bool synchronize();
    
    const std::string& lastError() const { return last_error_; }
    
    // =========================================================================
    // Coordinator Handles (for NCCL/RCCL sub-coordinators)
    // =========================================================================
    
    /**
     * @brief Get NCCLCoordinator for CUDA-only operations
     * 
     * Returns nullptr if no CUDA devices in this coordinator.
     */
    class NCCLCoordinator* ncclCoordinator() { return nccl_coord_.get(); }
    
    /**
     * @brief Get RCCLCoordinator for ROCm-only operations
     * 
     * Returns nullptr if no ROCm devices in this coordinator.
     */
    class RCCLCoordinator* rcclCoordinator() { return rccl_coord_.get(); }
    
protected:
    void enqueueWork(std::function<void()> work) override;
    
private:
    void coordinatorLoop();
    void initializeOnThread(const std::vector<DeviceSpec>& devices);
    void cleanupOnThread();
    
    // BAR memory management
    bool setupBARMappings();
    void* mapBARMemory(int src_device, int dst_device, size_t size);
    
    // Internal cross-vendor transfer
    bool doCrossVendorCopy(void* dst, const void* src, size_t bytes,
                           int src_idx, int dst_idx);
    
    // State
    std::vector<DeviceSpec> devices_;
    int num_devices_ = 0;
    std::atomic<bool> initialized_{false};
    std::string last_error_;
    
    // Device indices split by vendor
    std::vector<int> cuda_device_indices_;  // Indices into devices_
    std::vector<int> rocm_device_indices_;  // Indices into devices_
    
    // Sub-coordinators for homogeneous ops
    std::unique_ptr<NCCLCoordinator> nccl_coord_;
    std::unique_ptr<RCCLCoordinator> rccl_coord_;
    
    // Transfer streams (one per device)
    std::vector<void*> transfer_streams_;  // cudaStream_t or hipStream_t
    std::vector<void*> completion_events_; // cudaEvent_t or hipEvent_t
    
    // BAR staging buffers (for cross-vendor transfers)
    struct BARMapping {
        int src_device;
        int dst_device;
        void* src_bar_ptr;   // BAR-mapped pointer on src
        void* dst_bar_ptr;   // BAR-mapped pointer on dst
        size_t size;
    };
    std::vector<BARMapping> bar_mappings_;
    
    // Worker thread
    std::thread coordinator_thread_;
    std::atomic<bool> running_{false};
    
    // Work queue
    std::queue<std::function<void()>> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
};

} // namespace llaminar2
```

---

## Integration with NCCLBackend/RCCLBackend

The existing `NCCLBackend` and `RCCLBackend` classes become thin wrappers:

```cpp
// src/v2/collective/backends/NCCLBackend.h (modified)

class NCCLBackend : public ICollectiveBackend {
public:
    // ...existing interface...
    
    bool allreduceMulti(const std::vector<void*>& buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op) override {
        return coordinator_->allreduceMulti(buffers, count, dtype, op);
    }
    
    // Device workers use this to synchronize
    void* getCompletionEvent(int device_idx) const {
        return coordinator_->getCompletionEvent(device_idx);
    }
    
private:
    std::unique_ptr<NCCLCoordinator> coordinator_;
};
```

---

## Synchronization Flow Example

```cpp
// LocalTPContext::allreduceWithBarrierMultiGpu - refactored

void LocalTPContext::allreduceWithBarrierMultiGpu() {
    // Collect buffers from all devices (existing barrier logic)
    std::vector<void*> buffers = collectBarrierBuffers();
    
    // 1. Each device worker records "work done" event
    for (int i = 0; i < num_devices_; ++i) {
        auto& ctx = GPUDeviceContextPool::instance().getContext(devices_[i]);
        ctx.recordEvent(worker_done_events_[i]);
    }
    
    // 2. Coordinator waits for all device work
    for (int i = 0; i < num_devices_; ++i) {
        nccl_coordinator_->waitForDeviceEvent(i, worker_done_events_[i]);
    }
    
    // 3. Execute collective on coordinator thread
    nccl_coordinator_->allreduceMulti(buffers, count, dtype, op);
    
    // 4. Device workers wait for collective completion
    //    (getCompletionEvent already recorded by coordinator)
    for (int i = 0; i < num_devices_; ++i) {
        auto& ctx = GPUDeviceContextPool::instance().getContext(devices_[i]);
        ctx.waitEvent(nccl_coordinator_->getCompletionEvent(i));
    }
}
```

---

## Summary: Three Coordinators

| Coordinator | Owns | Purpose |
|-------------|------|---------|
| **NCCLCoordinator** | ncclComm_t[], CUDA streams, CUDA events | NCCL collectives for CUDA-only LOCAL TP |
| **RCCLCoordinator** | rcclComm_t[], HIP streams, HIP events | RCCL collectives for ROCm-only LOCAL TP |
| **PCIeBARCoordinator** | BAR mappings, transfer streams, sub-coordinators | Heterogeneous CUDA↔ROCm collectives |

This design:
1. ✅ Solves the threading mismatch (all ops on coordinator thread)
2. ✅ Supports ncclGroupStart/End (single thread = groups work)
3. ✅ Clean event synchronization with device workers
4. ✅ Heterogeneous support via PCIeBARCoordinator
