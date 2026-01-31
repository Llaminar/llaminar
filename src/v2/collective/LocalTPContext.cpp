/**
 * @file LocalTPContext.cpp
 * @brief Implementation of LOCAL tensor parallelism context
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LocalTPContext.h"
#include "backends/HostBackend.h"
#include "../tensors/TensorClasses.h"
#include "../backends/BackendManager.h" // For getCUDABackend, getROCmBackend
#include "../utils/Logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

// Conditionally include GPU-specific backends
#ifdef HAVE_CUDA
#include "backends/NCCLBackend.h"
#include <cuda_runtime.h> // For cudaMemcpy in zero-copy allreduce
#endif

#ifdef HAVE_ROCM
#include "backends/RCCLBackend.h"
#include "../backends/rocm/ROCmBackend.h" // For ROCmBackend::deviceToDevice
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/PCIeBARBackend.h"
#include "backends/HeterogeneousBackend.h"
#endif

namespace llaminar2
{
    // Helper function to get the appropriate backend for a device
    static IBackend *getBackendForDevice(DeviceId device)
    {
        if (device.is_cpu())
            return nullptr;

        if (device.is_cuda())
            return getCUDABackend();
        else if (device.is_rocm())
            return getROCmBackend();

        LOG_ERROR("[LocalTPContext] Unknown device type: " << device.toString());
        return nullptr;
    }

    // =========================================================================
    // Construction
    // =========================================================================

    LocalTPContext::LocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
        : devices_(std::move(devices))
    {
        // Validate devices
        if (devices_.empty())
        {
            throw std::invalid_argument("LocalTPContext: devices cannot be empty");
        }

        // Handle weights
        if (weights.empty())
        {
            // Equal distribution
            weights_.resize(devices_.size(), 1.0f / static_cast<float>(devices_.size()));
        }
        else if (weights.size() != devices_.size())
        {
            throw std::invalid_argument(
                "LocalTPContext: weights count (" + std::to_string(weights.size()) +
                ") must match device count (" + std::to_string(devices_.size()) + ")");
        }
        else
        {
            weights_ = normalizeWeights(weights);
        }

        // Handle backend
        if (backend == CollectiveBackendType::AUTO)
        {
            backend_ = autoDetectBackend(devices_);
        }
        else
        {
            backend_ = backend;
        }

        // Build lookup index
        buildDeviceIndex();

        LOG_DEBUG("LocalTPContext created: degree=" << degree()
                                                    << ", backend=" << collectiveBackendTypeToString(backend_));

        // Initialize backend for multi-device scenarios
        if (degree() > 1)
        {
            if (!initializeBackend())
            {
                LOG_WARN("LocalTPContext: Failed to initialize backend "
                         << collectiveBackendTypeToString(backend_)
                         << ", collectives will be no-ops");
            }
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    const std::vector<GlobalDeviceAddress> &LocalTPContext::devices() const
    {
        return devices_;
    }

    const std::vector<float> &LocalTPContext::weights() const
    {
        return weights_;
    }

    CollectiveBackendType LocalTPContext::backend() const
    {
        return backend_;
    }

    int LocalTPContext::degree() const
    {
        return static_cast<int>(devices_.size());
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool LocalTPContext::allreduce(TensorBase *tensor)
    {
        // Delegate to overload with empty stage name and default count (0 = use numel)
        return allreduce(tensor, "", 0);
    }

    bool LocalTPContext::allreduce(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::allreduce: null tensor");
            return false;
        }

        // Resolve count: 0 means use tensor->numel()
        const size_t effective_count = (count > 0) ? count : tensor->numel();

        std::unique_lock<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce: Backend not initialized, skipping");
            return true; // Return true to allow pipeline to continue
        }

        // ================================================================
        // Multi-GPU Backends (NCCL/RCCL): Use barrier-synchronized allreduce
        // ================================================================
        // For LOCAL TP with multiple threads (one per device), each thread calls
        // allreduce() with its OWN tensor. We CANNOT use getDeviceBuffers() on a
        // single tensor because TensorBase can only be on ONE GPU at a time.
        // Instead, use the barrier-synchronized approach where all device threads
        // rendezvous, collect their buffers, then the last arrival executes
        // allreduceMulti with all buffers.
        // ================================================================
        // PCIeBAR Backend: Use barrier-synchronized allreduce
        // ================================================================
        // CRITICAL: This check MUST happen BEFORE ensureOnDevice() because:
        // - Multiple threads call allreduce() concurrently, each with their OWN tensor
        // - Each thread may be running on behalf of a different device (CUDA:0 or ROCm:0)
        // - We cannot know which device "this thread" is without the barrier rendezvous
        // - If we call ensureOnDevice(devices_[0]) here, ALL threads would upload to CUDA:0
        // - The allreduceWithBarrier path handles device placement correctly via stage registration
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (backend_ == CollectiveBackendType::PCIE_BAR && degree() > 1)
        {
            // Release the main mutex before entering barrier (to avoid deadlock)
            // The barrier has its own mutex for synchronization
            lock.unlock();
            return allreduceWithBarrier(tensor, stage_name, effective_count);
        }
#endif

        // ================================================================
        // Multi-GPU Backends (NCCL/RCCL): Use barrier-synchronized allreduce
        // ================================================================
        // For LOCAL TP with multiple threads (one per device), each thread calls
        // allreduce() with its OWN tensor. We CANNOT use getDeviceBuffers() on a
        // single tensor because TensorBase can only be on ONE GPU at a time.
        // Instead, use the barrier-synchronized approach where all device threads
        // rendezvous, collect their buffers, then the last arrival executes
        // allreduceMulti with all buffers.
        if (backend_impl_->isMultiGpuSingleProcess() && degree() > 1)
        {
            // Release the main mutex before entering barrier (to avoid deadlock)
            // The barrier has its own mutex for synchronization
            lock.unlock();
            return allreduceWithBarrierMultiGpu(tensor, stage_name, effective_count);
        }
        else
        {
            // Fallback: single-buffer API (tensor must be on local device)
            // Ensure tensor is on the local device
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allreduce: Failed to ensure tensor on device");
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                LOG_ERROR("LocalTPContext::allreduce: No device buffer available");
                return false;
            }

            size_t reduce_count = effective_count;
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            LOG_DEBUG("LocalTPContext::allreduce: Single-buffer allreduce with "
                      << reduce_count << " elements (tensor numel=" << tensor->numel() << ")");

            bool result = backend_impl_->allreduce(
                buffer, reduce_count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allreduce: Backend allreduce failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::allreduce(const TensorBase *input, TensorBase *output)
    {
        if (!input || !output)
        {
            LOG_ERROR("LocalTPContext::allreduce: null input or output tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            // Copy input to output
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce (out-of-place): Backend not initialized, skipping");
            // Fall back to copy
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For out-of-place allreduce:
        // 1. Copy input to output
        // 2. Perform in-place allreduce on output
        LOG_DEBUG("LocalTPContext::allreduce (out-of-place): copying input to output first");

        // Copy on host first
        const float *src = input->data();
        float *dst = output->mutable_data();
        size_t count = std::min(input->numel(), output->numel());
        std::memcpy(dst, src, count * sizeof(float));

        // Now delegate to in-place allreduce (need to cast away const for the API)
        // The mutex is already held, so we call directly without re-locking
        return allreduceImpl(output);
    }

    // Private implementation that assumes lock is already held
    bool LocalTPContext::allreduceImpl(TensorBase *tensor)
    {
        if (!tensor)
        {
            return false;
        }

        if (degree() == 1)
        {
            return true;
        }

        if (!backend_initialized_ || !backend_impl_)
        {
            return true;
        }

        if (backend_impl_->isMultiGpuSingleProcess())
        {
            auto buffers = getDeviceBuffers(tensor);
            if (buffers.size() != static_cast<size_t>(degree()))
            {
                LOG_ERROR("LocalTPContext::allreduceImpl: Failed to get device buffers");
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            return backend_impl_->allreduceMulti(
                buffers, count, dtype, CollectiveOp::ALLREDUCE_SUM);
        }
        else
        {
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            // PCIeBAR Backend: Use registered allreduce path (same logic as main allreduce)
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            if (backend_ == CollectiveBackendType::PCIE_BAR)
            {
                if (!ensurePCIeBarBuffersRegistered(tensor))
                {
                    return false;
                }

                auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
                if (pcie_backend)
                {
                    // Copy tensor data to the registered BAR buffer on ROCm side
                    DeviceId rocm_device;
                    for (const auto &device : devices_)
                    {
                        DeviceId local_id = device.toLocalDeviceId();
                        if (local_id.is_rocm())
                        {
                            rocm_device = local_id;
                            break;
                        }
                    }

                    auto rocm_buf_opt = pcie_backend->getBuffer(pciebar_collective_id_, rocm_device);
                    if (rocm_buf_opt.has_value() && rocm_buf_opt->ptr)
                    {
                        const float *host_data = tensor->data();
                        std::memcpy(rocm_buf_opt->ptr, host_data, count * sizeof(float));
                    }
                }

                bool result = backend_impl_->allreduceRegistered(
                    pciebar_collective_id_, count, dtype, CollectiveOp::ALLREDUCE_SUM);

                if (result)
                {
                    tensor->mark_device_dirty_with_event();
                }
                return result;
            }
#endif

            bool result = backend_impl_->allreduce(
                buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty_with_event();
            }
            return result;
        }
    }

    // =========================================================================
    // PCIeBAR Barrier-Synchronized Allreduce
    // =========================================================================

    bool LocalTPContext::allreduceWithBarrier(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        const int num_participants = degree();

        std::unique_lock<std::mutex> lock(barrier_mutex_);

        // Capture current generation to detect spurious wakeups
        uint64_t my_generation = barrier_generation_.load();

        // Increment arrival count
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            // First arrival: initialize tensor collection vector and store stage name + count
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_tensor_ = tensor;         // Keep for backward compatibility
            barrier_stage_name_ = stage_name; // Store stage name for BAR-backed tensor lookup
            barrier_element_count_ = count;   // Store element count (0 = use tensor->numel())
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: First arrival (device thread), "
                      << "stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << ", count=" << count << " (0=use numel)"
                      << ", waiting for " << (num_participants - 1) << " more devices");
        }

        // Store this device's tensor in the collection
        // Each device thread stores its own tensor at its arrival slot
        barrier_tensors_[arrival_order] = tensor;

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Device arrival #" << (arrival_order + 1)
                                                                           << " of " << num_participants
                                                                           << " (tensor ptr=" << tensor << ")");

        if (arrival_order + 1 < num_participants)
        {
            // Not the last arrival: wait for completion with timeout
            constexpr auto BARRIER_TIMEOUT = std::chrono::seconds(30);

            bool completed = barrier_cv_.wait_for(lock, BARRIER_TIMEOUT, [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                // Timeout - likely a deadlock or missing participant
                LOG_ERROR("LocalTPContext::allreduceWithBarrier: TIMEOUT after 30s waiting for barrier! "
                          << "arrival_order=" << arrival_order << ", expected=" << num_participants
                          << " devices. Possible causes: missing device thread, kernel crash, or deadlock.");

                // Reset barrier state to allow recovery
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_tensor_ = nullptr;
                barrier_stage_name_.clear();
                barrier_element_count_ = 0;

                lock.unlock();
                barrier_cv_.notify_all(); // Wake any other waiters
                return false;
            }

            // Woke up - get the shared result
            bool result = barrier_result_;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Waiter released with result=" << result);
            return result;
        }

        // =====================================================================
        // LAST ARRIVAL: Execute the actual allreduce
        // =====================================================================
        // All other threads are waiting, so we have exclusive access to barrier_tensors_

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: All " << num_participants
                                                               << " devices arrived, executing PCIeBAR allreduce"
                                                               << " (count=" << barrier_element_count_ << ")");

        // The actual PCIeBAR transfer using all collected tensors and stored count
        bool success = executePCIeBarAllreduce(nullptr, barrier_element_count_);

        // Store result and signal completion
        barrier_result_ = success;
        barrier_tensors_.clear();
        barrier_tensor_ = nullptr;
        barrier_stage_name_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: PCIeBAR allreduce completed with result="
                  << success << ", releasing waiters (generation=" << barrier_generation_.load() << ")");

        lock.unlock();
        barrier_cv_.notify_all();

        return success;
    }

    // =========================================================================
    // Multi-GPU (NCCL/RCCL) Barrier-Synchronized Allreduce
    // =========================================================================
    //
    // For LOCAL TP with NCCL/RCCL, multiple device threads call allreduce()
    // concurrently, each with its OWN tensor. We need to:
    // 1. Collect all device buffers via barrier synchronization
    // 2. Have ONE thread execute allreduceMulti with all buffers
    // 3. All threads return after the collective completes
    //
    // This is necessary because TensorBase can only exist on ONE GPU at a time,
    // so we cannot use getDeviceBuffers() to gather buffers from a single tensor.
    // =========================================================================

    bool LocalTPContext::allreduceWithBarrierMultiGpu(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        const int num_participants = degree();

        // Determine which device index this tensor belongs to BEFORE taking the lock
        // This is critical: buffers must be ordered by device index, NOT arrival order
        int device_index = -1;
        auto tensor_device = tensor->current_device();
        if (tensor_device.has_value())
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i].toLocalDeviceId() == *tensor_device)
                {
                    device_index = static_cast<int>(i);
                    break;
                }
            }
        }

        if (device_index < 0 || device_index >= num_participants)
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: tensor device "
                      << (tensor_device.has_value() ? tensor_device->toString() : "none")
                      << " not found in LocalTPContext devices list (degree=" << num_participants << ")");
            return false;
        }

        std::unique_lock<std::mutex> lock(barrier_mutex_);

        // Capture current generation to detect spurious wakeups
        uint64_t my_generation = barrier_generation_.load();

        // Increment arrival count
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            // First arrival: initialize tensor collection vector and store count
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_element_count_ = count;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: First arrival (device thread), "
                      << "stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << ", count=" << count << " (0=use numel)"
                      << ", waiting for " << (num_participants - 1) << " more devices");
        }

        // Store this device's tensor at its DEVICE INDEX slot (not arrival order!)
        // This ensures buffers[i] corresponds to device_ordinals_[i] in RCCL
        barrier_tensors_[device_index] = tensor;

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Device arrival #" << (arrival_order + 1)
                                                                                   << " of " << num_participants
                                                                                   << " (tensor ptr=" << tensor
                                                                                   << ", device_index=" << device_index << ")");

        if (arrival_order + 1 < num_participants)
        {
            // Not the last arrival: wait for completion with timeout
            constexpr auto BARRIER_TIMEOUT = std::chrono::seconds(30);

            bool completed = barrier_cv_.wait_for(lock, BARRIER_TIMEOUT, [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                // Timeout - likely a deadlock or missing participant
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: TIMEOUT after 30s waiting for barrier! "
                          << "arrival_order=" << arrival_order << ", expected=" << num_participants
                          << " devices. Possible causes: missing device thread, kernel crash, or deadlock.");

                // Reset barrier state to allow recovery
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_element_count_ = 0;

                lock.unlock();
                barrier_cv_.notify_all(); // Wake any other waiters
                return false;
            }

            // Woke up - get the shared result
            bool result = barrier_result_;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Waiter released with result=" << result);
            return result;
        }

        // =====================================================================
        // LAST ARRIVAL: Execute the actual multi-GPU allreduce
        // =====================================================================
        // All other threads are waiting, so we have exclusive access to barrier_tensors_

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: All " << num_participants
                                                                       << " devices arrived, executing multi-GPU allreduce"
                                                                       << " (count=" << barrier_element_count_ << ")");

        // Collect device buffers from all tensors
        // CRITICAL: Each tensor must be on its own device, and we need to ensure that
        std::vector<void *> buffers;
        buffers.reserve(num_participants);

        // Determine effective count (use first tensor's numel if count is 0)
        size_t effective_count = barrier_element_count_;
        if (effective_count == 0 && barrier_tensors_[0] != nullptr)
        {
            effective_count = barrier_tensors_[0]->numel();
        }

        // Get buffer pointer from each tensor (they should already be on their respective devices)
        for (int i = 0; i < num_participants; ++i)
        {
            TensorBase *t = barrier_tensors_[i];
            if (!t)
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: null tensor at slot " << i);
                barrier_result_ = false;
                goto cleanup;
            }

            void *ptr = t->gpu_data_ptr();
            if (!ptr)
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: tensor at slot " << i
                                                                                          << " has no GPU buffer");
                barrier_result_ = false;
                goto cleanup;
            }

            // TRACE: Log detailed buffer info including device for debugging memory faults
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: BUFFER[" << i << "] "
                                                                              << "ptr=" << ptr << " tensor=" << static_cast<void *>(t)
                                                                              << " device=" << (t->current_device().has_value() ? t->current_device()->toString() : "none")
                                                                              << " name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName())
                                                                              << " numel=" << t->numel());

            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Buffer " << i
                                                                              << " ptr=" << ptr
                                                                              << " from tensor=" << t);
            buffers.push_back(ptr);
        }

        {
            // Execute the multi-GPU allreduce
            CollectiveDataType dtype = tensorDTypeToCollective(barrier_tensors_[0]);

            // TRACE: Log all buffer pointers before allreduce
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: ALLREDUCE START "
                      << "num_buffers=" << buffers.size() << " count=" << effective_count);
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                LOG_TRACE("  allreduce buffer[" << i << "] = " << buffers[i]);
            }

            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Calling allreduceMulti with "
                      << buffers.size() << " buffers, " << effective_count << " elements");

            bool success = backend_impl_->allreduceMulti(buffers, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (!success)
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: allreduceMulti failed: "
                          << backend_impl_->lastError());
                barrier_result_ = false;
                goto cleanup;
            }

            // TRACE: Log after allreduce completes
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: ALLREDUCE COMPLETE, syncing...");

            // Synchronize all GPU streams
            if (!backend_impl_->synchronize())
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: synchronize failed: "
                          << backend_impl_->lastError());
                barrier_result_ = false;
                goto cleanup;
            }

            // Mark all tensors as device-dirty (data was modified on GPU)
            for (int i = 0; i < num_participants; ++i)
            {
                if (barrier_tensors_[i])
                {
                    barrier_tensors_[i]->mark_device_dirty_with_event();
                }
            }

            barrier_result_ = true;
        }

    cleanup:
        // Clear barrier state and signal completion
        barrier_tensors_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        bool final_result = barrier_result_;

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Multi-GPU allreduce completed with result="
                  << final_result << ", releasing waiters (generation=" << barrier_generation_.load() << ")");

        lock.unlock();
        barrier_cv_.notify_all();

        return final_result;
    }

    bool LocalTPContext::executePCIeBarAllreduce(TensorBase * /* unused_tensor */, size_t count_param)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // ===========================================================================
        // PHASE 4 IMPLEMENTATION: Zero-Copy Allreduce using BAR-Backed Tensors
        // ===========================================================================
        //
        // For LOCAL TP with PCIeBAR, the key insight is:
        // - Each device has its OWN output tensor for row-parallel operations
        // - These tensors were allocated as BAR-backed and registered via registerBARBackedOutput()
        // - The stage name (stored in barrier_stage_name_) identifies which tensors to use
        //
        // Zero-Copy Flow:
        // 1. Lookup BAR-backed tensors by stage name
        // 2. CUDA device reads from ROCm's BAR region directly (zero-copy)
        // 3. Sum: cuda_output = cuda_partial + rocm_partial
        // 4. Write result back to ROCm via BAR (so both have the reduced value)
        // ===========================================================================

        // Lock the main mutex for backend operations
        std::lock_guard<std::mutex> lock(mutex_);

        // Find which device is CUDA and which is ROCm
        int cuda_idx = -1;
        int rocm_idx = -1;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            DeviceId local_id = devices_[i].toLocalDeviceId();
            if (local_id.is_cuda())
                cuda_idx = static_cast<int>(i);
            else if (local_id.is_rocm())
                rocm_idx = static_cast<int>(i);
        }

        if (cuda_idx < 0 || rocm_idx < 0)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Need exactly one CUDA and one ROCm device");
            return false;
        }

        // ===========================================================================
        // PRIMARY PATH: Use registered BAR-backed tensors for zero-copy operation
        // ===========================================================================
        if (!barrier_stage_name_.empty() && hasBARBackedOutputs(barrier_stage_name_))
        {
            auto bar_tensors = getBARBackedOutputs(barrier_stage_name_);

            FP32Tensor *cuda_tensor = bar_tensors.size() > static_cast<size_t>(cuda_idx) ? bar_tensors[cuda_idx] : nullptr;
            FP32Tensor *rocm_tensor = bar_tensors.size() > static_cast<size_t>(rocm_idx) ? bar_tensors[rocm_idx] : nullptr;

            if (cuda_tensor && rocm_tensor)
            {
                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Using BAR-backed tensors for stage '"
                          << barrier_stage_name_ << "' (CUDA tensor=" << cuda_tensor
                          << ", ROCm tensor=" << rocm_tensor << ")");

                // Validate tensors
                size_t tensor_numel = cuda_tensor->numel();
                if (rocm_tensor->numel() != tensor_numel)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor size mismatch: CUDA="
                              << tensor_numel << " ROCm=" << rocm_tensor->numel());
                    return false;
                }

                // Use count_param if provided, otherwise use tensor->numel()
                // CRITICAL: For decode with dynamic seq_len, count_param MUST be set correctly
                size_t count = (count_param > 0) ? count_param : tensor_numel;

                if (count > tensor_numel)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: count (" << count
                                                                                 << ") exceeds tensor capacity (" << tensor_numel << ")");
                    return false;
                }

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: count_param=" << count_param
                                                                                  << " tensor_numel=" << tensor_numel << " -> effective_count=" << count);

                // Ensure CUDA tensor is on the CUDA device
                DeviceId cuda_device = devices_[cuda_idx].toLocalDeviceId();
                if (!cuda_tensor->ensureOnDevice(cuda_device))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure CUDA tensor on device");
                    return false;
                }

                // Ensure ROCm tensor is on the ROCm device (should already be BAR-backed)
                DeviceId rocm_device = devices_[rocm_idx].toLocalDeviceId();
                if (!rocm_tensor->ensureOnDevice(rocm_device))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure ROCm tensor on device");
                    return false;
                }

                // Get pointers
                float *cuda_output = static_cast<float *>(cuda_tensor->gpu_data_ptr());

                // ===========================================================================
                // CRITICAL: ROCm kernels wrote to the HIP staging buffer, NOT directly to BAR!
                // We must copy from HIP staging → BAR so CUDA can read the ROCm partial result.
                //
                // Finding from Test__BARBackedHipAllocation exploration:
                //   - HIP kernels CANNOT dereference BAR mmap addresses (memory fault)
                //   - hipMemcpy(D2D) with BAR as destination WORKS at ~4+ GB/s (DMA engine)
                //   - This is the ONLY way for ROCm data to reach BAR
                // ===========================================================================
                const float *hip_staging_ptr = static_cast<const float *>(rocm_tensor->rocm_data_ptr());
                float *bar_ptr = static_cast<float *>(rocm_tensor->bar_address());

                if (!cuda_output || !hip_staging_ptr)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Invalid pointers "
                              << "(cuda=" << cuda_output << ", hip_staging=" << hip_staging_ptr << ")");
                    return false;
                }

                if (!bar_ptr)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm tensor has no BAR address "
                              << "(isBARBacked=" << rocm_tensor->isBARBacked() << ")");
                    return false;
                }

                size_t bytes = count * sizeof(float);

                // Step 1: Copy ROCm partial result from HIP staging buffer → BAR via hipMemcpy D2D
                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Copying ROCm staging → BAR: "
                          << bytes << " bytes, src=" << hip_staging_ptr << " dst=" << bar_ptr);

                // We need to use hipMemcpy to copy from HIP device memory to BAR
                // Note: We use the ROCm backend's deviceToDevice which wraps hipMemcpy
                ROCmBackend *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(rocm_device));
                if (!rocm_backend)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm backend not available");
                    return false;
                }

                rocm_backend->setDevice(rocm_device.toKernelDeviceIndex());

                // hipMemcpy(D2D) from HIP staging to BAR - this uses AMD DMA engine
                if (!rocm_backend->deviceToDevice(bar_ptr, hip_staging_ptr, bytes, rocm_device.toKernelDeviceIndex()))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to copy HIP staging → BAR");
                    return false;
                }

                // Synchronize HIP to ensure copy is complete before CUDA reads
                rocm_backend->synchronize(rocm_device.toKernelDeviceIndex());

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Reducing " << count << " elements "
                                                                               << "(cuda_ptr=" << cuda_output << ", bar_ptr=" << bar_ptr
                                                                               << ", ROCm BAR-backed=" << rocm_tensor->isBARBacked() << ")");

                // Get PCIeBAR backend for reduction kernel
                auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
                if (!pcie_backend)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Backend is not PCIeBARBackend");
                    return false;
                }

                // Step 2: Perform reduction: cuda_output += bar_ptr (CUDA reads from BAR)
                CollectiveDataType dtype = tensorDTypeToCollective(cuda_tensor);
                if (!pcie_backend->reduceOnCUDA(cuda_output, cuda_output, bar_ptr,
                                                count, dtype, CollectiveOp::ALLREDUCE_SUM))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA reduction kernel failed");
                    return false;
                }

                // Synchronize CUDA
                if (!pcie_backend->synchronize())
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA synchronization failed");
                    return false;
                }

                // Step 3: Write result back to ROCm via BAR
                cudaError_t err = cudaMemcpy(bar_ptr, cuda_output,
                                             bytes, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to write result to BAR: "
                              << cudaGetErrorString(err));
                    return false;
                }

                // Step 4: Copy result from BAR back to HIP staging buffer
                // so ROCm tensor has the final reduced value
                if (!rocm_backend->deviceToDevice(const_cast<float *>(hip_staging_ptr), bar_ptr,
                                                  bytes, rocm_device.toKernelDeviceIndex()))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to copy result BAR → HIP staging");
                    return false;
                }
                rocm_backend->synchronize(rocm_device.toKernelDeviceIndex());

                // Mark both tensors as having valid device data (with events for fine-grained sync)
                cuda_tensor->mark_device_dirty_with_event();
                rocm_tensor->mark_device_dirty_with_event();

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy allreduce completed successfully "
                          << "for stage '" << barrier_stage_name_ << "'");
                return true;
            }
            else
            {
                LOG_WARN("LocalTPContext::executePCIeBarAllreduce: BAR-backed tensors incomplete for stage '"
                         << barrier_stage_name_ << "' (cuda=" << cuda_tensor << ", rocm=" << rocm_tensor << ")");
                // Fall through to barrier_tensors_ path
            }
        }

        // ===========================================================================
        // FALLBACK PATH: Use barrier_tensors_[] (legacy - may have issues)
        // ===========================================================================
        // This path is used when:
        // - No stage name provided
        // - BAR-backed tensors not registered for this stage
        //
        // WARNING: This path has known issues when both threads pass the same tensor
        // (both CUDA and ROCm threads have reference to same params_.tensor)

        LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Using barrier_tensors_ fallback path "
                  << "(stage_name=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_) << ")");

        // Validate we have tensors from barrier
        if (barrier_tensors_.empty())
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: No tensors collected from barrier");
            return false;
        }

        // Identify CUDA and ROCm tensors from the barrier collection
        // NOTE: cuda_idx and rocm_idx were already determined above
        FP32Tensor *cuda_tensor = nullptr;
        FP32Tensor *rocm_tensor = nullptr;

        // The barrier_tensors_ stores by ARRIVAL order, not device order.
        // Try to identify by checking tensor properties.
        for (size_t i = 0; i < barrier_tensors_.size() && i < devices_.size(); ++i)
        {
            TensorBase *t = barrier_tensors_[i];
            if (!t)
            {
                LOG_WARN("LocalTPContext::executePCIeBarAllreduce: barrier_tensors_[" << i << "] is null");
                continue;
            }

            auto *fp32_t = dynamic_cast<FP32Tensor *>(t);
            if (!fp32_t)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor is not FP32Tensor");
                return false;
            }

            // Check if this tensor is BAR-backed (indicates ROCm origin for zero-copy)
            if (fp32_t->isBARBacked() && fp32_t->rocm_data_ptr() != nullptr)
            {
                rocm_tensor = fp32_t;
            }
            else if (fp32_t->gpu_data_ptr() != nullptr)
            {
                cuda_tensor = fp32_t;
            }
            else
            {
                // Fallback: assign by index matching device order
                DeviceId local_id = devices_[i].toLocalDeviceId();
                if (local_id.is_cuda())
                    cuda_tensor = fp32_t;
                else if (local_id.is_rocm())
                    rocm_tensor = fp32_t;
            }
        }

        // If we couldn't identify by properties, fall back to device index order
        if (!cuda_tensor && cuda_idx >= 0 && cuda_idx < static_cast<int>(barrier_tensors_.size()))
            cuda_tensor = dynamic_cast<FP32Tensor *>(barrier_tensors_[cuda_idx]);
        if (!rocm_tensor && rocm_idx >= 0 && rocm_idx < static_cast<int>(barrier_tensors_.size()))
            rocm_tensor = dynamic_cast<FP32Tensor *>(barrier_tensors_[rocm_idx]);

        // Check for shared buffer scenario (both devices passed the same tensor)
        if (cuda_tensor == rocm_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ARCHITECTURAL BUG - both devices "
                      "passed the same tensor pointer ("
                      << cuda_tensor << "). "
                                        "PCIeBAR allreduce requires each device to have its own buffer: "
                                        "CUDA device needs a CUDA-memory tensor, ROCm device needs a BAR-backed tensor. "
                                        "Fix: Ensure GraphBufferManager allocates per-device buffers for row-parallel "
                                        "outputs when using LOCAL TP with PCIeBAR backend.");
            return false;
        }

        // Final validation
        if (!cuda_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Could not identify CUDA tensor");
            return false;
        }
        if (!rocm_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Could not identify ROCm tensor");
            return false;
        }

        size_t tensor_numel = cuda_tensor->numel();
        if (rocm_tensor->numel() != tensor_numel)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor size mismatch: CUDA="
                      << tensor_numel << " ROCm=" << rocm_tensor->numel());
            return false;
        }

        // Use count_param if provided, otherwise use tensor->numel()
        // CRITICAL: For decode with dynamic seq_len, count_param MUST be set correctly
        size_t count = (count_param > 0) ? count_param : tensor_numel;

        if (count > tensor_numel)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: count (" << count
                                                                         << ") exceeds tensor capacity (" << tensor_numel << ")");
            return false;
        }

        LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Reducing " << count << " elements"
                                                                       << " (CUDA tensor=" << cuda_tensor
                                                                       << ", ROCm tensor=" << rocm_tensor
                                                                       << ", ROCm BAR-backed=" << rocm_tensor->isBARBacked()
                                                                       << ", count_param=" << count_param
                                                                       << ", tensor_numel=" << tensor_numel << ")");

        // ===========================================================================
        // ZERO-COPY PATH: If ROCm tensor is BAR-backed
        // ===========================================================================
        if (rocm_tensor->isBARBacked() && rocm_tensor->gpu_data_ptr() != nullptr)
        {
            // Zero-copy: CUDA reads directly from ROCm's BAR region
            //
            // The ROCm tensor's gpu_data_ptr() returns the CUDA-accessible pointer
            // to the BAR region (set up by initBARBackedDirect with bar_cuda_device_ptr_)
            //
            // Algorithm:
            // 1. Ensure CUDA tensor is on GPU
            // 2. CUDA kernel: cuda_output += rocm_bar_data
            // 3. Copy result back to ROCm (via BAR write)

            // Ensure CUDA tensor is on the CUDA device
            DeviceId cuda_device = devices_[cuda_idx].toLocalDeviceId();
            if (!cuda_tensor->ensureOnDevice(cuda_device))
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure CUDA tensor on device");
                return false;
            }

            float *cuda_output = static_cast<float *>(cuda_tensor->gpu_data_ptr());
            const float *rocm_bar_ptr = static_cast<const float *>(rocm_tensor->gpu_data_ptr());

            if (!cuda_output || !rocm_bar_ptr)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Invalid GPU pointers "
                          << "(cuda=" << cuda_output << ", rocm_bar=" << rocm_bar_ptr << ")");
                return false;
            }

            LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy path - "
                      << "CUDA reading from BAR at " << rocm_bar_ptr);

            // Get PCIeBAR backend for reduction kernel
            auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
            if (!pcie_backend)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Backend is not PCIeBARBackend");
                return false;
            }

            // Use the backend's reduceOnCUDA to perform: cuda_output += rocm_bar_ptr
            // This uses the CUDA vector add kernel
            CollectiveDataType dtype = tensorDTypeToCollective(cuda_tensor);
            if (!pcie_backend->reduceOnCUDA(cuda_output, cuda_output, rocm_bar_ptr,
                                            count, dtype, CollectiveOp::ALLREDUCE_SUM))
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA reduction kernel failed");
                return false;
            }

            // Synchronize CUDA
            if (!pcie_backend->synchronize())
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA synchronization failed");
                return false;
            }

            // Write result back to ROCm via BAR (so ROCm has the reduced result too)
            // For BAR-backed tensors, we can write directly through the BAR pointer
            // Use a CUDA memcpy from cuda_output to rocm_bar_ptr
            size_t bytes = count * sizeof(float);

            // For BAR-backed memory, the rocm_bar_ptr IS the CUDA-accessible pointer
            // We can use cudaMemcpy to write to it (CUDA writes through PCIe to AMD VRAM)
            cudaError_t err = cudaMemcpy(const_cast<float *>(rocm_bar_ptr), cuda_output,
                                         bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to write result to ROCm: "
                          << cudaGetErrorString(err));
                return false;
            }

            // Mark both tensors as having valid device data (with events for fine-grained sync)
            cuda_tensor->mark_device_dirty_with_event();
            rocm_tensor->mark_device_dirty_with_event();

            LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy allreduce completed successfully");
            return true;
        }

        // ===========================================================================
        // ERROR: ROCm tensor is not BAR-backed
        // ===========================================================================
        // PCIeBAR allreduce requires BAR-backed tensors for zero-copy operation.
        // The tensor must be allocated in the BAR region using TensorFactory with
        // BAR-backed allocation enabled.
        //
        // To fix this:
        // 1. Call TensorFactory::setDirectP2P(p2p_instance) before tensor allocation
        // 2. Ensure tensors are created with createFP32BARBacked() or similar
        // 3. Verify LocalTPContext tracks BAR-backed tensors correctly

        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm tensor is NOT BAR-backed! "
                  << "PCIeBAR allreduce requires zero-copy BAR-backed tensors. "
                  << "(rocm_tensor=" << rocm_tensor
                  << ", isBARBacked=" << rocm_tensor->isBARBacked()
                  << ", gpu_data_ptr=" << rocm_tensor->gpu_data_ptr() << ")");

        return false;
#else
        // Shouldn't be called without CUDA+ROCm
        (void)count_param;
        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: PCIeBAR requires both CUDA and ROCm");
        return false;
#endif
    }

    bool LocalTPContext::allgather(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        if (!local_shard || !global_tensor)
        {
            LOG_ERROR("LocalTPContext::allgather: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allgather: Backend not initialized, skipping");
            // Fall back to copy of local shard
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For LOCAL TP with Multi-GPU:
        // Each device sends its shard, receives all shards concatenated
        if (backend_impl_->isMultiGpuSingleProcess())
        {
            // For multi-GPU allgather, we need send buffers (one per device)
            // and recv buffers (one per device, each gets the full gathered result)

            // Note: In LOCAL TP, the local_shard and global_tensor parameters
            // represent the buffers for the "local" device. For true multi-GPU,
            // we would need separate buffers per device. For now, we use the
            // single-buffer API.
            DeviceId local_device = devices_[0].toLocalDeviceId();

            // Ensure local shard is on device (const-cast needed for ensureOnDevice)
            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            // Ensure global tensor is allocated on device
            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            LOG_DEBUG("LocalTPContext::allgather: allgather with "
                      << degree() << " devices, " << send_count << " elements per device");

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
        else
        {
            // Fallback to single-buffer allgather
            DeviceId local_device = devices_[0].toLocalDeviceId();

            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output)
    {
        if (shards.empty() || !output)
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: empty shards or null output");
            return false;
        }

        // Validate shard count matches device count
        if (static_cast<int>(shards.size()) != degree())
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: shard count (" << shards.size()
                                                                         << ") doesn't match device count (" << degree() << ")");
            return false;
        }

        // Verify all shards are non-null
        for (size_t i = 0; i < shards.size(); ++i)
        {
            if (!shards[i])
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: shard[" << i << "] is null");
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the shard to output
        if (degree() == 1)
        {
            const float *src = shards[0]->data();
            float *dst = output->mutable_data();
            size_t count = std::min(shards[0]->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Multi-device: concatenate all shards into output
        // For now, use CPU-side gather (works with any backend)
        // TODO: For GPU backends with allgatherMulti support, use device-side gather

        float *dst = output->mutable_data();
        size_t offset = 0;
        size_t output_capacity = output->numel();

        for (size_t i = 0; i < shards.size(); ++i)
        {
            const float *src = shards[i]->data();
            size_t shard_size = shards[i]->numel();

            // Check bounds
            if (offset + shard_size > output_capacity)
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: output buffer too small. "
                          << "Need " << (offset + shard_size) << ", have " << output_capacity);
                return false;
            }

            std::memcpy(dst + offset, src, shard_size * sizeof(float));
            offset += shard_size;
        }

        LOG_DEBUG("LocalTPContext::gatherFromDevices: gathered " << shards.size()
                                                                 << " shards, total " << offset << " elements");

        return true;
    }

    bool LocalTPContext::reduceScatter(const TensorBase *input, TensorBase *output_shard)
    {
        if (!input || !output_shard)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the appropriate shard
        if (degree() == 1)
        {
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = std::min(input->numel(), output_shard->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::reduceScatter: Backend not initialized, skipping");
            // Fall back to copy of first shard
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = output_shard->numel();
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // ReduceScatter: reduce across devices, each device gets a slice
        DeviceId local_device = devices_[0].toLocalDeviceId();

        // Ensure input is on device
        auto *mutable_input = const_cast<TensorBase *>(input);
        if (!mutable_input->ensureOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to ensure input on device");
            return false;
        }

        // Ensure output is allocated on device
        if (!output_shard->allocateOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to allocate output_shard on device");
            return false;
        }

        const void *send_buf = mutable_input->gpu_data_ptr();
        void *recv_buf = output_shard->gpu_data_ptr();

        if (!send_buf || !recv_buf)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: No device buffers available");
            return false;
        }

        size_t recv_count = output_shard->numel();
        CollectiveDataType dtype = tensorDTypeToCollective(input);

        LOG_DEBUG("LocalTPContext::reduceScatter: reduceScatter with "
                  << degree() << " devices, " << recv_count << " elements per device");

        bool result = backend_impl_->reduceScatter(
            send_buf, recv_buf, recv_count, dtype, CollectiveOp::ALLREDUCE_SUM);

        if (result)
        {
            output_shard->mark_device_dirty_with_event();
        }
        else
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Backend reduceScatter failed: "
                      << backend_impl_->lastError());
        }
        return result;
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void LocalTPContext::synchronize()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return;
        }

        // Synchronize via backend
        if (backend_initialized_ && backend_impl_)
        {
            LOG_DEBUG("LocalTPContext::synchronize: Synchronizing backend "
                      << collectiveBackendTypeToString(backend_));
            if (!backend_impl_->synchronize())
            {
                LOG_WARN("LocalTPContext::synchronize: Backend synchronize failed: "
                         << backend_impl_->lastError());
            }
        }
    }

    // =========================================================================
    // Device Management
    // =========================================================================

    int LocalTPContext::indexForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        if (it != device_to_index_.end())
        {
            return it->second;
        }
        return -1;
    }

    const GlobalDeviceAddress &LocalTPContext::deviceAt(int index) const
    {
        if (index < 0 || index >= static_cast<int>(devices_.size()))
        {
            throw std::out_of_range(
                "LocalTPContext::deviceAt: index " + std::to_string(index) +
                " out of range [0, " + std::to_string(devices_.size()) + ")");
        }
        return devices_[index];
    }

    float LocalTPContext::weightForDevice(const GlobalDeviceAddress &device) const
    {
        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::weightForDevice: device not found");
            return 0.0f;
        }
        return weights_[idx];
    }

    // =========================================================================
    // Weight Sharding Utilities
    // =========================================================================

    int LocalTPContext::headsForDevice(const GlobalDeviceAddress &device, int total_heads) const
    {
        if (total_heads <= 0)
        {
            return 0;
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::headsForDevice: device not found");
            return 0;
        }

        // Use cumulative counts to ensure exact distribution
        auto cumulative = computeCumulativeCounts(total_heads, weights_);
        return cumulative[idx + 1] - cumulative[idx];
    }

    std::pair<int, int> LocalTPContext::rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const
    {
        if (total_rows <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::rowRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_rows, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    std::pair<int, int> LocalTPContext::colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const
    {
        if (total_cols <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::colRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_cols, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    std::vector<float> LocalTPContext::normalizeWeights(const std::vector<float> &weights)
    {
        if (weights.empty())
        {
            return {};
        }

        // Check for non-positive weights
        for (float w : weights)
        {
            if (w <= 0.0f)
            {
                throw std::invalid_argument("LocalTPContext: weights must be positive");
            }
        }

        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            throw std::invalid_argument("LocalTPContext: weight sum must be positive");
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }

        return normalized;
    }

    std::vector<int> LocalTPContext::computeCumulativeCounts(
        int total, const std::vector<float> &norm_weights)
    {
        std::vector<int> cumulative(norm_weights.size() + 1);
        cumulative[0] = 0;

        // Distribute proportionally with rounding to ensure exact total
        int remaining = total;
        float remaining_weight = 1.0f;

        for (size_t i = 0; i < norm_weights.size(); ++i)
        {
            if (i == norm_weights.size() - 1)
            {
                // Last device gets the remainder to ensure exact total
                cumulative[i + 1] = total;
            }
            else
            {
                // Proportional distribution with proper rounding
                float proportion = norm_weights[i] / remaining_weight;
                int count = static_cast<int>(std::round(proportion * remaining));

                // Ensure at least 1 if there's work remaining
                if (count == 0 && remaining > 0)
                {
                    count = 1;
                }
                // Don't exceed remaining
                count = std::min(count, remaining);

                cumulative[i + 1] = cumulative[i] + count;
                remaining -= count;
                remaining_weight -= norm_weights[i];
            }
        }

        return cumulative;
    }

    CollectiveBackendType LocalTPContext::autoDetectBackend(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        bool has_cuda = false;
        bool has_rocm = false;
        bool has_cpu = false;

        for (const auto &dev : devices)
        {
            if (dev.isCUDA())
                has_cuda = true;
            else if (dev.isROCm())
                has_rocm = true;
            else if (dev.isCPU())
                has_cpu = true;
        }

        // If CPU is involved, use host-staged backend
        if (has_cpu)
        {
            return CollectiveBackendType::HOST;
        }

        // Mixed GPU types
        if (has_cuda && has_rocm)
        {
            // Count devices of each type to determine backend
            int num_cuda = 0;
            int num_rocm = 0;
            for (const auto &dev : devices)
            {
                if (dev.isCUDA())
                    num_cuda++;
                else if (dev.isROCm())
                    num_rocm++;
            }

            // Use simple PCIeBAR for exactly 1+1 case (2 devices total)
            if (num_cuda == 1 && num_rocm == 1)
            {
                return CollectiveBackendType::PCIE_BAR;
            }

            // Use hierarchical backend for N+M case (>2 devices)
            return CollectiveBackendType::HETEROGENEOUS;
        }

        // All CUDA - use NCCL
        if (has_cuda && !has_rocm)
        {
            return CollectiveBackendType::NCCL;
        }

        // All ROCm - use RCCL
        if (has_rocm && !has_cuda)
        {
            return CollectiveBackendType::RCCL;
        }

        // Default to HOST if nothing detected (shouldn't happen)
        return CollectiveBackendType::HOST;
    }

    void LocalTPContext::buildDeviceIndex()
    {
        device_to_index_.clear();
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            device_to_index_[devices_[i]] = static_cast<int>(i);
        }
    }

    // =========================================================================
    // Backend Initialization and Helper Methods
    // =========================================================================

    bool LocalTPContext::initializeBackend()
    {
        // Build device group from devices_
        DeviceGroupBuilder builder;
        builder.setName("LocalTP_" + std::to_string(degree()) + "_devices");
        builder.setScope(CollectiveScope::LOCAL);
        builder.setLocalRank(0); // In LOCAL TP, we manage all devices from rank 0

        for (const auto &device : devices_)
        {
            builder.addDevice(device.toLocalDeviceId());
        }

        device_group_ = builder.build();

        // Create appropriate backend based on type
        switch (backend_)
        {
        case CollectiveBackendType::NCCL:
#ifdef HAVE_CUDA
            LOG_DEBUG("LocalTPContext: Creating NCCL backend");
            backend_impl_ = std::make_unique<NCCLBackend>();
#else
            LOG_WARN("LocalTPContext: NCCL requested but CUDA not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::RCCL:
#ifdef HAVE_ROCM
            LOG_DEBUG("LocalTPContext: Creating RCCL backend");
            backend_impl_ = std::make_unique<RCCLBackend>();
#else
            LOG_WARN("LocalTPContext: RCCL requested but ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            LOG_DEBUG("LocalTPContext: Creating PCIe BAR backend");
            backend_impl_ = std::make_unique<PCIeBARBackend>();
#else
            LOG_WARN("LocalTPContext: PCIE_BAR requested but both CUDA and ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::HETEROGENEOUS:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            LOG_DEBUG("LocalTPContext: Creating Heterogeneous backend");
            backend_impl_ = std::make_unique<HeterogeneousBackend>();
#else
            LOG_WARN("LocalTPContext: HETEROGENEOUS requested but both CUDA and ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::HOST:
        case CollectiveBackendType::AUTO:
        default:
            LOG_DEBUG("LocalTPContext: Creating HOST backend");
            backend_impl_ = std::make_unique<HostBackend>();
            break;
        }

        // Check if backend is available
        if (!backend_impl_->isAvailable())
        {
            LOG_WARN("LocalTPContext: Backend " << collectiveBackendTypeToString(backend_)
                                                << " not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
            backend_ = CollectiveBackendType::HOST; // Update to reflect actual backend
        }

        // Initialize the backend with device group
        if (!backend_impl_->initialize(device_group_))
        {
            LOG_ERROR("LocalTPContext: Failed to initialize backend: "
                      << backend_impl_->lastError());
            backend_impl_.reset();
            backend_initialized_ = false;
            return false;
        }

        backend_initialized_ = true;
        LOG_INFO("LocalTPContext: Backend " << backend_impl_->name()
                                            << " initialized for " << degree() << " devices");
        return true;
    }

    bool LocalTPContext::ensurePCIeBarBuffersRegistered(TensorBase *tensor)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!tensor)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: null tensor");
            return false;
        }

        // Only needed for PCIeBAR backend
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            return true;
        }

        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (!pcie_backend)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Backend is not PCIeBARBackend");
            return false;
        }

        size_t buffer_bytes = tensor->numel() * sizeof(float);

        // If buffers are already registered for this size, we're done
        if (pciebar_buffers_registered_ && buffer_bytes <= pciebar_buffer_size_)
        {
            LOG_DEBUG("ensurePCIeBarBuffersRegistered: Reusing registered buffers for collective "
                      << pciebar_collective_id_);
            return true;
        }

        // If size changed, we need to re-register (unregister old first)
        if (pciebar_buffers_registered_ && buffer_bytes > pciebar_buffer_size_)
        {
            LOG_WARN("ensurePCIeBarBuffersRegistered: Buffer size increased from "
                     << pciebar_buffer_size_ << " to " << buffer_bytes
                     << ", need to re-register. This may leak BAR memory.");
            // Note: We can't actually reclaim BAR allocations (bump allocator),
            // but we can unregister the old collective_id
            for (const auto &device : devices_)
            {
                pcie_backend->unregisterBuffer(pciebar_collective_id_, device.toLocalDeviceId());
            }
            pciebar_buffers_registered_ = false;
        }

        // Generate a unique collective_id for this LocalTPContext
        pciebar_collective_id_ = "LocalTP_allreduce_" + std::to_string(reinterpret_cast<uintptr_t>(this));
        pciebar_buffer_size_ = buffer_bytes;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Registering buffers for " << pciebar_collective_id_
                                                                             << " (size=" << buffer_bytes << " bytes)");

        // For 2-device LOCAL TP (CUDA + ROCm):
        // 1. CUDA buffer: use tensor's existing GPU buffer (allocated by standard path)
        // 2. ROCm buffer: must be allocated in BAR region to get correct offset

        DeviceId cuda_device;
        DeviceId rocm_device;
        void *cuda_ptr = nullptr;

        // Identify CUDA and ROCm devices
        for (const auto &device : devices_)
        {
            DeviceId local_id = device.toLocalDeviceId();
            if (local_id.is_cuda())
            {
                cuda_device = local_id;
                // Get CUDA buffer from tensor
                if (!tensor->ensureOnDevice(cuda_device))
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to ensure tensor on CUDA device");
                    return false;
                }
                cuda_ptr = tensor->gpu_data_ptr();
                if (!cuda_ptr)
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: No CUDA buffer available");
                    return false;
                }
            }
            else if (local_id.is_rocm())
            {
                rocm_device = local_id;
            }
        }

        if (!cuda_device.is_valid() || !rocm_device.is_valid())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Need exactly one CUDA and one ROCm device");
            return false;
        }

        // Allocate ROCm buffer in BAR region
        auto bar_alloc = pcie_backend->allocateInBarRegion(buffer_bytes);
        if (!bar_alloc.has_value())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to allocate " << buffer_bytes
                                                                            << " bytes in BAR region");
            return false;
        }

        void *rocm_bar_ptr = bar_alloc->first;
        size_t rocm_bar_offset = bar_alloc->second;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Allocated BAR buffer at offset " << rocm_bar_offset);

        // Register CUDA buffer
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, cuda_device, cuda_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register CUDA buffer");
            return false;
        }

        // Register ROCm buffer (with BAR offset)
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, rocm_device, rocm_bar_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register ROCm buffer");
            pcie_backend->unregisterBuffer(pciebar_collective_id_, cuda_device);
            return false;
        }

        pciebar_buffers_registered_ = true;
        LOG_INFO("ensurePCIeBarBuffersRegistered: Successfully registered buffers for PCIeBAR allreduce"
                 << " (collective_id=" << pciebar_collective_id_
                 << ", size=" << buffer_bytes << " bytes"
                 << ", BAR offset=" << rocm_bar_offset << ")");

        return true;
#else
        (void)tensor;
        return true; // No-op when PCIeBAR not available
#endif
    }

    std::vector<void *> LocalTPContext::getDeviceBuffers(TensorBase *tensor)
    {
        std::vector<void *> buffers;
        buffers.reserve(devices_.size());

        // For LOCAL TP, we need to get the GPU buffer for each device
        // Current implementation assumes tensor has data on all devices
        // or we need to replicate it

        for (size_t i = 0; i < devices_.size(); ++i)
        {
            DeviceId device_id = devices_[i].toLocalDeviceId();

            // Ensure tensor data is on this device
            if (!tensor->ensureOnDevice(device_id))
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: Failed to ensure tensor on device "
                          << device_id.toString());
                return {}; // Return empty vector to indicate failure
            }

            void *ptr = tensor->gpu_data_ptr();
            if (!ptr)
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: No GPU buffer for device "
                          << device_id.toString());
                return {};
            }

            buffers.push_back(ptr);
        }

        return buffers;
    }

    CollectiveDataType LocalTPContext::tensorDTypeToCollective(const TensorBase *tensor) const
    {
        if (!tensor)
        {
            return CollectiveDataType::FLOAT32;
        }

        // Get tensor type and map to CollectiveDataType
        // Most common case is FP32 for activation tensors
        TensorType tt = tensor->native_type();
        switch (tt)
        {
        case TensorType::FP32:
            return CollectiveDataType::FLOAT32;
        case TensorType::FP16:
            return CollectiveDataType::FLOAT16;
        case TensorType::BF16:
            return CollectiveDataType::BFLOAT16;
        case TensorType::INT32:
            return CollectiveDataType::INT32;
        case TensorType::INT8:
        case TensorType::Q8_0:
        case TensorType::Q8_1:
            return CollectiveDataType::INT8;
        default:
            // For quantized types that don't have a direct mapping, use FLOAT32
            // (collectives are typically on dequantized activations)
            return CollectiveDataType::FLOAT32;
        }
    }

    // =========================================================================
    // BAR-Backed Tensor Registry
    // =========================================================================

    void LocalTPContext::registerBARBackedOutput(
        const std::string &stage_name,
        const GlobalDeviceAddress &device,
        TensorBase *tensor)
    {
        // Validate device is in our context
        int idx = indexForDevice(device);
        if (idx < 0)
        {
            throw std::invalid_argument(
                "Device " + device.toString() + " not in LocalTPContext");
        }

        // Validate tensor is non-null
        if (!tensor)
        {
            throw std::invalid_argument(
                "Tensor must be non-null");
        }

        // Cast to FP32Tensor (allreduce outputs are FP32)
        FP32Tensor *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor);
        if (!fp32_tensor)
        {
            LOG_WARN("[LocalTPContext] registerBARBackedOutput: tensor is not FP32Tensor, skipping");
            return;
        }

        // For PCIeBAR backend:
        // - CUDA device: tensor does NOT need to be BAR-backed (regular CUDA memory)
        // - ROCm device: tensor MUST be BAR-backed for zero-copy P2P access
        //
        // We register ALL tensors here. The actual BAR-backed requirement only
        // applies to the ROCm device, which is validated during allreduce execution.
        //
        // For non-PCIeBAR backends: require BAR-backed for testing consistency
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            // Non-PCIeBAR backends: still require BAR-backed for consistency with existing tests
            // This maintains backward compatibility with unit tests using HOST backend
            if (!fp32_tensor->isBARBacked())
            {
                throw std::invalid_argument(
                    "Tensor must be BAR-backed");
            }
        }

        // Log whether tensor is BAR-backed for debugging
        LOG_TRACE("[LocalTPContext] registerBARBackedOutput: stage='" << stage_name
                                                                      << "' device=" << device.toString()
                                                                      << " is_bar_backed=" << fp32_tensor->isBARBacked());

        // Initialize vector for this stage if needed
        auto &tensors = bar_output_tensors_[stage_name];
        if (tensors.empty())
        {
            tensors.resize(degree(), nullptr);
        }

        // Register
        tensors[idx] = fp32_tensor;

        LOG_DEBUG("[LocalTPContext] Registered BAR-backed output for stage '"
                  << stage_name << "' device " << device.toString());
    }

    std::vector<FP32Tensor *> LocalTPContext::getBARBackedOutputs(
        const std::string &stage_name) const
    {

        auto it = bar_output_tensors_.find(stage_name);
        if (it == bar_output_tensors_.end())
        {
            return std::vector<FP32Tensor *>(degree(), nullptr);
        }
        return it->second;
    }

    bool LocalTPContext::hasBARBackedOutputs(const std::string &stage_name) const
    {
        auto it = bar_output_tensors_.find(stage_name);
        if (it == bar_output_tensors_.end())
        {
            return false;
        }
        // Check if at least one entry is non-null
        for (auto *t : it->second)
        {
            if (t != nullptr)
                return true;
        }
        return false;
    }

    void LocalTPContext::clearBARBackedOutputs()
    {
        bar_output_tensors_.clear();
        LOG_DEBUG("[LocalTPContext] Cleared all BAR-backed output registrations");
    }

    std::shared_ptr<DirectP2PEngine> LocalTPContext::getDirectP2PEngine() const
    {
        // Only available for PCIeBAR backend
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            return nullptr;
        }

        if (!backend_impl_)
        {
            return nullptr;
        }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Cast to PCIeBARBackend and get the engine
        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (!pcie_backend)
        {
            return nullptr;
        }

        DirectP2PEngine *raw_ptr = pcie_backend->getDirectP2PEngine();
        if (!raw_ptr)
        {
            return nullptr;
        }

        // Return a shared_ptr that doesn't delete the engine (it's owned by PCIeBARBackend)
        // We use aliasing constructor with empty shared_ptr to create non-owning shared_ptr
        return std::shared_ptr<DirectP2PEngine>(std::shared_ptr<void>(), raw_ptr);
#else
        // PCIeBAR backend requires both CUDA and ROCm
        return nullptr;
#endif
    }

    bool LocalTPContext::reserveTempBufferBytes(size_t bytes)
    {
        if (!backend_impl_)
        {
            LOG_WARN("[LocalTPContext] Cannot reserve temp buffer: backend not initialized");
            return false;
        }

        LOG_DEBUG("[LocalTPContext] Reserving temp buffer: " << bytes << " bytes");
        return backend_impl_->reserveTempBufferBytes(bytes);
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ILocalTPContext> createLocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
    {
        return std::make_unique<LocalTPContext>(
            std::move(devices),
            std::move(weights),
            backend);
    }

} // namespace llaminar2
