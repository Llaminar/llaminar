/**
 * @file CrossDomainTransfer.cpp
 * @brief Implementation of cross-domain activation transfer utility
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CrossDomainTransfer.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../backends/ComputeBackend.h"
#include "../../../backends/BackendManager.h"
#include "../../../utils/Logger.h"
#include <cstring>
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    CrossDomainTransfer::CrossDomainTransfer(Config config)
        : config_(std::move(config)), stats_{}
    {
        LOG_DEBUG("[CrossDomainTransfer] Created with config: "
                  << "pinned=" << config_.use_pinned_memory
                  << ", async=" << config_.async_transfers
                  << ", numa_node=" << config_.target_numa_node);
    }

    CrossDomainTransfer::~CrossDomainTransfer()
    {
        // Wait for any pending async transfers
        synchronize();

        LOG_DEBUG("[CrossDomainTransfer] Destroyed. Stats: "
                  << "GPU→CPU: " << stats_.gpu_to_cpu_count << " transfers, "
                  << stats_.gpu_to_cpu_bytes << " bytes; "
                  << "CPU→GPU: " << stats_.cpu_to_gpu_count << " transfers, "
                  << stats_.cpu_to_gpu_bytes << " bytes");
    }

    // =========================================================================
    // GPU → CPU Transfer
    // =========================================================================

    bool CrossDomainTransfer::gpuToCpu(const TensorBase *src, TensorBase *dst)
    {
        if (!src || !dst)
        {
            LOG_ERROR("[CrossDomainTransfer::gpuToCpu] Null tensor pointer");
            return false;
        }

        // Get the CPU tensor base for coherence operations
        auto *src_cpu = dynamic_cast<const TensorBase *>(src);
        auto *dst_cpu = dynamic_cast<TensorBase *>(dst);

        if (!src_cpu || !dst_cpu)
        {
            LOG_ERROR("[CrossDomainTransfer::gpuToCpu] Tensors must be TensorBase derived");
            return false;
        }

        // Validate sizes match (use public ITensorBase::size_bytes())
        size_t src_bytes = src->size_bytes();
        size_t dst_bytes = dst->size_bytes();

        if (dst_bytes < src_bytes)
        {
            LOG_ERROR("[CrossDomainTransfer::gpuToCpu] Destination too small: "
                      << dst_bytes << " < " << src_bytes << " bytes");
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Use tensor coherence protocol: src->data() syncs GPU→host if needed
        // This leverages the existing ensureOnHost() mechanism which handles
        // mapped memory, event-based sync, and deviceToHost memcpy.
        const float *src_data = src_cpu->data();
        if (!src_data)
        {
            LOG_ERROR("[CrossDomainTransfer::gpuToCpu] Failed to get source data (coherence sync failed)");
            return false;
        }

        // Get destination pointer and copy
        float *dst_data = dst_cpu->mutable_data();
        if (!dst_data)
        {
            LOG_ERROR("[CrossDomainTransfer::gpuToCpu] Failed to get destination pointer");
            return false;
        }

        // Perform the copy
        std::memcpy(dst_data, src_data, src_bytes);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Update statistics
        stats_.gpu_to_cpu_count++;
        stats_.gpu_to_cpu_bytes += src_bytes;
        stats_.gpu_to_cpu_time_ms += elapsed_ms;

        LOG_DEBUG("[CrossDomainTransfer::gpuToCpu] Transferred " << src_bytes
                                                                 << " bytes in " << elapsed_ms << " ms ("
                                                                 << (src_bytes / 1e6) / (elapsed_ms / 1e3) << " MB/s)");

        return true;
    }

    // =========================================================================
    // CPU → GPU Transfer
    // =========================================================================

    bool CrossDomainTransfer::cpuToGpu(const TensorBase *src, TensorBase *dst, DeviceId device_id)
    {
        if (!src || !dst)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Null tensor pointer");
            return false;
        }

        if (!device_id.is_gpu())
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Target device must be GPU: " << device_id.to_string());
            return false;
        }

        // Get the CPU tensor base for coherence operations
        auto *src_cpu = dynamic_cast<const TensorBase *>(src);
        auto *dst_cpu = dynamic_cast<TensorBase *>(dst);

        if (!src_cpu || !dst_cpu)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Tensors must be TensorBase derived");
            return false;
        }

        // Validate sizes match (use public ITensorBase::size_bytes())
        size_t src_bytes = src->size_bytes();
        size_t dst_bytes = dst->size_bytes();

        if (dst_bytes < src_bytes)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Destination too small: "
                      << dst_bytes << " < " << src_bytes << " bytes");
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Get source host data (use public ITensorBase::raw_data())
        const void *src_data = src->raw_data();
        if (!src_data)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Failed to get source host pointer");
            return false;
        }

        // Ensure destination is allocated on GPU (but don't upload host data)
        // We use allocateOnDevice() instead of ensureOnDevice() because we're about
        // to overwrite the GPU buffer with src's data, not dst's host data.
        if (!dst_cpu->allocateOnDevice(device_id))
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Failed to allocate destination on device "
                      << device_id.to_string());
            return false;
        }

        // Get the GPU destination pointer
        void *gpu_dst = dst_cpu->gpu_data_ptr();
        if (!gpu_dst)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] GPU allocation succeeded but pointer is null");
            return false;
        }

        // Perform the transfer via backend
        bool ok = transferCpuToGpuImpl(src_data, gpu_dst, src_bytes, device_id);
        if (!ok)
        {
            LOG_ERROR("[CrossDomainTransfer::cpuToGpu] Backend H2D transfer failed");
            return false;
        }

        // Mark destination as having valid GPU data (with event for fine-grained sync)
        dst_cpu->mark_device_dirty_with_event();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Update statistics
        stats_.cpu_to_gpu_count++;
        stats_.cpu_to_gpu_bytes += src_bytes;
        stats_.cpu_to_gpu_time_ms += elapsed_ms;

        LOG_DEBUG("[CrossDomainTransfer::cpuToGpu] Transferred " << src_bytes
                                                                 << " bytes to " << device_id.to_string()
                                                                 << " in " << elapsed_ms << " ms ("
                                                                 << (src_bytes / 1e6) / (elapsed_ms / 1e3) << " MB/s)");

        return true;
    }

    // =========================================================================
    // Transfer with Automatic Allocation
    // =========================================================================

    std::unique_ptr<TensorBase> CrossDomainTransfer::transfer(
        const TensorBase *src,
        DeviceType target_device,
        DeviceId target_device_id)
    {
        if (!src)
        {
            LOG_ERROR("[CrossDomainTransfer::transfer] Null source tensor");
            return nullptr;
        }

        auto *src_cpu = dynamic_cast<const TensorBase *>(src);
        if (!src_cpu)
        {
            LOG_ERROR("[CrossDomainTransfer::transfer] Source must be TensorBase derived");
            return nullptr;
        }

        // Get source shape for allocation
        std::vector<size_t> shape = src_cpu->shape();

        std::unique_ptr<TensorBase> dst;

        if (target_device == DeviceType::CPU)
        {
            // Allocate CPU tensor with NUMA awareness if configured
            dst = allocateCpuTensor(shape);
            if (!dst)
            {
                LOG_ERROR("[CrossDomainTransfer::transfer] Failed to allocate CPU destination tensor");
                return nullptr;
            }

            // Transfer GPU→CPU
            if (!gpuToCpu(src, dst.get()))
            {
                LOG_ERROR("[CrossDomainTransfer::transfer] GPU→CPU transfer failed");
                return nullptr;
            }
        }
        else if (target_device == DeviceType::CUDA || target_device == DeviceType::ROCm)
        {
            // Validate device ID matches target type
            if (!target_device_id.is_gpu())
            {
                LOG_ERROR("[CrossDomainTransfer::transfer] Must provide valid GPU device ID for GPU target");
                return nullptr;
            }

            // Allocate FP32 tensor on the target GPU device
            dst = std::make_unique<FP32Tensor>(shape, target_device_id);
            if (!dst)
            {
                LOG_ERROR("[CrossDomainTransfer::transfer] Failed to allocate GPU destination tensor");
                return nullptr;
            }

            // Transfer CPU→GPU
            if (!cpuToGpu(src, dst.get(), target_device_id))
            {
                LOG_ERROR("[CrossDomainTransfer::transfer] CPU→GPU transfer failed");
                return nullptr;
            }
        }
        else
        {
            LOG_ERROR("[CrossDomainTransfer::transfer] Unsupported target device type: "
                      << static_cast<int>(target_device));
            return nullptr;
        }

        return dst;
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void CrossDomainTransfer::synchronize()
    {
        // If async transfers were enabled, we would synchronize streams here.
        // Current implementation uses synchronous transfers, so this is a no-op.
        //
        // Future enhancement for async_transfers=true:
        // - Track pending transfer streams/events
        // - Wait for all pending events to complete

        if (config_.async_transfers)
        {
            LOG_DEBUG("[CrossDomainTransfer::synchronize] Async sync requested (currently no-op)");
            // TODO: Implement async synchronization when async_transfers is fully supported
        }
    }

    // =========================================================================
    // Internal Implementation
    // =========================================================================

    bool CrossDomainTransfer::transferGpuToCpuImpl(
        const void *src, void *dst, size_t bytes, DeviceId src_device)
    {
        // Get backend for the source device
        IBackend *backend = nullptr;

#ifdef HAVE_CUDA
        if (src_device.is_cuda())
        {
            backend = getBackendForDeviceType(ComputeBackendType::GPU_CUDA);
        }
#endif

#ifdef HAVE_ROCM
        if (src_device.is_rocm())
        {
            backend = getBackendForDeviceType(ComputeBackendType::GPU_ROCM);
        }
#endif

        if (!backend)
        {
            LOG_ERROR("[CrossDomainTransfer::transferGpuToCpuImpl] No backend for device "
                      << src_device.to_string());
            return false;
        }

        // Use backend's deviceToHost for the transfer
        int device_ordinal = src_device.gpu_ordinal();
        return backend->deviceToHost(dst, src, bytes, device_ordinal);
    }

    bool CrossDomainTransfer::transferCpuToGpuImpl(
        const void *src, void *dst, size_t bytes, DeviceId dst_device)
    {
        // Get backend for the destination device
        IBackend *backend = nullptr;

#ifdef HAVE_CUDA
        if (dst_device.is_cuda())
        {
            backend = getBackendForDeviceType(ComputeBackendType::GPU_CUDA);
        }
#endif

#ifdef HAVE_ROCM
        if (dst_device.is_rocm())
        {
            backend = getBackendForDeviceType(ComputeBackendType::GPU_ROCM);
        }
#endif

        if (!backend)
        {
            LOG_ERROR("[CrossDomainTransfer::transferCpuToGpuImpl] No backend for device "
                      << dst_device.to_string());
            return false;
        }

        // Use backend's hostToDevice for the transfer
        int device_ordinal = dst_device.gpu_ordinal();
        return backend->hostToDevice(dst, src, bytes, device_ordinal);
    }

    std::unique_ptr<TensorBase> CrossDomainTransfer::allocateCpuTensor(
        const std::vector<size_t> &shape)
    {
        // Use NUMA allocator if configured
        if (config_.numa_allocator && config_.target_numa_node >= 0)
        {
            // TODO: Integrate with NUMAAllocator for NUMA-aware FP32 tensor allocation
            // For now, fall back to standard allocation
            LOG_DEBUG("[CrossDomainTransfer::allocateCpuTensor] NUMA allocation requested "
                      << "for node " << config_.target_numa_node << " (not yet implemented, using standard)");
        }

        // Standard FP32 tensor allocation on CPU
        return std::make_unique<FP32Tensor>(shape, DeviceId::cpu());
    }

} // namespace llaminar2
