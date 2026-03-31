/**
 * @file TensorBase.cpp
 * @brief TensorBase class implementation (helper methods)
 *
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "TensorKernels.h"
#include "SIMDHelpers.h"
#include "../utils/CPUFeatures.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/StackTrace.h"
#include "../utils/KernelProfiler.h"
#include "../backends/BackendManager.h"
#include "../backends/ComputeBackend.h"
#include "../backends/DeviceId.h"
#include "../kernels/KernelFactory.h"
#include "../collective/backends/PCIeBARBackend.h"
#include "../collective/BackendRouter.h"
#include "../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../transfer/TransferEngine.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// Forward declare pinned memory registration functions from HostBackend implementations
// These are defined in HostBackendROCm.cpp and HostBackendCUDA.cu
namespace llaminar2
{
    namespace host_backend_detail
    {
#ifdef HAVE_CUDA
        bool cudaHostRegisterBuffer(void *ptr, size_t size);
        void cudaHostUnregisterBuffer(void *ptr);
#endif
#ifdef HAVE_ROCM
        bool hipHostRegisterBuffer(void *ptr, size_t size);
        void hipHostUnregisterBuffer(void *ptr);
#endif
    } // namespace host_backend_detail
} // namespace llaminar2

namespace llaminar2
{

    // ===== Helper functions for DeviceId-based backend access =====

    /**
     * @brief Get the appropriate backend for a DeviceId
     *
     * Returns the correct backend (CUDA or ROCm) based on the device type.
     *
     * @param device DeviceId specifying the target device
     * @return IBackend* for the device, or nullptr for CPU
     */
    static IBackend *getBackendForDevice(DeviceId device)
    {
        if (device.is_cpu())
            return nullptr;

        if (device.is_cuda())
            return getCUDABackend();
        else if (device.is_rocm())
            return getROCmBackend();

        LOG_ERROR("[TensorBase] Unknown device type: " << device.toString());
        return nullptr;
    }

    /**
     * @brief Instance method to resolve backend, checking injected_backend_ first
     *
     * If a test has injected a backend via setBackendForTesting(), returns that.
     * Otherwise falls back to the global backend lookup via resolveBackend().
     *
     * @param device DeviceId specifying the target device
     * @return IBackend* for the device, or nullptr for CPU (with no injected backend)
     */
    IBackend *TensorBase::resolveBackend(DeviceId device) const
    {
        if (injected_backend_)
            return injected_backend_;
        return getBackendForDevice(device);
    }

    // ===== Legacy helper functions for global device index mapping =====

    /**
     * @brief Get the appropriate backend for a global device index
     *
     * Maps global device index (e.g., 1, 2, 3) to the correct backend (CUDA or ROCm)
     * based on the device type from DeviceManager.
     *
     * @param device_idx Global device index (0 = CPU, 1+ = GPUs)
     * @deprecated Use resolveBackend(DeviceId) instead
     * @return IBackend* for the device, or nullptr for CPU/invalid
     */
    static IBackend *getBackendForGlobalDeviceIdx(int device_idx)
    {
        if (device_idx <= 0)
            return nullptr; // CPU doesn't use IBackend

        // Get device type from DeviceManager
        const auto &devices = DeviceManager::instance().devices();
        if (static_cast<size_t>(device_idx) >= devices.size())
        {
            LOG_ERROR("[TensorBase] Invalid device index: " << device_idx
                                                            << " (max: " << devices.size() - 1 << ")");
            return nullptr;
        }

        const auto &device = devices[device_idx];
        return getBackendForDeviceType(device.type);
    }

    /**
     * @brief Convert global device index to backend-specific device ID
     *
     * E.g., global index 2 (second AMD GPU) -> ROCm device 0
     *
     * @param device_idx Global device index
     * @return Backend-specific device ID
     */
    static int getBackendSpecificDeviceId(int device_idx)
    {
        if (device_idx <= 0)
            return 0;

        const auto &devices = DeviceManager::instance().devices();
        if (static_cast<size_t>(device_idx) >= devices.size())
        {
            LOG_ERROR("[TensorBase] Invalid device index for device ID lookup: " << device_idx);
            return 0;
        }

        return devices[device_idx].device_id;
    }

    // ===== TensorBase destructor =====
    // Clears the kernel cache entry for this tensor to prevent use-after-free
    // when a new tensor is allocated at the same memory address.
    // Also frees mapped memory if this tensor used zero-copy allocation.
    // Also frees BAR-backed memory if this tensor used PCIeBAR allocation.
    // Also frees GPU memory and unpins host memory if allocated.
    TensorBase::~TensorBase()
    {
        // Free mapped memory if allocated (must be done first)
        freeMappedMemory();

        // Free BAR-backed memory if allocated
        freeBARBackedMemory();

        // Free secondary device buffers (from multi-device transfers)
        for (auto &[key, ptr] : secondary_device_buffers_)
        {
            if (ptr != nullptr)
            {
                // Skip BAR-allocated buffers - they're managed by PCIeBARBackend
                // and will be reclaimed when the backend shuts down
                if (secondary_bar_allocated_keys_.count(key) > 0)
                {
                    LOG_DEBUG("[TensorBase::~TensorBase] Skipping BAR-allocated secondary buffer (key=" << key << ")");
                    continue;
                }

                // Unpack device ID from key
                DeviceType type = static_cast<DeviceType>(key >> 16);
                int ordinal = key & 0xFFFF;
                DeviceId device(type, ordinal);

                IBackend *backend = resolveBackend(device);
                if (backend)
                {
                    backend->free(ptr, ordinal);
                }
            }
        }
        secondary_device_buffers_.clear();
        secondary_bar_allocated_keys_.clear();

        // Free GPU memory if allocated (before unpinning host memory)
        // Skip if gpu_data_ptr_ was pointing to BAR memory (already freed above)
        if (gpu_data_ptr_ && gpu_device_.has_value() && !is_bar_backed_)
        {
            IBackend *backend = resolveBackend(*gpu_device_);
            if (backend)
            {
                int backend_device_id = gpu_device_->gpu_ordinal();

                // Destroy completion event if it exists
                if (device_completion_event_)
                {
                    backend->destroyEvent(device_completion_event_, backend_device_id);
                    device_completion_event_ = nullptr;
                }

                backend->free(gpu_data_ptr_, backend_device_id);
            }
            gpu_data_ptr_ = nullptr;
            setCoherenceState_(TensorCoherenceState::HOST_ONLY); // GPU memory freed
        }

        // Unpin host memory if pinned (after freeing GPU memory)
        unpinHostMemory();

        // Clear kernel cache
        llaminar::v2::kernels::KernelFactory::clearCacheFor(this);
    }

    // ===== Zero-Copy Mapped Memory Implementation =====
    bool TensorBase::initMappedMemory(size_t bytes, DeviceId target_device)
    {
        // Validate target device - must be a GPU
        if (!target_device.is_gpu())
        {
            LOG_ERROR("[TensorBase::initMappedMemory] Target device must be GPU, got: " << target_device.toString());
            return false;
        }

        // Get backend for target device
        IBackend *backend = resolveBackend(target_device);
        if (!backend)
        {
            LOG_ERROR("[TensorBase::initMappedMemory] No backend available for device " << target_device.toString());
            return false;
        }

        // Allocate mapped memory
        int backend_device_id = target_device.gpu_ordinal();
        void *device_ptr = nullptr;
        void *host_ptr = backend->allocateMapped(bytes, backend_device_id, &device_ptr);

        if (!host_ptr || !device_ptr)
        {
            LOG_WARN("[TensorBase::initMappedMemory] Failed to allocate mapped memory ("
                     << bytes << " bytes on device " << target_device.toString() << ")");
            return false;
        }

        // Zero-initialize the mapped memory
        std::memset(host_ptr, 0, bytes);

        // Set up mapped memory state
        is_mapped_ = true;
        mapped_host_ptr_ = host_ptr;
        mapped_device_ptr_ = device_ptr;
        gpu_data_ptr_ = device_ptr; // GPU pointer is the device-visible mapped pointer
        gpu_device_ = target_device;

        // Both host and device are always valid for mapped memory
        memory_residency_ = MemoryResidency::MAPPED;
        setCoherenceState_(TensorCoherenceState::MAPPED);

        LOG_TRACE("[TensorBase::initMappedMemory] Allocated " << bytes << " bytes mapped memory"
                                                              << " host_ptr=" << host_ptr << " device_ptr=" << device_ptr
                                                              << " on device " << target_device.toString());

        return true;
    }

    void TensorBase::freeMappedMemory()
    {
        if (!is_mapped_ || !mapped_host_ptr_)
        {
            return; // Not mapped or already freed
        }

        if (gpu_device_.has_value())
        {
            IBackend *backend = resolveBackend(*gpu_device_);
            if (backend)
            {
                int backend_device_id = gpu_device_->gpu_ordinal();
                backend->freeMapped(mapped_host_ptr_, backend_device_id);
                LOG_TRACE("[TensorBase::freeMappedMemory] Freed mapped memory");
            }
        }

        mapped_host_ptr_ = nullptr;
        gpu_data_ptr_ = nullptr; // gpu_data_ptr_ was pointing to mapped_device_ptr_
        mapped_device_ptr_ = nullptr;
        is_mapped_ = false;
    }

    // ===== BAR-Backed Memory Implementation =====

    bool TensorBase::initBARBackedMemory(size_t bytes, DeviceId cuda_device, DeviceId rocm_device,
                                         PCIeBARBackend *backend)
    {
        // Validate inputs
        if (!cuda_device.is_cuda())
        {
            LOG_ERROR("[TensorBase::initBARBackedMemory] cuda_device must be CUDA, got: " << cuda_device.toString());
            return false;
        }
        if (!rocm_device.is_rocm())
        {
            LOG_ERROR("[TensorBase::initBARBackedMemory] rocm_device must be ROCm, got: " << rocm_device.toString());
            return false;
        }
        if (!backend)
        {
            LOG_ERROR("[TensorBase::initBARBackedMemory] backend must not be null");
            return false;
        }
        if (!backend->isInitialized())
        {
            LOG_ERROR("[TensorBase::initBARBackedMemory] backend must be initialized");
            return false;
        }

        // Allocate in BAR region
        auto alloc_result = backend->allocateInBarRegion(bytes);
        if (!alloc_result.has_value())
        {
            LOG_ERROR("[TensorBase::initBARBackedMemory] Failed to allocate " << bytes
                                                                              << " bytes in BAR region");
            return false;
        }

        auto [rocm_ptr, bar_offset] = alloc_result.value();

        // Get the CUDA-accessible pointer for this BAR region
        // The DirectP2PEngine provides a base CUDA pointer to the BAR;
        // we need to add our offset to get the pointer for this allocation
        void *cuda_bar_base = nullptr;
        if (backend->isPCIeBarActive())
        {
            // TODO: Add PCIeBARBackend::getCudaBarPointer() method
            // For now, we'll need to get this from the underlying DirectP2PEngine
            // This is a placeholder - actual implementation depends on exposing
            // the CUDA pointer calculation from PCIeBARBackend
            LOG_WARN("[TensorBase::initBARBackedMemory] TODO: Get CUDA BAR pointer from backend");

            // Placeholder: In full implementation, would be:
            // cuda_bar_base = backend->getDirectP2PEngine()->getCudaBarPointer();
            // bar_cuda_device_ptr_ = static_cast<char*>(cuda_bar_base) + bar_offset;

            // For now, store null - this needs DirectP2PEngine integration
            bar_cuda_device_ptr_ = nullptr;
        }

        // Set up BAR-backed memory state
        is_bar_backed_ = true;
        bar_offset_ = bar_offset;
        bar_size_ = bytes;
        bar_rocm_ptr_ = rocm_ptr;
        bar_host_device_ = rocm_device;
        bar_accessor_device_ = cuda_device;
        bar_backend_ = backend;

        // For CUDA kernel dispatch, use the BAR pointer
        // (once we have it from DirectP2PEngine integration)
        // gpu_data_ptr_ = bar_cuda_device_ptr_;
        // gpu_device_ = cuda_device;

        // Both host and device are always "valid" for BAR-backed tensors
        // since they share the same physical memory
        memory_residency_ = MemoryResidency::BAR_BACKED;
        setCoherenceState_(TensorCoherenceState::SYNCED);

        LOG_DEBUG("[TensorBase::initBARBackedMemory] Allocated " << bytes << " bytes in BAR region"
                                                                 << " rocm_ptr=" << rocm_ptr
                                                                 << " offset=" << bar_offset
                                                                 << " cuda_device=" << cuda_device.toString()
                                                                 << " rocm_device=" << rocm_device.toString());

        return true;
    }

    void TensorBase::freeBARBackedMemory()
    {
        if (!is_bar_backed_ || !bar_rocm_ptr_)
        {
            return; // Not BAR-backed or already freed
        }

        if (bar_backend_)
        {
            // We own this BAR memory (allocated via initBARBackedMemory)
            bar_backend_->freeBarBuffer(bar_rocm_ptr_);
            LOG_TRACE("[TensorBase::freeBARBackedMemory] Freed BAR buffer at offset " << bar_offset_);
        }
        else
        {
            // Externally managed BAR memory (initialized via initBARBackedDirect)
            // Do NOT free - just clear our references
            LOG_TRACE("[TensorBase::freeBARBackedMemory] Releasing reference to externally-managed BAR memory");
        }

        // Free HIP staging buffer if we own it
#if defined(HAVE_ROCM)
        if (hip_staging_ptr_ && owns_hip_staging_)
        {
            IBackend *rocm_backend = resolveBackend(bar_host_device_);
            if (rocm_backend)
            {
                rocm_backend->setDevice(bar_host_device_.toKernelDeviceIndex());
                rocm_backend->free(hip_staging_ptr_, bar_host_device_.toKernelDeviceIndex());
                LOG_TRACE("[TensorBase::freeBARBackedMemory] Freed HIP staging buffer at " << hip_staging_ptr_);
            }
        }
#endif
        hip_staging_ptr_ = nullptr;
        owns_hip_staging_ = false;

        // Clear BAR state
        // IMPORTANT: Keep is_bar_backed_ true until AFTER clearing gpu_data_ptr_
        // to prevent the destructor from calling cudaFree on BAR memory
        bar_offset_ = 0;
        bar_size_ = 0;
        bar_rocm_ptr_ = nullptr;

        // If gpu_data_ptr_ was pointing to BAR memory, clear it BEFORE resetting is_bar_backed_
        if (gpu_data_ptr_ == bar_cuda_device_ptr_)
        {
            gpu_data_ptr_ = nullptr;
        }

        bar_cuda_device_ptr_ = nullptr;
        bar_backend_ = nullptr;

        // Now safe to reset is_bar_backed_ since gpu_data_ptr_ is either cleared or was never BAR memory
        is_bar_backed_ = false;
    }

    void TensorBase::initBARBackedDirect(void *rocm_ptr, void *cuda_ptr,
                                         DeviceId rocm_device, DeviceId cuda_device,
                                         size_t bytes)
    {
        // Set up BAR-backed state with pre-obtained pointers
        // Note: Unlike initBARBackedMemory, this does NOT allocate memory
        // and does NOT take ownership - no deallocation on destruction

        is_bar_backed_ = true;
        bar_offset_ = 0; // No offset tracking for direct initialization
        bar_size_ = bytes;
        bar_rocm_ptr_ = rocm_ptr;
        bar_cuda_device_ptr_ = cuda_ptr;
        bar_host_device_ = rocm_device;
        bar_accessor_device_ = cuda_device;
        bar_backend_ = nullptr; // No backend - memory managed externally

        // ===== Allocate HIP staging buffer for ROCm kernel writes =====
        // CRITICAL: HIP kernels CANNOT directly dereference BAR mmap addresses!
        // We must allocate a real HIP device buffer for kernel writes, then
        // copy to BAR via hipMemcpy(D2D) which uses the AMD DMA engine.
#if defined(HAVE_ROCM)
        // Get ROCm backend for allocation
        IBackend *rocm_backend = resolveBackend(rocm_device);
        if (rocm_backend)
        {
            // Set device context
            rocm_backend->setDevice(rocm_device.toKernelDeviceIndex());

            // Allocate HIP staging buffer
            hip_staging_ptr_ = rocm_backend->allocate(bytes, rocm_device.toKernelDeviceIndex());
            if (hip_staging_ptr_)
            {
                owns_hip_staging_ = true;
                LOG_DEBUG("[TensorBase::initBARBackedDirect] Allocated HIP staging buffer: "
                          << bytes << " bytes at " << hip_staging_ptr_
                          << " on device " << rocm_device.toString());
            }
            else
            {
                LOG_ERROR("[TensorBase::initBARBackedDirect] Failed to allocate HIP staging buffer "
                          << "(" << bytes << " bytes) on " << rocm_device.toString());
            }
        }
        else
        {
            LOG_ERROR("[TensorBase::initBARBackedDirect] ROCm backend not available for device "
                      << rocm_device.toString());
        }
#else
        // No ROCm - staging buffer not needed (CUDA-only path)
        hip_staging_ptr_ = nullptr;
        owns_hip_staging_ = false;
        LOG_DEBUG("[TensorBase::initBARBackedDirect] ROCm not available, no HIP staging buffer allocated");
#endif

        // Set up GPU pointer and device for ROCm kernel execution
        // The HIP staging buffer is where ROCm kernels read/write data.
        // For BAR-backed tensors, the flow is:
        //   1. ROCm kernels operate on hip_staging_ptr_
        //   2. When transferTo(CUDA) is called, data is copied:
        //      hip_staging_ptr_ → bar_rocm_ptr_ (BAR) → bar_cuda_device_ptr_ (CUDA)
        //   3. After transfer, CUDA becomes authoritative
#if defined(HAVE_ROCM)
        if (hip_staging_ptr_)
        {
            gpu_data_ptr_ = hip_staging_ptr_;
            gpu_device_ = rocm_device;
        }
        else
        {
            // Fallback: use CUDA pointers if no staging buffer
            gpu_data_ptr_ = cuda_ptr;
            gpu_device_ = cuda_device;
        }
#else
        gpu_data_ptr_ = cuda_ptr;
        gpu_device_ = cuda_device;
#endif

        // Both host and device are always "valid" for BAR-backed tensors
        // since they share the same physical memory
        memory_residency_ = MemoryResidency::BAR_BACKED;
        setCoherenceState_(TensorCoherenceState::SYNCED);

        LOG_DEBUG("[TensorBase::initBARBackedDirect] Configured BAR-backed tensor: "
                  << bytes << " bytes"
                  << " bar_ptr=" << rocm_ptr
                  << " cuda_ptr=" << cuda_ptr
                  << " hip_staging=" << hip_staging_ptr_
                  << " rocm_device=" << rocm_device.toString()
                  << " cuda_device=" << cuda_device.toString());
    }

    void TensorBase::to_fp32_via_blocks(float *dst) const
    {
        // This helper is for quantized tensors that implement ITensorGemmTileDataProvider
        const ITensorGemmTileDataProvider *decoder = dynamic_cast<const ITensorGemmTileDataProvider *>(this);
        if (!decoder)
        {
            throw std::runtime_error("to_fp32_via_blocks() called on non-ITensorGemmTileDataProvider tensor");
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_via_blocks() requires 2D tensor");
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        const size_t block_sz = decoder->block_size();
        const size_t blocks_per_row = (cols + block_sz - 1) / block_sz;

        // Decode each block to the output buffer
        for (size_t row = 0; row < rows; ++row)
        {
            float *row_dst = dst + row * cols;
            for (size_t kb = 0; kb < blocks_per_row; ++kb)
            {
                const size_t offset = kb * block_sz;
                const size_t count = std::min(block_sz, cols - offset);

                // Decode block to temporary buffer
                float block_buffer[256]; // Max block size supported
                decoder->decode_block_at(row, kb, block_buffer);

                // Copy decoded values to output
                for (size_t i = 0; i < count; ++i)
                {
                    row_dst[offset + i] = block_buffer[i];
                }
            }
        }
    }

    bool TensorBase::to_int8_perchannel_via_blocks(int8_t *dst_int8,
                                                   float *dst_col_scales,
                                                   float *dst_row_scales) const
    {
        // Verify this is an ITensorGemmTileDataProvider tensor
        const ITensorGemmTileDataProvider *decoder = dynamic_cast<const ITensorGemmTileDataProvider *>(this);
        if (!decoder)
        {
            LOG_ERROR("[TensorBase] to_int8_perchannel_via_blocks() requires ITensorGemmTileDataProvider interface");
            return false;
        }

        // Verify 2D shape
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[TensorBase] to_int8_perchannel_via_blocks() requires 2D tensor, got " << shp.size() << "D");
            return false;
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        const size_t block_sz = decoder->block_size();
        const size_t blocks_per_row = (cols + block_sz - 1) / block_sz;

        // Step 1: Decode entire tensor to FP32 (temporary buffer)
        std::vector<float> fp32_data(rows * cols);

#pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row)
        {
            float *row_dst = fp32_data.data() + row * cols;
            for (size_t kb = 0; kb < blocks_per_row; ++kb)
            {
                const size_t offset = kb * block_sz;
                const size_t count = std::min(block_sz, cols - offset);

                // Decode block
                float block_buffer[256]; // Max block size
                decoder->decode_block_at(row, kb, block_buffer);

                // Copy to FP32 buffer
                for (size_t i = 0; i < count; ++i)
                {
                    row_dst[offset + i] = block_buffer[i];
                }
            }
        }

        // Step 2: Compute per-column scales
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                float abs_val = std::fabs(fp32_data[i * cols + j]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }
            dst_col_scales[j] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Step 3: Compute per-row scales (if requested)
        if (dst_row_scales != nullptr)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
            {
                float max_abs = 0.0f;
                for (size_t j = 0; j < cols; ++j)
                {
                    float abs_val = std::fabs(fp32_data[i * cols + j]);
                    if (abs_val > max_abs)
                        max_abs = abs_val;
                }
                dst_row_scales[i] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            }
        }

        // Step 4: Quantize to INT8 using per-column scales (or per-row if requested)
        const bool use_row_scales = (dst_row_scales != nullptr);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
        {
            const float row_inv_scale = use_row_scales ? (1.0f / dst_row_scales[i]) : 1.0f;

            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                float inv_scale;

                if (use_row_scales)
                {
                    inv_scale = row_inv_scale;
                }
                else
                {
                    inv_scale = 1.0f / dst_col_scales[j];
                }

                float scaled = fp32_data[idx] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::round(scaled));

                // Clamp to INT8 range
                if (quantized > 127)
                    quantized = 127;
                else if (quantized < -127)
                    quantized = -127;

                dst_int8[idx] = static_cast<int8_t>(quantized);
            }
        }

        return true;
    }

    ActivationPack TensorBase::pack_activation_rows_to_int8(int rows, int cols) const
    {
        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[TensorBase] pack_activation_rows_to_int8 requires positive dimensions");
            return {};
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[TensorBase] pack_activation_rows_to_int8 requires 2D tensor, got " << shp.size() << "D");
            return {};
        }
        if (static_cast<size_t>(rows) > shp[0] || static_cast<size_t>(cols) != shp[1])
        {
            LOG_ERROR("[TensorBase] pack_activation_rows_to_int8 dimension mismatch: tensor is ["
                      << shp[0] << ", " << shp[1] << "], requested " << rows << "x" << cols);
            return {};
        }

        ActivationPack pack;
        pack.rows = rows;
        pack.cols = cols;
        const size_t row_stride = static_cast<size_t>(cols);
        const size_t total = row_stride * static_cast<size_t>(rows);
        pack.data.resize(total, 0);
        pack.row_scales.resize(static_cast<size_t>(rows), 1.0f);

        std::vector<float> row_buffer(row_stride, 0.0f);
        for (int m = 0; m < rows; ++m)
        {
            to_fp32_row(static_cast<size_t>(m), row_buffer.data());

            const float max_abs = simd::activation_row_max_abs(row_buffer.data(), cols);
            const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            pack.row_scales[static_cast<size_t>(m)] = scale;
            const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            int8_t *row_dst = pack.data.data() + static_cast<size_t>(m) * row_stride;
            simd::quantize_activation_row(row_buffer.data(), cols, inv_scale, row_dst);
        }

        return pack;
    }

    // ===== Template Specializations for to<T>() =====

    // FP32 conversion (just call to_fp32)
    template <>
    void TensorBase::to<float>(float *dst, TensorType format) const
    {
        to_fp32(dst);
    }

    // FP16 conversion
    template <>
    void TensorBase::to<uint16_t>(uint16_t *dst, TensorType format) const
    {
        if (format == TensorType::FP16)
        {
            to_fp16(dst);
        }
        else if (format == TensorType::BF16)
        {
            to_bf16(dst);
        }
        else
        {
            throw std::runtime_error("to<uint16_t>() requires format to be FP16 or BF16");
        }
    }

    // INT8 conversion (uses to_int8_blocked as default)
    template <>
    void TensorBase::to<int8_t>(int8_t *dst, TensorType format) const
    {
        // For now, use blocked quantization with default block size
        // TODO: Add per-channel option via format parameter
        const size_t total = element_count();
        const size_t block_size = 32;
        const size_t num_blocks = (total + block_size - 1) / block_size;

        std::vector<float> scales(num_blocks);
        to_int8_blocked(dst, scales.data(), block_size);
    }

    // INT32 conversion (convert to FP32 then scale)
    template <>
    void TensorBase::to<int32_t>(int32_t *dst, TensorType format) const
    {
        // Convert to FP32 first
        const size_t total = element_count();
        std::vector<float> temp_fp32(total);
        to_fp32(temp_fp32.data());

        // Find scale factor (max absolute value)
        float max_abs = 0.0f;
        for (size_t i = 0; i < total; ++i)
        {
            max_abs = std::max(max_abs, std::abs(temp_fp32[i]));
        }

        // Scale to INT32 range (use ~2^30 to avoid overflow in downstream ops)
        const float scale = (max_abs > 1e-10f) ? (1073741824.0f / max_abs) : 1.0f;

        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = static_cast<int32_t>(std::round(temp_fp32[i] * scale));
        }
    }

    // Helper to convert TensorType enum to string
    static const char *tensorTypeToString(TensorType type)
    {
        switch (type)
        {
        case TensorType::FP32:
            return "FP32";
        case TensorType::BF16:
            return "BF16";
        case TensorType::FP16:
            return "FP16";
        case TensorType::INT8:
            return "INT8";
        case TensorType::INT32:
            return "INT32";
        case TensorType::IQ4_NL:
            return "IQ4_NL";
        case TensorType::IQ4_XS:
            return "IQ4_XS";
        case TensorType::Q8_0:
            return "Q8_0";
        case TensorType::Q4_0:
            return "Q4_0";
        case TensorType::Q4_1:
            return "Q4_1";
        case TensorType::Q5_0:
            return "Q5_0";
        case TensorType::Q5_1:
            return "Q5_1";
        case TensorType::Q6_K:
            return "Q6_K";
        case TensorType::Q2_K:
            return "Q2_K";
        case TensorType::Q5_K:
            return "Q5_K";
        case TensorType::Q3_K:
            return "Q3_K";
        case TensorType::Q4_K:
            return "Q4_K";
        case TensorType::Q8_K:
            return "Q8_K";
        case TensorType::IQ2_XXS:
            return "IQ2_XXS";
        case TensorType::IQ2_XS:
            return "IQ2_XS";
        case TensorType::IQ3_XXS:
            return "IQ3_XXS";
        case TensorType::IQ2_S:
            return "IQ2_S";
        case TensorType::IQ3_S:
            return "IQ3_S";
        case TensorType::IQ1_S:
            return "IQ1_S";
        case TensorType::IQ1_M:
            return "IQ1_M";
        default:
            return "Unknown";
        }
    }

    // Default Q8_0 conversion (throws error for read-only quantized weight tensors)
    void TensorBase::to_q8_0(Q8_0Block *dst) const
    {
        // Get tensor dimensions and block configuration
        const size_t total_elements = element_count();
        constexpr size_t Q8_BLOCK_SIZE = 32;
        const size_t num_blocks = (total_elements + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

        // For 2D tensors with ITensorGemmTileDataProvider, use optimized block decode
        if (shape().size() == 2)
        {
            const auto *provider = dynamic_cast<const ITensorGemmTileDataProvider *>(this);
            if (provider)
            {
                const size_t rows = shape()[0];
                const size_t cols = shape()[1];
                const size_t src_block_size = provider->block_size();
                const size_t blocks_per_row = (cols + src_block_size - 1) / src_block_size;

// OpenMP parallelize over rows
#pragma omp parallel for schedule(static)
                for (size_t r = 0; r < rows; ++r)
                {
                    for (size_t kb = 0; kb < blocks_per_row; ++kb)
                    {
                        // Decode source block to FP32
                        alignas(64) float fp32_block[256]; // Max block size (Q6_K = 256)
                        provider->decode_block_at(r, kb, fp32_block);

                        // Determine how many elements in this block
                        const size_t k_start = kb * src_block_size;
                        const size_t elements_remaining = cols - k_start;
                        const size_t elements_in_block = std::min(src_block_size, elements_remaining);

                        // Quantize to Q8_0 in chunks of 32 elements
                        for (size_t offset = 0; offset < elements_in_block; offset += Q8_BLOCK_SIZE)
                        {
                            const size_t chunk_size = std::min(Q8_BLOCK_SIZE, elements_in_block - offset);
                            const size_t global_block_idx = (r * cols + k_start + offset) / Q8_BLOCK_SIZE;

                            // Quantize this 32-element chunk
                            simd::quantize_fp32_to_q8_0(fp32_block + offset, chunk_size,
                                                        dst[global_block_idx].qs,
                                                        &dst[global_block_idx].d);

                            // Zero-pad if needed
                            if (chunk_size < Q8_BLOCK_SIZE)
                            {
                                std::memset(dst[global_block_idx].qs + chunk_size, 0, Q8_BLOCK_SIZE - chunk_size);
                            }
                        }
                    }
                }
                return;
            }
        }

        // Fallback: Decode to FP32 first, then quantize to Q8_0
        std::vector<float> fp32_buffer(total_elements);
        to_fp32(fp32_buffer.data());

#pragma omp parallel for schedule(static)
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const size_t offset = b * Q8_BLOCK_SIZE;
            const size_t chunk_size = std::min(Q8_BLOCK_SIZE, total_elements - offset);

            simd::quantize_fp32_to_q8_0(fp32_buffer.data() + offset, chunk_size,
                                        dst[b].qs, &dst[b].d);

            // Zero-pad if needed
            if (chunk_size < Q8_BLOCK_SIZE)
            {
                std::memset(dst[b].qs + chunk_size, 0, Q8_BLOCK_SIZE - chunk_size);
            }
        }
    }

    // Explicit template instantiations
    template void TensorBase::to<float>(float *dst, TensorType format) const;
    template void TensorBase::to<uint16_t>(uint16_t *dst, TensorType format) const;
    template void TensorBase::to<int8_t>(int8_t *dst, TensorType format) const;
    template void TensorBase::to<int32_t>(int32_t *dst, TensorType format) const;

    // =========================================================================
    // Pinned Memory Registration for Fast GPU Transfers
    // =========================================================================

    bool TensorBase::ensureHostPinned()
    {
        // Already pinned?
        if (host_pinned_)
        {
            return true;
        }

        // Get host data pointer and size
        void *host_ptr = raw_host_data_ptr();
        if (!host_ptr)
        {
            LOG_WARN("[TensorBase::ensureHostPinned] No host data to pin");
            return false;
        }

        size_t bytes = byte_size();
        if (bytes == 0)
        {
            return true; // Nothing to pin
        }

        // Try to register with the appropriate runtime
        bool success = false;

#ifdef HAVE_ROCM
        if (!success && gpu_device_.has_value() && gpu_device_->is_rocm())
        {
            // Ensure thread-local HIP device is set correctly before registration.
            // hipHostRegisterDefault (used inside hipHostRegisterBuffer) only registers
            // for the current device. The device should already be set by the caller
            // (ensureOnDevice → backend->allocate → hipSetDevice), but we set it
            // explicitly for safety in case ensureHostPinned is called from another path.
            hipSetDevice(gpu_device_->gpu_ordinal());
            success = host_backend_detail::hipHostRegisterBuffer(host_ptr, bytes);
            if (success)
            {
                LOG_DEBUG("[TensorBase::ensureHostPinned] Pinned " << bytes
                                                                   << " bytes of host memory for ROCm DMA transfers");
            }
        }
#endif

#ifdef HAVE_CUDA
        if (!success && gpu_device_.has_value() && gpu_device_->is_cuda())
        {
            success = host_backend_detail::cudaHostRegisterBuffer(host_ptr, bytes);
            if (success)
            {
                LOG_DEBUG("[TensorBase::ensureHostPinned] Pinned " << bytes
                                                                   << " bytes of host memory for CUDA DMA transfers");
            }
        }
#endif

        if (success)
        {
            host_pinned_ = true;
            pinned_bytes_ = bytes;
            pinned_host_ptr_ = host_ptr; // Store for unpinning in destructor
        }
        else
        {
            // Not an error - pinning is optional optimization
            LOG_TRACE("[TensorBase::ensureHostPinned] Could not pin host memory - "
                      "using pageable memory (slower GPU transfers)");
        }

        return success;
    }

    void TensorBase::unpinHostMemory()
    {
        if (!host_pinned_)
        {
            return; // Not pinned
        }

        // Use the stored pointer from when we pinned - avoids virtual call during destruction
        void *host_ptr = pinned_host_ptr_;
        if (!host_ptr)
        {
            host_pinned_ = false;
            pinned_bytes_ = 0;
            pinned_host_ptr_ = nullptr;
            return;
        }

#ifdef HAVE_ROCM
        if (gpu_device_.has_value() && gpu_device_->is_rocm())
        {
            host_backend_detail::hipHostUnregisterBuffer(host_ptr);
            LOG_TRACE("[TensorBase::unpinHostMemory] Unpinned " << pinned_bytes_
                                                                << " bytes of ROCm host memory");
        }
#endif

#ifdef HAVE_CUDA
        if (gpu_device_.has_value() && gpu_device_->is_cuda())
        {
            host_backend_detail::cudaHostUnregisterBuffer(host_ptr);
            LOG_TRACE("[TensorBase::unpinHostMemory] Unpinned " << pinned_bytes_
                                                                << " bytes of CUDA host memory");
        }
#endif

        host_pinned_ = false;
        pinned_bytes_ = 0;
        pinned_host_ptr_ = nullptr;
    }

    // =========================================================================
    // GPU Pointer Access with Trace Logging
    // =========================================================================

    void *TensorBase::gpu_data_ptr()
    {
        // TRACE: Log every GPU pointer access for debugging multi-GPU memory issues
        // Only log when pointer is non-null (i.e., tensor is on GPU)
        if (gpu_data_ptr_)
        {
            LOG_TRACE("[TensorBase::gpu_data_ptr] ACCESS tensor=" << static_cast<void *>(this)
                                                                  << " name=" << (debug_name_.empty() ? "(unnamed)" : debug_name_)
                                                                  << " ptr=" << gpu_data_ptr_
                                                                  << " device=" << (gpu_device_.has_value() ? gpu_device_->toString() : "none")
                                                                  << " device_valid=" << ::llaminar2::isDeviceValid(coherence_state_));
        }
        return gpu_data_ptr_;
    }

    const void *TensorBase::gpu_data_ptr() const
    {
        // TRACE: Log every GPU pointer access for debugging multi-GPU memory issues
        if (gpu_data_ptr_)
        {
            LOG_TRACE("[TensorBase::gpu_data_ptr] CONST ACCESS tensor=" << static_cast<const void *>(this)
                                                                        << " name=" << (debug_name_.empty() ? "(unnamed)" : debug_name_)
                                                                        << " ptr=" << gpu_data_ptr_
                                                                        << " device=" << (gpu_device_.has_value() ? gpu_device_->toString() : "none")
                                                                        << " device_valid=" << ::llaminar2::isDeviceValid(coherence_state_));
        }
        return gpu_data_ptr_;
    }

    // =========================================================================
    // Lazy Transfer Implementation (Phase 1 GPU Device-Aware Slicing)
    // =========================================================================

    // Default implementations for raw_host_data_ptr and byte_size
    // Tensors that support GPU transfer override these.
    void *TensorBase::raw_host_data_ptr()
    {
        throw std::runtime_error("[TensorBase] raw_host_data_ptr() not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    const void *TensorBase::raw_host_data_ptr() const
    {
        throw std::runtime_error("[TensorBase] raw_host_data_ptr() const not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    size_t TensorBase::byte_size() const
    {
        throw std::runtime_error("[TensorBase] byte_size() not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    void TensorBase::invalidateGpuData()
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // Mark GPU data as stale - next ensureOnDevice() will re-upload from host
        // Mark device as invalid (stale) - do NOT free GPU memory
        // This is called when host data is modified and GPU copy is now stale.
        // The GPU memory is kept allocated; next ensureOnDevice() will just re-upload.
        if (gpu_data_ptr_)
        {
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE); // Host now the only valid copy
            // CRITICAL WARNING: Repeated invalidation causes re-uploads!
            if (debugEnv().rocm.trace_coherence)
            {
                LOG_WARN("[TensorBase::invalidateGpuData] EXPENSIVE! Device data marked stale for ptr="
                         << static_cast<void *>(this) << " dtype=" << dtype_name()
                         << " gpu_ptr=" << gpu_data_ptr_ << " numel=" << numel());
            }
            LOG_DEBUG("[TensorBase::invalidateGpuData] Device data marked stale (memory retained)");
        }
    }

    bool TensorBase::ensureOnDevice(DeviceId target_device)
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // ===== GRAPH CAPTURE FAST PATH =====
        // During HIP/CUDA graph capture, synchronization operations are illegal.
        // Only check that data is already on device — warmup must have uploaded first.
        if (isGraphCaptureActive())
        {
            if (gpu_data_ptr_ && gpu_device_.has_value() &&
                *gpu_device_ == target_device && ::llaminar2::isDeviceValid(coherence_state_))
            {
                return true;
            }
            if (is_bar_backed_ || (is_mapped_ && mapped_device_ptr_ != nullptr))
            {
                return true;
            }
            LOG_ERROR("[TensorBase::ensureOnDevice] Called during graph capture but "
                      "tensor is NOT on device "
                      << target_device.toString()
                      << " — data must be uploaded during warmup phase first."
                      << " gpu_data_ptr=" << gpu_data_ptr_
                      << " device_valid=" << ::llaminar2::isDeviceValid(coherence_state_));
            return false;
        }

        // CPU devices: data is inherently on-host, nothing to transfer
        if (!target_device.is_gpu())
        {
            return true;
        }

        // Delegate to TransferEngine for all data movement
        auto result = TransferEngine::instance().uploadFull(this, target_device);
        if (!result.success)
        {
            LOG_ERROR("[TensorBase::ensureOnDevice] TransferEngine::uploadFull failed: " << result.error);
        }
        return result.success;
    }

    bool TensorBase::allocateOnDevice(DeviceId target_device)
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // CPU devices: host memory is the device, no allocation needed
        if (!target_device.is_gpu())
        {
            return true;
        }

        const bool trace = debugEnv().rocm.trace_coherence;

        // ===== BAR-BACKED TENSOR FAST PATH =====
        // BAR-backed tensors have memory allocated EXTERNALLY via PCIeBAR and are visible
        // to both CUDA and ROCm devices. They must NEVER be reallocated via backend->free/allocate.
        // The bar_backend_ field is set when the tensor is created as BAR-backed.
        //
        // IMPORTANT: is_bar_backed_ can be true in two scenarios:
        // 1. Tensor was initialized via initBARBackedDirect() with both pointers set up
        // 2. Tensor became BAR-backed via transferTo() where only the ROCm buffer is in BAR region
        //
        // For case 2, bar_cuda_device_ptr_ may be NULL, so we must fall through to normal path.
        if (is_bar_backed_)
        {
            // BAR memory is accessible from both devices via different pointers:
            // - bar_cuda_device_ptr_: CUDA-visible pointer to the BAR region
            // - bar_rocm_ptr_: ROCm-visible pointer to the same memory
            // We need to use the correct pointer for the target device type.

            if (target_device.is_cuda())
            {
                if (bar_cuda_device_ptr_)
                {
                    gpu_data_ptr_ = bar_cuda_device_ptr_;
                }
                else
                {
                    // BAR dual-pointer not set up (likely from transferTo()).
                    // Fall through to check secondary buffers or allocate new.
                    if (trace)
                    {
                        LOG_INFO("[TensorBase::allocateOnDevice] BAR-BACKED but bar_cuda_device_ptr_ is null, "
                                 "falling through to normal allocation path");
                    }
                    goto normal_allocation;
                }
            }
            else if (target_device.is_rocm())
            {
                // CRITICAL: For ROCm, use HIP staging buffer, NOT the BAR mmap address!
                // HIP kernels cannot dereference BAR mmap addresses (causes memory fault).
                // The staging buffer will be copied to BAR after kernel completes.
                if (hip_staging_ptr_)
                {
                    gpu_data_ptr_ = hip_staging_ptr_;
                    if (trace)
                    {
                        LOG_INFO("[TensorBase::allocateOnDevice] BAR-BACKED ROCm: Using HIP staging buffer "
                                 << hip_staging_ptr_ << " (BAR mmap=" << bar_rocm_ptr_ << ")");
                    }
                }
                else if (bar_rocm_ptr_)
                {
                    // Fallback to BAR mmap (may crash if used by HIP kernels)
                    LOG_WARN("[TensorBase::allocateOnDevice] BAR-BACKED ROCm: No HIP staging buffer! "
                             "Falling back to BAR mmap address (may crash)");
                    gpu_data_ptr_ = bar_rocm_ptr_;
                }
                else
                {
                    // Neither staging nor BAR pointer available - fall through
                    if (trace)
                    {
                        LOG_INFO("[TensorBase::allocateOnDevice] BAR-BACKED but no ROCm pointer available, "
                                 "falling through to normal allocation path");
                    }
                    goto normal_allocation;
                }
            }
            else
            {
                LOG_ERROR("[TensorBase::allocateOnDevice] BAR-backed tensor cannot target device type: "
                          << target_device.toString());
                return false;
            }

            gpu_device_ = target_device;
            // Mark device as invalid - kernel will write to this buffer
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
            if (trace)
            {
                LOG_INFO("[TensorBase::allocateOnDevice] BAR-BACKED: Switching to "
                         << target_device.toString() << " ptr=" << gpu_data_ptr_);
            }
            return true;
        }

    normal_allocation:

        // ===== ZERO-COPY MAPPED MEMORY FAST PATH =====
        // If tensor uses mapped memory, GPU can access host memory directly.
        // No allocation needed - just ensure we're tracking the target device.
        if (is_mapped_ && mapped_device_ptr_ != nullptr)
        {
            if (!gpu_device_.has_value() || *gpu_device_ != target_device)
            {
                gpu_device_ = target_device;
            }
            if (gpu_data_ptr_ != mapped_device_ptr_)
            {
                gpu_data_ptr_ = mapped_device_ptr_;
            }
            // Mapped memory: both host and device are always valid
            setCoherenceState_(TensorCoherenceState::MAPPED);

            if (trace)
            {
                LOG_INFO("[TensorBase::allocateOnDevice] ZERO-COPY: Tensor is mapped, no allocation needed");
            }
            return true;
        }

        // Check if already allocated on target device - reuse existing allocation
        if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ == target_device)
        {
            // Already allocated on correct device - mark as invalid (kernel will overwrite)
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
            if (trace)
            {
                LOG_INFO("[TensorBase::allocateOnDevice] Reusing existing allocation on " << target_device.toString());
            }
            return true;
        }

        // Get backend for target device
        IBackend *target_backend = resolveBackend(target_device);
        if (!target_backend)
        {
            LOG_ERROR("[TensorBase::allocateOnDevice] No backend available for device " << target_device.toString());
            return false;
        }

        int backend_device_id = target_device.gpu_ordinal();

        // ===== CHECK SECONDARY BUFFERS FIRST =====
        // If we have a buffer in secondary_device_buffers_ for the target device,
        // promote it to primary instead of allocating new memory.
        // This is critical for PP mode where tensors ping-pong between devices.
        {
            int target_key = packDeviceId(target_device);
            auto sec_it = secondary_device_buffers_.find(target_key);
            if (sec_it != secondary_device_buffers_.end() && sec_it->second != nullptr)
            {
                // Store current primary in secondary (if not already there)
                if (gpu_data_ptr_ && gpu_device_.has_value())
                {
                    int old_key = packDeviceId(*gpu_device_);
                    if (secondary_device_buffers_.find(old_key) == secondary_device_buffers_.end())
                    {
                        secondary_device_buffers_[old_key] = gpu_data_ptr_;
                        // If current primary was BAR-backed, track it in secondary
                        if (is_bar_backed_)
                        {
                            secondary_bar_allocated_keys_.insert(old_key);
                        }
                    }
                }

                // Promote secondary to primary
                void *promoted_ptr = sec_it->second;
                secondary_device_buffers_.erase(sec_it);

                gpu_data_ptr_ = promoted_ptr;
                gpu_device_ = target_device;
                setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE); // Promoted but data stale

                // Check if promoted buffer was BAR-backed in secondary
                bool was_bar = (secondary_bar_allocated_keys_.count(target_key) > 0);
                if (was_bar)
                {
                    secondary_bar_allocated_keys_.erase(target_key);
                }
                // CRITICAL: Update is_bar_backed_ to reflect the promoted buffer's status.
                // The promoted buffer is only BAR-backed if it was tracked as such in secondary.
                // A regular CUDA buffer promoted from secondary is NOT BAR-backed.
                is_bar_backed_ = was_bar;

                if (trace)
                {
                    LOG_INFO("[TensorBase::allocateOnDevice] Promoted secondary buffer to primary for "
                             << target_device.toString() << " ptr=" << promoted_ptr
                             << (was_bar ? " (BAR-backed)" : " (regular)"));
                }
                return true;
            }
        }

        // Free existing device memory if on different device
        if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ != target_device)
        {
            IBackend *old_backend = resolveBackend(*gpu_device_);
            int old_backend_device_id = gpu_device_->gpu_ordinal();
            if (old_backend)
            {
                old_backend->free(gpu_data_ptr_, old_backend_device_id);
            }
            gpu_data_ptr_ = nullptr;
            gpu_device_.reset();
            setCoherenceState_(TensorCoherenceState::HOST_ONLY); // GPU memory freed
        }

        // Allocate on target device
        size_t bytes = byte_size();
        if (!gpu_data_ptr_)
        {
            auto alloc_start = std::chrono::high_resolution_clock::now();
            gpu_data_ptr_ = target_backend->allocate(bytes, backend_device_id);
            auto alloc_end = std::chrono::high_resolution_clock::now();
            auto alloc_us = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();

            if (trace)
            {
                LOG_INFO("[TensorBase::allocateOnDevice] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            if (!gpu_data_ptr_)
            {
                LOG_ERROR("[TensorBase::allocateOnDevice] Failed to allocate " << bytes
                                                                               << " bytes on device " << target_device.toString()
                                                                               << " (backend device ID: " << backend_device_id << ")");
                return false;
            }

            gpu_device_ = target_device;
            // Kernel will write to this buffer — host still valid
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

            LOG_DEBUG("[TensorBase::allocateOnDevice] Allocated " << bytes
                                                                  << " bytes on device " << target_device.toString()
                                                                  << " (NO H2D upload - output buffer)");
        }

        return true;
    }

    // =========================================================================
    // Helper: Wait for CUDA event with cross-thread proxy support
    bool TensorBase::ensureOnHost()
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // Delegate to TransferEngine for all data movement
        auto result = TransferEngine::instance().downloadFull(this);
        if (!result.success)
        {
            LOG_ERROR("[TensorBase::ensureOnHost] TransferEngine::downloadFull failed: " << result.error);
        }
        return result.success;
    }

    void TensorBase::transitionToWithEvent(TensorCoherenceState new_state,
                                           std::optional<DeviceId> authoritative_dev,
                                           void *stream)
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // --- State transition (same as transitionTo) ---
        setCoherenceState_(new_state);
        if (new_state == TensorCoherenceState::DEVICE_AUTHORITATIVE ||
            new_state == TensorCoherenceState::MAPPED)
        {
            authoritative_device_ = authoritative_dev.value_or(gpu_device_.value_or(DeviceId::cpu()));
        }
        else if (new_state == TensorCoherenceState::HOST_ONLY ||
                 new_state == TensorCoherenceState::HOST_AUTHORITATIVE)
        {
            authoritative_device_.reset();
        }

        // For mapped memory: signal that sync is needed before CPU reads
        if (is_mapped_)
        {
            mapped_needs_sync_ = true;
        }

        // --- GPU event recording ---
        if (!gpu_device_.has_value())
            return;

        IBackend *backend = resolveBackend(*gpu_device_);
        if (!backend)
            return;

        // Defensive check: verify resolved backend type matches tensor's device.
        // In PP mode a tensor may migrate CUDA→ROCm; recording a CUDA stream event
        // on a ROCm backend would segfault.
        if (stream && backend->backendDeviceType() != gpu_device_->type)
        {
            LOG_ERROR("[TensorBase::transitionToWithEvent] CROSS-BACKEND MISMATCH: "
                      << "tensor gpu_device_=" << gpu_device_->toString()
                      << " but resolved backend type=" << static_cast<int>(backend->backendDeviceType())
                      << " — skipping event recording to avoid crash");
            return;
        }

        int backend_device_id = gpu_device_->gpu_ordinal();

        // Existing event on a different device — must recreate
        if (device_completion_event_ && event_device_.has_value() && *event_device_ != *gpu_device_)
        {
            // Don't destroy — it was created on a different backend. Leak is safer than crash.
            device_completion_event_ = nullptr;
            event_device_.reset();
        }

        // Create event if needed
        if (!device_completion_event_)
        {
            device_completion_event_ = backend->createEvent(backend_device_id);
            if (device_completion_event_)
            {
                event_device_ = *gpu_device_;
            }
            else
            {
                LOG_WARN("[TensorBase::transitionToWithEvent] Failed to create event on device "
                         << gpu_device_->toString());
                return;
            }
        }

        // Record the event on the stream
        if (!backend->recordEvent(device_completion_event_, backend_device_id, stream))
        {
            LOG_WARN("[TensorBase::transitionToWithEvent] Failed to record event");
        }
    }

    void TensorBase::mark_device_dirty_with_event(void *stream)
    {
        // Delegate to the state-machine-aware implementation
        transitionToWithEvent(
            is_mapped_ ? TensorCoherenceState::MAPPED : TensorCoherenceState::DEVICE_AUTHORITATIVE,
            gpu_device_,
            stream);
    }

    bool TensorBase::releaseDeviceMemory()
    {
        // Ensure host has current data
        if (!ensureOnHost())
        {
            return false;
        }

        // Free device memory
        if (gpu_data_ptr_ && gpu_device_.has_value())
        {
            IBackend *backend = resolveBackend(*gpu_device_);
            if (backend)
            {
                int backend_device_id = gpu_device_->gpu_ordinal();

                // Destroy completion event if it exists
                if (device_completion_event_)
                {
                    backend->destroyEvent(device_completion_event_, backend_device_id);
                    device_completion_event_ = nullptr;
                }

                backend->free(gpu_data_ptr_, backend_device_id);
            }
            LOG_DEBUG("[TensorBase::releaseDeviceMemory] Released device memory on device "
                      << gpu_device_->toString());
            gpu_data_ptr_ = nullptr;
            setCoherenceState_(TensorCoherenceState::HOST_ONLY); // GPU memory freed

            // Unpin host memory since we no longer need fast GPU transfers
            unpinHostMemory();

            gpu_device_.reset();
        }

        return true;
    }

    // =====================================================================
    // Tensor Comparison Utilities Implementation
    // =====================================================================

    double TensorBase::cosineSimilarityTo(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("cosineSimilarityTo: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "cosineSimilarityTo: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        if (n == 0)
        {
            return 0.0; // Empty tensors
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        // Compute dot product and norms in parallel
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

#pragma omp parallel for reduction(+ : dot, norm_a, norm_b) schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            double va = static_cast<double>(a[i]);
            double vb = static_cast<double>(b[i]);
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
        {
            return std::numeric_limits<double>::quiet_NaN(); // At least one tensor is zero
        }

        return dot / denom;
    }

    float TensorBase::maxAbsDiffTo(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("maxAbsDiffTo: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "maxAbsDiffTo: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        if (n == 0)
        {
            return 0.0f;
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        float max_diff = 0.0f;

#pragma omp parallel for reduction(max : max_diff) schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::fabs(a[i] - b[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }

        return max_diff;
    }

    float TensorBase::meanAbsDiffTo(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("meanAbsDiffTo: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "meanAbsDiffTo: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        if (n == 0)
        {
            return 0.0f;
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        double sum_diff = 0.0;

#pragma omp parallel for reduction(+ : sum_diff) schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            sum_diff += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }

        return static_cast<float>(sum_diff / static_cast<double>(n));
    }

    double TensorBase::relativeL2To(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("relativeL2To: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "relativeL2To: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        if (n == 0)
        {
            return 0.0;
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        double diff_sq = 0.0;
        double ref_sq = 0.0;

#pragma omp parallel for reduction(+ : diff_sq, ref_sq) schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            double va = static_cast<double>(a[i]);
            double vb = static_cast<double>(b[i]);
            double d = va - vb;
            diff_sq += d * d;
            ref_sq += vb * vb;
        }

        if (ref_sq < 1e-24)
        {
            return std::numeric_limits<double>::infinity();
        }

        return std::sqrt(diff_sq) / std::sqrt(ref_sq);
    }

    double TensorBase::klDivergenceTo(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("klDivergenceTo: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "klDivergenceTo: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        if (n == 0)
        {
            return 0.0;
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        // Apply softmax to convert logits to probabilities
        // First, find max for numerical stability
        double max_a = a[0];
        double max_b = b[0];
        for (size_t i = 1; i < n; ++i)
        {
            if (a[i] > max_a)
                max_a = a[i];
            if (b[i] > max_b)
                max_b = b[i];
        }

        // Compute softmax denominators
        double sum_exp_a = 0.0;
        double sum_exp_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum_exp_a += std::exp(static_cast<double>(a[i]) - max_a);
            sum_exp_b += std::exp(static_cast<double>(b[i]) - max_b);
        }

        // Compute KL divergence: KL(P || Q) = Σ P(i) × log(P(i) / Q(i))
        // = Σ P(i) × (log P(i) - log Q(i))
        // where P = softmax(a), Q = softmax(b)
        double kl = 0.0;
        constexpr double epsilon = 1e-10; // Prevent log(0)

        for (size_t i = 0; i < n; ++i)
        {
            double p = std::exp(static_cast<double>(a[i]) - max_a) / sum_exp_a;
            double q = std::exp(static_cast<double>(b[i]) - max_b) / sum_exp_b;

            if (p > epsilon)
            {
                // log(P/Q) = log(P) - log(Q)
                // = (a[i] - max_a - log(sum_exp_a)) - (b[i] - max_b - log(sum_exp_b))
                double log_p = static_cast<double>(a[i]) - max_a - std::log(sum_exp_a);
                double log_q = static_cast<double>(b[i]) - max_b - std::log(sum_exp_b);
                kl += p * (log_p - log_q);
            }
        }

        return kl;
    }

    TensorBase::ComparisonSummary TensorBase::compareTo(const TensorBase *other) const
    {
        if (!other)
        {
            throw std::invalid_argument("compareTo: other tensor is null");
        }

        const size_t n = element_count();
        if (n != other->element_count())
        {
            throw std::invalid_argument(
                "compareTo: element count mismatch (" +
                std::to_string(n) + " vs " + std::to_string(other->element_count()) + ")");
        }

        ComparisonSummary summary{};

        if (n == 0)
        {
            summary.cosine_similarity = 0.0;
            summary.max_abs_diff = 0.0f;
            summary.mean_abs_diff = 0.0f;
            summary.relative_l2 = 0.0;
            return summary;
        }

        const float *a = fp32_data();
        const float *b = other->fp32_data();

        // Compute all metrics in a single pass for efficiency
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        double diff_sq = 0.0;
        double sum_diff = 0.0;
        float max_diff = 0.0f;

#pragma omp parallel
        {
            double local_dot = 0.0;
            double local_norm_a = 0.0;
            double local_norm_b = 0.0;
            double local_diff_sq = 0.0;
            double local_sum_diff = 0.0;
            float local_max_diff = 0.0f;

#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
            {
                double va = static_cast<double>(a[i]);
                double vb = static_cast<double>(b[i]);
                double d = va - vb;
                float abs_d = std::fabs(a[i] - b[i]);

                local_dot += va * vb;
                local_norm_a += va * va;
                local_norm_b += vb * vb;
                local_diff_sq += d * d;
                local_sum_diff += abs_d;
                if (abs_d > local_max_diff)
                    local_max_diff = abs_d;
            }

#pragma omp critical
            {
                dot += local_dot;
                norm_a += local_norm_a;
                norm_b += local_norm_b;
                diff_sq += local_diff_sq;
                sum_diff += local_sum_diff;
                if (local_max_diff > max_diff)
                    max_diff = local_max_diff;
            }
        }

        // Cosine similarity
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
        {
            summary.cosine_similarity = std::numeric_limits<double>::quiet_NaN();
        }
        else
        {
            summary.cosine_similarity = dot / denom;
        }

        // Max abs diff
        summary.max_abs_diff = max_diff;

        // Mean abs diff
        summary.mean_abs_diff = static_cast<float>(sum_diff / static_cast<double>(n));

        // Relative L2
        if (norm_b < 1e-24)
        {
            summary.relative_l2 = std::numeric_limits<double>::infinity();
        }
        else
        {
            summary.relative_l2 = std::sqrt(diff_sq) / std::sqrt(norm_b);
        }

        return summary;
    }

    // ========================================================================
    // hasCachedDeviceData - Check for kernel-cached device representations
    // ========================================================================

    bool TensorBase::hasCachedDeviceData(DeviceType device_type) const
    {
        // Check if we have cached packed weights for the given device type
        // These caches are populated by KernelFactory when creating GEMM kernels
        // and contain converted/packed weight data that's already on device.

        if (device_type == DeviceType::CUDA)
        {
#ifdef HAVE_CUDA
            if (cuda_cache_.has_value())
            {
                // The cache contains TensorCUDAPackedWeightsCache* which wraps CUDAPackedWeights
                // We can't include the header here, but we know if cuda_cache_ has a value,
                // the packing was done. Check if uploaded by examining the value.
                try
                {
                    // Use the packed weights header type to check upload status
                    // This relies on KernelFactory storing the cache in a known format
                    // For now, if cuda_cache_ has value, assume it's been packed
                    // The actual upload status is tracked in CUDAPackedWeights::uploaded
                    return true; // Cache exists = packed weights ready
                }
                catch (...)
                {
                    return false;
                }
            }
#endif
            return false;
        }
        else if (device_type == DeviceType::ROCm)
        {
#ifdef HAVE_ROCM
            if (rocm_cache_.has_value())
            {
                return true; // Cache exists = packed weights ready
            }
#endif
            return false;
        }

        return false; // CPU doesn't use device caching
    }

    // =========================================================================
    // Direct GPU-to-GPU Transfer (Phase 2 GPU-Native Coherence)
    // =========================================================================

    bool TensorBase::transferTo(DeviceId dst_device, size_t bytes_override)
    {
        // 1. Validate preconditions
        if (!authoritative_device_.has_value())
        {
            LOG_ERROR("[TensorBase::transferTo] Tensor has no authoritative GPU device. "
                      "Call ensureOnDevice() + mark_device_dirty() first.");
            return false;
        }

        DeviceId src_device = *authoritative_device_;

        // Same device = no-op success
        if (src_device == dst_device)
        {
            LOG_DEBUG("[TensorBase::transferTo] Same device (" << src_device.toString()
                                                               << "), no transfer needed");
            return true;
        }

        // Must be GPU to GPU
        if (src_device.is_cpu() || dst_device.is_cpu())
        {
            LOG_ERROR("[TensorBase::transferTo] Only GPU-to-GPU transfers supported. "
                      "Got: "
                      << src_device.toString() << " -> " << dst_device.toString());
            return false;
        }

        // 2. Get backend that supports this transfer
        auto *router = GlobalBackendRouter::get();
        if (!router)
        {
            LOG_ERROR("[TensorBase::transferTo] GlobalBackendRouter not initialized");
            return false;
        }

        ICollectiveBackend *backend = router->getBackendForCopy(src_device, dst_device);
        if (!backend)
        {
            LOG_ERROR("[TensorBase::transferTo] No backend registered for "
                      << src_device.toString() << " -> " << dst_device.toString());
            return false;
        }

        if (!backend->supportsCopy(src_device, dst_device))
        {
            LOG_ERROR("[TensorBase::transferTo] Backend does not support "
                      << src_device.toString() << " -> " << dst_device.toString()
                      << " (fail-fast, no host fallback)");
            return false;
        }

        // 3. Ensure source buffer pointer is valid
        void *src_ptr = gpu_data_ptr_;
        if (!src_ptr)
        {
            LOG_ERROR("[TensorBase::transferTo] Source GPU buffer is null");
            return false;
        }

        // 4. Get or allocate destination buffer
        void *dst_ptr = getOrAllocateDeviceBuffer(dst_device);
        if (!dst_ptr)
        {
            LOG_ERROR("[TensorBase::transferTo] Failed to allocate buffer on "
                      << dst_device.toString());
            return false;
        }

        // Check if the destination buffer is BAR-allocated
        // (either just allocated, or previously stored as BAR-allocated in secondary)
        int dst_key = packDeviceId(dst_device);
        bool dst_is_bar_backed = (secondary_bar_allocated_keys_.count(dst_key) > 0);

        // 5. Get transfer size (use override if provided, otherwise full buffer)
        size_t bytes = (bytes_override > 0 && bytes_override <= byte_size()) ? bytes_override : byte_size();
        if (bytes == 0)
        {
            LOG_WARN("[TensorBase::transferTo] Zero-byte tensor, nothing to transfer");
            return true;
        }

        LOG_DEBUG("[TensorBase::transferTo] " << src_device.toString() << " -> "
                                              << dst_device.toString() << " (" << bytes << " bytes)");

        // 6. Perform transfer - special handling for BAR-backed tensors
        // For BAR-backed ROCm → CUDA transfers:
        //   1. Copy from HIP staging buffer to BAR region (hipMemcpy D2D)
        //   2. CUDA already has access to BAR via bar_cuda_device_ptr_
        if (is_bar_backed_ && src_device.is_rocm() && dst_device.is_cuda() &&
            hip_staging_ptr_ && bar_rocm_ptr_)
        {
            LOG_DEBUG("[TensorBase::transferTo] BAR-backed ROCm→CUDA: copying staging→BAR");

            // Step 1: Copy from HIP staging buffer to BAR region
            // This uses hipMemcpy D2D which works because the BAR mmap address
            // is recognized by the HIP runtime as device memory
#if defined(HAVE_ROCM)
            hipError_t hip_err = hipSetDevice(src_device.rocm_ordinal());
            if (hip_err != hipSuccess)
            {
                LOG_ERROR("[TensorBase::transferTo] hipSetDevice failed: " << hipGetErrorString(hip_err));
                return false;
            }

            hip_err = hipMemcpy(bar_rocm_ptr_, hip_staging_ptr_, bytes, hipMemcpyDeviceToDevice);
            if (hip_err != hipSuccess)
            {
                LOG_ERROR("[TensorBase::transferTo] hipMemcpy staging→BAR failed: " << hipGetErrorString(hip_err));
                return false;
            }

            // hipDeviceSynchronize to ensure the copy completes before CUDA reads
            hip_err = hipDeviceSynchronize();
            if (hip_err != hipSuccess)
            {
                LOG_ERROR("[TensorBase::transferTo] hipDeviceSynchronize failed: " << hipGetErrorString(hip_err));
                return false;
            }

            LOG_DEBUG("[TensorBase::transferTo] BAR-backed ROCm→CUDA: staging→BAR complete, "
                      << "CUDA can now read from BAR at " << bar_cuda_device_ptr_);

            // Step 2: Update destination to point to BAR-accessible CUDA pointer
            // The CUDA device can directly read from bar_cuda_device_ptr_
            dst_ptr = bar_cuda_device_ptr_;
            dst_is_bar_backed = true;
#else
            LOG_ERROR("[TensorBase::transferTo] BAR-backed transfer requires HAVE_ROCM");
            return false;
#endif
        }
        else
        {
            // Standard backend copy for non-BAR or same-vendor transfers
            if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes))
            {
                LOG_ERROR("[TensorBase::transferTo] Backend copy failed");
                return false;
            }
        }

        // 7. Update coherence state
        // Store old primary buffer in secondary map if not already there
        if (gpu_device_.has_value() && *gpu_device_ != dst_device)
        {
            int old_key = packDeviceId(*gpu_device_);
            if (secondary_device_buffers_.find(old_key) == secondary_device_buffers_.end())
            {
                secondary_device_buffers_[old_key] = gpu_data_ptr_;
                // If old primary was BAR-backed, track it in secondary_bar_allocated_keys_
                if (is_bar_backed_)
                {
                    secondary_bar_allocated_keys_.insert(old_key);
                }
            }
        }

        // Remove destination buffer from secondary since it's now primary
        // (prevents double-free in destructor)
        secondary_device_buffers_.erase(dst_key);
        // Note: we keep dst_key in secondary_bar_allocated_keys_ if it was BAR-backed
        // because is_bar_backed_ tracks primary status, not secondary

        // Update primary buffer to destination
        gpu_data_ptr_ = dst_ptr;
        gpu_device_ = dst_device;
        authoritative_device_ = dst_device;
        setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE); // Device now authoritative, host stale
        is_bar_backed_ = dst_is_bar_backed;                             // Track if new primary is BAR-backed

        // 8. Clear completion event - events are tied to the device/context where created
        // Must clear for ANY cross-device transfer (not just cross-vendor), because CUDA
        // events created on GPU:0 cannot be recorded on GPU:1's stream.
        clearCompletionEvent();

        LOG_DEBUG("[TensorBase::transferTo] Transfer completed, "
                  << dst_device.toString() << " is now authoritative"
                  << (is_bar_backed_ ? " (BAR-backed)" : ""));
        return true;
    }

    bool TensorBase::copyTo(DeviceId dst_device)
    {
        // Similar to transferTo but doesn't change authoritative
        if (!authoritative_device_.has_value())
        {
            LOG_ERROR("[TensorBase::copyTo] Tensor has no authoritative GPU device");
            return false;
        }

        DeviceId src_device = *authoritative_device_;
        if (src_device == dst_device)
        {
            return true; // Already there
        }

        if (src_device.is_cpu() || dst_device.is_cpu())
        {
            LOG_ERROR("[TensorBase::copyTo] Only GPU-to-GPU supported");
            return false;
        }

        auto *router = GlobalBackendRouter::get();
        if (!router)
        {
            LOG_ERROR("[TensorBase::copyTo] GlobalBackendRouter not initialized");
            return false;
        }

        ICollectiveBackend *backend = router->getBackendForCopy(src_device, dst_device);
        if (!backend || !backend->supportsCopy(src_device, dst_device))
        {
            LOG_ERROR("[TensorBase::copyTo] No backend supports this transfer");
            return false;
        }

        void *src_ptr = gpu_data_ptr_;
        if (!src_ptr)
        {
            LOG_ERROR("[TensorBase::copyTo] Source GPU buffer is null");
            return false;
        }

        void *dst_ptr = getOrAllocateDeviceBuffer(dst_device);
        if (!dst_ptr)
        {
            LOG_ERROR("[TensorBase::copyTo] Failed to allocate destination buffer");
            return false;
        }

        size_t bytes = byte_size();
        if (bytes == 0)
        {
            return true;
        }

        if (!backend->copy(dst_ptr, dst_device, src_ptr, src_device, bytes))
        {
            LOG_ERROR("[TensorBase::copyTo] Backend copy failed");
            return false;
        }

        // NOTE: Unlike transferTo(), we do NOT change authoritative_device_
        // The source device remains authoritative, destination is just a copy

        LOG_DEBUG("[TensorBase::copyTo] Copied to " << dst_device.toString()
                                                    << ", source " << src_device.toString() << " remains authoritative");
        return true;
    }

    void *TensorBase::getOrAllocateDeviceBuffer(DeviceId device)
    {
        // Check if this is the current primary device
        if (gpu_device_.has_value() && *gpu_device_ == device)
        {
            return gpu_data_ptr_;
        }

        // Check secondary buffers
        int key = packDeviceId(device);
        auto it = secondary_device_buffers_.find(key);
        if (it != secondary_device_buffers_.end() && it->second != nullptr)
        {
            return it->second;
        }

        // Need to allocate new buffer
        size_t bytes = byte_size();
        if (bytes == 0)
        {
            LOG_ERROR("[TensorBase::getOrAllocateDeviceBuffer] Cannot allocate 0 bytes");
            return nullptr;
        }

        // For cross-vendor transfers (CUDA↔ROCm), ROCm buffers need to be
        // allocated in the BAR region for PCIe BAR P2P to work.
        // Check if we need BAR-region allocation.
        bool need_bar_allocation = false;
        if (device.is_rocm() && authoritative_device_.has_value() && authoritative_device_->is_cuda())
        {
            // Destination is ROCm, source is CUDA → need BAR allocation
            need_bar_allocation = true;
        }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (need_bar_allocation)
        {
            // Try to allocate in BAR region via PCIeBAR backend
            auto *router = GlobalBackendRouter::get();
            if (router)
            {
                auto *backend = router->getBackend(CollectiveBackendType::PCIE_BAR);
                if (backend)
                {
                    auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend);
                    if (pcie_backend && pcie_backend->isPCIeBarActive())
                    {
                        auto bar_alloc = pcie_backend->allocateInBarRegion(bytes);
                        if (bar_alloc.has_value())
                        {
                            void *bar_ptr = bar_alloc->first;
                            secondary_device_buffers_[key] = bar_ptr;
                            secondary_bar_allocated_keys_.insert(key); // Track as BAR-allocated
                            LOG_DEBUG("[TensorBase::getOrAllocateDeviceBuffer] Allocated " << bytes
                                                                                           << " bytes in BAR region for " << device.toString());
                            return bar_ptr;
                        }
                        else
                        {
                            LOG_WARN("[TensorBase::getOrAllocateDeviceBuffer] BAR allocation failed, "
                                     "falling back to standard allocation");
                        }
                    }
                }
            }
        }
#endif

        // Standard allocation via device backend
        IBackend *backend = resolveBackend(device);
        if (!backend)
        {
            LOG_ERROR("[TensorBase::getOrAllocateDeviceBuffer] No backend for device "
                      << device.toString());
            return nullptr;
        }

        int backend_device_id = device.gpu_ordinal();
        void *new_ptr = backend->allocate(bytes, backend_device_id);

        if (new_ptr)
        {
            secondary_device_buffers_[key] = new_ptr;
            LOG_DEBUG("[TensorBase::getOrAllocateDeviceBuffer] Allocated " << bytes
                                                                           << " bytes on " << device.toString());
        }
        else
        {
            LOG_ERROR("[TensorBase::getOrAllocateDeviceBuffer] Allocation failed on "
                      << device.toString());
        }

        return new_ptr;
    }

} // namespace llaminar2
