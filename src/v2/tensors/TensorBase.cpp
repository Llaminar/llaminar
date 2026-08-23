/**
 * @file TensorBase.cpp
 * @brief Tensor storage, coherence, and transfer-event lifetime implementation.
 *
 * TensorBase owns host and device allocations while TransferEngine owns every
 * state transition that accompanies movement between them. This file enforces
 * the complementary lifetime rule: host storage cannot be mutated, unpinned,
 * or destroyed while an asynchronous GPU upload still reads it. The exact
 * copy-completion event is retired at that host reuse boundary; unrelated GPU
 * streams are never drained.
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
#include "../utils/VramBillOfMaterials.h"
#include "../backends/BackendManager.h"
#include "../backends/ComputeBackend.h"
#include "../backends/DeviceId.h"
#include "../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../transfer/TransferEngine.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <omp.h>

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
    // Also frees GPU memory and unpins host memory if allocated.
    TensorBase::~TensorBase()
    {
        /*
         * Concrete destructors already close this transition before their
         * storage members disappear. Keep the base call as an idempotent guard
         * for mapped or storage-free TensorBase implementations.
         */
        retireHostTransferLifetimeBeforeStorageDestruction();

        // Free mapped memory if allocated (must be done first)
        freeMappedMemory();

        // Free secondary device buffers (from multi-device transfers)
        for (auto &[key, ptr] : secondary_device_buffers_)
        {
            if (ptr != nullptr)
            {
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

        // Free GPU memory if allocated
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
                    event_device_.reset();
                    completion_event_protection_ =
                        CompletionEventProtection::None;
                }

                backend->free(gpu_data_ptr_, backend_device_id);
            }
            gpu_data_ptr_ = nullptr;
            applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE); // GPU memory freed
        }

        // Phase 10: TensorBase destructor NEVER touches global KernelFactory state.
        // Cleanup of KernelFactory registries is the exclusive responsibility of
        // PreparedWeightStore::releaseAllPreparedState() (called during orchestrator
        // shutdown). Only clear local packed-weights cache here.
        if (cache_.has_value())
        {
            std::lock_guard<std::mutex> lock(packed_cache_mutex_);
            cache_.reset();
        }
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

    void TensorBase::retireHostTransferLifetimeBeforeStorageDestruction() noexcept
    {
        try
        {
            /*
             * Async H2D may outlive the submitting host frame, but it may not
             * outlive the concrete vector that supplies its bytes. Wait only
             * the exact copy event before removing the runtime registration;
             * a stream/device synchronization would also drain unrelated work.
             */
            TransferEngine::waitForPendingHostSourceUseLocked(this);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[TensorBase] Cannot retire in-flight H2D host source before storage destruction: "
                      << error.what());
            std::terminate();
        }

        unpinHostMemory();
    }

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

        if (!gpu_device_.has_value() || !gpu_device_->is_gpu())
            return false;
        IBackend *const backend = resolveBackend(*gpu_device_);
        const bool success = backend && backend->pinHostMemory(
                                            host_ptr,
                                            bytes,
                                            gpu_device_->gpu_ordinal());

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

        if (gpu_device_.has_value() && gpu_device_->is_gpu())
        {
            IBackend *const backend = resolveBackend(*gpu_device_);
            if (!backend || !backend->unpinHostMemory(
                                host_ptr, gpu_device_->gpu_ordinal()))
            {
                LOG_ERROR("[TensorBase::unpinHostMemory] Failed to unregister "
                          << pinned_bytes_ << " bytes at " << host_ptr
                          << " from " << gpu_device_->toString());
                std::terminate();
            }
            LOG_TRACE("[TensorBase::unpinHostMemory] Unpinned " << pinned_bytes_
                                                                << " bytes at " << host_ptr
                                                                << " from " << gpu_device_->toString());
        }

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

        /*
         * mutable_data()/raw_mutable_data() reuse the host allocation.  If its
         * previous generation is still feeding an async H2D, only that copy's
         * completion event can release the source lifetime safely.
         */
        TransferEngine::waitForPendingHostSourceUseLocked(this);
        retireCompletionEvent_();

        // Mark GPU data as stale - next ensureOnDevice() will re-upload from host
        // Mark device as invalid (stale) - do NOT free GPU memory
        // This is called when host data is modified and GPU copy is now stale.
        // The GPU memory is kept allocated; next ensureOnDevice() will just re-upload.
        if (gpu_data_ptr_)
        {
            applyCoherenceOp_(CoherenceOp::MUTABLE_HOST_ACCESS); // Host now the only valid copy
            // CRITICAL WARNING: Repeated invalidation causes re-uploads!
            if (debugEnv().rocm.trace_coherence)
            {
                LOG_WARN("[TensorBase::invalidateGpuData] EXPENSIVE! Device data marked stale for ptr="
                         << static_cast<void *>(this) << " dtype=" << dtype_name()
                         << " gpu_ptr=" << gpu_data_ptr_ << " numel=" << numel());
            }
            LOG_TRACE("[TensorBase::invalidateGpuData] Device data marked stale (memory retained)");
        }
    }

    void TensorBase::publishHostWriteState()
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        /*
         * A queued H2D copy still reads the host allocation even though both
         * logical copies already contain the same generation. An external host
         * writer may not reuse that storage until the exact copy-completion
         * event has fired. TransferEngine owns the wait and event retirement so
         * this tensor hook cannot substitute a stream-wide synchronization.
         */
        TransferEngine::waitForPendingHostSourceUseLocked(this);
        retireCompletionEvent_();
        if (is_mapped_)
        {
            setCoherenceState_(TensorCoherenceState::MAPPED);
            mapped_needs_sync_ = false;
        }
        else
        {
            setCoherenceState_(gpu_data_ptr_ ? TensorCoherenceState::HOST_AUTHORITATIVE
                                             : TensorCoherenceState::HOST_ONLY);
        }
        authoritative_device_.reset();
    }

    bool TensorBase::ensureOnDevice(DeviceId target_device, void *stream)
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
            if (is_mapped_ && mapped_device_ptr_ != nullptr)
            {
                return true;
            }
            LOG_ERROR("[TensorBase::ensureOnDevice] Called during graph capture but "
                      "tensor is NOT on device "
                      << target_device.toString()
                      << " — data must be uploaded during warmup phase first."
                      << " tensor=" << static_cast<const void *>(this)
                      << " dtype=" << dtype_name()
                      << " numel=" << numel()
                      << " gpu_data_ptr=" << gpu_data_ptr_
                      << " gpu_device="
                      << (gpu_device_.has_value()
                              ? gpu_device_->toString()
                              : std::string("<none>"))
                      << " device_valid=" << ::llaminar2::isDeviceValid(coherence_state_));
            return false;
        }

        // CPU devices: data is inherently on-host, nothing to transfer
        if (!target_device.is_gpu())
        {
            return true;
        }

        // Delegate to TransferEngine for all data movement
        auto result = TransferEngine::instance().uploadFull(this, target_device, stream);
        if (!result.success)
        {
            LOG_ERROR("[TensorBase::ensureOnDevice] TransferEngine::uploadFull failed: " << result.error);
        }
        return result.success;
    }

    bool TensorBase::allocateOnDevice(DeviceId target_device, void *stream)
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // CPU devices: host memory is the device, no allocation needed
        if (!target_device.is_gpu())
        {
            return true;
        }

        const bool trace = debugEnv().rocm.trace_coherence;

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
                LOG_TRACE("[TensorBase::allocateOnDevice] ZERO-COPY: Tensor is mapped, no allocation needed");
            }
            return true;
        }

        // Check if already allocated on target device - reuse existing allocation
        if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ == target_device)
        {
            // Buffer already exists on target device — preserve current coherence state.
            // The caller (prepareForWrite) just needs the buffer allocated; the actual
            // state transition happens later via markWritten() after the kernel runs.
            // Resetting to HOST_AUTHORITATIVE here would cause stale H2D uploads if
            // a subsequent prepareForRead sees the reset state before markWritten runs.
            if (trace)
            {
                LOG_TRACE("[TensorBase::allocateOnDevice] Reusing existing allocation on " << target_device.toString());
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
                    }
                }

                // Promote secondary to primary
                void *promoted_ptr = sec_it->second;
                secondary_device_buffers_.erase(sec_it);

                gpu_data_ptr_ = promoted_ptr;
                gpu_device_ = target_device;
                setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE); // Promoted but data stale

                if (trace)
                {
                    LOG_TRACE("[TensorBase::allocateOnDevice] Promoted secondary buffer to primary for "
                              << target_device.toString() << " ptr=" << promoted_ptr);
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
            applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE); // GPU memory freed
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
                LOG_TRACE("[TensorBase::allocateOnDevice] backend->allocate(" << bytes << " bytes) took " << alloc_us << " us");
            }

            if (!gpu_data_ptr_)
            {
                LOG_ERROR("[TensorBase::allocateOnDevice] Failed to allocate " << bytes
                                                                               << " bytes on device " << target_device.toString()
                                                                               << " (backend device ID: " << backend_device_id << ")");
                return false;
            }

            gpu_device_ = target_device;
            // Kernel will write to this buffer — host still valid.
            // Use raw set because this is a fresh allocation (state is HOST_ONLY
            // or HOST_AUTHORITATIVE from the release above, and MUTABLE_HOST_ACCESS
            // from HOST_ONLY → HOST_ONLY which is correct but less descriptive).
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);

            if (vramBomEnabled())
            {
                std::ostringstream shape_text;
                shape_text << "[";
                const auto &tensor_shape = shape();
                for (size_t axis = 0; axis < tensor_shape.size(); ++axis)
                {
                    if (axis != 0)
                        shape_text << "x";
                    shape_text << tensor_shape[axis];
                }
                shape_text << "]";
                logVramBomLine(
                    "tensor_device_owner",
                    "action=bind device=" + target_device.toString() +
                        " ptr=" + vramBomPointer(gpu_data_ptr_) +
                        " name=" + (debugName().empty() ? std::string{"(unnamed)"} : debugName()) +
                        " shape=" + shape_text.str() +
                        " dtype=" + dtype_name() +
                        " " + vramBomBytes(bytes));
            }

            LOG_TRACE("[TensorBase::allocateOnDevice] Allocated " << bytes
                                                                  << " bytes on device " << target_device.toString()
                                                                  << " (NO H2D upload - output buffer)");
        }

        return true;
    }

    // =========================================================================
    // Helper: Wait for CUDA event with cross-thread proxy support
    bool TensorBase::ensureOnHost(void *stream)
    {
        std::lock_guard<std::mutex> lock(coherence_mutex_);

        // Delegate to TransferEngine for all data movement
        auto result = TransferEngine::instance().downloadFull(this, stream);
        if (!result.success)
        {
            LOG_ERROR("[TensorBase::ensureOnHost] TransferEngine::downloadFull failed: " << result.error);
        }
        return result.success;
    }

    void TensorBase::publishDeviceWriteStateWithEvent(
        DeviceId publication_device,
        void *stream)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "[TensorBase::publishDeviceWriteStateWithEvent] GPU publication "
                "requires the exact non-null producer stream");
        }

        std::lock_guard<std::mutex> lock(coherence_mutex_);

        if (!publication_device.is_gpu())
        {
            throw std::invalid_argument(
                "[TensorBase::publishDeviceWriteStateWithEvent] GPU publication "
                "requires a GPU device");
        }
        const TensorCoherenceState new_state =
            is_mapped_
                ? TensorCoherenceState::MAPPED
                : TensorCoherenceState::DEVICE_AUTHORITATIVE;

        /*
         * A captured stage does not own the externally visible completion
         * event. Recording here would add an event-record node to the graph;
         * the resulting handle would describe graph construction rather than
         * a completed replay. The owning graph execution boundary therefore
         * publishes completion after launch: direct tensor outputs receive
         * per-tensor events, while device-only cached replay consumers receive
         * the graph stream provenance used for their next event wait.
         */
        if (!gpu_device_.has_value())
        {
            throw std::runtime_error(
                "[TensorBase::publishDeviceWriteStateWithEvent] GPU publication "
                "requested for a tensor without GPU storage");
        }
        if (*gpu_device_ != publication_device)
        {
            throw std::runtime_error(
                "[TensorBase::publishDeviceWriteStateWithEvent] Publication device " +
                publication_device.toString() +
                " does not own tensor storage on " + gpu_device_->toString());
        }

        IBackend *publication_backend = resolveBackend(publication_device);
        if (!publication_backend)
        {
            throw std::runtime_error(
                "[TensorBase::publishDeviceWriteStateWithEvent] No backend for "
                "publication device " +
                publication_device.toString());
        }
        if (publication_backend->backendDeviceType() != publication_device.type)
        {
            throw std::runtime_error(
                "[TensorBase::publishDeviceWriteStateWithEvent] Backend/device "
                "mismatch for " +
                publication_device.toString() +
                ": refusing to publish an event through the wrong runtime");
        }

        if (!isGraphCaptureActive())
        {
            IBackend *backend = publication_backend;
            const int backend_device_id = publication_device.gpu_ordinal();

            /*
             * Re-recording an event handle creates a new logical generation.
             * A wait previously enqueued for that handle proves nothing about
             * the new producer, even when the runtime reuses the same address.
             */
            last_joined_completion_event_ = nullptr;
            last_joined_consumer_stream_ = nullptr;

            /*
             * Event handles are backend- and device-owned. A migrated tensor
             * must retire the old owner's handle before creating the new one;
             * silently dropping it leaks a runtime resource and obscures an
             * invalid cross-device lifecycle.
             */
            if (device_completion_event_ && event_device_.has_value() &&
                *event_device_ != publication_device)
            {
                retireCompletionEvent_();
            }

            bool created_event = false;
            if (!device_completion_event_)
            {
                device_completion_event_ = backend->createEvent(backend_device_id);
                if (!device_completion_event_)
                {
                    throw std::runtime_error(
                        "[TensorBase::publishDeviceWriteStateWithEvent] Failed to create completion "
                        "event on " +
                        publication_device.toString());
                }
                event_device_ = publication_device;
                created_event = true;
            }

            if (!backend->recordEvent(
                    device_completion_event_,
                    backend_device_id,
                    stream))
            {
                if (created_event)
                {
                    backend->destroyEvent(
                        device_completion_event_,
                        backend_device_id);
                    device_completion_event_ = nullptr;
                    event_device_.reset();
                }
                throw std::runtime_error(
                    "[TensorBase::publishDeviceWriteStateWithEvent] Failed to record completion "
                    "event on " +
                    publication_device.toString());
            }
            completion_event_protection_ =
                completionEventProtectsHostSource_()
                    ? CompletionEventProtection::DeviceValueAndHostSource
                    : CompletionEventProtection::DeviceValue;
        }

        /*
         * Outside capture, publish coherence only after event creation and
         * recording succeed. During capture the graph controller owns the
         * eventual replay event and replaces this provisional graph-owned
         * state immediately after launch.
         */
        setCoherenceState_(new_state);
        authoritative_device_ = publication_device;

        // Mapped host reads still require completion of the GPU producer.
        if (is_mapped_)
        {
            mapped_needs_sync_ = true;
        }
    }

    void TensorBase::retireCompletionEvent_()
    {
        TransferEngine::waitForPendingHostSourceUseLocked(this);
        if (!device_completion_event_)
        {
            event_device_.reset();
            completion_event_protection_ =
                CompletionEventProtection::None;
            return;
        }
        if (!event_device_.has_value() || !event_device_->is_gpu())
        {
            throw std::runtime_error(
                "[TensorBase::retireCompletionEvent_] Completion event has no "
                "valid owning GPU device");
        }

        IBackend *event_backend = resolveBackend(*event_device_);
        if (!event_backend ||
            event_backend->backendDeviceType() != event_device_->type)
        {
            throw std::runtime_error(
                "[TensorBase::retireCompletionEvent_] Cannot resolve the "
                "completion event's owning backend for " +
                event_device_->toString());
        }

        event_backend->destroyEvent(
            device_completion_event_,
            event_device_->gpu_ordinal());
        device_completion_event_ = nullptr;
        event_device_.reset();
        completion_event_protection_ = CompletionEventProtection::None;
        last_joined_completion_event_ = nullptr;
        last_joined_consumer_stream_ = nullptr;
    }

    void TensorBase::discardDeviceValueCompletionProtection_()
    {
        last_joined_completion_event_ = nullptr;
        last_joined_consumer_stream_ = nullptr;
        switch (completion_event_protection_)
        {
        case CompletionEventProtection::None:
        case CompletionEventProtection::HostSource:
            return;
        case CompletionEventProtection::DeviceValue:
            completion_event_protection_ = CompletionEventProtection::None;
            retireCompletionEvent_();
            return;
        case CompletionEventProtection::DeviceValueAndHostSource:
            completion_event_protection_ =
                CompletionEventProtection::HostSource;
            return;
        }
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
                    retireCompletionEvent_();
                }

                backend->free(gpu_data_ptr_, backend_device_id);
            }
            LOG_TRACE("[TensorBase::releaseDeviceMemory] Released device memory on device "
                      << gpu_device_->toString());
            gpu_data_ptr_ = nullptr;
            applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE); // GPU memory freed

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
            LOG_TRACE("[TensorBase::getOrAllocateDeviceBuffer] Allocated " << bytes
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
