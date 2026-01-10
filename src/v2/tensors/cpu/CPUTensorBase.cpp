/**
 * @file TensorBase.cpp
 * @brief TensorBase class implementation (helper methods)
 *
 * @author David Sanftenberg
 */

#include "CPUTensors.h"
#include "../TensorKernels.h"
#include "../SIMDHelpers.h"
#include "../../utils/CPUFeatures.h"
#include "../../utils/Logger.h"
#include "../../backends/BackendManager.h"
#include "../../backends/ComputeBackend.h"
#include "../../backends/DeviceId.h"
#include "../../kernels/KernelFactory.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
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

        LOG_ERROR("[CPUTensorBase] Unknown device type: " << device.toString());
        return nullptr;
    }

    // ===== Legacy helper functions for global device index mapping =====

    /**
     * @brief Get the appropriate backend for a global device index
     *
     * Maps global device index (e.g., 1, 2, 3) to the correct backend (CUDA or ROCm)
     * based on the device type from DeviceManager.
     *
     * @param device_idx Global device index (0 = CPU, 1+ = GPUs)
     * @deprecated Use getBackendForDevice(DeviceId) instead
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
            LOG_ERROR("[CPUTensorBase] Invalid device index: " << device_idx
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
            LOG_ERROR("[CPUTensorBase] Invalid device index for device ID lookup: " << device_idx);
            return 0;
        }

        return devices[device_idx].device_id;
    }

    // ===== TensorBase destructor =====
    // Clears the kernel cache entry for this tensor to prevent use-after-free
    // when a new tensor is allocated at the same memory address.
    CPUTensorBase::~CPUTensorBase()
    {
        llaminar::v2::kernels::KernelFactory::clearCacheFor(this);
    }

    void CPUTensorBase::to_fp32_via_blocks(float *dst) const
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

    bool CPUTensorBase::to_int8_perchannel_via_blocks(int8_t *dst_int8,
                                                      float *dst_col_scales,
                                                      float *dst_row_scales) const
    {
        // Verify this is an ITensorGemmTileDataProvider tensor
        const ITensorGemmTileDataProvider *decoder = dynamic_cast<const ITensorGemmTileDataProvider *>(this);
        if (!decoder)
        {
            LOG_ERROR("[CPUTensorBase] to_int8_perchannel_via_blocks() requires ITensorGemmTileDataProvider interface");
            return false;
        }

        // Verify 2D shape
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[CPUTensorBase] to_int8_perchannel_via_blocks() requires 2D tensor, got " << shp.size() << "D");
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

    ActivationPack CPUTensorBase::pack_activation_rows_to_int8(int rows, int cols) const
    {
        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[CPUTensorBase] pack_activation_rows_to_int8 requires positive dimensions");
            return {};
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[CPUTensorBase] pack_activation_rows_to_int8 requires 2D tensor, got " << shp.size() << "D");
            return {};
        }
        if (static_cast<size_t>(rows) > shp[0] || static_cast<size_t>(cols) != shp[1])
        {
            LOG_ERROR("[CPUTensorBase] pack_activation_rows_to_int8 dimension mismatch: tensor is ["
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
    void CPUTensorBase::to<float>(float *dst, TensorType format) const
    {
        to_fp32(dst);
    }

    // FP16 conversion
    template <>
    void CPUTensorBase::to<uint16_t>(uint16_t *dst, TensorType format) const
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
    void CPUTensorBase::to<int8_t>(int8_t *dst, TensorType format) const
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
    void CPUTensorBase::to<int32_t>(int32_t *dst, TensorType format) const
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
    void CPUTensorBase::to_q8_0(Q8_0Block *dst) const
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
    template void CPUTensorBase::to<float>(float *dst, TensorType format) const;
    template void CPUTensorBase::to<uint16_t>(uint16_t *dst, TensorType format) const;
    template void CPUTensorBase::to<int8_t>(int8_t *dst, TensorType format) const;
    template void CPUTensorBase::to<int32_t>(int32_t *dst, TensorType format) const;

    // =========================================================================
    // Lazy Transfer Implementation (Phase 1 GPU Device-Aware Slicing)
    // =========================================================================

    // Default implementations for raw_host_data_ptr and byte_size
    // Tensors that support GPU transfer override these.
    void *CPUTensorBase::raw_host_data_ptr()
    {
        throw std::runtime_error("[CPUTensorBase] raw_host_data_ptr() not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    const void *CPUTensorBase::raw_host_data_ptr() const
    {
        throw std::runtime_error("[CPUTensorBase] raw_host_data_ptr() const not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    size_t CPUTensorBase::byte_size() const
    {
        throw std::runtime_error("[CPUTensorBase] byte_size() not implemented for this tensor type. "
                                 "Override in derived class to support GPU transfer.");
    }

    void CPUTensorBase::invalidateGpuData()
    {
        // Mark GPU data as stale - next ensureOnDevice() will re-upload from host
        // Mark device as invalid (stale) - do NOT free GPU memory
        // This is called when host data is modified and GPU copy is now stale.
        // The GPU memory is kept allocated; next ensureOnDevice() will just re-upload.
        if (gpu_data_ptr_)
        {
            device_valid_ = false;
            LOG_DEBUG("[CPUTensorBase::invalidateGpuData] Device data marked stale (memory retained)");
        }
    }

    bool CPUTensorBase::ensureOnDevice(DeviceId target_device)
    {
        // Validate target device - must be a GPU
        if (!target_device.is_gpu())
        {
            LOG_ERROR("[CPUTensorBase::ensureOnDevice] Invalid device (must be GPU): " << target_device.toString());
            return false;
        }

        // Check if already on target device with valid data
        if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ == target_device && device_valid_)
        {
            // Already on correct device with valid GPU data - nothing to do
            return true;
        }

        // Get backend for target device directly from DeviceId type
        IBackend *target_backend = getBackendForDevice(target_device);
        if (!target_backend)
        {
            LOG_ERROR("[CPUTensorBase::ensureOnDevice] No backend available for device " << target_device.toString());
            return false;
        }

        // Get backend-specific device ordinal (e.g., CUDA:0 -> 0, ROCm:1 -> 1)
        int backend_device_id = target_device.gpu_ordinal();

        // Free existing device memory if on different device
        if (gpu_data_ptr_ && gpu_device_.has_value() && *gpu_device_ != target_device)
        {
            // Use the OLD device's backend for freeing
            IBackend *old_backend = getBackendForDevice(*gpu_device_);
            int old_backend_device_id = gpu_device_->gpu_ordinal();
            if (old_backend)
            {
                old_backend->free(gpu_data_ptr_, old_backend_device_id);
            }
            gpu_data_ptr_ = nullptr;
            gpu_device_.reset();
            device_valid_ = false;
        }

        // Allocate on target device if needed
        size_t bytes = byte_size();
        if (!gpu_data_ptr_)
        {
            gpu_data_ptr_ = target_backend->allocate(bytes, backend_device_id);
            if (!gpu_data_ptr_)
            {
                LOG_ERROR("[CPUTensorBase::ensureOnDevice] Failed to allocate " << bytes
                                                                                << " bytes on device " << target_device.toString()
                                                                                << " (backend device ID: " << backend_device_id << ")");
                return false;
            }
            gpu_device_ = target_device; // Store the actual DeviceId
            device_valid_ = false;       // Newly allocated - not yet populated
        }

        // Upload data from host if device data is stale
        if (!device_valid_)
        {
            // Sanity check: host must have valid data if device is stale
            if (!host_valid_)
            {
                LOG_ERROR("[CPUTensorBase::ensureOnDevice] COHERENCE ERROR: Both host and device are invalid!");
                return false;
            }

            const void *src = raw_host_data_ptr();
            if (!src)
            {
                LOG_ERROR("[CPUTensorBase::ensureOnDevice] Host data pointer is null");
                target_backend->free(gpu_data_ptr_, backend_device_id);
                gpu_data_ptr_ = nullptr;
                gpu_device_.reset();
                return false;
            }

            if (!target_backend->hostToDevice(gpu_data_ptr_, src, bytes, backend_device_id))
            {
                LOG_ERROR("[CPUTensorBase::ensureOnDevice] hostToDevice failed");
                target_backend->free(gpu_data_ptr_, backend_device_id);
                gpu_data_ptr_ = nullptr;
                gpu_device_.reset();
                return false;
            }

            device_valid_ = true; // Device now has current data
            // host_valid_ stays true - we uploaded FROM host, so both are now in sync

            LOG_DEBUG("[CPUTensorBase::ensureOnDevice] Uploaded " << bytes
                                                                  << " bytes to device " << target_device.toString()
                                                                  << " (backend device ID: " << backend_device_id << ")");
        }

        return true;
    }

    bool CPUTensorBase::ensureOnHost()
    {
        // If host is already valid, nothing to do
        if (host_valid_)
        {
            LOG_DEBUG("[CPUTensorBase::ensureOnHost] Host already valid, skipping sync");
            return true;
        }

        // Sanity check: device must be valid if host is invalid
        if (!device_valid_)
        {
            LOG_ERROR("[CPUTensorBase::ensureOnHost] COHERENCE ERROR: Both host and device are invalid!");
            return false;
        }

        // If GPU has data, download it
        if (gpu_data_ptr_ && gpu_device_.has_value())
        {
            IBackend *backend = getBackendForDevice(*gpu_device_);
            if (!backend)
            {
                LOG_ERROR("[CPUTensorBase::ensureOnHost] No backend available for device " << gpu_device_->toString());
                return false;
            }

            int backend_device_id = gpu_device_->gpu_ordinal();

            size_t bytes = byte_size();
            void *dst = raw_host_data_ptr();
            if (!dst)
            {
                LOG_ERROR("[CPUTensorBase::ensureOnHost] Host data pointer is null");
                return false;
            }

            if (!backend->deviceToHost(dst, gpu_data_ptr_, bytes, backend_device_id))
            {
                LOG_ERROR("[CPUTensorBase::ensureOnHost] deviceToHost failed");
                return false;
            }

            host_valid_ = true; // Host now has current data
            // device_valid_ stays true - we downloaded FROM device, so both are now in sync

            LOG_DEBUG("[CPUTensorBase::ensureOnHost] Downloaded " << bytes
                                                                  << " bytes from device " << gpu_device_->toString()
                                                                  << " (backend device ID: " << backend_device_id << ")");
        }

        return true;
    }

    bool CPUTensorBase::releaseDeviceMemory()
    {
        // Ensure host has current data
        if (!ensureOnHost())
        {
            return false;
        }

        // Free device memory
        if (gpu_data_ptr_ && gpu_device_.has_value())
        {
            IBackend *backend = getBackendForDevice(*gpu_device_);
            if (backend)
            {
                int backend_device_id = gpu_device_->gpu_ordinal();
                backend->free(gpu_data_ptr_, backend_device_id);
            }
            LOG_DEBUG("[CPUTensorBase::releaseDeviceMemory] Released device memory on device "
                      << gpu_device_->toString());
            gpu_data_ptr_ = nullptr;
            device_valid_ = false;
            gpu_device_.reset();
        }

        return true;
    }

    // =====================================================================
    // Tensor Comparison Utilities Implementation
    // =====================================================================

    double CPUTensorBase::cosineSimilarityTo(const CPUTensorBase *other) const
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

    float CPUTensorBase::maxAbsDiffTo(const CPUTensorBase *other) const
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

    float CPUTensorBase::meanAbsDiffTo(const CPUTensorBase *other) const
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

    double CPUTensorBase::relativeL2To(const CPUTensorBase *other) const
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

    double CPUTensorBase::klDivergenceTo(const CPUTensorBase *other) const
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

    CPUTensorBase::ComparisonSummary CPUTensorBase::compareTo(const CPUTensorBase *other) const
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

} // namespace llaminar2
