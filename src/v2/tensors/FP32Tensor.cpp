/**
 * @file FP32Tensor.cpp
 * @brief FP32 tensor implementation
 *
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "TensorKernels.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include "../backends/ComputeBackend.h"
#include "../backends/IBackend.h"
#include "../backends/BackendManager.h"
#include "../kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../collective/backends/PCIeBARBackend.h" // For BAR-backed tensor support

#include "../kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "../kernels/cpu/attention/CPUAttentionKernelT.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../utils/CPUFeatures.h"

namespace llaminar2
{

    // Helper to get the backend for a device index
    static IBackend *getBackendForDeviceIdx(int device_idx)
    {
        if (device_idx <= 0)
            return nullptr; // CPU doesn't use IBackend

        // Get device type from DeviceManager
        const auto &devices = DeviceManager::instance().devices();
        if (device_idx < 0 || static_cast<size_t>(device_idx) >= devices.size())
        {
            LOG_ERROR("[FP32Tensor] Invalid device index: " << device_idx);
            return nullptr;
        }

        const auto &device = devices[device_idx];
        return getBackendForDeviceType(device.type);
    }

    // Helper to get backend-specific device ID
    static int getBackendDeviceId(int device_idx)
    {
        if (device_idx <= 0)
            return 0;

        const auto &devices = DeviceManager::instance().devices();
        if (static_cast<size_t>(device_idx) >= devices.size())
        {
            return 0;
        }

        return devices[device_idx].device_id;
    }

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape, DeviceId device)
        : shape_(shape), device_(device),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        host_data_.resize(count, 0.0f);

        // GPU allocation is handled by TensorBase::ensureOnDevice() when needed
    }

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape,
                           DeviceId device,
                           AlignedVector<float> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<FP32Tensor> parent)
        : shape_(shape), device_(device),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_data_
        // They borrow from the parent via parent_data_ptr_
    }

    std::unique_ptr<FP32Tensor> FP32Tensor::createMapped(
        const std::vector<size_t> &shape,
        DeviceId target_device)
    {
        // Calculate tensor size
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        size_t bytes = count * sizeof(float);

        // Create tensor with regular constructor first (no host allocation for mapped)
        auto tensor = std::make_unique<FP32Tensor>(shape, target_device);

        // Try to initialize as mapped tensor via base class
        if (!tensor->initMappedMemory(bytes, target_device))
        {
            LOG_WARN("[FP32Tensor::createMapped] Failed to allocate mapped memory ("
                     << bytes << " bytes), using regular allocation");
            // tensor already has regular host allocation, just return it
        }

        return tensor;
    }

    std::unique_ptr<FP32Tensor> FP32Tensor::createBARBacked(
        const std::vector<size_t> &shape,
        DeviceId cuda_device,
        DeviceId rocm_device,
        PCIeBARBackend *backend)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Validate inputs
        if (!cuda_device.is_cuda())
        {
            LOG_ERROR("[FP32Tensor::createBARBacked] cuda_device must be CUDA, got: " << cuda_device.toString());
            return nullptr;
        }
        if (!rocm_device.is_rocm())
        {
            LOG_ERROR("[FP32Tensor::createBARBacked] rocm_device must be ROCm, got: " << rocm_device.toString());
            return nullptr;
        }
        if (!backend)
        {
            LOG_ERROR("[FP32Tensor::createBARBacked] backend must not be null");
            return nullptr;
        }
        if (!backend->isInitialized())
        {
            LOG_ERROR("[FP32Tensor::createBARBacked] backend must be initialized");
            return nullptr;
        }

        // Calculate tensor size
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        size_t bytes = count * sizeof(float);

        // Create tensor with regular constructor
        // Note: We use cuda_device as the "home" device since CUDA will write to it
        auto tensor = std::make_unique<FP32Tensor>(shape, cuda_device);

        // Try to initialize as BAR-backed tensor via base class
        if (!tensor->initBARBackedMemory(bytes, cuda_device, rocm_device, backend))
        {
            LOG_ERROR("[FP32Tensor::createBARBacked] Failed to allocate BAR-backed memory ("
                      << bytes << " bytes)");
            return nullptr; // Unlike createMapped, we don't fall back - BAR is required
        }

        LOG_DEBUG("[FP32Tensor::createBARBacked] Created BAR-backed tensor: "
                  << shape[0] << "x" << (shape.size() > 1 ? shape[1] : 1)
                  << " (" << bytes << " bytes)"
                  << " cuda=" << cuda_device.toString()
                  << " rocm=" << rocm_device.toString());

        return tensor;
#else
        // Stub when CUDA and ROCm are not both available
        LOG_ERROR("[FP32Tensor::createBARBacked] Requires both HAVE_CUDA and HAVE_ROCM");
        (void)shape;
        (void)cuda_device;
        (void)rocm_device;
        (void)backend;
        return nullptr;
#endif
    }

    FP32Tensor::~FP32Tensor()
    {
        // Mapped memory and BAR-backed memory cleanup is handled by TensorBase destructor
        // Nothing tensor-specific to clean up here
    }

    const float *FP32Tensor::data() const
    {
        assertValid("FP32Tensor::data");

        LOG_TRACE("[FP32Tensor::data] Called for tensor, host_valid=" << ::llaminar2::isHostValid(coherence_state_) << " device_valid=" << ::llaminar2::isDeviceValid(coherence_state_) << " gpu_data_ptr_=" << (gpu_data_ptr_ ? "set" : "null") << " is_mapped_=" << is_mapped_);

        // Use base class to ensure host has current data
        // For mapped tensors, ensureOnHost() is a no-op (data is always available)
        const_cast<FP32Tensor *>(this)->ensureOnHost();

        // Mapped tensors use mapped_host_ptr_ from base class
        if (is_mapped_ && mapped_host_ptr_)
        {
            return static_cast<const float *>(mapped_host_ptr_);
        }

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    float *FP32Tensor::mutable_data()
    {
        assertValid("FP32Tensor::mutable_data");

        // Ensure host has current data before modification
        // For mapped tensors, ensureOnHost() is a no-op
        ensureOnHost();

        // For non-mapped tensors: Invalidate GPU copy since host will be modified
        // For mapped tensors: Both host and device share memory, no invalidation needed
        if (!is_mapped_)
        {
            invalidateGpuData();
        }

        // Mapped tensors use mapped_host_ptr_ from base class
        if (is_mapped_ && mapped_host_ptr_)
        {
            return static_cast<float *>(mapped_host_ptr_);
        }

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    std::unique_ptr<ITensorGemm> FP32Tensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    std::unique_ptr<ITensorRoPE> FP32Tensor::createRoPE()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createRoPE(this, dev_type);
    }

    std::unique_ptr<ITensorSwiGLU> FP32Tensor::createSwiGLU()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createSwiGLU(this, dev_type);
    }

    std::unique_ptr<ITensorSoftmax> FP32Tensor::createSoftmax()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createSoftmax(this, dev_type);
    }

    std::unique_ptr<ITensorRMSNorm> FP32Tensor::createRMSNorm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createRMSNorm(this, dev_type);
    }

    std::unique_ptr<ITensorAttention> FP32Tensor::createAttention()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createAttention(this, dev_type);
    }

    std::unique_ptr<ITensorEmbedding> FP32Tensor::createEmbedding()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createEmbedding(this, dev_type);
    }

    // =========================================================================
    // Lazy Transfer Accessors (Phase 1 GPU Device-Aware Slicing)
    // TensorBase uses these to implement ensureOnDevice/ensureOnHost.
    // =========================================================================

    void *FP32Tensor::raw_host_data_ptr()
    {
        // Mapped tensors use mapped_host_ptr_ from base class
        if (is_mapped_ && mapped_host_ptr_)
        {
            return mapped_host_ptr_;
        }
        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    const void *FP32Tensor::raw_host_data_ptr() const
    {
        // Mapped tensors use mapped_host_ptr_ from base class
        if (is_mapped_ && mapped_host_ptr_)
        {
            return mapped_host_ptr_;
        }
        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    bool FP32Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Source tensor is null");
            return false;
        }

        // Validate shape compatibility
        const auto &src_shape = src->shape();
        if (src_shape != shape_)
        {
            std::string src_str = "[";
            for (size_t i = 0; i < src_shape.size(); ++i)
            {
                src_str += std::to_string(src_shape[i]);
                if (i + 1 < src_shape.size())
                    src_str += ", ";
            }
            src_str += "]";

            std::string dst_str = "[";
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                dst_str += std::to_string(shape_[i]);
                if (i + 1 < shape_.size())
                    dst_str += ", ";
            }
            dst_str += "]";

            LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Shape mismatch - src: " << src_str << ", dst: " << dst_str);
            return false;
        }

        size_t count = 1;
        for (auto dim : shape_)
        {
            count *= dim;
        }

        DeviceId src_device = src->home_device();
        DeviceId dst_device = device_;

        // Determine transfer type
        bool cpu_to_cpu = (src_device.is_cpu() && dst_device.is_cpu());
        bool cpu_to_gpu = (src_device.is_cpu() && dst_device.is_gpu());
        bool gpu_to_cpu = (src_device.is_gpu() && dst_device.is_cpu());
        bool gpu_to_gpu = (src_device.is_gpu() && dst_device.is_gpu());

        LOG_DEBUG("[FP32Tensor::copyFrom] Transfer: " << src_device.toString()
                                                      << " → " << dst_device.toString() << " (" << count << " elements)");

        if (cpu_to_cpu)
        {
            // CPU → CPU: Simple memcpy
            const float *src_data = src->data();
            std::memcpy(host_data_.data(), src_data, count * sizeof(float));
            // Host now has latest data; GPU copy (if any) is stale
            return true;
        }
        else if (cpu_to_gpu)
        {
            // CPU → GPU: Phase 4 CUDA
            LOG_ERROR("[FP32Tensor::copyFrom] CPU → GPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from CPU to GPU device " << dst_device);
            return false;
        }
        else if (gpu_to_cpu)
        {
            // GPU → CPU: Phase 4 CUDA
            LOG_ERROR("[FP32Tensor::copyFrom] GPU → CPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from GPU device " << src_device << " to CPU");
            return false;
        }
        else if (gpu_to_gpu)
        {
            // GPU → GPU: Phase 4 CUDA (peer-to-peer copy)
            if (src_device == dst_device)
            {
                LOG_DEBUG("[FP32Tensor::copyFrom] Same GPU device (" << src_device << "), no transfer needed");
                return true;
            }
            LOG_ERROR("[FP32Tensor::copyFrom] GPU → GPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from GPU " << src_device << " to GPU " << dst_device);
            return false;
        }

        // Should never reach here
        LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Unknown transfer type");
        return false;
    }

    std::shared_ptr<TensorBase> FP32Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Calculate total elements in the new view
        size_t view_elements = 1;
        for (auto dim : new_shape)
        {
            view_elements *= dim;
        }

        // Calculate available elements from offset
        size_t parent_elements = 1;
        for (auto dim : shape_)
        {
            parent_elements *= dim;
        }

        if (offset >= parent_elements)
        {
            LOG_ERROR("[FP32Tensor::create_view] ERROR: offset " << offset << " >= parent size " << parent_elements);
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            LOG_ERROR("[FP32Tensor::create_view] ERROR: view size " << view_elements
                                                                    << " > available elements " << available_elements
                                                                    << " (offset=" << offset << ", parent_size=" << parent_elements << ")");
            return nullptr;
        }

        // Determine the actual parent tensor and data pointer
        // If this is already a view, chain to the root parent
        std::shared_ptr<FP32Tensor> root_parent;
        if (is_view_)
        {
            root_parent = std::dynamic_pointer_cast<FP32Tensor>(parent_);
            if (!root_parent)
            {
                LOG_ERROR("[FP32Tensor::create_view] ERROR: Failed to cast parent to FP32Tensor (is_view=true)");
                return nullptr;
            }
        }
        else
        {
            // Get a proper shared_ptr to this object (increments ref count)
            try
            {
                auto self_ptr = shared_from_this();
                root_parent = std::dynamic_pointer_cast<FP32Tensor>(self_ptr);
                if (!root_parent)
                {
                    LOG_ERROR("[FP32Tensor::create_view] ERROR: Failed to cast shared_from_this to FP32Tensor");
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[FP32Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!");
                LOG_ERROR("[FP32Tensor::create_view] Exception: " << e.what());
                return nullptr;
            }
        }

        AlignedVector<float> *root_data = is_view_ ? parent_data_ptr_ : &host_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<FP32Tensor>(new FP32Tensor(
            new_shape,
            device_,
            root_data,
            root_offset,
            root_parent));

        return view_tensor;
    }

    // ===== Format Conversion Methods =====

    void FP32Tensor::to_fp32(float *dst) const
    {
        const size_t count = element_count();
        const float *src = data(); // Handles view offset if needed
        std::memcpy(dst, src, count * sizeof(float));
    }

    void FP32Tensor::to_bf16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const float *src = data();

        // Use SIMD-optimized conversion with OpenMP
        const size_t chunk_size = 4096;
        const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

#pragma omp parallel for
        for (size_t i = 0; i < num_chunks; ++i)
        {
            size_t start = i * chunk_size;
            size_t current_chunk = std::min(chunk_size, count - start);
            simd::convert_fp32_to_bf16(src + start, dst + start, current_chunk);
        }
    }

    void FP32Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const float *src = data();

        // Use SIMD-optimized conversion with OpenMP
        const size_t chunk_size = 4096;
        const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

#pragma omp parallel for
        for (size_t i = 0; i < num_chunks; ++i)
        {
            size_t start = i * chunk_size;
            size_t current_chunk = std::min(chunk_size, count - start);
            simd::convert_fp32_to_fp16(src + start, dst + start, current_chunk);
        }
    }

    void FP32Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        const size_t total_elements = element_count();
        const size_t num_blocks = (total_elements + block_size - 1) / block_size;
        const float *src = data();

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f; // Store inverse for faster dequant

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = src[offset + i] * scale;
                const float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst_int8[offset + i] = static_cast<int8_t>(std::round(clamped));
            }

            // Zero-fill partial block tail (if any)
            for (size_t i = count; i < block_size; ++i)
            {
                dst_int8[offset + i] = 0;
            }
        }
    }

    bool FP32Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[FP32Tensor] to_int8_perchannel() requires 2D tensor, got " << shp.size() << "D");
            return false;
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        const float *src = data();

        // Compute per-column scales
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                float abs_val = std::fabs(src[i * cols + j]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }
            dst_col_scales[j] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Compute per-row scales (if requested)
        if (dst_row_scales != nullptr)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                float max_abs = 0.0f;
                for (size_t j = 0; j < cols; ++j)
                {
                    float abs_val = std::fabs(src[i * cols + j]);
                    if (abs_val > max_abs)
                        max_abs = abs_val;
                }
                dst_row_scales[i] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            }
        }

        // Quantize to INT8
        // If row scales are requested, prefer using them (per-output-channel quantization for weights)
        // Otherwise use column scales (per-input-channel quantization)
        const bool use_row_scales = (dst_row_scales != nullptr);

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

                float scaled = src[idx] * inv_scale;
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

    void FP32Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_row() requires 2D tensor");
        }
        if (row_idx >= shp[0])
        {
            throw std::out_of_range("Row index out of bounds");
        }

        const size_t cols = shp[1];
        const float *src = data();
        std::memcpy(buffer, src + row_idx * cols, cols * sizeof(float));
    }

    void FP32Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        const float *src = data();
        std::memcpy(buffer, src + offset, count * sizeof(float));
    }

    ActivationPack FP32Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[FP32Tensor] to_int8_activation_pack requires positive dimensions");
            return {};
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[FP32Tensor] to_int8_activation_pack requires 2D tensor, got " << shp.size() << "D");
            return {};
        }
        if (static_cast<size_t>(rows) > shp[0] || static_cast<size_t>(cols) != shp[1])
        {
            LOG_ERROR("[FP32Tensor] to_int8_activation_pack dimension mismatch: tensor is ["
                      << shp[0] << ", " << shp[1] << "], requested " << rows << "x" << cols);
            return {};
        }

        ActivationPack pack;
        pack.rows = rows;
        pack.cols = cols;
        const size_t row_stride = static_cast<size_t>(cols);
        const size_t total = row_stride * static_cast<size_t>(rows);
        pack.data.resize(total);
        pack.row_scales.resize(static_cast<size_t>(rows));

        const float *src_data = data();

#pragma omp parallel for
        for (int m = 0; m < rows; ++m)
        {
            const float *row_src = src_data + static_cast<size_t>(m) * row_stride;
            int8_t *row_dst = pack.data.data() + static_cast<size_t>(m) * row_stride;

            // Calculate max abs directly from source
            float max_abs = simd::activation_row_max_abs(row_src, cols);

            const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            pack.row_scales[static_cast<size_t>(m)] = scale;
            const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            simd::quantize_activation_row(row_src, cols, inv_scale, row_dst);
        }

        return pack;
    }

    bool FP32Tensor::applyRMSNorm(
        const float *gamma,
        int seq_len,
        int d_model,
        float eps,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        LOG_ERROR("[FP32Tensor::applyRMSNorm] Removed — use ITensorRMSNorm::apply_tensor() via stages");
        return false;
    }

    bool FP32Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        if (!accum)
        {
            LOG_ERROR("[FP32Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[FP32Tensor::from_int32_with_scales] tensor must be 2D, got " << shape_.size() << "D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[FP32Tensor::from_int32_with_scales] shape mismatch: tensor=[" << shape_[0]
                                                                                      << ", " << shape_[1] << "] input=[" << rows << ", " << cols << "]");
            return false;
        }

        float *dst = is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_data_.data();
        simd::requantize_int32_matrix_to_fp32(
            accum,
            dst,
            rows,
            cols,
            row_scales,
            col_scales,
            bias);

        // Host now has latest data; GPU copy (if any) is stale
        return true;
    }

    bool FP32Tensor::applyRoPE(
        float * /*K*/,
        const int * /*position_ids*/,
        int /*seq_len*/,
        int /*n_heads*/,
        int /*n_kv_heads*/,
        int /*head_dim*/,
        float /*rope_theta*/,
        bool /*use_bf16*/,
        const IMPIContext * /*mpi_ctx*/,
        int /*device_idx*/)
    {
        LOG_ERROR("[FP32Tensor::applyRoPE] Legacy raw-pointer path removed. Use RoPEStage with apply_tensor() instead.");
        return false;
    }

    void FP32Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Calculate source offset in FP32 data
        const size_t cols = shape_[1];
        const size_t k_start = k_block_offset * Q8_0Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_0: row_idx out of range");
        }
        if (k_start + Q8_0Block::BLOCK_SIZE > cols)
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_0: k_block_offset exceeds tensor width");
        }

        // Get pointer to source FP32 data
        const float *fp32_data = data() + row_idx * cols + k_start;

        // Use vectorized quantization (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_fp32_to_q8_0(
            fp32_data,   // Input: FP32 values
            output->qs,  // Output: Q8_0 int8 values
            &output->d); // Output: Q8_0 FP16 scale
    }

    const Q8_1Block *FP32Tensor::decode_to_q8_1(size_t row_idx, size_t k_block_offset) const
    {
        // Calculate source offset in FP32 data
        const size_t cols = shape_[1];
        const size_t k_start = k_block_offset * Q8_1Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_1: row_idx out of range");
        }
        if (k_start + Q8_1Block::BLOCK_SIZE > cols)
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_1: k_block_offset exceeds tensor width");
        }

        // Get pointer to source FP32 data
        const float *fp32_data = data() + row_idx * cols + k_start;

        // Use thread-local storage to avoid heap allocation
        thread_local Q8_1Block q8_1_block;

        // Use vectorized quantization with pre-computed sum (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_fp32_to_q8_1(
            fp32_data,                                         // Input: FP32 values
            q8_1_block.qs,                                     // Output: Q8_1 int8 values
            &q8_1_block.d,                                     // Output: Q8_1 FP16 scale
            reinterpret_cast<uint16_t *>(&q8_1_block.sum_qs)); // Output: Q8_1 INT16 pre-computed sum (Nov 2024: changed from FP16 's')

        return &q8_1_block;
    }

    // ===== Bulk Q8_1 Quantization =====

    /**
     * @brief Bulk quantize FP32 tensor to Q8_1 format (SIMD-optimized)
     *
     * Dispatches to AVX-512, AVX2, or scalar implementation based on CPU features.
     * Uses OpenMP parallelization with adaptive strategy:
     * - Large M: Parallelize over rows (better cache locality)
     * - Small M: Collapse rows × blocks (more parallelism)
     *
     * @param q8_1_buffer Output buffer for Q8_1 blocks [m * k_blocks]
     * @param m Number of rows (batch_size * seq_len)
     * @param k Number of columns (must match tensor's K dimension)
     * @return true on success, false on failure
     */
    bool FP32Tensor::quantize_to_q8_1(void *q8_1_buffer, int m, int k) const
    {
        if (!q8_1_buffer || m <= 0 || k <= 0)
        {
            return false;
        }

        // Validate dimensions against tensor shape
        const size_t cols = shape_[1];
        const size_t rows = shape_[0];
        if (static_cast<size_t>(m) > rows || static_cast<size_t>(k) > cols)
        {
            return false;
        }

        const int k_blocks = (k + 31) / 32;
        Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);
        const float *fp32_data = data();

        // Check if K is a multiple of 32 (enables fast path with no bounds checking)
        const bool k_aligned = (k % 32 == 0);

#pragma omp parallel
        {
            // Parallelize over rows for large M, or collapse(2) for small M
            int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
            if (quant_thresh == 0)
                quant_thresh = omp_get_num_threads();

            if (m < quant_thresh)
            {
                // Small M: collapse rows × blocks for more parallelism
#pragma omp for collapse(2) schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        const float *a_row = fp32_data + i * cols + k_blk * 32;
                        Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                        int valid_elements = std::min(32, k - k_blk * 32);
                        simd::quantize_single_block(a_row, row_blocks[k_blk], valid_elements);
                    }
                }
            }
            else
            {
                // Large M: parallelize over rows, process all blocks per row sequentially
#pragma omp for schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    const float *a_row = fp32_data + i * cols;
                    Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                    if (k_aligned)
                    {
                        // Fast path: K is multiple of 32, no partial blocks
                        // Process blocks in pairs for 2-way ILP on AVX-512
                        int k_blk = 0;
                        if (activeISALevel() >= ISALevel::AVX512)
                        {
                            for (; k_blk + 1 < k_blocks; k_blk += 2)
                            {
                                simd::quantize_two_blocks_avx512(a_row + k_blk * 32,
                                                                 row_blocks[k_blk],
                                                                 row_blocks[k_blk + 1]);
                            }
                        }
                        // Handle remaining odd block (if any)
                        for (; k_blk < k_blocks; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }
                    }
                    else
                    {
                        // General path: handle partial last block
                        for (int k_blk = 0; k_blk < k_blocks - 1; ++k_blk)
                        {
                            simd::quantize_single_block(a_row + k_blk * 32, row_blocks[k_blk], 32);
                        }

                        // Last block may be partial
                        if (k_blocks > 0)
                        {
                            int last_k_blk = k_blocks - 1;
                            int valid_elements = k - last_k_blk * 32;
                            simd::quantize_single_block(a_row + last_k_blk * 32, row_blocks[last_k_blk], valid_elements);
                        }
                    }
                }
            }
        }

        return true;
    }

} // namespace llaminar2
