/**
 * @file FP32Tensor.cpp
 * @brief FP32 tensor implementation
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/Logger.h"
#include "TensorKernels.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include "../backends/ComputeBackend.h"
#include "../kernels/cpu/FP32GemmKernel.h"
#include "../kernels/cpu/CPUSoftmaxKernel.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include "../kernels/cpu/CPUSwiGLUKernel.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape, int device_idx)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          host_dirty_(false), device_dirty_(false),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        host_data_.resize(count, 0.0f);

        // TODO Phase 4: Allocate device_data_ if device_idx >= 0
        if (device_idx_ >= 0)
        {
            LOG_DEBUG("[FP32Tensor] GPU allocation not yet implemented (device " << device_idx_ << ")");
        }
    }

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           std::vector<float> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<FP32Tensor> parent)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          host_dirty_(false), device_dirty_(false),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_data_
        // They borrow from the parent via parent_data_ptr_
    }

    FP32Tensor::~FP32Tensor()
    {
        // TODO: Free device_data_ if allocated
        if (device_data_)
        {
            LOG_ERROR("[FP32Tensor] TODO: Free device data in destructor");
        }
    }

    bool FP32Tensor::set_device(int device_idx)
    {
        if (device_idx == device_idx_)
        {
            return true; // Already on target device
        }

        // TODO: Implement actual device transfer
        LOG_ERROR("[FP32Tensor] set_device not yet fully implemented");
        device_idx_ = device_idx;
        return true;
    }

    const float *FP32Tensor::data() const
    {
        // If on device, sync to host
        // TODO: Implement lazy sync when device support is added

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    float *FP32Tensor::mutable_data()
    {
        // Mark as dirty if on device
        // TODO: Implement dirty flag when device support is added
        host_dirty_ = true;

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    std::unique_ptr<ITensorGemm> FP32Tensor::createGemm()
    {
        return std::make_unique<FP32GemmKernel>(this);
    }

    std::unique_ptr<ITensorRoPE> FP32Tensor::createRoPE()
    {
        // TODO: Implement RoPE kernel creation
        LOG_ERROR("[FP32Tensor] createRoPE not yet implemented");
        return nullptr;
    }

    std::unique_ptr<ITensorSwiGLU> FP32Tensor::createSwiGLU()
    {
        // TODO: Implement SwiGLU kernel creation
        LOG_ERROR("[FP32Tensor] createSwiGLU not yet implemented");
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> FP32Tensor::createSoftmax()
    {
        // TODO: Implement Softmax kernel creation
        LOG_ERROR("[FP32Tensor] createSoftmax not yet implemented");
        return nullptr;
    }

    std::unique_ptr<ITensorRMSNorm> FP32Tensor::createRMSNorm()
    {
        // FP32 tensors use CPU RMSNorm kernel
        return std::make_unique<CPURMSNormKernel>();
    }

    bool FP32Tensor::sync_to_device()
    {
        // TODO: Implement sync_to_device
        LOG_ERROR("[FP32Tensor] sync_to_device not yet implemented");
        return false;
    }

    bool FP32Tensor::sync_from_device()
    {
        // TODO: Implement sync_from_device
        LOG_ERROR("[FP32Tensor] sync_from_device not yet implemented");
        return false;
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

        int src_device = src->device_index();
        int dst_device = device_idx_;

        // Determine transfer type
        bool cpu_to_cpu = (src_device == -1 && dst_device == -1);
        bool cpu_to_gpu = (src_device == -1 && dst_device >= 0);
        bool gpu_to_cpu = (src_device >= 0 && dst_device == -1);
        bool gpu_to_gpu = (src_device >= 0 && dst_device >= 0);

        LOG_DEBUG("[FP32Tensor::copyFrom] Transfer: device " << src_device
                                                             << " → device " << dst_device << " (" << count << " elements)");

        if (cpu_to_cpu)
        {
            // CPU → CPU: Simple memcpy
            const float *src_data = src->data();
            std::memcpy(host_data_.data(), src_data, count * sizeof(float));
            host_dirty_ = true; // Mark host as authoritative
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

        std::vector<float> *root_data = is_view_ ? parent_data_ptr_ : &host_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<FP32Tensor>(new FP32Tensor(
            new_shape,
            device_idx_,
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

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(src[i]);
        }
    }

    void FP32Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const float *src = data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(src[i]);
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

} // namespace llaminar2
