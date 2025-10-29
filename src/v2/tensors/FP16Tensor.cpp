/**
 * @file FP16Tensor.cpp
 * @brief FP16 tensor implementation (16-bit IEEE 754 half-precision)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/Logger.h"
#include <cstring>
#include <stdexcept>
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>

namespace llaminar2
{
    // ========== Constructors ==========

    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("FP16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        host_fp16_data_.resize(total, 0);
    }

    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &fp16_data)
        : shape_(shape), device_idx_(-1), device_data_(nullptr), host_fp16_data_(fp16_data),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("FP16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        if (fp16_data.size() != total)
        {
            throw std::invalid_argument("FP16Tensor: data size mismatch");
        }
    }

    // Private view constructor
    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           std::vector<uint16_t> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<FP16Tensor> parent)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_fp16_data_
        // They borrow from the parent via parent_data_ptr_
    }

    FP16Tensor::~FP16Tensor()
    {
        // TODO: Free device memory when device support is added
    }

    // ========== TensorBase Interface ==========

    bool FP16Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        // TODO: Upload to device when device support is added
        return true;
    }

    const float *FP16Tensor::data() const
    {
        // Lazy dequantization to FP32 cache
        if (dequant_cache_.empty())
        {
            // Get the actual data pointer (owned or borrowed)
            const uint16_t *fp16_ptr = is_view_
                                           ? (parent_data_ptr_->data() + view_offset_)
                                           : host_fp16_data_.data();

            // Calculate element count from shape
            size_t element_count = 1;
            for (size_t dim : shape_)
            {
                element_count *= dim;
            }

            dequant_cache_.resize(element_count);
            to_fp32(dequant_cache_.data(), element_count);
        }
        return dequant_cache_.data();
    }

    float *FP16Tensor::mutable_data()
    {
        throw std::runtime_error("FP16Tensor::mutable_data: FP16 tensors are immutable (use from_fp32 to update)");
    }

    bool FP16Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            return false;
        }

        // Check shape compatibility
        if (src->shape() != shape_)
        {
            return false;
        }

        // Get FP32 data from source and convert to FP16
        const float *src_data = src->data();
        if (!src_data)
        {
            return false;
        }

        // Calculate element count from shape
        size_t element_count = 1;
        for (size_t dim : shape_)
        {
            element_count *= dim;
        }

        // Convert FP32 to FP16 and store
        from_fp32(src_data, element_count);

        // Clear cache to force re-dequantization
        dequant_cache_.clear();

        return true;
    }

    std::unique_ptr<ITensorGemm> FP16Tensor::createGemm()
    {
        throw std::runtime_error("FP16Tensor: GEMM not yet implemented");
    }

    std::unique_ptr<ITensorRoPE> FP16Tensor::createRoPE()
    {
        throw std::runtime_error("FP16Tensor: RoPE not supported");
    }

    std::unique_ptr<ITensorSwiGLU> FP16Tensor::createSwiGLU()
    {
        throw std::runtime_error("FP16Tensor: SwiGLU not supported");
    }

    std::unique_ptr<ITensorSoftmax> FP16Tensor::createSoftmax()
    {
        throw std::runtime_error("FP16Tensor: Softmax not supported");
    }

    std::unique_ptr<ITensorRMSNorm> FP16Tensor::createRMSNorm()
    {
        throw std::runtime_error("FP16Tensor: RMSNorm not supported");
    }

    // ========== FP16-Specific Interface ==========

    void FP16Tensor::from_fp32(const float *fp32_data, size_t count)
    {
        // Get the actual data pointer (owned or borrowed)
        uint16_t *fp16_ptr = is_view_
                                 ? (parent_data_ptr_->data() + view_offset_)
                                 : host_fp16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
        {
            throw std::invalid_argument("FP16Tensor::from_fp32: size mismatch");
        }

        // Convert FP32 → FP16
        // TODO: Add SIMD/hardware acceleration
        for (size_t i = 0; i < count; ++i)
        {
            // Round to nearest even for FP32 → FP16
            uint32_t bits;
            std::memcpy(&bits, &fp32_data[i], sizeof(float));

            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112; // Signed!
            uint32_t mantissa = (bits >> 13) & 0x3FF;

            if (exponent <= 0)
            {
                // Subnormal or zero
                fp16_ptr[i] = static_cast<uint16_t>(sign);
            }
            else if (exponent >= 0x1F)
            {
                // Overflow → infinity
                fp16_ptr[i] = static_cast<uint16_t>(sign | 0x7C00);
            }
            else
            {
                // Normal value
                fp16_ptr[i] = static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
            }
        }

        // Invalidate FP32 cache
        dequant_cache_.clear();
    }

    void FP16Tensor::to_fp32(float *fp32_data, size_t count) const
    {
        // Get the actual data pointer (owned or borrowed)
        const uint16_t *fp16_ptr = is_view_
                                       ? (parent_data_ptr_->data() + view_offset_)
                                       : host_fp16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
        {
            throw std::invalid_argument("FP16Tensor::to_fp32: size mismatch");
        }

        // Convert FP16 → FP32 using utility function
        for (size_t i = 0; i < count; ++i)
        {
            fp32_data[i] = fp16_to_fp32(fp16_ptr[i]);
        }
    }

    bool FP16Tensor::sync_to_device()
    {
        // TODO: Implement device upload
        return false;
    }

    bool FP16Tensor::sync_from_device()
    {
        // TODO: Implement device download
        return false;
    }

    std::shared_ptr<TensorBase> FP16Tensor::create_view(
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
            LOG_ERROR("[FP16Tensor::create_view] ERROR: offset " << offset
                                                                 << " >= parent size " << parent_elements);
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            LOG_ERROR("[FP16Tensor::create_view] ERROR: view size " << view_elements
                                                                    << " > available elements " << available_elements
                                                                    << " (offset=" << offset << ", parent_size=" << parent_elements << ")");
            return nullptr;
        }

        // Determine the actual parent tensor and data pointer
        // If this is already a view, chain to the root parent
        std::shared_ptr<FP16Tensor> root_parent;
        if (is_view_)
        {
            root_parent = std::dynamic_pointer_cast<FP16Tensor>(parent_);
            if (!root_parent)
            {
                LOG_ERROR("[FP16Tensor::create_view] ERROR: Failed to cast parent to FP16Tensor (is_view=true)");
                return nullptr;
            }
        }
        else
        {
            // Get a proper shared_ptr to this object (increments ref count)
            try
            {
                auto self_ptr = shared_from_this();
                root_parent = std::dynamic_pointer_cast<FP16Tensor>(self_ptr);
                if (!root_parent)
                {
                    LOG_ERROR("[FP16Tensor::create_view] ERROR: Failed to cast shared_from_this to FP16Tensor");
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[FP16Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!");
                LOG_ERROR("[FP16Tensor::create_view] Exception: " << e.what());
                return nullptr;
            }
        }

        std::vector<uint16_t> *root_data = is_view_ ? parent_data_ptr_ : &host_fp16_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<FP16Tensor>(new FP16Tensor(
            new_shape,
            device_idx_,
            root_data,
            root_offset,
            root_parent));

        return view_tensor;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    // ===== Format Conversion Methods (TensorBase interface) =====

    void FP16Tensor::to_fp32(float *dst) const
    {
        const size_t count = element_count();
        const uint16_t *src = fp16_data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp16_to_fp32(src[i]);
        }
    }

    void FP16Tensor::to_bf16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to BF16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(temp_fp32[i]);
        }
    }

    void FP16Tensor::to_fp16(uint16_t *dst) const
    {
        // FP16 → FP16 is just a memcpy
        const size_t count = element_count();
        const uint16_t *src = fp16_data();
        std::memcpy(dst, src, count * sizeof(uint16_t));
    }

    void FP16Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Decode to FP32 first, then quantize to int8
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = temp_fp32[offset + i] * scale;
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

    void FP16Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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
        const uint16_t *src = fp16_data();
        const uint16_t *row_src = src + row_idx * cols;

        for (size_t i = 0; i < cols; ++i)
        {
            buffer[i] = fp16_to_fp32(row_src[i]);
        }
    }

    void FP16Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        const uint16_t *src = fp16_data();
        for (size_t i = 0; i < count; ++i)
        {
            buffer[i] = fp16_to_fp32(src[offset + i]);
        }
    }

} // namespace llaminar2
