/**
 * @file BF16Tensor.cpp
 * @brief BF16 tensor implementation (Brain Float 16)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/Logger.h"
#include "../utils/BFloat16.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include "../kernels/cpu/BF16GemmKernel.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace llaminar2
{
    // ========== Constructors ==========

    BF16Tensor::BF16Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("BF16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        host_bf16_data_.resize(total, 0);
    }

    BF16Tensor::BF16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &bf16_data)
        : shape_(shape), device_idx_(-1), device_data_(nullptr), host_bf16_data_(bf16_data),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("BF16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        if (bf16_data.size() != total)
        {
            throw std::invalid_argument("BF16Tensor: data size mismatch");
        }
    }

    // Private view constructor
    BF16Tensor::BF16Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           std::vector<uint16_t> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<BF16Tensor> parent)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_bf16_data_
        // They borrow from the parent via parent_data_ptr_
    }

    BF16Tensor::~BF16Tensor()
    {
        // TODO: Free device memory when device support is added
    }

    // ========== TensorBase Interface ==========

    bool BF16Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        // TODO: Upload to device when device support is added
        return true;
    }

    const float *BF16Tensor::data() const
    {
        // Lazy dequantization to FP32 cache
        if (dequant_cache_.empty())
        {
            // Get the actual data pointer (owned or borrowed)
            const uint16_t *bf16_ptr = is_view_
                                           ? (parent_data_ptr_->data() + view_offset_)
                                           : host_bf16_data_.data();

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

    float *BF16Tensor::mutable_data()
    {
        throw std::runtime_error("BF16Tensor::mutable_data: BF16 tensors are immutable (use from_fp32 to update)");
    }

    bool BF16Tensor::copyFrom(const TensorBase *src)
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

        // Get FP32 data from source and convert to BF16
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

        // Convert FP32 to BF16 and store
        from_fp32(src_data, element_count);

        // Clear cache to force re-dequantization
        dequant_cache_.clear();

        return true;
    }

    std::unique_ptr<ITensorGemm> BF16Tensor::createGemm()
    {
        return std::make_unique<BF16GemmKernel>(this);
    }

    std::unique_ptr<ITensorRoPE> BF16Tensor::createRoPE()
    {
        throw std::runtime_error("BF16Tensor: RoPE not supported");
    }

    std::unique_ptr<ITensorSwiGLU> BF16Tensor::createSwiGLU()
    {
        throw std::runtime_error("BF16Tensor: SwiGLU not supported");
    }

    std::unique_ptr<ITensorSoftmax> BF16Tensor::createSoftmax()
    {
        throw std::runtime_error("BF16Tensor: Softmax not supported");
    }

    std::unique_ptr<ITensorRMSNorm> BF16Tensor::createRMSNorm()
    {
        throw std::runtime_error("BF16Tensor: RMSNorm not supported");
    }

    std::unique_ptr<ITensorAttention> BF16Tensor::createAttention()
    {
        LOG_ERROR("[BF16Tensor] createAttention not supported for quantized tensors");
        return nullptr;
    }

    // ========== BF16-Specific Interface ==========

    void BF16Tensor::from_fp32(const float *fp32_data, size_t count)
    {
        // Get the actual data pointer (owned or borrowed)
        uint16_t *bf16_ptr = is_view_
                                 ? (parent_data_ptr_->data() + view_offset_)
                                 : host_bf16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
        {
            throw std::invalid_argument("BF16Tensor::from_fp32: size mismatch");
        }

        // Convert FP32 → BF16 using utility
        for (size_t i = 0; i < count; ++i)
        {
            bfloat16 bf = bfloat16::from_float(fp32_data[i]);
            bf16_ptr[i] = bf.data;
        }

        // Invalidate FP32 cache
        dequant_cache_.clear();
    }

    void BF16Tensor::to_fp32(float *fp32_data, size_t count) const
    {
        // Get the actual data pointer (owned or borrowed)
        const uint16_t *bf16_ptr = is_view_
                                       ? (parent_data_ptr_->data() + view_offset_)
                                       : host_bf16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
        {
            throw std::invalid_argument("BF16Tensor::to_fp32: size mismatch");
        }

        // Convert BF16 → FP32
        for (size_t i = 0; i < count; ++i)
        {
            bfloat16 bf;
            bf.data = bf16_ptr[i];
            fp32_data[i] = bf.to_float();
        }
    }

    bool BF16Tensor::sync_to_device()
    {
        // TODO: Implement device upload
        return false;
    }

    bool BF16Tensor::sync_from_device()
    {
        // TODO: Implement device download
        return false;
    }

    std::shared_ptr<TensorBase> BF16Tensor::create_view(
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
            LOG_ERROR("[BF16Tensor::create_view] ERROR: offset " << offset << " >= parent size " << parent_elements);
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            LOG_ERROR("[BF16Tensor::create_view] ERROR: view size " << view_elements
                                                                    << " > available elements " << available_elements
                                                                    << " (offset=" << offset << ", parent_size=" << parent_elements << ")");
            return nullptr;
        }

        // Determine the actual parent tensor and data pointer
        // If this is already a view, chain to the root parent
        std::shared_ptr<BF16Tensor> root_parent;
        if (is_view_)
        {
            root_parent = std::dynamic_pointer_cast<BF16Tensor>(parent_);
            if (!root_parent)
            {
                LOG_ERROR("[BF16Tensor::create_view] ERROR: Failed to cast parent to BF16Tensor (is_view=true)");
                return nullptr;
            }
        }
        else
        {
            // Get a proper shared_ptr to this object (increments ref count)
            try
            {
                auto self_ptr = shared_from_this();
                root_parent = std::dynamic_pointer_cast<BF16Tensor>(self_ptr);
                if (!root_parent)
                {
                    LOG_ERROR("[BF16Tensor::create_view] ERROR: Failed to cast shared_from_this to BF16Tensor");
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[BF16Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!");
                LOG_ERROR("[BF16Tensor::create_view] Exception: " << e.what());
                return nullptr;
            }
        }

        std::vector<uint16_t> *root_data = is_view_ ? parent_data_ptr_ : &host_bf16_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<BF16Tensor>(new BF16Tensor(
            new_shape,
            device_idx_,
            root_data,
            root_offset,
            root_parent));

        return view_tensor;
    }

    // ===== Format Conversion Methods =====

    void BF16Tensor::to_fp32(float *dst) const
    {
        const size_t count = element_count();
        const uint16_t *src = bf16_data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::bf16_to_fp32(src[i]);
        }
    }

    void BF16Tensor::to_bf16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const uint16_t *src = bf16_data();
        std::memcpy(dst, src, count * sizeof(uint16_t));
    }

    void BF16Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const uint16_t *src = bf16_data();

// BF16 → FP32 → FP16 (two-step conversion)
#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            float fp32_val = simd::bf16_to_fp32(src[i]);
            dst[i] = fp32_to_fp16(fp32_val);
        }
    }

    void BF16Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Convert to FP32 first, then quantize
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

    bool BF16Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        // BF16 → FP32 → INT8 per-channel quantization
        // Use the generic conversion path since BF16 doesn't implement IBlockDecoder

        if (shape_.size() != 2)
        {
            LOG_ERROR("[BF16Tensor::to_int8_perchannel] Only 2D tensors supported");
            return false;
        }

        const size_t rows = shape_[0];
        const size_t cols = shape_[1];

        // Decode entire tensor to FP32
        std::vector<float> temp_fp32(rows * cols);
        to_fp32(temp_fp32.data());

        // Compute per-column scales
        std::fill(dst_col_scales, dst_col_scales + cols, 0.0f);
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[i * cols + j]));
            }
            dst_col_scales[j] = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Compute per-row scales if requested
        if (dst_row_scales)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                float max_abs = 0.0f;
                for (size_t j = 0; j < cols; ++j)
                {
                    max_abs = std::max(max_abs, std::abs(temp_fp32[i * cols + j]));
                }
                dst_row_scales[i] = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1.0f;
            }
        }

        // Quantize using per-column scales
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                const float inv_scale = 1.0f / dst_col_scales[j];
                const float scaled = temp_fp32[idx] * inv_scale;
                const int32_t quantized = static_cast<int32_t>(std::round(scaled));

                // Clamp to INT8 range
                dst_int8[idx] = static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
        }

        return true;
    }

    void BF16Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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
        const uint16_t *src = bf16_data();
        const uint16_t *row_src = src + row_idx * cols;

        for (size_t i = 0; i < cols; ++i)
        {
            buffer[i] = simd::bf16_to_fp32(row_src[i]);
        }
    }

    void BF16Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        const uint16_t *src = bf16_data();
        for (size_t i = 0; i < count; ++i)
        {
            buffer[i] = simd::bf16_to_fp32(src[offset + i]);
        }
    }

} // namespace llaminar2
