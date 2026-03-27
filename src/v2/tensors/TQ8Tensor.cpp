/**
 * @file TQ8Tensor.cpp
 * @brief TurboQuant 8-bit tensor implementation
 */

#include "TQ8Tensor.h"
#include "../kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "../kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../utils/Logger.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    // =====================================================================
    // Constructor
    // =====================================================================

    TQ8Tensor::TQ8Tensor(const std::vector<size_t> &shape, int head_dim, DeviceId device)
        : shape_(shape), head_dim_(head_dim), device_(device)
    {
        if (shape.size() < 2)
        {
            throw std::invalid_argument("TQ8Tensor: shape must have at least 2 dimensions [rows, kv_dim]");
        }
        if (head_dim <= 0 || head_dim % 2 != 0)
        {
            throw std::invalid_argument("TQ8Tensor: head_dim must be positive and even");
        }

        const size_t kv_dim = shape[shape.size() - 1];
        if (kv_dim % static_cast<size_t>(head_dim) != 0)
        {
            throw std::invalid_argument("TQ8Tensor: kv_dim must be divisible by head_dim");
        }

        block_bytes_ = (head_dim == 128)
                           ? sizeof(TQ8Block_128)
                           : sizeof(TQ8Block_64);
        blocks_per_row_ = kv_dim / static_cast<size_t>(head_dim);

        // Calculate total rows
        size_t num_rows = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i)
        {
            num_rows *= shape[i];
        }

        // Pre-allocate raw block storage
        raw_blocks_.resize(num_rows * blocks_per_row_ * block_bytes_, 0);
    }

    // =====================================================================
    // TensorBase interface
    // =====================================================================

    const float *TQ8Tensor::data() const
    {
        if (dequant_cache_valid_)
        {
            return dequant_cache_.data();
        }

        if (!turboquant_ctx_)
        {
            LOG_ERROR("[TQ8Tensor::data] No TurboQuant context set — cannot dequantize");
            return nullptr;
        }

        const size_t num_rows = rows();
        const size_t kv_dim = cols();
        dequant_cache_.resize(num_rows * kv_dim);
        dequantize_to_fp32(dequant_cache_.data(), *turboquant_ctx_);
        dequant_cache_valid_ = true;
        return dequant_cache_.data();
    }

    float *TQ8Tensor::mutable_data()
    {
        invalidate_dequant_cache();

        const size_t num_rows = rows();
        const size_t kv_dim = cols();
        dequant_cache_.resize(num_rows * kv_dim);

        if (!turboquant_ctx_)
        {
            std::fill(dequant_cache_.begin(), dequant_cache_.end(), 0.0f);
        }
        else
        {
            dequantize_to_fp32(dequant_cache_.data(), *turboquant_ctx_);
        }
        return dequant_cache_.data();
    }

    bool TQ8Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
            return false;

        const auto *other = dynamic_cast<const TQ8Tensor *>(src);
        if (!other)
            return false;

        if (other->head_dim_ != head_dim_ || other->shape_ != shape_)
            return false;

        raw_blocks_ = other->raw_blocks_;
        turboquant_ctx_ = other->turboquant_ctx_;
        invalidate_dequant_cache();
        return true;
    }

    void TQ8Tensor::release_raw_data()
    {
        std::vector<uint8_t>().swap(raw_blocks_);
        raw_data_released_ = true;
    }

    void TQ8Tensor::to_fp32(float *dst) const
    {
        if (!turboquant_ctx_)
        {
            LOG_ERROR("[TQ8Tensor::to_fp32] No TurboQuant context set");
            return;
        }
        dequantize_to_fp32(dst, *turboquant_ctx_);
    }

    void TQ8Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        if (!turboquant_ctx_)
        {
            LOG_ERROR("[TQ8Tensor::to_fp32_row] No TurboQuant context set");
            return;
        }
        const size_t bpr = blocks_per_row_;
        const size_t bb = block_bytes_;
        const uint8_t *row_src = raw_blocks_.data() + row_idx * bpr * bb;
        alignas(64) float scratch[128];

        for (size_t h = 0; h < bpr; ++h)
        {
            float *head_dst = buffer + h * static_cast<size_t>(head_dim_);
            const uint8_t *block_src = row_src + h * bb;

            if (head_dim_ == 128)
            {
                const auto &head_ctx = turboquant_ctx_->for_layer(static_cast<int>(h));
                const TQ8Block_128 *block = reinterpret_cast<const TQ8Block_128 *>(block_src);
                turboquant_dequantize_tq8<128>(*block, head_ctx, head_dst, scratch);
            }
            else if (head_dim_ == 64)
            {
                const auto &head_ctx = turboquant_ctx_->for_layer(static_cast<int>(h));
                const TQ8Block_64 *block = reinterpret_cast<const TQ8Block_64 *>(block_src);
                turboquant_dequantize_tq8<64>(*block, head_ctx, head_dst, scratch);
            }
        }
    }

    void TQ8Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        const size_t kv_dim = cols();
        std::vector<float> row_buf(kv_dim);
        size_t written = 0;
        while (written < count)
        {
            const size_t elem = offset + written;
            const size_t row = elem / kv_dim;
            const size_t col = elem % kv_dim;
            to_fp32_row(row, row_buf.data());
            const size_t n = std::min(count - written, kv_dim - col);
            std::memcpy(buffer + written, row_buf.data() + col, n * sizeof(float));
            written += n;
        }
    }

    // =====================================================================
    // Quantization/Dequantization
    // =====================================================================

    std::shared_ptr<TQ8Tensor> TQ8Tensor::quantize_from_fp32(
        const float *src,
        const std::vector<size_t> &shape,
        int head_dim,
        const TurboQuantContext &turboquant_ctx)
    {
        if (!src || shape.size() < 2)
        {
            throw std::invalid_argument("TQ8Tensor::quantize_from_fp32: invalid arguments");
        }

        auto tensor = std::make_shared<TQ8Tensor>(shape, head_dim);
        tensor->set_turboquant_context(&turboquant_ctx);

        const size_t num_rows = tensor->rows();
        const size_t kv_dim = tensor->cols();
        const size_t bpr = tensor->blocks_per_row();
        const size_t bb = tensor->block_bytes();

#pragma omp parallel for if (num_rows * bpr >= 32)
        for (size_t r = 0; r < num_rows; ++r)
        {
            const float *row_src = src + r * kv_dim;
            uint8_t *row_dst = tensor->raw_blocks_.data() + r * bpr * bb;
            alignas(64) float scratch0[128];
            alignas(64) float scratch1[128];

            for (size_t h = 0; h < bpr; ++h)
            {
                const float *head_src = row_src + h * static_cast<size_t>(head_dim);
                uint8_t *block_dst = row_dst + h * bb;

                if (head_dim == 128)
                {
                    TQ8Block_128 *block = reinterpret_cast<TQ8Block_128 *>(block_dst);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_quantize_tq8<128>(head_src, head_ctx, *block, scratch0, scratch1);
                }
                else if (head_dim == 64)
                {
                    TQ8Block_64 *block = reinterpret_cast<TQ8Block_64 *>(block_dst);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_quantize_tq8<64>(head_src, head_ctx, *block, scratch0, scratch1);
                }
            }
        }

        return tensor;
    }

    bool TQ8Tensor::copyFrom_fp32_rows(const float *src_data, size_t num_rows, const TurboQuantContext &turboquant_ctx)
    {
        if (!src_data || num_rows > rows())
            return false;

        const size_t kv_dim = cols();
        const size_t bpr = blocks_per_row_;
        const size_t bb = block_bytes_;

#pragma omp parallel for if (num_rows * bpr >= 32)
        for (size_t r = 0; r < num_rows; ++r)
        {
            const float *row_src = src_data + r * kv_dim;
            uint8_t *row_dst = raw_blocks_.data() + r * bpr * bb;
            alignas(64) float scratch0[128];
            alignas(64) float scratch1[128];

            for (size_t h = 0; h < bpr; ++h)
            {
                const float *head_src = row_src + h * static_cast<size_t>(head_dim_);
                uint8_t *block_dst = row_dst + h * bb;

                if (head_dim_ == 128)
                {
                    TQ8Block_128 *block = reinterpret_cast<TQ8Block_128 *>(block_dst);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_quantize_tq8<128>(head_src, head_ctx, *block, scratch0, scratch1);
                }
                else if (head_dim_ == 64)
                {
                    TQ8Block_64 *block = reinterpret_cast<TQ8Block_64 *>(block_dst);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_quantize_tq8<64>(head_src, head_ctx, *block, scratch0, scratch1);
                }
            }
        }

        turboquant_ctx_ = &turboquant_ctx;
        invalidate_dequant_cache();
        return true;
    }

    void TQ8Tensor::dequantize_to_fp32(float *dst, const TurboQuantContext &turboquant_ctx) const
    {
        const size_t num_rows = rows();
        const size_t kv_dim = cols();
        const size_t bpr = blocks_per_row_;
        const size_t bb = block_bytes_;

#pragma omp parallel for if (num_rows * bpr >= 32)
        for (size_t r = 0; r < num_rows; ++r)
        {
            float *row_dst = dst + r * kv_dim;
            const uint8_t *row_src = raw_blocks_.data() + r * bpr * bb;
            alignas(64) float scratch[128];

            for (size_t h = 0; h < bpr; ++h)
            {
                float *head_dst = row_dst + h * static_cast<size_t>(head_dim_);
                const uint8_t *block_src = row_src + h * bb;

                if (head_dim_ == 128)
                {
                    const TQ8Block_128 *block = reinterpret_cast<const TQ8Block_128 *>(block_src);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_dequantize_tq8<128>(*block, head_ctx, head_dst, scratch);
                }
                else if (head_dim_ == 64)
                {
                    const TQ8Block_64 *block = reinterpret_cast<const TQ8Block_64 *>(block_src);
                    const auto &head_ctx = turboquant_ctx.for_layer(static_cast<int>(h));
                    turboquant_dequantize_tq8<64>(*block, head_ctx, head_dst, scratch);
                }
            }
        }
    }

} // namespace llaminar2
