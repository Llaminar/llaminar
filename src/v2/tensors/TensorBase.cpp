/**
 * @file TensorBase.cpp
 * @brief TensorBase class implementation (helper methods)
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace llaminar2
{

    void TensorBase::to_fp32_via_blocks(float *dst) const
    {
        // This helper is for quantized tensors that implement IBlockDecoder
        const IBlockDecoder *decoder = dynamic_cast<const IBlockDecoder *>(this);
        if (!decoder)
        {
            throw std::runtime_error("to_fp32_via_blocks() called on non-IBlockDecoder tensor");
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
        // Verify this is an IBlockDecoder tensor
        const IBlockDecoder *decoder = dynamic_cast<const IBlockDecoder *>(this);
        if (!decoder)
        {
            LOG_ERROR("[TensorBase] to_int8_perchannel_via_blocks() requires IBlockDecoder interface");
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

        // Step 4: Quantize to INT8 using per-column scales
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                const float inv_scale = 1.0f / dst_col_scales[j];
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

    // ===== Template Specializations for to<T>() =====

    // Forward declare helper for BF16/FP16 conversion
    namespace simd
    {
        extern float bf16_to_fp32(uint16_t bf16);
        extern float fp16_to_fp32(uint16_t fp16);
        extern uint16_t fp32_to_bf16(float fp32);
        extern uint16_t fp32_to_fp16(float fp32);
    }

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

    // Explicit template instantiations
    template void TensorBase::to<float>(float *dst, TensorType format) const;
    template void TensorBase::to<uint16_t>(uint16_t *dst, TensorType format) const;
    template void TensorBase::to<int8_t>(int8_t *dst, TensorType format) const;
    template void TensorBase::to<int32_t>(int32_t *dst, TensorType format) const;

} // namespace llaminar2
