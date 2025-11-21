/**
 * @file TensorBase.cpp
 * @brief TensorBase class implementation (helper methods)
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "SIMDHelpers.h"
#include "../utils/CPUFeatures.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <omp.h>

namespace llaminar2
{

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

} // namespace llaminar2
