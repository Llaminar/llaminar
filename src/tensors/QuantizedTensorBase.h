#pragma once

#include "TensorBase.h"
#include "../utils/BFloat16.h"
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>

// Forward declarations to avoid circular includes
namespace llaminar
{
    struct QuantBlockDescriptor; // Defined in TensorFactory.h
}

namespace llaminar
{
    /**
     * @brief Quantization format enumeration
     *
     * Matches GGML quantization types for compatibility.
     */
    enum class QuantType : uint8_t
    {
        Q4_0, ///< 4-bit uniform quantization (32 elements per block)
        Q4_K, ///< 4-bit K-quant (256 elements per block)
        Q5_0, ///< 5-bit uniform quantization (32 elements per block)
        Q5_K, ///< 5-bit K-quant (256 elements per block)
        Q6_K, ///< 6-bit K-quant (256 elements per block)
        Q8_0, ///< 8-bit uniform quantization (32 elements per block)
        Q8_K, ///< 8-bit K-quant (256 elements per block)
    };

    /**
     * @brief Abstract base class for quantized tensor formats
     *
     * Stores data in native compressed format. Provides streaming decode API
     * for row-wise or panel-wise access during computation.
     *
     * Design principles:
     * - Lazy dequantization: decode only when computing
     * - Streaming decode: row-by-row or panel-by-panel
     * - Type safety: cannot call data() - must use decodeRow()
     * - No global cache: each tensor is self-contained
     *
     * @author David Sanftenberg
     */
    class QuantizedTensorBase : public TensorBase
    {
    public:
        virtual ~QuantizedTensorBase() = default;

        // ===== Quantization Metadata =====

        /**
         * @brief Get quantization format (Q4_0, Q8_0, etc.)
         */
        virtual QuantType quant_type() const = 0;

        /**
         * @brief Get compression ratio (e.g., 4.0 for Q8_0, 8.0 for Q4_0)
         */
        virtual float compression_ratio() const = 0;

        // ===== Data Access (NOT SUPPORTED - use decode methods) =====

        /**
         * @brief FP32 data pointer - NOT SUPPORTED for quantized tensors
         * @throws std::runtime_error Always throws - use decodeRow/decodePanel instead
         */
        float *data() override
        {
            throw std::runtime_error("Cannot get FP32 pointer for quantized tensor - use decodeRow()");
        }

        const float *data() const override
        {
            throw std::runtime_error("Cannot get FP32 pointer for quantized tensor - use decodeRow()");
        }

        // ===== TensorBase Implementations =====

        std::string type_name() const override
        {
            return "QuantizedTensor[" + std::string(quant_type_name(quant_type())) + "]";
        }

        bool is_distributed() const override { return false; }
        void zero() override
        {
            throw std::runtime_error("Cannot zero quantized tensor - quantization is lossy");
        }
        void fill(float) override
        {
            throw std::runtime_error("Cannot fill quantized tensor - quantization is lossy");
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }

        // ===== Streaming Decode API (Primary Interface) =====

        /**
         * @brief Decode a single row to FP32
         *
         * This is the primary interface for row-wise operations (e.g., GEMM).
         * Buffer must be pre-allocated with at least shape()[1] elements.
         *
         * @param row_idx Row index (0 to shape()[0]-1)
         * @param buffer Pre-allocated FP32 buffer (size: shape()[1])
         */
        virtual void decodeRow(size_t row_idx, float *buffer) const = 0;

        /**
         * @brief Decode a single row to BF16
         *
         * Optional optimization: decode directly to BF16 without FP32 intermediate.
         * Default implementation decodes to FP32 then converts.
         *
         * @param row_idx Row index
         * @param buffer Pre-allocated BF16 buffer (size: shape()[1])
         */
        virtual void decodeRowToBF16(size_t row_idx, void *buffer) const
        {
            // Default: decode to FP32, then convert to BF16
            std::vector<float> fp32_buffer(shape()[1]);
            decodeRow(row_idx, fp32_buffer.data());

            bfloat16 *bf16_buffer = static_cast<bfloat16 *>(buffer);
            for (int i = 0; i < shape()[1]; i++)
            {
                bf16_buffer[i] = bfloat16::from_float(fp32_buffer[i]);
            }
        }

        /**
         * @brief Decode multiple rows as a panel
         *
         * For operations that benefit from batch decode (cache blocking, vectorization).
         *
         * @param row_start First row index
         * @param row_count Number of rows
         * @param buffer Pre-allocated buffer (size: row_count × shape()[1])
         */
        virtual void decodePanel(size_t row_start, size_t row_count, float *buffer) const
        {
            // Default: decode rows individually
            int cols = shape()[1];
            for (size_t i = 0; i < row_count; i++)
            {
                decodeRow(row_start + i, buffer + i * cols);
            }
        }

        /**
         * @brief Decode arbitrary span of elements
         *
         * For operations that need non-row-aligned access.
         * Offset is in logical element index (not block index).
         *
         * @param offset Starting element index
         * @param count Number of elements
         * @param buffer Pre-allocated FP32 buffer (size: count)
         */
        virtual void decodeSpan(size_t offset, size_t count, float *buffer) const = 0;

        // ===== Raw Block Access (Advanced) =====

        /**
         * @brief Get raw quantized data pointer
         *
         * For operations that want to implement custom decode logic.
         * Format is quantization-type-specific.
         */
        virtual const uint8_t *raw_data() const = 0;

        /**
         * @brief Get size of raw data in bytes
         */
        virtual size_t raw_size() const = 0;

        /**
         * @brief Get block descriptor
         *
         * Describes quantization block structure (elements per block, scale/offset layout).
         */
        virtual const QuantBlockDescriptor &block_descriptor() const = 0;

    protected:
        /**
         * @brief Helper: Convert FP16 to FP32
         *
         * Used by quantization formats that store scales as FP16.
         */
        static float fp16_to_fp32(uint16_t h)
        {
            // Portable half->float conversion
            uint32_t w = (uint32_t)h << 16;
            uint32_t sign = w & 0x80000000u;
            uint32_t two_w = w + w;
            uint32_t exp_offset = 0xE0u << 23;
            float exp_scale = 0x1.0p-112f;

            union
            {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = (two_w >> 4) + exp_offset;
            float normalized = cvt.f * exp_scale;

            uint32_t magic_mask = 126u << 23;
            float magic_bias = 0.5f;
            cvt.u = (two_w >> 17) | magic_mask;
            float denorm = cvt.f - magic_bias;

            uint32_t denorm_cut = 1u << 27;
            float result = (two_w < denorm_cut) ? denorm : normalized;

            union
            {
                uint32_t u;
                float f;
            } out;
            out.u = sign | ((uint32_t &)result & 0x7FFFFFFFu);
            return result;
        }

        /**
         * @brief Get quantization type name as string
         */
        static const char *quant_type_name(QuantType type)
        {
            switch (type)
            {
            case QuantType::Q4_0:
                return "Q4_0";
            case QuantType::Q4_K:
                return "Q4_K";
            case QuantType::Q5_0:
                return "Q5_0";
            case QuantType::Q5_K:
                return "Q5_K";
            case QuantType::Q6_K:
                return "Q6_K";
            case QuantType::Q8_0:
                return "Q8_0";
            case QuantType::Q8_K:
                return "Q8_K";
            default:
                return "UNKNOWN";
            }
        }
    };

} // namespace llaminar
