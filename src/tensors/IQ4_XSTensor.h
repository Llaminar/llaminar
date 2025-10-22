/**
 * @file IQ4_XSTensor.h
 * @brief IQ4_XS (Extra Small 4-bit) quantized tensor implementation
 *
 * IQ4_XS is a 4-bit quantization format with sub-block scaling:
 * - Block size: 256 elements (8 sub-blocks of 32 elements each)
 * - 16-entry lookup table (kvalues_iq4nl, shared with IQ4_NL)
 * - 8 × 6-bit sub-block scales (packed)
 * - 4.25 bits per weight (136 bytes per 256 elements)
 * - Compression ratio: ~7.5× vs FP32
 *
 * Block structure (136 bytes):
 *   - d (2 bytes): FP16 global scale factor
 *   - scales_h (2 bytes): High 2 bits of 8 sub-block scales (2 bits × 8 = 16 bits)
 *   - scales_l[4] (4 bytes): Low 4 bits of 8 sub-block scales (2 scales per byte)
 *   - qs[128] (128 bytes): Packed 4-bit indices (2 per byte, 16 bytes per sub-block)
 *
 * Scale extraction (6-bit scale per sub-block):
 *   For sub-block i (0-7):
 *     low_4_bits  = (scales_l[i/2] >> (4*(i%2))) & 0xF
 *     high_2_bits = (scales_h >> (2*i)) & 0x3
 *     scale_value = (high_2_bits << 4) | low_4_bits  // 6-bit value (0-63)
 *     effective_scale = global_scale * (scale_value - 32)  // Centered at 32
 *
 * Reference: ggml-quants.c line 2528 (dequantize_row_iq4_xs)
 *
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#pragma once

#include "QuantizedTensorBase.h"
#include "TensorFactory.h"
#include "IQQuantTables.h"
#include "../utils/SIMDHelpers.h"
#include <vector>
#include <cstring>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar
{

    /**
     * @brief IQ4_XS block structure (136 bytes, 256 elements)
     *
     * Matches GGML block_iq4_xs from ggml-common.h.
     */
    struct IQ4_XSBlock
    {
        uint16_t d;          ///< FP16 global scale factor
        uint16_t scales_h;   ///< High 2 bits of 8 sub-block scales
        uint8_t scales_l[4]; ///< Low 4 bits of 8 sub-block scales (2 per byte)
        uint8_t qs[128];     ///< Packed 4-bit indices (2 per byte)

        static constexpr size_t BLOCK_SIZE = 256; ///< Elements per block
    };

    static_assert(sizeof(IQ4_XSBlock) == 136, "IQ4_XSBlock must be 136 bytes");

    /**
     * @brief IQ4_XS quantized tensor (4.25 bpw, 7.5× compression)
     *
     * Implements 4-bit quantization with 8 sub-blocks per block.
     */
    class IQ4_XSTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct IQ4_XS tensor from shape and raw data
         *
         * @param shape Tensor dimensions (2D: [rows, cols])
         * @param raw_data Raw bytes (IQ4_XS blocks)
         * @throws std::invalid_argument if shape not 2D or size mismatch
         */
        IQ4_XSTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {

            if (shape_.size() != 2)
            {
                throw std::invalid_argument("IQ4_XSTensor only supports 2D tensors");
            }

            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(IQ4_XSBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "IQ4_XS raw data size mismatch: expected " + std::to_string(expected_size) +
                    " bytes, got " + std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ========== Shape and Metadata ==========

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::IQ4_XS; }
        float compression_ratio() const override { return 7.5f; }

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        // ========== Decode API ==========

        void decode_to_fp32(float *dst) const override
        {
            int rows = shape_[0];
            int cols = shape_[1];
            size_t num_blocks = (rows * cols + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());

// Parallel block-level decode for better load balancing
#pragma omp parallel for schedule(static) if (num_blocks > 8)
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
            {
                size_t element_offset = block_idx * IQ4_XSBlock::BLOCK_SIZE;
                size_t elements_to_decode = std::min(
                    static_cast<size_t>(IQ4_XSBlock::BLOCK_SIZE),
                    static_cast<size_t>(rows * cols) - element_offset);
                decodeBlock(blocks[block_idx], dst + element_offset);
            }
        }

        void decode_to_bf16(void *dst) const override
        {
            int rows = shape_[0];
            int cols = shape_[1];
            bfloat16 *bf16_dst = static_cast<bfloat16 *>(dst);
#pragma omp parallel for if (rows > 4)
            for (int row = 0; row < rows; ++row)
            {
                decodeRowToBF16(row, bf16_dst + row * cols);
            }
        }

        std::shared_ptr<TensorBase> copy() const override
        {
            return std::make_shared<IQ4_XSTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("IQ4_XSTensor::copy_from not supported - quantization is lossy");
        }

        // ========== Streaming Decode API ==========

        void decodeRow(size_t row_idx, float *buffer) const override
        {
            int cols = shape_[1];
            size_t global_start = row_idx * cols;
            size_t global_end = global_start + cols;

            size_t start_block = global_start / IQ4_XSBlock::BLOCK_SIZE;
            size_t end_block = (global_end - 1) / IQ4_XSBlock::BLOCK_SIZE;

            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());

            if (start_block == end_block)
            {
                // Single block case
                size_t offset_in_block = global_start % IQ4_XSBlock::BLOCK_SIZE;
                float temp[IQ4_XSBlock::BLOCK_SIZE];
                decodeBlock(blocks[start_block], temp);
                std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
            }
            else
            {
                // Multi-block case
                size_t offset = 0;
                for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
                {
                    float temp[IQ4_XSBlock::BLOCK_SIZE];
                    decodeBlock(blocks[block_idx], temp);

                    size_t block_start = block_idx * IQ4_XSBlock::BLOCK_SIZE;
                    size_t copy_start = std::max(global_start, block_start) - block_start;
                    size_t copy_end = std::min(global_end, block_start + IQ4_XSBlock::BLOCK_SIZE) - block_start;
                    size_t copy_count = copy_end - copy_start;

                    std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(float));
                    offset += copy_count;
                }
            }
        }

        void decodeSpan(size_t offset, size_t count, float *buffer) const override
        {
            if (offset + count > element_count())
            {
                throw std::out_of_range("IQ4_XSTensor::decodeSpan: range exceeds tensor bounds");
            }

            size_t start_block = offset / IQ4_XSBlock::BLOCK_SIZE;
            size_t end_block = (offset + count - 1) / IQ4_XSBlock::BLOCK_SIZE;

            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(raw_data_.data());

            size_t buffer_offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
            {
                float temp[IQ4_XSBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);

                size_t block_start = block_idx * IQ4_XSBlock::BLOCK_SIZE;
                size_t copy_start = std::max(offset, block_start) - block_start;
                size_t copy_end = std::min(offset + count, block_start + IQ4_XSBlock::BLOCK_SIZE) - block_start;
                size_t copy_count = copy_end - copy_start;

                std::memcpy(buffer + buffer_offset, temp + copy_start, copy_count * sizeof(float));
                buffer_offset += copy_count;
            }
        }

        // ========== Raw Block Access ==========

        const uint8_t *raw_data() const override
        {
            return raw_data_.data();
        }

        size_t raw_size() const override
        {
            return raw_data_.size();
        }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                256,  // elements_per_block
                136,  // bytes_per_block
                8,    // scale_count (8 sub-block scales)
                4,    // bits_per_value (4.25 bpw)
                false // is_k_quant
            };
            return desc;
        }

    private:
        std::vector<int> shape_;        ///< Tensor dimensions
        std::vector<uint8_t> raw_data_; ///< Raw quantized data

        /**
         * @brief Extract 6-bit scale for a sub-block
         *
         * @param block Block containing scales
         * @param sub_block_idx Sub-block index (0-7)
         * @return 6-bit scale value (0-63)
         */
        static inline int extractScale(const IQ4_XSBlock &block, size_t sub_block_idx)
        {
            // Low 4 bits from scales_l[] (2 scales per byte)
            const int low_4 = (block.scales_l[sub_block_idx / 2] >> (4 * (sub_block_idx % 2))) & 0x0F;

            // High 2 bits from scales_h (2 bits per scale)
            const int high_2 = (block.scales_h >> (2 * sub_block_idx)) & 0x03;

            // Combine to 6-bit value (0-63)
            return (high_2 << 4) | low_4;
        }

#if defined(__AVX512F__)
        /**
         * @brief AVX512-optimized IQ4_XS block decode
         *
         * Uses SIMD helper library for efficient int8 to float32 conversion.
         * Processes sub-blocks with 16 values at a time using AVX512.
         */
        static void decodeBlockAVX512(const IQ4_XSBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);

            const uint8_t *qs = block.qs;
            float *y = output;

            // Process 8 sub-blocks
            for (size_t ib = 0; ib < 8; ++ib)
            {
                // Extract 6-bit scale and compute effective scale
                const int scale_6bit = extractScale(block, ib);
                const float dl = d * static_cast<float>(scale_6bit - 32);

                // Prepare lookup buffer (32 int8 values)
                alignas(64) int8_t lookup_values[32];

                // Extract indices and lookup kvalues_iq4nl
                for (size_t j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = qs[j];
                    lookup_values[j] = kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
                    lookup_values[j + 16] = kvalues_iq4nl[qbyte >> 4]; // High nibble
                }

                // Convert and scale: 16 elements at a time (AVX512)
                simd::convert_i8_to_f32_scaled_avx512(lookup_values, dl, y);
                simd::convert_i8_to_f32_scaled_avx512(lookup_values + 16, dl, y + 16);

                y += 32;  // Move to next sub-block output
                qs += 16; // Move to next sub-block quantized data
            }
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2-optimized IQ4_XS block decode
         *
         * Uses SIMD helper library for efficient int8 to float32 conversion.
         * Processes sub-blocks with 8 values at a time using AVX2.
         */
        static void decodeBlockAVX2(const IQ4_XSBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);

            const uint8_t *qs = block.qs;
            float *y = output;

            // Process 8 sub-blocks
            for (size_t ib = 0; ib < 8; ++ib)
            {
                // Extract 6-bit scale and compute effective scale
                const int scale_6bit = extractScale(block, ib);
                const float dl = d * static_cast<float>(scale_6bit - 32);

                // Prepare lookup buffer (32 int8 values)
                alignas(32) int8_t lookup_values[32];

                // Extract indices and lookup kvalues_iq4nl
                for (size_t j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = qs[j];
                    lookup_values[j] = kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
                    lookup_values[j + 16] = kvalues_iq4nl[qbyte >> 4]; // High nibble
                }

                // Convert and scale: 8 elements at a time (AVX2)
                simd::convert_i8_to_f32_scaled_avx2(lookup_values, dl, y);
                simd::convert_i8_to_f32_scaled_avx2(lookup_values + 8, dl, y + 8);
                simd::convert_i8_to_f32_scaled_avx2(lookup_values + 16, dl, y + 16);
                simd::convert_i8_to_f32_scaled_avx2(lookup_values + 24, dl, y + 24);

                y += 32;  // Move to next sub-block output
                qs += 16; // Move to next sub-block quantized data
            }
        }
#endif

        /**
         * @brief Decode one IQ4_XS block (256 elements) to FP32
         *
         * Implements GGML dequantize_row_iq4_xs algorithm (ggml-quants.c line 2528).
         * Dispatches to AVX512/AVX2 version if available, otherwise uses scalar fallback.
         *
         * Algorithm:
         * 1. Extract FP16 global scale d
         * 2. Process 8 sub-blocks of 32 elements:
         *    a. Extract 6-bit scale for sub-block
         *    b. Compute effective_scale = d * (scale_value - 32)
         *    c. Process 16 bytes (each with 2 4-bit indices):
         *       - Low nibble  → output[j]
         *       - High nibble → output[j+16]
         *       - Lookup: kvalues_iq4nl[index]
         *       - Apply: y[j] = effective_scale * kvalues_iq4nl[index]
         *
         * @param block Input IQ4_XS block
         * @param output Output buffer (must have space for 256 floats)
         */
        static void decodeBlock(const IQ4_XSBlock &block, float *output)
        {
#if defined(__AVX512F__)
            if (simd::cpu_supports_avx512())
            {
                decodeBlockAVX512(block, output);
                return;
            }
#endif
#if defined(__AVX2__)
            if (simd::cpu_supports_avx2())
            {
                decodeBlockAVX2(block, output);
                return;
            }
#endif
            // Scalar fallback
            const float d = simd::fp16_to_fp32(block.d);

            const uint8_t *qs = block.qs;
            float *y = output;

            // Process 8 sub-blocks
            for (size_t ib = 0; ib < 8; ++ib)
            {
                // Extract 6-bit scale for this sub-block
                const int scale_6bit = extractScale(block, ib);

                // Effective scale: global_scale * (scale_value - 32)
                // Center value is 32, so scales range from -32 to +31
                const float dl = d * static_cast<float>(scale_6bit - 32);

// Decode 32 elements (16 bytes, 2 indices per byte)
#pragma omp simd
                for (size_t j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = qs[j];

                    // Low 4 bits -> first half of sub-block
                    const uint8_t idx_low = qbyte & 0x0F;
                    y[j] = dl * static_cast<float>(kvalues_iq4nl[idx_low]);

                    // High 4 bits -> second half of sub-block
                    const uint8_t idx_high = qbyte >> 4;
                    y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[idx_high]);
                }

                y += 32;  // Move to next sub-block output
                qs += 16; // Move to next sub-block quantized data
            }
        }

        /**
         * @brief Decode one IQ4_XS block to BF16
         */
        void decodeRowToBF16(size_t row_idx, bfloat16 *buffer) const
        {
            int cols = shape_[1];
            std::vector<float> temp(cols);
            decodeRow(row_idx, temp.data());

            for (int i = 0; i < cols; ++i)
            {
                buffer[i] = bfloat16(temp[i]);
            }
        }
    };

} // namespace llaminar
