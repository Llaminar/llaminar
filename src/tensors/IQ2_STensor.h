#pragma once

#include "QuantizedTensorBase.h"
#include "TensorFactory.h"
#include "IQQuantTables.h"
#include "../utils/BFloat16.h"
#include "../utils/SIMDHelpers.h"
#include <cstring>
#include <stdexcept>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar
{
    /**
     * @brief IQ2_S (2.5625-bit importance quantization) tensor format
     *
     * IQ2_S is the highest-quality variant in the IQ2 family, offering the best
     * compression-quality tradeoff at the cost of slightly larger block size and
     * more complex decode logic.
     *
     * **Block Structure (82 bytes per 256 elements):**
     * - d: 2 bytes (FP16 scale factor)
     * - qs[64]: 64 bytes (grid indices, low 8 bits)
     * - qh[8]: 8 bytes (grid indices, high 2 bits)
     * - scales[8]: 8 bytes (explicit scales as nibbles, 16 scales total)
     *
     * **Compression:**
     * - Input: 256 floats × 4 bytes = 1024 bytes
     * - Output: 82 bytes per block
     * - Ratio: 12.49× (vs 13.84× for IQ2_XS, 15.52× for IQ2_XXS)
     * - Bits per weight: 2.5625
     *
     * **Key Differences from IQ2_XS:**
     * - Larger grid: 1024 entries (10-bit indices) vs 512 entries (9-bit)
     * - Separate qh array: High 2 bits stored separately for cleaner decoding
     * - More complex bit extraction: Bit shift formula (8-2*l) for extracting high bits
     * - Best quality: Extra 8 bytes per block buys better quantization fidelity
     *
     * **Decode Algorithm:**
     * 1. Extract FP16 scale factor `d`
     * 2. Process 8 sub-blocks of 32 elements each:
     *    a. Extract dual scales from nibbles: scales[ib32] >> 0/4
     *    b. For each of 4 groups of 8 elements:
     *       - Extract 10-bit grid index: qs[l] | (qh[ib32] << (8-2*l) & 0x300)
     *       - Lookup grid: iq2s_grid[index] (1024 entries)
     *       - Apply alternating scale: db[l/2] (groups 0-1: db[0], groups 2-3: db[1])
     *       - Apply signs from ksigns_iq2xs[signs[l]] using kmask_iq2xs[j]
     * 3. Output 256 dequantized float values
     *
     * **Grid Index Bit Extraction:**
     * - qs[l]: Low 8 bits of 10-bit index
     * - qh[ib32]: Contains 2 bits for each of 4 groups (8 bits total)
     * - Bit shift formula: (8-2*l) where l ∈ [0,3]
     *   - l=0: shift 8 → bits 8-9
     *   - l=1: shift 6 → bits 6-7
     *   - l=2: shift 4 → bits 4-5
     *   - l=3: shift 2 → bits 2-3
     * - Mask 0x300: Extracts bits 8-9 (the high 2 bits of 10-bit index)
     * - Result: 10-bit index = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)
     *
     * **Quality Comparison (IQ2 Family):**
     * | Format    | Block Size | Compression | Bits/Weight | Quality | Complexity |
     * |-----------|------------|-------------|-------------|---------|------------|
     * | IQ2_XXS   | 66 bytes   | 15.52×      | 2.0625      | Good    | Simple     |
     * | IQ2_XS    | 74 bytes   | 13.84×      | 2.3125      | Better  | Medium     |
     * | IQ2_S     | 82 bytes   | 12.49×      | 2.5625      | Best    | Complex    |
     *
     * **When to Use:**
     * - Best quality at ~2.5 bits per weight
     * - Acceptable memory overhead (+11% vs IQ2_XS)
     * - Need highest fidelity in IQ2 family
     * - Can afford slightly more complex decode logic
     *
     * **GGML Compatibility:**
     * Compatible with llama.cpp GGML_TYPE_IQ2_S format.
     *
     * @author David Sanftenberg
     */
    class IQ2_STensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief IQ2_S block structure (82 bytes)
         */
        struct IQ2_SBlock
        {
            uint16_t d;          ///< FP16 scale factor (2 bytes)
            uint8_t qs[64];      ///< Grid indices low 8 bits (64 bytes)
            uint8_t qh[8];       ///< Grid indices high 2 bits (8 bytes)
            uint8_t scales[8];   ///< Explicit scales (8 bytes, 16 scales as nibbles)
        };
        static_assert(sizeof(IQ2_SBlock) == 82, "IQ2_SBlock must be 82 bytes");

        static constexpr size_t BLOCK_SIZE_ELEMENTS = 256;
        static constexpr size_t BLOCK_SIZE_BYTES = sizeof(IQ2_SBlock);

        /**
         * @brief Construct from raw quantized data
         *
         * @param shape Tensor shape (total elements must be multiple of 256)
         * @param raw_data Quantized data (must be 82 bytes per 256 elements)
         */
        IQ2_STensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape)
        {
            // Validate shape
            size_t total_elements = 1;
            for (int dim : shape)
            {
                if (dim <= 0)
                {
                    throw std::invalid_argument("IQ2_STensor: shape dimensions must be > 0");
                }
                total_elements *= dim;
            }

            if (total_elements % BLOCK_SIZE_ELEMENTS != 0)
            {
                throw std::invalid_argument("IQ2_STensor: total elements (" +
                                            std::to_string(total_elements) +
                                            ") must be multiple of 256");
            }

            num_blocks_ = total_elements / BLOCK_SIZE_ELEMENTS;
            size_t expected_size = num_blocks_ * BLOCK_SIZE_BYTES;

            if (raw_data.size() != expected_size)
            {
                throw std::invalid_argument("IQ2_STensor: expected " +
                                            std::to_string(expected_size) +
                                            " bytes, got " +
                                            std::to_string(raw_data.size()));
            }

            // Copy raw data
            raw_data_ = raw_data;
        }

        // ===== TensorBase Interface =====

        const std::vector<int> &shape() const override { return shape_; }
        
        int size() const override
        {
            int s = 1;
            for (int dim : shape_)
                s *= dim;
            return s;
        }
        
        int ndim() const override { return static_cast<int>(shape_.size()); }
        
        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        std::string type_name() const override { return "IQ2_STensor"; }

        std::shared_ptr<TensorBase> copy() const override
        {
            return std::make_shared<IQ2_STensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("IQ2_STensor::copy_from not supported - quantization is lossy");
        }

        // ===== QuantizedTensorBase Interface =====

        QuantType quant_type() const override { return QuantType::IQ2_S; }

        float compression_ratio() const override { return 12.49f; }

        void decode_to_fp32(float *dst) const override
        {
            int rows = shape_[0];
            int cols = shape_[1];
#pragma omp parallel for if (rows > 4)
            for (int row = 0; row < rows; ++row)
            {
                decodeRow(row, dst + row * cols);
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

        void decodeRow(size_t row_idx, float *buffer) const override
        {
            // Bounds check
            if (shape_.size() < 2)
            {
                throw std::out_of_range("decodeRow requires at least 2D tensor");
            }
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Row index " + std::to_string(row_idx) +
                                       " out of bounds for tensor with " +
                                       std::to_string(shape_[0]) + " rows");
            }

            int cols = shape_[1];
            size_t global_start = row_idx * cols;
            size_t global_end = global_start + cols;

            size_t start_block = global_start / BLOCK_SIZE_ELEMENTS;
            size_t end_block = (global_end - 1) / BLOCK_SIZE_ELEMENTS;

            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());

            if (start_block == end_block)
            {
                // Single block case
                size_t offset_in_block = global_start % BLOCK_SIZE_ELEMENTS;
                float temp[BLOCK_SIZE_ELEMENTS];
                decodeBlock(&blocks[start_block], temp);
                std::memcpy(buffer, temp + offset_in_block, cols * sizeof(float));
            }
            else
            {
                // Multi-block case
                size_t offset = 0;
                for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
                {
                    float temp[BLOCK_SIZE_ELEMENTS];
                    decodeBlock(&blocks[block_idx], temp);

                    size_t block_start = block_idx * BLOCK_SIZE_ELEMENTS;
                    size_t copy_start = std::max(global_start, block_start) - block_start;
                    size_t copy_end = std::min(global_end, block_start + BLOCK_SIZE_ELEMENTS) - block_start;
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
                throw std::out_of_range("IQ2_STensor::decodeSpan: range exceeds tensor bounds");
            }

            size_t start_block = offset / BLOCK_SIZE_ELEMENTS;
            size_t end_block = (offset + count - 1) / BLOCK_SIZE_ELEMENTS;

            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());

            size_t buffer_offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
            {
                float temp[BLOCK_SIZE_ELEMENTS];
                decodeBlock(&blocks[block_idx], temp);

                size_t block_start = block_idx * BLOCK_SIZE_ELEMENTS;
                size_t copy_start = std::max(offset, block_start) - block_start;
                size_t copy_end = std::min(offset + count, block_start + BLOCK_SIZE_ELEMENTS) - block_start;
                size_t copy_count = copy_end - copy_start;

                std::memcpy(buffer + buffer_offset, temp + copy_start, copy_count * sizeof(float));
                buffer_offset += copy_count;
            }
        }

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
                256, // elements_per_block
                82,  // bytes_per_block
                16,  // scale_count (16 scales: 8 sub-blocks × 2 scales each)
                2,   // bits_per_value (2.5625 bpw, rounded to 2)
                false // is_k_quant
            };
            return desc;
        }

    private:
        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
        size_t num_blocks_;

#if defined(__AVX2__)
        /**
         * @brief AVX2-optimized IQ2_S block decode
         */
        static void decodeBlockAVX2(const IQ2_SBlock *block, float *output)
        {
            const float d = simd::fp16_to_fp32(block->d);
            const uint8_t *qs = block->qs;
            const uint8_t *qh = block->qh;
            const uint8_t *signs = qs + 32;
            const uint8_t *scales_data = block->scales;

            for (size_t ib32 = 0; ib32 < 8; ++ib32)
            {
                const float db[2] = {
                    d * (0.5f + (scales_data[ib32] & 0xf)) * 0.25f,
                    d * (0.5f + (scales_data[ib32] >> 4)) * 0.25f
                };

                for (size_t l = 0; l < 4; ++l)
                {
                    const uint16_t grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);
                    const float scale = db[l / 2];
                    const uint8_t sign_byte = signs[l];
                    
                    // SIMD: Load 8 uint8 grid values and convert to float
                    __m128i grid_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(grid));
                    __m256i grid_i32 = _mm256_cvtepu8_epi32(grid_u8);
                    __m256 grid_f32 = _mm256_cvtepi32_ps(grid_i32);
                    
                    // Apply scale
                    const __m256 scale_vec = _mm256_set1_ps(scale);
                    __m256 result = _mm256_mul_ps(scale_vec, grid_f32);
                    
                    // Apply signs (direct byte masking, not lookup)
                    alignas(32) float result_arr[8];
                    _mm256_store_ps(result_arr, result);
                    for (size_t j = 0; j < 8; ++j) {
                        if (sign_byte & kmask_iq2xs[j]) result_arr[j] = -result_arr[j];
                    }
                    _mm256_storeu_ps(output, _mm256_load_ps(result_arr));
                    output += 8;
                }
                qs += 4;
                signs += 4;
            }
        }
#endif

        /**
         * @brief Decode a single IQ2_S block to FP32
         *
         * Dispatches to AVX2 SIMD if available, otherwise scalar fallback.
         * Implements the IQ2_S decode algorithm with 10-bit grid indexing.
         *
         * @param block Pointer to IQ2_SBlock (82 bytes)
         * @param output Pointer to output float array (256 elements)
         */
        static void decodeBlock(const IQ2_SBlock *block, float *output)
        {
#if defined(__AVX2__)
            decodeBlockAVX2(block, output);
#else
            // Scalar fallback
            const float d = simd::fp16_to_fp32(block->d);
            const uint8_t *qs = block->qs;
            const uint8_t *qh = block->qh;
            const uint8_t *signs = qs + 32;
            const uint8_t *scales_data = block->scales;

            for (size_t ib32 = 0; ib32 < 8; ++ib32)
            {
                const float db[2] = {
                    d * (0.5f + (scales_data[ib32] & 0xf)) * 0.25f,
                    d * (0.5f + (scales_data[ib32] >> 4)) * 0.25f
                };

                for (size_t l = 0; l < 4; ++l)
                {
                    const uint16_t grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                    const uint8_t *grid = reinterpret_cast<const uint8_t *>(&iq2s_grid[grid_idx]);
                    const float scale = db[l / 2];
                    const uint8_t sign_byte = signs[l];
                    
                    for (size_t j = 0; j < 8; ++j)
                    {
                        const float sign = (sign_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
                        output[j] = scale * static_cast<float>(grid[j]) * sign;
                    }
                    output += 8;
                }
                qs += 4;
                signs += 4;
            }
#endif
        }

        /**
         * @brief Decode a single IQ2_S block to BF16
         *
         * @param block Pointer to IQ2_SBlock (82 bytes)
         * @param output Pointer to output bfloat16 array (256 elements)
         */
        static void decodeBlockToBF16(const IQ2_SBlock *block, bfloat16 *output)
        {
            // Decode to FP32 first (BF16 decode not performance-critical)
            float temp[BLOCK_SIZE_ELEMENTS];
            decodeBlock(block, temp);

            // Convert FP32 → BF16
            for (size_t i = 0; i < BLOCK_SIZE_ELEMENTS; ++i)
            {
                output[i] = bfloat16::from_float(temp[i]);
            }
        }

        /**
         * @brief Decode a row to BF16
         *
         * @param row_idx Row index
         * @param buffer Output BF16 buffer
         */
        void decodeRowToBF16(size_t row_idx, bfloat16 *buffer) const
        {
            int cols = shape_[1];
            size_t global_start = row_idx * cols;
            size_t global_end = global_start + cols;

            size_t start_block = global_start / BLOCK_SIZE_ELEMENTS;
            size_t end_block = (global_end - 1) / BLOCK_SIZE_ELEMENTS;

            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());

            if (start_block == end_block)
            {
                // Single block case
                size_t offset_in_block = global_start % BLOCK_SIZE_ELEMENTS;
                bfloat16 temp[BLOCK_SIZE_ELEMENTS];
                decodeBlockToBF16(&blocks[start_block], temp);
                std::memcpy(buffer, temp + offset_in_block, cols * sizeof(bfloat16));
            }
            else
            {
                // Multi-block case
                size_t offset = 0;
                for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
                {
                    bfloat16 temp[BLOCK_SIZE_ELEMENTS];
                    decodeBlockToBF16(&blocks[block_idx], temp);

                    size_t block_start = block_idx * BLOCK_SIZE_ELEMENTS;
                    size_t copy_start = std::max(global_start, block_start) - block_start;
                    size_t copy_end = std::min(global_end, block_start + BLOCK_SIZE_ELEMENTS) - block_start;
                    size_t copy_count = copy_end - copy_start;

                    std::memcpy(buffer + offset, temp + copy_start, copy_count * sizeof(bfloat16));
                    offset += copy_count;
                }
            }
        }
    };

} // namespace llaminar
