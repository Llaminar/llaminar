#pragma once

#include "TensorFactory.h"
#include "QuantizedTensorBase.h"
#include "../utils/BFloat16.h"
#include "../utils/SIMDHelpers.h"
#include <cstring>
#include <algorithm>

namespace llaminar
{
    /**
     * @brief Q4_1 quantized tensor (4-bit with scale + min)
     *
     * Block format (32 elements per block):
     *   - scale: FP16 (2 bytes) - delta
     *   - min: FP16 (2 bytes) - minimum value
     *   - qs: 16 × uint8_t (16 bytes, each byte stores 2 4-bit values)
     *   - Total: 20 bytes per block
     *
     * Decoding formula:
     *   value[i] = scale * quant[i] + min
     *
     * Compression: ~6.4× (32 * 4 bytes FP32 = 128 bytes → 20 bytes)
     *
     * @author David Sanftenberg
     */
    class Q4_1Tensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q4_1 tensor from shape and raw Q4_1 blocks
         */
        Q4_1Tensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q4_1Tensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q4_1Block);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q4_1 raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q4_1; }
        float compression_ratio() const override { return 6.4f; } // 128 bytes → 20 bytes

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q4_1Tensor"; }
        bool is_distributed() const override { return false; }

        float *data() override
        {
            throw std::runtime_error("Q4_1Tensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            throw std::runtime_error("Q4_1Tensor: data() not supported - use decodeRow() instead");
        }

        void decode_to_fp32(float *dst) const override
        {
            int rows = shape_[0];
            int cols = shape_[1];
            #pragma omp parallel for if(rows > 4)
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
            #pragma omp parallel for if(rows > 4)
            for (int row = 0; row < rows; ++row)
            {
                decodeRowToBF16(row, bf16_dst + row * cols);
            }
        }

        std::shared_ptr<TensorBase> copy() const override
        {
            return std::make_shared<Q4_1Tensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q4_1Tensor::copy_from not supported - quantization is lossy");
        }

        // ===== Streaming Decode API =====

        /**
         * @brief Decode a single row to FP32 (AVX-512 optimized)
         */
#ifdef __AVX512F__
        void decodeRow_avx512(size_t row_idx, float *buffer, int cols) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q4_1Tensor: Row index out of bounds in decodeRow_avx512");
            }

            size_t element_offset = row_idx * cols;
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());

            int col = 0;
            // Process 32 elements (full block) at a time when aligned
            for (; col + 32 <= cols; col += 32)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                if (in_block_idx != 0)
                {
                    break;
                }

                const Q4_1Block *block = &blocks[block_idx];
                float scale = simd::fp16_to_fp32(block->scale);
                float min = simd::fp16_to_fp32(block->min);

                // Unpack nibbles and convert first 16 elements: (nibble * scale) + min
                __m128i interleaved_high = simd::unpack_nibbles_convert_f32_first16_avx512(
                    block->qs, scale, -min / scale, buffer + col);

                // Convert second 16 elements
                simd::convert_unpacked_nibbles_f32_avx512(
                    interleaved_high, scale, -min / scale, buffer + col + 16);
            }

            decodeRow_scalar(row_idx, buffer + col, cols - col, col);
        }
#endif

        /**
         * @brief Decode a single row to FP32 (AVX2 optimized)
         */
#ifdef __AVX2__
        void decodeRow_avx2(size_t row_idx, float *buffer, int cols) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q4_1Tensor: Row index out of bounds in decodeRow_avx2");
            }

            size_t element_offset = row_idx * cols;
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());

            int col = 0;
            for (; col + 8 <= cols; col += 8)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                if (in_block_idx + 8 > BLOCK_SIZE)
                {
                    break;
                }

                const Q4_1Block *block = &blocks[block_idx];
                float scale = simd::fp16_to_fp32(block->scale);
                float min = simd::fp16_to_fp32(block->min);

                uint32_t nibble_bytes;
                std::memcpy(&nibble_bytes, block->qs + in_block_idx / 2, 4);

                int8_t values[8];
                simd::extract_nibbles_scalar(nibble_bytes, values, 0);

                // Convert with scale and add min: scale * quant + min
                simd::convert_i8_to_f32_scaled_biased_avx2(values, scale, min, buffer + col);
            }

            decodeRow_scalar(row_idx, buffer + col, cols - col, col);
        }
#endif

        /**
         * @brief Decode a single row to FP32 (scalar fallback)
         */
        void decodeRow_scalar(size_t row_idx, float *buffer, int cols, int col_offset = 0) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q4_1Tensor: Row index out of bounds in decodeRow_scalar");
            }

            size_t element_offset = row_idx * shape_[1] + col_offset;
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());

            #pragma omp simd
            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q4_1Block *block = &blocks[block_idx];

                uint8_t byte_val = block->qs[in_block_idx / 2];
                int8_t quant;

                if (in_block_idx % 2 == 0)
                {
                    quant = (byte_val & 0x0F);
                }
                else
                {
                    quant = (byte_val >> 4);
                }

                float scale = simd::fp16_to_fp32(block->scale);
                float min = simd::fp16_to_fp32(block->min);
                buffer[col] = scale * quant + min;
            }
        }

        /**
         * @brief Decode a single row to FP32 (runtime dispatch)
         */
        void decodeRow(size_t row_idx, float *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q4_1Tensor: Row index out of bounds");
            }

            int cols = shape_[1];

#ifdef __AVX512F__
            if (simd::cpu_supports_avx512() && cols >= 32)
            {
                decodeRow_avx512(row_idx, buffer, cols);
                return;
            }
#endif
#ifdef __AVX2__
            if (simd::cpu_supports_avx2() && cols >= 8)
            {
                decodeRow_avx2(row_idx, buffer, cols);
                return;
            }
#endif
            decodeRow_scalar(row_idx, buffer, cols);
        }

        /**
         * @brief Decode a single row to BF16
         */
        void decodeRowToBF16(size_t row_idx, void *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q4_1Tensor: Row index out of bounds");
            }

            int cols = shape_[1];
            size_t element_offset = row_idx * cols;

            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());
            bfloat16 *bf16_buffer = static_cast<bfloat16 *>(buffer);

            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q4_1Block *block = &blocks[block_idx];

                uint8_t byte_val = block->qs[in_block_idx / 2];
                int8_t quant;

                if (in_block_idx % 2 == 0)
                {
                    quant = (byte_val & 0x0F);
                }
                else
                {
                    quant = (byte_val >> 4);
                }

                float scale = simd::fp16_to_fp32(block->scale);
                float min = simd::fp16_to_fp32(block->min);
                float fp32_val = scale * quant + min;
                bf16_buffer[col] = bfloat16::from_float(fp32_val);
            }
        }

        /**
         * @brief Decode arbitrary span of elements
         */
        void decodeSpan(size_t offset, size_t count, float *buffer) const override
        {
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(raw_data_.data());

            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q4_1Block *block = &blocks[block_idx];

                uint8_t byte_val = block->qs[in_block_idx / 2];
                int8_t quant;

                if (in_block_idx % 2 == 0)
                {
                    quant = (byte_val & 0x0F);
                }
                else
                {
                    quant = (byte_val >> 4);
                }

                float scale = simd::fp16_to_fp32(block->scale);
                float min = simd::fp16_to_fp32(block->min);
                buffer[i] = scale * quant + min;
            }
        }

        // ===== Raw Block Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = sizeof(Q4_1Block),
                .scale_count = 2, // scale + min
                .bits_per_value = 4,
                .is_k_quant = false};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 32;

        /**
         * @brief Q4_1 block structure (matches llama.cpp/ggml)
         *
         * Layout:
         *   - scale: FP16 (2 bytes) - delta
         *   - min: FP16 (2 bytes) - minimum value
         *   - qs: 16 × uint8_t (16 bytes, each byte stores 2 4-bit values)
         *
         * Total: 20 bytes per block
         */
        struct Q4_1Block
        {
            uint16_t scale; // FP16 scale factor (delta)
            uint16_t min;   // FP16 min value
            uint8_t qs[16]; // 16 bytes storing 32 4-bit values (nibbles)
        };

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
