#pragma once

#include "TensorFactory.h" // For QuantBlockDescriptor
#include "QuantizedTensorBase.h"
#include "../utils/BFloat16.h"
#include "../utils/SIMDHelpers.h"
#include <cstring>
#include <algorithm>

#ifdef __linux__
#include <execinfo.h> // For backtrace
#endif

namespace llaminar
{
    /**
     * @brief Q6_K quantized tensor (6-bit K-quant)
     *
     * Block format (256 elements per super-block):
     *   - 128 × uint8_t: lower 4 bits of quantized values (ql)
     *   - 64 × uint8_t: upper 2 bits of quantized values (qh)
     *   - 16 × int8_t: quantized scales (one per 16 elements)
     *   - 1 × FP16: super-block scale (d)
     *   - Total: 210 bytes per block
     *
     * Decoding formula:
     *   6bit_value = (ql[i] & 0xF) | ((qh[i/4] >> (2*(i%4))) & 0x3) << 4
     *   value[i] = d * scales[i/16] * (6bit_value - 32)
     *
     * Compression: ~5.33× (256 * 4 bytes FP32 = 1024 bytes → 210 bytes)
     *
     * @author David Sanftenberg
     */
    class Q6_KTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q6_K tensor from shape and raw Q6_K blocks
         */
        Q6_KTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q6_KTensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q6_KBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q6_K raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q6_K; }
        float compression_ratio() const override { return 5.33f; } // 1024 bytes FP32 → 210 bytes Q6_K

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q6_KTensor"; }
        bool is_distributed() const override { return false; }

        // Quantized tensors don't support direct data() access
        float *data() override
        {
            // Print stack trace to help debug where data() is being called from
            std::cerr << "[Q6_KTensor::data()] ERROR: data() called on Q6_KTensor!" << std::endl;
            std::cerr << "[Q6_KTensor::data()] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;

#ifdef __linux__
            void *callstack[128];
            int frames = backtrace(callstack, 128);
            char **strs = backtrace_symbols(callstack, frames);
            std::cerr << "[Q6_KTensor::data()] Call stack:" << std::endl;
            for (int i = 0; i < std::min(10, frames); i++)
            {
                std::cerr << "  " << strs[i] << std::endl;
            }
            free(strs);
#endif

            throw std::runtime_error("Q6_KTensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            throw std::runtime_error("Q6_KTensor: data() not supported - use decodeRow() instead");
        }

        // ===== TensorBase Required Methods =====

        void decode_to_fp32(float *dst) const override
        {
            // Decode entire tensor to FP32
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
            // Decode entire tensor to BF16
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
            return std::make_shared<Q6_KTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q6_KTensor::copy_from not supported - quantization is lossy");
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
                throw std::out_of_range("Q6_KTensor: Row index out of bounds in decodeRow_avx512");
            }

            size_t element_offset = row_idx * cols;
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());

            int col = 0;
            // Process 16 elements at a time (256 elements per block, so we can do 16 iterations)
            for (; col + 16 <= cols; col += 16)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                // Check if we cross block or scale segment boundary
                if (in_block_idx + 16 > BLOCK_SIZE || (in_block_idx / 16) != ((in_block_idx + 15) / 16))
                {
                    break;
                }

                const Q6_KBlock *block = &blocks[block_idx];
                float super_scale = simd::fp16_to_fp32(block->d);
                int scale_idx = in_block_idx / 16;
                float scale = super_scale * block->scales[scale_idx];

                // Extract 16 6-bit values using SIMDHelpers
                int8_t q6_values[16];
                simd::extract_q6k_values(block->ql, block->qh, in_block_idx, 16, 32, q6_values);

                // Convert 16 int8 → float32 with scaling
                simd::convert_i8_to_f32_scaled_avx512(q6_values, scale, buffer + col);
            }

            // Scalar fallback for remainder
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
                throw std::out_of_range("Q6_KTensor: Row index out of bounds in decodeRow_avx2");
            }

            size_t element_offset = row_idx * cols;
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());

            int col = 0;
            // Process 8 elements at a time
            for (; col + 8 <= cols; col += 8)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                // Check if we cross block or scale segment boundary
                if (in_block_idx + 8 > BLOCK_SIZE || (in_block_idx / 16) != ((in_block_idx + 7) / 16))
                {
                    break;
                }

                const Q6_KBlock *block = &blocks[block_idx];
                float super_scale = simd::fp16_to_fp32(block->d);
                int scale_idx = in_block_idx / 16;
                float scale = super_scale * block->scales[scale_idx];

                // Extract 8 6-bit values using SIMDHelpers
                int8_t q6_values[8];
                simd::extract_q6k_values(block->ql, block->qh, in_block_idx, 8, 32, q6_values);

                // Convert 8 int8 → float32 with scaling
                simd::convert_i8_to_f32_scaled_avx2(q6_values, scale, buffer + col);
            }

            // Scalar fallback for remainder
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
                throw std::out_of_range("Q6_KTensor: Row index out of bounds in decodeRow_scalar");
            }

            size_t element_offset = row_idx * shape_[1] + col_offset;
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());

            #pragma omp simd
            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q6_KBlock *block = &blocks[block_idx];

                // Extract 6-bit value using SIMDHelpers
                int8_t q6_value = simd::extract_q6k_value(block->ql, block->qh, in_block_idx, 32);

                int scale_idx = in_block_idx / 16;
                int8_t scale_quantized = block->scales[scale_idx];

                float super_scale = simd::fp16_to_fp32(block->d);
                buffer[col] = super_scale * scale_quantized * q6_value;
            }
        }

        /**
         * @brief Decode a single row to FP32 (runtime dispatch)
         *
         * Decodes 6-bit K-quant values to FP32 on-the-fly.
         * Each value is constructed from lower 4 bits (ql) + upper 2 bits (qh).
         *
         * @param row_idx Row index (0 to shape()[0]-1)
         * @param buffer Pre-allocated FP32 buffer with shape()[1] elements
         */
        void decodeRow(size_t row_idx, float *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q6_KTensor: Row index out of bounds");
            }

            int cols = shape_[1];

#ifdef __AVX512F__
            if (simd::cpu_supports_avx512() && cols >= 16)
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
         *
         * Optimized: decode directly to BF16 without FP32 intermediate.
         *
         * @param row_idx Row index
         * @param buffer Pre-allocated BF16 buffer
         */
        void decodeRowToBF16(size_t row_idx, void *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q6_KTensor: Row index out of bounds");
            }

            int cols = shape_[1];
            size_t element_offset = row_idx * cols;

            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());
            bfloat16 *bf16_buffer = static_cast<bfloat16 *>(buffer);

            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q6_KBlock *block = &blocks[block_idx];

                // Extract 6-bit value using SIMDHelpers
                int8_t q6_value = simd::extract_q6k_value(block->ql, block->qh, in_block_idx, 32);

                // Get scale
                int scale_idx = in_block_idx / 16;
                int8_t scale_quantized = block->scales[scale_idx];

                // Dequantize
                float super_scale = simd::fp16_to_fp32(block->d);
                float fp32_val = super_scale * scale_quantized * q6_value;
                bf16_buffer[col] = bfloat16::from_float(fp32_val);
            }
        }

        /**
         * @brief Decode arbitrary span of elements
         *
         * @param offset Starting element index
         * @param count Number of elements
         * @param buffer Pre-allocated FP32 buffer
         */
        void decodeSpan(size_t offset, size_t count, float *buffer) const override
        {
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(raw_data_.data());

            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q6_KBlock *block = &blocks[block_idx];

                // Extract 6-bit value using SIMDHelpers
                int8_t q6_value = simd::extract_q6k_value(block->ql, block->qh, in_block_idx, 32);

                // Get scale
                int scale_idx = in_block_idx / 16;
                int8_t scale_quantized = block->scales[scale_idx];

                // Dequantize
                float super_scale = simd::fp16_to_fp32(block->d);
                buffer[i] = super_scale * scale_quantized * q6_value;
            }
        }

        // ===== Raw Block Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = sizeof(Q6_KBlock),
                .scale_count = 16,
                .bits_per_value = 6,
                .is_k_quant = true};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 256; // Q6_K uses 256 elements per super-block

        /**
         * @brief Q6_K block structure (matches llama.cpp/ggml)
         *
         * Layout:
         *   - ql: 128 × uint8_t (lower 4 bits, storing 2 values per byte)
         *   - qh: 64 × uint8_t (upper 2 bits, storing 4 values per byte)
         *   - scales: 16 × int8_t (quantized scales, one per 16 elements)
         *   - d: FP16 super-block scale (2 bytes)
         *
         * Total: 210 bytes per block (128 + 64 + 16 + 2)
         */
        struct Q6_KBlock
        {
            uint8_t ql[128];  // Lower 4 bits of quantized values (256 values / 2 per byte = 128 bytes)
            uint8_t qh[64];   // Upper 2 bits of quantized values (256 values / 4 per byte = 64 bytes)
            int8_t scales[16]; // Quantized scales (16 scales for 256 elements)
            uint16_t d;        // FP16 super-block scale
        };

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
