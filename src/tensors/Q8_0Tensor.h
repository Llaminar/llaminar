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
     * @brief Q8_0 quantized tensor (8-bit uniform quantization)
     *
     * Block format (32 elements per block):
     *   - 1 × FP16 scale factor (2 bytes)
     *   - 32 × int8 quantized values (32 bytes)
     *   - Total: 34 bytes per block
     *
     * Decoding formula: value[i] = scale * quantized[i]
     *
     * Compression: 4× (32-bit FP32 → 8-bit + scale overhead)
     *
     * @author David Sanftenberg
     */
    class Q8_0Tensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q8_0 tensor from shape and raw Q8_0 blocks
         */
        Q8_0Tensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q8_0Tensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q8_0Block);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q8_0 raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q8_0; }
        float compression_ratio() const override { return 4.0f; } // 32-bit → 8-bit

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q8_0Tensor"; }
        bool is_distributed() const override { return false; }

        // Quantized tensors don't support direct data() access
        float *data() override
        {
            // Print stack trace to help debug where data() is being called from
            std::cerr << "[Q8_0Tensor::data()] ERROR: data() called on Q8_0Tensor!" << std::endl;
            std::cerr << "[Q8_0Tensor::data()] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;

// Print basic call stack info (requires execinfo.h on Linux)
#ifdef __linux__
            void *callstack[128];
            int frames = backtrace(callstack, 128);
            char **strs = backtrace_symbols(callstack, frames);
            std::cerr << "[Q8_0Tensor::data()] Call stack:" << std::endl;
            for (int i = 0; i < std::min(10, frames); i++)
            {
                std::cerr << "  " << strs[i] << std::endl;
            }
            free(strs);
#endif

            throw std::runtime_error("Q8_0Tensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            std::cerr << "[Q8_0Tensor::data() const] ERROR: data() called on Q8_0Tensor!" << std::endl;
            std::cerr << "[Q8_0Tensor::data() const] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;
            throw std::runtime_error("Q8_0Tensor: data() not supported - use decodeRow() instead");
        }

        // Quantized tensors are immutable
        void zero() override
        {
            throw std::runtime_error("Q8_0Tensor: zero() not supported - quantized tensors are immutable");
        }

        void fill(float) override
        {
            throw std::runtime_error("Q8_0Tensor: fill() not supported - quantized tensors are immutable");
        }

        void decode_to_fp32(float *dst) const override
        {
            // Decode entire tensor
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
            return std::make_shared<Q8_0Tensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q8_0Tensor::copy_from not supported - quantization is lossy");
        }

        // ===== Streaming Decode =====

#ifdef __AVX512F__
        // AVX-512 optimized decode (16 elements at a time)
        void decodeRow_avx512(size_t row_idx, float *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q8_0Tensor: Row index out of bounds in decodeRow_avx512");
            }
            
            int cols = shape_[1];
            size_t element_offset = row_idx * cols;
            int col = 0;

            // Process complete blocks (32 elements = 2 AVX-512 vectors)
            for (; col + 31 < cols; col += 32)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;

                // Check if entire 32-element span is within one block
                if ((elem_idx % BLOCK_SIZE) == 0)
                {
                    const Q8_0Block *block = get_block(block_idx);
                    float scale = simd::fp16_to_fp32(block->scale_bits);

                    // Convert first 16 elements using SIMD helper
                    simd::convert_i8_to_f32_scaled_avx512(block->values, scale, buffer + col);
                    
                    // Convert second 16 elements
                    simd::convert_i8_to_f32_scaled_avx512(block->values + 16, scale, buffer + col + 16);
                }
                else
                {
                    // Span crosses block boundary - use scalar fallback
                    for (int i = 0; i < 32; i++)
                    {
                        size_t idx = elem_idx + i;
                        size_t blk_idx = idx / BLOCK_SIZE;
                        size_t in_blk_idx = idx % BLOCK_SIZE;
                        const Q8_0Block *block = get_block(blk_idx);
                        float scale = fp16_to_fp32(block->scale_bits);
                        buffer[col + i] = scale * static_cast<float>(block->values[in_blk_idx]);
                    }
                }
            }

            // Handle remaining elements with scalar code
            for (; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;
                const Q8_0Block *block = get_block(block_idx);
                float scale = fp16_to_fp32(block->scale_bits);
                buffer[col] = scale * static_cast<float>(block->values[in_block_idx]);
            }
        }
#endif

#ifdef __AVX2__
        // AVX2 optimized decode (8 elements at a time)
        void decodeRow_avx2(size_t row_idx, float *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q8_0Tensor: Row index out of bounds in decodeRow_avx2");
            }
            
            int cols = shape_[1];
            size_t element_offset = row_idx * cols;
            int col = 0;

            // Process 8 elements at a time
            for (; col + 7 < cols; col += 8)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                // Check if 8 elements are within same block
                if ((elem_idx + 7) / BLOCK_SIZE == block_idx)
                {
                    const Q8_0Block *block = get_block(block_idx);
                    float scale = simd::fp16_to_fp32(block->scale_bits);

                    // Convert 8 elements using SIMD helper
                    simd::convert_i8_to_f32_scaled_avx2(block->values + in_block_idx, scale, buffer + col);
                }
                else
                {
                    // Span crosses block boundary - use scalar fallback
                    for (int i = 0; i < 8; i++)
                    {
                        size_t idx = elem_idx + i;
                        size_t blk_idx = idx / BLOCK_SIZE;
                        size_t in_blk_idx = idx % BLOCK_SIZE;
                        const Q8_0Block *block = get_block(blk_idx);
                        float scale = fp16_to_fp32(block->scale_bits);
                        buffer[col + i] = scale * static_cast<float>(block->values[in_blk_idx]);
                    }
                }
            }

            // Handle remaining elements
            for (; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;
                const Q8_0Block *block = get_block(block_idx);
                float scale = simd::fp16_to_fp32(block->scale_bits);
                buffer[col] = scale * static_cast<float>(block->values[in_block_idx]);
            }
        }
#endif

        // Scalar fallback (always available)
        void decodeRow_scalar(size_t row_idx, float *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q8_0Tensor: Row index out of bounds in decodeRow_scalar");
            }
            
            int cols = shape_[1];
            size_t element_offset = row_idx * cols;

#pragma omp simd
            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q8_0Block *block = get_block(block_idx);
                float scale = fp16_to_fp32(block->scale_bits);
                buffer[col] = scale * static_cast<float>(block->values[in_block_idx]);
            }
        }

        void decodeRow(size_t row, float *output) const override
        {
            size_t cols = shape_[1];
#ifdef __AVX512F__
            if (simd::cpu_supports_avx512() && cols >= 32)
            {
                decodeRow_avx512(row, output);
                return;
            }
#endif
#ifdef __AVX2__
            if (simd::cpu_supports_avx2() && cols >= 8)
            {
                decodeRow_avx2(row, output);
                return;
            }
#endif
            decodeRow_scalar(row, output);
        }

        void decodeRowToBF16(size_t row_idx, void *buffer) const override
        {
            // Optimized: decode directly to BF16 without FP32 intermediate
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q8_0Tensor::decodeRowToBF16: row_idx out of bounds");
            }

            bfloat16 *bf16_buffer = static_cast<bfloat16 *>(buffer);
            int cols = shape_[1];
            size_t element_offset = row_idx * cols;

            #pragma omp simd
            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q8_0Block *block = get_block(block_idx);
                float scale = fp16_to_fp32(block->scale_bits);
                float fp32_val = scale * static_cast<float>(block->values[in_block_idx]);
                bf16_buffer[col] = bfloat16::from_float(fp32_val);
            }
        }

        void decodeSpan(size_t offset, size_t count, float *buffer) const override
        {
            if (offset + count > static_cast<size_t>(size()))
            {
                throw std::out_of_range("Q8_0Tensor::decodeSpan: span out of bounds");
            }

            #pragma omp simd
            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q8_0Block *block = get_block(block_idx);
                float scale = fp16_to_fp32(block->scale_bits);
                buffer[i] = scale * static_cast<float>(block->values[in_block_idx]);
            }
        }

        // ===== Raw Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = static_cast<int>(sizeof(Q8_0Block)),
                .scale_count = 1,
                .bits_per_value = 8,
                .is_k_quant = false};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 32; // Q8_0 has 32 elements per block

        // Runtime CPU feature detection (cached)
        static bool cpu_supports_avx512()
        {
            static int result = -1;
            if (result == -1)
            {
#ifdef __AVX512F__
                result = __builtin_cpu_supports("avx512f") &&
                                 __builtin_cpu_supports("avx512bw")
                             ? 1
                             : 0;
#else
                result = 0;
#endif
            }
            return result == 1;
        }

        static bool cpu_supports_avx2()
        {
            static int result = -1;
            if (result == -1)
            {
#ifdef __AVX2__
                result = __builtin_cpu_supports("avx2") ? 1 : 0;
#else
                result = 0;
#endif
            }
            return result == 1;
        }

        /**
         * @brief Q8_0 block structure (34 bytes)
         *
         * Layout:
         *   - scale: FP16 scale factor (2 bytes)
         *   - values: 32 × int8 quantized values (32 bytes)
         */
        struct Q8_0Block
        {
            uint16_t scale_bits;       // FP16 scale (stored as uint16)
            int8_t values[BLOCK_SIZE]; // 32 quantized int8 values

            // Helper to get FP32 scale
            float get_scale() const
            {
                return fp16_to_fp32(scale_bits);
            }
        } __attribute__((packed));

        // Ensure struct packing matches GGML format
        static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

        /**
         * @brief Get block pointer with bounds checking
         */
        const Q8_0Block *get_block(size_t block_idx) const
        {
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(raw_data_.data());
            return &blocks[block_idx];
        }

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
