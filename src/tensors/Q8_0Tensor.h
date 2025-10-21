#pragma once

#include "TensorFactory.h" // For QuantBlockDescriptor
#include "QuantizedTensorBase.h"
#include "../utils/BFloat16.h"
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
            for (int row = 0; row < rows; ++row)
            {
                decodeRowToBF16(row, static_cast<uint8_t *>(dst) + row * cols * sizeof(bfloat16));
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

        void decodeRow(size_t row_idx, float *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range(
                    "Q8_0Tensor::decodeRow: row_idx " + std::to_string(row_idx) +
                    " out of bounds (rows=" + std::to_string(shape_[0]) + ")");
            }

            int cols = shape_[1];
            size_t element_offset = row_idx * cols;

            // Decode blocks covering this row
            for (int col = 0; col < cols; col++)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q8_0Block *block = get_block(block_idx);

                // Q8_0 decode: scale * quantized_value
                float scale = fp16_to_fp32(block->scale_bits);
                buffer[col] = scale * static_cast<float>(block->values[in_block_idx]);
            }
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
