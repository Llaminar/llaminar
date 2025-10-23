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
     * @brief Q5_K quantized tensor (5-bit K-quant with hierarchical scales and mins)
     *
     * Block format (256 elements per super-block):
     *   - 2 bytes: d (FP16) - super-block scale for quantized scales
     *   - 2 bytes: dmin (FP16) - super-block scale for quantized mins
     *   - 12 bytes: scales - 6-bit quantized scales and mins (hierarchical packing)
     *   - 128 bytes: qs - lower 4 bits of quantized values (2 values per byte)
     *   - 32 bytes: qh - upper 1 bit of quantized values (8 values per byte)
     *   - Total: 176 bytes per block
     *
     * Decoding formula (for each group of 32 elements):
     *   scale = d * sc (where sc is extracted 6-bit scale)
     *   min = dmin * m (where m is extracted 6-bit min)
     *   5bit_value = (qs[i] & 0xF) | ((qh[i/8] >> (i%8)) & 0x1) << 4
     *   value[i] = scale * 5bit_value - min
     *
     * Compression: ~5.82× (256 * 4 bytes FP32 = 1024 bytes → 176 bytes)
     *
     * @author David Sanftenberg
     */
    class Q5_KTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q5_K tensor from shape and raw Q5_K blocks
         */
        Q5_KTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q5_KTensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q5_KBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q5_K raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q5_K; }
        float compression_ratio() const override { return 5.82f; } // 1024 bytes FP32 → 176 bytes Q5_K

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q5_KTensor"; }
        bool is_distributed() const override { return false; }

        // Quantized tensors don't support direct data() access
        float *data() override
        {
            // Print stack trace to help debug where data() is being called from
            std::cerr << "[Q5_KTensor::data()] ERROR: data() called on Q5_KTensor!" << std::endl;
            std::cerr << "[Q5_KTensor::data()] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;

#ifdef __linux__
            void *callstack[128];
            int frames = backtrace(callstack, 128);
            char **strs = backtrace_symbols(callstack, frames);
            std::cerr << "[Q5_KTensor::data()] Call stack:" << std::endl;
            for (int i = 0; i < std::min(10, frames); i++)
            {
                std::cerr << "  " << strs[i] << std::endl;
            }
            free(strs);
#endif

            throw std::runtime_error("Q5_KTensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            throw std::runtime_error("Q5_KTensor: data() not supported - use decodeRow() instead");
        }

        // ===== Decode API =====

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
            return std::make_shared<Q5_KTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q5_KTensor::copy_from not supported - quantization is lossy");
        }

        // ===== Streaming Decode API =====

        /**
         * @brief Decode a single row to FP32 (scalar implementation)
         * 
         * Processes 256 elements per block in groups of 32 elements.
         * Each group has its own scale and min extracted hierarchically.
         */
        void decodeRow(size_t row_idx, float *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q5_KTensor: Row index out of bounds in decodeRow");
            }

            int cols = shape_[1];
            size_t element_offset = row_idx * cols;
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());

            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q5_KBlock *block = &blocks[block_idx];

                // Extract FP16 super-block scales
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                // Determine which group of 32 elements (0-7)
                int group_idx = in_block_idx / 32;
                
                // Extract 6-bit scale and min for this group
                uint8_t sc, m;
                simd::extract_scale_min_k4(group_idx, block->scales, &sc, &m);

                // Compute local scale and min
                float scale = d * sc;
                float min = dmin * m;

                // Extract 5-bit value (4 low bits + 1 high bit)
                uint8_t q5_value = simd::extract_q5k_value(block->qs, block->qh, in_block_idx);

                // Dequantize: scale * value - min
                buffer[col] = scale * q5_value - min;
            }
        }

        /**
         * @brief Decode a single row to BF16
         */
        void decodeRowToBF16(size_t row_idx, bfloat16 *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q5_KTensor: Row index out of bounds in decodeRowToBF16");
            }

            int cols = shape_[1];
            size_t element_offset = row_idx * cols;
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());

            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = element_offset + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q5_KBlock *block = &blocks[block_idx];

                // Extract FP16 super-block scales
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                // Determine which group of 32 elements (0-7)
                int group_idx = in_block_idx / 32;
                
                // Extract 6-bit scale and min for this group
                uint8_t sc, m;
                simd::extract_scale_min_k4(group_idx, block->scales, &sc, &m);

                // Compute local scale and min
                float scale = d * sc;
                float min = dmin * m;

                // Extract 5-bit value (4 low bits + 1 high bit)
                uint8_t q5_value = simd::extract_q5k_value(block->qs, block->qh, in_block_idx);

                // Dequantize: scale * value - min
                float fp32_val = scale * q5_value - min;
                buffer[col] = bfloat16::from_float(fp32_val);
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
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(raw_data_.data());

            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q5_KBlock *block = &blocks[block_idx];

                // Extract FP16 super-block scales
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                // Determine which group of 32 elements (0-7)
                int group_idx = in_block_idx / 32;
                
                // Extract 6-bit scale and min for this group
                uint8_t sc, m;
                simd::extract_scale_min_k4(group_idx, block->scales, &sc, &m);

                // Compute local scale and min
                float scale = d * sc;
                float min = dmin * m;

                // Extract 5-bit value (4 low bits + 1 high bit)
                uint8_t q5_value = simd::extract_q5k_value(block->qs, block->qh, in_block_idx);

                // Dequantize: scale * value - min
                buffer[i] = scale * q5_value - min;
            }
        }

        // ===== Raw Block Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = sizeof(Q5_KBlock),
                .scale_count = 8,       // 8 scale/min pairs for 256 elements (32 elements per pair)
                .bits_per_value = 5,
                .is_k_quant = true};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 256; // Q5_K uses 256 elements per super-block

        /**
         * @brief Q5_K block structure (matches llama.cpp/ggml)
         *
         * Layout:
         *   - d: FP16 super-block scale for quantized scales (2 bytes)
         *   - dmin: FP16 super-block scale for quantized mins (2 bytes)
         *   - scales: 12 bytes of 6-bit quantized scales and mins (hierarchical packing)
         *   - qs: 128 bytes of lower 4 bits (2 values per byte)
         *   - qh: 32 bytes of upper 1 bit (8 values per byte)
         *
         * Total: 176 bytes per block (2 + 2 + 12 + 128 + 32)
         *
         * Hierarchical scale extraction:
         *   - 8 scale/min pairs for 256 elements (32 elements per pair)
         *   - First 4 pairs: scale and min in lower 6 bits of separate bytes
         *   - Last 4 pairs: hierarchical extraction using upper 2 bits
         */
        struct Q5_KBlock
        {
            uint16_t d;          // FP16 super-block scale for scales
            uint16_t dmin;       // FP16 super-block scale for mins
            uint8_t scales[12];  // 6-bit quantized scales and mins (hierarchical packing)
            uint8_t qs[128];     // Lower 4 bits of quantized values (256 values / 2 per byte = 128 bytes)
            uint8_t qh[32];      // Upper 1 bit of quantized values (256 values / 8 per byte = 32 bytes)
        };

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
