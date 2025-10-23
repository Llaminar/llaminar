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
     * @brief Q3_K quantized tensor (3-bit K-quant with hierarchical scales)
     *
     * Block format (256 elements per super-block):
     *   - 32 bytes: hmask - high bit mask (8 values per byte)
     *   - 64 bytes: qs - lower 2 bits of quantized values (4 values per byte)
     *   - 12 bytes: scales - 6-bit quantized scales (hierarchical packing)
     *   - 2 bytes: d (FP16) - super-block scale for quantized scales
     *   - Total: 110 bytes per block
     *
     * Decoding formula (for each group of 16 elements):
     *   scale = d * (sc - 32) (where sc is extracted 6-bit scale)
     *   3bit_value = (qs[i] & 0x3) - (hmask[i] ? 0 : 4)
     *   value[i] = scale * 3bit_value
     *
     * Compression: ~9.31× (256 * 4 bytes FP32 = 1024 bytes → 110 bytes)
     *
     * @author David Sanftenberg
     */
    class Q3_KTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q3_K tensor from shape and raw Q3_K blocks
         */
        Q3_KTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q3_KTensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q3_KBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q3_K raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q3_K; }
        float compression_ratio() const override { return 9.31f; } // 1024 bytes FP32 → 110 bytes Q3_K

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q3_KTensor"; }
        bool is_distributed() const override { return false; }

        // Quantized tensors don't support direct data() access
        float *data() override
        {
            // Print stack trace to help debug where data() is being called from
            std::cerr << "[Q3_KTensor::data()] ERROR: data() called on Q3_KTensor!" << std::endl;
            std::cerr << "[Q3_KTensor::data()] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;

#ifdef __linux__
            void *callstack[128];
            int frames = backtrace(callstack, 128);
            char **strs = backtrace_symbols(callstack, frames);
            std::cerr << "[Q3_KTensor::data()] Call stack:" << std::endl;
            for (int i = 0; i < std::min(10, frames); i++)
            {
                std::cerr << "  " << strs[i] << std::endl;
            }
            free(strs);
#endif

            throw std::runtime_error("Q3_KTensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            throw std::runtime_error("Q3_KTensor: data() not supported - use decodeRow() instead");
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
            return std::make_shared<Q3_KTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q3_KTensor::copy_from not supported - quantization is lossy");
        }

        // ===== Streaming Decode API =====

        /**
         * @brief Decode a single row to FP32 (scalar implementation)
         * 
         * Mimics GGML dequantize_row_q3_K structure exactly.
         * Processes 256 elements in 2 groups of 128, each subdivided into
         * 4 sub-groups of 32 elements (16+16 each).
         */
        void decodeRow(size_t row_idx, float *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q3_KTensor: Row index out of bounds in decodeRow");
            }

            int cols = shape_[1];
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());

            // Process each element individually (mimics GGML structure)
            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = row_idx * cols + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q3_KBlock *block = &blocks[block_idx];
                float d = simd::fp16_to_fp32(block->d);

                // Determine position in GGML iteration structure
                int group_128 = in_block_idx / 128;     // 0 or 1
                int in_group_128 = in_block_idx % 128;
                
                int subgroup_32 = in_group_128 / 32;    // 0-3
                int in_subgroup_32 = in_group_128 % 32;
                
                int part_16 = in_subgroup_32 / 16;      // 0 or 1
                int in_part_16 = in_subgroup_32 % 16;
                
                // Scale index
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                int8_t sc = simd::extract_q3k_scale(block->scales, scale_idx);
                float scale = d * sc;
                
                // Calculate q offset and shift
                // q advances by 32 bytes every 128 elements
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;  // 0, 2, 4, 6
                
                // Position in qs: base + part*16 + offset
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                // Extract 2-bit value from qs
                int8_t q_low = (block->qs[qs_idx] >> shift) & 3;
                
                // Calculate hmask bit position using ORIGINAL element index (in_block_idx)
                // hmask layout: first 32 elements→bit0, next 32→bit1, ..., last 32→bit7
                int hmask_byte = in_block_idx % 32;
                uint8_t hmask_bit = 1 << (in_block_idx / 32);
                bool hmask_set = (block->hmask[hmask_byte] & hmask_bit) != 0;
                
                // Compute final value: (q_low - (hmask ? 0 : 4))
                int8_t q_value = q_low - (hmask_set ? 0 : 4);
                
                buffer[col] = scale * q_value;
            }
        }

        /**
         * @brief Decode a single row to BF16 (scalar implementation)
         * 
         * Same logic as decodeRow but converts output to BF16.
         */
        void decodeRowToBF16(size_t row_idx, bfloat16 *buffer) const
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q3_KTensor: Row index out of bounds in decodeRowToBF16");
            }

            int cols = shape_[1];
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());

            // Process each element individually (mimics GGML structure)
            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = row_idx * cols + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q3_KBlock *block = &blocks[block_idx];
                float d = simd::fp16_to_fp32(block->d);

                // Determine position in GGML iteration structure
                int group_128 = in_block_idx / 128;     // 0 or 1
                int in_group_128 = in_block_idx % 128;
                
                int subgroup_32 = in_group_128 / 32;    // 0-3
                int in_subgroup_32 = in_group_128 % 32;
                
                int part_16 = in_subgroup_32 / 16;      // 0 or 1
                int in_part_16 = in_subgroup_32 % 16;
                
                // Scale index
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                int8_t sc = simd::extract_q3k_scale(block->scales, scale_idx);
                float scale = d * sc;
                
                // Calculate q offset and shift
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;  // 0, 2, 4, 6
                
                // Position in qs
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                // Extract 2-bit value
                int8_t q_low = (block->qs[qs_idx] >> shift) & 3;
                
                // Calculate hmask bit position using ORIGINAL element index
                int hmask_byte = in_block_idx % 32;
                uint8_t hmask_bit = 1 << (in_block_idx / 32);
                bool hmask_set = (block->hmask[hmask_byte] & hmask_bit) != 0;
                
                // Compute value
                int8_t q_value = q_low - (hmask_set ? 0 : 4);
                
                float fp32_val = scale * q_value;
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
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(raw_data_.data());

            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q3_KBlock *block = &blocks[block_idx];
                float d = simd::fp16_to_fp32(block->d);

                // Determine position in GGML iteration structure
                int group_128 = in_block_idx / 128;     // 0 or 1
                int in_group_128 = in_block_idx % 128;
                
                int subgroup_32 = in_group_128 / 32;    // 0-3
                int in_subgroup_32 = in_group_128 % 32;
                
                int part_16 = in_subgroup_32 / 16;      // 0 or 1
                int in_part_16 = in_subgroup_32 % 16;
                
                // Scale index
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                int8_t sc = simd::extract_q3k_scale(block->scales, scale_idx);
                float scale = d * sc;
                
                // Calculate q offset and shift
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;  // 0, 2, 4, 6
                
                // Position in qs
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                // Extract 2-bit value
                int8_t q_low = (block->qs[qs_idx] >> shift) & 3;
                
                // Calculate hmask bit position using ORIGINAL element index
                int hmask_byte = in_block_idx % 32;
                uint8_t hmask_bit = 1 << (in_block_idx / 32);
                bool hmask_set = (block->hmask[hmask_byte] & hmask_bit) != 0;
                
                // Compute value
                int8_t q_value = q_low - (hmask_set ? 0 : 4);
                
                buffer[i] = scale * q_value;
            }
        }

        // ===== Raw Block Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = sizeof(Q3_KBlock),
                .scale_count = 16,      // 16 scales for 256 elements (16 elements per scale)
                .bits_per_value = 3,
                .is_k_quant = true};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 256; // Q3_K uses 256 elements per super-block

        /**
         * @brief Q3_K block structure (matches llama.cpp/ggml)
         *
         * Layout:
         *   - hmask: 32 bytes - high bit mask (8 values per byte)
         *   - qs: 64 bytes - lower 2 bits (4 values per byte)
         *   - scales: 12 bytes - 6-bit quantized scales (hierarchical packing)
         *   - d: FP16 super-block scale (2 bytes)
         *
         * Total: 110 bytes per block (32 + 64 + 12 + 2)
         *
         * Scale layout:
         *   - 16 scales for 256 elements (16 elements per scale)
         *   - Hierarchical packing allows 6-bit scales in 12 bytes
         *   - Scales are biased by 32 (actual value = stored_value - 32)
         *
         * Value encoding:
         *   - 3 bits per value: 2 low bits (qs) + 1 high bit (hmask)
         *   - High bit acts as sign modifier:
         *     * If hmask bit set: value = qs[i] & 3
         *     * If hmask bit clear: value = (qs[i] & 3) - 4
         *   - Range: -4 to 3 (signed 3-bit)
         */
        struct Q3_KBlock
        {
            uint8_t hmask[32];   // High bit mask (8 values per byte, 256 bits total)
            uint8_t qs[64];      // Lower 2 bits (4 values per byte, 256 values total)
            uint8_t scales[12];  // 6-bit quantized scales (hierarchical packing)
            uint16_t d;          // FP16 super-block scale
        };

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
