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
     * @brief Q2_K quantized tensor (2-bit K-quant with affine quantization)
     *
     * Block format (256 elements per super-block):
     *   - 64 bytes: qs - 2-bit quantized values (4 values per byte)
     *   - 16 bytes: scales - 4-bit scale + 4-bit min per 16 elements
     *   - 2 bytes: d (FP16) - super-block scale for quantized scales
     *   - 2 bytes: dmin (FP16) - super-block scale for quantized mins
     *   - Total: 84 bytes per block
     *
     * Decoding formula (for each element):
     *   scale = d * (scales[i] & 0xF)
     *   min = dmin * (scales[i] >> 4)
     *   value = scale * quant - min
     *
     * Compression: ~12.2× (256 * 4 bytes FP32 = 1024 bytes → 84 bytes)
     *
     * @author David Sanftenberg
     */
    class Q2_KTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct Q2_K tensor from shape and raw Q2_K blocks
         */
        Q2_KTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {
            if (shape_.size() != 2)
            {
                throw std::invalid_argument("Q2_KTensor only supports 2D tensors");
            }

            // Validate data size matches shape
            size_t num_elements = shape_[0] * shape_[1];
            size_t num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t expected_size = num_blocks * sizeof(Q2_KBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "Q2_K raw data size mismatch: expected " +
                    std::to_string(expected_size) + " bytes, got " +
                    std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ===== Shape and Metadata =====

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::Q2_K; }
        float compression_ratio() const override { return 12.2f; } // 1024 bytes FP32 → 84 bytes Q2_K

        // ===== TensorBase Required Methods =====

        size_t element_count() const override
        {
            size_t count = 1;
            for (int dim : shape_)
                count *= dim;
            return count;
        }

        TensorDataType native_type() const override { return TensorDataType::QUANTIZED; }
        std::string type_name() const override { return "Q2_KTensor"; }
        bool is_distributed() const override { return false; }

        // Quantized tensors don't support direct data() access
        float *data() override
        {
            // Print stack trace to help debug where data() is being called from
            std::cerr << "[Q2_KTensor::data()] ERROR: data() called on Q2_KTensor!" << std::endl;
            std::cerr << "[Q2_KTensor::data()] Shape: [" << shape_[0] << ", " << shape_[1] << "]" << std::endl;

#ifdef __linux__
            void *callstack[128];
            int frames = backtrace(callstack, 128);
            char **strs = backtrace_symbols(callstack, frames);
            std::cerr << "[Q2_KTensor::data()] Call stack:" << std::endl;
            for (int i = 0; i < std::min(10, frames); i++)
            {
                std::cerr << "  " << strs[i] << std::endl;
            }
            free(strs);
#endif

            throw std::runtime_error("Q2_KTensor: data() not supported - use decodeRow() instead");
        }

        const float *data() const override
        {
            throw std::runtime_error("Q2_KTensor: data() not supported - use decodeRow() instead");
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
            return std::make_shared<Q2_KTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("Q2_KTensor::copy_from not supported - quantization is lossy");
        }

        // ===== Streaming Decode API =====

        /**
         * @brief Decode a single row to FP32 (scalar implementation)
         * 
         * Implements scalar dequantization logic matching GGML's dequantize_row_q2_K.
         * Q2_K uses affine quantization: value = d * (sc & 0xF) * quant - dmin * (sc >> 4)
         * 
         * Block structure:
         * - 256 elements per block
         * - 16 scale/min pairs (each covers 16 elements)
         * - scales[i]: 8-bit value = (min_i << 4) | scale_i
         * - qs[64]: 2-bit quantized values (4 values per byte)
         * 
         * GGML iteration pattern:
         * - Outer loop: 2 groups of 128 elements (n = 0, 128)
         * - Inner loop: 4 iterations per 128-group (j = 0, 1, 2, 3)
         * - Each iteration: 2 sub-groups of 16 elements
         * - shift cycles: 0 → 2 → 4 → 6 (within 128-group)
         * - scale index increments by 2 per iteration
         */
        void decodeRow(size_t row_idx, float *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q2_KTensor: Row index out of bounds in decodeRow");
            }

            int cols = shape_[1];
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());

            // Process each element individually (mimics GGML structure)
            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = row_idx * cols + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q2_KBlock *block = &blocks[block_idx];
                
                // Extract FP16 scale factors
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                // Determine position in GGML iteration structure
                // 256 elements = 2 groups of 128
                int group_128 = in_block_idx / 128;     // 0 or 1
                int in_group_128 = in_block_idx % 128;
                
                // Within 128-group: 4 subgroups of 32 elements
                int subgroup_32 = in_group_128 / 32;    // 0-3
                int in_subgroup_32 = in_group_128 % 32;
                
                // Within 32-subgroup: 2 parts of 16 elements
                int part_16 = in_subgroup_32 / 16;      // 0 or 1
                int in_part_16 = in_subgroup_32 % 16;   // Position within 16-element part
                
                // Calculate scale index (16 total scales)
                // Advances by 2 each j iteration, offset by group_128*8
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                
                // Extract scale and min from packed byte
                uint8_t sc = block->scales[scale_idx];
                float dl = d * (sc & 0xF);       // Scale from low nibble
                float ml = dmin * (sc >> 4);     // Min from high nibble
                
                // Calculate q offset and shift
                // q advances by 32 bytes every 128 elements (same as Q3_K)
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;  // 0, 2, 4, 6
                
                // Position in qs: base + part*16 + offset
                // Elements 32 positions apart share same byte (same as Q3_K)
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                // Extract 2-bit value from qs
                int8_t quant = (block->qs[qs_idx] >> shift) & 3;
                
                // Apply affine quantization formula
                buffer[col] = dl * quant - ml;
            }
        }

        /**
         * @brief Decode a single row to BF16 (scalar implementation)
         * 
         * Same logic as decodeRow but converts output to BF16.
         */
        void decodeRowToBF16(size_t row_idx, void *buffer) const override
        {
            if (row_idx >= static_cast<size_t>(shape_[0]))
            {
                throw std::out_of_range("Q2_KTensor: Row index out of bounds in decodeRowToBF16");
            }

            int cols = shape_[1];
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());
            bfloat16 *bf16_buffer = static_cast<bfloat16 *>(buffer);

            for (int col = 0; col < cols; ++col)
            {
                size_t elem_idx = row_idx * cols + col;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q2_KBlock *block = &blocks[block_idx];
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                int group_128 = in_block_idx / 128;
                int in_group_128 = in_block_idx % 128;
                int subgroup_32 = in_group_128 / 32;
                int in_subgroup_32 = in_group_128 % 32;
                int part_16 = in_subgroup_32 / 16;
                int in_part_16 = in_subgroup_32 % 16;
                
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                uint8_t sc = block->scales[scale_idx];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                int8_t quant = (block->qs[qs_idx] >> shift) & 3;
                
                float fp32_val = dl * quant - ml;
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
            const Q2_KBlock *blocks = reinterpret_cast<const Q2_KBlock *>(raw_data_.data());

            for (size_t i = 0; i < count; i++)
            {
                size_t elem_idx = offset + i;
                size_t block_idx = elem_idx / BLOCK_SIZE;
                size_t in_block_idx = elem_idx % BLOCK_SIZE;

                const Q2_KBlock *block = &blocks[block_idx];
                float d = simd::fp16_to_fp32(block->d);
                float dmin = simd::fp16_to_fp32(block->dmin);

                int group_128 = in_block_idx / 128;
                int in_group_128 = in_block_idx % 128;
                int subgroup_32 = in_group_128 / 32;
                int in_subgroup_32 = in_group_128 % 32;
                int part_16 = in_subgroup_32 / 16;
                int in_part_16 = in_subgroup_32 % 16;
                
                int scale_idx = group_128 * 8 + subgroup_32 * 2 + part_16;
                uint8_t sc = block->scales[scale_idx];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                
                int q_offset = group_128 * 32;
                int shift = subgroup_32 * 2;
                int qs_idx = q_offset + part_16 * 16 + in_part_16;
                
                int8_t quant = (block->qs[qs_idx] >> shift) & 3;
                
                buffer[i] = dl * quant - ml;
            }
        }

        // ===== Raw Block Access =====

        const uint8_t *raw_data() const override { return raw_data_.data(); }
        size_t raw_size() const override { return raw_data_.size(); }

        const QuantBlockDescriptor &block_descriptor() const override
        {
            static QuantBlockDescriptor desc{
                .elements_per_block = BLOCK_SIZE,
                .bytes_per_block = sizeof(Q2_KBlock),
                .scale_count = 16,      // 16 scale/min pairs for 256 elements
                .bits_per_value = 2,
                .is_k_quant = true};
            return desc;
        }

    private:
        static constexpr int BLOCK_SIZE = 256; // Q2_K uses 256 elements per super-block

        /**
         * @brief Q2_K block structure (matches llama.cpp/ggml)
         *
         * Layout:
         *   - scales: 16 bytes - 4-bit scale + 4-bit min per 16 elements
         *   - qs: 64 bytes - 2 bits per value (4 values per byte)
         *   - d: FP16 super-block scale for scales (2 bytes)
         *   - dmin: FP16 super-block scale for mins (2 bytes)
         *
         * Total: 84 bytes per block (16 + 64 + 2 + 2)
         *
         * Scale/min layout:
         *   - 16 scale/min pairs for 256 elements (16 elements per pair)
         *   - Each byte: low nibble = scale, high nibble = min
         *   - Values are 4-bit unsigned integers
         *
         * Value encoding:
         *   - 2 bits per value (4 values per byte in qs)
         *   - Range: 0-3 (unsigned 2-bit)
         *   - Affine quantization: value = scale * quant - min
         */
        struct Q2_KBlock
        {
            uint8_t scales[16];  // 4-bit scale + 4-bit min per 16 elements
            uint8_t qs[64];      // 2 bits per value (4 values per byte)
            uint16_t d;          // FP16 super-block scale for scales
            uint16_t dmin;       // FP16 super-block scale for mins
        };

        std::vector<int> shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace llaminar
