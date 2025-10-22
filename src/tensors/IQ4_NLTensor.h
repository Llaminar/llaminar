/**
 * @file IQ4_NLTensor.h
 * @brief IQ4_NL (Non-Linear 4-bit) quantized tensor implementation
 *
 * IQ4_NL is a 4-bit non-linear quantization format with simple structure:
 * - Block size: 32 elements
 * - 16-entry lookup table (kvalues_iq4nl)
 * - Single FP16 scale per block
 * - 4.5 bits per weight (18 bytes per 32 elements)
 * - Compression ratio: ~7.1× vs FP32
 *
 * Block structure (18 bytes):
 *   - d (2 bytes): FP16 scale factor
 *   - qs[16] (16 bytes): Packed 4-bit indices (2 per byte)
 *
 * Decode algorithm:
 *   For each byte in qs[]:
 *     low_nibble  = qs[i] & 0xF
 *     high_nibble = qs[i] >> 4
 *     output[j]   = scale * kvalues_iq4nl[low_nibble]
 *     output[j+16] = scale * kvalues_iq4nl[high_nibble]
 *
 * PERFORMANCE OPTIMIZATIONS:
 *
 * Current status (October 2025):
 *   - ✅ AVX512/AVX2 FP32 SIMD (dot_product_simd)
 *   - ✅ Fused dequant+GEMM (cache-tiled, batch-aware)
 *   - ✅ Adaptive tile sizing (64×64 for large batches)
 *   - ⏳ AVX512 VNNI integer dot products (partial implementation)
 *
 * AVX512 VNNI Optimization Opportunity:
 *   Current path: int4 → int8 (LUT) → FP32 → FP32×FP32 (AVX512 FMA)
 *   VNNI path:    int4 → int8 (LUT) → reinterpret as uint8 → uint8×int8 (VNNI) → int32 → FP32
 *
 *   Benefits:
 *     - 2× throughput on Ice Lake+ (2 VNNI ops/cycle vs 1 FMA/cycle)
 *     - Skip FP32 conversion in hot loop
 *     - Better memory bandwidth (int8 vs float)
 *     - Expected speedup: 1.5-2× for small batches
 *
 *   Implementation approach (dot_product_block_vnni function added):
 *     1. Decode 4-bit indices to int8 using pshufb (existing)
 *     2. Quantize input activations (A) to int8 with adaptive scaling
 *     3. Use _mm512_dpbusd_epi32 for int8×uint8 → int32 accumulation
 *     4. Horizontal sum int32 accumulators
 *     5. Convert to FP32 and apply combined scale factor
 *
 *   Challenges:
 *     - dpbusd expects unsigned second operand (kvalues_iq4nl is signed)
 *     - Requires AVX512-VNNI CPU support detection
 *     - Needs -mavx512vnni compiler flag
 *     - Adaptive quantization of A adds overhead (mitigated by small blocks)
 *
 *   Best fit: Small batch path (m=2-16) where blocks are decoded on-the-fly
 *
 * Reference: ggml-quants.c line 2512 (dequantize_row_iq4_nl)
 *
 * @author David Sanftenberg
 * @date 2025-01-15
 */

#pragma once

#include "QuantizedTensorBase.h"
#include "TensorFactory.h"
#include "IQQuantTables.h"
#include "../utils/SIMDHelpers.h"
#include "../utils/CpuFeatures.h"
#include "../QuantizedGemm.h"
#include <vector>
#include <cstring>
#include <omp.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar
{

    /**
     * @brief IQ4_NL block structure (18 bytes, 32 elements)
     *
     * Matches GGML block_iq4_nl from ggml-common.h.
     */
    struct IQ4_NLBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< Packed 4-bit indices (2 per byte)

        static constexpr size_t BLOCK_SIZE = 32; ///< Elements per block
    };

    static_assert(sizeof(IQ4_NLBlock) == 18, "IQ4_NLBlock must be 18 bytes");

    /**
     * @brief IQ4_NL quantized tensor (4.5 bpw, 7.1× compression)
     *
     * Implements non-linear 4-bit quantization with simple lookup table.
     */
    class IQ4_NLTensor : public QuantizedTensorBase
    {
    public:
        /**
         * @brief Construct IQ4_NL tensor from shape and raw data
         *
         * @param shape Tensor dimensions (2D: [rows, cols])
         * @param raw_data Raw bytes (IQ4_NL blocks)
         * @throws std::invalid_argument if shape not 2D or size mismatch
         */
        IQ4_NLTensor(const std::vector<int> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data)
        {

            if (shape_.size() != 2)
            {
                throw std::invalid_argument("IQ4_NLTensor only supports 2D tensors");
            }

            // Per-row block counting: each row is independently padded to block boundary
            size_t rows = shape_[0];
            size_t cols = shape_[1];
            size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            size_t total_blocks = rows * blocks_per_row;
            size_t expected_size = total_blocks * sizeof(IQ4_NLBlock);

            if (raw_data_.size() != expected_size)
            {
                throw std::invalid_argument(
                    "IQ4_NL raw data size mismatch: expected " + std::to_string(expected_size) +
                    " bytes (" + std::to_string(rows) + " rows × " + std::to_string(blocks_per_row) +
                    " blocks/row), got " + std::to_string(raw_data_.size()) + " bytes");
            }
        }

        // ========== Shape and Metadata ==========

        const std::vector<int> &shape() const override { return shape_; }
        int size() const override { return shape_[0] * shape_[1]; }
        int ndim() const override { return 2; }

        QuantType quant_type() const override { return QuantType::IQ4_NL; }
        float compression_ratio() const override { return 7.1f; }

        /**
         * @brief Get logical K dimension (actual column count)
         * @return Actual number of columns in the tensor
         */
        size_t logical_k() const { return static_cast<size_t>(shape_[1]); }

        /**
         * @brief Get physical K dimension (padded to block boundary)
         * @return Padded column count (multiple of 32)
         *
         * For fused kernels: iterate over [0, padded_k) and mask tail in dot product.
         * Padded elements decode to 0.0f.
         */
        size_t padded_k() const
        {
            size_t cols = logical_k();
            return ((cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE) * IQ4_NLBlock::BLOCK_SIZE;
        }

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
            const int rows = shape_[0];
            const int cols = shape_[1];
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const auto &env = debugEnv();

            // If experimental microkernel enabled, use that path (now handles non-multiple columns)
            if (env.dequant.iq4_microkernel)
            {
                decode_to_fp32_microkernel(dst, blocks, rows, cols, blocks_per_row);
                return;
            }

// Row-level parallelization for improved cache locality
#pragma omp parallel for schedule(static) if (rows > 4)
            for (int row = 0; row < rows; ++row)
            {
                const size_t row_block_base = static_cast<size_t>(row) * blocks_per_row;
                float *row_out = dst + static_cast<size_t>(row) * cols;

                // Decode blocks, handling tail block specially
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t global_block_index = row_block_base + b;
                    size_t block_start_col = b * IQ4_NLBlock::BLOCK_SIZE;
                    size_t elements_in_block = std::min(
                        IQ4_NLBlock::BLOCK_SIZE,
                        static_cast<size_t>(cols) - block_start_col);

                    // Decode to temporary buffer (always 32 elements)
                    float temp[IQ4_NLBlock::BLOCK_SIZE];
                    decodeBlock(blocks[global_block_index], temp);

                    // Copy only the valid elements to output
                    std::memcpy(row_out + block_start_col, temp, elements_in_block * sizeof(float));
                }
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
            return std::make_shared<IQ4_NLTensor>(shape_, raw_data_);
        }

        void copy_from(const TensorBase &) override
        {
            throw std::runtime_error("IQ4_NLTensor::copy_from not supported - quantization is lossy");
        }

        // ========== Streaming Decode API ==========

        void decodeRow(size_t row_idx, float *buffer) const override
        {
            // Use per-row block layout: each row has blocks_per_row contiguous blocks
            const int cols = shape_[1];
            const size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const IQ4_NLBlock *row_blocks = blocks + row_idx * blocks_per_row;

            // Decode all blocks for this row
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                float temp[IQ4_NLBlock::BLOCK_SIZE];
                decodeBlock(row_blocks[b], temp);

                // Copy only valid elements (handle tail block)
                size_t block_start_col = b * IQ4_NLBlock::BLOCK_SIZE;
                size_t elements_to_copy = std::min(
                    IQ4_NLBlock::BLOCK_SIZE,
                    static_cast<size_t>(cols) - block_start_col);

                std::memcpy(buffer + block_start_col, temp, elements_to_copy * sizeof(float));
            }
        }

        void decodeSpan(size_t offset, size_t count, float *buffer) const override
        {
            if (offset + count > element_count())
            {
                throw std::out_of_range("IQ4_NLTensor::decodeSpan: range exceeds tensor bounds");
            }

            size_t start_block = offset / IQ4_NLBlock::BLOCK_SIZE;
            size_t end_block = (offset + count - 1) / IQ4_NLBlock::BLOCK_SIZE;

            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());

            size_t buffer_offset = 0;
            for (size_t block_idx = start_block; block_idx <= end_block; ++block_idx)
            {
                float temp[IQ4_NLBlock::BLOCK_SIZE];
                decodeBlock(blocks[block_idx], temp);

                size_t block_start = block_idx * IQ4_NLBlock::BLOCK_SIZE;
                size_t copy_start = std::max(offset, block_start) - block_start;
                size_t copy_end = std::min(offset + count, block_start + IQ4_NLBlock::BLOCK_SIZE) - block_start;
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
                32,   // elements_per_block
                18,   // bytes_per_block
                1,    // scale_count (single scale per block)
                4,    // bits_per_value (4.5 bpw)
                false // is_k_quant
            };
            return desc;
        }

        // ========== Fused Kernel Helpers ==========

        /**
         * @brief Decode a single block for a given row and K-block offset
         *
         * Used by fused GEMM kernels to decode on-the-fly during accumulation.
         *
         * @param row_idx Row index in tensor
         * @param k_block_offset K dimension block offset (in units of 32)
         * @param output Output buffer (must have space for 32 floats)
         */
        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        /**
         * @brief Get direct access to quantized block (for VNNI optimization)
         *
         * Returns const reference to the raw quantized block data without decoding.
         * Used by VNNI-optimized kernels to process integer data directly.
         *
         * @param row_idx Row index in tensor
         * @param k_block_offset K dimension block offset (in units of 32)
         * @return Const reference to IQ4_NLBlock
         */
        const IQ4_NLBlock &get_block_at(size_t row_idx, size_t k_block_offset) const
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return blocks[block_idx];
        }

        /**
         * @brief Decode multiple rows' blocks into SoA buffer for tiled GEMM
         *
         * Decodes tile_n consecutive rows at a given K-block offset.
         * Output layout: [tile_n][32] row-major (SoA across rows).
         *
         * @param row_start Starting row index
         * @param tile_n Number of rows to decode
         * @param k_block_offset K dimension block offset (in units of 32)
         * @param output Output buffer (must have space for tile_n * 32 floats)
         */
        void decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float *output) const
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());

            for (size_t i = 0; i < tile_n; ++i)
            {
                const size_t row_idx = row_start + i;
                const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
                float *row_output = output + i * IQ4_NLBlock::BLOCK_SIZE;
                decodeBlock(blocks[block_idx], row_output);
            }
        }

    private:
        std::vector<int> shape_;        ///< Tensor dimensions
        std::vector<uint8_t> raw_data_; ///< Raw quantized data

#if defined(__AVX512F__)
        /**
         * @brief AVX512-optimized IQ4_NL block decode
         *
         * Uses SIMD helper library for efficient int8 to float32 conversion.
         * Processes 16 values at a time with AVX512 intrinsics.
         */
        static void decodeBlockAVX512(const IQ4_NLBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);
            // Prepare lookup buffer (32 int8 values)
            alignas(64) int8_t lookup_values[32];
            for (size_t j = 0; j < 16; ++j)
            {
                const uint8_t qbyte = block.qs[j];
                lookup_values[j] = kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
                lookup_values[j + 16] = kvalues_iq4nl[qbyte >> 4]; // High nibble
            }
            // Convert and scale: 16 elements at a time (AVX512 helper)
            simd::convert_i8_to_f32_scaled_avx512(lookup_values, d, output);
            simd::convert_i8_to_f32_scaled_avx512(lookup_values + 16, d, output + 16);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief AVX2-optimized IQ4_NL block decode
         *
         * Uses SIMD helper library for efficient int8 to float32 conversion.
         * Processes 8 values at a time with AVX2 intrinsics.
         */
        static void decodeBlockAVX2(const IQ4_NLBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);
            // Prepare lookup buffer (32 int8 values)
            alignas(32) int8_t lookup_values[32];
            for (size_t j = 0; j < 16; ++j)
            {
                const uint8_t qbyte = block.qs[j];
                lookup_values[j] = kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
                lookup_values[j + 16] = kvalues_iq4nl[qbyte >> 4]; // High nibble
            }
            // Convert and scale: 8 elements at a time (AVX2 helper)
            simd::convert_i8_to_f32_scaled_avx2(lookup_values, d, output);
            simd::convert_i8_to_f32_scaled_avx2(lookup_values + 8, d, output + 8);
            simd::convert_i8_to_f32_scaled_avx2(lookup_values + 16, d, output + 16);
            simd::convert_i8_to_f32_scaled_avx2(lookup_values + 24, d, output + 24);
        }
#endif

        /**
         * @brief Decode one IQ4_NL block (32 elements) to FP32
         *
         * Implements GGML dequantize_row_iq4_nl algorithm (ggml-quants.c line 2512).
         * Dispatches to AVX512/AVX2 version if available, otherwise uses scalar fallback.
         *
         * Algorithm:
         * 1. Extract FP16 scale d
         * 2. Process 16 bytes (each contains 2 4-bit indices):
         *    - Low nibble  (bits 0-3) → output[j]
         *    - High nibble (bits 4-7) → output[j+16]
         *    - Lookup: kvalues_iq4nl[index] (int8_t values)
         *    - Apply: y[j] = d * kvalues_iq4nl[index]
         *
         * @param block Input IQ4_NL block
         * @param output Output buffer (must have space for 32 floats)
         */
        static void decodeBlock(const IQ4_NLBlock &block, float *output)
        {
            const auto &env = debugEnv();
            // Optional direct decode bypasses SIMD helper temp buffer
            if (env.dequant.iq4_direct_decode)
            {
                const float d = simd::fp16_to_fp32(block.d);
#pragma omp simd
                for (size_t j = 0; j < 16; ++j)
                {
                    const uint8_t qbyte = block.qs[j];
                    output[j] = d * static_cast<float>(kvalues_iq4nl[qbyte & 0x0F]);
                    output[j + 16] = d * static_cast<float>(kvalues_iq4nl[qbyte >> 4]);
                }
                return;
            }
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

// Decode 32 elements from 16 bytes
// Each byte contains 2 4-bit indices
#pragma omp simd
            for (size_t j = 0; j < 16; ++j)
            {
                const uint8_t qbyte = block.qs[j];

                // Low 4 bits -> first half of output
                const uint8_t idx_low = qbyte & 0x0F;
                output[j] = d * static_cast<float>(kvalues_iq4nl[idx_low]);

                // High 4 bits -> second half of output
                const uint8_t idx_high = qbyte >> 4;
                output[j + 16] = d * static_cast<float>(kvalues_iq4nl[idx_high]);
            }
        }

        // Experimental multi-block microkernel (AVX2/AVX512). Processes several blocks per row in one loop.
        static void decode_to_fp32_microkernel(float *dst, const IQ4_NLBlock *blocks, int rows, int cols, size_t blocks_per_row)
        {
            const bool has_avx512 = simd::cpu_supports_avx512();
            const bool has_avx2 = simd::cpu_supports_avx2();

#pragma omp parallel for schedule(static) if (rows > 4)
            for (int row = 0; row < rows; ++row)
            {
                float *out_row = dst + static_cast<size_t>(row) * cols;
                const IQ4_NLBlock *row_blocks = blocks + static_cast<size_t>(row) * blocks_per_row;
                size_t b = 0;

                // For tail blocks (cols not multiple of 32), we need to handle carefully
                // Process full blocks with vectorized path, then handle tail specially
                size_t full_blocks = (cols / IQ4_NLBlock::BLOCK_SIZE);

                // AVX512 path: process 2 blocks per iteration (32 + 32 = 64 outputs) with nibble vectorization
#if defined(__AVX512F__)
                if (has_avx512)
                {
                    for (; b + 2 <= full_blocks; b += 2)
                    {
                        decodeBlockVectorizedAVX512(row_blocks[b], out_row + b * IQ4_NLBlock::BLOCK_SIZE);
                        decodeBlockVectorizedAVX512(row_blocks[b + 1], out_row + (b + 1) * IQ4_NLBlock::BLOCK_SIZE);
                    }
                }
#endif
#if defined(__AVX2__)
                if (!has_avx512 && has_avx2)
                {
                    // AVX2: process 4 blocks per loop (unrolled) using shuffle-based nibble expansion
                    for (; b + 4 <= full_blocks; b += 4)
                    {
                        decodeBlockVectorizedAVX2(row_blocks[b + 0], out_row + (b + 0) * IQ4_NLBlock::BLOCK_SIZE);
                        decodeBlockVectorizedAVX2(row_blocks[b + 1], out_row + (b + 1) * IQ4_NLBlock::BLOCK_SIZE);
                        decodeBlockVectorizedAVX2(row_blocks[b + 2], out_row + (b + 2) * IQ4_NLBlock::BLOCK_SIZE);
                        decodeBlockVectorizedAVX2(row_blocks[b + 3], out_row + (b + 3) * IQ4_NLBlock::BLOCK_SIZE);
                    }
                }
#endif
                // Process remaining full blocks
                for (; b < full_blocks; ++b)
                {
                    decodeBlock(row_blocks[b], out_row + b * IQ4_NLBlock::BLOCK_SIZE);
                }

                // Handle tail block if present (cols not multiple of 32)
                if (b < blocks_per_row)
                {
                    float temp[IQ4_NLBlock::BLOCK_SIZE];
                    decodeBlock(row_blocks[b], temp);
                    size_t tail_elements = cols - (b * IQ4_NLBlock::BLOCK_SIZE);
                    std::memcpy(out_row + b * IQ4_NLBlock::BLOCK_SIZE, temp, tail_elements * sizeof(float));
                }
            }
        }

#if defined(__AVX2__)
        // Vectorized nibble expansion using pshufb for AVX2; eliminates intermediate per-block buffer
        static inline void decodeBlockVectorizedAVX2(const IQ4_NLBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);
            // Load 16 bytes of qs
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            // Mask for low nibbles
            __m128i low_mask = _mm_set1_epi8(0x0F);
            __m128i low_idx = _mm_and_si128(qs, low_mask);
            // High nibbles: shift right 4 bits per byte -> use 16-bit shift then mask
            __m128i high_shift = _mm_srli_epi16(qs, 4);
            __m128i high_idx = _mm_and_si128(high_shift, low_mask);
            // Load LUT (16 int8 entries) into vector
            __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
            // Shuffle to map indices → values (int8)
            __m128i low_vals = _mm_shuffle_epi8(lut, low_idx);
            __m128i high_vals = _mm_shuffle_epi8(lut, high_idx);
            // Store to temp contiguous array of 32 int8 values
            alignas(32) int8_t tmp[32];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp), low_vals);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp + 16), high_vals);
            // Convert in 4 chunks of 8 using existing helper
            simd::convert_i8_to_f32_scaled_avx2(tmp, d, output);
            simd::convert_i8_to_f32_scaled_avx2(tmp + 8, d, output + 8);
            simd::convert_i8_to_f32_scaled_avx2(tmp + 16, d, output + 16);
            simd::convert_i8_to_f32_scaled_avx2(tmp + 24, d, output + 24);
        }
#endif

#if defined(__AVX512F__)
        // AVX512 variant: expand low/high nibbles, then convert 16+16 using existing helpers
        static inline void decodeBlockVectorizedAVX512(const IQ4_NLBlock &block, float *output)
        {
            const float d = simd::fp16_to_fp32(block.d);
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);
            __m128i low_idx = _mm_and_si128(qs, low_mask);
            __m128i high_shift = _mm_srli_epi16(qs, 4);
            __m128i high_idx = _mm_and_si128(high_shift, low_mask);
            __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
            __m128i low_vals = _mm_shuffle_epi8(lut, low_idx);
            __m128i high_vals = _mm_shuffle_epi8(lut, high_idx);
            alignas(64) int8_t tmp[32];
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp), low_vals);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tmp + 16), high_vals);
            // Two wide conversions (16 each) using AVX512 helper; we reuse existing helper taking 16 int8
            simd::convert_i8_to_f32_scaled_avx512(tmp, d, output);
            simd::convert_i8_to_f32_scaled_avx512(tmp + 16, d, output + 16);
        }
#endif

        /**
         * @brief Create fused quantized GEMM implementation for this tensor
         *
         * @return Unique pointer to ITensorGemm implementation, or nullptr if not supported
         *
         * This enables adaptiveMatMul to use fused dequant+GEMM path instead of
         * full decode + BLAS.
         */
        std::unique_ptr<ITensorGemm> createGemm() const;

        /**
         * @brief Create fused quantized GEMM implementation for this tensor
         *
         * @return ITensorGemm pointer (caller takes ownership), or nullptr if not supported
         *
         * This enables adaptiveMatMul to use fused dequant+GEMM path instead of
         * full decode + BLAS.
         */
        ITensorGemm *createGemmRaw() const override;

        /**
         * @brief Decode one IQ4_NL block to BF16
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

    /**
     * @brief Fused quantized GEMM implementation for IQ4_NL tensors
     *
     * Implements tiled decode-then-accumulate pattern to avoid full weight materialization.
     * Uses IQ4_NLTensor::decode_block_at() for efficient block-by-block decoding.
     *
     * Memory footprint: Decodes 32-element blocks on-the-fly, no large intermediate buffers.
     *
     * Pattern:
     *   For each output element C[i,j]:
     *     C[i,j] = sum_k( A[i,k] * B[j,k] )   // B is IQ4_NL quantized
     *
     *   Tiled over K dimension (32 elements at a time):
     *     For each K-block:
     *       decode B[j, k_block] -> temp[32]
     *       accumulate dot(A[i, k_start:k_start+32], temp)
     */
    class IQ4_NLQuantizedGemm : public ITensorGemm
    {
    public:
        explicit IQ4_NLQuantizedGemm(const IQ4_NLTensor *tensor)
            : tensor_(tensor) {}

        /**
         * @brief Report BF16 activation support
         */
        bool supports_bf16() const override { return true; }

        /**
         * @brief FP32 activation × IQ4_NL weight multiply (standard path)
         */
        bool multiply(const float *A, float *C,
                      int m, int n, int k,
                      bool transpose_B,
                      float alpha,
                      float beta,
                      int row_offset = 0,
                      int row_count = -1) override
        {

            if (!tensor_)
                return false;

            // Default row_count to n if not specified
            if (row_count < 0)
                row_count = n;

            // Validate row range
            if (row_offset < 0 || row_offset + row_count > tensor_->shape()[0])
            {
                return false; // Invalid row range
            }

            // Validate dimensions against tensor shape
            // B is [total_rows, k] if transpose_B=true (most common: weight is [out_features, in_features])
            // We'll compute only rows [row_offset : row_offset+row_count]
            int expected_cols = transpose_B ? k : n;

            if (static_cast<int>(tensor_->logical_k()) != expected_cols)
            {
                return false; // Dimension mismatch
            }

            // Compute C = alpha * A @ B[row_offset:row_offset+row_count]^T + beta * C
            // A: [m, k], B: [row_count, k] (subset of tensor rows)
            // C: [m, row_count] (note: n == row_count in distributed case)

            const int num_k_blocks = (k + 31) / 32;

            // Strategy selection based on batch size:
            // - m=1: Row-wise (no reuse benefit from blocking)
            // - m=2-16: Cache-blocked algorithm (decode block, use immediately)
            // - m>16: Row-wise algorithm (decode full row, reuse across all m)

            if (m >= 2 && m <= 16)
            {
                // Cache-blocked algorithm: better for small m (2-16)
                // Keeps 32-element blocks hot in L1 cache, minimizes allocations
                //
                // VNNI optimization disabled for FP32 activations (use int8 overload for VNNI)
                const bool use_vnni = false;

#pragma omp parallel for schedule(static) if (n > 128)
                for (int j = 0; j < n; ++j)
                {
                    // Per-thread accumulator array (small: m≤16 → max 64 bytes)
                    float acc[16] = {0}; // Max m=16

                    // Process one K-block at a time (FP32 decode path)
                    for (int kb = 0; kb < num_k_blocks; ++kb)
                    {
                        size_t k_start = kb * 32;
                        size_t k_count = std::min(32, k - static_cast<int>(k_start));

                        // FP32 fallback path: decode to FP32 then compute
                        alignas(64) float B_block[32];
                        tensor_->decode_block_at(row_offset + j, kb, B_block); // Use row_offset

                        // Immediately use block for all M rows (hot in L1 cache)
                        for (int i = 0; i < m; ++i)
                        {
                            const float *A_row = A + i * k + k_start;
                            acc[i] += dot_product_simd(A_row, B_block, k_count);
                        }
                    }

                    // Write accumulated results to C
                    for (int i = 0; i < m; ++i)
                    {
                        size_t c_idx = i * n + j;
                        C[c_idx] = alpha * acc[i] + beta * C[c_idx];
                    }
                }
            }
            else
            {
                // Row-wise algorithm with hybrid adaptive cache blocking: optimal for m>16
                // Dynamically adjust tile sizes based on problem characteristics

                // Hybrid strategy: Detect compute-bound vs memory-bound operations
                // - Compute-bound (square/tall matrices): Optimize for compute reuse (larger tiles)
                // - Memory-bound (wide matrices): Optimize for cache locality (smaller N_TILE)

                int M_TILE, N_TILE;

                // Compute aspect ratio to determine workload characteristics
                // ratio > 1.5: Compute-bound (tall/square) - Q-projection (896×896)
                // ratio < 0.7: Memory-bound (wide) - FFN (896→4864)
                const float aspect_ratio = static_cast<float>(n) / static_cast<float>(m > 0 ? m : 1);
                const bool is_wide_output = aspect_ratio > 2.0f;                     // FFN-like: 4864/2048 = 2.37
                const bool is_square = aspect_ratio >= 0.5f && aspect_ratio <= 2.0f; // Q-proj-like

                if (is_wide_output)
                {
                    // MEMORY-BOUND PATH (FFN: wide output, limited by bandwidth)
                    // Strategy: Small N_TILE (reduce decode buffer), larger M_TILE (reuse decoded columns)
                    // Target: N_TILE * k * 4 ≤ 96KB, maximize M_TILE for compute reuse

                    if (m >= 4096)
                    {
                        N_TILE = 24; // ~85KB decode buffer for k=896
                        M_TILE = 32; // Aggressive cache-fit for very large batches
                    }
                    else if (m >= 2048)
                    {
                        N_TILE = 24; // Keep decode buffer small
                        M_TILE = 64; // Good reuse across many rows
                    }
                    else if (m >= 1024)
                    {
                        N_TILE = 32; // Moderate decode buffer
                        M_TILE = 96; // High reuse
                    }
                    else
                    {
                        N_TILE = 48;  // Larger decode for small batches
                        M_TILE = 128; // Minimize overhead
                    }
                }
                else if (is_square)
                {
                    // COMPUTE-BOUND PATH (Q-proj: square matrix, limited by compute)
                    // Strategy: Balanced tiling, prioritize L2 cache fit for entire working set
                    // Target: (M_TILE + N_TILE) * k * 4 ≤ 192KB total

                    if (m >= 4096 || n >= 4096)
                    {
                        M_TILE = 32;
                        N_TILE = 32; // Very aggressive for L2 fit
                    }
                    else if (m >= 2048 || n >= 2048)
                    {
                        M_TILE = 48;
                        N_TILE = 48; // Moderate tiling
                    }
                    else if (m >= 1024 || n >= 1024)
                    {
                        M_TILE = 64;
                        N_TILE = 64; // Balanced
                    }
                    else if (m >= 512 || n >= 512)
                    {
                        M_TILE = 96;
                        N_TILE = 96; // Standard
                    }
                    else
                    {
                        M_TILE = 128;
                        N_TILE = 128; // Minimize overhead for small ops
                    }
                }
                else
                {
                    // TALL MATRIX PATH (n < m/2: more rows than columns)
                    // Strategy: Small N_TILE, very large M_TILE for maximum reuse

                    if (m >= 4096)
                    {
                        M_TILE = 64;
                        N_TILE = 24;
                    }
                    else if (m >= 2048)
                    {
                        M_TILE = 96;
                        N_TILE = 32;
                    }
                    else
                    {
                        M_TILE = 128;
                        N_TILE = 48;
                    }
                }

#pragma omp parallel
                {
                    // Thread-local buffer for N_TILE B columns (decode multiple at once)
                    std::vector<float> B_tile(k * N_TILE);

#pragma omp for schedule(dynamic)
                    for (int jj = 0; jj < n; jj += N_TILE)
                    {
                        int n_block = std::min(N_TILE, n - jj);

                        // Decode N_TILE columns of B at once for better memory access pattern
                        for (int j_local = 0; j_local < n_block; ++j_local)
                        {
                            int j = jj + j_local;
                            float *B_col = B_tile.data() + j_local * k;

                            for (int kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * 32;
                                tensor_->decode_block_at(row_offset + j, kb, B_col + k_start); // Use row_offset
                            }
                        }

                        // Now process all M×N_block with decoded tile
                        for (int ii = 0; ii < m; ii += M_TILE)
                        {
                            int m_block = std::min(M_TILE, m - ii);

                            // Prefetch next M-tile of A for better cache behavior
                            if (ii + M_TILE < m)
                            {
                                const float *next_A = A + (ii + M_TILE) * k;
                                for (int pf = 0; pf < std::min(M_TILE, m - ii - M_TILE); pf += 8)
                                {
                                    __builtin_prefetch(next_A + pf * k, 0, 1); // Prefetch with low temporal locality
                                }
                            }

                            // Compute M_TILE × N_block outputs
                            for (int i_local = 0; i_local < m_block; ++i_local)
                            {
                                int i = ii + i_local;
                                const float *A_row = A + i * k;

                                for (int j_local = 0; j_local < n_block; ++j_local)
                                {
                                    int j = jj + j_local;
                                    const float *B_col = B_tile.data() + j_local * k;

                                    float acc = dot_product_simd(A_row, B_col, k);
                                    size_t c_idx = i * n + j;
                                    C[c_idx] = alpha * acc + beta * C[c_idx];
                                }
                            }
                        }
                    }
                }
            }

            return true;
        }

        /**
         * @brief Int8 activation × IQ4_NL weight multiply (VNNI-optimized path)
         *
         * Uses AVX512-VNNI for uint8×int8 dot products. Since activations are already
         * quantized to int8, we avoid the expensive FP32→int8 conversion overhead that
         * negates VNNI benefits for FP32 inputs.
         *
         * @param A_int8 Pre-quantized activations in int8 [m, k]
         * @param A_scale Per-row scale factors for A [m]
         * @param C Output in FP32 [m, n]
         * @param m Number of output rows
         * @param n Number of output columns
         * @param k Inner dimension
         * @param transpose_B Whether B (weights) are transposed
         * @param alpha Output scale factor
         * @param beta Accumulation factor for C
         * @return true if successful
         */
        bool multiply_int8(const int8_t *A_int8, const float *A_scale, float *C,
                           int m, int n, int k,
                           bool transpose_B,
                           float alpha,
                           float beta)
        {

            if (!tensor_)
                return false;

            // Validate dimensions
            int expected_rows = transpose_B ? n : k;
            int expected_cols = transpose_B ? k : n;

            if (tensor_->shape()[0] != expected_rows ||
                static_cast<int>(tensor_->logical_k()) != expected_cols)
            {
                return false;
            }

#if defined(__AVX512VNNI__)
            if (!simd::cpu_supports_avx512())
            {
                return false; // VNNI not available
            }

            const int num_k_blocks = (k + 31) / 32;

// VNNI path for pre-quantized int8 activations
#pragma omp parallel for schedule(static) if (n > 128)
            for (int j = 0; j < n; ++j)
            {
                float acc[16] = {0}; // Support up to m=16

                for (int kb = 0; kb < num_k_blocks; ++kb)
                {
                    size_t k_start = kb * 32;
                    const IQ4_NLBlock &block = tensor_->get_block_at(j, kb);

                    for (int i = 0; i < m; ++i)
                    {
                        const int8_t *A_row = A_int8 + i * k + k_start;
                        // Call VNNI kernel with pre-quantized input
                        float result = dot_product_block_vnni_prequantized(A_row, A_scale[i], block);
                        acc[i] += result;
                    }
                }

                // Write results
                for (int i = 0; i < m; ++i)
                {
                    size_t c_idx = i * n + j;
                    C[c_idx] = alpha * acc[i] + beta * C[c_idx];
                }
            }

            return true;
#else
            return false; // VNNI not compiled in
#endif
        }

        /**
         * @brief BF16 activation × IQ4_NL weight multiply (streaming conversion)
         *
         * Optimized BF16 path with streaming conversion:
         * 1. BF16→FP32 conversion is just a 16-bit left shift (cheap!)
         * 2. Integrate conversion into the GEMM loop (no upfront allocation)
         * 3. Decode IQ4_NL blocks to FP32 on-the-fly
         * 4. Accumulate in FP32 for full precision
         * 5. Optional: Output can be BF16 (downconvert at write)
         *
         * Performance: Should match FP32 path since BF16→FP32 is essentially free
         * (simple bit manipulation, no FP arithmetic).
         *
         * @param A_bf16 Activations in BF16 format [m, k]
         * @param C Output in FP32 [m, n]
         * @param m Number of output rows
         * @param n Number of output columns
         * @param k Inner dimension
         * @param transpose_B Whether B (weights) are transposed
         * @param alpha Output scale factor
         * @param beta Accumulation factor for C
         * @return true if successful
         */
        bool multiply_bf16(const uint16_t *A_bf16, float *C,
                           int m, int n, int k,
                           bool transpose_B,
                           float alpha,
                           float beta,
                           int row_offset = 0,
                           int row_count = -1) override
        {

            if (!tensor_)
                return false;

            // Default row_count to n if not specified
            if (row_count < 0)
                row_count = n;

            // Validate row range
            if (row_offset < 0 || row_offset + row_count > tensor_->shape()[0])
            {
                return false; // Invalid row range
            }

            // Validate dimensions against tensor shape
            int expected_cols = transpose_B ? k : n;

            if (static_cast<int>(tensor_->logical_k()) != expected_cols)
            {
                return false;
            }

            const int num_k_blocks = (k + 31) / 32;

            // Strategy: Repack entire rows once, amortize over all columns
            // Adaptive sizing: Detect L1 cache size but allow spill to L2
            // Empirical finding: 64KB working set optimal (may use L1+L2)
            // Conservative: Cap at 2× L1 cache size (fits comfortably in typical L2)

            // Calculate max rows that fit
            const size_t l1_cache = CpuFeatures::instance().l1_cache_size();
            const size_t target_working_set = l1_cache * 2; // Allow 2× L1 (spill to L2)
            const size_t bytes_per_row = k * sizeof(float); // k elements × 4 bytes
            const int max_repack_rows = std::max(2, std::min(32, static_cast<int>(target_working_set / bytes_per_row)));

            if (m >= 2 && m <= max_repack_rows)
            {
                // Repack all m rows completely (once, outside column loop)
                // Working set adapts to CPU cache hierarchy:
                //   32KB L1 → 16 rows @ 2×L1, 48KB L1 → 24 rows, 64KB L1 → 32 rows (k=1024)
                // Use stack-allocated buffer (256KB max = 32 rows × 2048 elements)
                alignas(64) float A_repacked_buffer[32 * 2048]; // Max possible size

                for (int i = 0; i < m; ++i)
                {
                    const uint16_t *A_row_bf16 = A_bf16 + i * k;
                    float *A_row_fp32 = A_repacked_buffer + i * k;
                    repack_bf16_to_fp32(A_row_bf16, A_row_fp32, k);
                }

// Cache-blocked algorithm: now using pre-repacked FP32 data
#pragma omp parallel for schedule(static) if (n > 128)
                for (int j = 0; j < n; ++j)
                {
                    float acc[32] = {0}; // Max 32 rows (stack-allocated)

                    // Process one K-block at a time
                    for (int kb = 0; kb < num_k_blocks; ++kb)
                    {
                        size_t k_start = kb * 32;
                        size_t k_count = std::min(32, k - static_cast<int>(k_start));

                        // Decode weight block to FP32
                        alignas(64) float B_block[32];
                        tensor_->decode_block_at(row_offset + j, kb, B_block); // Use row_offset

                        // Direct FP32×FP32 dot products (no conversion needed!)
                        for (int i = 0; i < m; ++i)
                        {
                            const float *A_row = A_repacked_buffer + i * k + k_start;
                            acc[i] += dot_product_simd(A_row, B_block, k_count);
                        }
                    }

                    // Write accumulated results
                    for (int i = 0; i < m; ++i)
                    {
                        size_t c_idx = i * n + j;
                        C[c_idx] = alpha * acc[i] + beta * C[c_idx];
                    }
                }
            }
            else
            {
                // Row-wise algorithm with cache tiling: optimal for m > max_repack_rows
                // (Fallback path when batch too large for L1-optimized repack)
                constexpr int M_TILE = 64;
                constexpr int N_TILE = 64;

#pragma omp parallel
                {
                    // Thread-local buffers
                    std::vector<float> B_tile(k * N_TILE); // Decoded weights
                    std::vector<float> A_tile(k * M_TILE); // Converted activations

#pragma omp for schedule(dynamic)
                    for (int jj = 0; jj < n; jj += N_TILE)
                    {
                        int n_block = std::min(N_TILE, n - jj);

                        // Decode N_TILE columns of B (weights)
                        for (int j_local = 0; j_local < n_block; ++j_local)
                        {
                            int j = jj + j_local;
                            float *B_col = B_tile.data() + j_local * k;

                            for (int kb = 0; kb < num_k_blocks; ++kb)
                            {
                                size_t k_start = kb * 32;
                                tensor_->decode_block_at(row_offset + j, kb, B_col + k_start); // Use row_offset
                            }
                        }

                        // Process M_TILE rows at a time
                        for (int ii = 0; ii < m; ii += M_TILE)
                        {
                            int m_block = std::min(M_TILE, m - ii);

// Convert BF16→FP32 for M_TILE rows in one pass
#if defined(__AVX512F__)
                            if (simd::cpu_supports_avx512())
                            {
                                // Vectorized conversion: 32 elements at a time per row
                                for (int i_local = 0; i_local < m_block; ++i_local)
                                {
                                    int i = ii + i_local;
                                    const uint16_t *A_row_bf16 = A_bf16 + i * k;
                                    float *A_row_fp32 = A_tile.data() + i_local * k;

                                    // Process in 32-element chunks
                                    size_t kk = 0;
                                    for (; kk + 32 <= static_cast<size_t>(k); kk += 32)
                                    {
                                        simd::convert_bf16_to_fp32_avx512(A_row_bf16 + kk, A_row_fp32 + kk);
                                    }

                                    // Scalar remainder
                                    for (; kk < static_cast<size_t>(k); ++kk)
                                    {
                                        A_row_fp32[kk] = simd::bf16_to_fp32(A_row_bf16[kk]);
                                    }
                                }
                            }
                            else
#endif
                            {
                                // Scalar conversion (still reasonably fast - just a shift)
                                for (int i_local = 0; i_local < m_block; ++i_local)
                                {
                                    int i = ii + i_local;
                                    const uint16_t *A_row_bf16 = A_bf16 + i * k;
                                    float *A_row_fp32 = A_tile.data() + i_local * k;

                                    for (int kk = 0; kk < k; ++kk)
                                    {
                                        A_row_fp32[kk] = simd::bf16_to_fp32(A_row_bf16[kk]);
                                    }
                                }
                            }

                            // Compute M_block × N_block tile (FP32×FP32 - no conversion overhead)
                            for (int i_local = 0; i_local < m_block; ++i_local)
                            {
                                int i = ii + i_local;
                                const float *A_row = A_tile.data() + i_local * k;

                                for (int j_local = 0; j_local < n_block; ++j_local)
                                {
                                    int j = jj + j_local;
                                    const float *B_col = B_tile.data() + j_local * k;

                                    float acc = dot_product_simd(A_row, B_col, k);
                                    size_t c_idx = i * n + j;
                                    C[c_idx] = alpha * acc + beta * C[c_idx];
                                }
                            }
                        }
                    }
                }
            }

            return true;
        }

    private:
        const IQ4_NLTensor *tensor_;

        /**
         * @brief Repack BF16 to spaced FP32 format for direct loading
         *
         * Converts packed BF16 (2 bytes/value) to FP32 format (4 bytes/value)
         * by placing each BF16's 16 bits in the upper half of an FP32 word.
         *
         * This allows direct FP32 SIMD loads without register shuffling.
         *
         * @param bf16_in Input BF16 data (packed, 2 bytes/element)
         * @param fp32_out Output FP32 buffer (4 bytes/element, must be 64-byte aligned)
         * @param count Number of elements to convert
         */
        static inline void repack_bf16_to_fp32(const uint16_t *bf16_in, float *fp32_out, size_t count)
        {
// Vectorized repack using intrinsics (same conversion, but store as FP32)
#if defined(__AVX512F__)
            if (simd::cpu_supports_avx512())
            {
                size_t i = 0;
                for (; i + 16 <= count; i += 16)
                {
                    // Load 16 BF16 values
                    __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_in + i));

                    // Zero-extend to 32-bit, shift left 16 to put in upper half
                    __m512i fp32_int = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_vec), 16);

                    // Store as FP32
                    _mm512_storeu_ps(fp32_out + i, _mm512_castsi512_ps(fp32_int));
                }

                // Scalar remainder
                for (; i < count; ++i)
                {
                    fp32_out[i] = simd::bf16_to_fp32(bf16_in[i]);
                }
                return;
            }
#endif

            // Scalar fallback
            for (size_t i = 0; i < count; ++i)
            {
                fp32_out[i] = simd::bf16_to_fp32(bf16_in[i]);
            }
        }

        /**
         * @brief Fused BF16→FP32 + dot product (inline conversion)
         *
         * Converts BF16 to FP32 inline with FMA, avoiding intermediate buffers.
         * BF16→FP32 is just a 16-bit left shift, so this should be nearly free.
         *
         * @param a_bf16 BF16 input vector
         * @param b_fp32 FP32 input vector
         * @param count Number of elements
         * @return Dot product result
         */
        static inline float dot_product_bf16_fp32(const uint16_t *a_bf16, const float *b_fp32, size_t count)
        {
#if defined(__AVX512F__)
            if (count == 32 && simd::cpu_supports_avx512())
            {
                // Load first 16 BF16 values (256 bits) and convert to FP32
                __m256i bf16_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_bf16));
                __m256i bf16_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_bf16 + 16));

                // BF16→FP32: Zero-extend 16-bit to 32-bit, then shift left by 16
                // This puts the BF16 bits in the upper 16 bits of FP32 (correct position)
                __m512i fp32_lo_int = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
                __m512i fp32_hi_int = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);

                // Reinterpret as FP32
                __m512 a_lo = _mm512_castsi512_ps(fp32_lo_int);
                __m512 a_hi = _mm512_castsi512_ps(fp32_hi_int);

                // Load B (FP32 weights) and compute dot product with FMA
                __m512 b_lo = _mm512_loadu_ps(b_fp32);
                __m512 b_hi = _mm512_loadu_ps(b_fp32 + 16);

                // Compute dot product: a_lo·b_lo + a_hi·b_hi
                __m512 prod_lo = _mm512_mul_ps(a_lo, b_lo);
                __m512 sum = _mm512_fmadd_ps(a_hi, b_hi, prod_lo); // sum = a_hi*b_hi + prod_lo

                // Horizontal reduction to scalar
                return _mm512_reduce_add_ps(sum);
            }
#endif
            // Scalar fallback
            float result = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                result += simd::bf16_to_fp32(a_bf16[i]) * b_fp32[i];
            }
            return result;
        }

        /**
         * @brief Vectorized dot product with SIMD acceleration
         *
         * Computes sum(a[i] * b[i]) for i in [0, count) using AVX512/AVX2/scalar fallback.
         *
         * @param a First vector (must have at least count elements)
         * @param b Second vector (must have at least count elements, 64-byte aligned for AVX512)
         * @param count Number of elements to process (≤32 for IQ4_NL blocks)
         * @return Dot product result
         */
        static inline float dot_product_simd(const float *a, const float *b, size_t count)
        {
#if defined(__AVX512F__)
            // AVX512: Process 16 floats at a time
            __m512 sum = _mm512_setzero_ps();

            // Main loop: 16 elements per iteration
            size_t i = 0;
            for (; i + 16 <= count; i += 16)
            {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_load_ps(b + i); // Aligned load
                sum = _mm512_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum of vector
            float result = _mm512_reduce_add_ps(sum);

            // Scalar tail (0-15 remaining elements)
            for (; i < count; ++i)
            {
                result += a[i] * b[i];
            }

            return result;

#elif defined(__AVX2__)
            // AVX2: Process 8 floats at a time
            __m256 sum = _mm256_setzero_ps();

            // Main loop: 8 elements per iteration
            size_t i = 0;
            for (; i + 8 <= count; i += 8)
            {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum of vector
            __m128 sum_high = _mm256_extractf128_ps(sum, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float result = _mm_cvtss_f32(sum128);

            // Scalar tail (0-7 remaining elements)
            for (; i < count; ++i)
            {
                result += a[i] * b[i];
            }

            return result;

#else
            // Scalar fallback
            float result = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                result += a[i] * b[i];
            }
            return result;
#endif
        }

#if defined(__AVX512VNNI__)
        /**
         * @brief VNNI-optimized block dot product: A (FP32) · B_block (quantized int8)
         *
         * Uses AVX512 VNNI instructions to compute dot products directly on int8 data.
         *
         * **Performance Note**: In practice, VNNI shows similar performance to FP32 path due to:
         * 1. A quantization overhead (FP32→int8: find max, scale, clamp) per block
         * 2. Correction loop overhead (even when vectorized)
         * 3. Descaling overhead (int32→FP32 conversion)
         *
         * VNNI would show 2× advantage if:
         * - A was pre-quantized (e.g., int8 activations in quantized models)
         * - Memory bandwidth limited (but cache-tiling keeps us compute-bound)
         * - Larger blocks amortized quantization cost
         *
         * Currently kept as proof-of-concept. Real gains require end-to-end int8 inference.
         *
         * @param A_row Input activation row (FP32, length k)

         * @param block Quantized IQ4_NL block (32 int4 values + FP16 scale)
         * @param k_offset Offset in A_row where this block starts
         * @return Dot product of A_row[k_offset:k_offset+32] · decoded(block)
         *
         * Algorithm:
         *  1. Decode 4-bit indices to int8 LUT values (kvalues_iq4nl: -127 to 113)
         *  2. Convert signed int8 → unsigned uint8 by adding 128 offset
         *  3. Quantize A_row to uint8 with adaptive scaling (centered at 128)
         *  4. Use _mm512_dpbusd_epi32 for uint8×uint8 → int32 accumulation
         *  5. Correct for the 128 offset in both operands
         *  6. Convert int32 result to FP32 and apply scales
         *
         * Offset Correction Math:
         *  Let A_orig[i] = original FP32 values, B_orig[i] = LUT int8 values
         *  Quantization: A_q[i] = A_orig[i] * scale_a + 128
         *                B_q[i] = B_orig[i] + 128
         *
         *  VNNI computes: sum_i(A_q[i] * B_q[i])
         *               = sum_i((A*scale_a + 128) * (B + 128))
         *               = sum_i(A*scale_a * B + 128*A*scale_a + 128*B + 128²)
         *               = scale_a * sum(A*B) + 128*scale_a*sum(A) + 128*sum(B) + 32*128²
         *
         *  Solving for sum(A*B):
         *  sum(A*B) = (vnni_result - 128*sum(A_q) - 128*sum(B_q) + 32*128²) / scale_a
         *
         * Performance: ~1.5-2× faster than FP32 path on Ice Lake+ CPUs
         */
        static inline float dot_product_block_vnni_prequantized(const int8_t *A_int8, float A_scale, const IQ4_NLBlock &block)
        {
            // Decode B quantized values to int8
            alignas(64) int8_t B_int8[32];

            // Unpack 4-bit indices and lookup int8 values
            __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));
            __m128i low_mask = _mm_set1_epi8(0x0F);
            __m128i low_idx = _mm_and_si128(qs, low_mask);
            __m128i high_shift = _mm_srli_epi16(qs, 4);
            __m128i high_idx = _mm_and_si128(high_shift, low_mask);

            // LUT lookup for int8 values (kvalues_iq4nl: -127 to 113)
            __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
            __m128i low_vals_i8 = _mm_shuffle_epi8(lut, low_idx);
            __m128i high_vals_i8 = _mm_shuffle_epi8(lut, high_idx);

            // Store int8 values (B stays signed for VNNI)
            _mm_store_si128(reinterpret_cast<__m128i *>(B_int8), low_vals_i8);
            _mm_store_si128(reinterpret_cast<__m128i *>(B_int8 + 16), high_vals_i8);

            // A is already quantized - just reinterpret as uint8 for VNNI
            const uint8_t *A_uint8 = reinterpret_cast<const uint8_t *>(A_int8);

            // VNNI: uint8 × int8 → int32 accumulation
            // Note: _mm512_dpbusd_epi32 expects A=unsigned, B=signed!
            __m512i acc = _mm512_setzero_si512();

            // Load A as uint8, B as int8
            __m512i a_512 = _mm512_setzero_si512();
            __m512i b_512 = _mm512_setzero_si512();

            // Load the 32 bytes into lower 256 bits of the 512-bit registers
            __mmask64 mask = 0xFFFFFFFF;                          // First 32 bytes
            a_512 = _mm512_mask_loadu_epi8(a_512, mask, A_uint8); // Unsigned A
            b_512 = _mm512_mask_loadu_epi8(b_512, mask, B_int8);  // Signed B

            // Perform VNNI dot product: uint8×int8 → int32
            // This processes 32 elements as 8 groups of 4-element dot products
            acc = _mm512_dpbusd_epi32(acc, a_512, b_512);

            // Horizontal sum of int32 accumulators
            // dpbusd produces 16 int32 outputs in acc, but only first 8 are valid (32 bytes / 4 bytes per group)
            // Extract and sum only the first 8 accumulators
            alignas(64) int32_t acc_array[16];
            _mm512_store_si512(reinterpret_cast<__m512i *>(acc_array), acc);

            int32_t vnni_result = 0;
            for (int i = 0; i < 8; ++i)
            { // Only sum first 8 valid accumulators
                vnni_result += acc_array[i];
            }

            //  VNNI computes unsigned×signed multiply:
            //    uint(A_i8) × B_i8 where uint(x) = x if x≥0 else 256+x
            //
            //  Since A can be negative, we need to correct:
            //    When A_i8 < 0: uint(A_i8) = 256 + A_i8
            //    VNNI gives: (256 + A_i8) * B_i8 = 256*B_i8 + A_i8*B_i8
            //    We want: A_i8 * B_i8
            //    Correction: subtract 256*B_i8 for each negative A_i8
            //
            //  Correction formula:
            //    corrected = vnni_result - 256 * sum(B_i8[i] where A_i8[i] < 0)

            // Vectorized correction for negative A values using AVX512
            // Load A and B as int8 vectors
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(A_int8));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(B_int8));

            // Create mask for negative A values (A_int8[i] < 0)
            __m256i zero = _mm256_setzero_si256();
            __mmask32 neg_mask = _mm256_cmplt_epi8_mask(a_vec, zero);

            // Mask B values where A is negative, zero out rest
            __m256i b_masked = _mm256_maskz_mov_epi8(neg_mask, b_vec);

            // Sign-extend int8 to int32 for accumulation (process in 8-element chunks)
            // Extract 8 bytes at a time and convert to int32
            __m128i b_low = _mm256_castsi256_si128(b_masked);       // Lower 16 bytes
            __m128i b_high = _mm256_extracti128_si256(b_masked, 1); // Upper 16 bytes

            // Convert int8 -> int32 (sign-extended)
            __m256i b_int32_0 = _mm256_cvtepi8_epi32(b_low); // Bytes 0-7
            __m128i b_mid = _mm_bsrli_si128(b_low, 8);
            __m256i b_int32_1 = _mm256_cvtepi8_epi32(b_mid);  // Bytes 8-15
            __m256i b_int32_2 = _mm256_cvtepi8_epi32(b_high); // Bytes 16-23
            __m128i b_top = _mm_bsrli_si128(b_high, 8);
            __m256i b_int32_3 = _mm256_cvtepi8_epi32(b_top); // Bytes 24-31

            // Sum all int32 values
            __m256i sum_01 = _mm256_add_epi32(b_int32_0, b_int32_1);
            __m256i sum_23 = _mm256_add_epi32(b_int32_2, b_int32_3);
            __m256i sum_all = _mm256_add_epi32(sum_01, sum_23);

            // Horizontal sum of 8 int32 values (manual reduction)
            __m128i sum_low = _mm256_castsi256_si128(sum_all);
            __m128i sum_high = _mm256_extracti128_si256(sum_all, 1);
            __m128i sum_128 = _mm_add_epi32(sum_low, sum_high);
            sum_128 = _mm_hadd_epi32(sum_128, sum_128);
            sum_128 = _mm_hadd_epi32(sum_128, sum_128);
            int32_t b_sum_where_a_neg = _mm_cvtsi128_si32(sum_128);

            // Apply correction: subtract 256 * sum(B where A<0)
            int32_t correction = -256 * b_sum_where_a_neg;
            int32_t corrected_dot = vnni_result + correction;

            // Convert to FP32 and descale
            float b_scale = simd::fp16_to_fp32(block.d);
            float result_fp32 = (static_cast<float>(corrected_dot) / A_scale) * b_scale;

            return result_fp32;
        }
#endif // __AVX512VNNI__

        bool supports(int m, int n, int k) const override
        {
            // Adaptive support based on operation characteristics

            // Skip very small ops where overhead dominates (< 1K elements total)
            size_t total_output = static_cast<size_t>(m) * n;
            if (total_output < 1024)
            {
                return false; // Too small, use BLAS directly
            }

            // Support all batch sizes now that we have cache-tiled implementation
            // - m=1: Single token decode (row-wise)
            // - m∈[2,16]: Small batch (cache-blocked, 32-elem blocks)
            // - m>16: Large batch (cache-tiled, 64×64 tiles)
            return true;
        }

        const char *name() const override
        {
            return "IQ4_NL_FusedGemm";
        }
    };

    // Implementation of createGemmRaw() method
    inline ITensorGemm *IQ4_NLTensor::createGemmRaw() const
    {
        return new IQ4_NLQuantizedGemm(this);
    }

} // namespace llaminar
