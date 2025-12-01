/**
 * @file IQ4_NLTensor.h
 * @brief Implementation of the IQ4_NL (Non‑Linear 4‑bit) quantized tensor format.
 *
 * IQ4_NL FORMAT SUMMARY
 * ---------------------
 *  Block size .......... 32 elements
 *  Bytes per block ..... 18 (2 bytes FP16 scale + 16 bytes packed 4‑bit indices)
 *  Bits per value ...... 4.5 (effective)
 *  Compression ratio ... ~7.1× vs FP32
 *  Lookup table ........ kvalues_iq4nl[16] (non‑linear int8 distribution in [-127, 113])
 *
 * Block Layout (18 bytes):
 *   struct IQ4_NLBlock {
 *       uint16_t d;       // FP16 scale factor
 *       uint8_t  qs[16];  // Packed 4-bit indices (2 per byte: low/high nibble)
 *   };
 *
 * Decode (Conceptual):
 *   for each byte b in qs:
 *       low  = b & 0x0F;          // index 0..15
 *       high = b >> 4;            // index 0..15
 *       out[j]     = fp16_to_fp32(d) * kvalues_iq4nl[low];
 *       out[j+16]  = fp16_to_fp32(d) * kvalues_iq4nl[high];
 *
 * @author David Sanftenberg
 * @date 2025-10-22 (cleanup/documentation pass)
 */

#pragma once

#include <vector>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <algorithm>

// V2 utilities (minimal portable implementations)
// #include "QuantTypes.h"  // Not needed
#include "TensorKernels.h"
#include "FP16Utils.h"
#include "../utils/CPUFeatures.h"
#include "IQQuantTables.h"
#include "../utils/DebugEnv.h"
#include "SIMDHelpers.h"
#include "AlignedVector.h"
#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

// Optional SIMD intrinsics (detected at runtime via CPUFeatures)
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief IQ4_NL block structure (exactly 18 bytes) representing 32 quantized elements.
     *
     * Layout mirrors GGML's block_iq4_nl. Two 4‑bit indices per byte in @p qs select entries
     * in kvalues_iq4nl, scaled by FP16 value @p d.
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
     * This is a v2 port focusing on core tensor structure and fused GEMM operations.
     */
    class IQ4_NLTensor : public TensorBase, public ITensorGemmTileDataProvider, public IQ8_0Decodable, public IINT8Unpackable
    {
    public:
        /**
         * @brief Construct tensor from a 2D shape and contiguous IQ4_NL block storage.
         *
         * @param shape 2D tensor dimensions: [rows, cols].
         * @param raw_data Raw block bytes (row-major blocks: each row padded to 32).
         * @throws std::invalid_argument If shape rank != 2 or raw size mismatches expected block count.
         */
        IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
            : shape_(shape), raw_data_(raw_data), device_idx_(-1), is_view_(false), raw_data_ptr_(nullptr), view_byte_offset_(0), parent_(nullptr)
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

        const std::vector<size_t> &shape() const { return shape_; }
        size_t size() const { return shape_[0] * shape_[1]; }
        size_t ndim() const { return 2; }

        float compression_ratio() const { return 7.1f; }

        /** @brief Logical (unpadded) column count (K dimension). */
        size_t logical_k() const { return shape_[1]; }

        /**
         * @brief Physical padded column count (multiple of 32).
         * @details Fused kernels iterate over [0, padded_k()) and safely process tail via min() when
         *          determining valid elements in the final block.
         */
        size_t padded_k() const
        {
            size_t cols = logical_k();
            return ((cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE) * IQ4_NLBlock::BLOCK_SIZE;
        }

        size_t element_count() const
        {
            return shape_[0] * shape_[1];
        }

        // ========== Raw data access ==========

        /** @brief Direct access to quantized blocks (for fused kernels) */
        const uint8_t *raw_blocks() const { return raw_data_.data(); }

        /** @brief Number of 18-byte blocks */
        size_t num_blocks() const { return raw_data_.size() / sizeof(IQ4_NLBlock); }

        // ========== Decode API ==========

        /**
         * @brief Fully decode tensor to a FP32 destination buffer.
         * @param dst Pointer to output buffer with capacity rows*cols floats.
         *
         * Production path: Parallel row-by-row decode. Each row is decoded independently
         * with OpenMP parallelization when rows > 4.
         */
        void decode_to_fp32(float *dst) const
        {
            const size_t rows = shape_[0];
            const size_t cols = shape_[1];
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const auto &env = debugEnv();

            // Experimental microkernel (disabled by default - enable via LLAMINAR_IQ4_MICROKERNEL=1)
            if (env.dequant.iq4_microkernel)
            {
                decode_to_fp32_microkernel(dst, blocks, rows, cols, blocks_per_row);
                return;
            }

// PRODUCTION PATH: Row-level parallelization for improved cache locality
#pragma omp parallel for schedule(static) if (rows > 4)
            for (size_t row = 0; row < rows; ++row)
            {
                const size_t row_block_base = row * blocks_per_row;
                float *row_out = dst + row * cols;

                // Decode blocks, handling tail block specially
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t global_block_index = row_block_base + b;
                    size_t block_start_col = b * IQ4_NLBlock::BLOCK_SIZE;
                    size_t elements_in_block = std::min(
                        IQ4_NLBlock::BLOCK_SIZE,
                        cols - block_start_col);

                    // Decode to temporary buffer (always 32 elements)
                    float temp[IQ4_NLBlock::BLOCK_SIZE];
                    decodeBlock(blocks[global_block_index], temp);

                    // Copy only the valid elements to output
                    std::memcpy(row_out + block_start_col, temp, elements_in_block * sizeof(float));
                }
            }
        }

        // decode_to_bf16() - Implementation in IQ4_NLTensor.cpp

        // copy() - Commented out (TensorBase interface not used in v2)
        // IQ4_NLTensor is value-semantic, use copy constructor if needed

        // copy_from() - Commented out (TensorBase interface not used in v2)

        // ========== Streaming Decode API ==========

        /**
         * @brief Decode a single row to FP32.
         * @param row_idx Row index in [0, rows).
         * @param buffer Output buffer with capacity = cols.
         */
        void decodeRow(size_t row_idx, float *buffer) const
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

        /**
         * @brief Decode an arbitrary contiguous span of elements (flattened indexing).
         * @param offset Starting element offset.
         * @param count Number of elements to decode.
         * @param buffer Output buffer (count floats).
         * @throws std::out_of_range if span exceeds tensor bounds.
         */
        void decodeSpan(size_t offset, size_t count, float *buffer) const
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

        /** @brief Raw underlying quantized byte storage. */
        const uint8_t *raw_data() const
        {
            return raw_data_.data();
        }

        /** @brief Size in bytes of raw quantized storage. */
        size_t raw_size() const
        {
            return raw_data_.size();
        }

        // block_descriptor() - Commented out (QuantBlockDescriptor not defined in v2)
        // Block layout info: 32 elements/block, 18 bytes/block, 4.5 bits/value

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
        /**
         * @brief Decode one 32‑element block at (row_idx, k_block_offset) to FP32.
         * @param row_idx Row index.
         * @param k_block_offset Block offset along K (0‑based, units of 32).
         * @param output Destination buffer (32 floats).
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
        /**
         * @brief Direct const access to a quantized block (no decode).
         * @param row_idx Row index.
         * @param k_block_offset Block offset along K.
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
        /**
         * @brief Decode a consecutive tile of rows (tile_n) for a single K block offset.
         * @param row_start First row.
         * @param tile_n Number of rows to decode.
         * @param k_block_offset Block offset along K.
         * @param output Output buffer sized tile_n*32.
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

        // ========== IINT8Unpackable Implementation ==========

        void unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const override;
        float get_block_scale(size_t row_idx, size_t k_block_offset) const override;
        float get_block_min(size_t row_idx, size_t k_block_offset) const override;

        /**
         * @brief Create fused quantized GEMM implementation
         */
        std::unique_ptr<ITensorGemm> createGemm();
        ITensorGemm *createGemmRaw();

    private:
        std::vector<size_t> shape_;       ///< Tensor dimensions (2D: [rows, cols])
        AlignedVector<uint8_t> raw_data_; ///< Raw quantized data (IQ4_NL blocks) - 64-byte aligned for SIMD

#if defined(__AVX512F__)
        /**
         * @brief AVX512-optimized IQ4_NL block decode
         *
         * Uses SIMD helper library for efficient int8 to float32 conversion.
         * Processes 16 values at a time with AVX512 intrinsics.
         */
        /**
         * @brief AVX512 helper: decode one block using a staging int8 buffer then wide convert.
         * @note Called only if CPU feature probe succeeds.
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
        /**
         * @brief AVX2 helper: decode one block using int8 staging buffer then convert in 8‑wide chunks.
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
        /**
         * @brief Generic block decode dispatch (direct / AVX512 / AVX2 / scalar).
         * @param block Source quantized block.
         * @param output Destination FP32 (32 floats).
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
        /**
         * @brief Experimental multi‑block microkernel (DISABLED by default).
         *
         * **Status**: NOT used in production benchmarks (requires LLAMINAR_IQ4_MICROKERNEL=1).
         *
         * Processes multiple blocks per iteration using AVX512/AVX2 nibble expansion to reduce
         * function call overhead. Retained for research but the standard per-block decode path
         * (used by default) has proven sufficient for current workloads.
         *
         * @warning Not the active code path - the production benchmarks use the standard row-parallel
         *          decode loop in `decode_to_fp32()`.
         */
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
        /** @brief Microkernel helper: AVX2 nibble expansion + staged conversion. */
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
        /** @brief Microkernel helper: AVX512 nibble expansion + staged conversion. */
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
            for (int i = 0; i < cols; ++i)
            {
                buffer[i] = bfloat16(temp[i]);
            }
        }
        */
    };

} // namespace llaminar2
