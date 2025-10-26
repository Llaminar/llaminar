/**
 * @file IQ4_NLTensor.cpp
 * @brief IQ4_NL quantized tensor implementation
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "FP16Utils.h"
#include "../utils/CPUFeatures.h"
#include "../utils/DebugEnv.h"
#include "SIMDHelpers.h"
#include "../backends/ComputeBackend.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace llaminar2
{

    // ========== Constructor & Destructor ==========

    IQ4_NLTensor::IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
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

    IQ4_NLTensor::~IQ4_NLTensor()
    {
        // TODO: Free device_blocks_ if allocated
        if (device_blocks_)
        {
            std::cerr << "[IQ4_NLTensor] TODO: Free device blocks in destructor\n";
        }
    }

    // ========== Shape and Metadata ==========

    size_t IQ4_NLTensor::padded_k() const
    {
        size_t cols = logical_k();
        return ((cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE) * IQ4_NLBlock::BLOCK_SIZE;
    }

    // ========== Device Management ==========

    bool IQ4_NLTensor::set_device(int device_idx)
    {
        // TODO: Implement device transfer for quantized tensors
        std::cerr << "[IQ4_NLTensor] set_device not yet implemented\n";
        device_idx_ = device_idx;
        return true;
    }

    // ========== Data Access ==========

    const float *IQ4_NLTensor::data() const
    {
        // Fully decode to cache
        size_t total_elements = element_count();
        if (dequant_cache_.size() != total_elements)
        {
            dequant_cache_.resize(total_elements);
        }

        decode_to_fp32(dequant_cache_.data());
        return dequant_cache_.data();
    }

    float *IQ4_NLTensor::mutable_data()
    {
        throw std::runtime_error("IQ4_NLTensor::mutable_data: quantized tensors are immutable");
    }

    // ========== Kernel Creation ==========

    std::unique_ptr<ITensorGemm> IQ4_NLTensor::createGemm()
    {
        // Return generic QuantizedGemmKernel using this tensor's IBlockDecoder interface
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    std::unique_ptr<ITensorRoPE> IQ4_NLTensor::createRoPE()
    {
        throw std::runtime_error("IQ4_NLTensor::createRoPE: not supported for quantized tensors");
    }

    std::unique_ptr<ITensorSwiGLU> IQ4_NLTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ4_NLTensor::createSwiGLU: not supported for quantized tensors");
    }

    std::unique_ptr<ITensorSoftmax> IQ4_NLTensor::createSoftmax()
    {
        throw std::runtime_error("IQ4_NLTensor::createSoftmax: not supported for quantized tensors");
    }

    std::unique_ptr<ITensorRMSNorm> IQ4_NLTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ4_NLTensor::createRMSNorm: not supported for quantized tensors");
    }

    // ========== Decode API ==========

    void IQ4_NLTensor::decode_to_fp32(float *dst) const
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

    void IQ4_NLTensor::decode_to_bf16(uint16_t *dst) const
    {
        const size_t rows = shape_[0];
        const size_t cols = shape_[1];
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
        const size_t blocks_per_row = (cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;

        // Decode to FP32 first, then convert to BF16
        // This is efficient since FP32->BF16 is just a truncation with rounding
#pragma omp parallel for schedule(static) if (rows > 4)
        for (size_t row = 0; row < rows; ++row)
        {
            const size_t row_block_base = row * blocks_per_row;
            uint16_t *row_out = dst + row * cols;

            // Decode blocks for this row
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const size_t global_block_index = row_block_base + b;
                size_t block_start_col = b * IQ4_NLBlock::BLOCK_SIZE;
                size_t elements_in_block = std::min(
                    IQ4_NLBlock::BLOCK_SIZE,
                    cols - block_start_col);

                // Decode block to FP32 temporary buffer
                float temp_fp32[IQ4_NLBlock::BLOCK_SIZE];
                decodeBlock(blocks[global_block_index], temp_fp32);

                // Convert FP32 to BF16
                for (size_t i = 0; i < elements_in_block; ++i)
                {
                    row_out[block_start_col + i] = simd::fp32_to_bf16(temp_fp32[i]);
                }
            }
        }
    }

    void IQ4_NLTensor::decodeRow(size_t row_idx, float *buffer) const
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

    void IQ4_NLTensor::decodeSpan(size_t offset, size_t count, float *buffer) const
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

    // ========== Fused Kernel Helpers ==========

    const IQ4_NLBlock &IQ4_NLTensor::get_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
        const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
        return blocks[block_idx];
    }

    void IQ4_NLTensor::decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());

        for (size_t r = 0; r < tile_n; ++r)
        {
            size_t row_idx = row_start + r;
            size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            float *row_output = output + r * IQ4_NLBlock::BLOCK_SIZE;
            decodeBlock(blocks[block_idx], row_output);
        }
    }

    // ========== Private Decode Helpers ==========

    void IQ4_NLTensor::decodeBlock(const IQ4_NLBlock &block, float *output)
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

#if defined(__AVX512F__)
    void IQ4_NLTensor::decodeBlockAVX512(const IQ4_NLBlock &block, float *output)
    {
        const float d = simd::fp16_to_fp32(block.d);
        // Use scalar fallback - kvalues_iq4nl are floats, not int8
        for (size_t j = 0; j < 16; ++j)
        {
            const uint8_t qbyte = block.qs[j];
            output[j] = d * kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
            output[j + 16] = d * kvalues_iq4nl[qbyte >> 4]; // High nibble
        }
    }
#endif

#if defined(__AVX2__)
    void IQ4_NLTensor::decodeBlockAVX2(const IQ4_NLBlock &block, float *output)
    {
        const float d = simd::fp16_to_fp32(block.d);
        // Use scalar fallback - kvalues_iq4nl are floats, not int8
        for (size_t j = 0; j < 16; ++j)
        {
            const uint8_t qbyte = block.qs[j];
            output[j] = d * kvalues_iq4nl[qbyte & 0x0F];    // Low nibble
            output[j + 16] = d * kvalues_iq4nl[qbyte >> 4]; // High nibble
        }
    }

    void IQ4_NLTensor::decodeBlockVectorizedAVX2(const IQ4_NLBlock &block, float *output)
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
    void IQ4_NLTensor::decodeBlockVectorizedAVX512(const IQ4_NLBlock &block, float *output)
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

    void IQ4_NLTensor::decode_to_fp32_microkernel(float *dst, const IQ4_NLBlock *blocks, int rows, int cols, size_t blocks_per_row)
    {
        // TODO: Implement experimental microkernel
        // For now, fall back to standard decode
        std::cerr << "[IQ4_NLTensor] Microkernel not yet implemented, using standard path\n";

#pragma omp parallel for schedule(static) if (rows > 4)
        for (int row = 0; row < rows; ++row)
        {
            const size_t row_block_base = row * blocks_per_row;
            float *row_out = dst + row * cols;

            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const size_t global_block_index = row_block_base + b;
                size_t block_start_col = b * IQ4_NLBlock::BLOCK_SIZE;
                size_t elements_in_block = std::min(
                    IQ4_NLBlock::BLOCK_SIZE,
                    static_cast<size_t>(cols) - block_start_col);

                float temp[IQ4_NLBlock::BLOCK_SIZE];
                decodeBlock(blocks[global_block_index], temp);
                std::memcpy(row_out + block_start_col, temp, elements_in_block * sizeof(float));
            }
        }
    }

    bool IQ4_NLTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ4_NLTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
