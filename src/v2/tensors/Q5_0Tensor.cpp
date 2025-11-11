#include "../utils/Logger.h"
/**
 * @file Q5_0Tensor.cpp
 * @brief Q5_0 quantized tensor implementation (5-bit uniform quantization)
 * @author David Sanftenberg
 * @date October 29, 2025
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm/GemmAutoTuner.h"
#include "../utils/DebugEnv.h"
#include "../utils/CPUFeatures.h"
#include "FP16Utils.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#include "SIMDHelpers.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    Q5_0Tensor::Q5_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape),
          is_view_(false),
          raw_data_(raw_data),
          raw_data_ptr_(nullptr),
          view_byte_offset_(0),
          parent_(nullptr),
          device_idx_(-1),
          device_blocks_(nullptr)
    {
        if (shape_.size() != 2)
        {
            throw std::invalid_argument("Q5_0Tensor requires 2D shape");
        }

        // Validate block alignment
        const size_t num_elements = shape_[0] * shape_[1];
        const size_t num_blocks = (num_elements + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
        const size_t expected_bytes = num_blocks * sizeof(Q5_0Block);

        if (raw_data_.size() != expected_bytes)
        {
            throw std::invalid_argument("Q5_0Tensor: raw_data size mismatch (expected " +
                                        std::to_string(expected_bytes) + ", got " +
                                        std::to_string(raw_data_.size()) + ")");
        }
    }

    Q5_0Tensor::Q5_0Tensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<TensorBase> parent)
        : shape_(shape),
          is_view_(true),
          raw_data_(),
          raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset),
          parent_(parent),
          device_idx_(-1),
          device_blocks_(nullptr)
    {
        if (shape_.size() != 2)
        {
            throw std::invalid_argument("Q5_0Tensor view requires 2D shape");
        }
    }

    Q5_0Tensor::~Q5_0Tensor()
    {
        // Device cleanup handled elsewhere
    }

    bool Q5_0Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q5_0Tensor::data() const
    {
        // Dequantize to cache
        const size_t num_elements = shape_[0] * shape_[1];
        dequant_cache_.resize(num_elements);

        const size_t num_blocks = (num_elements + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_0Block *blocks = reinterpret_cast<const Q5_0Block *>(data_ptr);

        for (size_t i = 0; i < num_blocks; ++i)
        {
            decodeBlock(blocks[i], &dequant_cache_[i * Q5_0Block::BLOCK_SIZE]);
        }

        return dequant_cache_.data();
    }

    float *Q5_0Tensor::mutable_data()
    {
        throw std::runtime_error("Q5_0Tensor is read-only (quantized format)");
    }

    bool Q5_0Tensor::copyFrom(const TensorBase *src)
    {
        throw std::runtime_error("Q5_0Tensor::copyFrom not implemented (read-only)");
    }

    std::unique_ptr<ITensorGemm> Q5_0Tensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    std::shared_ptr<TensorBase> Q5_0Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q5_0Tensor view requires 2D shape");
        }

        // Row-slice only: offset must be row-aligned
        if (offset % shape_[1] != 0)
        {
            throw std::invalid_argument("Q5_0Tensor view offset must be row-aligned");
        }

        const size_t row_offset = offset / shape_[1];
        const size_t blocks_per_row = (shape_[1] + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;
        const size_t byte_offset = row_offset * blocks_per_row * sizeof(Q5_0Block);

        const uint8_t *parent_data = is_view_ ? raw_data_ptr_ : raw_data_.data();
        std::shared_ptr<TensorBase> parent = is_view_ ? parent_ : std::static_pointer_cast<TensorBase>(shared_from_this());

        return std::shared_ptr<Q5_0Tensor>(new Q5_0Tensor(new_shape, parent_data, byte_offset, parent));
    }

    void Q5_0Tensor::decodeBlock(const Q5_0Block &block, float *output)
    {
        // Check debug environment for forced SIMD path
        const auto &env = debugEnv();

        if (env.dequant.simd_path == "scalar")
        {
            decodeBlockScalar(block, output);
            return;
        }

#if defined(__AVX512F__)
        if (env.dequant.simd_path == "avx512" ||
            (env.dequant.simd_path == "auto" && cpu_supports_avx512()))
        {
            decodeBlockAVX512(block, output);
            return;
        }
#endif

#if defined(__AVX2__)
        if (env.dequant.simd_path == "avx2" ||
            (env.dequant.simd_path == "auto" && cpu_supports_avx2()))
        {
            decodeBlockAVX2(block, output);
            return;
        }
#endif

        // Final fallback to scalar
        decodeBlockScalar(block, output);
    }

    void Q5_0Tensor::decodeBlockScalar(const Q5_0Block &block, float *output)
    {
        // Reference: llama.cpp ggml-quants.c dequantize_row_q5_0()
        // Block: 32 elements, FP16 scale, 4-byte qh (high bits), 16-byte qs (low 4 bits)

        const float d = fp16_to_fp32(block.d);

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Decode 32 elements (2 per iteration)
        for (int j = 0; j < 16; ++j)
        {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10; // Extract bit j from qh
            const uint8_t xh_1 = ((qh >> (j + 12)) & 0x10);     // Extract bit j+12 from qh

            const int32_t x0 = ((block.qs[j] & 0x0F) | xh_0) - 16; // Lower nibble + high bit - 16
            const int32_t x1 = ((block.qs[j] >> 4) | xh_1) - 16;   // Upper nibble + high bit - 16

            output[j + 0] = x0 * d;
            output[j + 16] = x1 * d;
        }
    }

#if defined(__AVX512F__)
    void Q5_0Tensor::decodeBlockAVX512(const Q5_0Block &block, float *output)
    {
        // AVX512 implementation for Q5_0: 32 values with 5-bit precision
        const float d = fp16_to_fp32(block.d);
        const __m512 vscale = _mm512_set1_ps(d);
        const __m512i voffset = _mm512_set1_epi32(16); // Q5_0 uses -16 bias

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Load 16 bytes containing 32 4-bit low values
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Extract high bits from qh and combine with low nibbles
        // First 16 elements: bits 0-15 of qh
        __m512i vi32_low = _mm512_cvtepu8_epi32(vlow);
        uint16_t qh_low = qh & 0xFFFF;
        __m512i vhigh_bits_low = _mm512_set_epi32(
            (qh_low >> 15) & 1, (qh_low >> 14) & 1, (qh_low >> 13) & 1, (qh_low >> 12) & 1,
            (qh_low >> 11) & 1, (qh_low >> 10) & 1, (qh_low >> 9) & 1, (qh_low >> 8) & 1,
            (qh_low >> 7) & 1, (qh_low >> 6) & 1, (qh_low >> 5) & 1, (qh_low >> 4) & 1,
            (qh_low >> 3) & 1, (qh_low >> 2) & 1, (qh_low >> 1) & 1, (qh_low >> 0) & 1);
        vhigh_bits_low = _mm512_slli_epi32(vhigh_bits_low, 4); // Shift to bit 4 position
        vi32_low = _mm512_or_si512(vi32_low, vhigh_bits_low);
        vi32_low = _mm512_sub_epi32(vi32_low, voffset); // Subtract 16 bias
        __m512 vf_low = _mm512_cvtepi32_ps(vi32_low);
        vf_low = _mm512_mul_ps(vf_low, vscale);
        _mm512_storeu_ps(&output[0], vf_low);

        // Second 16 elements: bits 16-31 of qh
        __m512i vi32_high = _mm512_cvtepu8_epi32(vhigh);
        uint16_t qh_high = (qh >> 16) & 0xFFFF;
        __m512i vhigh_bits_high = _mm512_set_epi32(
            (qh_high >> 15) & 1, (qh_high >> 14) & 1, (qh_high >> 13) & 1, (qh_high >> 12) & 1,
            (qh_high >> 11) & 1, (qh_high >> 10) & 1, (qh_high >> 9) & 1, (qh_high >> 8) & 1,
            (qh_high >> 7) & 1, (qh_high >> 6) & 1, (qh_high >> 5) & 1, (qh_high >> 4) & 1,
            (qh_high >> 3) & 1, (qh_high >> 2) & 1, (qh_high >> 1) & 1, (qh_high >> 0) & 1);
        vhigh_bits_high = _mm512_slli_epi32(vhigh_bits_high, 4);
        vi32_high = _mm512_or_si512(vi32_high, vhigh_bits_high);
        vi32_high = _mm512_sub_epi32(vi32_high, voffset);
        __m512 vf_high = _mm512_cvtepi32_ps(vi32_high);
        vf_high = _mm512_mul_ps(vf_high, vscale);
        _mm512_storeu_ps(&output[16], vf_high);
    }
#endif

#if defined(__AVX2__)
    void Q5_0Tensor::decodeBlockAVX2(const Q5_0Block &block, float *output)
    {
        // AVX2 implementation for Q5_0: process in 8-element chunks
        const float d = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(d);
        const __m256i voffset = _mm256_set1_epi32(16);

        uint32_t qh;
        std::memcpy(&qh, block.qh, sizeof(qh));

        // Load 16 bytes containing 32 4-bit low values
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Process first 8 low nibbles (output[0-7])
        __m256i vi32_low0 = _mm256_cvtepu8_epi32(vlow);
        uint8_t qh_bits_low0 = qh & 0xFF;
        __m256i vhigh_bits_low0 = _mm256_set_epi32(
            (qh_bits_low0 >> 7) & 1, (qh_bits_low0 >> 6) & 1,
            (qh_bits_low0 >> 5) & 1, (qh_bits_low0 >> 4) & 1,
            (qh_bits_low0 >> 3) & 1, (qh_bits_low0 >> 2) & 1,
            (qh_bits_low0 >> 1) & 1, (qh_bits_low0 >> 0) & 1);
        vhigh_bits_low0 = _mm256_slli_epi32(vhigh_bits_low0, 4);
        vi32_low0 = _mm256_or_si256(vi32_low0, vhigh_bits_low0);
        vi32_low0 = _mm256_sub_epi32(vi32_low0, voffset);
        __m256 vf_low0 = _mm256_cvtepi32_ps(vi32_low0);
        vf_low0 = _mm256_mul_ps(vf_low0, vscale);
        _mm256_storeu_ps(&output[0], vf_low0);

        // Process second 8 low nibbles (output[8-15])
        __m256i vi32_low1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vlow, 8));
        uint8_t qh_bits_low1 = (qh >> 8) & 0xFF;
        __m256i vhigh_bits_low1 = _mm256_set_epi32(
            (qh_bits_low1 >> 7) & 1, (qh_bits_low1 >> 6) & 1,
            (qh_bits_low1 >> 5) & 1, (qh_bits_low1 >> 4) & 1,
            (qh_bits_low1 >> 3) & 1, (qh_bits_low1 >> 2) & 1,
            (qh_bits_low1 >> 1) & 1, (qh_bits_low1 >> 0) & 1);
        vhigh_bits_low1 = _mm256_slli_epi32(vhigh_bits_low1, 4);
        vi32_low1 = _mm256_or_si256(vi32_low1, vhigh_bits_low1);
        vi32_low1 = _mm256_sub_epi32(vi32_low1, voffset);
        __m256 vf_low1 = _mm256_cvtepi32_ps(vi32_low1);
        vf_low1 = _mm256_mul_ps(vf_low1, vscale);
        _mm256_storeu_ps(&output[8], vf_low1);

        // Process first 8 high nibbles (output[16-23])
        __m256i vi32_high0 = _mm256_cvtepu8_epi32(vhigh);
        uint8_t qh_bits_high0 = (qh >> 16) & 0xFF;
        __m256i vhigh_bits_high0 = _mm256_set_epi32(
            (qh_bits_high0 >> 7) & 1, (qh_bits_high0 >> 6) & 1,
            (qh_bits_high0 >> 5) & 1, (qh_bits_high0 >> 4) & 1,
            (qh_bits_high0 >> 3) & 1, (qh_bits_high0 >> 2) & 1,
            (qh_bits_high0 >> 1) & 1, (qh_bits_high0 >> 0) & 1);
        vhigh_bits_high0 = _mm256_slli_epi32(vhigh_bits_high0, 4);
        vi32_high0 = _mm256_or_si256(vi32_high0, vhigh_bits_high0);
        vi32_high0 = _mm256_sub_epi32(vi32_high0, voffset);
        __m256 vf_high0 = _mm256_cvtepi32_ps(vi32_high0);
        vf_high0 = _mm256_mul_ps(vf_high0, vscale);
        _mm256_storeu_ps(&output[16], vf_high0);

        // Process second 8 high nibbles (output[24-31])
        __m256i vi32_high1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vhigh, 8));
        uint8_t qh_bits_high1 = (qh >> 24) & 0xFF;
        __m256i vhigh_bits_high1 = _mm256_set_epi32(
            (qh_bits_high1 >> 7) & 1, (qh_bits_high1 >> 6) & 1,
            (qh_bits_high1 >> 5) & 1, (qh_bits_high1 >> 4) & 1,
            (qh_bits_high1 >> 3) & 1, (qh_bits_high1 >> 2) & 1,
            (qh_bits_high1 >> 1) & 1, (qh_bits_high1 >> 0) & 1);
        vhigh_bits_high1 = _mm256_slli_epi32(vhigh_bits_high1, 4);
        vi32_high1 = _mm256_or_si256(vi32_high1, vhigh_bits_high1);
        vi32_high1 = _mm256_sub_epi32(vi32_high1, voffset);
        __m256 vf_high1 = _mm256_cvtepi32_ps(vi32_high1);
        vf_high1 = _mm256_mul_ps(vf_high1, vscale);
        _mm256_storeu_ps(&output[24], vf_high1);
    }
#endif

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q5_0Tensor::to_bf16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to BF16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(temp_fp32[i]);
        }
    }

    void Q5_0Tensor::to_fp16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to FP16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(temp_fp32[i]);
        }
    }

    void Q5_0Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Decode to FP32 first, then quantize to int8
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = temp_fp32[offset + i] * scale;
                const float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst_int8[offset + i] = static_cast<int8_t>(std::round(clamped));
            }

            // Zero-fill partial block tail (if any)
            for (size_t i = count; i < block_size; ++i)
            {
                dst_int8[offset + i] = 0;
            }
        }
    }

    void Q5_0Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_row() requires 2D tensor");
        }
        if (row_idx >= shp[0])
        {
            throw std::out_of_range("Row index out of bounds");
        }

        const size_t cols = shp[1];
        const size_t blocks_per_row = (cols + block_size() - 1) / block_size();

        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            const size_t offset = kb * block_size();
            const size_t count = std::min(block_size(), cols - offset);

            float temp[256]; // Max block size
            decode_block_at(row_idx, kb, temp);

            for (size_t i = 0; i < count; ++i)
            {
                buffer[offset + i] = temp[i];
            }
        }
    }

    void Q5_0Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        // Decode full tensor (inefficient but simple)
        std::vector<float> temp_fp32(element_count());
        to_fp32(temp_fp32.data());
        std::memcpy(buffer, temp_fp32.data() + offset, count * sizeof(float));
    }

    void Q5_0Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q5_0Tensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q5_0Tensor::decode_to_q8_0: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q5_0Tensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q5_0Block::BLOCK_SIZE - 1) / Q5_0Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q5_0Tensor::decode_to_q8_0: block offset out of bounds");
        }

        // Get Q5_0 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_0Block *blocks = reinterpret_cast<const Q5_0Block *>(data_ptr);
        const Q5_0Block &q5_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Use SIMD helper to decode Q5_0 → Q8_0
        uint16_t q8_scale_fp16;
        simd::decode_q5_0_to_q8_0(
            q5_block.qs,     // Q5 lower 4 bits
            q5_block.qh,     // Q5 high bits
            q5_block.d,      // Q5 scale (FP16)
            output->qs,      // Q8 output values
            &q8_scale_fp16); // Q8 output scale (FP16)

        output->d = q8_scale_fp16;
    }

} // namespace llaminar2
