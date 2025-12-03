/**
 * @file Q4_0Tensor.cpp
 * @brief Q4_0 quantized tensor implementation (4-bit per weight, 8.0x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/KernelFactory.h"
#include "Tensors.h"
#include "../kernels/cpu/ops/CPURoPEKernelT.h"
#include "../kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../utils/CPUFeatures.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    Q4_0Tensor::Q4_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape),
          is_view_(false),
          raw_data_(raw_data),
          raw_data_ptr_(nullptr),
          view_byte_offset_(0),
          parent_(nullptr),
          device_idx_(-1),
          device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q4_0Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_0Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_0Tensor: insufficient raw data");
        }
    }

    // Private view constructor
    Q4_0Tensor::Q4_0Tensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<TensorBase> parent)
        : shape_(shape),
          is_view_(true),
          raw_data_(), // Empty (view borrows parent data)
          raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset),
          parent_(parent),
          device_idx_(-1),
          device_blocks_(nullptr)
    {
    }

    std::shared_ptr<TensorBase> Q4_0Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validation: must be 2D
        if (shape_.size() != 2 || new_shape.size() != 2)
        {
            throw std::invalid_argument("Q4_0Tensor::create_view: only 2D row-slice views supported");
        }

        // Validation: K dimension must match
        if (new_shape[1] != shape_[1])
        {
            throw std::invalid_argument("Q4_0Tensor::create_view: column count (K) must match parent");
        }

        // Validation: offset must be row-aligned
        if (offset % shape_[1] != 0)
        {
            throw std::invalid_argument("Q4_0Tensor::create_view: offset must be row-aligned");
        }

        // Validation: bounds check
        if (offset + new_shape[0] * new_shape[1] > shape_[0] * shape_[1])
        {
            throw std::out_of_range("Q4_0Tensor::create_view: view exceeds parent bounds");
        }

        // Calculate block offset
        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
        const size_t first_row = offset / cols;
        const size_t block_offset = first_row * blocks_per_row;
        const size_t byte_offset = block_offset * sizeof(Q4_0Block);

        // Get root parent data pointer
        const uint8_t *root_data_ptr;
        size_t cumulative_byte_offset;
        std::shared_ptr<TensorBase> root_parent;

        if (is_view_)
        {
            // Chain to root parent
            root_data_ptr = raw_data_ptr_;
            cumulative_byte_offset = view_byte_offset_ + byte_offset;
            root_parent = parent_; // Already points to root
        }
        else
        {
            // Create view from owned tensor
            root_data_ptr = raw_data_.data();
            cumulative_byte_offset = byte_offset;
            root_parent = std::static_pointer_cast<TensorBase>(shared_from_this());
        }

        // Create view using private constructor
        return std::shared_ptr<Q4_0Tensor>(new Q4_0Tensor(
            new_shape,
            root_data_ptr,
            cumulative_byte_offset,
            root_parent));
    }

    std::unique_ptr<ITensorGemm> Q4_0Tensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void Q4_0Tensor::decodeBlock(const Q4_0Block &block, float *output)
    {
#if defined(__AVX512F__)
        if (cpu_supports_avx512())
        {
            decodeBlockAVX512(block, output);
            return;
        }
#endif

#if defined(__AVX2__)
        if (cpu_supports_avx2())
        {
            decodeBlockAVX2(block, output);
            return;
        }
#endif

        // Scalar fallback
        decodeBlockScalar(block, output);
    }

    void Q4_0Tensor::decodeBlockScalar(const Q4_0Block &block, float *output)
    {
        // Scalar fallback: decode formula is scale * (nibble - 8)
        // Layout: low nibbles in first half [0..15], high nibbles in second half [16..31]
        const float scale = fp16_to_fp32(block.d);

        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t byte = block.qs[i];

            // Extract two 4-bit nibbles (low and high)
            const int8_t v0 = static_cast<int8_t>((byte & 0x0F) - 8);
            const int8_t v1 = static_cast<int8_t>((byte >> 4) - 8);

            // Low nibble → first half, high nibble → second half (not interleaved!)
            output[i + 0] = scale * static_cast<float>(v0);
            output[i + 16] = scale * static_cast<float>(v1);
        }
    }

#if defined(__AVX512F__)
    void Q4_0Tensor::decodeBlockAVX512(const Q4_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m512 vscale = _mm512_set1_ps(scale);
        const __m512i vminus8 = _mm512_set1_epi8(-8);
        const __m512i vmask_low = _mm512_set1_epi8(0x0F);

        // Load 16 bytes (32 nibbles)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Broadcast to 512-bit registers for parallel processing
        __m512i v4bit_512 = _mm512_broadcast_i32x4(v4bit);

        // Extract low and high nibbles
        __m512i vlow = _mm512_and_si512(v4bit_512, vmask_low);
        __m512i vhigh = _mm512_and_si512(_mm512_srli_epi16(v4bit_512, 4), vmask_low);

        // Subtract 8 (Q4_0 bias)
        vlow = _mm512_add_epi8(vlow, vminus8);
        vhigh = _mm512_add_epi8(vhigh, vminus8);

        // Convert low nibbles (first 16 elements)
        __m128i vlow_128 = _mm512_castsi512_si128(vlow);
        __m512i vi32_low = _mm512_cvtepi8_epi32(vlow_128);
        __m512 vf_low = _mm512_cvtepi32_ps(vi32_low);
        vf_low = _mm512_mul_ps(vf_low, vscale);
        _mm512_storeu_ps(output, vf_low);

        // Convert high nibbles (second 16 elements)
        __m128i vhigh_128 = _mm512_castsi512_si128(vhigh);
        __m512i vi32_high = _mm512_cvtepi8_epi32(vhigh_128);
        __m512 vf_high = _mm512_cvtepi32_ps(vi32_high);
        vf_high = _mm512_mul_ps(vf_high, vscale);
        _mm512_storeu_ps(output + 16, vf_high);
    }
#endif

#if defined(__AVX2__)
    void Q4_0Tensor::decodeBlockAVX2(const Q4_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256i vminus8 = _mm256_set1_epi8(-8);

        // Process 16 bytes → 32 nibbles → 32 float outputs
        // Layout: low nibbles [0..15], then high nibbles [16..31]
        // Load 16 bytes (contains 32 4-bit values)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Broadcast to 256-bit for parallel processing
        __m256i v4bit_256 = _mm256_set_m128i(v4bit, v4bit);

        // Extract low nibbles (mask 0x0F)
        __m256i vlow = _mm256_and_si256(v4bit_256, _mm256_set1_epi8(0x0F));

        // Extract high nibbles (shift right 4)
        __m256i vhigh = _mm256_and_si256(_mm256_srli_epi16(v4bit_256, 4), _mm256_set1_epi8(0x0F));

        // Subtract 8 from both (Q4_0 bias)
        vlow = _mm256_add_epi8(vlow, vminus8);
        vhigh = _mm256_add_epi8(vhigh, vminus8);

        // Unpack and convert to float (2 chunks of 16 elements each)
        // Chunk 0: low nibbles [0..15]
        // Chunk 1: high nibbles [16..31]
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            __m256i vsrc = (chunk == 0) ? vlow : vhigh;
            __m128i vsrc_low = _mm256_castsi256_si128(vsrc);
            __m128i vsrc_high = _mm256_extracti128_si256(vsrc, 1);

            // Convert first 8 elements
            __m256i vi32_0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(&vsrc_low)));
            __m256 vf_0 = _mm256_cvtepi32_ps(vi32_0);
            vf_0 = _mm256_mul_ps(vf_0, vscale);
            _mm256_storeu_ps(&output[chunk * 16], vf_0);

            // Convert second 8 elements
            __m256i vi32_1 = _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(vsrc_low, vsrc_low));
            __m256 vf_1 = _mm256_cvtepi32_ps(vi32_1);
            vf_1 = _mm256_mul_ps(vf_1, vscale);
            _mm256_storeu_ps(&output[chunk * 16 + 8], vf_1);
        }
    }
#endif

    Q4_0Tensor::~Q4_0Tensor() {}

    bool Q4_0Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_0Tensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_0Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_0Tensor::mutable_data()
    {
        throw std::runtime_error("Q4_0Tensor::mutable_data: quantized tensors are immutable");
    }

    bool Q4_0Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q4_0Tensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q4_0Tensor::to_bf16(uint16_t *dst) const
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

    void Q4_0Tensor::to_fp16(uint16_t *dst) const
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

    void Q4_0Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q4_0Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q4_0Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void Q4_0Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q4_0Tensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_0Tensor::decode_to_q8_0: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_0Tensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_0Tensor::decode_to_q8_0: block offset out of bounds");
        }

        // Get Q4_0 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
        const Q4_0Block &q4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Use SIMD helper to decode Q4_0 → Q8_0
        uint16_t q8_scale_fp16;
        simd::decode_q4_0_to_q8_0(
            q4_block.qs,     // Q4 packed values
            q4_block.d,      // Q4 scale (FP16)
            output->qs,      // Q8 output values
            &q8_scale_fp16); // Q8 output scale (FP16)

        output->d = q8_scale_fp16;
    }

    void Q4_0Tensor::unpack_block_to_int8(
        size_t row_idx,
        size_t k_block_offset,
        int8_t *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q4_0Tensor::unpack_block_to_int8: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_0Tensor::unpack_block_to_int8: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_0Tensor::unpack_block_to_int8: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_0Tensor::unpack_block_to_int8: block offset out of bounds");
        }

        // Get Q4_0 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
        const Q4_0Block *q4_block = &blocks[row_idx * blocks_per_row + k_block_offset];

        // Unpack Q4_0 block to INT8 using SIMD dispatcher
        simd::unpack_q4_0_to_int8(*q4_block, output);
    }

    float Q4_0Tensor::get_block_scale(
        size_t row_idx,
        size_t k_block_offset) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_0Tensor::get_block_scale: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_0Tensor::get_block_scale: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_0Tensor::get_block_scale: block offset out of bounds");
        }

        // Get Q4_0 block and return original FP16 scale converted to FP32
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_0Block *blocks = reinterpret_cast<const Q4_0Block *>(data_ptr);
        const Q4_0Block &q4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        return fp16_to_fp32(q4_block.d);
    }

} // namespace llaminar2
