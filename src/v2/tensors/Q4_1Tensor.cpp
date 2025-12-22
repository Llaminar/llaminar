/**
 * @file Q4_1Tensor.cpp
 * @brief Q4_1 quantized tensor implementation (4-bit per weight with min, ~7.1x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/KernelFactory.h"
#include "../utils/DebugEnv.h"
#include "../utils/CPUFeatures.h"
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#include "../utils/Logger.h"
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#include "../utils/Logger.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    Q4_1Tensor::Q4_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
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
            throw std::invalid_argument("Q4_1Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_1Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_1Tensor: insufficient raw data");
        }
    }

    // Private view constructor
    Q4_1Tensor::Q4_1Tensor(const std::vector<size_t> &shape,
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

    std::shared_ptr<TensorBase> Q4_1Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validation: must be 2D
        if (shape_.size() != 2 || new_shape.size() != 2)
        {
            throw std::invalid_argument("Q4_1Tensor::create_view: only 2D row-slice views supported");
        }

        // Validation: K dimension must match
        if (new_shape[1] != shape_[1])
        {
            throw std::invalid_argument("Q4_1Tensor::create_view: column count (K) must match parent");
        }

        // Validation: offset must be row-aligned
        if (offset % shape_[1] != 0)
        {
            throw std::invalid_argument("Q4_1Tensor::create_view: offset must be row-aligned");
        }

        // Validation: bounds check
        if (offset + new_shape[0] * new_shape[1] > shape_[0] * shape_[1])
        {
            throw std::out_of_range("Q4_1Tensor::create_view: view exceeds parent bounds");
        }

        // Calculate block offset
        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
        const size_t first_row = offset / cols;
        const size_t block_offset = first_row * blocks_per_row;
        const size_t byte_offset = block_offset * sizeof(Q4_1Block);

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
        return std::shared_ptr<Q4_1Tensor>(new Q4_1Tensor(
            new_shape,
            root_data_ptr,
            cumulative_byte_offset,
            root_parent));
    }

    std::unique_ptr<ITensorGemm> Q4_1Tensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void Q4_1Tensor::decodeBlock(const Q4_1Block &block, float *output)
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

        // Final fallback to scalar
        decodeBlockScalar(block, output);
    }

    void Q4_1Tensor::decodeBlockScalar(const Q4_1Block &block, float *output)
    {
        // Scalar fallback: decode formula is scale * nibble + min
        // Layout matches llama.cpp: low nibbles in first half, high nibbles in second half
        const float scale = fp16_to_fp32(block.d);
        const float min = fp16_to_fp32(block.m);

        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t byte = block.qs[i];

            // Extract two 4-bit nibbles (no bias subtraction for Q4_1)
            const int x0 = byte & 0x0F;
            const int x1 = byte >> 4;

            // Interleaved layout: low nibbles then high nibbles
            output[i] = scale * x0 + min;
            output[i + 16] = scale * x1 + min;
        }
    }

#if defined(__AVX512F__)
    void Q4_1Tensor::decodeBlockAVX512(const Q4_1Block &block, float *output)
    {
        // AVX512 implementation: process all 32 values in 2 chunks of 16
        const float scale = fp16_to_fp32(block.d);
        const float min = fp16_to_fp32(block.m);
        const __m512 vscale = _mm512_set1_ps(scale);
        const __m512 vmin = _mm512_set1_ps(min);

        // Load 16 bytes (contains 32 4-bit values)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Process low nibbles: output[0-15] - all 16 at once with AVX512
        __m512i vi32_low = _mm512_cvtepu8_epi32(vlow);
        __m512 vf_low = _mm512_cvtepi32_ps(vi32_low);
        vf_low = _mm512_fmadd_ps(vf_low, vscale, vmin); // scale * value + min
        _mm512_storeu_ps(&output[0], vf_low);

        // Process high nibbles: output[16-31] - all 16 at once with AVX512
        __m512i vi32_high = _mm512_cvtepu8_epi32(vhigh);
        __m512 vf_high = _mm512_cvtepi32_ps(vi32_high);
        vf_high = _mm512_fmadd_ps(vf_high, vscale, vmin);
        _mm512_storeu_ps(&output[16], vf_high);
    }
#endif

#if defined(__AVX2__)
    void Q4_1Tensor::decodeBlockAVX2(const Q4_1Block &block, float *output)
    {
        // Match llama.cpp interleaved layout: low nibbles [0-15], high nibbles [16-31]
        const float scale = fp16_to_fp32(block.d);
        const float min = fp16_to_fp32(block.m);
        const __m256 vscale = _mm256_set1_ps(scale);
        const __m256 vmin = _mm256_set1_ps(min);

        // Load 16 bytes (contains 32 4-bit values)
        __m128i v4bit = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block.qs));

        // Extract low nibbles (first 16 values)
        __m128i vlow = _mm_and_si128(v4bit, _mm_set1_epi8(0x0F));

        // Extract high nibbles (second 16 values)
        __m128i vhigh = _mm_and_si128(_mm_srli_epi16(v4bit, 4), _mm_set1_epi8(0x0F));

        // Process low nibbles: output[0-15]
        // First 8 low nibbles
        __m256i vi32_low0 = _mm256_cvtepu8_epi32(vlow);
        __m256 vf_low0 = _mm256_cvtepi32_ps(vi32_low0);
        vf_low0 = _mm256_fmadd_ps(vf_low0, vscale, vmin); // scale * value + min
        _mm256_storeu_ps(&output[0], vf_low0);

        // Second 8 low nibbles
        __m256i vi32_low1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vlow, 8));
        __m256 vf_low1 = _mm256_cvtepi32_ps(vi32_low1);
        vf_low1 = _mm256_fmadd_ps(vf_low1, vscale, vmin);
        _mm256_storeu_ps(&output[8], vf_low1);

        // Process high nibbles: output[16-31]
        // First 8 high nibbles
        __m256i vi32_high0 = _mm256_cvtepu8_epi32(vhigh);
        __m256 vf_high0 = _mm256_cvtepi32_ps(vi32_high0);
        vf_high0 = _mm256_fmadd_ps(vf_high0, vscale, vmin);
        _mm256_storeu_ps(&output[16], vf_high0);

        // Second 8 high nibbles
        __m256i vi32_high1 = _mm256_cvtepu8_epi32(_mm_srli_si128(vhigh, 8));
        __m256 vf_high1 = _mm256_cvtepi32_ps(vi32_high1);
        vf_high1 = _mm256_fmadd_ps(vf_high1, vscale, vmin);
        _mm256_storeu_ps(&output[24], vf_high1);
    }
#endif

    Q4_1Tensor::~Q4_1Tensor() {}

    bool Q4_1Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_1Tensor::data() const
    {
        assertValid("Q4_1Tensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("Q4_1Tensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }

            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_1Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_1Tensor::mutable_data()
    {
        throw std::runtime_error("Q4_1Tensor::mutable_data: quantized tensors are immutable");
    }

    bool Q4_1Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q4_1Tensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q4_1Tensor::to_bf16(uint16_t *dst) const
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

    void Q4_1Tensor::to_fp16(uint16_t *dst) const
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

    void Q4_1Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q4_1Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q4_1Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void Q4_1Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q4_1Tensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_1Tensor::decode_to_q8_0: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_1Tensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_1Tensor::decode_to_q8_0: block offset out of bounds");
        }

        // Get Q4_1 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
        const Q4_1Block &q4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Use SIMD helper to decode Q4_1 → Q8_0
        uint16_t q8_scale_fp16;
        simd::decode_q4_1_to_q8_0(
            q4_block.qs,     // Q4 packed values
            q4_block.d,      // Q4 scale (FP16)
            q4_block.m,      // Q4 min (FP16)
            output->qs,      // Q8 output values
            &q8_scale_fp16); // Q8 output scale (FP16)

        output->d = q8_scale_fp16;
    }

    void Q4_1Tensor::unpack_block_to_int8(
        size_t row_idx,
        size_t k_block_offset,
        int8_t *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q4_1Tensor::unpack_block_to_int8: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_1Tensor::unpack_block_to_int8: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_1Tensor::unpack_block_to_int8: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_1Tensor::unpack_block_to_int8: block offset out of bounds");
        }

        // Get Q4_1 block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
        const Q4_1Block *q4_block = &blocks[row_idx * blocks_per_row + k_block_offset];

        // Unpack Q4_1 block to INT8 using SIMD dispatcher
        simd::unpack_q4_1_to_int8(*q4_block, output);
    }

    float Q4_1Tensor::get_block_scale(
        size_t row_idx,
        size_t k_block_offset) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_1Tensor::get_block_scale: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_1Tensor::get_block_scale: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_1Tensor::get_block_scale: block offset out of bounds");
        }

        // Get Q4_1 block and return original FP16 scale converted to FP32
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
        const Q4_1Block &q4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        return fp16_to_fp32(q4_block.d);
    }

    float Q4_1Tensor::get_block_min(
        size_t row_idx,
        size_t k_block_offset) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q4_1Tensor::get_block_min: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_1Tensor::get_block_min: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q4_1Block::BLOCK_SIZE - 1) / Q4_1Block::BLOCK_SIZE;

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q4_1Tensor::get_block_min: block offset out of bounds");
        }

        // Get Q4_1 block and return original FP16 min converted to FP32
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_1Block *blocks = reinterpret_cast<const Q4_1Block *>(data_ptr);
        const Q4_1Block &q4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        return fp16_to_fp32(q4_block.m);
    }

} // namespace llaminar2
