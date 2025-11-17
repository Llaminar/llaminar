/**
 * @file Q4_KTensor.cpp
 * @brief Q4_K quantized tensor implementation (4-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include <cstring>
#include <stdexcept>
#include "../utils/Logger.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    Q4_KTensor::Q4_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
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
            throw std::invalid_argument("Q4_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q4_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q4_KTensor: insufficient raw data");
        }
    }

    // Private view constructor (borrows parent's data)
    Q4_KTensor::Q4_KTensor(
        const std::vector<size_t> &shape,
        const uint8_t *parent_raw_data,
        size_t byte_offset,
        std::shared_ptr<TensorBase> parent)
        : shape_(shape),
          is_view_(true),
          raw_data_(), // Empty vector (view doesn't own data)
          raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset),
          parent_(parent),
          device_idx_(-1),
          device_blocks_(nullptr)
    {
    }

    std::shared_ptr<TensorBase> Q4_KTensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate: must be 2D
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q4_KTensor::create_view: only 2D views supported");
        }

        // Validate: K dimension must match
        if (new_shape[1] != shape_[1])
        {
            throw std::invalid_argument("Q4_KTensor::create_view: K dimension must match parent");
        }

        // Validate: offset must be row-aligned (multiple of K)
        if (offset % shape_[1] != 0)
        {
            throw std::invalid_argument("Q4_KTensor::create_view: offset must be row-aligned");
        }

        // Validate: view must fit within parent bounds
        size_t start_row = offset / shape_[1];
        size_t end_row = start_row + new_shape[0];
        if (end_row > shape_[0])
        {
            throw std::out_of_range("Q4_KTensor::create_view: view extends beyond parent bounds");
        }

        // Calculate byte offset for view
        // Q4_K: 256 elements per block, 144 bytes per block
        const size_t K = shape_[1];
        const size_t blocks_per_row = (K + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        const size_t bytes_per_row = blocks_per_row * sizeof(Q4_KBlock);
        size_t byte_offset = start_row * bytes_per_row;

        // If this is already a view, accumulate offsets and use the root parent
        const uint8_t *base_ptr;
        std::shared_ptr<TensorBase> root_parent;

        if (is_view_)
        {
            // Chain views: add offsets
            byte_offset += view_byte_offset_;
            base_ptr = raw_data_ptr_;
            root_parent = parent_; // Use root parent, not intermediate view
        }
        else
        {
            // First-level view
            base_ptr = raw_data_.data();
            root_parent = shared_from_this();
        }

        // Create view using private constructor
        return std::shared_ptr<Q4_KTensor>(new Q4_KTensor(new_shape, base_ptr, byte_offset, root_parent));
    }

    std::unique_ptr<ITensorGemm> Q4_KTensor::createGemm()
    {
        return std::make_unique<llaminar2::gemm_v4::OneDNNGemmKernel>(this);
    }

    void Q4_KTensor::decodeBlock(const Q4_KBlock &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        decodeBlockScalar(block, output);
#endif
    }

    void Q4_KTensor::decodeBlockScalar(const Q4_KBlock &block, float *output)
    {
        // Q4_K: 256 elements processed in 4 groups of 64 elements
        // Reference: llama.cpp ggml-quants.c dequantize_row_q4_K()
        // Uses get_scale_min_k4() for extracting 6-bit scales and mins

        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *q = block.qs;
        float *y = output;

        size_t is = 0;

        // Process 4 groups of 64 elements each (256 total)
        for (size_t j = 0; j < 256; j += 64)
        {
            // Extract scale and min for first 32 elements
            uint8_t sc1, m1;
            get_scale_min_k4(is + 0, block.scales, &sc1, &m1);
            const float d1 = d * sc1;
            const float m1_val = dmin * m1;

            // Extract scale and min for second 32 elements
            uint8_t sc2, m2;
            get_scale_min_k4(is + 1, block.scales, &sc2, &m2);
            const float d2 = d * sc2;
            const float m2_val = dmin * m2;

            // First 32 elements: lower 4 bits
            for (size_t l = 0; l < 32; ++l)
            {
                *y++ = d1 * (q[l] & 0xF) - m1_val;
            }

            // Second 32 elements: upper 4 bits
            for (size_t l = 0; l < 32; ++l)
            {
                *y++ = d2 * (q[l] >> 4) - m2_val;
            }

            q += 32;
            is += 2;
        }
    }

#if defined(__AVX2__)
    void Q4_KTensor::decodeBlockAVX2(const Q4_KBlock &block, float *output)
    {
        // AVX2: Process 8 floats at a time
        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *q = block.qs;
        float *y = output;

        size_t is = 0;

        // Process 4 groups of 64 elements each (256 total)
        for (size_t j = 0; j < 256; j += 64)
        {
            // Extract scale and min for first 32 elements
            uint8_t sc1, m1;
            get_scale_min_k4(is + 0, block.scales, &sc1, &m1);
            const float d1 = d * sc1;
            const float m1_val = dmin * m1;

            // Extract scale and min for second 32 elements
            uint8_t sc2, m2;
            get_scale_min_k4(is + 1, block.scales, &sc2, &m2);
            const float d2 = d * sc2;
            const float m2_val = dmin * m2;

            // First 32 elements: lower 4 bits (process 8 at a time)
            __m256 d1_vec = _mm256_set1_ps(d1);
            __m256 m1_vec = _mm256_set1_ps(m1_val);

            for (size_t l = 0; l < 32; l += 8)
            {
                // Load 8 bytes, extract lower 4 bits
                __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                __m256i low_bits = _mm256_and_si256(q_32, _mm256_set1_epi32(0xF));

                // Convert to float and apply formula: d1 * low_bits - m1_val
                __m256 q_float = _mm256_cvtepi32_ps(low_bits);
                __m256 result = _mm256_fmadd_ps(d1_vec, q_float, _mm256_sub_ps(_mm256_setzero_ps(), m1_vec));
                _mm256_storeu_ps(y + l, result);
            }
            y += 32;

            // Second 32 elements: upper 4 bits (process 8 at a time)
            __m256 d2_vec = _mm256_set1_ps(d2);
            __m256 m2_vec = _mm256_set1_ps(m2_val);

            for (size_t l = 0; l < 32; l += 8)
            {
                // Load 8 bytes, extract upper 4 bits
                __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + l));
                __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                __m256i high_bits = _mm256_srli_epi32(q_32, 4);

                // Convert to float and apply formula: d2 * high_bits - m2_val
                __m256 q_float = _mm256_cvtepi32_ps(high_bits);
                __m256 result = _mm256_fmadd_ps(d2_vec, q_float, _mm256_sub_ps(_mm256_setzero_ps(), m2_vec));
                _mm256_storeu_ps(y + l, result);
            }
            y += 32;

            q += 32;
            is += 2;
        }
    }
#endif

#if defined(__AVX512F__)
    void Q4_KTensor::decodeBlockAVX512(const Q4_KBlock &block, float *output)
    {
        // AVX512: Process 16 floats at a time
        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *q = block.qs;
        float *y = output;

        size_t is = 0;

        // Process 4 groups of 64 elements each (256 total)
        for (size_t j = 0; j < 256; j += 64)
        {
            // Extract scale and min for first 32 elements
            uint8_t sc1, m1;
            get_scale_min_k4(is + 0, block.scales, &sc1, &m1);
            const float d1 = d * sc1;
            const float m1_val = dmin * m1;

            // Extract scale and min for second 32 elements
            uint8_t sc2, m2;
            get_scale_min_k4(is + 1, block.scales, &sc2, &m2);
            const float d2 = d * sc2;
            const float m2_val = dmin * m2;

            // First 32 elements: lower 4 bits (process 16 at a time)
            __m512 d1_vec = _mm512_set1_ps(d1);
            __m512 m1_vec = _mm512_set1_ps(m1_val);

            for (size_t l = 0; l < 32; l += 16)
            {
                // Load 16 bytes, extract lower 4 bits
                __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                __m512i low_bits = _mm512_and_si512(q_32, _mm512_set1_epi32(0xF));

                // Convert to float and apply formula: d1 * low_bits - m1_val
                __m512 q_float = _mm512_cvtepi32_ps(low_bits);
                __m512 result = _mm512_fmadd_ps(d1_vec, q_float, _mm512_sub_ps(_mm512_setzero_ps(), m1_vec));
                _mm512_storeu_ps(y + l, result);
            }
            y += 32;

            // Second 32 elements: upper 4 bits (process 16 at a time)
            __m512 d2_vec = _mm512_set1_ps(d2);
            __m512 m2_vec = _mm512_set1_ps(m2_val);

            for (size_t l = 0; l < 32; l += 16)
            {
                // Load 16 bytes, extract upper 4 bits
                __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + l));
                __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);
                __m512i high_bits = _mm512_srli_epi32(q_32, 4);

                // Convert to float and apply formula: d2 * high_bits - m2_val
                __m512 q_float = _mm512_cvtepi32_ps(high_bits);
                __m512 result = _mm512_fmadd_ps(d2_vec, q_float, _mm512_sub_ps(_mm512_setzero_ps(), m2_vec));
                _mm512_storeu_ps(y + l, result);
            }
            y += 32;

            q += 32;
            is += 2;
        }
    }
#endif

    // Helper function matching llama.cpp's get_scale_min_k4
    inline void Q4_KTensor::get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
    {
        if (j < 4)
        {
            *d = q[j] & 63;
            *m = q[j + 4] & 63;
        }
        else
        {
            *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        }
    }

    Q4_KTensor::~Q4_KTensor() {}

    bool Q4_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q4_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q4_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q4_KTensor::mutable_data()
    {
        throw std::runtime_error("Q4_KTensor::mutable_data: quantized tensors are immutable");
    }

        bool Q4_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q4_KTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q4_KTensor::to_bf16(uint16_t *dst) const
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

    void Q4_KTensor::to_fp16(uint16_t *dst) const
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

    void Q4_KTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q4_KTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q4_KTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void Q4_KTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (output == nullptr)
        {
            throw std::invalid_argument("Q4_KTensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() < 2)
        {
            throw std::runtime_error("Q4_KTensor::decode_to_q8_0: tensor shape is invalid");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q4_KTensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t blocks_per_row = (shape_[1] + Q4_KBlock::BLOCK_SIZE - 1) / Q4_KBlock::BLOCK_SIZE;
        const size_t sub_blocks_per_super = Q4_KBlock::BLOCK_SIZE / Q8_0Block::BLOCK_SIZE; // 256 / 32 = 8
        const size_t total_q8_blocks = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

        if (k_block_offset >= total_q8_blocks)
        {
            throw std::out_of_range("Q4_KTensor::decode_to_q8_0: block offset exceeds row length");
        }

        const size_t superblock_idx = k_block_offset / sub_blocks_per_super;
        const size_t subblock_idx = k_block_offset % sub_blocks_per_super;

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q4_KBlock *blocks = reinterpret_cast<const Q4_KBlock *>(data_ptr);
        const Q4_KBlock &block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Decode using SIMD helper (auto-dispatches to scalar/AVX2/AVX512)
        simd::decode_q4_k_to_q8_0(block, subblock_idx, output->qs, &output->d);
    }

} // namespace llaminar2
