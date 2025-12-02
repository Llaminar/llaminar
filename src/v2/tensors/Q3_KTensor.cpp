/**
 * @file Q3_KTensor.cpp
 * @brief Q3_K quantized tensor implementation (3-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "SIMDHelpers.h"
#include <cstring>
#include <stdexcept>
#include "../utils/Logger.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    Q3_KTensor::Q3_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
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
            throw std::invalid_argument("Q3_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q3_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q3_KTensor: insufficient raw data");
        }
    }

    // Private view constructor
    Q3_KTensor::Q3_KTensor(const std::vector<size_t> &shape,
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
    }

    std::shared_ptr<TensorBase> Q3_KTensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        if (shape_.size() != 2 || new_shape.size() != 2)
        {
            throw std::invalid_argument("Q3_KTensor::create_view: only 2D row-slice views supported");
        }
        if (new_shape[1] != shape_[1])
        {
            throw std::invalid_argument("Q3_KTensor::create_view: column count (K) must match parent");
        }
        if (offset % shape_[1] != 0)
        {
            throw std::invalid_argument("Q3_KTensor::create_view: offset must be row-aligned");
        }
        if (offset + new_shape[0] * new_shape[1] > shape_[0] * shape_[1])
        {
            throw std::out_of_range("Q3_KTensor::create_view: view exceeds parent bounds");
        }

        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        const size_t first_row = offset / cols;
        const size_t block_offset = first_row * blocks_per_row;
        const size_t byte_offset = block_offset * sizeof(Q3_KBlock);

        const uint8_t *root_data_ptr;
        size_t cumulative_byte_offset;
        std::shared_ptr<TensorBase> root_parent;

        if (is_view_)
        {
            root_data_ptr = raw_data_ptr_;
            cumulative_byte_offset = view_byte_offset_ + byte_offset;
            root_parent = parent_;
        }
        else
        {
            root_data_ptr = raw_data_.data();
            cumulative_byte_offset = byte_offset;
            root_parent = std::static_pointer_cast<TensorBase>(shared_from_this());
        }

        return std::shared_ptr<Q3_KTensor>(new Q3_KTensor(
            new_shape, root_data_ptr, cumulative_byte_offset, root_parent));
    }

    std::unique_ptr<ITensorGemm> Q3_KTensor::createGemm()
    {
        // Use QuantisedGemmKernel - requires IINT8Unpackable interface
        return std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(this);
    }

    void Q3_KTensor::decodeBlockScalar(const Q3_KBlock &block, float *output)
    {
        // Scalar implementation for Q3_K: 256 elements in super-block
        // Q3_K uses 3 bits per value: 2 bits in qs[] + 1 high bit in hmask[]
        // Formula: output[i] = d * scale * (low_bits - (high_bit ? 0 : 4))

        const float d_all = fp16_to_fp32(block.d);

        const uint8_t *q = block.qs;
        const uint8_t *hm = block.hmask;
        uint8_t m = 1; // Mask for hmask bit testing

        // Unpack 16 6-bit scales from 12 bytes (complex bit manipulation)
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;

        uint32_t aux[4];
        memcpy(aux, block.scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const int8_t *scales = (const int8_t *)aux;

        float *y = output;
        int is = 0; // Scale index

        // Process 2 chunks of 128 elements each
        for (int n = 0; n < 256; n += 128)
        {
            int shift = 0;

            // 4 groups per chunk
            for (int j = 0; j < 4; ++j)
            {
                // First group of 16 elements
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                {
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                }

                // Second group of 16 elements
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    void Q3_KTensor::decodeBlock(const Q3_KBlock &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        decodeBlockScalar(block, output);
#endif
    }

#if defined(__AVX2__)
    void Q3_KTensor::decodeBlockAVX2(const Q3_KBlock &block, float *output)
    {
        // AVX2 implementation for Q3_K: 256 elements with 3-bit precision
        const float d_all = fp16_to_fp32(block.d);

        const uint8_t *q = block.qs;
        const uint8_t *hm = block.hmask;
        uint8_t m = 1;

        // Unpack scales (same as scalar)
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;

        uint32_t aux[4];
        memcpy(aux, block.scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const int8_t *scales = (const int8_t *)aux;

        float *y = output;
        int is = 0;

        // Process 2 chunks of 128 elements each
        for (int n = 0; n < 256; n += 128)
        {
            int shift = 0;

            // 4 groups per chunk
            for (int j = 0; j < 4; ++j)
            {
                // First group of 16 elements - process 8 at a time with AVX2
                float dl = d_all * (scales[is++] - 32);
                __m256 vdl = _mm256_set1_ps(dl);

                for (int l = 0; l < 16; l += 8)
                {
                    // Load 8 bytes
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&q[l]));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);

                    // Shift and mask to get 2-bit low values
                    __m256i shift_vec = _mm256_set1_epi32(shift);
                    __m256i low_bits = _mm256_and_si256(_mm256_srlv_epi32(q_32, shift_vec), _mm256_set1_epi32(3));

                    // Load hmask bits for high bit
                    __m128i hm_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&hm[l]));
                    __m256i hm_32 = _mm256_cvtepu8_epi32(hm_bytes);
                    __m256i high_bits = _mm256_and_si256(hm_32, _mm256_set1_epi32(m));

                    // Compute: low_bits - (high_bit ? 0 : 4)
                    // If high_bit == 0, subtract 4; if high_bit != 0, subtract 0
                    __m256i is_zero_mask = _mm256_cmpeq_epi32(high_bits, _mm256_setzero_si256());
                    __m256i offset = _mm256_and_si256(is_zero_mask, _mm256_set1_epi32(4));
                    __m256i q3_vals = _mm256_sub_epi32(low_bits, offset);

                    // Convert to float and apply scale
                    __m256 vq = _mm256_cvtepi32_ps(q3_vals);
                    __m256 result = _mm256_mul_ps(vdl, vq);

                    _mm256_storeu_ps(&y[l], result);
                }
                y += 16;

                // Second group of 16 elements
                dl = d_all * (scales[is++] - 32);
                vdl = _mm256_set1_ps(dl);

                for (int l = 0; l < 16; l += 8)
                {
                    __m128i q_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&q[l + 16]));
                    __m256i q_32 = _mm256_cvtepu8_epi32(q_bytes);
                    __m256i shift_vec = _mm256_set1_epi32(shift);
                    __m256i low_bits = _mm256_and_si256(_mm256_srlv_epi32(q_32, shift_vec), _mm256_set1_epi32(3));

                    __m128i hm_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&hm[l + 16]));
                    __m256i hm_32 = _mm256_cvtepu8_epi32(hm_bytes);
                    __m256i high_bits = _mm256_and_si256(hm_32, _mm256_set1_epi32(m));

                    __m256i is_zero_mask = _mm256_cmpeq_epi32(high_bits, _mm256_setzero_si256());
                    __m256i offset = _mm256_and_si256(is_zero_mask, _mm256_set1_epi32(4));
                    __m256i q3_vals = _mm256_sub_epi32(low_bits, offset);

                    __m256 vq = _mm256_cvtepi32_ps(q3_vals);
                    __m256 result = _mm256_mul_ps(vdl, vq);

                    _mm256_storeu_ps(&y[l], result);
                }
                y += 16;

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
#endif

#if defined(__AVX512F__)
    void Q3_KTensor::decodeBlockAVX512(const Q3_KBlock &block, float *output)
    {
        // AVX512 implementation for Q3_K: 256 elements with 3-bit precision
        const float d_all = fp16_to_fp32(block.d);

        const uint8_t *q = block.qs;
        const uint8_t *hm = block.hmask;
        uint8_t m = 1;

        // Unpack scales (same as scalar)
        const uint32_t kmask1 = 0x03030303;
        const uint32_t kmask2 = 0x0f0f0f0f;

        uint32_t aux[4];
        memcpy(aux, block.scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const int8_t *scales = (const int8_t *)aux;

        float *y = output;
        int is = 0;

        // Process 2 chunks of 128 elements each
        for (int n = 0; n < 256; n += 128)
        {
            int shift = 0;

            // 4 groups per chunk
            for (int j = 0; j < 4; ++j)
            {
                // First group of 16 elements - process all 16 with AVX512
                float dl = d_all * (scales[is++] - 32);
                __m512 vdl = _mm512_set1_ps(dl);

                // Load all 16 bytes
                __m128i q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&q[0]));
                __m512i q_32 = _mm512_cvtepu8_epi32(q_bytes);

                // Shift and mask to get 2-bit low values
                __m512i shift_vec = _mm512_set1_epi32(shift);
                __m512i low_bits = _mm512_and_epi32(_mm512_srlv_epi32(q_32, shift_vec), _mm512_set1_epi32(3));

                // Load hmask bits for high bit
                __m128i hm_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&hm[0]));
                __m512i hm_32 = _mm512_cvtepu8_epi32(hm_bytes);
                __mmask16 high_bits_mask = _mm512_test_epi32_mask(hm_32, _mm512_set1_epi32(m));

                // Compute: low_bits - (high_bit ? 0 : 4)
                // If high_bit is set (mask bit = 1), subtract 0; otherwise subtract 4
                __m512i q3_vals = _mm512_mask_sub_epi32(low_bits, ~high_bits_mask, low_bits, _mm512_set1_epi32(4));

                // Convert to float and apply scale
                __m512 vq = _mm512_cvtepi32_ps(q3_vals);
                __m512 result = _mm512_mul_ps(vdl, vq);

                _mm512_storeu_ps(y, result);
                y += 16;

                // Second group of 16 elements
                dl = d_all * (scales[is++] - 32);
                vdl = _mm512_set1_ps(dl);

                q_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&q[16]));
                q_32 = _mm512_cvtepu8_epi32(q_bytes);
                low_bits = _mm512_and_epi32(_mm512_srlv_epi32(q_32, shift_vec), _mm512_set1_epi32(3));

                hm_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&hm[16]));
                hm_32 = _mm512_cvtepu8_epi32(hm_bytes);
                high_bits_mask = _mm512_test_epi32_mask(hm_32, _mm512_set1_epi32(m));

                q3_vals = _mm512_mask_sub_epi32(low_bits, ~high_bits_mask, low_bits, _mm512_set1_epi32(4));

                vq = _mm512_cvtepi32_ps(q3_vals);
                result = _mm512_mul_ps(vdl, vq);

                _mm512_storeu_ps(y, result);
                y += 16;

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
#endif

    Q3_KTensor::~Q3_KTensor() {}

    bool Q3_KTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *Q3_KTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * Q3_KBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q3_KTensor::mutable_data()
    {
        throw std::runtime_error("Q3_KTensor::mutable_data: quantized tensors are immutable");
    }

    void Q3_KTensor::unpack_block_to_int8(
        size_t row_idx,
        size_t k_block_offset,
        int8_t *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q3_KTensor::unpack_block_to_int8: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q3_KTensor::unpack_block_to_int8: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q3_KTensor::unpack_block_to_int8: row index out of bounds");
        }

        // Map 32-element block index to 256-element super-block index
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;

        if (super_block_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q3_KTensor::unpack_block_to_int8: block offset out of bounds");
        }

        // Get Q3_K super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
        const Q3_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        // Transcode directly to int8 (fused dequant/requant)
        float scale, min_val;
        simd::transcode_q3_k_to_int8(block, sub_block_idx, output, &scale, &min_val);
    }

    float Q3_KTensor::get_block_scale(
        size_t row_idx,
        size_t k_block_offset) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q3_KTensor::get_block_scale: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q3_KTensor::get_block_scale: row index out of bounds");
        }

        // Map 32-element block index to 256-element super-block index
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;

        if (super_block_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q3_KTensor::get_block_scale: block offset out of bounds");
        }

        // Get Q3_K super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
        const Q3_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        // Transcode directly to int8 (fused dequant/requant)
        int8_t temp_i8[32];
        float scale, min_val;
        simd::transcode_q3_k_to_int8(block, sub_block_idx, temp_i8, &scale, &min_val);

        return scale;
    }

    float Q3_KTensor::get_block_min(
        size_t row_idx,
        size_t k_block_offset) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q3_KTensor::get_block_min: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q3_KTensor::get_block_min: row index out of bounds");
        }

        // Map 32-element block index to 256-element super-block index
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;

        if (super_block_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q3_KTensor::get_block_min: block offset out of bounds");
        }

        // Get Q3_K super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
        const Q3_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        // Transcode directly to int8 (fused dequant/requant)
        int8_t temp_i8[32];
        float scale, min_val;
        simd::transcode_q3_k_to_int8(block, sub_block_idx, temp_i8, &scale, &min_val);

        return min_val;
    }

    void Q3_KTensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q3_KTensor::unpack_superblock_to_int8: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q3_KTensor::unpack_superblock_to_int8: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q3_KTensor::unpack_superblock_to_int8: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;

        if (superblock_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q3_KTensor::unpack_superblock_to_int8: superblock index out of bounds");
        }

        // Get Q3_K super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
        const Q3_KBlock &block = blocks[row_idx * super_blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_q3_k_superblock_to_int8(block, output, scales, mins);
    }

    void Q3_KTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (output == nullptr)
        {
            throw std::invalid_argument("Q3_KTensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() < 2)
        {
            throw std::runtime_error("Q3_KTensor::decode_to_q8_0: tensor shape is invalid");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q3_KTensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t blocks_per_row = (shape_[1] + Q3_KBlock::BLOCK_SIZE - 1) / Q3_KBlock::BLOCK_SIZE;
        const size_t sub_blocks_per_super = Q3_KBlock::BLOCK_SIZE / Q8_0Block::BLOCK_SIZE; // 256 / 32 = 8
        const size_t total_q8_blocks = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

        if (k_block_offset >= total_q8_blocks)
        {
            throw std::out_of_range("Q3_KTensor::decode_to_q8_0: block offset exceeds row length");
        }

        const size_t superblock_idx = k_block_offset / sub_blocks_per_super;
        const size_t subblock_idx = k_block_offset % sub_blocks_per_super;

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q3_KBlock *blocks = reinterpret_cast<const Q3_KBlock *>(data_ptr);
        const Q3_KBlock &block = blocks[row_idx * blocks_per_row + superblock_idx];

        simd::decode_q3_k_to_q8_0(block, subblock_idx, output->qs, &output->d);
    }

    bool Q3_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q3_KTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q3_KTensor::to_bf16(uint16_t *dst) const
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

    void Q3_KTensor::to_fp16(uint16_t *dst) const
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

    void Q3_KTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q3_KTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q3_KTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

} // namespace llaminar2
