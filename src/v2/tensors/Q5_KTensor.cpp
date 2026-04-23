/**
 * @file Q5_KTensor.cpp
 * @brief Q5_K quantized tensor implementation (5-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
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

    Q5_KTensor::Q5_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape),
          is_view_(false),
          raw_data_(raw_data),
          raw_data_ptr_(nullptr),
          view_byte_offset_(0),
          parent_(nullptr),
          device_(DeviceId::cpu()),
          device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q5_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q5_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q5_KTensor: insufficient raw data");
        }
    }

    // Private view constructor (borrows parent's data)
    Q5_KTensor::Q5_KTensor(
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
          device_(DeviceId::cpu()),
          device_blocks_(nullptr)
    {
    }
    // Zero-copy constructor for mmap-backed data
    Q5_KTensor::Q5_KTensor(const std::vector<size_t> &shape,
                           const uint8_t *mmap_data,
                           size_t byte_size,
                           std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), raw_data_(), device_(DeviceId::cpu()), device_blocks_(nullptr),
          is_view_(true), raw_data_ptr_(mmap_data), view_byte_offset_(0),
          parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)), data_byte_size_(byte_size)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q5_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q5_KBlock);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("Q5_KTensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    std::shared_ptr<TensorBase> Q5_KTensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate: view must be 2D
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q5_KTensor::create_view: only 2D views supported");
        }

        // Compute effective 2D layout (supports both 2D and 3D parents).
        // GGUF 3D: shape=[ne0, ne1, ne2] where ne0=cols (fastest), ne2=outermost.
        // Flattened to 2D [ne1*ne2, ne0] = [total_rows, K].
        size_t K, total_rows;
        if (shape_.size() == 2)
        {
            K = shape_[1];
            total_rows = shape_[0];
        }
        else if (shape_.size() == 3)
        {
            // GGUF 3D: shape = [ne[0], ne[1], ne[2]], ne[0] is fastest-varying (cols/K)
            K = shape_[0];
            total_rows = shape_[1] * shape_[2];
        }
        else
        {
            throw std::invalid_argument("Q5_KTensor::create_view: parent must be 2D or 3D");
        }

        // Validate: K dimension must match
        if (new_shape[1] != K)
        {
            throw std::invalid_argument("Q5_KTensor::create_view: K dimension must match parent");
        }

        // Validate: offset must be row-aligned (multiple of K)
        if (offset % K != 0)
        {
            throw std::invalid_argument("Q5_KTensor::create_view: offset must be row-aligned");
        }

        // Validate: view must fit within parent bounds
        size_t start_row = offset / K;
        size_t end_row = start_row + new_shape[0];
        if (end_row > total_rows)
        {
            throw std::out_of_range("Q5_KTensor::create_view: view exceeds parent bounds");
        }

        // Calculate byte offset for view
        // Q5_K: 256 elements per block, 176 bytes per block
        const size_t blocks_per_row = (K + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        const size_t bytes_per_row = blocks_per_row * sizeof(Q5_KBlock);
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
        return std::shared_ptr<Q5_KTensor>(new Q5_KTensor(new_shape, base_ptr, byte_offset, root_parent));
    }

    std::unique_ptr<ITensorGemm> Q5_KTensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void Q5_KTensor::unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q5_KTensor::unpack_block_to_int8: output must not be null");
        }

        // Map 32-element block index to 256-element super-block index
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;

        if (super_block_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q5_KTensor::unpack_block_to_int8: block offset out of bounds");
        }

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
        const Q5_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        simd::unpack_q5_k_to_int8(block, sub_block_idx, output);
    }

    float Q5_KTensor::get_block_scale(size_t row_idx, size_t k_block_offset) const
    {
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
        const Q5_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        float scale, min_val;
        simd::get_q5_k_scale_min(block, sub_block_idx, &scale, &min_val);
        return scale;
    }

    float Q5_KTensor::get_block_min(size_t row_idx, size_t k_block_offset) const
    {
        size_t super_block_idx = k_block_offset / 8;
        size_t sub_block_idx = k_block_offset % 8;

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
        const Q5_KBlock &block = blocks[row_idx * super_blocks_per_row + super_block_idx];

        float scale, min_val;
        simd::get_q5_k_scale_min(block, sub_block_idx, &scale, &min_val);
        return -min_val;
    }

    void Q5_KTensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q5_KTensor::unpack_superblock_to_int8: output must not be null");
        }

        if (shape_.size() != 2)
        {
            throw std::runtime_error("Q5_KTensor::unpack_superblock_to_int8: tensor must be 2D");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q5_KTensor::unpack_superblock_to_int8: row index out of bounds");
        }

        const size_t cols = shape_[1];
        const size_t super_blocks_per_row = (cols + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;

        if (superblock_idx >= super_blocks_per_row)
        {
            throw std::out_of_range("Q5_KTensor::unpack_superblock_to_int8: superblock index out of bounds");
        }

        // Get Q5_K super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
        const Q5_KBlock &block = blocks[row_idx * super_blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_q5_k_superblock_to_int8(block, output, scales, mins);
    }

    void Q5_KTensor::decodeBlock(const Q5_KBlock &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        decodeBlockScalar(block, output);
#endif
    }

    void Q5_KTensor::decodeBlockScalar(const Q5_KBlock &block, float *output)
    {
        // Q5_K: 256 elements processed in 4 groups of 64 elements
        // Reference: llama.cpp ggml-quants.c dequantize_row_q5_K()
        // Each element is 5 bits: 4 bits in qs[], 1 bit in qh[]

        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *ql = block.qs;
        const uint8_t *qh = block.qh;
        float *y = output;

        size_t is = 0;
        uint8_t u1 = 1, u2 = 2; // Bit masks for high bit extraction

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

            // First 32 elements: lower 4 bits + high bit from u1 mask
            for (size_t l = 0; l < 32; ++l)
            {
                const uint8_t q_low = ql[l] & 0xF;
                const uint8_t q_high = (qh[l] & u1) ? 16 : 0;
                *y++ = d1 * (q_low + q_high) - m1_val;
            }

            // Second 32 elements: upper 4 bits + high bit from u2 mask
            for (size_t l = 0; l < 32; ++l)
            {
                const uint8_t q_low = ql[l] >> 4;
                const uint8_t q_high = (qh[l] & u2) ? 16 : 0;
                *y++ = d2 * (q_low + q_high) - m2_val;
            }

            ql += 32;
            is += 2;
            u1 <<= 2; // Shift masks for next group
            u2 <<= 2;
        }
    }

#if defined(__AVX2__)
    void Q5_KTensor::decodeBlockAVX2(const Q5_KBlock &block, float *output)
    {
        // AVX2: Process 8 floats at a time
        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *ql = block.qs;
        const uint8_t *qh = block.qh;
        float *y = output;

        size_t is = 0;
        uint8_t u1 = 1, u2 = 2;

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

            // First 32 elements: lower 4 bits + high bit from u1 mask
            __m256 d1_vec = _mm256_set1_ps(d1);
            __m256 m1_vec = _mm256_set1_ps(m1_val);
            __m256i u1_mask = _mm256_set1_epi32(u1);

            for (size_t l = 0; l < 32; l += 8)
            {
                // Load 8 bytes for lower 4 bits
                __m128i ql_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ql + l));
                __m256i ql_32 = _mm256_cvtepu8_epi32(ql_bytes);
                __m256i q_low = _mm256_and_si256(ql_32, _mm256_set1_epi32(0xF));

                // Load 8 bytes for high bits
                __m128i qh_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qh + l));
                __m256i qh_32 = _mm256_cvtepu8_epi32(qh_bytes);

                // Test high bit: (qh & u1) != 0 ? 16 : 0
                __m256i high_bit_set = _mm256_and_si256(qh_32, u1_mask);
                __m256i is_nonzero = _mm256_cmpeq_epi32(high_bit_set, _mm256_setzero_si256());
                __m256i q_high = _mm256_and_si256(_mm256_xor_si256(is_nonzero, _mm256_set1_epi32(-1)), _mm256_set1_epi32(16));

                // Combine: q_low + q_high
                __m256i q_combined = _mm256_add_epi32(q_low, q_high);
                __m256 q_float = _mm256_cvtepi32_ps(q_combined);

                // Apply formula: d1 * q_float - m1_val
                __m256 result = _mm256_fmadd_ps(d1_vec, q_float, _mm256_sub_ps(_mm256_setzero_ps(), m1_vec));
                _mm256_storeu_ps(y + l, result);
            }
            y += 32;

            // Second 32 elements: upper 4 bits + high bit from u2 mask
            __m256 d2_vec = _mm256_set1_ps(d2);
            __m256 m2_vec = _mm256_set1_ps(m2_val);
            __m256i u2_mask = _mm256_set1_epi32(u2);

            for (size_t l = 0; l < 32; l += 8)
            {
                // Load 8 bytes for upper 4 bits
                __m128i ql_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ql + l));
                __m256i ql_32 = _mm256_cvtepu8_epi32(ql_bytes);
                __m256i q_low = _mm256_srli_epi32(ql_32, 4);

                // Load 8 bytes for high bits
                __m128i qh_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(qh + l));
                __m256i qh_32 = _mm256_cvtepu8_epi32(qh_bytes);

                // Test high bit: (qh & u2) != 0 ? 16 : 0
                __m256i high_bit_set = _mm256_and_si256(qh_32, u2_mask);
                __m256i is_nonzero = _mm256_cmpeq_epi32(high_bit_set, _mm256_setzero_si256());
                __m256i q_high = _mm256_and_si256(_mm256_xor_si256(is_nonzero, _mm256_set1_epi32(-1)), _mm256_set1_epi32(16));

                // Combine: q_low + q_high
                __m256i q_combined = _mm256_add_epi32(q_low, q_high);
                __m256 q_float = _mm256_cvtepi32_ps(q_combined);

                // Apply formula: d2 * q_float - m2_val
                __m256 result = _mm256_fmadd_ps(d2_vec, q_float, _mm256_sub_ps(_mm256_setzero_ps(), m2_vec));
                _mm256_storeu_ps(y + l, result);
            }
            y += 32;

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
#endif

#if defined(__AVX512F__)
    void Q5_KTensor::decodeBlockAVX512(const Q5_KBlock &block, float *output)
    {
        // AVX512: Process 16 floats at a time
        const float d = fp16_to_fp32(block.d);
        const float dmin = fp16_to_fp32(block.dmin);

        const uint8_t *ql = block.qs;
        const uint8_t *qh = block.qh;
        float *y = output;

        size_t is = 0;
        uint8_t u1 = 1, u2 = 2;

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

            // First 32 elements: lower 4 bits + high bit from u1 mask
            __m512 d1_vec = _mm512_set1_ps(d1);
            __m512 m1_vec = _mm512_set1_ps(m1_val);
            __m512i u1_mask = _mm512_set1_epi32(u1);

            for (size_t l = 0; l < 32; l += 16)
            {
                // Load 16 bytes for lower 4 bits
                __m128i ql_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql + l));
                __m512i ql_32 = _mm512_cvtepu8_epi32(ql_bytes);
                __m512i q_low = _mm512_and_si512(ql_32, _mm512_set1_epi32(0xF));

                // Load 16 bytes for high bits
                __m128i qh_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh + l));
                __m512i qh_32 = _mm512_cvtepu8_epi32(qh_bytes);

                // Test high bit: (qh & u1) != 0 ? 16 : 0
                __m512i high_bit_set = _mm512_and_si512(qh_32, u1_mask);
                __mmask16 is_nonzero = _mm512_cmpneq_epi32_mask(high_bit_set, _mm512_setzero_si512());
                __m512i q_high = _mm512_maskz_set1_epi32(is_nonzero, 16);

                // Combine: q_low + q_high
                __m512i q_combined = _mm512_add_epi32(q_low, q_high);
                __m512 q_float = _mm512_cvtepi32_ps(q_combined);

                // Apply formula: d1 * q_float - m1_val
                __m512 result = _mm512_fmadd_ps(d1_vec, q_float, _mm512_sub_ps(_mm512_setzero_ps(), m1_vec));
                _mm512_storeu_ps(y + l, result);
            }
            y += 32;

            // Second 32 elements: upper 4 bits + high bit from u2 mask
            __m512 d2_vec = _mm512_set1_ps(d2);
            __m512 m2_vec = _mm512_set1_ps(m2_val);
            __m512i u2_mask = _mm512_set1_epi32(u2);

            for (size_t l = 0; l < 32; l += 16)
            {
                // Load 16 bytes for upper 4 bits
                __m128i ql_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql + l));
                __m512i ql_32 = _mm512_cvtepu8_epi32(ql_bytes);
                __m512i q_low = _mm512_srli_epi32(ql_32, 4);

                // Load 16 bytes for high bits
                __m128i qh_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh + l));
                __m512i qh_32 = _mm512_cvtepu8_epi32(qh_bytes);

                // Test high bit: (qh & u2) != 0 ? 16 : 0
                __m512i high_bit_set = _mm512_and_si512(qh_32, u2_mask);
                __mmask16 is_nonzero = _mm512_cmpneq_epi32_mask(high_bit_set, _mm512_setzero_si512());
                __m512i q_high = _mm512_maskz_set1_epi32(is_nonzero, 16);

                // Combine: q_low + q_high
                __m512i q_combined = _mm512_add_epi32(q_low, q_high);
                __m512 q_float = _mm512_cvtepi32_ps(q_combined);

                // Apply formula: d2 * q_float - m2_val
                __m512 result = _mm512_fmadd_ps(d2_vec, q_float, _mm512_sub_ps(_mm512_setzero_ps(), m2_vec));
                _mm512_storeu_ps(y + l, result);
            }
            y += 32;

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
#endif

    // Helper function matching llama.cpp's get_scale_min_k4
    inline void Q5_KTensor::get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
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

    Q5_KTensor::~Q5_KTensor() {}

    const float *Q5_KTensor::data() const
    {
        assertValid("Q5_KTensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("Q5_KTensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            // Use view-aware data pointer
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t row_offset = r * shape_[1];
                    const size_t col_offset = b * Q5_KBlock::BLOCK_SIZE;
                    const size_t remaining = shape_[1] > col_offset ? (shape_[1] - col_offset) : 0;
                    const size_t valid_count = std::min(static_cast<size_t>(Q5_KBlock::BLOCK_SIZE), remaining);

                    if (valid_count == 0)
                    {
                        continue;
                    }

                    if (valid_count == Q5_KBlock::BLOCK_SIZE)
                    {
                        decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[row_offset + col_offset]);
                    }
                    else
                    {
                        float temp[Q5_KBlock::BLOCK_SIZE];
                        decodeBlock(blocks[r * blocks_per_row + b], temp);
                        std::memcpy(&dequant_cache_[row_offset + col_offset], temp, valid_count * sizeof(float));
                    }
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q5_KTensor::mutable_data()
    {
        throw std::runtime_error("Q5_KTensor::mutable_data: quantized tensors are immutable");
    }

    bool Q5_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q5_KTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q5_KTensor::to_bf16(uint16_t *dst) const
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

    void Q5_KTensor::to_fp16(uint16_t *dst) const
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

    void Q5_KTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q5_KTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q5_KTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void Q5_KTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        if (output == nullptr)
        {
            throw std::invalid_argument("Q5_KTensor::decode_to_q8_0: output must not be null");
        }

        if (shape_.size() < 2)
        {
            throw std::runtime_error("Q5_KTensor::decode_to_q8_0: tensor shape is invalid");
        }

        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q5_KTensor::decode_to_q8_0: row index out of bounds");
        }

        const size_t blocks_per_row = (shape_[1] + Q5_KBlock::BLOCK_SIZE - 1) / Q5_KBlock::BLOCK_SIZE;
        const size_t sub_blocks_per_super = Q5_KBlock::BLOCK_SIZE / Q8_0Block::BLOCK_SIZE; // 256 / 32 = 8
        const size_t total_q8_blocks = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

        if (k_block_offset >= total_q8_blocks)
        {
            throw std::out_of_range("Q5_KTensor::decode_to_q8_0: block offset exceeds row length");
        }

        const size_t superblock_idx = k_block_offset / sub_blocks_per_super;
        const size_t subblock_idx = k_block_offset % sub_blocks_per_super;

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q5_KBlock *blocks = reinterpret_cast<const Q5_KBlock *>(data_ptr);
        const Q5_KBlock &block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Decode using SIMD helper (auto-dispatches to scalar/AVX2/AVX512)
        simd::decode_q5_k_to_q8_0(block, subblock_idx, output->qs, &output->d);
    }


    void Q5_KTensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const int sb_per_row = vnniSuperBlocksPerRow(ctx.K);
        const int sb_idx = b / 8;
        const int sub_idx = b % 8;
        const auto *blk = &typed_data()[static_cast<size_t>(n) * sb_per_row + sb_idx];

        const int group_idx = sub_idx / 2;
        const int is_high = sub_idx & 1;
        const uint8_t *src32 = blk->qs + group_idx * 32;

        uint8_t repacked_qs[16];
        if (is_high)
        {
            for (int i = 0; i < 16; ++i)
                repacked_qs[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
        }
        else
        {
            for (int i = 0; i < 16; ++i)
                repacked_qs[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
        }

        uint8_t repacked_qh[4] = {0, 0, 0, 0};
        for (int i = 0; i < 32; ++i)
        {
            const int bit_val = (blk->qh[i] >> sub_idx) & 1;
            repacked_qh[i / 8] |= static_cast<uint8_t>(bit_val << (i % 8));
        }

        uint8_t *dst = vnniPayloadDst(ctx, linear);
        std::memcpy(dst, repacked_qs, 16);
        std::memcpy(dst + 16, repacked_qh, 4);

        uint8_t sc, m_val;
        simd::get_scale_min_k4(sub_idx, blk->scales, &sc, &m_val);
        const float d = fp16_to_fp32(blk->d);
        const float dmin = fp16_to_fp32(blk->dmin);
        ctx.scales_array[linear] = fp32_to_fp16(d * static_cast<float>(sc));
        ctx.mins_array[linear] = fp32_to_fp16(-dmin * static_cast<float>(m_val));
    }

} // namespace llaminar2
