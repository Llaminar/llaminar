/**
 * @file Q6_KTensor.cpp
 * @brief Q6_K quantized tensor implementation (6-bit K-quant, 256-element super-blocks)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "FP16Utils.h"

#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
#include "SIMDHelpers.h"
#include "../utils/DebugEnv.h"
#include "../utils/CPUFeatures.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "../utils/Logger.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    Q6_KTensor::Q6_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
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
            throw std::invalid_argument("Q6_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q6_KBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q6_KTensor: insufficient raw data");
        }
    }

    // Private view constructor
    Q6_KTensor::Q6_KTensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<TensorBase> parent)
        : shape_(shape),
          is_view_(true),
          raw_data_(), // Empty (view borrows parent data)
          raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset),
          parent_(parent),
          device_(DeviceId::cpu()),
          device_blocks_(nullptr)
    {
    }
    // Zero-copy constructor for mmap-backed data
    Q6_KTensor::Q6_KTensor(const std::vector<size_t> &shape,
          const uint8_t *mmap_data,
          size_t byte_size,
          std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q6_KTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q6_KBlock);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("Q6_KTensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }


    std::shared_ptr<TensorBase> Q6_KTensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate: view must be 2D
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q6_KTensor::create_view: only 2D views supported");
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
            throw std::invalid_argument("Q6_KTensor::create_view: parent must be 2D or 3D");
        }

        // Validate: K dimension must match
        if (new_shape[1] != K)
        {
            throw std::invalid_argument("Q6_KTensor::create_view: K dimension must match parent");
        }

        // Validate: offset must be row-aligned
        if (offset % K != 0)
        {
            throw std::invalid_argument("Q6_KTensor::create_view: offset must be row-aligned");
        }

        // Validate: bounds check
        size_t total_elements = total_rows * K;
        if (offset + new_shape[0] * new_shape[1] > total_elements)
        {
            throw std::out_of_range("Q6_KTensor::create_view: view exceeds parent bounds");
        }

        // Calculate block offset
        const size_t blocks_per_row = (K + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const size_t first_row = offset / K;
        const size_t block_offset = first_row * blocks_per_row;
        const size_t byte_offset = block_offset * sizeof(Q6_KBlock);

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
        return std::shared_ptr<Q6_KTensor>(new Q6_KTensor(
            new_shape,
            root_data_ptr,
            cumulative_byte_offset,
            root_parent));
    }

    std::unique_ptr<ITensorGemm> Q6_KTensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void Q6_KTensor::decodeBlock(const Q6_KBlock &block, float *output)
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

    void Q6_KTensor::decodeBlockScalar(const Q6_KBlock &block, float *output)
    {
        // Q6_K: 256 elements per super-block, processed in 2 halves of 128 elements
        // Reference: llama.cpp ggml-quants.c dequantize_row_q6_K()
        // Each half uses interleaved layout with 4 values per iteration

        const float d = fp16_to_fp32(block.d);

        const uint8_t *ql = block.ql;
        const uint8_t *qh = block.qh;
        const int8_t *sc = block.scales;

        float *y = output;

        // Process 2 halves of 128 elements each
        for (size_t n = 0; n < 256; n += 128)
        {
            // Each half: 32 iterations producing 4 values each (32*4 = 128)
            for (size_t l = 0; l < 32; ++l)
            {
                const size_t is = l / 16; // Scale index (0 or 1 per half)

                // Reconstruct 4 6-bit values from ql[] and qh[]
                const int8_t q1 = static_cast<int8_t>((ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32;

                // Apply scales and output to interleaved positions
                // Scale indices: is+0, is+2, is+4, is+6
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }

            // Advance pointers for next half
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

#if defined(__AVX512F__)
    void Q6_KTensor::decodeBlockAVX512(const Q6_KBlock &block, float *output)
    {
        // AVX512 implementation for Q6_K
        // Note: Q6_K has complex bit unpacking that doesn't vectorize as cleanly
        // This implementation processes 8 elements at a time within the 32-iteration loop

        const float d = fp16_to_fp32(block.d);
        const __m512 vd = _mm512_set1_ps(d);
        const __m512i voffset = _mm512_set1_epi32(32);

        const uint8_t *ql = block.ql;
        const uint8_t *qh = block.qh;
        const int8_t *sc = block.scales;

        float *y = output;

        // Process 2 halves of 128 elements each
        for (size_t n = 0; n < 256; n += 128)
        {
            // Process in chunks of 8 iterations (32 values)
            for (size_t l = 0; l < 32; l += 8)
            {
                const size_t is = l / 16; // Scale index (0 or 1 per half)

                // Extract 8 q1 values
                __m256i vql_low = _mm256_cvtepu8_epi32(_mm_loadu_si64(&ql[l]));
                __m256i vqh_bytes = _mm256_cvtepu8_epi32(_mm_loadu_si64(&qh[l]));
                __m256i vq1_low = _mm256_and_si256(vql_low, _mm256_set1_epi32(0xF));
                __m256i vq1_high = _mm256_and_si256(vqh_bytes, _mm256_set1_epi32(0x3));
                vq1_high = _mm256_slli_epi32(vq1_high, 4);
                __m512i vq1 = _mm512_cvtepi32_epi64(_mm256_or_si256(vq1_low, vq1_high));
                vq1 = _mm512_sub_epi64(vq1, _mm512_cvtepi32_epi64(_mm256_set1_epi32(32)));

                // For simplicity in this complex format, fall back to scalar for remaining values
                // Full SIMD optimization requires significant complexity due to interleaved layout
                for (size_t i = 0; i < 8 && l + i < 32; ++i)
                {
                    const int8_t q1 = static_cast<int8_t>((ql[l + i] & 0xF) | ((qh[l + i] >> 0) & 3) << 4) - 32;
                    const int8_t q2 = static_cast<int8_t>((ql[l + i + 32] & 0xF) | ((qh[l + i] >> 2) & 3) << 4) - 32;
                    const int8_t q3 = static_cast<int8_t>((ql[l + i] >> 4) | ((qh[l + i] >> 4) & 3) << 4) - 32;
                    const int8_t q4 = static_cast<int8_t>((ql[l + i + 32] >> 4) | ((qh[l + i] >> 6) & 3) << 4) - 32;

                    y[l + i + 0] = d * sc[is + 0] * q1;
                    y[l + i + 32] = d * sc[is + 2] * q2;
                    y[l + i + 64] = d * sc[is + 4] * q3;
                    y[l + i + 96] = d * sc[is + 6] * q4;
                }
            }

            // Advance pointers for next half
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
#endif

#if defined(__AVX2__)
    void Q6_KTensor::decodeBlockAVX2(const Q6_KBlock &block, float *output)
    {
        // AVX2 implementation for Q6_K
        // Due to complex bit unpacking and interleaved layout, we use scalar fallback
        // Full AVX2 optimization would require significant code complexity for modest gains
        decodeBlockScalar(block, output);
    }
#endif

    Q6_KTensor::~Q6_KTensor() {}


    const float *Q6_KTensor::data() const
    {
        assertValid("Q6_KTensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("Q6_KTensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }

            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t row_offset = r * shape_[1];
                    const size_t col_offset = b * Q6_KBlock::BLOCK_SIZE;
                    const size_t remaining = shape_[1] > col_offset ? (shape_[1] - col_offset) : 0;
                    const size_t valid_count = std::min(static_cast<size_t>(Q6_KBlock::BLOCK_SIZE), remaining);

                    if (valid_count == 0)
                    {
                        continue;
                    }

                    if (valid_count == Q6_KBlock::BLOCK_SIZE)
                    {
                        decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[row_offset + col_offset]);
                    }
                    else
                    {
                        float temp[Q6_KBlock::BLOCK_SIZE];
                        decodeBlock(blocks[r * blocks_per_row + b], temp);
                        std::memcpy(&dequant_cache_[row_offset + col_offset], temp, valid_count * sizeof(float));
                    }
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q6_KTensor::mutable_data()
    {
        throw std::runtime_error("Q6_KTensor::mutable_data: quantized tensors are immutable");
    }

    bool Q6_KTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q6_KTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void Q6_KTensor::to_bf16(uint16_t *dst) const
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

    void Q6_KTensor::to_fp16(uint16_t *dst) const
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

    void Q6_KTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q6_KTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q6_KTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    // ========== Q8_0 Decode (for Integer GEMM) ==========

    void Q6_KTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Q6_K format: 256 elements per super-block, organized as 8 sub-blocks of 32 elements
        // Each element is 6-bit quantized (4 low bits + 2 high bits)
        // Hierarchical scaling: super-block scale (FP16) × sub-block scale (4-bit)

        constexpr size_t Q6K_SUPERBLOCK_SIZE = 256;
        constexpr size_t Q6K_SUBBLOCKS = 8; // 8 sub-blocks of 32 elements each

        // Calculate super-block index and sub-block index
        const size_t superblock_idx = (k_block_offset * 32) / Q6K_SUPERBLOCK_SIZE;
        const size_t sub_idx = k_block_offset % Q6K_SUBBLOCKS;

        // Get Q6_K super-block
        const size_t superblocks_per_row = (shape_[1] + Q6K_SUPERBLOCK_SIZE - 1) / Q6K_SUPERBLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);
        const Q6_KBlock &q6k_block = blocks[row_idx * superblocks_per_row + superblock_idx];

        simd::decode_q6_k_to_q8_0(q6k_block, sub_idx, output->qs, &output->d);
    }

    void Q6_KTensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("Q6_KTensor::unpack_superblock_to_int8: output must not be null");
        }

        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);
        const Q6_KBlock &block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_q6_k_superblock_to_int8(block, output, scales, mins);
    }

    void Q6_KTensor::unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const
    {
        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);

        // k_block_offset is in terms of 32-element sub-blocks
        // Q6_KBlock contains 8 sub-blocks (256 elements)

        const size_t super_block_idx = k_block_offset / 8;
        const size_t sub_idx = k_block_offset % 8;

        const Q6_KBlock &block = blocks[row_idx * blocks_per_row + super_block_idx];

        float scale, min;
        simd::transcode_q6_k_to_int8(block, sub_idx, output, &scale, &min);
    }

    float Q6_KTensor::get_block_scale(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);

        const size_t super_block_idx = k_block_offset / 8;
        const size_t sub_idx = k_block_offset % 8;

        const Q6_KBlock &block = blocks[row_idx * blocks_per_row + super_block_idx];

        int8_t dummy_int8[32];
        float scale, min;
        simd::transcode_q6_k_to_int8(block, sub_idx, dummy_int8, &scale, &min);
        return scale;
    }

    float Q6_KTensor::get_block_min(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + Q6_KBlock::BLOCK_SIZE - 1) / Q6_KBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q6_KBlock *blocks = reinterpret_cast<const Q6_KBlock *>(data_ptr);

        const size_t super_block_idx = k_block_offset / 8;
        const size_t sub_idx = k_block_offset % 8;

        const Q6_KBlock &block = blocks[row_idx * blocks_per_row + super_block_idx];

        int8_t dummy_int8[32];
        float scale, min;
        simd::transcode_q6_k_to_int8(block, sub_idx, dummy_int8, &scale, &min);
        return min;
    }

    void Q6_KTensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const int sb_per_row = vnniSuperBlocksPerRow(ctx.K);
        const int sb_idx = b / 8;
        const int sub_idx = b % 8;
        const auto *blk = &typed_data()[static_cast<size_t>(n) * sb_per_row + sb_idx];

        const int half = (sub_idx * 32) / 128;
        const int sub_in_half = (sub_idx * 32 % 128) / 32;
        const uint8_t *ql = blk->ql + half * 64;
        const uint8_t *qh = blk->qh + half * 32;

        uint8_t raw6[32];
        for (int l = 0; l < 32; ++l)
        {
            switch (sub_in_half)
            {
            case 0: raw6[l] = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
            case 1: raw6[l] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
            case 2: raw6[l] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
            case 3: raw6[l] = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
            }
        }

        uint8_t payload[24];
        for (int i = 0; i < 16; ++i)
            payload[i] = (raw6[i] & 0xF) | ((raw6[i + 16] & 0xF) << 4);
        for (int i = 0; i < 8; ++i)
            payload[16 + i] = ((raw6[4 * i + 0] >> 4) & 3) |
                              (((raw6[4 * i + 1] >> 4) & 3) << 2) |
                              (((raw6[4 * i + 2] >> 4) & 3) << 4) |
                              (((raw6[4 * i + 3] >> 4) & 3) << 6);

        std::memcpy(vnniPayloadDst(ctx, linear), payload, 24);

        const float d = fp16_to_fp32(blk->d);
        const int8_t *sc = blk->scales;
        const int sc_lo_idx = half * 8 + sub_in_half * 2;
        const int sc_hi_idx = sc_lo_idx + 1;
        ctx.scales_array[linear] = fp32_to_fp16(d * static_cast<float>(sc[sc_lo_idx]));
        ctx.mins_array[linear] = fp32_to_fp16(d * static_cast<float>(sc[sc_hi_idx]));
    }

} // namespace llaminar2
