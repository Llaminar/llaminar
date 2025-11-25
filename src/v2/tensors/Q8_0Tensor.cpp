/**
 * @file Q8_0Tensor.cpp
 * @brief Q8_0 quantized tensor implementation (8-bit per weight, 4.0x compression)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "../kernels/cpu/gemm_v4/Q8_1GemmKernel.h"
#include "Tensors.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#endif

namespace llaminar2
{

    Q8_0Tensor::Q8_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_0Tensor: shape cannot be empty");
        }

        // Calculate total elements
        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        // Each Q8_0Block contains 32 elements
        size_t n_blocks = (n_elems + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q8_0Block);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("Q8_0Tensor: insufficient raw data (" +
                                        std::to_string(raw_data_.size()) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    // Private view constructor
    Q8_0Tensor::Q8_0Tensor(const std::vector<size_t> &shape,
                           const uint8_t *parent_raw_data,
                           size_t byte_offset,
                           std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset), parent_(parent), device_idx_(-1), device_blocks_(nullptr)
    {
        // Views don't allocate raw_data_, they borrow via raw_data_ptr_
    }

    Q8_0Tensor::~Q8_0Tensor()
    {
        // Destructor
    }

    std::shared_ptr<TensorBase> Q8_0Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate 2D shape
        if (new_shape.size() != 2)
        {
            LOG_ERROR("[Q8_0Tensor::create_view] ERROR: View must be 2D (got "
                      << new_shape.size() << "D)");
            return nullptr;
        }

        // Validate K dimension matches (row-slice only)
        if (new_shape[1] != shape_[1])
        {
            LOG_ERROR("[Q8_0Tensor::create_view] ERROR: View must preserve K dimension\n"
                      << "  Parent K: " << shape_[1] << ", View K: " << new_shape[1]);
            return nullptr;
        }

        size_t K = shape_[1];

        // Validate offset is row-aligned
        if (offset % K != 0)
        {
            LOG_ERROR("[Q8_0Tensor::create_view] ERROR: Offset must be row-aligned (multiple of K="
                      << K << ")\n"
                      << "  Got offset: " << offset);
            return nullptr;
        }

        // Validate bounds
        size_t start_row = offset / K;
        size_t view_rows = new_shape[0];
        if (start_row + view_rows > shape_[0])
        {
            LOG_ERROR("[Q8_0Tensor::create_view] ERROR: View exceeds parent bounds\n"
                      << "  Parent rows: " << shape_[0] << ", Start row: " << start_row
                      << ", View rows: " << view_rows);
            return nullptr;
        }

        // Calculate byte offset
        size_t blocks_per_row = (K + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t block_offset = start_row * blocks_per_row;
        size_t byte_offset = block_offset * sizeof(Q8_0Block);

        // Determine root parent
        std::shared_ptr<TensorBase> root_parent;
        const uint8_t *root_data_ptr;
        size_t root_byte_offset;

        if (is_view_)
        {
            root_parent = parent_;
            root_data_ptr = raw_data_ptr_;
            root_byte_offset = view_byte_offset_ + byte_offset;
        }
        else
        {
            try
            {
                root_parent = shared_from_this();
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[Q8_0Tensor::create_view] ERROR: shared_from_this() failed");
                return nullptr;
            }
            root_data_ptr = raw_data_.data();
            root_byte_offset = byte_offset;
        }

        return std::shared_ptr<Q8_0Tensor>(new Q8_0Tensor(
            new_shape, root_data_ptr, root_byte_offset, root_parent));
    }

    bool Q8_0Tensor::set_device(int device_idx)
    {
        // TODO: Implement device transfer
        device_idx_ = device_idx;
        return true;
    }

    const float *Q8_0Tensor::data() const
    {
        // Dequantize to temp cache
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // Decode all blocks (parallelized for large tensors)
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

#pragma omp parallel for collapse(2) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const Q8_0Block &block = blocks[r * blocks_per_row + b];
                    decodeBlock(block, &dequant_cache_[r * shape_[1] + b * Q8_0Block::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *Q8_0Tensor::mutable_data()
    {
        throw std::runtime_error("Q8_0Tensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<ITensorGemm> Q8_0Tensor::createGemm()
    {
        // Use optimized Q8_1GemmKernel (repacks Q8_0 weights to Q8_1 format internally)
        return std::make_unique<llaminar2::gemm_v4::Q8_1GemmKernel>(this);
    }

    void Q8_0Tensor::decodeBlockScalar(const Q8_0Block &block, float *output)
    {
        // Scalar implementation for Q8_0: simple scale * int8 value
        const float scale = fp16_to_fp32(block.d);
        for (size_t i = 0; i < Q8_0Block::BLOCK_SIZE; ++i)
        {
            output[i] = scale * static_cast<float>(block.qs[i]);
        }
    }

    void Q8_0Tensor::decodeBlock(const Q8_0Block &block, float *output)
    {
#if defined(__AVX512F__)
        decodeBlockAVX512(block, output);
#elif defined(__AVX2__)
        decodeBlockAVX2(block, output);
#else
        decodeBlockScalar(block, output);
#endif
    }

#if defined(__AVX512F__)
    void Q8_0Tensor::decodeBlockAVX512(const Q8_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m512 vscale = _mm512_set1_ps(scale);

        // Process 16 int8 values at a time (2 iterations for 32 elements)
        for (size_t i = 0; i < 2; ++i)
        {
            // Load 16 int8 values and convert to int32
            __m128i vi8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block.qs[i * 16]));
            __m512i vi32 = _mm512_cvtepi8_epi32(vi8);

            // Convert to float and scale
            __m512 vf = _mm512_cvtepi32_ps(vi32);
            vf = _mm512_mul_ps(vf, vscale);

            // Store result
            _mm512_storeu_ps(&output[i * 16], vf);
        }
    }
#endif

#if defined(__AVX2__)
    void Q8_0Tensor::decodeBlockAVX2(const Q8_0Block &block, float *output)
    {
        const float scale = fp16_to_fp32(block.d);
        const __m256 vscale = _mm256_set1_ps(scale);

        // Process 8 int8 values at a time (4 iterations for 32 elements)
        for (size_t i = 0; i < 4; ++i)
        {
            // Load 8 int8 values from memory
            __m128i vi8_half = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&block.qs[i * 8]));

            // Sign-extend int8 → int32 (8 elements)
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8_half);

            // Convert int32 → float
            __m256 vf = _mm256_cvtepi32_ps(vi32);

            // Multiply by scale
            vf = _mm256_mul_ps(vf, vscale);

            // Store result
            _mm256_storeu_ps(&output[i * 8], vf);
        }
    }
#endif

    bool Q8_0Tensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[Q8_0Tensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    bool Q8_0Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
    }

    bool Q8_0Tensor::to_int8_rowmajor(int8_t *dst_int8, float *dst_row_scales) const
    {
        if (!dst_int8 || !dst_row_scales)
        {
            LOG_ERROR("[Q8_0Tensor] to_int8_rowmajor requires non-null output buffers");
            return false;
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[Q8_0Tensor] to_int8_rowmajor requires 2D tensor, got " << shp.size() << "D");
            return false;
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        if (rows == 0 || cols == 0)
        {
            return true;
        }

        const size_t blocks_per_row = (cols + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);

        // Pass 1: compute per-row max absolute value
        for (size_t row = 0; row < rows; ++row)
        {
            const Q8_0Block *row_blocks = blocks + row * blocks_per_row;
            float max_abs = 0.0f;
            size_t processed_cols = 0;

            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const Q8_0Block &block = row_blocks[b];
                const float block_scale = fp16_to_fp32(block.d);
                const size_t valid = std::min(static_cast<size_t>(Q8_0Block::BLOCK_SIZE), cols - processed_cols);

                for (size_t i = 0; i < valid; ++i)
                {
                    const float abs_val = std::fabs(block_scale * static_cast<float>(block.qs[i]));
                    max_abs = std::max(max_abs, abs_val);
                }

                processed_cols += valid;
            }

            dst_row_scales[row] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Pass 2: quantize each row using its scale
        for (size_t row = 0; row < rows; ++row)
        {
            const Q8_0Block *row_blocks = blocks + row * blocks_per_row;
            int8_t *row_dst = dst_int8 + row * cols;
            const float inv_scale = (dst_row_scales[row] > 0.0f) ? (1.0f / dst_row_scales[row]) : 0.0f;

            if (inv_scale == 0.0f)
            {
                std::memset(row_dst, 0, cols);
                continue;
            }

            size_t processed_cols = 0;
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const Q8_0Block &block = row_blocks[b];
                const float block_scale = fp16_to_fp32(block.d) * inv_scale;
                const size_t valid = std::min(static_cast<size_t>(Q8_0Block::BLOCK_SIZE), cols - processed_cols);

                for (size_t i = 0; i < valid; ++i)
                {
                    const float scaled = static_cast<float>(block.qs[i]) * block_scale;
                    int32_t quantized = static_cast<int32_t>(std::round(scaled));
                    quantized = std::max(-127, std::min(127, quantized));
                    row_dst[processed_cols + i] = static_cast<int8_t>(quantized);
                }

                processed_cols += valid;
            }
        }

        return true;
    }

    void Q8_0Tensor::to_bf16(uint16_t *dst) const
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

    void Q8_0Tensor::to_fp16(uint16_t *dst) const
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

    void Q8_0Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void Q8_0Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void Q8_0Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void Q8_0Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Q8_0 is already in Q8_0 format - just copy the block directly
        const size_t blocks_per_row = (shape_[1] + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("Q8_0Tensor::decode_to_q8_0: row_idx out of range");
        }
        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("Q8_0Tensor::decode_to_q8_0: k_block_offset exceeds blocks per row");
        }

        // Get pointer to source Q8_0 block
        const uint8_t *data_ptr = is_view_ ? raw_data_ptr_ + view_byte_offset_ : raw_data_.data();
        const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
        const Q8_0Block &source_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Direct copy - Q8_0 to Q8_0 (no conversion needed)
        *output = source_block;
    }

} // namespace llaminar2
