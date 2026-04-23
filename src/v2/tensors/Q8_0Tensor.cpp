/**
 * @file Q8_0Tensor.cpp
 * @brief Q8_0 quantized tensor implementation (8-bit per weight, 4.0x compression)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "../kernels/KernelFactory.h"
#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../utils/Logger.h"
#include "../utils/CPUFeatures.h"
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
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
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
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        // Views don't allocate raw_data_, they borrow via raw_data_ptr_
    }
    // Zero-copy constructor for mmap-backed data
    Q8_0Tensor::Q8_0Tensor(const std::vector<size_t> &shape,
                           const uint8_t *mmap_data,
                           size_t byte_size,
                           std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Q8_0Tensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(Q8_0Block);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("Q8_0Tensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    Q8_0Tensor::~Q8_0Tensor()
    {
        // Destructor
    }

    std::shared_ptr<TensorBase> Q8_0Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Validate: view must be 2D
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("Q8_0Tensor::create_view: only 2D views supported");
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
            throw std::invalid_argument("Q8_0Tensor::create_view: parent must be 2D or 3D");
        }

        // Validation: K dimension must match
        if (new_shape[1] != K)
        {
            throw std::invalid_argument("Q8_0Tensor::create_view: K dimension must match parent");
        }

        // Validation: offset must be row-aligned
        if (offset % K != 0)
        {
            throw std::invalid_argument("Q8_0Tensor::create_view: offset must be row-aligned");
        }

        // Validation: bounds check
        size_t start_row = offset / K;
        size_t view_rows = new_shape[0];
        if (start_row + view_rows > total_rows)
        {
            throw std::out_of_range("Q8_0Tensor::create_view: view exceeds parent bounds");
        }

        // Calculate byte offset
        size_t blocks_per_row = (K + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;
        size_t block_offset = start_row * blocks_per_row;
        size_t byte_offset = block_offset * sizeof(Q8_0Block);

        // Determine root parent
        const uint8_t *root_data_ptr;
        size_t root_byte_offset;
        std::shared_ptr<TensorBase> root_parent;

        if (is_view_)
        {
            root_data_ptr = raw_data_ptr_;
            root_byte_offset = view_byte_offset_ + byte_offset;
            root_parent = parent_;
        }
        else
        {
            root_data_ptr = raw_data_.data();
            root_byte_offset = byte_offset;
            root_parent = std::static_pointer_cast<TensorBase>(shared_from_this());
        }

        return std::shared_ptr<Q8_0Tensor>(new Q8_0Tensor(
            new_shape, root_data_ptr, root_byte_offset, root_parent));
    }

    const float *Q8_0Tensor::data() const
    {
        assertValid("Q8_0Tensor::data");
        // Dequantize to temp cache
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            // If so, we cannot dequantize - return nullptr
            if (raw_data_released_)
            {
                LOG_DEBUG("Q8_0Tensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }

            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // Decode all blocks (parallelized for large tensors)
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const Q8_0Block *blocks = reinterpret_cast<const Q8_0Block *>(data_ptr);
            size_t K = shape_[1];
            size_t blocks_per_row = (K + Q8_0Block::BLOCK_SIZE - 1) / Q8_0Block::BLOCK_SIZE;

#pragma omp parallel for if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const Q8_0Block &block = blocks[r * blocks_per_row + b];
                    size_t elem_offset = b * Q8_0Block::BLOCK_SIZE;

                    // Calculate how many elements to write for this block
                    // The last block in each row may be partial (when K is not a multiple of 32)
                    size_t elem_count = std::min(static_cast<size_t>(Q8_0Block::BLOCK_SIZE), K - elem_offset);

                    float *output = &dequant_cache_[r * K + elem_offset];

                    // Use scalar decode for partial blocks, SIMD for full blocks
                    if (elem_count == Q8_0Block::BLOCK_SIZE)
                    {
                        decodeBlock(block, output);
                    }
                    else
                    {
                        // Partial block - only decode elem_count elements
                        const float scale = fp16_to_fp32(block.d);
                        for (size_t i = 0; i < elem_count; ++i)
                        {
                            output[i] = scale * static_cast<float>(block.qs[i]);
                        }
                    }
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
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
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

    // ===== Efficient requantizeRowToInt8 for Q8_0 =====

    namespace detail
    {
        float q8_0_find_max_abs_scalar(const Q8_0Block *row_blocks, size_t blocks_per_row)
        {
            float max_abs = 0.0f;
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const float block_scale = fp16_to_fp32(row_blocks[b].d);
                int max_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    const int abs_qs = row_blocks[b].qs[i] < 0
                                           ? -row_blocks[b].qs[i]
                                           : row_blocks[b].qs[i];
                    if (abs_qs > max_qs)
                        max_qs = abs_qs;
                }
                const float val = block_scale * static_cast<float>(max_qs);
                if (val > max_abs)
                    max_abs = val;
            }
            return max_abs;
        }

#if defined(__AVX2__)
        float q8_0_find_max_abs_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row)
        {
            float max_abs = 0.0f;
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(row_blocks[b].qs));
                const __m256i abs_v = _mm256_abs_epi8(v);
                const __m128i lo = _mm256_castsi256_si128(abs_v);
                const __m128i hi = _mm256_extracti128_si256(abs_v, 1);
                __m128i mx = _mm_max_epu8(lo, hi);
                mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 8));
                mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 4));
                mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 2));
                mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 1));
                const float val = fp16_to_fp32(row_blocks[b].d) *
                                  static_cast<float>(_mm_extract_epi8(mx, 0));
                if (val > max_abs)
                    max_abs = val;
            }
            return max_abs;
        }
#endif

        void q8_0_requantize_scalar(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                    float inv_row_scale, int8_t *output)
        {
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const float rescale = fp16_to_fp32(row_blocks[b].d) * inv_row_scale;
                int8_t *dst = output + b * 32;
                for (int i = 0; i < 32; ++i)
                {
                    const float val = static_cast<float>(row_blocks[b].qs[i]) * rescale;
                    int32_t q = static_cast<int32_t>(val + (val >= 0.0f ? 0.5f : -0.5f));
                    q = q < -127 ? -127 : (q > 127 ? 127 : q);
                    dst[i] = static_cast<int8_t>(q);
                }
            }
        }

#if defined(__AVX2__)
        void q8_0_requantize_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                  float inv_row_scale, int8_t *output)
        {
            const __m256i clamp_lo = _mm256_set1_epi32(-127);
            const __m256i clamp_hi = _mm256_set1_epi32(127);
            const __m256i pack_perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const __m256 vrescale = _mm256_set1_ps(
                    fp16_to_fp32(row_blocks[b].d) * inv_row_scale);
                int8_t *dst = output + b * 32;
                __m256i groups[4];
                for (int q = 0; q < 4; ++q)
                {
                    const __m128i v8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(
                            row_blocks[b].qs + q * 8));
                    __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8));
                    vf = _mm256_mul_ps(vf, vrescale);
                    __m256i r = _mm256_cvtps_epi32(vf);
                    r = _mm256_max_epi32(r, clamp_lo);
                    groups[q] = _mm256_min_epi32(r, clamp_hi);
                }
                const __m256i p01 = _mm256_packs_epi32(groups[0], groups[1]);
                const __m256i p23 = _mm256_packs_epi32(groups[2], groups[3]);
                __m256i packed = _mm256_packs_epi16(p01, p23);
                packed = _mm256_permutevar8x32_epi32(packed, pack_perm);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), packed);
            }
        }
#endif

#if defined(__AVX512F__)
        void q8_0_requantize_avx512(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                    float inv_row_scale, int8_t *output)
        {
            const __m512i clamp_lo = _mm512_set1_epi32(-127);
            const __m512i clamp_hi = _mm512_set1_epi32(127);
            for (size_t b = 0; b < blocks_per_row; ++b)
            {
                const __m512 vrescale = _mm512_set1_ps(
                    fp16_to_fp32(row_blocks[b].d) * inv_row_scale);
                int8_t *dst = output + b * 32;
                for (int half = 0; half < 2; ++half)
                {
                    const __m128i v8 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(
                            row_blocks[b].qs + half * 16));
                    __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v8));
                    vf = _mm512_mul_ps(vf, vrescale);
                    __m512i r = _mm512_cvtps_epi32(vf);
                    r = _mm512_max_epi32(r, clamp_lo);
                    r = _mm512_min_epi32(r, clamp_hi);
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i *>(dst + half * 16),
                        _mm512_cvtsepi32_epi8(r));
                }
            }
        }
#endif

// --- ISA stubs for runtime dispatch ---
#if !defined(__AVX2__)
        float q8_0_find_max_abs_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row)
        {
            return q8_0_find_max_abs_scalar(row_blocks, blocks_per_row);
        }
        void q8_0_requantize_avx2(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                  float inv_row_scale, int8_t *output)
        {
            q8_0_requantize_scalar(row_blocks, blocks_per_row, inv_row_scale, output);
        }
#endif
        float q8_0_find_max_abs_avx512(const Q8_0Block *row_blocks, size_t blocks_per_row)
        {
            return q8_0_find_max_abs_avx2(row_blocks, blocks_per_row);
        }
#if !defined(__AVX512F__)
        void q8_0_requantize_avx512(const Q8_0Block *row_blocks, size_t blocks_per_row,
                                    float inv_row_scale, int8_t *output)
        {
            q8_0_requantize_avx2(row_blocks, blocks_per_row, inv_row_scale, output);
        }
#endif
    } // namespace detail
    using namespace detail;

    float Q8_0Tensor::requantizeRowToInt8(size_t row_idx, size_t K, int8_t *output) const
    {
        const size_t blocks_per_row = K / Q8_0Block::BLOCK_SIZE;
        const Q8_0Block *all_blocks = typed_data();
        const Q8_0Block *row_blocks = all_blocks + row_idx * blocks_per_row;

        // Phase 1: Find max absolute dequantized value across all blocks
        float max_abs = ISA_DISPATCH_RETVAL(q8_0_find_max_abs, row_blocks, blocks_per_row);

        const float row_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        const float inv_row_scale = 1.0f / row_scale;

        // Phase 2: Requantize each block
        ISA_DISPATCH_VOID(q8_0_requantize, row_blocks, blocks_per_row, inv_row_scale, output);

        return row_scale;
    }

    void Q8_0Tensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const auto *blk = &typed_data()[static_cast<size_t>(n) * ctx.blocks_per_row + b];
        std::memcpy(vnniPayloadDst(ctx, linear), blk->qs, 32);
        ctx.scales_array[linear] = blk->d;
    }

    // ===== Static FP32 → Q8_0 quantization =====

    void Q8_0Tensor::quantize_fp32_row(const float *src, Q8_0Block *dst, size_t count)
    {
        const size_t n_blocks = (count + 31) / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const size_t offset = b * 32;
            const size_t block_len = std::min<size_t>(32, count - offset);
            const float *block_src = src + offset;
            Q8_0Block &block = dst[b];

            float max_abs = 0.0f;
            for (size_t i = 0; i < block_len; ++i)
                max_abs = std::max(max_abs, std::abs(block_src[i]));

            if (max_abs < 1e-30f)
            {
                block.d = 0;
                std::memset(block.qs, 0, 32);
                continue;
            }

            const float scale = max_abs / 127.0f;
            block.d = fp32_to_fp16(scale);
            const float inv_scale = 127.0f / max_abs;

            for (size_t i = 0; i < block_len; ++i)
            {
                float v = block_src[i] * inv_scale;
                v = std::max(-127.0f, std::min(127.0f, v));
                block.qs[i] = static_cast<int8_t>(std::round(v));
            }
            for (size_t i = block_len; i < 32; ++i)
                block.qs[i] = 0;
        }
    }

} // namespace llaminar2
