/**
 * @file IQ4_NLTensor.cpp
 * @brief IQ4_NL quantized tensor implementation
 *
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "FP16Utils.h"
#include "../utils/CPUFeatures.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "SIMDHelpers.h"
#include "../backends/ComputeBackend.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

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
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
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

    // Private view constructor
    IQ4_NLTensor::IQ4_NLTensor(const std::vector<size_t> &shape,
                               const uint8_t *parent_raw_data,
                               size_t byte_offset,
                               std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(parent_raw_data),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        // Views don't allocate raw_data_, they borrow via raw_data_ptr_
    }
    // Zero-copy constructor for mmap-backed data
    IQ4_NLTensor::IQ4_NLTensor(const std::vector<size_t> &shape,
          const uint8_t *mmap_data,
          size_t byte_size,
          std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ4_NLTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ4_NLBlock);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("IQ4_NLTensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }


    IQ4_NLTensor::~IQ4_NLTensor()
    {
        // TODO: Free device_blocks_ if allocated
        if (device_blocks_)
        {
            LOG_DEBUG("[IQ4_NLTensor] TODO: Free device blocks in destructor");
        }
    }

    // ========== View Support ==========

    std::shared_ptr<TensorBase> IQ4_NLTensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // 1. Validate new_shape is 2D
        if (new_shape.size() != 2)
        {
            LOG_ERROR("[IQ4_NLTensor::create_view] ERROR: View must be 2D (got "
                      << new_shape.size() << "D)");
            return nullptr;
        }

        // 2. Validate K dimension matches parent (row-slice restriction)
        if (new_shape[1] != shape_[1])
        {
            LOG_ERROR("[IQ4_NLTensor::create_view] ERROR: View must preserve K dimension (column count)\n"
                      << "  Parent K: " << shape_[1] << ", View K: " << new_shape[1]);
            return nullptr;
        }

        size_t K = shape_[1];

        // 3. Validate offset is row-aligned
        if (offset % K != 0)
        {
            LOG_ERROR("[IQ4_NLTensor::create_view] ERROR: Offset must be row-aligned (multiple of K="
                      << K << ")\n"
                      << "  Got offset: " << offset << " (not divisible by " << K << ")");
            return nullptr;
        }

        // 4. Validate bounds
        size_t start_row = offset / K;
        size_t view_rows = new_shape[0];
        if (start_row + view_rows > shape_[0])
        {
            LOG_ERROR("[IQ4_NLTensor::create_view] ERROR: View exceeds parent bounds\n"
                      << "  Parent rows: " << shape_[0] << ", Start row: " << start_row
                      << ", View rows: " << view_rows);
            return nullptr;
        }

        // 5. Calculate byte offset in raw_data_
        size_t blocks_per_row = (K + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        size_t block_offset = start_row * blocks_per_row;
        size_t byte_offset = block_offset * sizeof(IQ4_NLBlock);

        // 6. Determine root parent and data pointer
        std::shared_ptr<TensorBase> root_parent;
        const uint8_t *root_data_ptr;
        size_t root_byte_offset;

        if (is_view_)
        {
            // Chain to existing parent
            root_parent = parent_;
            root_data_ptr = raw_data_ptr_;
            root_byte_offset = view_byte_offset_ + byte_offset;
        }
        else
        {
            // This is the root parent
            try
            {
                root_parent = shared_from_this();
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[IQ4_NLTensor::create_view] ERROR: shared_from_this() failed - "
                          << "object not managed by shared_ptr!\n"
                          << "  Exception: " << e.what());
                return nullptr;
            }
            root_data_ptr = raw_data_.data();
            root_byte_offset = byte_offset;
        }

        // 7. Create view using private constructor
        auto view_tensor = std::shared_ptr<IQ4_NLTensor>(new IQ4_NLTensor(
            new_shape,
            root_data_ptr,
            root_byte_offset,
            root_parent));

        return view_tensor;
    }

    // ========== Shape and Metadata ==========

    size_t IQ4_NLTensor::padded_k() const
    {
        size_t cols = logical_k();
        return ((cols + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE) * IQ4_NLBlock::BLOCK_SIZE;
    }

    // ========== Device Management ==========


    // ========== Data Access ==========

    const float *IQ4_NLTensor::data() const
    {
        assertValid("IQ4_NLTensor::data");
        // Check if raw data was released after GEMM packing
        // If so, we cannot dequantize - return nullptr
        if (raw_data_released_)
        {
            LOG_DEBUG("IQ4_NLTensor::data() called but raw data was released after GEMM packing");
            return nullptr;
        }

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

    // ========== IINT8Unpackable Implementation ==========

    void IQ4_NLTensor::unpack_block_to_int8(size_t row_idx, size_t k_block_offset, int8_t *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
        const IQ4_NLBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];

        simd::unpack_iq4_nl_to_int8(block, output);
    }

    float IQ4_NLTensor::get_block_scale(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
        return simd::fp16_to_fp32(blocks[row_idx * blocks_per_row + k_block_offset].d);
    }

    float IQ4_NLTensor::get_block_min(size_t row_idx, size_t k_block_offset) const
    {
        // IQ4_NL is symmetric around 0 (mostly), no min offset
        return 0.0f;
    }

    // ========== Kernel Creation ==========

    std::unique_ptr<ITensorGemm> IQ4_NLTensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    ITensorGemm *IQ4_NLTensor::createGemmRaw()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemmRaw(this, dev_type);
    }

    // ========== Decode API ==========

    void IQ4_NLTensor::decode_to_fp32(float *dst) const
    {
        const size_t rows = shape_[0];
        const size_t cols = shape_[1];
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
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
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
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
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
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

        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);

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

    void IQ4_NLTensor::to_fp16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to FP16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        decode_to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(temp_fp32[i]);
        }
    }

    void IQ4_NLTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Decode to FP32 first, then quantize to int8
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        decode_to_fp32(temp_fp32.data());

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

    bool IQ4_NLTensor::to_int8_perchannel(
        int8_t *dst_int8,
        float *dst_col_scales,
        float *dst_row_scales) const
    {
        return to_int8_perchannel_via_blocks(dst_int8, dst_col_scales, dst_row_scales);
    }

    // ========== Fused Kernel Helpers ==========

    const IQ4_NLBlock &IQ4_NLTensor::get_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
        const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
        return blocks[block_idx];
    }

    void IQ4_NLTensor::decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);

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
            decodeBlockScalar(block, output);
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
        decodeBlockScalar(block, output);
    }

    void IQ4_NLTensor::decodeBlockScalar(const IQ4_NLBlock &block, float *output)
    {
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
        LOG_DEBUG("[IQ4_NLTensor] Microkernel not yet implemented, using standard path");

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
        LOG_ERROR("[IQ4_NLTensor::copyFrom] Not implemented");
        return false;
    }

    // ========== Q8_0 Decode (for Integer GEMM) ==========

    void IQ4_NLTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Get IQ4_NL block
        const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(data_ptr);
        const IQ4_NLBlock &iq4_block = blocks[row_idx * blocks_per_row + k_block_offset];

        // Use vectorized decode (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_iq4nl_to_q8_0(
            iq4_block.qs, // Input: packed 4-bit indices
            iq4_block.d,  // Input: IQ4_NL FP16 scale
            output->qs,   // Output: Q8_0 int8 values
            &output->d);  // Output: Q8_0 FP16 scale
    }


    void IQ4_NLTensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const auto *blk = &typed_data()[static_cast<size_t>(n) * ctx.blocks_per_row + b];
        std::memcpy(vnniPayloadDst(ctx, linear), blk->qs, 16);
        ctx.scales_array[linear] = blk->d;
    }

} // namespace llaminar2
