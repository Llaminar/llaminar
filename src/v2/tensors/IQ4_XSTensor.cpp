/**
 * @file IQ4_XSTensor.cpp
 * @brief IQ4_XS quantized tensor implementation (4-bit extra-small IQ, 32-element blocks)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "../utils/CPUFeatures.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "SIMDHelpers.h"
#include <cstring>
#include <stdexcept>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#endif

namespace llaminar2
{

    // Note: IQ4_XS uses the same codebook as IQ4_NL (kvalues_iq4nl from IQQuantTables.h)
    // If IQ4_XS requires a different codebook, it should be added to IQQuantTables.h

    IQ4_XSTensor::IQ4_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr),
          view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ4_XSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ4_XSBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ4_XSTensor: insufficient raw data");
        }
    }

    // Private view constructor (borrows parent data)
    IQ4_XSTensor::IQ4_XSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                               size_t view_byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(view_byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
    }
    // Zero-copy constructor for mmap-backed data
    IQ4_XSTensor::IQ4_XSTensor(const std::vector<size_t> &shape,
                               const uint8_t *mmap_data,
                               size_t byte_size,
                               std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ4_XSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ4_XSBlock);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("IQ4_XSTensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }

    // View creation (row-slice only - preserves K dimension)
    std::shared_ptr<TensorBase> IQ4_XSTensor::create_view(
        const std::vector<size_t> &new_shape, size_t offset)
    {
        if (new_shape.size() != 2)
        {
            throw std::invalid_argument("IQ4_XSTensor::create_view: only 2D views supported");
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
            throw std::invalid_argument("IQ4_XSTensor::create_view: parent must be 2D or 3D");
        }

        if (new_shape[1] != K)
        {
            throw std::invalid_argument("IQ4_XSTensor::create_view: K dimension must match parent");
        }

        // Offset must be row-aligned
        if (offset % K != 0)
        {
            throw std::invalid_argument("IQ4_XSTensor::create_view: offset must be row-aligned");
        }

        const size_t offset_rows = offset / K;
        const size_t view_rows = new_shape[0];

        if (offset_rows + view_rows > total_rows)
        {
            throw std::out_of_range("IQ4_XSTensor::create_view: view exceeds parent bounds");
        }

        // Compute byte offset (blocks are row-major)
        const size_t blocks_per_row = (K + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        const size_t bytes_per_row = blocks_per_row * sizeof(IQ4_XSBlock);
        const size_t new_byte_offset = offset_rows * bytes_per_row;

        // Get raw data pointer from parent (or from this if we're the parent)
        const uint8_t *parent_data_ptr = is_view_ ? raw_data_ptr_ : raw_data_.data();
        const size_t cumulative_offset = is_view_ ? (view_byte_offset_ + new_byte_offset) : new_byte_offset;

        // Keep the ultimate parent alive (not intermediate views)
        std::shared_ptr<TensorBase> ultimate_parent = is_view_ ? parent_ : shared_from_this();

        return std::shared_ptr<TensorBase>(
            new IQ4_XSTensor(new_shape, parent_data_ptr, cumulative_offset, ultimate_parent));
    }

    std::unique_ptr<ITensorGemm> IQ4_XSTensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void IQ4_XSTensor::decodeBlock(const IQ4_XSBlock &block, float *output)
    {
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

    void IQ4_XSTensor::decodeBlockScalar(const IQ4_XSBlock &block, float *output)
    {
        // IQ4_XS: 256 elements per super-block, using kvalues_iq4nl codebook
        // Matches llama.cpp's dequantize_row_iq4_xs implementation
        const float d = fp16_to_fp32(block.d);
        const uint8_t *qs = block.qs;

        // Process in 8 iterations (QK_K/32 = 256/32 = 8)
        // Each iteration handles 32 elements (16 pairs of 4-bit values)
        for (size_t ib = 0; ib < 8; ++ib)
        {
            // Extract 6-bit scale: 4 bits from scales_l + 2 bits from scales_h
            const int ls = ((block.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((block.scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);

            // Decode 16 pairs of 4-bit values using kvalues_iq4nl lookup
            for (size_t j = 0; j < 16; ++j)
            {
                output[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                output[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            output += 32;
            qs += 16;
        }
    }

#ifdef __AVX512F__
    void IQ4_XSTensor::decodeBlockAVX512(const IQ4_XSBlock &block, float *output)
    {
        // For now, use scalar implementation
        // TODO: Implement AVX512 vectorized version
        decodeBlockScalar(block, output);
    }
#endif

#ifdef __AVX2__
    void IQ4_XSTensor::decodeBlockAVX2(const IQ4_XSBlock &block, float *output)
    {
        // For now, use scalar implementation
        // TODO: Implement AVX2 vectorized version
        decodeBlockScalar(block, output);
    }
#endif

    IQ4_XSTensor::~IQ4_XSTensor() {}

    const float *IQ4_XSTensor::data() const
    {
        assertValid("IQ4_XSTensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("IQ4_XSTensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // View-aware data pointer selection
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);

            size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t row_offset = r * shape_[1];
                    const size_t col_offset = b * IQ4_XSBlock::BLOCK_SIZE;
                    const size_t remaining = shape_[1] > col_offset ? (shape_[1] - col_offset) : 0;
                    const size_t valid_count = std::min(static_cast<size_t>(IQ4_XSBlock::BLOCK_SIZE), remaining);

                    if (valid_count == 0)
                    {
                        continue;
                    }

                    if (valid_count == IQ4_XSBlock::BLOCK_SIZE)
                    {
                        decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[row_offset + col_offset]);
                    }
                    else
                    {
                        float temp[IQ4_XSBlock::BLOCK_SIZE];
                        decodeBlock(blocks[r * blocks_per_row + b], temp);
                        std::memcpy(&dequant_cache_[row_offset + col_offset], temp, valid_count * sizeof(float));
                    }
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ4_XSTensor::mutable_data()
    {
        throw std::runtime_error("IQ4_XSTensor::mutable_data: quantized tensors are immutable");
    }

    bool IQ4_XSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[IQ4_XSTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void IQ4_XSTensor::to_bf16(uint16_t *dst) const
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

    void IQ4_XSTensor::to_fp16(uint16_t *dst) const
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

    void IQ4_XSTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void IQ4_XSTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void IQ4_XSTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void IQ4_XSTensor::decode_to_q8_0(
        size_t row_idx,
        size_t k_block_offset,
        Q8_0Block *output) const
    {
        if (shape_.size() != 2)
        {
            throw std::runtime_error("decode_to_q8_0 expects 2D tensor");
        }
        const size_t cols = shape_[1];
        const size_t blocks_per_row = (cols + block_size() - 1) / block_size();

        if (k_block_offset >= blocks_per_row)
        {
            throw std::out_of_range("decode_to_q8_0: k_block_offset out of range");
        }

        // Calculate super-block index (256 elements)
        const size_t super_blocks_per_row = (cols + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        const size_t super_block_idx = (row_idx * super_blocks_per_row) + (k_block_offset / 8);
        const size_t sub_block_idx = k_block_offset % 8;

        // Get IQ4_XS super-block
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);
        const IQ4_XSBlock &iq4xs_block = blocks[super_block_idx];

        // Decode ONLY the requested sub-block
        simd::decode_iq4xs_to_q8_0(iq4xs_block, sub_block_idx,
                                   output->qs, &output->d);
    }

    void IQ4_XSTensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("IQ4_XSTensor::unpack_superblock_to_int8: output must not be null");
        }

        const size_t blocks_per_row = (shape_[1] + IQ4_XSBlock::BLOCK_SIZE - 1) / IQ4_XSBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ4_XSBlock *blocks = reinterpret_cast<const IQ4_XSBlock *>(data_ptr);
        const IQ4_XSBlock &super_block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Use optimized fused implementation
        simd::unpack_iq4_xs_superblock_to_int8(super_block, output, scales);

        // Symmetric format - no min values
        if (mins)
            std::memset(mins, 0, 8 * sizeof(float));
    }

    void IQ4_XSTensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const int sb_per_row = vnniSuperBlocksPerRow(ctx.K);
        const int sb_idx = b / 8;
        const int sub_idx = b % 8;
        const auto *blk = &typed_data()[static_cast<size_t>(n) * sb_per_row + sb_idx];

        std::memcpy(vnniPayloadDst(ctx, linear), blk->qs + sub_idx * 16, 16);

        const int ls = ((blk->scales_l[sub_idx / 2] >> (4 * (sub_idx % 2))) & 0xf) |
                       (((blk->scales_h >> (2 * sub_idx)) & 3) << 4);
        const float combined_scale = fp16_to_fp32(blk->d) * static_cast<float>(ls - 32);
        ctx.scales_array[linear] = fp32_to_fp16(combined_scale);
    }

} // namespace llaminar2
