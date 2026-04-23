/**
 * @file IQ1_MTensor.cpp
 * @brief IQ1_M quantized tensor implementation (1-bit medium IQ)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
#include "VnniPackContext.h"
#include "../kernels/KernelFactory.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include <cstring>
#include <stdexcept>
#include "../utils/Logger.h"
#include "../utils/CPUFeatures.h"
#include "../utils/DebugEnv.h"
#include "SIMDHelpers.h"

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

    IQ1_MTensor::IQ1_MTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr), view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ1_MTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ1_MBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ1_MTensor: insufficient raw data");
        }
    }

    IQ1_MTensor::IQ1_MTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                             size_t byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr) {}
    // Zero-copy constructor for mmap-backed data
    IQ1_MTensor::IQ1_MTensor(const std::vector<size_t> &shape,
          const uint8_t *mmap_data,
          size_t byte_size,
          std::shared_ptr<void> mmap_lifetime_owner)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(mmap_data),
          view_byte_offset_(0), parent_(nullptr), mmap_owner_(std::move(mmap_lifetime_owner)),
          data_byte_size_(byte_size), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ1_MTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ1_MBlock);

        if (byte_size < expected_bytes)
        {
            throw std::invalid_argument("IQ1_MTensor: insufficient mmap data (" +
                                        std::to_string(byte_size) + " bytes, expected " +
                                        std::to_string(expected_bytes) + ")");
        }
    }


    std::shared_ptr<TensorBase> IQ1_MTensor::create_view(
        const std::vector<size_t> &view_shape, size_t offset_elements)
    {
        if (view_shape.size() != 2)
            throw std::invalid_argument("IQ1_MTensor::create_view: only 2D views supported");

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
            throw std::invalid_argument("IQ1_MTensor::create_view: parent must be 2D or 3D");
        }

        if (view_shape[1] != K)
            throw std::invalid_argument("IQ1_MTensor::create_view: K dimension must match parent");
        if (offset_elements % K != 0)
            throw std::invalid_argument("IQ1_MTensor::create_view: offset must be row-aligned");

        size_t offset_rows = offset_elements / K;
        size_t view_end_row = offset_rows + view_shape[0];
        if (view_end_row > total_rows)
            throw std::out_of_range("IQ1_MTensor::create_view: view exceeds parent bounds");

        size_t blocks_per_row = (K + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        size_t byte_offset_in_parent = offset_rows * blocks_per_row * sizeof(IQ1_MBlock);
        size_t new_total_byte_offset = view_byte_offset_ + byte_offset_in_parent;

        const uint8_t *parent_data_ptr = is_view_ ? raw_data_ptr_ : raw_data_.data();

        return std::shared_ptr<TensorBase>(
            new IQ1_MTensor(view_shape, parent_data_ptr, new_total_byte_offset,
                            is_view_ ? parent_ : shared_from_this()));
    }

    std::unique_ptr<ITensorGemm> IQ1_MTensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void IQ1_MTensor::decodeBlock(const IQ1_MBlock &block, float *output)
    {
#ifdef __AVX512F__
        if (simd::cpu_supports_avx512())
        {
            decodeBlockAVX512(block, output);
            return;
        }
#endif

#ifdef __AVX2__
        if (simd::cpu_supports_avx2())
        {
            decodeBlockAVX2(block, output);
            return;
        }
#endif

        decodeBlockScalar(block, output);
    }

    void IQ1_MTensor::decodeBlockScalar(const IQ1_MBlock &block, float *output)
    {
        // IQ1_M: 256 elements per super-block, using iq1s_grid[2048]
        // Matches llama.cpp's dequantize_row_iq1_m implementation

        // Extract global FP16 scale from packed scales array
        const uint16_t *sc = reinterpret_cast<const uint16_t *>(block.scales);
        uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                             ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = fp16_to_fp32(scale_u16);

        const uint8_t *qs = block.qs;
        const uint8_t *qh = block.qh;

        // Process in 8 iterations (QK_K/32 = 256/32 = 8)
        // Each iteration handles 32 elements (4 groups of 8)
        for (size_t ib = 0; ib < 8; ++ib)
        {
            // Two sub-scales per iteration
            const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);

            // Grid indices for 4 groups of 8
            uint16_t idx[4];
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            // Delta signs for 4 groups
            float delta[4];
            delta[0] = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;

            // First two groups use dl1
            for (size_t l = 0; l < 2; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl1 * (grid[j] + delta[l]);
                }
                output += 8;
            }

            // Last two groups use dl2
            for (size_t l = 2; l < 4; ++l)
            {
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + idx[l]);
                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl2 * (grid[j] + delta[l]);
                }
                output += 8;
            }

            qs += 4;
            qh += 2;
        }
    }

#ifdef __AVX512F__
    void IQ1_MTensor::decodeBlockAVX512(const IQ1_MBlock &block, float *output)
    {
        // Stub: call scalar implementation
        decodeBlockScalar(block, output);
    }
#endif

#ifdef __AVX2__
    void IQ1_MTensor::decodeBlockAVX2(const IQ1_MBlock &block, float *output)
    {
        // Stub: call scalar implementation
        decodeBlockScalar(block, output);
    }
#endif

    IQ1_MTensor::~IQ1_MTensor() {}


    const float *IQ1_MTensor::data() const
    {
        assertValid("IQ1_MTensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("IQ1_MTensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // View-aware data pointer selection
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    const size_t row_offset = r * shape_[1];
                    const size_t col_offset = b * IQ1_MBlock::BLOCK_SIZE;
                    const size_t remaining = shape_[1] > col_offset ? (shape_[1] - col_offset) : 0;
                    const size_t valid_count = std::min(static_cast<size_t>(IQ1_MBlock::BLOCK_SIZE), remaining);

                    if (valid_count == 0)
                    {
                        continue;
                    }

                    if (valid_count == IQ1_MBlock::BLOCK_SIZE)
                    {
                        decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[row_offset + col_offset]);
                    }
                    else
                    {
                        float temp[IQ1_MBlock::BLOCK_SIZE];
                        decodeBlock(blocks[r * blocks_per_row + b], temp);
                        std::memcpy(&dequant_cache_[row_offset + col_offset], temp, valid_count * sizeof(float));
                    }
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ1_MTensor::mutable_data()
    {
        throw std::runtime_error("IQ1_MTensor::mutable_data: quantized tensors are immutable");
    }

    bool IQ1_MTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[IQ1_MTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void IQ1_MTensor::to_bf16(uint16_t *dst) const
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

    void IQ1_MTensor::to_fp16(uint16_t *dst) const
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

    void IQ1_MTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void IQ1_MTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void IQ1_MTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void IQ1_MTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Get raw data pointer (view-aware)
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);

        // Calculate super-block index (8 Q8_0 blocks per IQ1_M super-block)
        size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        size_t super_block_idx = k_block_offset / 8;

        const IQ1_MBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

        // Extract global scale once (shared across all sub-blocks)
        float global_scale = simd::extract_iq1m_global_scale(super_block);

        // Decode all 8 sub-blocks (each sub-block = 32 elements = 1 Q8_0 block)
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq1m_to_q8_0(super_block, sub_idx, global_scale, output[sub_idx].qs, &output[sub_idx].d);
        }
    }

    void IQ1_MTensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("IQ1_MTensor::unpack_superblock_to_int8: output must not be null");
        }

        const size_t blocks_per_row = (shape_[1] + IQ1_MBlock::BLOCK_SIZE - 1) / IQ1_MBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ1_MBlock *blocks = reinterpret_cast<const IQ1_MBlock *>(data_ptr);
        const IQ1_MBlock &super_block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_iq1_m_superblock_to_int8(super_block, output, scales, mins);
    }


    void IQ1_MTensor::packVnniBlock(const VnniPackContext &ctx, int n, int b) const
    {
        const size_t linear = vnniLinearIdx(ctx, n, b);
        const int sb_per_row = vnniSuperBlocksPerRow(ctx.K);
        const int sb_idx = b / 8;
        const int sub_idx = b % 8;
        const auto *blk = &typed_data()[static_cast<size_t>(n) * sb_per_row + sb_idx];

        const uint8_t *qs = blk->qs + sub_idx * 4;
        const uint8_t *qh = blk->qh + sub_idx * 2;

        const uint16_t *sc = reinterpret_cast<const uint16_t *>(blk->scales);
        const uint16_t scale_u16 = static_cast<uint16_t>(
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
            ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d_val = fp16_to_fp32(scale_u16);

        const int sc_word_idx = sub_idx / 2;
        const int sc_bit_offset = 6 * (sub_idx % 2);
        const int sc3_lo = (sc[sc_word_idx] >> (sc_bit_offset + 0)) & 0x7;
        const int sc3_hi = (sc[sc_word_idx] >> (sc_bit_offset + 3)) & 0x7;
        const float dl1 = d_val * (2.0f * static_cast<float>(sc3_lo) + 1.0f);
        const float dl2 = d_val * (2.0f * static_cast<float>(sc3_hi) + 1.0f);

        uint8_t payload_buf[6];
        payload_buf[0] = qs[0];
        payload_buf[1] = qs[1];
        payload_buf[2] = qs[2];
        payload_buf[3] = qs[3];
        payload_buf[4] = qh[0];
        payload_buf[5] = qh[1];

        std::memcpy(vnniPayloadDst(ctx, linear), payload_buf, 6);
        ctx.scales_array[linear] = fp32_to_fp16(dl1);
        ctx.mins_array[linear] = fp32_to_fp16(dl2);
    }

} // namespace llaminar2
