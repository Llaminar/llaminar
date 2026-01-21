/**
 * @file IQ1_STensor.cpp
 * @brief IQ1_S quantized tensor implementation (1-bit small IQ)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
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

    IQ1_STensor::IQ1_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr), view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ1_STensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ1_SBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ1_STensor: insufficient raw data");
        }
    }

    IQ1_STensor::IQ1_STensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                             size_t byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr) {}

    std::shared_ptr<TensorBase> IQ1_STensor::create_view(
        const std::vector<size_t> &view_shape, size_t offset_elements)
    {
        if (view_shape.size() != 2)
            throw std::invalid_argument("IQ1_STensor::create_view: only 2D views supported");
        if (view_shape[1] != shape_[1])
            throw std::invalid_argument("IQ1_STensor::create_view: K dimension must match parent");
        if (offset_elements % shape_[1] != 0)
            throw std::invalid_argument("IQ1_STensor::create_view: offset must be row-aligned");

        size_t offset_rows = offset_elements / shape_[1];
        size_t view_end_row = offset_rows + view_shape[0];
        size_t parent_rows = shape_[0];
        if (view_end_row > parent_rows)
            throw std::out_of_range("IQ1_STensor::create_view: view exceeds parent bounds");

        size_t k = shape_[1];
        size_t blocks_per_row = (k + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        size_t byte_offset_in_parent = offset_rows * blocks_per_row * sizeof(IQ1_SBlock);
        size_t new_total_byte_offset = view_byte_offset_ + byte_offset_in_parent;

        const uint8_t *parent_data_ptr = is_view_ ? raw_data_ptr_ : raw_data_.data();

        return std::shared_ptr<TensorBase>(
            new IQ1_STensor(view_shape, parent_data_ptr, new_total_byte_offset,
                            is_view_ ? parent_ : shared_from_this()));
    }

    std::unique_ptr<ITensorGemm> IQ1_STensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void IQ1_STensor::decodeBlock(const IQ1_SBlock &block, float *output)
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

    void IQ1_STensor::decodeBlockScalar(const IQ1_SBlock &block, float *output)
    {
        // IQ1_S: 256 elements per super-block, using iq1s_grid[2048]
        // Matches llama.cpp's dequantize_row_iq1_s implementation
        const float d = fp16_to_fp32(block.d);
        const uint8_t *qs = block.qs;
        // Copy qh to avoid taking address of packed member (alignment warning)
        uint16_t qh[8];
        std::memcpy(qh, block.qh, sizeof(qh));

        // Process in 8 iterations (QK_K/32 = 256/32 = 8)
        // Each iteration handles 32 elements (4 groups of 8)
        for (size_t ib = 0; ib < 8; ++ib)
        {
            // Scale: 3 bits from qh[ib] bits 12-14
            const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);

            // Delta: sign bit from qh[ib] bit 15
            const float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            // Process 4 groups of 8 elements (32 elements total per ib)
            for (size_t l = 0; l < 4; ++l)
            {
                // Grid index: 8 bits from qs[l] + 3 bits from qh[ib] (bits 3*l to 3*l+2)
                const uint16_t grid_idx = qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8);

                // CRITICAL: iq1s_grid is uint64_t[2048], but we need to access it as int8_t*
                // Each uint64_t contains 8 int8_t values
                // llama.cpp does: (const int8_t *)(iq1s_grid + grid_idx)
                // This treats iq1s_grid as a byte array starting at iq1s_grid[grid_idx]
                const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + grid_idx);

                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = dl * (grid[j] + delta);
                }
                output += 8;
            }
            qs += 4;
        }
    }

#ifdef __AVX512F__
    void IQ1_STensor::decodeBlockAVX512(const IQ1_SBlock &block, float *output)
    {
        // Stub: call scalar implementation
        decodeBlockScalar(block, output);
    }
#endif

#ifdef __AVX2__
    void IQ1_STensor::decodeBlockAVX2(const IQ1_SBlock &block, float *output)
    {
        // Stub: call scalar implementation
        decodeBlockScalar(block, output);
    }
#endif

    IQ1_STensor::~IQ1_STensor() {}


    const float *IQ1_STensor::data() const
    {
        assertValid("IQ1_STensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("IQ1_STensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // View-aware data pointer selection
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ1_SBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ1_STensor::mutable_data()
    {
        throw std::runtime_error("IQ1_STensor::mutable_data: quantized tensors are immutable");
    }

    bool IQ1_STensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[IQ1_STensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void IQ1_STensor::to_bf16(uint16_t *dst) const
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

    void IQ1_STensor::to_fp16(uint16_t *dst) const
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

    void IQ1_STensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void IQ1_STensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void IQ1_STensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void IQ1_STensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Get raw data pointer (view-aware)
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);

        // Calculate super-block index (8 Q8_0 blocks per IQ1_S super-block)
        size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        size_t super_block_idx = k_block_offset / 8;

        const IQ1_SBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

        // Decode all 8 sub-blocks (each sub-block = 32 elements = 1 Q8_0 block)
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq1s_to_q8_0(super_block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }

    void IQ1_STensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("IQ1_STensor::unpack_superblock_to_int8: output must not be null");
        }

        const size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(data_ptr);
        const IQ1_SBlock &super_block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_iq1_s_superblock_to_int8(super_block, output, scales, mins);
    }

} // namespace llaminar2
