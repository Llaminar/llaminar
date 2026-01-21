/**
 * @file IQ2_STensor.cpp
 * @brief IQ2_S quantized tensor implementation (2-bit small IQ with high bits and scales)
 * @author David Sanftenberg
 */

#include "TensorClasses.h"
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

    IQ2_STensor::IQ2_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr), view_byte_offset_(0), parent_(nullptr), device_(DeviceId::cpu()), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ2_STensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ2_SBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ2_STensor: insufficient raw data");
        }
    }

    IQ2_STensor::IQ2_STensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                             size_t byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(byte_offset), parent_(parent), device_(DeviceId::cpu()), device_blocks_(nullptr) {}

    std::shared_ptr<TensorBase> IQ2_STensor::create_view(
        const std::vector<size_t> &view_shape, size_t offset_elements)
    {
        if (view_shape.size() != 2)
            throw std::invalid_argument("IQ2_STensor::create_view: only 2D views supported");
        if (view_shape[1] != shape_[1])
            throw std::invalid_argument("IQ2_STensor::create_view: K dimension must match parent");
        if (offset_elements % shape_[1] != 0)
            throw std::invalid_argument("IQ2_STensor::create_view: offset must be row-aligned");

        size_t offset_rows = offset_elements / shape_[1];
        size_t view_end_row = offset_rows + view_shape[0];
        size_t parent_rows = shape_[0];
        if (view_end_row > parent_rows)
            throw std::out_of_range("IQ2_STensor::create_view: view exceeds parent bounds");

        size_t k = shape_[1];
        size_t blocks_per_row = (k + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        size_t byte_offset_in_parent = offset_rows * blocks_per_row * sizeof(IQ2_SBlock);
        size_t new_total_byte_offset = view_byte_offset_ + byte_offset_in_parent;

        const uint8_t *parent_data_ptr = is_view_ ? raw_data_ptr_ : raw_data_.data();

        return std::shared_ptr<TensorBase>(
            new IQ2_STensor(view_shape, parent_data_ptr, new_total_byte_offset,
                            is_view_ ? parent_ : shared_from_this()));
    }

    std::unique_ptr<ITensorGemm> IQ2_STensor::createGemm()
    {
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    void IQ2_STensor::decodeBlock(const IQ2_SBlock &block, float *output)
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

    void IQ2_STensor::decodeBlockScalar(const IQ2_SBlock &block, float *output)
    {
        // IQ2_S: 256 elements per super-block, using iq2s_grid[1024]
        // Based on llama.cpp dequantize_row_iq2_s

        const float d = fp16_to_fp32(block.d);
        const uint8_t *qs = block.qs;
        const uint8_t *qh = block.qh;
        const uint8_t *signs = qs + 32; // signs pointer: QK_K/8 = 32 bytes after start of qs

        float db[2];

        for (int ib32 = 0; ib32 < 8; ++ib32)
        { // QK_K/32 = 8
            db[0] = d * (0.5f + (block.scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (block.scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; ++l)
            {
                const float dl = db[l / 2];
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j)
                {
                    output[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                output += 8;
            }
            qs += 4;
            signs += 4;
        }
    }

#ifdef __AVX512F__
    void IQ2_STensor::decodeBlockAVX512(const IQ2_SBlock &block, float *output)
    {
        // TODO: Implement AVX512 vectorization
        // For now, fall back to scalar
        decodeBlockScalar(block, output);
    }
#endif

#ifdef __AVX2__
    void IQ2_STensor::decodeBlockAVX2(const IQ2_SBlock &block, float *output)
    {
        // TODO: Implement AVX2 vectorization
        // For now, fall back to scalar
        decodeBlockScalar(block, output);
    }
#endif

    IQ2_STensor::~IQ2_STensor() {}


    const float *IQ2_STensor::data() const
    {
        assertValid("IQ2_STensor::data");
        if (dequant_cache_.empty())
        {
            // Check if raw data was released after GEMM packing
            if (raw_data_released_)
            {
                LOG_DEBUG("IQ2_STensor::data() called but raw data was released after GEMM packing");
                return nullptr;
            }
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // View-aware data pointer selection
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
#pragma omp parallel for schedule(static) if (total_elements > 10000)
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ2_SBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ2_STensor::mutable_data()
    {
        throw std::runtime_error("IQ2_STensor::mutable_data: quantized tensors are immutable");
    }

    bool IQ2_STensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[IQ2_STensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void IQ2_STensor::to_bf16(uint16_t *dst) const
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

    void IQ2_STensor::to_fp16(uint16_t *dst) const
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

    void IQ2_STensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void IQ2_STensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void IQ2_STensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void IQ2_STensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // IQ2_S: 256 elements per super-block → 8 Q8_0 sub-blocks of 32 elements
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);

        size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        size_t super_block_idx = k_block_offset / 8; // 8 Q8_0 blocks per IQ2_S super-block
        size_t sub_block_idx = k_block_offset % 8;   // Which Q8_0 sub-block within super-block

        const IQ2_SBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

        // Decode all 8 sub-blocks from the super-block
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq2s_to_q8_0(super_block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }

    void IQ2_STensor::unpack_superblock_to_int8(
        size_t row_idx,
        size_t superblock_idx,
        int8_t *output,
        float *scales,
        float *mins) const
    {
        if (!output)
        {
            throw std::invalid_argument("IQ2_STensor::unpack_superblock_to_int8: output must not be null");
        }

        const size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(data_ptr);
        const IQ2_SBlock &super_block = blocks[row_idx * blocks_per_row + superblock_idx];

        // Unpack all 8 sub-blocks (256 elements total)
        simd::unpack_iq2_s_superblock_to_int8(super_block, output, scales, mins);
    }

} // namespace llaminar2
