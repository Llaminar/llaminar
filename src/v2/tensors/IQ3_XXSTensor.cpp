/**
 * @file IQ3_XXSTensor.cpp
 * @brief IQ3_XXS quantized tensor implementation (3-bit extra-extra-small IQ)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "../kernels/cpu/gemm/GemmAutoTuner.h"
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

    IQ3_XXSTensor::IQ3_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), is_view_(false), raw_data_(raw_data), raw_data_ptr_(nullptr), view_byte_offset_(0), parent_(nullptr), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ3_XXSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ3_XXSBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ3_XXSTensor: insufficient raw data");
        }
    }

    IQ3_XXSTensor::IQ3_XXSTensor(const std::vector<size_t> &shape, const uint8_t *raw_data_ptr,
                                 size_t byte_offset, std::shared_ptr<TensorBase> parent)
        : shape_(shape), is_view_(true), raw_data_(), raw_data_ptr_(raw_data_ptr),
          view_byte_offset_(byte_offset), parent_(parent), device_idx_(-1), device_blocks_(nullptr) {}

    std::shared_ptr<TensorBase> IQ3_XXSTensor::create_view(
        const std::vector<size_t> &view_shape, size_t offset_elements)
    {
        if (view_shape.size() != 2)
            throw std::invalid_argument("IQ3_XXSTensor::create_view: only 2D views supported");
        if (view_shape[1] != shape_[1])
            throw std::invalid_argument("IQ3_XXSTensor::create_view: K dimension must match parent");
        if (offset_elements % shape_[1] != 0)
            throw std::invalid_argument("IQ3_XXSTensor::create_view: offset must be row-aligned");

        size_t offset_rows = offset_elements / shape_[1];
        size_t view_end_row = offset_rows + view_shape[0];
        size_t parent_rows = shape_[0];
        if (view_end_row > parent_rows)
            throw std::out_of_range("IQ3_XXSTensor::create_view: view exceeds parent bounds");

        size_t k = shape_[1];
        size_t blocks_per_row = (k + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        size_t byte_offset_in_parent = offset_rows * blocks_per_row * sizeof(IQ3_XXSBlock);
        size_t new_total_byte_offset = view_byte_offset_ + byte_offset_in_parent;

        const uint8_t *parent_data_ptr = is_view_ ? raw_data_ptr_ : raw_data_.data();

        return std::shared_ptr<TensorBase>(
            new IQ3_XXSTensor(view_shape, parent_data_ptr, new_total_byte_offset,
                              is_view_ ? parent_ : shared_from_this()));
    }

    std::unique_ptr<ITensorGemm> IQ3_XXSTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ3_XXSTensor::decodeBlock(const IQ3_XXSBlock &block, float *output)
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

    void IQ3_XXSTensor::decodeBlockScalar(const IQ3_XXSBlock &block, float *output)
    {
        // IQ3_XXS: 256 elements per super-block, using iq3xxs_grid[512]
        // Matches llama.cpp's dequantize_row_iq3_xxs implementation
        const float d = fp16_to_fp32(block.d);
        const uint8_t *qs = block.qs;
        const uint8_t *scales_and_signs = qs + 64; // QK_K/4 = 256/4 = 64

        // Process in 8 iterations (QK_K/32 = 256/32 = 8)
        // Each iteration handles 32 elements (4 groups of 8)
        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;

            for (size_t l = 0; l < 4; ++l)
            {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t *grid1 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t *grid2 = reinterpret_cast<const uint8_t *>(iq3xxs_grid + qs[2 * l + 1]);

                for (size_t j = 0; j < 4; ++j)
                {
                    output[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                    output[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                }
                output += 8;
            }
            qs += 8;
        }
    }

#ifdef __AVX512F__
    void IQ3_XXSTensor::decodeBlockAVX512(const IQ3_XXSBlock &block, float *output)
    {
        // TODO: Implement AVX512 vectorization
        // For now, fall back to scalar
        decodeBlockScalar(block, output);
    }
#endif

#ifdef __AVX2__
    void IQ3_XXSTensor::decodeBlockAVX2(const IQ3_XXSBlock &block, float *output)
    {
        // TODO: Implement AVX2 vectorization
        // For now, fall back to scalar
        decodeBlockScalar(block, output);
    }
#endif

    IQ3_XXSTensor::~IQ3_XXSTensor() {}

    bool IQ3_XXSTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ3_XXSTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);

            // View-aware data pointer selection
            const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);
            size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ3_XXSBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ3_XXSTensor::mutable_data()
    {
        throw std::runtime_error("IQ3_XXSTensor::mutable_data: quantized tensors are immutable");
    }

        bool IQ3_XXSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        LOG_ERROR("[IQ3_XXSTensor::copyFrom] Not implemented");
        return false;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    void IQ3_XXSTensor::to_bf16(uint16_t *dst) const
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

    void IQ3_XXSTensor::to_fp16(uint16_t *dst) const
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

    void IQ3_XXSTensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
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

    void IQ3_XXSTensor::to_fp32_row(size_t row_idx, float *buffer) const
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

    void IQ3_XXSTensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
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

    void IQ3_XXSTensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // IQ3_XXS: 256 elements per super-block → 8 Q8_0 sub-blocks of 32 elements
        const uint8_t *data_ptr = is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data();
        const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(data_ptr);

        size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        size_t super_block_idx = k_block_offset / 8; // 8 Q8_0 blocks per IQ3_XXS super-block
        size_t sub_block_idx = k_block_offset % 8;   // Which Q8_0 sub-block within super-block

        const IQ3_XXSBlock &super_block = blocks[row_idx * blocks_per_row + super_block_idx];

        // Decode all 8 sub-blocks from the super-block
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            simd::decode_iq3xxs_to_q8_0(super_block, sub_idx, output[sub_idx].qs, &output[sub_idx].d);
        }
    }

} // namespace llaminar2
