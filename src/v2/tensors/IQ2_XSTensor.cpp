/**
 * @file IQ2_XSTensor.cpp
 * @brief IQ2_XS quantized tensor implementation (2-bit extra-small IQ with per-block scales)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "../kernels/cpu/GemmAutoTuner.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    IQ2_XSTensor::IQ2_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ2_XSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ2_XSBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ2_XSTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> IQ2_XSTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ2_XSTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
        const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(raw_data_.data());
        const IQ2_XSBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ2_XSTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
        const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ2_XSTensor::decodeBlock(const IQ2_XSBlock &block, float *output)
    {
        // IQ2_XS: Similar to IQ2_XXS but with per-block scales, using iq2xs_grid[512]
        const float d = fp16_to_fp32(block.d);

        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            const uint16_t qs_val = block.qs[ib32];
            const float db = d * (block.scales[ib32 / 4] & 0x0F) / 16.0f;

            // Extract 9-bit grid index (8 bits from qs + 1 bit from scales)
            const size_t grid_idx = (qs_val & 0x01FF);
            const uint64_t grid_value = iq2xs_grid[grid_idx];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);

            // Extract 7-bit sign pattern
            const uint8_t signs = ksigns_iq2xs[(qs_val >> 9) & 127];

            for (size_t j = 0; j < 8; ++j)
            {
                output[ib32 * 8 + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
            }
        }
    }

    IQ2_XSTensor::~IQ2_XSTensor() {}

    bool IQ2_XSTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ2_XSTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ2_XSBlock *blocks = reinterpret_cast<const IQ2_XSBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ2_XSBlock::BLOCK_SIZE - 1) / IQ2_XSBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ2_XSBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ2_XSTensor::mutable_data()
    {
        throw std::runtime_error("IQ2_XSTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> IQ2_XSTensor::createRoPE()
    {
        throw std::runtime_error("IQ2_XSTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ2_XSTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ2_XSTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ2_XSTensor::createSoftmax()
    {
        throw std::runtime_error("IQ2_XSTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ2_XSTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ2_XSTensor: RMSNorm not supported");
    }

    bool IQ2_XSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ2_XSTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
