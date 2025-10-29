/**
 * @file IQ2_XXSTensor.cpp
 * @brief IQ2_XXS quantized tensor implementation (2-bit extra-extra-small IQ)
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
    IQ2_XXSTensor::IQ2_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ2_XXSTensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ2_XXSBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ2_XXSTensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> IQ2_XXSTensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ2_XXSTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(raw_data_.data());
        const IQ2_XXSBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ2_XXSTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
        const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ2_XXSTensor::decodeBlock(const IQ2_XXSBlock &block, float *output)
    {
        // IQ2_XXS: 256 elements per super-block, using iq2xxs_grid[256]
        const float d = fp16_to_fp32(block.d);

        uint32_t aux32[2];
        const uint8_t *aux8 = reinterpret_cast<const uint8_t *>(aux32);

        // Process 8 sub-blocks of 32 elements each
        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            std::memcpy(aux32, &block.qs[4 * ib32], 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

            // Process 4 groups of 8 elements
            for (size_t l = 0; l < 4; ++l)
            {
                const uint64_t grid_value = iq2xxs_grid[aux8[l]];
                const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (size_t j = 0; j < 8; ++j)
                {
                    output[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
                }
                output += 8;
            }
        }
    }

    IQ2_XXSTensor::~IQ2_XXSTensor() {}

    bool IQ2_XXSTensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ2_XXSTensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ2_XXSBlock *blocks = reinterpret_cast<const IQ2_XXSBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ2_XXSBlock::BLOCK_SIZE - 1) / IQ2_XXSBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ2_XXSBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ2_XXSTensor::mutable_data()
    {
        throw std::runtime_error("IQ2_XXSTensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> IQ2_XXSTensor::createRoPE()
    {
        throw std::runtime_error("IQ2_XXSTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ2_XXSTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ2_XXSTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ2_XXSTensor::createSoftmax()
    {
        throw std::runtime_error("IQ2_XXSTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ2_XXSTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ2_XXSTensor: RMSNorm not supported");
    }

    bool IQ2_XXSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ2_XXSTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
