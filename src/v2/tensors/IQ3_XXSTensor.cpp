/**
 * @file IQ3_XXSTensor.cpp
 * @brief IQ3_XXS quantized tensor implementation (3-bit extra-extra-small IQ)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "IQQuantTables.h"
#include "../kernels/cpu/QuantizedGemm.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    IQ3_XXSTensor::IQ3_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
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

    std::unique_ptr<ITensorGemm> IQ3_XXSTensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void IQ3_XXSTensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(raw_data_.data());
        const IQ3_XXSBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ3_XXSTensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ3_XXSBlock::BLOCK_SIZE - 1) / IQ3_XXSBlock::BLOCK_SIZE;
        const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ3_XXSTensor::decodeBlock(const IQ3_XXSBlock &block, float *output)
    {
        // IQ3_XXS: 256 elements per super-block, using iq3xxs_grid[256]
        const float d = fp16_to_fp32(block.d);

        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            const float db = d;

            // Process 8 groups of 4 values from qs
            for (size_t l = 0; l < 8; ++l)
            {
                const uint8_t grid_idx = block.qs[ib32 * 8 + l];
                const uint32_t grid_value = iq3xxs_grid[grid_idx];
                const int8_t *grid = reinterpret_cast<const int8_t *>(&grid_value);

                for (size_t j = 0; j < 4; ++j)
                {
                    output[ib32 * 32 + l * 4 + j] = db * grid[j];
                }
            }
        }
    }

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
            const IQ3_XXSBlock *blocks = reinterpret_cast<const IQ3_XXSBlock *>(raw_data_.data());
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

    std::unique_ptr<llaminar2::ITensorRoPE> IQ3_XXSTensor::createRoPE()
    {
        throw std::runtime_error("IQ3_XXSTensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ3_XXSTensor::createSwiGLU()
    {
        throw std::runtime_error("IQ3_XXSTensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ3_XXSTensor::createSoftmax()
    {
        throw std::runtime_error("IQ3_XXSTensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ3_XXSTensor::createRMSNorm()
    {
        throw std::runtime_error("IQ3_XXSTensor: RMSNorm not supported");
    }




    bool IQ3_XXSTensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ3_XXSTensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
