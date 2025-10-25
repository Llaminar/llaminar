/**
 * @file IQ1_STensor.cpp
 * @brief IQ1_S quantized tensor implementation (1-bit small IQ)
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
    IQ1_STensor::IQ1_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
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

    std::unique_ptr<ITensorGemm> IQ1_STensor::createGemm()
    {
        return std::make_unique<QuantizedGemmKernel>(this);
    }

    void IQ1_STensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(raw_data_.data());
        const IQ1_SBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ1_STensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
        const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ1_STensor::decodeBlock(const IQ1_SBlock &block, float *output)
    {
        // IQ1_S: 256 elements per super-block, using iq1s_grid[2048]
        const float d = fp16_to_fp32(block.d);

        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            const uint8_t qh = block.qh[ib32];
            const float db = d * (0.5f + (qh >> 4)) * 0.125f;

            // Extract 11-bit grid index (8 bits from qs + 3 bits from qh)
            const uint16_t grid_idx = block.qs[ib32] | ((qh & 0x07) << 8);
            const uint64_t grid_value = iq1s_grid[grid_idx];
            const int8_t *grid = reinterpret_cast<const int8_t *>(&grid_value);

            for (size_t j = 0; j < 8; ++j)
            {
                output[ib32 * 8 + j] = db * grid[j];
            }
        }
    }

    IQ1_STensor::~IQ1_STensor() {}

    bool IQ1_STensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ1_STensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ1_SBlock *blocks = reinterpret_cast<const IQ1_SBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ1_SBlock::BLOCK_SIZE - 1) / IQ1_SBlock::BLOCK_SIZE;
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

    std::unique_ptr<llaminar2::ITensorRoPE> IQ1_STensor::createRoPE()
    {
        throw std::runtime_error("IQ1_STensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ1_STensor::createSwiGLU()
    {
        throw std::runtime_error("IQ1_STensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ1_STensor::createSoftmax()
    {
        throw std::runtime_error("IQ1_STensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ1_STensor::createRMSNorm()
    {
        throw std::runtime_error("IQ1_STensor: RMSNorm not supported");
    }




    bool IQ1_STensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ1_STensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
