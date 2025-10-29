/**
 * @file IQ3_STensor.cpp
 * @brief IQ3_S quantized tensor implementation (3-bit small IQ with signs)
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
    IQ3_STensor::IQ3_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("IQ3_STensor: shape cannot be empty");
        }

        size_t n_elems = 1;
        for (auto dim : shape)
        {
            n_elems *= dim;
        }

        size_t n_blocks = (n_elems + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
        size_t expected_bytes = n_blocks * sizeof(IQ3_SBlock);

        if (raw_data_.size() < expected_bytes)
        {
            throw std::invalid_argument("IQ3_STensor: insufficient raw data");
        }
    }

    std::unique_ptr<ITensorGemm> IQ3_STensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ3_STensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
        const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(raw_data_.data());
        const IQ3_SBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ3_STensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
        const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ3_STensor::decodeBlock(const IQ3_SBlock &block, float *output)
    {
        // IQ3_S: 256 elements per super-block, using iq3s_grid[512]
        const float d = fp16_to_fp32(block.d);

        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            const uint8_t scale_idx = block.scales[ib32 / 4] >> (4 * (ib32 % 4));
            const float db = d * (0.5f + (scale_idx & 0x0F)) * 0.125f;

            // Process 8 groups of 4 values
            for (size_t l = 0; l < 8; ++l)
            {
                // Extract 9-bit grid index from qs
                const size_t byte_idx = ib32 * 12 + l * 3 / 2;
                uint16_t grid_idx;
                if (l % 2 == 0)
                {
                    grid_idx = (block.qs[byte_idx] | ((block.qs[byte_idx + 1] & 0x01) << 8));
                }
                else
                {
                    grid_idx = (block.qs[byte_idx] >> 1) | ((block.qs[byte_idx + 1] & 0x03) << 7);
                }

                const uint32_t grid_value = iq3s_grid[grid_idx & 0x1FF];
                const int8_t *grid = reinterpret_cast<const int8_t *>(&grid_value);

                for (size_t j = 0; j < 4; ++j)
                {
                    output[ib32 * 32 + l * 4 + j] = db * grid[j];
                }
            }
        }
    }

    IQ3_STensor::~IQ3_STensor() {}

    bool IQ3_STensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ3_STensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ3_SBlock *blocks = reinterpret_cast<const IQ3_SBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ3_SBlock::BLOCK_SIZE - 1) / IQ3_SBlock::BLOCK_SIZE;
            for (size_t r = 0; r < shape_[0]; ++r)
            {
                for (size_t b = 0; b < blocks_per_row; ++b)
                {
                    decodeBlock(blocks[r * blocks_per_row + b], &dequant_cache_[r * shape_[1] + b * IQ3_SBlock::BLOCK_SIZE]);
                }
            }
        }
        return dequant_cache_.data();
    }

    float *IQ3_STensor::mutable_data()
    {
        throw std::runtime_error("IQ3_STensor::mutable_data: quantized tensors are immutable");
    }

    std::unique_ptr<llaminar2::ITensorRoPE> IQ3_STensor::createRoPE()
    {
        throw std::runtime_error("IQ3_STensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ3_STensor::createSwiGLU()
    {
        throw std::runtime_error("IQ3_STensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ3_STensor::createSoftmax()
    {
        throw std::runtime_error("IQ3_STensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ3_STensor::createRMSNorm()
    {
        throw std::runtime_error("IQ3_STensor: RMSNorm not supported");
    }

    bool IQ3_STensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ3_STensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
