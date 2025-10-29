/**
 * @file IQ2_STensor.cpp
 * @brief IQ2_S quantized tensor implementation (2-bit small IQ with high bits and scales)
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
    IQ2_STensor::IQ2_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data)
        : shape_(shape), raw_data_(raw_data), device_idx_(-1), device_blocks_(nullptr)
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

    std::unique_ptr<ITensorGemm> IQ2_STensor::createGemm()
    {
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    void IQ2_STensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());
        const IQ2_SBlock &block = blocks[row_idx * blocks_per_row + k_block_offset];
        decodeBlock(block, output);
    }

    const void *IQ2_STensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        const size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
        const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());
        return &blocks[row_idx * blocks_per_row + k_block_offset];
    }

    void IQ2_STensor::decodeBlock(const IQ2_SBlock &block, float *output)
    {
        // IQ2_S: 256 elements per super-block, using iq2s_grid[1024]
        const float d = fp16_to_fp32(block.d);
        const float qh = fp16_to_fp32(block.qh);

        for (size_t ib32 = 0; ib32 < 8; ++ib32)
        {
            const uint16_t qs_val = block.qs[ib32];
            const float db = d * (0.5f + (qs_val >> 12)) * 0.25f;

            // Extract 10-bit grid index
            const size_t grid_idx = qs_val & 0x3FF;
            const uint64_t grid_value = iq2s_grid[grid_idx];
            const uint8_t *grid = reinterpret_cast<const uint8_t *>(&grid_value);

            // Extract sign pattern from qh
            const uint8_t signs = static_cast<uint8_t>((reinterpret_cast<const uint8_t *>(&block.qh))[ib32 / 2] >> (4 * (ib32 % 2)));

            for (size_t j = 0; j < 8; ++j)
            {
                output[ib32 * 8 + j] = db * grid[j] * (signs & (1 << j) ? -1.0f : 1.0f);
            }
        }
    }

    IQ2_STensor::~IQ2_STensor() {}

    bool IQ2_STensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        return true;
    }

    const float *IQ2_STensor::data() const
    {
        if (dequant_cache_.empty())
        {
            size_t total_elements = shape_[0] * shape_[1];
            dequant_cache_.resize(total_elements);
            const IQ2_SBlock *blocks = reinterpret_cast<const IQ2_SBlock *>(raw_data_.data());
            size_t blocks_per_row = (shape_[1] + IQ2_SBlock::BLOCK_SIZE - 1) / IQ2_SBlock::BLOCK_SIZE;
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

    std::unique_ptr<llaminar2::ITensorRoPE> IQ2_STensor::createRoPE()
    {
        throw std::runtime_error("IQ2_STensor: RoPE not supported");
    }

    std::unique_ptr<llaminar2::ITensorSwiGLU> IQ2_STensor::createSwiGLU()
    {
        throw std::runtime_error("IQ2_STensor: SwiGLU not supported");
    }

    std::unique_ptr<llaminar2::ITensorSoftmax> IQ2_STensor::createSoftmax()
    {
        throw std::runtime_error("IQ2_STensor: Softmax not supported");
    }

    std::unique_ptr<llaminar2::ITensorRMSNorm> IQ2_STensor::createRMSNorm()
    {
        throw std::runtime_error("IQ2_STensor: RMSNorm not supported");
    }

    bool IQ2_STensor::copyFrom(const TensorBase *src)
    {
        // Quantized tensors are read-only weights - no transfer needed
        (void)src;
        std::cerr << "[IQ2_STensor::copyFrom] Not implemented\n";
        return false;
    }

} // namespace llaminar2
